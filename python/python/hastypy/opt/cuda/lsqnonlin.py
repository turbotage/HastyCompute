import math

import cupy as cp
from jinja2 import Template
from symengine import sympify


#from ..sym import sym
from hastypy.opt.sym import util

from hastypy.opt.cuda import linalg
from hastypy.opt.cuda import solver

from hastypy.opt.cuda import cuda_program as cudap
from hastypy.opt.cuda.cuda_program import CudaFunction, CudaTensor
from hastypy.opt.cuda.cuda_program import CudaTensorChecking as ctc

import time
import copy

class Expr():
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], 
				nonlin_terms: list[str] | None, nonlin_pars: list[int] | None):
		self.expr = expr
		self.pars_str = pars_str
		self.consts_str = consts_str
		self.nonlin_terms = nonlin_terms
		self.nonlin_pars = nonlin_pars



class F_GradF():
	def __init__(self, f: CudaFunction, gradf: CudaFunction, nparam: int, nconst: int, name: str):
		self.f = f
		self.gradf = gradf
		self.nparam = nparam
		self.nconst = nconst
		self.name = name



def f_funcid(model_expr: Expr, ndata: int, dtype: cp.dtype):
	return 'f' + ctc.dim_dim_dim_type_funcid(len(model_expr.pars_str), len(model_expr.consts_str), ndata,
		dtype, 'f') + '_' + util.expr_hash(copy.deepcopy(model_expr.expr), 12)

def f_code(model_expr: Expr, ndata: int, dtype: cp.dtype):
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data,
	{{fp_type}}* f, int tid, int Nprobs) 
{
	{{fp_type}} pars[{{nparam}}];
	for (int i = 0; i < {{ndata}}; ++i) {
		pars[i] = params[i*Nprobs+tid];
	}

	{{fp_type}} res;

	for (int i = 0; i < {{ndata}}; ++i) {
{{sub_expr}}

{{res_expr}}

		f[tid] += res * res;
	}
}
""")

	exprobj = copy.deepcopy(model_expr)

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'res')

	nparam = len(exprobj.pars_str)
	nconst = len(exprobj.consts_str)

	funcid = f_funcid(exprobj, ndata, dtype)

	sym_expr = sympify(exprobj.expr)
	# convert parameter names to ease kernel generation
	for k in range(nparam):
		temp = exprobj.pars_str[k]
		exprobj.pars_str[k] = 'parvar_' + temp
		sym_expr = sym_expr.subs(temp, exprobj.pars_str[k])

	for k in range(nconst):
		temp = exprobj.consts_str[k]
		exprobj.consts_str[k] = 'convar_' + temp
		sym_expr = sym_expr.subs(temp, exprobj.consts_str[k])

	substs, reduced = util.res(str(sym_expr), exprobj.pars_str, exprobj.consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	res_str = '\t\tres = '+cuprint.tcs_f(reduced[0])+'-data[i*Nprobs+tid];'

	for k in range(nparam):
		p = exprobj.pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		res_str = res_str.replace(p, repl)

	for k in range(nconst):
		c = exprobj.consts_str[k]
		repl = 'consts['+str(k)+'*'+str(nconst)+'*Nprobs+i*Nprobs+tid]'
		sub_str = sub_str.replace(c, repl)
		res_str = res_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst, ndata=ndata,
		sub_expr=sub_str, res_expr=res_str)


	return rjh_kernel

class F(CudaFunction):
	def __init__(self, model_expr: Expr, ndata: int, dtype: cp.dtype, write_to_file: bool = False):

		exprobj = copy.deepcopy(model_expr)

		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = exprobj.pars_str.copy()
		self.consts_str = exprobj.consts_str.copy()
		self.ndata = ndata

		self.write_to_file = write_to_file

		self.funcid = f_funcid(exprobj.expr, self.pars_str, self.consts_str, ndata, dtype)
		self.code = f_code(exprobj.expr, self.pars_str, self.consts_str, ndata, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, data, step_type, f):
		if self.run_func == None:
			self.build()

		Nprobs = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(Nprobs / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, data, step_type, f, Nprobs))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const char* step_type,
	{{fp_type}}* f, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nprobs) {
		if (step_type[tid] == 0) {
			return;
		}

		{{dfuncid}}(params, consts, data, f, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def fghhl_funcid(model_expr: Expr, ndata: int, dtype: cp.dtype):
	return 'fghhl' + ctc.dim_dim_dim_type_funcid(len(model_expr.pars_str), len(model_expr.consts_str), ndata,
		dtype, 'fghhl') + '_' + util.expr_hash(model_expr.expr, 12)

def fghhl_code(model_expr: Expr, ndata: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lam,
	{{fp_type}}* f, {{fp_type}}* g, {{fp_type}}* h, {{fp_type}}* hl, int tid, int Nprobs) 
{
	{{fp_type}} pars[{{nparam}}];
	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*Nprobs+tid];
	}

	{{fp_type}} res;

	{{fp_type}} jac[{{nparam}}];
	{{fp_type}} hes[{{nhes}}];
	{{fp_type}} lambda = lam[tid];

	for (int i = 0; i < {{ndata}}; ++i) {
		
{{sub_expr}}

{{eval_expr}}

{{jac_expr}}

{{hes_expr}}

		// sum these parts
		f[tid] += res * res;

		int l = 0;
		for (int j = 0; j < {{nparam}}; ++j) {
			// this sums up hessian parts
			for (int k = 0; k <= j; ++k) {
				int lidx = l*Nprobs+tid;
				{{fp_type}} jtemp = jac[j] * jac[k];
				{{fp_type}} hjtemp = hes[l] + jtemp;
				h[lidx] += hjtemp;
				if (j != k) {
					hl[lidx] += hjtemp;
				} else {
					hl[lidx] += hjtemp + {{max_func}}(lambda*jtemp, {{min_scaling}});
				}
				++l;
			}

			g[j*Nprobs+tid] += jac[j] * res;
		}

	}
}
""")

	exprobj = copy.deepcopy(model_expr)

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'fghhl')

	nparam = len(exprobj.pars_str)
	nconst = len(exprobj.consts_str)
	nhes = round(nparam*(nparam+1)/2)

	funcid = fghhl_funcid(exprobj, ndata, dtype)

	min_scaling = ''
	max_func = ''
	if type == 'float':
		max_func = 'fmaxf'
		min_scaling = '1e-4f'
	else:
		max_func = 'fmax'
		min_scaling = '1e-4'

	sym_expr = sympify(exprobj.expr)
	# convert parameter names to ease kernel generation
	for k in range(nparam):
		temp = exprobj.pars_str[k]
		exprobj.pars_str[k] = 'parvar_' + temp
		sym_expr = sym_expr.subs(temp, exprobj.pars_str[k])

	for k in range(nconst):
		temp = exprobj.consts_str[k]
		exprobj.consts_str[k] = 'convar_' + temp
		sym_expr = sym_expr.subs(temp, exprobj.consts_str[k])

	substs, reduced = util.res_jac_hes(str(sym_expr), exprobj.pars_str, exprobj.consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	eval_str = '\t\tres = '+cuprint.tcs_f(reduced[0])+'-data[i*Nprobs+tid];'

	jac_str = ""
	for k in range(nparam):
		s = reduced[1+k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		jac_str += '\t\tjac['+str(k)+'] = '+ctstr+';\n'

	hes_str = ""
	for k in range(nhes):
		s = reduced[(nparam + 1) + k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		hes_str += '\t\thes['+str(k)+'] = '+ctstr+'*res;\n'

	for k in range(nparam):
		p = exprobj.pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		eval_str = eval_str.replace(p, repl)
		jac_str = jac_str.replace(p, repl)
		hes_str = hes_str.replace(p, repl)

	for k in range(nconst):
		c = exprobj.consts_str[k]
		repl = 'consts['+str(k)+'*'+str(nconst)+'*Nprobs+i*Nprobs+tid]'
		sub_str = sub_str.replace(c, repl)
		eval_str = eval_str.replace(c, repl)
		jac_str = jac_str.replace(c, repl)
		hes_str = hes_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst, ndata=ndata, nhes=nhes,
		sub_expr=sub_str, eval_expr=eval_str, jac_expr=jac_str, hes_expr=hes_str,
		max_func=max_func, min_scaling=min_scaling)

	return rjh_kernel

class FGHHL(CudaFunction):
	def __init__(self, model_expr: Expr, ndata: int, dtype: cp.dtype, write_to_file: bool = False):
		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = model_expr.pars_str.copy()
		self.consts_str = model_expr.consts_str.copy()
		self.ndata = ndata

		self.write_to_file = write_to_file

		self.funcid = fghhl_funcid(model_expr, self.pars_str, self.consts_str, ndata, dtype)
		self.code = fghhl_code(model_expr, self.pars_str, self.consts_str, ndata, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, data, lam, step_type, f, g, h, hl):
		if self.run_func == None:
			self.build()

		Nprobs = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(Nprobs / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, data, lam, step_type, f, g, h, hl, Nprobs))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lam, const char* step_type,
	{{fp_type}}* f, {{fp_type}}* g, {{fp_type}}* h, {{fp_type}}* hl, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nprobs) {
		if (step_type[tid] == 0) {
			return;
		}

		{{dfuncid}}(params, consts, data, lam, f, g, h, hl, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def gain_ratio_step_funcid(nparam: int, dtype: cp.dtype):
	return 'gain_ratio_step' + ctc.dim_type_funcid(nparam, dtype)

def gain_ratio_step_code(nparam: int, dtype: cp.dtype):
	temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* f, const {{fp_type}}* ftp, const {{fp_type}}* pars_tp, const {{fp_type}}* step,
	const {{fp_type}}* g, const {{fp_type}}* h, {{fp_type}}* pars, 
	{{fp_type}}* lam, char* step_type, {{fp_type}} mu, {{fp_type}} eta, {{fp_type}} acc, {{fp_type}} dec, int tid, int Nprobs) 
{

	{{fp_type}} actual = 0.5f * (f[tid] - ftp[tid]);
	{{fp_type}} predicted = 0.0f;

	int k = 0;
	for (int i = 0; i < {{nparam}}; ++i) {
		for (int j = 0; j <= i; ++j) {
			float entry = h[k*Nprobs+tid] * step[i*Nprobs+tid] * step[j*Nprobs+tid];
			if (i == j) {
				predicted -= entry;
			} else {
				predicted -= 2.0f * entry;
			}
			++k;
		}
	}
	predicted *= 0.5f;

	for (int i = 0; i < {{nparam}}; ++i) {
		int iidx = i*Nprobs+tid;
		predicted += step[iidx] * g[iidx];
	}

	{{fp_type}} rho = actual / predicted;

	if ((rho > mu) && (actual > 0)) {
		for (int i = 0; i < {{nparam}}; ++i) {
			int iidx = i*Nprobs+tid;
			pars[iidx] = pars_tp[iidx];
			if (tid == 0) {
				printf("pars copied ");
			}
		}
		if (rho > eta) {
			lam[tid] *= acc;
			step_type[tid] = 1;
		} else {
			step_type[tid] = 2;
		}
	} else {
		lam[tid] *= dec;
		step_type[tid] = 4;
	}

	if (predicted < 0) {
		lam[tid] *= dec;
		step_type[tid] |= 8;
	}

	if (tid == 0) {
		printf(" rho=%f, actual=%f, f=%f, \\n", rho, actual, f[tid]);
	}

}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = gain_ratio_step_funcid(nparam, dtype)
	return temp.render(funcid=funcid, fp_type=type, nparam=nparam)

class GainRatioStep(CudaFunction):
	def __init__(self, nparam: int, dtype: cp.dtype, write_to_file: bool = False):
		self.funcid = gain_ratio_step_funcid(nparam, dtype)
		self.code = gain_ratio_step_code(nparam, dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.nparam = nparam
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, f, ftp, pars_tp, step, g, h, pars, lam, step_type, mu = 0.25, eta = 0.75, acc = 5.0, dec = 2.0):
		if self.run_func == None:
			self.build()

		N = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(f, ftp, pars_tp, step, g, h, pars, lam, step_type, 
			cp.float64(mu).astype(self.dtype), cp.float64(eta).astype(self.dtype), cp.float64(acc).astype(self.dtype), cp.float64(dec).astype(self.dtype), N))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* f, const {{fp_type}}* ftp, const {{fp_type}}* pars_tp, const {{fp_type}}* step,
	const {{fp_type}}* g, const {{fp_type}}* h, {{fp_type}}* pars, 
	{{fp_type}}* lam, char* step_type, {{fp_type}} mu, {{fp_type}} eta, {{fp_type}} acc, {{fp_type}} dec, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nprobs) {
		if (step_type[tid] == 0) {
			return;
		}

		{{dfuncid}}(f, ftp, pars_tp, step, g, h, pars, lam, step_type, mu, eta, acc, dec, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def clamp_pars_funcid(nparam: int, dtype: cp.dtype):
	return 'clamp_pars' + ctc.dim_type_funcid(nparam, dtype)

def clamp_pars_code(nparam: int, dtype: cp.dtype):
	temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, {{fp_type}}* pars, int tid, int N) 
{
	for (int i = 0; i < {{nparam}}; ++i) {
		int index = i*N+tid;
		{{fp_type}} p = pars[index];
		{{fp_type}} u = upper_bound[index];
		{{fp_type}} l = lower_bound[index];

		if (p > u) {
			pars[index] = u;
		} else if (p < l) {
			pars[index] = l;
		}
	}
}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = clamp_pars_funcid(nparam, dtype)
	return temp.render(funcid=funcid, fp_type=type, nparam=nparam)

class ClampPars(CudaFunction):
	def __init__(self, nparam: int, dtype: cp.dtype, write_to_file: bool = False):
		
		self.funcid = clamp_pars_funcid(nparam, dtype)
		self.code = clamp_pars_code(nparam, dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.nparam = nparam
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, lower_bound, upper_bound, step_type, pars):
		if self.run_func == None:
			self.build()

		N = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(lower_bound, upper_bound, step_type, pars, N))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, const char* step_type, {{fp_type}}* pars, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nprobs) {
		if (step_type[tid] == 0) {
			return;
		}

		{{dfuncid}}(lower_bound, upper_bound, pars, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def gradient_convergence_funcid(nparam: int, dtype: cp.dtype):
	return 'gradient_convergence' + ctc.dim_type_funcid(nparam, dtype)

def gradient_convergence_code(nparam: int, dtype: cp.dtype):
	temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* pars, const {{fp_type}}* g, const {{fp_type}}* f, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, char* step_type, float tol, int tid, int N) 
{
	bool clamped = false;
	{{fp_type}} clamped_norm = 0.0f;
	{{fp_type}} temp1;
	{{fp_type}} temp2;
	for (int i = 0; i < {{nparam}}; ++i) {
		int iidx = i*N+tid;
		temp1 = pars[iidx];
		temp2 = g[iidx];
		temp2 = temp1 - temp2;
		{{fp_type}} u = upper_bound[iidx];
		{{fp_type}} l = lower_bound[iidx];
		if (temp2 > u) {
			clamped = true;
			temp2 = u;
		} else if (temp2 < l) {
			clamped = true;
			temp2 = l;
		}
		temp2 = temp1 - temp2;
		clamped_norm += temp2*temp2;
	}

	if (clamped_norm < tol*(1 + f[tid])) {
		if ((step_type[tid] & 1) || clamped) {
			step_type[tid] = 0;
		}
	}
}
""")

	type = ctc.type_to_typestr(dtype)
	funcid = gradient_convergence_funcid(nparam, dtype)
	return temp.render(funcid=funcid, fp_type=type, nparam=nparam)
	
class GradientConvergence(CudaFunction):
	def __init__(self, nparam: int, dtype: cp.dtype, write_to_file: bool = False):
		self.funcid = gradient_convergence_funcid(nparam, dtype)
		self.code = gradient_convergence_code(nparam, dtype)

		self.write_to_file = write_to_file

		self.type_str = ctc.type_to_typestr(dtype)
		self.nparam = nparam
		self.dtype = dtype

		self.mod = None
		self.run_func = None

	def run(self, pars, g, f, lower_bound, upper_bound, step_type, tol):
		if self.run_func == None:
			self.build()

		N = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, g, f, lower_bound, upper_bound, step_type, cp.float64(tol).astype(self.dtype), N))

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const float* pars, const float* g, const {{fp_type}}* f, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, char* step_type, float tol, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nprobs) {
		if (step_type[tid] == 0) {
			return;
		}
	
		{{dfuncid}}(pars, g, f, lower_bound, upper_bound, step_type, tol, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, fp_type=self.type_str, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)

		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def second_order_levenberg_marquardt_funcid(model_expr: Expr | None, f_gradf: F_GradF, ndata: int, dtype: cp.dtype):
	if model_expr is not None and f_gradf is None:
		return 'second_order_levenberg_marquardt' + ctc.dim_dim_dim_type_funcid(len(model_expr.pars_str), len(model_expr.consts_str), ndata,
			dtype, 'second_order_levenberg_marquardt') + '_' + util.expr_hash(model_expr.expr, 12)
	elif f_gradf is not None and model_expr is None:
		return 'second_order_levenberg_marquardt' + ctc.dim_dim_dim_type_funcid(f_gradf.nparam, f_gradf.nconst, ndata,
			dtype, 'second_order_levenberg_marquardt') + '_' + f_gradf.name
	else:
		raise RuntimeError('One and only one of model_expr or f_gradf must be non None')

def second_order_levenberg_marquardt_code(model_expr: Expr | None, f_gradf: F_GradF | None, ndata: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, 
	{{fp_type}} tol, {{fp_type}} mu, {{fp_type}} eta, {{fp_type}} acc, {{fp_type}} dec,
	{{fp_type}}* params, {{fp_type}}* params_tp, {{fp_type}}* step, {{fp_type}}* lam, char* step_type,
	{{fp_type}}* f, {{fp_type}}* ftp, {{fp_type}}* g, {{fp_type}}* h, {{fp_type}}* hl, int tid, int Nprobs)
{
	// Set gradients and objective functions to zero
	{
		f[tid] = 0.0f;
		ftp[tid] = 0.0f;
		for (int i = 0; i < {{nparam}}; ++i) {
			g[i*Nprobs+tid] = 0.0f;
		}
		for (int i = 0; i < {{nhes}}; ++i) {
			h[i*Nprobs+tid] = 0.0f;
			hl[i*Nprobs+tid] = 0.0f;
		}
	}

	// Calculate gradients
	{
		{{grad_funcid}}(params, consts, data, lam, f, g, h, hl, tid, Nprobs);
	}

	// Solve step
	{
		{{fp_type}}* mat = hl;
		{{fp_type}}* rhs = g;
		{{fp_type}}* sol = step;

		{{fp_type}} mat_copy[{{nparam}}*{{nparam}}];
		{{fp_type}} rhs_copy[{{nparam}}];
		{{fp_type}} sol_copy[{{nparam}}];

		for (int i = 0; i < {{nparam}}; ++i) {
			rhs_copy[i] = rhs[i*Nprobs+tid];
			sol_copy[i] = sol[i*Nprobs+tid];
		}
		int k = 0;
		for (int i = 0; i < {{nparam}}; ++i) {
			for (int j = 0; j <= i; ++j) {
				{{fp_type}} temp = mat[k*Nprobs+tid];
				mat_copy[i*{{nparam}}+j] = temp;
				if (i != j) {
					mat_copy[j*{{nparam}}+i] = temp;
				}
				++k;
			}
		}

		{{gmw81_funcid}}(mat_copy, rhs_copy, sol_copy);

		for (int i = 0; i < {{nparam}}; ++i) {
			sol[i*Nprobs+tid] = sol_copy[i];
		}
	}

	// Remove NaN and Infs
	{
		for (int i = 0; i < {{nparam}}; ++i) {
			int idx = i*Nprobs+tid;

			// Remove inf
			{{fp_type}} si = step[idx];
			if (isnan(si) || isinf(si)) {
				step[idx] = 0.0f;
			}
		}
	}

	// Check convergence
	{
		{{converg_funcid}}(params, g, f, lower_bound, upper_bound, step_type, tol, tid, Nprobs);
		if (step_type[tid] == 0) {
			return;
		}
	}

	// Subtract step from params
	{
		for (int i = 0; i < {{nparam}}; ++i) {
			int idx = i*Nprobs+tid;
			params_tp[idx] = params[idx] - step[idx];
		}
	}

	// Calculate error at new params
	{
		{{fobj_funcid}}(params_tp, consts, data, ftp, tid, Nprobs);
	}

	// Calculate gain ratio and determine step type
	{
		{{gain_funcid}}(f, ftp, params_tp, step, g, h, params, lam, step_type, mu, eta, acc, dec, tid, Nprobs);
	}

	// Clamp parameters to feasible region
	{
		{{clamp_funcid}}(lower_bound, upper_bound, params, tid, Nprobs);
	}

}
""")


	from_expr = model_expr is not None

	nparam = len(model_expr.pars_str) if from_expr else f_gradf.nparam
	nconst = len(model_expr.consts_str) if from_expr else f_gradf.nconst

	nhes = round(nparam*(nparam+1)/2)

	funcid = second_order_levenberg_marquardt_funcid(model_expr, f_gradf, ndata, dtype)

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'second_order_levenberg_marquardt')

	grad_funcid = fghhl_funcid(model_expr, ndata, dtype) if from_expr else f_gradf.gradf.get_device_funcid()
	gmw81_funcid = solver.gmw81_solver_funcid(nparam, dtype)
	fobj_funcid = f_funcid(model_expr, ndata, dtype) if from_expr else f_gradf.f.get_device_funcid()
	gain_funcid = gain_ratio_step_funcid(nparam, dtype)
	clamp_funcid = clamp_pars_funcid(nparam, dtype)
	converg_funcid = gradient_convergence_funcid(nparam, dtype)

	return codestr.render(funcid=funcid, nparam=nparam, nconst=nconst, nhes=nhes, grad_funcid=grad_funcid, fp_type=type,
		gmw81_funcid=gmw81_funcid, fobj_funcid=fobj_funcid, gain_funcid=gain_funcid, clamp_funcid=clamp_funcid, converg_funcid=converg_funcid)

class SecondOrderLevenbergMarquardt(CudaFunction):
	def __init__(self, model_expr: Expr | None, f_gradf: F_GradF | None, ndata: int, dtype: cp.dtype, write_to_file: bool = False):
		expr_true = model_expr is not None
		self.expr_true = expr_true

		self.nparam = len(model_expr.pars_str) if expr_true else f_gradf.nparam
		self.nhes = round(self.nparam * (self.nparam + 1) / 2)
		self.nconst = len(model_expr.consts_str) if expr_true else f_gradf.nconst
		self.ndata = ndata
		self.dtype = dtype
		self.type_str = ctc.type_to_typestr(dtype)

		self.write_to_file = write_to_file

		self.model_expr = model_expr
		self.f_gradf = f_gradf

		self.expr = model_expr.expr if expr_true else None
		self.pars_str = model_expr.pars_str.copy() if expr_true else None
		self.consts_str = model_expr.consts_str.copy() if expr_true else None

		self.funcid = second_order_levenberg_marquardt_funcid(model_expr, f_gradf, ndata, dtype)
		self.code = second_order_levenberg_marquardt_code(model_expr, f_gradf, ndata, dtype)

		self.build()

		self.Nprobs = int(1)

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound,
	{{fp_type}} tol, {{fp_type}} mu, {{fp_type}} eta, {{fp_type}} acc, {{fp_type}} dec,
	{{fp_type}}* params, {{fp_type}}* params_tp, {{fp_type}}* step, {{fp_type}}* lam, char* step_type,
	{{fp_type}}* f, {{fp_type}}* ftp, {{fp_type}}* g, {{fp_type}}* h, {{fp_type}}* hl, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nprobs) {

		if (step_type[tid] == 0) {
			return;
		}

		{{dfuncid}}(consts, data, lower_bound, upper_bound, tol, mu, eta, acc, dec, params, params_tp, step, lam, step_type, f, ftp, g, h, hl, tid, Nprobs);		
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid()+'.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return [
			FGHHL(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file) if self.expr_true else self.f_gradf.gradf,
			F(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file) if self.expr_true else self.f_gradf.f,
			solver.GMW81Solver(self.nparam, self.dtype, self.write_to_file),
			GainRatioStep(self.nparam, self.dtype, self.write_to_file),
			ClampPars(self.nparam, self.dtype, self.write_to_file),
			GradientConvergence(self.nparam, self.dtype, self.write_to_file)]

	def setup(self, pars_t, consts_t, data_t, lower_bound_t, upper_bound_t):
		self.Nprobs = data_t.shape[1]

		self.pars_t = pars_t
		self.consts_t = consts_t
		self.data_t = data_t
		self.lower_bound_t = lower_bound_t
		self.upper_bound_t = upper_bound_t

		self.first_f = cp.empty((1, self.Nprobs), dtype=self.dtype)
		self.last_f = cp.empty((1, self.Nprobs), dtype=self.dtype)

		self.lam_t = 5*cp.ones((1, self.Nprobs), dtype=self.dtype)
		self.step_t = cp.empty((self.nparam, self.Nprobs), dtype=self.dtype)
		self.f_t = cp.empty((1, self.Nprobs), dtype=self.dtype)
		self.ftp_t = cp.empty((1, self.Nprobs), dtype=self.dtype)
		self.g_t = cp.empty((self.nparam, self.Nprobs), dtype=self.dtype)
		self.h_t = cp.empty((self.nhes, self.Nprobs), dtype=self.dtype)
		self.hl_t = cp.empty((self.nhes, self.Nprobs), dtype=self.dtype)
		self.pars_tp_t = cp.empty((self.nparam, self.Nprobs), dtype=self.dtype)
		self.step_type_t = cp.ones((1, self.Nprobs), dtype=cp.int8)

	def run(self, iters: int, tol = cp.float32(1e-8), mu = cp.float32(0.25), eta = cp.float32(0.75), acc = cp.float32(0.2), dec = cp.float32(2.0)):
		if self.run_func == None:
			self.build()

		Nthreads = 32
		blockSize = math.ceil(self.Nprobs / Nthreads)

		for i in range(0,iters):
			self.run_func((blockSize,),(Nthreads,),(self.consts_t, self.data_t, self.lower_bound_t, self.upper_bound_t, 
				tol, mu, eta, acc, dec,
				self.pars_t, self.pars_tp_t, self.step_t, self.lam_t, self.step_type_t, self.f_t, self.ftp_t, self.g_t, self.h_t, self.hl_t, self.Nprobs))

			if i == 0:
				self.first_f = self.f_t.copy()
			if i == iters-1:
				self.last_f = self.f_t.copy()


def search_step_funcid(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	return 'random_search' + ctc.dim_dim_dim_type_funcid(len(pars_str), len(consts_str), ndata,
		dtype, 'random_search') + '_' + util.expr_hash(expr, 12)

def search_step_code(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, 
	{{fp_type}}* best_p, {{fp_type}}* p, {{fp_type}}* best_f, {{fp_type}}* f, int tid, int Nprobs)
{
	
	// Calculate error at new params
	{
		{{fobj_funcid}}(p, consts, data, f, tid, Nprobs);
	}

	// Check if new params is better and keep them in that case
	{
		if (f[tid] < best_f[tid]) {
			for (int i = 0; i < {{nparam}}; ++i) {
				int iidx = i*Nprobs+tid;
				best_p[iidx] = p[iidx];
			}
			best_f[tid] = f[tid];
		}
	}

}
""")

	nparam = len(pars_str)
	nconst = len(consts_str)
	nhes = round(nparam*(nparam+1)/2)

	funcid = search_step_funcid(expr, pars_str, consts_str, ndata, dtype)

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'search_step')

	fobj_funcid = f_funcid(expr, pars_str, consts_str, ndata, dtype)

	return codestr.render(funcid=funcid, nparam=nparam, nconst=nconst, nhes=nhes, fp_type=type,
		fobj_funcid=fobj_funcid)

class SearchStep(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):
		self.nparam = len(pars_str)
		self.nhes = round(self.nparam * (self.nparam + 1) / 2)
		self.nconst = len(consts_str)
		self.ndata = ndata
		self.dtype = dtype
		self.type_str = ctc.type_to_typestr(dtype)

		self.write_to_file = write_to_file

		self.expr = expr
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()

		self.funcid = search_step_funcid(expr, pars_str.copy(), consts_str.copy(), ndata, dtype)
		self.code = search_step_code(expr, pars_str.copy(), consts_str.copy(), ndata, dtype)

		self.Nprobs = int(1)

	def get_device_funcid(self):
		return self.funcid

	def get_kernel_funcid(self):
		funcid = self.funcid
		return 'k_' + funcid

	def get_device_code(self):
		return self.code

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const {{fp_type}}* consts, const {{fp_type}}* data, const {{fp_type}}* lower_bound, const {{fp_type}}* upper_bound, 
	{{fp_type}}* best_p, {{fp_type}}* p, {{fp_type}}* best_f, {{fp_type}}* f, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nprobs) {
		{{dfuncid}}(consts, data, lower_bound, upper_bound, best_p, p, best_f, f, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid, fp_type=self.type_str)

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid()+'.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return [F(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file)]

	def setup(self, consts_t, data_t, lower_bound_t, upper_bound_t):
		self.Nprobs = data_t.shape[1]

		self.consts_t = consts_t
		self.data_t = data_t
		self.lower_bound_t = lower_bound_t
		self.upper_bound_t = upper_bound_t

	def run(self, f, best_f, p, best_p):

		Nthreads = 32
		blockSize = math.ceil(self.Nprobs / Nthreads)

		self.run_func((blockSize,),(Nthreads,),(self.consts_t, self.data_t, self.lower_bound_t, self.upper_bound_t, best_p, p, best_f, f, self.Nprobs))


class RandomSearch(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):
		self.nparam = len(pars_str)
		self.nconst = len(consts_str)
		self.ndata = ndata
		self.dtype = dtype

		self.write_to_file = write_to_file

		self.expr = expr
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()

		self.ss = SearchStep(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file)
		self.ss.build()

		self.Nprobs = int(1)

	def setup(self, consts_t, data_t, lower_bound_t, upper_bound_t):
		self.Nprobs = data_t.shape[1]

		self.best_pars_t = cp.empty((self.nparam, self.Nprobs), dtype=self.dtype)
		self.consts_t = consts_t
		self.data_t = data_t
		self.lower_bound_t = lower_bound_t
		self.upper_bound_t = upper_bound_t

		self.best_f_t = (cp.finfo(self.dtype).max / 5) * cp.ones((1,self.Nprobs), dtype=self.dtype)
		self.f_t = cp.empty((1,self.Nprobs), dtype=self.dtype)
		
	def run(self, iters: int):

		self.ss.setup(self.consts_t, self.data_t, self.lower_bound_t, self.upper_bound_t)

		for i in range(0,iters):

			rand_pars = cp.random.rand(self.nparam, self.Nprobs).astype(dtype=self.dtype, order='C')
			rand_pars *= (self.upper_bound_t - self.lower_bound_t)
			rand_pars += self.lower_bound_t

			self.ss.run(self.f_t, self.best_f_t, rand_pars, self.best_pars_t)


class SecondOrderRandomSearch(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):
		self.nparam = len(pars_str)
		self.nhes = round(self.nparam * (self.nparam + 1) / 2)
		self.nconst = len(consts_str)
		self.ndata = ndata
		self.dtype = dtype

		self.write_to_file = write_to_file

		self.expr = expr
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()

		self.lm = SecondOrderLevenbergMarquardt(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file)
		self.lm.build()
		self.ss = SearchStep(self.expr, self.pars_str.copy(), self.consts_str.copy(), self.ndata, self.dtype, self.write_to_file)
		self.ss.build()

		self.Nprobs = int(1)

	def setup(self, consts_t, data_t, lower_bound_t, upper_bound_t):
		self.Nprobs = data_t.shape[1]

		self.best_pars_t = cp.empty((self.nparam, self.Nprobs), dtype=self.dtype)
		self.consts_t = consts_t
		self.data_t = data_t
		self.lower_bound_t = lower_bound_t
		self.upper_bound_t = upper_bound_t

		self.best_f_t = (cp.finfo(self.dtype).max / 5) * cp.ones((1,self.Nprobs), dtype=self.dtype)
		
	def run(self, iters: int, lm_iters: int = 20, tol = cp.float32(1e-8)):

		self.ss.setup(self.consts_t, self.data_t, self.lower_bound_t, self.upper_bound_t)

		for i in range(0,iters):

			rand_pars = cp.random.rand(self.nparam, self.Nprobs).astype(dtype=self.dtype, order='C')
			rand_pars *= (self.upper_bound_t - self.lower_bound_t)

			rand_pars += self.lower_bound_t

			self.lm.setup(rand_pars, self.consts_t, self.data_t, self.lower_bound_t, self.upper_bound_t)
			self.lm.run(lm_iters, tol)
			
			self.ss.run(self.lm.f_t, self.best_f_t, self.lm.pars_t, self.best_pars_t)





