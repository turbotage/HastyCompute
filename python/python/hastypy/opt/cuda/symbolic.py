import math
import numpy as np

import cupy as cp
from jinja2 import Template
from symengine import sympify


#from ..sym import sym
from hastypy.opt.sym import util

from hastypy.opt.cuda import linalg
from hastypy.opt.cuda import cuda_program as cudap
from hastypy.opt.cuda.cuda_program import CudaFunction, CudaTensor
from hastypy.opt.cuda.cuda_program import CudaTensorChecking as ctc


def eval_funcid(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	return 'eval' + ctc.dim_dim_dim_type_funcid(len(pars_str), len(consts_str), ndata,
		dtype, 'eval') + '_' + util.expr_hash(expr, 12)

def eval_code(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, {{fp_type}}* eval, 
	{{fp_type}}* jac, {{fp_type}}* hes, int tid, int N, int Nelem) 
{
	{{fp_type}} pars[{{nparam}}];
	int bucket = tid / {{ndata}};
	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*N+bucket];
	}

{{sub_expr}}

{{eval_expr}}

}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'eval')

	nparam = len(pars_str)
	nconst = len(consts_str)

	funcid = eval_funcid(expr, pars_str, consts_str, ndata, dtype)

	sym_expr = sympify(expr)
	# convert parameter names to ease kernel generation
	for k in range(0,len(pars_str)):
		temp = pars_str[k]
		pars_str[k] = 'parvar_' + temp
		sym_expr = sym_expr.subs(temp, pars_str[k])

	for k in range(0,len(consts_str)):
		temp = consts_str[k]
		consts_str[k] = 'convar_' + temp
		sym_expr = sym_expr.subs(temp, consts_str[k])

	substs, reduced = util.res(str(sym_expr), pars_str, consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	eval_str = '\teval[tid] = '+cuprint.tcs_f(reduced[0])+';'

	for k in range(len(pars_str)):
		p = pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		eval_str = eval_str.replace(p, repl)

	for k in range(len(consts_str)):
		c = consts_str[k]
		repl = 'consts['+str(k)+'*Nelem+tid]'
		sub_str = sub_str.replace(c, repl)
		eval_str = eval_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst,
		sub_expr=sub_str, eval_expr=eval_str)


	return rjh_kernel

class Eval(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):

		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()
		self.ndata = ndata

		self.write_to_file = write_to_file

		self.funcid = eval_funcid(expr, self.pars_str, self.consts_str, ndata, dtype)
		self.code = eval_code(expr, self.pars_str, self.consts_str, ndata, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, eval, jac, hes):
		if self.run_func == None:
			self.build()

		Nelem = hes.shape[1]
		N = round(Nelem / self.ndata)

		Nthreads = 32
		blockSize = np.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, eval, jac, hes, N, Nelem))

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
void {{funcid}}(const {{fp_type}}* pars, const {{fp_type}}* consts,
	{{fp_type}}* eval, int N, int Nelem) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(pars, consts, eval, jac, hes, tid, N, Nelem);
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
			with open(self.get_device_funcid(), "w") as f:
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


def eval_jac_hes_funcid(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	return 'eval_jac_hes' + ctc.dim_dim_dim_type_funcid(len(pars_str), len(consts_str), ndata,
		dtype, 'eval_jac_hes') + '_' + util.expr_hash(expr, 12)

def eval_jac_hes_code(expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, {{fp_type}}* eval, 
	{{fp_type}}* jac, {{fp_type}}* hes, int tid, int N, int Nelem) 
{
	{{fp_type}} pars[{{nparam}}];
	int bucket = tid / {{ndata}};
	for (int i = 0; i < {{nparam}}; ++i) {
		pars[i] = params[i*N+bucket];
	}

{{sub_expr}}

{{eval_expr}}

{{jac_expr}}

{{hes_expr}}

}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'eval_jac_hes')

	nparam = len(pars_str)
	nconst = len(consts_str)

	funcid = eval_jac_hes_funcid(expr, pars_str, consts_str, ndata, dtype)

	sym_expr = sympify(expr)
	# convert parameter names to ease kernel generation
	for k in range(0,len(pars_str)):
		temp = pars_str[k]
		pars_str[k] = 'parvar_' + temp
		sym_expr = sym_expr.subs(temp, pars_str[k])

	for k in range(0,len(consts_str)):
		temp = consts_str[k]
		consts_str[k] = 'convar_' + temp
		sym_expr = sym_expr.subs(temp, consts_str[k])

	substs, reduced = util.res_jac_hes(str(sym_expr), pars_str, consts_str)
	cuprint = util.CUDAPrinter()

	sub_str = ""
	for s in substs:
		sub_str += '\t'+type+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'

	eval_str = '\teval[tid] = '+cuprint.tcs_f(reduced[0])+';'

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
		jac_str += '\tjac['+str(k)+'*Nelem+tid] = '+ctstr+';\n'

	hes_str = ""
	for k in range(round(nparam*(nparam+1)/2)):
		s = reduced[(nparam + 1) + k]
		ctstr = ""
		if dtype == cp.float32:
			ctstr = cuprint.tcs_f(s)
		else:
			ctstr = cuprint.tcs_d(s)
		if ctstr == '0':
			ctstr = '0.0f'
		hes_str += '\thes['+str(k)+'*Nelem+tid] = '+ctstr+';\n'

	for k in range(len(pars_str)):
		p = pars_str[k]
		repl = 'pars['+str(k)+']'
		sub_str = sub_str.replace(p, repl)
		eval_str = eval_str.replace(p, repl)
		jac_str = jac_str.replace(p, repl)
		hes_str = hes_str.replace(p, repl)

	for k in range(len(consts_str)):
		c = consts_str[k]
		repl = 'consts['+str(k)+'*Nelem+tid]'
		sub_str = sub_str.replace(c, repl)
		eval_str = eval_str.replace(c, repl)
		jac_str = jac_str.replace(c, repl)
		hes_str = hes_str.replace(c, repl)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type,
		nparam=nparam, nconst=nconst,
		sub_expr=sub_str, eval_expr=eval_str, jac_expr=jac_str, hes_expr=hes_str)


	return rjh_kernel

class EvalJacHes(CudaFunction):
	def __init__(self, expr: str, pars_str: list[str], consts_str: list[str], ndata: int, dtype: cp.dtype, write_to_file: bool = False):

		self.type_str = ctc.type_to_typestr(dtype)
		self.pars_str = pars_str.copy()
		self.consts_str = consts_str.copy()
		self.ndata = ndata

		self.write_to_file = write_to_file

		self.funcid = eval_jac_hes_funcid(expr, self.pars_str, self.consts_str, ndata, dtype)
		self.code = eval_jac_hes_code(expr, self.pars_str, self.consts_str, ndata, dtype)

		self.mod = None
		self.run_func = None

	def run(self, pars, consts, eval, jac, hes):
		if self.run_func == None:
			self.build()

		Nelem = hes.shape[1]
		N = round(Nelem / self.ndata)

		Nthreads = 32
		blockSize = np.ceil(N / Nthreads)
		self.run_func((blockSize,),(Nthreads,),(pars, consts, eval, jac, hes, N, Nelem))

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
void {{funcid}}(const {{fp_type}}* pars, const {{fp_type}}* consts,
	{{fp_type}}* eval, {{fp_type}}* jac, {{fp_type}}* hes, int N) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < N) {
		{{dfuncid}}(pars, consts, eval, jac, hes, tid, N);
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
			with open(self.get_device_funcid(), "w") as f:
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


