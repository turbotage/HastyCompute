
from jinja2 import Template
import cupy as cp
#from numba import jit
import numpy as np
import math

from hastypy.opt.cuda.cuda_program import CudaFunction, CudaTensor
from hastypy.opt.cuda.cuda_program import CudaTensorChecking as ctc

from hastypy.opt.cuda import permute


def sum_every_n_upto_m_funcid(n: int, m: int, dtype: cp.dtype):
	return 'sum_every_n_upto_m' + ctc.dim_dim_type_funcid(n, m, dtype)

def sum_every_n_upto_m_code(n: int, m: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}({{fp_type}}* sred, unsigned int N) 
{
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int rem = tid % {{num_threads_per_m}};
	unsigned int nid = (tid - rem) * {{melem}} / {{num_threads_per_m}} + 2 * rem * {{nelem}};

	if (nid < N && rem != {{num_threads_per_m}}) {
		sred[nid] += sred[nid+{{nelem}}];
	}

}
""")

	ntpm = math.ceil(m / (2*n))

	type_str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'sum_every_n_upto_m')
	
	funcid = sum_every_n_upto_m_funcid(n, m, dtype)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type_str, num_threads_per_m=ntpm,
		melem=m, nelem=n)

	return rjh_kernel

class SumEveryNUptoM(CudaFunction):
	def __init__(self, n: int, m: int, dtype: cp.dtype):
		self.n = n
		self.m = m
		self.dtype = dtype

		self.funcid = sum_every_n_upto_m_funcid(n, m, dtype)
		self.code = sum_every_n_upto_m_code(n, m, dtype)

		self.mod = None
		self.run_func = None

	def run(self, sred):
		if self.run_func == None:
			self.build()

		batch_size = 1
		for s in sred.shape:
			batch_size *= s
		Nthreads = 32
		threads_per_m = np.ceil(self.m / (2*self.n))
		blockSize = np.ceil(batch_size / Nthreads / threads_per_m)
		self.run_func((blockSize,),(Nthreads,),(sred, batch_size))

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
void {{funcid}}({{fp_type}}* sred, unsigned int N) 
{
	{{dfuncid}}(sred, N);
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		self.mod = cp.RawModule(code=self.get_full_code())
		self.run_func = self.mod.get_function(self.get_kernel_funcid())

	def get_deps(self):
		return list()


def sum_upto_m_funcid(m: int, dtype: cp.dtype):
	return 'sum_every_n_upto_m' + ctc.dim_type_funcid(m, dtype)

def sum_upto_m_code(m: int, dtype: cp.dtype):
	
	rjh_temp = Template(
"""
__device__
void {{funcid}}({{fp_type}}* sred, unsigned int N) 
{
{{partial_sums}}
}
""")

	calls = ""
	n = 1
	while (m / (2*n)) >= 1:
		ntpm = math.ceil(m / (2*n))
		calls += '\t' + sum_every_n_upto_m_funcid(n, m, dtype) + '(sred, N);\n'
		n += 1

	type_str = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'sum_every_n_upto_m')
	
	funcid = sum_every_n_upto_m_funcid(n, m, dtype)

	rjh_kernel = rjh_temp.render(funcid=funcid, fp_type=type_str, partial_sums=calls)

	return rjh_kernel

class SumUptoM(CudaFunction):
	def __init__(self, m: int, dtype: cp.dtype):

		self.m = m
		self.dtype = dtype

		self.funcid = sum_upto_m_funcid(m, dtype)
		self.code = sum_upto_m_code(m, dtype)

		self.mod = None
		self.run_func = None

	def run(self, sred):
		if self.run_func == None:
			self.build()

		batch_size = 1
		for s in sred.shape:
			batch_size *= s
		Nthreads = 32
		blockSize = np.ceil(batch_size / Nthreads / np.floor(self.m / 2))
		self.run_func((blockSize,),(Nthreads,),(sred, batch_size))

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
void {{funcid}}({{fp_type}}* sred, unsigned int N) 
{
	{{dfuncid}}(sred, N);
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()

		return temp.render(funcid=fid, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		self.mod = cp.RawModule(code=self.get_full_code())
		self.run_func = self.mod.get_function(self.get_kernel_funcid())

	def get_deps(self):
		deps = []
		n = 1
		while (float(self.m) / 2.0*float(n)) >= 1.0:
			ntpm = math.ceil(self.m / (2*n))
			deps.append(SumEveryNUptoM(n, self.m, self.dtype))
			n += 1
		return deps


