from jinja2 import Template
import cupy as cp
#from numba import jit

import math

from hastypy.opt.cuda.cuda_program import CudaFunction, CudaTensor
from hastypy.opt.cuda.cuda_program import CudaTensorChecking as ctc


def lid_funcid():
	return 'lid'

def lid_code():
	codestr = Template(
"""
__device__
int lid(int i, int j) 
{
	return i*(i+1)/2 + j;
}
""")
	funcid = lid_funcid()
	return codestr.render(funcid=funcid)

class LID(CudaFunction):
	def __init__(self):
		self.funcid = lid_funcid()
		self.code = lid_code()

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


def max_diag_abs_funcid(ndim: int, dtype: cp.dtype):
	return 'max_diag_abs' + ctc.dim_type_funcid(ndim, dtype)

def max_diag_abs_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
int {{funcid}}(const {{fp_type}}* mat, int offset) 
{
	{{fp_type}} max_abs = -1.0f;
	int max_index = 0;
	for (int i = offset; i < {{ndim}}; ++i) {
		if ({{abs_fid}}(mat[i*{{ndim}}+i]) > max_abs) {
			max_index = i;
		}
	}
	return max_index;
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'max_diag_abs')

	funcid = max_diag_abs_funcid(ndim, dtype)
	abs_fid = 'fabsf' if dtype == cp.float32 else 'fabs'

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, abs_fid=abs_fid)

class MaxDiagAbs(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = max_diag_abs_funcid(ndim, dtype)
		self.code = max_diag_abs_code(ndim, dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


def row_interchange_i_funcid(ndim: int, dtype: cp.dtype):
	return 'row_interchange_i' + ctc.dim_type_funcid(ndim, dtype)

def row_interchange_i_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int ii, int jj) 
{
	for (int k = 0; k < {{ndim}}; ++k) {
		int ikn = ii*{{ndim}}+k;
		int jkn = jj*{{ndim}}+k;

		{{fp_type}} temp;
		temp = mat[ikn];
		mat[ikn] = mat[jkn];
		mat[jkn] = temp;
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'row_interchange_i')

	funcid = row_interchange_i_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class RowInterchangeI(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = row_interchange_i_funcid(ndim, dtype)
		self.code = row_interchange_i_code(ndim, dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


def col_interchange_i_funcid(ndim: int, dtype: cp.dtype):
	return 'col_interchange_i' + ctc.dim_type_funcid(ndim, dtype)

def col_interchange_i_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int ii, int jj) 
{
	for (int k = 0; k < {{ndim}}; ++k) {
		int kin = k*{{ndim}}+ii;
		int kjn = k*{{ndim}}+jj;

		{{fp_type}} temp;
		temp = mat[kin];
		mat[kin] = mat[kjn];
		mat[kjn] = temp;
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'col_interchange_i')

	funcid = col_interchange_i_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class ColInterchangeI(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = col_interchange_i_funcid(ndim, dtype)
		self.code = col_interchange_i_code(ndim, dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


def diag_pivot_funcid(ndim: int, dtype: cp.dtype):
	return 'diag_pivot' + ctc.dim_type_funcid(ndim, dtype)

def diag_pivot_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}({{fp_type}}* mat, int* perm) 
{
	for (int i = 0; i < {{ndim}}; ++i) {
		perm[i] = i;
	}
	for (int i = 0; i < {{ndim}}; ++i) {
		int max_abs = {{max_diag_abs_fid}}(mat, i);
		{{row_interchange_fid}}(mat, i, max_abs);
		{{col_interchange_fid}}(mat, i, max_abs);
		int temp = perm[i];
		perm[i] = perm[max_abs];
		perm[max_abs] = temp;
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'diag_pivot')

	funcid = diag_pivot_funcid(ndim, dtype)
	max_diag_abs_fid = max_diag_abs_funcid(ndim, dtype)
	row_fid = row_interchange_i_funcid(ndim, dtype)
	col_fid = col_interchange_i_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim, 
		max_diag_abs_fid=max_diag_abs_fid, row_interchange_fid=row_fid, col_interchange_fid=col_fid)

class DiagPivot(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = diag_pivot_funcid(ndim, dtype)
		self.code = diag_pivot_code(ndim, dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return [MaxDiagAbs(self.ndim, self.dtype), RowInterchangeI(self.ndim, self.dtype), 
			ColInterchangeI(self.ndim, self.dtype)]


def permute_vec_funcid(ndim: int, dtype: cp.dtype):
	return 'permute_vec' + ctc.dim_type_funcid(ndim, dtype)

def permute_vec_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* vec, const int* perm, {{fp_type}}* ovec) 
{
	for (int i = 0; i < {{ndim}}; ++i) {
		ovec[i] = vec[perm[i]];
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'permute_vec')

	funcid = permute_vec_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class PermuteVec(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = permute_vec_funcid(ndim, dtype)
		self.code = permute_vec_code(ndim, dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


def inv_permute_vec_funcid(ndim: int, dtype: cp.dtype):
	return 'inv_permute_vec' + ctc.dim_type_funcid(ndim, dtype)

def inv_permute_vec_code(ndim: int, dtype: cp.dtype):
	codestr = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* vec, const int* perm, {{fp_type}}* ovec) 
{
	for (int i = 0; i < {{ndim}}; ++i) {
		ovec[perm[i]] = vec[i];
	}
}
""")

	type = ctc.check_fp32_or_fp64(CudaTensor(None, dtype), 'inv_permute_vec')

	funcid = inv_permute_vec_funcid(ndim, dtype)

	return codestr.render(funcid=funcid, fp_type=type, ndim=ndim)

class InvPermuteVec(CudaFunction):
	def __init__(self, ndim: int, dtype: cp.dtype):
		self.ndim = ndim
		self.dtype = dtype
		self.funcid = inv_permute_vec_funcid(ndim, dtype)
		self.code = inv_permute_vec_code(ndim, dtype)

	def get_device_funcid(self):
		return self.funcid

	def get_device_code(self):
		return self.code

	def get_deps(self):
		return list()


