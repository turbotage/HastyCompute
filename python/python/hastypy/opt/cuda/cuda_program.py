import cupy as cp
from enum import Enum

import math


class CudaLinkage(Enum):
	GLOBAL = 1
	LOCAL = 2

class CudaTensor:
	def __init__(self, shape: list[int], dtype: cp.dtype):
		self.shape = shape
		self.dtype = dtype

class CudaFunction:
	def __init__(self):
		self.deps: dict[str, CudaFunction] = []

	def run(*args):
		raise NotImplementedError()

	def get_device_funcid(self):
		raise NotImplementedError()

	def get_kernel_funcid(self):
		raise NotImplementedError()

	def get_device_code(self):
		raise NotImplementedError()

	def get_kernel_code(self):
		raise NotImplementedError()

	def get_deps(self):
		return self.deps

class CudaTensorChecking:

	@staticmethod
	def type_funcid(dtype: cp.dtype, func_name: str = None):
		type_qualifier: str
		if dtype == cp.float32:
			type_qualifier = 'f'
		elif dtype == cp.float64:
			type_qualifier = 'd'
		else:
			raise RuntimeError('does only support fp32 and fp64 in ' + func_name if func_name != None else '')

		return '_' + type_qualifier

	@staticmethod
	def dim_type_funcid(ndim: int, dtype: cp.dtype, func_name: str = None):
		type_qualifier: str
		if dtype == cp.float32:
			type_qualifier = 'f'
		elif dtype == cp.float64:
			type_qualifier = 'd'
		else:
			raise RuntimeError('does only support fp32 and fp64 in ' + func_name if func_name != None else '')

		return '_' + str(ndim) + '_' + type_qualifier

	@staticmethod
	def dim_dim_type_funcid(ndim1: int, ndim2: int, dtype: cp.dtype, func_name: str = None):
		type_qualifier: str
		if dtype == cp.float32:
			type_qualifier = 'f'
		elif dtype == cp.float64:
			type_qualifier = 'd'
		else:
			raise RuntimeError('does only support fp32 and fp64 in ' + func_name if func_name != None else '')

		return '_' + str(ndim1) + '_' + str(ndim2) + '_' + type_qualifier

	@staticmethod
	def dim_dim_dim_type_funcid(ndim1: int, ndim2: int, ndim3: int, dtype: cp.dtype, func_name: str = None):
		type_qualifier: str
		if dtype == cp.float32:
			type_qualifier = 'f'
		elif dtype == cp.float64:
			type_qualifier = 'd'
		else:
			raise RuntimeError('does only support fp32 and fp64 in ' + func_name if func_name != None else '')

		return '_' + str(ndim1) + '_' + str(ndim2) + '_' + str(ndim3) + '_' + type_qualifier

	@staticmethod
	def check_integer(t: CudaTensor, func_name: str):
		if t.dtype != cp.int32:
			raise RuntimeError(t.name + ' was not integer in ' + func_name)

	@staticmethod
	def check_scalar(t: CudaTensor, func_name: str):
		if max(t.shape) != 1:
			raise RuntimeError(t.name + ' was not scalar in ' + func_name)

	@staticmethod
	def type_to_typestr(dtype: cp.dtype):
		type: str
		if dtype == cp.float32:
			type = 'float'
		elif dtype == cp.float64:
			type = 'double'
		else:
			raise RuntimeError('Invalid type')

		return type

	@staticmethod
	def check_fp32_or_fp64(t: CudaTensor, func_name: str):
		type: str
		if t.dtype == cp.float32:
			type = 'float'
		elif t.dtype == cp.float64:
			type = 'double'
		else:
			raise RuntimeError(t.name + ' was not fp32 and fp64 in ' + func_name)

		return type

	@staticmethod
	def check_square_mat(mat: CudaTensor, func_name: str):
		if len(mat.shape) != 3:
			raise RuntimeError(mat.name + ' shapes array must be 3 long in ' + func_name)
		if mat.shape[1] != mat.shape[2]:
			raise RuntimeError(mat.name + ' must be square in ' + func_name)

	@staticmethod
	def check_is_vec(vec: CudaTensor, func_name: str):
		if len(vec.shape) != 3:
			raise RuntimeError(vec.name + ' must be a vector in ' + func_name)

		if vec.shape[1] == 1 or vec.shape[2] == 1:
			return

		raise RuntimeError(vec.name + ' had no singleton dimension in ' + func_name)

	@staticmethod
	def check_is_same_shape(t1: CudaTensor, t2: CudaTensor, func_name: str):
		if t1.shape != t2.shape:
			raise RuntimeError(t1.name + ' and ' + t2.name + ' must have equal shape in ' + func_name)

	@staticmethod
	def check_matrix_multipliable(m1: CudaTensor, m2: CudaTensor, func_name: str):
		if len(m1.shape) != 3:
			raise RuntimeError(m1.name + ' shapes array must be 3 long in ' + func_name)
		if len(m2.shape) != 3:
			raise RuntimeError(m2.name + ' shapes array must be 3 long in ' + func_name)
		if m1.shape[2] != m2.shape[1]:
			raise RuntimeError(m1.name + ' and ' + m2.name + ' must agree along inner dimension in ' + func_name)
		if m1.shape[0] != m2.shape[0]:
			raise RuntimeError(m1.name + ' and ' + m2.name + ' must have the same batch dimension')

	@staticmethod
	def check_vecmat_multipliable(v: CudaTensor, m: CudaTensor, func_name: str):
		if len(m.shape) != 3:
			raise RuntimeError(m.name + ' shapes array must be 3 long in ' + func_name)
		if v.shape[2] != m.shape[1]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must agree along inner dimension in ' + func_name)
		if v.shape[0] != m.shape[0]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must have the same batch dimension')

	@staticmethod
	def check_matvec_multipliable(m: CudaTensor, v: CudaTensor, func_name: str):
		if len(m.shape) != 3:
			raise RuntimeError(m.name + ' shapes array must be 3 long in ' + func_name)
		if m.shape[2] != v.shape[1]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must agree along inner dimension in ' + func_name)
		if v.shape[0] != m.shape[0]:
			raise RuntimeError(v.name + ' and ' + m.name + ' must have the same batch dimension')

def from_cu_tensor(t: CudaTensor, rand=False, zeros=False, ones=False, empty=True):
	if rand:
		return cp.random.rand(*(t.shape), dtype=t.dtype)
	elif zeros:
		return cp.zeros(shape=tuple(t.shape), dtype=t.dtype)
	elif ones:
		return cp.ones(shape=tuple(t.shape), dtype=t.dtype)
	elif empty:
		return cp.empty(shape=tuple(t.shape), dtype=t.dtype)
	raise RuntimeError('from_cu_tensor can only construct uniform random, zeros, ones or empty')

def gen_deps_dict(func: CudaFunction, deps: list[CudaFunction], keys: set[str]):
	for f in func.get_deps():
		gen_deps_dict(f, deps, keys)

	fid = func.get_device_funcid()
	if fid not in keys:
		deps.append(func)
		keys.add(fid)

def code_gen_walking(func: CudaFunction, code: str):
	deps: list[CudaFunction] = []
	keys: set[str] = set()
	gen_deps_dict(func, deps, keys)

	for f in deps:
		code += f.get_device_code() + '\n'

	code += func.get_kernel_code() + '\n'

	return code

def print_shape_type_contig(tensor: cp.ndarray):
	print(tensor.shape, tensor.dtype, tensor._c_contiguous)

def compact_to_full(mat):
	nmat = mat.shape[0]
	n = math.floor(math.sqrt(2*nmat))
	retmat = cp.empty((n,n))
	k = 0
	for i in range(0,n):
		for j in range(0,i+1):
			retmat[i,j] = mat[k]
			if i != j:
				retmat[j,i] = mat[k]
			k += 1
	return retmat

def compact_to_LD(mat):
	nmat = mat.shape[0]
	n = math.floor(math.sqrt(2*nmat))
	L = cp.zeros((n,n))
	D = cp.zeros((n,n))
	k = 0
	for i in range(0,n):
		for j in range(0,i+1):
			if i != j:
				L[i,j] = mat[k]
			else:
				L[i,j] = 1.0
				D[i,j] = mat[k]
			k += 1
	return (L,D)