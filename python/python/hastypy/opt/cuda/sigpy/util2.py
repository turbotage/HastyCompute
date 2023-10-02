
import cupy as cp
import cupyx as cpx

import numpy as np
import math

import hastypy.opt.cuda.sigpy.linop as sp_linop
from hastypy.opt.cuda.sigpy.linop import Linop

from cupyx.scipy.sparse.linalg import LinearOperator

class CupyLinopWrapper(LinearOperator):
	def __init__(self, sp_linop: Linop, dtype: cp.dtype):
		self.sp_linop = sp_linop

		self.shape = (np.prod(sp_linop.oshape), np.prod(sp_linop.ishape))
		self.dtype = dtype

		super().__init__(self.dtype, self.shape)

	#@property
	#def shape(self):
	#	return (sp_linop.oshape[0], sp_linop.ishape[0])

	def _matvec(self, v):
		return self.sp_linop.apply(v.reshape(self.sp_linop.ishape)).flatten()

	@property
	def S(self):
		return self.sp_linop


