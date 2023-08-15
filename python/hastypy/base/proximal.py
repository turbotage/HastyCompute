import torch
import cupy as cp
import cupyx
import numpy as np

from torch.utils.dlpack import to_dlpack, from_dlpack

from hastypy.base.opalg import Vector, Operator, Linop
from hastypy.ffi.hasty_svt import Random3DBlocksSVT, Normal3DBlocksSVT


class ProximalOperator(Operator):
	def __init__(self, shape, lamda):
		self.alpha = torch.tensor(1.0, dtype=torch.float32)
		self.lamda = lamda
		super().__init__(shape, shape)
		
	def get_lamda(self):
		return self.lamda

	def set_alpha(self, alpha):
		self.alpha = alpha


class ProximalL1DCT(ProximalOperator):
	def __init__(self, shape, axis, lamda, inplace=True):
		self.axis = axis
		self.inplace = inplace
		super().__init__(shape, lamda)

	@staticmethod
	def tensor_apply(input: torch.Tensor, lamda: cp.array, axis: tuple[bool]):
		ntransdim = len(axis)

		axes = ()
		for i in range(ntransdim):
			if axis[i]:
				axes += (i,)

		def softdct(t: torch.Tensor):
			gpuimg = cupyx.scipy.fft.dctn(cp.array(t), axes=axes)
			gpuimg = cp.exp(1j*cp.angle(gpuimg)) * cp.maximum(0, (cp.abs(gpuimg) - lamda))
			gpuimg = cupyx.scipy.fft.idctn(gpuimg, axes=axes)
			t.copy_(from_dlpack(gpuimg.toDlpack()))
			
			
		def loop_dim(t):
			ndim = len(t.shape)
			if ndim == ntransdim:
				softdct(t)
			else:
				for i in range(t.shape[0]):
					loop_dim(t.select(0,i))

		loop_dim(input)

	def _apply(self, input: Vector):

		if self.inplace:
			output = input
		else:
			output = input.clone()

		al =  cp.array(self.alpha * self.lamda)

		def loop_list(inp: Vector):
			if inp.islist():
				for child in inp.children:
					loop_list(child)
			else:
				ProximalL1DCT.tensor_apply(inp.tensor, al, self.axis)

		loop_list(output)

		return output
