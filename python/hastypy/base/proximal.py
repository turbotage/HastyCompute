import torch
import cupy as cp
import cupyx
import numpy as np

from torch.utils.dlpack import to_dlpack, from_dlpack

from hastypy.base.opalg import Vector, Operator, Linop
from hastypy.base.svt import Random3DBlocksSVT, Normal3DBlocksSVT


class ProximalOperator(Operator):
	def __init__(self, shape, lamda):
		self.alpha = torch.tensor(1.0, dtype=torch.float32)
		self.lamda = lamda
		super().__init__(shape, shape)
		
	def get_lamda(self):
		return self.lamda

	def set_alpha(self, alpha):
		self.alpha = alpha


class UnitaryTransform(ProximalOperator):
	def __init__(self, prox: ProximalOperator, unitary: Linop, unitary_adjoint: Linop):
		self.prox = prox
		self.unitary = unitary
		self.unitary_adjoint = unitary_adjoint
		if unitary.ishape != unitary_adjoint.oshape:
			raise RuntimeError("Unitary input shape and UnitaryAdjoint output shape did not match")

		super().__init__(unitary.ishape, None)

	def get_lamda(self):
		return self.prox.get_lamda()

	def set_alpha(self, alpha):
		return self.prox.set_alpha(alpha)

	def _apply(self, input: Vector) -> Vector:
		return self.unitary_adjoint(self.prox(self.unitary(input)))


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

	def _apply(self, input: Vector) -> Vector:

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


class SVTOptions:
	def __init__(self, block_shapes=[16,16,16], block_strides=[16,16,16], block_iter: int = 4, 
			random=False, nblocks=1, thresh: float = 0.01, soft: bool = True):
		self.block_shape = block_shapes
		self.block_strides = block_strides
		self.block_iter = block_iter
		self.random = random
		self.nblocks = nblocks
		self.thresh = thresh
		self.soft = soft

class Proximal3DSVT(ProximalOperator):
	def __init__(self, shape, svt_options: SVTOptions, streams: list[torch.Stream] | None = None, inplace=True):

		self.inplace = inplace

		if streams is None:
			cudev = torch.device('cuda:0')
			streams = [torch.cuda.default_stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]

		if svt_options.random:
			self._svtop = Random3DBlocksSVT(shape, streams, svt_options.nblocks, 
				   svt_options.block_shape, svt_options.thresh, svt_options.soft)
		else:
			self._svtop = Normal3DBlocksSVT(shape, streams, svt_options.block_strides,
				   svt_options.block_shape, svt_options.block_iter, svt_options.thresh, svt_options.soft)

		super().__init__(shape, svt_options.thresh)

	def set_alpha(self, alpha):
		super().set_alpha(alpha)
		return self._svtop.update_thresh(self.alpha * self.lamda)
	
	def _apply(self, input: Vector) -> Vector:
		if self.inplace:
			output = input
		else:
			output = input.clone()

		self._svtop._apply(output)
		return output

