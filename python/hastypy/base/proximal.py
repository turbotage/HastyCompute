import torch
import cupy as cp
import cupyx
import numpy as np

from torch.utils.dlpack import to_dlpack, from_dlpack

from hastypy.base.opalg import Vector, Operator, Linop
from hastypy.base.svt import Random3DBlocksSVT, Normal3DBlocksSVT
import hastypy.base.opalg as opalg

# Proximal operator base class
class ProximalOperator:
	def __init__(self, shape, base_alpha, inplace):
		self.base_alpha = base_alpha if base_alpha is not None else 1.0
		self.inplace = inplace
		self.shape = shape

	def get_base_alpha(self):
		return self.base_alpha

	def set_base_alpha(self, base_alpha):
		self.base_alpha = base_alpha

	# True alpha will always be base_alpha * alpha so postcomposition is applied
	def apply(self, input: Vector, alpha=None) -> Vector:
		if opalg.get_shape(input) != self.shape:
			raise RuntimeError("Incompatible input shape for ProximalOperator")
		
		mul = self.base_alpha * (1.0 if alpha is None else alpha)

		output = self._apply(mul, input)

		if opalg.get_shape(output) != self.shape:
			raise RuntimeError("Incompatible output shape for ProximalOperator")
		
		return output

# proximal operator of zero function, is identity op
class ZeroFunc(ProximalOperator):
	def __init__(self, shape):
		super().__init__(shape, None, inplace=True)

	def _apply(self, alpha, input: Vector) -> Vector:
		return input

# proximal operator of f^*(y) = sup_{x \in X} y^Hx - f(x)
class ConvexConjugate(ProximalOperator):
	def __init__(self, prox: ProximalOperator, base_alpha=None):	
		self.prox = prox
		super().__init__(prox.shape, base_alpha)

	def _apply(self, alpha, input: Vector) -> Vector:
		return input - alpha * self.prox(1 / alpha, input / alpha)

# proximal operator of f(x) = alpha*phi(x) + beta, beta changes nothing
# also all proximal operators essentially performs postcomposition with base_alpha
class Postcomposition(ProximalOperator):
	def __init__(self, prox: ProximalOperator, base_alpha=None):
		self.prox = prox
		super().__init__(prox.shape, base_alpha)

	def _apply(self, alpha, input: Vector) -> Vector:
		return self.prox(alpha, input)

# proximal operator of f(x) = phi(a*x+b) with a != 0
class Precomposition(ProximalOperator):
	def __init__(self, prox: ProximalOperator, a, b, base_alpha=None):
		self.prox = prox
		self.a = a
		self.b = b
		super().__init__(prox.shape, base_alpha)

	def _apply(self, alpha, input: Vector) -> Vector:
		return (1 / self.a) * (self.prox((self.a*self.a)*alpha, self.a*input + self.b) - self.b)

# proximal operator of f(x) = phi(Qx) where Q is unitary
class UnitaryTransform(ProximalOperator):
	def __init__(self, prox: ProximalOperator, unitary: Linop, unitary_adjoint: Linop, base_alpha=None, inplace=False):
		self.prox = prox
		self.unitary = unitary
		self.unitary_adjoint = unitary_adjoint
		if unitary.ishape != unitary_adjoint.oshape:
			raise RuntimeError("Unitary input shape and UnitaryAdjoint output shape did not match")

		super().__init__(unitary.ishape, base_alpha, inplace)

	def _apply(self, alpha, input: Vector) -> Vector:
		return self.unitary_adjoint(self.prox(alpha, self.unitary(input)))

# proximal operator of f(x) = phi(x) + a^Hx + b, b doesn't matter
class AffineAddition(ProximalOperator):
	def __init__(self, prox: ProximalOperator, a, base_alpha=None):
		self.prox = prox
		self.a = a
		super().__init__(prox.shape, base_alpha)

	def _apply(self, alpha, input: Vector) -> Vector:
		return self.prox(alpha, input - alpha*self.a)

# proximal operator of f(x) = phi(x) + (rho/2)||x-a||_2^2
class L2Reg(ProximalOperator):
	def __init__(self, prox: ProximalOperator, rho, a, base_alpha=None):
		self.prox = prox
		self.rho = rho
		self.a = a
		super().__init__(prox.shape, base_alpha)

	def _apply(self, alpha, input: Vector) -> Vector:
		alphahat = alpha / (1 + alpha*self.rho)
		return self.prox(alphahat, (alphahat / alpha)*input + (self.rho*alphahat)*self.a)







class ProximalL1DCT(ProximalOperator):
	def __init__(self, shape, axis, lamda, inplace=True):
		self.axis = axis
		super().__init__(shape, lamda, inplace)

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
		if streams is None:
			cudev = torch.device('cuda:0')
			streams = [torch.cuda.default_stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]

		if svt_options.random:
			self._svtop = Random3DBlocksSVT(shape, streams, svt_options.nblocks, 
				   svt_options.block_shape, svt_options.thresh, svt_options.soft)
		else:
			self._svtop = Normal3DBlocksSVT(shape, streams, svt_options.block_strides,
				   svt_options.block_shape, svt_options.block_iter, svt_options.thresh, svt_options.soft)

		super().__init__(shape, svt_options.thresh, inplace)

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

