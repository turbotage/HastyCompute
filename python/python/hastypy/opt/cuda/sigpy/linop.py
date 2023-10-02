
import cupy as cp
import numpy as np
import math

import hastypy.opt.cuda.sigpy.fourier as sf
import hastypy.opt.cuda.sigpy.util as su



class Linop():
	def __init__(self, oshape, ishape, repr_str=None):
		self.oshape = list(oshape)
		self.ishape = list(ishape)

		su._check_shape_positive(oshape)
		su._check_shape_positive(ishape)

		if repr_str is None:
			self.repr_str = self.__class__.__name__
		else:
			self.repr_str = repr_str

		self.adj = None
		self.normal = None


	def _check_ishape(self, input):
		for i1, i2 in zip(input.shape, self.ishape):
			if i2 != -1 and i1 != i2:
				raise ValueError(
					'input shape mismatch for {s}, got {input_shape}'.format(
						s=self, input_shape=input.shape))

	def _check_oshape(self, output):
		for o1, o2 in zip(output.shape, self.oshape):
			if o2 != -1 and o1 != o2:
				raise ValueError(
					'output shape mismatch for {s}, got {output_shape}'.format(
						s=self, output_shape=output.shape))

	def _apply(self, input):
		raise NotImplementedError

	def apply(self, input):
		try:
			self._check_ishape(input)
			output = self._apply(input)
			self._check_oshape(output)
		except Exception as e:
			raise RuntimeError('Exceptions from {}.'.format(self)) from e

		return output

	def _adjoint_linop(self):
		raise NotImplementedError

	def _normal_linop(self):
		return self.H * self

	@property
	def H(self):
		if self.adj is None:
			self.adj = self._adjoint_linop()
		return self.adj

	@property
	def N(self):
		if self.normal is None:
			self.normal = self._normal_linop()
		return self.normal

	def __call__(self, input):
		return self.__mul__(input)

	def __mul__(self, input):
		if isinstance(input, Linop):
			return Compose([self, input])
		elif np.isscalar(input):
			M = Multiply(self.ishape, input)
			return Compose([self, M])
		elif isinstance(input, cp.get_array_module(input).ndarray):
			return self.apply(input)

		return NotImplemented

	def __rmul__(self, input):
		if np.isscalar(input):
			M = Multiply(self.oshape, input)
			return Compose([M, self])

		return NotImplemented

	def __add__(self, input):
		if isinstance(input, Linop):
			return Add([self, input])
		else:
			raise NotImplementedError

	def __neg__(self):
		return -1 * self

	def __sub__(self, input):
		return self.__add__(-input)

	def __repr__(self):
		return '<{oshape}x{ishape}> {repr_str} Linop>'.format(
			oshape=self.oshape, ishape=self.ishape, repr_str=self.repr_str)

class Identity(Linop):
	def __init__(self, shape):
		super().__init__(shape, shape)


	def _apply(self, input):
		return input

	def _adjoint_linop(self):
		return self

	def _normal_linop(self):
		return 

class Add(Linop):
	def __init__(self, linops):
		su._check_linops_same_ishape(linops)
		su._check_linops_same_oshape(linops)

		self.linops = linops
		oshape = linops[0].oshape
		ishape = linops[0].ishape

		super().__init__(
			oshape, ishape,
			repr_str=' + '.join([linop.repr_str for linop in linops]))


	def _apply(self, input):
		output = 0
		for linop in self.linops:
			output += linop(input)

		return output

	def _adjoint_linop(self):
		return Add([linop.H for linop in self.linops])

def _check_compose_linops(linops):
	for linop1, linop2 in zip(linops[:-1], linops[1:]):
		if (linop1.ishape != linop2.oshape):
			raise ValueError('cannot compose {linop1} and {linop2}.'.format(
				linop1=linop1, linop2=linop2))

def _combine_compose_linops(linops):
	combined_linops = []
	for linop in linops:
		if isinstance(linop, Compose):
			combined_linops += linop.linops
		else:
			combined_linops.append(linop)

	return combined_linops

class Compose(Linop):
	def __init__(self, linops):
		_check_compose_linops(linops)
		self.linops = _combine_compose_linops(linops)

		super().__init__(
			self.linops[0].oshape, self.linops[-1].ishape,
			repr_str=' * '.join([linop.repr_str for linop in linops]))


	def _apply(self, input):
		output = input
		for linop in self.linops[::-1]:
			output = linop(output)

		return output

	def _adjoint_linop(self):
		return Compose([linop.H for linop in self.linops[::-1]])

def _get_multiply_oshape(ishape, mshape):
	ishape_exp, mshape_exp = su._expand_shapes(ishape, mshape)
	max_ndim = max(len(ishape), len(mshape))
	oshape = []
	for i, m, d in zip(ishape_exp, mshape_exp, range(max_ndim)):
		if not (i == m or i == 1 or m == 1):
			raise ValueError('Invalid shapes: {ishape}, {mshape}.'.format(
				ishape=ishape, mshape=mshape))

		oshape.append(max(i, m))

	return oshape

def _get_multiply_adjoint_sum_axes(oshape, ishape, mshape):
	ishape_exp, mshape_exp = su._expand_shapes(ishape, mshape)
	max_ndim = max(len(ishape), len(mshape))
	sum_axes = []
	for i, m, o, d in zip(ishape_exp, mshape_exp, oshape, range(max_ndim)):
		if (i == 1 and (m != 1 or o != 1)):
			sum_axes.append(d)

	return sum_axes

class Multiply(Linop):
	def __init__(self, ishape, mult, conj=False):
		self.mult = mult
		self.conj = conj
		if np.isscalar(mult):
			self.mshape = [1]
		else:
			self.mshape = mult.shape

		oshape = _get_multiply_oshape(ishape, self.mshape)
		super().__init__(oshape, ishape)


	def _apply(self, input):
		mult = self.mult
		if np.isscalar(mult):
			if mult == 1:
				return input

			if self.conj:
				mult = mult.conjugate()
		else:
			if self.conj:
				mult = cp.conj(mult)

		return input * mult

	def _adjoint_linop(self):
		sum_axes = _get_multiply_adjoint_sum_axes(
			self.oshape, self.ishape, self.mshape)

		M = Multiply(self.oshape, self.mult, conj=not self.conj)
		S = Sum(M.oshape, sum_axes)
		R = Reshape(self.ishape, S.oshape)
		return R * S * M

class Sum(Linop):
	def __init__(self, ishape, axes):
		self.axes = tuple(i % len(ishape) for i in axes)
		oshape = [ishape[i] for i in range(len(ishape)) if i not in self.axes]

		super().__init__(oshape, ishape)


	def _apply(self, input):
		return cp.sum(input, axis=self.axes)

	def _adjoint_linop(self):
		return Tile(self.ishape, self.axes)

class Tile(Linop):
	def __init__(self, oshape, axes):
		self.axes = tuple(a % len(oshape) for a in axes)
		ishape = [oshape[d] for d in range(len(oshape)) if d not in self.axes]
		self.expanded_ishape = []
		self.reps = []
		for d in range(len(oshape)):
			if d in self.axes:
				self.expanded_ishape.append(1)
				self.reps.append(oshape[d])
			else:
				self.expanded_ishape.append(oshape[d])
				self.reps.append(1)

		super().__init__(oshape, ishape)


	def _apply(self, input): 
			return cp.tile(input.reshape(self.expanded_ishape), self.reps)

	def _adjoint_linop(self):
		return Sum(self.oshape, self.axes)

class Reshape(Linop):
	def __init__(self, oshape, ishape):
		super().__init__(oshape, ishape)


	def _apply(self, input):
		return input.reshape(self.oshape)

	def _adjoint_linop(self):
		return Reshape(self.ishape, self.oshape)

	def _normal_linop(self):
		return Identity(self.ishape)

class Resize(Linop):
	def __init__(self, oshape, ishape, ishift=None, oshift=None):
		self.ishift = ishift
		self.oshift = oshift

		super().__init__(oshape, ishape)


	def _apply(self, input):
		return su._resize(input, self.oshape,
			ishift=self.ishift, oshift=self.oshift)

	def _adjoint_linop(self):
		return Resize(self.ishape, self.oshape,
					ishift=self.oshift, oshift=self.ishift)


