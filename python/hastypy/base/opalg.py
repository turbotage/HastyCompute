import torch
import numpy as np
import math
from typing import Any, Self

"""
General Linops and Algorithms on General Linops
"""

class Vector:
	def __init__(self, input):
		self.children: list[torch.Tensor] = None
		self.tensor: torch.Tensor = None
		if isinstance(input, list):
			self.children = []
			for inp in input:
				self.children.append(inp if isinstance(inp, Vector) else Vector(inp))
		elif isinstance(input, torch.Tensor):
			self.tensor = input
		else:
			raise RuntimeError('Was not Vector, list or Tensor')
		
	def istensor(self):
		return self.tensor is not None

	def islist(self):
		return self.children is not None
	
	def clone(self):
		if self.istensor():
			return Vector(self.tensor.clone())
		else:
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i].clone())
			return Vector(ret)

	def __str__(self):
		if self.istensor():
			return str(self.tensor)
		else:
			ret = "["
			for i in range(len(self.children)):
				ret += self.children[i].__str__() + ","
			ret += "]"
			return ret
	
	def get_tensor(self):
		if self.istensor():
			return self.tensor
		else:
			raise RuntimeError("Can't get_tensor() when self is list")

	def get_tensorlist(self) -> list[torch.Tensor]:
		ret: list[torch.Tensor] = []
		if self.islist():
			for i in range(len(self.children)):
				ret.extend(self.children[i].get_tensorlist())
		else:
			return [self.tensor]

	@staticmethod
	def scale(input: Self, scale: torch.Tensor):
		if input.islist():
			ret = []
			for i in range(len(input.children)):
				ret.append(Vector.scale(input.children[i], scale))
			return Vector(ret)
		else:
			return Vector(input.tensor * scale)

	def __add__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")

		if self.istensor() and o.istensor():
			return Vector(self.tensor + o.tensor)
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self + o.children[i])
			return Vector(ret)
		elif self.islist() and o.istensor():
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] + o)
			return Vector(ret)
		else:
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] + o.children[i])
			return Vector(ret)
	
	def __sub__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")

		if self.istensor() and o.istensor():
			return Vector(self.tensor - o.tensor)
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self - o.children[i])
			return Vector(ret)
		elif self.islist() and o.istensor():
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] - o)
			return Vector(ret)
		else:
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] - o.children[i])
			return Vector(ret)

	def __mul__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")

		if self.istensor() and o.istensor():
			return Vector(self.tensor * o.tensor)
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self * o.children[i])
			return Vector(ret)
		elif self.islist() and o.istensor():
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] * o)
			return Vector(ret)
		else:
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] * o.children[i])
			return Vector(self.ret)

	def __div__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")

		if self.istensor() and o.istensor():
			return Vector(self.tensor / o.tensor)
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self / o.children[i])
			return Vector(ret)
		elif self.islist() and o.istensor():
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] / o)
			return Vector(ret)
		else:
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] / o.children[i])
			return Vector(ret)

	def __pow__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")

		if self.istensor() and o.istensor():
			return Vector(self.tensor ** o.tensor)
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self ** o.children[i])
			return Vector(ret)
		elif self.islist() and o.istensor():
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] ** o)
			return Vector(ret)
		else:
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i] ** o.children[i])
			return Vector(ret)

	def __iadd__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")
		if self.istensor() and o.istensor():
			self.tensor += o.tensor
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self + o.children[i])
			self.tensor = None
			self.children = ret
		elif self.islist() and o.istensor():
			for i in range(len(self.children)):
				self.children[i] += o
		else:
			for i in range(len(self.children)):
				self.children[i] += o.children[i]
		return self
	
	def __isub__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")
		if self.istensor() and o.istensor():
			self.tensor -= o.tensor
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self - o.children[i])
			self.tensor = None
			self.children = ret
		elif self.islist() and o.istensor():
			for i in range(len(self.children)):
				self.children[i] -= o
		else:
			for i in range(len(self.children)):
				self.children[i] -= o.children[i]
		return self
	
	def __imul__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")
		if self.istensor() and o.istensor():
			self.tensor *= o.tensor
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self * o.children[i])
			self.tensor = None
			self.children = ret
		elif self.islist() and o.istensor():
			for i in range(len(self.children)):
				self.children[i] *= o
		else:
			for i in range(len(self.children)):
				self.children[i] *= o.children[i]
		return self

	def __idiv__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")
		if self.istensor() and o.istensor():
			self.tensor /= o.tensor
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self / o.children[i])
			self.tensor = None
			self.children = ret
		elif self.islist() and o.istensor():
			for i in range(len(self.children)):
				self.children[i] /= o
		else:
			for i in range(len(self.children)):
				self.children[i] /= o.children[i]
		return self

	def __ipow__(self, o: Self) -> Self:
		if not isinstance(o, Vector):
			raise RuntimeError("Input was not a Vector")
		if self.istensor() and o.istensor():
			self.tensor **= o.tensor
		elif self.istensor() and o.islist():
			ret = []
			for i in range(len(o.children)):
				ret.append(self ** o.children[i])
			self.tensor = None
			self.children = ret
		elif self.islist() and o.istensor():
			for i in range(len(self.children)):
				self.children[i] **= o
		else:
			for i in range(len(self.children)):
				self.children[i] **= o.children[i]
		return self

	def __neg__(self):
		if self.istensor():
			self.tensor.neg_()
		else:
			for i in range(len(self.children)):
				self.children[i].__neg__()
		return self

	def __pos__(self):
		return self


def get_shape(inp: Vector | list | torch.Tensor):
	if isinstance(inp, Vector):
		if inp.islist():
			return get_shape(inp.children)
		else:
			return inp.tensor.shape
	if isinstance(inp, torch.Tensor) or isinstance(inp, np.ndarray):
		return inp.shape
	if isinstance(inp, list):
		shape_list = []
		for p in inp:
			shape_list.append(get_shape(p))
		return shape_list
	raise RuntimeError('Invalid type for get_shape')

def is_shape(inp):
	if isinstance(inp, list):
		if len(inp) == 0:
			return False
		isshape = True
		for p in inp:
			isshape = isshape and is_shape(p)
		return isshape
	elif isinstance(inp, tuple):
		if len(inp) > 0:
			for p in inp:
				if not isinstance(p, int):
					return False
		return True
	return False

def rand(shape, dtype):
	if not is_shape(shape):
		raise RuntimeError('Invalid shape')

	if isinstance(shape, list):
		ret = []
		for elm in shape:
			ret.append(rand(get_shape(elm), dtype))
		return Vector(ret)
	else: # shape is a tuple
		return Vector(torch.rand(shape, dtype=dtype))

def ones(shape, dtype):
	if not is_shape(shape):
		raise RuntimeError('Invalid shape')

	if isinstance(shape, list):
		ret = []
		for elm in shape:
			ret.append(rand(get_shape(elm), dtype))
		return Vector(ret)
	else: # shape is a tuple
		return Vector(torch.ones(shape, dtype=dtype))

def zeros(shape, dtype):
	if not is_shape(shape):
		raise RuntimeError('Invalid shape')

	if isinstance(shape, list):
		ret = []
		for elm in shape:
			ret.append(rand(get_shape(elm), dtype))
		return Vector(ret)
	else: # shape is a tuple
		return Vector(torch.zeros(shape, dtype=dtype))

def vdot(a: Vector, b: Vector) -> torch.Tensor:
	if a.istensor():
		return torch.vdot(a.tensor.flatten(), b.tensor.flatten())
	else:
		sum = torch.tensor(0.0)
		for i in range(len(a.children)):
			sum += vdot(a.children[i], b.children[i])
		return sum

def norm(a: Vector) -> torch.Tensor:
	if a.istensor():
		return torch.norm(a)
	else:
		sum = torch.tensor(0.0)
		for i in range(len(a.children)):
			sum += norm(a.children[i])
		return sum


class Operator:
	def __init__(self, ishape, oshape):
		self.ishape = ishape
		self.oshape = oshape

	def reinitialize(self, ishape, oshape):
		self.ishape = ishape
		self.oshape = oshape

	def _apply(self, input: Vector):
		raise NotImplementedError
	
	def apply(self, input: Vector):
		if get_shape(input) != self.ishape:
			raise RuntimeError("Incompatible input shape for Operator")
		
		output = self._apply(input)

		if get_shape(output) != self.oshape:
			raise RuntimeError("Incompatible output shape for Operator")
		
		return output

	def __call__(self, input: Vector):
		return self.apply(input)
	
	def __mul__(self, other: Self):
		return CompositeOp(self, other)
	
	def __add__(self, other: Self):
		return AdditiveOp(self, other)

class IdentityOp(Operator):
	def __init__(self, ishape):
		super().__init__(ishape, ishape)

	def _apply(self, input: Vector):
		return input

class CompositeOp(Operator):
	def __init__(self, left: Operator, right: Operator):
		if left.ishape != right.oshape:
			raise RuntimeError("Incompatible left and right ishape/oshape")
		
		self.left = left
		self.right = right

		super().__init__(right.ishape, left.oshape)

	def _apply(self, input: Vector):
		return self.left(self.right(input))

class AdditiveOp(Operator):
	def __init__(self, left: Operator, right: Operator):
		if left.ishape != right.ishape:
			raise RuntimeError("Incompatible left and right ishape/oshape")
		if left.oshape != right.oshape:
			raise RuntimeError("Incompatible left and right ishape/oshape")

		self.left = left
		self.right = right

		super().__init__(right.ishape, left.oshape)

	def _apply(self, input: Vector):
		return self.left(input) + self.right(input)

class ScaleOp(Operator):
	def __init__(self, op: Operator, scale: torch.Tensor):
		self.op = op
		self.scale = scale
		super().__init__(op.ishape, op.oshape)

	def _apply(self, input: Vector):
		return Vector.scale(self.op(input), self.scale)


class Linop(Operator):
	def __init__(self, ishape, oshape):
		super().__init__(ishape, oshape)


class IterativeAlg:
	def __init__(self, max_iter):
		self.max_iter = max_iter
		self.iter = 0
				
	def _update(self):
		raise NotImplementedError

	def _done(self):
		return self.iter >= self.max_iter

	def update(self):
		self._update()
		self.iter += 1

	def done(self):
		return self._done()
	
class MaxEig(IterativeAlg):
	def __init__(self, A: Linop, dtype=torch.float32, max_iter=30):
		self.A = A
		self.x = rand(A.ishape, dtype=dtype)
		self.max_eig = Vector(torch.inf)
		super().__init__(max_iter)
		
	def _update(self):
		y = self.A(self.x)
		self.max_eig = norm(y)
		self.x = y / self.max_eig

	def _done(self):
		return self.iter >= self.max_iter
	
	def run(self, callback=None, print_info=False):
		i = 0
		while not self.done():
			self.update()
			
			if print_info:
				print('MaxEig Iter: ', i,'/', self.max_iter, ' MaxEig: ', self.max_eig)

			if callback is not None:
				callback(self.x, i)
				
			i += 1
		return self.max_eig

class ConjugateGradient(IterativeAlg):
	def __init__(self, A: Linop, b: Vector, x: torch.Tensor, P: Linop | None = None, tol = 0.0, max_iter=100):
		self.A = A
		self.b = b
		self.x = x
		self.P = P
		self.tol = tol

		self.r = self.b - self.A(self.x)

		# Preconditioning
		if self.P is None:
			z = self.r
		else:
			z = self.P(self.r)

		if max_iter > 1:
			self.p = z.clone()
		else:
			self.p = z

		self.positive_definite = True
		self.rzold = torch.real(vdot(self.r, z))
		self.resid = torch.sqrt(self.rzold)

		super().__init__(max_iter)
	
	def _update(self):
		Ap = self.A(self.p)
		pAp = torch.real(vdot(self.p, Ap))
		if pAp <= 0:
			self.positive_definite = False
			return
		
		alpha = Vector(self.rzold / pAp)

		self.x += self.p * alpha
		if self.iter < self.max_iter - 1:
			self.r -= Ap * alpha
			#Preconditioning
			if self.P is not None:
				z = self.P(self.r)
			else:
				z = self.r
			rznew = torch.real(vdot(self.r, z))
			beta = Vector(rznew / self.rzold)

			self.p *= beta
			self.p += z

			self.rzold = rznew

		self.resid = torch.sqrt(self.rzold)

	def _done(self):
		return (
			self.iter >= self.max_iter or
			not self.positive_definite or
			self.resid <= self.tol
		)

	def run(self, callback=None, print_info=False):
		i = 0
		while not self.done():
			self.update()

			if print_info:
				print('CG Iter: ', i, '/', self.max_iter, ', Res: ', self.resids[-1])

			if callback is not None:
				callback(self.x, i)
			i += 1
		return self.x

class GradientDescent(IterativeAlg):
	def __init__(self, gradf: Operator, x: Vector, prox: Operator | None = None, alpha=1.0, accelerate=False, max_iter=100, tol=0.0):
		self.gradf = gradf
		self.x = x
		self.alpha = alpha
		self.accelerate = accelerate
		self.tol = tol
		self.prox = prox
		self.resids = [-1]
		self.rel_resids = [-1]
		if self.accelerate:
			self.z = self.x.clone()
			self.t = 1
		
		super().__init__(max_iter)

	def _update(self):
		x_old = self.x.detach().clone()

		if self.accelerate:
			self.x = self.z

		self.x -= self.alpha * self.gradf(self.x)

		if self.prox is not None:
			self.x = self.prox(self.alpha, self.x)

		if self.accelerate:
			t_old = self.t
			self.t = (1.0 + math.sqrt(1.0 + 4.0 * t_old*t_old)) / 2.0

			self.z = self.x - x_old
			self.resids.append(norm(self.z))
			self.z = self.x + ((t_old - 1) / self.t) * self.z
		else:
			self.resids.append(norm(self.x - x_old))

		if self.tol != 0.0:
			self.rel_resids.append(self.resids[-1] / norm(self.x))

	def _done(self):
		return self.iter >= self.max_iter or (self.rel_resids[-1] < self.tol and self.rel_resids[-1] > 0.0)
	
	def run(self, callback=None, print_info=False):
		i = 0
		while not self.done() or i == 0:
			self.update()
			
			if print_info:
				if self.accelerate:
					print('GD Iter: ', i, '/', self.max_iter, ', Res: ', self.resids[-1], ', t:  ', self.t, ',  ', end="")
				else:
					print('GD Iter: ', i, '/', self.max_iter, ', Res: ', self.resids[-1], ',  ', end="")

				if self.tol != 0.0:
					print(' RelRes: ', self.rel_resids[-1], ', ', end="")
				print("")

			if callback is not None:
				callback(self.x, i)
			
			i += 1

		return self.x


