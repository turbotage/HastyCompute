import torch
import numpy as np
import math
from typing import Any, Self

"""
General Linops and Algorithms on General Linops
"""

class Vector:
	def __init__(self, input):
		self.children: list[Vector] = None
		self.tensor: torch.Tensor = None
		if isinstance(input, list):
			self.children = []
			for inp in input:
				self.children.append(inp if isinstance(inp, Vector) else Vector(inp))
		elif isinstance(input, torch.Tensor):
			self.tensor = input
		elif isinstance(input, float):
			self.tensor = torch.tensor(input, dtype=torch.float32)
		else:
			raise RuntimeError('Was not Vector, list or Tensor')
		
	def istensor(self):
		return self.tensor is not None

	def islist(self):
		return self.children is not None
	
	def clone(self):
		if self.istensor():
			return Vector(self.tensor.detach().clone())
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
			return ret
		else:
			return [self.tensor]

	def __getitem__(self, key):
		if self.istensor():
			return self.tensor[key]

		if isinstance(key, list):
			ret = []
			for ikey in key:
				ret.append(self.children[ikey])
			return Vector(ret)
		elif isinstance(key, int):
			return self.children[key]
		else:
			raise RuntimeError('Illegal typeof key, operator[] on Vector')

	@staticmethod
	def tovector(other):
		return other if isinstance(other, Vector) else Vector(other)

	@staticmethod
	def scale(input: Self, scale: torch.Tensor):
		if input.islist():
			ret = []
			for i in range(len(input.children)):
				ret.append(Vector.scale(input.children[i], scale))
			return Vector(ret)
		else:
			return Vector(input.tensor * scale)

	def __add__(self, other) -> Self:
		o = Vector.tovector(other)

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
	
	def __sub__(self, other) -> Self:
		o = Vector.tovector(other)

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

	def __mul__(self, other) -> Self:
		o = Vector.tovector(other)

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
			return Vector(ret)

	def __truediv__(self, other: Self) -> Self:
		o = Vector.tovector(other)

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

	def __pow__(self, other) -> Self:
		o = Vector.tovector(other)

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

	def __iadd__(self, other) -> Self:
		o = Vector.tovector(other)
		
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
	
	def __isub__(self, other) -> Self:
		o = Vector.tovector(other)

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
	
	def __imul__(self, other) -> Self:
		o = Vector.tovector(other)

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

	def __idiv__(self, other) -> Self:
		o = Vector.tovector(other)

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

	def __ipow__(self, other) -> Self:
		o = Vector.tovector(other)

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

	def __radd__(self, other) -> Self:
		return Vector.tovector(other) + self

	def __rsub__(self, other) -> Self:
		return Vector.tovector(other) - self

	def __rmul__(self, other) -> Self:
		return Vector.tovector(other) * self

	def __rtruediv__(self, other) -> Self:
		return Vector.tovector(other) / self

	def __rpow__(self, other) -> Self:
		return Vector.tovector(other) ** self

	def __neg__(self):
		if self.istensor():
			return self.tensor.neg()
		else:
			ret = []
			for i in range(len(self.children)):
				ret.append(self.children[i].__neg__())
		return Vector(ret)

	def __pos__(self):
		return self

	def numel(self):
		return numel(self)
	
	def min(self):
		return min(self)
	
	def max(self):
		return max(self)

	def item(self):
		return item(self)

	def mean(self):
		return mean(self)
	
	def mean_median(self):
		return mean_median(self)
	
	def sum(self):
		return sum(self)

	def abs(self):
		return abs(self)

def get_shape(inp: Vector | list | torch.Tensor | tuple):
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
	if isinstance(inp, tuple):
		return inp
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
			ret.append(ones(get_shape(elm), dtype))
		return Vector(ret)
	else: # shape is a tuple
		return Vector(torch.ones(shape, dtype=dtype))

def zeros(shape, dtype):
	if not is_shape(shape):
		raise RuntimeError('Invalid shape')

	if isinstance(shape, list):
		ret = []
		for elm in shape:
			ret.append(zeros(get_shape(elm), dtype))
		return Vector(ret)
	else: # shape is a tuple
		return Vector(torch.zeros(shape, dtype=dtype))

def empty(shape, dtype):
	if not is_shape(shape):
		raise RuntimeError('Invalid shape')

	if isinstance(shape, list):
		ret = []
		for elm in shape:
			ret.append(empty(get_shape(elm), dtype))
		return Vector(ret)
	else: # shape is a tuple
		return Vector(torch.empty(shape, dtype=dtype))

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
		return torch.norm(a.tensor)
	else:
		sum = torch.tensor(0.0)
		for i in range(len(a.children)):
			sum += norm(a.children[i])
		return sum

def min(a: Vector) -> torch.Tensor:
	if a.istensor():
		return torch.min(a.tensor)
	else:
		smallest = min(a.children[0])
		for i in range(1,len(a.children)):
			smallest = torch.minimum(smallest, min(a.children[i]))
		return smallest

def max(a: Vector) -> torch.Tensor:
	if a.istensor():
		return torch.max(a.tensor)
	else:
		smallest = max(a.children[0])
		for i in range(1,len(a.children)):
			smallest = torch.maximum(smallest, max(a.children[i]))
		return smallest

def numel(a: Vector) -> int:
	if a.istensor():
		return a.tensor.numel()
	else:
		numelem = 0
		for i in range(len(a.children)):
			numelem += numel(a.children[i])
		return numelem

def item(a: Vector):
	if a.istensor():
		return a.tensor.item()
	else:
		raise RuntimeError("Can't run item on Vector that isn't tensor")

def mean(a: Vector):
	if a.istensor():
		return torch.mean(a.tensor)
	else:
		val = 0
		for i in range(len(a.children)):
			val += mean(a.children[i])
		return val / len(a.children)

def mean_median(a: Vector):
	if a.istensor():
		return torch.median(a.tensor)
	else:
		val = 0
		for i in range(len(a.children)):
			val += mean_median(a.children[i])
		return val / len(a.children)

def sum(a: Vector):
	if a.istensor():
		return torch.sum(a.tensor)
	else:
		val = 0
		for i in range(len(a.children)):
			val += sum(a.children[i])
		return val

def abs(a: Vector):
	if a.istensor():
		return Vector(torch.abs(a.tensor))
	else:
		ret = []
		for i in range(len(a.children)):
			ret.append(abs(a.children[i]))
		return Vector(ret)



class Operator:
	def __init__(self, ishape, oshape):
		self.ishape = ishape
		self.oshape = oshape

	def reinitialize(self, ishape, oshape):
		self.ishape = ishape
		self.oshape = oshape

	def _apply(self, input: Vector) -> Vector:
		raise NotImplementedError
	
	def apply(self, input: Vector) -> Vector:
		if get_shape(input) != self.ishape:
			raise RuntimeError("Incompatible input shape for Operator")
		
		output = self._apply(input)

		if get_shape(output) != self.oshape:
			raise RuntimeError("Incompatible output shape for Operator")
		
		return output

	def __call__(self, input: Vector) -> Vector:
		return self.apply(input)
	
	def __mul__(self, other: Self):
		return CompositeOp(self, other)
	
	def __add__(self, other: Self):
		return AdditiveOp(self, other)

class IdentityOp(Operator):
	def __init__(self, ishape):
		super().__init__(ishape, ishape)

	def _apply(self, input: Vector) -> Vector:
		return input

class CompositeOp(Operator):
	def __init__(self, left: Operator, right: Operator):
		if left.ishape != right.oshape:
			raise RuntimeError("Incompatible left and right ishape/oshape")
		
		self.left = left
		self.right = right

		super().__init__(right.ishape, left.oshape)

	def _apply(self, input: Vector) -> Vector:
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

	def _apply(self, input: Vector) -> Vector:
		return self.left(input) + self.right(input)

class ScaleOp(Operator):
	def __init__(self, op: Operator, scale: torch.Tensor):
		self.op = op
		self.scale = scale
		super().__init__(op.ishape, op.oshape)

	def _apply(self, input: Vector) -> Vector:
		return Vector.scale(self.op(input), self.scale)

class MultiplyOp(Operator):
	def __init__(self, mult: Vector):
		self.mult = mult
		shape = get_shape(mult)
		super().__init__(shape, shape)

	def _apply(self, input: Vector) -> Vector:
		return self.mult * input

class TranslationOp(Operator):
	def __init__(self, op: Operator, offset: Vector, neg: bool):
		self.op = op
		self.offset = offset
		self.neg = neg
		super().__init__(op.ishape, op.oshape)

	def _apply(self, input: Vector) -> Vector:
		if self.neg:
			return self.op(input) - self.offset
		else:
			return self.op(input) + self.offset
			


class Linop(Operator):
	def __init__(self, ishape, oshape):
		super().__init__(ishape, oshape)

def l2_reg_linop(linop: Linop, lamda: torch.Tensor):
	return linop + ScaleOp(IdentityOp, lamda)

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
		self.x = y / Vector(self.max_eig)

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

