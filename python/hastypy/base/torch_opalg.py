import torch
import numpy as np
import math
from typing import Any, Self

"""
Torch Linops
"""

class TorchLinop:
	def __init__(self, ishape, oshape):
		self.ishape = ishape
		self.oshape = oshape

	def _apply(self, input: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError
	
	def apply(self, input: torch.Tensor) -> torch.Tensor:
		if input.shape != self.ishape:
			raise RuntimeError("Incompatible input shape with Linop")
		
		output = self._apply(input)

		if output.shape != self.oshape:
			raise RuntimeError("Incompatible output shape with Linop")
		
		return output

	def __call__(self, input: torch.Tensor) -> torch.Tensor:
		return self.apply(input)

class TorchMatrixLinop(TorchLinop):
	def __init__(self, matrix: torch.Tensor):
		self.matrix = matrix
		super().__init__((matrix.shape[1],1), (matrix.shape[0],1))

	def _apply(self, input: torch.Tensor):
		return self.matrix @ input
	
class TorchScaleLinop(TorchLinop):
	def __init__(self, linop: TorchLinop, scaling: torch.Tensor):
		self.scaling = scaling
		self.linop = linop
		super().__init__(linop.ishape, linop.oshape)

	def _apply(self, input):
		return self.linop(input * self.scaling)
	
class TorchL2Reg(TorchLinop):
	def __init__(self, linop: TorchLinop, lamda = 0.01):
		self.lamda = lamda
		self.linop = linop
		if linop.ishape is not linop.oshape:
			raise RuntimeError("L2Reg linop must have equal input and output shapes")
		super().__init__(linop.ishape, linop.oshape)

	def _apply(self, input: torch.Tensor):
		if self.lamda == 0.0:
			return self.linop(input)
		else:
			ret = self.lamda * input
			return self.linop(input) + ret
		

"""
Torch Algorithms on TorchLinops
"""

class TorchIterativeAlg:
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
	
# This is not perfect, should use Rayleigh instead but this way uses less memory
class TorchMaxEig(TorchIterativeAlg):
	def __init__(self, A: TorchLinop, dtype=torch.float32, max_iter=30):
		self.A = A
		self.x = torch.rand(A.ishape, dtype=dtype)
		self.max_eig = torch.inf
		super().__init__(max_iter)
		
	def _update(self):
		y = self.A(self.x)
		self.max_eig = torch.linalg.norm(y)
		self.x = y / self.max_eig

	def _done(self):
		return self.iter >= self.max_iter
	
	def run(self):
		i = 0
		while not self.done():
			print('MaxEig Iter: ', i,'/', self.max_iter, ' MaxEig: ', self.max_eig)
			self.update()
			i += 1
		return self.max_eig

class TorchConjugateGradient(TorchIterativeAlg):
	def __init__(self, A: TorchLinop, b: torch.Tensor, x: torch.Tensor, P: TorchLinop | None = None, tol = 0.0, max_iter=100):
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
		self.rzold = torch.real(torch.vdot(self.r.flatten(), z.flatten()))
		self.resid = torch.sqrt(self.rzold)

		super().__init__(max_iter)
	
	def _update(self):
		Ap = self.A(self.p)
		pAp = torch.real(torch.vdot(self.p.flatten(), Ap.flatten()))
		if pAp <= 0:
			self.positive_definite = False
			return
		
		self.alpha = self.rzold / pAp

		self.x.add_(self.p, alpha=self.alpha)
		if self.iter < self.max_iter - 1:
			self.r.add_(Ap, alpha=self.alpha.neg())
			#Preconditioning
			if self.P is not None:
				z = self.P(self.r)
			else:
				z = self.r
			rznew = torch.real(torch.vdot(self.r.flatten(), z.flatten()))
			beta = rznew / self.rzold

			self.p.mul_(beta)
			self.p.add_(z)

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

class TorchGradientDescent(TorchIterativeAlg):
	def __init__(self, gradf, x, prox=None, alpha=1.0, accelerate=False, max_iter=100, tol=0.0):
		self.gradf = gradf
		self.x = x
		self.alpha = alpha
		self.accelerate = accelerate
		self.tol = tol
		self.prox = prox
		self.resids = [-1]
		self.rel_resids = [-1]
		if self.accelerate:
			self.z = self.x.detach().clone()
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
			self.resids.append(torch.norm(self.z))
			self.z = self.x + ((t_old - 1) / self.t) * self.z
		else:
			self.resids.append(torch.norm(self.x - x_old))

		if self.tol != 0.0:
			self.rel_resids.append(self.resids[-1] / torch.norm(self.x))

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

