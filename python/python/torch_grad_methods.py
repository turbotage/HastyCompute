import torch

import math
import numpy as np

import plot_utility as pu

import torch_linop as tl
from torch_linop import TorchLinop, TorchMatrixLinop

from torch_basics import TorchIterativeAlg

class TorchCG(TorchIterativeAlg):
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

	def run(self, plot=False):
		i = 0
		while not self.done():
			print('CG Iter: ', i, '/', self.max_iter)
			self.update()

			if plot:
				pu.image_nd(self.x.numpy())

			i += 1
		return self.x
	
	def run_with_prox(self, prox, plot=False):
		i = 0
		while not self.done():
			print('CG Iter: ', i, '/', self.max_iter)
			self.update()

			self.x = prox(self.x)

			if plot:
				pu.image_nd(self.x.numpy())

			i += 1
		return self.x

class TorchGD(TorchIterativeAlg):
	def __init__(self, gradf, x, alpha=1.0, accelerate=False, max_iter=100, tol=0.0):
		self.gradf = gradf
		self.x = x
		self.alpha = alpha
		self.accelerate = accelerate
		self.tol = 0.0
		
		super().__init__(max_iter)

		self.resid = np.infty
		if self.accelerate:
			self.z = self.x.clone()
			self.t = 1

	def _update(self):
		x_old = self.x.clone()

		if self.accelerate:
			self.x = self.z

		self.x -= self.alpha * self.gradf(self.x)

		if self.accelerate:
			t_old = self.t
			self.t = (1.0 + math.sqrt(1.0 + 4.0 * t_old*t_old)) / 2.0

			self.z = self.x + ((t_old - 1) / self.t) * (self.x - x_old)

		if self.tol != 0.0:
			self.resid = torch.norm(self.x - x_old).item() / self.alpha

	def _done(self):
		return self.iter >= self.max_iter or self.resid < self.tol
	
	def run(self, plot=False):
		i = 0
		while not self.done():
			print('GD Iter: ', i, '/', self.max_iter)
			self.update()

			if plot:
				pu.image_nd(self.x.numpy())

			i += 1
		return self.x
	
	def run_with_prox(self, prox, plot=False):
		i = 0
		while not self.done():
			print('GD Iter: ', i, '/', self.max_iter)
			self.update()

			self.x = prox(self.x)

			if plot:
				pu.image_nd(self.x.numpy())

			i += 1
		return self.x






def test_cg():

	A = torch.rand(10,5)
	Ah = A.transpose(0, 1).conj()

	AHA = Ah @ A

	x = torch.rand(5,1)

	b = AHA @ x

	mlinop = TorchMatrixLinop(AHA)

	x0 = torch.ones(5,1)

	tcg = TorchCG(mlinop, b, x0)

	while not tcg.done():
		tcg.update()

	result = tcg.x

	print(x)
	print(result)

