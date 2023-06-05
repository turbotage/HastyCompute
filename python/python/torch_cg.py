import torch

import torch_linop as tl
from torch_linop import TorchLinop, TorchMatrixLinop

from torch_iterative_alg import TorchIterativeAlg

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

	def run(self):
		i = 0
		while not self.done():
			print('CG Iter: ', i, '/', self.max_iter)
			self.update()
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

