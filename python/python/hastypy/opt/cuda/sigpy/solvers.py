
import cupy as cp
import numpy as np
import math


class Alg(object):
	def __init__(self, max_iter):
		self.max_iter = max_iter
		self.iter = 0


	def _update(self):
		raise NotImplementedError

	def _done(self):
		return self.iter >= self.max_iter

	def update(self):
		"""Perform one update step.

		Call the user-defined _update() function and increment iter.
		"""
		self._update()
		self.iter += 1

	def done(self):
		"""Return whether the algorithm is done.

		Call the user-defined _done() function.
		"""
		return self._done()


class ConjugateGradient(Alg):
	def __init__(self, A, b, x, P=None, max_iter=100, tol=0):
		self.A = A
		self.b = b
		self.P = P
		self.x = x
		self.tol = tol
		self.r = b - self.A(self.x)

		if self.P is None:
			z = self.r
		else:
			z = self.P(self.r)

		if max_iter > 1:
			self.p = z.copy()
		else:
			self.p = z

		self.not_positive_definite = False
		self.rzold = cp.real(cp.vdot(self.r, z))
		self.resid = self.rzold.item()**0.5

		super().__init__(max_iter)


	def _update(self):
		Ap = self.A(self.p)
		pAp = cp.real(cp.vdot(self.p, Ap)).item()
		if pAp <= 0:
			self.not_positive_definite = True
			return

		self.alpha = self.rzold / pAp
		self.x += self.alpha * self.p
		if self.iter < self.max_iter - 1:
			self.r -= self.alpha * Ap
			if self.P is not None:
				z = self.P(self.r)
			else:
				z = self.r

			rznew = cp.real(cp.vdot(self.r, z))
			beta = rznew / self.rzold
			self.p *= beta
			self.p += z
			self.rzold = rznew

		self.resid = self.rzold.item()**0.5

	def _done(self):
		return (self.iter >= self.max_iter or
				self.not_positive_definite or self.resid <= self.tol)

