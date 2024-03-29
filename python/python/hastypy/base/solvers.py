import torch
import numpy as np
import math
from typing import Any, Self
import time

import hastypy.base.opalg as opalg
from hastypy.base.opalg import IterativeAlg, Vector, Linop, Operator
from hastypy.base.proximal import ProximalOperator

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
		self.rzold = torch.real(opalg.vdot(self.r, z))
		self.resid = torch.sqrt(self.rzold)

		super().__init__(max_iter)
	
	def _update(self):
		Ap = self.A(self.p)
		pAp = torch.real(opalg.vdot(self.p, Ap))
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
			rznew = torch.real(opalg.vdot(self.r, z))
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
	def __init__(self, gradf: Operator, x: Vector, prox: ProximalOperator | None = None, alpha=1.0, accelerate=False, max_iter=100, tol=0.0):
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
		x_old = self.x.clone()

		if self.accelerate:
			self.x = self.z

		time_start = time.time()
		self.x -= self.alpha * self.gradf(self.x)
		time_stop = time.time()
		self.grad_time = time_stop - time_start

		if self.prox is not None:
			time_start = time.time()
			self.x = self.prox(self.x, self.alpha)
			time_stop = time.time()
			self.prox_time = time_stop - time_start

		if self.accelerate:
			t_old = self.t
			self.t = (1.0 + math.sqrt(1.0 + 4.0 * t_old*t_old)) / 2.0

			self.z = self.x - x_old
			self.resids.append(opalg.norm(self.z))
			self.z = self.x + ((t_old - 1) / self.t) * self.z
		else:
			self.resids.append(opalg.norm(self.x - x_old))

		if self.tol != 0.0:
			self.rel_resids.append(self.resids[-1] / opalg.norm(self.x))

	def _done(self):
		return self.iter >= self.max_iter or (self.rel_resids[-1] < self.tol and self.rel_resids[-1] > 0.0)
	
	def run(self, callback=None, print_info=False):
		i = 0
		while not self.done() or i == 0:
			self.update()
			
			if print_info:
				printr: str = ""
				if self.accelerate:
					printr += f"GD Iter: {i} / {self.max_iter}, , Res: {self.resids[-1]}, t: {self.t},  "
					#print('GD Iter: ', i, '/', self.max_iter, ', Res: ', self.resids[-1], ', t:  ', self.t, ',  ', end="")
				else:
					printr += f"GD Iter: {i} / {self.max_iter}, , Res: {self.resids[-1]},  "

				if self.tol != 0.0:
					printr += f"RelRes: {self.rel_resids[-1]},  "

				printr += f"GradTime:  {self.grad_time},  "

				if self.prox is not None:
					printr += f"ProxTime:  {self.prox_time},  "

				print(printr)

			if callback is not None:
				callback(self.x, i)
			
			i += 1

		return self.x

class PrimalDualHybridGradient(IterativeAlg):
	def __init__(self, proxfc: ProximalOperator, proxg: ProximalOperator, K: Linop, KH: Linop, x: Vector, y: Vector, 
	      tau: Vector, sigma: Vector, acceleration: float, max_iter=100, tol=0.0):
		self.proxfc = proxfc
		self.proxg = proxg
		self.K = K
		self.KH = KH
		self.x = x
		self.y = y

		self.acceleration = acceleration

		self.tau = tau
		self.sigma = sigma

		self.tau_mul = opalg.min(tau).item()

		self.xtemp = self.x.clone()

		self.tol = tol
		self.resids = [-1]
		self.rel_resids = [-1]

		super().__init__(max_iter)

	def _update(self):

		# Dual Update
		self.y += self.sigma * self.K(self.xtemp) # xtemp stored extrapolated x
		self.y = self.proxfc(self.y, self.sigma)

		# Primal Update
		self.xold = self.x.clone() # Stored x_old
		self.x -= self.tau * self.KH(self.y)
		self.x = self.proxg(self.x, self.tau)

		# Acceleration
		self.theta = 1.0 / math.sqrt(1.0 + 2.0 * self.acceleration * self.tau_mul)
		if not math.isclose(self.theta, 1.0):
			self.tau_mul *= self.theta
			self.tau *= self.theta
			self.sigma /= self.theta

		# xtemp holds difference
		self.xtemp = self.x - self.xtemp
		self.resids.append(opalg.norm(self.xtemp).item())
		# xtemp holds extrapolated x
		self.xtemp = self.x + self.theta*self.xtemp

		if self.tol != 0.0:
			self.rel_resids.append(self.resids[-1] / opalg.norm(self.x))

	def _done(self):
		return self.iter >= self.max_iter or (self.rel_resids[-1] < self.tol and self.rel_resids[-1] > 0.0)

	def run(self, callback=None, print_info=False):
		i = 0
		while not self.done() or i == 0:

			time_start = time.time()
			self.update()
			time_stop = time.time()
			
			if print_info:
				printr: str = ""
				printr += f"PDHG Iter: {i} / {self.max_iter}, , Res: {self.resids[-1]},  "

				if self.tol != 0.0:
					printr += f"RelRes: {self.rel_resids[-1]},  "

				if self.tau.numel() == 1:
					printr += f"Tau: {self.tau.item()},  "

				if self.sigma.numel() == 1:
					printr += f"Sigma: {self.sigma.item()},  "

				printr += f"Theta: {self.theta},  "

				printr += f"UpdateTime:  {time_stop - time_start},  "

				print(printr)

			if callback is not None:
				callback(self.x, i)
			
			i += 1

		return self.x

			

