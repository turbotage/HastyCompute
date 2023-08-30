import torch
import numpy as np

from hastypy.base.sense import BatchedSense, BatchedSenseAdjoint, BatchedSenseNormal, BatchedSenseNormalAdjoint, OuterInnerAutomorphism, InnerOuterAutomorphism
from hastypy.base.svt import Random3DBlocksSVT, Normal3DBlocksSVT

import hastypy.base.opalg as opalg
from hastypy.base.opalg import Vector, MaxEig, MultiplyOp
from hastypy.base.solvers import GradientDescent, PrimalDualHybridGradient
import hastypy.base.proximal as prox



class FivePointFULL:
	def __init__(self, smaps, coord_vec, kdata_vec, weights_vec=None, streams=None, solver='GD', lamda=0.0):
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.streams = streams
		self.solver = solver

		self.max_eig_run()
		self.stepsize = 1 / self.max_eig
		
		if self.solver == 'GD':
			self.sensenormalop = BatchedSenseNormal(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.streams)
			

		self.lamda = lamda / self.stepsize
		if self.lamda != 0.0:
			self._dctprox = prox.ProximalL1DCT((5,1) + self.smaps.shape[1:], (True,True,True), self.lamda)
		else:
			self._dctprox = None


	def max_eig_run(self, print_info=True):
		maxeigop = BatchedSenseNormal([self.coord_vec[0]], self.smaps)
		self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run(print_info=print_info).to(torch.float32)

	def run(self, image, iter=50, callback=None, accelerate=True):

		image = Vector(image)

		if self.solver == 'GD':
			gd = GradientDescent(self.sensenormalop, image, self._dctprox, 1 / self.max_eig, accelerate=accelerate, max_iter=iter)
			image = gd.run(callback, print_info=True)

		return image.get_tensor()



class FivePointLLR:
	def __init__(self, smaps, coord_vec, kdata_vec, svt_options: prox.SVTOptions | None, 
			weights_vec=None, grad_streams=None, prox_streams=None, solver='GD',
			tau=None, sigma=None):
		
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.svt_options = svt_options
		self.nframes = len(coord_vec) // 5

		self.grad_streams = grad_streams
		self.prox_streams = prox_streams

		self.solver = solver


		def init_gd():
			maxeigop = BatchedSenseNormal([self.coord_vec[0]], self.smaps)
			self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run(print_info=True).to(torch.float32)
			self.stepsize = 1 / self.max_eig
			self.sensenormalop = BatchedSenseNormal(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.grad_streams)
			# Rescale threshold after default stepsize
			self.svt_options.thresh /= self.stepsize

		def init_pdhg():
			self.gamma_primal = 0.0
			self.gamma_dual = 1.0

			datavec = Vector(self.kdata_vec)
			self.proxfc = prox.L2Reg(opalg.get_shape(datavec), a=-datavec, inplace=True)

			self.tau = tau
			self.sigma = sigma

			if self.tau is None:
				if self.sigma is None:
					sigmatemp = [torch.ones_like(self.kdata_vec[0])]
					self.sigma = Vector(1.0)
				else:
					sigmatemp = [sigma.get_tensorlist()[0]]

				A = BatchedSense([self.coord_vec[0]], self.smaps)
				AH = BatchedSenseAdjoint([self.coord_vec[0]], self.smaps)
				S = MultiplyOp(sigmatemp)

				max_eig = MaxEig(AH * S * A, torch.complex64, max_iter=5).run(print_info=True)
				
				self.tau = Vector(0.5 / max_eig)

			elif self.sigma is None:
				AAH = BatchedSenseNormalAdjoint([self.coord_vec[0]], self.smaps, [self.kdata_vec[0]], [self.tau[0]])
				max_eig = MaxEig(AAH, torch.complex64, max_iter=5).run(print_info=True)

				self.sigma = Vector(0.5 / max_eig)

			self.bsense = BatchedSense(self.coord_vec, self.smaps, None)
			self.bsenseadj = BatchedSenseAdjoint(self.coord_vec, self.smaps, None)


		# Setup ||Ax-b|| operators
		if self.solver == 'GD':
			init_gd()

		elif self.solver == 'PDHG':
			init_pdhg()

		
		self.svtshape = (self.nframes,5) + self.smaps.shape[1:]
		self.gradshape = (self.nframes*5,) + self.smaps.shape[1:]

		self.tosvtshape = InnerOuterAutomorphism(self.nframes, 5, self.smaps.shape[1:])
		self.togradshape = OuterInnerAutomorphism(self.nframes, 5, self.smaps.shape[1:])

		# Setup LLR proximality operators
		if self.svt_options is not None:
			self.svtprox = prox.UnitaryTransform(
				prox.Proximal3DSVT(self.svtshape, self.svt_options, self.prox_streams),
				self.tosvtshape,
				self.togradshape)
		else:
			self.svtprox = prox.ZeroFunc(self.gradshape)
			

	def run(self, image, iter=50, callback=None, accelerate=True):

		image = self.togradshape(Vector(image))

		if self.solver == 'GD':
			gd = GradientDescent(self.sensenormalop, image, self.svtprox, self.stepsize, accelerate=accelerate, max_iter=iter)
			image = gd.run(callback, print_info=True)
		elif self.solver == 'PDHG':
			dual = opalg.zeros(opalg.get_shape(Vector(self.kdata_vec)), torch.complex64)

			self.svtprox = prox.ZeroFunc(opalg.get_shape(image))
			pdhg = PrimalDualHybridGradient(self.proxfc, self.svtprox, self.bsense, self.bsenseadj, image, dual, 
				self.tau, self.sigma, gamma_primal=self.gamma_primal, gamma_dual=self.gamma_dual, max_iter=iter)
			
			image = pdhg.run(callback, print_info=True)

		return self.tosvtshape(image).get_tensor()



