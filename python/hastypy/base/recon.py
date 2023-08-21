import torch
import numpy as np

from hastypy.base.sense import BatchedSenseNormal, OuterInnerAutomorphism, InnerOuterAutomorphism
from hastypy.base.svt import Random3DBlocksSVT, Normal3DBlocksSVT

from hastypy.base.opalg import Vector, MaxEig
from hastypy.base.solvers import GradientDescent
from hastypy.base.proximal import UnitaryTransform, ProximalL1DCT, Proximal3DSVT, SVTOptions



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
			self._sensenormalop = BatchedSenseNormal(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.streams)
			

		self.lamda = lamda / self.stepsize
		if self.lamda != 0.0:
			self._dctprox = ProximalL1DCT((5,1) + self.smaps.shape[1:], (True,True,True), self.lamda)
		else:
			self._dctprox = None


	def max_eig_run(self, print_info=True):
		maxeigop = BatchedSenseNormal([self.coord_vec[0]], self.smaps)
		self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run(print_info=print_info).to(torch.float32)

	def run(self, image, iter=50, callback=None, accelerate=True):

		image = Vector(image)

		if self.solver == 'GD':
			gd = GradientDescent(self._sensenormalop, image, self._dctprox, 1 / self.max_eig, accelerate=accelerate, max_iter=iter)
			image = gd.run(callback, print_info=True)

		return image.get_tensor()



class FivePointLLR:
	def __init__(self, smaps, coord_vec, kdata_vec, svt_options: SVTOptions | None, 
			weights_vec=None, grad_streams=None, prox_streams=None, solver='GD'):
		
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.svt_options = svt_options
		self.nframes = len(coord_vec) // 5

		self.grad_streams = grad_streams
		self.prox_streams = prox_streams

		self.solver = solver

		self.max_eig_run()
		self.stepsize = 1 / self.max_eig

		# Setup ||Ax-b|| operators
		if self.solver == 'GD':
			self._sensenormalop = BatchedSenseNormal(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.grad_streams)
		
		self.svtshape = (self.nframes, 5) + self.smaps.shape[1:]
		self.gradshape = (self.nframes*5,) + self.smaps.shape[1:]

		self.tosvtshape = InnerOuterAutomorphism(self.nframes, 5, self.smaps.shape[1:])
		self.togradshape = OuterInnerAutomorphism(self.nframes, 5, self.smaps.shape[1:])

		# Setup LLR proximality operators
		if self.svt_options is not None:
			# Rescale threshold after default stepsize
			self.svt_options.thresh /= self.stepsize

			self.svtprox = UnitaryTransform(
				Proximal3DSVT(self.svtshape, self.svt_options, self.prox_streams),
				self.tosvtshape,
				self.togradshape)
		else:
			self.svtprox = None
			
		


	def max_eig_run(self):
		maxeigop = BatchedSenseNormal([self.coord_vec[0]], self.smaps)
		self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run().to(torch.float32)

	def run(self, image, iter=50, callback=None, accelerate=True):

		image = self.togradshape(Vector(image))

		if self.solver == 'GD':
			gd = GradientDescent(self._sensenormalop, image, self.svtprox, self.stepsize, accelerate=accelerate, max_iter=iter)
			image = gd.run(callback, print_info=True)

		return self.tosvtshape(image).get_tensor()



