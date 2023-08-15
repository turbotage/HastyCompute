import torch
import numpy as np

from hastypy.base.sense import BatchedSenseNormalLinop, OuterInnerAutomorphism, InnerOuterAutomorphism
from hastypy.base.svt import Random3DBlocksSVT, Normal3DBlocksSVT

from hastypy.base.opalg import Vector, MaxEig
from hastypy.base.solvers import GradientDescent
from hastypy.base.proximal import ProximalL1DCT



class FivePointFULL:
	def __init__(self, smaps, coord_vec, kdata_vec, weights_vec=None, streams=None, solver='GD', lamda=0.0):
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.streams = streams
		self.solver = solver

		self.lamda = lamda

		if self.solver == 'GD':
			self._sensenormalop = BatchedSenseNormalLinop(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.streams)
			
		if self.lamda != 0.0:
			self._dctprox = ProximalL1DCT((5,1) + self.smaps.shape[1:], (True,True,True), self.lamda)
		else:
			self._dctprox = None

		self.max_eig_run()

	def max_eig_run(self, print_info=True):
		maxeigop = BatchedSenseNormalLinop([self.coord_vec[0]], self.smaps)
		self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run(print_info=print_info).to(torch.float32)

	def run(self, image, iter=50, callback=None):

		if self.solver == 'GD':
			gd = GradientDescent(self._sensenormalop, Vector(image), self._dctprox, 1 / self.max_eig, accelerate=True, max_iter=iter)
			image = gd.run(callback, print_info=True).tensor

		#return image



class LLROptions:
	def __init__(self, block_shapes=[16,16,16], block_strides=[16,16,16], block_iter: int = 4, 
			random=False, nblocks=1, thresh: float = 0.01, soft: bool = True):
		self.block_shape = block_shapes
		self.block_strides = block_strides
		self.block_iter = block_iter
		self.random = random
		self.nblocks = nblocks
		self.thresh = thresh
		self.soft = soft

class FivePointLLR:
	def __init__(self, smaps, coord_vec, kdata_vec, llr_options: LLROptions, nframes: int, 
			weights_vec=None, streams=None, solver='GD'):
		
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.llr_options = llr_options
		self.nframes = nframes

		self.streams = streams

		self.solver = solver

		self._to_svtop = InnerOuterAutomorphism(self.nframes, 5, self.smaps.shape[1:])
		self._to_gradop = OuterInnerAutomorphism(self.nframes, 1, self.smaps.shape[1:])

		if self.solver == 'GD':
			self._sensenormalop = BatchedSenseNormalLinop(self.coord_vec, self.smaps, self.kdata_vec,
				self.weights_vec, self.streams)
		
		if self.llr_options.random:
			self._svtop = self._to_gradop * Random3DBlocksSVT((nframes, 5) + self.smaps.shape[1:], streams, self.llr_options.nblocks, 
				   self.llr_options.block_shape, self.llr_options.thresh, self.llr_options.soft) * self._to_svtop
		else:
			self._svtop = self._to_gradop * Normal3DBlocksSVT((nframes, 5) + self.smaps.shape[1:], streams, self.llr_options.block_strides,
				   self.llr_options.block_shape, self.llr_options.block_iter, self.llr_options.thresh, self.llr_options.soft) * self._to_svtop
		
		self.max_eig_run()

	def max_eig_run(self):
		maxeigop = BatchedSenseNormalLinop([self.coord_vec[0]], self.smaps)
		self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run().to(torch.float32)

	def run(self, image, iter=50, callback=None):

		if self.solver == 'GD':
			gd = GradientDescent(self._sensenormalop, Vector(image), self._svtop, 1 / self.max_eig, accelerate=True, max_iter=50)
			image = gd.run(callback, print_info=True).tensor

		#return image



