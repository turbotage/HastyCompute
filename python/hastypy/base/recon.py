import torch
import numpy as np

from hastypy.base.sense import BatchedSenseNormalLinop
from hastypy.base.svt import Random3DBlocksSVT, Normal3DBlocksSVT

from hastypy.base.opalg import MaxEig, GradientDescent

class LLROptions:
	def __init__(self, block_shapes=(16,16,16), block_strides=(16,16,16), random=False, nblocks=1):
		self.block_shapes = block_shapes
		self.block_strides = block_strides
		self.random = random
		self.nblocks = nblocks

class FivePointLLR:
	def __init__(self, smaps, coord_vec, kdata_vec, llr_options: LLROptions, 
	      random_coils=False, num_rand_coils=16, lamda=0.01, weights_vec=None, streams=None):
		
		self.smaps = smaps
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec

		self.llr_options = llr_options
		self.random_coils = random_coils
		self.num_rand_coils = num_rand_coils

		self.lamda = lamda
		self.streams = streams

		self._sensenormalop = BatchedSenseNormalLinop(self.coord_vec, self.smaps, self.kdata_vec,
			self.weights_vec, self.streams)
		self._sensenormalop.set_coil_randomization(self.random_coils, self.num_rand_coils)
		
		if self.llr_options.random:
			self._svtop = Random3DBlocksSVT(streams)
		else:
			self._svtop = Normal3DBlocksSVT(streams)

	def _svt_l1_prox(self, input):
		"""
		h
		"""
		pass


	def max_eig_run(self):
		max_eig: torch.Tensor
		maxeigop = BatchedSenseNormalLinop([self.coord_vec[0]], self.smaps)
		self.max_eig = MaxEig(maxeigop, torch.complex64, max_iter=5).run().to(torch.float32)

	def run(self):
		
