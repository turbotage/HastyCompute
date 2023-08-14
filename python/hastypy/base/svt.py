import torch
import numpy as np

from hastypy.base.opalg import Vector, Linop
from hastypy.ffi.hasty_svt import Random3DBlocksSVT, Normal3DBlocksSVT

class Random3DBlocksSVT(Linop):
	def __init__(self, input_shape, streams: list[torch.Stream], nblocks: int, 
			block_shape: list[int], thresh: float, soft: bool):
		self._svtop = Random3DBlocksSVT(streams)
		self.nblocks = nblocks
		self.block_shape = block_shape
		self.thresh = thresh
		self.soft = soft
		
		super().__init__(input_shape, input_shape)
		
	def _apply(self, input: Vector):
		self._svtop.apply(input.get_tensor(), self.nblocks, self.block_shape, self.thresh, self.soft)

class Normal3DBlocksSVT(Linop):
	def __init__(self, input_shape, streams: list[torch.Stream], block_strides: list[int], 
	      block_shape: list[int], block_iter: int, thresh: float, soft: bool):
		self._svtop = Normal3DBlocksSVT(streams)
		self.block_strides = block_strides
		self.block_shape = block_shape
		self.block_iter = block_iter
		self.thresh = thresh
		self.soft = soft
		
		super().__init__(input_shape, input_shape)

	def _apply(self, input: Vector):
		self._svtop.apply(input.get_tensor(), self.block_strides, self.block_shape, self.block_iter, self.thresh, self.soft)