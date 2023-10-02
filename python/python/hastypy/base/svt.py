import torch
import numpy as np
import random

from hastypy.base.opalg import Vector, Operator, Linop
import hastypy.ffi.hasty_svt as ffisvt

def extract_mean_block(images, imsize, blocksize, nblocks=200):
	Ss = []
	for i in range(nblocks):
		rb = (
				random.randint(imsize[0]//4, 3*imsize[0]//4 - blocksize[0]), 
				random.randint(imsize[1]//4, 3*imsize[1]//4 - blocksize[2]),
				random.randint(imsize[2]//4, 3*imsize[2]//4 - blocksize[2])
			)

		block = images[:,:,rb[0]:(rb[0]+blocksize[0]),rb[1]:(rb[1]+blocksize[1]),rb[2]:(rb[2]+blocksize[2])].contiguous()
		block = block.flatten(0,0).flatten(1).contiguous()
		block = block.transpose(0,1).contiguous()

		Ss.append(torch.linalg.svdvals(block))
	Stensor = torch.concat(Ss, dim=0)
	Smean = torch.mean(torch.stack(Ss, axis=0), axis=0)

	Stensor, _ = torch.sort(Stensor)

	return (torch.min(Stensor), torch.median(Stensor), torch.max(Stensor), Smean)

class Random3DBlocksSVT(Operator):
	def __init__(self, input_shape, streams: list[torch.Stream], nblocks: int, 
			block_shape: list[int], thresh: float, soft: bool):
		self._svtop = ffisvt.Random3DBlocksSVT(streams)
		self.nblocks = nblocks
		self.block_shape = block_shape
		self.thresh = thresh
		self.soft = soft
		
		super().__init__(input_shape, input_shape)
		
	def update_thresh(self, thresh):
		self.thresh = thresh

	def _apply(self, input: Vector):
		self._svtop.apply(input.get_tensor(), self.nblocks, self.block_shape, self.thresh, self.soft)

class Normal3DBlocksSVT(Operator):
	def __init__(self, input_shape, streams: list[torch.Stream], block_strides: list[int], 
	      block_shape: list[int], block_iter: int, thresh: float, soft: bool):
		self._svtop = ffisvt.Normal3DBlocksSVT(streams)
		self.block_strides = block_strides
		self.block_shape = block_shape
		self.block_iter = block_iter
		self.thresh = thresh
		self.soft = soft
		
		super().__init__(input_shape, input_shape)

	def update_thresh(self, thresh):
		self.thresh = thresh

	def _apply(self, input: Vector):
		self._svtop.apply(input.get_tensor(), self.block_strides, self.block_shape, self.block_iter, self.thresh, self.soft)