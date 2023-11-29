import torch

import hastypy.ffi.hasty_ffi as hasty_ffi

torch.classes.load_library(hasty_ffi.get_ffi_libfile())
torch.ops.load_library(hasty_ffi.get_ffi_libfile()) # only necessary for docstrings

_hasty_svt_mod = torch.classes.HastySVT

class Random3DBlocksSVT:
	def __init__(self, streams: list[torch.Stream] | None = None):
		self._svtop = _hasty_svt_mod.Random3DBlocksSVT(streams)
				
	def apply(self, input: torch.Tensor, nblocks: int, block_shape: list[int], thresh: float, soft: bool):
		self._svtop.apply(input, nblocks, block_shape, thresh, soft)

class Normal3DBlocksSVT:
	def __init__(self, streams: list[torch.Stream] | None = None):
		self._svtop = _hasty_svt_mod.Normal3DBlocksSVT(streams)

	def apply(self, input: torch.Tensor, block_strides: list[int], block_shape: list[int], block_iter: int, thresh: float, soft: bool):
		self._svtop.apply(input, block_strides, block_shape, block_iter, thresh, soft)


