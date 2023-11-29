import torch

import hastypy.ffi.hasty_ffi as hasty_ffi
import os

torch.classes.load_library(hasty_ffi.get_ffi_libfile())
#torch.ops.load_library(hasty_ffi.get_ffi_libfile()) # only necessary for docstrings

_hasty_nufft_mod = torch.classes.HastyInterface

class FunctionLambda:
	"""
	Example script could look like:
	def apply(a: List[Tensor], b: List[Tensor]):
		for i in range(len(b)):
			a[i] += b[i]
	"""
	def __init__(self, script: str, entry: str, captures: list[torch.Tensor]):
		self._funclam = _hasty_nufft_mod.FunctionLambda(script, entry, captures)
		
	def apply(self, input: list[torch.Tensor]):
		self._funclam.apply(input)