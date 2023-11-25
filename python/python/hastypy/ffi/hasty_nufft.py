import torch
from typing import Optional


#_dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
_dll_path = "D:/Documents/GitHub/HastyCompute/out/build/msvc-release-cuda/Release/HastyPyInterface.dll"


torch.classes.load_library(_dll_path)
torch.ops.load_library(_dll_path) # only necessary for docstrings

_hasty_nufft_mod = torch.classes.HastyNufft

from hastypy.ffi.hasty_interface import FunctionLambda


"""
Nufft operations, all tensors should reside on cuda device
"""

class NufftOptions:
	def __init__(self, type: int, positive: bool | None = None, tol: float | None = None):
		self._nufftopts = _hasty_nufft_mod.NufftOptions(type, positive, tol)

	@staticmethod
	def type2():
		return NufftOptions(2, False, 1e-5)
	
	@staticmethod
	def type1():
		return NufftOptions(1, True, 1e-5)


class Nufft:
	def __init__(self, coord: torch.Tensor, nmodes: list[int], opts: NufftOptions):
		self._nufftopt = _hasty_nufft_mod.Nufft(coord, nmodes, opts._nufftopts)
		
	def apply(self, input: torch.Tensor, output: torch.Tensor):
		self._nufftopt.apply(input, output)

class NufftNormal:
	def __init__(self, coord: torch.Tensor, nmodes: list[int], forward_opts: NufftOptions, backward_opts: NufftOptions):
		self._nufftopt = _hasty_nufft_mod.NufftNormal(coord, nmodes, forward_opts._nufftopts, backward_opts._nufftopts)

	def apply(self, input: torch.Tensor, output: torch.Tensor, storage: torch.Tensor, callable: FunctionLambda | None = None):
		self._nufftopt.apply(input, output, storage, callable._funclam if callable is not None else None)


