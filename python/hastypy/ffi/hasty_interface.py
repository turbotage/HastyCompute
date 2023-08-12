import torch

_dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"

torch.classes.load_library(_dll_path)
#torch.ops.load_library(_dll_path) # only necessary for docstrings

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