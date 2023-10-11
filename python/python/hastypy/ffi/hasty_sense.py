import torch

#_dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
_dll_path = "D:/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"


torch.classes.load_library(_dll_path)
#torch.ops.load_library(_dll_path) # only necessary for docstrings

_hasty_sense_mod = torch.classes.HastySense
_hasty_batched_sense_mod = torch.classes.HastyBatchedSense

"""
Just the Sense operation, no stacked Sense operators, all tensors should reside on cuda device
"""


class Sense:
	def __init__(self, coords: torch.Tensor, nmodes: list[int]):
		self._senseop = _hasty_sense_mod.Sense(coords, nmodes)
		
	def apply(self, input, output, smaps, coils, imspace_storage=None, kspace_storage=None):
		self._senseop.apply(input, output, smaps, coils, imspace_storage, kspace_storage)


class SenseAdjoint:
	def __init__(self, coords: torch.Tensor, nmodes: list[int]):
		self._senseop = _hasty_sense_mod.SenseAdjoint(coords, nmodes)
		
	def apply(self, input, output, smaps, coils, imspace_storage=None, kspace_storage=None):
		self._senseop.apply(input, output, smaps, coils, imspace_storage, kspace_storage)


class SenseNormal:
	def __init__(self, coords: torch.Tensor, nmodes: list[int]):
		self._senseop = _hasty_sense_mod.SenseNormal(coords, nmodes)
		
	def apply(self, input, output, smaps, coils, imspace_storage=None, kspace_storage=None):
		self._senseop.apply(input, output, smaps, coils, imspace_storage, kspace_storage)


class SenseNormalAdjoint:
	def __init__(self, coords: torch.Tensor, nmodes: list[int]):
		self._senseop = _hasty_sense_mod.SenseNormalAdjoint(coords, nmodes)
		
	def apply(self, input, output, smaps, coils, imspace_storage=None):
		self._senseop.apply(input, output, smaps, coils, imspace_storage)


"""
The batched operation, tensors should reside on cpu, outer batches are moved to gpu for computation,
by passing multiple streams, multiple gpus can be used. Otherwise the default cuda device is used
"""

class BatchedSense:
	def __init__(self, coords: list[torch.Tensor], smaps: torch.Tensor, 
	      kdata: list[torch.Tensor] | None = None, weights: list[torch.Tensor] | None = None,
		  streams: list[torch.Stream] | None = None):
		self._senseop = _hasty_batched_sense_mod.BatchedSense(coords, smaps, kdata, weights, streams)
		
	def apply(self, input: torch.Tensor, output: list[torch.Tensor], coils: list[list[int]] | None = None):
		self._senseop.apply(input, output, coils)


class BatchedSenseAdjoint:
	def __init__(self, coords: list[torch.Tensor], smaps: torch.Tensor, 
	      kdata: list[torch.Tensor] | None = None, weights: list[torch.Tensor] | None = None,
		  streams: list[torch.Stream] | None = None):
		self._senseop = _hasty_batched_sense_mod.BatchedSenseAdjoint(coords, smaps, kdata, weights, streams)
		
	def apply(self, input: list[torch.Tensor], output: torch.Tensor, coils: list[list[int]] | None = None):
		self._senseop.apply(input, output, coils)


class BatchedSenseNormal:
	def __init__(self, coords: list[torch.Tensor], smaps: torch.Tensor, 
	      kdata: list[torch.Tensor] | None = None, weights: list[torch.Tensor] | None = None,
		  streams: list[torch.Stream] | None = None):
		self._senseop = _hasty_batched_sense_mod.BatchedSenseNormal(coords, smaps, kdata, weights, streams)
		
	def apply(self, input: torch.Tensor, output: torch.Tensor, coils: list[list[int]] | None = None):
		self._senseop.apply(input, output, coils)


class BatchedSenseNormalAdjoint:
	def __init__(self, coords: list[torch.Tensor], smaps: torch.Tensor, 
	      kdata: list[torch.Tensor] | None = None, weights: list[torch.Tensor] | None = None,
	      imspace_weights: list[torch.Tensor] | None = None,
		  streams: list[torch.Stream] | None = None):
		self._senseop = _hasty_batched_sense_mod.BatchedSenseNormalAdjoint(coords, smaps, kdata, weights, imspace_weights, streams)
		
	def apply(self, input: list[torch.Tensor], output: list[torch.Tensor], coils: list[list[int]] | None = None):
		self._senseop.apply(input, output, coils)
 