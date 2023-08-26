import torch
import numpy as np

from hastypy.base.torch_opalg import TorchLinop
from hastypy.base.opalg import Vector, Linop
import hastypy.ffi.hasty_sense as hasty_sense



class SenseT(TorchLinop):
	def __init__(self, coords: torch.Tensor, smaps: torch.Tensor, coils):
		self._senseop = hasty_sense.Sense(coords, [1] + list(smaps.shape[1:]))
		self.nfreq = coords.shape[1]
		self.ndim = coords.shape[0]
		self.ncoil = smaps.shape[0]
		self.smaps = smaps
		self.coils = coils

		super().__init__((1,) + self.smaps.shape[1:], (self.ncoil, self.nfreq))

	def _apply(self, input: torch.Tensor):
		output = torch.empty((self.ncoil,self.nfreq), dtype=input.dtype, device=input.device)
		self._senseop.apply(input, output, self.smaps, self.coils, None, None)
		return output

class SenseAdjointT(TorchLinop):
	def __init__(self, coords: torch.Tensor, smaps: torch.Tensor, coils):
		self._senseop = hasty_sense.SenseAdjoint(coords, [1] + list(smaps.shape[1:]))
		self.nfreq = coords.shape[1]
		self.ndim = coords.shape[0]
		self.ncoil = smaps.shape[0]
		self.smaps = smaps
		self.coils = coils

		super().__init__((self.ncoil, self.nfreq), (1,) + self.smaps.shape[1:])

	def _apply(self, input: torch.Tensor):
		output = torch.empty((1,) + self.smaps.shape[1:], dtype=input.dtype, device=input.device)
		self._senseop.apply(input, output, self.smaps, self.coils, None, None)
		return output

class SenseNormalT(TorchLinop):
	def __init__(self, coords: torch.Tensor, smaps: torch.Tensor, coils):
		self._senseop = hasty_sense.SenseNormal(coords, [1] + list(smaps.shape[1:]))
		self.nfreq = coords.shape[1]
		self.ndim = coords.shape[0]
		self.ncoil = smaps.shape[0]
		self.smaps = smaps
		self.coils = coils

		super().__init__((1,) + self.smaps.shape[1:], (1,) + self.smaps.shape[1:])

	def _apply(self, input: torch.Tensor):
		output = torch.empty_like(input)
		self._senseop.apply(input, output, self.smaps, self.coils, None, None)
		return output





class Sense(Linop):
	def __init__(self, coords: torch.Tensor, smaps: torch.Tensor, coils):
		self._senseop = hasty_sense.Sense(coords, [1] + list(smaps.shape[1:]))
		self.nfreq = coords.shape[1]
		self.ndim = coords.shape[0]
		self.ncoil = smaps.shape[0]
		self.smaps = smaps
		self.coils = coils

		super().__init__((1,) + self.smaps.shape[1:], (1, self.ncoil, self.nfreq))

	def _apply(self, input: Vector):
		if not input.istensor():
			raise RuntimeError("Input must be a tensor Vector")
		inp = input.get_tensor()
		output = torch.empty((1,self.ncoil,self.nfreq), dtype=inp.dtype, device=inp.device)
		self._senseop.apply(input, output, self.smaps, self.coils, None, None)
		return Vector(output)

class SenseAdjoint(Linop):
	def __init__(self, coords: torch.Tensor, smaps: torch.Tensor, coils):
		self._senseop = hasty_sense.SenseAdjoint(coords, [1] + list(smaps.shape[1:]))
		self.nfreq = coords.shape[1]
		self.ndim = coords.shape[0]
		self.ncoil = smaps.shape[0]
		self.smaps = smaps
		self.coils = coils

		super().__init__((1, self.ncoil, self.nfreq), (1,) + self.smaps.shape[1:])

	def _apply(self, input: Vector):
		if not input.istensor():
			raise RuntimeError("Input must be a tensor Vector")
		inp = input.get_tensor()
		output = torch.empty((1,) + self.smaps.shape[1:], dtype=inp.dtype, device=inp.device)
		self._senseop.apply(input, output, self.smaps, self.coils, None, None)
		return Vector(output)

class SenseNormal(Linop):
	def __init__(self, coords: torch.Tensor, smaps: torch.Tensor, coils):
		self._senseop = hasty_sense.SenseNormal(coords, [1] + list(smaps.shape[1:]))
		self.nfreq = coords.shape[1]
		self.ndim = coords.shape[0]
		self.ncoil = smaps.shape[0]
		self.smaps = smaps
		self.coils = coils

		super().__init__((1,) + self.smaps.shape[1:], (1,) + self.smaps.shape[1:])

	def _apply(self, input: Vector):
		if not input.istensor():
			raise RuntimeError("Input must be a tensor Vector")
		inp = input.get_tensor()
		output = torch.empty_like(inp)
		self._senseop.apply(input, output, self.smaps, self.coils, None, None)
		return Vector(output)





class BatchedSense(Linop):
	def __init__(self, coord_vec: list[torch.Tensor], smaps: torch.Tensor, kdata_vec: list[torch.Tensor] | None = None, 
			weights_vec: list[torch.Tensor] | None = None, streams: list[torch.Stream] | None = None, 
			coils=None, ninner_batches=1):
		self.nouter_batches = len(coord_vec)
		self.ninner_batches = ninner_batches
		self.ncoils = smaps.shape[0]
		self.coils = coils
		self.dimensions = smaps.shape[1:]

		if coord_vec[0].dtype == torch.float32:
			self.dtype = torch.complex64
		elif coord_vec[0].dtype == torch.float64:
			self.dtype = torch.complex128
		else:
			raise RuntimeError('Non supported fp type for coord')

		self.nfreqs = []
		for i in range(len(coord_vec)):
			self.nfreqs.append(coord_vec[i].shape[1])

		self.output = []
		for i in range(self.nouter_batches):
			self.output.append(torch.empty(self.ninner_batches, self.ncoils, self.nfreqs[i]))

		self._senseop = hasty_sense.BatchedSense(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self._calc_shapes()

		super().__init__(self.ishape, self.oshape)
		
	def _calc_shapes(self):
		self.ishape = (self.nouter_batches, self.ninner_batches, *self.dimensions)
		self.oshape = []
		for i in range(self.nouter_batches):
			self.oshape.append((self.ninner_batches, self.ncoils, self.nfreqs[i]))
		
	def reinit(self, ninner_batches=1, coils=None):
		ninner_batches_changed = False
		if self.ninner_batches != ninner_batches:
			ninner_batches_changed = True
		self.ninner_batches = ninner_batches

		self.coils = coils

		if ninner_batches_changed:
			for i in range(len(self.nfreqs)):
				self.output[i] = torch.empty(self.ninner_batches, self.ncoils, self.nfreqs[i])
			self._calc_shapes()
			super().reinitialize(self.ishape, self.oshape)

	def set_coil_randomization(self, randomize=False, num_rand_coils=16):
		self.randomize_coils = randomize
		self.num_rand_coils = num_rand_coils

	def get_coils(self):
		if self.randomize_coils:
			coil_list = list()
			for i in range(self.nouter_batches):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.num_rand_coils].tolist())
			return coil_list
		else:
			return self.coils

	def _apply(self, input: Vector) -> Vector:
		self._senseop.apply(input.get_tensor(), self.output, self.get_coils())
		return Vector(self.output)
	

class BatchedSenseAdjoint(Linop):
	def __init__(self, coord_vec: list[torch.Tensor], smaps: torch.Tensor, kdata_vec: list[torch.Tensor] | None = None, 
			weights_vec: list[torch.Tensor] | None = None, streams: list[torch.Stream] | None = None, 
			coils=None, ninner_batches=1):
		self.nouter_batches = len(coord_vec)
		self.ninner_batches = ninner_batches
		self.ncoils = smaps.shape[0]
		self.coils = coils
		self.dimensions = smaps.shape[1:]

		if coord_vec[0].dtype == torch.float32:
			self.dtype = torch.complex64
		elif coord_vec[0].dtype == torch.float64:
			self.dtype = torch.complex128
		else:
			raise RuntimeError('Non supported fp type for coord')

		self.nfreqs = []
		for i in range(len(coord_vec)):
			self.nfreqs.append(coord_vec[i].shape[1])

		self.output = torch.empty((self.nouter_batches, self.ninner_batches, *self.dimensions), dtype=self.dtype)

		self._senseop = hasty_sense.BatchedSenseAdjoint(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self._calc_shapes()

		super().__init__(self.ishape, self.oshape)
		
	def _calc_shapes(self):
		self.ishape = []
		for i in range(self.nouter_batches):
			self.oshape.append((self.ninner_batches, self.ncoils, self.nfreqs[i]))
		self.oshape = (self.nouter_batches, self.ninner_batches, *self.dimensions)
		
	def reinit(self, ninner_batches=1, coils=None):
		ninner_batches_changed = False
		if self.ninner_batches != ninner_batches:
			ninner_batches_changed = True
		self.ninner_batches = ninner_batches

		self.coils = coils

		if ninner_batches_changed:
			self.output = torch.empty((self.nouter_batches, self.ninner_batches, *self.dimensions), dtype=self.dtype)
			self._calc_shapes()
			super().reinitialize(self.ishape, self.oshape)

	def set_coil_randomization(self, randomize=False, num_rand_coils=16):
		self.randomize_coils = randomize
		self.num_rand_coils = num_rand_coils

	def get_coils(self):
		if self.randomize_coils:
			coil_list = list()
			for i in range(self.nouter_batches):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.num_rand_coils].tolist())
			return coil_list
		else:
			return self.coils

	def _apply(self, input: Vector) -> Vector:
		self._senseop.apply(input.get_tensorlist(), self.output, self.get_coils())
		return Vector(self.output)
	

class BatchedSenseNormal(Linop):
	def __init__(self, coord_vec: list[torch.Tensor], smaps: torch.Tensor, kdata_vec: list[torch.Tensor] | None = None, 
			weights_vec: list[torch.Tensor] | None = None, streams: list[torch.Stream] | None = None, 
			coils=None, ninner_batches=1):
		self.nouter_batches = len(coord_vec)
		self.ninner_batches = ninner_batches
		self.ncoils = smaps.shape[0]
		self.coils = coils
		self.dimensions = smaps.shape[1:]

		if coord_vec[0].dtype == torch.float32:
			self.dtype = torch.complex64
		elif coord_vec[0].dtype == torch.float64:
			self.dtype = torch.complex128
		else:
			raise RuntimeError('Non supported fp type for coord')

		self.nfreqs = []
		for i in range(len(coord_vec)):
			self.nfreqs.append(coord_vec[i].shape[1])

		self.output = torch.empty((self.nouter_batches, self.ninner_batches, *self.dimensions), dtype=self.dtype)

		self._senseop = hasty_sense.BatchedSenseNormal(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self._calc_shapes()

		self.set_coil_randomization()

		super().__init__(self.ishape, self.ishape)
		
	def _calc_shapes(self):
		self.ishape = (self.nouter_batches, self.ninner_batches, *self.dimensions)
		
	def reinit(self, ninner_batches=1, coils=None):
		ninner_batches_changed = False
		if self.ninner_batches != ninner_batches:
			ninner_batches_changed = True
		self.ninner_batches = ninner_batches

		self.coils = coils

		if ninner_batches_changed:
			self.output = torch.empty((self.nouter_batches, self.ninner_batches, *self.dimensions), dtype=self.dtype)
			self._calc_shapes()
			super().reinitialize(self.ishape, self.ishape)

	def set_coil_randomization(self, randomize=False, num_rand_coils=16):
		self.randomize_coils = randomize
		self.num_rand_coils = num_rand_coils

	def get_coils(self):
		if self.randomize_coils:
			coil_list = list()
			for i in range(self.nouter_batches):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.num_rand_coils].tolist())
			return coil_list
		else:
			return self.coils

	def _apply(self, input: Vector) -> Vector:
		self._senseop.apply(input.get_tensor(), self.output, self.get_coils())
		return Vector(self.output)
	

class BatchedSenseNormalAdjoint(Linop):
	def __init__(self, coord_vec: list[torch.Tensor], smaps: torch.Tensor, kdata_vec: list[torch.Tensor] | None = None, 
			weights_vec: list[torch.Tensor] | None = None, streams: list[torch.Stream] | None = None, 
			coils=None, ninner_batches=1):
		self.nouter_batches = len(coord_vec)
		self.ninner_batches = ninner_batches
		self.ncoils = smaps.shape[0]
		self.coils = coils
		self.dimensions = smaps.shape[1:]

		if coord_vec[0].dtype == torch.float32:
			self.dtype = torch.complex64
		elif coord_vec[0].dtype == torch.float64:
			self.dtype = torch.complex128
		else:
			raise RuntimeError('Non supported fp type for coord')

		self.nfreqs = []
		for i in range(len(coord_vec)):
			self.nfreqs.append(coord_vec[i].shape[1])

		self.output = []
		for i in range(self.nouter_batches):
			self.output.append(torch.empty(self.ninner_batches, self.ncoils, self.nfreqs[i]))
		
		self._senseop = hasty_sense.BatchedSenseAdjoint(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self._calc_shapes()

		super().__init__(self.ishape, self.oshape)
		
	def _calc_shapes(self):
		self.ishape = []
		for i in range(self.nouter_batches):
			self.oshape.append((self.ninner_batches, self.ncoils, self.nfreqs[i]))
		self.oshape = (self.nouter_batches, self.ninner_batches, *self.dimensions)
		
	def reinit(self, ninner_batches=1, coils=None):
		ninner_batches_changed = False
		if self.ninner_batches != ninner_batches:
			ninner_batches_changed = True
		self.ninner_batches = ninner_batches

		self.coils = coils
		
		if ninner_batches_changed:
			for i in range(len(self.nfreqs)):
				self.output[i] = torch.empty(self.ninner_batches, self.ncoils, self.nfreqs[i])
				self._calc_shapes()
				super().reinitialize(self.ishape, self.oshape)

	def set_coil_randomization(self, randomize=False, num_rand_coils=16):
		self.randomize_coils = randomize
		self.num_rand_coils = num_rand_coils

	def get_coils(self):
		if self.randomize_coils:
			coil_list = list()
			for i in range(self.nouter_batches):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.num_rand_coils].tolist())
			return coil_list
		else:
			return self.coils

	def _apply(self, input: Vector) -> Vector:
		self._senseop.apply(input.get_tensorlist(), self.output, self.get_coils())
		return Vector(self.output)
	
#(nf,ne,...) -> (nf*ne,1,...)
class OuterInnerAutomorphism(Linop):
	def __init__(self, nouter_batches, ninner_batches, sp_dims):
		super().__init__((nouter_batches, ninner_batches) + sp_dims, 
		   (nouter_batches*ninner_batches,1) + sp_dims)

	def _apply(self, input: Vector) -> Vector:
		return Vector(input.get_tensor().view(self.oshape))

#(nf*ne,1,...) -> (nf,ne,...)
class InnerOuterAutomorphism(Linop):
	def __init__(self, nouter_batches, ninner_batches, sp_dims):
		super().__init__((nouter_batches*ninner_batches,1) + sp_dims, 
		   (nouter_batches, ninner_batches) + sp_dims)

	def _apply(self, input: Vector) -> Vector:
		return Vector(input.get_tensor().view(self.oshape))
	






