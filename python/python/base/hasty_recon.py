import torch
import numpy as np

import base.hasty_base as hasty_base
from base.hasty_base import Linop

from ffi.hasty_sense import BatchedSense, BatchedSenseAdjoint, BatchedSenseNormal


class BatchedSenseLinop(Linop):
	def __init__(self, smaps, coord_vec, kdata_vec=None, weights_vec=None, random=(False, None), streams=None, ninner_batches=1):
		self.nvoxels = smaps.shape[0] * smaps.shape[1] * smaps.shape[2]
		
		self.nfreqs = []
		for coord in coord_vec:
			self.nfreqs.append(coord.shape[1])
		self.ninner_batches = ninner_batches
		self.nouter_batches = len(coord_vec)
		self.inshape = tuple([self.nouter_batches, self.ninner_batches] + list(smaps.shape[1:]))
		self.outshape = 

		self.ncoils = smaps.shape[0]
		self.random = random

		self._senseop = BatchedSense(coord_vec, smaps, kdata_vec, weights_vec, streams)

		super().__init__(self.shape, self.shape)

	def coil_list(self):
		if self.random[0]:
			coil_list = []
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.random[1]].tolist())
			return coil_list
		return None

	def _apply(self, input: torch.Tensor, output: list[torch.Tensor] | None = None):
		if output is None:
			for nfreq in self.nfreqs:
				output.append(torch.empty(self.ninner_batches, self.ncoils, nfreq))

		self._senseop.apply(input, output, self.coil_list())

		return input

class BatchedSenseAdjointLinop(Linop):
	def __init__(self, smaps, coord_vec, kdata_vec=None, weights_vec=None, random=(False, None), streams=None, inner_batches=1):
		self.nvoxels = smaps.shape[0] * smaps.shape[1] * smaps.shape[2]

		self._senseop = BatchedSense(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self.nouter_batches = len(coord_vec)
		self.shape = tuple([self.nframes, inner_batches] + list(smaps.shape[1:]))

		self.ncoils = smaps.shape[0]
		self.random = random

		super().__init__(self.shape, self.shape)

	def coil_list(self):
		if self.random[0]:
			coil_list = []
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.random[1]].tolist())
			return coil_list
		return None

	def _apply(self, input: torch.Tensor, output: torch.Tensor | None = None):
		if output is None:
			output = torch.empty_like(input)

		self._senseop.apply(input, output, self.coil_list())

		return input

class BatchedSenseNormalLinop(Linop):
	def __init__(self, smaps, coord_vec, kdata_vec=None, weights_vec=None, random=(False, None), streams=None, inner_batches=1):
		self.nvoxels = smaps.shape[0] * smaps.shape[1] * smaps.shape[2]

		self._senseop = BatchedSenseNormal(coord_vec, smaps, kdata_vec, weights_vec, streams)

		self.nouter_batches = len(coord_vec)
		self.shape = tuple([self.nframes, inner_batches] + list(smaps.shape[1:]))

		self.ncoils = smaps.shape[0]
		self.random = random

		super().__init__(self.shape, self.shape)

	def coil_list(self):
		if self.random[0]:
			coil_list = []
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.random[1]].tolist())
			return coil_list
		return None

	def _apply(self, input: torch.Tensor, output: torch.Tensor | None = None):
		if output is None:
			output = torch.empty_like(input)

		self._senseop.apply(input, output, self.coil_list())

		return input