import torch
import numpy as np

import base.hasty_base as hasty_base
from base.hasty_base import Linop

from ffi.hasty_sense import BatchedSense, BatchedSenseAdjoint, BatchedSenseNormal

"""
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
"""


def crop_kspace(coord_vec, kdata_vec, weights_vec, im_size, crop_factor=1.0, prefovkmul=1.0, postfovkmul=1.0):
	max = 0.0
	for i in range(len(coord_vec)):
		maxi = coord_vec[i].abs().max().item()
		if maxi > max:
			max = maxi

	kim_size = tuple((e / 2) * crop_factor for e in im_size)

	for i in range(len(coord_vec)):
		coord = coord_vec[i] * prefovkmul
		idxx = torch.abs(coord[0,:]) < kim_size[0]
		idxy = torch.abs(coord[1,:]) < kim_size[1]
		idxz = torch.abs(coord[2,:]) < kim_size[2]

		idx = torch.logical_and(idxx, torch.logical_and(idxy, idxz))

		coord_vec[i] = postfovkmul * torch.pi * coord[:,idx] / maxi
		kdata_vec[i] = kdata_vec[i][:,:,idx]
		if weights_vec is not None:
			weights_vec[i] = weights_vec[i][:,idx]
	
	return (coord_vec, kdata_vec, weights_vec)

def translate(coord_vec, kdata_vec, translation):
	cudev = torch.device('cuda:0')
	for i in range(len(coord_vec)):
		coord = coord_vec[i].to(cudev)
		kdata = kdata_vec[i].to(cudev)
		mult = torch.tensor(list(translation)).unsqueeze(0).to(cudev)

		kdata *= torch.exp(1j*(mult @ coord)).unsqueeze(0)

		kdata_vec[i] = kdata.cpu()

	return kdata_vec

def center_weights(im_size, width, coord, weights):
	scaled_coords = coord * torch.tensor(list(im_size))[:,None] / (2*torch.pi)
	idx = torch.abs(scaled_coords) < width
	idx = torch.logical_and(idx[0,:], torch.logical_and(idx[1,:], idx[2,:]))
	return weights[0,idx]




