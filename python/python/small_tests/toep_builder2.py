import numpy as np

import itertools as itools
from numba import njit

import matplotlib.pyplot as plt

import torch
import torchkbnufft as tkbn

import finufft

def toeplitz_diagonal(weights, coords, im_size):
	ndim = coords.shape[0]

	def take_adjoint(input, coords):
		ndim = coords.shape[0]
		if ndim == 1:
			return finufft.nufft1d1(input, coords)
		elif ndim == 2:
			return finufft.nufft2d1(input, coords)
		elif ndim == 3:
			return finufft.nufft3d1(input, coords)
		
	if ndim > 1:
		diagonal = adjoint_flip_and_concat(1, weights, coords, take_adjoint)
	else:
		diagonal = take_adjoint(weights, coords)

	diagonal = reflect_conj_concat(2, diagonal)

	diagonal = hermitify(2, diagonal)
	
	
	
		
def adjoint_flip_and_concat(dim, weights, coords, take_adjoint):
	ndim = coords.shape[0]

	if dim < ndim-1:
		diagonal1 = adjoint_flip_and_concat(dim+1, weights, coords, take_adjoint)
		flipped_coords = coords.clone()
		flipped_coords[dim].neg_()
		diagonal2 = adjoint_flip_and_concat(dim+1, weights, flipped_coords, take_adjoint)
	else:
		diagonal1 = take_adjoint(weights, coords)
		flipped_coords = coords.clone()
		flipped_coords[dim].neg_()
		diagonal2 = take_adjoint(weights, coords)

	zero_shape = list(diagonal1.shape)
	zero_shape[dim] = 1
	zero_block = torch.zeros(tuple(zero_shape), dtype=diagonal1.dtype, device=diagonal1.device)

	diagonal2 = diagonal2.narrow(dim, 1, diagonal2.shape[dim] - 1)

	return torch.cat((diagonal1, zero_block, diagonal2.flip(dim)), dim)

def reflect_conj_concat(dim, diagonal):
	dtype, device = diagonal.dtype, diagonal.device
	flipdims = torch.arange(dim, diagonal.ndim, device=device)

	# calculate size of central z block
	zero_block_shape = torch.tensor(diagonal.shape, device=device)
	zero_block_shape[dim] = 1
	zero_block = torch.zeros(*zero_block_shape, dtype=dtype, device=device)

	# reflect the original block and conjugate it
	# the below code looks a bit hacky but we don't want to flip the 0 dim
	# TODO: make this better
	tmp_block = diagonal.conj()
	for d in flipdims:
		tmp_block = tmp_block.index_select(
			d,
			torch.remainder(
				-1 * torch.arange(tmp_block.shape[d], device=device), tmp_block.shape[d]
			),
		)
	tmp_block = torch.cat(
		(zero_block, tmp_block.narrow(dim, 1, tmp_block.shape[dim] - 1)), dim
	)

	# concatenate and return
	return torch.cat((diagonal, tmp_block), dim)

def hermitify(dim, diagonal):
	device = diagonal.device

	start = diagonal.clone()

	# reverse coordinates for each dimension
	# the below code looks a bit hacky but we don't want to flip the 0 dim
	# TODO: make this better
	for d in range(dim, diagonal.ndim):
		diagonal = diagonal.index_select(
			d,
			torch.remainder(
				-1 * torch.arange(diagonal.shape[d], device=device), diagonal.shape[d]
			),
		)

	return (start + diagonal.conj()) / 2