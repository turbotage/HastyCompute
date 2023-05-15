import numpy as np

import itertools as itools
from numba import njit

import matplotlib.pyplot as plt

import torch
import torchkbnufft as tkbn

import finufft

def take_adjoint(input, coords, im_size):
		coords_ = coords.numpy()
		input_ = input.numpy()
		ndim = coords.shape[0]
		output = torch.tensor([0])
		if ndim == 1:
			output = torch.tensor(finufft.nufft1d1(coords_[0,:], input_, n_modes=im_size))
		elif ndim == 2:
			output = torch.tensor(finufft.nufft2d1(coords_[0,:], coords_[1,:], input_, n_modes=im_size))
		elif ndim == 3:
			output =  torch.tensor(finufft.nufft3d1(coords_[0,:], coords_[1,:], coords_[2,:], input_, n_modes=im_size))
		return output
		#return torch.fft.fftshift(output)

def toeplitz_diagonal(weights, coords, im_size):
	ndim = coords.shape[0]
	
	adjnufft = lambda weights, coords: take_adjoint(weights, coords, im_size)

	if ndim > 1:
		diagonal = adjoint_flip_and_concat(1, weights, coords, adjnufft)
	else:
		diagonal = adjnufft(weights, coords)

	return diagonal
	diagonal = reflect_conj_concat(0, diagonal)

	#diagonal = hermitify(0, diagonal)
	
	return torch.fft.fftn(diagonal)
	
		
def adjoint_flip_and_concat(dim, weights, coords, adjnufft):
	ndim = coords.shape[0]

	if dim < ndim-1:
		diagonal1 = adjoint_flip_and_concat(dim+1, weights, coords, adjnufft)
		flipped_coords = coords.clone()
		flipped_coords[dim].neg_()
		diagonal2 = adjoint_flip_and_concat(dim+1, weights, flipped_coords, adjnufft)
	else:
		diagonal1 = adjnufft(weights, coords)
		flipped_coords = coords.clone()
		flipped_coords[dim].neg_()
		diagonal2 = adjnufft(weights, coords)

	zero_shape = list(diagonal1.shape)
	zero_shape[dim] = 1
	zero_block = torch.zeros(tuple(zero_shape), dtype=diagonal1.dtype, device=diagonal1.device)

	diagonal2 = diagonal2.narrow(dim, 1, diagonal2.shape[dim] - 1)

	return torch.cat((diagonal1, zero_block, diagonal2.flip(dim)), dim)

def reflect_conj_concat(dim, diagonal):
	dtype, device = diagonal.dtype, diagonal.device
	flipdims = torch.arange(0, diagonal.ndim, device=device)

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


ni = 64
nx = ni
ny = ni
nz = ni
nf = 80


ndim = 1
im_size = None
if ndim == 1:
	im_size = (nx,)
elif ndim == 2:
	im_size = (nx,ny)
elif ndim == 3:
	im_size = (nx,ny,nz)


coords = -3.141592 + 2*3.141592*torch.rand(ndim,nf, dtype=torch.float32)
coords2 = coords.clone()
weights = torch.rand(nf, dtype=torch.complex64).unsqueeze(0)
weights2 = weights.clone()

if True:
	diag = tkbn.calc_toeplitz_kernel(omega=coords, weights=weights, im_size=(nx,))

	weights2.squeeze_(0)
	diag2 = toeplitz_diagonal(weights=weights2, coords=coords2, im_size=(nx,))
	diag2 = torch.fft.ifftshift(diag2)

	print(diag.shape)
	print(diag2.shape)

	diag = diag.flatten()
	diag2 = diag2.flatten()



	plt.figure()
	plt.plot(np.real(diag), 'r-*')
	plt.plot(np.real(diag2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.imag(diag), 'r-*')
	plt.plot(np.imag(diag2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.abs(diag), 'r-*')
	plt.plot(np.abs(diag2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()


if False:
	adj_ob = tkbn.KbNufftAdjoint(
		im_size=im_size,
		n_shift=[0 for _ in range(coords.shape[0])],
		kbwidth=5,
		numpoints=12
	)
	adj_out1 = adj_ob(weights.unsqueeze(0), coords)

	angle = -torch.tensor(torch.pi) / 2
	shifted_coords = torch.sum(coords * angle, 1)
	shift = (torch.cos(shifted_coords) - 1j*torch.sin(shifted_coords))
	shift[:(nf // 2)] = 1
	weights = weights * shift
	adj_out2 = take_adjoint(weights.squeeze(0), coords, im_size)
	adj_out2 = torch.fft.ifftshift(adj_out2)

	adj_out1 = adj_out1.numpy().flatten()
	adj_out2 = adj_out2.numpy().flatten()

	plt.figure()
	plt.plot(np.real(adj_out1), 'r-*')
	plt.plot(np.real(adj_out2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.imag(adj_out1), 'r-*')
	plt.plot(np.imag(adj_out2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.abs(adj_out1), 'r-*')
	plt.plot(np.abs(adj_out2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()