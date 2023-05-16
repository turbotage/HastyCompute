import numpy as np

import itertools as itools
from numba import njit

import matplotlib.pyplot as plt

import torch
import torchkbnufft as tkbn
import math

import finufft

def take_adjoint(input, coords, im_size, my_nufft=True):
	
	if not my_nufft:
		adj_ob = tkbn.KbNufftAdjoint(
			im_size=im_size,
			n_shift=[0 for _ in range(coords.shape[0])],
			kbwidth=5,
			numpoints=12
		)
		return adj_ob(input.unsqueeze(0).unsqueeze(0), coords).squeeze(0).squeeze(0)
	else:
		ndim = coords.shape[0]

		angle = torch.tensor(0)
		if ndim == 1:
			angle = torch.tensor([im_size[0] // 2])
		elif ndim == 2:
			angle = torch.tensor([im_size[0] // 2, im_size[1] // 2])
		elif ndim == 3:
			angle = torch.tensor([im_size[0] // 2, im_size[1] // 2, im_size[2] // 2])

		def shifted_input(input, coord, angle):
			shift = torch.sum(coord * angle.unsqueeze(-1), 0)
			return input * (torch.cos(shift) + 1j*torch.sin(shift))

		coords_ = coords.numpy()
		input_ = shifted_input(input, coords, angle).numpy()
		output = torch.tensor([0])
		if ndim == 1:
			output = torch.tensor(finufft.nufft1d1(coords_[0,:], input_, n_modes=im_size))
		elif ndim == 2:
			output = torch.tensor(finufft.nufft2d1(coords_[0,:], coords_[1,:], input_, n_modes=im_size))
		elif ndim == 3:
			output =  torch.tensor(finufft.nufft3d1(coords_[0,:], coords_[1,:], coords_[2,:], input_, n_modes=im_size))

		return output
		

def toeplitz_diagonal(weights, coords, im_size):
	ndim = coords.shape[0]
	
	adjnufft = lambda weights, coords: take_adjoint(weights, coords, im_size, my_nufft=False)

	if ndim > 1:
		diagonal = adjoint_flip_and_concat(1, weights, coords, adjnufft)
	else:
		diagonal = adjnufft(weights, coords)

	return diagonal
	diagonal = reflect_conj_concat(0, diagonal)

	diagonal = hermitify(0, diagonal)

	return torch.fft.fftn(diagonal, dim=[-1], norm="forward")
	
		
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


ni = 6
nx = ni
ny = ni
nz = ni
nelem = nx*ny*nz
nf = 80


ndim = 3
im_size = None
if ndim == 1:
	im_size = (nx,)
elif ndim == 2:
	im_size = (nx,ny)
elif ndim == 3:
	im_size = (nx,ny,nz)


coords = -3.141592 + 2*3.141592*torch.rand(ndim,nf, dtype=torch.float32)
#weights = torch.rand(nf, dtype=torch.complex64).unsqueeze(0)
weights = tkbn.calc_density_compensation_function(coords, im_size).squeeze(0)


diag = np.array([0])
diag2 = np.array([0])

if True:

	#diag1 = tkbn.calc_toeplitz_kernel(omega=coords, im_size=im_size)
	diag1 = tkbn.calc_toeplitz_kernel(omega=coords, weights=weights, im_size=im_size)
	diag1 = diag1.squeeze(0).squeeze(0)

	diag2 = toeplitz_diagonal(weights=weights.squeeze(0), coords=coords, im_size=im_size)

	diag1_p = diag1.numpy().flatten()
	diag2_p = diag2.numpy().flatten()



	plt.figure()
	plt.plot(np.real(diag1_p), 'r-*')
	plt.plot(np.real(diag2_p), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.imag(diag1_p), 'r-*')
	plt.plot(np.imag(diag2_p), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.abs(diag1_p), 'r-*')
	plt.plot(np.abs(diag2_p), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	x1 = take_adjoint(input=weights.squeeze(0), coords=coords, im_size=im_size, my_nufft=True)
	x2 = take_adjoint(input=weights.squeeze(0), coords=coords, im_size=im_size, my_nufft=False)

	x1 = x1.numpy().flatten()
	x2 = x2.numpy().flatten()

	plt.figure()
	plt.plot(np.real(x1), 'r-*')
	plt.plot(np.real(x2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.imag(x1), 'r-*')
	plt.plot(np.imag(x2), 'g-*')
	plt.legend(['diag1', 'diag2'])
	plt.show()

	plt.figure()
	plt.plot(np.abs(x1), 'r-*')
	plt.plot(np.abs(x2), 'g-*')
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

	#angle = -torch.tensor(torch.pi) / 2
	#shifted_coords = torch.sum(coords * angle, 1)
	#shift = (torch.cos(shifted_coords) - 1j*torch.sin(shifted_coords))
	#shift[:(nf // 2)] = 1
	#weights = weights * shift
	adj_out2 = take_adjoint(weights.squeeze(0), coords, im_size)
	#adj_out2 = torch.fft.ifftshift(adj_out2)

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