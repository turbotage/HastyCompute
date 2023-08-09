import numpy as np

import itertools as itools
from numba import njit

import matplotlib.pyplot as plt

import torch
import torchkbnufft as tkbn
import math

import finufft


def take_adjoint(input, coords, im_size):
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

	#return torch.fft.fftshift(output)
	return output


ni = 16
nx = ni
ny = ni
nz = ni
nelem = nx*ny*nz
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
#weights = torch.rand(nf, dtype=torch.complex64).unsqueeze(0)
weights = torch.ones(nf, dtype=torch.complex64).unsqueeze(0)

adj_ob2 = tkbn.KbNufftAdjoint(
	im_size=im_size,
	n_shift=[0 for _ in range(coords.shape[0])],
	kbwidth=5,
	numpoints=12	
)

adj_out1 = take_adjoint(weights.squeeze(0), coords, im_size=im_size)

adj_out2 = adj_ob2(weights.unsqueeze(0), coords).squeeze(0).squeeze(0)


adj_out1 = adj_out1.numpy().flatten()
adj_out2 = adj_out2.numpy().flatten()

plt.figure()
plt.plot(np.real(adj_out1), 'r-*')
plt.plot(np.real(adj_out2), 'g-*')
plt.show()

plt.figure()
plt.plot(np.imag(adj_out1), 'r-*')
plt.plot(np.imag(adj_out2), 'g-*')
plt.show()

plt.figure()
plt.plot(np.abs(adj_out1), 'r-*')
plt.plot(np.abs(adj_out2), 'g-*')
plt.show()