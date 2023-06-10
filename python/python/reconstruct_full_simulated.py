import torch
import numpy as np
import math

import cupy as cp
import cupyx

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

from torch_linop import TorchLinop, TorchScaleLinop
from torch_grad_methods import TorchCG, TorchGD
from torch_maxeig import TorchMaxEig

import reconstruct_util as ru

coords, kdatas, nframes, nenc = simri.load_coords_kdatas('D:/4DRecon/dat/dat2')
smaps = torch.tensor(simri.load_smaps('D:/4DRecon/dat/dat2'))

im_size = (smaps.shape[1],smaps.shape[2],smaps.shape[3])

vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])
images = torch.zeros(vec_size, dtype=torch.complex64)


if False:
	print('Beginning weighted load')
	diagonals, rhs = load_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=0)
	print('Beginning weighted reconstruct')
	images = reconstruct_cg(diagonals, rhs, smaps, nenc, iter=100, lamda=100.0, images=images, plot=False)

	pu.image_5d(np.abs(images))

if False:
	print('Beginning unweighted load')
	diagonals, rhs = load_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=True, root=2)
	print('Beginning unweighted reconstruct')
	images = reconstruct_gd(diagonals, rhs, smaps, nenc, iter=100, lamda=5e5, images=images, plot=False)

	pu.image_5d(np.abs(images))

if True:
	print('Beginning unweighted load')
	diagonals, rhs = ru.load_simulated_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=1)
	print('Beginning unweighted reconstruct')
	images = ru.reconstruct_gd_full(diagonals, rhs, smaps, nenc, iter=100, lamda=5e4, images=images, plot=False)

	pu.image_5d(np.abs(images))
	pu.image_4d(np.mean(np.abs(images.numpy()), axis=0))


with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed_weighted.h5', "w") as f:
	f.create_dataset('images', data=images)





