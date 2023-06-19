import torch
import numpy as np
import math

import cupy as cp
import cupyx

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

import image_creation as ic

from torch_linop import TorchLinop, TorchScaleLinop
from torch_grad_methods import TorchCG, TorchGD
from torch_maxeig import TorchMaxEig

import reconstruct_util as ru

nframes = 30
nenc = 5

def run_framed(images_full, smaps, coord, kdata, gating, nframes, shift=(0.0, 0.0, 0.0),crop_factor=1.5):
	im_size = (images_full.shape[2], images_full.shape[3], images_full.shape[4])
	print('Gating')
	coord_vec, kdata_vec, weights_vec, gates = ru.gate(coord, kdata, None, gating, nframes)
	del coord, kdata, gating
	print('Stack frames for full image reconstruction')
	coord_vec, kdata_vec, weights_vec = ru.gated_full(
		coord_vec, kdata_vec, weights_vec, nframes)
	print('Crop kspace')
	coord_vec, kdata_vec, weights_vec = ru.crop_kspace(
		coord_vec, kdata_vec, weights_vec, im_size, crop_factor=crop_factor)
	print('Translate')
	coord_vec, kdata_vec = ru.translate(coord_vec, kdata_vec, shift)

	images = torch.empty(nenc*nframes, 1, im_size[0], im_size[1], 
		      im_size[2], dtype=torch.complex64)
	
	if True:
		l = 0
		for i in range(nframes):
			for j in range(nenc):
				images[l,...] = images_full[j,...]
				l += 1

	
	

	return image

def run_full(im_size, store=True, shift=(0.0, 0.0, 0.0), crop_factor=1.5):
	print('Loading coords, kdata, weights')
	smaps, coord, kdata, weights, gating = ru.load_real()
	print('Loading full')
	coord_vec, kdata_vec, weights_vec = ru.load_full_real(coord, kdata, weights)
	#del coord, kdata, weights
	print('Crop kspace')

	coord_vec, kdata_vec, weights_vec = ru.crop_kspace(
		coord_vec, kdata_vec, weights_vec, im_size, crop_factor=crop_factor)
	print('Translate')
	coord_vec, kdata_vec = ru.translate(coord_vec, kdata_vec, shift)

	print('Direct reconstruction')
	images = ru.direct_nufft_reconstruct(smaps, coord_vec, kdata_vec, weights_vec, im_size)

	print('Load toeplitz diagonal and rhs')
	diagonals, rhs = ru.load_real_full_diag_rhs(smaps, coord_vec, kdata_vec, 
					     weights_vec, use_weights=False, root=0)

	print('Reconstruct')
	images = ru.reconstruct_cg_full(diagonals, rhs, smaps, None,
				 iter=10, lamda=0.0, images=images, plot=True)

	pu.image_5d(images.numpy())

	if store:
		print('Storing fully reconstructed')
		with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_real.h5', "w") as f:
			f.create_dataset('images', data=images)

	return images, smaps, coord, kdata, gating



