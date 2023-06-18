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

def run_framed():
	print('Loading coords, kdata, weights')
	smaps, coord, kdata, weights, gating = ru.load_real()
	print('Gating')
	coord_vec, kdata_vec, weights_vec, gates = ru.gate(coord, kdata, weights, gating, nframes)
	del coord, kdata, weights, gating
	print('Stack frames for full image reconstruction')
	coord_vec, kdata_vec, weights_vec = ru.gated_full(
		coord_vec, kdata_vec, weights_vec, nframes)
	print('Crop kspace')
	coord_vec, kdata_vec, weights_vec = ru.crop_kspace(
		coord_vec, kdata_vec, weights_vec, (256,256,256), crop_factor=1.5)
	print('Translate')
	coord_vec, kdata_vec = ru.translate(coord_vec, kdata_vec, (0.25, 0.0, 0.0))

	print('Create full image diagonals and rhs')
	diagonals, rhs = ru.load_real_full_diag_rhs(
		smaps, coord_vec, kdata_vec, weights_vec, use_weights=True, root=1)
	print('Starting full image reconstruction')
	image = ru.reconstruct_cg_full(diagonals, rhs, smaps, nenc, iter=4, lamda=0, images=None, plot=False)
	
	pu.image_5d(np.abs(image))

	with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_real_reconstructed_weighted.h5', "w") as f:
		f.create_dataset('images', data=image)

	return image

def run():
	print('Loading coords, kdata, weights')
	smaps, coord, kdata, weights, _ = ru.load_real()
	print('Loading full')
	coord_vec, kdata_vec, weights_vec = ru.load_full_real(coord, kdata, weights)
	del coord, kdata, weights
	print('Crop kspace')
	#for i in range(5):
	#	coord_vec[i] = coord_vec[0]

	coord_vec, kdata_vec, weights_vec = ru.crop_kspace(
		coord_vec, kdata_vec, weights_vec, (256,256,256), crop_factor=1.5)
	print('Translate')
	coord_vec, kdata_vec = ru.translate(coord_vec, kdata_vec, (-50, 0.0, 0.0))

	images = ru.direct_nufft_reconstruct(smaps, coord_vec, kdata_vec, weights_vec, (256,256,256))
	
	#ming = torch.abs(torch.mean(images, dim=0)).numpy()

	#pu.image_5d(np.abs(images)) 

	#Prcnd_linop = None #ru.PrecondLinop(smaps, coord_vec, None)

	#diagonals, rhs = ru.load_real_full_diag_rhs(smaps, coord_vec, kdata_vec, weights_vec, use_weights=False, root=0)

	#toep_sense = ru.ToeplitzSenseLinop(smaps, diagonals)
	#sense = ru.SenseLinop(smaps, coord_vec, kdata_vec)

	#images1 = toep_sense(images)
	#images2 = sense(images)

	#pu.image_5d(np.abs(rhs))

	images = ru.reconstruct_gd_full(smaps, coord_vec, kdata_vec, weights_vec,
				 iter=20, lamda=0.0, images=None, plot=True)

	pu.image_5d(np.abs(images))

	with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_real_reconstructed_weighted.h5', "w") as f:
		f.create_dataset('images', data=images)

	return images

with torch.inference_mode():
	run()

def tested_batched_sense():
	dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
	torch.ops.load_library(dll_path)
	hasty_sense = torch.ops.HastySense

	N1 = 2
	N2 = 2
	N3 = 2
	NF = N1*N2*N3
	im_size = (N1,N2,N3)

	coord = np.empty((3,NF), dtype=np.float32)
	l = 0
	for x in range(N1):
		for y in range(N2):
			for z in range(N3):
				kx = -np.pi + x * 2 * np.pi / N1
				ky = -np.pi + y * 2 * np.pi / N2
				kz = -np.pi + z * 2 * np.pi / N3

				coord[0,l] = kx
				coord[1,l] = ky
				coord[2,l] = kz

				l += 1
	coord = torch.tensor(coord).to(torch.device('cuda:0'))


	diagonal = torch.ones((2*N1,2*N2,2*N3), dtype=torch.float32) #torch.real(tkbn.calc_toeplitz_kernel(omega=coord, im_size = im_size)).cpu()
	diagonal_vec = diagonal.unsqueeze(0)

	smaps = torch.rand((12,N1,N2,N3), dtype=torch.complex64)

	#toep_linop = ru.ToeplitzLinop((1,1,N1,N2,N3), smaps, 1, diagonal_vec)

	unity = torch.ones((1,1,N1,N2,N3), dtype=torch.complex64)

	#toep_linop(unity)
	coil_list = list()
	for i in range(nenc):
		inner_coil_list = list()
		for j in range(smaps.shape[0]):
			inner_coil_list.append(j)
		coil_list.append(inner_coil_list)

	hasty_sense.batched_sense_toeplitz_diagonals(unity, coil_list, smaps, diagonal_vec)

	SHS = torch.sum(smaps.conj() * smaps, dim=0)

	print('Hello')