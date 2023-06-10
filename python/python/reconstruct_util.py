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

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

def load_simulated_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=0):
	# mean flow
	ncoils = smaps.shape[0]

	coord_vec = []
	kdata_vec = []
	for encode in range(nenc):
		frame_coords = []
		frame_kdatas = []
		for frame in range(nframes):
			frame_coords.append(coords[frame][encode])
			frame_kdatas.append(kdatas[frame][encode])

		coord = np.concatenate(frame_coords, axis=1)
		kdata = np.concatenate(frame_kdatas, axis=2)
		coord_vec.append(torch.tensor(coord))
		kdata_vec.append(torch.tensor(kdata))

	diagonal_vec = []
	rhs_vec = []

	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	nvoxels = math.prod(list(im_size))
	cudev = torch.device('cuda:0')

	for i in range(nenc):
		print('Encoding: ', i)
		coord = coord_vec[i]
		kdata = kdata_vec[i]

		coord_cu = coord.to(cudev)

		weights: torch.Tensor
		if use_weights:
			print('Calculating density compensation')
			weights = tkbn.calc_density_compensation_function(ktraj=coord_cu, 
				im_size=im_size).to(torch.float32)
			
			for _ in range(root):
				weights = torch.sqrt(weights)
			
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, weights=weights.squeeze(0), im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

			weights = torch.sqrt(weights).squeeze(0)
		else:
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

		uimsize = [1,im_size[0],im_size[1],im_size[2]]

		print('Calculating RHS')
		rhs = torch.zeros(tuple(uimsize), dtype=torch.complex64).to(cudev)
		for j in range(ncoils): #range(nsmaps):
			print('Coil: ', j, '/', ncoils)
			SH = smaps[j,...].conj().to(cudev).unsqueeze(0)
			b = kdata[j,0,...].unsqueeze(0).to(cudev)
			if use_weights:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, weights * b, uimsize) / math.sqrt(nvoxels)
				rhs += rhs_j
			else:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, b, uimsize) / math.sqrt(nvoxels)
				rhs += rhs_j

		rhs_vec.append(rhs)

	rhs = torch.stack(rhs_vec, dim=0).cpu()
	diagonals = torch.stack(diagonal_vec, dim=0)

	return diagonals, rhs

def load_real_smaps_coords_kdata():
	smap_vec = []
	with h5py.File('D:/4DRecon/dat/dat2/SenseMapsCpp.h5') as f:
		smaps = f['Maps']
		ncoils = len(smaps.keys())
		for i in range(ncoils):
			smap_vec.append(smaps['SenseMaps_' + str(i)][()])
	smaps = np.stack(smap_vec, axis=0)
	del smap_vec

	coord = np.array([0])
	kdata = np.array([0])

	with h5py.File('D:/4DRecon/MRI_Raw.h5') as f:
		kdataset = f['Kdata']

		kx_vec = []
		ky_vec = []
		kz_vec = []

		kdata_vec = []

		for i in range(5):
			kx_vec.append(kdataset['KX_E' + str(i)][()])
			ky_vec.append(kdataset['KY_E' + str(i)][()])
			kz_vec.append(kdataset['KZ_E' + str(i)][()])

			kdata_enc = []

			for j in range(32):
				kdata_enc.append(kdataset['KData_E'+str(i)+'_C'+str(j)][()])

			kdata_vec.append(np.stack(kdata_enc, axis=0))

		kx = np.stack(kx_vec, axis=0)
		ky = np.stack(ky_vec, axis=0)
		kz = np.stack(kz_vec, axis=0)

		coord = np.stack([kx,ky,kz], axis=0)
		kdata = np.stack(kdata_vec, axis=0)

	return (smaps, coord, kdata)

class ToeplitzLinop(TorchLinop):
	def __init__(self, shape, smaps, nenc, diagonals):
		coil_list = list()
		for i in range(nenc):
			inner_coil_list = list()
			for j in range(smaps.shape[0]):
				inner_coil_list.append(j)
			coil_list.append(inner_coil_list)
		self.coil_list = coil_list
		self.smaps = smaps
		self.diagonals = diagonals

		super().__init__(shape, shape)

	def _apply(self, input):
		input_copy = input.clone()
		hasty_sense.batched_sense_toeplitz_diagonals(input_copy, self.coil_list, self.smaps, self.diagonals)
		return input_copy

def dct_l1_prox(image, lamda):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			gpuimg = cupyx.scipy.fft.dctn(cp.array(image[i,j,...].numpy()))
			gpuimg = cp.exp(1j*cp.angle(gpuimg)) * cp.maximum(0, (cp.abs(gpuimg) - lamda))
			image[i,j,...] = torch.tensor(cupyx.scipy.fft.idctn(gpuimg).get())
	
	return image

def reconstruct_cg_full(diagonals, rhs, smaps, nenc, iter = 50, lamda=0.1, images=None, plot=False):
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzLinop(vec_size, smaps, nenc, diagonals)

	scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=12).run()).to(torch.float32)

	scaled_linop = TorchScaleLinop(toep_linop, scaling)
	rhs *= torch.sqrt(scaling)

	tcg = TorchCG(scaled_linop, rhs, images, max_iter=iter)
	
	prox = lambda x: dct_l1_prox(x, lamda)

	if lamda != 0.0:
		return tcg.run_with_prox(prox, plot)
	else:
		return tcg.run(plot)

def reconstruct_gd_full(diagonals, rhs, smaps, nenc, iter = 50, lamda=0.1, images=None, plot=False):
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzLinop(vec_size, smaps, nenc, diagonals)

	scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=12).run()).to(torch.float32)
	rhs *= torch.sqrt(scaling)

	scaled_linop = TorchScaleLinop(toep_linop, scaling)

	gradf = lambda x: scaled_linop(x) -  rhs

	tgd = TorchGD(gradf, images, alpha=1.0, accelerate=True, max_iter=iter)

	prox = lambda x: dct_l1_prox(x, lamda)

	if lamda != 0.0:
		return tgd.run_with_prox(prox, plot)
	else:
		return tgd.run(plot)
