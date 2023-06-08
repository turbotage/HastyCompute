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

def load_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=0):
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

	nsmaps = smaps.shape[0]
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
		for j in range(nsmaps): #range(nsmaps):
			print('Coil: ', j, '/', nsmaps)
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
		return input_copy + 0.01*input


def dct_prox(image, lamda):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			gpuimg = cupyx.scipy.fft.dctn(cp.array(image[i,j,...].numpy()))
			gpuimg = cp.exp(1j*cp.angle(gpuimg)) * cp.maximum(0, (cp.abs(gpuimg) - lamda))
			image[i,j,...] = torch.tensor(cupyx.scipy.fft.idctn(gpuimg).get())
	
	return image


def reconstruct_cg(diagonals, rhs, smaps, nenc, iter = 50, images=None, plot=False):
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzLinop(vec_size, smaps, nenc, diagonals)

	scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=12).run()).to(torch.float32)

	scaled_linop = TorchScaleLinop(toep_linop, scaling)

	tcg = TorchCG(scaled_linop, scaling * rhs, images, max_iter=iter)
	
	prox = lambda x: dct_prox(x, 0.1)

	return tcg.run_with_prox(prox, plot)

def reconstruct_gd(diagonals, rhs, smaps, nenc, iter = 50, images=None, plot=False):
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzLinop(vec_size, smaps, nenc, diagonals)

	scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=12).run()).to(torch.float32)

	scaled_linop = TorchScaleLinop(toep_linop, scaling)

	gradf = lambda x: scaled_linop(x) - rhs

	tgd = TorchGD(gradf, images, alpha=1.0, accelerate=True, max_iter=iter)

	prox = lambda x: dct_prox(x, 0.1)

	return tgd.run_with_prox(prox, plot)


coords, kdatas, nframes, nenc = simri.load_coords_kdatas('D:/4DRecon/dat/dat2')
smaps = torch.tensor(simri.load_smaps('D:/4DRecon/dat/dat2'))

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

im_size = (smaps.shape[1],smaps.shape[2],smaps.shape[3])

use_weights = True

vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])
images = torch.zeros(vec_size, dtype=torch.complex64)

#print('View smaps')
#pu.image_4d(np.abs(smaps))

#print('Beginning weighted load')
#diagonals, rhs = load_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=True, root=0)

#print('Beginning weighted reconstruct')
#images = reconstruct_cg(diagonals, rhs, smaps, nenc, iter=50, images=images, plot=True)

#pu.image_5d(np.abs(images))

if True:
	print('Beginning weighted load')
	diagonals, rhs = load_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=0)
	print('Beginning weighted reconstruct')
	images = reconstruct_cg(diagonals, rhs, smaps, nenc, iter=500, images=images, plot=False)

	pu.image_5d(np.abs(images))

if False:
	print('Beginning unweighted load')
	diagonals, rhs = load_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=0)
	print('Beginning unweighted reconstruct')
	images = reconstruct_gd(diagonals, rhs, smaps, nenc, iter=150, images=images, plot=False)

	pu.image_5d(np.abs(images))


with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed_weighted.h5', "w") as f:
	f.create_dataset('images', data=images)





