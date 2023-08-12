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

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

def kspace_precond(smaps, coord, weights=None):
	mps_shape = list(smaps.shape)
	img_shape = mps_shape[1:]
	expanded_shape = [d*2 for d in img_shape]
	ndim = len(img_shape)
	cudev = torch.cuda('cuda:0')

	ones = torch.ones((1,coord.shape[1]), dtype=torch.complex64).to(cudev)
	if weights is not None:
		ones *= torch.sqrt(weights)

	coord_cu = coord.to(cudev)
	psf = hasty_sense.nufft1(coord_cu, ones, expanded_shape)

	smaps_cu = smaps.to(cudev)

	scale = math.prod(list(expanded_shape)) ** 1.5 / math.prod(list(img_shape))

	pinv = []
	for i in range(smaps_cu.shape[0]):
		mpsi = smaps_cu[i,...]
		mpsi_norm = torch.norm(mpsi) ** 2

		fourier_corr = 0.0
		for j in range(smaps_cu.shape[0]):
			mpsj = smaps_cu[j,...]
			fourier_corr += torch.fft.fftn(mpsi * mpsj.conj(), s=expanded_shape) ** 2

		corr = torch.fft.ifftn(fourier_corr)
		corr *= psf

		pinvi = hasty_sense.nufft2(coord, corr)

		if weights is not None:
			pinvi *= torch.sqrt(weights)

		pinv.append(pinvi * scale / mpsi_norm)

	pinv = torch.abs(torch.stack(pinv, dim=0))
	pinv[pinv == 0] = 1
	return 1 / pinv

def circulant_precond(smaps, coord, weights=None):
	mps_shape = list(smaps.shape)
	img_shape = mps_shape[1:]
	expanded_shape = [d*2 for d in img_shape]
	ndim = len(img_shape)
	cudev = torch.device('cuda:0')

	weigths_cu: torch.Tensor
	if weights is not None:
		weights_cu = weights.to(cudev)

	ones_cu = torch.ones((1,coord.shape[1]), dtype=torch.complex64).to(cudev)
	if weights is not None:
		ones_cu *= torch.sqrt(weights_cu)

	coord_cu = coord.to(cudev)
	psf = hasty_sense.nufft1(coord_cu, ones_cu, [1] + list(expanded_shape)).squeeze(0)
	del coord_cu, ones_cu
	torch.cuda.empty_cache()

	smaps_cu = smaps.to(cudev)

	scale = math.prod(list(expanded_shape)) ** 1.5 / math.prod(list(img_shape)) ** 2

	idx = (slice(None, None, 2),) * ndim
	pinv = torch.zeros(img_shape, dtype=torch.complex64).to(cudev)
	for i in range(smaps.shape[0]):
		mpsi = smaps_cu[i,...]
		fourier_corr = torch.abs(torch.fft.fftn(mpsi.conj(), s=expanded_shape)) ** 2
		corr = torch.fft.ifftn(fourier_corr)
		corr *= psf
		pinvi = torch.fft.fftn(corr)
		pinvi = pinvi[idx]
		pinvi *= scale
		#if weights is not None:
			#pinvi *= torch.sqrt(weights_cu)
		
		pinv += pinvi

	pinv[pinv == 0] = 1
	ret = (1 / pinv).cpu()
	del pinv, psf, smaps_cu
	if weights is not None:
		del weigths_cu
	torch.cuda.empty_cache()
	return ret

def test():
	N1 = 256
	N2 = 256
	N3 = 256

	smaps = torch.rand((32,N1,N2,N3), dtype=torch.complex64)
	coord = torch.rand((3,10000000), dtype=torch.float32)

	P = circulant_precond(smaps, coord)

	print(P.shape)

