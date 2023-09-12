import torch
import numpy as np
import math

import cupy as cp
import cupyx

import hastypy.util.plot_utility as pu
import hastypy.util.simulate_mri as simri

import hastypy.base.util as util

import hastypy.util.image_creation as ic
from hastypy.base.nufft import NufftT, NufftAdjointT, NufftNormalT

def kspace_precond(smaps, coord, weights=None):
	mps_shape = list(smaps.shape)
	img_shape = mps_shape[1:]
	expanded_shape = tuple([d*2 for d in img_shape])
	ndim = len(img_shape)
	cudev = torch.device('cuda:0')

	ones = torch.ones((1,coord.shape[1]), dtype=torch.complex64).to(cudev)
	if weights is not None:
		ones *= torch.sqrt(weights)

	coord_cu = coord.to(cudev)
	psf = NufftAdjointT(coord_cu, expanded_shape).apply(ones).squeeze(0)

	smaps_cu = smaps.to(cudev)

	scale = math.prod(list(expanded_shape)) ** 1.5 / math.prod(list(img_shape))

	pinv = []
	for i in range(smaps_cu.shape[0]):
		mpsi = smaps_cu[i,...]
		mpsi_norm = torch.norm(mpsi) ** 2

		fourier_corr = 0.0
		for j in range(smaps_cu.shape[0]):
			mpsj = smaps_cu[j,...]
			#fourier_corr += torch.fft.fftn(mpsi * mpsj.conj(), s=expanded_shape) ** 2
			fourier_corr += util.fftn(mpsi * mpsj.conj(), oshape=expanded_shape, center=True)

		corr = util.ifftn(fourier_corr, center=True)
		corr *= psf

		pinvi = NufftT(coord_cu, corr.shape).apply(corr.unsqueeze(0))


		if weights is not None:
			pinvi *= torch.sqrt(weights)

		pinv.append((pinvi * scale / mpsi_norm).squeeze(0))

	pinv = torch.abs(torch.stack(pinv, dim=0)).unsqueeze(0)
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
	psf = NufftAdjointT(coord_cu, expanded_shape).apply(ones_cu).squeeze(0)
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

def pipe_menon_dcf(coord, im_size, max_iter=30):
	cudev = torch.device('cuda:0')
	N = NufftT(coord.to(cudev), im_size)
	NH = NufftAdjointT(coord.to(cudev), im_size)
	w = torch.ones((1,coord.shape[1]), dtype=torch.complex64, device=cudev)
	for i in range(max_iter):
		NHNw = N(NH(w))
		w /= torch.abs(NHNw)
	return w


def test():
	N1 = 64
	N2 = 64
	N3 = 64

	smaps = torch.rand((32,N1,N2,N3), dtype=torch.complex64)
	coord = torch.rand((3,100000), dtype=torch.float32)

	P = kspace_precond(smaps, coord)

	print(P.shape)
