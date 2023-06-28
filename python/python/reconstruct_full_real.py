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

#shift=shift, 
#		crop_factor=1.0, prefovkmul=1.0, postfovkmul=1.0, plot=True

def run_framed(images_full, smaps, coord, kdata, gating, nframes, shift=(0.0, 0.0, 0.0), 
		crop_factor=1.5, prefovkmul=1.0,  postfovkmul=1.0, store=True, plot=False):
	nenc = 5
	im_size = (images_full.shape[2], images_full.shape[3], images_full.shape[4])
	print('Gating')
	coord_vec, kdata_vec, weights_vec, gates = ru.gate(coord, kdata, None, gating, nframes)

	if plot:
		pu.plot_gating(gating, gates)

	del coord, kdata, gating
	print('Crop kspace')
	coord_vec, kdata_vec, weights_vec = ru.crop_kspace(coord_vec, kdata_vec, weights_vec, im_size, 
		crop_factor=crop_factor, prefovkmul=prefovkmul, postfovkmul=postfovkmul)
	print('Translate')
	if shift != (0.0, 0.0, 0.0):
		data_vec = ru.translate(coord_vec, kdata_vec, shift)

	images = torch.empty(nframes, nenc, im_size[0], im_size[1], 
		      im_size[2], dtype=torch.complex64)
	
	for i in range(nframes):
		for j in range(nenc):
			images[i,j,...] = images_full[j,...]
	torch.cuda.empty_cache()
	print('Reconstructing frames')
	images = ru.reconstruct_frames(images, smaps, coord_vec, kdata_vec, nenc, nframes, iter=5, lamda=0.0, plot=plot)
	
	if plot:
		pu.image_nd(images.numpy())

	if store:
		print('Storing frame reconstructed')
		with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_real.h5', "w") as f:
			f.create_dataset('images', data=images.numpy())

	return images

def run_full(im_size, store=False, shift=(0.0, 0.0, 0.0), crop_factor=1.5, prefovkmul=1.0, postfovkmul=1.0, plot=False):
	print('Loading coords, kdata, weights')
	smaps, coord, kdata, weights, gating = ru.load_real()

	smaps = torch.permute(smaps, (0,3,2,1))
	#if plot:
	#	pu.image_nd(torch.sum(smaps, dim=0).numpy())

	print('Loading full')
	coord_vec, kdata_vec, weights_vec = ru.load_full_real(coord, kdata, weights)
	#del coord, kdata, weights
	print('Crop kspace')

	coord_vec, kdata_vec, weights_vec = ru.crop_kspace(
		coord_vec, kdata_vec, weights_vec, im_size, crop_factor=crop_factor, 
		prefovkmul=prefovkmul, postfovkmul=postfovkmul)
	
	kdatapoints = 0
	for kdatai in kdata_vec:
		kdatapoints += kdatai.shape[2]
	print(kdatapoints)

	print('Translate')
	if shift != (0.0, 0.0, 0.0):
		kdata_vec = ru.translate(coord_vec, kdata_vec, shift)

	center_weights = ru.center_weights(im_size, 0.1, coord_vec[0], weights_vec[0])

	#print('Direct reconstruction without Smaps')
	#images = ru.direct_nufft_reconstruct_encs(None, coord_vec, kdata_vec, weights_vec, im_size)
	#if plot:
	#	pu.image_nd(images.numpy())

	print('Direct reconstruction with Smaps')
	#images = ru.direct_nufft_reconstruct_encs(smaps, coord_vec, kdata_vec, weights_vec, im_size)

	#if plot:
	#	pu.image_nd(images.numpy())
	
	#images *= torch.mean(center_weights)
	#images *= torch.mean(center_weights)
	torch.cuda.empty_cache()
	images = ru.reconstruct_gd_full(smaps, coord_vec, kdata_vec, weights_vec,
			iter=3, lamda=0.0005, images=None, plot=False)
	torch.cuda.empty_cache()
	images = ru.reconstruct_gd_full(smaps, coord_vec, kdata_vec, None,
			iter=4, lamda=0.0001, images=images, plot=False)

	del center_weights, coord_vec, kdata_vec, weights_vec

	if plot:
		pu.image_nd(images.numpy())

	if store:
		print('Storing fully reconstructed')
		with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_real.h5', "w") as f:
			f.create_dataset('images', data=images.numpy())

	return images, smaps, coord, kdata, gating


if __name__ == "__main__":
	#shift = (0.0,0.0,0.0)
	shift = (-2*25.6, 0.0, 0.0)
	crop_factor=1.0
	prefovkmul=1.0
	postfovkmul=1.0

	images, smaps, coord, kdata, gating = run_full((256,256,256), shift=shift, 
		crop_factor=crop_factor, prefovkmul=prefovkmul, postfovkmul=postfovkmul, plot=False)
	images = run_framed(images, smaps, coord, kdata, gating, 15,
		shift=shift, crop_factor=crop_factor, prefovkmul=prefovkmul, postfovkmul=postfovkmul, plot=True)