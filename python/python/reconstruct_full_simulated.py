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

coords, kdatas, nframes, nenc = simri.load_coords_kdatas(dirpath='D:/4DRecon/dat/dat2')
smaps = torch.tensor(simri.load_smaps('D:/4DRecon/dat/dat2'))

im_size = (smaps.shape[1],smaps.shape[2],smaps.shape[3])

vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])
#images = torch.zeros(vec_size, dtype=torch.complex64)

#smaps = torch.permute(smaps, (0,1,2,1))

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

if False:
	print('Beginning unweighted load')
	diagonals, rhs = ru.load_simulated_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=1)
	print('Beginning unweighted reconstruct')
	images = ru.reconstruct_gd_full(diagonals, rhs, smaps, nenc, iter=100, lamda=0.0, images=images, plot=True)

	pu.image_5d(np.abs(images))
	pu.image_4d(np.mean(np.abs(images.numpy()), axis=0))

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

weights_vec = []
for coord in coord_vec:
	weights = tkbn.calc_density_compensation_function(ktraj=coord.to(torch.device('cuda:0')), im_size=im_size).cpu()
	weights_vec.append(weights.squeeze(0))

images = ru.reconstruct_gd_full(smaps, coord_vec, kdata_vec, weights_vec, iter=3, lamda=0.0, plot=False)

images_full = ru.reconstruct_gd_full(smaps, coord_vec, kdata_vec, None, iter=4, images=images, lamda=0.0, plot=True)

pu.image_nd(images)

coord_vec = []
kdata_vec = []
for i in range(nframes):
	for j in range(nenc):
		coord_vec.append(torch.tensor(coords[frame][encode]))
		kdata_vec.append(torch.tensor(kdatas[frame][encode]))

images = torch.empty(nframes, nenc, im_size[0], im_size[1], 
		      im_size[2], dtype=torch.complex64)

for i in range(nframes):
		for j in range(nenc):
			images[i,j,...] = images_full[j,...]

images = ru.reconstruct_frames(images, smaps, coord_vec, kdata_vec, nenc, 
			nframes, stepmul=1.0, rand_iter=0, iter=25, singular_index=2, lamda=0.1)

pu.image_nd(images)

with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed.h5', "w") as f:
	f.create_dataset('images', data=images)





