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

def run():
	coords, kdatas, nframes, nenc = simri.load_coords_kdatas(dirpath='D:/4DRecon/dat/dat2')
	smaps = torch.tensor(simri.load_smaps('D:/4DRecon/dat/dat2'))

	im_size = (smaps.shape[1],smaps.shape[2],smaps.shape[3])

	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	true_images: torch.Tensor
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_20f_cropped_interpolated.h5', "r") as f:
		true_images = torch.tensor(f['images'][()])
	true_norm = torch.norm(true_images)
	true_full_images = torch.mean(true_images, dim=0).view((nenc, 1, im_size[0], im_size[1], im_size[2]))
	true_full_norm = torch.norm(true_full_images)

	#pu.image_nd(true_images.numpy())

	#true_mvel: np.array
	#with h5py.File('D:\\4DRecon\\dat\\dat2\\images_mvel_20f_cropped_interpolated.h5', "r") as f:
	#	true_mvel = f['images'][()]

	#pu.image_nd(true_mvel.numpy())
	#pu.image_nd(np.sqrt(true_mvel[:,1,...]**2 + true_mvel[:,2,...]**2 + true_mvel[:,3,...]**2))


	coord_vec = []
	kdata_vec = []
	weights_vec = []
	if True:
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

		for coord in coord_vec:
			weights = tkbn.calc_density_compensation_function(ktraj=coord.to(torch.device('cuda:0')), im_size=im_size).cpu()
			weights_vec.append(weights.squeeze(0))

	relerr = []
	def full_error_callback(images):
		relerr.append(torch.norm(images - true_full_images) / true_full_norm)
		print('True RelErr: ', relerr[-1])

	images = ru.reconstruct_gd_full(smaps, coord_vec, kdata_vec, weights_vec, iter=40, lamda=0.0, 
				plot=False, callback=full_error_callback)

	#pu.image_nd(images.numpy())

	relerr = []
	images_full = ru.reconstruct_gd_full(smaps, coord_vec, kdata_vec, None, iter=10, images=images, lamda=0.0, 
					plot=False, callback=full_error_callback)

	#pu.image_nd(images_full.numpy())

	coord_vec = []
	kdata_vec = []
	if True:
		for frame in range(nframes):
			for encode in range(nenc):
				coord_vec.append(torch.tensor(coords[frame][encode]))
				kdata_vec.append(torch.tensor(kdatas[frame][encode]))

	images = torch.empty(nframes, nenc, im_size[0], im_size[1], 
				im_size[2], dtype=torch.complex64)
	if True:
		for frame in range(nframes):
			for enc in range(nenc):
				images[frame,enc,...] = images_full[enc,...]


	relerr = []
	def framed_error_callback(images):
		svt_shape = (nframes, nenc, im_size[0], im_size[1], im_size[2])
		relerr.append(torch.norm(images.view(svt_shape) - true_images) / true_norm)
		print('True RelErr: ', relerr[-1])


	images = ru.reconstruct_frames(images, smaps, coord_vec, kdata_vec, nenc, 
		nframes, stepmul=1.5, rand_iter=0, iter=40, nrestarts=1, singular_index=5, lamda=0.1,
		plot=False, callback=framed_error_callback)

	pu.image_nd(images.numpy())

	with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_simulated.h5', "w") as f:
		f.create_dataset('images', data=images)


with torch.no_grad():
	run()


