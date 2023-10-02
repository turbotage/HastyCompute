


import cupy as cp
import numpy as np
#from numba import jit

import math

from hastypy.opt.cuda import cuda_program as cudap
from hastypy.opt.cuda.cuda_program import CudaFunction, CudaTensor
from hastypy.opt.cuda.cuda_program import CudaTensorChecking as ctc

import hastypy.opt.cuda.sigpy.fourier_linops as fulinops
import hastypy.opt.cuda.sigpy.fourier as fourier
import hastypy.opt.cuda.sigpy.linop as linop

import h5py

def full_resampling(filename_images, filename_smaps, filename_coords, coorner, downsize_factor):
	smaps = np.array([0])
	images = np.array([0])
	coords = np.array([0])
 

	print('Reading images file')
	with h5py.File(filename_images, 'r') as h5_img:
		images = h5_img['images'][()]

	with h5py.File(filename_smaps, 'r') as h5_smap:
		ncoils = len(h5_smap['Maps'])
		shape = h5_smap['Maps']['SenseMaps_0'][()].shape
		smaps = np.empty((ncoils, *shape), dtype=cp.complex64)

		for i in range(ncoils):
			smap = h5_smap['Maps']['SenseMaps_' + str(i)][()]
			smaps[i,:,:,:] = (smap['real'] + 1j*smap['imag']).astype(np.complex64)

	return (images, smaps, coords)




def image_downsizeing(images, smaps, corner, downsize: int):
	shapei = list(images[0,0,:,:,:].shape)

	shapei[0] //= downsize
	shapei[1] //= downsize
	shapei[2] //= downsize

	imagesd = images[:,:,corner[0]+shapei[0], corner[1]+shapei[1], corner[2]+shapei[2]]

	smapsd = smaps[:,:,corner[0]+shapei[0], corner[1]+shapei[1], corner[2]+shapei[2]]

	return (imagesd, smapsd)


#images_out_shape = (nbin,1+3,nx,ny,nz)
# 1 + 3 a magnitude image and 3 velocities
def interpolate_images(images, images_per_interval: int):
	n_img = images.shape[0]

	dt = 1 / (images_per_interval + 1)

	ntot = n_img + (n_img - 1)*images_per_interval
	
	images_out = np.empty((ntot, *images.shape[1:]), dtype=np.float32)

	k = 0
	for i in range(n_img):
		for j in range(images_per_interval):
			images_out[k] = images[i]*(1 - j*dt) + images[i+1]*j*dt

	return images_out

# linear_coords_shape = (nencs, 3, nspokes, nsamps_per_spoke)
def bin_gate_coords(linear_coords, binidx, nspokes_lookup, verbose=False):

	nbin = nspokes_lookup.shape[0]
	nencs = linear_coords.shape[0]
	nspokes = linear_coords.shape[2]
	nsamp_per_spoke = linear_coords.shape[3]

	nstart = np.zeros((nbin), dtype=np.int32)
	coords = [np.array([0])]*nbin
	for i in nbin:
		coords[i] = np.empty(nencs,3,nspokes_lookup[i]*nsamp_per_spoke, dtype=np.float32)

	for i in range(nencs):
		if verbose:
			print('Binning encode: ' + str(i))
		for j in range(nspokes):
			bidx = binidx[j]
			start = nstart[bidx]
			end = start + nsamp_per_spoke
			nstart[bidx] = end

			spokefreq = linear_coords[i,:,j,:].squeeze()

			coords[bidx][i,:,start:end] = spokefreq

	return coords

# coords_shape = [(nencs,3,nspokes_in_bin * nsamp_per_spoke)]*nbin
# images_shape = 
def sample_kdata(coords, images, smaps, v_enc):
	nbin = len(coords)
	ncoils = smaps.shape[0]
	nencs = coords[0].shape[0]

	# Go from velocity and magnitude to phase and magnitude finaly complex image
	images_mag = images[:,0]
	images_vel = np.expand_dims(np.transpose(images[:,1:], axes=(2,3,4,0,1)), axis=-1)

	A = (1.0/v_enc) * np.array(
		[
		[ 0,  0,  0],
		[-1, -1, -1],
		[ 1,  1, -1],
		[ 1, -1,  1],
		[-1,  1,  1]
		], dtype=np.float32)

	images_enc = (A @ images_vel).squeeze(-1)

	images_enc = images_mag * (np.cos(images_enc) + 1j*np.sin(images_enc))

	# Preallocate kdata arrays
	kdatas = [np.array([0])]*nbin
	for i in range(nbin):
		kdatas[i] = np.empty(ncoils, nencs, 3, coords[i].shape[2], dtype=np.complex64)

	smaps = cp.array(smaps)
	for i in range(nbin):
		for j in range(nencs):
			image = cp.array(images[i, j])
			coord = cp.array(coords[i][j])
			for k in range(ncoils):

				kdatas[i][k,j] = fourier.nufft(image * smaps[k], coord, oversamp=2.0, width=16).get()

	return kdatas


# kdatas_shape [ncoil,nenc,n_spokes_in_bin * nsamples_per_spoke] * nbin
def unbin_gate_kdata(kdatas, binidx, nspokes_lookup, verbose=False):
	
	nbin = len(kdatas)
	ncoils = kdatas[0].shape[0]
	nencs = kdatas[0].shape[1]
	nspokes = binidx.shape[2]
	nsamp_per_spoke = kdatas[0].shape[2]


	nstart = np.zeros((nbin), dtype=np.int32)
	linear_kdata = np.empty((ncoils,nencs,nspokes,nsamp_per_spoke), dtype=np.complex64)

	for i in range(ncoils):
		if verbose:
			print('Unbinning coil: ' + str(i))
		for j in range(nencs):
			for k in range(nspokes):
				bidx = binidx[k]
				start = nstart[bidx]
				end = start + nsamp_per_spoke
				nstart[bidx] = end

				linear_kdata[i,j,k,:] = kdatas[bidx][i,j,start:end]

	return linear_kdata


	 

