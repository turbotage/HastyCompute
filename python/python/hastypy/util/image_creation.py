import numpy as np
from scipy.interpolate import interp1d
import scipy as sp

import hastypy.util.plot_utility as pu

import nibabel as nib
import math
import matplotlib.pyplot as plt

from numba import njit

def create_spoke(samp_per_spoke, method='PCVIPR', noise=0.005, scale_factor=1.0, crop_factor=1.0, xangle=None, zangle=None):
	if method == 'PCVIPR':
		def rx(angle):
			return np.array([
				[1.0, 	0.0, 	0.0],
				[0.0, np.cos(angle), -np.sin(angle)],
				[0.0, np.sin(angle), np.cos(angle)]
				]).astype(np.float32)
		
		def rz(angle):
			return np.array([
				[np.cos(angle), -np.sin(angle), 0.0],
				[np.sin(angle), np.cos(angle), 0.0],
				[0.0, 	0.0, 	1.0]
			]).astype(np.float32)

		spoke = np.zeros((3,samp_per_spoke), dtype=np.float32)
		spoke[0,:] = np.random.normal(scale=noise, size=samp_per_spoke)
		spoke[1,:] = np.random.normal(scale=noise, size=samp_per_spoke)
		spoke[2,:] = np.pi*np.linspace(-(1.0/3.0), 1.0, samp_per_spoke).astype(np.float32)
		spoke[2,:] += np.random.normal(scale=noise, size=samp_per_spoke)

		if xangle is None:
			xangle = np.pi*np.random.rand(1).astype(np.float32).item()
		if zangle is None:
			zangle = scale_factor*2*np.pi*np.random.rand(1).astype(np.float32).item()

		spoke = rz(zangle) @ rx(xangle) @ spoke
		
		if crop_factor != 1.0:
			spoke *= crop_factor

			xidx = np.abs(spoke[0,:]) < np.pi
			yidx = np.abs(spoke[1,:]) < np.pi
			zidx = np.abs(spoke[2,:]) < np.pi

			idx = np.logical_and(np.logical_and(xidx, yidx), zidx)

			spoke = spoke[:,idx]

		return spoke.astype(np.float32)
	else:
		raise RuntimeError("Not a valid method")

def create_coords(nspokes, samp_per_spoke, imsize, method='MidRandom', plot=False, crop_factor=1.0, xangles=None, zangles=None):
	nfreq = nspokes * samp_per_spoke

	if method == 'MidRandom':
		coord_vec = []
		L = np.pi / 8
		coord_vec.append(-L + 2*L*np.random.rand(3,nfreq // 4).astype(np.float32))
		L = np.pi / 4
		coord_vec.append(-L + 2*L*np.random.rand(3,nfreq // 4).astype(np.float32))
		L = np.pi / 2
		coord_vec.append(-L + 2*L*np.random.rand(3,nfreq // 4).astype(np.float32))
		L = np.pi
		coord_vec.append(-L + 2*L*np.random.rand(3,nfreq // 4).astype(np.float32))

		coord = np.concatenate(coord_vec, axis=1)

		if plot:
			pu.scatter_3d(coord)

		return coord
	elif method == 'PCVIPR':
		coord_vec = []

		for i in range(nspokes):
			if xangles is None and zangles is None:
				coord_vec.append(create_spoke(samp_per_spoke, method='PCVIPR', crop_factor=crop_factor))
			elif xangles is None and zangles is not None:
				coord_vec.append(create_spoke(samp_per_spoke, method='PCVIPR', crop_factor=crop_factor, zangle=zangles[i].item()))
			elif xangles is not None and zangles is None:
				coord_vec.append(create_spoke(samp_per_spoke, method='PCVIPR', crop_factor=crop_factor, xangle=xangles[i].item()))
			else:
				coord_vec.append(create_spoke(samp_per_spoke, method='PCVIPR', crop_factor=crop_factor, xangle=xangles[i].item(), zangle=zangles[i].item()))

		coord = np.concatenate(coord_vec, axis=1)

		if False:
			nfreq = nspokes * samp_per_spoke
			nsamp = coord.shape[1]
			nsamp_to_add = np.random.rand(3,nfreq-nsamp).astype(np.float32)

			coord = np.concatenate([coord, nsamp_to_add], axis=1)

		if plot:
			pu.scatter_3d(coord)

		return coord.astype(np.float32)	
	elif method == 'SubsampledCartesian':

		@njit
		def fill_coordinates():
			nx = imsize[0]
			ny = imsize[1]
			nz = imsize[2]
			nfreq = min(nspokes * samp_per_spoke, nx*ny*nz)
			choises = np.sort(np.random.permutation(nx*ny*nz)[:nfreq])
			coord = np.empty((3,nfreq), dtype=np.float32)
			l = 0
			c = 0
			for i in range(nx):
				for j in range(ny):
					for k in range(nz):
						if l != choises[c]:
							l += 1
							continue
						coord[0,c] = -np.pi + 2*np.pi*i/nx
						coord[1,c] = -np.pi + 2*np.pi*j/ny
						coord[2,c] = -np.pi + 2*np.pi*k/nz
						c += 1
						l += 1
			return coord.astype(np.float32)
		
		return fill_coordinates()	
	elif method == 'FullCartesian':
		@njit
		def fill_coordinates():
			nx = imsize[0]
			ny = imsize[1]
			nz = imsize[2]
			coord = np.empty((3,nx*ny*nz), dtype=np.float32)
			l = 0
			for i in range(nx):
				for j in range(ny):
					for k in range(nz):
						coord[0,l] = -np.pi + 2*np.pi*i/nx
						coord[1,l] = -np.pi + 2*np.pi*j/ny
						coord[2,l] = -np.pi + 2*np.pi*k/nz
						l += 1
			return coord.astype(np.float32)
		
		return fill_coordinates()
	else:
		raise RuntimeError("Not a valid method")
		
def get_CD(img, venc=1100, plot_cd=False, plot_mip=False):
	m = img[:,0,:,:,:].astype(np.float32)
	vx = img[:,1,:,:,:].astype(np.float32)
	vy = img[:,2,:,:,:].astype(np.float32)
	vz = img[:,3,:,:,:].astype(np.float32)

	cd = (m * np.sin(np.pi * np.minimum(np.sqrt(vx*vx + vy*vy + vz*vz), 0.5*venc) / venc)).astype(np.float32)

	if plot_cd:
		pu.image_4d(cd)
	if plot_mip:
		pu.maxip_4d(cd)

	return cd

def crop_5d_3d(img, box):
	new_img = img[:,:,box[0][0]:box[0][1],box[1][0]:box[1][1],box[2][0]:box[2][1]]
	return new_img

def crop_4d_3d(img, box):
	new_img = img[:,box[0][0]:box[0][1],box[1][0]:box[1][1],box[2][0]:box[2][1]]
	return new_img

def crop_3d_3d(img, box):
	new_img = img[box[0][0]:box[0][1],box[1][0]:box[1][1],box[2][0]:box[2][1]]
	return new_img

def plot_3view_maxip(img):
	cd = get_CD(img)
	pu.maxip_4d(cd,axis=1)
	pu.maxip_4d(cd,axis=2)
	pu.maxip_4d(cd,axis=3)

def numpy_to_nifti(img, file):
	nimg = nib.Nifti1Image(img, affine=np.eye(4))
	nib.save(nimg, file)

#images_out_shape = (nbin,1+3,nx,ny,nz)
# 1 + 3 a magnitude image and 3 velocities
def interpolate_images(images, num_img: int):
	n_img = images.shape[0]

	x = np.arange(0,n_img)

	f = interp1d(x, images, kind='linear',axis=0)

	c = np.arange(0,n_img-1, (n_img-1)/num_img)

	return f(c)

def convolve_3d_3x3x3(img, factor = 3, mode='same'):
	kernel = np.ones((3,3,3), dtype=np.float32) / (27 * factor)
	kernel[1,1,1] += (factor-1) / factor

	return sp.signal.convolve(img, kernel, mode=mode)

def convolve_4d_3x3x3(img, factor = 3, mode='same'):
	kernel = np.ones((3,3,3), dtype=np.float32) / (27 * factor)
	kernel[1,1,1] += (factor-1) / factor

	out = np.empty_like(img)

	for i in range(img.shape[0]):
		out[i,...] = sp.signal.convolve(img[i,...], kernel, mode=mode)

	return out

def convolve_5d_3x3x3(img, factor = 3, mode='same'):
	kernel = np.ones((3,3,3), dtype=np.float32) / (27 * factor)
	kernel[1,1,1] += (factor-1) / factor

	out = np.empty_like(img)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			out[i,j,...] = sp.signal.convolve(img[i,j,...], kernel, mode=mode)
		
	return out

#coord = create_coords(500, 50, method='PCVIPR', plot=True, crop_factor=1.5)
#print(coord.shape)
