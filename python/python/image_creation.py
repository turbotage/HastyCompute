import numpy as np
from scipy.interpolate import interp1d
import scipy as sp

import plot_utility as pu

import nibabel as nib


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
