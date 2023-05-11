import numpy as np
import h5py

import plot_utility as pu
import image_creation as ic

import torch

crop=False
img = np.array([0])
img_mag = np.array([0])

def plot_3view_maxip(img):
	cd = ic.get_CD(img)
	pu.maxip_4d(cd,axis=1)
	pu.maxip_4d(cd,axis=2)
	pu.maxip_4d(cd,axis=3)


if crop:
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f.h5', "r") as f:
		img = f['images'][()]
		img = np.transpose(img, axes=(4,3,2,1,0))

	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_mag.h5', "r") as f:
		img_mag = f['images'][()]
		img_mag = np.transpose(img_mag, (2,1,0))

	nlen = 150
	nx = 32
	ny = 32
	nz = 45
	crop_box = [(nx,nx+nlen),(ny,ny+nlen),(nz,nz+nlen)]
	new_img = ic.crop_5d_3d(img, crop_box)
	new_img_mag = ic.crop_3d_3d(img_mag, crop_box)

	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_cropped.h5', "w") as f:
		f.create_dataset('images', data=new_img)

	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_mag_cropped.h5', "w") as f:
		f.create_dataset('images', data=new_img_mag)

	img = new_img
	img_mag = new_img_mag
else:
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_cropped.h5', "r") as f:
		img = f['images'][()]
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_mag_cropped.h5', "r") as f:
		img_mag = f['images'][()]


img_enc = np.array([0])

create_enc_image = True
if create_enc_image:
	old_img = img

	img[:,0,...] *= (2.0 * np.expand_dims(img_mag, axis=0) / np.max(img_mag, axis=(0,1,2)))

	img = ic.interpolate_images(img, 15)

	#plot_3view_maxip(img)

	v_enc = 1100
	A = (1.0/v_enc) * np.array(
		[
		[ 0,  0,  0],
		[-1, -1, -1],
		[ 1,  1, -1],
		[ 1, -1,  1],
		[-1,  1,  1]
		], dtype=np.float32)

	imvel = np.expand_dims(np.transpose(img[:,1:], axes=(2,3,4,0,1)), axis=-1)

	imenc = (A @ imvel).squeeze(-1)
	imenc = np.transpose(imenc, axes=(3,4,0,1,2))

	imenc = np.expand_dims(img[:,0], axis=1) * (np.cos(imenc) + 1j*np.sin(imenc))

	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_15f_cropped_interpolated.h5', "w") as f:
		f.create_dataset('images', data=imenc)

	img_enc = imenc
else:
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_15f_cropped_interpolated.h5', "w") as f:
		img_enc = f['images'][()]


nufft_of_enced_image=True
if nufft_of_enced_image:
	dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
	torch.ops.load_library(dll_path)

	hasty_py = torch.ops.HastyPyInterface
	
	

