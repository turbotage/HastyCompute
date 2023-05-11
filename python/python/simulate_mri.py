import numpy as np
import h5py

import plot_utility as pu
import image_creation as ic

import torch

def simulate():

	create_crop_image = False
	load_crop_image = False

	create_enc_image = False
	load_enc_image = False

	create_nufft_of_enced_image = False
	load_nufft_of_neced_image = False

	img = np.array([0])
	img_mag = np.array([0])
	smaps = np.array([0])

	if create_crop_image:
		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f.h5', "r") as f:
			img = f['images'][()]
			img = np.transpose(img, axes=(4,3,2,1,0))

		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_mag.h5', "r") as f:
			img_mag = f['images'][()]
			img_mag = np.transpose(img_mag, (2,1,0))

		if True:
			smap_list = []
			with h5py.File('D:\\4DRecon\\dat\\dat2\\SenseMapsCpp.h5', "r") as hf:
				maps_base = hf['Maps']
				maps_key_base = 'SenseMaps_'
				for i in range(len(list(maps_base))):
					smap = maps_base[maps_key_base + str(i)][()]
					smap = smap['real'] + 1j*smap['imag']
					smap_list.append(smap)

			smaps = np.stack(smap_list, axis=0)

		nlen = 150
		nx = 32
		ny = 32
		nz = 45
		crop_box = [(nx,nx+nlen),(ny,ny+nlen),(nz,nz+nlen)]
		new_img = ic.crop_5d_3d(img, crop_box).astype(np.float32)
		new_img_mag = ic.crop_3d_3d(img_mag, crop_box).astype(np.float32)
		new_smaps = ic.crop_4d_3d(smaps, crop_box).astype(np.complex64)

		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_cropped.h5', "w") as f:
			f.create_dataset('images', data=new_img)

		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_mag_cropped.h5', "w") as f:
			f.create_dataset('images', data=new_img_mag)

		with h5py.File('D:\\4DRecon\\dat\\dat2\\SenseMapsCpp_cropped.h5', "w") as f:
			f.create_dataset('Maps', data=new_smaps)

		img = new_img
		img_mag = new_img_mag
		smaps = new_smaps
		print('\nCreated cropped images\n')
	elif load_crop_image:
		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_cropped.h5', "r") as f:
			img = f['images'][()]
		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_mag_cropped.h5', "r") as f:
			img_mag = f['images'][()]
		with h5py.File('D:\\4DRecon\\dat\\dat2\\SenseMapsCpp_cropped.h5', "r") as f:
			smaps = f['Maps'][()]
		print('\nLoaded cropped images\n')


	img_enc = np.array([0])

	if create_enc_image:
		old_img = img

		img[:,0,...] *= (2.0 * np.expand_dims(img_mag, axis=0) / np.max(img_mag, axis=(0,1,2)))

		img = np.real(ic.interpolate_images(img, 15).astype(np.complex64)).astype(np.float32)

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

		imenc = (np.expand_dims(img[:,0], axis=1) * (np.cos(imenc) + 1j*np.sin(imenc))).astype(np.complex64)

		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_15f_cropped_interpolated.h5', "w") as f:
			f.create_dataset('images', data=imenc)

		img_enc = imenc

		print('\nCreated encoded images\n')
	elif load_enc_image:
		with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_15f_cropped_interpolated.h5', "r") as f:
			img_enc = f['images'][()]
		print('\nLoaded encoded images\n')

	coords = np.array([0])
	kdatas = np.array([0])

	if create_nufft_of_enced_image:
		dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
		torch.ops.load_library(dll_path)

		hasty_py = torch.ops.HastyPyInterface

		nfreq = 400000

		nframes = img_enc.shape[0]
		nenc = img_enc.shape[1]
		nsmaps = smaps.shape[0]

		coords = np.empty((nframes, nenc,3,nfreq), dtype=np.float32)
		kdatas = np.empty((nframes,nenc,nsmaps,nfreq), dtype=np.complex64)

		for frame in range(nframes):
			for encode in range(nenc):
				coord = -np.pi + 2*np.pi*np.random.rand(3,nfreq).astype(np.float32)
				coords[frame,encode,...] = coord
				for smap in range(nsmaps):
					coiled_image = img_enc[frame,encode,...] * smaps[smap,...]

					coord_torch = torch.tensor(coord).to(torch.device('cuda:0'))
					coild_image_torch = torch.tensor(coiled_image).to(torch.device('cuda:0')).unsqueeze(0)

					kdata = hasty_py.nufft2(coord_torch,coild_image_torch)
					kdatas[frame,encode,smap,...] = kdata.cpu().numpy()

		with h5py.File('D:\\4DRecon\\dat\\dat2\\simulated_coords_kdatas.h5', "w") as f:
			f.create_dataset('coords', data=coords)
			f.create_dataset('kdatas', data=kdatas)

		print('\nCreated coords and kdatas\n')
	elif load_nufft_of_neced_image:
		with h5py.File('D:\\4DRecon\\dat\\dat2\\simulated_coords_kdatas.h5', "r") as f:
			coords = f['coords'][()]
			kdatas = f['kdatas'][()]
		print('\nLoaded coords and kdatas\n')

def load_coords_kdatas():
	coords = np.array([])
	kdatas = np.array([])
	with h5py.File('D:\\4DRecon\\dat\\dat2\\simulated_coords_kdatas.h5', "r") as f:
		coords = f['coords'][()]
		kdatas = f['kdatas'][()]
	return (coords, kdatas)

def load_smaps():
	smaps = np.array([])
	with h5py.File('D:\\4DRecon\\dat\\dat2\\SenseMapsCpp_cropped.h5', "r") as f:
		smaps = f['Maps'][()]
	return smaps
