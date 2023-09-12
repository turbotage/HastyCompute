import numpy as np
import h5py

import hastypy.util.plot_utility as pu
import hastypy.util.image_creation as ic

import math
import torch

import os
import fnmatch
import re

import hastypy.base.nufft as nufft

def rename_h5(name, appending):
	return name[:-3] + appending + '.h5'

def crop_image(dirpath, imagefile, create_crop_image=False, load_crop_image=False, just_plot=False, also_plot=False):
	map_joiner = lambda path: os.path.join(dirpath, path)

	img = np.array([0])
	img_mag = np.array([0])
	smaps = np.array([0])

	if create_crop_image and load_crop_image:
		raise RuntimeError('Can not both load and create crop images')

	if create_crop_image:
		print('Loading images')
		with h5py.File(map_joiner(imagefile), "r") as f:
			img = f['images'][()]
			img = np.transpose(img, axes=(4,3,2,1,0))

		print('Loading magnitude')
		with h5py.File(map_joiner(rename_h5(imagefile, '_mag')), "r") as f:
			img_mag = f['images'][()]
			img_mag = np.transpose(img_mag, (2,1,0))

		if True:
			smap_list = []
			print('Loading sense')
			with h5py.File(map_joiner('SenseMapsCpp.h5'), "r") as hf:
				maps_base = hf['Maps']
				maps_key_base = 'SenseMaps_'
				for i in range(len(list(maps_base))):
					smap = maps_base[maps_key_base + str(i)][()]
					smap = smap['real'] + 1j*smap['imag']
					smap_list.append(smap)

			smaps = np.stack(smap_list, axis=0)

		nlen = 64
		nx = 120
		ny = 50
		nz = 60 #45
		crop_box = [(nx,nx+nlen),(ny,ny+nlen),(nz,nz+nlen)]
		print('Cropping')
		new_img = ic.crop_5d_3d(img, crop_box).astype(np.float32)
		new_img_mag = ic.crop_3d_3d(img_mag, crop_box).astype(np.float32)
		new_smaps = ic.crop_4d_3d(smaps, crop_box).astype(np.complex64)

		if just_plot or also_plot:
			pu.image_nd(new_img)
			pu.image_nd(np.sqrt(new_img[:,1,...]**2 + new_img[:,2,...]**2 + new_img[:,3,...]**2))
			cd = ic.get_CD(new_img)
			pu.image_nd(cd)
			
			if just_plot:
				return (None, None, None)

		print('Writing cropped images')
		with h5py.File(map_joiner(rename_h5(imagefile, '_cropped')), "w") as f:
			f.create_dataset('images', data=new_img)

		print('Writing cropped mags')
		with h5py.File(map_joiner(rename_h5(imagefile, '_mag_cropped')), "w") as f:
			f.create_dataset('images', data=new_img_mag)

		print('Writing cropped sensemaps')
		with h5py.File(map_joiner('SenseMapsCpp_cropped.h5'), "w") as f:
			f.create_dataset('Maps', data=new_smaps)

		img = new_img
		img_mag = new_img_mag
		smaps = new_smaps
		print('\nCreated cropped images\n')
	elif load_crop_image:
		print('Loading cropped images')
		with h5py.File(map_joiner(rename_h5(imagefile, '_cropped')), "r") as f:
			img = f['images'][()]
		print('Loading cropped mags')
		with h5py.File(map_joiner(rename_h5(imagefile, '_mag_cropped')), "r") as f:
			img_mag = f['images'][()]
		print('Loading cropped sensemaps')
		with h5py.File(map_joiner('SenseMapsCpp_cropped.h5'), "r") as f:
			smaps = f['Maps'][()]
		print('\nLoaded cropped images\n')

	return img, img_mag, smaps

def enc_image(img, img_mag, out_images, dirpath, imagefile, 
	create_enc_image=False, load_enc_image=False, also_plot=False):
	
	map_joiner = lambda path: os.path.join(dirpath, path)

	if create_enc_image and load_enc_image:
		raise RuntimeError('Can not both load and create enc images')

	img_enc = np.array([0])
	img_mvel = np.array([0])

	if create_enc_image:
		old_img = img

		img[:,0,...] *= (2.0 * np.expand_dims(img_mag, axis=0) / np.max(img_mag, axis=(0,1,2)))

		
		if img.shape[0] != out_images:
			print('Interpolating')
			img = np.real(ic.interpolate_images(img, out_images).astype(np.complex64)).astype(np.float32)

		with h5py.File(map_joiner('images_mvel_' + str(out_images) + 'f_cropped_interpolated.h5'), "w") as f:
			f.create_dataset('images', data=img)
		img_mvel = img

		v_enc = 1.0
		A = (np.pi/(v_enc*np.sqrt(3))) * np.array(
			[
			[ 0,  0,  0],
			[-1, -1, -1],
			[ 1,  1, -1],
			[ 1, -1,  1],
			[-1,  1,  1]
			], dtype=np.float32)

		imvel = np.expand_dims(np.transpose(img[:,1:], axes=(2,3,4,0,1)), axis=-1)
		imvel /= np.mean(imvel)

		imvel = np.sign(imvel) * (imvel ** 3)
		max_vel = np.max(imvel)
		imvel /= max_vel
		imvel *= (v_enc / np.sqrt(3))

		if also_plot:
			pu.image_nd(np.sqrt(imvel[...,0,0]**2 + imvel[...,1,0]**2 + imvel[...,2,0]**2))

		print('Applying encoding matrix')
		imenc = (A @ imvel).squeeze(-1)
		imenc = np.transpose(imenc, axes=(3,4,0,1,2))

		imenc = (np.expand_dims(img[:,0], axis=1) * (np.cos(imenc) + 1j*np.sin(imenc))).astype(np.complex64)

		if also_plot:
			pu.image_nd(imenc)

		print('Writing interpolated encoded')
		with h5py.File(map_joiner('images_encs_' + str(out_images) + 'f_cropped_interpolated.h5'), "w") as f:
			f.create_dataset('images', data=imenc)

		img_enc = imenc

		print('\nCreated encoded images\n')
	elif load_enc_image:
		print('Loading mvel')
		with h5py.File(map_joiner('images_mvel_' + str(out_images) + 'f_cropped_interpolated.h5'), "r") as f:
			img_mvel = f['images'][()]
		print('Loading encs')
		with h5py.File(map_joiner('images_encs_' + str(out_images) + 'f_cropped_interpolated.h5'), "r") as f:
			img_enc = f['images'][()]
		print('\nLoaded encoded images\n')

	return img_enc, img_mvel

def nufft_of_enced_image(img_enc, smaps, dirpath, 
	nspokes, nsamp_per_spoke, method, crop_factor=1.0,
	create_nufft_of_enced_image=False, load_nufft_of_neced_image=False, also_plot=False):
	
	nfreq = nspokes * nsamp_per_spoke
	map_joiner = lambda path: os.path.join(dirpath, path)

	if create_nufft_of_enced_image and load_nufft_of_neced_image:
		raise RuntimeError('Can not both load and create nufft_of_enced_image images')

	frame_coords = []
	frame_kdatas = []

	if create_nufft_of_enced_image:
		nframes = img_enc.shape[0]
		nenc = img_enc.shape[1]
		nsmaps = smaps.shape[0]

		#coords = np.empty((nframes,nenc,3,nfreq), dtype=np.float32)
		#kdatas = np.empty((nframes,nenc,nsmaps,nfreq), dtype=np.complex64)

		im_size = (img_enc.shape[2], img_enc.shape[3], img_enc.shape[4])
		num_voxels = math.prod(list(im_size))

		print('Simulating MRI camera')
		for frame in range(nframes):
			print('Frame: ', frame, '/', nframes)

			encode_coords = []
			encode_kdatas = []
			for encode in range(nenc):
				print('Encode: ', encode, '/', nenc, '  Creating coordinates')

				coil_kdatas = []

				coord = np.ascontiguousarray(ic.create_coords(nspokes, nsamp_per_spoke, im_size, method, False, crop_factor))

				if also_plot:
					pu.scatter_3d(coord)

				coord_torch = torch.tensor(coord).to(torch.device('cuda:0')).contiguous()
				nufftobj = nufft.NufftT(coord_torch, smaps.shape[1:])
				for smap in range(nsmaps):
					print('\r Coil: ', smap, '/', nsmaps, end="")
					coiled_image = torch.tensor(img_enc[frame,encode,...] * smaps[smap,...]).to(
						torch.device('cuda:0')).unsqueeze(0).contiguous()
					coil_kdatas.append(nufftobj.apply(coiled_image).cpu().numpy())

				encode_coords.append(coord)
				encode_kdatas.append(np.stack(coil_kdatas, axis=1))

			frame_coords.append(encode_coords)
			frame_kdatas.append(encode_kdatas)

		with h5py.File(map_joiner('simulated_coords_kdatas.h5'), "w") as f:
			for i in range(nframes):
				for j in range(nenc):
					ijstr = str(i)+'_e'+str(j)
					f.create_dataset('coords_f'+ijstr, data=frame_coords[i][j])
					f.create_dataset('kdatas_f'+ijstr, data=frame_kdatas[i][j])

		print('\nCreated coords and kdatas\n')
	elif load_nufft_of_neced_image:
		frame_coords = np.array([], dtype=object)
		frame_kdatas = np.array([], dtype=object)
		with h5py.File(map_joiner('simulated_coords_kdatas.h5'), "r") as f:
			for i in range(nframes):
				encode_coords = np.array([], dtype=object)
				encode_kdatas = np.array([], dtype=object)
				for j in range(nenc):
					ijstr = str(i)+'_e'+str(j)
					np.append(encode_coords, f['coords_f'+ijstr])
					np.append(encode_kdatas, f['kdatas_f'+ijstr])
		print('\nLoaded coords and kdatas\n')

	return frame_coords, frame_kdatas

def simulate(dirpath='D:\\4DRecon\\dat\\dat2', imagefile='images_6f.h5',
		create_crop_image=False, load_crop_image=False, 
	    create_enc_image=False, load_enc_image=False,
	    create_nufft_of_enced_image=False, load_nufft_of_neced_image=False,
	    nimgout=20,
	    nspokes=500,
	    samp_per_spoke=489,
		method='PCVIPR',
		crop_factor=2.0,
		just_plot=False,
		also_plot=False):

	img, img_mag, smaps = crop_image(dirpath, imagefile, create_crop_image, load_crop_image, just_plot, also_plot)

	if just_plot:
		return

	img_enc, img_mvel = enc_image(img, img_mag, nimgout, dirpath, imagefile, create_enc_image, load_enc_image, also_plot)

	if also_plot:
		pu.image_nd(img_enc)
		pu.image_nd(ic.get_CD(img_mvel))

	coords, kdatas = nufft_of_enced_image(img_enc, smaps, dirpath, 
		nspokes, samp_per_spoke, method, crop_factor,
		create_nufft_of_enced_image, load_nufft_of_neced_image, also_plot)

def load_coords_kdatas(dirpath):
	map_joiner = lambda path: os.path.join(dirpath, path)

	def frames_and_encodes(keys):
		framelist = list()
		encodelist = list()
		for key in keys:
			m = re.findall(r'coords_f\d+_e0', key)
			if len(m) != 0:
				framelist.append(m[0])
			m = re.findall(r'coords_f0_e\d+', key)
			if len(m) != 0:
				encodelist.append(m[0])

		return (len(framelist), len(encodelist))
		
	nframes = 0
	nenc = 0
	frame_coords = []
	frame_kdatas = []
	with h5py.File(map_joiner('simulated_coords_kdatas.h5'), "r") as f:
		nframes, nenc = frames_and_encodes(list(f.keys()))
		for i in range(nframes):
			encode_coords = []
			encode_kdatas = []
			for j in range(nenc):
				ijstr = str(i)+'_e'+str(j)
				encode_coords.append(f['coords_f'+ijstr][()])
				encode_kdatas.append(f['kdatas_f'+ijstr][()])
			frame_coords.append(encode_coords)
			frame_kdatas.append(encode_kdatas)
	print('\nLoaded coords and kdatas\n')
	return (frame_coords, frame_kdatas, nframes, nenc)

def load_smaps(dirpath):
	map_joiner = lambda path: os.path.join(dirpath, path)
	smaps = np.array([])
	with h5py.File(map_joiner('SenseMapsCpp_cropped.h5'), "r") as f:
		smaps = f['Maps'][()]
	return smaps



