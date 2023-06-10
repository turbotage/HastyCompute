import numpy as np
import h5py

import plot_utility as pu
import image_creation as ic

import math
import torch

import os
import fnmatch
import re

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

def rename_h5(name, appending):
	return name[:-3] + appending + '.h5'

def crop_image(dirpath, imagefile, create_crop_image=False, load_crop_image=False, just_plot=False):
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

		nlen = 128
		nx = 40
		ny = 40
		nz = 40 #45
		crop_box = [(nx,nx+nlen),(ny,ny+nlen),(nz,nz+nlen)]
		print('Cropping')
		new_img = ic.crop_5d_3d(img, crop_box).astype(np.float32)
		new_img_mag = ic.crop_3d_3d(img_mag, crop_box).astype(np.float32)
		new_smaps = ic.crop_4d_3d(smaps, crop_box).astype(np.complex64)

		if just_plot:
			pu.image_5d(new_img)

			cd = ic.get_CD(new_img)
			pu.maxip_4d(cd, axis=1)
			pu.maxip_4d(cd, axis=2)
			pu.maxip_4d(cd, axis=3)
			
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
	create_enc_image=False, load_enc_image=False):
	
	map_joiner = lambda path: os.path.join(dirpath, path)

	if create_enc_image and load_enc_image:
		raise RuntimeError('Can not both load and create enc images')

	img_enc = np.array([0])
	img_mvel = np.array([0])

	if create_enc_image:
		old_img = img

		img[:,0,...] *= (2.0 * np.expand_dims(img_mag, axis=0) / np.max(img_mag, axis=(0,1,2)))

		print('Interpolating')
		img = np.real(ic.interpolate_images(img, out_images).astype(np.complex64)).astype(np.float32)

		with h5py.File(map_joiner('images_mvel_' + str(out_images) + 'f_cropped_interpolated.h5'), "w") as f:
			f.create_dataset('images', data=img)
		img_mvel = img

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

		print('Applying encoding matrix')
		imenc = (A @ imvel).squeeze(-1)
		imenc = np.transpose(imenc, axes=(3,4,0,1,2))

		imenc = (np.expand_dims(img[:,0], axis=1) * (np.cos(imenc) + 1j*np.sin(imenc))).astype(np.complex64)

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
	create_nufft_of_enced_image=False, load_nufft_of_neced_image=False):
	
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
				print('Encode: ', encode, '/', nenc)
				print('Creating coordinates')

				coil_kdatas = []

				coord = np.ascontiguousarray(ic.create_coords(nspokes, nsamp_per_spoke, method, False, crop_factor))
				coord_torch = torch.tensor(coord).to(torch.device('cuda:0')).contiguous()
				for smap in range(nsmaps):
					print('Coil: ', smap, '/', nsmaps)
					coiled_image = img_enc[frame,encode,...] * smaps[smap,...]

					coild_image_torch = torch.tensor(coiled_image).to(torch.device('cuda:0')).unsqueeze(0).contiguous()

					kdata = hasty_sense.nufft2(coord_torch,coild_image_torch) / math.sqrt(num_voxels)
					coil_kdatas.append(kdata.cpu().numpy())

				encode_coords.append(coord)
				encode_kdatas.append(np.stack(coil_kdatas, axis=0))

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
	    nspokes=500,
	    samp_per_spoke=489,
		method='PCVIPR',
		crop_factor=2.0,
		just_plot=False):

	img, img_mag, smaps = crop_image(dirpath, imagefile, create_crop_image, load_crop_image, just_plot)

	if just_plot:
		return

	img_enc, img_mvel = enc_image(img, img_mag, 15, dirpath, imagefile, create_enc_image, load_enc_image)

	pu.image_5d(np.abs(img_enc))
	pu.maxip_4d(ic.get_CD(img_mvel))

	coords, kdatas = nufft_of_enced_image(img_enc, smaps, dirpath, 
		nspokes, samp_per_spoke, method, crop_factor,
		create_nufft_of_enced_image, load_nufft_of_neced_image)

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

def load_simulated_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=0):
	# mean flow
	ncoils = smaps.shape[0]

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

	diagonal_vec = []
	rhs_vec = []

	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	nvoxels = math.prod(list(im_size))
	cudev = torch.device('cuda:0')

	for i in range(nenc):
		print('Encoding: ', i)
		coord = coord_vec[i]
		kdata = kdata_vec[i]

		coord_cu = coord.to(cudev)

		weights: torch.Tensor
		if use_weights:
			print('Calculating density compensation')
			weights = tkbn.calc_density_compensation_function(ktraj=coord_cu, 
				im_size=im_size).to(torch.float32)
			
			for _ in range(root):
				weights = torch.sqrt(weights)
			
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, weights=weights.squeeze(0), im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

			weights = torch.sqrt(weights).squeeze(0)
		else:
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

		uimsize = [1,im_size[0],im_size[1],im_size[2]]

		print('Calculating RHS')
		rhs = torch.zeros(tuple(uimsize), dtype=torch.complex64).to(cudev)
		for j in range(ncoils): #range(nsmaps):
			print('Coil: ', j, '/', ncoils)
			SH = smaps[j,...].conj().to(cudev).unsqueeze(0)
			b = kdata[j,0,...].unsqueeze(0).to(cudev)
			if use_weights:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, weights * b, uimsize) / math.sqrt(nvoxels)
				rhs += rhs_j
			else:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, b, uimsize) / math.sqrt(nvoxels)
				rhs += rhs_j

		rhs_vec.append(rhs)

	rhs = torch.stack(rhs_vec, dim=0).cpu()
	diagonals = torch.stack(diagonal_vec, dim=0)

	return diagonals, rhs
