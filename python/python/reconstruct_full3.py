import torch
import numpy as np

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

import random
import time


dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_py = torch.ops.HastyPyInterface

coords, kdatas = simri.load_coords_kdatas()
smaps = simri.load_smaps()

# mean flow
nframe = coords.shape[0]
nfreq = coords.shape[3]
nenc = coords.shape[1]
ncoils = kdatas.shape[2]

coord_vec = []
kdata_vec = []
weights_vec = []
start = time.time()
for i in range(nenc):
	coord_enc = coords[:,i,...]
	coord = np.transpose(coord_enc, axes=(1,0,2)).reshape(3,nframe*nfreq)
	coord = torch.tensor(coord)
	coord_vec.append(coord)

	weights = tkbn.calc_density_compensation_function(ktraj=coord.to(torch.device('cuda:0')), 
		im_size=(150,150,150)).squeeze(0).squeeze(0).cpu()
	#weights = torch.ones(1,nframe*nfreq)
	weights_vec.append(weights)

	kdata_enc = kdatas[:,i,...]
	kdata = np.transpose(kdata_enc, axes=(1,0,2)).reshape(ncoils,nframe*nfreq)
	kdata = torch.tensor(kdata).unsqueeze(0)
	kdata_vec.append(kdata)

torch.cuda.synchronize()
end = time.time()

print('Time Build Dcomp: ', end - start)

images = np.ones((nenc,1,150,150,150), dtype=np.complex64)

iters = 2

nsmap_div = 2
nsmaps = smaps.shape[0] // nsmap_div


errors = []
start = time.time()
for i in range(iters):

	coil_list = list()
	print('\nPython lists: ')
	for i in range(images.shape[0]):
		inner_coil_list = list()
		for j in range(smaps.shape[0]):
			inner_coil_list.append(j)
		random.shuffle(inner_coil_list)
		inner_coil_list = inner_coil_list[:(smaps.shape[0] // nsmap_div)]
		print("outer_batch: ", i, " coils: ", inner_coil_list)
		coil_list.append(inner_coil_list)
	print(' ')

	old_images = torch.tensor(images.copy())
	copied_images = torch.tensor(images.copy())
	hasty_py.batched_sense_weighted(copied_images, coil_list, torch.tensor(smaps), coord_vec, weights_vec, kdata_vec)
	images = images - 0.5 * (1.0 / nsmaps) * np.nan_to_num(copied_images.numpy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)

	error = torch.norm(old_images - images) / torch.norm(old_images)
	errors.append(error)

torch.cuda.synchronize()
end = time.time()

print('\nTime Reconstruct: ', end - start)
print('Errors: ', errors)

pu.image_5d(np.abs(images))

with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed.h5', "w") as f:
	f.create_dataset('images', data=images)





