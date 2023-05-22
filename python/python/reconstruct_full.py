import torch
import numpy as np

import plot_utility as pu
import simulate_mri as simri

import h5py

coords, kdatas = simri.load_coords_kdatas()
smaps = simri.load_smaps()

# mean flow
nframe = coords.shape[0]
nfreq = coords.shape[3]
nenc = coords.shape[1]
ncoils = kdatas.shape[2]

coord_vec = []
kdata_vec = []
for i in range(nenc):
	coord_enc = coords[:,i,...]
	coord = np.transpose(coord_enc, axes=(1,0,2)).reshape(3,nframe*nfreq)
	coord = torch.tensor(coord)
	coord_vec.append(coord)

	kdata_enc = kdatas[:,i,...]
	kdata = np.transpose(kdata_enc, axes=(1,0,2)).reshape(ncoils,nframe*nfreq)
	kdata = torch.tensor(kdata).unsqueeze(0)
	kdata_vec.append(kdata)


images = np.ones((nenc,1,150,150,150), dtype=np.complex64)


dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_py = torch.ops.HastyPyInterface

coil_list = list()
for i in range(images.shape[0]):
	inner_coil_list = list()
	for j in range(smaps.shape[0]):
		inner_coil_list.append(j)
	coil_list.append(inner_coil_list)

iters = 50
for i in range(iters):
	copied_images = torch.tensor(images.copy())
	hasty_py.batched_sense(copied_images, coil_list, torch.tensor(smaps), coord_vec, kdata_vec)
	images = images - np.nan_to_num(0.2*copied_images.numpy(), copy=False, nan=0.0, posinf=0.0, neginf=0.0)

pu.image_5d(np.abs(images))

with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed.h5', "w") as f:
	f.create_dataset('images', data=images)





