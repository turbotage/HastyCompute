import torch
import numpy as np

import plot_utility as pu
import simulate_mri as simri

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

copied_images = torch.tensor(images.copy())


dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_py = torch.ops.HastyPyInterface

hasty_py.batched_sense(copied_images, torch.tensor(smaps), coord_vec, kdata_vec)

images = images - copied_images.numpy()

pu.image_5d(np.abs(images))



