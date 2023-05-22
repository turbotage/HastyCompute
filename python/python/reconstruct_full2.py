import torch
import numpy as np

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

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
dev = torch.device('cuda:0')

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_py = torch.ops.HastyPyInterface

ss = np.sum(smaps * smaps, axis=0)

enc_images = list()
for j in range(nenc):
	coiled_images = list()
	coord = coord_vec[j].to(dev)
	dcomp = tkbn.calc_density_compensation_function(ktraj=coord, im_size=(150,150,150)).squeeze(0).squeeze(0)

	for i in range(smaps.shape[0]):
		kdat = kdata_vec[j][0,i,...].to(dev)
		coiled_images.append(hasty_py.nufft1(coord, 
			(kdat * dcomp).unsqueeze(0), (1,150,150,150)).cpu().numpy() / np.sqrt(150*150*150))
	
	cated = np.concatenate(coiled_images, axis=0)
	
	si = np.sum(cated * smaps, axis=0)

	enc_images.append(si / ss)


image_full = np.stack(enc_images, axis=0)


pu.image_4d(np.abs(image_full))

with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed2.h5', "w") as f:
	f.create_dataset('images', data=image_full)

	







