import torch
import numpy as np

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

import torch_linop as tlinop
from torch_cg import TorchCG

coords, kdatas = simri.load_coords_kdatas()
smaps = torch.tensor(simri.load_smaps())

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

# mean flow
nframe = coords.shape[0]
nfreq = coords.shape[3]
nenc = coords.shape[1]
ncoils = kdatas.shape[2]

im_size = (150,150,150)

diagonal_vec = []
rhs_vec = []
coord_vec = []
kdata_vec = []
for i in range(nenc):
	coord_enc = coords[:,i,...]
	coord = np.transpose(coord_enc, axes=(1,0,2)).reshape(3,nframe*nfreq)
	coord = torch.tensor(coord)
	coord_vec.append(coord)

	cudev = torch.device('cuda:0')

	coord_cu = coord.to(cudev)

	diagonal = tkbn.calc_toeplitz_kernel(coord_cu, im_size).cpu()
	diagonal_vec.append(diagonal)

	kdata_enc = kdatas[:,i,...]
	kdata = np.transpose(kdata_enc, axes=(1,0,2)).reshape(ncoils,nframe*nfreq)
	kdata = torch.tensor(kdata).unsqueeze(0)
	kdata_vec.append(kdata)

	rhs = torch.zeros((1,150,150,150), dtype=torch.complex64).to(cudev)
	for i in range(smaps.shape[0]):
		rhs += smaps[i,...].conj().to(cudev).unsqueeze(0) * hasty_sense.nufft1(coord_cu, kdata[0,i,...].unsqueeze(0).to(cudev), [1,150,150,150])

	rhs_vec.append(rhs)

rhs = torch.stack(rhs_vec, dim=0).cpu()
diagonals = torch.stack(diagonal_vec, dim=0)


images = torch.ones((nenc,1,150,150,150), dtype=torch.complex64)

class ToeplitzLinop(tlinop.TorchLinop):
	def __init__(self, shape, smaps, diagonals):
		coil_list = list()
		for i in range(images.shape[0]):
			inner_coil_list = list()
			for j in range(smaps.shape[0]):
				inner_coil_list.append(j)
			coil_list.append(inner_coil_list)
		self.coil_list = coil_list
		self.smaps = smaps
		self.diagonals = diagonals

		super().__init__(shape, shape)

	def _apply(self, input):
		input_copy = input.clone()
		hasty_sense.batched_sense_toeplitz_diagonals(input_copy, self.coil_list, self.smaps, self.diagonals)
		return input_copy

toep_linop = ToeplitzLinop((nenc,1,150,150,150), smaps, diagonals)

tcg = TorchCG(toep_linop, rhs, images, max_iter=500)

i = 0
while not tcg.done():
	print('Iter: ', i)
	tcg.update()
	i += 1

images = tcg.x

pu.image_5d(np.abs(images))

with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed.h5', "w") as f:
	f.create_dataset('images', data=images)





