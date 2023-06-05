import torch
import numpy as np

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

from torch_linop import TorchLinop, TorchScaleLinop
from torch_cg import TorchCG
from torch_maxeig import TorchMaxEig

def load_diag_rhs(coords, kdatas, use_weights=False, root=0):
	# mean flow
	nframe = coords.shape[0]
	nfreq = coords.shape[3]
	nenc = coords.shape[1]
	ncoils = kdatas.shape[2]

	diagonal_vec = []
	rhs_vec = []
	coord_vec = []
	kdata_vec = []

	for i in range(nenc):
		print('Encoding: ', i, '/', nenc)
		coord_enc = coords[:,i,...]
		coord = np.transpose(coord_enc, axes=(1,0,2)).reshape(3,nframe*nfreq)
		coord = torch.tensor(coord)
		coord_vec.append(coord)

		cudev = torch.device('cuda:0')

		coord_cu = coord.to(cudev)

		weights: torch.Tensor
		if use_weights:
			print('Calculating density compensation')
			weights = tkbn.calc_density_compensation_function(ktraj=coord_cu, 
				im_size=im_size).to(torch.float32)
			
			for i in range(root):
				weights = torch.sqrt(weights)
			
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, weights=weights.squeeze(0), im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

			weights = torch.sqrt(weights).squeeze(0)
		else:
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

		kdata_enc = kdatas[:,i,...]
		kdata = np.transpose(kdata_enc, axes=(1,0,2)).reshape(ncoils,nframe*nfreq)
		kdata = torch.tensor(kdata).unsqueeze(0)
		kdata_vec.append(kdata)

		print('Calculating RHS')
		rhs = torch.zeros((1,150,150,150), dtype=torch.complex64).to(cudev)
		nsmaps = smaps.shape[0]
		for i in range(nsmaps):
			print('Coil: ', i, '/', nsmaps)
			if use_weights:
				rhs += smaps[i,...].conj().to(cudev).unsqueeze(0) * hasty_sense.nufft1(coord_cu, weights * kdata[0,i,...].unsqueeze(0).to(cudev), [1,150,150,150])
			else:
				rhs += smaps[i,...].conj().to(cudev).unsqueeze(0) * hasty_sense.nufft1(coord_cu, kdata[0,i,...].unsqueeze(0).to(cudev), [1,150,150,150])

		rhs_vec.append(rhs)

	rhs = torch.stack(rhs_vec, dim=0).cpu()
	diagonals = torch.stack(diagonal_vec, dim=0)

	return diagonals, rhs

class ToeplitzLinop(TorchLinop):
	def __init__(self, shape, smaps, nenc, diagonals):
		coil_list = list()
		for i in range(nenc):
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

def reconstruct(diagonals, rhs, smaps, im_size, nenc, iter = 50, images=None):
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzLinop(vec_size, smaps, nenc, diagonals)

	scaling = (1 / TorchMaxEig(toep_linop, torch.complex64).run()).to(torch.float32)

	scaled_linop = TorchScaleLinop(toep_linop, scaling)

	tcg = TorchCG(scaled_linop, scaling * rhs, images, max_iter=iter)
	
	return tcg.run()


coords, kdatas = simri.load_coords_kdatas()
smaps = torch.tensor(simri.load_smaps())

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

nenc = coords.shape[1]
im_size = (smaps.shape[1],smaps.shape[2],smaps.shape[3])

use_weights = True

print('Beginning weighted load')
diagonals, rhs = load_diag_rhs(coords, kdatas, use_weights=True, root=0)
print('Beginning weighted reconstruct')
images = reconstruct(diagonals, rhs, smaps, im_size, nenc, iter=50)

pu.image_5d(np.abs(images))

print('Beginning unweighted load')
diagonals, rhs = load_diag_rhs(coords, kdatas, use_weights=False, root=0)
print('Beginning unweighted reconstruct')
images = reconstruct(diagonals, rhs, smaps, im_size, nenc, iter=200)

pu.image_5d(np.abs(images))

# diagonals, rhs = load_diag_rhs(coords, kdatas, use_weights=True, root=1)

# images = reconstruct(diagonals, rhs, smaps, im_size, nenc, iter=50)

# pu.image_5d(np.abs(images))

# diagonals, rhs = load_diag_rhs(coords, kdatas, use_weights=True, root=2)

# images = reconstruct(diagonals, rhs, smaps, im_size, nenc, iter=50)

# pu.image_5d(np.abs(images))afed


with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed_weighted.h5', "w") as f:
	f.create_dataset('images', data=images)





