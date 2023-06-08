import torch
import numpy as np
import math

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

from torch_linop import TorchLinop, TorchScaleLinop
from torch_grad_methods import TorchCG, TorchGD
from torch_maxeig import TorchMaxEig

coords, kdatas, nframes, nenc = simri.load_coords_kdatas('D:/4DRecon/dat/dat2')
smaps = torch.tensor(simri.load_smaps('D:/4DRecon/dat/dat2'))

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

vec_size = [1, smaps.shape[1], smaps.shape[2], smaps.shape[3]]
img_size = [nframes, nenc, smaps.shape[0], smaps.shape[1], smaps.shape[2], smaps.shape[3]]


img = np.empty(tuple(img_size), dtype=np.complex64)
for i in range(nframes):
	print('Frame: ', i, '/', nframes)
	for j in range(nenc):
		print('Encode: ', j, '/', nenc)
		C = torch.tensor(coords[i][j]).to(torch.device('cuda:0'))
		b = torch.tensor(kdatas[i][j]).to(torch.device('cuda:0'))
		for c in range(smaps.shape[0]):
			img[i,j,c,...] = hasty_sense.nufft1(C, b[c,...], vec_size).cpu().numpy()

pu.image_6d(np.abs(img))

