import torch
import numpy as np
import math

import plot_utility as pu
import simulate_mri as simri
import torchkbnufft as tkbn
import reconstruct_util as ru
import matplotlib.pyplot as plt


dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

nenc = 2
ncoils = 8
N1 = 16
N2 = 16
N3 = 16
im_size = (N1,N2,N3)
NF = N1*N2*N3
coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	for y in range(N2):
		for z in range(N3):
			kx = -np.pi + x * 2 * np.pi / N1
			ky = -np.pi + y * 2 * np.pi / N2
			kz = -np.pi + z * 2 * np.pi / N3

			coord[0,l] = kx
			coord[1,l] = ky
			coord[2,l] = kz

			l += 1
coord = torch.tensor(coord)

smaps = torch.rand((ncoils,N1,N2,N3), dtype=torch.complex64)
cudev = torch.device('cuda:0')

coord_vec = []
kdata_vec = []
diagonal_vec = []
for i in range(nenc):
	coord_cu = coord.to(cudev)
	coord_vec.append(coord.detach().clone())
	kdata_vec.append(torch.rand((1,ncoils,NF), dtype=torch.complex64))
	diag = tkbn.calc_toeplitz_kernel(omega=coord_cu, im_size=im_size).cpu()
	diagonal_vec.append(diag)
diagonals = torch.stack(diagonal_vec, dim=0)
diagonals = torch.ones((nenc,2*N1,2*N2,2*N3), dtype=torch.float32)
    

toep_sense = ru.ToeplitzSenseLinop(smaps, diagonals)
sense = ru.SenseLinop(smaps, coord_vec, kdata_vec)

images = torch.ones((nenc, 1, N1, N2, N3), dtype=torch.complex64)

images1 = toep_sense(images)
#images2 = sense(images)

i1 = images1[0,...].numpy().flatten()
#i2 = images2[0,...].numpy().flatten()
i2 = torch.sum(smaps.conj() * smaps, dim=0).numpy().flatten()

plt.figure()
plt.plot(i1, 'r-*')
plt.plot(i2, 'g-o')
plt.show()


print('Hello')