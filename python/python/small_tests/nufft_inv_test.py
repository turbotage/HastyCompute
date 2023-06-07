
import torch
import numpy as np

import ctypes as ct
import math

dll_path = "D:\\Documents\\GitHub\\HastyCompute\\out\\install\\x64-release-cuda\\bin\\HastyPyInterface.dll"

import matplotlib.pyplot as plt

torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

N1 = 10
N2 = 10
N3 = 10
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

#coord[0,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N1
#coord[1,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N2
#coord[2,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N3

#coord = np.transpose(coord, (1,0))
cudev = torch.device('cuda:0')

coord = torch.tensor(coord).to(cudev)
input = torch.rand((1,N1,N2,N3), dtype=torch.complex64).to(cudev)

temp = hasty_sense.nufft2(coord, input) / math.sqrt(NF)
output = hasty_sense.nufft1(coord, temp, (1,N1,N2,N3)) / math.sqrt(NF)

input_numpy = input.cpu().numpy()
output_numpy = output.cpu().numpy()

plt.figure()
plt.plot(np.abs(input_numpy.flatten()), 'r-*')
plt.plot(np.abs(output_numpy.flatten()), 'g-*')
plt.show()

