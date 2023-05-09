import torch
import time
import h5py
import numpy as np
import plot_utility as pu

def timer(func, input):
    torch.cuda.synchronize()
    start = time.time()
    func(input)
    torch.cuda.synchronize()
    end = time.time()
    return end - start


def svd_test():

	def run_svd(A):
		n = min(A.shape)
		U,S,Vh = torch.linalg.svd(A, full_matrices=False)
		S[0:(n // 2)] = 0.0
		Vh *= S.unsqueeze(1)
		return U @ Vh
	
	A = torch.rand(40,16*16*16*5, dtype=torch.complex64)

	for i in range(10):
		print(timer(run_svd, A))


	for i in range(10):
		print(timer(run_svd, A.transpose(0,1)))


import torch.nn.functional as tnf

plot_smaps = False
smap_list = []
with h5py.File('D:\\4DRecon\\SenseMapsCpp.h5', 'r') as hf:
	maps_base = hf['Maps']
	maps_key_base = 'SenseMaps_'
	for i in range(len(list(maps_base))):
		smap = maps_base[maps_key_base + str(i)][()]
		smap = smap['real'] + 1j*smap['imag']
	
		if plot_smaps:
			pu.image_3d(np.abs(smap))

		smap = torch.tensor(smap).unsqueeze(0).unsqueeze(0)
		smap_r = tnf.interpolate(torch.real(smap), (64,64,64), mode='trilinear')
		smap_i = tnf.interpolate(torch.imag(smap), (64,64,64), mode='trilinear')
		smap = (smap_r + 1j*smap_i).squeeze(0).squeeze(0).numpy()

		if plot_smaps:
			pu.image_3d(np.abs(smap))

		smap_list.append(smap)

smaps = np.stack(smap_list, axis=0)

pu.image_4d(np.abs(smaps))



