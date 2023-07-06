
import numpy as np
import h5py
import torch
import gc

import plot_utility as pu
import image_creation as ic

import torchkbnufft as tkbn

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense
hasty_svt = torch.ops.HastySVT


img = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_real.h5', "r") as f:
	img = f['images'][()]
	img = img[:,0,...][np.newaxis,...]
gc.collect()

pu.image_nd(img)
gc.collect()

mean = np.mean(img)
std = np.std(img)

img /= mean
img = img * ((0.2 + np.cos(np.linspace(0,3.141592/2.0,30)).astype(np.float32))[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])
gc.collect()
print('Created temporals')

pu.image_nd(img)
gc.collect()

img_copy = torch.tensor(img).detach().clone()

hasty_svt.random_blocks_svt(img_copy, 5000, 16, 10.0, False, None)

print('Performed SVT')

pu.image_nd(img_copy.numpy())

print('Hello')
