
import numpy as np
import h5py
import torch

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

pu.image_nd(img)

mean = np.mean(img)
std = np.std(img)

img /= mean
img = img * ((0.2 + np.cos(np.linspace(0,3.141592/2.0,30)))[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis])

pu.image_nd(img)

img_copy = torch.tensor(img).detach().clone()

hasty_svt.random_blocks_svt(img_copy, 5000, 16, 10.0, False, None)

pu.image_nd(img_copy)
