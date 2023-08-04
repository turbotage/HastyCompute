import torch
# import numpy as np
# import math
# import gc
# import random

# import cupy as cp
# import cupyx

#import plot_utility as pu
#import simulate_mri as simri

#import h5py
#import torchkbnufft as tkbn

#from torch_linop import TorchLinop, TorchScaleLinop, TorchL2Reg
#from torch_grad_methods import TorchCG, TorchGD
#from torch_maxeig import TorchMaxEig
#import torch_precond as tprcnd

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense
hasty_svt = torch.ops.HastySVT

tensorlist = []
for i in range(4):
	tensorlist.append(torch.rand(3,3))

print(tensorlist)
print('hello')

hasty_sense.dummy(tensorlist)

print('hello')
print(tensorlist)
