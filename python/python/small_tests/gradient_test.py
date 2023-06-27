
import torch
import numpy as np

import ctypes as ct
import math

dll_path = "D:\\Documents\\GitHub\\HastyCompute\\out\\install\\x64-release-cuda\\bin\\HastyPyInterface.dll"

import matplotlib.pyplot as plt

torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

N1 = 16
N2 = 16
N3 = 13
NF = N1*N2*N3

coord = torch.rand((3,NF), dtype=torch.float32)

cudev = torch.device('cuda:0')

input = torch.rand((1,N1,N2,N3), dtype=torch.complex64).to(cudev)

smaps = torch.rand((32,N1,N2,N3), dtype=torch.complex64)

output = hasty_sense.nufft2(coord, input * smaps)


