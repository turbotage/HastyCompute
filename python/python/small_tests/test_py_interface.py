import torch
import numpy as np

import sys
import os

from os.path import exists
from pathlib import Path

paths = [
	"c10.dll",
	"fbgemm.dll",
	"libiomp5md.dll",
	"uv.dll",
	"KERNEL32.dll",
	"MSVCP140.dll",
	"WS2_32.dll",
	"VCRUNTIME140.dll",
	"VCRUNTIME140_1.dll",
	"api-ms-win-crt-runtime-l1-1-0.dll",
	"api-ms-win-crt-heap-l1-1-0.dll",
	"api-ms-win-crt-math-l1-1-0.dll",
	"api-ms-win-crt-environment-l1-1-0.dll",
	"api-ms-win-crt-string-l1-1-0.dll",
	"api-ms-win-crt-convert-l1-1-0.dll",
	"api-ms-win-crt-stdio-l1-1-0.dll",
	"api-ms-win-crt-locale-l1-1-0.dll",
	"api-ms-win-crt-filesystem-l1-1-0.dll",
	"api-ms-win-crt-utility-l1-1-0.dll",
	"api-ms-win-crt-time-l1-1-0.dll",
	"api-ms-win-crt-process-l1-1-0.dll",
    "torch_cpu.dll"
]

base_path = Path("D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin")

dll_loop = False
if dll_loop:
	import ctypes as ct
	for path in paths:
		try:
			dll = ct.CDLL(path)
		except ...:
			try:
				tot_path = base_path / path
				dll = ct.CDLL(tot_path)
			except ...:
				print(path)
				raise

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"

print(os.getpid())

torch.ops.load_library(dll_path)

coord = -3.141592 + 2*3.141592*torch.rand(3,10000, dtype=torch.float32, device=torch.device('cuda:0'))

image = torch.rand(1,64,64,32, dtype=torch.complex64, device=torch.device('cuda:0'))

nufft2_func = torch.ops.HastyPyInterface.nufft2
nufft1_func = torch.ops.HastyPyInterface.nufft1
nufft_normal = torch.ops.HastyPyInterface.nufft2to1


kdata = nufft2_func(coord, image)
image_back = nufft1_func(coord, kdata, list((1,64,64,32)))

image_back2 = nufft_normal(coord, image)


