#import torch
import numpy as np

import sys
import os

from os.path import exists
from pathlib import Path

import ctypes as ct

print('PID: ', os.getpid())

paths1 = [
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

paths2 = [
	"python311.dll",
	"asmjit.dll",
	"c10.dll",
	"c10_cuda.dll",
	"cublas64_12.dll",
	"cublaslt64_12.dll",
	"cudart64_12.dll",
	"cudnn64_8.dll",
	"cufft64_11.dll",
	"cufinufft.dll",
	"cusolver64_11.dll",
	"cusparse64_12.dll",
	"fbgemm.dll",
	"libiomp5md.dll",
	"nvjitlink_120_0.dll",
	"nvtoolsext64_1.dll",
	"python311.dll",
	"torch_cpu.dll",
	"torch_cuda.dll",
	"uv.dll",
	"vcruntime140.dll",
	"zlib.dll",
	"HastyComputeLib.dll",
]

paths3 = [
	"MSVCP140.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
    "api-ms-win-crt-runtime-l1-1-0.dll",
    "api-ms-win-crt-heap-l1-1-0.dll",
    "api-ms-win-crt-string-l1-1-0.dll",
    "api-ms-win-crt-math-l1-1-0.dll",
    "KERNEL32.dll"
]

print(os.getcwd())

base_path = Path("D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin")

if False:
	dll_loop = True
	dlls1 = list()
	if dll_loop:
		for path in paths1:
			print(path)

			dll = ct.CDLL(str(path))
			dlls1.append(dll)

	print('\nLoaded first dependencies\n')
	dll_loop = True
	dlls2 = list()
	if dll_loop:
		for path in paths2:
			tot_path = base_path / path
			print(tot_path)
			dll = ct.CDLL(str(tot_path))
			dlls2.append(dll)


hasty_compute_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyComputeLib.dll"
dll = ct.CDLL(hasty_compute_path)

hasty_py_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
dll = ct.CDLL(hasty_py_path)


print('Hello')




