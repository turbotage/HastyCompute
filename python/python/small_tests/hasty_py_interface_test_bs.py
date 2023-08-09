import torch

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
#torch.ops.load_library(dll_path)

torch.classes.load_library(dll_path)

print(torch.classes.loaded_libraries)


hbs_mod = torch.classes.HastyBatchedSense

BatchedSense = hbs_mod.BatchedSense

nfreq = 500 * 489
nouter = 10
ncoil = 16

nx = 64
ny = 64
nz = 64

coords = []
for i in range(nouter):
	coords.append(torch.rand((3, nfreq), dtype=torch.float32))

smaps = torch.rand((ncoil, nx, ny, nz), dtype=torch.complex64)

image = torch.rand((nouter, 1, nx, ny, nz), dtype=torch.complex64)

out = []
for i in range(nouter):
	out.append(torch.empty((1,ncoil,nfreq), dtype=torch.complex64))

bs = BatchedSense(coords, smaps, None, None, None)
bs.apply(image, out, None)

print(len(out))

