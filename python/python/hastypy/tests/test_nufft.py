import test

import torch
import time

import hastypy.ffi.hasty_nufft as hnufft

nx = 128
nf = 100000

cudev = torch.device('cuda:0')

coord = torch.rand((3,nf), dtype=torch.float32, device=cudev)


nu = hnufft.Nufft(coord, [1,nx,nx,nx], hnufft.NufftOptions.type2())

output = torch.rand((32,nf), dtype=torch.complex64, device=cudev)
input = torch.rand((32,nx,nx,nx), dtype=torch.complex64, device=cudev)

start = time.time()
for i in range(32):
	nu.apply(input[i,...].unsqueeze(0), output[i,...].unsqueeze(0))
end = time.time()

print(end - start)


nu = hnufft.Nufft(coord, [32,nx,nx,nx], hnufft.NufftOptions.type2())

start = time.time()
nu.apply(input, output)
end = time.time()
print(end - start)


