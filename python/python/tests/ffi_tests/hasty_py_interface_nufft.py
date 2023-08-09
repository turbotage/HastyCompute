import torch

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
#torch.ops.load_library(dll_path)

torch.classes.load_library(dll_path)
torch.ops.load_library(dll_path)

print(torch.classes.loaded_libraries)


hi_mod = torch.classes.HastyInterface
hsvt_mod = torch.classes.HastySVT
hnufft_mod = torch.classes.HastyNufft
hsense_mod = torch.classes.HastySense
hbsense_mod = torch.classes.HastyBatchedSense

hi_ops = torch.ops.HastyInterface
hsvt_ops = torch.ops.HastySVT
hnufft_ops = torch.ops.HastyNufft
hsense_ops = torch.ops.HastySense
hbsense_ops = torch.ops.HastyBatchedSense

print(hnufft_ops.doc())

NufftOptions = hnufft_mod.NufftOptions

Nufft = hnufft_mod.Nufft
NufftNormal = hnufft_mod.NufftNormal


type1_opt = NufftOptions(1, None, None)
type2_opt = NufftOptions(2, None, None)

nx = 64
ny = 50
nz = 43
nmodes = [1,nx,ny,nz]

nfreq = 100000

input = torch.rand(1,nx,ny,nz, dtype=torch.complex64).to(torch.device('cuda:0'))
coord = torch.rand(3,nfreq, dtype=torch.float32).to(torch.device('cuda:0'))

midstore = torch.empty(1,nfreq, dtype=torch.complex64).to(torch.device('cuda:0'))
output = torch.empty(1,nx,ny,nz, dtype=torch.complex64).to(torch.device('cuda:0'))

nufft2 = Nufft(coord, nmodes, type2_opt)
nufft1 = Nufft(coord, nmodes, type1_opt)

nufft_normal = NufftNormal(coord, nmodes, type2_opt, type1_opt)

nufft2.apply(input, midstore)
nufft1.apply(midstore, output)

output_forward_backward = output.detach().clone()

nufft_normal.apply(input, output, midstore, None)


output_normal = output.detach().clone()

rel_err = torch.norm(output_normal - output_forward_backward) / torch.norm(output_normal)

print("Test gave relative error of: ", rel_err)
