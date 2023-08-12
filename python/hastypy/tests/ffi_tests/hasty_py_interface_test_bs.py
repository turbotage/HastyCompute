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

print(hbsense_ops.doc())

BatchedSense = hbsense_mod.BatchedSense
BatchedSenseAdjoint = hbsense_mod.BatchedSenseAdjoint
BatchedSenseNormal = hbsense_mod.BatchedSenseNormal
BatchedSenseNormalAdjoint = hbsense_mod.BatchedSenseNormalAdjoint


def test_normal():
	nx = 78
	ny = 127
	nz = 203
	nouter = 20
	nmodes = [1,nx,ny,nz]
	ncoil = 24

	nfreq = 300000

	input = torch.rand(nouter,1,nx,ny,nz, dtype=torch.complex64).to(torch.device('cuda:0'))
	coords = []
	midstore = []
	kdata = []
	for i in range(nouter):
		coords.append(torch.rand(3,nfreq+i, dtype=torch.float32).to(torch.device('cuda:0')))
		midstore.append(torch.zeros(1,ncoil,nfreq+i, dtype=torch.complex64).to(torch.device('cuda:0')))
		kdata.append(torch.rand(1,ncoil,nfreq+i, dtype=torch.complex64).to(torch.device('cuda:0')))

	smaps = torch.rand(ncoil,nx,ny,nz, dtype=torch.complex64)

	output = torch.zeros(nouter,1,nx,ny,nz, dtype=torch.complex64).to(torch.device('cuda:0'))

	bsense = BatchedSense(coords, smaps, kdata, None, None) 
	bsensea = BatchedSenseAdjoint(coords, smaps, None, None, None)
	bsensen = BatchedSenseNormal(coords, smaps, kdata, None, None)

	bsense.apply(input, midstore, None)
	bsensea.apply(midstore, output, None)

	output_forward_backward = output.detach().clone()

	bsensen.apply(input, output, None)

	output_normal = output.detach().clone()

	rel_err = torch.norm(output_forward_backward - output_normal) / torch.norm(output_normal)

	print("Normal: Test gave relative error of: ", rel_err)
	
def test_normal_adjoint():
	nx = 87
	ny = 127
	nz = 203
	nouter = 17
	nmodes = [1,nx,ny,nz]
	ncoil = 19

	nfreq = 300000

	input = []
	output = []
	coords = []
	for i in range(nouter):
		coords.append(torch.rand(3,nfreq+i, dtype=torch.float32).to(torch.device('cuda:0')))
		input.append(torch.rand(1,ncoil,nfreq+i, dtype=torch.complex64).to(torch.device('cuda:0')))
		output.append(torch.empty(1,ncoil,nfreq+i, dtype=torch.complex64).to(torch.device('cuda:0')))

	smaps = torch.rand(ncoil,nx,ny,nz, dtype=torch.complex64)
	midstore = torch.rand(nouter,1,nx,ny,nz, dtype=torch.complex64)

	bsense = BatchedSense(coords, smaps, None, None, None) 
	bsensea = BatchedSenseAdjoint(coords, smaps, None, None, None)
	bsensena = BatchedSenseNormalAdjoint(coords, smaps, None, None, None)

	bsensea.apply(input, midstore, None)
	bsense.apply(midstore, output, None)

	output_forward_backward = []
	for outp in output:
		output_forward_backward.append(outp.detach().clone())

	bsensena.apply(input, output, None)

	output_normal = []
	for outp in output:
		output_normal.append(outp.detach().clone())

	rel_err = torch.tensor(0.0)
	for i in range(len(output_normal)):
		rel_err += torch.norm(output_forward_backward[i] - output_normal[i]) / torch.norm(output_normal[i])
	rel_err /= len(output_normal)
	
	print("NormalAdjoint: Test gave relative error of: ", rel_err)





test_normal()
test_normal_adjoint()