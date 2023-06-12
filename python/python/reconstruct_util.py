import torch
import numpy as np
import math

import cupy as cp
import cupyx

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

from torch_linop import TorchLinop, TorchScaleLinop
from torch_grad_methods import TorchCG, TorchGD
from torch_maxeig import TorchMaxEig

dll_path = "D:/Documents/GitHub/HastyCompute/out/install/x64-release-cuda/bin/HastyPyInterface.dll"
torch.ops.load_library(dll_path)
hasty_sense = torch.ops.HastySense

def load_simulated_diag_rhs(coords, kdatas, smaps, nframes, nenc, use_weights=False, root=0):
	# mean flow
	ncoils = smaps.shape[0]

	coord_vec = []
	kdata_vec = []
	for encode in range(nenc):
		frame_coords = []
		frame_kdatas = []
		for frame in range(nframes):
			frame_coords.append(coords[frame][encode])
			frame_kdatas.append(kdatas[frame][encode])

		coord = np.concatenate(frame_coords, axis=1)
		kdata = np.concatenate(frame_kdatas, axis=2)
		coord_vec.append(torch.tensor(coord))
		kdata_vec.append(torch.tensor(kdata))

	diagonal_vec = []
	rhs_vec = []

	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	nvoxels = math.prod(list(im_size))
	cudev = torch.device('cuda:0')

	for i in range(nenc):
		print('Encoding: ', i)
		coord = coord_vec[i]
		kdata = kdata_vec[i]

		coord_cu = coord.to(cudev)

		weights: torch.Tensor
		if use_weights:
			print('Calculating density compensation')
			weights = tkbn.calc_density_compensation_function(ktraj=coord_cu, 
				im_size=im_size).to(torch.float32)
			
			for _ in range(root):
				weights = torch.sqrt(weights)
			
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, weights=weights.squeeze(0), im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

			weights = torch.sqrt(weights).squeeze(0)
		else:
			print('Building toeplitz kernel')
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

		uimsize = [1,im_size[0],im_size[1],im_size[2]]

		print('Calculating RHS')
		rhs = torch.zeros(tuple(uimsize), dtype=torch.complex64).to(cudev)
		for j in range(ncoils): #range(nsmaps):
			print('Coil: ', j, '/', ncoils)
			SH = smaps[j,...].conj().to(cudev).unsqueeze(0)
			b = kdata[j,0,...].unsqueeze(0).to(cudev)
			if use_weights:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, weights * b, uimsize) / math.sqrt(nvoxels)
				rhs += rhs_j
			else:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, b, uimsize) / math.sqrt(nvoxels)
				rhs += rhs_j

		rhs_vec.append(rhs)

	rhs = torch.stack(rhs_vec, dim=0).cpu()
	diagonals = torch.stack(diagonal_vec, dim=0)

	return diagonals, rhs

def load_real():
	smaps = np.array([0])
	with h5py.File('D:/4DRecon/dat/dat2/SenseMapsCpp.h5') as f:
		smap_vec = []
		smapsdata = f['Maps']
		ncoils = len(smapsdata.keys())
		for i in range(ncoils):
			smp = smapsdata['SenseMaps_' + str(i)][()]
			smap_vec.append(smp['real']+1j*smp['imag'])
		smaps = np.stack(smap_vec, axis=0)
		del smap_vec

	coord = np.array([0])
	kdata = np.array([0])
	weights = np.array([0])
	gating = np.array([0])

	with h5py.File('D:/4DRecon/MRI_Raw.h5') as f:
		kdataset = f['Kdata']

		# Coords
		if True:
			kx_vec = []
			ky_vec = []
			kz_vec = []
			for i in range(5):
				kx_vec.append(kdataset['KX_E' + str(i)][()])
				ky_vec.append(kdataset['KY_E' + str(i)][()])
				kz_vec.append(kdataset['KZ_E' + str(i)][()])
			kx = np.stack(kx_vec, axis=0)
			ky = np.stack(ky_vec, axis=0)
			kz = np.stack(kz_vec, axis=0)
			coord = np.stack([kx,ky,kz], axis=0)

		# K-Data
		if True:
			kdata_vec = []
			for i in range(5):
				kdata_enc = []
				for j in range(32):
					kde = kdataset['KData_E'+str(i)+'_C'+str(j)][()]
					kdata_enc.append(kde['real'] + 1j*kde['imag'])
				kdata_vec.append(np.stack(kdata_enc, axis=0))
			kdata = np.stack(kdata_vec, axis=0)

		# Density Compensation
		if True:
			kw_vec = []
			for i in range(5):
				kw_vec.append(kdataset['KW_E'+str(i)][()])
			weights = np.stack(kw_vec, axis=0)

		gatingset = f['Gating']
		# Gating
		if True:
			gating = gatingset['ECG_E0'][()][0,:]


	return (torch.tensor(smaps), coord, kdata, weights, gating)

def gate(coord, kdata, weights, gating, nframes):
	mean = np.mean(gating)
	upper_bound = 2*mean
	length = upper_bound / nframes

	spokelen = coord.shape[-1]
	nspokes = coord.shape[-2]
	ncoil = kdata.shape[1]	

	coord_vec = []
	kdata_vec = []
	weights_vec = []


	def add_idx_spokes(idx):
		nspokes_idx = np.count_nonzero(idx)
		nlast = nspokes_idx*spokelen

		coordf = coord[:,:,0,idx,:].reshape((3,5,nlast))
		kdataf = kdata[:,:,0,idx,:].reshape((5,ncoil,nlast))
		weightsf = weights[:,0,idx,:].reshape((5,nlast))

		for j in range(5):
			coord_vec.append(torch.tensor(coordf[:,j,:]))
			kdata_vec.append(torch.tensor(kdataf[j,:,:]).unsqueeze(0))
			weights_vec.append(torch.tensor(weightsf[j,:]).unsqueeze(0))

	len_start = length + length/2
	gates = [len_start]
	# First Frame
	if True:
		idx = gating < len_start
		add_idx_spokes(idx)

	# Mid Frames
	for i in range(1,nframes-1):
		new_len_start = len_start + length
		idx = np.logical_and(len_start <= gating, gating < new_len_start)
		len_start = new_len_start
		gates.append(len_start)

		add_idx_spokes(idx)

	# Last Frame
	if True:
		idx = len_start <= gating
		add_idx_spokes(idx)

	return (coord_vec, kdata_vec, weights_vec, gates)

def gated_full(coord_vec, kdata_vec, weights_vec, nframes):
	nenc = 5

	coord_vec_full1 = [
		coord_vec[0],coord_vec[1],coord_vec[2],coord_vec[3],coord_vec[4]]
	kdata_vec_full1 = [
		kdata_vec[0],kdata_vec[1],kdata_vec[2],kdata_vec[3],kdata_vec[4]]
	weights_vec_full1 = [
		weights_vec[0],weights_vec[1],weights_vec[2],weights_vec[3],weights_vec[4]]

	for i in range(1,nframes):
		for j in range(i*nenc,(i+1)*nenc):
			coord_vec_full1.append(coord_vec[j])
			kdata_vec_full1.append(kdata_vec[j])
			weights_vec_full1.append(weights_vec[j])

	coord_vec_full = []
	kdata_vec_full = []
	weights_vec_full = []
	for i in range(0,nenc):
		coord_vec_full.append(torch.concat(coord_vec_full1, dim=-1))
		kdata_vec_full.append(torch.concat(kdata_vec_full1, dim=-1))
		weights_vec_full.append(torch.concat(weights_vec_full1, dim=-1))

	return coord_vec_full, kdata_vec_full, weights_vec_full

def load_real_full_diag_rhs(smaps, coord_vec, kdata_vec, weights_vec, use_weights=False, root=0):
	nenc = 5
	diagonal_vec = []
	rhs_vec = []

	ncoils = smaps.shape[0]
	im_size = (smaps.shape[1], smaps.shape[1], smaps.shape[2])

	cudev = torch.device('cuda:0')

	for i in range(nenc):
		coord_cu = coord_vec[i].to(cudev)
		weights_cu = torch.tensor([0.0])
		kdata = kdata_vec[i]

		if use_weights:
			weights_cu = weights_vec[i].to(cudev)

			for _ in range(root):
				weights_cu = torch.sqrt(weights_cu)

			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, weights=weights_cu, im_size=im_size).cpu()
			torch.cuda.empty_cache()
			diagonal_vec.append(diagonal)

			weights_cu = torch.sqrt(weights_cu)
		else:
			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, im_size=im_size).cpu()
			diagonal_vec.append(diagonal)

		uimsize = [1,im_size[0],im_size[1],im_size[2]]
		nvoxel = math.prod(list(im_size))

		rhs = torch.zeros(tuple(uimsize), dtype=torch.complex64).to(cudev)
		for j in range(ncoils):
			SH = smaps[j,...].conj().to(cudev).unsqueeze(0)
			b = kdata[j,0,...].to(cudev).unsqueeze(0)
			if use_weights:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, weights_cu*b, uimsize) / math.sqrt(nvoxel)
				rhs += rhs_j
			else:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, b, uimsize) / math.sqrt(nvoxel)
				rhs += rhs_j

			rhs_vec.append(rhs)

		rhs_vec.append(rhs)

	rhs = torch.stack(rhs_vec, dim=0).cpu()
	diagonals = torch.stack(diagonal_vec, dim=0)

	return diagonals, rhs


class ToeplitzLinop(TorchLinop):
	def __init__(self, shape, smaps, nenc, diagonals):
		coil_list = list()
		for i in range(nenc):
			inner_coil_list = list()
			for j in range(smaps.shape[0]):
				inner_coil_list.append(j)
			coil_list.append(inner_coil_list)
		self.coil_list = coil_list
		self.smaps = smaps
		self.diagonals = diagonals

		super().__init__(shape, shape)

	def _apply(self, input):
		input_copy = input.clone()
		hasty_sense.batched_sense_toeplitz_diagonals(input_copy, self.coil_list, self.smaps, self.diagonals)
		return input_copy

def dct_l1_prox(image, lamda):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			gpuimg = cupyx.scipy.fft.dctn(cp.array(image[i,j,...].numpy()))
			gpuimg = cp.exp(1j*cp.angle(gpuimg)) * cp.maximum(0, (cp.abs(gpuimg) - lamda))
			image[i,j,...] = torch.tensor(cupyx.scipy.fft.idctn(gpuimg).get())
	
	return image

def reconstruct_cg_full(diagonals, rhs, smaps, nenc, iter = 50, lamda=0.1, images=None, plot=False):
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzLinop(vec_size, smaps, nenc, diagonals)

	scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=12).run()).to(torch.float32)

	scaled_linop = TorchScaleLinop(toep_linop, scaling)
	rhs *= torch.sqrt(scaling)

	tcg = TorchCG(scaled_linop, rhs, images, max_iter=iter)
	
	prox = lambda x: dct_l1_prox(x, lamda)

	if lamda != 0.0:
		return tcg.run_with_prox(prox, plot)
	else:
		return tcg.run(plot)

def reconstruct_gd_full(diagonals, rhs, smaps, nenc, iter = 50, lamda=0.1, images=None, plot=False):
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzLinop(vec_size, smaps, nenc, diagonals)

	scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=12).run()).to(torch.float32)
	rhs *= torch.sqrt(scaling)

	scaled_linop = TorchScaleLinop(toep_linop, scaling)

	gradf = lambda x: scaled_linop(x) -  rhs

	tgd = TorchGD(gradf, images, alpha=1.0, accelerate=True, max_iter=iter)

	prox = lambda x: dct_l1_prox(x, lamda)

	if lamda != 0.0:
		return tgd.run_with_prox(prox, plot)
	else:
		return tgd.run(plot)
