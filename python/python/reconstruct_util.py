import torch
import numpy as np
import math

import cupy as cp
import cupyx

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

from torch_linop import TorchLinop, TorchScaleLinop, TorchL2Reg
from torch_grad_methods import TorchCG, TorchGD
from torch_maxeig import TorchMaxEig
import torch_precond as tprcnd

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

def load_full_real(coord, kdata, weights):
	coord_vec = []
	kdata_vec = []
	weights_vec = []
	
	nlast = coord.shape[-1] * coord.shape[-2]
	ncoil = kdata.shape[1]
	coordf = coord[:,:,0,:,:].reshape((3,5,nlast))
	kdataf = kdata[:,:,0,:,:].reshape((5,ncoil,nlast))
	weightsf = weights[:,0,:,:].reshape((5,nlast))
	for j in range(5):
		coord_vec.append(torch.tensor(coordf[:,j,:]))
		kdata_vec.append(torch.tensor(kdataf[j,:,:]).unsqueeze(0))
		weights_vec.append(torch.tensor(weightsf[j,:]).unsqueeze(0))

	return coord_vec, kdata_vec, weights_vec


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
		if weights is not None:
			weightsf = weights[:,0,idx,:].reshape((5,nlast))

		for j in range(5):
			coord_vec.append(torch.tensor(coordf[:,j,:]))
			kdata_vec.append(torch.tensor(kdataf[j,:,:]).unsqueeze(0))
			if weights is not None:
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
	if weights_vec is not None:
		weights_vec_full1 = [
			weights_vec[0],weights_vec[1],weights_vec[2],
			weights_vec[3],weights_vec[4]]

	for i in range(1,nframes):
		for j in range(i*nenc,(i+1)*nenc):
			coord_vec_full1.append(coord_vec[j])
			kdata_vec_full1.append(kdata_vec[j])
			if weights_vec is not None:
				weights_vec_full1.append(weights_vec[j])

	coord_vec_full = []
	kdata_vec_full = []
	weights_vec_full = []
	for i in range(0,nenc):
		coord_vec_full.append(torch.concat(coord_vec_full1, dim=-1))
		kdata_vec_full.append(torch.concat(kdata_vec_full1, dim=-1))
		if weights_vec is not None:
			weights_vec_full.append(torch.concat(weights_vec_full1, dim=-1))

	return coord_vec_full, kdata_vec_full, weights_vec_full

def crop_kspace(coord_vec, kdata_vec, weights_vec, im_size, crop_factor=1.0):
	max = 0.0
	for i in range(len(coord_vec)):
		maxi = coord_vec[i].abs().max().item()
		if maxi > max:
			max = maxi

	kim_size = tuple((e / 2) * crop_factor for e in im_size)


	for i in range(len(coord_vec)):
		coord = coord_vec[i]
		idxx = coord[0,:] < kim_size[0]
		idxy = coord[1,:] < kim_size[1]
		idxz = coord[2,:] < kim_size[2]

		idx = torch.logical_and(idxx, torch.logical_and(idxy, idxz))

		coord_vec[i] = torch.pi * coord[:,idx] / maxi
		kdata_vec[i] = kdata_vec[i][:,:,idx]
		if weights_vec is not None:
			weights_vec[i] = weights_vec[i][:,idx]
	
	return (coord_vec, kdata_vec, weights_vec)

def translate(coord_vec, kdata_vec, translation):
	cudev = torch.device('cuda:0')
	for i in range(len(coord_vec)):
		coord = coord_vec[i].to(cudev)
		kdata = kdata_vec[i].to(cudev)
		mult = torch.tensor(list(translation)).unsqueeze(0).to(cudev)

		kdata *= torch.exp(1j*(mult @ coord)).unsqueeze(0)

		kdata_vec[i] = kdata.cpu()
		#coord_vec[i] = coord

	return coord_vec, kdata_vec

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
			weights_cu = weights_vec[i].to(cudev).contiguous()

			for _ in range(root):
				weights_cu = torch.sqrt(weights_cu)

			diagonal = tkbn.calc_toeplitz_kernel(omega=coord_cu, weights=weights_cu, im_size=im_size).cpu()
			torch.cuda.empty_cache()
			diagonal_vec.append(diagonal)

			#weights_cu = torch.sqrt(weights_cu)
		else:
			diagonal = (8 * tkbn.calc_toeplitz_kernel(omega=coord_cu, im_size=im_size)).cpu()
			torch.cuda.empty_cache()
			diagonal_vec.append(diagonal)

		uimsize = [1,im_size[0],im_size[1],im_size[2]]
		nvoxel = math.prod(list(im_size))

		rhs = torch.zeros(tuple(uimsize), dtype=torch.complex64).to(cudev)
		for j in range(ncoils):
			SH = smaps[j,...].conj().to(cudev).unsqueeze(0).contiguous()
			b = kdata[0,j,...].to(cudev).unsqueeze(0).contiguous()
			if use_weights:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, weights_cu*b, uimsize) / math.sqrt(nvoxel)
				rhs += rhs_j
			else:
				rhs_j = SH * hasty_sense.nufft1(coord_cu, b, uimsize) / math.sqrt(nvoxel)
				rhs += rhs_j

		rhs_vec.append(rhs)

	rhs = torch.stack(rhs_vec, dim=0).cpu()
	diagonals = torch.stack(diagonal_vec, dim=0)

	return diagonals, rhs


class SenseLinop(TorchLinop):
	def __init__(self, smaps, coord_vec, kdata_vec, weights_vec=None, clone=True, randomize_coils=False, num_rand_coils=16):
		self.smaps = smaps
		self.nframes = len(coord_vec)
		self.shape = tuple([self.nframes, 1] + list(smaps.shape[1:]))
		self.ncoils = self.smaps.shape[0]
		self.randomize_coils = randomize_coils
		self.num_rand_coils = num_rand_coils
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec
		self.clone = clone
		self.coil_list = self.create_coil_list()

		super().__init__(self.shape, self.shape)

	def create_coil_list(self):
		if self.randomize_coils:
			coil_list = list()
			for i in range(self.nframes):
				permuted = np.random.permutation(self.ncoils).astype(np.int32)
				coil_list.append(permuted[:self.num_rand_coils].tolist())
			return coil_list
		else:
			coil_list = list()
			for i in range(self.nframes):
				coil_list.append(np.arange(self.nframes).tolist())

	def _apply(self, input):
		input_copy: torch.Tensor
		if self.clone:
			input_copy = input.detach().clone()
		else:
			input_copy = input

		if self.randomize_coils:
			self.coil_list = self.create_coil_list()

		if self.weights_vec is not None:
			hasty_sense.batched_sense_weighted_kdata(input_copy, self.coil_list,
				self.smaps, self.coord_vec, self.weights_vec, self.kdata_vec, None)
		else:
			hasty_sense.batched_sense_kdata(input_copy, self.coil_list,
				self.smaps, self.coord_vec, self.kdata_vec, None)
		return input_copy
		
class ToeplitzSenseLinop(TorchLinop):
	def __init__(self, smaps, diagonals, clone=True):
		nenc = diagonals.shape[0]
		shape = tuple([nenc, 1] + list(smaps.shape[1:]))
		coil_list = list()
		for i in range(nenc):
			inner_coil_list = list()
			for j in range(smaps.shape[0]):
				inner_coil_list.append(j)
			coil_list.append(inner_coil_list)
		self.coil_list = coil_list
		self.smaps = smaps
		self.diagonals = diagonals
		self.clone = clone

		super().__init__(shape, shape)

	def _apply(self, input):
		input_copy: torch.Tensor
		if self.clone:
			input_copy = input.detach().clone()
		else:
			input_copy = input
		hasty_sense.batched_sense_toeplitz_diagonals(input_copy, self.coil_list, self.smaps, self.diagonals, None)
		return input_copy

class PrecondLinop(TorchLinop):
	def __init__(self, smaps, coord_vec, weights_vec = None, clone = True):
		shape = tuple([len(coord_vec), 1] + list(smaps.shape[1:]))
		self.clone = clone
		Pvec = []
		for i in range(len(coord_vec)):
			if weights_vec is not None:
				Pvec.append(tprcnd.circulant_precond(smaps, coord_vec[i], weights_vec[i]))
			else:
				Pvec.append(tprcnd.circulant_precond(smaps, coord_vec[i]))
		self.Pvec = Pvec
		super().__init__(shape, shape)
				
	def _apply(self, input):
		output: torch.Tensor
		if self.clone:
			output = torch.empty_like(input)

		for i in range(input.shape[0]):
			for j in range(input.shape[1]):
				if self.clone:
					output[i,j,...] = torch.fft.fftn(
						torch.fft.ifftn(input[i,j,...]) * self.Pvec[i])
				else:
					input[i,j,...] = torch.fft.fftn(
						torch.fft.ifftn(input[i,j,...]) * self.Pvec[i])
		
		if self.clone:
			return output
		else:
			return input

def dct_l1_prox(image, lamda):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			gpuimg = cupyx.scipy.fft.dctn(cp.array(image[i,j,...].numpy()))
			gpuimg = cp.exp(1j*cp.angle(gpuimg)) * cp.maximum(0, (cp.abs(gpuimg) - lamda))
			image[i,j,...] = torch.tensor(cupyx.scipy.fft.idctn(gpuimg).get())
	
	return image

def reconstruct_cg_full(diagonals, rhs, smaps, Prcnd = None, iter = 50, lamda=0.1, images=None, plot=False):
	nenc = diagonals.shape[0]
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	toep_linop = ToeplitzSenseLinop(smaps, diagonals)

	#scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=5).run()).to(torch.float32)
	#scaled_linop = TorchScaleLinop(toep_linop, scaling)
	#l2_regd_linop = TorchL2Reg(scaled_linop, lamda=0.0001)

	final_linop = toep_linop

	print('Scaling image')
	rhs_scalar = torch.sum(images.conj() * rhs)
	lhs_scalar = torch.sum(images.conj() * final_linop(images))
	scalar = (rhs_scalar / lhs_scalar)
	images *= torch.real(scalar).nan_to_num(nan = 1.0, posinf=1.0, neginf=1.0) + 1j*torch.imag(
		scalar).nan_to_num(nan = 0.0, posinf=0.0, neginf=0.0)

	tcg = TorchCG(final_linop, rhs, images, Prcnd, max_iter=iter)
	
	prox = lambda x: dct_l1_prox(x, lamda)

	if lamda != 0.0:
		return tcg.run_with_prox(prox, plot)
	else:
		return tcg.run(plot)

def reconstruct_gd_full(smaps, coord_vec, kdata_vec, weights_vec=None, iter = 50, lamda=0.1, images=None, plot=False):
	nenc = len(coord_vec)
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	vec_size = (nenc,1,im_size[0],im_size[1],im_size[2])
	nvoxels = smaps.shape[1] * smaps.shape[2] * smaps.shape[3]

	if images is None:
		images = torch.zeros(vec_size, dtype=torch.complex64)

	sense_linop = SenseLinop(smaps, coord_vec, kdata_vec, weights_vec)

	#scaling = (1 / TorchMaxEig(toep_linop, torch.complex64, max_iter=12).run()).to(torch.float32)
	#rhs *= torch.sqrt(scaling)
	final_linop = sense_linop

	gradf = lambda x: final_linop(x) #/ nvoxels

	tgd = TorchGD(gradf, images, alpha=1.0, accelerate=True, max_iter=iter)

	prox = lambda x: dct_l1_prox(x, lamda)

	if lamda != 0.0:
		return tgd.run_with_prox(prox, plot)
	else:
		return tgd.run(plot)

def direct_nufft_reconstruct_encs(smaps, coord_vec, kdata_vec, weights_vec, im_size):
	nframes = len(coord_vec)
	ncoils = kdata_vec[0].shape[1]
	cudev = torch.device('cuda:0')

	image = torch.empty(nframes, im_size[0], im_size[1], im_size[2], dtype=torch.complex64)
	nvoxels = im_size[0] * im_size[1] * im_size[2]

	for i in range(nframes):
		print('Encoding: ', i, '/', nframes)
		L = torch.zeros((ncoils, im_size[0], im_size[1], im_size[2]), dtype=torch.complex64)
		coord_cu = coord_vec[i].to(cudev)
		weights_cu = weights_vec[i].to(cudev)
		for j in range(ncoils):
			kdata_cu = kdata_vec[i][:,j,:].to(cudev)
			L[j, ...] = (hasty_sense.nufft1(coord_cu, weights_cu * kdata_cu, 
		    	[1, im_size[0], im_size[1], im_size[2]]) / math.sqrt(nvoxels)).cpu().squeeze(0)
			

		I = torch.sum(smaps.conj() * L, dim=0) / torch.sum(smaps.conj() * smaps, dim=0)
		image[i,...] = I.squeeze(0)

	return image.unsqueeze(1)

def reconstruct_frames(images, smaps, coord, kdata, nenc, nframes):
	im_size = (smaps.shape[1], smaps.shape[2], smaps.shape[3])
	ncoil = smaps.shape[0]
	nframes = images.shape[0]
	nenc = images.shape[1]
	grad_shape = (nframes*nenc,1,im_size[0], im_size[1], im_size[2])
	svt_shape = (nframes,nenc,im_size[0], im_size[1], im_size[2])

	sense_linop = SenseLinop(smaps, coord, kdata, randomize_coils=True, num_rand_coils=16)
	gradf = lambda x: sense_linop(x)

	images = images.view(grad_shape)

	print('Scaling')
	f1 = torch.sum(gradf(images*0.0)**2) #k+m
	f2 = torch.sum(gradf(images)**2) # m
	scaling = -(f2 / (f1 - f2))
	images *= scaling

	sgd = TorchGD(gradf, images,alpha=1.0, accelerate=True, max_iter=10)

	sgd.run(plot=True)

	return images.view(svt_shape)

	
	



