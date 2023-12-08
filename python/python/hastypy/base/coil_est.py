import torch
import h5py

import os
import math
import gc

from hastypy.base.opalg import Vector, Linop

from hastypy.base.recon import FivePointLLR, FivePointFULL
import hastypy.base.load_and_gate as lag
from hastypy.base.load_and_gate import FivePointLoader
from hastypy.base.nufft import NufftT, NufftAdjointT, NufftNormalT
from hastypy.base.sense import SenseAdjointT, SenseNormalT
from hastypy.ffi.hasty_interface import FunctionLambda
from hastypy.base.torch_opalg import TorchConjugateGradient

import hastypy.util.plot_utility as pu

import scipy
import numpy as np


class CoilMapNormal(Linop):
	def __init__(self, coord_vec: list[torch.Tensor], image: torch.Tensor, weights_vec=None):
		self.coord_vec = coord_vec
		self.image = image
		self.weights_vec = weights_vec
	
	def _apply(self, input: Vector):
		inp = input.get_tensor()
		output = torch.zeros_like(inp)
		for i in range(self.image.shape[0]):
			nn = NufftNormal(self.coord_vec[i], [1] + self.image.shape[1:])
			img = self.image.select(0,i)

			for j in range(inp.shape[0]):
				smap = inp.select(0,j)
				output[j,...] += img.conj() * nn.apply(Vector(img * smap)).get_tensor()
		return Vector(output)


# TODO: Alternating Minimization, This works poorly
class SenseEstimation:
	def __init__(self, smaps_est: torch.Tensor, coil_img_est: torch.Tensor, full_img_est: torch.Tensor,
	      coord: torch.Tensor, kdata: torch.Tensor, weights: torch.Tensor, im_size: tuple[int], sense_size: tuple[int] = (24,24,24), decay_factor=1.0):
		
		self.cudev = torch.device('cuda:0')
		self.smaps = smaps_est.to(self.cudev)
		self.coil_img = coil_img_est.to(self.cudev)
		self.full_img = full_img_est.to(device=self.cudev, dtype=torch.complex64)


		self.coord = coord.to(self.cudev, non_blocking=True)
		self.weights = weights.to(self.cudev, non_blocking=True)
		self.kdata = kdata.to(self.cudev, non_blocking=True)
		self.im_size = im_size
		self.sense_size = sense_size
		self.decay_factor = decay_factor
		self.ncoil = self.kdata.shape[1]
		self.coils = [i for i in range(self.ncoil)]


	def estimate_coil_images(self):
		pass
		# Forward

		ndim = len(self.im_size)

		kspace_length = 0.0
		for i in range(ndim):
			kspace_length += ((self.sense_size[i] / self.im_size[i]) * torch.pi)**2 #(0.125 * im_size[i] * torch.pi) ** 2
		kspace_length = math.sqrt(kspace_length)

		coord_length = torch.sum(torch.square(self.coord), dim=0).sqrt()

		idx = coord_length > kspace_length

		# Smoothing transient for smaps
		decaymult = torch.exp(torch.square(self.decay_factor*(coord_length[idx] - kspace_length) / kspace_length).neg())

		#weight_scaling_script = """
#def apply(a: List[Tensor], b: List[Tensor]):
#	a[0] *= b[0]
#"""
		#func_lamda = FunctionLambda(weight_scaling_script, "apply", [self.weights])

		nn = NufftNormalT(self.coord, self.im_size)
		na = NufftAdjointT(self.coord, self.im_size)

		print('Estimate coil images')
		for c in range(self.ncoil):
			kdatac = self.kdata[0,c,:].unsqueeze(0).detach().clone()
			kdatac[:,idx] *= decaymult
			rhs = na.apply(kdatac)
			
			print("\r", end="")
			print('Coil: ', c, '/', self.ncoil, end="")
			self.coil_img[c,...] = TorchConjugateGradient(nn, rhs, 
						     self.coil_img[c,...].unsqueeze(0), max_iter=20).run()
			
		pu.image_nd(self.coil_img.cpu().numpy())
		print('\nDone.')
	
	def image_and_smaps_update(self):
		snn = SenseNormalT(self.coord, self.smaps, self.coils)
		rhs = SenseAdjointT(self.coord, self.smaps, self.coils).apply(self.kdata.squeeze(0))

		self.full_img = TorchConjugateGradient(snn, rhs, self.full_img, max_iter=10).run()
		denominator = torch.abs(self.full_img)
		denominator += torch.max(denominator)*1e-5

		self.smaps = self.coil_img / denominator

	def run(self, iter=10):
		self.estimate_coil_images()

		for i in range(iter):
			print('\r Update image and smaps: ', i, end="")
			self.image_and_smaps_update()

		print('\nDone.')
		return self.smaps
			
			
def low_res_sensemaps(coord: torch.Tensor, kdata: torch.Tensor, weights: torch.Tensor, im_size: tuple[int], 
		kernel_size: tuple[int] = (24,24,24)):

	ndim = len(im_size)
	ncoil = kdata.shape[1]

	torch.cuda.empty_cache()

	#smaps = torch.zeros((ncoil,) + im_size, dtype=kdata_vec[0].dtype, device=torch.device('cuda:0'))
	coil_images = torch.zeros((ncoil,) + im_size, dtype=kdata.dtype, device=torch.device('cuda:0'))
	coordcu = coord.to(torch.device('cuda:0'), non_blocking=False)
	weightscu = weights.to(torch.device('cuda:0'), non_blocking=False)
	kdatacu = kdata.to(torch.device('cuda:0'), non_blocking=False)

	torch.cuda.synchronize()

	kernel = torch.ones(kernel_size, dtype=torch.float32, device=torch.device('cuda:0'))
	conv = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding='same', bias=False)
	conv.weights = kernel
	conv = conv.to(torch.device('cuda:0'))

	torch.cuda.synchronize()
	na = NufftAdjointT(coordcu, im_size)
	na.apply(torch.zeros_like(kdatacu[:,0,:]))
	torch.cuda.synchronize()

	for c in range(ncoil):
		kd = kdatacu[:,c,:] * weightscu
		torch.cuda.synchronize()
		coil_images[c,...] = na.apply(kd)
		torch.cuda.synchronize()
		if torch.logical_not(torch.isfinite(coil_images[c,...])).any():
			raise RuntimeError('coil image contained non finite value')
		
		# Apply smoothing filter
		#tempc = coil_images[c,...].unsqueeze(0)
		#coil_images[c,...] = conv(torch.real(tempc)) + 1j*conv(torch.imag(tempc)).squeeze(0).contiguous()

	del conv
	del kernel


	coil_images = coil_images.cpu()
	gc.collect()
	torch.cuda.empty_cache()

	pu.image_nd(coil_images.numpy())

	sos = torch.sqrt(torch.sum(torch.square(torch.abs(coil_images)), dim=0)).unsqueeze(0)
	sos += torch.max(sos)*1e-5

	return sos, coil_images, coil_images / sos
	


def walsh(coord: torch.Tensor, kdata: torch.Tensor, weights: torch.Tensor, im_size: tuple[int]):

	def extract_blocks(image, block_shape, block_stride):
		print('helo')

	
