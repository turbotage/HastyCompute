import torch
import h5py

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
	      coord: torch.Tensor, kdata: torch.Tensor, im_size: tuple[int], weights: torch.Tensor | None = None):
		
		self.cudev = torch.device('cuda:0')
		self.smaps = smaps_est.to(self.cudev)
		self.coil_img = coil_img_est.to(self.cudev)
		self.full_img = full_img_est.to(device=self.cudev, dtype=torch.complex64)


		self.coord = coord.to(self.cudev)
		self.weights = weights.to(self.cudev) if weights is not None else None
		self.kdata = kdata.to(self.cudev)
		self.im_size = im_size
		self.ncoil = self.kdata.shape[1]
		self.coils = [i for i in range(self.ncoil)]


	def estimate_coil_images(self):
		pass
		# Forward
		
		weight_scaling_script = """
def apply(a: List[Tensor], b: List[Tensor]):
	a[0] *= b[0]
"""
		func_lamda = FunctionLambda(weight_scaling_script, "apply", [self.weights])

		nn = NufftNormalT(self.coord, self.im_size, func_lamda)
		na = NufftAdjointT(self.coord, self.im_size)

		print('Estimate coil images')
		for c in range(self.ncoil):
			rhs = na.apply(self.weights * self.kdata[0,c,:].unsqueeze(0))
			
			print("\r", end="")
			print('Coil: ', c, '/', self.ncoil, end="")
			self.coil_img[c,...] = TorchConjugateGradient(nn, rhs, 
						     self.coil_img[c,...].unsqueeze(0), max_iter=20).run()
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
			
			

		



def low_res_sensemaps(coord: torch.Tensor, kdata: torch.Tensor, weights: torch.Tensor, im_size: tuple[int]):
	ndim = len(im_size)
	ncoil = kdata.shape[1]

	kspace_length = 0.0
	for i in range(ndim):
		kspace_length += (0.125 * im_size[i] * torch.pi) ** 2
	kspace_length = math.sqrt(kspace_length)

	#smaps = torch.zeros((ncoil,) + im_size, dtype=kdata_vec[0].dtype, device=torch.device('cuda:0'))
	coil_images = torch.zeros((ncoil,) + im_size, dtype=kdata.dtype, device=torch.device('cuda:0'))
	coord = coord.to(torch.device('cuda:0'), non_blocking=True)
	weights = weights.to(torch.device('cuda:0'), non_blocking=True)
	kdata = kdata.to(torch.device('cuda:0'), non_blocking=True)

	coord_length = torch.sum(torch.square(coord), dim=0).sqrt()

	idx = coord_length < kspace_length

	# Smoothing transient for smaps
	weights[:,idx] *= torch.exp(torch.square(2*(coord_length[idx] - kspace_length) / kspace_length).neg())

	na = NufftAdjointT(coord, im_size)

	for c in range(ncoil):
		kd = kdata[:,c,:] * weights
		coil_images[c,...] = na.apply(kd)

	gc.collect()

	sos = torch.sqrt(torch.sum(torch.square(torch.abs(coil_images)), dim=0)).unsqueeze(0)
	sos += torch.max(sos)*1e-5

	return sos, coil_images, coil_images / sos
	


		