import torch
import h5py

import gc

import runner

from hastypy.base.opalg import Vector, MaxEig
import hastypy.base.opalg as opalg

from hastypy.base.recon import FivePointLLR
import hastypy.base.load_and_gate as lag
from hastypy.base.load_and_gate import FivePointLoader

import hastypy.util.plot_utility as pu

from hastypy.base.recon import FivePointFULL
from hastypy.base.proximal import SVTOptions
from hastypy.base.svt import extract_mean_block
import hastypy.base.coil_est as coil_est

from hastypy.base.sense import BatchedSense, BatchedSenseAdjoint, BatchedSenseNormal

import hastypy.base.torch_precond as precond

class RunSettings:
	def __init__(self, im_size, crop_factors, prefovkmuls, postfovkmuls, shift):
		self.im_size = im_size
		self.crop_factors = crop_factors
		self.prefovkmuls = prefovkmuls
		self.postfovkmuls = postfovkmuls
		self.shift = shift

	def set_smaps_filepath(self, path):
		self.smaps_filepath = path
		return self

	def set_rawdata_filepath(self, path):
		self.rawdata_filepath = path
		return self

	def set_nframes(self, nframes):
		self.nframes = nframes
		return self


def run(settings: RunSettings):
	#smaps = FivePointLoader.load_smaps(settings.smaps_filepath, settings.im_size)
	#smaps = torch.permute(smaps, (0,3,2,1))

	#pu.image_nd(smaps.numpy())

	coords, kdata, weights, gating = FivePointLoader.load_raw(settings.rawdata_filepath)
	#normalize kdata
	kdata /= 0.5*(torch.mean(torch.real(kdata)) + torch.mean(torch.real(kdata)))

	coord_vec, kdata_vec, weights_vec = FivePointLoader.load_as_full(coords, kdata, weights)

	coord_vec, kdata_vec, weights_vec = lag.crop_kspace(coord_vec, kdata_vec, weights_vec, settings.im_size, 
		crop_factors=settings.crop_factors, prefovkmuls=settings.prefovkmuls, postfovkmuls=settings.postfovkmuls)

	kdatapoints = 0
	for kdatai in kdata_vec:
		kdatapoints += kdatai.shape[2]
	print('Num kdatapoints', kdatapoints)

	if settings.shift != (0.0, 0.0, 0.0):
		print('Translating... ', end="")
		kdata_vec = lag.translate(coord_vec, kdata_vec, settings.shift)
		print('Done.')

	# My smaps
	print('Estimating coil sensitivities...')
	_, _, smaps = coil_est.low_res_sensemaps(coord_vec[0], kdata_vec[0], weights_vec[0], im_size, sense_size=(16,16,16), decay_factor=1.0)
	smaps = (smaps / torch.mean(torch.abs(smaps))).cpu()
	print('Done estimating coil sensitivities')
	#pu.image_nd(smaps.numpy())

	if False:
		#A = BatchedSense(coord_vec, smaps)
		#AH = BatchedSenseAdjoint(coord_vec, smaps)
		#AHA = BatchedSenseNormal(coord_vec, smaps)
		#rand_input1 = opalg.rand((5,1) + smaps.shape[1:], dtype=torch.complex64)
		#rand_input2 = rand_input1.clone()
		#output1 = (AH * A)(rand_input1)
		#output2 = AHA(rand_input2)
		#relerr = opalg.norm(output1 - output2) / opalg.norm(output2)
		#print(relerr)
		pass
		#MaxEig(AH * A, torch.complex64, max_iter=8).run(print_info=True)
		#MaxEig(AHA, torch.complex64, max_iter=8).run(print_info=True)


	full_recon = FivePointFULL(smaps, coord_vec, kdata_vec, weights_vec, lamda=0.00005)

	image = torch.zeros((5,1) + settings.im_size, dtype=torch.complex64)

	def plot_callback(image: Vector, iter):
		pu.image_nd(image.tensor.numpy())

	gc.collect()
	torch.cuda.empty_cache()

	image = full_recon.run(image, iter=10, callback=None)

	pu.image_nd(image.numpy())

	return smaps, image, (coords, kdata, weights, gating)

def run_framed(settings: RunSettings, smaps, fullimage, rawdata, rescale):

	coord_vec, kdata_vec, weights_vec, gates = FivePointLoader.gate_ecg_method(rawdata[0], 
									    rawdata[1], rawdata[2], rawdata[3], settings.nframes)
	
	#pu.plot_gating(rawdata[3], gates)

	coord_vec, kdata_vec, weights_vec = lag.crop_kspace(coord_vec, kdata_vec, weights_vec, settings.im_size, 
		crop_factors=settings.crop_factors, prefovkmuls=settings.prefovkmuls, postfovkmuls=settings.postfovkmuls)

	if settings.shift != (0.0, 0.0, 0.0):
		print('Translating... ', end="")
		kdata_vec = lag.translate(coord_vec, kdata_vec, settings.shift)
		print('Done.')

	if rescale: # Implement mean shift
		pass

	images = torch.empty((settings.nframes, 5) + settings.im_size, dtype=torch.complex64)
	for i in range(settings.nframes):
		for j in range(5):
			images[i,j,...] = fullimage[j]
	
	_,_,_,smean = extract_mean_block(images, settings.im_size, (16,16,16))
	final_lamda = smean[0] * 8*1e-5

	svt_opts = SVTOptions(block_shapes=[16,16,16], block_strides=[16,16,16], block_iter=3, random=False, 
			nblocks=0, thresh=final_lamda, soft=True)

	gc.collect()
	torch.cuda.empty_cache()

	precond_pdhg = True
	precond_weights = False
	if precond_pdhg:
		preconds = []
		print('Preconditioning matrices...')
		for i in range(len(coord_vec)):
			print("\r", end="")
			print('Coil: ', i, '/', len(coord_vec), end="")
			P = precond.kspace_precond(smaps, coord_vec[i])
			P /= torch.mean(torch.abs(P))

			preconds.append(P.to(device='cpu', non_blocking=False))
		print(' Done.')

		framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='PDHG', sigma=Vector(preconds))
	elif precond_weights:
		preconds = []
		for i in range(len(coord_vec)):
			preconds.append(weights_vec[i].repeat(smaps.shape[0],1).unsqueeze(0))

		framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='PDHG', sigma=Vector(preconds))
	else:
		framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='PDHG')

	def plot_callback(image: Vector, iter):
		pu.image_nd(image.tensor.numpy())

	images = framed_recon.run(images, iter=45, callback=None)

	return images


if __name__ == '__main__':
	with torch.inference_mode():
		im_size = (256,256,256)
		shift = (-2*25.6, 0.0, 0.0)
		#im_size = (256,256,256)
		#shift = (-2*25.6, 0.0, 0.0)
		crop_factors = (1.1,1.1,1.1)
		prefovkmuls = (1.0,1.0,1.0)
		postfovkmuls = (1.0,1.0,1.0)

		settings = RunSettings(im_size, crop_factors, prefovkmuls, postfovkmuls, shift
			).set_nframes(10
			).set_smaps_filepath('D:/4DRecon/dat/dat2/SenseMapsCpp.h5'
			).set_rawdata_filepath('D:/4DRecon/MRI_Raw.h5')

		smaps, image, raw_data = run(settings)

		gc.collect()
		torch.cuda.empty_cache()

		images = run_framed(settings, smaps, image, raw_data, False)

		#pu.image_nd(images.numpy())

		store = True
		if store:
			print('Storing frame reconstructed')
			with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_real.h5', "w") as f:
				f.create_dataset('images', data=images.numpy())
