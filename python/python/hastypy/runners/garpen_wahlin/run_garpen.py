import torch
import h5py

import gc

import garpen_runner

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
from hastypy.base.sense import SenseT

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
	smaps /= torch.mean(torch.abs(smaps))
	pu.image_nd(smaps.numpy())
	print('Done estimating coil sensitivities')
	
	with h5py.File('D:\\4DRecon\\garpen\\my_smaps.h5', "w") as f:
		f.create_dataset('Maps', data=smaps.numpy())

	full_recon = FivePointFULL(smaps, coord_vec, kdata_vec, weights_vec, lamda=0.0005)

	image = torch.zeros((5,1) + settings.im_size, dtype=torch.complex64)

	def plot_callback(image: Vector, iter):
		pu.image_nd(image.tensor.numpy())

	gc.collect()

	torch.cuda.empty_cache()

	image = full_recon.run(image, iter=30, callback=None)

	#pu.image_nd(image.numpy())

	return smaps, image, (coords, kdata, weights, gating)

def run_framed(settings: RunSettings, smaps, fullimage, rawdata, rescale):

	coord_vec, kdata_vec, weights_vec, gates = FivePointLoader.gate_ecg_method(rawdata[0], 
									    rawdata[1], rawdata[2], rawdata[3], settings.nframes)
	
	coord_vec, kdata_vec, weights_vec = lag.crop_kspace(coord_vec, kdata_vec, weights_vec, settings.im_size, 
		crop_factors=settings.crop_factors, prefovkmuls=settings.prefovkmuls, postfovkmuls=settings.postfovkmuls)

	if settings.shift != (0.0, 0.0, 0.0):
		print('Translating... ', end="")
		kdata_vec = lag.translate(coord_vec, kdata_vec, settings.shift)
		print('Done.')

	cudev = torch.device('cuda:0')
	smaps_cu = smaps.to(cudev)
	coils = [i for i in range(smaps.shape[0])]
	for frame in range(settings.nframes):
		for enc in range(5):
			kdata = SenseT(coord_vec[frame*5 + enc].to(cudev), smaps_cu, coils).apply(fullimage[enc].to(cudev)).cpu()
			kdata_vec[frame*5 + enc] -= kdata
	del smaps_cu


	gc.collect()
	torch.cuda.empty_cache()

	svt_opts = SVTOptions(block_shapes=[16,16,16], block_strides=[16,16,16], block_iter=3, random=False, 
			nblocks=0, thresh=5e-6, soft=True)

	precond_pdhg = False
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
		framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='GD')

	def plot_callback(image: Vector, iter):
		pu.image_nd(image.tensor.view((settings.nframes,5) + settings.im_size).numpy())

	images = framed_recon.run(torch.zeros((settings.nframes,5) + settings.im_size, dtype=torch.complex64), iter=100, callback=None)

	return images


if __name__ == '__main__':

	base_path = 'D:/4Drecon/Garpen/Ena/'
	resol = 160 
	file_ending = '_' + str(resol)

	with torch.inference_mode():
		im_size = (resol,resol,resol)
		shift = (0.5*(resol / 10.0), 0.0, 0.0)
		#shift = (0.0, 0.0, 0.0)

		#im_size = (256,256,256)
		#shift = (-2*25.6, 0.0, 0.0)
		crop_factors = (1.0,1.0,1.0)
		prefovkmuls = (1.0,1.0,1.0)
		postfovkmuls = (1.0,1.0,1.0)

		settings = RunSettings(im_size, crop_factors, prefovkmuls, postfovkmuls, shift
			).set_nframes(20
			).set_smaps_filepath(base_path + 'SenseMaps.h5'
			).set_rawdata_filepath(base_path + 'MRI_Raw.h5')

		smaps, image, raw_data = run(settings)
		store = True
		if store:
			print('Storing full image')
			with h5py.File(base_path + 'my_full_real' + file_ending + '.h5', "w") as f:
				f.create_dataset('image', data=image)

		gc.collect()
		torch.cuda.empty_cache()

		images = run_framed(settings, smaps, image, raw_data, False)

		for frame in range(settings.nframes):
			for enc in range(5):
				images[frame,enc,...] += image[enc,0,...]

		images = images.numpy()

		if store:
			print('Storing frame reconstructed')
			with h5py.File(base_path + 'my_framed_real' + file_ending + '.h5', "w") as f:
				f.create_dataset('images', data=images)
		
		pu.image_nd(images)
