import torch
import h5py
import math

import gc

import runner

from hastypy.base.opalg import Vector, MaxEig
import hastypy.base.opalg as opalg

from hastypy.base.recon import stack_frame_datas_fivepoint, FivePointFULL, FivePointLLR
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
import hastypy.base.enc_to_vel as etv

import hastypy.opt.cuda.sigpy.dcf as dcf


class RunSettings:
	def __init__(self, im_size=None, crop_factors=None, prefovkmuls=None, postfovkmuls=None, shift=None, solver='GD'):
		self.im_size = im_size
		self.crop_factors = crop_factors
		self.prefovkmuls = prefovkmuls
		self.postfovkmuls = postfovkmuls
		self.shift = shift
		self.solver = solver
		
		self.smaps_filepath = None
		self.rawdata_filepath = None
		self.true_filepath = None

	def set_smaps_filepath(self, path):
		self.smaps_filepath = path
		return self

	def set_rawdata_filepath(self, path):
		self.rawdata_filepath = path
		return self

	def set_true_filepath(self, path):
		self.true_filepath = path
		return self

	def set_nframes(self, nframes):
		self.nframes = nframes
		return self

class Data:
	def __init__(self, coord_vec, kdata_vec, weights_vec, smaps, true_images):
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.weights_vec = weights_vec
		self.smaps = smaps
		self.true_images = true_images

	



def run_full(settings, data):

	full_coord_vec, full_kdata_vec = stack_frame_datas_fivepoint(data.coord_vec, data.kdata_vec)

	full_weights_vec = []
	for i in range(5):
		dcfw = dcf.pipe_menon_dcf(
			(torch.tensor(settings.im_size) // 2).unsqueeze(-1) * full_coord_vec[i] / torch.pi,
			settings.im_size,
			max_iter=30					  
			)

		full_weights_vec.append(dcfw / torch.mean(dcfw))

	images_full = FivePointFULL(data.smaps, full_coord_vec, full_kdata_vec, full_weights_vec, lamda=0.00001).run(
		torch.zeros((5,1) + settings.im_size, dtype=torch.complex64), iter=10)
	
	return images_full



def run_from_difference(settings: RunSettings, data: Data, img_full):
	cudev = torch.device('cuda:0')
	smaps_cu = data.smaps.to(cudev)
	coils = [i for i in range(data.smaps.shape[0])]
	for frame in range(settings.nframes):
		for enc in range(5):
			kdata = SenseT(data.coord_vec[frame*5 + enc].to(cudev), smaps_cu, coils).apply(img_full[enc].to(cudev)).cpu()
			data.kdata_vec[frame*5 + enc] -= kdata
	del smaps_cu, coils
	gc.collect()
	torch.cuda.empty_cache()

	true_images = Vector(data.true_images.view((settings.nframes*5,1)+data.smaps.shape[1:]))
	true_images /= opalg.mean(true_images)
	true_images_norm = opalg.norm(true_images)

	mean_images = torch.empty((settings.nframes,5) + settings.im_size, dtype=torch.complex64)
	for frame in range(settings.nframes):
		for enc in range(5):
			mean_images[frame,enc,...] = img_full[enc,0,...]
	mean_images = mean_images.view((settings.nframes*5,1) + settings.im_size)

	svt_opts = SVTOptions(block_shapes=[16,16,16], block_strides=[16,16,16], block_iter=6, random=False, 
			nblocks=0, thresh=0.001, soft=True)

	print('Kdata points per frame: ', data.coord_vec[0].shape[1])
	print('Kdata points kdata per pixel: ', data.coord_vec[0].shape[1] / math.prod(data.smaps.shape[1:]))

	if settings.solver == 'GD':
		use_weights = False
		if use_weights:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, data.weights_vec, solver='GD')
		else:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, solver='GD')
	elif settings.solver == 'Gridding':
		framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, weights_vec=data.weights_vec, solver='Gridding')
	else:
		raise RuntimeError('Unsupported solver type')

	relerrs = []

	def err_callback(image: Vector, iter):
		img = mean_images + image
		img /= opalg.mean(img)
		relerr = opalg.norm(img - true_images) / true_images_norm
		relerrs.append(relerr)
		print('RelErr: ', relerr)

	images = torch.zeros((settings.nframes, 5) + settings.im_size, dtype=torch.complex64)

	images = framed_recon.run(images, iter=45, callback=err_callback)

	#pu.image_nd(images.numpy())
	#images += mean_images.view((settings.nframes,5) + data.smaps.shape[1:])
	#return images

	return relerrs

def run_from_mean(settings: RunSettings, data: Data, img_full):
	true_images = Vector(data.true_images.view((settings.nframes*5,1)+data.smaps.shape[1:]))
	true_images /= opalg.mean(true_images)
	true_images_norm = opalg.norm(true_images)

	svt_opts = SVTOptions(block_shapes=[16,16,16], block_strides=[16,16,16], block_iter=6, random=False, 
			nblocks=0, thresh=0.001, soft=True)

	print('Kdata points per frame: ', data.coord_vec[0].shape[1])
	print('Kdata points kdata per pixel: ', data.coord_vec[0].shape[1] / math.prod(data.smaps.shape[1:]))

	if settings.solver == 'GD':
		use_weights = False
		if use_weights:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, data.weights_vec, solver='GD')
		else:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, solver='GD')
	elif settings.solver == 'Gridding':

		framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, weights_vec=data.weights_vec, solver='Gridding')
	else:
		raise RuntimeError('Unsupported solver type')

	relerrs = []

	def err_callback(image: Vector, iter):
		image /= opalg.mean(image)
		relerr = opalg.norm(image - true_images) / true_images_norm
		relerrs.append(relerr)
		print('RelErr: ', relerr)

	images = torch.zeros((settings.nframes, 5) + settings.im_size, dtype=torch.complex64)
	for frame in range(settings.nframes):
		for enc in range(5):
			images[frame,enc,...] = img_full[enc,0,...]

	images = framed_recon.run(images, iter=45, callback=err_callback)

	#pu.image_nd(images.numpy())
	#return images

	return relerrs

def run_from_zero(settings: RunSettings, data: Data):
	true_images = Vector(data.true_images.view((settings.nframes*5,1)+data.smaps.shape[1:]))
	true_images /= opalg.mean(true_images)
	true_images_norm = opalg.norm(true_images)

	svt_opts = SVTOptions(block_shapes=[16,16,16], block_strides=[16,16,16], block_iter=6, random=False, 
			nblocks=0, thresh=0.001, soft=True)

	print('Kdata points per frame: ', data.coord_vec[0].shape[1])
	print('Kdata points kdata per pixel: ', data.coord_vec[0].shape[1] / math.prod(data.smaps.shape[1:]))

	if settings.solver == 'GD':
		use_weights = False
		if use_weights:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, data.weights_vec, solver='GD')
		else:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, solver='GD')
	elif settings.solver == 'Gridding':

		framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, weights_vec=data.weights_vec, solver='Gridding')
	else:
		raise RuntimeError('Unsupported solver type')

	relerrs = []
	def err_callback(image: Vector, iter):
		image /= opalg.mean(image)
		relerr = opalg.norm(image - true_images) / true_images_norm
		relerrs.append(relerr)
		print('RelErr: ', relerr)

	images = torch.zeros((settings.nframes, 5) + settings.im_size, dtype=torch.complex64)

	images = framed_recon.run(images, iter=45, callback=err_callback)

	#pu.image_nd(images.numpy())
	#return images

	return relerrs

if __name__ == '__main__':
	with torch.inference_mode():
		settings = RunSettings(solver='GD',
			).set_smaps_filepath('D:/4DRecon/dat/dat2/SenseMapsCpp_cropped.h5'
			).set_rawdata_filepath('D:/4DRecon/dat/dat2/simulated_coords_kdatas.h5'
			).set_true_filepath('D:/4DRecon/dat/dat2/images_encs_20f_cropped_interpolated.h5')
		
		data = Data(*FivePointLoader.load_simulated(
			settings.rawdata_filepath, settings.smaps_filepath, settings.true_filepath))

		settings.im_size = data.smaps.shape[1:]
		settings.set_nframes(len(data.coord_vec) // 5)

		calc_full = True
		if calc_full:
			img_full = run_full(settings, data)
			with h5py.File('D:/4DRecon/dat/dat2/simulated_full_img.h5', 'w') as f:
				f.create_dataset('img_full', data=img_full)
		else:
			with h5py.File('D:/4DRecon/dat/dat2/simulated_full_img.h5', 'r') as f:
				img_full = f['img_full'][()]


		difference_err = run_from_difference(settings, data, img_full)
		mean_err = run_from_mean(settings, data, img_full)
		zero_err = run_from_zero(settings, data)

		


