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
	def __init__(self, coord_vec, kdata_vec, smaps, true_images):
		self.coord_vec = coord_vec
		self.kdata_vec = kdata_vec
		self.smaps = smaps
		self.true_images = true_images

	



def run_full(settings, data):

	full_coord_vec, full_kdata_vec = stack_frame_datas_fivepoint(data.coord_vec, data.kdata_vec)

	full_weights_vec = []
	for i in range(5):
		full_weights_vec.append(dcf.pipe_menon_dcf(
			(torch.tensor(settings.im_size) // 2).unsqueeze(-1) * full_coord_vec[i] / torch.pi,
			settings.im_size,
			max_iter=2					  
			))

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

	true_images = Vector(true_images.view((settings.nframes*5,1)+data.smaps.shape[1:]))
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

	if settings.solver == 'PDHG':
		precond_pdhg = False
		precond_density = False
		precond_ones = False
		if precond_pdhg:
			multicoil = False
			preconds = []
			print('Preconditioning matrices...')
			for i in range(len(data.coord_vec)):
				print("\r", end="")
				print('Coord: ', i, '/', len(data.coord_vec), end="")
				if multicoil:
					P = precond.kspace_precond(data.smaps, data.coord_vec[i])
				else:
					P = precond.kspace_precond(torch.ones_like(data.smaps), data.coord_vec[i])
				preconds.append(P.to(device='cpu', non_blocking=False))
			print(' Done.')

			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, solver='PDHG', sigma=Vector(preconds))
		elif precond_ones:
			preconds = []
			print('Preconditioning matrices...')
			for i in range(len(data.coord_vec)):
				print("\r", end="")
				print('Coord: ', i, '/', len(data.coord_vec), end="")
				preconds.append(torch.ones_like(data.kdata_vec[i], dtype=torch.float32))
			print(' Done.')

			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, solver='PDHG', sigma=Vector(preconds))
		else:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, solver='PDHG')
	elif settings.solver == 'GD':
		use_weights = False
		if use_weights:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, data.weights_vec, solver='GD')
		else:
			framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, solver='GD')
	elif settings.solver == 'Gridding':

		framed_recon = FivePointLLR(data.smaps, data.coord_vec, data.kdata_vec, svt_opts, weights_vec=data.weights_vec, solver='Gridding')
	else:
		raise RuntimeError('Unsupported solver type')

	def err_callback(image: Vector, iter):
		relerr = opalg.norm(mean_images + image - true_images) / true_images_norm
		print('RelErr: ', relerr)

	images = torch.zeros((settings.nframes, 5) + settings.im_size, dtype=torch.complex64)

	images = framed_recon.run(images, iter=45, callback=err_callback)

	pu.image_nd(images.numpy())

	images += mean_images.view((settings.nframes,5) + data.smaps.shape[1:])

	vel_images = etv.enc_to_vel_linear(images.numpy(), 1.0)

	pu.image_nd(vel_images)

	return images

def run_from_mean(settings: RunSettings):
	pass

def run_from_zero(settings: RunSettings):
	coord_vec, kdata_vec, smaps, true_images = FivePointLoader.load_simulated(
		settings.rawdata_filepath, settings.smaps_filepath, settings.true_filepath)
	
	settings.im_size = smaps.shape[1:]
	nframes = len(coord_vec) // 5
	
	#pu.image_nd(true_images.numpy())

	true_images = Vector(true_images.view((nframes*5,1)+smaps.shape[1:]))
	true_images_norm = opalg.norm(true_images)

	svt_opts = SVTOptions(block_shapes=[16,16,16], block_strides=[16,16,16], block_iter=6, random=False, 
			nblocks=0, thresh=0.001, soft=True)

	print('Kdata points per frame: ', coord_vec[0].shape[1])
	print('Kdata points kdata per pixel: ', coord_vec[0].shape[1] / math.prod(smaps.shape[1:]))

	if settings.solver == 'PDHG':
		precond_pdhg = False
		precond_density = False
		precond_ones = False
		if precond_pdhg:
			multicoil = False
			preconds = []
			print('Preconditioning matrices...')
			for i in range(len(coord_vec)):
				print("\r", end="")
				print('Coord: ', i, '/', len(coord_vec), end="")
				if multicoil:
					P = precond.kspace_precond(smaps, coord_vec[i])
				else:
					P = precond.kspace_precond(torch.ones_like(smaps), coord_vec[i])
				preconds.append(P.to(device='cpu', non_blocking=False))
			print(' Done.')

			framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='PDHG', sigma=Vector(preconds))
		elif precond_density:
			preconds = []
			print('Preconditioning matrices...')
			cudev = torch.device('cuda:0')
			for i in range(len(coord_vec)):
				print("\r", end="")
				print('Coord: ', i, '/', len(coord_vec), end="")
				P = tkbn.calc_density_compensation_function(ktraj=coord_vec[i].to(cudev), im_size=settings.im_size)
				P = P.repeat(1,smaps.shape[0],1)
				preconds.append(P.to(device='cpu', non_blocking=False))
			print(' Done.')

			framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='PDHG', sigma=Vector(preconds))
		elif precond_ones:
			preconds = []
			print('Preconditioning matrices...')
			for i in range(len(coord_vec)):
				print("\r", end="")
				print('Coord: ', i, '/', len(coord_vec), end="")
				preconds.append(torch.ones_like(kdata_vec[i], dtype=torch.float32))
			print(' Done.')

			framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='PDHG', sigma=Vector(preconds))
		else:
			framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='PDHG')
	elif settings.solver == 'GD':
		use_weights = True
		if use_weights:
			weights_vec = []
			print('Preconditioning matrices...')
			cudev = torch.device('cuda:0')
			for i in range(len(coord_vec)):
				print("\r", end="")
				print('Coord: ', i, '/', len(coord_vec), end="")
				weights = tkbn.calc_density_compensation_function(ktraj=coord_vec[i].to(cudev), im_size=settings.im_size)
				weights = torch.sqrt(weights)
				#weights = precond.pipe_menon_dcf(coord_vec[i], settings.im_size)
				weights_vec.append(weights.squeeze(0).to(device='cpu', non_blocking=False))
			print(' Done.')

			framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, weights_vec, solver='GD')
		else:
			framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, solver='GD')
	elif settings.solver == 'Gridding':
		weights_vec = []
		print('Weight matrices...')
		cudev = torch.device('cuda:0')
		for i in range(len(coord_vec)):
			print("\r", end="")
			print('Coord: ', i, '/', len(coord_vec), end="")
			weights = tkbn.calc_density_compensation_function(ktraj=coord_vec[i].to(cudev), im_size=settings.im_size)
			#weights = torch.sqrt(weights)
			#weights = precond.pipe_menon_dcf(coord_vec[i], settings.im_size)
			weights_vec.append(weights.squeeze(0).to(device='cpu', non_blocking=False))
		print(' Done.')

		framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts, weights_vec=weights_vec, solver='Gridding')
	else:
		raise RuntimeError('Unsupported solver type')

	def err_callback(image: Vector, iter):
		if iter==-1:
			relerr = torch.norm(image - true_images.get_tensor().view((nframes,5) + smaps.shape[1:])) / true_images_norm
		else:	
			relerr = opalg.norm(image - true_images) / true_images_norm
		print('RelErr: ', relerr)

	images = torch.zeros((nframes, 5) + settings.im_size, dtype=torch.complex64)

	images = framed_recon.run(images, iter=45, callback=err_callback)

	err_callback(images, -1)

	pu.image_nd(images.numpy())

	return images

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

		img_full = run_full(settings, data)

		run_from_difference(settings)


