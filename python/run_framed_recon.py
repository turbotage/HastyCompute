import torch
import h5py

import gc

from hastypy.base.opalg import Vector

from hastypy.base.recon import FivePointLLR
import hastypy.base.load_and_gate as lag
from hastypy.base.load_and_gate import FivePointLoader

import hastypy.util.plot_utility as pu

from hastypy.base.recon import FivePointFULL
from hastypy.base.proximal import SVTOptions
from hastypy.base.svt import extract_mean_block

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
	smaps = FivePointLoader.load_smaps(settings.smaps_filepath, settings.im_size)
	smaps = torch.permute(smaps, (0,3,2,1))

	pu.image_nd(smaps.numpy())

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

	full_recon = FivePointFULL(smaps, coord_vec, kdata_vec, weights_vec, lamda=10.0)

	image = torch.zeros((5,1) + settings.im_size, dtype=torch.complex64)

	def plot_callback(image: Vector, iter):
		pu.image_nd(image.tensor.numpy())

	gc.collect()
	torch.cuda.empty_cache()

	image = full_recon.run(image, iter=15, callback=None)

	pu.image_nd(image.numpy())

	return smaps, image, (coords, kdata, weights, gating)

def run_framed(settings: RunSettings, smaps, fullimage, rawdata, rescale):

	coord_vec, kdata_vec, _, gates = FivePointLoader.gate_ecg_method(rawdata[0], 
									    rawdata[1], None, rawdata[3], settings.nframes)
	
	coord_vec, kdata_vec, _ = lag.crop_kspace(coord_vec, kdata_vec, None, settings.im_size, 
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
	final_lamda = smean[0] * 0.00005

	svt_opts = SVTOptions(block_shapes=[16,16,16], block_strides=[16,16,16], block_iter=4, random=False, 
			nblocks=0, thresh=final_lamda, soft=True)

	framed_recon = FivePointLLR(smaps, coord_vec, kdata_vec, svt_opts)

	gc.collect()
	torch.cuda.empty_cache()

	def plot_callback(image: Vector, iter):
		pu.image_nd(image.tensor.numpy())

	images = framed_recon.run(images, iter=80, callback=None)

	return images


if __name__ == '__main__':
	with torch.inference_mode():
		im_size = (160,160,160)
		shift = (-2*16, 0.0, 0.0)
		#im_size = (256,256,256)
		#shift = (-2*25.6, 0.0, 0.0)
		crop_factors = (1.1,1.1,1.1)
		prefovkmuls = (1.0,1.0,1.0)
		postfovkmuls = (1.0,1.0,1.0)

		settings = RunSettings(im_size, crop_factors, prefovkmuls, postfovkmuls, shift
			).set_nframes(15
			).set_smaps_filepath('D:/4DRecon/dat/dat2/SenseMapsCpp.h5'
			).set_rawdata_filepath('D:/4DRecon/MRI_Raw.h5')

		smaps, image, raw_data = run(settings)

		gc.collect()
		torch.cuda.empty_cache()

		images = run_framed(settings, smaps, image, raw_data, False)

		pu.image_nd(images.numpy())

		store = True
		if store:
			print('Storing frame reconstructed')
			with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_real.h5', "w") as f:
				f.create_dataset('images', data=images.numpy())
