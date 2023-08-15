import torch

from hastypy.base.opalg import Vector

from hastypy.base.recon import FivePointLLR
import hastypy.base.load_and_gate as lag
from hastypy.base.load_and_gate import FivePointLoader

import hastypy.util.plot_utility as pu

from hastypy.base.recon import FivePointFULL

def run():
	im_size = (256,256,256)
	crop_factor = 1.5
	prefovkmul = 1.0
	postfovkmul = 1.0

	shift = (-2*25.6, 0.0, 0.0)

	smaps = FivePointLoader.load_smaps('D:/4DRecon/dat/dat2/SenseMapsCpp.h5', im_size)
	smaps = torch.permute(smaps, (0,3,2,1))

	coords, kdata, weights, gating = FivePointLoader.load_raw('D:/4DRecon/MRI_Raw.h5')
	#normalize kdata
	kdata /= 0.5*(torch.mean(torch.real(kdata)) + torch.mean(torch.real(kdata)))

	coord_vec, kdata_vec, weights_vec = FivePointLoader.load_as_full(coords, kdata, weights)

	coord_vec, kdata_vec, weights_vec = lag.crop_kspace(coord_vec, kdata_vec, weights_vec, im_size, 
		crop_factor=crop_factor, prefovkmul=prefovkmul, postfovkmul=postfovkmul)

	kdatapoints = 0
	for kdatai in kdata_vec:
		kdatapoints += kdatai.shape[2]
	print('Num kdatapoints', kdatapoints)

	if shift != (0.0, 0.0, 0.0):
		print('Translate:')
		kdata_vec = lag.translate(coord_vec, kdata_vec, shift)

	full_recon = FivePointFULL(smaps, coord_vec, kdata_vec, weights_vec, lamda=2000000.0)

	image = torch.zeros((5,1) + im_size, dtype=torch.complex64)

	def plot_callback(image: Vector, iter):
		pu.image_nd(image.tensor.numpy())

	full_recon.run(image, iter=5, callback=plot_callback)

	pu.image_nd(image.numpy())


if __name__ == '__main__':
	with torch.inference_mode():
		run()