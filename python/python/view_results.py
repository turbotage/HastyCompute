
import numpy as np
import h5py

import plot_utility as pu
import image_creation as ic

import torchkbnufft as tkbn

img = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_15f_cropped_interpolated.h5', "r") as f:
	img = f['images'][()]

#img_full = np.array([0])
#with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed_weighted.h5', "r") as f:#
#	img_full = f['images'][()].squeeze(1)
#	#img_full = f['images'][()]

#img_mean = np.mean(img, axis=0)

pu.image_5d(img)

#pu.image_4d(np.abs(img_full))

#pu.image_4d(np.abs(img_mean))


