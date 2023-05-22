
import numpy as np
import h5py

import plot_utility as pu
import image_creation as ic

import torchkbnufft as tkbn

img = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_15f_cropped_interpolated.h5', "r") as f:
	img = f['images'][()]

img_full = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed.h5', "r") as f:
	img_full = f['images'][()].squeeze(1)
	#img_full = f['images'][()]

img_full2 = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed2.h5', "r") as f:
	img_full2 = f['images'][()]

img_mean = np.mean(img, axis=0)

pu.image_4d(np.clip(np.abs(img_full), 0, 1e3))
pu.image_4d(np.clip(np.abs(img_full2), 1e5, 1e7))

pu.image_4d(np.abs(img_mean))

pu.image_4d(np.abs(img_full - img_mean))


