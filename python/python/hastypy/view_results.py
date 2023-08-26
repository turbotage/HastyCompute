
import numpy as np
import h5py

import hastypy.util.plot_utility as pu
import hastypy.util.image_creation as ic

import torchkbnufft as tkbn

img = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\images_mvel_15f_cropped_interpolated.h5', "r") as f:
	img = f['images'][()]

#img_full = np.array([0])
#with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_reconstructed_weighted.h5', "r") as f:#
#	img_full = f['images'][()].squeeze(1)
#	#img_full = f['images'][()]

#img_mean = np.mean(img, axis=0)

pu.image_nd(img)

mvel = np.sqrt(img[:,1,...]**2 + img[:,2,...]**2 + img[:,3,...]**2)

pu.image_nd(mvel)

#pu.image_4d(np.abs(img_full))

#pu.image_4d(np.abs(img_mean))


