
import torch
import numpy as np
import h5py

import math
import random
import gc

import plot_utility as pu
import image_creation as ic

import torchkbnufft as tkbn


#img_full = np.array([0])
#with h5py.File('D:\\4DRecon\\dat\\dat2\\my_full_real_reconstructed_weighted.h5', "r") as f:
#	img_full = f['images'][()].squeeze(1)
#	#img_full = f['images'][()]

img_full = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_real.h5', "r") as f:
	img_full = f['images'][()]

img_full *= np.exp(-1j*np.angle(img_full[:,0,...]))[:,np.newaxis,...]

pu.image_nd(img_full)


