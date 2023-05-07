import numpy as np
import h5py

import plot_utility as pu

with h5py.File('D:\\4DRecon\\images.h5', "r") as f:
	img = f['images'][()]

	pu.image_5d(img[:,1:,:,:,:])
	
