import numpy as np

img = (np.random.rand(5,3,32,32,16) + 1j*np.random.rand(5,3,32,32,16)).astype(np.complex64)

import plot_utility as pu

pu.image_nd(img)