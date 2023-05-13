from warnings import filterwarnings

from random import seed
from random import random
# seed random number generator
seed(1)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
from skimage.data import shepp_logan_phantom
import time

filterwarnings("ignore") # ignore floor divide warnings
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
nf = 40000
nx = 64
ncoil = 1


coords = -np.pi + 2*np.pi*np.random.rand(3,nf).astype(np.float32)
image = (np.random.rand(ncoil,nx,nx,nx).astype(np.float32) + 1j*np.random.rand(ncoil,nx,nx,nx).astype(np.float32)).astype(np.complex64)

# convert k-space trajectory to a tensor
coords = torch.tensor(coords).to(device).requires_grad_(False)
#print('coords shape: {}'.format(coords.shape))
image = torch.tensor(image).to(device).unsqueeze_(0).requires_grad_(False)
#print('image shape: {}'.format(image.shape))

im_size = (nx,nx,nx)

toep_ob = tkbn.ToepNufft()

dcomp = tkbn.calc_density_compensation_function(ktraj=coords, im_size=im_size)


normal_kernel = tkbn.calc_toeplitz_kernel(coords, im_size, norm="ortho")  # without density compensation

print('normal real: ', torch.real(normal_kernel).abs().max())
print('normal imag: ', torch.imag(normal_kernel).abs().max())

