
import sys
from pathlib import Path

def setup_test_environment():
	sys.path.append(str(Path(__file__).parent.parent.parent))

setup_test_environment()

import torch
from scipy.signal import tukey

import hastypy.util.plot_utility as pu

t1 = tukey(128, 0.9)
t2 = tukey(128, 0.9)
t3 = tukey(16, 0.9)

window_prod = torch.cartesian_prod(torch.tensor(t1), torch.tensor(t2), torch.tensor(t3))
window = (window_prod[:,0] * window_prod[:,1] * window_prod[:,2]).reshape(128,128,16)


#pu.image_nd(window.numpy()) 

fwindow = torch.fft.fftn(window)


pu.image_nd(fwindow.numpy())


fwindow = torch.fft.ifftshift(window)
fwindow = torch.fft.fftn(fwindow)
fwindow = torch.fft.fftshift(fwindow)

pu.image_nd(fwindow.numpy())
