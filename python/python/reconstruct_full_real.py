import torch
import numpy as np
import math

import cupy as cp
import cupyx

import plot_utility as pu
import simulate_mri as simri

import h5py
import torchkbnufft as tkbn

from torch_linop import TorchLinop, TorchScaleLinop
from torch_grad_methods import TorchCG, TorchGD
from torch_maxeig import TorchMaxEig

import reconstruct_util as ru

nframes = 30
nenc = 5

def run():
	print('Loading coords, kdata, weights')
	smaps, coord, kdata, weights, gating = ru.load_real()
	print('Gating')
	coord_vec, kdata_vec, weights_vec, gates = ru.gate(coord, kdata, weights, gating, nframes)
	print('Stack frames for full image reconstruction')
	coord_vec_full, kdata_vec_full, weights_vec_full = ru.gated_full(coord_vec, kdata_vec, weights_vec, nframes)
	print('Create full image diagonals and rhs')
	diagonals, rhs = ru.load_real_full_diag_rhs(
		smaps, coord_vec_full, kdata_vec_full, weights_vec_full, use_weights=True, root=0)
	print('Starting full image reconstruction')
	image = ru.reconstruct_cg_full(diagonals, rhs, smaps, nenc, iter=50, lamda=0, images=None, plot=False)


with torch.inference_mode():
	run()





