

import time
import sys
import os

import numpy as np
import cupy as cp

import pycompute.cuda.sigpy.fourier_linops as fulinops
from pycompute.cuda.sigpy.fourier_linops import NUFFT, NUFFTAdjoint, NormalNUFFT

import pycompute.cuda.sigpy.fourier as fourier

import pycompute.cuda.sigpy.linop as linop
from pycompute.cuda.sigpy.linop import Multiply




# SMOOTH_L0 [ WAVELET [ FFT [ENCODES_TO_VELOCITY [ IMAGE ] ] ] ]
class SL0_WFE():
	def __init__(self, y, x0, mps, coords, batch_size):

		# Sense operator

		self.ishape = mps.shape[1:]

		self.n_frames = x0.shape[3]
		self.x = x0
		
		self.mul_mps = [Multiply(mps[0].shape, mps[i]) for i in range(len(mps))]
		self.nufft_ops = [NUFFT(self.mul_mps[0].oshape, coords[i], oversamp=2.0) for i in range(len(coords))]
		self.normal_nufft_ops = [NormalNUFFT(self.mul_mps[0].oshape, coords[i], oversamp=2.0) for i in range(len(coords))]

		self.sigma_max = 1
		self.sigma_min = 1e-3

		self.sigma_k = self.sigma_max

		self.batch_size = batch_size


	def sparsity_gradient(self):



	def update():
		pass




