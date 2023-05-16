
import time
import sys
import os

import numpy as np

import torch
import torchkbnufft as tkbn

import matplotlib.pyplot as plt
import finufft

N1 = 6
N2 = 6
N3 = 6
NX = N1*N2*N3
NF = NX

coord = np.empty((3,NF), dtype=np.float32)
l = 0
for x in range(N1):
	for y in range(N2):
		for z in range(N3):
			kx = -np.pi + x * 2 * np.pi / N1
			ky = -np.pi + y * 2 * np.pi / N2
			kz = -np.pi + z * 2 * np.pi / N3

			coord[0,l] = kx
			coord[1,l] = ky
			coord[2,l] = kz

			l += 1

coord[0,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N1
coord[1,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N2
coord[2,:] = -np.pi + 2 * np.pi * np.random.rand(NF).astype(np.float32) / N3

coord = np.transpose(coord, (1,0))


psf1 = np.array([0])
if True:
	unity_vector = np.zeros((2*N1,2*N2,2*N3), dtype=np.complex64)
	unity_vector[0,0,0] = 1
	#print(unity_vector)
	unity_vector = np.fft.ifftshift(unity_vector)
	nuftt2_out = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], unity_vector) / np.sqrt(NX)
	nufft1_out = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], nuftt2_out, (2*N1,2*N2,2*N3)) / np.sqrt(NX)
	psf1 = np.fft.fftshift(nufft1_out)
	#psf2 = nufft1_out
	psf1 = np.fft.fftn(psf1)
	#psf1 = np.fft.fftshift(psf1)

psf2 = np.array([0])
if True:
	psf2 = tkbn.calc_toeplitz_kernel(omega=torch.tensor(coord).transpose(0,1), im_size=(N1,N2,N3))
	psf2 = psf2.squeeze(0).squeeze(0).numpy() * 8

psf1_p = psf1.flatten()
psf2_p = psf2.flatten()


plt.figure()
plt.plot(np.abs(psf1_p), 'r-')
plt.plot(np.abs(psf2_p), 'g-')
plt.show()

plt.figure()
plt.plot(np.real(psf1_p), 'r-')
plt.plot(np.real(psf2_p), 'g-')
plt.show()

plt.figure()
plt.plot(np.imag(psf1_p), 'r-')
plt.plot(np.imag(psf2_p), 'g-')
plt.show()


x = (np.random.rand(N1,N2,N3) + 1j*np.random.rand(N1,N2,N3)).astype(np.complex64)

y1 = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], x) /  np.sqrt(NX)
y1 = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], y1, (N1,N2,N3)) / np.sqrt(NX)


#psf2 = np.fft.ifftshift(psf2)
y2 = np.fft.ifftn(psf2 * np.fft.fftn(x, (2*N1,2*N2,2*N3)))
y2 = y2[:N1,:N2,:N3]

y1_p = y1.flatten()
y2_p = y2.flatten()


plt.figure()
plt.plot(np.abs(y1_p), 'r-')
plt.plot(np.abs(y2_p), 'g-')
plt.show()

plt.figure()
plt.plot(np.real(y1_p), 'r-')
plt.plot(np.real(y2_p), 'g-')
plt.show()

plt.figure()
plt.plot(np.imag(y1_p), 'r-')
plt.plot(np.imag(y2_p), 'g-')
plt.show()

z1 = np.abs(y1_p) / np.abs(x.flatten())
z2 = np.abs(y2_p) / np.abs(x.flatten())
plt.figure()
plt.plot(z1, 'r-')
plt.plot(z2, 'g-')
plt.show()
print(np.mean(z))


