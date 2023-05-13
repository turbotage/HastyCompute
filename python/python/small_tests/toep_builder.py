import numpy as np

import itertools as itools
from numba import njit

import matplotlib.pyplot as plt

import torch
import torchkbnufft as tkbn

ni = 9
nx = ni
ny = ni
nz = ni

nf = 200

coords = 2*np.pi*np.random.rand(3,nf).astype(np.float32)
#weights = np.random.rand(nf).astype(np.float32)
weights = np.ones(nf).astype(np.float32)

AWA_i0 = np.zeros((nx,ny,nz), dtype=np.complex64)

@njit
def expi(p):
	return np.cos(p) + 1j*np.sin(p)

@njit
def compute_AHWA_element(coords, weights,i,j):
	val = 0
	for k in range(nf):
		cx = coords[0,k]
		cy = coords[1,k]
		cz = coords[2,k]
		
		val += weights[k] * expi(cx*(i[0]-j[0])) * expi(cy*(i[1]-j[1])) * expi(cz*(i[2]-j[2]))
	return val

@njit
def compute_AHWA_row(coords,weights,rowshape):
	nx = rowshape[0]
	ny = rowshape[1]
	nz = rowshape[2]
	output = np.zeros(rowshape, dtype=np.complex64)
	for x in range(nx):
		for y in range(ny):
			for z in range(nz):
				i = (x,y,z)
				output[i[0],i[1],i[2]] = compute_AHWA_element(coords,weights,i,(0,0,0))
	return output

@njit
def compute_AHWA_col(coords,weights,colshape):
	nx = colshape[0]
	ny = colshape[1]
	nz = colshape[2]
	output = np.zeros(colshape, dtype=np.complex64)
	for x in range(nx):
		for y in range(ny):
			for z in range(nz):
				j = (x,y,z)
				output[j[0],j[1],j[2]] = compute_AHWA_element(coords,weights,(0,0,0),j)
	return output

@njit
def compute_AHWA(coords,weights,input):
	nx = input.shape[0]
	ny = input.shape[1]
	nz = input.shape[2]
	output = np.zeros_like(input)
	for xi in range(nx):
		for yi in range(ny):
			for zi in range(nz):
				i = (xi,yi,zi)
				val = 0
				for xj in range(nx):
					for yj in range(ny):
						for zj in range(nz):
							j = (xj,yj,zj)
							elem = compute_AHWA_element(coords, weights,i,j)
							val += elem * input[j[0],j[1],j[2]]
				output[i[0],i[1],i[2]] = val
	return output

#@njit
def compute_pad_to_circulant_z(matrix):
	nx = matrix.shape[0]
	ny = matrix.shape[1]
	nz = matrix.shape[2]
	output = np.zeros((nx,ny,2*nz), dtype=np.complex64)

	for x in range(nx):
		# Create first row
		row_vector = np.zeros((2*nz,), dtype=np.complex64)
		row_vector[:nz] = matrix[x,0,:]
		row_vector[(nz+1):] = np.flip(np.conjugate(row_vector[1:nz]))
		output[x,0,:] = row_vector
		for y in range(1,ny):
			output[x,y,:] = np.roll(row_vector,y)
	return output[:,:,nz:]



if True:
	print('Before compute_AHWA')
	x = (np.random.rand(nx,ny,nz) + 1j*np.random.rand(nx,ny,nz))
	y = compute_AHWA(coords, weights, x)


	oned = np.zeros((2*nx,2*ny,2*nz), dtype=np.complex64)
	oned[nx,ny,nz] = 1

	#print('Before compute_AHWA oned')
	#diag = compute_AHWA(coords, weights, oned)

	print('Before compute_AHWA_col')
	ahwa_col = compute_AHWA_col(coords,weights,(nx,ny,nz))

	ahwa_pad = compute_pad_to_circulant_z(ahwa_col)

	diag2 = np.zeros_like(oned)

	diag_block = np.conjugate(ahwa_col)

	diag2[:nx,:ny,:nz] = diag_block 													# 0,0,0
	diag2[(nx+1):,:ny,(nz+1):] = np.conjugate(diag_block[:0:-1,:,:0:-1]) 				# 1,0,1
	diag2[:nx,(ny+1):,:nz] = np.conjugate(diag_block[:,:0:-1,:])						# 0,1,0
	diag2[(nx+1):,(ny+1):,(nz+1):] = np.conjugate(diag_block[:0:-1,:0:-1,:0:-1])		# 1,1,1
	#diag2 = diag
	#diag2 = np.fft.ifftshift(diag2)

	#diag2[nx:,0:ny,0:nz] = diag_block[::-1,:,:]		# 1,0,0 
	#diag2[0:nx,0:ny,nz:] = diag_block[:,:,::-1]		# 0,0,1
	#diag2[nx:,ny:,0:nz] = diag_block[::-1,::-1,:]	# 1,1,0
	#diag2[0:nx,ny:,nz:] = diag_block[:,::-1,::-1]	# 0,1,1

	#diag2 = np.fft.fftn(diag2)
	#diag2 /= (nx*ny*nz)

	#yn = np.fft.ifftn(diag2 * np.fft.fftn(x, (2*nx,2*ny,2*nz)))
	#yn = yn[0:nx,0:ny,0:nz]

	diag3 = tkbn.calc_toeplitz_kernel(omega=torch.tensor(coords), weights=
				   torch.tensor(weights).unsqueeze(0), im_size=(nx,ny,nz)) # without
	diag4 = diag3
	diag3 = np.fft.ifftn(diag3)

	yn = np.fft.ifftn(diag4 * np.fft.fftn(x, (2*nx,2*ny,2*nz)))
	yn = yn[0:nx,0:ny,0:nz]

	if True:
		y /= 2*np.sqrt(2)*(nx*ny*nz)
		#yn *= np.sqrt(nx*ny*nz)
		yn *= 2*np.sqrt(2)

		z = np.abs(y).flatten() / np.abs(yn).flatten()

		plt.figure()
		plt.plot(z, 'r-')
		plt.legend(['diag1', 'diag2'])
		plt.show()

	if False:
		y /= (nx*ny*nz)
		#yn *= np.sqrt(nx*ny*nz)
		yn *= 1.2732*2*np.pi
		#yn *= (nx*ny*nz)


		plt.figure()
		plt.plot(np.real(y).flatten(), 'r-')
		plt.plot(np.real(yn).flatten(), 'g-')
		plt.legend(['diag1', 'diag2'])
		plt.show()

		plt.figure()
		plt.plot(np.imag(y).flatten(), 'r-')
		plt.plot(np.imag(yn).flatten(), 'g-')
		plt.legend(['diag1', 'diag2'])
		plt.show()

		plt.figure()
		plt.plot(np.abs(y).flatten(), 'r-')
		plt.plot(np.abs(yn).flatten(), 'g-')
		plt.legend(['diag1', 'diag2'])
		plt.show()

		z = np.abs(y).flatten() / np.abs(yn).flatten()

		plt.figure()
		plt.plot(z, 'r-')
		plt.legend(['diag1', 'diag2'])
		plt.show()


	if True:
		diag2 /= (2 * np.sqrt(2)*(nx*ny*nz))
		diag3 *= 2 * np.sqrt(2)

		diag_block1 = diag2[(nx):,:ny,(nz):]
		diag_block2 = diag3[(nx):,:ny,(nz):]

		diag_block1 = diag_block1.flatten()
		diag_block2 = diag_block2.flatten()

		plt.figure()
		plt.plot(np.real(diag_block1), 'r-*')
		plt.plot(np.real(diag_block2), 'g-*')
		plt.legend(['diag1', 'diag2'])
		plt.show()

		plt.figure()
		plt.plot(np.imag(diag_block1), 'r-*')
		plt.plot(np.imag(diag_block2), 'g-*')
		plt.legend(['diag1', 'diag2'])
		plt.show()

		plt.figure()
		plt.plot(np.abs(diag_block1), 'r-*')
		plt.plot(np.abs(diag_block2), 'g-*')
		plt.legend(['diag1', 'diag2'])
		plt.show()

