
import cupy as cp
import numpy as np
import math

import hastypy.opt.cuda.sigpy.util as su
import hastypy.opt.cuda.sigpy.gridding as sg
import hastypy.opt.cuda.sigpy.interpolate as si

def _fftc(input, oshape=None, axes=None, norm='ortho'):

    ndim = input.ndim
    axes = su._normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = su._resize(input, oshape)
    tmp = cp.fft.ifftshift(tmp, axes=axes)
    tmp = cp.fft.fftn(tmp, axes=axes, norm=norm)
    output = cp.fft.fftshift(tmp, axes=axes)
    return output


def _ifftc(input, oshape=None, axes=None, norm='ortho'):
    ndim = input.ndim
    axes = su._normalize_axes(axes, ndim)

    if oshape is None:
        oshape = input.shape

    tmp = su._resize(input, oshape)
    tmp = cp.fft.ifftshift(tmp, axes=axes)
    tmp = cp.fft.ifftn(tmp, axes=axes, norm=norm)
    output = cp.fft.fftshift(tmp, axes=axes)
    return output


def fft(input, oshape=None, axes=None, center=True, norm='ortho'):
	if not np.issubdtype(input.dtype, np.complexfloating):
		input = input.astype(np.complex64)

	if center:
		output = _fftc(input, oshape=oshape, axes=axes, norm=norm)
	else:
		output = cp.fft.fftn(input, s=oshape, axes=axes, norm=norm)

	if np.issubdtype(input.dtype,
						np.complexfloating) and input.dtype != output.dtype:
		output = output.astype(input.dtype, copy=False)

	return output


def ifft(input, oshape=None, axes=None, center=True, norm='ortho'):
	if not np.issubdtype(input.dtype, np.complexfloating):
		input = input.astype(np.complex64)

	if center:
		output = _ifftc(input, oshape=oshape, axes=axes, norm=norm)
	else:
		output = cp.fft.ifftn(input, s=oshape, axes=axes, norm=norm)

	if np.issubdtype(input.dtype,
						np.complexfloating) and input.dtype != output.dtype:
		output = output.astype(input.dtype)

	return output


def nufft(input, coord, oversamp=1.25, width=4, center=True):
	ndim = coord.shape[-1]
	beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
	os_shape = su._get_oversamp_shape(input.shape, ndim, oversamp)

	output = input.copy()

	# Apodize
	su._apodize(output, ndim, oversamp, width, beta)

	# Zero-pad
	output /= np.prod(input.shape[-ndim:], dtype=np.int64)**0.5
	output = su._resize(output, os_shape)

	# FFT
	output = fft(output, axes=range(-ndim, 0), norm=None, center=center)

	# Interpolate
	coord = su._scale_coord(coord, input.shape, oversamp)
	output = si._interpolate(output, coord, width=width, param=beta)
	output /= width**ndim

	return output


def nufft_adjoint(input, coord, oshape, oversamp=1.25, width=4, center=True):
	ndim = coord.shape[-1]
	beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
	oshape = list(oshape)

	os_shape = su._get_oversamp_shape(oshape, ndim, oversamp)

	# Gridding
	coord = su._scale_coord(coord, oshape, oversamp)
	output = sg._gridding(input, coord, os_shape, width=width, param=beta)
	output /= width**ndim

	# IFFT
	output = ifft(output, axes=range(-ndim, 0), norm=None, center=center)

	# Crop
	output = su._resize(output, oshape)
	output *= np.prod(os_shape[-ndim:], dtype=np.int64) / np.prod(oshape[-ndim:], dtype=np.int64)**0.5

	# Apodize
	su._apodize(output, ndim, oversamp, width, beta)

	return output


def toeplitz_psf(coord, shape, oversamp=1.25, width=4):
	ndim = coord.shape[-1]

	new_shape = su._get_oversamp_shape(shape, ndim, 2)
	new_coord = su._scale_coord(coord, new_shape, 2)

	idx = [slice(None)]*len(new_shape)
	for k in range(-1, -(ndim + 1), -1):
		idx[k] = new_shape[k]//2

	d = cp.zeros(new_shape, dtype=cp.complex64)
	d[tuple(idx)] = 1

	psf = nufft(d, new_coord, oversamp, width)
	psf = nufft_adjoint(psf, new_coord, d.shape, oversamp, width)
	fft_axes = tuple(range(-1, -(ndim + 1), -1))
	psf = fft(psf, axes=fft_axes, norm=None) * (2**ndim)

	return psf


