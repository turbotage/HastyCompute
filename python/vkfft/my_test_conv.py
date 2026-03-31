import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from scipy.datasets import ascent
from skimage.transform import resize
from skimage import data

import cupy
from pyvkfft.base import primes
from pyvkfft.fft import rfftn, irfftn, fftn, ifftn
from pyvkfft.cuda import VkFFTApp, vkfft_max_fft_dimensions
from ctypes import cast
from numpy.ctypeslib import as_ctypes, as_array

import time


def test_2d():
    img = data.camera().astype(np.float32) / 255.0
    # size = 8192
    size = 1024
    # size = 1234
    
    performZeropadding = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    # performZeropadding = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=pfUINT)
    print(performZeropadding)
    # performZeropadding = as_ctypes(performZeropadding)
    fft_zeropad_left = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    fft_zeropad_left[0] = size // 2
    fft_zeropad_left[1] = size // 2
    # fft_zeropad_left = as_ctypes(fft_zeropad_left)
    fft_zeropad_right = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    fft_zeropad_right[0] = size
    fft_zeropad_right[1] = size
    
    img = resize(img, (size, size), anti_aliasing=True)
    
    img[fft_zeropad_left[1]:fft_zeropad_right[1], :] = 0
    img[:, fft_zeropad_left[0]:fft_zeropad_right[0]] = 0
    
    ny, nx = img.shape
    x, y = np.meshgrid(np.arange(-nx//2,nx//2), np.arange(-ny//2, ny//2), indexing='xy')
    
    sigma = size // 100
    kernel = np.fft.fftshift(np.exp(-(x**2+y**2) / (2*sigma**2)))
    kernel /= kernel.sum()

    # Numpy convolution
    img, kernel = img.astype(np.complex64), kernel.astype(np.complex64)
    
    K_np = np.fft.fftn(kernel)
    gd_np = np.fft.ifftn(np.fft.fftn(img) * K_np, img.shape)
    
    gd_np = np.fft.fftshift(gd_np, axes=(-2, -1))
    gd_np = crop_center_2d(gd_np, (size//2, size//2))
    
    
    # move data to GPU
    d_gpu = cupy.asarray(img)
    k_gpu = cupy.asarray(kernel)
    
    K_gpu = fftn(k_gpu)

    # vkfft convolution
    # d_gpu = ifftn(fftn(d_gpu) * fftn(k_gpu))
    
    app_zp = VkFFTApp(
        img.shape, dtype=np.complex64, ndim=2, inplace=True, r2c=False, convolve=False, convolve_conj=0, 
        performZeropadding=performZeropadding,
        fft_zeropad_left=fft_zeropad_left,
        fft_zeropad_right=fft_zeropad_right)
    
    app = VkFFTApp(img.shape, dtype=np.complex64, ndim=2, inplace=True, r2c=False, convolve=False, convolve_conj=0)
    
    # # print(app.is_radix_transform())
    # print(app)
    # print('app.nb_axis_upload:', app.nb_axis_upload)
    # print('app.use_bluestein_fft:', app.use_bluestein_fft)
    # print('app.tmp_buffer_nbytes:', app.tmp_buffer_nbytes)
    # print('app.axis_split', app.axis_split)
    
    # app.fft(d_gpu, convolve_kernel=K_gpu)
    app_zp.fft(d_gpu)
    d_gpu *= K_gpu
    app_zp.ifft(d_gpu)
    
    gd0 = d_gpu.get()    
    gd0 = np.fft.fftshift(gd0, axes=(-2, -1))
    gd0 = crop_center_2d(gd0, (size//2, size//2))
    
    print(np.abs(gd0 - gd_np).max())
    print('np.allclose(gd0, gd_np): ',  np.allclose(gd0, gd_np, rtol=1e-6, atol=gd_np.max()*1e-6))

    plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(img.real, cmap='gray')
    plt.title('(inner) padded input')
    plt.subplot(142)
    plt.imshow(np.fft.fftshift(kernel).real, cmap='gray')
    plt.title('kernel')
    plt.subplot(143)
    plt.imshow(gd_np.real, cmap='gray', vmin=0, vmax=1)
    plt.title('numpy FFTs results')
    plt.tight_layout()    
    plt.subplot(144)
    plt.imshow(gd0.real, cmap='gray', vmin=0, vmax=1)
    plt.title('vKFFT results')
    plt.tight_layout()

    plt.show()

