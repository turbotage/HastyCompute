
import cupy as cp
import numpy as np
import math

import hastypy.opt.cuda.sigpy.util as su


_interpolate1_cuda = cp.ElementwiseKernel(
	'raw T input, raw S coord, raw S width, raw S param',
	'raw T output',
	"""
	const int ndim = 1;
	const int batch_size = input.shape()[0];
	const int nx = input.shape()[1];

	const int coord_idx[] = {i, 0};
	const S kx = coord[coord_idx];
	const int x0 = ceil(kx - width[ndim - 1] / 2.0);
	const int x1 = floor(kx + width[ndim - 1] / 2.0);

	for (int x = x0; x < x1 + 1; x++) {
		const S w = kernel(
			((S) x - kx) / (width[ndim - 1] / 2.0), param[ndim - 1]);
		for (int b = 0; b < batch_size; b++) {
			const int input_idx[] = {b, mod(x, nx)};
			const T v = (T) w * input[input_idx];
			const int output_idx[] = {b, i};
			output[output_idx] += v;
		}
	}
	""",
	name='interpolate1',
	preamble=su._kaiser_bessel_kernel_cuda + su._mod_cuda,
	reduce_dims=False)

_interpolate2_cuda = cp.ElementwiseKernel(
	'raw T input, raw S coord, raw S width, raw S param',
	'raw T output',
	"""
	const int ndim = 2;
	const int batch_size = input.shape()[0];
	const int ny = input.shape()[1];
	const int nx = input.shape()[2];

	const int coordx_idx[] = {i, 1};
	const S kx = coord[coordx_idx];
	const int coordy_idx[] = {i, 0};
	const S ky = coord[coordy_idx];

	const int x0 = ceil(kx - width[ndim - 1] / 2.0);
	const int y0 = ceil(ky - width[ndim - 2] / 2.0);

	const int x1 = floor(kx + width[ndim - 1] / 2.0);
	const int y1 = floor(ky + width[ndim - 2] / 2.0);

	for (int y = y0; y < y1 + 1; y++) {
		const S wy = kernel(
			((S) y - ky) / (width[ndim - 2] / 2.0),
			param[ndim - 2]);
		for (int x = x0; x < x1 + 1; x++) {
			const S w = wy * kernel(
				((S) x - kx) / (width[ndim - 1] / 2.0),
				param[ndim - 1]);
			for (int b = 0; b < batch_size; b++) {
				const int input_idx[] = {b, mod(y, ny), mod(x, nx)};
				const T v = (T) w * input[input_idx];
				const int output_idx[] = {b, i};
				output[output_idx] += v;
			}
		}
	}
	""",
	name='interpolate2',
	preamble=su._kaiser_bessel_kernel_cuda + su._mod_cuda,
	reduce_dims=False)

_interpolate3_cuda = cp.ElementwiseKernel(
	'raw T input, raw S coord, raw S width, raw S param', 'raw T output', """
	const int ndim = 3;
	const int batch_size = input.shape()[0];
	const int nz = input.shape()[1];
	const int ny = input.shape()[2];
	const int nx = input.shape()[3];

	const int coordz_idx[] = {i, 0};
	const S kz = coord[coordz_idx];
	const int coordy_idx[] = {i, 1};
	const S ky = coord[coordy_idx];
	const int coordx_idx[] = {i, 2};
	const S kx = coord[coordx_idx];

	const int x0 = ceil(kx - width[ndim - 1] / 2.0);
	const int y0 = ceil(ky - width[ndim - 2] / 2.0);
	const int z0 = ceil(kz - width[ndim - 3] / 2.0);

	const int x1 = floor(kx + width[ndim - 1] / 2.0);
	const int y1 = floor(ky + width[ndim - 2] / 2.0);
	const int z1 = floor(kz + width[ndim - 3] / 2.0);

	for (int z = z0; z < z1 + 1; z++) {
		const S wz = kernel(
			((S) z - kz) / (width[ndim - 3] / 2.0),
			param[ndim - 3]);
		for (int y = y0; y < y1 + 1; y++) {
			const S wy = wz * kernel(
				((S) y - ky) / (width[ndim - 2] / 2.0),
				param[ndim - 2]);
			for (int x = x0; x < x1 + 1; x++) {
				const S w = wy * kernel(
					((S) x - kx) / (width[ndim - 1] / 2.0),
					param[ndim - 1]);
				for (int b = 0; b < batch_size; b++) {
					const int input_idx[] = {b, mod(z, nz), mod(y, ny),
						mod(x, nx)};
					const T v = (T) w * input[input_idx];
					const int output_idx[] = {b, i};
					output[output_idx] += v;
				}
			}
		}
	}
	""", 
	name='interpolate3', 
	preamble=su._kaiser_bessel_kernel_cuda + su._mod_cuda,
	reduce_dims=False)

def _interpolate(input, coord, width=2, param=1):
	ndim = coord.shape[-1]

	batch_shape = input.shape[:-ndim]
	batch_size = np.prod(batch_shape, dtype=np.int64)

	pts_shape = coord.shape[:-1]
	npts = np.prod(pts_shape, dtype=np.int64)

	input = input.reshape([batch_size] + list(input.shape[-ndim:]))
	coord = coord.reshape([npts, ndim])

	output = cp.zeros([batch_size, npts], dtype=input.dtype)

	if np.isscalar(param):
		param = cp.array([param] * ndim, coord.dtype)
	else:
		param = cp.array(param, coord.dtype)

	if np.isscalar(width):
		width = cp.array([width] * ndim, coord.dtype)
	else:
		width = cp.array(width, coord.dtype)

	if ndim == 1:
		_interpolate1_cuda(input, coord, width, param, output, size=npts)
	elif ndim == 2:
		_interpolate2_cuda(input, coord, width, param, output, size=npts)
	elif ndim == 3:
		_interpolate3_cuda(input, coord, width, param, output, size=npts)

	return output.reshape(batch_shape + pts_shape)