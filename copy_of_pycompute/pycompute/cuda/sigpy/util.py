
import cupy as cp
import numpy as np
import math

_kaiser_bessel_kernel_cuda = """
__device__ inline S kernel(S x, S beta) {
	if (fabsf(x) > 1)
		return 0;

	x = beta * sqrt(1 - x * x);
	S t = x / 3.75;
	S t2 = t * t;
	S t4 = t2 * t2;
	S t6 = t4 * t2;
	S t8 = t6 * t2;
	if (x < 3.75) {
		S t10 = t8 * t2;
		S t12 = t10 * t2;
		return 1 + 3.5156229 * t2 + 3.0899424 * t4 +
			1.2067492 * t6 + 0.2659732 * t8 +
			0.0360768 * t10 + 0.0045813 * t12;
	} else {
		S t3 = t * t2;
		S t5 = t3 * t2;
		S t7 = t5 * t2;

		return exp(x) / sqrt(x) * (
			0.39894228 + 0.01328592 / t +
			0.00225319 / t2 - 0.00157565 / t3 +
			0.00916281 / t4 - 0.02057706 / t5 +
			0.02635537 / t6 - 0.01647633 / t7 +
			0.00392377 / t8);
	}
}
"""

_mod_cuda = """
__device__ inline int mod(int x, int n) {
	return (x % n + n) % n;
}
"""

def _apodize(input, ndim, oversamp, width, beta):
	output = input
	for a in range(-ndim, 0):
		i = output.shape[a]
		os_i = math.ceil(oversamp * i)
		idx = cp.arange(i, dtype=output.dtype)

		apod = (beta**2 - (np.pi * width * (idx - i // 2) / os_i)**2)**0.5
		apod /= cp.sinh(apod)
		output *= apod.reshape([i] + [1] * (-a - 1))

	return output

def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [math.ceil(oversamp * i) for i in shape[-ndim:]]

def _scale_coord(coord, shape, oversamp):
    ndim = coord.shape[-1]
    output = coord.copy()
    for i in range(-ndim, 0):
        scale = math.ceil(oversamp * shape[i]) / shape[i]
        shift = math.ceil(oversamp * shape[i]) // 2
        output[..., i] *= scale
        output[..., i] += shift

    return output

def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))

def _resize(input, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.

    Args:
        input (array): Input array.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.

    Returns:
        array: Zero-padded or cropped result.
    """

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = cp.zeros(oshape1, dtype=input.dtype)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output.reshape(oshape)

def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)

def _check_shape_positive(shape):

    if not all(s > 0 for s in shape):
        raise ValueError(
            'Shapes must be positive, got {shape}'.format(shape=shape))

def _check_linops_same_ishape(linops):
    for linop in linops:
        if (linop.ishape != linops[0].ishape):
            raise ValueError(
                'Linops must have the same ishape, got {linops}.'.format(
                    linops=linops))


def _check_linops_same_oshape(linops):
    for linop in linops:
        if (linop.oshape != linops[0].oshape):
            raise ValueError(
                'Linops must have the same oshape, got {linops}.'.format(
                    linops=linops))

