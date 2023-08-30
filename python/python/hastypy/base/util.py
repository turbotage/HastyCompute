import torch

#copy of sigpy centered fft

def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))

def _normalize_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)

def _expand_shapes(*shapes):
	shapes = [list(shape) for shape in shapes]
	max_ndim = max(len(shape) for shape in shapes)
	shapes_exp = [[1] * (max_ndim - len(shape)) + shape for shape in shapes]

	return tuple(shapes_exp)

def resize(input, oshape, ishift=None, oshift=None):
	ishape_n, oshape_n = _expand_shapes(input.shape, oshape)

	if ishape_n == oshape_n:
		return input.reshape(oshape)

	if ishift is None:
		ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape_n, oshape_n)]

	if oshift is None:
		oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape_n, oshape_n)]

	copy_shape = [
		min(i - si, o - so)
		for i, si, o, so in zip(ishape_n, ishift, oshape_n, oshift)
	]
	islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
	oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

	output = torch.zeros(oshape_n, dtype=input.dtype, device=input.device)
	inpute = input.reshape(ishape_n)
	output[oslice] = inpute[islice]

	return output.reshape(oshape)
	

def fftn(input, oshape=None, axes=None, center=False, norm="ortho"):
	if center:
		ndim = input.ndim
		axes = _normalize_axes(axes, ndim)

		if oshape is not None:
			tmp = resize(input, oshape)
		else:
			tmp = input

		tmp = torch.fft.ifftshift(tmp, dim=axes)
		tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
		return torch.fft.fftshift(tmp, dim=axes)
	else:
		return torch.fft.fftn(input, s=oshape, dim=axes, norm=norm)
	
def ifftn(input, oshape=None, axes=None, center=False, norm="ortho"):
	if center:
		ndim = input.ndim
		axes = _normalize_axes(axes, ndim)

		if oshape is not None:
			tmp = resize(input, oshape)
		else:
			tmp = input

		tmp = torch.fft.ifftshift(tmp, dim=axes)
		tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
		return torch.fft.fftshift(tmp, dim=axes)
	else:
		return torch.fft.ifftn(input, s=oshape, dim=axes, norm=norm)