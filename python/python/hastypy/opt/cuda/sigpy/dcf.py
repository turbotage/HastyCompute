import torch

import cupy as cp
import hastypy.opt.cuda.sigpy.interp_linops as intlinops

def pipe_menon_dcf(
	coord,
	img_shape=None,
	max_iter=30,
	n=128,
	beta=8,
	width=4,
	show_pbar=True,
):
	r"""Compute Pipe Menon density compensation factor.

	Perform the following iteration:

	.. math::

		w = \frac{w}{|G^H G w|}

	with :math:`G` as the gridding operator.

	Args:
		coord (array): k-space coordinates.
		img_shape (None or list): Image shape.
		device (Device): computing device.
		max_iter (int): number of iterations.
		n (int): Kaiser-Bessel sampling numbers for gridding operator.
		beta (float): Kaiser-Bessel kernel parameter.
		width (float): Kaiser-Bessel kernel width.
		show_pbar (bool): show progress bar.

	Returns:
		array: density compensation factor.

	References:
		Pipe, James G., and Padmanabhan Menon.
		Sampling Density Compensation in MRI:
		Rationale and an Iterative Numerical Solution.
		Magnetic Resonance in Medicine 41, no. 1 (1999): 179–86.


	"""

	coord = cp.array(coord.transpose(0,1).numpy())

	w = cp.ones(coord.shape[0], dtype=coord.dtype)

	G = intlinops.Gridding(img_shape, coord, param=beta, width=width)
	
	scale = cp.mean(G.H * G * w)

	print("Pipe Menon DCF:\n")
	for it in range(max_iter):
		GHGw = G.H * G * (w / scale)
		w /= cp.abs(GHGw)
		resid = cp.abs(GHGw - 1).max().item()
		print('\r Resid: ', resid, ' it: ', it, end="")
	print("")

	return torch.tensor(w.get())