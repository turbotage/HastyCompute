from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

import time

import numpy as np

def animate_matrix(mat, fps=24):
	fig = plt.figure()
	ax = fig.add_axes([0,0,1,1])
	figdata = ax.imshow(mat[:,:,0])

	def animate(i):
		ax.clear()
		ax.imshow(mat[:,:,i])

	ani = animation.FuncAnimation(fig, func=animate, frames=mat.shape[2], interval=(1/fps))
	plt.show()
	return ani


def plot_matrix(mat, perc=True,clim=None):
	pl = np.percentile(mat, 99)
	pu = np.percentile(mat, 1)
	fig = plt.figure()
	ax = fig.add_axes([0,0,1,1])
	figdata = ax.imshow(mat)
	if clim is not None:
		figdata.set_clim(clim)
	else:
		figdata.set_clim(pl, pu)
	fig.colorbar(figdata, ax=ax)

def param_plotter(pars, slicez=15, clim=[0.0, 16000], viewport=[[0.0, 1.0], [0.0, 1.0]], pars_idx: list[int] = [0]):
	xlen = pars.shape[0]
	ylen = pars.shape[1]
	if viewport is None:
		viewport=[[0.0, 1.0], [0.0, 1.0]]

	xstart = round(xlen*viewport[0][0])
	xend = round(xlen*viewport[0][1])
	ystart = round(ylen*viewport[1][0])
	yend = round(ylen*viewport[1][1])

	for i in pars_idx:
		fig = plt.figure()
		ax = fig.add_axes([0,0,1,1])
		figdata = ax.imshow(pars[xstart:xend,ystart:yend,slicez,i])
		ax.set_title("I'th parameter")
		if clim is not None:
			figdata.set_clim(clim[0], clim[1])
		fig.colorbar(figdata, ax=ax)

def ivim_curve_plot_expr():
	return 'p[0]*(p[1]*np.exp(-x*p[2])+(1-p[1])*np.exp(-x*p[3]))'

def plot_curve_fit(expr, pars, consts, data, num_samples=500, bounds=None):
	ilen = 0.01*(consts[-1] - consts[0])
	istart = consts[:,0] - ilen
	iend = consts[:,-1] + ilen

	domain = np.linspace(istart, iend, num=num_samples)
	values = eval(expr, {'p': pars, 'x': domain, 'np': np})
	cvalues = eval(expr, {'p': pars, 'x': consts, 'np': np})

	fig = plt.figure()
	ax = fig.add_axes([0,0,1,1])
	ax.plot(domain, values, 'r-')
	ax.plot(consts, data, 'bo')
	if bounds != None:
		ax.set_ylim(bounds)
	plt.show()

	return np.sum((cvalues - data)**2)

