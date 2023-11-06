
import test

import numpy as np
import nibabel as nib
import h5py

import weakref

#from nibabel.affines import voxel_sizes
#from nibabel.optpkg import optional_package
#from nibabel.orientations import aff2axcodes, axcodes2ornt

import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch

import hastypy.util.plot_utility as pu

class HastyOrthoSlicer:

	def __init__(self, data, title=None):
		"""
		Parameters
		----------
		data : array-like
			The data that will be displayed by the slicer. Should have 3+
			dimensions.
		title : str or None, optional
			The title to display. Can be None (default) to display no
			title.
		"""
		if len(data.shape) == 1:
			data = data[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
		if len(data.shape) == 2:
			data = data[:,:,np.newaxis,np.newaxis,np.newaxis]
		if len(data.shape) == 3:
			data = data[:,:,:,np.newaxis,np.newaxis]
		if len(data.shape) == 4:
			data = data[:,:,:,:,np.newaxis]

		if len(data.shape) > 5:
			raise RuntimeError("Can't use HastyOrthoSlicer for more than 5d data")


		# Use these late imports of matplotlib so that we have some hope that
		# the test functions are the first to set the matplotlib backend. The
		# tests set the backend to something that doesn't require a display.
		self._title = title
		self._closed = False
		self._cross = True

		self._mask_mode = False
		self._mask_idxs = []

		data = np.asanyarray(data)
		if data.ndim < 3:
			raise ValueError('data must have at least 3 dimensions')
		if np.iscomplexobj(data):
			raise TypeError('Complex data not supported')
		affine = np.eye(4)
		if affine.shape != (4, 4):
			raise ValueError('affine must be a 4x4 matrix')
		# determine our orientation
		self._affine = affine
		#codes = axcodes2ornt(aff2axcodes(self._affine))
		self._order = np.array([0,1,2], dtype=np.int64) #np.argsort([c[0] for c in codes])
		self._flips = [False, False, False, False] #np.array([c[1] < 0 for c in codes])[self._order]
		self._scalers = [1.0, 1.0, 1.0] #voxel_sizes(self._affine)
		self._inv_affine = np.linalg.inv(affine)
		# current volume info
		self._volume_dims = data.shape[3:]
		self._current_vol_data = data[:, :, :, ...] if data.ndim > 3 else data
		self._data = data
		self._clim = np.percentile(data, (1.0, 99.0))
		del data

		# ^ +---------+   ^ +---------+
		# | |         |   | |         |
		#   |   Sag   |     |   Cor   |
		# S |    0    |   S |    1    |
		#   |         |     |         |
		#   |         |     |         |
		#   +---------+     +---------+
		#        A  -->     <--  R
		# ^ +---------+     +---------+
		# | |         |     |         |
		#   |  Axial  |     |   Vol   |
		# A |    2    |     |    3    |
		#   |         |     |         |
		#   |         |     |         |
		#   +---------+     +---------+
		#   <--  R          <--  t  -->

		fig, axes = plt.subplots(2, 2)
		fig.set_size_inches((9, 9), forward=True)
		self._axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
		plt.tight_layout(pad=0.1)
		#if self.n_volumes <= 1:
		#	fig.delaxes(self._axes[3])
		#	self._axes.pop(-1)
		if self._title is not None:
			fig.canvas.manager.set_window_title(str(title))

		# Start midway through each axis, idx is current slice number
		self._ims, self._data_idx = list(), list()

		# set up axis crosshairs
		self._crosshairs = [None] * 3
		r = [
			self._scalers[self._order[2]] / self._scalers[self._order[1]],
			self._scalers[self._order[2]] / self._scalers[self._order[0]],
			self._scalers[self._order[1]] / self._scalers[self._order[0]],
		]
		self._sizes = [self._data.shape[order] for order in self._order]
		for ii, xax, yax, ratio, label in zip([0, 1, 2], [1, 0, 0], [2, 2, 1], r, ('SAIP', 'SRIL', 'ARPL')):
			ax = self._axes[ii]
			d = np.zeros((self._sizes[yax], self._sizes[xax]))
			im = self._axes[ii].imshow(
				d,
				vmin=self._clim[0],
				vmax=self._clim[1],
				aspect=1,
				cmap='gray',
				interpolation='nearest',
				origin='lower',
			)
			self._ims.append(im)
			vert = ax.plot(
				[0] * 2, [-0.5, self._sizes[yax] - 0.5], color=(0, 1, 0), linestyle='-'
			)[0]
			horiz = ax.plot(
				[-0.5, self._sizes[xax] - 0.5], [0] * 2, color=(0, 1, 0), linestyle='-'
			)[0]
			self._crosshairs[ii] = dict(vert=vert, horiz=horiz)
			# add text labels (top, right, bottom, left)
			lims = [0, self._sizes[xax], 0, self._sizes[yax]]
			bump = 0.01
			poss = [
				[lims[1] / 2.0, lims[3]],
				[(1 + bump) * lims[1], lims[3] / 2.0],
				[lims[1] / 2.0, 0],
				[lims[0] - bump * lims[1], lims[3] / 2.0],
			]
			anchors = [
				['center', 'bottom'],
				['left', 'center'],
				['center', 'top'],
				['right', 'center'],
			]
			for pos, anchor, lab in zip(poss, anchors, label):
				ax.text(
					pos[0], pos[1], lab, horizontalalignment=anchor[0], verticalalignment=anchor[1]
				)
			ax.axis(lims)
			ax.set_aspect(ratio)
			ax.patch.set_visible(False)
			ax.set_frame_on(False)
			ax.axes.get_yaxis().set_visible(False)
			ax.axes.get_xaxis().set_visible(False)
			self._data_idx.append(0)

		# Set up volumes axis
		self._data_idx.append(0)
		self._data_idx.append(0)
		ax = self._axes[3]
		try:
			ax.set_facecolor('k')
		except AttributeError:  # old mpl
			ax.set_axis_bgcolor('k')
		ax.set_title('Volumes')
		
		vol3 = self._data.shape[3]
		vol4 = self._data.shape[4]

		volwid = ax.imshow(np.mean(self._data, axis=(0,1,2)), 
			extent=(0,vol4,0,vol3), cmap='gray', aspect='auto')
	
		xy = (0.5, 0.5)
		patch = pltpatch.Rectangle(
			xy,
			1,
			1,
			fill=True,
			facecolor=(0, 1, 0),
			edgecolor=(1, 0, 0),
			alpha=0.5,
		)
		ax.add_patch(patch)
		self._volume_ax_objs = dict(volwid=volwid, patch=patch)

		self._figs = {a.figure for a in self._axes}
		for fig in self._figs:
			fig.canvas.mpl_connect('scroll_event', self._on_scroll)
			fig.canvas.mpl_connect('motion_notify_event', self._on_mouse)
			fig.canvas.mpl_connect('button_press_event', self._on_mouse)
			fig.canvas.mpl_connect('key_press_event', self._on_keypress)
			fig.canvas.mpl_connect('key_release_event', self._on_keyrelease)
			fig.canvas.mpl_connect('close_event', self._cleanup)

		# actually set data meaningfully
		self._position = np.zeros(4)
		self._position[3] = 1.0  # convenience for affine multiplication
		self._changing = False  # keep track of status to avoid loops
		self._links = []  # other viewers this one is linked to
		plt.draw()
		for fig in self._figs:
			fig.canvas.draw()
		self._set_volume_index((0,0), update_slices=False)
		self._set_position(0.0, 0.0, 0.0)
		self._draw()

	def show(self):
		"""Show the slicer in blocking mode; convenience for ``plt.show()``"""
		plt.show()
	
	def close(self):
		"""Close the viewer figures"""
		self._cleanup()
		for f in self._figs:
			plt.close(f)

	def _cleanup(self):
		"""Clean up before closing"""
		self._closed = True
		for link in list(self._links):  # make a copy before iterating
			self._unlink(link())

	def draw(self):
		"""Redraw the current image"""
		for fig in self._figs:
			fig.canvas.draw()

	@property
	def n_volumes(self):
		"""Number of volumes in the data"""
		return int(np.prod(self._volume_dims))

	@property
	def position(self):
		"""The current coordinates"""
		return self._position[:3].copy()

	@property
	def figs(self):
		"""A tuple of the figure(s) containing the axes"""
		return tuple(self._figs)

	@property
	def cmap(self):
		"""The current colormap"""
		return self._cmap
	
	@cmap.setter
	def cmap(self, cmap):
		for im in self._ims:
			im.set_cmap(cmap)
		self._cmap = cmap
		self.draw()

	@property
	def clim(self):
		"""The current color limits"""
		return self._clim

	@clim.setter
	def clim(self, clim):
		clim = np.array(clim, float)
		if clim.shape != (2,):
			raise ValueError('clim must be a 2-element array-like')
		for im in self._ims:
			im.set_clim(clim)
		self._clim = tuple(clim)
		self.draw()

	def link_to(self, other):
		"""Link positional changes between two canvases

		Parameters
		----------
		other : instance of OrthoSlicer3D
			Other viewer to use to link movements.
		"""
		if not isinstance(other, self.__class__):
			raise TypeError(
				f'other must be an instance of {self.__class__.__name__}, not {type(other)}'
			)
		self._link(other, is_primary=True)

	def _link(self, other, is_primary):
		"""Link a viewer"""
		ref = weakref.ref(other)
		if ref in self._links:
			return
		self._links.append(ref)
		if is_primary:
			other._link(self, is_primary=False)
			other.set_position(*self.position)

	def _unlink(self, other):
		"""Unlink a viewer"""
		ref = weakref.ref(other)
		if ref in self._links:
			self._links.pop(self._links.index(ref))
			ref()._unlink(self)

	def _notify_links(self):
		"""Notify linked canvases of a position change"""
		for link in self._links:
			link().set_position(*self.position[:3])

	def set_position(self, x=None, y=None, z=None):
		"""Set current displayed slice indices

		Parameters
		----------
		x : float | None
			X coordinate to use. If None, do not change.
		y : float | None
			Y coordinate to use. If None, do not change.
		z : float | None
			Z coordinate to use. If None, do not change.
		"""
		self._set_position(x, y, z)
		self._draw()

	def set_volume_idx(self, v):
		"""Set current displayed volume index

		Parameters
		----------
		v : int
			Volume index.
		"""
		self._set_volume_index(v)
		self._draw()

	def _set_volume_index(self, v, update_slices=True):
		"""Set the plot data using a volume index"""
		v = self._data_idx[3:] if v is None else (int(v[0]), int(v[1]))
		if v == self._data_idx[3:]:
			return
		max_ = np.prod(self._volume_dims)
		idx = (slice(None), slice(None), slice(None))
		if self._data.ndim > 3:
			idx = idx + v
		self._data_idx[3:] = v
		self._current_vol_data = self._data[idx]
		# update all of our slice plots
		if update_slices:
			self._set_position(None, None, None, notify=False)

	def _set_position(self, x, y, z, notify=True):
		"""Set the plot data using a physical position"""
		# deal with volume first
		if self._changing:
			return
		self._changing = True
		x = self._position[0] if x is None else float(x)
		y = self._position[1] if y is None else float(y)
		z = self._position[2] if z is None else float(z)

		# deal with slicing appropriately
		self._position[:3] = [x, y, z]
		idxs = np.dot(self._inv_affine, self._position)[:3]
		for ii, (size, idx) in enumerate(zip(self._sizes, idxs)):
			self._data_idx[ii] = max(min(int(round(idx)), size - 1), 0)
		for ii in range(3):
			# sagittal: get to S/A
			# coronal: get to S/L
			# axial: get to A/L
			data = np.rollaxis(self._current_vol_data, axis=self._order[ii])[self._data_idx[ii]]
			xax = [1, 0, 0][ii]
			yax = [2, 2, 1][ii]
			if self._order[xax] < self._order[yax]:
				data = data.T
			if self._flips[xax]:
				data = data[:, ::-1]
			if self._flips[yax]:
				data = data[::-1]
			self._ims[ii].set_data(data)
			# deal with crosshairs
			loc = self._data_idx[ii]
			if self._flips[ii]:
				loc = self._sizes[ii] - loc
			loc = [loc] * 2
			if ii == 0:
				self._crosshairs[2]['vert'].set_xdata(loc)
				self._crosshairs[1]['vert'].set_xdata(loc)
			elif ii == 1:
				self._crosshairs[2]['horiz'].set_ydata(loc)
				self._crosshairs[0]['vert'].set_xdata(loc)
			else:  # ii == 2
				self._crosshairs[1]['horiz'].set_ydata(loc)
				self._crosshairs[0]['horiz'].set_ydata(loc)

		# Update volume trace
		if self.n_volumes > 1 and len(self._axes) > 3:
			idx = [slice(None)] * 3
			for ii in range(3):
				idx[self._order[ii]] = self._data_idx[ii]
			idx += [slice(None)] * 2

			vdata = self._data[tuple(idx)]

			patchobj = self._volume_ax_objs['patch']
			patchobj.set_x(self._data_idx[4])
			patchobj.set_y(self._data_idx[3])

			volwidobj = self._volume_ax_objs['volwid']
			volwidobj.set_data(vdata)
			volwidobj.set_clim(vdata.min(), vdata.max())
			
		if notify:
			self._notify_links()
		self._changing = False

	# Matplotlib handlers ####################################################
	def _in_axis(self, event):
		"""Return axis index if within one of our axes, else None"""
		if getattr(event, 'inaxes') is None:
			return None
		for ii, ax in enumerate(self._axes):
			if event.inaxes is ax:
				return ii
			
	def _on_scroll(self, event):
		"""Handle mpl scroll wheel event"""
		assert event.button in ('up', 'down')
		ii = self._in_axis(event)
		if ii is None:
			return
		if event.key is not None and 'shift' in event.key:
			if self.n_volumes <= 1:
				return
			ii = 3  # shift: change volume in any axis
		assert ii in range(5)
		dv = 10.0 if event.key is not None and 'control' in event.key else 1.0
		dv *= 1.0 if event.button == 'up' else -1.0
		dv *= -1 if self._flips[ii] else 1
		val = self._data_idx[ii] + dv
		#if ii == 3:
		#	self._set_volume_index(val)
		#else:
		coords = [self._data_idx[k] for k in range(3)] + [1.0]
		coords[ii] = val
		self._set_position(*np.dot(self._affine, coords)[:3])
		self._draw()

	def _on_mouse(self, event):
		"""Handle mpl mouse move and button press events"""
		if event.button != 1:  # only enabled while dragging
			return
		ii = self._in_axis(event)
		if ii is None:
			return
		if ii == 3:
			if self._mask_mode:
				pass
			else:
				# volume plot directly translates
				self._set_volume_index((event.ydata,event.xdata))
		else:
			if self._mask_mode:
				self._mask_idxs.append((event.ydata,event.xdata))
			else:
				# translate click xdata/ydata to physical position
				xax, yax = [[1, 2], [0, 2], [0, 1]][ii]
				x, y = event.xdata, event.ydata
				x = self._sizes[xax] - x if self._flips[xax] else x
				y = self._sizes[yax] - y if self._flips[yax] else y
				idxs = [None, None, None, 1.0]
				idxs[xax] = x
				idxs[yax] = y
				idxs[ii] = self._data_idx[ii]
				self._set_position(*np.dot(self._affine, idxs)[:3])
		self._draw()

	def _on_keypress(self, event):
		"""Handle mpl keypress events"""
		if event.key is not None and 'escape' in event.key:
			self.close()
		elif event.key == 'up':
			new_idx = list(self._data_idx[3:])
			new_idx[0] += 1
			new_idx[0] = min(self._data.shape[3]-1, new_idx[0])
			self._set_volume_index(tuple(new_idx), update_slices=True)
			self._draw()
		elif event.key == 'down':
			new_idx = list(self._data_idx[3:])
			new_idx[0] -= 1
			new_idx[0] = max(0, new_idx[0])
			self._set_volume_index(tuple(new_idx), update_slices=True)
			self._draw()
		elif event.key == 'right':
			new_idx = list(self._data_idx[3:])
			new_idx[1] += 1
			new_idx[1] = min(self._data.shape[4]-1, new_idx[1])
			self._set_volume_index(tuple(new_idx), update_slices=True)
			self._draw()
		elif event.key == 'left':
			new_idx = list(self._data_idx[3:])
			new_idx[1] -= 1
			new_idx[1] = max(0, new_idx[1])
			self._set_volume_index(tuple(new_idx), update_slices=True)
			self._draw()
		elif event.key == 'ctrl+x':
			self._cross = not self._cross
			self._draw()
		elif event.key == 'm':
			self._mask_idxs = []
			self._mask_mode = True
			self._cross = False
			self._draw()

	def _on_keyrelease(self, event):
		if event.key == 'm':
			self._mask_idxs = []
			self._mask_mode = False

	def _draw(self):
		"""Update all four (or three) plots"""
		if self._closed:  # make sure we don't draw when we shouldn't
			return
		for ii in range(3):
			ax = self._axes[ii]
			ax.draw_artist(self._ims[ii])
			if self._cross:
				for line in self._crosshairs[ii].values():
					ax.draw_artist(line)
			ax.figure.canvas.blit(ax.bbox)
		if self.n_volumes > 1 and len(self._axes) > 3:
			ax = self._axes[3]
			ax.draw_artist(ax.patch)  # axis bgcolor to erase old lines
			for key in ('volwid', 'patch'):
				ax.draw_artist(self._volume_ax_objs[key])
			ax.figure.canvas.blit(ax.bbox)


def ortho_print_selector(data):
	plt.figure()




with h5py.File('D:/4DRecon/dat/dat2/for_plotting_tests.h5', 'r') as f:
	data = f['images'][()]

dataf = np.flip(np.abs(data).transpose((2,3,4,0,1)), axis=2)
slicer = HastyOrthoSlicer(dataf[:,:,:,:,:])

slicer.show()