import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider

def image_5d(image):
	lens = []
	for i in range(len(image.shape)):
		lens.append(image.shape[i])

	fig, ax = plt.subplots()
	fig.set_figwidth(14)
	fig.set_figheight(10)
	fig.subplots_adjust(left=0.05, bottom=0.12)

	figdata = ax.imshow(np.real(image[0, 0, lens[2] // 2, :,:]))

	def disc_slider(axis, name, length, color="red"):
		return Slider(axis, name, 0,length-1, valinit=0, 
			valstep=np.arange(0, length, dtype=np.int32), color=color)

	leftboxx = [0.2, 0.4]
	sliderh = 0.025
	rightboxx1 = [0.7, 0.03]
	boxh = 0.05

	ax0 = fig.add_axes([leftboxx[0], 0.025, leftboxx[1], sliderh])
	slider0 = disc_slider(ax0, "Slicer 0", lens[0], color="green")
	
	ax1 = fig.add_axes([leftboxx[0], 0.05, leftboxx[1], sliderh])
	slider1 = disc_slider(ax1, "Slicer 1", lens[1], color="green")
	
	ax_xyz = fig.add_axes([leftboxx[0], 0.075, leftboxx[1], sliderh])
	slider_xyz = disc_slider(ax_xyz, "XYZ", lens[2], color="red")

	boxax1 = fig.add_axes([rightboxx1[0], 0.035, rightboxx1[1], boxh])
	radbutton1 = RadioButtons(boxax1, ('x', 'y', 'z'))

	boxax2 = fig.add_axes([rightboxx1[0] - 0.05, 0.035, rightboxx1[1]+0.01, boxh])
	radbutton2 = RadioButtons(boxax2, ('real', 'imag', 'abs'))

	limax = fig.add_axes([0.15, 0.2, 0.015, 0.5])
	imin = np.minimum(np.real(image).min(), np.imag(image).min())
	imax = np.maximum(np.real(image).max(), np.imag(image).max())
	limslider = RangeSlider(limax, "CLim", imin, imax, orientation="vertical", valinit=(imin,imax))

	figdata.set_clim(limslider.val[0], limslider.val[1])

	def update(val):
		xyz = radbutton1.value_selected
		ria = radbutton2.value_selected

		s0 = slider0.val
		s1 = slider1.val
		s_xyz = slider_xyz.val

		img: np.array
		if xyz == 'x':
			img = image[s0,s1,s_xyz,:,:]
		elif xyz == 'y':
			img = image[s0,s1,:,s_xyz,:]
		elif xyz == 'z':
			img = image[s0,s1,:,:,s_xyz]

		if ria == 'real':
			img = np.real(img)
		elif ria == 'imag':
			img = np.imag(img)
		elif ria == 'abs':
			img = np.abs(img)

		figdata.set_data(img)
		clim = limslider.val
		figdata.set_clim(clim[0], clim[1])

		fig.canvas.draw_idle()

	slider0.on_changed(update)
	slider1.on_changed(update)
	slider_xyz.on_changed(update)
	radbutton1.on_clicked(update)
	radbutton2.on_clicked(update)
	limslider.on_changed(update)

	plt.show()

def image_6d(image, relim=False):

	wlen = image.shape[0]
	tlen = image.shape[1]
	zlen = image.shape[2]
	plen = image.shape[3]

	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.23)
	figdata = ax.imshow(image[0, 0, 0, plen // 2, :,:])

	ax_w = fig.add_axes([0.3, 0.05, 0.5, 0.03])
	ax_t = fig.add_axes([0.3, 0.1, 0.5, 0.03])
	ax_z = fig.add_axes([0.3, 0.15, 0.5, 0.03])
	ax_p = fig.add_axes([0.3, 0.20, 0.5, 0.03])

	# define the values to use for snapping

	# create the sliders
	wslider = Slider(
		ax_w, "W Slice", 0, wlen-1,
		valinit=0, valstep=np.arange(0, wlen, dtype=np.int32),
		color="red"
	)

	tslider = Slider(
		ax_t, "T Slice", 0, tlen-1,
		valinit=0, valstep=np.arange(0, tlen, dtype=np.int32),
		color="green"
	)

	zslider = Slider(
		ax_z, "Z Slice", 0, zlen-1,
		valinit=0, valstep=np.arange(0, zlen, dtype=np.int32),
		color="blue"
	)

	pslider = Slider(
		ax_p, "P Slice", 0, plen-1,
		valinit=plen // 2, valstep=np.arange(0, plen, dtype=np.int32),
		color="cyan"
	)

	def update(val):
		w = wslider.val
		t = tslider.val
		z = zslider.val
		p = pslider.val
		img = image[w,t,z,p,:,:]
		figdata.set_data(img)
		if relim:
			imax = np.max(img, axis=(0,1))
			imin = np.min(img, axis=(0,1))
			figdata.set_clim(imin, imax)
		fig.canvas.draw_idle()

	wslider.on_changed(update)
	tslider.on_changed(update)
	zslider.on_changed(update)
	pslider.on_changed(update)

	plt.show()

def image_4d(image, relim=False):

	tlen = image.shape[0]
	zlen = image.shape[1]

	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.23)
	figdata = ax.imshow(image[tlen-1, zlen // 2, :,:])

	ax_t = fig.add_axes([0.3, 0.1, 0.5, 0.03])
	ax_z = fig.add_axes([0.3, 0.15, 0.5, 0.03])

	# define the values to use for snapping

	# create the sliders
	tslider = Slider(
		ax_t, "T Slice", 0, tlen-1,
		valinit=0, valstep=np.arange(0, tlen, dtype=np.int32),
		color="green"
	)

	zslider = Slider(
		ax_z, "Z Slice", 0, zlen-1,
		valinit=zlen // 2, valstep=np.arange(0, zlen, dtype=np.int32),
		color="blue"
	)


	def update(val):
		t = tslider.val
		z = zslider.val

		img = image[t,z,:,:]
		figdata.set_data(img)
		if relim:
			imax = np.max(img, axis=(0,1))
			imin = np.min(img, axis=(0,1))
			figdata.set_clim(imin, imax)
		fig.canvas.draw_idle()


	tslider.on_changed(update)
	zslider.on_changed(update)

	plt.show()

def image_3d(image, relim=False):

	zlen = image.shape[0]

	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.23)
	figdata = ax.imshow(image[0,:,:])

	ax_z = fig.add_axes([0.3, 0.15, 0.5, 0.03])

	# define the values to use for snapping

	# create the sliders
	zslider = Slider(
		ax_z, "Z Slice", 0, zlen-1,
		valinit=0, valstep=np.arange(0, zlen, dtype=np.int32),
		color="green"
	)

	def update(val):
		z = zslider.val

		img = image[z,:,:]
		figdata.set_data(img)
		if relim:
			imax = np.max(img, axis=(0,1))
			imin = np.min(img, axis=(0,1))
			figdata.set_clim(imin, imax)
		fig.canvas.draw_idle()

	zslider.on_changed(update)

	plt.show()

def image_2d(image, relim=False):
	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.23)
	figdata = ax.imshow(image)
	plt.show()

def image_nd(image, relim=False):
	if image.dtype == np.complex64 or image.dtype == np.complex128:
		image = np.abs(image)

	if len(image.shape) == 5:
		image_5d(image)
	elif len(image.shape) == 4:
		image_4d(image, relim)
	elif len(image.shape) == 3:
		image_3d(image, relim)
	elif len(image.shape) == 2:
		image_2d(image, relim)

def scatter_3d(coord):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	
	ax.scatter(coord[0,:], coord[1,:], coord[2,:], marker='*')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

def maxip_5d(image, axis=4, relim=False):
	mip = np.max(image, axis=axis)
	image_4d(mip, relim=relim)

def minip_5d(image, axis=4, relim=False):
	mip = np.min(image, axis=axis)
	image_4d(mip, relim=relim)

def maxip_4d(image, axis=3, relim=False):
	mip = np.max(image, axis=axis)
	image_3d(mip, relim=relim)

def minip_4d(image, axis=3, relim=False):
	mip = np.min(image, axis=axis)
	image_3d(mip, relim=relim)

def plot_1vec(vec1):
	fig = plt.figure()
	plt.plot(vec1, 'r-*')
	plt.show()

def plot_2vec(vec1, vec2):
	fig = plt.figure()
	plt.plot(vec1, 'r-*')
	plt.plot(vec2, 'g-o')
	plt.show()

def plot_3vec(vec1, vec2, vec3):
	fig = plt.figure()
	plt.plot(vec1, 'r-*')
	plt.plot(vec2, 'g-o')
	plt.plot(vec3, 'b-^')
	plt.show()

def plot_gating(gating, bounds):
	plt.figure()
	plt.plot(gating, 'r-*')
	for bound in bounds:
		plt.plot(bound*np.ones(gating.shape), 'b-')
	plt.show()



