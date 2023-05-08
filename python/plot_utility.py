import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def image_5d(image):

	wlen = image.shape[0]
	tlen = image.shape[1]
	zlen = image.shape[2]

	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.23)
	figdata = ax.imshow(image[0, 0, zlen // 2, :,:])

	ax_w = fig.add_axes([0.3, 0.05, 0.5, 0.03])
	ax_t = fig.add_axes([0.3, 0.1, 0.5, 0.03])
	ax_z = fig.add_axes([0.3, 0.15, 0.5, 0.03])

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
		valinit=zlen // 2, valstep=np.arange(0, zlen, dtype=np.int32),
		color="blue"
	)


	def update(val):
		w = wslider.val
		t = tslider.val
		z = zslider.val
		figdata.set_data(image[w,t,z,:,:])
		fig.canvas.draw_idle()

	wslider.on_changed(update)
	tslider.on_changed(update)
	zslider.on_changed(update)

	plt.show()

def image_4d(image):

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
		figdata.set_data(image[t,z,:,:])
		fig.canvas.draw_idle()


	tslider.on_changed(update)
	zslider.on_changed(update)

	plt.show()

def image_3d(image):

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
		figdata.set_data(image[z,:,:])
		fig.canvas.draw_idle()

	zslider.on_changed(update)

	plt.show()
