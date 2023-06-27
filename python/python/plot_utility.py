import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider, CheckButtons, TextBox


def image_5d(image):
	lens = []
	for i in range(len(image.shape)):
		lens.append(image.shape[i])

	fig, ax = plt.subplots()
	fig.set_figwidth(14)
	fig.set_figheight(10)
	fig.subplots_adjust(left=0.05, bottom=0.12)

	figdata = ax.imshow(np.abs(image[0, 0,:,:,0]))

	def disc_slider(axis, name, length, color="red"):
		return Slider(axis, name, 0,length-1, valinit=0, 
			valstep=np.arange(0, length, dtype=np.int32), color=color)

	leftboxx = [0.2, 0.4]
	sliderh = 0.025
	rightboxx1 = [0.68, 0.03]
	boxh = 0.05

	ax0 = fig.add_axes([leftboxx[0], 0.025, leftboxx[1], sliderh])
	slider0 = disc_slider(ax0, "Slicer 0", lens[0], color="green")
	
	ax1 = fig.add_axes([leftboxx[0], 0.05, leftboxx[1], sliderh])
	slider1 = disc_slider(ax1, "Slicer 1", lens[1], color="green")
	
	ax_xyz = fig.add_axes([leftboxx[0], 0.075, leftboxx[1], sliderh])
	slider_xyz = disc_slider(ax_xyz, "XYZ", lens[2], color="red")

	boxax1 = fig.add_axes([rightboxx1[0], 0.035, rightboxx1[1], boxh])
	radbutton1 = RadioButtons(boxax1, ('x', 'y', 'z'), active=2)

	boxax2 = fig.add_axes([rightboxx1[0] - 0.04, 0.035, rightboxx1[1]+0.01, boxh])
	radbutton2 = RadioButtons(boxax2, ('real', 'imag', 'abs'), active=2)

	maxipax = fig.add_axes([rightboxx1[0] + 0.03, 0.035, rightboxx1[1]+0.01, boxh])
	maxipbutton = CheckButtons(maxipax, labels=['Maxip'])

	limax = fig.add_axes([0.15, 0.2, 0.015, 0.5])
	imin = np.minimum(np.real(image).min(), np.imag(image).min())
	imax = np.maximum(np.real(image).max(), np.maximum(np.imag(image).max(), np.abs(image).max()))
	limslider = RangeSlider(limax, "CLim", imin, imax, orientation="vertical", valinit=(imin,imax))

	pretextax = fig.add_axes([0.2, 0.93, 0.55, 0.04])
	pretextbox = TextBox(pretextax, 'Pre Options', initial='')

	posttextax = fig.add_axes([0.2, 0.89, 0.55, 0.04])
	posttextbox = TextBox(posttextax, 'Post Options', initial='')

	labels = ['Direct Command']
	checkboxax = fig.add_axes([0.77, 0.89, 0.15, 0.1])
	checkbox = CheckButtons(checkboxax, labels=labels)

	figdata.set_clim(limslider.val[0], limslider.val[1])

	def update(val):
		xyz = radbutton1.value_selected
		ria = radbutton2.value_selected

		s0 = slider0.val
		s1 = slider1.val
		s_xyz = slider_xyz.val

		if xyz == 'x':
			slider_xyz.valmax = lens[2]-1
		elif xyz == 'y':
			slider_xyz.valmax = lens[3]-1
		elif xyz == 'z':
			slider_xyz.valmax = lens[4]-1
		slider_xyz.ax.set_xlim(slider_xyz.valmin, slider_xyz.valmax)
		if s_xyz > slider_xyz.valmax:
			s_xyz = slider_xyz.valmax
			slider_xyz.val = slider_xyz.valmax

		img: np.array
		maxip = maxipbutton.get_status()[0]

		def riafunc(ria, input):
			if ria == 'real':
				return np.real(input)
			elif ria == 'imag':
				return np.imag(input)
			elif ria == 'abs':
				return np.abs(input)

		def modimgfunc(xyz, maxip, mod_image):
			if maxip:
				if xyz == 'x':
					return np.max(riafunc(ria, mod_image[s0,s1,:(s_xyz+1),:,:]),axis=0)
				elif xyz == 'y':
					return np.max(riafunc(ria, mod_image[s0,s1,:,:(s_xyz+1),:]),axis=1)
				elif xyz == 'z':
					return np.max(riafunc(ria, mod_image[s0,s1,:,:,:(s_xyz+1)]),axis=2)
			else:
				if xyz == 'x':
					return riafunc(ria, mod_image[s0,s1,s_xyz,:,:])
				elif xyz == 'y':
					return riafunc(ria, mod_image[s0,s1,:,s_xyz,:])
				elif xyz == 'z':
					return riafunc(ria, mod_image[s0,s1,:,:,s_xyz])


		mod_image: np.array
		img: np.array
		if pretextbox.text != '':
			try:
				mod_image = eval(pretextbox.text)
			except:
				print('Failed to evalueate PreTextbox')
				pretextbox.set_val('')
				pretextbox.stop_typing()
				mod_image = image
		else:
			mod_image = image

		checked = checkbox.get_status()
		if not checked[0]:
			img = modimgfunc(xyz, maxip, mod_image)
		else:
			if pretextbox.text == '':
				return

		if posttextbox.text != '':
			try:
				img = eval(posttextbox.text)
			except:
				posttextbox.set_val('')
				posttextbox.stop_typing()
				print('Failed to evalueate PostTextbox')


		ax.set(xlim=(0,img.shape[0]), ylim=(0,img.shape[1]))
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
	maxipbutton.on_clicked(update)
	pretextbox.on_submit(update)
	posttextbox.on_submit(update)
	checkbox.on_clicked(update)

	plt.show()

def image_nd(image):
	if len(image.shape) == 5:
		image_5d(image)
	elif len(image.shape) == 4:
		image_5d(image[np.newaxis,...])
	elif len(image.shape) == 3:
		image_5d(image[np.newaxis,np.newaxis,...])
	elif len(image.shape) == 2:
		image_5d(image[np.newaxis,np.newaxis,np.newaxis,...])
	else:
		raise RuntimeError("Only 2 <= n <= 5 is supported for image_nd")

def scatter_3d(coord):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	
	ax.scatter(coord[0,:], coord[1,:], coord[2,:], marker='*')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

def plot_vecs(vecs):
	fig = plt.figure()
	if len(vecs) == 1:
		plt.plot(vecs[0], 'r-*')
	elif len(vecs) == 2:
		plt.plot(vecs[0], 'r-*')
		plt.plot(vecs[1], 'g-o')
	elif len(vecs) == 3:
		plt.plot(vecs[0], 'r-*')
		plt.plot(vecs[1], 'g-o')
		plt.plot(vecs[2], 'b-^')
	else:
		for i in range(len(vecs)):
			plt.plot(vecs[i])
	plt.show()

def plot_gating(gating, bounds):
	plt.figure()
	plt.plot(gating, 'r-*')
	for bound in bounds:
		plt.plot(bound*np.ones(gating.shape), 'b-')
	plt.show()



