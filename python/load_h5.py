import numpy as np
import h5py

import plot_utility as pu

#with h5py.File('D:\\4DRecon\\images.h5', "r") as f:
#	img = f['images'][()]
#
#	pu.image_5d(img[:,1:,:,:,:])

def create_mean_img():
	with h5py.File('D:\\4DRecon\\ImagesPils1_time_f10_cf1.395.h5', "r") as f:
		
		img_mag = f['IMAGE_MAG'][()]
		img_phs = f['IMAGE_PHASE'][()]

		mean_mag = np.mean(img_mag, axis=1)

		encs = []
		for i in range(5):
			encs.append(mean_mag*(np.cos(img_phs[:,i,:,:,:]) + 1j*np.sin(img_phs[:,i,:,:,:])))
		
		img = np.stack(encs, axis=1)

		with h5py.File('D:\\4DRecon\\created_img.h5', "w") as f2:
			f2.create_dataset('image', data=img)

		return img

if False:
	CD = np.array([0])

	with h5py.File('D:\\4DRecon\\images.h5', "r") as f:
		img = f['images'][()]

		M = img[:,0,:,:,:].astype(np.float32)
		vx = img[:,1,:,:,:].astype(np.float32)
		vy = img[:,2,:,:,:].astype(np.float32)
		vz = img[:,3,:,:,:].astype(np.float32)

		CD = (M * np.sin(np.pi * np.sqrt(vx*vx + vy*vy + vz*vz) / 110)).astype(np.float32)

	pu.image_4d(CD)

if False:
	created_img = np.array([0])
	recreate = False
	if recreate:
		created_img = create_mean_img()
	else:
		with h5py.File('D:\\4DRecon\\created_img.h5', "r") as f2:
			created_img = f2['image'][()]

	pu.image_5d(np.real(created_img))
	pu.image_5d(np.imag(created_img))
	pu.image_5d(np.abs(created_img))

if True:
	with h5py.File('D:\\4DRecon\\SenseMapsCpp.h5', "r") as f2:
		maps = f2['Maps'][()]
