import numpy as np
import h5py

import plot_utility as pu
import image_creation as ic

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

def get_CD(img):
	m = img[:,0,:,:,:].astype(np.float32)
	vx = img[:,1,:,:,:].astype(np.float32)
	vy = img[:,2,:,:,:].astype(np.float32)
	vz = img[:,3,:,:,:].astype(np.float32)

	venc = 1100
	return (m * np.sin(np.pi * np.minimum(np.sqrt(vx*vx + vy*vy + vz*vz), 0.5*venc) / venc)).astype(np.float32)

if False:
	img = np.array([0])

	with h5py.File('D:\\4DRecon\\dat\\dat1\\images_20f.h5', "r") as f:
		img = f['images'][()]

	cd = get_CD(img)

	#pu.image_4d(cd)
	pu.mip_4d(cd)

	imgs = []
	for i in range(4):
		imgs.append(np.mean(img[i*4:(i*4+4),:,:,:], axis=0))

	img_mean = np.stack(imgs, axis=0)

	cd_mean = get_CD(img_mean)

	#pu.image_4d(cd_mean)
	pu.mip_4d(cd_mean)


if False:
	img = np.array([0])
	img_mag = np.array([0])

	#with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f.h5', "r") as f:
	#	img = f['images'][()]

	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_mag.h5', "r") as f:
		img_mag = f['images'][()]
		img_mag = np.transpose(img_mag, (2,1,0))

	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f.h5', "r") as f:
		img = f['images'][()]
		img = np.transpose(img, axes=(4,3,2,1,0))

	#img = np.transpose(img, axes=(4,3,2,1,0))
	#ic.numpy_to_nifti(img_mag, 'D:\\4DRecon\\dat\\dat2\\mag.nii')

	pu.image_5d(img, relim=True)
	pu.image_3d(img_mag, relim=True)

	#mask = img_mag > 1000
	#pu.image_3d(mask)

	#cd = get_CD(img)
 
	#pu.image_4d(cd)
	#pu.maxip_4d(cd)
	#pu.minip_4d(cd)

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

if False:
	with h5py.File('D:\\4DRecon\\SenseMapsCpp.h5', "r") as f2:
		maps = f2['Maps'][()]

if False:
	with h5py.File('D:\\4DRecon\\FullRecon.h5', "r") as f2:

		image = f2['IMAGE'][()].squeeze(1)
		image_mag = f2['IMAGE_MAG'][()].squeeze(1)
		image_phase = f2['IMAGE_PHASE'][()].squeeze(1)

		pu.image_5d(np.abs(image))
		pu.image_5d(image_mag)
		pu.image_5d(image_phase)



