import numpy as np
import h5py

import plot_utility as pu
import image_creation as ic

crop=False
img = np.array([0])

if crop:
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f.h5', "r") as f:
		img = f['images'][()]
		img = np.transpose(img, axes=(4,3,2,1,0))

	#pu.image_5d(img)

	new_img = ic.crop_mag_vel(img, [(30,200),(30,200),(30,200)], plot_cd=False)

	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_cropped.h5', "w") as f:
		f.create_dataset('images', data=new_img)

	img = new_img

if not crop:
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_6f_cropped.h5', "r") as f:
		img = f['images'][()]

plot_maxip = False
if plot_maxip:
	cd = ic.get_CD(img)
	pu.maxip_4d(cd,axis=1)
	pu.maxip_4d(cd,axis=2)
	pu.maxip_4d(cd,axis=3)

v_enc = 1100
create_encoded = True
if create_encoded:
	A = (1.0/v_enc) * np.array(
		[
		[ 0,  0,  0],
		[-1, -1, -1],
		[ 1,  1, -1],
		[ 1, -1,  1],
		[-1,  1,  1]
		], dtype=np.float32)
	
	imvel = np.expand_dims(np.transpose(img[:,1:], axes=(2,3,4,0,1)), axis=-1)

	imenc = (A @ imvel).squeeze(-1)
	imenc = np.transpose(imenc, axes=(3,4,0,1,2))

	imenc = np.expand_dims(img[:,0], axis=1) * (np.cos(imenc) + 1j*np.sin(imenc))

	pu.image_5d(np.abs(imenc))
	


