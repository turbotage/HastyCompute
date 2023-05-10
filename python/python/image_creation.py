import numpy as np

import plot_utility as pu

import nibabel as nib


def get_CD(img, venc=1100, plot_cd = False, plot_mip = False):
	m = img[:,0,:,:,:].astype(np.float32)
	vx = img[:,1,:,:,:].astype(np.float32)
	vy = img[:,2,:,:,:].astype(np.float32)
	vz = img[:,3,:,:,:].astype(np.float32)

	cd = (m * np.sin(np.pi * np.minimum(np.sqrt(vx*vx + vy*vy + vz*vz), 0.5*venc) / venc)).astype(np.float32)

	if plot_cd:
		pu.image_4d(cd)
	if plot_mip:
		pu.maxip_4d(cd)

	return cd

def crop_mag_vel(img, box, plot=False, plot_cd=True):
	new_img = img[:,:,box[0][0]:box[0][1],box[1][0]:box[1][1],box[2][0]:box[2][1]]

	if plot:
		pu.image_5d(new_img)
	if plot_cd:
		cd = get_CD(new_img)
		#pu.image_4d(cd)
		pu.maxip_4d(cd,axis=1)
		pu.maxip_4d(cd,axis=2)
		pu.maxip_4d(cd,axis=3)

	return new_img

def numpy_to_nifti(img, file):
	nimg = nib.Nifti1Image(img, affine=np.eye(4))
	nib.save(nimg, file)
