import torch

import numpy as np
import h5py

import garpen_runner

import hastypy.base.enc_to_vel as etv
import hastypy.util.plot_utility as pu
import hastypy.util.image_creation as ic

import hastypy.base.svt as svt

def test_data(noiceperc=0.0):
	nx = 32
	nvox = nx*nx*nx
	venc = 0.15
	true_venc = 1 / np.sqrt(3)
	vel = venc*(-1.0 + 2*np.random.rand(3,nvox)).astype(np.float32) / 10.0 #np.sqrt(3) / 5

	phase = etv.get_encode_matrix(etv.constant_from_venc(venc)) @ vel

	mag = 10*np.random.rand(1,nvox).astype(np.float32)

	pars_true = np.concatenate([mag, vel], axis=0)

	enc = mag * (np.cos(phase) + 1j*np.sin(phase))
	enc = np.reshape(enc, (1,5,nx,nx,nx))

	noise: np.array
	if noiceperc != 0.0:
		real_noise = np.random.normal(scale=0.5*np.sqrt(2), size=enc.shape)
		imag_noise = np.random.normal(scale=0.5*np.sqrt(2), size=enc.shape)
		noise = (real_noise + 1j*imag_noise)
		noise *= np.sqrt(noiceperc)*np.mean(np.abs(enc))

		enc += noise

	linear_pars = etv.enc_to_vel_linear(enc, venc)
	nonlinear_pars = etv.enc_to_vel_nonlinear(enc, venc)

	print('true_pars: ', pars_true[:,0])

	linear_pars = np.reshape(linear_pars, (4, nvox))
	nonlinear_pars = np.reshape(nonlinear_pars, (4, nvox))

	if False:
		true_norm = np.linalg.norm(pars_true[1:,...])
		linear_err = np.linalg.norm((pars_true - linear_pars)[1:,...]) / true_norm
		nonlinear_err = np.linalg.norm((pars_true - nonlinear_pars)[1:,...]) / true_norm
	else:
		true_norm = np.linalg.norm(pars_true)
		linear_err = np.linalg.norm((pars_true - linear_pars)) / true_norm
		nonlinear_err = np.linalg.norm((pars_true - nonlinear_pars)) / true_norm

	print(linear_err)
	print(nonlinear_err)





def test_real():

	#img_framed = np.array([0])
	#with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_real_100.h5', "r") as f:
	#	img_framed = f['images'][()]

	#pu.image_nd(img_full)

	img_full = np.array([0])
	with h5py.File('D:\\4DRecon\\Garpen\\Ena\\my_framed_real_160.h5', "r") as f:
		img_full = f['images'][()]

	#pu.image_nd(img_mean_img)

	#mean_block = svt.extract_mean_block(torch.tensor(img_framed), img_framed.shape[2:], [16,16,16])
	#print('Extremes: ', mean_block[0:3])
	#print('SVD Vals: ', mean_block[3])

	pu.image_nd(img_full)

	img_vel = etv.enc_to_vel_linear(img_full, 1100.0)

	pu.image_nd(img_vel)

	vmag = np.sqrt(img_vel[:,1,...]**2 + img_vel[:,2,...]**2 + img_vel[:,3,...]**2)
	pu.image_nd(vmag)
	

	cd = ic.get_CD(img_vel, 1100.0)
	pu.image_nd(cd)
	#pu.image_nd(vmag)


if __name__ == '__main__':
	test_real()