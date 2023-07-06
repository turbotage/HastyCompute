import sys
import numpy as np
import cupy as cp

import math
import h5py
import plot_utility as pu
import image_creation as ic

sys.path.append('C:\\Users\\TurboTage\\Documents\\GitHub\\pycompute')

from jinja2 import Template

from pycompute.cuda.lsqnonlin import F_GradF, SecondOrderLevenbergMarquardt
from pycompute.cuda.cuda_program import CudaFunction, CudaTensor

class EncVelF(CudaFunction):
	def __init__(self):
		return
	
	def get_device_funcid(self):
		return 'enc_to_vel_f'
	
	def get_device_code(self):
		f_temp = Template(
"""
__device__ 
void enc_to_vel_f(const float* params, const float* consts, const float* data, 
	float* f, int tid, int Nprobs)
{
	float M0 = params[tid];
	float vx = params[Nprobs+tid];
	float vy = params[2*Nprobs+tid];
	float vz = params[3*Nprobs+tid];

	float k = consts[0];

	float vel_term;
	float res;

	// Encoding 0
	res = M0 - data[tid];
	f[tid] += res * res;

	// Encoding 1
	vel_term = -k*(vx+vy+vz);
	// Real
	res = M0*cosf(vel_term) - data[Nprobs+tid];
	f[tid] += res * res;
	// Imag
	res = M0*sinf(vel_term) - data[2*Nprobs+tid];
	f[tid] += res * res;

	// Encoding 2
	vel_term = k*(vx+vy-vz);
	// Real
	res = M0*cosf(vel_term) - data[3*Nprobs+tid];
	f[tid] += res * res;
	// Imag
	res = M0*sinf(vel_term) - data[4*Nprobs+tid];
	f[tid] += res * res;
	
	// Encoding 3
	vel_term = k*(vx-vy+vz);
	// Real
	res = M0*cosf(vel_term) - data[5*Nprobs+tid];
	f[tid] += res * res;
	// Imag
	res = M0*sinf(vel_term) - data[6*Nprobs+tid];
	f[tid] += res * res;

	// Encoding 4
	vel_term = k*(vy+vz-vx);
	// Real
	res = M0*cosf(vel_term) - data[7*Nprobs+tid];
	f[tid] += res * res;
	// Imag
	res = M0*sinf(vel_term) - data[8*Nprobs+tid];
	f[tid] += res * res;
	
}
""")

		return f_temp.render()

	def get_deps(self):
		return list()

class EncVelF_GradF(CudaFunction):
	def __init__(self):
		return
	
	def get_device_funcid(self):
		return 'enc_to_vel_fghhl'
	
	def get_device_code(self):
		fghhl_temp = Template(
"""
__device__
void ghhl_add(const float* jac, const float* hes, float res, float lambda, float* h, float* hl, float* g, int tid, int Nprobs)
{
	int l = 0;
	for (int j = 0; j < 4; ++j) {
		// this sums up hessian parts
		for (int k = 0; k <= j; ++k) {
			int lidx = l*Nprobs+tid;
			float jtemp = jac[j] * jac[k];
			float hjtemp = hes[l] + jtemp;
			h[lidx] += hjtemp;
			if (j != k) {
				hl[lidx] += hjtemp;
			} else {
				hl[lidx] += hjtemp + fmaxf(lambda*jtemp, 1e-4f);
			}
			++l;
		}

		g[j*Nprobs+tid] += jac[j] * res;
	}
}


__device__
void enc_to_vel_fghhl(const float* params, const float* consts, const float* data, const float* lam,
	float* f, float* g, float* h, float* hl, int tid, int Nprobs)
{
	float res;
	float jac[4];
	float hes[10];
	float lambda = lam[tid];

	float M0 = params[tid];
	float vx = params[Nprobs+tid];
	float vy = params[2*Nprobs+tid];
	float vz = params[3*Nprobs+tid];

	/*
	if (tid == 0) {
		printf("M0: %f, vx: %f, vy: %f, vz: %f \\n", M0, vx, vy, vz);
		//for (int i = 0; i < 9; ++i) {
		//	printf("%f, ", data[i*Nprobs+tid]);
		//}
		//printf("\\n");
	}
	*/

	float k = consts[0];

	float vt[8];
	// Encoding 1
	vt[0] = cosf(-k*(vx+vy+vz));
	vt[1] = sinf(-k*(vx+vy+vz));
	// Encoding 2
	vt[2] = cosf(k*(vx+vy-vz));
	vt[3] = sinf(k*(vx+vy-vz));
	// Encoding 3
	vt[4] = cosf(k*(vx-vy+vz));
	vt[5] = sinf(k*(vx-vy+vz));
	// Encoding 4
	vt[6] = cosf(k*(vy+vz-vx));
	vt[7] = sinf(k*(vy+vz-vx));

	
	// Encoding 0
	res = M0 - data[tid];
	f[tid] += res * res;

	jac[0] = 1.0f;
	jac[1] = 0.0f;
	jac[2] = 0.0f;
	jac[3] = 0.0f;

	for (int i = 0; i < 10; ++i) {
		hes[i] = 0.0f;
	}

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);

	
	// Encoding 1 Real
	res = M0*vt[0] - data[Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[0];
	jac[1] = M0*vt[1]*k;
	jac[2] = jac[1];
	jac[3] = jac[1];

	hes[0] = 0.0f;
	hes[1] = vt[1]*k*res;
	hes[2] = -M0*vt[0]*k*k*res;
	hes[3] = hes[1];
	hes[4] = hes[2];
	hes[5] = hes[2];
	hes[6] = hes[1];
	hes[7] = hes[2];
	hes[8] = hes[2];
	hes[9] = hes[1];

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 1 Imag
	res = M0*vt[1] - data[2*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[1];
	jac[1] = -M0*vt[0]*k;
	jac[2] = jac[1];
	jac[3] = jac[1];

	hes[0] = 0.0f;
	hes[1] = -vt[0]*k*res;
	hes[2] = -M0*vt[1]*k*k*res;
	hes[3] = hes[1];
	hes[4] = hes[2];
	hes[5] = hes[2];
	hes[6] = hes[1];
	hes[7] = hes[2];
	hes[8] = hes[2];
	hes[9] = hes[2];
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);

	
	// Encoding 2 Real
	res = M0*vt[2] - data[3*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[2];
	jac[1] = -M0*vt[3]*k;
	jac[2] = jac[1];
	jac[3] = -jac[1];

	hes[0] = 0.0f;
	hes[1] = -vt[3]*k*res;
	hes[2] = -M0*vt[2]*k*k*res;
	hes[3] = hes[1];
	hes[4] = hes[2];
	hes[5] = hes[2];
	hes[6] = -hes[1];
	hes[7] = -hes[2];
	hes[8] = -hes[2];
	hes[9] = hes[2];
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 2 Imag
	res = M0*vt[3] - data[4*Nprobs+tid];
	f[tid] += res * res;
	
	jac[0] = vt[3];
	jac[1] = M0*vt[2]*k;
	jac[2] = jac[1];
	jac[3] = -jac[1];

	hes[0] = 0.0f;
	hes[1] = vt[2]*k*res;
	hes[2] = -M0*vt[3]*k*k*res;
	hes[3] = hes[1];
	hes[4] = hes[2];
	hes[5] = hes[2];
	hes[6] = -hes[1];
	hes[7] = -hes[2];
	hes[8] = -hes[2];
	hes[9] = hes[2];
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);

	// Encoding 3 Real
	res = M0*vt[4] - data[5*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[4];
	jac[1] = -M0*vt[5]*k;
	jac[2] = -jac[1];
	jac[3] = jac[1];

	hes[0] = 0.0f;
	hes[1] = vt[5]*k*res;
	hes[2] = -M0*vt[4]*k*k*res;
	hes[3] = -hes[1];
	hes[4] = -hes[2];
	hes[5] = hes[2];
	hes[6] = hes[1];
	hes[7] = hes[2];
	hes[8] = -hes[2];
	hes[9] = hes[2];

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 3 Imag
	res = M0*vt[5] - data[6*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[5];
	jac[1] = M0*vt[6]*k;
	jac[2] = -jac[1];
	jac[3] = jac[1];

	hes[0] = 0.0f;
	hes[1] = vt[6]*k*res;
	hes[2] = -M0*vt[5]*k*k*res;
	hes[3] = -hes[1];
	hes[4] = -hes[2];
	hes[5] = hes[2];
	hes[6] = hes[1];
	hes[7] = hes[2];
	hes[8] = -hes[2];
	hes[9] = hes[2];

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 4 Real
	res = M0*vt[6] - data[7*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[6];
	jac[1] = M0*vt[7]*k;
	jac[2] = -jac[1];
	jac[3] = -jac[1];

	hes[0] = 0.0f;
	hes[1] = vt[7]*k*res;
	hes[2] = -M0*vt[6]*k*k*res;
	hes[3] = -hes[1];
	hes[4] = -hes[2];
	hes[5] = hes[2];
	hes[6] = -hes[1];
	hes[7] = -hes[2];
	hes[8] = hes[2];
	hes[9] = hes[2];

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);

	
	// Encoding 4 Imag
	res = M0*vt[7] - data[8*Nprobs+tid];
	f[tid] += res * res;
	
	jac[0] = vt[7];
	jac[1] = -M0*vt[6]*k;
	jac[2] = -jac[1];
	jac[3] = -jac[1];

	hes[0] = 0.0f;
	hes[1] = -vt[6]*k*res;
	hes[2] = -M0*vt[7]*k*k*res;
	hes[3] = -hes[1];
	hes[4] = -hes[2];
	hes[5] = hes[2];
	hes[6] = -hes[1];
	hes[7] = -hes[2];
	hes[8] = hes[2];
	hes[9] = hes[2];

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);
	
}
""")

		return fghhl_temp.render()

	def get_deps(self):
		return list()


def constant_from_venc(venc):
	return venc

def get_encode_matrix(k):
	Emat = k * np.array(
		[
			[ 0,  0,  0],
			[-1, -1, -1],
			[ 1,  1, -1],
			[ 1, -1,  1],
			[-1,  1,  1]
		], dtype=np.float32)
	return Emat

def _enc_to_vel_linear(image, venc):
	Emat = get_encode_matrix(constant_from_venc(venc))
	pEmat = np.linalg.pinv(Emat)

	im_size = (image.shape[1], image.shape[2], image.shape[3])
	nvoxel = math.prod(list(im_size))

	base_phase = np.angle(image[0,...])
	base_corrected = (image * np.exp(-1j*base_phase)[np.newaxis,...]).reshape((5,nvoxel))

	phases = np.angle(base_corrected)
	imageout = pEmat @ phases

	mag = np.mean(np.abs(base_corrected), axis=0)[np.newaxis,:]

	imageout = np.concatenate([mag, imageout], axis=0)

	return (imageout, base_corrected)

def enc_to_vel_linear(images, venc):
	images_out = np.empty((images.shape[0], 4, images.shape[2], images.shape[3], images.shape[4]))
	for i in range(images.shape[0]):
		print('Frame: ', i)
		image = images[i,...]
		images_out[i,...] = _enc_to_vel_linear(image, venc)[0].reshape((4,images.shape[2], 
								  images.shape[3], images.shape[4]))

	return images_out

def _enc_to_vel_nonlinear(image, venc):
	im_size = (image.shape[1], image.shape[2], image.shape[3])
	nvoxel = math.prod(list(im_size))

	imageout, base_corrected = _enc_to_vel_linear(image, venc)

	parscu = cp.array(imageout)
	datacu: cp.array
	constscu = cp.array([constant_from_venc(venc)])
	lower_bound_cu = -1e6*cp.ones_like(parscu)
	upper_bound_cu = 1e6*cp.ones_like(parscu)

	if True:
		Edata = np.empty((9,nvoxel), dtype=np.float32)
		Edata[0,:] = np.real(base_corrected[0,:])
		Edata[1,:] = np.real(base_corrected[1,:])
		Edata[2,:] = np.imag(base_corrected[1,:])
		Edata[3,:] = np.real(base_corrected[2,:])
		Edata[4,:] = np.imag(base_corrected[2,:])
		Edata[5,:] = np.real(base_corrected[3,:])
		Edata[6,:] = np.imag(base_corrected[3,:])
		Edata[7,:] = np.real(base_corrected[4,:])
		Edata[8,:] = np.imag(base_corrected[4,:])
		datacu = cp.array(Edata)

	fobj = EncVelF()
	gradfobj = EncVelF_GradF()

	fgradfobj = F_GradF(fobj, gradfobj, 4, 0, "enc_vel_fgradf")
	solm = SecondOrderLevenbergMarquardt(None, fgradfobj, 9, cp.float32, write_to_file=True)

	solm.setup(parscu, constscu, datacu, lower_bound_cu, upper_bound_cu)
	solm.run(100)
	#last_error = solm.last_f.get()

	pars_out = parscu.get()

	return pars_out.reshape((4,im_size[0],im_size[1],im_size[2]))

def enc_to_vel_nonlinear(images, venc):
	images_out = np.empty((images.shape[0], 4, images.shape[2], images.shape[3], images.shape[4]))
	for i in range(images.shape[0]):
		print('Frame: ', i)
		image = images[i,...]
		images_out[i,...] = _enc_to_vel_nonlinear(image, venc)

	return images_out



def test_data():
	nvox = 40000
	venc = 0.15
	true_venc = 1 / np.sqrt(3)
	vel = venc*(-1.0 + 2*np.random.rand(3,nvox)).astype(np.float32)

	phase = A @ vel

	mag = 10*np.random.rand(1,nvox).astype(np.float32)

	pars_true = np.concatenate([mag, vel], axis=0)

	enc = mag * (np.cos(phase) + 1j*np.sin(phase))


	parscu = cp.ascontiguousarray(cp.random.rand(4,nvox).astype(cp.float32))
	lower_bound_cu = cp.ascontiguousarray(-true_venc*cp.ones((4,nvox), dtype=np.float32))
	lower_bound_cu[0,:] = 0.0
	upper_bound_cu = cp.ascontiguousarray(true_venc*cp.ones((4,nvox), dtype=np.float32))
	upper_bound_cu[0,:] = 1e6
	datacu = cp.ascontiguousarray(cp.array(Edata))

	#print('pars_before: ', parscu[:,0].get())
	#print('data_before: ', datacu[:,0].get())
	#print('pars_true: ', pars_true[:,0])
	#print('Relative error: ', np.linalg.norm(pars_out - pars_true) / np.linalg.norm(pars_true))
	#print('Maximum Velocity Parameter error: ', np.abs(pars_out[1:3,:] - pars_true[1:3,:]).max())
	#print('Maximum Magnitude Parameter error: ', np.abs(pars_out[0,:] - pars_true[0,:]).max())
	#print('Maximum Objective Error: ', np.abs(last_error).max())
	#print('Maximum Velocity Value: ', np.abs(pars_out[1:3,:]).max())
	#print('Hello')



img_full = np.array([0])
with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_real.h5', "r") as f:
	img_full = f['images'][()]

img_full = img_full / np.mean(np.abs(img_full))

pu.image_nd(img_full)

img_vel = enc_to_vel_nonlinear(img_full, 1)

pu.image_nd(img_vel)

vmag = np.sqrt(img_vel[:,1,...]**2 + img_vel[:,2,...]**2 + img_vel[:,3,...]**2)

pu.image_nd(vmag)

cd = ic.get_CD(img_vel, 1)

pu.image_nd(cd)

print('Hello')