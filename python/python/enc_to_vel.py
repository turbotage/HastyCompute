import sys
import numpy as np
import cupy as cp

import math
import h5py
import hastypy.util.plot_utility as pu
import hastypy.util.image_creation as ic

sys.path.append('C:\\Users\\TurboTage\\Documents\\GitHub\\pycompute')

from jinja2 import Template

from pycompute.cuda.lsqnonlin import F, FGradF, F_FGradF, SecondOrderLevenbergMarquardt, SecondOrderRandomSearch
from pycompute.cuda.cuda_program import CudaFunction, CudaTensor
from pycompute.cuda import cuda_program as cudap

class EncVelF(CudaFunction):
	def __init__(self):
		self.run_func = None
		self.write_to_file = False
	
	def get_device_funcid(self):
		return 'enc_to_vel_f'
	
	def get_kernel_funcid(self):
		funcid = self.get_device_funcid()
		return 'k_' + funcid

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

	/*
	if (tid == 0) {
		printf("M0: %f, vx: %f, vy: %f, vz: %f, ", M0, vx, vy, vz);
		printf(" k: %f,  ", consts[0]);
	}
	*/

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
	
	/*
	if (tid == 0) {
		printf("  f: %f \\n", f[tid]);
	}
	*/
}
""")

		return f_temp.render()

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const float* params, const float* consts, const float* data, 
	float* f, int Nprobs)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nprobs) {
		{{dfuncid}}(params, consts, data, f, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()
		return temp.render(funcid=fid, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code

	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()

	def run(self, pars, consts, data, f):
		if self.run_func == None:
			self.build()

		Nprobs = pars.shape[1]

		Nthreads = 32
		blockSize = math.ceil(Nprobs / Nthreads)

		self.run_func((blockSize,),(Nthreads,),(pars, consts, data, f, Nprobs))

class EncVel_FGradF(CudaFunction):
	def __init__(self):
		return
	
	def get_device_funcid(self):
		return 'enc_to_vel_fghhl'
	
	def get_kernel_funcid(self):
		funcid = self.get_device_funcid()
		return 'k_' + funcid

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

			/*
			if (tid == 0) {
				printf("h: %f, ", hl[lidx]);
			}
			*/
		}

		g[j*Nprobs+tid] += jac[j] * res;
	}
}

#define FULL_HESS

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

	float k = consts[0];
	
	/*
	if (tid == 0) {
		printf("M0: %f, vx: %f, vy: %f, vz: %f \\n", M0, vx, vy, vz);
		//for (int i = 0; i < 9; ++i) {
		//	printf("%f, ", data[i*Nprobs+tid]);
		//}
		//printf("\\n");
		//printf("k: %f", consts[0]);
	}
	*/
	
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

#ifdef FULL_HESS
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
#endif
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 1 Imag
	res = M0*vt[1] - data[2*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[1];
	jac[1] = -M0*vt[0]*k;
	jac[2] = jac[1];
	jac[3] = jac[1];

#ifdef FULL_HESS
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
#endif

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);

	
	// Encoding 2 Real
	res = M0*vt[2] - data[3*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[2];
	jac[1] = -M0*vt[3]*k;
	jac[2] = jac[1];
	jac[3] = -jac[1];

#ifdef FULL_HESS
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
#endif
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 2 Imag
	res = M0*vt[3] - data[4*Nprobs+tid];
	f[tid] += res * res;
	
	jac[0] = vt[3];
	jac[1] = M0*vt[2]*k;
	jac[2] = jac[1];
	jac[3] = -jac[1];

#ifdef FULL_HESS
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
#endif
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);

	// Encoding 3 Real
	res = M0*vt[4] - data[5*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[4];
	jac[1] = -M0*vt[5]*k;
	jac[2] = -jac[1];
	jac[3] = jac[1];

#ifdef FULL_HESS
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
#endif

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 3 Imag
	res = M0*vt[5] - data[6*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[5];
	jac[1] = M0*vt[6]*k;
	jac[2] = -jac[1];
	jac[3] = jac[1];

#ifdef FULL_HESS
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
#endif

	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);


	// Encoding 4 Real
	res = M0*vt[6] - data[7*Nprobs+tid];
	f[tid] += res * res;

	jac[0] = vt[6];
	jac[1] = M0*vt[7]*k;
	jac[2] = -jac[1];
	jac[3] = -jac[1];

#ifdef FULL_HESS
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
#endif
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);

	
	// Encoding 4 Imag
	res = M0*vt[7] - data[8*Nprobs+tid];
	f[tid] += res * res;
	
	jac[0] = vt[7];
	jac[1] = -M0*vt[6]*k;
	jac[2] = -jac[1];
	jac[3] = -jac[1];

#ifdef FULL_HESS
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
#endif
	
	ghhl_add(jac, hes, res, lambda, h, hl, g, tid, Nprobs);
	
}
""")

		return fghhl_temp.render()

	def get_kernel_code(self):
		temp = Template(
"""
extern \"C\" __global__
void {{funcid}}(const float* params, const float* consts, const float* data, const float* lam,
	float* f, float* g, float* h, float* hl, int Nprobs)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < Nprobs) {
		{{dfuncid}}(params, consts, data, lam, f, g, h, hl, tid, Nprobs);
	}
}
""")
		fid = self.get_kernel_funcid()
		dfid = self.get_device_funcid()
		return temp.render(funcid=fid, dfuncid=dfid)

	def get_full_code(self):
		code = self.get_device_code() + '\n'
		code += self.get_kernel_code()
		return code
	
	def build(self):
		cc = cudap.code_gen_walking(self, "")
		if self.write_to_file:
			with open(self.get_device_funcid() + '.cu', "w") as f:
				f.write(cc)
		try:
			self.mod = cp.RawModule(code=cc)
			self.run_func = self.mod.get_function(self.get_kernel_funcid())
		except:
			with open("on_compile_fail.cu", "w") as f:
				f.write(cc)
			raise
		return cc

	def get_deps(self):
		return list()


def constant_from_venc(venc):
	return (np.pi / (venc * np.sqrt(3)))

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
	print(imageout.min())
	print(imageout.max())
	print(venc * np.sqrt(3))

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
	constscu = cp.array([constant_from_venc(venc)]).astype(cp.float32)
	lower_bound_cu = cp.empty_like(parscu)
	lower_bound_cu[0,...] = cp.array(imageout[0,...] / 3)
	lower_bound_cu[1:,...] = -venc / 2 #-np.sqrt(3)*venc / 5

	upper_bound_cu = cp.empty_like(parscu)
	upper_bound_cu[0,...] = cp.array(imageout[0,...] * 3)
	upper_bound_cu[1:,...] = venc / 2 #np.sqrt(3)*venc / 5

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
	gradfobj = EncVel_FGradF()

	fgradfobj = F_FGradF(F(fobj, 4, "enc_vel_f"), FGradF(gradfobj, 4, "enc_vel_gradf"), "enc_vel_fgradf")
	
	#solm = SecondOrderLevenbergMarquardt(None, fgradfobj, 9, cp.float32, write_to_file=True)

	solm = SecondOrderRandomSearch(None, fgradfobj, 9, cp.float32, write_to_file=True)
	solm.setup(constscu, datacu, lower_bound_cu, upper_bound_cu)
	solm.run(iters=100, lm_iters=80, tol=cp.float32(0.0))
	solm.test(cp.array(imageout), lm_iters=80, tol=cp.float32(0.0))
	solm.direct_test(cp.array(imageout))

	#f = cp.zeros((1, parscu.shape[1]), dtype=cp.float32)
	#fobj.run(cp.array(imageout), constscu, datacu, f)

	#print('f: ', f[0,0])


	f = cp.zeros((1, parscu.shape[1]), dtype=cp.float32)
	fobj.run(cp.array(imageout), constscu, datacu, f)

	print('linear pars: ', imageout[:,0], '  f: ', f[0,0])

	#pars_out = parscu.get()
	pars_out = solm.best_pars_t.get()

	return pars_out.reshape((4,im_size[0],im_size[1],im_size[2]))

def enc_to_vel_nonlinear(images, venc):
	images_out = np.empty((images.shape[0], 4, images.shape[2], images.shape[3], images.shape[4]))
	for i in range(images.shape[0]):
		print('Frame: ', i)
		image = images[i,...]
		images_out[i,...] = _enc_to_vel_nonlinear(image, venc)

	return images_out

def objective_surface(noiceperc=0.0):
	nsamps = 20000

	nvox = 1
	venc = 0.15
	true_venc = 1 / np.sqrt(3)
	vel = venc*(-1.0 + 2*np.random.rand(3,nvox)).astype(np.float32) / np.sqrt(3)

	phase = get_encode_matrix(constant_from_venc(venc)) @ vel

	mag = 10*np.random.rand(1,nvox).astype(np.float32)

	pars_true = np.concatenate([mag, vel], axis=0)

	enc = mag * (np.cos(phase) + 1j*np.sin(phase))

	noice: np.array
	if noiceperc != 0.0:
		real_noice = np.random.normal(scale=np.abs(np.real(enc))*noiceperc)
		imag_noice = np.random.normal(scale=np.abs(np.imag(enc))*noiceperc)

		enc += (real_noice + 1j*imag_noice)

	datacu: cp.array
	if True:
		angle = np.angle(enc[0,0])
		enc *= np.exp(-1j*angle)

		Edata = np.empty((9,nsamps), dtype=np.float32)
		Edata[0,:] = np.real(enc[0,:])
		Edata[1,:] = np.real(enc[1,:])
		Edata[2,:] = np.imag(enc[1,:])
		Edata[3,:] = np.real(enc[2,:])
		Edata[4,:] = np.imag(enc[2,:])
		Edata[5,:] = np.real(enc[3,:])
		Edata[6,:] = np.imag(enc[3,:])
		Edata[7,:] = np.real(enc[4,:])
		Edata[8,:] = np.imag(enc[4,:])
		datacu = cp.array(Edata)

	#datacu = cp.repeat(datacu, nsamps, axis=1)
	constscu = cp.array([constant_from_venc(venc)]).astype(cp.float32)
	parscu = cp.empty((4, nsamps), dtype=cp.float32)
	parscu[0,:] = pars_true[0,0]
	parscu[1,:] = pars_true[1,0]
	#parscu[2,:] = -np.sqrt(3)*venc + 2*np.sqrt(3)*venc*cp.random.rand(nsamps).astype(cp.float32)
	#parscu[3,:] = -np.sqrt(3)*venc + 2*np.sqrt(3)*venc*cp.random.rand(nsamps).astype(cp.float32)
	parscu[2,:] = -venc + 2*venc*cp.random.rand(nsamps).astype(cp.float32)
	parscu[3,:] = -venc + 2*venc*cp.random.rand(nsamps).astype(cp.float32)

	fobj = EncVelF()
	f = cp.empty((1, nsamps), dtype=cp.float32)
	fobj.run(parscu, constscu, datacu, f)
	f = cp.nan_to_num(f, nan=1e28, posinf=1e28, neginf=1e28)

	pu.obj_surface(parscu[2:,:].get(), f.get())

def test_data(noiceperc=0.0):
	nx = 32
	nvox = nx*nx*nx
	venc = 0.15
	true_venc = 1 / np.sqrt(3)
	vel = venc*(-1.0 + 2*np.random.rand(3,nvox)).astype(np.float32) / 10.0 #np.sqrt(3) / 5

	phase = get_encode_matrix(constant_from_venc(venc)) @ vel

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

	linear_pars = enc_to_vel_linear(enc, venc)
	nonlinear_pars = enc_to_vel_nonlinear(enc, venc)

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

	#print('Hello')


def test():

	img_full = np.array([0])
	with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_simulated.h5', "r") as f:
		img_full = f['images'][()]

	img_full = img_full# / np.mean(np.abs(img_full))

	pu.image_nd(img_full)

	true_images: np.array([0])
	with h5py.File('D:\\4DRecon\\dat\\dat2\\images_encs_20f_cropped_interpolated.h5', "r") as f:
		true_images = f['images'][()]

	pu.image_nd(true_images - img_full)


	img_vel = enc_to_vel_linear(img_full, 1)

	mean_mag = np.mean(img_full, axis=(0,1))
	#with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_20f_mag.h5', "w") as f:
	#	f.create_dataset('images', data=np.transpose(mean_mag, (2,1,0)))
	
	#with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_20f.h5', "w") as f:
	#	f.create_dataset('images', data=np.transpose(img_vel, (4,3,2,1,0)))

	pu.image_nd(img_vel)

	vmag = np.sqrt(img_vel[:,1,...]**2 + img_vel[:,2,...]**2 + img_vel[:,3,...]**2)

	pu.image_nd(vmag)

	cd = ic.get_CD(img_vel, 1)

	pu.image_nd(cd)

	print('Hello')

def test_real():

	img_full = np.array([0])
	with h5py.File('D:\\4DRecon\\dat\\dat2\\my_framed_real.h5', "r") as f:
		img_full = f['images'][()]

	#pu.image_nd(img_full)

	img_vel = enc_to_vel_linear(img_full, 500)

	mean_mag = np.mean(img_full, axis=(0,1))

	vmag = np.sqrt(img_vel[:,1,...]**2 + img_vel[:,2,...]**2 + img_vel[:,3,...]**2)

	pu.image_nd(vmag)

	cd = ic.get_CD(img_vel, 500)

	pu.image_nd(cd)

	print('Hello')


if __name__ == '__main__':
	#objective_surface(0.0)
	#test_data(noiceperc=0.2)
	#test()
	test_real()