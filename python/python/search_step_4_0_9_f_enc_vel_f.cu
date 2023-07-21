
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
		printf("  f: %f \n", f[tid]);
	}
	*/
}

__device__
void search_step_4_0_9_f_enc_vel_f(const float* consts, const float* data, const float* lower_bound, const float* upper_bound, 
	float* best_p, const float* p, float* best_f, float* f, int tid, int Nprobs)
{
	
	// Calculate error at new params
	{
		enc_to_vel_f(p, consts, data, f, tid, Nprobs);
	}

	// Check if new params is better and keep them in that case
	{
		if (f[tid] < best_f[tid]) {
			for (int i = 0; i < 4; ++i) {
				int iidx = i*Nprobs+tid;
				best_p[iidx] = p[iidx];
			}
			best_f[tid] = f[tid];
		}
	}

}

extern "C" __global__
void k_search_step_4_0_9_f_enc_vel_f(const float* consts, const float* data, const float* lower_bound, const float* upper_bound, 
	float* best_p, float* p, float* best_f, float* f, int Nprobs) 
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < Nprobs) {
		search_step_4_0_9_f_enc_vel_f(consts, data, lower_bound, upper_bound, best_p, p, best_f, f, tid, Nprobs);
	}
}
