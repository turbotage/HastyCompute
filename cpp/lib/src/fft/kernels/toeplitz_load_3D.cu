struct cuFloatComplex { float x, y; };
__device__ inline cuFloatComplex complex_mult_toeplitz_load(cuFloatComplex a, cuFloatComplex b, int mult_type)
{
	cuFloatComplex c;
	if (mult_type == 1) {
		c.x = a.x * b.x - a.y * b.y;
		c.y = a.x * b.y + a.y * b.x;
	} else if (mult_type == 2) {
		c.x = a.x * b.x + a.y * b.y;
		c.y = a.y * b.x - a.x * b.y;
	} else {
		c = a;
	}
	return c;
}


extern "C" __global__ void toeplitz_load_3D(
	cuFloatComplex*       output,
	cuFloatComplex*       scratch,
	const cuFloatComplex* mult1,
	const cuFloatComplex* mult2,
	const cuFloatComplex* mult3,
	const cuFloatComplex* batch_mult,
	int input_mult1_type,  int output_mult1_type,
	int input_mult2_type,  int output_mult2_type,
	int input_mult3_type,  int output_mult3_type,
	int input_bm_type,     int output_bm_type,
	int batch_in,
	int batch_out,
	int NX, int NY, int NZ,
	bool accumulate,
	bool output_single_batch)
{
	const long int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < NX * NY * NZ) {
		const long int x = idx % NX;
		const long int y = (idx / NX) % NY;
		const long int z = idx / (NX * NY);

		const long int NX2 = 2 * NX;
		const long int NY2 = 2 * NY;

		cuFloatComplex temp;

		if (batch_out >= 0) {
			const long int out_batch_idx = (output_single_batch ? 0L : (long int)batch_out) * NX * NY * NZ + idx;
			const long int bm_batch_idx  = (long int)batch_out * NX * NY * NZ + idx;
			temp = scratch[z * NX2 * NY2 + y * NX2 + x];
			if (mult1 && output_mult1_type != 0)
				temp = complex_mult_toeplitz_load(temp, mult1[idx], output_mult1_type);
			if (mult2 && output_mult2_type != 0)
				temp = complex_mult_toeplitz_load(temp, mult2[idx], output_mult2_type);
			if (mult3 && output_mult3_type != 0)
				temp = complex_mult_toeplitz_load(temp, mult3[idx], output_mult3_type);
			if (batch_mult && output_bm_type != 0)
				temp = complex_mult_toeplitz_load(temp, batch_mult[bm_batch_idx], output_bm_type);
			if (accumulate) {
				output[out_batch_idx].x += temp.x;
				output[out_batch_idx].y += temp.y;
			} else {
				output[out_batch_idx] = temp;
			}
		}

		if (batch_in >= 0) {
			const long int bm_batch_idx = (long int)batch_in * NX * NY * NZ + idx;
			temp.x = 1.0f; temp.y = 0.0f;
			if (mult1 && input_mult1_type != 0)
				temp = complex_mult_toeplitz_load(temp, mult1[idx], input_mult1_type);
			if (mult2 && input_mult2_type != 0)
				temp = complex_mult_toeplitz_load(temp, mult2[idx], input_mult2_type);
			if (mult3 && input_mult3_type != 0)
				temp = complex_mult_toeplitz_load(temp, mult3[idx], input_mult3_type);
			if (batch_mult && input_bm_type != 0)
				temp = complex_mult_toeplitz_load(temp, batch_mult[bm_batch_idx], input_bm_type);
			scratch[z * NX2 * NY2 + y * NX2 + x] = temp;
		}
	}
}
