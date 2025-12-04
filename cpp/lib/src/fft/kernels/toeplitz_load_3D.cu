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
	const cuFloatComplex* input,
	cuFloatComplex* output,
	cuFloatComplex* scratch, 
	const cuFloatComplex* mult1,
	const cuFloatComplex* mult2,
	int input_output_mult_type,
	int input_mult1_type,
	int output_mult1_type,
	int input_mult2_type,
	int output_mult2_type,
	int batch_in,
	int batch_out,
	int NX, int NY, int NZ,
	bool accumulate) 
{
	const long int idx = blockIdx.x * blockDim.x + threadIdx.x; // Same as z * NX * NY + y * NX + x

	if (idx < NX * NY * NZ) {
		const long int x = idx % NX;
		const long int y = (idx / NX) % NY;
		const long int z = idx / (NX * NY);

		const long int NX2 = 2 * NX;
		const long int NY2 = 2 * NY;
		const long int NZ2 = 2 * NZ;

		cuFloatComplex temp;
		if (batch_out >= 0) {
			const long int batch_idx = batch_out * NX * NY * NZ + idx;
			const long int sx = x + NX - 1;
			const long int sy = y + NY - 1;
			const long int sz = z + NZ - 1;
			temp = scratch[sz * NX2 * NY2 + sy * NX2 + sx];
			if (mult1 != nullptr) {
				temp = complex_mult_toeplitz_load(temp, mult1[idx], output_mult1_type);
			}
			if (mult2 != nullptr) {
				temp = complex_mult_toeplitz_load(temp, mult2[idx], output_mult2_type);
			}
			if (input_output_mult_type != 0) {
				temp = complex_mult_toeplitz_load(temp, input[batch_idx], input_output_mult_type);
			}
			if (accumulate) {
				cuFloatComplex prev = output[batch_idx];
				temp.x += prev.x;
				temp.y += prev.y;
			}
			output[batch_idx] = temp;
		}
		if (batch_in >= 0) {
			const long int batch_idx = batch_in * NX * NY * NZ + idx;
			temp = input[batch_idx];
			if (mult1 != nullptr) {
				temp = complex_mult_toeplitz_load(temp, mult1[idx], input_mult1_type);
			}
			if (mult2 != nullptr) {
				temp = complex_mult_toeplitz_load(temp, mult2[idx], input_mult2_type);
			}
			scratch[z * NX2 * NY2 + y * NX2 + x] = temp;
		}
	}
}