

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <nvrtc.h>
#include <cuda.h>
#define VKFFT_BACKEND 1
#include "vkFFT.h"


#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; exit(1); } }
#define CUFFT_CHECK(err) { if (err != CUFFT_SUCCESS) { std::cerr << "cuFFT error: " << err << "\n"; exit(1); } }

import std;

// NVRTC kernel string for pointwise multiplication
const char* multiply_kernel_code = R"(
struct cuFloatComplex { float x, y; };
extern "C" __global__ void pointwise_multiply(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* out, float scaling, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		cuFloatComplex va = a[idx];
		cuFloatComplex vb = b[idx];
		out[idx].x = (va.x * vb.x - va.y * vb.y) / scaling;
		out[idx].y = (va.x * vb.y + va.y * vb.x) / scaling;
	}
}
)";

void launch_pointwise_multiply(const cuFloatComplex* a, const cuFloatComplex* b, cuFloatComplex* out, float scaling, int N) {
	static CUmodule cuModule = nullptr;
	static CUfunction cuFunction = nullptr;
	if (!cuModule) {
		nvrtcProgram prog;
		nvrtcCreateProgram(&prog, multiply_kernel_code, "multiply_kernel.cu", 0, nullptr, nullptr);
		nvrtcResult compileResult = nvrtcCompileProgram(prog, 0, nullptr);
		if (compileResult != NVRTC_SUCCESS) {
			size_t logSize;
			nvrtcGetProgramLogSize(prog, &logSize);
			std::vector<char> log(logSize);
			nvrtcGetProgramLog(prog, log.data());
			std::cerr << "NVRTC compile error: " << log.data() << std::endl;
			exit(1);
		}
		size_t ptxSize;
		nvrtcGetPTXSize(prog, &ptxSize);
		std::vector<char> ptx(ptxSize);
		nvrtcGetPTX(prog, ptx.data());
		nvrtcDestroyProgram(&prog);

		cuInit(0);
		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);
		CUcontext cuContext;
		cuCtxGetCurrent(&cuContext);
		// If no context, create one using CUDA 13 API
		if (!cuContext) {
			CUctxCreateParams params = {};
			cuCtxCreate(&cuContext, &params, 0, cuDevice);
		}
		cuModuleLoadData(&cuModule, ptx.data());
		cuModuleGetFunction(&cuFunction, cuModule, "pointwise_multiply");
	}
	void* args[] = { (void*)&a, (void*)&b, (void*)&out, (void*)&scaling, (void*)&N };
	int threads = 256;
	int blocks = (N + threads - 1) / threads;
	cuLaunchKernel(cuFunction, blocks, 1, 1, threads, 1, 1, 0, 0, args, 0);
	/*
	CUresult syncResult = cuCtxSynchronize();
	if (syncResult != CUDA_SUCCESS) {
		std::cerr << "cuCtxSynchronize failed: " << syncResult << std::endl;
		exit(1);
	}
	*/
}



bool compare_results(const std::vector<cuFloatComplex>& a, const std::vector<cuFloatComplex>& b, float tol = 1e-3f) {
	for (size_t i = 0; i < a.size(); ++i) {
		if (std::abs(a[i].x - b[i].x) > tol || std::abs(a[i].y - b[i].y) > tol) {
			std::cerr << "Mismatch at " << i << ": " << a[i].x << ", " << a[i].y << " vs " << b[i].x << ", " << b[i].y << std::endl;
			return false;
		}
	}
	return true;
}

bool print_n(const std::vector<cuFloatComplex>& output, const std::vector<cuFloatComplex>& input, int n = 10) {
    for (int i = 0; i < n && i < output.size(); ++i) {
        std::cout << "output: (" << output[i].x << ", " << output[i].y << ") input: (" << input[i].x << ", " << input[i].y << ")\n";
    }
    std::cout << std::endl;
    return true;
}

enum KernelType {
	IDENTITY = 0,
	OTHER = 1
};

std::array<std::vector<cuFloatComplex>,3> generate_data(int NX, int NY, int NZ, KernelType kernel_type = IDENTITY) {
    const int N = NX * NY * NZ;
	const int NX2 = 2 * NX;
	const int NY2 = 2 * NY;
	const int NZ2 = 2 * NZ;
	const int N2 = NX2 * NY2 * NZ2;
	std::vector<cuFloatComplex> h_cu_input(N2), h_vk_input(N2), h_kernel(N2);
	cuFloatComplex temp;
	for (int z = 0; z < NZ2; ++z) {
		for (int y = 0; y < NY2; ++y) {
			for (int x = 0; x < NX2; ++x) {
				int idx = z * NX2 * NY2 + y * NX2 + x;
				temp = { std::sin(float(idx*idx)), std::cos(float(idx)) };
                if ((z < NZ) && (y < NY) && (x < NX)) {
                    h_cu_input[idx] = temp;
					h_vk_input[idx] = temp;
                } else {
					//h_cu_input[idx] = temp; //{0.0f, 0.0f};
					h_cu_input[idx] = {0.0f, 0.0f};
					h_vk_input[idx] = {0.0f, 0.0f}; //temp;
                }
				if (kernel_type == IDENTITY)
					h_kernel[idx] = { 1.0f, 0.0f };
				else {
					h_kernel[idx] = { float(idx) / N - 0.5f, 0.0f };
				}
			}
		}
	}
    return {h_cu_input, h_vk_input, h_kernel};
}

std::pair<VkFFTConfiguration, VkFFTConfiguration> get_convolution_configurations(int NX, int NY, int NZ)
{
	const int N = 8 * NX * NY * NZ;
    static int cuda_device_id = 0;
    static pfUINT buf_size = N * sizeof(cuFloatComplex);
	
	int coalescedMemory = 128;

	VkFFTConfiguration configuration_kernel = {};
	configuration_kernel.device = &cuda_device_id;
	configuration_kernel.FFTdim = 3;
	configuration_kernel.size[0] = 2 * NX;
	configuration_kernel.size[1] = 2 * NY;
	configuration_kernel.size[2] = 2 * NZ;
	configuration_kernel.bufferSize = &buf_size;
	configuration_kernel.kernelConvolution = 1;
	configuration_kernel.coalescedMemory = coalescedMemory;

    VkFFTConfiguration configuration_convolution = {};
    configuration_convolution.device = &cuda_device_id;
    configuration_convolution.FFTdim = 3;
    configuration_convolution.size[0] = 2 * NX;
    configuration_convolution.size[1] = 2 * NY;
    configuration_convolution.size[2] = 2 * NZ;
    configuration_convolution.performZeropadding[0] = 1;
    configuration_convolution.performZeropadding[1] = 1;
    configuration_convolution.performZeropadding[2] = 1;
    configuration_convolution.fft_zeropad_left[0] = 0;
    configuration_convolution.fft_zeropad_left[1] = 0;
    configuration_convolution.fft_zeropad_left[2] = 0;
    configuration_convolution.fft_zeropad_right[0] = NX;
    configuration_convolution.fft_zeropad_right[1] = NY;
    configuration_convolution.fft_zeropad_right[2] = NZ;
    configuration_convolution.bufferSize = &buf_size;
	configuration_convolution.performConvolution = 1;
	configuration_convolution.coalescedMemory = coalescedMemory;

	return {configuration_kernel, configuration_convolution};
}


void benchmark_fft(int NX, int NY, int NZ, bool precission = true, bool speed = false) {
	const int N = NX * NY * NZ;
	const int NX2 = 2 * NX;
	const int NY2 = 2 * NY;
	const int NZ2 = 2 * NZ;
	const int N2 = NX2 * NY2 * NZ2;

	const int iterations = 20;

    auto gendat = generate_data(NX, NY, NZ, OTHER);

	std::vector<cuFloatComplex> h_cu_output(N2), h_vk_output(N2);
	cuFloatComplex *d_kernel;
	{
		CUDA_CHECK(cudaMalloc(&d_kernel, N2 * sizeof(cuFloatComplex)));
		CUDA_CHECK(cudaMemcpy(d_kernel, gendat[2].data(), N2 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
		cufftHandle plan;
		CUFFT_CHECK(cufftPlan3d(&plan, NX2, NY2, NZ2, CUFFT_C2C));
		cufftExecC2C(plan, (cufftComplex*)d_kernel, (cufftComplex*)d_kernel, CUFFT_INVERSE);
		CUDA_CHECK(cudaDeviceSynchronize());
		CUFFT_CHECK(cufftDestroy(plan));
	}

	// cuFFT benchmark
	{
		// Create the kernel
		cuFloatComplex *cu_kernel;
		{
			CUDA_CHECK(cudaMalloc(&cu_kernel, N2 * sizeof(cuFloatComplex)));
			cufftHandle plan;
			CUFFT_CHECK(cufftPlan3d(&plan, NX2, NY2, NZ2, CUFFT_C2C));
			cufftExecC2C(plan, (cufftComplex*)d_kernel, (cufftComplex*)cu_kernel, CUFFT_FORWARD);
			CUDA_CHECK(cudaDeviceSynchronize());
			CUFFT_CHECK(cufftDestroy(plan));
		}

		cuFloatComplex *d_cu_input;
		CUDA_CHECK(cudaMalloc(&d_cu_input, N2 * sizeof(cuFloatComplex)));
		CUDA_CHECK(cudaMemcpy(d_cu_input, gendat[0].data(), N2 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

		cufftHandle plan;
		CUFFT_CHECK(cufftPlan3d(&plan, NX2, NY2, NZ2, CUFFT_C2C));
		if (precission) {
			cufftExecC2C(plan, (cufftComplex*)d_cu_input, (cufftComplex*)d_cu_input, CUFFT_FORWARD);
			launch_pointwise_multiply(d_cu_input, cu_kernel, d_cu_input, (float)N2*N2, N2);
			cufftExecC2C(plan, (cufftComplex*)d_cu_input, (cufftComplex*)d_cu_input, CUFFT_INVERSE);
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaMemcpy(h_cu_output.data(), d_cu_input, N2 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
		}
		if (speed) {
			CUDA_CHECK(cudaDeviceSynchronize());
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < iterations; ++i) {
				cufftExecC2C(plan, (cufftComplex*)d_cu_input, (cufftComplex*)d_cu_input, CUFFT_FORWARD);
				launch_pointwise_multiply(d_cu_input, cu_kernel, d_cu_input, (float)N2*N2, N2);
				cufftExecC2C(plan, (cufftComplex*)d_cu_input, (cufftComplex*)d_cu_input, CUFFT_INVERSE);
			}
			CUDA_CHECK(cudaDeviceSynchronize());
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = end - start;
			std::cout << "cuFFT " << NX << "x" << NY << "x" << NZ << " average time over " << iterations << " iterations: " 
					  << (elapsed.count() * 1000.0 / iterations) << " ms\n";
		}

		CUFFT_CHECK(cufftDestroy(plan));
		CUDA_CHECK(cudaFree(cu_kernel));
		CUDA_CHECK(cudaFree(d_cu_input));
	}

	// VkFFT benchmark
	{
		auto configs = get_convolution_configurations(NX, NY, NZ);

		// Create the kernel
		cuFloatComplex *vk_kernel;
		{
			CUDA_CHECK(cudaMalloc(&vk_kernel, N2 * sizeof(cuFloatComplex)));
			VkFFTApplication app = {};
			VkFFTResult res = initializeVkFFT(&app, configs.first);
			if (res != VKFFT_SUCCESS) { 
				std::cerr << "VkFFT init failed\n"; return; 
			}
			static void* buffer_ptrs[1];
			buffer_ptrs[0] = vk_kernel;
			static void* kernel_ptrs[1];
			kernel_ptrs[0] = d_kernel;
			VkFFTLaunchParams launchParams = {};
			launchParams.buffer = buffer_ptrs;
			launchParams.kernel = kernel_ptrs;
			res = VkFFTAppend(&app, 0, &launchParams);
			if (res != VKFFT_SUCCESS) { 
				std::cerr << "VkFFT run failed, kernelCreation, code: " << res << "\n"; return; 
			}
			deleteVkFFT(&app);
		}

		cuFloatComplex *d_vk_input;
		CUDA_CHECK(cudaMalloc(&d_vk_input, N2 * sizeof(cuFloatComplex)));
		CUDA_CHECK(cudaMemcpy(d_vk_input, gendat[1].data(), N2 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

		VkFFTApplication app = {};
		VkFFTResult res = initializeVkFFT(&app, configs.second);
		if (res != VKFFT_SUCCESS) { 
			std::cerr << "VkFFT init failed\n"; return; 
		}
		static void* buffer_ptrs[1];
		buffer_ptrs[0] = d_vk_input;
		static void* kernel_ptrs[1];
		kernel_ptrs[0] = vk_kernel;
		VkFFTLaunchParams launchParams = {};
		launchParams.buffer = buffer_ptrs;
		launchParams.kernel = kernel_ptrs;
		if (precission) {
			res = VkFFTAppend(&app, 0, &launchParams);
			if (res != VKFFT_SUCCESS) { 
				std::cerr << "VkFFT run failed, code: " << res << "\n"; return; 
			}
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaMemcpy(h_vk_output.data(), d_vk_input, N2 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
		}
		if (speed) {
			CUDA_CHECK(cudaDeviceSynchronize());
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < iterations; ++i) {
				res = VkFFTAppend(&app, -1, &launchParams);
				if (res != VKFFT_SUCCESS) { 
					std::cerr << "VkFFT run failed, code: " << res << "\n"; return; 
				}
			}
			CUDA_CHECK(cudaDeviceSynchronize());
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed = end - start;
			std::cout << "VkFFT " << NX << "x" << NY << "x" << NZ << " average time over " << iterations << " iterations: " 
					  << (elapsed.count() * 1000.0 / iterations) << " ms\n";
		}

		deleteVkFFT(&app);
		CUDA_CHECK(cudaFree(vk_kernel));
		CUDA_CHECK(cudaFree(d_vk_input));
	}

	CUDA_CHECK(cudaFree(d_kernel));

	bool ok = compare_results(h_cu_output, h_vk_output, 1e-5f);

	print_n(h_cu_output, gendat[0], 10);
	print_n(h_vk_output, gendat[1], 10);
}

int main() {
	const int NX = 320, NY = 320, NZ = 320;
	std::cout << "Benchmarking 3D convolution: VkFFT vs cuFFT" << std::endl;

	benchmark_fft(NX, NY, NZ, true, true);

	return 0;
}
