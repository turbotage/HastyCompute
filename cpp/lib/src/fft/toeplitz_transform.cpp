module;

#include <battery/embed.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <nvrtc.h>
#include <cuda.h>
#define VKFFT_BACKEND 1
#include <vkFFT.h>

module hasty_fft_mod;

import hasty_nvrtc_mod;
import hasty_vkfft_mod;

inline void CUDA_CHECK(cudaError_t err) {
	if (err != cudaSuccess) {
		throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)) + "\n");
	}
}

inline void CUFFT_CHECK(cufftResult err) {
	if (err != CUFFT_SUCCESS) {
		throw std::runtime_error("cuFFT error: " + std::to_string(err) + "\n");
	}
}

inline void CUDA_PRINT_LAST_ERROR() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA last error: " << cudaGetErrorString(err) << "\n";
	}
}



namespace hasty {
namespace fft {

void transform_toeplitz_kernel_1D(Tensor& ker, bool clear_vkfft_plan)
{
	int NX = ker.size(0);
	if (NX <= 1) throw std::runtime_error("kernel dimension must be positive");
	auto device = ker.device();

	VkFFT_Cache::VkFFT_Key key(static_cast<int>(device.index));
	key.size[0]          = NX;
	key.FFTdim           = 1;
	key.kernelConvolution = 1;

	{
		cuda::CUDAGuard device_guard(device);
		cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.mutable_data_ptr<c64>());

		{
			cufftHandle plan;
			CUFFT_CHECK(cufftPlan1d(&plan, NX, CUFFT_C2C, /*batch=*/1));
			cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
			CUDA_CHECK(cudaDeviceSynchronize());
			CUFFT_CHECK(cufftDestroy(plan));
		}

		Tensor scratch = hasty::empty_like(ker);
		cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.mutable_data_ptr<c64>());

		VkFFTApplication& app = global_vkfft_cache[static_cast<int>(device.index)].get_or_create(key);
		VkFFTLaunchParams launchParams = {};
		static void* buffer_ptrs[1];
		buffer_ptrs[0] = scratch_ptr;
		static void* kernel_ptrs[1];
		kernel_ptrs[0] = ker_ptr;
		launchParams.buffer = buffer_ptrs;
		launchParams.kernel = kernel_ptrs;
		VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
		if (res != VKFFT_SUCCESS)
			throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
		CUDA_CHECK(cudaDeviceSynchronize());
		ker = std::move(scratch);
	}

	if (clear_vkfft_plan)
		global_vkfft_cache[static_cast<int>(device.index)].erase(key);
}

void transform_toeplitz_kernel_2D(Tensor& ker, bool clear_vkfft_plan) 
{   
	int NX = ker.size(1);
	int NY = ker.size(0);
	if (NY <= 1 || NX <= 1) throw std::runtime_error("kernel dimensions must be positive");
	auto device = ker.device();

	VkFFT_Cache::VkFFT_Key key(static_cast<int>(device.index));
	key.size[0] = NX;
	key.size[1] = NY;
	key.FFTdim = 2;
	key.kernelConvolution = 1;

	{
		cuda::CUDAGuard device_guard(device);
		cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.mutable_data_ptr<c64>());
		{
			cufftHandle plan;
			CUFFT_CHECK(cufftPlan2d(&plan, NY, NX, CUFFT_C2C));
			cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
			CUDA_CHECK(cudaDeviceSynchronize());
			CUFFT_CHECK(cufftDestroy(plan));
		}

		Tensor scratch = hasty::empty_like(ker);
		cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.mutable_data_ptr<c64>());

		VkFFTApplication& app = global_vkfft_cache[static_cast<int>(device.index)].get_or_create(key);
		VkFFTLaunchParams launchParams = {};
		static void* buffer_ptrs[1];
		buffer_ptrs[0] = scratch_ptr;
		static void* kernel_ptrs[1];
		kernel_ptrs[0] = ker_ptr;
		launchParams.buffer = buffer_ptrs;
		launchParams.kernel = kernel_ptrs;
		VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
		if (res != VKFFT_SUCCESS) {
			throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
		}
		CUDA_CHECK(cudaDeviceSynchronize());

		ker = std::move(scratch);
	}

	if (clear_vkfft_plan) {
		global_vkfft_cache[static_cast<int>(device.index)].erase(key);
	}

}

void transform_toeplitz_kernel_3D(Tensor& ker, bool clear_vkfft_plan) 
{
	int NX = ker.size(2);
	int NY = ker.size(1);
	int NZ = ker.size(0);
	if (NZ <= 1 || NY <= 1 || NX <= 1) throw std::runtime_error("kernel dimensions must be positive");
	auto device = ker.device();

	VkFFT_Cache::VkFFT_Key key(static_cast<int>(device.index));
	key.size[0] = NX;
	key.size[1] = NY;
	key.size[2] = NZ;
	key.FFTdim = 3;
	key.kernelConvolution = 1;

	{
		cuda::CUDAGuard device_guard(device);
		cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.mutable_data_ptr<c64>());
		{
			cufftHandle plan;
			CUFFT_CHECK(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C));
			cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
			CUDA_CHECK(cudaDeviceSynchronize());
			CUFFT_CHECK(cufftDestroy(plan));
		}

		Tensor scratch = hasty::empty_like(ker);
		cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.mutable_data_ptr<c64>());

		VkFFTApplication& app = global_vkfft_cache[static_cast<int>(device.index)].get_or_create(key);
		VkFFTLaunchParams launchParams = {};
		static void* buffer_ptrs[1];
		buffer_ptrs[0] = scratch_ptr;
		static void* kernel_ptrs[1];
		kernel_ptrs[0] = ker_ptr;
		launchParams.buffer = buffer_ptrs;
		launchParams.kernel = kernel_ptrs;
		VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
		if (res != VKFFT_SUCCESS) {
			throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		ker = std::move(scratch);
	}

	if (clear_vkfft_plan) {
		global_vkfft_cache[static_cast<int>(device.index)].erase(key);
	}
}




void transform_toeplitz_kernel(Tensor& kernel, bool clear_vkfft_plan)
{
	if (kernel.device().type != eDeviceType::CUDA) {
		throw std::runtime_error("Kernel tensor must be on CUDA");
	}
	if (kernel.scalar_type() != eScalarType::ComplexFloat) {
		throw std::runtime_error("Kernel tensor must be of type complex float");
	}
	if (kernel.ndimension() == 1) {
		transform_toeplitz_kernel_1D(kernel, clear_vkfft_plan);
	} else if (kernel.ndimension() == 2) {
		transform_toeplitz_kernel_2D(kernel, clear_vkfft_plan);
	} else if (kernel.ndimension() == 3) {
		transform_toeplitz_kernel_3D(kernel, clear_vkfft_plan);
	} else {
		throw std::runtime_error("Kernel tensor must be 1D, 2D or 3D");
	}
}

	
}
}