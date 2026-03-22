module;

#include <pch.hpp>

#include <battery/embed.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <nvrtc.h>
#include <cuda.h>
#define VKFFT_BACKEND 1
#include "vkFFT.h"

export module fft:toeplitz;

import std;
import util_mod;
import tensor_mod;
import nvrtc;
import vkfft;

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

    // If the following enums are changed, 
    // also change the corresponding kernel code in 
    // lib/fft/kernels/toeplitz_load_2D.cu and lib/fft/kernels/toeplitz_load_3D.cu
    export enum class ToeplitzMultType {
        NONE=0,
        MULT=1,
        MULT_CONJ=2
    };

    export enum class ToeplitzAccumulateType {
        NONE=0,
        ACCUMULATE=1
    };

}
}


namespace hasty {
namespace fft {

void transform_toeplitz_kernel_1D(Tensor& kernel, bool clear_vkfft_plan);
void transform_toeplitz_kernel_2D(Tensor& kernel, bool clear_vkfft_plan);
void transform_toeplitz_kernel_3D(Tensor& kernel, bool clear_vkfft_plan);

void toeplitz_multiplication_1D(
    const Tensor& 				    input,
    Tensor& 				        output,
    const Tensor& 				    kernel,
    OptRefW<Tensor> 			    scratch,
    OptCRefW<Tensor> 			    mult1,
    OptCRefW<Tensor> 			    mult2,
    ToeplitzMultType 			    input_output_mult_type,
    ToeplitzMultType 			    input_mult1_type,
    ToeplitzMultType 			    output_mult1_type,
    ToeplitzMultType 			    input_mult2_type,
    ToeplitzMultType 			    output_mult2_type,
    ToeplitzAccumulateType          accumulate_type
);
void toeplitz_multiplication_2D(
    const Tensor& 				    input,
    Tensor& 				        output,
    const Tensor& 				    kernel,
    OptRefW<Tensor> 			    scratch,
    OptCRefW<Tensor> 			    mult1,
    OptCRefW<Tensor> 			    mult2,
    ToeplitzMultType 			    input_output_mult_type,
    ToeplitzMultType 			    input_mult1_type,
    ToeplitzMultType 			    output_mult1_type,
    ToeplitzMultType 			    input_mult2_type,
    ToeplitzMultType 			    output_mult2_type,
    ToeplitzAccumulateType          accumulate_type
);
void toeplitz_multiplication_3D(
    const Tensor& 				    input,
    Tensor& 				        output,
    const Tensor& 				    kernel,
    OptRefW<Tensor> 			    scratch,
    OptCRefW<Tensor> 			    mult1,
    OptCRefW<Tensor> 			    mult2,
    ToeplitzMultType 			    input_output_mult_type,
    ToeplitzMultType 			    input_mult1_type,
    ToeplitzMultType 			    output_mult1_type,
    ToeplitzMultType 			    input_mult2_type,
    ToeplitzMultType 			    output_mult2_type,
    ToeplitzAccumulateType          accumulate_type
);

}
}


namespace hasty {
namespace fft {

export void transform_toeplitz_kernel(Tensor& kernel, bool clear_vkfft_plan = false)
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

export void toeplitz_multiplication(
    const Tensor&               input,
    Tensor&                     output,
    const Tensor&               kernel,
    OptRefW<Tensor>             scratch,
    OptCRefW<Tensor>            mult1,
    OptCRefW<Tensor>            mult2,
    ToeplitzMultType            input_output_mult_type,
    ToeplitzMultType            input_mult1_type,
    ToeplitzMultType            output_mult1_type,
    ToeplitzMultType            input_mult2_type,
    ToeplitzMultType            output_mult2_type,
    ToeplitzAccumulateType      accumulate_type
)
{
    if (kernel.ndimension() == 1) {
        if (input.ndimension() != 2)
            throw std::runtime_error("Input must be 2D for 1D kernel");
        if (output.ndimension() != 2)
            throw std::runtime_error("Output must be 2D for 1D kernel");
        toeplitz_multiplication_1D(
            input,
            output,
            kernel,
            scratch,
            mult1,
            mult2,
            input_output_mult_type,
            input_mult1_type,
            output_mult1_type,
            input_mult2_type,
            output_mult2_type,
            accumulate_type
        );
    }
    else if (kernel.ndimension() == 2) {
        if (input.ndimension() != 3)
            throw std::runtime_error("Input must be 3D for 2D kernel");
        if (output.ndimension() != 3)
            throw std::runtime_error("Output must be 3D for 2D kernel");
        toeplitz_multiplication_2D(
            input,
            output,
            kernel,
            scratch,
            mult1,
            mult2,
            input_output_mult_type,
            input_mult1_type,
            output_mult1_type,
            input_mult2_type,
            output_mult2_type,
            accumulate_type
        );
    }
    else if (kernel.ndimension() == 3) {
        if (input.ndimension() != 4)
            throw std::runtime_error("Input must be 4D for 3D kernel");
        if (output.ndimension() != 4)
            throw std::runtime_error("Output must be 4D for 3D kernel");
        toeplitz_multiplication_3D(
            input,
            output,
            kernel,
            scratch,
            mult1,
            mult2,
            input_output_mult_type,
            input_mult1_type,
            output_mult1_type,
            input_mult2_type,
            output_mult2_type,
            accumulate_type
        );
    }
    else {
        throw std::runtime_error("Kernel must be 1D, 2D or 3D");
    }
}

}
}

// VkFFT calling and NVRTC kernel calling implementations
namespace hasty {
namespace fft {

void launch_toeplitz_load_1D(
    const cuFloatComplex* input,
    const cuFloatComplex* output,
    cuFloatComplex*       scratch,
    const cuFloatComplex* mult1,
    const cuFloatComplex* mult2,
    int input_output_mult_type,
    int input_mult1_type,
    int output_mult1_type,
    int input_mult2_type,
    int output_mult2_type,
    int batch_in,
    int batch_out,
    int NX,
    bool accumulate,
    int device_idx,
    int threads_per_block = 256)
{
    static std::array<hasty::nvrtc::NVRTC_ModFunc, (std::size_t)device_alias::MAX_CUDA_DEVICES> nvrtc_modules = {};
    static const char* toeplitz_load_1D_code = b::embed<"src/fft/kernels/toeplitz_load_1D.cu">().data();

    if (!nvrtc_modules[device_idx].module) {
        nvrtc::compile_nvrtc_kernel(
            nvrtc_modules[device_idx],
            toeplitz_load_1D_code,
            "toeplitz_load_1D",
            device_idx
        );
    }

    cudaSetDevice(device_idx);
    int blocks = (NX + threads_per_block - 1) / threads_per_block;

    void* args[] = {
        (void*)&input,
        (void*)&output,
        (void*)&scratch,
        (void*)&mult1,
        (void*)&mult2,
        (void*)&input_output_mult_type,
        (void*)&input_mult1_type,
        (void*)&output_mult1_type,
        (void*)&input_mult2_type,
        (void*)&output_mult2_type,
        (void*)&batch_in,
        (void*)&batch_out,
        (void*)&NX,
        (void*)&accumulate
    };

    cuLaunchKernel(nvrtc_modules[device_idx].function,
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0, args, 0);
}

void launch_toeplitz_load_2D(
    const cuFloatComplex* input, 
    const cuFloatComplex* output, 
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
    int NX, int NY,
    bool accumulate,
    int device_idx,
    int threads_per_block = 256)
{
    static std::array<hasty::nvrtc::NVRTC_ModFunc, (std::size_t)device_alias::MAX_CUDA_DEVICES> nvrtc_modules = {};

    static const char* toeplitz_load_2D_code = b::embed<"src/fft/kernels/toeplitz_load_2D.cu">().data();

    if (!nvrtc_modules[device_idx].module) {
        nvrtc::compile_nvrtc_kernel(
            nvrtc_modules[device_idx],
            toeplitz_load_2D_code,
            "toeplitz_load_2D",
            device_idx
        );
    }

    CUdevice cudevice;
    cuDeviceGet(&cudevice, device_idx);
    cudaSetDevice(device_idx);

    int totalThreads = NX * NY;
    int blocks = (totalThreads + threads_per_block - 1) / threads_per_block;

    void* args[] = { 
        (void*)&input, 
        (void*)&output, 
        (void*)&scratch, 
        (void*)&mult1, 
        (void*)&mult2, 
        (void*)&input_output_mult_type,
        (void*)&input_mult1_type, 
        (void*)&output_mult1_type,
        (void*)&input_mult2_type, 
        (void*)&output_mult2_type, 
        (void*)&batch_in,
        (void*)&batch_out, 
        (void*)&NX, (void*)&NY,
        (void*)&accumulate
    };

    cuLaunchKernel(
        nvrtc_modules[device_idx].function,
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        args, 0
    );

}

void launch_toeplitz_load_3D(
    const cuFloatComplex* input, 
    const cuFloatComplex* output, 
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
    bool accumulate,
    int device_idx,
    int threads_per_block = 256)
{
    static std::array<hasty::nvrtc::NVRTC_ModFunc, (std::size_t)device_alias::MAX_CUDA_DEVICES> nvrtc_modules = {};

    static const char* toeplitz_load_3D_code = b::embed<"src/fft/kernels/toeplitz_load_3D.cu">().data();

    if (!nvrtc_modules[device_idx].module) {
        nvrtc::compile_nvrtc_kernel(
            nvrtc_modules[device_idx],
            toeplitz_load_3D_code,
            "toeplitz_load_3D",
            device_idx
        );
    }
    
    CUdevice cudevice;
    cuDeviceGet(&cudevice, device_idx);
    cudaSetDevice(device_idx);

    int totalThreads = NX * NY * NZ;
    int blocks = (totalThreads + threads_per_block - 1) / threads_per_block;

    void* args[] = { 
        (void*)&input, 
        (void*)&output, 
        (void*)&scratch, 
        (void*)&mult1,
        (void*)&mult2,
        (void*)&input_output_mult_type,
        (void*)&input_mult1_type, 
        (void*)&output_mult1_type, 
        (void*)&input_mult2_type, 
        (void*)&output_mult2_type, 
        (void*)&batch_in,
        (void*)&batch_out,
        (void*)&NX, (void*)&NY, (void*)&NZ,
        (void*)&accumulate
    };

    cuLaunchKernel(
        nvrtc_modules[device_idx].function,
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        args, 0
    );

}

void perform_toeplitz_multiplication_cuda_1D(
    const Tensor&                   input,
    Tensor                          output,
    const Tensor&                   kernel,
    OptRefW<Tensor>                 scratch,
    OptCRefW<Tensor>                mult1,
    OptCRefW<Tensor>                mult2,
    int input_output_mult_type =    (int)ToeplitzMultType::NONE,
    int input_mult1_type       =    (int)ToeplitzMultType::MULT,
    int output_mult1_type      =    (int)ToeplitzMultType::MULT_CONJ,
    int input_mult2_type       =    (int)ToeplitzMultType::MULT,
    int output_mult2_type      =    (int)ToeplitzMultType::MULT_CONJ,
    int accumulate_type        =    (int)ToeplitzAccumulateType::NONE
)
{
    auto device    = input.device();
    int  nbatch    = input.size(0);
    int  NX        = input.size(1);
    torch_check(NX > 1, "input dimension must be positive");
    int  device_idx = device.index();
    bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

    //const cuFloatComplex* in_ptr     = reinterpret_cast<const cuFloatComplex*>(input.data_ptr<hc10::complex<float>>());
    //cuFloatComplex*       out_ptr    = reinterpret_cast<cuFloatComplex*>(output.data_ptr<hc10::complex<float>>());
    //const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.data_ptr<hc10::complex<float>>());



    cuFloatComplex* scratch_ptr;
    Tensor scratchmem;
    if (scratch.has_value()) {
        torch_check(scr.is_cuda(),                                  "scratch must be a CUDA tensor");
        torch_check(scr.scalar_type() == hat::kComplexFloat,        "scratch dtype must be complex float");
        torch_check(scr.sizes().equals({ 2 * NX }),                 "scratch must have shape (2*NX)");
        torch_check(scr.device() == device,                         "scratch must be on the same device");
        torch_check(scr.is_contiguous(),                            "scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.data_ptr<hc10::complex<float>>());
    } else {
        scratchmem  = hat::empty({ 2 * NX }, input.options());
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem.data_ptr<hc10::complex<float>>());
    }

    const cuFloatComplex* mult1_ptr = nullptr;
    if (mult1.has_value()) {
        const hat::Tensor& m1 = (*mult1).get();
        torch_check(m1.is_cuda() && m1.scalar_type() == hat::kComplexFloat, "mult1 type error");
        torch_check(m1.sizes().equals({ NX }), "mult1 must have shape (NX)");
        torch_check(m1.device() == device && m1.is_contiguous(), "mult1 device/contiguity error");
        mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.data_ptr<hc10::complex<float>>());
    }
    const cuFloatComplex* mult2_ptr = nullptr;
    if (mult2.has_value()) {
        const hat::Tensor& m2 = (*mult2).get();
        torch_check(m2.is_cuda() && m2.scalar_type() == hat::kComplexFloat, "mult2 type error");
        torch_check(m2.sizes().equals({ NX }), "mult2 must have shape (NX)");
        torch_check(m2.device() == device && m2.is_contiguous(), "mult2 device/contiguity error");
        mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.data_ptr<hc10::complex<float>>());
    }

    // Prime: load batch 0 into scratch (batch_out=-1 skips writing output)
    launch_toeplitz_load_1D(
        in_ptr, out_ptr, scratch_ptr, mult1_ptr, mult2_ptr,
        input_output_mult_type, input_mult1_type, output_mult1_type,
        input_mult2_type, output_mult2_type,
        0, -1, NX, accumulate, device_idx);

    VkFFT_Cache::VkFFT_Key key(device_idx);
    key.performConvolution     = true;
    key.size[0]                = 2 * NX;
    key.FFTdim                 = 1;
    key.performZeropadding[0]  = true;
    key.fft_zeropad_left[0]    = 0;
    key.fft_zeropad_right[0]   = NX;

    VkFFTApplication& app = global_vkfft_cache[device_idx].get_or_create(key);

    static void* buffer_ptrs[1];
    buffer_ptrs[0] = scratch_ptr;
    static void* kernel_ptrs[1];
    kernel_ptrs[0] = const_cast<cuFloatComplex*>(kernel_ptr);

    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = buffer_ptrs;
    launchParams.kernel = kernel_ptrs;

    for (int b = 0; b < nbatch; ++b) {
        VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
        if (res != VKFFT_SUCCESS)
            throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));

        launch_toeplitz_load_1D(
            in_ptr, out_ptr, scratch_ptr, mult1_ptr, mult2_ptr,
            input_output_mult_type, input_mult1_type, output_mult1_type,
            input_mult2_type, output_mult2_type,
            (b < (nbatch - 1)) ? (b + 1) : -1, b,
            NX, accumulate, device_idx);
    }
}

void perform_toeplitz_multiplication_cuda_2D(
    const Tensor& 				    input,
    Tensor 					        output,
    const Tensor& 				    kernel,
    OptRefW<Tensor> 			    scratch,
    OptCRefW<Tensor> 			    mult1,
    OptCRefW<Tensor> 			    mult2,
    int input_output_mult_type  = 	(int)ToeplitzMultType::NONE,
    int input_mult1_type        = 	(int)ToeplitzMultType::MULT,
    int output_mult1_type       = 	(int)ToeplitzMultType::MULT_CONJ,
    int input_mult2_type        = 	(int)ToeplitzMultType::MULT,
    int output_mult2_type       = 	(int)ToeplitzMultType::MULT_CONJ,
    int accumulate_type         = 	(int)ToeplitzAccumulateType::NONE
)
{
    auto device = input.device();
    int nbatch = input.size(0);
    int dim = input.dim();
    int NX = input.size(dim - 1);
    int NY = input.size(dim - 2);
    torch_check(NY > 1 && NX > 1, "input dimensions must be positive");
    int device_idx = device.index();
    bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

    const cuFloatComplex* in_ptr = reinterpret_cast<const cuFloatComplex*>(input.data_ptr<hc10::complex<float>>());
    cuFloatComplex* out_ptr = reinterpret_cast<cuFloatComplex*>(output.data_ptr<hc10::complex<float>>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.data_ptr<hc10::complex<float>>());

    cuFloatComplex* scratch_ptr;
    hat::Tensor scratchmem;
    if (scratch.has_value()) {
        const hat::Tensor& scr = (*scratch).get();
        torch_check(scr.is_cuda(), "scratch was not a CUDA tensor");
        torch_check(scr.scalar_type() == hat::kComplexFloat, "scratch dtype must be complex float");
        torch_check(scr.sizes().equals({ 2 * NY, 2 * NX }), "scratch must have shape (2*NY, 2*NX)");
        torch_check(scr.device() == device, "scratch must be on the same device as input and output");
        torch_check(scr.is_contiguous(), "scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.data_ptr<hc10::complex<float>>());
    } else {
        scratchmem = hat::empty({ 2 * NY, 2 * NX }, input.options());
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem.data_ptr<hc10::complex<float>>());
    }

    const cuFloatComplex* mult1_ptr = nullptr;
    if (mult1.has_value()) {
        const hat::Tensor& m1 = (*mult1).get();
        torch_check(m1.is_cuda(), "mult1 was not a CUDA tensor");
        torch_check(m1.scalar_type() == hat::kComplexFloat, "mult1 dtype must be complex float");
        torch_check(m1.sizes().equals({ NY, NX }), "mult1 must have shape (NY, NX)");
        torch_check(m1.device() == device, "mult1 must be on the same device as input and output");
        torch_check(m1.is_contiguous(), "mult1 must be contiguous");
        mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.data_ptr<hc10::complex<float>>());
    }
    const cuFloatComplex* mult2_ptr = nullptr;
    if (mult2.has_value()) {
        const hat::Tensor& m2 = (*mult2).get();
        torch_check(m2.is_cuda(), "mult2 was not a CUDA tensor");
        torch_check(m2.scalar_type() == hat::kComplexFloat, "mult2 dtype must be complex float");
        torch_check(m2.sizes().equals({ NY, NX }), "mult2 must have shape (NY, NX)");
        torch_check(m2.device() == device, "mult2 must be on the same device as input and output");
        torch_check(m2.is_contiguous(), "mult2 must be contiguous");
        mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.data_ptr<hc10::complex<float>>());
    }

    launch_toeplitz_load_2D(
        in_ptr,
        out_ptr,
        scratch_ptr,
        mult1_ptr,
        mult2_ptr,
        input_output_mult_type,
        input_mult1_type,
        output_mult1_type,
        input_mult2_type,
        output_mult2_type,
        0,
        -1,
        NX, NY,
        accumulate,
        device_idx
    );

    // Create VkFFT plans and execute FFTs here
    VkFFT_Cache::VkFFT_Key key(device_idx);
    key.performConvolution = true;
    key.size[0] = 2 * NX;
    key.size[1] = 2 * NY;
    key.FFTdim = 2;
    key.performZeropadding[0] = true;
    key.performZeropadding[1] = true;
    key.fft_zeropad_left[0] = 0;
    key.fft_zeropad_left[1] = 0;
    key.fft_zeropad_right[0] = NX;
    key.fft_zeropad_right[1] = NY;

    VkFFTApplication& app = global_vkfft_cache[device_idx].get_or_create(key);

    static void* buffer_ptrs[1];
    buffer_ptrs[0] = scratch_ptr;
    static void* kernel_ptrs[1];
    kernel_ptrs[0] = const_cast<cuFloatComplex*>(kernel_ptr);

    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = buffer_ptrs;
    launchParams.kernel = kernel_ptrs;

    for (int b = 0; b < nbatch; ++b) {
        VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
        if (res != VKFFT_SUCCESS) {
            throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
        }

        // For other accumulate types, adjust outputbatch as needed
        launch_toeplitz_load_2D(
            in_ptr,
            out_ptr,
            scratch_ptr,
            mult1_ptr,
            mult2_ptr,
            input_output_mult_type,
            input_mult1_type,
            output_mult1_type,
            input_mult2_type,
            output_mult2_type,
            (b < (nbatch - 1)) ? (b + 1) : -1,
            b,
            NX, NY,
            accumulate,
            device_idx
        );
    }
}

void perform_toeplitz_multiplication_cuda_3D(
    const Tensor& 				    input,
    Tensor 					        output,
    const Tensor& 				    kernel,
    OptRefW<Tensor> 			    scratch,
    OptCRefW<Tensor> 			    mult1,
    OptCRefW<Tensor> 			    mult2,
    int input_output_mult_type  = 	(int)ToeplitzMultType::NONE,
    int input_mult1_type        = 	(int)ToeplitzMultType::MULT,
    int output_mult1_type       = 	(int)ToeplitzMultType::MULT_CONJ,
    int input_mult2_type        = 	(int)ToeplitzMultType::MULT,
    int output_mult2_type       = 	(int)ToeplitzMultType::MULT_CONJ,
    int accumulate_type         = 	(int)ToeplitzAccumulateType::NONE
)
{
    auto device = input.device();
    int nbatch = input.size(0);
    int dim = input.dim();
    int NX = input.size(dim - 1);
    int NY = input.size(dim - 2);
    int NZ = input.size(dim - 3);
    torch_check(NZ > 1 && NY > 1 && NX > 1, "input dimensions must be positive");
    int device_idx = device.index();
    bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

    const cuFloatComplex* in_ptr = reinterpret_cast<const cuFloatComplex*>(input.data_ptr<hc10::complex<float>>());
    cuFloatComplex* out_ptr = reinterpret_cast<cuFloatComplex*>(output.data_ptr<hc10::complex<float>>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.data_ptr<hc10::complex<float>>());

    cuFloatComplex* scratch_ptr;
    hat::Tensor scratchmem;
    if (scratch.has_value()) {
        const hat::Tensor& scr = (*scratch).get();
        torch_check(scr.is_cuda(), "scratch was not a CUDA tensor");
        torch_check(scr.scalar_type() == hat::kComplexFloat, "scratch dtype must be complex float");
        torch_check(scr.sizes().equals({ 2 * NZ, 2 * NY, 2 * NX }), "scratch must have shape (2*NZ, 2*NY, 2*NX)");
        torch_check(scr.device() == device, "scratch must be on the same device as input and output");
        torch_check(scr.is_contiguous(), "scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.data_ptr<hc10::complex<float>>());
    } else {
        scratchmem = hat::empty({ 2 * NZ, 2 * NY, 2 * NX }, input.options());
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem.data_ptr<hc10::complex<float>>());
    }

    const cuFloatComplex* mult1_ptr = nullptr;
    if (mult1.has_value()) {
        const hat::Tensor& m1 = (*mult1).get();
        torch_check(m1.is_cuda(), "mult1 was not a CUDA tensor");
        torch_check(m1.scalar_type() == hat::kComplexFloat, "mult1 dtype must be complex float");
        torch_check(m1.sizes().equals({ NZ, NY, NX }), "mult1 must have shape (NZ, NY, NX)");
        torch_check(m1.device() == device, "mult1 must be on the same device as input and output");
        torch_check(m1.is_contiguous(), "mult1 must be contiguous");
        mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.data_ptr<hc10::complex<float>>());
    }
    const cuFloatComplex* mult2_ptr = nullptr;
    if (mult2.has_value()) {
        const hat::Tensor& m2 = (*mult2).get();
        torch_check(m2.is_cuda(), "mult2 was not a CUDA tensor");
        torch_check(m2.scalar_type() == hat::kComplexFloat, "mult2 dtype must be complex float");
        torch_check(m2.sizes().equals({ NZ, NY, NX }), "mult2 must have shape (NZ, NY, NX)");
        torch_check(m2.device() == device, "mult2 must be on the same device as input and output");
        torch_check(m2.is_contiguous(), "mult2 must be contiguous");
        mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.data_ptr<hc10::complex<float>>());
    }

    launch_toeplitz_load_3D(
        in_ptr,
        out_ptr,
        scratch_ptr,
        mult1_ptr,
        mult2_ptr,
        input_output_mult_type,
        input_mult1_type,
        output_mult1_type,
        input_mult2_type,
        output_mult2_type,
        0,
        -1,
        NX, NY, NZ,
        accumulate,
        device_idx
    );

    // Create VkFFT plans and execute FFTs here
    VkFFT_Cache::VkFFT_Key key(device_idx);
    key.performConvolution = true;
    key.size[0] = 2 * NX;
    key.size[1] = 2 * NY;
    key.size[2] = 2 * NZ;
    key.FFTdim = 3;
    key.performZeropadding[0] = true;
    key.performZeropadding[1] = true;
    key.performZeropadding[2] = true;
    key.fft_zeropad_left[0] = 0;
    key.fft_zeropad_left[1] = 0;
    key.fft_zeropad_left[2] = 0;
    key.fft_zeropad_right[0] = NX;
    key.fft_zeropad_right[1] = NY;
    key.fft_zeropad_right[2] = NZ;

    VkFFTApplication& app = global_vkfft_cache[device_idx].get_or_create(key);
    
    static void* buffer_ptrs[1];
    buffer_ptrs[0] = scratch_ptr;
    static void* kernel_ptrs[1];
    kernel_ptrs[0] = const_cast<cuFloatComplex*>(kernel_ptr);

    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = buffer_ptrs;
    launchParams.kernel = kernel_ptrs;

    for (int b = 0; b < nbatch; ++b) {
        VkFFTResult res = VkFFTAppend(&app, 0, &launchParams);
        if (res != VKFFT_SUCCESS) {
            throw std::runtime_error("VkFFT run failed, code: " + std::to_string(res));
        }

        // For other accumulate types, adjust outputbatch as needed
        launch_toeplitz_load_3D(
            in_ptr,
            out_ptr,
            scratch_ptr,
            mult1_ptr,
            mult2_ptr,
            input_output_mult_type,
            input_mult1_type,
            output_mult1_type,
            input_mult2_type,
            output_mult2_type,
            (b < (nbatch - 1)) ? b+1 : -1,
            b,
            NX, NY, NZ,
            accumulate,
            device_idx
        );
    }
}

}
}


// TOEPLITZ KERNEL TRANSFORMATIONS
namespace hasty {
namespace fft {

void transform_toeplitz_kernel_1D(Tensor& ker, bool clear_vkfft_plan)
{
    int NX = ker.size(0);
    torch_check(NX > 1, "kernel dimension must be positive");
    auto device = ker.device();

    VkFFT_Cache::VkFFT_Key key(device.index());
    key.size[0]          = NX;
    key.FFTdim           = 1;
    key.kernelConvolution = 1;

    {
        hat::cuda::CUDAGuard device_guard(ker.device());
        cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.data_ptr<hc10::complex<float>>());

        {
            cufftHandle plan;
            CUFFT_CHECK(cufftPlan1d(&plan, NX, CUFFT_C2C, /*batch=*/1));
            cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUFFT_CHECK(cufftDestroy(plan));
        }

        hat::Tensor scratch = hat::empty_like(ker);
        cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.data_ptr<hc10::complex<float>>());

        VkFFTApplication& app = global_vkfft_cache[device.index()].get_or_create(key);
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
    }

    if (clear_vkfft_plan)
        global_vkfft_cache[device.index()].erase(key);
}

void transform_toeplitz_kernel_2D(Tensor& ker, bool clear_vkfft_plan) 
{   
    int NX = ker.size(1);
    int NY = ker.size(0);
    torch_check(NY > 1 && NX > 1, "kernel dimensions must be positive");
    auto device = ker.device();

    VkFFT_Cache::VkFFT_Key key(device.index());
    key.size[0] = NX;
    key.size[1] = NY;
    key.FFTdim = 2;
    key.kernelConvolution = 1;
    
    {
        hat::cuda::CUDAGuard device_guard(ker.device());
        cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.data_ptr<hc10::complex<float>>());
        {
            cufftHandle plan;
            CUFFT_CHECK(cufftPlan2d(&plan, NY, NX, CUFFT_C2C));
            cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUFFT_CHECK(cufftDestroy(plan));
        }

        hat::Tensor scratch = hat::empty_like(ker);
        cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.data_ptr<hc10::complex<float>>());

        VkFFTApplication& app = global_vkfft_cache[device.index()].get_or_create(key);
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
    }

    if (clear_vkfft_plan) {
        global_vkfft_cache[device.index()].erase(key);
    }

}

void transform_toeplitz_kernel_3D(Tensor& ker, bool clear_vkfft_plan) 
{
    int NX = ker.size(2);
    int NY = ker.size(1);
    int NZ = ker.size(0);
    torch_check(NZ > 1 && NY > 1 && NX > 1, "kernel dimensions must be positive");
    auto device = ker.device();

    VkFFT_Cache::VkFFT_Key key(device.index());
    key.size[0] = NX;
    key.size[1] = NY;
    key.size[2] = NZ;
    key.FFTdim = 3;
    key.kernelConvolution = 1;
    
    {
        hat::cuda::CUDAGuard device_guard(ker.device());
        cuFloatComplex* ker_ptr = reinterpret_cast<cuFloatComplex*>(ker.data_ptr<hc10::complex<float>>());
        {
            cufftHandle plan;
            CUFFT_CHECK(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C));
            cufftExecC2C(plan, (cufftComplex*)ker_ptr, (cufftComplex*)ker_ptr, CUFFT_INVERSE);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUFFT_CHECK(cufftDestroy(plan));
        }

        hat::Tensor scratch = hat::empty_like(ker);
        cuFloatComplex* scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratch.data_ptr<hc10::complex<float>>());

        VkFFTApplication& app = global_vkfft_cache[device.index()].get_or_create(key);
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
    }

    if (clear_vkfft_plan) {
        global_vkfft_cache[device.index()].erase(key);
    }
}

}
}


// MULTIPLICATION IMPLEMENTATION
namespace hasty {
namespace fft {

void toeplitz_multiplication_1D(
    const Tensor&      inp,
    Tensor&            out,
    const Tensor&      ker,
    OptRefW<Tensor>    scr,
    OptCRefW<Tensor>   m1,
    OptCRefW<Tensor>   m2,
    ToeplitzMultType        input_output_mult_type,
    ToeplitzMultType        input_mult1_type,
    ToeplitzMultType        output_mult1_type,
    ToeplitzMultType        input_mult2_type,
    ToeplitzMultType        output_mult2_type,
    ToeplitzAccumulateType  accumulate_type
)
{
    torch_check(inp.scalar_type() == hat::kComplexFloat, "input dtype must be complex float");
    torch_check(out.scalar_type() == hat::kComplexFloat, "output dtype must be complex float");
    torch_check(ker.scalar_type() == hat::kComplexFloat, "kernel dtype must be complex float");

    int NX = inp.size(1);
    auto device = inp.device();
    torch_check(inp.sizes().equals(out.sizes()),  "input and output must have the same size");
    torch_check(device == out.device(),           "input and output must be on the same device");
    torch_check(device == ker.device(),           "input and kernel must be on the same device");
    torch_check(ker.sizes().equals({ 2 * NX }),   "kernel must have shape (2*NX)");

    perform_toeplitz_multiplication_cuda_1D(
        inp, out, ker,
        std::move(scr), std::move(m1), std::move(m2),
        (int)input_output_mult_type,
        (int)input_mult1_type,
        (int)output_mult1_type,
        (int)input_mult2_type,
        (int)output_mult2_type,
        (int)accumulate_type
    );
}

void toeplitz_multiplication_2D(
    const Tensor&                  inp,
    Tensor&                        out,
    const Tensor&                  ker,
    OptRefW<Tensor>                scr,
    OptCRefW<Tensor>               m1,
    OptCRefW<Tensor>               m2,
    ToeplitzMultType                    input_output_mult_type,
    ToeplitzMultType                    input_mult1_type,
    ToeplitzMultType                    output_mult1_type,
    ToeplitzMultType                    input_mult2_type,
    ToeplitzMultType                    output_mult2_type,
    ToeplitzAccumulateType              accumulate_type
)
{
    torch_check(inp.scalar_type() == hat::kComplexFloat, "input dtype must be complex float");
    torch_check(out.scalar_type() == hat::kComplexFloat, "output dtype must be complex float");
    torch_check(ker.scalar_type() == hat::kComplexFloat, "kernel dtype must be complex float");

    int NX = inp.size(2);
    int NY = inp.size(1);
    auto device = inp.device();
    torch_check(inp.sizes().equals(out.sizes()), "input and output must have the same size");
    torch_check(device == out.device(), "input and output must be on the same device");
    torch_check(device == ker.device(), "input and kernel must be on the same device");
    torch_check(ker.sizes().equals({ 2 * NY, 2 * NX }), "scratch must have shape (2*NY, 2*NX)");

    perform_toeplitz_multiplication_cuda_2D(
        inp,
        out,
        ker,
        std::move(scr),
        std::move(m1),
        std::move(m2),
        (int)input_output_mult_type,
        (int)input_mult1_type,
        (int)output_mult1_type,
        (int)input_mult2_type,
        (int)output_mult2_type,
        (int)accumulate_type
    );
}

void toeplitz_multiplication_3D(
    const Tensor&                  inp,
    Tensor&                        out,
    const Tensor&                  ker,
    OptRefW<Tensor>                scr,
    OptCRefW<Tensor>               m1,
    OptCRefW<Tensor>               m2,
    ToeplitzMultType                    input_output_mult_type,
    ToeplitzMultType                    input_mult1_type,
    ToeplitzMultType                    output_mult1_type,
    ToeplitzMultType                    input_mult2_type,
    ToeplitzMultType                    output_mult2_type,
    ToeplitzAccumulateType              accumulate_type
) 
{
    torch_check(inp.scalar_type() == hat::kComplexFloat, "input dtype must be complex float");
    torch_check(out.scalar_type() == hat::kComplexFloat, "output dtype must be complex float");
    torch_check(ker.scalar_type() == hat::kComplexFloat, "kernel dtype must be complex float");

    int NX = inp.size(3);
    int NY = inp.size(2);
    int NZ = inp.size(1);
    auto device = inp.device();
    torch_check(inp.sizes().equals(out.sizes()), "input and output must have the same size");
    torch_check(device == out.device(), "input and output must be on the same device");
    torch_check(device == ker.device(), "input and kernel must be on the same device");
    torch_check(ker.sizes().equals({ 2 * NZ, 2 * NY, 2 * NX }), "scratch must have shape (2*NZ, 2*NY, 2*NX)");

    perform_toeplitz_multiplication_cuda_3D(
        inp,
        out,
        ker,
        std::move(scr),
        std::move(m1),
        std::move(m2),
        (int)input_output_mult_type,
        (int)input_mult1_type,
        (int)output_mult1_type,
        (int)input_mult2_type,
        (int)output_mult2_type,
        (int)accumulate_type
    );
}

}
}


