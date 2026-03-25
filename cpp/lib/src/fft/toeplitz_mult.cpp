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
    if (NX <= 1) throw std::runtime_error("input dimension must be positive");
    int  device_idx = static_cast<int>(device.index);
    bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

    const cuFloatComplex* in_ptr     = reinterpret_cast<const cuFloatComplex*>(input.const_data_ptr<c64>());
    cuFloatComplex*       out_ptr    = reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.const_data_ptr<c64>());

    cuFloatComplex* scratch_ptr;
    Opt<Tensor> scratchmem;
    if (scratch.has_value()) {
        Tensor& scr = (*scratch).get();
        if (scr.device().type != eDeviceType::CUDA) throw std::runtime_error("scratch must be a CUDA tensor");
        if (scr.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("scratch dtype must be complex float");
        if (scr.ndimension() != 1 || scr.size(0) != 2 * NX) throw std::runtime_error("scratch must have shape (2*NX)");
        if (scr.device().type != device.type || scr.device().index != device.index) throw std::runtime_error("scratch must be on the same device");
        if (!scr.is_contiguous()) throw std::runtime_error("scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.mutable_data_ptr<c64>());
    } else {
        scratchmem  = hasty::empty({ 2 * NX }, TensorOptions(input.device()).dtype(input.scalar_type()));
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem->mutable_data_ptr<c64>());
    }

    const cuFloatComplex* mult1_ptr = nullptr;
    if (mult1.has_value()) {
        const Tensor& m1 = (*mult1).get();
        if (m1.device().type != eDeviceType::CUDA || m1.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("mult1 type error");
        if (m1.ndimension() != 1 || m1.size(0) != NX) throw std::runtime_error("mult1 must have shape (NX)");
        if (m1.device().type != device.type || m1.device().index != device.index || !m1.is_contiguous()) throw std::runtime_error("mult1 device/contiguity error");
        mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.const_data_ptr<c64>());
    }
    const cuFloatComplex* mult2_ptr = nullptr;
    if (mult2.has_value()) {
        const Tensor& m2 = (*mult2).get();
        if (m2.device().type != eDeviceType::CUDA || m2.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("mult2 type error");
        if (m2.ndimension() != 1 || m2.size(0) != NX) throw std::runtime_error("mult2 must have shape (NX)");
        if (m2.device().type != device.type || m2.device().index != device.index || !m2.is_contiguous()) throw std::runtime_error("mult2 device/contiguity error");
        mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.const_data_ptr<c64>());
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
    int dim = input.ndimension();
    int NX = input.size(dim - 1);
    int NY = input.size(dim - 2);
    if (NY <= 1 || NX <= 1) throw std::runtime_error("input dimensions must be positive");
    int device_idx = static_cast<int>(device.index);
    bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

    const cuFloatComplex* in_ptr = reinterpret_cast<const cuFloatComplex*>(input.const_data_ptr<c64>());
    cuFloatComplex* out_ptr = reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.const_data_ptr<c64>());

    cuFloatComplex* scratch_ptr;
    Opt<Tensor> scratchmem;
    if (scratch.has_value()) {
        Tensor& scr = (*scratch).get();
        if (scr.device().type != eDeviceType::CUDA) throw std::runtime_error("scratch was not a CUDA tensor");
        if (scr.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("scratch dtype must be complex float");
        if (!scr.sizes().equals({ 2 * NY, 2 * NX })) throw std::runtime_error("scratch must have shape (2*NY, 2*NX)");
        if (scr.device().type != device.type || scr.device().index != device.index) throw std::runtime_error("scratch must be on the same device as input and output");
        if (!scr.is_contiguous()) throw std::runtime_error("scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.mutable_data_ptr<c64>());
    } else {
        scratchmem = hasty::empty({ 2 * NY, 2 * NX }, TensorOptions(input.device()).dtype(input.scalar_type()));
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem->mutable_data_ptr<c64>());
    }

    const cuFloatComplex* mult1_ptr = nullptr;
    if (mult1.has_value()) {
        const Tensor& m1 = (*mult1).get();
        if (m1.device().type != eDeviceType::CUDA) throw std::runtime_error("mult1 was not a CUDA tensor");
        if (m1.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("mult1 dtype must be complex float");
        if (!m1.sizes().equals({ NY, NX })) throw std::runtime_error("mult1 must have shape (NY, NX)");
        if (m1.device().type != device.type || m1.device().index != device.index) throw std::runtime_error("mult1 must be on the same device as input and output");
        if (!m1.is_contiguous()) throw std::runtime_error("mult1 must be contiguous");
        mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.const_data_ptr<c64>());
    }
    const cuFloatComplex* mult2_ptr = nullptr;
    if (mult2.has_value()) {
        const Tensor& m2 = (*mult2).get();
        if (m2.device().type != eDeviceType::CUDA) throw std::runtime_error("mult2 was not a CUDA tensor");
        if (m2.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("mult2 dtype must be complex float");
        if (!m2.sizes().equals({ NY, NX })) throw std::runtime_error("mult2 must have shape (NY, NX)");
        if (m2.device().type != device.type || m2.device().index != device.index) throw std::runtime_error("mult2 must be on the same device as input and output");
        if (!m2.is_contiguous()) throw std::runtime_error("mult2 must be contiguous");
        mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.const_data_ptr<c64>());
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
    int dim = input.ndimension();
    int NX = input.size(dim - 1);
    int NY = input.size(dim - 2);
    int NZ = input.size(dim - 3);
    if (NZ <= 1 || NY <= 1 || NX <= 1) throw std::runtime_error("input dimensions must be positive");
    int device_idx = static_cast<int>(device.index);
    bool accumulate = (accumulate_type != (int)ToeplitzAccumulateType::NONE);

    const cuFloatComplex* in_ptr = reinterpret_cast<const cuFloatComplex*>(input.const_data_ptr<c64>());
    cuFloatComplex* out_ptr = reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.const_data_ptr<c64>());

    cuFloatComplex* scratch_ptr;
    Opt<Tensor> scratchmem;
    if (scratch.has_value()) {
        Tensor& scr = (*scratch).get();
        if (scr.device().type != eDeviceType::CUDA) throw std::runtime_error("scratch was not a CUDA tensor");
        if (scr.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("scratch dtype must be complex float");
        if (!scr.sizes().equals({ 2*NZ, 2*NY, 2*NX })) throw std::runtime_error("scratch must have shape (2*NZ, 2*NY, 2*NX)");
        if (scr.device().type != device.type || scr.device().index != device.index) throw std::runtime_error("scratch must be on the same device as input and output");
        if (!scr.is_contiguous()) throw std::runtime_error("scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.mutable_data_ptr<c64>());
    } else {
        scratchmem = hasty::empty({ 2 * NZ, 2 * NY, 2 * NX }, TensorOptions(input.device()).dtype(input.scalar_type()));
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem->mutable_data_ptr<c64>());
    }

    const cuFloatComplex* mult1_ptr = nullptr;
    if (mult1.has_value()) {
        const Tensor& m1 = (*mult1).get();
        if (m1.device().type != eDeviceType::CUDA) throw std::runtime_error("mult1 was not a CUDA tensor");
        if (m1.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("mult1 dtype must be complex float");
        if (!m1.sizes().equals({ NZ, NY, NX })) throw std::runtime_error("mult1 must have shape (NZ, NY, NX)");
        if (m1.device().type != device.type || m1.device().index != device.index) throw std::runtime_error("mult1 must be on the same device as input and output");
        if (!m1.is_contiguous()) throw std::runtime_error("mult1 must be contiguous");
        mult1_ptr = reinterpret_cast<const cuFloatComplex*>(m1.const_data_ptr<c64>());
    }
    const cuFloatComplex* mult2_ptr = nullptr;
    if (mult2.has_value()) {
        const Tensor& m2 = (*mult2).get();
        if (m2.device().type != eDeviceType::CUDA) throw std::runtime_error("mult2 was not a CUDA tensor");
        if (m2.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("mult2 dtype must be complex float");
        if (!m2.sizes().equals({ NZ, NY, NX })) throw std::runtime_error("mult2 must have shape (NZ, NY, NX)");
        if (m2.device().type != device.type || m2.device().index != device.index) throw std::runtime_error("mult2 must be on the same device as input and output");
        if (!m2.is_contiguous()) throw std::runtime_error("mult2 must be contiguous");
        mult2_ptr = reinterpret_cast<const cuFloatComplex*>(m2.const_data_ptr<c64>());
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
    if (inp.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("input dtype must be complex float");
    if (out.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("output dtype must be complex float");
    if (ker.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("kernel dtype must be complex float");

    int NX = inp.size(1);
    auto device = inp.device();
    if (!inp.sizes().equals(out.sizes())) throw std::runtime_error("input and output must have the same size");
    if (device.type != out.device().type || device.index != out.device().index) throw std::runtime_error("input and output must be on the same device");
    if (device.type != ker.device().type || device.index != ker.device().index) throw std::runtime_error("input and kernel must be on the same device");
    if (!ker.sizes().equals({ 2 * NX })) throw std::runtime_error("kernel must have shape (2*NX)");

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
    if (inp.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("input dtype must be complex float");
    if (out.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("output dtype must be complex float");
    if (ker.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("kernel dtype must be complex float");

    int NX = inp.size(2);
    int NY = inp.size(1);
    auto device = inp.device();
    if (!inp.sizes().equals(out.sizes())) throw std::runtime_error("input and output must have the same size");
    if (device.type != out.device().type || device.index != out.device().index) throw std::runtime_error("input and output must be on the same device");
    if (device.type != ker.device().type || device.index != ker.device().index) throw std::runtime_error("input and kernel must be on the same device");
    if (!ker.sizes().equals({ 2 * NY, 2 * NX })) throw std::runtime_error("scratch must have shape (2*NY, 2*NX)");

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
    if (inp.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("input dtype must be complex float");
    if (out.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("output dtype must be complex float");
    if (ker.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("kernel dtype must be complex float");

    int NX = inp.size(3);
    int NY = inp.size(2);
    int NZ = inp.size(1);
    auto device = inp.device();
    if (!inp.sizes().equals(out.sizes())) throw std::runtime_error("input and output must have the same size");
    if (device.type != out.device().type || device.index != out.device().index) throw std::runtime_error("input and output must be on the same device");
    if (device.type != ker.device().type || device.index != ker.device().index) throw std::runtime_error("input and kernel must be on the same device");
    if (!ker.sizes().equals({ 2 * NZ, 2 * NY, 2 * NX })) throw std::runtime_error("scratch must have shape (2*NZ, 2*NY, 2*NX)");

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


void toeplitz_multiplication(
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