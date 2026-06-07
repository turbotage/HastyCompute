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

import hasty_viz_mod;

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
    int NX,
    bool accumulate,
    bool output_single_batch,
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
        (void*)&output,
        (void*)&scratch,
        (void*)&mult1,
        (void*)&mult2,
        (void*)&mult3,
        (void*)&batch_mult,
        (void*)&input_mult1_type,  (void*)&output_mult1_type,
        (void*)&input_mult2_type,  (void*)&output_mult2_type,
        (void*)&input_mult3_type,  (void*)&output_mult3_type,
        (void*)&input_bm_type,     (void*)&output_bm_type,
        (void*)&batch_in,
        (void*)&batch_out,
        (void*)&NX,
        (void*)&accumulate,
        (void*)&output_single_batch
    };

    CUresult cres = cuLaunchKernel(nvrtc_modules[device_idx].function,
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0, args, 0);
    if (cres != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed with error code " + std::to_string(cres));
    }
}


void launch_toeplitz_load_2D(
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
    int NX, int NY,
    bool accumulate,
    bool output_single_batch,
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

    cudaSetDevice(device_idx);
    int totalThreads = NX * NY;
    int blocks = (totalThreads + threads_per_block - 1) / threads_per_block;

    void* args[] = {
        (void*)&output,
        (void*)&scratch,
        (void*)&mult1,
        (void*)&mult2,
        (void*)&mult3,
        (void*)&batch_mult,
        (void*)&input_mult1_type,  (void*)&output_mult1_type,
        (void*)&input_mult2_type,  (void*)&output_mult2_type,
        (void*)&input_mult3_type,  (void*)&output_mult3_type,
        (void*)&input_bm_type,     (void*)&output_bm_type,
        (void*)&batch_in,
        (void*)&batch_out,
        (void*)&NX, (void*)&NY,
        (void*)&accumulate,
        (void*)&output_single_batch
    };

    CUresult cres = cuLaunchKernel(
        nvrtc_modules[device_idx].function,
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        args, 0
    );
    if (cres != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed with error code " + std::to_string(cres));
    }
}


void launch_toeplitz_load_3D(
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
    bool output_single_batch,
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

    cudaSetDevice(device_idx);
    int totalThreads = NX * NY * NZ;
    int blocks = (totalThreads + threads_per_block - 1) / threads_per_block;

    void* args[] = {
        (void*)&output,
        (void*)&scratch,
        (void*)&mult1,
        (void*)&mult2,
        (void*)&mult3,
        (void*)&batch_mult,
        (void*)&input_mult1_type,  (void*)&output_mult1_type,
        (void*)&input_mult2_type,  (void*)&output_mult2_type,
        (void*)&input_mult3_type,  (void*)&output_mult3_type,
        (void*)&input_bm_type,     (void*)&output_bm_type,
        (void*)&batch_in,
        (void*)&batch_out,
        (void*)&NX, (void*)&NY, (void*)&NZ,
        (void*)&accumulate,
        (void*)&output_single_batch
    };

    CUresult cres = cuLaunchKernel(
        nvrtc_modules[device_idx].function,
        blocks, 1, 1,
        threads_per_block, 1, 1,
        0, 0,
        args, 0
    );
    if (cres != CUDA_SUCCESS) {
        throw std::runtime_error("cuLaunchKernel failed with error code " + std::to_string(cres));
    }
}


void perform_toeplitz_multiplication_cuda_1D(
    Tensor&                 output,
    const Tensor&           kernel,
    OptRefW<Tensor>         scratch,
    OptCRefW<Tensor>        mult1,
    int input_mult1_type,   int output_mult1_type,
    OptCRefW<Tensor>        mult2,
    int input_mult2_type,   int output_mult2_type,
    OptCRefW<Tensor>        mult3,
    int input_mult3_type,   int output_mult3_type,
    OptCRefW<Tensor>        batch_mult,
    int input_bm_type,      int output_bm_type,
    int accumulate_type)
{
    auto device = output.device();
    int  NX     = output.size(output.ndimension() - 1);
    if (NX <= 1) throw std::runtime_error("spatial dimension must be positive");
    int device_idx = static_cast<int>(device.index);
    bool accumulate        = (accumulate_type != (int)ToeplitzAccumulateType::NONE);
    bool output_single_batch = (output.size(0) == 1);

    int nbatch;
    if (batch_mult.has_value()) {
        nbatch = (int)(*batch_mult).get().size(0);
    } else {
        if (output.size(0) != 1)
            throw std::runtime_error("output batch size must be 1 when batch_mult is not provided");
        nbatch = 1;
    }

    cuFloatComplex*       out_ptr    = reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.const_data_ptr<c64>());

    cuFloatComplex* scratch_ptr;
    Opt<Tensor> scratchmem;
    if (scratch.has_value()) {
        Tensor& scr = (*scratch).get();
        if (scr.device().type != eDeviceType::CUDA) throw std::runtime_error("scratch must be a CUDA tensor");
        if (scr.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("scratch dtype must be complex float");
        if (scr.ndimension() != 1 || scr.size(0) != 2 * NX) throw std::runtime_error("scratch must have shape (2*NX)");
        if (scr.device().index != device.index) throw std::runtime_error("scratch must be on the same device");
        if (!scr.is_contiguous()) throw std::runtime_error("scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.mutable_data_ptr<c64>());
    } else {
        scratchmem  = hasty::empty({ 2 * NX }, TensorOptions(output.device()).dtype(output.scalar_type()));
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem->mutable_data_ptr<c64>());
    }

    auto get_ptr = [&](OptCRefW<Tensor> m, const char* name, int size) -> const cuFloatComplex* {
        if (!m.has_value()) return nullptr;
        const Tensor& t = (*m).get();
        if (t.device().type != eDeviceType::CUDA || t.scalar_type() != eScalarType::ComplexFloat)
            throw std::runtime_error(std::string(name) + " type error");
        if (t.ndimension() != 1 || t.size(0) != size)
            throw std::runtime_error(std::string(name) + " must have shape (NX)");
        if (t.device().index != device.index || !t.is_contiguous())
            throw std::runtime_error(std::string(name) + " device/contiguity error");
        return reinterpret_cast<const cuFloatComplex*>(t.const_data_ptr<c64>());
    };

    const cuFloatComplex* mult1_ptr = get_ptr(mult1, "mult1", NX);
    const cuFloatComplex* mult2_ptr = get_ptr(mult2, "mult2", NX);
    const cuFloatComplex* mult3_ptr = get_ptr(mult3, "mult3", NX);

    const cuFloatComplex* bm_ptr = nullptr;
    if (batch_mult.has_value()) {
        const Tensor& bm = (*batch_mult).get();
        if (bm.device().type != eDeviceType::CUDA || bm.scalar_type() != eScalarType::ComplexFloat)
            throw std::runtime_error("batch_mult type error");
        if (bm.ndimension() != 2 || bm.size(0) != nbatch || bm.size(1) != NX)
            throw std::runtime_error("batch_mult must have shape (nbatch, NX)");
        if (bm.device().index != device.index || !bm.is_contiguous())
            throw std::runtime_error("batch_mult device/contiguity error");
        bm_ptr = reinterpret_cast<const cuFloatComplex*>(bm.const_data_ptr<c64>());
    }

    // Prime: load batch 0 into scratch
    launch_toeplitz_load_1D(
        out_ptr, scratch_ptr,
        mult1_ptr, mult2_ptr, mult3_ptr, bm_ptr,
        input_mult1_type, output_mult1_type,
        input_mult2_type, output_mult2_type,
        input_mult3_type, output_mult3_type,
        input_bm_type, output_bm_type,
        0, -1, NX, accumulate, output_single_batch, device_idx);

    VkFFT_Cache::VkFFT_Key key(device_idx);
    key.performConvolution    = true;
    key.size[0]               = 2 * NX;
    key.FFTdim                = 1;
    key.performZeropadding[0] = true;
    key.fft_zeropad_left[0]   = NX;
    key.fft_zeropad_right[0]  = 2 * NX;

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
            out_ptr, scratch_ptr,
            mult1_ptr, mult2_ptr, mult3_ptr, bm_ptr,
            input_mult1_type, output_mult1_type,
            input_mult2_type, output_mult2_type,
            input_mult3_type, output_mult3_type,
            input_bm_type, output_bm_type,
            (b < (nbatch - 1)) ? (b + 1) : -1, b,
            NX, accumulate, output_single_batch, device_idx);
    }
}


void perform_toeplitz_multiplication_cuda_2D(
    Tensor&                 output,
    const Tensor&           kernel,
    OptRefW<Tensor>         scratch,
    OptCRefW<Tensor>        mult1,
    int input_mult1_type,   int output_mult1_type,
    OptCRefW<Tensor>        mult2,
    int input_mult2_type,   int output_mult2_type,
    OptCRefW<Tensor>        mult3,
    int input_mult3_type,   int output_mult3_type,
    OptCRefW<Tensor>        batch_mult,
    int input_bm_type,      int output_bm_type,
    int accumulate_type)
{
    auto device = output.device();
    int dim = output.ndimension();
    int NX  = output.size(dim - 1);
    int NY  = output.size(dim - 2);
    if (NY <= 1 || NX <= 1) throw std::runtime_error("spatial dimensions must be positive");
    int device_idx = static_cast<int>(device.index);
    bool accumulate          = (accumulate_type != (int)ToeplitzAccumulateType::NONE);
    bool output_single_batch = (output.size(0) == 1);

    int nbatch;
    if (batch_mult.has_value()) {
        nbatch = (int)(*batch_mult).get().size(0);
    } else {
        if (output.size(0) != 1)
            throw std::runtime_error("output batch size must be 1 when batch_mult is not provided");
        nbatch = 1;
    }

    cuFloatComplex*       out_ptr    = reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.const_data_ptr<c64>());

    cuFloatComplex* scratch_ptr;
    Opt<Tensor> scratchmem;
    if (scratch.has_value()) {
        Tensor& scr = (*scratch).get();
        if (scr.device().type != eDeviceType::CUDA) throw std::runtime_error("scratch was not a CUDA tensor");
        if (scr.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("scratch dtype must be complex float");
        if (!scr.sizes().equals({ 2 * NY, 2 * NX })) throw std::runtime_error("scratch must have shape (2*NY, 2*NX)");
        if (scr.device().index != device.index) throw std::runtime_error("scratch must be on the same device as output");
        if (!scr.is_contiguous()) throw std::runtime_error("scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.mutable_data_ptr<c64>());
    } else {
        scratchmem  = hasty::empty({ 2 * NY, 2 * NX }, TensorOptions(output.device()).dtype(output.scalar_type()));
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem->mutable_data_ptr<c64>());
    }

    auto get_ptr = [&](OptCRefW<Tensor> m, const char* name) -> const cuFloatComplex* {
        if (!m.has_value()) return nullptr;
        const Tensor& t = (*m).get();
        if (t.device().type != eDeviceType::CUDA || t.scalar_type() != eScalarType::ComplexFloat)
            throw std::runtime_error(std::string(name) + " type error");
        if (!t.sizes().equals({ NY, NX }))
            throw std::runtime_error(std::string(name) + " must have shape (NY, NX)");
        if (t.device().index != device.index || !t.is_contiguous())
            throw std::runtime_error(std::string(name) + " device/contiguity error");
        return reinterpret_cast<const cuFloatComplex*>(t.const_data_ptr<c64>());
    };

    const cuFloatComplex* mult1_ptr = get_ptr(mult1, "mult1");
    const cuFloatComplex* mult2_ptr = get_ptr(mult2, "mult2");
    const cuFloatComplex* mult3_ptr = get_ptr(mult3, "mult3");

    const cuFloatComplex* bm_ptr = nullptr;
    if (batch_mult.has_value()) {
        const Tensor& bm = (*batch_mult).get();
        if (bm.device().type != eDeviceType::CUDA || bm.scalar_type() != eScalarType::ComplexFloat)
            throw std::runtime_error("batch_mult type error");
        if (!bm.sizes().equals({ nbatch, NY, NX }))
            throw std::runtime_error("batch_mult must have shape (nbatch, NY, NX)");
        if (bm.device().index != device.index || !bm.is_contiguous())
            throw std::runtime_error("batch_mult device/contiguity error");
        bm_ptr = reinterpret_cast<const cuFloatComplex*>(bm.const_data_ptr<c64>());
    }

    // Prime: load batch 0 into scratch
    launch_toeplitz_load_2D(
        out_ptr, scratch_ptr,
        mult1_ptr, mult2_ptr, mult3_ptr, bm_ptr,
        input_mult1_type, output_mult1_type,
        input_mult2_type, output_mult2_type,
        input_mult3_type, output_mult3_type,
        input_bm_type, output_bm_type,
        0, -1, NX, NY, accumulate, output_single_batch, device_idx);

    VkFFT_Cache::VkFFT_Key key(device_idx);
    key.performConvolution    = true;
    key.size[0]               = 2 * NX;
    key.size[1]               = 2 * NY;
    key.FFTdim                = 2;
    key.performZeropadding[0] = true;
    key.performZeropadding[1] = true;
    key.fft_zeropad_left[0]   = NX;
    key.fft_zeropad_left[1]   = NY;
    key.fft_zeropad_right[0]  = 2 * NX;
    key.fft_zeropad_right[1]  = 2 * NY;
    key.useLUT                = true;

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

        launch_toeplitz_load_2D(
            out_ptr, scratch_ptr,
            mult1_ptr, mult2_ptr, mult3_ptr, bm_ptr,
            input_mult1_type, output_mult1_type,
            input_mult2_type, output_mult2_type,
            input_mult3_type, output_mult3_type,
            input_bm_type, output_bm_type,
            (b < (nbatch - 1)) ? (b + 1) : -1, b,
            NX, NY, accumulate, output_single_batch, device_idx);
    }
}


void perform_toeplitz_multiplication_cuda_3D(
    Tensor&                 output,
    const Tensor&           kernel,
    OptRefW<Tensor>         scratch,
    OptCRefW<Tensor>        mult1,
    int input_mult1_type,   int output_mult1_type,
    OptCRefW<Tensor>        mult2,
    int input_mult2_type,   int output_mult2_type,
    OptCRefW<Tensor>        mult3,
    int input_mult3_type,   int output_mult3_type,
    OptCRefW<Tensor>        batch_mult,
    int input_bm_type,      int output_bm_type,
    int accumulate_type)
{
    auto device = output.device();
    int dim = output.ndimension();
    int NX  = output.size(dim - 1);
    int NY  = output.size(dim - 2);
    int NZ  = output.size(dim - 3);
    if (NZ <= 1 || NY <= 1 || NX <= 1) throw std::runtime_error("spatial dimensions must be positive");
    int device_idx = static_cast<int>(device.index);
    bool accumulate          = (accumulate_type != (int)ToeplitzAccumulateType::NONE);
    bool output_single_batch = (output.size(0) == 1);

    int nbatch;
    if (batch_mult.has_value()) {
        nbatch = (int)(*batch_mult).get().size(0);
    } else {
        if (output.size(0) != 1)
            throw std::runtime_error("output batch size must be 1 when batch_mult is not provided");
        nbatch = 1;
    }

    cuFloatComplex*       out_ptr    = reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>());
    const cuFloatComplex* kernel_ptr = reinterpret_cast<const cuFloatComplex*>(kernel.const_data_ptr<c64>());

    cuFloatComplex* scratch_ptr;
    Opt<Tensor> scratchmem;
    if (scratch.has_value()) {
        Tensor& scr = (*scratch).get();
        if (scr.device().type != eDeviceType::CUDA) throw std::runtime_error("scratch was not a CUDA tensor");
        if (scr.scalar_type() != eScalarType::ComplexFloat) throw std::runtime_error("scratch dtype must be complex float");
        if (!scr.sizes().equals({ 2*NZ, 2*NY, 2*NX })) throw std::runtime_error("scratch must have shape (2*NZ, 2*NY, 2*NX)");
        if (scr.device().index != device.index) throw std::runtime_error("scratch must be on the same device as output");
        if (!scr.is_contiguous()) throw std::runtime_error("scratch must be contiguous");
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scr.mutable_data_ptr<c64>());
    } else {
        scratchmem  = hasty::empty({ 2*NZ, 2*NY, 2*NX }, TensorOptions(output.device()).dtype(output.scalar_type()));
        scratch_ptr = reinterpret_cast<cuFloatComplex*>(scratchmem->mutable_data_ptr<c64>());
    }

    auto get_ptr = [&](OptCRefW<Tensor> m, const char* name) -> const cuFloatComplex* {
        if (!m.has_value()) return nullptr;
        const Tensor& t = (*m).get();
        if (t.device().type != eDeviceType::CUDA || t.scalar_type() != eScalarType::ComplexFloat)
            throw std::runtime_error(std::string(name) + " type error");
        if (!t.sizes().equals({ NZ, NY, NX }))
            throw std::runtime_error(std::string(name) + " must have shape (NZ, NY, NX)");
        if (t.device().index != device.index || !t.is_contiguous())
            throw std::runtime_error(std::string(name) + " device/contiguity error");
        return reinterpret_cast<const cuFloatComplex*>(t.const_data_ptr<c64>());
    };

    const cuFloatComplex* mult1_ptr = get_ptr(mult1, "mult1");
    const cuFloatComplex* mult2_ptr = get_ptr(mult2, "mult2");
    const cuFloatComplex* mult3_ptr = get_ptr(mult3, "mult3");

    const cuFloatComplex* bm_ptr = nullptr;
    if (batch_mult.has_value()) {
        const Tensor& bm = (*batch_mult).get();
        if (bm.device().type != eDeviceType::CUDA || bm.scalar_type() != eScalarType::ComplexFloat)
            throw std::runtime_error("batch_mult type error");
        if (!bm.sizes().equals({ nbatch, NZ, NY, NX }))
            throw std::runtime_error("batch_mult must have shape (nbatch, NZ, NY, NX)");
        if (bm.device().index != device.index || !bm.is_contiguous())
            throw std::runtime_error("batch_mult device/contiguity error");
        bm_ptr = reinterpret_cast<const cuFloatComplex*>(bm.const_data_ptr<c64>());
    }

    // Prime: load batch 0 into scratch
    launch_toeplitz_load_3D(
        out_ptr, scratch_ptr,
        mult1_ptr, mult2_ptr, mult3_ptr, bm_ptr,
        input_mult1_type, output_mult1_type,
        input_mult2_type, output_mult2_type,
        input_mult3_type, output_mult3_type,
        input_bm_type, output_bm_type,
        0, -1, NX, NY, NZ, accumulate, output_single_batch, device_idx);

    VkFFT_Cache::VkFFT_Key key(device_idx);
    key.performConvolution    = true;
    key.size[0]               = 2 * NX;
    key.size[1]               = 2 * NY;
    key.size[2]               = 2 * NZ;
    key.FFTdim                = 3;
    key.performZeropadding[0] = true;
    key.performZeropadding[1] = true;
    key.performZeropadding[2] = true;
    key.fft_zeropad_left[0]   = NX;
    key.fft_zeropad_left[1]   = NY;
    key.fft_zeropad_left[2]   = NZ;
    key.fft_zeropad_right[0]  = 2 * NX;
    key.fft_zeropad_right[1]  = 2 * NY;
    key.fft_zeropad_right[2]  = 2 * NZ;

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

        launch_toeplitz_load_3D(
            out_ptr, scratch_ptr,
            mult1_ptr, mult2_ptr, mult3_ptr, bm_ptr,
            input_mult1_type, output_mult1_type,
            input_mult2_type, output_mult2_type,
            input_mult3_type, output_mult3_type,
            input_bm_type, output_bm_type,
            (b < (nbatch - 1)) ? b+1 : -1, b,
            NX, NY, NZ, accumulate, output_single_batch, device_idx);
    }
}


void toeplitz_multiplication(
    Tensor&                     output,
    const Tensor&               kernel,
    OptRefW<Tensor>             scratch,
    const ToeplitzMultiplier&   mult1,
    const ToeplitzMultiplier&   mult2,
    const ToeplitzMultiplier&   mult3,
    const ToeplitzMultiplier&   batch_mult,
    ToeplitzAccumulateType      accumulate_type)
{
    bool any_mult = mult1.mult.has_value() || mult2.mult.has_value() ||
                    mult3.mult.has_value() || batch_mult.mult.has_value();
    if (!any_mult)
        throw std::runtime_error("at least one ToeplitzMultiplier must have a tensor");

    auto dispatch = [&](auto perform_fn) {
        perform_fn(
            output, kernel, scratch,
            mult1.mult, (int)mult1.input_mult_type, (int)mult1.output_mult_type,
            mult2.mult, (int)mult2.input_mult_type, (int)mult2.output_mult_type,
            mult3.mult, (int)mult3.input_mult_type, (int)mult3.output_mult_type,
            batch_mult.mult, (int)batch_mult.input_mult_type, (int)batch_mult.output_mult_type,
            (int)accumulate_type
        );
    };

    if (kernel.ndimension() == 1) {
        if (output.ndimension() != 2)
            throw std::runtime_error("output must be 2D (batch, NX) for 1D kernel");
        dispatch(perform_toeplitz_multiplication_cuda_1D);
    }
    else if (kernel.ndimension() == 2) {
        if (output.ndimension() != 3)
            throw std::runtime_error("output must be 3D (batch, NY, NX) for 2D kernel");
        dispatch(perform_toeplitz_multiplication_cuda_2D);
    }
    else if (kernel.ndimension() == 3) {
        if (output.ndimension() != 4)
            throw std::runtime_error("output must be 4D (batch, NZ, NY, NX) for 3D kernel");
        dispatch(perform_toeplitz_multiplication_cuda_3D);
    }
    else {
        throw std::runtime_error("kernel must be 1D, 2D or 3D");
    }
}

}
}
