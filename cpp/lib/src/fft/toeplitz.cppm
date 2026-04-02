module;

export module hasty_fft_mod:toeplitz;

import std;
import hasty_util_mod;
import hasty_tensor_mod;


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

export Tensor create_toeplitz_kernel(
    const Tensor&           coords,         // [ndim, npts] float on CUDA
    const Tensor&           weights,        // [npts] complex float on CUDA
    ArrayRef<i64>           im_size,                  // {NX} / {NY,NX} / {NZ,NY,NX}
    bool                    double_prec = false
);

export Tensor create_toeplitz_kernel_standard(
    const Tensor&           coords,
    const Tensor&           weights,
    ArrayRef<i64>           im_size
);

export void transform_toeplitz_kernel(Tensor& kernel, bool clear_vkfft_plan = false);

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
);

}
}

