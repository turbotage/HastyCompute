module;

#include <finufft.h>
#include <cufinufft.h>


export module fft_mod:nufft;

import util;
import tensor_mod:tensor;

namespace hasty {
namespace fft {

void verify_coords(const Tensor& coords)
{

}

template<is_device D, is_real_fp_tensor_type T>
requires std::is_same_v<T, f32> || std::is_same_v<T, f64>
struct finufft_plan_t {
    using IS_CUDA = std::bool_constant<std::is_same_v<D, cuda_t>>;
    using IS_FULL_PRECISION = std::bool_constant<std::is_same_v<T, f64>>;
    std::conditional_t<std::is_same_v<D, cuda_t>,
        std::conditional_t<std::is_same_v<T, f32>, cufinufftf_plan, cufinufft_plan>,
        std::conditional_t<std::is_same_v<T, f32>, finufftf_plan, finufft_plan>
    > plan;
};

template<is_device D>
struct finufft_opts_t {
    using IS_CUDA = std::bool_constant<std::is_same_v<D, cuda_t>>;
    std::conditional_t<std::is_same_v<D, cuda_t>, cufinufft_opts, finufft_opts> opts;
};

namespace nufft {
    export using T1 = empty_strong_typedef<struct T1_>;
    export using T2 = empty_strong_typedef<struct T2_>;
    export using T3 = empty_strong_typedef<struct T3_>;

    export using NTU = T1;
    export using UTN = T2;
    export using NTN = T3;

    export using NONUNIFORM_TO_UNIFORM = NTU;
    export using UNIFORM_TO_NONUNIFORM = UTN;
    export using NONUNIFORM_TO_NONUNIFORM = NTN;
    //export using UTU = // This is just an FFT

}

export template<typename T>
concept is_nufft_type = std::is_same_v<T, nufft::NTU> || std::is_same_v<T, nufft::UTN> || std::is_same_v<T, nufft::NTN>;

export enum struct eNufftSign : i8 {
    POS = 1,
    NEG = -1,
    DEFAULT_TYPE_1 = POS,
    DEFAULT_TYPE_2 = NEG
};

export enum struct eModeOrder : i8 {
    CMCL,
    FFT
};

export template<is_device D, is_real_fp_tensor_type T, is_dim3 N, is_nufft_type NT>
requires std::is_same_v<T, f32> || std::is_same_v<T, f64>
struct NufftPlan {};

export template<is_fp_real_tensor_type T, is_dim3 N, is_nufft_type NT>
struct NufftPlan<cpu_t, T, N, NT> {

private:
    finufft_plan_t<cpu_t, T> plan;

    

};





}
}