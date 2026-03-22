module;

#include <finufft.h>
#include <cufinufft.h>
#include <cuda_runtime.h>


export module fft_mod:nufft;

import std;
import util_mod;
import tensor_mod;

namespace hasty {
namespace fft {

template<is_device D, is_real_fp_tensor_type T, std::size_t N>
requires (std::is_same_v<T, f32> || std::is_same_v<T, f64>) && is_dim3<N>
void verify_coords(const Tensor& coords)
{
    if constexpr(std::is_same_v<D, cpu_t>) {
        if (coords.device().type != eDeviceType::CPU) {
            throw std::runtime_error("Coords tensor must be on CPU");
        }
    } else if constexpr(std::is_same_v<D, cuda_t>) {
        if (coords.device().type != eDeviceType::CUDA) {
            throw std::runtime_error("Coords tensor must be on CUDA");
        }
    }

    if constexpr (std::is_same_v<T, f32>) {
        if (coords.scalar_type() != eScalarType::Float) {
            throw std::runtime_error("Coords tensor must be of type float");
        }
    } else if constexpr (std::is_same_v<T, f64>) {
        if (coords.scalar_type() != eScalarType::Double) {
            throw std::runtime_error("Coords tensor must be of type double");
        }
    }

    if (coords.sizes().size() != 2) {
        throw std::runtime_error("Coords tensor must be 2D");
    }

    if (coords.sizes()[0] != N) {
        throw std::runtime_error("Coords tensor first dimension must be equal to N");
    }
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

export using NTU = empty_strong_typedef<struct NTU_>;
export using UTN = empty_strong_typedef<struct UTN_>;
export using NTN = empty_strong_typedef<struct NTN_>;

export using T1 = NTU;
export using T2 = UTN;
export using T3 = NTN;

export using NONUNIFORM_TO_UNIFORM = NTU;
export using UNIFORM_TO_NONUNIFORM = UTN;
export using NONUNIFORM_TO_NONUNIFORM = NTN;
    //export using UTU = // This is just an FFT


export template<typename T>
concept is_nufft_type = std::is_same_v<T, fft::NTU> || std::is_same_v<T, fft::UTN> || std::is_same_v<T, fft::NTN>;

i32 nufft_type_to_int(is_nufft_type auto t) {
    if constexpr (std::is_same_v<decltype(t), fft::NTU>) {
        return 1;
    } else if constexpr (std::is_same_v<decltype(t), fft::UTN>) {
        return 2;
    } else if constexpr (std::is_same_v<decltype(t), fft::NTN>) {
        return 3;
    } else {
        static_assert(always_false<decltype(t)>, "Invalid nufft type");
    }
}

export template<is_device D, is_real_fp_tensor_type T, std::size_t N, is_nufft_type NT>
requires (std::is_same_v<T, f32> || std::is_same_v<T, f64>) && is_dim3<N>
struct NufftPlan {};

export template<is_device D, is_real_fp_tensor_type T,is_nufft_type NT>
requires std::is_same_v<T, f32> || std::is_same_v<T, f64>
struct NufftOptions {};


export template<is_real_fp_tensor_type T, is_nufft_type NT>
requires std::is_same_v<T, f32> || std::is_same_v<T, f64>
struct NufftOptions<cuda_t, T, NT> {

    enum struct eNufftSign : i8 {
        POS = 1,
        NEG = -1,
        DEFAULT_TYPE_1 = POS,
        DEFAULT_TYPE_2 = NEG,
        DEFAULT_NTU = DEFAULT_TYPE_1,
        DEFAULT_UTN = DEFAULT_TYPE_2
    };

    enum struct eModeOrder : i8 {
        CMCL,
        FFT,
        DEFAULT
    };

    enum struct eSpreadInterpMethod : i8 {
        NUFFT,
        SPREAD_ONLY,
        INTERP_ONLY = SPREAD_ONLY,
        SPREAD_INTERP_ONLY = SPREAD_ONLY,
        DEFAULT
    };

    enum struct eNufftMethod : i8 {
        GM_SORT,
        GM_NOT_SORT,
        SM,
        OD,
        BLOCK_GATHER,
        DEFAULT
    };

    enum struct eKernelEvalMethod : i8 {
        DIRECT,
        HORNER,
        DEFAULT
    };

    enum struct eUppsamplingFactor : i8 {
        UPSAMP_2_0,
        UPSAMP_1_25,
        UPSAMP_1_0,
        DEFAULT
    };
    
    eNufftSign sign = std::is_same_v<NT, fft::NTU> ? eNufftSign::DEFAULT_NTU : eNufftSign::DEFAULT_UTN;
    i32 ntransf = 1;
    double tol = std::is_same_v<T, f32> ? 1e-6 : 1e-15;

    eModeOrder mode_order = eModeOrder::DEFAULT;
    DeviceIndex device_index = device_alias::CUDA0;
    eSpreadInterpMethod spread_interp_method = eSpreadInterpMethod::DEFAULT;
    eNufftMethod method = eNufftMethod::DEFAULT;
    eKernelEvalMethod kernel_eval_method = eKernelEvalMethod::DEFAULT;
    std::variant<eUppsamplingFactor, double> upsampling_factor = eUppsamplingFactor::DEFAULT;
    cudaStream_t stream = cudaDefaultStream;

};

export template<is_real_fp_tensor_type T, std::size_t N, is_nufft_type NT>
requires (std::is_same_v<T, f32> || std::is_same_v<T, f64>) && is_dim3<N>
struct NufftPlan<cuda_t, T, N, NT> {

    NufftPlan(
        const std::array<i64, N>& nmodes,
        const NufftOptions<cuda_t, T, NT>& options = NufftOptions<cuda_t, T, NT>()
    )   : m_nmodes(nmodes), m_options(options)
    {
        cufinufft_default_opts(&m_opts);

        switch (m_options.method) {
        case NufftOptions<cuda_t, T, NT>::eNufftMethod::GM_SORT:
            m_opts.gpu_method = 1;
            m_opts.gpu_sort = 1;
            break;
        case NufftOptions<cuda_t, T, NT>::eNufftMethod::GM_NOT_SORT:
            m_opts.gpu_method = 1;
            m_opts.gpu_sort = 0;
            break;
        case NufftOptions<cuda_t, T, NT>::eNufftMethod::SM:
            m_opts.gpu_method = 2;
            break;
        case NufftOptions<cuda_t, T, NT>::eNufftMethod::OD:
            m_opts.gpu_method = 3;
            break;
        case NufftOptions<cuda_t, T, NT>::eNufftMethod::BLOCK_GATHER:
            m_opts.gpu_method = 4;
            break;
        case NufftOptions<cuda_t, T, NT>::eNufftMethod::DEFAULT:
            break;
        default:
            throw std::runtime_error("Invalid CUDA method");
        }

        if (std::holds_alternative<NufftOptions<cuda_t, T, NT>::eUppsamplingFactor>(m_options.upsampling_factor)) {
            switch (std::get<NufftOptions<cuda_t, T, NT>::eUppsamplingFactor>(m_options.upsampling_factor)) {
            case NufftOptions<cuda_t, T, NT>::eUppsamplingFactor::UPSAMP_2_0:
                m_opts.upsampfac = 2.0;
                break;
            case NufftOptions<cuda_t, T, NT>::eUppsamplingFactor::UPSAMP_1_25:
                m_opts.upsampfac = 1.25;
                break;
            case NufftOptions<cuda_t, T, NT>::eUppsamplingFactor::UPSAMP_1_0:
                m_opts.upsampfac = 1.0;
                if (m_options.spread_interp_method != NufftOptions<cuda_t, T, NT>::eSpreadInterpMethod::SPREAD_INTERP_ONLY) {
                    throw std::runtime_error("Only spread/interp method supported with upsampling factor 1.0");
                }
                break;
            case NufftOptions<cuda_t, T, NT>::eUppsamplingFactor::DEFAULT:
                break;
            default:
                throw std::runtime_error("Invalid upsampling factor");
            }
        } else {
            throw std::runtime_error("Custom upsampling factor not supported yet");
            double upfac = std::get<double>(m_options.upsampling_factor);
            if (upfac <= 1.0) {
                throw std::runtime_error("Upsampling factor must be greater than 1.0");
            }
            m_opts.upsampfac = upfac;
            m_opts.gpu_kerevalmeth = 0; // Must use direct kernel eval for custom upsampling factors
        }

        if (m_options.spread_interp_method != NufftOptions<cuda_t, T, NT>::eSpreadInterpMethod::DEFAULT) {
            switch (m_options.spread_interp_method) {
            case NufftOptions<cuda_t, T, NT>::eSpreadInterpMethod::NUFFT:
                m_opts.gpu_spreadinterponly = 0;
                break;
            case NufftOptions<cuda_t, T, NT>::eSpreadInterpMethod::SPREAD_ONLY:
                m_opts.gpu_spreadinterponly = 1;
                break;
            case NufftOptions<cuda_t, T, NT>::eSpreadInterpMethod::INTERP_ONLY:
                m_opts.gpu_spreadinterponly = 1;
                break;
            case NufftOptions<cuda_t, T, NT>::eSpreadInterpMethod::SPREAD_INTERP_ONLY:
                m_opts.gpu_spreadinterponly = 1;
                break;
            case NufftOptions<cuda_t, T, NT>::eSpreadInterpMethod::DEFAULT:
                break;
            default:
                throw std::runtime_error("Invalid spread/interp method");
            }
        }

        if constexpr(std::is_same_v<T, f32>) {
            cufinufftf_makeplan(
                nufft_type_to_int<NT>(),
                N,
                m_nmodes.data(),
                static_cast<int>(m_options.sign),
                m_options.ntransf,
                m_options.tol,
                &m_plan.plan,
                &m_opts
            );
        } else if constexpr(std::is_same_v<T, f64>) {
            cufinufft_makeplan(
                std::is_same_v<NT, fft::NTU> ? 1 : (std::is_same_v<NT, fft::UTN> ? 2 : 3),
                N,
                m_nmodes.data(),
                static_cast<int>(m_options.sign),
                m_options.ntransf,
                m_options.tol,
                &m_plan.plan,
                &m_opts
            );
        }

    }

    ~NufftPlan() 
    {
        if constexpr(std::is_same_v<T, f32>) {
            cufinufftf_destroy(m_plan.plan);
        } else if constexpr(std::is_same_v<T, f64>) {
            cufinufft_destroy(m_plan.plan);
        }
    }

    void setpts(const Tensor& coords)
    requires(!std::is_same_v<NT, fft::NTN>)
    {
        verify_coords<cpu_t, T, N>(coords);
        m_coords = coords;

        if constexpr(std::is_same_v<T, f32>) {
            cufinufftf_setpts(
                m_plan.plan,
                coords.sizes()[1],
                coords.select(0, 0).template mutable_data_ptr<T>(),
                (N > 1) ? coords.select(0, 1).template mutable_data_ptr<T>() : nullptr,
                (N > 2) ? coords.select(0, 2).template mutable_data_ptr<T>() : nullptr,
                0,
                nullptr,
                nullptr,
                nullptr
            );
        } else if constexpr(std::is_same_v<T, f64>) {
            cufinufft_setpts(
                m_plan.plan,
                coords.sizes()[1],
                coords.select(0, 0).template mutable_data_ptr<T>(),
                (N > 1) ? coords.select(0, 1).template mutable_data_ptr<T>() : nullptr,
                (N > 2) ? coords.select(0, 2).template mutable_data_ptr<T>() : nullptr,
                0,
                nullptr,
                nullptr,
                nullptr
            );
        }
    }

    void setpts(const Tensor& coords_in, const Tensor& coords_out)
    requires(std::is_same_v<NT, fft::NTN>)
    {
        verify_coords<cpu_t, T, N>(coords_in);
        verify_coords<cpu_t, T, N>(coords_out);
        m_coords = std::make_pair(coords_in, coords_out);

        if constexpr(std::is_same_v<T, f32>) {
            cufinufftf_setpts(
                m_plan.plan,
                coords_in.sizes()[1],
                coords_in.select(0, 0).template mutable_data_ptr<T>(),
                (N > 1) ? coords_in.select(0, 1).template mutable_data_ptr<T>() : nullptr,
                (N > 2) ? coords_in.select(0, 2).template mutable_data_ptr<T>() : nullptr,
                coords_out.sizes()[1],
                coords_out.select(0, 0).template mutable_data_ptr<T>(),
                (N > 1) ? coords_out.select(0, 1).template mutable_data_ptr<T>() : nullptr,
                (N > 2) ? coords_out.select(0, 2).template mutable_data_ptr<T>() : nullptr
            );
        } else if constexpr(std::is_same_v<T, f64>) {
            cufinufft_setpts(
                m_plan.plan,
                coords_in.sizes()[1],
                coords_in.select(0, 0).template mutable_data_ptr<T>(),
                (N > 1) ? coords_in.select(0, 1).template mutable_data_ptr<T>() : nullptr,
                (N > 2) ? coords_in.select(0, 2).template mutable_data_ptr<T>() : nullptr,
                coords_out.sizes()[1],
                coords_out.select(0, 0).template mutable_data_ptr<T>(),
                (N > 1) ? coords_out.select(0, 1).template mutable_data_ptr<T>() : nullptr,
                (N > 2) ? coords_out.select(0, 2).template mutable_data_ptr<T>() : nullptr
            );
        }
    }

    void execute(const Tensor& input, const Tensor& output) const
    {
        // Both input and output must be contiguous
        if (!input.is_contiguous())
            throw std::runtime_error("Input tensor must be contiguous");
        if (!output.is_contiguous())
            throw std::runtime_error("Output tensor must be contiguous");

        // Number of elements must match specification in plan stage
        if constexpr(std::is_same_v<NT, fft::NTU>) {
            if (input.numel() != m_options.ntransf * m_coords.sizes()[1]) {
                throw std::runtime_error("Input tensor numel must match number of input coordinates for NTU");
            }
            if (output.numel() != m_options.ntransf * std::accumulate(m_nmodes.begin(), m_nmodes.end(), 1LL, std::multiplies<>())) {
                throw std::runtime_error("Output tensor numel must match number of output coordinates for NTU");
            }
        } else if constexpr(std::is_same_v<NT, fft::UTN>) {
            if (input.numel() != m_options.ntransf * std::accumulate(m_nmodes.begin(), m_nmodes.end(), 1LL, std::multiplies<>())) {
                throw std::runtime_error("Input tensor numel must match number of input coordinates for UTN");
            }
            if (output.numel() != m_options.ntransf * m_coords.sizes()[1]) {
                throw std::runtime_error("Output tensor numel must match number of output coordinates for UTN");
            }
        } else if constexpr(std::is_same_v<NT, fft::NTN>) {
            if (input.numel() != m_options.ntransf * m_coords.first.sizes()[1]) {
                throw std::runtime_error("Input tensor numel must match number of input coordinates for NTN");
            }
            if (output.numel() != m_options.ntransf * m_coords.second.sizes()[1]) {
                throw std::runtime_error("Output tensor numel must match number of output coordinates for NTN");
            }
        }

        // Execute the plan
        if constexpr(std::is_same_v<NT, fft::NTU>) {
            if constexpr(std::is_same_v<T, f32>) {
                cufinufftf_execute(
                    m_plan.plan, 
                    reinterpret_cast<cuFloatComplex*>(input.mutable_data_ptr<c64>()), 
                    reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>())
                );
            } else if constexpr(std::is_same_v<T, f64>) {
                cufinufft_execute(
                    m_plan.plan, 
                    reinterpret_cast<cuDoubleComplex*>(input.mutable_data_ptr<c64>()), 
                    reinterpret_cast<cuDoubleComplex*>(output.mutable_data_ptr<c64>())
                );
            }
        } else if constexpr(std::is_same_v<NT, fft::UTN>) {
            if constexpr(std::is_same_v<T, f32>) {
                cufinufftf_execute(
                    m_plan.plan, 
                    reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>()), 
                    reinterpret_cast<cuFloatComplex*>(input.mutable_data_ptr<c64>())
                );
            } else if constexpr(std::is_same_v<T, f64>) {
                cufinufft_execute(
                    m_plan.plan, 
                    reinterpret_cast<cuDoubleComplex*>(output.mutable_data_ptr<c64>()), 
                    reinterpret_cast<cuDoubleComplex*>(input.mutable_data_ptr<c64>())
                );
            }
        } else if constexpr(std::is_same_v<NT, fft::NTN>) {
            if constexpr(std::is_same_v<T, f32>) {
                cufinufftf_execute(
                    m_plan.plan, 
                    reinterpret_cast<cuFloatComplex*>(input.mutable_data_ptr<c64>()), 
                    reinterpret_cast<cuFloatComplex*>(output.mutable_data_ptr<c64>())
                );
            } else if constexpr(std::is_same_v<T, f64>) {
                cufinufft_execute(
                    m_plan.plan, 
                    reinterpret_cast<cuDoubleComplex*>(input.mutable_data_ptr<c64>()), 
                    reinterpret_cast<cuDoubleComplex*>(output.mutable_data_ptr<c64>())
                );
            }
        }
    }

private:
    finufft_plan_t<cpu_t, T> m_plan;
    cufinufft_opts m_opts;

    std::conditional_t<std::is_same_v<NT, fft::NTN>, std::pair<Tensor,Tensor>, Tensor> m_coords;
    
    std::array<i64, N> m_nmodes;
    NufftOptions<cuda_t, T, NT> m_options;

};

export template<is_real_fp_tensor_type T, is_nufft_type NT>
requires std::is_same_v<T, f32> || std::is_same_v<T, f64>
struct NufftOptions<cpu_t, T, NT> {
    
    enum struct eNufftSign : i8 {
        POS = 1,
        NEG = -1,
        DEFAULT_TYPE_1 = POS,
        DEFAULT_TYPE_2 = NEG,
        DEFAULT_NTU = DEFAULT_TYPE_1,
        DEFAULT_UTN = DEFAULT_TYPE_2
    };

    enum struct eModeOrder : i8 {
        CMCL,
        FFT,
        DEFAULT
    };

    enum struct eSpreadInterpMethod : i8 {
        NUFFT,
        SPREAD_ONLY,
        INTERP_ONLY = SPREAD_ONLY,
        SPREAD_INTERP_ONLY = SPREAD_ONLY,
        DEFAULT
    };

    eNufftSign sign = std::is_same_v<NT, fft::NTU> ? eNufftSign::DEFAULT_NTU : eNufftSign::DEFAULT_UTN;
    i32 ntransf = 1;
    double tol = std::is_same_v<T, f32> ? 1e-6 : 1e-15;

    eModeOrder mode_order = eModeOrder::DEFAULT;
    eSpreadInterpMethod spread_interp_method = eSpreadInterpMethod::DEFAULT;
    Opt<i32> nthreads = std::nullopt; // If nullopt, will use finufft default (which is currently all threads available)
    Opt<double> upsampling_factor = std::nullopt; // If nullopt, will use finufft default
};

export template<is_real_fp_tensor_type T, std::size_t N, is_nufft_type NT>
requires (std::is_same_v<T, f32> || std::is_same_v<T, f64>) && is_dim3<N>
struct NufftPlan<cpu_t, T, N, NT> {

    NufftPlan(
        const std::array<i64, N>& nmodes,
        const NufftOptions<cpu_t, T, NT>& options = NufftOptions<cpu_t, T, NT>()
    )   : m_nmodes(nmodes), m_options(options)
    {
        finufft_default_opts(&m_opts);

        if (m_options.modeord != NufftOptions<cpu_t, T, NT>::eModeOrder::DEFAULT) {
            switch (m_options.mode_order) {
            case NufftOptions<cpu_t, T, NT>::eModeOrder::CMCL:
                m_opts.modeord = 0;
                break;
            case NufftOptions<cpu_t, T, NT>::eModeOrder::FFT:
                m_opts.modeord = 1;
                break;
            default:
                throw std::runtime_error("Invalid mode order");
            }
        }

        if (m_options.spread_interp_method != NufftOptions<cpu_t, T, NT>::eSpreadInterpMethod::DEFAULT) {
            switch (m_options.spread_interp_method) {
            case NufftOptions<cpu_t, T, NT>::eSpreadInterpMethod::NUFFT:
                m_opts.spreadinterponly = 0;
                break;
            case NufftOptions<cpu_t, T, NT>::eSpreadInterpMethod::SPREAD_INTERP_ONLY:
                m_opts.spreadinterponly = 1;
                break;
            default:
                throw std::runtime_error("Invalid spread/interp method");
            }
        }

        if (m_options.nthreads.has_value()) {
            if (m_options.nthreads.value() <= 0) {
                throw std::runtime_error("Number of threads must be positive");
            }
            m_opts.nthreads = m_options.nthreads.value();
        }

        if (m_options.upsampling_factor.has_value()) {
            if (m_options.upsampling_factor.value() <= 1.2) {
                throw std::runtime_error("Upsampling factor must be greater than 1.2");
            }
            m_opts.upsampfac = m_options.upsampling_factor.value();
        }

        if constexpr(std::is_same_v<T, f32>) {
            finufftf_makeplan(
                nufft_type_to_int<NT>(),
                N,
                m_nmodes.data(),
                static_cast<int>(m_options.sign),
                m_options.ntransf,
                m_options.tol,
                &m_plan.plan,
                &m_opts
            );
        } else if constexpr(std::is_same_v<T, f64>) {
            finufft_makeplan(
                std::is_same_v<NT, fft::NTU> ? 1 : (std::is_same_v<NT, fft::UTN> ? 2 : 3),
                N,
                m_nmodes.data(),
                static_cast<int>(m_options.sign),
                m_options.ntransf,
                m_options.tol,
                &m_plan.plan,
                &m_opts
            );
        }

    }

private:
    finufft_plan_t<cpu_t, T> m_plan;
    finufft_opts m_opts;

    std::conditional_t<std::is_same_v<NT, fft::NTN>, std::pair<Tensor,Tensor>, Tensor> m_coords;
    
    std::array<i64, N> m_nmodes;
    NufftOptions<cpu_t, T, NT> m_options;

};


}
}