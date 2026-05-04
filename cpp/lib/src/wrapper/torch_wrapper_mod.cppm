module;

#include <ATen/core/ivalue.h>
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/library.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "c10/cuda/CUDAStream.h"
#include <c10/cuda/CUDAGuard.h>

namespace std {
    template<>
    void swap(std::pair<c10::IValue, c10::IValue>& lhs, std::pair<c10::IValue, c10::IValue>& rhs)
        noexcept(noexcept(lhs.swap(rhs))) {
        lhs.first.swap(rhs.first);
        lhs.second.swap(rhs.second);
    }
}

namespace c10 {
    inline void swap(
        std::pair<c10::IValue, c10::IValue>& lhs, 
        std::pair<c10::IValue, c10::IValue>& rhs
    ) noexcept 
    {
        using std::swap;
        swap(lhs.first, rhs.first);
        swap(lhs.second, rhs.second);
    }
}

export module hasty_torch_wrapper;


//export using at::Tensor;
//export using at::IntArrayRef;
//export using at::TensorOptions;

export template <typename Cond, typename... Args>
inline void torch_check(Cond&& cond, Args&&... args) {
    TORCH_CHECK(std::forward<Cond>(cond), std::forward<Args>(args)...);
}

export inline std::ostream& operator<<(std::ostream& os, const c10::FunctionSchema& schema) {
    return ::operator<<(os, schema); // calls the original operator<<
}

export namespace hat {
    // Types
    using at::Tensor;
    using at::Scalar;
    using at::ScalarType;
    using at::ArrayRef;
    using at::IntArrayRef;
    using at::TensorOptions;
    using at::Device;
    using at::DeviceType;
    using at::DeviceIndex;
    using at::Stream;
    using at::OptionalArrayRef;
    using at::InferenceMode;
    using at::MemoryFormat;
    using at::Layout;

    using at::empty;
    using at::zeros;
    using at::ones;
    using at::rand;
    using at::empty_like;
    using at::zeros_like;
    using at::ones_like;
    using at::rand_like;
    using at::real;
    using at::imag;
    using at::abs;
    using at::max;
    using at::min;
    using at::mean;
    using at::std;
    using at::median;
    using at::makeArrayRef;
    using at::from_blob;
    using at::scalar_tensor;
    using at::complex;
    using at::stack;
    using at::cat;
    using at::vdot;
    using at::exp;
    using at::arange;
    using at::view_as_complex;
    using at::allclose;
    using at::mm;
    using at::mv;
    using at::clamp;
    using at::log1p;
    using at::pow;
    using at::where;
    using at::sgn;




    namespace indexing {
        using at::indexing::TensorIndex;
        using at::indexing::TensorIndexType;
        inline constexpr auto None = at::indexing::None;
        using at::indexing::Ellipsis;
        using at::indexing::Slice;
    }
    namespace cuda {
        using at::cuda::CUDAGuard;
        namespace CUDACachingAllocator {
            using at::cuda::CUDACachingAllocator::getDeviceStats;
            using at::cuda::CUDACachingAllocator::emptyCache;
        }

        using at::cuda::getDefaultCUDAStream;
        using at::cuda::device_count;

        using at::cuda::CUDAStream;
        using at::cuda::CUDAStreamGuard;
    }

}

export namespace htorch {

    using torch::NoGradGuard;

    namespace cuda {
        using torch::cuda::synchronize;
    }
    namespace fft {
        using torch::fft::fftn;
        using torch::fft::ifftn;
        using torch::fft::rfftn;
        using torch::fft::irfftn;
        using torch::fft::hfftn;
        using torch::fft::ihfftn;

        using torch::fft::fftshift;
        using torch::fft::ifftshift;

        using torch::fft::fft;
        using torch::fft::rfft;
        using torch::fft::hfft;
        using torch::fft::ifft;
        using torch::fft::irfft;
        using torch::fft::ihfft;
    }
    namespace jit {
        using torch::jit::CompilationUnit;
        using torch::jit::Module;
        using torch::jit::IValue;

        using torch::jit::freeze;
        using torch::jit::optimize_for_inference;
        using torch::jit::load;
        using torch::jit::getAllOperators;
    }

}

export namespace hc10 {
    using c10::SymInt;
    using c10::Error;
    using c10::string_view;
    using c10::complex;
    using c10::List;
    using c10::Dict;
    using c10::IValue;
    using c10::ArrayRef;
    using c10::intrusive_ptr;
    using c10::StringType;
    using c10::Type;
    using c10::TypePtr;

    using c10::optional;

    namespace ivalue {
        using c10::ivalue::Tuple;
        using c10::ivalue::TupleElements;
    }

    namespace impl {
        using c10::impl::GenericList;
        using c10::impl::GenericDict;
        using c10::impl::toList;
    }
}

/*
export namespace hc10 {
    using c10::optional
}
*/