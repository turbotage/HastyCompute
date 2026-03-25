module;

export module hasty_tensor_mod:external;

import std;
import hasty_util_mod;
import hasty_torch_wrapper;

import :tensor;


namespace hasty {

    export Tensor empty(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::empty(sizes.to_torch(), options.to_torch()));
    }

    export Tensor empty(ArrayRef<i64> sizes, Opt<Device> device, Opt<eScalarType> dtype)
    {
        if (device.has_value() && dtype.has_value()) {
            return empty(sizes, TensorOptions(*device, *dtype));
        } else if (device.has_value()) {
            return empty(sizes, TensorOptions().device(*device));
        } else if (dtype.has_value()) {
            return empty(sizes, TensorOptions().dtype(*dtype));
        } else {
            return empty(sizes, TensorOptions());
        }
    }

    export Tensor empty_like(const Tensor& other)
    {
        return Tensor(hat::empty_like(other._base));
    }

    export Tensor zeros(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::zeros(sizes.to_torch(), options.to_torch()));
    }

    export Tensor zeros(ArrayRef<i64> sizes, Opt<Device> device, Opt<eScalarType> dtype)
    {
        if (device.has_value() && dtype.has_value()) {
            return zeros(sizes, TensorOptions(*device, *dtype));
        } else if (device.has_value()) {
            return zeros(sizes, TensorOptions().device(*device));
        } else if (dtype.has_value()) {
            return zeros(sizes, TensorOptions().dtype(*dtype));
        } else {
            return zeros(sizes, TensorOptions());
        }
    }

    export Tensor zeros_like(const Tensor& other)
    {
        return Tensor(hat::zeros_like(other._base));
    }

    export Tensor ones(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::ones(sizes.to_torch(), options.to_torch()));
    }

    export Tensor ones(ArrayRef<i64> sizes, Opt<Device> device, Opt<eScalarType> dtype)
    {
        if (device.has_value() && dtype.has_value()) {
            return ones(sizes, TensorOptions(*device, *dtype));
        } else if (device.has_value()) {
            return ones(sizes, TensorOptions().device(*device));
        } else if (dtype.has_value()) {
            return ones(sizes, TensorOptions().dtype(*dtype));
        } else {
            return ones(sizes, TensorOptions());
        }
    }

    export Tensor ones_like(const Tensor& other)
    {
        return Tensor(hat::ones_like(other._base));
    }

    export Tensor rand(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::rand(sizes.to_torch(), options.to_torch()));
    }

    export Tensor rand(ArrayRef<i64> sizes, Opt<Device> device, Opt<eScalarType> dtype)
    {
        if (device.has_value() && dtype.has_value()) {
            return rand(sizes, TensorOptions(*device, *dtype));
        } else if (device.has_value()) {
            return rand(sizes, TensorOptions().device(*device));
        } else if (dtype.has_value()) {
            return rand(sizes, TensorOptions().dtype(*dtype));
        } else {
            return rand(sizes, TensorOptions());
        }
    }

    export Tensor rand_like(const Tensor& other)
    {
        return Tensor(hat::rand_like(other._base));
    }

    // arange overloads
    export Tensor arange(i64 end, TensorOptions opts = TensorOptions())
    {
        return Tensor(hat::arange(end, opts.to_torch()));
    }
    export Tensor arange(i64 start, i64 end, TensorOptions opts = TensorOptions())
    {
        return Tensor(hat::arange(start, end, opts.to_torch()));
    }
    export Tensor arange(i64 start, i64 end, i64 step, TensorOptions opts = TensorOptions())
    {
        return Tensor(hat::arange(start, end, step, opts.to_torch()));
    }

    // cat / stack — accept initializer_list or std::vector of Tensors
    namespace detail {
        inline std::vector<hat::Tensor> to_hat(const std::vector<Tensor>& ts) {
            std::vector<hat::Tensor> r;
            r.reserve(ts.size());
            for (const auto& t : ts) r.push_back(t.to_torch());
            return r;
        }
        inline std::vector<hat::Tensor> to_hat(std::initializer_list<Tensor> ts) {
            std::vector<hat::Tensor> r;
            r.reserve(ts.size());
            for (const auto& t : ts) r.push_back(t.to_torch());
            return r;
        }
    }

    export Tensor cat(std::initializer_list<Tensor> tensors, i64 dim)
    {
        return Tensor(hat::cat(detail::to_hat(tensors), dim));
    }
    export Tensor cat(const std::vector<Tensor>& tensors, i64 dim)
    {
        return Tensor(hat::cat(detail::to_hat(tensors), dim));
    }

    export Tensor stack(std::initializer_list<Tensor> tensors, i64 dim = 0)
    {
        return Tensor(hat::stack(detail::to_hat(tensors), dim));
    }
    export Tensor stack(const std::vector<Tensor>& tensors, i64 dim = 0)
    {
        return Tensor(hat::stack(detail::to_hat(tensors), dim));
    }

    export Tensor view_as_complex(const Tensor& t)
    {
        return Tensor(hat::view_as_complex(t.to_torch()));
    }

    // fftn — wraps torch::fft::fftn; dims lifetime is caller's responsibility (ArrayRef is non-owning)
    export Tensor fftn(
        const Tensor&         t,
        Opt<ArrayRef<i64>>    s    = nullopt,
        Opt<ArrayRef<i64>>    dims = nullopt,
        Opt<std::string_view> norm = nullopt)
    {
        // c10::optional = std::optional, c10::ArrayRef = at::IntArrayRef = hat::IntArrayRef
        std::optional<hat::IntArrayRef> s_ref =
            s    ? std::optional<hat::IntArrayRef>(s->to_torch())    : std::nullopt;
        std::optional<hat::IntArrayRef> dim_ref =
            dims ? std::optional<hat::IntArrayRef>(dims->to_torch()) : std::nullopt;
        std::optional<std::string_view> norm_ref =
            norm ? std::optional<std::string_view>(*norm) : std::nullopt;
        return Tensor(htorch::fft::fftn(t.to_torch(), s_ref, dim_ref, norm_ref));
    }

}

