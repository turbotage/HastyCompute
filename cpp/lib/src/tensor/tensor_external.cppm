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

export Tensor randperm(i64 n, TensorOptions opts = TensorOptions())
{
    return Tensor(hat::randperm(n, opts.to_torch()));
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

export Tensor clamp(const Tensor& t, const Scalar& min, const Scalar& max)
{
    return Tensor(hat::clamp(t.to_torch(), min.to_torch(), max.to_torch()));
}

export Tensor view_as_complex(const Tensor& t)
{
    return Tensor(hat::view_as_complex(t.to_torch()));
}

export Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y)
{
    return Tensor(hat::where(condition.to_torch(), x.to_torch(), y.to_torch()));
}

export Tensor mm(const Tensor& a, const Tensor& b)
{
    return Tensor(hat::mm(a.to_torch(), b.to_torch()));
}

export Tensor mv(const Tensor& mat, const Tensor& vec)
{
    return Tensor(hat::mv(mat.to_torch(), vec.to_torch()));
}


export bool allclose(const Tensor& a, const Tensor& b, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
{
    return hat::allclose(a.to_torch(), b.to_torch(), rtol, atol, equal_nan);
}

// Returns {unique_values [M], inverse_indices [N]}
// inverse_indices[i] gives the index into unique_values for input element i.
export std::pair<Tensor, Tensor> unique_with_inverse(const Tensor& t, bool sorted = true)
{
    auto [u, inv, _cnt] = hat::_unique2(t.to_torch(), sorted, true, false);
    return {Tensor(u), Tensor(inv)};
}

// fftn — wraps torch::fft::fftn; dims lifetime is caller's responsibility (ArrayRef is non-owning)
export Tensor fftn(
    const Tensor&         t,
    Opt<ArrayRef<i64>>    s    = nullopt,
    Opt<ArrayRef<i64>>    dims = nullopt,
    Opt<std::string_view> norm = nullopt
)
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

export Tensor ifftn(
    const Tensor&       t,
    Opt<ArrayRef<i64>>  s    = nullopt,
    Opt<ArrayRef<i64>>  dims = nullopt,
    Opt<std::string_view> norm = nullopt
)
{
    std::optional<hat::IntArrayRef> s_ref =
        s    ? std::optional<hat::IntArrayRef>(s->to_torch())    : std::nullopt;
    std::optional<hat::IntArrayRef> dim_ref =
        dims ? std::optional<hat::IntArrayRef>(dims->to_torch()) : std::nullopt;
    std::optional<std::string_view> norm_ref =
        norm ? std::optional<std::string_view>(*norm) : std::nullopt;
    return Tensor(htorch::fft::ifftn(t.to_torch(), s_ref, dim_ref, norm_ref));
}

export Tensor fftshift(const Tensor& t, Opt<ArrayRef<i64>> dims = nullopt)
{
    std::optional<hat::IntArrayRef> dim_ref =
        dims ? std::optional<hat::IntArrayRef>(dims->to_torch()) : std::nullopt;
    return Tensor(htorch::fft::fftshift(t.to_torch(), dim_ref));
}

export Tensor ifftshift(const Tensor& t, Opt<ArrayRef<i64>> dims = nullopt)
{
    std::optional<hat::IntArrayRef> dim_ref =
        dims ? std::optional<hat::IntArrayRef>(dims->to_torch()) : std::nullopt;
    return Tensor(htorch::fft::ifftshift(t.to_torch(), dim_ref));
}

export Tensor isnan(const Tensor& t)
{
    return Tensor(hat::isnan(t.to_torch()));
}

// kernel_size, stride, padding, dilation: length-1 (broadcast) or length-3 [D,H,W].
export Tensor max_pool3d(const Tensor& t,
                         ArrayRef<i64> kernel_size,
                         ArrayRef<i64> stride   = {},
                         ArrayRef<i64> padding  = {0},
                         ArrayRef<i64> dilation = {1},
                         bool ceil_mode = false)
{
    return Tensor(hat::max_pool3d(t.to_torch(),
        kernel_size.to_torch(),
        stride.to_torch(),
        padding.to_torch(),
        dilation.to_torch(),
        ceil_mode));
}

export Tensor compress_hermitian(const Tensor& t, i64 dim)
{
    // T[k] = conj(T[N-k]) (joint over all dims).
    // Keep T[..., 0..N/2] along dim; the upper half is redundant.
    i64 n = t.size(dim);
    return t.narrow(dim, 0, n / 2 + 1).contiguous();
}

export Tensor decompress_hermitian(const Tensor& t, i64 dim, i64 original_size)
{
    // Recover T[..., N/2+1..N-1] from stored T[..., 0..N/2].
    //
    // The Hermitian symmetry is JOINT: T[k1,...,kd] = conj(T[N1-k1,...,Nd-kd]).
    // So T[k1,...,k_{d-1}, k_d] for k_d > N/2
    //   = conj(T[N1-k1, ..., N_{d-1}-k_{d-1}, N-k_d])
    //
    // Concretely, to build the tail:
    //   1. Narrow dim to the interior 1..N/2-1 and flip dim  → gives the right N-k_d indices
    //   2. Circ-reverse every OTHER dim d2                   → gives the right Nd2-k_d2 indices
    //   3. Conjugate                                          → satisfies T[k]=conj(T[N-k])
    //
    // circ_reverse: [a, b, c, ..., z] → [a, z, ..., c, b]  (index 0 fixed, rest reversed)
    // This maps k → (N-k) % N correctly (0 stays at 0, k>0 maps to N-k).

    i64 tail_len = original_size / 2 - 1;    // N/2-1 values
    if (tail_len <= 0)
        return t.contiguous();

    int ndim = (int)t.ndimension();

    auto circ_rev = [](const Tensor& x, i64 d) -> Tensor {
        i64 n = x.size(d);
        if (n <= 1) return x;
        return cat({x.narrow(d, 0, 1),
                    x.narrow(d, 1, n - 1).flip({d})}, d);
    };

    // Step 1: flip along compressed dim
    Tensor tail = t.narrow(dim, 1, tail_len).flip({dim});

    // Step 2: circ-reverse all other dims
    for (int d = 0; d < ndim; ++d)
        if (d != (int)dim)
            tail = circ_rev(tail, (i64)d);

    // Step 3: conjugate (no-op for real tensors, correct for complex)
    tail = tail.conj().clone();

    return cat({t, tail}, dim).contiguous();
}

// Unbiased N-D convolution dispatched by spatial rank (1, 2, or 3).
// stride/padding/dilation must each have length == weight.dim() - 2.
export Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    ArrayRef<i64> stride,
    ArrayRef<i64> padding,
    ArrayRef<i64> dilation,
    i64 groups = 1)
{
    std::optional<hat::Tensor> no_bias{};
    i64 spatial_ndim = (i64)weight.sizes().size() - 2;
    auto in = input.to_torch();
    auto wt = weight.to_torch();
    auto st = stride.to_torch();
    auto pd = padding.to_torch();
    auto dl = dilation.to_torch();
    if (spatial_ndim == 1)
        return Tensor(hat::conv1d(in, wt, no_bias, st, pd, dl, groups));
    if (spatial_ndim == 2)
        return Tensor(hat::conv2d(in, wt, no_bias, st, pd, dl, groups));
    return Tensor(hat::conv3d(in, wt, no_bias, st, pd, dl, groups));
}

}

