module;

export module hasty_tensor_mod:external_math;

import std;
import hasty_util_mod;
import hasty_torch_wrapper;


import :tensor;

namespace hasty {

export Tensor floor(const Tensor& t)
{
    return Tensor(hat::floor(t.to_torch()));
}

export Tensor frac(const Tensor& t)
{
    return Tensor(hat::frac(t.to_torch()));
}

export Tensor log1p(const Tensor& t)
{
    return Tensor(hat::log1p(t.to_torch()));
}

export Tensor pow(const Tensor& base, double exponent)
{
    return Tensor(hat::pow(base.to_torch(), exponent));
}

export Tensor exp(const Tensor& t)
{
    return Tensor(hat::exp(t.to_torch()));
}

export Tensor expm1(const Tensor& t)
{
    return Tensor(hat::expm1(t.to_torch()));
}

export Tensor nanmean(const Tensor& t)
{
    return Tensor(hat::nanmean(t.to_torch()));
}

export Tensor nanmean(const Tensor& t, i64 dim, bool keepdim = false)
{
    return Tensor(hat::nanmean(t.to_torch(), dim, keepdim));
}

export Tensor argsort(const Tensor& t, i64 dim = -1, bool descending = false)
{
    return Tensor(hat::argsort(t.to_torch(), dim, descending));
}


}