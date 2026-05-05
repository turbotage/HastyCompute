module;

export module hasty_tensor_mod:external_math;

import std;
import hasty_util_mod;
import hasty_torch_wrapper;


import :tensor;

namespace hasty {

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


}