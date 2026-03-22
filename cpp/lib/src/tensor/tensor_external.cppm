module;

export module tensor_mod:external;

import std;
import util_mod;
import torch_wrapper;

import :tensor;


namespace hasty {

    export Tensor empty(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::empty(sizes.to_torch(), options.to_torch()));
    }

    export Tensor empty_like(const Tensor& other)
    {
        return Tensor(hat::empty_like(other._base));
    }

    export Tensor zeros(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::zeros(sizes.to_torch(), options.to_torch()));
    }

    export Tensor zeros_like(const Tensor& other)
    {
        return Tensor(hat::zeros_like(other._base));
    }

    export Tensor ones(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::ones(sizes.to_torch(), options.to_torch()));
    }

    export Tensor ones_like(const Tensor& other)
    {
        return Tensor(hat::ones_like(other._base));
    }

    export Tensor rand(ArrayRef<i64> sizes, TensorOptions options)
    {
        return Tensor(hat::rand(sizes.to_torch(), options.to_torch()));
    }

    export Tensor rand_like(const Tensor& other)
    {
        return Tensor(hat::rand_like(other._base));
    }

}

