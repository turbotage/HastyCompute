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

}

