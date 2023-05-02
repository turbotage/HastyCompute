#pragma once

#include <torch/extension.h>
#include <torch/library.h>

#include "../lib/FFI/ffi.hpp"

using namespace at;


///Adds one to each element of a tensor
at::Tensor add_one(const at::Tensor& input) {
    auto output = torch::zeros_like(input);

    output = input + 1;

    return output;
}

at::Tensor nufft1(const at::Tensor& coords, const at::Tensor& input, const std::vector<int64_t>& nmodes)
{
    return hasty::ffi::nufft1(coords, input, nmodes);
}

at::Tensor nufft2(const at::Tensor& coords, const at::Tensor& input)
{
    return hasty::ffi::nufft2(coords, input);
}

at::Tensor nufft2to1(const at::Tensor& coords, const at::Tensor& input)
{
    return hasty::ffi::nufft2to1(coords, input);
}

void llr(const at::Tensor& coords, at::Tensor& input, const at::Tensor& smaps, const at::Tensor& kdata, int64_t iter)
{
    hasty::ffi::llr(coords, input, smaps, kdata, iter);
}