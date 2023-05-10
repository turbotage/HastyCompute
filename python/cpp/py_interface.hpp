#pragma once

#include <torch/extension.h>
#include <torch/library.h>

#include "../../lib/FFI/ffi.hpp"

using namespace at;


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

void batched_sense_toep(at::Tensor input, const at::Tensor& smaps, const std::vector<at::Tensor>& coords)
{
    hasty::ffi::batched_sense(input, smaps, coords);
}

void batched_sense(at::Tensor input, const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas)
{
    hasty::ffi::batched_sense(input, smaps, coords, kdatas);
}

void llr(const at::Tensor& coords, at::Tensor& input, const at::Tensor& smaps, const at::Tensor& kdata, int64_t iter)
{
    hasty::ffi::llr(coords, input, smaps, kdata, iter);
}