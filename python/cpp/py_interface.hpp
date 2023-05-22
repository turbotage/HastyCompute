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

void batched_sense_toep(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps, const std::vector<at::Tensor>& coords)
{
    std::vector<std::vector<int32_t>> coilss;
    coilss.reserve(coils.size());
    std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
        std::vector<int32_t> ret_coil(coil.size());
        std::copy(coil.begin(), coil.end(), ret_coil.begin());
        return ret_coil;
        });

    hasty::ffi::batched_sense(input, coilss, smaps, coords);
}

void batched_sense(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps, 
    const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas)
{
    std::vector<std::vector<int32_t>> coilss;
    coilss.reserve(coils.size());
    std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
        std::vector<int32_t> ret_coil(coil.size());
        std::copy(coil.begin(), coil.end(), ret_coil.begin());
        return ret_coil;
        });

    hasty::ffi::batched_sense(input, coilss, smaps, coords, kdatas);
}

void batched_sense_weighted(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
    const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& kdatas)
{
    std::vector<std::vector<int32_t>> coilss;
    coilss.reserve(coils.size());
    std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
        std::vector<int32_t> ret_coil(coil.size());
        std::copy(coil.begin(), coil.end(), ret_coil.begin());
        return ret_coil;
        });

    hasty::ffi::batched_sense(input, coilss, smaps, coords, weights, kdatas);
}

void random_blocks_svt(at::Tensor& input, int64_t nblocks, int64_t block_size, int64_t rank)
{
    hasty::ffi::random_blocks_svt(input, nblocks, block_size, rank);
}
