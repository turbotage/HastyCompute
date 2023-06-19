#pragma once

#include "py_util.hpp"

namespace nufft {

    at::Tensor nufft1(const at::Tensor& coords, const at::Tensor& input, const std::vector<int64_t>& nmodes)
    {
        return hasty::ffi::nufft1(coords, input, nmodes);
    }

    at::Tensor nufft2(const at::Tensor& coords, const at::Tensor& input)
    {
        return hasty::ffi::nufft2(coords, input);
    }

    at::Tensor nufft21(const at::Tensor& coords, const at::Tensor& input)
    {
        return hasty::ffi::nufft21(coords, input);
    }

    at::Tensor nufft12(const at::Tensor& coords, const at::Tensor& input, const std::vector<int64_t>& nmodes)
    {
        return hasty::ffi::nufft12(coords, input, nmodes);
    }

}


namespace bs {

    void batched_sense(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
        const std::vector<at::Tensor>& coords, const at::optional<std::vector<c10::Stream>>& streams)
    {
        std::vector<std::vector<int32_t>> coilss;
        coilss.reserve(coils.size());
        std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
            std::vector<int32_t> ret_coil(coil.size());
            std::copy(coil.begin(), coil.end(), ret_coil.begin());
            return ret_coil;
            });

        hasty::ffi::batched_sense(input, coilss, smaps, coords, hasty::ffi::get_streams(streams));
    }

    void batched_sense_weighted(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
        const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const at::optional<std::vector<c10::Stream>>& streams)
    {
        std::vector<std::vector<int32_t>> coilss;
        coilss.reserve(coils.size());
        std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
            std::vector<int32_t> ret_coil(coil.size());
            std::copy(coil.begin(), coil.end(), ret_coil.begin());
            return ret_coil;
            });

        hasty::ffi::batched_sense_weighted(input, coilss, smaps, coords, weights, hasty::ffi::get_streams(streams));
    }

    void batched_sense_kdata(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
        const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas, const at::optional<std::vector<c10::Stream>>& streams)
    {
        std::vector<std::vector<int32_t>> coilss;
        coilss.reserve(coils.size());
        std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
            std::vector<int32_t> ret_coil(coil.size());
            std::copy(coil.begin(), coil.end(), ret_coil.begin());
            return ret_coil;
            });

        hasty::ffi::batched_sense_kdata(input, coilss, smaps, coords, kdatas, hasty::ffi::get_streams(streams));
    }

    void batched_sense_weighted_kdata(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
        const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
        const std::vector<at::Tensor>& kdatas, const at::optional<std::vector<c10::Stream>>& streams)
    {
        std::vector<std::vector<int32_t>> coilss;
        coilss.reserve(coils.size());
        std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
            std::vector<int32_t> ret_coil(coil.size());
            std::copy(coil.begin(), coil.end(), ret_coil.begin());
            return ret_coil;
            });

        hasty::ffi::batched_sense_weighted_kdata(input, coilss, smaps, coords, weights, kdatas, hasty::ffi::get_streams(streams));
    }



    void batched_sense_toeplitz(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
        const std::vector<at::Tensor>& coords, const at::optional<std::vector<c10::Stream>>& streams)
    {
        std::vector<std::vector<int32_t>> coilss;
        coilss.reserve(coils.size());
        std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
            std::vector<int32_t> ret_coil(coil.size());
            std::copy(coil.begin(), coil.end(), ret_coil.begin());
            return ret_coil;
            });

        hasty::ffi::batched_sense_toeplitz(input, coilss, smaps, coords, hasty::ffi::get_streams(streams));
    }

    void batched_sense_toeplitz_diagonals(at::Tensor& input, const std::vector<std::vector<int64_t>>& coils, const at::Tensor& smaps,
        const at::Tensor& diagonals, const at::optional<std::vector<c10::Stream>>& streams)
    {
        std::vector<std::vector<int32_t>> coilss;
        coilss.reserve(coils.size());
        std::transform(coils.begin(), coils.end(), std::back_inserter(coilss), [](auto& coil) {
            std::vector<int32_t> ret_coil(coil.size());
            std::copy(coil.begin(), coil.end(), ret_coil.begin());
            return ret_coil;
            });

        hasty::ffi::batched_sense_toeplitz_diagonals(input, coilss, smaps, diagonals, hasty::ffi::get_streams(streams));
    }

}

namespace dummy {

    void dummy(at::Tensor& tensor) {
        std::cout << tensor << std::endl;
    }

}


using namespace at;

TORCH_LIBRARY(HastySense, m) {
    
    m.def("nufft1", nufft::nufft1);
    m.def("nufft2", nufft::nufft2);
    m.def("nufft21", nufft::nufft21);
    m.def("nufft12", nufft::nufft12);
    
    m.def("batched_sense", bs::batched_sense);
    m.def("batched_sense_weighted", bs::batched_sense_weighted);
    m.def("batched_sense_kdata", bs::batched_sense_kdata);
    m.def("batched_sense_weighted_kdata", bs::batched_sense_weighted_kdata);
    
    m.def("batched_sense_toeplitz", bs::batched_sense_toeplitz);
    m.def("batched_sense_toeplitz_diagonals", bs::batched_sense_toeplitz_diagonals);

}






