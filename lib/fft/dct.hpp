#pragma once

#include "../torch_util.hpp"

namespace hasty {

	at::Tensor dct_fft_impl(const at::Tensor v);

	at::Tensor idct_irfft_impl(const at::Tensor v);

	at::Tensor dct(at::Tensor x, const std::string& norm = "");

	at::Tensor idct(at::Tensor X, const std::string& norm="");

	at::Tensor dct_2d(at::Tensor x, const std::string& norm = "");

	at::Tensor idct_2d(at::Tensor X, const std::string& norm = "");

	at::Tensor dct_3d(at::Tensor x, const std::string& norm = "");

	at::Tensor idct_3d(at::Tensor X, const std::string& norm = "");


}


