module;

#include <torch/torch.h>

export module dct;

import torch_util;

namespace hasty {

	namespace fft {

		export at::Tensor dct_fft_impl(const at::Tensor v);

		export at::Tensor idct_irfft_impl(const at::Tensor v);

		export at::Tensor dct(at::Tensor x, const std::string& norm = "");

		export at::Tensor idct(at::Tensor X, const std::string& norm = "");

		export at::Tensor dct_2d(at::Tensor x, const std::string& norm = "");

		export at::Tensor idct_2d(at::Tensor X, const std::string& norm = "");

		export at::Tensor dct_3d(at::Tensor x, const std::string& norm = "");

		export at::Tensor idct_3d(at::Tensor X, const std::string& norm = "");

	}

}

