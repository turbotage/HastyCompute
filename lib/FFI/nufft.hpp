#pragma once

#include "ffi_defines.hpp"
#include "../torch_util.hpp"

namespace hasty {
	namespace ffi {

		LIB_EXPORT at::Tensor nufft1(const at::Tensor& coords, const at::Tensor& input, const std::vector<int64_t>& nmodes);

		LIB_EXPORT at::Tensor nufft2(const at::Tensor& coords, const at::Tensor& input);

		LIB_EXPORT at::Tensor nufft2to1(const at::Tensor& coords, const at::Tensor& input);

	}
}


