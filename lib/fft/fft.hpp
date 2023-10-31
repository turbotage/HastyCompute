#pragma once

#include "../torch_util.hpp"

namespace hasty {
	namespace fft {

		at::Tensor fftc(const at::Tensor& input, const at::OptionalArrayRef<int64_t>& oshape = at::nullopt,
			const at::OptionalArrayRef<int64_t>& axes = at::nullopt, const at::optional<at::string_view>& norm = "ortho");

		at::Tensor ifftc(const at::Tensor& input, const at::OptionalArrayRef<int64_t>& oshape = at::nullopt,
			const at::OptionalArrayRef<int64_t>& axes = at::nullopt, const at::optional<at::string_view>& norm = "ortho");

	}
}




