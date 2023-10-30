#pragma once

#include "../torch_util.hpp"

namespace hasty {
	namespace fft {

		at::Tensor fftc(const at::Tensor& input, const at::OptionalArrayRef<int64_t>& oshape,
			const at::OptionalArrayRef<int64_t>& axes, const at::optional<at::string_view>& norm);

		at::Tensor ifftc(const at::Tensor& input, const at::OptionalArrayRef<int64_t>& oshape,
			const at::OptionalArrayRef<int64_t>& axes, const at::optional<at::string_view>& norm);

	}
}




