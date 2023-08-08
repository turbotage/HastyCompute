#pragma once

#include "py_util.hpp"

namespace hasty {
	namespace dummy {

		void LIB_EXPORT dummy(at::TensorList tensorlist);

		void LIB_EXPORT stream_dummy(const at::optional<at::ArrayRef<at::Stream>>& streams, const torch::Tensor& in);
	}
}