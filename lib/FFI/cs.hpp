#pragma once

#include "ffi_defines.hpp"
#include "../torch_util.hpp"

namespace hasty {

	namespace ffi {

		LIB_EXPORT
		void batched_sense(at::Tensor input, const at::Tensor& smaps, const std::vector<at::Tensor>& coords);
		
		LIB_EXPORT
		void batched_sense(at::Tensor input, const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas);

		LIB_EXPORT 
		void llr(const at::Tensor& coords, at::Tensor& input, const at::Tensor& smaps, const at::Tensor& kdata, int64_t iter);

	}
}
