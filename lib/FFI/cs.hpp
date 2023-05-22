#pragma once

#include "ffi_defines.hpp"
#include "../torch_util.hpp"

namespace hasty {

	namespace ffi {

		LIB_EXPORT
		void batched_sense(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps, const std::vector<at::Tensor>& coords);

		LIB_EXPORT
		void batched_sense(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps, 
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas);

		LIB_EXPORT
		void batched_sense(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps, 
			const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, const std::vector<at::Tensor>& kdatas);

		LIB_EXPORT
		void random_blocks_svt(at::Tensor& input, int32_t nblocks, int32_t block_size, int32_t rank);

	}
}
