#pragma once

#include <torch/torch.h>

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <stdexcept>;
import <array>;
import <functional>;
import <optional>;
#endif


namespace hasty {
	namespace cuda {

		// tensors outer vector are frames inner are encodes
		at::Tensor extract_block(const std::vector<std::vector<std::reference_wrapper<const at::Tensor>>>& tensors, 
			const std::pair<std::vector<int64_t>, std::vector<int64_t>>& block);

		at::Tensor svt(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor,at::Tensor,at::Tensor>> storage);

		void svt_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		void insert_block(const std::vector<std::vector<std::reference_wrapper<at::Tensor>>>& tensors, 
			const std::pair<std::vector<int64_t>, std::vector<int64_t>>& block, const at::Tensor& block_tensor);
		
	}
}