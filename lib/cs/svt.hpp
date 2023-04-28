#pragma once

#include "block.hpp"
#include "../torch_util.hpp"

namespace hasty {

	at::Tensor extract_block(const at::Tensor& tensor, const Block<4>& block);

	at::Tensor extract_block(const at::Tensor& tensor, const Block<3>& block);

	at::Tensor svt(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

	void svt_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

	void insert_block(at::Tensor& in, const Block<4>& block, const at::Tensor& block_tensor);

	void insert_block(at::Tensor& in, const Block<3>& block, const at::Tensor& block_tensor);

}