#pragma once

#include "block.hpp"
#include "../torch_util.hpp"

namespace hasty {

	at::Tensor extract_block(const at::Tensor& tensor, const Block<4>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split = 1, bool transpose = false);

	at::Tensor extract_block(const at::Tensor& tensor, const Block<3>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split = 1, bool transpose = false);

	at::Tensor svt_hard(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

	void svt_hard_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

	at::Tensor svt_soft(const at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

	void svt_soft_inplace(at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

	void insert_block(at::Tensor& in, const Block<4>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose = false);

	void insert_block(at::Tensor& in, const Block<3>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose = false);

}

