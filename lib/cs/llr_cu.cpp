#include "llr_cu.hpp"


at::Tensor hasty::cuda::block_svt(const vec<refw<const at::Tensor>>& tensors, std::pair<vec<i64>,vec<i64>> block)
{
	using namespace at::indexing;

	std::vector<TensorIndex> indices;

	auto lower_bound = block.first;
	auto upper_bound = block.second;

	for (int i = 0; i < lower_bound.size(); ++i) {
		indices.emplace_back(Slice(lower_bound[i], upper_bound[i]));
	}
	auto block_indices = at::makeArrayRef(indices);

	std::vector<at::Tensor> stackable_blocks;
	for (int i = 0; i < tensors.size(); ++i) {
		stackable_blocks.emplace_back(tensors[i].get().index(block_indices));
	}

	

	at::Tensor final_block;

	at::Tensor U;
	at::Tensor S;
	at::Tensor Vh;

	std::tie(U, S, Vh) = at::linalg_svd(final_block, false, c10::nullopt);





	Vh.mul_(S.unsqueeze(1));

	at::mm_out(final_block, U, Vh);


}

