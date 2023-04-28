#include "llr_cu.hpp"

#include <random>


at::Tensor hasty::cuda::extract_block(const at::Tensor& in, const Block<4>& block)
{

}

at::Tensor hasty::cuda::extract_block(const std::vector<std::vector<std::reference_wrapper<const at::Tensor>>>& tensors,
	const std::pair<std::vector<int64_t>, std::vector<int64_t>>& block)
{
	using namespace at::indexing;

	int num_frame = tensors.size();
	int num_enc = tensors[0].size();

	std::vector<TensorIndex> indices;

	auto lower_bound = block.first;
	auto upper_bound = block.second;

	for (int i = 0; i < lower_bound.size(); ++i) {
		indices.emplace_back(Slice(lower_bound[i], upper_bound[i]));
	}
	auto block_indices = at::makeArrayRef(indices);

	std::vector<at::Tensor> stackable_blocks(tensors.size());
	std::vector<at::Tensor> catable_blocks(tensors[0].size());
	for (int nframe = 0; nframe < num_frame; ++nframe) {
		auto& enc_vec = tensors[nframe];
		for (int nenc = 0; nenc < num_enc; ++nenc) {
			catable_blocks[nenc] = enc_vec[nenc].get().index(block_indices).flatten();
		}
		stackable_blocks[nframe] = at::cat(catable_blocks);
	}
	return at::stack(stackable_blocks, 1);
}

at::Tensor hasty::cuda::svt(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
{
	using namespace at::indexing;

	at::Tensor U;
	at::Tensor S;
	at::Tensor Vh;

	if (storage.has_value()) {
		auto& sval = storage.value();
		U = std::get<0>(sval);
		S = std::get<1>(sval);
		Vh = std::get<2>(sval);
	}

	std::tie(U, S, Vh) = at::linalg_svd(in, false, "gesvd");

	S.index_put_({ Slice(rank,None) }, 0.0f);

	Vh.mul_(S.unsqueeze(1));

	return at::mm(U, Vh);
}

void hasty::cuda::svt_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
{
	using namespace at::indexing;

	at::Tensor U;
	at::Tensor S;
	at::Tensor Vh;

	if (storage.has_value()) {
		auto& sval = storage.value();
		U = std::get<0>(sval);
		S = std::get<1>(sval);
		Vh = std::get<2>(sval);
	}

	std::tie(U, S, Vh) = at::linalg_svd(in, false, "gesvd");

	S.index_put_({ Slice(rank,None) }, 0.0f);

	Vh.mul_(S.unsqueeze(1));

	at::mm_out(in, U, Vh);
}

void hasty::cuda::insert_block(const std::vector<std::vector<std::reference_wrapper<at::Tensor>>>& tensors,
	const std::pair<std::vector<int64_t>, std::vector<int64_t>>& block, const at::Tensor& block_tensor)
{
	using namespace at::indexing;

	int num_frame = tensors.size();
	int num_enc = tensors[0].size();

	std::vector<TensorIndex> indices;

	auto lower_bound = block.first;
	auto upper_bound = block.second;

	int nelem = 1;
	std::vector<int64_t> block_lens(lower_bound.size());
	for (int i = 0; i < lower_bound.size(); ++i) {
		int len = upper_bound[i] - lower_bound[i];
		block_lens[i] = len;
		nelem *= len;
		indices.emplace_back(Slice(lower_bound[i], upper_bound[i]));
	}
	auto block_indices = at::makeArrayRef(indices);

	for (int nframe = 0; nframe < num_frame; ++nframe) {
		int start = 0;
		for (int nenc = 0; nenc < num_enc; ++nenc) {
			auto& into = tensors[nframe][nenc].get();
			auto extracted = block_tensor.index({ Slice(start, start + nelem), nframe });
			into.index_put_(block_indices, extracted.view(block_lens));
			start += nelem;
		}
	}

}




hasty::cuda::LLRecon_4D_Encodes::LLRecon_4D_Encodes(
	at::Tensor& image,
	const at::Tensor& coords,
	const at::Tensor& smaps,
	const at::Tensor& kdata)
	:
	_image(image),
	_coords(coords),
	_smaps(smaps),
	_kdata(kdata)
{
	
}


void hasty::cuda::LLRecon_4D_Encodes::run(int iter)
{


}