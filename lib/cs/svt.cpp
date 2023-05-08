#include "svt.hpp"



at::Tensor hasty::extract_block(const at::Tensor& in, const Block<4>& block)
{
	using namespace at::indexing;
	std::array<TensorIndex, 5> idx = {
		"...",
		Slice(block.first_corner[0], block.second_corner[0]),
		Slice(block.first_corner[1], block.second_corner[1]),
		Slice(block.first_corner[2], block.second_corner[2]),
		Slice(block.first_corner[3], block.second_corner[3])
	};

	// Flatten the all but the first dimension
	return in.index(at::makeArrayRef(idx)).flatten(1);
}

at::Tensor hasty::extract_block(const at::Tensor& in, const Block<3>& block)
{
	using namespace at::indexing;
	std::array<TensorIndex, 4> idx = {
		"...",
		Slice(block.first_corner[0], block.second_corner[0]),
		Slice(block.first_corner[1], block.second_corner[1]),
		Slice(block.first_corner[2], block.second_corner[2])
	};

	// Flatten the all but the first dimension
	return in.index(at::makeArrayRef(idx)).flatten(1);
}

at::Tensor hasty::svt(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

void hasty::svt_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

void hasty::insert_block(at::Tensor& in, const Block<4>& block, const at::Tensor& block_tensor)
{
	using namespace at::indexing;

	constexpr size_t N = 4;

	std::array<TensorIndex, N+1> idx = {
		"...",
		Slice(block.first_corner[0], block.second_corner[0]),
		Slice(block.first_corner[1], block.second_corner[1]),
		Slice(block.first_corner[2], block.second_corner[2]),
		Slice(block.first_corner[3], block.second_corner[3])
	};

	std::vector<int64_t> view_lens = in.sizes().vec();
	int i = 0;
	for (auto it = view_lens.rbegin(); it != view_lens.rbegin() + N; ++it) {
		*it = block.first_corner[N - 1 - i];
		++i;
	}

	in.index_put_(at::makeArrayRef(idx), block_tensor.view(at::makeArrayRef(view_lens)));
}

void hasty::insert_block(at::Tensor& in, const Block<3>& block, const at::Tensor& block_tensor)
{
	using namespace at::indexing;

	constexpr size_t N = 3;

	std::array<TensorIndex, N + 1> idx = {
		"...",
		Slice(block.first_corner[0], block.second_corner[0]),
		Slice(block.first_corner[1], block.second_corner[1]),
		Slice(block.first_corner[2], block.second_corner[2])
	};

	std::vector<int64_t> view_lens = in.sizes().vec();
	int i = 0;
	for (auto it = view_lens.rbegin(); it != view_lens.rbegin() + N; ++it) {
		int block_idx = N - 1 - i;
		*it = (block.second_corner[block_idx] - block.first_corner[block_idx]);
		++i;
	}

	in.index_put_(at::makeArrayRef(idx), block_tensor.view(at::makeArrayRef(view_lens)));
}

