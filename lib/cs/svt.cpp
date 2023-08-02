#include "svt.hpp"


at::Tensor hasty::extract_block(const at::Tensor& in, const Block<4>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split, bool transpose)
{
	using namespace at::indexing;
	std::array<TensorIndex, 5> idx = {
		"...",
		Slice(block.first_corner[0], block.second_corner[0]),
		Slice(block.first_corner[1], block.second_corner[1]),
		Slice(block.first_corner[2], block.second_corner[2]),
		Slice(block.first_corner[3], block.second_corner[3])
	};

	auto extracted_block = in.index(at::makeArrayRef(idx));

	if (perms.has_value()) {
		extracted_block = at::permute(extracted_block, at::makeArrayRef(*perms));
	}

	extracted_block = extracted_block.flatten(0, flatten_split).flatten(1);

	if (transpose) {
		extracted_block.transpose_(0, 1);
	}

	return extracted_block;
}

at::Tensor hasty::extract_block(const at::Tensor& in, const Block<3>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split, bool transpose)
{
	using namespace at::indexing;
	std::array<TensorIndex, 4> idx = {
		"...",
		Slice(block.first_corner[0], block.second_corner[0]),
		Slice(block.first_corner[1], block.second_corner[1]),
		Slice(block.first_corner[2], block.second_corner[2])
	};

	auto extracted_block = in.index(at::makeArrayRef(idx));

	if (perms.has_value()) {
		extracted_block = at::permute(extracted_block, at::makeArrayRef(*perms)).contiguous();
	}

	extracted_block = extracted_block.flatten(0, flatten_split).flatten(1).contiguous();

	if (transpose) {
		extracted_block.transpose_(0, 1).contiguous();
	}

	return extracted_block;
}

at::Tensor hasty::svt_hard(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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


	std::tie(U, S, Vh) = at::linalg_svd(in, false);

	//std::cout << S << std::endl;

	S.index_put_({ Slice(rank,None) }, 0.0f);

	//std::cout << S << std::endl;

	Vh.mul_(S.unsqueeze(1));


	return at::mm(U, Vh);
}

void hasty::svt_hard_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

	std::tie(U, S, Vh) = at::linalg_svd(in, false);

	S.index_put_({ Slice(rank,None) }, 0.0f);

	Vh.mul_(S.unsqueeze(1));

	at::mm_out(in, U, Vh);

}

at::Tensor hasty::svt_soft(const at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

	std::tie(U, S, Vh) = at::linalg_svd(in, false);

	//std::cout << S << std::endl;

	auto ldiff = at::abs(S) - lambda;
	S = at::sign(S) * at::abs(0.5f * (at::abs(ldiff) + ldiff));

	//std::cout << S << std::endl;

	Vh.mul_(S.unsqueeze(1));

	return at::mm(U, Vh);
}

void hasty::svt_soft_inplace(at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

	std::tie(U, S, Vh) = at::linalg_svd(in, false);

	auto ldiff = at::abs(S) - lambda;
	S = at::sign(S) * at::abs(0.5f * (at::abs(ldiff) + ldiff));

	Vh.mul_(S.unsqueeze(1));

	at::mm_out(in, U, Vh);
}

void hasty::insert_block(at::Tensor& in, const Block<4>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose)
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

	block_tensor = block_tensor.contiguous();

	if (transpose) {
		block_tensor = block_tensor.transpose(0, 1).contiguous();
	}

	if (perms.has_value()) {
		auto permuted_view = torch_util::apply_permutation(view_lens, *perms);
		block_tensor = at::permute(block_tensor.view(permuted_view), torch_util::argsort(*perms)).contiguous();
		in.index_put_(at::makeArrayRef(idx), block_tensor);
	}
	else {
		in.index_put_(at::makeArrayRef(idx), block_tensor.view(at::makeArrayRef(view_lens)));
	}
}

void hasty::insert_block(at::Tensor& in, const Block<3>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose)
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

	block_tensor = block_tensor.contiguous();

	if (transpose) {
		block_tensor = block_tensor.transpose(0, 1).contiguous();
	}

	if (perms.has_value()) {
		auto permuted_view = torch_util::apply_permutation(view_lens, *perms);
		block_tensor = at::permute(block_tensor.view(permuted_view), torch_util::argsort(*perms)).contiguous();
		in.index_put_(at::makeArrayRef(idx), block_tensor);
	}
	else {
		in.index_put_(at::makeArrayRef(idx), block_tensor.view(at::makeArrayRef(view_lens)));
	}
}

