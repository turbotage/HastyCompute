module;

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

module svt;

import <random>;
import hasty_util;

at::Tensor hasty::svt::extract_block(const at::Tensor& in, const hasty::Block<4>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split, bool transpose)
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

at::Tensor hasty::svt::extract_block(const at::Tensor& in, const Block<3>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split, bool transpose)
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

at::Tensor hasty::svt::svt_hard(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

void hasty::svt::svt_hard_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

at::Tensor hasty::svt::svt_soft(const at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

void hasty::svt::svt_soft_inplace(at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage)
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

void hasty::svt::insert_block(at::Tensor& in, const Block<4>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose)
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

void hasty::svt::insert_block(at::Tensor& in, const Block<3>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose)
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

namespace {
	void special_tensor_printer(std::stringstream& ss, const at::Tensor& tensor)
	{
		ss << "contig: " << tensor.is_contiguous() << " ";
		ss << "sizes: " << tensor.sizes() << " ";
		ss << "type" << tensor.dtype() << " ";
	}
}



hasty::svt::BlocksSVTBase::BlocksSVTBase(const std::vector<DeviceContext>& contexts)
	: _dcontexts(contexts), _nctxt(contexts.size()), _tpool(nullptr)
{
	_tpool = std::make_unique<ContextThreadPool<DeviceContext>>(_dcontexts);
}



hasty::svt::Random3DBlocksSVT::Random3DBlocksSVT(const std::vector<DeviceContext>& contexts)
	: BlocksSVTBase(contexts)
{
}

void hasty::svt::Random3DBlocksSVT::apply(at::Tensor in, int64_t nblocks, std::array<int64_t, 3> block_shape, double thresh, bool soft)
{
	c10::InferenceMode inference_guard;

	std::deque<std::future<void>> futures;

	std::array<int64_t, 3> bounds{ in.size(2) - block_shape[0], in.size(3) - block_shape[1], in.size(4) - block_shape[2] };

	std::function<void(DeviceContext&)> blockrunner;

	Block<3> block;
	for (int i = 0; i < nblocks; ++i) {
		Block<3>::randomize(block, bounds, block_shape);

		blockrunner = [this, &in, block, thresh, soft](DeviceContext& context) {  
			block_svt_step(context, in, block, thresh, soft);
		};

		futures.emplace_back(_tpool->enqueue(blockrunner));

		if (futures.size() > 32 * _dcontexts.size()) {
			torch_util::future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		torch_util::future_catcher(futures.front());
		futures.pop_front();
	}
}

void hasty::svt::Random3DBlocksSVT::block_svt_step(DeviceContext& dctxt, at::Tensor& in, const Block<3>& block, double thresh, bool soft)
{
	c10::InferenceMode inference_guard;

	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor cuda_block;
	if (_nctxt > 1)
	{
		at::Tensor tmp_block;
		{
			std::lock_guard<std::mutex> lock(_mutex);
			tmp_block = hasty::svt::extract_block(in, block, std::nullopt, 0, true).detach().clone();
		}
		cuda_block = tmp_block.to(dctxt.stream.device());
	}
	else {
		cuda_block = hasty::svt::extract_block(in, block, std::nullopt, 0, true).to(dctxt.stream.device());
	}

	//std::cout << cuda_block << std::endl;

	at::Tensor low_ranked = soft ?
		hasty::svt::svt_soft(cuda_block, (float)(thresh), std::nullopt) :
		hasty::svt::svt_hard(cuda_block, (int)(thresh + 0.5), std::nullopt);

	//std::cout << low_ranked << std::endl;


	if (_nctxt > 1)
	{
		at::Tensor back_block = low_ranked.cpu();
		{
			std::lock_guard<std::mutex> lock(_mutex);
			hasty::svt::insert_block(in, block, back_block, std::nullopt, true);
		}
	}
	else {
		at::Tensor back_block = low_ranked.cpu();
		hasty::svt::insert_block(in, block, back_block, std::nullopt, true);
	}
}




hasty::svt::OldNormal3DBlocksSVT::OldNormal3DBlocksSVT(const std::vector<DeviceContext>& contexts)
	: BlocksSVTBase(contexts)
{

}

void hasty::svt::OldNormal3DBlocksSVT::apply(at::Tensor in, const std::array<int64_t, 3>& block_strides, const std::array<int64_t, 3>& block_shape,
	int64_t block_iter, double thresh, bool soft)
{
	c10::InferenceMode inference_guard;

	std::deque<std::future<void>> futures;

	if (block_iter < 1)
		throw std::runtime_error("Can't perform Normal3DBlocksSVT with block_iter < 1");

	std::array<int64_t, 3> bounds{ in.size(2) - block_shape[0], in.size(3) - block_shape[1], in.size(4) - block_shape[2] };

	int Sx = in.size(2);
	int Sy = in.size(3);
	int Sz = in.size(4);

	int bx = Sx / block_strides[0];
	int by = Sy / block_strides[1];
	int bz = Sz / block_strides[2];

	std::function<void(DeviceContext&)> blockrunner;

	std::array<std::vector<int>, 3> shifts;

	std::default_random_engine generator;

	// Randomize shifts
	for (int d = 0; d < 3; ++d) {
		shifts[d].resize(block_iter);
		shifts[d][0] = 0;
		
		if (block_iter > 1) {
			std::uniform_int_distribution<int> distribution(0, block_shape[d]);
			for (int biter = 1; biter < block_iter; ++biter) {
				shifts[d][biter] = distribution(generator);
			}
		}
	}

	// Create blocks
	for (int iter = 0; iter < block_iter; ++iter) {

		int shiftx = shifts[0][iter];
		int shifty = shifts[1][iter];
		int shiftz = shifts[2][iter];

		for (int nx = 0; nx < bx; ++nx) {
			int sx = nx * block_strides[0] + shiftx;
			int ex = sx + block_shape[0];

			if (ex >= Sx)
				continue;

			for (int ny = 0; ny < by; ++ny) {
				int sy = ny * block_strides[1] + shifty;
				int ey = sy + block_shape[1];

				if (ey >= Sy)
					continue;

				for (int nz = 0; nz < bz; ++nz) {
					int sz = nz * block_strides[2] + shiftz;
					int ez = sz + block_shape[2];

					if (ez >= Sz)
						continue;

					Block<3> block;
					block.first_corner[0] = sx; block.second_corner[0] = ex;
					block.first_corner[1] = sy; block.second_corner[1] = ey;
					block.first_corner[2] = sz; block.second_corner[2] = ez;

					// Actual block push

					blockrunner = [this, &in, block, thresh, soft](DeviceContext& context) { 
						block_svt_step(context, in, block, thresh, soft);
					};

					futures.emplace_back(_tpool->enqueue(blockrunner));

					if (futures.size() > 32 * _dcontexts.size()) {
						torch_util::future_catcher(futures.front());
						futures.pop_front();
					}
				}
			}
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		torch_util::future_catcher(futures.front());
		futures.pop_front();
	}

}

void hasty::svt::OldNormal3DBlocksSVT::block_svt_step(DeviceContext& dctxt, at::Tensor& in, const Block<3>& block, double thresh, bool soft)
{
	c10::InferenceMode inference_guard;

	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor cuda_block;
	if (_nctxt > 1)
	{
		at::Tensor tmp_block;
		{
			std::lock_guard<std::mutex> lock(_mutex);
			tmp_block = hasty::svt::extract_block(in, block, std::nullopt, 0, true).detach().clone();
		}
		cuda_block = tmp_block.to(dctxt.stream.device());
	}
	else {
		cuda_block = hasty::svt::extract_block(in, block, std::nullopt, 0, true).to(dctxt.stream.device());
	}

	//std::cout << cuda_block << std::endl;

	at::Tensor low_ranked = soft ?
		hasty::svt::svt_soft(cuda_block, (float)(thresh), std::nullopt) :
		hasty::svt::svt_hard(cuda_block, (int)(thresh + 0.5), std::nullopt);

	if (_nctxt > 1)
	{
		at::Tensor back_block = low_ranked.cpu();
		{
			std::lock_guard<std::mutex> lock(_mutex);
			hasty::svt::insert_block(in, block, back_block, std::nullopt, true);
		}
	}
	else {
		at::Tensor back_block = low_ranked.cpu();
		hasty::svt::insert_block(in, block, back_block, std::nullopt, true);
	}
}

