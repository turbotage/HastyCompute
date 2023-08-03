#include "llr.hpp"

#include <random>
#include <c10/cuda/CUDAGuard.h>

#include "svt.hpp"

import hasty_util;

namespace {
	void special_tensor_printer(std::stringstream& ss, const at::Tensor& tensor)
	{
		ss << "contig: " << tensor.is_contiguous() << " ";
		ss << "sizes: " << tensor.sizes() << " ";
		ss << "type" << tensor.dtype() << " ";
	}
}


hasty::RandomBlocksSVT::RandomBlocksSVT(std::vector<DeviceContext>& contexts,
	at::Tensor& image, int32_t nblocks, int32_t block_size, double thresh, bool soft)
	: _image(image), _nctxt(contexts.size())
{
	c10::InferenceMode inference_guard;

	std::deque<std::future<void>> futures;
	ContextThreadPool<DeviceContext> tpool(contexts);

	std::array<int64_t, 3> lens{ block_size,block_size,block_size };
	std::array<int64_t, 3> bounds{ image.size(2) - lens[0], image.size(3) - lens[1], image.size(4) - lens[2] };

	std::vector<Block<3>> blocks(nblocks);
	try {
		for (int i = 0; i < nblocks; ++i) {
			Block<3>::randomize(blocks[i], bounds, lens);
		}
	}
	catch (...) {
		std::cerr << "Oh No!";
	}

	std::function<void(DeviceContext&)> blockrunner;

	auto base_blockrunner = [this, thresh, soft](const Block<3>& block, DeviceContext& context) {
		block_svt_step(context, block, thresh, soft);
	};

	for (int i = 0; i < blocks.size(); ++i) {

		const auto& block = blocks[i];

		 blockrunner = [&block, &base_blockrunner](DeviceContext& context) 
			{ base_blockrunner(block, context); };

		futures.emplace_back(tpool.enqueue(blockrunner));

		if (futures.size() > 32 * contexts.size()) {
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

void hasty::RandomBlocksSVT::block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft)
{
	/*
	static int iter = 0;
	if (iter % 100 == 0)
		std::cout << "block: " << iter << std::endl;
	iter += 1;
	*/
	
	
	c10::InferenceMode inference_guard;

	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor cuda_block;
	if (_nctxt > 1)
	{
		at::Tensor tmp_block;
		{
			std::lock_guard<std::mutex> lock(_mutex);
			tmp_block = hasty::extract_block(_image, block, std::nullopt, 0, true).detach().clone();
		}
		cuda_block = tmp_block.to(dctxt.stream.device());
	}
	else {
		cuda_block = hasty::extract_block(_image, block, std::nullopt, 0, true).to(dctxt.stream.device());
	}

	//std::cout << cuda_block << std::endl;

	at::Tensor low_ranked = soft ?
		hasty::svt_soft(cuda_block, (float)(thresh), std::nullopt) :
		hasty::svt_hard(cuda_block, (int)(thresh + 0.5), std::nullopt);

	//std::cout << low_ranked << std::endl;


	if (_nctxt > 1)
	{
		at::Tensor back_block = low_ranked.cpu();
		{
			std::lock_guard<std::mutex> lock(_mutex);
			hasty::insert_block(_image, block, back_block, std::nullopt, true);
		}
	}
	else {
		at::Tensor back_block = low_ranked.cpu();
		hasty::insert_block(_image, block, back_block, std::nullopt, true);
	}
}




hasty::NormalBlocksSVT::NormalBlocksSVT(std::vector<DeviceContext>& contexts, at::Tensor& image, std::array<int64_t, 3> block_strides, 
	std::array<int64_t, 3> block_shape, int block_iter, double thresh, bool soft)
	: _image(image), _nctxt(contexts.size())
{
	c10::InferenceMode inference_guard;

	std::deque<std::future<void>> futures;
	ContextThreadPool<DeviceContext> tpool(contexts);

	std::array<int64_t, 3> bounds{ image.size(2) - block_shape[0], image.size(3) - block_shape[1], image.size(4) - block_shape[2] };

	int Sx = image.size(2);
	int Sy = image.size(3);
	int Sz = image.size(4);

	int bx = Sx / block_strides[0];
	int by = Sy / block_strides[1];
	int bz = Sz / block_strides[2];

	std::array<std::vector<int>, 3> shifts;
	std::vector<Block<3>> blocks;
	blocks.reserve(block_iter * bx * by * bz);
	try {

		std::default_random_engine generator;

		// Randomize shifts
		for (int d = 0; d < 3; ++d) {
			shifts[d].resize(block_iter);
			std::uniform_int_distribution<int> distribution(0, block_shape[d]);
			for (int biter = 0; biter < block_iter; ++biter) {
				shifts[d][biter] = distribution(generator);
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

						blocks.push_back(block);
					}
				}
			}
		}


	}
	catch (...) {
		std::cerr << "Oh No!";
	}

	std::function<void(DeviceContext&)> blockrunner;

	auto base_blockrunner = [this, thresh, soft](const Block<3>& block, DeviceContext& context) {
		block_svt_step(context, block, thresh, soft);
	};

	for (int i = 0; i < blocks.size(); ++i) {

		const auto& block = blocks[i];

		blockrunner = [&block, &base_blockrunner](DeviceContext& context)
		{ base_blockrunner(block, context); };

		futures.emplace_back(tpool.enqueue(blockrunner));

		if (futures.size() > 32 * contexts.size()) {
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

void hasty::NormalBlocksSVT::block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft)
{
	/*
	static int iter = 0;
	if (iter % 100 == 0)
		std::cout << "block: " << iter << std::endl;
	iter += 1;
	*/


	c10::InferenceMode inference_guard;

	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor cuda_block;
	if (_nctxt > 1)
	{
		at::Tensor tmp_block;
		{
			std::lock_guard<std::mutex> lock(_mutex);
			tmp_block = hasty::extract_block(_image, block, std::nullopt, 0, true).detach().clone();
		}
		cuda_block = tmp_block.to(dctxt.stream.device());
	}
	else {
		cuda_block = hasty::extract_block(_image, block, std::nullopt, 0, true).to(dctxt.stream.device());
	}

	//std::cout << cuda_block << std::endl;

	at::Tensor low_ranked = soft ?
		hasty::svt_soft(cuda_block, (float)(thresh), std::nullopt) :
		hasty::svt_hard(cuda_block, (int)(thresh + 0.5), std::nullopt);

	//std::cout << low_ranked << std::endl;


	if (_nctxt > 1)
	{
		at::Tensor back_block = low_ranked.cpu();
		{
			std::lock_guard<std::mutex> lock(_mutex);
			hasty::insert_block(_image, block, back_block, std::nullopt, true);
		}
	}
	else {
		at::Tensor back_block = low_ranked.cpu();
		hasty::insert_block(_image, block, back_block, std::nullopt, true);
	}
}

