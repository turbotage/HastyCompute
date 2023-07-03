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

std::string hasty::LLR_4DEncodes::DeviceContext::str()
{
	std::stringstream ss;
	ss << "device_index: " << device.index() << "\n";
	ss << "image:      "; special_tensor_printer(ss, image); ss << "\n";
	ss << "kdata:      "; special_tensor_printer(ss, kdata); ss << "\n";
	ss << "coords:     "; special_tensor_printer(ss, coords); ss << "\n";
	ss << "out:        "; special_tensor_printer(ss, out); ss << "\n";
	ss << "in_storage: "; special_tensor_printer(ss, in_storage); ss << "\n";
	ss << "smaps:      "; special_tensor_printer(ss, smaps); ss << "\n";
	return ss.str();
}



hasty::LLR_4DEncodes::LLR_4DEncodes(
	const Options& options,
	at::Tensor& image,
	const TensorVecVec& coords,
	const at::Tensor& smaps,
	const TensorVecVec& kdata)
	:
	_options(options),
	_image(image),
	_coords(coords),
	_smaps(smaps),
	_kdata(kdata),
	_nframe(image.size(0)),
	_nmodes(4)
{
	construct();
}

hasty::LLR_4DEncodes::LLR_4DEncodes(
	const Options& options,
	at::Tensor& image,
	const TensorVecVec& coords,
	const at::Tensor& smaps,
	const TensorVecVec& kdata,
	const TensorVecVec& weights)
	:
	_options(options),
	_image(image),
	_coords(coords),
	_smaps(smaps),
	_kdata(kdata),
	_weights(weights),
	_nframe(image.size(0)),
	_nmodes(4)
{
	construct();
}

hasty::LLR_4DEncodes::LLR_4DEncodes(
	const Options& options,
	at::Tensor& image,
	TensorVecVec&& coords,
	const at::Tensor& smaps,
	TensorVecVec&& kdata)
	:
	_options(options),
	_image(image),
	_coords(std::move(coords)),
	_smaps(smaps),
	_kdata(std::move(kdata)),
	_nframe(image.size(0)),
	_nmodes(4)
{
	construct();
}

hasty::LLR_4DEncodes::LLR_4DEncodes(
	const Options& options,
	at::Tensor& image,
	TensorVecVec&& coords,
	const at::Tensor& smaps,
	TensorVecVec&& kdata,
	TensorVecVec&& weights)
	:
	_options(options),
	_image(image),
	_coords(std::move(coords)),
	_smaps(smaps),
	_kdata(std::move(kdata)),
	_weights(std::move(weights)),
	_nframe(image.size(0)),
	_nmodes(4)
{
	construct();
}

void hasty::LLR_4DEncodes::construct()
{
	c10::InferenceMode inference_guard;

	_nmodes[0] = 1; _nmodes[1] = _image.size(2); _nmodes[2] = _image.size(3); _nmodes[3] = _image.size(4);

	auto base_options1 = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto base_options2 = c10::TensorOptions().dtype(c10::ScalarType::Float);

	for (auto& devicepair : _options.devices) {
		auto& device = devicepair.first;
		auto smaps = _smaps.to(device);
		for (auto& stream : devicepair.second) {
			auto& dcontext = _dcontexts.emplace_back(device, stream);

			c10::cuda::CUDAStreamGuard device_guard(stream);

			auto options1 = base_options1.device(device);
			auto options2 = base_options2.device(device);

			dcontext.image = at::empty(at::makeArrayRef(_nmodes), options1);
			dcontext.out = at::empty_like(dcontext.image);
			dcontext.in_storage = at::empty_like(dcontext.image);

			dcontext.smaps = smaps;
		}
	}

	_tpool = std::make_unique<ContextThreadPool<DeviceContext>>(_dcontexts);
}




void hasty::LLR_4DEncodes::step_llr(const std::vector<Block<3>>& blocks, const std::vector<int16_t>& ranks) 
{
	c10::InferenceMode inference_guard;

	std::deque<std::future<void>> futures;

	auto future_catcher = [](std::future<void>& fut) {
		try {
			fut.get();
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (...) {
			std::cerr << "caught something strange: " << std::endl;
		}
	};

	for (int i = 0; i < blocks.size(); ++i) {

		const int16_t& rank = ranks.size() == 1 ? ranks[0] : ranks[i];

		const auto& block = blocks[i];

		auto blockrunner = [this, &block, &rank](DeviceContext& context) { block_svt_step(context, block, rank); };

		futures.emplace_back(_tpool->enqueue(blockrunner));

		if (futures.size() > 32 * _dcontexts.size()) {
			future_catcher(futures.front());
			futures.pop_front();
		}

	}

	// we wait for all promises
	while (futures.size() > 0) {
		future_catcher(futures.front());
		futures.pop_front();
	}
}

void hasty::LLR_4DEncodes::block_svt_step(DeviceContext& dctxt, const Block<3>& block, int16_t rank)
{
	c10::InferenceMode inference_guard;

	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	auto tensor_block = hasty::extract_block(_image, block);
	auto cuda_block = tensor_block.to(dctxt.device);

	auto low_ranked = hasty::svt_hard(cuda_block, rank, std::nullopt);

	tensor_block.copy_(low_ranked, true);

	{
		std::lock_guard<std::mutex> lock(_copy_back_mutex);
		hasty::insert_block(_image, block, tensor_block);
	}

}





void hasty::LLR_4DEncodes::step_l2_sgd(const std::vector<
	std::pair<int, std::vector<int>>>& encode_coil_indices)
{
	c10::InferenceMode inference_guard;

	std::deque<std::future<void>> futures;

	auto future_catcher = [](std::future<void>& fut) {
		try {
			fut.get();
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (...) {
			std::cerr << "caught something strange: " << std::endl;
		}
	};

	for (int frame = 0; frame < _nframe; ++frame) {
		for (const auto& encode_pair : encode_coil_indices) {
			int encode = encode_pair.first;
			auto& coils = encode_pair.second;

			std::function<void(DeviceContext&)> coil_encode_stepper = [this, frame, encode, &coils](DeviceContext& context) {
				coil_encode_step(context, frame, encode, coils);
			};

			futures.emplace_back(_tpool->enqueue(coil_encode_stepper));

			if (futures.size() > 32 * _dcontexts.size()) {
				future_catcher(futures.front());
				futures.pop_front();
			}
			
		}
	}

	// we wait for all promises
	while(futures.size() > 0) {
		future_catcher(futures.front());
		futures.pop_front();
	}
	
}

void hasty::LLR_4DEncodes::coil_encode_step(DeviceContext& dctxt, int frame, int encode, const std::vector<int32_t>& coils)
{
	//printf("frame: %d, encode: %d\n", frame, encode);
	
	c10::InferenceMode inference_guard;

	// This shall be done on the correct device
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	// CPU Frame Encode View
	auto cpu_frame_encode_view = _image.select(0, frame).select(0, encode).unsqueeze(0); //_image.index({ frame, encode, "..." });

	// Copy image to device
	dctxt.image.copy_(cpu_frame_encode_view, true);

	// Copy coords to device
	auto& coords = _coords[frame][encode];
	if (dctxt.coords.sizes() == coords.sizes()) {
		dctxt.coords.copy_(coords, true);
	}
	else {
		dctxt.coords = coords.to(dctxt.device, true);
	}

	// Copy kdata to device
	auto& kdata = _kdata[frame][encode];
	if (dctxt.kdata.sizes() == kdata.sizes()) {
		dctxt.kdata.copy_(kdata, true);
	}
	else {
		dctxt.kdata = kdata.to(dctxt.device, true);
	}
	

	// Copy weights to device
	if (_weights.has_value()) {
		auto& weights = _weights.value()[frame][encode];
		auto& dctxt_weights = dctxt.weights.value();
		if (dctxt_weights.sizes() == weights.sizes()) {
			dctxt_weights.copy_(weights, true);
		}
		else {

			dctxt_weights = weights.to(dctxt.device, true);
		}
	}

	dctxt.sense = std::make_unique<SenseNormal>(dctxt.coords, _nmodes);

	auto freq_manip = [&dctxt](at::Tensor& in, int coil) {
		auto kdata_coil = dctxt.kdata.select(0, coil).unsqueeze(0);
		in.sub_(kdata_coil);
		if (dctxt.weights.has_value()) {
			in.mul_(dctxt.weights.value());
		}
	};

	dctxt.sense->apply(dctxt.image, dctxt.out, dctxt.smaps, coils, dctxt.in_storage, std::nullopt, freq_manip);

	{
		std::lock_guard<std::mutex> lock(_copy_back_mutex);
		cpu_frame_encode_view.add_(dctxt.out.to(c10::DeviceType::CPU));
	}

	dctxt.sense = nullptr;

}




hasty::RandomBlocksSVT::RandomBlocksSVT(std::vector<DeviceContext>& contexts,
	at::Tensor& image, int32_t nblocks, int32_t block_size, double thresh, bool soft)
	: _image(image)
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

	auto future_catcher = [](std::future<void>& fut) {
		try {
			fut.get();
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (...) {
			std::cerr << "caught something strange: " << std::endl;
		}
	};

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
			future_catcher(futures.front());
			futures.pop_front();
		}

	}

	// we wait for all promises
	while (futures.size() > 0) {
		future_catcher(futures.front());
		futures.pop_front();
	}
}

void hasty::RandomBlocksSVT::block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft)
{
	c10::InferenceMode inference_guard;

	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor cuda_block;
	{
		at::Tensor tmp_block;
		{
			std::lock_guard<std::mutex> lock(_mutex);
			tmp_block = hasty::extract_block(_image, block).detach().clone();
		}
		cuda_block = tmp_block.to(dctxt.stream.device());
	}

	at::Tensor low_ranked = soft ?
		hasty::svt_soft(cuda_block, (int)(thresh + 0.5), std::nullopt) :
		hasty::svt_hard(cuda_block, (float)thresh, std::nullopt);

	at::Tensor back_block = low_ranked.cpu();

	{
		std::lock_guard<std::mutex> lock(_mutex);
		hasty::insert_block(_image, block, back_block);
	}
}
