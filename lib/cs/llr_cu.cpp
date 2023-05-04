#include "llr_cu.hpp"

#include <random>
#include <c10/cuda/CUDAGuard.h>

import hasty_util;

hasty::cuda::LLR_4DEncodes::LLR_4DEncodes(
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
	_nmodes(4),
	_tpool(options.devices.size())
{
	construct();
}

hasty::cuda::LLR_4DEncodes::LLR_4DEncodes(
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
	_nmodes(4),
	_tpool(options.devices.size())
{
	construct();
}

hasty::cuda::LLR_4DEncodes::LLR_4DEncodes(
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
	_nmodes(4),
	_tpool(options.devices.size())
{
	construct();
}

hasty::cuda::LLR_4DEncodes::LLR_4DEncodes(
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
	_nmodes(4),
	_tpool(options.devices.size())
{
	construct();
}

void hasty::cuda::LLR_4DEncodes::construct()
{
	_nmodes[0] = 1; _nmodes[1] = _image.size(2); _nmodes[2] = _image.size(3); _nmodes[3] = _image.size(4);

	auto base_options1 = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto base_options2 = c10::TensorOptions().dtype(c10::ScalarType::Float);

	//_dcontexts.reserve(_options.devices.size());
	for (auto& device : _options.devices) {

		auto& dcontext = _dcontexts.emplace_back(device);

		c10::cuda::CUDAGuard device_guard(device);

		auto options1 = base_options1.device(device);
		auto options2 = base_options2.device(device);

		dcontext.image = at::empty(at::makeArrayRef(_nmodes), options1);
		dcontext.out = at::empty_like(dcontext.image);
		dcontext.in_storage = at::empty_like(dcontext.image);

		dcontext.smaps = _smaps.to(device, true);

	}
}




void hasty::cuda::LLR_4DEncodes::step_l2_sgd(const std::vector<
	std::pair<int, std::vector<int>>>& encode_coil_indices)
{
	c10::InferenceMode inference_guard;

	using dcontext_iterator = std::vector<DeviceContext>::iterator;

	auto it = _dcontexts.begin();

	auto circular_update = [this](const dcontext_iterator& it) {
		auto updated_it = it + 1;
		if (updated_it == this->_dcontexts.end()) {
			updated_it = this->_dcontexts.begin();
		}
		return updated_it;
	};
	

	std::vector<std::future<void>> futures;
	futures.reserve(_nframe * encode_coil_indices.size());

	for (int frame = 0; frame < _nframe; ++frame) {
		for (const auto& encode_pair : encode_coil_indices) {
			int encode = encode_pair.first;
			auto& coils = encode_pair.second;
			auto coil_encode_stepper = [this, it, frame, encode, &coils]() {
				try {
					coil_encode_step(it, frame, encode, coils);
				}
				catch (c10::Error& e) {
					std::cerr << e.what() << std::endl;
				}
				catch (std::exception& e) {
					std::cerr << e.what() << std::endl;
				}
				catch (...) {
					std::cerr << "what";
				}
			};

			//futures.emplace_back(_tpool.enqueue(coil_encode_stepper));
			coil_encode_stepper();

			// Move to the next device context
			it = circular_update(it);
		}
	}

	// we wait for all promises
	for (auto rit = std::begin(futures); rit != std::end(futures); ++rit) {
		rit->get();
	}
	
}


void hasty::cuda::LLR_4DEncodes::coil_encode_step(const std::vector<DeviceContext>::iterator& dit, int frame, int encode, const std::vector<int32_t>& coils)
{
	std::cout << " frame: " << frame << " encode: " << encode;
	c10::InferenceMode inference_guard;

	// Get the device context
	auto& dctxt = *dit;

	// Lock this device
	std::lock_guard<std::mutex> lock(*dctxt.device_mutex);
		
	// This shall be done on the correct device
	c10::cuda::CUDAGuard guard(dctxt.device.index());

	// CPU Frame Encode View
	auto cpu_frame_encode_view = _image.select(0, frame).select(0, encode).unsqueeze(0); //_image.index({ frame, encode, "..." });

	// Copy image to device
	//dctxt.image.copy_(cpu_frame_encode_view, true);
	dctxt.image = cpu_frame_encode_view.to(dctxt.device);

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
	dctxt.kdata = kdata.to(dctxt.device);
	/*
	if (dctxt.kdata.sizes() == kdata.sizes()) {
		dctxt.kdata.copy_(kdata, true);
	}
	else {
		dctxt.kdata = kdata.to(dctxt.device, true);
	}
	*/

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

	torch::cuda::synchronize(dctxt.device.index());

	//if (dctxt.sense == nullptr) {
	//dctxt.sense = nullptr;
	dctxt.sense = std::make_unique<SenseNormal>(dctxt.coords, _nmodes);
	//}

	auto freq_manip = [&dctxt](at::Tensor& in, int coil) {
		auto kdata_coil = dctxt.kdata.select(0, coil).unsqueeze(0);
		//in.sub_(kdata_coil);
		in.sub_(0.1f);
		if (dctxt.weights.has_value()) {
			in.mul_(dctxt.weights.value());
		}
	};

	try {
		torch::cuda::synchronize();
	}
	catch (c10::Error e) {
		std::cerr << "c10::Error: " << e.what() << std::endl;
	}

	try {
		dctxt.sense->apply(dctxt.image, dctxt.smaps, coils, dctxt.out, dctxt.in_storage, std::nullopt, freq_manip);
	}
	catch (c10::Error& e) {
		std::cerr << e.what() << std::endl;
	}

	{
		std::lock_guard<std::mutex> lock(_copy_back_mutex);
		cpu_frame_encode_view.add_(dctxt.out.to(c10::DeviceType::CPU));
	}

	dctxt.sense = nullptr;


	//torch::cuda::synchronize();
}
