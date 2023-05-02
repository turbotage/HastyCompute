#include "llr_cu.hpp"

#include <random>

import hasty_util;

hasty::cuda::LLR_4DEncodes::LLR_4DEncodes(
	const LLR_4DEncodesOptions& options,
	at::Tensor& image,
	const CTensorVecVec& coords,
	const at::Tensor& smaps,
	const CTensorVecVec& kdata,
	const CTensorVecVec& weights)
	:
	_options(options),
	_image(image),
	_coords(coords),
	_smaps(smaps),
	_kdata(kdata),
	_weights(weights),
	_nframe(image.size(0))
{
	
}

void hasty::cuda::LLR_4DEncodes::step_l2_sgd(const std::vector<
	std::pair<int, std::vector<int>>>& encode_coil_indices)
{
	
	using dcontext_iterator = std::vector<DeviceContext>::iterator;

	auto it = _dcontexts.begin();

	auto circular_update = [this](const dcontext_iterator& it) {
		auto updated_it = it + 1;
		if (updated_it == this->_dcontexts.end()) {
			updated_it == this->_dcontexts.begin();
		}
		return updated_it;
	};
	

	for (int frame = 0; frame < _nframe; ++frame) {
		for (const auto& encode_pair : encode_coil_indices) {
			int encode = encode_pair.first;

			// Get the device context
			auto& dctxt = *it;

			// Copy image to device
			dctxt.image.copy_(_image.index({frame, encode, "..." }), true);

			// Copy coords to device
			auto& coords = _coords[frame][encode].get();
			if (dctxt.coords.sizes() == coords.sizes()) {
				dctxt.coords.copy_(coords, true);
			} else {
				dctxt.coords = coords.to(dctxt.device);
			}

			// Copy kdata to device
			auto& kdata = _kdata[frame][encode].get();
			if (dctxt.kdata.sizes() == kdata.sizes()) {
				dctxt.kdata.copy_(kdata, true);
			} else {
				dctxt.kdata = kdata.to(dctxt.device);
			}

			// Copy weights to device
			auto& weights = _weights[frame][encode].get();
			if (dctxt.weights.sizes() == weights.sizes()) {
				dctxt.weights.copy_(weights, true);
			} else {

				dctxt.weights = weights.to(dctxt.device);
			}

			dctxt.normal_nufft = std::make_unique<NufftNormal>();


			// Move to the next device context
			it = circular_update(it);
		}
	}
}