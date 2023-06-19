#include "cs.hpp"

#include "../cs/llr.hpp"
#include "../cs/sense.hpp"
#include "../cs/svt.hpp"
#include <algorithm> 
#include <cstdlib>
#include <c10/cuda/CUDAGuard.h>


std::vector<hasty::BatchedSense::DeviceContext> get_batch_contexts(const std::vector<c10::Stream>& streams, const at::Tensor& smaps)
{
	std::unordered_map<c10::Device, at::Tensor> smaps_dict;
	std::vector<hasty::BatchedSense::DeviceContext> contexts(streams.begin(), streams.end());
	for (auto& context : contexts) {
		const auto& device = context.stream.device();
		if (smaps_dict.contains(device)) {
			context.smaps = smaps_dict.at(device);
		}
		else {
			smaps_dict.insert({ device, smaps.to(device, true) });
		}
	}
	return contexts;
}

void hasty::ffi::batched_sense(
	at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
	const std::vector<at::Tensor>& coords, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::nullopt);

	bsense.apply(input, coils, std::nullopt, std::nullopt, std::nullopt);
}

void hasty::ffi::batched_sense_kdata(
	at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
	const std::vector<at::Tensor>& coords,
	const std::vector<at::Tensor>& kdatas,
	const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;
	auto kdata_cpy = kdatas;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::move(kdata_cpy), std::nullopt);

	BatchedSense::FreqManipulator fmanip = [](at::Tensor& in, at::Tensor& kdata) {
		in.sub_(kdata);
	};

	bsense.apply(input, coils, std::nullopt, fmanip, std::nullopt);
}

void hasty::ffi::batched_sense_weighted(
	at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
	const std::vector<at::Tensor>& coords,
	const std::vector<at::Tensor>& weights,
	const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;
	auto weights_cpy = weights;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::move(weights_cpy));

	BatchedSense::WeightedManipulator wmanip = [](at::Tensor& in, at::Tensor& weights) {
		in.mul_(weights);
	};

	bsense.apply(input, coils, wmanip, std::nullopt, std::nullopt);
}

void hasty::ffi::batched_sense_weighted_kdata(
	at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
	const std::vector<at::Tensor>& coords,
	const std::vector<at::Tensor>& weights,
	const std::vector<at::Tensor>& kdatas,
	const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;
	auto kdata_cpy = kdatas;
	auto weights_cpy = weights;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::move(kdata_cpy), std::move(weights_cpy));

	BatchedSense::WeightedFreqManipulator wfmanip = [](at::Tensor& in, at::Tensor& kdata, at::Tensor& weights) {
		in.sub_(kdata);
		in.mul_(weights);
	};

	bsense.apply(input, coils, std::nullopt, std::nullopt, wfmanip);
}


void hasty::ffi::batched_sense_toeplitz(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps, 
	const std::vector<at::Tensor>& coords, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::nullopt);

	bsense.apply_toep(input, coils);
}

void hasty::ffi::batched_sense_toeplitz_diagonals(at::Tensor& input, const std::vector<std::vector<int32_t>>& coils, const at::Tensor& smaps,
	const at::Tensor& diagonals, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	at::Tensor diagonals_cpy = diagonals;

	hasty::BatchedSense bsense(std::move(contexts), std::move(diagonals_cpy));

	bsense.apply_toep(input, coils);
}












void hasty::ffi::random_blocks_svt(at::Tensor& input, int32_t nblocks, int32_t block_size, 
	double thresh, bool soft, const std::vector<c10::Stream>& streams)
{
	std::vector<RandomBlocksSVT::DeviceContext> contexts(streams.begin(), streams.end());

	RandomBlocksSVT(std::move(contexts), input, nblocks, block_size, thresh, soft);
}

