#include "cs.hpp"

#include "../cs/llr.hpp"
#include "../cs/sense.hpp"
#include "../cs/svt.hpp"
#include <algorithm> 
#include <cstdlib>
#include <c10/cuda/CUDAGuard.h>


/*
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
			context.smaps = smaps.to(device, true);
			smaps_dict.insert({ device, context.smaps });
		}
	}
	return contexts;
}


// FORWARD

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_forward(const at::Tensor& input, at::TensorList output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::nullopt);

	return bsense.apply_forward(input, output, sum, sumnorm, coils, std::nullopt, std::nullopt, std::nullopt);
}

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_forward_weighted(const at::Tensor& input, std::vector<at::Tensor>& output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
	bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;
	auto weights_cpy = weights;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::move(weights_cpy));

	BatchedSense::WeightedManipulator wmanip = [](at::Tensor& in, at::Tensor& weights) {
		in.mul_(weights);
	};

	return bsense.apply_forward(input, output, sum, sumnorm, coils, wmanip, std::nullopt, std::nullopt);
}

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_forward_kdata(const at::Tensor& input, std::vector<at::Tensor>& output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas, 
	bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;
	auto kdata_cpy = kdatas;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::move(kdata_cpy), std::nullopt);

	BatchedSense::FreqManipulator fmanip = [](at::Tensor& in, at::Tensor& kdata) {
		in.sub_(kdata);
	};

	return bsense.apply_forward(input, output, sum, sumnorm, coils, std::nullopt, fmanip, std::nullopt);
}

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_forward_weighted_kdata(const at::Tensor& input, std::vector<at::Tensor>& output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
	const std::vector<at::Tensor>& kdatas, bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
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

	return bsense.apply_forward(input, output, sum, sumnorm, coils, std::nullopt, std::nullopt, wfmanip);
}

// ADJOINT

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_adjoint(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::nullopt);

	return bsense.apply_adjoint(input, output, sum, sumnorm, coils, std::nullopt, std::nullopt, std::nullopt);
}

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_adjoint_weighted(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
	bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;
	auto weights_cpy = weights;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::move(weights_cpy));

	BatchedSense::WeightedManipulator wmanip = [](at::Tensor& in, at::Tensor& weights) {
		in.mul_(weights);
	};

	return bsense.apply_adjoint(input, output, sum, sumnorm, coils, wmanip, std::nullopt, std::nullopt);
}

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_adjoint_kdata(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas, 
	bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;
	auto kdata_cpy = kdatas;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::move(kdata_cpy), std::nullopt);

	BatchedSense::FreqManipulator fmanip = [](at::Tensor& in, at::Tensor& kdata) {
		in.sub_(kdata);
	};

	return bsense.apply_adjoint(input, output, sum, sumnorm, coils, std::nullopt, fmanip, std::nullopt);
}

LIB_EXPORT at::Tensor hasty::ffi::batched_sense_adjoint_weighted_kdata(const std::vector<at::Tensor>& input, at::Tensor& output, const std::optional<std::vector<std::vector<int64_t>>>& coils, 
	const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& weights, 
	const std::vector<at::Tensor>& kdatas, bool sum, bool sumnorm, const std::vector<c10::Stream>& streams)
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

	return bsense.apply_adjoint(input, output, sum, sumnorm, coils, std::nullopt, std::nullopt, wfmanip);
}

// NORMAL

void hasty::ffi::batched_sense_normal(
	at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
	const std::vector<at::Tensor>& coords, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::nullopt);

	bsense.apply_normal(input, coils, std::nullopt, std::nullopt, std::nullopt);
}

void hasty::ffi::batched_sense_normal_weighted(
	at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
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

	bsense.apply_normal(input, coils, wmanip, std::nullopt, std::nullopt);
}

void hasty::ffi::batched_sense_normal_kdata(
	at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
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

	bsense.apply_normal(input, coils, std::nullopt, fmanip, std::nullopt);
}

void hasty::ffi::batched_sense_normal_weighted_kdata(
	at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
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

	bsense.apply_normal(input, coils, std::nullopt, std::nullopt, wfmanip);
}

// TOEPLITZ

void hasty::ffi::batched_sense_toeplitz(at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
	const std::vector<at::Tensor>& coords, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	auto coords_cpy = coords;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::nullopt, std::nullopt);

	bsense.apply_toep(input, coils);
}

void hasty::ffi::batched_sense_toeplitz_diagonals(at::Tensor& input, const std::optional<std::vector<std::vector<int64_t>>>& coils, const at::Tensor& smaps,
	const at::Tensor& diagonals, const std::vector<c10::Stream>& streams)
{
	auto contexts = get_batch_contexts(streams, smaps);
	at::Tensor diagonals_cpy = diagonals;

	hasty::BatchedSense bsense(std::move(contexts), std::move(diagonals_cpy));

	bsense.apply_toep(input, coils);
}
*/

// LLR

void hasty::ffi::random_blocks_svt(at::Tensor& input, int64_t nblocks, int64_t block_size,
	double thresh, bool soft, const std::vector<c10::Stream>& streams)
{
	std::vector<RandomBlocksSVT::DeviceContext> contexts(streams.begin(), streams.end());

	RandomBlocksSVT(contexts, input, nblocks, block_size, thresh, soft);
}

void hasty::ffi::normal_blocks_svt(at::Tensor& input, std::vector<int64_t> block_strides, std::vector<int64_t> block_shapes,
	int64_t block_iter, double thresh, bool soft, const std::vector<c10::Stream>& streams)
{
	std::vector<NormalBlocksSVT::DeviceContext> contexts(streams.begin(), streams.end());

	std::array<int64_t, 3> ablock_strides;
	std::copy_n(block_strides.begin(), 3, ablock_strides.begin());

	std::array<int64_t, 3> ablock_shapes;
	std::copy_n(block_shapes.begin(), 3, ablock_shapes.begin());

	NormalBlocksSVT(contexts, input, ablock_strides, ablock_shapes, block_iter, thresh, soft);
}