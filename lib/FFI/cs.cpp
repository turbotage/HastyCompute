#include "cs.hpp"

#include "../cs/llr.hpp"
#include "../cs/sense.hpp"
#include <algorithm> 
#include <cstdlib>
#include <c10/cuda/CUDAGuard.h>


void hasty::ffi::batched_sense(at::Tensor input, const at::Tensor& smaps, const std::vector<at::Tensor>& coords)
{
	std::vector<BatchedSense::DeviceContext> contexts;

	auto device = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto& context = contexts.emplace_back(device, c10::cuda::getDefaultCUDAStream(device.index()));
	context.smaps = smaps.to(device);

	auto coords_cpy = coords;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy));

	bsense.apply_toep(input, std::nullopt);

}

void hasty::ffi::batched_sense(at::Tensor input, const at::Tensor& smaps, const std::vector<at::Tensor>& coords, const std::vector<at::Tensor>& kdatas)
{
	std::vector<BatchedSense::DeviceContext> contexts;

	auto device = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto& context1 = contexts.emplace_back(device, c10::cuda::getDefaultCUDAStream(device.index()));
	context1.smaps = smaps.to(device);
	//auto& context2 = contexts.emplace_back(device, c10::cuda::getStreamFromPool(false, device.index()));
	//context2.smaps = contexts[0].smaps;

	auto coords_cpy = coords;
	auto kdata_cpy = kdatas;

	hasty::BatchedSense bsense(std::move(contexts), std::move(coords_cpy), std::move(kdata_cpy));

	BatchedSense::FreqManipulator manip = [](at::Tensor& in, at::Tensor& kdata) {
		in.sub_(kdata);
	};

	bsense.apply(input, std::nullopt, std::nullopt, manip);
}


void hasty::ffi::llr(const at::Tensor& coords, at::Tensor& input, const at::Tensor& smaps, const at::Tensor& kdata, int64_t iter)
{
	using namespace hasty;
	int nframes = input.size(0);
	int nencodes = input.size(1);
	int ncoils = smaps.size(0);

	std::vector<int> coil_nums(ncoils);
	for (int i = 0; i < ncoils; ++i) {
		coil_nums[i] = i;
	}

	TensorVecVec coords_vec;
	coords_vec.reserve(nframes);
	//TensorVecVec weights_vec;
	//weights_vec.reserve(nframes);
	TensorVecVec kdata_vec;
	kdata_vec.reserve(nframes);

	//auto options = input.options().dtype(c10::kFloat);

	std::cout << "vectorizing coords and kdata: " << std::endl;

	for (int frame = 0; frame < nframes; ++frame) {
		auto& coords_encode_vec = coords_vec.emplace_back();
		coords_encode_vec.reserve(nencodes);
		//auto& weights_encode_vec = weights_vec.emplace_back();
		//weights_encode_vec.reserve(nencodes);
		auto& kdata_encode_vec = kdata_vec.emplace_back();
		kdata_encode_vec.reserve(nencodes);

		for (int encode = 0; encode < nencodes; ++encode) {
			auto coord = coords.select(0, frame).select(0, encode);
			coords_encode_vec.emplace_back(coord);
			//weights_encode_vec.emplace_back(at::ones({ 1,coord.size(1)}, options));
			auto kdat = kdata.select(0, frame).select(0, encode);
			kdata_encode_vec.emplace_back(kdat);
		}
	}

	auto device = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));

	auto llr_options = LLR_4DEncodes::Options(device, c10::cuda::getDefaultCUDAStream(device.index()));
	llr_options.push_back_device(device, c10::cuda::getStreamFromPool(device.index()));

	//llr_options.devices.emplace_back(c10::Device(c10::kCUDA));

	//LLR_4DEncodes llr(llr_options, input, std::move(coords_vec), smaps, std::move(kdata_vec), std::move(weights_vec));
	LLR_4DEncodes llr(llr_options, input, std::move(coords_vec), smaps, std::move(kdata_vec));

	std::random_device rd;
	std::mt19937 g(rd());
	std::uniform_int_distribution<std::mt19937::result_type> dist(ncoils / 4,ncoils);


	std::array<int64_t, 3> lens{ 16,16,16 };
	std::array<int64_t, 3> bounds{ smaps.size(1) - lens[0], smaps.size(2) - lens[1], smaps.size(3) - lens[2] };

	int nblocks = 1000;
	int16_t rank = 10;

	auto start = std::chrono::steady_clock::now();

	for (int k = 0; k < iter; ++k) {

		std::cout << "creates coil choices:" << std::endl;

		// Create coil choices fir L2 step
		std::vector<std::pair<int, std::vector<int>>> mhm;
		for (int i = 0; i < nencodes; ++i) {
			auto coil_copy = coil_nums;
			//std::shuffle(coil_copy.begin(), coil_copy.end(), g);
			//coil_copy = std::vector(coil_copy.begin(), coil_copy.begin() + dist(g));

			mhm.push_back(std::make_pair(i, std::move(coil_copy)));
		}

		std::cout << "takes L2 steps:" << std::endl;

		// Make L2 step
		llr.step_l2_sgd(mhm);

		std::cout << "creates blocks:" << std::endl;

		// Create SVT blocks
		std::vector<Block<3>> blocks(1000);
		for (int i = 0; i < nblocks; ++i) {
			Block<3>::randomize(blocks[i], bounds, lens);
		}

		std::cout << "step llr:" << std::endl;

		// Do SVT
		llr.step_llr(blocks, {rank});

		std::cout << "iter: " << k;

	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

}


