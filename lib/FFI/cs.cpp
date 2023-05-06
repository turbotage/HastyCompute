#include "cs.hpp"

#include "../cs/llr_cu.hpp"
#include <algorithm> 
#include <cstdlib>

void hasty::ffi::llr(const at::Tensor& coords, at::Tensor& input, const at::Tensor& smaps, const at::Tensor& kdata, int64_t iter)
{
	using namespace hasty::cuda;
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

	
	auto llr_options = LLR_4DEncodes::Options(c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0)));
	llr_options.devices.emplace_back(c10::Device(c10::kCUDA));

	//LLR_4DEncodes llr(llr_options, input, std::move(coords_vec), smaps, std::move(kdata_vec), std::move(weights_vec));
	LLR_4DEncodes llr(llr_options, input, std::move(coords_vec), smaps, std::move(kdata_vec));

	std::random_device rd;
	std::mt19937 g(rd());
	std::uniform_int_distribution<std::mt19937::result_type> dist(ncoils / 4,ncoils);

	for (int k = 0; k < iter; ++k) {
		std::vector<std::pair<int, std::vector<int>>> mhm;
		for (int i = 0; i < nencodes; ++i) {
			auto coil_copy = coil_nums;
			//std::shuffle(coil_copy.begin(), coil_copy.end(), g);
			//coil_copy = std::vector(coil_copy.begin(), coil_copy.begin() + dist(g));


			mhm.push_back(std::make_pair(i, std::move(coil_copy)));
		}

		std::cout << "iter: " << k;

		llr.step_l2_sgd(mhm);
	}


}


