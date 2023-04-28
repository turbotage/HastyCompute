#pragma once

#include "block.hpp"
#include "../torch_util.hpp"

namespace hasty {
	namespace cuda {

		using CTensorVec = std::vector<std::reference_wrapper<const at::Tensor>>;
		using TensorVec = std::vector<std::reference_wrapper<at::Tensor>>;

		using CTensorVecVec = std::vector<CTensorVec>;
		using TensorVecVec = std::vector<TensorVec>;

		
		// Runs 
		class LLR_4DEncodes {
		public:

			struct LLR_4DEncodesOptions {
				bool store_nufft_plans = false;
			};

		public:

			LLR_4DEncodes(
				const LLR_4DEncodesOptions& options,
				at::Tensor& image,
				const CTensorVecVec& coords,
				const at::Tensor& smaps,
				const CTensorVecVec& kdata);

			void step_l2_sgd(const std::vector<
				std::pair<int, std::vector<int>>>& coil_encode_indices);



			void iter();

		private:
			LLR_4DEncodesOptions _options;
			at::Tensor& _image;
			CTensorVecVec _coords;
			const at::Tensor& _smaps;
			CTensorVecVec _kdata;
		};

	}
}