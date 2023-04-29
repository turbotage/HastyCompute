#pragma once

#include "block.hpp"
#include "../torch_util.hpp"
#include "../fft/nufft_cu.hpp"

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
				const CTensorVecVec& kdata,
				const CTensorVecVec& weights);

			void step_l2_sgd(const std::vector<
				std::pair<int, std::vector<int>>>& encode_coil_indices);



			void iter();

		private:

			struct DeviceContext {
				at::Device device;
				at::Tensor image;
				at::Tensor kdata;
				at::Tensor weights;
				at::Tensor coords;

				std::unique_ptr<NufftNormal> normal_nufft;
			};

		private:

			LLR_4DEncodesOptions _options;
			at::Tensor& _image;
			CTensorVecVec _coords;
			const at::Tensor& _smaps;
			CTensorVecVec _kdata;
			CTensorVecVec _weights;

			int _nframe;


			std::vector<DeviceContext> _dcontexts;

		};

	}
}