#pragma once

#include "block.hpp"
#include "../torch_util.hpp"
#include "sense_cu.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {
	namespace cuda {

		using TensorVec = std::vector<at::Tensor>;

		using TensorVecVec = std::vector<TensorVec>;

		
		// Runs 
		class LLR_4DEncodes {
		public:

			struct Options {
				Options(const c10::Device& device)
				{
					devices.emplace_back(device);
				}

				std::vector<c10::Device> devices;
			};

		public:

			LLR_4DEncodes(
				const Options& options,
				at::Tensor& image,
				const TensorVecVec& coords,
				const at::Tensor& smaps,
				const TensorVecVec& kdata,
				const TensorVecVec& weights);

			LLR_4DEncodes(
				const Options& options,
				at::Tensor& image,
				TensorVecVec&& coords,
				const at::Tensor& smaps,
				TensorVecVec&& kdata,
				TensorVecVec&& weights);

			void step_l2_sgd(const std::vector<
				std::pair<int, std::vector<int>>>& encode_coil_indices);

		private:

			void construct();

		private:

			struct DeviceContext {

				DeviceContext(const c10::Device& dev) : device(dev) {}

				c10::Device device;
				at::Tensor image;
				at::Tensor kdata;
				at::Tensor weights;
				at::Tensor coords;

				at::Tensor out;

				at::Tensor in_storage; // TensorOptions as image
				//at::Tensor freq_storage; // TensorOptions as kdata

				at::Tensor smaps;

				std::unique_ptr<SenseNormal> sense;
			};

			void coil_encode_step(const std::vector<DeviceContext>::iterator& dit, int frame, int encode, const std::vector<int32_t>& coils);

		private:

			ThreadPool _tpool;

			Options _options;
			at::Tensor& _image;
			TensorVecVec _coords;
			const at::Tensor& _smaps;
			TensorVecVec _kdata;
			TensorVecVec _weights;

			std::vector<int64_t> _nmodes;

			int _nframe;
			std::mutex _copy_back_mutex;

			std::vector<DeviceContext> _dcontexts;

		};

	}
}