#pragma once

#include "../torch_util.hpp"
#include "sense_cu.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {
	namespace cuda {
		
		using TensorVec = std::vector<at::Tensor>;
		using TensorVecVec = std::vector<TensorVec>;

		class CGRecon {
		public:

			struct Options {
				Options(const c10::Device& device)
				{
					devices.emplace_back(device);
				}

				std::vector<c10::Device> devices;
			};

		public:

			CGRecon(const CGRecon&) = delete;
			CGRecon(CGRecon&&) = delete;
			CGRecon& operator=(const CGRecon&) = delete;

			CGRecon(
				const Options& options,
				at::Tensor& image,
				const TensorVec& coords,
				const at::Tensor& smaps,
				const TensorVec& kdata);

			CGRecon(
				const Options& options,
				at::Tensor& image,
				const TensorVec& coords,
				const at::Tensor& smaps,
				const TensorVec& kdata,
				const TensorVec& weights);

			CGRecon(
				const Options& options,
				at::Tensor& image,
				TensorVec&& coords,
				const at::Tensor& smaps,
				TensorVec&& kdata);

			CGRecon(
				const Options& options,
				at::Tensor& image,
				TensorVec&& coords,
				const at::Tensor& smaps,
				TensorVec&& kdata,
				TensorVec&& weights);

		private:

			void construct();

		private:



		private:

		};

	}
}

