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
				TensorVec&& coords,
				const at::Tensor& smaps,
				TensorVec&& kdata)

			

		private:

			void construct();

		private:

			struct DeviceContext {
				DeviceContext(const c10::Device& dev) :
					device(dev), device_mutex(std::make_unique<std::mutex>()) {}
				DeviceContext(const DeviceContext&) = delete;
				DeviceContext& operator=(const DeviceContext&) = delete;
				DeviceContext(DeviceContext&&) = default;

				std::string str();

				c10::Device device;
				at::Tensor image;
				at::Tensor kdata;
				at::Tensor coords;

				at::Tensor out;

				at::Tensor in_storage; // TensorOptions as image
				//at::Tensor freq_storage; // TensorOptions as kdata

				at::Tensor smaps;

				std::unique_ptr<SenseNormal> sense;

				// Probably have to rethink this
				std::unique_ptr<std::mutex> device_mutex;
			};

		private:

			Options _options;
			at::Tensor& _image;
			TensorVec _coords;
			const at::Tensor& _smaps;
			TensorVec _kdata;

			std::vector<int64_t> _nmodes;

			std::mutex _copy_back_mutex;

			std::vector<DeviceContext> _dcontexts;

		};

	}
}

