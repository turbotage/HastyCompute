#pragma once

#include "block.hpp"
#include "../torch_util.hpp"
#include "sense.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {

	using TensorVec = std::vector<at::Tensor>;
	using TensorVecVec = std::vector<TensorVec>;


	class RandomBlocksSVT {
	public:

		struct DeviceContext {

			DeviceContext(const c10::Stream& stream) : 
				stream(stream) {}
			DeviceContext(const DeviceContext&) = delete;
			DeviceContext& operator=(const DeviceContext&) = delete;
			DeviceContext(DeviceContext&&) = default;

			std::string str();

			c10::Stream stream;

		};
	public:

		RandomBlocksSVT(std::vector<DeviceContext>& contexts,
			at::Tensor& image, int32_t nblocks, int32_t block_size, double thresh, bool soft);

	private:

		void block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft);

	private:
		std::mutex _mutex;
		at::Tensor _image;
		int32_t _nctxt;
	};

	class NormalBlocksSVT {
	public:

		struct DeviceContext {

			DeviceContext(const c10::Stream& stream) :
				stream(stream) {}
			DeviceContext(const DeviceContext&) = delete;
			DeviceContext& operator=(const DeviceContext&) = delete;
			DeviceContext(DeviceContext&&) = default;

			std::string str();

			c10::Stream stream;

		};
	public:

		NormalBlocksSVT(std::vector<DeviceContext>& contexts,
			at::Tensor& image, std::array<int64_t, 3> block_strides, std::array<int64_t,3> block_shape, int block_iter, double thresh, bool soft);

	private:

		void block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft);

	private:
		std::mutex _mutex;
		at::Tensor _image;
		int32_t _nctxt;
	};


}