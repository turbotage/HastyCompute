#pragma once

#include "block.hpp"
#include "../torch_util.hpp"

namespace hasty {
	namespace svt {

		at::Tensor extract_block(const at::Tensor& tensor, const Block<4>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split = 1, bool transpose = false);

		at::Tensor extract_block(const at::Tensor& tensor, const Block<3>& block, const std::optional<std::vector<int64_t>>& perms, int flatten_split = 1, bool transpose = false);

		at::Tensor svt_hard(const at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		void svt_hard_inplace(at::Tensor& in, int rank, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		at::Tensor svt_soft(const at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		void svt_soft_inplace(at::Tensor& in, float lambda, std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> storage);

		void insert_block(at::Tensor& in, const Block<4>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose = false);

		void insert_block(at::Tensor& in, const Block<3>& block, at::Tensor& block_tensor, const std::optional<std::vector<int64_t>>& perms, bool transpose = false);

		class LIB_EXPORT RandomBlocksSVT {
		public:

			struct DeviceContext {

				DeviceContext(const c10::Stream& stream) :
					stream(stream) {}
				DeviceContext(const DeviceContext&) = delete;
				DeviceContext& operator=(const DeviceContext&) = delete;
				DeviceContext(DeviceContext&&) = default;

				c10::Stream stream;

			};

		public:

			RandomBlocksSVT(std::vector<DeviceContext>& contexts,
				at::Tensor& image, int64_t nblocks, std::vector<int64_t> block_shape, double thresh, bool soft);

		private:

			void block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft);

		private:
			std::mutex _mutex;
			at::Tensor _image;
			int32_t _nctxt;
		};

		class LIB_EXPORT NormalBlocksSVT {
		public:

			struct DeviceContext {

				DeviceContext(const c10::Stream& stream) :
					stream(stream) {}
				DeviceContext(const DeviceContext&) = delete;
				DeviceContext& operator=(const DeviceContext&) = delete;
				DeviceContext(DeviceContext&&) = default;

				c10::Stream stream;

			};
		public:

			NormalBlocksSVT(std::vector<DeviceContext>& contexts,
				at::Tensor& image, std::vector<int64_t> block_strides, std::vector<int64_t> block_shape, int block_iter, double thresh, bool soft);

		private:

			void block_svt_step(DeviceContext& dctxt, const Block<3>& block, double thresh, bool soft);

		private:
			std::mutex _mutex;
			at::Tensor _image;
			int32_t _nctxt;
		};

	}
}

