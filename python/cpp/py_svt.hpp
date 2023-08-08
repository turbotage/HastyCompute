#pragma once

#include "py_util.hpp"

namespace hasty {
	namespace ffi {

		class LIB_EXPORT Random3DBlocksSVT : public torch::CustomClassHolder {
		public:

			Random3DBlocksSVT(const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(at::Tensor in, int64_t nblocks, const std::array<int64_t, 3>& block_shape, double thresh, bool soft);

		private:
			std::unique_ptr<hasty::svt::Random3DBlocksSVT> _rbsvt;
		};

		class LIB_EXPORT Normal3DBlocksSVT : public torch::CustomClassHolder {
		public:

			Normal3DBlocksSVT(const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(at::Tensor in, const std::array<int64_t, 3>& block_strides, const std::array<int64_t, 3>& block_shape,
				int64_t block_iter, double thresh, bool soft);

		private:
			std::unique_ptr<hasty::svt::Normal3DBlocksSVT> _nbsvt;
		};

	}
}