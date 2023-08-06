#pragma once

#include "ffi_defines.hpp"
#include "../torch_util.hpp"
#include "../cs/batch_sense.hpp"

namespace hasty {

	namespace ffi {


		class LIB_EXPORT BatchedSense {
		public:

			BatchedSense(const at::TensorList& coords, const at::Tensor& smaps, 
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<std::vector<at::Stream>>& streams);

			void apply(const at::Tensor& in, at::TensorList out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		public:

			static BatchedSense create(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<std::vector<at::Stream>>& streams);

		private:
			std::unique_ptr<hasty::BatchedSense> _bs;
		};

	}
}

