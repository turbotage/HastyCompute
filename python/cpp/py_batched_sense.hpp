#pragma once

#include "py_util.hpp"


namespace hasty {

	namespace mri {
		class BatchedSense;
		class BatchedSenseAdjoint;
		class BatchedSenseNormal;
	}


	namespace ffi {

		class LIB_EXPORT BatchedSense : public torch::CustomClassHolder {
		public:

			BatchedSense(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(const at::Tensor& in, at::TensorList out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		private:
			std::unique_ptr<hasty::mri::BatchedSense> _bs;
		};

		class LIB_EXPORT BatchedSenseAdjoint : public torch::CustomClassHolder {
		public:

			BatchedSenseAdjoint(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(const at::TensorList& in, at::Tensor out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		private:
			std::unique_ptr<hasty::mri::BatchedSenseAdjoint> _bs;
		};

		class LIB_EXPORT BatchedSenseNormal : public torch::CustomClassHolder {
		public:

			BatchedSenseNormal(const at::TensorList& coords, const at::Tensor& smaps,
				const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
				const at::optional<at::ArrayRef<at::Stream>>& streams);

			void apply(const at::Tensor& in, at::Tensor out,
				const at::optional<std::vector<std::vector<int64_t>>>& coils);

		private:
			std::unique_ptr<hasty::mri::BatchedSenseNormal> _bs;
		};

	}
}


