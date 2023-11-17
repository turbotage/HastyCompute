module;

#include <torch/torch.h>

export module sense;

import <vector>;
import <functional>;

import torch_util;
import nufft;

namespace hasty {

	export using TensorVec = std::vector<at::Tensor>;

	namespace mri {

		export using CoilApplier = std::function<void(at::Tensor&, int32_t)>;

		export struct CoilManipulator {
			CoilManipulator() = default;

			CoilManipulator& setPreApply(const CoilApplier& apply) {
				if (preapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				preapplier = at::make_optional(apply);
				return *this;
			}

			CoilManipulator& setMidApply(const CoilApplier& apply) {
				if (midapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				midapplier = at::make_optional(apply);
				return *this;
			}

			CoilManipulator& setPostApply(const CoilApplier& apply) {
				if (postapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				postapplier = at::make_optional(apply);
				return *this;
			}

			at::optional<CoilApplier> preapplier;
			at::optional<CoilApplier> midapplier;
			at::optional<CoilApplier> postapplier;
		};

		export class Sense {
		public:

			Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);

		private:
			fft::Nufft _nufft;
			std::vector<int64_t> _nmodes;
		};

		export class CUDASense {
		public:

			CUDASense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);

		private:
			fft::CUDANufft _nufft;
			std::vector<int64_t> _nmodes;
		};


		export class SenseAdjoint {
		public:

			SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);

		private:
			fft::Nufft _nufft;
			std::vector<int64_t> _nmodes;
		};

		export class CUDASenseAdjoint {
		public:

			CUDASenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const at::optional<CoilApplier>& premanip,
				const at::optional<CoilApplier>& postmanip);

		private:
			fft::CUDANufft _nufft;
			std::vector<int64_t> _nmodes;
		};


		export class SenseNormal {
		public:

			SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<CoilApplier>& premanip = at::nullopt,
				const at::optional<CoilApplier>& midmanip = at::nullopt,
				const at::optional<CoilApplier>& postmanip = at::nullopt);

			void apply_forward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt);

			void apply_backward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt);

		private:

			fft::NufftNormal _normal_nufft;
			std::vector<int64_t> _nmodes;

		};

		export class CUDASenseNormal {
		public:

			CUDASenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<CoilApplier>& premanip = at::nullopt,
				const at::optional<CoilApplier>& midmanip = at::nullopt,
				const at::optional<CoilApplier>& postmanip = at::nullopt);

			void apply_forward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt);

			void apply_backward(const at::Tensor& in, at::Tensor& out,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& kspace_storage = at::nullopt,
				const at::optional<at::Tensor>& imspace_storage = at::nullopt);

		private:

			fft::CUDANufftNormal _normal_nufft;
			std::vector<int64_t> _nmodes;
		};


	}
}


