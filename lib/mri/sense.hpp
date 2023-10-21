#pragma once

#include "../fft/nufft.hpp"
#include "../threading/thread_pool.hpp"

#include "../export.hpp"

namespace hasty {

	using TensorVec = std::vector<at::Tensor>;

	namespace sense {

		using CoilApplier = std::function<void(at::Tensor&, int32_t)>;
		struct CoilManipulator {
			CoilManipulator() = default;

			CoilManipulator& setPreApply(const CoilApplier& apply) {
				if (preapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				preapplier = std::make_optional(apply);
				return *this;
			}

			CoilManipulator& setMidApply(const CoilApplier& apply) {
				if (midapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				midapplier = std::make_optional(apply);
				return *this;
			}

			CoilManipulator& setPostApply(const CoilApplier& apply) {
				if (postapplier.has_value())
					throw std::runtime_error("Tried to set non nullopt applier");
				postapplier = std::make_optional(apply);
				return *this;
			}

			std::optional<CoilApplier> preapplier;
			std::optional<CoilApplier> midapplier;
			std::optional<CoilApplier> postapplier;
		};

		class LIB_EXPORT Sense {
		public:

			Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const std::optional<CoilApplier>& premanip,
				const std::optional<CoilApplier>& postmanip);

		private:
			nufft::Nufft _nufft;
			std::vector<int64_t> _nmodes;
		};

		class LIB_EXPORT CUDASense {
		public:

			CUDASense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const std::optional<CoilApplier>& premanip,
				const std::optional<CoilApplier>& postmanip);

		private:
			nufft::CUDANufft _nufft;
			std::vector<int64_t> _nmodes;
		};


		class LIB_EXPORT SenseAdjoint {
		public:

			SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const std::optional<CoilApplier>& premanip,
				const std::optional<CoilApplier>& postmanip);

		private:
			nufft::Nufft _nufft;
			std::vector<int64_t> _nmodes;
		};

		class LIB_EXPORT CUDASenseAdjoint {
		public:

			CUDASenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const std::optional<CoilApplier>& premanip,
				const std::optional<CoilApplier>& postmanip);

		private:
			nufft::CUDANufft _nufft;
			std::vector<int64_t> _nmodes;
		};


		class LIB_EXPORT SenseNormal {
		public:

			SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const std::optional<CoilApplier>& premanip,
				const std::optional<CoilApplier>& midmanip,
				const std::optional<CoilApplier>& postmanip);

		private:

			nufft::NufftNormal _normal_nufft;
			std::vector<int64_t> _nmodes;

		};

		class LIB_EXPORT CUDASenseNormal {
		public:

			CUDASenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
				const std::optional<CoilApplier>& premanip,
				const std::optional<CoilApplier>& midmanip,
				const std::optional<CoilApplier>& postmanip);

		private:

			nufft::CUDANufftNormal _normal_nufft;
			std::vector<int64_t> _nmodes;

		};


	}
}


