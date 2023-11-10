module;

#include "../torch_util.hpp"

export module mriop;

import op;
import nufft;
import sense;

namespace hasty {
	namespace op {

		export class SenseOp : public AdjointableOp {
		public:

			static std::unique_ptr<SenseOp> Create(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			std::shared_ptr<AdjointableOp> adjoint() const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:

			SenseOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _opts;

			at::Tensor _smaps;
			std::vector<int64_t> _coils;

			std::unique_ptr<sense::Sense> _cpusense;
			std::unique_ptr<sense::CUDASense> _cudasense;
		};

		export class SenseHOp : public AdjointableOp {
		public:

			SenseHOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate = false,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			std::shared_ptr<AdjointableOp> adjoint() const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _opts;

			bool _accumulate;

			at::Tensor _smaps;
			std::vector<int64_t> _coils;

			std::unique_ptr<sense::SenseAdjoint> _cpusense;
			std::unique_ptr<sense::CUDASenseAdjoint> _cudasense;
		};

		export class SenseNOp : public AdjointableOp {
		public:

			SenseNOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt);

			Vector apply(const Vector& in) const;

			Vector apply_forward(const Vector& in) const;

			Vector apply_backward(const Vector& in) const;

			std::shared_ptr<AdjointableOp> adjoint() const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		protected:

			struct SenseNHolder {

				SenseNHolder(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
					const at::Tensor& smaps, const std::vector<int64_t>& coils,
					const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
					const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt);

				at::Tensor _coords;
				std::vector<int64_t> _nmodes;
				nufft::NufftOptions _forward_opts;
				nufft::NufftOptions _backward_opts;

				at::Tensor _smaps;
				std::vector<int64_t> _coils;

				std::unique_ptr<sense::SenseNormal> _cpusense;
				std::unique_ptr<sense::CUDASenseNormal> _cudasense;
			};

			SenseNOp(std::shared_ptr<SenseNHolder> shoulder);

		private:

			std::shared_ptr<SenseNHolder> _senseholder;
		};


	}
}

