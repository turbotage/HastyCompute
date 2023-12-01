module;

#include <torch/torch.h>

export module mriop;

import torch_util;
import op;
import nufft;
import sense;

namespace hasty {
	namespace op {

		export class SenseOp : public AdjointableOp {
		public:

			static uptr<SenseOp> Create(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			sptr<AdjointableOp> adjoint() const;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:

			SenseOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			fft::NufftOptions _opts;

			at::Tensor _smaps;
			std::vector<int64_t> _coils;

			uptr<mri::Sense> _cpusense;
			uptr<mri::CUDASense> _cudasense;
		};

		export class SenseHOp : public AdjointableOp {
		public:

			static uptr<SenseHOp> Create(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate = false,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			sptr<AdjointableOp> adjoint() const;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:
			
			SenseHOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate = false,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			fft::NufftOptions _opts;

			bool _accumulate;

			at::Tensor _smaps;
			std::vector<int64_t> _coils;

			uptr<mri::SenseAdjoint> _cpusense;
			uptr<mri::CUDASenseAdjoint> _cudasense;
		};

		export class SenseNOp : public AdjointableOp {
		public:

			static uptr<SenseNOp> Create(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt);

			Vector apply(const Vector& in) const;

			Vector apply_forward(const Vector& in) const;

			Vector apply_backward(const Vector& in) const;

			sptr<AdjointableOp> adjoint() const;

			sptr<Operator> to_device(at::Stream stream) const;

		protected:

			struct SenseNHolder {

				SenseNHolder(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
					const at::Tensor& smaps, const std::vector<int64_t>& coils,
					const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
					const at::optional<fft::NufftOptions>& backward_opts = at::nullopt);

				at::Tensor _coords;
				std::vector<int64_t> _nmodes;
				fft::NufftOptions _forward_opts;
				fft::NufftOptions _backward_opts;

				at::Tensor _smaps;
				std::vector<int64_t> _coils;

				uptr<mri::SenseNormal> _cpusense;
				uptr<mri::CUDASenseNormal> _cudasense;
			};

			SenseNOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt);

			SenseNOp(sptr<SenseNHolder> shoulder);

		private:

			sptr<SenseNHolder> _senseholder;
		};


	}
}

