module;

#include <torch/torch.h>

export module fftop;

import torch_util;
import op;
import nufft;

namespace hasty {
	namespace op {

		export class NUFFT : public Operator {
		public:

			NUFFT(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			sptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			fft::NufftOptions _opts;

			uptr<fft::Nufft> _cpunufft;
			uptr<fft::CUDANufft> _cudanufft;
		};

		export class NUFFTAdjoint : public Operator {
		public:

			NUFFTAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			sptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			fft::NufftOptions _opts;

			uptr<fft::Nufft> _cpunufft;
			uptr<fft::CUDANufft> _cudanufft;
		};

		export class NUFFTNormal : public Operator {
		public:

			NUFFTNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt,
				at::optional<std::function<void(at::Tensor&)>> func_between = at::nullopt);

			Vector apply(const Vector& in) const;

			void apply_inplace(Vector& in) const override;
			bool has_inplace_apply() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		private:

			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			fft::NufftOptions _forward_opts;
			fft::NufftOptions _backward_opts;
			at::optional<std::function<void(at::Tensor&)>> _func_between;
			uptr<at::Tensor> _storage;

			uptr<fft::NufftNormal> _cpunufft;
			uptr<fft::CUDANufftNormal> _cudanufft;
		};

		export class NUFFTNormalAdjoint : public Operator {
		public:

			NUFFTNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<fft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<fft::NufftOptions>& backward_opts = at::nullopt,
				at::optional<std::function<void(at::Tensor&)>> func_between = at::nullopt);

			Vector apply(const Vector& in) const;

			void apply_inplace(Vector& in) const override;
			bool has_inplace_apply() const override;

			sptr<Operator> to_device(at::Stream stream) const;

		private:

			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			fft::NufftOptions _forward_opts;
			fft::NufftOptions _backward_opts;
			at::optional<std::function<void(at::Tensor&)>> _func_between;
			uptr<at::Tensor> _storage;

			uptr<fft::NufftNormal> _cpunufft;
			uptr<fft::CUDANufftNormal> _cudanufft;
		};

		export class DCT : public Operator {
		public:



		private:


		};

	}
}



