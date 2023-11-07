#pragma once

#include "op.hpp"
#include "../fft/nufft.hpp"

namespace hasty {
	namespace op {

		class NUFFT : public Operator {
		public:

			NUFFT(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _opts;

			std::unique_ptr<nufft::Nufft> _cpunufft;
			std::unique_ptr<nufft::CUDANufft> _cudanufft;
		};

		class NUFFTAdjoint : public Operator {
		public:

			NUFFTAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _opts;

			std::unique_ptr<nufft::Nufft> _cpunufft;
			std::unique_ptr<nufft::CUDANufft> _cudanufft;
		};

		class NUFFTNormal : public Operator {
		public:

			NUFFTNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt,
				at::optional<std::function<void(at::Tensor&)>> func_between = at::nullopt);

			Vector apply(const Vector& in) const;

			void apply_inplace(Vector& in) const override;
			bool has_inplace_apply() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:

			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _forward_opts;
			nufft::NufftOptions _backward_opts;
			at::optional<std::function<void(at::Tensor&)>> _func_between;
			std::unique_ptr<at::Tensor> _storage;

			std::unique_ptr<nufft::NufftNormal> _cpunufft;
			std::unique_ptr<nufft::CUDANufftNormal> _cudanufft;
		};

		class NUFFTNormalAdjoint : public Operator {
		public:

			NUFFTNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt,
				at::optional<std::function<void(at::Tensor&)>> func_between = at::nullopt);

			Vector apply(const Vector& in) const;

			void apply_inplace(Vector& in) const override;
			bool has_inplace_apply() const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const;

		private:

			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _forward_opts;
			nufft::NufftOptions _backward_opts;
			at::optional<std::function<void(at::Tensor&)>> _func_between;
			std::unique_ptr<at::Tensor> _storage;

			std::unique_ptr<nufft::NufftNormal> _cpunufft;
			std::unique_ptr<nufft::CUDANufftNormal> _cudanufft;
		};

		class DCT : public Operator {
		public:



		private:


		};

	}
}
