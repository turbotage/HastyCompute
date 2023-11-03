#pragma once

#include "op.hpp"
#include "../fft/nufft.hpp"
#include "../mri/sense.hpp"

namespace hasty {
	namespace op {

		class Sense : public Operator {
		public:

			Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;
			
		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _opts;

			at::Tensor _smaps;
			std::vector<int64_t> _coils;

			std::unique_ptr<sense::Sense> _cpusense;
			std::unique_ptr<sense::CUDASense> _cudasense;
		};
		
		class SenseH : public Operator {
		public:

			SenseH(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate = false,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

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

		class SenseN : public Operator {
		public:

			SenseN(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt);

			Vector apply(const Vector& in) const;

			Vector apply_forward(const Vector& in) const;

			Vector apply_backward(const Vector& in) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _forward_opts;
			nufft::NufftOptions _backward_opts;

			at::Tensor _smaps;
			std::vector<int64_t> _coils;

			std::unique_ptr<sense::SenseNormal> _cpusense;
			std::unique_ptr<sense::CUDASenseNormal> _cudasense;
		};


	}
}