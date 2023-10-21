#pragma once

#include "op.hpp"
#include "../fft/nufft.hpp"
#include "../mri/sense.hpp"

namespace hasty {
	namespace op {

		class SENSE : public Operator {
		public:

			SENSE(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;
			
		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _opts;

			std::unique_ptr<sense::Sense> _cpusense;
			std::unique_ptr<sense::CUDASense> _cudasense;
		};
		
		class SENSE_H : public Operator {
		public:

			SENSE_H(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate = false,
				const at::optional<nufft::NufftOptions>& opts = at::nullopt);

			Vector apply(const Vector& in) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _opts;

			std::unique_ptr<sense::SenseAdjoint> _cpusense;
			std::unique_ptr<sense::CUDASenseAdjoint> _cudasense;
		};

		class SENSE_N : public Operator {
		public:

			SENSE_N(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
				const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<nufft::NufftOptions>& forward_opts = at::nullopt,
				const at::optional<nufft::NufftOptions>& backward_opts = at::nullopt);

			Vector apply(const Vector& in) const;

		private:
			at::Tensor _coords;
			std::vector<int64_t> _nmodes;
			nufft::NufftOptions _forward_opts;
			nufft::NufftOptions _backward_opts;

			std::unique_ptr<sense::SenseNormal> _cpusense;
			std::unique_ptr<sense::CUDASenseNormal> _cudasense;
		};

	}
}