#pragma once

#include "op.hpp"
#include "../fft/nufft.hpp"
#include "../mri/sense.hpp"

namespace hasty {
	namespace op {

		class NUFFT : public Operator {
		public:

			NUFFT(const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
				const std::optional<nufft::NufftOptions>& opts = std::nullopt);

		private:

		};

	}
}
