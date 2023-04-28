#pragma once

#include "../fft/nufft_cu.hpp"

namespace hasty {

	namespace cuda {

		class Sense {
		public:

		private:
		};

		class SenseNormal {
		public:

			SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, const std::vector<std::reference_wrapper<const at::Tensor>>& smaps, const at::Tensor& out,
				std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
				std::optional<std::function<void(at::Tensor&)>> freq_manip);

		private:

			NufftNormal _normal_nufft;

		};

	}


}


