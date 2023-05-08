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

			void apply(const at::Tensor& in, at::Tensor& out, const std::vector<std::reference_wrapper<const at::Tensor>>& smaps,
				std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
				std::optional<std::function<void(at::Tensor&,int)>> freq_manip);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int32_t>& coils,
				std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
				std::optional<std::function<void(at::Tensor&,int)>> freq_manip);

		private:

			NufftNormal _normal_nufft;

		};

		class SenseToeplitzNormal {
		public:

			SenseToeplitzNormal(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps,
				const std::optional<at::Tensor>& storage1, const std::optional<at::Tensor>& storage2);

		private:

			ToeplitzNormalNufft _normal_nufft;
		};


	}


}


