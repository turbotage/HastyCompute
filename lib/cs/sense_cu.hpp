
#include <torch/torch.h>
#include "../fft/nufft_cu.hpp"

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <stdexcept>;
import <array>;
import <functional>;
import <optional>;
#endif



namespace hasty {

	namespace cuda {

		class Sense {
		public:

		private:
		};

		class SenseNormal {
		public:

			SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, const std::vector<at::Tensor>& smaps, const at::Tensor& out, 
				std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage);

		private:

			NufftNormal _normal_nufft;

		};

	}


}


