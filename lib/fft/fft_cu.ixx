module;

#include <cufinufft.h>
#include <ATen/ATen.h>

export module fft_cu;

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <array>;
import <stdexcept>;
#endif

import hasty_util;
import hasty_compute;
import hasty_cu;

using namespace hasty::util;

namespace hasty {

	namespace cuda {

		export struct NufftOptions {
			i32 ntransf = 1;
			bool positive = true;
			double tol = 1e-6;

			int positive() { return positive ? 1 : -1; }
		};

		/// <summary>
		/// Nonuniform to uniform, also known as "adjoint" NUFFT
		/// </summary>
		export class NufftType1 {
		public:

			NufftType1(at::Tensor&& coords, const std::array<i32,1>& nmodes, const NufftOptions& opts = NufftOptions{})
				: _coords(coords), _opts(opts)
			{
				_nmodes[0] = nmodes[0];

				i8f ndim = _coords.size(0);
				if (ndim != 1) 
					throw std::runtime_error("Constructed as 1D Type-1 Nufft but 3D coords were passed");

				if (_nmodes[0] == _coords.size(1))
					throw std::runtime_error("nmodes[0] did not match coords.size(1)");

				if (!_coords.is_contiguous()) {
					throw std::runtime_error("coords must be contiguous");
				}

				if (cufinufft_makeplan(1, ndim, _nmodes.data(), _opts.positive(), _opts.ntransf, (float)_opts.tol, 0, &_plan, nullptr))
					throw std::runtime_error("cufinufft_makeplan failed");

				if (cufinufft_setpts(_nmodes[0], (float*)_coords.data_ptr(), ))

			}

			~NufftType1() {

			}

		private:

			at::Tensor _coords;
			std::array<i32, 3> _nmodes;
			NufftOptions _opts;

			cufinufft_plan_s _plan;


		};

		/// <summary>
		/// Uniform to nonuniform, also known as "forward" NUFFT
		/// </summary>
		export class NufftType2 {
		public:

		private:

		};


	}
}