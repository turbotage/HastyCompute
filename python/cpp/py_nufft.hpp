#pragma once

#include "py_util.hpp"
#include "py_interface.hpp"

namespace hasty {
	namespace ffi {

		class LIB_EXPORT NufftOptions : public torch::CustomClassHolder {
		public:

			NufftOptions(int64_t type, const at::optional<bool>& positive, const at::optional<double>& tol);

			const nufft::NufftOptions& getOptions() const;

		private:
			std::unique_ptr<nufft::NufftOptions> _opts;
		};

		class LIB_EXPORT Nufft : public torch::CustomClassHolder {
		public:

			Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const ffi::NufftOptions& opts);

			void apply(const at::Tensor& in, at::Tensor out) const;

		private:
			std::unique_ptr<nufft::Nufft> _nufftop;
		};

		
		class LIB_EXPORT NufftNormal : public torch::CustomClassHolder {
		public:

			NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
				const ffi::NufftOptions& forward, const ffi::NufftOptions& backward);

			void apply(const at::Tensor& in, at::Tensor out, at::Tensor storage,
				const at::optional<FunctionLambda>& func_between) const;

		private:
			std::unique_ptr<nufft::NufftNormal> _nufftop;
		};
		

		/*
		ADD TOEPLITZ LATER
		*/

	}
}