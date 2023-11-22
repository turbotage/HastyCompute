#pragma once

#include "py_util.hpp"

namespace hasty {
	namespace mri {
		class Sense;
		class SenseAdjoint;
		class SenseNormal;

		class CUDASense;
		class CUDASenseAdjoint;
		class CUDASenseNormal;
	}

	namespace ffi {

		class LIB_EXPORT Sense : public torch::CustomClassHolder {
		public:

			Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

		private:
			std::unique_ptr<hasty::mri::Sense> _senseop;
			std::unique_ptr<hasty::mri::CUDASense> _cudasenseop;
		};

		class LIB_EXPORT SenseAdjoint : public torch::CustomClassHolder {
		public:

			SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

		private:
			std::unique_ptr<hasty::mri::SenseAdjoint> _senseop;
			std::unique_ptr<hasty::mri::CUDASenseAdjoint> _cudasenseop;
		};

		class LIB_EXPORT SenseNormal : public torch::CustomClassHolder {
		public:

			SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

		private:
			std::unique_ptr<hasty::mri::SenseNormal> _senseop;
			std::unique_ptr<hasty::mri::CUDASenseNormal> _cudasenseop;
		};

	}
}