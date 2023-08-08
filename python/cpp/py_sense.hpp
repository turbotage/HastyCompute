#pragma once

#include "py_util.hpp"

namespace hasty {
	namespace ffi {

		class LIB_EXPORT Sense : public torch::CustomClassHolder {
		public:

			Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

		private:
			std::unique_ptr<hasty::sense::Sense> _senseop;
		};

		class LIB_EXPORT SenseAdjoint : public torch::CustomClassHolder {
		public:

			SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

		private:
			std::unique_ptr<hasty::sense::SenseAdjoint> _senseop;
		};

		class LIB_EXPORT SenseNormal : public torch::CustomClassHolder {
		public:

			SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

		private:
			std::unique_ptr<hasty::sense::SenseNormal> _senseop;
		};

		class LIB_EXPORT SenseNormalAdjoint : public torch::CustomClassHolder {
		public:

			SenseNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

			void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
				const at::optional<at::Tensor>& imspace_storage);

		private:
			std::unique_ptr<hasty::sense::SenseNormalAdjoint> _senseop;
		};

	}
}