#pragma once

#include "py_util.hpp"

namespace hasty {

	namespace op {
		class Admm;
	}

	namespace ffi {

		class LIB_EXPORT FivePointNuclearNormAdmm : public torch::CustomClassHolder {
		public:

			FivePointNuclearNormAdmm(std::vector<at::Tensor> coords, std::vector<int64_t> nmodes,
				std::vector<at::Tensor> kdata, at::Tensor smaps, at::optional<std::vector<at::Tensor>> preconds);



		private:
			std::unique_ptr<op::Admm> _admm;
		};

	}

}

