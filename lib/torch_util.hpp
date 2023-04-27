#pragma once

#include <ATen/ATen.h>

namespace hasty {

	namespace torch_util {
		std::stringstream print_4d_xyz(const at::Tensor& toprint);

		std::vector<int64_t> nmodes_from_tensor(const at::Tensor& tensor);

	}

}

