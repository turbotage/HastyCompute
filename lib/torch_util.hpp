#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

namespace hasty {

	namespace torch_util {

		std::vector<c10::Stream> get_streams(const at::optional<std::vector<c10::Stream>>& streams);

		std::stringstream print_4d_xyz(const at::Tensor& toprint);

		std::vector<int64_t> nmodes_from_tensor(const at::Tensor& tensor);

	}

}

