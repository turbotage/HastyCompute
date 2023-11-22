module;

#include <torch/torch.h>
#include <highfive/H5Easy.hpp>

export module hdf5;

import torch_util;

namespace hasty {

	export at::Tensor import_tensor(const std::string& filepath, const std::string& dataset);

}

