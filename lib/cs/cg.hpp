#pragma once

#include "../torch_util.hpp"
#include "sense.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {
	namespace cuda {
		
		using TensorVec = std::vector<at::Tensor>;
		using TensorVecVec = std::vector<TensorVec>;


	}
}

