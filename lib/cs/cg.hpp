#pragma once

#include "sense.hpp"
#include "../threading/thread_pool.hpp"

namespace hasty {
	namespace cuda {
		
		using TensorVec = std::vector<at::Tensor>;
		using TensorVecVec = std::vector<TensorVec>;


	}
}

