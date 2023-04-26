
#include <torch/torch.h>

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <stdexcept>;
import <array>;
import <functional>;
import <optional>;
#endif

import hasty_util;

namespace hasty {
	namespace cuda {

		at::Tensor block_svt(const vec<refw<const at::Tensor>>& tensors, std::pair<vec<i64>, vec<i64>> block);
		
	}
}