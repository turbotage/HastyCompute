#pragma once

#include <torch/extension.h>
#include <torch/library.h>

#include "../../lib/FFI/ffi.hpp"

namespace svt {

	void random_blocks_svt(at::Tensor& input, int64_t nblocks, int64_t block_size, int64_t rank)
	{
		hasty::ffi::random_blocks_svt(input, nblocks, block_size, rank);
	}

}
