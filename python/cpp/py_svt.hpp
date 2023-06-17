#pragma once

#ifdef _DEBUG
#undef _DEBUG
#include <torch/extension.h>
#define _DEBUG
#else
#include <torch/extension.h>
#endif
#include <torch/library.h>

#include "../../lib/FFI/ffi.hpp"

namespace svt {

	void random_blocks_svt(at::Tensor& input, int64_t nblocks, int64_t block_size, int64_t rank)
	{
		hasty::ffi::random_blocks_svt(input, nblocks, block_size, rank);
	}

    TORCH_LIBRARY(HastySVT, m) {
        m.def("random_blocks_svt", svt::random_blocks_svt);
    }

}
