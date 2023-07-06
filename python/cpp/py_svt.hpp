#pragma once

#include "py_util.hpp"

namespace svt {

	void random_blocks_svt(at::Tensor& input, int64_t nblocks, int64_t block_size, 
		double thresh, bool soft, const at::optional<std::vector<c10::Stream>>& streams)
	{
		//std::cout << nblocks << " " << block_size << " " << thresh << " " << soft << std::endl;

		hasty::ffi::random_blocks_svt(input, nblocks, block_size, thresh, soft, hasty::ffi::get_streams(streams));
	}

    TORCH_LIBRARY(HastySVT, m) {
        m.def("random_blocks_svt", svt::random_blocks_svt);
    }

}
