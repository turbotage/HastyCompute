#pragma once

#include "nufft.hpp"
#include "cs.hpp"
#include "batch_sense.hpp"

#ifdef INCLUDE_TESTS
#include "tests/tests1.hpp"
#include "tests/tests2.hpp"
#endif

namespace hasty {
	namespace ffi {

		LIB_EXPORT std::vector<c10::Stream> get_streams(const at::optional<std::vector<c10::Stream>>& streams);

	}
}