#pragma once

#include "../ffi_defines.hpp"

namespace hasty {
	namespace tests {

		LIB_EXPORT void test_deterministic_1d();

		LIB_EXPORT void test_deterministic_3d();

		LIB_EXPORT void test_speed();

		LIB_EXPORT void test_space_cufinufft();

		LIB_EXPORT void test_svd_speed();

		LIB_EXPORT void compare_normal_methods();

		LIB_EXPORT void test_nufft_speeds(bool toep);

		LIB_EXPORT void test_conway();

		LIB_EXPORT int test_torch_speed(int n, int nc);

		LIB_EXPORT int test_af_speed(int n, int nc);

		LIB_EXPORT int test_speed(int n, int nc, int nf = 200000);

		LIB_EXPORT void speed_comparisons_torch_af_cufinufft(int nres, int nrun);

		LIB_EXPORT void test_expression();

	}
}