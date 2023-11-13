#pragma once

#include "../export.hpp"

namespace hasty {
	namespace tests {

		void test_deterministic_1d();

		void test_deterministic_3d();

		void test_speed();

		void test_space_cufinufft();

		void test_svd_speed();

		void compare_normal_methods();

		void test_nufft_speeds(bool toep);

		int test_torch_speed(int n, int nc);

		int test_speed(int n, int nc, int nf = 200000);

		void speed_comparisons_torch_af_cufinufft(int nres, int nrun);

	}
}