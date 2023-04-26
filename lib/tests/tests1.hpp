#pragma once


#if defined(_WIN32)
#define LIB_EXPORT __declspec(dllexport)
#define LIB_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define LIB_EXPORT __attribute_((visibility("default")))
#define LIB_IMPORT
#else
#define LIB_EXPORT
#define LIB_IMPORT
#pragma warning Unknown dynamic link import/export semantics
#endif

namespace hasty {
	namespace tests {

		void test_deterministic_1d();

		void test_deterministic_3d();

		void test_speed();

		void test_space_cufinufft();

		void test_svd_speed();

		void compare_normal_methods();

		void test_nufft_speeds(bool toep);

		void test_conway();

		int test_torch_speed(int n, int nc);

		int test_af_speed(int n, int nc);

		int test_speed(int n, int nc, int nf = 200000);

		LIB_EXPORT void speed_comparisons_torch_af_cufinufft(int nrun);

	}
}