
#define INCLUDE_TESTS
#define INCLUDE_FFI
#include "../lib/hasty.hpp"

int main() {
	
	//compare_normal_methods();

	hasty::tests::speed_comparisons_torch_af_cufinufft(10);

	return 0;
}
