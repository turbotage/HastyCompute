
#define INCLUDE_TESTS
#include "../lib/hasty.hpp"

int main() {
	
	//compare_normal_methods();

	hasty::tests::speed_comparisons_torch_af_cufinufft(10);

	return 0;
}
