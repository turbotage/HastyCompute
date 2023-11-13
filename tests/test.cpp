
#define INCLUDE_TESTS
#include "../lib/hasty.hpp"

#include <chrono>
#include <iostream>

int main() {
	
	std::cout << "funcexprs:\n";
	hasty::tests::test_funcexprs();
	std::cout << "first_derivatives:\n";
	hasty::tests::test_first_derivatives();
	std::cout << "second_derivatives:\n";
	hasty::tests::test_second_derivatives();

	return 0;
}
