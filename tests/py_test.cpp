
#include <iostream>
#include "../python/cpp/py_sense.hpp"

void diagonal_test() {
	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);

	auto input = at::rand({ 5,1,32,32,32 }, complex_options);
	auto smaps = at::rand({ 20,32,32,32 }, complex_options);
	auto diagonals = at::rand({ 5,64,64,64 }, complex_options);

	std::vector<std::vector<int64_t>> coil_list;
	coil_list.reserve(5);
	for (int i = 0; i < 5; ++i) {
		std::vector<int64_t> coils(20);
		for (int j = 0; j < 20; ++j) {
			coils[j] = j;
		}
		coil_list.emplace_back(std::move(coils));
	}

	bs::batched_sense_toeplitz_diagonals(input, coil_list, smaps, diagonals);

	std::cout << "yas:" << std::endl;
}

int main() {

	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat).device(c10::Device("cuda:0"));
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float).device(c10::Device("cuda:0"));

	auto coord = at::rand({ 3, 30000000 }, real_options);
	auto input = at::rand({ 1, 30000000 }, complex_options);

	auto output = hasty::ffi::nufft1(coord, input, { 1, 256, 256, 256 });

	return 0;
}