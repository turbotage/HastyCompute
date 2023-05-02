
#include "../python/py_interface.hpp"

#include <iostream>

int main() {

	auto device = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));

	auto options = c10::TensorOptions().device(device).dtype(c10::kFloat);

	auto coords = at::rand({ 3,100 }, options);

	auto input = at::rand({ 1,256,256,256 }, options.dtype(c10::kComplexFloat));

	auto out = nufft2to1(coords, input);

	std::cout << "yas:" << std::endl;

	return 0;
}