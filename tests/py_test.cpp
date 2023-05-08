
#include "../python/py_interface.hpp"

#include <iostream>

void test_llr() {
	auto device_cuda = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto device_cpu = c10::Device(c10::DeviceType::CPU);
	auto options_real_cpu = c10::TensorOptions().device(device_cpu).dtype(c10::ScalarType::Float);
	auto options_complex_cpu = options_real_cpu.dtype(c10::ScalarType::ComplexFloat);

	int nencodes = 5;
	int nframes = 20;
	int nfreq = 400000;
	int nres = 256;
	int ncoil = 20;

	std::cout << "randing coords:" << std::endl;

	auto coords = -3.141592 + 2 * 3.141592 * at::rand({ nframes,nencodes,3,nfreq }, options_real_cpu);

	std::cout << "randing input:" << std::endl;

	auto input = at::rand({ nframes,nencodes,nres,nres,nres }, options_complex_cpu);

	std::cout << "randing kdata:" << std::endl;

	auto kdata = at::rand({ nframes,nencodes,ncoil,nfreq,}, options_complex_cpu);

	std::cout << "randing smaps:" << std::endl;

	auto smaps = at::rand({ ncoil, nres, nres, nres }, options_complex_cpu);

	std::cout << "starting llr: " << std::endl;

	llr(coords, input, smaps, kdata, 1);


}

int main() {

	test_llr();

	std::cout << "yas:" << std::endl;

	return 0;
}