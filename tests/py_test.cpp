
#include "../python/py_interface.hpp"

#include <iostream>

void test_llr() {
	auto device_cuda = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto device_cpu = c10::Device(c10::DeviceType::CPU);
	auto options_real_cpu = c10::TensorOptions().device(device_cpu).dtype(c10::ScalarType::Float);
	auto options_complex_cpu = options_real_cpu.dtype(c10::ScalarType::ComplexFloat);

	int nencodes = 3;
	int nframes = 5;
	int nfreq = 100000;
	int nres = 120;
	int ncoil = 12;

	auto coords = -3.141592 + 2 * 3.141592 * at::rand({ nframes,nencodes,3,nfreq }, options_real_cpu);

	auto input = at::rand({ nframes,nencodes,nres,nres,nres }, options_complex_cpu);

	auto kdata = at::rand({ nframes,nencodes,ncoil,nfreq,}, options_complex_cpu);

	auto smaps = at::rand({ ncoil, nres, nres, nres }, options_complex_cpu);

	llr(coords, input, smaps, kdata, 5);


}

int main() {

	test_llr();

	std::cout << "yas:" << std::endl;

	return 0;
}