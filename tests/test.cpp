
#define INCLUDE_TESTS
#define INCLUDE_FFI
#include "../lib/hasty.hpp"

#include <chrono>

int main() {
	
	auto device_cuda = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto device_cpu = c10::Device(c10::DeviceType::CPU);
	auto options_real_cpu = c10::TensorOptions().device(device_cpu).dtype(c10::ScalarType::Float);
	auto options_complex_cpu = options_real_cpu.dtype(c10::ScalarType::ComplexFloat);
	auto options_real_gpu = c10::TensorOptions().device(device_cuda).dtype(c10::ScalarType::Float);
	auto options_complex_gpu = options_real_gpu.dtype(c10::ScalarType::ComplexFloat);

	int nfreq = 100000;
	int nres = 256;

	auto coords = -3.141592 + 2 * 3.141592 * at::rand({ 3,nfreq }, options_real_gpu);
	auto input = at::rand({ 10,nres,nres,nres }, options_complex_gpu);

	
	auto output = hasty::ffi::nufft2(coords, input);

	torch::cuda::synchronize();
	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < 10; ++i) {
		output = hasty::ffi::nufft2(coords, input);
	}
	torch::cuda::synchronize();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return 0;
}
