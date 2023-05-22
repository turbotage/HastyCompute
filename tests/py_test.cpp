
#include "../python/cpp/py_interface.hpp"

#include <iostream>

/*
void test_llr() {
	auto device_cuda = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto device_cpu = c10::Device(c10::DeviceType::CPU);
	auto options_real_cpu = c10::TensorOptions().device(device_cpu).dtype(c10::ScalarType::Float);
	auto options_complex_cpu = options_real_cpu.dtype(c10::ScalarType::ComplexFloat);

	int nencodes = 5;
	int nframes = 20;
	int nfreq = 400000;
	int nres = 160;
	int ncoil = 32;

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
*/

void batched_sense() {
	auto device_cuda = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto device_cpu = c10::Device(c10::DeviceType::CPU);
	auto options_real_cpu = c10::TensorOptions().device(device_cpu).dtype(c10::ScalarType::Float);
	auto options_complex_cpu = options_real_cpu.dtype(c10::ScalarType::ComplexFloat);

	int inner_batches = 1;
	int outer_batches = 20;
	int nfreq = 400000;
	int nres = 160;
	int ncoil = 20;

	std::vector<at::Tensor> coords;
	std::vector<at::Tensor> kdatas;
	std::vector<std::vector<int64_t>> coils(outer_batches);
	for (int i = 0; i < outer_batches; ++i) {
		coords.push_back(-3.141592f + 2.0f * 3.141592f * at::rand({3,nfreq}, options_real_cpu));
		kdatas.push_back(at::rand({ inner_batches,ncoil,nfreq }, options_complex_cpu));
		std::iota(coils[i].begin(), coils[i].end(), 0);
	}

	auto input = at::rand({ outer_batches,inner_batches,nres,nres,nres }, options_complex_cpu);

	auto smaps = at::rand({ ncoil, nres, nres, nres }, options_complex_cpu);

	std::cout << "starting batched_sense: " << std::endl;
	auto start = std::chrono::steady_clock::now();
	batched_sense(input, coils, smaps, coords, kdatas);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

void batched_sense_toeplitz() {
	auto device_cuda = c10::Device(c10::DeviceType::CUDA, c10::DeviceIndex(0));
	auto device_cpu = c10::Device(c10::DeviceType::CPU);
	auto options_real_cpu = c10::TensorOptions().device(device_cpu).dtype(c10::ScalarType::Float);
	auto options_complex_cpu = options_real_cpu.dtype(c10::ScalarType::ComplexFloat);

	int inner_batches = 1;
	int outer_batches = 20;
	int nfreq = 400000;
	int nres = 160;
	int ncoil = 20;

	std::vector<at::Tensor> coords;
	std::vector<at::Tensor> kdatas;
	std::vector<std::vector<int64_t>> coils(outer_batches);
	for (int i = 0; i < outer_batches; ++i) {
		coords.push_back(-3.141592f + 2.0f * 3.141592f * at::rand({ 3,nfreq }, options_real_cpu));
		kdatas.push_back(at::rand({ inner_batches,ncoil,nfreq }, options_complex_cpu));
		std::iota(coils[i].begin(), coils[i].end(), 0);
	}

	auto input = at::rand({ outer_batches,inner_batches,nres,nres,nres }, options_complex_cpu);

	auto smaps = at::rand({ ncoil, nres, nres, nres }, options_complex_cpu);

	std::cout << "starting batched_sense: " << std::endl;
	auto start = std::chrono::steady_clock::now();
	batched_sense(input, coils, smaps, coords, kdatas);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

int main() {

	batched_sense();

	std::cout << "yas:" << std::endl;

	return 0;
}