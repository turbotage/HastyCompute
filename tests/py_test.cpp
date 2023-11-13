
#include <iostream>

#include "../python/cpp/py_batched_sense.hpp"
#include "../python/cpp/py_sense.hpp"
#include "../python/cpp/py_svt.hpp"


import svt;
import sense;
import batch_sense;


#include "py_test.hpp"

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>

void test_batched_sense()
{
	int32_t nfreq = 500 * 489;
	int32_t outer = 10;
	int32_t ncoil = 16;

	int32_t nx = 64;
	int32_t ny = 64;
	int32_t nz = 64;

	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float);

	std::vector<at::Tensor> coords;
	for (int i = 0; i < outer; ++i) {
		coords.push_back(at::rand({ 3,nfreq }, real_options));
	}
	
	at::Tensor smaps = at::rand({ ncoil, nx, ny, nz }, complex_options);

	at::Tensor image = at::rand({ outer, 1, nx, ny, nz }, complex_options);

	std::vector<at::Tensor> out;
	for (int i = 0; i < outer; ++i) {
		out.push_back(at::empty({ 1,ncoil,nfreq }, complex_options));
	}

	hasty::ffi::BatchedSense bs(at::makeArrayRef(coords), smaps, at::nullopt, at::nullopt, at::nullopt);

	bs.apply(image, at::makeArrayRef(out), at::nullopt);

}

void test_batched_sense_adjoint()
{
	int32_t nfreq = 500 * 489;
	int32_t outer = 10;
	int32_t ncoil = 16;

	int32_t nx = 64;
	int32_t ny = 64;
	int32_t nz = 64;

	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float);

	std::vector<at::Tensor> coords;
	for (int i = 0; i < outer; ++i) {
		coords.push_back(at::rand({ 3,nfreq }, real_options));
	}

	at::Tensor smaps = at::rand({ ncoil, nx, ny, nz }, complex_options);

	at::Tensor out = at::empty({ outer, 1, nx, ny, nz }, complex_options);

	std::vector<at::Tensor> in;
	for (int i = 0; i < outer; ++i) {
		in.push_back(at::rand({ 1,ncoil,nfreq }, complex_options));
	}

	hasty::ffi::BatchedSenseAdjoint bs(at::makeArrayRef(coords), smaps, at::nullopt, at::nullopt, at::nullopt);

	bs.apply(at::makeArrayRef(in), out, at::nullopt);

}

void test_batched_sense_normal()
{
	int32_t nfreq = 500 * 489;
	int32_t outer = 10;
	int32_t ncoil = 16;

	int32_t nx = 64;
	int32_t ny = 64;
	int32_t nz = 64;

	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float);

	std::vector<at::Tensor> coords;
	for (int i = 0; i < outer; ++i) {
		coords.push_back(at::rand({ 3,nfreq }, real_options));
	}

	at::Tensor smaps = at::rand({ ncoil, nx, ny, nz }, complex_options);

	at::Tensor out = at::empty({ outer, 1, nx, ny, nz }, complex_options);
	at::Tensor in = at::rand({ outer, 1, nx, ny, nz }, complex_options);

	hasty::ffi::BatchedSenseNormal bs(at::makeArrayRef(coords), smaps, at::nullopt, at::nullopt, at::nullopt);

	bs.apply(in, out, at::nullopt);
}

void batched_sense_tests() {
	try {
		test_batched_sense();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
	}

	try {
		test_batched_sense_adjoint();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
	}

	try {
		test_batched_sense_normal();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
	}
}

void test_random_svt() {
	int32_t outer = 20;

	int32_t nx = 256;
	int32_t ny = 256;
	int32_t nz = 256;

	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float);

	at::Tensor in = at::rand({ outer, 5, nx, ny, nz }, complex_options);

	std::vector<at::Stream> streams = { at::cuda::getDefaultCUDAStream() };
	/*
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	*/

	auto start = std::chrono::high_resolution_clock::now();

	hasty::ffi::Random3DBlocksSVT  rbsvt(at::make_optional(at::makeArrayRef(streams)));
	rbsvt.apply(in, 10000, { 16,16,16 }, 0.1, true);

	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;

}

void test_normal_svt() {
	int32_t outer = 20;

	int32_t nx = 256;
	int32_t ny = 256;
	int32_t nz = 256;

	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float);

	at::Tensor in = at::rand({ outer, 5, nx, ny, nz }, complex_options);

	std::vector<at::Stream> streams = { at::cuda::getDefaultCUDAStream() };
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	/*
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	streams.push_back(at::cuda::getStreamFromPool());
	*/

	auto start = std::chrono::high_resolution_clock::now();

	hasty::ffi::Normal3DBlocksSVT  rbsvt(at::make_optional(at::makeArrayRef(streams)));
	rbsvt.apply(in, { 16,16,16 }, {16, 16, 16}, 4, 0.1, true);

	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;

}

void svt_tests() {
	try {
		test_random_svt();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
	}

	try {
		test_normal_svt();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
	}
}

void test_nufft_speed() {

}


int main() {

	try {
		test_normal_svt();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
	}


	return 0;
}