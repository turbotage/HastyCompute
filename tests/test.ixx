
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <arrayfire.h>

import hasty_compute;
import solver_cu;
import permute_cu;

import <symengine/expression.h>;
import <symengine/simplify.h>;
import <symengine/parser.h>;

import <chrono>;

void test_torch_speed() {
	int n = 512;

	c10::InferenceMode im;

	auto device = c10::Device(torch::kCUDA);

	auto options = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	at::Tensor tensor = at::rand({ n,n,n }, options);
	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();

	auto fout = torch::fft_fftn(tensor, c10::nullopt, { 0,1,2 });
	
	for (int i = 0; i < 10; ++i) {
		fout = torch::fft_fftn(fout + 1.0f, c10::nullopt, { 0,1,2 });
	}
	

	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	torch::cuda::synchronize();

	start = std::chrono::high_resolution_clock::now();

	fout = torch::fft_fftn(tensor, c10::nullopt, { 0,1,2 });
	
	for (int i = 0; i < 10; ++i) {
		fout = torch::fft_fftn(fout + 1.0f, c10::nullopt, { 0,1,2 });
	}
	

	torch::cuda::synchronize();


	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;
}

void test_af_speed() {

	int n = 512;

	af::array arr1 = af::randu(512, 512, 512, af_dtype::c32);

	arr1.eval();
	af::sync();

	auto start = std::chrono::high_resolution_clock::now();

	auto fout = af::fft3(arr1);
	for (int i = 0; i < 10; ++i) {
		fout = af::fft3(fout + 1.0f);
	}

	fout.eval();
	af::sync();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	fout = af::fft3(arr1);
	for (int i = 0; i < 10; ++i) {
		fout = af::fft3(fout + 1.0f);
	}

	fout.eval();
	af::sync();

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;

}

int main() {

	test_torch_speed();
	test_af_speed();

	return 0;
}
