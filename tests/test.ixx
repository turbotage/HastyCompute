
#include <torch/torch.h>

import hasty_compute;
import solver_cu;
import permute_cu;

import <symengine/expression.h>;
import <symengine/simplify.h>;
import <symengine/parser.h>;

import <chrono>;


int main() {

	int n = 512;

	auto device = torch::Device(torch::kCUDA);

	auto options = c10::TensorOptions().device(device);

	torch::Tensor tensor = torch::rand({ n,n,n }, options);
	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();

	auto fout = torch::fft_fftn(tensor, c10::nullopt, { 0,1,2 });
	for (int i = 0; i < 10; ++i) {
		fout = torch::fft_fftn(fout, c10::nullopt, { 0,1,2 });
	}

	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	torch::cuda::synchronize();

	start = std::chrono::high_resolution_clock::now();

	fout = torch::fft_fftn(tensor, c10::nullopt, { 0,1,2 });
	for (int i = 0; i < 10; ++i) {
		fout = torch::fft_fftn(fout, c10::nullopt, { 0,1,2 });
	}

	torch::cuda::synchronize();
	
	

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;

	return 0;
}
