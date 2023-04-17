

/*
#include <cufinufft.h>
#include <ATen/ATen.h>
*/

#include "../lib/fft/nufft_cu.hpp"

import hasty_util;
import hasty_compute;
import solver_cu;
import permute_cu;

import <symengine/expression.h>;
import <symengine/simplify.h>;
import <symengine/parser.h>;

import <chrono>;



/*
void test_torch_speed() {

	int n1 = 256;
	int n2 = 2 * n1;
	//n = 400;

	c10::InferenceMode im;

	auto device = c10::Device(torch::kCUDA);

	auto options = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	at::Tensor ffter = at::rand({ 1, n1,n1,n1 }, options);
	//at::Tensor ffter = at::rand({ 1, n2,n2,n2 }, options);
	at::Tensor ffter_mul = at::rand({ 1, n2,n2,n2 }, options);
	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();

	ffter = torch::fft_ifftn(torch::fft_fftn(ffter, { n2,n2,n2 }, { 1,2,3 }) * ffter_mul, {n1,n1,n1}, { 1, 2, 3 });
	//ffter = torch::fft_ifftn(torch::fft_fftn(ffter, c10::nullopt, { 1,2,3 }) * ffter_mul, c10::nullopt, { 1,2,3 });

	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;
}
*/


void test_deterministic() {
	int nfb = 2;
	int nf = nfb * nfb * nfb;
	int nx = nfb;
	int nt = 1;

	auto device = c10::Device(torch::kCUDA);

	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::empty({ 3,nf }, options2);

	int l = 0;
	for (int x = 0; x < nfb; ++x) {
		for (int y = 0; y < nfb; ++y) {
			for (int z = 0; z < nfb; ++z) {
				float kx = -M_PI + x * M_PI / nfb;
				float ky = -M_PI + y * M_PI / nfb;
				float kz = -M_PI + z * M_PI / nfb;

				coords.index_put_({ 0,l }, kx);
				coords.index_put_({ 1,l }, ky);
				coords.index_put_({ 2,l }, kz);

				++l;
			}
		}
	}


	hasty::cuda::Nufft nu2(coords, { nt,nx,nx,nx }, hasty::cuda::NufftType::eType2, {true, 1e6});

	auto f = torch::empty({ nt,nf }, options1);
	auto c = torch::ones({ nt,nx,nx,nx }, options1);

	nu2.apply(c, f);

	std::cout << "Nufft Type 2: " << at::real(f) << std::endl << at::imag(f) << std::endl;

	//nu1.apply(f, c);

	//std::cout << "Nufft Type 1: " << at::real(c) << std::endl << at::imag(c) << std::endl;
}

void test_speed() {
	int nfb = 256;
	int nf = 200000;
	int nx = nfb;
	int nt = 1;

	auto device = c10::Device(torch::kCUDA);
	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);
	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::rand({ 3,nf }, options2);

	hasty::cuda::Nufft nu1(coords, { nt,nx,nx,nx }, hasty::cuda::NufftType::eType1, {true, 1e-5f});
	hasty::cuda::Nufft nu2(coords, { nt,nx,nx,nx }, hasty::cuda::NufftType::eType2, { true, 1e-5f });

	auto f = torch::rand({ nt,nf }, options1);
	auto c = torch::rand({ nt,nx,nx,nx }, options1);

	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();
	nu2.apply(c, f);
	nu1.apply(f, c);
	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;

}


int main() {

	//test_deterministic();
	int nblock = 1;
	int nx = 16;
	int nframes = 100;
	 
	bool full_svd = false;
	bool nruns = 10;

	auto device = c10::Device(torch::kCUDA);
	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);
	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	torch::cuda::synchronize();
	auto start = std::chrono::high_resolution_clock::now();
	auto A = torch::rand({ nblock, nx * nx * nx * 5, nframes }, options1);
	torch::cuda::synchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "rand 1: " << duration.count() << std::endl;


	//at::Tensor U;
	//at::Tensor Vh;
	//at::Tensor S;

	auto tup1 = torch::linalg::svd(A, full_svd, c10::nullopt);

	torch::cuda::synchronize();

	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i) {
		tup1 = torch::linalg::svd(A, full_svd, c10::nullopt);
	}
	torch::cuda::synchronize();

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "SVD 1: " << duration.count() << std::endl;

	torch::cuda::synchronize();
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i) {
		tup1 = torch::linalg::svd(A, full_svd, c10::nullopt);
	}
	torch::cuda::synchronize();

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "SVD 2: " << duration.count() << std::endl;






	/*
	for (int i = 0; i < 10; ++i) {
		test_speed();
	}

	for (int i = 0; i < 10; ++i) {
		test_torch_speed();
	}
	*/

	return 0;
}
