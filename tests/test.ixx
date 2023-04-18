

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




void test_torch_speed() {

	int n1 = 256;
	int n2 = 2 * n1;
	//n = 400;

	int nc = 20;

	c10::InferenceMode im;

	auto device = c10::Device(torch::kCUDA);

	auto options = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	at::Tensor ffter = at::rand({ nc, n1,n1,n1 }, options);
	at::Tensor ffter_out = at::empty({ nc, n1,n1,n1 }, options);
	//at::Tensor ffter = at::rand({ 1, n2,n2,n2 }, options);
	at::Tensor ffter_mul = at::rand({ n2,n2,n2 }, options);
	at::Tensor ffter_temp = at::empty({ n2,n2,n2 }, options);

	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < nc; ++i) {
		auto inview = ffter.select(0, i);
		auto outview = ffter_out.select(0, i);

		torch::fft_fftn_out(ffter_temp, inview, { n2, n2, n2 }, c10::nullopt);
		ffter_temp.mul_(ffter_mul);
		torch::fft_ifftn_out(outview, ffter_temp, { n1,n1,n1 }, c10::nullopt);
	}

	//ffter = torch::fft_ifftn(torch::fft_fftn(ffter, { n2,n2,n2 }, { 1,2,3 }) * ffter_mul, {n1,n1,n1}, { 1, 2, 3 });
	//ffter = torch::fft_ifftn(torch::fft_fftn(ffter, c10::nullopt, { 1,2,3 }) * ffter_mul, c10::nullopt, { 1,2,3 });

	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;
}

void test_deterministic_1d() {
	int nfb = 8;
	int nf = nfb;
	int nx = nfb;
	int nt = 1;

	auto device = c10::Device(torch::kCUDA);

	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::empty({ 1,nf }, options2);

	int l = 0;
	for (int x = 0; x < nfb; ++x) {
		float kx = -M_PI + 2 * x * M_PI / nfb;
		coords.index_put_({ 0,l }, kx);
		++l;
	}

	std::cout << "coords: " << coords << std::endl;

	hasty::cuda::Nufft nu2(coords, { nt,nx }, { hasty::cuda::NufftType::eType2, true, 1e-6 });
 
	std::cout << "\n" << (float*)coords.data_ptr() << std::endl;

	auto f = torch::empty({ nt,nf }, options1);
	auto c = torch::ones({ nt,nx }, options1); // *c10::complex<float>(0.0f, 1.0f);

	nu2.apply(c, f);

	torch::cuda::synchronize();

	std::cout << "Nufft Type 2:\nreal:\n" << at::real(f) << "\n\nimag:\n" << at::imag(f) << std::endl;
	std::cout << "Nufft Type 2:\nreal:\n" << at::real(c) << "\n\nimag:\n" << at::imag(c) << std::endl;

	//nu1.apply(f, c);

	//std::cout << "Nufft Type 1: " << at::real(c) << std::endl << at::imag(c) << std::endl;
}


void test_deterministic_3d() {
	int n1 = 3;
	int n2 = 5;
	int n3 = 4;
	int nf = n1 * n2 * n3;
	int nx = nf;
	int nt = 1;

	auto device = c10::Device(torch::kCUDA);

	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::empty({ 3,nf }, options2);

	int l = 0;
	for (int x = 0; x < n1; ++x) {
		for (int y = 0; y < n2; ++y) {
			for (int z = 0; z < n3; ++z) {
				float kx = (2 * x * M_PI / n1) - M_PI;
				float ky = (2 * y * M_PI / n2) - M_PI;
				float kz = (2 * z * M_PI / n3) - M_PI;

				coords.index_put_({ 0,l }, kx);
				coords.index_put_({ 1,l }, ky);
				coords.index_put_({ 2,l }, kz);

				++l;
			}
		}
	}


	hasty::cuda::Nufft nu2(coords, { nt,n3,n2,n1 }, { hasty::cuda::NufftType::eType2, true, 1e-6});

	auto f = torch::empty({ nt,nf }, options1);
	auto c = torch::ones({ nt,n3,n2,n1 }, options1); // *c10::complex<float>(0.0f, 1.0f);

	nu2.apply(c, f);

	std::cout << "Nufft Type 2:\nreal:\n" << at::real(f) << "\n\nimag:\n" << at::imag(f) << std::endl;

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

	hasty::cuda::Nufft nu1(coords, { nt,nx,nx,nx }, {hasty::cuda::NufftType::eType1, true, 1e-5f});
	hasty::cuda::Nufft nu2(coords, { nt,nx,nx,nx }, {hasty::cuda::NufftType::eType2, true, 1e-5f });

	auto f = torch::rand({ nt,nf }, options1);
	auto c = torch::rand({ nt,nx,nx,nx }, options1);

	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 100; ++i) {
		nu2.apply(c, f);
		nu1.apply(f, c);
	}
	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;

}

void test_svd_speed() {
	//test_deterministic();
	int nblock = 1;
	int nx = 16;
	int nframes = 40;

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


	//auto C = A - std::get<0>(tup1)

}


int main() {

	//test_deterministic_3d();
	//test_svd_speed();

	/*
	std::cout << "torch speed:\n";
	for (int i = 0; i < 10; ++i) {
		test_torch_speed();
	}
	*/
	
	std::cout << "cufinufft speed:\n";
	for (int i = 0; i < 10; ++i) {
		test_speed();
	}
	
	

	return 0;
}
