
#include <iostream>
#include "../python/cpp/py_sense.hpp"

#include "py_test.hpp"



void diagonal_test() {
	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);

	auto input = at::rand({ 5,1,32,32,32 }, complex_options);
	auto smaps = at::rand({ 20,32,32,32 }, complex_options);
	auto diagonals = at::rand({ 5,64,64,64 }, complex_options);

	std::vector<std::vector<int64_t>> coil_list;
	coil_list.reserve(5);
	for (int i = 0; i < 5; ++i) {
		std::vector<int64_t> coils(20);
		for (int j = 0; j < 20; ++j) {
			coils[j] = j;
		}
		coil_list.emplace_back(std::move(coils));
	}

	bs::batched_sense_toeplitz_diagonals(input, coil_list, smaps, diagonals);

	std::cout << "yas:" << std::endl;
}

void diagonal_ones_gives_shs() 
{
	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat); // .device(c10::Device("cuda:0"));
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float); // .device(c10::Device("cuda:0"));

	int nenc = 1;
	int ncoils = 4;
	int nx = 2;


	auto input = at::ones({ nenc,1,nx,nx,nx }, complex_options);
	auto smaps = at::rand({ ncoils,nx,nx,nx }, complex_options);
	auto diagonals = at::ones({ nenc,2 * nx,2 * nx,2 * nx }, real_options);

	std::vector<std::vector<int64_t>> coil_list;
	coil_list.reserve(nenc);
	for (int i = 0; i < nenc; ++i) {
		std::vector<int64_t> coils(ncoils);
		for (int j = 0; j < ncoils; ++j) {
			coils[j] = j;
		}
		coil_list.emplace_back(std::move(coils));
	}

	bs::batched_sense_toeplitz_diagonals(input, coil_list, smaps, diagonals);

	std::cout << input << std::endl;

	std::cout << (smaps.conj() * smaps).sum(0) << std::endl;

	std::cout << "yas:" << std::endl;
}

void nufft_tests()
{
	c10::InferenceMode im_guard{};

	auto complex_options = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat); // .device(c10::Device("cuda:0"));
	auto real_options = c10::TensorOptions().dtype(c10::ScalarType::Float); // .device(c10::Device("cuda:0"));

	c10::Device cudev("cuda:0");

	int nenc = 1;
	int ncoils = 4;
	int nx = 16;
	int ny = 16;
	int nz = 13;
	int nf = nx * ny * nz;

	using namespace at::indexing;
	auto coord = at::rand({ 3,nf }, real_options);
	bool uniform_sampling = true;
	if (uniform_sampling) {
		int l = 0;
		for (int i = 0; i < nx; ++i) {
			for (int j = 0; j < ny; ++j) {
				for (int k = 0; k < nz; ++k) {
					float kx = -M_PI + 2 * M_PI * i / nx;
					float ky = -M_PI + 2 * M_PI * j / ny;
					float kz = -M_PI + 2 * M_PI * k / nz;
				 
					coord.index_put_({ 0,l }, kx);
					coord.index_put_({ 1,l }, ky);
					coord.index_put_({ 2,l }, kz);
					++l;
				}
			}
		}
	}
	coord = coord.to(cudev);
	
	auto input1 = at::ones({ 1,nx,ny,nz }, complex_options.device(cudev));
	//auto input2 = input1.detach().clone();
	
	auto back1 = (nufft::nufft21(coord, input1) / nf).cpu();
	//auto back2 = (nufft::nufft1(coord, nufft::nufft2(coord, input2), {1, nx, ny, nz})).cpu();

	auto b1mean = at::mean(back1);
	//auto b2mean = at::mean(back2);

	std::cout << at::real(b1mean) << " " << at::imag(b1mean) << std::endl;
	//std::cout << at::real(b2mean) << " " << at::imag(b2mean) << std::endl;

	//plot1_1(coord.cpu(), -1);
	plot2_1(back1, input1.cpu(), 0);
}



int main() {

	nufft_tests();

	return 0;
}