
#include <iostream>
#include "../python/cpp/py_batched_sense.hpp"
#include "../python/cpp/py_sense.hpp"
#include "../python/cpp/py_svt.hpp"

#include "py_test.hpp"

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>

/*
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

	bs::batched_sense_toeplitz_diagonals(input, coil_list, smaps, diagonals, at::nullopt);

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

	bs::batched_sense_toeplitz_diagonals(input, coil_list, smaps, diagonals, at::nullopt);

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

void batched_test() {
	at::Tensor inp = at::rand({ 5,1,128,128,128 }, c10::ScalarType::ComplexFloat);
	at::Tensor smaps = at::rand({ 32,128,128,128 }, c10::ScalarType::ComplexFloat);
	at::Tensor diagonals = at::rand({ 5,256,256,256 }, c10::ScalarType::ComplexFloat);

	std::vector<std::vector<int64_t>> coils;
	for (int i = 0; i < 5; ++i) {
		auto& inner_coils = coils.emplace_back();
		for (int j = 0; j < 32; ++j) {
			inner_coils.emplace_back(j);
		}
	}

	bs::batched_sense_toeplitz_diagonals(inp, coils, smaps, diagonals, at::nullopt);
}

void batched_sense_test() 
{

	int nx = 32;
	int nf = 100000;
	at::Tensor inp1 = at::rand({ 1,1,nx,nx,nx }, c10::ScalarType::ComplexFloat);
	at::Tensor inp2 = inp1.detach().clone();

	at::Tensor smaps = at::rand({ 32,nx,nx,nx }, c10::ScalarType::ComplexFloat);
	at::Tensor weights = at::ones({ 1,nf }, c10::ScalarType::ComplexFloat);
	at::Tensor coord = at::rand({ 3,nf }, c10::ScalarType::Float);
	at::Tensor kdata = at::rand({ 1,32,nf }, c10::ScalarType::Float);

	std::vector<std::vector<int64_t>> coils;
	for (int i = 0; i < 5; ++i) {
		auto& inner_coils = coils.emplace_back();
		for (int j = 0; j < 32; ++j) {
			inner_coils.emplace_back(j);
		}
	}

	std::vector<at::Tensor> weights_vec = { weights };
	std::vector<at::Tensor> coord_vec = { coord };
	std::vector<at::Tensor> kdata_vec = { coord };

	bs::batched_sense_normal(inp1, at::nullopt, smaps, coord_vec, at::nullopt);
	bs::batched_sense_normal_weighted(inp2, at::nullopt, smaps, coord_vec, weights_vec, at::nullopt);

	std::cout << "Relerr: " << torch::norm(inp1 - inp2) / torch::norm(inp1) << std::endl;

	bs::batched_sense_normal_kdata(inp1, at::nullopt, smaps, coord_vec, kdata_vec, at::nullopt);
	bs::batched_sense_normal_weighted_kdata(inp2, at::nullopt, smaps, coord_vec, weights_vec, kdata_vec, at::nullopt);

	std::cout << "Relerr: " << torch::norm(inp1 - inp2) / torch::norm(inp1) << std::endl;
}

void test_large_batched() {

	int nx = 256;
	int nf = 300000;
	int nt = 75;
	at::Tensor inp1 = at::empty({ nt,1,nx,nx,nx }, c10::ScalarType::ComplexFloat);

	at::Tensor smaps = at::empty({ 32,nx,nx,nx }, c10::ScalarType::ComplexFloat);
	at::Tensor coord = -3.141592f + 2*3.141592f*at::rand({ 3,nf }, c10::ScalarType::Float);
	at::Tensor kdata = at::empty({ 1,32,nf }, c10::ScalarType::ComplexFloat);

	std::vector<std::vector<int64_t>> coils;
	for (int i = 0; i < nt; ++i) {
		auto& inner_coils = coils.emplace_back();
		for (int j = 0; j < 32; ++j) {
			inner_coils.emplace_back(j);
		}
	}

	//std::vector<at::Tensor> weights_vec = { weights };
	std::vector<at::Tensor> coord_vec;
	std::vector<at::Tensor> kdata_vec;
	for (int i = 0; i < nt; ++i) {
		coord_vec.push_back(coord.detach().clone());
		kdata_vec.push_back(kdata.detach().clone());
	}

	bs::batched_sense_normal_kdata(inp1, at::nullopt, smaps, coord_vec, kdata_vec, at::nullopt);

}

void random_svt_test() {

	auto image = at::rand({ 30,5,64,64,64 }, c10::ScalarType::ComplexFloat);

	svt::random_blocks_svt(image, 5000, 16, 0.05, true, at::nullopt);

}

void normal_svt_test() {

	auto image = at::rand({ 30,5,64,64,64 }, c10::ScalarType::ComplexFloat);

	svt::normal_blocks_svt(image, {16,16,16}, {16,16,16}, 4, 0.05, true, at::nullopt);
}
*/


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

void test_batched_sense_normal_adjoint()
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

	std::vector<at::Tensor> in;
	for (int i = 0; i < outer; ++i) {
		in.push_back(at::rand({ 1,ncoil,nfreq }, complex_options));
	}
	std::vector<at::Tensor> out;
	for (int i = 0; i < outer; ++i) {
		out.push_back(at::empty({ 1,ncoil,nfreq }, complex_options));
	}

	hasty::ffi::BatchedSenseNormalAdjoint bs(at::makeArrayRef(coords), smaps, at::nullopt, at::nullopt, at::nullopt, at::nullopt);

	bs.apply(at::makeArrayRef(in), at::makeArrayRef(out), at::nullopt);
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

	try {
		test_batched_sense_normal_adjoint();
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