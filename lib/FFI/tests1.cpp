#include "tests1.hpp"

#include "../fft/nufft.hpp"

import hasty_util;
import hasty_compute;
import solver;
import permute;

#include <iostream>

void hasty::tests::test_deterministic_1d() {
	int nfb = 8;
	int nf = nfb;
	int nx = nfb;
	int nt = 1;

	auto device = c10::Device(c10::kCUDA);

	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = at::empty({ 1,nf }, options2);

	int l = 0;
	for (int x = 0; x < nfb; ++x) {
		float kx = -M_PI + 2 * x * M_PI / nfb;
		coords.index_put_({ 0,l }, kx);
		++l;
	}

	std::cout << "coords: " << coords << std::endl;

	hasty::Nufft nu2(coords, { nt,nx }, { hasty::NufftType::eType2, true, 1e-6 });

	std::cout << "\n" << (float*)coords.data_ptr() << std::endl;

	auto f = at::empty({ nt,nf }, options1);
	auto c = at::ones({ nt,nx }, options1); // *c10::complex<float>(0.0f, 1.0f);

	nu2.apply(c, f);

	torch::cuda::synchronize();

	std::cout << "Nufft Type 2:\nreal:\n" << at::real(f) << "\n\nimag:\n" << at::imag(f) << std::endl;
	std::cout << "Nufft Type 2:\nreal:\n" << at::real(c) << "\n\nimag:\n" << at::imag(c) << std::endl;

	//nu1.apply(f, c);

	//std::cout << "Nufft Type 1: " << at::real(c) << std::endl << at::imag(c) << std::endl;
}

void hasty::tests::test_deterministic_3d() {
	int n1 = 2;
	int n2 = 2;
	int n3 = 2;
	int nf = n1 * n2 * n3;
	int nx = nf;
	int nt = 1;

	auto device = c10::Device(c10::kCUDA);

	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = at::empty({ 3,nf }, options2);
	std::vector<std::array<float, 3>> fixed_coords = {
		{0.98032004, 0.34147135, 0.61084439},
		{0.80602593, 0.65668227, 0.58148698},
		{0.11397716, 0.33431708, 0.98374713},
		{0.49615291, 0.28232985, 0.99452616},
		{0.6371665, 0.17878657, 0.60917358},
		{0.12547691, 0.80906753, 0.93934428},
		{0.27922639, 0.34255417, 0.93016733},
		{0.5219615, 0.09645867, 0.03214259},
		{0.27719558, 0.53159027, 0.17859567},
		{0.46942625, 0.34165946, 0.76490134},
		{0.75368101, 0.19968615, 0.28943047},
		{0.54910582, 0.42412554, 0.53432473},
		{0.87679761, 0.50310224, 0.97346862},
		{0.90748122, 0.19441631, 0.28978457},
		{0.88753202, 0.82203755, 0.9381962},
		{0.4025496, 0.1600768, 0.57524011},
		{0.65525699, 0.30591684, 0.61524885},
		{0.25528118, 0.40408854, 0.41270846},
		{0.83251984, 0.65801961, 0.59457825},
		{0.03506295, 0.9506098, 0.30317166},
		{0.93298131, 0.89051245, 0.26771778},
		{0.98649964, 0.86362155, 0.78970084},
		{0.63909318, 0.1392839, 0.86418488},
		{0.54289101, 0.63570404, 0.90898353},
		{0.82511889, 0.42676573, 0.32190359},
		{0.31232441, 0.97744591, 0.15726858},
		{0.97187048, 0.24543457, 0.63135005},
		{0.79675513, 0.36351348, 0.01525023},
		{0.82062195, 0.03481492, 0.59195562},
		{0.4450689, 0.8571279, 0.96007808},
		{0.9264222, 0.69303613, 0.10664806},
		{0.71865813, 0.54588828, 0.78308924},
		{0.68904944, 0.0625562, 0.61408031},
		{0.14991815, 0.97537933, 0.53627866},
		{0.12675083, 0.9740082, 0.9854024},
		{0.97176999, 0.8914261, 0.13173493},
		{0.1824385, 0.11983747, 0.38931872},
		{0.54571302, 0.17167634, 0.1974539},
		{0.6698734, 0.03353057, 0.52053646},
		{0.38529699, 0.56414308, 0.84647764},
		{0.2090367, 0.93103123, 0.4905254},
		{0.67817081, 0.9677812, 0.6214241},
		{0.32259623, 0.62142206, 0.82752577},
		{0.44352566, 0.58230826, 0.253879},
		{0.5019813, 0.7628638, 0.99163866},
		{0.43278585, 0.04287191, 0.57724611},
		{0.37305061, 0.85525078, 0.79450935},
		{0.54032247, 0.45156145, 0.51494839},
		{0.43547165, 0.47884727, 0.83920057},
		{0.90771454, 0.44304607, 0.10684609},
		{0.96914703, 0.99099193, 0.67660616},
		{0.12878615, 0.87274706, 0.63536308},
		{0.08763589, 0.96869032, 0.78334944},
		{0.89292773, 0.17549878, 0.63936669},
		{0.04983882, 0.84603837, 0.01729135},
		{0.28048693, 0.26340963, 0.37187648},
		{0.00647681, 0.65307172, 0.36669731},
		{0.51984174, 0.65083299, 0.22426562},
		{0.76060238, 0.54593353, 0.90458955},
		{0.06881982, 0.55241897, 0.20386456},
		{0.40423935, 0.19247943, 0.01989418},
		{0.51691671, 0.3127277, 0.59669834},
		{0.88841294, 0.98875191, 0.02759931},
		{0.33364495, 0.87896414, 0.66459484},
		{0.89172249, 0.98091925, 0.93169932},
		{0.23032556, 0.63535047, 0.85921909},
		{0.0318257, 0.5379664, 0.82014012},
		{0.10007126, 0.42387433, 0.57601986},
		{0.22774557, 0.36218039, 0.17861717},
		{0.03882169, 0.56421647, 0.56901133},
		{0.89507728, 0.11861613, 0.24141007},
		{0.99859803, 0.68317083, 0.09806365},
		{0.71303893, 0.73025946, 0.78945573},
		{0.1102953, 0.62754217, 0.73104685},
		{0.4376749, 0.77870555, 0.62067626},
		{0.41754434, 0.85778273, 0.06900474},
		{0.66257871, 0.09020336, 0.30395005},
		{0.46025083, 0.21331427, 0.52045147},
		{0.18048847, 0.67956526, 0.94928918},
		{0.37032696, 0.58974144, 0.31576817},
		{0.06239387, 0.22940668, 0.62886661},
		{0.25989873, 0.43018133, 0.34109312},
		{0.29890908, 0.62653838, 0.15048865},
		{0.26687828, 0.69478892, 0.35177644},
		{0.6200615, 0.59579733, 0.02966846},
		{0.40749241, 0.28749332, 0.12400579},
		{0.2824288, 0.68365401, 0.67101405},
		{0.10994612, 0.75164738, 0.32341936},
		{0.61737234, 0.02443164, 0.67057934},
		{0.02651275, 0.23172328, 0.06784856},
		{0.70089601, 0.88461766, 0.66256881},
		{0.5713635, 0.62115033, 0.47254448},
		{0.13134128, 0.27217286, 0.52773285},
		{0.45412968, 0.94835191, 0.0894173},
		{0.08491943, 0.39140606, 0.24737922},
		{0.21292069, 0.60481292, 0.18213903},
		{0.6853412, 0.13670174, 0.5994464},
		{0.08583449, 0.31034405, 0.02131659},
		{0.77788847, 0.15017276, 0.35985071},
		{0.13258452, 0.5643354, 0.79174118}
	};

	for (int l = 0; l < nf; ++l) {
		coords.index_put_({ 0,l }, -M_PI + 2 * M_PI * fixed_coords[l][0]);
		coords.index_put_({ 1,l }, -M_PI + 2 * M_PI * fixed_coords[l][1]);
		coords.index_put_({ 2,l }, -M_PI + 2 * M_PI * fixed_coords[l][2]);
	}
	std::cout << "coords: " << coords << std::endl;

	std::vector<int64_t> nmodes_ns = { nt, 2 * n3, 2 * n2, 2 * n1 };

	hasty::Nufft nu2(coords, nmodes_ns, { hasty::NufftType::eType2, false, 1e-6 });
	hasty::Nufft nu1(coords, nmodes_ns, { hasty::NufftType::eType1, true, 1e-6 });

	auto f = at::empty({ nt,nf }, options1);
	auto c = at::zeros(nmodes_ns, options1); // *c10::complex<float>(0.0f, 1.0f);
	c.index_put_({ 0,0,0,1 }, 2.0);
	c.index_put_({ 0,2,1,3 }, 3.0);

	/*
	using namespace at::indexing;
	std::vector<TensorIndex> indices;
	for (int i = 0; i < 4; ++i) {
		indices.emplace_back(i == 0 ? Slice() : TensorIndex(nmodes_ns[i] / 2));
		//indices.emplace_back(i == 0 ? Slice() : TensorIndex(0));
	}
	c.zero_();
	c.index_put_(at::makeArrayRef(indices), 1.0f);
	*/


	std::cout << c << std::endl;

	nu2.apply(c, f);

	std::cout << "Nufft Type 2:\nreal:\n" << at::real(f) << "\n\nimag:\n" << at::imag(f) << std::endl;

	nu1.apply(f, c);

	std::cout << "Nufft Type 2:\nreal:\n" << hasty::torch_util::print_4d_xyz(at::real(c)).str()
		<< "\n\nimag:\n" << hasty::torch_util::print_4d_xyz(at::imag(c)).str() << std::endl;

}

void hasty::tests::test_speed() {
	int nfb = 256;
	int nf = 200000;
	int nx = nfb;
	int nc = 40;

	auto device = c10::Device(c10::kCUDA);
	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);
	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = at::rand({ 3,nf }, options2);

	//hasty::Nufft nu1(coords, { nt,nx,nx,nx }, {hasty::NufftType::eType1, true, 1e-5f});
	//hasty::Nufft nu2(coords, { nt,nx,nx,nx }, {hasty::NufftType::eType2, false, 1e-5f });

	hasty::NufftNormal nufft_normal(coords, { 1, nx, nx, nx },
		{ hasty::NufftType::eType2, false, 1e-5f },
		{ hasty::NufftType::eType1, true, 1e-5f });

	auto freq = at::rand({ 1,nf }, options1);
	auto in = at::rand({ nc,nx,nx,nx }, options1);
	auto out = at::rand({ nc,nx,nx,nx }, options1);

	torch::cuda::synchronize();

	

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < nc; ++i) {
		using namespace at::indexing;
		auto inview = in.select(0, i).view({ 1,nx,nx,nx });
		auto outview = out.select(0, i).view({ 1,nx,nx,nx });

		nufft_normal.apply(inview, outview, freq, std::nullopt);
	}
	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << duration.count() << std::endl;

}

void hasty::tests::test_space_cufinufft() {
	int nfb = 256;
	int nf = 200000;
	int nx = nfb;
	int nt = 5;

	auto device = c10::Device(torch::kCUDA);
	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);
	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::rand({ 3,nf }, options2);

	hasty::Nufft nu1(coords, { nt,nx,nx,nx }, { hasty::NufftType::eType1, true, 1e-5f });
	hasty::Nufft nu2(coords, { nt,nx,nx,nx }, { hasty::NufftType::eType2, true, 1e-5f });

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

void hasty::tests::test_svd_speed() {
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

void hasty::tests::compare_normal_methods()
{
	using namespace hasty;

	int nx = 5;
	int ny = 2;
	int nz = 3;
	int nf = nx * ny * nz;
	int nt = 1;

	auto device = c10::Device(torch::kCUDA);
	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);
	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::empty({ 3,nf }, options2);
	int l = 0;
	for (int x = 0; x < nx; ++x) {
		for (int y = 0; y < ny; ++y) {
			for (int z = 0; z < nz; ++z) {
				float kx = (2 * x * M_PI / nx) - M_PI;
				float ky = (2 * y * M_PI / ny) - M_PI;
				float kz = (2 * z * M_PI / nz) - M_PI;

				coords.index_put_({ 0,l }, kx);
				coords.index_put_({ 1,l }, ky);
				coords.index_put_({ 2,l }, kz);

				++l;
			}
		}
	}
	coords = -M_PI + 2 * M_PI * torch::rand({ 3,nf }, options2);
	std::vector<std::array<float, 3>> fixed_coords = {
		{0.98032004, 0.34147135, 0.61084439},
		{0.80602593, 0.65668227, 0.58148698},
		{0.11397716, 0.33431708, 0.98374713},
		{0.49615291, 0.28232985, 0.99452616},
		{0.6371665, 0.17878657, 0.60917358},
		{0.12547691, 0.80906753, 0.93934428},
		{0.27922639, 0.34255417, 0.93016733},
		{0.5219615, 0.09645867, 0.03214259},
		{0.27719558, 0.53159027, 0.17859567},
		{0.46942625, 0.34165946, 0.76490134},
		{0.75368101, 0.19968615, 0.28943047},
		{0.54910582, 0.42412554, 0.53432473},
		{0.87679761, 0.50310224, 0.97346862},
		{0.90748122, 0.19441631, 0.28978457},
		{0.88753202, 0.82203755, 0.9381962},
		{0.4025496, 0.1600768, 0.57524011},
		{0.65525699, 0.30591684, 0.61524885},
		{0.25528118, 0.40408854, 0.41270846},
		{0.83251984, 0.65801961, 0.59457825},
		{0.03506295, 0.9506098, 0.30317166},
		{0.93298131, 0.89051245, 0.26771778},
		{0.98649964, 0.86362155, 0.78970084},
		{0.63909318, 0.1392839, 0.86418488},
		{0.54289101, 0.63570404, 0.90898353},
		{0.82511889, 0.42676573, 0.32190359},
		{0.31232441, 0.97744591, 0.15726858},
		{0.97187048, 0.24543457, 0.63135005},
		{0.79675513, 0.36351348, 0.01525023},
		{0.82062195, 0.03481492, 0.59195562},
		{0.4450689, 0.8571279, 0.96007808},
		{0.9264222, 0.69303613, 0.10664806},
		{0.71865813, 0.54588828, 0.78308924},
		{0.68904944, 0.0625562, 0.61408031},
		{0.14991815, 0.97537933, 0.53627866},
		{0.12675083, 0.9740082, 0.9854024},
		{0.97176999, 0.8914261, 0.13173493},
		{0.1824385, 0.11983747, 0.38931872},
		{0.54571302, 0.17167634, 0.1974539},
		{0.6698734, 0.03353057, 0.52053646},
		{0.38529699, 0.56414308, 0.84647764},
		{0.2090367, 0.93103123, 0.4905254},
		{0.67817081, 0.9677812, 0.6214241},
		{0.32259623, 0.62142206, 0.82752577},
		{0.44352566, 0.58230826, 0.253879},
		{0.5019813, 0.7628638, 0.99163866},
		{0.43278585, 0.04287191, 0.57724611},
		{0.37305061, 0.85525078, 0.79450935},
		{0.54032247, 0.45156145, 0.51494839},
		{0.43547165, 0.47884727, 0.83920057},
		{0.90771454, 0.44304607, 0.10684609},
		{0.96914703, 0.99099193, 0.67660616},
		{0.12878615, 0.87274706, 0.63536308},
		{0.08763589, 0.96869032, 0.78334944},
		{0.89292773, 0.17549878, 0.63936669},
		{0.04983882, 0.84603837, 0.01729135},
		{0.28048693, 0.26340963, 0.37187648},
		{0.00647681, 0.65307172, 0.36669731},
		{0.51984174, 0.65083299, 0.22426562},
		{0.76060238, 0.54593353, 0.90458955},
		{0.06881982, 0.55241897, 0.20386456},
		{0.40423935, 0.19247943, 0.01989418},
		{0.51691671, 0.3127277, 0.59669834},
		{0.88841294, 0.98875191, 0.02759931},
		{0.33364495, 0.87896414, 0.66459484},
		{0.89172249, 0.98091925, 0.93169932},
		{0.23032556, 0.63535047, 0.85921909},
		{0.0318257, 0.5379664, 0.82014012},
		{0.10007126, 0.42387433, 0.57601986},
		{0.22774557, 0.36218039, 0.17861717},
		{0.03882169, 0.56421647, 0.56901133},
		{0.89507728, 0.11861613, 0.24141007},
		{0.99859803, 0.68317083, 0.09806365},
		{0.71303893, 0.73025946, 0.78945573},
		{0.1102953, 0.62754217, 0.73104685},
		{0.4376749, 0.77870555, 0.62067626},
		{0.41754434, 0.85778273, 0.06900474},
		{0.66257871, 0.09020336, 0.30395005},
		{0.46025083, 0.21331427, 0.52045147},
		{0.18048847, 0.67956526, 0.94928918},
		{0.37032696, 0.58974144, 0.31576817},
		{0.06239387, 0.22940668, 0.62886661},
		{0.25989873, 0.43018133, 0.34109312},
		{0.29890908, 0.62653838, 0.15048865},
		{0.26687828, 0.69478892, 0.35177644},
		{0.6200615, 0.59579733, 0.02966846},
		{0.40749241, 0.28749332, 0.12400579},
		{0.2824288, 0.68365401, 0.67101405},
		{0.10994612, 0.75164738, 0.32341936},
		{0.61737234, 0.02443164, 0.67057934},
		{0.02651275, 0.23172328, 0.06784856},
		{0.70089601, 0.88461766, 0.66256881},
		{0.5713635, 0.62115033, 0.47254448},
		{0.13134128, 0.27217286, 0.52773285},
		{0.45412968, 0.94835191, 0.0894173},
		{0.08491943, 0.39140606, 0.24737922},
		{0.21292069, 0.60481292, 0.18213903},
		{0.6853412, 0.13670174, 0.5994464},
		{0.08583449, 0.31034405, 0.02131659},
		{0.77788847, 0.15017276, 0.35985071},
		{0.13258452, 0.5643354, 0.79174118}
	};

	for (int l = 0; l < nf; ++l) {
		coords.index_put_({ 0,l }, -M_PI + 2 * M_PI * fixed_coords[l][0]);
		coords.index_put_({ 1,l }, -M_PI + 2 * M_PI * fixed_coords[l][1]);
		coords.index_put_({ 2,l }, -M_PI + 2 * M_PI * fixed_coords[l][2]);
	}

	std::cout << "coords: " << coords.transpose(0, 1) << std::endl;

	auto freq_temp = torch::empty({ nt,nf }, options1);
	auto full_temp1 = torch::empty({ nt, 2 * nz, 2 * ny, 2 * nx }, options1);
	auto full_temp2 = torch::empty({ nt, 2 * nz, 2 * ny, 2 * nx }, options1);
	auto c = torch::ones({ nt,nz,ny,nx }, options1);
	//auto c = torch::rand({ nt,nz,ny,nx }, options1);

	auto out1 = torch::empty({ nt,nz,ny,nx }, options1);
	auto out2 = torch::empty({ nt,nz,ny,nx }, options1);
	auto out3 = torch::empty({ nt,nz,ny,nx }, options1);

	// Explicit forward backward
	{
		Nufft nu1(coords, { nt,nz,ny,nx }, { NufftType::eType1, false, 1e-6f });
		Nufft nu2(coords, { nt,nz,ny,nx }, { NufftType::eType2, true, 1e-6f });

		nu2.apply(c, freq_temp);
		nu1.apply(freq_temp, out1);
	}
	// Normal Nufft
	{
		NufftNormal nu_normal(coords, { nt,nz,ny,nx },
			{ NufftType::eType2, false, 1e-6 }, { NufftType::eType1, true, 1e-6 });

		nu_normal.apply(c, out2, freq_temp, std::nullopt);
	}
	// Toeplitz Normal Nufft
	{
		NormalNufftToeplitz top_normal(coords, { nt,nz,ny,nx }, 1e-6, std::nullopt, std::nullopt, std::nullopt);

		out3 = top_normal.apply(c, full_temp1, full_temp2);
		//top_normal.apply(torch::fft_fftshift(c), out3, full_temp1, full_temp2);
	}

	//std::cout << "out3 real: " << torch::real(out3) << std::endl;
	//std::cout << "ft2  real: " << torch::real(full_temp2) << std::endl;


	std::cout << "out2 real: " << torch::real(out2) << std::endl;
	std::cout << "out3 real: " << torch::real(out3) << std::endl;

	//std::cout << "out3 imag: " << torch::imag(out3) << std::endl;
	//std::cout << "ft2  imag: " << torch::imag(full_temp2) << std::endl;
	std::cout << "out2 imag: " << torch::imag(out2) << std::endl;
	std::cout << "out3 imag: " << torch::imag(out3) << std::endl;

	std::cout << "out2 abs: " << torch::abs(out2) << std::endl;
	std::cout << "out3 abs: " << torch::abs(out3) << std::endl;


	/*
	auto start = std::chrono::high_resolution_clock::now();
	torch::cuda::synchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << duration.count() << std::endl;
	*/
}

void hasty::tests::test_nufft_speeds(bool toep) {
	int nfb = 256;
	int nf = 200000;
	int nx = nfb;
	int nt = 1;
	int nrun = 20;
	c10::InferenceMode guard;

	auto device = c10::Device(torch::kCUDA);
	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);
	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::rand({ 3,nf }, options2);

	using namespace hasty;

	auto freq_temp = torch::empty({ nt, nf }, options1);
	auto in = torch::rand({ nt, nx, nx, nx }, options1);
	auto out = torch::empty({ nt, nx, nx, nx }, options1);

	auto storage1 = torch::empty({ nt, 2 * nx, 2 * nx, 2 * nx }, options1);
	auto storage2 = torch::empty({ nt, 2 * nx, 2 * nx, 2 * nx }, options1);

	if (!toep)
	{

		torch::cuda::synchronize();

		auto start = std::chrono::high_resolution_clock::now();

		NufftNormal normal_nufft(coords, { nt, nx, nx, nx }, NufftOptions::type2(1e-4), NufftOptions::type1(1e-4));

		for (int i = 0; i < nrun; ++i) {
			normal_nufft.apply(in, out, freq_temp, std::nullopt);
		}

		torch::cuda::synchronize();
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		std::cout << "2 NUFFTS: " << duration.count() << std::endl;

	}

	if (toep) {

		torch::cuda::synchronize();
		auto start = std::chrono::high_resolution_clock::now();

		NormalNufftToeplitz normal_nufft(coords, { nt, nx, nx, nx }, 1e-4, std::nullopt, freq_temp, storage1);
		for (int i = 0; i < nrun; ++i) {
			out = normal_nufft.apply(in, storage1, storage2);
		}

		torch::cuda::synchronize();
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		std::cout << "Toeplitz: " << duration.count() << std::endl;
	}
}
// Speed comparisons
int hasty::tests::test_torch_speed(int n, int nc) {

	int n1 = n;
	int n2 = 2 * n1;

	c10::InferenceMode im;

	auto device = c10::Device(c10::kCUDA);

	auto options = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);

	at::Tensor ffter = at::rand({ nc, n1,n1,n1 }, options);
	at::Tensor ffter_out = at::empty({ nc, n1,n1,n1 }, options);
	//at::Tensor ffter = at::rand({ 1, n2,n2,n2 }, options);
	at::Tensor ffter_mul = at::rand({ n2,n2,n2 }, options);
	at::Tensor ffter_temp = at::empty({ n2,n2,n2 }, options);

	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < nc; ++i) {
		using namespace at::indexing;
		auto inview = ffter.select(0, i);
		auto outview = ffter_out.select(0, i);

		at::fft_fftn_out(ffter_temp, inview, { n2, n2, n2 }, c10::nullopt);
		ffter_temp.mul_(ffter_mul);
		outview = at::fft_ifftn(ffter_temp, { n2,n2,n2 }, c10::nullopt).index(
			{ Slice(0,n1), Slice(0,n1), Slice(0,n1) });

	}


	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	int durt = duration.count();

	std::cout << "Time (ms): " << durt << std::endl;

	return durt;
}

int hasty::tests::test_speed(int n, int nc, int nf) {
	int nx = n;

	auto device = c10::Device(torch::kCUDA);
	auto options1 = c10::TensorOptions().device(device).dtype(c10::ScalarType::ComplexFloat);
	auto options2 = c10::TensorOptions().device(device).dtype(c10::ScalarType::Float);

	auto coords = torch::rand({ 3,nf }, options2);

	//hasty::Nufft nu1(coords, { nt,nx,nx,nx }, {hasty::NufftType::eType1, true, 1e-5f});
	//hasty::Nufft nu2(coords, { nt,nx,nx,nx }, {hasty::NufftType::eType2, false, 1e-5f });

	auto freq = torch::rand({ 1,nf }, options1);
	auto in = torch::rand({ nc,nx,nx,nx }, options1);
	auto out = torch::rand({ nc,nx,nx,nx }, options1);

	torch::cuda::synchronize();

	auto start = std::chrono::high_resolution_clock::now();

	hasty::NufftNormal nufft_normal(coords, { 1, nx, nx, nx },
		{ hasty::NufftType::eType2, false, 1e-5f },
		{ hasty::NufftType::eType1, true, 1e-5f });

	for (int i = 0; i < nc; ++i) {
		using namespace at::indexing;
		auto inview = in.select(0, i).view({ 1,nx,nx,nx });
		auto outview = out.select(0, i).view({ 1,nx,nx,nx });

		nufft_normal.apply(inview, outview, freq, std::nullopt);
	}
	torch::cuda::synchronize();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	int durt = duration.count();

	std::cout << "Time (ms): " << durt << std::endl;

	return durt;
}

void hasty::tests::speed_comparisons_torch_af_cufinufft(int nres, int nrun) {
	int time;
	int tot_time;

	std::cout << "LibTorch speed" << std::endl;
	tot_time = 0;
	for (int i = 0; i < nrun; ++i) {
		time = test_torch_speed(nres, 20);
		if (i != 0)
			tot_time += time;
	}
	std::cout << "AverageTime (ms): " << tot_time / (nrun - 1) << std::endl;

	std::cout << "cuFINUFFT speed:\n";
	tot_time = 0;
	for (int i = 0; i < nrun; ++i) {
		time = test_speed(nres, 20, 400000);
		if (i != 0)
			tot_time += time;
	}
	std::cout << "AverageTime (ms): " << tot_time / (nrun - 1) << std::endl;
}
