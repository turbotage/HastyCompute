#include "precond.hpp"
//#include "../fft/nufft.hpp"
#include "../op/fftop.hpp"

import hasty_util;

at::Tensor hasty::op::CirculantPreconditioner::build_diagonal(at::Tensor smaps, at::Tensor coord, const at::optional<at::Tensor>& weights, const at::optional<double>& lambda)
{
	auto mps_shape = smaps.sizes();
	auto img_shape = std::vector<int64_t>(mps_shape.begin() + 1, mps_shape.end());
	std::vector<int64_t> img2_shape = img_shape;
	std::for_each(img2_shape.begin(), img2_shape.end(), [](int64_t& l) {
		return l * 2;
		});
	int ndim = img_shape.size();

	auto scale = std::pow(std::accumulate(img2_shape.begin(), img2_shape.end(), 1, std::multiplies<int64_t>()), 1.5);
	scale /= std::pow(std::accumulate(img_shape.begin(), img_shape.end(), 1, std::multiplies<int64_t>()), 2.0);

	auto ones = at::ones({ coord.size(1) }, smaps.options());
	if (weights.has_value()) {
		ones *= at::sqrt(*weights);
	}

	auto nmodes = util::vec_concat<int64_t>({1}, img2_shape);
	at::Tensor psf = NUFFTAdjoint(coord, nmodes, nufft::NufftOptions::type1()).apply(ones).tensor();

	for (int i = 0; i < smaps.size(0); ++i) {

		auto mapsi = smaps.slice(0, i);
		


		auto corrf = at::abs(at::fft_fftn(, );
	}

}
