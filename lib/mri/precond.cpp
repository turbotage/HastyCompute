module;

#include <torch/torch.h>

module precond;

import hasty_util;
import fft;
import fftop;
import nufft;


std::unique_ptr<hasty::mri::CirculantPreconditionerOp> hasty::mri::CirculantPreconditionerOp::Create(at::Tensor diag, bool centered, const std::optional<std::string>& norm, bool adjointify)
{
	struct creator : public CirculantPreconditionerOp {
		creator(at::Tensor a, bool b, const std::optional<std::string>& c, bool d)
			: CirculantPreconditionerOp(std::move(a), b, c, d) {}
	};
	return std::make_unique<creator>(std::move(diag), centered, norm, adjointify);
}

hasty::mri::CirculantPreconditionerOp::CirculantPreconditionerOp(at::Tensor diag, bool centered, const std::optional<std::string>& norm, bool adjointify)
	: _diag(std::move(diag)), _centered(centered), _norm(norm), _adjointify(adjointify)
{
}

hasty::op::Vector hasty::mri::CirculantPreconditionerOp::apply(const op::Vector& in) const
{

	if (_adjointify) {
		if (_centered) {
			return fft::fftc(_diag.conj() * fft::ifftc(in.tensor(), at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt), 
				at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt);
		}
		else {
			return at::fft_fftn(_diag.conj() * at::fft_ifftn(in.tensor(), at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt),
				at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt);
		}
	}
	else {
		if (_centered) {
			return fft::ifftc(_diag * fft::fftc(in.tensor(), at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt),
				at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt);
		}
		else {
			return at::fft_ifftn(_diag * at::fft_fftn(in.tensor(), at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt),
				at::nullopt, at::nullopt, _norm.has_value() ? at::make_optional<at::string_view>(*_norm) : at::nullopt);
		}
	}
}

std::shared_ptr<hasty::op::Operator> hasty::mri::CirculantPreconditionerOp::to_device(at::Stream stream) const
{
	return CirculantPreconditionerOp::Create(_diag.to(stream.device()), _centered, _norm, _adjointify);
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::mri::CirculantPreconditionerOp::adjoint() const
{
	return CirculantPreconditionerOp::Create(_diag, _centered, _norm, !_adjointify);
}

/*
based on sigpy
*/
at::Tensor hasty::mri::CirculantPreconditionerOp::build_diagonal(at::Tensor smaps, at::Tensor coord, const at::optional<at::Tensor>& weights, const at::optional<double>& lambda)
{
	using namespace at::indexing;

	auto mps_shape = smaps.sizes();
	auto img_shape = std::vector<int64_t>(mps_shape.begin() + 1, mps_shape.end());
	std::vector<int64_t> img2_shape = img_shape;
	std::for_each(img2_shape.begin(), img2_shape.end(), [](int64_t& l) {
		return l * 2;
		});
	int ndim = img_shape.size();

	auto scale = std::pow(std::accumulate(img2_shape.begin(), img2_shape.end(), int64_t(1), std::multiplies<int64_t>()), 1.5);
	scale /= std::pow(std::accumulate(img_shape.begin(), img_shape.end(), int64_t(1), std::multiplies<int64_t>()), 2.0);

	auto ones = at::ones({ coord.size(1) }, smaps.options());
	if (weights.has_value()) {
		ones *= at::sqrt(*weights);
	}

	auto nmodes = util::vec_concat<int64_t>({1}, img2_shape);
	at::Tensor psf = op::NUFFTAdjoint(coord, nmodes, fft::NufftOptions::type1()).apply(ones).tensor();

	std::vector<TensorIndex> idx;
	idx.reserve(ndim);
	for (int i = 0; i < ndim; ++i) {
		idx.push_back(Slice(None, None, 2));
	}

	at::Tensor p_inv = at::zeros({coord.size(1)}, smaps.options());
	for (int i = 0; i < smaps.size(0); ++i) {

		auto mapsi = smaps.slice(0, i);
		auto xcorr = at::square(at::abs(fft::fftc(at::conj(mapsi), img2_shape, at::nullopt, at::nullopt)));
		xcorr = fft::ifftc(xcorr);
		xcorr *= psf;

		auto p_inv_i = fft::fftc(xcorr);
		p_inv_i = p_inv_i.index(idx);
		p_inv_i *= scale;

		if (weights.has_value())
			p_inv_i *= at::sqrt(*weights);

		p_inv.add_(p_inv_i);
	}

	if (lambda.has_value())
		p_inv += *lambda;
	p_inv.masked_fill_(p_inv == 0, 1.0);

	return at::reciprocal(p_inv);
}
