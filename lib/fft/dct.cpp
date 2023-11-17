module;

#include <torch/torch.h>

module dct;

at::Tensor hasty::fft::dct_fft_impl(const at::Tensor v)
{
	return at::view_as_real(at::fft_fft(v, at::nullopt, 1));
}

at::Tensor hasty::fft::idct_irfft_impl(const at::Tensor V)
{
	return at::fft_irfft(at::view_as_complex(V), V.size(1), 1);
}

at::Tensor hasty::fft::dct(at::Tensor x, const std::string& norm)
{
	using namespace torch::indexing;

	auto x_shape = x.sizes();
	int64_t N = x_shape.back();
	x = x.contiguous().view({ -1, N });

	auto v = at::cat({ x.slice(1, at::nullopt, at::nullopt, 2), x.slice(1, 1, at::nullopt, 2).flip({1}) }, 1);

	auto Vc = dct_fft_impl(v);

	auto k = -0.5 * at::arange(N, x.options()).unsqueeze(0) * M_PI / N;
	auto W_r = at::cos(k);
	auto W_i = at::sin(k);

	auto V = Vc.select(2, 0) * W_r - Vc.select(2, 1) * W_i;

	if (norm == "ortho") {
		V.select(1, 0) /= (std::sqrt(N) * 2);
		V.slice(1, 1) /= std::sqrt(N * 2);
	}

	V = 2 * V.view(x_shape);

	return V;
}

at::Tensor hasty::fft::idct(at::Tensor X, const std::string& norm)
{
	using namespace at::indexing;

	auto x_shape = X.sizes();
	int64_t N = x_shape.back();

	auto X_v = X.contiguous().view({ -1, x_shape.back() }) / 2;

	if (norm == "ortho") {
		X_v.select(1, 0) *= (std::sqrt(N) * 2);
		X_v.slice(1, 1) *= std::sqrt(N * 2);
	}

	auto k = at::arange(x_shape.back(), X.options()).unsqueeze(0) * M_PI / (2 * N);
	auto W_r = at::cos(k);
	auto W_i = at::sin(k);

	auto V_t_r = X_v;
	auto V_t_i = at::cat({ at::zeros_like(X_v.slice(1, 1)), -X_v.flip({1}).slice(1, at::nullopt, -1) }, 1);

	auto V_r = V_t_r * W_r - V_t_i * W_i;
	auto V_i = V_t_r * W_i - V_t_i * W_r;

	auto V = at::cat({ V_r.unsqueeze(2), V_i.unsqueeze(2) }, 2);

	auto v = idct_irfft_impl(V);

	auto x = v.new_zeros(v.sizes());

	x.slice(1, at::nullopt, at::nullopt, 2) += v.slice(1, at::nullopt, N - int64_t(N / 2));
	x.slice(1, 1, at::nullopt, 2) += v.flip({ 1 }).slice(1, at::nullopt, int64_t(N / 2));

	return x.view(x_shape);
}

at::Tensor hasty::fft::dct_2d(at::Tensor x, const std::string& norm)
{
	auto X1 = dct(x, norm);
	X1 = dct(X1.transpose(-1, -2), norm);
	return X1.transpose(-1, -2);
}

at::Tensor hasty::fft::idct_2d(at::Tensor X, const std::string& norm)
{
	auto x1 = idct(X, norm);
	x1 = idct(x1.transpose(-1, -2), norm);
	return x1.transpose(-1, -2);
}

at::Tensor hasty::fft::dct_3d(at::Tensor x, const std::string& norm)
{
	auto X1 = dct(x, norm);
	X1 = dct(X1.transpose(-1, -2), norm);
	X1 = dct(X1.transpose(-1, -3), norm);
	return X1.transpose(-1, -3).transpose(-1, -2);
}

at::Tensor hasty::fft::idct_3d(at::Tensor X, const std::string& norm)
{
	auto x1 = idct(X, norm);
	x1 = idct(x1.transpose(-1, -2), norm);
	x1 = idct(x1.transpose(-1, -3), norm);
	return x1.transpose(-1, -3).transpose(-1, -2);
}
