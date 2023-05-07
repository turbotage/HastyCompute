#include "nufft.hpp"

#include "../fft/nufft_cu.hpp"

at::Tensor hasty::ffi::nufft1(const at::Tensor& coords, const at::Tensor& input, const std::vector<int64_t>& nmodes)
{
	using namespace hasty::cuda;
	Nufft nufft(coords, nmodes, NufftOptions::type1());
	auto out = at::empty(at::makeArrayRef(nmodes), input.options());

	nufft.apply(input, out);

	return out;
}

at::Tensor hasty::ffi::nufft2(const at::Tensor& coords, const at::Tensor& input)
{
	using namespace hasty::cuda;
	Nufft nufft(coords, input.sizes().vec(), NufftOptions::type2());
	auto out = at::empty({ input.size(0), coords.size(1)}, input.options());

	nufft.apply(input, out);

	return out;
}

at::Tensor hasty::ffi::nufft2to1(const at::Tensor& coords, const at::Tensor& input)
{
	using namespace hasty::cuda;
	NufftNormal normal_nufft(coords, input.sizes().vec(), NufftOptions::type2(), NufftOptions::type1());
	auto out = at::empty(at::makeArrayRef(input.sizes()), input.options());

	auto freq_storage = at::empty({ input.size(0), coords.size(1)}, input.options());

	normal_nufft.apply(input, out, freq_storage, std::nullopt);

	return out;
}
