#include "nufft.hpp"

#include "../fft/nufft.hpp"

at::Tensor hasty::ffi::nufft1(const at::Tensor& coords, const at::Tensor& input, const std::vector<int64_t>& nmodes)
{
	using namespace hasty;
	Nufft nufft(coords, nmodes, NufftOptions::type1());
	auto out = at::empty(at::makeArrayRef(nmodes), input.options());

	nufft.apply(input, out);

	return out;
}

at::Tensor hasty::ffi::nufft2(const at::Tensor& coords, const at::Tensor& input)
{
	using namespace hasty;
	Nufft nufft(coords, input.sizes().vec(), NufftOptions::type2());
	auto out = at::empty({ input.size(0), coords.size(1)}, input.options());

	nufft.apply(input, out);

	return out;
}

at::Tensor hasty::ffi::nufft21(const at::Tensor& coords, const at::Tensor& input)
{
	using namespace hasty;
	NufftNormal normal_nufft(coords, input.sizes().vec(), NufftOptions::type2(false), NufftOptions::type1(true));
	//auto out = at::empty(at::makeArrayRef(input.sizes()), input.options());
	auto out = at::empty_like(input);

	auto storage = at::empty({ input.size(0), coords.size(1)}, input.options());

	normal_nufft.apply(input, out, storage, std::nullopt);

	return out;
}

at::Tensor hasty::ffi::nufft12(const at::Tensor& coords, const at::Tensor& input, const std::vector<int64_t>& nmodes)
{
	using namespace hasty;
	NufftNormal normal_nufft(coords, nmodes, NufftOptions::type1(), NufftOptions::type2());
	auto out = at::empty(at::makeArrayRef(input.sizes()), input.options());

	auto storage = at::empty(at::makeArrayRef(nmodes), input.options());

	normal_nufft.apply(input, out, storage, std::nullopt);

	return out;
}