#include "fft.hpp"


at::Tensor hasty::fft::fftc(const at::Tensor& input, const at::OptionalArrayRef<int64_t>& oshape,
	const at::OptionalArrayRef<int64_t>& axes, const at::optional<at::string_view>& norm)
{
	//auto ndim = input.ndimension();

	at::Tensor tmp = oshape.has_value() ? torch_util::upscale_with_zeropad(input, *oshape) : input;

	tmp = at::fft_ifftshift(tmp, axes);
	tmp = at::fft_fftn(tmp, at::nullopt, axes, norm);
	tmp = at::fft_fftshift(tmp, axes);
	return tmp;
}

at::Tensor hasty::fft::ifftc(const at::Tensor& input, const at::OptionalArrayRef<int64_t>& oshape, 
	const at::OptionalArrayRef<int64_t>& axes, const at::optional<at::string_view>& norm)
{
	//auto ndim = input.ndimension();


}
