#include "cg_cu.hpp"

using namespace hasty::cuda;

CGRecon::CGRecon(
	const Options& options,
	at::Tensor& image,
	const TensorVec& coords,
	const at::Tensor& smaps,
	const TensorVec& kdata)
	:
	_options(options),
	_image(image),
	_coords(coords),
	_smaps(smaps),
	_kdata(kdata),
	_nframe(image.size(0)),
	_nmodes(4)
{
	construct();
}

CGRecon(
	const Options& options,
	at::Tensor& image,
	const TensorVec& coords,
	const at::Tensor& smaps,
	const TensorVec& kdata,
	const TensorVec& weights)
{

}


void CGRecon::construct()
{
	_nmodes[0] = 1; _nmodes[1] = _image.size(2); _nmodes[2] = _image.size(3); _nmodes[3] = _image.size(4);

	auto base_options1 = c10::TensorOptions().dtype(c10::ScalarType::ComplexFloat);
	auto base_options2 = c10::TensorOptions().dtype(c10::ScalarType::Float);

	for (auto& device : _options.devices) {

		auto& dcontext = _dcontexts.emplace_back(device);

		c10::cuda::CUDAGuard device_guard(device);

		auto options1 = base_options1.device(device);
		auto options2 = base_options2.device(device);

		dcontext.image = at::empty(at::makeArrayRef(_nmodes), options1);
		dcontext.out = at::empty_like(dcontext.image);
		dcontext.in_storage = at::empty_like(dcontext.image);

		dcontext.smaps = _smaps.to(device, true);

	}

	_tpool = std::make_unique<ContextThreadPool<DeviceContext>>(_dcontexts);
}

