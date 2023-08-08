#include "py_nufft.hpp"

hasty::ffi::NufftOptions::NufftOptions(int64_t type, const at::optional<bool>& positive, const at::optional<double>& tol)
{
	if (type > 3 || type < 1) {
		throw std::runtime_error("Not available type");
	}

	_opts = std::make_unique<nufft::NufftOptions>((nufft::NufftType)type, positive, tol);
}

const hasty::nufft::NufftOptions& hasty::ffi::NufftOptions::getOptions() const
{
	return *_opts;
}


hasty::ffi::Nufft::Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const ffi::NufftOptions& opts)
	: _nufftop(std::make_unique<nufft::Nufft>(coords, nmodes, opts.getOptions()))
{}

void hasty::ffi::Nufft::apply(const at::Tensor& in, at::Tensor out) const
{
	auto func = [this, &in, &out]() {
		_nufftop->apply(in, out);
	};

	torch_util::future_catcher(func);
}

hasty::ffi::NufftNormal::NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const ffi::NufftOptions& forward, const ffi::NufftOptions& backward)
	: _nufftop(std::make_unique<nufft::NufftNormal>(coords, nmodes, forward.getOptions(), backward.getOptions()))
{}

void hasty::ffi::NufftNormal::apply(const at::Tensor& in, at::Tensor out, 
	at::Tensor storage, at::optional<std::function<void(at::Tensor&)>>& func_between) const
{
	auto func = [this, &in, &out, &storage, &func_between]() {
		_nufftop->apply(in, out, storage, func_between);
	};

	torch_util::future_catcher(func);
}


TORCH_LIBRARY(HastyNufft, hn) {

	hn.class_<hasty::ffi::NufftOptions>("NufftOptions")
		.def(torch::init<int64_t, const at::optional<bool>&, const at::optional<double>&>());

	hn.class_<hasty::ffi::Nufft>("Nufft")
		.def(torch::init<const at::Tensor&, const std::vector<int64_t>&, const hasty::ffi::NufftOptions&>())
		.def("apply", &hasty::ffi::Nufft::apply);

	hn.class_<hasty::ffi::NufftNormal>("NufftNormal")
		.def(torch::init<const at::Tensor&, const std::vector<int64_t>&, 
			const hasty::ffi::NufftOptions&, const hasty::ffi::NufftOptions&>())
		.def("apply", &hasty::ffi::NufftNormal::apply);
	
}
