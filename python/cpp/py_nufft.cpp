#include "py_nufft.hpp"

import torch_util;
import nufft;

hasty::ffi::NufftOptions::NufftOptions(int64_t type, const at::optional<bool>& positive, const at::optional<double>& tol)
{
	if (type > 3 || type < 1) {
		throw std::runtime_error("Not available type");
	}

	_opts = std::make_unique<fft::NufftOptions>((fft::NufftType)type, positive, tol);
}

const hasty::fft::NufftOptions& hasty::ffi::NufftOptions::getOptions() const
{
	return *_opts;
}


hasty::ffi::Nufft::Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const ffi::NufftOptions& opts)
{
	if (coords.is_cuda()) {
		_nufftop = nullptr;
	}

	_nufftop(std::make_unique<fft::Nufft>(coords, nmodes, opts.getOptions()))
}

void hasty::ffi::Nufft::apply(const at::Tensor& in, at::Tensor out) const
{
	auto func = [this, &in, &out]() {
		_nufftop->apply(in, out);
	};

	torch_util::future_catcher(func);
}


hasty::ffi::NufftNormal::NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const ffi::NufftOptions& forward, const ffi::NufftOptions& backward)
	: _nufftop(std::make_unique<fft::NufftNormal>(coords, nmodes, forward.getOptions(), backward.getOptions()))
{}

void hasty::ffi::NufftNormal::apply(const at::Tensor& in, at::Tensor out, at::Tensor storage,
	const at::optional<FunctionLambda>& func_between) const
{
	at::optional<std::function<void(at::Tensor&)>> lambda;
	if (func_between.has_value()) {
		lambda = at::make_optional<std::function<void(at::Tensor&)>>([func_between](at::Tensor& in) {
			(*func_between).apply(in);
		});
	}

	auto func = [this, &in, &out, &storage, &lambda]() {
		_nufftop->apply(in, out, storage, lambda);
	};

	torch_util::future_catcher(func);
}



TORCH_LIBRARY(HastyNufft, hn) {

	hn.class_<hasty::ffi::NufftOptions>("NufftOptions")
		.def(torch::init<int64_t, const at::optional<bool>&, const at::optional<double>&>());

	hn.class_<hasty::ffi::Nufft>("Nufft")
		.def(torch::init([](const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::intrusive_ptr<hasty::ffi::NufftOptions>& opts) {
				return at::make_intrusive<hasty::ffi::Nufft>(coords, nmodes, *opts);
			}))
		.def("apply", &hasty::ffi::Nufft::apply);

	hn.class_<hasty::ffi::NufftNormal>("NufftNormal")
		.def(torch::init([](const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
			const at::intrusive_ptr<hasty::ffi::NufftOptions>& forward_opts, const at::intrusive_ptr<hasty::ffi::NufftOptions>& backward_opts) {
				return at::make_intrusive<hasty::ffi::NufftNormal>(coords, nmodes, *forward_opts, *backward_opts);
			}))
		.def("apply", [](const at::intrusive_ptr<hasty::ffi::NufftNormal>& self, const at::Tensor& in, at::Tensor out, at::Tensor storage,
			const at::optional<at::intrusive_ptr<hasty::ffi::FunctionLambda>>& func_between) {
				at::optional<hasty::ffi::FunctionLambda> funclam;
				if (func_between.has_value()) {
					funclam = *(*func_between);
				}
				self->apply(in, out, storage, funclam);
			});

	hn.def("doc", []() -> std::string {
		return 
R"DOC(

// HastyNufft Module

class LIB_EXPORT NufftOptions : public torch::CustomClassHolder {
public:

	NufftOptions(int64_t type, const at::optional<bool>& positive, const at::optional<double>& tol);

	const nufft::NufftOptions& getOptions() const;

private:
	std::unique_ptr<nufft::NufftOptions> _opts;
};

class LIB_EXPORT Nufft : public torch::CustomClassHolder {
public:

	Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const ffi::NufftOptions& opts);

	void apply(const at::Tensor& in, at::Tensor out) const;

private:
	std::unique_ptr<nufft::Nufft> _nufftop;
};

class LIB_EXPORT NufftNormal : public torch::CustomClassHolder {
public:

	NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
		const ffi::NufftOptions& forward, const ffi::NufftOptions& backward);

	void apply(const at::Tensor& in, at::Tensor out, at::Tensor storage,
		const at::optional<FunctionLambda>& func_between) const;

private:
	std::unique_ptr<nufft::NufftNormal> _nufftop;
};

)DOC";
		});

}
