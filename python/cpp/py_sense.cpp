#include "py_sense.hpp"

import torch_util;

import sense;

hasty::ffi::Sense::Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
{
	if (coords.is_cuda()) {
		_cudasenseop = std::make_unique<hasty::mri::CUDASense>(coords, nmodes);
	}
	else {
		_senseop = std::make_unique<hasty::mri::Sense>(coords, nmodes);
	}
}

void hasty::ffi::Sense::apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage)
{
	auto func = [this, &in, &out, &smaps, &coils, &imspace_storage, &kspace_storage]() {
		if (_senseop) {
			_senseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, at::nullopt, at::nullopt);
		}
		else {
			_cudasenseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, at::nullopt, at::nullopt);
		}
	};

	torch_util::future_catcher(func);
}

hasty::ffi::SenseAdjoint::SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
{
	if (coords.is_cuda()) {
		_cudasenseop = std::make_unique<hasty::mri::CUDASenseAdjoint>(coords, nmodes);
	}
	else {
		_senseop = std::make_unique<hasty::mri::SenseAdjoint>(coords, nmodes);
	}
}

void hasty::ffi::SenseAdjoint::apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage)
{
	auto func = [this, &in, &out, &smaps, &coils, &imspace_storage, &kspace_storage]() {
		if (_senseop) {
			_senseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, at::nullopt, at::nullopt);
		}
		else {
			_cudasenseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, at::nullopt, at::nullopt);
		}
	};

	torch_util::future_catcher(func);
}

hasty::ffi::SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
{
	if (coords.is_cuda()) {
		_cudasenseop = std::make_unique<hasty::mri::CUDASenseNormal>(coords, nmodes);
	}
	else {
		_senseop = std::make_unique<hasty::mri::SenseNormal>(coords, nmodes);
	}
}

void hasty::ffi::SenseNormal::apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage)
{
	auto func = [this, &in, &out, &smaps, &coils, &imspace_storage, &kspace_storage]() {
		if (_senseop) {
			_senseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, at::nullopt, at::nullopt, at::nullopt);
		}
		else {
			_cudasenseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, at::nullopt, at::nullopt, at::nullopt);
		}
	};

	torch_util::future_catcher(func);
}


TORCH_LIBRARY(HastySense, hs) {

	hs.class_<hasty::ffi::Sense>("Sense")
		.def(torch::init<const at::Tensor&, const std::vector<int64_t>&>())
		.def("apply", &hasty::ffi::Sense::apply);

	hs.class_<hasty::ffi::SenseAdjoint>("SenseAdjoint")
		.def(torch::init<const at::Tensor&, const std::vector<int64_t>&>())
		.def("apply", &hasty::ffi::SenseAdjoint::apply);

	hs.class_<hasty::ffi::SenseNormal>("SenseNormal")
		.def(torch::init<const at::Tensor&, const std::vector<int64_t>&>())
		.def("apply", &hasty::ffi::SenseNormal::apply);


	hs.def("doc", []() -> std::string {
		return
R"DOC(

// HastySense Module

class LIB_EXPORT Sense : public torch::CustomClassHolder {
public:

	Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

	void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
		const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

private:
	std::unique_ptr<hasty::sense::Sense> _senseop;
};

class LIB_EXPORT SenseAdjoint : public torch::CustomClassHolder {
public:

	SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

	void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
		const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

private:
	std::unique_ptr<hasty::sense::SenseAdjoint> _senseop;
};

class LIB_EXPORT SenseNormal : public torch::CustomClassHolder {
public:

	SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes);

	void apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
		const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage);

private:
	std::unique_ptr<hasty::sense::SenseNormal> _senseop;
};

)DOC";
		});


}

