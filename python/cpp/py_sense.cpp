#include "py_sense.hpp"

hasty::ffi::Sense::Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _senseop(std::make_unique<hasty::sense::Sense>(coords, nmodes))
{
}

void hasty::ffi::Sense::apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage)
{
	auto func = [this, &in, &out, &smaps, &coils, &imspace_storage, &kspace_storage]() {
		_senseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, std::nullopt, std::nullopt);
	};

	torch_util::future_catcher(func);
}

hasty::ffi::SenseAdjoint::SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _senseop(std::make_unique<hasty::sense::SenseAdjoint>(coords, nmodes))
{
}

void hasty::ffi::SenseAdjoint::apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage)
{
	auto func = [this, &in, &out, &smaps, &coils, &imspace_storage, &kspace_storage]() {
		_senseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, std::nullopt, std::nullopt);
	};

	torch_util::future_catcher(func);
}

hasty::ffi::SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _senseop(std::make_unique<hasty::sense::SenseNormal>(coords, nmodes))
{
}

void hasty::ffi::SenseNormal::apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage)
{
	auto func = [this, &in, &out, &smaps, &coils, &imspace_storage, &kspace_storage]() {
		_senseop->apply(in, out, smaps, coils, imspace_storage, kspace_storage, std::nullopt, std::nullopt, std::nullopt);
	};

	torch_util::future_catcher(func);
}

hasty::ffi::SenseNormalAdjoint::SenseNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _senseop(std::make_unique<hasty::sense::SenseNormalAdjoint>(coords, nmodes))
{
}

void hasty::ffi::SenseNormalAdjoint::apply(const at::Tensor& in, at::Tensor out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage)
{
	auto func = [this, &in, &out, &smaps, &coils, &imspace_storage]() {
		_senseop->apply(in, out, smaps, coils, imspace_storage, std::nullopt, std::nullopt, std::nullopt);
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

	hs.class_<hasty::ffi::SenseNormalAdjoint>("SenseNormalAdjoint")
		.def(torch::init<const at::Tensor&, const std::vector<int64_t>&>())
		.def("apply", &hasty::ffi::SenseNormalAdjoint::apply);

}

