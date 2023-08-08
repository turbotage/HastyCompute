#include "py_svt.hpp"


namespace {

	std::vector<hasty::svt::BlocksSVTBase::DeviceContext> get_batch_contexts(const std::vector<c10::Stream>& streams)
	{
		return std::vector<hasty::svt::BlocksSVTBase::DeviceContext>(streams.begin(), streams.end());
	}

	std::vector<hasty::svt::BlocksSVTBase::DeviceContext> get_batch_contexts(const at::ArrayRef<c10::Stream>& streams, const at::Tensor& smaps)
	{
		return std::vector<hasty::svt::BlocksSVTBase::DeviceContext>(streams.begin(), streams.end());
	}

	std::vector<c10::Stream> get_streams(const at::optional<std::vector<c10::Stream>>& streams)
	{
		return hasty::torch_util::get_streams(streams);
	}

	std::vector<c10::Stream> get_streams(const at::optional<at::ArrayRef<c10::Stream>>& streams)
	{
		return hasty::torch_util::get_streams(streams);
	}

}



hasty::ffi::Random3DBlocksSVT::Random3DBlocksSVT(const at::optional<at::ArrayRef<at::Stream>>& streams)
	: _rbsvt(std::make_unique<hasty::svt::Random3DBlocksSVT>(get_batch_contexts(get_streams(streams))))
{
}

void hasty::ffi::Random3DBlocksSVT::apply(at::Tensor in, int64_t nblocks, const std::array<int64_t, 3>& block_shape, double thresh, bool soft)
{
	auto func = [this, &in, nblocks, &block_shape, thresh, soft]() {
		_rbsvt->apply(in, nblocks, block_shape, thresh, soft);
	};

	torch_util::future_catcher(func);
}


hasty::ffi::Normal3DBlocksSVT::Normal3DBlocksSVT(const at::optional<at::ArrayRef<at::Stream>>& streams)
	: _nbsvt(std::make_unique<hasty::svt::Normal3DBlocksSVT>(get_batch_contexts(get_streams(streams))))
{
}

void hasty::ffi::Normal3DBlocksSVT::apply(at::Tensor in, const std::array<int64_t, 3>& block_strides, const std::array<int64_t, 3>& block_shape,
	int64_t block_iter, double thresh, bool soft)
{
	auto func = [this, &in, &block_strides, &block_shape, block_iter, thresh, soft]() {
		_nbsvt->apply(in, block_strides, block_shape, block_iter, thresh, soft);
	};

	torch_util::future_catcher(func);
}



TORCH_LIBRARY(HastySVT, hsvt) {

	hsvt.class_<hasty::ffi::Random3DBlocksSVT>("Random3DBlocksSVT")
		.def(torch::init<const at::optional<at::ArrayRef<at::Stream>>&>())
		.def("apply", &hasty::ffi::Random3DBlocksSVT::apply);

	hsvt.class_<hasty::ffi::Normal3DBlocksSVT>("Normal3DBlocksSVT")
		.def(torch::init<const at::optional<at::ArrayRef<at::Stream>>&>())
		.def("apply", &hasty::ffi::Normal3DBlocksSVT::apply);

}


