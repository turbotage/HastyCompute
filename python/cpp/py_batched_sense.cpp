#include "py_batched_sense.hpp"


namespace {
	std::vector<hasty::BatchedSense::DeviceContext> get_batch_contexts(const std::vector<c10::Stream>& streams, const at::Tensor& smaps)
	{
		std::unordered_map<c10::Device, at::Tensor> smaps_dict;
		std::vector<hasty::BatchedSense::DeviceContext> contexts(streams.begin(), streams.end());
		for (auto& context : contexts) {
			const auto& device = context.stream.device();
			if (smaps_dict.contains(device)) {
				context.smaps = smaps_dict.at(device);
			}
			else {
				context.smaps = smaps.to(device, true);
				smaps_dict.insert({ device, context.smaps });
			}
		}
		return contexts;
	}

	std::vector<hasty::BatchedSense::DeviceContext> get_batch_contexts(const at::ArrayRef<c10::Stream>& streams, const at::Tensor& smaps)
	{
		std::unordered_map<c10::Device, at::Tensor> smaps_dict;
		std::vector<hasty::BatchedSense::DeviceContext> contexts(streams.begin(), streams.end());
		for (auto& context : contexts) {
			const auto& device = context.stream.device();
			if (smaps_dict.contains(device)) {
				context.smaps = smaps_dict.at(device);
			}
			else {
				context.smaps = smaps.to(device, true);
				smaps_dict.insert({ device, context.smaps });
			}
		}
		return contexts;
	}

	std::vector<c10::Stream> get_streams(const at::optional<std::vector<c10::Stream>>& streams)
	{
		return hasty::torch_util::get_streams(streams);
	}

	std::vector<c10::Stream> get_streams(const at::optional<at::ArrayRef<c10::Stream>>& streams)
	{
		return hasty::torch_util::get_streams(streams);
	}

	std::vector<std::vector<int64_t>> get_coils(int32_t nouter, int32_t ncoil) {
		std::vector<std::vector<int64_t>> coilsout(nouter);
		std::vector<int64_t> coilspec(ncoil);
		std::iota(coilspec.begin(), coilspec.end(), 0);
		for (int i = 0; i < nouter; ++i) {
			coilsout[i] = coilspec;
		}
		return coilsout;
	}

}

// BATCHED SENSE

hasty::ffi::BatchedSense::BatchedSense(const at::TensorList& coords, const at::Tensor& smaps,
	const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
	const at::optional<at::ArrayRef<at::Stream>>& streams)
{
	std::vector<hasty::BatchedSenseBase::DeviceContext> contexts = get_batch_contexts(get_streams(streams), smaps);

	_bs = std::make_unique<hasty::BatchedSense>(std::move(contexts), coords, kdata, weights);
}


void hasty::ffi::BatchedSense::apply(const at::Tensor& in, at::TensorList out, const at::optional<std::vector<std::vector<int64_t>>>& coils)
{
	auto func = [this, &in, &out, &coils]() {
		if (_bs->has_kdata() && _bs->has_weights()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSense::standard_weighted_kdata_manipulator());
		}
		else if (_bs->has_kdata()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSense::standard_kdata_manipulator());
		}
		else if (_bs->has_weights()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSense::standard_weighted_manipulator());
		}
		else {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()), OuterManipulator());
		}
	};
	
	torch_util::future_catcher(func);
}

// BATCHED SENSE ADJOINT

hasty::ffi::BatchedSenseAdjoint::BatchedSenseAdjoint(const at::TensorList& coords, const at::Tensor& smaps,
	const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
	const at::optional<at::ArrayRef<at::Stream>>& streams)
{
	std::vector<hasty::BatchedSenseBase::DeviceContext> contexts = get_batch_contexts(get_streams(streams), smaps);

	_bs = std::make_unique<hasty::BatchedSenseAdjoint>(std::move(contexts), coords, kdata, weights);
}


void hasty::ffi::BatchedSenseAdjoint::apply(const at::TensorList& in, at::Tensor out, const at::optional<std::vector<std::vector<int64_t>>>& coils)
{
	auto func = [this, &in, &out, &coils]() {
		if (_bs->has_kdata() && _bs->has_weights()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSenseAdjoint::standard_weighted_kdata_manipulator());
		}
		else if (_bs->has_kdata()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSenseAdjoint::standard_kdata_manipulator());
		}
		else if (_bs->has_weights()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSenseAdjoint::standard_weighted_manipulator());
		}
		else {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()), OuterManipulator());
		}
	};

	torch_util::future_catcher(func);
}

// BATCHED SENSE NORMAL

hasty::ffi::BatchedSenseNormal::BatchedSenseNormal(const at::TensorList& coords, const at::Tensor& smaps,
	const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
	const at::optional<at::ArrayRef<at::Stream>>& streams)
{
	std::vector<hasty::BatchedSenseBase::DeviceContext> contexts = get_batch_contexts(get_streams(streams), smaps);

	_bs = std::make_unique<hasty::BatchedSenseNormal>(std::move(contexts), coords, kdata, weights);
}


void hasty::ffi::BatchedSenseNormal::apply(const at::Tensor& in, at::Tensor out, const at::optional<std::vector<std::vector<int64_t>>>& coils)
{
	auto func = [this, &in, &out, &coils]() {
		if (_bs->has_kdata() && _bs->has_weights()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSenseNormal::standard_weighted_kdata_manipulator());
		}
		else if (_bs->has_kdata()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSenseNormal::standard_kdata_manipulator());
		}
		else if (_bs->has_weights()) {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()),
				hasty::BatchedSenseNormal::standard_weighted_manipulator());
		}
		else {
			_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()), OuterManipulator());
		}
	};

	torch_util::future_catcher(func);
}

// BATCHED SENSE NORMAL

hasty::ffi::BatchedSenseNormalAdjoint::BatchedSenseNormalAdjoint(const at::TensorList& coords, const at::Tensor& smaps,
	const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights,
	const at::optional<at::ArrayRef<at::Stream>>& streams)
{
	std::vector<hasty::BatchedSenseBase::DeviceContext> contexts = get_batch_contexts(get_streams(streams), smaps);

	_bs = std::make_unique<hasty::BatchedSenseNormalAdjoint>(std::move(contexts), coords, kdata, weights);
}


void hasty::ffi::BatchedSenseNormalAdjoint::apply(const at::TensorList& in, at::TensorList out, const at::optional<std::vector<std::vector<int64_t>>>& coils)
{
	auto func = [this, &in, &out, &coils]() {
		_bs->apply(in, out, coils.has_value() ? *coils : get_coils(_bs->nouter_batches(), _bs->ncoils()), OuterManipulator());
	};

	torch_util::future_catcher(func);
}







void dummy::dummy(at::TensorList tensorlist)
{
	std::cout << tensorlist << std::endl;
	for (auto tensor : tensorlist) {
		tensor += 1.0;
	}
}

void dummy::stream_dummy(const at::optional<at::ArrayRef<at::Stream>>& streams, const torch::Tensor& in)
{
	if (streams.has_value()) {
		for (auto& stream : *streams) {
			std::cout << stream << std::endl;
		}
	}
	std::cout << in << std::endl;
}



using namespace at;

TORCH_LIBRARY(HastyBatchedSense, hbs) {

	hbs.class_<hasty::ffi::BatchedSense>("BatchedSense")
		.def(torch::init<const at::TensorList&, const at::Tensor&,
			const at::optional<at::TensorList>&, const at::optional<at::TensorList>&,
			const at::optional<at::ArrayRef<at::Stream>>&>())
		.def("apply", &hasty::ffi::BatchedSense::apply);

	hbs.class_<hasty::ffi::BatchedSenseAdjoint>("BatchedSenseAdjoint")
		.def(torch::init<const at::TensorList&, const at::Tensor&,
			const at::optional<at::TensorList>&, const at::optional<at::TensorList>&,
			const at::optional<at::ArrayRef<at::Stream>>&>())
		.def("apply", &hasty::ffi::BatchedSenseAdjoint::apply);

	hbs.class_<hasty::ffi::BatchedSenseNormal>("BatchedSenseNormal")
		.def(torch::init<const at::TensorList&, const at::Tensor&,
			const at::optional<at::TensorList>&, const at::optional<at::TensorList>&,
			const at::optional<at::ArrayRef<at::Stream>>&>())
		.def("apply", &hasty::ffi::BatchedSenseNormal::apply);

	hbs.class_<hasty::ffi::BatchedSenseNormalAdjoint>("BatchedSenseNormalAdjoint")
		.def(torch::init<const at::TensorList&, const at::Tensor&,
			const at::optional<at::TensorList>&, const at::optional<at::TensorList>&,
			const at::optional<at::ArrayRef<at::Stream>>&>())
		.def("apply", &hasty::ffi::BatchedSenseNormalAdjoint::apply);

	//hbs.def("stream_dummy", &dummy::stream_dummy);
	hbs.def("dummy", dummy::dummy);
	//hbs.def("stream_dummy", dummy::stream_dummy);
}