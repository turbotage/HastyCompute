#include "batch_sense.hpp"



// BATCH SENSE BASE

hasty::BatchedSenseBase::BatchedSenseBase(std::vector<DeviceContext>&& contexts, 
	const at::TensorList& coords, const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights)
	:
	_dcontexts(std::move(contexts)),
	_coords(coords.vec()),
	_kdata(kdata.has_value() ? (*kdata).vec() : TensorVec()),
	_weights(weights.has_value() ? (*weights).vec() : TensorVec())
{
	auto& smap = _dcontexts[0].smaps;

	_ndim = smap.sizes().size() - 1;
	_nmodes.resize(smap.sizes().size());
	_nmodes[0] = 1;
	for (int i = 1; i < _nmodes.size(); ++i) {
		_nmodes[i] = smap.size(i);
	}
}


// BATCH SENSE

hasty::BatchedSense::BatchedSense(
	std::vector<DeviceContext>&& contexts,
	const at::TensorList& coords,
	const at::optional<at::TensorList>& kdata,
	const at::optional<at::TensorList>& weights)
	: BatchedSenseBase(std::move(contexts), coords, kdata, weights)
{}

void hasty::BatchedSense::apply(const at::Tensor& in, at::TensorList out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = in.size(0);
	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;


	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		std::function<void(DeviceContext&)> batch_applier = [&](DeviceContext& context) {
			apply_outer_batch(in.select(0, outer_batch), out[outer_batch], coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(tpool.enqueue(batch_applier));

		if (futures.size() > 32 * _dcontexts.size()) {
			torch_util::future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		torch_util::future_catcher(futures.front());
		futures.pop_front();
	}

}

void hasty::BatchedSense::apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
	DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor instore;

	// Apply outer preapplier if applicable
	{
		if (outmanip.preapplier.has_value()) {
			instore = in.detach().clone();
			(*outmanip.preapplier)(instore, outer_batch, dctxt.stream);
		}
		else {
			instore = in;
		}
	}

	InnerManipulator inmanip = outmanip.getInnerManipulator(outer_batch, dctxt.stream);

	int n_inner_batches = instore.size(1);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	Sense sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = instore.select(0, inner_batch).unsqueeze(0);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}

		at::Tensor out_inner_batch_cpu_view = out.select(0, inner_batch);
		at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage, kspace_storage,
			coilmanip.preapplier, coilmanip.postapplier);

		// no need to mutex synchronize in forward since devices split over vector not inside tensors
		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}

}

hasty::OuterManipulator hasty::BatchedSense::standard_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setPostApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0));
					});
				});
		});
}

hasty::OuterManipulator hasty::BatchedSense::standard_weighted_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setPostApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.mul_(data.weights);
					});
				});
		});
}

hasty::OuterManipulator hasty::BatchedSense::standard_weighted_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setPostApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0)).mul_(data.weights);
					});
				});
		});
}


// BATCH SENSE ADJOINT

hasty::BatchedSenseAdjoint::BatchedSenseAdjoint(
	std::vector<DeviceContext>&& contexts,
	const at::TensorList& coords,
	const at::optional<at::TensorList>& kdata,
	const at::optional<at::TensorList>& weights)
	: BatchedSenseBase(std::move(contexts), coords, kdata, weights)
{}

void hasty::BatchedSenseAdjoint::apply(const at::TensorList& in, at::Tensor out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = in.size();
	if (out.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		std::function<void(DeviceContext&)> batch_applier = [&](DeviceContext& context) {
			apply_outer_batch(in[outer_batch], out.select(0, outer_batch), coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(tpool.enqueue(batch_applier));

		if (futures.size() > 32 * _dcontexts.size()) {
			torch_util::future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		torch_util::future_catcher(futures.front());
		futures.pop_front();
	}
}

void hasty::BatchedSenseAdjoint::apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils, DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor instore;

	// Apply outer preapplier if applicable
	{
		if (outmanip.preapplier.has_value()) {
			instore = in.detach().clone();
			(*outmanip.preapplier)(instore, outer_batch, dctxt.stream);
		}
		else {
			instore = in;
		}
	}

	InnerManipulator inmanip = outmanip.getInnerManipulator(outer_batch, dctxt.stream);

	int n_inner_batches = instore.size(1);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	SenseAdjoint sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	at::Tensor out_outer_batch_cpu_view = out.select(0, outer_batch);

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = instore.select(0, inner_batch);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}

		at::Tensor out_inner_batch_cpu_view = out_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage, kspace_storage,
			coilmanip.preapplier, coilmanip.postapplier);

		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}
}

hasty::OuterManipulator hasty::BatchedSenseAdjoint::standard_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setPreApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0));
					});
				});
		});
}

hasty::OuterManipulator hasty::BatchedSenseAdjoint::standard_weighted_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setPreApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.mul_(data.weights);
					});
				});
		});
}

hasty::OuterManipulator hasty::BatchedSenseAdjoint::standard_weighted_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setPreApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0)).mul_(data.weights);
					});
				});
		});
}


// BATCH SENSE NORMAL

hasty::BatchedSenseNormal::BatchedSenseNormal(
	std::vector<DeviceContext>&& contexts,
	const at::TensorList& coords,
	const at::optional<at::TensorList>& kdata,
	const at::optional<at::TensorList>& weights)
	: BatchedSenseBase(std::move(contexts), coords, kdata, weights)
{}

void hasty::BatchedSenseNormal::apply(const at::Tensor& in, at::Tensor out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = in.size(0);
	if (out.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image output to apply should be (N+2)D tensor");
	}

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;


	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		std::function<void(DeviceContext&)> batch_applier = [&](DeviceContext& context) {
			apply_outer_batch(in.select(0, outer_batch), out.select(0, outer_batch), coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(tpool.enqueue(batch_applier));

		if (futures.size() > 32 * _dcontexts.size()) {
			torch_util::future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		torch_util::future_catcher(futures.front());
		futures.pop_front();
	}
}

/*
void hasty::BatchedSenseNormal::apply_addinplace(at::Tensor& inout, const at::Tensor& scalar, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = inout.size(0);
	if (out.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;


	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		std::function<void(DeviceContext&)> batch_applier = [&](DeviceContext& context) {
			apply_addinplace_outer_batch(inout, scalar, coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(tpool.enqueue(batch_applier));

		if (futures.size() > 32 * _dcontexts.size()) {
			torch_util::future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		torch_util::future_catcher(futures.front());
		futures.pop_front();
	}
}
*/

void hasty::BatchedSenseNormal::apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
	DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor instore;

	// Apply outer preapplier if applicable
	{
		if (outmanip.preapplier.has_value()) {
			instore = in.detach().clone();
			(*outmanip.preapplier)(instore, outer_batch, dctxt.stream);
		}
		else {
			instore = in;
		}
	}

	InnerManipulator inmanip = outmanip.getInnerManipulator(outer_batch, dctxt.stream);

	int n_inner_batches = instore.size(1);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	SenseNormal sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	at::Tensor in_outer_batch_cpu_view = instore.select(0, outer_batch);
	at::Tensor out_outer_batch_cpu_view = out.select(0, outer_batch);

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = in_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}

		at::Tensor out_inner_batch_cpu_view = out_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage, kspace_storage,
			coilmanip.preapplier, coilmanip.midapplier, coilmanip.postapplier);

		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}
}

/*
void hasty::BatchedSenseNormal::apply_addinplace_outer_batch(at::Tensor& inout, const at::Tensor& scalar, const std::vector<int64_t>& coils, DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor instore;

	// Apply outer preapplier if applicable
	{
		if (outmanip.preapplier.has_value()) {
			instore = inout.detach().clone();
			(*outmanip.preapplier)(instore, outer_batch, dctxt.stream);
		}
		else {
			instore = inout;
		}
	}

	InnerManipulator inmanip = outmanip.getInnerManipulator(outer_batch, dctxt.stream);

	int n_inner_batches = instore.size(1);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	SenseNormal sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	at::Tensor in_outer_batch_cpu_view = instore.select(0, outer_batch);

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = in_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}

		at::Tensor out_inner_batch_cpu_view = out_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage, kspace_storage,
			coilmanip.preapplier, coilmanip.postapplier);

		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}
}
*/

hasty::OuterManipulator hasty::BatchedSenseNormal::standard_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setMidApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0));
					});
				});
		});
}

hasty::OuterManipulator hasty::BatchedSenseNormal::standard_weighted_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setMidApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.mul_(data.weights);
					});
				});
		});
}

hasty::OuterManipulator hasty::BatchedSenseNormal::standard_weighted_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return CoilManipulator().setMidApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0)).mul_(data.weights);
					});
				});
		});
}


// BATCH SENSE NORMAL ADJOINT

hasty::BatchedSenseNormalAdjoint::BatchedSenseNormalAdjoint(
	std::vector<DeviceContext>&& contexts,
	const at::TensorList& coords,
	const at::optional<at::TensorList>& kdata,
	const at::optional<at::TensorList>& weights)
	: BatchedSenseBase(std::move(contexts), coords, kdata, weights)
{}

void hasty::BatchedSenseNormalAdjoint::apply(const at::TensorList& in, at::TensorList out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = in.size();
	if (out[0].sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image output to apply should be (N+2)D tensor");
	}

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;


	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		std::function<void(DeviceContext&)> batch_applier = [&](DeviceContext& context) {
			apply_outer_batch(in[outer_batch], out[outer_batch], coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(tpool.enqueue(batch_applier));

		if (futures.size() > 32 * _dcontexts.size()) {
			torch_util::future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		torch_util::future_catcher(futures.front());
		futures.pop_front();
	}
}

void hasty::BatchedSenseNormalAdjoint::apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
	DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor instore;

	// Apply outer preapplier if applicable
	{
		if (outmanip.preapplier.has_value()) {
			instore = in.detach().clone();
			(*outmanip.preapplier)(instore, outer_batch, dctxt.stream);
		}
		else {
			instore = in;
		}
	}

	InnerManipulator inmanip = outmanip.getInnerManipulator(outer_batch, dctxt.stream);

	int n_inner_batches = instore.size(1);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	SenseNormalAdjoint sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = instore.select(0, inner_batch).unsqueeze(0);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}

		at::Tensor out_inner_batch_cpu_view = out.select(0, inner_batch).unsqueeze(0);
		at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage,
			coilmanip.preapplier, coilmanip.midapplier, coilmanip.postapplier);

		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}
}

