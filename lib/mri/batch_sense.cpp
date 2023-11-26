module;

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

module batch_sense;

import <future>;
import precond;

// BATCH SENSE BASE

hasty::mri::BatchedSenseBase::BatchedSenseBase(const std::vector<DeviceContext>& contexts,
	const at::TensorList& coords, const at::optional<at::TensorList>& kdata, const at::optional<at::TensorList>& weights)
	:
	_dcontexts(contexts),
	_coords(coords.vec()),
	_kdata(kdata.has_value() ? (*kdata).vec() : TensorVec()),
	_weights(weights.has_value() ? (*weights).vec() : TensorVec())
{
	_tpool = std::make_unique<ContextThreadPool<DeviceContext>>(_dcontexts);

	auto& smap = _dcontexts[0].smaps;

	_ndim = smap.sizes().size() - 1;
	_nmodes.resize(smap.sizes().size());
	_nmodes[0] = 1;
	for (int i = 1; i < _nmodes.size(); ++i) {
		_nmodes[i] = smap.size(i);
	}
}


// BATCH SENSE

hasty::mri::BatchedSense::BatchedSense(
	const std::vector<DeviceContext>& contexts,
	const at::TensorList& coords,
	const at::optional<at::TensorList>& kdata,
	const at::optional<at::TensorList>& weights)
	: BatchedSenseBase(contexts, coords, kdata, weights)
{}

void hasty::mri::BatchedSense::apply(const at::Tensor& in, at::TensorList out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = in.size(0);
	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [this, &in, &out, &coils, &manips, outer_batch](DeviceContext& context) {
			apply_outer_batch(in.select(0, outer_batch), out[outer_batch], coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(_tpool->enqueue(batch_applier));

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

void hasty::mri::BatchedSense::apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
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

	int n_inner_batches = instore.size(0);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	mri::CUDASense sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	at::Tensor out_cu;
	{
		at::Tensor out_cu_view = out.select(0, 0);
		out_cu = at::empty_like(out_cu_view, out_cu_view.options().device(dctxt.stream.device()));
	}

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = instore.select(0, inner_batch).unsqueeze(0);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);
		at::Tensor out_inner_batch_cpu_view = out.select(0, inner_batch);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}


		mri::CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage, kspace_storage,
			coilmanip.preapplier, coilmanip.postapplier);

		// no need to mutex synchronize in forward since devices split over vector not inside tensors
		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}

}

hasty::mri::OuterManipulator hasty::mri::BatchedSense::standard_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setPostApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0));
					});
				});
		});
}

hasty::mri::OuterManipulator hasty::mri::BatchedSense::standard_weighted_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setPostApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.mul_(data.weights);
					});
				});
		});
}

hasty::mri::OuterManipulator hasty::mri::BatchedSense::standard_weighted_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setPostApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0)).mul_(data.weights);
					});
				});
		});
}


// BATCH SENSE ADJOINT

hasty::mri::BatchedSenseAdjoint::BatchedSenseAdjoint(
	const std::vector<DeviceContext>& contexts,
	const at::TensorList& coords,
	const at::optional<at::TensorList>& kdata,
	const at::optional<at::TensorList>& weights)
	: BatchedSenseBase(contexts, coords, kdata, weights)
{}

void hasty::mri::BatchedSenseAdjoint::apply(const at::TensorList& in, at::Tensor out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = in.size();
	if (out.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [this, &in, &out, &coils, &manips, outer_batch](DeviceContext& context) {
			apply_outer_batch(in[outer_batch], out.select(0, outer_batch), coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(_tpool->enqueue(batch_applier));

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

void hasty::mri::BatchedSenseAdjoint::apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils, DeviceContext& dctxt, int32_t outer_batch, const OuterManipulator& outmanip)
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

	int n_inner_batches = instore.size(0);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	mri::CUDASenseAdjoint sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	at::Tensor out_cu;
	{
		at::Tensor out_cu_view = out.select(0, 0).unsqueeze(0);
		out_cu = at::empty_like(out_cu_view, out_cu_view.options().device(dctxt.stream.device()));
	}
		

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = instore.select(0, inner_batch);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);
		at::Tensor out_inner_batch_cpu_view = out.select(0, inner_batch).unsqueeze(0);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}

		mri::CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage, kspace_storage,
			coilmanip.preapplier, coilmanip.postapplier);

		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}
}

hasty::mri::OuterManipulator hasty::mri::BatchedSenseAdjoint::standard_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setPreApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0));
					});
				});
		});
}

hasty::mri::OuterManipulator hasty::mri::BatchedSenseAdjoint::standard_weighted_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setPreApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.mul_(data.weights);
					});
				});
		});
}

hasty::mri::OuterManipulator hasty::mri::BatchedSenseAdjoint::standard_weighted_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setPreApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0)).mul_(data.weights);
					});
				});
		});
}


// BATCH SENSE NORMAL

hasty::mri::BatchedSenseNormal::BatchedSenseNormal(
	const std::vector<DeviceContext>& contexts,
	const at::TensorList& coords,
	const at::optional<at::TensorList>& kdata,
	const at::optional<at::TensorList>& weights)
	: BatchedSenseBase(contexts, coords, kdata, weights)
{}

void hasty::mri::BatchedSenseNormal::apply(const at::Tensor& in, at::Tensor out, const std::vector<std::vector<int64_t>>& coils, const OuterManipulator& manips)
{
	c10::InferenceMode im_mode;

	int n_outer_batch = in.size(0);
	if (out.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image output to apply should be (N+2)D tensor");
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;


	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [this, &in, &out, &coils, &manips, outer_batch](DeviceContext& context) {
			apply_outer_batch(in.select(0, outer_batch), out.select(0, outer_batch), coils[outer_batch], context, outer_batch, manips);
		};

		futures.emplace_back(_tpool->enqueue(batch_applier));

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

void hasty::mri::BatchedSenseNormal::apply_outer_batch(const at::Tensor& in, at::Tensor out, const std::vector<int64_t>& coils,
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

	int n_inner_batches = instore.size(0);
	at::Tensor coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	mri::CUDASenseNormal sense(coord_cu, _nmodes);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor imspace_storage = at::empty(at::makeArrayRef(_nmodes), instore.options().device(dctxt.stream.device()));
	at::Tensor kspace_storage = at::empty({ 1, coord_cu.size(1) }, instore.options().device(dctxt.stream.device()));

	at::Tensor out_cu;
	{
		at::Tensor out_cu_view = out.select(0, 0).unsqueeze(0);
		out_cu = at::empty_like(out_cu_view, out_cu_view.options().device(dctxt.stream.device()));
	}

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		at::Tensor in_inner_batch_cpu_view = instore.select(0, inner_batch).unsqueeze(0);
		at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);
		at::Tensor out_inner_batch_cpu_view = out.select(0, inner_batch).unsqueeze(0);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		InnerData data{ weights_cu, kdata_cu };

		if (inmanip.preapplier.has_value()) {
			(*inmanip.preapplier)(in_cu, inner_batch, data, dctxt.stream);
		}

		mri::CoilManipulator coilmanip = inmanip.getCoilManipulator(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, imspace_storage, kspace_storage,
			coilmanip.preapplier, coilmanip.midapplier, coilmanip.postapplier);

		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	if (outmanip.postapplier.has_value()) {
		(*outmanip.postapplier)(out, outer_batch, dctxt.stream);
	}
}

hasty::mri::OuterManipulator hasty::mri::BatchedSenseNormal::standard_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setMidApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0));
					});
				});
		});
}

hasty::mri::OuterManipulator hasty::mri::BatchedSenseNormal::standard_weighted_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setMidApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.mul_(data.weights);
					});
				});
		});
}

hasty::mri::OuterManipulator hasty::mri::BatchedSenseNormal::standard_weighted_kdata_manipulator()
{
	return OuterManipulator([](int32_t outer_batch, at::Stream stream)
		{
			return InnerManipulator([](int32_t inner_batch, const InnerData& data, at::Stream stream) {
				return mri::CoilManipulator().setMidApply([&data](at::Tensor& tensor, int32_t coil) {
					tensor.sub_(data.kdata.select(0, coil).unsqueeze(0)).mul_(data.weights);
					});
				});
		});
}


// SENSE LOADER FOR BATCH CG

hasty::mri::SenseBatchConjugateGradientLoader::SenseBatchConjugateGradientLoader(
	std::vector<at::Tensor> coords, std::vector<int64_t> nmodes,
	std::vector<at::Tensor> kdata, at::Tensor smaps,
	std::shared_ptr<op::Admm::Context> ctx,
	at::optional<std::vector<at::Tensor>> preconds)
	: _coords(std::move(coords)), _nmodes(std::move(nmodes)), 
	_kdata(std::move(kdata)), _smaps(std::move(smaps)), 
	_ctx(std::move(ctx)), _preconds(std::move(preconds))
{

}

hasty::op::ConjugateGradientLoadResult hasty::mri::SenseBatchConjugateGradientLoader::load(SenseDeviceContext& dctx, size_t idx)
{
	c10::InferenceMode im_guard;
	c10::cuda::CUDAStreamGuard guard(dctx.stream);

	auto device = dctx.stream.device();

	auto coords = _coords[idx].to(device, true);
	auto kdata = _kdata[idx].to(device, true);

	std::vector<int64_t> coils(dctx.smaps.size(0));
	std::generate(coils.begin(), coils.end(), [n = int64_t(0)]() mutable { return n++; });

	std::shared_ptr<op::Operator> CGop;

	std::shared_ptr<op::SenseNOp> SHS = op::SenseNOp::Create(coords, _nmodes, dctx.smaps, coils);
	std::shared_ptr<op::AdjointableOp> AHA;

	std::shared_ptr<op::AdjointableOp> AH;
	std::shared_ptr<op::AdjointableOp> A;
	std::shared_ptr<op::AdjointableOp> B;

	op::Vector z;
	op::Vector u;
	std::optional<op::Vector> c;

	op::Vector CGvec;

	{
		std::unique_lock<std::mutex> lock(_ctx->ctxmut);

		auto localA = op::downcast<op::AdjointableVStackedOp>(_ctx->A);
		if (!localA)
			throw std::runtime_error("SenseAdmmLoader requires A to be AdjointableVStackedOp");
		A = op::downcast<op::AdjointableOp>(localA->get_slice_ptr(idx)->to_device(dctx.stream));
		AH = A->adjoint();

		auto localB = op::downcast<op::AdjointableVStackedOp>(_ctx->B);
		if (!localB)
			throw std::runtime_error("SenseAdmmLoader requires B to be AdjointableVStackedOp");
		B = op::downcast<op::AdjointableOp>(localB->get_slice_ptr(idx)->to_device(dctx.stream));

		if (_ctx->AHA != nullptr) {
			auto localAHA = op::downcast<op::AdjointableVStackedOp>(_ctx->AHA);
			if (!localAHA)
				throw std::runtime_error("SenseAdmmLoader requires AHA to be AdjointableVStackedOp");

			AHA = op::downcast<op::AdjointableOp>(std::move(localAHA->get_slice_ptr(idx)->to_device(dctx.stream)));
		}
		else {
			AHA = op::upcast<op::AdjointableOp>(std::move(op::mul(AH, A)));
		}

		if (!AHA)
			throw std::runtime_error("AHA could not be cast to an AdjointableOp on this device");

		z = _ctx->z[idx].copy().to_device(dctx.stream);
		u = _ctx->u[idx].copy().to_device(dctx.stream);
		if (_ctx->c.has_value())
			c = (*_ctx->c)[idx].copy().to_device(dctx.stream);
	}

	AHA = op::mul(at::tensor(0.5 * _ctx->rho), AHA);

	CGop = op::staticcast<op::AdjointableOp>(op::add(std::move(SHS), std::move(AHA)));

	if (c.has_value())
		CGvec = _ctx->rho * AH->apply(*c - B->apply(z) - u);
	else
		CGvec = (0.5 * _ctx->rho) * AH->apply(B->apply(z) - u);

	CGvec += SHS->apply_backward(op::Vector(kdata));

	return op::ConjugateGradientLoadResult{ std::move(CGop), std::move(CGvec), _preconds.has_value() ?
		mri::CirculantPreconditionerOp::Create((*_preconds)[idx], true, std::nullopt, false) : nullptr
	};
}


hasty::mri::BatchSenseAdmmMinimizer::BatchSenseAdmmMinimizer(
	std::shared_ptr<ContextThreadPool<SenseDeviceContext>> sensethreadpool,
	std::vector<at::Tensor> coords, std::vector<int64_t> nmodes, 
	std::vector<at::Tensor> kdata, at::Tensor smaps, std::shared_ptr<op::Admm::Context> ctx, 
	at::optional<std::vector<at::Tensor>> preconds)
{
	auto senseloader = std::make_shared<SenseBatchConjugateGradientLoader>(
		coords, nmodes, kdata, smaps, ctx, preconds);
	
	_batchcg = std::make_unique<BatchConjugateGradient<SenseDeviceContext>>(
		std::move(senseloader), 
		std::move(sensethreadpool)
	);

	_iters = 10;
	_tol = 0.0;
}

void hasty::mri::BatchSenseAdmmMinimizer::set_iters(int iters)
{
	_iters = iters;
}

void hasty::mri::BatchSenseAdmmMinimizer::set_tol(double tol)
{
	_tol = tol;
}

void hasty::op::AdmmMinimizer::solve(Admm::Context& ctx) 
{
	_batchcg->run(ctx.x, _iters, _tol);
}

