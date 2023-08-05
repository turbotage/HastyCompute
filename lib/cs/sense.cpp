#include "sense.hpp"
#include <numeric>

#include <c10/cuda/CUDAGuard.h>

using namespace hasty;

Sense::Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _nufft(coords, nmodes, NufftOptions::type2()), _nmodes(nmodes)
{
	if (_nmodes[0] != 1) {
		throw std::runtime_error("Only ntransf==1 allowed for Sense operator");
	}
}

at::Tensor hasty::Sense::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
	const std::optional<CoilApplier>& premanip,
	const std::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	} else {
		imstore = at::empty_like(in);
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	} else {
		kstore = at::empty_like(out.select(0, 0).unsqueeze(0));
	}

	bool accumulate = out.size(0) == 1;

	for (auto coil : coils) {

		imstore.copy_(in);

		if (premanip.has_value()) {
			(*premanip)(imstore, coil);
		}

		at::Tensor smap = smaps.select(0, coil).unsqueeze(0);
		imstore.mul_(smap);

		_nufft.apply(imstore, kstore);

		if (postmanip.has_value()) {
			(*postmanip)(kstore, coil);
		}

		if (accumulate) {
			out.select(0, 0).unsqueeze(0).add_(kstore);
		}
		else {
			out.select(0, coil).unsqueeze(0).add_(kstore);
		}
	}

}


hasty::SenseAdjoint::SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _nufft(coords, nmodes, NufftOptions::type1()), _nmodes(nmodes)
{
	if (_nmodes[0] != 1) {
		throw std::runtime_error("Only ntransf==1 allowed for SenseAdjoint operator");
	}
}

at::Tensor hasty::SenseAdjoint::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage, 
	const std::optional<CoilApplier>& premanip, const std::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty_like(out.select(0, 0).unsqueeze(0));
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty_like(in.select(0,0).unsqueeze(0));
	}

	bool accumulate = out.size(0) == 1;

	for (auto coil : coils) {

		kstore.copy_(in.select(0, coil).unsqueeze(0));

		if (premanip.has_value()) {
			(*premanip)(kstore, coil);
		}
		
		_nufft.apply(kstore, imstore);

		at::Tensor smap = smaps.select(0, coil).unsqueeze(0);
		imstore.mul_(smap.conj());


		if (postmanip.has_value()) {
			(*postmanip)(imstore, coil);
		}

		if (accumulate) {
			out.select(0, 0).unsqueeze(0).add_(kstore);
		}
		else {
			out.select(0, coil).unsqueeze(0).add_(kstore);
		}
	}
}


SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _normal_nufft(coords, nmodes, NufftOptions::type2(), NufftOptions::type1()), _nmodes(nmodes)
{
	if (_nmodes[0] != 1) {
		throw std::runtime_error("Only ntransf==1 allowed for SenseNormal operator");
	}
}

void hasty::SenseNormal::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
	const std::optional<CoilApplier>& premanip,
	const std::optional<CoilApplier>& midmanip,
	const std::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	} else {
		imstore = at::empty_like(in);
	}

	out.zero_();


	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	} else {
		kstore = at::empty({ 1, _normal_nufft.nfreq() }, in.options());
	}

	for (auto coil : coils) {
		at::Tensor smap = smaps.select(0, coil).unsqueeze(0);

		if (premanip.has_value()) {
			imstore.copy_(in);
			(*premanip)(imstore, coil);
			imstore.mul_(smap);
		} else {
			at::mul_out(imstore, in, smap);
		}


		if (midmanip.has_value()) {
			_normal_nufft.apply_inplace(imstore, kstore, std::bind(*midmanip, std::placeholders::_1, coil));
		} else {
			_normal_nufft.apply_inplace(imstore, kstore, std::nullopt);
		}

		if (postmanip.has_value()) {
			imstore.mul_(smap.conj());
			(*postmanip)(imstore, coil);
			out.add_(imstore);
		} else {
			out.addcmul_(imstore, smap.conj());
		}
	}
}


hasty::SenseNormalAdjoint::SenseNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _normal_nufft(coords, nmodes, NufftOptions::type1(), NufftOptions::type1()), _nmodes(nmodes)
{
	if (_nmodes[0] != 1) {
		throw std::runtime_error("Only ntransf==1 allowed for SenseNormalAdjoint operator");
	}
}

void hasty::SenseNormalAdjoint::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const std::optional<at::Tensor>& imspace_storage,
	const std::optional<CoilApplier>& premanip,
	const std::optional<CoilApplier>& midmanip,
	const std::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	} else {
		imstore = at::empty(at::makeArrayRef(_nmodes), in.options());
	}

	at::Tensor kstore;

	for (auto coil : coils) {
		at::Tensor smap = smaps.select(0, coil).unsqueeze(0);
		
		kstore = out.select(0, coil).unsqueeze(0);
		kstore.copy_(in.select(0, coil).unsqueeze(0));

		if (premanip.has_value()) {
			(*premanip)(kstore, coil);
		}

		std::function<void(at::Tensor&)> total_midmanip = [coil, &smap, &midmanip](at::Tensor& in) {
			in.mul_(smap.conj());
			if (midmanip.has_value())
				(*midmanip)(in, coil);
			in.mul_(smap);
		};

		_normal_nufft.apply_inplace(kstore, imstore, total_midmanip);

		if (postmanip.has_value()) {
			(*postmanip)(kstore, coil);
		}

	}
}



SenseNormalToeplitz::SenseNormalToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, double tol)
	: _normal_nufft(coords, nmodes, tol, std::nullopt, std::nullopt, std::nullopt)
{

}

SenseNormalToeplitz::SenseNormalToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes)
	: _normal_nufft(std::move(diagonal), nmodes)
{

}

void SenseNormalToeplitz::apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage1, at::Tensor& storage2,
	const at::Tensor& smaps, const std::vector<int64_t>& coils) const
{
	c10::InferenceMode inference_guard;

	out.zero_();

	at::Tensor xstore = at::empty_like(in);

	for (auto coil : coils) {

		at::Tensor smap = smaps.select(0, coil).unsqueeze(0);
		at::mul_out(xstore, in, smap);

		_normal_nufft.apply_addcmul(xstore, out, smap.conj(), storage1, storage2);
	}
}




BatchedSense::BatchedSense(
	std::vector<DeviceContext>&& contexts,
	const at::TensorList& coords,
	const std::optional<at::TensorList>& kdata,
	const std::optional<at::TensorList>& weights)
	:
	_dcontexts(std::move(contexts)),
	_coords(coords.vec()),
	_kdata(kdata.has_value() ? (*kdata).vec() : TensorVec()),
	_weights(weights.has_value() ? (*weights).vec() : TensorVec())
{
	construct();
}

void BatchedSense::construct()
{
	auto& smap = _dcontexts[0].smaps;

	_ndim = smap.sizes().size() - 1;
	_nmodes.resize(smap.sizes().size());
	_nmodes[0] = 1; 
	for (int i = 1; i < _nmodes.size(); ++i) {
		_nmodes[i] = smap.size(i);
	}
}

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
			apply_outer_batch(in, out[outer_batch], coils[outer_batch], context, outer_batch, manips);
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
			instore = in.select(0, outer_batch).detach().clone();
			(*outmanip.preapplier)(instore, outer_batch, dctxt.stream);
		} else {
			instore = in.select(0, outer_batch);
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

		std::optional<CoilApplier> coil_applier = inmanip.getCoilApplier(inner_batch, data, dctxt.stream);

		sense.apply(in_cu, out_cu, coils, );

		// no need to mutex synchronize in forward since devices split over vector not inside tensors
		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}





	at::Tensor in_outer_batch_cpu_view = in.select(0, outer_batch);
	at::Tensor in_inner_batch_cpu_view = in_outer_batch_cpu_view.select(0, 0).unsqueeze(0);
	at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);
	at::Tensor storage_cu = at::empty_like(in_cu);

	at::Tensor out_inner_batch_cpu_view = out.select(0, 0);
	at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);


	at::Tensor normtensor = at::scalar_tensor(at::Scalar((double)0.0)).to(dctxt.stream.device());

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		in_inner_batch_cpu_view = in_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		out_inner_batch_cpu_view = out.select(0, inner_batch);
		out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		// no need to mutex synchronize in forward since devices split over vector not inside tensors
		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	return normtensor.cpu();
}






at::Tensor hasty::BatchedSense::apply_forward(const at::Tensor& in, at::TensorList out,
	bool sum, bool sumnorm,
	const std::optional<std::vector<std::vector<int64_t>>>& coils,
	const std::optional<WeightedManipulator>& wmanip, 
	const std::optional<FreqManipulator>& fmanip, 
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	int n_outer_batch = in.size(0);

	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<std::vector<int64_t>> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<std::vector<int64_t>>(n_outer_batch);
		for (int i = 0; i < coilss.size(); ++i) {
			std::iota(coilss[i].begin(), coilss[i].end(), 0);
		}
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	std::mutex summut;
	at::Tensor normtensor = at::scalar_tensor(at::Scalar((double)0.0));

	auto outer_capture = [this, &summut, &normtensor, &in, &out, sum, sumnorm, &coilss, &wmanip, &fmanip, &wfmanip](DeviceContext& context, int32_t outer_batch)
	{
		at::Tensor outer_norm = apply_outer_batch_forward(context, outer_batch, in, out[outer_batch], sum, sumnorm, coilss[outer_batch], wmanip, fmanip, wfmanip);
		{
			std::lock_guard<std::mutex> lock(summut);
			normtensor.add_(outer_norm);
		} 
	};

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [outer_batch, &outer_capture](DeviceContext& context) { outer_capture(context, outer_batch); };

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

	return normtensor;
}

at::Tensor hasty::BatchedSense::apply_outer_batch_forward(DeviceContext& dctxt, int32_t outer_batch, const at::Tensor& in, at::Tensor out,
	bool sum, bool sumnorm,
	const std::vector<int64_t>& coils, const std::optional<WeightedManipulator>& wmanip,
	const std::optional<FreqManipulator>& fmanip, 
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor coord_cu;
	if (_coords.size() > 0) {
		coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	}

	Sense sense(coord_cu, _nmodes, false);
	int n_inner_batches = in.size(1);

	at::Tensor in_outer_batch_cpu_view = in.select(0, outer_batch);
	at::Tensor in_inner_batch_cpu_view = in_outer_batch_cpu_view.select(0, 0).unsqueeze(0);
	at::Tensor in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);
	at::Tensor storage_cu = at::empty_like(in_cu);
	
	at::Tensor out_inner_batch_cpu_view= out.select(0, 0);
	at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor normtensor = at::scalar_tensor(at::Scalar((double)0.0)).to(dctxt.stream.device());

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		in_inner_batch_cpu_view = in_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		out_inner_batch_cpu_view = out.select(0, inner_batch);
		out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip;

		if (wmanip.has_value()) {
			freq_manip = [&wmanip, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				(*wmanip)(in, weights_cu);
			};
		}
		else if (fmanip.has_value()) {
			freq_manip = [&fmanip, &kdata_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*fmanip)(in, kdata_coil);
			};
		}
		else if (wfmanip.has_value()) {
			freq_manip = [&wfmanip, &kdata_cu, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*wfmanip)(in, kdata_coil, weights_cu);
			};
		}

		normtensor.add_(
			sense.apply(in_cu, out_cu, dctxt.smaps, coils, storage_cu, freq_manip, sum, sumnorm)
		);
		
		// no need to mutex synchronize in forward since devices split over vector not inside tensors
		out_inner_batch_cpu_view.copy_(out_cu.cpu());
	}

	return normtensor.cpu();
}


at::Tensor hasty::BatchedSense::apply_adjoint(const TensorVec& in, at::Tensor& out, 
	bool sum, bool sumnorm, const std::optional<std::vector<std::vector<int64_t>>>& coils,
	const std::optional<WeightedManipulator>& wmanip, 
	const std::optional<FreqManipulator>& fmanip, 
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	int n_outer_batch = out.size(0);

	if (out.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image output to apply should be (N+2)D tensor");
	}

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<std::vector<int64_t>> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<std::vector<int64_t>>(n_outer_batch);
		for (int i = 0; i < coilss.size(); ++i) {
			std::iota(coilss[i].begin(), coilss[i].end(), 0);
		}
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	std::mutex summut;
	at::Tensor normtensor = at::scalar_tensor(at::Scalar((double)0.0));

	auto outer_capture = [this, &summut, &normtensor, &in, &out, sum, sumnorm, &coilss, &wmanip, &fmanip, &wfmanip](DeviceContext& context, int32_t outer_batch)
	{
		at::Tensor outer_norm = apply_outer_batch_adjoint(context, outer_batch, in[outer_batch], out, sum, sumnorm, coilss[outer_batch], wmanip, fmanip, wfmanip);
		{
			std::lock_guard<std::mutex> lock(summut);
			normtensor.add_(outer_norm);
		}
	};

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [outer_batch, &outer_capture](DeviceContext& context) { outer_capture(context, outer_batch); };

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

	return normtensor;
}

at::Tensor hasty::BatchedSense::apply_outer_batch_adjoint(DeviceContext& dctxt, int32_t outer_batch, const at::Tensor& in, at::Tensor& out, 
	bool sum, bool sumnorm, const std::vector<int64_t>& coils,
	const std::optional<WeightedManipulator>& wmanip, 
	const std::optional<FreqManipulator>& fmanip,
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor coord_cu;
	if (_coords.size() > 0) {
		coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	}

	if (in.size(1) != out.size(0)) {
		throw std::runtime_error("apply_outer_batch_adjoint: in.size(1) != out.size(0)");
	}

	Sense sense(coord_cu, _nmodes, true);
	int n_inner_batches = out.size(1);

	at::Tensor in_inner_batch_cpu_view; // = in.select(0, 0).unsqueeze(0);
	at::Tensor in_cu; // = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

	at::Tensor out_outer_batch_cpu_view = out.select(0, outer_batch);
	at::Tensor out_inner_batch_cpu_view = out_outer_batch_cpu_view.select(0, 0).unsqueeze(0);
	at::Tensor out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);
	at::Tensor storage_cu = at::empty_like(out_cu);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	at::Tensor normtensor = at::scalar_tensor(at::Scalar((double)0.0)).to(dctxt.stream.device(), true);

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		out_inner_batch_cpu_view = out_outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		out_cu = out_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		in_inner_batch_cpu_view = out.select(0, inner_batch);
		in_cu = in_inner_batch_cpu_view.to(dctxt.stream.device(), true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip;

		if (wmanip.has_value()) {
			freq_manip = [&wmanip, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				(*wmanip)(in, weights_cu);
			};
		}
		else if (fmanip.has_value()) {
			freq_manip = [&fmanip, &kdata_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*fmanip)(in, kdata_coil);
			};
		}
		else if (wfmanip.has_value()) {
			freq_manip = [&wfmanip, &kdata_cu, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*wfmanip)(in, kdata_coil, weights_cu);
			};
		}

		normtensor.add_(
			sense.apply(in_cu, out_cu, dctxt.smaps, coils, storage_cu, freq_manip, sum, sumnorm)
		);

		
		out_inner_batch_cpu_view.copy_(out_cu.cpu());

		if (_nctxt > 1)
		{
			at::Tensor out_cpu = out_cu.cpu();
			{
				std::lock_guard<std::mutex> lock(_copy_back_mutex);
				out_inner_batch_cpu_view.copy_(out_cpu);
			}
		}
		else {
			out_inner_batch_cpu_view.copy_(out_cu.cpu());
		}

	}

	return normtensor.cpu();
}


void BatchedSense::apply_normal(at::Tensor& in,
	const std::optional<std::vector<std::vector<int64_t>>>& coils,
	const std::optional<WeightedManipulator>& wmanip,
	const std::optional<FreqManipulator>& fmanip,
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	int n_outer_batch = in.size(0);

	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<std::vector<int64_t>> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<std::vector<int64_t>>(n_outer_batch);
		for (int i = 0; i < coilss.size(); ++i) {
			std::iota(coilss[i].begin(), coilss[i].end(), 0);
		}
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	auto outer_capture = [this, &in, &coilss, &wmanip, &fmanip, &wfmanip](DeviceContext& context, int32_t outer_batch)
	{
		apply_outer_batch_normal(context, outer_batch, in, coilss[outer_batch], wmanip, fmanip, wfmanip);
	};

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [outer_batch, &outer_capture](DeviceContext& context) { outer_capture(context, outer_batch); };

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

void BatchedSense::apply_outer_batch_normal(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
	const std::vector<int64_t>& coils,
	const std::optional<WeightedManipulator>& wmanip,
	const std::optional<FreqManipulator>& fmanip,
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	//std::cout << std::endl << "outer_batch: " << outer_batch << "coils: ";

	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor coord_cu;
	if (_coords.size() > 0) {
		coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	}
	SenseNormal sense(coord_cu, _nmodes);
	int n_inner_batches = in.size(1);

	at::Tensor outer_batch_cpu_view = in.select(0, outer_batch);
	at::Tensor inner_batch_cpu_view = outer_batch_cpu_view.select(0, 0).unsqueeze(0);
	at::Tensor in_cu = inner_batch_cpu_view.to(dctxt.stream.device(), true);
	at::Tensor out_cu = at::empty_like(in_cu);
	at::Tensor storage_cu = at::empty_like(in_cu);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {
		
		inner_batch_cpu_view = outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		in_cu = inner_batch_cpu_view.to(dctxt.stream.device(), true);
		//inner_batch_cpu_view.zero_();

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip;

		if (wmanip.has_value()) {
			freq_manip = [&wmanip, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				(*wmanip)(in, weights_cu);
			};
		}
		else if (fmanip.has_value()) {
			freq_manip = [&fmanip, &kdata_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*fmanip)(in, kdata_coil);
			};
		}
		else if (wfmanip.has_value()) {
			freq_manip = [&wfmanip, &kdata_cu, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*wfmanip)(in, kdata_coil, weights_cu);
			};
		}

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, storage_cu, std::nullopt, freq_manip);

		//out_cu.div_(std::accumulate(_nmodes.begin(), _nmodes.end(), 1, std::multiplies<int64_t>()));

		if (_nctxt > 1)
		{
			at::Tensor out_cpu = out_cu.cpu();
			{
				std::lock_guard<std::mutex> lock(_copy_back_mutex);
				inner_batch_cpu_view.copy_(out_cpu);
			}
		}
		else {
			inner_batch_cpu_view.copy_(out_cu.cpu());
		}
	}

}


void hasty::BatchedSense::apply_normal_adjoint(TensorVec& in, 
	const std::optional<std::vector<std::vector<int64_t>>>& coils,
	const std::optional<WeightedManipulator>& wmanip,
	const std::optional<FreqManipulator>& fmanip,
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	int n_outer_batch = in.size();

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<std::vector<int64_t>> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<std::vector<int64_t>>(n_outer_batch);
		for (int i = 0; i < coilss.size(); ++i) {
			std::iota(coilss[i].begin(), coilss[i].end(), 0);
		}
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	auto outer_capture = [this, &in, &coilss, &wmanip, &fmanip, &wfmanip](DeviceContext& context, int32_t outer_batch)
	{
		apply_outer_batch_normal_adjoint(context, outer_batch, in[outer_batch], coilss[outer_batch], wmanip, fmanip, wfmanip);
	};

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [outer_batch, &outer_capture](DeviceContext& context) { outer_capture(context, outer_batch); };

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

void hasty::BatchedSense::apply_outer_batch_normal_adjoint(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in, 
	const std::vector<int64_t>& coils, 
	const std::optional<WeightedManipulator>& wmanip, 
	const std::optional<FreqManipulator>& fmanip, 
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor coord_cu;
	if (_coords.size() > 0) {
		coord_cu = _coords[outer_batch].to(dctxt.stream.device(), true);
	}
	SenseNormal sense(coord_cu, _nmodes, true);
	int n_inner_batches = in.size(0);

	at::Tensor inner_batch_cpu_view = in.select(0, 0).unsqueeze(0);
	at::Tensor in_cu = inner_batch_cpu_view.to(dctxt.stream.device(), true);
	at::Tensor out_cu = at::empty_like(in_cu);
	at::Tensor storage_cu = at::empty(at::makeArrayRef(_nmodes), out_cu.options());

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.stream.device(), true);
	}

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		inner_batch_cpu_view = in.select(0, inner_batch).unsqueeze(0);
		in_cu = inner_batch_cpu_view.to(dctxt.stream.device(), true);
		//inner_batch_cpu_view.zero_();

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.stream.device(), true);
		}

		std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip;

		if (wmanip.has_value()) {
			freq_manip = [&wmanip, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				(*wmanip)(in, weights_cu);
			};
		}
		else if (fmanip.has_value()) {
			freq_manip = [&fmanip, &kdata_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*fmanip)(in, kdata_coil);
			};
		}
		else if (wfmanip.has_value()) {
			freq_manip = [&wfmanip, &kdata_cu, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				(*wfmanip)(in, kdata_coil, weights_cu);
			};
		}
		
		sense.apply(in_cu, out_cu, dctxt.smaps, coils, storage_cu, std::nullopt, freq_manip);

		inner_batch_cpu_view.copy_(out_cu.cpu());
	}
}




void BatchedSense::apply_toep(at::Tensor& in, const std::optional<std::vector<std::vector<int64_t>>>& coils)
{
	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	int n_outer_batch = in.size(0);

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<std::vector<int64_t>> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<std::vector<int64_t>>(n_outer_batch);
		for (int i = 0; i < coilss.size(); ++i) {
			std::iota(coilss[i].begin(), coilss[i].end(), 0);
		}
	}

	std::deque<std::future<void>> futures;

	std::function<void(DeviceContext&)> batch_applier;

	auto outer_capture = [this, &in, &coilss](DeviceContext& context, int32_t outer_batch)
	{
		apply_outer_batch_toep(context, outer_batch, in, coilss[outer_batch]);
	};

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [outer_batch, &outer_capture](DeviceContext& context) { outer_capture(context, outer_batch); };

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

void BatchedSense::apply_outer_batch_toep(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
	const std::vector<int64_t>& coils)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	std::unique_ptr<SenseNormalToeplitz> psense;

	if (_coords.size()> 0) {
		psense = std::make_unique<SenseNormalToeplitz>(_coords[outer_batch].to(dctxt.stream.device(), true), _nmodes, 1e-5);

	}
	else {
		psense = std::make_unique<SenseNormalToeplitz>(
			std::move(
				_diagonals.select(0, outer_batch).unsqueeze(0).to(dctxt.stream.device(), true)
			), 
			_nmodes);
	}
	
	int n_inner_batches = in.size(1);

	at::Tensor outer_batch_cpu_view = in.select(0, outer_batch);
	at::Tensor inner_batch_cpu_view = outer_batch_cpu_view.select(0, 0).unsqueeze(0);
	at::Tensor in_cu = inner_batch_cpu_view.to(dctxt.stream.device(), true);
	at::Tensor out_cu = at::empty_like(in_cu);

	std::vector<int64_t> expanded_dims;
	expanded_dims.push_back(0);
	for (int i = 1; i < in_cu.sizes().size(); ++i) {
		expanded_dims.push_back(in_cu.sizes()[i]);
	}

	at::Tensor storage1 = at::empty(at::makeArrayRef(expanded_dims), in_cu.options());
	at::Tensor storage2 = at::empty(at::makeArrayRef(expanded_dims), in_cu.options());

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		inner_batch_cpu_view = outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		in_cu = inner_batch_cpu_view.to(dctxt.stream.device(), true);
		inner_batch_cpu_view.zero_();

		psense->apply(in_cu, out_cu, storage1, storage2, dctxt.smaps, coils);

		if (_nctxt > 1)
		{
			at::Tensor out_cpu = out_cu.cpu();
			{
				std::lock_guard<std::mutex> lock(_copy_back_mutex);
				inner_batch_cpu_view.add_(out_cpu);
			}
		}
		else {
			inner_batch_cpu_view.add_(out_cu.cpu());
		}

	}
}

 