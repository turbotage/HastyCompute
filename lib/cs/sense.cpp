#include "sense.hpp"
#include <numeric>

#include <c10/cuda/CUDAGuard.h>

using namespace hasty;

Sense::Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes, bool adjoint)
	: _nufft(coords, nmodes, adjoint ? NufftOptions::type1() : NufftOptions::type2()), _nmodes(nmodes), _adjoint(adjoint)
{
	if (_nmodes[0] != 1) {
		throw std::runtime_error("Only ntransf==1 allowed for Sense operator");
	}
}

at::Tensor hasty::Sense::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const std::optional<at::Tensor>& storage,
	const std::optional<std::function<void(at::Tensor&, int32_t)>>& manip, bool sum, bool sumnorm)
{
	c10::InferenceMode inference_guard;

	out.zero_();

	at::Tensor normtensor = at::scalar_tensor(at::Scalar((double)0.0)).to(in.device());
	bool has_manip = manip.has_value();

	if (!_adjoint) {
		
		at::Tensor instore;

		if (storage.has_value()) {
			instore = *storage;
		}
		else {
			instore = at::empty_like(in);
		}

		at::Tensor fstore;
		if (sum) {
			fstore = at::empty({ 1, _nufft.nfreq() }, in.options());
		}

		for (auto coil : coils) {
			at::Tensor smap = smaps.select(0, coil).unsqueeze(0);
			at::mul_out(instore, in, smap);

			if (!sum) {
				fstore = out.select(0, coil).unsqueeze(0);
			}

			_nufft.apply(instore, fstore);
			if (has_manip) {
				(*manip)(fstore, coil);
			}

			if (sumnorm) {
				auto norm = at::linalg_norm(fstore);
				normtensor.add_(norm.square_());
			}

			if (sum) {
				out.add_(fstore);
			}
		}
	}
	else {
		
		at::Tensor outstore;
		if (storage.has_value()) {
			outstore = *storage;
		}
		else {
			outstore = at::empty(at::makeArrayRef(_nmodes), in.options());
		}

		at::Tensor instore = at::empty({ 1, _nufft.nfreq() }, in.options());

		for (auto coil : coils) {
			
			instore.copy_(in.select(0, coil).unsqueeze(0));
			if (has_manip) {
				(*manip)(instore, coil);
			}

			_nufft.apply(instore, outstore);

			at::Tensor smap = smaps.select(0, coil).unsqueeze(0);
			outstore.mul_(smap.conj());

			if (sumnorm) {
				at::Tensor norm = at::linalg_norm(outstore);
				normtensor.add_(norm.square_());
			}

			if (sum) {
				out.add_(outstore);
			}
			else {
				out.select(0, coil).copy_(outstore);
			}
		}

	}

	return normtensor;
}


SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, bool adjoint)
	: _normal_nufft(coords, nmodes, adjoint ? NufftOptions::type1() : NufftOptions::type2(), adjoint ? NufftOptions::type2() : NufftOptions::type1()),
	_nmodes(nmodes), _adjoint(adjoint)
{
	
}

void SenseNormal::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const std::optional<at::Tensor>& in_storage, const std::optional<at::Tensor>& mid_storage,
	const std::optional<std::function<void(at::Tensor&, int32_t)>>& mid_manip)
{
	c10::InferenceMode inference_guard;

	out.zero_();

	if (!_adjoint) {
		at::Tensor xstore;
		at::Tensor midstore;

		if (in_storage.has_value()) {
			xstore = in_storage.value();
		}
		else {
			xstore = at::empty_like(in);
		}
		if (mid_storage.has_value()) {
			midstore = mid_storage.value();
		}
		else {
			midstore = at::empty({ 1, _normal_nufft.nfreq() }, in.options());
		}

		bool has_mid_manip = mid_manip.has_value();

		for (auto coil : coils) {

			at::Tensor smap = smaps.select(0,coil).unsqueeze(0);
			at::mul_out(xstore, in, smap);

			if (has_mid_manip) {
				_normal_nufft.apply_inplace(xstore, midstore, std::bind(mid_manip.value(), std::placeholders::_1, coil));
			}
			else {
				_normal_nufft.apply_inplace(xstore, midstore, std::nullopt);
			}

			out.addcmul_(xstore, smap.conj());

		}
	}
	else {

		at::Tensor midstore;
		if (mid_storage.has_value()) {
			midstore = *mid_storage;
		}
		else {
			midstore = at::empty(at::makeArrayRef(_nmodes), in.options());
		}

		bool has_mid_manip = mid_manip.has_value();
		std::function<void(at::Tensor&,int32_t)> total_mid_manip = [&smaps, &mid_manip](at::Tensor& in, int32_t coil) {

			at::Tensor smap = smaps.select(0, coil).unsqueeze(0);
			in.mul_(smap.conj());
			if (mid_manip.has_value()) {
				(*mid_manip)(in, coil);
			}
			in.mul_(smap);
		};

		for (auto coil : coils) {
			at::Tensor coil_out = out.select(0, coil).unsqueeze(0);
			_normal_nufft.apply(in.select(0, coil).unsqueeze(0), coil_out, midstore, std::bind(total_mid_manip, std::placeholders::_1, coil));
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
	at::Tensor&& diagonals)
	:
	_dcontexts(std::move(contexts)),
	_diagonals(std::move(diagonals))
{
	construct();
}

BatchedSense::BatchedSense(
	std::vector<DeviceContext>&& contexts,
	const std::optional<TensorVec>& coords,
	const std::optional<TensorVec>& kdata,
	const std::optional<TensorVec>& weights)
	:
	_dcontexts(std::move(contexts)),
	_coords(coords.has_value() ? std::move(*coords) : TensorVec()),
	_kdata(kdata.has_value() ? std::move(*kdata) : TensorVec()),
	_weights(weights.has_value() ? std::move(*weights) : TensorVec())
{
	construct();
}

void BatchedSense::construct()
{
	auto& smap = _dcontexts[0].smaps;

	_nctxt = _dcontexts.size();
	_ndim = smap.sizes().size() - 1;
	_nmodes.resize(smap.sizes().size());
	_nmodes[0] = 1; 
	for (int i = 1; i < _nmodes.size(); ++i) {
		_nmodes[i] = smap.size(i);
	}
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

 