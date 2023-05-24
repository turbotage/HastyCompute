#include "sense.hpp"
#include <numeric>

#include <c10/cuda/CUDAGuard.h>

using namespace hasty;

SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _normal_nufft(coords, nmodes, NufftOptions::type2(), NufftOptions::type1())
{
	
}

void SenseNormal::apply(const at::Tensor& in, at::Tensor& out, const std::vector<std::reference_wrapper<const at::Tensor>>& smaps,
	std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
	std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip)
{
	c10::InferenceMode inference_guard;

	out.zero_();
	
	at::Tensor xstore;
	at::Tensor fstore;

	if (in_storage.has_value()) {
		xstore = in_storage.value();
	} else {
		xstore = at::empty_like(in);
	}
	if (freq_storage.has_value()) {
		fstore = freq_storage.value();
	} else {
		fstore = at::empty({ 1, _normal_nufft.nfreq() }, in.options());
	}

	int smaps_len = smaps.size();

	bool has_freq_manip = freq_manip.has_value();

	for (int i = 0; i < smaps_len; ++i) {
		const at::Tensor& smap = smaps[i];
		at::mul_out(xstore, in, smap);
		if (has_freq_manip) {
			_normal_nufft.apply_inplace(xstore, fstore, std::bind(freq_manip.value(), std::placeholders::_1, i));
		} else {
			_normal_nufft.apply_inplace(xstore, fstore, std::nullopt);
		}
		out.addcmul_(xstore, smap.conj());
	}

}

void SenseNormal::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int32_t>& coils,
	std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
	std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip)
{
	c10::InferenceMode inference_guard;

	out.zero_();

	at::Tensor xstore;
	at::Tensor fstore;

	if (in_storage.has_value()) {
		xstore = in_storage.value();
	}
	else {
		xstore = at::empty_like(in);
	}
	if (freq_storage.has_value()) {
		fstore = freq_storage.value();
	}
	else {
		fstore = at::empty({ 1, _normal_nufft.nfreq() }, in.options());
	}

	bool has_freq_manip = freq_manip.has_value();

	for (auto coil : coils) {

		at::Tensor smap = smaps.select(0,coil).unsqueeze(0);
		at::mul_out(xstore, in, smap);

		if (has_freq_manip) {
			_normal_nufft.apply_inplace(xstore, fstore, std::bind(freq_manip.value(), std::placeholders::_1, coil));
		}
		else {
			_normal_nufft.apply_inplace(xstore, fstore, std::nullopt);
		}

		out.addcmul_(xstore, smap.conj());

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
	const at::Tensor& smaps, const std::vector<int32_t>& coils) const
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
	std::optional<TensorVec> coords,
	std::optional<TensorVec> kdata,
	std::optional<TensorVec> weights)
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

	_ndim = _coords[0].size(0);
	_nmodes.resize(smap.sizes().size());
	_nmodes[0] = 1; 
	for (int i = 1; i < _nmodes.size(); ++i) {
		_nmodes[i] = smap.size(i);
	}
}

void BatchedSense::apply(at::Tensor& in,
	const std::optional<std::vector<std::vector<int32_t>>>& coils,
	const std::optional<WeightedManipulator>& wmanip,
	const std::optional<FreqManipulator>& fmanip,
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	//std::cout << "BatchedSense::apply: " << std::endl;
	int n_outer_batch = in.size(0);

	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<std::vector<int32_t>> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<std::vector<int32_t>>(n_outer_batch);
		for (int i = 0; i < coilss.size(); ++i) {
			std::iota(coilss[i].begin(), coilss[i].end(), 0);
		}
	}

	std::deque<std::future<void>> futures;

	auto future_catcher = [](std::future<void>& fut) {
		try {
			fut.get();
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (...) {
			std::cerr << "caught something strange: " << std::endl;
		}
	};

	std::function<void(DeviceContext&)> batch_applier;

	auto outer_capture = [this, &in, &coilss, &wmanip, &fmanip, &wfmanip](DeviceContext& context, int32_t outer_batch)
	{
		apply_outer_batch(context, outer_batch, in, coilss[outer_batch], wmanip, fmanip, wfmanip);
	};

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [outer_batch, &outer_capture](DeviceContext& context) { outer_capture(context, outer_batch); };

		futures.emplace_back(tpool.enqueue(batch_applier));

		if (futures.size() > 32 * _dcontexts.size()) {
			future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		future_catcher(futures.front());
		futures.pop_front();
	}

}

void BatchedSense::apply_outer_batch(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
	const std::vector<int32_t>& coils,
	const std::optional<WeightedManipulator>& wmanip,
	const std::optional<FreqManipulator>& fmanip,
	const std::optional<WeightedFreqManipulator>& wfmanip)
{
	//std::cout << std::endl << "outer_batch: " << outer_batch << "coils: ";

	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor coord_cu;
	if (_coords.size() > 0) {
		coord_cu = _coords[outer_batch].to(dctxt.device, true);
	}
	SenseNormal sense(coord_cu, _nmodes);
	int n_inner_batches = in.size(1);

	at::Tensor outer_batch_cpu_view = in.select(0, outer_batch);
	at::Tensor inner_batch_cpu_view = outer_batch_cpu_view.select(0, 0).unsqueeze(0);
	at::Tensor in_cu = inner_batch_cpu_view.to(dctxt.device, true);
	at::Tensor out_cu = at::empty_like(in_cu);
	at::Tensor storage_cu = at::empty_like(in_cu);

	at::Tensor weights_cu;
	if (_weights.size() > 0) {
		weights_cu = _weights[outer_batch].to(dctxt.device, true);
	}

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {
		
		inner_batch_cpu_view = outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		in_cu = inner_batch_cpu_view.to(dctxt.device, true);

		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.device, true);
		}

		std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip;

		if (wmanip.has_value()) {
			freq_manip = [&wmanip, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				wmanip.value()(in, weights_cu);
			};
		}
		else if (fmanip.has_value()) {
			freq_manip = [&fmanip, &kdata_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				fmanip.value()(in, kdata_coil);
			};
		}
		else if (wfmanip.has_value()) {
			freq_manip = [&wfmanip, &kdata_cu, &weights_cu](at::Tensor& in, int32_t coil) {
				//std::cout << " " << coil;
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				wfmanip.value()(in, kdata_coil, weights_cu);
			};
		}

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, storage_cu, std::nullopt, freq_manip);

		out_cu.div_(std::accumulate(_nmodes.begin(), _nmodes.end(), 1, std::multiplies<int64_t>()));

		at::Tensor out_cpu = out_cu.cpu();
		{
			std::lock_guard<std::mutex> lock(_copy_back_mutex);
			inner_batch_cpu_view.add_(out_cpu);
		}
	}

}



void BatchedSense::apply_toep(at::Tensor& in, const std::optional<std::vector<std::vector<int32_t>>>& coils)
{
	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	int n_outer_batch = in.size(0);

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<std::vector<int32_t>> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<std::vector<int32_t>>(n_outer_batch);
		for (int i = 0; i < coilss.size(); ++i) {
			std::iota(coilss[i].begin(), coilss[i].end(), 0);
		}
	}

	std::deque<std::future<void>> futures;

	auto future_catcher = [](std::future<void>& fut) {
		try {
			fut.get();
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		catch (...) {
			std::cerr << "caught something strange: " << std::endl;
		}
	};

	std::function<void(DeviceContext&)> batch_applier;

	auto outer_capture = [this, &in, &coilss](DeviceContext& context, int32_t outer_batch)
	{
		apply_outer_batch_toep(context, outer_batch, in, coilss[outer_batch]);
	};

	for (int outer_batch = 0; outer_batch < n_outer_batch; ++outer_batch) {
		batch_applier = [outer_batch, &outer_capture](DeviceContext& context) { outer_capture(context, outer_batch); };

		futures.emplace_back(tpool.enqueue(batch_applier));

		if (futures.size() > 32 * _dcontexts.size()) {
			future_catcher(futures.front());
			futures.pop_front();
		}
	}

	// we wait for all promises
	while (futures.size() > 0) {
		future_catcher(futures.front());
		futures.pop_front();
	}
}

void BatchedSense::apply_outer_batch_toep(DeviceContext& dctxt, int32_t outer_batch, at::Tensor& in,
	const std::vector<int32_t>& coils)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	std::unique_ptr<SenseNormalToeplitz> psense;

	if (_coords.size()> 0) {
		psense = std::make_unique<SenseNormalToeplitz>(_coords[outer_batch].to(dctxt.device, true), _nmodes);

	}
	else {
		psense = std::make_unique<SenseNormalToeplitz>(_diagonals.select(0, outer_batch).to(dctxt.device, true), _nmodes);
	}
	
	int n_inner_batches = in.size(1);

	at::Tensor outer_batch_cpu_view = in.select(0, outer_batch);
	at::Tensor inner_batch_cpu_view = outer_batch_cpu_view.select(0, 0).unsqueeze(0);
	at::Tensor in_cu = inner_batch_cpu_view.to(dctxt.device, true);
	at::Tensor out_cu = at::empty_like(in_cu);

	at::Tensor storage1;
	at::Tensor storage2;

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {

		inner_batch_cpu_view = outer_batch_cpu_view.select(0, inner_batch).unsqueeze(0);
		in_cu = inner_batch_cpu_view.to(dctxt.device, true);

		psense->apply(in_cu, out_cu, storage1, storage2, dctxt.smaps, coils);

		at::Tensor out_cpu = out_cu.cpu();
		{
			std::lock_guard<std::mutex> lock(_copy_back_mutex);
			inner_batch_cpu_view.add_(out_cpu);
		}
	}
}

