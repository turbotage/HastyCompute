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



BatchedSense::BatchedSense(
	std::vector<DeviceContext>&& contexts,
	TensorVec&& coords)
	:
	_dcontexts(std::move(contexts)),
	_coords(std::move(coords))
{
	construct();
}

BatchedSense::BatchedSense(
	std::vector<DeviceContext>&& contexts,
	TensorVec&& coords,
	TensorVec&& kdata)
	:
	_dcontexts(std::move(contexts)),
	_coords(std::move(coords)),
	_kdata(std::move(kdata))
{
	construct();
}

BatchedSense::BatchedSense(
	std::vector<DeviceContext>&& contexts,
	TensorVec&& coords,
	TensorVec&& kdata,
	TensorVec&& weights)
	: 
	_dcontexts(std::move(contexts)),
	_coords(std::move(coords)),
	_kdata(std::move(kdata)),
	_weights(std::move(weights))
{
	construct();
}

void BatchedSense::construct()
{
	auto& smap = _dcontexts[0].smaps;

	_ndim = _coords[0].size(0);
	_nmodes[0] = 1; _nmodes[1] = smap.size(1); _nmodes[2] = smap.size(2); _nmodes[3] = smap.size(3);
}

void BatchedSense::apply(at::Tensor& in,
	const std::optional<std::vector<int32_t>>& coils,
	const std::optional<WeightedFreqManipulator>& wmanip,
	const std::optional<FreqManipulator>& manip)
{
	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<int32_t> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<int32_t>(num_smaps);
		std::iota(coilss.begin(), coilss.end(), 0);
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

	auto outer_capture = [this, &in, &coilss, &wmanip, &manip](DeviceContext& context, int32_t outer_batch)
	{
		apply_outer_batch(context, outer_batch, in, coilss, wmanip, manip);
	};

	int n_outer_batch = in.size(0);

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
	const std::optional<WeightedFreqManipulator>& wmanip,
	const std::optional<FreqManipulator>& manip)
{
	c10::InferenceMode inference_guard;
	c10::cuda::CUDAStreamGuard guard(dctxt.stream);

	at::Tensor coord_cu;
	if (_coords.size() > 0) {
		coord_cu = _coords[outer_batch].to(dctxt.device, true);
	}
	SenseNormal sense(coord_cu, _nmodes);
	int n_inner_batches = in.size(1);

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {
		at::Tensor cpu_batch_view = in.select(0, outer_batch).select(0, inner_batch).unsqueeze(0);

		at::Tensor in_cu = cpu_batch_view.to(dctxt.device, true);
		at::Tensor out_cu = at::empty_like(in_cu);
		at::Tensor storage_cu = at::empty_like(in_cu);
		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.device, true);
		}

		at::Tensor weights_cu;
		if (_weights.has_value()) {
			weights_cu = _weights.value()[outer_batch].to(dctxt.device, true);
		}

		std::optional<std::function<void(at::Tensor&, int32_t)>> freq_manip;

		if (wmanip.has_value()) {
			freq_manip = [&wmanip, &kdata_cu, &weights_cu](at::Tensor& in, int32_t coil) {
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				auto weights_coil = weights_cu.select(0, coil).unsqueeze(0);
				wmanip.value()(in, kdata_coil, weights_coil);
			};
		}
		else if (manip.has_value()) {
			freq_manip = [&manip, &kdata_cu](at::Tensor& in, int32_t coil) {
				auto kdata_coil = kdata_cu.select(0, coil).unsqueeze(0);
				manip.value()(in, kdata_coil);
			};
		}

		sense.apply(in_cu, out_cu, dctxt.smaps, coils, storage_cu, std::nullopt, freq_manip);

		at::Tensor out_cpu = out_cu.cpu();
		{
			std::lock_guard<std::mutex> lock(_copy_back_mutex);
			cpu_batch_view.add_(out_cpu);
		}
	}


}





void BatchedSense::apply_toep(at::Tensor& in, const std::optional<std::vector<int32_t>>& coils)
{
	if (in.sizes().size() != _ndim + 2) {
		throw std::runtime_error("For a ND-image input to apply should be (N+2)D tensor");
	}

	c10::InferenceMode im_mode;

	ContextThreadPool<DeviceContext> tpool(_dcontexts);

	std::vector<int32_t> coilss;
	if (coils.has_value()) {
		coilss = coils.value();
	}
	else {
		int num_smaps = _dcontexts[0].smaps.size(0);
		coilss = std::vector<int32_t>(num_smaps);
		std::iota(coilss.begin(), coilss.end(), 0);
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
		apply_outer_batch_toep(context, outer_batch, in, coilss);
	};

	int n_outer_batch = in.size(0);

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

	at::Tensor coord_cu;
	if (_coords.size() > 0) {
		coord_cu = _coords[outer_batch].to(dctxt.device, true);
	}


	//SenseNormal sense(coord_cu, _nmodes);
	int n_inner_batches = in.size(1);

	for (int inner_batch = 0; inner_batch < n_inner_batches; ++inner_batch) {
		at::Tensor cpu_batch_view = in.select(0, outer_batch).select(0, inner_batch).unsqueeze(0);

		at::Tensor in_cu = cpu_batch_view.to(dctxt.device, true);
		at::Tensor out_cu = at::empty_like(in_cu);
		at::Tensor storage_cu = at::empty_like(in_cu);
		at::Tensor kdata_cu;
		if (_kdata.size() > 0) {
			kdata_cu = _kdata[outer_batch].select(0, inner_batch).to(dctxt.device, true);
		}

		// Apply Toeplitz Sense

		at::Tensor out_cpu = out_cu.cpu();
		{
			std::lock_guard<std::mutex> lock(_copy_back_mutex);
			cpu_batch_view.add_(out_cpu);
		}
	}
}

