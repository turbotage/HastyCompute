#include "sense_cu.hpp"

using namespace hasty::cuda;

SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _normal_nufft(coords, nmodes, NufftOptions::type2(), NufftOptions::type1())
{

}

void SenseNormal::apply(const at::Tensor& in, const std::vector<std::reference_wrapper<const at::Tensor>>& smaps, const at::Tensor& out,
	std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
	std::optional<std::function<void(at::Tensor&,int)>> freq_manip)
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

void SenseNormal::apply(const at::Tensor& in, const at::Tensor& smaps, const std::vector<int32_t>& coils, const at::Tensor& out,
	std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
	std::optional<std::function<void(at::Tensor&,int)>> freq_manip)
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

		try {
			torch::cuda::synchronize(in.device().index());
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}

		std::cout << " coil: " << coil << std::endl;
		at::Tensor smap = smaps.select(0,coil).unsqueeze(0);
		at::mul_out(xstore, in, smap);

		try {
			torch::cuda::synchronize(in.device().index());
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}

		if (has_freq_manip) {
			_normal_nufft.apply_inplace(xstore, fstore, std::bind(freq_manip.value(), std::placeholders::_1, coil));
		}
		else {
			_normal_nufft.apply_inplace(xstore, fstore, std::nullopt);
		}

		try {
			torch::cuda::synchronize(in.device().index());
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}

		out.addcmul_(xstore, smap.conj());

		try {
			torch::cuda::synchronize(in.device().index());
		}
		catch (c10::Error& e) {
			std::cerr << e.what() << std::endl;
		}

	}

}
