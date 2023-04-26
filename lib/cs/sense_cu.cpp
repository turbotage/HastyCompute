#include "sense_cu.hpp"
#include "../torch_util.hpp"

using namespace hasty::cuda;

SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _normal_nufft(coords, nmodes, NufftOptions::type2(), NufftOptions::type1())
{

}

void SenseNormal::apply(const at::Tensor& in, const std::vector<std::reference_wrapper<const at::Tensor>>& smaps, const at::Tensor& out,
	std::optional<at::Tensor> in_storage, std::optional<at::Tensor> freq_storage,
	std::optional<std::function<void(at::Tensor&)>> freq_manip)
{
	out.zero_();
	
	at::Tensor xstore;
	at::Tensor fstore;

	if (in_storage.has_value()) {
		xstore = in_storage.value();
	} else {
		xstore = at::empty_like(in);
	}
	if (freq_storage.has_value()) {
		fstore = at::empty({ 1, _normal_nufft.nfreq() }, in.options());
	} else {
		fstore = freq_storage.value();
	}

	int smaps_len = smaps.size();

	for (int i = 0; i < smaps_len; ++i) {
		const at::Tensor& smap = smaps[i];
		at::mul_out(xstore, in, smap);
		_normal_nufft.apply_inplace(xstore, fstore, freq_manip);
		out.addcmul_(xstore, smap.conj());
	}


}
