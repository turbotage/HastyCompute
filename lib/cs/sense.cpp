#include "sense.hpp"
#include <numeric>

#include <c10/cuda/CUDAGuard.h>


hasty::Sense::Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
	: _nufft(coords, nmodes, NufftOptions::type2()), _nmodes(nmodes)
{
	if (_nmodes[0] != 1) {
		throw std::runtime_error("Only ntransf==1 allowed for Sense operator");
	}
}

void hasty::Sense::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const std::optional<at::Tensor>& imspace_storage, const std::optional<at::Tensor>& kspace_storage,
	const std::optional<CoilApplier>& premanip,
	const std::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty_like(in);
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
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

void hasty::SenseAdjoint::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
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
		kstore = at::empty_like(in.select(0, 0).unsqueeze(0));
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


hasty::SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
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
	}
	else {
		imstore = at::empty_like(in);
	}

	out.zero_();


	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ 1, _normal_nufft.nfreq() }, in.options());
	}

	for (auto coil : coils) {
		at::Tensor smap = smaps.select(0, coil).unsqueeze(0);

		if (premanip.has_value()) {
			imstore.copy_(in);
			(*premanip)(imstore, coil);
			imstore.mul_(smap);
		}
		else {
			at::mul_out(imstore, in, smap);
		}


		if (midmanip.has_value()) {
			_normal_nufft.apply_inplace(imstore, kstore, std::bind(*midmanip, std::placeholders::_1, coil));
		}
		else {
			_normal_nufft.apply_inplace(imstore, kstore, std::nullopt);
		}

		if (postmanip.has_value()) {
			imstore.mul_(smap.conj());
			(*postmanip)(imstore, coil);
			out.add_(imstore);
		}
		else {
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
	}
	else {
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



hasty::SenseNormalToeplitz::SenseNormalToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, double tol)
	: _normal_nufft(coords, nmodes, tol, std::nullopt, std::nullopt, std::nullopt)
{

}

hasty::SenseNormalToeplitz::SenseNormalToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes)
	: _normal_nufft(std::move(diagonal), nmodes)
{

}

void hasty::SenseNormalToeplitz::apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage1, at::Tensor& storage2,
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




