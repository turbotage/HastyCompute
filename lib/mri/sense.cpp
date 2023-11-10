module;

#include "../torch_util.hpp"
#include <c10/cuda/CUDAGuard.h>

module sense;

hasty::sense::Sense::Sense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& opts)
	: _nufft(coords, nmodes, opts.has_value() ? *opts : nufft::NufftOptions::type2()), _nmodes(nmodes)
{	
}

void hasty::sense::Sense::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
	const at::optional<CoilApplier>& premanip,
	const at::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), in.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ntransf, _nufft.nfreq() }, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		if (premanip.has_value()) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx);
			}
			if (premanip.has_value()) {
				(*premanip)(imstore, brun);
			}
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j).mul_(smaps.select(0, idx));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx) * smaps.select(0, idx);
			}
		}

		_nufft.apply(imstore, kstore);

		if (postmanip.has_value()) {
			(*postmanip)(kstore, brun);
		}

		if (accumulate) {
			for (int j = 0; j < ntransf; ++j) {
				out.select(0, 0).unsqueeze(0).add_(kstore.select(0, j));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				out.select(0, j).unsqueeze(0).add_(kstore.select(0, j));
			}
		}
	}

}


hasty::sense::CUDASense::CUDASense(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& opts)
	: _nufft(coords, nmodes, opts.has_value() ? *opts : nufft::NufftOptions::type2()), _nmodes(nmodes)
{
}

void hasty::sense::CUDASense::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
	const at::optional<CoilApplier>& premanip,
	const at::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), in.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _nufft.nfreq()}, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		if (premanip.has_value()) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx);
			}
			if (premanip.has_value()) {
				(*premanip)(imstore, brun);
			}
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j).mul_(smaps.select(0, idx));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx) * smaps.select(0, idx);
			}
		}

		_nufft.apply(imstore, kstore);

		if (postmanip.has_value()) {
			(*postmanip)(kstore, brun);
		}

		if (accumulate) {
			for (int j = 0; j < ntransf; ++j) {
				out.add_(kstore.select(0, j));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				out.select(0, j).unsqueeze(0).add_(kstore.select(0, j));
			}
		}
	}

}


hasty::sense::SenseAdjoint::SenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& opts)
	: _nufft(coords, nmodes, opts.has_value() ? *opts : nufft::NufftOptions::type1()), _nmodes(nmodes)
{
}

void hasty::sense::SenseAdjoint::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
	const at::optional<CoilApplier>& premanip, const at::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), out.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _nufft.nfreq() }, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			kstore.select(0, j) = in.select(0, idx);
		}

		if (premanip.has_value()) {
			(*premanip)(kstore, brun);
		}

		_nufft.apply(kstore, imstore);


		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.select(0, j).mul_(smaps.select(0, coils[idx]).conj());
		}

		if (postmanip.has_value()) {
			(*postmanip)(imstore, brun);
		}


		if (accumulate) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.add_(imstore.select(0, j));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.select(0, idx).add_(imstore.select(0, j));
			}
		}
	}
}


hasty::sense::CUDASenseAdjoint::CUDASenseAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& opts)
	: _nufft(coords, nmodes, opts.has_value() ? *opts : nufft::NufftOptions::type1()), _nmodes(nmodes)
{
}

void hasty::sense::CUDASenseAdjoint::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
	const at::optional<CoilApplier>& premanip, const at::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), out.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _nufft.nfreq() }, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			kstore.select(0, j) = in.select(0, idx);
		}

		if (premanip.has_value()) {
			(*premanip)(kstore, brun);
		}

		_nufft.apply(kstore, imstore);


		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.select(0,j).mul_(smaps.select(0, coils[idx]).conj());
		}

		if (postmanip.has_value()) {
			(*postmanip)(imstore, brun);
		}


		if (accumulate) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.add_(imstore.select(0, j));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.select(0, idx).add_(imstore.select(0, j));
			}
		}
	}
}


hasty::sense::SenseNormal::SenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& forward_opts, const at::optional<nufft::NufftOptions>& backward_opts)
	: _normal_nufft(coords, nmodes, 
		forward_opts.has_value() ? *forward_opts : nufft::NufftOptions::type2(),
		backward_opts.has_value() ? *backward_opts : nufft::NufftOptions::type1()),
		_nmodes(nmodes)
{
}

void hasty::sense::SenseNormal::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
	const at::optional<CoilApplier>& premanip,
	const at::optional<CoilApplier>& midmanip,
	const at::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();
	
	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), in.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _normal_nufft.nfreq() }, in.options());
	}

	for (int brun = 0; brun < batch_runs; ++brun) {

		if (premanip.has_value()) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx);
			}
			if (premanip.has_value()) {
				(*premanip)(imstore, brun);
			}
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j).mul_(smaps.select(0, idx));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx) * smaps.select(0, idx);
			}
		}


		if (midmanip.has_value()) {
			_normal_nufft.apply_inplace(imstore, kstore, std::bind(*midmanip, std::placeholders::_1, brun));
		}
		else {
			_normal_nufft.apply_inplace(imstore, kstore, at::nullopt);
		}

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.mul_(smaps.select(0, coils[idx]).conj());
		}

		if (postmanip.has_value()) {
			(*postmanip)(imstore, brun);
		}


		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			out.add_(imstore.select(0, j));
		}
	}
}

void hasty::sense::SenseNormal::apply_forward(const at::Tensor& in, at::Tensor& out, 
	const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage,
	const at::optional<at::Tensor>& kspace_storage)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), in.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _normal_nufft.nfreq() }, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.select(0, j) = in.select(0, idx) * smaps.select(0, idx);
		}

		_normal_nufft.apply_forward(imstore, kstore);

		for (int j = 0; j < ntransf; ++j) {
			out.select(0, j).unsqueeze(0).add_(kstore.select(0, j));
		}
	}
}

void hasty::sense::SenseNormal::apply_backward(const at::Tensor& in, at::Tensor& out, 
	const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& kspace_storage,
	const at::optional<at::Tensor>& imspace_storage)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), out.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _normal_nufft.nfreq() }, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			kstore.select(0, j) = in.select(0, idx);
		}

		_normal_nufft.apply_backward(kstore, imstore);


		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.select(0, j).mul_(smaps.select(0, coils[idx]).conj());
		}


		if (accumulate) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.add_(imstore.select(0, j));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.select(0, idx).add_(imstore.select(0, j));
			}
		}
	}
}


hasty::sense::CUDASenseNormal::CUDASenseNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& forward_opts, const at::optional<nufft::NufftOptions>& backward_opts)
	: _normal_nufft(coords, nmodes, 
		forward_opts.has_value() ? *forward_opts : nufft::NufftOptions::type2(),
		backward_opts.has_value() ? *backward_opts : nufft::NufftOptions::type1()),
		_nmodes(nmodes)
{
}

void hasty::sense::CUDASenseNormal::apply(const at::Tensor& in, at::Tensor& out, const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage, const at::optional<at::Tensor>& kspace_storage,
	const at::optional<CoilApplier>& premanip,
	const at::optional<CoilApplier>& midmanip,
	const at::optional<CoilApplier>& postmanip)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), in.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _normal_nufft.nfreq() }, in.options());
	}

	for (int brun = 0; brun < batch_runs; ++brun) {

		if (premanip.has_value()) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx);
			}
			if (premanip.has_value()) {
				(*premanip)(imstore, brun);
			}
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j).mul_(smaps.select(0, idx));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				imstore.select(0, j) = in.select(0, idx) * smaps.select(0, idx);
			}
		}


		if (midmanip.has_value()) {
			_normal_nufft.apply_inplace(imstore, kstore, std::bind(*midmanip, std::placeholders::_1, brun));
		}
		else {
			_normal_nufft.apply_inplace(imstore, kstore, at::nullopt);
		}

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.mul_(smaps.select(0, coils[idx]).conj());
		}

		if (postmanip.has_value()) {
			(*postmanip)(imstore, brun);
		}


		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			out.add_(imstore.select(0, j));
		}
	}
}

void hasty::sense::CUDASenseNormal::apply_forward(const at::Tensor& in, at::Tensor& out,
	const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& imspace_storage,
	const at::optional<at::Tensor>& kspace_storage)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), in.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _normal_nufft.nfreq() }, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.select(0, j) = in.select(0, idx) * smaps.select(0, idx);
		}

		_normal_nufft.apply_forward(imstore, kstore);

		for (int j = 0; j < ntransf; ++j) {
			out.select(0, j).unsqueeze(0).add_(kstore.select(0, j));
		}
	}
}

void hasty::sense::CUDASenseNormal::apply_backward(const at::Tensor& in, at::Tensor& out,
	const at::Tensor& smaps, const std::vector<int64_t>& coils,
	const at::optional<at::Tensor>& kspace_storage,
	const at::optional<at::Tensor>& imspace_storage)
{
	c10::InferenceMode inference_guard;

	if (coils.size() % _nmodes[0] != 0)
		throw std::runtime_error("Number of coils used in Sense::apply must be divisible by number of ntransf used for nufft");

	int ntransf = _nmodes[0];
	int batch_runs = coils.size() / ntransf;

	out.zero_();

	at::Tensor imstore;
	if (imspace_storage.has_value()) {
		imstore = *imspace_storage;
	}
	else {
		imstore = at::empty(at::makeArrayRef(_nmodes), out.options());
	}

	at::Tensor kstore;
	if (kspace_storage.has_value()) {
		kstore = *kspace_storage;
	}
	else {
		kstore = at::empty({ ntransf, _normal_nufft.nfreq() }, in.options());
	}

	bool accumulate = out.size(0) == 1;

	for (int brun = 0; brun < batch_runs; ++brun) {

		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			kstore.select(0, j) = in.select(0, idx);
		}

		_normal_nufft.apply_backward(kstore, imstore);


		for (int j = 0; j < ntransf; ++j) {
			int idx = brun * batch_runs + j;
			imstore.select(0, j).mul_(smaps.select(0, coils[idx]).conj());
		}


		if (accumulate) {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.add_(imstore.select(0, j));
			}
		}
		else {
			for (int j = 0; j < ntransf; ++j) {
				int idx = brun * batch_runs + j;
				out.select(0, idx).add_(imstore.select(0, j));
			}
		}
	}
}

