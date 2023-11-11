module;

#include "../torch_util.hpp"

module fftop;

// NUFFT

hasty::op::NUFFT::NUFFT(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::optional<nufft::NufftOptions>& opts)
	: _coords(coords), _nmodes(nmodes)
{
	if (opts.has_value()) {
		_opts = *opts;
		_opts.type = nufft::NufftType::eType2;
	}
	else {
		_opts = nufft::NufftOptions::type2();
	}

	if (_coords.is_cuda())
		_cudanufft = std::make_unique<nufft::CUDANufft>(_coords, _nmodes, _opts);
	else
		_cpunufft = std::make_unique<nufft::Nufft>(_coords, _nmodes, _opts);
}

hasty::op::Vector hasty::op::NUFFT::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_out(_coords, _nmodes[0]);
		if (_cudanufft != nullptr)
			_cudanufft->apply(access_vectensor(in), out);
		else
			_cpunufft->apply(access_vectensor(in), out);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply(child)));
	}

	return Vector(newchilds);
}

std::shared_ptr<hasty::op::Operator> hasty::op::NUFFT::to_device(at::Stream stream) const
{
	throw std::runtime_error("No to_device() for NUFFT operators");
}

// NUFFT ADJOINT

hasty::op::NUFFTAdjoint::NUFFTAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::optional<nufft::NufftOptions>& opts)
	: _coords(coords), _nmodes(nmodes)
{
	if (opts.has_value()) {
		_opts = *opts;
		_opts.type = nufft::NufftType::eType1;
	}
	else {
		_opts = nufft::NufftOptions::type1();
	}

	if (_coords.is_cuda())
		_cudanufft = std::make_unique<nufft::CUDANufft>(_coords, _nmodes, _opts);
	else
		_cpunufft = std::make_unique<nufft::Nufft>(_coords, _nmodes, _opts);
}

hasty::op::Vector hasty::op::NUFFTAdjoint::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_adjoint_out(_coords, _nmodes);
		if (_cudanufft != nullptr)
			_cudanufft->apply(access_vectensor(in), out);
		else
			_cpunufft->apply(access_vectensor(in), out);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply(child)));
	}

	return Vector(newchilds);
}

std::shared_ptr<hasty::op::Operator> hasty::op::NUFFTAdjoint::to_device(at::Stream stream) const
{
	throw std::runtime_error("No to_device() for NUFFT operators");
}

// NUFFT NORMAL

hasty::op::NUFFTNormal::NUFFTNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& forward_opts, const at::optional<nufft::NufftOptions>& backward_opts,
	at::optional<std::function<void(at::Tensor&)>> func_between)
	: _coords(coords), _nmodes(nmodes)
{
	if (forward_opts.has_value()) {
		_forward_opts = *forward_opts;
		_forward_opts.type = nufft::NufftType::eType2;
	}
	else {
		_forward_opts = nufft::NufftOptions::type2();
	}

	if (backward_opts.has_value()) {
		_backward_opts = *backward_opts;
		_backward_opts.type = nufft::NufftType::eType1;
	}
	else {
		_backward_opts = nufft::NufftOptions::type1();
	}

	_func_between = func_between;

	_storage = std::make_unique<at::Tensor>(nufft::allocate_normal_storage(_coords, _nmodes[0]));

	if (_coords.is_cuda())
		_cudanufft = std::make_unique<nufft::CUDANufftNormal>(_coords, _nmodes, _forward_opts, _backward_opts);
	else
		_cpunufft = std::make_unique<nufft::NufftNormal>(_coords, _nmodes, _forward_opts, _backward_opts);

}

hasty::op::Vector hasty::op::NUFFTNormal::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_normal_out(_coords, _nmodes);
		if (_cudanufft != nullptr)
			_cudanufft->apply(access_vectensor(in), out, *_storage, _func_between);
		else
			_cpunufft->apply(access_vectensor(in), out, *_storage, _func_between);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply(child)));
	}

	return Vector(newchilds);
}

void hasty::op::NUFFTNormal::apply_inplace(Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		if (_cudanufft != nullptr)
			_cudanufft->apply_inplace(access_vectensor(in), *_storage, _func_between);
		else
			_cpunufft->apply_inplace(access_vectensor(in), *_storage, _func_between);
	}

	for (auto& child : children) {
		apply(child);
	}
}

bool hasty::op::NUFFTNormal::has_inplace_apply() const
{
	return true;
}

std::shared_ptr<hasty::op::Operator> hasty::op::NUFFTNormal::to_device(at::Stream stream) const
{
	throw std::runtime_error("No to_device() for NUFFT operators");
}

// NUFFT NORMAL ADJOINT

hasty::op::NUFFTNormalAdjoint::NUFFTNormalAdjoint(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::optional<nufft::NufftOptions>& forward_opts, const at::optional<nufft::NufftOptions>& backward_opts,
	at::optional<std::function<void(at::Tensor&)>> func_between)
	: _coords(coords), _nmodes(nmodes)
{
	if (forward_opts.has_value()) {
		_forward_opts = *forward_opts;
		_forward_opts.type = nufft::NufftType::eType1;
	}
	else {
		_forward_opts = nufft::NufftOptions::type1();
	}

	if (backward_opts.has_value()) {
		_backward_opts = *backward_opts;
		_backward_opts.type = nufft::NufftType::eType2;
	}
	else {
		_backward_opts = nufft::NufftOptions::type2();
	}

	_func_between = func_between;

	_storage = std::make_unique<at::Tensor>(nufft::allocate_normal_adjoint_storage(_coords, _nmodes));

	if (_coords.is_cuda())
		_cudanufft = std::make_unique<nufft::CUDANufftNormal>(_coords, _nmodes, _forward_opts, _backward_opts);
	else
		_cpunufft = std::make_unique<nufft::NufftNormal>(_coords, _nmodes, _forward_opts, _backward_opts);

}

hasty::op::Vector hasty::op::NUFFTNormalAdjoint::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_normal_adjoint_out(_coords, _nmodes[0]);
		if (_cudanufft != nullptr)
			_cudanufft->apply(access_vectensor(in), out, *_storage, _func_between);
		else
			_cpunufft->apply(access_vectensor(in), out, *_storage, _func_between);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply(child)));
	}

	return Vector(newchilds);
}

void hasty::op::NUFFTNormalAdjoint::apply_inplace(Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		if (_cudanufft != nullptr)
			_cudanufft->apply_inplace(access_vectensor(in), *_storage, _func_between);
		else
			_cpunufft->apply_inplace(access_vectensor(in), *_storage, _func_between);
	}

	for (auto& child : children) {
		apply(child);
	}
}

bool hasty::op::NUFFTNormalAdjoint::has_inplace_apply() const
{
	return true;
}

std::shared_ptr<hasty::op::Operator> hasty::op::NUFFTNormalAdjoint::to_device(at::Stream stream) const
{
	throw std::runtime_error("No to_device() for NUFFT operators");
}