#include "mriop.hpp"


hasty::op::SenseOp::SenseOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const at::optional<nufft::NufftOptions>& opts)
	: _coords(coords), _nmodes(nmodes), _smaps(smaps), _coils(coils)
{
	if (opts.has_value()) {
		_opts = *opts;
		_opts.type = nufft::NufftType::eType2;
	}
	else {
		_opts = nufft::NufftOptions::type2();
	}

	if (_coords.is_cuda())
		_cudasense = std::make_unique<sense::CUDASense>(_coords, _nmodes, _opts);
	else
		_cpusense = std::make_unique<sense::Sense>(_coords, _nmodes, _opts);
}

hasty::op::Vector hasty::op::SenseOp::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_out(_coords, _nmodes[0]);
		if (_cudasense != nullptr)
			_cudasense->apply(access_vectensor(in), out, _smaps, _coils, 
				at::nullopt, at::nullopt, at::nullopt, at::nullopt);
		else
			_cudasense->apply(access_vectensor(in), out, _smaps, _coils,
				at::nullopt, at::nullopt, at::nullopt, at::nullopt);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply(child)));
	}

	return Vector(newchilds);
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::SenseOp::adjoint() const
{
	auto newops = _opts;
	newops.type = nufft::NufftType::eType2;
	return std::make_shared<SenseHOp>(_coords, _nmodes, _smaps, _coils, true, newops);
}

std::shared_ptr<hasty::op::Operator> hasty::op::SenseOp::to_device(at::Stream stream) const
{
	throw std::runtime_error("Sense operations may only be constructed on one device and never moved");
}

// SENSE ADJOINT

hasty::op::SenseHOp::SenseHOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate, 
	const at::optional<nufft::NufftOptions>& opts)
	: _coords(coords), _nmodes(nmodes), _smaps(smaps), _coils(coils), _accumulate(accumulate)
{
	if (opts.has_value()) {
		_opts = *opts;
		_opts.type = nufft::NufftType::eType1;
	}
	else {
		_opts = nufft::NufftOptions::type1();
	}

	if (_coords.is_cuda())
		_cudasense = std::make_unique<sense::CUDASenseAdjoint>(_coords, _nmodes, _opts);
	else
		_cpusense = std::make_unique<sense::SenseAdjoint>(_coords, _nmodes, _opts);
}

hasty::op::Vector hasty::op::SenseHOp::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_adjoint_out(_coords, _nmodes);
		if (_cudasense != nullptr)
			_cudasense->apply(access_vectensor(in), out, _smaps, _coils,
				at::nullopt, at::nullopt, at::nullopt, at::nullopt);
		else
			_cudasense->apply(access_vectensor(in), out, _smaps, _coils,
				at::nullopt, at::nullopt, at::nullopt, at::nullopt);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply(child)));
	}

	return Vector(newchilds);
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::SenseHOp::adjoint() const
{
	auto newops = _opts;
	newops.type = nufft::NufftType::eType2;
	return std::make_shared<SenseOp>(_coords, _nmodes, _smaps, _coils, newops);
}

std::shared_ptr<hasty::op::Operator> hasty::op::SenseHOp::to_device(at::Stream stream) const
{
	throw std::runtime_error("Sense operations may only be constructed on one device and never moved");
}

// SENSE NORMAL OP

hasty::op::SenseNOp::SenseNOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const at::optional<nufft::NufftOptions>& forward_opts,
	const at::optional<nufft::NufftOptions>& backward_opts)
	: _senseholder(std::make_shared<SenseNHolder>(coords, nmodes, smaps, coils, forward_opts, backward_opts))
{
}

hasty::op::SenseNOp::SenseNHolder::SenseNHolder(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::Tensor& smaps,
	const std::vector<int64_t>& coils, const at::optional<nufft::NufftOptions>& forward_opts, const at::optional<nufft::NufftOptions>& backward_opts)
	: _smaps(smaps), _coils(coils)
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
		_backward_opts.type = nufft::NufftType::eType2;
	}
	else {
		_backward_opts = nufft::NufftOptions::type2();
	}

	if (_coords.is_cuda())
		_cudasense = std::make_unique<sense::CUDASenseNormal>(_coords, _nmodes, _forward_opts, _backward_opts);
	else
		_cpusense = std::make_unique<sense::SenseNormal>(_coords, _nmodes, _forward_opts, _backward_opts);
}

hasty::op::SenseNOp::SenseNOp(std::shared_ptr<SenseNHolder> shoulder)
	: _senseholder(std::move(shoulder))
{
}

hasty::op::Vector hasty::op::SenseNOp::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor intens = access_vectensor(in);
		at::Tensor out = at::empty_like(intens);
		if (_senseholder->_cudasense != nullptr)
			_senseholder->_cudasense->apply(access_vectensor(in), out, _senseholder->_smaps, _senseholder->_coils,
				at::nullopt, at::nullopt, at::nullopt, at::nullopt, at::nullopt);
		else
			_senseholder->_cudasense->apply(access_vectensor(in), out, _senseholder->_smaps, _senseholder->_coils,
				at::nullopt, at::nullopt, at::nullopt, at::nullopt, at::nullopt);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply(child)));
	}

	return Vector(newchilds);
}

hasty::op::Vector hasty::op::SenseNOp::apply_forward(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_out(_senseholder->_coords, _senseholder->_nmodes[0]);
		if (_senseholder->_cudasense != nullptr)
			_senseholder->_cudasense->apply_forward(access_vectensor(in), out, _senseholder->_smaps, _senseholder->_coils,
				at::nullopt, at::nullopt);
		else
			_senseholder->_cudasense->apply_forward(access_vectensor(in), out, _senseholder->_smaps, _senseholder->_coils,
				at::nullopt, at::nullopt);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply_forward(child)));
	}

	return Vector(newchilds);
}

hasty::op::Vector hasty::op::SenseNOp::apply_backward(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = nufft::allocate_adjoint_out(_senseholder->_coords, _senseholder->_nmodes);
		if (_senseholder->_cudasense != nullptr)
			_senseholder->_cudasense->apply_backward(access_vectensor(in), out, _senseholder->_smaps, _senseholder->_coils,
				at::nullopt, at::nullopt);
		else
			_senseholder->_cudasense->apply_backward(access_vectensor(in), out, _senseholder->_smaps, _senseholder->_coils,
				at::nullopt, at::nullopt);
		return Vector(out);
	}

	std::vector<Vector> newchilds;
	newchilds.reserve(children.size());
	for (auto& child : children) {
		newchilds.emplace_back(std::move(apply_backward(child)));
	}

	return Vector(newchilds);
}

std::shared_ptr<hasty::op::AdjointableOp> hasty::op::SenseNOp::adjoint() const
{
	return downcast_shared_from_this<SenseNOp>();
}

std::shared_ptr<hasty::op::Operator> hasty::op::SenseNOp::to_device(at::Stream stream) const
{
	throw std::runtime_error("Sense operations may only be constructed on one device and never moved");
}