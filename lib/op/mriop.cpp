module;

#include <torch/torch.h>

module mriop;

std::unique_ptr<hasty::op::SenseOp> hasty::op::SenseOp::Create(const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
	const at::Tensor& smaps, const std::vector<int64_t>& coils, const at::optional<fft::NufftOptions>& opts)
{
	struct creator : public SenseOp {
		creator(const at::Tensor& a, const std::vector<int64_t>& b, const at::Tensor& c, 
			const std::vector<int64_t>& d, const at::optional<fft::NufftOptions>& e)
			: SenseOp(a, b, c, d, e) {}
	};
	return std::make_unique<creator>(coords, nmodes, smaps, coils, opts);
}

hasty::op::SenseOp::SenseOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const at::optional<fft::NufftOptions>& opts)
	: _coords(coords), _nmodes(nmodes), _smaps(smaps), _coils(coils)
{
	if (opts.has_value()) {
		_opts = *opts;
		_opts.type = fft::NufftType::eType2;
	}
	else {
		_opts = fft::NufftOptions::type2();
	}

	if (_coords.is_cuda())
		_cudasense = std::make_unique<mri::CUDASense>(_coords, _nmodes, _opts);
	else
		_cpusense = std::make_unique<mri::Sense>(_coords, _nmodes, _opts);
}

hasty::op::Vector hasty::op::SenseOp::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = fft::allocate_out(_coords, _nmodes[0]);
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
	newops.type = fft::NufftType::eType2;

	return SenseHOp::Create(_coords, _nmodes, _smaps, _coils, true, newops);
}

std::shared_ptr<hasty::op::Operator> hasty::op::SenseOp::to_device(at::Stream stream) const
{
	throw std::runtime_error("Sense operations may only be constructed on one device and never moved");
}

// SENSE ADJOINT

std::unique_ptr<hasty::op::SenseHOp> hasty::op::SenseHOp::Create(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate,
	const at::optional<fft::NufftOptions>& opts)
{
	struct creator : public SenseHOp {
		creator(const at::Tensor& a, const std::vector<int64_t>& b,
			const at::Tensor& c, const std::vector<int64_t>& d, bool e,
			const at::optional<fft::NufftOptions>& f)
			: SenseHOp(a, b, c, d, e, f) {}
	};
	return std::make_unique<creator>(coords, nmodes, smaps, coils, accumulate, opts);
}

hasty::op::SenseHOp::SenseHOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::Tensor& smaps, const std::vector<int64_t>& coils, bool accumulate, 
	const at::optional<fft::NufftOptions>& opts)
	: _coords(coords), _nmodes(nmodes), _smaps(smaps), _coils(coils), _accumulate(accumulate)
{
	if (opts.has_value()) {
		_opts = *opts;
		_opts.type = fft::NufftType::eType1;
	}
	else {
		_opts = fft::NufftOptions::type1();
	}

	if (_coords.is_cuda())
		_cudasense = std::make_unique<mri::CUDASenseAdjoint>(_coords, _nmodes, _opts);
	else
		_cpusense = std::make_unique<mri::SenseAdjoint>(_coords, _nmodes, _opts);
}

hasty::op::Vector hasty::op::SenseHOp::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor out = fft::allocate_adjoint_out(_coords, _nmodes);
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
	newops.type = fft::NufftType::eType2;
	return SenseOp::Create(_coords, _nmodes, _smaps, _coils, newops);
}

std::shared_ptr<hasty::op::Operator> hasty::op::SenseHOp::to_device(at::Stream stream) const
{
	throw std::runtime_error("Sense operations may only be constructed on one device and never moved");
}

// SENSE NORMAL OP

std::unique_ptr<hasty::op::SenseNOp> hasty::op::SenseNOp::Create(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::Tensor& smaps,
	const std::vector<int64_t>& coils, const at::optional<fft::NufftOptions>& forward_opts, const at::optional<fft::NufftOptions>& backward_opts)
{
	struct creator : public SenseNOp {
		creator(const at::Tensor& a, const std::vector<int64_t>& b, const at::Tensor& c,
			const std::vector<int64_t>& d, const at::optional<fft::NufftOptions>& e, const at::optional<fft::NufftOptions>& f)
			: SenseNOp(a, b, c, d, e, f) {}
	};
	return std::make_unique<creator>(coords, nmodes, smaps, coils, forward_opts, backward_opts);
}

hasty::op::SenseNOp::SenseNOp(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const at::optional<fft::NufftOptions>& forward_opts,
	const at::optional<fft::NufftOptions>& backward_opts)
	: _senseholder(std::make_shared<SenseNHolder>(coords, nmodes, smaps, coils, forward_opts, backward_opts))
{
}

hasty::op::SenseNOp::SenseNHolder::SenseNHolder(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::Tensor& smaps,
	const std::vector<int64_t>& coils, const at::optional<fft::NufftOptions>& forward_opts, const at::optional<fft::NufftOptions>& backward_opts)
	: _smaps(smaps), _coils(coils)
{
	if (forward_opts.has_value()) {
		_forward_opts = *forward_opts;
		_forward_opts.type = fft::NufftType::eType2;
	}
	else {
		_forward_opts = fft::NufftOptions::type2();
	}
	if (backward_opts.has_value()) {
		_backward_opts = *backward_opts;
		_backward_opts.type = fft::NufftType::eType2;
	}
	else {
		_backward_opts = fft::NufftOptions::type2();
	}

	if (_coords.is_cuda())
		_cudasense = std::make_unique<mri::CUDASenseNormal>(_coords, _nmodes, _forward_opts, _backward_opts);
	else
		_cpusense = std::make_unique<mri::SenseNormal>(_coords, _nmodes, _forward_opts, _backward_opts);
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
		at::Tensor out = fft::allocate_out(_senseholder->_coords, _senseholder->_nmodes[0]);
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
		at::Tensor out = fft::allocate_adjoint_out(_senseholder->_coords, _senseholder->_nmodes);
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
	struct creator : public SenseNOp { 
		creator(std::shared_ptr<SenseNHolder> _sh)
			: SenseNOp(std::move(_sh)) {}
	};
	return std::make_shared<creator>(_senseholder);
}

std::shared_ptr<hasty::op::Operator> hasty::op::SenseNOp::to_device(at::Stream stream) const
{
	throw std::runtime_error("Sense operations may only be constructed on one device and never moved");
}