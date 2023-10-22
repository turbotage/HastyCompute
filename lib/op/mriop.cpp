#include "mriop.hpp"


hasty::op::SENSE::SENSE(const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
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

hasty::op::Vector hasty::op::SENSE::apply(const Vector& in) const
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



hasty::op::SENSE_H::SENSE_H(const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
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

hasty::op::Vector hasty::op::SENSE_H::apply(const Vector& in) const
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



hasty::op::SENSE_N::SENSE_N(const at::Tensor& coords, const std::vector<int64_t>& nmodes, 
	const at::Tensor& smaps, const std::vector<int64_t>& coils, 
	const at::optional<nufft::NufftOptions>& forward_opts,
	const at::optional<nufft::NufftOptions>& backward_opts)
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

hasty::op::Vector hasty::op::SENSE_N::apply(const Vector& in) const
{
	const auto& children = access_vecchilds(in);

	if (children.empty()) {
		at::Tensor intens = access_vectensor(in);
		at::Tensor out = at::empty_like(intens);
		if (_cudasense != nullptr)
			_cudasense->apply(access_vectensor(in), out, _smaps, _coils,
				at::nullopt, at::nullopt, at::nullopt, at::nullopt, at::nullopt);
		else
			_cudasense->apply(access_vectensor(in), out, _smaps, _coils,
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

