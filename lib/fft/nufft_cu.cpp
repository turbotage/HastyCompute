
//#include <cufinufft.h>

#include "nufft_cu.hpp"

#include <cufinufft.h>


#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <array>;
import <stdexcept>;
import <optional>;
#endif

import hasty_util;

using namespace hasty::cuda;
using namespace hasty;

Nufft::Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& opts)
	: _coords(coords), _nmodes(nmodes), _opts(opts)
{
	_type = _coords.dtype().toScalarType();
	_ntransf = _nmodes[0];
	_ndim = _coords.size(0);
	_nfreq = _coords.size(1);

	if (_opts.get_type() == NufftType::eType3) {
		throw std::runtime_error("Type 3 Nufft is not yet supported");
	}

	if (_ndim + 1 != _nmodes.size()) {
		throw std::runtime_error("coords.size(0) must match number of nmodes given");
	}

	switch (_nmodes.size()) {
	case 2:
	{
		_ndim = 1;
		_nmodes_flipped[0] = _nmodes[1];
	}
	break;
	case 3:
	{
		_ndim = 2;
		_nmodes_flipped[0] = _nmodes[2];
		_nmodes_flipped[1] = _nmodes[1];
	}
	break;
	case 4:
	{
		_ndim = 3;
		_nmodes_flipped[0] = _nmodes[3];
		_nmodes_flipped[1] = _nmodes[2];
		_nmodes_flipped[2] = _nmodes[1];
	}
	break;
	default:
		throw std::runtime_error("Only 1,2,3-D Nufft's are supported");
	}

	if (!_coords.is_contiguous())
		throw std::runtime_error("coords must be contiguous");

	make_plan_set_pts();

}

Nufft::~Nufft() {
	try {
		switch (_type) {
		case c10::ScalarType::Float:
		{
			if (cufinufftf_destroy(_planf)) {
				std::exit(EXIT_FAILURE);
			}
		}
		break;
		case c10::ScalarType::Double:
		{
			if (cufinufft_destroy(_plan)) {
				std::exit(EXIT_FAILURE);
			}
		}
		break;
		default:
			throw std::runtime_error("Coordinates must be real!");
		}
	}
	catch (...) {
		std::exit(EXIT_FAILURE);
	}
}

void Nufft::apply(const at::Tensor& in, at::Tensor& out) const
{
	switch (_opts.get_type()) {
	case eType1:
	{
		apply_type1(in, out);
	}
	break;
	case eType2:
	{
		apply_type2(in, out);
	}
	break;
	default:
		throw std::runtime_error("Only Nufft Type 1 and Type 2 is supported");
	}
}

void Nufft::make_plan_set_pts()
{

	using namespace at::indexing;

	switch (_type) {
	case c10::ScalarType::Float:
	{
		if (cufinufftf_makeplan((int32_t)_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, 0, &_planf, NULL))
			throw std::runtime_error("cufinufft_makeplan failed");

		switch (_ndim) {
		case 1:
		{
			auto tx = _coords.select(0, 0);
			if (cufinufftf_setpts(_nfreq, (float*)tx.data_ptr(), NULL, NULL, 0, NULL, NULL, NULL, _planf)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 2:
		{
			auto tx = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			if (cufinufftf_setpts(_nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL, _planf)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tx = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tz = _coords.select(0, 2);
			if (cufinufftf_setpts(_nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), (float*)tz.data_ptr(), 0, NULL, NULL, NULL, _planf)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		default:
			throw std::runtime_error("Dimension must be 1,2 or 3");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		if (cufinufft_makeplan(_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, 0, &_plan, NULL)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}

		switch (_ndim) {
		case 1:
		{
			auto tx = _coords.select(0, 0);
			if (cufinufft_setpts(_nfreq, (double*)tx.data_ptr(), NULL, NULL, 0, NULL, NULL, NULL, _plan)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 2:
		{
			auto tx = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);

			if (cufinufft_setpts(_nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL, _plan)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tx = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tz = _coords.select(0, 2);

			if (cufinufft_setpts(_nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), (double*)tz.data_ptr(), 0, NULL, NULL, NULL, _plan)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		default:
			throw std::runtime_error("Dimension must be 1,2 or 3");
		}
	}
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}

void Nufft::apply_type1(const at::Tensor& in, at::Tensor& out) const
{
	// Checks
	{
		if (_coords.get_device() != in.get_device() || _coords.get_device() != out.get_device()) {
			throw std::runtime_error("All tensors must reside on same device");
		}

		if (!out.is_contiguous()) {
			throw std::runtime_error("out must be contiguous");
		}

		if (!in.is_contiguous()) {
			throw std::runtime_error("in must be contiguous");
		}

		if (out.sizes() != _nmodes) {
			throw std::runtime_error("out.sizes() must equal nmodes given at construct");
		}

		if (in.sizes() != c10::IntArrayRef{ _ntransf, _nfreq }) {
			throw std::runtime_error("in tensor must match ntransf in first dim and nfreq in second dim");
		}

		if (in.dtype().toScalarType() == c10::ScalarType::ComplexFloat && _type != c10::ScalarType::Float) {
			throw std::runtime_error("if type(in) is complex float type(coords) must be float");
		}
		if (out.dtype().toScalarType() == c10::ScalarType::ComplexFloat && _type != c10::ScalarType::Float) {
			throw std::runtime_error("if type(out) is complex float type(coords) must be float");
		}

		if (in.dtype().toScalarType() == c10::ScalarType::ComplexDouble && _type != c10::ScalarType::Double) {
			throw std::runtime_error("if type(in) is complex double type(coords) must be double");
		}
		if (out.dtype().toScalarType() == c10::ScalarType::ComplexDouble && _type != c10::ScalarType::Double) {
			throw std::runtime_error("if type(out) is complex double type(coords) must be double");
		}
	}

	switch (_type) {
	case c10::ScalarType::Float:
	{
		cuFloatComplex* c = (cuFloatComplex*)in.data_ptr();
		cuFloatComplex* f = (cuFloatComplex*)out.data_ptr();

		if (cufinufftf_execute(c, f, _planf)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		cuDoubleComplex* c = (cuDoubleComplex*)in.data_ptr();
		cuDoubleComplex* f = (cuDoubleComplex*)out.data_ptr();

		if (cufinufft_execute(c, f, _plan)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}

void Nufft::apply_type2(const at::Tensor& in, at::Tensor& out) const
{
	// checks
	{
		if (_coords.get_device() != in.get_device() || _coords.get_device() != out.get_device()) {
			throw std::runtime_error("All tensors must reside on same device");
		}

		if (!out.is_contiguous()) {
			throw std::runtime_error("out must be contiguous");
		}

		if (!in.is_contiguous()) {
			throw std::runtime_error("in must be contiguous");
		}

		if (in.sizes() != _nmodes) {
			throw std::runtime_error("in.sizes() must equal nmodes given at construct");
		}

		if (out.sizes() != c10::IntArrayRef{ _ntransf, _nfreq }) {
			throw std::runtime_error("out tensor must match ntransf in first dim and nfreq in second dim");
		}

		if (in.dtype().toScalarType() == c10::ScalarType::ComplexFloat && _type != c10::ScalarType::Float) {
			throw std::runtime_error("if type(in) is complex float type(coords) must be float");
		}
		if (out.dtype().toScalarType() == c10::ScalarType::ComplexFloat && _type != c10::ScalarType::Float) {
			throw std::runtime_error("if type(out) is complex float type(coords) must be float");
		}

		if (in.dtype().toScalarType() == c10::ScalarType::ComplexDouble && _type != c10::ScalarType::Double) {
			throw std::runtime_error("if type(in) is complex double type(coords) must be double");
		}
		if (out.dtype().toScalarType() == c10::ScalarType::ComplexDouble && _type != c10::ScalarType::Double) {
			throw std::runtime_error("if type(out) is complex double type(coords) must be double");
		}

	}

	switch (_type) {
	case c10::ScalarType::Float:
	{
		cuFloatComplex* c = (cuFloatComplex*)out.data_ptr();
		cuFloatComplex* f = (cuFloatComplex*)in.data_ptr();

		if (cufinufftf_execute(c, f, _planf)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		cuDoubleComplex* c = (cuDoubleComplex*)out.data_ptr();
		cuDoubleComplex* f = (cuDoubleComplex*)in.data_ptr();

		if (cufinufft_execute(c, f, _plan)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}




namespace {

	NufftOptions make_ops_type(const NufftOptions& ops, NufftType type)
	{
		auto ret = ops;
		ret.type = type;
		return ret;
	}

}


NufftNormal::NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& forward_ops, const NufftOptions& backward_ops)
	: _coords(coords), _forward(coords, nmodes, forward_ops), _backward(coords, nmodes, backward_ops)
{

}

void NufftNormal::apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, std::optional<std::function<void(at::Tensor&)>> func_between) const
{
	_forward.apply(in, storage);
	if (func_between.has_value()) {
		func_between.value()(storage);
	}
	_backward.apply(storage, out);
}

const Nufft& NufftNormal::get_forward()
{
	return _forward;
}

const Nufft& NufftNormal::get_backward()
{
	return _backward;
}


namespace {

	void build_diagonal_base(const at::Tensor& coords, const std::vector<int64_t>& nmodes_ns,
		at::Tensor& diagonal, at::Tensor& frequency_storage, at::Tensor& input_storage)
	{
		NufftNormal normal(coords, nmodes_ns, { NufftType::eType2, true, 1e-6 }, { NufftType::eType1, true, 1e-6 });

		using namespace at::indexing;

		std::vector<TensorIndex> indices;
		for (int i = 0; i < nmodes_ns.size(); ++i) {
			indices.emplace_back(i == 0 ? TensorIndex(Slice()) : TensorIndex(0));
		}

		input_storage.zero_();
		input_storage.index_put_(at::makeArrayRef(indices), 1.0f);

		normal.apply(input_storage, diagonal, frequency_storage, std::nullopt);
	}

}


void ToeplitzNormalNufft::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, at::Tensor& diagonal)
{
	std::vector<int64_t> nmodes_ns(nmodes.size());
	for (int i = 0; i < nmodes.size(); ++i) {
		nmodes_ns[i] = i == 0 ? nmodes[i] : nmodes[i] * 2;
	}
	c10::TensorOptions options;

	// Input Storage
	options = c10::TensorOptions().device(coords.device()).dtype(coords.dtype());
	at::Tensor input_storage = at::empty(c10::makeArrayRef(nmodes_ns), options);

	// Frequency Storage
	switch (c10::typeMetaToScalarType(coords.dtype())) {
	case c10::ScalarType::Float:
		options = c10::TensorOptions().device(coords.device()).dtype(c10::ScalarType::ComplexFloat);
		break;
	case c10::ScalarType::Double:
		options = c10::TensorOptions().device(coords.device()).dtype(c10::ScalarType::ComplexDouble);
		break;
	default:
		throw std::runtime_error("Unsupported type in build_diagonal");
	}
	auto frequency_storage = at::empty({ nmodes[0], coords.size(1) }, options);


	// Build
	build_diagonal_base(coords, nmodes_ns, diagonal, frequency_storage, input_storage);

}

void ToeplitzNormalNufft::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, at::Tensor& diagonal,
	at::Tensor& storage, bool storage_is_frequency)
{
	std::vector<int64_t> nmodes_ns(nmodes.size());
	for (int i = 0; i < nmodes.size(); ++i) {
		nmodes_ns[i] = i == 0 ? nmodes[i] : nmodes[i] * 2;
	}
	c10::TensorOptions options;

	if (storage_is_frequency) {

		// Frequency Storage
		switch (c10::typeMetaToScalarType(coords.dtype())) {
		case c10::ScalarType::Float:
			options = c10::TensorOptions().device(coords.device()).dtype(c10::ScalarType::ComplexFloat);
			break;
		case c10::ScalarType::Double:
			options = c10::TensorOptions().device(coords.device()).dtype(c10::ScalarType::ComplexDouble);
			break;
		default:
			throw std::runtime_error("Unsupported type in build_diagonal");
		}
		auto frequency_storage = at::empty({ nmodes[0], coords.size(1) }, options);

		// Build
		build_diagonal_base(coords, nmodes_ns, diagonal, frequency_storage, storage);
	}
	else {
		// Input Storage
		options = c10::TensorOptions().device(coords.device()).dtype(coords.dtype());
		at::Tensor input_storage = at::empty(c10::makeArrayRef(nmodes_ns), options);

		// Build
		build_diagonal_base(coords, nmodes_ns, diagonal, storage, input_storage);
	}

}

void ToeplitzNormalNufft::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, at::Tensor& diagonal,
	at::Tensor& frequency_storage, at::Tensor& input_storage)
{	
	std::vector<int64_t> nmodes_ns(nmodes.size());
	for (int i = 0; i < nmodes.size(); ++i) {
		nmodes_ns[i] = i == 0 ? nmodes[i] : nmodes[i] * 2;
	}

	// Build
	build_diagonal_base(coords, nmodes_ns, diagonal, frequency_storage, input_storage);
}



ToeplitzNormalNufft::ToeplitzNormalNufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes,
	std::optional<std::reference_wrapper<at::Tensor>> diagonal,
	std::optional<std::reference_wrapper<at::Tensor>> frequency_storage,
	std::optional<std::reference_wrapper<at::Tensor>> input_storage)
	: _nmodes(nmodes)
{
	_type = coords.dtype().toScalarType();
	_ntransf = _nmodes[0];
	_ndim = coords.size(0);
	_nfreq = coords.size(1);

	_transdims.resize(_ndim);
	for (int i = 0; i < _ndim; ++i) {
		_transdims[i] = i + 1;
	}

	_nmodes_ns.resize(_nmodes.size());
	for (int i = 0; i < _nmodes.size(); ++i) {
		_nmodes_ns[i] = i == 0 ? _nmodes[i] : _nmodes[i] * 2;
	}

	if (_ndim + 1 != _nmodes.size()) {
		throw std::runtime_error("coords.size(0) must match number of nmodes given");
	}

	if (!coords.is_contiguous())
		throw std::runtime_error("coords must be contiguous");

	if (diagonal.has_value()) {
		_diagonal = diagonal.value();
	}
	else {
		c10::TensorOptions options;

		switch (_type) {
		case c10::ScalarType::Float:
			options = c10::TensorOptions().device(coords.device()).dtype(c10::ScalarType::ComplexFloat);
			break;
		case c10::ScalarType::Double:
			options = c10::TensorOptions().device(coords.device()).dtype(c10::ScalarType::ComplexDouble);
			break;
		default:
			throw std::runtime_error("Unsupported type in build_diagonal");
		}

		_diagonal = at::empty(at::makeArrayRef(_nmodes_ns), options);
	}

	if (frequency_storage.has_value() && input_storage.has_value()) {
		ToeplitzNormalNufft::build_diagonal(coords, _nmodes, _diagonal, frequency_storage.value(), input_storage.value());
	}
	else if (frequency_storage.has_value()) {
		ToeplitzNormalNufft::build_diagonal(coords, _nmodes, _diagonal, frequency_storage.value(), true);
	}
	else if (input_storage.has_value()) {
		ToeplitzNormalNufft::build_diagonal(coords, _nmodes, _diagonal, input_storage.value(), false);
	}
	else {
		ToeplitzNormalNufft::build_diagonal(coords, _nmodes, _diagonal);
	}

}

void ToeplitzNormalNufft::apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage) const
{
	torch::fft_fftn_out(storage, in, at::makeArrayRef(std::vector<int64_t>{_nmodes_ns.begin()+1, _nmodes_ns.end()}), at::makeArrayRef(_transdims));
	storage.mul_(_diagonal);
	torch::fft_fftn_out(out, storage, at::makeArrayRef(std::vector<int64_t>{ _nmodes.begin()+1, _nmodes.end() }), at::makeArrayRef(_transdims));
}



