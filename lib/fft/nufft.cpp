
//#include <cufinufft.h>

#include "nufft.hpp"

#include <cufinufft.h>

import hasty_util;

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

	for (int i = 0; i < _ndim; ++i) {
		_nmodes_flipped[i] = _nmodes[_ndim - i];
		//_nmodes_flipped[i] = _nmodes[i + 1];
	}

	if (!_coords.is_contiguous())
		throw std::runtime_error("coords must be contiguous");

	make_plan_set_pts();

}

void Nufft::close()
{
	if (!_closed) {
		_closed = true;
		switch (_type) {
		case c10::ScalarType::Float:
		{
			if (_planf == nullptr) {
				throw std::runtime_error("Tried to close on Nufft with _planf == nullptr");
			}
			if (cufinufftf_destroy(_planf)) {
				throw std::runtime_error("cufinufftf_destroy() failed");
			}
			_planf = nullptr;
		}
		break;
		case c10::ScalarType::Double:
		{
			if (_plan == nullptr) {
				throw std::runtime_error("Tried to close on Nufft with _plan == nullptr");
			}
			if (cufinufft_destroy(_plan)) {
				throw std::runtime_error("cufinufft_destroy() failed");
			}
			_planf = nullptr;
		}
		break;
		default:
			throw std::runtime_error("Coordinates must be real! close()");
		}
	}
}

Nufft::~Nufft() {
	try {
		close();
	}
	catch (std::runtime_error& e) {
		std::cerr << e.what();
		std::exit(EXIT_FAILURE);
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

c10::ScalarType Nufft::coord_type()
{
	return _type;
}

c10::ScalarType Nufft::complex_type()
{
	switch (_type) {
	case c10::ScalarType::Float:
		return c10::ScalarType::ComplexFloat;
	break;
	case c10::ScalarType::Double:
		return c10::ScalarType::ComplexDouble;
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}

int32_t Nufft::nfreq()
{
	return _nfreq;
}

int32_t Nufft::ndim()
{
	return _ndim;
}



void Nufft::make_plan_set_pts()
{

	using namespace at::indexing;

	auto device = _coords.device();

	if (!device.is_cuda())
		throw std::runtime_error("Coordinates did not reside in cuda device");

	int cuda_device_idx = device.index();

	switch (_type) {
	case c10::ScalarType::Float:
	{
		if (cufinufftf_default_opts(_opts.get_type(), _ndim, &_finufft_opts))
			throw std::runtime_error("Failed to create cufinufft_default_opts");

		_finufft_opts.gpu_device_id = cuda_device_idx;

		if (cufinufftf_makeplan(_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, 0, &_planf, &_finufft_opts))
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
			auto ty = _coords.select(0, 0);
			auto tx = _coords.select(0, 1);
			if (cufinufftf_setpts(_nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL, _planf)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tz = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tx = _coords.select(0, 2);
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
		if (cufinufft_default_opts(_opts.get_type(), _ndim, &_finufft_opts))
			throw std::runtime_error("Failed to create cufinufft_default_opts");

		_finufft_opts.gpu_device_id = cuda_device_idx;

		if (cufinufft_makeplan(_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, 0, &_plan, &_finufft_opts))
			throw std::runtime_error("cufinufft_makeplan failed");

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
			auto ty = _coords.select(0, 0);
			auto tx = _coords.select(0, 1);
			if (cufinufft_setpts(_nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL, _plan)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tz = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tx = _coords.select(0, 2);
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
			throw std::runtime_error("cufinufft_execute failed");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		cuDoubleComplex* c = (cuDoubleComplex*)in.data_ptr();
		cuDoubleComplex* f = (cuDoubleComplex*)out.data_ptr();

		if (cufinufft_execute(c, f, _plan)) {
			throw std::runtime_error("cufinufft_execute failed");
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
	: _forward(coords, nmodes, forward_ops), _backward(coords, nmodes, backward_ops)
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

void NufftNormal::apply_inplace(at::Tensor& in, at::Tensor& storage, std::optional<std::function<void(at::Tensor&)>> func_between) const
{
	_forward.apply(in, storage);

	if (func_between.has_value()) {
		func_between.value()(storage);
	}

	_backward.apply(storage, in);
}

int32_t NufftNormal::nfreq()
{
	return _forward.nfreq();
}

int32_t NufftNormal::ndim()
{
	return _forward.ndim();
}

c10::ScalarType NufftNormal::coord_type()
{
	return _forward.coord_type();
}

c10::ScalarType NufftNormal::complex_type()
{
	return _forward.complex_type();
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

	void build_diagonal_base(const at::Tensor& coords, const std::vector<int64_t>& nmodes_ns, double tol,
		at::Tensor& diagonal, at::Tensor& frequency_storage, at::Tensor& input_storage)
	{
		c10::InferenceMode guard;

		NufftNormal normal(coords, nmodes_ns, { NufftType::eType2, false, 1e-6 }, { NufftType::eType1, true, 1e-6 });

		using namespace at::indexing;
		
		std::vector<TensorIndex> indices;
		for (int i = 0; i < nmodes_ns.size(); ++i) {
			indices.emplace_back(i == 0 ? Slice() : TensorIndex(nmodes_ns[i] / 2));
			//indices.emplace_back(i == 0 ? Slice() : TensorIndex(0));
		}

		input_storage.zero_();
		input_storage.index_put_(at::makeArrayRef(indices), 1.0f);

		//std::cout << "unity_vector:\n " << torch_util::print_4d_xyz(input_storage).str() << std::endl;

		normal.apply(input_storage, diagonal, frequency_storage, std::nullopt);
		
		/*
		std::cout << "after normal: " << std::endl;
		std::cout << torch_util::print_4d_xyz(at::real(diagonal)).str() << std::endl;
		std::cout << torch_util::print_4d_xyz(at::imag(diagonal)).str() << std::endl;
		*/

		std::vector<int64_t> transdims(coords.size(0));
		for (int i = 0; i < coords.size(0); ++i) {
			transdims[i] = i + 1;
		}

		auto transform_dims = at::makeArrayRef(transdims);

		input_storage = at::fft_ifftshift(diagonal, transform_dims);

		at::fft_fftn_out(diagonal, input_storage, c10::nullopt, transform_dims);

	}

}


void NormalNufftToeplitz::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal)
{
	std::vector<int64_t> nmodes_ns(nmodes.size());
	for (int i = 0; i < nmodes.size(); ++i) {
		nmodes_ns[i] = i == 0 ? nmodes[i] : nmodes[i] * 2;
	}
	c10::TensorOptions options;

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

	// Frequency Storage
	at::Tensor frequency_storage = at::empty({ nmodes[0], coords.size(1) }, options);

	// Input Storage
	at::Tensor input_storage = at::empty(c10::makeArrayRef(nmodes_ns), options);

	// Build
	build_diagonal_base(coords, nmodes_ns, tol, diagonal, frequency_storage, input_storage);

}

void NormalNufftToeplitz::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
	at::Tensor& storage, bool storage_is_frequency)
{
	std::vector<int64_t> nmodes_ns(nmodes.size());
	for (int i = 0; i < nmodes.size(); ++i) {
		nmodes_ns[i] = i == 0 ? nmodes[i] : nmodes[i] * 2;
	}
	c10::TensorOptions options;
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

	if (storage_is_frequency) {
		
		// Frequency Storage
		at::Tensor frequency_storage = at::empty({ nmodes[0], coords.size(1) }, options);

		// Build
		build_diagonal_base(coords, nmodes_ns, tol, diagonal, frequency_storage, storage);
	}
	else {
		// Input Storage
		at::Tensor input_storage = at::empty(c10::makeArrayRef(nmodes_ns), options);

		// Build
		build_diagonal_base(coords, nmodes_ns, tol, diagonal, storage, input_storage);
	}

}

void NormalNufftToeplitz::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
	at::Tensor& frequency_storage, at::Tensor& input_storage)
{	
	std::vector<int64_t> nmodes_ns(nmodes.size());
	for (int i = 0; i < nmodes.size(); ++i) {
		nmodes_ns[i] = i == 0 ? nmodes[i] : nmodes[i] * 2;
	}

	// Build
	build_diagonal_base(coords, nmodes_ns, tol, diagonal, frequency_storage, input_storage);
}

NormalNufftToeplitz::NormalNufftToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, std::optional<double> tol,
	std::optional<std::reference_wrapper<at::Tensor>> diagonal,
	std::optional<std::reference_wrapper<at::Tensor>> frequency_storage,
	std::optional<std::reference_wrapper<at::Tensor>> input_storage)
	: _nmodes(nmodes), _created_from_diagonal(false)
{
	c10::InferenceMode guard;
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

	_transform_dims = at::makeArrayRef(_transdims);
	_expanded_dims = at::makeArrayRef(_nmodes_ns.data()+1,3);
	
	using namespace at::indexing;

	for (int i = 0; i < _nmodes.size(); ++i) {
		int64_t start = _nmodes[i] / 2;
		int64_t end = start + _nmodes[i];
		_index_vector.emplace_back(i == 0 ? Slice() : Slice(0, _nmodes[i]));
	}
	_indices = at::makeArrayRef(_index_vector);

	double toler = tol.has_value() ? tol.value() : 1e-6;

	if (frequency_storage.has_value() && input_storage.has_value()) {
		NormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal, frequency_storage.value(), input_storage.value());
	}
	else if (frequency_storage.has_value()) {
		NormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal, frequency_storage.value(), true);
	}
	else if (input_storage.has_value()) {
		NormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal, input_storage.value(), false);
	}
	else {
		NormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal);
	}

}

NormalNufftToeplitz::NormalNufftToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes)
	: _diagonal(std::move(diagonal)), _nmodes(nmodes), _created_from_diagonal(true)
{
	c10::InferenceMode guard;
	_ntransf = _nmodes[0];
	_ndim = _diagonal.sizes().size() - 1;

	_transdims.resize(_ndim);
	for (int i = 0; i < _ndim; ++i) {
		_transdims[i] = i + 1;
	}

	_nmodes_ns.resize(_nmodes.size());
	for (int i = 0; i < _nmodes.size(); ++i) {
		_nmodes_ns[i] = i == 0 ? _nmodes[i] : _nmodes[i] * 2;
	}

	_transform_dims = at::makeArrayRef(_transdims);
	_expanded_dims = at::makeArrayRef(_nmodes_ns.data() + 1, 3);

	using namespace at::indexing;

	for (int i = 0; i < _nmodes.size(); ++i) {
		int64_t start = _nmodes[i] / 2;
		int64_t end = start + _nmodes[i];
		_index_vector.emplace_back(i == 0 ? Slice() : Slice(0, _nmodes[i]));
	}
	_indices = at::makeArrayRef(_index_vector);
}

at::Tensor NormalNufftToeplitz::apply(const at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	return storage2.index(_indices);
}

void NormalNufftToeplitz::apply_add(const at::Tensor& in, at::Tensor& add_to, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	add_to.add_(storage2.index(_indices));
}

void NormalNufftToeplitz::apply_addcmul(const at::Tensor& in, at::Tensor& add_to, const at::Tensor& mul, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	add_to.addcmul_(storage2.index(_indices), mul);
}

void NormalNufftToeplitz::apply_inplace(at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	in.copy_(storage2.index(_indices));
}

at::Tensor NormalNufftToeplitz::get_diagonal()
{
	return _diagonal;
}



