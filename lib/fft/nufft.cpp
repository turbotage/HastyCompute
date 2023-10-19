
//#include <cufinufft.h>

#include "nufft.hpp"

#include <finufft.h>
#include <cufinufft.h>


import hasty_util;



at::Tensor hasty::nufft::allocate_out(const at::Tensor& coords, int ntransf)
{
	return at::empty({ ntransf, coords.size(1) }, at::TensorOptions(coords.device()).dtype(
		torch_util::complex_type(coords.dtype().toScalarType(), {})));
}

at::Tensor hasty::nufft::allocate_adjoint_out(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
{
	return at::empty(at::makeArrayRef(nmodes), at::TensorOptions(coords.device()).dtype(
		torch_util::complex_type(coords.dtype().toScalarType(), {})));
}

// NUFFT

hasty::nufft::Nufft::Nufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& opts)
	: _coords(coords), _nmodes(nmodes), _opts(opts)
{
	_type = _coords.dtype().toScalarType();
	_ntransf = _nmodes[0];
	_ndim = _coords.size(0);
	_nfreq = _coords.size(1);
	_nvoxels = 1;
	for (int i = 0; i < _ndim; ++i) {
		_nvoxels *= nmodes[i + 1];
	}

	if (_opts.get_type() == NufftType::eType3) {
		throw std::runtime_error("Type 3 Nufft is not yet supported");
	}

	if (_ndim + 1 != _nmodes.size()) {
		throw std::runtime_error("coords.size(0) must match number of nmodes given");
	}

	for (int i = 0; i < _ndim; ++i) {
		_nmodes_flipped[i] = _nmodes[_ndim - i];
	}

	if (!_coords.is_contiguous())
		throw std::runtime_error("coords must be contiguous");

	make_plan_set_pts();

}

void hasty::nufft::Nufft::close()
{
	if (!_closed) {
		_closed = true;
		switch (_type) {
		case c10::ScalarType::Float:
		{
			if (_planf == nullptr) {
				throw std::runtime_error("Tried to close on Nufft with _planf == nullptr");
			}
			if (finufftf_destroy(_planf)) {
				throw std::runtime_error("finufftf_destroy() failed");
			}
			_planf = nullptr;
		}
		break;
		case c10::ScalarType::Double:
		{
			if (_plan == nullptr) {
				throw std::runtime_error("Tried to close on Nufft with _plan == nullptr");
			}
			if (finufft_destroy(_plan)) {
				throw std::runtime_error("finufft_destroy() failed");
			}
			_planf = nullptr;
		}
		break;
		default:
			throw std::runtime_error("Coordinates must be real! close()");
		}
	}
}

hasty::nufft::Nufft::~Nufft() {
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

void hasty::nufft::Nufft::apply(const at::Tensor& in, at::Tensor& out) const
{
	switch (_opts.get_type()) {
	case eType1:
	{
		apply_type1(in, out);
		out.div_((float)std::sqrt(_nvoxels));
	}
	break;
	case eType2:
	{
		apply_type2(in, out);
		out.div_((float)std::sqrt(_nvoxels));
	}
	break;

	default:
		throw std::runtime_error("Only Nufft Type 1 and Type 2 is supported");
	}
}

at::ScalarType hasty::nufft::Nufft::coord_type()
{
	return _type;
}

at::ScalarType hasty::nufft::Nufft::complex_type()
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

int32_t hasty::nufft::Nufft::nfreq()
{
	return _nfreq;
}

int32_t hasty::nufft::Nufft::ndim()
{
	return _ndim;
}

void hasty::nufft::Nufft::make_plan_set_pts()
{

	using namespace at::indexing;

	auto device = _coords.device();

	if (device.is_cuda())
		throw std::runtime_error("Coordinates did not reside in cpu");

	int cuda_device_idx = device.index();

	switch (_type) {
	case c10::ScalarType::Float:
	{
		finufftf_default_opts(&_finufft_opts);

		if (finufftf_makeplan(_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, &_planf, &_finufft_opts))
			throw std::runtime_error("finufft_makeplan failed");

		switch (_ndim) {
		case 1:
		{
			auto tx = _coords.select(0, 0);
			if (finufftf_setpts(_planf, _nfreq, (float*)tx.data_ptr(), NULL, NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("finufftf_setpts failed");
			}
		}
		break;
		case 2:
		{
			auto ty = _coords.select(0, 0);
			auto tx = _coords.select(0, 1);
			if (finufftf_setpts(_planf, _nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("finufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tz = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tx = _coords.select(0, 2);
			if (finufftf_setpts(_planf, _nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), (float*)tz.data_ptr(), 0, NULL, NULL, NULL)) {
				throw std::runtime_error("finufftf_setpts failed");
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
		finufft_default_opts(&_finufft_opts);

		if (finufft_makeplan(_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, &_plan, &_finufft_opts))
			throw std::runtime_error("cufinufft_makeplan failed");

		switch (_ndim) {
		case 1:
		{
			auto tx = _coords.select(0, 0);
			if (finufft_setpts(_plan, _nfreq, (double*)tx.data_ptr(), NULL, NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("finufftf_setpts failed");
			}
		}
		break;
		case 2:
		{
			auto ty = _coords.select(0, 0);
			auto tx = _coords.select(0, 1);
			if (finufft_setpts(_plan, _nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("finufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tz = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tx = _coords.select(0, 2);
			if (finufft_setpts(_plan, _nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), (double*)tz.data_ptr(), 0, NULL, NULL, NULL)) {
				throw std::runtime_error("finufftf_setpts failed");
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

void hasty::nufft::Nufft::apply_type1(const at::Tensor& in, at::Tensor& out) const
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
		std::complex<float>* c = (std::complex<float>*)in.data_ptr();
		std::complex<float>* f = (std::complex<float>*)out.data_ptr();

		if (finufftf_execute(_planf, c, f)) {
			throw std::runtime_error("cufinufft_execute failed");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		std::complex<double>* c = (std::complex<double>*)in.data_ptr();
		std::complex<double>* f = (std::complex<double>*)out.data_ptr();

		if (finufft_execute(_plan, c, f)) {
			throw std::runtime_error("cufinufft_execute failed");
		}
	}
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}

void hasty::nufft::Nufft::apply_type2(const at::Tensor& in, at::Tensor& out) const
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
		std::complex<float>* c = (std::complex<float>*)out.data_ptr();
		std::complex<float>* f = (std::complex<float>*)in.data_ptr();

		if (finufftf_execute(_planf, c, f)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		std::complex<double>* c = (std::complex<double>*)out.data_ptr();
		std::complex<double>* f = (std::complex<double>*)in.data_ptr();

		if (finufft_execute(_plan, c, f)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}


at::Tensor hasty::nufft::allocate_normal_out(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
{
	return at::empty(at::makeArrayRef(nmodes), at::TensorOptions(coords.device()).dtype(
		torch_util::complex_type(coords.dtype().toScalarType(), {})));
}

at::Tensor hasty::nufft::allocate_normal_storage(const at::Tensor& coords, int ntransf)
{
	return at::empty({ ntransf, coords.size(1) }, at::TensorOptions(coords.device()).dtype(
		torch_util::complex_type(coords.dtype().toScalarType(), {})));
}

at::Tensor hasty::nufft::allocate_normal_adjoint_out(const at::Tensor& coords, int ntransf)
{
	return at::empty({ ntransf, coords.size(1) }, at::TensorOptions(coords.device()).dtype(
		torch_util::complex_type(coords.dtype().toScalarType(), {})));
}

at::Tensor hasty::nufft::allocate_normal_adjoint_storage(const at::Tensor& coords, const std::vector<int64_t>& nmodes)
{
	return at::empty(at::makeArrayRef(nmodes), at::TensorOptions(coords.device()).dtype(
		torch_util::complex_type(coords.dtype().toScalarType(), {})));
}

// CUDA-NUFFT

hasty::nufft::CUDANufft::CUDANufft(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& opts)
	: _coords(coords), _nmodes(nmodes), _opts(opts)
{
	_type = _coords.dtype().toScalarType();
	_ntransf = _nmodes[0];
	_ndim = _coords.size(0);
	_nfreq = _coords.size(1);
	_nvoxels = 1;
	for (int i = 0; i < _ndim; ++i) {
		_nvoxels *= nmodes[i + 1];
	}

	if (_opts.get_type() == NufftType::eType3) {
		throw std::runtime_error("Type 3 Nufft is not yet supported");
	}

	if (_ndim + 1 != _nmodes.size()) {
		throw std::runtime_error("coords.size(0) must match number of nmodes given");
	}

	for (int i = 0; i < _ndim; ++i) {
		_nmodes_flipped[i] = _nmodes[_ndim - i];
	}

	if (!_coords.is_contiguous())
		throw std::runtime_error("coords must be contiguous");

	make_plan_set_pts();

}

void hasty::nufft::CUDANufft::close()
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

hasty::nufft::CUDANufft::~CUDANufft() {
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

void hasty::nufft::CUDANufft::apply(const at::Tensor& in, at::Tensor& out) const
{
	switch (_opts.get_type()) {
	case eType1:
	{
		apply_type1(in, out);
		out.div_((float)std::sqrt(_nvoxels));
	}
	break;
	case eType2:
	{
		apply_type2(in, out);
		out.div_((float)std::sqrt(_nvoxels));
	}
	break;

	default:
		throw std::runtime_error("Only Nufft Type 1 and Type 2 is supported");
	}
}

at::ScalarType hasty::nufft::CUDANufft::coord_type()
{
	return _type;
}

at::ScalarType hasty::nufft::CUDANufft::complex_type()
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

int32_t hasty::nufft::CUDANufft::nfreq()
{
	return _nfreq;
}

int32_t hasty::nufft::CUDANufft::ndim()
{
	return _ndim;
}

void hasty::nufft::CUDANufft::make_plan_set_pts()
{

	using namespace at::indexing;

	auto device = _coords.device();

	if (!device.is_cuda())
		throw std::runtime_error("Coordinates did not reside in cuda device");

	int cuda_device_idx = device.index();

	switch (_type) {
	case c10::ScalarType::Float:
	{
		cufinufft_default_opts(&_finufft_opts);

		_finufft_opts.gpu_device_id = cuda_device_idx;

		if (cufinufftf_makeplan(_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, &_planf, &_finufft_opts))
			throw std::runtime_error("cufinufft_makeplan failed");

		switch (_ndim) {
		case 1:
		{
			auto tx = _coords.select(0, 0);
			if (cufinufftf_setpts(_planf, _nfreq, (float*)tx.data_ptr(), NULL, NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 2:
		{
			auto ty = _coords.select(0, 0);
			auto tx = _coords.select(0, 1);
			if (cufinufftf_setpts(_planf, _nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tz = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tx = _coords.select(0, 2);
			if (cufinufftf_setpts(_planf, _nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), (float*)tz.data_ptr(), 0, NULL, NULL, NULL)) {
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
		cufinufft_default_opts(&_finufft_opts);

		_finufft_opts.gpu_device_id = cuda_device_idx;

		if (cufinufft_makeplan(_opts.get_type(), _ndim, _nmodes_flipped.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, &_plan, &_finufft_opts))
			throw std::runtime_error("cufinufft_makeplan failed");

		switch (_ndim) {
		case 1:
		{
			auto tx = _coords.select(0, 0);
			if (cufinufft_setpts(_plan, _nfreq, (double*)tx.data_ptr(), NULL, NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 2:
		{
			auto ty = _coords.select(0, 0);
			auto tx = _coords.select(0, 1);
			if (cufinufft_setpts(_plan, _nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), NULL, 0, NULL, NULL, NULL)) {
				throw std::runtime_error("cufinufftf_setpts failed");
			}
		}
		break;
		case 3:
		{
			auto tz = _coords.select(0, 0);
			auto ty = _coords.select(0, 1);
			auto tx = _coords.select(0, 2);
			if (cufinufft_setpts(_plan, _nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), (double*)tz.data_ptr(), 0, NULL, NULL, NULL)) {
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

void hasty::nufft::CUDANufft::apply_type1(const at::Tensor& in, at::Tensor& out) const
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

		if (cufinufftf_execute(_planf, c, f)) {
			throw std::runtime_error("cufinufft_execute failed");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		cuDoubleComplex* c = (cuDoubleComplex*)in.data_ptr();
		cuDoubleComplex* f = (cuDoubleComplex*)out.data_ptr();

		if (cufinufft_execute(_plan, c, f)) {
			throw std::runtime_error("cufinufft_execute failed");
		}
	}
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}

void hasty::nufft::CUDANufft::apply_type2(const at::Tensor& in, at::Tensor& out) const
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

		if (cufinufftf_execute(_planf, c, f)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	case c10::ScalarType::Double:
	{
		cuDoubleComplex* c = (cuDoubleComplex*)out.data_ptr();
		cuDoubleComplex* f = (cuDoubleComplex*)in.data_ptr();

		if (cufinufft_execute(_plan, c, f)) {
			throw std::runtime_error("cufinufft_makeplan failed");
		}
	}
	break;
	default:
		throw std::runtime_error("Coordinates must be real!");
	}
}




namespace {

	hasty::nufft::NufftOptions make_ops_type(const hasty::nufft::NufftOptions& ops, hasty::nufft::NufftType type)
	{
		auto ret = ops;
		ret.type = type;
		return ret;
	}

}


hasty::nufft::NufftNormal::NufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& forward_ops, const NufftOptions& backward_ops)
	: _forward(coords, nmodes, forward_ops), _backward(coords, nmodes, backward_ops)
{

}

void hasty::nufft::NufftNormal::apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const
{
	_forward.apply(in, storage);
	if (func_between.has_value()) {
		func_between.value()(storage);
	}
	_backward.apply(storage, out);
}

void hasty::nufft::NufftNormal::apply_inplace(at::Tensor& in, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const
{
	_forward.apply(in, storage);

	if (func_between.has_value()) {
		func_between.value()(storage);
	}

	_backward.apply(storage, in);
}

int32_t hasty::nufft::NufftNormal::nfreq()
{
	return _forward.nfreq();
}

int32_t hasty::nufft::NufftNormal::ndim()
{
	return _forward.ndim();
}

at::ScalarType hasty::nufft::NufftNormal::coord_type()
{
	return _forward.coord_type();
}

at::ScalarType hasty::nufft::NufftNormal::complex_type()
{
	return _forward.complex_type();
}

const hasty::nufft::Nufft& hasty::nufft::NufftNormal::get_forward()
{
	return _forward;
}

const hasty::nufft::Nufft& hasty::nufft::NufftNormal::get_backward()
{
	return _backward;
}






hasty::nufft::CUDANufftNormal::CUDANufftNormal(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const NufftOptions& forward_ops, const NufftOptions& backward_ops)
	: _forward(coords, nmodes, forward_ops), _backward(coords, nmodes, backward_ops)
{

}

void hasty::nufft::CUDANufftNormal::apply(const at::Tensor& in, at::Tensor& out, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const
{
	_forward.apply(in, storage);
	if (func_between.has_value()) {
		func_between.value()(storage);
	}
	_backward.apply(storage, out);
}

void hasty::nufft::CUDANufftNormal::apply_inplace(at::Tensor& in, at::Tensor& storage, at::optional<std::function<void(at::Tensor&)>> func_between) const
{
	_forward.apply(in, storage);

	if (func_between.has_value()) {
		func_between.value()(storage);
	}

	_backward.apply(storage, in);
}

int32_t hasty::nufft::CUDANufftNormal::nfreq()
{
	return _forward.nfreq();
}

int32_t hasty::nufft::CUDANufftNormal::ndim()
{
	return _forward.ndim();
}

at::ScalarType hasty::nufft::CUDANufftNormal::coord_type()
{
	return _forward.coord_type();
}

at::ScalarType hasty::nufft::CUDANufftNormal::complex_type()
{
	return _forward.complex_type();
}

const hasty::nufft::CUDANufft& hasty::nufft::CUDANufftNormal::get_forward()
{
	return _forward;
}

const hasty::nufft::CUDANufft& hasty::nufft::CUDANufftNormal::get_backward()
{
	return _backward;
}









namespace {

	void build_diagonal_base(const at::Tensor& coords, const std::vector<int64_t>& nmodes_ns, double tol,
		at::Tensor& diagonal, at::Tensor& frequency_storage, at::Tensor& input_storage)
	{
		c10::InferenceMode guard;

		hasty::nufft::CUDANufftNormal normal(coords, nmodes_ns, { hasty::nufft::NufftType::eType2, false, 1e-6 }, { hasty::nufft::NufftType::eType1, true, 1e-6 });

		using namespace at::indexing;
		
		std::vector<TensorIndex> indices;
		for (int i = 0; i < nmodes_ns.size(); ++i) {
			indices.emplace_back(i == 0 ? Slice() : TensorIndex(nmodes_ns[i] / 2));
			//indices.emplace_back(i == 0 ? Slice() : TensorIndex(0));
		}

		input_storage.zero_();
		input_storage.index_put_(at::makeArrayRef(indices), 1.0f);

		//std::cout << "unity_vector:\n " << torch_util::print_4d_xyz(input_storage).str() << std::endl;

		normal.apply(input_storage, diagonal, frequency_storage, at::nullopt);
		
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


void hasty::nufft::CUDANormalNufftToeplitz::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal)
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

void hasty::nufft::CUDANormalNufftToeplitz::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
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

void hasty::nufft::CUDANormalNufftToeplitz::build_diagonal(const at::Tensor& coords, std::vector<int64_t> nmodes, double tol, at::Tensor& diagonal,
	at::Tensor& frequency_storage, at::Tensor& input_storage)
{	
	std::vector<int64_t> nmodes_ns(nmodes.size());
	for (int i = 0; i < nmodes.size(); ++i) {
		nmodes_ns[i] = i == 0 ? nmodes[i] : nmodes[i] * 2;
	}

	// Build
	build_diagonal_base(coords, nmodes_ns, tol, diagonal, frequency_storage, input_storage);
}

hasty::nufft::CUDANormalNufftToeplitz::CUDANormalNufftToeplitz(const at::Tensor& coords, const std::vector<int64_t>& nmodes, const at::optional<double>& tol,
	const at::optional<std::reference_wrapper<at::Tensor>>& diagonal,
	const at::optional<std::reference_wrapper<at::Tensor>>& frequency_storage,
	const at::optional<std::reference_wrapper<at::Tensor>>& input_storage)
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
		CUDANormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal, frequency_storage.value(), input_storage.value());
	}
	else if (frequency_storage.has_value()) {
		CUDANormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal, frequency_storage.value(), true);
	}
	else if (input_storage.has_value()) {
		CUDANormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal, input_storage.value(), false);
	}
	else {
		CUDANormalNufftToeplitz::build_diagonal(coords, _nmodes, toler, _diagonal);
	}

}

hasty::nufft::CUDANormalNufftToeplitz::CUDANormalNufftToeplitz(at::Tensor&& diagonal, const std::vector<int64_t>& nmodes)
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

at::Tensor hasty::nufft::CUDANormalNufftToeplitz::apply(const at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	return storage2.index(_indices);
}

void hasty::nufft::CUDANormalNufftToeplitz::apply_add(const at::Tensor& in, at::Tensor& add_to, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	add_to.add_(storage2.index(_indices));
}

void hasty::nufft::CUDANormalNufftToeplitz::apply_addcmul(const at::Tensor& in, at::Tensor& add_to, const at::Tensor& mul, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	add_to.addcmul_(storage2.index(_indices), mul);
}

void hasty::nufft::CUDANormalNufftToeplitz::apply_inplace(at::Tensor& in, at::Tensor& storage1, at::Tensor& storage2) const
{
	c10::InferenceMode guard;
	at::fft_fftn_out(storage1, in, _expanded_dims, _transform_dims);
	storage1.mul_(_diagonal);
	at::fft_ifftn_out(storage2, storage1, c10::nullopt, _transform_dims);
	in.copy_(storage2.index(_indices));
}

at::Tensor hasty::nufft::CUDANormalNufftToeplitz::get_diagonal()
{
	return _diagonal;
}



