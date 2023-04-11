module;

#include <cufinufft.h>

#include <ATen/ATen.h>
//#include <Aten/cuda/CUDAContext.h>

export module fft_cu;

import hasty_util;

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <array>;
import <stdexcept>;
#endif

/*
namespace hasty {

	namespace cuda {

		export struct NufftOptions {
			bool positive = true;
			double tol = 1e-6;

			int get_positive() const { return positive ? 1 : -1; }

			double get_tol() const { return tol; }
		};

		export enum NufftType {
			eType1 = 1,
			eType2 = 2,
			eType3 = 3
		};

		export class Nufft {
		public:

			Nufft(const at::Tensor& coords, const std::vector<i32>& nmodes, const NufftType& type, const NufftOptions& opts = NufftOptions{})
				: _coords(coords), _nmodes(nmodes), _nufftType(type), _opts(opts)
			{
				_type = _coords.dtype().toScalarType();
				_ntransf = _nmodes[0];
				_nfreq = _coords.size(0);

				if (_nufftType == NufftType::eType3)
					throw std::runtime_error("Type 3 Nufft is not yet supported");

				if (_nfreq + 1 != _nmodes.size())
					throw std::runtime_error("coords.size(0) must match number of nmodes given");

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

			~Nufft() {
				try {
					switch (_type) {
					case c10::ScalarType::Float:
					{
						if (cufinufftf_destroy(_planf))
							std::exit(EXIT_FAILURE);
					}
					break;
					case c10::ScalarType::Double:
					{
						if (cufinufft_destroy(_plan))
							std::exit(EXIT_FAILURE);
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

			void apply(const at::Tensor& in, at::Tensor& out)
			{
				switch (_nufftType) {
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

		protected:

			void make_plan_set_pts() {

				using namespace at::indexing;

				switch (_type) {
				case c10::ScalarType::Float:
				{
					if (cufinufftf_makeplan(_nufftType, _ndim, _nmodes.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, 0, &_planf, nullptr))
						throw std::runtime_error("cufinufft_makeplan failed");

					switch (_ndim) {
					case 1:
					{
						if (cufinufftf_setpts(_nfreq, (float*)_coords.data_ptr(), nullptr, nullptr,
							0, nullptr, nullptr, nullptr, _planf))
							throw std::runtime_error("cufinufftf_setpts failed");
					}
					break;
					case 2:
					{
						auto tx = _coords.index({ 0, Slice() });
						auto ty = _coords.index({ 1, Slice() });
						if (!tx.is_view() || !ty.is_view())
							throw std::runtime_error("_coords[index] was not view");

						if (cufinufftf_setpts(_nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), nullptr, 0, nullptr, nullptr, nullptr, _planf))
							throw std::runtime_error("cufinufftf_setpts failed");
					}
					break;
					case 3:
					{
						auto tx = _coords.index({ 0, Slice() });
						auto ty = _coords.index({ 1, Slice() });
						auto tz = _coords.index({ 2, Slice() });
						if (!tx.is_view() || !ty.is_view())
							throw std::runtime_error("_coords[index] was not view");

						if (cufinufftf_setpts(_nfreq, (float*)tx.data_ptr(), (float*)ty.data_ptr(), nullptr, 0, nullptr, nullptr, nullptr, _planf))
							throw std::runtime_error("cufinufftf_setpts failed");
					}
					break;
					default:
						throw std::runtime_error("Dimension must be 1,2 or 3");
					}
				}
				break;
				case c10::ScalarType::Double:
				{
					if (cufinufft_makeplan(_nufftType, _ndim, _nmodes.data(), _opts.get_positive(), _ntransf, (float)_opts.tol, 0, &_plan, nullptr))
						throw std::runtime_error("cufinufft_makeplan failed");

					switch (_ndim) {
					case 1:
					{
						if (cufinufft_setpts(_nfreq, (double*)_coords.data_ptr(), nullptr, nullptr,
							0, nullptr, nullptr, nullptr, _plan))
							throw std::runtime_error("cufinufftf_setpts failed");
					}
					break;
					case 2:
					{
						auto tx = _coords.index({ 0, Slice() });
						auto ty = _coords.index({ 1, Slice() });
						if (!tx.is_view() || !ty.is_view())
							throw std::runtime_error("_coords[index] was not view");

						if (cufinufft_setpts(_nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), nullptr, 0, nullptr, nullptr, nullptr, _plan))
							throw std::runtime_error("cufinufftf_setpts failed");
					}
					break;
					case 3:
					{
						auto tx = _coords.index({ 0, Slice() });
						auto ty = _coords.index({ 1, Slice() });
						auto tz = _coords.index({ 2, Slice() });
						if (!tx.is_view() || !ty.is_view())
							throw std::runtime_error("_coords[index] was not view");

						if (cufinufft_setpts(_nfreq, (double*)tx.data_ptr(), (double*)ty.data_ptr(), nullptr, 0, nullptr, nullptr, nullptr, _plan))
							throw std::runtime_error("cufinufftf_setpts failed");
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

			void apply_type1(const at::Tensor& in, at::Tensor& out)
			{
				if (_coords.get_device() != in.get_device() || _coords.get_device() != out.get_device())
					throw std::runtime_error("All tensors must reside on same device");

				if (!out.is_contiguous())
					throw std::runtime_error("out must be contiguous");

				if (!in.is_contiguous())
					throw std::runtime_error("in must be contiguous");

				if (out.sizes() != std::vector<i64>(_nmodes.begin(), _nmodes.end()))
					throw std::runtime_error("out.sizes() must equal nmodes given at construct");

				if (in.sizes() != c10::IntArrayRef{ _ntransf, _nfreq })
					throw std::runtime_error("in tensor must match ntransf in first dim and nfreq in second dim");

				switch (caffe2::typeMetaToScalarType(_coords.dtype())) {
				case c10::ScalarType::Float:
				{
					cuFloatComplex* c = in.data_ptr<cuFloatComplex>();
					cuFloatComplex* f = out.data_ptr<cuFloatComplex>();

					if (cufinufftf_execute(c, f, _planf))
						throw std::runtime_error("cufinufft_makeplan failed");
				}
				break;
				case c10::ScalarType::Double:
				{
					cuDoubleComplex* c = in.data_ptr<cuDoubleComplex>();
					cuDoubleComplex* f = out.data_ptr<cuDoubleComplex>();

					if (cufinufft_execute(c, f, _plan))
						throw std::runtime_error("cufinufft_makeplan failed");
				}
				break;
				default:
					throw std::runtime_error("Coordinates must be real!");
				}
			}

			void apply_type2(const at::Tensor& in, at::Tensor& out)
			{
				if (_coords.get_device() != in.get_device() || _coords.get_device() != out.get_device())
					throw std::runtime_error("All tensors must reside on same device");

				if (!out.is_contiguous())
					throw std::runtime_error("out must be contiguous");

				if (!in.is_contiguous())
					throw std::runtime_error("in must be contiguous");

				if (in.sizes() != std::vector<i64>(_nmodes.begin(), _nmodes.end()))
					throw std::runtime_error("in.sizes() must equal nmodes given at construct");

				if (out.sizes() != c10::IntArrayRef{ _ntransf, _nfreq })
					throw std::runtime_error("out tensor must match ntransf in first dim and nfreq in second dim");

				switch (caffe2::typeMetaToScalarType(_coords.dtype())) {
				case c10::ScalarType::Float:
				{
					cuFloatComplex* c = out.data_ptr<cuFloatComplex>();
					cuFloatComplex* f = in.data_ptr<cuFloatComplex>();

					if (cufinufftf_execute(c, f, _planf))
						throw std::runtime_error("cufinufft_makeplan failed");
				}
				break;
				case c10::ScalarType::Double:
				{
					cuDoubleComplex* c = out.data_ptr<cuDoubleComplex>();
					cuDoubleComplex* f = in.data_ptr<cuDoubleComplex>();

					if (cufinufft_execute(c, f, _plan))
						throw std::runtime_error("cufinufft_makeplan failed");
				}
				break;
				default:
					throw std::runtime_error("Coordinates must be real!");
				}
			}

			c10::ScalarType			_type;
			NufftType				_nufftType;
			i16f					_ndim;
			i16f					_ntransf;
			i32						_nfreq;
			at::Tensor				_coords;
			std::vector<i32>		_nmodes;
			std::array<i32, 3>		_nmodes_flipped;
			NufftOptions			_opts;

			cufinufft_plan			_plan;
			cufinufftf_plan			_planf;
		};

	}
}
*/


