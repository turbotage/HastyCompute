#include "torch_util.hpp"
#include <c10/cuda/CUDAGuard.h>

#include <memory>

std::vector<at::Stream> hasty::torch_util::get_streams(const at::optional<std::vector<at::Stream>>& streams)
{
	if (streams.has_value()) {
		return *streams;
	}
	else {
		return { at::cuda::getDefaultCUDAStream() };
	}
}

std::vector<at::Stream> hasty::torch_util::get_streams(const at::optional<at::ArrayRef<at::Stream>>& streams)
{
	if (streams.has_value()) {
		return (*streams).vec();
	}
	else {
		return { c10::cuda::getDefaultCUDAStream() };
	}
}

std::stringstream hasty::torch_util::print_4d_xyz(const at::Tensor& toprint)
{
	std::stringstream printer;
	printer.precision(2);
	printer << std::scientific;
	int tlen = toprint.size(0);
	int zlen = toprint.size(1);
	int ylen = toprint.size(2);
	int xlen = toprint.size(3);

	auto closer = [&printer](int iter, int length, bool brackets, int spacing)
	{
		if (brackets) {
			for (int i = 0; i < spacing; ++i)
				printer << " ";
			printer << "]";
		}
		if (iter + 1 != length)
			printer << ",";
		if (brackets)
			printer << "\n";
	};

	auto value_printer = [&printer]<typename T>(T val)
	{
		if (val < 0.0)
			printer << val;
		else
			printer << " " << val;
	};

	for (int t = 0; t < tlen; ++t) {
		printer << "[\n";
		for (int z = 0; z < zlen; ++z) {
			printer << " [\n";
			for (int y = 0; y < ylen; ++y) {
				printer << "  [";
				for (int x = 0; x < xlen; ++x) {
					switch (toprint.dtype().toScalarType()) {
					case c10::ScalarType::Float:
						value_printer(toprint.index({ t,z,y,x }).item<float>());
					break;
					case c10::ScalarType::Double:
						value_printer(toprint.index({ t,z,y,x }).item<float>());
					break;
					case c10::ScalarType::ComplexFloat:
					{
						float val;
						val = at::real(toprint.index({ t,z,y,x })).item<float>();
						printer << "("; value_printer(val); printer << ",";
						val = at::imag(toprint.index({ t,z,y,x })).item<float>();
						value_printer(val); printer << ")";
					}
					break;
					case c10::ScalarType::ComplexDouble:
					{
						double val;
						val = at::real(toprint.index({ t,z,y,x })).item<double>();
						printer << "("; value_printer(val); printer << ",";
						val = at::imag(toprint.index({ t,z,y,x })).item<double>();
						value_printer(val); printer << ")";
					}
					break;
					default:
						printer << toprint.index({ t,z,y,x });
					}

					closer(x, xlen, false, 0);
				}
				closer(y, ylen, true, 0);
			}
			closer(z, zlen, true, 1);
		}
		closer(t, tlen, true, 0);
	}

	return printer;
}

std::vector<int64_t> hasty::torch_util::nmodes_from_tensor(const at::Tensor& tensor)
{
	auto ret = tensor.sizes().vec();
	ret[0] = 1;
	return ret;
}

at::ScalarType hasty::torch_util::complex_type(at::ScalarType real_type, std::initializer_list<at::ScalarType> allowed_types)
{
	at::ScalarType complex_type;
	if (real_type == at::ScalarType::Float)
		complex_type = at::ScalarType::ComplexFloat;
	else if (real_type == at::ScalarType::Double)
		complex_type = at::ScalarType::ComplexDouble;
	else if (real_type == at::ScalarType::Half)
		complex_type = at::ScalarType::ComplexHalf;
	else
		throw std::runtime_error("Type not implemented complex_type()");

	if (allowed_types.size() != 0) {
		for (auto& atype : allowed_types) {
			if (complex_type == atype)
				return complex_type;
		}
		throw std::runtime_error("complex_type converted to non allowable type");
	}
	return complex_type;
}

at::ScalarType hasty::torch_util::real_type(at::ScalarType complex_type, std::initializer_list<at::ScalarType> allowed_types)
{
	at::ScalarType real_type;
	if (complex_type == at::ScalarType::ComplexFloat)
		real_type = at::ScalarType::Float;
	else if (complex_type == at::ScalarType::ComplexDouble)
		real_type = at::ScalarType::Double;
	else if (complex_type == at::ScalarType::ComplexHalf)
		real_type = at::ScalarType::Half;
	else
		throw std::runtime_error("Type not implemented complex_type()");

	if (allowed_types.size() != 0) {
		for (auto& atype : allowed_types) {
			if (real_type == atype)
				return real_type;
		}
		throw std::runtime_error("complex_type converted to non allowable type");
	}
	return real_type;
}

void hasty::torch_util::future_catcher(std::future<void>& fut)
{
	try {
		fut.get();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (std::exception& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (...) {
		std::cerr << "caught something strange: " << std::endl;
		throw std::runtime_error("caught something strange: ");
	}
}

void hasty::torch_util::future_catcher(const std::function<void()>& func)
{
	try {
		func();
	}
	catch (c10::Error& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (std::exception& e) {
		std::string err = e.what();
		std::cerr << err << std::endl;
		throw std::runtime_error(err);
	}
	catch (...) {
		std::cerr << "caught something strange: " << std::endl;
		throw std::runtime_error("caught something strange: ");
	}
}