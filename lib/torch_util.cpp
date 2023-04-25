#include "torch_util.hpp"

#ifdef STL_AS_MODULES
import std.compat;
#else
import <memory>;
import <vector>;
import <string>;
import <array>;
import <stdexcept>;
import <optional>;
import <sstream>;
#endif

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
