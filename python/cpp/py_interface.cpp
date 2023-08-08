#include "py_interface.hpp"

#define FUNC_CASTER(x) static_cast<void(*)(x)>

void hasty::dummy::dummy(at::TensorList tensorlist)
{
	std::cout << tensorlist << std::endl;
	for (auto tensor : tensorlist) {
		tensor += 1.0;
	}
}

void hasty::dummy::stream_dummy(const at::optional<at::ArrayRef<at::Stream>>& streams, const torch::Tensor& in)
{
	if (streams.has_value()) {
		for (auto& stream : *streams) {
			std::cout << stream << std::endl;
		}
	}
	std::cout << in << std::endl;
}

TORCH_LIBRARY(HastyDummy, hd) {

	hd.def("dummy", hasty::dummy::dummy);

	hd.def("stream_dummy", hasty::dummy::stream_dummy);

}