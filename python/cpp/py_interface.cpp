#include "py_interface.hpp"

#define FUNC_CASTER(x) static_cast<void(*)(x)>


hasty::ffi::FunctionLambda::FunctionLambda(const std::string& script, const std::string& entry, at::TensorList captures)
	: _entry(entry), _cunit(torch::jit::compile(script)), _captures(captures.vec())
{
}

void hasty::ffi::FunctionLambda::apply(at::Tensor in) const
{

	auto ret = _cunit->run_method(_entry, in, _captures);
}

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



TORCH_LIBRARY(HastyInterface, hi) {
	
	hi.class_<hasty::ffi::FunctionLambda>("FunctionLambda")
		.def(torch::init<const std::string&, const std::string&, at::TensorList>())
		.def("apply", &hasty::ffi::FunctionLambda::apply);

	hi.def("doc", []() -> std::string {
		return
R"DOC(

// HastyInterface Module

class LIB_EXPORT FunctionLambda : public torch::CustomClassHolder {
public:

	FunctionLambda(const std::string& script, const std::string& entry, at::TensorList captures);

	void apply(at::Tensor in) const;

private:
	std::shared_ptr<at::CompilationUnit> _cunit;
	std::string _entry;
	std::vector<at::Tensor> _captures;
};

)DOC";
		});

}