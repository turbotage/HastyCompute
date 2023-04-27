

#include <torch/extension.h>
#include <torch/library.h>

#include <iostream>

using namespace at;


///Adds one to each element of a tensor
at::Tensor add_one(const at::Tensor& input) {
    auto output = torch::zeros_like(input);
    
    output = input + 1;

    return output;
}


///Note that we can have multiple implementations spread across multiple files, though there should only be one `def`
TORCH_LIBRARY(HastyPyInterface, m) {
    m.def("add_one", add_one);
}