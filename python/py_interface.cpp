#include "py_interface.hpp"

///Note that we can have multiple implementations spread across multiple files, though there should only be one `def`
TORCH_LIBRARY(HastyPyInterface, m) {
    m.def("add_one", add_one);
    m.def("nufft1", nufft1);
    m.def("nufft2", nufft2);
    m.def("nufft2to1", nufft2to1);
    m.def("llr", llr);
}

