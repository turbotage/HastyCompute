#include "py_interface.hpp"

#define FUNC_CASTER(x) static_cast<void(*)(x)>

/*
m.def("batched_sense",
    FUNC_CASTER(at::Tensor&, const std::vector<std::vector<int64_t>>&, const at::Tensor&,
        const std::vector<at::Tensor>&, const std::vector<at::Tensor>&)(&batched_sense));
m.def("batched_sense",
    FUNC_CASTER(at::Tensor&, const std::vector<std::vector<int64_t>>&, const at::Tensor&,
        const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&)(batched_sense));
*/

///Note that we can have multiple implementations spread across multiple files, though there should only be one `def`
TORCH_LIBRARY(HastyPyInterface, m) {
    m.def("nufft1", nufft1);
    m.def("nufft2", nufft2);
    m.def("nufft2to1", nufft2to1);
    m.def("batched_sense", batched_sense);
    m.def("batched_sense_weighted", batched_sense_weighted);
    m.def("batched_sense_toep", batched_sense_toep);
    m.def("random_blocks_svt", random_blocks_svt);
}

