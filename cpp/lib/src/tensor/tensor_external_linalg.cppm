module;

export module hasty_tensor_mod:external_linalg;

import std;
import hasty_util_mod;
import hasty_torch_wrapper;

import :tensor;

namespace hasty {

export Tensor linalg_solve(const Tensor& A, const Tensor& B)
{
    return Tensor(hat::linalg_solve(A.to_torch(), B.to_torch()));
}

}