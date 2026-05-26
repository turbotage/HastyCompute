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

export struct SVDResult {
    Tensor U;
    Tensor S;
    Tensor Vh;
};

export SVDResult linalg_svd(const Tensor& A, bool full_matrices = false)
{
    auto [U, S, Vh] = hat::linalg_svd(A.to_torch(), full_matrices);
    return SVDResult{Tensor{U}, Tensor{S}, Tensor{Vh}};
}

export struct EigResult {
    Tensor eigenvalues;
    Tensor eigenvectors;
};

export EigResult linalg_eig(const Tensor& A)
{
    auto [eigenvalues, eigenvectors] = hat::linalg_eig(A.to_torch());
    return EigResult{Tensor{eigenvalues}, Tensor{eigenvectors}};
}

export EigResult linalg_eigh(const Tensor& A)
{
    auto [eigenvalues, eigenvectors] = hat::linalg_eigh(A.to_torch());
    return EigResult{Tensor{eigenvalues}, Tensor{eigenvectors}};
}


}