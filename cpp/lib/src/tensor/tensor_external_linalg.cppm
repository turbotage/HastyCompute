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

// Batched matrix multiply — libtorch's matmul/bmm broadcast over leading
// batch dims natively; this is purely a re-export, no new math. Needed for
// butterfly factorization's per-level transfer-matrix application (apply
// the same small matrix to every same-level node's coefficients in one call
// instead of looping per node).
export Tensor matmul(const Tensor& A, const Tensor& B)
{
    return Tensor(hat::matmul(A.to_torch(), B.to_torch()));
}

export Tensor bmm(const Tensor& A, const Tensor& B)
{
    return Tensor(hat::bmm(A.to_torch(), B.to_torch()));
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

export struct QRResult {
    Tensor Q;
    Tensor R;
};

export QRResult linalg_qr(const Tensor& A, std::string_view mode = "reduced")
{
    auto [Q, R] = hat::linalg_qr(A.to_torch(), mode);
    return QRResult{Tensor{Q}, Tensor{R}};
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