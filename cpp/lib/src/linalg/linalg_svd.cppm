module;

export module hasty_linalg_mod:svd;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

import :op;

namespace hasty {
namespace linalg {

// ─── SVD result ──────────────────────────────────────────────────────────────

export struct SVDResult {
    Tensor U;   // (m, k)  left  singular vectors
    Tensor S;   // (k,)    singular values descending (real dtype)
    Tensor Vh;  // (k, n)  right singular vectors, Hermitian-transposed
};


// ─── operator_svd ────────────────────────────────────────────────────────────

// Compute the truncated SVD of a LinearOperator via SLEPc thick-restart Lanczos.
// Returns U (m,k), S (k,), Vh (k,n) in the same dtype as op.dtype().
// ncv, mpd: Krylov basis / projected dimension (-1 = let SLEPc choose).
// tol: convergence tolerance (-1 = SLEPc default ~1e-8).
// max_its: max Lanczos iterations (-1 = SLEPc default).
export SVDResult operator_svd(
    const LinearOperator& op,
    i64 k,
    i64 ncv     = -1,
    i64 mpd     = -1,
    double tol  = -1.0,
    i64 max_its = -1);



}
}