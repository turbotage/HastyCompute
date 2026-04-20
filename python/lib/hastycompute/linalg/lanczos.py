import torch
from typing import Callable, Optional, Tuple


def lanczos_svd(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    m: int,
    n: int,
    L: int,
    k: int = -1,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    max_iter: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a rank-L SVD approximation via Golub-Kahan bidiagonalization
    with full reorthogonalization (GKBSVD).

    Args:
        matvec:  x -> A @ x,  x shape (n,), output shape (m,)
        rmatvec: y -> A^H @ y, y shape (m,), output shape (n,)
        m: number of rows of A
        n: number of columns of A
        L: number of singular triplets to compute
        k: size of Krylov subspace beyond L. -1 => auto (2*L)
        dtype: floating-point dtype (default: torch.float64)
        device: compute device (default: cpu)
        max_iter: maximum bidiag steps (default: L + k)

    Returns:
        U:  (m, L) left singular vectors
        S:  (L,)   singular values, descending
        Vh: (L, n) right singular vectors (conjugate-transposed rows)
    """
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")
    if k < 0:
        k = 2 * L
    niters = L + k
    if max_iter is not None:
        niters = min(niters, max_iter)

    is_complex = dtype in (torch.complex64, torch.complex128)
    real_dtype = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    _eps = 1e-7 if real_dtype == torch.float32 else 1e-14

    def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if is_complex:
            return torch.dot(a.conj(), b)
        return torch.dot(a, b)

    def _norm(a: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(a)

    def _reorth(vec: torch.Tensor, basis: torch.Tensor, count: int) -> torch.Tensor:
        """Double-pass modified Gram-Schmidt against basis[0:count]."""
        for _ in range(2):
            for i in range(count):
                vec = vec - _dot(basis[i], vec) * basis[i]
        return vec

    def _fresh_orth(space_dim: int, basis: torch.Tensor, count: int,
                    rand_dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return a unit vector orthogonal to basis[0:count], or None if exhausted."""
        vec = torch.randn(space_dim, dtype=rand_dtype, device=device)
        vec = _reorth(vec, basis, count)
        nrm = _norm(vec)
        if nrm < _eps:
            return None
        return vec / nrm

    # Storage: rows are basis vectors
    U_basis = torch.zeros(niters + 1, m, dtype=dtype, device=device)
    V_basis = torch.zeros(niters + 1, n, dtype=dtype, device=device)

    # Bidiagonal scalars (real even for complex A)
    alpha = torch.zeros(niters, dtype=real_dtype, device=device)
    beta  = torch.zeros(niters, dtype=real_dtype, device=device)

    # Initialise with a random unit n-vector
    v = torch.randn(n, dtype=dtype, device=device)
    v = v / _norm(v)
    V_basis[0] = v

    beta_prev = torch.zeros((), dtype=real_dtype, device=device)
    actual = 0  # tracks how many steps complete successfully

    for j in range(niters):
        # ---- left step: u = A v_j - beta_{j-1} u_{j-1} ----
        u = matvec(V_basis[j])
        if j > 0:
            u = u - beta_prev * U_basis[j - 1]

        u = _reorth(u, U_basis, j)
        alpha_j = _norm(u)

        if alpha_j < _eps:
            # A maps V_basis[j] into span(U_basis[0:j]) — null-space hit.
            # Fill with a fresh direction so the basis stays orthonormal.
            fresh = _fresh_orth(m, U_basis, j, dtype)
            if fresh is None:
                # R^m is exhausted; cannot extend further.
                break
            u = fresh
            alpha[j] = torch.zeros((), dtype=real_dtype, device=device)
        else:
            alpha[j] = alpha_j.real if is_complex else alpha_j
            u = u / alpha_j

        U_basis[j] = u
        actual = j + 1

        # ---- right step: v = A^H u_j - alpha_j v_j ----
        v = rmatvec(u)
        v = v - alpha[j] * V_basis[j]

        v = _reorth(v, V_basis, j + 1)
        beta_j = _norm(v)

        if beta_j < _eps:
            # A^H maps U_basis[j] into span(V_basis[0:j+1]) — invariant subspace.
            fresh = _fresh_orth(n, V_basis, j + 1, dtype)
            if fresh is None:
                break
            v = fresh
            beta[j] = torch.zeros((), dtype=real_dtype, device=device)
        else:
            beta[j] = beta_j.real if is_complex else beta_j
            v = v / beta_j

        V_basis[j + 1] = v
        beta_prev = beta[j]

    # Build lower-bidiagonal matrix B (actual x actual)
    B = torch.zeros(actual, actual, dtype=real_dtype, device=device)
    for j in range(actual):
        B[j, j] = alpha[j]
        if j > 0:
            B[j, j - 1] = beta[j - 1]

    # SVD of the small bidiagonal matrix
    Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False)

    L_actual = min(L, actual)
    Ub  = Ub[:, :L_actual]    # (actual, L)
    Sb  = Sb[:L_actual]        # (L,)
    Vhb = Vhb[:L_actual, :]   # (L, actual)

    # Lift back to full-dimensional space
    U_mat = U_basis[:actual]   # (actual, m)
    V_mat = V_basis[:actual]   # (actual, n)

    U_full  = U_mat.T @ Ub.to(dtype)   # (m, L)
    Vh_full = Vhb.to(dtype) @ V_mat    # (L, n)

    return U_full, Sb, Vh_full
