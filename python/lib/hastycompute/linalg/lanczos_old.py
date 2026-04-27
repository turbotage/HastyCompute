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
    Rank-L SVD approximation via Golub-Kahan bidiagonalization with full
    double-pass reorthogonalization.

    Args:
        matvec:  x -> A @ x,   x shape (n,), output shape (m,)
        rmatvec: y -> A^H @ y, y shape (m,), output shape (n,)
        m: number of rows of A
        n: number of columns of A
        L: number of singular triplets to compute  (must be < min(m,n))
        k: Krylov oversampling beyond L.  -1 => auto (2*L)
        dtype: floating-point dtype (default: torch.float64)
        device: compute device (default: cpu)
        max_iter: hard cap on bidiagonalization steps (default: L + k)

    Returns:
        U:  (m, L) left  singular vectors (orthonormal columns)
        S:  (L,)   singular values, descending
        Vh: (L, n) right singular vectors (orthonormal rows)
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

    # When m < n, run GK on A^H instead of A.  This way the small dimension
    # (m) exhausts on the V-side (beta restart), which is numerically clean,
    # rather than the U-side (alpha ≈ 0), which corrupts the bidiagonal.
    transposed = m < n
    if transposed:
        _mv, _rmv = rmatvec, matvec   # A^H: R^m -> R^n
        _nrows, _ncols = n, m         # working dims of A^H
    else:
        _mv, _rmv = matvec, rmatvec   # A:   R^n -> R^m
        _nrows, _ncols = m, n

    def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.dot(a.conj(), b) if is_complex else torch.dot(a, b)

    def _norm(a: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(a)

    def _reorth(vec: torch.Tensor, basis: torch.Tensor, count: int) -> torch.Tensor:
        for _ in range(2):
            for i in range(count):
                vec = vec - _dot(basis[i], vec) * basis[i]
        return vec

    def _fresh(dim: int, basis: torch.Tensor, count: int) -> Optional[torch.Tensor]:
        vec = torch.randn(dim, dtype=dtype, device=device)
        vec = _reorth(vec, basis, count)
        nrm = _norm(vec)
        return (vec / nrm) if nrm > _eps else None

    # Basis storage: rows are vectors
    U_basis = torch.zeros(niters + 1, _nrows, dtype=dtype, device=device)
    V_basis = torch.zeros(niters + 1, _ncols, dtype=dtype, device=device)
    alpha   = torch.zeros(niters, dtype=real_dtype, device=device)
    beta    = torch.zeros(niters, dtype=real_dtype, device=device)

    v = torch.randn(_ncols, dtype=dtype, device=device)
    v = v / _norm(v)
    V_basis[0] = v
    beta_prev  = torch.zeros((), dtype=real_dtype, device=device)
    actual     = 0

    for j in range(niters):
        # ---- left step: u = A v_j - beta_{j-1} u_{j-1} ----
        u = _mv(V_basis[j])
        if j > 0:
            u = u - beta_prev * U_basis[j - 1]
        u = _reorth(u, U_basis, j)
        alpha_j = _norm(u)

        if alpha_j < _eps:
            fresh = _fresh(_nrows, U_basis, j)
            if fresh is None:
                break
            u = fresh
            alpha[j] = torch.zeros((), dtype=real_dtype, device=device)
        else:
            alpha[j] = alpha_j.real if is_complex else alpha_j
            u = u / alpha_j

        U_basis[j] = u
        actual = j + 1

        # ---- right step: v = A^H u_j - alpha_j v_j ----
        v = _rmv(u)
        v = v - alpha[j] * V_basis[j]
        v = _reorth(v, V_basis, j + 1)
        beta_j = _norm(v)

        if beta_j < _eps:
            fresh = _fresh(_ncols, V_basis, j + 1)
            if fresh is None:
                break
            v = fresh
            beta[j] = torch.zeros((), dtype=real_dtype, device=device)
        else:
            beta[j] = beta_j.real if is_complex else beta_j
            v = v / beta_j

        V_basis[j + 1] = v
        beta_prev = beta[j]

    # ---- Build lower-bidiagonal B and compute its SVD ----
    B = torch.zeros(actual, actual, dtype=real_dtype, device=device)
    for j in range(actual):
        B[j, j] = alpha[j]
        if j > 0:
            B[j, j - 1] = beta[j - 1]

    Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False)

    L_actual = min(L, actual)
    Sb  = Sb[:L_actual]
    Vhb = Vhb[:L_actual, :]        # (L, actual)
    V_mat = V_basis[:actual]        # (actual, _ncols)

    # Right Ritz vectors of the working operator (accurate from V Krylov subspace)
    Vh_work = Vhb.to(dtype) @ V_mat   # (L, _ncols)

    # Left vectors: A @ v_j / s_j  (avoids ill-conditioned U lift-back)
    U_work = torch.empty(_nrows, L_actual, dtype=dtype, device=device)
    for j in range(L_actual):
        s_j = Sb[j]
        if s_j.abs() > _eps:
            U_work[:, j] = _mv(Vh_work[j]) / s_j
        else:
            U_work[:, j] = (U_basis[:actual].T @ Ub[:, j].to(dtype))

    # ---- Map back to the original (non-transposed) SVD convention ----
    if transposed:
        # GK ran on A^H, so:
        #   V Krylov (V_basis, _ncols=m): right Krylov of A^H → right Ritz of A^H ≈ u_j
        #   U Krylov (U_basis, _nrows=n): left  Krylov of A^H → left  Ritz of A^H ≈ v_j
        #
        # Right Ritz of A are built from U Krylov × Ub (no division by s_j).
        # Then u_j = A v_j / s_j — one division, left residual = 0 by construction.
        # This is symmetric with the non-transposed case (Vh from V Krylov, U = A v / s).
        U_mat   = U_basis[:actual]                                      # (actual, n)
        V_right = (U_mat.T @ Ub[:, :L_actual].to(dtype))               # (n, L): v_j cols

        Vh_out = V_right.conj().T                                       # (L, n): rows = v_j^H

        U_out = torch.empty(_ncols, L_actual, dtype=dtype, device=device)  # (m, L)
        for j in range(L_actual):
            s_j = Sb[j]
            if s_j.abs() > _eps:
                U_out[:, j] = _rmv(V_right[:, j]) / s_j   # A v_j / s_j ≈ u_j
            else:
                U_out[:, j] = Vh_work[j].conj()
    else:
        U_out  = U_work                                    # (m, L)
        Vh_out = Vh_work.conj()                            # (L, n), rows = v_j^H

    return U_out, Sb, Vh_out
