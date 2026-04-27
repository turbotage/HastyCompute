import torch
from typing import Callable, Optional, Tuple



def lanczos_svd(
    matvec,
    rmatvec,
    m,
    n,
    L,
    k=-1,
    dtype=None,
    device=None,
    max_iter=None,
):
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
    eps = 1e-12 if real_dtype == torch.float64 else 1e-6

    def dot(a, b):
        return torch.dot(a.conj(), b) if is_complex else torch.dot(a, b)

    def norm(a):
        return torch.linalg.norm(a)

    def reorth(v, B, j):
        for _ in range(2):
            for i in range(j):
                v = v - dot(B[i], v) * B[i]
        return v

    def fresh(dim):
        v = torch.randn(dim, dtype=dtype, device=device)
        return v / norm(v)

    # --- restart vector (CRITICAL CHANGE) ---
    v = fresh(n)

    U_best = None
    S_best = None
    Vh_best = None

    for _restart in range(3):  # small fixed number, stable

        U_basis = torch.zeros(niters + 1, m, dtype=dtype, device=device)
        V_basis = torch.zeros(niters + 1, n, dtype=dtype, device=device)

        alpha = torch.zeros(niters, dtype=real_dtype, device=device)
        beta = torch.zeros(niters, dtype=real_dtype, device=device)

        V_basis[0] = v
        beta_prev = torch.zeros((), dtype=real_dtype, device=device)

        actual = 0

        # --- Lanczos bidiagonalization (UNCHANGED CORE) ---
        for j in range(niters):

            u = matvec(V_basis[j])
            if j > 0:
                u = u - beta_prev * U_basis[j - 1]

            u = reorth(u, U_basis, j)
            alpha_j = norm(u)

            if alpha_j < eps:
                u = fresh(m)
                alpha[j] = 0
            else:
                alpha[j] = alpha_j.real
                u = u / alpha_j

            U_basis[j] = u
            actual = j + 1

            v_next = rmatvec(u)
            v_next = v_next - alpha[j] * V_basis[j]
            v_next = reorth(v_next, V_basis, j + 1)

            beta_j = norm(v_next)

            if beta_j < eps:
                v_next = fresh(n)
                beta[j] = 0
            else:
                beta[j] = beta_j.real
                v_next = v_next / beta_j

            V_basis[j + 1] = v_next
            beta_prev = beta[j]

        # --- build bidiagonal ---
        B = torch.zeros(actual, actual, dtype=real_dtype, device=device)
        for j in range(actual):
            B[j, j] = alpha[j]
            if j > 0:
                B[j, j - 1] = beta[j - 1]

        Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False)

        # cast safely
        Ub = Ub.to(dtype)
        Vhb = Vhb.to(dtype)

        L_actual = min(L, actual)
        Sb = Sb[:L_actual]
        Ub = Ub[:, :L_actual]
        Vhb = Vhb[:L_actual]

        V_mat = V_basis[:actual]
        Vh = Vhb @ V_mat

        U = torch.empty(m, L_actual, dtype=dtype, device=device)

        for j in range(L_actual):
            s = Sb[j]
            if s > eps:
                U[:, j] = matvec(Vh[j]) / s
            else:
                U[:, j] = (U_basis[:actual].T @ Ub[:, j])

        # --- RESIDUAL CHECK (for restart decision) ---
        max_res = 0.0
        for j in range(L_actual):
            r = matvec(Vh[j]) - Sb[j] * U[:, j]
            max_res = max(max_res, norm(r).item())

        if U_best is None or max_res < 1e-6:
            U_best, S_best, Vh_best = U, Sb, Vh

        # ---------------------------
        # CRITICAL FIX: SAFE RESTART
        # ---------------------------
        # DO NOT inject full Vh block (this broke everything before)
        v = Vh[0].conj()
        v = v / norm(v)

    return U_best, S_best, Vh_best