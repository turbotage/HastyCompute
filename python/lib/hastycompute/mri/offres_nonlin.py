"""
offres_nonlin.py — Off-resonance and nonlinear-gradient phase approximation for MRI.

Signal model
------------
    S(t) = ∫ ρ(r) exp(Φ(r,t)) exp(-ik(t)·r) dr

    k(t)       = γ ∫₀ᵗ G⁰(τ) dτ                             (k-space trajectory)
    Φ(r, t)    = -z(r)·t  -  iγ Σ_q f_q(r) K_q(t)           (accumulated phase)
    z(r)       = 1/T₂(r) + iγ ΔB₀(r)                        (complex rate map)
    K_q(t)     = ∫₀ᵗ G_q(τ) dτ                               (integrated nonlin waveform)
    G_q(t)     = (α_q * G^ideal_q)(t)                        (convolved gradient)

The spatial functions f_q(r) can be any polynomial or harmonic basis evaluated at voxel
positions — solid harmonics, monomials, Zernike polynomials, etc.

Normal-operator approximation
------------------------------
The two-term phase products needed for A^H A are approximated as:

    exp(Φ(r_j,t_k) + Φ*(r_i,t_k)) ≈ Σ_{p,r} Λ_{k,p} Θ_L_{j,r,p} Θ_R*_{i,r,p}

where:

    exp(Φ) ≈ Σ_l Ω_{k,l} Υ_{j,l}                             [eq. 1 – rank-L SVD]
    Ω_{k,l1} Ω*_{k,l2} ≈ Σ_p C_{l1,l2,p} Λ_{k,p}            [eq. 2 – rank-P SVD over time]
    C_{l1,l2,p} = Σ_r Ĉ_L_{l1,r,p} Ĉ_R*_{l2,r,p}           [eq. 3 – two-factor eig per p]
    Θ_L/R_{j,r,p} = Σ_l Ĉ_L/R_{l,r,p} Υ_{j,l}              [eq. 4 – spatial combination]

Public API
----------
    convolve_waveform        – single 1-D causal convolution
    convolve_gradients       – convolve all gradient orders with ideal waveforms
    kspace_trajectory        – integrate linear waveforms to k(t)
    make_phi_matvec          – build chunked forward/adjoint operators for exp(Φ)
    randomized_svd           – proper Halko et al. randomized SVD via matvecs
    phi_lowrank              – randomized SVD of exp(Φ) → Ω, Υ
    normal_op_lowrank        – Ω, Υ → Λ, Θ_L, Θ_R
    offres_nonlin_approx     – full pipeline returning all approximants
"""

from __future__ import annotations
import math
from typing import Callable
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient waveform convolution
# ---------------------------------------------------------------------------

def convolve_waveform(
    kernel: torch.Tensor,
    waveform: torch.Tensor,
) -> torch.Tensor:
    """
    Causal 1-D convolution, output trimmed to the length of `waveform`.

    Args:
        kernel:   (K_kern,) float Tensor — impulse response α(t).
        waveform: (K,)      float Tensor — ideal gradient G^ideal(t).

    Returns:
        (K,) convolved waveform.
    """
    K = waveform.shape[0]
    pad = kernel.shape[0] - 1
    x = waveform.unsqueeze(0).unsqueeze(0)           # (1, 1, K)
    h = kernel.flip(0).unsqueeze(0).unsqueeze(0)     # (1, 1, K_kern)
    out = F.conv1d(x.float(), h.float(), padding=pad)
    return out[0, 0, :K].to(waveform.dtype)


def convolve_gradients(
    kernels: dict[tuple, torch.Tensor],
    ideal_waveforms: dict[str, torch.Tensor],
) -> dict[tuple, torch.Tensor]:
    """
    Convolve every ideal gradient axis waveform with its per-order kernel.

    Args:
        kernels:
            Mapping  key → (K_kern,) Tensor.  The last element of each key must be
            the axis label ('x', 'y', or 'z') used to look up `ideal_waveforms`.
            E.g. (0, 0, 'x') for the linear x-gradient, (1, 0, 'x') for a nonlinear
            x-gradient term, or any other hashable key whose last element is an axis.
        ideal_waveforms:
            {'x': (K,), 'y': (K,), 'z': (K,)} — G^ideal_β(t) per axis.

    Returns:
        Dict with the same keys as `kernels`; values are (K,) convolved waveforms.
    """
    return {
        key: convolve_waveform(kernel, ideal_waveforms[key[-1]])
        for key, kernel in kernels.items()
    }


# ---------------------------------------------------------------------------
# k-space trajectory
# ---------------------------------------------------------------------------

def kspace_trajectory(
    linear_waveforms: dict[str, torch.Tensor],
    gamma: float,
    dt: float,
) -> torch.Tensor:
    """
    Compute the k-space trajectory by cumulative integration of the linear
    gradient waveforms:   k(t) = γ ∫₀ᵗ G⁰(τ) dτ.

    Args:
        linear_waveforms:
            {'x': (K,), 'y': (K,), 'z': (K,)} — already-convolved linear
            gradient waveforms G⁰_β(t).
        gamma: Gyromagnetic ratio (rad s⁻¹ T⁻¹).
        dt:    Sampling interval (s).

    Returns:
        (K, 3) float32 Tensor — k-space coordinates [kx, ky, kz].
    """
    return gamma * dt * torch.stack(
        [torch.cumsum(linear_waveforms[b].float(), dim=0) for b in ('x', 'y', 'z')],
        dim=1,
    )


# ---------------------------------------------------------------------------
# Chunked forward/adjoint operators for exp(Φ)
# ---------------------------------------------------------------------------

def make_phi_matvec(
    z_map: torch.Tensor,
    spatial_maps: torch.Tensor,
    nonlin_waveforms: dict,
    waveform_order: list,
    t: torch.Tensor,
    gamma: float,
    dt: float,
    chunk_size: int = 512,
) -> tuple[Callable, Callable]:
    """
    Build forward and adjoint matrix-vector product operators for exp(Φ(r, t))
    without ever forming the full (K, N) matrix.

        fwd(x)  computes  exp(Φ) @ x,   x: (N, n) → (K, n)
        adj(y)  computes  exp(Φ)^H @ y, y: (K, n) → (N, n)

    The K axis is processed in chunks of `chunk_size` to control memory use.

    Args:
        z_map:
            (N,) complex Tensor — rate map 1/T₂(r) + iγ ΔB₀(r).
        spatial_maps:
            (N, Q) real Tensor — spatial basis functions f_q(r) evaluated at each
            voxel.  Can be solid harmonics, monomials, Zernike polynomials, etc.
            Column q corresponds to `waveform_order[q]`.
        nonlin_waveforms:
            Mapping  key → (K,) Tensor — already-convolved nonlinear gradient
            waveforms G_q(t).  Keys must appear in `waveform_order`.
        waveform_order:
            List of keys of length Q.  Entry q identifies which waveform in
            `nonlin_waveforms` pairs with column q of `spatial_maps`.
        t:
            (K,) float Tensor — time points (s).
        gamma:
            Gyromagnetic ratio (rad s⁻¹ T⁻¹).
        dt:
            Sampling interval (s).
        chunk_size:
            Number of time points processed per chunk (controls peak memory).

    Returns:
        fwd, adj — callable operators as described above.
    """
    device = z_map.device
    K = t.shape[0]
    N = z_map.shape[0]
    cdtype = torch.complex64

    z_c = z_map.to(cdtype).to(device)
    t_c = t.to(cdtype).to(device)

    # Precompute K_q(t) = iγ ∫₀ᵗ G_q(τ) dτ  for each nonlinear order, shape (Q, K)
    Q = len(waveform_order)
    if Q > 0:
        K_q = torch.stack(
            [
                1j * gamma * dt
                * torch.cumsum(nonlin_waveforms[key].to(cdtype).to(device), dim=0)
                for key in waveform_order
            ],
            dim=0,
        )  # (Q, K)
        sm_c = spatial_maps.to(cdtype).to(device)   # (N, Q)
    else:
        K_q = None
        sm_c = None

    def _exp_phi_chunk(k_start: int, k_end: int) -> torch.Tensor:
        """Compute exp(Φ[k_start:k_end, :]) → (chunk, N)."""
        t_chunk = t_c[k_start:k_end]                            # (chunk,)
        phi = -(t_chunk.unsqueeze(1) * z_c.unsqueeze(0))       # (chunk, N)
        if Q > 0:
            # (N, Q) @ (Q, chunk) → (N, chunk) → .T → (chunk, N)
            phi -= (sm_c @ K_q[:, k_start:k_end]).T
        return torch.exp(phi)                                    # (chunk, N)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        """Apply exp(Φ): (N, n) → (K, n)."""
        squeeze = x.dim() == 1
        x_c = x.to(cdtype).unsqueeze(1) if squeeze else x.to(cdtype)
        n_cols = x_c.shape[1]
        result = torch.zeros(K, n_cols, dtype=cdtype, device=device)
        for k0 in range(0, K, chunk_size):
            k1 = min(k0 + chunk_size, K)
            E = _exp_phi_chunk(k0, k1)                          # (chunk, N)
            result[k0:k1] = E @ x_c                             # (chunk, n)
        return result.squeeze(1) if squeeze else result

    def adj(y: torch.Tensor) -> torch.Tensor:
        """Apply exp(Φ)^H: (K, n) → (N, n)."""
        squeeze = y.dim() == 1
        y_c = y.to(cdtype).unsqueeze(1) if squeeze else y.to(cdtype)
        n_cols = y_c.shape[1]
        result = torch.zeros(N, n_cols, dtype=cdtype, device=device)
        for k0 in range(0, K, chunk_size):
            k1 = min(k0 + chunk_size, K)
            E = _exp_phi_chunk(k0, k1)                          # (chunk, N)
            result += E.conj().mT @ y_c[k0:k1]                 # (N, n)
        return result.squeeze(1) if squeeze else result

    return fwd, adj


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------

def randomized_svd(
    fwd: Callable,
    adj: Callable,
    N: int,
    K: int,
    rank: int,
    n_oversampling: int = 10,
    n_power_iter: int = 2,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD (Halko, Martinsson & Tropp, 2011, Algorithm 4.4 with subspace
    iteration) using only matrix-vector products.

    Finds the rank-`rank` approximation  A ≈ U diag(S) Vh  where:
        U  : (K, rank) — left singular vectors
        S  : (rank,)   — singular values (descending)
        Vh : (rank, N) — right singular vectors (conjugate-transposed)

    Args:
        fwd:  (N, n) → (K, n) — applies A.
        adj:  (K, n) → (N, n) — applies A^H.
        N, K: dimensions of the implicit matrix A.
        rank: target rank of the approximation.
        n_oversampling:
            Extra sketch columns beyond `rank` (default 10).  Larger values
            improve accuracy at the cost of more matvec calls.
        n_power_iter:
            Number of subspace power iterations (default 2).  Each iteration
            applies A A^H once more; significantly improves accuracy when singular
            values decay slowly.
        dtype, device:
            Dtype and device for the random test matrix.

    Returns:
        U  (K, rank), S (rank,), Vh (rank, N).
    """
    n = rank + n_oversampling

    # Complex Gaussian random test matrix
    Omega = torch.randn(N, n, dtype=torch.float32, device=device)
    if dtype.is_complex:
        Omega = (Omega + 1j * torch.randn(N, n, dtype=torch.float32, device=device)) \
                / math.sqrt(2)
    Omega = Omega.to(dtype)

    # Subspace iteration with alternating QR orthogonalization for numerical stability.
    # Each iteration applies (A A^H), which raises singular values to the power 2^q,
    # sharpening the gap and improving the range approximation.
    for _ in range(n_power_iter):
        Y = fwd(Omega)                          # (K, n) = A Omega
        Omega, _ = torch.linalg.qr(Y)          # (K, n) — orthonormal basis for range
        Z = adj(Omega)                          # (N, n) = A^H Q_K
        Omega, _ = torch.linalg.qr(Z)          # (N, n) — orthonormal basis for co-range

    # Final range approximation
    Y = fwd(Omega)                              # (K, n) = A Omega
    Q, _ = torch.linalg.qr(Y)                  # (K, n)

    # Project: B = Q^H A, computed as (A^H Q)^H
    B = adj(Q).mH                               # (n, N) = Q^H A

    # Small dense SVD
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)   # (n,n), (n,), (n,N)

    U  = Q @ Ub[:, :rank]   # (K, rank)
    S  = S[:rank]            # (rank,)
    Vh = Vh[:rank, :]        # (rank, N)

    return U, S, Vh


# ---------------------------------------------------------------------------
# Low-rank decomposition of exp(Φ)
# ---------------------------------------------------------------------------

def phi_lowrank(
    fwd: Callable,
    adj: Callable,
    N: int,
    K: int,
    L: int,
    n_oversampling: int = 10,
    n_power_iter: int = 2,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rank-L decomposition of exp(Φ(r, t)) via randomized SVD:

        exp(Φ) ≈ Ω Υᵀ     with   Ω: (K, L),  Υ: (N, L)

    so that  exp(Φ[k, j]) ≈ Σ_l Ω[k, l] · Υ[j, l]  (no conjugation on Υ).

    Singular values are split symmetrically:
        Ω[k, l] = U[k, l] · √S[l]
        Υ[j, l] = Vh[l, j] · √S[l]   (Vh, not Vh^H)

    Args:
        fwd:  (N, n) → (K, n) — applies exp(Φ); from `make_phi_matvec`.
        adj:  (K, n) → (N, n) — applies exp(Φ)^H; from `make_phi_matvec`.
        N:    Number of voxels.
        K:    Number of time points.
        L:    Target rank.
        n_oversampling: Extra sketch columns for `randomized_svd`.
        n_power_iter:   Power iterations for `randomized_svd`.
        dtype, device:  For the random test matrix.

    Returns:
        Omega:   (K, L) complex — temporal basis Ω_{k,l}.
        Upsilon: (N, L) complex — spatial basis Υ_{j,l}.
    """
    U, S, Vh = randomized_svd(
        fwd, adj, N, K, rank=L,
        n_oversampling=n_oversampling,
        n_power_iter=n_power_iter,
        dtype=dtype, device=device,
    )

    S_sqrt  = S.sqrt()
    Omega   = U   * S_sqrt.unsqueeze(0)    # (K, L)
    Upsilon = Vh.T * S_sqrt.unsqueeze(0)   # (N, L),  Υ[j,l] = Vh[l,j] · √S[l]

    return Omega, Upsilon


# ---------------------------------------------------------------------------
# Normal-operator low-rank decomposition
# ---------------------------------------------------------------------------

def normal_op_lowrank(
    Omega: torch.Tensor,
    Upsilon: torch.Tensor,
    P: int,
    R: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the compressed normal-operator approximants Λ_{k,p}, Θ_L_{j,r,p},
    and Θ_R_{j,r,p}.

    Given Ω (K, L) and Υ (N, L) from `phi_lowrank`, the steps are:

    **Step 1 — P-rank time compression.**
    Extract the L(L+1)/2 unique upper-triangle products Ω_{kl1} Ω*_{kl2} for each k,
    forming a complex (L(L+1)/2) × K matrix.  Split into a real augmented matrix
    [Re(upper); Im(off-diagonal)] of shape (L², K) so that the SVD gives real Λ
    (required for Hermitian consistency between upper and lower triangle):

        Ω_{kl1} Ω*_{kl2} ≈ Σ_p C_{l1,l2,p} Λ_{k,p}

    **Step 2 — two-factor eigendecomposition per p.**
    Reconstruct each Hermitian L×L matrix C_p from the SVD left singular vector.
    A sign flip of both C_p and Λ[:,p] (leaves the product unchanged) orients C_p
    to have positive trace.  Keep the R eigenvectors with largest |λ_r| and form
    two factor maps:

        Ĉ_L[l, r, p] = √|λ_r| · v_r[l]
        Ĉ_R[l, r, p] = √|λ_r| · sign(λ_r) · v_r[l]

    so that  Σ_r Ĉ_L[l1,r] conj(Ĉ_R[l2,r]) = Σ_r λ_r v_r[l1] v*_r[l2] = C_p[l1,l2]
    exactly for all eigenvalues (positive and negative).

    **Step 3 — spatial combination.**
        Θ_L[j, r, p] = Σ_l Ĉ_L[l, r, p] · Υ[j, l]
        Θ_R[j, r, p] = Σ_l Ĉ_R[l, r, p] · Υ[j, l]

    The normal-operator approximation is then:
        exp(Φ_j + Φ*_i) ≈ Σ_{p,r} Λ_{k,p} Θ_L[j,r,p] conj(Θ_R[i,r,p])

    Args:
        Omega:   (K, L) complex Tensor — temporal basis from `phi_lowrank`.
        Upsilon: (N, L) complex Tensor — spatial basis from `phi_lowrank`.
        P:       Rank for time compression of the normal operator.
        R:       Rank for eigendecomposition of each C_p (≤ L).

    Returns:
        Lambda:  (K, P) complex Tensor — Λ_{k,p}  (real values stored as complex).
        Theta_L: (N, R, P) complex Tensor — Θ_L_{j,r,p}.
        Theta_R: (N, R, P) complex Tensor — Θ_R_{j,r,p}.
    """
    _, L = Omega.shape
    device = Omega.device
    cdtype = Omega.dtype

    # ----- Step 1: real SVD of Hermitian-product matrix -----
    triu_i, triu_j = torch.triu_indices(L, L, offset=0, device=device)
    off_mask = triu_i < triu_j                       # off-diagonal upper-triangle

    M_complex = (Omega[:, triu_i] * Omega[:, triu_j].conj()).T   # (n_triu, K)
    M_re  = M_complex.real                           # (n_triu, K)
    M_im  = M_complex[off_mask].imag                 # (n_off, K)
    M_aug = torch.cat([M_re, M_im], dim=0)           # (L², K) real matrix

    P = min(P, min(M_aug.shape))
    U_aug, S, Vh = torch.linalg.svd(M_aug, full_matrices=False)
    Lambda = (S[:P].unsqueeze(1) * Vh[:P, :]).T.to(cdtype)       # (K, P) real→complex

    # ----- Step 2: two-factor eigendecomposition per p -----
    R = min(R, L)
    C_hat_L = torch.zeros(L, R, P, dtype=cdtype, device=device)
    C_hat_R = torch.zeros(L, R, P, dtype=cdtype, device=device)

    n_triu  = triu_i.shape[0]
    off_idx = torch.where(off_mask)[0]

    for p in range(P):
        # Reconstruct Hermitian C_p from the p-th left singular vector of M_aug.
        Cp = torch.zeros(L, L, dtype=cdtype, device=device)
        Cp[triu_i, triu_j]              = U_aug[:n_triu, p].to(cdtype)
        Cp[triu_i[off_idx], triu_j[off_idx]] += 1j * U_aug[n_triu:, p].to(cdtype)
        Cp = Cp + Cp.conj().T - torch.diag(Cp.diagonal())

        # Sign fix: orient so trace is positive (leaves Λ[:,p] · C_p unchanged).
        if Cp.real.trace() < 0:
            Cp = -Cp
            Lambda[:, p] = -Lambda[:, p]

        eigvals, eigvecs = torch.linalg.eigh(Cp)              # ascending, real eigvals
        top_idx = torch.argsort(eigvals.abs(), descending=True)[:R]
        R_eff   = top_idx.shape[0]

        lam_r  = eigvals[top_idx[:R_eff]]                     # (R_eff,) real
        vec_r  = eigvecs[:, top_idx[:R_eff]]                  # (L, R_eff)

        scale_abs  = lam_r.abs().sqrt().to(cdtype)
        scale_sign = torch.sign(lam_r).to(cdtype)

        C_hat_L[:, :R_eff, p] = vec_r * scale_abs.unsqueeze(0)
        C_hat_R[:, :R_eff, p] = vec_r * (scale_abs * scale_sign).unsqueeze(0)

    # ----- Step 3: spatial combination -----
    Theta_L = torch.einsum('nl,lrp->nrp', Upsilon, C_hat_L)    # (N, R, P)
    Theta_R = torch.einsum('nl,lrp->nrp', Upsilon, C_hat_R)    # (N, R, P)

    return Lambda, Theta_L, Theta_R


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def offres_nonlin_approx(
    kernels: dict[tuple, torch.Tensor],
    ideal_waveforms: dict[str, torch.Tensor],
    z_map: torch.Tensor,
    spatial_maps: torch.Tensor,
    waveform_order: list,
    t: torch.Tensor,
    gamma: float,
    dt: float,
    L: int,
    P: int,
    R: int,
    n_oversampling: int = 10,
    n_power_iter: int = 2,
    chunk_size: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Full off-resonance / nonlinear-gradient approximation pipeline.

    Args:
        kernels:
            Mapping  key → (K_kern,) Tensor.  Keys whose last element is 'x', 'y',
            or 'z' and first two elements are (0, 0) are treated as linear orders
            used to build k(t).  All other keys contribute to Φ(r, t).
        ideal_waveforms:
            {'x': (K,), 'y': (K,), 'z': (K,)} — G^ideal_β(t) per axis.
        z_map:
            (N,) complex Tensor — rate map 1/T₂(r) + iγ ΔB₀(r).
        spatial_maps:
            (N, Q) real Tensor — spatial basis functions f_q(r) at each voxel.
            Column q must correspond to `waveform_order[q]`.  Any polynomial or
            harmonic basis is accepted (solid harmonics, monomials, Zernike, …).
        waveform_order:
            List of Q hashable keys, one per column of `spatial_maps`.  Each key
            identifies the corresponding entry in the nonlinear part of `convolved`.
        t:
            (K,) float Tensor — time points in seconds.
        gamma:
            Gyromagnetic ratio (rad s⁻¹ T⁻¹), e.g. 2π × 42.577e6 for ¹H.
        dt:
            Sampling interval (s).
        L:    Rank of the exp(Φ) approximation.
        P:    Rank for normal-operator time compression.
        R:    Eigendecomposition rank per p.
        n_oversampling: Extra sketch columns for the randomized SVD.
        n_power_iter:   Subspace power iterations for the randomized SVD.
        chunk_size:     Time-axis chunk size for the phi matvec operators.

    Returns:
        dict with keys:

        'k_traj'   – (K, 3) float32 — k-space coordinates k(t).
        'Omega'    – (K, L) complex — Ω_{k,l}, temporal basis (forward op).
        'Upsilon'  – (N, L) complex — Υ_{j,l}, spatial basis (forward op).
        'Lambda'   – (K, P) complex — Λ_{k,p}, temporal basis (normal op).
        'Theta_L'  – (N, R, P) complex — Θ_L_{j,r,p}, left spatial factor.
        'Theta_R'  – (N, R, P) complex — Θ_R_{j,r,p}, right spatial factor.

        Normal-op usage:
            exp(Φ_j + Φ*_i) ≈ Σ_{p,r} Λ[k,p] · Θ_L[j,r,p] · conj(Θ_R[i,r,p])
    """
    device = z_map.device
    dtype  = torch.complex64

    # 1. Convolve all gradient orders
    convolved = convolve_gradients(kernels, ideal_waveforms)

    # 2. k-space trajectory from linear (0, 0, beta) terms
    linear_waveforms = {
        key[-1]: convolved[key]
        for key in convolved
        if key[0] == 0 and key[1] == 0
    }
    k_traj = kspace_trajectory(linear_waveforms, gamma, dt)

    # 3. Nonlinear waveforms for Φ
    nonlin_waveforms = {
        key: val for key, val in convolved.items()
        if not (key[0] == 0 and key[1] == 0)
    }

    # 4. Build chunked matvec operators for exp(Φ)
    fwd, adj = make_phi_matvec(
        z_map, spatial_maps, nonlin_waveforms, waveform_order,
        t, gamma, dt, chunk_size=chunk_size,
    )

    N = z_map.shape[0]
    K = t.shape[0]

    # 5. Rank-L decomposition of exp(Φ) via randomized SVD
    Omega, Upsilon = phi_lowrank(
        fwd, adj, N, K, L,
        n_oversampling=n_oversampling,
        n_power_iter=n_power_iter,
        dtype=dtype, device=device,
    )

    # 6. Normal-operator decomposition → Λ, Θ_L, Θ_R
    Lambda, Theta_L, Theta_R = normal_op_lowrank(Omega, Upsilon, P, R)

    return {
        'k_traj':  k_traj,
        'Omega':   Omega,
        'Upsilon': Upsilon,
        'Lambda':  Lambda,
        'Theta_L': Theta_L,
        'Theta_R': Theta_R,
    }
