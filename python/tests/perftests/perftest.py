import math
import time

import torch


def theta_grad_orthogonal(
    Theta,
    Lambda,
    g_generator,
    k_batch_size=128,
):
    """
    Gradient assuming orthonormal columns of Theta.

    Assumes:
        sum_i Theta[i,p].conj() * Theta[i,q] = delta_pq

    Computes:
        dL / dTheta*

    Parameters
    ----------
    Theta : complex tensor, shape (Ni, P)

    Lambda : complex tensor, shape (Nk, P)

    g_generator : callable
        g_generator(k_indices) -> tensor shape (Kb, Ni)

        returns:
            g[k,i] = exp(Phi(r_i, t_k))

    k_batch_size : int

    Returns
    -------
    grad : complex tensor, shape (Ni, P)
    """

    device = Theta.device
    dtype = Theta.dtype

    Ni, P = Theta.shape
    Nk = Lambda.shape[0]

    grad = torch.zeros_like(Theta)

    for k0 in range(0, Nk, k_batch_size):

        k1 = min(k0 + k_batch_size, Nk)

        ks = torch.arange(k0, k1, device=device)

        # shape: (Kb, Ni)
        g_batch = g_generator(ks)

        # alpha[k,p] = sum_i g*[k,i] Theta[i,p]
        #
        # shape: (Kb, P)
        alpha = g_batch.conj() @ Theta

        # first term:
        #
        # sum_k |Lambda_kp|^2 Theta_jp
        #
        coeff = torch.sum(torch.abs(Lambda[ks]) ** 2, dim=0)

        grad += Theta * coeff[None, :]

        # second term:
        #
        # sum_k Lambda*_kp g_jk alpha_kp
        #
        # Rewrite as matmul to avoid (Kb, Ni, P) intermediate:
        #   grad -= g_batch.T @ (Lambda*.conj() * alpha)
        #         = (Ni, Kb) @ (Kb, P)  -->  (Ni, P)
        #
        coeff2 = Lambda[ks].conj() * alpha  # (Kb, P)
        grad -= g_batch.T @ coeff2          # (Ni, P) — plain .T, g not conjugated

    return grad


def theta_grad_nonorthogonal(
    Theta,
    Lambda,
    g_generator,
    k_batch_size=128,
):
    """
    Gradient without orthogonality assumption.

    Computes:
        dL / dTheta*

    Parameters
    ----------
    Theta : complex tensor, shape (Ni, P)

    Lambda : complex tensor, shape (Nk, P)

    g_generator : callable
        g_generator(k_indices) -> tensor shape (Kb, Ni)

    Returns
    -------
    grad : complex tensor, shape (Ni, P)
    """
    device = Theta.device

    Ni, P = Theta.shape
    Nk = Lambda.shape[0]

    grad = torch.zeros_like(Theta)

    #
    # Gram matrix:
    #
    # G[p',p] = sum_i Theta*[i,p'] Theta[i,p]
    #
    G = Theta.conj().T @ Theta   # (P, P) Gram matrix

    # M[s,p] = sum_k Lambda[k,s] * Lambda[k,p]  (P x P, real)
    # Accumulated in the same batch loop to avoid second pass over k.
    M = torch.zeros(G.shape, dtype=Lambda.dtype, device=device)

    for k0 in range(0, Nk, k_batch_size):

        k1 = min(k0 + k_batch_size, Nk)

        ks = torch.arange(k0, k1, device=device)

        # shape: (Kb, Ni)
        g_batch = g_generator(ks)

        # shape: (Kb, P)
        alpha = g_batch.conj() @ Theta

        # Accumulate M (outer product of Lambda rows summed over k)
        M += Lambda[ks].T @ Lambda[ks]   # (P, P)

        # Second term: sum_k Lambda[k,p] alpha[k,p] g[k,j]
        coeff2 = Lambda[ks] * alpha      # (Kb, P)  — Lambda real, no .conj()
        grad -= g_batch.T @ coeff2       # (Ni, P)

    # First term: Theta @ (G * M)
    # G * M is element-wise product (P x P), not matrix multiply
    grad += Theta @ (G * M)

    return grad


def orthonormalize_theta(Theta, Lambda=None):
    """
    Orthonormalize columns of Theta using QR.

    If Lambda is supplied, rescales Lambda so that
    the represented operator stays approximately invariant.

    Original:
        sum_p Lambda_kp Theta_p Theta_p^

    After QR:
        Theta = Q R

    We absorb R into Lambda approximately.

    Parameters
    ----------
    Theta : complex tensor, shape (Ni, P)

    Lambda : complex tensor, optional
        shape (Nk, P)

    Returns
    -------
    Theta_orth : complex tensor

    Lambda_new : complex tensor or None
    """

    #
    # QR decomposition
    #
    # Theta = Q R
    #
    Q, R = torch.linalg.qr(Theta)

    if Lambda is None:
        return Q

    #
    # We approximately absorb diagonal scaling into Lambda.
    #
    # Exact absorption is impossible because:
    #
    # Theta_p Theta_p^H
    #
    # is quadratic in Theta.
    #
    # But diagonal scaling works well in practice.
    #
    diagR = torch.diagonal(R)

    Lambda_new = Lambda * (torch.abs(diagR)[None, :] ** 2)

    return Q, Lambda_new


# ---------------------------------------------------------------------------
# Update Lambda (closed-form optimal for fixed Theta)
#
# Lambda_kp = |sum_i g*[k,i] Theta[i,p]|^2
# ---------------------------------------------------------------------------

def update_lambda(Theta, g_generator, Nk, k_batch_size=512, device="cpu"):
    """
    Compute optimal Lambda for fixed Theta.

    Lambda[k,p] = |alpha[k,p]|^2
    where alpha[k,p] = sum_i g*[k,i] Theta[i,p]

    Parameters
    ----------
    Theta : complex tensor, shape (Ni, P)
    g_generator : callable, returns (Kb, Ni)
    Nk : int
    k_batch_size : int

    Returns
    -------
    Lambda : real tensor, shape (Nk, P)
    """
    P = Theta.shape[1]
    Lambda = torch.zeros(Nk, P, dtype=Theta.real.dtype, device=device)

    for k0 in range(0, Nk, k_batch_size):
        k1 = min(k0 + k_batch_size, Nk)
        ks = torch.arange(k0, k1, device=device)
        g_batch = g_generator(ks)          # (Kb, Ni)
        alpha = g_batch.conj() @ Theta     # (Kb, P)
        Lambda[k0:k1] = alpha.abs() ** 2

    return Lambda


# ---------------------------------------------------------------------------
# Loss computation (with optimal Lambda already applied)
#
# L = sum_k ( ||g_k||^4 - sum_p |alpha_kp|^4 )
#
# Derivation:
#   ||A^k - hat_A^k||_F^2
#   = ||A^k||_F^2 - 2 Re<hat_A^k, A^k> + ||hat_A^k||_F^2
#
#   ||A^k||_F^2 = ||g_k g_k^H||_F^2 = (g_k^H g_k)^2 = ||g_k||^4
#
#   With Lambda_kp = |alpha_kp|^2 and orthonormal Theta:
#     ||hat_A^k||_F^2 = sum_p |Lambda_kp|^2 = sum_p |alpha_kp|^4
#     2 Re<hat_A^k, A^k>   = 2 sum_p Lambda_kp |alpha_kp|^2
#                           = 2 sum_p |alpha_kp|^4
#
#   => L = sum_k ( ||g_k||^4 - sum_p |alpha_kp|^4 )   >= 0
# ---------------------------------------------------------------------------

def compute_loss(Theta, g_generator, Nk, k_batch_size=512, device="cpu"):
    """
    Compute loss + energy fraction assuming Lambda is at its optimum.

    Loss (4th-order, with optimal Lambda):
        L = sum_k ( ||g_k||^4 - sum_p |alpha_kp|^4 )   >= 0

    WARNING — for unit-magnitude exponentials (|g[i,k]| = 1):
        ||g_k||^4 = Ni^2  for ALL k, so L0 = Nk * Ni^2 regardless of Theta.
        The "cross-term floor"  ~= Nk * Ni^2 * (1 - 1/P)  for uniform spreading,
        meaning more P can give a HIGHER floor!  Use energy_frac for quality.

    Energy fraction (2nd-order):
        energy_frac = sum_k sum_p |alpha_kp|^2 / (Nk * Ni)
        = fraction of g-energy captured in span(Theta).
        Good approximation quality requires energy_frac ≈ 1.

    Returns
    -------
    loss          : float
    energy_frac   : float  (in [0, 1])
    """
    loss = 0.0
    energy_cap = 0.0
    energy_tot = 0.0

    for k0 in range(0, Nk, k_batch_size):
        k1 = min(k0 + k_batch_size, Nk)
        ks = torch.arange(k0, k1, device=device)
        g_batch = g_generator(ks)                      # (Kb, Ni)

        g_norm_sq = (g_batch.abs() ** 2).sum(dim=1)   # (Kb,)
        gnorm4    = g_norm_sq ** 2                     # (Kb,)

        alpha = g_batch.conj() @ Theta                 # (Kb, P)
        a2    = alpha.abs() ** 2                       # (Kb, P)

        loss       += (gnorm4 - a2.pow(2).sum(dim=1)).sum().item()
        energy_cap += a2.sum().item()
        energy_tot += g_norm_sq.sum().item()

    energy_frac = energy_cap / energy_tot if energy_tot > 0 else 0.0
    return loss, energy_frac


# ---------------------------------------------------------------------------
# Operator error on random test images
#
# True normal operator:
#   (N rho)[i] = sum_{j,k} g[j,k] * conj(g[i,k]) * rho[j]
#              = G^H (G rho)   where G[k,i] = g[i,k] = g_batch[kb,i]
#   Forward:  G  rho   = g_batch     @ rho   (no conjugate)
#   Adjoint:  G^H a    = g_batch.mH  @ a     (conjugate-transpose)
#
# Approximate normal operator:
#   (N_hat rho)[i] = conj(Theta) @ (mu * (Theta.T @ rho))
#   where mu[p] = sum_k Lambda[k,p],  Theta.T (NOT .H) mirrors g vs conj(g)
#
# Relative error: mean over n_test random unit rho of
#   ||N rho - N_hat rho||^2 / ||N rho||^2
# ---------------------------------------------------------------------------

def compute_operator_error(Theta, Lambda, g_generator, Nk,
                           k_batch_size=512, device="cpu", n_test=8, seed=99):
    """
    Estimate relative operator error on random complex images.

    Approximation of N = G^H G:
        N_hat = conj(Theta) @ diag(mu) @ Theta.T
    where mu[p] = sum_k Lambda[k,p].

    Returns
    -------
    rel_err : float  (‖N rho − N_hat rho‖² / ‖N rho‖²  averaged over rho)
    """
    torch.manual_seed(seed)
    Ni       = Theta.shape[0]
    cdtype   = Theta.dtype
    fdtype   = Theta.real.dtype

    rho = (torch.randn(Ni, n_test, dtype=fdtype, device=device)
           + 1j * torch.randn(Ni, n_test, dtype=fdtype, device=device)
           ).to(cdtype)
    rho = rho / rho.norm(dim=0, keepdim=True)  # unit norm columns

    # N_hat rho = conj(Theta) @ (mu * (Theta.T @ rho))
    mu       = Lambda.sum(dim=0)               # (P,)
    fwd_hat  = Theta.T @ rho                   # (P, n_test)
    Nrho_hat = Theta.conj() @ (mu[:, None] * fwd_hat)  # (Ni, n_test)

    # True N rho  (G^H G rho, batched)
    Nrho_true = torch.zeros_like(rho)
    for k0 in range(0, Nk, k_batch_size):
        k1 = min(k0 + k_batch_size, Nk)
        ks = torch.arange(k0, k1, device=device)
        g_batch = g_generator(ks)              # (Kb, Ni)
        fwd     = g_batch @ rho                # (Kb, n_test)
        Nrho_true += g_batch.mH @ fwd          # (Ni, n_test)

    err_sq  = (Nrho_true - Nrho_hat).norm(dim=0).pow(2)   # (n_test,)
    base_sq = Nrho_true.norm(dim=0).pow(2)                 # (n_test,)
    return (err_sq / base_sq).mean().item()


# ---------------------------------------------------------------------------
# g-generator factories
# ---------------------------------------------------------------------------

def make_random_exp_generator(Ni, Nk, device, seed, cdtype, fdtype):
    """
    Fully random field map: field[i] ~ U(0, 2pi*500 Hz), T=20ms.

    High bandwidth (~20pi rad) → effective rank >> P.
    Expected: slow / poor convergence.
    """
    torch.manual_seed(seed)
    max_freq = 2.0 * math.pi * 500.0
    T        = 20e-3
    field = max_freq * torch.rand(Ni, dtype=fdtype, device=device)
    t     = torch.linspace(0, T, Nk, dtype=fdtype, device=device)

    def g_generator(k_indices):
        phase = t[k_indices, None] * field[None, :]
        return torch.exp(1j * phase).to(cdtype)

    bw = max_freq * T / (2 * math.pi)
    desc = f"random exp  (bandwidth={bw:.1f} cycles, max_freq=500Hz, T=20ms)"
    return g_generator, desc


def make_bandlimited_exp_generator(Ni, Nk, P, device, seed, cdtype, fdtype,
                                   bandwidth_cycles=None):
    """
    Exponential with bandwidth tuned to P:

        bandwidth_cycles = P / 2   (default)
        max_freq = bandwidth_cycles / T

    P/2 cycles across readout → P basis functions suffice → good convergence.
    """
    torch.manual_seed(seed)
    if bandwidth_cycles is None:
        bandwidth_cycles = P / 2.0
    T        = 20e-3
    max_freq = bandwidth_cycles * 2.0 * math.pi / T
    field = max_freq * torch.rand(Ni, dtype=fdtype, device=device)
    t     = torch.linspace(0, T, Nk, dtype=fdtype, device=device)

    def g_generator(k_indices):
        phase = t[k_indices, None] * field[None, :]
        return torch.exp(1j * phase).to(cdtype)

    desc = (f"bandlimited exp  (bandwidth={bandwidth_cycles:.1f} cycles,"
            f" max_freq={max_freq/(2*math.pi):.1f}Hz, T=20ms)")
    return g_generator, desc


def make_rank_p_exact_generator(Ni, Nk, P, device, seed, cdtype, fdtype):
    """
    Exactly rank-P ground truth — algorithm should converge to loss ≈ 0.

    Construction:
        Theta_true  : (Ni, P) random orthonormal  — true spatial modes
        group[k]    : (Nk,)   random in {0,...,P-1} — which mode is active at time k
        A[k]        : scalar random complex         — amplitude at time k

        g[i, k] = A[k] * Theta_true[i, group[k]]

    Then A^k_ij = g[j,k] * g*[i,k]
               = |A[k]|^2 * Theta_true[j, p(k)] * conj(Theta_true[i, p(k)])

    This is EXACTLY sum_p Lambda[k,p] * Theta_true[j,p] * conj(Theta_true[i,p])
    with Lambda[k, group[k]] = |A[k]|^2 and Lambda[k, p≠group[k]] = 0.

    Cross-terms vanish (one mode per k) → perfect rank-P decomposition.
    """
    torch.manual_seed(seed)

    # Random orthonormal Theta_true
    Q, _ = torch.linalg.qr(
        torch.randn(Ni, P, dtype=cdtype, device=device)
    )
    Theta_true = Q  # (Ni, P)

    # Group assignment: one mode per time point
    group = torch.randint(0, P, (Nk,), device=device)  # (Nk,)

    # Random complex amplitude per k
    amp = (torch.randn(Nk, dtype=fdtype, device=device)
           + 1j * torch.randn(Nk, dtype=fdtype, device=device)).to(cdtype)

    def g_generator(k_indices):
        # g[kb, i] = amp[k] * Theta_true[i, group[k]]
        # active_mode[kb, i] = Theta_true[i, group[k_indices[kb]]]
        active_mode  = Theta_true[:, group[k_indices]].T  # (Kb, Ni)
        active_amp   = amp[k_indices]                     # (Kb,)
        return active_amp[:, None] * active_mode           # (Kb, Ni)

    desc = f"exact rank-{P}  (one mode active per k, loss should → 0)"
    return g_generator, desc, Theta_true


# ---------------------------------------------------------------------------
# Gradient of loss w.r.t. Lambda (real, non-negative)
#
# dL/dLambda[k,p] = sum_q |<theta_p, theta_q>|^2 * Lambda[k,q] - |alpha[k,p]|^2
#                 = (Gabs2 @ Lambda[k,:])[p]  -  |alpha[k,p]|^2
#
# For orthonormal Theta: Gabs2 = I → grad = Lambda[k,p] - |alpha[k,p]|^2
#   (optimal at Lambda = |alpha|^2, same as update_lambda)
# ---------------------------------------------------------------------------

def lambda_grad(Lambda, Theta, g_generator, Nk, k_batch_size=512, device="cpu"):
    """
    Gradient of loss w.r.t. Lambda (real tensor, shape Nk x P).

    Parameters
    ----------
    Lambda : real tensor, shape (Nk, P)
    Theta  : complex tensor, shape (Ni, P)  — need not be orthonormal
    """
    G      = Theta.conj().T @ Theta          # (P, P) Gram
    Gabs2  = G.abs() ** 2                    # |G[p,q]|^2, symmetric

    grad = torch.zeros_like(Lambda)

    for k0 in range(0, Nk, k_batch_size):
        k1 = min(k0 + k_batch_size, Nk)
        ks = torch.arange(k0, k1, device=device)
        g_batch    = g_generator(ks)              # (Kb, Ni)
        alpha      = g_batch.conj() @ Theta       # (Kb, P)
        alpha_abs2 = alpha.abs() ** 2             # (Kb, P)

        # (Kb, P) @ (P, P) - (Kb, P)
        grad[k0:k1] = Lambda[k0:k1] @ Gabs2 - alpha_abs2

    return grad


# ---------------------------------------------------------------------------
# Theta update A: 4th-moment power iteration  (MRI_Physics.tex step 3)
#
# Maximises sum_k sum_p |alpha_kp|^4  (tex loss objective).
# Does NOT minimise ||N - N_hat||_op directly.
# Use when you want to approximate individual A^k slices.
# ---------------------------------------------------------------------------

def theta_power_iterate(Theta, g_generator, Nk, k_batch_size=512, device="cpu"):
    """
    One 4th-moment power-iteration step.

    Theta_new[j,p] = sum_k |alpha_kp|^2 * alpha_kp * g[j,k]

    NOT yet orthonormalized.
    """
    Theta_new = torch.zeros_like(Theta)

    for k0 in range(0, Nk, k_batch_size):
        k1 = min(k0 + k_batch_size, Nk)
        ks = torch.arange(k0, k1, device=device)
        g_batch  = g_generator(ks)             # (Kb, Ni)
        alpha    = g_batch.conj() @ Theta      # (Kb, P)
        coeff    = alpha * (alpha.abs() ** 2)  # Λ * alpha, (Kb, P)
        Theta_new += g_batch.T @ coeff         # (Ni, P)

    return Theta_new


# ---------------------------------------------------------------------------
# Theta update B: randomized SVD on N = G^H G
#
# N = sum_k g_k g_k^H  (the true normal operator, Ni × Ni)
#
# Optimal Theta for minimising ||N - N_hat||_F^2 = P leading eigenvectors of N.
# (Eckart-Young theorem for Hermitian PSD matrices)
#
# Apply N to a vector x:
#   (N x)[i] = sum_k conj(g[i,k]) * sum_j g[j,k] * x[j]
#            = G^H (G x)
#   Forward:  G  x = g_batch     @ x   (no conjugate)
#   Adjoint:  G^H a = g_batch.mH @ a
#
# Cost: O((P + oversampling) * n_power_iter * Nk * Ni)
#   ≈ same as ~5 tex power iterations, but targets the right objective.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Time-segmentation initialisation  (Theta + Lambda)
#
# Theta[:,p] = g(tau_p) / ||g(tau_p)||    column-normalised segment basis
#              = exp(i * field * tau_p) / sqrt(Ni)  for unit-magnitude g
#
# Lambda[k,p] = c_p[k]^2
#   where c_p[k] are LINEAR interpolation weights between adjacent segment
#   centres: for k in [tau_p, tau_{p+1}], c_p[k] + c_{p+1}[k] = 1.
#
#   This is Fessler's diagonal O(P) normal-operator approximation:
#       A^k ≈ sum_p |c_p[k]|^2 theta_p theta_p^H
#
#   NOT the same as update_lambda (|alpha|^2), which is the OPTIMISED Lambda
#   for given Theta.  Time-seg Lambda comes from the physical interpolation.
#
# tau_p = p * (Nk-1) / (P-1)  — P evenly-spaced time indices in [0, Nk-1]
# ---------------------------------------------------------------------------

def time_seg_init(g_generator, Nk, P, device):
    """
    Time-segmentation basis: P evenly-spaced segment vectors, column-normalised.

    Theta[:,p] = g(tau_p) / ||g(tau_p)||

    Lambda is NOT returned here — always use update_lambda(Theta, ...) afterward.
    Linear-interpolation c_p^2 weights are suboptimal; |alpha|^2 from update_lambda
    is the closed-form optimal Lambda for any fixed Theta.

    Returns
    -------
    Theta : complex tensor, shape (Ni, P)
    """
    tau_long = torch.linspace(0, Nk - 1, P, device=device).long()

    cols = []
    for p in range(P):
        col = g_generator(tau_long[p:p+1])[0]   # (Ni,)
        col = col / col.norm()
        cols.append(col)

    return torch.stack(cols, dim=1)   # (Ni, P)


def apply_N(X, g_generator, Nk, k_batch_size, device):
    """Apply N = G^H G to matrix X of shape (Ni, l)."""
    Y = torch.zeros_like(X)
    for k0 in range(0, Nk, k_batch_size):
        k1 = min(k0 + k_batch_size, Nk)
        ks = torch.arange(k0, k1, device=device)
        g_batch = g_generator(ks)      # (Kb, Ni)
        fwd = g_batch @ X              # (Kb, l)
        Y  += g_batch.mH @ fwd         # (Ni, l)
    return Y


def theta_rsvd(g_generator, Ni, Nk, P, k_batch_size=512, device="cpu",
               oversampling=10, n_power_iter=2, seed=7):
    """
    P leading eigenvectors of N = G^H G via randomized SVD.

    Optimal Theta for ||N - Theta diag(mu) Theta^H||_F^2.

    Parameters
    ----------
    oversampling : int
        Extra columns for accuracy (l = P + oversampling).
    n_power_iter : int
        Subspace power iterations for accuracy on flat spectra.

    Returns
    -------
    Theta : (Ni, P) orthonormal  — P leading eigenvectors of N
    mu    : (P,)  real           — corresponding eigenvalues
    """
    torch.manual_seed(seed)
    cdtype = torch.complex64
    fdtype = torch.float32
    l = P + oversampling

    # Random Gaussian test matrix
    Omega = (torch.randn(Ni, l, dtype=fdtype, device=device)
             + 1j * torch.randn(Ni, l, dtype=fdtype, device=device)
             ).to(cdtype)

    # Range approximation Y = (N^{n_power_iter+1}) Omega
    Y = apply_N(Omega, g_generator, Nk, k_batch_size, device)
    for _ in range(n_power_iter):
        Q, _ = torch.linalg.qr(Y)
        Y    = apply_N(Q, g_generator, Nk, k_batch_size, device)

    # Orthonormal basis for range of N^{...} Omega
    Q, _ = torch.linalg.qr(Y)  # (Ni, l)

    # Projected small matrix B = Q^H N Q  (l × l)
    NQ = apply_N(Q, g_generator, Nk, k_batch_size, device)  # (Ni, l)
    B  = Q.mH @ NQ                                           # (l, l)

    # Eigendecompose B (Hermitian PSD)
    eigvals, eigvecs = torch.linalg.eigh(B)           # ascending order

    # Return P largest — conjugated because our N_hat formula uses conj(Theta):
    #   N_hat[i,j] = sum_p mu_p conj(Theta[i,p]) Theta[j,p]
    # Optimal Theta minimises ||conj(N) - Theta diag(mu) Theta^H||_F^2
    # = eigenvectors of conj(N) = conj(eigenvectors of N).
    Theta = (Q @ eigvecs[:, -P:]).conj()   # (Ni, P)
    mu    = eigvals[-P:].real              # (P,)

    return Theta, mu


# ---------------------------------------------------------------------------
# Main alternating optimisation loop
# ---------------------------------------------------------------------------

def run_exp_decomposition(
    Ni: int,
    Nk: int,
    P: int,
    g_generator,
    label: str = "",
    n_iter: int = 20,
    k_batch_size: int = 512,
    print_every: int = 5,
    compute_op_err: bool = True,
    use_rsvd_init: bool = False,
    rsvd_oversampling: int = 10,
    rsvd_power_iter: int = 2,
    use_time_seg_init: bool = False,
    device: str = "cpu",
    seed: int = 42,
):
    """
    4th-moment power iteration for the exponential decomposition.

    use_time_seg_init=True:
        Init Theta from P evenly-spaced time-segment basis vectors
        (Fessler's analytical basis: g(tau_p)).  Good starting point for
        exponential g.  Overrides rsvd/random init if set.

    use_rsvd_init=True:
        Init Theta from rsvd of N = G^H G  (P leading eigenvectors).

    default (both False):
        Random orthonormal init for Theta.

    Metrics:
        loss%   = 4th-power loss vs random-Theta baseline (ignore absolute %)
        efrac   = fraction of g-energy captured in span(Theta)  [0 → 1]
        op_err  = E[‖N rho − N_hat rho‖² / ‖N rho‖²]  on random images
    """

    cdtype = torch.complex64
    fdtype = torch.float32

    if use_time_seg_init:
        init_str = "time-seg-init"
    elif use_rsvd_init:
        init_str = "rsvd-init"
    else:
        init_str = "rand-init"
    print(f"\n{label}  [{init_str}, power-iter]")
    print(f"  Ni={Ni:,}  Nk={Nk:,}  P={P}  n_iter={n_iter}  device={device}")

    t0_wall = time.perf_counter()

    # ------------------------------------------------------------------
    # Initialise Theta
    # ------------------------------------------------------------------
    if use_time_seg_init:
        Theta = time_seg_init(g_generator, Nk, P, device)
        Theta = orthonormalize_theta(Theta)
    elif use_rsvd_init:
        print(f"  [rsvd init: oversampling={rsvd_oversampling}, "
              f"power_iter={rsvd_power_iter}]")
        Theta, _ = theta_rsvd(g_generator, Ni, Nk, P,
                              k_batch_size=k_batch_size, device=device,
                              oversampling=rsvd_oversampling,
                              n_power_iter=rsvd_power_iter)
    else:
        torch.manual_seed(seed + 1)
        Theta = (torch.randn(Ni, P, dtype=fdtype, device=device)
                 + 1j * torch.randn(Ni, P, dtype=fdtype, device=device)).to(cdtype)
        Theta = orthonormalize_theta(Theta)

    # ------------------------------------------------------------------
    # Baseline loss at iter 0
    # ------------------------------------------------------------------
    loss0, efrac0 = compute_loss(Theta, g_generator, Nk,
                                 k_batch_size=k_batch_size, device=device)
    print(f"  iter   0 | loss = {loss0:.4e}  (100.00%)  efrac = {efrac0:.4f}")

    # ------------------------------------------------------------------
    # Power iteration loop (shared by both init paths)
    # ------------------------------------------------------------------
    t_iter_total = 0.0

    for it in range(1, n_iter + 1):
        t0 = time.perf_counter()
        Theta = orthonormalize_theta(
            theta_power_iterate(Theta, g_generator, Nk,
                                k_batch_size=k_batch_size, device=device)
        )
        t_iter = time.perf_counter() - t0
        t_iter_total += t_iter

        if it % print_every == 0 or it == 1:
            loss, efrac = compute_loss(Theta, g_generator, Nk,
                                       k_batch_size=k_batch_size, device=device)
            pct = 100.0 * loss / loss0
            print(
                f"  iter {it:4d} | loss = {loss:.4e}  ({pct:6.2f}%)"
                f"  efrac = {efrac:.4f}"
                f"  [{t_iter:.2f}s, avg {t_iter_total/it:.2f}s]"
            )

    # ------------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------------
    loss_final, efrac_final = compute_loss(Theta, g_generator, Nk,
                                           k_batch_size=k_batch_size, device=device)
    Lambda = update_lambda(Theta, g_generator, Nk,
                           k_batch_size=k_batch_size, device=device)
    t_elapsed = time.perf_counter() - t0_wall
    pct_str   = f"{100.0 * loss_final / loss0:.2f}% of L0"

    if compute_op_err:
        op_err = compute_operator_error(Theta, Lambda, g_generator, Nk,
                                        k_batch_size=k_batch_size, device=device)
        op_err_str = f"{op_err:.4e}"
    else:
        op_err_str = "(skipped)"

    print(f"  Final: loss = {loss_final:.4e}"
          + f"  ({pct_str})"
          + f"  efrac = {efrac_final:.4f}"
          + f"  op_err = {op_err_str}"
          + f"  |  total = {t_elapsed:.2f}s")

    return Theta, Lambda, loss_final


# ---------------------------------------------------------------------------
# Joint gradient descent on Theta and Lambda
# ---------------------------------------------------------------------------

def compute_full_loss(Theta, Lambda, g_generator, Nk, k_batch_size, device):
    """
    Full loss (no orthonormality assumption):
        L = sum_k ||g_k g_k^H - sum_p Lambda_kp theta_p theta_p^H||_F^2
          = sum_k [||g_k||^4 - 2 sum_p Lambda_kp |alpha_kp|^2
                   + sum_{p,q} Lambda_kp Lambda_kq |<theta_p,theta_q>|^2]

    Differentiable w.r.t. Theta and Lambda.
    """
    G_gram = Theta.conj().T @ Theta       # (P, P)
    Gabs2  = G_gram.abs() ** 2            # |<theta_p, theta_q>|^2

    loss = torch.zeros(1, device=device, dtype=torch.float32)
    for k0 in range(0, Nk, k_batch_size):
        k1       = min(k0 + k_batch_size, Nk)
        ks       = torch.arange(k0, k1, device=device)
        g_batch  = g_generator(ks).detach()        # (Kb, Ni) — no grad through g
        alpha    = g_batch.conj() @ Theta           # (Kb, P)

        gnorm4   = (g_batch.abs() ** 2).sum(dim=1) ** 2          # (Kb,)
        cross    = (Lambda[k0:k1] * alpha.abs() ** 2).sum(dim=1)  # (Kb,)
        hat_sq   = (Lambda[k0:k1] @ Gabs2 * Lambda[k0:k1]).sum(dim=1)  # (Kb,)

        loss = loss + (gnorm4 - 2.0 * cross + hat_sq).sum()

    return loss


def run_exp_decomposition_gd(
    Ni: int,
    Nk: int,
    P: int,
    g_generator,
    label: str = "",
    n_iter: int = 100,
    k_batch_size: int = 512,
    print_every: int = 10,
    compute_op_err: bool = True,
    lr_theta: float = 1e-4,
    lr_lambda: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    init_theta=None,
    use_rsvd_init: bool = False,
    rsvd_oversampling: int = 10,
    rsvd_power_iter: int = 2,
    use_time_seg_init: bool = False,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Joint Adam on Theta (complex, Wirtinger) and Lambda (real, clamped >= 0).

    Manual Adam — no autograd.  Autograd stores every g_batch for backward
    (O(Nk * Ni * P) memory), which is untenable at scale.  Manual gradients
    process one k-batch at a time and discard it immediately.

    Adam suppresses the scaling gauge instability (Theta -> c*Theta,
    Lambda -> |c|^-2*Lambda is loss-invariant) far better than fixed-lr SGD
    because per-coordinate adaptive lr normalises the gradient scale.
    """
    cdtype = torch.complex64
    fdtype = torch.float32

    print(f"\n{label}  [Adam, lr_theta={lr_theta:.1e}, lr_lambda={lr_lambda:.1e}]")
    print(f"  Ni={Ni:,}  Nk={Nk:,}  P={P}  n_iter={n_iter}  device={device}")

    t0_wall = time.perf_counter()

    # ------------------------------------------------------------------
    # Init Theta
    # ------------------------------------------------------------------
    if init_theta is not None:
        Theta = init_theta.clone()
        print("  [external init]")
    elif use_time_seg_init:
        print("  [time-seg init]")
        # Column-normalised basis + analytical interp Lambda — no QR.
        # QR would mix columns, invalidating Lambda_ts.
        # Adam handles non-orthonormal Theta freely.
        Theta = time_seg_init(g_generator, Nk, P, device)
    elif use_rsvd_init:
        print(f"  [rsvd init: oversampling={rsvd_oversampling}, "
              f"power_iter={rsvd_power_iter}]")
        Theta, _ = theta_rsvd(g_generator, Ni, Nk, P,
                              k_batch_size=k_batch_size, device=device,
                              oversampling=rsvd_oversampling,
                              n_power_iter=rsvd_power_iter)
    else:
        torch.manual_seed(seed + 1)
        Theta = (torch.randn(Ni, P, dtype=fdtype, device=device)
                 + 1j * torch.randn(Ni, P, dtype=fdtype, device=device)).to(cdtype)
        Theta = orthonormalize_theta(Theta)

    Lambda = update_lambda(Theta, g_generator, Nk,
                           k_batch_size=k_batch_size, device=device)

    # Adam states — complex for Theta (Wirtinger), real for Lambda
    m_th = torch.zeros_like(Theta)           # 1st moment, complex
    v_th = torch.zeros(Ni, P, dtype=fdtype, device=device)  # 2nd moment, real
    m_lm = torch.zeros_like(Lambda)
    v_lm = torch.zeros_like(Lambda)

    # Init is orthonormal so compute_full_loss == compute_loss here,
    # but use full_loss throughout for consistency.
    loss0 = compute_full_loss(Theta, Lambda, g_generator, Nk,
                              k_batch_size=k_batch_size, device=device).item()
    print(f"  iter   0 | loss = {loss0:.4e}  (100.00%)")

    t_iter_total = 0.0

    for it in range(1, n_iter + 1):
        t0 = time.perf_counter()

        # Gradients (manual, O(k_batch_size * Ni * P) peak memory)
        g_th = theta_grad_nonorthogonal(Theta, Lambda, g_generator,
                                        k_batch_size=k_batch_size)
        g_lm = lambda_grad(Lambda, Theta, g_generator, Nk,
                           k_batch_size=k_batch_size, device=device)

        # Adam update — Theta (complex Wirtinger Adam)
        m_th = beta1 * m_th + (1.0 - beta1) * g_th
        v_th = beta2 * v_th + (1.0 - beta2) * g_th.abs() ** 2
        bc1  = 1.0 - beta1 ** it
        bc2  = 1.0 - beta2 ** it
        Theta = Theta - lr_theta * (m_th / bc1) / (torch.sqrt(v_th / bc2) + eps)

        # Adam update — Lambda (real)
        m_lm = beta1 * m_lm + (1.0 - beta1) * g_lm
        v_lm = beta2 * v_lm + (1.0 - beta2) * g_lm ** 2
        Lambda = (Lambda - lr_lambda * (m_lm / bc1) / (torch.sqrt(v_lm / bc2) + eps)
                  ).clamp(min=0.0)

        t_iter = time.perf_counter() - t0
        t_iter_total += t_iter

        if it % print_every == 0 or it == 1:
            loss = compute_full_loss(Theta, Lambda, g_generator, Nk,
                                     k_batch_size=k_batch_size, device=device).item()
            pct = 100.0 * loss / loss0
            if compute_op_err:
                # Normalize Theta columns before computing op_err.
                # Adam lets ||theta_p|| drift → Lambda from Adam state scales as c^2
                # → N_hat scales as c^4 → op_err explodes.  Column-normalize + fresh
                # update_lambda removes the gauge artifact and shows true subspace quality.
                norms      = Theta.norm(dim=0, keepdim=True).clamp(min=1e-8)
                Theta_n    = Theta / norms
                Lambda_n   = update_lambda(Theta_n, g_generator, Nk,
                                           k_batch_size=k_batch_size, device=device)
                op_err_it  = compute_operator_error(Theta_n, Lambda_n, g_generator, Nk,
                                                    k_batch_size=k_batch_size,
                                                    device=device)
                op_str = f"  op_err={op_err_it:.4e}  |nrm|={norms.mean().item():.3f}"
            else:
                op_str = ""
            print(
                f"  iter {it:4d} | loss={loss:.4e}  ({pct:6.2f}%)"
                + op_str
                + f"  [{t_iter:.2f}s]"
            )

    loss_final = compute_full_loss(Theta, Lambda, g_generator, Nk,
                                   k_batch_size=k_batch_size, device=device).item()
    t_elapsed  = time.perf_counter() - t0_wall
    pct_str    = f"{100.0 * loss_final / loss0:.2f}% of L0"

    # Return col-normalised Theta with fresh optimal Lambda
    norms_f   = Theta.norm(dim=0, keepdim=True).clamp(min=1e-8)
    Theta_ret = Theta / norms_f
    Lambda_ret = update_lambda(Theta_ret, g_generator, Nk,
                               k_batch_size=k_batch_size, device=device)

    print(f"  Final: loss={loss_final:.4e}  ({pct_str})  |  total={t_elapsed:.2f}s")

    return Theta_ret, Lambda_ret, loss_final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cdtype = torch.complex64
    fdtype = torch.float32

    SEED = 42
    Ni_small, Nk_small = 2_000,   5_000
    Ni_large, Nk_large = 100_000, 400_000

    Test1 = False
    if Test1:
        # ------------------------------------------------------------------
        # 1. Exact rank-P  (correctness: loss → 0, op_err → 0)
        # ------------------------------------------------------------------
        print("=" * 70)
        print("TEST 1 — exact rank-P  (loss → 0, op_err → 0 expected)")
        print("=" * 70)
        P = 8
        g_gen, desc, _ = make_rank_p_exact_generator(
            Ni_small, Nk_small, P, device, SEED, cdtype, fdtype
        )
        run_exp_decomposition(
            Ni=Ni_small, Nk=Nk_small, P=P,
            g_generator=g_gen, label=desc,
            n_iter=15, k_batch_size=256,
            print_every=5, device=device, seed=SEED,
        )

    Test2 = False
    if Test2:
        # ------------------------------------------------------------------
        # 2. P-sweep — bandlimited exp, fixed bandwidth = 4 cycles
        #    Shows: efrac and op_err vs P  (loss% is misleading here)
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("TEST 2 — P-sweep  (bandlimited 4 cycles, fixed bandwidth)")
        print("  Note: loss% increases with P for unit exponentials (expected)")
        print("        efrac and op_err are the meaningful accuracy metrics")
        print("=" * 70)
        for P in [2, 4, 8, 16, 32]:
            g_gen, desc = make_bandlimited_exp_generator(
                Ni_small, Nk_small, P, device, SEED, cdtype, fdtype,
                bandwidth_cycles=4.0,
            )
            run_exp_decomposition(
                Ni=Ni_small, Nk=Nk_small, P=P,
                g_generator=g_gen, label=f"  P={P:3d}  {desc}",
                n_iter=15, k_batch_size=256,
                print_every=9999,   # suppress per-iter output, show only Final
                device=device, seed=SEED,
            )

    Test3 = False
    if Test3:
        # ------------------------------------------------------------------
        # 3. Random exponential  (high bandwidth)
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("TEST 3 — random exp  (500 Hz / 20 ms, high bandwidth)")
        print("=" * 70)
        P = 8
        g_gen, desc = make_random_exp_generator(
            Ni_small, Nk_small, device, SEED, cdtype, fdtype
        )
        run_exp_decomposition(
            Ni=Ni_small, Nk=Nk_small, P=P,
            g_generator=g_gen, label=desc,
            n_iter=15, k_batch_size=256,
            print_every=5, device=device, seed=SEED,
        )

    Test4 = False
    if Test4:
        # ------------------------------------------------------------------
        # 4. Full-scale perf test — rand-init vs rsvd-init
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("TEST 4 — perf test at scale  (Ni=100k, Nk=400k)")
        print("  rand-init vs rsvd-init + power-iter refinement")
        print("=" * 70)
        P = 10
        g_gen, desc = make_bandlimited_exp_generator(
            Ni_large, Nk_large, P, device, SEED, cdtype, fdtype
        )

        # Large k_batch_size for GPU: fewer Python loop iters (400k/4096 = ~98 vs 781)
        # print_every=9999: skip per-iter loss eval at this scale (too expensive)
        kbs_large = 4096

        run_exp_decomposition(
            Ni=Ni_large, Nk=Nk_large, P=P,
            g_generator=g_gen, label=f"[rand-init]  {desc}",
            n_iter=50, k_batch_size=kbs_large,
            print_every=9999, compute_op_err=True,
            use_rsvd_init=False,
            device=device, seed=SEED,
        )

        run_exp_decomposition(
            Ni=Ni_large, Nk=Nk_large, P=P,
            g_generator=g_gen, label=f"[rsvd-init]  {desc}",
            n_iter=50, k_batch_size=kbs_large,
            print_every=9999, compute_op_err=True,
            use_rsvd_init=True, rsvd_oversampling=10, rsvd_power_iter=2,
            device=device, seed=SEED,
        )

    Test5 = True
    if Test5:
        # ------------------------------------------------------------------
        # 5. Time-seg quality: col-norm → QR-orth → Adam iters
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("TEST 5 — time-seg init quality stages  (Ni=100k, Nk=400k)")
        print("=" * 70)
        P = 14
        kbs5 = 4096
        g_gen5, desc5 = make_bandlimited_exp_generator(
            Ni_large, Nk_large, P, device, SEED, cdtype, fdtype
        )
        print(f"  {desc5}  P={P}")

        # Stage 1: time-seg col-normalised Theta + optimal Lambda (|alpha|^2)
        Theta_ts  = time_seg_init(g_gen5, Nk_large, P, device)
        Lambda_ts = update_lambda(Theta_ts, g_gen5, Nk_large,
                                  k_batch_size=kbs5, device=device)
        op_ts = compute_operator_error(Theta_ts, Lambda_ts, g_gen5, Nk_large,
                                       k_batch_size=kbs5, device=device)
        print(f"\n  [stage 1: time-seg col-norm]   op_err={op_ts:.4e}")

        # Stage 2: QR-orthonormalize; Lambda recomputed — columns mixed, op_err may change
        Theta_orth  = orthonormalize_theta(Theta_ts)
        Lambda_orth = update_lambda(Theta_orth, g_gen5, Nk_large,
                                    k_batch_size=kbs5, device=device)
        op_orth = compute_operator_error(Theta_orth, Lambda_orth, g_gen5, Nk_large,
                                         k_batch_size=kbs5, device=device)
        print(f"  [stage 2: time-seg QR-orth]    op_err={op_orth:.4e}")

        # Stage 3: rsvd — directly optimal for ||N-N_hat||_F^2
        Theta_rsvd, _ = theta_rsvd(g_gen5, Ni_large, Nk_large, P,
                                    k_batch_size=kbs5, device=device,
                                    oversampling=10, n_power_iter=2)
        Lambda_rsvd = update_lambda(Theta_rsvd, g_gen5, Nk_large,
                                    k_batch_size=kbs5, device=device)
        op_rsvd = compute_operator_error(Theta_rsvd, Lambda_rsvd, g_gen5, Nk_large,
                                         k_batch_size=kbs5, device=device)
        print(f"  [stage 3: rsvd P={P}]          op_err={op_rsvd:.4e}")

        # Stage 4: Adam from QR-orthonormalized time-seg init
        print(f"\n  [stage 4: Adam from time-seg init, print_every=10]")
        run_exp_decomposition_gd(
            Ni=Ni_large, Nk=Nk_large, P=P,
            g_generator=g_gen5,
            label=f"[time-seg+Adam]  {desc5}",
            n_iter=200, k_batch_size=kbs5,
            print_every=10, compute_op_err=True,
            lr_theta=1e-3, lr_lambda=1e-1,
            use_time_seg_init=True,
            device=device, seed=SEED,
        )
