"""
test_offres_nonlin_3d.py — End-to-end accuracy test comparing the low-rank
off-resonance approximation against a direct naive NUFFT.

Setup
-----
- 16×16×16 ball phantom, 8 cm FOV, 5 mm voxels
- 250 koosh-ball spokes × 500 samples → K = 125 000 time points
- Off-resonance: ±200 Hz linear in z
- Nonlinear field: z² spatial term with a BIPOLAR gradient waveform (zero-sum
  per spoke) so that the cumulative integral K_q(t) resets each spoke.

Time model
----------
Each spoke has its own RF excitation that resets the transverse magnetisation.
The off-resonance phase therefore accumulates from t=0 at the START OF EACH
SPOKE, not from the start of the entire acquisition.

To be consistent with `make_phi_matvec` (which uses cumsum over the full
waveform), both the time vector and the nonlinear waveform must be PERIODIC
with period n_samp:
  t[s*n_samp + j]       = j * dt           (within-spoke time)
  G_nl[s*n_samp + j]    = G_nl_spoke[j]    (repeats each spoke)

The bipolar G_nl_spoke sums to zero, so cumsum(G_nl) also resets to zero at
the start of each new spoke. This makes K_q[k] = f(k % n_samp), consistent
with the per-spoke time model.

Peak phases (2 ms readout):
  Off-resonance (±200 Hz): ≈ ±2.5 rad
  Nonlinear (z², bipolar): ≈ ±π/4 rad at echo centre
  → rank L=12 gives < 1% approximation error.

Run from repo root:
    python -m python.tests.offres_nonlin.test_offres_nonlin_3d
"""

import math
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

from hastycompute.mri.offres_nonlin import make_phi_matvec, phi_lowrank


def report(name: str, err: float, tol: float):
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}]  {name:55s}  rel err = {err:.3e}  (tol={tol:.0e})")


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def make_3d_problem(n_dim: int = 16, n_spokes: int = 250,
                    n_samp: int = 500, device: str = 'cpu') -> dict:
    """
    3-D MRI problem with off-resonance and a z² nonlinear field term.

    Time convention: t resets at the start of each spoke.
    Waveform convention: G_nl is bipolar (zero-sum per spoke) so that
    cumsum(G_nl) is also periodic → K_q resets each spoke.
    """
    gamma = 2 * math.pi * 42.577e6   # ¹H (rad s⁻¹ T⁻¹)
    dt    = 4e-6                      # 4 µs dwell time
    FOV   = 0.08                      # 8 cm
    dx    = FOV / n_dim
    K     = n_spokes * n_samp

    # ---- Voxel grid ----
    xi = torch.linspace(-FOV/2 + dx/2, FOV/2 - dx/2, n_dim,
                        dtype=torch.float32, device=device)
    X, Y, Z = torch.meshgrid(xi, xi, xi, indexing='ij')
    coords  = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    N = coords.shape[0]

    # ---- Ball phantom ----
    phantom = (coords.norm(dim=1) < FOV * 0.4).to(torch.complex64)

    # ---- 3-D koosh-ball k-space trajectory (rad/m) ----
    k_max = math.pi / dx    # Nyquist: π/dx rad/m

    idx   = torch.arange(n_spokes, dtype=torch.float32, device=device)
    theta = torch.arccos(1.0 - 2.0 * idx / n_spokes)
    phi_a = idx * math.pi * (3.0 - math.sqrt(5.0))
    dirs  = torch.stack([
        torch.sin(theta) * torch.cos(phi_a),
        torch.sin(theta) * torch.sin(phi_a),
        torch.cos(theta),
    ], dim=1)  # (n_spokes, 3)

    k_r    = torch.linspace(-k_max, k_max, n_samp, dtype=torch.float32, device=device)
    k_traj = (dirs.unsqueeze(1) * k_r.unsqueeze(0).unsqueeze(-1)).reshape(-1, 3)

    # ---- Per-spoke time vector ----
    # t[s*n_samp + j] = j*dt  (resets at the start of each spoke)
    t_spoke = torch.arange(n_samp, dtype=torch.float32, device=device) * dt
    t = t_spoke.unsqueeze(0).expand(n_spokes, n_samp).reshape(-1).contiguous()

    # ---- Off-resonance: ±200 Hz linear in z ----
    # Peak imaginary phase over 2 ms readout: 200·2π·0.002 ≈ 2.51 rad
    dB0_hz = 200.0 * coords[:, 2] / (FOV / 2)
    T2     = 0.06 * torch.ones(N, dtype=torch.float32, device=device)
    z_map  = (1.0 / T2 + 1j * 2 * math.pi * dB0_hz).to(torch.complex64).to(device)

    # ---- Nonlinear field: z² term with bipolar gradient ----
    # Bipolar waveform: first half +G_amp, second half -G_amp → sum = 0 per spoke.
    # cumsum(G_nonlin) therefore also resets to 0 at each spoke boundary, so
    # K_q(t) = iγ·dt·cumsum(G_nonlin) is periodic with the same period as t.
    #
    # Peak K_q occurs at mid-spoke (j = n_samp//2):
    #   |K_q|_peak = γ·G_amp·dt·(n_samp/2)
    # We set the resulting phase amplitude to π/4:
    #   γ·G_amp·dt·(n_samp/2)·z_max² = π/4
    z_max_sq = (FOV / 2) ** 2
    half     = n_samp // 2
    G_nl_amp = (math.pi / 4) / (gamma * dt * half * z_max_sq)

    G_nl_spoke  = torch.cat([
        G_nl_amp * torch.ones(half, dtype=torch.float32, device=device),
        -G_nl_amp * torch.ones(n_samp - half, dtype=torch.float32, device=device),
    ])                                                      # (n_samp,) zero-sum
    G_nonlin = G_nl_spoke.repeat(n_spokes)                 # (K,) periodic

    spatial_maps     = (coords[:, 2] ** 2).unsqueeze(1).float()   # (N, 1) m²
    nonlin_waveforms = {'z_sq': G_nonlin}
    waveform_order   = ['z_sq']

    return dict(
        N=N, K=K, coords=coords, phantom=phantom,
        k_traj=k_traj, t=t,
        z_map=z_map, spatial_maps=spatial_maps,
        nonlin_waveforms=nonlin_waveforms, waveform_order=waveform_order,
        gamma=gamma, dt=dt, k_max=k_max, dx=dx, FOV=FOV,
        n_spokes=n_spokes, n_samp=n_samp,
    )


# ---------------------------------------------------------------------------
# Naive NUFFT
# ---------------------------------------------------------------------------

def naive_nufft(image: torch.Tensor,
                coords: torch.Tensor,
                k_vec: torch.Tensor) -> torch.Tensor:
    """
    Direct O(N) NUFFT for a single k-space location:

        S(k) = Σ_j image[j] · exp(-i k · r_j)

    Args:
        image:  (N,) complex.
        coords: (N, 3) float — voxel positions (m).
        k_vec:  (3,)  float — k-space coordinate (rad/m).
    """
    phase = (coords * k_vec.unsqueeze(0)).sum(-1)
    return (image * torch.exp(-1j * phase.to(image.dtype))).sum()


# ---------------------------------------------------------------------------
# Exact vs approximate signal
# ---------------------------------------------------------------------------

def compute_signals(sample_idx: list,
                    prob: dict,
                    Omega: torch.Tensor,
                    Upsilon: torch.Tensor,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact and approximate k-space signals at the given sample indices.

    Exact:
        S[m] = Σ_j ρ_j · exp(Φ_j(t_k)) · exp(-i k_t · r_j)

    Φ is computed using the SAME K_q = iγ·dt·cumsum(G_nonlin) formula as
    make_phi_matvec, so the two branches are guaranteed consistent.

    Approx:
        S[m] ≈ Σ_l Ω[k,l] · Σ_j Υ[j,l] · ρ_j · exp(-i k_t · r_j)
    """
    coords           = prob['coords']
    phantom          = prob['phantom']
    k_traj           = prob['k_traj']
    z_map            = prob['z_map']
    sm               = prob['spatial_maps'].to(torch.complex64)
    nonlin_waveforms = prob['nonlin_waveforms']
    waveform_order   = prob['waveform_order']
    t                = prob['t']
    gamma            = prob['gamma']
    dt               = prob['dt']

    if waveform_order:
        K_q = torch.stack([
            1j * gamma * dt
            * torch.cumsum(nonlin_waveforms[key].to(torch.complex64), dim=0)
            for key in waveform_order
        ], dim=0)   # (Q, K)
    else:
        K_q = None

    M        = len(sample_idx)
    S_exact  = torch.zeros(M, dtype=torch.complex64)
    S_approx = torch.zeros(M, dtype=torch.complex64)

    for i, k in enumerate(sample_idx):
        k_vec   = k_traj[k]
        phase   = (coords * k_vec.unsqueeze(0)).sum(-1)
        exp_ikr = torch.exp(-1j * phase.to(torch.complex64))

        # Exact Φ(r, t_k)
        phi_j = -z_map * t[k].to(torch.complex64)
        if K_q is not None:
            phi_j = phi_j - (sm @ K_q[:, k])
        exp_phi_j = torch.exp(phi_j)

        S_exact[i] = (phantom * exp_phi_j * exp_ikr).sum()

        # Approx: Ω[k] ⊙ (Υᵀ @ (ρ · exp(-ik·r)))
        nufft_basis = (Upsilon * (phantom * exp_ikr).unsqueeze(1)).sum(0)  # (L,)
        S_approx[i] = (Omega[k] * nufft_basis).sum()

    return S_exact, S_approx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nufft_sanity(prob: dict):
    """NUFFT at k=0 must equal the total image mass (no phase modulation)."""
    print("\n[naive NUFFT — sanity]")
    S0   = naive_nufft(prob['phantom'], prob['coords'],
                       torch.zeros(3, dtype=torch.float32))
    mass = prob['phantom'].sum()
    err  = (S0 - mass).abs() / mass.abs()
    report("NUFFT(k=0) = Σ_j ρ_j", err.item(), tol=1e-5)


def test_signal_accuracy(prob: dict, L: int = 12, n_sample: int = 100):
    """
    Build the rank-L decomposition and compare exact vs approximate k-space
    signals at n_sample randomly chosen points.
    """
    print(f"\n[signal accuracy  L={L}  n_sample={n_sample}]")
    N, K   = prob['N'], prob['K']
    device = prob['z_map'].device

    fwd, adj = make_phi_matvec(
        prob['z_map'], prob['spatial_maps'],
        prob['nonlin_waveforms'], prob['waveform_order'],
        prob['t'], prob['gamma'], prob['dt'],
    )

    print(f"  Building rank-{L} decomposition  (N={N}, K={K}) …", flush=True)
    Omega, Upsilon = phi_lowrank(
        fwd, adj, N, K, L,
        n_oversampling=10, n_power_iter=2,
        dtype=torch.complex64, device=device,
    )
    print(f"  Omega {tuple(Omega.shape)},  Upsilon {tuple(Upsilon.shape)}")

    torch.manual_seed(42)
    idx = torch.randperm(K)[:n_sample].tolist()

    print(f"  Evaluating {n_sample} exact signals …", flush=True)
    S_exact, S_approx = compute_signals(idx, prob, Omega, Upsilon)

    err = (S_exact - S_approx).norm() / S_exact.norm()
    report("approx vs exact k-space signal", err.item(), tol=0.05)
    return Omega, Upsilon, err.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cpu'

    print("=" * 75)
    print("  3-D off-resonance approximation — end-to-end signal accuracy test")
    print("=" * 75)

    prob = make_3d_problem(n_dim=16, n_spokes=250, n_samp=500, device=device)
    t_max_ms = prob['n_samp'] * prob['dt'] * 1e3
    print(f"\n  N = {prob['N']} voxels,  K = {prob['K']} time points")
    print(f"  per-spoke readout = {t_max_ms:.1f} ms,  k_max = {prob['k_max']:.1f} rad/m")
    print(f"  off-resonance ±200 Hz → ±{200*2*math.pi*t_max_ms*1e-3:.2f} rad")
    print(f"  nonlinear z² (bipolar) → ±π/4 rad peak")

    test_nufft_sanity(prob)
    Omega, Upsilon, err = test_signal_accuracy(prob, L=12, n_sample=100)

    print("\n" + "=" * 75)
    print(f"  Signal approximation error: {err:.3e}")
    print("=" * 75)
