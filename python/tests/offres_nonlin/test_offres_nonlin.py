"""
test_offres_nonlin.py — Numerical verification of the off-resonance /
nonlinear-gradient normal-operator approximation.

Run from the repo root:
    python -m python.tests.offres_nonlin.test_offres_nonlin

What it does
------------
1.  Builds a synthetic 1-D (N voxels) problem with a complex rate map z(r),
    arbitrary (non-harmonic) spatial basis functions, and sinusoidal gradient
    waveforms.
2.  Verifies the gradient convolution and k-space trajectory shapes.
3.  Tests the phi matvec operators (fwd, adj) via an adjoint consistency check.
4.  Verifies the randomized SVD via a sample-based reconstruction error.
5.  Checks the normal-operator approximation accuracy.
6.  Runs the full pipeline and prints output shapes.
"""

import math
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

from hastycompute.mri.offres_nonlin import (
    convolve_gradients,
    kspace_trajectory,
    make_phi_matvec,
    phi_lowrank,
    normal_op_lowrank,
    offres_nonlin_approx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_kernel(length: int, sigma: float, device='cpu') -> torch.Tensor:
    """Unit-area Gaussian impulse response."""
    x = torch.arange(length, dtype=torch.float32, device=device) - length // 2
    h = torch.exp(-0.5 * (x / sigma) ** 2)
    return h / h.sum()


def report(name: str, err_rel: float, tol: float):
    status = "PASS" if err_rel < tol else "FAIL"
    print(f"  [{status}]  {name:50s}  rel err = {err_rel:.3e}  (tol={tol:.0e})")


# ---------------------------------------------------------------------------
# Synthetic problem setup
# ---------------------------------------------------------------------------

def make_problem(N: int = 128, K: int = 256, device: str = 'cpu') -> dict:
    """
    Create synthetic inputs.  Spatial basis functions are arbitrary polynomials
    in r — not solid harmonics — to confirm the library is basis-agnostic.

    Realistic MRI regime:
      - T2 = 50–80 ms, readout = 1 ms  → weak T2 decay
      - Off-resonance ±200 Hz           → dominant imaginary Φ
    """
    gamma = 2 * math.pi * 42.577e6   # ¹H gyromagnetic ratio (rad/s/T)
    dt    = 4e-6                      # 4 µs dwell time

    t = torch.arange(K, dtype=torch.float32, device=device) * dt
    r = torch.linspace(-0.1, 0.1, N, dtype=torch.float32, device=device)

    # Rate map z(r) = 1/T2(r) + iγ ΔB0(r)
    T2     = 0.05 + 0.03 * torch.sin(math.pi * r / r.max())
    dB0_hz = 200.0 * r / r.abs().max()
    z_map  = (1.0 / T2 + 1j * 2 * math.pi * dB0_hz).to(torch.complex64)

    # Arbitrary polynomial spatial basis: [r, r²]  (not spherical harmonics)
    f0 = r.unsqueeze(1)           # (N, 1) — linear in r
    f1 = (r ** 2).unsqueeze(1)    # (N, 1) — quadratic in r
    spatial_maps = torch.cat([f0, f1], dim=1).float()    # (N, 2)

    # Ideal gradient waveforms
    freq    = 1.0 / (K * dt)
    G_ideal = {
        'x': 1e-3 * torch.sin(2 * math.pi * freq * t),
        'y': 5e-4 * torch.cos(2 * math.pi * freq * t),
        'z': torch.zeros(K, dtype=torch.float32, device=device),
    }

    # Impulse-response kernels — keyed so last element is the axis.
    # Nonlinear keys can be any hashable type; here we use descriptive tuples.
    kern_len = 32
    sigma    = 5.0
    kern     = gaussian_kernel(kern_len, sigma, device=device)
    kernels  = {
        (0, 0, 'x'): kern.clone(),
        (0, 0, 'y'): kern.clone(),
        (0, 0, 'z'): kern.clone(),
        ('nonlin_x', 'x'): gaussian_kernel(kern_len, sigma * 0.8, device=device) * 0.1,
        ('nonlin_y', 'y'): gaussian_kernel(kern_len, sigma * 1.2, device=device) * 0.05,
    }

    # waveform_order: maps column q of spatial_maps to a key in nonlin_waveforms.
    # Must use the exact same keys that appear in the (filtered) convolved dict.
    waveform_order = [('nonlin_x', 'x'), ('nonlin_y', 'y')]

    return dict(
        gamma=gamma, dt=dt, t=t, r=r,
        z_map=z_map, spatial_maps=spatial_maps,
        waveform_order=waveform_order,
        G_ideal=G_ideal, kernels=kernels,
        N=N, K=K,
    )


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_convolution(prob: dict):
    print("\n[convolution]")
    convolved = convolve_gradients(prob['kernels'], prob['G_ideal'])
    for key, wfm in convolved.items():
        axis = key[-1]
        assert wfm.shape == prob['G_ideal'][axis].shape, f"Shape mismatch for {key}"
    print("  [PASS]  output shapes match input waveform length")
    return convolved


def test_kspace(prob: dict, convolved: dict):
    print("\n[k-space trajectory]")
    linear = {key[-1]: convolved[key] for key in convolved if key[0] == 0 and key[1] == 0}
    k_traj = kspace_trajectory(linear, prob['gamma'], prob['dt'])
    assert k_traj.shape == (prob['K'], 3), f"Expected ({prob['K']}, 3), got {k_traj.shape}"
    print(f"  [PASS]  k_traj shape = {tuple(k_traj.shape)},  "
          f"max |k| = {k_traj.abs().max().item():.3e} m⁻¹")
    return k_traj


def test_phi_matvec(prob: dict, fwd, adj):
    """
    Adjoint consistency test:  <y, A x> = <A^H y, x>
    This validates both fwd and adj without forming the full matrix.
    """
    print("\n[phi matvec — adjoint consistency]")
    N, K = prob['N'], prob['K']
    cdtype = torch.complex64
    device = prob['z_map'].device

    torch.manual_seed(0)
    x = (torch.randn(N, 1, dtype=torch.float32, device=device)
         + 1j * torch.randn(N, 1, dtype=torch.float32, device=device)).to(cdtype)
    y = (torch.randn(K, 1, dtype=torch.float32, device=device)
         + 1j * torch.randn(K, 1, dtype=torch.float32, device=device)).to(cdtype)

    Ax  = fwd(x)   # (K, 1)
    AHy = adj(y)   # (N, 1)

    lhs = (y.conj() * Ax).sum()
    rhs = (AHy.conj() * x).sum()
    err = (lhs - rhs).abs() / lhs.abs()
    report("<y, Ax> = <A^H y, x>", err.item(), tol=1e-5)
    return err.item()


def test_phi_lowrank(prob: dict, fwd, adj, L: int = 12):
    """
    Verify exp(Φ) x ≈ Ω Υᵀ x on a batch of random vectors.
    No full K×N matrix is formed.
    """
    print(f"\n[phi_lowrank  L={L}]")
    N, K = prob['N'], prob['K']
    cdtype = torch.complex64
    device = prob['z_map'].device

    Omega, Upsilon = phi_lowrank(
        fwd, adj, N, K, L,
        n_oversampling=10, n_power_iter=2,
        dtype=cdtype, device=device,
    )
    assert Omega.shape   == (K, L), f"Omega shape: {Omega.shape}"
    assert Upsilon.shape == (N, L), f"Upsilon shape: {Upsilon.shape}"

    # Sample-based reconstruction error: compare fwd(X) vs Omega @ (Upsilon.T @ X)
    torch.manual_seed(1)
    n_probe = 8
    X = (torch.randn(N, n_probe, dtype=torch.float32, device=device)
         + 1j * torch.randn(N, n_probe, dtype=torch.float32, device=device)).to(cdtype)

    E_exact  = fwd(X)                          # (K, n_probe)
    E_approx = Omega @ (Upsilon.T @ X)         # (K, n_probe)

    err = (E_exact - E_approx).norm() / E_exact.norm()
    report("exp(Φ) x  ≈  Ω Υᵀ x", err.item(), tol=0.05)
    return Omega, Upsilon, err.item()


def test_normal_op(
    Omega: torch.Tensor, Upsilon: torch.Tensor,
    P: int = 6, R: int = 3,
):
    """
    Verify  exp(Φ_j + Φ*_i) ≈ Σ_{p,r} Λ_{kp} Θ_L_{jrp} Θ_R*_{irp}
    using a probe-vector approach instead of forming the full (K, N, N) tensor.

    We pick a small number of (j, i) pairs and compare the k-time series directly.
    """
    print(f"\n[normal_op_lowrank  P={P}  R={R}]")
    K, L = Omega.shape
    N    = Upsilon.shape[0]
    device = Omega.device

    Lambda, Theta_L, Theta_R = normal_op_lowrank(Omega, Upsilon, P, R)
    assert Lambda.shape  == (K, P),    f"Lambda:  {Lambda.shape}"
    assert Theta_L.shape == (N, R, P), f"Theta_L: {Theta_L.shape}"
    assert Theta_R.shape == (N, R, P), f"Theta_R: {Theta_R.shape}"

    # Exact exp(Φ_j + Φ*_i) at a few probe voxel pairs via the rank-L approximation.
    # Reconstruction: exp(Φ) ≈ Ω Υᵀ, so exp(Φ[k,j]) ≈ (Ω Υᵀ)[k,j]
    n_probe = min(16, N)
    idx_j   = torch.arange(n_probe, device=device)
    idx_i   = torch.arange(n_probe, device=device).flip(0)

    E = Omega @ Upsilon.T                          # (K, N), rank-L approx of exp(Φ)
    exact   = E[:, idx_j] * E[:, idx_i].conj()    # (K, n_probe)

    approx  = torch.einsum(
        'kp,jrp,jrp->kj',
        Lambda,
        Theta_L[:, :, :][ idx_j],
        Theta_R[:, :, :][ idx_i].conj(),
    )  # (K, n_probe)

    err = (exact - approx).norm() / exact.norm()
    report("normal-op Λ Θ_L Θ_R* vs Ω Υᵀ", err.item(), tol=0.15)
    return Lambda, Theta_L, Theta_R, err.item()


def test_full_pipeline(prob: dict, L: int = 12, P: int = 6, R: int = 3):
    print(f"\n[full pipeline  L={L}  P={P}  R={R}]")
    nonlin_waveforms_keys = prob['waveform_order']

    out = offres_nonlin_approx(
        kernels=prob['kernels'],
        ideal_waveforms=prob['G_ideal'],
        z_map=prob['z_map'],
        spatial_maps=prob['spatial_maps'],
        waveform_order=nonlin_waveforms_keys,
        t=prob['t'],
        gamma=prob['gamma'],
        dt=prob['dt'],
        L=L, P=P, R=R,
    )
    for key, val in out.items():
        print(f"  {key:10s}  shape={tuple(val.shape)}  dtype={val.dtype}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    device = 'cpu'

    print("=" * 70)
    print("  Off-resonance / nonlinear gradient approximation — tests")
    print("=" * 70)

    prob = make_problem(N=128, K=256, device=device)

    convolved = test_convolution(prob)
    k_traj    = test_kspace(prob, convolved)

    # Build phi matvec operators from convolved nonlinear waveforms
    nonlin_waveforms = {
        key: val for key, val in convolved.items()
        if not (key[0] == 0 and key[1] == 0)
    }
    fwd, adj = make_phi_matvec(
        prob['z_map'], prob['spatial_maps'],
        nonlin_waveforms, prob['waveform_order'],
        prob['t'], prob['gamma'], prob['dt'],
    )

    adj_err              = test_phi_matvec(prob, fwd, adj)
    L, P, R              = 12, 6, 3
    Omega, Upsilon, err_phi = test_phi_lowrank(prob, fwd, adj, L=L)
    Lambda, Theta_L, Theta_R, err_nop = test_normal_op(Omega, Upsilon, P=P, R=R)
    out = test_full_pipeline(prob, L=L, P=P, R=R)

    print("\n" + "=" * 70)
    print(f"  Adjoint consistency error    : {adj_err:.3e}")
    print(f"  exp(Φ) reconstruction error  : {err_phi:.3e}")
    print(f"  Normal-op approximation error: {err_nop:.3e}")
    print("=" * 70)
