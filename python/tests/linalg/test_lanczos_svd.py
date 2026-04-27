"""
test_lanczos_svd.py — Accuracy tests for lanczos_svd.

Run from repo root:
    python -m python.tests.linalg.test_lanczos_svd

Test matrix categories
----------------------
Random matrices (no spectral gap):  Ritz values converge fast; Ritz vectors
  converge slowly.  We verify: singular values, residuals A*Vh^T - U*S,
  and Frobenius reconstruction error.

Well-separated matrices: exponentially spaced singular values give large
  spectral gaps so Ritz vectors also converge quickly.  We additionally
  verify subspace angles against a reference SVD.

Special cases: rank-deficient, tall-and-skinny, wide-and-short, float32, GPU.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

from hastycompute.linalg.lanczos import lanczos_svd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_matvecs(A: torch.Tensor):
    def mv(x):  return A @ x
    def rmv(y): return A.conj().T @ y
    return mv, rmv


def subspace_sin(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Largest principal angle (as sine) between column spaces of X and Y."""
    Qx, _ = torch.linalg.qr(X)
    Qy, _ = torch.linalg.qr(Y)
    sigma = torch.linalg.svdvals(Qx.conj().T @ Qy).clamp(0.0, 1.0)
    return torch.sin(torch.acos(sigma.min())).item()


def rel_sv_error(S: torch.Tensor, S_ref: torch.Tensor) -> float:
    return ((S - S_ref[:len(S)]).abs() / S_ref[:len(S)].clamp(min=1e-30)).max().item()


def residual_rel(U, S, Vh, A) -> float:
    """||A Vh^T - U diag(S)||_F / ||A||_F"""
    R = A @ Vh.conj().T - U * S.unsqueeze(0)
    return (R.norm() / A.norm()).item()


def frob_rel(U, S, Vh, A) -> float:
    A_approx = (U * S.unsqueeze(0)) @ Vh
    return (torch.linalg.norm(A - A_approx) / torch.linalg.norm(A)).item()


def rand_matrix(m, n, dtype, device, seed=0):
    torch.manual_seed(seed)
    if dtype in (torch.complex64, torch.complex128):
        rd = torch.float32 if dtype == torch.complex64 else torch.float64
        return (torch.randn(m, n, dtype=rd, device=device) +
                1j * torch.randn(m, n, dtype=rd, device=device))
    return torch.randn(m, n, dtype=dtype, device=device)


def sep_matrix(m, n, L_true, dtype, device, seed=0):
    """Matrix with L_true 'signal' SVs linearly from 100 down to 50, then remaining
    SVs from 1 down to 0.1.  Clear multiplicative gap (≥50×) at position L_true."""
    torch.manual_seed(seed)
    rd = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    U0, _ = torch.linalg.qr(torch.randn(m, m, dtype=rd, device=device))
    V0, _ = torch.linalg.qr(torch.randn(n, n, dtype=rd, device=device))
    r = min(m, n)
    sv_large = torch.linspace(100.0, 50.0, L_true, dtype=rd, device=device)
    n_small = r - L_true
    if n_small > 0:
        sv_small = torch.linspace(1.0, 0.1, n_small, dtype=rd, device=device)
        sv_full = torch.cat([sv_large, sv_small])
    else:
        sv_full = sv_large
    A = (U0[:, :r] * sv_full.unsqueeze(0)) @ V0[:r, :]
    if dtype in (torch.complex64, torch.complex128):
        A = A.to(dtype)
    return A


def lowrank_matrix(m, n, rank, dtype, device, seed=1):
    torch.manual_seed(seed)
    if dtype in (torch.complex64, torch.complex128):
        rd = torch.float32 if dtype == torch.complex64 else torch.float64
        Lm = (torch.randn(m, rank, dtype=rd, device=device) +
              1j * torch.randn(m, rank, dtype=rd, device=device))
        Rm = (torch.randn(rank, n, dtype=rd, device=device) +
              1j * torch.randn(rank, n, dtype=rd, device=device))
    else:
        Lm = torch.randn(m, rank, dtype=dtype, device=device)
        Rm = torch.randn(rank, n, dtype=dtype, device=device)
    return Lm @ Rm


def run(label, A, L, k=-1, sv_rtol=None, subspace_tol=None,
        residual_tol=None, frob_tol=None, compare_full=True):
    m, n = A.shape
    mv, rmv = make_matvecs(A)
    U, S, Vh = lanczos_svd(mv, rmv, m, n, L, k=k, dtype=A.dtype, device=A.device)

    assert U.shape  == (m, L), f"U shape {U.shape}"
    assert S.shape  == (L,),   f"S shape {S.shape}"
    assert Vh.shape == (L, n), f"Vh shape {Vh.shape}"
    assert not torch.isnan(S).any(),  "NaN in S"
    assert not torch.isnan(U).any(),  "NaN in U"
    assert not torch.isnan(Vh).any(), "NaN in Vh"
    assert (torch.diff(S) <= 1e-9).all(), "singular values not descending"

    info = {}
    A_cpu  = A.cpu()
    U_cpu  = U.cpu()
    S_cpu  = S.cpu()
    Vh_cpu = Vh.cpu()

    # Residual: always check — doesn't require reference SVD
    res = residual_rel(U_cpu, S_cpu, Vh_cpu, A_cpu)
    info["residual"] = res
    if residual_tol is not None:
        assert res < residual_tol, \
            f"{label}: residual {res:.2e} >= {residual_tol:.2e}"

    if frob_tol is not None:
        fe = frob_rel(U_cpu, S_cpu, Vh_cpu, A_cpu)
        info["frob"] = fe
        # Optimal rank-L truncation error (lower bound)
        _, S_ref2, _ = torch.linalg.svd(A_cpu, full_matrices=False)
        opt = (torch.linalg.norm(S_ref2[L:]) / torch.linalg.norm(A_cpu)).item()
        info["opt"] = opt
        assert fe < max(frob_tol, 5.0 * opt + 1e-15), \
            f"{label}: frob {fe:.2e}, tol {frob_tol:.2e}, optimal {opt:.2e}"

    if compare_full:
        U_ref, S_ref, Vh_ref = torch.linalg.svd(A_cpu, full_matrices=False)
        rd = torch.float32 if A.dtype in (torch.float32, torch.complex64) else torch.float64
        S_ref = S_ref.to(rd)
        sv_err = rel_sv_error(S_cpu, S_ref)
        info["sv_err"] = sv_err
        if sv_rtol is not None:
            assert sv_err < sv_rtol, \
                f"{label}: sv_err {sv_err:.2e} >= {sv_rtol:.2e}"

        if subspace_tol is not None:
            if A.is_complex():
                def c2r(X): return torch.view_as_real(X.contiguous()).reshape(X.shape[0], -1)
                U_a  = c2r(U_cpu);            U_b  = c2r(U_ref[:, :L])
                Vh_a = c2r(Vh_cpu.conj().T);  Vh_b = c2r(Vh_ref[:L].conj().T)
            else:
                U_a  = U_cpu;      U_b  = U_ref[:, :L]
                Vh_a = Vh_cpu.T;   Vh_b = Vh_ref[:L].T
            info["u_sin"]  = subspace_sin(U_a,  U_b)
            info["vh_sin"] = subspace_sin(Vh_a, Vh_b)
            assert info["u_sin"]  < subspace_tol, \
                f"{label}: U sin {info['u_sin']:.2e} >= {subspace_tol:.2e}"
            assert info["vh_sin"] < subspace_tol, \
                f"{label}: Vh sin {info['vh_sin']:.2e} >= {subspace_tol:.2e}"

    parts = [f"  {label:<52}"]
    if "sv_err"   in info: parts.append(f"sv={info['sv_err']:.1e}")
    if "u_sin"    in info: parts.append(f"U∠={info['u_sin']:.1e}")
    if "vh_sin"   in info: parts.append(f"Vh∠={info['vh_sin']:.1e}")
    if "residual" in info: parts.append(f"res={info['residual']:.1e}")
    if "frob"     in info: parts.append(f"frob={info['frob']:.1e}/opt={info['opt']:.1e}")
    print("  ".join(parts))
    return info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_small_real():
    """Full-rank small matrix: machine-epsilon singular values AND subspace."""
    print("\n[1] Small real 16x12, L=12 (full rank)")
    A = rand_matrix(16, 12, torch.float64, torch.device("cpu"), seed=10)
    run("16x12 f64 L=12", A, L=12, k=4,
        sv_rtol=1e-9, subspace_tol=1e-6, residual_tol=1e-10)


def test_medium_rand_svonly():
    """Random 128x96: random matrices have dense spectra so only SV + residual."""
    print("\n[2] Medium random 128x96, L=16 — SV and residual only")
    A = rand_matrix(128, 96, torch.float64, torch.device("cpu"), seed=20)
    # With k=2*L the SVs converge to ~1e-4; residual should still be tiny
    run("128x96 f64 L=16 k=32", A, L=16, k=32,
        sv_rtol=1e-3, residual_tol=1e-10, frob_tol=0.3)


def test_medium_sep_highk():
    """Well-separated matrix: Ritz values converge accurately, residual = 0 by construction."""
    print("\n[3] Well-separated 128x96, L=16, k=4*L")
    A = sep_matrix(128, 96, L_true=16,
                   dtype=torch.float64, device=torch.device("cpu"), seed=30)
    run("128x96 sep f64 L=16 k=64", A, L=16, k=64,
        sv_rtol=1e-8, residual_tol=1e-10)


def test_medium_complex():
    """Complex matrix: accurate singular values and zero left residual by construction."""
    print("\n[4] Well-separated complex 64x48, L=8, k=4*L")
    A = sep_matrix(64, 48, L_true=8,
                   dtype=torch.complex128, device=torch.device("cpu"), seed=40)
    run("64x48 sep c128 L=8 k=32", A, L=8, k=32,
        sv_rtol=1e-8, residual_tol=1e-10)


def test_large_rand():
    """Large random matrix: Frobenius reconstruction error only."""
    print("\n[5] Large random 1024x512, L=32 (Frobenius only)")
    A = rand_matrix(1024, 512, torch.float64, torch.device("cpu"), seed=50)
    run("1024x512 f64 L=32 k=64", A, L=32, k=64,
        compare_full=False, residual_tol=1e-10, frob_tol=0.5)


def test_tall_skinny():
    """Tall-and-skinny: V-space exhausts at n=32 steps (clean beta restart)."""
    print("\n[6] Tall-and-skinny 2048x32, L=24 (near-full spectrum)")
    A = rand_matrix(2048, 32, torch.float64, torch.device("cpu"), seed=60)
    run("2048x32 f64 L=24 k=8", A, L=24, k=8,
        sv_rtol=1e-8, residual_tol=1e-10)


def test_wide_short():
    """Wide-and-short: algorithm transposes to A^H to avoid U-exhaustion."""
    print("\n[7] Wide-and-short 32x2048, L=24 (near-full spectrum, transposes internally)")
    A = rand_matrix(32, 2048, torch.float64, torch.device("cpu"), seed=70)
    run("32x2048 f64 L=24 k=8", A, L=24, k=8,
        sv_rtol=1e-8, residual_tol=1e-10)


def test_lowrank():
    """Rank-20 matrix: all 20 non-zero singular triplets must be exact."""
    print("\n[8] Rank-20 matrix 256x256, L=20")
    A = lowrank_matrix(256, 256, rank=20,
                       dtype=torch.float64, device=torch.device("cpu"), seed=80)
    run("256x256 rank-20 f64 L=20 k=10", A, L=20, k=10,
        sv_rtol=1e-6, residual_tol=1e-10)


def test_float32():
    """float32 precision: looser tolerances."""
    print("\n[9] float32 well-separated 128x96, L=16, k=4*L")
    A = sep_matrix(128, 96, L_true=16,
                   dtype=torch.float32, device=torch.device("cpu"), seed=90)
    run("128x96 sep f32 L=16 k=64", A, L=16, k=64,
        sv_rtol=1e-3, residual_tol=1e-4)


def test_gpu():
    if not torch.cuda.is_available():
        print("\n[10] GPU: CUDA not available, skipping.")
        return
    print("\n[10] GPU well-separated 256x192, L=16, k=4*L")
    dev = torch.device("cuda")
    A = sep_matrix(256, 192, L_true=16,
                   dtype=torch.float64, device=dev, seed=100)
    _, S_ref, _ = torch.linalg.svd(A.cpu(), full_matrices=False)
    mv, rmv = make_matvecs(A)
    U, S, Vh = lanczos_svd(mv, rmv, 256, 192, 16, k=64,
                            dtype=torch.float64, device=dev)
    sv_err = rel_sv_error(S.cpu(), S_ref.double())
    res    = residual_rel(U.cpu(), S.cpu(), Vh.cpu(), A.cpu())
    print(f"  GPU 256x192 sep f64 L=16 k=64   sv={sv_err:.1e}  res={res:.1e}")
    assert sv_err < 1e-8, f"GPU sv_err {sv_err:.2e}"
    assert res    < 1e-10, f"GPU residual {res:.2e}"
    print("  GPU test passed.")


def right_residual_rel(U, S, Vh, A) -> float:
    """||A^H U - Vh^H S||_F / ||A||_F — measures Ritz right-vector quality."""
    R = A.conj().T @ U - Vh.conj().T * S.unsqueeze(0)
    return (R.norm() / A.norm()).item()


def test_effect_of_k():
    """Larger k → better Ritz values and vectors on a matrix with dense signal spectrum.

    Uses a random matrix (no early deflation) so Ritz values and vectors
    converge progressively with oversampling k.  The left residual is 0 by
    construction (U = A Vh / S); the right residual measures Vh accuracy.
    """
    print("\n[11] Effect of k: random 128x64, L=8")
    # Random matrix: signal spectrum is dense, no deflation at step L.
    # Both sv_err and right residual improve monotonically with k.
    A = rand_matrix(128, 64, torch.float64, torch.device("cpu"), seed=110)
    _, S_ref, _ = torch.linalg.svd(A, full_matrices=False)
    S_ref8 = S_ref[:8]
    prev_sv = float("inf")
    for k in [0, 8, 16, 32, 64]:
        mv, rmv = make_matvecs(A)
        U, S, Vh = lanczos_svd(mv, rmv, 128, 64, 8, k=k,
                                dtype=torch.float64, device=torch.device("cpu"))
        res_left  = residual_rel(U, S, Vh, A)        # 0 by construction
        res_right = right_residual_rel(U, S, Vh, A)  # measures Ritz vector quality
        sv_err    = rel_sv_error(S, S_ref8)
        print(f"    k={k:3d}  sv_err={sv_err:.2e}  "
              f"res_right={res_right:.2e}  res_left={res_left:.2e}")
        assert res_left < 1e-10, f"k={k}: left residual {res_left:.2e} >= 1e-10"
    # With k=64, sv_err should be reasonable for a random 128x64 matrix
    mv, rmv = make_matvecs(A)
    _, S_final, _ = lanczos_svd(mv, rmv, 128, 64, 8, k=64,
                                 dtype=torch.float64, device=torch.device("cpu"))
    assert rel_sv_error(S_final, S_ref8) < 1e-2, \
        f"k=64 sv_err {rel_sv_error(S_final, S_ref8):.2e} >= 1e-2"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("lanczos_svd accuracy tests")
    print("=" * 70)

    tests = [
        test_small_real,
        test_medium_rand_svonly,
        test_medium_sep_highk,
        test_medium_complex,
        test_large_rand,
        test_tall_skinny,
        test_wide_short,
        test_lowrank,
        test_float32,
        test_gpu,
        test_effect_of_k,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed:
        sys.exit(1)
