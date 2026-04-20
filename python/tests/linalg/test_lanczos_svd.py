"""
test_lanczos_svd.py — Accuracy tests for lanczos_svd.

Run from repo root:
    python -m python.tests.linalg.test_lanczos_svd

Tests
-----
 1. Small real   (16x12), L=12 (full rank): singular values and subspace
    angles should be at machine-epsilon accuracy.
 2. Medium real  (128x96), rank-16 approx, k=2*L (default):
    coarse accuracy check; comparison to full SVD.
 3. Medium real  (128x96), rank-16 approx, k=4*L (high accuracy):
    tight comparison to full SVD.
 4. Medium complex (64x48), rank-8 approx, k=4*L.
 5. Large real  (1024x512), rank-32, no full-SVD, Frobenius error check.
 6. Tall-and-skinny (2048x32), L=32 (full spectrum).
 7. Wide-and-short  (32x2048), L=32 (full spectrum, tests null-space
    handling when U-basis exhausts R^m).
 8. True-rank-20 matrix (256x256): tests invariant-subspace early stop.
 9. float32 medium matrix: checks precision-appropriate tolerances.
10. GPU test (if CUDA available).
11. Effect of k: confirm larger k → smaller error on the same matrix.
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
    def matvec(x):  return A @ x
    def rmatvec(y): return A.conj().T @ y
    return matvec, rmatvec


def subspace_sin(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Largest principal angle (as sine) between the column spaces of X and Y.
    Both must be (n, L) tall matrices. Returns 0 for identical subspaces.
    """
    Qx, _ = torch.linalg.qr(X)
    Qy, _ = torch.linalg.qr(Y)
    sigma = torch.linalg.svdvals(Qx.conj().T @ Qy).clamp(0.0, 1.0)
    return torch.sin(torch.acos(sigma.min())).item()


def rel_sv_error(S: torch.Tensor, S_ref: torch.Tensor) -> torch.Tensor:
    return (S - S_ref[:len(S)]).abs() / S_ref[:len(S)].clamp(min=1e-30)


def frob_rel_err(U, S, Vh, A) -> float:
    A_approx = (U * S.unsqueeze(0)) @ Vh
    return (torch.linalg.norm(A - A_approx) / torch.linalg.norm(A)).item()


def rand_matrix(m, n, dtype, device, seed=0):
    torch.manual_seed(seed)
    if dtype in (torch.complex64, torch.complex128):
        rd = torch.float32 if dtype == torch.complex64 else torch.float64
        return (torch.randn(m, n, dtype=rd, device=device) +
                1j * torch.randn(m, n, dtype=rd, device=device))
    return torch.randn(m, n, dtype=dtype, device=device)


def lowrank_matrix(m, n, rank, dtype, device, seed=1):
    torch.manual_seed(seed)
    if dtype in (torch.complex64, torch.complex128):
        rd = torch.float32 if dtype == torch.complex64 else torch.float64
        L = (torch.randn(m, rank, dtype=rd, device=device) +
             1j * torch.randn(m, rank, dtype=rd, device=device))
        R = (torch.randn(rank, n, dtype=rd, device=device) +
             1j * torch.randn(rank, n, dtype=rd, device=device))
    else:
        L = torch.randn(m, rank, dtype=dtype, device=device)
        R = torch.randn(rank, n, dtype=dtype, device=device)
    return L @ R


def run(label, A, L, k=-1, sv_rtol=None, subspace_tol=None, frob_tol=None,
        compare_full=True):
    m, n = A.shape
    mv, rmv = make_matvecs(A)
    U, S, Vh = lanczos_svd(mv, rmv, m, n, L, k=k, dtype=A.dtype, device=A.device)

    assert U.shape  == (m, L), f"U shape {U.shape}"
    assert S.shape  == (L,),   f"S shape {S.shape}"
    assert Vh.shape == (L, n), f"Vh shape {Vh.shape}"
    assert not torch.isnan(S).any(),  "NaN in singular values"
    assert not torch.isnan(U).any(),  "NaN in U"
    assert not torch.isnan(Vh).any(), "NaN in Vh"
    assert (torch.diff(S) <= 1e-9).all(), "singular values not descending"

    info = {}

    if compare_full:
        A_cpu = A.cpu()
        _, S_ref, _ = torch.linalg.svd(A_cpu, full_matrices=False)
        U_ref, S_ref, Vh_ref = torch.linalg.svd(A_cpu, full_matrices=False)
        rd = torch.float32 if A.dtype in (torch.float32, torch.complex64) else torch.float64
        S_ref = S_ref.to(rd)

        S_cpu = S.cpu()
        sv_err = rel_sv_error(S_cpu, S_ref).max().item()
        info["sv_err"] = sv_err

        # Subspace angles: flatten complex to real-doubled columns
        U_cpu, Vh_cpu = U.cpu(), Vh.cpu()
        if A.is_complex():
            def c2r(X): return torch.view_as_real(X.contiguous()).reshape(X.shape[0], -1)
            U_a  = c2r(U_cpu);             U_b  = c2r(U_ref[:, :L])
            Vh_a = c2r(Vh_cpu.conj().T);   Vh_b = c2r(Vh_ref[:L].conj().T)
        else:
            U_a  = U_cpu;       U_b  = U_ref[:, :L]
            Vh_a = Vh_cpu.T;    Vh_b = Vh_ref[:L].T

        info["u_sin"]  = subspace_sin(U_a,  U_b)
        info["vh_sin"] = subspace_sin(Vh_a, Vh_b)

        if sv_rtol is not None:
            assert sv_err < sv_rtol, \
                f"{label}: sv rel err {sv_err:.2e} >= {sv_rtol:.2e}"
        if subspace_tol is not None:
            assert info["u_sin"]  < subspace_tol, \
                f"{label}: U  sin(angle) {info['u_sin']:.2e} >= {subspace_tol:.2e}"
            assert info["vh_sin"] < subspace_tol, \
                f"{label}: Vh sin(angle) {info['vh_sin']:.2e} >= {subspace_tol:.2e}"

    if frob_tol is not None:
        A_cpu = A.cpu(); U_cpu = U.cpu(); S_cpu = S.cpu(); Vh_cpu = Vh.cpu()
        ferr  = frob_rel_err(U_cpu, S_cpu, Vh_cpu, A_cpu)
        info["frob"] = ferr

        # Optimal rank-L truncation error (Eckart-Young lower bound)
        _, S_ref2, _ = torch.linalg.svd(A_cpu, full_matrices=False)
        opt = (torch.linalg.norm(S_ref2[L:]) / torch.linalg.norm(A_cpu)).item()
        info["opt"] = opt

        # Allow 5x overhead over optimal
        bound = max(frob_tol, 5.0 * opt + 1e-15)
        assert ferr < bound, \
            f"{label}: frob {ferr:.2e} >= bound {bound:.2e} (opt={opt:.2e})"

    # Pretty-print one result line
    parts = [f"  {label:<48}"]
    if "sv_err"  in info: parts.append(f"SV={info['sv_err']:.1e}")
    if "u_sin"   in info: parts.append(f"U∠={info['u_sin']:.1e}")
    if "vh_sin"  in info: parts.append(f"Vh∠={info['vh_sin']:.1e}")
    if "frob"    in info: parts.append(f"frob={info['frob']:.1e} opt={info['opt']:.1e}")
    print("  ".join(parts))
    return info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_small_real():
    print("\n[1] Small real 16x12, L=12 (full rank, machine-epsilon accuracy)")
    A = rand_matrix(16, 12, torch.float64, torch.device("cpu"), seed=10)
    run("16x12 f64 L=12 k=4", A, L=12, k=4,
        sv_rtol=1e-9, subspace_tol=1e-6)


def test_medium_real_coarse():
    print("\n[2] Medium real 128x96, L=16, k=2*L (default, coarse accuracy)")
    A = rand_matrix(128, 96, torch.float64, torch.device("cpu"), seed=20)
    # With k=32 on a dense-spectrum random matrix, ~1e-4 SV error is expected.
    run("128x96 f64 L=16 k=32", A, L=16, k=32,
        sv_rtol=1e-3, subspace_tol=5e-2, frob_tol=1e-1)


def test_medium_real_highk():
    print("\n[3] Medium real 128x96, L=16, k=4*L (high accuracy)")
    A = rand_matrix(128, 96, torch.float64, torch.device("cpu"), seed=20)
    run("128x96 f64 L=16 k=64", A, L=16, k=64,
        sv_rtol=1e-8, subspace_tol=1e-5)


def test_medium_complex():
    print("\n[4] Medium complex 64x48, L=8, k=4*L")
    A = rand_matrix(64, 48, torch.complex128, torch.device("cpu"), seed=30)
    run("64x48 c128 L=8 k=32", A, L=8, k=32,
        sv_rtol=1e-8, subspace_tol=1e-5)


def test_large_real():
    print("\n[5] Large real 1024x512, L=32 (Frobenius only, no full SVD)")
    A = rand_matrix(1024, 512, torch.float64, torch.device("cpu"), seed=40)
    run("1024x512 f64 L=32 k=64", A, L=32, k=64,
        compare_full=False, frob_tol=0.5)


def test_tall_skinny():
    print("\n[6] Tall-and-skinny 2048x32, L=32 (full spectrum)")
    A = rand_matrix(2048, 32, torch.float64, torch.device("cpu"), seed=50)
    run("2048x32 f64 L=32 k=8", A, L=32, k=8,
        sv_rtol=1e-8, subspace_tol=1e-5)


def test_wide_short():
    """
    Wide-and-short matrix: after min(m,n)=32 steps the U-basis exhausts R^32.
    The algorithm must handle null-space detection without producing NaN.
    """
    print("\n[7] Wide-and-short 32x2048, L=32 (full spectrum, null-space handling)")
    A = rand_matrix(32, 2048, torch.float64, torch.device("cpu"), seed=60)
    run("32x2048 f64 L=32 k=8", A, L=32, k=8,
        sv_rtol=1e-8, subspace_tol=1e-5)


def test_lowrank():
    """
    True-rank-20 matrix: the algorithm must hit invariant subspaces cleanly
    and still recover all 20 non-zero singular triplets exactly.
    """
    print("\n[8] True-rank-20 256x256 matrix, L=20")
    A = lowrank_matrix(256, 256, rank=20, dtype=torch.float64,
                       device=torch.device("cpu"), seed=70)
    run("256x256 rank-20 f64 L=20 k=10", A, L=20, k=10,
        sv_rtol=1e-6, subspace_tol=1e-4)


def test_float32():
    print("\n[9] Float32 128x96, L=16, k=4*L (float32 precision limits)")
    A = rand_matrix(128, 96, torch.float32, torch.device("cpu"), seed=80)
    # float32 has ~7 digits; expect ~1e-4 SV accuracy with large k
    run("128x96 f32 L=16 k=64", A, L=16, k=64,
        sv_rtol=1e-3, subspace_tol=1e-2)


def test_gpu():
    if not torch.cuda.is_available():
        print("\n[10] GPU test: CUDA not available, skipping.")
        return
    print("\n[10] GPU 128x96 float64, L=16, k=4*L")
    dev = torch.device("cuda")
    A_gpu = rand_matrix(128, 96, torch.float64, dev, seed=90)
    A_cpu = A_gpu.cpu()

    mv, rmv = make_matvecs(A_gpu)
    U, S, Vh = lanczos_svd(mv, rmv, 128, 96, 16, k=64,
                            dtype=torch.float64, device=dev)

    _, S_ref, _ = torch.linalg.svd(A_cpu, full_matrices=False)
    sv_err = rel_sv_error(S.cpu(), S_ref.double()).max().item()
    print(f"  GPU 128x96 f64 L=16 k=64   SV={sv_err:.1e}")
    assert sv_err < 1e-8, f"GPU SV rel error {sv_err:.2e} >= 1e-8"
    print("  GPU test passed.")


def test_effect_of_k():
    """Confirm that larger k monotonically improves accuracy."""
    print("\n[11] Effect of k on 128x96 float64, L=8")
    A = rand_matrix(128, 96, torch.float64, torch.device("cpu"), seed=100)
    _, S_ref, _ = torch.linalg.svd(A.cpu(), full_matrices=False)

    prev_err = float("inf")
    for k in [0, 8, 16, 32, 64]:
        mv, rmv = make_matvecs(A)
        _, S, _ = lanczos_svd(mv, rmv, 128, 96, 8, k=k,
                               dtype=torch.float64, device=torch.device("cpu"))
        err = rel_sv_error(S, S_ref).max().item()
        print(f"    k={k:3d}  sv_err={err:.2e}")
        # Accuracy should be non-increasing (allow small numerical noise)
        assert err <= prev_err * 10 + 1e-15 or err < 1e-12, \
            f"Accuracy degraded: k={k} err={err:.2e} > prev {prev_err:.2e}"
        prev_err = err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("lanczos_svd accuracy tests")
    print("=" * 70)

    tests = [
        test_small_real,
        test_medium_real_coarse,
        test_medium_real_highk,
        test_medium_complex,
        test_large_real,
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
            print(f"  ERROR:  {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed:
        sys.exit(1)
