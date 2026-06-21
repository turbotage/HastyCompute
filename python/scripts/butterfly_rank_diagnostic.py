"""
Rank-decay diagnostic for a potential butterfly-factorization speedup of
CoordinateWarp's r<->q transform.

The transform evaluates, for every q-grid point:
    m_q(q) = (1/N) * Sum_k C_k * exp(i * Phi(k, q)),   Phi(k, q) = (2*pi/N) * k * u_inv(q)
i.e. a matvec with matrix M[q, k] = exp(i*Phi(k,q)) -- exactly the canonical
form of a Fourier integral operator (FIO) matrix that butterfly factorization
targets (see ButterflyLab / Candes-Demanet-Ying / Li-Yang-Ying). Note the
2*pi/N normalization -- k and q are integer PIXEL indices in [-N/2, N/2), and
the actual phase (matching make_nufft_coords' convention) divides by N. This
is NOT optional bookkeeping: it sets the scale at which admissible boxes are
actually low-rank, even for plain (undistorted) FFT.

Butterfly's whole speedup rests on M restricted to small "admissible" box
pairs (box_k x box_q, sized so |box_k|*|box_q| ~ N) being numerically
low-rank, even though M is NOT globally low-rank. With the 2*pi/N scaling,
this holds for PLAIN FFT (delta=0) by construction -- FFT itself is a
butterfly algorithm. So the real question isn't "is M low-rank" (plain FFT
already answers yes) -- it's "how much EXTRA rank does the warp's
displacement field ADD on top of plain FFT's own baseline". This script
reports both, so the warp's actual contribution isn't conflated with
FFT's own (already fine) structure.

No dependency on the C++ build -- this is a pure-math check on a
representative quadratic GNL-like displacement field. Plug in your own
max-displacement number (voxels) from CoordinateWarp::max_abs_displacement_pix()
to make this match your actual problem.

Run: python butterfly_rank_diagnostic.py
"""

import numpy as np


def u_inv(q, delta_max, half_n):
    """
    Representative GNL-like inverse warp: quadratic displacement, zero at
    isocenter, growing to delta_max voxels at the FOV edge (q = +-half_n).
    This mirrors the x^2/y^2/z^2-type GNL channels used in the actual warp --
    swap in the real field_fn evaluation if you want this exact rather than
    representative.
    """
    delta = delta_max * (q / half_n) ** 2
    return q - delta


def phase_matrix(k_vals, q_vals, delta_max, half_n, n_grid):
    """M[i,j] = exp(i * (2*pi/N) * k_vals[i] * u_inv(q_vals[j]))."""
    uinv_q = u_inv(q_vals, delta_max, half_n) if delta_max > 0 else q_vals
    Phi = (2.0 * np.pi / n_grid) * np.outer(k_vals, uinv_q)
    return np.exp(1j * Phi)


def singular_values(box_k_center, box_q_center, box_size_k, box_size_q,
                     delta_max, half_n, n_grid, n_samples=48):
    k_vals = box_k_center + np.linspace(-box_size_k / 2, box_size_k / 2, n_samples)
    q_vals = box_q_center + np.linspace(-box_size_q / 2, box_size_q / 2, n_samples)
    M = phase_matrix(k_vals, q_vals, delta_max, half_n, n_grid)
    return np.linalg.svd(M, compute_uv=False)


def rank_for_tol(s, tol):
    s_norm = s / s[0]
    idx = np.searchsorted(-s_norm, -tol)
    return int(idx), len(s)


def sanity_checks(N, half_n, delta_max, tol):
    """
    Two checks that the diagnostic itself isn't just trivially reporting low
    rank regardless of input -- run these BEFORE trusting the main result.
    """
    print("=== Sanity check 1: NON-admissible box (should be near FULL rank) ===")
    # box_k * box_q >> N (deliberately mismatched, not admissible) -- even
    # PLAIN FFT should show high rank here. If this ALSO comes out low-rank,
    # the diagnostic is broken (e.g. always producing low rank regardless of
    # input), not confirming anything real.
    s_full = singular_values(0.0, 0.0, N, N, 0.0, half_n, N)
    r_full, total = rank_for_tol(s_full, tol)
    print(f"  box_k={N} box_q={N} (NOT admissible): plain-FFT rank = {r_full}/{total}")
    if r_full < total * 0.5:
        print("  UNEXPECED: still low-rank on a non-admissible box -- diagnostic"
              " may be broken, treat results below with suspicion.\n")
    else:
        print("  As expected: high rank when admissibility is violated -- the"
              " low-rank results below are because of admissibility, not a"
              " degenerate/broken matrix construction.\n")

    print("=== Sanity check 2: OSCILLATORY (non-smooth) displacement (should INFLATE rank) ===")
    # A rapidly-oscillating delta(q) is NOT what butterfly's smoothness
    # premise assumes -- this should show real extra rank on an ADMISSIBLE
    # box, proving the diagnostic CAN detect rank inflation when it's
    # actually there, rather than always reporting ~0 regardless of input.
    bk, bq = N / 4, 4.0  # admissible: product = N
    qc = half_n * 0.9
    k_vals = np.linspace(-bk / 2, bk / 2, 48)
    q_vals = qc + np.linspace(-bq / 2, bq / 2, 48)
    osc_delta = delta_max * np.sin(q_vals * 2.0)  # oscillates several times across a tiny box
    Phi_osc = (2.0 * np.pi / N) * np.outer(k_vals, q_vals - osc_delta)
    s_osc = np.linalg.svd(np.exp(1j * Phi_osc), compute_uv=False)
    s0 = singular_values(0.0, qc, bk, bq, 0.0, half_n, N)
    r0, total = rank_for_tol(s0, tol)
    r_osc, _ = rank_for_tol(s_osc, tol)
    print(f"  plain-FFT rank = {r0}/{total}, oscillatory-warp rank = {r_osc}/{total}")
    if r_osc - r0 < total * 0.1:
        print("  UNEXPECTED: oscillatory displacement did NOT inflate rank --"
              " diagnostic is likely insensitive, treat the main result with"
              " suspicion.\n")
    else:
        print("  As expected: a non-smooth displacement DOES inflate rank --"
              " the diagnostic is sensitive, so the smooth-GNL result below"
              " is meaningful, not a default-low-rank artifact.\n")


def main():
    # ---- Problem parameters: EDIT to match your actual setup ----
    N = 256              # representative grid size per axis (use your real N)
    delta_max = 60.0      # max GNL displacement in VOXELS at FOV edge (use your
                          # real max_abs_displacement_pix() value here)
    half_n = N / 2.0
    tol = 1e-3           # accuracy target for the low-rank approximation

    sanity_checks(N, half_n, delta_max, tol)

    # Admissible box sizes: |box_k| * |box_q| ~ N, matching butterfly's
    # multilevel admissibility condition (this is what makes plain FFT itself
    # low-rank on these boxes -- see module docstring). Sweep several scales,
    # like the levels of an octree from coarsest to finest.
    box_sizes_k = [N, N / 2, N / 4, N / 8, N / 16, N / 32]

    # Test box_q CENTERS across the FOV, including the EDGE (worst case for
    # GNL displacement) -- not just the center, where displacement is ~0 and
    # the matrix trivially looks linear/low-rank regardless of the warp.
    q_centers = [0.0, half_n * 0.5, half_n * 0.9]

    print(f"N={N}  delta_max={delta_max} voxels  tol={tol}\n")
    print(f"{'box_k':>8} {'box_q':>8} {'q_center':>9}  {'plain FFT':>10}  {'warped':>10}  {'EXTRA':>7}")
    print("-" * 60)

    worst_extra = 0.0
    for bk in box_sizes_k:
        bq = N / bk  # keep product ~ N (admissibility)
        for qc in q_centers:
            s0 = singular_values(0.0, qc, bk, bq, 0.0, half_n, N)         # plain FFT baseline
            s1 = singular_values(0.0, qc, bk, bq, delta_max, half_n, N)   # warped
            r0, total = rank_for_tol(s0, tol)
            r1, _     = rank_for_tol(s1, tol)
            extra = (r1 - r0) / total
            worst_extra = max(worst_extra, extra)
            print(f"{bk:8.1f} {bq:8.2f} {qc:9.1f}  {r0:4d}/{total:<5d}  {r1:4d}/{total:<5d}  {extra:7.2f}")

    print("\n--- Verdict ---")
    print("(\"EXTRA\" = rank the WARP adds beyond plain FFT's own baseline "
          "rank on the same admissible box -- this isolates the warp's "
          "actual contribution, since plain FFT is already butterfly-"
          "compatible by construction.)\n")
    if worst_extra < 0.1:
        print(f"Worst-case extra rank fraction = {worst_extra:.2f} -- warp adds "
              f"almost nothing on top of FFT's own structure. Butterfly premise "
              f"looks strong; worth scoping the real build.")
    elif worst_extra < 0.3:
        print(f"Worst-case extra rank fraction = {worst_extra:.2f} -- the warp "
              f"adds real but moderate extra rank. Butterfly should still beat "
              f"NUFFT's kernel-width cost, but by less margin than the "
              f"\"free\" FFT case alone -- worth a closer cost comparison "
              f"before committing.")
    else:
        print(f"Worst-case extra rank fraction = {worst_extra:.2f} -- the warp "
              f"itself is destroying the admissible boxes' low-rank structure, "
              f"not just riding on FFT's own. Butterfly's premise is weak for "
              f"this displacement magnitude; the multi-month build risk "
              f"probably isn't worth it here.")


if __name__ == "__main__":
    main()
