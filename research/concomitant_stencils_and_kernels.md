# Round 2: faster concomitant-field handling — butterfly, stencil NUFFT, kernel design, commutator modes

Scripts (all in `research/experiments/`, run with the repo `.venv`): `butterfly_split.py`, `stencil_width2.py`, `kernel_design.py`, `stencil_rank.py`
(plus round 1: `conc_structure.py`, `pair_rank.py`, `cp_rank.py`, `operator_frontier.py`, `identities_check.py`).
Everything is a small-scale model (1-D stencil studies with N=64 voxels; 3-D rank studies on 12³–32³ grids); numbers show *scaling and ratios*, not final performance.
Cost numbers use the measured FFT timings (RTX 4090, f32): 640³ = 14.6 ms, 400³ = 3.7 ms, fused Toeplitz conv pass N=320 = 11.1 ms (`rustlib/bench/baselines.md`).

## 0. Summary

| Thread | Verdict | Evidence |
|---|---|---|
| **Butterfly / hierarchical splitting** of the sample set | **No.** Splitting by time and/or direction *increases* the total number of spatial bases (= FFTs). Only pays off if the global rank is in the hundreds. | `butterfly_split.py`: e.g. twisty Φ≈9 rad: global L=42 vs Σ over 2/4/8 time bins 51/80/132, vs 4×4 (time×direction) 253. |
| **Chirp-aware stencil NUFFT** (one shared FFT, per-sample compact k-space stencil that absorbs `e^{-iΦ_j}`) | **Best FFT-count result: 1 FFT pair per coil independent of L and of Φ.** Cost moves to wider stencils: per axis J ≈ 8 (Φ=1 rad), 10 (2 rad), 13–14 (4 rad), 18 (8 rad) at tol 1e-3, vs J=4 for a plain NUFFT. | `stencil_width2.py` (σ=2…1.25), `stencil_rank.py`: chirp spectrum is *exactly separable* for the diagonal fields (z², x²+y²) and CP-rank 3–4 with the xz,yz cross terms. |
| **Different kernel (not ES/KB)** | **Second-order gain.** Jointly designed correction function: 1.4–4× lower error at fixed width (≈ 1 tap/axis saved ≈ 25–30% fewer 3-D taps). The width is set by the chirp's spectral spread, not the window family. For *derivative* (Taylor) schemes ES is the wrong family (√-singular derivatives): use KB or a smooth window. A complex-Gaussian window gives closed-form stencils but is untested and has poor edge conditioning. | `kernel_design.py` |
| **Commutator / Weyl expansion** (round 1) | **Right tool for small Φ (≲ 0.5–1 rad):** 4 FFT pairs (order 1) remove ~94% of the concomitant effect at Φ=0.3; 10 pairs (order 2) ~99.7%. Exactly Hermitian, no SVD, no stored basis. | `operator_frontier.py`, `concomitant_weyl_commutator.md` |
| **Defect correction / mixed fidelity** (new) | **Likely the cheapest end-to-end:** use the plain (or low-order) Toeplitz operator for the bulk of CG/ISTA work and the exact operator only for residuals, because `N − N0` is a few-% perturbation. | Argument + `operator_frontier.py` (effect size ‖N−N⁰‖/‖N‖ = 0.8 % at Φ=0.3, 2.8 % at 1, 7.9 % at 3). Needs a reconstruction-level test. |

## 1. Banded-Gram picture (why everything lines up)

For quadratic phases the k-space representation of the normal operator on the `(σN)³` grid is a **banded matrix**, `N̂(k,k+δ) = ν_δ(k)`, `|δ| ≤ s`, with `s ≈` the chirp's spectral radius (3–5 cells for Φ ≲ 2). Equivalently `N̂ = Γᴴ W Γ` where `Γ` is the sparse (K × grid) stencil matrix.
- Plain Toeplitz: only `δ=0` survives (`N̂ = diag(K̂)`): the exact sinc-interpolation limit of the NUFFT Gram.
- Concomitant: `ν_δ(k) = Σ_μ c_{δμ} K̂^μ(k)` with a **universal fixed table** `c_{δμ}` (independent of trajectory) and trajectory-dependent moment kernels `K̂^μ` (w-moments of the sample density) — the Hankel/moment form, now read in k-space as *universal small convolutions × moment-weighted diagonals*.
- Product-convolution rank, Weyl order, number of Taylor terms and stencil width are four views of the same number: the Slepian count `(spectral spread of e^{-iΦ}) × (FOV)`.

## 2. Why not a butterfly algorithm

The butterfly/FIO machinery earns O(N log N) when kernel blocks have rank O(1) *after* recursive subdivision, i.e. when the global rank is large. Here the global rank is `≈ (Φ_max/π)·c` ≈ 5–40.
Measured (`butterfly_split.py`, ε=10⁻³): subdividing the samples never helps — total bases grow (1.2–1.5× for 2 time bins, 2.9–4.5× for 8 bins, 6–9× with time×direction clusters) because each sub-group needs its own FFTs and the groups share nothing. A real butterfly factorisation of the whole `K × N³` operator would also cost
≈ r² × (FFT work) per application for block rank r≈8 plus a heavy per-trajectory factorisation. It becomes interesting only for Φ ≳ 100 rad (not concomitant fields at clinical B0/gradients); your `butterfly_*.cppm` experiments are the right tool for *large*-phase problems (e.g. strong off-resonance), not this one.

## 3. Chirp-aware stencil NUFFT (the main result)

Model: sample j needs `y_j = Σ_n ρ_n e^{-i(k_j·r_n + Φ_j(r_n))}`. NUFFT represents `e^{-ik·r}` as `c(r)⁻¹… Σ_u φ(k−u) e^{-iu·r}`; a modulated row needs `Σ_u Γ_j(u) e^{-iu·r}` with a stencil `Γ_j = φ ⋆ ĝ(·; w_j)` where `ĝ(·;w)` is the (compact) spectrum of the chirp `e^{-iΦ_w(r)}` over the FOV. Everything else (one FFT of the whole image, deconvolution `c(r)`, coil maps in image space) is unchanged.

**3.1 Required width (1-D model, LS-optimal stencil, shared correction `c(x)`, worst fractional k), taps per axis `J`:**

| Φ_max [rad] | σ=2: tol 1e-2 / 1e-3 / 1e-4 | σ=1.5: 1e-3 / 1e-4 | σ=1.25: 1e-3 / 1e-4 |
|---|---|---|---|
| 0 (plain NUFFT) | 4 / 4 / 6 | 6 / 8 | 8 / 10 |
| 1 | 6 / 8 / 10 | 8 / 10 | 8 / 10 |
| 2 | 8 / 10 / 12 | 10 / 12 | 10 / 12 |
| 4 | 10 / 13 / 16 | 14 / 16 | 14 / 16 |
| 8 | 16 / 18 / 22 | 18 / 22 | 18 / 22 |

- At σ≈1.25 the chirp is **free up to ~1 rad at tol 1e-3** (the plain stencil is already J=8 wide); in general the width behaves like `max(plain width, chirp spread)`.
- Width is roughly independent of σ once the chirp dominates → at Φ≥2 use the *smallest σ the plain kernel allows* (σ=1.25: FFT grid 400³ instead of 640³, 4× less FFT traffic).
- 3-D taps ≈ J³: Φ=1 → ~500, Φ=2 → ~1000, Φ=4 → ~2200 (plain: 64).

**3.2 Structure of the stencil (3-D, `stencil_rank.py`).** With only the diagonal fields (z², x²+y²) the chirp spectrum is *exactly separable* (Tucker/CP rank 1). With the cross fields (xz, yz): Tucker ranks 3–5 per mode, CP rank R=3 → error 4·10⁻³, R=4 → 1·10⁻³ (Φ≤2); R=6 → 10⁻⁴ (Φ=4). The stencil is therefore generated from `R ≈ 3–4` triples of 1-D vectors (each a 1-D convolution of the ES weights with a short `a_ρ(m; w)` vector), and `ĝ(·;w)` itself is a smooth function of the 4-vector `w` (tabulated on a small grid or Taylor/Chebyshev in `w`, a few hundred numbers).

**3.3 Cost model (per frame, per coil, per application = type-2 then type-1).**
`T_stencil ≈ 2·FFT(σN)³ + 2·K·J³·t_tap`, sample-side generation of `Γ_j` is **coil-independent** (computed once per sample, reused for all C coils; per coil only the J³ gather/scatter).
Example N=320, K=3·10⁶, C=32, tol 1e-3, assuming `t_tap` ≈ 5 ps (2·10¹¹ tap/s, **unmeasured guess**):
- σ=2, Φ≈2 (J=10): `2·14.6 ms + 2·3·10⁶·1000·5 ps = 29 + 30 ms` ≈ 60 ms/coil ⇒ 1.9 s per application.
- Toeplitz-CP, `L'≈10` (my fused conv pass 11.1 ms): 111 ms/coil ⇒ 3.6 s; `L'≈20` ⇒ 7.1 s; plain Toeplitz (no correction) 11 ms/coil ⇒ 0.36 s.
So the stencil route is roughly **2–4× faster than CP-Toeplitz at Φ≈2 and K/N³≈9 %**, and the gap grows as K/N³ shrinks or Φ grows; it loses when K·J³ is large (fully sampled data: Toeplitz cost is K-independent).
Memory: no kernels at all; the per-sample stencils are regenerated on the fly (storing them would be K·J³·8 B ≈ 24 GB).
Sample-side lever: Φ_j grows with readout time, so early samples (Φ_j ≪ Φ_max) use J≈4–6 — the average cost is a fraction of the worst case (time-adaptive J).

**3.4 Kernel design (the "other kernel" question), `kernel_design.py`.** With the shared correction `c(x)` jointly optimised for the whole family `{frac(k)} × Φ∈[0,Φ_max]` (alternating least squares, closed form per voxel):
worst-case error at fixed J drops 1.6–2.6× (Φ_max=2) and 2.4–4.2× (Φ_max=4); at σ=1.5 slightly less (1.4–2.0× and 1.9–4.0×). That is about one tap per axis. Conclusions:
1. The window family matters at second order; the chirp's spectral spread (a physical quantity) sets J.
2. Cheap win: use a *designed* `c(x)` (a 64-number table, computed once per FOV/σ/tol/Φ_max) instead of the ES/KB closed form.
3. For schemes that need derivatives of the interpolation kernel (Taylor/commutator in the sample domain) pick a kernel whose derivatives are bounded at the support edge: Kaiser–Bessel (entire in z) or the designed kernel, not ES.
4. Complex-Gaussian window: `Gaussian × chirp` is a Gaussian with complex variance `(a + i b)`, so stencils are closed-form for any `M_j` (3×3 complex matrix inverse) with no tables. Costs: a Gaussian needs ~1.5–2× the taps of ES/KB for the same tol and its image-domain correction `e^{+τ r²}` amplifies edge values (precision trouble in f32). Untested here; worth a prototype only if table lookup of `ĝ(·;w)` proves awkward.

## 4. Commutator / Weyl modes (small Φ) and defect correction

Round-1 results stand: with the exact commutator identities, order 1 costs 4 FFT pairs (7 Δ-moment kernels + base), order 2 about 10, and each order is exactly Hermitian. `N − N0` is small: ‖N−N⁰‖/‖N‖ = 0.8 % (Φ=0.3), 2.8 % (1), 7.9 % (3) in the toy problem.
**Defect correction:** write `A^H(Ax−b) = N0 x − A^Hb + (N−N0)x`. The last term is small, so it only needs *relative* accuracy ~10 % to give ≲1 % overall error:
- order-1 commutator mode (4 pairs) is enough when Φ ≲ 0.5 rad (6·10⁻² of the effect at Φ=0.3);
- for larger Φ evaluate `(N−N0)x` with the stencil NUFFT at *loose tolerance* (J smaller: tol 1e-2 needs J=6–10, i.e. 200–1000 taps) — and only every few iterations (the correction changes slowly with x);
- a two-level CG: inner iterations with cheap `N0` (or `N0 +` order-1), outer residual refreshes with the exact operator; convergence factor per outer iteration ≈ ‖I − N0⁻¹N‖ ≈ Φ-sized.
This needs a *reconstruction-level* study (image error vs operator error); the operator-level numbers above do not tell by themselves how accurate the correction must be.

## 5. Recommended routes by regime (hypotheses to test, in this order)

| Regime (max Φ, K/N³) | First choice | Why |
|---|---|---|
| Φ ≲ 0.5 rad (≥3 T, short readout) | Toeplitz + commutator order 1–2 (4–10 pairs), or defect correction | no SVD, exact structure, 4–10× plain Toeplitz |
| 0.5 ≲ Φ ≲ 4, any K/N³ ≲ 10 % | **Chirp-aware stencil NUFFT**, σ 1.25–2, time-adaptive J, designed `c(x)`; defect-correct if possible | FFT count independent of Φ, no kernels, scales with K |
| Φ ≲ 4, K/N³ ≳ 20 % (well sampled) | Toeplitz-CP / SVD-Tucker (`L'≈2L`) with real/fp16 kernels | Toeplitz cost is K-independent |
| Φ ≳ 10 | stencil NUFFT with large J, or SVD-Tucker L≈40–80 | butterfly only if Φ ≳ 100 |
Off-resonance multiplies in: low-rank `L_z` temporal terms × (any of the above), or fold `e^{-iω(r)t_j}` into the stencil spectrum `ĝ(·; t_j, w_j)` (5-parameter table).

## 6. What the libraries need (generic, no MRI inside)

- **NUFFT** (`hasty-nufft`): per-sample *stencil generator* hook (device code that returns the J taps per axis from `(k_j, params_j)`, CP-rank-R form: R × three 1-D vectors), width J chosen per sample, multi-RHS gather/scatter across coils with shared stencil, free σ down to 1.25, designed correction function `c(x)` as an input table, kernel evaluators with bounded derivatives.
- **FFT** (`hasty-fft`): done in F4: fused pruned convolution with pre/post diagonals, packed kernels. Add: coordinate-monomial diagonals generated from the voxel index (for the commutator modes), several kernels sharing one forward transform (Hankel contraction in the k-space step), multi-coil batching that reuses `K` through L2.
- **Research code → library validation:** every claim above has a script; each should become a Rust test against a small dense reference before optimisation.

## 7. Open / unverified

- `t_tap` (stencil tap throughput with sorting, shared-memory tiles and atomics) decides the stencil-vs-Toeplitz crossover; measure on the 4090 with a prototype type-2/type-1 using J up to 14.
- All stencil-width numbers are 1-D models; 3-D with cross terms should be re-measured (rank R≈3–4 suggests ≤ J³ but the combined error budget across axes needs checking).
- Reconstruction-level error budget (how accurate must `(N − N0)x` be?).
- Real sequences: Φ_max and its distribution over samples and FOV offsets; whether time-adaptive J helps as much as hoped.
