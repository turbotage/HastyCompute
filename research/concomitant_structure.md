# Concomitant fields when every spoke is a linear transform of a base spoke

Companion to `RESEARCH_fast_normal_operator.md`. Experiments: `research/experiments/{conc_structure,pair_rank,cp_rank}.py`
(torch/numpy, run with the repo `.venv`). All rank numbers are on small voxel grids (12³–20³ over a 0.32 m FOV) with random SO(3) rotations of a
synthetic 2 ms prototype — they show *trends*; absolute ranks depend on your real sequences, FOV offsets and B0. Rerun with real waveforms before trusting a number.

## 0. Findings in one place

1. **Probable bug in the main pipeline (check first).** `cpp/test/src/non_fourier_interp_test.cpp:302-305` (and `notes/MRI_Physics.tex` eq. `conc_maxwell`) use
   `(Gx²+Gy²)/(8B0)·z²` and `+GxGz/(2B0)·xz`. Deriving from `B = (B0+G·r) ẑ + (Gx z − Gz x/2) x̂ + (Gy z − Gz y/2) ŷ` (∇·B = 0, ∇×B = 0) gives
   `B_c = [ (Gx z − Gz x/2)² + (Gy z − Gz y/2)² ] / (2B0) = (Gx²+Gy²)/(2B0) z² + Gz²/(8B0)(x²+y²) − GxGz/(2B0) xz − GyGz/(2B0) yz`.
   The transverse `z²` term is **4× larger** than the code/notes, and the cross-term sign is **opposite**. `cpp/test/src/concomitant_coordinate_warp_test.cpp:106-109` has the correct coefficients,
   so the two tests disagree. `conc_structure.py [1]` checks the exact |B| expansion numerically (leading-order error ~2·10⁻⁴ relative at 3 T, 20 mT/m, 10 cm).
   This changes `max_phase_z2` and every rank/accuracy conclusion drawn from the main generator.
2. **Exact lifted-linear structure (Sec. 1).** If `k_s(τ) = A_s k_0(τ)` then the concomitant coefficients are `w_s(τ) = Θ_s W_0(τ)` with `Θ_s` a fixed 4×6 matrix built from `A_s`
   (`W_0(τ) = ∫G_0G_0ᵀ`, 6 comps). In the lifted feature space every spoke is a **linear image of one base curve** and every spoke kernel is the same function evaluated at a linearly transformed argument.
   Consequence: trajectories, waveform coefficients and kernels are all *matrix-free*, generated from a `T×(3+6)` prototype table and `S` matrices. **It does not reduce any rank** (Sec. 3 — measured).
3. **Where the cost really comes from (measured).** The spatial rank needed by a *global* basis grows much faster with phase than the rank of one spoke or of the shared temporal family:
   at max|Φ|≈9.7 rad (twisty) global 41 vs single-spoke 11 vs temporal-family 14 (ε=10⁻³). The spoke-dependence of the spatial functions is a ~3–4× penalty at large phase, ~1.2–1.7× at ≤3 rad.
4. **Kernel count vs FFT count frontier (measured).**
   - Tucker (your naive/Hermitian-squeezed): `L(L+1)/2` kernels, `L` FFT pairs/coil.
   - Optimal general decomposition: only `R_HS ≈ 1.6–2.5·L` kernels are needed (4–8× fewer than Tucker), but the atoms then have high rank, so FFT pairs/coil grow (e.g. 38–576 vs `L`=6–19).
   - Rank-1 atoms (Fessler IV-B generalised, "Hermitian CP"): `L' ≈ 2L` kernels **and** `L'` FFT pairs/coil at ~1–2·10⁻² HS error (optimiser untuned → upper bound).
   So the choice is a real tradeoff, not a free reduction: kernels ↓ costs FFTs ↑ (see Sec. 3 table).
5. **Moment (Hankel) structure of the Taylor route (exact, Sec. 2).** Kernel `T_{αα'}` depends only on `α+α'`: `C(4+2P, 2P)` distinct kernels instead of `L(L+1)/2`
   (70 vs 120 at P=2, 210 vs 630 at P=3). Free, exact, but only a constant factor.
6. **Stencil width for the modulated-row NUFFT (measured).** For one row `exp(−iΦ_j(r))` on a hard FOV box: ≤3 rad needs radius ~3–4 k-cells for 10⁻³ of the energy outside;
   8.5 rad needs ~8 cells. (A tapered window would shrink the tails; hard-box leakage inflates these.)

## 1. Lifted-linear structure (exact)

Gradient `G_s(τ) = A_s G_0(τ)`, `k_s = A_s k_0` (`A_s` rotation, or any linear map — calibrated scale/shear included).
`B_c = Gᵀ S(r) G /(2B0)` with `S(r) = Mr(r)ᵀ Mr(r)`, `Mr = [[z,0,−x/2],[0,z,−y/2]]`, entries `{z², z², (x²+y²)/4, −xz/2, −yz/2}`.
So `Φ_s(r,τ) = (γ/2B0)·⟨W_s(τ), S(r)⟩` with `W_s = A_s W_0 A_sᵀ` (verified numerically, `conc_structure.py`).
In the 4 lab-frame fields `m = (z², x²+y², xz, yz)`: `w_{s,q}(τ) = (γ/2B0) tr(E_q A_s W_0(τ) A_sᵀ) = Σ_p Θ_{s,qp} W_{0,p}(τ)`.

Lifted features: voxel `μ(r) = (r, m(r), ω(r))`, sample `κ_s(τ) = (A_s k_0(τ), Θ_s W_0(τ), τ) = L_s κ_0(τ)` with `L_s = diag(A_s, Θ_s, 1)`. Then

```
a_{sτ}(r)            = exp(−i ⟨κ_s(τ), μ(r)⟩)
(A^H W A)(r, r')     = Σ_s Ĝ_0( L_sᵀ (μ(r) − μ(r')) ),    Ĝ_0(η) = Σ_τ w_τ e^{i⟨κ_0(τ), η⟩}      (one universal function, FT of a 1-D curve)
```

Exact local form (quadratic `m`): `m(r̄+Δ/2) − m(r̄−Δ/2) = J_m(r̄) Δ`, so the kernel at `r̄` is the PSF of the effective trajectory
`k^eff_s(τ; r̄) = A_s k_0(τ) + J_m(r̄)ᵀ Θ_s W_0(τ)` — linear in `r̄`. (Exact for degree ≤ 2; cubic GNL corrections give an `O(Δ³∂³Φ)` remainder.)

What you get from it: nothing but *generation*: `k_j`, `g_α(j)`, `w_{s,q}(τ)` are computed in-kernel from `(A_s, Θ_s)` and the prototype table; kernels
`T^{mom}_γ = Σ_β Σ_s c^{(s)}_{γβ} P_β∘A_sᵀ` with prototype moment-PSFs `P_β(Δ') = Σ_τ w_τ W_0(τ)^β e^{ik_0(τ)·Δ'}` (for radial, `P_k` are 1-D profiles).
Rank per sample is unchanged: any sample, spoke-structured or not, has a 4-vector `w_j` of concomitant coefficients.

## 2. Moment (Hankel) structure of the exact Taylor expansion

`e^{−iΦ_j(r)} = Σ_α c_α(j) F_α(r)`, `F_α = Π m_q^{α_q}`, `c_α(j) = (−i)^{|α|} g_α(j)/α!`, `g_α(j) = Π_q w_{j,q}^{α_q}` (real).
`conj(c_α) c_{α'} = i^{|α|−|α'|}/(α!α'!) · g_{α+α'}(j)`, therefore

```
N = Σ_{α,α'}  i^{|α|−|α'|}/(α!α'!) · F_α^H · T^{mom}_{α+α'} · F_{α'},     T^{mom}_γ(Δ) = Σ_j w_j g_γ(j) e^{i k_j·Δ}
```

Only `γ = α+α'`, `|γ| ≤ 2P` appear (`C(4+2P,2P)` kernels). With off-resonance time segmentation the same trick applies per factor (Fessler IV-B).
Cost per coil: `2L` FFTs (L = C(4+P,P)); the k-space stage is a 15×15 (P=2) Hankel-structured matvec per voxel, memory-bound if kernels are streamed per coil.
Taylor is only attractive for small phase (remainder `|Φ|^{P+1}/(P+1)! e^{|Φ|}`); SVD wins for large phase (measured ranks above vs `C(4+P,P)`).

## 3. Measured tradeoffs (`pair_rank.py`, `cp_rank.py`; 12³ voxels, 50 random rotations × 40 time samples)

`L` = rank of `E=exp(−iΦ)` at ε=10⁻²/10⁻³. Kernels/FFT pairs are per frame / per coil.

| case | max Φ (rad) | L | Tucker kernels | R_HS kernels | atom FFT pairs (R_HS route) | CP L' (error≈1–2·10⁻²) |
|---|---|---|---|---|---|---|
| twisty 3 T | 0.9 | 4 / 6 | 10 / 21 | 5 / 9 | 11 | ~8 |
| twisty 1 T | 2.6 | 6 / 13 | 21 / 91 | 9 / 21 | 38 | ~12 |
| twisty 0.5 T | 5.4 | 11 / 23 | 66 / 276 | 19 / 44 | 148 | — |
| curly 1 T | 5.8 | 9 / 20 | 45 / 210 | 16 / 39 | 105 | ~16 |
| curly 0.5 T | 11.5 | 19 / 39 | 190 / 780 | 43 / 97 | 576 | — |

Global vs single-spoke vs temporal-family rank (`conc_structure.py`, ε=10⁻³, 20³ voxels):

| case | max Φ | global (all spokes) | one spoke | spoke-independent time basis |
|---|---|---|---|---|
| twisty 3 T | 1.0 | 5 | 4 | 6 |
| twisty 1 T | 2.9 | 12 | 7 | 8 |
| twisty 0.3 T | 9.7 | 41 | 11 | 14 |
| curly 1 T | 6.4 | 21 | 10 | 14 |
| curly 0.3 T | 21 | 79 | 19 | 24 |

Reading: (a) FFT pairs/coil ≈ L is the floor for any shared-spatial-basis scheme; (b) kernel count can be cut 3–8× below Tucker at ~2× the FFTs (CP) — worthwhile only when kernel *memory*, not FFT time, binds;
(c) the global/per-spoke ratio (the price of spoke-dependent spatial functions) is what makes concomitant harder than off-resonance — off-resonance has *one* spatial map for all samples.
Clustering spokes by direction does not help: 20 nearest-direction spokes still need ~60% of the global rank (24 vs 41), so G clusters cost `G·L_c ≫ L`.

## 4. What this suggests (ranked by expected payoff, all unproven)

1. **Fix the coefficients first**, regenerate the problem generator and re-measure `L` (item 0.1). Likely changes the whole picture; at the correct magnitude transverse gradients matter 4× more.
2. **Modulated-row NUFFT (M5)** is the one route whose FFT count is independent of `L` and of the spoke-dependence penalty: the spoke-dependent row `e^{−iΦ_j(r)}` is a compact k-space stencil that depends on `j` only through the 4-vector `w_j`.
   So tabulate/interpolate (or Taylor/Chebyshev-expand) the stencil as a function of `w_j ∈ R⁴` **once**, independent of S and trajectory; per-sample generation is a 4-D lookup. Radius 3–4 cells at ≤3 rad ⇒ taps ≈ 10³–3·10³ including ES width (Sec. 6).
   Break-even vs FFT route: `K·taps` vs `L·FFT(2N)` work (see `RESEARCH_fast_normal_operator.md`, Sec. 2). Needs a prototype; the stencil table is cheap to compute offline (3-D FFT of a chirp on a ~16³ grid per node `w`).
3. **Kernel count reduction (CP/Hankel) only if kernel memory binds** — and build kernels on the fly (Sec. 3.7 of the other note); the rotated-prototype structure makes `T^{mom}_γ` cheap to generate.
4. **Do not expect rank reduction from rotation symmetry for undersampled frames** (no continuous symmetry; measured penalty is geometric, not removable by relabelling).

## 5. Open questions / things I could not resolve

- Whether SVD-derived spatial bases at 320³ stay as low-rank as at 12³–20³ (should: smooth functions, rank set by phase range, not resolution) — untested.
- A genuinely better-than-CP decomposition of the pair tensor with node selection in `w`-space (Fessler's frequency-segmentation analog in 4-D, nodes covering only the **visited** set of `w_j`, a 3-D cone-like set): not attempted.
- Metaplectic/phase-space view (quadratic phase = shear `k → k + M r` of phase space, 4-parameter family of shears): suggests a Radon-type structure over `(r,k)`; no algorithm found.
- Real trajectories: LTI-corrected gradients are still `A_s`-linear per axis (`[k_s]_a = Σ_c R_{ac,s} κ_{ac}(τ)`), but spoke-dependent eddy/heating residuals break exactness; measure their size.
