# Time-segment basis of the segmented NUFFT: noise-referenced rule, weighting and smooth-field absorption

Round of 2026-11-10 on ph0001 (factory_arms, 2 mm thorax, 3 T, `sim_flow_10min`). CPU only: the GPU benchmark
(`segbudget`, §7) is written and builds but its run was not permitted in this session, so seconds per set below come
from the cost model of the production run (§6), not from a new measurement.

## 0. Decision table

R = model error of the operator in the data, in units of the noise: ‖(A − Ã) x‖ / (σ √(M C)) for the realistic image
x (warm start, encode 0), real trajectory, whitened true coil maps, σ = the sim's noise per whitened coil (§2). R = 0.2
adds 4 % to the residual energy; R = 1 means the model error is as large as the noise. Cost per set (forward +
adjoint, 31 coils, 2 mm) ≈ 0.077 s × L (FFT-bound; production: L = 22 → 1.7 s, §6). Absorption has no kernel cost.

Water (ph0001 field incl. the mean concomitant phase), exact-metric R (§3) and s/set:

| absorption | basis | L | R | s / set |
|---|---|---|---|---|
| none | fixed, `--nseg auto` (today) | 17 | 0.021 | 1.31 |
| none | fixed | 14 / 13 / 12 | 0.094 / 0.148 / 0.221 | 1.08 / 1.00 / 0.92 |
| none | svdw (existing option) | 12 / 10 | 0.176 / 0.328 | 0.92 / 0.77 |
| none | fixed + energy-weighted LS (`--seg-weight energy`) | 14 / 12 / 11 | 0.026 / 0.065 / 0.088 (Rust pred.) | 1.08 / 0.92 / 0.85 |
| none | Chebyshev nodes | 14 / 12 | 0.155 / 0.272 | 1.08 / 0.92 |
| none | **svdn + energy** (`--seg-basis svdn --seg-weight energy`) | 10 / **9** / 8 | 0.079 / **0.150** / 0.254 | 0.77 / **0.69** / 0.62 |
| degree 1 | fixed | 14 / 12 | 0.077 / 0.183 | 1.08 / 0.92 |
| degree 1 | svdn + energy | 10 / 9 / 8 | 0.083 / 0.158 / 0.263 | 0.77 / 0.69 / 0.62 |
| degree 2 (ΔC Taylor) | any | any | ≈ 1400 (Taylor argument Q·T/2 up to 22 rad) | — |
| none, + linear shim residual ±150 / ±100 Hz | svdn + energy / fixed | 10 / 12 | 0.280 / 0.371 | 0.77 / 0.92 |
| degree 1, + the same residual | svdn + energy / fixed | 10 / 12 | 0.083 / 0.183 (= ph0001 + degree 1, exactly) | 0.77 / 0.92 |
| none, + Z2 residual ±150 Hz (widening) | svdn + energy / fixed | 10 / 12 | 0.178 / 0.405 | 0.77 / 0.92 |

Water + fat (frame-0 fat-fraction map folded; R2* not in the exact metric):

| absorption | basis | L | R | s / set |
|---|---|---|---|---|
| none | fixed, `--nseg auto` (today, production) | 22 | 0.007 | 1.70 |
| none | fixed | 18 / 16 / 14 | 0.047 / 0.109 / 0.247 | 1.39 / 1.23 / 1.08 |
| none | fixed + energy-weighted LS (per-frame maps possible) | 16 / 14 | 0.053 / 0.138 | 1.23 / 1.08 |
| none | Chebyshev nodes | 16 / 14 | 0.089 / 0.285 | 1.23 / 1.08 |
| none | **svdn + energy** (one map only) | 13 / **12** / 11 | 0.072 / **0.133** / 0.231 | 1.00 / **0.92** / 0.85 |
| degree 1 | svdn + energy | 12 | ≈ as none (predicted 0.103 vs 0.110) | 0.92 |

Rule `--nseg auto:noise:0.2` (prediction × 1.5 ≤ 0.2) as implemented picks, on the real tables (GNL, encode-mean
concomitant coefficients, 31 virtual coils; fat with R2*): water fixed 14, fixed + energy 11, **svdn + energy 9**;
water + fat fixed 18, fixed + energy 15 (the per-frame-map path), **svdn + energy 12** (one folded map). Today's worst-voxel rule gives 17 / 22.
Recommendation (§5): `--nseg auto:noise:0.2 --seg-basis svdn --seg-weight energy --b0-absorb 1` → **−47 % (water),
−45 % (water + fat, one map), −32 % (per-frame fat maps, fixed + energy, L = 15)** operator time per set.


## 1. What exists

* `fixed` (default): diagonals P_l = e^{−iψ(·, τ_l)} at L uniform node times over the ADC window (both ends included),
  interpolators a(τ) by least squares over the fit voxels (support/3, equal weights) at every one of the 3990 table
  entries. Equal voxel weights over the support = the field-histogram-weighted least-squares interpolator of
  Sutton, Noll & Fessler (IEEE TMI 22:178, 2003); their min–max variant (worst case over a frequency band) is the
  thinking behind today's segment count.
* `svd` / `svdw`: the optimal rank-L separable model of g(v, τ) on 96 uniform times (`build_segments_svd_weighted`;
  svdw: time weights = measured data energy per ADC time, floor 10⁻⁶; voxel weights count or `mass` = smoothed |x̄| /
  median clamped to [0.3, 3]).
* Joint water + fat: one fat-fraction map folded into g (any basis), or per-frame maps through the ff-independent
  table of `build_segments_fat` (fixed nodes only: the operator applies the fat factor 1 + f (c_l − 1) per node).
* `--nseg auto`: L = ⌈1.3 · Δf·T + 3.5⌉ with Δf the **full** range of the phase rate over the fit voxels (every 7th),
  including the mean concomitant phase and, joint, the main fat peak. On ph0001 the range is set by < 0.1 % of the
  voxels (air–tissue interfaces): field over the support 0 / 1 / 99 / 100 % = −760 / −204 / 435 / 804 Hz →
  L = 17 (water), 22 (joint water + fat, the production run).
* Per-sample tables: constant phase φ0, k-shift (exact) and the quadratic ΔC term (Taylor order 1/2 inside the kernel)
  for the concomitant *deviation* from the encode-mean coefficients; degree 3/4 to first order with order 3.
  Before this round the B0 field went entirely into the segments.

Related literature (from memory): time segmentation (Noll et al., IEEE TMI 10:629, 1991); multi-frequency
interpolation (Man, Pauly & Macovski, MRM 37:785, 1997) — the transposed separable form (fixed temporal exponentials,
fitted spatial coefficients); Sutton et al. 2003 (interpolators for iterative reconstruction: min–max over a band,
least squares over the field histogram); Fessler et al., IEEE TSP 53:3393, 2005 (Toeplitz normal operator with the
time-segmented model); the rank-L SVD of the exponential matrix as the optimal separable model (e.g. the SVD options
of MIRT `mri_exp_approx`). All of these weight frequencies (voxels) by count or worst case, and time uniformly; none
references the error to the data noise.

## 2. The metric

Per sample j = (readout r, ADC index n) and coil c the model error is
e_cj = F_{k_j}[S_c x δ_n], δ_n(v) = g(v, τ_n) − Σ_l a_l(τ_n) D_l(v); per ADC index the error energy over all readouts
and coils is (readouts at index n are quasi-uniform on the shell |k| = κ_n)

  E(n) = Σ_c mean_r |F_{k_rn}[S_c x δ_n]|²,   R² = mean_n E(n) / (C σ²).

The readout goes out and back: |k| = 0 at the first and last sample, π/Δ in the middle. Measured whitened data energy
per ADC sample: 86.6 % in the first quarter, 13.4 % in the last, 0.0 in the middle half; 67 % of the energy in the 51
samples with |k| < 20 rad/m (one Cartesian cell around DC), 97 % within |k| < 100 rad/m. Per-sample SNR (data energy
over noise energy) ≈ 2.4·10⁵ at the first sample, 2·10⁴ at the last, 1–3 in the middle (noise 2 % of the k-space rms).
So a relative model error of 10⁻³ matters at the k-space centre and 0.3 is harmless in the middle.

Evaluation (scratch `segeval.py`, CPU): near the centre (|k| < 120 rad/m, 78 ADC indices) a direct DFT of 4×4×4-block
sums of S_c x δ_n at the real sample positions of 48 readouts with all 32 whitened coils; elsewhere (63–90 indices)
the shell average |k| = κ_n of the full-resolution FFT of √(Σ_c|S_c|²) x δ_n. Checked against the exact full-resolution
DFT at the 48 readouts (fixed and SVD basis, L = 12, 16 ADC indices): block sums 0.91–1.00 of exact below 120 rad/m;
rss-FFT shells 0.65–1.3 above (the exact reference itself has ±20–30 % sampling noise from 48 directions; all 32 coil
FFTs gave the same as rss at 224, 645, 359 rad/m). A white-error model Σ_v |x|²Σ|S|²|δ|² is right only above
≈ 600 rad/m and underestimates 3–50× near the centre; the proportional model (relative voxel error × data energy of
that sample) overestimates 2–4× near the centre.

Two limits explain the per-sample error: a voxel-wise error that is smooth over the object reaches a sample in
proportion to that sample's signal (energy E(n)); a rough one (field interfaces) reaches every sample at the white level
E_vol = Σ_v |x|² Σ_c |S_c|² (by Parseval the volume-average data energy over the k-space ball, computable from the data
alone: `vol_energy`, E_vol / (C σ²) = 21–38 on ph0001). Hence the separable weights

  voxel  w_v = |x̄_v|² Σ_c |S_c(v)|²  (smoothed warm-start magnitude × coil energy; `--seg-weight energy`)
  time   w_n = 1 + (E(n) + E_vol) / (C σ²)  (`noise_time_weights`; E(n) from the whitened data)

and the prediction R_pred(L) = ε_w(L) √(mean_n w_n), ε_w the relative weighted separable error. For the weighted SVD
on that metric ε_w(L)² is the tail Σ_{l≥L} λ_l / Σ λ_l of the Gram eigenvalues — no segments need to be built to
choose L. With SNR-only weights (no white floor) the prediction is 2–4× optimistic and the basis leaves the middle of
the readout unweighted (svdn_e vs svdm_e below).

## 3. Measured R vs L (exact-metric evaluation, CPU)

Candidates (scratch `segeval.py` ports of the Rust builders; fit voxels = dilated support / 3, as `recon_p2np`):
`fixed` = today (uniform nodes, equal-weight LS); `fixed_e` = uniform nodes, energy-weighted LS; `cheb` = Chebyshev
nodes, equal-weight LS; `svd` = unweighted rank-L (96 uniform times); `svdw` = today's svdw (data-energy time weights,
count); `svdn_e` = SNR-only time weights 1 + E(n)/Cσ² on `seg_time_grid`, energy voxel weights; `svdm_e` / `svdm` /
`svdm_m` = the recommended mixed weights 1 + (E(n) + E_vol)/Cσ² on `seg_time_grid` with energy / count / mass voxel
weights (`svdm_e` = `--seg-basis svdn --seg-weight energy`). Field variants: `abs1` = degree-1 part removed (= its
exact per-sample absorption); `shimlin` = + ±150 Hz (y) ±100 Hz (z) linear residual over the tissue box; `shimquad` /
`shimquadn` = + / − a Z2-like ±150 Hz residual (`shimquadn` widens the ph0001 field, `shimquad` happens to narrow it);
`+abs2` = degree-2 part removed exactly (ideal). Degree 2 through the Taylor path (`abs2t`, linear exact + quadratic
per sample at order 2): R ≈ 1360–1370 at every L and basis — omitted. Each variant's R is referenced to the noise of
the real data (the factor in the header rescales the variant's own model-to-data energy match to the raw field's).
Concomitant phase in this evaluation: isotropic rotation average of the readout's Maxwell terms (≤ 0.31 cycles); no
R2*, no GNL (both are in the Rust predictions of §5).

Exact-metric R per (variant, basis, L); near-centre / far split and the separable proxies per configuration in
`experiments/segment_basis/res3_*.jsonl` (scripts `segeval.py`, `sweep3.py`, `tables.py`; Rust predictions
`rule_*.json`) (for every weighted SVD basis > 90 % of R² is in the far region |k| ≥ 120 rad/m; for fixed
nodes about a quarter of R² is near the centre).

#### water, field raw (R = data error rms / noise rms; noise-reference correction x1.000)

| basis | L=6 | L=7 | L=8 | L=9 | L=10 | L=11 | L=12 | L=13 | L=14 | L=16 | L=17 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cheb |  |  | 0.958 |  | 0.455 |  | 0.272 |  | 0.155 |  |  |
| fixed | 3.013 |  | 1.100 |  | 0.509 |  | 0.221 | 0.148 | 0.094 | 0.040 | 0.021 |
| fixed_e |  |  |  |  |  |  | 0.065 |  | 0.026 |  |  |
| svd | 11.702 |  | 2.001 |  | 0.433 |  | 0.121 |  |  |  |  |
| svdm | 2.065 |  | 0.611 | 0.387 | 0.261 |  |  |  |  |  |  |
| svdm_e | 0.883 | 0.470 | 0.254 | 0.150 | 0.079 | 0.035 | 0.016 |  | 0.004 |  |  |
| svdm_m |  |  | 0.470 | 0.274 | 0.188 |  |  |  |  |  |  |
| svdn_e | 0.987 |  | 0.315 |  | 0.104 |  | 0.019 |  | 0.005 |  |  |
| svdw | 2.320 |  | 0.844 |  | 0.328 |  | 0.176 |  | 0.074 |  |  |

#### water + fat (frame-0 fat-fraction map folded into g), ph0001 field (R = data error rms / noise rms; noise-reference correction x1.000)

| basis | L=8 | L=10 | L=11 | L=12 | L=13 | L=14 | L=16 | L=18 | L=20 | L=22 |
|---|---|---|---|---|---|---|---|---|---|---|
| cheb |  | 1.145 |  | 0.800 |  | 0.285 | 0.089 |  |  |  |
| fixed |  |  |  | 0.703 |  | 0.247 | 0.109 | 0.047 | 0.017 | 0.007 |
| fixed_e |  |  |  |  |  | 0.138 | 0.053 |  |  |  |
| svdm_e | 0.753 | 0.353 | 0.231 | 0.133 | 0.072 | 0.034 | 0.007 | 0.002 |  |  |

#### water, field abs1 (R = data error rms / noise rms; noise-reference correction x0.985)

| basis | L=6 | L=8 | L=9 | L=10 | L=12 | L=14 |
|---|---|---|---|---|---|---|
| fixed | 3.054 | 0.954 |  | 0.395 | 0.183 | 0.077 |
| svdm_e |  | 0.263 | 0.158 | 0.083 |  |  |
| svdn_e | 1.070 | 0.334 |  | 0.109 | 0.021 | 0.004 |
| svdw | 2.393 | 0.763 |  | 0.278 | 0.154 | 0.063 |

#### water, field shimlin (R = data error rms / noise rms; noise-reference correction x0.826)

| basis | L=8 | L=10 | L=12 | L=14 | L=16 |
|---|---|---|---|---|---|
| fixed | 3.281 | 1.105 | 0.371 | 0.157 | 0.055 |
| svdm_e | 0.946 | 0.280 | 0.052 |  |  |
| svdn_e | 0.972 | 0.317 | 0.065 | 0.011 | 0.002 |

#### water, field shimlin+abs1 (R = data error rms / noise rms; noise-reference correction x0.985)

| basis | L=8 | L=10 | L=12 | L=14 |
|---|---|---|---|---|
| fixed | 0.954 | 0.395 | 0.183 | 0.077 |
| svdn_e | 0.334 | 0.109 | 0.021 | 0.004 |

#### water, field shimquad (R = data error rms / noise rms; noise-reference correction x0.884)

| basis | L=8 | L=10 | L=12 | L=14 |
|---|---|---|---|---|
| fixed |  | 0.293 | 0.134 | 0.059 |
| svdm_e | 0.156 | 0.038 | 0.008 |  |

#### water, field shimquadn (R = data error rms / noise rms; noise-reference correction x0.500)

| basis | L=8 | L=10 | L=12 | L=14 |
|---|---|---|---|---|
| fixed |  | 0.857 | 0.405 | 0.190 |
| svdm_e | 0.518 | 0.178 | 0.056 |  |

#### water, field shimquadn+abs2 (R = data error rms / noise rms; noise-reference correction x0.204)

| basis | L=8 | L=10 |
|---|---|---|
| svdm_e | 0.170 | 0.038 |

## 4. Smooth-field absorption

What the tables can carry exactly: the constant part c₀τ_j (phase φ0) and the linear part (c₁·ρ)τ_j (k-space shift
c₁τ_j of the sample: exact for the NUFFT, like the concomitant linear terms). The quadratic part Q(ρ)τ_j can only go
through the ΔC term, which the kernel applies as a Taylor expansion of e^{−iρᵀΔCρ} (order 1/2, built for the
≤ 0.3 rad concomitant deviation); splitting Q τ_j = Q τ_ref (static voxel phase, rank-neutral, left in the diagonals)
+ Q (τ_j − τ_ref) leaves |Q| T/2 as the Taylor argument. Degree 3/4 would go through the first-order-only order-3
path. `--b0-absorb 1|2` implements degree 1 (and 2, with the Taylor phase reported); the fit is signal-energy
weighted over the support.

Measured (water, R in noise rms; shim residuals are synthetic, added over the tissue box; R of the variants with a
changed field is referenced to the noise of the real data, factor in the table header of §3):

| field | absorbed | fixed L = 10 / 12 / 14 | svdm_e L = 8 / 10 / 12 |
|---|---|---|---|
| ph0001 | none | 0.51 / 0.22 / 0.094 | 0.25 / 0.079 / 0.016 |
| ph0001 | degree 1 | 0.40 / 0.18 / 0.077 | 0.26 / 0.083 (L = 8 / 10; 0.16 at 9) |
| + linear ±150 Hz (y), ±100 Hz (z) | none | 1.11 / 0.37 / 0.157 | 0.95 / 0.28 / 0.052 |
| + linear residual | degree 1 | 0.40 / 0.18 / 0.077 (identical to ph0001 + degree 1) | = ph0001 + degree 1 |
| + Z2-like ±150 Hz widening the field | none | 0.86 / 0.41 / 0.19 | 0.52 / 0.18 / 0.056 |
| + Z2-like residual | degree 2 removed exactly (ideal, not realisable) | — | 0.17 (L = 8) |
| + Z2-like ±150 Hz (opposite sign, narrows the field) | none | 0.29 / 0.13 / 0.059 | 0.16 / 0.038 / 0.008 |
| same | degree 2 via the ΔC Taylor path | ≈ 1400 at every L | ≈ 1400 |

* On ph0001 the smooth part is small (degree-1 weighted fit 234 Hz + [9, 199, −767] Hz/m; 1–99 % field range 639 →
  648 Hz), so degree-1 absorption gains ≈ 20 % in R for fixed nodes (≈ ½ segment) and nothing for the weighted SVD
  basis (it is already insensitive to a global frequency and nearly so to a gentle ramp; Rust prediction 0.208 vs
  0.230 at L = 8). The coordinator's degree-1…4 fits (wrms 93 → 78 Hz at degree 4) say the same: the field is
  shimmed; what is left is local susceptibility structure.
* A linear shim residual of ±150 / ±100 Hz costs ≈ 1.5–2 segments with either basis, and degree-1 absorption removes
  it exactly (the R of "residual + degree 1" equals that of "ph0001 + degree 1" to all digits): free robustness for
  real data with an imperfect linear shim. Recommended on (`--b0-absorb 1`).
* Quadratic absorption through the Taylor path is unusable at realistic curvature: the weighted degree-2 fit has
  |Q| up to 870 Hz at the edge of the tissue box (21.8 rad Taylor argument) and R ≈ 1400. Even the ideal (exact)
  removal of the degree-2 part would only recover ≈ 1–2 segments of a ±150 Hz Z2 residual (0.52 → 0.17 at L = 8). An
  exact quadratic correction per sample would need a chirp (Fresnel) convolution in k-space per sample — not a
  kernel tweak. Degree 2 is kept as an option for curvature of a few Hz only (the run reports the Taylor phase).


## 5. Recommendation

`--nseg auto:noise:0.2 --seg-basis svdn --seg-weight energy --b0-absorb 1`

* **Rule.** L = the smallest L whose predicted data error, ×1.5 margin, is ≤ R = 0.2 noise rms:
  R_pred(L) = ε_w(L) √(mean_n w_n) with w_n = 1 + (E(n) + E_vol)/(Cσ²) and voxel weights |x̄|²Σ|S|². For the SVD basis
  ε_w(L) is the eigenvalue tail (no segments built); for fixed nodes the grid error `nodes_weighted_error`. R = 0.2
  means the model error adds 4 % to the residual energy; the margin covers the separable surrogate's optimism (≤ 1.3×
  on the CPU-exact metric in Python, ≤ 2× between the Rust prediction and the Python exact values at L = 9–10). The
  noise σ is the per-whitened-coil noise (sim: `noise_sigma`; real data whitened with a measured Ψ: 1).
* **Basis.** The weighted rank-L basis with those weights on the `seg_time_grid` (96 samples, a third uniform, the
  rest equidistributing √w — the k-space-centre samples get ADC-sample resolution) and energy-weighted
  interpolators. On ph0001 it needs 9 segments where today's basis needs 13–14 at the same error. The ingredients
  matter in this order: energy voxel weights (≈ 2 segments vs count, ≈ 1 vs mass), the white floor in the time
  weights (≈ ½–1 segment vs SNR-only, and it makes the prediction honest), the noise referencing itself (svdw's
  data-energy weights on the 96-point uniform grid: 12 segments for R = 0.18).
* **Fixed nodes** (needed for joint fat with per-frame maps, whose operator applies the fat factor per node): uniform
  nodes with energy-weighted interpolators (`--seg-weight energy`; `svdn` falls back to this there) gain ≈ 2½
  segments over today's unweighted ones (water L = 12: 0.065 vs 0.22). Chebyshev nodes (clustered at the k-space
  centre) are *worse* under this metric (water L = 12: 0.27): the middle of the readout matters at the white level.
* **Absorption** degree 1 always (free, exact; recovers the 1.5–2 segments a linear shim residual costs); degree 2
  only for curvature of a few Hz (§4).
* **Outliers** need no special handling: the < 0.1 % interface voxels that set today's full-range count enter with
  their signal energy, and their error is part of every R above (the fit and the evaluation cover the whole dilated
  support). The worst-voxel range rule over-provisions: at today's L = 17 the water operator is at R = 0.02.

Expected L for ph0001 (rule as implemented, Rust `--rule-only`, real tables with GNL and the encode-mean concomitant
coefficients, 31 virtual coils; fat with R2* and the frame-0 map):

| case | today (`--nseg auto`) | fixed, rule | fixed + energy, rule | svdn + energy, rule |
|---|---|---|---|---|
| water | 17 (R 0.021) | 14 (R 0.094) | 11 (R ≈ 0.1) | **9** (R 0.15) |
| water + fat | 22 (R 0.007) | 18 (R 0.047) | 15 (R ≈ 0.09) | **12** (R 0.13, one map) |

(R in parentheses: the CPU-exact value at that L from §3 where measured; fat values without R2*.) Cost per set:
water 1.31 → 0.69 s (−47 %), joint water + fat 1.70 → 0.92 s with one folded map (−45 %), 1.70 → 1.16 s (−32 %) with
per-frame maps (fixed nodes + energy weights, L = 15).


## 6. Cost

The operator is FFT-bound: each segment costs 2·C fine-grid FFTs per set and gradient (C = 31, grid 318 × 180 × 143),
plus its pad / crop pass and its share of the tile kernels (they interpolate L segment channels per sample). The
production run (L = 22, joint water + fat, per-frame maps) measured ≈ 1.7 s per set forward + adjoint, ≈ 0.09 s of it
in the tile kernels; with everything proportional to L, t_set ≈ 0.077 s × L (taking the kernels as L-independent
instead: 0.09 + 0.073 L — the same to ±4 % in the range below).

| configuration | L | s / set | vs today |
|---|---|---|---|
| water, `--nseg auto` (fixed, full field range) | 17 | 1.31 | — |
| water, fixed, rule (R ≤ 0.2 incl. margin; exact R 0.094) | 14 | 1.08 | −18 % |
| water, fixed + energy, rule | 11 | 0.85 | −35 % |
| water, svdn + energy, rule (exact R 0.15) | 9 | 0.69 | −47 % |
| water + fat, `--nseg auto` (production) | 22 | 1.70 | — |
| water + fat, fixed, rule | 18 | 1.39 | −18 % |
| water + fat, per-frame maps, fixed + energy, rule | 15 | 1.16 | −32 % |
| water + fat, one map, svdn + energy, rule (exact R 0.13) | 12 | 0.92 | −45 % |

Absorption costs nothing per set: degree 1 changes only the sample coordinates and φ0 before the tile lists are
built; degree 2 writes into the ΔC terms that order 2 already evaluates for the concomitant deviation. (The order-3
path — degree-3/4 terms to first order, +25 FMA per tile row in the staging per PLAN.md — would add a few ms per set,
far below one segment's ≈ 77 ms, but it is first order only and gains nothing here, §4.) Building the segments is a
setup cost: the weighted Gram is one pass over the fit voxels × 96 times (≈ 1–4 s), the rule then needs no further
work for the SVD basis (fixed nodes: ≈ 2–3 s per candidate L at full CPU, one LS solve per grid time).


## 7. Implementation (additive; defaults unchanged)

All in `rustlib/.agents/seg` (merge by diff); no existing default, signature or output changes.

* `crates/hasty-nufft/src/segbasis.rs` (new module, exported from `lib.rs`):
  `seg_time_grid` (time samples + quadrature weights resolving a peaked per-ADC weight), `snr_time_weights`
  (1 + E/Cσ²), `vol_energy` + `noise_time_weights` (1 + (E + E_vol)/Cσ², the recommended weights),
  `weighted_spectrum` / `tail_errors` / `seg_count_for` (the rule from the Gram eigenvalues),
  `build_segments_svd_grid` (weighted rank-L basis on an explicit grid, voxel-weighted LS table), `cheb_nodes`,
  `build_segments_nodes` (fixed diagonals at any node times, weighted LS), `build_segments_fat_nodes` (ff-independent
  joint water + fat table at any nodes with voxel weights, for the per-frame fat-map operator), `nodes_weighted_error`
  (weighted error of fixed nodes on the grid without building a table: the rule for fixed / Chebyshev nodes),
  `segment_model_error` (weighted error of built segments at the ADC times). `segnufft.rs`: one `pub(crate)` wrapper
  of the private `ls_interpolators`.
* `crates/hasty-recon/src/segphys.rs`: `FieldPoly` (+ `residual`, `residual_at`), `fit_field_poly(_at)`,
  `absorb_field_poly(_at)`: degree 0/1/2 B0 part into φ0 / k-shift / ΔC of the sample tables (the static part Q τ_ref
  stays in the diagonals). `kload.rs`: `SimData::adc_energy_whitened` (un-normalised whitened energy per ADC sample).
* `recon_p2np` (new values only): `--nseg auto:noise[:R]` (default R 0.2), `--seg-basis svdn` (weights
  `noise_time_weights` on `seg_time_grid`, voxel weights of `--seg-weight`), `--seg-basis cheb`, `--seg-weight energy`
  (smoothed warm-start magnitude² × coil energy / median), `--b0-absorb 0|1|2` (fit weighted by the same energy; the
  plain `auto` count also uses the residual field). `--seg-basis fixed --seg-weight energy`: the uniform nodes with
  energy-weighted interpolators (`build_segments_nodes`). Joint fat with per-frame maps: `cheb`, `svdn` (→ uniform
  nodes with its voxel weights) and `--seg-weight energy` use `build_segments_fat_nodes`; `fixed|svd|svdw` with the
  old weights keep `build_segments_fat` exactly as before.
  σ for the weights: `HASTY_NOISE_SIGMA`, else the sim's `noise_sigma`, else 1 (data whitened with a measured Ψ).
* `hasty-bench/src/bin/segbudget.rs` (new): the real-operator benchmark per (absorption, basis, L): s/set forward +
  adjoint, R against a high-L reference on the warm start (scaled to the measured k-space centre), error against an
  exact f64 direct sum (true per-sample concomitant coefficients, fat, R2*) on an energy-sampled voxel subset;
  `--rule-only` = the CPU prediction R_pred(L) of the rule (used for §5).
* Tests (CPU, pass): `hasty-nufft --test seg_rule_cpu` (time-grid quadrature on a peaked weight, tail = built-basis
  error within ×1.5, noise-weighted basis ≤ unweighted SVD and fixed nodes in the noise metric, rule monotone in the
  field spread and L = 1 for one frequency, grid error of fixed / Chebyshev nodes = tabulated error, white-floor
  weights: flat-profile identity, volume average ≪ time average for a centre-peaked profile), `hasty-recon
  --test seg_absorb_cpu` (fit recovers degree-1/2 fields; absorbed tables + residual segments reproduce every
  (sample, voxel) phase to 2e-5 rad, the quadratic Taylor phase is reported); existing `seg_basis_cpu` still passes.
  GPU tests: none run (§8).

GPU measurement to run (writes JSON; ≈ 10–20 min, ≤ 20 GB device):

    cd /home/turbotage/Documents/GitHub/HastyCompute && HASTY_REPO=$PWD systemd-run --user --scope -q \
      -p MemoryMax=16G -p MemorySwapMax=0 nice -n 10 rustlib/.agents/seg/target/release/segbudget \
      data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan \
      --mask data/hasty/phantom/factory_arms/ph0001/lock_2mm/llr/mask_maps.npy \
      --x data/hasty/phantom/factory_arms/ph0001/fix_2mm/llr_s/warm_full.npy --ref 0:fixed:28 \
      --cfgs "0:fixed:17;0:fixed:14;0:fixed_e:11;0:fixed_e:12;0:svdn_e:8;0:svdn_e:9;0:svdn_e:10;1:svdn_e:9;2:svdn_e:9;0:svdn_e:auto" \
      --json segbudget_water.json
    # water + fat (frame-0 map folded) with R2*: add --fat truth --r2s truth and e.g.
    #   --cfgs "0:fixed:22;0:fixed:18;0:fixed_e:15;0:svdn_e:11;0:svdn_e:12;0:svdn_e:13;1:svdn_e:12;0:svdn_e:auto"
    # (segbudget's `svdn_e` = recon_p2np's `--seg-basis svdn --seg-weight energy`; HASTY_SEG_WEIGHTS=snr for SNR-only)


## 8. Limitations

* **No GPU run.** The GPU was announced free, but launching the benchmark was refused by the session's permission
  check, so the real-operator s/set and the operator-vs-direct-sum check are not measured; `segbudget` builds but is
  untested on the device. Seconds per set are the production run's 1.7 s at L = 22 scaled linearly in L (every
  segment adds 2·C FFTs of 318 × 180 × 143, its pad / crop and its share of the tile kernels).
* The exact metric is a CPU emulation (block-DFT near the centre, rss-coil FFT shells elsewhere; ±30 % per sample,
  validated against the full DFT on 48 readouts). R values are for ph0001's warm start, encode 0, one cardiac frame's
  fat map, and an isotropic rotation-average concomitant phase (≤ 0.31 cycles) instead of the encode-mean w̄; GNL and
  R2* are only in the Rust `--rule-only` predictions.
* The warm start `fix_2mm/llr_s/warm_full.npy` is not in data units under my forward model (its k-space centre has 16×
  the measured energy, ±30 % over the readout): all R are referenced to the measured data (model/measured energy ratio
  fitted at |k| < 30 rad/m). Origin not traced (map normalisation or data scaling of that run).
* The rule's prediction R_pred = ε_w √(mean w) is a separable surrogate. On ph0001 it tracks the exact R within ~15 %
  for the svdm basis at L ≤ 8 and is optimistic up to ×2 at L = 9–10 (exact 0.15 / 0.079 vs predicted 0.097 / 0.037
  in the Rust run; hence the 1.5 margin, which does not change L at R = 0.2); for uniform unweighted fixed nodes it is
  pessimistic ×1.2–2.5, for energy-weighted fixed nodes ×1.2 optimistic; for Chebyshev nodes it is ×2.7 optimistic
  (do not use the rule with `cheb`). One phantom: the white-floor term and the
  energy voxel weights are physically motivated, but the calibration has not been repeated on a second anatomy.
* Joint water + fat with **per-frame** maps (production) cannot use the rank-L SVD basis: the operator applies the fat
  factor per node (P_l (1 + f (c_l − 1))); an SVD needs separate water and fat diagonal sets (two diagonals per
  segment in the pad / crop kernels, not done: GPU code I could not test). Until then that path gets uniform nodes +
  energy-weighted interpolators (L = 15 instead of 12).
* R is a data-space criterion. A systematic error at the densely sampled k-space centre is averaged less than the
  noise there; in image space the low-resolution bias can be relatively larger than R suggests (and is common to all
  encodes, so it largely cancels in phase-contrast velocity). Not quantified here.
* Degree-2 absorption through the ΔC Taylor path is only valid for a few Hz of curvature (§4); degree 3/4 would need
  the first-order-only order-3 path and was not implemented for B0.

