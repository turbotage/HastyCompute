# Why the pooled image showed no anatomy (2026-11-22)

**Question.** The §12–§13 dev case (temporal_pyramid.md) has a pooled start x̄ (one image per encode from all 20 cardiac
frames, `--warm-full 60`) whose magnitude correlates with the proton density at 0.02, and whose velocity error grows
with iterations, even on a noise-free inverse crime. Pooled over 20 frames the data should give an image with clear
anatomy. Bug or conditioning?

**Answer.** Two causes, both outside the pyramid code.
1. **A plain bug in the inverse crime (fixed).** With staged data (`--plan host`) the grouped gradient that pooled PCG
   uses formed its residuals as E x − y_store, i.e. from the MEASURED data in the k-space store, not from the
   synthetic residual E(x − x★). So every pooled solve of a §13 "inverse crime" (`--synthetic --synth-x pc --plan
   host --warm-full`) actually solved the measured problem. That is why the crime reproduced the measured numbers "to
   2 digits" (1.757 vs 1.756 m/s median).
2. **The circulant metric is a harmful preconditioner for this pooled solve (switch added, default unchanged).** On
   the static object (no blood dynamics, no noise, the recon's own operator) pooled PCG with the per-encode circulant
   makes the image error GROW: relative error 1.07 after 1 iteration, 2.42 after 30, 2.95 after 60; correlation of |x|
   with the PD in the body −0.12 → 0.01. Plain CG on the same system converges: 0.31 after 30 iterations, PD
   correlation 0.87. The image the §13 note measured (0.02) was truly without anatomy; the analysis was not the
   problem.

All numbers below are noise-free inverse crimes with a realistic object (x★ = PD_water + PD_fat box means at each
voxel's true, GNL-inverted position; with `--synth-x pc` plus the truth blood signal per frame). Scripts:
`research/experiments/pooled_diag/run_pool_diag.sh` (stages test, a–e), `run_dev_solvable.sh`, `err_spec.py`. Run
directories: `data/hasty/phantom/factory_arms/ph0001/pool_diag/` (pooled images deleted after scoring). GPU ≈ 52 min.

## 1. Common setup
- **Data:** ph0001 sim_flow_10min (yarn readouts, 3990 ADC samples each), 128 × 72 × 60 grid (`--grid-scale 0.5`,
  552 960 voxels), k-space cropped to the grid's ball |k·Δ| ≤ π (`--kcrop`, keeps 22.8 % of the samples), 8 virtual
  coils, 20 cardiac frames × 7 encodes = 140 sets. No held-out readouts in these runs.
- **Sampling levels** (`--keep K` = 1/K of the readouts, hash-spread):

| keep | readouts | per (frame, encode) set | samples per set after the crop | samples per voxel per set (per coil) | pooled per encode, samples per voxel |
|---|---|---|---|---|---|
| 16 | 3301 | 13–35 | 11.8 k – 31.8 k | 0.02–0.06 | 0.78 |
| 4 | 13228 | 77–121 | 70 k – 110 k | 0.13–0.20 | 3.1 |
| 2 | 26674 | 162–214 | 147 k – 195 k | 0.27–0.35 | 6.3 |

  (The ball holds 52 % of the cube's Fourier modes; the corners have no samples at all.)
- **Unknowns of the pooled solve:** one complex image per encode (7 × 552 960).
- **Operator:** full physics unless stated: 6 ratemap/B0 segments, concomitant fields (degree-4 fit), GNL ("full").
  "Plain": `--phys none`, no GNL (one segment, plain NUFFT).
- **Algorithm:** pooled PCG per encode on N_e x_e = Σ_{f} A_{f,e}ᴴ y_{f,e} from zero, 30 or 60 iterations.
- **Preconditioner:** "circ" = the per-encode circulant metric M (the ZMSHR1-M metric: inverse of the Fejér-smoothed
  sampling density convolved with the coil power, floored at ε = 1e-3 λ_max); "none" = plain CG
  (`--pool-metric none`, new).
- **Measures** (printed every 5 pooled iterations, new, scoring only): relative error ‖x − x̄★‖/‖x̄★‖ against the
  frame mean of x★ (whole grid, static body, blood voxels, air relative to ‖x★‖), the complex LS scale ⟨x̄★, x⟩/‖x̄★‖²,
  the Pearson correlation of |x| (encode 0) with the PD over the body. The pc velocity score (`--score-pc`, §13.1 of
  temporal_pyramid.md) at it 0 of a 1-iteration P²nP run is the pooled image.

## 2. Controls (static object `--synth-x pd`, keep 16 unless stated)

| run | operator | preconditioner | rel error it 1 / 10 / 30 / 60 | static body it 30 | PD corr (body) it 30 | LS scale it 30 |
|---|---|---|---|---|---|---|
| a_k16_pd | full | circ | 1.07 / 1.58 / 2.42 / 2.95 | 2.35 | 0.035 (it 60: 0.014) | 0.28 |
| d_full_circ3d (= a, see §3) | full | circ | 1.07 / 1.58 / 2.42 | 2.35 | 0.035 | 0.28 |
| c_full_none | full | **none** | 0.92 / 0.54 / **0.31** | 0.30 | **0.87** | 0.87 |
| b_plain | plain | circ | 1.19 / 1.54 / 0.92 | 0.52 | 0.67 | 0.95 |
| c_plain_none | plain | none | 0.92 / 0.40 / 0.20 | 0.19 | 0.90 | 0.94 |
| c_gnl | GNL only | circ | 1.22 / 1.60 / 0.97 | 0.56 | 0.65 | 0.95 |
| c_phys_nognl | ratemap + concomitant, no GNL | circ | 1.06 / 1.56 / 2.40 | 2.34 | 0.03 | 0.27 |
| e_k16_eps1e2 | full | circ, ε = 1e-2 | 1.02 / 1.19 / 1.41 | 1.34 | 0.07 | 0.46 |
| e_k16_eps1e1 | full | circ, ε = 1e-1 | 0.96 / 0.77 / 0.57 | 0.54 | 0.69 | 0.78 |
| e_k4_circ | full, keep 4 | circ | 1.06 / 1.58 / 2.35 | 2.31 | 0.06 | 0.29 |
| e_k4_none | full, keep 4 | none | 0.92 / 0.54 / 0.29 | 0.28 | 0.88 | 0.88 |
| e_k2_circ | full, keep 2 | circ | 1.06 / 1.56 / 2.19 | 2.15 | 0.07 | 0.29 |
| e_k2_none | full, keep 2 | none | 0.92 / 0.55 / 0.29 | 0.29 | 0.88 | 0.88 |

- **Adjoint test of the exact solver operator** (`--adj-test`, new; random x, y): |⟨Ex, y⟩ − ⟨x, Eᴴy⟩| / (‖Ex‖‖y‖)
  = 2.4e-7, 9.4e-8, 5.6e-7 (sets 0, 70, 139, full model); 7e-7, 3e-7, 1e-7 (plain). Grouped vs member-wise forward /
  adjoint (4 sets of encode 0): 6.9e-5 / 1.7e-4 (full), 2.8e-6 / 8.2e-5 (plain). The operator and the grouped path
  are fine.
- **Plain CG converges at every sampling level** (rel error 0.29–0.31 at 30 iterations, LS scale 0.87–0.88, PD
  correlation 0.87–0.88), and the three levels are nearly equal: at 30 iterations CG is limited by its own
  convergence (the low frequencies come first), not by the data.
- **With the circulant the error grows at every level** (keep 16, 4, 2: 2.42, 2.35, 2.19 at 30 iterations) and with
  the full physics more than with the plain model (λ_max(M N) = 51 full vs 9.8 plain). The ratemap / B0 / concomitant
  part causes it (c_phys_nognl ≈ full), not GNL (c_gnl ≈ plain).
- **A weaker metric is less harmful, never helpful:** ε = 1e-2: 1.41; ε = 1e-1: 0.57; none: 0.31.
- **Where the error goes** (`err_spec.py`, error energy / object energy per shell of the normalised radius
  r = |k|/k_Nyquist, encode 0):

| shell r | 0–0.125 | 0.125–0.25 | 0.25–0.5 | 0.5–0.75 | 0.75–0.9 | 0.9–1.0 | 1.0–1.2 | > 1.2 (corners) |
|---|---|---|---|---|---|---|---|---|
| circ, keep 16, 60 it | 0.87 | 2.87 | 4.98 | 10.6 | 17.1 | 22.3 | 34.8 | 45.5 |
| none, keep 16, 30 it | 0.16 | 0.57 | 0.74 | 1.00 | 1.29 | 1.46 | 1.83 | 2.28 |
| circ, keep 2, 30 it | 0.90 | 2.20 | 4.07 | 8.47 | 12.3 | 14.9 | 21.1 | 26.8 |
| none, keep 2, 30 it | 0.15 | 0.53 | 0.69 | 0.95 | 1.23 | 1.40 | 1.78 | 2.22 |

  With the circulant, 35 % of the pooled image's energy sits outside the sampled ball (r > 1) and only 3.5 % below
  r = 0.125 (the object: ≈ 86 %). The low frequencies, where the anatomy is, are barely reached (error 0.87–0.90 of
  the object there after 30–60 iterations, LS scale 0.28).
- **Why:** the density spectrum of this trajectory is extremely peaked: 80 % of the Fourier modes sit at the ε floor
  (99.8 % below 1e-2 λ_max). M therefore damps the k = 0 region by up to 1/ε = 1000 relative to everything else and
  amplifies the outer shell and the unsampled corners (whose diagonal is coil leakage only). N is far from diagonal in
  k (coils, and the ratemap/B0 phase over the 8 ms readout), so M N has λ_max = 51 in directions the circulant gets
  wrong, and PCG fills the weakly determined outer modes long before the anatomy. PCG still decreases ‖e‖ in the M⁻¹
  norm; with M's dynamic range 1000, ‖e‖₂ may grow up to ≈ 30×, and it does grow ≈ 3×. This is a property of the
  metric, not a coding error; I did not change it. (A 3D floor of the unsampled corners, generalising the in-plane
  spiral rule, was tried and reverted: the boundary λ is 1e-4 λ_max, below ε, so it changed nothing, d_full_circ3d.)
- **Old §13 analysis.** The 0.02 correlation (warm.npy, ad hoc, no script) agrees with the in-binary measure on an
  exact reference (0.01–0.04 at 30–60 iterations): the image had no anatomy; the measure was not the issue.

## 3. Blood dynamics (pooled static model on dynamic data, `--synth-x pc`, keep 2, plain CG, 60 iterations)
- Relative error against the frame mean x̄★ at it 1 / 10 / 30 / 60: whole 0.92 / 0.54 / 0.29 / **0.166**; static body
  0.907 / 0.514 / 0.281 / **0.159**; blood voxels 1.009 / 0.644 / 0.369 / **0.197**; air 0.038 of ‖x★‖. PD correlation
  (body) 0.93, LS scale 0.96.
- So in 60 noise-free iterations the inconsistency of a static model for pulsating blood costs little: blood voxels
  0.197 vs static tissue 0.159, and the error is still falling in both. No semi-convergence visible yet.
- **pc velocity score of the pooled image:** RMSE 0.381 m/s (time mean 0.138, detail 0.355), median 0.126, p90 0.554,
  phase 0.508 rad. The time-averaged-truth oracle: 0.378 (0.129 / 0.355), median 0.137, p90 0.545, phase 0.519. **The
  pooled image reaches the time-averaged truth.** (§13: 2.21 / median 1.76 / phase 1.59 at keep 16 with the bug and
  the circulant.)

## 4. The fix and tests
- `hasty-recon/src/setop.rs`: `SetOp::residual_ignores_data()` (default false; `synth::SynthSets` true).
  `hasty-recon/src/p2np.rs` `gradient_groups_host`: staged data + such an operator → `forward_residual_group` (the
  synthetic residual) instead of forward minus the staged store data. No switch: plain bug.
- Regression test `tests/grouped_sets.rs::pooled_pcg_synthetic_ignores_staged_data` (GPU): a store holding a DIFFERENT
  object; staged grouped, staged replicate and resident grouped pooled PCG must agree (1.2e-4, 1.3e-4) and reach x★
  (0.23 after 12 iterations), not the store's object (1.11). Passes with the fix (not run against the old code).
- New switches (default off, no change to existing runs): `--synth-x pd` (static object only), `--pool-metric
  circ|none` (pooled PCG preconditioner of `--warm-full` / `--tpyramid`; default circ = old behaviour), `--adj-test`.
  New prints: the pooled image error vs x★ (synthetic runs only), `HASTY_PRECOND_DIAG=1` (share of modes at the
  circulant floor). `synth_pd.npy` (the static object, 2 MB) is written by `--synth-x pc|pd` runs.
- Not fixed: `RcSets` (`--rcompress`) over `SynthSets` has the same pattern (its default `forward_residual` subtracts
  the passed data); not used in these runs.

## 5. The solvable dev case (`run_dev_solvable.sh`, KEEP=2)
- Data and operator as §1, keep 2: 26 674 readouts, 162–214 per (frame, encode) set, 0.27–0.35 samples per voxel per
  set per coil; pooled per encode 6.3 samples per voxel. Noise-free inverse crime x★ = PD + truth blood.
- Pooled start: 60 plain-CG pooled iterations (`--pool-metric none`), 210 s. Then P²nP, LLR κ 1, 30 iterations, from
  the pooled start and from zero, with the circulant-metric preconditioner (zmshr1m) and without (zmshr1).
- Acceptance: pooled image = §3 (median 0.126 vs oracle 0.137, RMSE 0.381 vs 0.378, PD correlation 0.93) ✔.
- **Per-frame P²nP** (pc score; zero image 0.425 RMSE / median 0.131 / p90 0.698; time-averaged truth 0.378 /
  0.137 / 0.545):

| start | preconditioner | it 3 RMSE / median | it 12 | it 30 RMSE (time mean / detail) | it 30 median / p90 | it 30 phase rad |
|---|---|---|---|---|---|---|
| pooled (plain CG) | none (zmshr1) | 0.378 / 0.123 | 0.349 / 0.095 | **0.353** (0.126 / 0.330) | **0.065 / 0.269** | 0.321 |
| pooled (plain CG) | circulant (zmshr1m) | 0.380 / 0.124 | 0.374 / 0.116 | 0.366 (0.141 / 0.338) | 0.100 / 0.479 | 0.459 |
| zero | none (zmshr1) | 0.421 / 0.147 | 0.405 / 0.133 | 0.382 (0.197 / 0.327) | 0.110 / 0.553 | 0.496 |
| zero | circulant (zmshr1m) | 0.539 / 0.205 | 0.549 / 0.193 | 0.475 (0.229 / 0.416) | 0.158 / 0.700 | 0.594 |

- **Acceptance met:** from the pooled start without the metric, 30 iterations beat the zero image clearly on median
  (0.065 vs 0.131) and p90 (0.27 vs 0.70), and on RMSE by 17 % (0.353 vs 0.425), and they beat the time-averaged truth.
  The temporal detail is only beginning to come in (0.330 vs the truth's 0.355 RMS): 30 iterations are short.
- **The circulant metric also hurts per-frame P²nP here:** from zero it ends worse than the zero image (0.475); from
  the pooled start it is behind the unpreconditioned run at every iteration. The training f falls more slowly too
  (it 30: 3.0e12 vs 2.0e12 from the pooled start).
- Cost: setup 38 s, pooled start 210 s, ≈ 5 s per P²nP iteration (2.7 s gradient, 1.9 s LLR); the four runs ≈ 17 min.
- Tables: `research/experiments/pooled_diag/dev_k2/pcscore_*.csv`; all run summaries `pool_diag_results.txt`.

## 6. Consequences and open problems
- **Every pooled / warm start with `--warm-full` and a metric preconditioner so far (§12, §13, meanres, anchor `mean`)
  started from an image without anatomy**, on measured data too (the circulant problem is data-independent). The
  §12–§13 rankings of starts and anchors are void; rerun them on the dev case above with `--pool-metric none`.
- The same circulant is the default P²nP metric (ZMSHR1-M); §5 shows it hurts there too on this trajectory. Every
  earlier yarn result with zmshr1m should be read with that in mind (velocity_shrink_cause chose the circulant for
  its PSF; that was a different question and dataset). A better metric for this trajectory (one that does not damp
  the k-space centre by 1000, e.g. coil-aware, or a much milder density inverse) is open; for now use
  `--pool-metric none` and `--precond zmshr1` on the yarn dev case. Defaults are unchanged.
- Not run: the dev case's held-out k-space score, more than 30 P²nP iterations (does the detail error keep falling, or
  does it semi-converge?), keep 1, noisy data at the measured SNR.
- Measured-data pooled solve with plain CG not run (budget).
