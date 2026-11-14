# Velocity shrinkage: preconditioning, not physics (ph0001, 2 mm, 2026-11-12)

Follow-up of `research/recon_diag_2026-11-11.md` §5. Question: why are the reconstructed velocities biased low (slope
0.3–0.7) when a plain NUFFT reconstruction (PC-VIPR) of real data is close to the truth? Arrays and logs:
`data/hasty/phantom/factory_arms/ph0001/pyr_diag/` (`psf/`, `rowin/`, `warm100/`, `wfull_circ/`).

## 1. Yardstick

`research/experiments/recon_diag/sig_fit.py`: time-averaged images per encode (all frames pooled) against the ideal
frame-mean phase-contrast signal of the sim, `truth_pc_signal.npz` (`rustlib/tools/phantom/bias_truth.py --signal`:
Σ over the beat library of w s e^{−i k_e·u} over the six moment-matched sub-sources, spin history and intravoxel velocity
spread included). Slope / correlation of Δφ_e = arg(x_e x̄_0) against the prediction, deep-lumen depth bands
(voxels; lumen fraction > 0.5), only voxels where every |Δφ_pred| < π/2.

- Signal weighting itself is not the issue: signal-weighted vs equal-weight unit-magnitude prediction slope
  0.996–1.016 over all lumen voxels.
- **Partial coverage is**: 41 % of the voxels ≥ 5 deep in the static lumen mask have LBM coverage < 0.9 in some frame
  (wall / chamber motion puts tissue there; the prediction has no tissue term). `COVMIN=0.99` keeps the voxels that are
  blood in every frame (2382 at depth ≥ 5). All numbers below use it. Earlier "slope 0.85 deep" for the plain warm start
  mixed both effects.

## 2. The plain model on PC-VIPR-like data is accurate

`rowindow` (hasty-bench): plain SENSE NUFFT, PCGLS 30 it + circulant (eps 1e-3), every encode pooled over all frames,
12 strongest coils (78.8 % of the lumen sensitivity energy), every 2nd readout, only samples in a window after the echo.
The yarn readout is out-and-back: |k| reaches Nyquist at 4 ms and returns to k = 0 at 7.98 ms (samples.npy), i.e. low k
is sampled again ≈ 9.6 ms after the excitation.

| window (ms after the echo) | depth 1–2 | depth 2–3 | depth 3–5 | depth ≥ 5 |
|---|---|---|---|---|
| 0–1.5 (k ≤ 0.65 Nyquist) | 0.834 / 0.729 | 0.943 / 0.801 | 0.921 / 0.897 | **0.959 / 0.948** |
| 0–4 (out half, full res) | 0.826 / 0.696 | 0.958 / 0.775 | 0.923 / 0.867 | **0.958 / 0.936** |
| 4–8 (back half) | 0.578 / 0.511 | 0.595 / 0.531 | 0.701 / 0.669 | 0.790 / 0.827 |
| 0–8 (all) | 0.580 / 0.411 | 0.660 / 0.445 | 0.744 / 0.512 | 0.754 / 0.567 |
| warm.npy, 100 it, 31 vcoils, all readouts | 0.698 / 0.511 | 0.789 / 0.552 | 0.821 / 0.619 | 0.848 / 0.716 |

No implementation error in NUFFT, coil maps, encoding or decoder: the out half alone gives 0.96 deep (the rest: 2 mm
resolution, respiratory shift of the two bins against the truth at r = 0.49 [CHECK]). The late low-k samples of the back
half are inconsistent with a plain model (lumen field median −49 Hz, p5–p95 −160…+230 Hz; fat fraction 0.33 in a
10 mm ring around the lumen, −434 Hz = 3.5 cycles at 8 ms; blood displacement v·8 ms): adding them drops the
correlation 0.94 → 0.57.

## 3. Effective PSF of the warm-start path (`psfcheck`)

Point (130, 47, 33) (aorta, 8.2 voxels deep), encode 0 pooled, every 2nd readout, 12 strongest coils (95.9 % of
Σ|S(q0)|²), y = A δ, PCGLS:

| | it 1 | it 10 | it 20 | it 100 |
|---|---|---|---|---|
| x(q0), circulant eps 1e-3 | 0.024 | 0.506 | 0.540 | 0.561 |
| x(q0), circulant eps 1e-5 | 0.024 | 0.503 | 0.538 | 0.560 |
| x(q0), no preconditioner | 0.000 | 0.001 | 0.005 | 0.246 |

- Circulant: energy share 0.565 in the centre voxel, 0.835 within r ≤ 1, half-maximum radius < 1 voxel, converged by
  ≈ 20 it. Limit x(q0) ≈ 0.56 ≈ the k-ball to Nyquist / cube (π/6 = 0.52).
- λ(k) / λ_max median per |k| shell: 4.0e-2 (0.05 Nyquist) → 5.3e-3 (0.2) → 2.0e-3 (1.0); the 1e-3 floor clips 0 %
  inside Nyquist (only the cube corners): the floor is not the problem.
- The Re-sum share within r oscillates around 1 (1.51 / 0.85 / 1.29 / 1.15 / 0.72 / 0.60 at r = 1, 2, 3, 5, 8, 12) and
  does not decay: the 3-D sharp spherical cutoff (S(r) = (2/π)(Si(Kr) − sin Kr) → 1 − (2/π) sin Kr), not a broad PSF.

## 4. The production solver path is not preconditioned in Fourier

`--precond zmshr1s` (fix_2mm, pyr_2mm, pyr_inittruth, algo_2026-11-12) is ZMSHR1 per set in the **identity** metric;
only `circ`, `zmshr1m`, `zmshr1ms` are metric preconditioners (`Precond::metric`, p2np.rs). With a non-metric choice
`need_circ` is false (recon_p2np.rs `let need_circ`), so

- the time-resolved solver takes quasi-Newton steps without the Fourier-diagonal metric (per-set scalar + rank one), and
- the physics-model pooled warm start (`--warm-full`) runs plain CG: `pooled_pcg_host_with(.., metric = false)` copies
  z = r (hoststack.rs).

fix_2mm/llr_s/warm_full.npy (10 unpreconditioned pooled PCG iterations of the full model from the plain 20-it warm):

| | depth 1–2 | depth 2–3 | depth 3–5 | depth ≥ 5 |
|---|---|---|---|---|
| plain warm.npy (20 it, circulant) | 0.681 / 0.522 | 0.760 / 0.538 | 0.800 / 0.590 | 0.787 / 0.617 |
| warm_full.npy (full model, no precond) | **0.414** / 0.437 | 0.489 / 0.576 | 0.615 / 0.757 | 0.708 / **0.901** |

The full model fixes the inconsistency of the late samples (correlation 0.62 → 0.90 deep) but the unconverged high k
blends static tissue into the lumen: the shrink grows towards the wall, the signature of a broad effective PSF.
Same for the time-resolved ZMSHR1s runs (`recon_diag_2026-11-11.md` §5: slope 0.32 at the wall → 0.64–0.67 deep).
The inverse-crime test (`--synthetic`, slope 0.99) started from a resampled coarse reconstruction of a smooth x★
(warm magnitudes + true phases) and is not sensitive to slow high-k convergence.

Earlier note: `research/synthetic_4dflow_pipeline.md` (P²nP follow-up) found `zmshr1m` (ZMSHR1 in the circulant metric)
the fastest data fit of all variants.

## 5. Cost and sharing of the circulant metric

- `circbench` (2 mm, 254×144×114): z = M r or M⁻¹ r 0.77 ms device-resident, 4.26 ms host-resident (two N-grid
  arrays uploaded per application, `--circ-share encode`); build 2.0 s for one encode's 3.0e7 samples. `zmshr1ms`
  applies M / M⁻¹ five times per set per iteration (three would do: M⁻¹s, Mu, Mg; the secant is recomputed in pass 2):
  280 × 5 × 4.26 ms ≈ 6 s per iteration vs ≈ 303 s gradient + ≈ 150 s LLR → ≈ 1.3 %.
- `research/experiments/recon_diag/circ_share.py` (CPU, sample histograms on the N grid, σ = 0.5 bin): the pooled
  densities of the 7 encodes are identical (ratio 1.000 per shell: readout i plays rotation ⌊i/7⌋, encode i mod 7) —
  the 7 per-encode circulants coincide. One set (cardiac frame, both respiratory bins) vs the pooled encode × n_t/n_e:
  median 1.00 [p10 0.76, p90 1.23] below 0.05 Nyquist, 0.96 [0.45, 1.6] at 0.05–0.1, 0.62–0.68 [0.07, 2.4] beyond 0.2
  (angular undersampling, 380 of 7592 rotations). The shared metric is right up to the scalar n_e/n_t (absorbed by τ_t).
- Estimator: without prior unchanged (ML); with the fixed-strength Euclidean LLR prox the fixed point satisfies
  0 ∈ ∇f + (θ/α) P⁻¹ ∂R, i.e. it depends on the metric (reconstruction_operator_section.tex,
  Proposition `prop:metricfp`).

## 6. Result of `pyr_diag/wfull_circ` (circulant in the pooled full-model warm start)

fix_2mm/llr_s arguments with `--precond zmshr1ms` (per-encode circulant, 10 pooled PCG iterations from the same plain
20-it warm start, ‖r‖/‖r₀‖ 0.096 at the end). Slope / corr vs the signal-weighted truth (COVMIN = 0.99):

| depth | plain warm | full, no precond (fix_2mm) | full, circulant |
|---|---|---|---|
| 1–2 | 0.682 / 0.523 | 0.414 / 0.437 | 0.413 / 0.425 |
| 2–3 | 0.760 / 0.538 | 0.489 / 0.576 | 0.501 / 0.444 |
| 3–5 | 0.800 / 0.590 | 0.615 / 0.757 | 0.717 / 0.542 |
| ≥ 5 | 0.787 / 0.617 | 0.708 / 0.901 | 0.882 / 0.725 |

The circulant fixes most of the deep shrink (0.71 → 0.88) but not the wall (0.41 either way), and lowers the
correlation (high k enters only part-converged). The wall shrink is a full-model effect: the plain model gives 0.68 and
its outgoing half 0.83. Candidates [CHECK]: joint water + fat with the fat ring around the vessels; GNL registration of
the image vs the truth grid; mismodelled late samples. Next: full model on the outgoing half (window env), and
`--fat-model none` on the same window.

## 7. Running / next

- `pyr_diag/wfull_circ`: fix_2mm/llr_s arguments with `--precond zmshr1ms` (per-encode circulant in the pooled PCG).
  Expected: correlation ≈ 0.9 kept, slope towards the out-half 0.96 [CHECK].
- Then: time-resolved runs with `zmshr1ms`; `--warm-full` should use the circulant whatever the main preconditioner
  (additive switch or a warning); inverse-crime test from a sharp / zero start.
