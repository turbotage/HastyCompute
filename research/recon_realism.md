# De-oracling the reconstruction (2026-11-08)

Until now every reconstruction of the synthetic scans used three things a real scan does not have: the true coil
sensitivities, the true off-resonance field, and the true ECG. This note covers the first two, plus coil compression
and a first look at stochastic coil sampling. Implementation summary: rustlib/PLAN.md, "De-oracling the
reconstruction".

## 1. Coil compression by singular values

Virtual coil v = u_vᴴ y with u_v the eigenvectors of the coil Gram G = Σ_n y_n y_nᴴ (whitened data, so the rows are
orthonormal and white noise stays white). Keep the fewest v with Σ_{v<n} σ_v² ≥ e · Σ σ² (default e = 0.999).

Which Gram matters. The k-space samples of a centre-dense trajectory are dominated by k ≈ 0, where
y(0) = Σ_r S(r) ρ(r) is one vector. Its Gram is then nearly rank one. "99.9 % of the data energy" keeps far too few
coils (2 of 32 on the point test sim).

The density-neutral choice is the Gram of the Cartesian calibration region. By Parseval it equals Σ_r ρ_lr ρ_lrᴴ over
the low-resolution coil images:
- it uses the measured data, so no true maps are needed;
- the same calibration feeds ESPIRiT;
- it is the recon_p2np default (`--coil-gram calib`).

Measured with whitened true maps (map-energy criterion) of the 32-coil array:

| sim | 99 % | 99.5 % | 99.9 % | 99.95 % | 99.99 % | 12 coils keep |
|---|---|---|---|---|---|---|
| factory/ph0001 sim_point (160×160×64) | 9 | 11 | 16 | 21 | 28 | 99.70 % |
| factory/ph0001 b0test6 | — | — | 29 | — | — | ≈ 98 % |

So 12 virtual coils discard 0.3–2 % of the map energy, while 99.9 % needs 16–29.

The cost is linear in the coil count: per-coil gridding, FFTs and data. At 29 coils a recon is ≈ 2.4× the cost and
memory of 12. This is the reason to look at coil sampling (§4).

## 2. ESPIRiT

`hasty_recon::espirit`, CPU, with the conventions of the NUFFT (centred modes). The steps:
1. **Calibration windows.** Each k³ window of the calibration k-space is a row a_pᵀ of A.
2. **Signal subspace.** A = UΣVᴴ means a_p = Σ_j U_pj σ_j conj(v_j), so the window subspace is spanned by conj(v_j).
   This conjugate was the first-run bug: maps came out uncorrelated with the truth.
3. **Image-space operator.** W(r) = (1/k³) Σ_j ṽ_j(r) ṽ_j(r)ᴴ with ṽ_{j,c}(r) = Σ_d v_j[c,d] e^{+2πi d·r/FOV}.
   It is evaluated on a coarse grid with the same FOV.
4. **Maps.** The top eigenvector per voxel is the map. Its phase is aligned to a virtual reference coil (the dominant
   direction of all maps). It is interpolated to the recon grid and normalised to Σ|s|² = 1 where λ > eig_min
   (default 0.8).
5. **Eigensolver.** Only about 5 % of the n = nc·k³ kernels pass σ > 0.02 σ_max. Randomized subspace iteration with
   Rayleigh–Ritz (block 256, doubling until the threshold is reached) replaces the dense eigendecomposition:
   - production size (16 coils, 6³ kernel, 24³ calibration, 130 × 80 × 44 output): 311 s → 5.9 s;
   - same 179 kernels;
   - maps equal to < 1e-3.
6. **Non-Cartesian calibration.** `calib_from_noncart` takes the samples inside the box. It solves for the
   low-resolution coil images by CG with a small ridge and no density compensation, then takes a centred FFT.
   Synthetic test: 1 − |⟨ŝ, s⟩| = 3e-5 inside the object.

**Caveat for flow data.** Calibration uses encode 0 only. Mixing velocity encodes would make blood k-space
inconsistent, lower the eigenvalue in vessels, and could mask them. Lungs have low signal; if the mask cuts
pulmonary vessels, lower eig_min (`--smaps espirit:0.5`).

**Next.** Joint estimation of coil maps and image (the user's direction): poor maps leak image error into the coil
model. Candidates:
- nonlinear inversion (NLINV, Uecker 2008);
- alternating ESPIRiT-initialised refinement of the maps between P²nP blocks.

Both raise the cost; the evaluation in §5 will show how much error is due to the maps.

## 3. Measured fields

`hasty_recon::fieldmap::FieldMaps` reads a recon_b0 run: ω (Hz), R2*, fat fraction per breathing bin, and the
support. It samples them at each flow voxel's true scanner position:
- the B0 scan's prescription and gradient nonlinearity (GNL) come from its meta.json;
- recon_b0 does not model GNL, so its maps sit at apparent positions, apparent = GNL(x).

Holes are filled by onion peeling followed by smoothing. Lungs and arms need values even where the B0 fit failed.

The simulator applies e^{−R2* τ} during each readout, but the flow operator modelled only the phase. `--r2s`
(`truth` or `est:`) folds e^{−R2* (τ − τ_0)} into the segment time function through `build_segments_complex`, with
no new kernels and the same segment count. R2* ≪ |ω| range, so L is unchanged.

## 4. Stochastic coil sampling

**Estimator.** ∇̃f = Σ_{c < keep} E_cᴴ r_c + w Σ_{c ∈ D} E_cᴴ r_c, where D is `draw` coils of the remaining ones and
w = 1/P(c ∈ D):
- with random reshuffling, w = number of groups, and every coil enters once per epoch;
- the epoch mean equals the full gradient (test: 7e-6, float32).

How it fits the existing machinery:
- **Step size.** α must be stable for every subset, α = 1 / max_S λ_max(Ñ_S) (`P2np::step_norm_sampled`).
  - Drawing half of all energy-ordered virtual coils can include the dominant coil with weight 2: λ_max × 1.42 in
    the test, so α shrinks and the cost saving is eaten.
  - Keeping the leading coils and drawing from the tail gives 0.90, no inflation. Hence the `keep` parameter.
- **Circulant preconditioner M.** It is built from all coils and approximates E[Ñ]⁻¹ = N⁻¹; unchanged.
- **ZMSHR1.** The secant pair needs both gradients from the same subset; for a quadratic data term, m = Ñ_S s.
  - `hold` keeps a subset for ≥ 2 iterations.
  - On a new subset, `Objective::curvature_pair_ok` makes the step fall back to the metric step.
  - Subsets are a function of (seed, k), so resumed runs are exact.
- **Denoiser / LLR.** These act on the iterate after the gradient step, unchanged. The stochastic step adds gradient
  noise the prior partly absorbs. The fixed point is then a neighbourhood of the full one (online / stochastic PnP).
  `full_last` iterations with all coils end at the full fixed point.

**First result** (8 energy-ordered virtual coils, B0 segments, identity prior, 60 gradient-equivalents of coil
applications, error vs a 400-iteration solution):

| method | full gradients | best sampled schedule | worst sampled |
|---|---|---|---|
| PnP-ISTA (no preconditioner) | 1.77e-1 | 1.48e-1 (keep 3, draw 2) | 1.64e-1 |
| P²nP-ZMSHR1 | 5.51e-2 | 5.47e-2 (keep 0, draw 4) | 7.63e-2 |

Without curvature information, sampling buys 10–16 % at equal cost. With ZMSHR1 it buys nothing, because every
subset change discards the secant pair.

What could still make it pay:
1. The VRAM it frees. Residual and data buffers scale with the coils used, if the data of the unused coils stay in
   host memory: not implemented yet, data are gathered from device copies.
2. Problems where the learned prior, not the data term, limits convergence.
3. Higher coil counts with a flatter tail, e.g. 29 virtual coils at 99.9 %, where the tail is cheap to subsample.
4. A variance-reduced estimator (SVRG anchor) that keeps exact curvature pairs.

The real-data test with the denoiser is `recon_p2np --coil-sampling keep:draw:hold:last`.

## 5. Evaluation pipeline

`rustlib/tools/recon/eval_matrix.sh <flow sim> <recon_b0 run> <out> -- <recon args>` runs these configurations:

| configuration | field | coil maps | R2* |
|---|---|---|---|
| oracle | true | true | none |
| est_field | measured ω | true | measured |
| est_smaps | true | ESPIRiT | none |
| est_both | measured ω | ESPIRiT | measured |
| true_r2s | true | true | true |

Every run is resumable and finished configurations are skipped. `eval_table.py` collects the final velocity RMSE,
the change against the oracle, and the estimate diagnostics (field error vs truth, ESPIRiT agreement, coils kept).

It is waiting for the realistic scans (10-min flow, then the 2.5-min B0 scan) and their memory handling: data
streamed from disk with compression on load.

**Smoke test** (factory/ph0001 sim_point and b0test6, 3 iterations): every path runs.
- That B0 test sim gives a poor ω (RMSE 376 Hz, many fat/water swaps). It is old and was run here on a coarse grid
  with few iterations.
- The B0 fit quality itself is the recon_b0 / mbq_b0 work package.

## 6. Data path for full-length scans

The 10-minute factory_arms scan is 53 144 readouts × 3 990 ADC samples × 32 coils = 212 M samples, 54 GB raw. The
legacy loader holds the file twice while loading and then a whitened copy (> 150 GB), so the reconstruction now
streams (implementation: rustlib/PLAN.md, "K-space data path"):
- **Load.** Readouts are read on demand, then whitened, given noise, and coil-compressed on the way in (`kload`).
  The noise is added in the whitened domain, where the Ψ-correlated noise is white. It is counter-based, so any
  split, order or subset of the scan gets the same noise. The calibration region and the compressed data share
  it, as in a real scan.
- **Hold.** The data sit in a `KStore` in one of these representations:
  - dense RAM (cluster nodes);
  - lossless lz4 / zstd blocks in RAM (×1.2–1.3 on noisy floats);
  - a file on the local NVMe, reusable across runs;
  - optionally noise-relative quantisation in steps of q·σ. It adds q²/12 of the noise variance (q = 0.25:
    +0.5 %) and stores ≈ 4–6 bits per component. It is lossy but by a known, tiny, white amount; off by default.
- **Use.** The data stay resident when they fit; otherwise they are staged per set: double-buffered pinned upload,
  with the next set decoded while the GPU works. Stochastic coil sampling then moves only the sampled coils'
  blocks: the VRAM and bandwidth saving that §4 lacked.

Memory per sample of the 10-min scan:

| item | bytes / sample | total |
|---|---|---|
| raw coil data (32 coils) | 256 | 54 GB |
| data, 29 virtual coils | 232 | 49 GB |
| data, 29 virtual coils, zstd1 (lossless) | ≈ 186 | 39 GB |
| operator sample tables (SegPoints), pinned | 187 | 40 GB |
| sample tables packed (zstd1, ×6.2) | ≈ 30 | 6.5 GB |
| host sample records (coords, τ, concomitant coefficients) | 52 | 11 GB, released after the tables are built |

The tables are trajectory data and compress ≈ ×6, so they get the same treatment as the data: packed host blocks
decoded into a pinned staging buffer on use. The 10-minute scan then needs ≈ 46 GB of RAM (lossless data in RAM)
plus the 11 GB of records during setup. That fits next to nothing else on this workstation, and easily on a
cluster node. Quantised data or a disk store take it far below that.

## 7. Terminology: T2* vs T2

Every sequence here (the flow scan and the multi-echo B0 scan) is a spoiled gradient echo: there is no refocusing
pulse, so the transverse signal decays with T2*: 1/T2* = 1/T2 + 1/T2′, where T2′ is the dephasing from field
variation within the voxel.
- The tissue table (`hasty_phantom::tissue`, `tissue_variation.py`) stores T2* per tissue (literature values at
  3 T). The simulator applies e^{−τ/T2*} per source.
- T2 only bounds T2* (T2* ≤ T2 in the variation script); no signal equation uses it.
- The B0 scan fits R2* = 1/T2* (`maps_r2`, the "rate map") next to ω and the fat fraction. A spin-echo
  measurement would be needed for T2 itself.

Caveat: the simulator also resolves B0 explicitly on sources finer than a voxel. The intravoxel dephasing from that
field already appears in the summed signal. A literature T2* measured in mm voxels contains part of the same effect,
so it is counted twice in tissues with strong field gradients (lung and air interfaces). The physically consistent
source property would be R2 + R2′ (mesoscopic), not the voxel-level R2*. This is a small effect elsewhere; revisit
with the phantom work.
