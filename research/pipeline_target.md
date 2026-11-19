# Target pipeline: status and gaps (single source of truth)

The whole forward chain, its approximations, the gradient cost and the status of every factor are in research/mathematics_of_paper/forward_chain_section.tex. This file tracks the work.

Started 2026-11-18. Every component change and every agent brief is checked against this file. Update the matrix whenever a component's status changes.

## 1. Target

The final reconstruction is 4D flow of the thorax, aimed at accurate coronary flow:
- 1 mm full torso (640×384×384) or a 320³ / 384³ head grid;
- 10–12 min of data, gated by bellows and PPG;
- the velocity model (magnitude, phase, velocity) on a 2-D spline basis: 6 respiratory knots (linear) × 16 cardiac knots (cubic);
- the full physics: ratemap (off-resonance, decay and fat together), concomitant fields, GNL;
- priors on all parameters;
- 5–6 h or more on 8×5090, starting from a pyramid or an ML initial guess.

Dev runs use the same model and physics. They are smaller in space (heart box), in levels (pyramid) and in data (keep subsets), never in physics.

## 2. Error budget (replaces the misread "1e-4 everywhere")

- **NUFFT:** 1e-4 is the NUFFT's accuracy rule (kernel W, σ), and only the NUFFT's. Its error must stay well below the physics error.
- **Every other approximation** follows a budget at each pyramid level, relative to the dominant model error at that level:
  - time segments of the ratemap;
  - operator frames of the velocity model;
  - coil and sample compression;
  - physics simplifications.

  It starts loose (about the same size as that error) and tightens towards the NUFFT level in the last levels. It is reported as achieved, signal-weighted error.
- **Why not tighten early:** late, tight levels barely move the solution. Gradient steps per knot are small when each knot holds little data.

Measured reference: the 16 cubic cardiac knots' own error against the truth signal is 4.3e-2 (frames_caveat, ph0001). The table gives the operator-frame error of the velocity model (cubic frames):

| frames per cycle | signal error | velocity error / dynamic RMS | cost per encode pass (2 mm, 30 coils) |
|---|---|---|---|
| 12 | 9.3e-2 | 0.29 | 119 s |
| 16 | 4.4e-2 | 0.14 | 126 s |
| 24 | 1.8e-2 | 0.06 | 140 s |
| 48 | 1.4e-3 | 0.007 | 180 s |

So a frame pyramid of 12 → 16 → 24 follows the budget.

**Correction (2026-11-18): the costs above hold only at 2 mm, where the sample side dominates.**
- **Image side:** FFT plus pad / crop per (coil, segment, image), forward + adjoint (§22.3 of temporal_operator.md):
  - 2 mm: about 1 ms;
  - head 400³ fine grid: about 8 ms (scaled by cells);
  - 1 mm torso: about 23 ms.
- **Per image** (30 coils, L 10): 0.3 s / about 2.4 s / 6.9 s.
- **Velocity model:** it needs (respiratory rows × F) images. That is 9 rows today because of the segnufft table-key limit, 6 once the key is widened.
- **1 mm torso, image side per encode pass:**

| | images | image side |
|---|---|---|
| complex knots | 96 | 660 s |
| F 16 | 144 (96) | 990 s (660 s) |
| F 24 | 216 (144) | 1490 s (990 s) |
| F 48 | 432 (288) | 2980 s (1990 s) |

  The values in brackets are with the widened table key.
- **Sample side:** about 500 s, independent of F, because each sample touches 8 images.

So at 1 mm the frame count is a first-order cost. Options are in §5.

## 3. Component matrix

Status values: **done** = used in target runs; **built** = exists but is off by default or unverified; **gap** = missing.

| # | component | target | status | gap / next |
|---|---|---|---|---|
| 1 | Ratemap estimation (field + decay + fat maps, breathing-resolved) | from the multi-echo ratemap scan, real-data only, per bellows state | built (recon_b0), **not usable** | First run on ph0001 (2026-11-18): field RMSE 193 Hz, 37 % of the signal voxels off by more than 150 Hz (swaps / aliasing). An agent is diagnosing and fixing it, and adding breathing-resolved maps (research/ratemap_estimation.md). |
| 1b | Simulator: breathing-dependent field | field recomputed from χ per respiratory state | **missing** (the field is static per source, only advected) | An agent is adding it (research/resp_field_sim.md), with new dev datasets. |
| 2 | Ratemap per operator frame + segments | one ratemap per (b, c) frame, g = (1−φ)e^{−z_w τ} + φ Σ α_p e^{−z_f,p τ} × concomitant mean × readout-motion mean; SVD segments per frame | engine: `SegNufft::segments_frames` (no extra FFTs); recon_p2np `--ratemap truth-bc` builds per-knot segments from the simulator's TRUE per-state ratemap (oracle; estimation is shelved, user 2026-11-18) | GPU test seg_frames queued. Truth maps are being written. Velocity-model frames / respiratory groups: not yet. Readout-motion mean (lagged v): not yet. Per-frame coil maps: not yet. Per-frame segment count rule: static rule plus the achieved error printed for 2 knots. |
| 3 | Concomitant + GNL | degree-4 fit under GNL (order 3) | done (default) | — |
| 4 | Respiration | bellows-driven 6 linear knots | done (default) | — |
| 5 | Temporal basis + velocity model | 6×16 knots on m, ψ, v; operator frames from the budget | built (`--vmodel knots`, `--vop frames:F`) | The smoke run is broken (NaN objective, initial guess worse than zero, blow-up); it is being debugged. The frames default changes to 16, with a frame pyramid 12 → 16 → 24 to add. |
| 6 | Priors | LLR, divergence-free wavelets on v, db4 on m, static anchor, knot static lock, VENC continuation | built (on by default in the velocity model) | Untuned. |
| 7 | Coil compression | ≥ 99.9 % of the energy of the region of interest (prescription box Gram) | built (`--coil-gram box`) | Not run yet. Coil subsampling: SVRG recommended, not built. |
| 8 | Sample compression (`--rcompress`) | exact per-readout compression for box or support ROIs | built | Box ROIs 1.4–2.6×, full FOV about 1× (readouts already at Nyquist); masked support 1.3–1.7×. Real-data and driver tests are queued. |
| 9 | Joint coil × segment basis | fewer channels per pass | built (prototype) | 5–8× slower: the kernels need about 100 channels per pass; diagnostics queued. Low priority. |
| 10 | Pyramid schedule | space (4 → 2 → 1 mm), frames, segments and coils by level; box dynamic + outside static | partial (`--grid-scale`, `--init-from`, `--kcrop`) | No single driver runs the whole schedule. |
| 11 | NUFFT accuracy | error ≪ physics error | W 10, σ 1.25 | About 2e-4 on the normal operator for full-band images (rcompress check), so it needs a decision; reference runs are queued. |
| 12 | Multi-GPU | 8×5090 | plan (plan_cluster.md) | Later. |

## 4. Order of work

1. **Ratemap path end-to-end** (components 1–2):
   - Reconstruct `sim_b0_2.5min` with recon_b0 to get the estimated field, decay and fat maps.
   - Feed them into the knot modes and the velocity model. Fold the fat with the static ff map, and take the decay from the estimated maps.
   - Size the segments from the budget and build the basis on the GPU.
   - Report the achieved error per encode.
2. **Velocity model working** (5): fix the smoke failures, then add the frame pyramid.
3. **One integrated dev driver** (10):
   - heart box, levels 4 → 2 → 1 mm with frames, segments and coils scheduled per level;
   - all physics with the estimated ratemap, box coil Gram and rcompress box;
   - a pre-registered comparison against the complex-knot baseline.
4. **Full-FOV speed and scale:** joint-basis kernels, SVRG coils, the NUFFT accuracy decision and multi-GPU.

## 5. Velocity model: keeping the spline work in k-space

The knot-major engine already does the temporal spline in k-space:
- **Weights per sample:** each sample takes its own B-spline weights over the knot images. This also covers motion during the readout at the knot resolution.
- **FFTs:** one per knot image.

This is exact only for a model that is linear in the images, i.e. complex knots. In the velocity model the signal is m e^{i(ψ − k_e·v)}, which is nonlinear in time. The operator frames are a patch for that: complex images at F time points, interpolated in k-space.

Ways to cut that cost:
1. **Box-restricted frame corrections.**
   - Model x(t) = complex knot spline (full grid, 96 images) + d(t). The correction d is nonzero only where k_e·v changes fast between knots: lumen and myocardium inside the heart box.
   - d is evaluated at F frames on a box-sized NUFFT grid. A box-supported image needs only a box-sized FFT (200³ fine for a 160 mm box), about 23× cheaper per image than the torso grid.
   - The sample side gains one more gather per sample.
2. **Widen the segnufft table key** so the respiratory rows drop from 9 to 6: 1.5× on the image side.
3. **A frame pyramid by the budget** (12 → 16 → 24).
4. **To measure first:**
   - A least-squares complex-knot fit of the parametric model. F measured a cubic L2 projection about 1.2× better than interpolation, so this alone is likely not enough.
   - A splitting scheme: the data term on complex knots, with the velocity model as an image-space coupling.

## 7. Resume note (2026-11-18 evening, laptop closed)

- **Detached units that keep running:**
  - `rc-gpu-queue`: sample-compression checks; logs in research/experiments/sample_compress/logs/.
  - `sj-gpu`: joint-basis diagnostics, counts and gains; research/experiments/cl_compress/gpu/out/.
  - `seg_frames_test`: GPU test of the per-frame segments; log research/experiments/seg_frames/test.log.
- **Agents stopped with the session:**
  - The simulator breathing/cardiac field agent: see research/resp_field_sim.md "status / resume". If it did not write one, check its dev sim under data/hasty/phantom/factory_arms/ph0001/ and the truth_ratemap_bc output of sim_flow_10min/scan.
  - The full-chain accuracy agent: see §6 of this file, if written.
- **Stopped dev chain:** vknot_dev2_B2 is stopped on purpose. Its binary predates the velocity fixes (the B2a/B2b STOP files are in place).
- **Next on resume:**
  1. Read test.log.
  2. Once scan/truth_ratemap_bc/ exists, run a 4 mm box smoke with `--knots2d 6,1,16,3 --ratemap truth-bc`. Check the field correlation printout, the per-knot segment error and the build time.
  3. Continue the open items of row 2, then the frame pyramid and the temporal velocity prior.

## 6. Operator accuracy (full chain)

Goal: a full-chain f64 check of the per-frame-ratemap operator (forward_chain_section.tex, eq. op) in
`rustlib/crates/hasty-bench/tests/seg_physics_direct.rs`, on a 2-D knot engine (`SegNufft::new_temporal2d`, respiratory
linear × cardiac cubic, KR 3 × MC 5, order 3, `segments_frames` + `points2d_hi`). One table, every row a signal-weighted
relative data error (`rel3`).

**Status (2026-11-18): design done, no code written yet.** The test file is unchanged and compiles as before.

Design (ready to implement):
- Problem: `problem()` with `Spec { b0 2.89, oblique off-centre, gnl, field: false, conc, frames: MC, n_ro 24, adc_step 16 }`
  (own ratemap, so the sim field map is off). Breathing state per readout b_r = frac(0.382 r + 0.07); cardiac time from
  `seg_tables` (`tab.t`); frame weights (k, c) = resp hat × `bspline_weight(MC, 3, c, t)`, frame index k·MC + c. Own
  image x of KR·MC frames (fixed seed).
- Synthetic ratemap `rate_at(x_log_true, b, c)` → (ω, R^w, R^f, φ): static smooth ±150 Hz map; air-boundary dome
  (tanh, 40 Hz, 12 mm) moving 20 mm with b (breathing part, tens of Hz); heart Gaussian × 3 Hz cos 2πc (cardiac part);
  R^w 30 + 40 near the boundary, R^f 55; fat φ = subcutaneous band shifting ±1 voxel in y with b, epicardial band with
  radius 42 ± 8 mm over c; six-peak spectrum (`FatSpectrum::six_peak`). Phases since excitation, decay from τ0 (as
  recon_p2np's rm_knots branch).
- Own frame-aware direct sum `dsum(base(j,q), frame(j,q,f))` (rayon, forward + exact adjoint).
- Operator segments per frame: `build_segments_complex` of g_f(v,τ) = rate part × e^{−i conc̄} (`segment_phase`,
  with_b0 false), ntab = na; the shared basis = the same call over stacked (frame, voxel) rows.
- Rows: (1) segments only (exact ΔC), stencil only (exact g_f), segments + stencil (emulation, order 3), NUFFT + f32
  (L 1, order 0, no physics), implementation (GPU vs emulation), total, adjoint identity — both sets, the only
  asserted rows; (2) direct sum with vs without: static field, breathing part, cardiac part, per-frame ω vs knot-mean ω,
  whole ratemap vs knot mean, decay, fat, fat motion, concomitant, GNL; (3) per-frame model vs state-exact ratemap
  (bilinear in (b, c) from the knot maps; also analytic); (4) lumen cylinder (r 15 mm) at 0.2 / 0.5 / 1 m/s: truth
  e^{−i γ m1_j(τ)·v} (record p[3..6]) vs encode constant only, vs encode-mean m̄1e(τ) folded in, vs + first-order
  per-readout rest; whole data and lumen only; plus the emulated operator with the mean term in its segments;
  (5) L sweep 8…32, per-frame vs shared basis, L at 1e-3 / 1e-4.
- Check vs the simulator: per sample, `direct_sum_cpu` with sources (frame × voxel × {water, 6 fat peaks}, vt, r2s,
  amp scaled by e^{R τ0}) must equal the reference (≤ 1e-9).

Next: implement the above (new code only, after the existing tests), develop on the CPU emulation (ignored test),
then one GPU run under the GPU lock; set tolerances on the rows of (1) from the measured values; fill the table here.
Engine question to verify on the first GPU run: `segments_frames` with order 3 (`points2d_hi`) on the 2-D engine has
only been tested at order 2 (hasty-nufft/tests/seg_frames.rs).

### 7.1 Night of 2026-11-18 → 19 (all GPU/CPU jobs must end by 06:30)

- **Stopped on purpose:** `rf-respfield-datasets`, at the b0 step. Ratemap estimation is shelved, so the 2.5 h multi-echo scan with the new field is not needed tonight. To resume: `QFROM=0 rustlib/tools/phantom/resp_field_runs.sh b0` with GPU_LOCK set (see research/resp_field_sim.md).
- **Done:** sim_flow_10min/scan/truth_ratemap_bc, 120 states, written cleanly after a resume at phase 18 (`--q-from`).
- **Running:**
  - `sj-gpu`: joint-basis gain rows, resumable per row.
  - `rmbc_smoke`: the first `--ratemap truth-bc` run, 8 mm, 2 iterations; log research/experiments/ratemap_bc/smoke8mm.log.
- **Guard:** the `deadline_guard.timer` stops every remaining job unit at 06:15; all of them are resumable.
