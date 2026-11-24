# Preconditioner comparison (dev config): NOT RUN, blocked after the smoke test (2026-11-22)

Only the smoke test ran (run.sh smoke: --holdout 25, 2 pooled plain-CG iterations, then zmshr1 and zmshr1m, 2
P²nP iterations each, one process). Log: data/hasty/phantom/factory_arms/ph0001/precond_compare/smoke/log.txt.

## Blockers (each needs a code change; none made)
1. Held-out residual cannot be measured in this config. `--holdout-eval` is a separate run whose ESPIRiT maps are
   computed from the held-out readouts only (the holdout filter runs before the calibration; there is no option to
   load the solve run's maps, only `--cc-load` for the coil basis). With `--smaps espirit` the eval operator differs
   from the solve operator. The solve loop itself prints no held-out residual, and there are no per-iteration image
   snapshots, so the curve at it 10/20/40/60 is not available either.
   Needed: either the held-out residual computed in the solve loop (every N iterations), or `--smaps <file>` loading
   plus snapshots at given iterations.
2. Arm 3 (`--pp-metric p2`, Jacobi metric) is rejected by the code: recon_p2np.rs line ~6552 asserts the metric
   base is a circulant (p0, p1, p1n, p12, p12n, pe<ε>_<p>); p2 is only allowed for `--pp-pool`. Arm dropped.
3. OOM at the 24 G cap: after zmshr1 finished its 2 iterations, the zmshr1m run in the same process was killed by the
   OOM killer at its first iteration (exit 137). Not investigated.

## What the smoke test established
- ADC part out: samples 0..1991 of 3990 per readout; with --kcrop: 22 905 944 of 203 075 040 samples kept (11.3 %),
  ADC window 0..478 (1.600..2.556 ms).
- Sampling: 560 sets (4 resp bins × 20 cardiac frames × 7 encodes); with --holdout 25, 50 896 readouts solve /
  2 248 held out; 75..112 readouts and 33 756..50 334 samples per set (149 325..222 992 before the crop).
- Coils: `--vcoils auto --coil-energy 0.999` keeps 31 of 32 virtual coils (99.952 %).
- ESPIRiT: 638 kernels, 174 s; vs true maps: mean 1 − |⟨ŝ, s⟩| = 0.326 over 552 950 voxels, 135 943 masked.
- Segments: --nseg auto → L = 5 at the segment-basis step, SegNufft reports L = 6.
- Field: the pooled_diag scripts pass no field flag (default --field truth, one static true field); used the same.
- Setup: 708 s the first time (k-space store build 129 s, ESPIRiT 174 s, autotune 454 s of NVRTC compiles; the
  store and autotune are now cached, so the next setup is roughly 4 min).
- Pooled plain CG: ~14 s per iteration (2 iterations: ‖r‖/‖r₀‖ 0.27).
- zmshr1 P²nP: 39–41 s per iteration = gradient 12.2 s + prior 24–26 s (LLR + db4 + DFW 7.4 s + lock) + report 2.5 s.
  The prior, not the data term, dominates the iteration.
- pc score references (974 voxels): zero image RMSE 0.425 m/s (median 0.131), time-averaged truth RMSE 0.378
  (median 0.137); after the 2-iteration warm start + 2 iterations: RMSE 0.422.
- Campaign estimate if unblocked: setup 4 min + 60 pooled iterations 14 min + 60 iterations × 40 s = 40 min per arm.

## Overnight results (2026-11-22/23), computed by analyze.py on the run logs

| arm | s/it (gradient + prior + step + report) | held-out rel. residual it 0 / 10 / 20 / 40 / 60 | min (it) | within 1 % of min: it, wall s | pc RMSE / median m/s | peak RSS GiB |
|---|---|---|---|---|---|---|
| zmshr1m | 50.5 (14.1 + 31.9 + 1.8 + 2.7) | 0.07887 / 0.06497 / 0.06433 / 0.06538 / 0.06886 | 0.06433 (20) | 10, 467 | 0.410 / 0.124 (it 60) | 20.7 |
| zmshr1 | 48.6 (17.3 + 28.3 + 0.2 + 2.8) | 0.07887 / 0.06772 / 0.06654 / 0.06411 / 0.06355 | 0.06355 (60) | 40, 1884 | 0.277 / 0.089 (it 60) | 21.0 |
| p2 | 50.7 (17.3 + 28.9 + 1.8 + 2.8) | 0.07987 / 0.06834 / 0.06723 / 0.06652 / 0.06396 | 0.06396 (60) | 60, 3049 | 0.279 / 0.093 (it 60) | 21.6 |
| pe | 56.5 (20.0 + 31.8 + 1.8 + 3.0) | 0.07917 / 0.06686 / 0.06571 / 0.06532 / 0.06369 | 0.06369 (60) | 55, 3027 | 0.267 / 0.090 (it 60) | 20.2 |
| zmshr1_120 | 48.2 (17.3 + 28.6 + 0.2 + 2.1) | 0.07912 / 0.06770 / 0.06653 / 0.06555 / 0.06956 | 0.06147 (120) | 85, 4033 | 0.242 / 0.086 (it 120) | 21.3 |

- zmshr1_120: held-out spikes at it 35 (0.071), 60 (0.070), 90 (1.07) and 115 (0.84), recovered by the next report; at it 110 f = 1.4e16. The iterate jumps and comes back (no step safeguard in ZMSHR1).
- Static map (it 85–110): static over cardiac ~18 %, over breathing ~15 %, 'motion not detectable' ~63 % of voxels, so the lock acts on a minority of voxels.

## With the air prior (2026-11-23, chain `admm/chain_air.sh`)

Same data and config as above (out-half ADC, all readouts, 4 resp × 20 cardiac × 7 encodes, 128 × 72 × 60, 31 coils,
LLR + db4 mix 0.3 + DFW + static lock), now with `--air auto` (air map from the iterate, zeroed in every frame before
and after the prior). Held-out = relative residual on the 2 248 held-out readouts; pc = velocity RMSE / median in the
974 lumen voxels (scoring only). Wall time counts the iterations only (setup excluded).

| run | solver | coil maps | iterations, wall s | held-out (final) | pc RMSE / median m/s |
|---|---|---|---|---|---|
| g_esp (no air, earlier) | P²nP, no metric, step guard | ESPIRiT | 60 | 0.0613 | 0.258 / 0.091 |
| g_esp_air | P²nP, no metric, step guard | ESPIRiT | 60, 2 761 | 0.0593 | 0.315 / 0.089 |
| g_truth (no air, earlier) | P²nP, no metric, step guard | true | 60 | 0.0598 | 0.346 / 0.101 |
| g_truth_air | P²nP, no metric, step guard | true | 60, 3 315 | 0.0571 | 0.283 / 0.085 (still falling) |
| A_none_air | ADMM, ρ = ρ₁, 3 inner PCG steps per set, no inner metric | ESPIRiT | 15 of 30, 1 479 (killed) | z 0.0605, x 0.0561 | z 0.289 / –, x 0.291 / 0.095 |
| B_clip_air | ADMM, as A with the clipped-density inner metric | ESPIRiT | 3 of 30, 295 (killed) | z 0.0594, x 0.0586 | z 0.335, x 0.337 / 0.103 |

- Air helps the true-map run a lot (pc 0.346 → 0.283) and lowers held-out in both runs. With ESPIRiT maps the held-out
  improves (0.0613 → 0.0593) but pc is worse (0.258 → 0.315). Cause not known. The no-air run's pc fell
  0.320 → 0.285 between it 20 and 25; its step guard fired only from it 48 on, so that drop is not a guard event.
- True maps now beat ESPIRiT on both scores (with air), as expected. The earlier "true maps worse" result came from
  the run without air.
- ADMM without inner metric, at 1 479 s (0.54 × the P²nP wall time), is already below the P²nP final on pc
  (0.289 vs 0.315) and on held-out of x (0.0561 vs 0.0593). That meets the pre-registered rule (reference quality in
  ≤ 0.6 × wall time), on one run cut at outer 15.
- Caveats:
  - The inner solve is not solving: ‖r_K‖/‖r₀‖ ≈ 0.8–1.4 per outer iteration after the first (0.39 at outer 0). The
    x-step is far from exact, so the estimator-invariance argument for ADMM does not apply yet.
  - The primal gap ‖x − z‖/‖x‖ grows 1.5 % → 11 % over 15 outer iterations, and held-out of z rose at outer 15
    (0.0590 → 0.0605). Not converged.
  - Clip vs none: only 3 outer iterations of clip; equal within noise there.
- Both ADMM arms were OOM-killed by the kernel at the 24 GiB cgroup cap (anon RSS 23.7 GiB; journal 14:07 and 14:32;
  the script logged exit 143). The P²nP arms already peak at 22.8–23.3 GiB, so 24 GiB had no margin. OOM kills with
  SIGKILL, so no checkpoint (SIGINT / SIGTERM / STOP do checkpoint). Rerun `admm/chain_air2.sh`: cap 32 GiB, host RSS
  and peak logged every outer iteration.

### ADMM rerun to 30 outer iterations (2026-11-23, `admm/chain_air2.sh`, cap 32 GiB)

Same arguments as above (ESPIRiT maps, air, ρ = ρ₁ fixed, 3 inner PCG steps per set). x = data-step image, z = denoised
image D(x + u). Wall time = iteration loop only. Host memory: 16.0 GiB between iterations (0.1 GiB fragmentation),
peak 23.7 / 23.8 GiB (`/usr/bin/time`), so the 24 GiB cap could not hold it.

| arm | outer it (s) | held-out z | held-out x | pc z RMSE / median | pc z, hierarchical decoder | pc x |
|---|---|---|---|---|---|---|
| none | 12 (1 328) | 0.0593 | 0.0570 | 0.308 / 0.095 | 0.286 | 0.309 |
| none | 30 (3 445) | 0.0635 (min 0.0592 at 9) | 0.0565 (min 0.0559 at 18) | 0.264 / 0.099 | 0.254 | 0.259 |
| clip | 12 (1 294) | 0.0588 | 0.0556 | 0.308 / 0.099 | 0.287 | 0.311 |
| clip | 30 (3 398) | 0.0616 (min 0.0588 at 12) | 0.0552 | 0.273 / 0.101 | 0.268 | 0.273 |
| P²nP reference (g_esp_air) | 60 (2 761) | 0.0593 | – | 0.315 / 0.089 | 0.302 | – |

- Pre-registered rule (reference quality in ≤ 0.6 × the P²nP wall time): both arms reach held-out 0.0593 and pc ≤ 0.315
  at outer 12, 1 294–1 328 s = 0.47–0.48 ×. Pass.
- At the P²nP wall time (2 761 s, outer ≈ 24): none pc z 0.270, clip 0.283, P²nP 0.315.
- Not converged: the primal gap ‖x − z‖ / ‖x‖ grows to 11.5 % (none) and 21.8 % (clip) by outer 30. Held-out of z
  rises after outer 9–12 while held-out of x stays flat and pc keeps improving to outer ≈ 27.
- Clip vs none: clip solves the inner problem better (‖r_K‖/‖r₀‖ 0.55–0.85 vs 0.75–1.4) and fits the data better
  (held-out x 0.0552 vs 0.0565, f(x) 1.58e13 vs 1.85e13), but its pc is worse (0.273 vs 0.264) and its x–z gap is
  twice as large. The two arms do not reach the same image: with 3 inner steps the x-step is inexact, so the metric
  still changes the result.

Direct image comparison of the two K = 3 final denoised images z (CPU, `recon_*.hv`, 146 294 signal voxels = time-mean
magnitude of the reference encode > 10 % of its 99th percentile):
- none vs clip: ‖z_none − z_clip‖ / ‖z_none‖ = 0.131 in the signal voxels (0.11–0.14 per encode).
- Split into the time-mean image and the dynamic part (each frame minus the time mean): the time means agree to 1.9 %,
  the dynamic parts differ by 80 % of the dynamic part's norm.
- P²nP (g_esp_air) vs ADMM: 0.157 (none), 0.162 (clip).
- So the data fix the time-mean image, while the frame-to-frame variation (the motion and flow content) depends on the
  solver path. That fits the per-set sampling: a set is dense only for |k| ≲ 0.15 of the ball radius.

## Where the large velocity errors come from (ADMM none, K = 3, outer 30; CPU analysis 2026-11-23)

Encoding (scan meta.json): `coprime7`, per axis venc 1.0 and 1.5 m/s, unaliased range ±3.0 m/s. The recon decoder is
the same scheme: a first stage on the venc-1.5 encodes (±1.5 m/s), then a maximum-likelihood alias search over ±3.0 m/s
(margin 1). The pc score's "hierarchical" variant is the first stage without the search.

- Squared error: 0.86 % of voxel-frames with |Δv| > 1 m/s carry 60 % of it; 0.61 % with |Δv| > 1.5 m/s carry 54 %.
  The median error is 0.099 m/s. Without the > 0.5 m/s voxel-frames the RMSE is 0.146.
- Truth (decoded from the ideal PC signal pc_c, native grid, coverage ≥ 0.99): speeds up to 4.7 m/s. 1.3 % of
  voxel-frames exceed 1.5 m/s on some axis, 0.10 % exceed 2.5 m/s. The p99 speed is 2.0–2.8 m/s in frames 2–6. So
  the alias search is needed, and the truth stays inside ±3.
- At the outliers (|Δv| > 1.5):
  - the per-axis phase cost Σ(1 − cos) at the ML solution is 0.22, against 0.010 for all other voxel-frames (noise
    only at SNR 20 ≈ 0.015). The two venc pairs disagree: the recon phases fit no single velocity.
  - The first stage alone also fails there (median error 2.1 m/s), so this is not a wrong choice by the alias search.
  - 7.4 % of voxel-frames in cardiac frame 2 and 2.5 % in frame 3 (early systole); ≤ 0.5 % in every other frame.
  - The recon reference magnitude is 0.55 × the voxel's own median over frames; the truth's is 0.83 × at the same
    voxel-frames. The recon has lost signal there that the truth does not lose.
- Reading: the coprime decoding works wherever the recon phases are consistent (99.4 % of voxel-frames, noise-level
  cost). The outliers are recon errors in the dynamic part at the systolic upstroke: the signal drops and the encodes
  become mutually inconsistent. Untested hypothesis for the mechanism: temporal mixing of frames with very different
  velocities (a sum of phasors fits no single velocity in both venc pairs).

## ADMM with K = 10 inner steps (2026-11-23, `admm/chain_k10.sh`, no inner metric; the clip arm was skipped for the controls)

Same as `A_none_air` except `--admm-inner 10`, 21 outer iterations, 248 s each (inner 221 s).

| outer it (s) | held-out z | held-out x | pc z RMSE / median | pc x |
|---|---|---|---|---|
| 3 (704) | 0.0586 | 0.0572 | 0.287 / 0.096 | 0.290 |
| 6 (1 230) | 0.0663 | 0.0611 | 0.257 / 0.097 | 0.262 |
| 12 (2 284) | 0.0697 | 0.0687 | 0.246 / 0.096 | 0.251 |
| 21 (3 979) | 0.0810 | 0.0753 | 0.239 / 0.098 | 0.246 |

- Inner solve: ‖r_K‖/‖r₀‖ 0.11 at outer 0, then 0.44–0.47 every outer iteration (K = 3: 0.75–1.4).
- pc improves faster and further than K = 3 (0.239 at outer 21 vs 0.264 at outer 30), the best pc of all runs.
- The held-out residual rises steadily for both x and z (x 0.0572 → 0.0753), while K = 3 kept x at 0.055–0.057. The
  x–z gap reaches 22 %.
- Reading: with ρ = ρ₁ the data step of one set is barely coupled to z. Solved more exactly, x fits that set's own
  ~95 readouts (overfitting the sparse per-set data), and the held-out readouts get worse. The lumen velocities still
  improve. So the result depends on K: ADMM is not converged at this ρ, and K is acting as a regulariser.
- Next for ADMM: a larger or increasing ρ (x pulled to z), judged by held-out of x.

## Queued 2026-11-23 evening (`chain_ctrl.sh`, then `chain_rm.sh`)

- `c_base_llr_ga05`: practice baseline (LLR only, 20 cardiac frames from the quietest 50 % of breathing, air zeroing,
  same operator and solver as g_esp_air).
- `c_crime_air`: inverse crime in the g_esp_air config (data E x★ + measured-σ noise, x★ = PD + truth blood signal,
  no respiratory motion in x★). The first try failed at startup (`--synthetic` needs staged data: now `--plan host`).
- `per_set_segs` GPU unit test: `setop::PerSetSegs` (one host segment table per operator frame, uploaded per set)
  must equal the operator with all tables resident, to < 1e-6 (forward, adjoint, residual, group calls).
- `g_esp_air_rm` (only if the test passes): the g_esp_air reference with the decided model, the TRUE ratemap per
  frame (`--ratemap truth-bc`, now in frame mode: field, water / fat decay and fat fraction of each frame's
  breathing × cardiac state; the segment count rule on every frame's model). This replaces g_esp_air as the
  reference; it is not a per-frame vs static A/B.

## Controls done (2026-11-23 evening)

Scores at the last iteration (60) unless stated; pc score = 974 blood voxels, 20 truth frames, ML coprime decode
(hierarchical in brackets). All three runs: measured or synthetic data on the 2 mm dev scan cropped to 128 × 72 × 60,
31 virtual coils, ESPIRiT maps, full physics with ONE static field, out half, air auto, zmshr1, 60 iterations.

| run | data | frames | readouts / set | prior | pc RMSE m/s | median | p90 | frame 2 (early systole) | held-out rel | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| g_esp_air | measured | 4 resp × 20 card | 75..112 | LLR + wavelet 0.3 + DFW + static lock | 0.315 (0.302) | 0.089 | 0.353 | 0.89..1.01 | 0.0593 | 65 min |
| c_base_llr_ga05 | measured, 50 % quietest breathing | 1 × 20 | 162..202 | LLR only | 0.210 (0.229); best 0.1995 at it 40 | 0.074 (0.064 at 40) | 0.222 | 0.56 | 0.0476 (min at it 40) | 16 min |
| c_crime_air | E x★ + n, sim σ | 4 × 20 | 78..117 | as g_esp_air | 0.421 (0.283) | 0.072 | 0.298 | 1.30..1.40 | n/a | 63 min |

- The practice recipe beats the full method on every score, 4× faster. Held-out rel is not comparable across the
  binnings (different held-out sets).
- Inverse crime: with the exact operator the full method still fails in early systole (1.3–1.4 m/s in frames 2–3)
  and the ML RMSE grows from 0.37 to 0.42 over the iterations (hierarchical falls to 0.28). Caveat: x★'s blood is the
  truth box means (median |blood| / median PD 0.234), which may be darker than the measured blood.
- Data term: crime floor ½E‖n‖² = 5.92e11; crime f 1.73e12 (2.9× floor). Measured g_esp_air f 2.71e13 = 46× the same
  floor (same σ, similar sample count) → the operator mismatch dominates the measured residual. Suspect: the static
  field (per-state mismatch 37–77 Hz rms, p99 152–359 Hz).
- Two separate problems: (1) operator mismatch (tested by g_esp_air_rm, true ratemap per frame); (2) the estimator
  (4 bins × ~95 readouts per set + the full prior) — tested by the factorial below.

per_set_segs GPU test: first run failed on a too-tight bar (1e-6). Re-run with a control engine (second engine, every
table resident): PerSetSegs 3.8e-6 worst vs control floor 3.4e-6 (fp32 spreading atomics); wrong table 0.64. Passed
(per_set_segs_test2.log).

## Queued (pre-registered 2026-11-23 ~22:45)

1. g_esp_air_rm: g_esp_air with `--ratemap truth-bc` (frame mode). Question: how much of the 46× residual and the pc
   error is the static field? Decision: if f drops toward the floor and pc improves clearly, every later run uses the
   per-frame ratemap; else look for the next mismatch (coil maps, GNL, concomitant fit).
2. 2×2 factorial on the static field, binning {4 bins, 1 gated bin 50 %} × prior {LLR only, full}:
   f_ga05_full (gated + full) and f_r4_llr (4 bins + LLR), with g_esp_air and c_base_llr_ga05 as the other two cells.
   Question: is the full prior or the binning what loses against the practice recipe? Decision: drop or retune the
   component that loses (lock / DFW / wavelet), and set the respiratory binning from readouts per set.

## Results 2026-11-24 ~00:40

**g_esp_air_rm** (true ratemap per frame, frame mode) vs g_esp_air (static field), it 60:
- data term f 2.71e13 → 2.16e13 (−20 %; still 36× the crime noise floor); held-out rel 0.0593 → 0.0545 (warm start
  0.0753 → 0.0578).
- pc unchanged: RMSE 0.315 → 0.327, median 0.089 → 0.087, p90 0.353 → 0.348, frame 2 0.94 → 1.07.
- Per-frame field vs the static field (fit voxels, readout-weighted state mean): rms 22–53 Hz. L = 5 (1.15 cycles).
- **Invalid as the answer**: frame-mode bug. The per-frame tables were built from the all-encode mean concomitant
  coefficients w̄, the per-sample tables still took deviations from the per-encode means w̄_e → the per-encode part
  w̄_e − w̄ (max |Δ| 9.8 of 99 rad/m²) was missing from the model, and it differs per encode (velocity bias). The
  assumption "the encodes' means agree" was wrong. Fixed (snap_rm2: per-sample tables relative to w̄ in frame mode;
  the run prints the per-encode phase now carried by the per-sample fit). Rerun g_esp_air_rm2 queued.

**f_ga05_full** (1 gated bin 50 % + full prior), the factorial cell, it 60: pc RMSE 0.314, median 0.094, p90 0.357,
held-out 0.0496. Against c_base_llr_ga05 (same data, same held-out set, LLR only): 0.210 / 0.074 / 0.222, held-out
0.0480. Against g_esp_air (4 bins + full prior): 0.315 / 0.089 / 0.353. → **the full prior (wavelet mix + DFW + static
lock), not the binning, is what loses** — on pc and on held-out k-space. f_r4_llr (4 bins + LLR) completes the 2×2.

## 2×2 factorial complete (2026-11-24 01:00), static field, it 60 (pc RMSE / median / p90 m/s; held-out rel)

| | LLR only | full prior (LLR + wav 0.3 + DFW + lock group μ 20) |
|---|---|---|
| 4 resp bins | f_r4_llr 0.239 / 0.088 / 0.238; 0.0519 | g_esp_air 0.315 / 0.089 / 0.353; 0.0593 |
| 1 gated bin 50 % | c_base_llr_ga05 0.210 / 0.074 / 0.222; 0.0480 | f_ga05_full 0.314 / 0.094 / 0.357; 0.0496 |

- The full prior loses in both binnings, on pc and on held-out (held-out comparable within a row only).
- The binning comparison is confounded: the pc truth is ONE truth at the scan-mean breathing state 0.493
  (tpyr_score/pc), scored against every bin (audit 2026-11-24; true states 0.061/0.308/0.655/0.949, gated 0.184,
  up to ~4.7 mm heart displacement). Within-row comparisons are like-for-like.

## Audits 2026-11-24 (independent reviewers, read-only)

- Frame-mode truth-bc: (1) plain `--nseg auto` in frame mode sized L from the static field without fat (the
  per-frame rule only ran under auto:noise/err), seg_err printed nothing → g_esp_air_rm used L = 5 with fat in the
  tables, unverified; (2) the w̄ fix moved the per-encode concomitant part into the approximate per-sample fit;
  (3) frame mode changed field + fat + decay at once vs the static path; (5) checkpoints carry no config fingerprint
  (reuse risk with the same --out). Fixed: frame mode defaults to auto:noise:0.2 + svdn + energy (as knot mode), seg_err
  on, one table per (frame, encode) with the encode's own w̄_e (exact, as the static path), startup self-check
  (degenerate frame table must reproduce the static table on two real sets, abort otherwise). Re-review pending.
- Prior chain: no coding error found. Config error: `--static-lock 20 --lock-mode group` (20 is the quadratic-mode
  value; group documented at 1.5) → over-locking of partly weighted voxels. Plausible losses: per-frame db4 threshold
  at the aliasing level (detail pulled down), DFW writing decoder alias errors back into the images (margin rule),
  ZMSHR1 secant step including the denoiser's displacement. Gap opens before the lock starts (it 4–5).
- Scoring: one truth at the mean breathing state for every bin (above); phase metric without the GNL Jacobian
  (constant floor); held-out code sound.
- Config drift: non-knot runs default to `--seg-basis fixed --seg-weight count` and the plain static L rule, not the
  svdn basis decided for production.

## Queued: one component at a time (pre-registered 2026-11-24 01:05)

Gated cell, LLR k1 base + one of: wavelet mix 0.3 (k_ga05_wav), DFW sure (k_ga05_dfw), lock group μ 1.5
(k_ga05_lock1p5), lock group μ 20 (k_ga05_lock20). Question: which component causes the 0.21 → 0.31 loss? Decision:
drop components that lose on held-out AND pc; keep only ones that win on held-out.

## Queued after chain_comp (pre-registered 2026-11-24 ~01:30): the field question, one factor

Re-review of the frame-mode fixes: all five correct; added a host-memory check before the 560-table build (~27–30 GB
at L 11–12; dev-scale oracle only, 2 mm would need ~220 GB) and a non-vacuous self-check (non-empty set, the table index
f·ne + e rebuilt and compared).
1. s_esp_air_svdn: g_esp_air (static field, full prior) with `--nseg auto:noise --seg-basis svdn --seg-weight energy`.
2. g_esp_air_rm3: the same with `--ratemap truth-bc` (frame mode, fixed).
Question: does the true per-frame ratemap (field + water/fat decay + fat fraction) lower the data term toward the noise
floor (5.9e11; static 2.71e13) and improve held-out and pc? The pair differs only in the ratemap. Decision: if yes,
every later run uses it; if no, look for the next mismatch. Both are scored against the one mean-state truth (fine
within the pair); recon arrays kept for rescoring with per-bin truths (--score-pc-bins, built, truths not yet made).

## Paused 2026-11-24 (user: pause + sleep)

- k_ga05_wav stopped by STOP at ~it 7 (checkpoint in its dir; no DONE). Resume: `bash research/experiments/precond_compare/chain_comp.sh`
  (run_ctrl skips DONE runs; k_ga05_wav resumes from its checkpoint, same --out, same config).
- Not started: k_ga05_dfw, k_ga05_lock1p5, k_ga05_lock20 (chain_comp.sh), then s_esp_air_svdn and g_esp_air_rm3
  (chain_rm3.sh, binary snap_rm3).

## Gated-cell campaign (pre-registered 2026-11-24, user back; replaces chain_comp + chain_rm3)

User: only runs that really help understanding; cluster / Vast runs soon, so no more wasted runs.
Cheaper design: everything on the gated cell (1 bin, 50 % quietest breathing, 20 cardiac frames, 140 sets of 162–202
readouts; ~16 min per run instead of 65), which showed the same prior loss as 4 bins. Production segment settings
(auto:noise, svdn, energy). Scored against the truth at breathing state 0.181 (gated true state 0.184) with
`--score-pc-bins`; the old mean-state truth printed as an oracle (size of the old scoring bias). Binary snap_gc.
New guards in the binary: config fingerprint for resumes; leftover/repeated arguments abort (found: chain_rm3.sh had
`--nseg auto` twice-defined → s_esp_air_svdn would silently have kept plain `auto`).

| run | changed factor vs gc_llr | question | decision |
|---|---|---|---|
| gc_llr | — (LLR k1 only) | reference | — |
| gc_llr_rm | `--ratemap truth-bc` (true field + water/fat decay + fat fraction per frame) | how much of the residual (≈30× noise) and the pc error is the static ratemap? | f / held-out toward the noise floor and pc better → per-frame ratemap in every later run; else find the next mismatch |
| gc_wav | + db4 mix 0.3 | does the wavelet mix lose? | drop / retune if it loses on held-out and pc |
| gc_dfw | + DFW sure | does DFW lose? | same |
| gc_lock | + static lock group μ 1.5 (documented value) | does the lock lose at the right μ? | same |

Revision after the independent review (2026-11-24): scoring code correct; fixed the fingerprint (shared warm starts
not checked, --iters/--every/--ckpt-every may change on a resume; 5 dry cases on CPU behave), --synth-x pc refuses
--score-pc-bins, the bin truth directory must carry pc_resp_state.txt within 0.02 of its bin's true state. The prior
components now run on the TRUE-ratemap operator (the decided dev model) against gc_llr_rm, and only if gc_llr_rm
finishes: gc_rm_wav, gc_rm_dfw, gc_rm_lock (chain_gc.sh).

### gc_llr (2026-11-24): reference of the gated-cell campaign
svdn + energy + auto:noise → L = 4 (static ratemap), per-state truth: pc RMSE 0.262 at it 55 (0.251 at 35), median
0.079, p90 0.253, frame 2 0.78; held-out 0.0480 (min at it 25–35); f 9.52e12 = 33.5× the noise floor 2.84e11.
c_base_llr_ga05 (fixed L = 5, count weights, mean-state truth) on the same data: 0.210 / 0.074 / 0.222, frame 2 0.56,
held-out 0.0476, f 9.24e12. Oracle: mean-state vs per-state truth RMSE 0.055 (median 0.009) → most of the gap is
probably the segment operator. Queued gc_llr_fixed5 (chain_gc2.sh): gc_llr with fixed L = 5 count. Question: does the
production rule under-segment? Decision: if fixed5 wins on held-out and pc, tighten the auto:noise rule (or its
margin) before any cluster run; gc_llr_fixed5 vs c_base_llr_ga05 isolates the scoring truth.
gc_llr_rm setup: frame-mode self-check passed (degenerate 5.0e-6, true table vs static 0.45); per-frame field vs
static rms 22–25 Hz; per-frame segment model error 5.6e-4 at L = 4 (static 9.5e-5), target 1.08e-3.

### gc_llr_rm vs gc_llr (2026-11-24): the true ratemap per frame, one factor (LLR, gated, it 55)
f 9.52e12 → 7.84e12 (33.5× → 27.6× the noise floor); held-out 0.0482 → 0.0447 (monotone); pc RMSE 0.262 → 0.241,
median 0.079 → 0.075, p90 0.253 → 0.207, frame 2 0.78 → 0.64. **The true ratemap helps on every score, but most of the
mismatch remains** (28× noise). Queued gc_rm_tmaps (chain_gc3.sh): + true coil maps. Question: how much of the rest is
the ESPIRiT maps (31k zero-map voxels, mean 1 − |⟨ŝ, s⟩| 0.109)? Decision: if f drops a lot → coil-map estimation
(joint maps / --smaps-fit) is the next work item before the cluster; if not → the rest is likely intra-bin dynamics
(motion and flow within a frame, which no binned model fits) or physics the operator lacks; then measure that floor.

### gc_llr_fixed5 vs gc_llr (2026-11-24): segment options only (LLR, static ratemap, gated)
fixed L 5 count vs svdn energy auto:noise L 4, at it 35 / 55: pc RMSE 0.212 / 0.221 vs 0.251 / 0.262, median 0.064 vs
0.074, p90 0.210 vs 0.246, frame 2 0.62 vs 0.78; held-out 0.04778 vs 0.04803 (−0.5 %); f −1 %. The segment error
is a systematic, per-encode phase in blood: it biases velocity while hardly raising the data residual → the
auto:noise rule (data error ≤ 0.2 noise rms) is the wrong yardstick for velocity; the segment budget needs a
velocity-phase criterion before cluster runs.
Scoring truth alone (gc_llr_fixed5 vs c_base_llr_ga05, same operator and flags): 0.212 vs 0.204, median 0.064 vs 0.066
→ within run-to-run spread. Note: identical flags gave f 9.37e12 vs 9.30e12 (new autotune → other fp order → LLR rank
decisions diverge): comparison noise ≈ ±0.01 in pc RMSE.
Queued gc_llr_svdn5 (chain_gc4.sh): svdn energy at L = 5. Question: rule's L or basis / weights? Decision: if svdn L 5
≈ fixed L 5 → keep svdn, replace the rule's yardstick; else investigate the svdn basis for blood phase.

### gc_rm_wav vs gc_llr_rm (2026-11-24): + db4 mix 0.3 on the true-ratemap operator
pc RMSE 0.241 → 0.323 (it 55), median 0.075 → 0.086, p90 0.207 → 0.296; held-out 0.0447 → 0.0461. Loses on both →
drop the wavelet mix (pre-registered decision).

### Encode trains and frames (2026-11-24, user question)
Readout used at dev scale: TE 1.6 ms, out half, cropped to the 4 mm grid's band → τ 1.600–2.556 ms (0.96 ms); at 2 mm
the out half is ~3.9 ms. Model errors are small, as expected for such a readout: segment model relative error 9.5e-5
(static, L 4) / 5.6e-4 (per frame, L 4), concomitant degree-4 fit leaves ≤ 0.035 rad. My explanation of the L 4 / L 5
gap as a per-encode phase error was not supported by these numbers (withdrawn); the gap is unexplained until
gc_llr_svdn5 and the repeat gc_llr_rep.
Encodes: coprime7 (encode 0 reference, encodes 1–6 venc 1.0 / 1.5 m/s per axis), played as consecutive TRs of one
rotation (readout i = rotation i / 7, encode i % 7, order 0..6; same trajectory for all 7). Binning is per readout
(cardiac phase and bellows at its own time). RR 853 ms → frame 42.6 ms; train 7 × 11.29 = 79.0 ms → 0 % of trains in
one frame, 41 % in two, 59 % in three: every frame's encodes come from different rotations. recon_diag_2026-11-13 §5–8
measured with a plain per-frame recon: --bin-group corr 0.36 → 0.56; its cost is intra-train acceleration cross-talk
(0.087 m/s rms with order 0..6; the ref-middle order (3,5,0,1,6,2,4) cuts it 8×). Never tested in P²nP on the
validated pc score. Stopped gc_rm_dfw / gc_rm_lock (measured on this framing); queued chain_gc5.sh: gc_rm_bg
(+ --bin-group), gc_llr_rep (repeat). Decision: if bin-group wins on pc → it becomes the dev default and the prior
components are re-measured with it.

## Exact-time temporal model (implemented 2026-11-24; user: "implement everything and then measure once")

`--temporal linear|cubic` now runs the gated dev config: knot-major engine (`--temporal-engine`, default knot), gating
with one respiratory bin, `--holdout-res`, the pooled warm start (hasty_recon::p2np::ConstOverFrames: every knot the
same image = the time-static problem through the full operator), per-knot TRUE ratemaps (rm_knots from the B-spline
weights at each readout's cardiac time, existing knot path), scoring on frame box averages Σ_k W_fk c_k
(W_fk = ∫_{f−½}^{f+½} B_p(t − k) dt; cubic 0.599 / 0.198 / 0.0026), recon_*.npy = frame box averages. Startup
self-check: set 0 and set ne−1 through the temporal engine vs an independent per-knot reference (static engine with
knot k's table, host weights B_p(t_s − k)), a one-knot table-shift control that must differ ≫, an adjoint dot test.
Independent review: no break for this config; fixed its findings (shift control, static case skipped where the
reference does not apply, recon array, device-memory check, held-out check asserted, interval + truth-bc refused).
Dry start (2 + 2 iterations, previous self-check version): forward vs reference 6.0e-6, adjoint 1.9e-5, held-out
check 0; the pooled warm start's residuals equal frame mode's to 3 digits (it 2: 2.71e-1 2.76e-1 … identical) → same
static problem. Disk: /mnt/hastydata was full (3 gated runs failed: ENOSPC behind "writer thread ended", now reported
properly); deleted the crime k-space store (46 GB).

Queued (chain_t3.sh), one factor vs gc_llr_rm (binned, per-frame true ratemap, LLR, gated):
| run | time model | question | decision |
|---|---|---|---|
| t_cubic_rm | cubic B-spline, 20 knots, exact sample times | does exact-time modelling (no binning error, shared undersampling across a train's encodes, no bin-group cross-talk) improve pc and held-out? | if yes on both → the temporal model replaces binning in dev; next its 4-bin / knot version |
| t_linear_rm | linear, 20 knots | degree | cubic vs linear trade-off (cost 2.0× vs 1.35× per knot image) |
Comparison also with gc_rm_bg (binned + --bin-group, rerun queued).
Full-run self-check (t_cubic_rm, snap_t1): sets 0 and 6: forward vs reference 6.0e-6 / 6.1e-6, one-knot shift control
3.6e-2 (≈ 6000×), adjoint 1.7e-5 / 1.6e-5; per-knot diagonals 4.6 GiB on the device (solve + held-out), 15.9 GiB free.
gc_rm_tmaps failed again in the frame-mode self-check (6.75e-5): with --smaps truth the support is the whole box
(552 801 of 552 960 voxels), so the support mask did not remove the air voxels where the f32-stored per-frame field
differs from the static f64 field. Fixed properly: RateFrame.om is f64 (snap_t2; only gc_rm_tmaps uses it).

### t_cubic_rm vs gc_llr_rm (2026-11-24): exact-time cubic model vs 20 bins (true ratemap, LLR, gated, it 55)
held-out 0.0447 → 0.0439 (−1.8 %, still falling at it 45); pc RMSE 0.241 → 0.226 (hier 0.277 → 0.238); detail 0.224
→ 0.207, time mean 0.089 → 0.091; median 0.075 → 0.070; p90 0.207 → 0.229 (worse); frame 1 0.21 → 0.31 (worse,
systolic upstroke), frame 2 0.64 → 0.56, frame 3 0.55 → 0.49; f 7.84e12 → 8.26e12 (higher data term, lower
held-out: less overfit); wall 17 → 29 min (gradient ≈ 2×). Better on the data-only metric and on most pc numbers;
significance pending the repeat (gc_llr_rep, spread).

### More gated one-factor results (2026-11-24, it 55; pc RMSE, held-out, f ÷ noise floor 2.84e11)
| run | vs | pc RMSE | median / p90 | held-out | f ÷ floor |
|---|---|---|---|---|---|
| gc_llr_svdn5 (svdn energy, L 5) | gc_llr (svdn L 4: 0.262) | 0.273 | 0.074 / 0.247 | 0.0482 | 33.4× |
| gc_rm_bg (+ --bin-group) | gc_llr_rm (0.241) | 0.303 | 0.088 / 0.319 | 0.0432 (other held-out readouts) | 26.0× |
| gc_rm_tmaps (+ true coil maps) | gc_llr_rm (0.241) | 0.203 | 0.065 / 0.181 | 0.0403 | 22.0× |
- svdn at L 5 is no better than svdn at L 4 → the fixed-L5 gain (0.221) is the basis / voxel weights (fixed + count vs
  svdn + energy), not L — or run-to-run spread (repeat pending). The printed segment error is energy-weighted (blood
  is dark): measure the segment error on the lumen voxels before concluding.
- bin-group: frame 0 / 1 errors 0.36 / 0.53 (vs 0.10 / 0.21) → suspicious (timing stamp of the group, or the
  acceleration cross-talk with the 0..6 encode order); not adopted; the exact-time model covers the same issue.
- True coil maps: the largest single operator gain so far (f −20 %, held-out −10 %, pc −16 %), but the residual is
  still 22× the noise floor with true ratemap + true maps.

### What the simulator models, and background phase (2026-11-24, user questions)
sim_flow_10min (meta + simphantom docs): GIRF ideal (no eddy currents / gradient errors); breathing-dependent field off
(static ω per source); spin history ON (beat library default; `--beat-spin off` would remove it); blood motion inside
the readout ON (x + v δ); motion between excitations in 10 ms bins; concomitant fields, GNL, six-peak fat per sample;
static coil maps; tissue as box elements on the 2 mm source grid (vs the 4 mm Dirac recon grid). → Eddy currents and
breathing-field changes are NOT error sources here (my earlier list was wrong).
Background phase: the complex-image decode uses phase differences to encode 0 (FlowDecoder, ML alias search); no
static-tissue polynomial correction. Measured on finished runs (bgvel.py: static = support, magnitude > 40th pct,
lowest 30 % temporal velocity std, outside the 6-voxel-dilated lumen; degree-1 fit per axis of the time-mean velocity):
| run | static voxels | mean (m/s) | poly half-range (m/s) | residual rms (m/s) |
|---|---|---|---|---|
| gc_llr | 71 930 | ≤ 1e-4 | ≤ 5e-4 | 2.2–2.7e-3 |
| gc_llr_rm | 71 848 | ≤ 1e-4 | ≤ 6e-4 | 2.2–2.8e-3 |
| t_cubic_rm | 71 807 | ≤ 1e-4 | ≤ 3e-4 | 0.7–1.4e-3 |
| gc_rm_tmaps | 93 585 | ≤ 1e-4 | ≤ 3e-4 | 0.7–1.5e-3 |
→ no smooth encode-dependent phase error in the sim reconstructions; a practice-style background correction would
change nothing here (needed for real data). Velocity model: φ_e(r) (--vpoly) + static anchor (--vanchor).

## Run-to-run spread (2026-11-24): the pc score at 60 iterations is not reproducible
gc_llr_rep = gc_llr exactly (same config, same tune record, frame-mode path numerically unchanged between the two
binaries). it 55: pc RMSE 0.262 vs 0.173, median 0.079 vs 0.057, p90 0.253 vs 0.171, frame 2 0.78 vs 0.52; held-out
0.0482 vs 0.0476 (−1.3 %); f 9.52e12 (rising) vs 9.12e12 (falling). Divergence starts at fp noise (warm-start CG
residuals at 1e-3, f at it 0 differs 3e-4) and grows from it ~15. → a 0.5–4 % data-term difference carries up to 35 % in
pc: the data barely constrain the velocity-relevant directions, and 60 iterations are not converged.
Consequences for today's comparisons: pc differences below ~0.1 m/s are NOT established (svdn vs fixed, cubic vs
binned, bin-group, true maps on pc). Held-out spread ≈ 1.3 % → held-out effects > ~3 % hold: true ratemap −7 %, true
coil maps −10 % (real); wavelet +3 % (borderline); cubic −1.8 % and svdn/fixed 0.4 % (not established).
Queued chain_conv.sh: conv_a / conv_b, identical gc_llr config to 200 iterations (question / decision in the script).

## Physics: simulated vs modelled, verified in code (2026-11-24, no runs)
Sim phase per sample and source (simphantom.rs header, direct_sum.cu): φ = k·(u + δ J v) + m1·J v + ½ m2·J a + w·m(x)
+ ω τ, u = f(x − iso) − c (GNL), records from excitation (seqsched::records_at → played_moments(tr_wave(e, rot), t_rf)).
| effect | sim | recon | size (from the sequence) |
|---|---|---|---|
| concomitant incl. the encoding lobes (static per-encode offset) | yes | yes, exact: per-sample w (s[9..12]) from excitation, per-encode tables w̄_e, deviation fit ≤ 0.035 rad | 0.13–0.21 rad at 10 cm; encodes differ by ≤ 0.08 rad (= 4 cm/s ∝ r² if unmodelled); measured background ≈ 0 |
| GNL on trajectory and encoding vector | yes (u = f(x), J v) | yes: voxel → true x (Newton on f), m(x), ω(x), J(x); v = J⁻¹ v_app | exact for the linear-in-v phase |
| acceleration (½ ΔM2 of the lobes) | yes | no | = v measured 0.84 ms early: 1.7 cm/s per 20 m/s² (negligible vs 42 ms frames) |
| blood motion in the readout (readout M1(τ)·v) | yes | no | displacement v·τ′ ≤ 1 mm per m/s over the 0.96 ms dev window (~3.9 mm at 2 mm's out half); common to a rotation's encodes, so it cancels in phase differences only if a frame's encodes share rotations (0 % of trains do with per-readout binning) |
| eddy currents | no (GIRF ideal) | yes via the trajectory: HASTY_TRAJ=played / girf:model gives k, M1, concomitant, B0 term (slot 15); the decoder's encode moments for flow schedules are the per-encode mean of M1 at the first ADC sample of THAT trajectory (data.rs) | correction 2026-11-24: only the old spiral sims use the nominal acq.params.encode_m1(); GIRF encode moments are already used |
| spin history (blood magnitude TR to TR) | yes (beat library) | no | not computed yet |
| breathing-dependent field | no | (per-frame truth-bc) | — |

## Solver: why identical runs differ (mathematics, p2np.rs)
x_{k+1} = D(x_k − P_k α ∇f(x_k)), D Euclidean (LLR prox + air mask), P_k = ZMSHR1 rank-1 from the last secant pair
(rebuilt every iteration). Fixed points of x = D(x − P α ∇f(x)) depend on P (Proposition G, metric displacement), and P_k
depends on the trajectory: two runs that differ by fp noise follow different P_k and settle at different points; the
displacement lives where the data do not constrain the image (velocity-relevant directions), consistent with held-out
equal to 1.3 % and pc 35 % apart. The air mask (re-estimated every 2 iterations) adds discrete changes. → evaluation
needs a solver with a history-independent fixed point (fixed metric, or a fixed-point-preserving acceleration).

## Fixed-objective solver test (pre-registered 2026-11-24; research/solver_choice.md)
Implemented (binary snap_fx after review): --llr-shifts S (proximal average over fixed shifts, all prior modes),
--llr-thr freeze (penalty λ = κσ̂/s from the first call; every call thresholds at λ·s with s = the prox step:
α for P²nP, ατ_k for the metric prox, 1/ρ for ADMM; hasty_recon::p2np::prox_step), --prox-metric (global ZMSHR1 with
the prox in the step's metric S⁻¹ = (1/τ)I + κ w wᵀ, Becker–Fadili–Ochs root by secant; unit test metric_prox.rs:
equals the brute-force prox in V to 2e-7 / 1e-8, 3–4 prox calls at 1e-6). Air from the start image, frozen.
Baseline irreproducibility (compare_runs.py gc_llr vs gc_llr_rep): velocity difference RMS 0.215 m/s on the 909 pc
voxels, 0.53 on the support, image 4.4 %.
Runs (gated gc_llr config + fixed objective, 100 iterations): fx_bfo_a, fx_admm (ADMM, ρ = ρ₁, 3 inner PCG, clip
circulant inner), fx_bfo_b (= fx_bfo_a, fp-noise repeat), fx_p2np (Euclidean prox, the metric still displaces).
- V1 reproducibility: fx_bfo_a vs fx_bfo_b velocity difference < 0.01 m/s on pc voxels and image < 1e-3.
- V2 solver equivalence: fx_bfo_a vs fx_admm < 0.01 m/s (if both converged; else note which is still moving).
- fx_p2np vs fx_admm: expected to differ (Euclidean prox + dynamic metric); size = the metric displacement.
Decision: V1 and V2 pass → the fixed objective + metric prox (or ADMM) is the evaluation solver for every later
comparison; V1 fails → find the remaining non-determinism before anything else; V2 fails with V1 passing → one solver
has not converged or the objectives differ (check λ·s bookkeeping).
Revised after the independent review (2026-11-24): (1) λ is fixed at setup in objective units from the shared start
stack (λ = κσ̂(x0) λ_max(N)) — per-arm freezing would have given ADMM a ~166× weaker prior; (2) one fixed partition
(--llr-shifts 1): a proximal average over S > 1 shifts is not homogeneous in the prox step (different solvers would
minimise different objectives, and the metric prox's prior would vary with τ_k); (3) binary air mask from the start
stack (--air start; the ramp is not a projection, and per-arm estimation gave different masks); (4) safeguarded root
(slope bounds, bracket, best point, tolerance 1e-4, ≤ 6 prox calls); (5) --state device. Air-map oracle line added
(zeroed voxels holding true tissue, scoring only). Arms: fx_admm_a, fx_admm_b (V1), fx_bfo (V2 + prox calls per
iteration), fx_p2np (displacement size), after a 3-iteration dry start of fx_bfo.
Dry start 1 stopped at the λ guard: σ̂ of the start stack is 0 (every frame the same image → block Casorati rank ≤ 7, lower singular values 0). λ is now measured on PnP-ISTA's first prox input z0 = x0 − ∇f(x0)/λ_max (one data gradient at setup, shared by all arms).
Dry start 2 (fx_dry_bfo, 3 iterations) passed: λ = κσ̂(z0) λ_max = 1 × 1.604e-3 × 1.725e8 = 2.767e5; prox threshold
1.604e-3 at step α and 1.343e-3 at step ατ (τ 0.837) — the λ·s bookkeeping works; metric prox 3 prox calls per
iteration, |L| ~ 1e-8; prior 4.3 s vs 1.5 s, iteration ≈ 1.7× (gradient 2.7 s). Air map oracle: 1 764 of 137 713
zeroed voxels hold true tissue (PD > 5 % of the body median, max 1.57× the median) → the mask is not as conservative
as intended; to localise (the truth PD is at the phantom's reference pose, the data at the gated breathing range).
Air oracle localised (airloc.py, approximate grid mapping): 1 439 flagged; 1 262 at the true body surface, 177
inside. The surface ones are all lateral (|x| > 83 mm: the arms) and 1 152 of them in the lowest 13 mm of the box
(z −104..−91 mm), up to 8–12 mm deep in the zeroed region at full PD. The sim excites a slab (Bloch-simulated
profile over the FOV z), so the data have ~no signal at the slab edge while the truth PD (reference pose, no slab
profile) does: zeroing there agrees with the data; the oracle overcounts. Not the breathing issue; per-bin
estimation + a wider margin remain needed for multi-bin runs (user, 2026-11-24). TODO: weight the oracle PD by the
slab profile; inspect the 177 inside-body voxels.

### Fixed-objective test: outcome (2026-11-24)
All four arms ran 100 iterations (fx_admm_a/b 29 min each, fx_bfo 17 min, fx_p2np 15 min). compare_runs.py, on the
909 pc voxels / the support / the image:

| pair | velocity diff on pc voxels (m/s) | on support (m/s) | relative image diff |
|---|---|---|---|
| fx_admm_a vs fx_admm_b (identical config) | 0.064 | 0.34 | 9.4e-3 |
| fx_bfo vs fx_admm_a / _b | 0.166 / 0.168 | 0.75 | 3.9e-2 |
| fx_p2np vs fx_admm_a | 0.160 | 0.73 | 3.4e-2 |
| fx_p2np vs fx_bfo | 0.123 | 0.63 | 2.2e-2 |

- **V1 fails** (criterion < 0.01 m/s and image < 1e-3).
- **Cause: the setup is not deterministic.** Identical configurations give:
  - a different start stack: f(x0) 1.2426e13 vs 1.2413e13 (0.1 %);
  - a different segment-basis fit set: 149 133 vs 149 130 voxels, from the 0-iteration warm start;
  - a different λ: 2.7653e5 vs 2.7669e5;
  - a different air mask: 137 722 vs 137 783 zeroed voxels across arms.
  The GPU adjoint's atomic summation differs at ~1e-7, and the 60-iteration pooled PCG warm start amplifies this to
  1e-3.
- **The two runs' data terms stay a constant 1.2e10 apart from iteration 5 to 60,** about the offset at x0. At 100
  iterations the iterate still remembers its start: not converged. f still drops ≈ 0.1 % per 5 iterations in every arm.
- **V2 cannot be judged.** The solver distances (0.12–0.17 m/s) exceed the start-noise distance (0.064) but are
  path differences at 100 iterations, not fixed-point differences.
- **The velocity RMSE vs truth is 0.36–0.38 m/s (rel 0.84–0.89) in every arm,** the same as at iteration 0 (0.38).
- **Decision (pre-registered: V1 fails → remove the non-determinism first):**
  - one shared setup for all arms: write the warm start, segment basis, air mask and λ once and load them;
  - before any further solver comparison, estimate the distance to the minimiser: one long reference arm, or the
    objective value F = f + λ Σ‖·‖_* printed per iteration. This is not measured yet; F is not logged.
- **Not re-run.** The flow phantom has unphysiological pulmonary speeds (research/experiments/flow_figs/README.md),
  to be settled first.
