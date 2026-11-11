# Reconstruction diagnostics, 2026-11-11: pyramid trial, κ, truth start, velocity decoder

Data: factory_arms/ph0001 10-min simulated scan, current (out-and-back) readout, oracle physics (truth fat per frame,
truth table R2*, GNL), 2 respiratory × 20 cardiac frames, 7 encodes (coprime7: per axis VENC 1.0 and 1.5 m/s, combined
unaliased range ±3 m/s). Scripts: `research/experiments/recon_diag/` (`vel_error.py`, `vel_vs_mean.py`,
`vel_vs_depth.py`, `redecode.py`, `decode_margin.py`). Runs: `data/hasty/phantom/factory_arms/ph0001/{pyr_2mm,
pyr_kappa, pyr_inittruth, fix_2mm/llr_s}`.

## 1. Pyramid with the audit follow-ups (`pyr_2mm`)

Levels 8 / 4 / 2 mm (grid scale 0.25 / 0.5 / 1, coarse levels band-cropped), physics warm start on the coarsest level,
air map seeded from the previous level, `--nseg auto:noise:0.2 --seg-basis svdn --seg-weight energy --b0-absorb 1`,
`--precond zmshr1s`, κ 1.

| level | segments (old rule: 22) | s / iteration | rel. velocity error (hierarchical decoder) |
|---|---|---|---|
| 8 mm (10 it) | 14 | ≈ 20 | 0.941 → 0.871 |
| 4 mm (10 it) | 15 | ≈ 90 | 0.905 → 0.856 |
| 2 mm (20 it) | 16 | (running) | |

Air map: 1.7 k voxels at 8 mm, no lumen zeroed. Driver fix: `pyramid.sh --next-args "--air-init {prev_air}"`
(the air map's file name carries the level's tag).

## 2. The prior is not what limits the result (`pyr_kappa`)

8 mm level, κ = 0 / 0.05 / 0.25 / 1: f after 10 iterations 4.576e14 / 4.577e14 / 4.581e14 / 4.557e14, rel. error
0.898 / 0.895 / 0.891 / 0.871. Residual at that level: f = 4.58e14 against a noise floor ½σ²MC = 4.0e11 for its 16.3 M
cropped samples × 31 coils, i.e. 34× the noise rms (2 mm baseline: 6.5×; the readout-physics budget allows ≈ 2 for this
design, dominated by table R2*).

## 3. Starting from the truth (`pyr_inittruth`, `--init-truth`)

Warm image × true velocity phases in the lumen, 8 mm, κ 1 and 0 (identical): diastole frames start at 0.04–0.09 m/s
and drift to 0.15–0.30 m/s within 10 iterations; overall rel. error 0.646 → 0.868 (the normal start ends at 0.871). So
the 8 mm data term pulls the solution away from the truth: model error of the coarse grid (intravoxel field / fat / R2*
evolution at 8 mm [CHECK]); the coarse levels are starting points, not velocity estimates. The same test at 2 mm is
open.

## 4. Velocity decoder bug (fixed)

`hasty_recon::flow::FlowDecoder` unwrapped hierarchically from the least sensitive encodes (VENC 1.5: ±1.5 m/s per
axis). The coprime design's combined range is ±3 m/s, so every voxel with a component above 1.5 m/s aliased. On
noise-free true phases (2 mm lumen, `redecode.py` check): systole (cardiac 3, 7.8 % of the lumen above 1.5 m/s)
0.80 m/s rms error vs 0.0000 for a maximum-likelihood decode over ±3 m/s; diastole 0.10 vs 0. Every systolic velocity
metric reported before 2026-11-11 is inflated by this (metrics only: the reconstructed images are unaffected; velocity
priors such as `--dfw` / the static lock used the same decoder).

Fix (additive): `FlowDecoder::with_alias_search(AliasSearch { vext, margin })`. Candidates are the first stage's
aliases v₀ + 2π K₁⁻¹ n inside |v_a| ≤ vext, each refined through all stages. Cost Σ(1 − cos(φ_e − k_e·v)); an alias
wins only with a cost advantage > margin. `recon_p2np --vdecode auto|hier|ml[:MARGIN]` (auto = ml:1.0 when the
sim's `sequence.encoding.unaliased` exceeds the first stage's range); the log prints the hierarchical RMSE alongside.
Tests: `flow::tests::coprime_alias_search_covers_the_combined_range` (exact to 1e-9 over ±2.99 m/s; the hierarchical
decoder fails on > 2000 / 4000 draws), `alias_search_keeps_in_range_answers_and_reference_layout`.

Margin on today's images (`decode_margin.py`, fix_2mm/llr_s, 6 frames): mean rms 0.915 (margin 0, plain ML) / 0.771
(0.5) / 0.673 (1.0) / 0.613 (in range only); |z₀|²-weighted cost: 0.771 / 0.666 / 0.634 / 0.613. On these noisy,
shrunk images no alias decision helps (their systolic velocities rarely exceed 1.5 m/s); on accurate images the
in-range answer is wrong for 7.8 % of the systolic lumen. The margin (default 1.0) trades the two; a proper treatment
is spatial / model-based unwrapping (the velocity-parametrised model) [CHECK: re-tune once the images are good].

## 5. Velocity shrinkage (decoder-independent part)

fix_2mm/llr_s (2 mm, κ 1, 20 it), hierarchical decoder:
- recon vs truth slope 0.33–0.48 per component, correlation 0.23–0.55; error rises with speed and in low-|x| voxels
  (rms 1.10 / 0.69 / 0.53 / 0.33 m/s by |x| quartile, systole);
- the recon tracks the time-averaged truth more than each frame (rms 0.40 vs 0.45 m/s), and even the time-mean
  velocity is halved (slope 0.44);
- by depth inside the lumen (time mean): slope 0.32 at the wall (corr 0.50) → 0.64–0.67 at ≥ 4 voxels (corr 0.93–0.97):
  partial volume with static tissue, consistent with unconverged high k.

## 6. Next

- 2 mm level of `pyr_2mm` (running) vs the baseline; re-decode both with the new decoder.
- `--init-truth` at 2 mm: does the full-resolution data term also leave the truth?
- Start from per-frame data (density-weighted adjoint, LS-scaled), consistent LLR prox, momentum
  (`research/convergence_p2np.md` §5) after the recon-physics agent's patch lands.
- `--synthetic` with streamed data (inverse crime at 2 mm) to separate solver and model.
