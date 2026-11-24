# Dev configuration for the preconditioner evaluation (draft for review, 2026-11-22)

Purpose: compare preconditioners (current circulant metric `zmshr1m`, plain step `zmshr1`, Jacobi, and the coil + segment
surrogate) inside the real regularised reconstruction. The score is wall time to the same final quality.

Every knob is listed against the production target. **Deviations from production are marked ⚠.**

| knob | dev setting | production target | note |
|---|---|---|---|
| data | ph0001 sim_flow_10min, measured (simulated) k-space, with its noise (noise_rel 0.02) | real scan | no inverse crime, no NOISE_SCALE=0 |
| readout part | **out half only**: \|k\| increasing, first centre pass → first \|k\| maximum (new `--adc-part out`, default) | same | return / spoiler / encoder samples reserved for navigators |
| sampling | all readouts (`--keep 1`), 53 144 readouts, 10 min | same | readouts per (resp bin, cardiac frame, encode) set: filled in after the agent's count |
| frames | 20 cardiac × R respiratory bins (`--resp-bins R`, bellows), R = 4 | resp × cardiac always | ⚠ R = 4 to keep sets ≥ ~70 readouts; production R from the error budget |
| model | complex per-frame images, P²nP (frames × encodes) | velocity-parametrised P²nP with temporal basis | ⚠ the per-frame complex model is the baseline the user named; preconditioner lessons carry over to the data term |
| grid | 128 × 72 × 60 (`--grid-scale 0.5`, `--kcrop`, 4 mm) | 254 × 144 × 114 (2 mm), later ~1 mm | ⚠ dev scale (fast dev phantoms) |
| coils | SV compression keeping ≥ 99.9 % energy (`--vcoils auto --coil-energy 0.999`) | same | not the fixed `--vcoils 8` of the earlier dev runs |
| coil maps | ESPIRiT from the data (`--smaps espirit`) | ESPIRiT now, joint later | not truth maps |
| physics | `--phys full --gnl --conc-fit deg4`; **actually ONE static true field** (`--field truth` = field_ppm_recon, the reference pose) for every frame, not the true ratemap per frame: `--ratemap truth-bc` exists only with `--knots2d` (found 2026-11-23) | estimated ratemap per frame | ⚠ per-frame true ratemap (the decided model) to be implemented for the frame-mode path; `--nseg auto` |
| prior | LLR (κ) + db4 wavelet (+ DFW on velocity) + static-tissue lock + **air zeroing (`--air auto`)** | learned denoiser + air | ⚠ until the denoiser is trained; air was missing in the runs of 2026-11-22/23 (added 2026-11-23) |
| iterations | to a fixed quality, measured on held-out k-space | same | `--holdout`, `--holdout-eval` |
| score | held-out k-space residual (data-only) + pc velocity score vs truth (`--score-pc`, scoring only) | held-out only | no truth inside the method |

## The prior (user, 2026-11-22: "LLR + wavelet and the static averaging and averaging within cardiac or resp bin")

What exists in `recon_p2np` now:
- `--prior llr --kappa κ`: LLR over frames × encodes (blocks, rank-thresholded).
- `--prior wavelet --wavelet THR:LEVELS`: db4 complex shrinkage. It is currently an alternative to LLR, not a
  combination. The `--wavelet-mix β` mix exists only together with `--learned`.
- `--dfw sure`: divergence-free wavelet shrinkage of the decoded velocity, after the prior.
- `--static-lock MU[:IT0[:EVERY]]` (`hasty_recon::tlock`): the per-axis static / dynamic lock. This IS the
  "average within the static temporal dimension" prior; my earlier draft wrongly said it was missing.
  - It estimates the map from the iterate (no labels): cardiac variation, respiratory variation, the noise level from
    cardiac second differences, the detection power, and a flow criterion so steady flow is never locked.
  - It pulls each frame toward the mean over the axis on which the voxel is static. The decomposition is
    x = x̄ + (bin mean − x̄) + (frame − bin mean): the cardiac lock within each respiratory bin uses w_c, the
    respiratory lock on the bin means uses w_r.
  - The spine is locked over both axes, tissue that moves only with breathing over the cardiac axis only, and vice
    versa.
  - Quadratic or group prox (`--lock-mode`). The knot-model version is `knotlock` (G_s-metric, additive / nested
    coupling).
  - Choice for dev: the prox form (inside the denoiser step), `--lock-mode group` (a static voxel goes exactly to its
    mean; a mis-mapped moving voxel keeps its motion minus the threshold). The respiratory lock needs `--resp-bins` > 1.
- `--mag-couple`: weak magnitude coupling of the encodes.

Gap to implement (small, behind a switch):
1. **LLR + wavelet together without `--learned`**: x ← x + κ((1 − β) LLR(x) + β W(x) − x), the same mix form as
   `--wavelet-mix`. Then the DFW velocity shrinkage and the static lock follow, as they do now.

Strengths (κ, THR, MU, β) are set once on held-out k-space. They are not tuned per preconditioner: the estimator must be
the same for all preconditioners.

## Preconditioner comparison inside this config
- Arms: `--precond zmshr1` (no metric), `zmshr1m` (current circulant metric), Jacobi metric, and the surrogate metric
  in its fixed Chebyshev form (fixed SPD, so it is valid inside P²nP).
- Same start (pooled plain-CG warm start), same prior, same iterations budget.
- Report:
  - wall time and iterations to the held-out k-space plateau, and to fixed pc-score levels;
  - the fixed-point check: the final images of all arms should agree to within the held-out noise floor; otherwise the
    metric changed the estimator.
