# Field absorption before the segment SVD: per-knot segment counts

Round of 2026-11-20, CPU only. Tool: `rustlib/crates/hasty-bench/src/bin/fieldabsorb.rs` (new). Runs, logs and JSON:
`research/experiments/field_absorb/` (`run_all.sh`, `summarize.py`).

## Question (pre-registered)

How many time segments L does the knot engine's segment model need when the constant and linear spatial parts of
the field are moved into the per-sample tables (constant → per-sample phase φ0, linear → per-sample coordinate shift)
before the segment SVD? How large are the coordinate shifts?

Decision it informs: build per-knot sample tables with per-knot absorption (written before sorting) if L drops
noticeably. Fewer segments means fewer FFTs per gradient.

## Answer

**No. On ph0001, absorbing the constant and linear part, per knot or globally, does not lower L.** With the
least-squares fit that `--b0-absorb 1` uses it raises L by one segment at the coarse grid. Even the best possible
linear term (found by minimising the segment tail directly) changes L by at most one, and it changes nothing at 4 mm
or 2 mm. The constant part cannot change L at all: a phase that is the same for every voxel is a unitary factor
on the time axis, so it leaves the SVD spectrum unchanged. What sets L is the local field structure (susceptibility
tails, up to about ±600 Hz in signal voxels) and the fat term, not a smooth trend.

Two side findings matter more than absorption:

1. **Fat costs 2–3 segments per knot.** Water-only models need L = 8–9; with the knot's fat fraction and the six-peak
   spectrum they need L = 10–11.
2. **Today's `--ratemap truth-bc` engine uses too few segments.** It picks L from the static field with phase only,
   without decay and fat: L = 8 here (recon log: 9). The per-knot models it then builds need L = 11 under the same
   noise rule. This is consistent with the per-knot error the recon itself prints: 9.5e-3 and 1.0e-2 against a
   target of 2.8e-3 (research/experiments/ratemap_bc/ab_truth-bc.log). The rule should be evaluated on the per-knot
   g (with decay and fat), taking the maximum over knots.

## Data and method, in words

- **Scan:** phantom ph0001, `sim_flow_10min` (10-minute thorax 4D-flow simulation, 7 velocity encodes, 3990 ADC
  samples over 1.60–9.58 ms, 32 coils, GNL on, full concomitant physics).
- **Ratemap:** the TRUE per-knot ratemap from the simulator's per-state maps (`scan/truth_ratemap_bc/`: 6 breathing
  × 20 cardiac states). It is evaluated at each knot's state exactly as recon_p2np's `truth_ratemap_knots` does:
  the basis-weighted mean of the true breathing amplitude and the true cardiac time over the readouts. The ratemap
  is off-resonance, water and fat decay rates and fat fraction. Its field is the static field advected with the
  motion; the simulator does not recompute susceptibility as the lungs inflate.
- **Time basis:** the default `--knots2d 6:1,16:3`: 6 linear breathing knots on the bellows signal × 16 cubic
  cardiac knots, 96 knots in total. The frame alternative is `6:0,16:0`: 6 breathing bins × 16 disjoint cardiac
  frames, also 96, with a separate fit per frame.
- **Segment model of one knot and encode:** the same function recon_p2np builds for `--ratemap truth-bc`. It is the
  knot's water and fat decay plus the fat spectrum, times the phase of the knot's field plus the encode-mean
  concomitant phase (conc0).
- **Rule:** the default noise rule of knot mode, `auto:noise:0.2`. L is the smallest number of segments whose
  noise-weighted SVD tail predicts at most 0.2 noise rms in the data, with the 1.5 margin. That is a relative
  weighted error of 2.83e-3.
  - Voxel weights: smoothed warm-start magnitude² × coil energy over the fit region, divided by the median.
  - ADC-sample weights: 1 + (E(n) + E_vol)/(Cσ²) on the 95-point time grid.
  - L is also given at relative tolerances 1e-3 and 1e-4.
- **Weights and grids:** the warm start, the support and the fit region come from existing runs on three grids,
  with the support recomputed by the recon's own code. It matches the run's `mask_fit` exactly at the coarse grid.
  - matrix 64×36×30: `research/experiments/ratemap_bc/ab_truth-bc`, 8 coils;
  - matrix 128×72×60: `vknot/smoke4mm`, 31 coils;
  - matrix 254×144×114, the native grid: `vknot/dev_k8/B_vknot`, 12 coils. Here the rule uses every 12th fit voxel;
    the check against every 3rd voxel gives identical L, below.
- **Coil compression:** the coil energy uses the true whitened maps, compressed by their own Gram over the fit
  region. The run's calibration-region compression needs the GPU.
- **Coverage:** all 96 knots × 7 encodes at 64×36×30, every 3rd knot at 128×72×60, every 6th knot at the native
  grid. Every subset keeps all 6 breathing knots. The encodes give identical spectra (explained under the
  concomitant-phase finding), so the finer grids use encodes 0 and 3.

Variants (what the segments still carry):

| | phase in the segments |
|---|---|
| a | today: knot field + conc0, nothing absorbed |
| b | `--b0-absorb 1` as implemented: one global constant + linear fit of the static field |
| c | per knot constant + linear (weighted least squares on the knot's own field, same weights) |
| d | per knot constant + linear + quadratic, quadratic through ΔC relative to the readout centre (as `absorb_field_poly`) |
| e | c + the constant + linear part of conc0 (per encode, weighted fit of the monomials) |
| f | d + the constant + linear + quadratic part of conc0 |
| g | per knot constant + linear, L8 fit (close to minimax: minimises the field range, not the variance) |
| h | per knot linear term that directly minimises the weighted SVD tail at L_a − 1 (Nelder–Mead, a best-case bound) |

`_nofat` = the same model with fat fraction 0 (water only), on every 4th knot.

**Control:** a synthetic ratemap that is exactly constant + linear, 2π·(40 Hz + 1500 Hz/m·y + 800 Hz/m·z), with
knot 0's decay and fat.
- Without absorption: L = 10 / 11 / 13 (noise rule / 1e-3 / 1e-4).
- After per-knot absorption: L = 3 / 4 / 6, identical to a zero field (3 / 4 / 6) on every encode and grid.
- The weighted tails agree to ≤ 3.3e-5 relative.

The absorption path is therefore exact; the absence of a gain on the real ratemap is a property of the field.

## L per variant

L under the noise rule: maximum over knots and encodes (the engine needs one L for all knots), with the median in
brackets. Then L at 1e-3 and 1e-4 (maximum).

| variant | 64×36×30 (96 knots × 7 enc) | 128×72×60 (32 knots × 2 enc) | 254×144×114 (16 knots × 2 enc) |
|---|---|---|---|
| today's rule (static field, phase only) | 8 | 9 | 8 |
| a: none | **11** (10) · 12 · 14 | **11** (11) · 13 · 15 | **11** (11) · 13 · 15 |
| b: global const + lin | 11 (11) · 13 · 15 | 11 (11) · 12 · 15 | 11 (11) · 13 · 15 |
| c: per knot const + lin (LSQ) | 11 (11) · 12 · 15 | 11 (11) · 12 · 15 | 11 (11) · 12 · 15 |
| d: c + quadratic (ΔC) | 10 (10) · 11 · 14 † | 10 (10) · 12 · 15 † | 11 (11) · 12 · 16 † |
| e: c + conc0 const + lin | 11 (11) · 12 · 15 | 11 (11) · 12 · 15 | 11 (11) · 12 · 15 |
| f: d + conc0 quadratic | 10 (10) · 11 · 14 † | 10 (10) · 12 · 15 † | — |
| g: per knot L8 fit | 10 (10) · 12 · 14 | 11 (11) · 12 · 15 | 11 (11) · 12 · 15 |
| h: tail-optimal linear (subset) | 10 (10) · 12 · 14 | 11 (10.5) · 12 · 15 | 11 (11) · 12 · 15 |
| a, water only | 8 (8) · 9 · 11 | 9 (9) · 10 · 12 | 9 (8.5) · 10 · 12 |
| c, water only | 9 (8) · 10 · 12 | 8 (8) · 10 · 12 | 8 (8) · 10 · 12 |

† The quadratic part goes through the ΔC Taylor correction with an argument |Q|·|τ − τ_ref| of 8.9 / 10.7 /
15.4 rad (median over knots; maximum 10 / 11.7 / 16.2 rad). That is far outside its validity (≪ 1 rad), so d and
f are not realisable. The L shown assumes a correction that is exact. Even so, the gain is at most one segment and
zero at the native grid.

Paired changes against a, per knot and encode, under the noise rule:
- c − a: +0.83 mean (always 0 or +1) at 64×36×30; 0 at the two finer grids.
- g − a: −0.17 / −0.34 / 0.
- h − a: −0.17 / −0.5 / 0.

The tail-optimal linear term lowers the weighted tail at L_a − 1 only to ×0.93 / ×0.86 / ×0.87 of no absorption.
The least-squares term raises it to ×1.30 / ×1.15 / ×1.19. The tail-optimal term is small: median |18, 50, 41| Hz
across the FOV on x, y, z at 64×36×30, against the LSQ fit's |17, 83, 280| Hz.

Why least squares makes things worse: it minimises the signal-weighted variance of the field (std 111 → 82 Hz at
64×36×30, 138 → 122 Hz at the native grid). But the SVD tail follows the weighted range, which widens: the
0.1–99.9 % span goes from 581 to 614 Hz. The z-gradient the fit removes is a real large-scale trend, but tissue
with extreme field values sits on both ends of it.

**Frames instead of knots** (`6:0,16:0`, a separate fit per frame): identical results.
- 64×36×30: a 11, c 11, d 10, e 11, g 10; water only a 8, c 9.
- 128×72×60: a 11, c 11, g 11.
- The per-frame coefficients match the per-knot ones to within 1 Hz.

Frames give no absorption advantage. The disjoint frames' field is no smoother in space than the overlapping
knots' field.

## Coordinate shifts

The shift of sample j along axis a, in cells of the oversampled grid (σ = 1.25), is
|c₁,a| · Δ_a · τ_j · σ N_a / 2π = (Hz across the FOV along a) · τ_j · σ. It depends on the field across the FOV, not
on the matrix size.

Maximum over the knots, per-knot LSQ fit, x / y / z:

| grid | τ from excitation (τ up to 9.58 ms) | τ − τ_ref (readout centre, ±3.99 ms) |
|---|---|---|
| 64×36×30 | 0.24 / 1.04 / 3.61 | 0.10 / 0.43 / 1.50 |
| 128×72×60 | 0.09 / 1.26 / 2.67 | 0.04 / 0.52 / 1.11 |
| 254×144×114 | 0.16 / 1.17 / 3.01 | 0.06 / 0.49 / 1.26 |

The global fit (b) gives the same: 0.25 / 0.95 / 3.63, 0.02 / 1.05 / 2.70 and 0.16 / 0.87 / 3.08 cells. Referencing
the linear term to the readout centre leaves a static linear phase ramp in the segment diagonals. That ramp does not
change the rank, and it cuts the shift by 2.4×, to at most 1.5 cells.

## Fitted coefficients and their variation across knots

Per-knot LSQ fit (constant + linear), over all knots:

| grid | constant (Hz) | x / y / z (Hz across the FOV) | x / y / z (Hz per voxel, max \|·\|) |
|---|---|---|---|
| 64×36×30 | 270–281 | 13…20 / 80…87 / −301…−262 | 0.32 / 2.42 / 10.0 |
| 128×72×60 | 268–278 | −7…−2 / 88…105 / −223…−195 | 0.06 / 1.46 / 3.72 |
| 254×144×114 | 258–271 | 7…13 / 75…98 / −252…−206 | 0.05 / 0.68 / 2.21 |

- **Across cardiac knots,** at a fixed breathing knot: essentially constant. The standard deviation is at most
  0.6 Hz for any coefficient.
- **Across breathing knots:** a monotone drift from exhale to inhale. At the native grid the constant goes from 271
  to 258 Hz. The y term goes from 75 to 97 Hz across the FOV, and z from −251 to −206 Hz across the FOV, which is
  about 45 Hz of change in the z gradient.
- **Shifts:** this drift moves the z shift by about 0.5 cells between the end breathing knots.
- **Caveat:** the phantom's field only follows the motion of the static field. Real breathing also changes the
  susceptibility field, so these per-knot differences are a lower bound.

Signal-weighted field range per knot (median over knots, Hz):

| grid | metric | none | global lin | per-knot lin | per-knot quad | per-knot L8 |
|---|---|---|---|---|---|---|
| 64×36×30 | 0.1–99.9 % span | 581 | 625 | 614 | 480 | 543 |
| 64×36×30 | 1–99 % span | 476 | 440 | 431 | 364 | 430 |
| 64×36×30 | weighted std | 111 | 82 | 82 | 61 | 96 |
| 254×144×114 | 0.1–99.9 % span | 640 | 646 | 632 | 551 | 612 |
| 254×144×114 | 1–99 % span | 515 | 515 | 507 | 422 | 491 |
| 254×144×114 | weighted std | 138 | 122 | 122 | 90 | 124 |

## Encode-mean concomitant phase (conc0)

- **Linear part: negligible.** In logical coordinates, the constant + linear fit of conc0, with the same weights,
  corresponds to a shift of at most 0.13 cells (z), 0.06 cells (y) and 0.04 cells (x) at any time on any grid. The
  FOV is centred at the isocentre and the support is nearly symmetric.
- **Constant part:** up to 0.6–0.76 rad. It cannot change L.
- **Effect on L: none.** Removing the linear part (variant e) gives exactly the L of c.
- **Encodes:** they differ only through the encoding lobes, whose concomitant phase is constant during the ADC
  window. A per-voxel phase that is constant in time leaves the spectrum unchanged, so all 7 encodes give
  identical spectra (to 4 digits) and identical L.
- **Quadratic part:** conc0 is almost purely quadratic. Removing its quadratic part shrinks its weighted spatial
  span from 1.8 to 0.08–0.12 rad, but this is worth no segment: d → f is unchanged. Its Taylor argument relative to
  the readout centre is 0.87–0.88 rad, which is also outside the ΔC validity.

## Checks

- **Stride (native grid, knots 0 and 48, encode 0):** using every 3rd instead of every 12th fit voxel gives the
  same L for a and c. The tail at L = 10 is within 7 %.
- **Coil weighting:** with all 32 uncompressed coils, today's rule still gives L = 8 at 64×36×30, and a still needs
  11. The recon log's L = 9 for today's rule differs from my 8 by one segment. The likely causes are its
  calibration-region coil basis and its `--keep 8` readout subset in the concomitant mean; this was not resolved.
  All comparisons above are paired within the same weights.
- **Control (above):** absorption of an exactly constant + linear field reproduces the zero-field spectrum.

## Recommendation for the per-knot tables

- **Do not build per-knot sample tables for field absorption now.** On ph0001 they buy no segments: 0 at 4 mm and
  2 mm, and at best one segment at 8 mm, and only with a non-LSQ fit.
  - A per-knot coordinate shift makes the tables knot-dependent. Each sample has weight on up to 8 knots, so that
    means up to 8 sorted coordinate copies per sample, against today's one.
  - It changes nothing about the FFT count.
- **Spend the effort on the rule instead.** In `--ratemap truth-bc` mode, compute L from the per-knot g, with decay
  and fat, as the maximum over knots. That is a cheap spectrum per knot, about 0.1 s at 64×36×30.
  - Today the engine runs at L = 8–9 while the per-knot models need 11. This is an accuracy issue (model error about
    3× the target), not a speed issue.
  - Fat is the larger lever: 2–3 segments. A fat-aware basis, such as the union water + fat SVD, is where L can come
    down.
- **Keep `--b0-absorb 1` for real data only as shim robustness.** segment_basis.md §4 shows it removes a synthetic
  linear shim residual exactly. If it is used, consider a range-minimising fit (L8) instead of least squares,
  because least squares widens the extreme range here.
- **If per-knot tables are built later,** for example for motion or a breathing-dependent linear shim on real data:
  - write the shift relative to the readout centre (at most 1.5 cells; the static ramp goes into the diagonals);
  - use one fit per breathing knot only, since the cardiac variation is below 1 Hz;
  - this gives 6 table variants, not 96, and the shift differences between breathing knots are about 0.5 cells.
- **Never absorb the quadratic part through ΔC at this curvature.** Its Taylor argument is 9–16 rad.

## Limitations

- **One phantom**, and its field only follows the motion of the static field (no breathing susceptibility change).
  Real subjects may have a larger breathing-dependent linear part, which absorption would remove exactly (see the
  control).
- **Predicted, not measured, data error:** L comes from the noise rule's spectrum prediction, the same quantity
  recon_p2np uses, not from an exact data-space error on the operator. No GPU was used.
- **Coil energy** uses the true maps compressed over the fit region, not the run's calibration-region basis (a
  one-segment offset in today's rule at 64×36×30).
- **Coverage:** the finer grids use knot subsets (all breathing knots, every 3rd or 6th cardiac knot) and 2 encodes.
