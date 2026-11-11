# Readout physics: error budget and the cheapest correct handling (ph0001, current vs v2 readout)

Round of 2026-11-10/11 on factory_arms/ph0001 (2 mm thorax, 3 T). Question: which physical terms of the data model
are as large as the errors we already model (B0, concomitant fields, fat) or the segment-rule tolerance R ≈ 0.2, and
what is the cheapest correct handling of each. Both readouts: `current` (`data/hasty/seq/flow_thorax`, out-and-back,
7.98 ms) and `v2` (`flow_thorax_v2`, centre-out 9.49 ms reconstruction window + M1-nulled rewinder; not simulated,
evaluated from its waveform). Scripts: `research/experiments/readout_physics/`; arrays, figures, `tables.md`:
`data/hasty/phantom/factory_arms/ph0001/physics_budget_2026-11-10/` (README there). Code (additive):
`rustlib/.agents/seg`, diff in `rustlib/.agents/phys.patch`.

**Summary.** On one noise-referenced scale (R = data error rms / noise rms; R_loc for errors from one region,
§1), against the bar "comparable to what we already model, or the segment tolerance R ≈ 0.2":

* **Already handled well**: B0 with the svdn rule (L 9 current / **10 v2**, R ≈ 0.11), concomitant fields (residual
  0.02 / 0.002). v2's centre-out readout cuts the concomitant omission 14× and the R2* omission 3.6×.
* **Pass the bar, cheap fixes exist**:
  * **Fat with one map for all cardiac frames** (R 0.65–0.67, R_loc 1.2–1.9 around the heart; frame-0 map
    1.14–1.19): per-frame maps with a
    *union* water + fat noise-weighted SVD in cardinal form, **L 15 on v2 (R 0.11) = one segment more than one folded
    map, three fewer than today's per-frame path** (+7 % per set vs one map, −17 % vs today's per-frame path).
    Implemented (Rust builder + `--seg-basis svdn-union`), CPU-tested.
  * **R2* / intravoxel decay** (omission 1.66 on v2, table map 1.40): mostly a **field-map definition** issue — the
    signal-weighted voxel frequency instead of the map's gives 0.83, with the table R2* 0.56, a fitted complex rate
    0.46. Free in the operator (per-voxel functions in the segments, no change of L); the B0 map must be the
    effective frequency. The coordinator's hypothesis (decay from the map gradient) is **refuted** here (worse in every
    combination).
* **Pass the bar locally, only partly fixable at 2 mm**: **T5, in-readout displacement of blood** (v2: vessels R_loc
  0.64 cycle / 1.11 systole, coronaries 0.49–0.62; image space: coronary velocity bias median 16–22 mm/s, rms
  34–53 % of the coronary speed at 2 mm). The lagged warp in cardinal segments is built and tested (GPU, +2–3 % per set
  measured, additive), handles T6 (acceleration) for free and T7 is negligible, but it removes only **20–30 %** of the
  T5 error at 2 mm on this phantom: the rest is sub-voxel velocity structure (→ the ≤ 1 mm coronary grid). T12 (path
  through the B0 gradient) cannot be handled from the 2 mm map.
* **Below the bar → ignore**: T7 M1⊥ (R_loc ≤ 0.21), T8 tissue motion (myocardium R_loc ≤ 0.16), T9 voxel basis
  (a convention; its R 0.42–0.49 does not survive as an error once x is redefined, except through B0, which is T4).
* **Largest unmodelled term outside the readout**: GIRF (illustrative chain: omission R 33–37; 5 % of the error left
  → 1.9–2.1): the GIRF needs ≈ 0.5 % accuracy of the trajectory error for R < 0.2.

## 1. Scale and method

**Metric.** R = ‖(A − A_model) x‖ / (σ √(M C)), the data error rms over the noise rms per sample and whitened coil
(σ = 39.98, 32 true coil maps whitened with the sim's Ψ), the same scale as `research/segment_basis.md`. R = 0.2 adds
4 % to the residual energy. Every number is an exact direct sum (GPU, f32 values, f64 phases) at the real sample
positions of 48 readouts spread over the 10-min scan (their own rotations and cardiac phases; 16–96 in some
experiments, stated there), on ~450–650 ADC samples per readout (all samples with |k| < 150 rad/m, every 8th elsewhere, trapezoid weights). The R split over |k| bands
(< 0.1, 0.1–0.5, 0.5–0.9, > 0.9 kmax) adds in R² and is in `tables.md`.

**Local scale.** For an error that comes from a region V only (vessels, coronaries, myocardium),
R_loc = R_V √(N_obj / N_V), N_obj = 2 364 426 support voxels, N_V = the 2 mm voxels of V: the global R the whole object
would have if every region of that size behaved like V. A per-voxel bias over the per-voxel noise of an unregularised
least-squares fit would be larger still by √(MC / N_obj) for an error in the range of A (not used for decisions).
Regions: blood sources of the Eulerian frame by their 1 mm truth label (chambers; vessels = artery / vein / pulmonary;
coronary = within r + 1 mm of a coronary centreline point of that phase: RCA, LAD, LCx, branches, cardiac veins,
coronary sinus; ≈ 1 500 voxels at 2 mm); myocardium = 2 mm majority label 18 at frame 0. Phases: systole = truth frame 3
(blood frame 15, aortic p95 speed 1.53 m/s), diastole = frame 12 (blood frame 60, E wave / coronary diastolic flow);
"cycle" = each readout at its own cardiac frame.

**The image.** The warm start of the 10-min run (`fix_2mm/llr_s/warm_full.npy`) is unusable as "the realistic image":
its model data have 3.7–4.4× the measured energy above 0.3 kmax (`calib.json`; noise and streaks of a 10-iteration
reconstruction). Since the segment basis R² is > 90 % at high |k|, the CPU-exact R values of `segment_basis.md` are
too high by ~1.4× (water svdn L = 9: 0.150 there, 0.105 here). Instead `truth_image.py` builds the band-limited 2 mm
projection of the simulator's own sources (x = IDFT₂ₘₘ[Σ_s a_s e^{−ik·r_s}] at the 2 mm DFT frequencies; static
tissue at the reference pose, PD × spoiled steady state at the design's TR, rect slab; one Eulerian blood frame,
steady state at the blood T1, inflow not modelled [CHECK]) and the per-voxel effective signal of its tissue sources
(H(v, τ) = Σ_s a_s e^{−τ/T2*_s} e^{−i(ω_s − ω̄_v)τ} / Σ_s a_s, water and fat separately). Against the measured
noise-free data of the simulated current scan, **without any fitted scale**: energy ratio model / measured per |k| band
1.02 / 0.95 / 1.00 / 1.04 / 0.95 (out leg, 0–0.02–0.1–0.3–0.6–1 kmax) and 0.95 / 0.94 / 1.00 / 0.94 / 0.99 (return
leg); the fitted complex scale is 0.97 e^{−0.09i} (`calib_truth.json`, `figs/calib_truth.png`). The per-sample complex
residual is 22–30 % at low |k| (the static image ignores motion, the 7 encodes and inflow) and > 100 % at high |k|
(moving fine structure), so the image is right in energy distribution, not sample by sample. v2 uses the same image
with its own steady state (TR 14.1 ms: water +14 %, fat +9 %, blood +9 % in amplitude).

**Designs.** Both readouts are synthesised from the seqdesign waveform (tr_grad, readout part about the echo, 2 µs) and
rotated with the scan's rotations (Procrustes on sched_nominal.npy). Check against the simulated current scan's
per-sample records: k 5·10⁻⁸ kmax, readout M1 about the echo 1·10⁻⁷ relative, τ 5·10⁻¹⁰ s, concomitant coefficients
of the readout part 1·10⁻⁶ relative (`pb.check_design`). v2 uses the same rotations (paired comparison).

## 2. Phase 1 — the budget

Main numbers (`tables.md` has all; R global; R_loc vessels / coronary at systole and diastole; myocardium for T8).
Cost per set: today's operator ≈ 0.077 s per segment (production scaling of segment_basis.md §6), v2 at L 10 ≈ 0.77 s.

| term | design | omission R | omission R_loc vessels sys / dia; coronary sys / dia | proposed handling: residual | added cost per set | verdict |
|---|---|---|---|---|---|---|
| T1 B0 | current | 59 | (global term) | svdn L 9: 0.105 | baseline (0.69 s) | model (done) |
| | v2 | 24 | | svdn L 10: 0.107 | baseline (0.77 s) | model (done) |
| T2 concomitant | current | 7.5 | | encode mean + ΔC Taylor 2: 0.019 | 0 | model (done) |
| | v2 | 0.54 | | 0.0016 | 0 | model (done) |
| T3 fat | current | 64; one map vs frames 0.65 | | union svdn per frame, L 13: 0.17 | +1 segment vs one map (+0.08 s, +8 %) | model per frame |
| | v2 | 63; one map 0.67 | | union svdn per frame, L 15: 0.11 | +1 segment (+0.08 s, +7 %); −3 vs today's per-frame path | model per frame |
| T4 R2* | current | 5.96 | | table 1.99; complex rate 0.85 | 0 (in the segment functions) | model (effective map + R2*) |
| | v2 | 1.66 | | table 1.40; eff. frequency + table 0.56; complex rate 0.46 | 0 | model; gradient factor: no |
| T5 displacement (blood) | current | 0.33 cycle / 0.60 sys | 3.2 / 0.66; 0.64 / 0.53 | not evaluated | — | — (v2 is the design) |
| | v2 | 0.13 cycle / 0.22 sys | 1.11 / 0.47; 0.62 / 0.58 | warp, Lanczos-3: −20 to −30 % (coronary sys R_loc 0.36 → 0.25 on the model set) | +0.024 (cubic) – 0.030 s (Lanczos-3) measured, idle GPU: +2.3–2.9 % of the benchmark operator, +3–4 % of 0.77 s | model (opt-in, partial) + ≤ 1 mm grid |
| T6 acceleration | current | 0.14 sys | 0.79 / 0.12; 0.06 / 0.05 | — | — | — |
| | v2 | 0.10 sys | 0.53 / 0.12; 0.10 / 0.08 | â in the warp: removed (0.194 → 0.187 = the no-acceleration value) | 0 | model with T5 |
| T7 M1⊥ | both | ≤ 0.042 | ≤ 0.21 / 0.07; ≤ 0.08 / 0.07 | (exact phase changes the warp residual by 0.003) | — | ignore |
| T8 tissue motion | current | 0.052 / 0.067 (sys / dia, cardiac + resp) | myocardium 0.16 / 0.15 | — | — | ignore |
| | v2 | 0.043 / 0.055 | myocardium 0.14 / 0.14 | — | — | ignore |
| T9 voxel basis | current / v2 | 0.42 / 0.49 | | no separate handling (§T9) | — | ignore (convention) |
| T10 GIRF (illustrative) | current / v2 | 37 / 33 | | 5 % left: 2.1 / 1.85 | — | model (planned: played trajectory) |
| T12 path through ∇B0 | current | 0.29 sys | 1.52 / 0.29; 0.21 / 0.15 | — | — | — |
| | v2 | 0.11 sys | 0.51 / 0.19; 0.18 / 0.17 | map-gradient phase in the warp: no gain (0.203 → 0.205) | — | defer (not in the sim data) |
| T11 within-frame dynamics | — | — | | temporal B-spline frames (exists) | — | covered |

**Local view, v2** (static image, every readout at its own frame; R_loc of the errors from each region, 2 mm regions at
the reference pose dilated by 1 voxel except the myocardium; `run_global.py --region`; T5 from the table above, cycle):

| R_loc | blood (279 k voxels) | vessels (129 k) | coronary (9.8 k) | myocardium (32 k) |
|---|---|---|---|---|
| T1 B0, svdn L 10 residual | 0.13 | 0.12 | 0.21 | 0.25 |
| T2 concomitant omitted / residual | 0.30 / 0.000 | 0.35 / 0.001 | 0.21 / 0.000 | 0.16 / 0.000 |
| T3 one time-mean fat map vs per-frame | **1.48** | **1.16** | **1.64** | **1.89** |
| T3 joint svdn (folded map) segment error at L 14 | 0.11 | 0.11 | 0.15 | 0.15 |
| T4 R2* omitted / table / effective complex rate | 0.57 / 0.91 / 0.17 | 0.54 / **1.10** / 0.17 | 0.61 / 0.56 / 0.18 | 0.45 / 0.36 / 0.12 |
| T5 displacement, cycle (blood sources) | 0.52 (all blood) | 0.64 | 0.49 | (T8: 0.14) |

Around the heart the per-frame fat map is the largest local error (epicardial fat moves with the heart; ×1.2–1.9 the
global value), larger than T5 in the coronaries. The table R2* map makes the vessels *worse* than no R2* (1.10 vs 0.54):
mixed blood / tissue voxels get the tissue's table rate [CHECK the table in blood voxels]; the effective complex rate
brings every region to ≈ 0.17. B0 with the svdn rule is at the tolerance locally (0.12–0.25).

### T1 B0 off-resonance

Omission R = 59 (current) / 24 (v2): the reference scale of everything else. With today's recommendation
(`--nseg auto:noise:0.2 --seg-basis svdn --seg-weight energy --b0-absorb 1`; rule ported to Python, the encode-mean
concomitant phase in the segments): rule L = 9 (current, R 0.105; the Rust `--rule-only` also gave 9) and **L = 10 for
v2 (R 0.107)**; v2 at L = 9 is at 0.20. Fixed nodes need 17 (current 0.023) / 17 (v2 0.064); fixed + energy LS 14
(0.040 / 0.091). The degree-1 fit (≈ 44 Hz + [72, 150, 101] Hz/m) narrows the 1–99 % field only from [−207, 361] to
[−249, 309] Hz: the field is shimmed, the rest is local structure (as in segment_basis.md).

### T2 concomitant fields

| | current | v2 |
|---|---|---|
| readout part omitted | 7.5 | 0.54 |
| per-sample deviation from the encode mean omitted (segments keep the mean) | 2.2 | 0.15 |
| today's model (encode-mean segments + ΔC Taylor order 2) | 0.019 | 0.0016 |

Largest per-sample deviation phase over the support 0.81 / 0.94 rad. The centre-out readout samples the
high-energy low |k| before the concomitant phase grows: v2's whole concomitant omission (0.54) is a quarter of the
current design's deviation alone. Today's handling is ample (residual ≪ 0.2); for v2 the per-sample ΔC term could
even be dropped (0.15) but it is free (order 2 ≈ order 0 in cost), so keep it. GNL-induced non-quadratic monomials were
not in this evaluation (GNL off in both truth and model) [CHECK: the `--conc-fit deg4` floor of segnufft §7].

### T3 fat (six-peak)

Omission (fat treated as water) R = 64 / 63: fat is 57 % of the steady-state signal (77 % of its energy). The joint
water + fat svdn basis with one folded map needs L = 12 (current; R 0.24 — the rule's ×1.5 margin was not enough here)
/ 14 (v2; R 0.105). **One map for all cardiac frames is the bigger error**: the fat fraction moves with the heart
(epicardial / pericardial fat), and a single time-mean map against the per-frame truth gives R = 0.65 / 0.67, the
frame-0 map 1.14 / 1.19 (2 mm truth fat-fraction frames). That is 3–6× the rule tolerance, so the per-frame maps
matter. Today's per-frame path (fixed uniform nodes + energy-weighted interpolators) needs L = 18 on v2 for R 0.15
(L 15: 0.45). Fix in §3.3.

### T4 R2* and the effective decay

Truth = the voxel's own tissue sources (1 mm, 8 per 2 mm voxel in the static pose: each with its T2* and its field),
image frozen at the echo, then a per-voxel time factor:

| model of the voxel's signal after the echo | current | v2 |
|---|---|---|
| none (`--r2s off`, every realistic run so far) | 5.96 | 1.66 |
| table R2* map (truth_r2s, PD-weighted mean of 1/T2*) | 1.99 | 1.40 |
| per-voxel effective real R2* (fitted to the sources, noise-weighted) | 1.87 | 1.28 |
| per-voxel effective complex rate z = R2* + iΔω (fitted) | 0.85 | 0.46 |
| signal-weighted source frequency instead of the map (what a data-estimated map gives), no decay | 5.70 | 0.83 |
| signal-weighted frequency + table R2* | 0.91 | 0.56 |
| map-gradient intravoxel factor Π sinc(g_a D τ / 2) alone / with the table | 4.82 / 2.61 | 1.57 / 1.48 |
| the same with the simulator's two-point geometry Π cos(g_a D τ / 4), with the table | 3.16 | 1.64 |
| complex rate refitted on top of the sinc factor | 1.81 | 1.22 |

* The table map barely helps on v2 (1.66 → 1.40) and a fitted real rate not much more (1.28): the error is not in the
  decay rate. The complex rate removes most of it (0.46): the dominant part is a **frequency** offset — the map field
  ω̄ (field_ppm_recon) is not the signal-weighted frequency of the voxel's sources (Δf_eff 1–99 % −131 … +82 Hz on v2,
  −226 … +211 Hz on the current design whose return leg sees it longer). In a real reconstruction the B0 map is
  estimated from the data, i.e. it *is* the effective frequency; part of this term is then absorbed by the map
  definition [CHECK on real data: the map's spatial resolution and its own intravoxel bias]. The effective R2* median
  is 30.7 s⁻¹ vs 34.7 s⁻¹ of the table (energy-weighted mean −19 s⁻¹; mixed voxels: the PD-weighted mean of rates
  over-states the decay of the mixture).
* Representativeness of the simulator [CHECK]: a 2 mm voxel is 8 point sources at ±D/4 per axis (in the static pose),
  each with a constant field. In a linear field this is a cos(g_a D τ / 4) product per axis (it recovers after
  g D τ / 4 = π / 2), whereas a continuous voxel gives sinc(g_a D τ / 2) (monotone decay). The two agree to second order
  (1 − (gDτ)²/32 vs 1 − (gDτ)²/24 per axis) for small arguments and differ completely in strong gradients
  (≈ 500 Hz/cm × 10 ms). On v2, 8.6 % of the signal energy sits in voxels where g D τ_max / 4 > 1 (2 mm map
  gradient), the regime where the two geometries part ways; the phantom's intravoxel field there is not linear either
  (next item).
* **Segments with complex rates.** Folding the table decay into g_v(τ) (as `--r2s truth` does) does not change the
  rule's L (9 / 10 with and without; the effective complex rate also 10 on v2) and lowers R at a given L slightly
  (svdn L 10 on v2: 0.107 → 0.096). The histogram-weighted LS interpolators (fixed nodes, equal voxel weights:
  Sutton–Noll–Fessler) gain more from the decay (v2 L 12: 0.31 → 0.19) — decayed late samples need less fit — but stay
  far behind svdn (0.041 at L 12). So a 2-D (f, R2*) histogram is not needed: the weighted SVD of the actual per-voxel
  functions already is that 2-D weighting. **No double counting**: the voxel weights are |x̄(TE)|² Σ|S|² (image at the
  echo) and the decay is in g; using decay-weighted voxel weights instead gives the same L (9 / 10).

### T5 in-readout displacement, T7 M1⊥, T6 acceleration, T12 motion through the B0 gradient (blood)

Source level: each 1 mm blood source of the Eulerian frame moves with its own velocity; carrier = its own field, T2*
and the readout concomitant phase. Omission errors (`tables.md` has the |k| split and the chambers):

| design | phase | readout M1 (T5+T7) R / R_loc vessels / R_loc coronary | T5 displacement | T7 M1⊥ | T6 acceleration | T12 path ∇B0 |
|---|---|---|---|---|---|---|
| current | cycle | 0.329 / 1.65 / 0.46 | 0.328 / 1.65 / 0.45 | 0.023 / 0.11 / 0.06 | 0.074 / 0.35 / 0.04 | 0.156 / 0.78 / 0.15 |
| current | systole | 0.599 / 3.23 / 0.64 | 0.598 / 3.22 / 0.64 | 0.041 / 0.21 / 0.08 | 0.142 / 0.79 / 0.06 | 0.285 / 1.52 / 0.21 |
| current | diastole | 0.245 / 0.67 / 0.53 | 0.244 / 0.66 / 0.53 | 0.018 / 0.07 / 0.07 | 0.073 / 0.12 / 0.05 | 0.112 / 0.29 / 0.15 |
| v2 | cycle | 0.133 / 0.65 / 0.49 | 0.131 / 0.64 / 0.49 | 0.024 / 0.11 / 0.06 | 0.050 / 0.25 / 0.07 | 0.059 / 0.28 / 0.15 |
| v2 | systole | 0.227 / 1.13 / 0.63 | 0.224 / 1.11 / 0.62 | 0.042 / 0.21 / 0.08 | 0.096 / 0.53 / 0.10 | 0.107 / 0.51 / 0.18 |
| v2 | diastole | 0.109 / 0.47 / 0.59 | 0.108 / 0.47 / 0.58 | 0.018 / 0.07 / 0.07 | 0.044 / 0.12 / 0.08 | 0.042 / 0.19 / 0.17 |

(each cell: R global (blood) / R_loc vessels / R_loc coronaries; chambers in `tables.md`. Blood speed at systole: p50 0.19 m/s, p95 1.54 m/s; coronary p95 0.32 m/s; |a| p95 49 m/s², vessels p99 103 m/s².)

* **The readout M1 term is almost entirely displacement** (T5 = the part parallel to M0): T5 alone 0.131 of the
  0.133 total (v2 cycle). T7 (M1⊥, rotating with the readout) is R ≤ 0.042 globally and R_loc ≤ 0.21 (vessels, systole)
  in both designs — at the tolerance in the great vessels at peak systole only, 0.06–0.08 in the coronaries. **Ignore**
  (its handling would need 3 Taylor channels = +300 % FFTs, or a direct sum).
* **v2 has 2.5× less displacement error than today's readout** (cycle 0.13 vs 0.33 globally): the current design's
  return leg revisits the k-space centre with a large readout M1 (R 0.17 at |k| < 0.1 kmax for current, 0.001 for v2).
  On v2 the error lives above 0.5 kmax (late samples, low SNR per sample), locally still large: vessels 0.65 (cycle) /
  1.11 (systole), coronaries 0.49–0.62 — 3–5× the tolerance, the same size as the one-map fat error.
* T6 (acceleration, Lagrangian a from blood frames ±8.5 ms, not in the simulated data): v2 vessels 0.25 (cycle) /
  0.53 (systole). T12 (½ (∇ω·v) t², the source-field gradient; not in the simulated data, whose sources keep their bin
  field): v2 vessels 0.28 / 0.51. Both pass the bar locally in systole; both ride on the same displacement mechanism
  (§3.1). The phantom's in-blood field gradients are large (median 35 Hz/cm, p95 190 Hz/cm inside blood, p99
  1300 Hz/cm at the pulmonary vessels) [CHECK: the 1 mm susceptibility model near the lungs].
* **Image-space consequence** (`imgbias.py`): ideal fully sampled least-squares reconstruction on the 2 mm grid of a box
  around the coronaries (69 × 68 × 64 voxels; every blood voxel of the box, band-limited 2 mm blood image of the
  frame, its voxel velocity), each DFT frequency in the ball observed with the design's displacement at that |k|
  (current: both visits, density-weighted); 4-point encoding at VENC 3 m/s (linear regime); bias = v̂(displaced data)
  − v̂(static data), so the partial-volume / band-limit part common to both cancels. Voxels with |x| > 20 % of their
  region's p95:

  | | coronary: speed p50 / p95; \|bias\| p50 / p95 / p99; rms bias / rms speed | vessels: same | chambers: same |
  |---|---|---|---|
  | v2 systole | 81 / 267; 22 / 117 / 268; **0.53** | 354 / 1609; 38 / 198 / 370; 0.15 | 205 / 574; 14 / 77 / 180; 0.14 |
  | v2 diastole | 91 / 223; 16 / 65 / 115; **0.34** | 78 / 396; 7 / 52 / 171; 0.27 | 239 / 882; 21 / 98 / 196; 0.14 |
  | current systole | 81 / 267; 11 / 45 / 99; 0.25 | 355 / 1610; 35 / 210 / 377; 0.15 | 205 / 574; 16 / 80 / 281; 0.18 |
  | current diastole | 92 / 223; 8 / 30 / 56; 0.21 | 78 / 396; 5 / 38 / 117; 0.20 | 239 / 882; 27 / 171 / 245; 0.19 |

  (mm/s.) The mean bias along v is ≈ 0 (≤ 2.4 mm/s; ≤ 9 mm/s current): a scatter, not a scale error — fast blood of
  the neighbouring chambers / aortic root, displaced along its own velocity, leaks into the 1–2-voxel coronaries
  (figs/imgbias_*.png: the bias sits on the rims of the blood pools and in the coronaries). **In image space v2 is worse
  than the current design for the coronaries** (rms 0.53 vs 0.25 at systole): its high |k| is sampled late (t1 up to
  9.4 ms vs ≤ 3.9 ms on the current out-leg), while the current design's data-space error (0.33 vs 0.13) sits at low
  |k| on the return leg, where it is a smooth, common-to-all-encodes error that cancels in the velocity. Data-space R
  and image-space velocity bias rank the designs differently here [CHECK in a real reconstruction; 2 mm is too coarse
  for coronaries anyway].

### T8 tissue motion during the readout

Tissue velocity at the reference pose: cardiac v_c = ∂U/∂t of the phantom's cardiac motion field (48 periodic samples,
central differences), respiratory ṙ Σ_k B_k(X) at the 99th percentile of |ṙ| of the scan's breathing trace
(`tissue.py`; truth tissue image, H tables, 48 readouts at the phase):

| | current systole | current diastole | v2 systole | v2 diastole |
|---|---|---|---|---|
| cardiac, all tissue: R | 0.046 | 0.038 | 0.034 | 0.028 |
| cardiac, myocardium: R / R_loc | 0.018 / 0.155 | 0.017 / 0.145 | 0.016 / 0.138 | 0.015 / 0.132 |
| M1⊥ part, myocardium R_loc | 0.007 | 0.006 | 0.007 | 0.006 |
| respiratory at peak ṙ, all / myocardium R_loc | 0.046 / 0.032 | 0.046 / 0.032 | 0.041 / 0.031 | 0.041 / 0.031 |
| cardiac + respiratory, all / myocardium R_loc | 0.052 / 0.157 | 0.067 / 0.151 | 0.043 / 0.137 | 0.055 / 0.141 |

Below the bar everywhere (myocardium R_loc ≤ 0.16 < 0.2; the wall moves at ≲ 0.1 m/s). **Ignore**; if wanted it costs
nothing extra in the warp of §3.1 (the moving set would include the myocardium, ~2× the voxels of the blood set).

### T9 voxel basis (rect vs Dirac)

Rect-voxel weight Π_a sinc(k_a D/2) per sample vs the Dirac model, same image: R = 0.42 / 0.49 (above 0.5 kmax).
"After the image redefinition" (Dirac model with x′ = IDFT[sinc · DFT x], the image a Dirac operator would converge to)
R = 0.82 is larger — not a bug: the redefinition does not commute with the per-voxel time phase. The rect-voxel signal
at sample time τ is sinc(k) F[C(τ) x] with C = e^{−iω̄τ − iw·m}; the redefined Dirac model is F[C(τ) (sinc ∗ x)]; the
commutator [C(τ), sinc ∗] lives wherever ω̄ varies between neighbours and reaches the low-|k|, high-energy samples
(the absorbed residual has R 0.43–0.62 at |k| < 0.1 kmax, the omission 0.02). Check (`t9check.py`, v2, 24 readouts):
with C ≡ 1 the omission is 0.39 and the redefinition absorbs it to **0.084** (interpolation + coil-map commutator);
with C, 0.78 → 0.83.

Interpretation: for a static object without off-resonance the Dirac model on the band-limited image is *exact* for
every sample inside the 2 mm band, whatever the sub-voxel structure (the sampled ball lies inside the band), so the
voxel basis is only a convention for x (partial volume, the prior's training target). What is physical is the
sub-voxel structure of the *time-evolving* magnetisation — intravoxel dephasing, i.e. T4's H, which is in the
simulated data (8 sources with their own fields per voxel) and which the effective complex rate captures to R 0.46.
Verdict: no separate T9 handling; a rect basis would be a free per-sample weight but the evidence here does not say
it fits data better than Dirac [CHECK: a source-level comparison of Dirac / rect operators against the full source sum
with fields would settle it; not run].

### T10 GIRF, gradient nonlinearity, coils at moved positions; T11; T12

* GIRF (the illustrative clinical chain of seqdesign, rotation-0 trajectory error rotated with each readout [CHECK]):
  omission R 37 / 33; leaving 5 % of the trajectory error gives 2.1 / 1.85, so a measured GIRF must be good to
  ≈ 0.5 % of the trajectory error for R < 0.2 (R is ~linear in Δk here). This is the largest unmodelled term after B0
  and fat, and it is planned (played trajectory, HASTY_TRAJ=played); the GIRF QA plan should target that accuracy.
* GNL: not evaluated (the operator has it via the physics grid with `--gnl`).
* Coil sensitivity at the moved position: within a 1 cm displacement |∇S| δ / |S| ≲ 0.1 near the surface coils, on
  the blood only; it comes free with the warp (§3.1), whose scatter uses S at the displaced voxel. Not quantified
  separately.
* T11 (velocity change within a temporal bin): covered by the existing temporal B-spline frames of SegNufft (per-sample
  interpolation between frame images); the warp of §3.1 is per frame and inherits that interpolation.

## 3. Phase 2 — handling

### 3.1 Displacement (T5) with acceleration (T6) and path field (T12): lagged warp in cardinal segments

**Mechanism** (designed in concomitant_segmented_first_order.md, now built and tested). Any segment basis can be put
in **cardinal form** without changing the operator: QDEIM nodes on the interpolator table, T = a(τ_nodes),
a′ = a T⁻¹, D′ = T D (a′D′ = aD to 8·10⁻⁸; v2 svdn L 10: nodes 0, 0.70, 1.62, 2.79, 3.94, 5.12, 6.31, 7.51, 8.62,
9.49 ms after TE, condition 31). Then D′_l ≈ g(·, τ_l) and each segment has a time τ_l, so the moving voxels' segment
image can be pushed to their position at τ_l: u_l = D′_l x + Δu_l, Δu_l = (W_l − I)(D′_l x),
d_l(v) = v̂ t1(τ_l) + â t2(τ_l) (t1, t2 = the readout's M1, M2/2 parallel to M0 at the node). Same L channels and
FFTs; Δu is a sparse image scattered into the segment's fine-grid channel after the pad (adjoint: gathered after the
inverse FFT). The pushed value can carry a per-(segment, voxel) factor: the path field e^{−i½(∇ω·v̂)(τ_l−TE)²} (T12).
The coil map is taken at the displaced voxel (the "coils at moved positions" item comes free).

**Accuracy against the exact source-level truth** (`warp_eval.py`, v2, blood frame of the phase, 1 mm sources with
their own velocity; model = the band-limited 2 mm blood image of the same frame and v̂ = the amplitude-weighted source
velocity per voxel, i.e. a perfect lagged estimate). Two set-ups: (a) sources with their own field and T2*, (b)
"matched": sources carry the field and R2* of their 2 mm voxel, which isolates the motion handling. The static
floor (2 mm model vs 1 mm sources, no motion) is R 0.25 in (a) and 0.165 in (b) for the blood alone (the remainder in
(b) is the band-limit / B0 commutator of §T9); it is common to every row. "Motion error" = R of
[(model − truth) − (model − truth) without motion], global R (R_loc = ×4.1 blood, ×4.3 vessels, ×23 coronaries):

| v2, matched carrier (b) | systole: blood / vessels / coronary | diastole: blood / vessels / coronary |
|---|---|---|
| not modelled (= source-level omission) | 0.221 / 0.184 / 0.0155 | 0.108 / 0.078 / 0.0152 |
| warp, ideal sub-voxel shift, exact sample times | 0.186 / 0.153 / 0.0108 | 0.082 / 0.057 / 0.0105 |
| warp, Lanczos-3 stencil, exact times | 0.179 / 0.147 / 0.0105 | 0.081 / 0.055 / 0.0105 |
| warp, cubic stencil | 0.183 / — / — | 0.086 / — / — |
| warp in the svdn segments (cardinal, L 10), Lanczos-3 | 0.181 / — / — | 0.081 / — / — |
| warp in the segments, cubic (vessels, coronary) | — / 0.150 / 0.0115 | — / 0.058 / 0.0115 |
| + the exact M1⊥ phase | 0.182 / 0.149 / 0.0106 | 0.080 / 0.056 / 0.0104 |
| + sheared voxel (intravoxel gradient of v̂, cos / sinc) | 0.175 / 0.141 / 0.0108 (ideal) | — |
| Lanczos-3 + sheared voxel | 0.172 / 0.139 / 0.0105 | — |
| truth with acceleration: warp without / with â | 0.194 / 0.187 (omission 0.217) | 0.082 / 0.080 (0.103) |
| truth with the path field: warp without / with the map-gradient phase / with the source gradient | 0.203 / 0.205 / 0.197 (omission 0.210) | 0.089 / 0.091 / 0.087 (0.103) |
| everything in the truth, segmented Lanczos-3 warp with â and the map path field | 0.196 (omission 0.206) | 0.087 (0.098) |

and with the unmatched carrier (a), systole, 32 readouts: omission 0.227, ideal 0.226, Lanczos-3 0.219, nearest-voxel
0.36, trilinear 0.24; v̂ biased × 0.8: 0.228; v̂ + 5 cm/s white noise per voxel and component: **0.335** (worse than
not modelling); v̂ smoothed (Gaussian σ = 1 voxel): 0.242.

Readings:
* The mechanism works as designed — segmenting the warp costs nothing in accuracy (segmented = exact-time to
  ≤ 0.002), Lanczos-3 is as good as the ideal sub-voxel shift (the Nyquist-edge loss of the stencil does not matter at
  the SNR of those samples), â removes the T6 part (0.194 → 0.187 ≈ the no-acceleration value), M1⊥ is negligible.
* **But a 2 mm voxel-level warp removes only 20–30 % of the displacement error amplitude** (systole blood 0.221 →
  0.179, vessels 0.184 → 0.147, coronaries 0.0155 → 0.0105 = R_loc 0.36 → 0.25; diastole 0.108 → 0.081). In
  quadrature against the static floor the excess falls by ~40 % (0.225 → 0.138 at systole). The remainder lives above
  0.5 kmax and is the motion of **sub-voxel structure**: 1 mm sources with their own velocities (laminar profiles
  across 2–4 voxels in the coronaries) are displaced by up to 9 mm at kmax; a voxel moved by its mean velocity cannot
  represent that. A linear intravoxel velocity gradient (sheared voxel) recovers only a further 0.007.
* T12 cannot be handled from the 2 mm field map (its gradient in vessels next to the lungs is wrong: p95 660 Hz/cm);
  even the source gradient recovers 0.006 of 0.210. Leave T12 unmodelled; it is not in the simulated data anyway
  [CHECK the phantom's in-vessel field].
* **Sensitivity to v̂ is the practical risk**: white per-voxel noise of 5 cm/s makes the warped operator worse than
  the static one (random high-k phase k·δ ≈ 0.7 rad at kmax). The lagged v̂ must be denoised or the warp tapered
  (shrink v̂ towards 0 where |v̂| is not ≫ its noise) [open; untested].

**Cost** (`seg_warp_gpu::warp_cost_realistic`, RTX 4090 idle, ph0001 grid 254 × 144 × 114, σ 1.25, 31 coils, coil
batch 9, L = 10, 1.32 M samples; |V| = 150 000 moving voxels ≈ the blood of one frame, displacements up to 7 voxels):
forward + adjoint 1.048 s plain → **+24 ms with the cubic stencil (+2.3 %), +30 ms with Lanczos-3 (+2.9 %)**
(minimum of 5 runs each, interleaved; footprint 230 000 / 260 000 voxels). Against the production estimate of
0.77 s per set at L 10 that is +3–4 %, independent of the number of samples, and proportional to the moving set (a
speed threshold or the coronary region alone would cut it ~10×). The correction images are built once per frame and
set and reused by every coil batch; the adjoint gathers over batches and applies the transposed push once. With
temporal frames (`frames > 1`) the cost is per frame image.

**Verdict T5 (+T6)**: model it with the warp, opt-in, because it is nearly free, additive and never hurts with an
accurate v̂ — but expect only a partial gain at 2 mm. The rest of T5 is a resolution problem: the user's plan of
≤ 1 mm grids for the coronaries is also the T5 fix (sub-voxel velocity structure shrinks with the voxel; the
stencil's Nyquist edge moves out) [hypothesis, not tested: needs a finer-source phantom]. T12, T7: ignore.

### 3.2 T7

**Confirmed ignorable.** Omission R_loc ≤ 0.21 (great vessels, peak systole, both designs), ≤ 0.08 in the coronaries,
≤ 0.007 in the myocardium. In the warp experiment (§3.1, v2 systole) adding the exact M1⊥ phase to the ideal warp
changes the remaining error by 0.004 globally (0.186 → 0.182, matched carrier; 0.226 → 0.223 unmatched), 0.003 in the
vessels and 0.0002 in the coronaries:
M1⊥ is small next to what remains of T5. Handling it would need three Taylor channels (+300 % FFT) or a direct sum on
the fast voxels; not worth it. If ever needed, the cheaper route is the sequence: M1⊥ is a property of the base
readout's curvature (0.41 rad/(m/s) at most on v2).

### 3.3 Per-frame fat maps with the noise-weighted SVD basis

The joint per-frame-map operator (`SegNufft::segments_fat` / `fat_maps`) applies the factor (1 − f) + f c(τ_l) per
segment node, so it needs an interpolatory basis; today that is uniform nodes. **Any basis in cardinal form is
interpolatory** (QDEIM, `cardinal_form`, §3.1), so the noise-weighted SVD can serve the per-frame path if its time span
holds both the water functions g_v(t) and the fat functions g_v(t) c(t): a *union* Gram (water rows weighted
w_v (1 − f_min)², fat rows w_v f_max², f over the 20 frames), interpolators by least squares over the same union rows,
then QDEIM nodes (`fat_union.py`; Rust `build_segments_svd_fat_union`). Against the per-frame truth (each readout with
its own frame's fat map), 48 readouts:

| basis for per-frame maps | current L: R | v2 L: R |
|---|---|---|
| union svdn, cardinal (proposed) | 12: 0.37, **13: 0.165** (rule), 14: 0.076 | 13: 0.46, 14: 0.23, **15: 0.114** (rule), 16: 0.059 |
| uniform nodes + union LS interpolators (≈ today's per-frame path) | 14: 0.25, 15: 0.17, 16: 0.12 | 14: 0.52, 15: 0.38, 16: 0.28, 18: 0.15 (run_global) |
| one folded time-mean map + svdn at its rule L (today's cheap path) | 12: 0.24 segments + 0.65 map → ≈ 0.69 | 14: 0.105 segments + 0.67 map → ≈ 0.68 |

So per-frame maps cost **one segment more than one folded map** with the union basis (v2: 15 vs 14), and three fewer
than today's per-frame path at its R ≈ 0.15 (18). Cost per set (≈ 0.077 s per segment): v2 one map 1.08 s →
per-frame union 1.16 s (+0.08 s, +7 %), vs 1.39 s (fixed nodes, L 18). The union rule (Gram tail × √mean w × 1.5 ≤
0.2) picks 13 / 15 and lands at R 0.165 / 0.114. No GPU change: the output (P, a, c_l) plugs into the existing
`segments_fat`. Recon side: `recon_p2np --seg-basis svdn-union` (new value; plain `svdn` keeps its uniform-node
fallback for per-frame maps) uses it, with the union Gram tail as the `--nseg auto:noise` rule. Built, not run.

### 3.4 R2* / effective decay

**The hypothesis (a decay from the field-map gradient, no extra unknowns) is refuted on this phantom.** Against the
sources' own decay, the box-voxel factor Π_a sinc(g_a D τ/2) with g = ∇ω̄ of the 2 mm map makes every model worse
(v2: table 1.37 → 1.48, complex rate 0.46 → 1.22; current: 2.02 → 2.61, 1.02 → 1.81; `r2grad.py`, 48 readouts); the simulator's own two-point geometry
(cos) is worse still (1.64). The 2 mm central-difference gradient (p50 / p90 / p99: 51 / 326 / 1306 Hz/cm) is
dominated by tissue boundaries (lung, air): it spans two tissues and predicts a dephasing the voxel's sources do not have.
A gradient from a smoother map, or only where the map is locally linear, might do better [not tested].

What does close the gap is the **frequency**: replacing the map field by the signal-weighted frequency of the voxel's
sources gives (v2) 0.83 alone and 0.56 with the table R2*, against 1.37 (table) and 0.46 (fitted complex rate); on the
current design, whose return leg sees the decay for 8 ms, the frequency alone is not enough (5.70) but with the table
it is the best model (0.91, vs 1.02 for the fitted complex rate). So:

1. In a real reconstruction the B0 map must be the signal-weighted effective frequency at the readout's own TE range —
   which a map estimated from the data (multi-echo or joint) is, and the simulator's `field_ppm_recon` is not. This is
   not an operator change; it is the field-map estimation target [CHECK: field_ppm_recon's definition, and the map
   estimator's bias in mixed voxels]. Fold the table (or estimated) R2* into the segment functions (`--r2s`): free,
   no change of L (§T4).
2. The remaining 0.46 (fitted complex rate) / 0.56 (table) on v2 is non-exponential intravoxel decay of the 8-source
   voxels, mostly above 0.5 kmax. A per-voxel complex rate z = R2* + iΔω is a free per-voxel function of τ in the
   segments (no cost); estimating it needs the data (R2*/B0 mapping work package) [open].
3. Representativeness [CHECK]: 8 point sources per voxel with piecewise-constant fields is a coarse intravoxel model;
   real tissue is continuous (sinc-type decay in a linear field). The 0.46 floor is therefore a property of the
   phantom as much as of the operator; the B0/R2* mapping study should re-measure it on a finer-source phantom.

## 4. Implementation and tests (rustlib/.agents/seg, additive)

All in `rustlib/.agents/seg` (diff: `rustlib/.agents/phys.patch`); no existing default, signature or output changes
(the existing `segnufft`, `seg_basis_cpu`, `seg_rule_cpu`, `seg_multi`, `seg_b0_direct` tests pass in the agent tree;
hasty-recon and hasty-bench build).

* `crates/hasty-nufft/src/segwarp.rs` (new, exported):
  * `cardinal_form(segph, atab, nseg, nv)`: QDEIM (greedy pivoted Gram–Schmidt on the interpolator table) → nodes,
    T = a(nodes), a′ = a T⁻¹, D′ = T D; the same operator (a′D′ = aD to f32 rounding), interpolatory (a′(τ_m) = δ).
  * `WarpStencil` (trilinear / Keys cubic / Lanczos-3, unit DC gain; `response(θ, f)` = the push's k-response
    relative to the exact shift).
  * `SegWarp` (one frame): moving voxels, node displacements d_l(v) = v̂ t1_l + â t2_l (voxels), footprint,
    optional per-(segment, voxel) factor of the pushed value (`path_field`: −½(∇ω·v̂)(τ_l − TE)², T12);
    `forward_cpu` / `adjoint_cpu` (Δu = (W − I)(D′x) on the footprint and its exact transpose), `macs`.
  * GPU: `warp_build` (thread per (segment, moving voxel), atomics into Δu), `warp_scatter` (after the pad, before the
    FFT: grid[c, l] += deapod · S_c · Δu_l at the footprint), `warp_gather` (after the inverse FFT, accumulated over
    coil batches and evictions), `warp_apply_adj` (thread per moving voxel, no atomics). Δu is built once per frame
    and set (first coil batch) and reused by the other batches.
* `segnufft.rs`: `SegNufft::set_warp(st, Option<&[SegWarp]>)` (one per frame; `None` restores the plain operator),
  `has_warp()`; hooks in the private `load_frame` / `flush_frame` and at the end of `adjoint` (inactive without a
  warp); the multi-image, fat and order-3 paths return an error while a warp is set (not extended).
* `segbasis.rs`: `build_segments_svd_fat_union` (union water + fat noise-weighted SVD → cardinal form → (P, a, c_l) for
  `SegNufft::segments_fat` / `fat_maps`, per-frame fat maps), `fat_union_spectrum` (its Gram eigenvalues, for the rule).
* `hasty-bench/src/bin/recon_p2np.rs`: new value `--seg-basis svdn-union` (= svdn; with per-frame fat maps the union
  basis and its rule instead of the uniform-node fallback; `union_weights`: water w (1 − f_min)², fat w f_max² over the
  frames). Builds; not run (no reconstructions in this round). Not wired: the warp (`set_warp`) into the
  reconstruction loop (needs the lagged v̂ per frame from the iterate, the moving-set threshold and the refresh
  schedule; design in §3.1).

Tests (all pass):
* `hasty-nufft --test seg_warp_cpu` (CPU): `cardinal_form_is_the_same_operator` (L 7: max |a′D′ − aD| 8e-8, cond
  3.3, a′(nodes) = I to 1e-4), `stencil_response` (integer shifts exact, DC gain 1, the measured fall-off: max |ρ − 1|
  at k_aΔ = π/2: trilinear 0.29, cubic 0.12, Lanczos-3 0.022; at 3π/4: 0.62 / 0.45 / 0.24), `warp_adjoint` (all
  stencils, with acceleration and the path field: rel 1e-8–8e-8), `moving_vessel_against_direct_sum` (a 1 m/s
  vessel in an off-resonant object, 9.5 ms centre-out, L = 9 svd in cardinal form, f64 direct sum of moving point
  voxels: error of the static operator 0.040 (segment floor 1e-6); with the warp trilinear 0.66, cubic 0.40,
  Lanczos-3 0.22 of it).
* `hasty-nufft --test seg_warp_gpu` (GPU): `warp_correction_matches_cpu` (GPU forward with warp − without = CPU direct
  sum of the correction images, rel 3.7e-6, with the path field; `set_warp(None)` returns to the plain operator within
  its own run-to-run atomic noise 5e-8), `warp_adjoint_correction_matches_cpu` (GPU adjoint correction vs CPU type-1
  direct sum + transposed push, rel 1.2e-5), `warp_adjoint_gpu` (⟨y, A_w x⟩ vs ⟨A_wᴴ y, x⟩ within the plain operator's
  own f32-atomic envelope, 1 and 3 frames), `warp_moving_vessel_gpu` (the toy on the real operator: static 0.058,
  warped trilinear 0.67, cubic 0.46, Lanczos-3 0.33 of it), `warp_cost_realistic` (ignored; the production-size
  benchmark of §3.1).
* `hasty-nufft --test seg_fat_union_cpu` (CPU): `fat_union_tracks_gram_tail` (interpolatory at the nodes; with per-frame
  maps the error follows the union Gram tail within 2.5× and falls with L: 0.40 / 0.14 / 0.081 / 0.018 at L 8 / 10 /
  12 / 14).

## 5. Open questions

1. **T5 at ≤ 1 mm.** The voxel-level warp leaves 70–80 % of the displacement error at 2 mm because the blood's
   velocity varies inside the voxel. Is the ≤ 1 mm coronary grid (already planned) enough to make the warp effective?
   Testing needs a phantom with blood sources finer than the grid (0.5 mm) — worth it before relying on the warp.
2. **v̂ for the warp.** 5 cm/s of per-voxel noise makes the warped operator worse than none. What v̂ do we trust in
   early iterations (the warm start's 7-point fit? the denoiser's output?), and should the warp be tapered by the
   v̂ uncertainty? Refresh every few iterations (lagged) as designed — untested in a reconstruction.
3. **Image-space ranking of the readouts.** Data-space R prefers v2 (T5 0.13 vs 0.33), the ideal-reconstruction
   velocity bias in the coronaries prefers the current design (rms 0.25 vs 0.53 of the speed at systole). A real
   reconstruction of both (the planned v2 re-simulation) should decide; this round's numbers say the long centre-out
   leg costs coronary velocity accuracy unless T5 is modelled at a fine grid.
4. **Field map definition.** Most of the "R2*" error on this phantom is the map frequency not being the
   signal-weighted voxel frequency (`field_ppm_recon`). Is the B0 estimator of the GIRF / B0 / qfit work package
   going to deliver the effective frequency at the readout's TE range (and in blood, where the static floor of the
   2 mm blood model was R 0.25 with the map field and 0.165 with matched fields)?
5. **Phantom realism** [CHECK]: 8 point sources with constant fields per 2 mm tissue voxel; in-blood field gradients
   p95 190–660 Hz/cm; blood amplitude without inflow; no acceleration and no path field in the simulated data. Each
   affects the T4, T5, T12 numbers.
6. **GIRF accuracy target**: R < 0.2 needs the played trajectory to ≈ 0.5 % of the GIRF error of the illustrative
   chain (evaluated with one rotation's error rotated per readout [CHECK]).
7. Local numbers of the global terms (B0, concomitant, fat, R2*) were computed for v2 only (blood, coronary,
   myocardium, vessels; `global_v2_<region>.json`, §2 "Local view"); the current design's were not run.
