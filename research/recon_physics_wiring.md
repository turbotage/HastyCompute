# Reconstruction-operator physics for the v2 readout: wiring, validation and a GIRF refit (interrupted 2026-11-11)

Round of 2026-11-11 on factory_arms/ph0001 (2 mm thorax, 3 T), v2 readout (`data/hasty/seq/flow_thorax_v2`, centre-out
9.49 ms reconstruction window + M1-nulled rewinder navigator). Basis: `research/readout_physics_budget.md` (error budget
on the noise-referenced scale R = ‖(A − A_model)x‖/(σ√(MC)); R ≈ 0.2 = the segment-rule tolerance) and
`research/girf_sensitivity.md` (GIRF measurement study). Code (additive): `rustlib/.agents/seg`, diff
`rustlib/.agents/recon2.patch`. Scripts: `research/experiments/recon_physics/`. Arrays, tables, figures:
`data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11/` (README there). No reconstruction was run.

**Status (interrupted 2026-11-11, session ended by the coordinator).** Items 1–4 implemented and tested; item 6
(operator validation) measured except the two L variants; item 5 (GIRF refit) implemented (library + binary + unit
tests) but its GPU experiments (Jacobian check, recovery vs CRB, refit on the exact data, navigator monitor, cost on a
production set) were **not run** — so there is **no adopt / reject verdict yet**. Resume commands in §8.

| item | status | key numbers |
|---|---|---|
| 1 reconstruction window | done, tested | v2: 5 395 ADC samples → window 4 746 (0–9.49 ms), 649 navigator samples kept reachable |
| 2 played trajectory with TR history + `Girf::from_measurement` | done, tested | exact to 2e-15 vs explicit concatenation; 10 vs 1 TR: 0.089 cycles/m rms; model records = simulator records bitwise |
| 3 per-frame fat, union svdn | test added (GPU, direct sum), pass | model error 0.11 / 1.0e-3 / 4e-6 at L 8 / 12 / 16, implementation 1e-4, adjoint 2e-6 |
| 4 complex rate input | `recon_p2np --rate`, tested | fitted rate R 0.42 vs table 1.50 vs none 1.82 (§6) |
| 5 GIRF refit | code + 2 unit tests; experiments not run | CPU tables 3.9 s per 300-readout set (6 threads); GPU cost unmeasured |
| 6 validation | done except L 13 / 17 | operator vs its own model **0.038**; everything on, true chain **0.42**; estimated GIRF **6.5 raw** (per-encode offsets) |

Findings worth acting on:
* The full v2 operator reproduces the direct sum of its own model to R 0.038 (= the predicted union-Gram tail 0.040 at
  L 15): the segments, per-frame fat, complex rates, concomitant and B0-term phase are wired correctly. Against the
  true physics (same trajectory) R 0.42 — the intravoxel (T4) floor of the 8-source phantom, measured directly as 0.42.
* The estimated GIRF (simulated WP2 measurement, 200 ms protocol) leaves R 6.5 at fixed image, 6.45 of it below
  0.1 kmax: per-encode coherent k offsets (0.24 cycles/m of the 0.26 rms trajectory error; the unmeasured cross terms
  and the encode blocks), which per-encode images / the per-encode background polynomial absorb [the absorbed level is
  what `refit --enc-only` measures: not run]. Without any GIRF: 43.7. With only one previous TR: 6.85 (true chain with
  one TR: 1.58 instead of 0.42 — the 10-TR history is necessary).
* Measured CPU cost: the union svdn build takes 240 s per encode (6 threads, 788 k fit voxels, full table); every 9th
  support voxel as fit voxel and an interpolator table every 4th sample gave the same R (0.0377 vs 0.0375) at ≈ 10×
  less CPU — recon_p2np's svdn-union setup would be ≈ 28 min for v2 at its current settings [the recon defaults were
  not changed].


## 1. Reconstruction window (item 1)

The simulator plays v2's whole ADC (5 395 samples at 2 µs = 10.79 ms, the rewinder included). seqdesign's
`best.recon` = [[0, 949]] (10 µs raster) marks the reconstruction samples.

* Simulator: `FlowSchedule` (hasty-bench `seqsched.rs`) reads `best.recon` (default the whole window) and writes
  `sequence.recon_raster` and `sequence.recon_adc` into the scan meta: half-open ADC-index ranges of the samples with
  n·dwell ∈ [a·dt, b·dt] (closed interval, 1e-9 of slack). v2 at 2 µs: [[0, 4746]] (t = 0 … 9.49 ms), the remaining
  649 samples (9.492–10.788 ms) are navigator. Older designs: the whole ADC (identical behaviour).
* Loader: `SimData` (hasty-recon `data.rs`, `kload.rs`) presents readouts of `na` = the window's samples: `na_file`
  (the files' `n_adc`), `adc_keep` (kept file indices), every accessor (`sample_set`, `load_set`, `readout_raw`, the
  energy profiles) uses the window, so recon_p2np and the k-space store see shorter readouts with no other change (the
  store key already contains the set sizes, so stores with and without the window never mix). The navigator stays
  reachable: `adc_navigator()`, `readout_records_full()`, `readout_raw_full()`. The counter-based noise of the lazy
  loader keeps its file index (the same noise with or without a window). `HASTY_RECON_WINDOW=all` disables the window.
* Test: `hasty-recon data::tests::schedule_based_flow_sim_loads` (extended: window [0, 2) of 3 samples; sample sets,
  raw / full raw data, navigator indices, the streamed loader against the in-memory one, malformed ranges rejected) —
  pass; `hasty-bench seqsched::tests::flow_schedule_history_and_recon_window` (ranges from raster windows) and
  `model_records_match_the_schedule` (v2: n_adc 5 395, window [[0, 4746]]) — pass.

## 2. Played trajectory with the TR history (item 2)

**`PlayedStream`** (hasty-seq `girf.rs`, new section at the end of the file). The chain is LTI, so the played gradient
of TR i is the sum of the chain's responses to every earlier TR. A flow TR of type e with readout rotation R is
F_e + R·B: the rotation-independent part F_e (encode, flow-compensated tail, spoiler on physical axes, readout samples
zeroed) and the base readout B placed at i_ro; chain(R·B) = Σ_b Σ_c R_bc chain(B_c on input axis b). The responses of
the 7 F_e and of the 9 (b, c) pairs are computed once over hist + 1 TRs (`Girf::apply_batch`: the same arithmetic as
`Girf::apply`, every FFT-path transfer function evaluated once); a readout then costs (hist + 1)·n_tr·30 multiply-adds
plus the moment integration (`played_from`, the same integration as `played_moments`).
* Exactness: `girf::tests_stream::stream_matches_the_concatenated_waveform` — against `played_moments` on the
  explicitly concatenated history (1 and 3 previous TRs, a chain with every response type: time-domain parametric,
  high-pass cross / B0 terms, FFT resonances, a table): M0 within 1.4e-20 T·s/m (relative 2e-15), B0 phase 7e-16 rad,
  P 3e-22. `apply_batch_equals_apply` (bitwise), `ideal_stream_is_the_nominal_trajectory` — pass.
* Simulator: `FlowSchedule::with_history(N)` / `simphantom --girf-hist N` (default 0 = the original one-TR path,
  byte-identical); `with_history(1)` reproduces the original path exactly (f32 records identical). 10 vs 1 TR on the toy
  flow TR of the unit test: up to 5.4 rad/m of k. On the 48 budget readouts of v2 with the girf.rs `resonance` chain:
  10 vs 1 previous TR differ by 0.089 cycles/m rms (girf_sensitivity.md: 0.08). Through the chain the trajectory is
  off the nominal one by 2.6 cycles/m at the echo (median), 8.0 cycles/m rms over the window, B0-term phase 0.10 rad rms
  (study: 1.0–2.6 / 8–14 / 0.03–0.10).
* Cost: 1.1–1.4 s to build the stream of a chain (v2, 10 TRs, 6 threads), ≈ 1 ms per readout for its records.
* Measured / estimated chains: **`Girf::from_measurement(path)`** loads a `hasty-girf-1` file (§2.1);
  `Girf::to_measurement` writes one; `GirfMeasurement::load` returns the metadata (resonance list, note);
  `simphantom --girf <file.json>` accepts it. Test `measurement_file_roundtrip` (inline arrays and `.npy` arrays,
  Hermitian extension, bad files rejected; the table chain plays within 2e-3 of the parametric one) — pass.
* Reconstruction: **`HASTY_TRAJ=girf:<girf.json|variant>`** recomputes every readout's records through the model
  chain on the fly (hasty-recon `played.rs`: the seqdesign run from `meta.sequence.run`, encode and rotation per readout
  from readouts.npy, `HASTY_GIRF_HIST` previous TRs, default 10): played k, the B0-term phase (removed from the data on
  load, as for `played`), the concomitant coefficients from the played ∫GGᵀ, γM1 / ½γM2; nothing is written to disk.
  Test `seqsched::tests::model_records_match_the_schedule`: on the real v2 design the reconstruction's model records
  equal the simulator's `FlowSchedule::sample_params` (same chain, 10 TRs) bitwise. `HASTY_TRAJ=nominal|played` are
  unchanged.

### 2.1 The `hasty-girf-1` file

One JSON file; arrays inline or as `.npy` files next to it (relative names). Proposed as the common format of WP2's GIRF
estimator and the reconstruction [the WP2 / seq agent may extend it; keep it backwards compatible]:

```json
{
  "format": "hasty-girf-1",
  "convention": "continuous",            // H(f) of the continuous waveform; the raster alias sum is applied on use
  "f": [0.0, 5.0, ...],                  // Hz, ascending, >= 0 (or "f.npy", float64)
  "self": [T, T, T],                     // per physical axis x, y, z
  "cross": [{"out": 0, "in": 2, "h": T}], // axis `in`'s input appearing on axis `out` (optional)
  "b0": [T | null, T | null, T | null],  // B0 term per input axis, T per T/m (optional)
  "tail": 0.05,                          // s of response kept after the input (>= the slowest eddy term x 5)
  "resonances": [{"axis": 0, "f0": 1400.0, "zeta": 0.03, "gain": 0.06}],  // the parametric fit's list (refit seeds)
  "note": "free text: protocol, date, scanner, thermal state"
}
```
T = `{"re": [...], "im": [...]}` or `"h_x.npy"` (complex128 / complex64, or float64 [n, 2]) or `{"f": [...], "h": ...}`
(its own grid). Every term becomes a `Response::Table` (linear interpolation of re / im, Hermitian, held constant beyond
the grid): tabulate to ≳ 40/dt (the alias sum of `at_pl` reaches 4 MHz on the 10 µs raster); the study's hybrid tables
(nonparametric below f_trust, parametric to 5 MHz) do.

Estimated GIRF for the next resimulation's reconstruction (`make_girf_est.py`, one simulated WP2 thin-slice measurement
of the girf.rs `resonance` chain, nominal nuisances): `girf_est_resonance_T200.json` (the recommended 200 ms window, 25
averages) and `…_T80.json` (80 ms, 50 averages). Both find the three resonances to < 0.003 Hz; the group delay at
1–3 kHz is off by −98 / −25 / +88 ns (x / y / z) with 200 ms and 0.2 / 5.3 / −3.0 ns with 80 ms: the 200 ms estimate
trusts the nonparametric table only to 4.4 kHz (80 ms: 12 kHz), as the study noted, and its blend into the parametric
extension costs H errors of ≈ 9·10⁻⁴ at 2–5 kHz on this broad-resonance chain. Neither has cross terms (not measured by
self-term slices). Against the true chain on the 48 readouts: 0.26 cycles/m rms of k, 3.4 / 1.9 mrad of B0 phase.

## 3. Per-frame fat with the union basis (item 3)

`--seg-basis svdn-union` was wired in recon_p2np (per-frame maps → `build_segments_svd_fat_union` → `segments_fat`
→ `forward_fat` / `adjoint_fat` with the set's frame map) but never run. New GPU test
`hasty-nufft --test seg_fat_union_gpu` (`fat_union_per_frame_maps_match_direct_sum`): 24 × 22 × 16 grid, 4 coils,
±200 Hz field, a fat shell with mixed voxels whose fraction moves over 3 frames, 12 centre-out 9.5 ms readouts; the
GPU operator per frame against the f64 direct sum of the exact signal, split into the segment model (f64 emulation
of Σ_l a_l(τ) P_l (1 + f (c_l − 1))) and the implementation:

| L | model error fwd / adj (frames 0–2) | implementation fwd / adj | adjoint identity ‖⟨y,Ax⟩−⟨Aᴴy,x⟩‖/(‖y‖‖Ax‖) |
|---|---|---|---|
| 8 | 0.11–0.16 / 0.09–0.13 | 1.1e-4 / 5.9e-4 | 1.4–2.2e-6 |
| 12 | 1.0–1.4e-3 / 1.1–1.3e-3 | 1.1e-4 / 4.9e-4 | 2.6–4.3e-6 |
| 16 | 4.0–5.0e-6 / 4.1–5.1e-6 | 1.1e-4 / 7.6e-4 | 3.9–4.8e-6 |

The implementation error is the plain operator's (seg_b0_direct: 8.6e-5 / 2.8e-4, f32); the per-frame maps add
nothing. Pass. `seg_fat_union_cpu` unchanged (the same numbers as the budget note: 0.40 / 0.14 / 0.081 / 0.018).
Refactor: `build_segments_svd_fat_union` and `fat_union_spectrum` are now wrappers of `…_g` variants that take a
complex water function g(v, t) (item 4); with g = e^{−iψ} the arithmetic is the same (the CPU test's numbers are
identical).

Not wired before and found here: with per-frame fat maps recon_p2np's union (and uniform-node) paths take the phase ψ
only, i.e. `--r2s` is silently ignored with per-frame fat [the existing behaviour; left unchanged; with the new
`--rate` the union basis carries the decay].

## 4. Complex rate input (item 4)

`recon_p2np --rate <z.npy>`: a per-voxel complex rate z = R2* + iΔω (complex64 [X, Y, Z], x fastest, s⁻¹ and rad/s;
native or reconstruction grid; sampled at the voxels' true positions like the truth maps), the voxel's signal after
the first ADC sample τ0 (the echo for v2) relative to the field's e^{−iωτ}: the segment function becomes
g_v(τ) = e^{−z_v (τ − τ0)} e^{−iω_v τ − i w̄(τ)·m(v)} (R2* into the decay as `--r2s` does; Δω onto the field of
`--field truth|est` with its static part Δω τ0 removed, so the image keeps its meaning at the echo — a first version
added Δω τ and left a static per-voxel phase Δω·TE of up to 1 rad that a free image absorbs but the fixed-image
validation of §6 measured as R 1.6). With per-frame fat and `svdn-union` the union basis is built from that g
(`build_segments_svd_fat_union_g`). Excludes `--r2s`; run
directory suffix `_rate`. Static per voxel: a per-frame rate would need per-frame diagonals beyond the fat factor
[not done; the evidence for needing it is missing — the budget's T4 was evaluated on a static image].
Test: `fat_union_with_complex_rates_matches_direct_sum` (R2* 20–60 s⁻¹, Δω ±2π·40 Hz) — model 0.09–0.12 / 1.0–1.4e-3 /
4–5e-6 at L 8 / 12 / 16, implementation 1e-4 / 5–11e-4, adjoint 2–7e-6: pass.

Rate maps from the simulator's sources (`case.py prepare`; the budget's H tables of the voxel's own tissue sources,
water + blood carrier at R2B = 31 s⁻¹, fat separately, normalised at the echo): `z_fit` (per-voxel complex rate
fitted to the frame-mean carrier, noise-time-weighted Gauss–Newton as run_global.py), `z_wt` (table R2* + the
signal-weighted source frequency: what a data-estimated B0 map gives), `z_tab` (table R2* only). Median R2* 34 s⁻¹,
Δf 1–99 % −110 … +57 Hz (fit) / −96 … +54 Hz (wt); ≥ 1 % of the signal voxels hit the fit's R2* bound (5 000 s⁻¹: strongly
dephased voxels at the lung boundaries; 6.5 % of the voxels have |h(TE)| ≤ 0.2) [CHECK: the 8-source intravoxel model].

## 5. GIRF refit in the reconstruction (item 5)

### 5.1 What is refitted and how

The model chain is the estimated GIRF (a `hasty-girf-1` file) with a small parameter vector θ on top
(hasty-recon `girf_refit.rs`, `ChainModel`):

* self term a: H_a(f; θ) = (1 + s_a) e^{−i2πf δ_a} H_a^est(f) Π_{j on a} R_j(f; f0_j (1 + φ_j), g_j (1 + γ_j)) / R_j(f; f0_j, g_j);
* cross term a ← b: (1 + s_a) e^{−i2πf δ_a} H_ab^est(f) (delay and gain act on the output axis);
* B0 term b: (1 + β) H0_b^est(f).

R_j = 1 + g·bp(f; f0, ζ) is the resonance factor of `Response::Resonant`, the list (f0, ζ, g) comes from the estimate's
parametric fit (the file's `resonances`). θ = (δ_x, δ_y, δ_z, [s_x, s_y, s_z], β, φ_1, γ_1, …): 10 parameters for the
v2 chain (3 delays, B0 amplitude, f0 and height of 3 resonances), 13 with gains. θ = 0 is the estimate.

**Sensitivities ∂k/∂θ, ∂φ0/∂θ per sample.** A delay of output axis a moves its played gradient,
∂k_a/∂δ_a(t) = −γ[G_a(t) − G_a(t_exc)]; a gain scales it, ∂k_a/∂s_a = k_a(t); the B0 amplitude scales the k0 phase,
∂φ0/∂β = φ0(t): all from the readout's played window, no extra work. A resonance parameter needs the played response
of the derivative chain ∂H/∂θ_p (tabulated analytically): by linearity one more `PlayedStream` per parameter, i.e. the
"small base-spoke sensitivity tables per base axis combined with each readout's rotation on the fly" of the brief
(the stream's 9 base-axis responses × R_bc). Records and sensitivities: `girf_refit::sensitivities`.

**Data Jacobian.** The operator's samples are y(n, c) = e^{−iφ0_n} [A x](κ_n), κ = kΔ (rad / voxel). Since the
segment diagonals P_l and the coil maps commute with multiplication by the voxel coordinate r_a (index − n_a/2),
∂_{κ_a} y = A(−i r_a ⊙ x): **three extra forward transforms per set** (kernel `refit_ramp` + the operator). The brief
proposed derivative interpolation weights (no extra FFTs); that needs a fourth kernel variant in the SegNufft tile
engine (three more weight arrays in the shared-memory stage, the ΔC Taylor terms differentiated to one order more)
and was not built here — the ramp form uses only the public operator, is exact for the segmented model, and its cost is
measured below (§5.4) [the fused kernel would remove the FFT part of the three forwards: hypothesis, not built].
Then ∂y/∂θ_p = Σ_a ∂_{κ_a}y · ∂κ_a/∂θ_p − i y ∂φ0/∂θ_p (ignored: θ's effect on M1, on the concomitant
coefficients and on the segments — second order). Per sample, g = (∂_{κ0}y, ∂_{κ1}y, ∂_{κ2}y, −iy) is reduced over
the coils on the GPU (`refit_reduce`: G = Re Σ_c g_a* g_b, h = Re Σ_c g_a* r, 14 floats per sample, r = data − model,
the residual of the gradient step), then on the CPU H = Σ_n Sᵀ G S, b = Σ_n Sᵀ h (S = the 4 × P sensitivity of the
sample): a Levenberg–Marquardt-damped Gauss–Newton step Δθ = (H + λ diag H)⁻¹ b with the image fixed. Fisher
information F = 2H/σ² (complex Gaussian noise of variance σ² per whitened coil sample), CRB = F⁻¹ (image known).

**Per-encode level.** As in girf_sensitivity.md, a coherent per-encode k offset (3) and phase (1) is absorbed by the
per-encode images (exactly a linear / constant image phase per encode); the refit optionally fits those 28 directions
too (`--enc`): the level to judge the remaining error on. Cross terms are not parametrised (the estimate has none; the
study's 12 cross-term directions were not built) — their per-encode coherent part is what `--enc` captures.

**Navigator monitor** (`girf_refit::navigator_fit`, `opval_v2 nav`): the same per-sample G, h summed over each TR's
rewinder navigator samples (v2: 649 samples after the window; here every 2nd) give a per-TR constant k offset (3) and
global phase. It needs an image (the data model), as the study's Fisher numbers did ("image known"); in a
reconstruction the current iterate serves, and segments must cover the navigator time (a separate small segment table
over 9.49–10.79 ms, L 6 here).

### 5.2 Tests run

* `girf_refit::tests::chain_at_zero_is_the_estimate_and_derivatives_match_differences` (θ = 0 is the estimate to 1e-9;
  the analytic resonance derivative chains match central differences to 2e-3) — pass.
* `girf_refit::tests::normal_step_solves_a_linear_problem` (the coil reduction + normal equations recover θ of a
  linear problem to 1e-4) — pass.
* CPU cost of the per-sample tables (`opval_v2 sens`, 300 readouts × 4 745 samples, 10 parameters, 6 threads): 3.9 s
  per set, of which ≈ 1.6 s per derivative stream (7 streams, shared by all sets of one refresh) — the per-readout
  part is ≈ 5 ms. For a 10 % subset of the scan's 42.5 k readouts: ≈ 20 s CPU per refresh, overlappable with the GPU.

### 5.3 Not run (interrupted) — the verdict is open

Planned (script `run_refit.sh`, every step resumable): (1) Jacobian vs central differences of the operator
(`jac_check.json`); (2) mismatched chain (delays +100 / −100 / +100 ns, B0 amplitude +20 %, resonance f0 ±0.5 %):
operator data at θ*, σ = 39.98 noise, 6 seeds → θ̂ − θ* vs CRB (`refit_recovery.json`); (3) the exact data with the
estimated GIRF: R before / after, with and without the per-encode directions (`refit_exact*.json`); (4) navigator
monitor per TR (`nav_est.json`, `nav_ideal.json`); (5) GPU cost of the refit pass vs a gradient pass on a 300-readout
set, 16 and 24 coils (`cost_*.json`). The "≤ 2–5 % of the reconstruction time" estimate is **neither verified nor
refuted**: the refit pass is three extra forwards per refitted set (≈ 1.5 gradient passes [hypothesis, unmeasured]),
so with a fraction f of the sets every N iterations the GPU overhead is ≈ 1.5 f / N (f = 0.1, N = 10: 1.5 %) — to be
measured by step (5).



## 6. Operator validation (item 6)

**Set-up** (`case.py`, `opval_v2`). The budget's image and readouts: the band-limited 2 mm truth image of v2
(water + fat + blood, regenerated by `truth_image_lowmem.py`), 32 whitened true coil maps, the budget's 48 readouts
(pb.scan_readouts(48, offset 7): encode, cardiac frame, rotation composed as pb composes v2 — the records with the
ideal chain reproduce pb's v2 trajectory exactly), their 10 previous TRs (girf_sensitivity's scan rotations), 652
selected window samples per readout (|k| < 150 rad/m all, every 8th else, trapezoid weights) and every 2nd navigator
sample (325).

* **Exact data** (direct sum, f64 phases, GPU; `case.py exact`):
  y_c(n) = e^{−iφ0_n} Σ_v S_c(v) X_f(v, τ_n) e^{−i[k_n·r_v + ω̄_v τ_n + w_n·m(v)]},
  X_f(v, τ) = x(v)[(1 − ff_f(v)) h_w(v, τ) + ff_f(v) c(τ) h_f(v, τ)], with k, the concomitant coefficients w (from the
  played ∫GGᵀ since the excitation), φ0 and τ from the records of the **true chain** (girf.rs `resonance`, 10 previous
  TRs); h_w, h_f = the intravoxel signal of the voxel's own tissue sources (8 per voxel, their T2* and field relative
  to the map field; water with the blood fraction at R2B = 31 s⁻¹), normalised at the echo; ff_f = the truth fat
  fraction of the readout's cardiac frame (2 mm). Static image: no flow / motion terms (T5–T8, T12 of the budget),
  no GNL [CHECK: these remain as in the budget].
* **Operator** (`opval_v2 forward`, what recon_p2np builds with `--model seg --phys full --seg-basis svdn-union
  --seg-weight energy --b0-absorb 1 --rate`, per readout = one set at its frame): SegNufft (σ 1.25, W 10, order 2),
  union svdn segments per encode in cardinal form from the encode-mean concomitant coefficients (energy voxel weights,
  noise time weights from the exact data energy; fit voxels every 3rd support voxel; interpolator table every 4th window
  sample), per-frame fat maps, the fitted complex rate z_fit, the degree-1 field part and the per-sample concomitant
  deviation (φ0, k shift, ΔC order 2) in the sample tables, the model trajectory's B0-term phase; the model trajectory
  through the **estimated** GIRF (T200 file, 10 previous TRs). Ablations: one term changed at a time.
* R = √(Σ_r Σ_n w_n Σ_c |y_op − y_exact|² / (N_r · 4745 · 32 · σ²)), σ = 39.98 (the budget's metric, image fixed).
* Two references isolate the operator itself: `y_exact_modelfit_truth` = the direct sum of the operator's own model
  (x[(1 − f) + f c(τ)] e^{−z(τ − TE)}) on the true-chain records.

**Results** (`case_v2/R.json`, `tables.md`, `figs/ablation.png`). Column "true chain" = the operator with the true
gradient chain (isolates the non-GIRF terms), "estimated" = the reconstruction's realistic model (T200 estimate).
Excess = √(R² − R_ref²) over the true-chain everything-on row.

| variant | R, true chain | excess | R, estimated GIRF | \|k\| < 0.1 kmax (est.) |
|---|---|---|---|---|
| **everything on** (union L 15, fitted rate, concomitant mean + ΔC 2, 10 TRs) | **0.425** | — | **6.55** | 6.45 |
| operator vs the direct sum of its own model | 0.0375 | | | |
| fit voxels every 9th instead of every 3rd | 0.425 | 0.005 | 6.55 | |
| GIRF: nominal trajectory | | | 43.7 | 42.4 |
| GIRF: 1 previous TR | 1.58 | 1.52 | 6.85 | 6.75 |
| GIRF: estimate of the 80 ms protocol | | | 6.58 | 6.48 |
| fat: one time-mean map | 0.766 | 0.64 (budget 0.67) | 6.57 | |
| fat: none | 72.6 | 72.6 | 71.2 | |
| rate: none (map field, no R2*) | 1.82 | 1.77 (budget omission 1.66) | 6.80 | |
| rate: table R2* only | 1.50 | 1.44 (budget 1.40) | 6.72 | |
| rate: table R2* + signal-weighted frequency | 0.828 | 0.71 (budget 0.56 [CHECK carrier]) | 6.59 | |
| concomitant: off (whole history since excitation) | 4.14 | 4.12 (3.998 below 0.1 kmax: the encode lobes' static phase) | 7.59 | |
| concomitant: encode mean only (no per-sample ΔC) | 0.448 | 0.14 (budget 0.15) | 6.55 | |
| segments L 13 / L 17 | not run | | not run | |

Readings: every non-GIRF term behaves as the budget predicted (fat per frame needed, complex rate the lever,
concomitant per-sample terms small but free); the total with a perfect chain is the phantom's intravoxel floor 0.42
(of which the operator's own approximation is 0.04). With the estimated GIRF the raw fixed-image R is 6.5, ≈ all of it
a per-encode coherent low-|k| error: whether per-encode absorption plus the refit brings it to the study's
0.1–0.2 is exactly what the unrun steps (3) of §5.3 answer.



## 7. Implementation, tests, patch

All in `rustlib/.agents/seg` (diff `rustlib/.agents/recon2.patch`, 18 files, `git apply --check` passes against the main
tree of 2026-11-11 13:5x; regenerate with `rustlib/.agents/seg/merge/mkpatch_recon2.sh`); without the new flags / env
variables every existing path is unchanged. **Build status**: the agent tree builds (every hasty-bench binary and test
target, hasty-recon, hasty-seq, hasty-nufft). recon_p2np.rs in the patch is a clean 3-way merge (`git merge-file`, no
conflicts) of my changes onto the main tree's newer file (with `--init-truth`, `--vdecode`, `recursion_limit`); that
merged file was **not compiled** here. hasty-recon `flow.rs` (main-tree change) is not in the patch. The seq agent's
`girfqa.patch` also edits hasty-seq `Cargo.toml` / `lib.rs` and `Cargo.lock`: expect trivial conflicts there (both
add dependencies / modules); my girf.rs changes are an appended section only.

| crate / file | change |
|---|---|
| hasty-seq `girf.rs` | **appended section only** (lines 1–516 unchanged): `use crate::Mat3`; `Girf::apply_batch`; `PlayedStream` (`new`, `n_types`, `played_tr`, `moments`); `played_from`; private `norm`; `ResonanceSpec` (+ `factor`); `GirfMeasurement` (+ `load`); `Girf::from_measurement`, `Girf::to_measurement`; private `npy_1d`; test module `tests_stream` (4 tests) |
| hasty-seq `Cargo.toml` | + `serde_json = "1"` (the measurement file) |
| hasty-bench `seqsched.rs` | `FlowSchedule`: fields `recon`, `hist`, `stream`; `set_recon`, `recon_adc`, `with_history`, `played_stream`, `history`; meta `recon_raster`, `recon_adc`, `girf_history`; `recon_adc_ranges`; tests `flow_schedule_history_and_recon_window`, `model_records_match_the_schedule` |
| hasty-bench `simphantom.rs` | `--girf <file.json>`, `--girf-hist N` |
| hasty-bench `recon_p2np.rs` | `--rate <z.npy>` (+ union basis with decay when given) |
| hasty-bench `opval_v2.rs` (new) | validation / refit / navigator / cost binary |
| hasty-bench `opbench.rs`, `tests/seg_physics_direct.rs` | the three new `SimData` fields in their struct literals |
| hasty-recon `data.rs`, `kload.rs` | reconstruction window (`na_file`, `adc_keep`, `adc_navigator`, `readout_records_full`, `readout_raw_full`, `recon_window`), `HASTY_TRAJ=girf:` (`model_traj`); test extended |
| hasty-recon `played.rs` (new) | `FlowDesign`, `ModelTraj` |
| hasty-recon `girf_refit.rs` (new) | `ChainModel`, `RefitOpts`, `Param`, `sensitivities`, `Reducer` (GPU), `reduce_cpu`, `Normal` (step, CRB, correlation), `to_operator_units`, `navigator_fit`; 2 tests |
| hasty-recon `Cargo.toml`, `lib.rs` | + hasty-seq dependency; `pub mod played, girf_refit` |
| hasty-nufft `segbasis.rs`, `lib.rs` | `build_segments_svd_fat_union_g`, `fat_union_spectrum_g` (the old functions are wrappers, identical arithmetic) |
| hasty-nufft `tests/seg_fat_union_gpu.rs` (new) | 2 GPU tests |

Test runs (all pass): hasty-seq lib (38 + the 4 new), hasty-recon lib (51), hasty-bench lib (8), hasty-nufft
`seg_basis_cpu`, `seg_rule_cpu`, `seg_fat_union_cpu`, `seg_warp_cpu`, `seg_fat_union_gpu`; every hasty-bench binary and
test target builds. Not re-run: the other GPU suites of hasty-nufft (their code is unchanged).

## 8. Status of the jobs and how to resume (interrupted 2026-11-11)

Everything under `data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11/` (README); binaries in
`rustlib/.agents/seg/target/release` (or the main tree after the patch). Set `HASTY_REPO=/home/turbotage/Documents/GitHub/HastyCompute`.

* Done: estimated GIRF files, `truth_x_v2.npz`, the case (`case.py prepare`), records (truth, truth_h1, ideal,
  estT200, estT80), exact sums (`y_exact`, `ynav_exact`, `y_exact_modelfit_truth`), 25 operator variants, `R.json`.
* Unfinished: L 13 / L 17 ablation (stopped during its segment build) and the whole refit / navigator / cost series.
  Resume (each step skips finished outputs; ≈ 2 h with a shared CPU; GPU ≤ 5 GB per step):

      cd research/experiments/recon_physics
      ./run_ablation.sh   > ../../../data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11/logs/ablation.log 2>&1
      ../../../rustlib/.venv/bin/python case.py eval
      ./run_refit.sh      > ../../../data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11/logs/refit.log 2>&1
      ../../../rustlib/.venv/bin/python summarize.py && ../../../rustlib/.venv/bin/python figs.py

* Not started: a 1-s end-to-end simulator check of the new flags, then the ignored test
  `HASTY_TEST_SIM=<out>/scan cargo test -p hasty-recon --lib real_sim_window_and_model_trajectory -- --ignored`
  (writes ≈ 1.3 GB, mostly smaps_true):

      simphantom data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/sim_input.hv <out>/scan --seq flow \
        --flow-run data/hasty/seq/flow_thorax_v2 --girf resonance --girf-hist 10 --flip 15 --scan-time 1.0

**Blocking the v2 resimulation?** Nothing found in the code: `simphantom --seq flow --flow-run
data/hasty/seq/flow_thorax_v2 --girf resonance --girf-hist 10 --flip 15` (+ the realistic run's `--beats …`) writes
the window into the meta and plays every TR with 10 previous TRs (≈ 1 ms per readout of CPU). Not yet run end to end
(above). Disk: with `--girf`, sched_nominal.npy and sched_played.npy are written in full (≈ 14.7 GB each for 10 min of
v2) — the reconstruction no longer needs sched_played.npy for the estimated chain (`HASTY_TRAJ=girf:`), only for
oracle runs [consider an option to skip it].


## 9. Open questions

1. Refit verdict (adopt / opt-in / reject): open until §5.3 runs; the brief's cost estimate is unverified.
2. Per-encode coherent errors of the estimated GIRF (unmeasured cross terms) dominate the fixed-image R (6.5): does
   the reconstruction absorb them (complex per-encode images: yes exactly, but the LLR / learned prior couples the
   encodes; the velocity model: its per-encode degree-1 background polynomial)? Or should the refit get the study's
   cross-term directions?
3. The 200 ms GIRF protocol estimate is worse than the 80 ms one on this broad-resonance chain (group delay ±100 ns vs
   ≤ 5 ns at 1–3 kHz): the study's two-window estimator idea, or the delay refit, has to cover it.
4. svdn-union setup cost in recon_p2np (≈ 4 min per encode at its fit-voxel step 3 and full table): adopt the coarser
   fit-voxel step / table (same R here) as defaults?
5. Per-frame complex rates (moving mixed voxels) — not implemented; no evidence yet that they are needed.
6. With per-frame fat maps recon_p2np ignores `--r2s` (existing behaviour; `--rate` handles it).
7. Encode 3 has no readout among the 48 budget readouts (its segments were built but unused) — a property of the
   budget's readout selection.
8. The recon_p2np changes (`--rate`) must be merged with the main tree's later additions (`--init-truth`, `--vdecode`,
   `#![recursion_limit]`; hasty-recon flow.rs `with_alias_search`) — the patch is against the tree of the brief.


