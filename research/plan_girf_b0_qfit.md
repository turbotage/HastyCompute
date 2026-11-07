# Plan: GIRF-robust TR, GIRF measurement sequence, B0/R2* mapping, qMRI fitting library (2026-11-06)

Context: scans 8–12 min, goal = accuracy at high temporal/spatial resolution; error sources are modelled (B0, R2*,
concomitant, GNL, motion). The reconstruction's concomitant model uses that every readout is a rotation of one base
waveform; the gradient chain (GIRF) breaks that slightly.

## WP1 — TR design that the GIRF changes as little as possible

What actually breaks the rotation structure (linear time-invariant chain, per axis h_a, cross terms h_ab, B0 terms):
- If all three axes had the *same* response h and no cross terms, the played readout would still be an exact
  rotation of the played base readout: h * (R g) = R (h * g). Only axis **differences** (delays differ by ~0.5–1 µs,
  bandwidth, eddy terms) and cross terms break it — and they act most on high-frequency content.
- The encode and spoiler blocks are on physical axes (same for every rotation). Their eddy-current tails leak into the
  following readout as an encode-dependent additive gradient e_e(t); the previous TR's spoiler leaks into the next
  readout. In the concomitant term (∝ G Gᵀ) the cross product R g e_eᵀ is not rotation-structured. Their B0 eddy
  terms give encode-dependent phase offsets (the classic PC background phase).
- So the user's gut feeling is right: the sharp trapezoid/triangle corners of encodes and spoilers are the main
  lever; axis delay differences are the second (and trivially compensated by per-axis timing).

Steps:
1. Baseline GIRF (until measured): per-axis parametric chains with literature-typical values and explicit axis
   differences (delays, first-order bandwidth, 2–3 eddy terms 0.1–10 ms at 0.1–0.5 %, small cross terms, B0 terms),
   plus a "mechanical resonance" term (a damped peak around 1–1.5 kHz) since Loecher 2025 found those matter for PC.
   Several plausible variants, not one, so conclusions do not hinge on one guess.
2. Metrics (`girfcheck` bench) on a designed TR over the PNS rotation set:
   - rotation-structure error of the played readout: max |k_R(t) − R k_base(t)| in Nyquist cells;
   - concomitant-structure error: phase at 15 cm of P_R − R P_base Rᵀ;
   - encode/spoiler leakage into the readout: k error and B0 phase per encode at the echo and over the readout;
   - velocity background phase difference between encode pairs (what PC sees), in mm/s.
3. Mitigations in the designer, each measured with step 2:
   a. smooth ramps (sine / jerk-limited) for encode lobes and the per-encode spoiler block; jerk limit at the readout
      start and at the closing blip; smoother turn-around in the out-and-back;
   b. eddy-current nulling of the encode and spoiler blocks for a band of time constants (GrOpt-style constraint),
      and — once a GIRF exists — the Loecher 2025 approach: choose bipolar/spoiler shapes whose GIRF-predicted
      background phase cancels between encode pairs at the echo;
   c. a short settle gap only if a/b are not enough (costs TE);
   d. per-axis delay compensation now; full GIRF pre-emphasis (Stich 2018) after measurement, so the played readout
      ≈ the rotated nominal one (needs ~10–20 % slew headroom).
4. Feed the residual (after pre-emphasis) to the recon side: if the concomitant-structure error is below a threshold
   (e.g. 0.05 rad at 15 cm) keep the rotation structure; else a first-order correction term per rotation.

### WP1 status (2026-11-06)

- `Girf::variants()` (hasty-seq): six illustrative chains, each adding one effect class — axis delays (2.0 / 2.6 /
  3.1 µs) → + low-pass (7–9 µs) → + eddy currents (0.1–0.3 %, 40 µs – 3 ms) → + cross terms (0.04–0.05 %) → + B0
  eddy terms (1.5–3·10⁻⁵ m, 1–4 ms) → + mechanical resonances (band-pass peaks of 5–8 % at 0.9 / 1.25 / 1.4 kHz,
  ζ 0.03). Response types `Resonant` (H·(1 + g·BP(f)): peak g at f0, no change at DC or high f, as in measured
  GIRFs — an earlier form 1 + g(r − 1) also scaled everything above f0 by 1 − g, which is not what GIRFs show),
  `HighPass`, `Scaled`. The FFT path now filters the *piecewise-linear* waveform (alias sum Σ_m sinc²(f·dt + m)
  H(f + m/dt), `Response::at_pl`); it matches the exact time-domain path to 1e-7 without delay.
  `Girf::perturbed(eps)` / `perturbations(eps)` give models of a chain that are off by eps (all classes at once /
  one class at a time) for "what is left after modelling" studies.
- Metric fix: what phase contrast sees is not the k offset *at* TE but the encode-dependent k error over the readout,
  weighted as a least-squares linear-phase fit weights the samples (`robust::ball_weights`; a 12 cm ball = body-scale
  background: half the weight within 89 µs of the echo; a 2 mm ball = vessel scale: within 4 ms). Low-pass and delay
  transients are gone after ≈ 30 µs, so the old "Δk at TE" overstated them ~100× (121 → 1.4 mm/s).
- `girfcheck` (CPU, 0.2 s; `data/hasty/seq/studies/girfcheck`), on the final designs:

  | chain | flow: rotation-structure k error [cells] | concomitant structure @15 cm [rad] | encode-to-encode weighted Δk [1/m] / Δφ_B0 [mrad] | background velocity @10 cm, body scale [mm/s] | B0 train: rotation k [cells] / Δk at the crossings [cells] / ω bias [Hz] / phase misfit [mrad] |
  |---|---|---|---|---|---|
  | delays | 0.18 | 0.007 | 0 / 0 | 0 | 0.18 / 0.02 / 0 / 0 |
  | + low-pass | 0.48 | 0.020 | 0.007 / 0 | 1.4 | 0.50 / 0.19 / 0 / 0 |
  | + eddy | 0.51 | 0.021 | 0.36 / 0 | 73 | 0.53 / 0.29 / 0 / 0 |
  | + cross | 0.54 | 0.021 | 0.37 / 0 | 73 | 0.54 / 0.29 / 0 / 0 |
  | + B0 eddy | 0.54 | 0.021 | 0.37 / 22 | 80 | 0.54 / 0.29 / 0.01 / 18 |
  | + resonance | 0.68 | 0.018 | 0.32 / 22 | 66 | 0.72 / 0.82 / 0.01 / 18 |

  Reading: (1) the concomitant term stays rotation-structured (≤ 0.03 rad ≪ 0.05 rad) → the reconstruction's
  rotation assumption holds for the Maxwell part; (2) the *trajectory* breaks it at 0.5–0.7 Nyquist cells → the
  reconstruction needs the played (GIRF) trajectory per rotation, or pre-emphasis; (3) the encode blocks' background
  velocity is dominated by eddy currents with τ ≳ 1 ms (and the B0 eddy phase), not by the low-pass; (4) the B0 map is
  insensitive to B0 eddy terms (bias 0.01 Hz), but its lobes' k = 0 crossings move by up to 0.8 cells with
  resonances → model the played trajectory there too.

### WP1 step 3 status: GIRF-robust encodes (2026-11-06)

- `hasty_seq::robust`: every first-order part of the chain leaves after an encode block (ending at T, G = 0) the
  k offset of one functional, E(τ) = ∫ g(s) e^{−(T−s)/τ} ds (low-pass −E e^{−(t−T)/τ}, eddy −aE e^{−(t−T)/τ}, B0 eddy
  γkE e^{−(t−T)/τ}); resonances act through Ĝ(f) near f0 (and dĜ/df once modelled with an f0 error). Designer: per
  axis and encode a QP on the raster samples — exact M0 / M1, |G| ≤ G_max, |ΔG| ≤ S_max·dt, minimise Σ (a_τ γ̄ E_τ)² +
  band terms (|Ĝ|, δf·|dĜ/df| over 0.5–2 kHz, or per-axis weights from a known GIRF, `with_resonances`) + a small jerk
  term; ADMM (dense, OSQP-type splitting), polished to exact moments. `robust::played_offsets` /
  `weighted_offset` = the metric above.
- `encrobust` (CPU, ≈ 2.5 min; `data/hasty/seq/studies/encrobust`, figures `figs/encrobust_*`
  (tools/seq/plot_encrobust.py)): the final TR's encodes (coprime 7, VENC_lo 1 m/s, 1.52 ms two-lobe, TE 1.60 ms)
  against gaps, robust blocks at +0.2…0.8 ms (at the design's encode slew and ×1.25, which the smoother blocks' PNS
  allows) and blocks tuned to the variant chain's resonances as if measured. Background velocity at 10 cm, body
  scale [mm/s], unmodelled | modelled (RSS over single-class 10 % model errors; resonance f0 1 %):

  | design | TE [ms] | encode PNS [%] | eddy chain | full chain (known resonances) | resonance anywhere 0.4–2 kHz: mean / worst |
  |---|---|---|---|---|---|
  | two-lobe (current) | 1.60 | 79 | 73 \| 7.4 | 66 \| 32 | 202 / 715 \| 30 / 96 |
  | + 30 / 60 µs gap | 1.63 / 1.66 | 79 | 69 / 67 \| 7.1 / 6.9 | 71 / 104 \| 29 / 25 | ≈ unchanged |
  | robust +0.2 ms | 1.80 | 65 | 68 \| 6.9 | 81 \| 15 | 134 / 555 \| 21 / 75 |
  | robust +0.5 ms | 2.10 | 60 | 60 \| 6.1 | 25 \| 8.4 | 89 / 433 \| 14 / 58 |
  | robust +0.8 ms | 2.40 | 56 | 53 \| 5.4 | 26 \| 8.8 | 57 / 366 \| 8.9 / 49 |
  | robust ×1.25 slew +0.5 ms | 2.10 | 70 | 60 \| 6.1 | 23 \| 8.4 | 88 / 420 \| 13 / 57 |
  | tuned to the known resonances +0.1 ms | 1.70 | 81 | 71 \| 7.2 | 60 \| 8.9 | (not the goal) |
  | tuned +0.2 ms | 1.80 | 72 | 67 \| 6.8 | 59 \| 7.8 | (not the goal) |

  Vessel scale: 19–23 unmodelled, 2.2–2.7 modelled for every design (long eddy terms over the whole readout).
- Reading:
  1. Long eddy currents (τ ≳ 1 ms) set a floor no shape removes: E(τ) → −M1_T/τ for τ ≫ the block, the classic
     per-encode linear background phase ∝ ΔM1/τ (≈ 70 mm/s body / 20 vessel scale at 10 cm with these uncompensated
     0.2–0.3 % amplitudes; a scanner's own pre-emphasis lowers them). It has to be **modelled**: played encode moments
     from the GIRF in the encoding operator and a per-encode linear background phase (3 coefficients per encode,
     shared over all TRs, fitted jointly / on static tissue — the residual-phase polynomial of the velocity model).
     With a 10 % GIRF error the remainder is ≈ 7 mm/s body, 2.5 mm/s vessel scale.
  2. Resonances are the design lever, but only for **known** frequencies: an encode with this M1 needs 1.5–2.5 ms of
     bipolar, so its energy sits at 0.3–0.7 kHz whatever the shape; no block is robust to a resonance anywhere in
     0.4–2 kHz (worst cases 350–700 mm/s unmodelled, 50–100 modelled, all at the low end). Tuned to the measured
     resonances, +0.1–0.2 ms TE takes the resonance part down to the eddy floor (modelled 32 → 7.8 mm/s); a broad band
     needs +0.8 ms for the same. → after WP2 measures the GIRF, design the encodes against it (`with_resonances`).
  3. A gap before the readout only helps the low-pass, which is already negligible → dropped. Low-pass / delays need
     nothing beyond modelling.
  4. The smoother blocks have PNS headroom (56–72 % vs 79 %), so the TE cost of a tuned block can be partly bought
     back with slew.
- Done (2026-11-07): `TrSpec::encode_shape` (`EncodeShape::{TwoLobe, Robust}`; the flow-compensated tail is re-solved
  around it), `seqdesign --encode robust[:extra_ms[:slew_scale]]` (final build only; the search stays two-lobe).
  Exported for comparison: `data/hasty/seq/flow_thorax_robust` (robust +0.5 ms at the design slew: block 2.02 ms,
  TE 2.10, TR 11.79 ms, PNS 79.8 %, encode part 60.6 %; at ×1.25 slew the readout part exceeded 80 %). Not adopted
  (see reading 2). Played-sequence moments: `girf::played_moments` / `Tr::played_moments` (M0, M1, M2, P and the B0
  phase of one played TR, integrated from its excitation, the previous TR's tails included) — what a GIRF-aware
  simulator and the reconstruction's played trajectory use.
- Full played sequence (`girfcheck`, each encode's TR after its real predecessor; 16 rotations; body | vessel scale,
  mm/s at 10 cm; coherent = the same every rotation, i.e. a background phase; rms = TR-to-TR fluctuation):

  | chain | coherent | coherent, modelled (10 % single-class errors, RSS) | TR-to-TR rms | rms, modelled |
  |---|---|---|---|---|
  | encode block only, eddy (for comparison) | 73 \| 20 | 7.4 \| 2.5 | – | – |
  | eddy | 158 \| 161 | 18 \| 18 | 1.6 \| 1.9 | 0.2 \| 0.3 |
  | eddy + cross + B0 eddy | 173 \| 178 | 19 \| 19 | 1.8 \| 2.1 | 0.2 \| 0.3 |
  | + resonances | 486 \| 396 | 105 \| 48 | 518 \| 382 | 126 \| 82 |
  | eddy, every TR with encode 0's tail (`--same-tail`) | 73 \| 21 | | 1.6 \| 1.9 | |

  Reading: (5) the per-encode flow-compensated tail (decision G, sequence_design.md §9b: it rewinds each encode's M1
  for RF spoiling with flow) leaks its eddy currents into the next TR's readout — coherent, about the encode's own
  size at body scale and 8× it at vessel scale (it lasts through the readout). G stays (without it, flowing spins get
  up to 192 mm/s from spoiling); the leak is deterministic and the same every rotation, i.e. modellable: the played
  sequence with history (`played_moments`) in the encoding operator. (6) Resonances excite the yarn readout itself:
  rotation-dependent trajectory errors of ≈ 0.8 Nyquist cells (rms 518 mm/s equivalent). (7) Sensitivities, linear in
  the model error (largest classes): eddy amplitude 16 mm/s per 10 %, eddy τ 9.6 per 10 %, resonance f0 112 per 1 %,
  resonance height 56 per 10 % → WP2 precision targets below; and the readout design should keep its spectral energy
  away from the measured resonances (a seqdesign objective term once the GIRF exists), since 0.02 % in f0 is hard.
- Done (2026-11-07): `hasty_bench::seqsched::FlowSchedule` — the final yarn TR of a seqdesign run as a simulator
  schedule (readout i = rotation i / 7, encode i % 7; super-Fibonacci rotations sized to the scan, acquisition order;
  per-sample moments through an optional GIRF with the previous TR included; the B0 terms as a constant phase in
  record slot 15, added by the simphantom kernel, module v4). `simphantom --seq flow [--flow-run DIR] [--girf
  ideal|clinical|<variant>] [--dwell 2e-6]`: samples.npy = the nominal trajectory, samples_played.npy
  [readout][adc][8] = played γM0, γM1, τ, B0 phase. `--plan` writes readouts / samples / plan.json without GPU work:
  the dev phantom, 30 s, gives 2653 readouts × 3990 samples × 32 coils × 3.6 M sources ≈ 4.4 min of GPU. With the
  `resonance` chain the played trajectory is 2.5 Nyquist cells (median, max 5.4) from the nominal one — mostly the
  ≈ 10 µs effective delay of delay + low-pass (γ̄|G|·10 µs), the rotation-breaking part is the 0.5–0.7 cells above.
  Test `flow_schedule_moments_and_played_chain` (k = 0 at the echo, design M1 per encode, R·base readout, delay bound,
  B0 phase).
- Done (2026-11-07): hasty-recon `SimData` reads schedule-based sims (`sequence.kind = flow`: readouts.npy, the full
  per-sample records `sched_nominal.npy` / `sched_played.npy` that simphantom now writes, `HASTY_TRAJ=nominal|played`;
  logical k = R_prescᵀ k_phys; the played B0-term phase is removed from the data as a known phase; encode M1 from the
  records at the echo; cardiac times from the regenerated ECG). `acq` became optional (`SimData::spiral()` for the
  spiral-only tools), `ReadoutInfo` carries encode / frame / cardiac time for both. recon_p2np runs on either (CPU
  unit test `schedule_based_flow_sim_loads`; the GPU path is unchanged). Velocity decoding stays hierarchical: ±VENC_hi
  (1.5 m/s) on the coprime-7 data, the CRT unwrapping (±3 m/s) is in hasty-seq `venc` and not yet in hasty-recon.
- Done (2026-11-07): resonance-aware readout search. `robust::band_exposure` = sqrt(Σ_f (w_f γ̄ |Ĝ(f)|)²) of the base
  readout's vector spectrum (rotation-invariant), × FOV in Nyquist cells; with `with_resonances` weights it
  approximates the k error a peak leaves (γ̄·2gζ·|Ĝ(f0)|). `seqdesign --res-girf band|<variant> --res-weight μ`:
  objective N·TR·(1 + μ·exposure) (the cache fingerprint changes only with the option). Study
  (`data/hasty/seq/studies/resaware`, tuned to the `resonance` variant as if measured, evaluated with `girfcheck` on the
  same chain; ≈ 25 min CPU per search):

  | design | N | Nyquist-complete scan | PNS | exposure [cells] | resonance chain: TR-to-TR rms body \| vessel [mm/s] | modelled (10 %, f0 1 %) | coherent | res_f0 sensitivity [mm/s per 1 %] |
  |---|---|---|---|---|---|---|---|---|
  | current (μ = 0) | 1104 | 87.2 s | 79.8 % | 0.332 | 518 \| 382 | 126 \| 82 | 486 \| 395 | 112 |
  | μ = 1 | 1195 | 94.5 s (+8 %) | 79.6 % | 0.208 | 214 \| 213 | 40 \| 38 | 510 \| 414 | 59 |
  | μ = 3 | 1237 | 98.0 s (+12 %) | 79.7 % | 0.185 | 228 \| 176 | 43 \| 36 | 509 \| 409 | 64 |

  Reading: (8) the readout term works where it should — the TR-to-TR (rotation-dependent) resonance error drops
  2–2.5× for +8 % scan time; μ = 3 buys nothing more (the exposure is a proxy). The search puts a spectral notch on the
  strongest resonance (z, 0.9 kHz), moves the main lobe to ≈ 620 Hz and energy to ≈ 2 kHz (`figs/resaware_spectra.png`).
  It is tuned to and evaluated on the same chain (inverse crime): it shows what a measured GIRF buys, not robustness
  to unknown f0; a real resonance near the 600–700 Hz main lobe would hit every yarn readout. (9) The coherent
  resonance part (≈ 500 mm/s on this chain) comes from the encode and the flow-compensated tail (physical axes), which
  the readout term cannot touch → after WP2 also the tail block resonance-tuned (`robust` with the spoil M0 and the
  rewind M1 as its constraints, E/band terms at the next echo), with the encode tuned as in step 3.
- Next (GPU, when allowed): simulate played (`simphantom --seq flow --girf eddy|resonance`, 30 s dev ≈ 4.4 min),
  reconstruct with `HASTY_TRAJ=nominal` vs `played`, compare the background velocity in static tissue with `girfcheck`'s
  prediction. The encode-robust treatment of the tail block is of limited use (long τ dominate after the 1.6 ms to the
  next echo).

## WP2 — GIRF measurement sequence (QA, ~1 h, power/temperature dependent)

Method (no field camera assumed): phantom thin-slice method (Duyn 1998) as extended by Rahmer 2019 / Kronthaler 2021:
- spherical phantom; per axis 4 thin slices at ±8.75 / ±26.25 mm (phase-encoded into voxels for cross and 2nd-order
  terms à la Rahmer 2019 in a later stage);
- test inputs: slew-limited chirps (0.1 → ~20 kHz) plus triangles of several widths (Mazurkewitz/Rahmer combination),
  both polarities (difference cancels B0 drift and non-gradient phase), long ADC (~80 ms) to see slow eddy currents,
  many averages;
- self terms, B0 terms, cross terms (test on axis b, slices along a);
- estimation: per time point fit phase vs slice position → k_a(t), B0(t); regularised least-squares deconvolution
  over all inputs → H_aa(f), H_ab(f), H_0a(f).
Power / temperature dependence (Stich 2020, Nussbaum 2022: self terms ~linear in coil temperature, B0 cross term not):
- interleave GIRF frames (a few seconds each) with heating blocks: high-RMS, low-frequency sinusoids per axis (no PNS,
  within the scanner's duty-cycle supervision), stepping through power levels, then a cool-down — ~1 h total;
- covariates: coil temperature sensors if the site can export them (best, per Stich/Nussbaum); otherwise the
  dissipated-power history (Σ G_rms² per axis) through a fitted thermal low-pass model;
- model: H(f; θ) = H₀(f) + Σ_k θ_k H_k(f) (linear thermal expansion).
Precision targets (from WP1 `girfcheck`, full played sequence, background velocity at 10 cm ≈ 2 mm/s per class):
eddy amplitudes ≈ 1 %, eddy time constants ≈ 2 %, resonance height ≈ 0.4 % and f0 ≈ 0.02 % (≈ 0.2 Hz) — the last two
are hard, which is why the readout should also avoid the measured resonances; low-pass and delays are uncritical. A
per-encode linear background phase fitted in the reconstruction absorbs part of the coherent remainder, not the
TR-to-TR part.
Prerequisites: a Pulseq `.seq` writer (needed for every sequence we make; check by reading back with pypulseq), a raw
data reader for the Siemens files, and a full simulation of the measurement with the baseline GIRF to verify the
estimator recovers H and its power dependence before scanner time.
Safety: heating blocks stay inside the scanner's gradient supervision; protocol reviewed with the site physicist.

## WP3 — B0 / R2* (/ fat) mapping sequence, model-based

**Readout choice (2026-11-06: the user leans to it, I agree; adopted unless revisited): short readouts, not the 4D-flow yarn.** Reasons:
- Errors that are identical in every echo (same lobe waveform, same GIRF / concomitant / eddy phase) only change the
  complex amplitude ρ, not ω: ω comes from the phase *evolution* across echoes. So the "shared errors" benefit of
  reusing the flow readout is small. What does bias ω is evolution *inside* a readout (phase ω·τ, which couples ω
  to the k-space encoding: the voxel model e^{−zt} no longer holds per echo image) and echo-to-echo differences in
  the gradient history. Short lobes keep τ small (fat at kmax: 1.4 rad for 2 mm lobes, vs ≈ 11 rad mid-way through
  an 8 ms readout) and make every interior echo identical.
- A long readout has only 2 k = 0 crossings per TR, so its time diversity at any k comes from TE shifts across TRs.
  CRLB per unit scan time (`echodesign`, below): even when idealised, an 8 ms out-and-back needs 2.6× (ω, R2*) and
  16× (water–fat swap / ω alias margin) more scan time than a K = 16 short train; at 4 ms, 17× and 7×.
- Maps of the physical fields (ω, R2*, fat) are reusable by any later sequence and reconstruction, whatever its own
  readout; each sequence's GIRF / concomitant effects stay in its own operator (consistent with "model the errors").
- Costs: the train is gradient-heavy (bipolar switching near ~1 kHz → PNS-limited slew, heating; check against the
  GIRF's mechanical resonances in WP1, and avoid forbidden lobe periods as in EPI). B0 drifts with gradient heating
  between the B0 scan and the flow scan → a global (low-order) drift term, tracked by the flow scan's own navigator
  ADC (encode window). The flow data can later refine a low-dimensional δω(x) on top of this map.
- Train (`hasty-seq` `multiecho`): after a short hard pulse (TE₁ = 80 µs), K out-and-back lobes, each leaving and
  re-entering k = 0 at rest (G = 0), so the direction changes freely between lobes; echo e = crossing e, owns the
  half lobes on either side (time-symmetric echo); ADC on through the whole train (duty 0.93–0.95); constant physical
  spoiler; slew fitted to PNS (scalar, golden-means directions, SAFE example hw, 80 %).
- Flow: an out-and-back lobe has M1 = −A·T/2 ≠ 0. Lobe pairs (d, −d) along one diameter (a new diameter per pair,
  2D golden means) make every even crossing first-order flow compensated and the train's total M1 = 0 (RF spoiling
  unaffected by flow); odd crossings carry 0.85 rad per m/s (2 mm; 0.43 rad at 3 mm) along d. In the echo images that
  phase varies over TRs → odd-echo attenuation in fast blood (spherical average sinc(γM1|v|): 0.88 at 1 m/s, 0.75 at
  1.5 m/s). Handled by the model (k-space-domain model-based recon with v from the flow scan, or an odd-echo factor),
  or by fitting blood from the even (compensated) echoes only.
- A ½-lobe TE shift on alternate TRs costs ≈ nothing on all echoes and keeps the even-only subset well posed: without
  it the even echoes are 2.16 ms apart ≈ the fat period (2.39 ms at 2.89 T) and water / fat become inseparable
  (swap margin 10³–10⁴× worse); with it, even-only costs 2–3× scan time.
- `echodesign` results (2 mm, PNS-fitted slew 0.42, lobe 1.08 ms; 3 mm: 0.49, 0.82 ms): ω / R2* precision per unit
  time is set by the echo span, K = 32 needs 0.2–0.35× the time of K = 16 for blood / myocardium / liver (no gain for
  iron liver / lung, whose signal is gone by then); FF and the swap margin improve more slowly (0.6–0.8×). Sine lobes
  allow a higher PNS-limited slew but are ~20 % less time efficient at equal K. Choice for now: 2 mm, K = 24–32
  trapezoid pairs, ½-lobe shift on alternate TRs (TR 28–36 ms, ΔTE 1.08 ms, 0.54 ms interleaved).
  Run: `data/hasty/seq/studies/echodesign` (json + figs, `tools/seq/plot_echodesign.py`). The chosen train itself:
  `b0seq data/hasty/seq/b0_multiecho` (figures `tools/seq/plot_final.py`).
- Thorax: respiratory-resolved maps (self-gating from the k = 0 samples), since B0 changes with breathing.
- Reconstruction, two stages:
  1. subspace (low-rank in time, from a dictionary of e^{−z t} curves) CS reconstruction with the same operator
     family, then voxel-wise nonlinear fits with WP4 (GPU) → ρ, R2*, Δf, fat;
  2. later fully model-based: maps fitted directly from k-space with the complex-rate segmentor (z = R2* + iω), the
     WP4 solver used inside the outer iterations; intra-lobe evolution and the odd-echo flow phase are exact there.
- Output feeds the 4D-flow segmentor's rate map.
- Simulation pipeline (2026-11-06): `hasty_mrsim::sequence::Schedule` (what the direct-sum simulator needs: readout
  schedule + per-sample moment records) implemented by the old spiral `Acquisition` and by `hasty_bench::seqsched::
  EchoSchedule` (the train; exact moments at the ADC times via `Grad::moments_at`). `simphantom --seq b0` simulates it
  on a factory phantom, fat sources split into the six-peak spectrum of the fit (`--fat single` for the model-error
  study), writes samples.npy (k, τ). Point-phantom check: k = 0 crossings at the design TEs, R2* recovered (19.8 vs 20).
- The scan's own concomitant field: the lobes' G Gᵀ integral grows through the train like an off-resonance (≈ 6 Hz
  at the FOV edge for 3 mm lobes, ~10 Hz expected at 2 mm). Deterministic from the waveform; `recon_b0` removes it from
  the echo images (mean w over each set's low-k samples) so the map is the field B0 only.
- `recon_b0` (stage 1): whitened + coil-compressed data, 50 echo sets (2 shifts × 25 crossings, nearest crossing),
  P²nP (ZMSHR1) with LLR across echo times, concomitant correction, GPU water–fat fit, maps vs the phantom's field
  (figures `tools/b0/plot_b0recon.py`). First run: ph0001, 30 s static, 2.5 mm (`sim_b0/scan_static_30s`).
- First full-phantom runs (2026-11-06) exposed a simulator artefact: the receive loops' Biot–Savart near field was
  singular (ideal filament, 48 segments, loops placed by the body surface at their centre only, so curved surfaces
  came within mm of the wire): coil energy max/median 1.5e7, the images showed only coil hot spots with radial
  streaks. Fixed in `hasty_phantom::coils` (5 mm Plummer softening, 96 segments, the whole loop footprint clears the
  surface by ≥ 1.2 cm): max/median 238. The reconstruction geometry itself was verified on a 6 s point-phantom scan
  (both points at their exact voxels). recon_b0 also normalises the maps to Σ|S|² = 1 (as recon_p2np) and uses the
  circulant P²nP preconditioner (ZMSHR1 in its metric).
- Undersampling per echo image is the real difficulty: a set gets 2 half-lobes per TR of its TE shift, so a 30 s scan
  gives ~1240 half-spokes per set vs ~80 000 for Nyquist at 2.5 mm / FOV 0.4 m (R ≈ 65; at 5 mm R ≈ 16). Jointly all
  50 sets are near Nyquist, so the temporal prior carries the reconstruction: LLR across echoes now; a subspace
  (low-rank e^{−zt} dictionary after demodulating a coarse B0) or the model-based stage 2 next; B0 maps on a 5 mm grid
  (`--grid-scale 2`, samples beyond its band dropped) are the robust first product.
- Free breathing (user, 2026-11-06): the B0 / R2* scan runs free-breathing, so it yields *dynamic* maps for the flow
  reconstruction. Prior: the change over the breathing cycle is essentially a low-parameter diffeomorphism of the
  tissue (the field and R2* move with it), so the dynamic maps are a reference-pose map set warped by a few motion
  parameters, not independent maps per state. Plan:
  1. self-gating from the FID at the first crossing of every TR (`hasty_recon::selfgate`: band-passed coil channels,
     PCA, skew-fixed sign, 0 = end-expiration); amplitude bins; `recon_b0 --resp-bins N` (one LLR prior over all
     (bin, shift, crossing) images) → maps per bin. Validated against the simulator's r(t) (`resp.npy`).
  2. motion model from the bins: register each bin's magnitude to end-expiration (B-spline FFD), PCA → K = 1–3
     displacement modes S_k; motion x = X + Σ θ_k(t) S_k(X) with θ(t) from the surrogate (r and ṙ for hysteresis).
  3. motion-compensated, model-based maps in the reference pose: ω_t(x) = ω_ref(φ_t⁻¹(x)) + δω_t(x) with δω_t of
     low order (the susceptibility field of the lungs / diaphragm changes with lung volume and is not a pure warp);
     fitted jointly over all TRs (warp inside the operator), so every TR's data informs one map set.
  4. the 4D/5D-flow reconstruction uses ω(x, t) = the reference map warped by its own motion estimate.
  Phantom caveat: the simulator attaches the field to the tissue (Lagrangian), i.e. a pure warp; for step 3's δω add
  a breathing-dependent field (dipole of the χ map recomputed at a few breathing states, interpolated in r).
- Next: free-breathing scan (running), motion model (steps 2–3), exact τ / model-based stage 2, GIRF sensitivity of
  the train (WP1 variants), echo-image weights and noise covariance in the fit.

- Redirect (user, 2026-11-06): per-echo images are the wrong unknowns. Model-based reconstruction of tissue
  properties in a reference pose (ρ_w, ρ_f, R2* at high resolution; B0 as basis coefficients or via χ), motion
  (respiratory + cardiac) as a low-parameter diffeomorphism, all per-sample physics, dynamic preconditioning, 2–3 min
  scans. Design and stages: research/b0_r2s_model_based.md. `recon_b0` (LLR per echo) stays as the baseline and the
  low-resolution initialiser.

## Research goal (user, 2026-11-06)

When the reconstruction pipeline is shown to be good for the thorax, the target is: the full pipeline (B0 / R2* /
fat scan → dynamic maps → 4D/5D-flow reconstruction with all modelled effects) yields images that accurately reflect
flow even in the **coronary arteries and veins**. Implications:
- the phantom already contains them (Z-Anatomy graph: left / right coronary artery, anterior interventricular (LAD),
  circumflex, septal and right inferolateral branches; coronary sinus, great and middle cardiac veins); check their
  radii (≈ 2–4 mm arteries) and flow: coronary arterial flow is diastole-dominant (intramyocardial compression in
  systole). Checked on ph0001 (2026-11-06): LCA r 2.4 mm, mean 2.6 mL/s; LAD 2.3 mm, 2.2 mL/s; RCA 2.2 mm, 2.0 mL/s;
  LCx only 1.3 mm, 0.36 mL/s (physiologically ~⅓ of the LCA); coronary sinus 3.2 mm, −4.4 mL/s. The waveform follows the
  aortic pressure: 48–50 % of the volume passes in systole (first 35 % of the cycle) vs ≈ 25–30 % physiologically →
  phantom fidelity item: an intramyocardial pump (time-varying distal resistance / compliance driven by LV pressure)
  in the network model, and a larger LCx;
- 1–4 mm vessels need ~1 mm effective resolution (or the velocity model's sub-voxel / super-resolution route),
  cardiac *and* respiratory motion compensation (coronaries move 10–20 mm with breathing, 10+ mm per beat),
  accurate B0 and water–fat at the heart surface (epicardial fat surrounds the coronaries: chemical shift and
  partial-volume fat signal), and low-velocity sensitivity (coronary velocities 10–30 cm/s → a low VENC encode);
- evaluation: flow / velocity profiles through the coronary segments vs the phantom truth (moving planes, as for the
  great vessels), per cardiac phase.

## WP4 — `hasty-qfit`: symbolic models + second-order LM, CPU and CUDA

Old pipeline (copy_of_pycompute, read only): symengine diff + CSE → jinja CUDA; per data point residual, Jacobian,
residual-Hessian; full H = JᵀJ + Σ r ∇²r; Marquardt damping diag += max(λ J_j², 1e-4); GMW81 with diagonal pivoting
and scaling → LDLᵀ solve; gain-ratio λ update (accept if ρ > μ, shrink λ if ρ > η, grow otherwise, grow if the
predicted reduction is negative); clamp to bounds; random search.

New crate:
1. `expr` — own small CAS: hash-consed expression DAG (params, per-data constants like TE/b, numeric constants;
   + − × ÷ pow exp log sqrt sin cos tanh abs …), parser from strings, symbolic derivative, simplification + constant
   folding, CSE (from hash-consing), Hessian = derivative of the gradient (upper triangle). Complex models as
   real/imag residual pairs (helper expands e^{(−R2*+iω)t} etc.).
2. Code generation from one DAG: (a) CPU straight-line bytecode VM (no compile step); (b) CUDA C device functions
   compiled at runtime with NVRTC (`hasty-cuda::compile_cubin`), f32 and f64.
3. Objective interface, three ways to provide it: symbolic model; user residual+Jacobian(+residual-Hessian) code;
   user objective giving f, g, H directly (e.g. **VarPro**: separable models with linear coefficients eliminated per
   voxel — built in as an adapter whose basis functions can themselves be symbolic; IVIM as the test case). On CUDA
   the custom path is a user device-function source with a fixed signature, inserted into the solver kernel.
4. Solver: second-order LM, trust-region λ update from the gain ratio (old scheme + Nielsen's continuous update as an
   option), full Hessian or Gauss–Newton, Marquardt scaling; indefinite H handled by a modified Cholesky —
   GMW81 (as before) and Schnabel–Eskow 1999 as the modern alternative, both per-thread for n ≤ ~12, plus a small
   Jacobi-eigen spectral modification for very small n; box bounds by projection (later: active-set); convergence on
   projected gradient / step / Δf; multi-start or grid initialisation for non-convex models (Δf wrapping, IVIM).
5. Batched execution: CPU over voxels with threads; CUDA one problem per thread with the whole LM loop in-kernel,
   SoA data layout, parameters/Hessian in registers.
6. Tests: mono-exponential T2/R2*, IVIM (full and VarPro), complex multi-echo (ρ, R2*, Δf), water–fat; CPU vs CUDA
   agreement; benchmark ~10⁶ voxels.

## Proposed order

1. WP4 core on CPU (expr, derivative/CSE/VM, LM + GMW81/SE99, bounds, VarPro) with tests.
2. WP4 CUDA codegen + kernel, CPU/GPU agreement, benchmark.
3. WP1 baseline GIRF variants + `girfcheck` metrics + smooth/eddy-nulled encode and spoiler blocks; redesign the thorax
   TR and report.
4. Pulseq writer (+ pypulseq round-trip check).
5. WP2 measurement sequence + simulated validation of the estimator (incl. power dependence).
6. WP3 sequence (CRLB design) + subspace recon + WP4 voxel fits; fully model-based later.

## User answers (2026-11-06)

- No scanner access now (a later one is likely GE); no field camera; no coil temperature sensors → GIRF heating model
  driven by gradient power / duty-cycle history. Everything simulated in Rust; Pulseq export deferred.
- Assume reasonable GIRF and PNS coefficients. The end-to-end synthetic pipeline is the evaluation: flow phantom →
  simulator (with the assumed GIRF/PNS, B0, R2*, flow, dynamics) → B0/R2* sequence + reconstruction → maps into the
  4D-flow reconstruction → accuracy against the phantom truth.
- qfit is GPU-first (one problem per thread, whole LM loop in-kernel); the CPU path is a reference for tests.
- For the GIRF measurement, the simulation uses a uniform QA-sphere phantom (what the thin-slice method needs).

## Open questions (answered above)

- Field camera available, or phantom thin-slice method only?
- Pulseq interpreter on the scanner (and version)? Can raw data be exported (twix)?
- Can gradient-coil temperature sensor logs be read out on this system (Stich 2020 used the vendor's 12 sensors)?

## Status

- [x] WP4 step 1–2 (2026-11-06): `rustlib/crates/hasty-qfit`
  - `expr`: hash-consed DAG (automatic CSE), simplifying constructors, parser (`+ − * / ^ **`, exp log sqrt sin cos
    tan tanh sinh cosh atan abs sign pow, `pi`), symbolic derivatives; leaves: params, per-data vars, per-problem
    fixed inputs.
  - `program`: straight-line programs (CPU interpreter = exact reference; CUDA C emission).
  - `model`: outputs (complex = re/im pair), symbolic Jacobian and second derivatives; value / GN / full programs.
  - `chol`: GMW81 (pivoted) and SE99 modified Cholesky; property-tested (P(A+E)Pᵀ = LLᵀ, E ≥ 0 diagonal, SE99 leaves
    PD matrices unmodified) on 1800 random matrices.
  - `lm`: second-order LM, gain-ratio λ (classic or Nielsen), full or GN Hessian, Marquardt scaling, box bounds by
    projection, parameter scaling (default from the bounds; essential in f32).
  - `gpu`: whole LM loop per thread in one NVRTC kernel, f32/f64, custom-objective hook (`GpuFit::with_source`).
  - Tests: parser; derivatives vs finite differences; IVIM and complex multi-echo (ρ, R2*, Δf) noise-free fits on
    CPU (all modes) and GPU; GPU f64 = CPU to ~1e-14 with identical iteration counts; GPU f32 converges 2000/2000.
    GMW81 sends 1/2000 IVIM starts into the swapped D ↔ D* basin (SE99: 0).
  - Benchmark (RTX 4090, f32): 10⁶ IVIM voxels × 13 b-values, full-Hessian LM, 23 ms incl. transfers (GN 14 ms).
- Deferred (user, 2026-11-06 — do not stray into general qMRI): VarPro adapter and IVIM. VarPro was only an example
  of an objective that cannot be generated from the signal equation alone (the `with_source` hook covers that case).
  If IVIM is ever needed, use a topological global method (SHGO, Endres et al. 2018) for its multi-modality rather
  than multi-start LM.
- [x] WP4 now (2026-11-06): only what B0/R2*/fat estimation needs — water–fat multi-peak signal models (simulator sign convention
  e^{−(R2*+iω)t}), profile-likelihood grid initialisation over (ω, R2*) for wraps and water/fat swaps (CPU + GPU),
  multi-start LM, CRLB; scaling audit (damping in the scaled variables; scaled problem has the same MLE; stopping
  rule with a statistical meaning).
- Scaling and statistics (2026-11-06), `lm.rs` module docs:
  - the solver works in u = S⁻¹p: gradient S g, Hessian S H S, damping λ diag(max(S²D, floor)) (Marquardt part
    identical to the unscaled λD; the isotropic floor and the modified-Cholesky shift act in units of each parameter's
    typical size), box [lo/S, hi/S]; fixed points = KKT points of the original problem → same MLE for every S.
    Test: the scaled solve and a hand-reparametrised model (u = p/S, unit scale) give identical iteration counts
    and status 2000/2000, |Δp| ~ 3e-14 relative;
  - new statistical stop `tol_stat` (default 1e-4): remaining projected Newton step ≤ tol_stat standard errors
    (δ² = gᵀH⁻¹g ≤ tol² σ̂², σ̂² = 2f/(m − n)); invariant to parameter, data and weight scaling. Default, unit,
    data × 1e4 and data × 1e-6 scalings all land 1.05e-4 SE from a tight reference; iterations 9.9 → 3.3;
  - bounds: projected Newton with an active set (variables held at a bound by an outward gradient are removed from
    the step system). Without it, fits at R2* = 0 stalled to MaxIter (111/4000 at SNR 10); now all converge;
  - statistics fixed by the objective: w = 1/σ², complex data as two residuals, only uniform per-voxel data
    scaling (the water–fat driver normalises by the voxel RMS, then scales amplitudes back).
- B0/R2*/fat (`waterfat.rs`, `stats.rs`):
  - s(t) = (ρ_w + ρ_f c(t)) e^{−(R2*+iω)t}, c(t) = Σ α_p e^{−iω_p t}, same convention as hasty-mrsim/phantom
    (checked against a simulator-style source sum); six-peak fat (Hamilton 2011) or single peak; species models:
    water, water–fat (complex amplitudes), common phase, dual R2*;
  - profile likelihood over a (ω, R2*) grid (amplitudes eliminated exactly, nodes shared by all voxels), best
    3 local minima over ω → multi-start LM → lowest objective; Fisher information / CRLB from the symbolic Jacobian;
  - tests (12 echoes, 3 T): MLE efficient — sample variance / CRLB 0.99–1.01 over 40 000 realisations, no bias
    beyond O(σ²); SNR 50: 100 % right basin; SNR 10: 99.2 % with 3 starts (98.5 % with 1), 0 lost optima — the
    remaining 0.7 % are voxels whose own MLE is the swapped solution (needs the spatial prior in the recon);
  - GPU device-resident (normalise, profile kernel writing starts, LM per start, selection): 1e6 voxels, f32,
    469 nodes, 3 starts: 42 ms on the device, 0.18 s incl. transfers and f64 conversion; f64 GPU = CPU to 1e-13 SE.
- WP3 notes from this work:
  - after a subspace (low-rank) reconstruction the echo images' noise is correlated across echoes (rank K): fit
    the subspace coefficients with their noise covariance (or fully model-based), not per-echo LS on projected
    images, which is not the MLE;
  - the phantom's fat is single-peak −3.4 ppm; switch the simulator to the six-peak spectrum so data and fit agree
    (keep the single-peak case only as a deliberate model-error study);
  - R2* ≥ 0 makes the fit a constrained MLE (pile-up at 0 at low SNR); `r2_min` is configurable;
  - echo design by CRLB (`stats::crlb`), incl. non-uniform spacing to break the 2π/ΔTE periodicity in ω.
