# GIRF QA measurement sequence for GE / pge2: protocol, simulation, estimator (WP2, 2026-11-11, interrupted)

Goal (plan_girf_b0_qfit.md WP2, girf_sensitivity.md §6): a GIRF measurement that runs rarely (once a quarter) and
may therefore be long, and that measures everything the v2 flow reconstruction and the sequence designer need: the
3D GIRF incl. cross and B0 terms, mechanical resonances to sub-Hz in the readout band, their thermal dependence, the
amplitude linearity, the receiver timing per dwell and coil, and a direct check on the played v2 readouts.
Everything below is simulated (no scanner, no field camera, no measured GE GIRF). Code: Rust
`hasty-seq/src/{pulseq,pnsge,girfqa,girfmeas}.rs` + `hasty-bench` bin `girfqa` (patch `rustlib/.agents/girfqa.patch`);
Python `research/experiments/girf_qa/`; outputs `data/hasty/seq/girf_qa/` (README there).

**(interrupted 2026-11-11)** — the session ended during the Monte Carlo. Status per work item:

| item | status |
|---|---|
| (a) Rust Pulseq writer | done: `hasty-seq/src/pulseq.rs` (1.5.0 / 1.4.1, ROTATIONS, labels, signature) + reader; unit tests pass; pypulseq 1.5 round trip of every protocol part clean (§7) |
| (b) generator `girfqa` | done: 11 pass-1 parts (95.8 min) + pass 2 from a target list (example: 4 targets, 13.3 min); PNS ≤ 80 % on SAFE-example and GE-hrmw models for every part; protocol.json, frames, inputs, power timeline, figures |
| (c) simulation + estimators | implemented (`qasim.py`, `qaest.py`, `qamc.py`); debugged on single draws; **the 18-draw Monte Carlo was stopped before any draw completed** (draw 0 reached R) — no MC statistics yet |
| (d) measured-GIRF file + loader | done: `girfmeas.rs` (`MeasuredGirf::load`, `girf(x)`, `resonance_list(x)`, `state_of(power)`, `from_measurement`), Python writer `girfmeas_io.py`; unit test passes; the Python-file cross-check test was not run on a final file |
| note / patch | this note (partial results §5); `rustlib/.agents/girfqa.patch` (`git apply --check` OK against the main tree) |

Results so far (single draws, **not statistics**; §5): receiver-delay differences per dwell recovered to 0.05–3 ns
(truth −4.6 / +9.5 / +28 µs vs 4 µs), body-vs-array offset to < 1 ns, β to ≈ 4 %; with the fixed estimator, draw 0:
R_refined 0.21 (cold, no X stage) → 0.17 (cold + X stage, i.e. cross terms measured) against the cold truth, 0.12 with a
perfect GIRF except cross terms; the same cold GIRF used at the flow scan's steady state (+6.9 °C) 0.79, warm survey
0.43, warm + pass-2 + thermal model 0.36; without the receiver-delay calibration 3.7. The v2-validation block
reproduces the true model error (R_V,meas ≈ R_V,true, noise rms 0.065 rad/m per sample) — it detects GIRF errors at
the scanner. Not yet shown: f0 precision statistics vs 0.2 Hz, thermal-model precision vs 0.5 °C, cross-term
precision statistics.

## 1. GE / pge2 constraints that shape the sequence

Found online today (agent search; each item with its source; [CHECK] = not confirmed for our scanner):

- **Rasters** (HarmonizedMRI design rules, https://github.com/HarmonizedMRI/SequenceExamples-GE, `pge2/docs/
  pge2-sequence-design-rules.md`; `pge2/matlab/+pge2/opts.m`): gradient and block raster 4 µs, RF raster 2 µs, ADC
  dwell a multiple of 2 µs, ADC delay 1 µs raster. RF dead 72 µs / ringdown 54 µs, ADC dead 40 µs. We use 100 / 60 /
  40 µs, and also 40 µs after every ADC (Pulseq / pypulseq convention, `check_timing`).
- **Segments**: a segment is the block range from one `TRID` label to the next; all instances must have the same
  blocks and events, only amplitudes, phases and frequencies change (shapes fixed: use a scaled base waveform); a
  rotation applies to the whole segment; gradients start and end at 0 per segment; **amplitudes must not be exactly
  0** (we use 1e-6 of the shape for "zero" levels, e.g. the zero-input reference and the k = 0 phase-encode step);
  the interpreter **adds 117 µs at the end of every segment instance** (opts.m: 12 + 105 µs).
- **Consequence for the flow scan itself**: the v2 TR has physical-axis encode / tail blocks and a rotated readout;
  pge2 rotates whole segments, so one TR = 3 segments (encode | rotated readout | tail) and +3 × 117 µs = 0.35 ms per
  TR (+2.5 % of 14.1 ms) — unless the encodes are allowed to rotate with the readout. The generator plays exactly this
  segmented TR (heating, v2 validation), so what we measure is what the GE flow scan would play.
- **Safety evaluation**: the interpreter checks only the first 40 000 blocks; compliance beyond is the user's
  responsibility (SequenceExamples-GE README). Our long parts (heating 150 k blocks) repeat one structure. No gradient
  heating / duty-cycle check in pge2's MATLAB code; the scanner's own supervision decides whether the 1.5 × load runs
  [CHECK with the site physicist].
- **PNS**: pge2/TOPPE use GE's IEC 60601-2-33 model (TOPPE `+toppe/pns.m`): per axis
  P(l) = 100 Δt c/SR_min Σ_k SR[k] / (c + (l−k+½)Δt)², SR_min = rheobase/α, RSS over axes, error above 80 % (normal
  mode). Premier coil `hrmw`: chronaxie 642.4 µs, rheobase 17.9 T/s, α 0.310 m (SR_min 57.7 T/m/s). Implemented as
  `hasty_seq::pnsge` (test: direct sum = FFT convolution; constant-slew limit). TOPPE's header lists HRMW at
  70 mT/m [CHECK: the user's spec is 80 mT/m; read the scanner's `Scandbdt.cfg`]. Every part is checked against both
  this model and the SAFE example hardware (≤ 80 % for both; table §2).
- **Format version**: no official statement; indirect evidence (spspiral-cest runs on GE with PyPulseq 1.5.1 + the
  rotation extension "required on GE due to memory constraints") → we write 1.5.0, rotations as the 1.5.1 `ROTATIONS`
  extension [CHECK the exact 1.5.1 layout against MATLAB Pulseq; pypulseq 1.5.0 cannot read it, so every part also
  gets a `_check.seq` with the rotations baked in].
- **Not found** [CHECK at the scanner]: maximum samples per ADC event (GE historically 8–16 k?) → the 200 ms window is
  split into 12 500-sample ADC events (50 ms) with 84 µs gaps (gaps are harmless: the residual phase is unwrapped
  against the model); maximum waveform memory / number of segments; whether the receiver filter delay depends on the
  dwell (measured anyway, t_rx); body-coil receive selection (TOPPE: coil from the normal prescription UI); gradient
  coil temperature access (none found → power-history model, §3.5).
- **Published GE resonances**: none with numbers. Campbell-Washburn et al. (GE MR750, https://pmc.ncbi.nlm.nih.gov/
  articles/PMC5378680/) give a group delay ≈ 8 µs and resonances only in figures; OPTIKS (arXiv 2505.07117, Premier /
  UHP) shows Y-axis acoustic peaks below 1 kHz only in figures. Typical whole-body resonances 0.5–1.2 kHz (from the
  brief; PMC7245110). So the truth chains stay random (girf_sensitivity §1.2) and the protocol finds the peaks itself
  (two passes).

## 2. The protocol

All parts are separate `.seq` files run back to back (the phantom stays; 20 s assumed per changeover [CHECK]). Times on
the as-played clock (incl. the 117 µs segment gaps). PNS: max over one instance of every segment type of the part, SAFE
example hw / GE hrmw. Data: complex64, 2 channels for the body coil [CHECK the channel count], 48 for an array.

| part | measures (brief item) | duration [min] | frames | blocks | coil | data body / 48-ch [GB] | PNS SAFE / GE [%] | mean power / P_flow |
|---|---|---|---|---|---|---|---|---|
| c_cal | 7 (prescan: centre, B0 / shim check, T1, T2, slice positions, reference images, rx FIDs) | 2.0 | 3276 | 24248 | array (images) + body (rx delay repeat) | 0.01 / 0.3 | 65 / 63 | 0.20 |
| t_rx | 6 (receiver delay per dwell 2/4/8/16 µs; run with body coil and array) | 1.4 | 768 | 3072 | body + array (run twice) | 0.03 / 0.7 | 71 / 52 | 0.14 |
| s_cold | 1, 2 (survey: self + B0 terms, resonance list), 3 (cold state) | 12.4 | 3456 | 24192 | body | 2.76 / 66.4 | 78 / 52 | 0.08 |
| x_3d | 1 (cross terms, B0, 2nd/3rd-order spatial terms) | 9.2 | 4800 | 24000 | array | 1.92 / 46.1 | 71 / 52 | 0.15 |
| l_lin | 4 (odd nonlinearity β) | 2.2 | 1152 | 5760 | body | 0.46 / 11.1 | 64 / 52 | 0.14 |
| h1_flow | 3 (heating with the v2 flow TR train) | 12.0 | 0 | 149430 | - | 0.00 / 0.0 | 78 / 77 | 1.00 |
| s_warm | 1, 3 (survey at the flow-scan load) | 11.3 | 768 | 108416 | body | 0.61 / 14.7 | 78 / 77 | 0.99 |
| v_v2 | 5 (played v2 readouts, 48 × 3 axes, 10-TR history) | 6.7 | 2304 | 78336 | body | 0.11 / 2.8 | 79 / 76 | 1.01 |
| h2_hot | 3 (1.5 × load) | 8.0 | 0 | 94637 | - | 0.00 / 0.0 | 79 / 77 | 1.46 |
| s_hot | 3 (survey at 1.5 ×) | 2.8 | 192 | 25520 | body | 0.15 / 3.7 | 78 / 77 | 1.41 |
| d_cool | 3 (cool-down tracking, every 2 min) | 24.0 | 288 | 3396 | body | 0.23 / 5.5 | 68 / 52 | 0.00 |
| **pass 1 total** | | **95.8** (incl. 11 × 20 s changeovers) | 17004 | | | | | |
| h3_flow + r_target (pass 2; example with 4 targets in `protocol_p2_example/`) | 2 (targeted resonances at the flow load) | 8 + 1.15 per target (4 targets: 13.3 incl. changeovers) | 96 per target | 99 621 + 41 184 | body | | 78 / 77 | ≈ 1.0 |

Common frame (one pge2 segment): Hann-sinc RF (TBW 4, 3 ms, 90°) with the slice select on axis a (G_ss 20.9 mT/m for
1.5 mm, ramps at 0.25 S_max), slice position via the RF frequency (per instance), rephaser (+ in-plane PE blips in
x_3d), the test input σ·ℓ·u(t) on axis a with ADC event 1 in the same block, further ADC events, a staggered
low-slew spoiler (4 cycles across the slice on every axis), wait. ADC delay 42 µs so that every sample lies on the
4 µs gradient raster (sample i of the window = corner sample 11 + i of the input; no half-raster ambiguity between
input and output). Ordering: 12 slices (4 per axis at ±8.75 / ±26.25 mm) interleaved over the axes (each slice
recovers 11 frames ≈ 2.4 s); test polarity σ in the pattern + − − + over consecutive frames, shifted by one per
sweep, so the previous frame's tail cancels in the ± difference and each slice gets both polarities twice per 4
sweeps; slice-select polarity η alternates every 4 sweeps (removes the first-order B0 warp of the slices,
girf_sensitivity §5.5).

Inputs (PNS-fitted: the slew fraction is lowered in 7 % steps until both models are ≤ 80 % on every axis; none needed
it except the pass-2 tone burst): log chirp 0.1 → 20 kHz over 40 ms at 0.5 S_max, |k| ≤ 0.8 k_lim; prephased
triangles with 60 / 100 µs ramps at 0.9 S_max and 160 / 240 µs at 0.55 S_max; pass 2: per detected resonance a
linear chirp f0 ± max(6 ζ f0, 15 Hz) over 30 ms (sin² envelope) and a 20 ms tone burst at f0 (`figs/protocol_inputs.png`).

### 2.1 What each part measures (the eight items of the brief)

1. **3D GIRF incl. cross and B0 terms (x_3d + s_*)**: Rahmer-style phase-encoded thin slices. With slices ⟂ the test
   axis a, the test gradient only dephases through the slice (k_lim as before); its cross outputs (gradients on the
   in-plane axes) and 2nd / 3rd-order eddy fields vary *in plane*, which a 5 × 5 PE grid over 200 mm resolves. The
   estimator does not reconstruct voxels: per time sample it fits all 16 real solid-harmonic coefficients (B0, x, y, z,
   five 2nd-order, seven 3rd-order terms; 2nd / 3rd normalised to r0 = 10 cm) to the ± phase differences of all
   (slice, η, PE) units, with design rows A_l = Re(M_l / M) computed from the prescan geometry (receive weighting,
   B0 map, the input's own concomitant phase, the PE weighting). 4 slice positions × 5 × 5 PE identify up to cubic
   terms along a and in plane. Cross / higher-order responses are then deconvolved over the two inputs (chirp,
   tri100) and fitted with two high-pass terms each. These estimates also remove the centroid bias of the B0 term
   (girf_sensitivity §3: in-plane centroid × cross term) and the higher-order contributions from the survey's
   per-slice phases. Array coil (SNR per PE "voxel").
2. **Resonances (two passes)**: pass 1 = the survey (200 ms windows); its parametric fit lists the resonances in
   0.3–2 kHz. Pass 2 (separate file generated from that list by `girfqa --targets res.json --only p2`): after 8 min
   at the flow-scan load, per resonance a narrow chirp and a tone burst on its axis, 200 ms ring-down, 3 groups
   (6 averages per slice and polarity), heating between sweeps; estimator: a time-domain LS fit of (f0, ζ, gain,
   delay) of that resonance on the k increments, with the pass-1 table as background (its own resonance divided out).
3. **Thermal dependence**: cold survey (s_cold), 12 min of flow-scan load (h1_flow: the v2 TR as pge2 segments with
   the scan's super-Fibonacci rotations in acquisition order), survey at the flow load (s_warm, heating between sweeps
   at 1.3 × so that the time-averaged power equals the flow scan's), the v2 validation (≈ the flow load itself),
   1.5 × load (h2_hot: flow TRs + low-slew heater TRs; s_hot), cool-down tracking (d_cool: chirp on the outer slices
   every 2 min for 24 min). Covariate: the protocol's own gradient power Σ_a ∫G_a² in 1-s bins (`power.npy`, exact
   from the .seq). Model: f0_j(t) = F_j + A_j [w x₁(t) + (1−w) x₂(t)], x_k = (P ∗ e^{−t/τ_k}/τ_k)/P_flow (two
   compartments shared by all resonances, per-resonance F_j, A_j; x = 1 is the flow scan's steady state), fitted to
   per-set resonance fits (chirp data of every set) + the pass-2 points. **Temperature-sensor hook**: the frame table
   carries absolute times; a sensor log (fibre-optic probe on the bore liner, or GE service logs if available) can
   replace x(t) in the same fit.
4. **Amplitude linearity (l_lin)**: tri100 / tri240 at 0.25, 0.5, 0.75, 1.0, both polarities; odd nonlinearity
   β g|g|/G_ref from K(ℓ)/ℓ − K(1) = β (ℓ − 1) k₂ (k₂ = the linear chain's response to g|g|/G_ref). Even terms show
   in the ± *sum* (concomitant + any G² eddy field) [not evaluated].
5. **Direct measurement of the v2 readout (v_v2)**: per readout (the 48 budget readouts of girf_sensitivity, encodes
   and rotations of the real scan order) 10 previous TRs as played on GE (3 segments each, all axes), then a 1.0 mm
   slice ⟂ axis a and the measured TR's *axis-a components* (encode + readout incl. the rewinder / navigator), ADC over
   all of it; 3 axes × 4 slices × ± → k_a(t), φ0(t) of every readout. Only axis-a components can be played after the
   excitation: an in-plane readout component (up to 1600 rad/m) dephases the slice cross-section completely (its
   in-plane FT at 250 cycles/m is ~1e-3); the history is played in full 3D (before the excitation it does not
   dephase). So V validates self + B0 terms with the real history, amplitudes and thermal state; the readout's own
   cross terms come from x_3d. 1.0 mm because the encode lobes reach |k| ≈ 3600 rad/m on one axis.
6. **Receiver timing (t_rx, run twice: body coil, array)**: thin slices, tri60 + tri100, ADC dwell 2 / 4 / 8 / 16 µs;
   the delay of k_a(t) per dwell relative to 4 µs (LS on the model's dk/dt) = δ_rx(d) − δ_rx(4 µs), referenced to the
   gradients (vendor-independent). The calibration part also has ±20 kHz off-resonance FIDs per dwell (filter group
   delay if the NCO mixes before the decimation filter — vendor-dependent interpretation [CHECK]).
7. **Phantom-agnostic calibration (c_cal, 2 min)**: noise; projections ±x ±y ±z (centre, radius, receive-weighted
   centroid); 3D dual-echo GRE 5 mm (B0 map → 2nd-order shim check: target ≤ 3 Hz rms over the sphere, re-shim
   otherwise); receiver FIDs (above); T1 inversion recovery (8 TIs); T2 CPMG (32 echoes, 10 ms); slice positions
   (each thin slice read out along its normal, ± readout and ± slice-select polarity: p_s(η) incl. the B0 shift, to
   ≪ 1 µm); thin-slice reference images (64 × 64 over 256 mm, dual echo: ρ, receive weighting, in-plane B0 for the
   x_3d design rows).
   **Phantom requirements / checks**: sphere Ø 15–20 cm (the slices at ±26 mm need ≥ 60 mm radius of signal);
   single spectral line (doped water, or silicone oil at ≈ −4.7 ppm: set the centre frequency on it [CHECK the GE QA
   sphere's filling]); T1 ≤ 1.5 s (each slice recovers 2.4 s; longer T1 → lower flip), T2 ≥ 100 ms and thin-slice T2*
   ≥ 100 ms after shimming (else shorten the window: the 200 ms window was chosen for narrow resonances); no air
   bubbles (projections / B0 map); thermally equilibrated in the bore ≥ 30 min before (the sphere's own drift
   cancels in the ± difference, but changes T1 / T2); centred ±5 mm (the estimator uses the measured centre).
8. **Coil plan**: body-coil receive for everything that is a phase difference of a whole-slice FID (s_*, l_lin,
   v_v2, d_cool, r_target, t_rx): stable, uniform phase, small receive-weighting centroid, 2 channels → small data
   (2.8 GB for s_cold vs 66 GB with 48 channels); SNR₀ ≈ 300 is enough (girf_sensitivity: noise is not the limit).
   Array for x_3d (per-PE-voxel SNR; 46 GB at 48 channels — the largest data item, [CHECK] storage) and for the
   reference images and the B0 map (same coil as x_3d, since the receive weighting enters A_l). t_rx is run with
   both, so the body-vs-array receive delay is known when the GIRF (body) is used for imaging (array).

## 3. Simulation (what is assumed)

`research/experiments/girf_qa/qasim.py` (truth, sessions, signals), `qamc.py` (one Monte Carlo draw = one truth
chain + one session through every part of the protocol, then all estimators and metrics; resumable JSONL).

- **Protocol**: read from the generator's output (frames, inputs, timings incl. the pge2 gaps, power timeline, flow-TR
  segments, v2-validation layout) — the simulation plays exactly what the `.seq` files contain (the pypulseq round
  trip checks the files against the same arrays).
- **Truth chains**: girf_sensitivity `truths.draw` (seeds 5000 + d: the same 48-chain family as that study, paired):
  per axis delay, 1st-order low-pass, 2nd-order roll-off, three eddy terms (incl. 10–60 ms residuals), 2–4 resonances
  0.5–4 kHz with ζ 0.004–0.05 and peaks 1–8 %, smooth ripple; all six cross terms; B0 terms; odd nonlinearity
  β ≤ 3·10⁻⁴. Added here [assumptions, CHECK against a measured 3D GIRF]: 2nd-order eddy fields (five terms per input
  axis, high-pass, |k| ~ N(0, 10⁻⁴) of G at r0 = 10 cm, τ 0.3–3 ms) and 3rd-order (seven, |k| ~ N(0, 3·10⁻⁵)).
- **Thermal truth** [assumption]: θ(t) = three compartments driven by the protocol's own Σ_a G_a² (τ 2–6 min,
  15–40 min, and a slow 70–130 min one the fitted model does not have), steady-state rise 4–10 °C at the flow-scan
  power; coefficients girf_sensitivity's ×U[0.5, 1.5], resonance f0 per resonance N(−0.1, 0.03) %/°C, gain
  N(+0.5, 0.2) %/°C. The thermal model therefore has to extrapolate from the protocol's load history to the flow
  scan's state; it is not in-family.
- **Receiver** [assumption]: ADC delay of the GIRF protocol (4 µs, body) U[−2, 2] µs vs the gradient raster; per
  dwell slope 0.5–2.5 × (dwell − 4 µs) (decimation-filter group delay ~ samples × dwell) + 50 ns jitter per dwell;
  array vs body N(0, 30 ns).
- **Phantom / session** (girf_sensitivity `Nuisance`): Ø 17 cm sphere off-centre N(0, 3 mm), T2 80–250 ms, residual
  shim 2–8 Hz rms (random 2nd + 3rd order), receive-weighting gradient ≤ 30 %, RF frequency error N(0, 3 Hz), G_ss
  scale N(0, 0.2 %), per-slice offsets 5 µm, slice refocusing error N(0, 20 rad/m); SNR₀ (central 1.5 mm slice FID,
  one average, 4 µs) 300 with the body coil, 1000 with the array. Prescan errors given to the estimator: centre
  0.1 mm, B0 map 0.3 Hz (smooth), slice positions 1 µm (C4 CRLB ≈ 0.1–0.2 µm), receive weighting 5 %.
- **Signal**: per frame group and test polarity the exact in-plane sum over the slice (4 mm grid) of
  P̂(σk_a + κ_r t + k_ss) e^{−t/T2} e^{−i[ψ_r + σΦ_r]} with the PE weight e^{ik_q·r}; Φ_r = Σ_l k_l(t) Y_l(r) over the
  16 spatial basis functions (self term with the slice warp −η ΔB0(r)/(γ̄G_ss) and the B0-dependent slice position,
  cross and higher-order terms from the chain at the frame's thermal state), ψ_r = residual shim + concomitant
  (γ∫g_a² · z²/2B0 or (x²+y²)/8B0), κ_r the local through-slice B0 gradient. Evaluated exactly to 2nd order in the
  small in-plane phase (≤ 0.05 rad; checked against the brute-force sum: 7.6·10⁻⁵ relative, 4.6·10⁻⁴ rad) on a
  0.1 ms grid for the slow static phase; the output fields on a 1 µs internal raster (exact for the piecewise-linear
  input; sample times off the raster by the receiver delay), zero padding ≥ 6 ring-down times of the narrowest
  resonance / longest eddy term (a 20 ms padding wrapped a ζ = 0.0045 resonance around: 13–18 rad/m errors — fixed).
  Noise per sample 1/(SNR₀ √n_avg). Each thermal bin (3–4 survey groups, one cool-down set, ...) is simulated at
  its own mean θ.
- **Not simulated**: the excitation's own gradients (σ-even: cancel in the ± difference; their tails are part of the
  history in V), cross-frame leakage (cancelled by the polarity pattern), B0 drift within a ± pair, gradient
  nonlinearity, eddy-field patterns beyond 3rd order, the calibration part's own data (its outputs enter as the
  prescan errors above), the ±20 kHz FID receiver check, the array's coil combination.

## 4. Estimators

`qaest.py`:
- **Survey (self + B0)**: per sample WLS over units (thermal bin, slice, η) of −½ arg(s₊ s₋*) on (p̂_{s,η}, 1) → k_a(t),
  φ0(t), with the X stage's in-plane / higher-order contributions Σ_l k̂_l(t) Ȳ_l(s, η) removed (Ȳ_l: receive-weighted
  slice means from the prescan geometry). Two passes: A against the clinical-example chain with the residual
  unwrapped along time; B against A with the wrapped residual (no temporal unwrap). **Window cut**: the
  frequency-domain LS (girf_sensitivity) weights all samples of an input alike; with SNR₀ 300 and a 3 Hz shim the
  late samples of a 200 ms window are dephased (|s| → 10⁻³), and in pass A a single 2π slip there corrupted every
  frequency bin (group-delay errors of 0.4–0.9 µs on x / y in the first runs). The nonparametric H therefore uses the
  window up to where the median k sd reaches 0.1 rad/m (CUT), which beat 0.3 / 1 / 3 rad/m on the test draw; the long
  window is used where it matters, for the narrow resonances, through sample-weighted time-domain fits.
  Then increments-domain LS over the inputs, PL → continuous, hybrid table (nonparametric below the trusted band,
  parametric above), B0 term with the slice cross-talk term — as girf_sensitivity §1.3.
- **X stage**: per sample WLS of all 16 coefficients over the 200 (slice, η, PE) units of the ± phase differences,
  design rows A_l(t) = Re(M_l/M) from the estimated geometry (0.5 ms grid for the normal equations, interpolated),
  first 45 ms; deconvolution over chirp + tri100; each cross / 2nd / 3rd-order response fitted with two high-pass
  terms.
- **Resonances**: pass-1 list = the parametric fit's resonances in 0.3–2 kHz with |gain| > 0.2 %; pass 2 and every
  thermal set: time-domain LS of (f0, ζ, gain, delay) (ζ fixed to the pass-2 value for the per-set tracking fits) on
  the k increments over the full 200 ms, weights 1/sd per increment, ADC gaps masked, background = the pass-1 table
  with its own resonance divided out.
- **Thermal**: the two-compartment power model of §2.1 item 3 fitted to all per-set f0 values (shared τ₁, τ₂, w;
  per-resonance F_j, A_j); prediction at the flow scan's steady state (x = 1) and at any time from a power history.
- **Flow-state GIRF** (what the reconstruction would get): the warm-survey tables with each targeted resonance
  replaced by (f0 from the thermal model at x, ζ and gain from pass 2), cross terms from X, B0 terms from the warm
  survey; the receiver-delay difference (imaging dwell / coil vs the GIRF protocol's) from t_rx.
- **v2 validation**: per readout and axis the WLS over the four 1 mm slices against the flow-state prediction
  (wrapped residual; samples with k sd > 1 rad/m masked: in-plane dephasing, below); residual measured − predicted
  (observable) vs true − predicted (simulation only), both mapped to the R metric of the girf_sensitivity Gram.
- **Linearity**: β from K(ℓ)/ℓ − K(1) = β(ℓ − 1) k₂ over 0–6 ms of the triangles.
- **Receiver timing**: Gauss–Newton on the sample-time shift of k_a(t) per dwell against the cold estimate.

## 5. Results so far (single draws; interrupted before the Monte Carlo statistics)

Debug draws with the final simulator and estimator, unless stated (seeds = girf_sensitivity's chains):

- **Pulseq / PNS / protocol**: §2, §7 (complete).
- **Receiver timing (t_rx)** (draw 0, old estimator sign fixed): δ(2 µs) − δ(4 µs) −4648.0 ns vs truth −4648.6;
  8 µs 9485.9 vs 9485.6; 16 µs 28157 vs 28154; array vs body at 4 µs −21.0 vs −21.4 ns. Required: the imaging ADC
  (2 µs, array) timing relative to the GIRF protocol's (4 µs, body) to ≪ 70 ns → met by ~2 orders of magnitude.
  Without this calibration the dwell-dependent filter delay (assumed µs-scale [CHECK]) costs R ≈ 3.7 (draw 0).
- **Linearity**: β 2.63·10⁻⁴ vs 2.75·10⁻⁴ (all axes), 0.89–1.14·10⁻⁴ vs 0.98·10⁻⁴ (draw 1, perfect reference).
- **Self terms (survey)**: first versions were biased by (i) a circular-convolution wrap of narrow resonances in the
  truth simulator (fixed) and (ii) 2π slips of the time-unwrapped residual in dephased late samples, which corrupt
  every frequency bin (0.4–0.9 µs group-delay errors). With the window cut at a median k sd of 0.1 rad/m (best of
  0.1 / 0.3 / 1 / 3), draw 0: |ΔH| 1.3–1.7·10⁻⁴ (x), 1.5–2.8·10⁻⁴ (z) at 0.01–2 kHz, group delay 10 ns; y 1.3–2·10⁻³
  flat over all bands — consistent with a long eddy term (10–60 ms) the cut window cannot resolve, which mostly
  enters through the 10–15 Hz gain normalisation of the metric [hypothesis, not checked]. Target 10⁻⁴ over
  0.3–15 kHz: not yet met in this draw; the cut costs low-frequency information. A time-weighted (non-stationary
  noise) deconvolution is the proper fix [not implemented].
- **R (draw 0, refined level)**: cold, no X 0.209; cold + X (cross terms measured) 0.167; perfect except cross
  0.119; cold GIRF at the flow steady state 0.791; warm survey 0.433; flow-state model (warm + pass 2 + thermal)
  0.362; flow model without the receiver calibration 3.729. Reading: the X stage pays; the thermal state matters
  more than anything else (0.17 → 0.79 for +6.9 °C); the flow-state model only moved the *targeted* resonances (draw 0:
  one target at 1.62 kHz; a narrow 2.1 kHz resonance was outside the 0.3–2 kHz target band) and extrapolated nothing
  else. Two fixes are in `qamc.py` but **untested**: targets chosen by exposure |g|/ζ·|G_v2(f0)| > 0.02 up to 3 kHz,
  and a flow-state table interpolated per frequency between the warm and the hot survey (their states bracket x = 1).
- **Pass 2 / tracking** (old debug run): pass-1 f0 0.06 Hz off (cold), pass-2 f0 0.47 Hz off (flow load) with a
  0.97 Hz Jacobian sd, thermal-model prediction at the steady state 1.8 Hz off — this was before the per-sample
  weights / gap masks in the resonance fits and before the wrap fix; no valid numbers yet for the 0.2 Hz target.
- **v2 validation**: measured − predicted reproduces true − predicted (R_V 6.94 vs 6.94 with a poor estimate; noise
  0.065 rad/m rms per sample). Finding: the history's in-plane eddy tails dephase the thin slice (|s| dips to 10⁻³ in
  some readouts; 1.9 % of samples masked at sd > 1 rad/m); a per-sample V comparison is noise-limited for R (R ≈ 5.8
  from noise alone), so V is a model-consistency test (χ²) and a source for refitting a few parameters, not a
  stand-alone trajectory. A 2–5 ms settle gap before the V excitation or in-plane outer-volume saturation would reduce
  the dropouts [hypothesis].

### Resume (exact commands, from the repo root)

1. Generator (code in `rustlib/.agents/seq`, built): `rustlib/.agents/seq/target/release/girfqa --out
   data/hasty/seq/girf_qa/protocol` (≈ 25 s; regenerate once: the 2 µs-dwell ADC delay now sits on the RF raster —
   pypulseq `check_timing` flagged 43 µs in t_rx), then `rustlib/.venv/bin/python
   research/experiments/girf_qa/pulseq_roundtrip.py data/hasty/seq/girf_qa/protocol`.
2. Monte Carlo (≈ 40–45 min per draw single-threaded; 6 workers): `OMP_NUM_THREADS=1 systemd-run --user --scope -q -p
   MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -u research/experiments/girf_qa/qamc.py --tag main --ndraw
   12 --workers 6 --save > data/hasty/seq/girf_qa/mc/main.log 2>&1` (resumable; `touch data/hasty/seq/girf_qa/mc/STOP`
   ends after the current draw; `mc_main.jsonl` is empty now). Then `qaanalyze.py main` and `qafigs.py mc main`.
3. Loader cross-check: `HASTY_GIRF_MEASURED=data/hasty/seq/girf_qa/mc/measured_main_d0.json cargo test --release -p
   hasty-seq --lib -- girfmeas` (in `rustlib/.agents/seq`).
4. Speed: the truth fields use a 1 µs internal raster with up to 1 s zero padding (`qasim.UP`, `Truth.pad_s`); UP = 2
   would halve the cost (prediction error 4× the 1 µs one, ≈ 0.008 rad/m).



## 6. Output for downstream use

- **Measured-GIRF file** `hasty-girf-measured` v1 (JSON; written by `girfmeas_io.write`, read by
  `hasty_seq::girfmeas::MeasuredGirf::load`): per axis the continuous self term with the targeted resonances divided
  out (table, linear re / im interpolation on |f|, Hermitian — `Response::Table` semantics), the resonance list
  (f0 at the tables' thermal state x_ref, ζ, gain, df0/dx, sd), cross terms and 2nd / 3rd-order terms as high-pass
  sums, B0 terms as tables, the thermal model (τ₁, τ₂, w, P_flow, x_ref), receiver delays per dwell, β per axis.
  `MeasuredGirf::girf(x)` builds a `Girf` (tables, nested `Resonant` factors at f0(x), the high-pass cross terms as
  separate entries, B0 tables) at the thermal state x (0 = cold, 1 = the flow scan's steady state);
  `state_of(power)` computes x at the end of a power history (1-s bins of Σ_a G_a²), so a reconstruction can follow
  the flow scan's own heating; `resonance_list(x)` is the designer's list (`seqdesign --res-girf` would read it — not
  wired, seqdesign is not mine to change here). Free function `girfmeas::from_measurement(path, x)`; a
  `Girf::from_measurement` wrapper in girf.rs is a one-liner left out on purpose (the reconstruction agent may add
  history / loading code to girf.rs; this keeps the patch conflict-free).
- Checks: Rust unit test with an embedded file (resonance peak at f0(x) for two states, table interpolation ×
  resonance factor to 1e-9, high-pass B0 table to 10⁻³, x of a long flow-scan load = 1, chain applies); and a test that
  loads a file written by the Python estimator (`HASTY_GIRF_MEASURED=data/hasty/seq/girf_qa/mc/measured_main_d0.json`)
  and compares H at six frequencies and the B0 terms with the Python chain built from the same file.

## 7. Implementation and tests

Rust (`rustlib/.agents/seq`, patch `rustlib/.agents/girfqa.patch`; additive: four new modules, one new bin, lib.rs
module lines, `serde_json` added to hasty-seq; girf.rs untouched):
- `pulseq.rs`: writer (format 1.5.0 or 1.4.1; blocks, trapezoids, arbitrary gradients as corner-sample "extended
  trapezoids" with time shapes or centre-sampled shapes, RF with mag / phase / time shapes, ADC, LABELSET / LABELINC,
  ROTATIONS extension or baked rotations, definitions, md5 signature, pypulseq shape compression), events encoded
  into de-duplicated libraries at `add` (a 150 000-block heating train costs one row per block), raster / dead-time /
  limit checks, waveform and ADC-time reconstruction; reader (`SeqFile`) for round trips; helpers (sinc / block
  pulses, area trapezoids). Tests `md5_vectors`, `g_format_like_c`, `shape_compression_round_trip`,
  `write_read_round_trip` (both systems × both versions × baked / extension rotations: durations exact, raster
  waveforms to 2·10⁻⁷ relative, ADC times exact, labels), `dedup_and_trap_area` — pass.
- `pnsge.rs`: GE / TOPPE PNS model (hrmw, xrm, hrmbuhp); test `constant_slew_approaches_rheobase_scaling_and_matches_
  direct_sum` — pass.
- `girfqa.rs`: protocol pieces (inputs incl. pass-2 narrow chirp / tone burst with PNS fitting, frame builder with the
  raster-aligned ADC delay, PE steps, the v2 TR as pge2 segments, v-frames, heater TR, sweep order, quaternions); tests
  `inputs_respect_limits_and_pns`, `frame_is_valid_pulseq_and_round_trips`, `quaternion_round_trip` — pass.
- `girfmeas.rs`: loader; tests `parse_and_evaluate` — pass; `python_written_file_matches` passes trivially (skipped
  without `HASTY_GIRF_MEASURED`; not yet run on an estimator-written file).
- `hasty-bench` bin `girfqa`: writes all parts (≈ 25 s), check files, frames / inputs / power / flow-TR arrays,
  protocol.json with PNS (both models) per input and per part.
- `cargo test -p hasty-seq`: 45 passed, 2 ignored (all, incl. the existing ones); `cargo build -p hasty-bench` builds
  every bin (no warnings from the new code).

pypulseq 1.5.0.post1 round trip (`pulseq_roundtrip.py`, `protocol/roundtrip.json`): every `_check.seq` (one
instance of each of the 4–28 segment types per part, rotations baked in) is read by pypulseq; md5 signatures valid;
gradient waveforms at the raster points equal to 2.8·10⁻⁶ relative (≤ 1.0·10⁻⁷ T/m: the 9-digit shape text and the
derivative compression); ADC sample times identical (< 1 ps); `check_timing` clean with the GE rasters; total
durations identical. The full files without rotations (c_cal, t_rx, s_cold, x_3d, l_lin, d_cool) are read by
pypulseq with identical block counts and durations; the files with ROTATIONS (pypulseq 1.5.0 cannot parse that
extension) pass an independent parser (signature, block / duration sums, unit quaternions, every extension chain
resolves) and the Rust reader.

Python (`research/experiments/girf_qa/`): `prep_inputs.py` (v2-validation layout), `pulseq_roundtrip.py`, `qainputs.py`
(port of the pass-2 input generators + PNS; identical to the Rust output, checked by running `girfqa --targets`),
`qasim.py`, `qaest.py`, `qamc.py`, `girfmeas_io.py`, `qafigs.py`.

## 8. Open questions for the user (hardware / site)

1. **pge2 limits**: maximum samples per ADC event and per scan, waveform memory, number of segments; does pge2 /
   the GE interpreter accept the Pulseq 1.5.1 ROTATIONS extension (or which rotation mechanism does it use)?
2. **Receive coils**: can the body coil be selected for receive with a Pulseq sequence (and how many channels does it
   deliver)? Which array (channel count, data volume: x_3d is 46 GB at 48 channels)? Can the scanner store ~70 GB of
   raw data from one session, and in which format (ScanArchive)?
3. **Thermal**: any access to gradient-coil temperature sensors or GE's gradient heating model? If not, can a
   fibre-optic probe be taped to the bore liner for this session? Does the scanner's gradient supervision allow the
   1.5 × flow-scan load for 8 min (h2_hot) — site physicist?
4. **Gradient coil**: SuperG on the target Premier — 80 or 70 mT/m (TOPPE lists hrmw at 70)? Its `Scandbdt.cfg` PNS
   parameters (chronaxie, rheobase, α)?
5. **Phantom**: which QA sphere (diameter, filling: doped water or silicone oil, T1 / T2)? Can it be centred to ±5 mm
   and left in the bore ≥ 30 min before the session?
6. **Session**: 96 min pass 1 + ~20 min pass 2 with a ~10 min offline estimation in between (or pass 2 another day).
   Acceptable? A "core" plan (`--plan core`: fewer averages, no l_lin / h2_hot / s_hot, 6 cool-down sets) is 54 min.

## References

- Rahmer J et al. Rapid acquisition of the 3D MRI GIRF using a simple phantom measurement. MRM 2019;82:2146–2159.
  https://doi.org/10.1002/mrm.27902
- Kronthaler S et al. Trajectory correction based on the GIRF improves high-resolution UTE imaging. MRM 2021;85:
  2001–2015. https://doi.org/10.1002/mrm.28566
- Campbell-Washburn AE et al. (GE MR750 GIRF via SPI; group delay ≈ 8 µs). https://pmc.ncbi.nlm.nih.gov/articles/
  PMC5378680/
- OPTIKS (GE Premier / UHP vibration and acoustic spectra; numbers only in figures). https://arxiv.org/abs/2505.07117
- PMC7245110 (whole-body GIRF resonances ≈ 500–650 Hz and 1.13 kHz; from the brief).
- Bacon JB et al. Open-source framework for measurement of the GIRF. bioRxiv 2026.
  https://doi.org/10.64898/2026.08.24.745510 (code https://github.com/jbbacon/GIRF_PE_Python)
- HarmonizedMRI pge2 design rules and options: https://github.com/HarmonizedMRI/SequenceExamples-GE (pge2/docs/
  pge2-sequence-design-rules.md), https://github.com/HarmonizedMRI/pge2 (matlab/+pge2/opts.m, checksegment.m).
- TOPPE PNS model: https://github.com/toppeMRI/toppe/blob/main/+toppe/pns.m (hrmw: chronaxie 642.4 µs, rheobase
  17.9 T/s, α 0.310 m).
- spspiral-cest (PyPulseq 1.5.1 + rotation extension on GE): https://github.com/jweigandwhittier/spspiral-cest
- Pulseq file format and pypulseq: https://pulseq.github.io/, pypulseq 1.5.0.post1 (rustlib/.venv).
- Stich M et al. MRM 2020 (temperature dependence), Nussbaum J et al. MRM 2022 (thermal variation), Scholten H et al.
  MRM 2023 (7 T GSTF), Vannesjo SJ et al. MRM 2013 — see girf_sensitivity.md for the URLs.
