# Sequence designer (2026-11-05)

Goal: a Rust sequence calculator and optimizer for our own 4D/5D flow sequence. The readout is a 3D **yarnball-type
spoke**: every readout is a rotation of one base spoke. The design should extract as much information per spoke as the
gradient system and PNS allow. One crate, `rustlib/crates/hasty-seq`, shared later by the simulator (`hasty-mrsim`),
the reconstruction and the scanner export.

## 1. Components

| module | content | validated against |
|---|---|---|
| `system` | gradient limits (G_max, S_max, raster 10 µs), ADC raster, RF / dead times, B0, γ | — |
| `wave` | 3-axis piecewise-linear gradient on a raster; exact cumulative M0, M1, M2 and P = ∫GGᵀ (concomitant), with a chosen time origin; k(t); rotation; concatenation; limit checks (per axis and vector norm) | closed forms (trapezoids, bipolars), pypulseq |
| `encode` | velocity encoding: minimum-time lobes for a target (M0, M1) per axis (bipolars, prephaser + flow encode merged), joint multi-axis timing, encoding schemes (referenced / Hadamard 4-point, 7-point dual VENC), M2 (acceleration) and readout M1 reporting | moments of the designed waveforms (exact) |
| `girf` | LTI gradient chain: per-axis impulse response (delay, low-pass, multi-exponential eddy currents, or a measured GIRF table, with optional cross terms and the B0 term k0); applied by zero-padded FFT convolution → actual gradients and actual k(t) | analytic step responses |
| `pns` | SAFE model (Hebrank & Gebhardt 2000; implementation of Szczepankiewicz & Witzel, BSD-3): per axis, three RC low-pass branches of the slew; combined root-sum-square in % of the threshold. Hardware: `.asc` parameters per gradient system; the built-in set is pypulseq's *example*, not a real scanner | pypulseq `safe_pns_prediction` on the same waveforms |
| `traj` | time-optimal traversal of a k-space curve under vector G_max / S_max (Lustig 2008), extended with a per-sample slew cap (for PNS shaping); base-spoke families (yarnball variants); rotation sets (Fibonacci directions × golden in-plane angle, …) | existing `hasty-mrsim` version |
| `metrics` | per-spoke and per-set quality (§2), concomitant anisotropy, T2* / off-resonance cost, GIRF sensitivity | — |
| `optim` | parametric search over base-spoke parameters with constraints (§3) | — |

Limits under rotation: every readout is a rotated base spoke, so the base spoke is designed with **vector** limits
‖G‖ ≤ G_max and ‖dG/dt‖ ≤ S_max. These hold on every physical axis for every rotation (conservative by up to √3 on a
single axis). PNS is **not** rotation-invariant (axis-specific SAFE parameters), so it is evaluated over the actual
rotation set: worst case, and the distribution.

## 2. Information per spoke

For a centre-out base spoke rotated uniformly over SO(3), what counts is curve length, not samples (shrinking the
dwell adds samples on the same curve but no coverage). A spoke crossing the shell r with path length q(r) per unit
radius (q = 1/cos α, α the angle to the radial direction) leaves, rotation-averaged, L₁(r) = q / (4π r²) of curve
length per unit volume. Nyquist for a FOV D needs curves ≤ 1/D apart (L ≥ D²) and samples ≤ 1/D apart along the
curve (v·Δt ≤ 1/D, the ADC bandwidth cap). So N spokes reach Nyquist at density w(r) if

  N ≥ N* = max_r D² w(r) · 4π r² / q(r).

Check: straight spokes (q = 1) give N* = 4π k_max² D² = π M², the known count of centre-out radial half-spokes.
The base-spoke family (`traj::Yarn`) builds the radius law in: q = √(1 + f²), f = 4π r² D² w / N (a smooth
max(1, f), so the curvature stays finite where winding starts), and the direction moves on the unit sphere along a
path of geodesic curvature k_g(σ) = k_g0 + A sin(ωσ) (A = 0, k_g0 = 0: great circle = rotated planar spiral; k_g0:
cone-like; A: meander / yarnball, isotropic concomitants).

So the radial-speed profile v_r(r) sets the number of spokes. Radial lines (v_r = γ̄G_max) oversample the centre. A
uniform design needs v_r ∝ 1/r²: the spoke has to move tangentially more and more as r grows (the yarnball
principle), while the total speed ‖dk/dt‖ ≤ γ̄ G_max and the curvature–slew bound cap how much tangential motion
fits. The spoke's information is the k-space volume it covers to the target density; the cost per spoke is the TR:

  TR = t_RF + t_enc + T_ro + t_spoil,   scan time to Nyquist = N* · TR · (number of encodes),

and T_ro is bounded by T2* decay and off-resonance blur.

Primary objective: minimise the time to the target density, N*·TR. Secondary objectives:
- angular uniformity / incoherence of the rotation set (point-spread function sidelobes);
- concomitant anisotropy of the base spoke's P tensor (what the reconstruction operator cannot absorb; the paper
  section, `yarn_study`);
- GIRF sensitivity: ‖k_actual − k_nominal‖, or the trajectory error after delay correction;
- the readout first moment M1^ro(τ): flow displacement during the readout;
- acoustic resonances (spectrum of G) — later.

## 3. Optimizer

1. **Parametric (first):** base-spoke family with a handful of parameters (kmax, radial profile exponent, azimuthal
   turns, polar sweep and wobble, readout length). The time-optimal design enforces G_max / S_max exactly. PNS enters
   as a constraint: the waveform is redesigned with a per-sample slew cap lowered where SAFE exceeds the limit (fixed
   point), or as a penalty. Derivative-free search (CMA-ES / Nelder–Mead) over the parameters, objective from §2.
2. **Waveform-domain (later):** the gradient samples are the variables.
   - G_max and S_max are box / linear constraints.
   - SAFE PNS is a convex constraint (absolute values and the root-sum-square of linear filters of the slew).
   - Density / coverage objectives are evaluated by differentiable gridding.
   - Solved by projected gradient / ADMM, so the "camera" capabilities are used fully, not only within one
     parametric family.

### PNS under rotation (caution)

SAFE differs per physical axis, so PNS is **not** rotation-invariant. Never derate per axis in the base frame: a
spoke tuned to one axis's headroom breaks the limit once it is rotated onto a more sensitive axis.

Here:
- Only a scalar, rotation-invariant factor scales the vector slew of each TR part.
- G_max and S_max are designed as vector limits, so they hold on every axis for every rotation.
- PNS is evaluated on the physical axes of the played sequence over a rotation set (400 rotations in the search, 3200
  in the final check), with encodes and per-rotation tails.
- **Gate before export:** PNS must be checked on exactly the acquisition's rotations in their order (every spoke), not
  a subset.
- The worst orientation sets the factor, so most rotations run below the limit. A planar spoke concentrates slew in
  one plane, a 3D yarn spreads it; on the example hardware this gained only 2–15 % slew (readout factor 0.45 planar vs
  0.46–0.52), less than the yarn's extra curvature costs.

## 4. Velocity encoding

- **Moments are designed on the actual waveforms.**
  - Bipolars give ΔM1 = π / (γ VENC). The encodes differ only in M1 (their M0 is equal and is 0 at the echo), so ΔM1
    does not depend on the time origin.
  - The echo is the readout start (k = 0, first ADC sample of a centre-out spoke). Every encode's M0 = 0 there, and
    its M1 equals the scheme value. `seqdesign` prints both from the designed TR and asserts that the pair VENCs match.
  - Later readout samples add the readout's own M1(τ). It is the same for all encodes, so it cancels in velocity
    differences, but it displaces fast flow (§5).
- **Schemes:**
  - referenced and Hadamard 4-point;
  - classic dual VENC 7-point;
  - **coprime dual VENC** (Beghella Bartoli, Chechin, Boccalini, Sigovan, JCMR 2025), with layouts:
    - *seven*: shared reference, the denoiser's 7-point layout;
    - *eight*: one reference per VENC, interleaved, as in the paper;
    - *eight-Hadamard*: two balanced Hadamard sets.
- **Coprime decoding (`venc.rs`).**
  - VENC_lo / VENC_hi = p / (p + 1), so the phases fix v modulo 2·VENC_lo and 2·VENC_hi, hence modulo
    2p·VENC_hi (Chinese remainder). The range becomes ±p·VENC_hi = ±(p + 1)·VENC_lo, and neither VENC needs to
    exceed v_max.
  - m = nint((p φ_lo − (p + 1) φ_hi) / 2π) = (p + 1) n_hi − p n_lo, then a lookup table gives (n_hi, n_lo). This
    reproduces the paper's Table 1.
  - The two unwrapped estimates are combined by generalised least squares. Inverse covariance weights the low VENC
    more and includes the shared-reference correlation. The paper's Eq. 6 weights do not look inverse-variance; worth
    checking against the original.
- **Unwrap-failure model (not in the paper).**
  - The CRT statistic's noise is Σ c_e δφ_e with integer c_e. Its failure probability P(|·| > π) is computed from the
    exact phase-noise distribution by convolution; it matches Monte Carlo within ~10 %.
  - The Gaussian small-noise version underestimates it by up to 3× near SNR 3.
- **Hadamard caveat.**
  - Hadamard pairs measure v_i + v_j. Pairwise decoding then protects |v| ≤ VENC/√2 in the worst direction, so for
    the same protection Hadamard needs √2 higher VENC, which cancels its √2 noise advantage.
  - A lattice (nearest-point) decoder would protect √3/2·VENC. Not implemented.
- **`vencstudy`: equal protection** (no aliasing up to v_max in any direction). Noise is normalised to the scan time
  of 4 encodes.

  | thorax, v_max 3 m/s | enc | VENC lo / hi | σv time-norm @SNR20 | SNR for 1e-3 unwrap failures | encode ms |
  |---|---|---|---|---|---|
  | referenced | 4 | 3.0 | 47.7 mm/s | – | 0.68 |
  | classic dual 7p (1:2) | 7 | 1.5 / 3.0 | 31.6 | 2.5 | 0.86 |
  | coprime p=2, seven | 7 | 1.0 / 1.5 | 20.7 | 3.3 | 0.98 |
  | coprime p=2, eight | 8 | 1.0 / 1.5 | 18.7 | 4.2 | 0.98 |
  | coprime p=3, seven | 7 | 0.75 / 1.0 | 15.2 | 4.3 | 1.08 |
  | coprime p=4, seven | 7 | 0.6 / 0.75 | 11.9 | 5.4 | 1.17 |

  - Intracranial (v_max 1 m/s): same ratios. Coprime p=3 seven gives VENC 0.25 / 0.33, 5.1 mm/s, encode block 1.56 ms.
  - Each step of p gives ~1.4× less noise, needs ~+1 in SNR to unwrap, and adds ~0.1 ms of encode.
  - Seven unwraps at the lowest SNR, because the shared reference largely cancels in the CRT statistic.
- **The paper's limitations carry over.** A low VENC means a large M1: more intravoxel dephasing and acceleration
  errors near walls and after stenoses. The robustness step in the paper (a 9³ boxcar consistency check) belongs in
  the reconstruction. There, the velocity-model P²nP and the denoiser prior can take over the unwrapping, with CRT as
  the initialiser.

## 5. Rest of the TR

- **Steady state:** RF-spoiled GRE.
  - bSSFP would need balanced M0 and M1 every TR, conflicts with the encodes and the rotated readouts, and bands at 3 T.
- **Gradient spoiling:** constant *physical* total M0 per TR, which RF spoiling theory assumes.
  - Implemented in the per-rotation tail (ramp-down + rewinder + spoiler in one lobe).
  - Default 2 cycles / voxel on z.
  - The spoil direction is free: to do, let the optimiser choose it for the shortest tail / lowest PNS.
  - RF spoiling: quadratic phase over all TRs (117° or 50°), not reset per encode.
- **M1 "balancing":** the TR's total M1 differs between encodes and rotations.
  - Spoiling of moving spins becomes velocity dependent.
  - Residual coherences (stimulated echoes) can carry the previous encode's velocity encoding into the next →
    velocity bias.
  - Plan: extended phase graph (EPG) simulation with flow over the interleaved encode / rotation order, to size the
    spoil moment and RF increment. Add an M1-constant tail (one more lobe) only if the bias is relevant.
- **Readout M1:** centre-out readouts are not flow compensated.
  - γM1 = 0.06 rad/(m/s) at 25 % of kmax, but 4.7 at kmax for a 3 ms readout. Jets at 3 m/s are displaced by
    ~v·t (mm).
  - With the velocity-parametrised operator the readout M1 is part of the model (phase γ(M0(t)·x + M1(t)·v)), so it
    does not by itself force a short T_adc (§7). Without that model, jets favour short readouts.
- **Excitation:**
  - Thorax: slab-selective along SI (short sinc), so abdomen and neck signal does not streak into the spherical FOV.
    The slab rephaser merges into the encode lobes (encode.rs supports M0 / M1 offsets; TrSpec is still a hard pulse
    — to do).
  - Head: non-selective hard pulse.
- **Fat at 3 T** (440 Hz): a 3 ms readout blurs fat over ~1.3 cycles of phase.
  - Options: water excitation (+~1 ms TR, cheap at an 8–10 ms readout); fat modelled in the reconstruction (a long
    readout spreads chemical-shift phase and improves the conditioning of single-readout water–fat separation).
  - It matters in the thorax (epicardial fat); hardly intracranially.
- **Self-gating:** every centre-out spoke samples k = 0 at TE. This gives a free DC navigator per coil for
  cardiac / respiratory binning.
- **Eddy currents and Maxwell terms:**
  - Encodes and tails differ between TRs, so eddy-current (B0-term) phase offsets differ between encodes → velocity
    offsets.
  - With a measured GIRF including B0 terms this is predictable for the designed TR, and can become a design
    objective.
  - The encode-to-encode concomitant ΔP is reported. The reconstruction's concomitant model handles it.

## 6. Status

- [x] design (this note)
- [x] `hasty-seq` crate (17 tests):
  - system; wave with exact (Simpson) M0 / M1 / M2 / P;
  - encode: two-lobe min-time for any (M0, M1); referenced / Hadamard / dual-VENC;
  - girf: parametric chain as exact time-domain recursions, measured tables by FFT;
  - pns: SAFE + `.asc` parser;
  - traj:
    - time-optimal with vector limits, speed cap and arc-length slew cap;
    - Yarn family;
    - per-rotation tail;
    - rotation sets with per-set acquisition order;
  - tr: TR assembly, PNS of the played sequence;
  - metrics: N*, Nyquist-cell coverage, anisotropy, readout M1, GIRF error;
  - optim: N fit to the ADC budget, part-wise PNS derating, Nelder–Mead.
- [x] Cross-check against pypulseq 1.5.0 (`tools/seq/check_pypulseq.py`):
  - SAFE agrees to 1e-15 relative on a trapezoid, a triangle train and 3 rotations × 4 encodes of the designed
    sequence;
  - moments agree with fine-grid integration to its own O(h²) error (≤ 3e-8).
- [x] `seqdesign` bench + `tools/seq/plot_seqdesign.py`. Resumable: evaluation cache, STOP / SIGINT.
- [x] parametric optimizer (first version)
- [x] coprime dual VENC (`encode::Scheme::Coprime`, `venc.rs`, `vencstudy`), wired into `seqdesign` (`--scheme coprime7|coprime8|coprime8h --p --v-ext`)
- [ ] EPG-with-flow check of spoiling / interleaving (§5)
- [ ] slab-selective excitation in TrSpec; spoil direction as an optimiser variable; readout-M1 objective
- [ ] waveform-domain optimizer
- [ ] Pulseq export (`.seq`) of the designed TR loop
- Open: the scanner's PNS file (`.asc`), a measured GIRF, and the target resolution / FOV / T_adc. Results with the
  pypulseq EXAMPLE PNS hardware are not representative of the Prisma.

### First run (2026-11-05, `data/hasty/seq/example_hw`)

Setup: 2 mm iso, FOV 320 mm, T_adc 3 ms, VENC 1.5 m/s Hadamard, spoil 2 cycles / voxel, PNS ≤ 80 % on the
**example** hardware.

- **PNS binds every part.** Slew factors: encode 0.29, readout 0.45 (90 T/m/s), tail 0.53. The readout runs at the
  slew limit throughout (curvature-limited, |G| ≤ 23 mT/m), and the SAFE trace is flat near the target over the whole
  readout. So a local arc-length slew cap would gain little there; the encode and the tail are the levers.
- **Optimum ≈ rotated planar variable-density spiral:**
  - k_g0 0.04, A 0.09, ~2.5 turns;
  - N = 2978 (N* from the actual k: 3003), TR 5.81 ms, TE 1.38 ms;
  - N·TR = 17.2 s per encode.
- **Isotropy costs efficiency.** The anisotropy 0.71 of the planar shape drops to 0.33 at +8 % N·TR (A = 1, ω = 2)
  and to 0.24 at +25 %. `--aniso-weight` sets the trade-off.
- **Merged tail (ramp-down + rewinder + spoiler to a constant physical moment)** shortened the TR from 6.58 to 5.81 ms
  (−12 %).
- **Coverage:**
  - At N = N* the full rotation sets leave 23 % of the outer Nyquist cells empty (Poisson: 37 %).
  - Contiguous N/8 subsets (one bin) are near Poisson (84 % vs 88 %).
  - Fibonacci×golden sorted by frac(i/Φ) put each bin in one azimuth wedge; fixed with per-set ordering.
- **GIRF:** the clinical-example chain (≈ 10 µs effective delay) moves k by up to 3.4 Nyquist cells, so the
  reconstruction needs the GIRF-corrected trajectory.
- **Readout first moment:** γM1 is 0.06 rad/(m/s) at 25 % of kmax and 4.7 rad/(m/s) at kmax. This is the usual
  centre-out flow sensitivity at the k-space edge.

### Coprime runs (2026-11-06, example hardware, PNS ≤ 80 %)

| | Thorax `thorax_coprime7` | Intracranial `head_coprime7` |
|---|---|---|
| Res / FOV / T_adc | 2 mm / 320 mm / 3 ms | 1 mm / 220 mm / 2.5 ms |
| Scheme | coprime seven p=2, v_ext 3 m/s | coprime seven p=3, v_ext 1 m/s |
| VENC lo / hi | 1.0 / 1.5 m/s | 0.25 / 0.333 m/s |
| \|M1\| ref / hi-pair offset (mT·ms²/m) | 5.87 / 1.96 | 23.49 / 11.74 |
| N, TR, TE | 2968, 5.98 ms, 1.59 ms | 11074, 7.12 ms, 2.59 ms |
| Encode / tail | 1.51 / 1.24 ms | 2.51 / 1.88 ms |
| Slew factors (enc, ro, tail) | 0.28, 0.47, 0.56 | 0.24, 0.41, 0.49 |
| N·TR per encode; ×7 | 17.6 s; 124 s | 78.4 s; 549 s |
| σv at SNR 20 (per component) | 15.6 mm/s | 3.8 mm/s |
| SNR for 1e-3 unwrap failures | 3.3 | 4.3 |
| Encode-to-encode Maxwell ΔP at 15 cm | 0.44 rad | 1.54 rad |
| GIRF max \|Δk\| | 3.45 cells | 3.14 cells |

- Moments at TE recomputed from the designed TR waveforms: M0 ≤ 1.2e-14, pair VENCs exact in both runs.
- Intracranial: the PNS-derated encode (0.24 slew, all three axes bipolar at once in the balanced layout) is 2.51 ms
  vs 1.56 ms at full slew, and sets TE. Levers: unbalanced reference (fewer simultaneous axes), PNS-aware encode
  shape (trapezoid vs triangle), real Prisma `.asc`.
- Intracranial tail: 2 cycles/voxel at 1 mm is a 2000 /m spoil moment (58 mT/m peak on z). 1 cycle/voxel, or a
  spoil direction chosen by the optimiser, should shorten it; needs the EPG check (§5).
- Intracranial search cost: 3.2 h (≈ 108 s per evaluation at N ≈ 11k). Cheaper PNS proxy during the search
  (fewer rotations, then full gate at the end) before the next head run.

## 7. Readout length and sampling the whole TR (2026-11-06)

Premise (user): B0, concomitant fields and the GIRF are part of the reconstruction operator, so the readout can be
much longer than PC-VIPR's 2–3 ms, which corrects nothing but a post-hoc low-order polynomial on static tissue. And the
ADC runs during the encode and the tail too.

`adcstudy` sweep (`data/hasty/seq/adc_{thorax,head}`, `tools/seq/plot_adcstudy.py`), example PNS hw, best shape per
T_adc. Planar and the optimiser's shape agree within 1 % at every T_adc; the 3D shape (k_g0 0.5, A 1) costs ~11 %.

| | Thorax 3 → 8 → 12 ms | Head 2.5 → 8 → 12 ms |
|---|---|---|
| N·TR per encode | 17.7 → 11.7 → 10.5 s | 78.4 → 38.5 → 33.3 s |
| All 7 encodes, Nyquist-complete | 2.1 → 1.4 → 1.2 min | 9.1 → 4.5 → 3.9 min |
| ADC duty (readout only) | 0.50 → 0.73 → 0.80 | 0.35 → 0.63 → 0.72 |
| SNR at equal scan time (∝ √duty) | 1 → 1.21 → 1.26 | 1 → 1.34 → 1.43 |
| Mean sample time of outer shell | 2.7 → 7.1 → 10.6 ms | 2.3 → 7.1 → 10.6 ms |
| Off-resonance for π/2 at outer shell | 93 → 35 → 24 Hz | 110 → 35 → 24 Hz |
| Readout γ\|M1\| at ADC end | 4.8 → 12.6 → 18.8 rad/(m/s) | 7.8 → 25 → 38 rad/(m/s) |
| Readout concomitant scale at 15 cm | 0.37 → 1.0 → 1.5 rad | 0.50 → 1.8 → 2.7 rad |

- Diminishing returns past ~8–10 ms: the fixed overhead (encode + tail + RF) is ~3 ms thorax, ~4.7 ms head. The
  head gains more because its overhead is larger.
- The yarn samples the outer shell at ≈ 0.89·T_adc on average (radial: only the last sample is at kmax). An
  uncorrected yarn at 3 ms is already more exposed to off-resonance than PC-VIPR at 3 ms, so operator modelling is
  needed at any length for this trajectory.
- Not removed by the operator, grows with T_adc:
  - T2* (k-space magnitude filter): outer-shell weight 0.84 / 0.70 / 0.49 for T2* 40 / 20 / 10 ms at 8 ms.
    Modelling R2* (complex field map) adds unknowns. Short T2* near sinuses (head) and lung interfaces (thorax).
  - Fat: modelled species or water excitation.
  - Field-map error and its time variation (respiratory B0 in the thorax): error phase 2π·δf·t, so accuracy needs
    to be well below ~35 Hz at 8 ms; respiratory-resolved or jointly estimated field map.
  - Flow during the readout: needs the velocity-parametrised operator (per-sample M1); a per-encode image model
    cannot carry it. γ|M1| grows ~linearly with T_adc (oscillating readout), not ∝ T².
  - Operator cost: time-segmentation and concomitant ranks grow ∝ T_adc (per rotation for the concomitant term).
- Plan: decide T_adc with the synthetic phantom (B0 map, fat, R2*, flow in the simulation; reconstruct with and
  without each term in the operator) over T_adc ∈ {3, 6, 8, 10, 12} ms.

Sampling the encode and the tail, as currently designed:
- Encode lobes reach 2.3 kmax (thorax) / 2.8 kmax (head); 49 % / 44 % of encode samples lie inside kmax.
- The tail runs from the spoke end out to the spoil moment (2 cycles / voxel = 4 kmax): 0–20 % inside kmax.
- Duty counting only |k| ≤ kmax samples: thorax 3 ms 0.50 → 0.62, 8 ms 0.73 → 0.79; head 2.5 ms 0.35 → 0.51, 8 ms
  0.63 → 0.74.
- Encode samples: M1 changes during the encode, so they belong only in the velocity-parametrised operator. The
  encodes do not rotate, so the same physical k lines repeat every TR: a free FID + 3-axis projection navigator,
  but without DCW those lines weigh heavily in the data term, and systematic errors there (B0 eddy term, GIRF of
  the high-slew lobes) can bias. Check in simulation.
- Tail samples beyond kmax are off the reconstruction grid. To make the tail pay, design it as a return path
  inside kmax (spoil at the end, spoil moment sized by the EPG check) and count its samples in the information
  metric.
- Siemens caps samples per ADC object; long windows are split into segments (MATLAB Pulseq `calcAdcSeg`). The exact
  limit for the Prisma software version is to check.

## 8. Out-and-back yarn: no sample outside kmax (2026-11-06)

Rule (user): every sample used for reconstruction lies inside the kmax ball. Encode samples are kept as navigator data
only (same physical lines every TR), not reconstructed.

Design (`traj::Yarn::curve_return`, `seqdesign --return R`):
- The base curve goes centre → kmax (yarn), turns around on the sphere, and comes back along the same curve rotated by π
  about its end direction and reversed, down to R·kmax (R = 0: back to the centre), ending at rest (v_end = 0).
  The rotation keeps the tangential velocity at the turn; only the small radial component (≈ 1/q) flips, Gaussian
  smoothed (convex combinations, so still inside kmax).
- The return leg is a congruent copy, so every shell above R·kmax is crossed twice per spoke: N = n_design / 2.
- Per-rotation tail: one straight trapezoid from the rest point to the constant physical spoil moment. Both ends lie
  inside kmax, so the whole move does (convexity). Spoil |M0| ≤ kmax/γ̄ → 0.5 cycles / voxel by default. EPG check
  still needed for whether that spoils well enough with RF spoiling.
- With R = 0 every tail samples the same physical line (0 → spoil point on z): an SI-projection navigator each TR, like
  the encode. Not for reconstruction.
- Correction (user, 2026-11-06): the return leg is the out leg rotated by π about its end direction, i.e. mirrored
  across that line in the spoke's plane. It crosses every *shell* a second time, at *different* points: no k location
  is sampled twice within a spoke (only k ≈ 0). Over the rotation set, each k-space region gets samples from out legs
  (early) and return legs (late) of different spokes: two echo-time groups locally, not repeated locations. k = 0 is
  sampled at TE and again at TE + T_adc (γM1 ≈ 0 there).

Thorax sweep (`data/hasty/seq/sweep_ret`, 2 mm, FOV 320 mm, coprime seven p=2, v_ext 3 m/s, example PNS hw,
`tools/seq/plot_sweep.py`):

| T_adc ms | N | TR ms | N·TR s | 7 enc min | readout duty | tail ms | old centre-out N·TR s |
|---|---|---|---|---|---|---|---|
| 4 | 2389 | 6.12 | 14.62 | 1.71 | 0.65 | 0.38 | 15.22 |
| 6 | 1502 | 8.11 | 12.18 | 1.42 | 0.74 | 0.37 | 12.83 |
| 8 | 1101 | 10.10 | 11.12 | 1.30 | 0.79 | 0.37 | 11.66 |
| 10 | 866 | 12.11 | 10.49 | 1.22 | 0.83 | 0.37 | 10.97 |
| 12 | 717 | 14.09 | 10.11 | 1.18 | 0.85 | 0.37 | 10.52 |

- Over 3200 PNS-check rotations: readout max |k| 0.998–0.9997 kmax, tail exactly kmax, 0 raster samples outside.
- 4–5 % shorter N·TR than centre-out at every T_adc, and the tail shrinks from ~1.25 to 0.37 ms.
- Optimum still near-planar (spiral-in/out in one plane per spoke).
- Visible small gradient wiggle at the turn-around (within limits; a gentler smoothing could remove it).

Designer speed (same day): one evaluation 26 s → 1.9 s (centre-out) / 3.3 s (out-and-back).
- encode solver started its block-length scan at 4 samples (O(n³) per length): now starts at the bipolar-triangle bound
  and is memoised;
- PNS of the played sequence runs its chunks on threads;
- PNS derating scales every TR part above the target at once;
- warm-started N bisection.
Nothing on the GPU: the time-optimal traversal is a sequential sweep along the curve, and the SAFE filter is a
recursion in time (parallel only across chunks). `--threads T` splits cores, so several designs run in parallel
(`sweep_ret/run.sh`: 9 runs × 3 threads; a single run with all cores ≈ 1 min).

## 9. Spoiling: where the spoil moment has to go (2026-11-06)

`spoilcheck` (`hasty_seq::spoil`, `data/hasty/seq/spoilcheck_8ms`, `tools/seq/plot_spoilcheck.py`): multi-isochromat
RF-spoiled steady state (16384 isochromats on an R3 lattice in a 2 mm voxel, no diffusion) driven by the *actual* per-TR
net M0 and M1 of 8 ms thorax designs (coprime seven, rotations in acquisition order, all 7 encodes per rotation), flip
10°, RF increments 117° and 50°; myocardium (1.45 s / 45 ms), static blood (1.7 s / 200 ms), plug-flowing blood
0.71 m/s. Per-encode mean complex echo → pair velocity error.

| variant | |S|/ideal | TR-to-TR fluct. | worst velocity error |
|---|---|---|---|
| spoil inside kmax (0.5 cyc/voxel), constant z or rotating end point | 1.06–1.15 (blood up to 1.8) | 35–50 % | up to 110 mm/s |
| no net moment | 1.5–3.2 | 40–54 % | up to 130 mm/s |
| constant z 2 cyc/voxel (classic) | 0.98–1.00 | 0 static, 4–7 % flowing | 0 static, 36–59 mm/s flowing |
| same, every encode's TR M1 equal (what-if) | 0.98–1.03 | ≤ 0.4 % | ≤ 0.02 mm/s |
| trajectory continues out to 2 / 4 cyc/voxel (rotates with the spoke) | 1.000 | 2–4 % / 1–3 % | 1–13 mm/s, encode-locked |
| same, encode order cycled per rotation group | 1.000 | 2–4 % / 1–3 % | ≤ 1.9 mm/s |
| rotating, but a new rotation every TR | 1.00–1.03 | 3–22 % | 1–11 mm/s |

Conclusions:
- 0.5 cycles / voxel is not enough in any direction. Spoiling needs ≥ 2 cycles / voxel, so the spoiler part of the TR
  necessarily lies outside kmax; those samples are flagged (or ADC off), not reconstructed. The rule becomes: every
  *reconstruction* sample inside kmax.
- Constant physical spoiler: ideal for static tissue, but for flowing blood the encode-dependent TR first moment breaks
  RF spoiling (velocity bias tens of mm/s at 0.7 m/s). Fix: an M1-equalising lobe (every encode's TR has the same
  total M1); not designed yet (time cost).
- Rotating spoiler = the trajectory simply continues out along its end direction (`--spoil radial`): mean signal ideal,
  flow fine, and the whole TR after the encode is one rotated base waveform (no per-rotation tail). Needs the encode
  order cycled per rotation group (else the encode right after each direction change is biased). Leaves 1–4 % random
  TR-to-TR modulation (less with a bigger moment).
- A new rotation every TR is worse than repeating a rotation over the 7 encodes: RF spoiling needs a few TRs of
  repeated moment.
- The out-and-back yarn (§8) cannot spoil inside kmax and does not combine with a radial continuation (it ends at the
  centre). With spoiling outside kmax anyway, the natural design is centre-out + continuation:
  `data/hasty/seq/radial_spoil_8ms`: N 1066, TR 10.93 ms, tail 1.18 ms (to 4 kmax), N·TR 11.65 s.
- Not yet modelled: velocity spread within a voxel, diffusion, inflow (fresh spins), pulsatility.

Per-axis vs vector slew (why planar): hardware limits are per axis, but the base spoke is rotated into every direction,
so some rotation maps the slew vector onto a single axis. Measured on the 8 ms spoke over 2000 random rotations: worst
per-axis slew = 1.000 × the vector norm (median 0.83). A rotation-safe design therefore has a vector budget ‖Ġ‖ ≤ S,
and under it speed is curvature-limited, v ≤ √(γ̄S/κ). A curve on a shell of radius r has κ ≥ 1/r (Meusnier), with
equality only on great circles: planar is the minimum-curvature way to wind around a shell. 3D meanders add geodesic
curvature (+~11 % N·TR for k_g0 0.5, A 1).

### 9b. Velocity sweep, the closure bug, and the decision (2026-11-06, later)

Context (user): scans will be 8–12 min; the goal is quality (accuracy, high temporal and spatial resolution), not a
shorter scan. At fixed time N·TR is the undersampling per cardiac phase and the ADC duty the SNR; between designs the
one whose residual errors are deterministic / modellable wins over one with random TR-to-TR effects.

- One velocity is not enough: flow through a *constant* spoil moment adds a per-TR phase that grows by 2π (k_net·v) TR
  each TR, i.e. a quadratic phase on top of the RF-spoiling one, so the effective RF increment shifts by
  ≈ 69 rad per m/s (2 cyc/voxel at 2 mm, TR 11 ms). Sweep: 40 plug-flow velocities (0.05–2.5 m/s, 4 directions) ×
  RF increments 117° / 50°, blood T1 1.7 s / T2 200 ms (`spoilcheck --vsweep`, `--offset` for an off-centre voxel).
- Any TR-to-TR jitter of the net moment acts with the spin position: 2π δk·x. The first flow-compensated design
  missed by a 0.1 /m discretisation residual at the readout end (rotating with the spoke), enough to give 8–12 %
  fluctuation and up to 47 mm/s errors for flowing spins. Fixed with `Spoke::close_to` (a few-raster-point vector
  blip closes the readout exactly at its curve end point).

| over 40 velocities × 2 increments, isocentre and 10 cm off | |S|/ideal | fluct. median / max | velocity err median / max |
|---|---|---|---|
| constant z 2 cyc, TR M1 differs per encode (B) | 0.98–1.19 | 4.6 % / 20 % | 35 / 192 mm/s |
| **out-and-back + per-encode block: z 2 cyc + equal TR M1 (G, `--spoil flowcomp`)** | 0.98–1.57 | 0.1 % / 0.8 % | 0.00 / 0.14 mm/s |
| trajectory continues out to 2 cyc, encode order cycled (Dp) | 1.000–1.002 | 4.2 % / 4.6 % | 1.5 / 3.9 mm/s |
| same, 4 cyc (Ep) | 1.000–1.001 | 2.6 % / 3.0 % | 1.0 / 2.1 mm/s |

Decision: G. Velocities are essentially exact and the TR-to-TR signal is consistent (what the forward model assumes);
the flowing-blood magnitude depends on velocity (the RF-spoiling shift) but is constant over TRs and equal across
encodes, so it is absorbed by the per-bin magnitude the model estimates anyway. The rotating spoiler keeps the magnitude
ideal but leaves 2.6–4.6 % random TR-to-TR modulation that no model captures.

Still unmodelled in the check: velocity spread within a voxel (laminar profile), pulsatility (velocity changing over
the RF-spoiling memory), inflow / residence time, acceleration (the per-encode blocks equalise M1, not M2), diffusion.
