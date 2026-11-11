# GIRF sensitivity: what the thin-slice measurement captures, and what is left for the v2 readout (2026-11-11)

Question (user): the GIRF is a big potential error. Given a physical phantom and our GIRF reconstruction sequence, how
likely are we to capture the actual GIRF, and how good will the correction be?

Everything is simulated: no scanner, no measured GIRF. Scripts: `research/experiments/girf_sensitivity/` (Python;
`pb.py` of the budget is imported, not modified). Arrays, tables, figures:
`data/hasty/seq/studies/girf_sensitivity_2026-11-11/` (README there; `tables.md` has every number quoted here).
Metric as in `readout_physics_budget.md`: R = ‖(A − Ã)x‖ / (σ√(MC)); the segment rule tolerates R ≈ 0.2.

## Summary (simulation only; illustrative / random chains, assumed nuisances — see §1 for what is assumed)

1. **Capture.** With the WP2 thin-slice protocol (Ø 17 cm sphere, 4 slices per axis, chirp + 4 triangles, both
   polarities, 10 min of frames, SNR₀ ≈ 1 000 per FID sample) the self terms come out to ≈ 1·10⁻⁴ in 0.1–5 kHz
   (nonparametric; 5·10⁻⁴ at 5–10 kHz), the group delay to ≈ 1 ns (parametric, in-family) / 2–4 ns (nonparametric),
   ms eddy amplitudes to 0.3–0.8 %, broad resonances (ζ ≳ 0.01) far better than needed, B0 terms to ≈ 1.5·10⁻⁶ m
   (5–10 %), the DC gain only to 1.4·10⁻³ (slice positions; harmless, absorbed by the image). Cross terms are not
   measured by self-term slices at all. **Noise is not the limit**: 10 → 60 min changes little; the residual shim
   (fixable by alternating the slice-select polarity) and the **80 ms window** (narrow resonances, ζ ≲ 0.008) are.
2. **Correction quality** for the v2 readout, 48 random realistic chains, at the level the velocity model absorbs
   anyway (per-encode linear background phase) plus a per-axis delay refit: **80 ms window: median R 0.25,
   P(R ≤ 0.2) = 0.33, P(R ≤ 0.5) = 0.65**, with a heavy tail (p90 1.9) from narrow resonances under the readout's main
   lobe; **200 ms window, same 11 min: median 0.22, P = 0.42 / 0.90, p90 0.52**; 200 ms with 66 min of frames
   (pooled over a 1 h thermal session): median 0.17, P = 0.75 / 0.92 (24 chains); vessel-local R median 0.06–0.07,
   P(≤ 0.2) = 0.96–1.00. The floor set by the unmeasured cross terms is 0.10. A one-step reconstruction refit of
   cross / B0 amplitudes and resonance f0 / height on top takes the 200 ms case to median 0.17, P = 0.69 / 0.94 (16
   chains). In the budget's raw metric (fixed image) R stays 2.4–7.7 after correction, almost all of it per-encode coherent k offsets from cross terms on
   the physical encode / tail blocks (R_raw 6–9 with cross terms omitted vs 0.11–0.18 at the per-encode level).
3. **Dominant error sources for R**: (a) mechanical resonances under the readout spectrum (main lobe 688 Hz,
   vulnerable 0.4–1.05 kHz): f0 must be known to ≈ 0.1–0.2 Hz for Q ≈ 100, and with an assumed −0.1 %/°C drift R grows
   ≈ 0.37 per °C of unmodelled warming; (b) the group delay (tolerance ≈ 70 ns; a per-axis refit from the scan data
   removes it, CRLB 0.004–0.008 ns); (c) the high-frequency shape beyond a 1st-order low-pass (2nd-order roll-off,
   ripple; needs ≈ 1·10⁻⁴ — only a nonparametric H delivers it; the parametric fit gives median R 0.56 / 0.37);
   (d) unmeasured cross terms (0.11–0.18). Eddy and B0 terms act per encode (absorbed for R; they remain the velocity
   background problem of WP1). Gain errors, nonlinearity (β ≤ 3·10⁻⁴) and ADC timing (if refit) are uncritical.
4. **What to change** (ranked in §5): recon-side refit of per-axis delays and of resonance f0 / height (cheap, very
   well determined); readout design that keeps its spectral energy off the measured resonances; a 200 ms observation
   window (2nd-order shim; or Scholten's re-excitation); alternate the slice-select polarity; nonparametric H with
   raster compensation; GIRF in the flow scan's thermal state; per-TR navigator monitoring. Protocol in §6.
5. **The old budget number re-judged**: "a measured GIRF must be good to ≈ 0.5 % of the trajectory error" holds for the
   raw metric; at the per-encode level with a delay refit, a 5 % residual of the full omission gives 0.24–0.52, and what
   matters is the frequency-dependent part near the readout band, not the delays.


## 1. Method

### 1.1 Played trajectory and the R evaluator

- **Chains.** `gmodel.py` mirrors `hasty-seq/src/girf.rs` (`Response` incl. `at_pl`, `Girf::apply`, `variants()`,
  single-class `perturbations`), FFT path for every response. Checks: the TR of `girf.npy` through `clinical_example`
  matches the Rust output to 7.8·10⁻⁵ T/m of 0.043 (the Rust time-domain delay interpolates linearly at the corners);
  without a delay the time-domain recursion and the FFT path agree to 1·10⁻⁸ T/m. Added responses for truths outside
  the fitted family: several resonances per axis, a 2nd-order roll-off, a smooth complex ripple, sums of high-pass
  terms, an odd amplitude nonlinearity g + β g|g|/70 mT/m.
- **Played v2 TR** (`v2seq.py`): physical TR(e, R) = [encode e (physical)][R · base readout incl. the rewinder leg]
  [flow-compensated tail e (physical)], the 48 budget readouts with their own encode / rotation and **10 previous
  TRs** in the chain (141 ms of history; with only 2 previous TRs the trajectory of a chain with 10-20 ms eddy terms
  is off by 0.08 cycles/m rms, about what R = 0.2 tolerates). ADC 2 µs from the echo: reconstruction window 0–9.49 ms,
  navigator (rewinder) 9.49–10.79 ms.
- **R, linearised** (`rgram.py`, `rcalc.py`): per readout and ADC sample the 4×4 Gram of the data derivatives with
  respect to (k_x, k_y, k_z, global phase), whole object (2 364 426 voxels, 32 whitened true coil maps, the budget's
  truth image of v2) and, separately, the vessel voxels (129 021) for R_loc. Then R² = Σ_n w_n mean_r qᵀGq / (M C σ²)
  for any residual q = (Δk, Δφ₀). Carrier simplified (B0 map, table R2*, six-peak fat, blood decay; no intravoxel H
  table, no concomitant phase). **Calibration** against the budget's exact T10 numbers (same girf_err.npy, same
  readouts): exact evaluation with this carrier R = 33.16 / 1.840 (omission / 5 % residual) vs the budget's
  33.41 / 1.846; the linearised Gram gives 1.835 for the 5 % case (36.7 for the full omission: linearisation 11 % high
  at that size, exact below ~20 % of the omission). The Gram took 42 min on 6 CPU threads (the GPU was full).
- **What the reconstruction absorbs** (projections in the R metric, `rcalc.Rset`):
  `affine` = minus a global k → A k + b and phase c (the same for every readout: an affine map of the image
  coordinates and a global linear phase, absorbed by x); `enc` = + per-encode k offset and phase (the velocity model's
  per-encode linear background-phase polynomial, coherent over rotations); `field` = + a degree-1 field error over the
  readout (Δk = c t, Δφ = ω t: a constant gradient offset is a linear B0 field); `refined` = + a per-axis delay of the
  played gradient fitted from the scan data (Δk_a = −γ δ_a g_a(t), a per-sample k shift, essentially free per
  iteration). **`enc` is the level the velocity reconstruction has anyway; `refined` needs three extra scalars.**
  The raw level (fixed x, the budget's convention) is dominated by per-encode coherent errors of the physical encode and
  tail blocks (eddy, cross and B0 terms), which matter for the velocity background (WP1) but not for data consistency.

### 1.2 Truth chains

- girf.rs `variants()` / `clinical_example` (illustrative; the in-family case for the parametric fit).
- `truths.draw` (`rand`): random chains deliberately richer than any fitted model, ranges are assumptions informed by
  the literature [CHECK each against a measured GE GIRF]: per axis delay 1.5–4 µs, 1st-order low-pass 4–10 µs plus a
  2nd-order roll-off (fc 25–60 kHz, Q 0.5–0.8), eddy terms (0.05–0.3 %, 20–150 µs), (0.05–0.3 %, 0.5–5 ms), (0.01–0.08 %,
  10–60 ms: residual long terms after vendor pre-emphasis), 2–4 resonances 0.5–4 kHz with ζ 0.004–0.05 and peaks 1–8 %,
  a smooth complex ripple (|c| ≈ 0.2 %, bumps at 0.3–15 kHz), all six cross terms (high-pass, |k| ≈ 0.03 %, 0.5–3 ms),
  B0 terms (two high-pass terms, ~2·10⁻⁵ and ~1·10⁻⁵ m), odd nonlinearity β ≤ 3·10⁻⁴. Literature anchors (partly from
  memory): delays of a few µs and kHz-range mechanical resonances (Vannesjo 2013, Loecher 2025), resonances narrower
  at higher field (Scholten 2023), first-order phantom GIRFs suffice (Rahmer 2019, Bacon 2026).
- `truths.thermal(g, ΔT)`: linear temperature dependence (Stich 2020: self terms linear in temperature, B0 term not;
  left out). Coefficients are **assumptions** [CHECK]: delay +0.5 ns/°C, low-pass τ +0.1 %/°C, eddy amplitudes
  +0.3 %/°C, resonance f0 −0.1 %/°C, resonance peak +0.5 %/°C. Nussbaum 2022 report resonance shifts "in the same
  range as the linewidths" for heating to 32–65 °C, i.e. of order −0.05 to −0.2 %/°C for ζ ≈ 0.01–0.03.

### 1.3 Simulated measurement (`thinslice.py`)

WP2 protocol as planned: spherical phantom (Ø 17 cm, doped water), per axis 4 slices at ±8.75 / ±26.25 mm, 1.5 mm
(Hann-windowed sinc RF, TBW 4, 3 ms → G_ss 21 mT/m), test input on the slice axis, both polarities, FID from the
excitation, 80 ms ADC at 4 µs (GE raster), 100 ms per frame, the 12 slices interleaved over axes (each slice
re-excited every 1.2 s, T1 of doped water ≲ 0.3 s). Inputs (`figs/inputs.png`): a log chirp 0.1→20 kHz over 40 ms
at 0.5 S_max with the amplitude capped so |k| ≤ 0.8 k_lim (k_lim = the slice-dephasing limit, sinc(0.4)); four
triangles (ramps 60 / 100 µs at 0.9 S_max, 160 / 240 µs at 0.55 S_max), each with a prephaser so k runs −A/2 → +A/2.
SAFE PNS of every input ≤ 77 % (pypulseq example hardware, not GE's model [CHECK with pge2's PNS check]); with the
long triangles at 0.9 S_max it was 98–121 % on y. One set (3 axes × 4 slices × 2 polarities × 5 inputs) = 120 frames =
12 s; 50 averages = 10 min.

Signal per frame: s(t) = A_s e^{−t/T2} P̂(σk_a + k_ss + κ t) I_s(t; σ) e^{−i[σ(k_a p'_s + φ₀) + ψ_s]} + noise, with
k_a, φ₀ from the truth chain, P̂ the slice-profile FT (real: magnitude only), κ the through-slice B0 gradient, k_ss a
refocusing error; I_s the in-plane integral over the sphere's cross section with the receive weighting, the B0
inhomogeneity, the concomitant field (q = z²/2B0 for x/y inputs, (x²+y²)/8B0 for z), the in-plane cross-term gradients
(they meet the slice centroid) and the slice warp by the B0 field (dp = −ΔB0/G_ss), linearised in the small
σ-dependent terms; ψ_s (mean off-resonance, receiver phase, concomitant mean) cancels in the ± difference. Nuisances
per draw (assumptions): phantom offset N(0, 3 mm), T2 80–250 ms, residual shim 2–8 Hz rms over the sphere (random 2nd /
3rd-order field), receive-weighting gradient up to 30 % across the phantom, RF-frequency error / drift N(0, 3 Hz)
(shifts every slice by −Δf/(γ̄G_ss)), G_ss scale error N(0, 0.2 %), per-slice offsets N(0, 5 µm), k_ss N(0, 20 rad/m).
Not modelled: gradient nonlinearity (for self terms it cancels: the excited slice is an iso-surface of the same coil's
field, so the test field is k·p on it exactly; higher-order *eddy* field patterns would bias the linear fit [CHECK]),
cross-frame eddy leakage (use a + − − + polarity order), B0 drift between paired frames (≪ noise with interleaved ±).

**SNR (per sample of the central slice's FID, one average, 4 µs).** SNR₀ = SNR_vox · (V_slice/V_vox) ·
√(Δt / T_acq,vox). Example [assumption]: a 1 × 1 × 1.5 mm GRE voxel with SNR 40 from 1.05 s of total ADC time
(256 lines × 4.1 ms) in a head array; V_slice = π·85²·1.5 mm³ = 34 cm³ → SNR₀ ≈ 40 · 22 700 · 1.95·10⁻³ ≈ 1 800. Body-coil
receive or a lower voxel SNR: a few hundred. Range used: 300 / 1 000 (nominal) / 3 000.

**Estimator.** d_s(t) = arg(s₊ s₋*) unwrapped as a residual against a reference chain's prediction (clinical example)
**along time** (a first version wrapped per sample and failed for truths ≳ 3 µs of effective delay from the reference:
3 of 48 draws with R 3–57; fixed, those draws now 0.14–0.25); weighted LS over the slices per sample → k_a(t),
φ₀(t); frequency-domain LS over the inputs on the k *increments* (input and output then carry the same dwell box, so
no dwell-time compensation is needed; the regularisation must stay ≤ 10⁻¹² of the peak information, 10⁻⁶ biased H by
6 % at 10–20 kHz) → H_pl(f) on the 4 µs raster; converted to the continuous response by removing the PL alias sum
(H(f) ≈ [H_pl − Σ_{m≠0} sinc²(fΔt+m) H_par(f+m/Δt)] / sinc²(fΔt): the "dwell-time compensation" of Stich 2020 in
the PL convention, 0.5 % at 10 kHz); nonparametric table below the frequency where its predicted sd exceeds 2·10⁻³
(9–16 kHz with the 80 ms window, 3–7 kHz with 200 ms; 5th–95th percentile), parametric above. Parametric fit on top (free DC gain, delay, 1st-order low-pass, 2 eddy terms,
resonances added greedily; weights 1/√(sd² + (2·10⁻⁴)²)). B0 term: two high-pass terms + c·Ĥ_aa (the slice-position
cross-talk k·p̄, absorbable). Cross terms are **not** measured by self-term slices (in-plane gradients need in-plane
resolution, §5); left at 0.

## 2. What the readout is sensitive to

All numbers: global R at the `enc` level (what the velocity reconstruction absorbs anyway), ranges over the girf.rs
`resonance` chain and three random chains (`sens.json`, `tables.md`); R is linear in the error in this regime (1 %
and 10 % cases agree to ≤ 2 %), so a tolerance scales directly. "For R = 0.2" means that class alone; independent
classes add in R².

**Omission (no GIRF in the model).** With the full played TR and history the k error is already 1.0–2.6 cycles/m at
the echo (encode, flow-compensated tail and previous TRs; the budget's T10 played the readout spoke from rest, where it
is 0) and 8–14 cycles/m rms over the readout; the B0-term phase is 0.03–0.10 rad rms. R_raw 38–86 (linearised, ~10 %
high at this size; spoke from rest 35–36 as in the budget), enc 30–71. Leaving 5 % of it: raw 1.9–2.7, enc 1.5–2.1,
refined 0.24–0.52 — the budget's "the GIRF must be known to ≈ 0.5 %" holds for raw/enc, but most of what remains after
a per-axis delay refit is the frequency-dependent part.

| class | test error | R enc | tolerance for R_enc = 0.2 | R refined (delay refit) | comment |
|---|---|---|---|---|---|
| group delay, all axes | 10 ns | 0.026–0.029 | ≈ 70 ns | 0.005–0.008 | = an ADC/receiver timing mismatch between GIRF and imaging protocols; refit from data (CRLB 0.004–0.008 ns) |
| delay, one axis | 10 ns | x 0.016–0.018, y 0.018–0.020, z 0.009 | 100–220 ns | 0.002–0.006 | |
| 1st-order low-pass τ | 1 % (40–100 ns) | 0.18–0.20 | ≈ 1 % | 0.064–0.094 per 1 % | the low-pass mostly acts as delay; its shape remains after the refit (≈ 3 %) |
| 2nd-order roll-off (fc 25–60 kHz) | omitted | 15–20 | must be measured | 3–5 | the high-frequency *shape*; a 1st-order parametric model cannot represent it |
| smooth ripple (bumps of 0.2 %, 0.3–15 kHz) | omitted | 0.32–0.45 | ≈ 1·10⁻⁴ in H | 0.23–0.29 | sets the needed nonparametric accuracy |
| eddy amplitudes | 10 % | 0.04–0.08 | 25–50 % | 0.04–0.05 | coherent per encode → absorbed; raw 1.2–1.8 |
| ms eddy amplitude | +0.01 % abs | 0.016–0.037 | 0.05–0.1 % abs | 0.014–0.025 | raw 0.6–0.7 |
| eddy time constants | 10 % | 0.04–0.07 | 30–50 % | 0.03–0.04 | |
| long eddy (10–60 ms) | omitted | 0.14–0.20 | | 0.010–0.021 | absorbed by the degree-1 field level (0.02–0.04) |
| cross terms | 10 % / omitted | 0.011–0.018 / 0.11–0.18 | ≈ 100 % | 0.011–0.015 / 0.032–0.041 with 12 refit directions | raw 6–9 when omitted: a per-encode k offset (physical blocks) |
| B0 terms | 10 % / omitted / +1·10⁻⁶ m | 0.008–0.012 / 0.08–0.12 / 0.004 | ≈ 50 % | 0.008–0.011 / 0.011–0.029 with 6 refit directions | phase coherent per encode; matters for velocity (WP1), not for R |
| resonance f0 (ζ 0.01–0.03, peak 3–8 %) | +1 Hz | 0.06 / 7.0 / 0.47 | 0.03–3 Hz | 0.05 / 6.8 / 0.43 | rand1: peak 7 % at 700 Hz, ζ 0.0045 |
| resonance f0, girf.rs chain | +0.1 % (1–1.4 Hz) | 0.39 | 0.05 % | 0.35 | |
| resonance f0, ζ = 0.005, peak 5 % (`figs/res_sweep.png`) | +0.05 % | 0.3–2.7 at 0.4–1.05 kHz; ≤ 0.1 above 1.1 kHz (0.21 at 1.5 kHz) | 0.004–0.03 % (≤ 0.2 Hz) in the readout's main band | | the v2 readout's spectrum peaks at 688 Hz (relative |G| 0.27 at 500 Hz, 0.21 at 1 kHz, 0.05 at 1.5 kHz) |
| resonance peak height | +10 % rel / +0.1 % abs | 0.3–3.3 / 0.04–0.46 | 0.6–7 % rel | 0.28–2.6 | |
| resonance damping ζ | ×1.1 | 0.17–2.6 | 1–10 % | 0.17–2.3 | |
| DC gain | 10⁻³ | raw 0.37 → affine 0.03–0.08 | absorbed (image scale) | 0.007–0.027 | slice-position errors land here |
| odd amplitude nonlinearity | β ≤ 3·10⁻⁴ omitted | ≤ 0.008 | uncritical | | |
| thermal drift, assumed coefficients | ΔT = 1 / 5 / 10 °C | 0.37 / 1.85 / 3.7 (girf.rs chain); 20.6 at 10 °C (rand1) | ≈ 0.5 °C | 0.32 / 1.63 / 3.4 | 95 % from the resonance f0 shift (−0.1 %/°C assumed); delay, low-pass, eddy coefficients give ≤ 0.1 at 5 °C |

Reading:
1. **Resonances near the readout's spectrum dominate** — their f0, height and damping, and the temperature drift of f0.
   A narrow resonance (Q ≳ 100) in 0.4–1.05 kHz needs f0 to ~0.1–0.2 Hz and its temperature known to a fraction of a
   degree; above ~1.1 kHz the v2 readout has little energy and the same error gives ≤ 0.1. WP1's resonance targets
   (f0 ≈ 0.02 %, height ≈ 0.4 %) are confirmed for R as well.
2. **Delay and low-pass are not uncritical for R** (WP1 judged them uncritical for the velocity background, which is
   right there): the group delay must be right to ≈ 70 ns — or refitted from the scan data, which costs three scalars
   and is extremely well determined (§5). After the refit what is left of the delay / low-pass class is its
   frequency-dependent shape (2nd-order roll-off, ripple): the nonparametric H must be good to ~10⁻⁴ over 0.3–15 kHz.
3. Eddy, cross and B0 terms act mostly through the physical encode and tail blocks: a per-encode k offset / phase,
   absorbed by the per-encode background-phase polynomial (`enc`). They stay critical for the *velocity background*
   (WP1 targets: eddy amplitudes ≈ 1 %, τ ≈ 2 %), not for data consistency.
4. Unmeasured cross terms (|k| ≈ 0.03 %) cost R_enc 0.11–0.18 — tolerable, and a 12-direction refit takes it to 0.03.
   Gain errors (slice positions) are absorbed by the image.


## 3. How well the thin-slice measurement captures the chain

In-family truth (girf.rs `resonance` chain), SNR₀ 1 000, nominal nuisances, 24 draws per row (`acc_res_*`), rms
error of the parametric fit's parameters (delay and low-pass trade off; their sum, the group delay, is what matters):

| run | delay [ns] | group delay par / np [ns] | low-pass τ [%] | short eddy a / τ [%] | ms eddy a / τ [%] | resonance f0 [Hz] / ζ / peak [%] | B0 term, 0.5–2 kHz [m] | DC gain |
|---|---|---|---|---|---|---|---|---|
| 10 min | 3.6 | 0.82 / 4.1 | 0.068 | 11 / 7.5 | 0.76 / 2.3 | 0.0018 / 0.016 / 0.006 | 1.7·10⁻⁶ | 1.4·10⁻³ |
| 20 min | 3.1 | 0.80 / 2.9 | 0.060 | 9.9 / 7.2 | 0.76 / 2.3 | 0.0018 / 0.017 / 0.006 | 1.6·10⁻⁶ | 1.4·10⁻³ |
| 60 min | 2.5 | 0.82 / 1.6 | 0.052 | 8.6 / 7.0 | 0.76 / 2.3 | 0.0018 / 0.018 / 0.006 | 1.5·10⁻⁶ | 1.4·10⁻³ |
| 10 min, alternating slice-select polarity | 2.9 | 0.46 / 4.1 | 0.058 | 8.4 / 6.8 | 0.28 / 0.97 | 0.0015 / 0.014 / 0.006 | 1.3·10⁻⁶ | 1.4·10⁻³ |
| 10 min, noise only (no nuisances) | 0.8 | 0.26 / 1.8 | 0.018 | 3.8 / 3.3 | 0.05 / 0.31 | 0.0005 / 0.009 / 0.001 | 6.6·10⁻⁸ | 2.6·10⁻⁶ |

Nonparametric self term, |Ĥ − H| with the DC gain removed, median over draws:

| run | 10–100 Hz | 0.1–0.5 kHz | 0.5–2 kHz | 2–5 kHz | 5–10 kHz |
|---|---|---|---|---|---|
| 10 min | 8.0·10⁻⁵ | 6.9·10⁻⁵ | 7.8·10⁻⁵ | 1.2·10⁻⁴ | 5.6·10⁻⁴ |
| 20 min | 7.8·10⁻⁵ | 7.0·10⁻⁵ | 7.8·10⁻⁵ | 1.0·10⁻⁴ | 4.0·10⁻⁴ |
| 60 min | 7.7·10⁻⁵ | 7.2·10⁻⁵ | 7.8·10⁻⁵ | 9.2·10⁻⁵ | 2.5·10⁻⁴ |
| 10 min, alternating G_ss | 3.0·10⁻⁵ | 2.8·10⁻⁵ | 4.3·10⁻⁵ | 9.7·10⁻⁵ | 5.6·10⁻⁴ |
| 10 min, noise only | 3.9·10⁻⁶ | 3.7·10⁻⁶ | 2.2·10⁻⁵ | 3.5·10⁻⁵ | 2.4·10⁻⁴ |
| realistic chains (`R_rand_10`, 48 draws; p90) | 8.5·10⁻⁵ (2.5·10⁻⁴) | 9.1·10⁻⁵ (2.3·10⁻⁴) | 1.9·10⁻⁴ (6.6·10⁻⁴) | 1.5·10⁻⁴ (3.1·10⁻⁴) | 6.2·10⁻⁴ (1.0·10⁻³) |

Attribution (one nuisance at a time, `acc_res_10min_only_*`):
- **Residual shim (2–8 Hz rms) is the dominant systematic for the self terms**: it shifts and warps each slice by
  −ΔB0/G_ss in a way that flips with the test polarity (σ k_a dp in the phase), giving H errors ~7·10⁻⁵ at low
  frequency, 0.75 % in the ms eddy amplitude and 0.7 ns of group delay. Alternating the slice-select polarity over the
  averages (the shift flips with G_ss) removes the first-order part: 2.5–3× smaller (table). A better shim does the
  same (GE Premier has five 2nd-order shim coils).
- Phantom offset (3 mm) and receive-weighting asymmetry (≤ 30 %) put the slice's in-plane centroid off-centre; the
  in-plane cross-term gradients then add k_c · r̄_c to the B0-term intercept: 1.2·10⁻⁶ / 5.5·10⁻⁷ m, i.e. 5–10 % of the
  B0 terms. Uncritical for R (B0 terms are absorbed per encode), relevant for the velocity background (WP1).
- Slice-position and G_ss-scale errors only change the DC gain (1.4·10⁻³ rms here): absorbed. RF-frequency drift moves
  all slices together → cross-talk k·p̄ into the B0 term, removed exactly by the c·Ĥ_aa term of the B0 fit.
- Noise: not limiting at SNR₀ 1 000 except above ~5 kHz (and for the nonparametric group delay, 1.8 ns); averages
  beyond 10 min buy little because the nuisances dominate.
- Not captured by this table: **narrow resonances**. With the 80 ms window the response of a resonance with ring-down
  time 1/(ζω₀) ≳ 20 ms (ζ ≲ 0.008 at 1 kHz) is truncated; frequency resolution 1/T = 12.5 Hz vs half-widths of a few
  Hz. These are the draws that fail in §4.


## 4. R after correction (Monte Carlo)

Random realistic chains (§1.2) with nuisances and noise; one draw = one chain + one session; nonparametric estimate
unless stated; R at the levels of §1.1 (`raw` is dominated by per-encode coherent errors and shown for completeness).
`figs/R_cdf.png`, `figs/R_vs_resonance.png`, `tables.md`.

| run | QA time | R enc median [p10, p90] | P(R_enc ≤ 0.2 / 0.5) | R refined median [p10, p90] | P(R_ref ≤ 0.2 / 0.5) | vessel-local R refined median, P(≤ 0.2 / 0.5) |
|---|---|---|---|---|---|---|
| ADC 80 ms, 50 averages, SNR₀ 1 000 (48 draws) | 10 min | 0.30 [0.17, 1.95] | 0.21 / 0.65 | 0.25 [0.13, 1.87] | 0.33 / 0.65 | 0.08; 0.77 / 0.92 |
| **ADC 200 ms, 25 averages, SNR₀ 1 000 (48 draws)** | 11 min | 0.26 [0.19, 0.62] | 0.15 / 0.88 | **0.22 [0.15, 0.52]** | **0.42 / 0.90** | **0.07; 0.96 / 0.98** |
| ADC 200 ms, 150 averages (24 draws) | 66 min | 0.20 [0.17, 0.46] | 0.58 / 0.92 | 0.17 [0.12, 0.44] | 0.75 / 0.92 | 0.06; 1.00 / 1.00 |
| ADC 80 ms, 50 averages, SNR₀ 300 (24 draws) | 10 min | 0.32 [0.19, 3.9] | 0.17 / 0.58 | 0.29 [0.14, 3.8] | 0.25 / 0.58 | 0.07; 0.71 / 0.83 |
| ADC 80 ms, 10 min, no session nuisances (24 draws) | 10 min | 0.29 [0.15, 3.9] | 0.25 / 0.58 | 0.26 [0.13, 3.8] | 0.33 / 0.58 | 0.07; 0.71 / 0.83 |
| same chains, *perfect* GIRF except the unmeasured cross terms | – | 0.14 [0.09, 0.22] | 0.83 / 1.00 | 0.10 [0.06, 0.15] | 0.96 / 1.00 | 0.05; 1.00 / 1.00 |
| parametric estimate instead (80 ms / 200 ms) | | 0.77 / 0.48 | 0.00 / 0.19, 0.00 / 0.58 | 0.56 / 0.37 | 0.00 / 0.38, 0.02 / 0.81 | 0.20 / 0.11 |
| in-family girf.rs `resonance` chain (ζ 0.03), 80 ms (24 draws) | 10 min | 0.13 [0.12, 0.15] | 1.00 / 1.00 | 0.11 [0.11, 0.12] | 1.00 / 1.00 | 0.06; 1.00 / 1.00 |

What decides the outcome is the chain, not the session. **With the 80 ms window the per-draw R follows the truth's
resonance exposure** max_j (peak_j / ζ_j)·|G_readout(f0_j)| (Spearman 0.77): exposure < 1 (21 of 48 chains: no narrow
resonance under the readout's main lobe) R_refined 0.16 [0.12, 0.26], P(≤ 0.2 / 0.5) = 0.62 / 0.90; exposure ≥ 1
0.72 [0.18, 3.6], P = 0.11 / 0.44. **The 200 ms window removes that dependence** (Spearman 0.24; exposure ≥ 1:
0.24 [0.15, 0.55], P = 0.33 / 0.89) and the tail (p90 1.87 → 0.52) at the same QA time. Without session nuisances
the 80 ms result is the same (0.26 vs 0.25): shim, positioning, coil weighting and frequency drift do not limit R; the
window and the unmeasured cross terms do. A perfect self- and B0-term measurement leaves the cross-term row
(refined 0.10); with broad resonances only (girf.rs chain) the measured correction reaches that floor (0.11, every
draw ≤ 0.2). The remaining gap at 200 ms (0.22 vs 0.10, i.e. ≈ 0.2 in quadrature) is the measurement error of the
frequency-dependent part (smooth ripple / roll-off shape at 1–2·10⁻⁴, residual narrow-resonance error); the refit below
says how much of it the reconstruction can take back.

Paired on the same 24 chains (R_refined median; P(≤ 0.2 / 0.5)): 80 ms 10 min 0.27 (0.25 / 0.58); SNR₀ 300 0.29
(0.25 / 0.58); no nuisances 0.26 (0.33 / 0.58); 200 ms 11 min 0.23 (0.42 / 0.92); 200 ms 66 min 0.17 (0.75 / 0.92). With
the long window the averages do pay (its late samples are low-SNR); with 80 ms they do not.

A hard per-input cut of the late, noise-dominated samples (stop where the k variance grew 25× / 100×) was tried and
not adopted: it rescued one short-T2* draw (1.05 → 0.16) and hurt nine with narrow resonances (e.g. 0.33 → 2.5;
`old_cut100/`). The right tool is a time-weighted (time-domain WLS) estimator [not implemented].

Recon-side refit on top (one linearised step on 16 draws of the 200 ms run, `mc_refit.py`, `mcrefit_R_rand_T200.jsonl`):
| level (16 chains, 200 ms, 11 min) | R median [p10, p90] | P(≤ 0.2 / 0.5) |
|---|---|---|
| refined (enc + field + per-axis delays) | 0.23 [0.17, 0.40] | 0.44 / 0.94 |
| + 12 cross-term and 6 B0-term directions | 0.20 [0.13, 0.39] | 0.50 / 0.94 |
| + f0 and peak height of every resonance the parametric fit found (3–12 per chain) | 0.17 [0.11, 0.32] | 0.69 / 0.94 |
| vessel-local, all of the above | 0.03 [0.02, 0.07] | 1.00 / 1.00 |

The refit directions are derivatives of the *model's* played trajectory (computable before the reconstruction), one
linearised step, image fixed; a joint fit with the image will lose some of this [CHECK].


## 5. Mitigations, ranked

Effect on R at the `enc` level unless stated; "cost" = scan / QA time and reconstruction cost.

1. **Refit per-axis delays in the reconstruction** (3 scalars; Δk_a = −γ δ_a g_a(t), a per-sample k shift).
   Removes the group-delay class (tolerance 70 ns → 250–400 ns) and an ADC-timing mismatch between the GIRF and imaging
   protocols; in the MC the median R goes 0.30 → 0.25 (80 ms) and 0.26 → 0.22 (200 ms window). Fisher information of the 10-min scan: CRLB 0.004–
   0.008 ns per axis from the reconstruction samples, 0.01–0.02 ns from the navigator samples alone (image known up to
   the absorbable directions; a joint image fit will be less precise, by how much is open [CHECK]). Cost: three
   scalars per outer iteration, the k shift is free in the NUFFT. Needs the played gradient per sample (from the
   played records).
2. **Keep the readout's spectral energy away from measured resonances** (resonance-aware design,
   `seqdesign --res-girf <measured>`, WP1 step 3: +8 % scan time halved the rotation-dependent resonance error). The
   v2 readout's main lobe is at 688 Hz with sidelobes to ~1.05 kHz; a resonance there with ζ ≈ 0.005 costs R 0.3–2.7
   for a 0.05 % f0 error, above 1.1 kHz ≤ 0.1 (`figs/res_sweep.png`). This is the only mitigation that also makes the
   result robust to thermal drift and to resonances the measurement misses. Cost: a re-design per scanner after WP2,
   scan time +0–10 % [CHECK on the measured GIRF].
3. **Long enough observation of the response** (narrow resonances). ADC 200 ms instead of 80 ms, same chains:
   R_refined 8.0 → 0.68 (peak 7 % at 700 Hz, ζ 0.0045), 7.95 → 0.82, 3.99 → 0.45, 2.12 → 0.28; a benign chain
   0.15 → 0.17. Needs a slice
   T2* of ≳ 100 ms (good 2nd-order shim; the Premier has the coils) or Scholten 2023's re-excitation after the probing
   gradient; Bacon 2026 reconstructed 175 ms windows (with phase encoding). MC (48 chains, same QA time): R_refined
   median 0.25 → 0.22, p90 1.87 → 0.52, P(≤ 0.5) 0.65 → 0.90, vessel-local P(≤ 0.2) 0.77 → 0.96. Cost: frame 220 ms;
   at the same QA time half the averages. Side effect in this estimator: the late, low-SNR samples enter every
   frequency bin, so the nonparametric band shrinks (trusted up to 3–7 kHz instead of 9–16 kHz; the parametric
   extension takes over above) — a two-window estimator (200 ms below ~2 kHz for the resonances, the first 80 ms above)
   or a time-weighted one should give both [hypothesis, not run]; more averages do help here (66 min: median 0.17).
4. **Refit resonance frequencies / heights in the reconstruction** (linearised directions, 2 per resonance, on top of
   1): thermal drift of 5 °C with the assumed −0.1 %/°C f0 coefficient: R 1.85 → 0.16 (girf.rs chain), 0.55 → 0.16
   (rand0); f0 +1 Hz on every resonance: 0.06 → 0.003, 0.47 → 0.02; fails as a one-step linearisation for a narrow
   resonance at the main lobe (rand1: 7.0 → 1.2 for +1 Hz, 15 → 7.4 for 5 °C) — an iterated (Gauss–Newton) refit may
   do better [hypothesis, not run]. Cost: one played-trajectory derivative per resonance (precomputable), a handful of
   scalars per outer iteration.
5. **Alternate the slice-select polarity over the averages** (free): removes the first-order B0-warp bias of the
   self terms; low-frequency H error 6.9·10⁻⁵ → 2.8·10⁻⁵, ms eddy amplitude 0.76 % → 0.28 %, eddy τ 2.3 % → 1.0 %,
   group delay 0.82 → 0.46 ns. Matters more for the velocity background (eddy amplitudes) than for R.
6. **Nonparametric H (not a 1st-order parametric fit) up to ~15 kHz**, with the PL raster compensation and a
   parametric extension only above. On realistic chains the parametric model is biased (missing 2nd-order roll-off,
   ripple, a third eddy term): median R_enc 0.77 vs 0.30 nonparametric; its group delay is off by 20–150 ns. Cost: none.
7. **Cross terms**: not measurable by self-term slices (the cross output is an in-plane gradient on a slice ⟂ the test
   axis). Unmeasured they cost R_enc 0.11–0.18 (raw 6–9 via per-encode offsets); a 12-direction recon refit takes this
   to 0.03–0.04. Measuring them needs in-plane resolution: Rahmer / Bacon phase-encode each slice (7×7: 90 min per
   axis). A cheaper variant [hypothesis, not simulated]: two in-plane phase-encode offsets ±k_pe per in-plane axis at
   the slope of the slice's in-plane FT (k_pe ≈ 2.3/R_phantom ≈ 4 cycles/m), where the cross gradient changes the
   magnitude linearly (∂|s|/∂k_c ≈ 0.35 R_phantom ≈ 30 mm lever arm vs 39 mm rms slice lever arm for the self term):
   4 extra frames per slice and input instead of 49. Worth it for the velocity background, not for R.
8. **Per-scan check with the navigator** (rewinder samples, `recon_mask = 0`): one TR's navigator determines a k offset
   to 0.0004–0.0008 cycles/m and the global phase to 0.23 mrad (Fisher, image known) — a per-TR monitor of k0 drift
   (B0 drift with heating, B0-term phase) and of k offsets; aggregated over the scan it fixes the per-axis delays to
   0.01–0.02 ns. Cost: none in scan time (the samples are acquired anyway); a small fit per TR.
9. **Thermal model**: with the assumed coefficients R grows 0.37 per °C of unmodelled warming, almost entirely the
   resonance f0 shift. Without temperature sensors the GIRF should be measured in the thermal state of the flow scan
   (run the flow sequence's own gradient load as the heating block before / between GIRF frames) and the residual f0
   drift refitted (4). A per-°C model H₀ + θ H₁ from heating blocks works if θ is known to ≈ 0.5 °C [CHECK whether
   GE exposes gradient-coil temperatures].
10. Averages beyond 10 min, SNR above 300, more slices: little (nuisance- and window-dominated; §3 and §4). Slice
    thickness [hypothesis, not simulated in the final runs]: at low frequency the H precision ∝ σ_k / k_lim does not
    depend on the thickness (signal ∝ thk, k_lim ∝ 1/thk), at high frequency (slew-limited inputs) it improves ∝ thk,
    but the B0 slice shift −ΔB0/G_ss doubles at 3 mm — with alternating G_ss and noise not limiting, 1.5–2 mm is fine.
11. Pre-emphasis (Stich 2018): for a model error δH the played error is (H Ĥ⁻¹ − 1) g ≈ (H − Ĥ) g, the same as the
    reconstruction with the modelled trajectory to first order; it helps the concomitant / segment structure (rotation
    of one waveform) and the encode blocks, not R.


## 6. Recommended WP2 protocol

For the GE Premier (80 mT/m, 200 T/m/s, 4 µs raster), Pulseq via pge2. Numbers in brackets are what this study
simulated.

- **Phantom**: Ø 16–18 cm sphere, NiCl₂- or CuSO₄-doped water (T1 ≲ 0.3 s, T2 ≳ 100 ms), centred to ±2 mm (localiser),
  receive with the body coil or a known, fixed array combination (the receive-weighted in-plane centroid enters the B0
  terms); 2nd-order shim to ≤ 3 Hz rms over the sphere [2–8 Hz]; centre frequency re-measured before every block.
- **Slices**: per axis 4 at ±8.75 / ±26.25 mm, 1.5 mm, TBW-4 sinc 3 ms (G_ss ≈ 21 mT/m); **slice-select polarity
  alternated every average**; the 12 slices interleaved over axes so each recovers ≥ 1 s.
- **Inputs** (both polarities, order + − − + so a frame's preceding tail cancels in the difference): log chirp
  0.1 → 20 kHz, 40 ms, 0.5 S_max, amplitude capped at |k| ≤ 0.8 k_lim; prephased triangles with ramps 60 / 100 µs at
  0.9 S_max and 160 / 240 µs at 0.55 S_max (SAFE ≤ 77 % on the example hardware; re-check with GE's PNS model in pge2).
- **ADC 200 ms** (narrow resonances; [80 ms fails for ζ ≲ 0.008 near the readout band]), dwell **equal to the imaging
  dwell** (same receiver filter; else the filter's delay difference must be calibrated — or left to the
  reconstruction's delay refit, which absorbs it). Frame ≈ 220 ms.
- **Averages**: 25 per (slice, input, polarity) ≈ 11 min per thermal state, pooled over the states by the thermal
  model (H₀ from all frames). With the 200 ms window averages pay: 66 min of frames gave R_refined median 0.17,
  P(≤ 0.2) = 0.75 vs 0.23 / 0.42 at 11 min (same 24 chains); with 80 ms they do not (nuisance / window limited).
  A 4-slice Hadamard multiband encoding (√4 SNR at equal time [hypothesis, not simulated]) is the cheap way to buy
  the same if the session is short.
- **Thermal plan (~60 min)**: (1) cold set, 11 min; (2) heating with the flow sequence's own gradient load (its TR
  waveform, rotations, duty) until the coil temperature settles (~10 min [CHECK time constant]); (3) "warm" set, 11 min,
  with short heating bursts between frames to hold the state; (4) one higher load (≈ 1.5× the flow scan's RMS) and a set;
  (5) cool-down: a short chirp-only set (one slice pair per axis, ~1 min) every 2–3 min to follow the resonance
  frequencies. Fit H(f; θ) = H₀ + θ H₁ with θ from a first-order thermal model of the dissipated power history
  (sensor logs if GE exposes them). Report the resonance list (f0, ζ, height, their thermal slopes) to the designer.
- **Estimator**: as §1.3 (residual unwrapping along time, increment-domain LS, PL compensation, nonparametric to
  ~15 kHz, parametric above; B0 term with the c·Ĥ_aa cross-talk term).
- **Cross terms**: optional second stage (in-plane phase encoding, or the ±k_pe variant of §5.7) — needed for the
  velocity background, not for R.
- **Per scan**: the reconstruction refits per-axis delays (+ resonance f0 / height, cross / B0 amplitudes when
  available) and monitors k0 / k offsets on the rewinder navigator.


## 7. Proposed code changes (not made; the warm Rust workspace belongs to another agent)

1. `hasty-seq` `girf.rs` — **history**: `played_moments` takes one previous TR; with eddy terms of 10–60 ms the played
   trajectory needs ≳ 100 ms of history (2 vs 10 TRs differ by 0.08 cycles/m rms, about the R = 0.2 level). Sketch:
   ```rust
   /// LTI chain state carried through the whole acquisition (parametric sections exactly, tables by overlap-save)
   pub struct PlayedStream<'a> { girf: &'a Girf, hist: Vec<Vec3>, t: f64 }
   impl PlayedStream<'_> {
       /// append one TR (physical axes) and return its played moments at `times` (from this TR's excitation)
       pub fn push_tr(&mut self, g: &Grad, t_exc: f64, times: &[f64]) -> Vec<Played> {
           self.hist.extend_from_slice(&g.g);
           let keep = (self.girf.tail / g.dt).ceil() as usize + g.len();     // ≥ 5 τ_max of the chain
           if self.hist.len() > keep { self.hist.drain(..self.hist.len() - keep); }
           let off = self.hist.len() - g.len();
           played_moments(self.girf, &self.hist[..off], g, t_exc, times)       // or an IIR state for Parametric
       }
   }
   ```
   (`FlowSchedule` / `simphantom --girf` would use it in acquisition order.)
2. `girf.rs` — **measured responses**: `Girf::from_measurement(path)` reading the nonparametric continuous H (already
   raster-compensated: H = [H_pl − Σ_{m≠0} sinc²(fΔt+m) H_par(f+m/Δt)] / sinc²(fΔt)) per axis, B0 term per axis, the
   parametric extension above f_trust, the resonance list; and `Response::MultiRes` (several resonances on one base),
   used by the designer's `with_resonances`.
3. `hasty-recon` — **refinement of a few chain parameters**: per-axis delay δ_a as a per-sample k shift
   Δk = −γ δ_a g_a(t) (needs the played gradient per sample in the `sched_played` records, or a finite difference of
   γM0), optionally per-resonance (f0, height) and cross / B0 amplitudes through precomputed trajectory derivatives
   (∂k/∂θ per readout, from the GIRF model). Gauss–Newton step on these scalars once per outer iteration, the image
   fixed; this study's numbers are for one linearised step.
4. A thin-slice WP2 sequence writer (Pulseq / pge2) and this estimator in Rust (later; the Python version here is the
   reference).


## 8. Open questions

1. Is there any measured GIRF of a GE Premier (SuperG coil) — resonance frequencies, widths, heights? Everything about
   resonances here is a random assumption, and they decide the outcome.
2. Does GE expose gradient-coil temperatures (service logs / pge2)? Without them the thermal state comes from a power
   model, and the resonance drift (assumed −0.1 %/°C) costs R ≈ 0.37 per °C.
3. pge2: which ADC dwells are possible, and does the receiver's filter delay depend on the dwell? If the GIRF protocol
   can use the imaging dwell, the timing mismatch disappears; otherwise the reconstruction's delay refit has to carry it.
4. Accept a per-axis delay (and resonance) refit inside recon_p2np? It is cheap and the Fisher information is
   enormous; the open part is how it interacts with the image (confounding beyond the affine / per-encode directions).
5. How much scan time is acceptable to keep the readout's main lobe (now 688 Hz) away from the measured resonances?
6. Phantom: which sphere (size, doping) is available, and can the body coil receive?
7. The realistic-chain ranges (§1.2): the 2nd-order roll-off (fc 25–60 kHz) adds 4–10 µs of group delay — possibly
   more than a real chain; a measured GE GIRF would replace all of §1.2.


## References

- Duyn JH, Yang Y, Frank JA, van der Veen JW. Simple correction method for k-space trajectory deviations in MRI.
  J Magn Reson 1998;132:150–153 (thin-slice method; from memory).
- Vannesjo SJ et al. Gradient system characterization by impulse response measurements with a dynamic field camera.
  MRM 2013;69:583–593. https://onlinelibrary.wiley.com/doi/10.1002/mrm.24263 (prediction error < 0.2 %, mechanical
  oscillations visible).
- Campbell-Washburn AE et al. Real-time distortion correction of spiral and echo planar images using the GIRF.
  MRM 2016;75:2278–2285. https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.25788 (phantom, triangle inputs).
- Rahmer J, Mazurkewitz P, Börnert P, Nielsen T. Rapid acquisition of the 3D MRI GIRF using a simple phantom
  measurement. MRM 2019;82:2146–2159. https://onlinelibrary.wiley.com/doi/10.1002/mrm.27902 (thin slices + phase
  encoding into voxel "probes"; chirp input; second-order terms negligible).
- Kronthaler S et al. Trajectory correction based on the GIRF improves high-resolution UTE imaging. MRM
  2021;85:2001–2015. https://onlinelibrary.wiley.com/doi/10.1002/mrm.28566 (Ø 166 mm CuSO4 sphere, 4 slices per axis,
  chirp, TR 2 s).
- Stich M et al. The temperature dependence of gradient system response characteristics. MRM 2020;83:1519–1527.
  https://onlinelibrary.wiley.com/doi/10.1002/mrm.28013 (self terms linear in temperature, B0 term not; 12 vendor
  sensors).
- Stich M et al. Field camera versus phantom-based measurement of the GSTF with dwell time compensation. MRI
  2020;71:125–131. https://pubmed.ncbi.nlm.nih.gov/32534067 (phantom dwell 4.8–8.7 µs; sinc compensation).
- Nussbaum J, Dietrich BE, Wilm BJ, Pruessmann KP. Thermal variation in gradient response: measurement and modeling.
  MRM 2022;87:2224–2238. https://pmc.ncbi.nlm.nih.gov/articles/PMC9303907/ (1 s per frame, 40 ms sweep, 5 Hz
  resolution; resonance shifts ~ linewidths on heating; global thermal model 0.2–0.3 ‰ RMSE).
- Scholten H et al. Fast measurement of the GSTF at 7 T. MRM 2023;89:1644–1659.
  https://onlinelibrary.wiley.com/doi/10.1002/mrm.29523 (shifted excitations after the probing gradient for long-lived
  fields; 2 slices ±16.5 mm, 3 mm).
- Scholten H et al. Phantom-based gradient waveform measurements with compensated variable-prephasing. MRM 2025.
  https://arxiv.org/abs/2409.07203 (prephasing to extend the measurable k range).
- Loecher M, Ennis DB. Phase contrast MRI with minimized background phase errors. MRM 2025;93:1104–1116.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11870205/ (field-camera GIRF, Siemens Vida; prediction within 0.6
  [−3.4, 5.4] mm/s; mechanical oscillations dominate; temperature not modelled).
- Bacon JB et al. Open-source framework for measurement of the GIRF (bioRxiv 2026).
  https://doi.org/10.64898/2026.08.24.745510 (PyPulseq thin-slice + 7×7 PE, 1 mm slices at ±8 / ±16 mm, 5 µs dwell,
  90 min per axis with PE, ~3 min per axis without; vs field probes: RMS trajectory error 2.6–2.8 → 0.10–0.18 cycles/m
  over the first 6.6 ms, a common slow linear trend over 90 ms; B0 phase RMS 0.052 rad).
- GE SIGNA Premier: 80 mT/m, 200 T/m/s, 5 second-order shim coils in the gradient coil (spec sheet,
  https://nyspi.org/sites/default/files/inline-files/SIGNA%20Premier_Spec%20Sheet_FDA%20Cleared_Final%20071717.pdf).
