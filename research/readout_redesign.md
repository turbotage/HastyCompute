# Readout redesign: most of the ADC time on the way out (2026-11-10)

Design files: `out/flow_thorax_v2/` (format of `data/hasty/seq/flow_thorax`), candidates `out/fx_*` (made by
`out/run_candidates2.sh`), the current design replayed `out/flow_thorax_ref`, metrics
`out/readout_study/` (`metrics_table.md`, `metrics.json`, `figs/`), spoiling checks `out/spoilcheck_v2*`. Paths are
relative to the agent copy `rustlib/.agents/seq/`; after the merge they belong under `data/hasty/seq/` and
`data/hasty/seq/studies/`.

## 1. Problem

The current TR (`flow_thorax`, §8–9b of sequence_design.md) is an out-and-back yarn: |k| rises to kmax at 5.58 ms after
the RF centre and comes back to k = 0 at 9.58 ms, followed by the per-encode flow-compensated tail (z spoiler 2 cyc/voxel
plus equal TR M1 for every encode). The return leg was meant to balance the moments. The user's intent is the opposite
weighting: most sampling time on the way out, still about 10 ms of sampling per TR, few off-resonance segments, low
B0 / T2* error.

The measured data say why the out-and-back is the worst case for that. On the simulated 10-min scan (factory_arms/
ph0001, noise-free `data_clean.npy`, 448 readouts × 32 coils; `tools/seq/energy_profile.py`):

| ADC quarter | 0–2 ms | 2–4 ms | 4–6 ms | 6–8 ms |
|---|---|---|---|---|
| share of the data energy | 0.870 | 0.0002 | 0.0002 | 0.129 |

- The high-energy samples sit at both ends of the 8 ms window: k = 0 is visited at τ = 1.6 ms and again at 9.6 ms.
- At the second visit the low-k energy is 0.18–0.41 of the first one at the same |k| (B0 / T2* dephasing of the bulk
  signal over 8 ms). Those 13 % of the energy carry 8 ms of field evolution, so they are the samples most sensitive to
  any field-map, R2* or flow model error.
- Measured: energy-weighted mean τ 2.65 ms and SD 2.65 ms; the "unmodelled-T2*" data error (below) is 10 % for
  myocardium.

## 2. What the reconstruction needs

**Energy profile.** The per-sample energy of the out leg falls as |k|^−2.25 between 0.05 and 0.9 kmax. Per sample and
coil, E/σ² runs from ≈ 3·10⁵ at the centre to ≈ 2 at kmax (σ² = noise_sigma²).
- 99 % of the out-leg energy lies within |k| < 0.084 kmax, i.e. the first 0.11 ms after the echo; 99.9 % lies within
  0.33 kmax (0.42 ms).
- Yet 96 % of all samples are above the per-coil noise (E > σ²).

**Two different "effective windows"** follow from this.

1. *Energy window* (where the information is): the shortest interval holding 99.9 % of the data energy (T_E99.9), and
   the energy-weighted SD of τ (σ_τ).
   - These set how much the data feel a field-map error δf (RMS phase error ≈ 2π δf σ_τ), unmodelled R2*, or flow
     during the readout.
   - Low k must be sampled at the shortest τ, and only once.
2. *Segmentation window* (what the operator must model): the span of the reconstruction samples, T_rec.
   - Every sample above the noise needs a model phase error below ≈ ησ/√E_n. That is 3·10⁻⁴ rad at the centre but only
     ≈ 0.07 rad at kmax (η = 0.1), still far below the 2π·Δf·T of an unmodelled field.
   - So the time-segment rank still scales with span × T_rec.
   - Energy weighting gains only a log term. Rank ≈ B·T + c·log(1/δ_min) over the window holding the tight samples.
     Centre-out holds the tight tolerance only over the first 0.4 ms; the out-and-back holds it at both ends of the
     8 ms window.

**Definition used for L** ("the cycle count of the 1–99 % field over the effective window"): the effective window for
the segment count is T_rec, the full reconstruction span, because 96 % of the samples are above the noise. Two estimates
are given:

- **L_rule** = ⌈1.3·span·T_rec + 3.5⌉: recon_p2np's `--nseg auto`. With the worst-voxel span 1236 Hz (9.86 cycles over
  7.98 ms, from the recon logs) it reproduces today's L = 17, and 21 with fat (the recon logged 22).
- **L_svdw**: the smallest rank of the data-energy-weighted separable fit of e^{i2πΔfτ} whose model-error energy is at
  most η² × the noise energy of the readout's reconstruction samples.
  - Δf is uniform over the span; for water + fat, the span plus its copy shifted by the main fat peak, −434 Hz.
  - η = 0.1 and 0.03.
  - This is `--seg-basis svdw` with the noise as the yardstick, computed from the designed k(τ) and the measured
    spectrum.

Energy model for the candidates: E(|k|, τ) = P(|k|)·D, where
- P is the measured out-leg profile, or ∝ 1/(|k|² + k0²) as a "|k|^−2" check scaled to the same SNR;
- D = exp(−Δτ/4.0 ms) for |k| < 0.1 kmax when the centre is revisited later. 4.0 ms was fitted to the out / return
  ratio; the model gives a return-half share of 0.16 against 0.13 measured.

Other metrics:
- T2* weighting e^{−τ/T2*} at the centre and at kmax: lung 1.5 ms, myocardium 25 ms, blood 40 ms;
- unmodelled-T2* data error √(Σ E (1 − e^{−(τ−TE)/T2*})² / Σ E);
- readout γ|M1| relative to the echo, at the first reach of kmax and energy-weighted;
- N (Nyquist spokes), N·TR·7, readouts / 10 min, density vs |k| (density.npy);
- PNS and limits, moment balance (TR M0, TR M1 across encodes and across rotations).

All of it is computed CPU-side from the designed waveforms (`tools/seq/readout_metrics.py`).

## 3. Candidates

Common to all candidates:
- coprime seven p = 2 (VENC 1.0 / 1.5 m/s), 2 mm, FOV 320 mm, kmax 250 /m, dwell 2 µs;
- example SAFE hardware ≤ 80 %, System::prisma_xr limits (80 mT/m, 200 T/m/s, vector design);
- the same per-encode FlowComp tail;
- the same base-spoke shape as the current optimum (fixed, for an apples-to-apples comparison);
- `--t-adc` bounds the span of the reconstruction samples. Prephasers and rewinders are navigator ADC, flagged in
  `recon_mask.npy` and in `summary.json` `best.recon`.

| id | readout (`seqdesign --leg`) |
|---|---|
| current | out-and-back yarn (no `--leg`) |
| H80 | centre-out yarn to kmax at rest + straight rewind to k = 0 (`rewind`), 8 ms out |
| J65 / J80 / J90 / J95 | centre-out + **M1-nulled rewinder** (`rewind-m1`), 6.5 / 8 / 9 / 9.5 ms out |
| B95 | asymmetric: out 8.4 ms, return leg = the yarn for 4× the spokes (coarse, fast) sampled down to 0.5 kmax, then the M1-nulled rewinder (`asym:4:0.5:m1`) |
| B95c | asymmetric to the centre: return leg for 2× the spokes, all the way back to k = 0, M1-nulled (`asym:2:0:m1`) |
| C95 | two centre-out legs per readout (each ~4.5 ms, rewind in between), the second turned by π (`double`) |
| D80 | in-out: prephaser to −k_end, in through k = 0 (echo in the middle), out, rewind (`inout`) |

Why the rewinder must null M1 (spoilcheck, §4):
- A centre-out readout plus a straight rewind leaves the readout with a net first moment m1 (γ|m1| = 0.89 rad/(m/s) for
  H80).
- It rotates with the spoke, so the TR's net M1 changes every rotation group. The FlowComp block equalises M1 only
  across encodes.
- `Spoke::rewind_m1` brings both M0 and M1 of the readout to zero. It uses two-lobe blocks along the rewind direction
  and along the orthogonal part of the required M1, at a searched split of the vector limits.
- The out-and-back was balanced by symmetry (γ|m1| = 0.026 rad/(m/s)).

## 4. Spoiling check (no phantom; `spoilcheck --vsweep`, blood T1 1.7 s / T2 200 ms, 40 plug-flow velocities × RF 117° / 50°)

| variant | net-M1 spread over rotations (rad/(m/s)) | \|S\|/ideal | TR-to-TR fluct. median / max | velocity error median / 90 % / max (mm/s) |
|---|---|---|---|---|
| G current | 0.013 | 0.982–1.568 | 0.1 % / 0.8 % | 0.00 / 0.03 / 0.10 |
| H centre-out + straight rewind | 1.51 | 0.985–1.215 | 6.5 % / 26 % | 6.5 / 20.6 / 42.5 |
| **J centre-out + M1-nulled rewind** | 5·10⁻¹⁴ | 0.986–1.291 | 0.1 % / 0.1 % | 0.00 / 0.00 / 0.20 |
| I asymmetric 4×:0.5 + straight rewind | 0.54 | 0.983–1.045 | 4.1 % / 13 % | 2.3 / 9.1 / 20.1 |

- The same holds 10 cm off-centre: G 0.14, H 42.5, J 0.16, I 20.1 mm/s max.
- J re-run with the final rewinder on the tail's slew factor (`out/spoilcheck_v2b*`): N 1081, TR 12.60 ms; fluctuation
  0.1 % / 0.1 %, velocity error max 0.19 mm/s at isocentre and 0.16 mm/s 10 cm off-centre.
- H and I are rejected: a rotation-dependent net M1 gives random TR-to-TR modulation for flowing spins, which no model
  captures (the decision criterion of §9b).

## 5. Metrics (measured spectrum unless marked; full table in `out/readout_study/metrics_table.md`)

| metric | current | H80 | J65 | J80 | J90 | **J95** | B95 | B95c | C95 | D80 |
|---|---|---|---|---|---|---|---|---|---|---|
| TR ms | 11.29 | 11.88 | 11.14 | 12.65 | 13.66 | 14.15 | 13.82 | 13.52 | 13.37 | 12.41 |
| readouts / 10 min | 53144 | 50505 | 53859 | 47430 | 43923 | 42402 | 43415 | 44378 | 44876 | 48348 |
| reconstruction window ms | 7.98 | 8.0 | 6.5 | 8.0 | 9.0 | 9.5 | 9.5 | 9.5 | 9.5 (8.96 sampled) | 8.0 |
| ADC per TR ms (incl. navigator) | 7.98 | 8.57 | 7.83 | 9.34 | 10.35 | 10.84 | 10.51 | 10.21 | 10.06 | 9.10 |
| reconstruction duty / ADC duty | 0.71 / 0.71 | 0.67 / 0.72 | 0.58 / 0.70 | 0.63 / 0.74 | 0.66 / 0.76 | 0.67 / 0.77 | 0.69 / 0.76 | 0.70 / 0.76 | 0.67 / 0.75 | 0.65 / 0.73 |
| echo (k = 0) after RF ms | 1.6 (+9.6) | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 (+11.1) | 1.6 (+6.7) | 6.14 |
| N (Nyquist spokes) | 1104 | 1081 | 1345 | 1081 | 957 | 905 | 1094 | 918 | 1028 | 1160 |
| N·TR·7 s (one Nyquist set, 7 encodes) | 87.2 | 89.9 | 104.9 | 95.7 | 91.5 | 89.7 | 105.8 | 86.9 | 96.3 | 100.8 |
| density / Nyquist at 0.25 / 0.5 / 0.75 kmax | 1.09 / 1.01 / 1.01 | 1.02 / 1.00 / 1.00 | 1.03 / 1.00 / 1.01 | 1.02 / 1.00 / 1.00 | 1.02 / 1.01 / 1.01 | 1.02 / 1.01 / 1.00 | 1.02 / 1.12 / 1.24 | 1.07 / 1.00 / 1.00 | 1.08 / 1.01 / 1.00 | 1.10 / 1.01 / 1.00 |
| τ at \|k\| < 0.05 kmax ms | 1.6–1.7, 9.5–9.6 | 1.6–1.7 | 1.6–1.7 | 1.6–1.7 | 1.6–1.7 | 1.6–1.7 | 1.6–1.7 | 1.6, 11.1 | 1.6, 6.7 | 6.1–6.2 |
| τ at \|k\| ≈ 0.5 kmax ms | 2.3, 8.9 | 2.8–3.5 | 2.6–3.2 | 2.8–3.5 | 3.0–3.7 | 3.0–3.8 | 2.8, 11.1 | 2.6, 10.5 | 2.4, 7.7 | 5.2, 7.1 |
| τ at \|k\| > 0.95 kmax ms | 5.1–6.1 | 8.5–9.6 | 7.2–8.1 | 8.5–9.6 | 9.4–10.6 | 9.8–11.1 | 8.5–9.6 | 7.1–8.2 | 5.4–11.1 | 2.1–10.1 |
| E-weighted SD of τ ms (meas / \|k\|⁻²) | 2.89 / 3.06 | 0.15 / 0.86 | 0.11 / 0.64 | 0.15 / 0.86 | 0.18 / 1.02 | 0.19 / 1.10 | 0.21 / 1.22 | 2.83 / 3.09 | 2.11 / 2.26 | 0.17 / 0.88 |
| 99.9 %-energy window ms (meas) | 7.98 | 1.99 | 1.06 | 1.99 | 2.71 | 3.07 | 3.06 | 9.49 | 5.41 | 5.37 |
| cycles over T_rec / T_E99.9 at 100 Hz | 0.80 / 0.80 | 0.80 / 0.20 | 0.65 / 0.11 | 0.80 / 0.20 | 0.90 / 0.27 | 0.95 / 0.31 | 0.95 / 0.31 | 0.95 / 0.95 | 0.95 / 0.54 | 0.80 / 0.54 |
| … at 300 Hz | 2.39 / 2.39 | 2.40 / 0.60 | 1.95 / 0.32 | 2.40 / 0.60 | 2.70 / 0.81 | 2.85 / 0.92 | 2.84 / 0.92 | 2.85 / 2.85 | 2.85 / 1.62 | 2.40 / 1.61 |
| … at 500 Hz | 3.99 / 3.99 | 4.00 / 0.99 | 3.25 / 0.53 | 4.00 / 0.99 | 4.50 / 1.36 | 4.75 / 1.53 | 4.74 / 1.53 | 4.75 / 4.74 | 4.75 / 2.71 | 4.00 / 2.68 |
| … fat (434 Hz) | 3.46 / 3.46 | 3.47 / 0.86 | 2.82 / 0.46 | 3.47 / 0.86 | 3.90 / 1.18 | 4.12 / 1.33 | 4.11 / 1.33 | 4.12 / 4.12 | 4.12 / 2.35 | 3.47 / 2.33 |
| **L water, 1–99 % (488 Hz): rule / svdw η 0.1 / 0.03** | 9 / 7 / 8 | 9 / 6 / 7 | 8 / 6 / 6 | 9 / 6 / 7 | 10 / 7 / 8 | 10 / 7 / 8 | 10 / 7 / 8 | 10 / 8 / 9 | 10 / 7 / 8 | 9 / 6 / 7 |
| **L water + fat, 1–99 %** | 14 / 11 / 12 | 14 / 10 / 11 | 12 / 9 / 10 | 14 / 10 / 11 | 15 / 11 / 12 | 15 / 11 / 12 | 15 / 12 / 13 | 15 / 12 / 13 | 15 / 11 / 12 | 14 / 9 / 10 |
| L water, worst voxel (1236 Hz) | 17 / 13 / 14 | 17 / 13 / 14 | 14 / 11 / 12 | 17 / 13 / 14 | 18 / 14 / 15 | 19 / 15 / 16 | 19 / 15 / 16 | 19 / 15 / 16 | 19 / 15 / 16 | 17 / 12 / 13 |
| L water + fat, worst voxel | 21 / 17 / 18 | 21 / 16 / 17 | 18 / 14 / 15 | 21 / 16 / 17 | 24 / 18 / 19 | 25 / 19 / 20 | 25 / 19 / 20 | 25 / 19 / 20 | 25 / 19 / 20 | 21 / 15 / 16 |
| T2* centre / kmax: myocardium | 0.94 / 0.80 | 0.94 / 0.70 | 0.94 / 0.74 | 0.94 / 0.70 | 0.94 / 0.67 | 0.94 / 0.66 | 0.94 / 0.70 | 0.94 / 0.74 | 0.94 / 0.72 | 0.78 / 0.79 |
| T2* centre / kmax: blood | 0.96 / 0.87 | 0.96 / 0.80 | 0.96 / 0.83 | 0.96 / 0.80 | 0.96 / 0.78 | 0.96 / 0.77 | 0.96 / 0.80 | 0.96 / 0.83 | 0.96 / 0.82 | 0.86 / 0.86 |
| T2* centre / kmax: lung | 0.34 / 0.025 | 0.34 / 0.002 | 0.34 / 0.006 | 0.34 / 0.002 | 0.34 / 0.001 | 0.34 / 0.001 | 0.34 / 0.002 | 0.34 / 0.006 | 0.34 / 0.011 | 0.017 / 0.099 |
| unmodelled-T2* data error: myocardium / blood / lung (meas) | 0.109 / 0.072 / 0.40 | 0.006 / 0.004 / 0.045 | 0.004 / 0.003 / 0.041 | 0.006 / 0.004 / 0.045 | 0.007 / 0.004 / 0.048 | 0.007 / 0.005 / 0.050 | 0.008 / 0.005 / 0.048 | 0.100 / 0.067 / 0.32 | 0.087 / 0.057 / 0.46 | 0.007 / 0.004 / 0.24 |
| same, \|k\|⁻² spectrum | 0.118 / 0.078 / 0.45 | 0.032 / 0.021 / 0.19 | 0.024 / 0.016 / 0.16 | 0.032 / 0.021 / 0.19 | 0.037 / 0.024 / 0.21 | 0.040 / 0.026 / 0.22 | 0.044 / 0.029 / 0.21 | 0.112 / 0.075 / 0.39 | 0.095 / 0.062 / 0.49 | 0.036 / 0.022 / 1.37 |
| readout γ\|M1\| at kmax / E-RMS, rad/(m/s) | 8.5 / 0.33 | 14.4 / 0.26 | 12.1 / 0.21 | 14.4 / 0.26 | 15.9 / 0.30 | 16.7 / 0.32 | 14.3 / 0.33 | 11.9 / 0.41 | 9.1 / 0.55 | 3.6 / 0.81 |
| readout net γ\|M1\| (rotates with the spoke) | 0.026 | **0.89** | 0 | 0 | 0 | 0 | 0 | 0 | **0.30** | 0.029 |
| PNS % (part) | 79.8 (readout) | 79.8 (ro) | 79.8 (ro) | 79.8 (ro) | 79.8 (ro) | 79.8 (ro) | 79.8 (ro) | 79.9 (ro) | 79.3 (enc) | 79.5 (enc) |
| slew factors enc / ro / tail | .276/.447/.375 | .276/.444/.374 | .275/.445/.373 | .275/.444/.373 | .275/.444/.373 | .276/.444/.374 | .274/.444/.374 | .275/.445/.374 | .276/.426/.374 | .275/.418/.375 |
| \|G\| / \|S\| max (readout incl. rewinder) | 22.6 / 90 | 22.6 / 91 | 27.5 / 90 | 27.7 / 91 | 27.4 / 91 | 28.3 / 91 | 22.7 / 91 | 22.6 / 90 | 22.1 / 86 | 21.8 / 84 |
| TR net k / M1 spread across encodes | 1000 /m / 3e-14 | same | same | same | same | same | same | same | same | same |

Readings:
- **B0 / T2* / flow error of the data** (energy-weighted): centre-out (H, J) samples all low k within 0.1 ms of the echo.
  - σ_τ drops from 2.9 ms to 0.15–0.19 ms (×15–20; ×3 with the flatter |k|⁻² spectrum).
  - The unmodelled-T2* data error drops ×15–25 (myocardium 11 % → 0.6 %).
  - The 99.9 %-energy window drops from 8 ms to 2–3 ms.
  - A 10 Hz field-map error gives an RMS data phase error of 0.18 rad today against 0.01 rad.
- **Segments follow the reconstruction window**, not the energy.
  - At equal window (J80 vs current) the rule gives the same L (9 / 14 / 17 / 21). The weighted fit gives one segment
    fewer for centre-out (svdw η 0.1: 6 / 10 / 13 / 16 against 7 / 11 / 13 / 17), and one more for designs that revisit
    the centre late (B95c).
  - Each further ms of window costs ≈ +0.6 (1–99 %) to +1.6 (worst voxel + fat) segments.
- **Undersampling per unit time.**
  - The out-and-back gets its return leg's coverage for free.
  - Centre-out pays for the rewinder (1.34 ms with M1 nulling, 0.57 ms straight) and amortises it with a longer out
    leg: N·TR·7 is +9.7 % at 8 ms, +4.9 % at 9 ms, +2.9 % at 9.5 ms (vs current).
  - Readouts per 10 min drop by 11 % / 17 % / 20 %.
- **T2* at kmax** is lower for the long centre-out legs: myocardium 0.66 against 0.80, blood 0.77 against 0.87. It is
  modelled by R2* in the operator and costs high-k SNR / resolution for short T2*.
  - The readout γ|M1| at kmax doubles (16.7 against 8.5 rad/(m/s)). The velocity-parametrised operator models it; for
    a per-encode image model the jets blur more at high k.
- B95 (fast coarse return to 0.5 kmax): its return only adds density above 0.5 kmax (1.12–1.24 × Nyquist), so N·TR gets
  worse (105.8 s).
- B95c (return to the centre): N·TR as today, but the late centre visit brings back the current design's error profile.
- C95: late second centre and a non-zero net M1.
- D80 (in-out):
  - It is M1-balanced by symmetry, has the narrowest energy window around its echo, and the lowest svdw L at its length.
  - But the echo moves to TE 6.1 ms (myocardium 0.78, lung 0.02 at the centre), N·TR is +16 %, and the readout M1 at
    the echo includes the prephaser (rotation-dependent velocity offset, common to all encodes).
  - Not recommended for the thorax; possibly interesting where T2* is long.

## 6. Recommendation

**Centre-out yarn + M1-nulled rewinder (navigator ADC) + the unchanged per-encode FlowComp tail** (`--leg rewind-m1`),
with a reconstruction window of 9.5 ms: `out/flow_thorax_v2`.

Why:
- Every reconstruction sample is on the way out. The k-centre is sampled once, at TE = 1.6 ms; nothing returns to the
  centre late.
- The data's sensitivity to field-map, R2* and flow model errors falls 15–25× (measured spectrum).
- About 10 ms of sampling per TR: 9.5 ms reconstruction + 1.3 ms navigator (the rewinder, ending at k = 0: a second DC
  sample per coil at τ ≈ 12.4 ms for field drift / respiratory B0).
- Undersampling per unit time within 3 % of today (N·TR·7 89.2 s against 87.2 s), reconstruction duty 0.67 (0.71
  today; SNR −3 %).
- RF spoiling with flow is as good as today: the readout's net M1 is exactly zero for every rotation, so the TR's net M1
  is the same in every TR.
- PNS 79.8 % on the example hardware. Peak |G| ≤ 46 mT/m anywhere in the TR (below the 70–80 mT/m of GE Premier /
  Prisma XR); slew ≤ 91 T/m/s.

Cost: one more time segment than today by the rule (water 1–99 % 10 against 9; worst voxel 19 against 17). With the
svdw basis the 1–99 % count is equal to today's (7, η 0.1).

If segments or readouts per minute matter more than coverage per minute, the same design at an **8 ms window (J80)** has
today's segment count by the rule (and −1 with svdw) and 12 % more readouts than J95, for N·TR +9.7 % against today. The
window length is the one free parameter; it should be settled by the phantom study (§8), as planned for T_adc.

Rejected:
- H, the straight rewind: rotation-dependent net M1 breaks RF spoiling for flow (up to 42 mm/s, 26 % fluctuation).
- B (asymmetric return): no coverage gain where it matters (below the return radius); returning to the centre
  re-creates the problem.
- C (double centre-out): late second centre, net M1.
- D (in-out): long TE, N·TR.

Possible improvement (not implemented):
- Merge the readout's M1 compensation into a per-rotation FlowComp block, instead of nulling it in the rewinder
  (`optim::merged_tail_study`). Straight rewinder 0.57 ms + per-rotation block 1.85 ms (longest of 64 rotations) against
  M1-nulled rewinder 1.34 ms + block 1.56 ms saves 0.48 ms / TR (≈ 3.4 %).
- It needs rotation-dependent tails: Tr::build per rotation, FlowSchedule per-rotation tail waveforms, and the GIRF eddy
  model per rotation.
- A waveform-domain (QP) M1-nulled rewinder under vector limits and PNS would likely shorten the 1.34 ms as well.

### 6b. Final design `out/flow_thorax_v2`

Command:

    seqdesign out/flow_thorax_v2 --t-adc 9.5e-3 --scheme coprime7 --p 2 --v-ext 3 --return 0 --spoil flowcomp \
        --spoil-cycles 2 --leg rewind-m1 --iters 40 --threads 2 --check

Shape search: grid of 30 + Nelder–Mead, 98 evaluations. Optimum near-planar (k_g0 0.005, wobble 0.098, ω 2.17).

| | current `flow_thorax` | `flow_thorax_v2` |
|---|---|---|
| TR / TE | 11.29 / 1.60 ms | 14.10 / 1.60 ms |
| encode / readout / tail | 1.52 / 7.98 / 1.56 ms | 1.52 / 9.49 reconstruction + 1.30 navigator rewinder / 1.56 ms |
| ADC samples per TR (2 µs) | 3990 | 5395, of which 4745 reconstruction |
| readouts / 10 min, rotations | 53 144, 7592 | 42 553, 6079 |
| N (design), N* (from the actual k) | 1104, 1119 | 904, 923 |
| N·TR·7 | 87.2 s | 89.2 s (+2.3 %) |
| reconstruction duty | 0.71 | 0.67 |
| PNS (example SAFE), slew factors enc / ro / tail | 79.8 % readout, .276 / .447 / .375 | 79.9 % readout, .274 / .444 / .372 |
| peak \|G\| readout incl. rewinder / whole TR; \|S\| | 22.6 / 45.2 mT/m; 90 T/m/s | 27.8 / ≤ 46 mT/m; 91 T/m/s |
| \|k\| ≤ kmax over 3200 rotations (readout incl. rewinder) | 0.9992 | 0.9999 (0 samples outside) |
| net readout γ\|M1\| (rotation-dependent part of the TR's M1) | 0.026 rad/(m/s) | 7·10⁻¹⁵ |
| TR M0 / M1 across encodes | 1000 /m constant / 3·10⁻¹⁴ | 1000 /m constant / 9·10⁻¹⁴ |
| readout γM1 at first reach of 10 / 25 / 50 / 99 % kmax | 0.014 / 0.074 / 0.71 / 6.1 | 0.013 / 0.16 / 1.4 / 14.2 rad/(m/s) |
| concomitant anisotropy, phase scale at 15 cm (end of readout) | 0.70, 0.96 rad | 0.71, 1.27 rad |
| GIRF (illustrative clinical chain) max \|Δk\| over the ADC | 3.4 cells | 4.1 cells |
| coverage: empty outer Nyquist cells (full set / 1/8 subset) | 0.243 / 0.840 | 0.236 / 0.838 |
| E-weighted SD of τ; 99.9 %-energy window | 2.89 ms; 7.98 ms | 0.19 ms; 3.07 ms |
| unmodelled-T2* data error myocardium / blood / lung | 0.109 / 0.072 / 0.40 | 0.007 / 0.005 / 0.050 |
| L water 1–99 %: rule / svdw | 9 / 7 | 10 / 7 |
| L water + fat 1–99 %: rule / svdw | 14 / 11 | 15 / 11 |
| L worst voxel (water; + fat): rule / svdw | 17 / 13; 21 / 17 | 19 / 15; 25 / 19 |

Encoding unchanged: the moments at TE and the pair VENCs (1.0 / 1.5 m/s) come out exactly as before (seqdesign
asserts them).

Checks:
- seqdesign's own checks: limits, PNS over 3200 acquisition rotations, kmax, moments at TE, pair VENC assertion.
- `tools/seq/check_pypulseq.py`: SAFE PNS identical to pypulseq to 1.5·10⁻¹⁵ relative on 3 rotations × 7 encodes
  (79.12 %); moments agree with fine-grid integration (M1 rel. 4·10⁻¹¹). All OK.
- `spoilcheck --vsweep G,J` (§4).
- hasty-seq unit tests: 34 pass; hasty-bench `seqsched` tests (FlowSchedule on a FlowComp TR) pass.
- Regression: the current design replayed from its cache is byte-identical (spoke_grad, tr_grad, k_adc, density,
  coverage).

Figures: `out/flow_thorax_v2/figs/final_*.png` and the seqdesign figures.

## 7. What changes for the reconstruction

- **Sample selection.**
  - The ADC window now holds the reconstruction part and a navigator part. `summary.json` `best.recon` gives the
    raster ranges ([[0, 949]] for v2, 10 µs raster) from the readout start; `recon_mask.npy` gives the same per raster segment.
  - The simulator records the whole window (FlowSchedule unchanged: `i_adc` is the full spoke). The reconstruction must
    drop samples with τ − TE ≥ t_recon (or use them as a navigator).
  - `--nseg auto` must count only the reconstruction window. Today it takes `crop_adc.window` or the whole ADC.
- **Segment count.** Equal window → equal L_rule. The 9.5 ms window → +1 (1–99 %) / +2 (worst voxel). `--seg-basis
  svdw` (data-energy time weights) gains one segment more for centre-out than for the out-and-back.
  - The pyramid's `--kcrop prefix` (outgoing part only) is now the whole reconstruction window. A coarse level can
    simply end the window at its own kmax, since τ(|k|) is monotone: L_coarse ∝ τ(k_coarse) ≈ T_rec·(k/kmax)^2.5.
    That is a large saving for coarse levels: 0.5 kmax is reached at 3.0–3.8 ms.
- **Density / preconditioning.** One pass per spoke instead of two: the centre is less oversampled (radial region once),
  shells are at Nyquist with N ≈ 905 rotations per set (instead of 1104 half-coverage pairs). Circulant / P²nP
  preconditioners are recomputed from the new k; no DCW.
- **Timing.** TE unchanged (1.6 ms).
  - kmax is sampled at 9.8–11.1 ms after RF (5.1–6.1 ms today). The R2* map matters more at high k; T2* weights at
    kmax are in the table.
  - The readout γ|M1| at kmax is 16.7 rad/(m/s): per-sample M1 in the velocity-parametrised operator matters at high k.
  - The concomitant P(τ) grows monotonically over 9.5 ms (it is not a return to small k). Its energy-weighted effect is
    smaller, but the late-τ concomitant ranks per rotation grow ∝ T_rec, the same scaling as today.
- **Navigator data.** Rewinder samples stay inside kmax (two-lobe path, dips to ≈ 0.3 kmax, then back to the centre)
  and end at k = 0 at τ ≈ 12.4 ms after RF.
  - Per coil and TR that is a late DC sample: field drift / respiratory B0, T2* of the bulk.
  - The encode excursions stay navigator data as before.
  - The two-lobe rewinder has higher slew-time content than a straight line; its eddy-current tail into the FlowComp
    block and the next TR is modelled through `girf::played_moments` (previous TR included).
- **Readouts.** 42 553 readouts in 10 min (53 144 today), 6079 rotations of 7 encodes. Each readout covers 22 % more
  curve (one Nyquist set per 89.2 s for 7 encodes).
  - Data volume: 5395 ADC samples per readout, 2.30·10⁸ in total (+8 %), or 2.02·10⁸ (−5 %) if the navigator samples
    are not stored.

## 8. To re-simulate later (not done now)

1. The 10-min flow scan on factory_arms/ph0001 with `flow_thorax_v2`, and the same at J80 (window study). Same physio,
   field and fat. GIRF ideal and one illustrative chain.
2. Reconstruct both, and today's design, with `--nseg auto` restricted to the reconstruction window, and with svdw.
   Compare:
   - the data-space model error vs noise at equal L;
   - image and velocity error at the coronaries / great vessels;
   - with and without R2* in the operator, and with a perturbed field map (δf 10–20 Hz, respiratory).
   This is where the T_rec choice (8 vs 9.5 ms) gets decided.
3. The navigator: estimate per-TR global frequency from the late DC sample and check it against the simulated
   respiratory B0.
4. Spoiling was checked with the J readout at 8 ms (§4). Repeat with v2's 9.5 ms window and its searched shape (a
   small `spoilcheck` change: take the design from a run directory). Unmodelled as before: laminar velocity spread,
   pulsatility, inflow, acceleration (M2), diffusion.
5. GIRF-robust variants (`--encode/--tail tuned=…`) and `--res-girf` re-run on the new readout once a GIRF is measured.

## 9. Open questions

**Answered by the user (2026-11-10):**
- Window: **9.5 ms (v2)** is the default for the later resimulation (J80 stays a documented alternative).
- Rewinder samples: **keep as navigator** (stored, `recon_mask.npy` excludes them from the reconstruction).
- Per-rotation FlowComp tails: **not now** (keep the rotation-independent tail; simpler GIRF / eddy model).

Original questions (kept for context):

- Window length: 9.5 ms (≈ today's coverage per minute, +1–2 segments) or 8 ms (today's segment count, 12 % more
  readouts, −9 % coverage per minute)? The phantom study above decides; the defaults here are a judgement.
- Is the late rewinder DC sample wanted as a navigator (per-TR B0 drift), or should the ADC stop at the end of the
  reconstruction window (less data to store: ≈ 12 % of the ADC samples)?
- Per-rotation FlowComp tails (−0.48 ms / TR) are worth implementing only if the rotation-dependent eddy-current history
  is acceptable for the GIRF model. Is it?
- GE specifics not modelled:
  - GE's own PNS model (the example SAFE hardware binds every part here);
  - the 4 µs gradient raster (design raster is 10 µs);
  - receiver limits (max samples per ADC window / readout, ±250 kHz at 2 µs dwell).
  The Premier's slew / amplitude limits do not bind (peak 46 mT/m, 91 T/m/s).

## 10. Code (additive; existing designs reproduce byte-for-byte)

- `hasty-seq::traj`:
  - `Spoke::{recon, i_echo}` (reconstruction ranges, echo index);
  - `t_recon`, `t_recon_samples`, `is_recon`, `push`, `rewind` (straight), `rewind_m1` (M0 and M1 nulled), `m1_end`;
  - `Yarn::{curve_asym, curve_in_out, n_eff_asym}`; `smooth_kink` (shared, new constructions only; near-duplicate
    junction points removed).
- `hasty-seq::optim`:
  - `Leg {Default, Rewind, RewindM1, Asym{back, ret, null_m1}, Double, InOut}` and `Problem::leg`;
  - `spoke_rw` / `fit_budget_rw`: the final rewinder at the tail's slew factor, designed once after the N bisection;
  - `budget_time` (reconstruction span for legs);
  - tests `leg_tests` (rest at k = 0, recon inside kmax, limits, curve_asym ≡ curve_return for equal legs, FlowComp
    with constant M0 and equal M1, M1 equal across rotations with `RewindM1`);
  - `merged_tail_study` (ignored).
- `hasty-seq::tr`: `Tr::part` counts a leg readout's final rewinder as the tail (PNS derating).
- `seqdesign`:
  - `--leg rewind|rewind-m1|asym:B:R[:m1]|double|inout`;
  - `--shape kg0,wobble,freq` (no search);
  - `recon_mask.npy`; `summary.json` `best.{leg, recon, i_echo, t_recon, t_recon_samples, te_echo,
    readout_gamma_m1_end}`;
  - density / coverage over the reconstruction samples (legs);
  - the fingerprint is unchanged for default designs.
- `spoilcheck`: variants H, J, I; `--vsweep` designs only the selected variants.
- `adcstudy`: the new `Problem` field.
- `tools/seq/energy_profile.py`, `tools/seq/readout_metrics.py` (new).
