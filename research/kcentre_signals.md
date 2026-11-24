# k-centre signals of the 10-min flow scan (ph0001, sim_flow_10min)

2026-11-23. CPU only. Scripts: `research/experiments/kcentre/{extract,predict,state_variants,analyze}.py`.
Arrays: `data/hasty/phantom/factory_arms/ph0001/kcentre/` (k0.npz, pred.npz, pred_state_variants.npz, raw_first2048.npz, numbers.json).

## Data and what was extracted

- **Data used:** `data_clean.hv` (noise-free). No noisy data file exists. The recon adds Ψ-correlated noise on load
  (σ = 39.98 per coil, `data.rs`). I added noise from the same model with my own RNG as a check. It changes nothing
  that matters here: the implied Δf moves by < 0.01 Hz, and by +0.2 Hz with the fit.
- **ADC inventory:** each readout has 3990 samples, all inside the readout. They run from 1.600 to 9.578 ms after
  the excitation, with a 2 µs dwell. The simulator records **no ADC during the velocity encoders or the spoiler/tail**
  (no navigator ADC). That part of the task is therefore skipped.
- **k units:** rad/m (k = γM0, max 1570 = 2π·250). One FOV step is 2π/FOV_max = 2π/0.508 m = 12.4 rad/m.
- **Out-centre (start of the readout):** samples 0–11, t = 1.600–1.622 ms, |k| = 0, 0.05, 0.19, … 5.75 rad/m
  (≤ 0.47 FOV steps). Sample 0 is exactly k = 0.
- **In-centre (end of the return half):** samples 3969–3989, t = 9.538–9.578 ms, |k| from 5.3 down to 0.005 rad/m.
  The last 15 samples (30 µs) stay at |k| ≤ 0.27 rad/m.
- **k = 0 values per readout and coil** (53 144 × 32 × 2):
  - `exact`: sample 0 and sample 3989.
  - `fit`: the intercept of a least-squares fit s = a + g·k over each window.

  On clean data the fit differs from the exact samples by 1.8 % (out; the window is too wide for a linear model) and
  0.33 % (in). With noise, the single exact sample has 0.19 % (out) and 0.67 % (in) relative error. The analysis
  below uses `exact`.

## Prediction model (for comparison only)

The prediction uses the simulator's own sign convention:

s_c(τ) = Σ_x S_c(x) W(x) [pd_w e^{−R_w τ} + pd_f F(τ) e^{−R_f τ}] e^{−i2πΔf τ} e^{−i(w·q + m1·v)}

- F is the six-peak fat model, the same one the simulator uses.
- W is the spoiled-GRE steady-state weight. It uses T1 from the source list and flip 12°. The slab profile is the
  small-tip Fourier transform of the Hamming sinc, displaced by off-resonance.
- w·q is the concomitant-field phase, with w taken per readout from `sched_nominal` at samples 0 and 3989.
- m1·v is the flow phase, using `truth_vel_logical` for the readout's cardiac frame.

The variants step from the static truth maps to per-state maps, then add the concomitant and flow terms. The per-state
maps come from `truth_ratemap_bc` (6 breathing × 20 cardiac states); each readout gets its nearest state. The
concomitant and flow terms are computed on a subset of 3325 readouts (every 16th rotation group, all 7 encodes).

Approximations:
- W comes from the template positions; the tissue is not moved.
- Blood uses the steady-state signal; the simulator uses the LBM spin history.
- The voxel-mean velocity is applied to the whole 2 mm voxel.
- Dephasing within a voxel is not modelled.

## Numbers

Δt = 7.978 ms. Coil-combined ratio ρ = Σ s_in·conj(s_out) / Σ |s_out|².
Implied Δf = −arg ρ / (2πΔt); effective decay rate = −ln|ρ| / Δt.

| source | implied Δf (Hz) | effective decay rate (1/s) |
|---|---|---|
| **measured** (all readouts) | **5.97** | **330** |
| static maps, PD only | 7.0 | 374 |
| static maps, steady-state weight | −0.3 | 309 |
| per state (breathing × cardiac), no concomitant/flow | −1.0 | 390 |
| per state + concomitant | 8.7 | 357 |
| per state + flow | −1.8 | 384 |
| **per state + concomitant + flow** | **7.8** | **353** |
| static maps, water only (PD / steady-state weight) | 41.5 / 41.2 | 329 / 318 |
| truth, signal-weighted mean (per state, W·Σ_c\|S_c\|²) | Δf 28.2 | decay rate 37 |

- **The in/out ratio is not a direct B0 or decay-rate readout here.** Fat is 54 % of the signal weight. The
  water–fat beating moves the implied Δf from about 41 Hz (water only) to about 7 Hz. The concomitant field at
  9.58 ms adds about +9.7 Hz equivalent (the z² coefficient grows from 19 to 121 rad/m²).
- **The effective decay rate (330 /s) is about 9× the true mean decay rate (37 /s).** The magnitude loss is
  dephasing: water–fat beating, B0 spread over the slab, and the quadratic concomitant phase. It is not decay.
- **The full model reproduces the measurement:** 7.8 vs 6.0 Hz and 353 vs 330 /s.
- **Per-readout implied Δf:** measured mean 5.9 Hz, standard deviation 1.5 Hz. The full prediction explains R² = 0.85
  of the per-readout variation. That variation is concomitant phase, which depends on the readout rotation. The
  measured spread is about 0.6× the predicted spread.
- **Coil-map consistency:** fit one complex scale per readout between the measured coil vector and the predicted one;
  the median relative residual is:

  | centre | full model | per state only | static weighted map |
  |---|---|---|---|
  | out | 5.7 % | 7.9 % | 12.9 % |
  | in | 7.8 % | 23 % | 22 % |

**Breathing:**
- The out-centre signal tracks breathing strongly per coil (reference encode). Median R² with the bellows is 0.57 for
  |s_out| and 0.35 for arg s_out; the best coil reaches 0.94 and 0.96. Cardiac phase (2-harmonic fit) gives median
  R² 0.33 for |s_out| and 0.58 for arg s_out; the best coil reaches 0.96 and 0.90.
- The in-centre per-coil signals hardly correlate with breathing (median R² about 0.02). Their readout-to-readout
  variation is dominated by the rotation-dependent concomitant phase at 9.58 ms.
- **The implied Δf does not vary with breathing:** R² with the bellows is 0.0002 and the slope is 0.04 Hz per unit r.
  The truth-based predictions give +1.6 Hz/r (full) and +2.2 Hz/r (per state). The truth signal-weighted mean Δf
  itself changes from 28.8 to 27.1 Hz over r = 0 → 1.4.
- **This discrepancy is not explained.** One cause is ruled out: the steady-state weight map. Predictions with
  W = 1 (+4.4 Hz/r) and water only (+3.2 Hz/r) also have a slope. I did not chase it further.
- The coil-combined implied Δf of the reference encode correlates with cardiac phase (R² 0.46).

**Encodes:** the coil-combined k-centre phase of each encode, relative to encode 0 in the same rotation group. All
values are in rad and small, because static tissue dominates k = 0.

| encode | out: measured | predicted full | flow only | concomitant only | in: measured | predicted full |
|---|---|---|---|---|---|---|
| 1 | +0.002 | −0.000 | −0.007 | +0.007 | +0.021 | +0.021 |
| 2 | +0.024 | +0.022 | −0.003 | +0.024 | +0.033 | +0.033 |
| 3 | −0.012 | −0.013 | −0.000 | −0.013 | +0.007 | +0.008 |
| 4 | +0.016 | +0.015 | +0.003 | +0.011 | +0.025 | +0.025 |
| 5 | −0.006 | −0.008 | −0.004 | −0.004 | +0.032 | +0.033 |
| 6 | +0.030 | +0.030 | +0.003 | +0.026 | +0.055 | +0.055 |

- Measured and predicted agree to within 0.002 rad.
- The **concomitant phase from the encoder** (cross terms xz and yz, Δw up to 18.7 rad/m²) is most of the difference.
  Flow is |Δφ| ≤ 0.007 rad: typically 2–20 % of the difference, up to about 50 % where the concomitant part is small
  (encode 5 at the out-centre, encode 3 at the in-centre). In encode 1 at the out-centre, flow and concomitant cancel.

## Figures (`research/experiments/kcentre/figs/`)

- `a_ratio_phase.png`: per-coil phase of s_in/s_out, measured vs four prediction variants, with the coil-combined
  implied Δf as bars.
- `b_ratio_magnitude.png`: per-coil |s_in/s_out| (log–log), and the implied effective decay rate next to the truth
  mean decay rate (37 /s).
- `c_timeseries.png`: magnitude and phase at the out and in centres, for 3 coils, over the first 120 s of the
  reference encode, coloured by the true breathing state.
- `c2_correlations.png`: per-coil R² of the four k-centre features with the bellows and with cardiac phase.
- `d_df_vs_resp.png`: implied Δf per readout over time, against breathing (measured and predicted, binned), and per
  readout measured vs predicted (R² 0.85).
- `e_encode_phase.png`: per-encode k-centre phase vs encode 0, measured vs predicted (full, flow only, concomitant only).

## Possible uses

- **Respiratory / cardiac navigator.** The out-centre (TE) samples per coil carry breathing and cardiac motion with
  R² up to 0.95, at every TR, for free. Use the out-centre, not the in-centre.
- **B0 / ratemap estimation.** This is only possible with a model that includes fat and the concomitant phase. The
  in/out phase alone is not the mean field; the magnitude ratio measures dephasing, not decay rate. As a
  consistency check, the full forward model matches to about 2 Hz and 7 %.
- **Coil-map calibration consistency.** The predicted k = 0 coil vectors match the measured ones to 6–8 % per
  readout (after one complex scale). This is a cheap per-readout check of the coil maps and motion state.
- The in-centre signal needs the concomitant phase removed before any physiological use.
- Navigator ADC during the encoders and spoilers is not simulated. It would need to be added to the simulator
  (see motion_navigators).
