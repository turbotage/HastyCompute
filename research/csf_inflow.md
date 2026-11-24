# CSF inflow signal simulator (Söderström et al., MRM 2026)

Code: `rustlib/crates/hasty-mrsim/src/inflow.rs` (module `hasty_mrsim::inflow`, CPU only, independent of the k-space
direct-sum simulator), experiments `rustlib/crates/hasty-mrsim/examples/inflowsim.rs`, outputs
`research/experiments/csf_inflow/<cmd>/` (npy ignored by git, `summary.json` kept).
Run: `cargo build --release -p hasty-mrsim --example inflowsim`, then `inflowsim <knee|phantom|alias|sweeps|roundtrip> <out>`.

## Model
1D plug flow along the slice normal, ζ(t) = ∫v dt, v = Σ v_k cos(φ_k + δ_k) (cardiac from `Ecg`, respiratory and slow
clocks wandering, optional non-sinusoidal harmonics). A continuum of elements carries M_z; per excitation event T1
recovery + Gaussian dispersion (D = 6 cm²/min), then M_z → M_z cos α(z − z_slice), α from the Bloch profile of a sinc
pulse (`rf::SlabPulse`), perfect spoiling. Bottom-slice signal S = e^(−TE/T2) ∫ sin α M_z dz. All 37 slices are timed
(interleaved), the first `n_sim` (default 6) are simulated. Optional cross-section profile A(z) (volume-flux plug).
The paper's Eq. 5 is garbled in the text I read; the ΔT factor is taken as e^(−ΔT/T1) (verified against the closed form).

## Checks (unit tests, `cargo test --release -p hasty-mrsim --lib inflow`)
- single slice, no flow: matches the spoiled-GRE closed form per label (2e-5 relative).
- 50 mm/s plug: signal = fresh-spin value (1e-5), above static.
- neighbouring slices lower the bottom-slice signal (cross-talk, 3 %).
- continuum engine vs independent Monte-Carlo random walk (periodic domain): 1.6 % max difference.

## Results (paper acquisition: TR 2 s, TE 30 ms, 80°, 3.4 mm + 0.5 mm gap, interleaved, T1 4.738 s)
- `knee`: static signal = 0.53 of fresh; saturates at v ≈ 2–3 mm/s (v·TR ≈ one slice + profile tails).
- `phantom` (0.93 / 0.23 / 0.077 Hz, six amplitudes): modulation is linear in amplitude (r ≥ 0.998 for all three; cardiac
  14.3 % at 20 mm/s, respiratory 16.5 % at 5 mm/s, low 18.0 % at 2.5 mm/s). The paper's phantom lost linearity of the low
  frequency above ≈ 1.5 mm/s (r = 0.58) and calls it a phantom artifact because their digital phantom did not show it;
  this simulator agrees (no compression).
- `alias`: cardiac 0.93 Hz at TR 2 s appears as a peak at 0.070 Hz (8.7 % at 12 mm/s), i.e. inside the low-frequency band.
  At TR 0.5 s it stays at 0.93 Hz.
- `sweeps` (12 mm/s cardiac / 1.2 mm/s resp modulation): dispersion 0 → 8 cm²/min: 22.5 → 7.6 % (dispersion is the
  dominant model uncertainty); T1 3.5 → 5.2 s: 7.2 → 9.3 %; flip 50° → 90°: 4.5 → 10 %; gap 0 → 2 mm: 9.7 → 6.9 %;
  sequential 9.1 % vs interleaved 8.8 %.
- `roundtrip` (truth = Monte-Carlo engine with random-walk dispersion, physiology-like clocks, 2 % noise; estimator =
  continuum engine + quasi-stationary sinusoids of the same clocks, LM): sinusoidal truth R² 0.91, v 12.3 / 1.10 / 0.13
  vs truth 12.0 / 1.2 / 0.18 mm/s; non-sinusoidal cardiac truth (harmonics 0.5, 0.25, 0.12) R² 0.83, v 11.7 / 1.06 / 0.16.
  Phases recovered within 0.05–0.25 rad. The slow component is the least well determined.

## Not modelled
Real EPI (in-plane encoding, ghosting, T2* in the voxel, intravoxel velocity dephasing), non-plug profiles (a Lagrangian
radial profile would be the extension), net flow, diffusion in a funnel (reference-area width only), BOLD-derived slow phase.

## Figures
`research/experiments/csf_inflow/plot.py` (run from the repo root with `.venv/bin/python`, matplotlib in the gitignored `.venv`) writes
`fig1_estimated_vs_prescribed.png` (against the paper's Table 2 regression lines), `fig2_modulation_vs_amplitude.png`,
`fig3_aliasing.png`, `fig4_knee_and_timeseries.png`, `fig5_sweeps.png`. Data for fig1: `inflowsim phantomfit`.

## Mismatch study (`inflowsim mismatch`, fig6_mismatch.png)
Truth: Monte-Carlo engine, physiology clocks, non-sinusoidal cardiac, 2 % noise, v = 12 / 1.2 / 0.18 mm/s. Nominal fit 11.7 / 1.06 / 0.16.
Variants are warm-started from the nominal fit (local sensitivity). Estimated v (mm/s) cardiac / resp / slow, R² in brackets:
- noise seeds (same truth): 11.0–11.7 / 1.06–1.32 / 0.15–0.22, i.e. about ±3 % / ±10 % / ±17 %.
- estimator D 3 / 9 cm²/min (truth 6): 7.6 / 0.67 / 0.10 and 15.6 / 1.44 / 0.22 — R² 0.829 / 0.830, **unchanged**.
- estimator T1 4.2 / 5.3 s (truth 4.74): 12.6 / 10.9 (±8 %). Flip 70° / 90° (truth 80°): 14.3 / 10.0 (+22 % / −14 %), R² unchanged.
  Gap 1.0 instead of 0.5 mm: +6 %.
- recorded-physiology timing: a late clock only shifts δ by 2π f τ (0.10 s cardiac: +0.57 rad; 1 s respiratory: +1.4 rad) and leaves
  v and R² untouched. Random R-peak jitter 30 / 60 ms lowers cardiac v by 3 / 10 % and R² to 0.81 / 0.76.
- slow component vs a confound locked to its regressor (0.25 / 0.5 / 1 % amplitude, arbitrary phase): 0.13 / 0.10 / 0.04 mm/s.
  A BOLD-like 1 % rms slow noise not tied to the regressor does not move it (R² 0.83 → 0.80).
- non-plug truth (laminar m = 2, blunt m = 6, mean velocity prescribed): cardiac 11.4 / 11.7, resp 1.02: plug assumption costs ≤ 3 %.
Conclusions: fit quality cannot calibrate D, T1 or flip, so absolute velocities carry the assumed values (≈ D^0.65; ±10° of
effective flip ≈ ±18 %), while phases, relative changes and the plug assumption are robust. The slow component is limited
by noise (±17 %) and by anything locked to its regressor (a 0.5 % BOLD residual in the CSF voxels removes about 40 % of it).
