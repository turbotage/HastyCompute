# Temporal rank of the velocity-encode images (interim report, 2026-11-19)

Status: the cardiac-axis results below are final. The regime sweep was 3 of 19 runs done at hand-back. The joint
breathing × cardiac runs, the whole-FOV breathing run, the beat library, HRV and estimability are queued in a detached
unit and had not run yet. Section 9 says how to finish. The sections marked *pending* get filled from
`research/experiments/temporal_rank/out/*.json` when the queue is done.

## 1. Question (pre-registered)

How many temporal basis images K per velocity encode are needed to represent the complex encode image over
(breathing, cardiac) time to a given accuracy, and how does K depend on the flow regime? Decision informed: is a
data-derived low-rank temporal basis worth building into the operator, and what K would it need? Today the operator sees
6 linear breathing knots × 16 cubic cardiac frames = 96 images per encode.

## 2. Data used

- **Cardiac truth.** The phantom ph0001's 1 mm truth `simblood_lib/truth4d.hv` is the input the k-space simulator draws
  its blood from. It has 100 cardiac frames (8.5 ms apart) at end-expiration (r = 0). It contains:
  - LBM chamber and great-vessel-stub flow, which is the phase-binned mean of 18 LBM cycles at 50 bins, linearly
    interpolated to 100 frames, so the chamber content has at most 50 independent time samples;
  - LBM coronary flow solved in the heart's reference frame;
  - the 1-D network flow (25 harmonics) in the other vessels;
  - moving tissue labels, and texture.
- **Breathing.** `motion/resp_motion.hv`: the displacement u = r Σ B_k(X), which is linear in the breathing amplitude r,
  with 13 / 8 / 3 mm SI / AP / LR at r = 1. The amplitude distribution r(t) comes from the scan's `sim_input.hv`
  breathing trace: range 0–1.4, median 0.49.
- **Encoding** (from `meta.json`): coprime7 dual-VENC, 7 encodes. VENC_lo = 1.0 m/s, VENC_hi = 1.5 m/s, unaliased
  range ±3 m/s.
- **Region.** A 2 mm heart box of 82 × 65 × 66 voxels (352 k voxels). It covers the heart ± 10 mm, the aortic root,
  arch and descending aorta, the pulmonary trunk and its branches, SVC, IVC, the pulmonary veins and the coronaries.
  Each 2 mm voxel is the complex mean of its 8 fine voxels, so partial volume and intravoxel dephasing are part of the
  signal.
- **Magnitude.** Spoiled GRE at TE with the phantom's tissue table and texture, and blood × the inflow gain 1.5. No
  inflow time dependence was modelled, and no B0 / ratemap phase was included.

## 3. Method

- The temporal Gram XᴴX (T × T) is computed per encode and pooled over encodes, and eigen-decomposed.
- K at an error level is the smallest K whose relative Frobenius tail is at or below that level, over the whole box.
- Each basis is scored by orthogonal projection of every encode's time series onto it, which is the best any
  coefficients could do. The scores are:
  - relative error over signal voxels and over lumen voxels;
  - p99 per-voxel relative error;
  - the velocity error after dual-VENC decoding of the projected encodes, compared with the decoded truth, in the lumen
    (mean, p99, split into chamber and vessel blood);
  - the flow-integral error per named vessel, max_t |ΔQ| / max_t |Q|, with Q = Σ blood weight × tangent · decoded
    velocity.
- Bases compared at equal K:
  - SVD shared by all 7 encodes, and SVD per encode;
  - periodic cubic B-splines with the operator's knot convention;
  - Fourier.
- **Control.** A synthetic complex signal of exact rank 7 came out as K = 7 at every level ≤ 1e-2, with singular values
  ≤ 2e-8 beyond the seventh. With noise of 1e-3 added, the eighth singular value equals the noise (7.9e-5). The pipeline
  measures rank correctly.
- **Noise reference.** meta.json gives σ_v = 1.56 cm/s per axis at encode-image SNR 20, which is about 2.7 cm/s as a
  vector magnitude, scaling with 20/SNR. The SNR of the real reconstruction per frame is not known from the data and
  is an assumption.

## 4. Results: ph0001 cardiac axis (r = 0, 100 frames)

### 4.1 K at each error level (heart box, 7 encodes pooled)

| signal | 1e-1 | 3e-2 | 1e-2 | 3e-3 | 1e-3 |
|---|---|---|---|---|---|
| truth as stored (step-wise 1 mm labels) | 9 | 30 | 68 | 93 | 99 |
| truth, magnitude moved continuously (see 4.3) | 8 | 22 | 44 | 74 | 89 |
| velocity phase only (magnitude frozen at time mean) | 7 | 20 | 42 | 73 | 89 |
| magnitude / motion only (no velocity phase) | 3 | 18 | 62 | 91 | 99 |
| velocity only, chamber blood excluded | 4 | 14 | 30 | 56 | 82 |
| network-flow overlay reference (4.4) | 8 | 21 | 49 | 77 | 93 |
| same, velocities × 0.25 (p99 speed / VENC_lo 1.46 → 0.36) | 3 | 10 | 23 | 51 | 74 |

Per-encode K differs from the shared K by at most ±2. Per-encode SVD is only marginally better than one basis shared by
all encodes (for example, at K = 16 the lumen error is 5.9 % vs 6.1 %). **A single shared basis is enough.**

### 4.2 Error versus K (stored truth)

Mean |Δv| is in cm/s; the flow columns are max_t |ΔQ| / max|Q|.

| basis | K | rel. lumen | mean \|Δv\| | p99 \|Δv\| | RCA | LAD | LCx | Asc. aorta |
|---|---|---|---|---|---|---|---|---|
| SVD shared | 8 | 12.8 % | 5.3 | 26 | 0.22 | 0.23 | 0.36 | 0.10 |
| SVD shared | 12 | 8.3 % | 3.2 | 17 | 0.09 | 0.14 | 0.22 | 0.03 |
| SVD shared | 16 | 6.1 % | 2.3 | 12 | 0.06 | 0.08 | 0.15 | 0.02 |
| SVD shared | 24 | 3.8 % | 1.2 | 7.7 | 0.05 | 0.02 | 0.05 | 0.01 |
| SVD shared | 32 | 2.7 % | 0.8 | 5.2 | 0.03 | 0.02 | 0.03 | 0.01 |
| SVD shared | 48 | 1.7 % | 0.4 | 2.8 | 0.02 | 0.01 | 0.02 | 0.004 |
| cubic B-spline (today's cardiac part) | 16 | 8.1 % | 2.8 | 19 | 0.11 | 0.10 | 0.23 | 0.07 |
| cubic B-spline | 24 | 4.6 % | 1.4 | 10 | 0.07 | 0.07 | 0.13 | 0.03 |
| cubic B-spline | 32 | 3.2 % | 0.9 | 7.0 | 0.06 | 0.04 | 0.07 | 0.03 |
| cubic B-spline | 48 | 2.0 % | 0.55 | 4.0 | 0.02 | 0.02 | 0.03 | 0.01 |
| Fourier | 16 | 7.6 % | 2.7 | 16 | 0.12 | 0.18 | 0.25 | 0.07 |

Read across, **an SVD basis buys about a factor 1.5 in K over cubic B-splines at equal error**: SVD 16 ≈ B-spline 24,
and SVD 32 ≈ B-spline 48. The SVD spectrum does not fall off steeply, so it does not give a factor of 4–8. In the
network-flow overlay reference at K = 16:

- SVD gives a vessel-blood mean |Δv| of 1.4 cm/s and a chamber-blood mean of 2.6 cm/s, against 1.6 / 3.1 for
  B-splines.
- The coronary flow errors are 7–13 % for SVD and 17–21 % for B-splines.

The SVC flow error stays at 13–30 % for every basis up to K = 32. The SVC group includes the RA inflow stub, where LBM
and partial-volume voxels dominate a small flow, so this metric is not trusted.

### 4.3 What drives the rank (`diag_regions`)

- **Energy above temporal harmonic 16 (the stored truth).** By region, it is:
  - chamber blood 1.2e-3 of its energy, which is 36 % of the image's high-frequency energy;
  - vessel blood 6.4e-4 (19 %);
  - myocardium 2.6e-3 (18 %);
  - other tissue 3.7e-4 (16 %).
- **Step-wise 1 mm labels.** These are a phantom artefact. The 1 mm labels are pulled back to the nearest label, so a
  wall crossing a fine-voxel centre makes a step in magnitude (traces in `out/diag_voxel_traces.png`). Steps have a
  1/h spectrum, and they alone account for the magnitude-only K = 62 at 1e-2.
- **Continuous-motion magnitude.** Replacing the magnitude by the frame-0 magnitude moved continuously with the
  cardiac motion field (`motion/cardiac_motion.hv`, trilinear) lowers K at 1e-2 from 68 to 44. This version differs
  from the stored frames by 9–13 % at 2 mm, mostly at edges. **Real moving edges are still a genuine rank driver:** the
  phantom-free edge model (section 5) gives K ≈ 10–13 at 1e-2 for a 10–15 mm wall excursion.
- **Velocity phase.** The velocity phase alone gives K = 42 at 1e-2. The LBM chamber flow carries the most diverse
  waveforms; without the chambers the velocity-only K is 30. Part of the chamber tail is LBM bin noise: an 18-cycle
  mean at 50 bins. The beat-library run (pending) is meant to size that.

### 4.4 Regime sweep (partial)

The sweep replaces the vessel blood with a controllable overlay on the true moving centrelines:

- the vessel's own motion (frames_geom wall velocity), plus
- blood fraction × a Poiseuille profile × the 1-D network mean velocity along the tangent.

Chambers and tissue keep the truth. The magnitude is the continuous-motion version.

Done so far:

- reference: K = 8 / 21 / 49 / 77 / 93 at 1e-1 … 1e-3;
- velocities × 0.25: K = 3 / 10 / 23 / 51 / 74.

Phase excursion (v/VENC) is a strong lever: quartering the speed halves K at 1e-2.

*Pending* (queued in `run_regimes.py`):

- VENC ratio × 0.5 / 0.75 / 1.5 / 2;
- per-vessel delays 25 / 50 / 100 ms;
- added random harmonics 0.1 / 0.2 / 0.4;
- coronary diastolic-dominant waveform with 60 / 30 / 15 ms ramps, and a 2× speed variant;
- one combined case.

## 5. Phantom-free model of the velocity-phase rank (`generic_model.py`, final)

The ensemble is x(t) = exp(−iπ a W(t − τ)), with a uniform in [0, v_max/VENC] and τ uniform in [0, spread]. K at 1e-2 is
in the table below; the full grid is in `out/generic_model.json`.

| waveform | spread 0 ms | 50 ms | 100 ms |
|---|---|---|---|
| aortic jet, v_max/VENC 1 | 3 | 6 | 7 |
| aortic jet, v_max/VENC 2 | 4 | 8 | 11 |
| SVC, v_max/VENC 2 | 4 | 12 | 17 |
| coronary 15 ms ramp, v_max/VENC 1 | 3 | 9 | 14 |
| coronary 15 ms ramp, v_max/VENC 2 | 4 | 13 | 22 |

- **One waveform shape has very low rank.** With a single shape scaled by amplitude, K stays 3–5 even at v_max = 2 VENC
  and with steep coronary ramps.
- **Rank comes from waveform diversity.** Timing spread and shape variety across voxels drive it, multiplied by the
  phase excursion. Steepness matters only when combined with delays: a 15 ms ramp with 100 ms spread at v_max = 2 VENC
  needs 22 at 1e-2 and 40 at 1e-3.
- **Moving edges** (partial-volume ramps of a wall through 2 mm voxels): K at 1e-2 is 4, 7, 10 and 13 for 2, 5, 10 and
  15 mm excursions.
- The 3-D phantom (K ≈ 44–49 at 1e-2) is much richer than any of these ensembles. Chamber flow and the sum of all
  vessels and edges add up.

## 6. Baseline adequacy (cardiac part of today's 6 × 16 basis)

- **Image error.** 16 cubic cardiac frames leave a lumen relative error of 6.6–8 % at the best coefficients. That is
  comparable to or above the noise per frame at SNR 15–20.
- **Velocity error.**
  - mean |Δv| is 2.3–2.8 cm/s and p99 is 16–19 cm/s, against a noise σ_v of about 2.7 cm/s (vector magnitude, SNR 20);
  - in vessel blood, the mean is 1.6 cm/s and p99 is 14 cm/s.
- **Coronary flow error** (max over the cycle): 10–23 %.
- Truncation is a bias, not noise, so it does not average out. **Today's 16-frame cardiac basis is not well below the
  noise for coronary flow.**
- The operator interpolates its 16 frames from the model rather than projecting, which is somewhat worse than the
  projection numbers here. Reaching about 2–3 % coronary flow error and about 1 cm/s mean |Δv| takes about 32 SVD
  images, or about 48 cubic B-spline frames.
- The breathing half of the baseline (6 linear hats) is *pending*: `joint_rank.py` compares linear hats × cubic
  B-splines, a separable SVD and a joint SVD.

## 7. Estimability (*pending*)

`estimability.py` learns the SVD basis from four kinds of source:

- a 4 / 6 / 8 mm version of the scene;
- the scene plus noise at lumen SNR 20 / 10 / 5;
- a different regime (overlay, and overlay + delays + coronary waveform);
- the stored step-wise truth.

Each basis is scored on the continuous-motion target at K = 8–32, against B-spline and Fourier bases.

## 8. Implications and decision (cardiac axis; may change with the breathing results)

- **Decision: a data-derived SVD basis is worth having, but as a rank-for-accuracy trade, not an FFT cut.** At equal
  accuracy it needs about 2/3 of the B-spline count: 32 SVD images instead of 48 cubic frames, or 16 instead of 24.
  Nothing in this data supports K ≈ 6–10 per encode. The image is genuinely high-rank, because chamber flow, diverse
  vessel waveforms and moving edges add up.
- **More cardiac resolution is the bigger issue than the basis type.** The current 16 cardiac frames are probably too
  few for the coronary target. Raising them to about 24–32 (B-spline), or using 16–24 SVD images, roughly halves the
  coronary flow error.
- **Basis structure.** One basis shared by all 7 encodes loses almost nothing against per-encode bases.
- **Regime dependence.** Lower v/VENC lowers K strongly: at 0.25× speed, K at 1e-2 drops from 49 to 23. Dual-VENC
  low-VENC encodes that wrap are the high-rank part. Timing and shape diversity raise K.
- **Phantom vs reality.** ph0001's step-wise 1 mm labels inflate the magnitude rank, and the continuous-motion version
  is used for the regime runs. Further caveats:
  - LBM bin noise sets part of the chamber tail;
  - inflow, ratemap phase and breathing hysteresis are not in this cardiac analysis;
  - real beats vary, and that floor (*pending*) caps what any phase basis can reach.

## 9. Files and how to finish

Code is in `research/experiments/temporal_rank/`:

- `common.py`, `rank_lib.py`, `experiment.py`: shared definitions, signals, metrics;
- `extract_cache.py`: fine-field cache;
- `run_truth.py`, `diag_regions.py`, `run_regimes.py`, `regimes.py`: the cardiac runs;
- `joint_rank.py`, `resp_fov.py`, `beats_rank.py`, `run_hrv.py`, `estimability.py`, `generic_model.py`: the other runs;
- `plot_all.py`, `queue.sh`: figures and the run queue.

Outputs are in `out/` (JSON, logs, figures).

The cache `cache/` is about 5 GB of .npy plus pickles, gitignored. Delete it once the study is finished.

Detached units at hand-back:

- `tr-regimes` (the regime sweep);
- `tr-queue`, which waits for `tr-regimes` and then runs the joint, whole-FOV breathing, beats, estimability and HRV
  runs, then `plot_all.py`. Its log is `out/queue.log`.

Both units resume by skipping runs whose JSON exists. `touch out/STOP` stops them between runs. To restart after a stop:

```
cd research/experiments/temporal_rank
systemd-run --user --unit tr-regimes -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% --working-directory=$PWD rustlib/.venv/bin/python3 run_regimes.py
systemd-run --user --unit tr-queue -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% --working-directory=$PWD bash queue.sh
```

Use the absolute path to `rustlib/.venv/bin/python3`.
