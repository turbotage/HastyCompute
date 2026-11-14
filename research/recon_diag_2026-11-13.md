# Why the per-frame reconstruction does not behave (2026-11-13)

Symptom: per-frame velocity error grows over the iterations while the data residual falls, slowly with
ZMSHR1-s (2 mm: rel 0.96 → 1.10 in 20 it), fast with ZMSHR1-ms (2 mm: 1.00 → 2.01 in 10 it; phase error 1 rad
at it 10, `frame_fit.py` on the checkpoint). Diagnostics at the pyramid level L0 (grid 0.25, `--kcrop 1`, ~15 s per
iteration), which reproduces the symptom (zmshr1s 0.99 → 1.10, zmshr1ms → 1.55, circ flat at 1.03 in 15 it).
Run dirs: `data/hasty/phantom/factory_arms/ph0001/pyr_mag/L0_*`.

## 1. The LLR prior is effectively off at κ = 1 (and up to κ ≈ 10) [measured]

LLR: 8³ blocks, Casorati 512 voxels × T columns, T = all images of the stack = 2 respiratory bins × 20 cardiac
frames × 7 encodes = 280; threshold κσ̂, σ̂ = median over blocks of each block's s[T/4].

- Pure noise (512 × 280, i.i.d. σ): s[T/4] = 13.8σ, edge 39σ, so κ = 2.8 removes all noise; κ = 1 keeps 75 % of the
  noise components (shrunk). For T = 24 the bulk is narrow (κ = 1.3 reaches the edge): a fixed κ is a much weaker
  prior at T = 280.
- The iterates are not noise + low rank. Block spectra (`research/experiments/recon_diag/llr_spectra.py`):

  | iterate | comps > σ̂ (median of 280) | > 10σ̂ | > 30σ̂ | energy removed at κ = 1 |
  |---|---|---|---|---|
  | L0 zmshr1s κ 10, it 15 | 198 | 78 | 33 | 0.1 % |
  | 2 mm fix_2mm (zmshr1s κ 1, it 20) | 207 | 58 | 11 | 1.4 % |

  The spectra decay continuously (no noise plateau); s[T/4] differs 0.4–2.7× between blocks; bright blocks carry the
  most components (2 mm: 73 above 10σ̂ in the bright third, 18 in the dim third): per-frame aliasing, ∝ brightness.
  σ̂ lands inside this artifact spectrum and moves with it.
- The truth is low rank: the ideal blood signal (`truth_pc_frames.npz`, T = 7 × 20 cardiac frames), 8³ blocks:
  rank for 99 % of the energy median 2 (p90 5, max 15), for 99.9 % median 7 (p90 14, max 40).
- At L0 the GPU LLR keeps k ≤ 106–192 components even at κ = 10.

## 2. Inverse crime at L0, start at x★ (exact model, `--synthetic --init-truth --plan host`)

x★ = warm images with the true per-frame velocity phases in the lumen; y = E x★ + n.

| velocity rel, it 0 → 15 | κ 0 | κ 1 | κ 3 | κ 10 |
|---|---|---|---|---|
| zmshr1s | 0.197 → 0.237 | → 0.229 | → 0.223 | → 0.202 |
| zmshr1ms | 0.197 → 0.280 | → 0.238 | → 0.227 | → 0.200 |

f stays at the noise floor (4.0e11). Without an active prior the iterate leaves the truth by noise fitting in the
per-frame null space, about twice as fast with the metric. κ = 10 holds it. The inverse-crime run at 2 mm
(pyr_synth, κ 1) drifted the same way (0.114 → 0.152 m/s in 8 it).

## 3. Real data from the time-averaged start: no κ recovers the dynamics

| zmshr1s, velocity rel at it 15 | κ 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|
| | 1.095 | 1.084 | 1.077 | 1.078 | 1.042 | 1.010 |

(zmshr1ms: 1.55 / 1.39 / 1.27 for κ 1 / 3 / 10.) Start 0.993. Systole frames stay at 0.73–0.87 m/s error, as in
the time average. f = 7.6e14 at it 15, about 1900× the inverse crime's floor at L0 (2 mm: 50× the floor at it 20).
A stronger prior only slows the decline; it does not recover the cardiac dynamics.

## 4. Inverse crime from the time-averaged start: slow recovery of the dynamics [measured]

x0 = warm (time average per encode), x★ = warm + true per-frame lumen phases (1102 lumen voxels at L0), y = E x★ + n.

| zmshr1s, velocity rel | it 0 | 5 | 10 | 15 | 20 | f at it 20 |
|---|---|---|---|---|---|---|
| κ 0 | 0.938 | 0.908 | 0.889 | 0.857 | 0.854 | 4.53e11 |
| κ 10 | 0.938 | 0.905 | 0.886 | 0.849 | 0.812 | 4.66e11 |
| κ 100 | 0.938 | 0.908 | 0.871 | 0.859 | 0.839 | 4.81e11 |

Target 0.197 (x★'s own error on the L0 grid). The noise floor is 4.0e11 and f(x_5) = 6.25e11: the dynamics' data
component is ≈ 0.7σ rms per sample, plenty of information in aggregate, yet 20 iterations recover ~10 % of it — the
iteration moves slowly along the per-frame directions.

**Correction (same day).** I first read the real-data residual (L0: 7.6e14, ~3000× the dynamics' excess) as "model
mismatch dominates the gradient". Not supported: (i) the L0 crop keeps only low k, yet its residual exceeds the whole
2 mm residual over all of k (2.3e14 at it 20), so it is mostly L0's own coarse-grid representation error; (ii) with
only each readout's first 0.3 ms (`--kcrop 1:prefix`, `L0_prefix`, where off-resonance, fat and flow during the
readout are negligible) f is still 6.3e14 and the dynamics are still not recovered (rel 0.99 → 1.06); (iii) the 2 mm f
was still falling at it 20, so "50× the floor" can be slow convergence. Low-k residual of this size is what
respiratory motion inside two broad bins gives, and practitioners live with it: it is common to all encodes and
cancels in the phase differences. A large f is not evidence for the velocity failure. L0 is not a valid test bed for
real-data dynamics. `--bin-group` at L0 (`L0_bingroup`): 1.095 → 1.032 (κ 1), 1.077 → 1.009 (κ 10), no recovery.

## 5. Reality check: a practitioner's per-frame reconstruction of the same data (2 mm)

`framerecon` (hasty-bench): every cardiac frame × encode on its own (respiratory states pooled), plain SENSE NUFFT, 12
strongest coils, circulant-preconditioned PCGLS, 10 it, no regulariser; scored with `frame_fit.py` (deep lumen ≥ 5
voxels, blood in every frame):

| | slope | corr | wrapped phase error rms |
|---|---|---|---|
| plain, all samples, bin-group | 0.33 | 0.31 | 0.61 rad |
| plain, out half (0–4 ms), bin-group | 0.43 | 0.56 | 0.46 rad |
| plain, out half, frames as recon_p2np bins them | 0.46 | 0.36 | 0.63 rad |
| P²nP fix_2mm/llr_s (all samples, recon_p2np binning, LLR κ 1, 20 it) | 0.65 | 0.64 | 0.42 rad |

- Our P²nP already beats the plain per-frame reconstruction: the per-frame problem of this protocol is that hard
  (≈ 380 readouts per frame and encode for a 254 × 144 × 114 grid, 7 encodes; 3-D radial needs ~10⁵ for Nyquist).
  PC-VIPR-style results rely on temporal view sharing / filtering, i.e. a prior.
- Binning every rotation's encodes into one frame (shared undersampling per frame, as PC-VIPR / Cartesian 4D flow)
  helps the plain per-frame phases strongly (corr 0.36 → 0.56, rms 0.63 → 0.46 rad).
- The back half of the out-and-back readout hurts per frame as it did time-averaged (0.61 → 0.46 rad with bin-group).
- No semi-convergence in the plain per-frame reconstruction: it improves steadily with the iterations (out half +
  bin-group, deep: slope / corr / rms 0.10 / 0.44 / 0.55 rad at 3 it, 0.43 / 0.56 / 0.46 at 10, 0.69 / 0.64 / 0.43 at
  25; systole slope 0.05 → 0.39 → 0.67). Per-frame velocities converge slowly; 25 plain CG iterations match P²nP's 20.
- Consequence for sections 2–4: the in-run "velocity RMSE rel" (all lumen voxels, through the decoder; at L0 nearly
  all partial-volume) was never validated against frame_fit. "Velocity gets worse over the iterations" may be a
  property of that metric. Test: `pyr_mag/traj_2mm` (P²nP at 2 mm, `--save-lumen` every 5 it, frame_fit per
  iteration).

## 6. The pooled frame_fit score mostly measures the time-mean flow [measured]

`frame_fit.py`'s slope / corr pool arg(x_e x̄_0) over voxels and frames, so they credit the spatial pattern of the
time-mean flow. Oracle: the truth's own temporal (circular) mean as the reconstruction, i.e. a perfect time-averaged
image with no cardiac dynamics, scores deep slope 0.43, corr 0.64, rms 0.47 rad, as good as every "per-frame"
result in section 5. Added: "dynamics" (each voxel's temporal mean removed on both sides) and "ROI dynamics" (phases
averaged over 8 mm blocks first, as a practitioner reads ROI flow curves); `dyn_fit.py` also takes one truth per
respiratory bin (`bias_truth.py --resp R --frames-out`). Deep lumen (depth ≥ 5), truth dynamics rms 0.405 rad
(ROI 0.364):

| | dyn slope | dyn corr | dyn err rms | ROI dyn corr | ROI err rms |
|---|---|---|---|---|---|
| zero dynamics (perfect time average) | 0 | 0 | 0.405 | 0 | 0.364 |
| plain per frame, out half, bin-group, 3 it | 0.08 | 0.26 | 0.397 | 0.30 | 0.352 |
| same, 10 it | 0.41 | 0.42 | 0.385 | 0.63 | 0.291 |
| same, 25 it | 0.67 | 0.52 | 0.388 | 0.72 | 0.268 |
| same, 25 it, gated per respiratory bin (half the data each) | 0.60 | 0.49 | 0.395 | 0.69 | 0.255 |
| plain, all samples, bin-group, 10 it | 0.34 | 0.25 | 0.513 | 0.48 | 0.330 |
| plain, out half, no bin-group, 10 it | 0.49 | 0.29 | 0.583 | 0.46 | 0.364 |
| P²nP traj_2mm it 0 (time-averaged start) | 0 | 0 | 0.405 | 0 | 0.364 |
| P²nP it 5 | 0.37 | 0.47 | 0.360 | 0.53 | 0.314 |
| P²nP it 10 | 0.37 | 0.45 | 0.370 | 0.49 | 0.331 |

- Nobody recovers the per-voxel dynamics well, and 8 mm averaging helps little: the errors are spatially coherent
  (bias), not noise.
- The in-run "velocity RMSE rel" rose 1.00 → 1.14 over the same iterations in which the dynamics went 0 → 0.37: it
  is not a usable measure of the per-frame result (noise through the alias-searching decoder, partial-volume voxels).
- P²nP (all samples, no bin-group) is on par with the plain per-frame reconstruction with the same two handicaps; at
  depth 2–5 it is worse (dyn corr 0.19 vs 0.41 for plain + both levers).
- The traj_2mm run stopped at it 12 (CUDA out of memory in the gradient; device 4.5 GiB free before the solver).
  Since then: the GPU LLR retries smaller batches on OOM and returns its pool memory; a failing gradient writes a raw
  checkpoint at that iteration.

## 7. Coherent errors, a bin-group timing bug, and encode-order cross-talk [measured]

- The dynamics error is coherent: averaging the phases over ROIs of n voxels (cubes of 1–12 voxels, depth ≥ 2) takes
  the plain recon's error from 0.61 (n = 1) to 0.27 rad (n ≈ 550); incoherent noise would fall to 0.03. P²nP's
  dynamics are strongly attenuated (gain 0.12–0.26 vs truth, plain 25 it: 0.33–0.50): from the time-averaged start
  the iterations move little along the per-frame directions (as section 4's inverse crime).
- Timing (`lagtest.py`, `roiscale.py`: circular cross-correlation of ROI waveforms): with `--bin-group` the plain
  recon's waveforms lag the truth by ~0.5 frame (corr 0.53 at lag 0, 0.50 at +1) and are blurred ~1 frame; P²nP without
  bin-group shows no lag. Cause: the referenced 7-point scheme's encode 0 carries a first moment
  (k_0 = (−π/2, −π/2, −π/2) rad per m/s), so each velocity component mixes the times of encode 0 and encode e. With
  u(t) = u0 + u't the least-squares velocity is u0 + T u', T = pinv(ΔK) D, D_e = k_e t_e − k_0 t_0: mean diag(T) =
  1.35 TR after encode 0, while `--bin-group` stamped the group with the middle readout (3 TR): 18.7 ms = 0.44 frame
  late. Fixed: `hasty_recon::data::velocity_time` (generic, unit-tested), `--bin-group` now takes the frame of the
  readout nearest that time (`--bin-group-mid` = old behaviour) in recon_p2np and framerecon.
- Inherent cross-talk when encodes share a frame: T's off-diagonal (up to 3.1 TR = 35 ms) mixes the other axes'
  acceleration into each component. With the truth velocities (u_vol per frame, central differences, voxels covered
  in every frame): 0.087 m/s rms (p99 0.36) against 0.21 m/s rms of cardiac dynamics (systole 0.11 vs 0.33).
  Per-encode framing (no bin-group) avoids it to first order, at the price of encode-specific undersampling.
- Sequence side: over all 5040 encode orders, the reference in the middle with symmetric axes (encode → slot
  (3, 5, 0, 1, 6, 2, 4), current (0, 1, …, 6)) cuts the error to 0.011 m/s rms (8×). Worth adopting in the
  flow_thorax sequence if frames share encodes.

## 8. What the per-frame dynamics are limited by [measured, plain per-frame recon unless noted]

Deep lumen (depth ≥ 5), dynamics only; truth at the scan-mean breathing state unless noted; 25 PCGLS it, out half:

| | dyn slope | dyn corr | dyn err rms | ROI dyn corr | ROI err rms |
|---|---|---|---|---|---|
| bin-group, middle-readout stamp (old) | 0.65 | 0.51 | 0.395 | 0.72 | 0.248 |
| bin-group, velocity-time stamp (fixed) | 0.67 | 0.53 | 0.385 | 0.75 | 0.236 |
| per-encode framing | 0.77 | 0.39 | 0.614 | 0.65 | 0.308 |
| bin-group (old stamp) + view sharing ±1 frame (3× data) | 0.64 | 0.64 | 0.301 | 0.78 | 0.217 |
| gated bin 0 (half the data), truth at the mean state 0.49 | 0.58 | 0.52 | 0.368 | 0.68 | 0.255 |
| gated bin 0, truth at the bin's state 0.18 | 0.66 | 0.57 | 0.355 | 0.74 | 0.236 |
| P²nP traj_2mm it 5, bin 0, truth at 0.49 / 0.18 | 0.35 / 0.37 | 0.46 / 0.49 | 0.357 / 0.351 | 0.54 / 0.58 | 0.293 / 0.282 |

- Data: 3× the readouts per frame (view sharing) is the largest single gain: undersampling (≈ 380 readouts per frame
  and encode at 254 × 144 × 114) limits the per-frame result, as for any unregularised reconstruction.
- Bin-group timing: the fix removes the lag (corr at lag 0 / ±1: 0.56 / 0.42, 0.38) and gains a little; the remaining
  cross-talk (0.18–0.23 rad per encode phase from the truth's accelerations, §7) is of the size of the ROI residual
  (0.24 rad): with bin-group it is the coherent error floor.
- Per-encode framing: no timing error, less attenuation (slope 0.77), but encode-specific aliasing (voxel error 0.61).
- Respiratory misregistration of the scoring truth: minor (+0.03–0.05 in corr).
- P²nP from the time average: attenuated (dyn gain 0.2–0.4); next test `pyr_mag/pf_rank_2mm` (per-encode framing,
  out half, per-frame PCG start `--tpyramid 2x20:10`, LLR with a rank cap of 16, `--llr-rank`).

## 9. P²nP from a per-frame start with an LLR that acts (`pyr_mag/pf_rank_2mm`) [measured]

Per-encode framing, out half (`--kcrop 0.99:prefix`: 48.6 % of the samples, ADC 3.87 ms, L = 13; FRAC 1 keeps
everything at the native grid since the readout never leaves the Nyquist ball), start `--tpyramid 2x20:10` (per-frame
circulant PCG through the full model from the time average, 48 min), then zmshr1s κ 1 with a soft rank cap of 16
(`--llr-rank 16`: the LLR now keeps k ≤ 16 per block, was ≤ 277). 250 s per iteration. Deep lumen, dynamics:

| | dyn slope | dyn corr | dyn err rms | ROI dyn corr | ROI err rms |
|---|---|---|---|---|---|
| it 0 (the per-frame PCG start) | 0.46 | 0.37 | 0.451 | 0.48 | 0.346 |
| it 5 | 0.34 | 0.47 | 0.350 | 0.55 | 0.293 |
| it 10 | 0.31 | 0.43 | 0.361 | 0.52 | 0.299 |
| (traj_2mm it 5: time-average start, LLR inactive, all samples) | 0.37 | 0.47 | 0.360 | 0.53 | 0.314 |
| (plain per-encode framing, 10 / 25 PCGLS it) | 0.48 / 0.77 | 0.29 / 0.39 | 0.578 / 0.614 | 0.49 / 0.65 | 0.341 / 0.308 |

- The full model's per-frame PCG start is no better than plain NUFFT per frame (aliasing-limited, not model-limited).
- The capped LLR removes noise (error 0.45 → 0.35) but shrinks the dynamics, more with every iteration (slope 0.46 →
  0.34 → 0.31): the soft rule subtracts s_17 from every kept singular value, and the cardiac components are the weak
  ones. Next: the hard rule (`--llr-hard`, keep the top 16 unchanged), `pyr_mag/pf_rankh_2mm` (same start, the pyramid
  level hardlinked).
- The in-run "velocity rel" called it 0 the worst (1.44) and it 10 better (1.05): opposite to the dynamics scores.

## 10. LLR cannot separate the per-frame error from the dynamics; a cardiac-time prior can [measured]

- Hard rank cap 16 (`pf_rankh_2mm`, same start): worse than soft at it 5 (deep dyn slope / corr / err 0.37 / 0.25 /
  0.547 vs 0.34 / 0.47 / 0.350): stopped.
- Offline, one LLR step on the per-frame start (`llr_offline.py`: unshifted 8³ blocks around the truth lumen, SVD per
  block, the pyramid level `--tpyr`; note: a pyramid level is stored in group order e · F + f): deep lumen

  | rule (T = 280) | R | dyn slope | corr | err | energy removed |
  |---|---|---|---|---|---|
  | none | – | 0.46 | 0.37 | 0.451 | 0 |
  | soft | 8 / 16 / 32 / 64 | 0.37 / 0.40 / 0.42 / 0.44 | 0.46 / 0.44 / 0.42 / 0.40 | 0.359 / 0.375 / 0.396 / 0.421 | 21 / 16 / 11 / 6 % |
  | hard | 8 / 16 / 32 | 0.43 / 0.44 / 0.45 | 0.41 / 0.39 / 0.38 | 0.405 / 0.429 / 0.441 | 16 / 11 / 7 % |

  Per respiratory bin (T = 140): the same. Hard truncation to 8 components leaves most of the error: the error lies in
  the block's leading temporal components, with the dynamics; soft thresholding lowers it only by shrinking both.
- Temporal smoothing over the cardiac frames of the same start (per voxel, encode, respiratory bin): Gaussian w = 1
  frame: corr 0.37 → 0.60, err 0.451 → 0.314 (blurred-truth floor 0.237); low-pass to the cyclic harmonics |k| ≤ 4:
  slope 0.43 (kept), corr 0.57, err 0.322; |k| ≤ 3: 0.41 / 0.61 / 0.311. The truth needs |k| ≤ 6 (0.12 rad) to 8
  (0.08 rad). The per-frame error is incoherent across cardiac frames: a prior over cardiac time is the lever, LLR is not.
- Implemented `hasty_recon::tlp::cardiac_lowpass` (projection onto |k| ≤ K, unit-tested) as `--prior tlp --tlp-k K`;
  run `pyr_mag/pf_tlp_2mm` (same start, K = 4, κ 1: least squares constrained to band-limited cardiac cycles).

Result (`pf_tlp_2mm`, deep lumen, dynamics):

| | dyn slope | corr | err | ROI corr | ROI err |
|---|---|---|---|---|---|
| it 0 (per-frame start) | 0.46 | 0.37 | 0.451 | 0.48 | 0.346 |
| it 5, cardiac low-pass \|k\| ≤ 4 | 0.45 | 0.57 | 0.319 | 0.68 | 0.256 |
| it 10 | 0.53 | 0.45 | 0.402 | 0.59 | 0.301 |
| (it 5 soft LLR rank 16 / traj_2mm) | 0.34 / 0.37 | 0.47 / 0.47 | 0.350 / 0.360 | 0.55 / 0.53 | 0.293 / 0.314 |

Best P²nP so far at it 5 (as predicted offline: 0.43 / 0.57 / 0.322), then semi-convergence: the attenuation
falls (slope 0.45 → 0.53) while the per-frame aliasing inside the kept band (9 of 20 harmonics) is fitted again.
The per-frame problem at ≈ 380 readouts per frame and encode is variance-limited; the prior has to carry more than a
band limit: candidates a weighted (Tikhonov) harmonic decay instead of the hard band, |k| ≤ 3, the learned/spatial
prior on top, shared aliasing across encodes (bin-group with the reordered encode sequence, §7), more readouts per
frame (sequence).

## 11. The per-frame inverse crime, and what the warm image is [measured]

Question: is the per-frame error a null-space (sampling) limit or data inconsistency? Test: framerecon `--synthetic`,
data y = A x★ from the reconstruction's own plain operator, no noise, same sampling, start 0, 50 PCGLS it.

First version (x★ = warm image `wfull_circ/warm_full.npy` with the true per-frame phases in the lumen,
`synth_lumen.py`): deep-lumen dynamics slope 0.01 / 0.03 / 0.09 and corr 0.03 / 0.05 / 0.13 at 10 / 25 / 50 it.
That is far worse than the measured noise-free data (0.68 / 0.56 at 25 it), with consistent data. The lumen values
are 2.7× off in norm from x★ (phase difference rms 1.58 rad). The cause is x★ itself:

| image (encode 0) | energy beyond ½ k_max | beyond k_max (sphere) | corr(\|x\|, PD) in body |
|---|---|---|---|
| truth PD (water + fat) | 3.9 % | 1.1 % | 1 |
| \|warm_full\| | 11.7 % | 3.5 % | 0.18 |
| warm_full (complex) | 43 % | 8.2 % | |
| warm_full inside the body (+6 mm) | 46 % | 8.8 % | |
| `tpyramid_mean` (pf_tlp_2mm start) | 41 % | 7.9 % | |

- Outside the body mask (PD or lumen, dilated 6 mm) the warm image holds 8.9 % of its energy. A first count of 31 %
  used the PD mask alone, which counts the bright vessels and the breathing margin as outside.
  - The 0–6 mm rim is brighter than the body: median |x| 0.44–0.47 against 0.27 inside. That is the breathing
    displacement of the surface, plus fat.
  - Haze further out: median |x| 0.07 at 12–24 mm and 0.03–0.04 beyond 24 mm, i.e. 13 % of the body median.
- The complex image is broadband inside the body (46 % beyond ½ k_max, magnitude alone 12 %), so its phase varies on
  the voxel scale.
- The magnitude barely correlates with PD (0.18). PD is not the image contrast (T1 weighting, inflow, T2*), so this is
  only an upper hint, not a measurement of error.
- As x★, this broadband object aliases into the lumen of every per-frame reconstruction far more than a smooth object
  would. So the first inverse crime is not the per-frame problem of our object and is discarded.
- [CHECK] Why the warm image's phase is this rough. Candidates:
  - motion inconsistency of the time-mean model (one image fit to all respiratory and cardiac states);
  - the circulant metric amplifying unsampled k-space (1/(λ + ε) where λ ≈ 0 outside the sphere): 8 % of the energy
    sits beyond the sampled sphere;
  - the full-physics phase model.
  Next: the same spectra for a plain pooled CG of one respiratory bin, and the air map's view of the haze.
- Second version (`synth_pd.py`): x★ = PD_water + PD_fat (real) everywhere, plus the true blood signal C (same units;
  blood is not in the static PD maps) with the true phase differences in the lumen. Oracle check: slope 0.97,
  corr 0.99. Deep-lumen dynamics (slope / corr / err rad, 8 mm ROI corr), against the measured noise-free data with
  the same sampling, operator and iterations (§8):

  | PCGLS it | inverse crime (PD object) | measured, noise-free |
  |---|---|---|
  | 10 | 0.27 / 0.70 / 0.322 (0.87) | 0.42 / 0.45 / 0.371 |
  | 25 | 0.65 / 0.57 / 0.341 (0.86) | 0.68 / 0.56 / 0.369 (0.75) |
  | 50 | **0.94 / 0.87 / 0.180 (0.96)** | 0.65 / 0.48 / 0.424 |
  | 100 | — | 0.50 / 0.34 / 0.503 |

  **The per-frame problem is not sampling-limited.** With consistent data and an object of realistic spectrum, plain
  per-frame PCGLS from 0 recovers the lumen dynamics (slope 0.94, corr 0.87 at 50 it) despite 19 % of the Nyquist
  cells per frame. The measured noise-free data, with the same operator, sampling and data residual (frame 0
  encode 0: 2.3e-3 vs 1.5e-3 at 50 it), stall at 25 it and then degrade. **The limit is model mismatch between
  the plain per-frame operator and the data.** The mismatch is fitted into the image: the residual keeps falling while
  the lumen error grows.
  - At 25 it the two curves coincide: early iterations are limited by convergence, not by mismatch. Every 10–25 it
    comparison so far was therefore blind to the mismatch.
  - What the plain operator leaves out (framerecon: `SenseOp`, true coil maps, no physics grid):
    - respiratory motion (all breathing states pooled per cardiac frame, ~10 mm);
    - cardiac motion within the frame window;
    - B0 off-resonance, fat shift and T2* during the 4 ms window;
    - readout motion;
    - gradient nonlinearity.
  - Ladder queued (`scratchpad/motion_ladder.sh`, 50 it, measured vs inverse crime with the same sampling): one
    respiratory bin of 4; a 1 ms window; both.

## 12. RED and the temporal subspace on the 2 mm data

- `pyr_mag/red_tlp_2mm`: pf_tlp_2mm with `--prior-mode red --red-rho 1` (g = ∇f + λ(I − P)x, λ = 1/α = 1.5e9), and
  zmshr1s with 14 secant groups, one per (respiratory bin, encode). Stable.
  - The out-of-band part ‖x − Px‖/‖x‖ falls from 0.147 at the start to 1.1e-3 after one step, and stays at 1e-3. That
    is the Newton-like step on the quadratic penalty.
  - ‖λ(x − Px)‖/‖∇f‖ ≈ 0.4 from then on: the prior is active in every step.

  Deep-lumen dynamics, slope / corr / err (8 mm ROI corr), and f:

  | | it 5 | it 10 | f at it 10 |
  |---|---|---|---|
  | P²nP, projection after the step (pf_tlp_2mm) | 0.450 / 0.574 / 0.319 (0.681) | 0.531 / 0.454 / 0.402 (0.585) | 2.69e14 |
  | RED, penalty in the gradient | 0.449 / 0.576 / 0.318 (0.681) | **0.485 / 0.585 / 0.314 (0.704)** | 2.91e14 |

  - The two are identical at it 5.
  - At it 10, P²nP degrades (corr 0.57 → 0.45, err 0.32 → 0.40, in-run velocity RMSE 0.45 → 0.53 m/s) while RED
    still improves slightly (in-run RMSE 0.449 → 0.448).
  - P²nP's data term falls further (2.69e14 against 2.91e14). P²nP fits the model mismatch of §11 into the band
    while RED's iterate stays nearer the penalised optimum, which is the expected direction [CHECK: one run, one ρ].
    - RED's secant pairs see the constraint's curvature, so its steps along the band are not inflated.
    - In P²nP the projection after the step makes the secants of the next step mix out-of-band gradient differences.
- `pyr_mag/tsub_2mm`: `--tsub fourier:4 --kappa 0`: 18 coefficients for 40 frames per encode, 14 secant groups, T/Tᴴ
  0.6 s per iteration, stable. Deep-lumen dynamics against the data term f(x_k) of all three runs:

  | run | it | f | slope / corr / err (ROI corr) |
  |---|---|---|---|
  | start (pyramid level projected onto the band) | 0 | | 0.426 / 0.573 / 0.322 (0.671) |
  | P²nP + projection | 5 | 3.34e14 | 0.450 / 0.574 / 0.319 (0.681) |
  | RED | 5 | 3.15e14 | 0.449 / 0.576 / 0.318 (0.681) |
  | subspace | 5 | 3.03e14 | 0.488 / 0.585 / 0.314 (0.704) |
  | RED | 10 | 2.91e14 | 0.485 / 0.585 / 0.314 (0.704) |
  | P²nP + projection | 10 | 2.69e14 | 0.531 / 0.454 / 0.402 (0.585) |
  | subspace | 10 | 2.44e14 | 0.582 / 0.529 / 0.359 (0.686) |

  - **Semi-convergence in all three.** The dynamics improve until f ≈ 2.9–3.0e14, then the correlation falls while
    f keeps falling: the model mismatch of §11 is fitted into the band-limited image. RED at it 10 is "better" than
    P²nP only because it is slower: RED at it 10 = subspace at it 5.
  - Speed: the subspace (exact constraint) is ≈ 2× RED at ρ 1 (a penalty, with stiffer curvature for the shared
    secant group).
  - **At equal data fit P²nP is worse.** At f = 2.69e14, P²nP reaches corr 0.45 / err 0.40, against ≈ 0.56 / 0.34
    interpolated between the subspace's it 5 and it 10. Projecting after per-set-scaled steps leaves the iterate
    further from the constrained least-squares path. This matches the theory: P²nP's fixed point depends on the
    metric. [CHECK: interpolation, one run each]
  - So the prior's form matters less than the data model. Conclusions about RED or subspace against P²nP need the
    mismatch removed (inverse crime at 2 mm, or the motion / readout ladder's winner modelled).
- `pyr_mag/tsub_2mm`: `--tsub fourier:4 --kappa 0`, x = T c with 18 coefficients for 40 frames, no prior. This is the
  constrained problem that P²nP-tlp approximates by projecting after each step.
- Found on the way: per-set ZMSHR1 diverges when the Hessian couples sets (a temporal prior in the gradient, or a
  temporal basis). Its secant pairs assume H = blockdiag H_t, and a coupled set's ⟨s_t, m_t⟩ mixes in the other
  sets' steps. Fixed by secant groups: the coupled sets share one pair (tests/tsub.rs: diverged to NaN, now 6e-8).

## 13. Respiratory motion is a large part of the mismatch (2026-11-14) [measured]

Breathing (physio.npy: bellows = column 3, the simulator's true state r = column 4; corr 0.97). r = 1 moves the heart
centre ≈ 10 mm (resp_motion.json: (0.3, −3.7, −9.5) mm at end inspiration); r ranges over 0–1.4, std 0.36 over the scan.
Spread of the true r inside a bellows bin:

| binning | std of r per bin (≈ mm × 10 at the heart) |
|---|---|
| none | 0.36 |
| 2 bins (used so far) | 0.165 / 0.195 |
| 4 bins | 0.065 / 0.14 / 0.135 / 0.12 |
| 8 bins | 0.039 / 0.074 / 0.10 / 0.12 / 0.12 / 0.10 / 0.083 / 0.098 |
| gate: lowest 30 % / 40 % / 50 % of the bellows | 0.082 / 0.12 / 0.165 |

Bellows bins stop sharpening in mid-breath: 8 bins still leave ≈ 1 mm rms there, because inhale and exhale pass the same
bellows amplitude at different r (hysteresis). End-expiration is the quietest state, which is why practice gates there.

Per-frame, noise-free, 50 PCGLS it, `--bin-group`, 0–4 ms window, one bellows bin of 4 (`--resp-bin 0/4`: 25 % of the
readouts, mean r 0.061). Deep lumen; the truths are at r 0.493 (scan mean) and 0.181 (the nearest available, still 1.2
mm off):

| data | it 25 | it 50 |
|---|---|---|
| all breathing states (2-bin-free, §11) | 0.68 / 0.56 / 0.369 | 0.65 / 0.48 / 0.424 (falling) |
| one bin of 4, truth r 0.493 | 0.50 / 0.55 / 0.336 | 0.56 / 0.59 / 0.325 |
| one bin of 4, truth r 0.181 | 0.57 / 0.60 / 0.320 (ROI 0.75) | **0.65 / 0.65 / 0.304 (ROI 0.77)** (rising) |

With a quarter of the data the per-frame dynamics improve and no longer degrade with iterations: breathing motion is a
large part of the §11 mismatch. A truth at r 0.07 is queued to score it properly.

Inverse crime with the same quarter of the readouts (x★ from `synth_pd.py`, scored against its own truth at r 0.493):

| PCGLS it | slope / corr / err (ROI corr) |
|---|---|
| 25 | 0.34 / 0.38 / 0.390 (0.73) |
| 50 | 0.43 / 0.40 / 0.399 (0.75) |

- This is below the measured quarter (0.56 / 0.59 / 0.325 against the same truth) and still rising: with 25 % of the
  readouts per frame, plain PCGLS has not converged at 50 it.
  - With all readouts it reached 0.94 at 50 it, but only 0.27 at 10 it.
- So the quarter-data inverse crime is not an upper bound at 50 it. It does not decide whether gating to 25 % is
  sampling-limited [CHECK: 150 it].
- Why the measured data converge faster on the lumen dynamics is open. The objects differ: x★ is PD + blood at r 0.493
  with real static tissue, against the simulated contrast and phase at r ≈ 0.06.

Runs (2026-11-14):
- `recon_p2np --resp-accept FRAC` (new): gating, keeps the lowest FRAC of the breathing amplitude.
- `pyr_mag/gate30_2mm`: 30 % gate, 1 bin × 20 frames, temporal subspace fourier:4, 20 it. k-space store in RAM.
- The rest of the ladder (1 ms window; window + 1 bin of 4) and the truth at r 0.07 follow it.

## 14. Night run 2026-11-14 (01:12–07:00) — `pyr_mag/night_1114/`

- `night.sh` (log: `night.log`):
  - stops gate30_2mm at it 9 (checkpoint);
  - CPU: truth at r 0.07;
  - GPU: per-frame one bin of 4 at 150 it, measured (`fr_meas_rb04`) and inverse crime (`fr_ic_rb04`), snapshots at
    50 / 100 / 150;
  - then gate30_2mm resumed by `gate30_2mm/run2.sh` (up to 300 it, snapshots every 10);
  - scores in `scores.txt` if the truth is ready before 06:50.
- 07:00: `stop7.sh`, run by the user timer `hasty-stop7` and by `watchdog.sh` (flock: once):
  - kills the queue, framerecon and the truth build;
  - recon_p2np: STOP file + SIGUSR1, checkpoint at the end of its iteration, 20 min grace, then SIGTERM / SIGKILL.
- Resume the gated run: `gate30_2mm/run2.sh`. Score: dyn_fit.py with `--truth truth_pc_frames_r0.07.npz` on
  `gate30_2mm/it*_lumen.npy` and `night_1114/fr_*_lumen.npy`.

## 15. Open

- The P²nP frame_fit trajectory at 2 mm (`traj_2mm`, running) and the plain reconstruction beyond 25 it.
- P²nP at 2 mm with `--bin-group` and an LLR that acts (threshold rule), scored with frame_fit, against the 0.65 /
  0.42 rad baseline.
- The threshold rule: σ̂ from the iterate's own lower quartile is the wrong scale when the spectrum is a continuous
  artifact spectrum. Candidates: a noise scale propagated from the data noise, a rank cap from the truth statistics,
  T split (per respiratory bin, encodes into the rows).
