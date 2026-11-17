# I/O precision policy and storage audit (2026-11-17)

## 1. Storage audit of data/hasty (measured 2026-11-17, read-only)

Method (`research/experiments/io_policy/audit.py` → `rustlib/target/release/examples/ioaudit`): every file ≥ 1 MB under
data/hasty (/mnt/hastydata; hard links once, on-disk bytes) is classified; each of the 285 files ≥ 64 MB in a
measured class was **sampled across the whole file** (≈ 384 MB per file in chunks spread evenly from start to end; whole
file if smaller; .hv trees: a spread subset of every (dtype, shape) group of tensors), encoded with the real codec and
the class policy of §2 (zstd-1 + shuffle, `hasty_value::store::encode_chunk`), decoded and compared with the source.
Ratios are extrapolated per file; files < 64 MB and the three unmeasured large files take their class's size-weighted
ratio. "max err / rms" is the worst |error| over the sampled data divided by the rms of the same data (bound
1.73e-4 + float32 rounding of the decoded value for Image / k-space; for .hv trees the pooled value mixes tensors of
different scale — per tensor the Image bound holds); "rms err / rms" is the relative rms error (target 1e-4).

**Total: 367.9 GB on disk → 188.6 GB projected (ratio 0.513), 179.3 GB freed.** (du reports 342 GiB = 368 GB.)

> **Revised after the end-to-end check (§4, §5):** complex images now use 13-bit mantissas (ratio 0.52 instead of 0.39,
> phase-safe), records 24 bits (0.34 instead of 0.27), existing `.hv` trees are not re-encoded (simulator inputs), the
> beat library and checkpoints need writer changes. Claimable now by `convert.py`: **≈ 115–120 GB**; with the beat
> format change (field-major, lossless) another ≈ 28 GB. The table below is the first audit (uniform 1e-4 everywhere).

| class | files | on disk (GB) | measured (GB) | ratio | projected (GB) | saved (GB) | worst rel err (max/rms) | rms rel err |
|---|---|---|---|---|---|---|---|---|
| hv tree | 141 | 92.75 | 72.00 | 0.720 | 66.78 | 25.97 | 2.04e-03 | 1.06e-04 |
| kspace | 62 | 88.26 | 82.62 | 0.284 | 25.11 | 63.15 | 1.74e-04 | 1.00e-04 |
| LBM beat library | 42 | 66.25 | 66.25 | 0.630 | 41.74 | 24.51 | 0.00e+00 | 0.00e+00 |
| image | 331 | 61.13 | 58.26 | 0.311 | 19.00 | 42.14 | 1.99e-04 | 1.09e-04 |
| traj | 217 | 28.44 | 27.67 | 0.269 | 7.65 | 20.78 | 1.33e-06 | 3.69e-07 |
| checkpoint (.hv) | 2 | 12.58 | 12.58 | 0.930 | 11.70 | 0.88 | 0.00e+00 | 0.00e+00 |
| not ours (models, pdf, meshes, ...) | 283 | 8.85 | 0.00 | — | 8.85 | 0.00 | 0.00e+00 | 0.00e+00 |
| checkpoint (.hsk raw) | 1 | 6.30 | 6.30 | 0.834 | 5.26 | 1.05 | 0.00e+00 | 0.00e+00 |
| npz (numpy zip) | 74 | 2.54 | 0.00 | — | 2.54 | 0.00 | 0.00e+00 | 0.00e+00 |
| integer/f16 npy | 16 | 0.81 | 0.24 | 0.005 | 0.00 | 0.80 | 0.00e+00 | 0.00e+00 |
| **total** | 1169 | **367.91** | | **0.513** | **188.62** | **179.29** | | |

Notes: k-space ratio 0.24–0.30 (19 ADC bits; §2); "image" = recon outputs, warm starts, lumen series, coil maps,
truth volumes, fields, sdf (0.07–0.40; smooth or sparse maps compress most); "hv tree" = existing `.hv` files
(truth4d, sim_input, flow/blood frames, cardiac motion, viewer volumes): their float16 / uint8 content is kept and only
re-packed (zstd-1), the gain comes from float32 tensors (sim_input 0.46) — truth4d 0.83–0.89; "LBM beat library" =
lossless repack of the f16 records + delta-coded cell indices (run_lib and run_lib_v2 hold 33 GB each); checkpoints
lossless (0.83–0.93); "not ours" (onnx/pth models, pdfs, meshes, external data) and .npz are left as they are.
Largest single items: data_clean.npy of the 10-min scan 54.3 → 15.8 GB, its sched_nominal.npy 13.6 → 3.9 GB, the two
9.3 GB 2 mm recon stacks → 3.6 GB each.

Files ≥ 1 GB:

| file (under data/hasty/phantom) | class | on disk (GB) | measured ratio | projected (GB) | saved (GB) | max err / rms | rms err / rms |
|---|---|---|---|---|---|---|---|
| factory_arms/ph0001/sim_flow_10min/scan/data_clean.npy | kspace | 54.28 | 0.291 | 15.78 | 38.50 | 1.73e-04 | 1.00e-04 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/data_clean.npy | kspace | 14.48 | 0.236 | 3.42 | 11.07 | 1.74e-04 | 1.00e-04 |
| factory_arms/ph0001/sim_flow_10min/scan/sched_nominal.npy | traj | 13.57 | 0.285 | 3.87 | 9.70 | 1.33e-06 | 2.65e-07 |
| factory_arms/ph0001/sim_b0_2.5min/scan/data_clean.npy | kspace | 11.96 | 0.303 | 3.62 | 8.34 | 1.74e-04 | 1.00e-04 |
| factory_arms/ph0001/simblood_lib/truth4d.hv | hv tree | 10.57 | not measured (OOM at the 16 G cap), class ratio | 7.61 | 2.96 | | |
| factory_arms/ph0001/pyr_synth/L2_s1/recon_seg_zmshr1s_k1.npy | image | 9.34 | 0.387 | 3.62 | 5.72 | 1.73e-04 | 1.00e-04 |
| factory_arms/ph0001/fix_2mm/llr_s/recon_seg_zmshr1s_k1.npy | image | 9.34 | 0.391 | 3.65 | 5.69 | 1.73e-04 | 1.00e-04 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/truth_fine/truth4d.hv | hv tree | 9.16 | not measured (OOM at the 16 G cap), class ratio | 6.59 | 2.56 | | |
| factory_arms/ph0001/pyr_mag/pf_rankh_2mm/state_tpyr_done_0_2x20.hv | checkpoint (.hv) | 8.39 | 0.930 | 7.80 | 0.58 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/pyr_mag/gate30_2mm/state_seg_zmshr1s_k0_tsfourier4_ga0.3.hsk | checkpoint (.hsk raw) | 6.30 | 0.834 | 5.26 | 1.05 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/sim_flow_10min_box1mm/scan/data_clean.part.npy | kspace | 4.54 | sparse (running sim), class ratio | 1.29 | 3.25 | | |
| factory_arms/ph0001/pyr_mag/gate30_2mm/state_tpyr_done_0_1x20.hv | checkpoint (.hv) | 4.19 | 0.930 | 3.90 | 0.29 | 0.00e+00 | 0.00e+00 |
| factory/ph0001/realtime_1mm/truth4d.hv | hv tree | 3.75 | 0.825 | 3.09 | 0.66 | 9.36e-05 | 3.53e-05 |
| factory/ph0004/realtime_1mm/truth4d.hv | hv tree | 3.74 | 0.868 | 3.24 | 0.49 | 9.41e-05 | 3.47e-05 |
| factory/ph0002/realtime_1mm/truth4d.hv | hv tree | 3.72 | 0.870 | 3.24 | 0.48 | 9.36e-05 | 3.64e-05 |
| factory/ph0003/realtime_1mm/truth4d.hv | hv tree | 3.71 | 0.870 | 3.22 | 0.48 | 9.36e-05 | 3.62e-05 |
| factory/ph9001/realtime_1mm/truth4d.hv | hv tree | 3.65 | 0.892 | 3.25 | 0.39 | 9.43e-05 | 3.50e-05 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/sched_played.npy | traj | 3.62 | 0.255 | 0.92 | 2.70 | 1.33e-06 | 2.62e-07 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/sched_nominal.npy | traj | 3.62 | 0.238 | 0.86 | 2.76 | 1.33e-06 | 2.62e-07 |
| factory_arms/ph0001/sim_flow_10min/scan/samples.npy | traj | 3.39 | 0.283 | 0.96 | 2.43 | 6.68e-07 | 2.65e-07 |
| factory/ph0001/sim_test/sim_input.hv | hv tree | 2.40 | 0.462 | 1.11 | 1.29 | 3.62e-04 | 9.91e-05 |
| factory_arms/ph0001/sim_flow_10min/sim_input.hv | hv tree | 2.37 | 0.464 | 1.10 | 1.27 | 5.30e-04 | 1.00e-04 |
| factory_arms/ph0001/sim_b0_2.5min/sim_input.hv | hv tree | 2.36 | 0.464 | 1.10 | 1.27 | 5.30e-04 | 1.00e-04 |
| factory_arms/ph0001/sim_b0_2.5min/scan/truth_fine/truth4d.hv | hv tree | 2.33 | 0.855 | 1.99 | 0.34 | 1.64e-04 | 9.58e-05 |
| factory_arms/ph0001/sim_flow_10min/scan/truth_fine/truth4d.hv | hv tree | 2.33 | 0.855 | 1.99 | 0.34 | 1.64e-04 | 9.59e-05 |
| factory/ph0103/gated_1mm/truth4d.hv | hv tree | 2.19 | 0.856 | 1.87 | 0.31 | 1.67e-04 | 1.01e-04 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/smaps_true.npy | image | 2.18 | 0.263 | 0.58 | 1.61 | 1.74e-04 | 1.00e-04 |
| factory/ph0102/gated_1mm/truth4d.hv | hv tree | 2.13 | 0.856 | 1.82 | 0.31 | 1.67e-04 | 1.01e-04 |
| factory/ph0004/gated_1mm/truth4d.hv | hv tree | 2.11 | 0.853 | 1.80 | 0.31 | 1.67e-04 | 1.01e-04 |
| factory/ph9001/gated_1mm/truth4d.hv | hv tree | 2.11 | 0.854 | 1.80 | 0.31 | 1.67e-04 | 1.01e-04 |
| factory/ph0001/gated_1mm/truth4d.hv | hv tree | 2.11 | 0.854 | 1.80 | 0.31 | 1.67e-04 | 1.01e-04 |
| factory/ph0002/gated_1mm/truth4d.hv | hv tree | 2.10 | 0.855 | 1.80 | 0.30 | 1.67e-04 | 1.01e-04 |
| factory/ph0003/gated_1mm/truth4d.hv | hv tree | 2.10 | 0.857 | 1.80 | 0.30 | 1.67e-04 | 1.01e-04 |
| factory/ph0101/gated_1mm/truth4d.hv | hv tree | 2.10 | 0.853 | 1.79 | 0.31 | 1.67e-04 | 1.01e-04 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/truth_vel_logical.npy | image | 2.05 | 0.051 | 0.10 | 1.94 | 1.74e-04 | 3.05e-05 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0010.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0009.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0010.bin | LBM beat library | 1.86 | 0.664 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0009.bin | LBM beat library | 1.86 | 0.664 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0008.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0017.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0007.bin | LBM beat library | 1.86 | 0.666 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0008.bin | LBM beat library | 1.86 | 0.664 | 1.23 | 0.63 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0017.bin | LBM beat library | 1.86 | 0.664 | 1.23 | 0.63 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0007.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0015.bin | LBM beat library | 1.86 | 0.664 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0005.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0016.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0014.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0006.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0004.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0015.bin | LBM beat library | 1.86 | 0.664 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0005.bin | LBM beat library | 1.86 | 0.664 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0016.bin | LBM beat library | 1.86 | 0.664 | 1.23 | 0.63 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0014.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0006.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0004.bin | LBM beat library | 1.86 | 0.663 | 1.23 | 0.63 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0011.bin | LBM beat library | 1.86 | 0.664 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0003.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0013.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0011.bin | LBM beat library | 1.86 | 0.664 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0003.bin | LBM beat library | 1.86 | 0.663 | 1.23 | 0.63 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0013.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib/beats/beat_0012.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/lbm/run_lib_v2/beats/beat_0012.bin | LBM beat library | 1.86 | 0.665 | 1.24 | 0.62 | 0.00e+00 | 0.00e+00 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/sim_input.hv | hv tree | 1.85 | 0.430 | 0.80 | 1.05 | 1.73e-03 | 1.00e-04 |
| p0/flow_0.5mm_heart.hv | hv tree | 1.83 | 0.201 | 0.37 | 1.46 | 7.46e-04 | 7.15e-05 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/samples_played.npy | traj | 1.81 | 0.274 | 0.50 | 1.32 | 9.45e-07 | 2.62e-07 |
| factory_arms/ph0001/sim_flow_10min_box1mm/sim_input.hv | hv tree | 1.64 | 0.465 | 0.76 | 0.88 | 1.59e-03 | 1.00e-04 |
| factory_arms/ph0001/conv/in_b10f/sim_input.hv | hv tree | 1.52 | 0.485 | 0.74 | 0.78 | 1.69e-03 | 1.00e-04 |
| factory_arms/ph0001/simblood_lib/flow_frames.hv | hv tree | 1.20 | 0.564 | 0.68 | 0.52 | 3.53e-04 | 8.72e-05 |
| factory_arms/ph0001/box_recon/R1/frame_lumen_it25.npy | image | 1.19 | 0.418 | 0.50 | 0.69 | 1.73e-04 | 1.00e-04 |
| factory_arms/ph0001/box_recon/R1/frame_lumen.npy | image | 1.19 | 0.421 | 0.50 | 0.69 | 1.73e-04 | 1.00e-04 |
| factory_arms/ph0001/box_recon/R1/frame_lumen_it10.npy | image | 1.19 | 0.415 | 0.49 | 0.70 | 1.74e-04 | 1.00e-04 |
| factory_arms/ph0001/sim_flow_10min_box1mm_b/scan/truth_fine/flow_frames.hv | hv tree | 1.13 | 0.528 | 0.59 | 0.53 | 3.43e-04 | 9.05e-05 |
| factory/ph0001/simblood_1mm/flow_frames.hv | hv tree | 1.08 | 0.567 | 0.61 | 0.47 | 3.52e-04 | 9.21e-05 |
| factory_arms/ph0001/sim_b0_2.5min/scan/smaps_true.npy | image | 1.07 | 0.233 | 0.25 | 0.82 | 1.73e-04 | 9.99e-05 |
| factory_arms/ph0001/sim_flow_10min/scan/smaps_true.npy | image | 1.07 | 0.233 | 0.25 | 0.82 | 1.73e-04 | 9.99e-05 |
| factory_arms/ph0001/sim_b0_2.5min/scan/truth_vel_logical.npy | image | 1.00 | 0.021 | 0.02 | 0.98 | 1.74e-04 | 1.71e-05 |
| factory_arms/ph0001/sim_flow_10min/scan/truth_vel_logical.npy | image | 1.00 | 0.021 | 0.02 | 0.98 | 1.75e-04 | 1.71e-05 |


## 2. Precision policy per data class

Rule from the user: nothing we compute needs more than ≈ 1e-4 relative precision; simulated k-space need not exceed a
real scanner ADC (16–24 bit). Everything is stored with `hasty_value::store` (Rust) / `hastycompute.store` (Python):
uniform integer quantisation (tag-10 chunks, `FORMAT.md`) + byte shuffle + zstd-1, chunked for random access.
Integer / bool / float16 data and float arrays that hold only small integers (masks stored as f32) are always
lossless; chunks containing NaN / inf are stored losslessly; the resolved step is recorded in the file (`"loss"`).

| class | policy | error guarantee | measured (real data, sampled) |
|---|---|---|---|
| k-space (`data_clean`, `data`) | `Abs(kspace_step(σ, rms))`, one chunk per readout `[1, n_adc, n_coil]` | step = min(σ/(8√2), 1e-4·√12·rms/√2); \|err\| ≤ step/2 | 10-min scan: σ = 39.98, rms = 2015 per complex sample → step 0.494 (the 1e-4 term binds; the σ/8 term would be 3.53). rms err / rms = 1.00e-4, rms err / σ_component = 0.50 % (noise variance +0.0025 %), max \|x\| / step → **19 ADC bits**, ratio 0.29 |
| per-sample records (`sched_*`, `samples*`) | `Bits(24)` per field and readout (chunk `[1, n_adc, 1]`) + delta filter (simphantom and convert.py) | \|err\| ≤ max_field / (2²⁴ − 2) per readout ≈ float32 precision at each field's maximum | k: max err 2.4e-4 rad/m (1 mm box, kmax ≈ 3e3) → phase error ≤ 7.6e-5 rad over the FOV; τ: 9.3e-10 s; ratio 0.31–0.37 (measured on the converted 1 mm scan). Bits(22) (audit run): 4× the error, ratio 0.24–0.29; Bits(20): ≈ 8e-4 rad, not ≪ 1e-3; Mantissa(20): ratio 0.62 |
| images / recon outputs / maps / truth (`recon_*`, `warm*`, `*_lumen`, `vel_*`, `smaps*`, `truth_*`, `field_*`) | `Image { rel: 1e-4 }`: **real** data uniform step over the tensor; **complex** data mantissa rounding to 13 bits | real: quantisation noise rms = 1e-4·rms(tensor), \|err\| ≤ 1.73e-4·rms; complex: pointwise \|err\| ≤ 2⁻¹⁴ = 6.1e-5 of each component | why complex differs (measured on box_recon/R1/frame_lumen, 300 k voxels × 7 encodes × 20 frames): the uniform step gave phase-difference errors median 1.1e-4, p99 1.2e-3, **max 0.66 rad** in low-magnitude voxels (|x| down to 0.03 rms); mantissa 13: max 6e-5·|x| → phase max ≈ 1e-4 rad; ratio 0.52 (uniform 0.42, mantissa 12: 0.50, relative-asinh 1e-4/1e-3: 0.52). Real maps: rms err / rms = 1.00e-4 on every file, ratio 0.05–0.39 |
| high-dynamic-range data where small values matter relatively | `Relative { rel, floor }` (asinh, knee = floor·rms) | \|err\| ≲ rel·√(knee² + x²) | Python test: max err / √(k² + x²) = 1.0e-4 over 6 decades |
| existing `.hv` trees (truth4d, sim_input, flow_frames, …) | floats → Image(1e-4); float16 / uint8 → lossless zstd-1 re-encode | as above per tensor | float16 tensors are already below the budget (rel 4.9e-4) and are kept as is |
| solver checkpoints (`state_*.hv`, `.hsk`) | `Uniform { rel: 1e-6 }` **per stored vector** (each set of each stack on its own scale; tensors < 65536 elements and non-float data bit-exact); the quasi-Newton states store the secant step s = x − x_prev instead of x_prev | quantisation noise 1e-6 of that vector's rms; s keeps 1e-6 of ‖s‖ however small the step | Real ZMSHR1 state (gate30_2mm, k = 163, ‖s‖/‖x‖ = 1.5e-2): x 0.63, s 0.57, g_prev 0.61 of raw (lossless 0.84 / 0.63 / 0.82); x and x_prev stored separately at 1e-6 would give s a 9.5e-5 relative error, stored as s: 1e-6. Resume equivalence (CPU test `checkpoint_lossy`, Barzilai–Borwein on a 262 k complex quadratic, stop at 25 of 45): resumed vs uninterrupted 4.4e-6 after 20 iterations (state error 1.0e-6), lossless control 0; the GPU test `host_state` (host-stack checkpoints) is listed in §4. Old raw `HSTKCK01` and lossless states still load |
| LBM beat library (`beats/*.bin`) | **lossless** — the near-lossless criterion (≤ 1e-6 relative) cannot be met by any quantiser: the records are already float16 (11-bit mantissa, 4.9e-4 relative) | bit-exact | measured on beat_0010 (6 blocks spread): lossless as written 0.663, **field-major [28][n] + delta-coded cells 0.580** (−27.8 GB over 66 GB instead of −22.3 GB); per-field uniform quantisers trade small values' precision: 16 bit 0.36 (max err 1.5e-5 of the field max, but median pointwise 0.9 %), 14 bit 0.28, 12 bit 0.20. Converting needs the beat format change in hasty-cfd (writer + reader), not done here |

**Throughput** (10 threads of the 1000 % CPU quota, zstd-1; encode / decode in memory): k-space 5.3 / 13.3 GB/s,
images 3.2 / 7.1 GB/s, records 4.8 / 11 GB/s; lz4 is no faster at decode and 17 % larger, zstd-3 costs 27 % encode
speed for < 0.2 % size. Disks (2 GB, O_DIRECT): rustlib/target (nvme1) write 3.7 GB/s, read 6.5 GB/s; data/hasty
(nvme0, read-only test) read 6.1 GB/s. At 10 threads decode delivers 7–13 GB/s of raw data while consuming only
2–4 GB/s of file (ratio 0.3), so loading a `.hv` is CPU-bound but 1.2–2× faster than reading the raw `.npy` at disk
speed; saturating the disk would take ≈ 20 GB/s raw (15–25 threads). Encode (3–5 GB/s raw = 1–1.5 GB/s of file) is
below the 3.7 GB/s write speed: writes are CPU-bound too, and write 3× fewer bytes (SSD wear).

## 3. Compressed in-RAM storage (jobs above the RAM budget)

94 GB total, heavy jobs capped by `systemd-run MemoryMax` (memory_ledger.md). Candidates, with the access pattern
that decides whether decode-on-use pays:

| array | size | access | compressed in RAM | decision |
|---|---|---|---|---|
| k-space (10-min scan) | 54 GB raw | once per run into the KStore, then per set per iteration | KStore `zstd+qREL` tiers (noise-relative, ≈ 4–6 bits/component) exist; the `.hv` file (ratio 0.29, 15.8 GB) now feeds kload directly (one chunk = one readout: no read amplification) | done (KStore) — nothing new |
| sample tables (1 mm box) | 9.3 GiB | per set per iteration | packed tier P exists (2.42 GiB) | done |
| solver state per frame (x, g, x_prev, g_prev) | 4 × 17.8 GiB at 1 mm | every iteration, read + write | per-vector 1e-6 quantisation (the checkpoint policy) gives 0.57–0.63 of raw: −28 GiB, for ≈ 4 × 17.8 GB × (1/3.2 + 1/7.1) s ≈ 30 s of CPU per iteration at 10 threads | not done: only worthwhile if an iteration takes minutes and RAM is the binding limit; the ledger's §3.1 (store s, y in half precision) or the temporal basis are cheaper |
| simphantom beat cache | ≈ 40 GB at `--beat-cache 15` | per bin chunk, random | records already f16; lossless 0.58–0.66 | possible −14 GB, decode ≈ 0.35 GB/s per thread for the f16 shuffle path: measure the per-chunk hit rate before building it (not done) |

## 4. Migration status (2026-11-17)

New files are `.hv` by default (`HASTY_IO=npy`: plain `.npy`; `HASTY_IO_LOSSLESS=1`: `.hv` without quantisation).
Readers take either: `hasty_value::store::resolve` / `hastycompute.store.resolve` map `x.npy` ↔ `x.hv`, so paths in
scripts, meta files and command lines keep working; a writer that rewrites an array removes the stale sibling.

**Rust writers**
- `simphantom`: `data_clean.hv` (k-space, `Abs(kspace_step)`, one readout per chunk; converted from the resumable
  `data_clean.part.npy` after the run, then the raw file is removed), `data.hv` (noisy copy), `sched_*` / `samples*`
  (Bits(24) per field and readout), `smaps_true`, `field_ppm_recon`, `truth_r2s/ff/pd_water/pd_fat` (Image);
  meta.json records the storage (`"storage"`). Small tables (readouts, resp, physio) stay `.npy`.
- `recon_p2np`: recon_*, vel_*, warm, warm_full, warm_vel, tpyramid_mean, smaps, smaps_axial, masks, it*_lumen (Image:
  complex → mantissa 13, real → 1e-4 rms); static / air maps reloaded on resume: lossless `.hv`.
- `framerecon`, `splinerecon` (lumen series), `motionfit` (motion maps): Image.
- checkpoints: `RunCtl::save` (all `state_*.hv`) and the host-stack checkpoints (`state_*.hsk`, now `.hv` content):
  Uniform 1e-6 per vector, secant step stored as s (§2). Old checkpoints load.
- not migrated (small or other owners): sim4d, simimg, opval*, seqdesign, girfqa and other bench tables; hasty-cfd
  (beat library, case sdf), hasty-phantom subject volumes, hasty-server.

**Rust readers**: `hasty_recon::data::read_npy` (every bench binary: sim maps, truths, warm starts, lumen idx, motion
maps), `kload::NpyFile` (k-space and per-sample records streamed readout by readout from `.npy` or chunked `.hv`:
`store::RowFile`), `pyramid::InitImages` (recon_*.hv / warm*.hv), existence checks in data.rs, b0data, recon_b0,
pyramid_dry, recon_p2np (`store::exists_any`).

**Python**: `hastycompute.store` (`load_any`, `save_array`, `resolve`, `exists_any`, `glob_any`, `load_meta`,
`LazyArray` = the memory-map counterpart: `load_any(p, mmap_mode="r")` of a `.hv` decodes only the chunks an index
touches); `value_format` reads quantised tensors (tag 10) and digest trailers, decodes chunks on threads, and uses the
rebuilt Rust reader (`libhasty_value_capi.so`) when present. Readers migrated: every `np.load` under rustlib/tools,
research/experiments and python/lib (85 files) → `load_any`, plus 14 `.npy` existence checks / globs (run_figs,
opval_figs, show_run, static_map_eval, physics_figs, plot_seqdesign, plot_b0recon, sim_conv, llr_spectra, case.py,
mag_var, vel_error, score_pair). Writers migrated: `sim_truth.py` (truth_vel_logical, truth_lumen_fraction,
truth_fat_fraction). Not migrated: `.npz` outputs (bias_truth, truth_pc_frames), training / subject / mesh tools.

**Tests** (CPU): `cargo test -p hasty-value` (store: image / complex / k-space rows / per-field bits / masks / NaN /
value writer), `cargo test -p hasty-recon --lib` (61, incl. `.hv` sim files loading like `.npy`, eager and streamed),
`--test kload_stream`, `--test checkpoint_lossy` (resume equivalence), the Python cross-language test (Rust ↔ Python
lossy files decode bit-identically). GPU test to run when the GPU is free: `cargo test --release -p hasty-recon
--test host_state -- --nocapture` (host-stack checkpoint stop / resume vs uninterrupted; tolerance 4e-5 / 1.5e-4,
dominated by float32 atomics).

**End-to-end check on real data** (sim_flow_10min_box1mm_b/scan converted into a temporary copy, then removed; logs in
research/experiments/io_policy/verify_*):
- conversion: 29.4 → 7.4 GB (data_clean 14.5 → 3.4 GB, records 0.31–0.37, smaps 0.44, truth velocity 0.05), every
  file within its bound;
- Rust loaders (`--test hv_equivalence`, `SimData::load_lazy` + `load_set` on 256 readouts): coordinates max |Δ| 4.8e-7
  rad/voxel, τ 9.3e-10 s, k-space rms Δ / rms 9.0e-5 (0.15 % of the noise σ), maps within 1.75e-4 of rms;
- `splinerecon --time-only --encodes 0` on the converted scan: runs (k-space streamed from the `.hv`), the same
  segments; 263 673 fit voxels vs 263 674 (one coil-map support voxel at the threshold);
- Python scorer `roi_score.py` (box_recon/R1/frame_lumen converted, truths from the converted scan): every score equal
  to the printed digits except one LCx mean ratio (−0.41 → −0.42); a random 1e-4 relative perturbation of the same
  file changes a different ratio by 0.01 too, a 1e-6 one nothing: the scores' own sensitivity at the policy's
  precision (hierarchical velocity unwrapping in a noise-dominated reconstruction). This check is also what showed the
  uniform step to be wrong for complex images (§2).

## 5. Converting existing data (not yet run: needs the user's go-ahead)

The in-place conversion with deletion of the `.npy` sources was prepared and verified on a copy (§4) but **not run**:
deleting data under data/hasty needs the user's own permission. `research/experiments/io_policy/convert.py`
classifies each `.npy` like the audit, writes the `.hv` next to it, checks the whole array against the policy bound
(k-space / real images: |err| ≤ step/2 + float32 rounding; complex: pointwise 2⁻¹⁴; records: per field and readout),
and only then, with `--delete-source`, removes the `.npy`. It skips files < 64 MB, `*.part.npy` and directories holding
one (in-progress simulations), files open or memory-mapped by any process, and directories whose readers lack `.hv`
support (lbm, subjects, models, external, train, totalseg, denoiser). Existing `.hv` trees are not re-encoded (their
float tensors include simulator inputs such as source positions, where a 1e-4-of-rms error is 35 µm ≈ 0.1 rad of k-space
phase at 1 mm; the audit's "hv tree" saving is therefore not claimed); the beat library needs a hasty-cfd format change.

```bash
cd /home/turbotage/Documents/GitHub/HastyCompute
# dry run: class and policy per file
rustlib/.venv/bin/python -I research/experiments/io_policy/convert.py data/hasty/phantom
# convert + verify + delete the sources, logged (≈ 150 MB/s: ≈ 20 min for the ≈ 180 GB of .npy)
HASTY_IO_THREADS=10 systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 nice -n 10 \
  rustlib/.venv/bin/python -I research/experiments/io_policy/convert.py --apply --delete-source \
  --log research/experiments/io_policy/convert_log.txt data/hasty/phantom
```

Expected (audit ratios with the final policies): k-space 88 → 25 GB, records 28 → ≈ 9.5 GB (Bits 24), real images
and complex reconstructions 61 → ≈ 23 GB (complex 0.52 instead of 0.39): ≈ 115–120 GB freed by the `.npy` conversion; the
beat library (−27.8 GB lossless, field-major) and checkpoints (−≈ 5 GB at 1e-6 per vector) need their writers'
formats changed / the runs re-checkpointed.
