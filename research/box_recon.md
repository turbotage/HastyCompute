# Reconstruction of the 1 mm box scan (2026-11-15)

Scan: `factory_arms/ph0001/sim_flow_10min_box1mm_b/scan` (research/box_benchmark.md, research/sim_convergence.md):
- 198 × 189 × 228 voxels at 1 mm (8.53 M);
- 9891 readouts × 5720 ADC samples × 32 coils;
- noise σ 39.98, flow_thorax_v2_1mm, 7 encodes;
- truth from the LBM beat library (`truth_bias.npz`).

Runs: `factory_arms/ph0001/box_recon/R*`. Lumen index for per-iteration scoring: `box_recon/lumen_idx_f32.npy`.
It holds 1 023 475 voxels, those with coverage.max over frames > 0, in the order of bias_truth's `truth_pc_frames.npz` idx.

## R0: feasibility smoke (2 iterations)

Question: does the full-physics reconstruction fit this machine at 1 mm, with 2 respiratory bins × 20 cardiac frames
(280 sets), and what does one iteration cost?

Final command (attempt 4):

```
recon_p2np <scan> --gnl --phys full --kstore disk:zstd1 --table-store zstd --plan host --state host
  --resp-bins 2 --resp bellows --fat-fraction truth --fat-model joint --r2s truth
  --nseg auto:noise:0.2 --seg-basis svdn-union --seg-weight energy --b0-absorb 1 --seg-fit-stride 9 --seg-table-stride 4
  --tsub fourier:4 --kappa 0 --iters 2 --precond zmshr1ms --circ-share encode --no-full-save
```

What failed on the way. Logs are in R0/:

1. **CUDA OOM building the segments** (`log_try1_L27_oom.txt`). Two causes:
   - The default worst-voxel rule gave L = 27: field plus the main fat peak span 17.65 cycles over the 9.48 ms
     readout.
   - The planner had put the sample tables on the device (46.9 M samples, ≈ 8 GB), leaving 3 GiB.

   Fixes: `--plan host`, and the segment rule of research/segment_basis.md §5, which gives **L = 11**. The tables then
   take 2.42 GiB packed on the host (×3.84) and 0.08 GiB of device slots.
2. **Host OOM, per-frame state** (`log_try2_hoststate_oom.txt`). The host-state solver keeps x, g, x_prev and g_prev,
   at 17.8 GiB per stack (8.53 M × 280 sets × 8 B). With 16.7 GiB of setup that is ≈ 88 GB > 80 GB cap (94 GB host).
   **Per-frame P²nP at 1 mm with 40 frames does not fit this host.**
3. **Host OOM, temporal subspace** (`log_try3_tsub_framestacks_oom.txt`). `tsub::Subspace` held two full frame stacks
   for its gradient (x = T c and ∇f(x), 2 × 17.8 GiB), plus the frame buffer and 4 coefficient stacks of 8 GiB:
   ≈ 102 GB.

   Fix (code): `Subspace::gradient_host` now works in frame blocks of ≤ 2 GiB. Per block it synthesises the frames,
   takes their gradient through the new `Objective::gradient_host_sets` (P2np: the same set streamer, over a subset
   of sets), and accumulates g (+)= B[f0..f1, :]ᵀ g_block (`tsub::apply_sets_into`). It falls back to full stacks
   when the inner objective has no subset form. Test `hasty-recon/tests/tsub.rs blocked_gradient_matches_full_stacks`:
   blocked vs full gradient rel 1e-7 (f32), f rel 1e-16.

Result (attempt 4):

| | |
|---|---|
| peak RSS | 70.6 GiB: setup 16.6 + frame buffer 17.8 + 4 × 8.0 coefficients + 2 × 1.9 blocks |
| device | ≈ 16 GiB used: tables streamed, state streamed |
| setup | 478 s (attempt 3) / 1364 s (attempt 4: SegNufft build 300 → 1201 s with identical settings) [CHECK cause; nothing else ran at the check afterwards] |
| iteration | gradient 219 s (280 sets, 0.78 s per set: L = 11, 22 virtual coils, joint fat), step 3.5–16 s, report 41 s |
| λ_max(M N) | 6.02 (circulant metric per encode, 40 sets pooled, dynamic range 1e3) |
| f | 3.03e14 → 1.11e14 (it 1 → 2, from zero) |

The velocity RMSE rises from 0.43 to 0.49 m/s over 2 iterations from a zero start. This is expected: no warm start,
the first metric steps. It says nothing about the method.

Budget: about 4 min per iteration, so 40 iterations ≈ 2.7 h plus setup.

## Scoring tools

- `research/experiments/box_recon/coronary_roi.py` builds the per-frame coronary ROIs on the box grid. Input:
  phantom_4d's deformed vessel tree (`truth_fine/frames_geom.hv`).
  - Arteries: the coronary LBM domain (lbm_coronary.py NAMES), grouped as LM / LAD (+ septal) / LCx / RCA
    (+ inferolateral). Cardiac veins separately.
  - Core masks are radius − 0.5 voxel; centreline tangents point distally.
  - Output: `box_recon/coronary_roi.npz`, and `lumen_idx_ext_f32.npy` = the bias_truth idx + 41 719 coronary voxels,
    1 065 194 in total.
  - Geometry check: aorta centreline voxels have truth lumen fraction 1.00. Per frame: arteries 5.8–6.8 k voxels
    (core 2.5–3.0 k), veins 5.8–6.9 k.
- `research/experiments/box_recon/roi_score.py` scores the coronaries and veins:
  - velocity decode: dual-VENC 7-point, the sim's phase −k·u;
  - voxel rms error;
  - ROI-mean axial velocity curves per branch against truth_vel_logical, which includes the coronary LBM;
  - a sign check on fast LBM voxels against truth_bias u_pc.
- `frame_fit.py` (recon_diag) now reads the truth rows of extended lumen files.
- Caveat: the truth (velocities, ROIs, truth_pc_frames) is at the scan-mean breathing state r 0.49. R1 pools all
  breathing states, so the comparison is consistent. R2's bins sit at true states r 0.18 / 0.80, so honest coronary
  scores need truths at those states: phantom_4d `--resp-state` into separate directories, 2 GPU runs.

## R1: practice baseline (framerecon, 2026-11-15)

Setup: per cardiac frame (20, respiratory states pooled) × encode, plain SENSE NUFFT, circulant PCGLS, 12 coils,
50 iterations. Each (frame, encode) has 64–78 readouts. 1628 s, 13 GB.

frame_fit (the sim's own ideal PC signal, independent of any decode), slope / corr / wrapped error:

| it | pooled depth ≥ 5 | dynamics depth ≥ 5 | ROI dynamics (8 mm) depth ≥ 5 |
|---|---|---|---|
| 10 | 0.23 / 0.16 / 0.73 rad | 0.16 / 0.09 / 0.72 (truth 0.48) | 0.15 / 0.13 / 0.55 (truth 0.45) |
| 25 | 0.33 / 0.21 / 0.75 | 0.26 / 0.13 / 0.74 | 0.25 / 0.20 / 0.55 |

roi_score:
- sign check on fast LBM voxels: corr 0.05–0.12 per axis;
- coronaries: voxel rms error 30–120 cm/s against true speeds of 11–20 cm/s;
- axial ROI-mean curves: corr −0.4…0.4, errors above the flat-truth error.

**Reading.** At 1 mm the per-frame problem is undersampled by about R ≈ 50 (≈ 70 of ≈ 3600 readouts per
(frame, encode)). At 2 mm the same plain per-frame reconstruction reached corr ≈ 0.6 (recon_diag §5). The slope grows
with depth and with iterations, so this looks like real but aliasing-dominated signal, not a mapping error.
**A practitioner's per-frame reconstruction gives no usable per-frame velocity at 1 mm.** The ROI dynamics error
exceeds the truth's own dynamics.

[CHECK] A fairer practice baseline is view sharing (`framerecon --share K`, as PC-VIPR does).

## R2: full-physics temporal subspace, warm start (recon_p2np, 2026-11-16)

Setup: the R0 configuration with `--tsub fourier:4`, `--warm 20` (per-encode PCGLS on all data), 40 iterations,
zmshr1ms with `--circ-share encode`, 2 respiratory bins, joint water + fat with truth fat fractions, no spatial prior.
- Warm start finished 2026-11-15. The rest ran 2026-11-16 00:30–02:37 on the rebuilt binary (coalesced scatter / pad
  / crop kernels, research/temporal_operator.md §9). Same operator up to f32 rounding.
- L = 10 by the noise rule. Gradient 152.5 s per iteration (R0: 219.9 s at L = 11). Wall time 2 h 08 min for setup +
  40 iterations. Peak RSS 77.8 GiB under the 80 G cap: the host ledger is tight.

frame_fit (slope / corr / wrapped error; truth dyn rms 0.479 at depth ≥ 5, 0.448 for 8 mm ROIs):

| it | dynamics depth ≥ 5 | ROI dynamics (8 mm) depth ≥ 5 |
|---|---|---|
| 0 (warm: static) | 0 / 0 / 0.479 | 0 / 0 / 0.448 |
| 10 | 0.058 / 0.050 / 0.586 | 0.057 / 0.081 / 0.485 |
| 20 | 0.082 / 0.049 / 0.686 | 0.080 / 0.084 / 0.520 |
| 30 | 0.104 / 0.051 / 0.772 | 0.100 / 0.091 / 0.543 |
| 40 | 0.116 / 0.051 / 0.836 | 0.113 / 0.095 / 0.559 |
| R1, it 25 | 0.26 / 0.13 / 0.74 | 0.25 / 0.20 / 0.55 |

roi_score at it 40, per respiratory bin, against the mean-state truth (caveat above):
- sign check: corr 0.06–0.11;
- coronaries: voxel rms error 62–113 cm/s against true speeds of 11–20 cm/s;
- axial ROI-mean curves: corr −0.19 … 0.28, |err| 2–6× the flat-truth error.

The in-driver velocity RMSE grew from rel 1.77 (it 10) to 2.32 (it 40).

**Reading.** R2 does **not** beat R1. The pre-registered rule says diagnose before adding a prior.
- The dynamics slope grows slowly while the error grows faster, every 10 iterations: semi-convergence. Noise and
  aliasing enter faster than the cardiac variation is fitted.
- Hypotheses [CHECK, one factor each, dev scale first]:
  1. the per-encode circulant preconditioner amplifies the poorly determined dynamic directions;
  2. the static warm start plus an unregularised Fourier subspace at R ≈ 50 per frame;
  3. the 20-bin binning error under fourier:4 (temporal_operator.md §3: 0.09–0.35 for h = 1–4);
  4. model mismatch of the full physics.
- Cheapest first test: the same run with the per-frame metric off (plain PCGLS steps) at 2 mm, and the R1 baseline at
  the same scale.

Lumen saves (it 0–40, 12 GB) deleted after scoring. The driver removed its warm-start checkpoints
(`state_warm_done_*.hv`) at completion. `warm.npy` / `warm_vel.npy`, figs, metrics and timing CSVs remain (660 MB).

## Next

- **R1, practice baseline** (`framerecon`): done (above). Possible R1b: `--share 2`.
- **R2**: done (above). It does not beat R1. Diagnose (hypotheses above) before R3.

  Question: does the full-physics temporal subspace (no spatial prior) beat R1 on the frame_fit dynamics and the
  coronary / great-vessel ROIs?

  Decision: if yes, a prior (wavelet DFW + static lock) in R3; if no, diagnose before adding anything.
