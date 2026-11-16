# Small-FOV benchmark at coronary resolution (2026-11-14)

User decision (2026-11-14): method selection on a box cropped out of the flow phantom: the heart and aortic arch,
with breathing, cardiac and vessel motion kept, at the local (fine) resolution needed for the coronaries. Not a
coarser grid: that changes partial volume, the vessel-to-voxel size ratio and the PSF.

## What exists

- `prep_sim.py --keep-box heart:M`:
  - tissue and blood sources only inside the bounding box of the heart and great-vessel labels in the slab, dilated
    by M mm;
  - the FOV is the box plus a margin;
  - B0 is computed from the whole body.
- Local-resolution sources:
  - `--src-octree`: 1 mm voxels in the heart box (+12 mm); label-homogeneous 2 / 4 / 8 mm box elements elsewhere,
    each with the exact sinc of a uniform cube with a linear field;
  - coronaries: the reference-frame LBM library (0.30 mm cells, 3³ points per blood source, `--beats-ref`).
- An earlier box run (`sim_flow_10min_box1mm`, heart:30, 1 mm, flow_thorax_1mm) stopped at bin 5000 of 60001.
  - Measured cost: 7.4 M sources per readout (3.03 M tissue + 6 × 0.73 M blood sub-sources), 4000 samples, 32 coils.
  - ≈ 2.4 s per readout, with the software pacer at duty 0.7 → ≈ 35 h for 53 k readouts.
  - (Not 770 h as I first said: I misread a cumulative timer.)

## Cost model and speed-ups

Direct sum per readout ≈ sources × samples × coils. A motion bin is 10 ms, so there is about one readout per source
configuration.

**Why not a NUFFT.** A type-3 NUFFT ("local DFT") would cost ≈ L · C · (S w³ + M w³ + N log N) against S · M · C. With
L ≈ 12 time segments (for the B0, M1 and fat terms, which depend on τ) and w = 6, the gain is only M / (L w³) ≈ 2, for
an approximation and a large code change. Not done.

Implemented (2026-11-14, not built yet: quiet mode):

| lever | how | expected gain |
|---|---|---|
| readout stride | `simphantom --readout-stride auto`: every n-th encode group, n = full-FOV volume / box FOV volume (prep_sim records `fov_full`). Keeps the per-frame undersampling and the samples per voxel of the full protocol, so image noise per voxel matches at the same absolute σ. ECG, breathing and excitation times are unchanged | ≈ 5–6× |
| tight box | `--keep-box heart:15 --keep-box-fov 15` (FOV margin 15 mm per side for the breathing excursion, ≈ 10 mm at the heart) | ≈ 1.4× |
| packed blood sub-sources | `bsig` packs its output: a source without a moment record gets 1 sub-source, not 6 (5 had zero weight: ≈ 0.24 of 0.73 M per frame); `--beat-collapse SD` also collapses covered sources with velocity sd < SD on every axis | ≈ 1.2× (more with collapse) |
| virtual sim coils | `--sim-coils auto[:EPS]`: the dominant eigenvectors of the sensitivity Gram matrix over the FOV (discarded energy ≤ 1e-8), expanded to the 32 physical coils after the sum. The coil multiply-adds dominate the kernel | ≈ 1.5–2× |
| no software pacer | leave HASTY_GPU_WATTS / DUTY unset (the user caps power with nvidia-smi -pl 350) | ≈ 1.3× |
| absolute noise σ | `--noise-sigma 39.98` (the full scan's receiver σ). The default (2 % of the data rms) depends on the object size | realism |

Estimate: 35 h / (5.7 · 1.4 · 1.2 · 1.7 · 1.3) ≈ 1.5 h; range 1.5–4 h. To be checked with a short timed run and the
`--plan` estimate.

## Open choices

- **Sequence.**
  - `flow_thorax_1mm` exists: the v1 out-and-back yarn, design FOV 230 mm.
  - The adopted protocol is v2 (centre-out, 9.5 ms window + M1-nulled rewinder, flip 15°, GIRF resonance). It has no
    1 mm design yet; a seqdesign run is minutes of CPU.
- **Reconstruction size.** A 1 mm grid over a ≈ 190 × 180 × 170 mm FOV is ≈ 5.8 M voxels, 1.4× the 2 mm full-FOV grid.
  Samples per set are ≈ 6× fewer than a full 1 mm scan. An iteration costs about as much as today's 2 mm runs: that
  is the price of 1 mm.
- **Truth.** bias_truth on the 1 mm box grid: RAM to be checked (38.5 GB for the 2 mm full grid).

Driver: `realistic_scan.sh <ph> --res 0.001 --keep-box heart:15 --keep-box-fov 15 --octree --flow-run <1 mm design>
--readout-stride auto --sim-coils auto --noise-sigma 39.98 --flow-only --tag _box1mm_b`.
