# Convergence of the k-space simulation (2026-11-14)

User request (2026-11-14): before the box benchmark simulation, check that the simulated signal has converged in
the discretisation choices (source resolution, sub-sources per blood voxel, LBM resolution). Vary each choice and
measure the signal change on a few readouts. A change much smaller than the noise, or than what the physics model can
capture anyway, means the physics is captured.

## Pre-registration

**Question.** At the box setup:
- prep_sim `--keep-box great:15 --keep-box-fov 15 --src-octree --res 0.001`;
- v2 1 mm design `flow_thorax_v2_1mm`;
- blood frames at 1 mm, 100 per cycle;
- tissue elements of 1 mm, plus octree 2 / 4 / 8 mm away from the heart;
- six moment-matched blood sub-sources;
- coronaries sampled at 3³ points per blood source;
- 10 ms motion bins.

Does any discretisation choice change the simulated readouts by a non-negligible fraction of the noise, in the image
region it affects?

**Readouts.** Two encode groups of 7 readouts (`simphantom --window`):
- systole: t = 48.0 s, ECG phase 0.15, breathing r = 0.45;
- diastole: t = 32.2 s, phase 0.65, r = 0.96.

Each readout runs centre-out to kmax, so all k radii are covered.

**Measures** (`rustlib/tools/phantom/sim_conv.py`):
- e_σ = rms(d)/σ over the reconstruction-window samples, where d = y_var − y_ref and σ = 39.98 per complex sample (the
  full scan's receiver noise);
- e_σ per |k|/kmax band;
- image-domain ratio: the rms change over the K voxels of the region the change lives in, in units of the per-voxel
  noise of one reconstructed frame (one encode, one cardiac bin, R readouts):
  - ratio = ‖d_frame‖ / (σ √K) with ‖d_frame‖² ≈ R/n_test · ‖d_test‖²;
  - this follows from A^H A ≈ αI for the frame operator A: then ‖Δx‖² = ‖d‖²/α and the noise variance per voxel is
    σ²/α. It is independent of the coil scaling, and needs d in range(A) with K ≤ the frame's information;
  - regions: all recon voxels (8.53 M), blood (0.74 M per frame), coronary (the sources the reference library covers),
    heart.

**Yardsticks:**
- the noise (ratio 1);
- physical beat-to-beat variability: `--beat-seed 2` gives another library beat at the same time. The reconstruction
  bins over beats, so it cannot resolve this. Beat-to-beat changes are incoherent across a frame's readouts, so in the
  image they average down by about √R against a coherent discretisation error.

**Decision rule:**
- image ratio < 0.1 in the change's region: converged, keep the cheap setting;
- 0.1–1: report it, prefer the refined setting if it is affordable;
- > 1: the refined setting is required for the box simulation.

**Variants** (one factor each, against the baseline):

| # | change | tests | region |
|---|---|---|---|
| T1 | `--tissue-sub 2` (0.5 mm boxes) | tissue intravoxel motion, field, coil variation | all, heart |
| T2 | blood frames at 0.5 mm (`phantom_4d --vox 0.5`) | blood discretisation: 8 LBM cells per 1 mm voxel instead of the centre cell, vessel partial volume | blood |
| T3 | `--beat-collapse 1e3` (1 sub-source) | size of the intravoxel velocity distribution itself | blood |
| T4 | `--beat-ref-sub 5` | coronary points per source | coronary |
| T5 | `--bin-ms 2.5` | motion-bin extrapolation | all |
| T6 | `--sim-coils auto` | coil compression in the sum (speed-up) | all |
| T7 | `--beat-seed 2` | yardstick: physical beat-to-beat variability | blood |
| T8 | chamber LBM at 1 mm vs 0.5 mm (same geometry, `lbm_case --dx-mm 1`) | LBM resolution | blood |
| T9 | coronary LBM at 0.3 mm vs 0.15 mm | coronary LBM resolution | coronary |

- T1–T7 run on the existing libraries with `--beat-spin off`: the existing libraries' spin history is the v1
  protocol's (TR 11.29 ms, 12°), and v2 at 1 mm has TR ≈ 16.5 ms and 15°. The spin history changes only blood
  amplitudes, not the discretisation.
- T8 and T9 compare libraries solved with the same settings.

Fixes found on the way:
- prep_sim gave blood sources PD × fraction without the (blood voxel / 1 mm)³ factor, so 0.5 mm blood frames would
  have been 8× too bright. It now scales by (bh/h)³.
- `--keep-box heart:M` cut off the aortic arch: the atlas labels stop at the vessel roots, and the arch top is at
  z ≈ 1397 mm against the heart box top at 1345 mm. The new `great:M` adds the vessel graph's vessels of radius
  ≥ 7 mm in the slab.
- A box FOV axis wider than the full protocol's FOV is clamped to it (z: 1.3 × slab).

## Results

Run data: `data/hasty/phantom/factory_arms/ph0001/conv/`. The run directories and the queue scripts that made them
are `queue1.sh` to `queue6.sh`. The comparison tables and figures are `cmp*/conv.md` and `cmp*/figs/`. Every run has
14 readouts and costs ≈ 4.5 min on the GPU, ≈ 95 s of it setup.

### Bugs found by the study

- **Blood in partial-volume voxels had the tissue's relaxation.** prep_sim took a blood source's label from its voxel,
  and a partial-volume voxel's label is the tissue's. About 14 % of the blood volume at 1 mm is in partial-volume
  voxels, and ≈ 2.6 % had a non-blood label, e.g. lung (T2* 0.8 ms) or myocardium (T1 1.04 s). Fix: each such source
  takes the nearest blood voxel's label (cKDTree). Inputs `in_*f` are label-fixed. T1 and T3–T7 ran on the old input;
  they are differences of one factor, so the label error cancels to first order.
- **The empty-frame crash.** With `--blood-frames-keep`, a run whose bins use another frame launched a kernel with 0
  blocks (`CUDA_ERROR_INVALID_VALUE`). simphantom now asserts with a message naming the frame.
- Also the blood (bh/h)³ scaling, the arch cut-off and the FOV clamp (see Fixes above).

### Coil compression (T6): not a speed-up here

`--sim-coils auto` keeps all 32 coils at 1e-8 and at 1e-6 discarded energy (image ratio 1.2e-3). Over a 20 cm box
the body array's 32 channels are not compressible:

| coils kept | 16 | 24 | 28 |
|---|---|---|---|
| discarded energy | 1.2e-2 | 1.4e-3 | 2.7e-4 (≈ 0.3 σ per sample) |

The box simulation runs with `--sim-coils off`.

### Settled

| # | change | image ratio, region | verdict |
|---|---|---|---|
| T3 | 1 sub-source instead of 6 | 0.53, blood | the intravoxel velocity spread is real signal: keep 6 |
| T4 | coronary 5³ vs 3³ points per source | 0.12, coronary | ≈ converged; 5³ costs nothing, so use it |
| T1 | tissue sub-boxes 2³, all elements | 0.088, all | see below |
| T1o | tissue sub-boxes for the octree (≥ 2 mm) elements only | identical to T1 (e_σ 7.88e-2 both, same bands) | sub-boxing the 1 mm elements changes nothing |
| T1m | tissue sub-boxes where cardiac displacement ≥ 2 mm | 0.097, heart | converged: myocardial strain inside 1 mm elements is negligible |
| T7 | another library beat (seed 2) | 0.66, blood, but incoherent | yardstick, see below |

- **T1 / T1o.** The whole tissue change is in the 2 / 4 / 8 mm octree elements away from the heart. Those elements
  fill ≈ 3.5 M voxel-volumes, so the ratio in their own region is 0.088 · √(8.53 / 3.5) ≈ 0.14. Of that, 0.044 is
  without motion (`Bs` / `T1s`). The rest is respiratory strain across an 8 mm element.
  - This region is outside the benchmark's measurement regions. Its effect inside the heart is only its aliasing,
    which spreads, so it is much smaller still.
- **T7, the yardstick.** A different beat changes every readout of a frame differently, because the frame's ≈ 70
  readouts come from ≈ 70 heartbeats. The reconstruction sees only the mean, so the beat-to-beat part is ≈ 0.66 / √70
  ≈ 0.08 per frame.

### Time discretisation of the blood

| change | input | image ratio, blood |
|---|---|---|
| motion bins 2.5 vs 10 ms | old | 0.81 |
| motion bins 1 vs 2.5 ms | old | 0.40 |
| motion bins 2.5 vs 10 ms, blood only | old | 0.81 (= the full pair: the whole effect is blood) |
| motion bins 0.5 vs 1 ms, with dμ/dt | fixed | 0.27 |
| 200 vs 100 blood frames per cycle, 1 ms bins, with dμ/dt | fixed | 0.50 |
| beat-sub 2 vs 1 | fixed | 0.41 |
| beat-sub 3 vs 2 | old | 0.17 |

- With TR 15 ms every readout is in its own bin from ≈ 7 ms bins down, so 1 ms bins cost nothing. The library mean
  velocity is now carried to the bin's time with its rate (dμ/dt from the two library bins).
- The blood sources are already advected to first order: x += v · Δt from the frame time, and x += v · δ inside the
  readout. A pure timing error should therefore fall off fast. The bin 0.5 vs 1 change (0.27) is about twice the
  first-order extrapolation of 1 vs 2.5 (0.40 · 0.5 / 1.5 ≈ 0.13).
- Hypothesis (rejected): the frame and bin effects are the per-frame point sampling of the library moments.
  - The idea: another frame, or a bin edge that flips the nearest frame, puts the sources at other points of the
    moment field, which is the beat-sub error again.
  - Test (queue6) at beat-sub 3: 200 vs 100 frames 0.48 (0.50 at beat-sub 1); bins 0.5 vs 1 ms 0.26 (0.27).
    Unchanged, so not sampling.
- Hypothesis [CHECK] (queue8): the frame → bin advection is the error.
  - The blood frames are Eulerian: a source is the blood in its voxel at the frame time. simphantom moved it to the
    bin time by its own velocity (x += v Δt, |Δt| ≤ 4.3 ms at 100 frames). That shifts a 1 m/s jet's velocity pattern
    4 mm downstream and pushes wall voxels into the wall.
  - With dμ/dt the library velocity at the fixed voxel is already carried to the bin time: v(X, t_f) + ∂ₜv Δt is
    first-order correct. Moving the position as well adds an error of (v·∇)v Δt.
  - `--blood-advect off` keeps the frame positions. The motion inside the readout (x + v δ, the displacement artefact)
    stays.
- **Confirmed** (cmp12, 1 ms bins, beat-sub 1):

  | change | advection on | advection off |
  |---|---|---|
  | 200 vs 100 frames | 0.50 | 0.20 |
  | bins 0.5 vs 1 ms | 0.27 | 0.077 (converged) |
  | advection off vs on | — | 0.49 (the advection was itself the error) |

  The remaining 0.20 is the time quantisation of the chamber walls. With the advection off, an interior voxel is the
  same in every frame (Eulerian library cells carried by dμ/dt), and only the voxel set at the walls jumps from one
  frame to the next.
  - 200 frames in production would double the moments per beat (5.6 GB), and the 15-beat cache would no longer fit
    in 94 GB of RAM.
  - Instead, `--blood-blend on` takes the two frames around the bin, weighted linearly in time. The walls then move
    continuously, with O(Δt²) error, at ≈ 1.5× the blood sub-sources.
- **Blend test** (cmp14):

  | change | image ratio, blood |
  |---|---|
  | 200 vs 100 frames, blended | 0.115 (0.20 with the nearest frame) |
  | blended vs nearest, 100 frames | 0.15 |
  | blended vs nearest, 200 frames | 0.084 |

  The nearest frame's error halves with Δt (first order). The remaining 0.115 is probably the frames' own content
  [CHECK]: phantom_4d cleans every frame's chamber labels on its own (valves, opening, largest component).

### Blood partial volume at the chamber walls

| change | image ratio, blood |
|---|---|
| 0.5 mm vs 1 mm blood frames, label-fixed, 10 ms bins (T2f10 vs Bf10) | 0.51 |
| the same with the old labels (T2 vs B) | 0.62 |

- phantom_4d makes chamber blood binary per voxel (`bfrac = where(chamber, 1, Bf)`). Only the vessels get a
  sub-sampled fraction (phantomflow `--sub 3`).
- At 1 mm the endocardial wall is therefore a 1 mm staircase. The 0.5 mm frames come from the same 1 mm label map
  (`--labels-src tissue_1mm.hv`) and draw a 2× finer staircase. The difference is a quadrature error, not a different
  anatomy.
- Object space, 1 mm frames vs the 0.5 mm frames binned into 1 mm voxels (frames 27 and 72):
  - total blood magnetisation equal to ±0.2 %;
  - rms per-voxel difference 9–10 % of a full voxel;
  - |Δ| > 25 % in 3 % of the voxels;
  - 2.6–2.8 % of the magnetisation sits in other voxels;
  - the children's blood centroid is up to 0.31 mm (p90) off the voxel centre.
- Fix: `prep_sim --blood-collapse 2` makes one 1 mm source per 2³ voxels of the 0.5 mm frames. Its magnetisation is
  the children's sum, its position their centroid, its velocity their weighted mean. It costs the same as 1 mm in the
  sum.
- Test (cmp13; 10 ms bins, advection off; the reference is the 0.5 mm frames):

  | 1 mm sources | image ratio, blood | e_σ per k band (0–0.1 … 0.75–1) |
  |---|---|---|
  | 1 mm frames | 0.52 | 0.65 / 0.27 / 0.10 / 0.077 / 0.090 |
  | collapsed, beat-sub 1 | 0.42 | 0.45 / 0.23 / 0.087 / 0.070 / 0.085 |
  | collapsed, beat-sub 2 | 0.25 | 0.11 / 0.040 / 0.041 / 0.052 / 0.077 |
  | collapsed, beat-sub 2, advection on (vs 0.5 mm with it on) | 0.24 | |

  - A collapsed source must sample its children's library cells. Beat-sub 2 puts its 2³ points at the children's
    positions for an interior voxel.
  - With it, the low-k error is gone (band 0–0.1: 0.65 → 0.11). What remains grows with |k|.
- Hypothesis [CHECK]: the high-k remainder is the point-source lattice.
  - Blood sources are points, while the tissue uses sinc box elements. A lattice of points at h = 1 mm has its first
    replica at 2π/h, which is −kmax for k = kmax, so it aliases near kmax. A 0.5 mm lattice barely does.
  - `--blood-box on` makes every blood (sub-)source a cube of the blood voxel's edge (the tissue's sinc form factor).
- **Box test** (cmp15):

  | comparison | image ratio, blood |
  |---|---|
  | collapsed + beat-sub 2 vs 0.5 mm, both as boxes | 0.20 (0.25 as points) |
  | box vs point, 0.5 mm frames | 0.025 (the 0.5 mm lattice is converged) |
  | box vs point, collapsed 1 mm | 0.11 |

  - The boxes remove part of the high-k error: band 0.75–1 goes from 0.077 to 0.059.
  - The low-k remainder (0.11) stays. Hypothesis [CHECK]: one 1 mm source carries a single velocity mixture of its
    eight children, which loses the correlation between position and velocity inside the voxel.
  - The next step would be 0.5 mm sources in production. They need 8× the GPU and host memory (100 frames × 5.3 M
    sources, ≈ 22 GB of moments per beat), which does not fit. The study stops here at 0.20.
- The coronaries are not affected by the staircase: vessel fractions are sub-sampled, and the epicardial coronaries are
  ≥ 8 mm from the endocardium.

### Tissue sub-boxes

| change | image ratio, octree region (≈ 3.5 M voxel-volumes) |
|---|---|
| sub-boxes for all elements (T1) | 0.137 |
| sub-boxes for the ≥ 2 mm elements (T1o) | 0.137 (identical) |
| sub-boxes for the 4 / 8 mm elements only (T1o3: 21 k elements, +5 % sources) | 0.130 |

The 2 mm elements leave √(0.137² − 0.130²) ≈ 0.04. Production: `--tissue-sub 2 --tissue-sub-min-mm 3`.

### LBM resolution (v2-spin libraries, spin history on)

| change | image ratio, region |
|---|---|
| T9: coronary LBM 0.3 vs 0.15 mm | 0.155, coronary |
| T8: chamber LBM 1 vs 0.5 mm | 0.91, blood |
| T7s: another library beat (seed 2) | 0.50, blood |

- **T9.** At second order, a step of 0.155 leaves ≈ 0.05 between 0.15 mm and converged. Converged.
- **T8.** This includes chaos: another resolution is another turbulent realisation, like another beat. A rough split is
  √(0.91² − 0.50²) ≈ 0.77 systematic, but only 2 heartbeats are in the test. Queue7 averages over seeds 1–4 to
  separate the systematic part.
- **T8 over seeds 1–4:**

  | quantity | image ratio, blood |
  |---|---|
  | per seed | 0.905–0.936 |
  | mean over the 4 seeds | 0.81 |
  | chaos per seed | 0.49 |
  | **systematic, unbiased** | **0.78** |

  - Chamber LBM 1 mm → 0.5 mm changes the mean flow coherently.
  - Converged 0.5 mm would need a 0.25 mm LBM: ≈ 16× the 72 min, and more GPU memory than the 24 GB.
  - This is flow realism, not simulator consistency: the truth maps (bias_truth) come from the same library as the
    signal, so the benchmark stays self-consistent at either resolution. The coronaries have their own 0.15 mm
    library (T9: converged).

## Decision: the box simulation

`box_scan.sh` defaults (SIMDISC), with every refinement that was affordable:

| setting | flag | evidence: refined vs this setting |
|---|---|---|
| 1 ms motion bins | `--bin-ms 1` | 0.5 vs 1 ms: 0.077 (advection off) |
| Eulerian blood frames | `--blood-advect off` | the advection was itself an error of 0.49 |
| frames blended in time | `--blood-blend on` | 200 vs 100 frames: 0.115 (nearest: 0.20) |
| 1 mm sources collapsed from 0.5 mm frames | prep_sim `--blood-collapse 2` | vs 0.5 mm sources: 0.20 (1 mm frames: 0.52) |
| blood sources as boxes | `--blood-box on` | in the 0.20 above |
| 2³ library points per blood source | `--beat-sub 2` | 3 vs 2: 0.17 (old input) |
| 5³ points per coronary source | `--beat-ref-sub 5` | 5³ vs 3³: 0.12 |
| 6 moment-matched sub-sources | default | 1 vs 6: 0.53, which is real signal |
| sub-boxes for the 4 / 8 mm octree elements | `--tissue-sub 2 --tissue-sub-min-mm 3` | rest (2 mm): ≈ 0.04 |
| no tissue sub-boxes elsewhere | — | moving heart elements: 0.097 |
| chamber LBM 0.5 mm, coronary LBM 0.15 mm | libraries `run_lib_v2`, `run_cor_v2` | coronary: ≈ 0.05; chamber: 1 → 0.5 mm 0.78 (not converged, unaffordable) |
| all 32 coils | `--sim-coils off` | compression keeps 32 |

- Remaining, larger than 0.1 per frame:
  - the chamber LBM resolution (flow realism, consistent with the truths);
  - blood partial volume at 1 mm sources (0.20);
  - the blood frames' content (0.115).
- All of these are blood in the chambers and great vessels. The coronary items are converged: T4, T9, and the
  vessel fractions are sub-sampled.
- For scale: the reconstruction pools ≈ 70 heartbeats per frame, so the physical beat-to-beat variability is
  ≈ 0.5 / √70 ≈ 0.06 per frame.
- Input: `sim_flow_10min_box1mm_b/sim_input.hv` (see PROVENANCE.txt there).
- Cost: ≈ 0.4 s per readout × 9.9 k readouts, plus 15 library beats × ≈ 70 s of moments, ≈ 1.5 h.

### Production run (2026-11-14/15)

- Wall time 6583 s ≈ 1 h 50 min for 9891 readouts, from bin 4750 on with the loader fix below. Rate ≈ 10.7 s per
  scan second, GPU 97 %.
- Beat loader fix: `BeatLibrary::load_bin` read the whole beat file (≈ 1.6 GB) for every bin, ≈ 85 s of idle GPU per
  new library beat. It now keeps a block index per beat file and reads only the bin's block, and `prefetch` loads a
  beat's bins in parallel.
  - Check run PRODw2: 305 → 136 s; difference to the old loader 3e-5 σ rms (summation order only).
- Memory: ≈ 80 GB resident at `--beat-cache 15`, cgroup peak 83.9 of the 84 GB cap. That is more than the ≈ 57 GB
  estimate; the beat cache is most of it. [CHECK] which part of the rest is the blood frames on the host.
- The first finish was OOM-killed while writing the outputs: the beat cache was still held when the true coil maps
  (4.4 GB, plus a 2.2 GB byte copy) were built. Fixes in simphantom:
  - the beat source and its caches are dropped before the outputs;
  - a run that finds a complete `data_clean.npy` (and no `.part`) resumes from it: every chunk is done, only the
    outputs are written;
  - sample tables already on disk with the right size are kept.

  `box_scan.sh` now marks the sim step done by `meta.json`, which is written last. The resumed run gives the same
  data rms (6.0426e2), so the data file is unchanged.
- No noisy copy is written (no `--write-noisy`); the noise σ 39.98 is in `meta.json`.

### Truths for the box scan (2026-11-15)

- Bug (mine): the first fix above dropped the beat source before `meta.json` was written, so `meta.json` lost its
  `"beats"` record. Then:
  - `sim_truth.py` fell back to potential flow (no LBM);
  - `bias_truth.py` stopped with "not a --beats scan".

  Fix: the beat record is built before the drop. The resumed finalisation now starts its chunk loop at the end
  (`b0 = nbins`). Before, the cleared chunk states would have made it re-simulate everything.
- `phantom_4d.py`'s `LbmFlow` needed `flow_bins.f32`, which `box_scan.sh` deletes after the v2 LBM runs. Without the
  file it now sums the library beats (ww = v₀·s·fct³ and ww·u per cell and bin), which equals the accumulated
  `flow_bins` to f16 rounding (`lbm_beats.py`).
- `phantom_4d.py` hit a CUDA OOM on the box: 19.3 GB allocated plus 2.8 GB fragmentation. The coronary LBM field is
  ≈ 0.75 GB per bin, and four were cached. Fixes:
  - the bin cache is now two bins;
  - `sim_truth.py` runs it with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `sim_truth.py` was OOM-killed at its 48 GB cap: it made eager f32 copies of the 0.5 mm fine grid (68 M voxels × 20
  frames). It now reads the fine truth frame by frame (`open_lazy`). An `args.json` stamp, or `--reuse-fine`, skips
  re-running `phantom_4d` when the command is unchanged.
- Sanity check of the 1 mm truth (198 × 189 × 228, 20 frames, r 0.49, 705 686 blood voxels):
  - LBM coverage 0.98–0.995 of the blood per frame;
  - ≈ 52–56 k coronary reference voxels;
  - speeds as in the full-FOV truth (box frame 5: p99 1.98, p99.9 4.26, max 6.10 m/s; full scan: p99 1.80,
    max 5.88 m/s);
  - ≈ 0.25 % of the blood voxels are above 3 m/s at systole. [CHECK] where these voxels are.
- `truth_bias.npz` (451 MB; 11.6 M covered voxel-frames):
  - |u_vol| median 0.139 m/s;
  - signal weighting |κ1 − ū|: median 0.03 cm/s, p95 0.54 cm/s;
  - PC skew + mixing |v̂ − κ1|: median 0.03 cm/s, p95 1.31 cm/s.

  The large values sit at vessel rims (aortic ring ≈ 5 cm/s) and in jets.
