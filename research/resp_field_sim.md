# Breathing- and cardiac-dependent off-resonance in simphantom (2026-11-18)

Status-table row "ratemap per state (b, c)" of `mathematics_of_paper/forward_chain_section.tex`: the simulator now
recomputes ω from the deformed susceptibility per state, and writes the truth ratemap per state for oracle use.

## Model

- **States:** a grid (r_k, c_q). r_k = linspace(0, r_max, K), with r_max the largest r in the input's breathing table
  (the whole scan; r = 0 is end-expiration, r = 1 nominal end-inspiration, the Lujan depths reach 1.3–1.4).
  c_q = q T / Q is the template cardiac time, periodic. Default K = 6, Q = 20. Q = 1 is the breathing-only mode
  (no cardiac deformation of χ).
- **Field of a state:** prep_sim's model, with χ deformed by the full motion of the state:
  x = X + u_c(X, c) + r S(X + u_c). χ is pulled back by fixed-point iteration (12 iterations) and sampled trilinearly
  on the 1 mm atlas grid. Then the dipole convolution (pad 64, B0 along z). The heart moving against the lung changes
  the field at the heart–lung border, so cardiac deformation is included.
- **Shim:** one 2nd-order (10-term) shim, fitted on the reference pose (end-expiration) over the tissue in the
  prescribed box, exactly as prep_sim. It is held for every state, as a scanner shims once before the scan. So
  F_state = dipole(χ_state) − poly_ref.
- **Per source:** the change of the off-resonance relative to the static (prep_sim) value:
  - Tissue (Lagrangian, reference X): Δ_qk(X) = F_qk(X + u_c(X, c_q) + r_k S) − F_ref(X), trilinear. The table is
    [Q, K, n_tissue] f16; f16 precision is ≤ 0.06 Hz at 1 ppm.
  - Blood (Eulerian frames at their own cardiac time c_f, r = 0 positions p):
    Δ_k(p) = F(r_k, c_f)(p + r_k S(p)) − F_ref(p), linear in c between the two cardiac states around c_f.
    The table is [n_f, K] per frame.
- **simphantom `--resp-field <file>`** (default off; the old path is untouched):
  - Tissue: ω = ω_static + Σ_i w_i Δ_i, bilinear in the bin's (r, c) (`hasty_bench::respfield::bilinear`, kernel
    `bilerp`). The four state columns are expanded to the simulated sources (fat peaks, `--tissue-sub` boxes) and
    uploaded on use, in an LRU of 8 columns.
  - Blood frames: linear in r (kernel `rlerp`), into the ω the prep and bsig kernels read.
  - With K = Q = 1 the table is zero and ω is bit-identical to the static path (unit tests).
- **Not modelled:**
  - The box elements' field gradient (`field_grad_ppm_m`) stays the static one.
  - Blood uses its frame's cardiac time c_f, not the bin's exact c (at most half a frame, 4 ms).
  - χ of blood vs myocardium differs by 0.03 ppm only, so deforming the chambers with u_c is adequate.

## Truth ratemap per state (`--truth-dir`, evaluation / oracle)

`<scan>/truth_ratemap_bc/` holds the following, lab frame, on simphantom's reconstruction grid:

| file | content |
|---|---|
| `mask.npy` | bool [x, y, z] (F order): voxels that hold signal in any state |
| `pd_water.npy`, `pd_fat.npy` | [S, nv] float32: PD sums (nearest voxel, box elements split, as `truth_pd_*`) |
| `ff.npy` | pd_fat / (pd_water + pd_fat) |
| `r_water.npy`, `r_fat.npy` | PD-weighted mean 1/T2* of the water / fat sources (1/s); the phantom's fat has its own rate |
| `omega_hz.npy` | PD-weighted mean of the sources' field off-resonance (Hz, no chemical shift): the ω the signal sees |
| `field_hz.npy` | the lab field at the voxel centre (Hz); only when the field is recomputed |
| `meta.json` | the (b, c) grid, the layout, the definitions, the stats |

- The state index is s = q K + k: cardiac phase outer, breathing amplitude inner.
- nv = mask.sum(). Column j of each file is the j-th true voxel of the mask in F order (x fastest).
- NaN means no source of that kind in that voxel and state.
- Tissue sources sit at their state position. Blood comes from the frame nearest c_q, at p + r_k S(p).
- `--advected-only` writes the same maps for a dataset simulated with the static field. Its ω is the advected static
  field per source, so the ratemap still changes per (b, c) through tissue and fat motion.

## Magnitudes

### Breathing and cardiac field changes (ph0001)

Dev input (`sim_flow_dev_respfield`, 3.54 M tissue sources).

For comparison, the first breathing-only version, run on `sim_b0_2.5min`'s 11.9 M sources (1 mm heart ROI, plain
pull-back) at r = 1:

| region | p5 / p50 / p95 | \|Δ\| p95 |
|---|---|---|
| heart | −21.7 / +1.6 / +17.4 | 26.5 |
| liver dome | −1.5 / +8.4 / +25.4 | 25.5 |
| chest wall | −22.9 / +10.6 / +23.5 | 38.0 |

The full-input 6 × 20 numbers are in `sim_b0_2.5min_respfield/resp_field.log` once that step has run.

- Values are signal-weighted percentiles over tissue sources (spoiled GRE at 12°, TR 10 ms), in Hz at 3 T.
- "Breathing" is Δ(r = 1) − Δ(r = 0), pooled over 4 cardiac phases.
- "Cardiac" is Δ(r = 0, c_q) minus its cycle mean, pooled over 20 phases.

| region | breathing p5 / p50 / p95 | breathing \|Δ\| p95 | cardiac \|Δ\| p95 |
|---|---|---|---|
| heart (myocardium, valves, walls) | −27.6 / −4.4 / +16.2 | 30.1 | 12.9 |
| liver dome (top 30 mm of the liver) | −3.6 / +8.7 / +28.1 | 28.5 | 17.6 |
| chest wall (anterior to the heart) | −23.2 / +11.1 / +20.7 | 27.0 | 6.9 |
| all tissue | −14.4 / +4.1 / +16.6 | 20.9 | 3.1 |

- At the deepest breaths of the scan (r = 1.30) the breathing values scale by about 1.3. In the first, breathing-only
  run: heart \|Δ\| p95 33.8 Hz, liver dome 33.0 Hz, chest wall 43.6 Hz.
- Over the reconstruction grid (Eulerian, voxel level), the lab field changes by p1 / p99 = −374 / +191 Hz between
  r = 0 and r_max. These are voxels whose content changes, lung against liver or chest wall.
- The cardiac change is not negligible near the heart: 13 Hz p95 in the heart, 18 Hz at the liver dome under it.
  This is why the user asked for (b, c) states.

### Literature at 3 T

The phantom sits at the upper end of the reported range, which is consistent: the phantom's breathing depth is
13 mm SI and 8 mm AP at r = 1, up to 1.4× deeper breaths, with no shim update.

- Liver: median 4–15 Hz between end-exhale and end-inhale, 20–30 Hz peak-to-peak in earlier studies (Stelter et al.,
  MRM 2025, B0 navigator for liver T1 mapping).
- Breast at 4 T: 10–30 Hz, mean 24 Hz (≈ 0.1 ppm), up to 70 Hz next to the chest wall (Bolan et al., MRM 2004).
  That is ≈ 18 Hz typical at 3 T.
- Cervical cord: 74 Hz (0.58 ppm) at C7 between inspired and expired breath-holds (Verma et al., MRM 2014).
- Heart: the 3 T motion-adaptive shimming study (PMC12892406) finds a significant inspiration/expiration change, at
  the heart–lung interface strongest in the lateral wall, and no significant systole/diastole change of the shim-level
  field.
- Brain: ≈ 0.01 ppm, 1–3 Hz (Van de Moortele et al., MRM 2002), for scale.

The phantom's cardiac term (13 Hz p95 in the myocardium) is a local, voxel-level change at the heart–lung border. It
does not contradict the shim-level finding.

### Interpolation error of the state grid

Signal-weighted tissue \|e\| (Hz), on the dev input:

| grid | p95 | p99 | heart p95 |
|---|---|---|---|
| cardiac Q = 10 (odd nodes of Q = 20 against their neighbours) | 0.52 | 2.10 | 3.26 |
| breathing Δr = 0.52 (K ≈ 3.5, inner nodes against their neighbours) | 1.86 | 7.79 | 1.81 |
| **6 × 20 grid, 8 random off-grid states (direct fields)** | **1.09** | **4.81** | **1.75** |

- K = 6 × Q = 20 keeps the interpolation error at about 1 Hz p95, below the cardiac and breathing physics (13–30 Hz).
- Halving Q doubles the heart error.
- The pull-back converges with the damped iteration: residual ≤ 0.12 mm. The plain iteration oscillates where
  |∇u_c| ≈ 1 (residual up to 21 mm).

### Checks

- **Unit tests** (`cargo test -p hasty-bench --lib respfield`, 5 tests): K = Q = 1 reproduces ω bit for bit; r = 0
  with K > 1 is exact; the nodes are exact and linear / bilinear tables are reproduced in between; periodic wrap in c;
  f16 decoding; the source-map expansion.
- **End-to-end identity:** simphantom on the dev input, window 20–20.25 s, with a K = Q = 1 field file against
  without `--resp-field`. The k-space is bit-identical (max \|diff\| 0). With the 6 × 20 field, the same window differs
  by 10.5 % relative rms.
- **Reproduction:** the CPU-recomputed reference field matches prep_sim's within 4e-6 ppm (the first version's check).

## Datasets and commands

All datasets are under `data/hasty/phantom/factory_arms/ph0001/`. The binary snapshot is
`rustlib/target/snap/simphantom_respfield`; run that, since the shared one is rebuilt by other agents.

1. **`sim_flow_dev_respfield/`**: dev-scale 4D-flow scan with the field.
   - prep_sim `--slab auto --fov auto --res 0.002 --src-step 2 --scan-time 120` (3.54 M tissue sources,
     100 blood frames).
   - `resp_field.py … --k 6 --q 20 --check 8 --truth-dir scan/truth_ratemap_bc` (GPU, 11 min, 2.5 GB tables).
   - `simphantom … scan --seq flow --flow-run data/hasty/seq/flow_thorax --flip 12 --resp-field resp_field.hv`.
   - 32 coils, no beat library: the LRU of beats would cost ≈ 70 s of moments per heartbeat within the 40 GB cap.
     The blood keeps the frames' LBM mean velocities.
   - The truth ratemap is 7.5 GB (120 states × 2.39 M voxels × 7 files).
2. **`sim_b0_2.5min_respfield/`**: the multi-echo ratemap scan of `sim_b0_2.5min`. It uses the same input
   (`sim_b0_2.5min/sim_input.hv`), the same `--seq b0 --flip 12`, and adds `--resp-field`.
   - No beat library: the original's 15-beat cache needs an 84 GB cap.
   - Driver: `rustlib/tools/phantom/resp_field_runs.sh b0`, resumable.
3. **`sim_flow_10min/scan/truth_ratemap_bc/`**: the truth ratemap of the existing advected-field scan,
   `resp_field.py … --advected-only` (`resp_field_runs.sh adv10`). Nothing else in that scan is touched.

Steps 2 and 3 run in the detached unit `rf-respfield-datasets`:

```
systemd-run --user --unit=rf-respfield-datasets -p MemoryMax=40G -p MemorySwapMax=0 --setenv=GPU_LOCK=<gpu.lock> \
  bash -c "rustlib/tools/phantom/resp_field_runs.sh adv10 && rustlib/tools/phantom/resp_field_runs.sh b0"
```

- Logs: `sim_b0_2.5min_respfield/{resp_field,sim}.log` and `logs/truth_ratemap_bc_10min.log`.
- To resume after a stop, rerun the same command. simphantom continues from its `state_sim_bins_*` chunks.

## Status / resume (2026-11-18 ≈ 19:50, session closed)

**Done**
- Code:
  - `rustlib/tools/phantom/resp_field.py` (fields, tables, truth ratemap).
  - `rustlib/crates/hasty-bench/src/respfield.rs` (interpolation, kernels, tests).
  - simphantom `--resp-field`.
  - `rustlib/tools/phantom/resp_field_runs.sh` (driver).
  - The snapshot binary `rustlib/target/snap/simphantom_respfield`.
- Dev dataset inputs: `sim_flow_dev_respfield/{sim_input.hv, resp_field.hv, scan/truth_ratemap_bc/}`, with
  `resp_field.log` holding the stats.
- The identity check passes (bit-identical). Its outputs are in the session scratchpad, which is temporary.

**Running (detached units, GPU lock `…/scratchpad/gpu.lock`; both wait for it)**

1. `rf-dev-flow-scan`: the dev flow scan.
   - It resumes from bins 2000 / 12001, at ≈ 36 s per 250 bins: ≈ 25 min of GPU once it holds the lock.
   - Log: `sim_flow_dev_respfield/sim.log`. Done when `sim_flow_dev_respfield/scan/data_clean.hv` exists.
2. `rf-respfield-datasets`: `resp_field_runs.sh adv10`, then `resp_field_runs.sh b0`.
   - adv10 (truth ratemap of `sim_flow_10min`) takes a few minutes. Log: `logs/truth_ratemap_bc_10min.log`.
   - b0 field step takes ≈ 15 min. Log: `sim_b0_2.5min_respfield/resp_field.log`.
   - b0 scan takes ≈ 2 h of GPU (1.93e16 terms at ≈ 3e12 /s). Log: `sim_b0_2.5min_respfield/sim.log`.

**Check / resume**

```
systemctl --user status rf-dev-flow-scan rf-respfield-datasets
# after a stop / reboot, the same commands resume (finished steps are skipped):
systemd-run --user --unit=rf-dev-flow-scan --collect -p MemoryMax=40G -p MemorySwapMax=0 --working-directory=$PWD bash -c \
  "flock <gpu.lock> rustlib/target/snap/simphantom_respfield data/hasty/phantom/factory_arms/ph0001/sim_flow_dev_respfield/sim_input.hv \
   data/hasty/phantom/factory_arms/ph0001/sim_flow_dev_respfield/scan --seq flow --flow-run data/hasty/seq/flow_thorax --flip 12 \
   --resp-field data/hasty/phantom/factory_arms/ph0001/sim_flow_dev_respfield/resp_field.hv >> data/hasty/phantom/factory_arms/ph0001/sim_flow_dev_respfield/sim.log 2>&1"
systemd-run --user --unit=rf-respfield-datasets --collect -p MemoryMax=40G -p MemorySwapMax=0 --setenv=GPU_LOCK=<gpu.lock> --working-directory=$PWD \
  bash -c "rustlib/tools/phantom/resp_field_runs.sh adv10 && rustlib/tools/phantom/resp_field_runs.sh b0"
```

To stop gracefully, send SIGINT to simphantom or touch `<scan>/STOP`. It saves the current chunk.

**Next**
- After the units finish, read the b0 field stats (full input) from `sim_b0_2.5min_respfield/resp_field.log`.
- The scratchpad `gpu.lock` path is session-specific. If the scratchpad is cleared, use any shared lock file.

**Open issues**
- The b0 and dev scans run without the beat library (the 40 GB cap). Blood uses the frames' mean velocities.
- The box elements keep their static field gradient.
- The truth ratemap is large: 7.5 GB for the dev dataset, about the same for the others.
- The breathing grid's p99 interpolation error is 7.8 Hz at Δr = 0.52. K = 6 gives 1.1 Hz p95 and 4.8 Hz p99
  overall. If the operator needs better, use K = 8.
