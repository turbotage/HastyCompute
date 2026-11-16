# Memory ledger and setup cost (2026-11-15, design + measured segment-build timers)

Why: the 1 mm box reconstruction (research/box_recon.md, R0) failed three times by running out of memory, twice after
10–20 minutes of setup. Running out of memory is a planning bug, not a hardware limit. The gradient is bounded by GPU
compute and VRAM bandwidth, and moving the state over PCIe costs about 1 % of an iteration (§2). So every buffer can be
placed where it fits, at essentially full speed, provided placement is decided up front from the sizes, which are all
known within seconds of start.

## 1. Post-mortem of the R0 failures

| try | what ran out | why the planner did not prevent it |
|---|---|---|
| 1 | VRAM, building the segments (L = 27) | The segment diagonals of all 7 encodes are device-resident (`SegSegments::segph`): 7 × L × n_v × 8 B = **12.9 GiB at L = 27** (5.25 GiB at L = 11). The sample tables (8 GB) had been placed on the device before L and the diagonals were known. `plan::Need::fixed` is "whatever is resident now", so the planner never sees the diagonals, coil maps, fat maps or fine grids as placeable. |
| 2 | host RAM (cgroup 80 G), per-frame host state | x, g, x_prev, g_prev at 17.8 GiB each + 16.7 GiB setup ≈ 88 GB. The device planner has no host budget; nothing compared 88 with 80 before the 20-minute setup. |
| 3 | host RAM, temporal subspace | `tsub::Subspace` held two full frame stacks (2 × 17.8 GiB) next to the coefficients. Not in any plan (fixed in code: frame-blocked gradient, research/box_recon.md). |

Common cause: placement is decided **late and locally**, each component allocating when it is built. The tables'
host/device choice is made from set 0's measured size *during* the build; the fine-grid budget is hard-coded
(`max_grid_bytes: 6 << 30` in recon_p2np). Host memory is checked only as `available_memory()` for packing heuristics.

## 2. Streaming costs almost nothing

R0 attempt 4 (1 mm, 280 sets, host state, tables streamed). The log shows "set streaming: 560 sets, 35.6 GiB up /
35.6 GiB down" over 2 gradients. Per iteration:

| stream | bytes / iteration | time at ≈ 25 GB/s (PCIe 4 x16, pinned) [CHECK probe value] |
|---|---|---|
| state up / down | 17.8 + 17.8 GiB | 1.5 s |
| sample tables (2.42 GiB packed → 9.30 GiB) | 9.3 GiB | 0.4 s (+ zstd decode on the CPU, overlapped) |
| k-space (zstd store) | 7.7 GiB | 0.3 s |
| total | ≈ 53 GiB | ≈ 2.2 s vs a **219 s** gradient (1 %) |

Per set: 0.78 s of GPU work against ≈ 190 MB of traffic (8 ms). Each byte of state drives 242 channel FFTs (22 coils ×
11 segments) and the gridding of ≈ 167 k samples, so the arithmetic intensity is very high. Overlap requirement: transfers
on a second stream, double-buffered slots, one set ahead (the existing SetStreamer / Stager pattern). Under that rule,
**the device needs only the working set of one or two sets plus the operator's per-encode constants**.

Host RAM is limited by capacity, not bandwidth. At 1 mm with 40 frames one complex stack is 17.8 GiB.

## 3. The ledger

One object, built before any heavy work, that owns every buffer ≥ 64 MiB of the run.

**Entries.** Each entry has:
- a tag;
- a size formula in the problem dimensions (n_v, sets, samples, C, L, E, F or K, coil batch);
- the allowed tiers;
- the bytes it reads and writes per iteration;
- its lifetime (setup phase, or solver).

**Tiers:**
- **D** device resident.
- **S** device slots streamed from pinned host, double-buffered.
- **H** host resident.
- **P** host packed (zstd; read-mostly data).
- **K** disk store, read-only (k-space data, written once and reused across runs, `rustlib/target/kstore`).
- **R** recompute on use.

There is no disk tier for state: state is rewritten every iteration, and SSD writes are a standing constraint.

**Budgets:**
- Device: `cuMemGetInfo` minus a measured reserve for the CUDA context, FFT plans and kernels.
- Host: the cgroup headroom (`hasty_site::cgroup_headroom`; it already reads `memory.max` up the tree) minus the
  page-cache share that the k-space store needs to stream.

**Choice.** Pick the tier per class that minimises the predicted time per iteration, subject to both budgets. Predicted
time = max(compute, transfer) per pipeline stage. The space is small, so the brute-force enumeration of today's
`plan::choose` extends to it.

**When.** The plan must be built and checked before any heavy work:
- After the sample records (≈ 4 s, which gives the sets and samples).
- After the segment count. The `auto:noise` rule needs only the union Gram's spectrum. With the cheaper fit of §4 that
  takes seconds; otherwise plan with L_max and refine.
- **Before** the warm start, the segment build and the solver.
- `--plan-only` prints the ledger and exits. An infeasible configuration fails in seconds with the table, not after
  20 minutes.

**Enforcement:**
- Device and host buffers ≥ 64 MiB are allocated through `Ledger::alloc(tag, bytes)`, a cudarc / `HostStack` wrapper
  that checks against the plan.
- At every phase boundary it prints, per tag, planned vs actual, VRAM free and RSS.
- An unplanned allocation, or growth more than 5 % over plan, is an error that prints the ledger. There is no silent
  OOM, and no SIGKILL from the cgroup.
- A CUDA OOM anywhere also prints the ledger before failing.
- Setup temporaries are ledger entries with their phase. Examples: the per-voxel coefficient rows of the segment fit,
  the union Gram, and the table construction.

### Ledger of the 1 mm box (sizes from the formulas and logs; R2 configuration)

| entry | size | today | could be |
|---|---|---|---|
| k-space data | 7.69 GiB (6.83 stored) | K, staged per set | — |
| sample tables, 213 B/sample × 46.9 M | 9.30 GiB (2.42 packed) | P → S | D on ≥ 48 GB cards |
| segment diagonals, E × L × n_v × 8 B | 5.25 GiB (L 11) | D, all encodes | S per encode (0.75 GiB slot, sets processed in encode order), or R (§4) |
| coil maps, 22 × n_v × 8 B | 1.50 GiB | D | D |
| fat-fraction maps, 20 × n_v × 4 B | 0.68 GiB | D | D / S |
| fine grids, cb × L × 17.3 M × 8 B | 6 GiB (cb 4) | D, fixed budget | D, size from the plan |
| solver state, per frame (4 stacks) | 4 × 17.8 GiB | H → S | does not fit the 80 G host with setup: needs §3.1 or the temporal basis |
| solver state, tsub fourier:4 | 4 × 8.0 GiB + 17.8 frame buffer + 2 × 1.9 blocks | H | frame buffer → blocks only |
| setup residue | 16.6 GiB RSS after setup | H, unlabelled | itemise [CHECK what it holds] |

Device in R0: 15.4 GiB used before the solver ("8.11 GiB free of 23.51"). No log line attributes it [CHECK]; the ledger
would.

### 3.1 Host capacity: fewer full stacks

The ZMSHR1 secant pair needs only inner products and one rank-one direction:
- Store s = x − x_prev and y = g − g_prev instead of x_prev and g_prev.
- Store them in f16 / bf16 with a scale per set. That gives 2 full stacks + 2 half stacks: 53 GiB instead of 71 GiB at
  1 mm per frame.
- Accuracy of the quasi-Newton step with half-precision pairs [CHECK on the 2 mm dev scan before use].

Alternatively the temporal basis keeps only coefficient stacks (`research/temporal_operator.md`).

## 4. What the segment build does, and why it took 300–1200 s

"SegNufft … built in 1201.5 s" (R0 attempt 4) vs 300.5 s (attempt 3), with identical settings.

**The variance is CPU contention.** From 13:46 to 14:08, during attempt 4's build, my `nav_motion_sim.py` ran in the
same app.slice. The journal says it used 2 h 37 min of CPU in 22 min, i.e. ≈ 7 of the slice's 10 CPUs, leaving about 3
for recon_p2np: × 3–4 slower. Rule: one heavy CPU job at a time, my side tools included.

What the build does (recon_p2np.rs ≈ l. 1547–1900):
1. Quadratic monomial fit of the concomitant field over 8.49 M voxels.
2. Fat-fraction maps on the grid (per frame) and the six-peak spectrum.
3. `--b0-absorb 1`: a linear field polynomial moved into the per-sample tables.
4. Per-sample SNR weights (512 readouts).
5. `auto:noise` segment count: per encode, the union water + fat Gram over 943 k fit voxels (stride 9) on a 96-point
   time grid, then its eigenvalues. Result: L = 11.
6. Per encode (7×), `build_segments_svd_fat_union`:
   - **the same Gram again** (duplicate of 5) and its eigenvectors;
   - all 8.53 M voxels projected onto them (96 phase-model evaluations per voxel, one heap `Vec` per voxel), giving the
     diagonals P_l;
   - the least-squares interpolator table a_l(τ) at 1431 readout times. **For each table time the full phase model is
     re-evaluated for all 1.89 M rows** (943 k water + 943 k fat), and the 333 MB matrix Bᴴ is streamed through.
   - cardinal form, then upload of 0.75 GiB of diagonals to the device.
7. Per set (280, **sequential**): `seg_tables` on the CPU (per-sample concomitant phase, k-shift, ΔC, cardiac time),
   `absorb_field_poly`, `points` on the GPU (tile CSR, 213 B/sample), copy to the host, zstd pack.
8. Kernel autotune.

**Measured** (run T0, 2026-11-15 19:55, uncontended; phase timers `[build timer]` in recon_p2np and `[seg timer]` in
`build_segments_svd_fat_union_g` with `HASTY_SEG_TIMERS=1`; `box_recon/T0/log.txt`). Built in **306.2 s**:

| step | measured | share | work |
|---|---|---|---|
| 1 monomial fit | 1.2 s | 0 % | |
| 5 `auto:noise` rule (7 union Grams + eigenvalues) | 27.6 s | 9 % | duplicated in 6 |
| 6 per encode × 7: union Gram + eigen | 2.0 s each, 14 s | 5 % | the same Gram as 5 |
| 6 projection of all voxels (diagonals) | 4.9 s each, 34 s | 11 % | 8.53 M × 96 evaluations + per-voxel `Vec`s |
| 6 **least-squares interpolator table** | **≈ 20 s each, 140 s** | **46 %** | 1431 × 1.89 M phase evaluations + Bᴴ y per table time |
| 6 cardinal form | 4.3 s each, 30 s | 10 % | |
| 7 sample tables (280 sets, sequential) | 51.8 s | 17 % | |
| 8 autotune | 1.8 s | 1 % | |

[CHECK]: rayon runs 32 threads (i9-13900K, 32 logical) under a 10-CPU quota. CFS throttling with that many threads can
cost extra; size the pool from `cpu.max`.

How to get this to seconds. It is a dense, embarrassingly parallel evaluation of a known phase model:
1. **Least-squares table by one GEMM on the GPU**: Bᴴ Y with Y = [g_r(τ_i)], 1.89 M × 1431. Evaluating g is 2.7e9
   complex exponentials per encode (< 1 s on the GPU); the GEMM is 3e10 MAC (≈ 0.1 s).
2. **Subsample the fit.** 11 unknowns per table time against 1.89 M rows is overdetermined by 10⁵. An energy-weighted
   random subset of 20–50 k voxels should reproduce the table to ≲ 1e-4 [CHECK: compare the predicted error and the
   per-voxel residual with today's full fit].
3. Compute the union Gram once and pass the eigen-decomposition from the `auto:noise` rule to the build.
4. **Diagonals on the GPU from per-voxel parameters** (ω, 4 monomials, R2*, fat) and the 96 × L eigenvector matrix:
   9e9 MAC, < 0.1 s. Then the diagonals need not be stored for all encodes (tier R, or S per encode).
5. Tables: `seg_tables` over sets in parallel, and the GPU `points` / copy / pack pipelined over two streams.
6. **Cache the basis.** The eigenvectors (96 × L), interpolator table (1431 × L) and nodes are kilobytes, keyed by a hash
   of the inputs (field, fat maps, concomitant coefficients, L, weights). A resume then skips the fit; R2's resume
   repeats it today.

## 5. Order of work (when allowed)

1. Phase timers in the segment build: **done** (T0, §4). Next targets by measured share: the least-squares table (46 %),
   then the sequential tables (17 %), projection plus cardinal form (21 %), and the duplicate Gram (9 %).
2. Ledger skeleton: entries and formulas for the classes in §3's table, `--plan-only`, and phase-boundary reports
   (planned vs actual). No placement changes yet; it must reproduce today's choices.
3. Placement: segment diagonals per encode (S); fine-grid budget from the plan; tables tier from the plan before the
   build.
4. Segment build on the GPU (§4 items 1–4) + basis cache. Test: the basis and tables equal today's to f32 rounding on the
   2 mm scan.
5. Secant pair in half precision (§3.1), behind a flag, A/B on the 2 mm dev scan.

## 7. Update 2026-11-16: knot-major temporal operator

- The knot-major operator (research/temporal_operator.md §10) needs **one** grid slot of cb · L channels, whatever the
  B-spline degree. The interval-major one needs s = p + 1 slots, so the ledger term "grid slots" becomes
  1 × cb · L × fine grid, and cb is no longer traded against s:
  - cubic at 1 mm: cb 4 (6.1 GB) instead of cb 1;
  - interval-major cubic needed 4 × 1.5 GB at cb 1.
- The adjoint write-back now stages through shared memory. It needs no extra device memory, but the tile kernels'
  shared memory now has a floor of 8 · 64 · ((cpba · L) | 1) bytes, so `TileCfg` drops coil groups that exceed it.
- Knot-major tables: one interval-sorted tile list per unit (resp bin × encode). A 1 mm unit merges 20 cardiac sets:
  3.34 M samples, 0.7 GB of tables. That is the same total as the per-set tables, but one unit must be resident (or
  streamed) at once.
