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

## 8. Update 2026-11-20: BFP storage in the ledger (research/bfp_storage.md §9)

Block floating point (`hasty_bfp`) shrinks three ledger entries. Each one is switchable, and the default is f32 (the plan
is then the same entry for entry as before; a unit test checks this).

- **Host solver state** (`--state-store` / `HASTY_STATE_STORE`, `hasty_recon::bfpstate`):
  - x stays f32 (the master copy); g, g_prev and the secant record are BFP.
  - Size per stack: `Bfp::bytes`, i.e. (4b + 1) / 16 bytes per complex value, 4.06 B at bfp16 against 8 B.
  - The set-sized f32 temporaries of the metric steps go from two to three.
  - This is the §3.1 item ("store s and y instead of x_prev and g_prev, in reduced precision"). It is done with
    BFP, and the secant step is stored as the exact difference x_new − x_old whenever there is no post-step denoiser.
- **Gradient download:** the device packs each set's gradient before the copy (`hasty_bfp::xfer`), so the PCIe bytes
  of g fall to the same size.
- **k-space data** (`--kstage` / `HASTY_KSTAGE`, `kstage::Stager`):
  - Packed staging: the upload is BFP with the width from the σ rule, decoded on the device. Device bytes per staged
    set: the f32 set plus its packed upload.
  - `:resident` keeps every set packed on the device: the BFP size of all data plus one decoded set.
  - `--kstore bfp` is the host-RAM form of the same bytes. It needs no host decode, but zstd + q stays 3–4 × tighter
    for host or disk.
- **Ledger:** `vknot::plan_with(n, store, kstage, …)` holds these sizes; `plan` reads the two switches.

### 8.1 Dry plan, 1 mm box (R2 sizes of §3; `vknot::tests::ledger_bfp_state_and_kspace_1mm_box`)

Inputs to the plan:
- 280 sets of 8.53 M voxels; one stack 17.8 GiB;
- k-space 7.69 GiB, tables 9.30 GiB, the operator 13.4 GiB resident;
- a 23 GiB card;
- host headroom 57.9 GiB (80 G cgroup less the 16.6 GiB setup residue).

Log: `research/experiments/bfp_host/ledger_1mm.log`.

| configuration | host state | host peak | fits the host? | device peak |
|---|---|---|---|---|
| f32 (today) | 4 × 17.8 = 71.2 GiB | 73.2 GiB | no (R0 attempt 2) | 15.6 GiB |
| bfp16, objective with a gradient sink (P2np per-frame streaming) | 17.8 + 3 × 9.04 = 45.0 GiB | 47.0 GiB | yes | 15.6 GiB |
| bfp16, objective without a sink (velocity model, temporal subspace: g, g_prev stay f32) | 3 × 17.8 + 9.04 = 62.5 GiB | 64.5 GiB | no | 15.6 GiB |
| bfp16 + packed k-space staging | 45.0 GiB | 47.0 GiB | yes | 15.7 GiB |
| bfp16 + k-space packed resident on the device | 45.0 GiB | 47.0 GiB | yes | 20.0 GiB (k-space 4.42 GiB instead of 7.69) |

So the §1 failure (R0 attempt 2, 88 GB host) fits with the BFP state. It does so only for an objective that hands
its gradient over set by set. The velocity objective (`vobj::VelProblem`) needs that path too
(`Objective::gradient_host_sink`); until then the knot-mode plan counts its g and g_prev as f32. With resident packed
k-space, the data no longer need the PCIe at all and still fit the 23 GiB card next to the 1 mm operator. The f32
resident copy does not: 23.3 GiB peak with the tables streamed.

Planning assumption: a σ-rule store is planned at 18 bits (0.570 of f32). On ph0001 the rule picks 15 bits
(0.477, §9 of the BFP doc).

## 9. Update 2026-11-20: BFP grid-FFT intermediates (research/bfp_storage.md §8.1)

`HASTY_SEG_FFT_STORE = bfpN` stores the y ↔ z intermediates of the engine's grid FFT as block floating point.

| entry | size | where it is counted |
|---|---|---|
| in place (default when a plane's lines n0 · nch are a multiple of 16) | exponent bytes only: (block z planes) × n1 × ⌈n0 · nch / 16⌉ bytes, about 1/128 of the slot's image-block planes (29 MB at the 288 × 288 × 140 grid, 50 channels) | engine coil-batch budget (`fused::fit_store_budget` inside `max_grid_bytes`); the knot ledger's `fixed` term, which is measured from device memory after the operator is built |
| dense (`HASTY_SEG_FFT_BFP_LAYOUT=dense`, or n0 · nch not a multiple of 16) | (block z planes) × n1 × ⌈n0 · nch / 16⌉ × (4b + 1) bytes: 0.56 of a slot at bfp22, σ 1.25 (2.58 GB at that grid) | the same two places |

Formula for planners: `PrunedChanFft3::store_bytes_for(store, dense, [n2, n1, n0], nch, block z planes)`.

Not counted yet:
- the extra grid transforms of the multi-image passes (`multi_plan`, one exponent array or dense buffer each);
- a store switched after construction (`SegNufft::set_fft_store` does not re-budget the coil batch).

With in place both gaps are a few tens of MB; with dense they are a slot fraction each.

## 10. Update 2026-11-20: the grid slot as a ledger entry; the velocity objective's gradient sink (research/sample_kernels.md §20)

- **Knot grid slot sized by the ledger** (recon_p2np `--grid-budget auto`, now the default; `legacy` keeps the
  (free − work) / 3 ≤ 6 GiB rule). Before the knot engine is built, `hasty_recon::vknot::slot_budget` runs the knot
  plan with the operator WITHOUT its slot (resident now + coil maps + segment diagonals, per-knot ratemaps: every
  knot's + 0.5 GiB of plans / kernels), the data and tables from the sample counts (220 B of tables per sample), the
  velocity model's parameter stack, and the slot of `cb` coils as one more entry (`hasty_nufft::seg_slot_bytes`: cb × L
  channels of the fine grid plus the default store's BFP bytes). The largest batch whose plan fits (5 % of the device
  kept free for the estimate's error) bounds the choice; under it the engine planner's rule picks (fewest transformed
  channels, then passes; no ragged last batch). The real ledger still checks the measured plan before the solver; if
  the estimate refuses, the legacy budget is used.
- **In-place BFP exponents are not in `max_grid_bytes`.** The engine's own y ↔ z intermediates in place cost only the
  exponent bytes (≤ 1/128 of a slot); they live outside the coil-batch budget like the FFT plans' buffers, so a budget
  of exactly n coils keeps n coils. A dense buffer (n0 · nch not a multiple of 16, or `HASTY_SEG_FFT_BFP_LAYOUT=dense`)
  is counted in full. A multi-image pass with its own transform (cbm ≠ cb) counts that transform's bytes in full
  (`multi_plan_with`). `seg_slot_bytes` (planners) counts the exponents too. The knot ledger measures all of them in
  `fixed`.
- **Velocity objective with a gradient sink** (`VelProblem::gradient_host_sink`, at-knots mode): with a BFP host state
  (`--state-store bfpN`) g and g_prev are now stored at their width in knot mode too. `vknot::plan_sink(n, has_sink, …)`
  plans them so; the 1 mm knot-mode row of §8.1 (62.5 GiB with bfp16 because g, g_prev stayed f32) becomes the
  sink row, 45.0 GiB with bfp16 (58.3 GiB with bfp24, the P²nP recommendation). Spline mode (operator frames) keeps g
  f32 (no sink path there yet). The host state default stays f32.

## 11. Update 2026-11-21: compressed-DRAM switches (research/bfp_storage.md §10)

- **Defaults: every on-device store is f32 again** (`HASTY_SEG_FFT_STORE` default f32): no exponent bytes. The slab
  discard (`HASTY_FFT_SLAB_DISCARD`) allocates nothing. With `HASTY_SEG_FFT_STORE=bfp24` or `=auto`, the in-place
  exponent bytes are 1/128 of the slot, as before.
- **`HASTY_SEG_GRID_STORE=bfpN`.**
  - Every grid slot grows by its exponent tail: cells / 16 bytes, 1/128 of the slot (82 MB at 320³ × 50 channels).
  - `seg_slot_bytes` counts it.
  - The engine's coil-batch budget does not; it is below the 5 % margin.
- **`HASTY_SEG_MAPS_STORE` / `HASTY_SEG_DIAG_STORE`=bfpN.**
  - A dense BFP copy is kept next to the f32 maps, at (4b + 1) / 128 of their size: 0.76 × at bfp24, 0.51 × at bfp16.
  - Examples: 256³, 10 coils: maps +1.0 GB, diagonals +1.0 GB per encode at bfp24. 640 × 384 × 320: diagonals +4.8 GB
    per encode at bfp24.
  - The ledger's maps and diagonals entries must add these when the switches are on. They are off by default.
  - Dropping the f32 copies (VRAM −0.24 × / −0.49 × instead of +0.76 × / +0.51 ×) needs every pad / crop variant to decode.
