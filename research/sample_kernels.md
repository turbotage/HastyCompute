# Sample-side kernels of the segmented NUFFT: exact-support designs (prototype, 2026-11-19)

Status (2026-11-20; §18 = tenth session (grid sizes): one-frame gradient (forward + adjoint, FFTs included, one (frame, encode) set of 1.52 M samples, 300 W) at 192³ / 256³ / 320³ / 512³ / 640×384×320 measured and tuned; static order 3 through the knot kernels (one-bin knot engine, HASTY_SEG_STATIC_KNOT, default off): order 3 −16 … −40 % (192³ 411 → 295 ms, 256³ 737 → 481, 320³ 1105 → 924, 640×384×320 per coil 280 → 208); segment batches inside a coil (SegSplit) make 512³ run (per coil 608 ms at 2 segments per batch); planner plan_batch picks the largest equal batch that fits (the measured optimum everywhere; ragged batches 5–10 % slower); grid FFT passes at 84–91 % of DRAM peak, slabs L2-limited; tests, sanitizer clean, regression as known (+ a flaky warp-adjoint bound). §17 = ninth session: knot-path layout. "Stage each sample once for all its knots" counted and measured to need more sample visits at a fixed grid budget (the coil batch is the lever: 5 → 10 coils = −16 % dense gradient); built instead: skip of empty tiles (HASTY_SEG_KL, autotuned) and a whole-tile kl adjoint; velocity-recon config 2-D 6:1,16:3 o3 gradient −1.3 % (13.3 M per encode) / −3.4 % (1.33 M) at 300 W; tests, control and sanitizer clean. §16 = eighth session: ra and vt integrated in the engine as autotuned variants (HASTY_SEG_VSA, HASTY_SEG_VT): static 4.3 M o2 gradient 772 → 558 ms, knot o2 / o3 gradient −4.6 / −5.2 % (ra only; vt does not win on the knot path: items ≈ 4× emptier); tests and sanitizer clean. §15 = seventh session: adjoint design 7 `ra` (row-blocked output tiles, exact rows, lanes = channel blocks × x-groups): 1.35–2.2× the production knot adjoint at order 0, 1.6× at order 2, 1.8× at order 3 (4 M 10 × 10: 190 vs 342 ms); accumulator-stationary duals ruled out by a measured RED rate (≈ 240 G float/s); not yet in the engine. §13 = fifth session: vs with a per-(sample, plane) complex-weight table (vt): order 2 165.6 ms vs production 236.6 (1.43×), order 3 186 ms vs 299.8 (1.59×) at 4 M 10 × 10; order-0 L2 / occupancy tunables worth 0–3 %; latency-bound at 7 warps / SM. §12 = fourth session: design 6 `vs`, value-stationary rows: 1.4–1.9× ps, 1.9× the production knot forward at 4 M 10 × 10 o0, 25–30 % of peak; order 2 no win; sanitizer clean. §11 = third session: limiter measured): **design 5 (`ps`, plane-streamed home tile) written, compiled (nvcc sm_89: no
spills), queued with its smoke gate, the knot adjoint acb blocking and the ablation as one detached unit
(`proto-ps`, §8); not yet measured** — the GPU lock was held by another agent's recon the whole session. Measured so
far: the `base` rows and one `gm` / `halo` row of §4.

Question: is there a faster, architecture-tunable form of the gather (forward) and scatter (adjoint) than today's TT = 4
tile kernels (segnufft.cu `seg_forward_t` / `seg_adjoint_t`), which spend ≈ half their MACs on zero weights
(temporal_operator.md §9) and whose forward time follows its atomic count, ≈ 34–40 atomics per (sample, coil)
(coil_segment_compression.md §10.2, §12)? Decision this answers: whether to build an exact-support kernel family into
the engine, and with which shape parameters per GPU.

## 1. Designs

All prototypes use exact W³ supports (no zero-weight MACs) and write each (sample, coil) output once (plain store when a
thread owns all NSEG segments of its coils, else one atomic per (sample, coil piece); no per-tile atomics).

| # | name | idea |
|---|---|---|
| 1 | `base` | today's engine tile kernels, timed through `SegNufft::autotune` (kernel time only, one coil batch = all coils, static) |
| 2 | `halo` (`pg_fwd_halo`) | home tile: sample j belongs to the TT³ tile holding its support corner i0 = ceil(sc − W/2); the block stages the (TT+W−1)³ halo of a channel chunk (CH channels) in shared memory and finishes every one of its samples' outputs for that chunk |
| 3 | `gm` (`pg_fwd_gm`) | the same exact-support loop reading the grid through L1/L2 (channel-contiguous float4 loads, `__ldg`, L1 carve-out preferred), samples sorted by home tile; no staging |
| 2b/3b | `SB`, `RX` | **sample blocking**: a work unit holds ≤ SB samples with the same (j0, k0) support rows and i0 in one RX-wide bin; the unit loops over the union (W+RX−1)·W·W and every value load serves all SB samples (zero weights only in the RX − 1 extra x cells and in dummy slots) |
| 4 | `smspread` (`pg_adj_sm`) | adjoint dual of 2: samples spread into a shared-memory halo accumulator (shared atomics), halo added to the grid with global atomics (grid zeroed first) |
| 5 | `ps` (`pg_fwd_ps`) | home tile TX × TY × TZ, halo z-planes streamed through a double-buffered shared plane, lanes = (sample slot, channel group) with broadcast plane loads, accumulators across planes in registers, one store per (sample, coil) (§6) |
| 6 | `vs` (`pg_fwd_vs`) | home tile; lanes own channels and hold 5 grid rows × (TX + 9) values in registers, samples visited one per warp with broadcast weights (exact x / y / z), accumulators per (sample, channel) in shared memory, no barriers in the main loop (§12) |

Tunables (all NVRTC defines, autotuned over a candidate set; nothing architecture-specific, compiles sm_80–sm_120):
TT, CH (channels per chunk, divides NCH), TPS (threads per sample/unit; FC = CH/TPS channels per thread), NT (block
threads), SMAX (samples per work item), PADC (halo point-stride padding), SB, RX. Weights: order 0 real separable
φ(zx)φ(zy)φ(zz); order 2 the complex second-order corrected weights with exactly the arithmetic of
`stage_weights` in segnufft.cu (5 kernel derivatives per axis, 15 e-coefficients).

## 2. Why exact supports alone are not enough (the bound that decides the design)

Per useful complex MAC an exact-support kernel needs one distinct 8 B grid value per lane (each value is used once per
sample). An SM issues 4 warp-FFMA per clock but serves one 128 B shared-memory / L1 wavefront per clock; a warp-wide
8 B load is 2 wavefronts. So with one sample per value load:
- order 0 (real weight, 2 FFMA per MAC): ≤ 25 % of FP32 FMA peak; order 2 (complex weight, 4 FFMA): ≤ 50 %.
- Time floor = M · W³ · NCH · 8 B / (L1/shared bandwidth ≈ 128 B/clk/SM ≈ 41 TB/s on the 4090), e.g. 4 M samples,
  100 channels: ≈ 78 ms per pass (both orders).
- Today's tile kernel instead broadcasts each staged tile value to all lanes working on different samples (one
  wavefront serves the warp) and pays 2.2× MACs (TT = 4 tiles under W = 10) plus the per-tile atomics; at ≈ 60 % of peak
  it would take ≈ 71 ms (real weights) / ≈ 142 ms (complex weights) for the same MACs, before atomics.

So exact supports win clearly only together with **value reuse across samples in registers** (design 2b/3b) or at
order ≥ 1 (complex weights double the MAC cost of the zero-weight waste). The reuse is available because real
readouts are oversampled along the readout (measured below).

## 3. Sample clustering of the real trajectory (measured, CPU, `--mode stats`)

Trajectory: encode-0 readouts of data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/samples.hv (contiguous
block of rotations from rotation 3796), k scaled so max |k_a| → 0.98 π on the 200³ grid (the 2 mm yarn ball fills the
1 mm grid; its centre-dense shape is kept). Log: research/experiments/sample_kernels/stats.log.

| samples | RX | groups (same j0,k0, i0 bin) | samples / group | in groups ≥ 2 | ≥ 4 | ≥ 8 | x-union efficiency W/(W+RX−1) |
|---|---|---|---|---|---|---|---|
| 4.0 M | 1 | 1.97 M | 2.04 | 0.79 | 0.27 | 0.05 | 1.00 |
| 4.0 M | 2 | 1.40 M | 2.86 | 0.91 | 0.52 | 0.12 | 0.91 |
| 4.0 M | 4 | 0.90 M | 4.46 | 0.97 | 0.78 | 0.31 | 0.77 |
| 1.3 M | 1 | 0.79 M | 1.65 | 0.67 | 0.12 | 0.03 | 1.00 |
| 1.3 M | 2 | 0.64 M | 2.03 | 0.79 | 0.25 | 0.05 | 0.91 |

TT = 4 tiles: 66 k of 125 k occupied (the yarn ball fills the sphere), median 57 samples per tile at 4 M (18 at 1.3 M),
max 12 516 (k-space centre) → items are cut at SMAX samples.

Expected effect: at 4 M with RX = 2 a unit of SB ≈ 3 cuts the value loads per useful MAC ≈ 2.6× (2.86 × 0.91), which
moves the order-0 load bound from 25 % to ≈ 65 % of peak.

## 4. Measurements (RTX 4090, 2.56 GHz, 42.0 T FMA/s; out/results.jsonl)

% = useful FMAs (2 per complex MAC at order 0, 4 at order 2) / FP32 FMA peak.

| M | coils × seg | order | base fwd (ms, %) | base adj (ms, %) | gm fwd | halo fwd | ps fwd | rel. err |
|---|---|---|---|---|---|---|---|---|
| 4 M | 16 × 1 | 0 | 75.3 (4.0) | 75.9 (4.0) | | | queued | |
| 4 M | 10 × 10 | 0 | 201.7 (9.4) | 342.0 (5.6) | queued | | queued | |
| 4 M | 48 × 1 | 0 | 218.3 (4.2) | 217.6 (4.2) | queued | | queued | |
| 4 M | 16 × 12 | 0 | 336.5 (10.9) | 338.8 (10.8) | | | queued | |
| 4 M | 10 × 10 | 2 | 235.9 (16.1) | 375.5 (10.1) | | | queued | |
| 1.3 M | 16 × 1 | 0 | 22.6 (4.4) | 24.3 (4.1) | | | | |
| 1.3 M | 10 × 10 | 0 | 68.9 (9.0) | 114.8 (5.4) | | | queued | |
| 1.3 M | 48 × 1 | 0 | (timed out) | | 33.5 (8.9; tt 8, 16 ch, sb 2 rx 2) | 44.8 (6.6; tt 6) | queued | 7.9e-7 |
| 1.3 M | 192 × 1 | 0 | 264.8 (4.5) | 292.6 (4.1) | | | queued | |

Base shapes: forward kv 0 (complex weights at order 0), cpbf 6–24, fcb 1–4; adjoint cpba = batch / acb 4–8 for nseg 1.
The earlier 4 M `fwd` steps and the smoke steps failed on the timing-event bug (fixed); the 1.3 M 48 × 1 `base` step
hit the driver timeout, which then counted the wait for the GPU lock (the new driver starts the timeout under the lock).

## 5. Files

- `rustlib/crates/hasty-nufft/src/proto_gather.cu` — kernels `pg_fwd_halo` / `pg_fwd_gm` (one source, `HALO`
  define), `pg_adj_sm`, `pg_fill` (hash grid values shared with the host reference). Not referenced by the engine.
- `rustlib/crates/hasty-bench/src/bin/proto_gather.rs` — trajectory loader, sorting / units / items, autotune, f64
  reference, `--mode base|fwd|adj|stats`, JSON lines (`--json`).
- `research/experiments/sample_kernels/run_all.sh` — the first driver (designs 1–4); `run1.sh` one locked invocation.
- `research/experiments/sample_kernels/run_ps.sh` — the second driver (§8): ps smoke gate, `seg_knot_kv` GPU test,
  ps autotune per case, gm at 4 M, ablation of today's kernels, knot-major kernels on one knot (`--knot`).
- `proto_gather --mode base --knot`: today's production knot-major kernels on a 2-bin (degree-0) knot engine, all
  samples in bin 0, knot 0 timed by the engine autotune (variants 0/1; HASTY_SEG_KV restricts). `--tag` labels JSON rows.
- `research/experiments/sample_kernels/run_ps_diag.sh` (ps ablations, FSB probes, 16 × 12) and `run_ps_fix.sh` (fix
  probes, `ldsbench`, `AUTOTUNE=1` re-tune), §11. `--fixed tx,ty,tz,ncg,fc,fsb,nt[,acfg[,pfs[,minb]]]`; `PG_DEFS=` adds
  ablation defines; the `RES` / `WORK` lines print registers, local bytes, blocks per SM and the issued-work counts.
- Outputs: `research/experiments/sample_kernels/out/<step>.log`, `results.jsonl`, `<step>.done`.

## 6. Design 5 `ps`: plane-streamed home tile (2026-11-19)

Why (counted before writing it). Every design without a shared-memory broadcast has one grid-value load per useful
MAC per sample in flight, so its intensity is the number of samples sharing a support (§2–3: 1.6–2.9) and it stays
L1/L2-bound below ≈ 25 % at order 0. Today's tile kernels have the broadcast (one smem wavefront feeds the whole warp)
but multiply 64 % zero weights at TT = 4 and pay one atomic per (sample, coil, tile). `ps` keeps the broadcast and
removes the other two:
- home tile TX × TY × TZ (sample = tile of its wrapped support corner, sorted (z, y, x) inside); the block streams the
  TZ + W − 1 halo z-planes of a channel chunk through a double-buffered shared plane ((TX+W−1)(TY+W−1) points × CHP
  channels; register prefetch of the next plane, one barrier per plane), so the halo depth costs no shared memory;
- lanes = (sample slot, channel group), channel group fastest: each float4 plane load (2 channels) is broadcast to the
  32 / NCG slots of the warp; FSB samples per slot reuse it in registers;
- x: the halo row unrolled (HX = TX + W − 1) with the sample's x weights in registers (zeros outside); y: the warp's
  union of support rows; z: only planes where a sample of the warp is active (warp-uniform skip); separable order 0
  (row partial Σ_x φx v, then × φy φz: no per-point weight multiply); CORR 2: complex weights per point from the row
  factors R (5), Q (3) of `stage_weights`;
- accumulators live in registers across the planes: one store per (sample, coil) — no atomics when FC holds whole
  coils (FC % NSEG = 0), else one atomic per (sample, coil piece).

Expected useful fraction of the issued FMAs (TX = 4): x 10/13 × y ≈ 0.85 × z ≈ 0.9 ≈ 0.6 (TX = 2: ≈ 0.7) vs 0.36
today; per point per lane FC/2 LDS.128 + 2 FC FSB FFMA (FMA-bound for FC ≥ 4 or FSB ≥ 2). Halo staging from L2
(`--mode stats`, items of ≤ 128 samples; useful FMA at order 0):

| home tile | 4 M: samples / tile (sample-weighted) | 4 M: staging B / FMA | 1.3 M: samples / tile | 1.3 M: B / FMA |
|---|---|---|---|---|
| 2 × 4 × 4 | 247 | 0.243 | 84 | 0.683 |
| 4 × 4 × 4 | 342 | 0.150 | 115 | 0.432 |
| 4 × 4 × 8 | 678 | 0.134 | 225 | 0.299 |
| 4 × 8 × 8 | 1347 | 0.150 | 444 | 0.216 |
| 8 × 8 × 8 | 2683 | 0.177 | 880 | 0.226 |

The 4090's L2 delivers ≈ 15–20 B/clk/SM (≈ 5–6 TB/s, literature value, not measured here) against 128 FMA/clk/SM,
i.e. ≈ 0.12–0.16 B/FMA at full FMA rate: at 4 M the staging allows ≈ 80 % of peak, at 1.3 M ≈ 40–55 % with the deeper
tiles (4 × 8 × 8). So the prediction to check: ps ≥ 30–40 % of peak at 4 M order 0 (vs 4–11 % base), the deeper tiles
winning at 1.3 M; order 2 roughly half the order-0 speed per MAC (complex weights, 8 weight FMA per point and sample).

Tunables (NVRTC defines, `--fixed tx,ty,tz,ncg,fc,fsb,nt`): TX, TY, TZ, NCG (divides 32), FC (even), FSB, NT;
SMAX = NT / NCG · FSB samples per item. Registers (nvcc sm_89, no spills): 95–125 at FC 2, 253 at FC 10 / FSB 2.

## 7. Knot variant-1 adjoint: coil register blocking (engine change, 2026-11-19, test queued)

The variant-1 adjoint (segnufft_knot.cu, sub-box warps, z = y a β) required acb = 1: at nseg = 1 each thread does
2 FFMA per (sample, point) against 2 shared loads (weight + z), 8× below the FMA:LSU balance. Now ACB consecutive coils
per thread (ACB · NSEG accumulators, float4 z loads over the coil run, `adj_store<1, CPBA, ACB>`); the warp's coil
groups count thread groups (CGA = CPBA / ACB). Rust: `TileCfg::fits` allows acb > 1 with kv ≥ 1 and the autotune times
it; `default_kv` still requires acb = 1, so no default changes until measured. Test: `tests/seg_knot_kv.rs` gains
6 acb shapes (cpba 2–8, acb 2–8) against variant 0 (forward, adjoint, adjointness; same bounds), queued in `run_ps.sh`.

## 8. Queue and resume

    systemd-run --user --unit=proto-ps --collect -p MemoryMax=24G -p MemorySwapMax=0 \
      /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels/run_ps.sh

- Steps (out/<step>.log, results.jsonl rows tagged by step): `smoke_ps_*` (4 fixed shapes, orders 0/2, own and atomic
  outputs); gate: every BEST rel err < 1e-5, else no ps tuning; `test_kv` (seg_knot_kv binary); `ps_<case>` autotune
  for 9 cases; `gm_` at 4 M; `abl_{noatomic,nomac,nostage}_<case>` (today's static kernels autotuned under each
  PROF_ switch: confirms (a) atomics, (b) MACs, (c) staging shares); `knot_kv{0,1}_<case>` (production kernels on
  one knot; kv 1 includes acb > 1).
- Each step runs under the session gpu.lock, timeout 900 s from acquiring it; `touch STOP` stops between steps;
  finished steps are skipped on restart.
- Then: fill §4; decide integration (next section).

## 9. Integration plan (if ps wins) and portability

- Tables: home-tile CSR in `SegPoints` next to the tile lists (static: (tile, first, count) items over samples sorted
  (tile, z, y, x); knot-major: sorted (tile, interval, z, y, x) and filtered per knot as `knot_window`, ≤ 2 runs).
- Kernel: a new forward variant (TileCfg::kv = 3, or a separate `PsCfg {tx, ty, tz, ncg, fc, fsb, nt}`), record staging
  as `stage_kv` (a_l(τ), φ0, β folded), output y += o (one owner per (sample, coil): plain read-modify-write), autotuned
  over the tunables above; never replaces variants 0–2.
- The adjoint keeps output-driven tiles (a ps dual needs one global atomic per (halo point, channel, item): ≈ 15 G at
  4 M × 100 channels); its levers are §7 and fewer zero weights.
- Portability (expected; to be measured): the staging budget is L2 bytes per FMA ≈ L2 bandwidth / (SMs · clock · FMA
  per clock). RTX 5090 is close to the 4090 (≈ 0.12–0.15) → deep tiles (4 × 8 × 8); H100 / GH200 (≈ 0.3) and A100
  (≈ 0.5) have 2–4× more L2 per FMA → smaller tiles (4 × 4 × 4) and larger items are fine. Shared memory per SM (228 KB
  H100 / GH200, 164 KB A100, 100 KB 4090, 128 KB 5090) sets CHP and blocks per SM: plane = 2 (TX+9)(TY+9) CHP · 8 B, so
  H100 fits CHP ≈ 2× the 4090's at the same occupancy. FP32:bandwidth only enters through the B/FMA budget above.

## 10. Results of the 2026-11-19 queue (run_ps.sh, finished 12:42; out/results.jsonl)

Correctness: all 4 ps smoke shapes rel err 6.6e-7 (order 0) / 1.0e-6 (order 2) against the f64 reference; seg_knot_kv
(incl. the 6 acb > 1 shapes) passed. ps at 4 M, 16 × 12 crashed with CUDA out-of-memory in proto_gather (prototype
buffer sizing, not a kernel fault); not timed.

Forward, ms (% of FP32 FMA peak, useful FMAs):

| M | coils × seg | order | base static | knot kv0 | knot kv1 (production default) | ps (autotuned) | gm |
|---|---|---|---|---|---|---|---|
| 4 M | 10 × 10 | 0 | 201.7 (9.4) | 196.0 (9.7) | 140.4 (13.6) | 132.1 (14.4) | 531.3 (3.6) |
| 4 M | 48 × 1 | 0 | 218.3 (4.2) | 224.0 (4.1) | 205.7 (4.4) | 58.1 (15.7) | 81.7 (11.2) |
| 4 M | 16 × 1 | 0 | 75.3 (4.0) | | | 21.0 (14.5) | |
| 1.3 M | 10 × 10 | 0 | 68.9 (9.0) | 67.5 (9.2) | 47.1 (13.1) | 63.9 (9.7) | |
| 1.3 M | 48 × 1 | 0 | | | | 25.3 (11.8) | 33.5 (8.9) |
| 1.3 M | 192 × 1 | 0 | 264.8 (4.5) | | | 107.8 (11.0) | |
| 4 M | 10 × 10 | 2 | 235.9 (16.1) | 236.1 (16.1) | 236.0 (16.1) | 599.7 (6.3) | |
| 1.3 M | 10 × 10 | 2 | | | | 282.1 (4.4) | |

Adjoint, ms (%): knot kv1 4 M 10 × 10 o0 131.9 (14.4) vs kv0 370.7 (5.1); 4 M 48 × 1 144.2 (6.3) vs 210.6; 1.3 M
10 × 10 45.4 (13.6) vs 124.3; 4 M 10 × 10 o2 288.4 (13.2) vs 366.9.

Ablation of the static base kernels (each switch removes one part; time left = the rest), ms:

| case | dir | base | no atomics | no MAC loop | no weight staging |
|---|---|---|---|---|---|
| 4 M 10 × 10 o0 | fwd | 201.7 | 189.1 | 69.5 | 151.6 |
| | adj | 342.0 | — | 61.5 | 270.7 |
| 4 M 48 × 1 o0 | fwd | 218.3 | 86.9 | 187.1 | 202.0 |
| | adj | 217.6 | — | 112.7 | 153.1 |
| 1.3 M 10 × 10 o0 | fwd | 68.9 | 65.1 | 21.8 | 53.1 |
| | adj | 114.8 | — | 22.2 | 91.7 |
| 4 M 10 × 10 o2 | fwd | 235.9 | 227.2 | 101.0 | 158.4 |
| | adj | 375.5 | — | 98.7 | 298.1 |

(The adjoint has no forward atomics; its "no atomics" run times the same adjoint kernel.)

Reading:
- Many coils, one segment, forward: the per-tile atomics are ≈ 60 % of the time (218 → 87 ms). ps removes them:
  3.6–3.8× (58 vs 218 ms; 21 vs 75 ms).
- 10 coils × 10 segments: the MAC loop is 66 % (fwd) / 82 % (adj) of the time; atomics only 6 %. ps gains 1.5× over the
  static base but only 6 % over the production knot kv1 forward; at 1.3 M it is slower than kv1.
- Order 2: ps is 2.5× slower than the base (per-point complex weights from the row factors cost more than they save).
- Prediction §6 (ps ≥ 30–40 % at 4 M order 0) missed by 2–3×: best 15.7 %. Cause not measured. Nsight Compute is
  blocked for non-admin users (RmProfilingAdminOnly = 1), so stall reasons need either admin profiling or PROF_
  ablation switches inside pg_fwd_ps (plane load, barrier, MAC, store).

## 11. Why `ps` stops at ≈ 15–18 %: ablations, resources, a shared-load microbenchmark (2026-11-19, third session)

Question: what limits `pg_fwd_ps` (§6 predicted 30–40 % of peak at 4 M order 0, §10 measured ≤ 15.7 %)? Decision: which
cheap tunable fixes it, or whether the inner loop needs a different structure. No profiler (RmProfilingAdminOnly = 1):
limiter found from timing-only ablations, ptxas / SASS counts, the driver occupancy API, a host-side count of the
issued work, and a microbenchmark of the shared loads. Logs: out/diag_*.log, out/fix*.log, out/ldsbench.log
(results.jsonl rows tagged by step); queues run_ps_diag.sh, run_ps_fix.sh. % below = useful FMAs / FP32 peak at the
device's nominal clock as in §4 and §10; the card actually runs at ≈ 2.79 GHz, so true fractions are ≈ 0.92 × these.

### 11.1 What each switch removes (defines in proto_gather.cu, set by `PG_DEFS=A,B`; results are wrong, timing only)

- `PS_NOLOAD`: the global plane loads (the prefetch registers hold a per-thread constant; shared stores and barriers stay).
- `PS_NOSYNC`: the barrier between planes (races; the barrier per channel chunk stays).
- `PS_NOMAC`: the whole multiply-accumulate block (plane loads, stores, barriers, sample staging and output stay).
- `PS_NOSTORE`: the output store (accumulators kept live by a never-true compare).
- `PS_FMAONLY`: the x loop reads its values once per row and reuses them for all 13 x positions: same FFMA count,
  13× fewer shared loads in the hot loop.

### 11.2 Ablation table (fixed best shapes of §10, one session, median of 5; ms)

| case (shape tx,ty,tz,ncg,fc,fsb,nt) | base | no load | no sync | no load + no sync | no MAC | no store | FMA-only | FMA-only + no load + no sync |
|---|---|---|---|---|---|---|---|---|
| 4 M 48 × 1 (4,4,8,4,2,2,256) | 50.1 (18.2 %) | 47.9 | 46.4 | 42.5 | 17.2 | 48.1 | 37.3 | 41.7 |
| 4 M 10 × 10 (4,4,8,2,2,2,128) | 118.8 (16.0 %) | 118.1 | 104.6 | 99.4 | 55.0 | 121.7 | 107.5 | 97.0 |
| 1.3 M 10 × 10 (4,4,8,2,2,1,128) | 57.4 (10.8 %) | 43.5 | 52.0 | 39.7 | 41.3 | 61.4 | 52.8 | 28.8 |

(The base times are 7–14 % below the §10 BEST lines of the same shapes; §10's tuning loop already showed 55.5 / 127.9 ms
for these shapes, the rest is run-to-run spread. All comparisons below are within this session.)

### 11.3 Resources and instruction mix (no profiler needed)

Registers / spills from ptxas (`nvcc -Xptxas -v`, same count as NVRTC's `num_regs`), resident blocks from
cuOccupancyMaxActiveBlocksPerMultiprocessor, SASS from cuobjdump (hot loop = innermost backward branch = one y row):

| case | regs | spills | smem / block | blocks / SM | warps / SM (of 48) | limited by |
|---|---|---|---|---|---|---|
| 4 M 48 × 1 | 127 | 0 | 39.5 KB | 2 | 16 | registers and shared memory |
| 4 M 10 × 10 | 127 | 0 | 38.0 KB (27 KB sample records) | 2 | 8 | shared memory |
| 1.3 M 10 × 10 | 78 | 0 | 24.4 KB | 4 | 16 | shared memory |

Hot loop per y row (FSB 2, FC 2): 140 instructions = 112 FFMA (104 x loop + 8 y/z combine) + 13 LDS.128 + 2 LDS + 13
integer / branch (FFMA 80 % of issue). FSB 1: 79 = 56 FFMA + 13 LDS.128 + 1 LDS + 9. The prefetch of the next plane
costs ≈ 47 registers (127 → 80 with `PS_NOLOAD`).

Work actually issued (host-side count from the sample layout with the kernel's loop bounds, `WORK` line):

| case | useful / issued MAC slots | x (W / HXP) | y rows / warp | z planes / warp | live slots | active (warp, plane) pairs | block fill | FFMA-pipe use of issued work |
|---|---|---|---|---|---|---|---|---|
| 4 M 48 × 1 | 0.538 | 0.769 | 12.41 | 10.79 | 0.94 | 0.45 | 0.67 | 34 % |
| 4 M 10 × 10 | 0.463 | 0.769 | 12.67 | 11.53 | 0.88 | 0.52 | 0.67 | 35 % |
| 1.3 M 10 × 10 | 0.432 | 0.769 | 12.37 | 12.00 | 0.84 | 0.46 | 0.55 | 25 % |

So the useful fraction is 0.43–0.54 (§6 guessed 0.6), and the issued FFMAs run at only 25–35 % of the FFMA pipe.
"Active pairs": of all (warp, plane) slots of the launched blocks only ≈ half do MACs — samples are z-sorted, so the
warps of one block have staggered z windows (each active ≈ 11 of the 17 planes), and the last item of a tile is
partly empty (block fill 0.55–0.67). Idle warps hold registers and shared memory, so the active warps per scheduler
are ≈ 1–2.

### 11.4 Shared-load microbenchmark (`--mode ldsbench`, kernel `pg_lds_bench`)

A warp issues LDS.128 whose 32 lanes read `ncg` distinct consecutive float4 (as ps's channel groups do), with KF
independent FFMA per load; 8–48 warps per SM. Cost per warp-level LDS.128, in SM clocks at 2.79 GHz:

| distinct float4 per warp | 1 | 2 | 4 | 8 | 32 |
|---|---|---|---|---|---|
| clk per LDS.128 (no FFMA, any warps 8–48) | 2.4 | 2.4 | 4.1 | 4.1 | 4.1 |
| FFMA rate with 8 FFMA per load (16–24 warps) | 64 % | 64 % | 50 % | 34 % | 34 % |
| FFMA rate with 16 FFMA per load | 84 % | 84 % | 58 % | 57 % | 57 % |

A broadcast does not make a float4 load cheap: once a warp reads 4 distinct float4 (only 64 B of distinct data) the
instruction costs the full 4 clocks of a 512 B delivery, and even a single shared address costs 2.4. That is the
assumption of §6 that failed ("one wavefront serves the warp", FMA-bound for FSB ≥ 2).

### 11.5 The limiter

- **4 M, many coils (48 × 1, NCG 4): shared-memory delivery.** The x loop issues 1 LDS.128 per 8 FFMA (FSB 2). At
  4.1 clk each, the 2.8 G warp-level LDS.128 of the pass occupy the shared pipe for ≈ 32 ms of the 50 ms; the
  microbenchmark caps this mix at 50 % of the FFMA rate, i.e. 0.50 × 0.538 ≈ 27 % useful (≈ 29 % in table units).
  Measured 18.2 %: the rest is the staging / barrier part that does not overlap (no-MAC alone 17 ms) and the half-idle
  warps. Consistent with the ablations: removing 12 of 13 shared loads (FMA-only) is the largest single gain (−25 %).
- **4 M, 10 coils × 10 segments (NCG 2, 2.4 clk loads): too few warps.** 8 warps per SM, because the 128-sample
  records (20 coefficient words each) take 27 KB of shared memory; with half of them idle per plane the schedulers
  have ≈ 1 warp each and the issue slots are < 50 % busy. Adding resident warps without changing the work speeds it
  up (below).
- **1.3 M: staging.** Items are half empty (fill 0.55), so plane loads per useful MAC are ≈ 2× those at 4 M: the no-MAC
  kernel alone takes 41 of 57 ms, removing the global plane loads saves 24 %.
- Outside 1.3 M the plane loads, the barriers and the store are second-order (≤ 12 % each).

### 11.6 Changes (new tunables; nothing removed) and new numbers

- `PS_ACFG` (1 = segment coefficients read from global at the output instead of staged in the sample record): record
  2·NSEG words shorter. 4 M 10 × 10, same shape: 2 → 3 blocks/SM (8 → 12 warps), **118.8 → 104.3 ms (16.0 → 18.3 %)**.
  Autotuned on for nseg ≥ 4 (full tuning tries both).
- FSB 3 (FC ≤ 4) added to the autotune: 1.5× FFMA per shared load at the same 128 registers; −4 % at 4 M 48 × 1 in the
  tuning loop (49.4 vs 51.6 ms), at the cost of emptier items (fill 0.57).
- `PS_PFS` (next plane copied one float4 at a time inside the row loop, load at one row and store at the next): 127 →
  94–98 registers, but +24 instructions per row and no extra block (shared memory still limits): 50.1 → 55.1 ms
  (48 × 1), 104.3 → 123.2 ms (10 × 10 with ACFG). Full tuning only.
- `PS_MINB` (`__launch_bounds__` minimum blocks): forces 80 registers with 16–88 B spills, slower (77 ms); `--fixed`
  only.
- Flat home tiles (tz 1–2: 4×8×2, 4×16×2, 4×16×1, 4×8×1): every warp is active in nearly every plane (active pairs
  0.63–0.76), but items get emptier (fill 0.23–0.45) and bigger planes cost a block per SM: 63–111 ms at 48 × 1,
  176–270 ms at 10 × 10. Full tuning only.
- Fixed the 16 × 12 OOM: the per-sample arrays (sorted positions, ΔC, coefficients, permutation; 0.56 GB at 4 M × 12
  segments) were copied once per (tile shape, item size, FSB) during the autotune; now shared per tile shape. 4 M
  16 × 12 ps now runs: 213.9 ms (17.1 %) with the §10 candidate set, vs base static 336.5 ms.

Autotuned with the extended candidate set (`fixtune_*`; same session):

| case | ps §10 | ps now | shape now | base static (§10) |
|---|---|---|---|---|
| 4 M 48 × 1 | 58.1 (15.7) | 51.1 (17.9) | 4,4,8 ncg 4 fc 2 fsb 3 nt 256 | 218.3 |
| 4 M 10 × 10 | 132.1 (14.4) | 108.6 (17.5) | 4,4,8 ncg 2 fc 2 fsb 2 nt 128 acfg | 201.7 (kv1 140.4) |
| 4 M 16 × 1 | 21.0 (14.5) | 16.3 (18.6) | 4,4,8 ncg 4 fc 2 fsb 3 nt 256 | 75.3 |
| 1.3 M 10 × 10 | 63.9 (9.7) | 58.7 (10.5) | 4,8,8 ncg 2 fc 2 fsb 1 nt 256 acfg | 68.9 (kv1 47.1) |
| 4 M 16 × 12 | OOM (213.9 (17.1) with the §10 set) | 194.3 (18.8) | 4,4,8 ncg 8 fc 2 fsb 3 nt 256 acfg | 336.5 |
| 1.3 M 48 × 1 | 25.3 (11.8) | 22.7 (13.1) | 4,8,8 ncg 8 fc 2 fsb 3 nt 256 | (timed out) |

All rel. errors 6.4–6.5e-7. Most of the "§10 → now" gain at 48 × 1 and 16 × 1 is run-to-run spread (the §10 shapes time 50.1 / ≈ 18 ms in this
session); the real changes are ACFG at many segments (−12 % at 10 × 10, −9 % at 16 × 12 against the §10 set in this session) and
FSB 3 (≈ −4 %).

### 11.7 What it means and what next

The ps inner loop is bounded by shared-memory delivery, not FMAs: any lane-per-(sample, channel) loop that reads each
grid value per sample from shared memory as a float4 pays ≥ 2.4–4 clk per 8–12 FFMA. The cheap tunables are
exhausted (ACFG / FSB 3 / NCG trade the bound against item fill, union waste and occupancy). Next, in order:
1. Value-stationary inner loop: lanes hold grid values in registers (a halo row × FC channels) and the samples' weights
   are broadcast (one address per warp: the 2.4 clk case, or LDS.32), each loaded weight feeding FC MACs; this
   inverts the FFMA : shared-byte ratio (≈ 1 → 2–4 FFMA per byte) and is the structural fix to measure next.
2. Keep all warps of a block busy: split each item's samples into warps with equal z windows (or stream planes per
   warp group), and pack tiles so that items are full (fill 0.55–0.67 today).
3. 1.3 M: fewer plane loads per useful MAC (larger items, or the knot / multi-frame batching that the engine already
   does).

### 11.x Nsight Compute cross-check (sudo ncu, 2026-11-19 13:16–13:30, out/ncu/*.ncu-rep)

ncu (`--set full`, clocks locked at base 2.23 GHz) on the §10 best ps shapes agrees with §11.5:

| case | time (ms, at 2.23 GHz) | regs | warps active (% of 48) | L1/shared throughput | FMA pipe | shared wavefronts per LDS instr. | stall samples: barrier / short scoreboard (shared latency) / issuing |
|---|---|---|---|---|---|---|---|
| 4 M 48 × 1 | 60.8 | 127 | 33 % | 76.6 % | 38.6 % | 3.6 | 41 % / 19 % / 12 % |
| 4 M 10 × 10 | 144.9 | 127 | 17 % | 46.4 % | 38.3 % | 1.9 | 37 % / 17 % / 26 % |
| 1.3 M 10 × 10 | 67.2 | 78 | 33 % | 68.8 % | 31.4 % | 2.0 | 42 % / 29 % / 12 % |

- 48 × 1: the shared pipe is the limiter (3.6 wavefronts per LDS.128 although the lanes read only 4 distinct float4,
  as the §11.4 microbenchmark found); barrier waits are the largest stall class in all three.
- ncu also flags the output stores: 4.8 of 32 B per sector used (one scattered (sample, coil) store); small in absolute
  time (no-store ablation −4 %).
- The second ps launch of 4 M 10 × 10 and both knot-kernel profiles (seg_forward_k) ended with `LaunchFailed` after
  2 replay passes: not yet known whether this is ncu's replay (memory save/restore with a 6 GiB grid) or a real
  out-of-bounds access; to check with compute-sanitizer. NVRTC builds lack -lineinfo, so no source-line attribution.

## 12. Design 6 `vs`: value-stationary rows, lanes own channels (2026-11-19, fourth session)

Question: can a layout in which the grid values are the stationary register operand (lanes own channels, sample
weights broadcast) break the shared-delivery bound of `ps` (§11.5: ≤ 50 % of the FFMA rate, × 0.43–0.54 useful) and
reach ≥ 30 % useful FP32 peak at 4 M order 0? Decision: whether this replaces ps as the forward candidate for the
engine, and whether it has an order-2 path. Written before the kernel; §12.5 onward are the measurements.

### 12.1 Constraints that fixed the geometry (measured / counted first)

- **Footprint.** `--mode stats` now also prints the vs item model (items = runs of ≤ SMAX z-sorted samples of one home
  tile; an item loads only its planes and rows: (TX + 9)(Δy + 10)(Δz + 10) points). L2 staging, B per useful FMA (o0,
  per channel), and samples active per (item, plane):

  | footprint TX × TY (column) | 4 M, SMAX 64 | 4 M, SMAX 128 | 1.3 M, SMAX 64 | 1.3 M, SMAX 128 | active / plane (4 M, 128) |
  |---|---|---|---|---|---|
  | 1 × 1 | 0.665 | 0.638 | 1.84 | 1.83 | 10.7 |
  | 2 × 2 | 0.261 | 0.225 | 0.636 | 0.603 | 31.1 |
  | 2 × 4 | 0.199 | 0.156 | 0.425 | 0.384 | 49.3 |
  | 4 × 4 | 0.171 | 0.120 | 0.308 | 0.257 | 70.6 |
  | (ps 4 × 4 × 8, items 128, §6) | | 0.134–0.150 | | 0.43 | |

  The L2 budget (§6) is ≈ 0.12–0.16 B / FMA at full FMA rate, so at a 40 % target ≈ 0.3–0.4. A z-column tile (1 × 1,
  every sample of a block on the same x / y rows, the simplest value-stationary layout) is L2-bound at ≈ 20–25 % of
  peak at 4 M. The footprint must be ≈ 2 × 8 / 4 × 4 → samples of a block have TX·TY different (x, y) offsets
  ("classes").
- **Shared-load instruction cost** (new `ldsbench` rows, `LDSB_W=1|2`, out/ldsbench_w12.log): every warp-wide shared
  load costs ≥ 2.0–2.2 SM clocks whatever its width or address pattern: LDS.32 2.2 clk (1–32 distinct words), LDS.64
  2.0 clk broadcast / 3.2–3.9 clk with ≥ 16 distinct float2 in an FFMA mix, LDS.128 2.4 / 4.1 (§11.4). With 8 FFMA per
  LDS.32: 80 % FFMA rate. So what counts is shared-load *instructions* per FFMA, not bytes: the design minimises the
  number of LDS / STS / SHFL per sample, and packs broadcasts into LDS.128.

### 12.2 The design

- Home tile TX × TY × TZ; items = ≤ SMAX samples of one tile, cut in z order; inside an item the samples are
  re-sorted by class (oy, ox) and then oz, so that for every (plane, class) the active samples are one contiguous range
  (a per-item table of NPL × TX·TY ranges, built in shared memory by one thread per class with a two-pointer sweep).
- Warp = 16 channel lanes × 2 row halves (lane = h · 16 + c). Lane (h, c) holds, for its channel, the values of 5
  consecutive grid rows × HX = TX + 9 points of the current plane in registers (2·5·HX floats; 110 at TX = 2): half 0
  the rows oy..oy + 4 of a class, half 1 the rows oy + 5..oy + 9, so the two halves together hold exactly the 10 support
  rows of class oy. Classes are visited in window order st = oy = 0..TY − 1: the window slides by one row per step per
  half (static register-slot rotation row r → slot r mod 5 in the unrolled st loop), every (plane, half) loads TY + 4
  rows.
- Inner loop over the samples of class (oy, ox) active in the plane, one sample per warp at a time (all lanes the same
  sample: the x shift ox is a compile-time variant, no zero weights in x, y or z). Per sample ("body"), per lane:
  x contraction of its 5 rows against the 10 x weights (5 × 20 FFMA), y combination (10 FFMA), z scale (2 FMUL); the
  two halves' partials are combined with one SHFL (half 0 sends its imaginary, half 1 its real part) and the lane
  read-modify-writes one float of the sample's accumulator (half 0 the real, half 1 the imaginary plane of
  acc[s][c]): one LDS.32 + one STS.32, 32 distinct words = one wavefront.
- Weights per body, all broadcast: the x row (φx 0..9, oz, φy of row 4 of the half; duplicated per half) = 3 LDS.128;
  φy rows 0..3 of the half = 1 LDS.128; φz(tz) = 1 LDS.32. Total per body: **5 weight loads + 1 SHFL + 1 LDS + 1 STS = 8
  shared-pipe instructions against 112 FFMA**.
- Grid values are read straight from global/L2 into registers (no shared staging, no barriers in the main loop: each
  warp walks planes and classes on its own); per (plane, half) TY + 4 rows × HX points (coalesced: 16 lanes × 8 B
  consecutive channels).
- Accumulators in shared memory, per channel group: SMAX × 16 × 8 B. Warps: NWC channel groups (each 16 channels) × NWS
  splits of the window positions st (warps with different st own different samples, so they share the accumulator rows
  without races). One barrier at the end of the item; epilogue Σ_l a_jl acc → y, plain store when the block holds
  every segment of the coil, else one atomic per (sample, coil piece). Channels padded to multiples of 16 (10 × 10:
  100 of 112 lanes, 0.89).
- Order 2: the same body with x-moments. Per (sample, plane) the lane forms E_ab = e_ab wz_{4−a−b}(tz) (21 FMUL) and per
  row R_a = Σ_b E_ab wy_b, Q_a (21 FFMA); per derivative a = 0..4: moments m_a(row) = Σ_t kx_a(t) v (20 FFMA per row),
  r += (R_a + i Q_a) m_a.
- Tunables (NVRTC defines, autotuned; nothing arch-specific): TX, TY, TZ, SMAX, NWC, NWS, VXPAD (1 = zero-padded x
  weights over HX instead of the ox variants: less code, x waste 10/HX).

### 12.3 Count per useful MAC (per lane; one body = 5 rows × 10 x = 50 useful complex MACs of one channel)

| | o0 vs body | o0 ps (§11.3, FSB 2 FC 2, per x-point) | o2 vs body | o2 production tile |
|---|---|---|---|---|
| FFMA issued | 112 (100 useful) | 8 per 8 issued slots, 0.43–0.54 useful | 706 (200 useful) | 4 per MAC + weight build, 2.2× zero-weight MACs |
| shared-pipe instr. | 8 (5 broadcast LDS.128/32, 1 SHFL, LDS.32 + STS.32) | 1 LDS.128 (4 distinct) | 33 (15 kx + 7 wy + 5 wz + 6 e/d bcast; SHFL, RMW) | |
| shared clk (§11.4 / §12.1) | ≈ 17.6 | 4.1 | ≈ 73 | |
| FFMA clk (4 / clk / SM) | 28 | 2 | 177 | |
| shared / FFMA time | 0.63 | 2.0 | 0.41 | |
| other issue (ALU, branch, SHFL add, FMUL) | ≈ 11 | | ≈ 40 | |
| useful FFMA / issued instructions | 0.76 | ≈ 0.4 | 0.26 | |
| registers (TX 2) | 110 values + ≈ 45 → ≈ 155 | 127 | 110 + 25 wy + 15 E + ≈ 40 → ≈ 190 | |
| shared bytes / block | SMAX · 176 (records) + NPL·TX·TY·4 + NWC · SMAX · 128 | | SMAX · 760 + … | |

Per (plane, half, warp) the window loads (TY + 4)·HX LDG.64 and reads TX·TY range words: ≈ 2–4 % of the bodies' issue
at 70 active samples per plane (4 M), ≈ 6 % at 1.3 M.

### 12.4 Prediction (before measuring)

- o0, 4 M: the body is issue-bound (0.76 useful / issued) with the shared pipe ≈ 63 % busy; by the §11.4 / §12.1 rows
  (14 FFMA per shared instruction lies between "8 FFMA per LDS: 80–88 %" and "16: 84 %") the FFMA pipe should stay
  ≈ 85–90 % fed. Occupancy is set by the accumulator rows (SMAX 64: 8 KB per channel group) to ≈ 8–12 warps / SM; no
  barrier in the main loop, but each window step starts with a dependent L2 load. Guess for the latency/occupancy and
  tail loss ×0.6–0.7 (ps measured ×0.6 of its bound). **Predicted useful fraction: 0.76 × 0.88 × 0.65 ≈ 43 % (range
  35–50 %) at 4 M 48 × 1 / 16 × 1 / 16 × 12; × 0.89 channel padding → ≈ 38 % at 4 M 10 × 10; 1.3 M 10 × 10: staging
  0.26–0.31 B / FMA, fewer active samples per plane → ≈ 25 %.** In ms: 4 M 48 × 1 ≈ 21, 4 M 10 × 10 ≈ 50, 4 M 16 × 1
  ≈ 7, 1.3 M 10 × 10 ≈ 25, 4 M 16 × 12 ≈ 85.
- o2, 4 M 10 × 10: 0.26 useful / issued × 0.88 × 0.65 × 0.89 ≈ 13 % → ≈ 290 ms, i.e. **no win over the production
  tile kernel (236 ms)**. The reason is structural: with lanes owning channels, the complex weight (rank 5 in x, rank
  3 imaginary) is rebuilt by every channel lane: 5 x-moments × 10 points × 2 = 100 FFMA per row per channel, against
  40 useful. Order 2 wins only if the weight is shared across channels (§12.9, item 3).

### 12.5 As built (proto_gather.cu `pg_fwd_vs`, `#ifdef VS`; proto_gather.rs `--designs vs`, `sort_vs`, `vs_work`)

- Correctness: `--check` against the f64 reference, rel err 6.2–6.8e-7 at order 0 and 6.6e-7 at order 2 (smoke: own and
  atomic epilogue, NWS 1–4, VXPAD 0 / 1, partial channel groups; every tuned BEST below).
- ptxas (nvcc sm_89, no spills): TX 2 o0 158 registers, TX 4 o0 226, o2 254. Two compiler fixes were needed: the
  (row, x) address offsets of the whole window do not depend on the plane and were hoisted out of the plane loop
  (255 registers, 230–600 B spills; fixed by an opaque per-step zero from `asm volatile`), and `__launch_bounds__(NT)`
  with NT > 384 caps registers below the need (candidate filter).
- SASS of one sample body (TX 2, TY 8, o0): 133 instructions = 111 FFMA + 6 LDS + 1 STS + 1 SHFL + ≈ 14 integer /
  branch: 0.75 useful FFMA per issued instruction (counted 0.76).
- Tunables picked by the autotune: tz 16 everywhere, SMAX 32–64, NWS 1, one block = all 16-channel groups of a pass
  (NWC 3 / 7 / 4), TX 4 × TY 4 except 48 × 1 (TX 2 × TY 4).
- New `ldsbench` rows (`LDSB_W=1|2`), `--fixed tx,ty,tz,smax,nwc,nws[,vxpad]`, `HASTY_NVRTC_LINEINFO=1` (hasty-cuda
  `compile_cubin`: adds `-lineinfo`, part of the cache key; applies to proto_gather and to the engine; default off).
- Queue: research/experiments/sample_kernels/run_vs.sh (unit `proto-vs`, finished 15:04); logs out/{smoke_vs_*,vs_*,
  psref_*,knotref_*,san_*}.log, rows in out/results.jsonl tagged by step; ncu reports out/ncu/vs_m4M_c{48,16}.ncu-rep.

### 12.6 Predicted vs measured (RTX 4090; % = useful FMAs / FP32 peak at the nominal clock, as §4; same session)

| case | predicted vs | **vs measured** | shape (tx,ty,tz,smax,nwc,nws) | ps (§11.6 shapes, same session) | production knot kv1 | base static (§10) |
|---|---|---|---|---|---|---|
| 4 M 48 × 1 o0 | 21 ms (43 %) | **34.3 ms (26.7 %)** | 2,4,16,32,3,1 | 48.8 (18.7) → 1.42× | 205.7 (§10) | 218.3 |
| 4 M 10 × 10 o0 | 50 ms (38 %) | **74.7 ms (25.5 %)** | 4,4,16,64,7,1 | 105.3 (18.1) → 1.41× | **140.5 (13.6) → 1.88×** | 201.7 |
| 4 M 16 × 1 o0 | 7 ms (43 %) | **12.3 ms (24.8 %)** | 4,4,16,32,1,1 | 17.5 (17.4) → 1.42× | | 75.3 |
| 1.3 M 10 × 10 o0 | 25 ms (25 %) | **30.7 ms (20.2 %)** | 4,4,16,64,7,1 | 58.7 (10.5) → 1.92× | 47.1 (§10) → 1.54× | 68.9 |
| 4 M 16 × 12 o0 | 85 ms (43 %) | **120.4 ms (30.4 %)** | 4,4,16,64,4,1 | 195.2 (18.7) → 1.62× | | 336.5 |
| 4 M 10 × 10 o2 | 290 ms (13 %) | **281.9 ms (13.5 %)** | 2,4,8,48,7,1 | 599.7 (6.3, §10) → 2.1× | **236.6 (16.1) → 0.84×** | 235.9 |
| 1.3 M 10 × 10 o2 | | 105.5 ms (11.7 %) | 2,4,8,48,7,1 | 282.1 (4.4, §10) | | |

- Order 0: 1.4–1.9× faster than ps in every case, 1.9× faster than the production knot forward at 4 M 10 × 10 (1.5× at
  1.3 M). The 30 % target is reached only at 16 × 12 (30.4 %); 25–27 % at the other 4 M cases. The prediction missed
  by 1.4–1.7× (the per-body count was right, §12.5; the ×0.65 latency/occupancy factor was too optimistic, §12.7).
- Order 2: as predicted, no win: 282 ms against 237 ms for the production tile kernel (2.1× faster than ps o2). The
  x-moment body is FFMA-bound by construction (706 FFMA per 200 useful).

### 12.7 Limiter (Nsight Compute, now available to non-admin; targeted sections, `--launch-count 1`, line info)

| case (shape) | time under ncu | warps active / SM | issue slots busy | FMA pipe | L2 throughput | L1 hit | top stalls per issue |
|---|---|---|---|---|---|---|---|
| 4 M 48 × 1 (2,4,16,32,3,1) | 37.2 ms | 11.6 (theoretical 12: registers 168 → 4 blocks) | 58 % | 40 % | **80 %** | 5 % | wait 0.58, dispatch 0.52, barrier 0.51, short scoreboard 0.50, long scoreboard 0.46 |
| 4 M 16 × 1 (4,4,16,32,1,1) | 14.0 ms | 7.6 (registers 171–226, 1-warp blocks, smem) | 49 % | 32 % | 68 % | 6 % | wait 1.03, long scoreboard 0.49, short scoreboard 0.40 |

- The shared-memory pipe is no longer the limiter (mio_throttle 0.03 per issue; LSU pipe 33 %): the structural fix
  worked; per useful MAC the design issues 0.08 shared instructions against ps's 0.25–0.5.
- What limits now is **L2 delivery together with low occupancy**. The window rows come straight from L2 (L1 hit
  5–6 %), at 68–80 % of L2 throughput; and 7–12 resident warps (≈ 2–3 per scheduler: registers for the 5 × HX window
  plus the shared accumulator rows, SMAX × 128 B per channel group) are too few to hide the L2 latency at each window
  step (5.7–11 bodies per step, `WORK` line) and the FFMA chain latency ("wait" stalls: each row's 10-FFMA x
  contraction is a dependent chain, ILP 10 per body). At 48 × 1 the small items (SMAX 32) add the per-item serial
  parts (stage, one thread per class building the range table, epilogue): barrier 0.51 per issue.
- The tuner trades the two: larger items (fewer L2 bytes per FMA, §12.1) cost shared memory and hence warps.

### 12.8 compute-sanitizer (memcheck, 200 k samples, `HASTY_NVRTC_LINEINFO=1`)

**0 errors** for: the production knot kernels with `HASTY_SEG_KV=1` (`--mode base --knot`, 10 × 10, order 0 and order
2: every forward and adjoint configuration the engine autotune runs), ps (10 × 10, §11.6 shape) and vs (order 0 and
2). out/san_*.log. So the `LaunchFailed` of the earlier ncu replay (§11.x) was the replay of the 6 GiB grid, not an
out-of-bounds access.

### 12.9 Next

1. L2 bytes. The two halves load rows 5 .. TY + 3 twice per plane (TY 4: 16 rows instead of 13), and items are cut
   small for shared-memory reasons. Options: give half 1 the rows of half 0 through the L1 (larger L1 carve-out
   while the accumulator is the main shared consumer), or a block-wide plane in shared memory loaded once with
   coalesced float4 for all channel groups (one barrier per plane again, but ≈ 1/3 fewer L2 bytes).
2. Occupancy and latency: prefetch the next window row into spare registers (TX 2: +22) so a step's first body does not
   wait on L2; process two samples per iteration (independent FFMA chains against "wait"); TX 2 at 158 registers
   with NWS > 1 for 12–16 warps; build the per-item range table in parallel (one thread per (plane, class), binary
   search) to cut the barrier time at small items.
3. Order 2 needs a layout where the complex weight is shared by the channels: per (sample, plane) the warp builds the
   100 complex weights cooperatively (≈ 3 points per lane: 8 FFMA each plus the row factors) in a warp-private
   800 B table, then each lane reads them as broadcast LDS.128 (2 complex weights per load, 8 FFMA per load → ≈ 80 %
   FFMA by §12.1). Count: ≈ 70 FFMA + 35 shared instr. for the build, 25 LDS.128 + 200 FFMA for the MACs per half-body:
   shared ≈ 115 clk against 50 clk of useful FFMA → ≈ 40 % bound, ≈ 25 % realistic → ≈ 150 ms at 4 M 10 × 10 (vs
   237 production). To measure before any engine work; until then the production kernel stays the order-2 forward.
4. Engine: vs is the first exact-support forward that beats the production knot forward at order 0 (1.9× at 4 M,
   1.5× at 1.3 M, 10 × 10). Integration path as §9 (home-tile CSR in SegPoints, a new forward variant, autotuned
   tunables), order 0 / 1 only.
5. Portability: nothing is 4090-specific (NVRTC defines; the barrier-free design depends on L2 bandwidth and shared
   memory per SM). H100 / A100 have 2–4× the L2 bytes per FMA and 164–228 KB shared memory per SM → larger items and
   more warps; expected to sit nearer the issue bound (0.75). To be measured on the cluster nodes.

## 13. vs order 2 / 3 with a shared weight table, and the order-0 L2 / occupancy tunables (2026-11-19, fifth session)

Questions. (A) Can the vs layout beat the production knot forward at order 2 (236.6 ms, 4 M 10 × 10) and at order 3
(the velocity reconstruction runs CORR 3) once the complex weight is built once per (sample, plane) for all channel
lanes instead of by every lane (§12.4: ≈ 3.5 FFMA per useful FFMA)? (B) Which of the §12.9 L2 / occupancy fixes speed
up order 0? Decision: whether vs gets an order-2/3 path in the engine, and which tunables enter the autotune set.
§13.1–13.2 were written before the kernels.

### 13.1 Design `vt` (VT 1): per (sample, plane) complex-weight table, built by the warp, read by broadcast

- Everything of §12.2 stays (home tile, items, class ranges, lanes = 16 channels × 2 row halves, 5 × HXV value window
  in registers, barrier-free main loop, shared accumulators, epilogue). Only the sample body changes.
- Weight algebra (exactly the terms of the engine's `stage_weights` / `stage_weights_hk`, segnufft_knot.cu): for row
  (ty, tz) the weight at x point t is w = Σ_a kx_a(t) (R_a + i Q_a), a = 0..4, with
  R_a = Σ_b ER_ab(tz) wy_b(ty), Q_a = Σ_b EQ_ab(tz) wy_b(ty) (a + b ≤ 4) and per-plane coefficients
  ER_ab = e_ab wz_{4−a−b} (+ wz_0 at a = b = 0) [+ hc3_ab wz_{3−a−b} at CORR 3],
  EQ: the six CORR-2 ΔC terms (d_yy wz_0 at (0, 2), d_zz wz_2 at (0, 0), d_yz wz_1 at (0, 1), d_xy wz_0 at (1, 1),
  d_xz wz_1 at (1, 0), d_xx wz_0 at (2, 0)) [− hc4_ab wz_{4−a−b} at CORR 3]. hc = Σ_q u_q H'_q (25 per sample, staged
  from uw / htab with the engine's fma chain, `stage_hc_k`).
- Build, per (sample, plane), lanes 0..29 = (row i = 0..9, x group g = 0..2, points 4 g .. 4 g + 3, group 2 has 2):
  every builder lane forms ER / EQ (o2 22, o3 47 FMUL/FFMA, broadcast inputs), its row's R_a, Q_a (o2 21, o3 30
  FFMA) and its ≤ 4 points (o2 8, o3 10 FFMA each) and stores them with 2 STS.128 into a warp-private table of
  10 × 10 complex = 800 B (double-buffered, one `__syncwarp` per sample). wz(tz) is folded into the table.
- MAC: lane (h, c) reads the 50 weights of its half as 25 LDS.128 (2 distinct addresses per instruction: the two
  halves) and does 4 FFMA per point = 200 FFMA in 10 independent chains; halves combined by one SHFL, one
  read-modify-write of the accumulator (as §12.2, no z factor).
- Record (floats): kx[5][12] (t padded to 12), wy[10][8] = wy_0..4 + wz_0..2 of the same index, wz_3..4 [10][2],
  e[15] + d[6] (+ 3 pad), at CORR 3 hc[25] (+ 3 pad), ox / oy / oz: 188 (o2) / 216 (o3) floats.
- New tunable `VT` (vs, CORR ≥ 2, VXPAD 0; candidates add vt 0 / 1); order 3 runs vt only.

Count per body (one sample × one plane × 16 channels × 10 rows; per lane; 50 useful complex MACs = 200 useful FFMA):

| | vs o2 (§12, per-lane x-moments) | **vt o2** | **vt o3** |
|---|---|---|---|
| FFMA / FMUL issued | 706 | 200 MAC + 75 build + ≈ 12 sums = 287 | 200 + 117 + 12 = 329 |
| shared-pipe instr. | 33 | 25 LDS.128 MAC + 17 build (6 e/d, 2 wz, 2 wy, 5 kx, 2 STS.128) + 3 acc/SHFL + 1 oz = 46 | 46 + 7 hc = 53 |
| shared clk (2.4 broadcast LDS.128, 4.1 for ≥ 3 distinct, 2.2 LDS.32 / SHFL) | ≈ 73 | 60 + 41 + 7 + 2 ≈ 110–122 | ≈ 127–139 |
| issue clk (instr. / 4 per SM clk) | ≈ 190 | (287 + 46 + ≈ 10) / 4 ≈ 86 | ≈ 98 |
| useful FFMA clk | 50 | 50 | 50 |
| bound (useful FFMA / max(shared, issue)) | 26 % | **≈ 41 % (shared pipe)** | **≈ 36 % (shared pipe)** |
| registers (TX 2) | 254 | 110 window + ≈ 20 MAC + ≈ 60 build ≈ 190 | ≈ 205 |
| shared bytes / block | SMAX · 756 + … | SMAX · 752 + NWV · 1600 + NWC · SMAX · 128 | SMAX · 864 + … |
| shared bytes delivered per useful MAC (per lane) | | 8 B weight + ≈ 2 B build | ≈ 8 + 2.5 B |

The table halves the issued work per useful FFMA (3.5 → 1.4 at o2, 1.6 at o3) and moves the bound from the FFMA pipe
to the shared pipe: the 25 MAC loads alone are 60 clk per 50 clk of useful FFMA (a lane owns one channel, so one
LDS.128 of 2 complex weights feeds 8 FFMA; §11.4: 64 % FFMA rate at that ratio).

### 13.2 Prediction (before measuring)

- Achieved / bound: vs o0 reached 0.38 of its bound (25.5 % of 0.76 × 0.89); here the L2 bytes per issued instruction
  are 2.6× smaller (the same window feeds 343 instead of 133 instructions per body), so ≈ 0.5–0.6.
- **o2, 4 M 10 × 10: 41 % × 0.89 (channel padding) × 0.55 ≈ 20 % → ≈ 190 ms (range 150–240), vs production
  236.6 ms: ≈ 1.25× faster.** 1.3 M 10 × 10 o2 ≈ 70 ms (55–90; vs §12 105.5 ms).
- **o3, 4 M 10 × 10: 36 % × 0.89 × 0.55 ≈ 17.5 % → ≈ 220 ms (170–270)**; production order 3 to be measured
  (`--mode base --knot --order 3`, `points_hi`).
- If measured ≥ 240 ms at o2, the next variant is a block-shared table (one build per (sample, plane) for all NWC
  channel-group warps, one barrier per plane): shared ≈ 60 + 7 + 53 / NWC clk → bound ≈ 65 %.

### 13.3 Order-0 tunables (Task B), counted before measuring

ncu (§12.7, 4 M 48 × 1): L2 throughput 80 % with L2 hit 97.7 % and DRAM 13 %: the window loads are L2-bandwidth bound,
not DRAM; L1 hit 5.5 %; 2.9 active / 1.1 eligible warps per scheduler.

- `VSHR` (1 = half 1 hands the row it evicts to half 0 by SHFL): at window step st the row half 0 needs (st + 4) is
  exactly the row half 1 drops, so half 0 never loads after the first step; half 1's loads are predicated (PTX
  `@p ld.global.nc`). L2 rows per (plane, warp) 2 (TY + 4) → TY + 9: TY 4 16 → 13 (−19 %), TY 8 24 → 17 (−29 %); cost
  2 HXV SHFL per step (≈ 22–26 against ≈ 46–90 shared instr. of the step's bodies). Expected −10–15 % where L2-bound
  (48 × 1, 16 × 1), less at 10 × 10 / 16 × 12 (7–4 warps per block share nothing anyway). A block-wide plane in
  shared memory loads the same unique rows (warps of different channel groups load disjoint channels), so after VSHR it
  saves no L2 bytes; not built.
- `carve` (driver attribute CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 0 / 50 % vs default): more L1 for
  re-hits across window splits and neighbouring items, at the risk of fewer resident blocks. Expected ±5 %.
- `VPF` (1 = the next step's row is loaded into 2 HXV spare registers one step ahead): removes the dependent L2 wait at
  the start of each step (5.7–11 bodies per step); +22–26 registers (TX 2 only below 255). Expected −5–10 % at TX 2.
- `VS2` (1 = two samples per iteration, o0): two independent bodies interleaved (ILP 10 → 20 chains); +≈ 20 registers.
  Expected −0–8 % (the "wait" stall class).
- `VPT` (1 = range table built in parallel: one thread per (plane, class), four binary searches over the item's sorted
  keys, instead of NCLS threads with serial two-pointer sweeps of ≈ 100–150 dependent loads): removes ≈ 3 k clk of
  single-thread latency per item behind a barrier. Expected −2–5 % at small items (48 × 1, 16 × 1; SMAX 32).
- Smaller window: TX 1 tiles (HXV 10: 100 window registers, ≈ 145 total → up to 14 warps / SM) with TY 8–16; with VSHR
  their L2 rows per class (TY + 9) · 10 / TY = 15.6 point-rows at TY 16 match TX 2 × TY 4 (17.9). Expected to win
  where occupancy limits (10 × 10, 16 × 12 at 7–8 warps / SM).

## 14. Engine integration of vs (2026-11-19, sixth session)

Question: does the §12 vs forward keep its kernel-level gain (1.5–1.9× the production knot forward at order 0) as an
engine variant, operator-level (pad + FFT + gather) and in the gradient (forward + adjoint)? Decision: whether the
autotune should pick vs by default at orders 0 / 1, and what to build next.

### 14.1 What was added (nothing removed; kv 0 / 1 / 2 unchanged and selectable)

- Kernel `seg_forward_vs` in rustlib/crates/hasty-nufft/src/segnufft_vs.cu (standalone NVRTC module): the §12.2 body at
  CORR 0 (as the prototype) and a new CORR 1 body (three x-moments φx, φx', φx'' per row; w·v = R0 m0 + i Σ_a Q_a m_a
  with exactly the R0 / Q_a terms of `stage_weights`), static images and knot-major engines (1-D and 2-D knots, β_b(t)
  as `knot_beta`), ES and KB windows, W ≤ 10 (rows ≥ W zero). CORR ≥ 2: `#error` hook (§12.9 item 3 / §13).
- Staging per item: per sample index j from the vs list, records from the engine's `rec` (original order), a_l(τ) table
  position, cos / sin φ0 and β in the shared record; epilogue y[c0 + coil][j] += Σ_l β e^{−iφ0} a_l(τ) acc — a plain
  read-modify-write, one owner per (sample, coil): the host only launches channel blocks holding whole coils (one block
  row, or 16 · NWC a multiple of NSEG). Same sink semantics as the tile kernels (y initialised by the host: forward,
  residual, sink / chunked, group).
- Host (rustlib/crates/hasty-nufft/src/segnufft/vs.rs): `VsGeom {tx, ty, tz, smax}` (table geometry), `VsCfg {nwc, nws,
  vxpad}` (launch tunables, NVRTC defines), tables built once in `points` / `points2d` when the engine has a vs geometry:
  per knot b (static: one list) the samples of b's window sorted (home tile, oz, oy, ox, j) = `sort_vs`, items of ≤ SMAX
  samples of one tile, in-item order (class, oz, j); entries key << 25 | j (knot) / j (static); per-knot item offsets.
  `SegPoints::vs_table_bytes`, included in `table_bytes` / `device_bytes` (so the knot-mode memory plan's "sample tables,
  resident" counts them), moved by `to_host` / `to_device`.
- Dispatch `SegNufft::gather`: every forward gather (forward, residual, forward_knot, group, forward_sink) uses vs when
  it is active and the set has device vs tables of its geometry, else the tile kernel; host-resident sets streamed into a
  table slot run the tile kernels (the slots carry no vs tables).
- Autotune: after the tile forward search, every vs candidate (nwc 1–4 and the largest valid, nws 1 / 2, vxpad 0 / 1) is
  timed on the same interval / knot; vs is kept only where faster. Env HASTY_SEG_VS = 0 (off: no tables, not timed) /
  1 (on: always the best vs shape, active from construction) / unset (auto); HASTY_SEG_VS_TILE = tx,ty,tz,smax,
  HASTY_SEG_VS_CFG = nwc,nws,vxpad, HASTY_SEG_VS_LOG = 1 (per-candidate times). API: `set_vs_geom`, `set_vs`,
  `vs_active`, `vs_tune_times`, `vs_tune_cfg`.
- Default geometry: order 0 4 × 4 × 16, SMAX 64 (≥ 64 channels per batch) else 32; order 1 2 × 4 × 16 × 32.
- opbench: `--knot` (knot-major temporal engine), `--cb-cap N`, `--vs-compare` (forward / adjoint / gradient with vs off
  and on after the autotune; prints the autotune's tile and vs kernel times).
- Test rustlib/crates/hasty-nufft/tests/seg_vs.rs.

### 14.2 Tests (RTX 4090; rustlib/crates/hasty-nufft/tests/seg_vs.rs, all pass)

- vs forward vs the tile forward (variant 0) at orders 0 and 1: static (10 × 10 one batch, 12 coils in batches of 4,
  20 × 1), knot 1-D cubic (8 × 4; 6 coils in batches of 3 × 5 segments), knot 2-D linear × cubic (4 × 3); geometries
  4 × 4 × 16 × 64, 2 × 4 × 8 × 32, 4 × 2 × 8 × 48; launch shapes nwc 1 / 2 / 3 / 5 / 7 (incl. two block rows), nws 1 / 2 / 4,
  vxpad 0 / 1. W = 8: ≤ 7e-6 (bound 3e-5); W = 10: 4.6–8e-5, equal to the difference between two tile launch shapes on
  the same case (4–8e-5, random images: f32 noise of the tile kernels themselves), so the bound is max(3e-5, 2 × that
  floor). Adjointness against the unchanged adjoint 2e-8 – 3e-6 (= the tile forward's own).
- residual, forward_sink (host-summed parts) and forward_group through the vs gather equal the tile path (7e-6).
- autotune keeps vs only where faster (static 10 × 10, 30 k samples: tile 1.69 ms, vs 0.57 ms; knot 0.91 / 0.49 ms).
- Regression: seg_knot_kv, seg_tile_block, seg_stream, segnufft, seg_knot, seg_knot2d pass. seg_joint_chunks fails at
  its Φ-CGEMM-vs-kernel check (7.6e-5 > 1e-5) identically with HASTY_SEG_VS=0 (no vs tables): not caused by vs.
- compute-sanitizer memcheck (HASTY_NVRTC_LINEINFO=1, 3000 samples, every valid shape of the case): static o0, static
  o1, knot 1-D o0, knot 2-D o1: 0 errors (out/../vs_engine/san_*.log).

### 14.3 Operator-level timing (opbench `--vs-compare`, flow_thorax 2 mm, 450 × 450 × 224 mm, GNL; ms)

| case | M | autotune forward kernel, tile → vs | forward off → on | adjoint | gradient (fwd + adj) off → on |
|---|---|---|---|---|---|
| static 48 × 1 o0 (cb 48) | 4.32 M | 265.1 → 47.1 (5.6×) | 296.3 → 71.9 | 255 | 550.1 → 329.9 (1.67×) |
| static 1.3 M 10 × 10 o0 | 1.31 M | 72.0 → 34.6 (2.1×) | 121.2 → 81.7 | 175 | 296.1 → 257.3 (1.15×) |
| static 4 M 10 × 10 o0 | 4.32 M | run hit the 900 s timeout (first-time NVRTC of the tile autotune + vs) — rerun pending | | | |
| knot 4.0 M 10 × 10 o0 (cubic, 20 knots; kernel = knot 0) | 4.03 M | 39.4 → 26.5 (1.49×) | 1686.8 → 1462.9 | 1847 | 3533.3 → 3308.0 (1.07×) |
| knot 4.0 M 48 × 1 o0 (cb 48) | 4.03 M | 30.9 → 13.4 (2.3×) | 1146.5 → 741.6 | 1110 | 2260.6 → 1852.8 (1.22×) |
| knot 1.3 M 10 × 10 o0 | 1.33 M | 15.1 → 16.0: autotune kept the tile kernel | | | |
| order-1 cases | | queued (unit vs-engine, research/experiments/vs_engine/run_vs_engine.sh, resumable) | | | |

Knot path: per knot only KS / 20 of the samples (≈ 0.8 M) are active, so items are small and the per-knot launch /
FFT / pad overhead (20 knots × FFT + pad) dominates the operator: the forward kernel gain is 1.5–2.3× but the gradient
only 1.07–1.22×; at 1.3 M the vs kernel is slightly slower and the autotune correctly keeps the tile kernel. Measured
knot table memory 18–22 B / sample.

The autotune picked vs in every measured case (geometry 4 × 4 × 16; 10 × 10: nwc 7 = all channel groups in one block
row; 48 × 1: nwc 1, three block rows). The adjoint is unchanged, so the gradient gain is bounded by the
forward share: 1.67× at 48 × 1, 1.15× at 1.3 M 10 × 10, where the adjoint (175 ms) now dominates.

### 14.4 Table memory

Static: one u32 entry per sample + 16 B per item ≈ 4.4–4.6 B / sample measured (4 M: ≈ 19 MB per set); knot-major:
KS entries per sample (each sample is in the list of every knot of its support) ≈ 4 · KS + 16 KS / SMAX B / sample
(cubic: ≈ 17 B / sample), against ≈ 48 B records + ≈ 27–64 × 4 B tile-list entries of the existing tables. Counted in
`SegPoints::table_bytes` (memory ledger). Built in `points` only when the engine has a vs geometry (HASTY_SEG_VS unset
or 1, order ≤ 1, one frame slot, grid ≥ tile + 9 per axis).

### 14.5 Against the prototype

- 1.3 M 10 × 10 o0: engine vs kernel 34.6 ms vs prototype 30.7 ms (§12.6; +13 %); engine tile static 72.0 vs 68.9 ms. The
  engine kernel stages records through the sample index (random 48 B record reads instead of a pre-sorted copy), interpolates
  a_l(τ) and applies β e^{−iφ0} in the epilogue (prototype: pre-multiplied coefficients) and does a read-modify-write of y
  instead of a store; different grid (2 mm thorax, 282 × 282 × 140 fine grid vs 200³) and sample density.
- 4 M 48 × 1 o0: 47.1 ms vs prototype 34.3 ms (+37 %): the engine's default geometry is 4 × 4 × 16 × 32 while the
  prototype's tuned one was 2 × 4 × 16 × 32 (TX 2: 158 registers, more warps); the geometry is not autotuned in the
  engine (it fixes the tables). To check with HASTY_SEG_VS_TILE=2,4,16,32.

### 14.6 Next

1. Finish the queue (4 M 10 × 10 static, knot 4 M / 1.3 M, order 1) — rerun st_4m_10x10_o0 with a warm kernel cache.
2. Geometry rule: sweep HASTY_SEG_VS_TILE (2 × 4 vs 4 × 4, SMAX 32 / 64) per (channels, order); order 1 default
   (2 × 4 × 16 × 32) is a guess, not measured at scale.
3. Host-resident sets: stream the vs tables into the table slots (now: tile fallback).
4. The adjoint is now the larger half of the gradient: its levers (§7, fewer zero weights) matter more than further vs
   forward gains. Order ≥ 2: hook in segnufft_vs.cu (`#error`), the §13 vt table design is the candidate.


### 13.4 As built (proto_gather.cu `pg_fwd_vs`, defines VT / VTP / VSHR / VPF / VS2 / VPT / VMINB; proto_gather.rs)

- Correctness: `--check` against the f64 reference, rel err 6.5–6.8e-7 at orders 2 and 3, 6.2–6.3e-7 at order 0, for
  every smoke shape (own / atomic epilogue, NWS 2, TX 1 / 2 / 4, every tunable) and every tuned BEST. The f64 reference
  (`weights`) now has the order-3 terms (degree-3 real, degree-4 imaginary, as `stage_weights_hk`); the order-3 test
  data (|u| ≤ 0.5, |H'| ≤ 0.05) make the degree-3/4 terms dominate the weights (rms 134 vs 10 at order 2), so the
  check exercises them strongly. compute-sanitizer memcheck: 0 errors (vt o3 with VTP / VSHR / VPT, o0 with every
  tunable; out/san_vt_o3.log, out/san_vs_o0t.log).
- ptxas: vt o2 / o3 TX 2 190–198 registers, TX 4 244–254, no spills. SASS of one vt o2 body: 366 instructions
  (229 FFMA, 46 FMUL, 42 LDS, 3 STS, 1 SHFL): counted 343.
- Driver: `--fixed` takes several `;`-separated vs configurations with `key=value` tunables (vt, vtp, shr, pf, s2, pt,
  minb, carve), `--each` times / checks / logs every one; `--order 3` (vs only) builds u / H' and passes uw / htab;
  `--mode base --order 3` uses the engine's `points_hi`. vs autotune now has two stages: the shape search with
  default tunables, then `vs_refine` (the kept tunables and their combinations) on the best shape. `ldsbench` has
  `LDSB_BLK=1` (blocked address pattern).
- Queues: run_vt.sh (smoke, Task A autotunes, production references, Task B A/B), run_vt2.sh (VTP smoke, vt A/B),
  run_vt3.sh (sanitizer, final two-stage autotune), run_vt_ncu.sh (ncu, application replay). Logs out/{smoke_vt*,
  smoke_o0t*, smoke_vtp*, vt_*, vtab_*, knotref2_*, ab_*, fin_*, san_vt_o3, san_vs_o0t, ldsbench_blk}.log, rows in
  out/results.jsonl, out/ncu/vt_m4M_c10_o2.ncu-rep. Binaries: frozen copies in the session scratchpad (another agent
  rebuilds hasty-nufft).

### 13.5 Orders 2 / 3: predicted vs measured (RTX 4090, same session; % = useful FMAs / FP32 peak at the nominal clock)

| case | predicted vt | **vt (final two-stage autotune)** | shape (tx,ty,tz,smax,nwc) + tunables | vs §12 per-lane body | **production knot forward (same session)** | vt vs production |
|---|---|---|---|---|---|---|
| 4 M 10 × 10 o2 | 190 ms (150–240), 20 % | **165.6 ms (23.0 %)** | 4,4,16,48,7 shr pt | 276.4 | 236.6 | **1.43×** |
| 1.3 M 10 × 10 o2 | 70 ms (55–90) | **59.6 ms (20.8 %)** | 4,4,16,48,7 pt | 102.0 | 80.9 | **1.36×** |
| 4 M 10 × 10 o3 | 220 ms (170–270), 17.5 % | **186.1–188.7 ms (20.2 %)** | 2,4,16,48,7 (4,4,8 shr: 186.1) | — | 299.8 | **1.59×** |
| 1.3 M 10 × 10 o3 | | **67.6 ms (18.3 %)** | 4,4,16,48,7 pt | — | 99.9 | **1.48×** |

- The prediction held (measured 13–15 % faster than the central guess, inside the range). The table removes the
  per-lane weight rebuild: 4 M o2 276 → 166 ms against the §12 body.
- Order 3 costs vt only +12–14 % over order 2 (the extra terms sit in the per-(sample, plane) build, shared by the 16
  channel lanes), but costs production +27 % (299.8 vs 236.6 ms; its weight build is per (sample, row) per block, and
  the tile kernel spends 2.2× MACs on zero weights). So vt's lead grows to 1.5–1.6× at the order the reconstruction
  runs.
- Every best shape is one block of 7 warps per SM (NWC 7 for 112 channel lanes; 92–98 KB of shared memory: SMAX ×
  (752–864 B records + 896 B accumulators) + 11 KB tables; 190–250 registers). Larger items win throughout (SMAX
  48 > 32 > 16: 169.7 / 180.2 / 209.2 ms at 4 M o2); SMAX 64 does not fit 99 KB.
- vt A/B at fixed shapes (vtab_*, ms; 4 M o2 / 1.3 M o2 / 4 M o3 / 1.3 M o3):

  | shape | plain | VTP | VSHR | VTP + VSHR |
  |---|---|---|---|---|
  | 4,4,8,48,7 | 172.0 / 61.3 / 189.9 / 67.9 | 164.4 / 61.7 / 191.0 / 71.1 | 166.1 / 61.5 / 186.1 / 69.0 | 164.3 / 62.3 / 190.6 / 70.3 |
  | 2,4,8,48,7 | 169.1 / 65.7 / 188.2 / 71.8 | 174.7 / 68.4 / 198.9 / 77.4 | | 175.5 / 68.9 / 204.0 / 77.4 |
  | 4,4,16,48,7 | 165.5 / 61.0 / 186.3 / 67.6 | 162.7 / 60.1 / 190.0 / 69.5 | | |

  VTP (software-pipelined build, next sample's table during this sample's MACs) is −4 % at 4 M o2 TX 4 and +2–8 %
  elsewhere (registers: TX 4 hits the 255 cap, TX 2 gains nothing); VSHR −1–3 %; TZ 16 −1–4 %. All kept only as
  refinement candidates.
- ncu (4 M o2, 4,4,8,48,7, application replay; out/ncu/vt_m4M_c10_o2): issue slots 50 % busy, FMA pipe 38 %, LSU
  (shared) wavefronts 53 % of peak, L2 22 %, DRAM 5 %; 1.75 active / 0.80 eligible warps per scheduler; stalls per
  issue: wait 0.46, short scoreboard (shared-load latency) 0.43, dispatch 0.33, barrier 0.20, long scoreboard 0.09.
  **Latency-bound at 7 warps / SM**: neither the shared pipe (counted bound 41 %) nor the FFMA pipe is saturated;
  each warp waits on its own chain build → `__syncwarp` → 25 LDS.128 → 10 FFMA chains → SHFL → accumulator RMW.
  From the measured time: ≈ 197 SM clocks per warp-body against ≈ 115 shared / ≈ 92 issue clocks counted.
- Blocked shared loads (new `ldsbench` rows, `LDSB_BLK=1`, out/ldsbench_blk.log): a warp-wide LDS.128 whose 32 lanes
  read 4 distinct float4, one per quarter-warp (8 lanes), costs 2.5–2.6 clk, the same as a full broadcast; the same 4
  addresses interleaved (lane % 4) cost 4.3–4.5 clk. With 16 FFMA per load the blocked pattern runs the FFMA pipe at
  79 % (interleaved 65 %). §11.4's "≥ 4 distinct float4 cost 4 clk" holds only for interleaved patterns.

### 13.6 Order 0: what each tunable did (ab_*: one invocation per case at the §12.6 shape, ms; final = two-stage autotune)

| change (alone) | 4 M 48 × 1 | 4 M 10 × 10 | 4 M 16 × 1 | 1.3 M 10 × 10 | 4 M 16 × 12 |
|---|---|---|---|---|---|
| §12.6 shape, this session | 34.48 (26.5 %) | 78.68 (24.2 %) | 12.28 (24.8 %) | 31.68 (19.5 %) | 119.02 (30.7 %) |
| VSHR (row hand-off by SHFL) | **32.89 (−4.6 %)** | 81.44 (+3.5 %) | 12.94 (+5.4 %) | 32.79 (+3.5 %) | **116.10 (−2.5 %)** |
| carve-out 50 % | 36.55 (+6.0 %) | 78.91 (+0.3 %) | 15.97 (+30 %) | 31.82 (+0.4 %) | 160.11 (+35 %) |
| carve-out 0 % | 72.21 (+109 %) | 78.26 (−0.5 %) | 68.64 (+459 %) | 31.97 (+0.9 %) | 160.95 (+35 %) |
| VPF (next row one step ahead) | 43.57 (+26 %; 243 regs → 6 warps) | **76.64 (−2.6 %)** | 12.74 (+3.7 %) | 32.37 (+2.2 %) | **115.58 (−2.9 %)** |
| VS2 (two samples per iteration) | **33.15 (−3.9 %)** | **73.95 (−6.0 %)** | 16.01 (+30 %) | 32.43 (+2.4 %) | 117.36 (−1.4 %) |
| VPT (parallel range table) | 34.09 (−1.1 %) | **75.61 (−3.9 %)** | 12.53 (+2.0 %) | 31.38 (−0.9 %) | **113.69 (−4.5 %)** |
| TX 1 × TY 16 (100-value window) | 53.50 (+55 %) | 129.10 (+64 %) | 18.15 (+48 %) | 54.71 (+73 %) | 208.06 (+75 %) |
| TX 1 + VSHR | 46.07 (+34 %) | 124.29 (+58 %) | 16.44 (+34 %) | 52.45 (+66 %) | 170.30 (+43 %) |
| TX 1 + VSHR + VMINB (128 regs, 12–16 warps) | 60.53 (+76 %, 72 B spills) | 135.61 (+72 %) | 21.65 (+76 %) | 59.91 (+89 %) | 240.86 (+102 %) |
| **final two-stage autotune** | **33.52 (27.3 %)** shr s2 pt | **74.10 (25.7 %)** shr s2 pt | **12.11 (25.1 %)** shr pf | **31.19 (19.8 %)** pt | **116.58 (31.4 %)** pt |
| §12.6 (previous session) | 34.26 | 74.73 | 12.28 | 30.65 | 120.41 |

- Run-to-run spread between sessions is 1–5 % (the §12.6 shapes: 34.5 / 78.7 / 12.3 / 31.7 / 119.0 here against
  34.3 / 74.7 / 12.3 / 30.7 / 120.4), the size of most single effects. Net result of Task B: **0–3 %** over §12.6.
- VSHR cuts the window's L2 rows by 19 % (TY 4) and wins where the shape is TX 2 / L2-bound (48 × 1); at TX 4 the
  extra 26 SHFL per step and +4 registers cost more than the bytes save.
- The carve-out only removes shared memory: with 1-warp or small blocks it cuts resident blocks (16 × 1: 8 → 1–5
  warps / SM, 16 × 12: 8 → 4); at 1 block / SM (10 × 10) it changes nothing. L1 reuse does not appear (L1 hit rate
  1.6–5.5 % with any split).
- TX 1 loses everywhere although it frees 10–40 registers: items get emptier (items 135 k vs 72 k at 4 M), samples
  per class drop and the window still loads (TY + 9) rows per plane for half the reuse in x. Forcing 12–16 warps with
  VMINB spills (64–80 B) and is slower still. A smaller register window does not buy occupancy without spills; the
  occupancy limit at 10 × 10 is shared memory (71 KB / block) as much as registers.
- Kept (autotune refinement stage, `vs_refine`): VSHR, VPT, VS2, VPF and their combinations; carve-out (driver
  attribute, `--fixed ... carve=`) and VMINB remain available but not in the autotune set; TX 1 tiles stay in the
  full set only.

### 13.7 Limiter now

- **Order 2 / 3 (vt): latency at 7 warps / SM.** One block of 7 warps per SM, fixed by shared memory (accumulators
  SMAX × 112 lanes × 8 B + records) and by 190–250 registers; issue 50 % busy, shared pipe 53 %, FMA 38 %, L2 22 %.
  Pipelining the build inside a warp (VTP) does not help because the 255-register cap leaves no room for two bodies'
  live state.
- **Order 0: the same structure.** 7–12 warps / SM; L2-bound at 48 × 1 (80 %, §12.7); none of the six levers moves it
  by more than 5 %.

### 13.8 Next

1. vt is the order-2 / 3 forward candidate for the engine: 1.36–1.59× faster than the production knot forward
   (1.59× at 4 M order 3, the reconstruction's order) at f32 reference accuracy. Engine path as §12.9 item 4 plus the
   record layout of §13.1 and uw / htab trailing arguments (as the knot kernels).
2. Occupancy for vt: halve the shared accumulators (store a channel group's sample accumulators as bf16x2 / f16x2
   pairs only if the error budget allows: probably not; better split each item's samples into two 24-sample halves
   with the window reloaded, if L2 allows), or move the records out of shared memory (they are only read by the
   build: L1-resident global records, 36 KB / block freed → 2 blocks / SM on the 4090; on H100 / A100 (164–228 KB) 2–3
   blocks fit as is).
3. Layout with 2 channels per lane and 4 quarter-warp groups (2 row halves × 2 x halves): §13.5's blocked-load
   result makes each LDS.128 feed 16 FFMA at broadcast cost (MAC loads 25 → 15 per body at 2.6 clk), registers
   20 (TX + 4) for the window. Counted: shared ≈ 105 vs 120 clk per body; worth it only together with item 2 (more
   warps), since vt is latency- rather than shared-bound at 7 warps.
4. Block-shared tables (one build per (sample, plane) for all 7 channel-group warps, one barrier per plane): cuts the
   build's share of issue and shared time by 7×, but needs 2 × SMAX × 800 B more shared memory, which only fits on
   the larger-SM GPUs; to try on the cluster nodes.

## 15. A faster adjoint (spreading) kernel (2026-11-19, seventh session)

Question: is there an adjoint (k-space → grid) kernel for the segmented NUFFT that beats the production adjoint at
orders 0–3 (velocity reconstruction: order 3) on the static and knot-major paths, now that the vs forward made the
adjoint the larger half of the gradient (opbench 2 mm thorax: static 4.3 M 48 × 1 forward 72 / adjoint 255 ms; prototype
production knot adjoint, 4 M 10 × 10: 132 ms o0 (14 % of peak), 288.8 ms o2, 341.7 ms o3 (`knotref2_*`, §13))?
Decision: which adjoint design to build into the engine as a new autotuned variant. §15.1–15.3 were written before the
kernel; measurements from §15.4 on.

Notation: z_j[ch] = conj(a_jl) β y_c(j) (per sample and channel, built once per staged sample), G[p][ch] += Σ_j
conj(w_j(p)) z_j[ch]: per tile a sparse GEMM (points × channels × samples). Useful MACs as §4 (M · W³ · NCH; 2 FFMA
each at order 0, 4 at orders 1–3).

### 15.1 Counts that decide the design

**Global RED throughput** (new `--mode redbench`, kernel `pg_red_bench`: every home tile flushes its footprint box
(TX + 9)(TY + 9)(TZ + 9) for 112 channels, tiles in x-fastest order over the 200³ grid; out/../adjoint/redbench.log):

| tile | RED.ADD.F32, stride 8 B (re, im per lane) | RED packed (lanes cover whole 32-B sectors) | plain read-modify-write | plain store |
|---|---|---|---|---|
| 4 × 4 × 16 | 190 G float/s | 236 | 623 | 491 |
| 2 × 4 × 16 | 211 | 307 | 721 | 535 |
| 4 × 4 × 8 | 201 | 265 | 605 | 528 |

So one complex RED costs ≈ 8 ps of the whole GPU (packed), against ≈ 0.19 ns of useful MACs per (sample, channel) at
25 % of peak (1000 MACs × 2 FFMA / (0.25 · 42 T/s)).

**Accumulator-stationary dual of vs** (lanes own channels, a 5 × (TX + 9) register window of accumulators per half,
weights by broadcast, z per lane). The body is cheap — per lane 100 FFMA + 12 FMUL + 6 shared instr. (5 weight loads,
1 z; no SHFL, no accumulator RMW), and its 100 FFMA update independent accumulators (no dependent chains) — but every
point of an item's footprint must leave the registers once per item. Points flushed per sample = item footprint /
samples = the §12.1 staging column × 250: 4 M: 43 (4 × 4, SMAX 64), 30 (SMAX 128), 50 (2 × 4, 64); 1.3 M: 77 / 64. The
lower bound with infinite items is the x/y halo overlap of the tile columns, ((W + TX − 1)/TX)((W + TY − 1)/TY) ≈ 11
(4 × 4 columns) — still ≥ 11 complex flushes per (sample, channel).
- global REDs: 4 M 10 × 10, 43 points: 4e6 · 43 · 100 · 2 = 3.4e10 float REDs ≈ 146 ms at 236 G/s — more than the whole
  production adjoint (132 ms). Even the infinite-item bound (11) costs ≈ 37 ms. **Dead.**
- two-phase colouring (plain RMW, no atomics): footprints of concurrently running tiles must not overlap, so a phase is
  spread over the whole grid (4 × 4 × 2 colours for 4 × 4 × 16 tiles); 128 resident blocks × 3.8 MB footprint each
  (13 · 13 · 25 points × 112 ch × 8 B) ≈ 485 MB ≫ 72 MB L2: the RMW goes to DRAM, 3.4e10 × 8 B ≈ 270 GB at ≈ 1 TB/s.
  **Dead.**
- per-block shared halo with one global add per halo point per block: the halo is (TX + 9)(TY + 9)(npl) points ×
  16 · NWC channels ≈ 3.8 MB at 4 × 4 × 16: does not fit; with lanes owning channels a block cannot hold a halo.
  **Dead.**
- owner-computes with lanes = channels and the tile's own points in registers (each block processes every sample whose
  support reaches its tile, over its exact rows): per (sample, tile, plane) visit only ≈ 3 × 3 points per lane get a
  MAC (≈ 105 visits per sample at 4 × 4 tiles) against 4–5 shared instructions per visit (z, x / y / z weights): ≈ 2 FFMA
  per shared instruction. **Dead in that form** — but the same ownership with lanes owning channels *and* several points
  × rows in registers is the GEMM register blocking below.

**Output-driven, production (knot variant 1).** Per (sample, sub-box warp) a lane does 1 weight LDS + ACB · NSEG / 2
broadcast LDS.128 of z for 2 · ACB · NSEG FFMA: 4 FFMA per LDS.128 whatever ACB (z is per (sample, channel), every lane
reloads it), i.e. ≈ 2.4 clk of shared pipe per 1 clk of FFMA: ≤ 30–40 % FFMA rate × ≈ 0.55 useful (sub-box skip: ≈ 54
points processed per (sample, tile), 29 of them in the support) → ≈ 20 % bound, 14.4 % measured. The lever is register blocking in points:
a thread that holds P points reuses each z load P times.

### 15.2 Design 7 `ra` (row-blocked output adjoint): exact rows, lanes = channel blocks × x-groups

- Output tile TX × TY × TZ (TX ∈ {2, 4}, TY ∈ {4, 8}, TZ ∈ {4, 8, 16}); per tile the list of every sample whose support
  meets it (as the production tile lists), items of ≤ LMAX samples; an unsplit tile stores, a split one adds with
  atomics (only the dense k-space centre).
- Block = (item, channel-block row); warp = one plane tz of the tile (TZ warps); lane = (x-group g of NXG, channel block
  cb of 32 / NXG); thread holds acc[TY rows][PT = TX / NXG points][C channels] complex in registers (o0 TX 4, TY 4, NXG 2,
  C 7: 112 floats).
- Samples staged per chunk (records: kernel values on the tile's cells, oy / oz, z for the block's channels, built as
  y_c · conj(a_l) in shared memory). Per staged sample a warp skips it unless its plane is in the support (warp-uniform),
  then loads z (C / 2 LDS.128, 16–32 lanes distinct), φz, φx at its PT points, φy of the TY rows (broadcast), and for
  every **covered row** (warp-uniform test: exact y / z, no zero-weight rows) forms PT weights (FMUL) and 2 · PT · C FFMA.
  x: lanes of different x-groups see different points, so the x-waste is W / (W + TX − 1) (TX 4: 0.77, TX 2: 0.91);
  NXG 1 (all lanes the same points) makes x exact too by a warp-uniform per-point test.
- Orders 1–3: per (sample, plane, tile) visit the warp builds the TY × TX complex weights cooperatively into a
  warp-private shared table (one lane per point: ER / EQ of the plane, R_a / Q_a of its row, its point: the vt terms of
  §13.1, CORR 1 = e terms zero) and reads them as broadcast LDS.128 (PT = 2: one per row); 4 · PT · C FFMA per row.
- Tunables (NVRTC defines, autotune): TX, TY, TZ, NXG, C, CHK (staged chunk), LMAX; nothing architecture-specific.

### 15.3 Count and prediction (per warp-visit = one sample × one plane × one tile; per lane)

| | o0, 4 × 4 × 8, NXG 2, C 7 (10 × 10) | o0 48 × 1 (C 3) | o0 16 × 1 (C 1) | o3, C 7 |
|---|---|---|---|---|
| covered rows Rc (exact) | 3.08 | 3.08 | 3.08 | 3.08 |
| FFMA (useful = × 0.77 x-waste) | 86 (66) | 37 (28) | 12 (9.5) | 172 (133) |
| FMUL / build | 9 | 9 | 9 | ≈ 87 build (16 of 32 lanes, vt terms) |
| shared instr. | 4 z + 3 w = 7 (≈ 19 clk) | 5 | 4 | 4 z + 3 w + ≈ 10 build = 17 |
| int / branch (skip, row tests) | ≈ 15 | ≈ 15 | ≈ 15 | ≈ 20 |
| issue clk (instr / 4) | 29 | 17 | 10 | 74 |
| useful FFMA clk | 16.5 | 7 | 2.4 | 33 |
| bound (useful / max(issue, shared, FFMA)) | **57 %** | 41 % | 24 % | **45 %** |
| registers | 112 acc + ≈ 40 | 48 + 40 | 16 + 40 | 112 + ≈ 60 |

Visits per sample: ((W + TX − 1)/TX)((W + TY − 1)/TY) · W = 105.6 at 4 × 4; staging per (sample, tile) (≈ 16 kernel
evaluations, z for 112 channels = 448 FFMA) ≈ 10 % of the useful FFMA at TZ 8. Skipped samples (plane not in the
support: 1 − W / (W + TZ − 1) = 41 % at TZ 8) cost ≈ 4 instructions each.

Achieved / bound so far: vs 0.38 (o0), vt 0.5–0.6. With ≈ 0.45 and the staging, **predicted**: o0 4 M 10 × 10 ≈ 23 %
of issued → with channel padding 112 / 100 ≈ 21 % useful → **≈ 90 ms (range 70–120) vs production 132 ms (1.5×)**; 4 M
48 × 1 ≈ 18 % → ≈ 50 ms vs 144 ms (≈ 2.9×); 4 M 16 × 1 ≈ 10 % → ≈ 30 ms vs 75.9 static; 1.3 M 10 × 10 ≈ 30 ms vs
45.4; o2 / o3 4 M 10 × 10 ≈ 20 % → **≈ 190 ms vs 288.8 / 341.7 (1.5–1.8×)**.

### 15.4 As built (proto_gather.cu `pg_adj_ra`, `#ifdef RA`; proto_gather.rs `--mode adj --designs ra`)

- Correctness: f64 adjoint at 48 grid points × every channel (the exact transpose of the `weights` forward reference,
  now with the order-1 terms: Q only, no e terms) and adjointness |⟨A x, y⟩ − ⟨x, Aᴴ y⟩| / |⟨A x, y⟩| with y supported on
  a strided sample subset, ⟨A x, y⟩ from the f64 forward of the hash grid x, ⟨x, Aᴴ y⟩ over the whole GPU grid. Every
  smoke shape (TX 2 / 4, NXG 1 / 2 / 4, split items with atomics at LMAX 64) and every tuned BEST: rel err 0.8–2.1e-6,
  adjointness 2.5e-7 – 1.4e-6, orders 0–3 (out/../adjoint/smoke*_ra.log, out/ra_*.log).
- compute-sanitizer memcheck (100 k samples, `HASTY_NVRTC_LINEINFO=1`, three shapes incl. split items): 0 errors at
  orders 0 and 3 (adjoint/out/san_ra_o{0,3}.log).
- Driver: `--fixed tx,ty,tz,nxg,cb,chk,lmax[;...]`, `--each` (times and checks every configuration), `PG_DEFS=` adds
  RA_NOMAC (no MAC loop), RA_NOZ (no z build), RA_SPLITB (orders ≥ 1: weights in two steps). Timings are kernel only;
  the grid memset (6.8 ms for 6.4 GB) is measured separately, as the production autotune times only its kernel.
  `--mode redbench`: §15.1.
- Three fixes found by ablation on the way (4 M 10 × 10 o0, best shape 4 × 4 × 4, NXG 2, CB 7; ms):

  | version | full | no MAC | no z build |
  |---|---|---|---|
  | first (z: [s][cb][CB] blocks 56–64 B apart → 4–8-way bank conflicts; z built per (sample, channel) from 3 dependent global loads) | 142.6 | — | — |
  | z interleaved [s][pair][cb] (conflict-free LDS.128) | 128.4 | 32–50 | 86 |
  | raw inputs (position, ΔC, a_l, y per coil) prefetched into registers one chunk ahead, z built from shared memory | 123.2 | 24–55 | 92 |
  | z build with the per-channel index math hoisted out of the sample loop (≈ 40 → ≈ 10 instructions per element) | **96.6** | 22–27 | |

  The z build runs once per (sample, tile visit, channel): 34 visits × 112 channels per sample at 4 × 4 × 4 tiles, so
  its per-element instruction count matters as much as the MAC's (§15.3 had counted it as 4 FFMA).
- Order ≥ 1: staging the weights in two steps (per-(sample, plane) ER / EQ, then rows; ≈ 70 instead of ≈ 150 FFMA per
  row) was slower (o3: 192 → 207 ms; one more barrier, half the block idle in the plane step): kept as RA_SPLITB, off.

### 15.5 Predicted vs measured (RTX 4090, one session; kernel time, % = useful FMAs / FP32 peak at the nominal clock)

Queue research/experiments/adjoint/run_ra.sh (unit `adj-ra`), logs adjoint/out/{ra,knot,base,san}_*.log, rows in
adjoint/results.jsonl. ra: quick autotune (≈ 100 candidates at o0, fewer at o ≥ 1), BEST re-timed with 5 reps and
checked. Production: `--mode base --knot` (the knot-major kernels, variants 0 / 1 timed by the engine autotune, incl. the
§7 acb blocking) and the static tile kernels (`--mode base`), same session.

| case | predicted ra | **ra measured** | shape (tx,ty,tz, nxg, cb, chk) | production knot adjoint | static adjoint | ra vs knot |
|---|---|---|---|---|---|---|
| 4 M 10 × 10 o0 | 90 ms (70–120) | **92.0 (20.7 %)** | 4,4,4, 2, 7, 16 | 133.9 (14.2 %) | 342.0 | **1.46×** |
| 4 M 48 × 1 o0 | 50 | **66.0 (13.8 %)** | 4,8,4, 2, 3, 16 (4,4,4: 77.5) | 145.2 (6.3 %) | 209.1 | **2.20×** |
| 4 M 16 × 1 o0 | 30 | **31.0 (9.8 %)** | 4,8,4, 4, 2, 16 (4,4,4: 32.3) | 52.1 (5.8 %) | (75.9, §4) | **1.68×** |
| 1.3 M 10 × 10 o0 | 30 | **32.5 (19.0 %)** | 4,4,4, 2, 7, 16 | 45.6 (13.6 %) | | **1.40×** |
| 4 M 16 × 12 o0 | | **163.7 (22.3 %)** | 4,4,4, 2, 6, 32 | 220.8 (16.6 %) | (338.8, §4) | **1.35×** |
| 4 M 10 × 10 o2 | 190 | **181.7 (20.9 %)** | 4,4,4, 2, 7, 16 | 288.5 (13.2 %) | | **1.59×** |
| 1.3 M 10 × 10 o2 | | **64.7 (19.1 %)** | 4,4,4, 2, 7, 16 | 97.8 (12.7 %) | | **1.51×** |
| 4 M 10 × 10 o3 | 190 | **190.3 (20.0 %)** | 4,4,4, 2, 7, 16 | 342.3 (11.1 %) | | **1.80×** |
| 1.3 M 10 × 10 o3 | | **64.4 (19.2 %)** | 4,4,4, 2, 7, 16 | 114.2 (10.8 %) | | **1.77×** |

- Every case is faster than the production adjoint: 1.35–1.46× at order 0 with 10 × 10 / 16 × 12, 1.7–2.2× with few
  coils, 1.5–1.6× at order 2 and **1.8× at order 3** (the reconstruction's order). rel err 0.96–1.34e-6 (< 1e-5),
  adjointness 1.8e-7 – 1.5e-6, every row.
- Order 3 costs ra +5 % over order 2 (190 vs 182 ms) against +19 % for production: the order-3 terms sit in the
  per-(sample, row) weight staging, shared by all channels and the four points of a row.
- The prediction held for the 10 × 10 cases at all orders (within 5 %); it was 30 % optimistic at 48 × 1 (the few-channel
  bodies are short, §15.3: 41 % bound, and staging per channel is relatively larger).
- 4 × 4 × 4 wins every 10 × 10 / 16 × 12 case: it is exactly the engine's TT = 4 tile (lists, items, split flags).
- The memset of the grid (6.8 ms for 6.4 GB) is needed only because split items add atomically (here 55.8 k of the
  109 k items at LMAX 2048); unsplit tiles store. The production knot adjoint has the same split-item requirement.

### 15.6 Limiter (Nsight Compute, 4 M 10 × 10 o0, 4,4,4, 2, 7, 16; targeted sections, `--launch-count 1`)

110.7 ms under ncu; registers 162 → 3 blocks × 4 warps = **12 warps / SM** (registers and shared memory both limit at
3 blocks); issue slots 60 % busy (IPC 2.41), FMA pipe 39 %, LSU 41 %, L1 / shared 54 %, L2 10 %. Stalls per issued
instruction: wait 1.15 (fixed-latency dependencies), barrier 0.85 (per-chunk staging), not selected 0.74, short
scoreboard (shared loads) 0.70, branch 0.16. So: **latency at 12 warps plus the per-chunk barriers**, not a pipe. The
ablation agrees: without the MAC loop the kernel takes 22–27 ms (staging, barriers, store); with it 92 ms. Per chunk
of 16 samples the block stages (two barriers), then each of its 4 plane-warps does ≈ 12 sample visits of ≈ 120
instructions; the staging cost is per (sample, tile visit) — 34 visits per sample at 4 × 4 × 4 — and cannot be shared
across tiles.

### 15.7 Engine integration: not done this session

The prototype wins clearly at the orders that matter (1.8× at order 3), so integration is justified, but it was not
attempted: another agent is changing segnufft.rs (slot zeroing, pad / crop fusion, FFT plans), and the integration
needs a new kernel file plus autotune / dispatch plumbing and GPU tests. What it needs (counted, not built):
- **No new tables for the 4 × 4 × 4 shape**: the ra items are the engine's TT = 4 tile lists (samples whose support
  meets the tile, items ≤ max_item, split flag), and on the knot path the existing `knot_window` filter applies
  unchanged. Shapes 4 × 8 × 4 (few coils: 48 × 1 66 vs 77.5 ms) would need their own lists (+ ≈ 22.5 u32 per sample).
- Kernel in a new file (segnufft_ra.cu, as segnufft_vs.cu): the engine staging (`stage_kv<-1, true>`: kernel factors
  on the tile, zc = y a_l β per (sample, coil, segment), `STAGE_HC`, `STAGE_WEIGHTS_K` for the complex weights at
  orders ≥ 1; real weights at order 0 come from the axis factors in the MAC) is already what ra stages; new is the MAC
  layout (warp = tile plane, lane = (x-group, channel block), acc[4][2][CB] in registers, exact row / plane skips), the
  conflict-free zc layout [s][pair][cb], the register prefetch of the raw inputs, and an output park that writes runs
  of cpba · NSEG channels per point (store / add / atomic as `adj_store`).
- Host: an `RaCfg {nxg, cb, chk}` next to TileCfg, autotuned after the tile adjoint like vs (§14.1), env switch
  `HASTY_SEG_VSA=0/1`, tests as tests/seg_vs.rs (against variant 0, adjointness, orders 0–3, static + knot 1-D / 2-D).
  The slot-zeroing change of the other agent matters: ra relies on zeroed split tiles exactly as the production adjoint.

### 15.8 Next

1. Engine integration as §15.7 (after the FFT-side changes land), then opbench `--vs-compare`-style forward / adjoint /
   gradient tables: with vs forward + ra adjoint the 4 M 10 × 10 order-3 gradient kernels would be ≈ 186 + 190 ms
   against 300 + 342 ms today.
2. Occupancy / barriers (§15.6): fewer registers per lane (CB 7 → 162–222 registers) or deeper chunks with cp.async
   staging (no prefetch registers), and LMAX 4096 (half the items are split at 2048: atomics + memset).
3. Few-coil cases (16 × 1: 9.8 %): bodies are too short at CB 1–3; lanes over 2 planes or 2 tiles per warp.
4. Portability: nothing is 4090-specific (NVRTC defines; occupancy is set by registers and 100 KB shared memory).
   H100 / A100 have 164–228 KB shared memory per SM → more resident blocks of the same shape; to be measured on the
   cluster nodes.

## 16. Engine integration of ra and vt (2026-11-19, eighth session)

Question: do the §15 `ra` adjoint and the §13 `vt` forward (orders 2 / 3) keep their kernel-level gains as engine
variants, operator-level and in the gradient, on the paths the reconstruction runs? Decision: whether the autotune
should offer them by default, and what limits them in the engine.

### 16.1 Which kernels the velocity reconstruction launches

recon_p2np with `--vmodel knots` builds a 2-D knot-major engine (`SegNufft::new_temporal2d`, respiratory × cardiac
knots) at correction order 3 (`conc_hi`) and its sample tables with `points2d_hi`. `SegSets` calls `forward`,
`forward_residual` and `adjoint` (group calls are refused on knot engines). On a knot engine these go to
`forward_knot` / `adjoint_knot`, i.e. the knot kernels `seg_forward_k` / `seg_adjoint_k` at CORR 3 (uw / htab trailing
arguments, `stage_hc_k`, `stage_weights_hk`), per knot and coil batch; the multi-image kernels (segnufft_multi.cu) are
only used by static / interval-major engines at order 3 and by the fat paths. ra and vt cover exactly the knot path
(1-D and 2-D knots, orders 0–3) and the static path at orders 0–2.

### 16.2 What was added (nothing removed; kv 0 / 1 / 2, the vs order 0 / 1 body and the multi kernels unchanged)

- **ra** (rustlib/crates/hasty-nufft/src/segnufft_ra.cu, kernel `seg_adjoint_ra`; host src/segnufft/ra.rs). The §15
  design on the engine's own TT = 4 tile lists, items and split flags (no new tables). Block = (work item, channel-block
  row), warp = tile plane, lane = (x group, channel block), acc[4 rows][4 / NXG points][CB channels] in registers.
  Engine staging: the item's list is compacted into shared memory (knot engines: the knot window, as `knot_window` /
  `knot_window2d`), records and coil values prefetched into registers one chunk ahead, kernel values on the tile cells
  with `wrapdiff` (as `stage_kv`), w_l = β conj(a_l(τ)) e^{iφ0}, z = y_c w_l per (sample, channel) in the conflict-free
  [s][pair][cb][2] layout, complex weights per covered (sample, tile row) with exactly the `stage_weights` /
  `stage_weights_hk` terms (orders 1–3). Output parked per warp row and written as runs of 32 / NXG · CB channels:
  split items `atomicAdd`, unsplit items store (bit 0 of addmask: read-modify-write), an item with nothing in the knot
  window stores zeros (store mode) — the HASTY_SEG_ZERO=tiles contract of temporal_operator.md §23.
  Tunables (NVRTC defines): NXG 1 | 2 | 4, CB (4 / NXG · CB ≤ 16), CHK 16 | 32; `RaCfg {nxg, cb, chk}`.
  Dispatch: one new `SegNufft::scatter` (ra when active and applicable, else the tile kernel) replaces the five direct
  tile-scatter calls in segnufft.rs (`adjoint`, `adjoint_knot`, `group_adj_in`, both branches of `adjoint_source`).
  Applicable: one frame slot (static or knot-major), TT = 4, order ≤ 2 static / ≤ 3 knot; host-resident sets streamed
  into a table slot run ra too (it needs only the tile lists).
- **vt** (segnufft_vs.cu at CORR 2 / 3, replacing the `#error` hook; host segnufft/vs.rs). The §13.1 body: record with
  kx[5][12], wy / wz derivatives, e[15], ΔC[6] (+ hc[25] at CORR 3, from `uw` / `htab` of the original sample index),
  per (sample, plane) a warp-built double-buffered 10 × 10 complex table (1600 B per warp after the accumulators),
  25 broadcast LDS.128 per lane; W < 10 via zero weights and a guarded MAC; vxpad 0 only. Same tables, items, launch
  shapes, epilogue and dispatch (`SegNufft::gather`) as vs. Default geometry at orders 2 / 3: 4 × 4 × 16, SMAX 48
  (prototype best, §13.5). Tables are now also built at orders 2 / 3 (static order ≤ 2, knot order ≤ 3).
- Autotune: after the tile adjoint search, every ra candidate (per NXG the CB that covers the batch's channels in 1–4
  block rows, CHK 16 / 32) is timed on the same interval / knot in store mode; kept only where faster. vt is timed in
  the existing vs step.
- Switches: `HASTY_SEG_VSA = 0 | 1 | auto` (ra: off / always the best ra shape, active from construction / autotune
  choice), `HASTY_SEG_VSA_CFG = nxg,cb,chk`, `HASTY_SEG_VSA_LOG = 1`; `HASTY_SEG_VT = 0 | 1 | auto` (as HASTY_SEG_VS but
  for engines of order ≥ 2 only; unset: follow HASTY_SEG_VS). API: `set_ra`, `ra_active`, `ra_candidates`,
  `ra_tune_times`, `ra_tune_cfg`, `RaCfg` (exported).
- opbench: `--new-compare` (forward / adjoint / gradient with vs-vt and ra each off / on: four rows, autotune's best
  shapes), the autotune's adjoint kernel times (tile vs ra); `--order 3` now builds `points_hi` tables (u, H' zero:
  the full order-3 arithmetic, order-2 values) so the reconstruction's order is measurable.

### 16.3 Tests (RTX 4090; rustlib/crates/hasty-nufft/tests/seg_ra_vt.rs; logs research/experiments/integrate_ra_vt/)

All pass (test_*.log; vt_static_order2 after lowering two shape-count assertions to the number of valid shapes,
test_vt_static_order2b.log).
- ra vs the variant-0 tile adjoint, orders 0–2 static (10 × 10 dense with split lists and sparse without, 12 coils in
  batches of 4 × 3 segments at W = 8, 20 × 1) and 0–3 knot (1-D 8 × 4 W = 8, 1-D 6 coils in batches of 3 × 5 segments
  with split lists, 2-D 4 × 3), every candidate shape plus five extra (incl. odd CB, several block rows), each under
  `zero_tiles` true and false. Every ra shape gives the same result (same per-point summation order), and on knot
  engines its distance to variant 0 equals the variant-1 tile adjoint's distance to variant 0 to two digits in all 12
  cases (ra uses variant 1's z = y a β arithmetic): 3.6e-6 – 6e-4 relative in the image. These numbers are large only
  where the image is ill-conditioned: moving the input data by one ulp per element changes the variant-0 adjoint by
  1/4–1/5 of that (W = 10, random data: the deapodisation amplifies grid rounding, 1.1e-6 – 1.9e-4). Bound: max(3e-5,
  6 × one-ulp floor, 3 × kv1-vs-kv0). Adjointness against the tile forward 1e-8 – 1e-6, at or below the tile pair's.
- vt vs the tile forward, order 2 static (10 × 10, 12(4) × 3 at W = 8, 20 × 1; geometries 4×4×16×48, 2×4×8×32,
  4×2×8×48) and orders 2 / 3 knot 1-D / 2-D (incl. 10 coils in batches of 5 × 10 segments): 1.2e-6 – 1.0e-5, equal to
  the tile kernels' own floor (two tile shapes: 1.1e-6 – 9.7e-6; one-ulp image: 1.3e-6 – 1.1e-5). The order-2 / 3 test
  data are strong (|d| ≲ 0.1, hc ≈ 0.05–0.1 in kernel units): at order 3 the tile forward with vs without the degree-3/4
  data differs by 0.93–0.99, so the hc terms dominate what is compared. Adjointness vs the tile adjoint and vs ra (the
  new pair) 9e-9 – 2.2e-7, ≤ the tile pair's.
- Fused pad / crop and L2 slabs on / off (fuse 1 / 0 × slab auto / 0 / 2; static o2 and knot-2-D o3): vt 7e-6 / 1e-5,
  ra 4.3e-5 / 4.5e-5 vs the tile pair in every combination, pair adjointness 5e-8 – 1.4e-7.
- Grouped adjoint (static, two members: store then add) and `adjoint_source` (static and knot) through ra, ZERO tiles
  and full: 4.5e-6 / 3.7e-6 (tile vs ra).
- Autotune: keeps ra / vt iff faster (static 10 × 10 o2, 30 k samples: adjoint tile 3.32 → ra 1.74 ms, forward tile
  2.03 → vt 1.40 ms; knot 1-D o3: 2.05 → 1.43, 1.90 → 1.16 ms).
- compute-sanitizer memcheck (HASTY_NVRTC_LINEINFO=1, 3000 samples; static o0 / o2, knot 2-D o3, knot 1-D o1; every ra
  shape under ZERO tiles and full, every vt shape): 0 errors without and with split lists (HASTY_SEG_MAXITEM=64)
  (san_ra_vt.log, san_ra_vt_split.log).
- Regression (frozen binaries, defaults incl. ra / vt in the autotune): seg_vs, seg_knot_kv, seg_knot_hi, seg_knot,
  seg_knot2d, seg_tile_block, seg_stream, segnufft, seg_fft_traffic, seg_pruned, seg_multi, seg_frames, seg_kernel,
  seg_b0_direct, seg_wf, seg_warp_gpu, seg_joint, seg_fat_union_gpu pass; seg_joint_chunks fails at its known
  Φ-CGEMM check (pre-existing, §14.2 / temporal_operator.md §23.6).

### 16.4 Operator timing (opbench `--new-compare`, flow_thorax 2 mm, GNL, 10 virtual coils, L 10, ms; ob_*.log)

| case | M | autotune kernels (one coil batch): adjoint tile → ra, forward tile → vt/vs | forward off → on | adjoint off → on | gradient off → on |
|---|---|---|---|---|---|
| knot o2 (cubic, 20 knots, cb 5, `--scan-min 0.44`) | 1.33 M | 15.11 → 11.99 (1.26×); 13.83 → 14.28: vt not kept | 1203.6 → 1209.0 | 1318.9 → 1197.9 | **2523.8 → 2408.4 (−4.6 %)** |
| knot o3 (same, `points_hi` tables) | 1.33 M | 17.84 → 13.02 (1.37×); 16.25 → 15.76: vt kept | 1249.8 → 1254.2 | 1372.9 → 1236.4 | **2627.5 → 2490.5 (−5.2 %)** |
| static o2 (`--frames 7`, one frame × encode set) | 4.32 M | 204.1 → 124.7 (1.64×); 139.1 → 105.5 (1.32×) | 323.4 → 258.5 | 450.9 → 296.0 | **772.3 → 558.1 (1.38×)** |
| static o2, small set (`--scan-min 0.44`) | 0.19 M | 13.97 → 9.02 (1.55×); 10.72 → 10.81 | 54.1 → 54.0 | 63.5 → 53.8 | 117.7 → 108.1 |
| 48 coils, seg-none (L 1, order 0, `--cb-cap 48 --frames 7`; forward = the §14 order-0 vs) | 4.32 M | 222.1 → 162.0 (1.37×); 264.9 → 44.1 (6.0×) | 285.3 → 65.0 | 251.9 → 184.4 | **544.0 → 251.9 (2.16×)** |

"on" = both new variants at the autotune's best shapes (rows "fwd new" / "adj new" in the logs separate them). Knot
o2 "both off" (2523.8) and static "both off" (772.3) reproduce §23's 2512 / 771 ms. ra was the fastest adjoint in
every case (NXG 4, CB 7, CHK 16 — 104 channels per block row — for 50 / 100 channels; NXG 4, CB 3 for 48).

### 16.5 Against the prototype

- **ra**: kernel gain 1.64× on the static 4.3 M set (prototype 4 M 10 × 10 o2: 1.59×), 1.26× / 1.37× per knot at
  orders 2 / 3 (prototype 1.59× / 1.80×). The knot engine launches 20 knots × 2 coil batches; per launch only ≈ 0.27 M
  samples (4 / 20 of the set) are in a knot's window and every one of the ≈ 72 × 72 × 35 tiles is visited (items with
  an empty window store zeros, as the tile kernel), so per-tile staging, the window compaction and the zero stores weigh
  more than in the prototype's one 4 M launch. The order-3 terms cost ra +9 % (prototype +5 %), the tile kernel +18 %.
- **vt**: kernel gain 1.32× on the static 4.3 M set at order 2 (prototype 1.43×), none on the knot path (0.97× o2,
  1.03× o3 at knot 0; operator unchanged). Measured from the table sizes: knot tables 21.8 B / sample = 4 B × 4 knots
  + 16 B × items / sample → ≈ 11 samples per item, static 4.4 B → ≈ 40 per item (prototype ≈ 48 = SMAX). vt loads its
  register window (2 (TY + 4) = 16 rows × 13 points per plane and warp) per item, so on the knot path, where a knot sees ≈ 1/5 of the
  samples, the window loads are spread over 4× fewer sample bodies. The 2-D knots of the reconstruction (respiratory ×
  cardiac) thin each knot further; vt will not help there in this form, and the autotune correctly declines it where
  it is slower (it decides on knot 0).
- The gradient gain on the knot path is bounded by the FFT side (≈ 50 % of the gradient after §23) and the forward,
  where vt does not win: −4.6 % / −5.2 %.

### 16.6 Memory

ra adds no tables. vt uses the vs tables, which are now also built by default at orders 2 / 3: measured 21.8 B / sample
on the knot engine (cubic, 4 knots per sample) and 4.4 B / sample static, counted in `SegPoints::table_bytes`. On the
velocity reconstruction's knot path, where vt does not win, `HASTY_SEG_VT=0` saves them.

### 16.7 Next

1. Knot path forward: items are ≈ 4× emptier than vt needs. Options: home tiles sized per knot (2 × 4 × 8: fewer
   window rows per item), or one launch over all knots of a coil batch with the knot as an extra item dimension (the
   per-knot slots are not resident together today, so this needs the knot-batched grids of the FFT side).
2. ra on the knot path: skip items whose knot window is empty in add / split mode already; in store mode they write
   zeros. A per-knot list of nonempty tiles plus a zero pass over the rest (or the FFT's k-support) would remove those
   blocks; the tile kernels pay the same.
3. 48 × 1: ra kernel 162 ms (NXG 4, CB 3: two block rows of 24 channels) against the prototype's 77.5 ms at the same
   4 × 4 × 4 tile (§15.5, 4 M, 200³ grid) — a 2× gap not yet explained by measurement. Candidates to check with ncu: the
   engine's item length (max_item 1024 vs the prototype's LMAX 2048: more split items, atomics), the 2 mm thorax grid
   (288 × 288 × 140, other tile occupancy than 200³), the per-item list copy; the prototype's best 48 × 1 shape was
   4 × 8 × 4 tiles (own lists, +22.5 u32 per sample).
4. Cluster GPUs (H100 / A100 / 5090): more shared memory → larger vt items and more ra blocks per SM; to be measured.

## 17. Knot-path layout (2026-11-19/20, ninth session)

Question: can the knot-path NUFFT stage each sample once and apply its spatial / concomitant weights to all ≤ 8 knot
grids the sample touches, instead of once per knot launch? Does that close the gap to the prototype rates of §13 / §15?
This is the path the velocity reconstruction runs (`recon_p2np --vmodel knots`: 2-D knots 6:1,16:3, order 3,
`points2d_hi`). Decision: which knot-path layout goes into the engine, and behind which switch.

Setup:
- Data: opbench, flow_thorax 2 mm (image 225 × 225 × 112, fine grid 288 × 288 × 140, 181 440 tiles of 4³), 10 virtual
  coils, L 10, coil batch 5 (50 grid channels, one 4.6 GB slot), one encode with all frames.
- Two sample densities:
  - `--scan-min 0.44`: 1.33 M samples per encode, the reduced scan used in §16;
  - `--scan-min 4.4`: 13.3 M per encode. The 10-min scan has 30 M.
- Knots: 1-D, 20 cubic knots; 2-D, `--knots2d 6:1,16:3` (new in opbench, see 17.4), the reconstruction's config.
- Power limit: the user lowered it from 450 W to 300 W at 00:02 during the session (logged every 30 s in
  research/experiments/knot_layout/power.log). Every number compared below comes from one opbench invocation at
  300 W. The 450 W numbers of §16 and of the first baseline runs (base_*.log) are not compared with them.
- Logs, scripts and `table.py`: research/experiments/knot_layout/.

### 17.1 What one knot launch sees (measured, `SegPoints::tile_stats`)

| | 1.33 M per encode | 13.3 M per encode |
|---|---|---|
| work items (tiles 181 440) | 184 233 | 573 218 |
| zero-count items (tiles with no sample at all) | 65 103 (35.9 % of tiles) | 62 588 (34.5 %) |
| split items (max_item 1024) | 4 013 | **494 275** |
| list entries per nonempty tile | 394 | 3 862 |
| 1-D 20 knots: entries per knot window (mean) | 9.17 M (79 per nonempty tile) | 91.8 M |
| 2-D 6 × 16: entries per knot window (min / mean / max) | 1.91 / 3.82 / 4.93 M | 21.5 / 38.2 / 47.7 M |
| 2-D: tiles with samples but an empty window (mean) | 18 527 | 3 185 |

Per knot launch, the ra adjoint (§16) spends about 1.1–1.4 ns per (sample, tile) list entry at 50 channels. That rate
was measured at 450 W: 13.06 ms for 9.17 M entries at 1.33 M per encode (this includes storing the 4.6 GB slot), and
99.4 ms for 91.8 M entries at 13.3 M. The prototype runs at 1.41 ns per entry with 112 channels per block row (§15.5,
4 M 10 × 10 o3: 192 ms). So per visit, the engine is already as fast as the prototype. A visit there simply carries
twice the channels.

The §16 statement "a tile holds about 11 samples against 48 in the prototype" is about the vt items (home tiles of one
knot's window) of the 0.44-min scan. At 13.3 M per encode the vt forward beats the tile forward by 1.33–1.38× per knot
(1-D 91.0 vs 121.1 ms, 2-D 29.3 vs 40.5 ms), close to the prototype's 1.43–1.59×.

### 17.2 Why "stage once, apply to all knots" was not built: counting and measurements

**Counting.** Write C for the grid channels of one launch's resident slots and K_j for the knots sample j touches (4 for
1-D cubic, 8 for 2-D linear × cubic). NC = coils × L is the total channel count. A block can hold the accumulators of
at most CPW channel × knot outputs per tile (registers: 56–112 at TT 4, §15).

The cost that dominates is the per-visit staging F: records, kernel values on the tile cells, the order-3 row weights.
ncu attributes 55–60 % of the ra time to staging and barriers (17.6).

- Knot-major (today): one knot per launch with C channels. A sample is visited K_j · NC / C times per tile (times
  ⌈C / CPW⌉ block rows).
- Knot groups of G knots with C' channels each: the resident memory is G · C' channel grids. A sample's knot support
  meets on average (K + G − 1) / G groups (1-D; 2-D: ((G_r + 1) / G_r)((G_c + 3) / G_c)). So it is visited
  (K + G − 1) / G · NC / C' times.
- At equal memory (G · C' = C) this is (K + G − 1) · NC / C ≥ K · NC / C, with equality only at G = 1. At equal
  accumulator capacity the bound is the same. The knot dimension has a support boundary (a sample touches 4 consecutive
  knots, not all of them); the channel dimension has none.
- So for a fixed grid budget the knot-major order with the largest coil batch is already the visit-minimal layout.
  Staging once for all 8 knots would need 8 resident knot grids per channel (17.3). What a knot group does save is the
  channel-proportional part of the staging (z = y_c a_l, built once for all knots of the group), about 5 % of ra (17.6).

**Measured: the coil batch is the lever.** 1-D, 13.3 M per encode, order 3, 300 W, one opbench run each:

| coil batch (channels, engine memory) | knot-0 adjoint kernel (best of tile / ra) | per channel | forward | adjoint | gradient |
|---|---|---|---|---|---|
| 2 (20 ch, 2.19 GiB) | 75.5 ms (tile) | 3.8 ms | 7270 | 8924 | 16 190 |
| 5 (50 ch, 4.78 GiB) | 110.2 ms (ra) | 2.2 ms | 4656 | 5481 | 10 158 |
| 10 (100 ch, 9.13 GiB, `--max-grid-gb 10`) | 167.3 ms (ra) | 1.7 ms | 4115 | 4407 | **8534** |

- The per-channel cost of a visit falls by 2.2× from 20 to 100 channels per visit.
- Knot grouping at the same 4.6 GB would mean coil batches of 2 with G = 2. Per visit that is the 20-channel cost times
  2 knots, ≈ the 50-channel cost, and the visits grow by (4 + 1) / 2 · 5 / (4 · 2) = 1.56×. Predicted ≈ 1.5× the
  knot-major kernel time.
- Doubling the slot memory buys −16 % of the gradient here with no new code. That is a planning question for
  recon_p2np's `knot_grid_budget` (today (free − work) / 3, clamped to 6 GiB).

### 17.3 Memory ledger (knot grids resident at once)

One channel grid is 92.9 MB at 2 mm (288 × 288 × 140 × 8 B) and 1.47 GB at 1 mm (torso 800 × 480 × 480, temporal_operator.md §22).

| layout | resident | 2 mm, 10 coils × L 10 | 1 mm torso, L 10 |
|---|---|---|---|
| knot-major, coil batch cb (today) | cb · L channel grids | cb 5: 4.6 GB | cb 1: 14.7 GB (one coil already above a 24 GB card's slot budget) |
| all K_j = 8 knots of a 2-D sample, cb coils | 8 · cb · L | cb 1: 7.4 GB; cb 5: 37 GB | 118 GB |
| knot groups G (degrade by window batching) | G · cb · L | G 2, cb 2: 3.7 GB | ≥ 29 GB |
| layout built here (17.4): `tcount` + column hulls | 4 B per tile + 8 B per tile column | 0.73 MB + 41 KB | 11.5 MB + 0.2 MB |

At 1 mm no knot grouping fits a 24–32 GB card. The knot-major order plus window batching degrades to the same thing as
today's cb = 1. The 1 mm run was not done (time; the slot of one coil alone is 14.7 GB).

### 17.4 What was built (new files; old paths unchanged and selectable)

- **Skip of empty tiles** (`segnufft/kl.rs`, kernel `seg_zero_empty` in `segnufft_kl.cu`):
  - The knot adjoint launches only the items with samples. Items are sorted heaviest first, so `Interval::nnz` is a
    prefix.
  - The pruned inverse FFT reads, per tile column, the wrapped hull of its nonempty tiles in the engine's support
    union. A zero-count tile is read only when it lies inside that hull. That happens, for example, when another set's
    samples touch it. `seg_zero_empty` zeroes exactly those tiles. It checks the hull table on the device, one block
    per zero-count item, so no per-set state is cached; streamed slots have id 0.
  - The §23 contract is kept: tiles with samples are stored, and split tiles are zeroed and added atomically.
- **Whole-tile kl adjoint** (`segnufft_kl_adj.cu`, `seg_adjoint_kl`). It is the ra adjoint generated from
  segnufft_ra.cu with the same arithmetic, block / lane mapping and output park, plus three changes:
  - one block per tile and knot over the tile's whole key-sorted list: later parts of a split tile exit, the first part
    reads the tile's total from `tcount` (`seg_tile_count`); so there are no split-tile atomics and no
    `seg_zero_split`;
  - the knot window is read in place: ≤ 2 (1-D) or ≤ 4 (2-D) runs by binary search, no LISTMAX;
  - htab is held in shared memory and u is prefetched with the records.
  - Tunables are as ra (`RaCfg`).
- **Switch and autotune.**
  - `HASTY_SEG_KL = 0 | 1 | auto` (default auto). `HASTY_SEG_KL_CFG = nxg,cb,chk`. `HASTY_SEG_KL_LOG = 1`.
  - `HASTY_SEG_KL_NOZERO = 1` is a debug control only; it gives wrong results.
  - After the ra / vt steps the autotune times three paths on a representative knot, split-tile zeroing included:
    the engine's adjoint over all items, the same with the skip, and every kl shape. It keeps the fastest.
  - The representative knot is knot 0 for 1-D. For 2-D it is the first cardiac knot of the middle respiratory knot.
    Knot 0 of a 2-D engine sees only one respiratory interval; on 2 mm 13.3 M it chose the kl kernel, which is slower
    over all knots (kl3 runs).
  - API: `set_kl`, `set_kl_kernel`, `kl_active`, `kl_kernel`, `kl_candidates`, `kl_tune_times`, `kl_tune_cfg`,
    `SegPoints::tile_stats`.
- **Dispatch.** The edits in segnufft.rs are the dispatch in `adjoint_knot`, the autotune hook, `Interval::nnz`, the
  state field and the module line. Only `adjoint_knot` uses the layout; `adjoint_source` and the group / multi / fat
  paths keep theirs.
- **opbench.**
  - `--knots2d KR:RDEG,MC:CDEG` builds the 2-D engine with `points2d(_hi)`. A synthetic breathing amplitude places the
    respiratory knots at its quantiles; this also fixes the image size for 2-D, which needs KR × MC knot images.
  - `--kl-compare` prints three rows: previous path, skip, and the autotune's kl shape.
  - Also printed: tile-list statistics and the autotune's layout times.

### 17.5 Tests (tests/seg_kl.rs; logs t5_*.log; RTX 4090)

Samples are drawn in a ball of radius 0.62 π, so about half the tiles are zero-count, as in the real grid. Order-2/3
data are strong. All tests pass.

- **Layout on vs off, orders 0–3.** Cases: 1-D 8 × 4 W 8; 1-D 6 coils in batches of 3 × 5 segments (odd 15-channel
  blocks, c0 > 0); 2-D 4 × 3. Both the tile adjoint and ra (candidate plus extra shapes), each under ZERO tiles and
  full. On vs off differ by 0–3.5e-5, within 4 × the path's own run-to-run spread. Adjointness equals the off path.
- **Split lists** (`HASTY_SEG_MAXITEM = 64`): the same comparisons, passed.
- **kl kernel vs ra of the same shape:**
  - unsplit lists: bitwise equal (0.0) at every shape, orders 0–3, 1-D / 2-D (`kl_unsplit_equals_ra`);
  - split lists against ra on unsplit lists of the same samples: within the build-to-build floor
    (`kl_split_equals_unsplit_ra`). Two engines built from identical data already differ by ≈ 2.4e-4 in the W 10
    adjoint image; host-side builds are not bitwise reproducible.
  - In-engine, split lists: kl vs ra 6–7e-4, bounded by 10 × the one-ulp-data floor (1.5e-4). Only the summation
    grouping differs.
- **Fused pad / crop and L2 slabs** on / off (five combinations, 1-D o2 and 2-D o3): layout ≤ 1.1e-6, kl kernel
  ≤ 4.8e-5, adjointness ≤ 3.4e-7.
- **Support union** (a half-ball set inside a full-ball set's support): on vs off ≤ 4.5e-6.
  - Control with `HASTY_SEG_KL_NOZERO = 1`: 1.1e17 / NaN under ZERO tiles. So the test sees the stale cells that
    `seg_zero_empty` removes.
- **Autotune** keeps a layout iff it is fastest.
- **compute-sanitizer memcheck:** 0 errors. Coverage: tile, ra and every kl shape; ZERO tiles / full; with and without
  split lists; the union case.
- **Regression** (frozen binaries, reg_*.log; defaults with the layout in auto). Pass: seg_vs, seg_knot_kv,
  seg_knot_hi, seg_knot, seg_knot2d, seg_tile_block, seg_stream, segnufft, seg_fft_traffic, seg_pruned, seg_multi,
  seg_frames, seg_kernel, seg_b0_direct, seg_wf, seg_warp_gpu, seg_joint, seg_fat_union_gpu.
  - seg_joint_chunks fails at its known Φ-CGEMM check (pre-existing, §16.3).
  - seg_ra_vt `ra_static_orders` fails at its bound. The case is static o1, nxg 1, cb 4: 1.8e-4 vs 1.7e-4, then
    2.3e-4 vs 2.2e-4 on a rerun; the bound is 6 × a one-ulp floor that itself varies run to run. The previous
    session's frozen binary, built without this session's code, fails the same check now (reg3_*.log: 1.8e-4 vs
    1.5e-4). So it is a pre-existing marginal bound on the static path, which the layout does not touch.

### 17.6 Operator timing (opbench `--kl-compare`, 300 W, flow_thorax 2 mm, 10 coils, L 10, coil batch 5; ms)

Rows within one case come from one run. "Previous" = the autotune's ra / vt kernels over all items: the §16 path.

| case | M per encode | adjoint previous → skip → kl kernel | gradient previous → skip | autotune |
|---|---|---|---|---|
| **2-D 6:1,16:3 o3 (velocity recon)** | 13.3 M | 13 865 → **13 530** → 13 698 | **26 480 → 26 146 (−1.3 %)** | skip |
| **2-D 6:1,16:3 o3** | 1.33 M | 5265 → **4915** → 4925 | **10 268 → 9921 (−3.4 %)** | kl kernel (= skip) |
| 2-D 6:1,16:3 o2 | 1.33 M | 5189 → 4834 → **4734** | 10 086 → 9731 / 9632 (−4.5 % with the kl kernel) | kl kernel |
| 1-D 20 cubic o2 | 1.33 M | 1275 → 1203 → **1166** | 2574 → 2501 / 2467 (−4.2 % with the kl kernel) | kl kernel |
| 1-D 20 cubic o3 | 13.3 M | 5481 → **5414** → 5758 | 10 158 → 10 088 (−0.7 %) | skip |
| 1-D 20 cubic o3 | 1.33 M | 1333 → **1260** → 1265 | 2678 → 2608 (−2.6 %) | kl kernel (= skip) |

Kernel-only numbers (one coil batch, the representative knot, split-tile zeroing included):

| case | all items | skip | kl kernel |
|---|---|---|---|
| 2-D o3, 13.3 M | 59.8 | 58.5 | 60.5 |
| 2-D o3, 1.33 M | 8.56 | 7.34 | 6.80 |
| 1-D o3, 13.3 M | 117.5 | 113.6 | 118.3 |
| 1-D o3, 1.33 M | 13.65 | 12.95 | 11.96 |

- The forward is unchanged; vt is kept by the autotune in all cases.
- The FFTs: 2-D 13.3 M, 2970 / 3181 ms per call (192 transforms each way), 24 % of the gradient. 2-D 1.33 M: 2955 /
  3195 ms, 60 %.
- The skip removes the zero-count tiles' blocks and their 1.7 GB of zero stores per launch. That is worth −7 % of the
  adjoint at 1.33 M and −2.4 % at 13.3 M.
- The whole-tile kl kernel removes the split atomics and zeroing (8 ms of 117 ms at 13.3 M, 1-D). It loses that and
  more, so it only ties at 1.33 M. The likely causes are the long single-block tiles (center tiles up to tens of
  thousands of entries) and the in-place window reads. These were not separated by measurement.

### 17.7 Limiter of ra on a knot launch, and the 48 × 1 gap (Nsight Compute, targeted sections, line info)

**ra, one 1-D knot launch, order 3** (3 mm, coil batch 10 → two block rows of 56 channels, 830 entries per tile; ncu/ra_knot*).

- 12 warps per SM; issue 0.52 per scheduler (latency-bound).
- Stall samples by source region (ncu/ra_knot.regions.txt):

| region | share of stall samples |
|---|---|
| MAC | 40.1 % |
| row weights (order 3) | 9.9 % |
| chunk-head barrier | 8.1 % |
| hc staging (52 % of it long scoreboard on the uw / htab global loads) | 7.4 % |
| stage barrier | 7.0 % |
| inlined kernel-window helpers | 6.5 % |
| z build | 5.1 % |
| window scan (1-D `__syncthreads_count` over the whole list) | 4.7 % |
| w_l | 3.9 % |
| output | 3.8 % |

- Staging and barriers are ≈ 55 % and do not depend on the knot. They are the same per visit as in the prototype.

**48 coils × 1 segment, static, order 0** (4.3 M; the §16.7 item 3 question: engine 162 ms vs prototype 77.5 ms).

- The prototype's best shape (nxg 2, cb 3: one block row of 48 channels) spills in the engine build: 128 registers,
  140 M local-memory requests, issue 0.27 per scheduler, 198 ms under ncu.
- The autotune therefore takes nxg 4, cb 3: no spills, but two block rows of 24 channels, so every sample is staged
  twice. That is 180 ms under ncu.
- The launch has 279 907 items for 181 440 tiles (max_item 1024: split lists, atomics and zeroing on every dense tile).
  The prototype used LMAX 2048 on its own 200³ grid.
- Not resolved: why the engine source spills at (2, 3) where the prototype's does not. The static path is outside this
  session's knot-path scope.

### 17.8 Next

1. **Coil batch per gradient pass** (largest measured lever): 4.8 → 9.1 GiB of slot memory = −16 % of the dense 1-D
   gradient. Make recon_p2np's knot slot budget a ledger entry instead of (free − work) / 3 ≤ 6 GiB. On 32 / 80 GB cards
   it allows cb ≥ 10.
2. **ra staging** (≈ 55 % of a knot launch at order 3; 17.7): htab in shared memory and uw prefetched (in the kl
   kernel already; isolate its gain); 1-D windows by binary search; z build over all threads; fewer barriers per
   chunk. Measure each against ra, as a new variant.
3. **Autotune of ra / vt on 2-D engines** times knot 0 (an edge respiratory knot with half the window). Use the
   representative knot as the kl step now does.
4. **FFT side** on the 2-D low-density case (60 % of the gradient): §23 items 1–3.
5. **The 48 × 1 spill** (17.7), and the 1 mm check (not run).

## 18. One-frame gradient at realistic grid sizes (2026-11-20, tenth session)

Question: what does one gradient evaluation — forward plus adjoint of the segmented NUFFT, FFTs included — cost for
one frame at realistic image matrices (192³, 256³, 320³, 512³, 640 × 384 × 320) and realistic per-frame sample
counts, where does the time go, and what is the fastest channel batch per matrix within the card's memory? Decision:
which batch the engine / planner should choose, and which paths to use for a one-frame (one knot per GPU) gradient.

### 18.1 Setup

- **What one measurement is.** opbench, static engine (one image), one (frame, encode) sample set: the median set of
  the scan's sets for the given frame count. Times are CUDA-event medians of 3 calls after a warm-up, and cover the
  whole call: pad (fused into the first FFT pass), forward FFTs, gather, scatter (with split-tile zeroing), inverse
  FFTs and crop (fused into the last pass), for all coils. They do not cover table building, segment building or the
  autotune. "Gradient" is one forward followed by one adjoint (`normal` in opbench). A frame of the scan has 7
  encodes, each with its own segment diagonals and therefore its own FFTs, so **one frame's gradient = 7 × the
  numbers below** (the sets are independent; same work per encode).
- **Engine.** σ = 1.25, W = 10 (ES window), k-support-pruned grid FFTs, fused pad / crop, L2 slabs (auto), zeroing of
  split tiles only (all §23 defaults), the autotune's kernels. Fine grid per axis: the smallest 2·3·5·7-smooth even
  size ≥ 1.25 N (192 → 240, 256 → 320, 320 → 400, 512 → 640, 640 × 384 × 320 → 800 × 480 × 400).
- **Coils and segments.** 10 virtual coils × L 10 segments (100 channels) as the baseline; 16 × 16 (256 channels)
  as a second point. Orders 0 and 3 (order 3 = `points_hi` tables: degree-3/4 concomitant terms; the values are those
  of the order-2 operator, the arithmetic is the order-3 kernels').
- **Sample density.** From the user's 10-minute scan (data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/
  meta.json: 53 144 readouts × 3990 ADC samples = 2.12 × 10⁸ samples, 7 encodes → 30.3 M per encode). Per (frame,
  encode): 20 frames → 1.52 M (the baseline), 48 → 0.63 M, 96 → 0.32 M. The sweep ×0.3 / ×1 / ×3 uses 67 / 20 / 7
  frames: 0.45 / 1.52 / 4.32 M (median sets). Per frame (all 7 encodes): 3.2 / 10.6 / 30.3 M.
- **How samples per frame scale with the matrix: they do not.** Assumption, stated plainly: the scan time (10 min),
  the TR (11.29 ms), the ADC window (7.98 ms) and the dwell (2 µs) stay fixed, so the number of readouts and samples
  per readout stay fixed, and so does the count per frame. k-max grows with the matrix: the readouts are the flow
  design's (opbench `--seq flow`, the same yarn readouts and encode order as the 10-min scan), and their coordinates
  k·Δ (radians; |k·Δ| ≤ π at the design's k-max of 250 /m) are kept while the matrix N changes. That is the same
  readout shape stretched in k so that its k-max sits at the new matrix's Nyquist edge. A larger matrix is then more
  undersampled; the per-frame cost does not grow from samples, only from the grid. (At the largest matrices this
  stretched readout needs ≈ 2.3× the design's gradient amplitude; a real design would need longer or more
  interleaved readouts. The point here is the count and k-space distribution of the samples, which set the cost.)
- **`--matrix NX,NY,NZ`** (new in opbench): the engine matrix is set directly; physics and tables come from the design
  grid as before, coil maps and segment phase are sampled onto the matrix (values do not change the cost).
- **Card.** RTX 4090 (24 GB, 72 MB L2, DRAM peak 1008 GB/s), power limit **300 W** for every run (logged per case and
  every 30 s in research/experiments/grid_sizes/power.log). All numbers compared below come from that limit.
- Files: research/experiments/grid_sizes/ — `a/` before, `b/` after, `p/` nsys profiles, `c/` planner, `fft/` FFT pass
  bench, `tests/`, `table.py` / `tables_md.py` / `fftpass.py` / `trace.py`, the case lists and runners.

### 18.2 Memory per matrix (device, one encode; formulas of research/temporal_operator.md §22.4)

| matrix → grid | one channel grid | coil maps 10 / 16 | diagonals L 10 / 16 | x and Aᴴy | all 100 channels | one coil (10 segments) |
|---|---|---|---|---|---|---|
| 192³ → 240³ | 0.11 GB | 0.57 / 0.91 GB | 0.57 / 0.91 GB | 0.11 GB | 11.1 GB | 1.1 GB |
| 256³ → 320³ | 0.26 GB | 1.34 / 2.15 | 1.34 / 2.15 | 0.27 | 26.2 | 2.6 |
| 320³ → 400³ | 0.51 GB | 2.62 / 4.19 | 2.62 / 4.19 | 0.52 | 51.2 | 5.1 |
| 512³ → 640³ | 2.10 GB | 10.7 / 17.2 | 10.7 / 17.2 | 2.15 | 210 | 21.0 |
| 640×384×320 → 800×480×400 | 1.23 GB | 6.3 / 10.1 | 6.3 / 10.1 | 1.26 | 123 | 12.3 |

Sample tables add ≈ 220 B per sample (0.33 GB at 1.5 M, 0.95 GB at 4.3 M). Consequences on a 24 GB card:
- 192³: all 100 channels fit in one batch. 256³: 5 coils (50 channels). 320³: 3 coils at most (30 channels).
- **512³ with 10 coils does not fit at all**: maps + diagonals + x + Aᴴy = 23.6 GB before any grid. With one coil's
  map resident (the per-coil pass; maps streamed per coil, §22.5 option 4) the slot has ≈ 4–8 GB left: **less than
  one coil** (21 GB). The engine could not batch a fraction of a coil (a batch was whole coils of all L segments; one
  coil was always allocated however large) — added here (18.5).
- 640 × 384 × 320 with 10 coils' maps: 13.8 GB fixed, ≈ 8 GB for the slot → also less than a coil (12.3 GB). With one
  coil's map resident a whole coil fits.
- 16 segments at 512³: the diagonals alone are 17.2 GB; not run.

### 18.3 Summary: one-frame gradient per matrix (after tuning; 300 W)

One (frame, encode) set of 1.52 M samples (20 frames), forward + adjoint, all coils. "Per frame" = × 7 encodes.

| matrix → grid | coils × segments, resident | best channel batch | order 0 | order 3 before → after | order 3 per frame (7 encodes) |
|---|---|---|---|---|---|
| 192³ → 240³ | 10 × 10, all maps | 10 coils × 10 = 100 channels (one batch) | 198 ms | 411 → **295 ms** (−28 %) | 2.1 s |
| 256³ → 320³ | 10 × 10, all maps | 5 coils × 10 = 50 channels (2 batches) | 349 ms | 737 → **481 ms** (−35 %) | 3.4 s |
| 320³ → 400³ | 10 × 10, all maps | 2 coils × 10 = 20 channels (5 batches) | 675 ms | 1105 → **924 ms** (−16 %) | 6.5 s |
| 512³ → 640³ | 10 × 10, one coil's map at a time (per-coil pass × 10) | 1 coil × 2 segments = 2 channels (5 groups per coil) | 10 × 344 = 3.4 s | did not fit → **10 × 608 = 6.1 s** | 43 s |
| 640×384×320 → 800×480×400 | 10 × 10, one coil's map at a time | 1 coil × 10 = 10 channels | 10 × 151 = 1.5 s | 10 × 280 → **10 × 208 = 2.1 s** (−26 %) | 14.6 s |
| 640×384×320 | 10 × 10, all 10 maps resident | 1 coil × 5 segments (2 groups per coil) | — | did not fit → **2.7 s** | 19.1 s |

"Before" is the static engine (order 3 on the multi-image kernels) at its best batch; "after" is the one-bin knot
engine (18.5) at the planner's batch. At the same batch order 0 changes by −7 … +0 % (the static order-0 path already has vt / ra);
all the gain is at order 3. The 512³ and 640 × 384 × 320 "one coil's map at a time" rows assume the coil maps are
streamed per coil pass (§22.5 option 4: ≈ 1 GB per coil at ≈ 21 GB/s, 50 ms, hidden by double buffering; not built);
on the 24 GB card only the run with all 10 maps resident at 640 × 384 × 320 is the real configuration. 512³ with all
maps resident does not fit (18.2).


### 18.4 The grid FFTs run at DRAM bandwidth (fftpass_bench under nsys; fft/, `fftpass.py`)

The transform alone (block-pruned, no k-support pruning, forward and inverse, one channel batch), per pass: time per
transform (mean of forward and inverse) and the bytes of the pass model (x: read the image block, write n0·N1·N2; y:
read that, write n0·n1·N2; z: read that, write the grid; with L2 slabs the x → y intermediate is not counted).

| grid (block) | channels | slabs | x pass ms (GB/s, % peak) | y pass | z pass | transform ms |
|---|---|---|---|---|---|---|
| 240×240×240 (192×192×192) | 100 | 0 | 13.96 (913, 91 %) | 18.02 (884, 88 %) | 23.34 (853, 85 %) | 55.3 |
| 240×240×240 (192×192×192) | 100 | 1 | 9.44 (600, 60 %) | 12.59 (703, 70 %) | 23.31 (854, 85 %) | 45.3 |
| 240×240×240 (192×192×192) | 20 | 0 | 2.82 (903, 90 %) | 3.49 (914, 91 %) | 4.65 (856, 85 %) | 11.0 |
| 240×240×240 (192×192×192) | 20 | 1 | 2.73 (415, 41 %) | 3.15 (562, 56 %) | 4.67 (853, 85 %) | 10.5 |
| 320×320×320 (256×256×256) | 50 | 0 | 16.58 (911, 90 %) | 21.68 (871, 86 %) | 27.19 (868, 86 %) | 65.4 |
| 320×320×320 (256×256×256) | 50 | 1 | 11.32 (593, 59 %) | 13.71 (765, 76 %) | 27.29 (865, 86 %) | 52.3 |
| 320×320×320 (256×256×256) | 20 | 1 | 5.07 (530, 53 %) | 6.19 (678, 67 %) | 10.91 (865, 86 %) | 22.2 |
| 400×400×400 (320×320×320) | 30 | 0 | 19.32 (916, 91 %) | 24.79 (892, 89 %) | 31.16 (887, 88 %) | 75.3 |
| 400×400×400 (320×320×320) | 30 | 1 | 16.85 (467, 46 %) | 20.67 (594, 59 %) | 31.12 (889, 88 %) | 68.6 |
| 400×400×400 (320×320×320) | 10 | 1 | 6.41 (409, 41 %) | 7.94 (516, 51 %) | 10.42 (884, 88 %) | 24.8 |
| 640×640×640 (512×512×512) | 2 | 0 | 5.29 (914, 91 %) | 6.73 (898, 89 %) | 8.89 (849, 84 %) | 20.9 |
| 640×640×640 (512×512×512) | 4 | 0 | 10.56 (915, 91 %) | 13.66 (884, 88 %) | 17.85 (846, 84 %) | 42.1 |
| 640×640×640 (512×512×512) | 4 | 2 | 5.73 (750, 74 %) | 11.84 (567, 56 %) | 17.85 (846, 84 %) | 35.4 |
| 800×480×400 (640×384×320) | 10 | 0 | 15.49 (914, 91 %) | 20.13 (879, 87 %) | 25.13 (880, 87 %) | 60.7 |
| 800×480×400 (640×384×320) | 10 | 1 | 13.23 (476, 47 %) | 14.22 (691, 69 %) | 25.14 (880, 87 %) | 52.6 |
| 800×480×400 (640×384×320) | 5 | 1 | 7.47 (421, 42 %) | 7.99 (615, 61 %) | 12.49 (885, 88 %) | 28.0 |

- **Whole-grid passes run at 84–91 % of the 1008 GB/s peak at every grid size and channel count**, including 2
  channels at 640³ (the x pass reads 16 B per line there and still reaches 91 %). There is no launch-shape or
  coalescing loss at large grids, although lengths ≥ 400 run 8 lines per block (4 at 800) for shared-memory reasons.
- **L2 slabs** save 1–23 % of the transform. Their x and y passes run at 41–76 % of the DRAM-equivalent rate: a slab
  pass is limited by the L2, not by DRAM. Slabs of a few small planes (20 channels at 240³: 9 MB per plane) gain
  almost nothing (11.0 → 10.5 ms); large planes gain most (100 channels at 240³: 55.3 → 45.3 ms).
- **Channel sub-slabs** (temporal_operator.md §23.6 item 2: slabs of (plane, channel tile) when a plane × channels
  exceeds the L2) are not worth building on this card at these matrices: memory already limits the batch to the
  channel count where one plane fits (18.2), and smaller planes measured no better per channel (400³: 10 channels 24.8
  ms per transform vs 30 channels 68.6 ms, i.e. 2.48 vs 2.29 ms per channel). Not changed.
- In the engine (opbench's FFT column, the k-support-pruned transforms the engine runs) the same model gives 64–80 %
  (18.8): those transforms carry the k-support line tables and the slabs.

So the FFT side is at its byte floor; it only gets cheaper with fewer bytes (fewer transformed channels: 18.5; a
smaller storage format: the block-floating-point work that another agent has started).

### 18.5 What was built (nothing removed; old paths unchanged and default)

- **Static image through the knot kernels (one-bin knot engine).** A knot-major engine with one frame of degree 0
  (one bin, knot support 1) is now allowed: `SegNufft::new_temporal(dev, dims, res, nc, SegNufftOpts { frames: 1,
  bspline: 0, .. }, true)`, or env **`HASTY_SEG_STATIC_KNOT=1`** (default off), which builds every static engine that
  way. It is the static operator (weight 1 for every sample) run by `forward_knot` / `adjoint_knot`: the vt / vs
  forward and ra adjoint at every order including 3, the kl skip of empty tiles, the fused pad / crop, and zeroing of
  split tiles only — instead of the static path, whose order 3 runs the multi-image kernels (no vt / ra / kl, separate
  pad / crop, full-slot memset when tiles are split). The knot kernels already supported KS = 1 (bins); the change is
  one condition in `new_impl` (plus the env switch). The multi-image (K > 1), grouped, fat and warp paths are not
  available on a knot-major engine (they return an error), which is why the switch is off by default. opbench
  `--static-knot`.
- **Segment batches inside a coil** (new `hasty-nufft/src/segsplit.rs`): `SegSplit` runs an engine built with `lb`
  segments once per group of segments; the groups sum exactly (A = Σ_g A_g; forward calls after the first add into y
  through the new `SegNufft::forward_accumulate` flag, adjoint calls through the existing `adjoint_accumulate`). A slot
  then holds cb × lb channels, so a batch can be a fraction of one coil. Works on static, knot-major and one-bin
  engines (the segment diagonals of every group stay resident, as before). Cost: one pass over the samples per (coil
  batch, group); the image side is unchanged. When lb does not divide L the last group is zero-padded. opbench
  `--seg-batch LB`.
- **Batch planner** (`plan_batch`, `BatchPlan`, `SegSplit::planned(.., budget, ..)`): for a slot budget it takes
  whole coils of all L segments while one coil fits, else one coil in segment groups; among the batches that fit, the
  fewest **transformed** channels, then the fewest passes. Every coil batch transforms the full slot, so a ragged last
  batch (10 coils in batches of 3: 3, 3, 3, 1 → 120 transformed channels) or a padded segment group costs FFT time for
  nothing; measured at 320³ (18.6) this rule picks 2 coils per batch over 3. opbench `--plan-budget-gb G`.
- **segnufft.rs edits** (dispatch only): the `forward_accumulate` field and its test in `init_rows`; the one-bin
  condition and `HASTY_SEG_STATIC_KNOT` in `new_impl`; doc lines.
- **recon_p2np `--grid-budget legacy|auto|SIZE`**: the knot engine's slot budget. Default `legacy` (unchanged: (free −
  work) / 3, at most 6 GiB); `auto` = (free − work − 2 GiB) / 1.5 with no 6 GiB cap (the knot ledger still refuses a
  plan that does not fit, before the solver); or an explicit size. Not made the default: the recon allocates more
  after the engine and only the ledger knows how much; a planned slot entry in the ledger is the proper fix
  (memory_ledger.md §3).
- **opbench**: `--matrix`, `--cb-list` (one engine per coil batch; the grid budget is set to exactly that batch so the
  order-3 multi path cannot take a larger one with extra buffers), `--jsonl` (one record per measurement: times,
  breakdown, memory, the autotune's variants), `--profile` (one gradient between cuProfilerStart / Stop for nsys),
  `--seg-batch`, `--static-knot`, `--plan-budget-gb`.
- **fftpass_bench** (new bin): one PrunedChanFft3 configuration, forward / inverse time and the pass byte model.
- No edits to hasty-fft.
### 18.6 Before / after per case

| matrix (grid) | coils × L | order | samples | before: batch, gradient (fwd / adj) | after: batch, gradient (fwd / adj) | after: interp fwd / adj, FFT fwd / inv, rest fwd / inv | change |
|---|---|---|---|---|---|---|---|
| 192×192×192 (240×240×240) | 10 × 10 | 0 | 0.45 M | 10×10 = 100 ch: **138** (68 / 66) | 10×10 = 100 ch: **132** (69 / 64) | 22 / 20, 38 / 39, 8 / 5 | -4 % |
| 192×192×192 (240×240×240) | 10 × 10 | 0 | 1.52 M | 10×10 = 100 ch: **197** (93 / 102) | 10×10 = 100 ch: **198** (89 / 102) | 42 / 53, 39 / 39, 8 / 10 | +0 % |
| 192×192×192 (240×240×240) | 10 × 10 | 0 | 4.32 M | 10×10 = 100 ch: **357** (147 / 207) | 10×10 = 100 ch: **351** (149 / 198) | 91 / 147, 39 / 40, 19 / 12 | -2 % |
| 192×192×192 (240×240×240) | 10 × 10 | 3 | 0.45 M | 10×10 = 100 ch: **209** (99 / 112) | 10×10 = 100 ch: **164** (88 / 74) | 39 / 31, 38 / 39, 12 / 4 | -22 % |
| 192×192×192 (240×240×240) | 10 × 10 | 3 | 1.52 M | 10×10 = 100 ch: **411** (197 / 217) | 10×10 = 100 ch: **295** (152 / 137) | 89 / 89, 39 / 40, 24 / 8 | -28 % |
| 192×192×192 (240×240×240) | 10 × 10 | 3 | 4.32 M | 10×10 = 100 ch: **948** (456 / 498) | 10×10 = 100 ch: **612** (298 / 310) | 229 / 244, 39 / 40, 31 / 26 | -35 % |
| 192×192×192 (240×240×240) | 16 × 16 | 0 | 1.52 M | 8×16 = 128 ch: **483** (220 / 241) | — |  |  |
| 192×192×192 (240×240×240) | 16 × 16 | 3 | 1.52 M | 8×16 = 128 ch: **860** (414 / 443) | 8×16 = 128 ch: **710** (379 / 319) | 225 / 187, 101 / 103, 53 / 28 | -17 % |
| 256×256×256 (320×320×320) | 10 × 10 | 0 | 0.45 M | 5×10 = 50 ch: **271** (131 / 140) | 5×10 = 50 ch: **252** (125 / 128) | 30 / 28, 85 / 90, 10 / 9 | -7 % |
| 256×256×256 (320×320×320) | 10 × 10 | 0 | 1.52 M | 5×10 = 50 ch: **363** (172 / 182) | 5×10 = 50 ch: **349** (169 / 167) | 66 / 63, 85 / 90, 18 / 14 | -4 % |
| 256×256×256 (320×320×320) | 10 × 10 | 0 | 4.32 M | 5×10 = 50 ch: **554** (254 / 297) | 5×10 = 50 ch: **555** (255 / 297) | 135 / 167, 85 / 91, 34 / 38 | +0 % |
| 256×256×256 (320×320×320) | 10 × 10 | 3 | 0.45 M | 5×10 = 50 ch: **445** (198 / 242) | 5×10 = 50 ch: **302** (158 / 144) | 64 / 45, 84 / 90, 10 / 9 | -32 % |
| 256×256×256 (320×320×320) | 10 × 10 | 3 | 1.52 M | 5×10 = 50 ch: **737** (360 / 377) | 5×10 = 50 ch: **481** (250 / 231) | 144 / 124, 85 / 91, 21 / 16 | -35 % |
| 256×256×256 (320×320×320) | 10 × 10 | 3 | 4.32 M | 5×10 = 50 ch: **1523** (761 / 760) | 5×10 = 50 ch: **920** (452 / 459) | 322 / 333, 86 / 91, 43 / 36 | -40 % |
| 256×256×256 (320×320×320) | 16 × 16 | 3 | 1.52 M | 4×16 = 64 ch: **1348** (666 / 678) | 4×16 = 64 ch: **1109** (591 / 513) | 291 / 240, 213 / 223, 87 / 50 | -18 % |
| 320×320×320 (400×400×400) | 10 × 10 | 0 | 0.45 M | 3×10 = 30 ch: **644** (313 / 334) | 2×10 = 20 ch: **553** (276 / 277) | 52 / 52, 195 / 195, 29 / 30 | -14 % |
| 320×320×320 (400×400×400) | 10 × 10 | 0 | 1.52 M | 2×10 = 20 ch: **714** (348 / 364) | 2×10 = 20 ch: **675** (342 / 327) | 98 / 96, 196 / 197, 48 / 34 | -6 % |
| 320×320×320 (400×400×400) | 10 × 10 | 0 | 4.32 M | 3×10 = 30 ch: **1027** (478 / 561) | 2×10 = 20 ch: **936** (446 / 481) | 202 / 239, 197 / 198, 47 / 44 | -9 % |
| 320×320×320 (400×400×400) | 10 × 10 | 3 | 0.45 M | 3×10 = 30 ch: **919** (428 / 494) | 2×10 = 20 ch: **632** (319 / 314) | 94 / 87, 197 / 198, 29 / 29 | -31 % |
| 320×320×320 (400×400×400) | 10 × 10 | 3 | 1.52 M | 2×10 = 20 ch: **1105** (500 / 607) | 2×10 = 20 ch: **924** (465 / 464) | 226 / 227, 196 / 197, 43 / 40 | -16 % |
| 320×320×320 (400×400×400) | 10 × 10 | 3 | 4.32 M | 3×10 = 30 ch: **2019** (1015 / 1004) | 2×10 = 20 ch: **1715** (854 / 859) | 577 / 618, 198 / 200, 79 / 42 | -15 % |
| 320×320×320 (400×400×400) | 16 × 16 | 3 | 1.52 M | 1×16 = 16 ch: **3570** (1571 / 1992) | 1×16 = 16 ch: **2675** (1379 / 1291) | 768 / 666, 502 / 497, 109 / 129 | -25 % |
| 512×512×512 (640×640×640) | 1 × 10 | 0 | 1.52 M | — | 1×2 = 2 ch: **344** (159 / 185) | 68 / 83, 68 / 73, 22 / 29 |  |
| 512×512×512 (640×640×640) | 1 × 10 | 3 | 0.45 M | — | 1×2 = 2 ch: **395** (208 / 190) | 122 / 86, 69 / 74, 17 / 30 |  |
| 512×512×512 (640×640×640) | 1 × 10 | 3 | 1.52 M | — | 1×2 = 2 ch: **608** (306 / 300) | 218 / 195, 69 / 73, 20 / 32 |  |
| 512×512×512 (640×640×640) | 1 × 10 | 3 | 4.32 M | — | 1×2 = 2 ch: **1150** (563 / 583) | 460 / 473, 70 / 74, 33 / 36 |  |
| 640×384×320 (800×480×400) | 1 × 10 | 0 | 1.52 M | 1×10 = 10 ch: **160** (73 / 81) | 1×10 = 10 ch: **151** (72 / 76) | 17 / 21, 45 / 47, 10 / 8 | -6 % |
| 640×384×320 (800×480×400) | 1 × 10 | 3 | 0.45 M | 1×10 = 10 ch: **211** (88 / 122) | 1×10 = 10 ch: **153** (79 / 74) | 27 / 21, 45 / 46, 7 / 7 | -28 % |
| 640×384×320 (800×480×400) | 1 × 10 | 3 | 1.52 M | 1×10 = 10 ch: **280** (116 / 164) | 1×10 = 10 ch: **208** (105 / 104) | 51 / 48, 44 / 46, 10 / 10 | -26 % |
| 640×384×320 (800×480×400) | 1 × 10 | 3 | 4.32 M | 1×10 = 10 ch: **458** (187 / 271) | 1×10 = 10 ch: **350** (172 / 177) | 113 / 119, 45 / 47, 14 / 12 | -24 % |
| 640×384×320 (800×480×400) | 1 × 16 | 3 | 1.52 M | — | 1×8 = 8 ch: **337** (174 / 161) | 91 / 74, 69 / 70, 13 / 18 |  |
| 640×384×320 (800×480×400) | 10 × 10 | 3 | 1.52 M | — | 1×5 = 5 ch: **2723** (1415 / 1303) | 841 / 652, 446 / 468, 128 / 182 |  |

Reading:
- **Order 3 is the lever: −16 to −40 %.** The static order-3 path runs the multi-image kernels (`seg_forward_mt` /
  `seg_adjoint_mt`), which have none of the §12–§17 work (no vt / ra / kl), the separate pad / crop and, on split tile
  lists, a memset of the whole slot. The one-bin knot engine runs vt or the tile kernel forward (autotune's choice), ra
  or the kl kernel adjoint, the fused pad / crop and split-tile-only zeroing. The gain grows with samples per frame
  (192³: −22 / −28 / −35 % at 0.45 / 1.52 / 4.32 M).
- **The 320³ rows change two things**: the path and the batch (before 3 coils per batch at 0.45 / 4.32 M, after 2
  coils). At 1.52 M, where the baseline has both batches, the path alone gives −16 %.
- **16 × 16** (256 channels): −17 / −18 / −25 % at order 3.

### 18.7 Gradient vs channel batch

| matrix | coils × L | order | samples | path | channels per batch (coils × segments): gradient ms |
|---|---|---|---|---|---|
| 192×192×192 | 10 × 10 | 0 | 1.52 M | static | 10 (1×10): 338, 20 (2×10): 264, 50 (5×10): 224, 100 (10×10): 197 |
| 192×192×192 | 10 × 10 | 3 | 1.52 M | one-bin knot | 20 (2×10): 514, 50 (5×10): 346, 100 (10×10): 295 |
| 192×192×192 | 10 × 10 | 3 | 1.52 M | static | 10 (1×10): 1044, 20 (2×10): 570, 50 (5×10): 550, 100 (10×10): 411 |
| 192×192×192 | 16 × 16 | 3 | 1.52 M | static | 64 (4×16): 934, 128 (8×16): 860 |
| 256×256×256 | 10 × 10 | 0 | 1.52 M | static | 10 (1×10): 493, 20 (2×10): 406, 50 (5×10): 363 |
| 256×256×256 | 10 × 10 | 3 | 1.52 M | one-bin knot | 20 (2×10): 655, 50 (5×10): 481 |
| 256×256×256 | 10 × 10 | 3 | 1.52 M | static | 10 (1×10): 1281, 20 (2×10): 750, 50 (5×10): 737 |
| 256×256×256 | 16 × 16 | 3 | 1.52 M | static | 32 (2×16): 1725, 64 (4×16): 1348 |
| 320×320×320 | 10 × 10 | 0 | 1.52 M | static | 10 (1×10): 824, 20 (2×10): 714, 30 (3×10): 772 |
| 320×320×320 | 10 × 10 | 3 | 1.52 M | one-bin knot | 10 (1×10): 1322, 20 (2×10): 924, 30 (3×10): 995 |
| 320×320×320 | 10 × 10 | 3 | 1.52 M | static | 10 (1×10): 1729, 20 (2×10): 1105, 30 (3×10): 1200 |
| 512×512×512 | 1 × 10 | 3 | 1.52 M | one-bin knot | 1 (1×1): 1324, 2 (1×2): 608 |
| 640×384×320 | 1 × 10 | 3 | 1.52 M | one-bin knot | 5 (1×5): 277, 10 (1×10): 208 |

- **More channels per batch is faster as long as the batches are equal.** Each extra coil batch is one more pass over
  the samples (staging, kernel weights, order-3 row weights: §17.2), and the FFT work per channel is the same.
  192³ order 0: 10 → 100 channels = 338 → 197 ms; one-bin order 3: 20 → 100 channels = 514 → 295 ms.
- **A ragged last batch is slower**: at 320³ 3 coils per batch (batches 3, 3, 3, 1 = 120 transformed channels,
  because every batch transforms the full slot) is slower than 2 coils (5 batches of 2 = 100): 772 vs 714 ms (static
  order 0), 1200 vs 1105 (static order 3), 995 vs 924 (one-bin order 3). The planner now minimises transformed
  channels first, then passes (18.5); the old rule (`balanced_batch`: fewest passes, smallest equal batch) picked 3.
- **512³, one coil: 2 segments per batch (5 groups) 608 ms, 1 segment (10 groups) 1324 ms**: with one channel the
  engine falls back to the unpruned full transform (`GridFft::new` builds the pruned plan only for ≥ 2 channels) and
  the sample side runs 10 passes. The static (multi-kernel) path with the same 2 × 5 split: 791 ms.
- **The best batch per matrix on the 24 GB card**: 192³ 100 channels (10 coils), 256³ 50 (5 coils), 320³ 20 (2
  coils), 512³ 2 (one coil, 5 groups; one coil's map resident), 640 × 384 × 320 10 (one coil) with one map resident or
  5 (one coil, 2 groups) with all 10 maps resident. In every case it is the largest equal batch that fits, i.e. the
  speed optimum is the memory limit here; no measured case had a smaller batch faster.

### 18.8 Where the time goes (nsys, exactly one gradient, `--profile`; p/)

| case (one-bin knot unless static) | GPU time of the gradient ms | FFT passes (fused pad / crop included) | forward interpolation | adjoint interpolation | separate pad / crop | zeroing / memsets |
|---|---|---|---|---|---|---|
| 192³ order 0, 100 ch | 191 | 82 (43 %) | 45 (`seg_forward_vs`) | 64 (`seg_adjoint_ra`) | 0 | 0.3 |
| 192³ order 3, 100 ch | 276 | 82 (30 %) | 95 (`seg_forward_vs`) | 100 (`seg_adjoint_ra`) | 0 | 0.3 |
| 192³ order 3, 100 ch, static | 394 | 77 (19 %) | 142 (`seg_forward_mt`) | 148 (`seg_adjoint_mt`) | 16 | 12.1 |
| 256³ order 3, 50 ch × 2 | 469 | 188 (40 %) | 152 (`seg_forward_k`) | 129 (`seg_adjoint_kl`) | 0 | 0.6 |
| 320³ order 3, 20 ch × 5 | 910 | 444 (49 %) | 234 (`seg_forward_k`) | 230 (`seg_adjoint_kl`) | 0 | 2.3 |
| 512³ order 3, one coil, 2 ch × 5 groups | 604 | 181 (30 %) | 226 (`seg_forward_k`) | 188 (`seg_adjoint_kl`) | 0 | 10.1 |
| 640×384×320 order 3, one coil, 10 ch | 202 | 101 (50 %) | 53 (`seg_forward_k`) | 46 (`seg_adjoint_kl`) | 0 | 1.6 |

and the bandwidth / lower-bound view of every after case:

| matrix | coils × L | order | samples | batch | engine GiB | free GiB after | FFT model BW % of 1008 GB/s fwd / inv (slabs) | lower bound ms (FFT bytes at peak + measured interp) | measured | variants |
|---|---|---|---|---|---|---|---|---|---|---|
| 192×192×192 | 10 × 10 | 0 | 0.45 M | 10×10 | 10.88 | 11.22 | 76 / 73 (y) | 99 | 132 | vs/vt off, ra on, kl skip |
| 192×192×192 | 10 × 10 | 0 | 1.52 M | 10×10 | 10.88 | 10.87 | 75 / 73 (y) | 152 | 198 | vs/vt on, ra on, kl skip |
| 192×192×192 | 10 × 10 | 0 | 4.32 M | 10×10 | 10.88 | 10.16 | 75 / 73 (y) | 296 | 351 | vs/vt on, ra on, kl skip + kl kernel |
| 192×192×192 | 10 × 10 | 3 | 0.45 M | 10×10 | 10.88 | 11.22 | 76 / 73 (y) | 126 | 164 | vs/vt on, ra on, kl skip |
| 192×192×192 | 10 × 10 | 3 | 1.52 M | 10×10 | 10.88 | 10.87 | 74 / 73 (y) | 236 | 295 | vs/vt on, ra on, kl skip |
| 192×192×192 | 10 × 10 | 3 | 4.32 M | 10×10 | 10.88 | 10.09 | 75 / 73 (y) | 531 | 612 | vs/vt on, ra on, kl skip + kl kernel |
| 192×192×192 | 16 × 16 | 3 | 1.52 M | 8×16 | 14.06 | 7.31 | 73 / 72 (y) | 560 | 710 | vs/vt on, ra on, kl skip + kl kernel |
| 256×256×256 | 10 × 10 | 0 | 0.45 M | 5×10 | 13.53 | 7.69 | 79 / 74 (y) | 192 | 252 | vs/vt off, ra off, kl skip |
| 256×256×256 | 10 × 10 | 0 | 1.52 M | 5×10 | 13.53 | 7.44 | 79 / 75 (y) | 264 | 349 | vs/vt on, ra off, kl skip |
| 256×256×256 | 10 × 10 | 0 | 4.32 M | 5×10 | 13.53 | 6.62 | 79 / 74 (y) | 438 | 555 | vs/vt on, ra off, kl skip |
| 256×256×256 | 10 × 10 | 3 | 0.45 M | 5×10 | 13.53 | 7.69 | 79 / 74 (y) | 241 | 302 | vs/vt off, ra on, kl skip |
| 256×256×256 | 10 × 10 | 3 | 1.52 M | 5×10 | 13.53 | 7.37 | 79 / 74 (y) | 402 | 481 | vs/vt off, ra on, kl skip + kl kernel |
| 256×256×256 | 10 × 10 | 3 | 4.32 M | 5×10 | 13.53 | 6.56 | 78 / 74 (y) | 790 | 920 | vs/vt on, ra on, kl skip + kl kernel |
| 256×256×256 | 16 × 16 | 3 | 1.52 M | 4×16 | 17.69 | 2.41 | 81 / 77 (y) | 876 | 1109 | vs/vt on, ra on, kl skip + kl kernel |
| 320×320×320 | 10 × 10 | 0 | 0.45 M | 2×10 | 12.03 | 7.72 | 66 / 66 (y) | 361 | 553 | vs/vt on, ra off, kl skip |
| 320×320×320 | 10 × 10 | 0 | 1.52 M | 2×10 | 12.03 | 7.44 | 66 / 66 (y) | 453 | 675 | vs/vt on, ra off, kl skip |
| 320×320×320 | 10 × 10 | 0 | 4.32 M | 2×10 | 12.03 | 6.66 | 66 / 66 (y) | 701 | 936 | vs/vt on, ra off, kl skip |
| 320×320×320 | 10 × 10 | 3 | 0.45 M | 2×10 | 12.03 | 7.72 | 65 / 65 (y) | 438 | 632 | vs/vt off, ra on, kl skip + kl kernel |
| 320×320×320 | 10 × 10 | 3 | 1.52 M | 2×10 | 12.03 | 7.44 | 66 / 66 (y) | 713 | 924 | vs/vt off, ra off, kl skip + kl kernel |
| 320×320×320 | 10 × 10 | 3 | 4.32 M | 2×10 | 12.03 | 6.59 | 66 / 65 (y) | 1456 | 1715 | vs/vt off, ra off, kl skip + kl kernel |
| 320×320×320 | 16 × 16 | 3 | 1.52 M | 1×16 | 11.60 | 6.34 | 66 / 67 (y) | 2098 | 2675 | vs/vt off, ra on, kl skip + kl kernel |
| 512×512×512 | 1 × 10 | 0 | 1.52 M | 1×2 | 4.97 | 5.47 | 77 / 72 (y) | 256 | 344 | vs/vt off, ra off, kl skip |
| 512×512×512 | 1 × 10 | 3 | 0.45 M | 1×2 | 4.97 | 5.69 | 75 / 70 (y) | 312 | 395 | vs/vt off, ra on, kl skip + kl kernel |
| 512×512×512 | 1 × 10 | 3 | 1.52 M | 1×2 | 4.97 | 5.47 | 76 / 71 (y) | 517 | 608 | vs/vt off, ra on, kl skip + kl kernel |
| 512×512×512 | 1 × 10 | 3 | 4.32 M | 1×2 | 4.97 | 4.87 | 75 / 71 (y) | 1037 | 1150 | vs/vt off, ra on, kl skip + kl kernel |
| 640×384×320 | 1 × 10 | 0 | 1.52 M | 1×10 | 12.06 | 3.37 | 68 / 66 (y) | 100 | 151 | vs/vt on, ra on, kl skip + kl kernel |
| 640×384×320 | 1 × 10 | 3 | 0.45 M | 1×10 | 12.06 | 3.59 | 68 / 65 (y) | 109 | 153 | vs/vt off, ra on, kl skip |
| 640×384×320 | 1 × 10 | 3 | 1.52 M | 1×10 | 12.06 | 3.37 | 69 / 67 (y) | 160 | 208 | vs/vt off, ra on, kl skip + kl kernel |
| 640×384×320 | 1 × 10 | 3 | 4.32 M | 1×10 | 12.06 | 2.75 | 68 / 66 (y) | 294 | 350 | vs/vt off, ra on, kl skip + kl kernel |
| 640×384×320 | 1 × 16 | 3 | 1.52 M | 1×8 | 9.78 | 2.12 | 71 / 70 (y) | 264 | 337 | vs/vt off, ra on, kl skip + kl kernel |
| 640×384×320 | 10 × 10 | 3 | 1.52 M | 1×5 | 11.60 | 3.69 | 69 / 66 (y) | 2109 | 2723 | vs/vt off, ra on, kl skip + kl kernel |

- **Lower bound** = all transforms' model bytes at the 1008 GB/s peak + the measured interpolation kernels. The
  measured gradient is 1.1–1.5× of it (bound / measured 65–90 %; 77–85 % at order 3 and 1.52 M). The gap: the L2 slab
  passes run L2-limited (65–81 % in the byte model, which counts the slab's saved bytes as absent; in isolation the
  slabs still save 1–23 % of the transform, 18.4), the fused x passes, and zeroing. Slabs on vs off was not measured
  inside the engine in this session.
- **FFT share** of the gradient: 30 % (192³ order 3) to 50 % (320³ order 3, 640 × 384 × 320); the engine's
  k-support-pruned transforms reach 65–80 % of the peak in the byte model (66 % at 400³, where slabs of 25.6 MB planes
  gain little, 18.4).
- **512³ is sample-side bound because of the split**: 5 passes over 1.5 M samples per coil for only 2 channels each:
  forward 45 ms and adjoint 38 ms per pass (414 of 604 ms), FFT 181 ms. A 32 GB card holds 5 segments per batch
  (2 passes); diagonals computed on the fly (§22.5 option 2) would free 10.7 GB here and allow the same on this card.
- **Kernel variants the autotune picked** (one-bin engine; per case in the table above, tile shapes in each
  record's `tiles` field): forward — vt / vs at 192³ (both orders), at 256³ / 320³ for order 0 and the densest
  order-3 set, the tile forward (`seg_forward_k`) for order 3 from 320³ up and at 512³ / 640 × 384 × 320; adjoint —
  at order 3 the ra adjoint or its whole-tile form (the kl kernel, generated from ra; it took over in most cases from
  256³ up), at order 0 on 256³ / 320³ / 512³ the tile adjoint; the kl skip of empty tiles in every case.

### 18.9 The planner's choice (`SegSplit::planned`, opbench `--plan-budget-gb`; one-bin engine, order 3, 1.52 M; c/)

| matrix | coils × L | budget GiB | chosen coils × segments (groups, passes) | gradient ms |
|---|---|---|---|---|
| 192×192×192 | 10 × 10 | 18 | 10 × 10 (1, 1) | 291 |
| 256×256×256 | 10 × 10 | 16 | 5 × 10 (1, 2) | 480 |
| 320×320×320 | 10 × 10 | 14.5 | 2 × 10 (1, 5) | 930 |
| 512×512×512 | 1 × 10 | 4.5 | 1 × 2 (5, 5) | 612 |
| 640×384×320 | 10 × 10 | 7 | 1 × 5 (2, 20) | 2738 |
| 640×384×320 | 1 × 10 | 12 | 1 × 10 (1, 1) | 208 |

In every case it picks the batch measured fastest in 18.7 (including 2 coils at 320³, where the old rule took 3), and
the times match the hand-chosen batches within run-to-run spread (192³ 291 vs 295 ms, 320³ 930 vs 924).

### 18.10 Static order 3: the old multi kernels, and what it takes to route it through vt / ra / kl

Static (one-image) engines at order 3 dispatch `forward` / `adjoint` to the multi-image path (`forward_multi_in` with
one image), because order 3 lived only in the multi kernels when it was written; vt / ra / kl and the fused pad / crop
were later built for the knot-major kernels, which do carry order 3. Measured (18.6, 18.8): the multi kernels cost
≈ 2× order 0 (192³ 1.52 M: 411 vs 197 ms; kernels 290 of 394 ms in the profile, plus 16 ms separate pad / crop and
12 ms full-slot memsets).

Routing it is a small dispatch change, done behind a switch: a static image is a knot-major operator with one bin of
degree 0 (every sample weight 1), and the knot kernels already support a knot support of 1. `new_impl` now allows
`frames == 1 && bspline == 0` on a knot-major engine, and env `HASTY_SEG_STATIC_KNOT=1` builds every static engine
that way (default off). Tests: 18.11. Result: −16 to −40 % at order 3 (18.6).

Why the switch is not the default yet, and what the remaining step is:
- a knot-major engine refuses the multi-image calls (`forward_multi` / `adjoint_multi` with K > 1 images and
  per-sample weights), the grouped calls (`forward_group`, `forward_sink`, `adjoint_source` with several sets), the
  water + fat pair at order 3 (`forward_fat`) and the warp. Callers that use any of them on a static order-3 engine
  would fail with the switch on;
- so the next step is either (a) a per-call fallback: a static engine that holds both kernel sets and dispatches
  single-image order-3 calls to the knot kernels and the rest to the multi kernels (the tables differ: knot-major
  tables carry the interval key, so the set would need both, or the knot table layout for the multi kernels), or (b)
  K-image weights in the knot kernels (what the multi kernels do), which is real kernel work. (a) is the cheaper one;
  until then use the switch (or `SegNufft::new_temporal(.., frames 1, bspline 0, .., true)`) wherever only the
  single-image calls are made — which is the one-frame / one-knot-per-GPU gradient measured here.

### 18.11 Tests (logs research/experiments/grid_sizes/tests/)

- `tests/seg_split.rs` (new; t1_seg_split.log, all pass):
  - `split_matches_plain`: SegSplit with groups of 1, 2 and 3 of L 5 (two zero-padded) against the plain engine,
    static orders 0 / 2 / 3, 1-D cubic knot orders 0 / 3, the one-bin engine order 3, coil batches 1 and 3: forward
    ≤ 1.1e-5, adjoint ≤ 2.4e-5 (bounds 3e-5 … 8e-5 = 6 × the one-ulp floor), the fused residual and adjoint
    accumulation the same, adjointness equal to the plain engine's.
  - `one_bin_knot_matches_static`: orders 0–3, two coil / segment shapes, tile kernels and the autotune's vt / vs,
    ra, ra + skip, kl kernel each forced on: forward ≤ 6.9e-6, adjoint ≤ 4.0e-5 (order 0; bound 7.7e-5) / 6.8e-6
    (order 3); adjointness as the static engine's.
  - `one_bin_sanitize` under compute-sanitizer memcheck (t2_sanitize.log): tile, vs, ra, ra + skip, kl kernel at
    orders 0 and 3 and a 2 × 2 segment split: **0 errors**.
  - `static_knot_env_switch` (HASTY_SEG_STATIC_KNOT=1 equals the static path at orders 0 and 3): in the regression run.
- `segsplit::tests::plan_batch_rules` (CPU): whole coils, the ragged-batch and padded-group rules, caps.
- Regression on frozen binaries (scratchpad/bin/reg, built from this tree; logs tests/reg_*.log): note first: my first frozen set (12:04) was overwritten in the shared scratchpad by another agent; the set run here
was frozen at 15:33 from the tree as it then stood, which also holds other agents' concurrent, uncommitted edits
(hasty-bfp, block-floating-point storage in hasty-fft, segnufft*.cu / segnufft.rs changes that are not mine).
  - Pass: hasty-nufft lib tests (13, incl. `plan_batch_rules`), segnufft, seg_vs, seg_knot_kv, seg_knot_hi,
    seg_knot, seg_knot2d, seg_tile_block, seg_stream, seg_fft_traffic, seg_pruned, seg_multi, seg_frames,
    seg_kernel, seg_b0_direct, seg_wf, seg_joint, seg_fat_union_gpu, seg_kl, seg_split (4, incl.
    `static_knot_env_switch`), accuracy.
  - Known pre-existing failures: seg_joint_chunks (Φ-CGEMM check), seg_ra_vt `ra_static_orders` (the same marginal
    case as §17.5: static order 1, nxg 1, cb 4, 2.2e-4 against a 2.0e-4 bound).
  - **New failure: seg_warp_gpu `warp_adjoint_gpu`** (warped adjoint mismatch 2.0e-4 against 3 × the plain
    envelope 5.5e-5 + 1e-5 = 1.75e-4). Rerun of that test alone (tests/warp3.log): passes twice in a row, with the plain operator's own envelope at
    1.5e-4 and 3.9e-4 (5.5e-5 in the failing run): the bound is 3 × a run-to-run atomics envelope that itself varies
    7×, i.e. a flaky bound. The warp path does not touch this session's code (its forward runs `init_rows` with
    `forward_accumulate` false; its adjoint is the static interval-major path).

### 18.12 Gaps and next

1. **Static order 3 by default** (18.10): per-call fallback so the switch can be on everywhere.
2. **512³ on a 24 GB card is sample-pass bound** (5 passes of 2 channels per coil): free the 10.7 GB of resident
   diagonals (§22.5 option 2, diagonals from the field on the fly) or use a 32 GB card to run 5 segments per batch;
   stream coil maps per coil (§22.5 option 4, not built) so that 10 coils need not be resident at 512³ / 640 × 384 ×
   320 at all.
3. **One-channel batches fall back to the unpruned transform** (`GridFft::new` needs ≥ 2 channels for the pruned
   plan): 1324 vs 608 ms at 512³. The planner avoids them when 2 channels fit.
4. **recon_p2np's knot slot** stays at the legacy 6 GiB cap by default; `--grid-budget auto` is there, but a slot entry
   in the knot ledger (memory_ledger.md §3) is what should size it.
5. **FFT side**: at bandwidth (18.4). Fewer bytes per channel (block-floating-point storage, another agent) is the
   remaining lever; channel sub-slabs are not worth it on this card.
6. Not measured: the knot-major (temporal) engines at these matrices (the per-knot cost is this one-frame cost at the
   knot's sample count, §17), other cards (L2 / memory change the batch and slab choices).

## 19. Phase 2: shared-memory footprint, cp.async, spill screening (2026-11-20, tenth session)

Question: do the three phase-2 levers of research/arch_targets.md §4 (items 2 and 3) make the ra / kl adjoint and the
vt / vs forward faster on the 4090, and do they give every target architecture spill-free, higher-occupancy
candidates? The levers are a smaller shared-memory footprint (§5.3 there: the velocity ra holds 12 warps per SM where
its registers allow 16; the default vt holds 4), cp.async double-buffered staging, and spill screening with a per-shape
launch-bounds hint. Decision: which variants the autotune offers by default, and what the first cluster hour has to
re-time.

Every variant is a new candidate behind a switch. The previous kernels and candidates are unchanged and are timed
first.

### 19.1 What was built

**ra (segnufft_ra.cu, segnufft/ra.rs).** A staging variant `RaVar { inplace, stages, lbmin }` sits next to the launch
shape `RaCfg { nxg, cb, chk }`. `RaVar::BASE` = (false, 1, 0) is the kernel of §16, with the same source text under
its defines.
- `RA_INPLACE 1`: the item's list is read in place. For knot engines that is knot b's window, found by binary search
  of the key-sorted list as ≤ 1 / 2 / 4 runs (`ra_runs`, `ra_entry`; the same entries in the same order as the old
  compaction). There is no LISTMAX buffer in shared memory: −4.1 KB at max_item 1024. The chunk's entries, uw (order 3)
  and htab are kept in shared memory, as in the kl kernel.
- `RA_STAGES 2` (needs in place): records (3 × 16 B), coil values (8 B each) and uw (16 B) of chunk k + 1 are copied by
  `cp.async` (16-byte `.cg` / 8-byte `.ca`) into the second of two buffers while chunk k is staged and MACed. Only the
  list entries are prefetched into registers, one chunk further ahead (three 32-bit registers at chunk ≤ 42 instead of
  float4 / complex prefetch registers). One `cp.async.wait_group 0` plus one barrier at the chunk head replaces the
  register-to-shared copy and its barrier: three barriers per chunk instead of four. Below sm_80 the same structure
  uses plain loads / stores, so the define is portable.
- `LBMIN m > 0`: `__launch_bounds__(128, m)`.
- Chunks (CHK) are any 1 … 64. The code was already CHK-generic; the tests run 8, 12, 13, 16 and 32.

**kl (segnufft_kl_adj.cu, segnufft/kl.rs).** The same `RA_STAGES 2` pipeline and `LBMIN` on the whole-tile kernel. It
always reads in place, so `inplace` is ignored.

**vs / vt (segnufft_vs.cu, segnufft/vs.rs).** `VSCH < SMAX` stages an item in sub-chunks of ≤ VSCH samples. Records,
class table and per-sample accumulators are sized for VSCH, so the vt default (4 × 4 × 16, SMAX 48, nwc 4) shrinks from
76 KB to 42 KB at VSCH 24. The tables are the same. Each sub-chunk is a contiguous, still class-sorted part of the
item's list, and each sample's sum is formed exactly as in one pass, so the result is bitwise the same. The cost is
that the register windows are reloaded once per sub-chunk of items longer than VSCH. API: `set_vs_v(cfg, sch)`,
`vs_active_v`, `vs_resources`.

**Spill screening (all three).** After each candidate is compiled, `KernRes` = registers and local bytes from
`cuFuncGetAttribute`, and blocks / warps per SM from the driver's occupancy calculator, all in microseconds.

The bare local size is not a spill test. Every ra / kl / tile kernel, and the vt sub-chunk kernel, carries the
28-byte local array of the `sincosf` slow path (Payne–Hanek reduction): a 32-byte stack frame without spills. This was
found by SASS line info (STL / LDL at `sincosf(r1.x, …)`). The base vt keeps that array in registers; the sub-chunk
loop does not.

So a kernel with local memory gets its spill-store bytes from the ptxas report (`ptxas_spill`): one extra NVRTC
compile with `--ptxas-options=-v` per source, architecture and NVRTC version. The result is memoised and cached as
`<kernel cache>/ptxinfo/<name>-<hash>.spill`; a repeated query takes 0.3 ms. Kernels with no local memory need no
compile at all.

**Autotune (`autotune_ra`, `autotune_kl`, `autotune_vs`).**
1. Phase A: the previous candidates, with spilling ones skipped and logged.
2. Phase B: for the fastest two (nxg, cb) of phase A, the staging variants. These are chunks 8, 12 and 16, plus the
   largest chunk ≤ 32 at which shared memory does not cut the register-limited block count on this device (`chunk_fit`:
   driver occupancy for a probe compile, so it is larger on sm_80 / sm_90), each with the register prefetch and with
   cp.async.
3. Then the hints 3 and 4 for the fastest three unhinted kernels, kept only if they do not spill and keep at least the
   unhinted warps.
4. vs: for the fastest two one-pass shapes, per extra block count that shared memory allows (up to the register limit),
   the largest sub-chunk reaching it.

If every candidate spills, all are timed.

**Switches (env).**
- `HASTY_SEG_VSA_STAGE = 0 | inplace | cpasync | auto` (ra and kl; default auto = all).
- `HASTY_SEG_LBMIN = auto | 0 | <m>`.
- `HASTY_SEG_SPILLSKIP = 1 | 0`.
- `HASTY_SEG_VS_SCH = auto | 0 | <n>`.
- `HASTY_SEG_VSA_CFG = nxg,cb,chk[,inplace,stages,lbmin]`, `HASTY_SEG_KL_CFG = nxg,cb,chk[,stages,lbmin]`,
  `HASTY_SEG_VS_CFG = nwc,nws,vxpad[,sch]`.

With `HASTY_SEG_VSA_STAGE=0 HASTY_SEG_LBMIN=0 HASTY_SEG_SPILLSKIP=0 HASTY_SEG_VS_SCH=0`, the candidate set and the
kernels are exactly the previous ones.

**API.** `RaVar`, `KernRes` (exported); `set_ra_v`, `ra_active_v`, `ra_resources`, `set_kl_kernel_v`,
`kl_kernel_v`, `kl_resources`. The old `set_ra` / `set_kl_kernel` / `set_vs` take the base variant.

### 19.2 Compile audit of the new shapes (kernaudit families `ra2`, `kl2`, `vs2`; NVRTC 13.0, no GPU)

Run: `kernaudit --families ra2,kl2,vs2` → research/experiments/arch_audit/out_p2/ (2970 kernels, 0 compile errors);
digest `analyze_p2.py` → research/experiments/phase2/audit_tables.md. Shapes covered per case: one block row at nxg 4
and nxg 2 and two rows at nxg 4 (every shape the 4090 autotune picks in §16–§17). Variants per shape: chunks 8 / 12 /
16 / 24 × register prefetch / cp.async × hint 0 / 4, plus hint 3 at (12, cp.async) and (16, prefetch), plus the base
kernel with hints 3 / 4. vs / vt: the default geometry's candidates × sub-chunks 12 / 16 / 24 / 32.

The base kernels are unchanged by the new code: 608 base-variant rows (ra / kl / vs at vel2d_o3_cb5 and
static_o0_48x1, all five architectures) re-audited (out_p2base/) have the same registers, stack, spills and shared
memory as phase 1.

Cell = registers / warps per SM, S = spill-store bytes. Velocity reconstruction, ra nxg 4, cb 7 (50 channels):
| variant | smem (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|
| base chk 16 | 29632 | 156/12 | 128/12 | 125/16 | 123/16 | 122/12 |
| base chk 16, hint 4 | 29632 | 128/16 | 127/12 | 125/16 | 122/16 | 123/12 |
| base chk 16, hint 3 | 29632 | 156/12 | 157/12 | 158/12 | 158/12 | 152/12 |
| in place chk 16 | 26256 | 159/12 | 159/12 | 158/12 | 119/16 | 120/12 |
| in place chk 12 | 19792 | 159/12 | 159/12 | 158/12 | 119/16 | 120/16 |
| in place chk 12, hint 4 | 19792 | 128/16 | 126/16 | 126/16 | 122/16 | 122/16 |
| cp.async chk 8 | 14192 | 156/12 | 125/16 | 124/16 | 124/16 | 121/16 |
| cp.async chk 12 | 21088 | 156/12 | 125/16 | 126/16 | 121/16 | 119/16 |
| cp.async chk 12, hint 4 | 21088 | 126/16 | 125/16 | 126/16 | 124/16 | 124/16 |
| cp.async chk 16 | 27984 | 156/12 | 125/12 | 124/16 | 126/16 | 124/12 |
| cp.async chk 24 | 41776 | 156/12 | 125/8 | 126/16 | 121/16 | 119/8 |

Velocity reconstruction, kl nxg 4, cb 7:
| variant | smem (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|
| base chk 16 | 26256 | 157/12 | 160/12 | 157/12 | 123/16 | 125/12 |
| chk 12 | 19792 | 157/12 | 160/12 | 157/12 | 123/16 | 125/16 |
| chk 12, hint 4 | 19792 | 128/16 | 125/16 | 127/16 | 123/16 | 124/16 |
| cp.async chk 12 | 21088 | 156/12 | 123/16 | 125/16 | 123/16 | 127/16 |
| cp.async chk 12, hint 4 | 21088 | 126/16 | 126/16 | 128/16 | 124/16 | 123/16 |

vt forward 4 × 4 × 16, SMAX 48 (velocity reconstruction, 50 channels):
| variant | smem (B) | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|
| nwc 4 one pass | 76096 | 254/8 | 254/4 | 254/8 | 254/8 | 236/4 |
| nwc 4, sub-chunk 32 | 53568 | 254/8 | 254/4 | 244/8 | 244/8 | 246/4 |
| nwc 4, sub-chunk 24 | 42304 | 254/8 | 254/8 | 244/8 | 244/8 | 246/8 |
| nwc 4, sub-chunk 16 | 31040 | 254/8 | 254/8 | 244/8 | 244/8 | 246/8 |
| nwc 4 nws 2 one pass | 82496 | 232/8 | 236/8 | 240/8 | 252/8 | 238/8 |

What the audit says, per architecture:
- **sm_89 / sm_120 (4090, L40S, 5090; 100 KB):** the velocity ra and kl reach 16 warps per SM, from 12, at chunk 12
  with cp.async: 125 / 119 registers, 21.1 KB, no spills. Chunk 8 does the same at 14.2 KB. Chunk 16 with cp.async stays
  at 12 warps on these parts (28 KB > 24 KB per block for 4 blocks). The vt default reaches 8 warps from 4 with
  sub-chunks of ≤ 29 samples (2 blocks of 42–49 KB). So on 100 KB parts the footprint was the limiter, as §5.3 of
  arch_targets.md predicted.
- **In place with the register prefetch is not enough.** The kl-style entry / uw prefetch costs ≈ 30 registers
  (128 → 159 on sm_89), so in-place chunk 12 stays at 12 warps by registers. Only cp.async (which drops the prefetch
  registers: 125) or the hint 4 (126–128, no spills) gets 16 warps.
- **sm_80 (A100):** the cp.async variant keeps 156 registers (ptxas on sm_80 allocates ≈ 30 more), so 12 warps. The
  hint 4 brings it to 126 registers and 16 warps without spills. On sm_80 the hint is what matters, and the
  autotune's hint step covers it.
- **sm_90 / sm_100 (H100, H200, B200; 228 KB):** the base kernel already holds 16 warps. cp.async keeps 16 at every
  chunk ≤ 24, with fewer barriers per chunk. Larger chunks fit there (chunk 24 still 16 warps), and `chunk_fit` picks
  them.
- **100 channels (coil batch 10, nxg 4, cb 13):** register-bound everywhere (160–225 registers: 8–12 warps). The hint
  4 forces 128 registers with 228–376 B of spills on every architecture, so the screen rejects it. The phase-2
  footprint levers do nothing for this shape; it needs fewer accumulators per lane (two block rows), which the
  autotune already offers.
- **The 48 × 1 spill case (ra nxg 2, cb 3):** the base spills 20 / 24 B on sm_89 / sm_90. cp.async chunk 12 removes
  the spill on every architecture (119–128 registers, 16 warps), as does the hint 4 on the base.
- **Spills by family and arch** (kernels / spilling): ra2 sm_80 360 / 69, sm_89 360 / 56, sm_90 360 / 54, sm_100
  360 / 38, sm_120 360 / 40 (mostly hint-4 shapes with more than 128 registers of accumulators); kl2 162 / 27, 19,
  19, 18, 17; vt2 0 everywhere except sm_120 8 (4-byte spills at 255 registers); vs2 sm_90 3, sm_120 6. Spills are per
  architecture, as in phase 1: the device-side screen (19.1) is needed, not a static list.
- **Spill-free stack frames are not constant:** ra2 32, 40 or 48 B (the in-place run arrays), kl2 48 B, vs2 / vt2 32 B
  (sincosf) without spills. This confirms that the screen must read ptxas rather than compare the local size to a
  constant.

### 19.3 Tests (rustlib/crates/hasty-nufft/tests/seg_phase2.rs; logs research/experiments/phase2/t4_*, t5_*)

Every new variant is compared with the kernel it extends on the same engine and data. Cases:
- static order 0–2 (10 coils × 10 segments; 12 coils in batches of 4 × 3 segments at W = 8);
- 1-D knots order 0–3 (6 coils in batches of 3 × 5 segments: odd 15-channel blocks, c0 > 0);
- 2-D knots order 0–3 (4 × 3);
- for vt / vs: static and 2-D knot cases at 10 × 10 and 4 × 3, plus nwc 5 / 7 with and without x padding at order 0 / 1.

For each case: every (nxg, cb) of the first two candidates plus odd channel blocks (nxg 2 cb 3, nxg 4 cb 5, nxg 1 cb
1); variants chunk 8 / 12 / 13 / 32 × register prefetch / cp.async, cp.async chunk 12 with hint 4, the base kernel
with hint 4; vs / vt sub-chunks 8, 13, smax / 2, smax − 1. Each under HASTY_SEG_ZERO = tiles and full, and each with
unsplit lists (HASTY_SEG_MAXITEM = 16384) and split lists (64).

| test | runs | result |
|---|---|---|
| ra static, unsplit | 520 | all bitwise equal to the base ra of the same (nxg, cb) |
| ra static, split | 420 | within the bound (max 0.27 of max(3e-5, 6 × one-ulp floor, 4 × run-to-run spread)); split tiles add atomically in a run-dependent order |
| ra knot 1-D / 2-D, unsplit | 600 | all bitwise |
| ra knot, split | 480 | max 0.19 of the bound |
| kl knot, unsplit / split | 480 / 480 | all bitwise (no atomics) |
| vs / vt sub-chunks | 52 + 60 (t4) + 168 (t5) | all bitwise equal to the one-pass kernel |
| adjointness | every run | no worse than the variant's base (ra ≤ 4.8e-6, kl ≤ 5.3e-6 = the base kl's own value, vs ≤ 3.7e-6); new pair (vt sub-chunk forward × ra cp.async adjoint) at the one-pass pair's value |
| spill screen | | ra 48 × 1 nxg 2 cb 3 on sm_89: 128 registers, 56 B local, **20 B spill** (ptxas); nxg 4 cb 3: 32 B local, 0 spill; repeat query 0.3 ms |
| autotune | 3 engines | keeps a variant only where faster; result equal to the base within 2e-4 |

Notes:
- The static adjoint image is not deterministic even without split lists (6e-8 run to run): the fused crop adds
  into the image atomically. The bitwise checks therefore run with the unfused crop, which is deterministic. The fused
  path is covered by p2_autotune and by the regression suite.
- With MAXITEM 16384 the 50-channel tile kernels exceed the shared-memory opt-in (a test-setup limit, not a kernel
  error), so the vt test uses the default lists as its unsplit case.

- Independent review of the kernel code (a subagent given the code only; races, out-of-bounds, missed
  synchronisation, cp.async wait-group use): no bug found. It traced every stage-buffer read / write across the chunk
  loop (first / last chunk, n = 0, n < CHK, the output park), the cp.async pairing and alignment, the block-uniformity
  of every barrier and early return, the OFF_* layouts against the host formulas, and the equivalence of `ra_runs` with
  the old window compaction. Applied from its remarks: the masked coil slots of the cp.async buffer are zeroed (never
  read, but the register path zeroes them), the cp.async global operand goes through `__cvta_generic_to_global`, the
  host caps LBMIN at 8, and a comment records that 1-D `ra_runs` relies on key-sorted item lists (as kl does).

### 19.4 Timing on the 4090 (opbench, 300 W, flow_thorax 288×288×140 grid, 10 coils × L 10, coil batch 5, order 3, 2-D knots 6:1,16:3)

Logs: research/experiments/phase2/ob_{lo,hi}_{old,new}.log (run_ob.sh). "old" = `HASTY_SEG_VSA_STAGE=0 HASTY_SEG_LBMIN=0
HASTY_SEG_SPILLSKIP=0 HASTY_SEG_VS_SCH=0` (exactly the previous candidates); "new" = defaults. One opbench run each, same
session. Kernel times are the autotune's (best of 3; ra / vs on knot 0, kl on the representative knot, one coil batch).

Kernels (ms):

| kernel | 1.33 M old → new (chosen shape) | 13.3 M old → new (chosen shape) |
|---|---|---|
| ra adjoint, knot 0 | 7.29 (nxg 1 cb 2 chk 16) → **6.91** (nxg 4 cb 7, cp.async chk 12, hint 4), −5 % | 30.32 (nxg 4 cb 7 chk 16) → **28.61** (cp.async chk 12), −5.6 % |
| kl adjoint, representative knot | 6.75 (chk 16) → **6.40** (cp.async chk 12), −5 % | 61.2 → 56.6 (chk 20, hint 4); the skip path (55.2) still wins |
| vt forward, knot 0 | 5.20 (nws 2, one pass) → **4.26** (nwc 4, sub-chunk 29: 49 KB, 2 blocks), −18 % | 29.39 → **27.41** (sub-chunk 29), −6.7 % |

Gradient (forward + adjoint, one encode with all frames, 192 knot × coil-batch launches each way):

| case | forward | adjoint | gradient |
|---|---|---|---|
| 1.33 M per encode, old | 4981.9 | 4908.6 | 9919.0 |
| 1.33 M per encode, new | 4586.4 | 4799.0 | **9404.2 (−5.2 %)** |
| 13.3 M per encode, old | 12548.7 | 13492.4 | 26061.5 |
| 13.3 M per encode, new | 12536.8 | 12993.0 | **25556.3 (−1.9 %)** |

The old rows reproduce §17.6 (9921 / 26146 ms). FFT time is unchanged (2.98–3.14 s per direction).

### 19.5 Achieved occupancy (Nsight Compute, 4090)

Setup: one knot launch of each fixed variant (opbench `--no-tune`, env `HASTY_SEG_VSA_CFG` / `HASTY_SEG_KL_CFG`). Case:
2-D 6:1,16:3, order 3, reduced scan at 3 mm, a 192×192×96 grid. Sections: LaunchStats, Occupancy, SpeedOfLight,
WarpStateStats, SchedulerStats. Output: research/experiments/phase2/ncu/*.details.txt.

At this grid opbench runs coil batch 10 (100 channels), so nxg 4 cb 7 is two block rows of 56 channels. Registers and
shared memory are those of the 50-channel shape within a few bytes. ncu times are at its locked base clock.

| kernel | registers | dynamic smem (KB) | block limit regs / smem | theoretical warps | achieved warps / SM | no-eligible cycles | time (ms) |
|---|---|---|---|---|---|---|---|
| ra base chk 16 | 128 | 29.9 | 4 / 3 | 12 | 11.85 | 54.0 % | 8.05 |
| ra in place chk 16 | 161 | 26.5 | 3 / 3 | 12 | 11.86 | 54.9 % | 8.17 |
| ra in place chk 12 | 161 | 20.0 | 3 / 3 | 12 | 11.92 | 55.0 % | 8.43 |
| ra in place chk 12, hint 4 | 128 | 20.0 | 4 / 4 | 16 | 15.74 | 47.4 % | 7.32 |
| ra cp.async chk 16 | 125 | 28.5 | 4 / 3 | 12 | 11.89 | 53.7 % | 8.12 |
| **ra cp.async chk 12** | 125 | 21.5 | 4 / 4 | 16 | **16.27** | 46.8 % | **7.20 (−10.6 %)** |
| ra cp.async chk 8 | 125 | 14.5 | 4 / 4 | 16 | 15.78 | 45.9 % | 7.44 |
| kl base chk 16 | 158 | 26.5 | 3 / 3 | 12 | 11.84 | 54.4 % | 8.10 |
| **kl cp.async chk 12** | 123 | 21.5 | 4 / 4 | 16 | **15.99** | 44.9 % | **6.98 (−13.8 %)** |
| kl chk 12, hint 4 | 125 | 20.0 | 4 / 4 | 16 | 15.82 | 46.1 % | 7.24 |

The operator times of the same runs (all 96 knots × coil batches, the fixed variant in every launch, 3 mm, 100
channels, no autotune) give the adjoint per call:
- ra: base 2230.9 ms; cp.async chunk 12 2166.8 ms (−2.9 %); in place chunk 12 with hint 4 2175.9 ms; in place chunk 12
  2280.7 ms;
- kl: base 2212.9 ms; cp.async chunk 12 2112.5 ms (−4.5 %).

- **Prediction confirmed.** The phase-1 prediction (arch_targets.md §4 item 2: shared memory caps ra at 12 warps on
  100 KB parts; a ≤ 24 KB footprint reaches 16) holds. Achieved occupancy follows the theoretical number within 2 %.
- **In place needs cp.async or the hint.** Reading the list in place alone does not help: the kl-style prefetch of the
  entries and uw costs about 33 registers (128 → 161), so the block becomes register-limited at 3. The cp.async double
  buffer frees those registers instead (125). The hint 4 does the same at the same speed when the in-place register
  prefetch is kept.
- **Chunk size.** Chunk 16 with cp.async stays at 3 blocks (28.5 KB > 24 KB). Chunk 8 fits, but its extra barriers per
  sample cost 3 % against chunk 12.
Repeated with coil batch 5 (`--cb-cap 5`: 50 channels, one block row, exactly the audit's velocity shape; ncu5/):

| kernel | registers | smem (KB) | achieved warps / SM | no-eligible | time (ms) | adjoint per call (ms) |
|---|---|---|---|---|---|---|
| ra base chk 16 | 128 | 29.6 | 11.88 | 52.3 % | 3.92 | 2221.0 |
| ra in place chk 12 | 159 | 19.8 | 11.88 | 53.6 % | 4.11 | 2272.0 |
| ra in place chk 12, hint 4 | 126 | 19.8 | 15.85 | 44.2 % | 3.52 | 2158.1 |
| ra cp.async chk 16 | 125 | 28.0 | 11.90 | 52.4 % | 3.97 | 2235.1 |
| **ra cp.async chk 12** | 125 | 21.1 | **15.84** | 44.0 % | **3.50 (−10.7 %)** | **2154.0 (−3.0 %)** |
| ra cp.async chk 8 | 125 | 14.2 | 15.85 | 44.1 % | 3.61 | 2188.2 |
| kl base chk 16 | 160 | 26.3 | 11.85 | 53.1 % | 3.95 | 2199.1 |
| **kl cp.async chk 12** | 123 | 21.1 | **15.84** | 43.4 % | **3.42 (−13.4 %)** | **2097.7 (−4.6 %)** |

vt forward, the same case with coil batch 5 (`--cb-cap 5`: 50 channels, the velocity shape; ncu/vt5_*):

| kernel | registers | dynamic smem (KB) | block limit regs / smem | theoretical warps | achieved warps / SM | no-eligible cycles | time (ms) |
|---|---|---|---|---|---|---|---|
| vt nwc 4, one pass | 254 | 76.1 | 2 / 1 | 4 | 3.98 | 71.0 % | 5.72 |
| vt nwc 4 nws 2, one pass | 236 | 82.5 | 1 / 1 | 8 | 7.93 | 64.3 % | 5.31 |
| **vt nwc 4, sub-chunk 24** | 254 | 42.3 | 2 / 2 | 8 | **7.91** | 57.1 % | **3.99 (−30 %)** |

The forward over all knots with the vt shape fixed: one pass 2511.9 ms, nws 2 2538.6 ms, sub-chunk 24 **2278.1 ms
(−9.3 %)**. So the sub-chunk wins on every knot of the 1.33 M case, not only on knot 0.

- **Where the kernel gain goes.** The kernel gain per launch (−11 % / −14 %) is larger than the operator gain (−3 % /
  −4.5 %). In the operator the adjoint also contains the zero passes, the FFTs (≈ 1.05 s of the 2.2 s at this grid),
  pad and crop.

### 19.6 compute-sanitizer and regression

compute-sanitizer runs on `p2_sanitizer_case` of tests/seg_phase2.rs, with HASTY_NVRTC_LINEINFO=1. The case has 3000
samples on the 30×28×30 grid of the 24×22×24 image. Each case covers every ra / kl variant of the first three (nxg, cb)
under ZERO tiles and full, and the vs sub-chunks 8 / 13 / 24 / 47. There are four cases:
- static order 0;
- static order 2;
- 1-D knots order 1;
- 2-D knots order 3.

Logs: research/experiments/phase2/san/.

| tool | unsplit lists (MAXITEM 16384) | split lists (64) |
|---|---|---|
| memcheck | 0 errors, all 4 cases | 0 errors, all 4 cases |
| racecheck (filtered to seg_adjoint_ra / seg_adjoint_kl / seg_forward_vs) | 0 hazards: static order 0 (316 s), static order 2 (562 s); **not finished**: 1-D order 1, 2-D order 3 | not run |
| synccheck | not run | not run |

- **racecheck host memory.** The racecheck run of the 1-D order-1 case was killed twice by the 24 GB memory cap of its
  unit:
  - once with all three kernel families in one process;
  - once with the ra variants alone (`P2_SAN_FAM=ra`).

  Racecheck's host memory grows with every instrumented module loaded in the process, and each variant is a new module.
  The peak RSS per invocation was not recorded. Sizing it is the first step of the next campaign
  (research/experiments/phase2/HANDOFF.md).
- **Kernel filter.** Without the filter the first racecheck run (static order 0, all kernels instrumented) did not
  finish within 900 s. The filter limits instrumentation to the three phase-2 kernel entries.
- **What is covered.** The cp.async / smaller-buffer paths that racecheck matters most for are covered so far by:
  - the clean static runs, which include every ra cp.async variant;
  - the bitwise test results;
  - the code review.

  The knot-engine paths (kl, 2-D window runs) have memcheck only.

**Regression (the seg_* GPU suite with frozen binaries): not run.** The chain that was to run it after the sanitizer was
stopped by the out-of-memory kill. The coordinator then reserved all further GPU work for one combined campaign. The
frozen binaries (built 2026-11-20 13:3x from this session's source) are in the session scratchpad (bin/reg/). The
known pre-existing failures to expect are seg_joint_chunks (Φ-CGEMM) and seg_ra_vt::ra_static_orders (marginal
≈ 1.05× bound).

### 19.7 What it means and what is default

- **Default (HASTY_SEG_VSA_STAGE / LBMIN / SPILLSKIP / VS_SCH all auto, as the code stands).** The autotune times the
  previous candidates first, then the phase-2 variants, and keeps a variant only where it measured faster. On the 4090 it chose cp.async at
  chunk 12 for ra and kl and the vt sub-chunk 29 on the velocity reconstruction at both densities.
- **4090, velocity reconstruction (2-D knots, order 3, coil batch 5).**
  - Kernels: ra −5 %, kl −5 %, vt −18 % (1.33 M) / −7 % (13.3 M) per autotune launch. In ncu, one launch at fixed
    shape: ra −11 %, kl −14 %, with 12 → 16 achieved warps.
  - Gradient: −5.2 % at 1.33 M and −1.9 % at 13.3 M samples per encode.
  - At 13.3 M the forward did not gain, although the vt kernel was 7 % faster on knot 0. The ra / vs autotune still
    times knot 0, an edge respiratory knot with about half the window (§17.8 item 3; the fix is in segnufft.rs, owned by
    another agent this session). A sub-chunk chosen on a thin knot can lose on the dense interior knots, where items
    are longer than the sub-chunk and the windows are reloaded.
- **Other architectures (compile audit, 19.2).**
  - Every target has a spill-free 16-warp ra / kl variant for the 50-channel velocity shape. On sm_89 / sm_120 it is
    cp.async chunk 12 (or 8); on sm_80 it is the hint 4 (with or without cp.async); on sm_90 / sm_100 the base already
    holds 16, and cp.async adds the larger chunks.
  - vt reaches 8 warps on sm_89 / sm_120 with sub-chunks (already 8 on sm_80 / sm_90).
  - 100-channel shapes stay register-bound everywhere.
- **Cost.** The first autotune of an engine shape compiles about 2× more kernels, plus one ptxas-report compile per
  kernel that has local memory: tens of seconds of NVRTC per new shape, cached on disk afterwards. Arch / driver keyed
  tune caching (arch_targets.md §4 item 0) would remove the re-timing as well.

### 19.8 Next

1. Representative-knot autotune for ra / vs on 2-D engines (segnufft.rs autotune hook: pass the kl step's `kb`
   instead of 0). This is the likely reason the 13.3 M forward did not see the vt gain.
2. Warp-specialised ra / kl (arch_targets.md §4 item 4). cp.async now carries the raw inputs, so a producer warp can
   build w_l, z and the row weights for chunk k + 1 while the consumer warps MAC chunk k. It also removes the remaining
   two inner barriers.
3. The 100-channel shapes (coil batch 10): fewer accumulators per lane (e.g. nxg 4 cb 7 in two rows already exists;
   the autotune chooses among them). No footprint lever helps there.
4. On the cluster: re-time all variants (the tune differs per architecture: on sm_80 the hint, on sm_90 the larger
   chunks). Run `kernaudit --families ra2,kl2,vs2 --arch <sm>` before booking (research/experiments/arch_audit/run.sh).

### 19.9 Files

- Kernels:
  - `rustlib/crates/hasty-nufft/src/segnufft_ra.cu` (RA_INPLACE, RA_STAGES, LBMIN; `ra_runs`, `ra_entry`, `cpa*`);
  - `segnufft_kl_adj.cu` (RA_STAGES, LBMIN);
  - `segnufft_vs.cu` (VSCH).
- Host:
  - `segnufft/ra.rs`: `RaVar`, `KernRes`, `kern_res`, `ptxas_spill`, `chunk_fit`, `ra_variant_plan`, two-phase
    `autotune_ra`, env switches;
  - `segnufft/kl.rs`: `kl_*_v`, `kl_variant_plan`, `autotune_kl`;
  - `segnufft/vs.rs`: `vs_*_v`, `vs_sch_plan`, `autotune_vs`.
- Outside the owned files (small):
  - segnufft.rs line 36 exports `KernRes` and `RaVar` (`pub use ra::{KernRes, RaCfg, RaVar}`);
  - lib.rs re-exports them;
  - hasty-nufft/Cargo.toml adds the `hasty-site` dependency (the kernel cache path for the ptxas sidecar files).
- Tests: `rustlib/crates/hasty-nufft/tests/seg_phase2.rs`.
- Audit:
  - `rustlib/crates/hasty-bench/src/bin/kernaudit.rs` (families `ra2`, `kl2`, `vs2`; `--grep`);
  - `research/experiments/arch_audit/out_p2/`, `out_p2base/`, `analyze_p2.py`.
- Runs: `research/experiments/phase2/`:
  - `run_tests_split.sh`, `run_ob.sh`, `run_ncu.sh`, `run_ncu5.sh`, `run_san.sh`, `run_reg.sh`, `chain2.sh`, `chain3.sh`;
  - logs `t4_*`, `t5_*`, `ob_*`, `ncu/`, `ncu5/`, `san/`, `reg/`;
  - `audit_tables.md`.

## 20. Integration: defaults and campaign (2026-11-20, eleventh session)

Question: make the measured-better paths of §15–§19 and research/bfp_storage.md §8 the defaults (each with a switch back), fix the truth-bc segment rule and the velocity objective's gradient sink, then check the whole operator once (correctness, sanitizers, accuracy against the one-ulp floor, timing old vs new). Status: implemented and built; the campaign (research/experiments/integration/) was started 19:03 and is running; its results are to be filled in below (logs/, campaign.log, perf.jsonl).

### 20.1 What was built (nothing removed; every new default has a switch back)

| # | default now | switch back (old path) | where |
|---|---|---|---|
| 1 | Phase 2 candidates in the autotune (cp.async staging, chunks 8 / 12 / 16 / fit, launch-bounds hints 3 / 4, vt / vs sub-chunks, spill screen): already the default after §19 | `HASTY_SEG_VSA_STAGE=0 HASTY_SEG_LBMIN=0 HASTY_SEG_SPILLSKIP=0 HASTY_SEG_VS_SCH=0` | segnufft/ra.rs, kl.rs, vs.rs (unchanged here) |
| 1b | the tile / vs / vt / ra candidates of a 2-D knot engine are timed on the first cardiac knot of the middle respiratory knot (as kl already was), not knot 0 | `HASTY_SEG_TUNE_KNOT=old` (knot 0 for tile / vs / ra, middle for kl); `=<index>`: that knot for every step | `SegNufft::tune_knot` (segnufft.rs), kl.rs |
| 2 | static order-3 engines are one-bin knot engines (vt / vs forward, ra / kl adjoint, fused pad / crop, split-tile zeroing at order 3) | `HASTY_SEG_STATIC_KNOT=0` (never); `=1` every order (the old opt-in); thread form `set_thread_static_knot(Some(StaticKnot::…))` | `static_knot_default`, `new_impl` |
| 2b | per-call fallback: on a one-bin engine the multi-image (`forward_multi` / `adjoint_multi` / `autotune_multi` / `multi_info`), grouped (`forward_group`, `forward_residual_group`, `adjoint_group`, the fat groups) and water + fat (`forward_fat` / `adjoint_fat`) calls run through the multi-image kernels of the same engine, on the same tables (one bin: the knot-major tables equal the static ones; the interval key is 0). The K = 1 multi shapes are tuned at the first such call after `autotune` | `HASTY_SEG_FB_TUNE=0` (no K = 1 tune) | `SegNufft::as_static`, `as_static_k1`, `is_one_bin`; `group_multi` forces the multi kernels in fallback mode |
| 3 | grid-FFT y ↔ z intermediates in bfp24, in place in the grid slot | `HASTY_SEG_FFT_STORE=f32` (or `set_thread_fft_store(Some(Store::F32))`, `SegNufft::set_fft_store`) | fused.rs `fft_store_default` |
| 3b | dense fallback (separate buffer) when a plane's lines n0 · nch are not whole 16-line tiles; f32 with one printed note per process where the passes do not cover BFP (y / z lengths > 396 on 99 KB parts) | `HASTY_SEG_FFT_STORE_LOG=1` prints the note for every transform | fused.rs `apply_store` |
| 3c | the multi-image passes' own transforms (cbm ≠ cb) count their BFP bytes in `multi_plan`; `set_fft_store` after construction re-budgets the coil batch (rebuilds slots / kernels at default shapes when the batch changes, with a note) and re-plans the multi passes when the store needs more bytes | — (f32 store: no change) | `multi_plan_with`, `SegNufft::rebatch`, fused.rs `set_fft_store` |
| 4 | engine batch: a batch whose planes are whole 16-line tiles (in place) is preferred over the dense fallback when it needs no more passes (equal batches, fewest passes, then fewest transformed channels) | `HASTY_SEG_FFT_INPLACE_PREF=0` | fused.rs `prefer_inplace` |
| 4b | recon_p2np `--grid-budget auto` (default): the knot slot sized by the knot ledger before the engine is built (`vknot::slot_budget`: the largest slot that fits next to the estimated state, data, tables, scratch with 5 % kept free; the engine planner's rule picks the batch under it; legacy if the estimate refuses) | `--grid-budget legacy` (the previous (free − work) / 3 ≤ 6 GiB), `free` (the §18 formula), or a size | recon_p2np.rs, vknot.rs, `hasty_nufft::seg_slot_bytes` |
| 5 | `--ratemap truth-bc`: the segment rule on every knot's own model (field, water and fat decay, fat fraction), maximum over knots and encodes; prints the chosen L, the per-knot L and the per-knot predicted error | `HASTY_SEG_RULE_KNOTS=static` (the static-field, phase-only rule) | recon_p2np.rs (the rm_knots block now precedes the rule) |
| 6 | the velocity objective (at-knots mode) has a gradient sink: the knots' chunks are partial sums in the sink (`GradSink::get` / `put_partial`, new trait methods with defaults), completed by one `put` per chunk; the knot ledger plans g and g_prev at the BFP width (`vknot::plan_sink`) | `HASTY_VOBJ_SINK=0`; host state default stays f32 (`--state-store` unchanged) | vobj.rs, bfpstate.rs, vknot.rs |

Unchanged on purpose: W, σ, every kernel source, the solver defaults (host state f32; P²nP with a BFP state still
warns below 20 bits, bfp24 recommended, research/bfp_storage.md §9), the tile-kernel candidate set.

### 20.2 Campaign

Driver research/experiments/integration/run.sh, cases in cases.txt (tests t*, regression r*, sanitizers s*, accuracy a*, performance p*), unit int-campaign; stop.sh / resume.sh. Frozen binaries: session scratchpad int/bin (freeze.sh). Results: pending.

#### 20.2.1 Partial results (2026-11-20 / 21)

The campaign was stopped after the correctness tests and part of the regression. Everything after that was run by hand.

**Smoke timing.** One gradient (forward + adjoint, FFTs included) for one (frame, encode) set:
- 256³ image (320³ grid), 1.52 M samples (one set of the 20-frame split of the 10-min scan);
- 10 coils × 10 segments, coil batch 5 (50 channels), order 3;
- one opbench binary for all three rows, power limit 450 W;
- "old" = `HASTY_SEG_STATIC_KNOT=0 HASTY_SEG_FFT_STORE=f32 HASTY_SEG_VSA_STAGE=0 HASTY_SEG_LBMIN=0 HASTY_SEG_SPILLSKIP=0 HASTY_SEG_VS_SCH=0 HASTY_SEG_TUNE_KNOT=old`.

| paths | forward | adjoint | gradient | kernels fwd / adj | FFT fwd / adj | pad/crop/rest | autotune |
|---|---|---|---|---|---|---|---|
| old | 330.8 | 346.1 | **677.0 ms** | 228.1 / 212.2 | 81.5 / 87.8 | 21.2 / 46.2 | 11.9 s |
| new defaults, f32 store | 209.5 | 191.8 | **403.5 ms (−40 %)** | 108.8 / 95.6 | 81.6 / 87.8 | 19.1 / 8.4 | 122 s |
| new defaults, bfp24 (fixed) | 211.0 | 186.9 | **399.7 ms (−41 %)** | 108.8 / 95.7 | 83.9 / 83.7 | 18.3 / 7.4 | 126 s |

What the rows show:
- **The gain is in the kernels.** Static order 3 runs on the one-bin knot engine with the Phase-2 shapes.
- **BFP adds about 1 % here.** With the L2 slabs, the x ↔ y intermediates stay in L2, and the z side reads and writes the f32 grid, so only part of the traffic is compressed.
- **The autotune now takes about 2 min per engine build**, because it has more candidates. The per-GPU tune cache (arch_targets item 0) is the fix.

**BFP store slowdown: cause and fix.**
- **The symptom.** The first in-place bfp24 version was *slower* than f32 (FFT pair at 320³, 24 ch: forward +26 %).
  - Nsight Compute: the y pass that writes BFP took 66 µs per slab against 31 µs for f32. It was L2-bound: L2 bandwidth 74 % while DRAM was at 24 %.
- **The cause.**
  - Each lane packed a run of 8 values into 6 words and wrote them with 6 separate 4-byte stores.
  - A warp's store instruction therefore scattered 4-byte pieces at a 24-byte stride across every 96-byte block.
  - Each 32-byte sector took about 6 partial-sector writes.
- **The fix (codegen.rs BFP store epilogue, `bfp_block_word_q` in hasty-bfp/src/bfp.cuh).** The bit layout is unchanged, and the bfp_store tests are bit-exact against the host codec.
  - Phase 1 quantises each run in place in shared memory and writes the exponent byte.
  - After a barrier, phase 2 has consecutive lanes write consecutive uint4 / uint2 pieces of each block's mantissa words, so every warp store covers whole sectors.

FFT pair after the fix (320³ grid, 256³ image, 24 channels; per-pass times in ms):

| store | pair, slab 1 | pair, whole-grid | x | y (writes) | z (reads) | z⁻¹ (writes) | y⁻¹ (reads) | x⁻¹ |
|---|---|---|---|---|---|---|---|---|
| f32 | 51.3 ms | 62.8 ms | 7.85 | 10.25 | 13.78 | 13.27 | 10.08 | 7.86 |
| bfp24, scattered stores | 60.6 ms | 69.5 ms | 7.85 | 14.09 | 12.37 | 20.36 | 8.68 | 7.86 |
| bfp24, fixed | 49.8 ms | 58.3 ms | 7.84 | 9.15 | 12.41 | 12.95 | 8.68 | 7.85 |

After the fix every BFP pass runs at 79–91 % of DRAM peak.

**Default policy change.** The default store is bfp24 in place where a plane's lines n0 · nch are whole 16-line tiles; elsewhere the transform stays f32. The automatic dense fallback is removed: its buffer (≈ 0.56 of a slot) shrank the coil batch, e.g. 6 → 4 in seg_tile_block, and the coil batch is the bigger lever. The dense layout is used only with `HASTY_SEG_FFT_BFP_LAYOUT=dense`.

**Regression (hasty-fft + hasty-nufft, all suites, current tree).** 39 suites pass. The only failures are the two known pre-existing ones: seg_joint_chunks (Φ-CGEMM) and seg_ra_vt::ra_static_orders (marginal bound).

Test changes:
- **seg_fft_traffic `fft_slabs_match_whole_grid` and seg_pruned `pruned_grid_fft_and_fused_residual`** are f32-equivalence checks, so they now pin the f32 store. BFP rounding differs between slab and pruned layouts by design, and the BFP error is tested in bfp_store, seg_fft_store and the phantom runs.
- **seg_defaults** follows the new policy: in place or f32, and the re-budget case asks for the dense layout explicitly.
- **seg_stream** now checks streaming correctness with the f32 store, and checks each path's BFP error separately: 8.0–8.4e-6, the same on streamed and resident tables, bound 2.5e-5.
  - Its f32 floor rose from 2e-5 to 3e-5. With every new variant off, streamed vs resident already differ by 0.9–2.4e-5 in the adjoint and ≈ 6e-7 in the forward, against a resident repeat of ≈ 5e-7.
  - **OPEN:** why streamed and resident tables are not processed identically. The forward has no atomics, yet differs.

**Still pending:** the accuracy cases (bfp_phantom --compare-defaults), the sanitizers on the new defaults, the 2-D-knot velocity-recon timing (bfp24 keep rule ≥ 0.58 s), and 640×384×320.

#### 20.2.2 Compressed DRAM traffic (2026-11-21; research/bfp_storage.md §10)

Defaults changed (each has a switch back). Campaign (450 W): 256³ gradient 397.8 → 390.9 ms, 640 × 384 × 320 180.1 →
177.2 ms, 2-D knot velocity config 21.78 → 21.54 s (old = the binary before this session, f32 store).

| # | new default | switch back | effect (256³ image, 50 channels, one set, order 3, 450 W) |
|---|---|---|---|
| 1 | under L2 slabs, the hooked inverse drops each slab's dead y-pass output from the L2 after the fused crop read it (`discard.global.L2`, sm_80+, a no-op below) | `HASTY_FFT_SLAB_DISCARD=0` | DRAM writes per gradient 82.4 → 66.7 GB; adjoint 186.6 → 179.7 ms, gradient 402.1 → 394.1 ms |
| 2 | the y ↔ z store is f32 again (the user's decision: fastest gradient; every on-device store f32 by default) | `HASTY_SEG_FFT_STORE=bfp24` (the §20 default) or `=auto` (bfp24 for whole-grid passes, f32 under slabs) | with the discard, bfp24 no longer pays under slabs: 256³ 391.8 against 390.9 ms, 2-D knots 21.68 against 21.54 s (the L2-resident BFP y pass costs 11.6 ms, the BFP z passes save 6.6 ms) |

New switches, default off (measured slower on the 4090, or not yet measured where they could pay):

| switch | what | measured |
|---|---|---|
| `HASTY_SEG_GRID_STORE=bfp16 … bfp24` | the forward z pass stores the grid slot as BFP in place (exponent bytes in the slot's tail, 1/128); the tile / knot tile / vs / vt gathers decode | vs 57 → 120 ms, gradient +6 % |
| `HASTY_SEG_MAPS_STORE`, `HASTY_SEG_DIAG_STORE` `=bfp16 … bfp24` | BFP copies of the coil maps / segment diagonals decoded by the fused pad / crop staging (f32 arrays kept for the other paths) | 256³: +0.5 (bfp16) to +2.5 ms (bfp24); 640 × 384 × 320: campaign |
| `HASTY_FFT_BFP_STORE=fused \| pack` | other BFP store epilogues (fused into the last stage / words packed per run) | bit-identical; no faster than the default `epi` |
| `HASTY_FFT_BFP_LOAD=vec` | coalesced vector loads of the BFP blocks through shared memory | bit-identical; 4 % slower on the z pass |
| `HASTY_FFT_BFP_DISCARD=1` | the BFP y pass drops its tile's cell runs from L2 before rewriting them (no dirty last sector) | −20 % slab writes, +1.1 ms issue time per transform |

The 640 × 384 × 320 grid (800 × 480 × 400) does not use the BFP passes at all: y / z lengths above 396 do not fit
16-line BFP tiles in 99 KB of shared memory.

Campaign: research/experiments/compressed_dram/ (unit cz-campaign), results in research/bfp_storage.md §10.6.

## 21. Tensor-core forward contraction (`vtc`) and balanced ra staging (2026-11-21, twelfth session)

Question: can the two interpolation kernels of one gradient get faster at an added arithmetic error of at most about
1e-5 relative? That budget is set against the W 10 / σ 1.25 kernel's own truncation error of about 1e-5, so the total
stays near 1e-5, well below 1e-4. W and σ are unchanged. Lever B is to run the forward contraction on tensor cores.
Lever A3 removes control flow and barrier stalls from the ra adjoint. Levers A1 / A2 (block-shared weight table and
lower register pressure in the FMA vt) were not built; see §21.7.

The case for every number below:
- **Data:** one sample set (frame 4, encode 2 of flow_thorax) with 1 516 200 samples.
- **Operator:** order 3 with segments, at matrix 256 × 256 × 256, fine grid 320 × 320 × 320.
- **Channels:** 10 coils × 10 segments; coil batch 5, so 50 channels per launch and 2 launches per forward.
- **Run:** opbench `--models seg-full --order 3 --sets 1 --matrix 256,256,256 --cb-cap 5 --max-grid-gb 14`.
- **Hardware:** RTX 4090, power limit 450 W (the user's setting, not changed).

The forward kernels are the vs geometry 4 × 4 × 16 with items of ≤ 48 samples: 73 978 items, about 20.5 samples per
item. The best FMA vt is nwc 4, nws 1, sub-chunk 24, as the autotune picks it at this size.

### 21.1 What the profile says before

These Nsight Compute numbers come from the brief and were re-checked here. The ncu clock is 2.23 GHz.

**vt forward, FMA, sub-chunk 24:**
- 65.0 ms, 36.1 G warp instructions executed, 254 registers, 49.3 KB shared memory.
- 2 blocks per SM, so 16.7 % occupancy; issue slots 49 % busy.
- The instruction mix is dynamic (executed instructions, recomputed from vs_sass.csv): FFMA 45.5 %, LDS 8.6 %, FMUL 8.1 %, IMAD 7.9 %, IADD3 5.5 %, LEA 4.5 %, BRA 1.0 %.
- Counting: the contraction itself is 12 G of the FFMAs. That is 200 per lane and 800 per (sample, plane) over the 4 channel-group warps. The redundant per-warp weight build is about 4.4 G more. The other 20 G are addressing, loads and loop control, about 400 instructions per (sample, plane, warp).
- With one-pass staging (sub-chunk 48, 76.1 KB): 1 block per SM, 8.3 % occupancy, 91.2 ms, 34.2 G instructions.

**ra adjoint (nxg 4, cb 7, chk 14):** the "34 % control flow" in the brief was the static opcode count. Executed, it is:
- BRA 4.6 %, BSSY 1.6 %, BSYNC 1.85 %, about 8 % in total.
- ncu agrees: 2.67 G branch instructions of 40.2 G, branch efficiency 94.7 %.
- The executed mix is FFMA 42.7 %, IMAD 10.8 %, LDS 9.2 %, ISETP 6.8 %.
- The barrier stall (1.61 per issued instruction) comes from load imbalance in the staging loops. Every loop starts at `q = threadIdx.x`, so the tails all land on warp 0.

### 21.2 What was built (nothing removed; default unchanged)

**`VTC` in `segnufft_vs.cu`** (host: `segnufft/vs.rs`; env `HASTY_SEG_VT_MMA` = 0 | bf16 | fp16 | auto; `SegNufft::set_vt_mma`):
- It is a new main loop of the vt forward, orders ≥ 2. The staging and the epilogue are the vt kernel's own.
- With VTC unset, the preprocessed FMA source is unchanged. nvcc cubins are byte-identical to the previous source for four shapes: vt order 3 knot, vt order 2 sub-chunk 24, vs order 0 nws 2, vs order 1 nwc 7.

Steps per block, where a block is one item × one channel row:
- **Stage 0:** the item's list entries are re-sorted by oz in shared memory. Sub-chunks, and the samples active on each plane, are then contiguous.
- **Per sub-chunk** (≤ 32 samples; `nws` = 1 | 2 K-chunk splits; warps = nwc · nws):
  - The block keeps a table of union-window point offsets. This is the tile's 13 × 13 = 169 points, padded to 176 = 11 chunks of 16.
  - It also keeps the active range [a, b) of every plane.
- **Per plane:** the active samples' weights are built once, a warp per sample, with the vt kernel's own `vt_build` arithmetic. They go into a double-buffered table `Bw[parity][sample][chunk][pair]`.
  - Each entry is a uint4 = (Re w hi, Re w lo, Im w hi, Im w lo) × 2 points, in mma B-fragment order. Points outside a sample's 10 × 10 are zero.
  - There is one barrier per plane. This build is the "block-shared weight table" of item A1 in the tensor kernel.
- **Contraction:** warp (channel group, split) runs Y[(c, Re | Im)][s] += Σ_p G[(c, Re | Im)][p] · W[p][s] with mma.sync.m16n8k16.
  - An m16 tile is the real parts of 8 channels, then their imaginary parts. A warp holds 2 m16 tiles = its 16-channel group; tiles with no channel are skipped.
  - N is 8 samples, K is 16 union points.
  - Re y = Σ Gr Wr − Gi Wi and Im y = Σ Gr Wi + Gi Wr come from two products (A·Wr and A·Wi) in the same thread.
  - The grid values are read from global memory (L1 / L2), prefetched one chunk ahead, and split in registers.
  - The f32 accumulators stay in registers across all planes. They are written to the shared accumulator plane of the vt epilogue at the end; with 2 splits this is 0 + a + b, the same in either order.
- **Split precision:**
  - `bf16` (VTC 1): hi = rn_bf16(x), lo = rn_bf16(x − hi), products hi·hi + hi·lo + lo·hi, about 2^-16 per product. No scaling is needed (bf16 has the f32 exponent range).
  - `fp16` (VTC 2): the same split in fp16, 11 + 11 bits, about 2^-22 per product. Weights are scaled by 2^8 (|w| < 255). Grid values are scaled by 2^E per (channel, plane), with max |G| · 2^E < 2^15.
  - For fp16, E starts each plane at 40 and is lowered chunk by chunk when larger values appear. That costs a quad max reduction per chunk. The accumulators are rescaled by the exact power of two.
- **Autotune:** with `auto` it also times fp16 at nwc ∈ {min(4, groups), largest}, nws ∈ {1, 2} and sub-chunk ∈ {8, 16, 24}. The fastest of all shapes wins. With `bf16` / `fp16` it times only that kind.
- **Arch gate:** sm_80+ only; no BFP grid slots. Compile-checked for sm_80, sm_89, sm_90 and sm_120 with nvcc 13.0, no spills.

Tests and benchmarks:
- **Test:** `tests/seg_phase2.rs` `p2_vtc`. It compares against the FMA vt of the same channel groups on the same tables, and prints the f32 one-ulp floor (image moved by one ulp) and the vt-vs-tile difference.
- **opbench `--mma-compare`:** forward and gradient times, and the added error of the forward and of the adjoint-of-forward, for fp16 / bf16 × nws × sub-chunk. The baseline is the FMA vt of the active shape (`HASTY_SEG_VS_CFG`).

### 21.3 Added arithmetic error (absolute, relative 2-norm vs the FMA vt)

| problem | fp16 (VTC 2) | bf16 (VTC 1) | f32 one-ulp floor | FMA vt vs tile forward |
|---|---|---|---|---|
| opbench 256³, order 3, 50 channels: forward | 3.0e-6 (nws 1), 1.6e-6 (nws 2) | 4.9e-6 (nws 1), 4.0e-6 (nws 2) | 1.1e-7 | 3.6e-6 |
| same: adjoint of the forward (gradient path) | 2.9e-6, 1.6e-6 | 3.3e-6, 1.7e-6 | 2.3e-7 | – |
| p2_vtc static order 2, 10 × 10, 100 ch | 8.3e-6 | 2.9e-4 | 5.6e-6 | 6.3e-6 |
| p2_vtc knot 1-D order 3, 6 × 5 | 1.3e-5 (nws 1), 1.9e-5 (nws 2) | 5.4e-4 | 1.1e-5 | 1.2e-5 |
| p2_vtc knot 2-D order 2 / 3, 10 × 10, 50 ch | 8.3e-6 / 9.8e-6 (nws 2: 9.3e-6 / 1.2e-5) | 2.9e-4 / 3.6e-4 | 5.7e-6 / 7.2e-6 | 6.5e-6 / 8.3e-6 |

The p2_vtc problems are random synthetic sets. Their segments cancel strongly: the one-ulp floor is about 1e-5, about
100 times the input change. The opbench image is well conditioned: floor about 1e-7. Adjointness against the tile
adjoint was ≤ 2.2e-7 for fp16 and ≤ 2.2e-6 for bf16.

What it means:
- **fp16** adds 1.6–3e-6 on the 256³ problem, below the 1e-5 budget and below the existing vt-vs-tile difference. On the ill-conditioned synthetic sets it adds about 1.5 × the f32 floor.
  - The added error halves with 2 K-chunk splits. This points at the tensor core's f32 accumulation (truncating, so the error grows linearly with the chain length: about 330 MMA steps per sample), not at the 2^-22 split.
  - A per-plane flush of the accumulators into f32 round-to-nearest sums would shorten the chains about 10×. Not built.
- **bf16** fits the budget on the well-conditioned problem (4–5e-6). It misses by 30–50× where segments cancel, so it is not a safe default and `auto` never picks it.

### 21.4 Timing (opbench, 450 W; forward = 2 launches + FFT 81 ms + pad / crop)

| forward kernel | shape | registers | warps / SM | forward ms | gradient ms |
|---|---|---|---|---|---|
| FMA vt (baseline, the autotune's choice) | nwc 4, nws 1, sub-chunk 24 | 254 | 8 | **211.5** | **423.0** |
| FMA vt, one pass | nwc 4, nws 1, sub-chunk 48 | 254 | 4 | 234.7 | 441.8 |
| fp16 vtc | nws 1, sub-chunk 8 / 16 / 24 | 107 / 148 / 146 | 12 / 4 / 4 | 225.3 / 304.4 / 290.1 | 441.1 / 511.7 / 497.7 |
| fp16 vtc | nws 2, sub-chunk 8 / 16 / 24 | 117 / 148 / 162 | 16 / 8 / 8 | 224.8 / 242.9 / 233.7 | 438.8 / 451.0 / 441.6 |
| bf16 vtc | nws 1 / 2, sub-chunk 16 | 120 / 120 | 4 / 8 | 252.6 / **214.3** | 461.2 / **420.7** |

On the 4090 the tensor-core contraction does **not** beat the best FMA vt:
- fp16 is +13 ms per forward (+6.5 ms per launch) at its best shape.
- bf16 is level with it (+2.8 ms forward, −2.3 ms gradient), within run-to-run noise.

The autotune keeps the FMA vt (`HASTY_SEG_VT_MMA` default 0). An early comparison against the one-pass FMA shape
(234.7 ms) had looked like a 10 % win; that was the wrong baseline.

Nsight Compute, one launch, fp16 nws 2, sub-chunk 8, against FMA sub-chunk 24 (§21.1):

| | FMA vt sub-chunk 24 | fp16 vtc nws 2 sub-chunk 8 |
|---|---|---|
| duration (ms) | 65.0 | 74.3 |
| warp instructions executed | 36.1 G | 45.5 G |
| registers / achieved occupancy | 254 / 16.6 % | 117 / 33.3 % |
| issue slots busy | 49 % | 53.7 % |
| tensor pipe active | – | 29 % |
| pipe use (ALU / FMA / LSU) | – | 44.6 / 19.5 / 28.5 % |
| L2 throughput | – | 62 % |
| top stalls per issued instruction | selected 1.0, wait 0.60, not selected 0.45, short scoreboard 0.44 | math pipe throttle 1.86 (ALU), wait 1.40, barrier 1.19, not selected 0.76 |

Why the tensor kernel is still issue-bound, by counting:
- The kernel issues about 1.2 G MMA instructions, 2.7 % of its instructions.
- Converting the grid operand costs about 130 instructions per lane per 16-point chunk. That is 8 loads, 8 hi / lo splits of about 8 instructions each, and the fp16 exponent reduction. This is paid for every (sub-chunk, plane) and amortised over only one n8 tile (6 MMAs per m16 tile) at sub-chunk 8, or 2–3 tiles at 16–24.
- The weight build is about 250 instructions per (sample, plane) and was 4 × redundant in the FMA vt.

Items carry only about 20 samples spread over 25 planes, so about 8 samples are active per plane. The grid
operand's split is therefore reused by too few MMAs. Larger sub-chunks would amortise it, but they need the records
and two weight tables in shared memory: sub-chunk 24 is 92 KB, so 1 block per SM, and the barrier per plane then
serialises build and MMA. That is why sub-chunks 16 / 24 are slower.

### 21.5 What this means for the other GPUs (counting; nothing measured off the 4090)

At equal SM count and clock the vtc kernel needs:
- 1.2 G MMA × 32 cycles per SM sub-partition (4090 fp16 with f32 accumulation: 165 TFLOPS dense, half the fp16-accumulate rate; Ada whitepaper, recalled, **UNVERIFIED**).
- About 44 G other instructions.

The tensor pipe is 29 % busy. So even a GPU with an 8 × faster tensor core per SM (H100 SXM: 989 TFLOPS fp16 dense
with f32 accumulation on 132 SMs at about 1.8 GHz, **UNVERIFIED**) saves at most the part of the 74 ms where MMAs are
on the critical path. The rest is issue-bound and scales with SMs × clock, as the FMA vt does.

Expected gains:
- **H100 / A100:** at best about 1.3 × over the FMA vt, not the 5–10 × that rates alone suggest, until the operand conversion and the weight build stop dominating the issue.
- **5090:** same per-SM tensor rate as the 4090 (GeForce half-rate f32 accumulation; **UNVERIFIED**), so no gain.

This replaces the old arch_targets.md §3.5 estimate, which assumed an FMA-bound kernel.

### 21.6 ra adjoint: balanced staging (`RaVar.stages` 3 / 4; a sub-agent of this session)

What was built (files `segnufft_ra.cu`, `segnufft/ra.rs`, `tests/seg_phase2.rs`):
- **`stages 3`** builds on the cp.async in-place pipeline (stages 2).
  - Each chunk's staging loops are cut into 32-item warp units, heaviest units first. Warps take them round robin.
  - The coil-value stage is split into 2 sample slices per channel, so it spreads over all warps.
  - The MAC loads the next sample's box one sample ahead.
- **`stages 4`** additionally fuses the MAC of chunk k − 1 between the same barriers as staging phase 1 of chunk k: two barriers per chunk instead of three.
- The previous variants are unchanged: their preprocessed source and nvcc cubins are byte-identical.
- **Selection:**
  - One kernel: `HASTY_SEG_VSA_CFG=nxg,cb,chk,1,3,lbmin` (or `,1,4,`).
  - Autotune: `HASTY_SEG_VSA_STAGE` = cpasync (the previous candidates) | bal (+ stages 3) | dyn (+ stages 3, 4).
  - The default `auto` now means `bal`. This only adds autotune candidates; set `cpasync` to get the previous candidate set back.
  - An atomic-counter work grab (`HASTY_SEG_RA_DEFS=RA_GRAB=1`) measured 7–16 % slower and is not in the autotune.

Tests:
- Without split lists: 1300 / 1300 runs bitwise equal to the base ra, orders 0–3, static and knot engines, zero mode tiles and full.
- With split lists: at most 0.27 of the bound (6 × one-ulp floor, 4 × run-to-run spread).

Timing (opbench autotune times, knot 0, but at coil batch 2 = 20 channels; opbench picked that batch without `--max-grid-gb 14`):
- Best ra: 39.43 ms (stages 2, nxg 4, cb 3, chk 12) → 36.71 ms (stages 3, chk 8), −6.9 %. ptxas also gave the chk 8 shape 72 registers, so 28 warps instead of 20.
- At equal occupancy, balancing alone is −1 % … +0.7 %, and fusion is ± 1 %. Fewer barriers did not translate into time: the latency-bound MAC header is the limiter. That header is about 38 instructions per sample against 116 MAC instructions.
- The 50-channel shape (nxg 4, cb 7, chk 14; stages 3 / 4 compile to 124–125 registers, no spill) is unmeasured; it is in the campaign.

### 21.7 Not built, and next

- **A1 in the FMA vt** (block-shared weight table) and **A2** (≤ 128 registers through a shared-memory window plane, precomputed offsets): not built.
  - The vtc kernel shows the cost of the block-wide build. With a barrier per plane and 1–2 blocks per SM, the build and the contraction serialise.
  - An FMA variant should build the next plane while the current one is contracted (producer / consumer warps, or one plane of lookahead with mbarriers) rather than adding a barrier.
- **For the tensor path to win anywhere**, the grid operand's split must be reused by more MMAs:
  - (a) Store the forward z pass's output already split (bf16 hi / lo = 4 bytes per real, the same bytes as f32). The kernel then loads fragments directly with no conversion, at the bf16 accuracy of §21.3.
  - (b) Larger items with the records for all samples resident, so one split serves every active sample of a plane.
  - (c) A per-plane f32 flush of the accumulators, to shorten the truncating accumulation chains.
- **The adjoint as a GEMM** (G[c, p] += Σ_s Y[c, s] W[s, p]) has K = active samples per plane, about 8. That is below one k16 step, so it is not worth building at this sample density.

### 21.8 Campaign (prepared, not started)

The campaign lives in research/experiments/tensor_interp/: `start.sh` (runs `freeze.sh` once: copies opbench and the
test executables), `stop.sh`, `resume.sh`, `cases.txt`, and unit tic-campaign. Each case runs under the GPU lock with
timeout 900 and logs the power limit. About 75 min at 450 W:
- **Regression, 8 cases:** p2_vtc, p2_vs_subchunks, the ra variants, kl / autotune / spill, seg_vs, seg_ra_vt, seg_kl, seg_split.
- **opbench `--mma-compare`** at 256³ order 3, 192³ order 3 and 256³ order 2.
- **One full autotune** with `HASTY_SEG_VT_MMA=auto` and `HASTY_SEG_VSA_STAGE=dyn`, logging every candidate's kernel time.
- **ra at 50 channels:** stages 2 / 3 / 4, nxg 4, cb 7, chk 14.
- **Three ncu launches**, with SourceCounters for per-line attribution: FMA vt, fp16 vtc, bf16 vtc.

Dev logs: scratchpad `vtc/` (t3.log tests, ob2.log opbench, n_*.ncu-rep).

### 21.9 Short campaign result (2026-11-21, research/experiments/tensor_interp/, trimmed case list)

**What ran.** Only the regression tests and the ra timing at 50 channels. The case lines of the mma comparisons and the ncu runs are prefixed `#SKIP` in `cases.txt`; the full list is `cases_full.txt`.

**Regression.**
- These passed: p2_vtc, p2_vs_subchunks, p2_kl / p2_autotune / p2_spill, seg_vs 5/5, seg_ra_vt 8/8 (ra_static_orders passes here), seg_kl 8/8 and seg_split 4/4.
- p2_ra (new variants) hit the 900 s timeout while still running. p2_ra_knot passed, and p2_ra_static had logged about 1470 checks with no failure, bitwise equal to the base ra without split lists.
  - It is too long for one case; split it before the next campaign.

**ra at 50 channels** (256³, one set of 1.52 M samples, order 3, 10 coils × 10 segments, coil batch 5, nxg 4, cb 7, chk 14). The table gives the ra launch time in the autotune.

| ra staging | launch | adjoint per gradient |
|---|---|---|
| stages 2 (previous cp.async double buffer) | 53.05 ms | 182.4 ms |
| stages 3 (balanced) | 58.42 ms (+10 %) | 183.9 ms |
| stages 4 (balanced + fused MAC) | 54.83 ms (+3 %) | 184.0 ms |

- The −6.9 % measured at 20 channels (§21.6) does not carry over to 50 channels. There, the previous staging is fastest.
- The variants stay as autotune candidates (the autotune drops them where slower), at the cost of a longer autotune until the per-GPU tune cache exists.
- With stages 3 / 4 the knot-path layout picked the whole-tile kl kernel (48.2 ms) over the engine adjoint, so the adjoint per gradient barely moved.
- These runs pinned the forward to `HASTY_SEG_VS_CFG=4,1,0` (the one-pass shape), so their forward times (235 ms) are not the default's.

## 22. Warp-specialised forward and adjoint, and the z-pass / gather fusion count (2026-11-21, thirteenth session)

Question: can the two interpolation kernels of one gradient (forward vt ≈ 2 × 55–57 ms, adjoint ra ≈ 2 × 53–56 ms at
the shape below) get faster by splitting the warps of a block into producers and consumers? And is fusing the
forward's last FFT pass (z) with the gather worth building? Added arithmetic error allowed: ≲ 1e-6 relative (these
are pure re-orderings in f32 FMA). W 10, σ 1.25, f32 storage and accumulation unchanged.

The case for every number below (as §21):
- **Data:** one sample set (frame 4, encode 2 of flow_thorax) with 1 516 200 samples.
- **Operator:** order 3 with segments, matrix 256 × 256 × 256, fine grid 320 × 320 × 320.
- **Channels:** 10 coils × 10 segments; coil batch 5, so 50 channels per launch and 2 launches per forward.
- **Run:** opbench `--models seg-full --order 3 --sets 1 --matrix 256,256,256 --cb-cap 5 --max-grid-gb 14`, RTX 4090 at
  the user's 450 W limit (not changed). Kernel times are the autotune's (best of 3 launches).
- **Baselines in the same runs:** FMA vt nwc 4, sub-chunk 29 (the autotune's pick here): 54.5–57.1 ms, 254 registers,
  8 warps / SM, 9 616 SASS instructions. ra nxg 4, cb 7, chk 14, cp.async (stages 2): 53.1–56.4 ms, 124 registers,
  16 warps / SM.

Verdict first: **on the 4090 neither warp-specialised kernel beats the kernel it replaces**, and the fusion does not
fit any L2. Everything built is bitwise equal to the kernel it restructures, stays selectable, and the autotune keeps
the old kernels (it never picked a new one here).

### 22.1 Forward `vw` (VW 1): build once per block, plane window in shared memory, consumers read shared memory only

Files: `segnufft_vs.cu` (`#if VW == 1`), host `segnufft/vs.rs` (`VwCfg`, `vw_ok`, `vs_compile_w`, `set_vs_w`,
`vw_candidates`), test `tests/seg_phase2.rs` `p2_vw`, opbench `--ws-compare`.

Block = one item sub-chunk × the channel row (all 16-channel slices of the batch). Three roles:
- **Build warps** (`nbw`): per plane, each active sample's 10 × 10 complex weight table, built **once per block** with
  vt's own `vt_build`, into a ring of `nwt` tables (≥ the sub-chunk). The vt kernel built it once per channel group,
  4 × at 50 channels.
- **Copy warps** (`ncp`): per (plane, 16-channel slice) the union window (13 × 13 points × 16 channels = 21.6 KB)
  into one of `nst` shared stages by cp.async. The wrapped point offsets are computed once per item.
- **Consumer warps** (`ncw`): lane = (channel, row half) as in vt. Every grid value and weight is read from shared
  memory at compile-time offsets, so there is no register window: **115–141 registers** (vt: 254), **3.0 k SASS
  instructions** (vt: 9.6 k). Sample s is always contracted by consumer s mod ncw (no accumulator races).
- Items are re-sorted by oz first (as vtc), so the samples active on a plane are a contiguous range.
- Per sample the planes, rows and points are summed in vt's order: **the result is bitwise vt's** (p2_vw: 39 / 39
  shapes bitwise on static order 2 and knot orders 2 / 3, odd channel counts, rings larger than the sub-chunk).

Hand-over, in the order it was built (time per launch at the 50-channel shape, best shape):

| hand-over | ms | what limited it (Nsight Compute, one launch) |
|---|---|---|
| mbarrier (try / test_wait), 1 copy warp | 93.7–97.7 | `mbarrier.test_wait` on sm_89 is a shared-memory poll: 8.1 G of 22.7 G shared wavefronts were spin loads, plus NANOSLEEP / BRA / LOP3 = 35 % of the instructions; shared pipe at 103 % |
| + 2–3 copy warps, 64 ns back-off | 80.1 | still polling; mio throttle 1.62 |
| named barriers (`bar.arrive` / `bar.sync`, no polling) | 67.6 | 29.9 G instructions (vt 36.1 G); 15.0 G shared wavefronts (65 % of the pipe); barrier stall 7.8 per issued instruction: starved |
| copies published at once (no lag of one slice) | **62.7** (other runs 62.8, 67.4) | the window pipeline is only 2 stages deep |

Why it cannot reach vt here, by counting (per launch, 60.6 M (sample, plane, 16-channel slice) units):
- A consumer unit reads 50 LDS.64 grid values and 25 LDS.128 weights; with two row halves per warp each is 2 shared
  wavefronts: **150 wavefronts per unit, 9.1 G per launch = 28 ms** at one wavefront per clock per SM.
- The cp.async window copies cost 3.76 G wavefronts (12.4 per instruction against 4 ideal: with 50 channels a grid
  point is 400 B, so a slice's 128 B straddle two 128-B lines) = 12 ms more. Shared-memory floor ≈ 45 ms.
- vt avoids most of this by keeping its window in registers and reusing it across the ≈ 2 samples of a window step
  and 5 steps, at the price of 254 registers.
- The block needs 97.8 KB at sub-chunk 24 (records 21.5, accumulators 12.8, table ring 19.2, two window stages 43.3):
  one block per SM, and no room for a third 21.6 KB stage. With two stages the consumers wait for the copies (barrier
  stall 7.8). Smaller sub-chunks free shared memory but repeat the window loads: sub-chunk 16 68.8 ms, 12 (3 stages)
  74.0 ms.

What would change it: 228 KB parts (H100 / A100 / H200) take 2 blocks or 4 stages, and L2 delivery there is dearer
than on the 4090 (vw reads 13 window rows per plane once, vt reads 16 rows at 50 of 64 lanes, L1 hit 1–6 %). Not
measured. `HASTY_SEG_VT_WS = auto` (default) keeps three vw shapes in the autotune for that reason (≈ 1 s of autotune,
plus ≈ 20 s of NVRTC per shape on a cold kernel cache); `0` drops them, `1` times only vw shapes.

### 22.2 Forward `vw 2`: only the build shared, vt's register window kept

Same files (`#if VW == 2`). The vt channel-group warps keep their register window and contraction; `nbw` build warps
fill a ring of tables one plane ahead; a per-plane byte map gives each active sample its ring slot. Bitwise vt.

| shape (50 channels) | registers | warps / SM | ms |
|---|---|---|---|
| 1 build warp, sub-chunk 24 / 20 / 16, `__launch_bounds__(160, 2)` | 168, 36 B spill | 5 / 10 / 10 | 101.2 / 109.6 / 118.2 |
| 1 build warp, `__maxnreg__(200)` | 200, no spill | 5 (one block) | 80.3 / 87.8 / 92.0 |
| 2 build warps | 168, spill | 6 / 12 / 12 | 97.5 / 106.2 / 114.7 |

Why: without the build the window still needs ≈ 200 registers, and the 4090's register file is per scheduler
(16 K each): 200-register warps fit 2 per scheduler, so a 5-warp block runs alone on the SM (5 warps). Two blocks need
≤ 168 registers, and then the window spills inside the step loop. The 4 × redundant build that vt pays costs less than
either. Only with `HASTY_SEG_VT_WS=all` (or `HASTY_SEG_VW_CFG=nwc,0,nbw,0,sch`).

### 22.3 Adjoint ra, warp-specialised (`RaVar.stages` 5 / 6 = 4 / 2 producer warps)

Files: `segnufft_ra.cu` (`RA_WS`), host `segnufft/ra.rs` (`RaVar::ws`, `threads`, `ws_stages`), tests `p2_ra_*` with
`P2_RA_VARS=ws`.
- The 4 MAC warps (one per tile plane, accumulators in registers) are unchanged.
- 4 (or 2) producer warps run the whole staging of chunk k + 1 (cp.async raw copies, kernel values, boxes, w_l, z,
  weights) while the MAC warps work on chunk k. The MAC inputs (boxes, z, weights; order 0 kernel values) are
  double-buffered. Hand-over by named barriers (full / free per parity); the producers' own phases use a producer-only
  named barrier. No block barrier in the chunk loop.
- The old variants' preprocessed source is identical (stages 1–4 checked with `nvcc -E`).
- Bitwise equal to the base ra without split lists. With split lists the difference is within the base's floors, as
  every other variant: knot orders 2 / 3, 342 checks (`p2_ra_knot_o23`, `P2_RA_VARS=ws`, 8.9 min; log
  experiments/warp_spec/dev_t03_p2_ra_ws_knot_o23.log). Orders 0 / 1 are campaign case t02.
- `__launch_bounds__(256, 2)`: 124–128 registers, no spill on sm_80 / 89 / 90 / 120. The 2-producer shape needs
  ≈ 160 registers (3 blocks spill at 96).

| ra at 50 channels (nxg 4, cb 7) | registers | warps / SM | shared KB | chk 8 / 12 / 14 / 16 (ms) |
|---|---|---|---|---|
| stages 2 (cp.async, the autotune's choice) | 124–125 | 16 | 14–28 | 60.8 / 57.6 / **56.4** / 64.7 |
| stages 5 (4 producers) | 128 | 16 | 22–44 | 110.9 / 106.5 / 101.1 / 101.0 |
| stages 6 (2 producers) | 157–161 | 12 | 22–44 | 141.3 / 122.7 / 120.7 / 113.9 |

Why 1.8 × slower, by counting per chunk of 14 samples:
- The MAC is ≈ 154 instructions per (sample, plane warp): ≈ 2.2 k per MAC warp per chunk.
- The staging is ≈ 57 k thread instructions (estimated from the loop bodies; the weights of the covered (sample, row)
  pairs are ≈ 60 % of it): ≈ 450 per warp when 4 producer warps share it.
- So the MAC is ≈ 80 % of the work. Keeping 16 warps per SM at 128 registers means 2 blocks of 4 MAC + 4 producer
  warps: **8 MAC warps per SM instead of 16**. The MAC is latency-bound (issue 61 % with 16 warps), so halving its warps
  nearly doubles the time, while the producers idle.
- The barrier stall the brief targeted (1.61 of 6.5 cycles per instruction) is real, but the 4 co-resident blocks of
  the old kernel already overlap each other's staging and MAC.
- On sm_89 every warp gets the kernel's register count (no `setmaxnreg`), so a producer warp costs a MAC warp.
- `HASTY_SEG_VSA_WS = 0` (default: never timed) | auto | 4 | 2. The 50-channel shape decided this, as asked: the
  20-channel shape had misled §21.6.

### 22.4 Fusing the forward z pass with the gather: counted, not built

What fusion could save: the forward z pass writes the whole fine grid. That is 320³ × 50 channels × 8 B = 13.1 GB per
coil batch, ≈ 15 ms of its 27.3 ms (§18.4: z pass at 868 GB/s). The gather reads it once more, ≈ 12–13 GB from DRAM
(VW profile: DRAM 16 % busy). But the gather is not DRAM-bound (55 ms for 13 GB), so only the write counts: at best
≈ 2 × 15 = 30 ms per gradient, 7.7 % of 391 ms.

What it needs: a gather item (4 × 4 × 16 home tile) reads a 13 × 13 × 25 window. A z line can only be transformed
whole (320 points), so the gather of a tile can start only when all 13 × 13 = 169 z columns of its halo are done.
- **L2-slab fusion (z output kept in L2 until gathered):** a tile row needs 13 rows of complete z columns. One row of
  the 320 × 320 × 320 grid at 50 channels is 320 × 320 × 50 × 8 B = 41 MB, so 13 rows are **533 MB**. L2 is 72 MB
  (4090), 96 MB (5090), 50 MB (H100). Even one channel takes 10.6 MB for 13 rows, so a slab fits 6 / 9 / 4 channels.
  The gather would then run in 8–12 channel passes. Each pass rebuilds the per-(sample, plane) weights (vt: ≈ 250
  instructions each, 15.2 M per pass), which costs more than the whole saving.
- **z-column blocks with halo recompute:** blocks of B × B z columns transformed per block, gathering the tiles fully
  inside. Recompute factor B² / (B − 9)², with the block's output (B² × 128 KB) in L2.
  - B = 23 (4090): 2.7 ×. B = 27 (5090): 2.25 ×. B = 19 (H100): 3.6 ×.
  - The extra z-pass reads (8.4 GB of y-pass output per batch × 1.25–2.6) cost as much as or more than the 13.1 GB
    write saved.
- **Bands of B tile rows with all SMs on one band:** the halo overhead is (4B + 9) / 4B, i.e. 1.28 × at B = 8, for a
  resident set of (4B + 9) rows × 13 columns × 128 KB = 68 MB.
  - But 128 SMs work on different tiles at once. Spread over the band, they cover 128 / B tiles of x, so the resident
    set is 41 rows × 73 columns × 128 KB = 383 MB.
  - Keeping all SMs inside a compact xy region needs a persistent kernel with global dependency tracking (z columns
    done → tiles ready) and load balance across very uneven items (k-space centre). Even if all that worked, it saves
    at best 5–6 % on the 4090.
- `discard.global.L2` (chanfft.rs) helps only when the reader comes while the lines are still resident. The whole-grid
  z pass finishes long before the gather starts.

Verdict: below the ≈ 5 % bar on every listed GPU once the halo or channel passes are paid, and much more complex than
anything else on the list. Not built.

### 22.5 Tests and the over-long p2_ra

- `seg_vs` 5 / 5 pass with vw in the default autotune (274 s on a cold kernel cache: NVRTC of the vw shapes; 36 s before).
- `p2_vw` (new, 83 s): vw 1 and vw 2 against the FMA vt, bitwise, plus adjointness against the tile adjoint ≤ 1.2e-7.
- `p2_ra_*` with `P2_RA_VARS=ws`: the warp-specialised ra variants only.
- p2_ra split by order: `p2_ra_static_o0`, `_o1`, `_o2`, `p2_ra_knot_o01`, `_o23` (run with `--exact`). The old
  `p2_ra_static` / `p2_ra_knot` stay as all-order wrappers. The 900 s timeout hit p2_ra_static and p2_ra_knot running
  together (§21.9). One part measured: knot o23 with only the ws variants, 8.9 min; the full variant list per part
  is not yet timed.
- vt / vs / vtc / ra stages 1–4 sources: unchanged after preprocessing; the vt order-3 SASS is identical.

### 22.6 Added error

Every new kernel is bitwise equal to the kernel it restructures on the tests (campaign o01 / o02 print the same check,
the bitwise flag and the relative difference, on the 256 × 256 × 256 and 192 × 192 × 192 problems).
- Added error is therefore 0 against the FMA vt / base ra. For comparison, the FMA vt vs tile forward difference is
  3.6e-6 and the f32 one-ulp floor is 1.1e-7 (§21.3).
- Split-list ra runs differ only by atomic order, within the base ra's own run-to-run spread.

### 22.7 Campaign (prepared, not started): research/experiments/warp_spec/

`start.sh` (runs `freeze.sh` once: copies opbench, seg_phase2, seg_vs), `stop.sh`, `resume.sh`, `cases.txt`, unit
wsp-campaign. Each case runs under the GPU lock with timeout 900 and logs the power limit. About 30 min:
- **Regression, 3 cases:** p2_vw; ra-ws on knot orders 0 / 1; seg_vs (the default autotune now includes the vw
  shapes). Knot o23 ran in dev; static o2 is listed but skipped (over budget).
- **opbench `--ws-compare`** at 256³ and 192³. The FMA vt (4,1,0, sub-chunk 29) and the ra (4,7,14, cp.async) are
  pinned; every vw / vw 2 / ra-ws shape is compared against them (forward / adjoint / gradient ms, added error, bitwise
  flag).
- **One full autotune** with every new candidate logged (`HASTY_SEG_VT_WS=all`, `HASTY_SEG_VSA_WS=auto`, spill
  screening off).
- **Two ncu launches** (targeted sections, SourceCounters): vw 1 (10 / 3 / 3, sub-chunk 24) and ra stages 5.

### 22.8 Open

- vw 1 on a 228 KB part: 2 blocks or 4 window stages; and the cp.async line straddle, which a 128-byte-aligned channel
  stride would remove. A grid-layout change: not done.
- ra with fewer registers per producer needs `setmaxnreg` (sm_90a / sm_120a).
  - With it, 16 MAC warps + producers fit only if MAC warps keep 128 and producers drop to ≈ 40 registers: 16 × 128 +
    4 × 40 ≈ 69 K > 64 K, so even then it is 12 MAC warps.
  - A persistent ra (producers prefetch the next item) does not change that count.

**Tune cache (2026-11-22, research/arch_targets.md item 0).** The autotune's choices (TileCfg, vs / vt / vw, ra +
RaVar, knot-path layout / kl, multi-image shapes) are cached per GPU in `rustlib/target/hasty-tune/<gpu>/<key>.json`,
keyed by engine shape + `HASTY_SEG_*` + kernel-source hash + table-density bucket; `HASTY_TUNE = reuse | retune | off`.
On a hit no candidate is timed: 59 s → 0 s per engine on the 4 mm velocity case (123 / 306 s at 2 mm). After a kernel
edit, run once with the default (a miss) or `HASTY_TUNE=retune` before timing campaigns.

