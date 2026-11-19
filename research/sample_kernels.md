# Sample-side kernels of the segmented NUFFT: exact-support designs (prototype, 2026-11-19)

Status (2026-11-19; §12 = fourth session: design 6 `vs`, value-stationary rows: 1.4–1.9× ps, 1.9× the production knot forward at 4 M 10 × 10 o0, 25–30 % of peak; order 2 no win; sanitizer clean. §11 = third session: limiter measured): **design 5 (`ps`, plane-streamed home tile) written, compiled (nvcc sm_89: no
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

