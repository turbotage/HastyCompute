# Sample-side kernels of the segmented NUFFT: exact-support designs (prototype, 2026-11-19)

Status (2026-11-19, second session; §11 = third session: limiter measured): **design 5 (`ps`, plane-streamed home tile) written, compiled (nvcc sm_89: no
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
  40 useful. Order 2 wins only if the weight is shared across channels (§12.8).
