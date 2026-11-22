# Architecture targets for the gradient kernels: facts, per-GPU settings, phase-2 list (2026-11-20)

Status 2026-11-20: phase 1 complete (facts, mapping, recommendations, ranked list, full compile audit with what-if
variants, first-hour draft). Rerun the audit after phase-2 kernel changes: research/experiments/arch_audit/run.sh.
Phase 2, items 2 and 3: **built, tested, measured on the 4090, audited on sm_80 / 89 / 90 / 100 / 120**
(research/sample_kernels.md §19; status per item in §4 below). Items 0, 1, 4–13: not started.

Phase 1 of the multi-GPU-architecture effort: study and offline audit only. No GPU was used. Every kernel number for a
card other than the RTX 4090 is a projection or a compile-time count until it is measured on that card.

Question: which settings and kernel designs of the segmented-NUFFT gradient (forward and adjoint interpolation plus the
grid FFTs) should change per GPU architecture, and what should be built and tested on the 4090 before cluster hours are
spent? Decision: the ranked phase-2 implementation list (§4) and the first-hour procedure for a new GPU (§6).

Inputs: research/sample_kernels.md §10–§17 (kernel designs and limiters), research/temporal_operator.md §20 and §23
(FFT side), research/memory_ledger.md, research/plan_cluster.md, the engine sources (crates/hasty-nufft/src/segnufft*.cu,
segnufft/*.rs, crates/hasty-fft), and an offline compile audit of every production kernel family at its autotune
candidates for sm_80, sm_89, sm_90, sm_100 and sm_120 (§5; tool `kernaudit`).

## 0. Answer

1. **Registers per SM are 64 K on every target**, and our kernels are latency-bound at 4–16 warps per SM
   (sample_kernels.md §11.5, §12.7, §15.6, §17.7). Hopper and Ampere raise the thread limit (2048 per SM) and shared
   memory (228 KB / 164 KB per SM against 100 KB), but not the register file. The audit (§5) shows that the shapes the
   velocity reconstruction runs are **shared-memory-limited on the 100 KB parts** (4090, L40S, 5090): the ra adjoint
   (nxg 4, cb 7, chk 16: 128 registers, 30 KB) holds 3 blocks = 12 warps where its registers allow 16; the vt forward
   (4 × 4 × 16, SMAX 48: 254 registers, 76 KB) holds one block = 4 warps. On sm_90 the same kernels hold 16 and 8 warps.
   So on Hopper the larger shared memory does buy occupancy for these two kernels (+33 % and 2× warps), and on the
   100 KB parts a smaller shared-memory footprint is a lever that can be built and measured on the 4090 (§4 item 2).
2. **For latency- and issue-bound kernels, SM count × clock is the first-order scale**, and on that scale the datacentre
   Hopper parts are *slower* than a 4090 (H100 SXM ≈ 0.81×, H100 PCIe ≈ 0.62×, A100 ≈ 0.47× and ≈ 0.24× where FMA-bound,
   5090 ≈ 1.27×, L40S ≈ 1.11×; §1.4). The FFT side, which is DRAM-bound, scales with bandwidth (H100 SXM 3.3×, H200
   4.8×). Projected gradient of the velocity reconstruction's 2-D knot case at 13.3 M samples per encode: about the same
   on an H100 as on the 4090, ≈ 1.4× faster on a 5090 (§2.1). This is a hypothesis; the first cluster hour must measure it
   (§6).
3. **What the large cards really offer is VRAM and shared memory.** The measured lever on the knot path is the coil batch
   (5 → 10 coils: −16 % of the dense gradient, sample_kernels.md §17.2). On the 800×480×480 grid (1 mm torso) one coil's knot slot is 14.7 GB, so a
   24–48 GB card runs coil batch 1 (L40S 1–2), an 80 GB card 3, a 141 GB H200 5 (§2.3). Shared memory: on sm_90 every ra / vt / kl
   candidate fits; ra / kl item lists of 2048–4096 (fewer split tiles and atomics) cost no occupancy there, while
   chunks of 48–64 samples and vt items above SMAX 64 halve the warps (§5.3).
4. **Ranked phase-2 list (§4):** a per-GPU tune cache and the coil batch from the ledger (enablers, all cards); a smaller
   shared-memory footprint and cp.async staging in ra / kl (staging is ≈ 55 % of a knot launch; 12 → 16 warps on the
   100 KB parts); per-arch spill screening plus a per-shape
   launch-bounds candidate (the audit shows spills are arch-specific, and a per-shape hint removes all ra / kl spills); a shared-plane vt/vs forward
   (cp.async on sm_80+, TMA on sm_90); block-floating-point storage of the FFT intermediates (−19 % to −39 % of the FFT-side
   DRAM bytes, most useful on 4090 / 5090 / L40S); channel sub-slabs for the L2 two-pass FFT; then the sm_90-only items
   (vector float2 atomics, cluster/DSMEM reduction of split tiles, TMA) that can only be compile-checked here.
5. **Compile audit (§5):** 5611 kernels (six engine cases × every autotune candidate × five architectures) compile
   without errors. Spills are arch-specific: the same source spills on one arch and not another (ra nxg 2 cb 3 for 48
   coils × L 1 spills on sm_89 and sm_90 only; the tile forward has 57 spilling candidates on sm_80, 35 on sm_120, 5 on
   sm_89). So spilling candidates must be detected and skipped per card, not per source. Every ra and tile kernel carries
   a 32-byte stack frame and every kl kernel 48 bytes without spilling; vs / vt and the FFT passes have none. A
   blanket launch-bounds minimum-blocks hint adds spills; chosen per shape it removes every ra / kl spill on every arch
   (§5.4).

## 1. Architecture facts

Sources are numbered at the end (§8). [n] are NVIDIA documents, [Sn] secondary. **UNVERIFIED** = no NVIDIA source
found. "Derived" = SMs × FP32 lanes × 2 × boost clock.

### 1.1 Products

| GPU | CC | SMs | base / boost MHz | FP32 TFLOPS (non-tensor) | L2 | DRAM | bandwidth | host link | GPU-GPU |
|---|---|---|---|---|---|---|---|---|---|
| RTX 4090 (AD102) | 8.9 | 128 [25] | 2230 / 2520 [25][27] | 82.6 [25] | 72 MB [25] | 24 GB GDDR6X | 1008 GB/s [25] | PCIe 4 x16 (measured here: 21.1 / 19.8 GB/s pinned h2d / d2h) | none |
| RTX 5090 (GB202) | 12.0 | 170 [26] | 2010 / 2407 [26][28] | 104.8 [26] | 96 MB [26] | 32 GB GDDR7 | 1792 GB/s [26] | PCIe 5 x16 | none (no P2P on stock driver, plan_cluster.md §8.2) |
| H100 SXM5 80 GB | 9.0 | 132 [17] | — / 1980 **UNVERIFIED** [S1] | 67 [18][19] | 50 MB, two partitions [6][17] | 80 GB HBM3 | 3.35 TB/s [18] | PCIe 5 x16 | NVLink 4, 900 GB/s [18] |
| H100 PCIe 80 GB | 9.0 | 114 [17] | 1125 / 1755 [20] | 51 [19] | 50 MB [17] | 80 GB HBM2e | 2.0 TB/s [20] | PCIe 5 x16 [20] | bridge 600 GB/s [19] |
| GH200 (96 GB HBM3) | 9.0 | 132 **UNVERIFIED** [S6] | not found | 67 [22] | not found (H100: 50 MB) | 96 GB HBM3 (144 GB HBM3e variant) [22][23] | 4.0 TB/s (4.9 HBM3e) [22] | NVLink-C2C to Grace, 450 GB/s per direction, coherent [23]; ≈ 250 GiB/s measured on Arrhenius (plan_cluster.md §1) | NVLink 4 in NVL2/32 |
| H200 SXM 141 GB | 9.0 | 132 **UNVERIFIED** [S7] | — / 1980 **UNVERIFIED** [S7] | 67 [21] | 50 MB **UNVERIFIED** [S7] | 141 GB HBM3e | 4.8 TB/s [21] | PCIe 5 | NVLink 900 GB/s [21] |
| A100 SXM 80 GB | 8.0 | 108 [15] | — / 1410 [15] | 19.5 [15][16] | 40 MB, two partitions [15] | 80 GB HBM2e | 2039 GB/s [16] | PCIe 4 x16 | NVLink 3, 600 GB/s |
| A100 40 GB | 8.0 | 108 [15] | — / 1410 [15] | 19.5 [16] | 40 MB [15] | 40 GB HBM2 | 1555 GB/s [16] | PCIe 4 x16 | 600 GB/s (SXM) |
| L40S 48 GB | 8.9 | 142 (derived from 18176 cores) [29] | 1065 / 2520 [30] (brief seen via a mirror) | 91.6 [29] | **UNVERIFIED** (L40, same die: 96 MB [25]) | 48 GB GDDR6 ECC | 864 GB/s [29] | PCIe 4 x16 | none |

### 1.2 Per-SM limits by compute capability [1] (CUDA Programming Guide v13.5)

| limit | 8.0 (A100) | 8.9 (4090, L40S) | 9.0 (H100, H200, GH200) | 10.0 (B200, reference) | 12.0 (5090) |
|---|---|---|---|---|---|
| resident threads / warps / blocks per SM | 2048 / 64 / 32 | 1536 / 48 / 24 | 2048 / 64 / 32 | 2048 / 64 / 32 | 1536 / 48 / 24 (Blackwell tuning guide: 32 [9]) |
| 32-bit registers per SM / per block / per thread | 64 K / 64 K / 255 | same | same | same | same |
| FP32 lanes per SM [5] | **64** | 128 | 128 | 128 | 128 |
| L1 + shared per SM | 192 KB | 128 KB | 256 KB | 256 KB | 128 KB [1][26] |
| max shared per SM / per block (opt-in) | 164 / 163 KB | 100 / 99 KB | 228 / 227 KB | 228 / 227 KB | 100 / 99 KB (tuning guide says 128 KB per SM [9]) |
| reserved shared per block | 1 KB (all sm_80+) [5–9] | | | | |
| carveout steps (KB) | 0–164 | 0, 8, 16, 32, 64, 100 | 0–228 | 0–228 | 0, 8, 16, 32, 64, 100 |
| thread-block clusters / DSMEM | no | no | yes, 8 portable, 16 non-portable [6][17] | yes | listed as supported [1]; community reports conflict (1 × 1 × 1 in CUTLASS sm_120 kernels, cluster 4 working on a 5090 laptop) [S8–S11] |

The audit (§5) uses exactly these limits for its occupancy column; the conflicting CC 12.0 values (24 vs 32 blocks, 100
vs 128 KB) do not change any result here (no kernel of ours is block-count-limited, and 99 KB per block is the opt-in
either way).

### 1.3 Features (PTX ISA [10], Programming Guide [1]; compile-checked with nvcc 13.0 on this machine)

| feature | sm_80 | sm_89 | sm_90 | sm_120 | use for us |
|---|---|---|---|---|---|
| cp.async (LDGSTS) + mbarrier | yes | yes | yes | yes | staging of ra / kl / vt chunks without prefetch registers |
| TMA bulk copies (`cp.async.bulk`, `memcpy_async_tx`) | no | no | yes | yes | plane / tile loads of the forward |
| clusters + distributed shared memory | no | no | yes | compiles; runtime unverified | split-tile reduction in the adjoint |
| `atomicAdd(float2*)`, `atomicAdd(float4*)` (global, per-component atomic) | no | no | yes | yes | complex atomics in split tiles and the fused crop: one RED instead of two |
| bf16 atomics | no | no | yes | yes | — |
| `redux.sync` integer max | yes | yes | yes | yes | BFP block exponent: max of the float bits of \|x\| |
| `redux.sync.f32` | no | no | no | no (sm_100a only) | — |
| `setmaxnreg` (warp-specialised register split) | no | no | `sm_90a` only | `sm_120a` only | producer / consumer warps; needs an "a" target in our NVRTC arch string |
| `wgmma`, `tcgen05` / TMEM | no | no | wgmma sm_90a | neither | not used (§3.10) |
| L2 persistence / access-policy window | yes | yes | yes | yes | FFT slabs; not needed so far (temporal_operator.md §23.5) |

Compile check (nvcc 13.0, `-cubin`, CPU only; test sources in the session scratchpad): float2/float4 atomicAdd fails on
sm_80 / sm_89, builds on sm_90 / sm_100 / sm_120; cp.async builds on all; cluster + DSMEM and TMA bulk copy fail on
sm_80 / sm_89 and build on sm_90 / sm_100 / sm_120; setmaxnreg builds only for sm_90a and sm_120a; redux.sync.max.abs.f32
fails on every target tested (it is sm_100a-only).

Global atomics: executed in L2 on all parts [14]; no NVIDIA document quantifies a change in float-atomic throughput
from Ada to Hopper or Blackwell. Our own number is the 4090's: 190–310 G float RED/s (sample_kernels.md §15.1);
`proto_gather --mode redbench` measures it on each new card (§6).

### 1.4 Derived ratios that decide the balance (relative to the 4090)

| GPU | SMs × boost (rel.) | FP32 (rel.) | DRAM BW (rel.) | FP32 FLOP per DRAM byte | L2 per SM | VRAM | shared / SM |
|---|---|---|---|---|---|---|---|
| RTX 4090 | 1.00 | 1.00 | 1.00 | 82 | 0.56 MB | 24 GB | 100 KB |
| RTX 5090 | 1.27 | 1.27 | 1.78 | 58 | 0.56 MB | 32 GB | 100 KB |
| L40S | 1.11 | 1.11 | 0.86 | 106 | 0.68 MB (unverified L2) | 48 GB | 100 KB |
| H100 SXM | 0.81 | 0.81 | 3.32 | 20 | 0.38 MB, split | 80 GB | 228 KB |
| H100 PCIe | 0.62 | 0.62 | 1.98 | 26 | 0.44 MB, split | 80 GB | 228 KB |
| GH200 | ≈ 0.81 (clock unverified) | 0.81 | 3.97 | 17 | 0.38 MB (unverified) | 96 GB | 228 KB |
| H200 | ≈ 0.81 (unverified) | 0.81 | 4.76 | 14 | 0.38 MB (unverified) | 141 GB | 228 KB |
| A100 80 | 0.47 | **0.24** | 2.02 | 9.6 | 0.37 MB, split | 80 GB | 164 KB |
| A100 40 | 0.47 | 0.24 | 1.54 | 12.5 | 0.37 MB, split | 40 GB | 164 KB |

Our measured 4090 clocks: ≈ 2.79 GHz at 450 W (sample_kernels.md §11), lower at the 300–350 W caps used since
2026-11-20. Datacentre parts run near their boost clock under load; the ratios use boost clocks on both sides, which
favours the 4090 slightly less than reality at 450 W and slightly more at 300 W.

## 2. What the facts mean for our gradient

### 2.1 Where a gradient's time goes on the 4090, and how each part scales

Measured on the 4090 (300 W, flow_thorax on the 288×288×140 grid (image 225×225×112), 10 coils × L 10, coil batch 5; sample_kernels.md §17.6, temporal_operator.md §23):

| case | gradient | FFT side (pad + FFT + crop) | sample kernels and the rest |
|---|---|---|---|
| 2-D 6:1,16:3 order 3, 13.3 M per encode (velocity recon, dense) | 26.1 s | 6.15 s (24 %) | 19.9 s |
| same, 1.33 M per encode | 9.9 s | 6.15 s (60 %) | 3.8 s |
| static order 2, 4.3 M | 0.77 s | ≈ 0.1 s | ≈ 0.67 s (tile-kernel bound) |

The sample kernels (vt / vs forward, ra / kl / tile adjoint) are latency- and issue-bound: 12 warps per SM, issue slots
52–60 % busy, FMA pipe ≈ 40 %, staging and barriers ≈ 55 % of a knot launch (sample_kernels.md §15.6, §17.7). Their
speed scales, to first order, with SMs × clock (same per-SM register file, same warp count), not with DRAM bandwidth.
The FFT passes run at 57–93 % of DRAM bandwidth and scale with it.

Projection (first order; the hypothesis to test first on each card):

| GPU | 2-D 13.3 M gradient | 2-D 1.33 M gradient | comment |
|---|---|---|---|
| RTX 4090 | 26.1 s (measured) | 9.9 s (measured) | |
| RTX 5090 | 6.15 / 1.78 + 19.9 / 1.27 ≈ 19.1 s (1.37×) | 3.5 + 3.0 ≈ 6.5 s (1.5×) | same per-SM limits as the 4090: the autotune shapes carry over |
| L40S | 7.2 + 17.9 ≈ 25 s (1.04×) | 7.2 + 3.4 ≈ 10.6 s (0.93×) | the FFT side is relatively more expensive than on the 4090 |
| H100 SXM | 1.85 + 24.6 ≈ 26.4 s (≈ 1.0×) | 1.85 + 4.7 ≈ 6.5 s (1.5×) | interpolation is ≈ 93 % of the gradient |
| H200 | 1.3 + 24.6 ≈ 25.9 s (1.0×) | 1.3 + 4.7 ≈ 6.0 s (1.65×) | as H100; the VRAM is the gain (§2.3) |
| A100 80 | 3.0 + 42–83 s (0.3–0.6×) | 3.0 + 8–16 s (0.5–0.9×) | half the FP32 lanes per SM: the MAC loops become FMA-bound |

The datacentre parts gain from three things this projection ignores: shared memory (larger items, deeper chunks, more
blocks where shared memory is the limiter today), a larger coil batch from VRAM (§2.3), and fewer stalls on L2 / DRAM
latency at the same warp count (unknown, to measure). They lose from the split L2 (far-partition hits cost up to ≈ 2×
the latency [S2][S3]). The H100 being no faster than a 4090 per GPU on the dense knot path is the single most important
number to confirm or refute in the first cluster hour.

### 2.2 FFT-side bytes per architecture

Per grid load (pad + forward FFT) and flush (inverse FFT + crop), knot case on the 288×288×140 grid, 50 channels, with today's defaults
(fused pad / crop, one-plane L2 slabs): 10.6 + 10.7 GB (temporal_operator.md §23.2). The table includes the
block-floating-point storage of §4 item 6 (6.0625 / 5.0625 / 4.0625 bytes per complex with 16-value blocks; the scatter's
output and the image-side inputs stay f32):

| storage | load + flush DRAM bytes | 4090 | 5090 | L40S | H100 SXM | H200 | A100 80 |
|---|---|---|---|---|---|---|---|
| f32 (today) | 21.2 GB | 32 ms | 18 ms | 37 ms | 9.6 ms | 6.7 ms | 16 ms |
| bfp24 | 17.2 GB (−19 %) | 26 ms | 15 ms | 30 ms | 7.8 ms | 5.4 ms | 13 ms |
| bfp20 | 15.1 GB (−29 %) | 23 ms | 13 ms | 26 ms | 6.8 ms | 4.8 ms | 11 ms |
| bfp16 | 13.0 GB (−39 %) | 20 ms | 11 ms | 23 ms | 5.9 ms | 4.1 ms | 9.6 ms |

ms = bytes / (bandwidth × 0.66), the efficiency the 4090 reaches on the whole FFT side today (6.15 s for 192 load +
flush pairs of 21.2 GB at 1.008 TB/s). A 2-D velocity gradient has 192 pairs: on the 4090 bfp24 saves ≈ 1.2 s of
26.1 s (dense) or of 9.9 s (sparse, −12 %); bfp16 ≈ 2.3 s. On HBM parts the same fraction is worth 3–5× fewer seconds.

L2 slabs (two-pass x → y through L2, temporal_operator.md §23.5): one z plane of all channels must fit
`HASTY_SEG_FFT_L2MAX` = 0.85 of the L2 (else slabs turn off and the x → y intermediate goes through DRAM: +5.8 GB per
pair). Plane sizes against the 0.85 × L2 budget:

| plane (all channels of one coil batch) | f32 | bfp24 | bfp16 | 4090 (61 MB) | 5090 / L40S (82 MB) | H100 / H200 (42.5 MB) | A100 (34 MB) |
|---|---|---|---|---|---|---|---|
| 288×288×140 grid, 50 ch (plane 288 × 288) | 33.2 MB | 25.2 | 16.9 | all fit | all fit | all fit | f32 just fits |
| 288×288×140 grid, 100 ch (coil batch 10) | 66.4 MB | 50.3 | 33.7 | bfp24 / bfp16 | all fit | bfp16 | bfp16 |
| 800×480×480 grid, 50 ch (plane 480 × 480) | 92.2 MB | 69.9 | 46.8 | bfp16 | bfp24 / bfp16 | none | none |
| 800×480×480 grid, 10 ch (coil batch 1) | 18.4 MB | 14.0 | 9.4 | all fit | all fit | all fit | all fit |

So BFP keeps the slab path on where it would turn off, and on the 800×480×480 grid with more than ≈ 20 channels only channel sub-slabs
(§4 item 7) keep it on on the HBM parts. On H100 / A100 a slab that relies on the far L2 partition may lose part of
its gain; `HASTY_SEG_FFT_L2FRAC` must be swept there (§6).

### 2.3 Channel batch from VRAM

One knot slot (all resident grid channels) is cb × L channel grids: 92.9 MB per channel on the 288×288×140 grid, 1.47
GB on the 800×480×480 grid (1 mm torso; sample_kernels.md §17.3). Measured on the 4090 (1-D, 13.3 M, order 3): per-channel
cost of a visit falls 2.2× from 20 to 100 channels per launch, and coil batch 10 (9.1 GiB) gives −16 % of the gradient
against batch 5. Batch at L = 10 with ≈ 60 % of the card for the slot (the rest: tables, state, FFT plans, per the
ledger, memory_ledger.md):

| GPU | VRAM | 288×288×140: coils per batch that fit (0.93 GB each; engine cap 16) | 800×480×480: coil batch |
|---|---|---|---|
| 4090 | 24 GB | 15 | 1 (14.7 GB, above the 60 % budget) |
| 5090 | 32 GB | 20 | 1 |
| L40S | 48 GB | 31 | 1–2 |
| A100 40 | 40 GB | 25 | 1 |
| H100 / A100 80 | 80 GB | 51 | 3 |
| GH200 | 96 GB | 61 | 3–4 |
| H200 | 141 GB | 90 | 5 |

On the 800×480×480 grid the coil batch is where the large cards win: batch 5 on an H200 against batch 1 on a 5090 is ≥ 2× cheaper per
channel visit (2.2× measured from 20 to 100 channels on the 288×288×140 grid) (§17.2's counting), on top of the FFT bandwidth. recon_p2np's slot budget is fixed today
(`knot_grid_budget` = (free − work) / 3, ≤ 6 GiB): it must come from the ledger (§4 item 1).

### 2.4 Host link

The gradient's host traffic (state up / down, streamed tables, k-space) is ≈ 1 % of an iteration of the 1 mm box (198×189×228 image) over PCIe 4
(memory_ledger.md §2). PCIe 5 (5090, H100) halves that; NVLink-C2C on GH200 makes it ≈ 10× cheaper and coherent: the
GPU can read pinned or even plain host memory at ≈ 250–450 GB/s, an eighth to a tenth of HBM. That allows a new tier on
GH200 only: sample tables (read once per pass, sequentially) read in place from Grace memory, freeing VRAM for a larger
coil batch. Elsewhere host streaming is not a bottleneck and needs no architecture-specific work.

## 3. Per-architecture recommendations

The autotune already times every candidate on the card it runs on, so most settings need no per-arch code; what
changes is the candidate set (shared memory admits more shapes on sm_80 / sm_90), a few compile-time tunables, and the
memory plan. Items are numbered as in the ranked list (§4).

### 3.1 sm_89: RTX 4090, L40S

- 4090: today's tuned state. Everything in §4 marked "4090: full" is built and measured here first.
- The velocity reconstruction's kernels are shared-memory-limited here (§5.3): ra at 12 warps (3 × 30 KB blocks;
  registers would allow 16) and the default vt at 4 warps (one 76 KB block). A ra block of ≤ 24 KB would hold 4 blocks
  (100 KB / 4 − 1 KB reserved); vt with NWS 2 already holds 8 warps (one 82 KB block of 8 warps) and is in the candidate
  set. Item 2 targets the ra footprint.
- L40S: same per-SM limits and compiler output as the 4090 (same sm_89 cubins), 1.11× SMs × clock, 0.86× bandwidth,
  48 GB. The FFT side is relatively the most expensive of all targets: BFP storage (item 6) and channel sub-slabs
  (item 7) matter most here. Coil batch: all 10 coils on the 288×288×140 grid, 1–2 on the 800×480×480 grid.

### 3.2 sm_120: RTX 5090

- Same shared memory (99 KB per block), threads (1536) and registers as sm_89, so every 4090 candidate set carries over;
  the tunables are rerun by the autotune. ptxas allocates differently for sm_120 (§5.3: median −5 registers for ra and +22 for vs against sm_89, with
  per-shape differences of −60 to +88; e.g. ra nxg 2 cb 3 for 48 × 1: 116 registers, no spill, against 128 registers
  with spills on sm_89; the tile forward has 35 spilling candidates on sm_120 against 5 on sm_89), so tuned choices must
  not be copied from the 4090: tune on the card, skip spilling candidates, and cache per GPU (items 0, 3).
- Shared memory per SM as on sm_89: the ra / vt occupancy limits of §3.1 apply unchanged.
- 96 MB L2: 100-channel planes of the 288×288×140 grid fit the slab budget without BFP; coil batch 10+ there (32 GB), 1 on the 800×480×480 grid.
- TMA builds for sm_120; clusters build but whether a cluster > 1 runs on a 5090 is unverified. Nothing in the
  first-pass list depends on it.
- No P2P: multi-GPU stays host-mediated (plan_cluster.md §8.2).

### 3.3 sm_90: H100 SXM / PCIe, H200, GH200

- Shared memory 228 KB per SM (§5.3): of the ra candidates that are shared-memory-limited on sm_89, 66 of 98 become
  register-limited on sm_90 (median 1.75× the warps); the velocity ra rises from 12 to 16 warps, the default vt from 4
  to 8. Because vt was declined on the knot path on the 4090 at 4 warps (sample_kernels.md §16.5), **re-time vt on the
  knot path on sm_90**: it may win there.
- Add to the autotune on sm_90: ra / kl LISTMAX 2048 and 4096 (no occupancy cost at CHK 16; fewer split tiles: the
  13.3 M case has 494 k split items at 1024, sample_kernels.md §17.1). Do not expect CHK 48 / 64 or vt SMAX ≥ 96 to help:
  they fit, but at half the warps (ra nxg 4 cb 7 at CHK 48: 8 warps; vt SMAX 96 / 128: 4 warps).
- The FFT passes on the 288×288×140 grid run 2 blocks per SM instead of 1 (register-limited at 56 registers, §5.2).
- 2048 threads per SM are usable only below 32 registers per thread: irrelevant for the sample kernels, relevant for
  the FFT passes (36 warps per SM at 56 registers against 18 on sm_89).
- Split 50 MB L2: slabs of one 50-channel plane of the 288×288×140 grid (33 MB) pass only through the 0.85 rule; sweep `HASTY_SEG_FFT_L2FRAC`
  0.4 / 0.5 / 0.85 and slab off. Channel sub-slabs (item 7) are needed for 100 channels and for the 800×480×480 grid.
- FFTs are 3.3–4.8× faster than on the 4090; interpolation is ≈ 90 % of the gradient. Spend the cluster hour on the
  sample kernels' candidates, not on the FFT.
- Vector atomics (item 9) and cluster / DSMEM split-tile reduction (item 10) are sm_90 features worth trying here.
- GH200: NVLink-C2C table tier (item 12); aarch64 host build (plan_cluster.md §3.1).
- H200 / GH200: coil batch 3–5 on the 800×480×480 grid (§2.3); the ledger must size the slot.

### 3.4 sm_80: A100 40 / 80 GB (Berzelius)

- 64 FP32 lanes per SM and 108 SMs at 1.41 GHz: 19.5 TFLOPS against 82.6 on the 4090. Our MAC loops run at
  12–30 % of the 4090's peak, i.e. 10–25 TFLOPS of issued FMAs: on an A100 they hit the FMA pipe. Expect the sample
  kernels at 0.25–0.5× the 4090 and prefer shapes with the fewest issued FMAs per useful MAC (NXG 1 in ra: exact x
  points; vs over vt at orders ≤ 1).
- 164 KB shared memory: the occupancy gains of sm_90 apply partly (ra: median 1.33× the warps of sm_89). ptxas uses
  more registers on sm_80 (ra median +16, tile forward +14 against sm_89; 57 spilling tile-forward candidates), so the
  velocity ra (156 registers) holds 12 warps by registers here.
- Use: FFT-heavy, sample-sparse work (1.33 M class) or memory-hungry setup; not the dense velocity gradient.

### 3.5 Interpolation vs FFT balance, and what not to do

- Do not move work from the FFT side to the interpolation on HBM parts (e.g. smaller σ for a smaller grid would also
  break the accuracy rule); if anything the reverse: on H100 / H200 a larger grid (σ_eff 1.43 via `HASTY_NUFFT_GRID`) is
  cheaper there than on the 4090 and is strictly more accurate (temporal_operator.md §20.3).
- Tensor cores. The earlier estimate here (3xTF32 on the adjoint's per-tile GEMM: "≈ 1.1× on H100, none on 4090")
  assumed an FMA-bound kernel. That premise is wrong: both interpolation kernels are bound by instruction issue and
  latency (sample_kernels.md §21.1). One mma.sync replaces 32+ FFMA, so tensor cores can win where tensor rate ≈ FP32
  rate.
  - **Built and measured** (sample_kernels.md §21): the forward vt contraction on tensor cores (`VTC`,
    `HASTY_SEG_VT_MMA`).
    - **Precision:** fp16 hi / lo with power-of-two scaling, 3 MMAs, adds 1.6–3e-6 at 256³ order 3. bf16 hi / lo adds
      4–5e-6 there but 3–5e-4 where segments cancel. Plain TF32 / bf16 / fp16 (3e-4–4e-3) stays excluded.
    - **4090 speed:** no faster than the best FMA vt (fp16 +6.5 ms per launch, bf16 level).
    - **Why:** the tensor pipe is 29 % busy. The issue slots go to splitting the grid operand per (sub-chunk, plane) and
      to building the weights, because items carry only about 20 samples, about 8 active per plane.
  - **On H100 / A100** a faster tensor core therefore buys at most about 1.3× (counting, unmeasured; datasheet rates
    **UNVERIFIED**) until that overhead is removed: grid stored pre-split by the z pass, larger resident items, a
    per-plane f32 flush.
  - **The adjoint as a GEMM** (K = active samples per plane, about 8) is not worth building at this density.

## 4. Ranked phase-2 list

Ranked by expected gain × likelihood on the target cards. "4090: full" = can be built, tested and benchmarked here;
"4090: compile" = sm_90-only or sm_120-only feature, compile- and resource-checked here (kernaudit, nvcc), correctness and
timing only on the target. Every item is a new variant behind a switch; the previous path stays selectable and the
autotune keeps a variant only where it is faster (the house rule).

| # | item | targets that gain most | gain × likelihood | 4090 |
|---|---|---|---|---|
| 0 | per-GPU tune cache and first-hour sweep | all | enabler (saves cluster minutes) | full |
| 1 | coil batch / knot slot budget from the ledger | 80–141 GB parts, 800×480×480 grid | −16 % measured on 288×288×140 (cb 5 → 10); more on 800×480×480 | full (288×288×140, cb 10) |
| 2 | ra / kl staging: ≤ 24 KB footprint on 100 KB parts, cp.async double buffer | all | high × high: staging ≈ 55 % of a knot launch; 12 → 16 warps on sm_89 / sm_120 | full — **done** (§19 of sample_kernels.md) |
| 3 | per-arch spill screening + per-shape launch-bounds candidate (ra / kl) | all, sm_80 / sm_120 most | medium × high (audit-backed) | full — **done** (§19) |
| 4 | warp-specialised ra / kl (producer staging warps, consumer MAC warps) | all; sm_90a / sm_120a add setmaxnreg | high × medium | full (without setmaxnreg) |
| 5 | vt / vs forward with a shared-memory plane (cp.async; TMA on sm_90 / sm_120) | sm_90 (split L2), all | medium × medium | full (cp.async) |
| 6 | block-floating-point storage of the FFT intermediates | 4090, 5090, L40S | medium × medium (accuracy across channels to measure first) | full |
| 7 | channel sub-slabs for the L2 two-pass FFT | 800×480×480 everywhere; 100-ch 288×288×140 on 4090 / H100 / A100 | medium × high on 800×480×480 | full |
| 8 | fused x pass at 2 blocks per SM on 100 KB parts | 4090, 5090, L40S | low–medium × high | full |
| 9 | float2 vector atomics (split tiles, fused crop) | sm_90, sm_120 | low–medium × high | compile |
| 10 | cluster / DSMEM reduction of split tiles | sm_90 | medium × low–medium | compile |
| 11 | FFT strided passes with 32 lines per block on big-smem parts | sm_80, sm_90 | low × medium | partial (z pass only) |
| 12 | GH200 C2C table tier (tables read in place from Grace memory) | GH200 | medium (VRAM for coil batch) × medium | correctness only (mapped pinned memory) |
| 13 | L2 persistence window for slabs | sm_90 (split L2) | low × low | full |

### Item 0. Per-GPU tune cache and the first-hour sweep

- Today the autotune re-times every candidate at each engine construction (kernels come from the arch-keyed NVRTC
  disk cache, choices do not). Design: after `autotune`, write the chosen (TileCfg forward / adjoint, VsCfg / geometry,
  RaCfg, kl choice) and their times to `$HASTY_KERNEL_CACHE/../hasty-tune/<gpu-name>-<cc>-<driver>-nvrtc<ver>/<key>.json`,
  key = hash(engine shape: nseg, order, cb, ks, rdeg, W, σ, grid dims; table statistics bucket: samples per nonempty
  tile, split fraction). On construction with a matching key, load the choice and skip the timing.
- Switch `HASTY_TUNE = reuse` (default) | `retune` | `off`.
- 4090: full. Test: two runs, second one equal results bit-for-bit and no timing calls; changed shape → miss.
- Target: nothing to measure; the sweep of §6 fills the cache, which is copied back into the repo
  (research/experiments/arch_runs/<gpu>/tune/).

**Status (2026-11-22): built (`hasty-nufft/src/segnufft/tune.rs`).**
- Where setup time goes (velocity knot-mode opbench, research/experiments/compressed_dram/logs/p17 / p18). The "setup
  908 / 1081 s" there is opbench's projection for a full recon, not a measurement: table builds scaled to all 93 M
  samples (128 s) + segments (1.4 s) + autotune (123 / 306 s) + 15 power iterations on two sets (645 s). The measured
  part of the 7-minute run is the autotune: 306 s of 430 s wall (p18). The 306 − 123 s difference between the two
  binaries is kernel compiles: the new binary's kernel sources were not in the NVRTC cache yet. Split of one
  autotune, 2-D knot order 3, 4 mm grid, 1-minute scan, cb 10, 236 candidates (instrumented: opbench / recon_p2np
  print an `autotune split` line):
  - kernels not yet compiled (new sources): 212 s = timing launches 57 s + NVRTC compiles 94 s (78 kernels) + 61 s
    rest, which is the ptxas spill reports (one more NVRTC compile per new kernel; cached in `ptxinfo/`, not counted);
  - kernels already compiled: 59.4 s, of which 59.3 s are timing launches (≈ 0.25 s per candidate: warm-up + 3
    repetitions). Cubin reads and module loads take < 0.1 s. Some compiles also come back on a warm cache: the adjoint
    candidates are built around the forward winner, and the forward tile winner varies from run to run (cpbf 8 /
    chunk 32 against 5 / 48 within 1 %), so a retune compiled 24 new adjoint kernels (42 s).
- Cached: everything an autotune decides, with its measured times. That is the forward and adjoint TileCfg; the vs /
  vt / vw forward (geometry, VsCfg, sub-chunk, tensor-core contraction, VwCfg) or "tile kernel"; the ra adjoint
  (RaCfg + RaVar: in-place list, stages, launch-bounds hint) or "tile kernel"; the knot-path layout (skip, kl kernel
  RaCfg + RaVar) and its times. A step that did not run is stored as "unchanged". The multi-image K-image shapes
  (`autotune_multi`, the one-bin fallback tune, water + fat) are stored separately under their own key.
- Key: FNV-64 of a canonical description, which is also stored in the file and compared on load. It covers the GPU
  (name, cc, driver, NVRTC), a hash of the kernel sources and of the tuning code (segnufft.rs, vs / ra / kl / tune.rs,
  kernel.rs, BFP header), and the engine. Engine fields: nseg, order, coils, coil batch, frame slots, frames, knot
  support / kr / mc / rdeg, W, β, kernel kind, fine grid, smem opt-in, start tile edge / item length, vs geometry,
  vt / vw modes, zero / prune / fused pad-crop / FFT-store / grid-store / map / diagonal storage, warp, water-fat,
  accumulate flags. It also covers every `HASTY_SEG_*` switch (except `*_LOG` and `KINFO`), `HASTY_NVRTC_LINEINFO`,
  the knot layout of the tables, and a table-statistics bucket. The bucket holds half-octaves (⌊2 log₂ x⌋) of the
  sample count, of the nonempty tiles and of the entries per nonempty tile, eighths of the split-item fraction, and a
  half-octave of the vs items of the tuned knot. Any change of density or k-space extent by ≥ √2 moves to another
  bucket. Inside a bucket every candidate's time scales with the same per-tile work, so their order does not change.
  The 2 mm and 4 mm thoraxes (3862 vs 4205 entries per tile but 10× the tiles) get different keys.
- Location: `hasty_site::tune_cache()` = `HASTY_TUNE_CACHE`, else next to the kernel cache:
  `rustlib/target/hasty-tune/NVIDIA-GeForce-RTX-4090-cc89-drv580.178.04-nvrtc13.0/<key>.json` (≈ 2.5 KB each).
  Written to a temporary file and renamed; an unreadable / foreign / missing file is a miss. A second construction in
  the same process is served from memory.
- Switch `HASTY_TUNE = reuse` (default) | `retune` (time and overwrite) | `off` (no reads, no writes, no stats copy:
  the previous code path). One stderr line per autotune: `[tune] hit | miss (<why>) | retune | stale (<why>) <key>`.
  A loaded choice is checked first: tile edge / item length, shapes fit the shared memory, the vs geometry matches the
  set's tables, the ra / vw / kl shape is valid on this engine. A record that fails the check is retuned ("stale").
- Tests: `cargo test -p hasty-nufft --lib tune` (CPU: JSON round trip with NaN times, bucket boundaries, 8
  concurrent writers + a reader that never sees a partial file); `--test seg_tune` (GPU, 3 min, 2-D knot order 3).
  - A miss times and writes one file. A second engine loaded from the file times 0 candidates, makes the same choice
    and gives a bitwise-identical forward.
  - The adjoint is not bitwise repeatable even on one engine (f32 atomics of split tiles: 5e-6 between two calls). Two
    engines differ by 3e-5; the test asserts ≤ 10× the engine's own repeatability.
  - An in-process hit times 0 candidates. Another coil batch misses. A corrupted file is retuned and rewritten without
    a panic. Retune times again. Off times and leaves the files untouched.
- Saving, measured: opbench `--models seg-full --nseg 10 --vcoils 10 --order 3 --knots2d 6:1,16:3 --scan-min 1 --res
  4e-3`, kernels already compiled:
  - miss: autotune 59.4 s, wall 81.6 s;
  - hit: autotune 0.0 s, wall 22.2 s, operator times unchanged (forward 1865.9 / 1864.8 ms).
  - On the 2 mm velocity case (p17 / p18) the same hit removes 123 s with all kernels compiled and 306 s right after a
    kernel edit (the first run after an edit misses and compiles anyway).
- Open:
  - The forward tile winner is noisy (only matters where vs loses). The cache now freezes one sample of that noise:
    `retune` to refresh.
  - The power limit and clocks are not part of the key (choices measured at 350 W and 450 W share a file).
  - The source hash includes segnufft.rs, so any edit there retunes once (conservative).
  - The 645 s of projected power iterations are the larger part of a recon's setup: addressed by the Lanczos
    estimator, temporal_operator.md §24.

### Item 1. Coil batch from the ledger

- recon_p2np: `knot_grid_budget` from the memory ledger (memory_ledger.md §3) instead of (free − work) / 3 ≤ 6 GiB;
  opbench `--max-grid-gb` already exists for the study.
- Switch: `--knot-grid-gb auto | <GB>`.
- 4090: full on 288×288×140 (cb 5 / 10, measured −16 %). Target: the gradient against cb on 80 / 96 / 141 GB on 288×288×140 and 800×480×480
  (`opbench --kl-compare --max-grid-gb 10 / 20 / 40 / 80`).

### Item 2. ra / kl staging: smaller footprint on 100 KB parts, cp.async double buffer

- Today ra prefetches the next chunk's raw inputs (records, ΔC, a_l, y per coil) into registers one chunk ahead, then
  stages through shared memory behind two barriers per chunk (sample_kernels.md §15.4, §16.2). The chunk-head and stage
  barriers are 15 % of stall samples, staging and barriers ≈ 55 % of a knot launch (§17.7).
- The audit (§5.3) shows what limits the velocity reconstruction's ra shape (nxg 4, cb 7, chk 16; 50 and 100 channels):
  on sm_89 / sm_120 it is **shared memory**, 29.6–29.9 KB per block → 3 blocks, 12 warps, while its 122–128 registers
  allow 16; on sm_90 it is registers (125) → 16 warps. Freeing registers alone therefore gains nothing on the 100 KB
  parts, and a double buffer that adds shared memory would cost a block there.
- Design, two steps, each a variant:
  1. **Footprint ≤ 24 KB** (4 blocks on 100 KB parts): read the item's list in place instead of copying it (the kl kernel
     already does: binary search of the knot window runs; −4.1 KB at LISTMAX 1024), and stage 12 instead of 16 samples
     (≈ 1.6 KB of staging per sample at order 3, cb 7, of which the z and row-weight buffers are 512 B each: −6.4 KB).
     By the engine's `ra_smem` layout: 29.6 KB → 25.5 KB (list in place, CHK 16) → 19.2 KB (CHK 12), i.e. 12 → 16 warps
     on sm_89 / sm_120 (then register-limited), with a third more chunk barriers per sample (to measure). The kernel has only run at CHK 16 / 32;
     other chunk sizes need the test suite first.
  2. **cp.async double buffer** (16-byte LDGSTS, sm_80+): while the warps MAC chunk k, chunk k + 1's raw records and coil
     values are copied into the second buffer; `cp.async.wait_group` + one barrier at the chunk head. Removes the
     prefetch registers (≈ 20–40) and the stage barrier. On sm_90 the extra buffer is free; on 100 KB parts it must fit
     inside the 24 KB of step 1 (raw records only, not the derived z).
- Switch: defines `RA_LISTINPLACE`, `RA_CPASYNC` (NVRTC), candidate dimensions in `ra_candidates` / `kl_candidates`; env
  `HASTY_SEG_VSA_STAGE = 0 | inplace | cpasync | auto`.
- 4090: full (tests as tests/seg_ra_vt.rs and seg_kl.rs, sanitizer). Check with kernaudit (registers, shared memory,
  warps per SM) before timing; time the adjoint kernel on the 2-D knot order-3 case at 1.33 M and 13.3 M samples.
- Target: the same rows on sm_90 (register-limited there: only step 2 matters) and sm_120.

**Status (2026-11-20, sample_kernels.md §19): built and measured.**
- As built: `RaVar { inplace, stages, lbmin }` (defines `RA_INPLACE`, `RA_STAGES`, `LBMIN`), the same in kl (stages,
  lbmin), and the vt / vs sub-chunk staging `VSCH` (the same item-2 idea for the forward). Switch
  `HASTY_SEG_VSA_STAGE = 0 | inplace | cpasync | auto`, as proposed; vs: `HASTY_SEG_VS_SCH`.
- 4090:
  - ra / kl on the velocity shape reach 16 warps from 12 at chunk 12 with cp.async (125 / 123 registers, 21.1 KB).
    ncu: 11.85 → 16.27 achieved warps; kernel −11 % / −14 % per launch.
  - vt: 4 → 8 warps with sub-chunk 29; −7 … −18 % per autotune launch.
  - Gradient of the velocity reconstruction: −5.2 % at 1.33 M and −1.9 % at 13.3 M samples per encode.
- Finding against the design: the in-place list with the register prefetch alone costs about 33 registers (kl-style
  entry / uw prefetch: 128 → 161), so it stays at 12 warps. The footprint only pays together with cp.async (which drops
  the prefetch registers) or with the hint 4.
- Audit (2970 kernels): every target has a spill-free 16-warp velocity ra / kl variant:
  - sm_89 / sm_120: cp.async chunk 12;
  - sm_80: hint 4, since ptxas there keeps 156 registers with cp.async;
  - sm_90 / sm_100: already 16; cp.async allows chunk 24.
- 100-channel shapes stay register-bound on every architecture.
- Verification:
  - correctness: 3260 variant runs against their base kernels (bitwise where no atomics are involved);
  - memcheck: clean;
  - racecheck: clean on the static cases; the knot cases are not finished (out of host memory, see
    research/experiments/phase2/HANDOFF.md);
  - synccheck and the seg_* regression: not run yet. They are left for the combined GPU campaign.
- Open: the ra / vs autotune still times knot 0 on 2-D engines (an edge knot). The 13.3 M forward did not see the vt
  kernel gain.

### Item 3. Per-arch spill screening and a launch-bounds candidate for ra / kl

- Audit findings (§5.3, §5.4): spills depend on the arch (the tile forward has 57 spilling candidates on sm_80, 35 on
  sm_120, 5 on sm_89; ra nxg 2 cb 3 for 48 coils × L 1 spills on sm_89 and sm_90 only). A blanket
  `__launch_bounds__(NT, m)` makes things worse (more spills, fewer warps), but per shape the best of {no hint, m 3, m 4}
  removes every ra / kl spill on every arch and adds warps without spilling in 1–17 shapes per arch.
- Design:
  1. Spill screening in the autotune: compile with `--ptxas-options=-v` (or read `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`
     against the shape's known stack frame: 32 B ra / tile, 48 B kl) and skip spilling candidates before timing; log them.
  2. A define `LBMIN` (0 = today) in ra and kl: `__launch_bounds__(NTA, LBMIN)`; the candidate set adds LBMIN 3 / 4 for a
     shape only where the resource check shows no spill and at least the warps of LBMIN 0. Not for vs / vt (m 2 spills
     there on every arch).
  3. While editing ra / tile / kl: find and remove the 32 / 48-byte stack frames (no spills; probably a dynamically
     indexed small array).
- Switch: `HASTY_SEG_LBMIN = auto | 0 | 3 | 4`, `HASTY_SEG_SPILLSKIP = 1 | 0`.
- 4090: full; kernaudit gives the expected per-arch picture before any timing. Target: nothing new to build; the
  autotune applies it on the card and the tune cache keeps the choice.

**Status (2026-11-20, sample_kernels.md §19): built.** The plan was to read `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` against
a fixed per-family frame. That turned out to be unreliable: the spill-free frame varies (ra 32 / 40 / 48 B, kl 48 B,
vs / vt 0 or 32 B), and the 32 B is the `sincosf` slow-path array.

So the screen works as follows:
- registers, local size and occupancy come from the driver;
- the spill bytes come from a ptxas `-v` report (one NVRTC compile per kernel that has local memory, cached on disk next
  to the cubins);
- it applies to ra, kl and vs / vt candidates.

On the 4090 it detects the known 48 × 1 spill (20 B). LBMIN 3 / 4 is a candidate dimension for the three fastest ra /
kl kernels, kept only without spills and with at least the unhinted warps.

Switches: `HASTY_SEG_LBMIN = auto | 0 | <m>`, `HASTY_SEG_SPILLSKIP = 1 | 0`. Not done: the tile-kernel candidates in
segnufft.rs are not screened yet (owned by another agent this session), and the 32-byte frames were not removed (they
are the `sincosf` slow path, never executed for |φ0| ≪ 1e5).

### Item 4. Warp-specialised ra / kl

- The structural form of item 2: one or two producer warps per block stage chunk k + 1 (cp.async or TMA, kernel values,
  w_l, z build) while TT consumer warps (one per tile plane) MAC chunk k; handover through mbarrier (sm_80+:
  `cuda::barrier` / named barriers `bar.sync id, n`). Removes the block-wide barrier per chunk and overlaps the 55 %
  staging with the MAC. With `sm_90a` / `sm_120a` targets, `setmaxnreg` gives the producers ≈ 40 and the consumers
  ≈ 200 registers.
- Needs: the engine's NVRTC arch string with the "a" suffix on demand (`Device::arch` + option, e.g. `HASTY_NVRTC_ARCH_A=1`);
  "a" cubins run only on that exact CC.
- Switch: define `RA_WS` + candidate; env `HASTY_SEG_VSA_WS`.
- 4090: full without setmaxnreg. Target: with setmaxnreg on sm_90a / sm_120a.
- **Built and measured 2026-11-21 (sample_kernels.md §22.3):** `RaVar.stages` 5 / 6 (4 / 2 producers, named barriers),
  bitwise; 1.8 × slower at 50 channels on the 4090. At equal registers a producer warp costs a MAC warp (8 instead of 16
  MAC warps per SM), and the MAC is ≈ 80 % of the work. Default off (`HASTY_SEG_VSA_WS`).

### Item 5. vt / vs forward with a shared-memory plane

- vs / vt read their register windows straight from L2 (L1 hit 5–6 %, L2 throughput 68–80 % on the 4090: §12.7), and
  the two row halves load rows 5 … TY + 3 twice (16 rows for 13). Design (§12.9 item 1): per (item, plane) the block loads
  the plane's (TX + 9) × (Δy + 10) × channels box once into shared memory (cp.async; on sm_90 / sm_120 one TMA bulk
  tensor copy per box with an mbarrier), lanes then read their channel's rows with conflict-free LDS.64. ≈ 1/3 fewer L2
  bytes, one barrier per plane.
- On sm_90 the split L2 makes L2 delivery more expensive than on the 4090; this item is the forward's main lever there.
- Shared-memory budget (§5.3): vt is already shared-memory-limited on the 100 KB parts (one 76 KB block, 4 warps; NWS 2:
  8 warps). The plane box (13 × 13 points × 16 channels × 8 B ≈ 22 KB per channel group and plane at a 4 × 4 tile) must replace
  part of the record staging there, or be used only on sm_80 / sm_90, where the same vt shape runs 2 blocks.
  The 228 KB also allow SMAX 96–192 items (fewer window loads per sample: the knot path's items are ≈ 4× emptier than vt
  needs, sample_kernels.md §16.5).
- Switch: define `VS_PLANE = 0 | 1 (cp.async) | 2 (TMA)`; `HASTY_SEG_VS_PLANE`.
- 4090: full for 1; 2 compile-only. Target: vt forward on the 2-D knot path, SMAX sweep.
- **Built and measured 2026-11-21 (sample_kernels.md §22.1):** `VW 1` (plane window in shared memory by cp.async,
  weights built once per block, consumers read shared memory only; 115–141 registers), bitwise. 62.7 vs 54.5–57.1 ms on
  the 4090: shared-memory bound (150 wavefronts per (sample, plane, 16-channel slice)) and only 2 window stages fit in
  97.8 KB. Stays an autotune candidate for the 228 KB parts. TMA not built.

### Item 6. Block-floating-point storage of the FFT intermediates (proposed by the user)

- Design is owned by a separate agent: **research/bfp_storage.md**. Here only what bears on the architectures.
- Idea: grid intermediates stored as blocks of complex values sharing one exponent byte, with 24 / 20 / 16-bit integer
  mantissas (6 / 5 / 4 bytes per complex instead of 8), packed in the FFT pass store and unpacked in the next load
  (hasty-fft `IoHook`). The user's storage-only measurement: bfp24 3.9e-7 relative L2 error (f32 5.0e-8, bfp20 6.3e-6,
  bfp16 1.0e-4).
- Thread layout found here (hasty-fft codegen): every grid pass is strided over the channel-fastest layout with 16 lines
  per block (`li = tid % 16`), so one half-warp holds 16 consecutive fast-axis values and a 32-value block maximum
  cannot be formed inside one pass on 99 KB parts (32 lines at n = 288 need 147 KB). 16-value blocks work everywhere;
  32-value blocks only with item 11 on sm_80 / sm_90.
- Per-arch effect: FFT-side DRAM bytes −19 % (bfp24) to −39 % (bfp16), §2.2; worth most on 4090 / 5090 / L40S; keeps the
  L2 slab path on for larger planes (§2.2 table). Arch-neutral code; fully testable on the 4090.
- Switch (proposed there): `HASTY_FFT_STORE`.

### Item 7. Channel sub-slabs for the L2 two-pass FFT

- temporal_operator.md §23.6 item 2: when a plane of all channels exceeds the L2 budget (100 channels on 288×288×140 on 4090 /
  H100 / A100; 800×480×480 everywhere above ≈ 20 channels), run the x and y passes per (slab of planes, channel tile). The x
  pass's lines are already channel tiles; the y pass's lines are (x, channel) flattened and need a line layout
  (channel tile) × (kx, z) so a channel tile's plane is contiguous in its lines (`AxisPlan::launch_lines` with a channel
  stride).
- Switch: `HASTY_SEG_FFT_SLAB_CH = auto | 0 | <channels>`.
- 4090: full (288×288×140 with coil batch 10; the 1 mm box). Target: H100 / A100 (split L2) on 800×480×480, sweep channels per sub-slab.

### Item 8. Fused x pass at 2 blocks per SM on 100 KB parts

- The fused pad / crop x passes run at 55–69 % of DRAM bandwidth, latency-bound at one block per SM (73.7 KB of shared
  memory for 16 lines × 288 × 2 buffers; §23.4). The audit confirms one block (18 warps) on sm_89 / sm_120 and two
  (36 warps, register-limited) on sm_80 / sm_90 for the plain passes (§5.2).
- Design (§23.6 item 1): an in-place stage order with one shared buffer (radix plan whose radices all divide the largest;
  288 = 8 · 4 · 9 would need a radix-9 or a 2 × 144 split), or line prefetch into registers.
- Switch: `HASTY_FFT_XPASS = 2buf | inplace | prefetch`.
- 4090: full. Target: on sm_90 check the plain passes already reach the bandwidth roofline at 2 blocks.

### Item 9. float2 vector atomics (sm_90+, sm_120)

- The split-tile atomics (ra / tile adjoint: `atomicAdd(&gp->x); atomicAdd(&gp->y)`) and the fused crop's per-voxel
  atomic into x become one `atomicAdd(float2*)` (per-component atomic, global memory, CC 9.x+ [4]) under
  `#if __CUDA_ARCH__ >= 900`. Halves the RED instructions; L2 traffic per atomic unchanged.
- Switch: define `ATOM_V2` (auto on CC ≥ 9.0).
- 4090: compile only (sm_89 has no vector float atomics; checked with nvcc). Test on target: seg_ra_vt / seg_kl /
  seg_fft_traffic equal to the scalar path within the atomics' order spread; kernel time of split-heavy launches (13.3 M:
  494 k split items).

### Item 10. Cluster / DSMEM reduction of split tiles (sm_90)

- Split tiles (lists > max_item) today: `seg_zero_split` + atomics (ra), or one long block per tile (kl, which loses on the
  long centre tiles: §17.6). Design: launch each split tile's parts as one thread-block cluster (2–8 CTAs); every CTA
  accumulates its part in registers as now, parks it in its shared memory, `cluster.sync()`, then CTA r reduces the r-th
  slice of the tile's points over all CTAs' shared memory (DSMEM) and stores it once. No zeroing, no atomics, balanced
  long tiles. Unsplit tiles keep the plain path.
- Cost: clusters need a cluster-aware item list (parts of a tile adjacent, padded to the cluster size).
- Switch: define `RA_CLUSTER = <cluster size>`; `HASTY_SEG_VSA_CLUSTER`.
- 4090: compile only (kernaudit for sm_90 / sm_120 + nvcc). Target: H100 at 13.3 M (split-heavy). sm_120: only after a
  cluster > 1 is confirmed to run.

### Item 11. FFT strided passes with 32 lines per block

- `choose_shape` caps strided passes at 16 lines. With 163 / 227 KB, 32 lines (147 KB at n = 288) give 256-byte runs per
  element position and allow 32-value BFP blocks. Design: cap = 32 when the device's opt-in allows it.
- Switch: `HASTY_FFT_LPB = 16 | 32 | auto`.
- 4090: partial (the z pass of 288×288×140, n = 140: 71.7 KB, fits). Target: x / y passes on sm_90.

### Item 12. GH200: tables read in place over NVLink-C2C

- On GH200 the GPU reads Grace memory coherently at up to 450 GB/s per direction. The packed sample tables
  (213 B / sample) are read once per pass, sequentially: reading them in place costs ≈ 1/8 of HBM bandwidth on a small
  share of the traffic and frees their VRAM (9.3 GiB for the 1 mm box, 198×189×228 image) for the coil batch.
- Design: a ledger tier "M" (host-mapped) for the tables; kernels take the same pointers (cuMemHostGetDevicePointer or
  system-allocated memory under ATS).
- 4090: correctness only (mapped pinned memory over PCIe 4 is ≈ 40× slower than VRAM). Target: GH200 gradient with
  tables in tier M vs D, and the coil batch it frees.

### Item 13. L2 persistence window for slabs

- `cudaAccessPolicyWindow` on the slab's planes (hitProp persisting). The 4090 did not need it (plain caching kept one
  plane, §23.5). On the split H100 L2 it may help keep a 33 MB slab resident. Low expectation; cheap to try.
- Switch: `HASTY_SEG_FFT_PERSIST = 0 | 1`. 4090: full. Target: H100.

## 5. Compile audit (offline, NVRTC 13.0, no GPU)

Run 2026-11-20 (`research/experiments/arch_audit/run.sh`, 4 NVRTC threads at CPUQuota 400 %): 5611 kernels, 0 compile
errors. Rows: `out/results.jsonl`; tables: `out/summary.md`; digest: `analyze.py out/results.jsonl [variant …]`.
What-if variants: `out_lb3/`, `out_lb4/` (ra / kl with `__launch_bounds__(NTA, 3 | 4)`), `out_vlb2/` (vs / vt with
`__launch_bounds__(NTV, 2)`).

Cases (engine shapes; grid channels = coil batch × L):
- `vel2d_o3_cb5` / `vel2d_o3_cb10`: the velocity reconstruction, 2-D knots (respiratory linear × cardiac cubic), order 3,
  L 10, coil batch 5 / 10 (50 / 100 channels);
- `knot1d_o2_cb5`: 1-D cubic knots, order 2, 50 channels;
- `static_o2_cb5`: static image, order 2, 50 channels;
- `static_o0_10x10`: static, order 0, 100 channels (the prototype's 10 × 10);
- `static_o0_48x1`: static, order 0, 48 coils × L 1 (the known ra spill case).

### 5.1 Tool

`kernaudit` (rustlib/crates/hasty-bench/src/bin/kernaudit.rs) builds each kernel's source exactly as the engine does
(the same define prefix and .cu file), compiles it with NVRTC 13.0 for each arch with `--ptxas-options=-v --no-cache`,
parses registers, stack frame, spill stores / loads, static shared memory and barriers per entry, adds the dynamic
shared memory the engine requests for the shape, and computes the theoretical occupancy from §1.2 (registers in
256-register units per warp, warps floored to 4, +1 KB reserved shared memory per block). Families: tile forward /
adjoint (the autotune's loops), vs / vt (default and test geometries × vs_candidates), ra and kl (ra_candidates /
kl_candidates), FFT axis passes (axes of the 288×288×140 and 800×480×480 grids, plain and pruned), and `whatif` (larger vt SMAX, ra LISTMAX 2048 /
4096, CHK 48 / 64). Options: `--opt` (extra NVRTC options), `--subst from=>to` (what-if source edit), `--cuobjdump`
(cross-check each cubin), `--list`.

Checks done: the occupancy rules reproduce the 4090 numbers of sample_kernels.md (ra 162 registers → 3 blocks × 4 warps;
vs 168 registers, 3-warp blocks → 4 blocks; ps 127 registers, 8-warp blocks → 2 blocks); ptxas numbers equal cuobjdump's
`--dump-resource-usage` on every ra cubin checked. NVRTC 13 caches cubins itself and a cache hit returns no ptxas
report, hence `--no-cache` (found when repeated runs gave empty reports).

### 5.2 Spot checks: one spill case and the FFT passes

ra adjoint, case 48 coils × L 1, order 0 (static), LISTMAX 1024, all 16 candidates:

| nxg cb chk | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|
| 2 3 16 (prototype best) | 147 regs, 12 warps | **128 regs, spills 20 / 36 B**, 16 warps | **128 regs, spills 24 / 40 B** | 121, no spill | 116, no spill, 16 warps |
| same, `__launch_bounds__(NTA, 4)` | 128, 16 warps | 126, no spill, 16 warps | 127, no spill | | 116 |
| same, `__launch_bounds__(NTA, 1)` | 147 | 147, no spill, 12 warps | 149 | | 152 |

`--maxrregcount` does not remove the sm_89 spill.

FFT axis passes (plain loads / stores; the fused pad / crop hooks are not audited), registers / warps per SM:

| pass | sm_80 | sm_89 | sm_90 | sm_120 |
|---|---|---|---|---|
| 288×288×140 x / y, n 288, 16 lines × 36 threads, 73.7 KB | 56 / 36 (2 blocks) | 56 / 18 (1 block, shared memory) | 56 / 36 (2 blocks, registers) | 46 / 18 (1 block) |
| 288×288×140 z, n 140, 35.8 KB | 48 / 40 | 48 / 20 | 44 / 40 | 46 / 20 |
| 800×480×480 x, n 800 | 8 lines, 102 KB: 58 / 25 | 4 lines, 51 KB: 58 / 13 | 8 lines: 58 / 25 | 4 lines: 48 / 13 |
| 800×480×480 y / z, n 480 | 16 lines, 123 KB: 59 / 30 | 8 lines, 61 KB: 53 / 15 | 16 lines: 58 / 30 | 8 lines: 40 / 15 |

The shape rule already picks more lines per block on sm_80 / sm_90 for the 800×480×480 grid; on 100 KB parts its x pass
runs at 13 warps per SM (4 lines per block): a candidate for items 8 / 11.

### 5.3 Full run: spills, stack frames, occupancy

Flags per family and arch: candidates / spilling / stack frame without spill / below 8 warps per SM:

| family | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|
| tile forward | 500 / 57 / 443 / 155 | 496 / 5 / 491 / 228 | 500 / 21 / 479 / 90 | 500 / 7 / 493 / 90 | 496 / 35 / 461 / 228 |
| tile adjoint | 98 / 12 / 86 / 5 | 98 / 8 / 90 / 20 | 98 / 15 / 83 / 0 | 98 / 14 / 84 / 0 | 98 / 9 / 89 / 20 |
| vs forward (orders 0–1) | 68 / 0 / 0 / 16 | 68 / 0 / 0 / 30 | 68 / 0 / 0 / 16 | 68 / 0 / 0 / 16 | 68 / 2 / 0 / 42 |
| vt forward (orders 2–3) | 35 / 0 / 0 / 8 | 33 / 0 / 0 / 15 | 35 / 0 / 0 / 5 | 35 / 3 / 0 / 5 | 33 / 3 / 0 / 15 |
| ra adjoint | 114 / 16 / 98 / 0 | 114 / 5 / 109 / 21 | 114 / 13 / 101 / 0 | 114 / 3 / 111 / 0 | 114 / 3 / 111 / 21 |
| kl adjoint | 58 / 0 / 58 / 0 | 58 / 2 / 56 / 12 | 58 / 4 / 54 / 0 | 58 / 5 / 53 / 0 | 58 / 1 / 57 / 12 |
| FFT axis passes | 12 / 0 / 0 / 0 | same | same | same | same |

(The candidate counts differ slightly by arch because shared memory admits different shapes.)

- **Spills are per arch, not per source.** The same candidate spills on one architecture and not on another, in both
  directions (e.g. vel2d tile forward cpbf 2 ppf 4 chunk 64 fsb 2: 71 registers on sm_89, 64 with 32 / 32 B spills on
  sm_120; ra nxg 4 cb 2 for the velocity case: spills on sm_89 / sm_90, not on sm_100 / sm_120). sm_80 has the most
  spilling tile and ra candidates (ptxas there uses ≈ 14–16 more registers per thread). The velocity reconstruction's
  chosen shapes do not spill on any arch (key-shape table below).
- **Stack frames:** every ra and tile kernel has a 32-byte stack frame and every kl kernel a 48-byte one, without
  spills, on every arch; vs, vt and the FFT passes have none. A stack frame is local memory (L1-cached); a dynamically
  indexed small array is the usual cause. Small, but worth removing when those kernels are next edited (§4 item 3).
- **Shared memory limits the 100 KB parts.** Share of candidates whose occupancy is limited by shared memory (alone or
  tied with registers): ra 98 / 114 on sm_89 and 103 / 114 on sm_120, but 32 / 114 on sm_90; vt 33 / 33 on sm_89 and
  sm_120, 6 / 35 on sm_90. Of the ra candidates limited by shared memory on sm_89, 66 of 98 become register-limited on
  sm_90 (median 1.75× the warps, up to 4×); sm_80: median 1.33×.
- Registers against sm_89 (same shape, median / range): ra sm_80 +16 (−4 … +45), sm_90 +1, sm_120 −5 (−52 … +30); vs
  sm_120 +22 (−60 … +88); tile forward sm_80 +14. The autotune must be rerun on each card (item 0).

Key shapes (the autotune's choices on the 4090 for the velocity path and the documented prototype bests): registers /
warps per SM (limiter), S = spill stores / loads in bytes:

| case | kernel | shape | sm_80 | sm_89 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|---|---|---|
| vel2d_o3_cb5 | ra | nxg 4 cb 7 chk 16 (29.6 KB) | 156 / 12 (regs) | 128 / 12 (smem) | 125 / 16 (regs) | 123 / 16 (regs) | 122 / 12 (smem) |
| vel2d_o3_cb10 | ra | nxg 4 cb 7 chk 16 | 156 / 12 (regs) | 128 / 12 (smem) | 125 / 16 (regs) | 123 / 16 (regs) | 122 / 12 (smem) |
| vel2d_o3_cb5 | kl | nxg 4 cb 7 chk 16 | 157 / 12 (regs) | 160 / 12 (regs + smem) | 157 / 12 (regs) | 123 / 16 (regs) | 125 / 12 (smem) |
| vel2d_o3_cb5 | vt | 4 × 4 × 16, SMAX 48, nwc 4 (76 KB) | 254 / 8 | **254 / 4 (smem)** | 254 / 8 (regs) | 254 / 8 | **236 / 4 (smem)** |
| vel2d_o3_cb5 | vt | same, nws 2 (82 KB) | 232 / 8 | 236 / 8 | 240 / 8 | | 238 / 8 |
| vel2d_o3_cb10 | vt | 4 × 4 × 16, SMAX 48, nwc 7 | 250 / 7 | 252 / 7 | 254 / 7 | 249 / 7 | 254 / 7 |
| static_o0_10x10 | vs | 4 × 4 × 16, SMAX 64, nwc 7 | 240 / 7 | 240 / 7 | 254 / 7 | 254 / 7 | 248 / 7 |
| static_o0_48x1 | ra | nxg 2 cb 3 chk 16 | 147 / 12 | 128 / 16 S 20 / 36 | 128 / 16 S 24 / 40 | 121 / 16 | 116 / 16 |
| static_o2_cb5 | ra | nxg 4 cb 7 chk 16 | 128 / 16 | 121 / 12 (smem) | 123 / 16 | 121 / 16 | 121 / 12 (smem) |

The vt forward at 4 warps per SM on sm_89 is consistent with its measured loss on the sparse knot path
(sample_kernels.md §16.5: items 4× emptier than vt needs; vt kept only at 13.3 M per encode); on sm_90 the same shape
holds 8 warps.

Larger shapes for big shared memory (`whatif`, velocity case 50 channels): ra LISTMAX 2048 / 4096 at CHK 16 keep 16
warps on sm_90 (8 on sm_89: they cost a block there); ra CHK 48 / 64 fit on sm_90 but drop to 8 warps (nxg 4 cb 7 at
CHK 48: 80.7 KB); vt SMAX 96 / 128 fit only on sm_80 / sm_90 and hold 4 warps (143.7 / 188.7 KB, one block). So the
useful sm_90 extension is the longer item list, not the bigger chunk or item.

### 5.4 Launch-bounds minimum blocks (what-if runs)

`__launch_bounds__(NT, m)` asks ptxas to fit m blocks per SM, i.e. caps registers at 64 K / (m · NT).

| variant | effect over all candidates |
|---|---|
| ra / kl, m = 4 (≤ 128 registers) | spills rise on every arch (ra sm_89 5 → 15, sm_100 3 → 12; kl sm_80 0 → 17); warps down in more shapes than up |
| ra / kl, m = 3 (≤ 170 registers) | registers +23 … +35 median (ptxas uses the allowance); warps down in most shapes |
| vs / vt, m = 2 (≤ 128 registers at 256 threads) | vt spills 0 → 18–20 on every arch (−43 registers median); vs spills 0 → 10–17 |

So a blanket hint is harmful. Per shape it is useful: choosing the best of {no hint, m 3, m 4} without spills removes
**every** baseline ra / kl spill on every arch (sm_80 16, sm_89 7, sm_90 17, sm_100 8, sm_120 4 shapes) and adds warps
without spilling in 1 (sm_89), 3 (sm_120), 11 (sm_80), 13 (sm_90) and 17 (sm_100) shapes. The velocity ra shape is
unchanged by m 4 (127–128 registers, still shared-memory-limited) and loses a block with m 3 on sm_90. For vs / vt the
hint helps nowhere that matters (at the vs 10 × 10 shape m 2 forces 128 registers with 508–616 B of spills).
Hence §4 item 3: a per-shape candidate dimension for ra / kl, selected by a resource check, not a default.

### 5.5 Not audited

- The fused pad / crop hooked x passes (hook code is private to segnufft/fused.rs): their shared memory (73.7 KB at
  n 288) is known; their registers are not.
- The multi-image kernels (static order 3, fat paths), chirp and the plain NUFFT.
- sm_120 cluster runtime, every timing: compile-time counts only.

## 6. First hour on a new GPU (draft)

Goal: arrive with built binaries and a script, spend the hour measuring, leave with results and a tune cache in the
repo. Times are budgets.

0. **Before booking (local, CPU):** build the release binaries for the node's CPU architecture (aarch64 for GH200)
   and the container (plan_cluster.md §3.3); run the audit for the node's arch
   (`research/experiments/arch_audit/run.sh sm90 --arch sm_90` → `out_sm90/`) and check the flags of §5.3; run the GPU test binaries once on the 4090 so they are known-good.
1. **Node facts (5 min):** `nvidia-smi -q` (driver, clocks, power cap, ECC, MIG off, persistence mode),
   `nvidia-smi topo -m`, `lscpu`, `numactl -H`, free disk on the scratch volume. Set `HASTY_KERNEL_CACHE` and the tune
   cache on persistent storage, `HASTY_SITE`, `CUDA_VISIBLE_DEVICES`; unset `HASTY_GPU_WATTS` / `HASTY_GPU_DUTY`.
   Save everything to `research/experiments/arch_runs/<gpu-slug>-<date>/node.txt`.
2. **Correctness (10 min):** the GPU test binaries of hasty-nufft and hasty-fft (seg_ra_vt, seg_vs, seg_kl,
   seg_knot_kv, seg_knot_hi, seg_fft_traffic, seg_pruned, chanfft), first run also fills the kernel cache.
   compute-sanitizer memcheck on one small ra / vt / kl case with `HASTY_NVRTC_LINEINFO=1`.
3. **Microbenchmarks (10 min):** `proto_gather --mode ldsbench` (shared-load cost per instruction width), `--mode
   redbench` (global RED throughput: the atomics vs owner-computes trade), `fft_bench` against `cufft_baseline`, and the
   PCIe / C2C rate opbench prints at start.
4. **Operator sweep (25 min):** opbench on flow_thorax (288×288×140 grid), 10 virtual coils, L 10, order 3, 2-D knots 6:1,16:3:
   `--kl-compare` at `--scan-min 0.44` and `4.4`, each with `--max-grid-gb` for coil batch 5 / 10 / all;
   `--new-compare` for the static order-2 case; `HASTY_SEG_FFT_L2FRAC` 0.4 / 0.5 / 0.85 and `HASTY_SEG_FFT_SLAB=0`;
   all with `HASTY_SEG_VSA_LOG=1 HASTY_SEG_VS_LOG=1 HASTY_SEG_KL_LOG=1` so every candidate's time is in the log, and
   `--jsonl` for the rows. This answers §2.1's projection directly (kernel vs FFT split per card). On sm_90 add a
   `HASTY_SEG_VT=1` row on the sparse knot case: vt holds 8 warps there against 4 on the 4090 (§3.3).
5. **Phase-2 variants (remaining time):** the items of §4 built by then, each as an A/B against its off switch in the
   same opbench invocation.
6. **Save (5 min):** logs, `--jsonl` rows, `nvidia-smi` during the run, the tune cache directory, and the kernaudit
   summary for that arch, all under `research/experiments/arch_runs/<gpu-slug>-<date>/`; copy back and commit. The
   engine reuses the tune cache on the next run on the same GPU model, driver and NVRTC version (item 0).

A script `research/experiments/arch_runs/first_hour.sh` should run steps 1–4 and 6 unattended with a per-step time
limit (phase 2; it needs item 0 for the cache).

## 7. Files

- `rustlib/crates/hasty-bench/src/bin/kernaudit.rs`: the audit tool (new file; mirrors the engine's source
  generation and shared-memory formulas, see its header).
- `research/experiments/arch_audit/run.sh`: rerun script (`run.sh` → `out/`; `run.sh <tag> --subst … --families …` for
  what-if variants → `out_<tag>/`).
- `research/experiments/arch_audit/analyze.py`: digest of results.jsonl (flags, limiter shifts, register differences,
  key shapes, what-if comparisons): `analyze.py out/results.jsonl out_lb4/results.jsonl …`.
- `research/experiments/arch_audit/out/` (full run), `out_lb3/`, `out_lb4/`, `out_vlb2/` (launch-bounds what-ifs):
  `results.jsonl`, `summary.md`; `out*.log` progress logs.
- `research/experiments/arch_audit/STATUS.md`: status and how to rerun.

## 8. Sources

NVIDIA:
1. CUDA Programming Guide v13.5, Compute Capabilities: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
2. CUDA Programming Guide v13.5, Advanced Kernel Programming: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
3. CUDA Programming Guide v13.5, L2 Cache Control: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
4. CUDA Programming Guide v13.5, C++ Language Extensions (atomics): https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
5. CUDA C++ Programming Guide 12.9.1 (legacy): https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html
6. Hopper Tuning Guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
7. Ampere Tuning Guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
8. Ada Tuning Guide: https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
9. Blackwell Tuning Guide: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
10. PTX ISA 9.5: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
11. CUDA 13.0 Release Notes: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
12. NVRTC 13.0: https://docs.nvidia.com/cuda/archive/13.0.0/nvrtc/index.html
13. nvcc 13.0: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-compiler-driver-nvcc/index.html
14. Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
15. A100 whitepaper: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
16. A100 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
17. NVIDIA Hopper Architecture In-Depth: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
18. H100 product page: https://www.nvidia.com/en-us/data-center/h100/
19. H100 datasheet (reseller mirror): https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf
20. H100 PCIe product brief PB-11133: https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf
21. H200 product page: https://www.nvidia.com/en-us/data-center/h200/
22. GH200 datasheet: https://dam-cdn.nvd.orangelogic.com/AssetLink/tkno03fa645gb7f33w60rf200d330420.pdf
23. GH200 whitepaper v1.21 (third-party mirror): https://catalogone.com/wp-content/uploads/2024/06/NVIDIA-Grace-Hopper-Superchip-the-Whitepaper.pdf
24. GH200 product page: https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/
25. Ada GPU architecture whitepaper: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
26. RTX Blackwell GPU architecture whitepaper v1.1: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
27. RTX 4090 product page: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
28. RTX 5090 product page: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
29. L40S product page: https://www.nvidia.com/en-us/data-center/l40s/
30. L40S product brief PB-11470 (PNY mirror, search snippet only): https://www.pny.com/File%20Library/Company/Support/product-briefs/data-center-gpus/NVIDIA-L40S-Product-Brief.pdf

Secondary (not NVIDIA; used only where marked):
- S1. compute-verification/verity PR #10802 (quotes H100 whitepaper clocks): https://github.com/compute-verification/verity/pull/10802
- S2. Chips and Cheese, H100 L2: https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth
- S3. arXiv 2501.12084, Dissecting Hopper: https://arxiv.org/html/2501.12084v1
- S4. UBC MICRO 2024, GPU NoC: https://people.ece.ubc.ca/aamodt/publications/papers/realgpu-noc.micro2024.pdf
- S6. NASA HECC knowledge base: https://www.nas.nasa.gov/hecc/support/kb/basics-on-nvidia-gpu-hardware-architecture_704.html
- S7. H200 aggregator pages: https://hmc-tech.com/gpus/nvidia-h200-sxm, https://videocardz.net/nvidia-h200-sxm-141gb
- S8–S11. sm_120 cluster reports: https://github.com/jesdga95/basalt/issues/85, https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471, https://github.com/NVIDIA/TransformerEngine/issues/3765, https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/

Not verified: H100 SXM / H200 / GH200 clocks and SM counts of H200 / GH200 (secondary only), L40S L2 size, H200 L2
size, tensor-core TF32 rates (§3.5, from memory), the sm_120 cluster runtime behaviour, per-arch global-atomic rates.
