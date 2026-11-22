# Block-floating-point storage of the gradient's buffers (2026-11-20)

> **Correction (2026-11-21, the user's): segment diagonals are always the stored SVD-basis diagonals** (the
> reconstruction uses the noise-weighted SVD segment basis, `svdn`). Their only compressed form is BFP storage,
> decoded in the pad / crop staging. The "on the fly from fixed-point turns + ratemap" option for node bases
> (§3.3, §5 row 3, §6, §7 step 3 below) does not apply and is withdrawn: not to be implemented, planned or
> recommended. Those passages are kept only as the record of what was measured.

Question (the user's): store the float buffers of the reconstruction as b-bit integer mantissas sharing one exponent
per block ("block floating point", BFP), so DRAM traffic, VRAM, host RAM and PCIe bytes fall by about (32 − b) / 32,
while all arithmetic stays f32 in registers. Three things had to be settled: (1) whether the packing cost depends on b;
(2) which b each buffer can take, measured on real data along the engine's real blocking axis, combined into one
gradient error and compared with the NUFFT error and the noise; (3) where the conversions go.

Decision this document supports: the codec (done, tested, new crate `hasty-bfp`), the b per buffer (§6) and the
integration plan (§7) for the next phase. Nothing in the engine was changed in this phase.

## 0. Answer

1. **Packing cost does not depend on b.** A block of 32 reals (16 complex) is exactly b 32-bit words for every b. Lane
   j of a warp loads word j (one coalesced 4b-byte access); value i is rebuilt from words ⌊ib/32⌋ and the next with two
   shuffles, one funnel shift and a sign extension. Measured on the RTX 4090 (§2): pack and unpack of 2^27 reals run at
   85–91 % of DRAM peak for every b from 8 to 24, at 0.97–1.13 × the time a plain copy of the same bytes takes. 22 bits
   cost the same as 20 or 24, also in executed instructions (Nsight Compute: 75 per real to pack, 36 to unpack in the
   warp layout for b = 18 … 24; only b = 8 and 16, which never straddle, are cheaper). Two qualifications: (a) the warp
   and half-warp layouts cost ≈ 110 instructions per real for a load + store, which would push the FFT passes (today
   14–23 % of issue slots) towards issue-bound; the per-thread run layout through shared memory costs 3–5 × fewer
   (25–43 per real for b = 18 … 24) and is the one to integrate there; (b) per-thread runs of 16 or 32 values loaded straight from
   global memory (b = 18, 22, odd b) pack at only 60–76 % of peak; through shared memory this does not arise.
2. **Accuracy (§3, real phantom ph0001, 10 virtual coils × 10 segments, blocks along the engine's flat channel-fastest
   index).** Every stored buffer contributes a relative error of about c · 2^−b with c ≈ 1.0–2.0, independently of the
   buffer: image-side buffers (image, coil maps, segment diagonals) 1.0–1.6 · 2^−b, FFT intermediates and the grid
   1.95–2.0 · 2^−b per buffer (bfp16 3.0e-5, bfp20 1.9e-6, bfp22 4.7e-7, bfp24 1.2e-7), forward and adjoint alike.
   Buffers add in quadrature (chained runs match to 3–4 digits). Channel blocks are slightly better than the user's
   x-axis blocks. [Withdrawn, see the correction at the top: segment diagonals are SVD-basis diagonals, stored, BFP at
   most.] k-space data: bfp16 keeps the worst block's step ≤ σ / 4.
   Solver state: quantised iterates do not accumulate error (they stall at ≈ 3 · 2^−b); a BFP gradient and secant
   pair leave the fixed point exact even at 12 bits.
3. **End-to-end (§4).** Early iterations: the Gram A^H A x with every buffer rounded has 2.5e-5 / 1.6e-6 / 9.8e-8
   relative error at b = 16 / 20 / 24 (the crop's coherent coil sum suppresses the grid buffers 12 ×). Near convergence
   the forward error is amplified by rms(A x) / σ_k = 4.0 like any forward error: relative to the noise-driven gradient
   2.0e-4 / 1.3e-5 / 3.2e-6 / 8.6e-7 at b = 16 / 20 / 22 / 24, against 4e-5 from the NUFFT's own 1e-5 forward error.
   Against the noise every b ≥ 16 is invisible (bfp16 adds 2e-4 σ_k).
4. **Recommended b (§6).** bfp22 for the DRAM-resident FFT buffers, the grid, the image working copy and the coil
   maps at the final pyramid level (1/13 of the NUFFT error, −24 % of the FFT-side bytes); bfp20 / bfp16 at earlier
   levels; segment diagonals stored as BFP (SVD basis; on-the-fly withdrawn); f32 for the L2-resident intermediates, the scatter output (atomics) and the residual;
   bfp16 for device k-space, the gradient download and the host g, s, y; f32 master x. Step 5 (§9) refines this for
   the host state: bfp16 without a post-step prior, bfp24 with P²nP; device k-space by the σ rule (15 bits on ph0001).
5. **Where to convert (§5).** Only inside the producer's store and the consumer's load (pack in an FFT pass epilogue,
   unpack in the next pass's first-stage load or the gather's grid load). A standalone conversion pass costs a full
   read + write and cancels the saving on every DRAM-bound buffer. The one exception is a host link: a pack kernel
   before a device → host copy costs DRAM time ≈ 1/40 of the PCIe time it saves.

## 1. Format and codec

Files: `rustlib/crates/hasty-bfp/` (new crate; one line added to the workspace members).

| file | content |
|---|---|
| `src/bfp.cuh` | device codec for NVRTC inclusion (`hasty_bfp::BFP_CUH`), templates for b = 8 … 24 |
| `src/lib.rs` | host codec (`Bfp`, `BfpBuf`), bit-identical to the device codec, rayon-parallel |
| `src/kernels.cu`, `src/gpu.rs` | test and benchmark kernels (pack / unpack per layout, random-access decode, copies) |
| `src/bin/bfp_bench.rs` | microbenchmark (`bench`) and CPU round-trip tool (`roundtrip`, cross-check of the Python emulation) |
| `tests/cpu.rs`, `tests/gpu.rs` | bit layout, bounds, edge cases; GPU = CPU bit for bit |

**Format.**
- A block is 32 consecutive reals (16 complex values, re / im interleaved) of a buffer, aligned to 32 reals in the
  buffer's flat index. The writer pads a tail block with zeros.
- Mantissas: b-bit two's complement integers q_i. Value i occupies bits [ib, ib + b) of the block's bit stream, which is
  b little-endian 32-bit words (block k at words kb … kb + b − 1). Values straddle words, never blocks.
- Exponent: one byte per block in a separate array (4 per 32-bit word), es = e + 125, value x_i ≈ q_i · 2^(e − b + 1),
  every |x_i| < 2^e.
- Choice of e: the smallest e in [b − 127, 128] with max |x| < 2^e and round(max |x| · 2^(b − 1 − e)) ≤ 2^(b − 1) − 1.
  So nothing saturates and every value obeys **|x − x̂| ≤ 2^(e − b)** (half a step); the CPU test checks this bound and
  the minimality of e for every block at every b. The lower limit keeps the scale factors normal floats (tiny blocks
  quantise on a 2^−126 grid and may decode to subnormals, so kernels using the codec must not flush subnormals). The
  upper limit keeps every decoded value finite: only values within half a step of 2^128 saturate, with an error below
  one step.
- Rounding: to nearest, ties to even, of the exact power-of-two scaling x · 2^(b − 1 − e) (`__float2int_rn` on the
  device, `round_ties_even` on the host). The block max is taken on the integer bits of |x|, so a NaN is never dropped
  (an `fmaxf` reduction would drop it).
- Non-finite values: a block with an Inf or NaN gets es = 255 and decodes as all-NaN. Nothing is silently saturated;
  the solver's NaN checks see it.
- Size: (4b + 1) / 16 bytes per complex value: bfp24 6.0625 B, bfp22 5.5625 B, bfp20 5.0625 B, bfp16 4.0625 B, against
  8 B for f32.

**Access patterns over the same bytes** (the consumer chooses; the memory format is the same):

| layout | who holds what | mantissa traffic | shuffles per value (pack / unpack) | fits |
|---|---|---|---|---|
| warp block (`bfp_warp_*`) | lane j = value j; lane j < b loads / stores word j | one coalesced 4b-byte access per block | ⌈32/b⌉ + 1 (≤ 3 for b ≥ 16; 4 for b 11–15) / 2 (1 when b divides 32) | one value per lane in registers, no shared memory at hand; 75 / 36 instructions per real (§2) |
| half warp, complex (`bfp_hw_*`) | lane t of a 16-lane group = complex t; lane t holds words t and t + 16 | one 4b-byte access per half warp | 2–3 shuffles of 64-bit chunks per word / 3 word fetches | strided FFT passes in place (`FFT_INPLACE`, no free shared buffer): load in the first stage; 68 / 40 instructions per real |
| per-thread run (`bfp_run_*`) | a thread owns V = 32 / gcd(b, 32) consecutive values = W = Vb / 32 whole words (b 24: 4 values in 3 words; 20: 8 in 5; 16: 2 in 1) | W words per thread | none for mantissas; the exponent is shared by 32 / V lanes (group max for packing) | **the FFT passes through shared memory** (store from the last stage's shared output, decode into the free shared buffer before stage 1), pad / crop row staging; 15–28 / 10–15 instructions per real (§2); `bfp_run_pack` / `bfp_run_unpack` work on registers from any source |
| random access (`bfp_load_c`, `bfp_load_r`) | any lane, any index | 1–3 word loads (in the block) + the exponent byte | none | the gather (vs / vt lanes own channels of one cell; 16-lane groups align with blocks only when the channel count is a multiple of 16) |

The half-warp functions use the half warp's own shuffle mask, so the two halves of a warp may diverge (one half
outside a pass's input range); the GPU test exercises that at the buffer tail. `nvcc -Xptxas -v` for sm_89: no stack
frame and no spills in any kernel at any b (14–48 registers).

## 2. Packing cost against b: microbenchmark

RTX 4090, power limit 300 W (as found), `bfp_bench bench 27 21`: 2^27 reals (537 MB as f32), median of 21 runs, CUDA
events, grid-stride kernels of 256 threads, 8 blocks per SM. Bytes per kernel: 4n (f32 side) + (b/8 + 1/32) n (BFP
side). "t / t_copy" is the kernel time over the time the f32 copy kernel takes for the same number of bytes (the
copy reaches 892 GB/s, 89 % of the 1008 GB/s peak). Logs: `research/experiments/bfp/bench.log` (run 2: `run2/`).

Each cell: median time, achieved bandwidth as % of the 1008 GB/s peak, time over the f32-copy time for the same bytes.

| b | warp pack | warp unpack | half-warp pack | half-warp unpack | run pack | run unpack | random-access unpack (complex per thread) |
|---|---|---|---|---|---|---|---|
| 8 | 0.792 ms, 85 %, 1.04 | 0.842 ms, 80 %, 1.11 | 0.770 ms, 87 %, 1.02 | 0.799 ms, 84 %, 1.05 | 0.733 ms, 91 %, 0.97 | 0.821 ms, 82 %, 1.08 | 0.784 ms, 85 %, 1.03 |
| 12 | 0.850 ms, 87 %, 1.02 | 0.934 ms, 79 %, 1.12 | 0.822 ms, 90 %, 0.99 | 0.942 ms, 78 %, 1.13 | 0.814 ms, 90 %, 0.98 | 0.856 ms, 86 %, 1.03 | 0.908 ms, 81 %, 1.09 |
| 16 | 0.921 ms, 87 %, 1.01 | 0.961 ms, 84 %, 1.06 | 0.899 ms, 89 %, 0.99 | 0.953 ms, 84 %, 1.05 | 0.907 ms, 89 %, 1.00 | 0.931 ms, 86 %, 1.02 | 0.935 ms, 86 %, 1.03 |
| 18 | 0.972 ms, 86 %, 1.03 | 1.019 ms, 82 %, 1.08 | 0.965 ms, 87 %, 1.02 | 1.034 ms, 81 %, 1.09 | 1.103 ms, 76 %, 1.16 | 0.953 ms, 88 %, 1.01 | 1.030 ms, 81 %, 1.09 |
| 20 | 1.019 ms, 85 %, 1.03 | 1.059 ms, 82 %, 1.07 | 1.009 ms, 86 %, 1.02 | 1.062 ms, 82 %, 1.08 | 0.981 ms, 89 %, 1.00 | 0.997 ms, 87 %, 1.01 | 1.068 ms, 81 %, 1.08 |
| 22 | 1.046 ms, 86 %, 1.02 | 1.101 ms, 82 %, 1.08 | 1.036 ms, 87 %, 1.01 | 1.127 ms, 80 %, 1.10 | 1.531 ms, 59 %, 1.50 | 1.034 ms, 87 %, 1.01 | 1.097 ms, 82 %, 1.07 |
| 24 | 1.096 ms, 85 %, 1.03 | 1.120 ms, 84 %, 1.06 | 1.086 ms, 86 %, 1.02 | 1.135 ms, 83 %, 1.07 | 1.084 ms, 86 %, 1.02 | 1.099 ms, 85 %, 1.04 | 1.127 ms, 83 %, 1.06 |

So the packing cost is flat in b: the codec adds nothing measurable to a bandwidth-bound kernel's time for any b in
8 … 24, in any layout except the per-thread pack with 16- or 32-value runs. Straddling (b not dividing 32) costs
nothing visible: b = 18 and 22, where blocks are not even 32-byte-sector aligned, run at the same bandwidth as 16 and
24 (L2 merges the partial sectors that neighbouring blocks complete).

**Instruction cost.** At DRAM speed the codec hides behind the memory traffic, so the microbenchmark cannot show what
it costs inside a kernel that also does other work. Executed instructions per real value (Nsight Compute,
`smsp__inst_executed` × 32 / n, n = 2^22; `ncu_inst.csv`, `ncu_inst_summary.txt`; whole kernels including loads,
stores and index arithmetic; a float4 copy is 6.25):

| b | warp pack | warp unpack | half-warp pack | half-warp unpack | run pack | run unpack | random-access unpack | run length V |
|---|---|---|---|---|---|---|---|---|
| 8 | 72.9 | 31.9 | 63.6 | 29.7 | 25.2 | 12.9 | 23.9 | 4 |
| 12 | 82.3 | 36.3 | 59.5 | 35.3 | 18.9 | 11.7 | 26.1 | 8 |
| 16 | 64.9 | 31.9 | 48.4 | 25.4 | 42.8 | 20.7 | 16.9 | 2 |
| 18 | 75.1 | 36.3 | 67.9 | 39.9 | 15.1 | 10.0 | 26.6 | 16 |
| 20 | 75.1 | 36.3 | 67.9 | 39.9 | 19.7 | 11.9 | 26.6 | 8 |
| 22 | 75.1 | 36.3 | 68.2 | 40.1 | 15.2 | 10.1 | 26.6 | 16 |
| 24 | 74.8 | 36.3 | 67.9 | 39.9 | 27.7 | 15.4 | 26.6 | 4 |

- Warp and half-warp layouts: **identical instruction counts for b = 18, 20, 22 and 24**; only b = 8 and 16 (values
  never straddle a word) are cheaper. This confirms the expected answer: 22 bits cost the same as 20 or 24.
- Per-thread runs: 3–5 × fewer instructions (no shuffles, all shifts compile-time). Their cost depends on the run
  length V = 32 / gcd(b, 32), not on b: long runs amortise the per-run index work and the shorter group-max (b = 16
  with V = 2 is the worst case; any multiple of V is a valid run, so b = 16 can use 8-value runs of 4 words).
- L2-resident timing (2^22 reals, `bench_l2.log`): ≈ 10 µs per kernel, dominated by launch overhead. It shows the
  warp and half-warp layouts at 1.1–1.8 × the copy time and 16-value runs at up to 5 × (their per-thread 64-byte global
  loads); it is not used further.

**What this means inside the FFT passes.** The passes themselves, profiled with `fftpass_bench 288 288 140 225 225 112
50` (2 mm knot grid, 50 channels, no slabs; `ncu_fftpass.csv`):

| pass | time | DRAM throughput | issue slots busy | thread instructions per complex value |
|---|---|---|---|---|
| x (25200 rows × 4 channel tiles) | 5.59 ms | 93.5 % | 23 % | 126 |
| y (900 line tiles × 112 planes) | 7.46 ms | 90.1 % | 17 % | 99 |
| z (259200 column tiles) | 9.82 ms | 86.4 % | 14 % | 86 |

A BFP load plus a BFP store in the warp / half-warp layouts adds ≈ 2 × (68 + 40) ≈ 215 instructions per complex value:
the y pass would go from 99 to ≈ 315 per complex, and since BFP also shortens the pass (bfp20: × 0.71), its issue
utilisation would rise from 17 % to ≈ 75 %. That is too close to issue-bound for a DRAM-bound pass to stay at 90 % of
bandwidth. With per-thread runs through shared memory (store: the last stage already writes its outputs to shared
memory on the epilogue path, and a 16-line tile entry is 32 contiguous reals there; load: decode the tile into the
second shared buffer in a prologue, which is free until stage 1), the codec adds ≈ 2 × (20 + 12) ≈ 63 instructions
per complex at b = 20: ≈ 162 per complex, ≈ 39 % of the issue slots. **So the integration should use the per-thread run
layout through shared memory in the FFT passes**, and the warp / half-warp layouts only where values sit one per lane
in registers and shared memory is not available (`FFT_INPLACE` passes).

### Block geometry against the consumers

- **Strided FFT passes (y, z; forward and inverse).** A CUDA block holds 16 lines (`choose_shape` caps strided passes
  at LPB = 16), the line index is the flattened (x, channel) of a plane, so at each element position a half warp holds
  16 consecutive complex values of the flat index: **exactly one block**, aligned whenever the tile's first line is a
  multiple of 16 (rows of Nx · nch values; 320 · 50 and 288 · 50 are multiples of 16). Store: the epilogue packs the
  block from shared memory (`fin[e · LPB + k]`, 32 contiguous reals) in per-thread runs and writes b words + one byte.
  Load: a prologue decodes the tile's blocks in per-thread runs into the free shared buffer; in place passes use
  `bfp_hw_load_c` in the first stage (one word per lane, three shuffles, one exponent byte).
- **x pass.** Its lines are the channels of one (y, z) row; a 16-line tile is 16 channels of one cell. Blocks of the
  flat index line up with it only when nch is a multiple of 16 (coil batch 8 × L 10 = 80, or L 16). With nch = 50 (coil
  batch 5) the x pass's tiles straddle blocks. The x → y intermediate stays in L2 with the slabs (temporal_operator.md
  §23.5), so it does not need BFP: keep it f32 and pack in the y pass's epilogue.
- **Pad / crop rows.** The fused pad stages the image row, the coil rows and the segment rows along x
  (`segnufft/fused.rs`). Rows of N0 complex values start at v = (a2 · N1 + a1) · N0, so when N0 is not a multiple of 16
  (254, 225) rows straddle blocks: the prologue decodes the covering blocks (per-thread runs into the staging buffer)
  and keeps the row's range.
- **Gather (vs / vt / tile kernels).** The grid is read as cells of nch channels, 16-channel lane groups per cell.
  Random-access decode per lane works for any nch (1–3 L1-resident word loads); when nch is a multiple of 16 a group
  reads one block and can use the half-warp decode.
- **128-byte lines.** A block is 4b bytes: whole 32-byte sectors when b is a multiple of 8, four consecutive blocks are
  three whole 128-byte lines at b = 24. The benchmark shows no penalty for b = 18 or 22 either, so b need not be chosen
  for alignment.

## 3. Accuracy per buffer

### 3.1 Setup

- Data: ph0001 `sim_flow_10min/scan`: image 254 × 144 × 114 (water + fat proton density, fat at a fixed phase, smooth
  background phase), fine grid 320 × 180 × 144 (σ = 1.25). True coil maps (32 coils, `smaps_true`) compressed to 10
  virtual coils by the signal-region Gram (keeps 67 % of the energy; last / first eigenvalue 0.25: the virtual coils
  have a moderate spread). Ratemap: the recon field map (−206 … 283 Hz between the 1st and 99th percentile of the signal
  voxels, ±760 Hz extremes) and the true R2* (up to 6250 s⁻¹). L = 10 segments over 0.5 … 8.5 ms: the fixed-node basis
  (D_l(v) = e^{−(R2* + 2πi f) τ_l}, the non-knot default) and a rank-10 SVD basis (as svdn, unweighted).
- Engine layout: one coil batch of 5 virtual coils × 10 segments = 50 channels, grid [z][y][x][channel] with the
  channel fastest (la = c · L + l), image-side buffers [row][x]. **Blocks = 16 consecutive complex values of each
  buffer's flat index** (the codec), i.e. along channels for the grid buffers (straddling cells, as the y / z passes'
  tiles do) and along x for the image, coil maps and diagonals.
- Computation in f64, every buffer rounded with an emulation of the codec. The emulation equals the Rust codec value
  for value on 4.1 M reals (phantom rows plus values spread over 2^±87), for b = 8, 12, 16, 20, 22, 24; the only bit
  differences are −0.0 against +0.0 (`pycheck.log`). The f32 input rounding is not applied (it is the "f32" row).
- Metric per buffer: the relative L2 error the buffer's rounding puts into the operator output. The transforms after a
  buffer are unitary up to scale, so the relative rounding error of a grid buffer is the relative error it adds to the
  output. A chained run (every buffer rounded, errors propagated through the following passes) checks that the
  buffers' errors add in quadrature.
- Script: `research/experiments/bfp/acc.py` (phases image, forward, adjoint, kspace), results `acc_*.json`, logs
  `acc_*.log`.

### 3.2 Image-side buffers (pad output and crop, 50 channels)

Relative error of the pad output S_c P_l x (all 50 channels) when one buffer is stored in the format, and of the crop
Σ conj(S_c P_l) g of the exact pad output (the crop's reading of the maps):

| format | image x: pad | coil maps: pad / crop | diagonals, fixed nodes: pad / crop | diagonals, SVD basis: pad / crop | all three: pad / crop |
|---|---|---|---|---|---|
| f32 | 2.5e-08 | 2.5e-08 / 1.6e-08 | 2.3e-08 / 7.2e-09 | 2.6e-08 / 2.0e-08 | 4.2e-08 / 1.8e-08 |
| fp24 | 6.5e-06 | 6.5e-06 / 4.1e-06 | 5.8e-06 / 1.8e-06 | 6.6e-06 / 5.2e-06 | 1.1e-05 / 4.5e-06 |
| bf16 | 1.7e-03 | 1.7e-03 / 1.0e-03 | 1.5e-03 / 4.7e-04 | 1.7e-03 / 1.3e-03 | 2.8e-03 / 1.2e-03 |
| bfp8 | 4.6e-03 | 4.7e-03 / 2.9e-03 | 4.1e-03 / 1.3e-03 | 6.1e-03 / 4.1e-03 | 7.7e-03 / 3.2e-03 |
| bfp10 | 1.1e-03 | 1.2e-03 / 7.3e-04 | 9.9e-04 / 3.2e-04 | 1.5e-03 / 1.0e-03 | 1.9e-03 / 7.9e-04 |
| bfp12 | 2.8e-04 | 2.9e-04 / 1.8e-04 | 2.4e-04 / 7.7e-05 | 3.8e-04 / 2.5e-04 | 4.7e-04 / 2.0e-04 |
| bfp14 | 7.1e-05 | 7.3e-05 / 4.5e-05 | 5.9e-05 / 1.9e-05 | 9.5e-05 / 6.3e-05 | 1.2e-04 / 4.9e-05 |
| bfp16 | 1.8e-05 | 1.8e-05 / 1.1e-05 | 1.5e-05 / 4.7e-06 | 2.4e-05 / 1.6e-05 | 2.9e-05 / 1.2e-05 |
| bfp18 | 4.5e-06 | 4.6e-06 / 2.8e-06 | 3.6e-06 / 1.2e-06 | 5.9e-06 / 4.0e-06 | 7.3e-06 / 3.1e-06 |
| bfp20 | 1.1e-06 | 1.1e-06 / 7.1e-07 | 9.1e-07 / 2.9e-07 | 1.5e-06 / 9.9e-07 | 1.8e-06 / 7.7e-07 |
| bfp22 | 2.8e-07 | 2.9e-07 / 1.8e-07 | 2.3e-07 / 7.2e-08 | 3.7e-07 / 2.5e-07 | 4.6e-07 / 1.9e-07 |
| bfp24 | 6.9e-08 | 7.1e-08 / 4.4e-08 | 5.7e-08 / 1.8e-08 | 9.3e-08 / 6.2e-08 | 1.1e-07 / 4.8e-08 |

The error per buffer on the pad output is 1.0–1.6 · 2^−b (image 1.17, coil maps 1.2, fixed-node diagonals 0.96, SVD
diagonals 1.55); on the crop it is lower (the crop's channel sum). The coil maps' smoothness does not lower it: a block of 16 voxels along x of a
smooth map has max / rms close to 1, which BFP already uses fully; smoothness would only help a predictive (delta)
coder, not BFP. So low b for the coil maps is a choice by the error budget, not a property of smooth data.

### 3.3 Segment diagonals: stored, phase angle, or computed on the fly

Fixed-node diagonals are e^{−R2* τ_l} e^{−2πi f τ_l}: a known decay times a unit-modulus phase. Signal-weighted relative
error of the diagonals:

| representation | bytes per voxel for L = 10 | error |
|---|---|---|
| stored f32 (today) | 80 | 2.3e-08 |
| BFP 16 | 40.6 | 1.5e-05 |
| BFP 20 | 50.6 | 9.1e-07 |
| BFP 22 | 55.6 | 2.3e-07 |
| BFP 24 | 60.6 | 5.7e-08 |
| phase angle 12 bits + decay from R2* (4 B) | 19 | 4.4e-04 |
| phase angle 16 bits + decay from R2* (4 B) | 24 | 2.8e-05 |
| phase angle 20 bits + decay from R2* (4 B) | 29 | 1.7e-06 |
| phase angle 24 bits + decay from R2* (4 B) | 34 | 1.1e-07 |
| on the fly, f32 radians (f, R2*) | 8 | 1.8e-07 |
| on the fly, u32 turns (phase0, dphase, R2*) | 12 | 8.3e-08 |

- The phase angle in b bits has 1.8 · 2^−b, about twice BFP's error at the same b, but takes b bits per value instead
  of BFP's 2b: at equal bytes it is far more accurate (a 16-bit angle, 2 B per value: 2.8e-5; bfp8, 2.06 B per
  value: 4.1e-3). It needs a sincos per value and a separate decay.
- On the fly, from per-voxel 32-bit fixed-point turns (phase0 = frac(−f τ_0), dphase = frac(−f Δτ), exact integer wrap
  for phase_l = phase0 + l · dphase) plus R2*, the diagonals come out at the f32 rounding level (8.3e-8) from 12 bytes
  per voxel for all L segments: 5 × fewer bytes than bfp24 storage (6.7 × fewer than f32) and exact to f32. With radians in f32 (f · τ
  rounded) it is 1.8e-7. The cost is L sincos + exp per voxel per pad / crop tile, on the XU pipe that is 0.5–2.7 %
  busy. This works for any node set (store f in fixed point and the τ_l as constants), not for the SVD basis, whose
  diagonals need ~96 exponentials per voxel: those stay stored (BFP).

### 3.4 FFT intermediates and the grid (forward)

Relative rounding error of each forward buffer (input = the exact pad output, so these are the grid side only), the
chained run with all three buffers rounded and propagated through the following passes, the quadrature sum of the three
single-buffer errors, and the chained run's rms error over the per-channel k-space noise σ_k at image SNR 20 (unnormalised
FFT, σ_k = σ_image √N; the grid's rms over the k-space ball is 7.4 σ_k):

| format | x → y | y → z | grid (read by the gather) | all three chained | quadrature of the three | chained rms error / σ_k |
|---|---|---|---|---|---|---|
| f32 | 2.5e-08 | 2.5e-08 | 2.5e-08 | — | — | — |
| fp24 | 6.5e-06 | 6.5e-06 | 6.5e-06 | — | — | — |
| bf16 | 1.7e-03 | 1.7e-03 | 1.7e-03 | — | — | — |
| bfp8 | 7.9e-03 | 7.6e-03 | 7.7e-03 | — | — | — |
| bfp10 | 2.0e-03 | 1.9e-03 | 1.9e-03 | — | — | — |
| bfp12 | 4.9e-04 | 4.8e-04 | 4.8e-04 | — | — | — |
| bfp14 | 1.2e-04 | 1.2e-04 | 1.2e-04 | — | — | — |
| bfp16 | 3.1e-05 | 3.0e-05 | 3.0e-05 | 5.2e-05 | 5.2e-05 | 2.1e-04 |
| bfp18 | 7.7e-06 | 7.4e-06 | 7.5e-06 | — | — | — |
| bfp20 | 1.9e-06 | 1.9e-06 | 1.9e-06 | 3.3e-06 | 3.3e-06 | 1.3e-05 |
| bfp22 | 4.8e-07 | 4.6e-07 | 4.7e-07 | — | — | — |
| bfp24 | 1.2e-07 | 1.2e-07 | 1.2e-07 | 2.0e-07 | 2.0e-07 | 8.2e-07 |

- Each grid buffer adds **1.95–2.0 · 2^−b**, the same for the three buffers (x → y is still in the image domain along
  z and y, y → z along z only, the grid in full k-space: the channel-blocked dynamic range does not change with the
  domain, unlike blocks along a spatial axis, where k-space blocks near the centre span decades).
- The chained errors equal the quadrature sums to 3–4 digits: the buffers' errors are independent and add in
  quadrature, so the per-buffer table is all one needs to budget.
- Blocks along the channels are slightly better than the user's blocks along x of one channel (storage after the pad
  and after each of the three passes: bfp24 3.9e-7, bfp20 6.3e-6, bfp16 1.0e-4; channel blocks for the same four
  stores: √4 · 2 · 2^−b = 2.4e-7, 3.8e-6, 6.1e-5).
- The grid buffers' constant (2.0) is higher than the image-side buffers' (1.1–1.5) because a 16-channel block mixes
  coils of different sensitivity and segments at one cell, so the block max exceeds the typical value more.
- Against the noise: bfp16 adds an rms error of 2e-4 σ_k (a noise-variance increase of 4e-8), bfp24 8e-7 σ_k. Noise is
  never the binding constraint; the NUFFT's own error is (§4).

### 3.5 Adjoint

Two residuals: (a) data-like, the forward grid itself (r ≈ A x: early iterations, the residual is coherent with the
image); (b) noise-like, white complex Gaussian noise on the k-space ball |k| ≤ image Nyquist (a converged
iteration). Buffers: the scatter output (the inverse z pass's input), z → y, y → x; the inverse x pass feeds the crop
with exact maps. The chained run rounds all three and compares the crop output (the gradient's image) with the exact
one.

(a) data-like residual:

| format | scatter output | z → y | y → x | chained, crop output | quadrature of the three |
|---|---|---|---|---|---|
| f32 | 2.5e-08 | 2.5e-08 | 2.5e-08 | — | — |
| fp24 | 6.5e-06 | 6.5e-06 | 6.5e-06 | — | — |
| bf16 | 1.7e-03 | 1.7e-03 | 1.7e-03 | — | — |
| bfp8 | 7.7e-03 | 7.6e-03 | 7.9e-03 | — | — |
| bfp10 | 1.9e-03 | 1.9e-03 | 2.0e-03 | — | — |
| bfp12 | 4.8e-04 | 4.8e-04 | 4.9e-04 | — | — |
| bfp14 | 1.2e-04 | 1.2e-04 | 1.2e-04 | — | — |
| bfp16 | 3.0e-05 | 3.0e-05 | 3.1e-05 | 4.3e-06 | 5.2e-05 |
| bfp18 | 7.5e-06 | 7.4e-06 | 7.7e-06 | — | — |
| bfp20 | 1.9e-06 | 1.9e-06 | 1.9e-06 | 2.7e-07 | 3.3e-06 |
| bfp22 | 4.7e-07 | 4.6e-07 | 4.8e-07 | — | — |
| bfp24 | 1.2e-07 | 1.2e-07 | 1.2e-07 | 1.7e-08 | 2.0e-07 |

(b) noise-like residual:

| format | scatter output | z → y | y → x | chained, crop output | quadrature of the three |
|---|---|---|---|---|---|
| f32 | 2.5e-08 | 2.5e-08 | 2.5e-08 | — | — |
| fp24 | 6.5e-06 | 6.5e-06 | 6.5e-06 | — | — |
| bf16 | 1.7e-03 | 1.7e-03 | 1.7e-03 | — | — |
| bfp8 | 8.3e-03 | 8.0e-03 | 8.1e-03 | — | — |
| bfp10 | 2.1e-03 | 2.0e-03 | 2.0e-03 | — | — |
| bfp12 | 5.2e-04 | 5.0e-04 | 5.0e-04 | — | — |
| bfp14 | 1.3e-04 | 1.3e-04 | 1.3e-04 | — | — |
| bfp16 | 3.2e-05 | 3.1e-05 | 3.1e-05 | 5.5e-05 | 5.5e-05 |
| bfp18 | 8.1e-06 | 7.8e-06 | 7.9e-06 | — | — |
| bfp20 | 2.0e-06 | 2.0e-06 | 2.0e-06 | 3.4e-06 | 3.4e-06 |
| bfp22 | 5.0e-07 | 4.9e-07 | 4.9e-07 | — | — |
| bfp24 | 1.3e-07 | 1.2e-07 | 1.2e-07 | 2.1e-07 | 2.1e-07 |

- Per buffer the grid-domain error is again 1.95–2.0 · 2^−b, the same as in the forward.
- **What reaches the gradient depends on the residual.** With a data-like residual the crop output has 1/12 of the
  grid-domain error (bfp16: 4.3e-6 against 5.2e-5). The crop sums the 50 channels with the weights conj(S_c P_l), so the
  signal adds coherently and the independent rounding errors do not; half of the error energy also falls outside the
  image block, which the crop drops. With a noise-like residual the signal is incoherent too, and the crop output's
  error equals the grid-domain quadrature (5.5e-5 at bfp16).
- So near convergence the adjoint's BFP error is the full ≈ √k · 2 · 2^−b for k stored buffers, relative to A^H n.

### 3.6 k-space data

256 readouts × 3990 samples from the middle of the scan (`data_clean`, noise-free sim) plus complex Gaussian noise
σ = 40.0 per real component (`meta.json` noise_sigma), coil-compressed 32 → 10 with the same signal-region basis. The
data's rms is 85 σ, the k-space peak 3800 σ. Error per real component over σ:

| format | blocks along samples (one coil): rms / σ | max / σ | blocks along coils (one sample): rms / σ | max / σ |
|---|---|---|---|---|
| f32 | 1.5e-06 | nan | 1.5e-06 | nan |
| bfp8 | 3.5e-01 | 3.5e+01 | 4.8e-01 | 3.6e+01 |
| bfp10 | 8.7e-02 | 8.6e+00 | 1.2e-01 | 8.8e+00 |
| bfp12 | 2.2e-02 | 2.2e+00 | 3.0e-02 | 2.2e+00 |
| bfp14 | 5.5e-03 | 5.5e-01 | 7.6e-03 | 5.6e-01 |
| bfp16 | 1.4e-03 | 1.4e-01 | 1.9e-03 | 1.4e-01 |
| bfp18 | 3.4e-04 | 3.3e-02 | 4.7e-04 | 3.5e-02 |
| bfp20 | 8.5e-05 | 8.5e-03 | 1.2e-04 | 8.8e-03 |
| bfp22 | 2.1e-05 | 2.0e-03 | 2.9e-05 | 2.1e-03 |
| bfp24 | 5.3e-06 | 5.1e-04 | 7.4e-06 | 5.3e-04 |

- The device layout of k-space and residual is coil-major (nc × m), so blocks run along the samples of one coil, and
  that is also the better blocking (1.4 × lower error than blocks across coils at one sample).
- The rms error is far below the noise at any b ≥ 12 (bfp12: 0.022 σ, a noise-variance increase of 5e-4). But the
  error is not uniform: blocks at the k-space centre (values up to 3800 σ) get a step of up to 4 · peak · 2^−b. The
  noise-tied criterion is the worst block's step: step_max ≤ q σ needs **b ≥ log2(peak / σ) + 2 + log2(1 / q)**. For
  q = 1/4 (the KStore quantiser's default step, rms 0.072 σ) and peak 3800 σ: b ≥ 15.9, so bfp16 (max error 0.14 σ,
  rms 1.4e-3 σ: the KStore quantiser's worst case with 50 × lower rms). bfp12 would raise the noise in the centre
  blocks by tens of percent.
- The rule is per dataset (peak / σ grows with SNR and coil compression); the encoder can compute b from the noise
  scan and the data peak before streaming. On the host and on disk the KStore's zstd + uniform quantiser stays better
  (≈ 4–6 bits per component, entropy-coded); BFP is for the device copy and the PCIe transfer, where decode must be
  free.

### 3.7 Solver state

2-D toy on the real phantom (`state_toy.py`, `state_toy.log`): slice z = 57 (144 × 254), 10 virtual coils,
variable-density Cartesian undersampling R 4.4, noise at image SNR 20, min ½‖M F S x − y‖² + λ/2 ‖x‖², solved by
Barzilai–Borwein gradient steps (the recon's quasi-Newton family) for 300 iterations; reference x* by f64 CG to 1e-13.
Error ‖x_k − x*‖ / ‖x*‖ at iterations 30 / 100 / 300 (f64 run: 1.9e-3 / 1.3e-10 / 1.0e-13):

| state handling | b = 12 | b = 16 | b = 20 | b = 24 |
|---|---|---|---|---|
| f32 everything | 2.0e-7 (any b) | | | |
| x stored as BFP between iterations (no master copy) | 8.1e-4 | 4.9e-5 | 2.7e-6 | 1.6e-7 |
| f32 master x, operator reads a BFP copy of x | 5.9e-4 | 3.6e-5 | 2.0e-6 | 1.7e-7 |
| f32 master x, gradient returned as BFP | 8.1e-8 | 8.0e-8 | 8.9e-8 | 8.6e-8 |
| f32 master, BB secant pair (s, y) as BFP | 8.1e-8 | 8.2e-8 | 8.2e-8 | — |
| all three BFP (master x f32) | 6.3e-4 | 3.5e-5 | 1.5e-6 | 1.6e-7 |

(values at iteration 300; they equal the iteration-100 values within 3 %: no drift.)

- **Quantised iterates do not accumulate error**: x_{k+1} = Q(x_k − α g_k) re-rounds the current value, it does not add
  roundings up. The iterate stalls at ≈ 3 · 2^−b of x* (updates below half a step are lost) and stays there.
- **A BFP copy of x read by the operator** biases the fixed point by ≈ 2.4 · 2^−b (the operator sees Q(x)), also flat.
  The f32 master does not remove this bias; it removes the stall, which matters when steps are small relative to a
  block's step for a long time (slow convergence, dark voxels next to bright ones in a block).
- **A BFP gradient is free**: Q(g) has error 2^−b relative to g, which vanishes as g → 0, so the fixed point is exact
  even at 12 bits. The same holds for the secant pair (it only sets step lengths). This is the useful result for host
  RAM: g, s and y can be stored at 12–16 bits. **This holds only where g → 0 (no prior, RED).** With P²nP (a denoiser
  after the step), g does not vanish at the fixed point and s is a difference of non-vanishing iterates. A bfp16
  state then biases the fixed point by ≈ 2 · 2^−16, so P²nP needs bfp24 (measured in §9.4).
- Policy: f32 master x (host), BFP for the operator's working copy of x and for g, s, y; b for the working copy from
  the error budget (§6). Caveat: the toy converges in ~100 iterations and has no prior; ZMSHR1's rank-one direction uses
  the pair like BB does (inner products), which the toy does not test separately.

## 4. End-to-end gradient error

The gradient is g = A^H (A x − y). BFP changes it by δg = (Â^H − A^H) r + Â^H (Â − A) x, and how large that is
relative to g depends on the residual r:

- **Early iterations** (r coherent with the image): measured directly as the Gram application A^H A x with every buffer
  rounded, against the exact Σ_la |S_c P_l|² x (fully sampled grid; `acc.py e2e`, `acc_e2e.log`): all buffers
  (x, S, P, x → y, y → z, grid, scatter, z → y, y → x) at b = 16 / 20 / 24 give 2.5e-5 / 1.6e-6 / 9.8e-8; all f32
  4.0e-8. Here the crop's coherent channel sum suppresses the grid buffers' errors (§3.5), and the image-side buffers,
  which act coherently, dominate. With the maps held at bfp20 and the rest at b (configuration "proposed" in the
  log), the error floors at 1.0e-6 from b = 22 on: **the coil maps must follow b** at the final level.
- **Near convergence** (r ≈ noise n): the adjoint part is relative to A^H n with no coherence gain (§3.5), and the forward
  error enters as extra "noise" of rms ε_f · rms(A x), i.e. amplified by rms(A x) / σ_k = 4.0 at image SNR 20 (this
  amplification applies to every forward error, the NUFFT's included). So ‖δg‖ / ‖A^H n‖ ≈ √(ε_a² + (4.0 ε_f)²).

For the configuration of §6 (image working copy, coil maps, y → z, grid and z → y at b; diagonals on the fly; x → y,
y → x and the scatter output f32), from the per-buffer tables (they add in quadrature, §3.4–3.5):

| b | forward ε_f (data samples) | gradient, early iteration (A^H A x, measured) | adjoint ε_a, noise residual | gradient at convergence, / ‖A^H n‖ | ε_f / NUFFT forward error (1e-5) |
|---|---|---|---|---|---|
| 16 | 4.9e-05 | 2.5e-05 (all buffers at b) | 3.3e-05 | 2.0e-04 | 4.93 |
| 18 | 1.2e-05 | — (all buffers at b) | 8.3e-06 | 5.0e-05 | 1.23 |
| 20 | 3.1e-06 | 1.6e-06 (all buffers at b) | 2.1e-06 | 1.3e-05 | 0.31 |
| 22 | 7.8e-07 | — (all buffers at b) | 5.3e-07 | 3.2e-06 | 0.08 |
| 24 | 2.1e-07 | 9.8e-08 (all buffers at b) | 1.6e-07 | 8.6e-07 | 0.02 |

(The early-iteration column is the measured all-buffers-at-b run, an upper bound for the §6 configuration, which keeps
three buffers f32.)

- **Against the NUFFT.** The NUFFT's forward error at W 10 is ≈ 1e-5 (arch_targets.md item 6), and its measured
  normal-operator error for full-band images ≈ 2e-4 (pipeline_target.md row 11). The same amplification applies to it,
  so the ratio ε_f / 1e-5 in the last column is also the ratio of the converged-gradient errors. bfp22: 1/13 of the
  NUFFT error, bfp24 1/48, bfp20 1/3, bfp16 5 × the NUFFT's forward error (but 1/8 of its normal-operator error).
- **Against the noise.** The forward error as data noise is 4.0 ε_f σ_k: bfp16 2e-4 σ_k, bfp22 3e-6 σ_k. The
  fixed point moves by that fraction of the noise-induced image error: invisible at every b ≥ 16.
- **Against the solver state** (§3.7): the operator's working copy of x adds a fixed-point bias of ≈ 2.4 · 2^−b
  (bfp22: 5.7e-7), the same order as ε_f; the gradient and secant pairs add nothing measurable at any b ≥ 12.

## 5. Conversion points

Rule: convert only inside the producer's store and the consumer's load. With c = (4b + 1) / 16 bytes per complex, a
buffer of n complex values between a producer and a consumer moves 16n bytes as f32 and 2cn with in-kernel
conversion; with a standalone convert pass it moves 8n (producer writes f32) + 8n + cn (convert) + cn (consumer) =
16n + 2cn, more than f32 at every b. So on a DRAM-bound buffer a separate pass never pays. The
exception is a host link (PCIe ≈ 25 GB/s against ≈ 900 GB/s DRAM): a device pack kernel before a device → host copy
costs ≈ 1/40 of the PCIe time it saves.

Every producer / consumer pair of one gradient (knot-major engine, plain single-image path, fused pad / crop, L2
slabs) and of the host streaming, with the kernel that would carry the codec. "DRAM" means the buffer goes through
DRAM in today's default path; the per-load / per-flush bytes are those of temporal_operator.md §23.2 (2 mm, 50
channels).

| # | buffer | producer (store) | consumer (load) | where it lives | codec in | proposal |
|---|---|---|---|---|---|---|
| 1 | image x, operator input (per knot) | host solver update (CPU) | x pass with the pad hook: prologue staging of the x row | host state → device slot (Stager) | host codec in the update loop (the update writes it anyway); device: pad prologue decodes the row's covering blocks | BFP working copy; f32 master on the host |
| 2 | coil maps S_c | setup (host) | pad prologue staging; crop epilogue staging | device, whole run | host codec at setup; decode in both staging loops (the same decoded values, so pad and crop stay an exact adjoint pair) | BFP |
| 3 | segment diagonals P_l (per encode, per frame) | setup `build_segments_*` (host) | pad prologue, crop epilogue | device | node bases: no stored diagonals, the staging loop computes them from 32-bit turns + R2*; SVD bases: host codec at setup, decode in staging | on the fly / BFP |
| 4 | x → y intermediate (forward) | x pass (fused pad) | y pass, first stage | L2 slab (2 × 2.90 GB per load saved by §23.5) | — | f32 (no DRAM bytes to save; the x pass's 16-channel tiles align with blocks only if nch % 16 = 0) |
| 5 | y → z intermediate | y pass, last stage | z pass, first stage | DRAM: 3.72 GB written, 3.15 GB read per load | new store epilogue of the y pass (per-thread runs from `fin`); prologue decode into shared memory in the z pass (half-warp decode in the first stage for in-place passes) | BFP |
| 6 | grid (forward result) | z pass, last stage | gather: vs / vt lane-group channel loads; tile kernels' staging | DRAM slot: 2.98 GB written, read by the gather per knot | z pass store epilogue; gather: `bfp_load_c` per lane (any nch) or `bfp_hw_load_c` (nch % 16 = 0) | BFP |
| 7 | k-space data y | KStore (host: zstd + noise-relative quantiser, decoded per set) | forward residual init (r = −y) | host tier K / P → device per set | host codec in the Stager thread right after the KStore decode; device: the gather's accumulator starts from decode(y) instead of a separate r = −y copy | BFP tied to σ (§3.6) |
| 8 | residual r (nc × m) | forward gather (one owner thread per (sample, coil), y += o) | adjoint scatter (ra / kl / tile) staging of samples | DRAM | — (no lane group owns 16 consecutive samples of a coil; the sample kernels are latency-bound, so fewer bytes would buy little) | f32 |
| 9 | scatter output (adjoint grid) | ra / kl / tile: owner tiles store, split tiles add atomically on `seg_zero_split` cells | inverse z pass, first stage | DRAM slot: 2.98 GB read per flush | — (atomics need native f32) | f32; later option: owner tiles BFP + a small f32 split overlay |
| 10 | z → y intermediate | inverse z pass, last stage | inverse y pass, first stage | DRAM: 3.15 GB written, 3.72 GB read per flush | inverse z store epilogue (per-thread runs); prologue decode in the inverse y pass | BFP |
| 11 | y → x intermediate | inverse y pass | inverse x pass (fused crop) | L2 slab | — | f32 |
| 12 | image gradient g (device) | crop epilogue, f32 atomic adds per voxel and channel tile | device → host download, solver update | device → host | device pack kernel before the D2H copy (the PCIe exception); host decode in the update loop | BFP |
| 13 | host state: x master; g, s = x − x_prev, y = g − g_prev | host solver | host solver | host RAM (4 × 17.8 GiB at 1 mm) | host codec | x f32; g, s, y BFP (§3.7) |
| 14 | sample tables | setup | gathers / scatters | packed tier P → slots | — (integer coordinates and weights; already zstd-packed) | unchanged |

Totals for one grid load + flush at 2 mm, 50 channels (today 10.6 + 10.7 = 21.3 GB): BFP on rows 5, 6, 10 only, so
forward 0.73 + 9.85 f and inverse 3.83 + 6.87 f GB with f = (4b + 1) / 128 (BFP bytes per complex over 8):

| storage of rows 5, 6, 10 | load + flush DRAM bytes | saved | 4090 time per pair (0.66 × 1008 GB/s) | 2-D velocity gradient (192 pairs) |
|---|---|---|---|---|
| f32 (today) | 21.3 GB | — | 32.0 ms | 6.15 s FFT side |
| bfp24 | 17.2 GB | −19 % | 25.9 ms | −1.2 s |
| bfp22 | 16.2 GB | −24 % | 24.3 ms | −1.5 s |
| bfp20 | 15.1 GB | −29 % | 22.7 ms | −1.8 s |
| bfp16 | 13.0 GB | −39 % | 19.6 ms | −2.4 s |

The gather's read of the grid (row 6) falls by the same factor; the gather kernels are latency-bound, so the time it
buys there is to be measured, but the grid slot also shrinks in VRAM by the same factor (a larger coil batch at equal
VRAM: arch_targets.md §2.3) and a plane of all channels fits the L2 slab budget in more cases (arch_targets.md §2.2).

## 6. Recommended b per buffer

The binding constraint is the NUFFT's own numerical error, not the noise (§4): storage error at the final pyramid
level ≤ 1/10 of the NUFFT's forward error (1e-5) gives b = 22. Earlier pyramid levels follow the error budget of
pipeline_target.md §2 (loose, relative to that level's dominant model error), so b is a per-level knob.

| buffer | final level | earlier levels | measured error at the final-level b | bytes vs f32 |
|---|---|---|---|---|
| y → z, z → y FFT intermediates | bfp22 | bfp20, coarse bfp16 | 4.6e-7 per buffer | 0.70 |
| grid read by the gather | bfp22 | bfp20, coarse bfp16 | 4.7e-7 | 0.70 |
| x → y, y → x (L2 slabs) | f32 | f32 | — (no DRAM bytes) | 1 |
| scatter output | f32 | f32 | — (atomics) | 1 |
| residual r | f32 | f32 | — (sample kernels, latency-bound) | 1 |
| image: the operator's working copy | bfp22, f32 master on the host | bfp16–20 | 2.8e-7 on the pad output; fixed-point bias 5.7e-7 | 0.70 (copy) |
| coil maps | bfp22 (one decoded copy for pad and crop) | bfp16 | 2.9e-7 pad, 1.8e-7 crop | 0.70 |
| segment diagonals, node bases (fixed, cheb) | on the fly from 32-bit turns + R2* | same | 8.3e-8 | 12 B per voxel for all L instead of 8 L (0.15 at L 10) |
| segment diagonals, SVD bases | bfp22 | bfp16 | 3.7e-7 | 0.70 |
| k-space data, device copy and PCIe | b ≥ log2(peak / σ) + 4 → bfp16 on ph0001 | same | worst block step ≤ σ / 4; rms 1.4e-3 σ | 0.51 |
| image gradient download, host g | bfp16 for no prior / RED; bfp24 for P²nP (§9.4) | same | fixed point exact without a post-step prior; ≈ 2 · 2^−b bias with one | 0.51 / 0.76 |
| host secant pair s = x − x_prev, y = g − g_prev | bfp16 (difference record) for no prior / RED; bfp24 (x_prev record) for P²nP | same | §9.4 | 0.51 / 0.76 |
| host master x | f32 | f32 | — | 1 |

bfp24 is the conservative alternative where an extra 6 % of bytes is acceptable (1/48 of the NUFFT error); bfp20 is
within a factor 3 of the NUFFT error and suits all but the last level.

## 7. Integration plan (next phase)

Order by bytes saved per unit of work; every step behind a switch whose default stays f32 until its test passes, the
old path kept selectable.

1. **FFT intermediates (rows 5, 10), hasty-fft codegen.**
   - `AxisCfg` gets an input and an output storage, `Store::{F32, Bfp(b)}`, for strided passes whose 16-line tiles
     start at multiples of 16 in the flat line index (the y and z passes; the x pass only when nch % 16 = 0). Kernel
     parameters: mantissa and exponent pointers instead of `in` / `out`.
   - Store (§2, instruction cost): the last stage writes to shared memory as on the epilogue path
     (`fin[e · LPB + k]`: one entry of the 16-line tile = 32 contiguous reals = one block), then each thread packs
     runs of 8–16 values with `bfp_run_store` (exponent by a group max over the run's lanes) and writes whole words +
     one exponent byte per block. A built-in store epilogue keeps `IoHook` free for callers.
   - Load: a prologue decodes the tile's blocks with `bfp_run_load` into the second shared buffer (free until stage 1
     when `FFT_INPLACE` = 0), and the first stage reads shared memory. Fallback for in-place passes:
     `bfp_hw_load_c<B>(mant, ex[blk], blk, valid)` in the first stage (one word per lane, three shuffles).
   - Pre-registered check with ncu: the y and z passes' issue utilisation stays ≤ 50 % and DRAM throughput ≥ 85 %.
   - `PrunedChanFft3::set_store(Store)` for the y → z (forward) and z → y (inverse) buffers; the slab scheduling is
     unchanged (x → y and y → x stay f32 in L2).
   - k-support pruning: the pruned y / z passes start their line groups at kx_lo · nch (and per-row column ranges),
     which are not multiples of 16. With BFP every range must be rounded out to whole 16-line blocks, and the extra
     lines must read defined data (the producing pass writes the rounded range, or the rounded lines read zeros): one
     block is written by one tile, and a block holding garbage that happens to be Inf / NaN would poison its valid
     neighbours (non-finite blocks decode as NaN). Cost: ≤ 15 extra lines per range.
   - Switch `HASTY_SEG_FFT_STORE = f32 | bfp24 | bfp22 | bfp20 | bfp16` (name of arch_targets.md item 6).
   - Tests: forward and inverse against the f32 path at every b with the bound 2.2 · 2^−b per stored buffer (§3.4
     measured 1.95–2.0); slabs = whole grid bit for bit with BFP on; adjointness of the pair; `chanfft` and
     `seg_fft_traffic` suites with the switch on; opbench FFT column and gradient (pre-registered decision: keep if the
     knot gradient falls by ≥ 0.6 × the byte model's saving).
2. **Grid (row 6).** The z pass's output storage plus the gather's decode: vs / vt read their 16-channel lane groups
   with `bfp_load_c` (any nch; 1–3 word loads per complex, L1-resident) or `bfp_hw_load_c` when nch % 16 = 0; the tile
   kernels decode in their staging loop. Switch `HASTY_SEG_GRID_STORE` (same values). Test: forward within the
   predicted error against the f32 grid, at 1.33 M and 13.3 M samples; opbench forward time, since the gather is
   latency-bound (the extra integer work may or may not hide).
3. **[WITHDRAWN: segment diagonals are SVD-basis, stored; only `HASTY_SEG_DIAG = bfpN`-style BFP storage applies]**
   ~~Segment diagonals on the fly (row 3).~~ The pad prologue / crop epilogue staging computes P_l from per-voxel
   (turn0, dturn) as u32 and R2* (12 B per voxel for all L), `__sinf` / `__cosf` of the reduced turn or `sincospif`;
   per-frame ratemaps (`segments_frames`) store one such triple per frame. VRAM: E × L × n_v × 8 B → E × 12 × n_v B
   (1 mm box, 7 encodes, L 11: 5.25 GiB → 0.72 GiB). Switch `HASTY_SEG_DIAG = stored | turns`. SVD-basis diagonals:
   `HASTY_SEG_DIAG = bfpN`. Test: pad / crop equal to the stored path within 1e-7.
4. **Coil maps (row 2).** Host codec at setup, decode in the staging loops. Switch `HASTY_SEG_MAPS_STORE = f32 |
   bfpN`. Test: adjoint identity unchanged (the same decoded maps on both sides).
5. **Host side (rows 1, 7, 12, 13).**
   - Solver state: master x f32; g, s, y as `hasty_bfp::BfpBuf` (§3.7); the update loop encodes the operator's working
     copy of x as it writes x. Switch `HASTY_STATE_STORE = f32 | bfp`. Host RAM at 1 mm per frame: 4 × 17.8 GiB =
     71 GiB → 17.8 + 3 × 9.0 = 45 GiB with bfp16 stacks.
   - Gradient download: a device pack kernel before the D2H copy, decode on the host in the update (bfp16: 17.8 →
     9.0 GiB per iteration over PCIe).
   - k-space (kload / KStore): a device tier `bfpN` for the k-space data on the device (host BFP encode in the Stager
     right after the KStore decode; the residual initialisation decodes it). VRAM and PCIe fall by the factor of §3.6;
     the host and disk tiers keep zstd + the noise-relative quantiser, which compresses far better there.
   - Ledger (memory_ledger.md): size formulas take `Bfp::bytes`; the slot, maps, diagonals and state entries get a
     storage field.
6. **Later, only if measured worth it:** owner-tile BFP for the scatter output with an f32 overlay for split tiles
   (row 9); the x pass BFP for nch % 16 = 0 when slabs are off (1 mm, many channels, small L2).

## 8. Tests and how to rerun

| test | what | result |
|---|---|---|
| `cargo test -p hasty-bfp --test cpu` | bit layout against the spec (hand-built 12-bit block), per-value bound 2^(e−b) and exponent minimality for every block at every b in 8 … 24 (128 k mixed-scale blocks per b), zeros and tails, −0.0, subnormal blocks (lower exponent limit), all-equal blocks, powers of two exact, the rounding bump at max → 2^e, ties to even, max-exponent blocks finite with error < one step, Inf / NaN blocks decode as NaN (NaN not dropped by the max), complex round trip and byte count | 9 / 9 pass |
| `cargo test -p hasty-bfp --test gpu` | GPU = CPU **bit for bit** for every b in 8 … 24: pack in the warp, half-warp complex and per-thread layouts (mantissa words and exponent bytes), unpack in all three plus the random-access real and complex decoders, on 1300 blocks with every edge case and a tail, at 3 CUDA blocks (many grid-stride iterations) and with half warps diverging at the tail; GPU pack → unpack = CPU round trip on 1 M reals at the default grid | 2 / 2 pass (`gpu_test.log`) |
| `compute-sanitizer --tool memcheck --leak-check full` on the GPU test | | 0 errors, 0 leaks (`memcheck.log`; the first run found an out-of-bounds store of the half-warp layout for b < 16, fixed) |
| `bfp_bench roundtrip` against `acc.py`'s emulation | 4.1 M reals, b = 8 … 24 | identical values (`pycheck.log`) |

Rerun (GPU parts under the GPU lock):

```
cd rustlib && cargo test --release -j 4 -p hasty-bfp            # cpu + gpu tests
compute-sanitizer --tool memcheck --leak-check full target/release/deps/gpu-<hash> --test-threads=1
target/release/bfp_bench bench 27 21                               # DRAM-sized microbenchmark
target/release/bfp_bench bench 22 201                              # L2-resident (instruction cost)
cd research/experiments/bfp && ../../../rustlib/.venv/bin/python3 acc.py image|forward|adjoint|kspace
../../../rustlib/.venv/bin/python3 state_toy.py; python3 tables.py <image|P|forward|adjoint|kspace|bench>
```

`acc.py forward` and `adjoint` hold one 50-channel f64 grid (6.6 GB) plus one f32 or f64 copy: run them under a 16 GB
memory cap; each takes 9–30 minutes at 2 workers. Nothing large is written to disk (json and logs only).

### 8.1 Step 1 implemented: results (2026-11-20)

Step 1 of §7 is done: the y → z (forward) and z → y (inverse) intermediates of the grid FFT can be stored as BFP. The
x ↔ y intermediates stay f32 in the L2 slabs. The switch is off by default. With it off, the generated kernel source is
identical to the old code (1152 non-BFP configurations compared), and the engine's loaded grid slot is bit-identical
before and after a BFP round trip. Timing has **not** been measured yet: it is left to the combined GPU campaign
(handoff: `research/experiments/bfp_fft/HANDOFF.md`).

**Switches**

| switch | values (default first) | what it does |
|---|---|---|
| `HASTY_SEG_FFT_STORE` | `f32` · `bfp16` … `bfp24` | storage of the y ↔ z intermediates of every grid FFT of the engine (tests: `set_thread_fft_store`; per engine: `SegNufft::set_fft_store`) |
| `HASTY_SEG_FFT_BFP_LAYOUT` | `inplace` · `dense` | in place in the grid slot (needs n0 · nch to be a multiple of 16, else dense), or a separate buffer |

**Design (hasty-fft).**
- The BFP pass layout is per-thread runs through shared memory, as §2 recommends.
  - The writing pass's last stage writes to shared memory. An epilogue packs whole 16-line blocks: runs of
    max(32 / gcd(b, 32), 8) values, the exponent taken as a group max over the block's runs.
  - The reading pass decodes the tile's blocks into the free shared buffer before stage 1.
  - The shared-memory row stride is padded by one complex (at most 2-way bank conflicts). The padding is dropped
    where it does not fit (e.g. a y length of 384 at 99 KB per block).
- A block is the 16 complex values of one 16-line tile at one entry. Tiles are aligned to 16 in the flat line index
  of a plane, lines (x, channel). The last tile of a row may be partial: its missing lines are zeros.
- **K-support pruning.** The BFP passes round each row's line range out to whole tiles; lines outside the range store
  zeros.
  - With per-line entry ranges, a block is written when any line of its tile writes that entry.
  - Blocks the writer skips are marked by exponent byte 0 (the codec never produces it); they decode as zeros.
  - A k-support plan clears the exponent bytes once per call (about 1 % of the buffer).
- **Short coil batches.** The writers take a runtime channel mask. A short last batch leaves stale data in the unused
  channels, and a block of 16 lines mixes channels, so those lines must store zeros. Without the mask a stale channel
  could set a block's exponent: `seg_fft_traffic` under bfp22 showed this, fused vs separate pad differing by 7.4e-4.
- Files: `rustlib/crates/hasty-fft/src/codegen.rs` (`Store`, `BfpIo`, `AxisCfg::bfp_in` / `bfp_out`),
  `plan.rs` (`AxisPlan::from_spec_bfp`, `LaunchPtrs::bin` / `bout` / `bch`), `chanfft.rs`
  (`PrunedChanFft3::set_store`, `set_store_opts`, `set_active_channels`, `store_bytes_for`). Engine wiring:
  `hasty-nufft/src/segnufft/fused.rs` plus dispatch lines in `segnufft.rs`.

**In-place layout and VRAM.**
- The first version used a separate buffer: planes of the image block × n1 rows × the plane's tiles × (4b + 1) bytes.
  At the 2 mm thorax grid (288 × 288 × 140, 50 channels, image block 112 planes) that is 2.58 GB at bfp22, 0.56 of the
  4.6 GB slot, and it added VRAM.
- The default now keeps the mantissas inside the grid slot. Block (z, y, tile) uses the first 4 · b bytes of its own
  16 lines' 128 bytes at (z, y).
  - Every y or z pass reads and writes only its own tile's cells, so the transform stays in place as in f32.
  - Only the exponent bytes are extra: one per 16 complex values, 29 MB at that grid.
- Cells outside the k-support, which the forward leaves undefined anyway, then hold mantissa bits rather than stale
  floats. The gather never reads them.
- **Budget.** The engine's coil batch now counts the BFP bytes in `max_grid_bytes`
  (`fused::fit_store_budget`); with f32 the batch is unchanged. Not counted yet: the extra multi-image transforms of
  `multi_plan`, and a store switched after construction (`set_fft_store` does not re-budget).

**Byte model, corrected.** The 21.3 → 16.2 GB of §5 for bfp22 includes the grid itself (row 6), which is step 2. Step 1
covers rows 5 and 10 only: 7.54 + 13.74 f GB per grid load + flush. In place, the mantissas of b = 17 … 24 cover 3 of the
4 DRAM sectors of their 128-byte line (f = 97 / 128 with the exponent byte); bfp16 covers 2 of 4.

| storage of rows 5 and 10 | f | load + flush | time per pair (0.66 × 1008 GB/s) | 2-D-knot gradient (192 pairs) | keep threshold (0.6 × model saving) |
|---|---|---|---|---|---|
| f32 | 1 | 21.3 GB | 32.0 ms | — | — |
| bfp20 / 22 / 24, in place | 0.758 | 17.95 GB | 27.0 ms | −0.97 s | 0.58 s |
| bfp22, dense buffer | 0.695 | 17.1 GB | 25.7 ms | −1.21 s | 0.73 s |
| bfp20, dense buffer | 0.633 | 16.2 GB | 24.4 ms | −1.47 s | 0.88 s |
| bfp16, in place | 0.508 | 14.5 GB | 21.8 ms | −1.96 s | 1.18 s |

So in place, bfp24 costs the same bytes as bfp22, and its 96-byte mantissa run writes 3 whole sectors (bfp22's 88
bytes end in a partial sector; whether that partial sector costs a read-modify-write is for ncu to show).

**Accuracy on real data.**
- Tool: `bfp_phantom` (new, `rustlib/crates/hasty-bench/src/bin/bfp_phantom.rs`). Inputs:
  - ph0001 `sim_flow_10min/scan`: image 254 × 144 × 114, fine grid 320 × 180 × 144;
  - the truth image (water + fat at a fixed phase, smooth background phase, as `acc.py`);
  - the true maps compressed 32 → 10 virtual coils (0.926 of the signal-region energy);
  - L = 10 segments of the field map over the readout window;
  - the real coordinates and readout times of encode 3, and at order 3 the real concomitant tables;
  - coil batch 5, so 50 channels.
- Cases: static, one (frame, encode) set of 1.51 M samples; and 2-D knots 3:1,8:3 (24 knot images, 340 readouts of the
  encode over the scan, 1.36 M samples).
- One engine; the store is switched on it.
- Every error is relative to f32. Next to it is the operator's own one-ulp floor: how much the same quantity moves in
  f32 when x (or n) is moved by one ulp with random signs.
- Near convergence: y = A x + n with rms(A x) / rms(n) = 4.

| case | stored buffer (loaded slot) | forward A x (one-ulp floor) | early gradient Aᴴ A x (floor) | adjoint of the noise Aᴴ n (floor) | converged gradient Aᴴ(A x − y), bfp22 / bfp24 (floor) |
|---|---|---|---|---|---|
| static, order 0 | 1.94 · 2^−22 | 4.45e-7 (3.5e-7) | 1.06e-5 (4.1e-6) | 8.3e-6 (7.0e-6) | 8.4e-5 / 9.5e-5 (5.9e-5) |
| static, order 3 | 1.93 · 2^−22 | 5.15e-7 (3.9e-7) | 1.07e-5 (4.6e-6) | 8.3e-6 (7.0e-6) | 1.2e-4 / 6.7e-5 (8.9e-5) |
| 2-D knots, order 0 | 1.94 · 2^−22 | 4.26e-7 (3.0e-7) | 1.04e-5 (3.1e-6) | 9.2e-6 (5.7e-6) | 3.0e-5 / 3.3e-5 (1.8e-5) |
| 2-D knots, order 3 | 1.93 · 2^−22 | 4.52e-7 (3.1e-7) | 1.00e-5 (3.2e-6) | 9.2e-6 (5.7e-6) | 3.3e-5 / 2.1e-5 (2.0e-5) |

- The stored buffer matches the prediction (1.95–2.0 · 2^−b) in every case, and at every width: 1.87 at bfp16 and
  bfp20.
- At the samples, the forward error is 1.8–2.2 · 2^−22 at bfp22. The bfp16 rows (0.76–0.80 · 2^−16) show the part due
  to BFP; at bfp24 the forward sits on the f32 floor (3.2–3.9e-7).
- The early gradient and the noise adjoint carry 35–45 · 2^−b, 17–40 × the doc's prediction. They scale exactly as
  2^−b: noise adjoint 5.2e-4 at bfp16, 3.2e-5 at bfp20, 8.3e-6 at bfp22.
- **Why the adjoint numbers exceed §3.5–§4.** The cause is the engine's conditioning, not a BFP defect.
  - In f32 alone, one ulp on n moves Aᴴ n by 5.7–7.0e-6, about 100 × the input change. The engine's f32 adjoint
    already carries rounding noise of that size relative to a noise-driven gradient.
  - Every grid-level perturbation is amplified the same way: about 17 × into Aᴴ n and 22 × into Aᴴ A x. This holds for
    the BFP rounding and for f32's own.
  - The emulation of §3.5 fed independent noise per channel to an f64 crop. It saw neither this amplification nor
    f32's own floor.
  - Checks that ruled out the other explanations:
    - In place and dense, and k-support pruning on and off, give identical numbers.
    - The segment interpolators are well conditioned: ‖a(τ)‖ = 0.88–1.08.
    - An SVD segment basis gives the same factor (29 · 2^−b for the noise adjoint).
- **Near convergence.**
  - The gradient is dominated by coherent forward errors: |Aᴴ A x| / |Aᴴ n| = 446 (static) and 148 (knots).
  - A one-ulp change of x already moves it by 1.8–8.9e-5.
  - Against that floor, bfp22 adds 1.4–1.7 × and bfp24 0.75–1.8 ×. bfp24 is at the floor in every quantity.
  - bfp20 gives 2.0e-4 / 1.4e-4 / 7.3e-5 / 8.0e-5 for the four cases (3.3 × the floor static, 4 × knots). bfp16 gives
    1–2.7e-3.
- Not measured: the NUFFT's own error on the same quantities (the doc's 1e-5 forward is an f64 figure). The comparison
  "BFP error ≪ NUFFT error" therefore holds for the forward, but for the gradient it rests on the assumption that
  NUFFT errors are amplified the same way.

**Recommendation.** bfp24 in place for the final level:
- the same DRAM bytes as bfp22 in place (the −16 % of the table above);
- whole sectors;
- every measured error at the f32 one-ulp floor.

bfp22 in place has the same cost and is 1.2–3 × the floor. The dense layout buys 4 % more bytes (bfp22) at 0.56 of a
slot of VRAM; it is not recommended at the final level. Earlier pyramid levels: bfp20 in place has the same bytes as
bfp24 in place, so it gains nothing there. Only dense bfp20 (−24 %) or bfp16 in place (−32 %) save more, at 3–4 × and
≈ 40 × the floor.

**Tests.**

| test | what | result |
|---|---|---|
| `hasty-fft --test bfp_store` | 13 layouts × b = 16 / 20 / 22 / 24 / 17 / 23, forward and inverse. Layouts: block-pruned; ball, cube, non-convex and full k-supports; 2–8 channels; partial tiles; in-place y / z passes; z block without tail; y 384 unpadded. Checks: dense buffer = host codec bit for bit; output = f32 passes on the decoded buffer bit for bit; NaN-filled buffer before every call (every block read was written or cleared); slabs 1 and 3 = whole grid; in place = dense bit for bit (5 whole-tile layouts); short coil batch; switch tests. Buffer error 1.71–2.17 · 2^−b | pass |
| `hasty-fft --test chanfft` | the f32 pruned, k-support and slab tests | pass |
| `hasty-nufft --test seg_fft_store` | bfp22 vs f32 on one engine: static and 2-D knots, orders 0 and 3, fused pad / crop on / off, slabs, k-pruning on / off, in place and dense. Checks: stored buffer < 2.6 · 2^−22 (measured 1.86); forward < 2 κ × slot error; adjoint and gradient < 16 × their one-ulp sensitivity plus the atomics spread; adjointness; switch-off bit-identity of the loaded slot | pass |
| `hasty-nufft --test seg_fft_traffic` with f32 and with `HASTY_SEG_FFT_STORE=bfp22` | the FFT-side switches of §23 | pass |
| compute-sanitizer memcheck, 12 runs | bfp_store layouts 0, 1, 7, 10, 11, 12 (dense and in place) at one width each, the short batch, 4 engine cases | 0 errors (peak RSS ≤ 0.2 GB) |
| compute-sanitizer racecheck, 10 runs | the same bfp_store set and engine cases 0, 1 | 0 hazards (peak RSS 1.1–4.4 GB per bfp_store run, 9.1 GB short batch, 22–23 GB per engine case) |
| racecheck engine case 6 (2-D knots order 3), synccheck (all) | | not run: case 6 was killed by the 24 GB cap (handoff note) |

Logs: `research/experiments/bfp_fft/` (`t_*`, `ph_*`, `san/`, `diag/`; earlier rounds in `run1/`, `run2/`).

**Open.**
- The timing campaign: opbench, bfp_fft_bench and ncu, cases in the handoff note.
- Racecheck engine case 6, and synccheck.
- Budget of the multi-image transforms, and re-budget after `set_fft_store`.
- Tile lengths above 396 on 99 KB-per-block cards (e.g. a 480-point y or z axis at 1 mm) fall back to f32: BFP needs
  16-line tiles.

## 9. Step 5 host side: results (2026-11-20)

What was built, each part behind a switch whose default is f32. With the switches off, the old code runs unchanged;
the f32 plan and the f32 transfers are byte for byte the same as before.

### 9.1 Switches

| switch | values | what it does |
|---|---|---|
| `--state-store` (recon_p2np) = `HASTY_STATE_STORE` | `f32` (default) · `bfp` = `bfp16` · `bfpN` · `G:P` (each `f32` or `bfpN`) | host solver state: x is the f32 master; g and g_prev at G bits, the secant record at P bits (`hasty_recon::bfpstate::run_objective_host_bfp`; `hoststack::run_objective_host` dispatches to it) |
| `HASTY_GRAD_PACK` | `1` (default when the store is BFP) · `0` | device pack of each set's gradient before the download; `0`: f32 download plus host encode (the same bytes) |
| `--kstage` (recon_p2np) = `HASTY_KSTAGE` | `f32` (default) · `bfp` (σ rule, step ≤ σ/4) · `bfp+qREL` · `bfpN`, optional `:resident` | k-space staged as BFP and decoded on the device at upload (`kstage::Stager::with_packing`); `:resident` keeps every set packed in VRAM (`make_resident`) |
| `--kstore bfp…` | `bfp` · `bfp+qREL` · `bfpN` (RAM only) | host k-space store in BFP: the Stager uploads its words as they are, with no host decode |

### 9.2 What was built

- **`hasty_bfp` host container.**
  - `BfpBuf` holds b-bit storage per buffer.
  - `view` / `view_mut` give block-aligned f32 views, decoded on demand.
  - `par_views` / `par_views_mut` give the same views as rayon iterators.
  - Also: `zeros`, `encode_from` / `decode_into`, `copy_packed`, `bytes_for`, `Bfp::bits_for_step` (the k-space rule).
- **Faster block loops, same bits.**
  - `round_ties_even` on f32 compiles to a libm call on baseline x86-64. It is replaced by an exact f64 magic-number
    rounding, with the scale hoisted and a streaming 64-bit bit packer.
  - b = 16 has a fast path.
  - The parallel encode and decode run the serial block loop per chunk, so both give the same bits by construction.
- **Transfer kernels, new files** (`hasty-bfp/src/xfer.cu`, `xfer.rs`).
  - Per-thread-run pack and unpack of `rows` rows of n reals, each row its own block sequence. The gradient is one
    row; k-space has one row per coil.
- **Host solver state** (`hasty-recon/src/bfpstate.rs`). The ZMSHR1 / P²nP loop of `run_objective_host`, branch for
  branch (none, circulant, first step, ZMSHR1, ZMSHR1 per set, ZMSHR1-M, step limit, checkpoints):
  - Every pass decodes 64 Ki-complex chunks into per-thread scratch. Set-sized f32 temporaries exist only for the
    metric (three instead of two).
  - The gradient arrives set by set through a `GradSink` (`Objective::gradient_host_sink`, implemented for `P2np`
    streaming and for RED). Objectives without one (the velocity model, the temporal subspace) keep g and g_prev f32
    (`StateStore::effective`).
  - G and G_prev swap after the step; there is no copy.
  - **Secant record as an exact difference.** A BFP x_prev makes s = x − x̂_prev wrong by 2^−b ‖x‖, which near
    convergence exceeds ‖s‖. So the loop stores s = x_new − x_old (an exact f64 difference of the f32 iterates)
    whenever the denoiser leaves x untouched. A fingerprint of all bits of x before and after the denoiser decides
    this every iteration.
    - The first iteration stores x_prev.
    - If a denoiser later changes x, the next step is a first step and x_prev is stored from then on.
  - Checkpoints (`state_<name>.hsb`) store the BFP stacks bit-exact. x is stored bit-exact while the record is
    x_prev, and at the f32 loop's 1e-6 policy otherwise.
  - Resuming across store kinds or preconditioners is refused before anything is read.
- **Streaming.** `SetStreamer::for_each_sink` packs on the copy stream before each download (two packed slots);
  `for_each` is the f32 sink.
- **k-space.** `KStore` gains a `bfp` representation: per set, the width from the σ rule (the set's peak, the store's
  σ). `read_packed` returns the words; `bfp_summary` gives the widths, worst step / σ and predicted rms. The Stager has
  packed staging, a resident mode, and host encode right after a zstd decode for non-BFP stores. σ is the store's
  noise level, i.e. the whitened per-component noise the pipeline records (the same σ the `+q` quantiser uses).
  Never phantom truth.
- **The decode inside the engine's residual kernels is not done.** Decoding y in the `init_rows` / `forward_residual`
  load instead of into an f32 set copy is the next step. Those kernels belong to the engine (segnufft), so the
  device decode happens at upload; the residual sees an ordinary f32 set.
- **Ledger.** `vknot::plan_with(n, store, kstage, …)` sizes the state and the data from `Bfp` bytes (memory_ledger.md
  §8).

### 9.3 Tests

| test | what | result |
|---|---|---|
| `hasty-bfp --test cpu` (9), `--test host_buf` (6) | layout, bounds and edge cases. The parallel encode / decode equals the serial loops bit for bit (sizes around the chunk and block boundaries, tails). Chunked read and write views equal the whole-buffer coding. Zeros, NaN / Inf blocks inside views, write views refused inside a block. The step rule: minimal, step ≤ qσ, never coarser than log2(peak/σ) + 2 + log2(1/q) | pass |
| `hasty-bfp --test xfer_gpu` | device pack and decode = host codec bit for bit: all widths 8 … 24, 1 … 10 rows, odd and even row lengths, tails, NaN, Inf, f32::MAX, 2^21 reals; mantissa and exponent buffers poisoned. Rerun after the 16-bit fast path | 153 cases pass (`xfer_gpu.log`) |
| compute-sanitizer memcheck, leak check, on `xfer_gpu` | | 0 errors, 0 leaks (`xfer_gpu_memcheck.log`) |
| `hasty-recon --test bfp_xfer_gpu` | streamer, 5 sets of 70 001 values in shuffled order, b = 12 / 16 / 20 / 24: the device-packed download = host encode bit for bit; the f32 path is exact. Stager over `bfp`, `zstd1+q0.25` and `lz4` stores: packed staging = host reference bit for bit, plus coil subsets, `upload` and resident mode | pass (`bfp_xfer_gpu.log`) |
| `hasty-recon --lib kstore` (5) | `bfp` / `bfpN` / `bfp+q` round trips within half the worst step, coil subsets, selection views. σ rule on k-space-like rows (peak 4000 σ): worst step = 0.250 σ, max error 0.125 σ, rms 4.33e-3 σ against the predicted 4.30e-3 | pass |
| `hasty-recon --lib vknot::tests::ledger_*` | the 1 mm dry plan (§9.5); f32 plans unchanged | pass |
| `hasty-recon --test bfp_state` (6 + 1 ignored) | the solver tests of §9.4, checkpoint resume, refusals across store kinds | pass (`bfp_state.log`) |
| `hasty-recon --test kspace_bfp_real -- --ignored` | the σ rule on a real store (§9.6) | pass (`kspace_real.log`) |

The solver tests need a CUDA stream handle only (the toy objectives are host arithmetic). They ran in short runs under
the GPU lock.

### 9.4 Solver: fixed point and history against f32

Two toys: a 2-D multi-coil Cartesian problem (3 frames of 50 × 38, 4 coils, variable-density masks, f64-CG
reference), and a diagonal problem with 3 sets of 150 001 / 70 000 / 150 001 values. In the diagonal one every set
spans several 64 Ki chunks, and the P²nP fixed point with a support projection is known exactly.

**No prior (the stationary point is x★; g → 0).** Every width leaves the fixed point at the f32 floor, as §3.7
predicts. In the table, the error is ‖x − x★‖ / ‖x★‖ at the end; parentheses give the distance from the f32 run where
both have not converged.

| preconditioner (2-D toy, iterations) | f32 | bfp16 | bfp12 | bfp24 | iterations to 1e-5, f32 / bfp16 / bfp12 / bfp24 |
|---|---|---|---|---|---|
| none (300, not converged) | 2.73e-3 | 2.73e-3 (2.6e-6) | 2.73e-3 (4.1e-5) | 2.73e-3 (1.3e-7) | — |
| circulant (150, not converged) | 1.54e-3 | 1.54e-3 (3.8e-6) | 1.54e-3 (6.1e-5) | 1.54e-3 (8.1e-8) | — |
| ZMSHR1 (150) | 8.6e-7 | 1.0e-6 | 1.0e-6 | 8.8e-7 | 83 / 104 / 94 / 86 |
| ZMSHR1 per set (150) | 9.5e-7 | 9.7e-7 | 9.9e-7 | 8.9e-7 | 106 / 87 / 90 / 90 |
| ZMSHR1-M (100) | 3.4e-7 | 3.4e-7 | 3.6e-7 | 3.6e-7 | 58 / 60 / 56 / 58 |
| ZMSHR1-M per set (100) | 3.5e-7 | 3.4e-7 | 3.5e-7 | 3.4e-7 | 55 / 51 / 55 / 54 |

- Diagonal toy, multi-chunk sets: ZMSHR1 bfp16 2.9e-8 against f32 3.6e-8, with the same 13 iterations to 1e-5.
  ZMSHR1-M: 2.7e-8 against 2.8e-8 (3 iterations against 2).
- The BFP loop with f32 stores equals the f32 loop exactly (0 difference).
- The iteration count to 1e-5 moves by −18 % … +25 % (bfp16). The quantised secant pair changes the step sequence,
  not the floor.

**RED** (soft threshold inside the gradient, 300 its, against a 1500-iteration f32 run): bfp16 8.9e-7 and bfp12 9.0e-7
with ZMSHR1, against the f32 floor of 1.0e-6 (ZMSHR1 per set: 1.0e-6 and 9.7e-7).

**P²nP: the correction to §3.7.** The toy of §3.7 had no prior: there g → 0 at the fixed point, so a BFP gradient is
free. With a denoiser after the step, g does not vanish at the fixed point, and the secant record must be x_prev, a
difference of non-vanishing iterates. Both change the answer:

- **A BFP gradient biases the P²nP fixed point**, in proportion to 2^−b.
  - 2-D toy, plain and circulant steps (no pair), against the f32 run at the same iteration. Converged case (blur
    κ 0.3, circulant; f32 drift 6.8e-8): bfp16 5.9e-6, bfp12 9.4e-5, bfp24 1.5e-7, i.e. 0.38 · 2^−b. Over all eight
    denoiser and step cases: bfp16 0.7e-6 – 3.4e-5, bfp12 1.4e-5 – 5.8e-4, bfp24 7.5e-8 – 1.7e-7. The coefficient
    0.05–2.2 depends on ‖α g★‖ / ‖x★‖.
  - Diagonal toy, every ZMSHR1 variant with a support projection (P²nP converges here; f32 2.7–4.5e-8 of the
    projected minimiser). A bfp16 gradient with an f32 record (`bfp16:f32`) stalls at 2.5–3.1e-5 ≈ 2 · 2^−16.
    bfp24 with an f32 record: 1.1e-7.
- **A BFP x_prev record is worse.**
  - `f32:bfp16`: 1.7e-5 with ZMSHR1; 7.9e-3 per set; 1.8e-2 – 4.7e-2 with the metric steps (the rank-one term
    amplifies the noisy s).
  - `f32:bfp24`: 1.9–2.1e-7, except plain ZMSHR1, which ended on a spike at 1.1e-4 after holding 2e-7.
- **bfp16 for everything:** stalls at 3.3–6.3e-5. The projection switched on at iteration 15 (difference record,
  then an invalid pair, then x_prev): bfp16 gives 4.7e-5 with ZMSHR1 and 2.8e-2 per set, but stays at the f32 floor
  with the metric steps.
- **bfp24 for everything:** 1.2–2.4e-7 of the minimiser for every preconditioner, and 0.4–2.4e-7 with the late switch
  (both asserted < 5e-7).
- P²nP with ZMSHR1 does not converge on the 2-D toy at any denoiser strength (soft 0.002 … 0.02, blur 0.03 … 0.3),
  not even in f32 (drift between iterations 200 and 400: 2e-2 to 0.4). So the 2-D toy gives P²nP numbers only for
  the plain and circulant steps. That is a property of the f32 solver, not of BFP.

Recommendation, replacing the host rows of §6:

| prior | store | host state at 1 mm (per frame) | fixed point |
|---|---|---|---|
| none, RED | `bfp16` (difference record) | 17.8 + 3 × 9.04 = 45.0 GiB | f32 floor |
| P²nP (denoiser after the step) | `bfp24` | 17.8 + 3 × 13.5 = 58.3 GiB | ≤ 2.4e-7 of x (1/40 of the NUFFT's 1e-5) |

Below 20 bits with a post-step denoiser, the loop prints a warning. Predicted by the 2 · 2^−b scaling (not
measured): bfp20 ≈ 2e-6, bfp22 ≈ 5e-7.

### 9.5 Memory: dry plan

At 1 mm, per frame (280 sets of 8.53 M voxels, 17.8 GiB per stack), from the ledger (memory_ledger.md §8.1, the
`StateStore::state_bytes` formula; `ledger_1mm.log`):
- f32: 71.2 GiB, refused against 57.9 GiB of host headroom;
- bfp16: 45.0 GiB (predicted 45), fits;
- bfp24 (P²nP): 58.3 GiB;
- an objective without a gradient sink (the knot-mode velocity model), bfp16: 62.5 GiB, since g and g_prev stay f32.

VRAM:
- packed staging adds one packed set (+0.02 GiB);
- resident packed k-space is 4.42 GiB on the device instead of 7.69 GiB resident f32. The 1 mm plan then fits the
  23 GiB card with the data resident (20.0 GiB peak; the f32 resident copy: 23.3 GiB, refused).

### 9.6 k-space: the σ rule on real data

The 1 mm box k-space store the reconstruction built (`target/kstore/69d8d33581799b5c.kst`: whitened, coil-compressed
to 22 virtual coils, lossless zstd, 46.9 M samples, σ = 28.27 per component from the store key). Every 4th set (70
sets); `kspace_real.log`:

| width | rms error / σ | max error / σ | bytes per complex (of f32) |
|---|---|---|---|
| σ rule (step ≤ σ/4): 15 bits for all 70 sets | 8.2e-4 | 0.071 | 3.81 (0.477) |
| bfp16 | 4.1e-4 | 0.035 | 4.06 (0.508) |
| bfp14 | 1.6e-3 | 0.141 | 3.56 (0.445) |
| bfp12 | 6.5e-3 | 0.566 | 3.06 (0.383) |

- peak / σ = 1763, so the rule gives b = 15, inside the doc's prediction (§3.6: rms ≈ 1e-3 σ, worst ≲ 0.14 σ).
- A data-only σ (median |component| of the weakest virtual coil / 0.6745) is 1.025 × the pipeline's σ: the store's
  noise level is consistent with the data.

### 9.7 Throughput and cost per iteration

**Host codec** (`bfp_bench host`, 2^26 reals, GB/s of f32 bytes; the CPU was shared with other jobs, so the
single-thread numbers vary by about 30 %; `host_codec_bench.log`):

| | 1 thread | 4 threads | 10 threads |
|---|---|---|---|
| encode, bfp16 (fast path) | 1.33 | 3.70 | — |
| decode, bfp16 (fast path) | 3.41 | 11.9 | — |
| encode, bfp12 / 20 / 24 (general path) | 0.6–1.4 | 1.3–1.4 | 13.6–14.0 |
| decode, bfp12 / 20 / 24 (general path) | 1.0–2.6 | 5.5–8 | 24–27 |
| real k-space, 15 bits, 8 threads | | | encode 9.1, decode 15.0 |

**Host update per iteration** (`step_cost_at_scale`; everything but the gradient: the sink encode of g, the
fingerprints, the ZMSHR1 passes; 3 sets × 20 M complex, 4 threads; `step_cost.log`):

| | f32 loop | bfp16 loop |
|---|---|---|
| ZMSHR1 | 0.42 s (7.0 ns per complex) | 1.22 s (20.2 ns) |
| ZMSHR1-M | 0.55 s (9.2 ns) | 1.59 s (26.5 ns) |

- At 1 mm (2.39 G complex per stack), scaled linearly: ≈ 17 s (f32) and 48 s (bfp16) on 4 threads, ≈ 7 s and 19 s
  on 10.
- Against the 219 s gradient of R0 attempt 4, the BFP update costs about 5 % more per iteration on 10 threads.
- It is decode-bound: 10 chunk decodes and 2 encodes per ZMSHR1 iteration.
- Faster encode (vectorised max and rounding) and a sampled fingerprint are the next steps if it matters.

**PCIe bytes per gradient at 1 mm** (measured ratios from `bfp_xfer_gpu.log`; time at 22 GB/s pinned):

| stream | f32 | BFP | saved | time saved | cost |
|---|---|---|---|---|---|
| gradient download, bfp16 | 17.8 GiB | 9.04 GiB (0.508) | 8.8 GiB | 0.43 s | device pack ≈ 27 GiB of DRAM traffic, ≈ 32 ms |
| gradient download, bfp24 | 17.8 GiB | 13.5 GiB (0.758) | 4.3 GiB | 0.21 s | same |
| k-space staging, σ rule (15 bits) | 7.69 GiB | 3.67 GiB (0.477) | 4.0 GiB | 0.20 s | device decode of 7.69 GiB of f32 output, ≈ 12 ms |
| k-space resident packed | 7.69 GiB | 0 (3.67 GiB once) | 7.69 GiB | 0.37 s | 3.67 GiB VRAM |

The transfer time saved is small against the gradient (≈ 0.6 s of 219 s). The point of the PCIe side is the host
copies it removes and the VRAM it frees (resident data). The memory gain is host RAM: 71 → 45 GiB (RED / no prior)
or 58 GiB (P²nP).

### 9.8 Review

An independent review of the code found the following; all are fixed:

1. The planner and the solver assumed a BFP gradient for objectives without a sink path (the velocity model). The
   solver then fell back to an f32 g while the BFP g was still allocated, a peak 18 GiB above the plan. Fix:
   `StateStore::effective(has_gradient_sink)` before allocation, and the knot plan counts g and g_prev as f32.
2. A resumed x_prev record was corrupted by x's lossy checkpoint (1e-6 rms of x in s). Fix: x is bit-exact in the
   checkpoint while the record is x_prev. P²nP resume test: 0 difference.
3. RED failed hard when the inner objective declined the sink (unequal sets). Fix: `has_gradient_sink` mirrors the
   sink's conditions, and RED falls back.
4. In the fallback, g_prev was quantised while g was not. Fix: both f32.
5. The trust-region comment overstated exactness. Corrected: with an x_prev record, a shrunk element carries the
   record's rounding.
6. `SumSink::put_packed` skipped ‖g‖². Fixed.
7. The checkpoint format check came after reading x. It now runs first and includes the preconditioner.
8. Packed `Stager::upload` made a device-to-device copy (one extra f32 set of VRAM). Fixed.
9. The mantissa buffer in the device test was not poisoned. Now it is.

The review also named three untested paths, now covered by the diagonal toy and the P²nP resume test: multi-chunk
sets, pair invalidation, and resume with an x_prev record. The toy work exposed one bug of my own: the test kit's
`uniform()` is in [−1, 1), which had made the first diagonal toy ill-conditioned.

### 9.9 Files

- `rustlib/crates/hasty-bfp/src/lib.rs` (container, views, faster block loops); `src/xfer.cu`, `src/xfer.rs` (new);
  `src/bin/bfp_bench.rs` (`host`); `tests/host_buf.rs`, `tests/xfer_gpu.rs` (new).
- `rustlib/crates/hasty-recon/src/bfpstate.rs` (new). `hoststack.rs`: `for_each_sink` and packed download, the
  dispatch. `p2np.rs`: the sink trait methods and P2np streaming. `red.rs`: the RED sink. `kstore.rs`: the `bfp`
  representation. `kstage.rs`: packed and resident staging, `HASTY_KSTAGE`. `vknot.rs`: `plan_with` and the 1 mm
  test. `Cargo.toml`: the `hasty-bfp` dependency.
- `rustlib/crates/hasty-recon/tests/bfp_state.rs`, `bfp_xfer_gpu.rs`, `kspace_bfp_real.rs` (new).
- `rustlib/crates/hasty-bench/src/bin/recon_p2np.rs`: `--state-store`, `--kstage` (13 lines).
- Logs: `research/experiments/bfp_host/`.

Rerun:

```
cd rustlib
cargo test --release -j 4 -p hasty-bfp --test cpu --test host_buf
cargo test --release -j 4 -p hasty-recon --lib kstore ledger
cargo test --release -j 4 -p hasty-recon --test kspace_bfp_real -- --ignored --nocapture        # CPU, needs the store
# GPU (short, under the lock):
cargo test --release -j 4 -p hasty-bfp --test xfer_gpu; cargo test --release -j 4 -p hasty-recon --test bfp_xfer_gpu
cargo test --release -j 4 -p hasty-recon --test bfp_state -- --test-threads=1 --nocapture     # CUDA stream handle only
RAYON_NUM_THREADS=4 cargo test --release -p hasty-recon --test bfp_state step_cost_at_scale -- --ignored --nocapture
target/release/bfp_bench host 26 5
```

## 10. Compressed DRAM traffic: budget, fixes, results (2026-11-21)

Question (the user's): make BFP as fast as its byte saving should make it; find out which f32 buffers of the gradient
really need f32; store as many DRAM loads and stores as possible compressed. Every earlier optimisation had to keep
working (block and k-support pruning, fused pad / crop, L2 slabs, split-tile zeroing, the empty-tile skip, the one-bin
static path, the vs / vt / ra / kl kernels).

Short answer:
- **Only the atomically accumulated cells need f32**: the split tiles of the adjoint scatter (about 0.1 % of the
  scatter's cells at 256³) and the crop's accumulation into the image. Every other buffer is f32 by construction only.
- **Bytes are not the limiter of most of the gradient on the 4090.** The gathers and scatters run at 130–270 GB/s,
  the L2-slab passes are bound by latency and issue slots with one 82 KB block per SM, and the fused pad / crop passes
  are latency-bound. A BFP decode or pack inside those kernels costs more time than the bytes it saves.
- **The one large waste was not a format question.** With L2 slabs, the inverse y pass's f32 output is dead once the
  fused crop has read it, but it was still written back from L2 to DRAM: 8.4 GB per coil batch at 256³ × 50 channels.
  An L2 discard after each slab removes it: gradient 402.1 → 394.1 ms (−2 %), adjoint −3.7 %. This is the new
  default (`HASTY_FFT_SLAB_DISCARD`).
- **The y ↔ z BFP store no longer pays under slabs.** With the discard on:
  - 256³: f32 390.9 ms, bfp24 391.8 ms per gradient (campaign).
  - 2-D knot velocity config: f32 21.54 s, bfp24 21.68 s.
  - Why: the L2-resident BFP y pass costs 11.6 ms more per gradient than the f32 one, and the z passes save 6.6 ms
    (ncu).
  - Decision (the user asked for the fastest gradient): **every on-device store defaults to f32**, `HASTY_SEG_FFT_STORE`
    included. The previous bfp24 default is `HASTY_SEG_FFT_STORE=bfp24`. `=auto` gives bfp24 for whole-grid passes and
    f32 under slabs.
- **The new storage switches work but are off by default (f32).** They cover the grid itself (z pass and gathers), the
  coil maps and the segment diagonals. All are bit-exact against the host codec.
  - Accuracy on ph0001 (§10.7): the maps and diagonals are at the f32 one-ulp floor down to bfp20. The grid at bfp24 is
    2 × the floor in the forward and 2.2–5.2 × in the near-convergence gradient.
  - Time: the BFP grid makes the vs gather 2.1 × slower (57 → 120 ms), so the gradient gets 6–8 % slower. The BFP maps
    cost 0.5–2.5 ms at 256³, and at 640 × 384 × 320 their copy next to the f32 one does not fit in device memory.
- **The 640 × 384 × 320 grid cannot use the BFP passes at all.** Its y and z lengths (480, 400) exceed the 396 entries
  a 16-line BFP tile allows in 99 KB of shared memory.
  - There the segment diagonals are 15 % of the gradient's bytes. The BFP diagonals are the one switch that could pay
    at that size.
  - But with the f32 copy kept, the extra 4.8 GB (bfp24) does not fit next to the engine on 24 GB.
  - To pay there, the f32 copy has to go (follow-up).

### 10.1 Byte budget of one gradient (Nsight Compute, DRAM bytes per kernel, summed)

How it was measured:
- opbench `--profile`: one gradient (forward + adjoint) of one (frame, encode) set, order 3, 1.52 M samples, on the
  RTX 4090.
- `ncu --profile-from-start off --cache-control none`, metrics `dram__bytes_read.sum`, `dram__bytes_write.sum`,
  `gpu__time_duration.sum`. One pass, so no replay; times are at ncu's locked clocks.
- Sizes:
  - 320³ grid (256³ image), 10 coils × 10 segments, coil batch 5 (50 channels, two batches per gradient);
  - 800 × 480 × 400 grid (640 × 384 × 320 image), 1 coil × 10 segments (10 channels, one batch).
- k support: 0.597 / 0.559 of the cells, 0.844 / 0.815 of the z columns.
- Files: research/experiments/compressed_dram/ncu/ (`*.phases.txt` per run, `*.seq.txt` kernel by kernel; scripts
  `eng.sh`, `phase.py`).

In the slab rows, the x and y passes keep their tile in L2. Their DRAM bytes are the write-back of what the y pass
wrote, which L2 evicts during the following slabs.

**256³ image, 50 channels, per gradient (GB read / written):**

| buffer (producer → consumer) | slab, f32 store (new default) | slab, bfp24 store (§20 default) | whole grid, bfp24 store | f32 needed? |
|---|---|---|---|---|
| pad inputs: image x, 5 coil maps, 10 diagonals per batch (fused x pass) | 4.30 / – | 4.30 / – | 4.30 / – | no (maps / diagonals: switch, §10.4) |
| x → y intermediate | in L2 | in L2 | 16.8 / 16.8 (x writes, y reads) | no (format: §10.5) |
| y → z intermediate (y pass writes, z pass reads) | 21.6 W / 17.7 R | 20.7 W / 13.4 R | 13.4 W / 13.4 R | no |
| forward grid (z pass writes, gather reads) | 15.6 W / 34.8 R | 15.6 W / 34.6 R | 15.6 W / 34.6 R | no (switch, §10.3) |
| samples, tables, residual (gather, scatter) | 7.8 R + 0.2 W | same | same | (integers; residual f32: latency-bound kernels) |
| adjoint grid (ra scatter stores, z⁻¹ reads) | 15.6 W / 15.7 R | same | same | **only the split tiles (0.02 GB zeroed, added atomically)** |
| z → y intermediate (z⁻¹ writes, y⁻¹ reads) | 17.7 W / 17.7 R | 13.5 W / 13.4 R | 13.5 W / 13.4 R | no |
| y → x intermediate (y⁻¹ writes, crop x pass reads) | 0.3 W (14.8 before the discard) | 0.3 W | 16.8 W / 16.8 R | no |
| crop inputs: coil maps + diagonals; image accumulation | 4.30 R / 0.75 W | same | 4.30 R / 0.34 W | maps: no; **the image accumulation (atomics)** |
| **total** | **102.2 R + 71.9 W = 174 GB, 438.4 ms** | **93.5 + 66.7 = 160 GB, 446.3 ms** | **219 GB, 450.6 ms** | |

**640 × 384 × 320 image, 10 channels, slab (one batch):**
- image-side inputs, read by the pad and again by the crop: 7.55 GB each. Of this, the diagonals are 6.29, the coil
  maps 0.63 and the image 0.63.
- y → z: 8.8 W / 8.0 R (f32: the BFP passes do not cover these lengths).
- forward grid: 6.9 W / 11.3 R.
- adjoint grid: 6.9 W (kl) / 6.9 R.
- z → y: 8.0 W / 8.0 R.
- y → x: 0.2 W.
- total: 83.2 GB, 208.5 ms (whole grid: 190.1 against 183.4 ms in opbench).

Ranked by bytes saved per effort, with what happened to each:

| # | change | bytes saved / gradient (256³) | outcome |
|---|---|---|---|
| 1 | drop the dead y⁻¹ output from L2 (slabs) | 14.5 GB of writes | **done, default**: −7 ms adjoint |
| 2 | grid itself in BFP, forward (z pass + gathers) | 3.9 W + up to 8.7 R | done, switch: the vs decode doubles the gather |
| 3 | grid in BFP, adjoint (owner stores + split overlay) | 3.9 W + 3.9 R | designed (§10.5), not built: same decode risk, five writer kernels |
| 4 | x ↔ y in BFP (whole grid only) | 8.4 (16.8 at 640, if covered) | not built (§10.5): slabs are the default at both sizes |
| 5 | maps + diagonals in BFP | 2.0 (bfp24) – 4.0 (bfp16); 640: 3.8 – 7.6 | done, switch: 256³ neutral to +2.5 ms; 640 in the campaign |
| 6 | y-pass write-back of the dirty last sector (slab, BFP) | 2.1 | in-kernel discard done, switch, off: +1.1 ms issue time |
| 7 | coalesced BFP loads, cheaper BFP stores | 0 (L1 / L2 traffic only) | done, switches, off: no gain (§10.2) |

### 10.2 Why the BFP passes are not faster, and the z⁻¹ store

Each pass of the y / z FFT runs one 82–87 KB block per SM: two stage buffers of N × 16 lines, N = 320, and 20 warps.
The load, the stages and the store of a block run one after the other, and no second block hides them.

Nsight Compute on the passes (320³ grid, 24 channels, ms, `ncu/passes.txt`):

| pass | f32 | bfp24, previous epilogue | bfp24, fused store | bfp24, vector loads |
|---|---|---|---|---|
| y, whole grid (writes BFP) | 10.25 | 9.10 (DRAM 87 %) | 8.55 (DRAM 93 %; per-word phase 2) | — |
| z⁻¹, whole grid (writes BFP) | 13.27 | 12.98 (DRAM 79 %) | 13.18 / 13.20 | — |
| z, whole grid (reads BFP) | 13.78 | 12.37 | — | 12.89 |
| y⁻¹, whole grid (reads BFP) | 10.08 | 8.63 | — | 8.74 |
| y, one L2 slab (writes BFP) | 23.7 µs | 26.7 µs (issue 54 %, 2.2 × instructions) | slower | — |

What was tried:
- **Fused store** (`HASTY_FFT_BFP_STORE=fused`). The half warp already holds the 16 lines of a block in the last
  stage. It takes the block's exponent by a half-warp max in registers and writes quantised integers into unpadded
  shared rows. After one barrier, each thread writes one mantissa word; consecutive lanes write consecutive words.
  - This removes the float re-read and one barrier. It helps the DRAM-bound whole-grid y pass (−6 %).
  - It does nothing for z⁻¹ (+1.5 %) and costs 21 % on the issue-bound slab y pass, because the shuffles and the
    per-word assembly add instructions.
  - The 98 M shared-memory bank conflicts of every BFP store come from the per-word assembly
    (`bfp_block_word_q`, 2-way between the entries of a warp), not from the per-run phase.
- **Packed store** (`=pack`). The per-run phase packs whole words in registers (`bfp_run_pack`), and the coalesced phase
  only copies words. Bit-identical, and no faster: slab pair 50.46 against 49.88 ms.
- **Vector loads** (`HASTY_FFT_BFP_LOAD=vec`). Phase A copies the block words into the first shared buffer with
  uint4 / uint2 loads (3.3 sectors per request instead of 7.6), then one barrier, then the runs are unpacked from shared
  memory. Bit-identical. It is 4 % slower on the z pass: the extra barrier and the shared round trip in a
  one-block-per-SM pass cost more than the L1 wavefronts it saves.
- **Why z⁻¹ reaches only 79 % of DRAM peak.** It reads 6.29 GB of f32 and writes 3.79 GB of BFP in 12.98 ms. The
  f32 z⁻¹ moves 1.24 GB more in 13.27 ms, so the BFP pass is not slower per byte than its own read stream allows. With
  one block per SM, the time is set by the serial load → stages → quantise → write of each block; the write phase
  overlaps nothing.
- **The fix for both passes is a second block per SM, not a different store.** That needs in-place stages in one
  buffer (radices that all divide the largest one; 320 = 4 · 4 · 4 · 5 does not) or 8-line tiles for the f32 passes.
  Not done.
- **In-kernel L2 discard in the BFP y pass** (`HASTY_FFT_BFP_DISCARD=1`, `BfpIo::discard`). Under slabs, the x pass
  leaves the whole 128-byte run of every block dirty in L2. The in-place BFP y pass rewrites only 3 of its 4 sectors,
  so the stale fourth sector is still written back. That is why the slab-mode BFP store saved only 4 % of the writes.
  Dropping the runs after stage 0 cuts the slab passes' DRAM writes by 20 % (4.68 → 3.76 GB per transform at
  24 channels), but the discards cost +1.1 ms of issue time per transform. Off by default.

All variants are bit-exact against the host codec (`hasty-fft --test bfp_store` under each switch).

### 10.3 The grid itself in BFP (`HASTY_SEG_GRID_STORE = f32 | bfp16 … bfp24`, default f32)

- **Store.** `PrunedChanFft3::set_grid_store` builds a second forward z pass whose output is BFP.
  - It writes in place: block (z, y, 16-line tile) holds its mantissas in the first 4 · b bytes of its own 16 cells.
  - Its exponent byte sits at index (flat cell index / 16), in the grid slot's tail. The slot grows by 1/128 (82 MB at
    256³ × 50 channels).
  - The exponent address is passed per call (`set_grid_exponents`). A call without one writes the f32 grid, so the
    multi-image and fat-group paths, which load through their own code, stay f32.
  - With a k support the exponent bytes are cleared every call: blocks the pass does not write decode as zeros.
- **Gathers.** The tile, knot-tile and vs / vt kernels are compiled with `GBFP` and decode while they stage grid
  values (`gload_bfp`: two or three word loads, the exponent byte, unpack). The adjoint is unchanged.
- **Invariant.** The engine records the width its gathers were compiled for (`gbfp_compiled`). A re-plan keeps the grid
  storage, and a load asserts that the transform agrees. This came from the review.
- **Accuracy.** The grid slot is bit-identical to the host codec over every whole block, and its error is 1.89–1.98 ·
  2^−b. On the random test problem the forward error is 66–86 · 2^−16 at bfp16 (κ ≈ 17), and bfp24 is at the f32 floor
  there. On ph0001 the near-convergence gradient is 2.2–5.2 × the floor at bfp24 (§10.7).
- **Time (256³, opbench, 450 W).** vs with decode 120.4 ms against 57.1 ms; tile 70.7 against 72.6 ms. The autotune then
  picks the tile kernel: forward 238.3 against 210.2 ms, gradient 417.4 against 394.7 ms (+6 %).
- **Why it loses.** The gathers are not DRAM-bound: vs reads 17.3 GB in 65 ms, 266 GB/s. Its window loads become 3–4
  load instructions plus integer work each.
- **What would make it pay.** A warp-cooperative decode, where the 32 channel lanes of a cell cover 2–3 consecutive
  blocks: load the words once, shuffle. Or bfp16, where one word is one complex value: one load, two sign extensions.

### 10.4 Coil maps and segment diagonals in BFP (`HASTY_SEG_MAPS_STORE`, `HASTY_SEG_DIAG_STORE` = f32 | bfp16 … bfp24, default f32)

- **What is stored.** Segment diagonals are treated only as stored maps P_l(r), whatever basis produced them.
  - When a switch is on, the engine keeps a dense BFP copy of the flat array next to the f32 array:
    - the coil maps when they are first used after any change (`set_maps`, `select_coils`, `smaps_mut`, `swap_smaps`
      invalidate the copy);
    - the diagonals in `SegSegments` (`segments`, `segments_frames`).
  - The copy is packed on the device by the warp codec (`bfpmaps.rs`), bit-identical to the host codec.
- **Who decodes it.** The fused pad / crop hooks, in their staging loops and in the gather fallbacks (`HK_S` / `HK_P`
  in `fused.rs`). Pad and crop read the same decoded values, so the pair stays adjoint.
- **What stays f32.** Every other consumer: separate pad / crop, water + fat, warp, multi-image. The f32 arrays are
  kept, so VRAM grows by 0.76 × (bfp24) or 0.51 × (bfp16) of the maps. Dropping the f32 copy once those paths can
  decode is the follow-up.
- **Accuracy (test problem).**
  - bfp16 maps: forward 1.51 · 2^−16; bfp16 diagonals: 1.0 · 2^−16; both: 1.78 · 2^−16.
  - bfp24 both: forward at the f32 one-ulp floor.
  - Adjointness unchanged (1e-7).
- **Time (256³).** Gradient 397.2 ms (bfp24) and 395.2 ms (bfp16) against 394.7 ms f32. The fused x passes are
  latency-bound (§23.4: 55–69 % of DRAM bandwidth), so the 2–4 GB they save per gradient do not shorten them, and the
  decode adds to the staging loop.
- **640 × 384 × 320.** There the diagonals are 15 % of the bytes, but the bfp24 copy (+4.8 GB) next to the f32 one ran
  out of device memory (opbench panic at its first allocation after the engine; campaign case p16). A bfp16 case
  (p16b) runs on resume. Paying there needs the f32 copy dropped.

### 10.5 Designed, not built

- **Adjoint grid in BFP.**
  - Owner-stored blocks (all 16 lines inside one writer's tile row) would be quantised by the writer from its staged
    output rows (ra and kl already park rows in shared memory) and stored in place with their exponent.
  - Blocks that straddle two tiles (TT · NCH not a multiple of 16: every other tile edge at 50 channels, every edge at
    10), blocks of channel-row splits, split tiles (atomics on zeroed cells) and zeroed empty tiles would store f32
    cells with the exponent marker 254. The codec never produces 254.
  - The z⁻¹ prologue would copy f32 blocks and decode the others.
  - Expected: about 96 % of the blocks compressed at 50 channels and 80 % at 10, saving about 7 GB per gradient at
    256³ (4 % of the bytes).
  - Work: five writer kernels (ra, kl, tile, knot tile, multi), two zero kernels, the z⁻¹ prologue. Given §10.3 (the
    scatter is not DRAM-bound either: ra writes 15.6 GB in 115 ms), it was not built.
- **x ↔ y in BFP for whole-grid passes.**
  - The x pass's tile is 16 channels of one (y, z) line; the y pass's tile is 16 consecutive (x, channel) lines. The
    blocks match only when nch is a multiple of 16 (neither 50 nor 10).
  - Otherwise the forward x pass writes channel-group blocks per cell (the last group truncated to its words, so it
    fits in place), and the y pass decodes them by random access. The inverse is the other way round.
  - Worth 8.4 GB per coil batch at 256³ in whole-grid mode, which today runs as fast as the slab mode (450.6 against
    446.3 ms at ncu clocks).
  - It matters where a plane does not fit the L2 (1 mm torso, 92 MB planes at 50 channels). Not built.
- **BFP passes for long axes.** y / z lengths above 396 (the 640 × 384 × 320 grid) need 16-line tiles in one shared
  buffer, or two-pass tiles of 8 lines with blocks spanning two tiles.

### 10.6 Why bytes do not buy time here: achieved bandwidth per kernel

DRAM bytes over kernel time, from the ncu runs of §10.1 (f32 store, slabs; ncu clocks; peak 1008 GB/s):

| kernel | 256³, 50 channels | 640 × 384 × 320, 10 channels | bound by |
|---|---|---|---|
| forward z pass (whole grid) | 33.4 GB / 38.3 ms = 870 GB/s (86 %) | 14.9 GB / 18.7 ms = 798 GB/s (79 %) | DRAM |
| inverse z pass | 33.4 GB / 38.4 ms = 869 GB/s (86 %) | 14.9 GB / 18.6 ms = 802 GB/s | DRAM |
| inverse y pass, slabs (reads DRAM, writes L2) | 18.0 GB / 27.4 ms = 657 GB/s | 8.2 GB / 15.3 ms = 536 GB/s | DRAM + L2 latency |
| fused pad + x pass, slabs (incl. the y pass's write-back) | 22.1 GB / 39.4 ms = 560 GB/s | 15.4 GB / 25.6 ms = 602 GB/s | latency (one block per SM) |
| forward y pass, slabs | in L2: 3.8 GB / 19.7 ms | 1.0 GB / 11.4 ms | L2 latency, issue |
| fused crop + x pass, slabs | 5.1 GB / 27.7 ms = 183 GB/s | 8.0 GB / 18.9 ms = 425 GB/s | latency (staging, reductions, atomics) |
| gather (vs / vt) | 35.0 GB / 130.0 ms = **269 GB/s (27 %)** | 11.4 GB / 56.0 ms = **203 GB/s (20 %)** | latency, issue (window loads, MACs) |
| scatter (ra / kl) | 23.4 GB / 114.7 ms = **204 GB/s (20 %)** | 9.4 GB / 41.9 ms = **225 GB/s (22 %)** | latency, issue |

What the table shows:
- **Only the whole-grid z passes are DRAM-bound.** They are the passes where BFP did save time (13.8 → 12.4 ms at 24
  channels).
- **The gathers and scatters reach 20–27 % of DRAM peak.** Compressing their grid moves fewer bytes through a pipe
  that is not full, and the decode adds issue work to a kernel that is issue- and latency-bound: the vs gather went
  from 57 to 120 ms.
- **The slab passes live in L2.** They gain nothing from fewer DRAM bytes. Their cost is latency with one block per SM,
  and the BFP epilogue adds instructions there.
- **Bytes do buy time where a DRAM-bound stream is pure waste.** The dead write-back removed by the slab discard is such
  a case.

### 10.7 Campaign (cz-campaign; research/experiments/compressed_dram/)

RTX 4090 at the user's 450 W power limit (logged per case, never changed). OLD = the binaries frozen before this
session (§20 kernels, no slab discard) run with the f32 store; NEW = this session's binaries at their defaults (slab
discard on; f32 store under slabs, which equals the final f32 default for every case below, all of which run with
slabs).

**FFT pair, 320³ grid, 256³ image, 24 channels** (bfp_fft_bench, plain passes without the fused pad / crop, so the
slab discard does not apply; median of 15, ms):

| store | slab 1 | whole grid |
|---|---|---|
| f32 | 51.38 | 62.78 |
| bfp24, default epilogue | 49.87 | 58.34 |
| bfp24, fused store | 50.98 | 58.28 |
| bfp24, vector loads | 51.09 | 58.51 |
| bfp24, y-pass discard | 50.62 | — |
| bfp22, dense buffer | — | 61.35 |

**One-set gradient, order 3** (opbench, forward + adjoint incl. FFTs; one binary per row; ms):

| case | forward | adjoint | gradient |
|---|---|---|---|
| 256³, 10 coils × 10 segments, batch 5: OLD | 206.3 | 191.0 | 397.8 |
| NEW | 207.1 | 181.7 | **390.9 (−1.7 %)** |
| NEW with the y ↔ z store bfp24 | 211.2 | 179.6 | 391.8 |
| NEW with every BFP buffer (y ↔ z, grid, maps, diagonals bfp24) | 238.3 | 182.2 | 420.5 (+7.6 %) |
| 640 × 384 × 320, 1 coil × 10 segments: OLD | 94.8 | 85.5 | 180.1 |
| NEW | 94.8 | 82.4 | **177.2 (−1.6 %)** |
| NEW with every BFP buffer | — | — | out of device memory: the BFP diagonal copy (+4.8 GB) next to the f32 one |
| 2-D knots 6:1,16:3, 288 × 288 × 140 grid, 10 coils × 10 segments, 13.3 M samples: OLD | 10275.8 | 11458.3 | 21783.7 |
| NEW | 10277.3 | 11219.3 | **21537.4 (−1.1 %)** |
| NEW with the y ↔ z store bfp24 | 10520.7 | 11150.4 | 21677.9 |

The gain is the slab discard (adjoint −2 to −5 %).

**Accuracy on ph0001** (bfp_phantom `--switches`).
- Setup: image 254 × 144 × 114, grid 320 × 180 × 144, 10 virtual coils × 10 segments, coil batch 5, order 3. Each
  switch configuration is compared with an all-f32 engine; no autotune, so the default kernels run.
- Floor = the f32 engine with x (or n) moved by one ulp. Spread = the f32 engine run to run and engine to engine.
- Cells give the error / floor. Floors for the static case (forward, Aᴴ n, Aᴴ A x, Aᴴ(A x − y)): 3.6e-7, 6.9e-6, 4.5e-6,
  5.1e-5; the spread is up to 9.5e-6. For 2-D knots 3:1,8:3 (24 knot images, 1.36 M samples): 3.1e-7, 5.7e-6, 3.2e-6,
  2.0e-5.
- Logs: `logs/a01_static_o3.log`, `logs/a02_knot2d_o3.log`.

| configuration | static: forward / Aᴴ n / Aᴴ A x / converged | 2-D knots: forward / Aᴴ n / Aᴴ A x / converged |
|---|---|---|
| y ↔ z bfp24 | 0.98 / 1.49 / 1.32 / 1.31 | 1.03 / 0.91 / 1.57 / 1.05 |
| grid bfp24 | 2.01 / 1.40 / 1.12 / **5.15** | 1.82 / 0.81 / 1.48 / **2.24** |
| grid bfp22 | 5.78 / 1.38 / 1.18 / 16.4 | 6.91 / 0.79 / 1.74 / 13.1 |
| grid bfp20 | 23.3 / 1.43 / 2.35 / 59.6 | 27.6 / 0.80 / 2.56 / 43.7 |
| coil maps bfp24 | 0.89 / 1.34 / 1.73 / 0.84 | 0.90 / 0.80 / 1.46 / 0.79 |
| diagonals bfp24 | 0.91 / 1.39 / 1.19 / 1.05 | 0.89 / 0.79 / 1.45 / 0.88 |
| maps + diagonals bfp20 | 1.06 / 1.46 / 1.24 / 1.32 | 1.07 / 0.83 / 1.47 / 1.00 |
| maps + diagonals bfp16 | 4.21 / 3.89 / 5.02 / 2.62 | 4.83 / 4.43 / 7.03 / 2.69 |
| all four at bfp24 | 2.09 / 1.44 / 1.73 / 4.97 | 2.04 / 0.89 / 1.63 / 3.12 |

What the table shows:
- **At the floor (within the run-to-run and engine-to-engine spread):** the y ↔ z store at bfp24, and the coil maps
  and diagonals down to bfp20.
- **The grid is the one buffer that does not reach the floor.** At bfp24 its forward error is 2 × the floor and the
  near-convergence gradient 5.2 × (static) and 2.2 × (knots); at bfp22 the gradient is 13–16 ×.
- **Why the grid differs.** A grid block is 16 lines of one (z, y) row (1.3 cells at 12 channels, a third of a cell at
  50). Its step is set by the block's largest value. The near-convergence gradient is noise-driven and weights the low
  k-space periphery, where values are small. In the intermediates, by contrast, the next FFT pass spreads every
  block's error over the whole line.
- **Consequences for the defaults:**
  - The grid store is off by default on accuracy as well as on time.
  - A grid store would need bfp24 or wider, or a per-cell exponent.
  - Maps and diagonals at bfp20 cost nothing in accuracy; they are off by default only because they did not save time
    (§10.4).
