# Speed of the sampling operator at 2 mm (L = 13), and of the prior step (2026-11-12, interrupted)

Scope: wall time of one P²nP iteration of the 2 mm ph0001 reconstruction (280 sets = 40 frames × 7 encodes, 31 virtual
coils, `--seg-basis svdn-union`, joint water + fat with per-frame maps, correction order 2, RTX 4090 at 350 W) at fixed
accuracy. Algorithmic levers (coil subsets, convergence) are the other agent's. Code: `rustlib/.agents/seg`, patch
`rustlib/.agents/opspeed.patch`. Data: `data/hasty/studies/operator_speed_2026-11-12/` (README there).

**Status.** The reconstruction queue held the GPU until 03:35; then quiet mode (no GPU, builds, tests or heavy CPU).
So: the analysis, the implementation, three new GPU tests (passed, run under the queue's load), an accuracy scan of the
NUFFT on a small grid and the GPU-LLR check on real data are done; **no uncontended timing has been measured**. Every
time below that is not from a recon log is an estimate and marked [CHECK]; the runs that settle them are scripted (§9).

## 0. Summary

| component of one iteration (pyr_2mm L2, log) | today | source / reasoning | with the levers below [CHECK] |
|---|---|---|---|
| gridding (tile gather + scatter) | ≈ 155–174 s | autotune 54.5 / 69.8 ms per **coil batch of 7** × 5 batches (7, 7, 7, 7, 3) × 280 sets | ≈ 75–90 s (W 7: (W+3)³ tile points 1000 vs 2197) |
| FFT + pad / crop + other GPU | ≈ 129–148 s | 303 s − gridding; FFT roofline 81 s for the 455 channel-FFTs actually run per set (403 needed) | ≈ 85–95 s (balanced batches: 403 channels; roofline 72 s) |
| gradient pass | **303 s** | log, 20 iterations, ±0.3 s | **≈ 165–185 s** |
| step (host ZMSHR1) | 2.5 s | log | 2.5 s |
| prior (CPU LLR + air map) | **145–193 s** (mean ≈ 167) | log; air map 0.4 s | **≈ 20–30 s** (GPU LLR, f64) |
| iteration | ≈ 470 s | | ≈ 190–215 s |

Ranked levers (expected gain per iteration; accuracy):
1. **LLR prior on the GPU** (implemented, default `--llr-device auto`): ≈ −140 s [CHECK timing]; equal to the CPU path to
   f32 rounding on real 2 mm data (8.4e-5 of the update, 1.8e-6 of max |x|).
2. **ES kernel width W 10 → 7 at σ 1.25** (flag `--seg-nufft 1.25:7`; default unchanged): ≈ −70 to −90 s of gridding
   [CHECK]; NUFFT error (small-grid scan, §5) forward 4.5e-6 (phantom) / 1.5e-4 (white image), adjoint 1.45e-4 = today's
   1.6e-4 → expected R change ≤ +0.1 % to +3 % of 0.0375 [CHECK with `run_accuracy.sh`]. W 8 is *more* accurate than
   W 10 in the white / adjoint case (3.9e-5) at 0.61× the tile work; W 6 fails the guard in the worst case.
3. **Balanced coil batches** (implemented, default on): the short remainder batch of 3 coils paid the FFTs of all
   7 × 13 channels → 455 → 403 channel-FFTs per set (−11 % FFT, ≈ −9 to −15 s); together with the autotuner now trying
   every divisor of the batch, the prime batch 7 no longer forces idle coil lanes [CHECK].
4. **k-space read-ahead** (implemented, default on): the store decode of set t+1 (66–289 ms per set measured, 6
   threads, under load) runs during set t instead of before it; today hidden behind the previous adjoint (≈ 0.5 s),
   after levers 2–3 it would not be (adjoint ≈ 0.25–0.3 s [CHECK]).
5. Not adopted, with numbers: σ < 1.25 (≤ 11 % of the FFT at the roofline, needs W ≥ 8–9 and worsens the f32 floor),
   σ > 1.25 (FFT × 1.4 at 1.5), f16 / bf16 FFT storage (the f32 rounding already sets the adjoint error at σ 1.25),
   pruning to the object support (the masks span the whole FOV), Toeplitz normal operator (24 GB of kernels per set;
   2.4× today's FFT work even in the O(L) form), FFT precision games, coil batch 8 (`--seg-grid-gb 7`, small, [CHECK]).

## 1. Where the gradient time goes

Per set the operator does, per coil batch of cb = 7 coils (the 6 GiB grid budget / (13 segments × 320·180·144 × 8 B)):
pad (S_c P_l x / φ̂, 91 channels) → pruned channel-interleaved 3-D FFT → tile gather (forward); tile scatter → inverse
FFT → crop (adjoint). 31 coils = 5 batches.

* **Gridding.** The `kernels 60.3/66.9 ms (set 0)` / `54.5/69.8 ms` lines of the log are `autotune` timings of the
  busiest interval for **one coil batch** (c0 = 0, ncb = cb; `SegNufft::autotune_multi_in`), not of a set. Per set:
  5 batches → 551–622 ms (the 3-coil batch costs between 3/7 and 1 of a full one), i.e. 155–174 s per pass.
  fix_2mm (L 22, cb 4): 8 batches × 95.2 ms = 738–762 ms per set. So per set the gridding got **faster** (−20 %)
  from fix_2mm to pyr_2mm; the per-call number grew because a call now handles 7 coils × 13 segments (91 channels)
  instead of 4 × 22 (88). Per channel it did get 27 % slower (1.37 vs 1.08 ms per coil-segment): batch 7 is prime and
  the autotuner tried only coils-per-block ∈ {1, 2, 3, 4, 6}, i.e. idle coil lanes (4: 8 slots for 7 coils; 6: 12) or
  the tile weights recomputed per coil (1) [CHECK: nsys of cb 7 vs 6]. `--seg-table-stride 4` only shortens the
  interpolator table (same per-sample linear interpolation in the kernel); the union-fat cardinal path uses the same
  kernels (`forward_fat`) as fix_2mm. Neither changes the kernel cost.
* **FFT.** hasty-fft's pruned channel FFT moves, per channel and direction, 289 MB at σ 1.25 (x pass over the 144·114
  block lines, y pass over the 114 block planes, z pass over all lines). 403 channels × 2 directions = 233 GB per set =
  259 ms at 0.9 TB/s = **72 s per pass** at the roofline; today 455 channels run (the remainder batch transforms all
  91 channels, its pad writes 39) → 81 s. pad + crop ≈ 37.7 GB per set → 12 s.
* **Host side.** GPU utilisation during the gradient phase is 100 % (nvidia-smi, 1 s samples over a 10-min window:
  373 × 100 %, the 224 zero samples are the prior phase). k-space staging (decode on the main thread before the
  forward) and table unpacking overlap the previous set's adjoint today. Image stacks: 2 × 8.7 GiB per pass over PCIe
  (≈ 1 s), negligible.
* The 267 s "all else" of the task statement = FFT + pad/crop + other, **plus** the gridding hidden in the
  per-batch kernel numbers (the statement multiplied them by 280, not by 280 × 5).
* The GPU is power-limited (350 W, clocks 2.5–2.8 GHz under the gridding kernels): less FMA work also means higher
  clocks [CHECK].

## 2. Why the prior is 3× slower (56 → 165 s)

The CPU quota. `app.slice` got `CPUQuota=1000%` on 2026-11-10 20:12:55 (drop-in mtime,
`/run/user/1000/systemd/user.control/app.slice.d/50-CPUQuota.conf`); fix_2mm/llr_s finished 19:34 that day (log mtime),
so it ran unthrottled; the pyramid runs are `systemd-run --user --scope` jobs in `app.slice`. `Llr::apply` spawns
`available_parallelism()` = 32 threads (8 P-cores × 2 + 16 E-cores, ≈ 19 P-core equivalents) and is pure CPU
(nalgebra, f64 Gram + Hermitian eigensolver per 8³ block). Same work in both runs (same grid, block, 280 images,
support 3.05 vs 3.10 M voxels). Under the quota the 32 threads get 10 CPUs of time: ≈ 19 / 7 ≈ 2.7× [CHECK: not
separately timed under both settings; the measured ratio is 2.95]. The prior also varied 145–193 s with other load on
the machine (my builds / tests during iterations 12–14 cost the queue ≈ +30–100 s per iteration; sorry).

CPU cost per block (`examples/llr_prof.rs`, single thread, 512 × 280, under load): Gram 65–101 ms, eigensolver 85–179 ms,
filter 15–26 ms, A F 29–56 ms → 0.2–0.36 s per block; ≈ 9 500 blocks per call at 2 mm (≈ 627 per z-slab × 15.25
slabs; 1 626 on a 16-plane crop) [CHECK the exact count on the full stack].

## 3. The GPU LLR (lever 1)

`hasty_recon::llr_gpu::LlrGpu` (new; `llr_gpu.rs`, `llr_gpu.cu`): the **same operator** as `Llr::apply` — the block
enumeration and the cycle-spinning shift are now shared methods (`Llr::next_shift`, `Llr::block_boxes`; the CPU path
uses them, behaviour unchanged), σ̂ = median over blocks of each block's lower-quartile singular value, soft threshold
κσ̂. Per z-slab of blocks (all 280 images of the slab uploaded through a pinned buffer): gather the 512 × 280 Casorati
matrices (zero rows for partial blocks), Gram AᴴA by cuBLAS `ZgemmStridedBatched` (f64), eigendecomposition by
cuSOLVER `cusolverDnXsyevBatched` (f64), weights w = (s − thr)/s, F = V diag(w) Vᴴ and A F by `CgemmStridedBatched`
(f32, as the CPU path), scatter. Two passes per call (pass 1 eigenvalues only → σ̂; pass 2 the filter): the stack never
lives on the device. Device memory ≈ 16 MB per block of a batch (8.3 MB of it cuSOLVER workspace) + one slab
(0.65 GB at 2 mm): ≈ 1.7 GB at `--llr-batch 64`, freed after every call (`release_device`; the pinned host slab
stays). Blocks whose eigensolve fails are left unchanged, as on the CPU.

* Equality: `tests/llr_gpu.rs` (synthetic, nt 16 / 24 / 280, partial blocks, two consecutive shifts) max |Δ| ≤ 4.4e-6
  on values ≈ 1, ≤ 6.7e-5 of the update; on real data (z 48:64 of `fix_2mm/llr_s/recon_seg_zmshr1s_k1.npy`,
  1 626 blocks, the run's support mask) ‖gpu − cpu‖ / ‖cpu − x‖ = 8.4e-5, max |gpu − cpu| / max |x| = 1.8e-6, σ̂ equal
  to 7 digits (`llr_compare.json`).
* f32 eigensolver modes (`LlrPrec::Gram64`, `F32`, kept in the code, not reachable from recon_p2np): **not replicas.**
  The absolute eigenvalue error ≈ 1e-6 λ_max of a block swamps the lower quartile: on partial blocks (fewer voxels
  than the 210 needed for a non-zero s[n/4]) σ̂ became 0 on a real 8-plane crop (CPU: small but non-zero), and on
  synthetic data the output moved by 1e-2–6e-2 of the update. They are ≈ 4× faster in the eigensolver.
* Speed, measured under the queue's load only: cuSOLVER batched f64 n = 280 ≈ 2.0 ms per matrix with vectors at batch
  64 (2.4 ms without vectors at batch 16; f32 0.8–1.0 ms) (`examples/syev_probe.rs`); the real 16-plane crop took 14.4 s
  at batch 16 (8.9 ms per block for both passes). Expected uncontended at batch 64–128: 2.5–3 ms per block for both
  passes → **20–30 s per call** [CHECK `bench_rest.sh llr`]; the f64 Gram adds ≈ 5 TFlop f64 per call (≈ 4–5 s at the
  4090's 1.3 TFlop/s) [CHECK].
* recon_p2np: `--llr-device auto|gpu|cpu` (default auto = GPU; the CPU path if the device buffers cannot be allocated,
  before the shift is drawn, so the fallback call uses the same shift), `--llr-batch B` (default 64). It logs
  `LLR (GPU, f64 eigensolver): … s per call` on calls 1, 10, 20, …. The velocity-model path (`llr_z`, `llr_p`) still
  uses the CPU.

## 4. Coil batching and the autotuner (lever 3)

* `SegNufft::balance_coils` (default on; env `HASTY_SEG_BALANCE=0` off): the single-image multi path (`forward_fat` /
  `adjoint_fat`, the recon's path) runs ⌈nc / cb⌉ batches of ⌈nc / nb⌉ or ⌊nc / nb⌋ coils: 31 coils at cb 7 →
  7, 6, 6, 6, 6. Each batch size has its own kernels (NCB), FFT plans (cbm · 13 channels) and autotuned tiles, in the
  engine's slot memory (no extra grid memory). `autotune_multi(…, 1)` tunes every size in the schedule. Order-3 and
  grouped (warm-start pooling) paths are unchanged. A coil subset (`select_coils`) of a size not seen before gets its
  states compiled on first use with the default tile shape (not autotuned) [CHECK for the stochastic-coil work].
* The autotuner (`autotune`, `autotune_multi_in`) now also tries every divisor of the batch as coils per block
  (before: only {1, 2, 3, 4, 6, 8, 12, 16} ≤ cb, which excludes 7).
* Test `hasty-nufft tests/seg_balance_gpu.rs`: (nc, cb) = (7, 3), (10, 4) with host-packed streamed tables, (5, 2):
  balanced = plain schedule to ≤ 1.3e-6, adjoint identity ≤ 3.2e-6. **Pass.**
* `--seg-grid-gb 7` (cb 8 → 8, 8, 8, 7): one batch fewer for the gridding's per-(sample, tile) staging, same FFT
  channels; ≈ 0.9 GB more device memory [CHECK].

## 5. NUFFT accuracy vs (σ, W) (lever 2)

`opspeed acc`: the joint water + fat SegNufft (L 13, order-2 ΔC ≈ 0.1 rad over the FOV, a fat map, smooth coil maps
and segments) on a 96 × 64 × 48 grid with the real sample positions (k·Δ) of set (cardiac 0, encode 0), every 4th
sample (189 525), 4 coils; relative ℓ2 error against σ 2, W 14 (2:14 vs 2:16 agree to 1e-6). Forward on a phantom
(smooth inside an ellipsoid) and on a white image (every mode populated: the worst case of the forward); adjoint of
white data (what the residual looks like). `accuracy_small.json`.

| σ | W | grid | fwd, phantom | fwd, white | adjoint, white data | tile points / sample (TT 4) |
|---|---|---|---|---|---|---|
| 1.25 | 10 (today) | 120·80·60 | 2.8e-6 | 5.7e-5 | 1.6e-4 | 2197 |
| 1.25 | 8 | | 2.8e-6 | 3.7e-5 | 3.9e-5 | 1331 |
| 1.25 | 7 | | 4.5e-6 | 1.46e-4 | 1.45e-4 | 1000 |
| 1.25 | 6 | | 2.4e-5 | 6.6e-4 | 6.6e-4 | 729 |
| 1.25 | 5 | | 1.6e-4 | | 3.1e-3 | 512 |
| 1.375 | 6 | 140·90·70 | 1.5e-5 | | 1.3e-4 | 729 |
| 1.5 | 7 | 144·96·72 | 2.9e-6 | | 1.4e-5 | 1000 |
| 1.5 | 6 | | 1.0e-5 | 7.9e-5 | 7.8e-5 | 729 |
| 2 | 6 | 192·128·96 | 4.8e-6 | | 1.2e-5 | 729 |
| 2 | 5 | | 2.7e-5 | 1.1e-4 | 1.1e-4 | 512 |

(ΔC off: the same numbers to 5 %; the ΔC correction is not what limits W.)

On the noise-referenced scale R ≈ ε × ‖Ax‖/(σ√(MC)) ≈ ε × 47 at 2 mm ph0001 (mean per-sample weight 1 + E/(Cσ²) =
2.21e3 in the recon log) [CHECK: the factor for a real image is between the phantom and white cases]. The guard
R ≤ 1.1 × 0.0375 allows an independent extra R of √(0.04125² − 0.0375²) = 0.017, i.e. ε ≲ 3.7e-4. W 7: 2e-4–7e-3 of R
(0.1–3 % of 0.0375 in quadrature) → passes; W 6: up to 0.031 in the white case → fails the worst case; W 8: better
than today in the white / adjoint cases. The adjoint error at σ 1.25 does not fall below ≈ 1.5e-4 for W ≥ 7 (except
W 8); at σ 1.5 it is 10× lower — consistent with f32 rounding amplified by the deconvolution 1/φ̂ (φ̂(0)/φ̂(edge) = 86
per axis at W 10, 20 at W 7) plus the f32 accumulation in the tile scatter [CHECK: the W 10 > W 8 inversion is not
explained; the quadrature of φ̂ is accurate to 1e-9–1e-7, ruled out]. The deciding run is R on the v2 validation case
(`run_accuracy.sh`: 32 coils, real image, union L 15, against the direct sum of the operator's own model and against
the exact physics).

Gridding cost ∝ (W + TT − 1)³ tile points per sample per channel (W³ useful): TT 2 / 3 may beat TT 4 at W 6–7
(512 / 729 points vs 1000 at W 7, but 64 / 27 (sample, tile) pairs instead of 15.6) [CHECK `bench_op.sh w7_tt2 w7_tt3`].
The tile lists shrink with W too: 189 B/sample at W 10 → ≈ 112 at W 7 [CHECK], i.e. host tables 37 → ≈ 22 GiB raw,
less upload per set, and the setup's tile-list build (336 s of CPU) ≈ 2× shorter [CHECK].

## 6. k-space read-ahead (lever 4)

`Stager::upload_ahead(st, t, next)`: set t from the decode started by the previous call (or decoded now), the decode of
`next` started on a worker; two extra pinned buffers of the largest set (≈ 0.4 GB). `P2np::gradient_host` uses it
with next = (t + 1) mod n (default; `HASTY_KSTAGE_AHEAD=0`: the old synchronous `upload`). Decode measured
(`opspeed kread`, zstd1 store from disk, 6 threads, under the queue's load): 66, 250, 135, 289 ms for four sets of
≈ 200 MB. Test `hasty-recon tests/kstage_ahead.rs` (lz4 and zstd1 stores, two passes with wrap-around, an
out-of-order request): identical to `upload`. **Pass.** The packed sample tables are still unpacked on the main thread
(`--table-prefetch` uploads them on a second stream; measure with `opspeed op --prefetch`) [CHECK].

## 7. Levers rejected, with numbers

* **σ.** FFT traffic per channel and direction (pruned): σ 1.15 256 MB, 1.2 287, 1.25 289, 1.375 343, 1.5 405, 2 720
  (smooth sizes 294·168·140, 320·180·140, 320·180·144, 350·200·160, 384·216·180, 512·288·240). Per pass at the roofline
  (403 channels): 64 / 72 / 72 / 86 / 102 / 181 s. Below 1.25 at most −8 s of FFT for a wider kernel (W 8–9 for the same
  error) and a larger deconvolution amplification; above, the FFT grows faster than the gridding shrinks (σ 1.5 W 6:
  ≈ 0.33 × gridding + 1.4 × FFT ≈ 183 s vs ≈ 169 s at 1.25 W 7) [CHECK].
* **f16 / bf16 FFT storage.** The f32 rounding through 1/φ̂ already sets the adjoint error at σ 1.25 (≈ 1.5e-4 for every
  W ≥ 7); f16 (unit roundoff 4.9e-4 vs 6e-8) would raise it ≈ 8 000× [CHECK: estimate from the roundoff ratio, not run].
  No.
* **Pruning to the object support.** The coil-map and support masks of fix_2mm and pyr_2mm span the full grid
  (bounding box 254·144·114, pyr support 252·144·114; `--coil-mask none` makes S_c x non-zero everywhere): the FFT is
  already pruned to the image block, nothing more to drop for this FOV. The z-FOV (228 mm) vs the 175 mm slab would
  give ≈ 20 % of the x/y passes only if the image were zero outside the slab, which the support says it is not.
* **Toeplitz normal operator.** Per set L(L+1)/2 = 91 kernels on the (2N)³ = 33.4 M grid = 24.3 GB (6.8 TB for 280 sets):
  cannot be stored; regenerated per set and iteration the kernels cost a type-1 NUFFT of 91 weight vectors + 91 FFTs of
  33.4 M. Applying even the O(L′) form (Fessler 2005 eq. 24, research/RESEARCH_fast_normal_operator.md) takes
  2 L′ × 31 FFTs of the zero-padded 2N grid per set: with L′ = L = 13 that is 33.4 M × 0.44 (pruning at σ 2) vs
  8.29 M × 0.73 per FFT → **2.4× today's FFT work**, before the larger L′ the pair exponential needs and the per-sample ΔC
  terms (not Toeplitz-separable). K/N³ = 0.76 M / 4.17 M = 18 % per set (2.3 % of the 2N grid). **No.**
* **Fusing pad / crop into the FFT's first / last pass** (hasty-fft has pre/post diagonal operands, not the 91-channel
  coil × segment products): saves one write + read of the block per channel, ≈ 67 of ≈ 320 MB per channel-direction
  → ≈ 12–15 s per pass [CHECK]. Not done (new codegen); the next FFT-side lever after the above.
* **Forward + adjoint fusion / Aᴴy reuse.** The gradient needs Aᴴ(Ax − y): the residual of a sample needs all its
  tiles before it can be scattered, so gather and scatter stay two passes; Aᴴy precomputed would not remove any
  transform. Nothing to gain.

## 8. Implementation and tests

All in `rustlib/.agents/seg`, additive; defaults changed: GPU LLR (auto), balanced batches, autotune candidates,
k-space read-ahead — each equal to the old path to rounding (tests below), each with an off switch.

| file | change |
|---|---|
| `Cargo.toml` (workspace) | cudarc features + `cublas`, `cusolver` (dynamic loading; libcublas / libcusolver of CUDA 13) |
| `hasty-recon/src/llr.rs` | `next_shift`, `block_boxes` shared by CPU and GPU (CPU behaviour unchanged) |
| `hasty-recon/src/llr_gpu.rs`, `llr_gpu.cu` | GPU LLR (§3) |
| `hasty-recon/src/kstage.rs`, `p2np.rs` | `Stager::upload_ahead`, used by `gradient_host` (§6) |
| `hasty-nufft/src/segnufft.rs` | balanced coil batches (`balance_coils`, `multi_index_cb`, `coil_groups`, `for_coil_groups`, `forward_multi_rng` / `adjoint_multi_rng`), autotune candidates incl. the batch's divisors |
| `hasty-bench/src/bin/recon_p2np.rs` | `--llr-device`, `--llr-batch`, `--seg-nufft σ:W`, `--seg-grid-gb G`; log line names "per coil batch" and the batch count |
| `hasty-bench/src/bin/opval_v2.rs` | `forward --nufft σ:W,…` (one SegNufft per variant on the same segments, tags `<tag>_s<σ>w<W>`) |
| `hasty-bench/src/bin/opspeed.rs` | new bench: `op` (per-set operator + pipeline with k-space staging), `acc` (§5), `llr`, `kread` |
| `hasty-recon/examples/llr_prof.rs`, `syev_probe.rs` | CPU LLR cost per block; cuSOLVER batched eigensolver probe |

Tests (run while the queue loaded the GPU; correctness only):
* `hasty-recon --test llr_gpu` `llr_gpu_matches_cpu` — **pass**
* `hasty-nufft --test seg_balance_gpu` `balanced_coil_batches_match_the_plain_schedule` — **pass**
* `hasty-recon --test kstage_ahead` `upload_ahead_matches_upload` — **pass**
* not yet run (quiet mode): the existing suites the change touches — `hasty-nufft` segnufft
  (`segnufft_matches_direct_sum`, `segnufft_is_adjoint`, …), seg_fat_union_gpu, seg_stream, seg_multi;
  `hasty-recon` host_state, grouped_sets, fat_direct, stream_tables, coil_sampling, `--lib llr`; `hasty-bench`
  seg_physics_direct (`bench_rest.sh tests`).

## 9. What still has to run (scripted; each waits for `data/hasty/GPU_BENCH_LOCK`)

1. Build `rustlib/.agents/seg` (`cargo build --release -j 4 -p hasty-bench --bin opspeed --bin recon_p2np --bin opval_v2`;
   the last build is current except for nothing — all edits since were scripts / notes).
2. `run_accuracy.sh` (≈ 12 min, CPU segment build ≈ 6 min then 7 variants): R of σ 1.25 W 10/8/7/6/5 and σ 1.5 W 6/5 on
   case_v2 vs the operator's own model (today 0.0375) and vs the exact physics (0.425). Decides W.
3. `bench_op.sh` (≈ 40–55 min, 11 variants; split in two lock periods): per-set operator and pipeline time for
   W 10 plain (today), W 10 / 8 / 7 / 6 balanced, W 7 with TT 2 / 3, W 6 TT 2, cb 8 at W 7 and W 10, σ 1.5 W 6; plus
   one `--prefetch` and one `--no-ahead` run of the chosen variant.
4. `bench_rest.sh nsys llr tests`: the kernel split (today, balanced, W 7), the full-stack GPU LLR at batch 32–256 and on
   the pyr_2mm L2 result, the GPU test suites of §8.
5. `figs.py` and a `breakdown.json` from 3–4 (per-set times × 280, nsys shares, LLR call time); then fill §0.

## 10. Open questions

* The adjoint (white-data) error floor ≈ 1.5e-4 at σ 1.25 and its W 10 > W 8 inversion [CHECK]: f32 accumulation in
  the tile scatter or 1/φ̂ amplification? An f64 accumulation test of the scatter would tell; if it is the scatter,
  Kahan / pairwise accumulation per tile would buy accuracy for free.
* After levers 1–4 the iteration is ≈ 45 % gridding, ≈ 45 % FFT [CHECK]; the next steps would be the fused pad/crop FFT
  (≈ −12–15 s) and TT tuning; beyond that only fewer channels (L, coils: the other agent's algorithmic side).
* Should W become the default (`--seg-nufft 1.25:7`, or 1.25:8 if R prefers it)? Only after `run_accuracy.sh`.
* The GPU LLR's f64 eigensolver is latency-bound in cuSOLVER (2 ms per 280 × 280 vs ≈ 0.1 ms of f64 flops); a
  custom batched eigensolver or an exact f32 + f64-refinement scheme could take the prior to a few seconds. Not tried.
* The velocity-model (vmodel) path's LLR is still on the CPU.
