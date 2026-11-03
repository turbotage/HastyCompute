# Baselines (RTX 4090, CUDA 13.0, driver as installed 2026-11-02)

Rule: every Rust implementation is compared against these on the same machine. "eff GB/s" counts one read + one write of the array.
A 3-pass 3D FFT moves ~3× that, so ~300 "eff" GB/s ≈ 0.9 TB/s real ≈ memory roofline of this GPU.

## cuFFT 3D C2C f32, out-of-place, median of 10 (`cargo run --release -p hasty-bench --bin cufft_baseline`)
| shape | MiB | ms | eff GB/s |
|---|---|---|---|
| 256³ | 128 | 0.892 | 301.0 |
| 384³ | 432 | 3.232 | 280.3 |
| 512³ | 1024 | 7.663 | 280.3 |
| 768³ | 3456 | 28.258 | 256.5 |
| 1024³ | 8192 | 58.137 | 295.5 |

Realistic padded (2N) sizes for N = 320, 512×512×320, 512×512×256 (primary targets; 1024³ is the extreme):
| shape | MiB | ms | eff GB/s |
|---|---|---|---|
| 320³ | 250 | 1.761 | 297.7 |
| 400³ (σ=1.25 grid for N=320) | 488 | 3.656 | 280.1 |
| 640³ (N=320) | 2000 | 14.566 | 287.9 |
| 512×512×256 | 512 | 3.816 | 281.3 |
| 1024×1024×512 (N=512×512×256) | 4096 | 29.052 | 295.7 |
| 1024×1024×640 (N=512×512×320) | 5120 | 36.722 | 292.4 |

## TODO baselines
- VkFFT (current C++ path): 3D `performConvolution` + zero-pad at 2N for N ∈ {96,128,192,256}, batch 1/8/32.
- cuFINUFFT: type 1/2, 3D radial & spiral, 1e6–1e8 points, tol 1e-3/1e-6, ntransf 1/8/32.
- cuFFT: 1D/2D batched, non-pow2 (3,5,7-smooth) sizes.

## hasty-fft vs VkFFT vs cuFFT (2026-11-02, RTX 4090, out-of-place C2C forward, median of 10)
Reproduce: `cargo run --release -p hasty-bench --bin fft_bench -- [--f64] [1d|3d|all|specs...]`, `bench/cpp/build.sh && bench/cpp/vkfft_bench`.
Speed column = cuFFT time / other time (>1 = faster than cuFFT). Plain C2C is memory-bound; all three sit at the ~917 GB/s roofline for 1-D and ~300 "eff" GB/s (3 passes) for 3-D.

| f32 case | hasty ms | VkFFT ms | cuFFT ms | hasty × | VkFFT × |
|---|---|---|---|---|---|
| 1-D batched, 128 MiB, N=16…2048, 384, 640, 768, 1000, 1536 | 0.291–0.294 | 0.292–0.295 | 0.292–0.296 | 1.00–1.01 | 1.00–1.01 |
| 3-D 128³ | 0.042 | 0.037 | 0.036 | 0.85 | 0.98 |
| 3-D 192³ | 0.223 | 0.201 | 0.193 | 0.86 | 0.97 |
| 3-D 256³ | 0.842 | 0.820 | 0.891 | 1.06 | 1.09 |
| 3-D 320³ | 1.719 | 1.709 | 1.762 | 1.02 | 1.03 |
| 3-D 384³ | 3.053 | 3.042 | 3.238 | 1.06 | 1.06 |
| 3-D 400³ | 3.396 | 3.476 | 3.671 | 1.08 | 1.06 |
| 3-D 512³ | 7.177 | 7.259 | 7.631 | 1.06 | 1.05 |
| 3-D 640³ | 13.93 | 15.10 | 14.55 | 1.04 | 0.96 |
| 3-D 512×512×256 | 3.581 | 3.616 | 3.817 | 1.07 | 1.06 |
| 3-D 1024×1024×512 | 29.31 | 29.66 | 29.04 | 0.99 | 0.98 |

f64 (hasty vs cuFFT Z2Z; compute-bound on this GPU, FP64 is 1/64 rate): 1-D 1.0–1.3× faster than cuFFT, 3-D 1.11–1.31× faster (320³: 5.48 vs 6.09 ms; 640³: 51.1 vs 65.3 ms).
Single axis passes (f32): 920 GB/s for contiguous and tiled-strided N=512/640/1024 lines; the outermost (inner=524288) N=1024 pass 817 GB/s. Small L2-resident cubes (≤192³) are latency-bound where hasty is ~15% behind.

## Fused, pruned zero-padded convolution (Toeplitz-multiplication shape), f32, pad = 2N, 2026-11-02
Operation: `out += conjmult( crop( IFFT( K . FFT( pad( x . d1 ) ) ) ) / prod(pad), d2 )`. Rust: `conv_bench` (`ConvPlan`, 5 pruned passes, conv fused in the slowest-axis pass, no load/store kernels, scratch never cleared).
C++: `bench/cpp/conv_baseline` — (A) HastyCompute's path: load kernel + VkFFT `performConvolution` with zero-padding + store kernel; (B) naive cuFFT (memset, load, FFT, ×K, IFFT, store).

| N (pad 2N) | pad MiB | **hasty complex-K** | hasty real-K | hasty fp16-K | VkFFT path (A) | cuFFT naive (B) | hasty/A (complex) |
|---|---|---|---|---|---|---|---|
| 64³ | 16 | 0.038 | 0.037 | 0.037 | 0.046 | 0.096 | 1.21× |
| 128³ | 128 | 0.530 | 0.431 | 0.436 | 0.549 | 2.387 | 1.04× |
| 160³ | 250 | 1.293 | 1.152 | 1.152 | 1.278 | 4.869 | 0.99× |
| 192³ | 432 | 2.601 | 2.151 | 2.150 | 2.564 | 8.889 | 0.99× |
| 256³ | 1024 | 5.373 | 4.853 | 4.856 | 6.093 | 21.06 | 1.13× |
| 320³ | 2000 | 11.15 | 10.45 | 10.42 | 12.89 | 40.32 | 1.16× |
| 512×512×256 | 4096 | 21.56 | 19.96 | 20.35 | 24.57 | 80.86 | 1.14× |
| 512×512×320 | 5120 | 28.03 | 26.15 | 26.71 | 30.52 | 102.1 | 1.09× |

(ms; median of 10.) Per-pass at N=320: [1.12, 1.72, 5.08, 1.76, 1.42] ms = 935/911/826/893/922 GB/s — every pass is at the memory roofline except the kernel-multiply pass (826); total traffic ≈ 9.7 GB = 4.7 × (padded array).
Consequence: with this pass structure the only remaining lever is bytes — packed kernels (real/fp16: −6…−19% above), sharing the kernel across coils (amortises 2.1 GB of 9.7 GB per coil at N=320), fewer intermediate bytes.
VkFFT already prunes zeros, so the win over the existing path is ~10–16% (complex) / ~17–23% (real or fp16 kernel), and ~3.5× over a naive padded cuFFT pipeline.

## 3-D NUFFT: hasty-nufft vs cuFINUFFT v2.5.1 (Release build, `bench/cpp/build_cufinufft.sh`), f32, tol 1e-6, 128³ modes, ntransf 1, 2026-11-02
ms per transform (execute only, setpts separate: 0.2–1.6 ms both). Trajectories: `uniform` random in the cube, `radial` = 512-sample spokes through the centre. Reproduce: `nufft_bench`, `bench/cpp/nufft_baseline`.

| case | type 1 hasty (auto) | type 1 cuFINUFFT (default / SM) | type 2 hasty | type 2 cuFINUFFT (default / SM) |
|---|---|---|---|---|
| uniform, M=10⁶ | 13.8 | 9.3 / 9.3 | 2.49 | 2.30 / 1.95 |
| radial, M=10⁶ | 11.8 | 11.5 / 11.4 | 1.43 | 1.32 / 1.61 |
| uniform, M=10⁷ | **30.6** | 52.8 / 55.0 | 8.7 | 6.2 / 6.5 |
| radial, M=10⁷ | **37.9** | 73.6 / 73.7 | 4.5 | 3.2 / 3.6 |

Type 1 is 1.7–1.9× faster than cuFINUFFT at 10⁷ points (shared-memory tiles with sub-problem splitting), 10–30% slower at 10⁶ (atomics-bound in both); type 2 (plain global gather) is 10–40% slower than cuFINUFFT at large M — next steps: shared-memory tile gather, Horner kernel evaluation, vectorised row loads.

## Normal-operator families on the GPU, one coil, f32, N=128 (pad 256³), `toeplitz_bench` (random kernels/bases; timing only)
| operator | FFT pairs (fields) | kernels | kernel memory GB | ms / coil | × fused plain |
|---|---|---|---|---|---|
| plain, fused conv pass (`ConvPlan`) | 1 | 1 | 0.13 | 0.462 | 1.0 |
| plain through the multi-field engine | 1 | 1 | 0.13 | 1.035 | 2.2 |
| Weyl/commutator n=1 | 4 | 8 | 1.07 | 4.89 | 10.6 |
| Taylor–Hankel P=1 | 5 | 15 | 2.01 | 6.84 | 14.8 |
| Weyl n=2 | 10 | 36 | 4.83 | 14.9 | 32 |
| Taylor–Hankel P=2 | 15 | 70 | 9.40 | 26.6 | 58 |
| Tucker L=4 / L=6 | 4 / 6 | 10 / 21 | 1.3 / 2.8 | 5.3 / 8.8 | 11.5 / 19 |
| CP L'=8 / L'=12 | 8 / 12 | 8 / 12 | 1.1 / 1.6 | 8.9 / 13.4 | 19 / 29 |
**Fused multi-field Z pass (2026-11-02, `MixZPlan`)**: same rows after fusing forward-Z / mixing / inverse-Z into one shared-memory pass (ms/coil, unfused in parentheses): plain-through-engine 0.53 (1.03), Weyl n=1 3.12 (4.89), Taylor P=1 4.77 (6.84), Tucker L=4 3.58 (5.32), CP L'=8 4.98 (8.94), CP L'=12 7.48 (13.4) — 1.4–1.95×. Weyl n=2, Taylor P=2, Tucker L=6 do not fit / tile < 8 lines and keep the unfused numbers (14.9, 26.6, 8.85).

Before fusion: the multi-field engine (forward transforms, pointwise mixing, inverse transforms) cost ≈ 1.1–1.3 ms per field at N=128 vs 0.46 ms for the fused conv pass: fusing the mixing into the slowest-axis pass would roughly halve every row (measured upside 2.2×).
Memory is the binding constraint at real sizes: every field is a padded (2N)³ array (2 GB at N=320), so ≤ 8 fields and a handful of kernels fit on a 24 GB card.

## Chirp-aware NUFFT normal operator A^H A (v1: global-memory gather/scatter, one block per sample), f32, radial, `chirp_bench`
| N : K : C | ES tol / S | stencil taps | forward ms/coil | adjoint ms/coil | A^H A ms/coil |
|---|---|---|---|---|---|
| 96 : 10⁶ : 1 | 1e-4 / 4 | 2197 | 50.5 | 70.9 | 121 |
| 96 : 10⁶ : 8 | 1e-4 / 4 | 2197 | 12.6 | 41.3 | 53.9 |
| 128 : 10⁶ : 8 | 1e-4 / 4 | 2197 | 13.2 | 33.6 | 46.8 |
| 128 : 3·10⁶ : 8 | 1e-4 / 4 | 2197 | 37.1 | 98.3 | 135 |
| 128 : 10⁶ : 8 | 1e-3 / 3 | 1000 | 7.3 | 17.1 | 24.4 |
| 128 : 3·10⁶ : 8 | 1e-3 / 3 | 1000 | 20.3 | 49.8 | 70.2 |
| 128 : 10⁶ : 8, σ=1.25 | 1e-3 / 3 | 1728 | 8.6 | 26.1 | 34.8 |
Accuracy (N=14×12×10 test): forward/adjoint error 1.3e-3 relative where the concomitant effect itself is 0.42 (99.7% removed). At N=128 with K/N³ ≈ 0.5–1.4 the Toeplitz families above are 2–50× cheaper; the stencil route scales with K, Toeplitz does not (crossover discussion: `research/implementation_results.md`).

**Stored stencils + tiled scatter, sequence-trained tables (2026-11-02)**, N=128, K=10⁶, C=8, ES tol 1e-3 / S=3 (1000 taps), `chirp_bench --stored [--seqtrain]`:
| tables | L | row-repr err (max) | forward ms/coil | adjoint ms/coil | A^H A ms/coil | stencil store | setup |
|---|---|---|---|---|---|---|---|
| box-trained (|w|≤0.5) | 31 | 2.0e-3 | 6.93 | 11.28 | 18.2 | 4.0 GB, 0.02–0.03 s | 2.7 s |
| sequence-trained | 13 | 1.1e-3 | 6.93 | 11.28 | 18.2 | 4.0 GB, 0.02–0.03 s | 5.0 s |
(v1 on-the-fly: 7.3 / 17.1 / 24.4.) With stored stencils L only affects the one-off precompute. 18.2 ms = 39× the fused plain Toeplitz application (0.462 ms) at K/N³ = 0.5.

## Segmented physics NUFFT (`SegNufft`, f32, σ = 1.25, W = 10, autotuned tile shapes), `segnufft_bench` (2026-11-03)
ms per call, all coils. Details and history: `research/concomitant_segmented_first_order.md` §7.
| case | L / order | forward | adjoint |
|---|---|---|---|
| 180×180×28, M = 2.2e5, 12 coils (plain σ=2 W=6: 6.0 / 24.7) | 1 / 0 | 5.7 | 8.7 |
| | 8 / 0, 8 / 2 | 30.1, 31.7 | 31.2, 30.1 |
| | 10 / 2, 12 / 2 | 40.4, 49.2 | 35.3, 41.0 |
| 128³, M = 1e6, 8 coils (plain σ=2 W=6: 12.0 / 54) | 1 / 0 | 17.4 | 24.5 |
| | 8 / 0, 8 / 2 | 65.6, 67.5 | 53.9, 57.1 |
| | 10 / 2, 12 / 2 | 80.8, 103.8 | 68.9, 87.5 |
