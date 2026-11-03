# rustlib — plan: Rust FFT (VkFFT-rs) and NUFFT (cuFINUFFT-rs)

Status: planning only, nothing scaffolded yet. Scope of this document: the FFT and NUFFT
layers. Toeplitz / MRI operators / optimisation library come after and are only mentioned
where they constrain the design of these two layers.

Facts below marked **(verify)** are from memory of upstream projects, not checked against
source in this session. Facts about the current C++ were read from the repo.

---

## 1. Goals / non-goals

**Goals**
- CUDA-only (sm_89 on the dev box, RTX 4090, CUDA 13.0 at `/usr/local/cuda-13.0`) FFT and NUFFT in Rust,
  matching what HastyCompute actually uses today, then beating it by fusing things the C++
  stack cannot fuse (section 7).
- Core crates independent of torch. A separate bridge crate adapts `tch` tensors.
- Correctness oracle-driven: every kernel has a CPU f64 reference and golden data.
- Performance parity gate against current C++ (VkFFT + cuFINUFFT) *on the cases you actually run*
  before any new feature is declared done.

**Non-goals (for now)**
- Vulkan / OpenCL / HIP / Metal backends (VkFFT's raison d'être, irrelevant here).
- DCT/DST, quad/half FFT, Rader's algorithm, multi-GPU FFT of a single transform.
- Full CPU FINUFFT parity. A simple CPU reference is required as an oracle; a fast CPU backend is a later decision.

---

## 2. Inventory: what the C++ actually uses today

This defines Tier 1 (must-match). Everything else upstream supports is Tier 2+.

### FFT (VkFFT, `cpp/lib/src/fft/toeplitz_*.cpp`, `cpp/lib/src/wrapper/vkfft_mod.cppm`)
| Used | Detail |
|---|---|
| Backend | CUDA only (`VKFFT_BACKEND 1`), kernels JIT via NVRTC |
| Type / precision | C2C, **f32 only** in Toeplitz path |
| Dims | 1D, 2D, 3D |
| Sizes | `2N` per axis (zero-padded image size), so 128–1024 typical; N arbitrary → need non-pow2 |
| Zero-padding | `performZeropadding`: input occupies `[0,N)`, `[N,2N)` implicit zero |
| Convolution mode | `performConvolution` = FFT → ×kernel → IFFT in one plan (saves memory passes); `kernelConvolution` to pre-transform the kernel into VkFFT's private layout |
| LUT twiddles | `useLUT` in 2D path |
| Coalescing | `coalescedMemory` 64/128 tuned (see `kernel_*_coaleced.txt`) |
| Plan cache | `global_vkfft_cache[device]` keyed on a ~20-field struct |

Hand-written NVRTC kernels `toeplitz_load_{1,2,3}D.cu` do gather/scatter between the unpadded
tensor and the `(2N)^d` scratch, applying up to 4 elementwise multipliers (`MULT` / `MULT_CONJ`,
on input side and output side) plus optional accumulate. Per batch element the C++ launches
`VkFFTAppend` + one load kernel.

Warts to not reproduce: `static void* buffer_ptrs[1]` in the launch path (not thread-safe),
`cudaDeviceSynchronize` inside `transform_toeplitz_kernel`, scratch allocated inside the multiply
if not supplied, every launch on the default stream (`cuLaunchKernel(..., 0, 0, ...)`), PTX
(not cubin) loaded each process.

### NUFFT (cuFINUFFT v2.5.1 via `NufftPlan` in `cpp/lib/src/fft/nufft.cppm`; CPU finufft too)
| Used | Detail |
|---|---|
| Types | type 1 (`NTU`) and type 2 (`UTN`). Type 3 (`NTN`) is declared, **never used** |
| Dims | 1, 2, 3 |
| Precision | f32 mostly; f64 in tests and `double_prec` kernel build |
| Mode order | CMCL (default) and FFT |
| Tolerance | 1e-6 default f32; coordinate-warp code loosens it (1e-2) for PnP steps |
| Batch | `ntransf` |
| Upsampling | 2.0 default; enum has 1.25 / 1.0; "custom" throws |
| Methods | GM / GM-sort / SM / OD / block-gather enums mapped to `gpu_method` 1–4 |
| Spread/interp-only | used in an earlier warp implementation, since replaced by full type-2 |
| CPU path | `use_cpu_nufft` for Toeplitz kernel build to save GPU memory |
| Plan lifetime | `coordinate_warp` caches a plan + setpts and reuses it across many executes, because plan/setpts cost is significant |

Toeplitz kernel build (`toeplitz_creation.cpp`): mirrors TorchKbNufft — `2^(d-1)` spatially phase-shifted
type-1 NUFFTs on `N^d` modes (CMCL + weight modulation `exp(i·Σ N_d/2·ω_d)`), then
flip-and-concat / reflect-conj-concat into the `(2N)^d` kernel. Note this is `d`-fold redundant spreading of the same points.

---

## 3. Decisions to make (recommendation first)

| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | Port style for VkFFT | **Re-architect, don't transliterate.** Port the algorithms (Stockham radix stages, shared-memory exchange, 4-step, Bluestein, convolution fusion), not the string-macro structure | VkFFT's generator is ~50 headers of C string concatenation with global state; a typed IR in Rust is far easier to test, fuse and extend (the fusion hooks are the point of the exercise) |
| D2 | GPU kernel language | **CUDA C++ source generated/templated by Rust, compiled at runtime with NVRTC** to cubin for the exact arch, with an on-disk cache | Needs runtime specialisation (size, radix plan, w, dim, fused epilogues). VkFFT and your current code already work this way. Rust-CUDA (`rustc_codegen_nvvm`) is nightly-pinned and would make fusion/specialisation harder. Revisit later for static kernels (scan, sort) |
| D3 | CUDA bindings | `cudarc` (driver, nvrtc, cufft for the bootstrap) behind a thin `hasty-cuda` layer so it can be swapped | Avoids writing bindgen. **(verify)** CUDA 13 support in the cudarc version available; otherwise bindgen the driver/nvrtc headers ourselves — small surface |
| D4 | Torch coupling | Core crates take raw device pointers + stream (`DevicePtr<T>`, `Stream`). `hasty-torch` crate wraps `tch::Tensor`, zero-copy | Keeps FFT/NUFFT testable without libtorch; tch version must match libtorch exactly (you have torch 2.9.1/cu130 conda env) **(verify tch-rs version mapping)**. Getting torch's *current CUDA stream* from `tch` likely needs a tiny C++ shim; interim: coarse `Cuda::synchronize` at boundaries |
| D5 | NUFFT FFT dependency during bootstrap | Define `trait FftBackend`; start with cuFFT (via cudarc), swap to our FFT at milestone N4 | Decouples the two workstreams; NUFFT correctness can be proven without waiting for the FFT |
| D6 | Precision | f32 + f64 from the start in generic code (`trait Real`) | f64 needed for oracles and `double_prec` path; retrofitting is painful |
| D7 | Oracle data | Golden files generated once (Python in the existing venvs: numpy/scipy; `finufft` pip package in a scratch venv; plus current C++ via small dump programs), read by Rust tests | Avoids linking the C++ module build into Rust. Tests stay hermetic and reproducible |
| D8 | Licenses | Keep VkFFT (MIT) and FINUFFT (Apache-2.0) attribution + NOTICE for derived algorithms/tables (Horner coefficients etc.) **(verify both)** | Port is a derivative work |
| D9 | Toolchain | Update Rust (installed 1.85.0, Feb 2025). Pin via `rust-toolchain.toml` | Current stable; edition 2024 |

---

## 4. Workspace layout (all under `rustlib/`)

```
rustlib/
  Cargo.toml                workspace
  rust-toolchain.toml
  PLAN.md                   this file
  NOTICE / licenses/
  crates/
    hasty-cuda/             Context, Stream, Event, DeviceBuffer<T>, pinned host mem, stream-ordered pool,
                            NVRTC compile -> cubin, module/function cache (memory + disk), launch helper
    hasty-testkit/          CPU references (naive DFT/NDFT f64, rustfft-based big-N ref), error metrics,
                            .npy golden loaders, trajectory generators (radial, spiral, stack-of-stars, random,
                            clustered), bench harness + cuFFT/cuFINUFFT baselines
    hasty-fft/              plan IR, codegen, scheduler, launch; public: FftPlan<T>
    hasty-nufft/            kernel math, sort, spread/interp, deconv, plans; public: NufftPlan<T,D>
    hasty-torch/            tch bridge (later)
  tools/
    gen_golden/             scripts (python) that write golden .npy into rustlib/testdata/ (git-lfs or gitignored + regenerate)
    bench/                  benchmark binaries; results logged to rustlib/bench/baselines.md
```

Design rules that apply to every crate:
- Plans are immutable after build, `Send + Sync`; per-call state is passed into `execute(&self, stream, bufs)`.
- No hidden synchronisation. Everything enqueues on the caller's stream. Scratch comes from caller or stream-ordered pool.
- Errors via `Result`, no panics across the GPU boundary; every CUDA/NVRTC error carries the kernel name and the generated source hash.
- Generated source retained and dumpable (`HASTY_DUMP_KERNELS=dir`) — your VkFFT `keepShaderCode` equivalent.
- Kernel cache key = hash(generated source + arch + nvrtc version).

---

## 5. Track F — FFT ("VkFFT-rs")

### 5.1 Scope tiers
- **Tier 1** (matches today): C2C f32/f64, 1–3D, any size 2..~1024 per axis (pow2 + 3/5/7 smooth mandatory, other sizes via Bluestein), batched, zero-pad, convolution mode, fwd/inv, in-place and out-of-place.
- **Tier 2**: sizes beyond one shared-memory pass (4-step), R2C/C2R, `norm` options, strided/advanced layouts like cuFFT.
- **Tier 3** (new capability): fusion hooks (section 7), compressed kernel storage, pruned transforms.
- Explicitly deferred: Rader, DCT/DST, half precision compute (maybe half *storage*).

### 5.2 Architecture
1. **Plan IR**: a transform = list of *axis passes*. Each pass = (axis length, factorisation into radices, layout/stride info, prologue hook, epilogue hook, twiddle source). Scheduler picks single-kernel-per-axis (fits in shared memory) or 4-step.
2. **Radix library**: butterflies for 2,3,4,5,7,8,(11,13) emitted from generated code with constants derived in Rust at build time (no hand-typed constants). Stockham autosort, register-resident per thread, shared-memory exchange between radix stages.
3. **Strided axes**: coalescing by processing a tile of adjacent lines (the `coalescedMemory` knob you already tuned; expose as plan option, autotune later).
4. **Twiddles**: `sincospi` / LUT choice per precision; f32 accuracy study is part of F2 (error vs N must grow like ~log N, not N).
5. **Bluestein**: chirp-z via padded power-of-2 convolution, using our own pow2 pass.
6. **NVRTC pipeline**: codegen → hash → cache → compile → load. Compile latency matters (hundreds of ms per kernel): parallel compile of a plan's kernels, disk cache, warm-up API.
7. **Launch**: `FftPlan::execute(stream, in, out, scratch)`; descriptor carries pointers by value (kills the `static buffer_ptrs` problem).

### 5.3 Milestones (each ends with tests + a number)
| M | Content | Exit criteria |
|---|---|---|
| F0 | `hasty-cuda`, `hasty-testkit`, NVRTC→cubin→launch smoke test, baseline bench of cuFFT and current VkFFT Toeplitz path recorded | trivial kernel runs; baselines file committed |
| F1 | Single-axis C2C, pow2 ≤ 4096, radix 2/4/8, batched, in/out-of-place, fwd/inv | rel-L2 error vs f64 ref within ~2× cuFFT's error for all pow2 sizes; ≥ 70% of cuFFT throughput |
| F2 | Radix 3,5,7(,11,13); Bluestein; twiddle study | every N in 2..2048 correct; error curves plotted |
| F3 | 2D/3D via axis passes, strided coalescing, advanced layout | 3D sizes in your range (128–1024 per axis, incl. `2N` for N=96…384) ≥ cuFFT, ≥ 90% of VkFFT |
| F4 | **Fusion**: zero-pad/crop pruning, prologue/epilogue hooks, convolution mode (FFT·K·IFFT), accumulate-store | reproduces `toeplitz_multiplication` outputs from the C++ goldens (all mult/mult_conj combos, batch_mult, accumulate); ≥ 1.3× faster than current load-kernel + VkFFT path |
| F5 | 4-step for large N, R2C/C2R, f64 perf pass | as needed |

### 5.4 Tests
- Reference: naive O(N²) f64 DFT (small N), `rustfft` f64 for large N, cuFFT f32 for perf and for "error no worse than vendor".
- Properties: linearity, Parseval, shift theorem, impulse/constant inputs, inverse∘forward = N·id, conjugate symmetry for real input.
- Sweep: all N ≤ 2048; selected up to 8192; batch ∈ {1,2,7,64}; strides; 1–3D.
- Convolution mode vs explicit zero-pad + FFT + ×K + IFFT + crop.
- Nsight Compute / `scripts/nsys_profile.sh` for every perf claim; no LLM-guessed tuning without a measurement.

---

## 6. Track N — NUFFT ("cuFINUFFT-rs")

### 6.1 Scope tiers
- **Tier 1** (matches today): type 1 & 2, 1–3D, f32/f64, `ntransf` batching, CMCL + FFT mode order, ES kernel with width from `tol`, upsampfac 2.0, sign option, GM(+sort) and SM spreading, stream-aware plan/setpts/execute/execute_adjoint.
- **Tier 2**: spread-only / interp-only, upsampfac 1.25, type 3, exposed kernel params, kernel evaluation direct vs Horner.
- **Tier 3** (new): section 7.

### 6.2 Algorithm pieces to port (from FINUFFT v2.5.1 — pinned reference)
1. **ES kernel** `φ(z)=exp(β(√(1−z²)−1))`; `w` and `β` chosen from `tol`/upsampfac (w ≤ 16) **(verify constants)**; Horner piecewise-polynomial evaluation option.
2. **Fourier series of the kernel** (deconvolution factors) by quadrature at plan time.
3. **Fine grid sizing** (`nf = next smooth ≥ max(σ·N, 2w)`), uses our FFT sizes — ties to F2 (smooth sizes).
4. **Spread (type 1) / interp (type 2)**: GM nupts-driven; GM with bin-sort; SM with subproblem splitting and shared-memory atomics. Which of GM/SM/OD/block-gather are actually live in v2.5.1 for each type: **verify before committing**. Fold-and-wrap at periodic boundaries.
5. **Bin sort**: histogram + scan + scatter. cuFINUFFT leans on Thrust/CUB here; we need own scan (decoupled look-back or simple 3-pass) and deterministic option.
6. **Deconvolve + mode copy** between fine grid and `N^d` modes (CMCL/FFT order).
7. **Batching**: `ntransf` with `maxbatchsize`, stream-overlapped.

### 6.3 Milestones
| M | Content | Exit criteria |
|---|---|---|
| N0 | Kernel math crate module (ES, β/w tables, Fourier series, Horner fit) + CPU reference NDFT f64 + simple CPU NUFFT | matches golden values from `finufft` (python) to ~1e-14 for kernel params; CPU NUFFT meets `tol` vs NDFT |
| N1 | GPU type 2 (GM), deconv, `FftBackend` = cuFFT, 1/2/3D f32 | err ≤ tol vs NDFT for tol ∈ {1e-2…1e-6}, all trajectory generators |
| N2 | GPU type 1 (GM, GM-sort, SM), bin-sort, atomics, adjointness test | `⟨Ax,y⟩ = ⟨x,A^H y⟩` to f32 eps; same accuracy as cuFINUFFT |
| N3 | f64, batching, mode orders, plan/setpts reuse, stream semantics, upsampfac 1.25 | parity run vs C++ `NufftPlan` on coordinate-warp and nufft_dft tests |
| N4 | Replace cuFFT with `hasty-fft`, fuse deconv/zero-pad into FFT prologue/epilogue | no extra full-grid passes; perf ≥ N3 |
| N5 | Spread-only/interp-only, type 3 if needed | as needed |
| N6 | Tier 3 features | section 7 |

### 6.4 Performance reality check
MRI trajectories (radial / spiral / stack-of-stars) are *highly* non-uniform: dense near the k-space centre means
heavy atomic contention in naive GM spreading. SM + sorting is what makes cuFINUFFT fast there. The test set must
include clustered points from day one, otherwise a uniform-random benchmark will mislead us. Expect N2 to be the
hardest milestone; N1 (interp) is comparatively easy because it has no atomics.

### 6.5 Tests
- Reference: direct NDFT f64 (small problems), `finufft` CPU goldens (medium), current C++ cuFINUFFT outputs (regression).
- Accuracy vs `tol` curve across (dim, type, precision, upsampfac, trajectory).
- Adjointness, linearity, translation property, zero/delta inputs.
- Timing: points ∈ {1e5…1e8}, grid ∈ {64³…512³}, `ntransf` ∈ {1, 8, 32}; report vs cuFINUFFT.

---

## 7. Additions for compressed sensing / concomitant fields (design hooks now, build later)

These drive API shape in F4 and N6, so they matter *now* even though they are not first.

1. **Fused prologue/epilogue hooks in the FFT** (F4). Replace `toeplitz_load_*D.cu` entirely:
   `diag_in(voxel) → FFT → ×kernel → IFFT → diag_out(voxel) → accumulate`. Diagonals come from
   buffers, scalars, or a generated expression of the voxel index (e.g. `exp(-iΦ_DC(r))` from the
   DC/AC split computed in registers instead of read from a stored tensor). Keep the expression
   surface restricted (elementwise, captured buffers/scalars) to avoid scope creep.
2. **Pruned zero-padded transforms**. Today `(2N)^d` is processed in full. Input has only `N^d`
   non-zeros and output only `N^d` is kept, so first-pass line count and last-pass line count drop; roughly
   40% fewer FFT line-passes in 3D on each side.
3. **Batched multi-term Toeplitz** matching the normal-operator structure in
   `normal_op_coil_factorization` (coil sum outermost, per-term `A_l`, `kernel_l`, `B_l` coil-free). API should
   take a list of terms sharing the same grid, with one launch sequence and accumulate in the epilogue, instead of
   `P×R` independent calls. Exact sum structure to be pinned down with you before the API freezes (open Q).
4. **Kernel storage is the real bandwidth bottleneck.** Each `kernel_l` is `(2N)^3` complex f32 (1 GB at N=256) and
   is streamed per application. Make the convolution hook read a *packed* format (Hermitian half `[2nx,2ny,nz+1]`,
   optionally f16/bf16 or block-scaled) and expand in registers. Our own layout, not VkFFT's private one.
5. **Multi-RHS NUFFT with shared kernel evaluation.** For `C` coils × `L` terms, evaluate the ES kernel weights and
   bin sort once per nonuniform point and apply to all `ntransf` channels (register blocking), instead of
   re-evaluating per transform.
6. **Fused weighted term-sum on the nonuniform side.** Type 2: `y_j = Σ_l w_l(t_j)·interp_l(k_j)` accumulated in registers
   (never materialising `L` outputs). Type 1 adjoint: spread `conj(w_l(t_j))·y_j` into `L` grids sharing sort and kernel weights.
   This is the time-segmented / low-rank off-resonance structure from `notes/MRI_Physics.tex`.
7. **Plan reuse**. Cheap `setpts` and a plan that can retarget point sets (warp code reuses plans because rebuild is costly).
8. **On-the-fly coordinates** (later): compute `k_j` from gradient waveforms (incl. concomitant/GNL warps) inside the
   spread/interp kernel instead of storing `[3,K]` per term.
9. **Explicit adjoints, not autograd**: every op exposes `apply` + `adjoint` (a `LinearOp`-style trait) for use by the
   later optimisation library; gradient w.r.t. trajectory (kernel derivative) is a Tier-3 item.
10. **Async / overlap**: stream-ordered API so kernel streaming from host/file (`CachedTensor`) overlaps with compute;
    pinned staging buffers provided by `hasty-cuda`.

---

## 8. Cross-cutting

- **Baselines first (F0).** Before writing FFT code: run current C++ and record (a) 3D Toeplitz multiply at N ∈ {96,128,192,256}, batched 1/8/32,
  (b) cuFINUFFT type 1/2, 3D radial & spiral, 1e6–1e8 points, tol 1e-3/1e-6, `ntransf` 1/8/32, (c) cuFFT 3D C2C for the same `2N`. Store in `rustlib/bench/baselines.md`. Everything later is measured against this. **Need your typical sizes** (open Q).
- **Golden data**: generated by `rustlib/tools/gen_golden/` and the existing C++ tests via small dump programs; stored as `.npy`; large files git-ignored with a regenerate script.
- **LLM-authored numerics discipline**: every kernel lands with (1) reference test, (2) error-vs-size sweep, (3) Nsight number for perf claims. Property tests catch the "plausible but wrong" class of bug. Review generated CUDA source by dumping it; don't trust only the Rust side.
- **CI**: GPU tests behind a feature flag (`--features gpu-tests`); CPU-only tests (kernel math, reference DFT, scheduler logic, codegen snapshot tests) run anywhere.
- **Dev environment**: `nvcc` is not on `PATH` (CUDA 13.0 lives at `/usr/local/cuda-13.0`); NVRTC route doesn't need nvcc, but profiling (`ncu`/`nsys`) and `CUDA_PATH` must be configured.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Matching VkFFT/cuFFT performance is months of tuning | Parity gate only on *your* size set; use cuFFT as fallback `FftBackend` while FFT matures |
| NVRTC compile latency makes small/one-off transforms slow | disk cubin cache, parallel compile, plan cache keyed on structure |
| f32 twiddle accuracy at large N | F2 study; `sincospi` or LUT with f64 generation |
| Atomic contention in spreading on MRI trajectories | clustered test sets from N0; SM/sort; consider warp-aggregated atomics |
| `tch` / libtorch version & ABI coupling, stream sharing | core crates torch-free; bridge crate isolated; coarse sync first |
| cudarc lagging CUDA 13 | thin abstraction; fallback bindgen |
| Scope creep from section 7 | Tier gating; nothing from section 7 starts before its prerequisite parity milestone passes |
| Licence/attribution | NOTICE file from day one |

---

## 10. Suggested order of work

1. **F0**: toolchain update, workspace scaffold, `hasty-cuda`, `hasty-testkit`, NVRTC smoke test, baselines recorded.
2. **N0** in parallel (pure CPU, no GPU risk): kernel math + CPU references + golden generation.
3. **F1 → F2 → F3** and **N1 → N2 → N3** as two independent streams (NUFFT uses cuFFT until N4).
4. **F4** (fusion) — first real payoff; validate against the C++ `toeplitz_multiplication` goldens.
5. **N4**, then Toeplitz crate on top (`create_toeplitz_kernel`, `toeplitz_multiplication`), then section 7 items in order of measured benefit.

## 11. Answers received (2026-11-02) and their consequences

1. **Kernel language**: CUDA C++ generated by Rust + NVRTC. Confirmed (D2).
2. **Problem sizes**: highly variable; typical N = 320³, 512×512×320, 512×512×256 (padded 640³ / 1024×1024×640 / 1024×1024×512), 512³ is the extreme; single local GPU up to multi-node multi-GPU clusters; images up to 512³ ⇒ `1024³` padded (8.6 GB per complex-f32 field).
   Consequences: 64-bit indexing everywhere; stream-ordered memory pool; **distributed FFT (slab/pencil) is a planned first-class track (F6)** and the plan
   interface carries global/local shape + process grid from the start; NUFFT needs slab ownership + halo exchange (N7). No assumption that a field fits in one GPU.
3. **Normal-operator sum structure**: unknown (research phase) ⇒ **no Toeplitz/MRI specialisation in these crates.**
4. **Generality over specialisation; performance is the main metric.** Fusion is provided only as *generic* hooks (load/store functors, convolution mode, user stencil
   generators) that specialisations plug into. Section 7 below is therefore a list of *consumers* of those hooks, not features of the core.
5. **Process**: start by implementing the libraries with tests and benchmarks against VkFFT / cuFFT / cuFINUFFT; judge parity on measured numbers before adding anything else.

Still open: f64 priority (assume f32+f64 generic), CPU backend (assume reference-only), license, toolchain update permission, and the multi-node transport
choice (NCCL vs CUDA-aware MPI vs NVSHMEM) which only matters at F6.

## 12. Revised milestone additions
- **F6** distributed FFT (pencil/slab, comm/compute overlap), after F3.
- **N7** multi-GPU NUFFT (grid slab ownership + halo).
- NUFFT kernel evaluation is a pluggable device function (needed by research note Sec. 3.1) — design this into N1/N2 rather than retrofitting.

## 13. Implementation status (2026-11-02)
- `hasty-cuda`: device, NVRTC→cubin with disk cache (`HASTY_KERNEL_CACHE`, `HASTY_DUMP_KERNELS`), interleaved complex types, `Real` trait (f32/f64). Smoke tests pass.
- `hasty-fft`: runtime-generated shared-memory Stockham kernels. Radices 2,3,4,5,7,8,11,13 (any n with those prime factors, up to the shared-memory limit — 2048+ f32 verified), forward/inverse, normalisation,
  f32/f64, contiguous and strided axes (tile of adjacent lines for coalescing), in-place and out-of-place, ND via one pass per axis. Correctness vs f64 rustfft: all supported n ≤ 600 at 1.2e-7 worst f32 relative error; ND/strided/f64 tests pass.
  Single-pass in-place variant when all radices divide the largest (halves shared memory → wider tiles).
- Performance: matches cuFFT/VkFFT at the memory roofline for 1-D and for 3-D ≥ 256³ (0.99–1.08×), 10–30% faster in f64; 15% behind on L2-resident small cubes. Details: `bench/baselines.md`.
- **F4 done**: `ConvPlan` — pruned zero-pad/crop (zero rows never read/written, scratch never cleared), forward→×K→inverse fused in the slowest-axis pass, up to 4 `pre` and 4 `post` diagonal operands (multiply / conj-multiply), accumulate, normalisation, kernel formats `Complex | Real | ComplexHalf | ComplexBf16` with runtime `kscale`; ranks 1–3, f32/f64. Tests vs rustfft reference (all ranks, ops, accumulate, garbage scratch, Toeplitz-matvec equivalence, packed kernels). Perf: 1.0–1.16× the existing VkFFT path (1.17–1.23× with real/fp16 kernel), ~3.5× a naive cuFFT pipeline; all passes at the memory roofline (`bench/baselines.md`).
- Not yet: Bluestein/large prime n, n beyond one-pass shared memory (4-step), R2C/C2R, multi-coil batching in the conv pass (share K across coils through L2), coordinate-monomial `pre/post` generated from voxel index, autotuning (tile width, radix order, small-size latency), distributed (F6).
- Next: NUFFT track N0 (CPU reference + kernel math) and the research prototypes in `research/` that decide which hooks matter.

## 14. Implementation status — round 3 (2026-11-02)
- `hasty-nufft`: 1–3 D type 1/2 NUFFT (ES, bin-sort, GM + SM spreading, multi-coil, FFT-order output) validated vs NDFT; 1.7–1.9× faster than Release cuFINUFFT for type 1 at 10⁷ points, 10–40% slower type 2; chirp-aware NUFFT (modulated rows) validated.
- `hasty-toeplitz`: kernel builder + multi-field mixing engine + plain / Taylor-Hankel / commutator-Weyl / Tucker / CP expansions, each validated against CPU series.
- `hasty-fft`: added polynomial-in-coordinates diagonals, runtime term tables, `PadFft3`.
- `hasty-bench`: `nufft_bench`, `toeplitz_bench`, `chirp_bench`, `recon_sched`; C++ baselines for VkFFT, cuFFT conv, cuFINUFFT (Release).
- Results and decisions: `../research/implementation_results.md`. Open: fused multi-field pass, tiled chirp gather/scatter, P²nP integration, large-N kernel streaming, distributed FFT (F6), multi-GPU NUFFT (N7).

## 15. Implementation status — round 4 (2026-11-02)
- `hasty-fft::MixZPlan` + `hasty-toeplitz::MixOp`: fused multi-field Z pass (forward-Z / k-space mixing / inverse-Z in shared memory per connected field group), automatic fallback to the unfused path if the group does not fit or tiles are < 8 lines; 1.4–1.95× on every fusable family.
- `hasty-nufft::ChirpNufft`: least-squares FOV stencils, stored fp16 stencils (`precompute_stencils`) with tiled shared-memory scatter (adjoint 1.5×), sequence-trained POD tables (L 31→13) with `ChirpTables::save/load`. A^H A = 18.2 ms/coil at N=128, K=10⁶.
- `recon_sched` rewritten to the P²nP setting: no density compensation, noise-whitened multi-coil model, fixed F-1/F-Cheb and dynamic ZMSHR1 preconditioners, lagged-correction gradient with adaptive anchor. Findings: ZMSHR1 ≥ F-Cheb at half the operator applications; cheap-only gradients diverge under the dynamic preconditioner; lagged plain surrogate + adaptive anchor matches exact error at 4.5–6.7× lower cost (`../research/implementation_results.md` §5).
- Open: GPU-resident solver loop with ZMSHR1 and the anchor rule; real denoiser + N ≥ 64 with realistic coil maps; forward-gather tiling for the chirp NUFFT; type-2 plain NUFFT tile gather; scan-independent precompute of concomitant/GNL tables; kernel streaming for N=320.

## 16. Synthetic 4D-flow pipeline — slice M0 (2026-11-02)
Design and status: `../research/synthetic_4dflow_pipeline.md` (§7). New crates:
- `hasty-mrsim` — gradient moments, trapezoid/bipolar design, Lustig time-optimal readouts, spiral 4D-flow sequence,
  ECG, synthetic GNL, GPU direct-sum truth simulator.
- `hasty-phantom` — subject loading (from `tools/synth/prep_subject.py`), tissue model, χ→B0 + shim, Biot–Savart coils +
  noise covariance, slab prescription, `FlowField` trait (CFD will implement it; `eval_lagrangian` for analytic
  acceleration) with the centerline Womersley model, simulator sources. Time enters only through per-time source sets,
  so wall motion / respiration (5D) are additions, not rewrites.
- `hasty-recon` — loader, whitening, coil compression, SENSE operator with a generic CUR low-rank phase model (B0 +
  concomitant), GNL via apparent-coordinate reconstruction + inverse warp + J⁻¹, CG.
- binaries: `sim4d`, `recon_m0`, `flowcheck`, `blooddiag`.
Python side (`rustlib/.venv`): torch 2.13 (matches tch 0.26), TotalSegmentator, kimimaro. Data under `../data/hasty/` (git-ignored; kernel cache in `target/hasty-kernels`).

## 17. Segmented physics NUFFT (2026-11-03)
`hasty-nufft::SegNufft`: multi-coil encoding operator with L time segments (B0 + τ-dependent concomitant in the
segment diagonals), per-sample linear/constant concomitant (k-shift, phase) and first/second-order ΔC correction in the
interpolation kernel (no extra FFTs). Channel-interleaved grid + partial-axis FFT, exact tile→sample lists, shared
per-(sample, tile point) weights, atomic-free adjoint, planner bounded by 48 KB static shared memory + `autotune()`.
L=8 order 2 at 180×180×28, 2.2e5 samples, 12 coils: 31.7 / 30.1 ms (first version 187 / 903). Accuracy 8e-5 (L=8, o2).
Status, numbers, dead ends and the next levers: `../research/concomitant_segmented_first_order.md` §7.
- Next: temporal interpolation over frames (channels = segments × frame slots), accuracy tests vs direct sums, P²nP
  integration; multi-channel pruned FFT with fused pad/crop (≈ half of the call is FFT/pad/crop).
