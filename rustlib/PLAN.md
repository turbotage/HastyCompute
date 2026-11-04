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

## 18. Operator validation and Bluestein (2026-11-03)
- `hasty-fft`: Bluestein (chirp-z) core for axis lengths with a prime factor > 13 (`bluestein.rs`): same load/store
  semantics as the mixed-radix kernel (strides, pruning, pre/post/polynomial diagonals, scale, accumulate, fused
  convolution), core = FFT_m · B̂ · IFFT_m in shared memory, m ≥ 2n − 1 smooth. f32 1.2–1.9e-7, f64 ~1e-15 (tests
  `tests/bluestein.rs`, `conv_bluestein_lengths`). The fused multi-field Z pass declines such lengths (unfused path).
- Operator validation on RACLETTE n001 (`tools/raclette/export_case.py` → `simimg` → `opval`): dense model-exact data
  with B0 + concomitant + GNL; full model 4.3e-3 image error (CG-limited), velocity RMSE 2.4 mm/s, bias 0.0000.

## 19. Learned prior: equivariant 7-point denoiser (2026-11-03)
Design: `../research/denoiser_design.md` §10; training data: `../research/cfd_training_data.md`.

**Data**
- `tools/raclette/build_training.py` streams RACLETTE into `../data/hasty/train/raclette`: download, fp16 cases, delete
  the raw files.
- `index.json` splits by background patient.
- `tools/raclette/train_figs.py` writes QA montages.

**Python** (`tools/denoiser`, plain PyTorch, no equivariance library):
- `oh.py`: O_h / O equivariant kernels for monomial field types. Orbits are computed once; each layer is a signed gather
  + conv3d / conv_transpose3d / einsum, and `freeze()` makes the weights constants.
- `model.py`: 7p lift to reference-relative features, then a bias-free regular-field U-Net with temporal mixing and
  multiplicative FiLM on (VENC_lo, VENC_hi, h), then the inverse lift. `denoise_volume` evaluates the whole volume or
  exact overlap tiles.
- `dataset.py`: encode-aligned on-the-fly B-spline patches (RACLETTE encodes are about 20° oblique to its grid). The
  `Patches` stage is device-agnostic, meant to run on the GPU.
- `train.py`: run dirs with metrics, checkpoints and figures.
- `export.py`: TorchScript; `torch.export` .pt2 with dynamic shapes; AOTInductor package; reference volume for Rust.
- `_env.py`: torch / inductor / triton caches and TMPDIR go to `target/`.

**Tests** (`test_model.py`, `test_dataset.py`, CPU)
- Equivariance and exactness, all at machine precision:
  - kernel constraint for every g ∈ O_h and O;
  - O_h equivariance in raw 7p space (incl. the flip conjugation);
  - positive homogeneity D(µx) = µD(x);
  - cyclic time shift;
  - untrained model = identity;
  - tiled = whole volume.
- Export paths:
  - TorchScript traced at one shape and run at another: exact;
  - `torch.compile` (CPU inductor): exact;
  - torch.export .pt2 at a new shape: exact vs TorchScript.
- 3-level U-Net: structural receptive field ±26 voxels / ±3 frames.

**Rust**
- `hasty-denoise` (workspace, torch-free) converts layouts: the x-fastest stack is read as C-order [Z, Y, X], which is
  exact thanks to O_h equivariance; encode pairs are reordered, axes zero-padded to ≡ 1 mod 2^(levels−1), and the
  exp(−ikv) convention of hasty-mrsim is mapped by the exact involution x_e → conj(x_e) x_ref/conj(x_ref).
- `hasty-denoise-torch` is a separate cargo workspace: a C ABI shim (`csrc/shim.cpp`, C++20) over libtorch's C++ API,
  compiled against the venv's libtorch.
  - It loads TorchScript or an AOTInductor package. Both reproduce Python on the reference volume with rel. error
    9e-7, including the −1 phase-convention path.
  - tch-rs 0.26 is unusable: it only supports libtorch 2.13, its generated shim does not compile against 2.14 (the
    venv's torch, pinned by TotalSegmentator / nnU-Net / torchvision), and it only loads TorchScript, which is
    deprecated upstream.
  - Build: `CARGO_HOME=$PWD/target/cargo-home CARGO_TARGET_DIR=$PWD/target/torch TMPDIR=$PWD/target/tmp cargo test --release --manifest-path crates/hasty-denoise-torch/Cargo.toml`

**hasty-mrsim:** `Encoding::DualVenc { venc_hi }` (7 encodes in the denoiser layout) and `sim4d --venc-hi`.

**Not done yet**
- AOTI CUDA package (export on CUDA, bf16);
- running the denoiser on the P²nP CUDA stream with device pointers instead of the host round trip.
- Done since: training (§24), `recon_p2np` with any encoding and the learned prior (§24).

## 20. Services, IO, visualisation: transition plan for the C++ server stack (2026-11-03)

### Inventory of the C++ / Python / web side

| C++ module | lines | what it does |
|---|---|---|
| `generic_value` | ~1700 | value tree None / Tensor / Vector / Dict / Tuple / String; chunked binary wire format; slice-path language (`["key"][0][1:5, :, ...]`); HDF5 (gv_type attributes) and blosc2 file formats |
| `server` | ~1400 | `hasty.HastyService` (`protos/hasty_service.proto`) |
| `io` | ~600 | NIfTI read/write, blosc2 streams, HDF5 → JSON, default directories |
| `python` | ~270 | runs venv scripts as subprocesses |
| `viz` | ~470 | plotlypp line/heatmap figures; `orthoslicer(volume)` |
| `threading` | ~1100 | thread pool, dependency-graph pool, storage pool, thread streams, CUDA load balancer |
| `tensor` | 14 files | libtorch tensor wrapper, comprep (ui16 quantisation for transfer), disk-cached tensors, masking |
| `plotting_website` | ~3700 JS | orthoslicer (WebGPU/WGSL, lazy remote volume with ring-buffer slice cache); node editor (litegraph graph of commands); hand-written grpc-web client (Push, Read, ExecuteCommand, BankQuery, Delete, metadata) |
| `python/lib/hastycompute` | | grpcio client; GenericValue (de)serialiser |

Details of the `server` module:
- **RPCs:** Push / Read / Write are chunked streams; Read and Write accept a slice string. Also BankQuery, ExecuteCommand (int id + options string + input UUIDs → output UUIDs), Delete, and metadata read / write / delete.
- **Value bank:** UUID → (value, metadata string, per-key hooks).
- **Command registry:** int ids; base arithmetic / creation / reduction / FFT and ui16 compress commands.
- **HTTP side:** cpp-httplib HTTPS server serving the website, plus a hand-rolled grpc-web proxy over a loopback channel.

Details of the `python` module:
- It pre-registers output UUID slots and a log "mailbox" value; the script writes back over gRPC.
- NIfTI registration.

Details of `orthoslicer(volume)`:
- It pushes the volume to the bank, writes metadata, and opens `https://localhost:<port>/orthoslicer/index.html?uuid=…`.

### Rust targets (CPU-only work: can proceed while the GPU is off limits)

1. **`hasty-value`.** `enum Value { None, Tensor, List, Dict, Tuple, Str }`.
   - `Tensor` = dtype (torch ScalarType codes) + shape + host bytes or a device buffer.
   - Serialisation is byte-compatible with the C++/Python GenericValue wire format, and the slice-path parser is ported.
   - comprep ui16 quantisation lives here too.
   - Tested against `python/lib/hastycompute/generic_value.py`.
2. **`hasty-io`.**
   - One N-D npy reader/writer (C/F order, complex) replacing the ad-hoc readers in hasty-phantom / hasty-recon.
   - NIfTI.
   - Zarr v3 via `zarrs` for our own chunked, compressed volumes. It supports partial reads (what the viewer's slice fetching wants), is readable from Python, and replaces blosc `.gv` / HDF5 dumps.
   - Content-addressed disk cache (CachedTensor).
   - HDF5 (`hdf5-metno`) only behind a feature: ISMRMRD raw data and legacy gv files.
   - Later: twix / ISMRMRD readers for scanner data.
3. **`hasty-server`.**
   - tonic + prost on the *same proto*, so the existing website and Python client work unchanged.
   - axum on the same port: static files (`tower-http` ServeDir) and `tonic-web` for unary/server-streaming. tonic-web cannot do client streaming, so a small axum route buffers grpc-web frames for Push / Write, which the website uses, as the C++ proxy did.
   - Bank = `RwLock<HashMap<Uuid, Entry>>` plus a `broadcast` event channel. The channel replaces per-key callbacks; the script log mailbox becomes a subscriber.
   - Registry keyed by name, with stable legacy ids served by `get_available_commands`.
   - Commands run on a tokio blocking pool (CPU) or on one worker thread per GPU that owns its CUDA context and streams (actor + job queue). This replaces the thread pools and the load balancer.
   - Additive proto extensions: job submit / status and a server-streaming `Watch` (bank events, recon progress).
   - `tracing` logs to `data/hasty/logs`.
   - TLS optional (rustls). `http://localhost` is already a secure context for WebGPU; reach a remote GPU box through an SSH tunnel.
4. **Scripts / Python.**
   - Same subprocess pattern: server address and output UUIDs passed in the environment; logs via the event channel.
   - Later a pyo3 module, so research scripts call the Rust operators in-process (numpy zero-copy).
5. **Viz.**
   - `viz::show(value, meta)` → bank + viewer URL, from any binary; recon runs can publish iterates live.
   - matplotlib run-dir figures stay for reports.
   - The website gains a run browser (metrics, figures) and recon/denoiser nodes once the GPU commands exist.

### Order
1. hasty-value + hasty-io npy/NIfTI, with cross-language round-trip tests.
2. hasty-server serving the unchanged website and Python client, with base commands on the host; e2e tests with the Python client.
3. Zarr volumes and the viewer on runs.
4. GPU commands (FFT, NUFFT, P²nP, denoiser) through GPU workers.
5. pyo3; scanner-data readers.

## 21. Values, server and LAN viewer: implemented (2026-11-03)

**`hasty-value`**
- Value tree: None / Bool / Int / Float / Str / Tensor / List / Tuple / Dict.
- Streaming, block-compressed format v1 (`crates/hasty-value/FORMAT.md`):
  - independent blocks, each with its own header (codec, filter, word size, raw length, compressed length, optional CRC32);
  - filters: byte shuffle (complex values by component) and integer delta;
  - codecs: stored, LZ4, zstd; raw fallback for blocks that don't compress;
  - parallel encoding in bounded groups;
  - an incremental decoder that decodes complete blocks as soon as they arrive (in parallel when several are ready).
- Also: numpy-style slice paths with assignment, u16 display quantisation, N-D npy (C or Fortran order).
- `hvtool` (info / digest / convert / gen).
- Sizes: smooth complex64 volume, shuffle + LZ4 63 % of raw, zstd 60 %; u16 display volume, delta + shuffle + LZ4 26 %.

**Cross-language implementations**, verified against Rust files by digest, both directions:
- Python `python/lib/hastycompute/value_format.py`, all codecs;
- JS `plotting_website/common/value_format.js`, stored and LZ4 (zstd is rejected in the browser).

**`hasty-server`** (proto `hasty.v2`, `crates/hasty-server/proto/hasty.proto`; codegen via protox, no protoc)
- One TLS port serves native gRPC (HTTP/2), gRPC-web (tonic-web; browsers use `PutUnary`), the website (`ServeDir`) and `/hasty-ca.pem`.
- Bank with copy-on-write slice writes and broadcast events; `Watch` streams them to clients.
- Name-keyed command registry on the blocking pool: summary, slice, quantize_u16, stats, abs, real, imag, angle, neg, conj, add, sub, mul, div, zeros, ones, rand, fft, ifft (rustfft).
- TLS: a persistent local CA in `data/hasty/certs` (created once; import `ca.pem` once per viewing machine) signs a fresh server certificate each start for localhost, the host name and every interface IP.
- Binaries:
  - `hasty-serve [--bind 0.0.0.0:8443] [--no-tls]`;
  - `hasty-show <file.npy|.hv> [--slice …]`.
- `viz::show(name, value)` / `viz::volume(name, dims, data)` from any binary: publishes to `HASTY_SERVER`, or to a server already running locally, or starts an embedded LAN server. `viz::wait()` keeps it alive.
- Async Rust `client::Client`.

**Clients and website**
- Python `hastycompute.v2` (`Client`, `show`; stubs via `v2/generate.sh`).
- Browser `plotting_website/common/hasty_client.js` (minimal protobuf + gRPC-web over fetch, incremental value decoding).
- The orthoslicer is ported:
  - `?id=` (legacy `?uuid=`), `&path=` for a sub-value, `&mode=` for complex data;
  - u16 display copy plus `stats` for the window;
  - LZ4 + delta slice fetches;
  - follows `show` events, reloads on updates.
- The landing page lists the bank live and links the CA.

**Tests**
- `cargo test -p hasty-value -p hasty-server`.
- `python -I python/tests/value_format/test_value_format.py`.
- `node plotting_website/common/test_value_format.js`.
- With a running `hasty-serve`:
  - `node plotting_website/common/test_client.js https://<lan-ip>:<port>` (`NODE_EXTRA_CA_CERTS=ca.pem`);
  - `python -I python/tests/client/test_client_v2.py`.
- `test_orthoslicer_flow.js` replays the viewer's calls. On RACLETTE n001 it fetches a display slice in 7 ms with a dequantisation error of half a step.

**Pending**
- The node editor still speaks the old C++ protocol and value layout (its NIfTI upload and raw parsing need porting to the v2 client).
- GPU commands via per-device workers.
- `recon_p2np --viz` (iterates live).
- Zarr volumes.
- Build note: new crates were fetched into `CARGO_HOME=rustlib/target/cargo-home`.

## 22. comprep, node editor, compressed training store (2026-11-04)

**comprep** (`hasty_value::quant`): exact port of the C++ `tensor_comprep`, plus the inverse of the orthoslicer
shader.
- Transfer: piecewise `[clampa, a] → [0, t0]` (linear / gamma / log tail), `[a, b] → [t0, t1]` linear,
  `[b, clampb] → [t1, 1]` tail; `t0 = (1 − focus)/2`.
- Auto config as in C++: p1 / p99 window, [min, max] clamp, ≤ 16-bit integers mapped directly.
- Server commands `comprep_u16` (full or partial config; the message returns the full one) and `decomprep_u16`.
- The orthoslicer passes `?config=` and `?prepped=1`.
- The earlier linear `quantize_u16` and the viewer's use of it were wrong (the window was decoded as the full range),
  and are replaced.

**Node editor:** ported to v2 (`node_editor/src/hasty.js`; ids are hex strings; the old gRPC client is deleted).
- FFT / IFFT nodes now work (numpy-style `norm`, axes).
- NIfTI load / save, uint16 and complex64 included.
- Compress UI16 → `comprep_u16`.
- Bank meta / clear go through List / Delete.
- `?selftest=1` runs a test graph. `tools/web/headless.py` drives headless Chrome over the DevTools pipe (software
  rendering) and reports it: whole graph ok, FFT/IFFT round trip 1.2e-7.
- Headless WebGPU canvases fail even with SwiftShader, so orthoslicer rendering can only be checked in a real browser.
  Its data path (comprep, stats, slice streaming) passes headless.

**Chunked tensors** (format tag 9; `FORMAT.md`): N-D chunks with an index.
- `encode_with(value, o, plan)`.
- Random access: `LazyFile::read_box` in Rust, `open_lazy()[...]` in Python. The Python fast path calls the Rust
  reader through `crates/hasty-value-capi` (ctypes, one thread pool per process); a pure-Python fallback remains.
- A training patch box (50×50×26 voxels × 12 frames × 7 × 2, f16, 21.8 MB) reads in 80 ms on 1 thread and 21 ms on
  8 (Python and Rust alike).

**RACLETTE store:** `hvtool pack-tree` → one `case.hv` per case.
- zstd-19 + byte shuffle; `x` chunked `[16,16,16,1,E,2]`, `mask` `[16,16,16,1]`.
- Verified byte-exact before the npy files are deleted; `case.json` gains a `storage` section.
- `dataset.py` and `train_figs.py` read it lazily (npy fallback).
- Done 2026-11-04 (8 threads, 39 min): all 314 cases verified, 114.7 GB → 76.4 GB (0.666); free disk 120 → 154 GB.
- Still npy: the breathing motion fields of the 200 image-less bins (2.1 GB).
- `test_dataset.py` passes on case.hv. Worker cost is 182 ms/sample including the first open of each case, against
  82 ms from npy in the page cache. Use ≥ 8 DataLoader workers (HASTY_VALUE_THREADS=2–4 each).

Lossless compression measured on fp16 images with in-vivo backgrounds:

| setting | ratio |
|---|---|
| shuffle + LZ4 | ≈ 0.91 |
| shuffle + zstd-3 | ≈ 0.87 |
| shuffle + zstd-19 | ≈ 0.72 per case with mask / maps (0.78 for x alone) |

- Bitshuffle gains < 1 % at level 19, so it isn't implemented.
- Lossy option (not applied): rounding the fp16 mantissa to 8 / 7 bits gives 0.64 / 0.57. The error stays below the
  training noise: median relative error 0.07 / 0.15 %, phase error ≤ 2 / 4 mrad.

## 23. Interruptible, resumable runs (2026-11-04)

Every long task can be paused and continued: Ctrl-C, SIGTERM or `touch <run dir>/STOP` → finish the current step,
write a checkpoint, exit 0. The same command line resumes.

**Denoiser training** (`tools/denoiser/train.py`)
- The run directory is `runs/denoiser/<name>` (stable); it auto-resumes from `ckpt_last.pt`. `run.json` holds the
  hyperparameters, so a resume can't silently change them. `--fresh` starts over.
- Atomic checkpoints every `--ckpt-every` steps, plus `status.json` (iteration / target / it/s / ETA).
- The sample stream is deterministic: sample k is a function of (seed, k), so the resumed stream continues exactly.
- DataLoader workers ignore SIGINT.
- Tested: SIGINT at 21, STOP at 43, completion at 80, no errors.

**Reconstruction** (`hasty_recon::checkpoint::RunCtl`)
- State files `state_<name>.hv` (LZ4, atomic). `P2np::run_from` saves k, x, x_prev, g_prev (the ZMSHR1 secant pair)
  and α. `cg::pcgls_ctl` saves x, r, p, γ, ‖s₀‖ and its history.
- `recon_p2np` defaults to a stable run directory:
  - warm start resumable per encode (finished encodes kept);
  - each preconditioner / κ configuration resumes and appends its metrics CSV;
  - finished configurations are skipped;
  - `--fresh`, `--ckpt-every n` (default 10).
- The LLR cycle-spinning shift is a function of the iteration (`Llr::set_iteration`), so it repeats after a resume.
- GPU reductions are not bit-deterministic: λ_max differs in the 7th digit between runs, and ZMSHR1 amplifies this to
  ~1.4 % max image difference after 12 iterations, between two *uninterrupted* runs. A resumed run must agree with an
  uninterrupted one to within that spread.
- GPU sharing: training (12.6 GB) and a full recon (> 11 GB) don't fit together on 24 GB. Pause training with its
  STOP file, run the reconstruction, then rerun the training command.

## 24. Denoiser training restart, 7-point reconstruction, learned prior plugin (2026-11-04)

**Training NaN (first 7p_v1 run, NaN from iteration ~340)**
- Cause 1: `inv_abs` (1/|x_ref| in the lift / unlift) has derivative ~|z|⁻³. Near-zero (but nonzero) reference voxels
  — resampled patch edges outside the FOV — overflow it; 0 · ∞ then turns masked-out gradients into NaN.
  Fix: |z|² is floored at 1e-12 × its per-sample mean. That keeps degree-2 homogeneity and O_h invariance exact, and
  tiling stays exact (tests).
- Cause 2: the noise σ was derived from each patch's own lumen / p90 level, so air patches got σ ≈ 1e-8, and the
  σ²-normalised loss then weighted them 10⁴×.
  Fix: σ = (case blood level, mean |x_ref| in the lumen over 4 frames) × |gain| / SNR. The blood level is computed
  per case in the worker and passed with the sample — a property of the acquisition, not of the patch.
- Guard: a step whose loss or gradient norm is non-finite is skipped and logged; 20 in a row stop the run.
- Verified: 800-iteration reproduction, 0 skipped steps. 7p_v1 was then restarted `--fresh`: 2.9 it/s alone,
  2.2 it/s while sharing the GPU.
- Learning rate. The 1000-iteration train mean rose 9.6 → 30 while val improved (0.98).
  - Per-iteration log (`loss_iter.csv`: loss, gradient norm, per-sample loss / SNR / case): the loss is
    heavy-tailed. At iteration 3, two Adam steps from the identity init already cost 130 σ².
  - Sensitivity: a 1e-3 perturbation of the head weights moves the output by 0.3–2.6 σ (orbit-shared parameters
    drive many weights coherently), so Adam at lr 1e-3 jitters several σ on high-SNR samples.
  - 7p_v1 is paused at 3303 (resumable). `7p_v2` runs with peak lr 3e-4, otherwise the same.

**Reconstruction with any encoding** (`hasty_recon::flow::FlowDecoder`)
- Velocity from phase differences to encode 0, φ_e = k_e·v mod 2π, k_e = γ(M1_e − M1_0), by hierarchical unwrapping:
  1. the least-sensitive group spanning 3D gives the first estimate;
  2. each more sensitive group is unwrapped against the current estimate and joins the least-squares solve.
- Tests cover referenced, Hadamard and dual-VENC: dual-VENC unwraps exactly up to 2.5 VENC_lo, and its noise
  matches the ML prediction σ_φ/(π √(1 + (VENC_lo/VENC_hi)²)).
- `recon_p2np` no longer assumes 4 encodes: sets, warm start, segments, synthetic data, outputs and metrics all
  follow `encode_m1()`. It writes `warm_vel.npy`, and `run_figs.py` uses it.
- `sim4d` is interruptible and resumable: one `state_sim_bins_<b0>_<b1>.hv` per finished chunk of flow-time bins;
  rerunning the same command skips those chunks.
- 7-point R4 simulation: `R4_dv_ideal_noro_pad` (`--venc 1.0 --venc-hi 3.5 --ideal --no-readout-motion`). Truth peak
  speed is 3.1 m/s; the RACLETTE 7p training data has VENC_lo = 1.0 and VENC_hi 1.5–7.0 at 2.0–2.5 mm, so this sits
  inside the training distribution.

**Learned prior in P²nP** (`--learned <export_dir>`)
- Plugin: `hasty-denoise-torch` also builds a cdylib, `libhasty_denoise_torch.so`, with a C ABI (v2):
  `hdt_open`, `hdt_denoise`, `hdt_set_autocast`, `hdt_on_cuda`, `hdt_close`. `hasty_recon::learned` dlopens it, so
  libtorch stays out of the main build.
- Update rule: x ← x + κ (D(x) − x) per iteration, on the host stack in `images[frame·7 + e]` order (the temporal
  model's encode-major stack is reordered).
- Exact tiling (`hasty_denoise::tiled`) bounds memory, with the same result as the whole volume:
  - spatial overlap tiles, margin 28 on the coarsest lattice;
  - cyclic time windows, margin ±3 frames, because the temporal convs have no stride and only the margin frames
    see the window's circular pad.
  - Measured: 0.0 relative difference, Rust TorchScript CPU, tiles × windows.
- Memory: about 6 full-resolution 96-channel tensors per voxel-frame (fp32), so a 61² × 29 × 20-frame tile needs
  more than 5 GB. With 28-voxel margins small spatial tiles are inefficient (tile 61 keeps 5 voxels), so time
  windows are the cheap lever.
- bf16 autocast is the default on CUDA: the training regime, half the activation memory.
- Export fixes: checkpoints load with `weights_only=False` (our own files); the reference volume is read from
  `case.hv`.
- Early checkpoint (iteration 2.5k) exported to `data/hasty/denoiser/7p_v1_early`: Rust reproduces Python to 8e-8.

## 25. 4D flow visualisation (2026-11-04)

**Backend** (`hasty-server/src/flow.rs`, CPU / rayon, so it runs next to GPU work; registered in `Registry::base`)

Flow value: a dict with
- `kind = "flow4d"`;
- `vel`: f32, hasty `[X, Y, Z, 3, F]`, m/s along the array axes;
- `mag` (optional): `[X, Y, Z, F]`;
- `spacing` (m), `dt` (s per frame, cyclic), `venc` (optional);
- `mask` (optional): `[X, Y, Z]`.

Commands:

| command | what it does |
|---|---|
| `flow_make` | builds a flow value from tensors |
| `flow_info` | dims, p99 / p99.9 speed, mean speed per frame in the vessel region, peak frame |
| `flow_pcmra` | PC-MRA: time mean of \|mag\|·speed, mean speed, or max speed |
| `flow_streamlines` | per frame, RK4 on the unit direction field, traced both ways from the seeds |
| `flow_pathlines` | RK4 in space and cyclic time; several releases per cycle; exact end time |
| `flow_plane` | analysis disc (see below) |

- Interpolation: trilinear in space, linear and cyclic in time.
- Seeding: the PC-MRA region above a quantile (or the mask), a disc, or explicit points. Tracing stops outside that
  region, dilated by one voxel.
- `flow_plane` reports Q(t) in ml/s, mean and peak through-plane velocity, forward / backward / net volume, regurgitant
  fraction and area. It also returns the through-plane velocity image per frame and the disc basis.
- Tests use analytic fields: uniform-flow pathlines are exact; solid-body rotation closes (< 0.05 voxel after one period)
  and its streamlines stay at constant radius (< 1e-3); cyclic time interpolation; Poiseuille Q(t) within 1 % for every
  frame, plus reversed normal and peak velocity; region seeding and value round trip.
- Publishers:
  - Rust `viz::show_flow`;
  - Python `hastycompute.v2.flow_value` / `show_flow` (hasty or C layout);
  - `tools/flow/show_run.py <run_dir>`: simulator truth plus every finished reconstruction of a `recon_p2np` run;
  - `train.py --show`: validation sample 0 (clean / noisy / denoised, dual-VENC unwrapped), overwritten in place at
    every validation so open viewers reload.

**Frontend** (`plotting_website/flow/`): three.js 0.186.1, vendored under `plotting_website/vendor/` (MIT); WebGL2.
- Why WebGL2: headless Chrome renders it in software (SwiftShader), so `tools/web/headless.py --screenshot` can test
  the page. WebGPU canvases cannot be tested headless here.
- PC-MRA MIP: a ray-marched 3D texture with level and opacity.
- Axial / coronal / sagittal slice planes:
  - magnitude with an overlay of speed or a single component (signed colormap), a threshold and an opacity;
  - instanced cone glyphs on the first visible slice.
- Streamlines per frame, cached, following the cine.
- Animated pathlines:
  - screen-space wide instanced segments with comet tails and particle heads;
  - the cycle wraps, so a release late in the cycle continues at the start.
- Analysis planes:
  - click a slice to set the cursor; the normal follows the local velocity;
  - the disc is coloured by through-plane velocity per frame;
  - a Q(t) plot with a frame cursor, plus net / forward volume, RF, area and peak.
- Streamlines and pathlines can be seeded from the active plane (emitter).
- Cine: play / pause (space), ← / →, ¼×–2× cardiac speed.
- Colormaps: turbo / viridis / inferno / coolwarm / gray, with a legend.
- Live: switches to newly shown flow values; reloads when the current value is updated.
- `?selftest=1&layers=…` runs the whole pipeline and reports JSON. On R4 truth: 600 streamlines, 1598 pathlines,
  plane net 108.6 ml per cycle (aortic stroke-volume range).
- The index page has a Flow card; flow values in the live bank list get a "flow" link.

**First 7-point reconstruction** (G1: R4 dual-VENC 1.0 / 3.5, same scan time, configuration and LLR as the 4-point F1)

| | 4-point F1 | 7-point G1 |
|---|---|---|
| velocity RMSE at true positions | 0.160 | 0.183 |
| on-grid median \|e\| | 0.046 | 0.043 |
| RMSE without \|e\| > 0.8 m/s | 0.091 | 0.087 |
| voxels with unwrap failures (\|e\| > 0.8) | 0.03 % | 0.18 % |

- Typical errors are lower with 7 points, but dual-VENC unwrap failures (2·VENC_lo jumps) dominate the RMSE.
- The lever is the joint 7-point prior, or spatial / ML unwrapping instead of per-voxel rounding.

## 26. Next research thread: velocity-parametrised reconstruction + finer grids (2026-11-04)
Plan: `../research/velocity_model_sr.md`.
- Unknowns z (complex reference), v (spline / Fourier in cardiac time) and a per-encode residual-phase polynomial.
- Same P²nP algorithm: pointwise Gauss–Newton blocks plus ZMSHR1 in the block metric.
- Learned prior in (|z|, p = |z|·v): O_h scalar + vector fields, exact intensity equivariance.
- `--grid-scale` (anisotropic, non-integer), `--voxel-basis`, coarse-to-fine, ROI dual grid.
- Milestones S0 / V0–V5 / B0–B2 with gates.

## 27. Velocity model and memory planner: implemented (2026-11-04)

**Velocity model** (`hasty-recon`)
- `vmodel.rs` / `vmodel.cu`:
  - periodic B-spline time bases for z and v (any K_z, K_v, degree 0–3, independent of the operator frames);
  - a Legendre residual-phase basis (degree ≤ 3, per encode);
  - GPU kernels: evaluation, vector–Jacobian product, Jacobian–vector product, the polynomial reduction, and the
    Gauss–Newton weights and metric;
  - parameters packed into `Stack` chunks, so the P²nP algebra and checkpointing apply unchanged.
- `vobj.rs` `VelProblem`, a P²nP `Objective`:
  - the data term through any `SetOp` (per-frame sets t = f·E + e);
  - the gradient, the Gauss–Newton Hessian-vector product, and λ_max(MH);
  - the pointwise Gauss–Newton metric (z scalar, v (c·σ_k)⁻¹(Σ k_e k_eᵀ)⁻¹, residual phase diagonal, Levenberg
    damping, refreshed every 5 gradients);
  - per-set weights for VENC continuation.
- `p2np.rs`: the loop is generic over an `Objective` (`run_objective[_with]`); the complex model is one
  implementation.
- **Tests:**
  - CPU f64 finite differences: gradient, JVP, adjoint;
  - GPU vs CPU < 2e-6;
  - full objective through NUFFT `SenseSets` (gradient vs finite differences, Hessian symmetry);
  - solve from a perturbed start: exact recovery (0.0865 → 0.0000 m/s) when identifiable. With a free degree-1
    residual phase the data term converges, but the time-mean linear velocity trades against it, as predicted
    (§A3 of the plan).
- `recon_p2np --vmodel KZ,KV [--vdeg DZ,DV] [--vpoly d] [--vcont n]`:
  - initialised from the warm start (decoder);
  - stages for VENC continuation;
  - resumable;
  - writes vel / recon / vparams and the metrics (incl. wrap fraction) and figures.

**Memory planner** (`plan.rs`, `store.rs`)
- `probe`: free / total VRAM and pinned H2D / D2H bandwidth.
- `Need` / `choose`:
  - placeable buffers: k-space data (device or host), the ZMSHR1 secant pair (device or host), the rank-one direction
    w (device, host, or recomputed in a third pass);
  - the cost model is gradient time + bytes / bandwidth;
  - it picks the fastest plan that fits the budget, and prints a clear error with remedies when nothing fits.
- The ZMSHR1 / ZMSHR1-M step is evaluated chunk by chunk in 2–3 passes over a device / pinned-host `Store`. The device
  minimum is x, g and the step plus chunk scratch; the previous loop kept 10 full stacks.
- `VelProblem::data_to_host`: streamed k-space data. `lambda_max` uses two stacks.
- `recon_p2np`: `--vram-budget 8G` (emulates smaller cards) and `--plan auto|device|host`. Data placement is decided at
  setup; state placement is refined with the measured gradient time.
- **Tests:** every placement (host secant + host w, host secant + recomputed w, device + recomputed w, host-streamed
  data) reproduces the all-device solve to float rounding.
- Not placeable yet: operator tables (SegPoints), coil maps, NUFFT grids. The coil batch and segment count are the
  operator's own knobs, through the SegNufft planner's `max_grid_bytes`.

**First velocity-model results** (R4 7-point dual-VENC 1.0 / 3.5, 60 s, R ≈ 8 per frame; 30 iterations; no prior;
ZMSHR1 in the Gauss–Newton metric; initialised from the 30-iteration time-averaged warm start).
Reference: the complex model with the LLR prior (G1) reaches 0.183 m/s.

| z (nodes, degree) | v (nodes, degree) | VENC continuation | velocity RMSE | wrapped (\|e\| > 0.8 m/s) |
|---|---|---|---|---|
| 1 (static) | 20, linear | – | 0.343 (semi-convergent: 0.248 at iteration 5) | 0.61 % |
| 20, linear | 20, linear | – | 0.237 | 0.51 % |
| 20, linear | 8, cubic | – | 0.182 | 0.39 % |
| 8, cubic | 8, cubic | – | 0.229 | 0.46 % |
| 8, cubic | 8, cubic | first 10 iterations reference + high-VENC only | **0.138** | **0.11 %** |

- Fewer, smoother velocity unknowns plus continuation beat complex + LLR by 25 % without any prior, and the wrap
  failures drop 4×.
- A static z can't follow pulsatile magnitude: its data term stalls at 7.3e11 against 4.7e11.
- Gradient cost: 1.33 s (140 sets, SegNufft L = 1).

**Residual-phase injection (S0, degree ≤ 1, exact):**
- Convention, measured by test (residual 5e-7): E_k(x·e^{i g·(r − n/2)}) = E_{k−g}(x).
- Implementation: a per-encode phase c + g·(r − n/2) shifts that encode's sample coordinates by +g and multiplies its
  data by e^{ic}. `recon_p2np --inject-phase SEED,AMP` draws Legendre P0 / P1 coefficients per encode, applies them
  before the operators and the warm start are built, and records them in run.json.
- Degree 2 needs the simulator (the concomitant slots of the sample parameters).
- Static-tissue anchor: `--vanchor μ` adds ½λ Σ w |v_k|². λ = μ × the mean Gauss–Newton curvature of one velocity
  node; the mask is simulator truth for now (pulsatility-based estimate next).

**Residual phase (S0): first results** (injected amplitude 0.6 rad, degree 1; 8×8 cubic, continuation 10, 30 iterations)

| | velocity RMSE (m/s) | wrapped | residual-phase error (rad) |
|---|---|---|---|
| not modelled | 0.847 | 6.0 % | — |
| `--vpoly 1` | 0.283 | 0.37 % | 0.295 (degenerate) |
| + anchor, metric without the anchor curvature | 0.338 | 0.59 % | 0.270 |
| + anchor, curvature in the v metric | 0.370 | 1.06 % | **0.099** |

- **Normal-operator scale.** The anchor needs an absolute scale. `VelProblem` estimates n̂ = mean diagonal of EᴴE per
  unit coil energy with a random probe; it came out at 1.18e5 here. λ = μ n̂ c σ tr/3, and λ w goes into the v
  metric block (trace-matched), otherwise λ_max(MH) is set by static tissue.
- **The residual phase is identified once the anchor is in the metric.** Velocity is still worse: in continuation
  stage 0 the high-VENC encodes carry the unestimated residual phase (×3.5 velocity bias), which then misguides the
  low-VENC unwrapping.
- **Next:**
  - initialise a by a static-tissue least-squares fit to the warm-start encode phase differences (plan §A3 step 1)
    before iterating;
  - estimate the static mask from pulsatility instead of the simulator truth;
  - degree 2 via the simulator.

**Added 2026-11-04 (evening)**
- `vinit.rs`: robust static-tissue fit of the residual phase to the warm-start encode phase differences.
  - IRLS with Tukey weights on the wrapped residual, signal-weighted. Vessels and noise become outliers.
  - The accepted voxels form the static mask, which replaces the simulator truth.
  - `remove_phase` and `decode` complete the initialisation.
  - Test: coefficients < 5e-3 rad, no vessel voxel in the mask, > 90 % of static tissue kept.
  - `recon_p2np --vpoly d` starts from it; `--vanchor μ[,auto|truth]`.
- `--vkappa k,…`: LLR prior in the velocity loop.
  - It runs separately on the z nodes and on p = |z̄|·v (3K_v real images), so units never mix; then v = p / |z̄|.
- `--grid-scale s | sx,sy,sz` (`SimData::set_recon_grid`).
  - The reconstruction grid (`dims`, per-axis `vox`) is separate from the native truth grid (`tdims`, `res`).
  - Smooth 2·3·5·7 sizes, FOV kept exactly; sample coordinates k·Δ_a follow the reconstruction voxel.
  - Truth is always sampled at true positions; `--synthetic` (direct truth indexing) is native-grid only.
- `--voxel-basis dirac|rect|bspline3`: `setop::WeightedSets` wraps any SetOp with per-sample weights
  Π sinc(k_a Δ_a / 2)^p (p = 1 rect, 4 cubic B-spline). The scratch is allocated per call, with no data copy.

**Results 2026-11-04 (night)** (7-point R4 unless stated; 30 iterations; velocity RMSE at true positions)

| experiment | RMSE (m/s) | note |
|---|---|---|
| 8×8 cubic + continuation, native grid, no prior | 0.138 | reference |
| + LLR on (z, p), κ 0.5 / 1 / 2 | 0.138 / 0.182 / 0.179 | LLR is not a useful prior here |
| grid 1.25× (1.6 mm), Dirac / rect voxels | **0.130** / 0.133 | gradient 2.0 s vs 1.34 s |
| native grid, rect voxels | 0.139 | voxel basis neutral at this resolution |
| continuous time (`--temporal linear`) | 0.142 | 20 bins are already fine |
| injected residual phase 0.6 rad, `--vpoly 1`: no init / robust init | 0.283 / 0.191 | init error 0.206 rad |
| … + anchor, auto mask (fit only) | 0.254 | 29k blood voxels in the mask |
| … + anchor, refined by pulsatility, 1.5× median, dilated 3 (now default) | **0.166** | 1.5k blood voxels; oracle mask 0.157 |
| 4-point ideal (vs complex + LLR F1 0.160) | 0.165 | no dual-VENC advantage |
| 4-point full physics + GNL (vs complex F2 0.196) | 0.690 | diverges, see below |

Notes:
- The LLR NaN came from nalgebra's f32 Hermitian eigensolver on exactly rank-deficient blocks (identical spline
  nodes). The Gram matrices are now decomposed in f64, and non-finite blocks are skipped.
- The velocity path no longer builds the unused circulant kernels (≈ 8 GB at the native grid, OOM at 1.25×).
- **Full-physics divergence: an encode-common phase error leaks into v.**
  - The v gradient contains −|z|² sin φ_common Σ_e k_e. For referenced and dual-VENC encodings Σ_e k_e ≠ 0, and
    the block-diagonal Gauss–Newton metric drops this z–v cross term.
  - Fix: **centred encodings** k̃_e = k_e − mean_e k_e. z becomes the mean-encode reference, the cross term vanishes
    exactly, and velocity is unchanged. `HASTY_VM_UNCENTRED=1` restores the old parametrisation.
- `DenoiserZP` (tools/denoiser/model.py): the velocity-space prior is an O_h scalar ⊕ vector field U-Net on
  [|z|, p = |z| v] per node, bias-free (D(µ·) = µD(·)), conditioned on h only. Tested: equivariance 6e-16,
  homogeneity 1e-15, identity at init. Training waits for the CFD truth data.
- **Centred encodings: rejected as the default.** z must then carry the velocity-dependent mean-encode phase k̄·v(t),
  which breaks its temporal smoothness: 7-point 0.138 → 0.174 m/s; full physics only 0.69 → 0.49.
- **z warm-up (`--vzwarm n`)** runs n iterations updating z only (v and the residual phase frozen: a linear
  subproblem), so z matches the operator before velocity moves.
  - Full-physics 4-point: 0.69 → **0.232** m/s with 10 z-only + 30 joint iterations, wraps 0.01 %. The complex model
    with LLR (F2) reaches 0.196.
  - 7-point ideal: 0.135 m/s, wraps 0.06 % (5 z-only), no harm.
  - A z-only stage of 10 iterations had not converged yet (data term 7.4e12 → 3.6e12).
