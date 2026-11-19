# Running on clusters and rented GPU nodes: portable multi-GPU plan (2026-11-10)

Goal: one simulation or reconstruction runs fast on **one node with 4–8+ GPUs**, on whatever node we get:
Kebnekaise (HPC2N), Arrhenius (NAISS), a rented cloud node, or the local RTX 4090 box. No fast interconnect needed;
the work splits into independent pieces and the data is streamed from host RAM. Cost effectiveness is the criterion.

Facts below come from a web survey on 2026-11-10 (numbered sources at the end). Prices move weekly; Vast numbers are
one snapshot of its public offer API.

## 0. Decisions in one place

1. **One process per node, one worker thread per GPU, shared host RAM.** No MPI. Multi-node use means independent jobs
   (simulation bin ranges, separate reconstructions), never a distributed solve.
2. **Dynamic work queue over the natural unit** (recon: the set; sim: the 250-bin chunk). Faster GPUs take more units,
   so mixed GPU models balance themselves. Reductions run in unit order, so results do not depend on the GPU count.
3. **No compile-time paths at runtime.** Everything resolves from env / a site profile (`HASTY_REPO`, `HASTY_DATA`,
   `HASTY_RUNS`, `HASTY_TOOLS`, `HASTY_PYTHON`, `HASTY_SCRATCH`, `HASTY_KERNEL_CACHE`, `HASTY_TARGET`,
   `HASTY_DENOISE_PLUGIN`), then the checkout the binary lives in (crate `hasty-site`).
4. **Two CPU architectures, two CUDA floors.** x86_64 and aarch64 (Arrhenius GH200) builds. cudarc already loads CUDA
   libraries at runtime, so the Rust build needs no CUDA SDK and cross-compiles. Default NVRTC 13 (driver ≥ 580); a
   `cuda12` feature for clusters with older drivers. Kernels stay compiled to cubin per device arch at runtime.
5. **One OCI image, two flavours** (core; core + libtorch for the learned prior), built for amd64 and arm64. Clusters
   run it with Apptainer (`apptainer pull docker://…`), cloud hosts run it with Docker.
6. **On big-RAM nodes the k-space never touches disk.** 768 GB nodes hold the 43 GB compressed store in RAM; the
   disk store tier is chosen only when RAM is short (also the SSD-wear rule).
7. **Slurm preemption through a signal, not the time limit.** `--signal=B:USR1@<lead>` → the run stops at the next
   iteration boundary, checkpoints, the job requeues itself.
8. **Real patient data only on Arrhenius SENS or an EU provider under a DPA**, never on marketplace hosts.

## 1. Targets

| Target | GPUs per node | Host | Host↔GPU | Notes |
|---|---|---|---|---|
| Kebnekaise H100 (2 nodes) | 4× H100 SXM5 80 GB, 3.35 TB/s | 2× EPYC 9454, 768 GiB (620 usable) | PCIe 5 x16 | `--gpus=h100:4`; charge 3 per GPU-h [15] |
| Kebnekaise A40 (1 node) | 8× A40 48 GB, 0.70 TB/s | 2× EPYC 9334, 768 GiB | PCIe 4 | `--gpus=a40:8`; charge 1.5 |
| Kebnekaise L40s (2 nodes) | 6× L40s 48 GB, 0.86 TB/s | 2× EPYC 9334, max 10 cores per GPU | PCIe 4 | `--gpus=l40s:6` |
| Arrhenius GPU (382 nodes) | 4× GH200, 96 GB HBM3, 4 TB/s each | 4× Grace (72 Arm cores, 128 GB each) = 512 GB, 1.8 TB NVMe | NVLink-C2C, ~250 GiB/s measured | **aarch64**; `--gpus N`, account `*-gpu`; nodes shared [1][4][5][11] |
| Arrhenius SENS | 4× L40S (20 nodes) or 4× H100 (1 node) | x86 EPYC 9335, 384/768 GB | PCIe | sensitive data; restricted internet [1] |
| Vast.ai 8× RTX 5090 | 32 GB, 1.79 TB/s, 105 TFLOPS FP32 | median 504 GB | x16 on 38 of 43 hosts | ~$0.52/GPU-h on demand, $0.40 interruptible [24] |
| Vast.ai 8× RTX 3090 / 4090 | 24 GB, 0.94 / 1.0 TB/s | 252 / 504 GB | 3090 rigs often x4/x8 | $0.18 / $0.46 per GPU-h |
| EU clouds (Verda, Nebius, Scaleway, OVH, Genesis) | H100 / L40S / PRO 6000 | datacentre | PCIe 5 / SXM | $1.5–5 per GPU-h; DPA possible [35–40,43] |
| Berzelius (NSC, KAW) | 8× A100 40/80 GB (94 DGX nodes) or 8× H200 141 GB (16 DGX nodes) | x86 DGX | NVLink, PCIe 4/5 | PI must be ≥ assistant professor with a KAW/WASP link (PhD student = proxy PI); project primarily AI/ML; no sensitive data; ≤ 240 GPU-h/month fast-tracked; ≤ 6-month projects; CUDA 13.0 driver [48] |
| Local | 1× RTX 4090 | 32 cores, 94 GB | PCIe 4 x16 | development, small runs |

Kebnekaise: Ubuntu 22.04, Apptainer without fakeroot (build images elsewhere), `/scratch` ≈ 170 GB per GPU node
shared by its jobs, `/proj/nobackup` for project data, 7-day walltime [14–19]. CUDA modules 12.4–12.6 are documented;
**the driver version is unverified** (decides whether the NVRTC 13 build runs there; ask HPC2N support).
Arrhenius: modules include `GPU/buildenv-nvhpc/25.9-cu13.0` (suggests driver ≥ 580) [4][6].

### Which hardware fits which workload

| Workload | Bound by | Best value |
|---|---|---|
| Simulation (`simphantom`, direct sum over samples × sources × coils) | FP32 / SFU throughput | **consumer Blackwell**: 8× 5090 = 840 TFLOPS for ~$4.2/h; an H100 has 67 TFLOPS at ~$2.6/h |
| Reconstruction (SegNufft spreading, FFT, host streaming) | memory bandwidth + host→device | GH200 (C2C makes host streaming ~5× faster than PCIe 5), H100, then 5090 |
| LBM (`cardiac_lbm`) | memory bandwidth | one run per GPU (beat-library cases in parallel) |

Rough numbers, from the local 4090 (1.0 TB/s, 83 TFLOPS):
- 1 mm development scan ≈ 36 h at 350 W → 8× 5090 ≈ 3.5–4 h ≈ $15–20.
- 2 mm reconstruction ≈ 9 min per iteration → 8× 5090 ≈ 1 min per iteration, **if the CPU prior step keeps up**
  (measure first, §2.5).

Bandwidth per dollar (price / TB/s): 3090 $0.19, 5090 $0.29–0.37, 4090 $0.34–0.53, H100 $0.67–1.5 [24][31]. On
Kebnekaise per charge unit: H100 1.12 TB/s, L40s 0.51, A40 0.46 → the H100 node is the best use of an allocation.

## 2. Architecture

### 2.1 Device pool (`hasty-cuda`)
- `DevicePool::open(sel)`: `--gpus all|N|0,2,5`, env `HASTY_GPUS`; respects `CUDA_VISIBLE_DEVICES` (UUIDs included).
- Per device: name, compute capability, VRAM, memory bandwidth (attributes), PCIe gen/width and NUMA node (NVML /
  sysfs), a measured host→device rate (one 256 MB copy at startup, all devices at once).
- One worker thread per device, pinned to the GPU's NUMA node (dual-socket EPYC and the four Grace chips are NUMA). Its
  pinned buffers are allocated from that thread (first touch on the near memory).
- Per-device pacer: the global `PACER` becomes one per device; no sleeping while holding a lock; NVML handle per device.
- NVRTC: temp file names unique per thread (today `tmp{pid}` races); the disk cache key already includes the arch.

### 2.2 Work queue
- Shared `AtomicUsize` over unit indices, or blocks of units for locality (sim beat cache).
- Results land in per-unit slots; scalar reductions (f, ZMSHR1 dot products) are summed in unit order on the host →
  bit-for-bit independent of the GPU count and scheduling (except for in-kernel atomics, as today).
- A slow or failing device does not stall the others; a device error aborts the iteration with a checkpointable state.

### 2.3 Host memory shared by all devices
- Today every pinned allocation is tied to device 0's context (flags 0); cudarc's `PinnedHostSlice` records its own
  context's event. Two fixes, both needed:
  - Large shared, read-mostly host arrays (KStore in RAM, packed sample tables, host stacks): plain `Vec` registered
    once with `cuMemHostRegister(…, PORTABLE)`, so every context can DMA from it.
  - Per-device staging slots (Stager, SetStreamer, SegNufft table slots) stay per device.
- `SegPoints` gets a host-only form behind `Arc` (packed tables), uploaded into per-device slots.

### 2.4 Reconstruction (`recon_p2np`)
- Per device: `Kernels`, a SegNufft engine autotuned for that GPU (coil batch, shared memory, grid budget from its own
  `mem_get_info` instead of the fixed 6 GiB), coil maps, segment tables, a circulant workspace, a Stager, a SetStreamer.
- Static model (e.g. 280 sets): sets are pulled from the queue. With N GPUs each device holds 6·S/N of the stacks, so
  the state often fits on the devices again.
- Temporal model (≤ 7 sets): split inside a set by coil groups, sum the image per set.
- Warm start: one encode per GPU.
- Prior step: today the whole stack goes to the CPU each iteration (LLR, wavelet, DFW, static lock). On 64–192-core
  nodes this is parallel (rayon), but it is the Amdahl term. The learned prior tiles frames across the GPUs.
- Velocity model: per-device partial parameter gradients, summed on the host.
- **First step: per-iteration timing** (gradient GPU time, prior CPU time, host↔device bytes and seconds) in the log
  and metrics CSV. It decides whether the prior needs to move to the GPUs before more GPUs help.

### 2.5 Simulation (`simphantom`)
- 250-bin chunks from the queue, contiguous blocks per GPU (beat-moment upload cache).
- `write_at` (positioned writes) instead of seek + write on one `&mut File`; the data file created once; the final
  rename and noise pass after all chunks.
- The host beat cache (`BeatSrc`, ~2.5 GB per beat) shared behind a lock rather than copied per GPU.
- `--bins b0:b1` for splitting one scan over several nodes (independent jobs, merged by the chunk markers).

### 2.6 LBM (`cardiac_lbm`)
No domain decomposition. A node runs the independent cases (library beats, `run_lib` and `run_cor`) one per GPU.

## 3. Portability

### 3.1 Builds
- Targets `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`. No CUDA SDK at build time (cudarc dynamic loading);
  the CFD CPU kernel (`lbm_cpu.c`) and the server's protox are plain cross-compiles.
- `hasty-denoise-torch` (C++ shim over libtorch) builds on the target architecture from the PyTorch wheel's libtorch
  (aarch64 CUDA wheels exist from PyTorch 2.11 [47]).
- CUDA features: `cuda-13000` (default) and `cuda12` (cudarc `cuda-12080`) for drivers below 580. CUDA 13 dropped
  compute capability < 7.5 [45d], so the minimum GPU is Turing (Kebnekaise's V100s are out).
- Compile to cubin per device (as today), never PTX: PTX from a newer NVRTC fails on older drivers [45a]. The
  forward-compatibility package does not cover GeForce cards [45c], so on cloud hosts filter by driver.

### 3.2 Runtime configuration
- Crate `hasty-site` (done, C0): `HASTY_REPO` (checkout or container prefix with the same layout), `HASTY_DATA`
  (default `<repo>/data/hasty`), `HASTY_RUNS`, `HASTY_TOOLS`, `HASTY_PYTHON` (figures; default `rustlib/.venv`, else
  `python3`), `HASTY_SCRATCH` (node-local), `HASTY_KERNEL_CACHE` (default `$HASTY_SCRATCH/hasty-kernels`, else
  `rustlib/target/hasty-kernels`), `HASTY_TARGET`, `HASTY_DENOISE_PLUGIN`. Without env the checkout is found from the
  binary's own path (`<repo>/rustlib/target/<profile>/<bin>`); the compile-time path is the last fallback.
- Site profiles `rustlib/sites/<site>.env` (local, kebnekaise, arrhenius, vast, runpod): scratch and data paths,
  pacing (off on clusters), memory-cap mechanism, image path, Slurm defaults. `hasty-site` picks one from
  `HASTY_SITE` or detects it (Slurm cluster name, hostname).
- Memory caps: `systemd-run` locally (no swapping), Slurm `--mem` on clusters. `hasty_site::available_memory` (done)
  takes the minimum headroom over the process's cgroup and its ancestors, v1 and v2 (verified: 3.00 GiB inside a
  `MemoryMax=3G` scope).
- The run scripts (`realistic_scan.sh`, queue scripts) call a `hasty-run` wrapper that applies the site's cap instead
  of hard-coding `systemd-run --user --scope`.

### 3.3 Container
- `rustlib/deploy/Dockerfile`: build stage (Rust toolchain, cross targets), runtime stage
  `nvidia/cuda:13.0.x-runtime-ubuntu24.04` (libnvrtc + builtins, libcufft) + the release binaries + a small Python venv
  (numpy, matplotlib, the `hastycompute` value-format package) for figures. Flavour `-torch` adds libtorch and the
  denoiser plugin (~5 GB).
- Pushed to a registry (GHCR); clusters `apptainer pull docker://ghcr.io/…` into project storage (set
  `APPTAINER_CACHEDIR`; the default fills the 25 GB home quota on Kebnekaise [19]); run with `apptainer exec --nv`.

### 3.4 Job control
- Slurm templates `rustlib/deploy/slurm/{recon,sim,lbm}.sbatch`: `--gpus`, `--mem`, `--signal=B:USR1@<lead>`
  (lead ≥ one iteration + checkpoint), self-requeue (`scontrol requeue` or a dependency chain), stage-in / stage-out.
- The binaries handle SIGUSR1 like the STOP file (stop at the next safe point, checkpoint; repeated SIGUSR1 never
  aborts; done in `RunCtl`). A second SIGINT / SIGTERM still exits 130 without saving (interactive abort).
- Interruptible cloud instances: checkpoint by wall time (`--checkpoint-minutes`), to the instance's persistent
  volume, only on those sites (locally the SSD-wear rule stands).

### 3.5 Data movement
- Per phantom: LBM beat library 59 GB, blood library 12 GB, anatomy / motion / flow < 0.3 GB; a 10-min 2 mm scan
  output is 74 GB. KStores are caches, never shipped (rebuilt on the node, in RAM where it fits).
- Pipelines stay node-local: ship the phantom inputs (or run the LBM on the node), simulate, reconstruct, ship back
  the results (velocity, magnitude, figures: a few GB).
- `hasty-sync` (rsync / rclone with a manifest + checksums), resumable; on Kebnekaise stage into `/proj/nobackup`
  (170 GB `/scratch` is too small for sim + store), on Arrhenius into the 1.8 TB local NVMe.

## 4. Validation without a multi-GPU machine
1. **N workers on one GPU** (several contexts on device 0): exercises the queue, per-unit reductions and per-device
   state locally. Acceptance: N = 1, 2, 3 give identical metrics (up to the known atomics noise).
2. **One cheap 2-GPU rental** (2× 3090/4090, ~$0.5/h, 1–2 h): the real multi-context pinned-memory paths.
3. **`nodecheck` bin** run first on every new node: GPUs, driver, concurrent host→device rate per GPU, NVRTC compile,
   FFT / SegNufft / direct-sum microbenchmarks, JSON report. It decides what a node is worth before a long run.

## 5. Work packages (order)

| WP | Content | Needs GPU |
|---|---|---|
| C0 | Runtime paths (`hasty-site`), SIGUSR1 graceful stop, unique NVRTC temp names, cgroup v1/v2 probe, per-iteration timing (`timing_<name>.csv`) — **done 2026-11-10** (cargo check; GPU tests pending) | no |
| C1 | `DevicePool`, NUMA pinning, portable host registration, per-device pacer, `nodecheck` | short tests |
| C2 | Multi-GPU simulation (chunk queue, `write_at`, shared beat cache, `--bins`) | yes |
| C3 | Multi-GPU reconstruction (per-device engines, set queue, reductions, warm start per encode) | yes |
| C4 | Container (amd64 + arm64, core / torch), site profiles, Slurm templates, `hasty-run`, `hasty-sync` | build only |
| C5 | First real node: Kebnekaise H100 or a rented 8× 5090 — `nodecheck`, then the 1 mm scan end to end | rental / allocation |

## 6. Actions for the user (accounts)
- **Arrhenius:** NAISS Small (≤ 500 GPU-h/month; PhD student or above, supervisor named) via SUPR; Medium
  (≤ 5,000 GPU-h/month, assistant professor or above), next deadline 2026-11-15 [3][7][8].
- **Kebnekaise GPU:** HPC2N Local 2026 round (partner sites, ≤ 1,000 GPU-h/month, open until 2026-12-15) [23]. Ask
  HPC2N support for the GPU node driver version.
- **Berzelius:** only through a WASP-faculty supervisor as PI (the user as proxy PI), framed as learned-prior /
  denoiser development on synthetic data [48]. x86 8-GPU nodes: no aarch64 port needed.
- **WARA Ops** (Ericsson, free for WASP members): interactive Jupyter GPU environment for prototyping; quotas not
  published [49].
- **Later:** EuroHPC Benchmark / Development access (monthly cut-offs; Arrhenius, LUMI, …), then AI for Science
  (≤ 360k GPU-h / 6 months, needs a prior Benchmark allocation) [50].
- **Cloud:** a Vast account with a small credit for C1/C5 tests. Filter: whole machine, `gpu_lanes=16`,
  `pcie_bw ≥ 20` GB/s, ≥ 512 GB RAM, `cuda_max_good ≥ 13.0`, datacentre where possible [24][25].

## 7. Open
- How much of an iteration the CPU prior takes (C0 timing) → whether the prior moves to the GPUs.
- Kebnekaise driver version → whether the `cuda12` build is needed there.
- GH200 specifics: whether host-resident state through C2C (or unified memory) beats explicit streaming.
- Real-data pipeline: Arrhenius SENS vs an EU provider; data processing agreement.

## 8. Audit 2026-11-17: Vast.ai 8× RTX 5090

This section checks §0–§7 against the code as of 2026-11-17 (branch RustBranch, read-only) and against a new snapshot
of Vast's public offer API [51]. Timings come from local run logs on the RTX 4090. **Every cost and speed figure for
the 5090 is an estimate** until the first rental measures it (action A5). A 5090 is about 1.65× a 4090 in memory
bandwidth (Vast-measured median 1454 vs 879 GB/s) and 1.33× in FP32 (108 vs 81 TFLOPS) [51]. The estimates below
assume 1.4× per GPU for the reconstruction (bandwidth-bound, with atomics) and 1.3× for the simulation (FP32/SFU-bound).

### 8.1 Will it build and run on sm_120?

**Yes, with no code change, on hosts with driver ≥ 580.** How it works today:

- **No CUDA SDK at build time.** Kernels are compiled at runtime.
  - cudarc 0.19.10 is built with `cuda-13000` and `dynamic-loading` (`rustlib/Cargo.toml:11`, `Cargo.lock`).
  - `Device::new` reads the compute capability and sets `arch = sm_{major}{minor}` (`hasty-cuda/src/lib.rs:43-51`).
    NVRTC then compiles for exactly that arch: `--gpu-architecture=sm_120` gives a **cubin, not PTX**
    (`hasty-cuda/src/nvrtc.rs:85,119-128`). Building to cubin rather than PTX matches §3.1.
- **Kernel cache.** The cache key includes the arch and the NVRTC version (`nvrtc.rs:53-60`). Cache files are written
  then renamed (`nvrtc.rs:138-144`), so 8 processes or threads can safely share one cache directory.
  - The key is a hash of the generated, size-specialised source. A new node therefore compiles on first use, so put
    `HASTY_KERNEL_CACHE` on the persistent instance disk.
- **No hard-coded arch** in library or binary code. The only one is a test, `hasty-cfd/src/gpu.rs:362` (`sm_89`,
  NVRTC only; harmless).
- **Shared memory is queried at runtime**, via `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`:
  - `hasty-nufft/src/plan.rs:187`, `hasty-nufft/src/segnufft.rs:1207`;
  - `hasty-fft/src/plan.rs:296`, `hasty-fft/src/mixz.rs:175`.

  CC 12.0 has the same limits as Ada 8.9: 99 KB per block, 100 KB per SM, 1536 threads per SM [61]. The tile
  configurations and autotune therefore carry over. The 99 KB literals at `segnufft.rs:2746,2777` are tests.
- **CUDA libraries.** None are linked; they are loaded by name:
  - `libnvrtc.so.13` with its builtins (every kernel);
  - cuBLAS and cuSOLVER (`hasty-recon/src/llr_gpu.rs:18-19`, batched `cusolverDnXsyevBatched` at :166/:238; GPU LLR);
  - cuFFT only in the benchmark bins (`cufft_baseline`, `fft_bench`, `eigbench`).
  - cudarc's name list includes `.so.13` and `.so.12` (CUDA 13 ships `libcusolver.so.12` and `libcufft.so.12`). The
    `nvidia/cuda:13.0.x-runtime-ubuntu24.04` image contains all of them; the driver's `libcuda.so` comes from the host
    through the NVIDIA container toolkit.
- **No Vulkan anywhere.** VkFFT went with the C++ stack. `hasty-fft` is our own NVRTC FFT (rustlib/PLAN.md §5, D2), so
  whether Vast containers expose Vulkan does not matter.
- **Torch side.** The venv has `torch 2.14.1+cu130` (`rustlib/.venv`, 7.1 GB), whose wheels include sm_120. The
  libtorch denoiser plugin is built against that venv (`hasty-denoise-torch/build.rs`). Torch 2.7+/cu128 is the floor
  for Blackwell.

**Minimums.**
- sm_120 needs CUDA ≥ 12.8 and driver ≥ 570.26 [60]. Our CUDA 13 build needs driver ≥ 580 [45d].
- 8× 5090 offers today: drivers 575 (2), 580 (18), 595 (12), 610 (4), 615 (4). 38 of 40 report
  `cuda_max_good ≥ 13.0` [51].

**Blockers and fixes:**

| # | Issue | Fix | Size |
|---|---|---|---|
| B1 | Hosts with driver < 580 (2 of 40) cannot run the CUDA 13 build | Filter `cuda_max_good >= 13.0`, or build the planned `cuda12` feature (cudarc `cuda-12080`; NVRTC 12.8 knows sm_120) | none / small |
| B2 | Nothing has been run on sm_120 yet: numerics, autotune choices and atomics are unverified | Run the test suite and `nodecheck` on a 1× 5090 first (A5) | small (rental) |
| B3 | The GPU pacer reads NVML index 0 (`hasty-cuda/src/pace.rs:143`), which is not the CUDA device when `CUDA_VISIBLE_DEVICES` is set | Keep `HASTY_GPU_WATTS` / `HASTY_GPU_DUTY` unset on rented nodes (the pacer is off by default) | none |
| B4 | No image, site profile or deploy directory exists yet (`rustlib/deploy`, `rustlib/sites` are absent) | A1, A2 | small–medium |

### 8.2 Multi-GPU readiness today

**Status of the work packages:**
- **C0 is done.** `hasty-site`, SIGUSR1 in `RunCtl` (`hasty-recon/src/checkpoint.rs:21-41`), unique NVRTC temp names,
  and the per-iteration `timing_*.csv` (R2 wrote it).
- **C1–C5 are not started.** No `DevicePool`, `HASTY_GPUS`, `--gpus` or `nodecheck` exists anywhere in `rustlib/crates`.

What the code assumes today:
- **Every binary opens device 0.** Examples:
  - `recon_p2np.rs:769` (then one `default_stream()`, :770);
  - `splinerecon.rs:274,606` (being edited by another agent);
  - `simphantom.rs:1341`;
  - `cardiac_lbm.rs:112,126`;
  - about 35 other bench bins.

  Only `llr_gpu.rs:119` takes an ordinal.
- **The libraries are stream-parameterised.** `hasty-fft`, `hasty-nufft` and `hasty-recon` take `&Device` / `&Arc<CudaStream>`;
  only 16 `default_stream()` calls exist in library code. The one global cache, `hasty-nufft/src/gridsize.rs:76`, is
  pure arithmetic. The libraries are therefore ready for per-device engines; the drivers are not.
- **Pinned host memory uses flags 0, from device 0's context.** Sites: `hasty-recon/src/store.rs:39`, `hoststack.rs:118`,
  `kstage.rs:40`, `plan.rs:38`, `vobj.rs:217`, `llr_gpu.rs:200,284`; `hasty-nufft/src/segnufft.rs:673,868`.
  - Correction to §2.3: under unified addressing, which every 64-bit Linux process has, `cuMemHostAlloc` memory is
    automatically portable to all devices [63]. A shared read-mostly pinned array does not strictly need
    `cuMemHostRegister(PORTABLE)`.
  - cudarc's `PinnedHostSlice` does tie an event to its context. Keep per-device staging slots (as planned) and pass
    shared host data as plain slices.
- **The fine-grid budget is fixed** at `max_grid_bytes: 6 << 30` (`recon_p2np.rs:1719`). On 32 GB cards it should come
  from `mem_get_info` or the ledger.
- **Checkpoints are off by default.** `--ckpt-every` defaults to 0 (`recon_p2np.rs:278`), which is right locally (SSD
  wear) and wrong on a rented node (§8.4).

**What works on 8 GPUs today, with no code change:** one process per GPU, each started with
`CUDA_VISIBLE_DEVICES=k`, so its device 0 is GPU k. That covers:
- independent runs: denoiser training arms, LBM cases (`run_lib` / `run_cor`), eval-matrix configurations,
  different phantoms;
- sharing the k-space store on disk through the page cache, and the kernel cache.

Limits of process-per-GPU:
- **Host RAM per process.** R2 needs 78 GiB, so 8 copies need 624 GiB. That is more than the median 504 GB of an
  8× 5090 host, so only 5–6 such runs fit at once.
- **CPU oversubscription.** Each process's rayon pool sees all host cores. Set `RAYON_NUM_THREADS` per process.

**What must change for one job on 8 GPUs:**

| Change | Where | Size |
|---|---|---|
| `--gpu k` / honour `HASTY_GPUS` in the four drivers (until the pool exists) | recon_p2np, splinerecon, simphantom, cardiac_lbm | small |
| `splinerecon --encodes` subset → 7 processes for 7 encodes (per-encode CGLS is independent) | splinerecon | small (stop-gap) |
| `DevicePool` + worker thread per device + NUMA pinning + `nodecheck` (C1) | hasty-cuda, new bin | medium |
| Fine-grid budget and table tier from the device's own free memory (ledger skeleton, memory_ledger.md §5 step 2) | recon_p2np, hasty-recon::plan | medium |
| Per-device engines (`Kernels`, SegNufft, coil maps, segments, Stager, SetStreamer) + unit queue over (resp bin, encode[, coil batch]) + host reductions in unit order (C3, temporal_operator.md §14) | recon_p2np / knot driver, hasty-recon `Objective::gradient_host*` | large |
| Step / metric work per unit instead of on one GPU (R2: the step takes 15 s of 169 s per iteration; it becomes the Amdahl term once the gradient is ÷ 8) | hasty-recon precond / p2np | medium |
| Simulation chunk queue, positioned writes instead of seek (`simphantom.rs:1562,1867`), shared beat cache, `--bins` (C2) | simphantom | medium |
| `train.py` DistributedDataParallel (sample k depends only on (seed, k), so shard by k mod world) | rustlib/tools/denoiser/train.py | small–medium |

**No NVLink, no P2P.**
- GeForce 5090s report no peer access with the stock driver; only patched community drivers enable it [62]. Every
  8× 5090 offer has `bw_nvlink = 0` [51].
- Consequences:
  - Every GPU-to-GPU byte goes device → pinned host → device.
  - `cudaMemcpyPeer` and NCCL's P2P transport are off the table. NCCL falls back to shared memory. Some stacks fail
    instead of falling back, so set `NCCL_P2P_DISABLE=1`.
- The plan's design already assumes this: independent units per GPU, state and reductions in host RAM, no split FFTs
  or domain decomposition.
- Cost of the reductions at 1 mm:
  - the partial knot images of a coil-split unit (1.4 GB) at 22–54 GB/s take 30–65 ms against about 10 s of unit
    compute;
  - the velocity-model gradient (5 GB) takes about 0.2 s.
- What does matter:
  - all 8 GPUs streaming at once through one or two CPU root complexes;
  - NUMA placement on the dual-socket EPYC hosts.

  Measure both with `nodecheck`.
- For training, the denoiser is tiny (`ckpt_last.pt` 4.7 MB including Adam state), so a gradient all-reduce through host
  shared memory is negligible.

### 8.3 Host resources

**Measured local needs:**

| Workload | Host RAM | GPU | Disk | Evidence |
|---|---|---|---|---|
| R2: 1 mm box, 2 resp × 20 cardiac (280 sets), readout stride 4, `fourier:4`, 40 iterations | peak RSS **77.8 GiB** (80 G cap) | 12–16 GiB of 24 | KStore 7.3 GB, run dir 0.4 GB | `box_recon/R2/time.txt`, `timing_*.csv`: 2 h 07 min wall, gradient 152.5 s per iteration, step 15.2 s, prior 1.0 s |
| Target recon: 8 resp × 20 cardiac, full readout stride (estimate) | ≈ 200–250 GiB (4 coefficient stacks × 32 GiB + KStore in RAM ≈ 29 GB + packed tables ≈ 10 GB + setup ≈ 20–40 GiB); cubic-16 knots → ≈ 330 GiB | per-GPU slots | KStore ≈ 29 GB | scaled from R2 and temporal_operator.md §16 [CHECK with `--plan-only` once the ledger exists] |
| simphantom, 2 mm 10-min | 84 G cap with `--beats 15` (2.5–3 GB per beat; 60 G OOMed) | — | output 26 GB | `realistic_scan_run` notes, `logs/box_sim.log` |
| cardiac_lbm library run | small | 17.9 M nodes | 26 GB output per library | `logs/realistic_lbm_lib.log`: 4303 s on the 4090 |
| Denoiser training | 8 loader workers; data ≤ 35 GB (fits the page cache) | small model | 1.5 GB phantom + ≈ 33 GB RACLETTE tier 1 | `runs/denoiser/*/status.json` |

**What a typical 8× 5090 offers** (40 on-demand offers, 2026-11-17; p10 / p50 / p90) [51]:

| | 8× 5090 | 4× 5090 | 1× 5090 |
|---|---|---|---|
| host RAM | 282 / **504** / 755 GB (only 5 ≥ 700 GB) | 141 / 252 / 504 | 31 / 63 / 126 |
| effective CPU cores | 144 / 224 / 320 | 48 / 112 / 192 | 12 / 32 / 48 |
| disk available | 1.9 / 2.9 / 6.0 TB; disk bandwidth 0.6–26 GB/s | 0.9 / 2.1 / 6.2 TB | 0.1 / 0.6 / 1.7 TB |
| internet down / up | 0.4 / 1.7 / 7.4 Gb/s down, 0.4 / 0.9 / 7.0 up | | |
| host↔GPU | PCIe gen 4 on 21 hosts (≈ 22 GB/s), gen 5 on 19 (≈ 54 GB/s); x16 on 39 of 40 | | |
| GPU memory bandwidth (Vast test) | ≈ 1450 GB/s (spec 1792) | | |

**Mismatches:**
- **RAM per GPU is about 63 GB on the median 8× host.** The single-job target recon fits (≈ 250 GiB of 504). Process
  per GPU at R2 size does not (8 × 78 GiB). The plan's "one process, shared host RAM" is the right shape. Filter
  ≥ 500 GB, or ≥ 750 GB for the cubic-16 knot basis.
- **1× 5090 rentals have 63 GB of RAM** (median). A 1 mm recon does not fit; training and 2 mm work do.
- **Disk bandwidth varies 40×.** At 0.6 GB/s a full-stride disk KStore (≈ 29 GB per iteration) would take ≈ 45 s per
  iteration, about as long as the compute. Put the KStore in RAM on these nodes (§0 decision 6), or filter
  `disk_bw ≥ 2000`.
- **Disk is billed by size, while stopped too.** The median is $0.20 per GB-month [51][55], so 1 TB ≈ $0.27/h. Size the
  disk at about 300–500 GB, not the 2–3 TB on offer.
- **CPU-heavy setup.** The R2 segment build took 306 s on a 10-CPU quota (memory_ledger.md §4). With ≥ 144 cores it
  should shrink, but the sequential per-set tables (17 %) and the least-squares table (46 %) remain until the GPU
  build lands.
- **PCIe gen 4 hosts are fine.** Streaming is ≈ 1 % of an iteration (memory_ledger.md §2), even at 22 GB/s.

### 8.4 Moving our stuff there

**Image.**
- Vast runs any Docker image (`repo/image:tag`, private registries with login) [57]. It does not support
  Docker-in-Docker [54], so images are built elsewhere.
- Recommended build: GitHub Actions on the public repo (`github.com/turbotage/HastyCompute` is public) →
  GHCR public image. That costs no local CPU (quiet-mode rule) and needs no registry credentials on hosts.
- Builder stage:
  - Rust 1.99.1 (`rustlib/rust-toolchain.toml`) and `build-essential` (cc builds `lbm_cpu.c`; protox needs no protoc);
  - `cargo build --release -p hasty-bench`, plus the server only if needed. No CUDA SDK is required.
- Runtime stage: `nvidia/cuda:13.0.x-runtime-ubuntu24.04` + binaries + a slim venv (numpy, matplotlib, `hastycompute`)
  for figures.
- Flavour `-torch` (training and the learned prior):
  - the cu130 torch wheel set (≈ 5–7 GB);
  - gcc and Python headers for `torch.compile`;
  - `hasty-denoise-torch` built against that libtorch.
- Stop-gap for the first test: start from a stock CUDA 13 image in SSH mode, `git clone`, and `cargo build --release`
  on the node (a few minutes on 144+ cores).

**Launch modes** [57]:
- *SSH* or *Jupyter + SSH*: the image ENTRYPOINT is **not** run; `onstart` is.
- *Entrypoint (args)*: the image runs as-is.
- Use SSH mode with an idempotent `onstart` that re-runs the same resumable command. Rerunning resumes in all our
  drivers, so a resumed interruptible instance continues by itself.
- Env vars set in the template are not visible in SSH sessions unless exported to `/etc/environment`.

**Data to ship** (measured `du`, 2026-11-17):

| Purpose | Contents | Size |
|---|---|---|
| R2-style recon of the existing 1 mm box scan | `sim_flow_10min_box1mm_b/scan` without `truth_fine`:<br>data_clean 3.2 G, sched_nominal 1.1 G, sched_played 1.2 G, samples_played 0.64 G, samples 0.28 G, smaps_true 0.9 G, field / physio / meta < 0.1 G<br>+ scoring truth: truth_pc_frames 0.9 G, truth_bias 0.43 G, truth_vel_logical 0.1 G, maps 0.2 G, `box_recon/*idx*`, roi 0.02 G | **≈ 9 GB** (the dir is 21 GB; `truth_fine` 9.7 G only for `flow_truth`) |
| 2 mm 10-min scan | `sim_flow_10min/scan` | 26 GB (data_clean 15 G) |
| Simulate on the node (full stride) | `sim_input.hv` 1.85 G + `lbm/run_lib_v2` 26 G + `run_cor_v2` 4.9 G + `seq/flow_thorax_v2_1mm` 10 M | ≈ 33 GB, or **≈ 2.5 GB** if the LBM runs on the node (cases 0.65 G; one library run per GPU, ≈ 1 h each) |
| Denoiser training | `train/phantom` 1.5 G; RACLETTE fetched on the node from its public share (`tools/raclette/fetch.py`, `build_training.py`) | 1.5 GB upload |
| Never ship | KStores (`rustlib/target/kstore` 51 G: derived caches), `.venv` (7.1 G: rebuilt in the image), `lbm/run_lib` v1, `pyr_*`, `conv/` | — |

**Transfer.**
- `rsync` over SSH, or `vastai copy` (rsync-based; local ↔ instance ↔ instance) [56].
- *Cloud Sync* / `vastai cloud copy` moves data between the instance and S3, Backblaze, GCS, Dropbox or Hugging Face,
  even while the instance is stopped [56].
- Vast's prices are small: ingress ≈ $2.7/TB and egress ≈ $4/TB at the median on 8× offers; some hosts charge $12/TB
  [51].
- The real limit is our own uplink [CHECK]. 9 GB at 100 Mb/s takes ≈ 12 min; 33 GB takes ≈ 45 min. Ship inputs and
  compute on the node; bring back only results (R2's run dir is 0.4 GB).

**Persistence** [54][55]:
- **Container disk** survives *stop*, including an interruptible pause, and is deleted on *destroy*. It is billed
  while stopped.
- **Local volumes** survive destroy but are tied to one physical machine.
- Anything that must outlive the rental goes to object storage (B2 / R2 / S3): results, checkpoints of multi-day
  runs, the kernel cache if wanted. Then destroy the instance.

**Interruptions.**
- Interruptible instances are *stopped* when outbid or when an on-demand rental claims the GPUs. Processes are killed,
  disk data is kept, and the instance resumes when it regains priority, possibly much later [52][53].
- Vast documents **no SIGTERM or grace period** [52][54][64]. Treat it as SIGKILL.
- Rental end dates also stop the instance (offer lifetime p10 28 days, median 113 days) [54][51].
- Our graceful paths (STOP file, SIGINT / SIGTERM / SIGUSR1 → checkpoint at the next iteration,
  `hasty-recon/src/checkpoint.rs:1-7`; `train.py:65-72`) **will not fire**. Only periodic checkpoints protect a run:
  - **Recon:** `--ckpt-every N` exists but defaults to 0. Turn it on for rented sites, ideally by wall time (§3.4
    `--checkpoint-minutes`, not implemented).
    - Checkpoints are written atomically, so a kill mid-write keeps the previous one.
    - A resume repeats the setup: segment build plus warm start, ≈ 10 min at R2 size. The basis cache
      (memory_ledger.md §4 item 6) makes resumes cheap.
  - **simphantom:** resumes from per-250-bin-chunk state (`state_sim_bins_*.hv`; `logs/box_sim.log` shows
    stop / resume). It loses ≤ 1 chunk (≈ 3 min).
  - **train.py:** `--ckpt-every 250` by default. It loses ≤ 250 iterations (≈ 2 min).
- On 8× nodes the bid discount is small: median $0.635 vs $0.707 per GPU-h. Use on-demand for reconstructions
  (1–2 h, setup-heavy). Use interruptible for 1× training runs (median $0.53 vs $0.74).

### 8.5 Training the denoisers

**Stack today.**
- Python + PyTorch: `rustlib/tools/denoiser/{train,model,oh,dataset,export}.py`, torch 2.14.1+cu130, bf16 autocast
  (`--amp`), `torch.compile`, a `DataLoader` with 8 workers.
- Patches are resampled on the GPU (`Patches(..., device=dev)`).
- The run is resumable: STOP / SIGTERM, and `ckpt_last.pt` every 250 iterations.
- Export goes to TorchScript / AOTI for the Rust libtorch plugin. ORT is only for the server's segmentation models
  (plan_models.md) and is not needed here.

**No DDP.** `train.py` is single-GPU. On an 8× node, run 8 independent arms (seeds, ablations, widths, the phantom
weight) with `CUDA_VISIBLE_DEVICES=k`. That works today.

**Measured speed.** On the 4090:
- 0.32–0.49 s per iteration at batch 4 (2.1–3.1 it/s; 7p_v1 / v2 / v3ph);
- so 100 k iterations take ≈ 9–14 h. Pacing may have been on [CHECK].
- Whether that is GPU- or loader-bound is unmeasured. Log GPU utilisation on the first rental before buying more GPUs
  per run.

**What the box needs:**
- the `-torch` image;
- `/dev/shm`, which Vast sizes by the GPU fraction [58]. On 1× rentals set `--workers` low or
  `file_system` sharing if loaders crash;
- the data: phantom cases (1.5 GB) plus RACLETTE tier 1 (≈ 33 GB, built on the node).

**Data pipeline.** Cases are zstd-chunked `case.hv`, read lazily by chunk. 35 GB fits the page cache of any 8× host,
so after the first epoch the loader is CPU-bound (zstd decode, crop): 8 workers × 8 arms = 64 cores of 144–320.
Throughput is not a concern on 8× hosts. On 1× hosts (median 32 cores) it is fine for one run.

### 8.6 Cost (snapshot 2026-11-17, Vast public API [51]; per GPU-hour, p10 / p50)

| Offer | On demand | Interruptible (min bid) | Notes |
|---|---|---|---|
| 8× RTX 5090 (40 / 30 offers) | $0.534 / $0.707 → **$4.3 / $5.7 per node-h** | $0.373 / $0.635 | Higher than the 2026-11-10 figure (§1: $0.52; method may differ). Filtered on CUDA ≥ 13, ≥ 480 GB, verified, reliability ≥ 0.98: **9 hosts, $3.85–11.15/h**. PCIe 5 + ≥ 645 GB: $6.14–8.54/h. Datacenter: $6.09/h (GB, 503 GB) and $8.00/h (CA, 645–755 GB) |
| 4× RTX 5090 (47) | $0.507 / $0.667 | $0.373 / $0.560 | median 252 GB RAM |
| 1× RTX 5090 (64) | $0.509 / $0.741 | $0.334 / $0.533 | median 63 GB RAM |
| 1× H100 SXM (6) | $1.74 / $2.34 | $1.47 / $1.73 | 2.9 TB/s measured, 54 TFLOPS FP32, 126 GB RAM |
| 1× H200 (6) | $3.66 / $4.61 | | |
| 8× H100 SXM (1) | $3.33 | $1.33 | a single host, unverified |
| 8× RTX PRO 6000 S, 96 GB (7) | $1.73 / $1.73 | $0.67 / $1.60 | 1–1.5 TB RAM, PCIe 5; the choice if 32 GB per card ever limits |

Value per dollar:
- **Bandwidth:** 5090 ≈ 2.0 TB/s per $/h vs H100 SXM ≈ 1.2.
- **FP32:** 5090 ≈ 150 TFLOPS per $/h vs H100 ≈ 23.
- The 5090 stays the cost-effective choice for both the reconstruction and the simulation. A single H100 only wins
  where 80 GB of VRAM avoids streaming, and our design streams anyway.

**Rough job costs** (on-demand median $5.7 per 8× node-h; plus ≈ 15–30 min per rental of image pull, data in and
compile; all [CHECK] until A5):

| Job | Local 4090 | 8× 5090 | Cost |
|---|---|---|---|
| Validation: `nodecheck`, tests, R2 single-GPU, 1 k training iterations on **1× 5090** | — | 2–3 h | **≈ $1.5–2.5** |
| R2-scale recon (1 mm box, 2 × 20, stride 4, 40 iterations), after C3 | 2 h 08 min | gradient 152.5 s → ≈ 15 s; with the step ≈ 30 s per iteration → ≈ 20 min + ≈ 10 min setup | **≈ $3–5** |
| Same, **today** (single-GPU process on a 1× 5090) | 2 h 08 min | ≈ 1.5 h | ≈ $1.1 |
| Target recon (8 × 20, full stride, 40 iterations), after C3 | does not fit 94 GB RAM; ≈ 8 h of gradient | ≈ 70–90 s per iteration → ≈ 1–1.5 h with setup | **≈ $7–12** |
| 2 mm 10-min sim, after C2 | 11.5 h (240 chunks × 172 s at 350 W) | ≈ 1.2 h | ≈ $7–9 |
| Denoiser, 100 k iterations, 1 arm | 9–14 h | 1× 5090: ≈ 6–10 h | $3–5 (interruptible), $4.5–7.5 (on demand) |
| Denoiser, 8 arms in parallel | — | ≈ 6–10 h on one 8× node | ≈ $35–60 |
| Disk, 500 GB | — | billed per hour, also while stopped | ≈ $0.14/h |

### 8.7 Security and data governance

**Vast's isolation protects renters from each other, not from the host.**
- Containers are unprivileged and namespaced [59].
- Vast's own FAQ answers "how is my data protected from providers?" with: security varies, use Secure Cloud, encrypt
  data, keep credentials off instances [59].
- Secure Cloud partners are in ISO 27001 / Tier 3–4 facilities and sign DPAs **with Vast**, not with us. Community
  hosts are not audited [59][65].
- The container also carries a Vast API key in `CONTAINER_API_KEY` [58].

**What may go there:**
- **Allowed:** our own phantoms and simulations; RACLETTE (public, CC BY-NC 4.0, research use, fetched from source);
  AVT / TotalSegmentator-derived anatomy (CC BY 4.0); Z-Anatomy (CC BY-SA); trained denoiser weights; code (the repo
  is public).
- **Never (decision 8 stands):**
  - real scanner data (future GE / Pulseq scans, physio logs);
  - anything under `data/hasty/studies` or `data/hasty/subjects` that is not a public dataset;
  - anything not explicitly allowlisted.

  Real data goes to Arrhenius SENS or an EU provider under our own DPA.
- **Mechanism.** The sync tool ships an **allowlist manifest** (paths and checksums), never a denylist. It refuses
  paths outside `phantom/`, `train/phantom/`, `seq/` and listed `external/` sets.

**Secrets:**
- No write-scoped GitHub token: clone anonymously, pull a public image.
- A per-rental SSH key pair; never `ssh -A` (a host root could use the forwarded agent).
- Object storage through a bucket-scoped key (B2 application key / R2 token limited to one bucket). Cloud Sync copies
  credentials onto the host [56].
- No Vast account API key inside the instance beyond Vast's own.
- Destroy the instance (not just stop it) after copy-out.

### 8.8 Prioritised actions

| # | Action | Needs | Size |
|---|---|---|---|
| A1 | Site profile `vast.env` + `hasty-run` launcher:<br>• `HASTY_*` paths on the instance disk, `HASTY_KERNEL_CACHE` persistent;<br>• pacer off;<br>• `CUDA_VISIBLE_DEVICES` and `RAYON_NUM_THREADS` per process;<br>• `--ckpt-every` on;<br>• `onstart.sh` that re-runs the resumable command | no GPU | small |
| A2 | `rustlib/deploy/Dockerfile` (core + `-torch`), built by GitHub Actions → public GHCR | CI only | small–medium |
| A3 | Wall-clock checkpoints (`--checkpoint-minutes`) in recon_p2np and the knot driver. Resume test by `kill -9` mid-run, not by the STOP file | CPU test at dev scale | small |
| A4 | `nodecheck` (C1 subset), output as JSON:<br>• driver, CC, NVRTC sm_120 compile;<br>• concurrent host→device rate per GPU, NUMA;<br>• cgroup memory and CPU limits, disk bandwidth;<br>• FFT / SegNufft / direct-sum microbenchmarks | short GPU test | small–medium |
| A5 | First rental, **1× 5090, 2–3 h, ≈ $2**:<br>• nodecheck and the test suite on sm_120;<br>• R2 single-GPU (≈ 9 GB shipped);<br>• 1 k training iterations with GPU utilisation logged.<br>Replaces every estimate above with a measured 5090/4090 factor | account + credit (user) | small |
| A6 | Allowlist sync (`hasty-sync`: rsync / `vastai copy` + object-storage results, checksums, manifest) | no GPU | small |
| A7 | Process-per-GPU use of an 8× node: 8 training arms, LBM cases, `splinerecon --encodes`, 4–6 eval configurations within RAM | A1, A5 | small |
| A8 | Ledger skeleton + device-sized grid budget (memory_ledger.md §5 step 2) | dev scale | medium |
| A9 | C3 multi-GPU recon: per-device engines, unit queue, host reductions, step per unit. It unlocks the 8 × 20 target recon, which cannot run locally | A4, A8 | large |
| A10 | C2 simulation queue + shared beat cache + `--bins` | A4 | medium |
| A11 | `train.py` DDP (`NCCL_P2P_DISABLE=1`), only if single-run wall time matters | A5 | small–medium |

## Sources
1. https://www.naiss.se/resources/arrhenius-technical-description/
2. https://www.naiss.se/resource/arrhenius/
3. https://supr.naiss.se/resource/arrhenius-gpu/
4. https://user.naiss.se/arrhenius_hpc/quickstart/
5. https://hpc.pages.naiss.se/user-documentation/support-docs/arrhenius_hpc/job_management/scheduling/
6. https://hpc.pages.naiss.se/training/ai-intro/lessons/basics/
7. https://supr.naiss.se/round/2026small/
8. https://supr.naiss.se/round/2026medium/
11. https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip
14. https://www.hpc2n.umu.se/resources/hardware/kebnekaise
15. https://docs.hpc2n.umu.se/documentation/batchsystem/resources/
16. https://hpc2n.github.io/intro-course/batch/
17. https://docs.hpc2n.umu.se/documentation/batchsystem/submit_file_design/
18. https://docs.hpc2n.umu.se/documentation/filesystems/
19. https://docs.hpc2n.umu.se/software/containers/
23. https://supr.naiss.se/round/2026hpc2n/
24. Vast public offer API, snapshot 2026-11-10: https://console.vast.ai/api/v0/bundles/
25. https://docs.vast.ai/documentation/host/verification-stages
31. https://www.runpod.io/pricing
35. https://verda.com/pricing
36. https://europeanstack.com/software/verda
37. https://nebius.com/prices
39. https://getdeploying.com/scaleway
40. https://www.ovhcloud.com/en/public-cloud/gpu/
43. https://computeprices.com/compare/genesis-vs-google
45a. https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html
45c. https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html
45d. https://docs.nvidia.com/cuda/archive/13.0.3/pdf/CUDA_Toolkit_Release_Notes.pdf
47. https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/
48. https://www.nsc.liu.se/support/systems/berzelius-resource-allocation/ ; https://supr.naiss.se/round/berzelius2026/
49. https://wasp-sweden.org/updates-in-the-wara-ops-portal/ ; https://wara-ops.org/
50. https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-ai-science-and-collaborative-eu-projects_en
51. Vast public offer API, snapshot 2026-11-17 (queries per GPU model × `num_gpus` 1/4/8 × on-demand / bid; fields
    `dph_total`, `min_bid`, `cpu_ram`, `cpu_cores_effective`, `disk_space`, `disk_bw`, `inet_down/up`, `pcie_bw`,
    `pci_gen`, `driver_version`, `cuda_max_good`, `gpu_mem_bw`, `verification`, `hosting_type`, `storage_cost`,
    `internet_*_cost_per_tb`): https://console.vast.ai/api/v0/bundles/
52. https://docs.vast.ai/guides/instances/choosing/instance-types
53. https://vast.ai/article/Rental-Types
54. https://docs.vast.ai/documentation/reference/faq/instances
55. https://docs.vast.ai/documentation/instances/storage/types ; https://docs.vast.ai/guides/instances/storage/volumes
56. https://docs.vast.ai/guides/instances/storage/data-movement ; https://docs.vast.ai/documentation/instances/storage/cloud-sync
57. https://docs.vast.ai/documentation/templates/template-settings ; https://docs.vast.ai/api-reference/creating-instances-with-api
58. https://docs.vast.ai/documentation/instances/templates/docker-environment
59. https://docs.vast.ai/guides/reference/faq/security ; https://vast.ai/article/security-and-compliance-at-vast-ai
60. https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html ;
    https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
61. https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html ;
    https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
62. https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776 ;
    https://github.com/tinygrad/open-gpu-kernel-modules/issues/29 ; https://github.com/verl-project/verl/issues/2803
63. https://docs.nvidia.com/cuda/archive/8.0/cuda-c-programming-guide/index.html (Unified Virtual Address Space:
    "allocations via cudaHostAlloc() are automatically portable") ;
    https://docs.nvidia.com/cuda/archive/8.0/cuda-runtime-api/group__CUDART__UNIFIED.html
64. https://docs.vast.ai/cli/reference/stop-instance
65. https://fast.io/resources/vast-ai-review-2026/ (third party: community hosts not independently audited)
