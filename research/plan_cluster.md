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
