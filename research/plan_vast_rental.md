# 8× RTX 5090 rental (24–48 h): scenarios, readiness, work split, speed-ups (2026-11-27)

Use case: LBM flow libraries and 4D-flow / B0 k-space scans for a set of scenarios — the healthy phantom, aortic variants
(diffeomorphisms), valve disease — on one rented 8× RTX 5090 node (Vast.ai, no NVLink / P2P). Builds on the
2026-11-17 audit in plan_cluster.md §8 (sm_120 builds, host resources, cost, data governance), which still holds:
nothing of its action list A1–A11 has been built since.

## 0. Answer

- **Decision (user, 2026-11-27): one simulation uses all GPUs** (not 8 scenarios side by side), host RAM ≤ 256 GB,
  and **every ADC sample is recorded** (readout, velocity encodes, rewinder / read-in, spoiler tail): `--nav off` is
  not an option (the navigator data are for later research).
- **k-space scan over all GPUs: built** (`simphantom --gpus 0,1,…`, default all visible devices, 2026-11-27): one
  worker thread per GPU takes 250-bin chunks from a shared queue; host tables and the beat-library cache are shared
  (one copy in RAM), each GPU holds its own uploads; rows are written at their offsets (`pwrite`), chunk markers keep
  the run resumable. 240 chunks for a 10-min scan → 30 per GPU, dynamic balance. Equivalence test against the
  single-GPU binary: §2.1.
- **LBM library over all GPUs: not built.** Easy route: an ensemble of forks from one warmed checkpoint (§4.4).
- **Still missing:** a site profile / launcher / container (plan_cluster A1, A2), a sync tool (A6).
- **Efficiency today** (measured on the 4090):
  - *LBM* ≈ 620 MLUPS at 1 mm (≈ 1000 at 0.5 mm) with the moving heart, against 5600 MLUPS (FP32 storage) / 11 100
    (FP16) for a plain box in FluidX3D [1]. The geometry update every 4 steps is 35 % of the time.
  - *Direct sum* ≈ 1–1.9·10¹² coil-terms / s ≈ 22 clock cycles per (sample, source) pair per SM sub-partition,
    against ≈ 5 for an FMA-bound kernel (128 coil FMAs + ≈ 35 phase FMAs per pair at 32 coils): ≈ 23 % of FP32 peak.
    Nsight Compute profile: §4.2. The navigator ADC is 24 % of the time and stays on.
- **Hardware specificity: low.** Every kernel is NVRTC-compiled at run time for the device's own arch (sm_120 cubin on a
  5090), with shared memory queried at run time and no hard-coded arch (plan_cluster §8.1). Neither kernel uses tensor
  cores. The LBM is memory-bandwidth-bound (5090/4090 ≈ 1.65×); the direct sum is FP32/SFU-bound (≈ 1.3×).
- **Scenario cost** (5090 estimates, see §3): one GPU-scenario ≈ 15–16 GPU-h (phantom 1 h CPU, LBM 1 h, 10-min flow
  scan ≈ 12 h with every ADC sample of the v2 design, 2.5-min B0 scan ≈ 2 h). On 8 GPUs one scenario's scans take
  ≈ 1.8 h wall; the LBM library ≈ 1 h until the ensemble split exists (then ≈ 15 min). ≈ 2.5–3 h per scenario → 8–9
  scenarios in 24 h. The direct sum is near its FP32 limit on GeForce (§4.2: Nsight, register blocking 1.035×), so
  more GPUs, not kernel work, shorten the scan part.

## 1. Scenarios and their readiness

| scenario | how | state |
|---|---|---|
| healthy (seed 1 test phantom) | `make_phantom 1 --clearance` | full16: 183 vessel pairs (≤ 1.77 mm), 12 tissue contacts (≤ 0.31 mm); LBM gates pass at 1 and 0.5 mm with the wall pump |
| habitus (body shape) | `--diffeo habitus` | production path cleared (373 pairs on the full11 graph); not rerun with today's rules |
| ascending aortic aneurysm | `--diffeo aneurysm` | production path with adaptive carve rounds: 230 pairs (≤ 3.59 mm, the pulmonary bifurcation), 29 tissue (≤ 0.43 mm) |
| supravalvular narrowing + dilated pulmonary trunk | `--diffeo narrow` | preset exists; not run through the production path |
| coarctation | `narrow` feature on the aortic isthmus (spec file) | no preset yet (small) |
| mitral regurgitation | `--valves mr=0.3` (+ `--diffeo lv_overload` for a realistic forward stroke) | LBM checked at 1 mm (mr 0.2, regurgitant volume ≈ 30 mL as the model); `lv_overload` never run end to end |
| aortic stenosis | `--valves as=1.0` | LBM checked at 1 mm (jet 5 m/s, gates pass) |
| bicuspid aortic valve (± stenosis) | `--valves bav=0.1,as=1.2` | geometry checked (open area); no LBM run |
| aortic regurgitation | `--valves ar=0.3` | network + model implemented; no LBM run |
| mitral / tricuspid stenosis | `--valves ms=1.0` | model implemented; no LBM run |
| arrhythmia | — | design only (flow_phantom.md "Arrhythmias") |

Known measurement caveat: the report's material-disk valve flux over-reads a regurgitant jet ≈ 1.4×, so the mitral
gate of a regurgitant case is not meaningful; the cut balances and the mitral-plane link flux are.

## 2. Pipeline per scenario and its parallel shape

| step | tool | 4090 time (measured) | GPUs | host RAM | notes |
|---|---|---|---|---|---|
| phantom (graph, network, motion, clearance with carve rounds) | `make_phantom … --clearance` | 54 min (full16, 10-core quota) | 1 (torch, clearance audits) | ≤ 40 GB | CPU-heavy: on a 144–320-core host 8 builds in parallel fit |
| LBM case | `lbm_case.py` | ≈ 5 min CPU | 0 | ≈ 15 GB | |
| LBM library (0.5 mm, 15 beats + 3 warm-up) | `cardiac_lbm --beats --spin` | 72 min | 1 | small | 26 GB output |
| coronary library | `lbm_coronary.py` + `cardiac_lbm` | ≈ 40 min | 1 | small | |
| blood frames + prep | `phantom_4d.py`, `prep_sim.py` | ≈ 6 min | 1 / 0 | ≤ 48 GB | |
| 10-min 4D-flow scan (2 mm) | `simphantom --seq flow` | 11.5 h at 350 W (172 s per 250-bin chunk), v1 readout samples only | 1 | **84 GB** with `--beat-cache 15` | v1 run of 2026-11-08 predates the navigator ADC; with every sample of the v2 design: ≈ 15.7 h (below) |
| 2.5-min B0 / ratemap scan | `simphantom --seq b0` | 2.5 h | 1 | 84 GB | |
| truths, bias | `sim_truth`, … | minutes | 1 | | |

`rustlib/tools/phantom/realistic_scan.sh` chains everything from the LBM case on (resumable, a STOP file / deadline ends
it between steps). It has not been run on the current phantom generation (cleared and carved atlas, wall pump, new cut
rules): **a local dry run with a short scan is the first readiness item** (§5 P1).

**Work split for the rental: one simulation over all GPUs** (user decision, 2026-11-27).
- Scans: `simphantom --gpus all` (§0). Host RAM is one process's: ≈ 84 GB with `--beat-cache 15` today, plus
  ≈ 0.3 GB per worker for its chunk buffers → ≈ 90 GB on 8 GPUs, inside the 256 GB limit. Set `--beat-cache` to the
  library's beat count: the workers are at different scan times and draw different library beats.
- The beat moments (CPU, rayon) are computed under one lock: at the start the workers wait for the first beats
  (≈ one computation per library beat, once per run); a prefill of all library beats in parallel is a small follow-up.
- LBM: one library over all GPUs needs the ensemble fork (§4.4); until then, one scenario's LBM per GPU while another
  scenario's scan uses the others is the fallback.
- Phantom builds are CPU work (clearance, carving); they overlap with GPU steps of other scenarios.

### 2.1 Multi-GPU scan: equivalence test

Dev input `sim_flow_dev_respfield` (3.5 M tissue sources, 32 coils), 1 s of scan in 40-bin chunks, resp field and
navigator ADC on; scripts `jobs/sim/mgpu_test.sh`, `mgpu_test2.sh`; "2 workers" = two workers on the one local GPU.

| comparison | max \|diff\| / rms | rms diff / rms | verdict |
|---|---|---|---|
| no beat library: pre-change binary vs `--gpus 0` | 0 | 0 | bit-identical |
| no beat library: pre-change vs `--gpus 0,0` | 0 | 0 | bit-identical |
| beat library: pre-change binary vs itself (two runs) | 4.3e-5 | 3.7e-7 | run-to-run spread of the existing code |
| beat library: pre-change vs `--gpus 0` | 4.3e-5 | 3.7e-7 | same spread |
| beat library: `--gpus 0` vs 2 workers stopped (one hit a CUDA OOM: two full copies on one 24 GB card) and resumed with 1 | 4.3e-5 | 4.1e-7 | same spread; resume across worker counts works |

- The spread with a beat library was old: `bsig` packed the blood sub-sources through an atomic counter (slot order
  arbitrary), so the per-thread f32 sums over ≈ 10⁵ sources rounded differently each run. **Fixed (2026-11-27):**
  `bsig_nk` + a device exclusive prefix sum give every source its slots in source order (`SIMPH_BSIG_ATOMIC=1`: the
  old counter order); self-checks: the slot total per bin, and the prefix sum of each worker's first four slots.
  Tests (`jobs/sim/bsig_det_test.sh`): beat library 1 vs 2 workers bit-identical, `--blood-blend` twice bit-identical,
  det vs atomic order rms 1.2e-6 of rms (f32 rounding), same packed count, same speed. So every simphantom run is now
  bit-reproducible and independent of the worker count.
- Timing: 2 workers on one GPU 35.2 s vs 1 worker 35.0 s (no queue / write overhead); the pre-change binary 34.1 s.
- On a real node each worker has its own GPU; VRAM per GPU is as for one process today.

**Samples per TR (2 µs dwell, GE pge2 minimum):**

| design | TR | encode (nav) | readout ADC | tail (nav) | total ADC | 10-min readouts | samples / 10 min |
|---|---|---|---|---|---|---|---|
| v1 `flow_thorax` (the 11.5 h run, readout only) | 11.29 ms | 685 | 3990 (7.98 ms) | 720 | 5395 (10.79 ms, 96 %) | 53 144 | 212 M simulated (287 M with navigators) |
| v2 `flow_thorax_v2` (next sims) | 14.10 ms | 685 | 5395 (10.79 ms: 4745 recon + 650 rewinder nav) | 720 | 6800 (13.6 ms, 96 %) | 42 553 | 289 M |

The direct sum scales with the samples: a v2 10-min scan with every sample ≈ 289 / 212 × 11.5 h ≈ **15.7 h on the 4090**
at 350 W.

## 3. Estimated scenario cost on a 5090

Factors (plan_cluster §8, to be measured on the first rental): LBM ×1/1.65 (bandwidth), direct sum ×1/1.3.

| step | today's code | with P3–P5 (§4) |
|---|---|---|
| phantom build | ≈ 1 h | ≈ 1 h |
| LBM library + coronary | ≈ 1.1 h | ≈ 0.7 h (geometry every 16 steps) |
| 10-min v2 flow scan, every ADC sample (GPU-hours; ÷ 8 on all GPUs) | ≈ 12 h | ≈ 12 h (register blocking measured 1.035×) |
| B0 scan | ≈ 1.9 h | ≈ 1 h |
| total | ≈ 16 h | ≈ 15 h |

Over 8 GPUs: ≈ 16 GPU-h → ≈ 2–2.5 h wall per scenario (LBM ensemble built) or ≈ 3 h (LBM on one GPU, others busy
with scans). 24 h ≈ 8–16 scenarios; 48 h ≈ 16–30, or 1 mm scans of a few.

## 4. Performance: what to improve

### 4.1 LBM (measured split, 1 mm, one cycle of 53 k steps: 55 s)

| item | gain | effort | notes |
|---|---|---|---|
| geometry update every 16 steps instead of 4 | **1.5×** (55 → 36 s measured) | trivial (`run.geom_every`) | the wall moves ≲ 0.1 lattice units per 16 steps at 0.5 mm (≈ 0.1 m/s × 16 × 16 µs); validate flows / gates against 4 |
| FP16 population storage (FluidX3D's FP16S: shifted DDFs, FP32 arithmetic) | up to ≈ 1.8× on the step | medium (kernel + both backends; GPU = CPU test) | [1]: 5624 → 11 091 MLUPS on a 4090; accuracy shown adequate for low-Mach flows |
| leaner per-node traffic | ≈ 1.3–2× on the step | medium | today per step and node: 2 × 19 populations, the macroscopic ρ, u of the previous and current step, wall velocity, q fractions, flags; FluidX3D moves 153 B (FP32) per step. Profile with ncu (works non-admin) before choosing |
| higher lattice Mach (u_lattice 0.08 → 0.12) | 1.5× fewer steps | trivial | compressibility error ∝ Ma²; check jets / valve gates; the cumulant operator (§4.3) makes it safer |
| WALE off | 1.04× | trivial | keep: it stabilises the jets |
| AA / esoteric-twist streaming (one population buffer) | memory ÷ 2, speed ≈ same | medium | only needed for larger domains (whole torso at 0.5 mm) |
| grid refinement (0.25 mm near valves / jets, 1 mm in the trees) | large for jets | large | multi-block refinement with τ scaling; not before the rental |

### 4.2 Direct sum (the k-space time)

The algorithm is the right one: within a 10 ms bin the sources are fixed and a bin holds about one readout (≈ 4000
samples), so a type-3 NUFFT would save little per bin (O(N_src W³) spreading against O(N_src N_samples)) and the
per-sample physics (T2*/ratemap decay, off-resonance, velocity phase, box sinc) breaks its Fourier structure. The levers
are the kernel and the term count:

| item | gain | effort | notes |
|---|---|---|---|
| register blocking: 2–4 samples per thread (`SIMPH_DSUM_RB`, built 2026-11-27) | **measured: RB 2 1.035×, RB 4 0.32×** | done | bit-identical to `dsum`; with 32 coils RB 2 needs 208 registers (occupancy 16 → 8 warps / SM) and RB 4 spills (255 + 488 B stack); kept off by default |
| virtual coils (`--sim-coils auto`) | **none for this array** | none (exists) | auto (ε = 1e-8) kept all 32 coils on the 1 mm box run (`sim_coils: 32`); looser ε trades accuracy |
| octree / box-element sources for static tissue (`--octree`) | 2–4× fewer tissue sources | none–small (exists; benchmark) | 11.9 M tissue sources at 1 mm / 2 mm today; exact box-integrated elements |
| tensor-core coil accumulation (phasor tile × amplitude tile, FP32 accumulate) | **none on 4090 / 5090**; large on A100 / H100 | large | see below |
| half-split / f64 atomics | small | small | the f64 atomics at the end are not the limit |

**Nsight Compute profile of `dsum` (2026-11-27, 4090, 32 coils, a readout and a navigator launch, `jobs/sim/ncu_dsum.sh`):**

| metric | readout | navigator |
|---|---|---|
| issue slots busy | 71 % | 72 % |
| FMA pipe / LSU (shared loads) / XU (sincos, exp) | 55 % / 60 % / 10 % | 56 % / 60 % / 10 % |
| registers, occupancy | 128, 33 % (16 warps / SM) | same |
| stalls | not selected 36 %, selected 18 %, dispatch 16 %, long scoreboard 8 %, barrier 5 % | similar |
| shared-memory bank conflicts | 0 | 0 |

Per warp and source ≈ 215 instructions issued, of them ≈ 142 FFMA (128 for the 32 complex coil multiply-adds, the
minimum in FP32), ≈ 45 shared loads, the rest integer / control; ≈ 303 cycles. So the kernel is issue-bound with
FMA and shared loads at about half throughput, not SFU- or memory-bound; the FP32 floor is 142 cycles (≈ 2.1×), of
which only ≈ 1.4× is reachable without removing FFMAs. Packed FP32×2 FMA (FFMA2, which would halve the coil loop's
issue slots) exists only on sm_100 / sm_103; on sm_120 (5090) ptxas splits it into two FFMAs [7].

**Tensor cores on GeForce (datasheets [4], [5]; measured 5090 ratio 0.54 [6]):** FP16 tensor with FP32 accumulate
runs at half the FP16-accumulate rate on GeForce: 4090 165 TFLOPS against 82.6 FP32, 5090 209.5 against 104.8, i.e.
only 2× the FP32 cores. The coil sum is a complex GEMM (samples × 2·sources) · (2·sources × 2·coils); keeping today's
≈ 1e-6 accuracy needs split FP16 (hi / lo, 3 MMAs, as sample_kernels.md §21), i.e. 3 × the work at 2 × the rate =
1.5× *slower* than an FMA-bound coil loop. Plain FP16 inputs (1 MMA, 2× faster on that part) leave ≈ 3e-4 relative
error per term, above the 1e-4 noise-free floor the inverse-crime controls need. FP16 accumulation is excluded. So on
the rental's GPUs neither tensor cores nor register blocking (measured 1.035×) buy much; on A100 / H100 (FP32-
accumulate tensor rate 16× / 15× the FP32 cores) the 3-MMA split would make the phase arithmetic the limit (≈ 3–4×).
Worth building only for a datacenter-GPU target.

### 4.3 More advanced LBM variants

| variant | what it buys us | cost |
|---|---|---|
| **cumulant collision** (Geier et al.; cascaded / central-moment family) | the most stable operator at τ → ½; robust stenotic and regurgitant jets at coarser lattices and higher lattice Mach; no LES tuning; same memory traffic (bandwidth-bound kernel, extra FLOPs free) [2] | medium: new collision in lbm_core.h; validate Poiseuille, Womersley, jets |
| TRT with "magic" Λ = 3/16 | wall position of bounce-back independent of viscosity; cheapest accuracy fix at the walls | small |
| recursive regularized (already regularized + bulk relaxation) | marginal | — |
| He–Luo incompressible equilibrium | removes the O(Ma²) density-coupling error in the pressure / mass balance; helps the chamber pump (compressibility stored ±few % of mass) | small |
| entropic KBC | stability like cumulants; less accurate in shear | medium |
| FP16 DDF compression (FluidX3D) | bandwidth ÷ 2 | medium |
| mass-conserving moving boundaries | removes the need for the wall pump's correction (the bounce-back injected 0.83–0.89 of the stroke) | large (research) |
| HemeLB-style LES (Smagorinsky) at scale [3] | we use WALE already | — |

Recommendation: before the rental only the trivial item (geometry interval); after it,
cumulant + FP16 storage as one LBM work package, validated on the healthy and stenosis cases.

### 4.4 LBM over several GPUs

| route | speed-up on 8 GPUs | effort | notes |
|---|---|---|---|
| **ensemble of forks**: warm up once (3 cycles), checkpoint, then each GPU resumes the checkpoint with a seeded perturbation and writes its own library cycles (beat_<g>_<cycle>) | ≈ 15 + 3 → 3 + 2 cycles of wall time ≈ **3.6×** for 15 beats, more for longer libraries | small–medium: a fork driver (thread per GPU), unique cycle ids, controller state in the checkpoint (pump gain is not yet) | scans draw library beats at random (`beat_seed`), so consecutive library cycles were never used as a continuous series; forks are as valid. If the flow is nearly periodic the forks converge to the same beat: then fewer beats are needed anyway |
| **domain decomposition** (slabs of tiles, halo populations exchanged every step through pinned host memory; no P2P on Vast) | ≈ 3–5× at 0.5 mm, less at 1 mm | large: ghost tiles in every kernel and both backends, per-step D2H/H2D, global reductions (cuts, pump, Windkessel), load balance with the moving heart | 0.5 mm ≈ 3–4 M fluid nodes, ≈ 2.3 ms per step on the 4090 → ≈ 0.3 ms per GPU vs ≈ 0.1–0.2 ms halo round-trip + launch syncs; only worth it for one long beat or a domain that does not fit one GPU (whole torso / head at ≤ 0.5 mm) |

Recommendation: the ensemble fork before the rental; domain decomposition only when a single domain outgrows one
GPU's 32 GB.

## 5. Before renting (ordered)

| # | item | effort | why |
|---|---|---|---|
| P1 | **dry run of `realistic_scan.sh` on the current phantom generation**, 2 library beats, a 20 s scan, locally | ≈ 1.5 h GPU | the pipeline has not run since the clearance / carve / pump changes |
| P2 | scenario runner: the resumable steps of each scenario in order, each step on all GPUs (`--gpus all`), logs, STOP / deadline, plus the site env (plan_cluster A1) | small | scenarios back to back, unattended |
| P2b | LBM ensemble fork (§4.4) incl. pump gain in the checkpoint | small–medium | LBM ≈ 3.6× on 8 GPUs |
| P3 | `geom_every` 16: validate gates and flows against 4 at 1 mm, then make it the default | small + 2 GPU runs | 1.5× LBM |
| P4 | ~~register-blocked `dsum`~~ built and measured: 1.035× (§4.2); the direct sum is near its FP32 limit on GeForce — more GPUs are the lever | done | |
| P5 | scenario specs: coarctation preset, `lv_overload` + `mr` end to end, AR / BAV LBM bursts at 1 mm | ≈ 3–4 h GPU | §1 gaps |
| P6 | environment bootstrap script (venv with torch cu130, `cargo build --release`, data allowlist ≈ 10–20 GB: p0 atlas, Z-Anatomy export, seq designs, phantoms) | small | plan_cluster §8.4 |
| P7 | optional: 1× 5090 smoke test (plan_cluster A5, ≈ $2) | 2 h rental | measures the 5090 factors above |

## Sources

[1] FluidX3D README, benchmark table (RTX 4090: 5624 MLUPS FP32/FP32, 11 091 FP32/FP16S, 85 % of the bandwidth roofline; 153 / 77 B per step): https://github.com/ProjectPhysX/FluidX3D/blob/master/README.md ; RTX 5090 ≈ 18 456 MLUPS FP16S (user results): https://openbenchmarking.org/performance/test/pts/fluidx3d/deec21f23313b845450e67faf66183f2e66c849d
[2] MacLAB / review of cascaded and cumulant LBM stability at small viscosity: https://arxiv.org/pdf/1901.02716
[4] NVIDIA RTX Blackwell GPU architecture whitepaper (RTX 5090: FP32 104.8 TFLOPS; FP16 tensor 419 dense with FP16
accumulate, 209.5 with FP32 accumulate): https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
[5] NVIDIA Ada GPU architecture whitepaper (RTX 4090: FP16 tensor 330.3 / 165.2 with FP16 / FP32 accumulate, FP32 82.6): https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
[6] Measured FP32/FP16-accumulate tensor ratio 0.53–0.54 on 3090 / 4090 / 5090: https://www.julien.org/blog/industry-perspectives/2026-10-28_geforce-vs-workstation-gpus-nvidias-half-rate-priced/
[7] NVIDIA developer forum, "Does Blackwell SM 120 have native f32x2 support?" (FADD2 / FFMA2 SASS only on sm_100 / sm_103): https://forums.developer.nvidia.com/t/does-blackwell-sm-120-have-native-f32x2-support/344788
[3] Exascale-ready HemeLB for large-eddy simulation of full-human arterial blood flow: https://arxiv.org/pdf/2611.07840
