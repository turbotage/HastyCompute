# NAISS Small Compute 2026: proposal draft (to be rewritten by the PI)

Draft structured after the SUPR form fields. `[CHECK]` marks a fact or choice the PI must confirm. Every GPU-hour
number traces to a measured runtime on the development machine (RTX 4090, 350 W) in §6, with the scaling assumptions
stated there.

---

## 1. Title

Convergence and recovery guarantees for plug-and-play reconstruction with learned priors, validated on simulated
4D flow MRI

*(alternative, shorter: "Learned priors in compressed sensing: theory and large-scale validation in 4D flow MRI")*

## 2. PI and affiliation

- PI: Viktor Vigren, doctoral student, `[CHECK: department, university]` (WASP graduate school).
- Main supervisor (required for doctoral students): `[CHECK: name, affiliation]`.
- Other NAISS projects of the PI: `[CHECK: project id]` (geometric neural ODEs / PolyNODE). This proposal is a
  separate project; no resources are shared between the two.

## 3. Requested duration and start date

12 months, starting `[CHECK: e.g. 2026-12-01]`.

## 4. Research Council classification

`[CHECK against the SUPR list]` Primary: 10105 Computational Mathematics. Secondary: 20603 Medical Image Processing.

## 5. Abstract (English)

Plug-and-play (PnP) methods reconstruct images from undersampled measurements by alternating a data-consistency step
with a learned denoiser. They are among the most accurate methods in MRI, but their guarantees lag behind practice.
Classical compressed-sensing theory (restricted isometry, incoherence) assumes a fixed sparsifying basis. A learned
denoiser instead defines a signal model that changes along the iterates, and the published convergence bounds for
preconditioned PnP become vacuous in the undersampled regime.

This project develops (i) sharper convergence bounds for preconditioned PnP with quasi-Newton (ZMSHR1)
preconditioning, based on an exact spectral description of the preconditioner, and (ii) recovery guarantees for
learned priors stated through the local geometry of the denoiser: its Jacobian acts as a projector onto a tangent
space of the signal set. These yield sample-complexity and error bounds, with probability, that depend on measurable
quantities: tangent dimension, leakage, restricted conditioning and block coherence.

The theory is tested on a demanding application, time-resolved 3D phase-contrast (4D flow) MRI, where ground truth
is unavailable in vivo. We use a physics-based simulation pipeline: anatomical phantoms, lattice-Boltzmann blood flow,
and a full MR signal model with 32 receive coils, off-resonance and gradient nonlinearity. On it we measure the
theoretical quantities for trained denoisers and test the falsifiable predictions of the theory. The simulation and
reconstruction codes are our own GPU implementations (Rust/CUDA).

`[CHECK: ~200 words; rewrite in your own voice]`

## 6. Resource usage and justification

### 6.1 Resources

| Resource | Request | Notes |
|---|---|---|
| Arrhenius GPU | **300 GPU-h/month** | within the 500 GPU-h/month Small limit |
| Arrhenius CPU | default / none `[CHECK]` | CPU work runs inside the GPU jobs |
| Arrhenius Disk | **2,000 GiB** | ≤ 5,000 GiB limit; data management plan in §8 |

### 6.2 Measured runtimes (development machine, one RTX 4090 at 350 W)

| Task | Measured | Source |
|---|---|---|
| k-space simulation, 10-min 2 mm scan, 60,001 time bins, 32 coils | 172 s per 250 bins → **11.5 h** | `realistic_scan.sh` logs, phantom ph0001 |
| k-space simulation, 1 mm small-FOV scan | ≈ 2.3 s per bin → **38 h** | same driver, `--res 0.001` |
| Lattice-Boltzmann cardiac flow, 0.5 mm | 277 s per 5 cardiac cycles | `cardiac_lbm` |
| Reconstruction, 2 mm, 280 sets (40 frames × 7 encodes), 31 virtual coils | setup ≈ 8 min, **≈ 9 min per iteration** | `recon_p2np` logs |
| Denoiser training (3D patch network, 32³ × 12 frames) | 2.9 iterations/s | training logs |

### 6.3 Conversion to GH200 GPU-hours (assumptions)

- **Simulation** is FP32/SFU-bound. A GH200 has 67 TFLOPS FP32 and the 4090 83 TFLOPS. Assume **1.1 GH200-h per
  4090-h**.
- **Reconstruction and LBM** are memory-bandwidth-bound and stream data from host memory. The GH200 has 4 TB/s versus
  1 TB/s, plus NVLink-C2C to host memory. Assume a conservative **2.5× speed-up**.
- **Training** uses tensor-core convolutions. Assume **2× speed-up**.

### 6.4 Monthly plan

| Activity | Count per month | GPU-h each | GPU-h/month |
|---|---|---|---|
| Simulated 2 mm scans (training phantoms with tissue / physiology variation, held-out phantom) | 4 | 12.5 | 50 |
| Simulated 1 mm scans (coronary-scale resolution) | 1 | 42 | 42 |
| LBM flow libraries (new phantoms / pathologies) | 3 | 3 | 9 |
| Reconstruction studies at 2 mm (preconditioner variants, cold vs warm start, priors, convergence measurements) | 30 | 2.2 | 66 |
| Reconstructions at 1 mm | 4 | 8 | 32 |
| Denoiser training runs (≈ 100k iterations incl. validation) | 6 | 8 | 48 |
| Theory measurements: Jacobian probes (tangent dimension via Hutchinson trace estimates, leakage, restricted conditioning, coherence) | — | — | 20 |
| Porting and benchmarking (aarch64 build, multi-GPU validation; mainly month 1) | — | — | 10 |
| **Total** | | | **≈ 280 → request 300** |

### 6.5 Efficient use
- Jobs are interruptible and resumable: per-iteration checkpoints, a STOP file, and SIGUSR1 handled for Slurm
  `--signal`. Preempted or time-limited jobs lose at most one iteration.
- Measured k-space data is compressed and streamed from host memory (the GH200's C2C link is a good fit). Coil
  compression keeps 99.9 % of the energy (32 → ~31 virtual coils at 2 mm).
- GPU kernels are compiled at run time for the exact architecture (sm_90) and cached.
- Jobs start on single GPUs (Arrhenius nodes are shared). 4-GPU jobs follow once multi-GPU support, currently in
  development, has been validated.

## 7. Software and licences

- **Own code:** Rust + CUDA (kernels compiled at run time with NVRTC from the CUDA toolkit) and Python tools. Licence
  MIT OR Apache-2.0 `[CHECK: the repository is not public yet; say so or publish]`.
- **Third party:**
  - PyTorch (BSD) for denoiser training and inference.
  - NumPy / SciPy / Matplotlib.
  - Apptainer containers built from public CUDA base images.
  - No licensed commercial software is needed.
- **Architecture:** aarch64 builds of the code are needed for the GH200 nodes. The Rust code cross-compiles, and
  PyTorch provides aarch64 CUDA wheels.

## 8. Storage and data management plan

- **Volumes:** per phantom, the flow library is about 70 GB (LBM beat library 59 GB, blood-tracer library 12 GB). A
  simulated 10-min scan is 74 GB, and a reconstruction output about 10 GB. With 3–4 phantoms active plus
  reconstruction studies: **≈ 1.5–2 TB**.
- **Regenerability:**
  - Everything stored is derived data, regenerable from the code version, the phantom parameters and the random
    seeds.
  - Primary data is the code (version-controlled in git) and the small configuration / parameter files, which are
    backed up outside NAISS.
  - Large intermediates on Arrhenius therefore need no backup.
- **Downstream:**
  - Final results (velocity fields, figures, metrics; a few GB per study) are copied to the PI's institutional
    storage `[CHECK]`.
  - Results that support publications are deposited with the papers (e.g. Zenodo).
- **Removal:**
  - Intermediates are deleted when a study ends (reconstruction caches, k-space stores).
  - All project data is removed at the end of the project.

## 9. Personal data / GDPR

- **All data processed in this project is synthetic.** It is generated from public anatomical atlas meshes
  (Z-Anatomy / BodyParts3D, CC BY-SA) and computed flows. **No personal data is processed.** No data processing
  agreement is needed.
- `[CHECK / DECIDE]` RACLETTE (Dirix, Buoso, Kozerke, ETH Zürich; CC BY-NC 4.0) is synthetic CFD flow embedded in
  in-vivo image backgrounds. Two options:
  - Train only on our own phantoms on Arrhenius. Then the statement above stays true.
  - Check with the dataset authors / your university's data protection officer whether the backgrounds count as
    personal data, and declare them if so.
- No real patient data. Real scanner data would need a separate proposal on the sensitive-data partition (Arrhenius
  SENS).

## 10. Expected outcomes
- **Paper 1:** sharpened convergence analysis for preconditioned PnP (exact ZMSHR1 spectrum, restricted contraction,
  and why tight preconditioner bounds slow convergence in practice).
- **Paper 2:** recovery guarantees for learned priors through denoiser tangent geometry, with numerical verification
  of the predicted quantities.
- **Software:** an open GPU simulation and reconstruction pipeline for 4D flow MRI with synthetic ground truth.

`[CHECK: keep only the outcomes you actually intend]`
