# Synthetic 4D-flow pipeline: anatomy → CFD → MR truth simulation → reconstruction benchmark → 4D-flow denoiser

Status: design (2026-11-02). Nothing here is implemented yet.

Goal: a set of synthetic "patients" with known ground truth (anatomy, vessel tree, time-resolved velocity field, B0 field, coil sensitivities, gradient nonlinearity (GNL), concomitant fields), from which we
1. simulate spiral 4D-flow raw data at higher fidelity than any reconstruction model (no inverse crime),
2. benchmark reconstructions (P²nP with dynamic preconditioning, fidelity schedules, physics on/off),
3. train a purpose-built 4D-flow denoiser (4DFlowNet-like) that runs inside the reconstruction as the plug-and-play prior.

The same anatomy + flow sets feed both the denoiser training and the reconstruction benchmark, split **by subject** so the benchmark never sees a training geometry.

---

## 0. What exists that we can reuse (checked 2026-11-02)

| resource | what it gives | licence | use |
|---|---|---|---|
| AVT, Aortic Vessel Tree CTA (Radl et al. 2022, figshare 14806362) | 56 CTA scans, expert aorta + branch masks (arch branches, abdominal aorta, iliacs) | CC BY 4.0 | primary vessel trees with matching body CT |
| TotalSegmentator (Wasserthal et al.; model Apache-2.0, data CC BY 4.0, Zenodo) | 1228 CT subjects with 117 labels (lungs, heart, aorta, carotids, subclavians, SVC/IVC, trachea, bones, organs…); the model itself can be run on AVT scans | data CC BY 4.0, code Apache-2.0 (`heartchambers_highres` and some subtasks need a licence) | whole-body context labels that match the vessel tree (run on the AVT CT scans) |
| Vascular Model Repository (SimVascular) | ~150 projects: image + centerlines + segmentation + model + mesh, ~110 with CFD results | licence not stated on the pages found, check before use | extra geometries, CFD cross-checks |
| RACLETTE (Dirix, Buoso, Kozerke, ETH, 2026) | public paired HR/LR synthetic aortic 4D flow (healthy + graded stenosis), OpenFOAM CFD embedded in in-vivo backgrounds | gitlab.ethz.ch/ibt-cmr/publications/raclette, licence not stated | external test set for the denoiser; sanity check of our CFD statistics |
| 4DFlowNet (Ferdian et al. 2020, MIT) + Ensemble4DFlowNet (Ericsson et al. 2024) | residual CNN for SR/denoising of (magnitude, velocity), pretrained weights (aorta, cardiac, cerebro), example data | MIT | architecture baseline, pretrained baseline denoiser |
| Saitta et al. 2023, 4D inflow-profile SSM | 437 synthetic time-resolved aortic inlet velocity profiles + scripts | GitHub (check licence) | inlet boundary conditions with realistic skew/helicity |
| Pajaziti et al. 2023 | SSM of coarctation aortas, 3000 synthetic shapes + CFD (code on GitHub) | check | geometry augmentation (pathology) |
| CMRsim (Weine, McGrath et al. 2024) | GPU particle Bloch simulator incl. turbulent flow (≈1.5 M particles, 7 encodings, 29 min on a Titan RTX) | open source (Python/TensorFlow) | independent cross-check of our MR simulator on a few cases |
| Divergence-free wavelet denoising (Ong et al. MRM 2015) | classical velocity-domain denoiser, code (MATLAB) | Lustig lab page | classical baseline prior |
| FDA nozzle benchmark | geometry + inter-laboratory PIV for Re 500 – 6500 | public | CFD validation |
| XLB (Autodesk, Apache-2.0, JAX/Warp) / OpenFOAM / svMultiPhysics | existing CFD solvers | permissive / GPL / BSD-like | reference solutions; not the production solver (see §2) |

Not available publicly: raw multi-coil 4D-flow k-space; vendor GNL coefficients (Siemens `coeff.grad` is proprietary; it is on every scanner under `C:\MedCom\MriSiteData\GradientCoil\`). XCAT/MRXCAT require a commercial XCAT licence.

---

## 1. Anatomy ("patients")

**Per subject we need** on one physical grid with a defined scanner frame (orientation + isocentre offset):
- tissue label volume: body outline, air (lungs, trachea, bowel gas), fat, muscle, bone, myocardium, blood pools, organs;
- vessel lumen as a signed distance function (SDF) at ≤ 0.5 mm, plus centerlines and named inlet/outlet planes;
- property maps: proton density, T1, T2, T2*, susceptibility χ, fat fraction / chemical species.

**Source**: AVT CTA (vessel trees) + TotalSegmentator inference on the same CT (context labels) → the vessel tree matches the rest of the anatomy by construction. CT HU also gives fat (−190…−30 HU), air and bone directly, which avoids the licensed TotalSegmentator `tissue_types` subtask. Optional extra subjects: TotalSegmentator's own CT set (aorta + arch branch labels at lower fidelity), VMR.

**Geometry processing** (offline, Python is fine here: TotalSegmentator, VMTK for centerlines):
mask → upsample + smooth → SDF → centerlines (VMTK) → inlet plane at the aortic root, outlet planes perpendicular to the centerline at each branch truncation (with short flow extensions in the CFD domain only).

**MR appearance** (for the static background): label → (PD, T1, T2*) + within-label texture taken from the CT HU deviations (random gain/sign per label) + smooth bias field. The contrasts are randomised per subject (SynthSeg-style domain randomisation) so a learned denoiser cannot overfit to one synthetic look. Blood signal: steady-state spoiled-GRE value with an inflow-enhancement factor depending on through-plane velocity (Eulerian mode), or explicit magnetisation history (particle mode, §3.4).

**Regions outside the CFD domain that contain blood** (LV, pulmonary artery, veins): they appear in the image, so they need some velocity. Plan: smooth random divergence-free low-speed fields (or a second, cheap CFD run on those vessels) and mark them "unknown" so that neither the denoiser loss nor the benchmark metrics use them.

**Augmentation for training diversity**: random smooth diffeomorphisms of the aorta (radius ±15%, curvature), local stenosis (coarctation), dilation (aneurysm), bicuspid-like eccentric jets (via inlet profile), pathology shapes from the coarctation SSM.

**B0 from susceptibility**: χ from labels/HU (tissue −9.05 ppm, air +0.36, fat ≈ −8.4, bone ≈ −11.3, lung parenchyma as an air/tissue mixture), body padded beyond the CT FOV, then the dipole convolution ΔB/B0 = F⁻¹[(1/3 − k_z²/|k|²) F χ] (our FFT) in the **physical** frame (B0 along scanner z), then a simulated shim (fit and remove the 1st/2nd-order spherical harmonics inside the shim box) → realistic residual off-resonance (lung/heart interfaces, ±100–300 Hz at 3 T). Fat chemical shift as a separate species (−3.4 ppm, optional multi-peak).

**Coils**: Biot–Savart loops (anterior + posterior arrays, e.g. 18 + 12 elements) placed on the body surface, quasi-static B1⁻; noise covariance Ψ from a coupling model (correlations 0.1–0.4). The reconstruction estimates its maps from a simulated calibration scan (ESPIRiT), never from the truth maps.

---

## 2. CFD

**Regime**: ascending aorta peak Re ≈ 3000–7000, Womersley α ≈ 10–20; stenotic jets up to 4 m/s and turbulent. Rigid walls first (as in 4DFlowNet and RACLETTE); wall motion later.

**Recommendation: own GPU lattice-Boltzmann solver in Rust + NVRTC** (same codegen route as the FFT/NUFFT crates), with existing solvers only as references. Reasons:
- geometry is already an SDF on a voxel grid → no meshing pipeline; interpolated bounce-back (Bouzidi) uses the SDF wall distance directly;
- throughput: estimated ≈ 7 min per pulsatile simulation (3 cycles) for a thoracic aorta at 0.4 mm (≈ 4 M fluid nodes in sparse 8³ tiles, ≈ 110 k steps at u_lattice ≤ 0.1, ≈ 1.5–2 G node-updates/s for D3Q27 fp32 on the 4090; to be measured) → hundreds of flow configurations are feasible; stenotic jets ≈ 3× more;
- the CFD can run coupled with the MR simulator (particles advected inside the solver) when we need turbulence-resolved magnetisation history without storing terabytes of snapshots.
FluidX3D is non-commercial-only (and OpenCL); XLB is Apache-2.0 and a good cross-check; OpenFOAM/svMultiPhysics for a FEM/FVM reference on one geometry.

**Numerics**: ν_lattice ≈ 5·10⁻⁴ (τ − ½ ≈ 2·10⁻³) is far into the under-resolved regime → BGK is unusable; use a cumulant or recursive-regularised / central-Hermite-moment collision with Smagorinsky LES (the FDA-nozzle LBM study of Huang et al. 2022 used central Hermite moments). Sparse tiles, AA-pattern in-place streaming, optional fp16 population storage.

**Boundary conditions**: inlet velocity from a waveform library (stroke volume 50–110 mL, HR 50–110, varied systolic shape) × spatial profile (parabolic / plug / Saitta SSM profiles with skew and helicity); outlets: 3-element Windkessel (RCR) per outlet integrated each step from the outlet flux, parameters tuned so flow splits (~30% supra-aortic) and pulse pressure are physiological. Newtonian first, Carreau–Yasuda optional.

**Validation gates** (must pass before data production): Poiseuille and Womersley in a straight tube (< 1% error), Dean flow in a curved pipe, FDA nozzle at Re 500/3500/6500 vs PIV, one aorta vs OpenFOAM or XLB (phase-averaged velocity and flow splits).

**Output**: phase-averaged velocity on the 0.4–0.5 mm grid every 5 ms (fp16, ≈ 2–4 GB per simulation, fluid nodes only) + per-voxel Reynolds-stress tensor for turbulent cases (intravoxel dephasing). Storing every LBM step is out of the question (≈ 10 GB per cycle at 2 ms); for turbulence-resolved particle simulation the CFD is re-run coupled (it is cheaper than the storage).

---

## 3. MR truth simulator

### 3.1 Sequence (physical frame)
Spiral 4D-flow GRE: per TR: RF (α), velocity-encoding bipolars (4-point referenced or balanced/Hadamard; VENC per anatomy, 150–200 cm/s aorta), spiral readout, spoiler. Trajectories: gradient-limited variable-density spiral design (Gmax, Smax) as stack-of-spirals or rotated 3D spirals/cones (golden-angle rotations; each readout is a linear transform of a base readout — the structure the operator work already exploits). The simulator keeps the **gradient waveforms G(t)**, not only k(t), in the logical frame plus the logical→physical rotation and the FOV offset. ECG gating with HR variability, retrospective binning (20–30 frames). Respiration: none at first (gated), 5D-flow later.

### 3.2 Physics, per readout sample at absolute time t and source point r with velocity v(r, t)
phase = γ[ M0(t)·u(r) + M1(t)·L(r)ᵀv + ½ M2(t)·a ] + φ_conc(r, t) + ω_B0(r)(t − t_exc) + φ_Maxwell,bipolar,e(r)
- M0, M1, M2: moments of the **whole** gradient waveform since excitation (velocity encoding *and* the spiral readout's own first moment → displacement/misregistration artefacts of long spirals are included to first order: at 1 m/s a 10 ms readout moves spins by 1 cm);
- u(r): GNL warp from spherical-harmonic coil fields; L(r) = ∂u/∂r also distorts the velocity-encoding direction and magnitude (the known GNL VENC error);
- φ_conc: exact Maxwell terms from the physical gradients during the readout; φ_Maxwell,bipolar: the per-encode quadratic phase offset from the encoding lobes (the clinically corrected "Maxwell velocity offset"; it is the DC part of the concomitant model);
- T2* decay over the readout, optional fat species;
- intravoxel dephasing: sub-voxel sampling of v on the fine grid + Gaussian turbulence factor exp(−½ k_vᵀ R k_v) with k_v = γ M1 and Reynolds stress R;
- GNL coefficients: the user's `coeff.grad` if available, otherwise a synthetic spherical-harmonic set with realistic deviation (a few to ~10% at 20 cm);
- noise with covariance Ψ; GIRF/eddy currents as a later knob (true G = GIRF ⊛ nominal, recon uses nominal or a perturbed estimate).

### 3.3 Exactness: brute-force GPU direct summation
The truth signal is evaluated as a **direct sum over a fine point cloud** (no NUFFT, no Toeplitz, no expansions, nothing shared with the reconstruction operators):
- blood: vessel voxels at 0.25–0.5 mm (≈ 1–10 M points; ~150 cm³ of lumen);
- static tissue: 2× the reconstruction resolution (≈ 2·10⁷ points at 1 mm over a 35×35×15 cm FOV).
Cost: a 10-min spiral scan has ≈ 1.5·10⁸ samples → ≈ 3·10¹⁵ phase terms ≈ 10–25 min on the 4090 at ≈ 2–5·10¹² terms/s (fast sincos after an exact range reduction; phase in double-single arithmetic, per-sample accumulation in f64). That is fine for benchmark datasets (dozens), and it removes any question of the truth simulator's own approximation error. Checked against an f64 CPU sum on random sample subsets.

### 3.4 Particle mode (later)
Lagrangian particles advected in the (coupled) CFD with Bloch magnetisation history (inflow saturation, acceleration, turbulence) for validation of the moment model on a few cases; CMRsim as an independent cross-check.

### 3.5 Ground-truth definitions for metrics
Defined in the **complex** domain (RACLETTE notes that filtering real-valued velocities is methodologically wrong): per encode, the fine-grid complex signal ideally low-passed to the reconstruction grid → velocity by the encoding-matrix inversion of the phases (includes realistic partial volume). Also: volume-averaged true velocity per voxel and exact flow rates through centerline planes from the CFD.

### 3.6 Fast simulator for training data
Training the denoiser needs many more samples than the benchmark but not truth-level acquisition fidelity: image-domain synthesis + the fast reconstruction operators with injected model mismatch (perturbed B0 and coil maps) is enough to generate realistic aliasing/noise for training inputs.

---

## 4. Reconstruction model and benchmark

Unknown: x_{t,e}(r) complex (frames × encodes × voxels). Data: y_{t,e,c} = A_{t,e}(S_c x_{t,e}) + n, with A including the estimated B0 map, GNL (known coil model), concomitant fields (known from the waveforms, incl. bipolar DC offsets). The model is exact physics but static within the readout, so readout motion, intravoxel dephasing, partial volume and estimated maps are realistic mismatch.

Scenarios: field strength (1.5 / 3 T), gradient system (Gmax 40–80 mT/m), readout length 4–16 ms, undersampling, FOV obliquity and offset from isocentre, VENC. Physics ablation: each of B0 / GNL / concomitant {absent, present and modelled, present and ignored}.

Metrics: velocity RMSE and bias in the lumen (full and eroded), flow-rate and stroke-volume error at centerline planes (ascending, arch, descending), peak velocity, divergence, magnitude NRMSE/SSIM, later WSS; cost as wall-clock on the 4090 and operator applications. Comparisons at equal cost, not equal iterations. Reference: "oracle" reconstruction with the truth operator.

Denoiser priors to compare: (1) classical — divergence-free wavelets (Ong 2015, cycle spinning, noise-referenced SURE thresholds), locally-low-rank across encodes × time; (2) learned 4D-flow denoiser (§5). The fidelity-schedule study (lagged correction, anchor rule) is re-run with each prior: with a real prior, the stability of cheap-only gradients may change in either direction.

---

## 5. 4D-flow denoiser inside P²nP

**Base**: 4DFlowNet-style residual CNN (Ferdian 2020; v2: separate fluid/non-fluid MSE, linear output to allow phase wrapping; ensembles generalise across aorta/cardiac/cerebral — Ericsson 2024). Reimplemented in PyTorch; `res_increase = 1` is plain denoising; SR is a later option (reconstruct on a coarse grid, super-resolve after, or reconstruct on the fine grid).

**Interface to P²nP**: the optimisation variable is complex multi-encode images, not (magnitude, velocity). Two options:
- (a) wrapper: x → (|x|, background phase, v = E⁺Δφ·VENC/π) → network → recompose x̂_e = |x̂|·exp(i(φ̂_ref + π E_e v̂ / VENC)); simple, but the phase extraction is unstable where |x| is small;
- (b) network on real/imag of all encodes directly, with the complex→velocity map inside the loss (velocity + magnitude + divergence terms) — preferred.
Required properties: magnitude-normalisation equivariance (normalise each patch by its norm, multiply back — exact by construction), equivariance to a global phase, robustness to the iteration-dependent input distribution.

**Training inputs** must look like what the denoiser sees inside the iteration (undersampled-spiral aliasing + noise + model mismatch, varying strength), not white Gaussian noise: generate them by running the fast simulator (§3.6) and the first iterations of the actual reconstruction (artefact-aware training), with a noise-level/iteration conditioning input or a blind normalisation-equivariant network. Patches 3D (+ a few frames for temporal context); losses on velocity inside the lumen, magnitude everywhere, weak divergence penalty.

**Runtime**: inference from the Rust reconstruction through a `Denoiser` trait in the torch-free core; a separate adapter crate (tch-rs/libtorch or TensorRT) passes CUDA device pointers zero-copy.

---

## 6. Order of work and gates

| step | content | gate |
|---|---|---|
| M0 | vertical slice on one AVT subject: labels + SDF + centerline-based analytic pulsatile flow (Womersley profile along the centerline, no CFD yet) → direct-sum simulator (B0, concomitant, GNL, moments) → existing P²nP testbed on GPU operators | end-to-end numbers on one realistic case |
| M1 | anatomy pipeline for all 56 AVT subjects: property maps, χ→B0 with shim, coils, fat | B0 range and coil g-factors physiologically plausible |
| M2 | GPU LBM: collision + LES, sparse tiles, SDF bounce-back, inlet profiles, RCR outlets | Womersley, Dean, FDA nozzle, one aorta vs reference |
| M3 | sequence designer (spirals, bipolars, gating, waveforms → moments/Maxwell), full truth simulator, ground-truth definitions | direct sum vs f64 subsets; CMRsim cross-check on one case |
| M4 | benchmark harness, metrics, classical priors (DFW, LLR), GPU-resident P²nP + anchor rule | physics ablation table, schedule study with real priors |
| M5 | learned denoiser: training-data generator, PyTorch model, inference adapter, evaluation inside P²nP (plus RACLETTE as external test) | beats DFW/LLR at equal cost, no failure on held-out subjects |
| M6 | wall/cardiac motion, respiration (5D flow), cerebrovascular domain, turbulence-heavy stenoses | — |

Rough budget: 56 subjects × ~10 flow configurations ≈ 560 CFD runs ≈ 70–200 GPU-hours; benchmark truth simulations ≈ 0.5 h each; storage ≈ 2–4 GB per CFD run (phase-averaged, fp16).

---

## 7. Implementation status (2026-11-02, slice M0 in progress)

Decisions (user): thoracic aorta first; own Rust GPU CFD later (must not block 5D flow / moving walls); 3 T clinical
gradients (80 mT/m, 200 T/m/s), synthetic GNL until real coefficients exist; rigid walls first; PyTorch training with
inference from Rust (tch 0.26 ↔ torch 2.13 venv in `rustlib/.venv`).

Built:
- `rustlib/tools/synth/prep_subject.py`: AVT case → HU fix (some cases are stored as HU + 1024, which silently broke
  TotalSegmentator's lungs), TotalSegmentator labels, thoracic crop, 1 mm tissue grid, 0.5 mm lumen SDF, centerline
  (kimimaro) with radii. Data in `~/hasty-data/` (outside the repo). First subject: Rider R4 (healthy, 0.62 mm slices).
- `hasty-mrsim`: exact piecewise-linear gradient moments (M0, M1, M2, ∫GGᵀ), minimum-time trapezoids/bipolars,
  Lustig time-optimal design for arbitrary k-space curves, variable-density spiral, spiral 4D-flow GRE (referenced or
  Hadamard encoding, stack-of-spirals or 3D-rotated), ECG with RR variability, synthetic harmonic GNL (−14 mm at
  z = 20 cm), GPU direct-sum simulator (verified against an f64 CPU sum: 1.2e-5). Fixed a sequence bug on the way: the
  kz blip must sit at the same time for every encode, otherwise its M1 differs between encodes (kz-dependent velocity
  error).
- `hasty-phantom`: tissue classes/properties at 3 T (post-contrast blood), χ → B0 by dipole convolution with a 2nd-order
  shim (sphere test exact), Biot–Savart loop arrays (32 coils) with body-loss noise covariance, slab prescription fitted
  to the main aortic path, centerline Womersley flow with overflow-safe complex Bessel functions (unit mean flow up to
  α = 80), Murray splits with a velocity cap on side branches, simulator sources (1 mm tissue, 0.5 mm blood; velocity,
  ∂v/∂t and Lagrangian acceleration through the GNL Jacobian).
- `hasty-recon`: simulator output loader (sequence regenerated deterministically), whitening, coil compression,
  multi-coil NUFFT SENSE operator with a generic low-rank phase model (CUR/skeleton by adaptive cross approximation:
  off-resonance + concomitant fields in one decomposition, L ≈ 20 at 2e-3), GNL by reconstructing in apparent
  coordinates + inverse warp + J⁻¹ velocity correction, CG.
- Binaries: `sim4d` (truth simulation), `flowcheck`, `recon_m0` (time-averaged physics ablation).

R4 numbers: slab 56 mm sagittal-oblique, TE 2.2 ms, TR 7.6 ms, 4.4 ms spiral ADC (16 interleaves, 2 mm, slew-limited
at 31 mT/m), off-resonance in tissue −199…537 Hz (1–99%), 3.9 M tissue + 1.8 M blood sources. A 60 s scan (7880
readouts × 2222 samples × 32 coils) simulates in ≈ 17 min (direct sum, ≈ 2e12 coil-terms/s; kernel not yet tuned).

### M0 sanity check (R4, 60 s spiral scan, time-averaged least squares per encode, 100 PCGLS iterations)
Purpose: validate simulator + forward model before the time-resolved P²nP benchmark (not a reconstruction method).

| model in the reconstruction | GNL corrected | velocity RMSE in lumen | bias (logical x, y, z) |
|---|---|---|---|
| none | no | 0.211 m/s | −0.00, +0.10, +0.10 |
| B0 | no | 0.249 | (slower convergence with the segmented operator) |
| B0 + concomitant | no | 0.131 | −0.01, +0.01, −0.00 |
| none | yes | 0.194 | |
| B0 + concomitant | yes | 0.127 | −0.02, +0.00, −0.00 |
| B0 + concomitant on data without B0/GNL | — | 0.060 | −0.01, +0.00, −0.00 |

- The 0.1 m/s offsets are the Maxwell (concomitant) velocity offsets of the bipolars; the concomitant model removes them.
- Bugs found on the way (all fixed): golden-ratio partition order and golden-angle rotation driven by the same shot
  counter (each kz plane got only a few angles → genuine null space, swirl artefacts); `--ideal` did not remove
  concomitant fields; AVT HU offset; velocity sign; finite-difference accelerations.
- Without density compensation the spiral normal operator spans ~10⁴–10⁶ (every readout dwells near k = 0). With the
  T. Chan circulant preconditioner, least squares still needs ~100 iterations; the segmented (B0) operator converges
  slower than the plain one under the same preconditioner. This is the conditioning problem P²nP's preconditioning
  must handle in the time-resolved reconstruction.

### Time-resolved P²nP-ISTA, first runs (R4, 60 s, 20 frames × 4 encodes, plain NUFFT model, `recon_p2np`)
- Implementation: one NUFFT plan whose points are switched per (frame, encode) set (≈ 2 s per iteration for all 80
  sets, 12 virtual coils), gradient E_t^H(E_t x_t − y_t) from explicit residuals, preconditioners none / circulant /
  ZMSHR1, LLR prior over encodes × frames (8³ blocks, random shifts, normalisation-equivariant threshold), optional
  bipolar concomitant phase per encode folded into the coil maps (`--conc-dc`), warm start from the time-averaged
  PCGLS image (`--warm`).
- Cold start, 100 iterations: per-frame velocity RMSE 0.64 (none) / 1.39 (circulant) / 0.54 m/s (ZMSHR1) — phases in
  the lumen still noise.
- Warm start (time-averaged image), 60 iterations, + bipolar concomitant phase: 0.30 m/s for every preconditioner and
  with/without LLR; the frame-to-frame dynamics are not recovered (0.303 → 0.295 m/s best, ZMSHR1 without prior).
- Diagnosis: (1) α = 1/λ_max(N) is set by the DC eigenvalue of the un-weighted spiral sampling (λ_max ≈ 1e10, dynamic
  range 10⁴–10⁶), so dynamic/high-frequency directions get tiny steps; ZMSHR1 (rank one) cannot fix that and the
  circulant M is too crude (λ_max(MN) ≈ 38). (2) The residual is ~180× above the noise floor: the plain model's
  mismatch (readout-time B0/concomitant, GNL, T2*, readout motion) dominates the gradient.

### P²nP follow-up: ZMSHR1 in the circulant metric, ideal data, model-consistent data
- `zmshr1m`: P_k = τ_k M + rank one (ZMSHR1 on M^{-1/2}s, M^{1/2}m; needs M, M⁻¹ = circulant forward). Fastest data
  fit of all variants (ideal 60 s data, 60 it: f 5.2e11 none / 3.1e11 ZMSHR1 / 1.5e11 ZMSHR1-M), adaptive τ ≈ 0.5–4.
- Ideal data (no B0/concomitant/GNL): time-averaged LS velocity error 0.048 m/s (simulator ↔ model consistent).
  Per-frame P²nP from the time-averaged warm start: 0.286 → 0.283–0.296 m/s after 60–300 it; the misfit stalls at
  ~1e11 (≈100× noise) — residual flow physics in the data (spiral readout M1·v of several rad, velocity change within a
  45 ms frame) that a static-per-frame model cannot fit.
- Model-consistent synthetic data (y = E x_true + noise): misfit reaches 1.4× the noise floor but the velocity improves
  only 0.286 → 0.268 m/s; with 4× lower noise 0.269 m/s at 9× the floor after 100 it. The frame-to-frame signal sits in
  poorly conditioned (high-k, undersampled) directions: convergence there is still the bottleneck, and the simulated
  noise (2 % of the k-space RMS) is on the high side.

### Preconditioner and SNR findings (2026-11-03)
- The circulant (T. Chan) inverse-density preconditioner is poor for spirals with coils: λ_max(MN) ≈ 38. Causes, fixed in
  turn: (1) the extreme k = 0 density spike is smeared onto neighbouring modes by the coil maps → use the exact
  Fourier-diagonal ⟨e_k, N e_k⟩ = (ρ ⊛ P_coil)(k) (density convolved with the coil-map power spectrum) instead of ρ
  (a max filter fixes λ_max but kills the near-DC modes = the vessel-scale flow dynamics); (2) raw Biot–Savart maps blow
  up next to the loops → ESPIRiT-style normalised maps Σ|S|² = 1, masked to the dilated support; (3) unsampled in-plane
  k-space corners → floor at the outer-ring level. λ_max(MN) 38 → 9. The rest is coupling between kz planes with very
  different per-frame interleave counts (≈ 3.5 interleaves per plane per frame), which a Fourier-diagonal cannot capture.
- Model-consistent synthetic test then fits the data to the noise floor within 50 iterations but the dynamics are not
  recovered: the deep aortic blood has a per-frame SNR of only a few with the current noise (2 % of the k-space RMS,
  arbitrary). Velocity noise VENC/(π·SNR) ≈ 0.3–0.6 m/s explains the observed errors. Next: calibrated SNR — the
  simulator now also writes `data_clean.npy`, and `HASTY_NOISE_SIGMA` adds Ψ-correlated noise at load time.
- Simulator: `--no-readout-motion` (velocity encoded by the bipolars only) and truth = velocity averaged over each
  frame's cardiac window.
