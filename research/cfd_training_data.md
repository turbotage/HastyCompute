# Training data for the 4D-flow denoiser: RACLETTE first, own moving-wall CFD second

Status: research note (2026-11-03). Companion to `research/denoiser_design.md` (what the network is) and
`research/synthetic_4dflow_pipeline.md` (anatomy → CFD → MR simulator → benchmark). This note decides **what the
training set is, how its targets are defined, and how the CFD must handle wall, cardiac and respiratory motion**.

## 0. Recommendation

1. **Tier 1: RACLETTE images as they are.** 164 subjects (139 static, 25 breathing), 25 frames, 7-point dual VENC,
   clean complex targets. We add the noise. This gets the denoiser v0 training without any CFD work.
2. **Tier 2: re-synthesise from RACLETTE ground truth.** Their 0.5 mm CFD (velocity, pressure, diagonal Reynolds
   stress, wall motion included) is combined with their backgrounds and coil maps through **our** image model: our
   encodings (4-point, balanced Hadamard, icosahedral), VENCs, frame rates, resolutions and point-spread function.
   This removes the single-protocol bias of tier 1.
3. **Tier 3: own GPU lattice-Boltzmann CFD with prescribed (kinematic, one-way) wall motion.** Covers what RACLETTE
   lacks:
   - other vascular beds (carotids, intracranial, pulmonary, LV);
   - a healthy-weighted pathology distribution;
   - cycle-to-cycle variation;
   - our own geometry and motion statistics.
   Motion enters in three ways:
   - (a) wall deformation that changes the lumen (distension, aortic-root motion) inside the CFD, as moving
     interpolated bounce-back;
   - (b) near-rigid motion (respiration, bulk heart displacement) added afterwards in the lab frame;
   - (c) cardiac wall motion of the surrounding tissue in the MR image stage only. LV/atrial blood CFD driven by
     prescribed endocardial motion comes last.
4. **Tier 4: operator-shaped iterates.** Any of tiers 1–3 pushed through our k-space simulator and a few P²nP
   iterations, so that the denoiser also sees the coloured, aliased noise it meets in the loop. GPU-heavy: when the
   fans may run.
5. **Never store CFD snapshots.** Accumulate the per-voxel signal moments (and, optionally, the exact intravoxel
   characteristic function at the encoding moments) on the fly inside the solver. Store ~0.1 GB per simulation.

## 1. What the training set must contain

The denoiser acts on complex multi-encode stacks x_e(r, t) (see `denoiser_design.md` §2). A training pair is
(x_true, x_true + noise), so the target must reproduce everything the in-loop iterate contains, except the noise:

- **Magnitude realism.** Real tissue texture, fat, air, coil shading. In-vivo backgrounds (RACLETTE) or the
  CT-derived MR appearance of `synthetic_4dflow_pipeline.md` §1.
- **Flow realism.** Laminar/helical systolic flow, stenotic jets, recirculation, turbulence (intravoxel dephasing),
  diastolic near-stasis.
- **Motion.** Moving walls (partial volume at the wall changes over the cycle), moving tissue (nonzero tissue
  velocity in the phase), through-plane displacement of the heart and aortic root, breathing for 5D.
- **Background phase.** Smooth, shared by all encodes (B0, coil phase, eddy-current residual).
- **Acquisition variety.** Encoding schemes, VENCs (including wraps: complex targets need no unwrapping), 15–40
  frames, 1.5–3 mm voxels, different SNR and coil geometries.
- **Correct target definition (§5).** The target is the image the reconstruction converges to with infinite SNR,
  not the voxel-sampled velocity.

## 2. Tier 1: RACLETTE as is

Facts checked on our copy (`data/hasty/external/raclette_data`, metadata of all subjects, Data of n001):

| | 7p_2Venc | 7p_2Venc_Breathing |
|---|---|---|
| subjects / distinct geometries / distinct backgrounds | 139 / 139 / **33** | 25 / 24 / 19 |
| resolution, frames | 2.02 × 2.00 × 2.33 mm, 25 × 40 ms | 2.01 × 2.06 × 2.07 mm, 25 frames |
| stenosis grades | 24 healthy, 4 sclerosis, 20 mild, 24 moderate, 39 severe, 27 critical, 1 unclassified | 5 / – / 5 / 5 / 3 / 7 |
| peak velocity (min / median / max) | 1.09 / **3.89** / 7.70 m/s | 1.20 / 3.59 / 6.45 m/s |
| ascending-aorta area change | 2.0 / 6.6 / 28 % | 2.9 / 7.5 / 17 % |
| descending-aorta area change | 0.8 / 2.6 / 9.4 % | 0.8 / 3.4 / 7.5 % |
| aortic-root displacement | 0.9 / 4.6 / 10.5 mm | 1.9 / 4.9 / 9.3 mm |
| jet tilt | 0.2 / 7.9 / 38° | 1.3 / 8.5 / 27° |

Observations relevant for training:
- Encodes are [ref, x@1.0, x@2.5, y@1.0, y@2.5, z@1.0, z@2.5] m/s, axis-aligned (k_v = π/VENC = 3.1, 1.3).
- **Backgrounds move.** Tissue magnitude changes 3–6 % between frames. Tissue phase is nonzero (≈ 0.11–0.15 rad at
  VENC 1.0, ≈ 0.05 rad at 2.5). It scales with 1/VENC, so it is velocity or eddy-current-like phase, not noise.
- **Effectively clean.** Air signal is ≈ 0.6 % of tissue p99, spatially smooth (neighbour correlation 0.3–0.5) and
  almost identical across encodes (correlation 0.96). There is no independent per-encode noise, so the images are
  usable as targets and we add calibrated noise ourselves.
- **Distribution is stenosis-heavy.** Only 24/139 healthy; the median peak velocity is 3.9 m/s. Reweight when sampling
  if the denoiser is meant for general 4D flow, and split train/validation/test **by background patient** (33
  groups), not by subject, to avoid leakage.
- **Licence CC BY-NC 4.0** (DOI 10.3929/ethz-c-000799752). Research use is fine. Weights trained on it inherit the
  non-commercial question; tier 3 is the clean path if that ever matters.

Training tensors: complex fp16 at native grid, ≈ 120 × 100 × 24 × 25 × 7 × 4 B ≈ 0.2 GB per subject → ≈ 33 GB for all
164 subjects. The conversion is CPU-only: zstd decode, crop to body, reorder.

Tier-1 limitation: one protocol. A network trained only on 7-point Cartesian dual-VENC at 2 mm/40 ms would learn that
protocol. The denoiser architecture takes k_e as metadata, but generalisation needs tier 2.

## 3. Tier 2: re-synthesis from RACLETTE ground truth

Per subject the GroundTruth gives, on 0.5 mm, 100 frames:
- velocity U;
- pressure p;
- diagonal Reynolds stress (RST), only the diagonal;
- the lumen and its wall motion.

With the background images and coil maps from Data, we compose:

  x_e(r,t) = PSF ∗ [ m_bg(x,t) e^{iθ_bg,e(x,t)} (1 − χ(x,t)) + m_b(x,t) χ(x,t) exp(ik_eᵀU) exp(−½ k_eᵀΣk_e) ] (r)

- χ: lumen indicator on the 0.5 mm grid at time t (moving with the wall).
- m_b: blood magnitude, taken from RACLETTE's own blood signal or an inflow model.
- Σ: diagonal RST. Spatial intravoxel spread needs no model: it is resolved by integrating the 0.5 mm field over the
  recon PSF, so only the sub-grid part relies on the diagonal-RST approximation.
- θ_bg,e: background phase per encode. Split it into the shared part (θ_bg,0) and the velocity-like tissue part; for
  new encodings re-encode the tissue part as k_eᵀv_tissue, with v_tissue estimated from the RACLETTE background phases.

Background frames are only available at 25 × 40 ms on the native grid; for other frame rates interpolate them in time
(cyclic cubic). GT flow is available at 10 ms.

What tier 2 adds:
- any encoding set (4-point referenced, balanced Hadamard (T_d), icosahedral 13-point; RACLETTE also ships an ICOSA
  variant to check against);
- any VENC;
- 15–40 frames;
- 1.5–3 mm isotropic or anisotropic voxels;
- our exact PSF (§5).

Storage: stream subject by subject (GT is 1.3–3.3 GB per subject, ≈ 300 GB for all; the disk has 229 GB free).
Convert to per-voxel moments at 1 mm (fluid band only, ≈ 0.1 GB per subject) and delete the raw GT.

## 4. Tier 3: own CFD with motion

### 4.1 Solver

Own Rust/NVRTC lattice-Boltzmann (decided in `synthetic_4dflow_pipeline.md` §2; FluidX3D is non-commercial-only and
cannot be a dependency):
- D3Q27 cumulant collision (Geier et al. 2015), with WALE (Nicoud & Ducros 1999) as the explicit sub-grid model.
  WALE is preferred over Smagorinsky because it vanishes in pure shear at walls and in laminar regions.
- Sparse 8³ tiles over the lumen band, AA-pattern in-place streaming, fp16 population storage optional.

Resolution and cost on the 4090, to be measured:
- Thoracic aorta at 0.4 mm ≈ 3–4 M fluid nodes.
- u_lattice ≤ 0.1 at 1.5 m/s peak gives dt ≈ 2.7·10⁻⁵ s, about 37 k steps per 1 s cycle.
  ν_lattice ≈ 6·10⁻⁴, so we are firmly in the LES regime.
- ≈ 4·10¹¹ node updates for 3 cycles, a few minutes at 1.5–2 GLUPS.
- Stenotic jets (4–7 m/s) need ≈ 3–5× more steps and local refinement (0.25 mm in the jet): roughly 30–60 min per
  case.
- Cycles after the first 2 warm-up cycles are used for the signal ensemble (§4.5).

### 4.2 Moving walls inside the CFD

The wall is a time-dependent signed distance φ(x,t) = φ₀(Φ_t⁻¹(x)), where Φ_t is the prescribed motion (§4.3).

- **Interpolated bounce-back with wall velocity** (Bouzidi, Firdaouss, Lallemand 2001), plus the moving-wall momentum
  term (Ladd 1994): f_ī(x_f, t+1) gains −2 w_i ρ (c_i·u_w)/c_s². The wall fraction q and the wall velocity
  u_w = ∂_tΦ ∘ Φ⁻¹ come from φ and Φ at each step (cheap: the analytic motion is evaluated in-kernel).
- **Refill** of nodes uncovered by the receding wall: equilibrium at the extrapolated ρ, u (Lallemand & Luo 2003), or
  Caiazzo's extrapolation. **Removal** of nodes covered by the advancing wall: drop them and account for their mass.
- **Size of the effect.** Root motion of ≈ 10 mm in ≈ 150 ms is ≈ 7 cm/s of wall speed, i.e. ≈ 0.005 dx per step.
  Refill events are rare and isolated, so the spurious pressure noise known from moving-boundary LBM should be small.
  Measure it (§6).
- **Mass consistency.** With a distensible domain, Q_out = Q_in − dV/dt. RCR Windkessel outlets already model
  downstream compliance: when the domain's own compliance is explicit, reduce C by the domain's dV/dp so the total
  arterial compliance is not counted twice.
- **Inlet on a moving plane.** The inlet profile is prescribed in the root frame. The lab-frame inlet velocity is that
  profile plus the plane velocity, and the inlet plane position follows Φ_t.

The ALE finite-volume reference is the approach of the synthetic 4D-flow study in Sci Rep 2022 (OpenFOAM ALE, LES
WALE, wall motion registered from cine, inlet from 2D PC-MRI, Gaussian intravoxel signal model). Use it on one
geometry to cross-check our LBM, not as the production solver (≈ 60 h on 48 cores per case there).

### 4.3 Motion models (prescribed kinematics, calibrated to RACLETTE statistics)

FSI is not needed for training data: the denoiser needs **realistic statistics** of moving walls, not patient-specific
wall mechanics. All motions are smooth displacement fields Φ_t built from a few parameters per subject, sampled from
distributions that match the RACLETTE table in §2 and the literature.

1. **Distension (pressure-driven radial motion).** Bramwell–Hill gives dA/A = dp/(ρc²). Use a pulse-wave speed c
   that grows with age and stiffness (4–10 m/s) and a pressure waveform p(s,t) = p_in(t − s/c) delayed along the
   centreline arc length s. Displacement is normal to the wall: δ = r₀(√(1 + ΔA/A) − 1).
   Check: a pulse pressure of 40 mmHg at c = 5 m/s gives ΔA/A ≈ 20 %, at c = 8 m/s ≈ 8 %. This matches RACLETTE's
   ascending 2–28 %, median 6.6 %, and its descending values are about 2.5× smaller.
   p_in comes from the outlet Windkessels of a previous rigid run, or from a template waveform; one iteration is
   enough, since the coupling is weak.
2. **Aortic-root motion.** Systolic displacement of the root towards the apex plus a small rotation.
   - Amplitude: 1–10 mm (RACLETTE median 4.6 mm). Literature on aortic-root longitudinal motion (e.g. Beller et al.
     2004) reports ≈ 9 mm in healthy volunteers.
   - Attenuation along the centreline: full at the sinotubular junction, decaying through the ascending aorta and arch,
     ≈ 0 where the supra-aortic branches and the spine tether the descending aorta.
   - Time course: follows the LV long-axis (base) motion, i.e. the AV-plane descent curve.
3. **Respiration (5D, tier 3 only).** Near-rigid SI translation of the heart and aorta (5–15 mm) plus smaller AP
   motion, plus the corresponding deformation of the surrounding tissue (liver/diaphragm 10–20 mm). RACLETTE's
   breathing variant provides displacement fields to calibrate against.
   Rigid motion leaves the flow relative to the walls unchanged: respiratory accelerations of ~0.05 m/s² are
   negligible next to systolic pressure gradients. So apply it **after** the CFD, in the lab frame:
   v_lab(x,t) = v_rel(Φ_t⁻¹x, t) + u̇_rigid(x,t). It needs no CFD re-run.
4. **Cardiac wall motion of the surrounding tissue** (myocardium, atria, great veins). This matters in aortic scans
   because the heart is in the FOV and its tissue has nonzero velocity (≈ 5–15 cm/s). That breaks the zero-velocity
   assumption for "static tissue".
   - Source A, analytic LV kinematics (Arts et al. 1992 style): long-axis shortening 10–20 %, radial wall thickening
     30–50 %, torsion 10–20°, in a prolate-spheroidal frame fitted to the TotalSegmentator heart chambers of the CT
     subject.
   - Source B, motion fields registered from public cine (e.g. ACDC, 150 subjects, ED/ES segmentations), transferred to
     the CT anatomy.
   - Applied in the MR image stage only: magnitude is warped by Φ_t, and phase gets k_eᵀ(∂_tΦ).
5. **Intracardiac blood (later).** LV CFD driven by prescribed endocardial motion from source A or B. The volume change
   drives mitral inflow and aortic outflow, and valves are open/closed planes with timing from the cycle. Same solver,
   larger moving-boundary motion (wall speed up to ≈ 10 cm/s, still ≪ 1 dx per step). A recent LV-CFD study driven by
   cine-registration wall motion shows the approach is viable.

### 4.4 Boundary conditions and variability

- **Inlet waveform library.** Stroke volume 50–110 mL, HR 50–110 /min, varied systolic shape.
- **Inlet profiles.** Parabolic, plug, or skewed/helical profiles from the Saitta et al. SSM (437 profiles).
- **Valve jets.** Area 0.4–3.5 cm² (RACLETTE spans 0.37–3.71) and tilt 0–40°.
- **Outlets.** RCR per outlet, tuned to physiological flow splits (~30 % supra-aortic) and pulse pressure.
- **Geometry sources.**
  - AVT (56 CTA, CC BY 4.0);
  - Vascular Model Repository (vascularmodel.com, ~200 models; check licences per model);
  - the 139 RACLETTE lumen masks themselves (NC);
  - aorta shape models (SynthAorta, HUG-VAS, Celi et al. SSM) for sampling new shapes;
  - random smooth diffeomorphisms, stenosis, dilation and coarctation for augmentation.
- **Blood.** Newtonian first, Carreau–Yasuda optional.

### 4.5 Signal accumulation inside the solver (no snapshot storage)

Real 4D flow is an ensemble average over hundreds of heartbeats, so the image sees ⟨e^{ik·v}⟩ over sub-voxel positions
and over beats. Inside the solver, every N steps and for every recon-grid voxel r (a 1 mm accumulation grid, decimated
later) and cardiac bin t, accumulate:
- M₀ = Σ w, M₁ = Σ w v, M₂ = Σ w v vᵀ (mean, 6-component covariance);
- optionally the exact characteristic function C(k_j) = Σ w e^{ik_j·v} for a fixed set of ≈ 26–40 encoding moments
  (3 axes × 2–3 VENCs, Hadamard and icosahedral directions). This covers non-Gaussian intravoxel spectra such as jet
  shear layers and recirculation.

Weights w are the fine-grid PSF weights and the lumen fraction, so partial volume at a moving wall is accumulated
correctly. Cycle-to-cycle variation: run ≥ 5 cycles after warm-up with jittered HR (±3 %), stroke volume (±5 %) and a
small random inlet perturbation, and average over cycles. Storage ≈ (10 + 2·n_k) fp16 values × fluid-band voxels ×
frames ≈ 0.05–0.2 GB per simulation, independent of how long the solver runs.

## 5. Target definition: match the reconstruction's PSF

The training target must be what the reconstruction converges to at infinite SNR, given dense k-space and our
operator. That is the **band-limited projection** of the continuous object onto the recon grid, not a box average and
not point-sampled velocities:

  x_true = D · F⁻¹[ W(k) · F x_fine ],

- x_fine: the object on the 0.5 mm (or 1 mm) grid, built from §3 or §4.5.
- W(k): the k-space support of the trajectory (spherical for stack-of-spirals/3D spirals, cube for Cartesian), with
  the same taper or apodisation the reconstruction uses.
- D: decimation.

Why this matters:
- Box-averaged targets ask the denoiser to remove Gibbs ringing the data cannot support. That biases the prior against
  the data term, the opposite of "respecting the data".
- Velocity at the wall is the phase of a band-limited sum of complex signals, not the mean velocity. Point-sampled
  velocity targets get partial-volume voxels wrong.
- This is the same object we validated the operator against (`opval`, RACLETTE n001: unbiased to 2.4 mm/s with full
  physics), so the denoiser and the data term share one definition of "truth".

## 6. Validation gates for the CFD

1. Rigid: Poiseuille, Womersley (< 1 %), Dean flow, FDA nozzle Re 500/3500/6500 vs PIV.
2. Moving walls:
   - a tube with prescribed uniform radius R(t) and one closed end: the axial flux grows linearly from the closed end,
     Q(z) = 2πRṘz, the radial wall velocity equals Ṙ, and the mass balance Q_in − Q_out = dV/dt holds to 10⁻³;
   - an oscillating rigid tube: zero relative flow and lab velocity equal to the wall velocity (Galilean check);
   - spurious pressure noise from refill below 1 % of the systolic pressure gradient;
   - the Womersley elastic-tube solution for a distensible pipe.
3. RACLETTE cross-check: run our solver on 3–5 RACLETTE geometries with matched inflow (their flow-rate curve is in
   `meta.json`, 100 samples) and compare peak velocity, TPG, flow splits and TKE maps against their GT.
4. One geometry against an OpenFOAM ALE run (the Sci Rep 2022 recipe).
5. Signal accumulation: the moments path against our brute-force DirectSim on one case.

## 7. Disk and compute budget

- Free disk 229 GB (`data/hasty` currently 22 GB).
- Tier 1: ≈ 33 GB of training tensors (the raw Data download is ≈ 70 GB; stream and delete).
- Tier 2: ≈ 15–30 GB of moments (GT is streamed one subject at a time and deleted after conversion; never hold the
  full ≈ 300 GB).
- Tier 3: ≈ 0.1 GB per simulation, so 500 simulations ≈ 50 GB.
- Training patches are generated on the fly from moments and backgrounds: the noise draw, encoding set, VENC, frame
  count and voxel size are sampled per batch. No pre-baked pair dataset.

## 8. Order of work

| step | what | GPU | gate |
|---|---|---|---|
| C0 | RACLETTE tier-1 converter (stream, crop, fp16, split by background) | no | tensors load, round-trip error < fp16 eps |
| C1 | denoiser v0 (`denoiser_design.md` ablations 2–5) on tier 1 | yes, training | beats LLR in P²nP on the RACLETTE n001 validation case |
| C2 | tier-2 re-synthesis (moments at 1 mm, band-limited targets, new encodings) | light | tier-2 render of the native protocol ≈ RACLETTE images in the lumen |
| C3 | LBM rigid + validation gate 1 | yes | gate 1 |
| C4 | moving walls (distension, root motion) + gates 2–3 | yes | gates 2–3 |
| C5 | production tier 3 (geometries, motion sampling, in-solver accumulation) | yes | statistics match §2 table |
| C6 | tier 4 operator-shaped iterates | yes | denoiser on in-loop iterates |
| C7 | LV CFD with endocardial motion, 5D respiration | yes | later |

C0 and the CPU parts of C2 can run while the fans must stay quiet; everything from C1 onwards waits.

## References

- RACLETTE: Dirix, Buoso, Kozerke (ETH Zürich), synthetic aortic 4D-flow dataset, CC BY-NC 4.0,
  DOI 10.3929/ethz-c-000799752.
- Synthetic 4D flow with ALE CFD and moving walls (OpenFOAM, LES WALE, cine-derived wall motion, Gaussian intravoxel
  model), Sci Rep 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC9513106/
- LV CFD driven by cine-registration wall motion: https://pmc.ncbi.nlm.nih.gov/articles/PMC13278472/
- FluidX3D (licence: non-commercial only): https://github.com/ProjectPhysX/FluidX3D
- Vascular Model Repository: https://www.vascularmodel.com
- Bouzidi, Firdaouss, Lallemand, *Momentum transfer of a Boltzmann-lattice fluid with boundaries*, Phys. Fluids 2001.
- Ladd, *Numerical simulations of particulate suspensions via a discretized Boltzmann equation*, J. Fluid Mech. 1994.
- Lallemand, Luo, *Lattice Boltzmann method for moving boundaries*, J. Comput. Phys. 2003.
- Geier, Schönherr, Pasquali, Krafczyk, *The cumulant lattice Boltzmann equation in three dimensions*, Comput. Math.
  Appl. 2015.
- Nicoud, Ducros, *Subgrid-scale stress modelling based on the square of the velocity gradient tensor* (WALE), Flow
  Turbul. Combust. 1999.
- Bramwell, Hill, *The velocity of the pulse wave in man*, Proc. R. Soc. B 1922.
- Beller et al., *Role of aortic root motion in the pathogenesis of aortic dissection*, Circulation 2004.
- Arts et al., *Description of the deformation of the left ventricle by a kinematic model*, J. Biomech. 1992.
- Bernard et al., ACDC challenge, IEEE TMI 2018.
- Saitta et al. 2023, inflow-profile SSM; SynthAorta (IEEE TMI 2025); HUG-VAS; Celi et al. aorta SSM; AVT (Radl et
  al. 2022) — as listed in `synthetic_4dflow_pipeline.md` §0.
