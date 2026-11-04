# Synthetic truth for super-resolution, vessel scale generalisation, and datasets beyond the aorta

Status: research note (2026-11-04). Reading-only session, nothing implemented. Extends
`research/synthetic_4dflow_pipeline.md`, `research/cfd_training_data.md`, `research/denoiser_design.md` and
`research/velocity_model_sr.md`. Claims marked *(memory)* come from my own knowledge, not from a checked source.
Numbers marked *(estimate)* are order-of-magnitude derivations in this note.

## 0. Summary

1. **Truth in three layers**: continuous CFD object (0.25–0.5 mm, 5–10 ms, B-spline in cycle time) → SR target
   (band-limited projection onto the reconstruction grid, possibly finer than the acquisition supports) →
   acquisition input (direct-sum simulator with true spiral timing, gating, HRV). Spatial blur and temporal-window blur
   then come from physics, not from k-space cropping or frame dropping.
2. **Scale (§3)**: a denoiser trained on a few geometries and flow conditions *can* generalise to other vessels, but
   only along the axes the training set covers. The flow field depends on dimensionless numbers (Re, Wo, geometry in
   units of D, waveform shape), the image depends on further dimensionless numbers (D/h, VENC/U, SNR, frame/cycle
   ratio). Build the CFD library **non-dimensionally** and choose physical size at the imaging stage, so scale
   coverage is by construction. What does not scale: regime (Re, Wo), waveform class per bed, geometry class, inflow
   physics, tissue surroundings, absolute rigid motion.
3. **Visibility (§4)**: rule of thumb ≥ 4–6 voxels across the lumen to be usable, ≥ 7 for < 5 % flow error. Condition
   on d/h and mark d/h < 3 as "unresolved".
4. **Datasets (§6)**: intracranial (TopCoW, TopBrain, IntrA, Aneurisk, AneuX, CTA+CFD set, TOF sets, 7T SMILE-UHURA),
   neck (Zenodo carotid bifurcations, TotalSegmentator on neck CTA), abdomen/legs (TotalSegmentator CT/MR labels, VMR
   aortofemoral, procedural trees). There is no public CTA set with femoral/popliteal masks.

## 1. Super-resolution and temporal super-resolution truth

**What published pipelines do.**
- 4DFlowNet: CFD as HR truth, LR by k-space cropping + noise ([arXiv 2004.07035](https://arxiv.org/pdf/2004.07035)).
- Intracardiac 4D-MRAR: k-space truncation for 4× spatial reduction (128³ at 0.625 mm → 32³ at 2.5 mm), temporal 4×
  by selecting time steps (10 → 40 ms), valves binary open/closed, ALE CFD (svFSI), wall motion from SDF4CHD (99
  coarse sims) or gated CT (8 fine sims). Authors list the sparse temporal selection as a limitation
  ([arXiv 2611.04053](https://arxiv.org/html/2611.04053)).
- Temporal SR (TMI): 4DFlowNet adapted to 2D+t; the CFD→MRI pipeline accumulates the readout pattern over
  consecutive frames (temporal averaging, retrospective gating); authors attribute the gain mainly to this realistic
  temporal degradation ([ResearchGate](https://www.researchgate.net/publication/403148138_Deep_learning_for_temporal_super-resolution_4D_Flow_MRI)).
- Intracranial multi-cohort SR: only 4 patient CFD sims for pretraining, k-space truncation + clinically matched noise
  + dual VENC (60/120 cm/s), in vivo HR 0.6 mm / LR 1.2 mm; then transfer learning on 18 subjects
  ([arXiv 2610.35396](https://arxiv.org/html/2610.35396)).
- Fourier-shell metrics separate real resolution gain from denoising
  ([PMC13418888](https://pmc.ncbi.nlm.nih.gov/articles/PMC13418888/)); RACLETTE publishes paired HR 1.25 mm / LR 2.5 mm.

**What we do instead.** The simulator already integrates true readout times and gradient moments, so no shortcut is
needed:

| layer | definition | where it lives |
|---|---|---|
| L1 continuous object | CFD velocity at 0.25–0.5 mm, 5–10 ms, periodic in cycle time (periodic cubic B-spline in τ, as in `velocity_model_sr.md` A2), per-voxel signal moments accumulated in-solver (`cfd_training_data.md` §4.5) | CFD / moment store |
| L2 SR target | `D · F⁻¹[W(k) · F x_fine]` on the reconstruction grid; for SR the grid is finer than the acquisition (s ≥ 1) and W is the *target* support, not the acquisition support | target generator |
| L3 acquisition | direct-sum simulation with spiral timing, gating, HRV, B0/concomitant/GNL, coil noise | `hasty-mrsim` |

Consequences:
- Temporal blur = window average of v(t) over the readouts that fill a frame, plus the M1 phase of the readout itself.
  Temporal SR then means "recover the B-spline coefficients of v(t) at node spacing finer than the frame width".
- SR content above the acquisition k_max is **prior, not data**. The SR must stay inside P²nP so the data term binds;
  evaluate with Fourier-shell metrics against L2 truth, and report the denoising gain separately.
- Rigid-wall CFD teaches the wrong partial-volume statistics at the wall. The wall voxel is where SR matters, so
  moving walls (distension, root motion) are required for SR targets, not only for denoising.
- Sub-resolved branches matter (§5): partial-volume signal from vessels below the segmentation limit sits next to
  resolved lumens.

## 2. Heart wall and tissue motion (additions)

`cfd_training_data.md` §4.3 already lists distension (Bramwell–Hill), root motion, rigid respiration, tissue motion in
the image stage and LV CFD last. Added sources:
- **SDF4CHD** generative whole-heart CT model: LV geometry + motion at scale (used for 99 sims in 4D-MRAR).
- **Gated CT** LV meshes + motion (8 patient sets in 4D-MRAR).
- **Cine registration**: ACDC (150 subjects) ([PMC10198131](https://pmc.ncbi.nlm.nih.gov/articles/PMC10198131/)).
- **Bai bi-ventricular atlas** (> 1000 healthy subjects) as a motion SSM, and the open 4D heart SSM of Unberath et al.
  ([ResearchGate](https://www.researchgate.net/publication/283758608_Open-source_4D_statistical_shape_model_of_the_heart_for_x-ray_projection_imaging)).
- Calibration target for aortic motion: RACLETTE table (`cfd_training_data.md` §2).
- Valves as binary open/closed by phase is the accepted simplification (4D-MRAR).

Order unchanged: aortic wall + root → tissue motion in the image stage → LV CFD.

## 3. The scale issue: resolution versus real-world size

The question: if the network learns "how flow in a vessel looks" for a few geometries, pulsatile conditions and
pressures, will it work on other vessels? **Largely yes, if the training set spans the right dimensionless
numbers.** This section separates what is scale-free from what is not.

### 3.1 Flow physics: what depends on size

Incompressible Navier–Stokes with no-slip walls. Non-dimensionalise with D (diameter), U (peak mean velocity),
ω = 2π f (heart rate), ν (kinematic viscosity, ≈ 3.3·10⁻⁶ m²/s, *(memory)*):

- Re = U D / ν
- Wo = (D/2) √(ω/ν)  (Womersley number)
- geometry in units of D: curvature D/R_c, stenosis area ratio, branching angles and area ratios, aneurysm
  neck/dome ratios
- waveform shape over the cycle (dimensionless in t/T), and wall compliance ΔA/A (already dimensionless)

The non-dimensional velocity field u/U(x/D, t/T) is a function of these only. **Dynamic similarity**: two vessels with
the same (Re, Wo, geometry/D, waveform) have identical non-dimensional flow, whatever D is.

Typical ranges *(memory, order of magnitude)*:

| bed | D | peak U | Re peak | Wo |
|---|---|---|---|---|
| ascending aorta | 25–35 mm | 1–1.5 m/s | 3000–7000 | 15–20 |
| abdominal aorta | 15–20 mm | 0.6–1 m/s | 2000–4000 | 10–13 |
| common carotid | 6–8 mm | 0.7–1 m/s | 1000–2000 | 4–6 |
| MCA / ICA intracranial | 2.5–5 mm | 0.4–0.8 m/s | 300–1000 | 2–4 |
| smaller cerebral | 1–2 mm | 0.2–0.4 m/s | 60–250 | 1–2 |

Observations:
- Peak velocity varies by only about 5× across beds while D varies by 15×. Re falls with D; Wo ∝ D at fixed HR.
- Stokes boundary layer δ = √(2ν/ω) ≈ 1 mm at 72 bpm *(estimate)*, **independent of D**. In the aorta it is a thin
  layer (δ/D ≈ 0.04, plug-like core); in a 3 mm cerebral artery it fills the lumen (δ/D ≈ 0.3, near-parabolic
  quasi-steady profile). In the image the layer is ≈ 0.5 voxel at 2 mm and ≈ 1.5 voxels at 0.6 mm. So the *profile
  shape the network must restore* is regime-dependent and not removable by rescaling the picture.
- The low-Re / low-Wo end is the favourable case. In Stokes flow (Re → 0) there is no length scale in the equations:
  geometry scaling is exact and a rescale augmentation is physically valid there. Poiseuille and quasi-steady
  Womersley profiles are the same function of r/R at any size. Cerebral and carotid flow is mostly in this
  near-laminar regime except at stenoses and aneurysms.
- The high-Re end (aorta, stenotic jets) is turbulent; turbulence only appears in the image as intravoxel dephasing
  (Σ, see below), a function of Re, not of D separately.

### 3.2 What the image adds (also dimensionless)

With voxel size h, VENC, frame time T_f and cycle time T:

| parameter | meaning | scale behaviour |
|---|---|---|
| D/h | voxels across lumen | the main "visibility" axis, set by resolution, free to sample |
| U/VENC | wrap ratio | scale-free; dual-VENC wraps depend on it |
| intravoxel shear dephasing | Δv ≈ γ̇ h, with wall shear rate γ̇ ≈ 8 U/D for Poiseuille *(estimate)*, so Δv/VENC ≈ 8 (U/VENC)(h/D) | depends on h/D and U/VENC only |
| frames per cycle F = T/T_f | temporal sampling | scale-free at fixed F; a *different* F (e.g. 40 ms frames at HR 60 vs 100) must be sampled |
| partial volume at wall | fraction of boundary voxels | ∝ h/D |
| SNR | ∝ voxel volume, ∝ √(scan time) | not scale-free; sample SNR independently |
| wall motion in voxels | δ_r / h with δ_r = (D/2)(√(1+ΔA/A) − 1) | δ_r scales with D, so it shrinks to sub-voxel for small vessels; ΔA/A is dimensionless |
| bulk rigid motion (root displacement, respiration, brain pulsation) | absolute mm | **not** scale-free; compare with h and with D separately |
| inflow enhancement | depends on U·TR versus slab thickness | **not** scale-free; depends on the sequence |

### 3.3 What generalises, what does not

Generalises across D (at fixed regime):
- no-slip + incompressibility structure (divergence-free field, v → 0 at the wall in the lumen interior limit);
- smooth profile forms (parabolic, flattened, helical skew) as functions of r/R;
- partial-volume and Gibbs behaviour as functions of h/D and the PSF;
- phase ↔ velocity relation and wrap structure;
- noise behaviour, if SNR is sampled independently.

Does **not** generalise automatically:
1. **Regime (Re, Wo).** Train on aorta and iliacs (Wo 10–20) and test on carotid (Wo 4–6) or cerebral (Wo 2–4): the
   profile family differs. The training set must cover the (Re, Wo) plane up to the test points, not only D.
2. **Waveform class per bed.** Aorta: sharp systolic peak with diastolic near-stasis. Carotid ICA / cerebral:
   low pulsatility, continuous diastolic flow. Renal / visceral: low-resistance. Legs: triphasic with diastolic
   reversal *(memory)*. Veins: low-pulsatility, respiratory modulation. The prior over t/T shapes must contain them.
3. **Geometry class.** Carotid siphon, bifurcation with plaque, aneurysm sacs, abdominal side branches, iliac
   bifurcation. Procedural and real geometries needed (§5, §6).
4. **Surrounding anatomy.** Brain tissue versus thoracic air/fat/bone changes magnitude contrast and background
   phase; pure domain randomisation of tissue appearance (SynthSeg-style, already planned).
5. **Non-Newtonian blood** at small D and low shear (Fåhræus–Lindqvist, shear thinning). Probably second-order for
   4D-flow resolution; Carreau–Yasuda optional in the solver.
6. **Acquisition-specific effects** (spiral PSF, inflow, readout motion) are not vessel-size effects but interact with
   it; they are simulated, not learned.

### 3.4 Building scale coverage into the data

**Non-dimensional CFD library.** LBM runs in lattice units anyway. Index each simulation by (Re_peak, Wo, waveform
shape, wall ΔA/A, geometry), with geometry in units of D. At the imaging stage choose the physical size and derive the
rest:

  U = Re ν / D,  ω = 4 ν Wo² / D²  (so HR = ω / 2π)

For a given D this fixes U and HR. Rejection-sample (D, Re, Wo) so that U ∈ [0.05, 3] m/s and HR ∈ [45, 130] bpm
*(estimate bounds)*. One CFD run then serves many physical sizes, which is the scale coverage "by construction".
Check: D = 3 mm, Wo = 3 gives ω ≈ 14 rad/s, HR ≈ 130 bpm, only just physiological, so the independent range of Wo at
small D is narrow and the bed-specific (D, Wo) pairs in §3.1 are the main region. Do not rescale CFD geometries
without re-matching Re and Wo: they cannot both be preserved while changing D at fixed U and f.

**Sampling.** Draw per patch: bed class (waveform family), geometry, (D, Re, Wo) from the physiological joint, h
(so D/h ∈ ~[2, 15]), VENC (U/VENC ∈ ~[0.2, 3]), F, SNR, encodes. This is the existing "no pre-baked pairs" plan
(`cfd_training_data.md` §7) with the dimensionless axes made explicit.

**Network conditioning.** The convolutional net sees structures in voxel units, so D/h is implicit in the image.
Regime information is not implicit when the profile is unresolved. Give it, multiplicatively (degree-0 FiLM, no
additive terms, per `denoiser_design.md` §10): h in mm, HR (known from ECG), VENC set. D follows from the mask size
times h, so Wo is derivable by the network; alternatively compute Wo and Re from the denoiser's current mask estimate
and pass them in. Do not pass the absolute scale through normalisation shifts.

**Leave-one-regime-out validation.** The generalisation claim is testable:
- train on {ascending aorta, iliacs} (Wo 10–20), test on carotid;
- train on {aorta, carotid}, test on intracranial;
- train on healthy, test on stenotic and aneurysm geometries;
- hold out a waveform class (e.g. legs).
The gap measures extrapolation. RACLETTE alone cannot do this (aorta only), so this is a tier-3 deliverable. Real
checks: the in-vivo varied-voxel data at [Zenodo 4882572](https://zenodo.org/records/4882572) and downsampled
subject data where available.

## 4. How many vessels are visible: voxels across the lumen

Published thresholds (d/h):

| source | statement |
|---|---|
| QC review ([PubMed 35197395](https://pubmed.ncbi.nlm.nih.gov/35197395/)) | ≥ 4 voxels, 6 preferred; temporal resolution < 40 ms |
| In-vitro 2–5 mm tubes ([PMC8880336](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8880336/)) | < 7 voxels overestimates flow by > 5 % |
| Pulsatile phantom (same source family) | voxel 7.5–14 % of d for < 10 % error (7–13 voxels) |
| Cerebral review ([PMC8369999](https://pmc.ncbi.nlm.nih.gov/articles/PMC8369999/)) | vessels ≤ 3 voxels, or voxel ≥ 0.75 mm in CoW, bias significantly; one cited source: ≥ 16 voxels over the lumen for < 10 % error (likely over the area) |

Abdominal diameters (celiac 7–8 mm, hepatic artery 3–4.5 mm, splenic 5–6 mm, portal vein 14–15 mm in liver disease)
from the [abdominal 4D-flow review](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/a-1271-7405.pdf);
hepatic and splenic arteries reproduce poorly (CV 11–38 %, [PMC9674613](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9674613/)).

| grid | resolved (d/h ≥ 6) | marginal (3–6) | unresolved |
|---|---|---|---|
| 2 mm (thoracic) | aorta, brachiocephalic, SVC/IVC, pulmonary trunk, portal vein | left carotid, subclavian | carotids |
| 1.2 mm (neck, abdomen) | CCA, subclavian, celiac, splenic | ICA, vertebral | renal and hepatic arteries |
| 0.6–0.8 mm (cerebral) | ICA, basilar | M1, A1, P1 | distal MCA/ACA/PCA, lenticulostriates |

(Vessel diameters not cited above are from memory.) Typical intracranial acquisitions are 0.5–1.1 mm isotropic.

Design rules:
- condition and stratify on d/h, not on region; pool all vessels;
- per-voxel "unresolved" label (d/h < 3): excluded from velocity loss and metrics, the network may return a
  low-confidence/identity output there;
- report metrics per d/h bin so a result like "good for d/h ≥ 5, bias below" is explicit.

## 5. Sub-resolved branches and procedural trees

Segmentation limits sit at 0.3–1 mm, but partial-volume signal and inflow come from vessels below that. Plan:
- CFD only for the resolved trunk (inlet at the proximal end, RCR outlets at truncated branches);
- grow the tree below the segmentation limit by Murray-law branching or constrained constructive optimisation
  *(memory)*, anchored to the real trunk centerlines;
- give the sub-resolved branches the centerline Womersley flow that `hasty-phantom` already implements, with
  Murray flow splits and a velocity cap (already in `hasty-phantom`);
- calibrate branching statistics (counts, tortuosity, radii) on 7T TOF distal-tree data (SMILE-UHURA, §6).

For legs, renal and mesenteric arteries where no public masks exist, the same procedural route anchored to the CT
abdominal aorta and iliacs gives plausible branching.

## 6. Datasets

### Intracranial

| dataset | content | licence / note |
|---|---|---|
| [TopCoW 2024](https://zenodo.org/records/10990868) | 125 paired CTA/MRA (250 scans), 13 CoW classes, graph labels; from OpenData Swiss | non-commercial, credit source; Zenodo record text CC BY 4.0, images follow data owner; subset shares CROWN terms |
| [TopBrain 2025](https://topbrain2025.grand-challenge.org/data/) | 90 annotated volumes in total, 50 public ([Zenodo 16878417](https://zenodo.org/records/16878417)), about 40 (CTA) / 42 (MRA) vessel classes incl. veins | built from TopCoW cases; cite medRxiv summary |
| [IntrA](https://github.com/intra3d2019/IntrA) | 103 whole-brain vessel models from MRA, 1909 segments (215 aneurysm) | raw MRA not released |
| AneuX | 750 aneurysm models with morphology | |
| [CTA + CFD set](https://pmc.ncbi.nlm.nih.gov/articles/PMC11557944/) | raw CTA, 3D models, CFD hemodynamics per patient and control | |
| Aneurisk | 103 models, 23 3DRA, hemodynamics for 8 | |
| IntrA-derived CFD benchmark | 105 aneurysm geometries with CFD fields (Zenodo doi 10.5281/zenodo.15697196, CC BY 4.0) | only a fragmentary hit: verify |
| [Di Noto TOF-MRA](https://www.nature.com/articles/s41597-024-03397-8) | 63 patients (24 with follow-up), aneurysm + vessel segmentations, 3D models | |
| ADAM | 254 TOF-MRA | |
| [SMILE-UHURA](https://arxiv.org/pdf/2411.09593) | 20 healthy 7T 3D multislab TOF at 300 µm (StudyForrest), small-vessel labels | download location unconfirmed; use for distal tree statistics |

Intracranial CFD is mostly laminar (§3.1), so an LBM with light LES is sufficient and cheap.

### Neck

- [Zenodo carotid bifurcations](https://zenodo.org/records/7634644): 62 cases (≈ 110 inner-wall models), lumen + plaque
  + centerline tree from CTA, stenosis spectrum, CC BY 4.0, ≈ 399 MB. Bifurcation only, no vertebrals.
- HNCTA: 358 head-and-neck CTAs, 9 vessel classes incl. vertebral, ICA, CCA, basilar (used in DeformCL,
  [arXiv 2506.05820](https://arxiv.org/pdf/2506.05820)); public status unconfirmed.
- HaN-Seg: 42 public CTs (carotid labels), radiotherapy, not CTA.
- MICCAI 2009 carotid bifurcation challenge: 56 head-and-neck CTAs, lumen at the bifurcation; availability unconfirmed.
- Route to a connected tree: run TotalSegmentator on full neck-to-chest CTA (carotids, subclavians,
  brachiocephalic are classes *(memory)*) and join with the AVT aortic arch.

### Abdomen, pelvis, legs

- TotalSegmentator CT/MR (class lists differ per version): aorta, IVC, portal/splenic vein, iliac arteries and veins,
  pulmonary artery. No renal or mesenteric artery class found. MR data at
  [Zenodo 14710732](https://zenodo.org/records/14710732) (80 structures, about 50 main classes in `total_mr`).
- [AVT](https://figshare.com/articles/dataset/Aortic_Vessel_Tree_AVT_CTA_Datasets_and_Segmentations/14806362):
  arch + branches, abdominal aorta + iliacs; stops at the iliacs.
- No public CTA set with femoral/popliteal masks found (TransONet segments to the knee but data availability is unclear).
- [Vascular Model Repository](https://www.researchgate.net/publication/262844310_The_Vascular_Model_Repository_A_Public_Resource_of_Medical_Imaging_Data_and_Blood_Flow_Simulation_Results):
  126 models (2013 count) in aorta, cerebral, coronary, aortofemoral, pulmonary, single-ventricle; CFD for 100, 29
  exercise states for 17. Aortofemoral models are the leg/pelvis route. Licence per model: check.
- From memory, unchecked: Parse2022 (pulmonary arteries), 3D-IRCADb and MSD hepatic vessel (liver), KiTS (kidney),
  ImageCAS and ASOCA (coronary).

### In vivo test data

- RACLETTE (external test, aorta).
- [4D flow with varied isotropic voxel size and acceleration](https://zenodo.org/records/4882572) (Zenodo).

## 7. Recommended order

1. Adopt the three-layer truth (§1) in the benchmark and training generators; keep SR inside P²nP.
2. Add the dimensionless sampling (§3.4) and the d/h + "unresolved" mask (§4) to the dataset code now (no CFD needed).
3. Waveform library per bed (aorta, carotid, intracranial, renal/visceral, leg triphasic, venous) as a dimensionless
   t/T family; extend `hasty-phantom` flow before the LBM exists.
4. Geometry sources by cost: VMR and Zenodo carotid (ready lumens) → TopCoW/TopBrain + TotalSegmentator-style prep →
   procedural sub-trees (§5).
5. Non-dimensional LBM library, laminar cerebral/carotid first (cheap), aorta + jets after.
6. Leave-one-regime-out experiments (§3.4) as the acceptance test for "generalises to other vessels".
7. Licences: TopCoW and RACLETTE are non-commercial; weights inherit the question.

## Open questions

- How far can near-Stokes rescaling augmentation be used in practice before inertial effects show (a CFD check at
  Re 50–500 in a bifurcation)?
- Is HR + h conditioning enough for the network to infer Wo, or does an explicit Wo/Re input help (ablation)?
- How much do veins and venous respiratory modulation matter for the intended applications?
- Is the TotalSegmentator "vessels" coverage on neck/abdominal CTA good enough, or are manual/semi-automatic masks needed?
