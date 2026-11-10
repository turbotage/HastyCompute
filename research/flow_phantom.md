# Synthetic 4D/5D flow phantom — design (2026-11-05)

Purpose: ground truth that is **realistic in the features the reconstruction must capture**. It need not be
anatomically exact. One generator serves two consumers:
- **benchmark** — reconstruction tests with exact truth at any resolution;
- **training data** — for the velocity-space prior (`DenoiserZP`).

It supersedes the aorta-only, centreline-Womersley truth of M0 (`synthetic_4dflow_pipeline.md`), and complements
the real-scan-plus-CFD plan (`cfd_training_data.md` §4) with a procedural phantom we fully control.

## 1. Requirements

| feature | why the reconstruction cares |
|---|---|
| **many vessels of all calibres** (aorta ≈ 30 mm down to ≈ 0.5 mm), arteries and veins, through the whole FOV | partial volume, sub-voxel vessels, crowded flow, priors that must not smear neighbouring vessels |
| pulsatile flow, realistic waveforms, phase lags along the tree, diastolic backflow | temporal model (spline nodes, continuous time), VENC continuation |
| complex flow: helical / secondary flow in bends, recirculation, valve jets, stenotic jets | dual-VENC unwrapping, intravoxel dephasing (Σ), no-slip near walls |
| **tissue motion**: cardiac contraction, respiration (5D) | static-tissue anchor (myocardium is *not* static), motion artefacts, binning |
| **vessel motion**: distension, root motion, translation with the surrounding tissue | moving walls, partial volume over the cycle |
| **valves**: opening / closing orifices, leaflets | jets, regurgitation, fast velocity transients |
| **resolution-free definition** (implicit geometry, analytic or high-resolution fields) | super-resolution truth at any grid (§B of `velocity_model_sr.md`), no inverse crime |
| MR properties: PD / T1 / T2*, textures, fat, air / lung, susceptibility → B0 | realistic magnitude, B0, coil maps |
| **randomisable** (seeded): shapes, trees, waveforms, heart rate, motion amplitudes | many training cases, held-out benchmark cases |

## 2. Representation: everything is a function of (x, t)

- **Geometry.** Signed distance functions.
  - Vessels: smooth unions of tapered tubes along spline centrelines (closed-form distance to the segment plus a smooth
    minimum at junctions).
  - Heart chambers: deformable superquadric / ellipsoid unions.
  - Valves: moving surfaces.
  - Evaluated on demand at any resolution; masks, partial-volume fractions and wall distances follow.
- **Motion.** A displacement field Φ_t(x) composed of:
  - respiration: diaphragm SI translation with an AP / LR expansion of the torso;
  - cardiac: chamber contraction from a volume curve, plus base descent and twist;
  - vessel distension: radius r(s, t) from a delayed pressure wave;
  - root motion.

  Tissue and vessels are defined in a reference frame and pushed forward by Φ_t. Wall velocity is ∂_tΦ.
- **Flow,** in tiers by calibre:
  - **network vessels** (most of them):
    - a 1D pulsatile network (resistances, compliances, Murray's law, wave delays) gives Q(s, t) per segment;
    - each segment carries an exact Womersley profile for its Q(t) and r(t);
    - bends add a Dean secondary-flow term; junctions blend over a short length;
    - divergence-free inside straight segments; mass conserved at junctions by construction.
  - **great vessels and chambers** (aorta, pulmonary trunk, ventricles, atria): GPU lattice-Boltzmann LES with moving
    boundaries (`cfd_training_data.md` §4, Bouzidi + moving-wall bounce-back).
    - Valves are resistive immersed surfaces whose open area follows the cycle (later: kinematic leaflets).
    - Resolution 0.25–0.4 mm; the outlets drive the network inlets.
  - **Tier A (no CFD):** the chambers get a parametric flow (inflow jet with a ring vortex in diastole, outflow tract in
    systole). Cheap, for quick tests and bulk training data; Tier B replaces it with CFD.
- **MR properties.** Labels map to PD / T1 / T2*. Within-label textures are procedural (band-limited noise) or taken
  from CT HU deviations when an anatomical scaffold is used. χ comes from the labels → B0 by the existing dipole code.
  Fat is a separate species.

## 3. Scaffold options

1. **Procedural torso** (default):
   - ellipsoidal body with a fat layer, muscle, lungs, liver, spine;
   - a heart placed and oriented by parameters;
   - a vessel forest grown into every tissue.
   - Unlimited variety, no licensing questions.
2. **Anatomical scaffold:** R4 (AVT CTA + TotalSegmentator labels, already prepared) for the outer anatomy, heart and
   great vessels, with the procedural forest grown into its organs. More realistic context; a few subjects.

Both share all generators below; the scaffold only provides the domains and the large structures.

## 4. Vessel forest generator

- **Growth.** Space colonisation (attraction points sampled in each tissue, density per tissue type) or constrained
  constructive optimisation (CCO: add terminals one by one, minimising total volume under flow and pressure
  constraints).
  - CCO gives physiological branching statistics, space colonisation is faster.
  - Plan: space colonisation for the topology, then a CCO-style radius / position relaxation.
- **Radii.** Murray's law r³ = r₁³ + r₂³ with asymmetry; terminal radius ≥ 0.25 mm. Arteries and veins grow as
  paired trees (veins somewhat larger, slower, little pulsatility).
- **Flow boundary conditions.**
  - Inflow at the roots (aorta or CFD outlets, a cardiac waveform); outflow to terminal resistances (organ beds).
  - The venous return closes the loop.
  - Heart rate and stroke volume per case.
- **Output.** Segment list (centreline spline, r(s), material, parent / children) plus the network flow solution
  Q(s, t), p(s, t).

## 5. Truth export and the MR simulator

- **Eulerian truth on any grid:**
  - per time: v(x, t) (Womersley / CFD), lumen fraction (super-sampled SDF), tissue labels and fractions;
  - per voxel: mean velocity and intravoxel covariance (from sub-voxel sampling);
  - B0 and coil maps.
  - Exported as flow values (`flow4d`) straight into the flow viewer, and as `.hv` for training.
- **Lagrangian sources for the direct-sum simulator (`sim4d`):**
  - tissue points pushed by Φ_t;
  - blood particles advected through v (GPU RK4, the same integrator as `flow_pathlines`) with spin history
    (inflow saturation);
  - per readout: positions and velocities, so readout motion and acceleration are exact.

  The current simulator builds blood sources on the CPU (95 % of its runtime); this moves to the GPU.
- **Sequence-agnostic:** any trajectory and encoding (stack of spirals, rotated spirals, radial; 4-, 7- or 13-point).
  It is the testbed for your own sequence.

## 6. Visualisation

- **Flow viewer** (`plotting_website/flow`): every exported truth is a flow value, so PC-MRA, streamlines, pathlines,
  planes and Q(t) work as they are.
- **Phantom view** (new layer in the flow viewer):
  - vessel centrelines as lines coloured by radius or flow, with the segment's Q(t) on click;
  - chamber and valve surfaces over the cycle (marching cubes of the SDF at display resolution);
  - motion field glyphs.
- `physfigs` and `run_figs` for the reconstruction side: B0, sequence, concomitant fields, GNL, coverage, and errors
  against truth.

## 7. Milestones

| # | content | gate |
|---|---|---|
| P0 | vessel forest (growth, Murray radii); 1D pulsatile network; Womersley segments; SDF; truth export at any grid; viewer | flow conserved at junctions (< 1e-6); divergence of the exported field in straight segments ≈ 0; Q(t) through planes equals the network solution; statistics of radii / branching |
| P1 | procedural torso + labels / textures + B0; motion (respiration, cardiac, distension); moving sources in `sim4d` (GPU particles) | k-space from the phantom; static and moving cases; reconstruction runs end to end |
| P2 | heart chambers + valves: Tier A parametric flow, then Tier B LBM with moving walls and resistive valves | CFD validation gates (`cfd_training_data.md` §6); jets and regurgitation visible in the reconstruction |
| P3 | random phantom factory (seeded); benchmark set (fixed) and training set; `DenoiserZP` training | train / test split by phantom; prior beats the no-prior reconstruction |

Order: P0 → P1 (first end-to-end reconstruction on the phantom) → P2 → P3. Tier A chambers can enter P1 so the heart
moves from the start.

## 8. Open decisions

- Procedural torso vs R4 scaffold for the first benchmark phantom. Proposal: procedural first; R4 second for
  realism checks.
- Smallest vessel radius, which sets how many segments there are: 0.25 mm gives ~10⁴ segments in a thorax —
  affordable for analytic flow; the cost is in sampling.
- Truth sampling resolution for the MR simulation: 0.25 mm blood particles (~50 M) as in `sim4d`; ≈ 1 h per 10-min
  scan on our GPU once particles are on the GPU.

## 9. Decision and status (2026-11-05)

**Decision:** the scaffold is the **Z-Anatomy** atlas (CC BY-SA 4.0, built on BodyParts3D CC BY-SA 2.1 JP). No XCAT
licence is available. The "procedural torso" is dropped. Fine vessels are grown into the organs from the atlas
vessels' anatomical entry points. Tissue parameters come from 3T literature tables per label.

**Licence:**
- Attribute Z-Anatomy and BodyParts3D everywhere.
- Share-alike applies to *distributed* material derived from the atlas: phantom geometry, label volumes, and probably
  the simulated datasets. Our code is unaffected.
- `Kidney.l/.r` (CC BY-NC, Lissie Cowley) is excluded; BodyParts3D kidneys replace it later.

**Export** (`rustlib/tools/phantom/blend_export.py`, run in a portable Blender 4.5.15 LTS under
`rustlib/target/tools/`, with `--disable-autoexec`, so no scripts embedded in the .blend run):
`data/hasty/external/zanatomy/export/`
- `vessels.npz` / `vessels.json`:
  - 654 named vessel curves, 250k samples at ≤ 0.5 mm, world coordinates in m;
  - radius = bevel depth 0.5 mm × the point radius, so: p5 0.26 mm, median 0.88 mm, p95 4.1 mm, max 14.7 mm
    (ascending aorta); 122 m of centreline in total;
  - parent links, grouped by region with roots such as carotids, subclavians, abdominal aorta, femorals, vena cavae.
    They still have to be stitched into whole trees by endpoint proximity.
- `meshes_<system>.npz` / `meshes.json`: all meshes, triangulated, with a closed-surface flag.
  - Heart chambers, valve leaflets (aortic, pulmonary, mitral, tricuspid), papillary muscles, organs, bones, muscles,
    body regions.
  - Most organ and chamber meshes are not watertight, so voxelisation uses the generalised winding number (robust to
    holes).
- `figs/vessels.png`: front / side / thorax projections.

**P0 status (2026-11-05, evening):**

1. *Vessel graph* (`vessel_graph.py`):
   - Same-system node merge only. Coronary arteries and veins had been fused along the grooves, which put the
     coronary sinus into the arterial tree.
   - Proximal free-end joins: 13, e.g. left carotid → aortic arch, azygos → SVC, internal thoracic veins →
     brachiocephalic veins, ileocolic → SMA.
   - Heart connectors: pulmonary trunk → its valve (40 mm), SVC → RA (11 mm), each pulmonary vein → LA (2–7 mm).
   - Roots per component (`roots` in graph.json).
2. *Tissue labels* (`phantomvox` CUDA winding number + `phantom_labels.py`, 3T table
   `crates/hasty-phantom/data/atlas_tissues_3t.json`):
   - Mirrored atlas objects (det < 0, the `.r` copies) had inverted normals → fixed at export.
   - Shells (skin, myocardium, gut walls) are filled by ray-cast outer-triangle classification plus the
     outer-only winding number. Each heart cavity is assigned to the chamber whose reversed inner surface
     encloses it most.
   - Thorax at 2 mm in 8 s on the GPU. Chamber volumes: LA 62, LV 118, RA 95, RV 136 mL; myocardium 261 cm³;
     lungs 3.5 L.
   - Figures in `data/hasty/phantom/p0/figs_2mm/`.
3. *1D pulsatile network* (`network_flow.py`, CPU, seconds):
   - Womersley transmission-line π-segments ≤ 5 mm, Olufsen wall stiffness; nodal admittance per harmonic
     (the graph has cycles).
   - Arteries: ventricular outflow at the root, 3-element Windkessels at the terminals.
   - Veins: atrial pressure waves at the roots, steady capillary sources.
   - Flow split by first-order territories (e.g. renal 9.5 %, coeliac 10 %, carotid 6.5 % each), r³ within.
   - Results at HR 70 / SV 75 mL:
     - aortic root 127/62 mmHg, peak 425 mL/s;
     - brachial 139/52, femoral 136/64 mmHg, femoral flow triphasic;
     - renal 9.4, coeliac 9.6, SMA 8.2, coronaries 2.5 / 2.0 mL/s;
     - SVC / IVC 23 / 49 mL/s with S, D and a-wave reversal; portal 17.5 mL/s;
     - pulmonary trunk peak 370 mL/s, RPA / LPA 49 / 31 mL/s.
   - Gates: solve residual 8e-16; junction mass conservation 2e-11 × CO (gate 1e-6).
   - Figures in `data/hasty/phantom/p0/flow/figs/`.

**Known issues / tuning:**
- Descending-aortic diastolic oscillation is too large (+80 mL/s late-diastolic forward flow) and PA pressure is
  37/8 mmHg (normal ≈ 25/10). Tune the peripheral compliance and Rc split.
- Coronary flow is not yet diastolic-dominant (no intramyocardial pump).
- Pulmonary-vein pulsatility is weak (capillary beds are ideal sources; a PA-terminal → PV-terminal capillary link
  would transmit the RV pulse).
- Atlas radii are low for some veins (portal vein 2.75 mm → 74 cm/s mean); a radius-correction table for named
  vessels is needed.
- No aortic-valve leaflet mesh matched the connector regex (the root stays at the coronary ostia).
- No oesophagus mesh in the atlas (to add as a tube).

**Truth export (2026-11-05, later):** `rustlib/tools/phantom/run_p0.sh [--graph] [--labels] [--publish]` runs the
chain; outputs in `data/hasty/phantom/p0/`.
- *Graph fixes* (`vessel_graph.py`):
  - radius corrections and floors for undersized named vessels;
  - aortic-valve connector (leaflets are "Left/Right coronary leaflet", "Non-coronary leaflet");
  - Hermite valve connectors along the valve axis;
  - pulmonary-root separation from the aorta (≈ 8 mm, Gaussian-weighted over the pulmonary tree);
  - bridges prefer targets ≥ 1.5 × the fragment radius; renal fragments bridge to the renal vessels;
  - de-overlap (`deoverlap.py`): side-by-side lumens of non-adjacent vessels shrink ≤ 15 %.
- *Network* (`network_flow.py`):
  - viscoelastic wall (loss tangent 0.15);
  - physiological territory fractions (≈ 7 % unassigned);
  - two Murray-floor iterations (`murray_floor.py`): r ≥ 0.75 c Q̄^{1/3} fixes atlas segments that are too thin for
    their subtree's flow.
- *Pack* (`pack_vessels.py`):
  - Womersley profile table g(ξ; α);
  - branches the atlas draws inside their parent hand their flow to the parent until they emerge.
- *Truth* (`phantomflow`, CUDA):
  - capsules along the centrelines; S³ sub-samples per voxel;
  - ownership by signed depth (min d − r);
  - per-capsule, per-harmonic contact compensation (∫g dA / ∫_owned g dA, ≤ 2×), so pressed vessels keep their flux;
  - output: sparse per-voxel velocity harmonics (continuous in time), blood and wall fractions, class, owning sample.
  - Thorax at 1 mm in 1 s; heart at 0.5 mm (400 × 400 × 480) in 3 s.
- *Assembly* (`phantom_truth.py`): final labels (vessel lumens / walls over the tissue labels), gates, figures, and a
  flow-viewer export (box-averaged).
- *Gates:*
  - plane flux, i.e. mass conservation through a junction-free disk, vs Σ network Q;
  - at 0.5 mm: aorta 0.1–0.4 %, carotid / subclavian / brachiocephalic < 0.1 %, RPA / LPA 0.5–0.9 %, right coronary
    0.9 % (r/h 3);
  - vessels pressed together (pulmonary trunk vs aorta, SVC, IVC, right superior pulmonary vein): 3–6 %;
  - divergence |∇·v| h / |v| median 0.4 %, p95 13 % (junction and contact zones).

**P1 — motion (2026-11-05):** outputs in `data/hasty/phantom/p1/`.
- **Cardiac kinematics** (`cardiac_motion.py` → `motion/cardiac_motion.hv`): displacement of the atlas pose on a
  2 mm heart grid at 48 samples per cycle (float16 mm, periodic Catmull-Rom in time).
  - Modes:
    - AV-plane descent, 14 mm;
    - twist, 9°;
    - per-chamber volume modes B_c = ∇Γ_c with ΔΓ_c = 1_c (Newtonian potential by FFT: divergence only inside
      the chamber).
  - AV descent and twist are Helmholtz-projected so the myocardium stays incompressible.
  - Per time sample: Newton on the four chamber volumes, then a nonlinear correction U ← U + ∇ψ with
    Δψ = −log det(I + ∇U) on the myocardium and valves.
  - Results:
    - volumes hit their targets to 0.001 mL; myocardium volume change ≤ 0.4 %;
    - EF 67 / 58 % (LV / RV);
    - mitral annulus travel 14 mm, tricuspid 19 mm, apex 2 mm, annulus velocity ≈ 10 cm/s;
    - wall thickening visible.
  - Volume curves are mass-consistent with the network:
    - V_LV = EDV + ∫(Q_MV − Q_AO), V_LA from the pulmonary-vein inflow;
    - right heart alike;
    - E/A filling lobes.
- **Respiration** (`resp_motion.py` → `motion/resp_motion.hv`, 4 mm trunk grid):
  - internal organs SI (diaphragm 15 mm; heart 10.7 mm; lung apex 0; abdomen tapering), AP 7 mm (spine fixed),
    lower-rib LR 3 mm;
  - Lujan cos⁴ breaths with period and depth variability (`motion.Breathing`).
- `motion.py`: position / velocity / inverse at arbitrary points and times (composition Φ_r ∘ Φ_c).
- **Moving truth** (`phantom_4d.py`, GPU / torch; connected components on the CPU):
  - vessels move with the tissue and distend with the network pressure (aortic radius swing ≈ 10 %);
    `phantomflow --frames` voxelises each frame (wall velocity added to the Womersley flow).
  - Tissue labels are pulled back through the inverse motion from a high-resolution label source (1 mm thorax,
    0.5 mm heart); tissue velocity is Eulerian.
  - Chambers carry potential flow v = ∇φ, solved by matrix-free Jacobi-CG:
    - walls: the tissue velocity, corrected per chamber to dV/dt taken from the defining flows
      (Q_MV − Q_AO, …);
    - ostia: each heart root's network flow (only roots within 10 mm of a chamber; fragment roots are fed by
      their tree);
    - valve planes cut the outflow tracts; mitral / tricuspid faces open while filling > 0;
    - chambers cleaned with a 2 mm opening.
  - Network fix: detached fragments now draw their flow from the main tree at their feeding node (coupled), so the
    pulmonary root carries the full RV output.
  - Speed: 1 mm thorax (20 M voxels) in 1–2 s per frame; 0.5 mm heart (53.5 M voxels) in 4–11 s per frame.
  - Gates:

    | gate | gated 1 mm | 0.5 mm heart | real-time 1 mm (8 s, breathing r 0–1.07) |
    |---|---|---|---|
    | max per-component mass imbalance [mL/s] | 7.9 (end-diastole valve transition; else < 2) | 7.9 | 7.9 |
    | mitral / tricuspid flux vs filling model | 0.17 % | < 0.5 % | 0.11 % |
    | chamber speed p99.9 / max [m/s] | 1.65 / 3.9 | 1.63 / 5.3 | 1.65 / 4.1 |
    | vessel flux through moving planes (thoracic aorta, brachiocephalic, carotid) | 0.1–0.24 % | 0.04–0.28 % | 0.07–0.23 % |
    | same, vessels pressed against the aorta (RPA, IVC, SVC) | 4.6 / 1.7 / 11 % | 4.5 / 1.7 / 10.8 % | 4.1 / 1.7 / 9.9 % |

    The moving-plane gate does not yet filter voxel ownership, so aortic blood contaminates the RPA and SVC disks.
  - Known limitations:
    - The atlas mitral orifice is small (3.8 cm² of faces at peak E), so E-wave speeds are 1–1.6 m/s
      (physiological 0.6–1.0).
    - Potential flow has singular velocities at orifice rims (max 4–5 m/s in a few voxels).
    - No vortices, turbulence or jets: that is the CFD tier.

**P3 — phantom factory (`make_phantom.py <seed>`):** `data/hasty/phantom/factory/ph<seed>`. Seed 9001 is held out
(never in training / validation).
- Sampled per seed:
  - heart: HR 55–85, SV 60–85 mL, E share, AV descent, twist;
  - circulation: mean systemic / pulmonary pressure, compliance, territory jitter 15 %;
  - vessels: radius jitter 6 % per vessel;
  - breathing: period, cos^{2n} shape, variability, SI / AP / LR depths;
  - tissue (`tissue_variation.py`): PD / T1 / T2 / T2* / ff jitter, contrast native / gadolinium / ferumoxytol,
    texture gain;
  - textures: random Fourier modes per tissue class in the atlas frame, resolution-free, moving with the tissue,
    modulating PD (field a) and T2* / T1 (field b).
- Pipeline: graph copy + jitter → network (+ Murray floor) → pack → cardiac motion → respiratory motion →
  `phantom_4d` truths (gated / real-time; the truth carries the textures `tex`).
- Cost: ≈ 6 min per phantom (cardiac calibration on the CPU), ≈ 1.5 min per 1 mm truth.

**k-space from the moving phantom:**
- `prep_sim.py`:
  - tissue sources in the reference pose: 1 mm, lumen-excluded, water / fat split, the phantom's table and
    textures;
  - B0 from the χ map (FFT dipole, 2nd-order shim) attached to the tissue;
  - Eulerian blood frames from `phantom_4d --blood-export`;
  - the breathing table, body mask, prescription.
- `simphantom` (Rust / CUDA):
  - readouts are binned in absolute time (10 ms); per bin, tissue moves by cardiac (ECG phase) and respiratory
    motion, and blood comes from the nearest cardiac frame, shifted by respiration and advanced by v Δ;
  - gradient nonlinearity, slab profile and coil sensitivities are applied at the moved positions;
  - the direct sum is split over source chunks (f64 atomics), with k·(u + δ J v) moving the position to each
    excitation;
  - outputs follow the `sim4d` format (recon_p2np, show_run); resumable chunks.
- `sim_truth.py`: the truth on the reconstruction grid (gated, mean breathing state, 2 × box average).

**Next:**
1. Chamber flow beyond potential flow: diastolic vortex ring (divergence-free add-on), then LBM (Tier B).
2. Moving sources in `sim4d` (GPU particles / tissue points through `motion.py`) → k-space from the moving phantom,
   5D reconstruction tests.
3. 1 mm moving truth (GPU label pull-back), respiratory modulation of venous return.
4. Fine-vessel growth; phantom layer in the viewer.

## Tier B: cardiac lattice-Boltzmann flow (hasty-cfd, 2026-11-07)

Chamber and great-vessel-root flow driven by the phantom's prescribed wall motion, with valves; replaces the
potential flow of `phantom_4d` on request (`--chamber-flow lbm:<run>`), the old path stays the default. Validated on
the CPU, GPU = CPU (test `gpu_matches_cpu`), and run on ph0001 (see "GPU runs on ph0001" below: the first run exposed
leaks and boundary problems that changed the case builder, the valves and the inlets).

**Solver** (`rustlib/crates/hasty-cfd`):
- D3Q19, regularized BGK (Latt & Chopard) + WALE LES; sparse 8³ tiles over the union of the moving domain, A/B
  population arrays (pull), shifted populations (f − w) in fp32, one thread per node.
- **Bulk viscosity:** the trace of Π^neq relaxes with its own τ_b = 2 (ν_b ≈ 0.33 lattice); at τ0 ≈ 0.5008 the
  chambers' acoustic standing waves otherwise ring for whole cycles (±45 mmHg between the caval cuts).
- **One source for CPU and GPU:** the per-node physics is `kernels/lbm_core.h`, compiled by NVRTC for CUDA
  (`lbm_cuda.cu`) and by the C compiler for the CPU backend (`lbm_cpu.c`, rayon over tile ranges). The CPU path is
  the test oracle; the CUDA source is compile-checked by a test (NVRTC, no device).
- **Moving walls:** φ(x, t) = φ0(Φ_t⁻¹ x), with the reference signed distance of the blood domain pulled through
  `cardiac_motion.hv` (Catmull-Rom in time, trilinear) every 4 steps. The inverse is a Newton iteration (analytic
  trilinear Jacobian, warm-started per node, steps capped at one motion voxel): the fixed point X = x − U(X) diverges
  where ‖∇U‖ > 1, and the phantom's motion reaches 1.35 inside the ventricles at end-systole.
  Bouzidi interpolated bounce-back with the wall velocity ∂_tU; refill at equilibrium (neighbours' density, wall
  velocity).
- **Mass consistency of the moving walls is a gate**, because the chamber flows *are* the wall volume changes.
  - The dilating closed tube meets Q(x) = 2πRṘx to 3–7 %.
  - Two alternatives were tried and rejected: exact stores of the wall mass (stale stores on sliding walls biased the
    bulk flow), and the half-way wall term for all q (27 % too much inflow).
- **Valves:** kinematic, prescribed opening.
  - *Leaflets*: each anatomical leaflet is a fan of strips hinged on short chords along the annulus arc. A rigid
    leaflet on its commissure chord caps the open orifice at ≈ 41 % of the annulus. Strips rotate by θ = θ_open ·
    opening(t), with the exact solid velocity (annulus motion + rotation).
  - *Orifice*: a plate with an elliptical hole.
  - In both, the valve core is solid (bounce-back with the leaflet velocity) and the edges are partially saturated
    cells. A sheet of partially saturated cells alone leaked 77 % of the open flow; the hybrid leaks 0.0 %.
  - **Tissue frame** (default): the valve lives in the reference pose and is evaluated at each node's reference
    position, solid velocity = tissue velocity + leaflet rotation; it deforms with the collar and walls it seals
    against (a rigid disk on the moved centre opened gaps at the deformed annulus). Thicknesses across the sheet are
    measured in lab units (|F n|), since the valve centre sits in the cavity's motion field, which squeezes it ~3×.
  - **Closed** (opening < 0.05): a full solid disk ≥ 2 mm thick in the tissue frame plus a rigid disk on the plane
    fitted to the moved annulus (radius + 3 mm); the closed strip fans do not tile the disk exactly, and their partial
    cells bridged atrium and ventricle.
  - **Quasi-static leaflets** (`push` 0, default): the leaflet geometry follows the schedule, the wall velocity is the
    tissue's only. A prescribed swing velocity pumps the swept volume through the valve. In a sealed tube a 30 ms
    opening built −7 % / +9 % density across it (test `opening_valve_conserves_mass`, 0 with push 0). In the heart it
    put +6 % density in the RV and a 6 m/s backflow at tricuspid opening.
  - **Pressure trigger** (optional, off by default): opening = max(schedule, o_p), with o_p from the filtered
    up/downstream probe gradient (a 0D valve law with hysteresis). It acts only while one side of the valve is a
    sealed compartment (region graph: no open valve, no cut).
    - Off because the LBM pressures are gauge pressures about the outlets: no 120-vs-10 mmHg systolic difference, so
      unrestricted it held the AV valves open in systole.
    - With quasi-static leaflets it no longer improved anything.
  - Leaflet FSI is not attempted; the kinematic leaflets give realistic jets and closure with prescribed timing.
- **Inlets and outlets:** cut planes moving with Φ_t; slab, flux layer and sponge membership decided at the node's
  reference position (a neighbouring chamber the motion brings near a cut became outlet nodes inside the RA). The flux
  layer is one reference unit thick, so fluxes and sections are divided by its lab thickness 1/|F⁻ᵀn| (F at the cut
  centre). Uncorrected, the aortic outlet read 110 against 96 mL/s where the motion stretches the root.
  - A slab of equilibrium nodes beyond each cut. Inlets take ρ from the interior, outlets take u from the interior.
  - Venous inlets: a plug normalised by the measured section, with a flux controller (the equilibrium slab alone
    delivered ≈ 8 % too little), and **inertial**: q(t) = share × Q_PV / Q_VR plus one correction ΔQ per atrium,
    dΔQ/dt = (−p̄_cuts − R ΔQ) / L − ΔQ / 1 s, L = ρ ℓ / A_veins (ℓ = 6 cm pulmonary, 10 cm caval), R for ζ = 0.7
    against the atrium's LBM compliance V / (ρ c_s²), split by the shares. Plain velocity inlets leave a closed atrium
    or ventricle without a pressure reference: the LBM's chamber volume change differs from the model's by ≈ 10 % and
    the excess compressed them (mean Δρ +1 %, bursts when valves open). Pressure inlets (zero upstream inertia) let the
    veins slosh against each other; per-cut inertial inlets sloshed SVC against IVC. Both stay available.
  - Arterial outlets: pressure or RCR, with a 6 mm sponge (τ + 0.3) before the plane.
- **Regions in the solver:** the builder's region map (reference pose) is sampled at each node's reference position.
  - Diagnostics: per-region fluid volume and mean density (chamber volume curves against the cardiac model).
  - The region graph (free contacts, the two sides of each valve, cuts) gives compartments and their sealed phases.
- **Wall shear stress (native):** at wall nodes (fluid next to a wall or leaflet, not partially saturated):
  - stress σ = −(τ0 − ½)/τ Π^neq_dev, from the post-collision populations: the strain at the node's τ (molecular +
    sponge), the stress with the molecular viscosity only — the sponge's artificial viscosity had inflated the outlet
    WSS ~190×;
  - normal from ∇φ, τ_w = σn − (n·σn)n;
  - accumulated per output cell and phase bin (accumulator slots 10–14, `acc_stride` 15 in the meta).

  Evaluated at the first fluid node, about ½ cell from the wall. Poiseuille: −8.5 % on R = 6 cells, which is (R − ½)/R.
- **Output:** phase-binned Eulerian velocity (Σw, Σwu, Σwuuᵀ: mean and intravoxel covariance) per output cell (2³
  nodes by default), accumulated over the cycles after warm-up into `flow_bins.f32`. Diagnostics go to `log.jsonl`
  (cut fluxes relative to the moving cut vs expected, density, max speed). Checkpoint/resume through a STOP file or
  Ctrl-C (checked).
- **CPU tests** (`cargo test -p hasty-cfd`):
  - rest state;
  - Poiseuille: centre = 2 × section mean; delivered flux = target;
  - translating capsule: fluid = wall velocity to 2 %;
  - dilating tube: within 3–7 %;
  - valve geometry: closed seals; open clears the annulus plane;
  - closed valve flux ≈ 0 against open;
  - the lattice, the motion interpolation, the leaflet presets.

**Genericity rule** (user, 2026-11-07): fixes must survive a change of geometry, i.e. be equivariant under a
diffeomorphism of the phantom (labels, vessel graph, motion deformed together; the planned way to make more phantoms:
bulges, stenoses) and work outside the thorax later.
- Geometric decisions are made in the reference pose from labels, the vessel graph and the region graph, then
  carried by the motion: tissue-frame valves, reference-frame cut membership, region diagnostics.
- Lengths are physical (mm) with lattice minimums, never voxel counts: walls ≥ 2 mm and ≥ 4 cells, separations
  ≥ 1.5 mm and ≥ 3 cells, leaflets ≥ 1.5 mm and ≥ 3 cells.
- Valve and BC laws carry no anatomy constants.
- Still anatomy-specific:
  - the AV radius caps (16 / 17 mm, ≈ 0.73 × the contact-patch radius) — to be replaced by a fit to the atlas valve
    tissue, which comes as one connected component per valve;
  - the heart module of the builder (chamber names, valve definitions). Other body regions need their own module; the
    solver is generic.
- Check: the default builder at `--coarsen 2` (1 mm) seals with the same topology. Its valve fluxes agree with
  0.5 mm (corr ≥ 0.99, strokes within 2 mL); its jets are 12–15 % faster (thicker walls and leaflets in mm).

**Case from a phantom** (`rustlib/tools/phantom/lbm_case.py <phantom> <case>`, CPU, ≈ 90 s):
- **Domain:** chambers (+ valve tissue) from the 0.5 mm heart labels, plus great-vessel stubs from the heart roots
  (aorta 45, pulmonary trunk 30, venae cavae 20, pulmonary veins 15 mm along the main branch).
- **Walls the atlas lacks:** every blood voxel gets a region (chambers, stubs). Near each valve, the upstream /
  downstream boundary is the valve plane. Walls go wherever two regions touch, except:
  - venous stub ↔ its atrium;
  - a valve's two sides inside the valve disk.

  Solid collars sit around the annuli. Since the GPU runs:
  - walls on every D3Q19 link (face and edge contacts), grown to ≥ 2 mm; a 1 mm wall thins below a lattice spacing
    under smoothing, motion and sampling;
  - proximity walls: regions that must not exchange fluid and come within 3 voxels through a thin atlas gap get a
    wall too (RA ↔ aortic root, RA ↔ right superior PV and aortic ↔ pulmonary root leaked under the motion);
  - the vessel lumen takes precedence over a neighbouring chamber's labels, and an arterial stub over its own
    ventricle's labels beyond the valve plane (the aortic root was walled down to half its lumen);
  - each valve cylinder (radius R, ± 2.5 mm) is carved into the blood (31 % of the capped mitral disk was atlas
    myocardium); caps remove only their own stub beyond the slab (the 2.5 R kill cylinders deleted 11 mL of
    neighbouring vessels), and a thin slab beyond each cut is cleared of other regions;
  - checks with 18-connectivity (the 6-connected check had passed while left and right heart were linked);
  - `region.npy` saved with the case (diagnostics: `lbm_leaks.py`).
  - The atlas's left and right blood touch over 2849 faces with no septum.
  - The LA–LV contact is 15.7 cm², 4.9 mm out of plane.
  - Without these walls the domain would shunt and every valve would leak.
  - Checked: removing any valve's disk disconnects its two sides; left and right heart are separate components.
  - Stub passages ≥ their lumen; arterial stubs are not pinched.
- **Valves:**
  - aortic and pulmonary at the root connectors (three leaflets × 4 strips);
  - mitral (two leaflets, 5 + 7 strips) and tricuspid (three) on planes fitted to the AV interfaces, radius capped at
    16 / 17 mm;
  - opening from the cardiac model's valve flows (> 3 % of peak, 30 ms ramps); AV valves stay ≥ half open through
    diastasis (closed there they sealed atrium and ventricle into boxes the LBM compressed);
  - AV centre on the blood passage (atrium + ventricle + valve labels, planes −3…+8 mm, iterated). The contact
    centroid put the mitral disk 7 mm beside the inflow tract: atlas myocardium filled 30 % of the orifice 4 mm into
    the LV. Valve cylinders are carved from −2.5 to +4 mm (into the downstream chamber).
  - Moving the centre took the mitral jet p95 from 1.9 to 1.2 m/s.
  - Leaflets stay ≥ 3 cells and ≤ 85° / 80° open: a 2-cell leaflet is porous partial cells; near-parallel AV
    leaflets pinch gaps at the wall when they move to half open for diastasis. Either gave 4–7 m/s spikes.
  - Atlas valve tissue (the `valve` label, one connected component per valve) as geometry source: semilunar cusps are
    closed rings that agree with the vessel radii (the pulmonary ring is larger: 16.6 mm against 13.1); the AV
    leaflets hang open with partial hinge arcs and do not give a reliable annulus. Not used.
- **Venous inlets** carry the cardiac model's Q_PV / Q_VR split by the network's mean shares (inertial, see above).
  The network's own stub flows (64 of 98 mL/s for the PVs) miss the branches without a stub, and the mismatch drove
  the atria into suction.
- **ph0001 at 0.5 mm:** 430 mL domain, 13.4 k active tiles = 6.9 M nodes, ≈ 2.1 GB on the device, dt 20 µs,
  42.6 k steps per cycle, τ0 = 0.5008. 5 cycles take 275 s on the RTX 4090 (2.1 G fluid-node updates/s).

**GPU runs on ph0001** (2026-11-07; `data/hasty/phantom/factory/ph0001/lbm/{case,run}`, history in its README):
- **Tools:**
  - `cardiac_lbm --probe <steps>[:every] --snap s1,…`: the fastest nodes with their surroundings, and sparse snapshots;
  - `lbm_report.py <case> <run> [--snap]`: cut fluxes against targets, valve fluxes relative to the annulus against the
    cardiac model, jet speeds, closed-valve leakage, speed MIPs at four phases;
  - `lbm_leaks.py <case> <snaps>`: D3Q19 links between regions that must not exchange fluid, in the reference pose.
- **First run** (the case and kernels as first written): stable, but semilunar jets 2.8 m/s, mean Δρ +1 %, isolated
  4–5 m/s spots. Each was traced, in this order:
  - the aortic root walled down to 2.8 cm² 6 mm above the valve;
  - closed boxes behind velocity inlets;
  - pinholes through thin walls, and left and right heart linked through edge contacts;
  - closed valves bridged by partial cells and by the rigid valve frame;
  - outlet slabs inside the RA;
  - venous sloshing;
  - acoustic ringing;
  - a mitral annulus one third in myocardium.
- **Third round** (v18–v22):
  - AV valves centred on the blood passage: the mitral jet p95 1.9 → 1.36 m/s;
  - leaflets kept at ≥ 3 cells and 80° / 85°;
  - annulus cylinders carved to R + 2 mm (late-E gap jets gone);
  - native WSS, with molecular viscosity.

  Current (v22 = `run`): valve correlation 0.992–0.999, max speed ≤ 2.9 m/s; WSS median 0.3–0.6 Pa, p95 1.9–3.4 Pa,
  max ≈ 15 Pa.

  | jet (p95) | LBM |
  |---|---|
  | aortic | 1.98 m/s |
  | pulmonary | 1.49 m/s |
  | mitral | 1.36 m/s |
  | tricuspid | 1.25 m/s |
- **Second round** (each step one run, history in the data README):
  - pressure-triggered valves (helped the bursts, but held the AV valves open in systole);
  - per-region volumes (LBM chamber volumes match the model: LV 27.5–108.5 vs 26.9–111.6 mL);
  - the RV density rising 6 % while the tricuspid opened, then a sealed-tube test isolating leaflet pumping →
    quasi-static leaflets;
  - the outlet flux layer's stretch.
- **Current** (v17, cycles 4–5 of 5):
  - **Topology:** correct at every phase. RA | RV + PA | LV + Ao | LA in systole; each ventricle isolated in the
    isovolumic phases; LA + LV and RA + RV in diastasis with the roots isolated; left and right never connect.
  - **Valve flows against the cardiac model:** correlation 0.991–0.998, strokes 79–81 mL against 84.
  - **Arterial outlets:** aorta 96.6 / 96.2 mL/s mean, corr 0.993, peak 503 / 503; pulmonary 97.1 / 96.3, corr 0.993.
  - **Veins:** pulmonary veins corr 0.97–0.98, means within 2–4 %; venae cavae corr 0.85, means within 6 %.
  - **Closed-valve flux:** rms 0.1–2.8 mL/s.
  - **Density:** mean −0.02 %, range ±0.55 %. Largest speed 2.6 m/s.
  - **Jets, 95th percentile at peak:**

    | valve | LBM | physiological |
    |---|---|---|
    | aortic | 1.9 m/s | 1.0–1.7 m/s |
    | pulmonary | 1.5 m/s | 0.6–0.9 m/s |
    | mitral E / A | 1.9 m/s | 0.6–1.0 m/s |
    | tricuspid | 1.2 m/s | 0.4–0.7 m/s |
- **Open:**
  - **Jet speeds** (orifice areas):
    - AV annulus size from the atlas valve tissue instead of caps;
    - leaflet blockage;
    - the motion contracting the mitral annulus at E (R 16 → 13.5 mm);
    - resolution (0.5 → 1 mm: +15 %).
  - **Caval waveforms:** corr 0.85.
  - **Stroke deficit:** 4–6 %.
  - **Right-side imbalance:** caval in 104 vs PA out 97 mL/s.

**Viewer export:** `rustlib/tools/phantom/lbm_export.py <run> [--step k] [--frames F] [--publish]`. It writes the
flow4d value of phantom_4d's viewer_flow.hv:
- `vel`: [F, 3, Z, Y, X] m/s, the bins as frames;
- `mag`: blood coverage per cell and frame (the moving heart);
- `mask`, `spacing`, `dt`, `venc`;
- `wss` / `wss_mag` (Pa) when the run has them.

`--publish` puts it on the hasty server for `plotting_website/flow`.

**Resolution:**
- `lbm_case.py --dx-mm` decouples the lattice from the label voxel (the sdf is sampled trilinearly). 0.25 mm on ph0001
  is 41.8 M nodes, 10.8 GB on the device, ≈ 45 min for 4 cycles alone on the GPU: the uniform reference.
- **Result** (v17 geometry, cycles 3–4):

  | quantity | 0.25 mm vs 0.5 mm | 0.5 mm: cycles 5–6 vs 3–4 |
  |---|---|---|
  | jets (p95) | ≤ 4 % (aortic 1.87 / 1.95 m/s, mitral 1.86 / 1.88) | — |
  | kinetic energy | 1–3.5 % | 0–3.5 % |
  | p99 speeds | 3 % | — |
  | fluxes, strokes | agree | — |
  | pointwise phase-averaged velocity | 25–35 % rms, 62 % in diastasis | 16–34 %, 59 % in diastasis |

  - **0.5 mm is converged for the integral quantities.** The pointwise difference matches the cycle-to-cycle
    variability at the same resolution, so it is statistics, not resolution: the flow is transitional (jets,
    diastolic vortices).
  - A gated phantom needs the ensemble mean: ≈ 30 % per-cycle spread → tens of averaged cycles for a few-% mean (the
    accumulator's Σuuᵀ keeps the cycle / intravoxel covariance).
  - `run_long` (v22 case, 24 cycles, 20 averaged, 23 min): the 2-cycle mean is 14–20 % off it (39 % in diastasis), as
    the variability predicts; the 20-cycle mean is thus ≈ 6–7 % pointwise (≈ 13 % in diastasis). Integrals unchanged.
  - Refinement is not the priority now. It stays the plan for small vessels (lumen < ~16 cells) and a whole-body
    model.
- Local refinement, if the reference says it is needed:
  - 2:1 block levels on the 8³ tiles, acoustic scaling (dt halves on the fine level);
  - a mass-conserving (volumetric) interface;
  - static regions, the union over the cycle of the valve, root and jet zones.

  Criteria must stay generic: cells per lumen diameter from the vessel graph, distance to valves, radius drops, later
  a posteriori from a coarse run's strain / vorticity / WSS. Gain: valve / root / jet zones are ≈ 10–15 % of the
  volume, so ≈ 3× the 0.5 mm cost against 16× for uniform 0.25 mm. Essential for a whole-body model.

**Use:**
- `cardiac_lbm <case> <run>` (GPU), resumable.
- Then `phantom_4d.py … --chamber-flow lbm:<run> [--lbm-vessels]`:
  - chamber voxels get the LBM velocity, linear between phase bins, pulled back through the respiration with its
    velocity on top;
  - uncovered voxels keep the potential flow;
  - the coverage is reported per frame in gates.json.

**Next (flow phantom only; no reconstruction or k-space for now, user 2026-11-07):**
1. Valve geometry from the atlas valve tissue (annulus ring of the semilunar components, the atrial end of the AV
   funnels; the open AV orifice from the leaflet surface), replacing the radius caps.
2. Jets to physiological ranges; the LV diastolic vortex ring; caval waveforms.
3. Motion accuracy: the cardiac motion inside the cavities (‖∇U‖ up to 1.35, det down to 0.04), the annulus dynamics.

**Later:**
- Distension of the stubs.
- Coronary ostia.
- RCR outlets tuned to the network.
- Cumulant collision if the low-τ stability needs it.
- Intramyocardial coronary pump for the coronary goal.


## Tier B2: beat library, spin history, diffeomorphic phantoms, training cases (2026-11-07)

**Beat library** (`cardiac_lbm --beats`): instead of only the ensemble mean, every cycle's phase bins are stored per
occupied output cell (1 mm at 0.5 mm lattice), f16, `<run>/beats/beat_<cycle>.bin`, appended bin by bin (resumable
at bin boundaries). 28 fields: coverage w, mean u (3), covariance (6), third central moments (10), WSS (3), spin
history mz, mz_std, cov(s, u) (3). Older libraries: 23 (no spin) / 25 (no cov(s, u)); readers take all three.
- `run_beats` (ph0001, v22 geometry, 30 beats, no spin, 42 min): ≈ 0.9 GB / beat. `run_spin` (case_v23, spin for the
  flow protocol): ≈ 1.1 GB / beat.
- `simphantom --beats <run>`: each heartbeat of the scan is a library beat (seeded shuffle, no back-to-back repeat,
  RR-stretched); each covered blood source becomes six sub-sources (two per principal axis of the cell covariance,
  weights matching variance and axial skew: exact mean and covariance). A beat's moments are computed on first use
  (LRU `--beat-cache`, default 6; precomputing 30 beats took 58 GB). meta.json records the heartbeat → beat sequence.
- Why: a gated reconstruction of a long scan averages many distinct beats; the truth is their mean, and the bias of
  the reconstruction against it (intravoxel dephasing of skewed distributions, beat-to-beat mixing in a bin, signal
  weighting) can be studied against the stored moments.

**Spin history** (`cardiac_lbm --spin [k=v,…]`): relative M_z s = M_z/M0 as a passive scalar.
- Per step: BFECC semi-Lagrangian advection (a single trace diffuses like upwind, D ≈ u Δx/2: centimetres over a
  chamber residence), then RF saturation (1 − cos α(z)) dt/TR and T1 recovery. α(z) from the Bloch-simulated slab
  profile (`hasty_mrsim::rf::SlabPulse`, windowed sinc), slab of any normal.
- Inflow: each venous inlet has an `upstream` path from the builder (vessel graph beyond the cut, main branch to its
  end, transit times from the network mean flow; systemic: capillary 2 s, fresh start; pulmonary: from the pulmonary
  arteries' flux-weighted outflow s (`from`, cycle-averaged in the solver), through the lung, total = pulmonary mean
  transit time 5 s). The slab saturates whatever part of the path lies inside it.
- Validation: pipe in a uniform slab matches the exact transit solution to 0.3 % along the centreline streamline;
  a recirculating `from` loop converges to its fixed point; GPU = CPU (RMS Δs 1e-6).
- T1 is the phantom's blood T1 (case.json "blood", its contrast state): ph0001 is gadolinium (T1 0.39 s), so
  saturation recovers within ≈ 0.2 s; native blood (1.9 s) would show strong inflow contrast. Example at native T1,
  TR 6.8 ms, 12°, 160 mm axial slab: IVC s_in 0.92, SVC 0.31, pulmonary veins 0.14–0.37, aortic outflow 0.15.
- simphantom: a library with a spin history replaces the steady-state blood signal: each sub-source has
  s = E[s] + cov(s,u)ᵀ Σ⁻¹ (u − μ) (exact E[s], E[s u] over the six points), amplitude M0 s sin α(z), and the blood
  PD loses the constant inflow gain. The library's protocol (TR, flip, slab) must match the scan (checked).
- Not modelled: respiration moves blood relative to the slab (the LBM runs in the cardiac frame); matters near slab
  edges only.

**Diffeomorphic phantoms** (`diffeo.py`, `make_phantom.py <seed> --diffeo <preset|spec>[:seed]`):
- φ = exp(v) of a stationary velocity field on a 3 mm grid over the atlas box + 150 mm margin (tapered to zero in the
  margin only: φ = identity outside, where the vessel graph continues): habitus (Gaussian random field + anisotropic
  scaling), vessel features placed on a named vessel at an arclength fraction (anisotropic radial kernels: "bulge",
  "narrow"; "rotate" in a ball). Presets: habitus, aneurysm (ascending aorta × 1.45), narrow (ascending aorta × 0.7 +
  pulmonary trunk × 1.25).
- Applied to the atlas inputs only: labels L(φ⁻¹(x)) sampled from the finest atlas volume containing the point, graph
  points φ(p), lumen radii from a ring of wall points mapped through φ (finite strain). Every later step (network,
  cardiac / respiratory motion, truth, LBM case) re-derives its physics on the deformed anatomy; prep_sim, sim_truth
  and lbm_case pick `<phantom>/atlas` up by themselves.
- Gates (aneurysm, seed 101): det J 0.83–2.68, inverse consistency 0.13 mm, aorta R 12.5 → 17.8 mm (asked × 1.45).
  ph0101 built in 10 min (network Kirchhoff 1.6e-11, cardiac fit 6e-4 mL, mitral error 0.13 %), its LBM case seals
  all four valves without any change to the builder.

**Training cases** (`to_training.py`): truth4d → 7-point dual-VENC complex case in the RACLETTE tier-1 layout
(`data/hasty/train/phantom`, `--index` writes index.json; `hvtool pack-tree` packs). Spoiled-GRE magnitude from the
phantom's tissue table with textures, drawn flip / TR / TE / blood inflow gain, smooth random background phase,
VENC_lo 1, VENC_hi 1.5–3.5 m/s, complex mean of the 1 mm truth voxels per 2 mm case voxel (intravoxel dephasing).
`tools/denoiser/dataset.py` joins `data/hasty/train/phantom` to RACLETTE (weight 0.2 of the case draw by default;
`HASTY_TRAIN_PHANTOM=0` to leave it out).

**Results of 2026-11-07 (evening):**
- Diffeomorphic phantoms through the whole chain (factory → LBM case → 10-cycle LBM with beats + spin → LBM truth →
  training case), no per-geometry fixes:

  | phantom | variant | contrast (blood T1) | LBM | chamber coverage of the truth | notes |
  |---|---|---|---|---|---|
  | ph0101 | aneurysm (aorta 12.5 → 17.8 mm) | gadolinium (0.37 s) | 10 cycles, flux corr aortic 0.995, outlet means ±2 % | 98.6 % | |
  | ph0102 | narrow (aorta → 7.7 mm) + PA × 1.3 | ferumoxytol (0.10 s) | phase-averaged stenosis jet 3.75 m/s | 99.0 % | node spikes to 6.4 m/s |
  | ph0103 | habitus (det J 0.79–1.22) | native (1.68 s) | inflow s 0.24–0.88 | 99.2 % | |

- Lattice velocity scale: the builder now takes v_max = max(2, Q_peak / A_min) from the anatomy (`--v-max`
  overrides). A stenosis at v_max 5 (τ0 0.50032) gave the same phase-averaged jet but instantaneous spikes to 22 m/s;
  τ0 closer to ½ is the larger risk than Mach at u ≈ 0.15 (see ph0102/lbm/README.md).
- First physics-bias numbers (ph0001, 20 s dev scan, VENC 1.0 / 1.5, 2.5 mm voxels; research/velocity_bias.md):
  PC skew + beat mixing |v̂ − κ1| median 0.03 cm/s, p95 0.8 cm/s; signal weighting with the gadolinium spin history
  |κ1 − ū| median 0.04, p95 0.46 cm/s. Sub-cm/s: at this resolution the reconstruction, not the measurement physics,
  will dominate the bias.
- The 20 s dev scan (≈ 12 readouts per frame and encode) reconstructs to noise (velocity RMSE 0.69 m/s): it checks
  the pipeline, not the method. Benchmarks need minutes of data (`sim_lbm_2min`).

## Tier B3: great-vessel trees, coronary zone, scanner-like prescription (2026-11-07, code; GPU runs pending)

**Vessel trees** (`lbm_case.py`, default; `--stubs` = the old 45/30/20/15 mm stubs). The LBM domain is now the
chambers plus every branch of radius ≥ `--tree-rmin-mm` (2.5 mm = 5 cells) from each heart root inside the 0.5 mm label
box (less 8 mm). The ph0101 aneurysm 60 mm up the aorta was outside the 45 mm stub.
- Rules:
  - the continuation at a junction is always kept;
  - side branches shorter than max(4 mm, 2 R) are dropped;
  - dead-end branches without network flow (graph artefacts) are dropped;
  - cap kills reach at most max(4 R, 6 cells) beyond the cut plane.
- Outlets:
  - per arterial tree, every cut but the largest gets the 1D network's outflow waveform (a velocity cut). The flows
    are scaled so the tree's mean equals the root flow: flow below R_cut is shared out;
  - the largest cut (the abdominal aorta; a segmental pulmonary artery) is a pressure outlet and takes the rest;
  - veins keep the share-grouped inertial inlets.
  - Windkessel outlets were rejected: in the weakly compressible lattice the pulse pressure becomes density
    (≈ 0.4 mL/mmHg artificial compliance at c_s = 14 m/s).
- ph0001: 42 cuts, 686 mL (430 before), 2 components, all valves sealed, 17.9 M nodes, 5.5 GB, ≈ 8–11 min per
  cycle on the 4090 (plan).

**Coronary zone** (`lbm_coronary.py`): nested fine domain at 0.15 mm (sdf 0.25 mm). It is local refinement by zones;
in-solver multi-level refinement is deferred.
- Domain: the coronary arteries of the graph from both ostia, R ≥ 0.6 mm (4 cells). ph0001: 76 branches, 38 outlets,
  6.8 mL, 262 mL/min.
- Outlets: lumped Kim-type model with an intramyocardial pump, P_im = κ P_LV (left, κ 1) / κ P_RV (right, κ 0.5).
  P_ao comes from the network root; P_LV / P_RV from the valve phases. With τ_im = 1 s the left systolic / diastolic
  flow ratio is 0.37 and the right 1.18. Ostia are pressure inlets.
- Solved in the heart's reference frame (static geometry): a moving lattice would cover the swept volume (329 M
  nodes, 87 GB). With prescribed outlet fluxes in a loop-free tree, frame accelerations only add pressure.
  phantom_4d `--ref-flow` maps the result to the lab as F v_rel + u_wall.
- Size: 15.9 M nodes, 4.7 GB, ≈ 10–12 min per cycle.
- Not yet: simphantom beat-library sources for the reference-frame (coronary) library.

**Scanner-like prescription** (`prep_sim.py`):
- `--slab auto`: from 10 mm below the heart (a bit of the IVC) to 5 mm above the aortic arch. ph0001: z 1227–1403 mm,
  176 mm. The prescription records the slab; simphantom centres its RF profile there.
- `--fov auto`: the body's x, y extent in the slab + 10 mm; z = 1.3 × slab.
- `--src-roi heart --src-step-out k`: 1 mm sources in the heart / great-vessel box, coarser elsewhere. The direct-sum
  cost scales with the number of sources: 1 mm everywhere is ≈ 15 h for a 10-min scan.
- The atlas cut the arms at x = ±200 mm (the meshes have them: body at ±247 mm in the thorax). `run_p0.sh` now voxelises
  520 × 320 × 360 mm. Rebuilding p0 and the phantoms needs the GPU (phantomvox).

**simphantom k-space to disk:** data_clean.part.npy, pre-sized; finished chunks write their rows and leave only a
marker state; RMS and the noisy copy are streamed. A 10-min scan is ≈ 54 GB at 32 coils and was held in RAM up to 4×.
Older chunk states are moved into the file on resume.

**Realistic scan driver** (`rustlib/tools/phantom/realistic_scan.sh <phantom> [--beats 15] [--flow-min 10] [--b0-min
2.5] [--atlas]`). One resumable command for the GPU queue. Steps:
1. optional atlas with the arms;
2. slab from the anatomy (`slab.py`, shared with `prep_sim --slab auto`);
3. vessel-tree LBM beat library with the flow protocol's spin history;
4. coronary zone and its library;
5. 1 mm blood frames (LBM chambers, great vessels, coronaries via `--ref-flow`);
6. 10-min 4D flow: prescription, simphantom with both libraries, truths incl. fat fraction and the bias truths;
7. 2.5-min B0 / R2* / fat: same prescription; the flow library's velocities with the steady-state blood signal
   (`simphantom --beat-spin off`), since its spin history is the flow protocol's.

Each step is skipped when its output exists and runs in a memory-capped scope without swap. simphantom caches all
library beats (≈ 1.3 GB each): about 700 heartbeats drawn from 15 beats would otherwise recompute ≈ 27 s per miss.
