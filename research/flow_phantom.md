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

## Tier B4: flow-phantom defects found in the 10-min ph0001 simulation (2026-11-25)

Found from velocity-weighted MIPs of the truth (research/experiments/flow_figs/README.md), then traced in the LBM run
`lbm/run_lib_v2` (beat library, per-cut log) and the case `lbm/case_tree`.

| symptom (1 mm truth) | measured in the LBM | cause |
|---|---|---|
| pulmonary arteries p99 3.1 m/s, max 6.2 m/s (normal peak 0.6–1.0) | 3.3–6.3 m/s at the right hilum in systole; 4.9–8.5 m/s at the right-lung pressure outlet when the valve is closed | (1) outlet BC, (2) lumen pinch |
| pulmonary veins 2–3 m/s all cycle at the right hilum | 2.3–3.2 m/s in the right superior PV at (−27, −2, 1320) mm during the S and D waves | (2) lumen pinch |
| diastolic hot spots at the box edge (x = −71 mm) | pressure outlet "Lateral segmental artery of right lung #3888" (r 3.4 mm) carries −118…+54 mL/s | (1) |
| thoracic aorta retrograde −15…−29 mL/s for ~250 ms | abdominal-aorta pressure outlet feeds the prescribed head/arm outflow backwards in diastole | (1) |
| RPA peaks 50 ms after the aorta | network roots: aortic 110 ms, pulmonary 160 ms = physiological (PA acceleration time ~130 ms vs aortic ~90 ms); the further delay at the RPA plane comes from (1) | not a defect by itself |

**(1) Rigid domain + one pressure outlet per arterial tree.** Every cut but the largest gets the 1D network's outflow
waveform. The network has compliance, so its distal waveforms are delayed and damped relative to the valve flow. The
LBM domain is rigid (apart from the cardiac motion field's volume change of the tree, ±6 mL for the PA tree), so the
whole storage term Q_valve − Σ Q_cut passes through the single pressure outlet. For the pulmonary tree: RV ejection
peaks at 425 mL/s, the prescribed cuts at 342 mL/s 60 ms later; the imbalance reaches +186 / −131 mL/s, through
0.36 cm². On the systemic side the head/arm cuts keep their diastolic forward flow with the aortic valve closed, so
the abdominal aorta flows backwards.

**(2) Interpenetrating atlas tubes.**
- **The pinch at the right hilum:** the right PA / middle lobar artery (R 7.1–7.4 mm) and the right superior PV / apical vein
  (R 5.3 / 3.4 mm) have centrelines 4.2–6.9 mm apart, i.e. 4.6–6.5 mm of overlap.
- **Why de-overlap missed it:** `deoverlap.py` shrinks radii by at most 15 % (floor 0.85) and exempts pairs whose
  axis lies inside the other's lumen (meant for branches the atlas draws along their parent).
- **What the case builder does with it:** it separates disallowed regions by walls (interface growth + 1.5 mm proximity
  walls), carving both lumens down to slivers: wall distance at the centreline 1.0 mm for R 7.4 mm (middle lobar
  artery) and 0.5 mm for R 5.3 mm (right superior PV).
- **Whole graph:** 455 edge pairs of different components overlap, 41 k sample pairs deeply. Examples: iliac artery/vein
  7.5 mm, superior mesenteric artery/vein 6.6 mm, internal carotid/jugular 4.4 mm.
- **Fragment component:** component 109 (apical / anterior / posterior segmental arteries of the left lung, 13 edges)
  is separate from the main pulmonary tree but starts on the left PA's axis (distance 0): a missing join.

Great-vessel diameters are realistic (pulmonary trunk 24.5, RPA 14.2, LPA 19.0, ascending aorta 24.8, PVs 9–13 mm),
so the speeds do not come from undersized vessels.

**Fix plan (generic: mm-based, region-agnostic, diffeomorphism-equivariant):**
1. Graph: separate interpenetrating vessels of different components by moving centrelines, not by shrinking. Pairs
   closer than r_i + r_j + walls are pushed apart radially from the larger vessel's axis; the smaller vessel moves
   more; displacements are smoothed along each edge (≈ 2 r) and kept consistent at graph nodes; mobility is reduced
   near the heart roots. The existing radius shrink stays for the residue. Gate: report remaining inter-component
   overlap > 0.5 mm. Same-system fragments touching a tree at distance ≈ 0 are reported as missing joins.
2. LBM case: lumen gate — along every in-domain centreline (away from cut planes and junctions) the wall distance must be
   ≥ 0.7 r; the build fails otherwise.
3. LBM compliance: wall transpiration on each arterial tree. A wall-normal velocity u_n(X, t) = q_w(t) w(X) is added to
   the material velocity of the nodes within 1.5 cells of the tree wall (Ladd's moving-wall term then injects ρ u_n per
   area: exact for a flat wall). Definitions:
   - storage q_w(t) = Q_valve(t) − Σ_cuts Q_net,cut(t) − dV_tree^motion/dt, made zero-mean;
   - Q_valve(t) = −dV_ventricle/dt from the motion field while the outflow valve opens, else 0;
   - the pressure cut's target is its scaled network flow;
   - weight w ∝ local radius, normalised so that ∮ w dA = 1, i.e. thin-wall compliance per wall area ∝ r.
   The pressure outlet then carries only the residual. Test: closed rigid tube with constant u_n, so that the flux
   Q(x) = 2πR u_n x, as in the dilating-tube test.

### Tier B4 status (2026-11-25, end of day)

Implemented (not yet run on the GPU):
- **Graph separation** (`deoverlap.separate`, `vessel_graph.py`; `--no-separate` = old graph, `--out DIR`; regression:
  `--no-separate` reproduces the export graph bit-exactly):
  - lumen-overlap sample pairs between components 72 193 → 1 275, worst 14.5 → 1.45 mm;
  - heart-root ends pinned (0.0 mm); largest moves 7.3 mm (SVC, brachiocephalic veins), 5.9 mm (PA bifurcation);
  - right hilum improved but not cleared: RSPV ↔ middle lobar artery gap −3.3 → +0.2 mm, apical vein ↔ RPA −4.7 →
    −1.2 mm. The vein side has low mobility near its LA root, and the per-sample cap binds.
- **Fragment fallback bridge:** a fragment node inside a larger same-system lumen joins there when the free-end rule fails
  (+5 bridges: left upper-lobe segmental arteries, ascending pharyngeal ×2, circumflex femoral ×2; no existing bridge
  changed).
- **LBM wall transpiration** (hasty-cfd `Transpiration`):
  - tests: CPU `transpiring_closed_tube_flux_grows_linearly` (Q(x) = −2πR u0 (x − x_s) within 2–8 %, past the band
    2 %); `gpu_matches_cpu` with transpiration (relative RMS 4.4e-4).
  - GPU bug found and fixed in the process: `B == 0` excluded nodes where fast math leaves the valve fraction at
    2e-6; the guard is now `B < 1e-3`.
- **Case builder** (`lbm_case.py`):
  - `--sep-split half` (mid-surface proximity walls; each side gives up (SEP − gap)/2);
  - `--transpiration on` (storage waveforms per arterial tree, band weight ∝ r; the pressure outlet's q_ref = its
    network target);
  - lumen gate `--lumen-min 0.6`.
- **Test phantom** `data/hasty/phantom/factory_test/ph0001` (seed 1 on the separated graph via `make_phantom --graph`):
  - transpiration, aorta: wall 205 cm², q_w −118…+196 mL/s, stored swing 17 mL, wall speed ≤ 1.4 cm/s;
  - transpiration, pulmonary: wall 277 cm², q_w −215…+150 mL/s, swing 25 mL, wall speed ≤ 1.8 cm/s;
  - motion volumes: aortic tree 110–117 mL, PA tree 99–110 mL.

**Lumen gate fails on 16 branches** (so the GPU rebuild would still carry pinches):
- right hilum: RSPV 0.27 r, apical vein 0.33 r, middle lobar artery 0.45 r, RPA 0.46 r. Part of it is a planar cut through
  the RSPV trunk at x ≈ −28 mm, apparently the cap / kill zone of the venous inlet "bridge: Anterior vein of right lung",
  which runs along the RSPV / RA interface;
- vessel–chamber overlaps the graph separation does not see: thoracic aorta 0.46 r (LA lumen 6.7 mm from its axis,
  R 9.3 mm), coronary sinus (centreline inside wall, 0.5 mm from the RV);
- PA bifurcation 0.52 r, LPA 0.55 r, left apical segmental artery 0.55 r.

Next:
- chambers as separation obstacles (vessel samples pushed out of the chamber lumens except their own root chamber);
- cut-cap ownership check for inlets that run along another branch;
- then a GPU probe run + `lbm_leaks.py`, then the LBM library, frames and the 10-min scan.

### Tier B5: clearance at every instant (2026-11-25)

User requirement (2026-11-25):
- not a single overlap, vessel–vessel or vessel–heart, at any instant (cardiac motion, respiration, wall distension);
- coronaries and the vessels around the heart move with the heart;
- general rules, no per-vessel cases.

**`clearance.py <phantom>`** (new):
- **Geometry:** frame-by-frame as phantom_4d builds it — centrelines moved by the cardiac and respiratory fields, radii
  at their largest pressure distension, pack_vessels walls. 24 cardiac phases × r ∈ {0, 0.5, 1}.
- **Vessel–vessel:** different components, or one component more than 2(r_i + r_j) + 2 mm apart along the graph; gap
  = walls, ≥ 1.5 mm where both r ≥ 2.5 mm.
- **Vessel–heart:** chamber blood and valves are off-limits except in a tree's own connection zone. The myocardium is
  off-limits except connection zones and the heart's own circulation (side subtrees leaving a trunk inside a
  connection zone, from the shortest-path tree; graph topology, not names).
- **Solve:** pushes, then a capped radius shrink. The audit uses every sample and every frame.

Audit of the test phantom (factory_test/ph0001, separated graph):

| | pairs | worst | vessel–heart | worst |
|---|---|---|---|---|
| reference pose | 216 869 (158 814 within one tree, 58 055 between trees; 1 517 edge pairs) | 13.0 mm | 1 629 (LA 588, RA 416, RV 230, LV 171, myocardium 209) | 12.4 mm |
| all 72 frames | 270 094 | 24.2 mm | 2 688 | 25.8 mm |

Three causes, all general:
1. **Graph geometry within a tree.** Branches the atlas draws running inside or alongside their parent well past the
   junction: right PA ↔ superior lobar artery 11.7 mm, left PA ↔ its segmental arteries ~10 mm, external ↔ internal
   iliac 9.6 mm. Further:
   - bridges end at the target edge's end node and run inside the parent (ascending aorta ↔ bridge: right coronary
     artery 9.9 mm);
   - a confluence the graph misses (hepatic veins ↔ IVC 13 mm, joined elsewhere);
   - self-overlaps (superior mesenteric vein, right inferior PV).
2. **Atlas overlaps between trees and with the heart.** Ascending aorta into the LA / RA blood ≈ 12 mm, iliac
   artery / vein, mesenteric artery / vein.
3. **The cardiac motion field spills into the great vessels with unphysical strain.** Smallest stretch over the cycle at
   the vessel samples (`strain.py` in the session scratchpad):

   | vessel | smallest stretch | displacement |
   |---|---|---|
   | IVC | 0.44 | 13.1 mm |
   | ascending aorta | 0.52 | 16.4 mm |
   | PA bifurcation | 0.55 | 17.6 mm |
   | thoracic aorta | 0.57 | 13.5 mm (behind the heart) |
   | SVC, RSPV | 0.58–0.59 | 6–8 mm |
   | coronaries | 0.59–0.64 | 16–18 mm |

   Gaps close by up to 2× under motion; this is what the LBM leaks under the moving lattice were.

Plan:
- **A. Motion model outside the myocardium.** Keep the heart's own motion (chambers, myocardium, the epicardial
  surface that carries the coronaries). Extend it into the surroundings as a smooth, near-rigid / bounded-strain field
  that decays with distance. Great-vessel segments are stiff inclusions that translate and rotate with their root.
  Gates: det J > 0 everywhere, outside-heart stretch ≥ ~0.85.
- **B. Graph geometry:**
  - co-running branch → junction moved to where the branch leaves the parent's lumen (the inside part dropped);
  - bridges end at the nearest point of the target (edge split there);
  - a same-system end inside another lumen joins it there;
  - self-overlaps flagged.
- **C. Clearance solve** (as above) for what remains.
- **D. End-to-end gate** in phantom_4d: vessel voxels lost to chambers and phantomflow contact splits per frame = 0.

**Time in the clearance check (user, 2026-11-25: "the truth must be much smoother than 20 cardiac × 8–12 resp
bins").** The truth is continuous: the cardiac field is a periodic Catmull-Rom spline of its 48 samples, r(t) a
continuous breath model with depths clipped to [0.5, 1.4], and simphantom moves the sources every 10 ms; LBM flow is
linear between 50 phase bins. The first clearance check sampled 24 phases × r ∈ {0, 0.5, 1}, which was too coarse and
stopped at r = 1. Now:
- **breathing:** exact over r ∈ [0, 1.4]; at a fixed cardiac phase a point moves linearly in r, so the closest
  approach of two points has a closed form;
- **cardiac phase:** 96 phases (2 per motion sample), minus half the largest relative displacement between neighbouring
  phases, which covers the instants between them;
- **radii:** the largest over the cycle.

Implemented since:
- `vessel_graph.py` junction relocation, generalised: a branch's chain runs through degree-2 nodes; any vessel of the
  same tree at least as large counts as a parent; chains lying entirely inside a parent and ending free are dropped as
  duplicates; heart connections are never touched;
- `cardiac_motion.py --taper-mm` (default 25) and an outside-heart strain print (stretch, det);
- `make_phantom.py --clearance` (solve → network → pack → audit gate; opt-in until validated);
- `graph_test_chain.sh <tag>`: test export, test phantom and both audits, without touching the export graph.

### Tier B5 progress (2026-11-25, evening)

**Rules (all general):**
- *Allowed overlap:* only a bifurcation's merged lumen (contiguous from the junction cluster along both edges, in
  `clearance.merged_runs`) and a vessel entering its own chamber (contiguous from its root, the "ostium").
- *Chambers yield to great vessels* (user decision): `clearance --carve` relabels another chamber's blood or the
  myocardium around a violating vessel sample as fat, in `<phantom>/atlas/tissue_*.hv`; the motion is refitted to it.
  Test phantom: LA 3.6, RA 7.7, RV 4.5, LV 0.1, myocardium 0.4 mL.
- *Graph*:
  - junction relocation in two passes, "inside" (axis in the parent's lumen) and "alongside" (lumen within
    parent + walls + gap, moved when the exit is > 2 (r_p + r_c) along the parent);
  - self-loops dropped (≥ 80 % inside the other vessels at their node) or split into rings;
  - folds excised (closer than max r, > 1.5 max r apart along an edge);
  - kinks smoothed (bend radius ≥ 1.2 r + 1 mm).
- *Motion:* the AV descent / twist fade harmonically outside the heart (`--outside harmonic`, default). Free-space
  chamber modes. The insulated-vessel variant folds (`--insulate-vessels`, experimental).
- *Radius shrink* (the solver's last resort) is floored at 0.9 of the radius before the first clearance run
  (`radius_ref` in graph.npz). Velocity ∝ 1/r²; the earlier 0.6 floor compounded across runs, leaving the thoracic
  aorta at r 3.1 mm, and was the same kind of distortion as the too-fast pulmonary arteries. What needs more fails the
  gate.

**Speed (one test phantom from the Z-Anatomy export):**

| stage | before | now | what changed |
|---|---|---|---|
| whole chain | ~55 min | 19 min | the items below |
| graph build | 11 min | 3 min | candidate pairs by per-sample reach instead of a 30 mm `query_pairs`, bit-identical; vectorised spread |
| cardiac motion fit | 690 s | 55–91 s | torch float64 on the GPU; identical to the numpy fit (U diff 0, coefficients 5e-15) |
| clearance setup | 120 s | 33 s | 12 EDTs in parallel |
| full audit | 231 s | 39 s | motion, pair distances, stretch in torch, 96 phases batched; closed-form 3×3 eigenvalues; stretch cached |
| clearance iteration | 34 s | 2.3 s | as above, plus plateau stop |

Cardiac and respiratory motion are fitted concurrently.

**Gate on the test phantom** before the alongside pass / fold / shrink-floor fixes:
- 8 171 vessel–vessel violations: 598 between trees; 141 kinks, 732 bifurcation re-contacts, 2 207 / 2 161 / 2 332
  contacts ≤ 50 / 50–120 / > 120 mm apart along the graph;
- 65 vessel–heart violations.

**Open:**
- a missed confluence (hepatic veins cross the IVC axis, joined 89 mm away along the graph);
- the remaining within-tree contacts of small vessels.

### Tier B5 progress (2026-11-25, night)

Test chain `rustlib/tools/phantom/graph_test_chain.sh <tag>`:
1. the graph from the Z-Anatomy export;
2. the network flow;
3. the test phantom (seed 1) with clearance, carve and motion refit;
4. the final audit (every sample, every one of the 96 cardiac phases, breathing r ∈ [0, 1.4]).

**Graph (vessel_graph.py)**
- *Inside-chain collapse:* a chain lying entirely inside its parent between two junctions is removed. Its far junction moves onto the parent at the nearest point, provided the tree stays connected. `--no-collapse` turns it off.
- *Crossing joins:* two edges of one tree whose axes cross (closer than half the smaller radius) while more than max(2 (r_i + r_j) + 2 mm, 20 mm) apart along the graph are joined there.
  - Both edges are split at the crossing.
  - Nodes closer than 0.5 mm are merged instead of joined by an edge; a zero-length edge broke the node re-join.
  - 59 joins on the test graph.
  - Some look like drawing crossings rather than confluences: internal ↔ external iliac at 2.1 mm, the right coronary artery ↔ unnamed branches. `--no-crossing-join` turns it off.
- *Bridges join where they enter the target lumen.* The old rule (`--bridge-end-node`) ran the bridge to the target edge's nearer end node. The 23 posterior intercostal bridges then ran inside the thoracic aorta (up to 38 mm) to nodes 114 mm away along the graph. That was 31 % of all remaining overlaps.
  - Now the target edge is split at the first lumen ≥ 1.5× the fragment's own that the straight bridge enters.
  - Targets are ranked by distance to the lumen wall, not the axis.
  - A ray along the free end's own tangent is tried first.
- The builder now asserts that every edge's polyline runs from node a to node b.

**Clearance (clearance.py)**
- *Merged lumen across nodes* (`--merged-runs graph`, default). A bifurcation's allowed overlap is followed past the branches' next nodes.
  - Two left superior pulmonary vein branches leave one junction side by side. The first edge ends after 11 mm while they still overlap by 5 mm; the old per-edge rule (`edge`) flagged the rest.
  - Same geometry, same audit: 23 644 → 10 841 pair violations.
  - The walk is lazy: only first edges, unless a run covers a whole first edge. 2 s per evaluation, recomputed every 4th solver iteration.
- *Exact tissue check* (`--tissue-check exact`, default). The heart tissues' boundary voxels (563 k at 0.5 mm) move with the same cardiac + respiratory motion as the vessel samples, and every (sample, voxel) distance is minimised over the frames and r.
  - The linearised check (`linear`: reference distance × the smallest stretch |F n| at the vessel's axis) was wrong both ways.
  - Test phantom: it carved 47 mL of chamber, of which the exact motion needs 6.4 mL (ascending aorta: RA 8.3 mL carved / 2.0 needed, LV 3.2 / 0.0). On the same geometry it missed 695 contacts (104 vs 799).
  - Candidates are boundary voxels within 3 (need + 1 mm) in the reference pose. Reach 2, 3 and 4 find the same 9 576 violating pairs. The gate fails if any violating pair lies beyond 0.75 of the reach.
- *Exact carve:* the violating boundary voxels become fat, then their tissue neighbours are tested in turn, one 0.5 mm layer per pass. The 1 mm and 2 mm grids follow by nearest neighbour.
- *Carve loop* in make_phantom: carve → refit the motion → solve, repeated until no tissue contact remains (`--carve-rounds`, 3). One pass was not enough, because the refit moves the heart differently.
- *Heart's own circulation* (`--cardiac-rule epicardial`, default):
  - side branches leaving a root's trunk within 2 r_root + 10 mm of the root;
  - kept only if ≥ 50 % of their subtree lies within lumen + 5 mm of the myocardium.
  - In the atlas the right coronary artery joins the aorta 31 mm from the root node, outside the R_root + 10 mm zone. It was not exempt, and was 512 of the 799 exact tissue violations.
  - The wider zone alone flagged the lower-body veins through the IVC's trunk choice. The epicardial test leaves exactly the coronary arteries and cardiac veins, plus the phrenic arteries, which the atlas bridges onto the RCA.
  - The old zone rule had flagged pulmonary vein branches.

**Gate on the test phantom (full audit)**

| run | pairs (worst) | tissue (worst) | carved chamber (mL) |
|---|---|---|---|
| full5: before today's fixes | 41 497 (17.6 mm) | 259 (5.5 mm) | LA 12.7, RA 17.6, RV 16.6, LV 3.7 |
| full6: along-edge smoothing, directional stretch, collapse, crossing joins | 23 644 (11.7 mm) | 343 (7.8 mm) | LA 6.5, RA 8.7, RV 11.1, LV 0.4 |
| full7: + merged lumen across nodes, carve loop, bridges at the entry | 3 902 (4.3 mm) | 104 linearised / 799 exact (1.8 / 4.4 mm) | LA 6.5, RA 19.4, RV 14.3, LV 4.2 (linearised, 3 rounds) |

| full8: + exact tissue check / carve, epicardial heart circulation, ray bridges | 3 892 (5.4 mm) | 98 exact (1.1 mm) | LA 1.9, RA 7.0, RV 1.4, LV 0.1, myocardium 2.0 (exact, 3 rounds) |

The full6 solve also removed the solver-made folds (bends and kinks 1 621 → 0).

full8 remainder:
- *Vessel pairs:*
  - deficits: 68 % < 0.5 mm, 86 % < 1 mm, 122 pairs > 2 mm;
  - 75 % involve a vessel with r < 1 mm (sub-voxel in the 1 mm truth);
  - worst: a left upper-lobe pulmonary vein touching itself (5.4 mm), small arteries drawn along their parent past the junction (maxillary branches, anterior inferior pancreaticoduodenal ↔ superior mesenteric artery, ~3 mm), the right coronary artery ↔ its unnamed branches (2.9 mm), intrarenal veins.
- *Heart tissue:* 98 samples, all ≤ 1 mm, mostly coronary arteries within 0.5 mm of RV / LA blood.
  - Three carve rounds leave a remainder because each motion refit moves the heart a little.
- *Chain time:* 35 min; the exact tissue check makes each solve slower.
- *Open question* (asked 2026-11-25): gate per consumer instead of per centreline-sample pair — blood voxels merged between vessels or lost to chambers in the 1 mm truth, connected lumens in the LBM domain — with this count kept as a diagnostic.

### Tier B5 progress (2026-11-26, overnight repair loop)

Rules added (all general; the old behaviour kept behind an option):
- **Graph (vessel_graph.py):**
  - *Origins against the parent's wall* (`--attach-axis`: old). A spline end attaches to a vessel at least as large when it lies within max(1.5 mm, 2 r) of that vessel's lumen wall, not its axis. The node sits on the parent's axis (before: halfway between the end and the axis, which kinked the parent).
    - The maxillary artery's pterygoid-canal / buccal / descending-palatine branches start 0.7–1 mm outside its wall and had been bridged to each other and 131 mm away.
    - The ophthalmic fragments (lacrimal, ciliary, supratrochlear arteries) are now attached.
  - *No stage may split a tree.* Components are counted after each stage, and the build fails (`--allow-split` to continue). `--reloc-debug` names the step.
    - The alongside relocation had moved the right inferior basal vein's junction onto the superior basal vein, which drains through it. That cut the right inferior pulmonary vein from both. Same for the pericallosal arteries.
    - A relocation now requires the junction node to reach the new parent without the moved chain. The pericallosal arteries now join the ACA.
- **Clearance (clearance.py):**
  - *Crossing tubes separate along their common normal* (`--push-dir radial`: old). The radial push from the larger axis lies mostly along the smaller vessel's own tangent near a crossing, so it only slid the crossing along. An unnamed RCA branch crossing the RCA's axis at 0.3 mm stayed put.
  - *Cluster core:* the samples of a junction cluster's own short edges belong to its merged lumen. A 7 mm pulmonary vein edge between two close junctions had no side, and every contact on it counted (5.4 mm, the worst remaining pair).
  - *Merged lumens over the motion* (`--merged-static`: old): the contiguity test also uses the smallest distance over the frames.
  - *Solver:*
    - the plateau stop measures the total deficit (`--plateau count`: old). In full9, 1 310 of the 1 343 violating samples ended at the 0.9 shrink floor, two thirds of them still below their displacement cap;
    - pushes split by the displacement left under each sample's cap (`--cap-blind`: old).
- `flow_check.py <phantom>`: named-vessel mean flows against adult resting ranges (scaled with the cardiac output), per-lung arterial in / venous out, and the peak cross-sectional velocity per class.

| run | pairs (worst) | tissue (worst) | carved chamber (mL) |
|---|---|---|---|
| full8 | 3 892 (5.4 mm) | 98 (1.1 mm) | 12.4 |
| full8 geometry, core + dynamic merged lumens (audit only) | 3 178 (3.2 mm) | 98 | — |
| full9: + common-normal push, wall attach | 2 397 (3.7 mm) | 62 (1.1 mm) | 12.4 (LA 1.7, RA 7.6, RV 1.3, LV 0.1, myocardium 1.7) |

**Flow check of the network (full8 graph)**

Thorax and abdomen are all in range:
- cardiac output 87.5 mL/s;
- SVC 23, IVC 60, portal vein 17.5 mL/s;
- coronary arteries 3.6 mL/s (4.1 %);
- carotids 5.8 / 6.1, renal arteries 8.3, coeliac 9.6, superior mesenteric artery 9.2 mL/s;
- lungs: right share 0.55 venous; 0.60 arterial counting only the named LPA, 0.55 with the bridged upper-lobe fragment;
- peak cross-sectional velocity: systemic arteries 1.38 m/s (abdominal aorta), veins 0.38 m/s.

Out of range:
- *head arteries:*
  - ACA 0.2 mL/s (0.8–1.9): its continuation, the pericallosal artery, was a disconnected fragment, now attached;
  - basilar 0.8 mL/s (1.8–3.4), right vertebral 0.9 mL/s;
- *dural sinuses:* superior sagittal 0.2, transverse 0.1, straight 0.1 mL/s (3.5–7.5 / 2.5–6 / 0.7–2.2). The atlas has no cortical veins, and terminal flows follow the drawn end radii (Murray r³ inside each first-order territory), so the sinuses drain almost nothing. The petrosal sinuses take it instead;
- one middle-lobe pulmonary artery at 1.33 m/s cross-sectional mean (limit 1.2).

For head MRI the terminal flows should come from the tissue each terminal supplies or drains (brain labels, regional perfusion), not from the drawn end radii. The torso label grid has no brain. Open.

More rules (2026-11-26, overnight):
- *Push spreading along the vessels* (`--spread spatial`: old). A push reaches only its own edge and the edges sharing a node with it. The spatial Gaussian also reached a crossing branch of the same tree, and the two opposite pushes cancelled: the RCA and an unnamed branch, 0.33 mm apart, had both moved past their caps, together. This was the largest single gain (full10 → full11).
- *Coronary sinus as its own root.* The coronary sinus tree is classed systemic venous (it was "coronary", which the network ignores unless bridged) and gets a heart connector to the RA. A root vessel's heart end is never a bridge source (`--bridge-heart-ends`: old). It had been bridged 22 mm through the RA to the IVC root, and the LBM builder's walls pinched that bridge.
- *Hooks* (relocation pass "hook", `--no-hook`). A branch that comes back across the vessel it left within 3 (R + r) + 5 mm of its junction leaves it where it last crosses; the hook is dropped.
  - The parent must pass through the junction: its name on two edges there, not the branch's own. A first version moved the internal iliac onto the external iliac, the ECA onto the ICA, and the PCA onto its own medial occipital branch.
  - On the atlas there are 27 hooks: both inferior epigastric arteries, intercostals, genicular, superficial epigastric and lumbar veins.
- *Floating-junction joins* (`--float-join`, opt-in; rejected as default): a junction near a same-tree vessel joined to it. It also joined the facial artery to the ICA and a lumbar artery to the testicular artery; vessels lie side by side.
- *LBM case (lbm_case.py):* a cut never sits on a junction. A branch ending at a junction node ends 1.5 r + 1 mm before it, if its cap then stays clear of the parent's lumen; otherwise a side branch is left out (`--cut-at-node`: old). The left superior pulmonary vein's inlet disk lay across its lingular / anterior crotch: 30 % fluid, 1 % of its flow delivered.
- `lbm_burst.sh <phantom> <out> [dx_mm] [cycles]`: the case at 1 mm, a few cycles, the report. `lbm_report.py` now sizes its figure to the cut count.

| run | pairs (worst) | tissue (worst) | within-tree contacts > 50 mm along the graph |
|---|---|---|---|
| full10: + deficit plateau, cap-aware split, relocation reachability guard | 2 221 (3.6 mm) | 44 (0.54 mm) | 1 208 |
| full11: + along-vessel spread, coronary-sinus root | **489** (3.7 mm) | 67 (0.60 mm) | **2** |

full11 remainder:
- 283 bifurcation re-contacts;
- 73 contacts ≤ 50 mm along the graph;
- 20 bends;
- 111 artery–vein pairs.

About 200 of the remainder are the left inferior epigastric artery's hook at the external iliac junction, which the hook pass handles in full12.

**Diffeomorphism of a cleared geometry** (habitus, det J 0.75–1.27; full10 graph and carved atlas, `make_phantom --base … --diffeo`, no re-solve):
- 4 022 pairs / 96 tissue contacts, against the base's 2 221 / 44;
- artery–vein pairs 110 → 1 234.

Clearance keeps exactly the required gap, so a map that compresses by up to 25 % closes some of it, and the radii follow the mean transverse stretch only. The production path therefore runs clearance after the diffeomorphism (`make_phantom --graph … --diffeo … --clearance`); its gate is the one that counts.
- Production path on the habitus variant of the full11 graph: **373 pairs (3.65 mm) / 31 tissue contacts (0.62 mm)**, carve 8.1 mL in total, against 489 / 67 for the undeformed phantom.
- So the variants clear as well as the base, with the same kind of remainder.

Further LBM-case work:
- *orphan cuts:* a branch whose onward edges were all left out ends in a cut (`--no-orphan-cuts`: old). The coronary sinus continues into the 2.2 mm great cardiac vein, and its tree was a sealed pocket; the pulmonary trees gained 7 cuts.
- *lumen-gate failures on the full11 phantom (not caused by the cut rules):*
  - the coronary sinus at 0.39 r: its own lumen voxels, labelled myocardium, end up solid 0.9–1.3 mm from its centreline. Probably the stub claims voxels only beyond its root plane, and the curved sinus comes back behind it;
  - the middle lobar artery at 0.53 r: solid lung voxels 4 mm from its axis, next to its medial segmental branch. Open.

**LBM short burst** (full10 phantom, 1 mm lattice, 4 cycles in 115 s):
- *in agreement:*
  - carotid / subclavian / segmental cut fluxes 0.98–1.03 of their targets (corr ≥ 0.97);
  - venous cuts 0.92 (the known 4–6 % stroke deficit);
  - pulmonary jet 1.9 m/s;
  - mitral / tricuspid 1.6 m/s.
- *off:*
  - aortic jet p95 2.9 / max 3.5 m/s (0.5 mm reference: 2.0 / 2.6); probably the 1 mm lattice narrowing the leaflet opening, to be checked at 0.5 mm;
  - mitral closed-valve leak 9 mL/s mean;
  - the left superior pulmonary vein inlet blocked (fixed: cut at a junction).
- *open lumen-gate failures on the full11 phantom:*
  - the coronary sinus at 0.39 r, where it meets RA and RV regions near the tricuspid annulus: probably the builder's AV wall / annulus collar;
  - the middle lobar artery at 0.53 r inside its own tree.

| run | pairs (worst) | tissue (worst) | edge pairs |
|---|---|---|---|
| full12: + hook pass | 262 (1.92 mm) | 52 (0.67 mm) | 39 |
| full13: + near-junction exemption with the cycle-maximum radii (`--near-rest`: old) | 267 (1.75 mm) | 39 (0.66 mm) | 35 |

full13 remainder:
- 202 of the 267 pairs are < 0.5 mm deep, none > 1.75 mm;
- the deeper ones are siblings touching again beyond their junction as the lumens distend: the left inferior pulmonary vein's two branches, the left and right pulmonary arteries at the bifurcation, the left coronary artery on the distended aortic root, the right internal jugular / subclavian veins at the venous angle.

**LBM at 0.5 mm** (full12 phantom, 4 cycles, 980 s; 55 cuts):
- *jets:* aortic p95 2.44 / max 2.73 m/s, close to the reference 2.0 / 2.6, so the 1 mm values (2.9 / 3.5) were the lattice; pulmonary 1.55, mitral 1.26, tricuspid 1.18 m/s;
- *the right lateral segmental cut* (the pulmonary tree's pressure outlet) takes 0.24 of its target. It absorbs the pulmonary stroke deficit: 76.3 vs 83.8 mL, 9 %, larger than the 4–6 % noted before;
- *mitral leak* 11 mL/s mean (17 rms) while closed, at 0.5 mm too: real (reference ≤ 3 mL/s rms). Open;
- *hot spots* 3.1–3.3 m/s: at the left subclavian inlet (its slab 76 % fluid) and in the left upper-lobe segmental arteries in late systole. Open.

**Network flow (full13 phantom):** regressions against the full8 graph.
- *posterior head circulation:* vertebral arteries 0.17 / 0.09 mL/s (were 1.16 / 0.86), basilar 0.40 mL/s, its top reversed (fed from the ICAs through the posterior communicating arteries). The atlas posterior communicating arteries (r 1.85 mm) are thicker than the vertebrals (1.2–1.5 mm), against roughly 0.6 vs 1.5–2 mm in reality. With r³ terminal shares and no tissue territories, the network takes the easy path. Head-phase item, together with the sinuses.
- *right pulmonary artery:* 1.90 m/s cross-sectional peak. It tapers to r 6.1 mm, the Murray floor 0.75 · c · Q^{1/3} after the 0.9 clearance shrink, while carrying the whole right-lung flow. The floor factor 0.75 allows up to 1.8× the Murray-consistent velocity. Decision pending.

**Aneurysm variant** (`--diffeo aneurysm`: habitus + a 1.45× bulge of the ascending aorta, det J 0.80–2.77; full13 graph, production path):
- 332 pairs (3.44 mm) / 88 tissue contacts (2.26 mm), against 267 / 39 for the undeformed phantom.
- The carve rounds are not monotone. After the second carve and motion refit the audit rose (543 → 1 106 pairs, 210 → 354 tissue contacts) before the third round brought it back.
- The strong local deformation next to the heart needs more than three carve / refit / solve rounds. Open (adaptive rounds, or carving with a margin for the refit).

**Within-tree gap** (`--tree-gap`: old). Lumens of one tree need only their two walls between them. The 1.5 mm applied only because the LBM builder separates and carves *different* trees' regions; branches of one tree share a region.
- The merged-lumen (bifurcation) test keeps the generous 1.5 mm. With the relaxed gap there too, the allowed runs shrank faster than the test relaxed: 267 → 1 647 pairs on the full13 geometry.
- With it kept: 267 → 207 pairs (worst 1.64 mm), tissue unchanged.

### Mitral closed-valve leak and valve disease (2026-11-26, plan)

**Observation.** The full12 phantom (clearance + carve) leaks 11 mL/s mean (17 rms) through the closed mitral valve, at 1 mm and at 0.5 mm. The earlier ph0001 (v22, no clearance) leaked ≤ 2.8 mL/s rms. Tricuspid 1 mL/s, semilunar valves ≤ 0.3. So the regression is in this phantom's mitral geometry, not the lattice.

**How the closed AV valve seals** (lbm_case.py + hasty-cfd valves):
- the valve plane is fitted to the LA–LV label interface (radius from the interface area × 1.1, capped at 16 mm), centred on the blood passage;
- a cylinder of radius R + 2 mm (± 2.5 mm) is carved into blood;
- elsewhere an AV wall goes wherever the atrium and ventricle regions touch, plus a solid collar around the annulus;
- closed (opening < 0.05) means a full solid disk ≥ 2 mm in the tissue frame plus a rigid disk (R + 3 mm) on the plane fitted to the moved annulus;
- leaks come from fluid links LA ↔ LV that bypass the disk: around its rim where the passage is wider or tilted against the fitted plane, or through gaps in the AV wall / collar.

**Suspect.** The clearance carve relabels chamber blood near vessels as fat. In full12 it carved 1.6–1.9 mL of LA, much of it along the left AV groove (left coronary / circumflex artery, great cardiac vein). That changes the LA–LV interface the plane and radius are fitted to, and the motion refit changes the annulus motion. The disk + rim may then no longer cover the passage at some phases.

**Plan:**
1. Locate the leak: `cardiac_lbm --probe … --snap` at isovolumic contraction / mid-systole, then `lbm_leaks.py` (D3Q19 links LA ↔ LV not through the disk) on the full12 case.
2. Compare the fitted plane / radius / centre on the uncarved atlas labels with the carved ones; check the rim coverage against the passage over the cycle (moved annulus).
3. Fix generally:
   - fit the AV valve geometry on the uncarved chamber labels (the carve is about vessels, not the valve);
   - or size the closed seal from the passage over all phases (rim ≥ the largest passage radius + 1 mm);
   - and add a gate: closed-valve rms ≤ 3 mL/s per valve in the LBM report.

**Valve disease (later).** The valve model is kinematic, which makes pathologies parameters of the case's `valves`, plus a matching cardiac model:
- *stenosis:* a smaller maximum opening (`theta_open`) or orifice (`hole`), and a slower opening ramp;
- *regurgitation:* a closed state that leaves a coaptation gap (closed opening > 0, or a hole of a given area), with the regurgitant volume in the cardiac model (Q_MV negative in systole, the LV stroke raised so the forward stroke is kept);
- *prolapse / flail leaflet:* one leaflet's strips swinging past the annulus plane in systole (per-strip angle schedule).

So the leak fix comes first: a closed valve must seal to the gate before a regurgitant one can be prescribed on purpose.

| run | pairs (worst) | tissue (worst) |
|---|---|---|
| full14: + within-tree gap of the walls only | **210 (1.59 mm)** | 54 (0.61 mm) |

full14 remainder:
- 141 bifurcation re-contacts, 33 contacts ≤ 50 mm along the graph, 2 at 50–120 mm, 34 artery–vein pairs;
- no bends, nothing farther apart along the graph.

### Valves, LBM case fixes, valve disease and variants (2026-11-26, day)

**The mitral "leak" was the measurement, not the valve.** The report integrated the flux over a rigid disk of radius R
that moves with the valve centre (its velocity subtracted, its axes from the Jacobian there). A closed tissue-frame
valve deforms with the motion field around it (the cavity modes squeeze and shear the AV plane), so the closed sheet
sweeps volume through that rigid disk. Measured instead through the *material* disk, the reference disk carried by the
tissue, relative to the tissue velocity — the cardiac model's own definition of $Q_{MV}$ (the flux into the chamber as
a material region):

$$Q_\mathrm{mat}(t) = \int_{\varphi_t(D_0)} (\mathbf u - \dot\varphi_t)\cdot \mathbf n \,\mathrm dA ,$$

with $\mathrm d\mathbf A = \partial_\rho\varphi_t \times \partial_\theta\varphi_t\,\mathrm d\rho\,\mathrm d\theta$ from the forward map of a polar grid on $D_0$.

| valve (full14 phantom, 1 mm, 4 cycles) | rigid disk: stroke / closed mean, rms | material disk: stroke / closed mean, rms | model stroke |
|---|---|---|---|
| aortic | 80.7 mL / 0.01, 0.06 mL/s | 84.2 mL / 0.01, 0.06 mL/s | 83.8 mL |
| pulmonary | 78.8 / 0.10, 0.65 | 84.6 / −0.00, 0.06 | 83.8 |
| mitral | 77.9 / **−11.0, 17.6** | 83.9 / −0.15, **0.20** | 83.8 |
| tricuspid | 80.0 / −0.62, 0.70 | 82.7 / −0.04, 0.07 | 83.8 |

- The closed valves seal (0.06–0.2 mL/s rms, against the plan's gate of 3 mL/s). The plan's suspect (the carve changing the fitted AV plane) is not needed. *Corrected later the same day:* the material disk's strokes (≈ 84 mL) over-read by 6–10 %; the old "4–9 % stroke deficit" is real — the chambers deliver ≈ 0.9 of their stroke (exact link fluxes, "Where the stroke goes" below).
- The ascending aorta's flux at three stations (binned field, material disks) is 97–101 mL/s, the LV region's volume swing in the lattice ≥ 81.9 mL per beat (both over-read or not what passes the valve; see below).
- `lbm_report.py` keeps the rigid numbers (`stroke_ml`, `closed_*_q_ml_s`) and adds the material ones (`stroke_mat_ml`, `closed_*_q_mat_ml_s`) and a **valve gate**: closed flux within max(3 mL/s, 15 % of the model's closed-phase peak) of the model (0 for a competent valve, the backflow for a regurgitant one), stroke within 5 %.

**Open: the cut flux layers read low.** Both sides balance at the cuts (left: pulmonary-vein inlets 90.6 mL/s in, aortic-tree cuts 89.7 out; right 90.7 / 90.0) while the valves and the chamber volume swings say 97–99 mL/s.
- The binned field 2.5 mm inside each cut (material disks) reads a median 1.06 × the flux layer; along the arch branches the flux is constant along each branch and drops at its cut (right subclavian 7.1 vs 6.74, left common carotid 8.35 vs 7.79 mL/s).
- In the still Poiseuille pipe (hasty-cfd test, now printing the section flux along the pipe) the velocity-inlet's layer reads only 1.4 % below the interior trend, at R/dx 3 and 6 alike. So in the moving, pulsatile heart something else lowers the layer value by 5–8 %; the same deficit appeared at 0.5 mm, so it is not the lattice resolution.
- A pipe translating across or along its axis (new test `moving_pipe_cut_flux`) and the expanding closed tube reproduce neither: the inlet layer reads the prescribed flux, moving walls deliver their swept volume (3–7 % *more* in the expanding tube, its section having 5 % more nodes than πR²).
- The lattice's own region volumes side with the cut layers: on full14 the LV region swings 81.9 mL per beat (log samples every 30 ms), the aortic-tree cut layers sum to 76.5 mL per beat, the material aortic disk 84.2. With mr 0.2 (below) the material disks sum to 87.4 mL against an LV swing of 79.2. So the binned 2 mm output cells over-read small cross-sections (by ~6 % at R 3–5 mm, coverage weighting near the wall) and somewhat the valves too; the measures disagree by 5–10 % and none is exact.
- *Exact link flux (hasty-cfd, done):* the full diagnostics now also sum, over every D3Q19 link from a fluid node into a cut's equilibrium slab, the populations that cross it in the next streaming ($f_i(\mathbf x)$ out, the slab's $f_{\bar\imath}$ in): the mass the boundary actually injects or removes (`q_link_ml_s` in the log). In the pipes it matches the section flux (still: 3.461 vs 3.465 mL/s; moving across: 1.561 vs 1.555), while the u-based layer reads 2–3.4 % low at a velocity inlet (and +3 % at an outlet moving along its normal). The inlet controller κ now steers the link flux (`run.kappa_flux: "layer"`: old): the moving pipe's inlet then delivers 1.511 for 1.508 prescribed, its mid section 1.507. `lbm_report.py` uses the link flux for the cuts (the layer value kept as `layer_mean_ml_s`). The heart cases are to be rerun with it.

**LBM case builder (lbm_case.py), general rules (old behaviour behind options):**
- *Valve-plane split zone = the passage* (`--split-zone wide`: old). Near each valve the region boundary between its two chambers is the valve plane within the carved cylinder (R + rim + 2 voxels) only; beyond it the labels decide (walls wherever regions touch, so the seal does not depend on it). The old zone (R + 6 mm, ±20 mm) turned the RA voxels at the coronary sinus ostium, 7 mm below the flat tricuspid plane, into RV region, and the proximity walls pinched the sinus to 0.39 r.
- *Cut clearance* (`--cut-overlap`: old). A cut's slab and flux layer (lateral < 1.3 R, −1.5 h … slab + 3 h) and its capsule end must hold no other lumen (all trees; a ball of the other sample's radius reaching > 0.5 mm in). An end branch slides its cut back along its samples until clear; a side branch with no clear position is left out (its flow leaves through the tree's outlets, as with the junction rule); a root or continuation keeps its cut. The medial segmental artery's cut had removed part of the middle lobar artery (0.52 r). On full14: 3 side branches left out, 54 cuts.
- *Geometric slab kill* (`--cut-kill owner`: old). Voxels inside a branch's own capsule are its lumen whatever their nearest sample. A 1.3 mm continuation of the left subclavian artery past a junction had lost the parent-owned voxels of its own lumen in its slab (76 % fluid, 3 m/s jets).
- New diagnostics: `--debug-pt x,y,z[,r]` (what holds the voxels around a point after each build stage), `--debug-cuts`, and the open share of every cut disk (all 54 at 1.00 on full14).
- Lumen gate on full14: 0 pinched branches (was 2).

**Velocity floor for the radii (murray_floor.py, default; `--no-vfloor`: Murray only).** Murray's cube law holds for small vessels; the large ones are not WSS-optimal (flow ∝ r^2…2.5 from the aorta to its main branches) and what bounds them in a healthy adult is the section-mean velocity at peak flow. Each edge's radius is also raised to $\sqrt{\hat Q/(\pi v_\mathrm{cap})}$, $\hat Q$ its peak |Q(t)|, with healthy resting peak section-mean velocities $v_\mathrm{cap}$ = 1.2 (systemic arteries), 0.95 (pulmonary arteries), 0.5 (systemic veins), 0.6 m/s (pulmonary veins). After the clearance solver's 0.9 shrink floor (×1.23 velocity) they stay within flow_check.py's limits. On the full14 graph it raises 7 edges beyond the Murray floor (445 vs 438), the right pulmonary artery's narrowest radius from 6.2 to 8.6 mm (adult RPA r ≈ 7.5–9 mm); the aorta and the venae cavae are unchanged. Lesions (stenoses, aneurysms, diffeomorphic variants) and vessel variants come after or are exempt, so the floors do not undo them.

**Valve disease** (`valve_disease.py`, one spec read by the network, the cardiac model and the LBM case; `make_phantom --valves "mr=0.3"`, `"as=0.9,ar=0.15"`, …; stored in params.json `pathology.valves`). The model is kinematic, like the rest of the pipeline: the LBM is driven by the wall motion and has no physiological pressures, so the lesion is prescribed as flows plus valve geometry.
- *Regurgitation* (EROA, cm²): $Q_\mathrm{reg}(t) = 0.8\,\mathrm{EROA}\sqrt{2\Delta P(t)/\rho}$, with $\Delta P$ from simple chamber / artery pressure waveforms built on the phantom's mean pressures (ventricle − atrium over isovolumic contraction, ejection and isovolumic relaxation for MR / TR; artery − ventricle over diastole for AR / PR). EROA is the peak (PISA) orifice; with its cycle mean taken as 0.8 of it, RVol / EROA ≈ 150 cm (the usual CW VTI), so the grading thresholds line up: MR EROA 0.2 → RVol 30 mL, 0.4 → 61 mL at 70 bpm; AR 0.3 → 53 mL.
  - The forward stroke stays the network's SV; the ventricle ejects SV + RVol. MR / TR: the cardiac model adds the systolic backflow to $Q_{MV}$ / $Q_{TV}$ and the ventricle fills SV + RVol. AR / PR: the network's root flow reverses in diastole.
  - The atrium of an AV regurgitation enlarges instead of emptying further: its maximum rises so that its minimum stays at the healthy share of the atlas volume (mr 0.2 on ph0001: LA 26–79 mL instead of 13–65).
  - With the atlas EDV (112 mL) a ventricle cannot eject SV + RVol: make_phantom caps the forward SV so that the end-systolic volume stays ≥ 0.25 EDV (mr 0.2: forward SV 54 mL instead of 84), and refuses below 30 mL. A realistic chronic regurgitation needs the remodelled heart: the new diffeomorphism feature `dilate` (a label's volume × factor about its centroid) and the preset `lv_overload` (LV and LA × 1.5).
- *Stenosis* (effective orifice area, cm², as graded by continuity / Gorlin): the leaflets open to θ with $\pi R^2(1-\cos\theta)$ = area / 0.75 (the geometric orifice) (the closed strip fan tiles the disk; a strip turned by θ projects cos θ of its area). AV stenosis also slows the E wave: deceleration time = PHT / 0.29 with PHT = 220 ms / area (Hatle). Semilunar stenosis changes no flow; the jet follows from the orifice and the prescribed ejection.
- *LBM:* the closed valve keeps the regurgitant orifice open (hasty-cfd valve record slots 34–37, case JSON `regurg_hole` [x, y, a, b] in valve-local units of R: the mitral one a slit on the coaptation line, aspect 3; the others central, aspect 1.5; anatomic area = EROA / 0.75). The affected side's arterial tree gets no pressure outlet: every cut prescribed, so mass conservation forces the ventricle's excess back through the orifice; the venous inertial inlets keep the pressure reference. The jets (Q / (0.75 A) for a stenosis, peak backflow / EROA for a regurgitation) enter the velocity scale. CPU test `closed_valve_with_a_regurgitant_hole`.
- *Bicuspid aortic valve* (`bav=<offset>`, alone or with `as` / `ar`): two cusps (strip fans as the mitral preset, the coaptation line at the offset in units of R). Fully open it passes about the trileaflet's area (2.98 vs 2.87 cm² at 1 mm); stenotic, its orifice is a slit along the coaptation line (the fish-mouth BAV orifice) and the jet eccentric.
- *Later:* prolapse / flail (one leaflet's strips past the annulus plane: a per-strip angle schedule, free slot 7 of the leaflet record), commissural fusion (per-strip opening scale), the LV dilation chosen from the RVol automatically.

**Vessel variants** (`vessel_variants.py`, `make_phantom --vessel-variant <preset|spec.json>`, stored in params.json `pathology.vessels`): ops on named vessels (fnmatch names, or an edge at a junction of two named vessels): `scale` (hypoplasia), `radius` (a drawn calibre corrected), `remove` (aplasia; refused when a piece would lose its root), `connect` (a straight segment between two junction nodes). Applied to the graph copy before the jitter, network and clearance, so a variant is cleared and its flows follow from the network. Edited edges are flagged (`variant_fixed`) and the Murray / velocity floors leave them alone.
- The atlas circle of Willis has no anterior communicating artery (the A1 / A2 junctions are 10.6 mm apart) and posterior communicating arteries of r 1.7 mm (adult ≈ 0.5–0.75 mm, a fetal-type one ≈ P1). Presets: `cow_atlas` (as drawn), `cow_complete` (ACoA r 0.75 mm, PCoAs r 0.65 mm), `cow_no_pcoa`, `cow_fetal_pcoa_l` (left PCoA 1.1 mm, left P1 × 0.4), `cow_a1_hypo_r` (right A1 × 0.4).
- Stenoses and aneurysms anywhere are the existing diffeomorphism features `narrow` / `bulge` on a named vessel (e.g. `{"kind": "narrow", "vessel": "Internal carotid artery.l", "at": 0.2, "factor": 0.5, "length_mm": 10}`); a diffeomorphism cannot create crossings, and the production path clears again afterwards.
- Limit for the head: the diffeomorphism lives on the atlas label box (the torso; φ is the identity outside it, where the graph continues into the head and limbs), so head diffeomorphisms and intracranial aneurysms / stenoses need head label volumes in the atlas first. Graph-only variants (`vessel_variants.py`) work anywhere.
- Head flows need the terminal flows from the tissue each terminal supplies (perfusion territories with brain labels), not from the drawn end radii; that is the head-phase item and decides whether the CoW variants redistribute flow correctly.

**Arrhythmias (design, not built).** The cardiac model is one periodic beat (heart-rate harmonics); the LBM runs it for N cycles (the beat library) and the scan simulator plays beats. An arrhythmia is a sequence of different beats:
- a beat family from the cardiac model: templates over RR (systole ≈ fixed, QT by Bazett; the diastole shortens or lengthens; the E wave truncated by an early systole), the stroke from the preceding RR (Frank–Starling), and an atrial kick or none (AF: no A wave);
- the LBM runs each template periodically (its own beat library; the transient between beats neglected at first);
- a rhythm model for the scan: sinus with respiratory HRV, AF (irregularly irregular RR, no A), PVCs (premature beat, low stroke, compensatory pause), bigeminy — a sequence of (template, phase) over the scan time, also driving the simulated PPG trigger.

**Valve-disease LBM checks** (same cleared full14 geometry, network and cardiac model rerun with the disease, 1 mm, 4 cycles; `data/hasty/scratch/vd_variant.py`):
- *mr 0.2* (forward SV capped to 54 mL; LV ejects 82.5 mL): the slit (10 × 3.4 mm on the coaptation line) passes 28.9 mL per beat while closed against the model's 27.9 (closed phase) / 30 (whole beat); the LV region's swing in the lattice, 79.2 mL, equals the aortic-tree cut layers (≈ 51.5 mL) plus that. v_max 5 m/s (jet 4.2 m/s), 213 k steps in 206 s. Valve gate: mitral ok; aortic "FAIL" on the material disk (58.5 vs 53.7 mL), the measurement spread above (the material disks sum to 87.4 mL against an LV swing of 79.2).
- *as 1.0* (first try): the LV blew up (density ×2, 23 600 mmHg, 33 m/s). The opening angle came from πR²(1 − cos θ) (37°), which counts the radial slits that open between turned strips; leaflets 3 lattice units thick close them, and the valve opened ≈ 0.24 cm². The angle now comes from the strips' actual projection along the valve axis, thickness and the ½-cell ramp included (`open_area` in lbm_case.py, bisection on θ; `--stenosis-formula`: the old rule). It also shows the healthy valves' resolution dependence: the aortic valve opens 2.87 cm² at 1 mm (leaflets 3 mm thick) and 3.54 cm² at 0.5 mm — the 1 mm lattice's inflated aortic jets (2.9 vs 2.4 m/s).
- `lbm_report.py` adds a *density gate*: per region the late-cycle |ρ − 1| ≤ 0.05 (a chamber the motion squeezes with no way out compresses the weakly compressible lattice).

**Clearance: root mobility** (`--root-mob all`: old). The displacement ramp (0 within 5 mm of a heart root, full at 20 mm) anchors the great vessels where they meet the heart; it also froze every thin branch near a root. The left coronary artery could not leave the distending aortic root (1.59 mm, the worst pair of full14). Samples thinner than half the root's radius are now anchored only at their ostium (free 2–6 mm from the root).

| run | pairs (worst) | tissue (worst) |
|---|---|---|
| full15: + velocity floor, root mobility (thin branches free beyond their ostium) | 210 (1.66 mm) | 49 (0.62 mm) |
| full16: + `--plateau worst`, adaptive carve rounds | 183 (1.77 mm, the pulmonary bifurcation's own branches) | 12 (0.31 mm) |
| full17: + solver sampling ≤ 2 mm (`--sub-max-mm 2`) | **162 (0.50 mm)** | 25 (0.48 mm) |
| full18: + SV = EF × atlas EDV (seed 1: 77.3 mL, EF 0.69), E share 0.65–0.80, physio E / A filling | 137 (0.86 mm, the LCA on the ascending-aorta connector) | 33 (0.63 mm) |

full15:
- flow check: every class within its peak-velocity limit now (pulmonary arteries 1.18 m/s at the right pulmonary artery, was 1.90; systemic arteries 1.44, veins 0.55, pulmonary veins 0.64); out of range only the head (vertebral, basilar, ACA, dural sinuses: the head-phase perfusion item);
- the left coronary artery against the aortic root (1.66 mm) did not clear with the free mobility. Its junction node sits on the aortic axis (the wall-attach rule), its lumen runs 17 mm inside the root and leaves the wall at ≈ 35°; the contact is where the merged lumen ends under the root's distension. Being looked at with the audit's pair dump.

**Campaign (full15 phantom, link-flux κ, numeric stenosis angle; 1 mm, 4 cycles):**

| case | valves (material disk): closed rms / stroke vs model | net strokes from the link-flux cut balances | lattice |
|---|---|---|---|
| healthy | ≤ 0.23 mL/s; 82.9–84.3 / 83.8 mL | 74.9–78.7 mL (−6 … −11 %) | density ok, 85 k steps 78 s |
| mr 0.2 | mitral regurgitant 29.8 / 27.9 mL; aortic 57.6 / 53.7 | aortic tree (all prescribed) 53.0 / 53.7 | v_max 5, 213 k steps 193 s |
| as 1.0 | leaflets 63°, jet 5.0 m/s; aortic 84.8 / 83.7 | 77.7–78.3 | v_max 6, 256 k steps 228 s |

- GPU and CPU backends agree on the new diagnostics (`gpu_matches_cpu`: link fluxes identical to 7 digits).
- Gate fix: a competent valve is gated against zero flux while closed. The model's own closing backflow (a 50 ms lobe after ejection, ≤ 26 mL/s, 0.9 mL) falls in the scheduled closed phase; comparing against it had failed every aortic valve at 6.4 mL/s rms.
- *The arterial trees lose a fixed share of their throughput.* The material valve fluxes agree with the model and with the LV region's volume swing. The cut balances are now exact (link flux), and they are 6–11 % lower. The MR case separates the two: its aortic tree has every outlet prescribed, and the cuts deliver 53.0 mL (the target) while the material aortic disk passes 57.6 mL, a ratio of 0.92 (healthy: 75.5 / 83.6 = 0.90). So mass disappears inside the trees. Suspect: the wall transpiration (±300 mL/s compliance storage, zero mean as prescribed, but realised through Ladd's moving-wall term on curved stair-cased walls, where its efficiency may depend on the phase). Test queued: the same burst with `--transpiration off`. The venous side shows a smaller gap (pulmonary-vein inflow 78.7 vs mitral 83.9): next, the exact link flux through the valve planes too.

**Aneurysm variant with adaptive carve rounds** (`--carve-rounds auto`, default: up to 6 carve / refit / solve rounds until no tissue violation, stop after two without a better worst depth; an integer as before). full15 graph, production path, 3381 s:
- tissue worst per round 3.35 → 3.54 → 1.97 → 1.21 → 0.90 mm, final audit **29 tissue contacts (0.43 mm)**, against 88 (2.26 mm) with the fixed 3 rounds;
- **230 pairs (3.59 mm)**: the worst are the pulmonary bifurcation's own branches (3.59, and 2.28 mm against the left pulmonary artery), squeezed between the bulging ascending aorta and the LA, then the left coronary artery on the aortic root (1.77 mm). Open.

**Where the stroke goes (2026-11-26, night).** The exact diagnostics were corrected first: a link the step bounces back (a wall or a leaflet between two fluid nodes) carries no mass and is now excluded; before, the pressure difference across a closed leaflet showed up as a spurious backflow. Healthy full15 phantom, 1 mm:

| | transpiration on | off |
|---|---|---|
| aortic tree: valve-plane link flux in / cuts out | 86.9 / 88.5 mL/s (−0.7 %) | 87.2 / 86.8 (+0.7 %) |
| pulmonary tree | 85.8 / 87.9 (−2.3 %) | 86.8 / 88.8 (−2.8 %) |
| net strokes from the cut balances (model 83.8 mL) | 74.1–78.7 | 74.1–78.9 |

- *The trees conserve mass*; the wall transpiration is no sink (the earlier suspicion was wrong).
- *The chambers deliver ≈ 0.9 of their stroke* to the trees, although the LV region's volume swings ≈ 82 mL. The binned material disks (≈ 84 mL) over-read by 6–10 % (the 2 mm output cells and their coverage weighting); the valve-plane link fluxes are exact for the semilunar valves (74 mL = the aortic cuts), but not for a moving AV plane, which sweeps nodes through cover / refill (mitral 67.5 mL).
- *Mechanism (CPU tests):*
  - a sphere scaled uniformly with an outlet pipe pumps 0.973 of its volume change (`contracting_chamber_pumps_its_volume_change`, c_s 11.5 m/s; 0.93 in a soft lattice, where the chamber's compliance delays the outflow by 24°);
  - a closed valve carried along a static tube pumped "0.876" of its swept volume (`moving_closed_valve_pumps_like_a_piston`) — *corrected the next night:* the test's Gaussian displacement bump (σ 7 mm) moved the sheet's faces only 0.92 as far as its centre, which the expected value used. With a flat-top bump (the valve region moving rigidly) it pumps **0.981**.
  
  So the moving sheet loses ≈ 2 %, not 12 %: the closed AV plane is *not* the main cause of the heart's ≈ 9 % deficit (the paragraph that called it the root cause was wrong).
- *Next:* mass-conserving cover / refill for moving solids, for valves and walls alike. A covered node hands its mass to its fluid neighbours ahead of the solid, a refilled node takes its mass from its neighbours, and the moving-wall term stays for the momentum only. The two CPU tests are the gates (≥ 0.98), then the heart's cut balances against the model's stroke.
- *Wall mass (hasty-cfd `run.wall_mass`, default off until validated on the heart):* the step accumulates each wall link's bounce-back mass change, a covered node its lost mass, a refilled one its new mass; at every geometry update the sum per region (the chambers, `blood_*`; the trees keep their transpiration, a deliberate wall mass flux) goes back to that region's fluid nodes as rest mass. A region's walls then change its volume, never its mass, and its pressure drives the displaced volume out. CPU tests: contracting chamber 0.973 → 0.976, the piston (flat-top bump) 0.981 → **0.998** on both sides.
- *First fix attempt (negative):* a valve core's bounce-back wall at the solid fraction's ½ crossing along the link (`run.valve_q: true`, kernel GeomParams.vq; default off) instead of half-way: the piston pumps 0.854 instead of 0.876. With q spread over the cell the interpolated bounce-back's moving-wall term injects $\tfrac{1}{2q}$ of the ideal mass for $q > \tfrac12$ (its interpolation part makes up the rest only where the fluid at the node already moves with the wall); averaged over q uniform on (0, 1] that is $\tfrac12 + \tfrac12\ln 2 = 0.847$, close to both numbers. In front of the sheet sits a ramp of partially saturated cells whose fluid does not yet move with the sheet. The fix belongs in the cover / refill mass bookkeeping (above), not in q.

**The chambers' pump (2026-11-27, night).**
- *Neutral wall mass is wrong for the heart.* The cover / refill / bounce-back sums made mass-neutral per chamber (`run.wall_mass`) cut the strokes to 52.6–57 mL. The per-chamber split (diagnostics `HASTY_CFD_WM_DEBUG`, `HASTY_CFD_WM_NOAPPLY`) shows why. In LV systole the moving-wall bounce-back injects ≈ 71 node volumes per 4-step update, while covering removes only ≈ 44; the rest of the region's node loss is region switching at the moving valve planes.
- *Mass balance of a region* (fluid nodes; density steady): the outflow through its openings equals its bounce-back injection. Cover, refill and region switching only change its volume. So **the bounce-back is the chambers' pump**. Over the third cycle the LV's systolic injection was 74.8 mL (the aortic tree's cuts: 75.5) and the RV's 73.5, against the model's 83.8: an efficiency ≈ 0.89 on the heart's curved, stair-cased moving walls (the interpolated bounce-back's moving-wall term carries $\tfrac{1}{2q}$ of the ideal mass for $q > \tfrac12$).
- *Wall pump* (hasty-cfd `run.wall_pump`, Case `wall_pump`: per chamber the cardiac model's dV/dt table; lbm_case writes them and sets it on, `--no-wall-pump`: old). At every geometry update each chamber's measured bounce-back injection is corrected to −dV/dt · Δt of the model, the difference spread over the chamber's fluid nodes as rest mass; cover / refill are left alone.

| healthy full15, 1 mm | before | wall pump | model |
|---|---|---|---|
| aortic tree cuts (mL / beat) | 75.5 | **84.7** | 83.8 |
| pulmonary tree cuts | 75.0 | **83.4** | 83.8 |
| systemic venous inlets | 76.4 | **84.7** | 83.8 |
| pulmonary venous inlets | 78.7 | 89.2 | 83.8 |
| valve-plane link strokes (aortic / pulmonary / mitral / tricuspid) | 74.1 / 73.2 / 67.5 / 66.4 | 83.4 / 81.2 / 85.6 / 81.7 | 83.8 |

All four valve gates pass, the density gate too; the trees balance within 0.6 / 2.6 %. Open: the pulmonary-vein inflow +6 %. Overnight: the same through freshly built cases (healthy, mr 0.2, as 1.0 at 1 mm; healthy at 0.5 mm).

**Wall pump through fresh cases (2026-11-27, morning).**

| case | cut strokes (aortic / pulmonary / pulmonary-venous / systemic-venous, mL) | gates |
|---|---|---|
| healthy 1 mm | 84.7 / 83.4 / 89.2 / 84.7 (model 83.8) | all pass |
| as 1.0, 1 mm | 88.3 / 87.8 / 87.1 / 85.4 (83.7) | all pass (strokes +4–5 %) |
| mr 0.2, 1 mm | aortic tree 53.0 (its target); mitral plane while closed −33.8 mL per beat (link, moving plane), LV mass balance ≈ 29.5 (model 27.9–30) | material disk reads the jet ×1.4 |
| healthy 0.5 mm | 91.6 / 90.1 / 91.6 / 88.5 (83.8) | stroke +8–9 % |

- The over-pump grows as the time step shrinks (1 mm dt 40 µs: +1 %; as 1 mm dt 13 µs: +5 %; 0.5 mm dt 20 µs: +9 %).
- At 0.5 mm the raw bounce-back delivered 0.833 (LV) / 0.841 (RV) of the target and the corrected chambers 1.09: another resolution-dependent mass source beyond the bounce-back, not yet identified.
- Remedy, general: a *pump gain* per side, set every cycle from the venous inlets' exact link inflow against their model targets (the inertial inlets deliver what the chambers draw), × (want / measured)^0.7 within 0.7–1.5, logged as `pump_gain`. Under test.

**Clearance: the left coronary artery on the aortic root** (`clearance.py --watch "A|B"`: per iteration the worst pair between two vessels, its push, step, displacement against cap and mobility):
- the coronary sample got the full push (1.52 mm) and was free (cap 4 mm, mobility 1), but the spreading along the vessel turned it into 0.1–0.2 mm per iteration; its deficit fell steadily (1.52 → 1.04 mm), until the global plateau stop at iteration 21 ended the solve;
- the total deficit is dominated by pairs that no longer move. New default `--plateau worst`: stop only when the deepest violation has also stopped improving (< 3 % over 10 iterations); `--plateau deficit`: the previous rule. Under test.

**Pump gain results** (healthy full15, 6 cycles with 3 warm-up; the gain settles in about two cycles):

| | gain left / right | valve-plane strokes (aortic / pulmonary / mitral / tricuspid, mL) | cut strokes (mL) | model |
|---|---|---|---|---|
| 1 mm | 0.955 / 0.99 | 81.5 / 82.1 / 83.1 / 83.2 | 82.1 / 83.6 / 84.6 / 83.6 | 83.8 |
| 0.5 mm | 0.91 / 0.94 | 83.0 / 83.1 / 84.5 / 84.6 | 85.3 / 86.3 / 84.4 / 83.9 | 83.8 |

All gates pass at both resolutions; the trees balance within 0.1–3.7 %. The gain is not in the checkpoint: a resumed run starts again from 1 (about three cycles to settle).

**Clearance rerun with `--plateau worst`** (the full15 phantom copy, solve only, no carve):
- 27 iterations instead of 21;
- the left coronary artery on the aortic root: 1.66 → 1.20 mm in the audit. In the solver's view it stalls at ≈ 1.0 mm with 0.85 of its 4 mm cap used, sandwiched between the anchored root and the LA (0.65 mm tissue contact): the LA has to yield (carve rounds);
- the worst pair is now the pulmonary bifurcation's own branches (1.52 mm).

Next: the test phantom through the production path with both rules (full16).

**full16 / full17 (2026-11-27), production path on the test phantom (full15 graph export):**
- full16 (`--plateau worst` + adaptive carve rounds, 3243 s): 183 pairs ≤ 1.77 mm, 12 tissue ≤ 0.31 mm. The worst pair, the
  pulmonary bifurcation against itself, was never pushed: the solver's sub-sampling (spacing up to r / 2 on the 13 mm
  trunk) skipped the contact samples.
- full17 (+ solver sample spacing ≤ min(r / 2, 2 mm), 3275 s, 7 solve / carve rounds): **162 pairs ≤ 0.50 mm** (left
  coronary artery against the ascending-aorta connector 0.50, IVC → RA 0.48, coronary sinus / middle cardiac vein 0.48;
  pulmonary bifurcation 0.30), **25 tissue contacts ≤ 0.48 mm** (LA 14, RV 5, myocardium 5, RA 1). Tissue got a little
  worse than full16 (more contacts, still < ½ voxel at 1 mm); the pairs are now all ≤ ½ mm.
- Flow check: as before except the pulmonary arterial peak 1.20 m/s at the left apical segmental artery, on its limit
  (1.18 in full16; more samples shrunk to the 0.9 floor: 6086). Head flows (vertebral, basilar, ACA, dural sinuses)
  remain the open head-phase item.
- Logs: `data/hasty/scratch/chain_full16.log`, `chain_full17.log`; chain script `jobs/phantom/chain_full17.sh`.

**LBM on the full17 geometry** (1 mm, 6 cycles with 3 warm-up, wall pump + pump gain, `lbm_burst.sh`, 133 s at 601 MLUPS):
valve-link strokes aortic 81.7 / pulmonary 81.6 / mitral 83.2 / tricuspid 82.9 mL (model 83.8), cuts 82.7–84.7;
closed-valve rms ≤ 0.17 mL/s; trees balance −0.7 % (aortic) / −2.0 % (pulmonary); density gate ok; cut cross-check
median 1.049 (10–90 % 0.98–1.09). All gates pass. Open realism item: the early-diastolic peak speed 1.98 m/s 14 mm
past the mitral valve (1.60 before the wall pump); a healthy E wave is ≈ 0.6–1.0 m/s. The model's LV filling peaks at
590 mL/s through a 16 mm annulus (8 cm²; bicuspid leaflets `bi:0.2,5,7`), i.e. the effective open area is ≈ 3–4 cm²
where a normal valve opens to 4–6 cm² — check the leaflet opening and the E-wave rate.

## Valve jets, filling and ejection fraction (2026-11-27)

**Finding.** On the full17 test phantom at 1 mm every valve jet was about twice the healthy value (report `jet_max`: the
largest |u| on the valve disk and 4 / 8 / 12 mm downstream over the cycle): aortic 2.78 m/s (healthy 1.0–1.7; ≥ 2.5 is
graded sclerosis / stenosis), pulmonary 1.93 (0.6–1.2), mitral 1.91 (E 0.6–1.0), tricuspid 1.77 (0.3–0.7). Causes, measured:
- *Semilunar cusps inside the lumen.* The hinge circle was at 1.05 × the lumen radius and the open 3-cell (3 mm) cusps
  stood inside it: 2.85 of 4.87 cm² open (aortic), 3.01 of 5.10 (pulmonary). Real cusps open into the sinuses of
  Valsalva, so the orifice is about the annulus.
- *Filling shape.* The E wave was a symmetric sin² lobe of 0.26 T and A one of 0.14 T; with seed 1's E share 0.58 the
  A wave peaked at 590 mL/s against E 437 (E/A 0.74: the impaired-relaxation pattern), peak filling rate ≈ 5 EDV/s.
- *Ejection fraction.* SV was drawn independently of the atlas EDV: LV EF 0.75, RV 0.66 for seed 1 (healthy LV
  0.55–0.70).
- *Profile.* Even at a physiological mean through-flow the 1 mm jets are 1.7–2.2 × the disk-mean normal velocity
  (mitral E: disk mean 0.90 m/s, disk max 1.67, 12 mm downstream 1.94): the 3-cell plates act as a sharp-edged orifice.
  The leaflet kinematics are physiological (opening / closing over 33 ms, tip speeds 0.64–0.71 m/s).

**Changes** (old behaviour behind flags):
- `lbm_case --sl-hinge lumen` (default; `annulus` before): semilunar hinge circle half a leaflet + half a cell outside
  the lumen; open-area accounting (`open_area(r_in)`) and the stenosis angle count only the lumen disk; every valve's open
  area, lumen and mean orifice velocity at the model's peak are printed.
- `cardiac_motion --filling physio` (default; `sin2` before): E rises over 90 ms and decelerates linearly over DT = 180 ms
  (stenosis: PHT / 0.29), A a 150 ms sin² lobe ending 40 ms before ejection, all × √(T / 0.857 s); the RV uses the LV's E
  share (sin2: 0.05 less). The motion summary records E / A peaks.
- `make_phantom --fe-range 0.65,0.80` (was 0.55–0.75): E/A peak ≈ 1.1–2 with the physio shapes.
- `make_phantom --sv-mode ef` (default; `draw` before): the sampled SV's place in 60–85 mL picks the LV EF in 0.55–0.70 and
  SV = EF × the LV EDV of the atlas / diffeomorphism (minus the LV-side regurgitant volume). Seed 1: EF 0.69, SV 77 mL
  (CO 5.4 L/min) instead of 83.8 mL.
- `lbm_report`: per valve `jet_where` (time, depth, radius share, normal share of the maximum) and `at_peak_flow` (mean /
  max normal velocity over the disk's fluid area); `lbm_case --motion <file>` (another cardiac model, tests);
  `lbm_burst.sh` passes `CASE_ARGS`.

**1 mm results** (full17 geometry, 6 cycles, wall pump; `jobs/phantom/valve_realism.sh`):

| run | aortic jet | pulmonary | mitral | tricuspid | gates |
|---|---|---|---|---|---|
| baseline (annulus hinge, sin2, fe 0.58) | 2.78 m/s | 1.93 | 1.91 | 1.77 | pass |
| + sinus hinge | 2.36 | 1.58 | 1.91 | 1.79 | pass |
| + physio filling (fe 0.672; mitral peak 590 → 417 mL/s) | 2.48 | 1.59 | 1.94 | 1.98 (an opening transient at 401 ms, backwards near the centre; at peak flow 1.30) | pass, trees −3.2 / −4.6 % |

Open areas at 1 mm / 0.5 mm (`open_area`, thickness 3 cells + 1): aortic 4.07 / 4.71 cm² at 85°, pulmonary 4.26 / 4.93,
mitral 4.81 / 5.65 at 80°, tricuspid 5.24 / 6.17 — at 0.5 mm all within the healthy ranges.

**0.5 mm and eccentricity** (same changes, `lbm_full17_hf05`, 6 cycles, 1360 s at 940 MLUPS; all gates pass, strokes
82.7–84.1 / 83.8 mL). New report metrics at the time of peak model flow, on the valve plane and one lumen diameter
downstream (≈ the sinotubular junction for the aortic valve), as graded in 4D-flow studies: normalised flow
displacement (forward-flux centroid offset / diameter), jet angle (flux-weighted velocity vs the axis), and the lumen
share carrying half the forward flux (plug 0.50, Poiseuille 0.29):

| | aortic jet | angle at plane | displacement / A50 at 1 diameter | mitral 12 mm at E peak |
|---|---|---|---|---|
| 1 mm | 2.48 m/s | 30° | 0.27 / 0.16 | 1.94 m/s |
| 0.5 mm | 2.41 m/s | 29° | 0.26 / 0.16 | 1.65 m/s |

Resolution is not the cause. Healthy references: mid-ascending peak systolic velocity 1.12 ± 0.2 m/s, central with
almost no displacement; bicuspid valves 1.96 ± 0.7 / 2.11 ± 0.9 m/s with eccentric helical jets (Hope et al.,
PMC3559505; Radiology 2010) — the healthy phantom looked bicuspid. Cause: the atlas LVOT is ≈ 7.1 cm² right up to the
annulus (8.7 at −10 mm, 12.2 at −20 mm) against the 4.9 cm² lumen, so the flow meets a sudden contraction (area ratio
0.7) and enters 30° off the axis; the RVOT likewise (7.8 → 5.1 cm²). A real LVOT is about the annulus diameter (the echo
continuity equation uses it as such) and narrows over ≈ 2 cm. In disease peaked, eccentric jets are physical (AS
2.5 → > 4 m/s, BAV ≈ 2 m/s eccentric, MR / AR 4–6 m/s, MS 1.5–2.5 m/s, coarctation / carotid stenoses 2–4 m/s;
aneurysms slow and vortical): the target is the right physics in each case, not uniformly blunt profiles.
Next: `lbm_case --ot-funnel L,k` (an outflow-tract funnel, below), then a contraction-coefficient validation (sharp
orifice ≈ 0.61, smooth nozzle ≈ 1) so stenoses get their jets from the geometry instead of the assumed CC 0.75.

**Outflow-tract funnel: worse, off by default** (`--ot-funnel L,k`, 1 mm, `jobs/phantom/funnel_test*.sh`). A cone on the
valve axis narrowing the ventricle to the lumen over 20 mm (7.4 mL of LV walled) made the aortic jet 3.48 m/s: the LVOT
lies 38° off the stub axis (its centroid 15.6 mm off at 20 mm), so the straight cone cut a new narrowing (4.42 cm² at
−12 mm). Bent to the tract's centroid: 2.89 m/s, angle at the plane still 28°. The step contraction is not the main
cause. Velocity slices (`lbm_report --flow-slices`, report/flow_slices.png) show why: the LVOT and the ascending aorta
form one straight channel and only the valve axis (the stub's first two centreline samples) is tilted ~30° to it. The
pulmonary valve, whose axis matches its tract (2°), already looks healthy: central, ≈ flat, 1.0–1.3 m/s.

**Displacement measured as in the 4D-flow literature** (flux centroid relative to the lumen's own centroid in that
cross-section, over the lumen's equivalent diameter; the lumen = the fluid component holding the peak forward flux): on
the baseline 1 mm run the aortic displacement one diameter downstream is 0.083 (A50 0.31), not 0.27 — most of that was
the vessel curving away from the valve axis. Effective orifice areas (peak flow / peak jet, the echo continuity
equation; report "effective orifice"): aortic 2.50 cm² of 4.07 open (0.61), pulmonary 2.96 of 4.26 (0.69), mitral 2.13
of 4.81 (0.44), tricuspid 3.23 of 5.24 (0.62); healthy valves ≈ 0.9–1.

**Next: `lbm_case --sl-axis channel`** (default): the semilunar axis from the outflow tract's centroid 20 mm upstream to
the vessel centreline two diameters downstream (`stub`: before).

**full18** (2026-11-27, `jobs/phantom/chain_full18.sh`, 3194 s): the realism defaults of the day in the production
path — 137 pairs ≤ 0.86 mm (LCA on the aortic-root connector 0.86, the coronary sinus into the myocardium 0.63, the rest
≤ 0.45), 33 tissue contacts ≤ 0.63 mm (myocardium 15, LA 13). Fewer pairs than full17, a deeper worst one (the LCA root
contact moves with the heart motion; next to look at). Flow check: every class within its peak-velocity limit
(pulmonary arterial 1.15 m/s, was 1.20 at the larger SV); the head flows remain the head-phase item.

**Coronary flow timing** (`network_flow --coronary-imp κ`, off by default until calibrated). The network ended the
coronaries in ordinary Windkessels to venous pressure, so their flow was systole-dominant like any artery (full17: LCA
peak 7.5 mL/s at mean 1.68). Real LCA flow is ≈ 70–75 % diastolic (the contracting myocardium squeezes the intramural
vessels), RCA less (the RV pressure is low). Model (as Kim et al. 2010's lumped coronary terminals): the terminal
compliance is referenced to an intramyocardial pressure κ·P_LV(t) (left territory) / κ·P_RV(t) (right), a source current
iωC·P_im / (1 + Rc/R + iωC·Rc) per harmonic at each coronary terminal, mean flows unchanged. P_LV / P_RV come from a first
pass: the root pressure while ejecting, isovolumic relaxation (80 / 60 ms) and contraction (40 ms) ramps, filling
pressure 5 → 10 (LV) / 3 → 6 mmHg (RV). The network summary now records each key vessel's diastolic share.

Results (full17 network, `jobs/phantom/coronary_imp_test.sh`; κ = 0 reproduces the stored network bit-exactly):

| terminal, κ | LCA peak / min (mL/s) | LCA peak velocity | LCA diastolic share | RCA diastolic share |
|---|---|---|---|---|
| 3-element, 0 (before) | 7.5 / 0.2 | 65 cm/s | 0.48 | 0.34 |
| compliance on P_im, κ 1 | 8.4 / −4.7 (backflow) | 73 | 0.63 | 0.43 |
| **5-element (Ra – Ca – Ram – Cim – Rv, 0.32 / 0.52 / 0.16 of R, 0.11 / 0.89 of C), κ 1** | 4.8 / 0.2 | 41 | 0.57 | 0.42 |

The single compressed compliance pushed blood back into the epicardial artery; the 5-element terminal (the microvascular
resistance between the arterial and the intramyocardial compliance) has no backflow and a proximal LCA peak velocity in
the Doppler-wire range of normal LADs (≈ 30–50 cm/s). The diastolic share saturates near 0.57–0.58 with κ (0.75 → 1.25),
somewhat below normal LAD values (diastolic / systolic velocity-time integral ≈ 1.5–2, share ≈ 0.6–0.67); the RCA is
lower, as it should be. Default κ 1 with the 5-element terminals (`--coronary-imp 0`: before).

**Semilunar channel axis and opening angles** (full17, 1 mm, one factor at a time against the stub-axis run; jets are
the maximum while the valve is ≥ half open, EOA = peak flow / peak jet):

| run | aortic jet | aortic angle at plane | displacement at 1 diameter | aortic EOA / GOA | mitral jet | mitral EOA / GOA |
|---|---|---|---|---|---|---|
| stub axis, 85 / 80° | 2.48 m/s | 30° | 0.08 | 2.50 / 4.07 | 1.94 | 2.13 / 4.81 |
| channel axis (24° turned; the LVOT approached 38° off) | 2.29 | 11° | 0.04 | 2.74 / 4.07 | 1.95 | 2.12 / 4.81 |
| + opening 90 / 85° | 2.29 | 12° | 0.03 | 2.78 / 4.60 | 1.71 | 2.42 / 5.34 |

Both kept as defaults (`--sl-axis channel`, `--theta-open-deg 90,85`). The aortic flow now enters nearly on axis and
leaves central (healthy displacement); what remains is the profile: peaks ≈ 1.9–2.3 × the disk mean, EOA / GOA ≈ 0.6
(healthy ≈ 0.9–1) — the 3-cell plates at 1 mm behave as sharp edges. full18 (new phantom, SV 77 mL, these defaults
except the opening angle): aortic jet 1.88 m/s, pulmonary 1.44, mitral 1.60 (E disk mean 0.83), all gates pass.

**full18 at 0.5 mm** (channel axis, opening 90 / 85°, sinus hinge, physio filling, SV 77.3 mL; 6 cycles, 1372 s at 936
MLUPS; all gates pass, valve-link strokes 76.2–77.6 / 77.3 mL; trees −4.9 / −6.0 %):

| | aortic | pulmonary | mitral (E) | tricuspid |
|---|---|---|---|---|
| jet | 1.87 m/s | 1.44 | 1.29 | 2.05 (opening transient; p95 1.14) |
| disk mean / max at peak flow | 1.02 / 1.60 | 0.74 / 1.24 | 0.68 / 1.29 | 0.66 / 1.08 |
| EOA / GOA (cm²) | 2.87 / 4.63 | 2.84 / 4.84 | 2.95 / 6.22 | 3.33 / 6.90 |
| displacement at 1 diameter | 0.03 | 0.03 | 0.18 | 0.15 |
| healthy peak | 1.0–1.7 | 0.6–1.2 | 0.6–1.0 (young ≤ 1.3) | 0.3–0.7 |

From the morning's 2.78 / 1.93 / 1.91 / 1.77 m/s: aortic and pulmonary central (healthy displacement), mitral in range;
peaks still ≈ 20–40 % high with EOA / GOA ≈ 0.5–0.6 (healthy ≈ 0.9–1).

**Tricuspid opening transient** (2.0–2.3 m/s backward near the centre at 401 ms, at both resolutions): the opening table
was a centred 30 ms moving average, so every valve started opening 15 ms before its flow — the leaflets (3 cells thick,
≈ 2.7 mL of plate on a 9 cm² annulus at 1 mm) swept into a ventricle that was not filling yet and their volume left
backwards through the slit. `lbm_case --open-ramp causal` (default; `centred` before): the opening starts at the flow
onset (the pressure crossover); the closing keeps the centred ramp (min of the two smoothings). Under test.

Results (full17, 1 mm; `jobs/phantom/funnel_retest.sh`):

| run | tricuspid jet | mitral jet | aortic jet | aortic EOA / GOA | gates |
|---|---|---|---|---|---|
| centred opening | 1.97 m/s backward at 401 ms | 1.71 at E peak (p95 1.62) | 2.29 | 2.78 / 4.60 | pass |
| causal opening | 1.37 forward at E peak | 1.94 at opening, 418 ms (p95 1.63) | 2.29 | 2.83 / 4.60 | pass |
| + outflow funnel on the channel axis | 1.37 | 1.95 | 2.29 (disk max 1.89) | 2.67 / 4.60 | density FAIL (LV −5.6 %) |

Causal opening kept (default): the backward tricuspid jet is gone; the mitral shows a shorter forward pinch at opening
instead (p95 unchanged) — at 0.5 mm (half-thickness leaflets) it should shrink. The outflow funnel stays off: aligned
with the channel it still does not help and it breaks the LV density gate.

**Where the valves stand** (healthy phantom): mean through-flow, filling pattern, EF, jet direction and eccentricity are
physiological; peak velocities are ≈ 20–40 % above healthy at 0.5 mm (EOA / GOA ≈ 0.5–0.6 against ≈ 0.9–1) because the
leaflets are rigid plates three cells thick with square edges. The remaining levers are the leaflet representation
(sub-cell thin membranes, rounded edges) or local refinement around the valves — kernel work, not case-builder rules.

## Membrane leaflets (hasty-cfd, 2026-11-27)

Leaflets as zero-thickness membranes instead of partially saturated plates three cells thick (`"membrane": true` per
valve; `lbm_case --leaflets membrane`). Kernel (`lbm_core.h`): `valve_solid` keeps only the fixed annulus ring and the
circular segments behind the hinge chords as solid; `node_geom2` tests every link between two fluid nodes near a membrane
valve against the closed disk (opening < 0.05, out to the ring radius) or the strips turned by θ (exact segment–plane
crossing, polygon test with a 0.05-unit tolerance so shared strip edges are never missed; regurgitant orifice open), in
the valve's frame (reference positions for tissue-frame valves, so the closed membrane stays topologically sealed under
the motion). A crossing is an interpolated bounce-back wall link (Bouzidi, q = the crossing fraction) with the leaflet's
velocity (tissue velocity + the push × rotation term). Nothing else changes: the step is the walls' bounce-back.

Tests (all pass; `cargo test -p hasty-cfd --lib`, 20 CPU tests): closed membrane valves (tri, bi) pass 0.0000 mL/s, open
ones carry the outlet's flow through the exact valve-plane link flux (within 1 %), no solid nodes left in the lumen; the
closed membrane carried as a piston pumps 0.980 of its swept volume (plates 0.981); open-valve steady flow peak / mean
2.26 (plates) → 1.91 (membrane) with 2.3 × the flow at the same pressure; GPU = CPU: identical link bits and q
fractions, velocities within 1e-6 with WALE off (with WALE 6e-4: its gradient-based viscosity amplifies fast-math
rounding in the faster jet; tolerance documented in the test).

Heart (full17, 1 mm, all defaults, against the same with plates):

| | plates | membrane |
|---|---|---|
| mitral jet / p95 | 1.94 (opening pinch) / 1.63 m/s | 1.39 / 1.31 (max at E peak) |
| mitral EOA | 2.42 cm² | 2.97 cm² |
| tricuspid jet / p95 | 1.37 / 1.27 | 1.12 / 1.03 |
| aortic / pulmonary jet | 2.29 / 1.58 | 2.34 / 1.54 |

All gates pass; closing regurgitation 0.1–0.2 mL per beat (real valves: a few mL). The AV valves lose the leaflet-volume
and thickness artefacts; the semilunar valves do not change — their open cusps were already outside the lumen. Their
remaining contraction: the carved valve cylinder (hinge radius + 2 mm rim) ended 4 mm downstream in a step down to the
lumen (8.5 → 4.9 cm² aortic). `--sl-sinus none` (default with membranes): the downstream carve of a sinus-hinged semilunar
valve stops at the lumen. Under test.

`--sl-sinus none` result (membranes, full17, 1 mm): aortic 2.38 m/s, pulmonary 1.53 — unchanged (2.34 / 1.54 with the
pocket); kept as the membrane default (no step, no measurable cost). The semilunar profile is peaked at the valve plane
itself (disk max / mean at peak flow 2.0 at 1 mm, 1.57 at 0.5 mm): a resolution / effective-viscosity effect (12.5
lumen cells at 1 mm), not the valve geometry. Membranes are the `lbm_case` default from now (`--leaflets plates`:
before). Under test: full18 at 0.5 mm with membranes; WALE off at 1 mm.

**Production numbers with membranes** (full18 at 0.5 mm, all defaults; 6 cycles; all gates pass; valve-link strokes
74.9–76.0 / 77.3 mL; trees −5.3 / −6.3 %):

| | aortic | pulmonary | mitral (E) | tricuspid (E) |
|---|---|---|---|---|
| morning (full17, 1 mm, old model) | 2.78 m/s | 1.93 | 1.91 | 1.77 |
| 0.5 mm, plates | 1.87 | 1.44 | 1.29 | 2.05 (opening transient) |
| 0.5 mm, membranes | 1.81 (p95 1.71) | 1.43 | 1.19 | 1.01 (disk mean 0.56) |
| healthy | 1.0–1.7 | 0.6–1.2 | 0.6–1.0 (young ≤ 1.3) | 0.3–0.7 |

Displacement one diameter downstream 0.03 at both semilunar valves (central). WALE off (1 mm membranes): aortic 2.30 vs
2.38, pulmonary 1.51 vs 1.53 — the eddy viscosity is not the cause. What remains at the semilunar valves is resolution:
the profile at the valve plane (peak / mean at peak flow) 2.0 at 1 mm → 1.64 at 0.5 mm. The mitral is in range; the
tricuspid ≈ 40 % above its healthy upper bound with a physiological disk mean.

## Local grid refinement (hasty-cfd, 2026-11-27)

The remaining semilunar peaks are resolution (above), and the coronaries and small vessels need finer lattices than the
chambers: local refinement instead of a finer lattice everywhere. Levels of spacing $\Delta x / 2^l$ (2:1 between
neighbours, at most 3), each its own sparse 8³-tile lattice taking two steps per step of the next coarser one (acoustic
scaling: same lattice velocity, $\tau_l - \tfrac12 = 2^l (\tau_0 - \tfrac12)$, the bulk and sponge relaxation scaled
alike). Cell-centred nesting: fine node $2g + \{0,1\}^3$ lies in coarse cell $g$.

Code: `refine.rs` (regions → node roles and interface lists), `kernels/lbm_core.h` "grid refinement" (the kernels and
the derivation), `solver.rs` `step_level` (recursive), `case.json` `"refine"` (boxes / spheres per level),
`tools/phantom/refine_spheres.py` (spheres around valves along their path through the cycle and downstream over the
jet; `lbm_case --refine aortic,pulmonary|all`, or as a script on an existing case with symlinked arrays).

Scheme (volumetric, Rohde et al. 2006; waLBerla, Schornbaum & Rüde 2016), per coarse step:

1. explosion — each coarse rim cell (owned, next to the refined region) fills its 8 fine ghost cells: the equilibrium
   at ρ, u linear to the ghost's centre, the post-collision non-equilibrium copied;
2. two fine steps — the ghosts stream with their wall links but do not collide, so after two fine steps a ghost
   packet has moved one coarse cell, like the coarse packet;
3. restriction — ρ, u, s of the hidden coarse cells next to owned ones from their fine cells;
4. coalescence — a rim cell's packets from a hidden neighbour are the mean of its ghosts' twice-streamed values;
5. the coarse step.

Mass is exact: the rim cell's rest population gets $L - R - (I - O)/8$ (packets it sent into hidden cells, packets it
pulls from them, the fine level's net intake from its ghosts counted link by link) — the residual of the interface edges
and corners.

What the tests decided (`cargo test -p hasty-cfd --lib refined_`):

| | interface pressure step (pipe, mm of pipe), coarse → fine / fine → coarse | stable at τ0 0.501 |
|---|---|---|
| piecewise-constant explosion, copied neq | 0.75 / 0.86 | yes |
| piecewise-constant, Dupuis–Chopard neq × (τ_f − 1)/(2(τ_c − 1)) | 1.16 / 2.22 | yes (× ½ only) |
| … and the full × ≈ 2 fine → coarse | — | no (τ ≤ 0.52 grows, 0.501 blows up) |
| **linear explosion, copied neq (default)** | **0.14 / −0.16** | **yes** |
| linear, per-layer neq ("layers") | 0.01 / −0.38 | yes |

The Chapman–Enskog stress itself scales as expected (post-collision $\Pi^*_{xy}$ fine / coarse measured 0.370, theory
0.375), but in this scheme the copy is the consistent transfer: the far ghost layer reaches the fine level after two
fine cells without collision (one more equilibrium shift), wanting $(\tau_f - 2)/(2(\tau_c - 1)) \approx 3/2$ of the
coarse stress where the near layer wants $\approx \tfrac12$ — they average to exactly 1; the fine → coarse packets come
from ¾ of a coarse cell away, which closes the stress gap exactly ($\tau_f = 2\tau_c - \tfrac12$). The equilibrium's
interpolation matters more than any rescaling: a piecewise-constant copy errs by $O(\Delta x\,\nabla u)$, the order of
the viscous non-equilibrium itself.

Further checks: closed box at rest stays exactly at rest; mass drift with a pulse across the interface (inside the fluid
and crossing the walls) at float level, as the uniform lattice; pipe: section flux continuous within 1 %, centre / mean
2.00 in the fine part; open membrane valve, flow-driven: peak / mean one radius downstream 1.90 (1 mm), 2.02 (1 mm +
0.5 mm box), 1.99 (0.5 mm); GPU = CPU (velocity moments 1e-4; WSS apart at the test's τ_f 1.1, documented). Independent
code review: one bug (a hidden cell without fluid fine cells kept ρ = 0 — thin vessels across the boundary), fixed.

Production test (one factor): full18 at 0.5 mm with membranes (the table above) + level 1 (0.25 mm) around the aortic and
pulmonary valves (72 spheres, valve radius + 6 mm, 10 mm downstream, along the cycle): 932 k coarse cells refined →
9.1 M fine nodes (+2.3 GB), 2.0 × the node updates; 3110 s for 6 cycles on the 4090 (uniform 0.5 mm: 1467 s), all
gates pass:

| full18, membranes | aortic jet / p95 | disk max / mean at peak flow | EOA | pulmonary jet / p95 | disk max / mean | EOA | mitral | tricuspid |
|---|---|---|---|---|---|---|---|---|
| 0.5 mm | 1.81 / 1.71 m/s | 1.66 | 3.00 cm² | 1.43 / 1.25 | 1.64 | 2.95 | 1.19 | 1.01 |
| + 0.25 mm at the semilunar valves | **1.71 / 1.64** | 1.56 | 3.24 | **1.33 / 1.21** | 1.62 | 3.08 | 1.20 | 1.05 |
| healthy | 1.0–1.7 | | | 0.6–1.2 | | | 0.6–1.0 | 0.3–0.7 |

Valve-link strokes 75.0 / 74.2 mL (uniform 76.0 / 75.1; model 77.4 / 77.3), trees −5.6 / −6.5 % (−5.3 / −6.3 %). The
aortic jet reaches the healthy upper bound, the pulmonary drops 7 %. The unrefined AV valves are the control (±0.04).
One open point: the pulmonary jet one diameter downstream moved off axis (displacement 0.03 → 0.12, angle 13° → 26°;
aortic unchanged) — that plane lay at the refinement boundary (0.7 mm outside). Test: the same with 25 mm downstream
reach (the plane 12 mm inside; 1.23 M cells, 11.9 M fine nodes, 2.3 ×; 3398 s):

| pulmonary, one diameter downstream | displacement | jet angle | A50 |
|---|---|---|---|
| 0.5 mm | 0.034 | 12.7° | 0.35 |
| refined, plane at the boundary | 0.120 | 25.6° | 0.34 |
| refined, plane 12 mm inside | 0.031 | 14.4° | 0.37 |

An interface artefact: a fast jet crossing the fine → coarse boundary is deflected there (the aortic jet, crossing
nearly normal to the boundary, was not). Refinement regions must contain the jets until they have spread
(`--refine-down-mm` ≥ the measurement / jet region); the interface itself is a to-do for jets that must cross one.
Numbers with the 25 mm reach (all gates pass; strokes 75.1 / 74.2 mL): aortic 1.74 m/s (p95 1.66, disk max / mean
1.57, EOA 3.22 cm²), pulmonary 1.33 (p95 1.23, 1.56, 3.21 cm²), mitral 1.18, tricuspid 1.03. Domain peak speed: higher
start-up spikes in the warm-up cycles 0–1 (5.0 vs 3.7 m/s), the accumulated cycles 3–5 as the uniform run (max 3.0–3.4,
p90 2.83–2.89).

All four valves refined (the same spheres around the AV valves; 3.0 M coarse cells, 28 M fine nodes, 4.1 × the node
updates; 5877 s; all gates pass):

| full18, membranes | aortic | pulmonary | mitral jet / p95, EOA | tricuspid jet / p95, EOA |
|---|---|---|---|---|
| 0.5 mm | 1.81 | 1.43 | 1.19 / 1.06, 3.20 cm² | 1.01 / 0.90, 3.71 |
| semilunar refined (25 mm) | 1.74 | 1.33 | 1.18 / 1.06, 3.23 | 1.03 / 0.87, 3.62 |
| all four refined | 1.74 | 1.36 | **1.10 / 0.99, 3.47** | **0.95 / 0.81, 3.92** |
| healthy | 1.0–1.7 | 0.6–1.2 | 0.6–1.0 (young ≤ 1.3) | 0.3–0.7 |

The AV jets drop with resolution too (mitral p95 into the healthy range; tricuspid still ≈ 15 % above). The AV-valve
strokes fell 2 mL (73.7 / 73.6 vs 75.5 / 75.2) and the closed-mitral leak rose (rms 2.6 vs 1.6 mL/s); the semilunar
disk max / mean moved 1.57 → 1.64 (aortic) and 1.56 → 1.65 (pulmonary) between the two refined runs with identical
semilunar refinement — inflow change or cycle-to-cycle variability, not separable without a noise estimate (a repeat).

Interface fix under test: the WALE eddy viscosity dominates the jets' shear layers (≈ 16 × ν), and for the same physical
gradient it is 4 × larger on the coarse level (ν_t ∝ Δ²), so a jet meeting the boundary obliquely sees a one-sided 4 ×
jump. `run.refine_wale_ramp n`: Δ² of the coarse level at the interface ramped down to the fine one over n fine nodes
(refine.rs wale_ramp, per-node scale in node_step). Test: the boundary-plane case (10 mm reach) with n = 8 (2632 s):
pulmonary one diameter downstream displacement 0.137, angle 27.5° (0.120 / 25.6° without the ramp, 0.031 / 14.4° with
the plane inside) — not the cause. The ramp stays available, off by default; the rule stays: refinement regions contain
the jets (`--refine-down-mm` ≥ 25). Remaining suspects: an under-resolved shear layer leaving the fine level breaks
down asymmetrically on the coarse lattice where the boundary meets the jet obliquely (resolution, not the scheme).

**Correction (2026-11-28, repeat run):** the deflection is not reproducible. 05r again, same case and code, averaging
cycles 7–9 instead of 4–6 (`jobs/phantom/refine_repeat.sh`, `lbm_full18_05r/run_late`): pulmonary 1-diameter
displacement 0.120 → **0.025**, angle 25.6° → 14.4° (05ra, plane inside the fine region: 0.027, 13.7°). The 0.12–0.14
of 05r / 05rw was beat-to-beat variability (or a late transient) of a 3-cycle phase average, not the interface; the
containment rule has no evidence behind it (keep `--refine-down-mm 25` only because it costs little). Pre-registered
comparison |late − early| vs |05ra − 05r| per metric:

| metric | 05r cycles 4–6 | 05r cycles 7–9 | 05ra cycles 4–6 | verdict |
|---|---|---|---|---|
| aortic jet max (m/s) | 1.71 | 1.71 | 1.74 | difference ≈ noise |
| aortic disk max / mean | 1.56 | 1.53 | 1.64 | 05ra difference 2× the window spread |
| pulmonary displacement | 0.120 | 0.025 | 0.027 | **noise** |
| pulmonary angle (°) | 25.6 | 14.4 | 13.7 | **noise** |
| mitral jet max (m/s) | 1.20 | 1.17 | 1.10 | refinement effect (0.10 > 0.03) |
| tricuspid jet max (m/s) | 1.05 | 1.01 | 0.95 | refinement effect (0.10 > 0.04) |
| strokes, valve link (mL) | 75.0 / 74.2 / 75.5 / 75.2 | 74.1 / 73.2 / 74.4 / 74.0 | 76.1 / 75.2 / 73.7 / 73.6 | −1 mL over 3 cycles: slow drift |

Decision: jet-direction metrics (displacement, angle, disk max/mean) need ≥ 6 averaged cycles before they are compared;
peak speeds are stable to ±0.04 m/s. The link strokes drift down ~1 mL per 3 cycles (pulmonary 73.2 / 77.3 mL, −5.3 %,
fails the 5 % gate in cycles 7–9; material disk and cuts stay at 78–81 mL): watch in longer runs.


## Motion realism: tethering (2026-11-28)

User: "the aorta moves further up but is quite static lower down … a weird kink". Measured along the aorta's centreline
(`jobs/tools/vessel_motion_check.py`: displacement, axial stretch of 2 mm segments, change of the bending angle over
±5 mm, cardiac × breathing; `jobs/tools/aorta_motion_modes.py`: the per-mode split of the stored cardiac field):

- the descending thoracic aorta behind the left atrium moved 13–20 mm with the heartbeat while it is static above and
  below: stretch 0.6–1.5, bends 14–23° — the harmonic extension of the AV descent + twist (13 mm there) and the LA's
  free-space volume mode (4 mm). Measured 4D CTA: 6.2 mm at the coronary ostia, 3.8 mm mid-ascending, the least in the
  descending aorta (Weber et al. 2014, Int J Cardiovasc Imaging 30:185);
- breathing pulled the retroperitoneal abdominal aorta 5 mm caudally and 3.6 mm anteriorly (stretch 0.85–1.20): the
  organs' smoothed mask reaches it.

Fix (`motion_inclusions.tether_mask` / `tether`, `--tether on` default in cardiac_motion.py and resp_motion.py; off: as
before): tethered cells — bone (heartbeat: all bone; breathing: the vertebral column only, the rib cage moves) and the
lumens of great vessels (radius ≥ 3 mm) whose wall lies within 8 mm of that bone and that lie ≥ 100 mm along the vessel
tree from the heart (graph path distance from the `connector:` attachments; a smoothstep hold over 70–130 mm). The
path criterion is what separates the ascending aorta (2–10 mm behind the sternum in this atlas, 0–80 mm along) from the
descending one (on the spine, ≥ 140 mm along); below the label field of view large vessels are held without the bone
test (retroperitoneal). Per basis field B: C = W B on the tethered cells, 0 beyond a band (heartbeat 30 mm, breathing
40 mm; 15 / 12 mm folded the fat between the ribs and the chest wall, det J −0.17 / −0.35), harmonic between; B ← B − C.
The heart is not held back: its walls near bone or the held aorta move less, and the Newton fit takes each chamber's
volume elsewhere (volumes exact).

ph0001 (factory_test18), tethered vs before: descending thoracic and abdominal aorta ≤ 0.8 mm, stretch 1.00, no bends
(was 13–20 mm, 0.6–1.5, 23°); ascending aorta and arch a smooth decay 15 → 0 mm over 120 mm (stretch ≤ 1.30 cardiac);
carotids / subclavian / brachiocephalic ≤ 3.6 mm; SVC, IVC, pulmonary trunk unchanged; chamber volumes exact; det J
outside the heart ≥ 0.27 (was 0.38), breathing ≥ 0.24 (0.48); the organs' breathing amplitudes unchanged.

Open: the aortic root lies inside the atlas's LV blood label, so it moves with the full AV descent (15 mm to the
sinotubular level; measured annulus 11.6 ± 2.9 mm, root 9.1 ± 1.9, ostia 6.2 ± 2.0) and is compressed with the chamber
(stretch 0.73, a 27° bend of the reference centreline's zig-zag at 10 mm); the ascending aorta then stretches up to
30 % in systole. Needs a root label (or a root region) with its own kinematics. New phantoms get the tethered motion;
the existing ones keep theirs until regenerated (make_phantom passes --graph to resp_motion).
## Vessel tree realism: portal system, venous entries, root and caval kinematics (2026-11-28)

User: "look at the vena cava downstream … like some other vessels has gone into it at some weird angle and they have
been smoothed together … we need to look at the vessel tree so it actually looks and moves realistically" — and that my
diagnostics do not always capture what a viewer shows. Two new looking tools, both CPU, no X server:

- `jobs/tools/vessel_render.py <graph_dir> <out.png> --center x,y,z --half hx,hy,hz [--rmin-mm r] [--only regex]
  [--organs blood_RA,liver --labels <tissue.hv>] [--motion <phantom> --states t:r,...]`: shaded first-hit renders of
  the capsule lumen (the model phantomflow paints), one colour per vessel name, anterior / lateral / posterior views
  (patient right on the viewer's left in the anterior view), optional atlas organs and motion states.
- `jobs/tools/vessel_contacts.py <graph_dir> "<name>"`: every vessel touching a named lumen — graph-adjacent
  (tributary) or foreign, centreline depth, contact length, angle. `jobs/tools/graph_junction_audit.py`: hairpins at
  degree-2 joints, branches leaving along a parent, bridge / join / connector lengths.

**Found (renders: data/hasty/scratch/vc_geom/):**
1. *Portosystemic shunt.* The atlas portal vein ends at the porta hepatis (z 1199 mm). The graph builder joined that
   free end to the nearest same-system vessel, the IVC: in the export graph (factory_arms, the 10-min phantom) a 27 mm
   bridge running back down into the IVC at 42° (a hairpin); in the full15/18 graphs the attachment wall rule
   (tolerance 2 r = 12 mm) attached it directly, and junction relocation made it an elbow straight into the IVC
   (z 1190). The network then sent 20 % of the cardiac output from the gut through the portal vein into the IVC.
   The superior mesenteric vein also lay 2–3 mm inside the IVC lumen for 40 mm in the arms phantom (the clearance solve
   has since separated it).
2. *Venous connectors pointing outward.* The atlas draws the venae cavae and the pulmonary veins 3–15 mm into their
   atrium; the connector then ran from that end to the nearest atrial-wall vertex: the SVC 9 mm sideways at 103° (a
   ball on top of the RA), the pulmonary veins out into the myocardium, the IVC a 15 mm stub inside the RA.

**Fixes (vessel_graph.py; old behaviour behind flags):**
- `ORGAN_ROOTS = {"Hepatic portal vein": "portal_venous"}`: an organ-rooted tree. Its vessels attach nothing (their
  tributaries attach to them), it is never bridged, its free organ end is never joined and is the tree's root; class
  `portal_venous` (pack_vessels code 2, murray_floor / flow_check as systemic veins). `--portal-shunt`: old.
- network_flow.py: the portal root is a sink at RA mean + the hepatic venous pressure gradient (`--hvpg`, 4 mmHg) with
  0.15 of the atrial wave; portal territory 20 % of the output; the hepatic veins drain 25 % (portal + hepatic artery)
  into the IVC. full19: portal vein 17.5 mL/s (14.8–19.4), mean 15.5 cm/s (normal 15–20); IVC thoracic 53.6 mL/s
  (abdominal 31.7 + liver 21.9), unchanged at the atrium.
- Venous roots already inside their chamber's blood (atlas labels p0/tissue_1mm.hv, `--chamber-labels`) are trimmed
  where the axis enters the chamber, root one sample inside, no connector (`--venous-connectors`: old): SVC 3.1 mm,
  IVC 15.3 mm, pulmonary veins 6–11 mm trimmed. The motion tether's path distance now seeds from the heart-class roots
  too (`motion_inclusions.heart_roots`), since these trees have no `connector:` edge any more.
- Pulmonary veins whose end node carries 2–4 branch edges are not trimmed (trimming one edge and moving the node left
  the others behind — independent review): their root stays at the atlas end inside the LA, no connector.

**Motion (cardiac_motion.py / resp_motion.py; tests on factory_test18 into data/hasty/scratch/motion_tether/root,
`jobs/phantom/motion_root.sh`, `motion_resp_art.sh`; `jobs/tools/aorta_root_profile.py`, `jobs/tools/resp_det.py`):**
- *Aortic root* (`--root-hold on`, `--root-s 0,15,50,80`, `--root-g 0.77,0.45,0.25,0.10`, `--root-classes
  systemic_arterial`): the root lumen + 2 mm wall along the aortic tree from the valve gets a prescribed motion —
  `tether(target=…)`: in the AV basis g(s)·(−e₁), held at 0 in the other bases (twist, chamber volumes; other chambers'
  cells excluded). Tried first: a hold weight 1 − g/|B_av| (the chamber modes and the Newton refit put the motion
  back: 0.73 instead of 0.45 at 15 mm, det J 0.12) and the pulmonary root too (folded the lung at the left pulmonary
  artery, det J −0.36; the trunk moved realistically without it). Result (s from the valve, × the AV descent 15.7 mm):
  0 mm 0.68 (10.6 mm; CMR annulus 11.6 ± 2.9), 15 mm 0.61 (9.6; CTA ostia 6.2 ± 2.0 — high), 50 mm 0.28 (4.5; CTA
  mid-ascending 3.8 ± 1.9), 100 mm 0.09; ascending stretch ≤ 1.20 (was 1.30); det J outside the heart ≥ 0.35 (was 0.27).
  (That test still seeded s = 0 at both connector ends; fixed after review: s from the valve only.)
- *Breathing*: the intrahepatic IVC moved 0–6 mm while the liver 15–35 mm beside it moved 8–13 mm; stretch 0.58 and a
  16° kink below the atrium. Cause: the atlas holds no vessel labels (a lumen is fat), and the clearance carve adds
  fat around great vessels, so the IVC's liver groove (open toward the crus) lay outside the "internal cavity" that
  carries the SI motion. Fixes: (1) `--embedded-vessels on`: a vessel's lumen joins the internal cavity where its
  shell (1–6 mm outside the wall), averaged ±30 mm along the edge, is ≥ 20 % internal organ (165 mL on ph0001);
  (2) the breathing tether holds arteries only (`--tether-veins on`: veins too); (3) `--tether-keep
  liver,spleen,kidney`: no tether correction inside the solid organs. Tried and rejected: mean pooling of the tether
  onto the 4 mm grid (det J 0.08). IVC SI motion at end-inspiration now: infrarenal 1.7–4.5 mm, renal level 4.5–5.8,
  coeliac level 2.4 (4.5 mm from the held aorta — a dip the 4 mm grid cannot resolve), intrahepatic 5.2–10.3, atrial
  junction 12.1; aorta below the arch 0; det J ≥ 0.40 (was 0.24).
- Path distance for the tether now also seeds from the heart roots (`--tether-seeds connectors`: as before): the IVC
  root never had a connector, so the IVC counted 636 mm from the heart through the azygos and its atrial end could be
  held as "far from the heart".

### Round 2: what the full19 renders showed (2026-11-28, afternoon)

**full19 phantom** (`jobs/phantom/chain_full19.sh`, factory_test19/ph0001: export_full19 graph, seed 1, clearance):
audit worst 0.45 mm pair / 0.21 mm tissue; aortic root motion now 0.67 / 0.42 / 0.23 / 0.09 × the AV descent at
$s$ = 0 / 15 / 50 / 80 mm (target 0.77 / 0.45 / 0.25 / 0.10; the review fix of the $s$ seeding moved 15 mm from 0.61 to
0.42); IVC–RA kink 3.7° (was 11°); breathing det J ≥ 0.47; flow check: every class within its velocity limit (portal
0.24 m/s), the head out of range as before. Renders: `data/hasty/scratch/vc_geom/ivc_full19_phantom.png`,
`ivc_full19_motion.png`.

**Breathing: the IVC still spot came back.** New tool `jobs/tools/vessel_resp_neighbours.py <phantom> <tissue_2mm.hv>
<regex>`: per sample of a vessel, its SI displacement at end-inspiration vs the median of each tissue 2–15 mm around it.
full19: infrarenal 0.1–2.5 mm, at the pancreas 2.2 mm (pancreas beside it 6 mm), between pancreas and liver **0.3 mm**
(shell 70 % fat, 16 % crus muscle, 0 % organ), intrahepatic 8.5–11.5 (liver 11–13). On full18 the renal level moved
4.5–5.8 — the portal → IVC shunt bridge was itself an "embedded vessel" and dragged the IVC. Measured motion for
comparison: IVC caudal excursion in quiet inspiration 21.7 mm (95 % CI 18.3–25.1; ultrasound, below the hepatic veins;
Blehar 2012, *Crit Ultrasound J* 4:18), markers near the renal arteries 5–7 mm, kidneys 2.5–20.5 mm (4D CT, SpringerPlus
2014). A tube does not stand still between two organs that move 6 and 12 mm.

**Fix — vessel rod** (`resp_motion.py --vessel-rod-mm 30`, default; 0: off, bit-identical to before): per vessel of the
classes `--vessel-rod-classes systemic_venous,portal_venous` and median radius ≥ `--vessel-rod-rmin-mm 4`, every 2 mm:
$a$ = median breathing motion of the internal-organ voxels in the shell 1–6 mm outside the wall, weighted by their share
$f$, blended with the field at the centreline (weight $w_0$ = 0.02); the centreline motion minimises

$$\int (f + w_0)\,|u - a|^2\,ds + \ell^2 \int |\partial_s u|^2\,ds,\qquad \ell = 30\ \text{mm}$$

on the vessel graph (edges joined at their nodes; data weight = half the neighbouring intervals). Where organs touch it
the vessel follows them over $\approx \ell/\sqrt f$, across gaps it interpolates. Its lumen cells (4 mm grid) are then held
at $u$ with the tether's harmonic correction (16 mm band; tethered cells and kept organs unchanged). ph0001: 632 samples
(SVC, IVC + hepatic / renal / iliac veins, portal + splenic + SMV, left intrarenal veins), 149 mL held.

Tests (`jobs/phantom/motion_rod.sh`, data/hasty/scratch/motion_rod/, one factor at a time):

| run | IVC resp SI (confluence → renal → coeliac → intrahepatic → RA) | IVC stretch (r = 1, + cardiac) | det J min | aorta resp |
|---|---|---|---|---|
| off (= full19) | 0.1 → 2.2 → **0.3** → 8.5 → 11.5 | 0.66 | 0.47 | 0–7.9 |
| rod, all classes, r ≥ 3 mm | 2.5 → 4.6 → 7.3 → 9–11 → 11.5 | 0.84 | **0.04** | 0–**5.4** |
| rod, veins, r ≥ 4 mm | 3.4 → 5.5 → 7.7 → 9.8 → 11.6 | 0.84 | 0.34 | 0–7.9 |
| below-FOV tether great arteries only | 0.0 → 3.0 → 2.8 → 8.5 → 11.5 | 0.67 | 0.48 | 0–7.9 |
| both (defaults) | 3.8 → 6.1 → 8.6 → 10–11 → 11.5 | 0.84 | **0.48** | 0–7.9 |

- All classes folded: a 3.7 mm pulmonary vein lying in the fat at the lung base was pulled to the lung's 10 mm against
  the static chest wall (det 0.04); a 3 mm suprarenal artery near the aorta held against the kept liver; the ascending
  aorta was smoothed toward the tethered descending aorta. Small vessels ride on their tissue; arteries have the tether
  and the root hold.
- Veins only: det 0.34 at the pancreas just above the SMA. Cause (pre-existing, 0.47 before): below the label field of
  view (z < 1120 mm) the breathing tether held *every* artery ≥ 3 mm — the SMA stood still for 36 mm and then jumped
  5 mm within 10 mm; the rod moved the SMV and pancreas more (5.6 → 7.7 mm), steepening that. Fix:
  `--tether-below-rmin-mm 4` (default; 3: as before) — below the FOV only arteries whose *edge median* radius is ≥ 4 mm
  (aorta, iliacs ≥ 4.4 mm) are held; the SMA (≤ 3.5 mm) moves with the mesentery. Review: a per-point radius test
  switched the hold on and off along the left external iliac (4.4–4.7 mm) → per edge.
- IVC stretch 0.84 (target 0.85) is at the atrial end and appears only with the cardiac field added (breathing alone
  ≤ 5 % strain, cardiac alone 0.91): a composition effect, open.
- Independent review: N = 0 crash (guarded + class assert), per-point below-FOV test (→ per edge), anchor weight, $w_0 > 0$
  assert, interval-weighted data term — fixed. Left renal vein (median 3.9 mm) is outside the rod (open).
- Render `data/hasty/scratch/vc_geom/ivc_full19_motion_rod.png`: IVC, portal, splenic vein and SMV shift caudally
  together, the aorta stays.

**Radius steps at name boundaries.** The atlas radius is per named mesh, RFIX scales single names (left subclavian
×1.29, pulmonary veins ×1.4–1.7) and make_phantom jitters each name by its own lognormal factor (σ 6 %): factory_test19's
IVC stepped 8.2 | 10.1 mm (thoracic ×0.91 | abdominal ×1.14), the aortic arch | thoracic aorta 9.4 | 11.0, left
subclavian | axillary 3.3 | 2.4, right superior | inferior pulmonary vein 2.8 | 4.4. Fix: `radius_taper.joint_taper`
— at degree-2 joints and along the parent's through-line at branch nodes (two largest edges ≥ 120° apart, all others
< 0.8× the smaller) both ends meet at the geometric mean, each ramping back multiplicatively over max(15 mm, 2 r) (≤ 45 %
of the edge); side branches untouched. In vessel_graph (`--joint-taper-mm 0`: off, bit-identical to export_full19 —
checked; `--joint-taper-through off`: degree-2 only) and again after the jitter in make_phantom
(`--radius-taper-mm 0`: off). Graph steps ≥ 15 % at r ≥ 2 mm: 5 → 0 (degree-2 version).
Independent review: the end radius was a median over max(2 mm, r) — ≈ r/2 from the node, and on short edges inside the
other end's ramp — so the factor applied at the node reversed or worsened 30 / 37 joints of export_full19 (abdominal
aorta 8.2 | 9.0 → 8.9 | 8.3 mm; an 11.3 | 11.3 mm pulmonary joint → 10.9 | 11.6), and a second pass was not a no-op;
the report keyed joints by name pair (189 shown of 360). Fixed: end radius = intercept at the node of a linear fit over
min(max(2 mm, r), 0.25 L); report keyed by node. Now 317 joints on export_full19, 373 on factory_test19: 0 reversed,
0 worse, a second pass changes nothing; factors 0.42–2.40, both extremes on a 0.15 / 0.85 mm intercostal-vein
two-edge cycle (atlas artefact).
The Murray floor is one value per edge, so it re-creates the steps after any taper (export_full19's 317 joints > 5 %
are post-floor; the thoracic aorta | arch 11.0 | 9.3 mm is the velocity floor raising one edge): murray_floor.py now
tapers after flooring (`--taper-mm 15`, default; 0: off; vessel-variant calibres untouched), iterated with the network
as before; make_phantom `--radius-taper-mm 0` switches both tapers off. On export_full20 + two floor / network rounds:
degree-2 node steps ≥ 10 % at r ≥ 2 mm 1 → 0, 270 joints tapered, network residual 3e-16, Kirchhoff 3e-11.

**Atlas gaps (open):** no intrahepatic portal branches in the Z-Anatomy export (only "Hepatic portal vein", 52 mm,
r 6.2 mm): the portal tree ends in a blunt cap at the porta hepatis (the liver sink).

**Next:** `jobs/phantom/chain_full20.sh` — export_full20 (taper) → factory_test20/ph0001 (rod, per-edge below-FOV
tether, taper after jitter) → audit, flow check, kinematics, renders → 0.5 mm all-valve refined LBM, compared with
lbm_full19_05ra against the full18 run-to-run spread.
