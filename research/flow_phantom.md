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

**Next (P0, needs compute — paused while the machine must stay quiet):**
1. Vessel lumens into the label volume (capsule distance to the centrelines; artery / vein / pulmonary labels,
   vessel wall).
2. Womersley profiles per segment from the network harmonics → Eulerian truth v(x, t) at any grid (junction
   blending) → `flow4d` export → flow viewer (gates: Q(t) through planes = network; divergence ≈ 0 in straight
   segments).
3. Finer label grids (1 mm / 0.5 mm: GPU, or a CPU fast winding number with a tree + dipole far field).
4. Fine-vessel growth into organs from the atlas terminals.
5. Phantom layer in the viewer.
