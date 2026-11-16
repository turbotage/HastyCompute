# Geometry viewer and mesh nodes (built 2026-11-14)

**Mesh value** (dict, `rustlib/crates/hasty-server/src/mesh.rs`): `kind "mesh"`, `vertices` f32 [N,3] mm, `faces` u32 [M,3] (outward CCW), `normals`,
optional `scalars` {name: f32 [N], NaN = no data}, `space` (`ras` when built through a NIfTI affine, `voxel_mm`, `voxel`), `affine`, `spacing_mm`.
Line sets (`kind "lines"`, flow centerlines) are shown too.

**Server commands (CPU):** `mesh_from_mask` (label or threshold, Kuhn marching tetrahedra on the bounding box, box blur, keep largest component,
Taubin smoothing, affine → RAS), `mesh_smooth`, `mesh_largest`, `mesh_info` (area, signed volume, open edges, components, bbox, scalar stats),
`mesh_sample` (volume → per-vertex scalar, trilinear). Tests: sphere area / volume within 8 %, closed + outward, components, affine inverse.

**Viewer** `plotting_website/mesh/` (`?id=a&id=b&scalar=name`): several objects, per-object colour / opacity / wire / flat, colour by scalar
(range editable, colormaps shared with the flow viewer), clip plane, point picking with distances, bbox + axes, view presets, PNG export.

**Node editor, category Geometry:** Mesh From Mask, Mesh Smooth, Mesh Largest Part, Mesh Sample, Mesh Info, Mesh Viewer (4 inputs), Mesh Load
(OBJ / STL / PLY parsed in the browser, `common/meshio.js`, tested), Mesh Save (OBJ / STL / PLY download).

## Second round (2026-11-14)

- **Server (`mesh_ops.rs`):** `mesh_decimate` (quadric edge collapse; closed stays closed / manifold, open borders kept; sphere at 20 %: area within 5 %,
  volume within 6 %), `mesh_cut` (plane clip with triangle splitting, scalars interpolated, optional cap by fans: fine for vessel cross-sections, not for
  sections with holes), `centerline_from_mask` (TEASAR: exact EDT, penalised Dijkstra, branch removal balls; lines value with `radius`; `coordinates: voxel`
  gives the flow viewer's centerline convention), `meshseq_from_mask` (one surface per frame of a 4-D mask, `kind mesh_seq`).
- **Viewer:** mesh sequences play (slider, speed), line `radius` colours, **Edit** section (erase around the last pick, keep the picked piece, undo, save a copy
  to the bank), flow values with `centerlines` open.
- **Nodes:** Mesh Decimate, Mesh Cut, Centerline, Mesh Sequence; Mesh Load got file units (mm / cm / m / um); Mesh Viewer takes meshes, lines and sequences.
- **Checked end to end** (`?geomtest=1`, headless, CPU): decimate 21376 → 6412 triangles, capped cut closed, centerline 2 branches in RAS, 4-frame sequence,
  Mesh Load as a browser step (cm → mm), Mesh Save OBJ / STL / PLY round trip, flow value with centerlines in the viewer, edit tools.

## Third round (2026-11-14)

- **Hole filling and caps** (`mesh_fill.rs`): boundary loops are projected to a plane, nested loops become holes (bridged), ear clipping triangulates; no
  vertices are added. `mesh_fill_holes` (loops up to `max_edges`, default 80) and `mesh_cut` caps now handle annular sections (a cavity gives a ring cap).
  Fixed on the way: a loop-size guard that stopped ear clipping early and left 18 open edges on a 474-edge cap.
- **Booleans** (`mesh_boolean`, `mesh_voxelize`): both meshes are voxelised on a shared grid (scanline parity), combined, and re-meshed. Resolution-limited
  (about one voxel, default longest side / 160), not exact; tests use the identities |A∪B| + |A∩B| = |A| + |B| and |A∖B| + |A∩B| = |A| (3 %).
- **Coherent moving walls** (`meshseq_track`): frame 0 is the iso-surface, later frames move the same vertices onto each frame's 0.5 level (Newton steps
  along the blurred indicator gradient, at most one voxel each, plus Laplacian relaxation): same triangles in every frame, vertices keep their identity, and the
  viewer blends positions between frames for smooth playback. Sphere test: mean radius within 0.8 mm of the true radius in every frame.
- **Centerline tree:** `arclength` (mm from the root along the tree), `branch` id per point, and `branch_info` per branch (parent, junction point, length, mean /
  min / max / tip radius). The viewer plots diameter against arc length (one curve per branch), picks points on lines and marks them in the plot. End points
  of a branch read small radii (the voxel next to the wall), so tip radii are not diameters.
- **Viewer sculpting:** push out / pull in / smooth around the last pick (Gaussian falloff, radius in mm), with undo and save copy.
- **Timing** (release, 914k-triangle sphere): mask → mesh 1.0 s, decimate to 10 % 4.1 s, capped cut 0.1 s, voxelize 0.1 s.
- **Checked end to end** (`?geomtest=2`, headless, CPU): tree − ball boolean (3 closed parts), open + fill (closed, 1 part), voxelise → re-mesh (3.00 ml vs 3.15 ml
  at 0.8 mm), coherent sequence, centerline branch table; in the viewer: line picking + profile, coherent playback ("+0.72" between frames), sculpt + undo + save.

## Fourth round (2026-11-14): centerline fix and the remaining items

- **Centerline bug (reported from a screenshot): the two daughters were joined by a corner-cutting U instead of leaving the trunk at the bifurcation.** Two causes:
  the root was the tip farthest from the thickest point, i.e. a daughter tip, so the "main" path ran daughter → daughter across the crotch; and the path penalty used
  the *global* maximum radius, so inside a wide trunk the paths to two daughters could run side by side and separate far below the bifurcation. Fixed by (1) rooting
  at the tip of the thickest branch (a first pass at the thickest point finds the trunk), (2) a penalty normalised by the *local* inscribed radius (distance to the
  medial axis), (3) centring each junction on the largest inscribed ball of the parent path near where the daughter attached. The Y-tree test now asserts: root at
  the stem bottom, junction within 4 voxels of the true bifurcation, arm points within 1.8 voxels of the arm axes.
- **`centerline_measure`:** cross-section area and equivalent diameter at every point (plane ⟂ to the line, connected lumen cross-section on the mask), tortuosity
  per segment between junctions (core only: 1.5 radii in from each end), bifurcation angles from limb chords away from the junction (angle between the daughters,
  each daughter against the trunk axis). Y tree (arms 71° apart in mm): trunk equivalent diameter 10.2 mm (true 10), arm 7.0 mm (true 7), angle 65°.
- **Exact-ish booleans:** the voxel result is snapped onto the nearest input surface within 1.5 voxels (`refine`), so faces lie on the inputs and only the crease is
  voxel-limited (test: > 90 % of union vertices on an input surface).
- **Hole filling with interior vertices and fairing:** the ear-clipped patch is refined to the boundary edge length and its interior relaxed (bounded to avoid
  exploding on huge openings such as a whole cut face, where the plain triangulation is kept).
- **Large motion in tracking:** vertices are carried by the centroid shift, then snapped coarse-to-fine (blur + 4, + 2, + 0). A sphere moving 6–7 voxels per frame
  and changing radius: centroid within 0.8, radius within 1.0 voxel.
- **Viewer:** grab tool (drag the surface with the mouse, radius from the Edit section, undo), profile plots equivalent diameter when present, picking works on a
  playing sequence (bounding boxes follow the frames).

## Fifth round (2026-11-14): the remaining items

- **Boolean crease** (`mesh_boolean`, `refine`): the exact intersection curve of the two meshes (triangle–triangle crossings through a grid) is computed; result vertices
  about equally close to both surfaces are moved onto it, the rest onto the nearest input surface. Test: > 40 result vertices lie within 1 µm of the exact curve. The
  triangulation itself is still the voxel result's, so this is crease-accurate, not an exact CSG remesh.
- **Cotangent fairing** (`mesh_fill_holes`, `cotan`): cotangent-weighted relaxation of the refined patch (minimal-surface flow); its patch area is not larger than the
  uniform-weight one.
- **Topology changes in tracking** (`meshseq_track`): a split or merge (different number of connected pieces between frames) or a fit the fixed topology cannot reach
  (> 5 % of vertices off the 0.5 level) starts a new surface run. `links` (u8 [T−1]) tells which transitions keep vertex identity; the viewer blends only across linked
  transitions. Test: ball → two balls → grown → shrunk gives links [0, 1, 1], 1 → 2 components, identical triangles within the runs.
- **Scalar-preserving decimation** (`mesh_decimate`, `preserve_scalars`): vertices that are strict local extremes of a scalar within the top / bottom 1 % of its range are
  never merged away. Test: a narrow spike survives 5 % decimation, and is flattened without the option.
- **Flow-based measurements** (`mesh_flow.rs`):
  - `mesh_wss`: tangential velocity at depths d and 2d inside the wall (along the inward normal) gives τ = μ (4 v₁ − v₂) / (2 d); per-vertex `tawss`, `peak_wss`, `osi`.
    Poiseuille tube (R = 8 voxels, 1 mm): 0.784 Pa against 0.875 analytic (−10 %: the iso-surface is slightly inside the true wall and the first sample sits one voxel in).
  - `centerline_pwv`: flow waveforms at cross-sections along a branch (plane flow through the lumen), delays by circular cross-correlation (×16 upsampled), PWV from the
    slope of delay against arc length. Travelling wave with c = 3 m/s: 3.00 m/s, r² = 1.000. Lines now remember the affine / spacing they came from, so the editor
    needs no extra options.
  - Editor nodes: Mesh Wall Shear, Pulse Wave Velocity. End to end on the server: tube with a travelling wave → TAWSS 1.24 Pa (analytic 1.17; the end caps and the closed-ends
    of the mesh raise the mean), PWV 3.00 m/s.

## Sixth round (2026-11-14): pulse wave velocity in depth, and the rest of "not covered"

- **Pulse wave velocity** (`centerline_pwv`, rewritten):
  - Waveforms (through-plane flow or mean velocity) at N cross-sections of a branch, cyclic cubic (Catmull–Rom) interpolation ×16.
  - Five delay estimators: **foot** (the line through the 20 % and 80 % points of the upstroke meets the baseline; a tangent at one sample is unstable at coarse frame rates),
    **halfmax** (time to 50 % of the upstroke), **maxslope** (steepest upstroke, centroid of the part above 80 % of the maximum slope), **xcorr** (circular cross-correlation) and
    **phase** (amplitude-weighted phase of the first two harmonics). Crossing times are unwrapped around the steepest point so an upstroke that straddles the start of the cycle works.
  - PWV = 1 / slope of delay against arc length, Theil–Sen with a 5–95 % bootstrap interval over the cross-sections, robust r². The consensus is the median of the methods with r² > 0.8;
    a method more than 30 % from it gets a warning (it is ignored by the median); a total delay below a quarter of the frame interval also warns.
  - The output lines carry `pwv_waveforms`, `pwv_arclength`, `pwv_delay_ms`; the viewer's "Pulse waves" panel plots the waveforms and the delay line. `all: true` measures every branch.
  - Tests: sine wave c = 3 m/s: all five methods 3.00 (a landmark test checks that each landmark follows a shifted waveform to 0.2 ms); a systolic pulse with a dicrotic bump at 40 ms frames,
    c = 5 m/s (the whole delay is a third of a frame): consensus 5.00 (foot 4.47, halfmax 5.46, xcorr 4.90, phase 5.00, maxslope 23: flagged and ignored); with noise, 20 ms frames, c = 4:
    consensus 3.92. In the editor on the server (50 frames, noise, c = 4): 4.03 m/s [3.9–4.1], r² 0.999, all methods 4.00–4.23.
  - What it does not do: reflections / dispersion (a single wave speed per branch), and delays far below a frame interval are only as good as the interpolation (maxslope fails there).
- **Wall shear, third order** (`order 3`, default): three depths d, 2d, 3d, a quadratic through the tangential velocity, and the wall is where the fitted velocity vanishes (so a
  surface that sits a fraction of a voxel inside the true wall is corrected). Poiseuille tube: 0.872 Pa against 0.875 analytic (order 2: 0.784). Bifurcation with oblique
  Poiseuille legs: trunk 1.390 vs 1.400, arms 1.971 and 1.963 vs 2.000 Pa (all within 1.5 %).
- **Scalar-aware decimation** (`scalar_weight`): merging vertices with different scalars costs (Δs / range × mean edge × weight)²; a wavy scalar step keeps its place (mismatch with the
  nearest original vertex 4.8 % → 0.0 %).
- **Tracking through splits and merges:** a new surface run records, for every vertex, the nearest vertex of the previous frame (`parent`); the viewer morphs the new surface from
  those positions. Test: the left piece of a split ball descends from the left half of the ball (< 2 % wrong).
- **Centerline quality** (reported from a screenshot): the lines now (1) root at the trunk, (2) use a local-radius path cost, (3) centre junctions, (4) are re-centred on the centroid of the
  lumen cross-section at every point, (5) have the hook around a tip's cap cut and replaced by a straight continuation. A straight tube stays within 1.3 voxels of its axis up to both
  ends; Y tree: bifurcation 67° (arms 71.5° apart), straight segments tortuosity 1.00.

## Seventh round (2026-11-14): exact booleans, wall shear against the LBM, wave separation

- **Exact CSG** (`mesh_csg`, Mesh Boolean "method: exact", `mesh_csg.rs`): corefinement. Triangle pairs that cross give segments; each cut triangle is re-triangulated along them
  (planar graph of the triangle's edges, split at the nodes on them, plus the segments; faces by angle-sorted half-edges, holes from separate closed pieces of the curve, ear clipping with
  bridged holes); every patch of the refined surface (connected without crossing a crease edge) is classified inside / outside the other mesh once, by its biggest triangle and a ray
  parity test; the kept patches are welded. Two overlapping spheres (12k triangles each): all three operations watertight with outward windings; |A∪B| + |A∩B| = |A| + |B| and
  |A∖B| + |A∩B| = |A| to 1e-4; the crease vertices lie on both input surfaces to 1 µm; within 6 % of the voxel boolean. Nested and disjoint meshes exact to 1e-6. In the editor,
  tree minus ball: 2.42 ml in both, but 20 178 triangles (the originals, cut) against 94 156 for the voxel result. Limits: general position only (no coplanar overlapping faces, no
  vertex exactly on the other surface): otherwise it returns an error that says to use the voxel method; scalars are dropped. A first version classified single triangles and left 16
  open edges (a sliver next to the crease has its centroid within rounding of the other surface): classifying whole patches fixed it.
- **Wall shear against the LBM phantom's own wall shear** (`tools/flow/lbm_crop.py` + the ignored test `lbm_wall_shear_against_the_simulation`): a 56³ window (0.3 mm cells, 1393 tiles) of
  the coronary tree run `phantom/factory_arms/ph0001/lbm/run_cor/flow_bins.f32` (10 of its 50 bins; velocity from Σw·u / Σw, wall |τ| from Σ|τ_w| / Σw_wall), mesh from the lumen
  cells, `mesh_wss` compared cell by cell with the simulation's time-mean wall |τ| (4653 wall cells): order 3: mean 0.784 Pa against 0.767 (ratio 1.02, median cell ratio 1.07,
  Pearson r 0.74); order 2: ratio 0.70 (r 0.77). An independent solver's own wall treatment, so this is a real check of level and pattern, not of the algorithm against itself. The
  remaining scatter is the 0.3 mm resolution (three samples span 0.9 mm of a 3 mm vessel) and the time averaging over 10 of 50 bins.
- **Wave separation** (`centerline_pwv`, `separate`, default on): flow-only forward / backward decomposition. Per harmonic X_k(s_i) = F_k e^{−iθ} + B_k e^{+iθ}, θ = 2πk s_i / (T c),
  complex least squares over the cross-sections; output per-harmonic amplitudes and ratio, the forward and backward waveforms at the first cross-section, the reflection index
  (rms backward / (rms forward + rms backward)), the fit residual, and the arrays in the lines value (the viewer plots them). A 260 mm tube with a forward pulse and a 40 % backward
  pulse (three harmonics, c = 2 m/s): backward/forward 0.400 at every harmonic with c given; with c taken from the (reflection-biased) delay fit (1.88 m/s): reflection index 0.280
  against 0.286. It warns and skips harmonics when the cross-sections span too little of the wavelength (a 100 mm segment cannot separate waves of 1 Hz at 4 m/s), and, as 4D flow has no
  pressure, it is not the pressure–velocity wave intensity analysis.

**Still not built:** the pressure-based wave analysis (needs a pressure estimate), exact CSG for meshes with coplanar faces (a tolerance-based plane-merge step), wall shear on the
full-resolution LBM data (finer than the 0.3 mm output cells) and over all 50 bins, vertex scalars through the exact boolean (transfer from the parent triangle).
