# Moving blood-pool surfaces in the flow viewer (stage 1, 2026-11-27)

Goal: per-frame heart / vessel surfaces for the flow viewer from **`vel` + `mag` alone** (no atlas, no labels), so it works on reconstructed
data. Atlas / LBM labels are only for scoring on phantoms.

**Server** `flow_bloodpool` (`rustlib/crates/hasty-server/src/flow_surface.rs`): hierarchical Otsu on the magnitude histogram (air | tissue | blood),
flow evidence (non-air voxel faster than q50 + 3 (q50 − q10) of the tissue-class speed is blood: dephased / saturated blood), components below
`min_voxels` dropped unless a resolved structure that sits where blood is in `persist` of the frames, small enclosed holes filled; then
`track_sequence` (same tracker as `meshseq_track`, `keep_first_all` keeps every piece of frame 0). Output `mesh_seq` in mm, voxel centres at
index × spacing = the flow viewer's world; `links[i] = 0` where the topology changes (a valve closing splits the pool).

**Viewer** (opt-in section "Surface (from the data)"): `flow/src/surface_layer.js` (headlight shader, cut plane in the fragment shader, vertices blended
between linked frames), `surface_ui.js` (compute / load stored / `&mesh=<id>`), `common/surface_seq.js` (pure frame / blend logic, `test_surface_seq.js`).
Result is kept in the bank as "<flow name> blood pool".

**Checked:** 3 Rust tests (pulsating ball with noise + speckle, noise-free control, fast-but-dark blood); `rustlib/tools/flow/synth_heart.py`
(chamber + outflow tube + closing valve disc, noise, speckle) end to end in headless Chrome: components 1 → 2 at the valve closing, one broken link
exactly there, cut-through shows the inside. Surface volume is 4–6 % below the true volume for small radii (binary indicator + blur + Taubin).

## Stage 2 (2026-11-27): check on a phantom, wall shell

**Check** (`rustlib/tools/flow/score_bloodpool.py`, ph0001 gated 2 mm, 20 frames, `viewer_flow.hv` against `truth4d.hv` blood / labels; the method never sees the truth;
magnitude is the noise-free SPGR image of the phantom with gadolinium contrast, not a reconstruction):
- A global magnitude Otsu failed (Dice 0.29): fat and skin are nearly as bright as blood in a gradient echo image. Fix: classify the magnitude only within `reach_mm` (6) of
  voxels that move fast in ≥ 2 frames, with a floor at the 10th percentile of the magnitude of the fast voxels; speed threshold from the non-air speed distribution
  (noise-free tissue has speed 0: at least 3 % of its 99.9th percentile). Result: **Dice 0.91, precision 0.90, recall 0.92, surface 1.07 mm mean / 2.3 mm p95 from the truth boundary
  (2 mm voxels)**. With recon-like noise (blood SNR 20 / 8): Dice 0.88 / 0.74 (the speed threshold rises with the velocity noise of dark tissue and slow blood is lost: weight the speed
  by the local magnitude next).
- Is the wall in the magnitude? Myocardium vs its 1-voxel neighbours, contrast in units of the myocardium's std: blood +3.3…+4.8 σ (endocardium clear), fat +2.3 σ (brighter), lung −1.6 σ
  (darker), muscle +0.4, liver +0.6, thymus +1.1, vessel wall +1.2 (no edge). Outside the blood about 80 % of the neighbours are fat or lung: the epicardial edge is there for most of the surface,
  not against muscle / liver.

**`flow_wall_shell`** (`flow_wall.rs`): per vertex of the blood-pool mesh, magnitude profile along the outward normal (1-ring averaged), change-point fit of "wall layer, then something else" (two-sample
t statistic = confidence), constraints: wall darker than 0.7 × blood beside it, brighter than the dark-class threshold (not air / lung), roughly constant (cv ≤ 0.5), ≥ 3 mm and not at the first position the fit can place an
edge; thickness / confidence diffused over the mesh as numerator / denominator, smoothed over time on linked frames; outer surface = vertex + T · normal, same triangles, scalars thickness_mm, confidence, support,
wall_rel, out_rel. Viewer: second layer, hides vertices below a support slider.
Phantom score (frames 0, 5, 10, 15; blood-pool vertices on the true myocardium): **78 % have a confident edge, outer surface median |d| 2.1 mm (one voxel), p90 ≈ 4.8 mm, mean signed −0.1 … −1.2 mm** against the boundary of
blood ∪ myocardium. Synthetic tests: fat side 8.21 mm and lung side 8.22 mm for a true 8 mm wall; a muscle quarter without an edge is low-confidence and filled from its neighbours; a 2 mm wall is not reported.

**Vessel cuffs (fixed 2026-11-27):** the shell also appeared around peripheral vessels wherever a fat / muscle edge sat behind a dark layer. Feature check on the phantom (confident edges, chord = local lumen
diameter found by walking inwards until the magnitude drops below halfway between blood and wall): chord median 28–33 mm on the myocardium against 16–19 mm elsewhere (thickness and confidence do not separate them; wall_rel does
partly, 0.32 vs 0.55, but is contrast dependent). Fix: `chord_mm` per vertex, averaged over the mesh neighbourhood (40 diffusion steps), `support` fades to 0 below `min_chord_mm` (25, fade from 20). Result (frames 0, 5, 10, 15): the viewer
shows 56–62 % of the vertices on the myocardium (was 73–83 %) and 4.7–6.4 % of the vertices more than 8 mm from it (was about 19 %); the remaining ones sit on large vessels (aorta, venae cavae, pulmonary trunk) whose
lumen is chamber-sized and whose wall is invisible. Tests: a tube of 6 mm radius with a clear edge gets no shell, and does with `min_chord_mm: 1`.

**Noise (2026-11-27):** phase noise is ∝ 1/magnitude, so a raw speed threshold is set by dark tissue. Speed is now weighted by min(1, mag / m_ref) (m_ref = p95 of the non-air magnitude; `norm_speed: false` = old
behaviour). Dice on the phantom, same scorer: noise-free 0.908 → 0.911, blood SNR 20 0.883 → 0.891 (speed threshold 0.106 → 0.072 m/s), SNR 8 0.739 → 0.809 (recall 0.61 → 0.71). What is left at SNR 8: the threshold is ≈ 3.8 σ of
the voxel-wise speed (0.18 m/s), so blood slower than that in a frame is lost; per-voxel testing cannot do better — next is to detect the flow region from spatially / temporally pooled velocity (3³ mean reduces σ by ~5) and keep the
per-frame test only for the surface position.

**Known problems:** (1) large vessels (chord ≥ 25 mm) still get a shell where a fat / lung edge sits behind a dark layer; a shape criterion (tube vs blob: chord in several directions, or distance to the centerline) would
split them from chambers; the shell also loses the small / systolic chambers (chord below 25 mm), which is the price of the threshold; (2) septum: the other chamber's blood behind the wall is an edge too, thickness there is septum thickness, not epicardium (the score's reference R includes both chambers, so septal vertices look too thin);
(3) at 2 mm voxels a wall of 1–2 voxels is at the resolution limit; (4) one-voxel error is the sampling limit, a soft indicator and sub-voxel profile fits could halve it.

**Not yet:** soft (non-binary) indicator for sub-voxel accuracy, faces stored once for coherent runs, rayon in the tracker, caps on the cut, cut applied to the MIP / slices, scalar colouring (speed, WSS) on the surface,
scores on 1 mm data and on real reconstructions, speed evidence weighted by magnitude (noise), valve plane / jet surface.
