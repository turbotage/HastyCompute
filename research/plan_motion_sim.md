# Plan: simulate patient motion and the navigator ADC (for `simphantom`, hand-off to the implementing model)

Context: `research/motion_navigators.md` (why, numbers). User decisions 2026-11-15:
- Treat motion as **non-rigid at least in parts**: the head is nearly rigid, the neck / shoulders never are.
- Dropping a couple of seconds around a move is fine. Consequence (accepted): frequent minor movement is not handled by this
  scheme (moves closer together than ≈ 10 s leave too little stationary data; that regime needs motion-resolved bins like the
  respiratory ones, a different method).
- **Do not implement now.** This file is the specification; another model does the work.

Nothing below is implemented. Line references are to `rustlib/crates/hasty-bench/src/bin/simphantom.rs` at commit cc1d6ef + working tree
(search by name, lines drift).

## 0. Deliverables

1. `simphantom --pose <pose.json>`: an object-attached deformation Φ(X, t) applied on top of cardiac and respiratory motion,
   with the physics of §3.
2. `simphantom --nav`: simulate the encode-window and tail ADC samples of every TR (flow) or the full train (B0), written as separate
   files (§2), so existing `data_clean.npy`, the loader and every recon keep working untouched.
3. Truth outputs for scoring (§4).
4. A generator for pose schedules and deformation bases (`tools/phantom/pose_motion.py`, §1).
5. Loader support `SimData::nav()` in hasty-recon (§5) and tests (§6).

## 1. Pose model and schedule

Φ(X, t) maps a reference-pose (atlas) point to the pose at time t; the existing motion is applied after it, so the order is
X → Φ(X, t) → cardiac u_c, respiratory r S (both defined in the reference pose; evaluate their fields at X, push forward with
the Jacobian of Φ) → scanner-frame effects. Write it as a composition, document the order in the meta.

Parametrisation (all piecewise-constant poses joined by smooth transitions):
- **Pose clusters** p = 0..P−1, each a deformation Φ_p; schedule = list of `{t_start, t_end}` transitions (≥ 1 s, smoothstep or
  minimum-jerk in the interpolation parameter) between clusters, plus optional return to an earlier cluster (patient moves and comes back).
  Φ(t) is the interpolation of the two clusters' *stationary velocity fields* (exp of the lerped velocity field via scaling and
  squaring), so it stays a diffeomorphism and a transition is a smooth path. Never lerp displacements for rotations.
- A cluster's field v_p(X) is built from:
  1. a rigid part about a centre c (head: the C1 level): rotation vector ω_p and translation t_p;
  2. an **articulated / skinned part** with smooth weights w_j(X) from the anatomy labels (head, neck, upper torso, arms, shoulders):
     v_p(X) = Σ_j w_j(X) (ω_j × (X − c_j) + t_j), log-Euclidean blending (no linear blend of rotation matrices). The neck gets the
     transition weights between head and torso, which is what makes it non-rigid. Weight maps on the 4 mm grid used by `resp_motion.hv`;
  3. an optional **free-form part**: low-order cubic B-spline lattice (spacing 30–60 mm), amplitude set per cluster, to represent soft-tissue
     non-rigidity the skinning does not (reuse `rustlib/tools/phantom/diffeo.py` / `tissue_variation.py` for smooth diffeos).
- Presets in the generator: `head_nod` (head rotates 1–6° about LR through the neck, torso still), `head_turn`, `head_shift` (5 mm),
  `shoulder_shrug`, `torso_shift_5mm`, `arm_move`, plus a random generator with a seed. Every preset reports the displacement
  statistics (mean / max |u| in mm, max rotation, min det J) in the JSON. Require min det J > 0.2.
- Phantom note: ph0001 (factory_arms) is a thorax phantom with no head; the head presets need the intracranial phantom (planned).
  Until then use `torso_shift`, `arm_move`, `shoulder` clusters. The code must not assume a head exists.

File format: `pose.json` {version, reference "atlas", clusters: [{name, rigid:{centre_m, rotvec_rad, t_m}, skin:[{label, weight_map, rotvec, t, centre}], ffd:{spacing_m, coeff_file}}], schedule:[{from, to, t0, t1}], seed}. Fields stored as `.hv` like `resp_motion.hv` (`[k][3][z][y][x]`, 4 mm, origin_m, voxel_m) so the Python / Rust readers exist.

## 2. Navigator ADC output

Current writes only the readout window (+ the v2 rewinder). Add the ADC during the encode block and the tail:
- Flow (`FlowSchedule`, `seqsched.rs`): per TR a second ADC segment [i_enc … i_ro) and a third [i_tail … n) (and, if the design has a rewinder
  outside `recon`, that is already in the readout ADC). Raster 10 µs, dwell 2 µs as now. Sample times relative to the TR start and to
  the RF; per-sample record is the existing 16-float `sample_record` (k = γM0, γM1, γM2/h terms, τ since excitation, B0-eddy
  phase). The encode k path is the *physical* encode gradient (no rotation); the tail is the per-encode FlowComp block (no rotation).
  Use `PlayedStream` for the played version exactly as for the readout (`--girf`), previous TRs included.
- B0 train (`EchoSchedule`): the ADC already runs through the whole train; nothing to add, only the pose.
- Files: `nav_data_clean.npy` complex64 [readout][n_nav][coil], `nav_samples.npy` [readout][n_nav][4] (k rad/m, τ),
  `nav_sched_nominal.npy` / `nav_sched_played.npy` [readout][n_nav][16], and `meta.sequence.nav` = {segments: [{part, raster_from, raster_to, adc_from, adc_to}], n_nav}.
  `n_nav` ≈ 760 + 780 per TR (+28 % of the readout volume). Same σ per sample as the readouts.
- Cost control for development: `--nav-stride n` simulates the navigator of every n-th TR only (always whole sets of 7 encodes,
  so the encode comparisons exist); default 1. With `--readout-stride` (box sims) the nav follows the same kept groups.
- Physics identical to the readout path (they are the same kernels: `prep` + `dsum`), including R2*, B0 field, fat six-peak, flow
  phase from per-sample M1, gradient nonlinearity, coil maps at the moved position. The encode samples must carry M1 of the encode
  (flow phase differs between the 7 encodes; that is real and the navigator analysis has to cope).
- Check on the first implementation: a no-motion, no-noise sim must reproduce `nav_crlb.py` / `nav_motion_sim.py`'s coil-free model
  (rho-sum over the truth grid) in magnitude at the encode k-lines for a static phantom to < 1 % (blood and fat off).

## 3. Physics of a move in the simulator (what must differ from the recon's assumptions, no inverse crime)

Per bin and source, after the cardiac and respiratory displacement:
1. **Positions**: x = Φ(X + u_c(X) + r S(X + u_c), t). Tissue sources are Lagrangian points with fixed weight (mass), so no Jacobian weight;
   density changes by point density. Blood sources (Eulerian frame positions p): x = Φ(p + …) with the same map.
2. **Velocities**: v_total = J_Φ v_old + ∂_t Φ (the bulk velocity during a transition, mm/s; matters for phase). Blood moment
   statistics (mean μ, covariance Σ, the `beats.rs` moments) transform as J μ, J Σ Jᵀ, exactly as the cardiac path already does with F.
   Accelerations: J a + ∂_tt Φ (first order, ignore ∂_tt unless the transition is < 0.5 s).
3. **Scanner-frame effects use the moved position** (they already take x at that point): gradient nonlinearity, slab profile, coil
   sensitivities, the concomitant term, B0 shim field. Verify in the kernel `prep` that none of these were precomputed on reference positions.
4. **B0 map** (`field_ppm`): phase 1 attach it to the tissue (Lagrangian, as for respiration). Phase 2 (realism, recommended before any
   claim about B0 handling): recompute the susceptibility part per pose from the χ map with the dipole kernel for the rotated B0 direction
   in the object frame; the shim / scanner part stays in the scanner frame. A pure rigid shift of a Lagrangian field makes the
   recon's "field attached to tissue" assumption exact; label results of phase 1 as such.
5. **Slab and spin history** (thorax slab-selective along SI): sources whose slab weight changes pass through the slab edge; give a source
   that enters the slab the saturation-recovery amplitude factor 1 − (1 − E)·exp(−Δt_out / T1) (one term, T1 per tissue) for the TRs after
   it entered, and let sources that leave stop contributing through the slab weight. Phase 1 may ignore it; the meta must say so.
6. Cardiac and respiratory fields stay defined on the reference anatomy (the heart beats the same after the patient shifts).
7. Truth maps in the output are in the **reference pose** and stay that way (recon reconstructs in the reference pose).

## 4. Truth and scoring outputs

- `pose_truth.npy` [readout][..]: cluster id, transition fraction, for each readout the Φ parameters at its excitation time (rigid
  part: rotvec, t about the centre; skin: per-label rotvec / t; FFD: coefficient vector) and `dphi_<cluster>.npy`: the dense displacement of every cluster on the 4 mm (and a 2 mm) grid,
  forward (reference → pose) and inverse.
- `pose_clusters.json`: per cluster the readout index ranges (and how many are transition), displacement statistics.
- `truth_pose_<p>_{pd,r2s,vel}.npy`: truth maps of cluster p in the *scanner* frame (reference truth pushed forward by Φ_p with
  the velocity vectors rotated by J), for registration and image-quality scoring of per-segment reconstructions.
- `physio.npy` unchanged: the bellows and PPG do **not** see the bulk move (as in reality).

## 5. Loader (hasty-recon `data.rs`)

`SimData::nav()` returning an iterator / accessors: per readout the nav samples (data [n_nav][coil], k, τ, per-sample moments),
the segment table, and `SimData::pose_truth()`; reading is lazy (same chunking as the readout data, the nav volume is 28 % extra).
Do not touch `adc_navigator()` (that is the rewinder inside the readout ADC). Add a unit test next to the existing
`recon_adc` test: a tiny sim with a known nav block.

## 6. Tests and acceptance (cheap first)

1. Unit: the schedule evaluator: Φ(t) continuity, det J > 0, return-to-cluster gives identity composition, stationary clusters constant.
2. Kernel: rigid translation 10 mm, `--static`, no coils effects, single tissue source: the signal phase ramp equals e^{−2πi k·t} (per
   sample, 1e-4) for readout and nav samples; rotation by R: s(k) of a point cloud equals the sum at Rᵀk.
3. Non-rigid: a two-source test where one source moves rigidly and the other stays; the nav lines must show both.
4. End-to-end small: `--keep-box` + `--readout-stride` dev phantom, 3 clusters (10 mm shift at 40 %, return at 70 %), compare the encode-line
   nav rigid fit (`tools/seq/nav_motion_sim.py` logic, now on simulated data) with the truth pose to < 0.1 mm.
5. Performance: report the cost per bin before / after (`--plan` should print the nav sample counts and the cost estimate).
6. Reproducibility: the pose file, the seed and the sim version go in `meta.json`; a run without `--pose` is bit-identical to today.

## 7. Experiments the data enables (for the later reconstruction work)

Pre-register each; one factor at a time, dev scale first; controls: no-move sim, noise-free sim, oracle pose, ablation of physio.
1. Detection: delay and false alarms over move size (0.2–10 mm, 0.1–5°), move type (rigid / skinned neck / FFD), with respiration,
   cardiac and flow in the encode lines. Compare pose-only with pose + χ² residual and with bellows regression.
2. Non-rigid identifiability: the Fisher information of the navigators for the skinning / FFD parameters (not only the 6 rigid
   ones); which neck-bending modes the 3 axis-line navigators can see, and what must come from image-domain data. Expect: rigid + a few
   low-order modes from the nav, the rest from data consistency / registration of segment images.
3. Reconstruction at a move at 50 % of 10 min: ignore / discard the second half / oracle Φ / nav-initialised rigid Φ / consensus two-image
   with non-rigid registration / joint shared-image with refined Φ. Metrics: velocity and magnitude error in the great vessels and
   coronaries (dev: in the box), sharpness at the neck, with and without dropped transition TRs (0, 2, 4 s).
4. Frequency of moves: scans with 1, 2, 4, 8 moves in 10 min, to find where "drop ≈ 2 s around each move" stops paying (expected break at
   moves closer than ≈ 10 s).
5. B0 handling: phase 1 vs phase 2 field model (does recomputing the susceptibility field per pose matter at the neck?).

## 8. Order of work

1. pose generator + schedule evaluator + tests (Python / Rust, no GPU).
2. positions, velocities, blood moments (§3.1–3.3) behind `--pose`; rigid-only test 4.2.
3. nav ADC output for flow (§2) without pose; the static comparison check.
4. nav + pose together; end-to-end test 6.4.
5. loader (§5), truth outputs (§4).
6. B0 field per pose (§3.4), slab / spin history (§3.5).
Quiet-mode and GPU rules apply while testing: dev phantoms, one factor per run, outputs under `data/hasty/phantom/.../`, no periodic big writes.
