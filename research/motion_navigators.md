# Navigators from the encode / tail ADC: detecting patient motion and using it in the reconstruction (2026-11-15)

Question (user): the ADC runs during the velocity encode and the spoiler / return to k = 0 as well as during the readout.
Can that data detect that the patient moved, say how, and be used so a 10-minute scan with a move in the middle becomes
two 5-minute scans that are reconstructed together (gradient + denoiser steps on both, joined by a deformation) instead of
throwing data away?

Short answer: yes, and the thermal noise is not the limit. Numbers below come from `tools/seq/nav_crlb.py`,
`nav_dc_crlb.py`, `nav_motion_sim.py` (outputs in `data/hasty/seq/studies/nav_crlb/`). The limits are model error
(respiration, cardiac, flow phase, coil / field changes), the spin history after a move, and non-rigidity of a real
patient. Recommended structure: **one shared reference-pose image, a per-pose-cluster rigid operator in k-space (rotate
the trajectory, add a phase ramp, warp the coil maps), poses initialised by the navigators and refined by data
consistency, a non-rigid residual only if needed.** The "two images + a matching deformation" idea is the right
*estimation* stage; as the final model a shared image uses all data in one fit.

## 1. What navigator data exists (flow_thorax_v2, TR 14.1 ms; b0_multiecho)

| part | duration | k path | repeats | use |
|---|---|---|---|---|
| encode (7 encodes) | 1.52 ms, ≈ 760 samples at 2 µs | 3D out-and-back curve per encode, ≤ 1.32 kmax per axis (100 of 153 raster points inside the 2 mm Nyquist box), returns to k = 0 at TE | **same physical curve every TR for a given encode** (encodes do not rotate); one set of 7 per 98.7 ms | projection / line navigator, image-free differences |
| readout | 9.5 ms | rotating centre-out yarn | never repeats | imaging data; low-k part is self-navigation |
| rewinder (v2) | 1.3 ms | M1-nulled return to k = 0 at ≈ 12.4 ms | rotation dependent | late DC sample |
| tail (FlowComp block) | 1.56 ms, ≈ 780 samples | constant physical path 0 → spoil on z (4 kmax) with ≤ 1.28 kmax transverse | **identical every TR for a given encode** | second line navigator; samples above kmax unused here |
| B0 train (24 lobes, TR 27.75 ms) | whole TR (duty 0.93) | 25 crossings of k = 0 per TR, lobes along diameters that change per pair | directions differ across TRs | per-coil DC at 900 /s, short lines; no velocity encode |

Facts that matter:
- The encode and tail curves repeat exactly, so **differences between TRs of the same encode contain only motion, flow,
  physiology, B0 drift and the known RF-spoiling phase** (eddy / GIRF errors of the lobes are identical and cancel).
- Encode M1 differs between the 7 encodes (flow phase), so compare each encode with its own history, never across encodes.
- The k = 0 samples (TE, the rewinder end) are one complex value per coil: FIDnav-type data (Kober 2011; Wallace 2019).
- The simulator currently writes only the readout window plus the rewinder (`recon_adc`); encode and tail ADC are **not
  simulated yet**, so nothing below has been run on simulated flow data (see §6).

## 2. What rigid motion does to the data

Object ρ(x) → ρ(R⁻¹(x − t)) about a centre c: with the coil maps fixed in the scanner,
s_c(k) = Σ S_c(x) ρ(R⁻¹(x−t)) e^{−2πi k·x}. Substituting x = Ty gives the form used as an operator

  s_c(k) = e^{−2πi k·t} · Σ_y [S_c(Ty) ρ(y)] e^{−2πi (Rᵀk)·y}.

So a pose change = rotate the sample locations, multiply a phase ramp, and use coil maps resampled into the object frame.
No image resampling. Without coils the translation part is image-free: s'(k) = e^{−2πik·t} s(k) at the *same* sampled k,
and rotations need s at rotated k (hence a reference image). First-order pose derivatives are cheap:
∂s/∂t_a = −2πi k_a s(k), ∂s/∂r_a = 2πi (e_a × k)·S(k) with S_j = FT of x_j ρ (3 extra NUFFTs).

## 3. How precisely, from noise alone (`nav_crlb.py`, `nav_dc_crlb.py`)

Assumptions: phantom ph0001 on the 2 mm grid, image SNR 20 (2 mm voxel, 10 min, 2·10⁸ samples ⇒ σ = 5.5·10³ per
sample, 90th percentile tissue 7.7), coil-free unit-sensitivity model (conservative for the lines), samples with any
|k_i| > 250 /m dropped (lower bound), per-TR nuisance: global phase, amplitude, linear phase in time (B0 drift) profiled out.
Cramér–Rao standard deviation, rotation given as displacement at 100 mm lever (σ ∝ 1/SNR):

| data | 0.1 s (one set of 7) | 1 s | 10 s |
|---|---|---|---|
| encode lines, translation | 8.5 µm | 2.7 µm | 0.9 µm |
| encode lines, rotation @100 mm | 7–21 µm | 2–6 µm | 0.7–2 µm |
| tail line, translation | 24–34 µm | 8–11 µm | 2.4–3.4 µm |
| readout (6 stored rotations, average) | 6–8 µm | 2 µm | 0.6–0.8 µm |
| per-coil DC only, flow (1 per TR) | 40–70 µm | 13–21 µm | 4–7 µm |
| per-coil DC, B0 train (25 per TR) | 12–19 µm | 4–6 µm | 1–2 µm |

Reading: the thermal floor is two orders of magnitude below a voxel for every data class and every window ≥ 0.1 s. The
line data (encode, tail) are about as informative per second as the whole readout, and they need no image. The DC
numbers assume *known* coil maps (smap error will dominate them: calibrate or use as a model-based term).

## 4. What limits detection in practice (`nav_motion_sim.py`, SNR 20, phantom respiration field)

Simulated: ρ_r(x) = ρ(x − r u(x)) with the phantom's respiratory field (SI 13.3 mm, AP 7.9 mm), plus optional bulk
rigid moves; the 7 encode curves fitted with a rigid pose (+ complex gain per encode), coarse to fine in |k|.
χ² = residual energy / noise energy per sample (1 = the rigid model explains the data).

| case | fitted t (mm) [x, y(AP), z(SI)] | fitted rot (deg) [x, y, z] | χ² rigid | χ² no motion |
|---|---|---|---|---|
| r = 0 | 0.03, 0.05, 0.03 | −0.01, 0, 0 | 0.98 | 0.99 |
| r = 0.25 | 0, −0.71, 0 | −0.03, 0.02, 0.07 | 2.2 | 3.3 |
| r = 0.5 | 0, −1.39, 0.26 | −0.26, 0.04, 0.13 | 6.1 | 10.6 |
| r = 1 (end-inspiration) | 0.05, −2.79, 0.81 | −0.60, 0.12, 0.22 | 21 | 40 |
| bulk 10 mm RL | 9.98, 0.01, −0.01 | 0.02, −0.02, 0 | 1.01 | 182 |
| bulk 5 mm AP + 2° about x | 0, 5.28, 0.39 | 1.50, −0.14, 0.19 | 3.2 | 66 |
| r = 0.5 + bulk 5 mm AP + 2° | 0, 3.86, 0.83 | 1.17, −0.08, 0.31 | 6.7 | 43 |

- A 10 mm shift is recovered to 0.02 mm with χ² = 1.01. Poses add linearly (r = 0.5 + bulk ≈ sum of the two rows).
- Respiration (13 mm SI organ motion) appears as only ≈ 3 mm rigid-equivalent (the lines mostly see the body surface,
  which breathes little) and is **linear in r**: a clean 1-D curve in pose space, plus a χ² excess from the non-rigid
  part. Bulk motion leaves that curve. A detector = pose (and χ²) minus the bellows-driven prediction (r, ṙ for hysteresis).
- The bulk rotation came back as 1.5° for 2° (exact-model data recover 2.00°, `--diag`). Likely cause (not verified):
  the simulated rotation of the cropped 228 mm slab moves tissue across the grid edge, which a pure-rotation model cannot
  represent. The same effect exists in the real scan through the slab-selective excitation (§5).
- A bulk motion exactly along the respiration direction (SI, correlated with breathing) is the hard case for the pose
  alone; the bellows (which do not see a table / body shift) and the χ² pattern separate them.
- Not simulated: cardiac motion (heart is a small share of the line signal; check with `cardiac_motion.hv`), flow phase
  in the encode lines, B0 drift, coil-sensitivity changes. These set the real threshold, probably 0.1–0.5 mm, to measure.

## 5. Physics that a move changes (must be in the operator)

- Scanner-frame quantities stay where they are: GIRF-played trajectory, concomitant fields, eddy / B0 terms, shim field,
  gradient nonlinearity, coil maps, slab profile. Evaluate them at scanner coordinates T y; object-attached ones (ρ, R2*,
  susceptibility B0 part, velocity vectors) move.
- Velocity is a vector field: v_scanner = R v_ref. The encode axes are scanner axes, so the velocity-parametrised operator
  needs R in the encode term; the flow viewer and denoiser work in the reference pose.
- B0: the object-attached part of ΔB0 rotates with a rotation about axes ⟂ B0, but its *value* changes (susceptibility
  field ∝ orientation to B0). Small for the head at a few degrees; thorax / non-rigid moves need a low-order re-fit.
- Slab-selective excitation (thorax): a move along SI changes which tissue is excited, so a pure warp is wrong at the slab
  faces. Put the slab profile in the operator as a scanner-frame weight.
- Spin history: after a move the steady state is disturbed for ≈ 1–3 T1(blood) ≈ 1 s; give the transition TRs weight 0
  (the navigators mark them).
- Coil maps: resample S_c∘T per pose cluster (32 coils, cheap on GPU), or re-run ESPIRiT per cluster.

## 6. Plan

Detection (stage 0): per set of 7 TRs build a feature vector from the encode / tail / DC data (each encode against its own
rolling reference, RF-spoil phase removed, per-TR global phase / amplitude / linear phase profiled out), regress the
bellows and PPG terms, run a change-point detector (CUSUM / PELT / BOCPD) on the residual and on the model-based pose.
Output: pose cluster per rotation set (clusters in pose space, not time: a patient who moves and returns merges back),
transition TRs flagged.

Pose estimation (stage 1): model-based. Reference nav predicted from the first cluster's own reconstruction (image + coil
maps through the encode-window operator, magnitude / static tissue; blood and flow phase excluded or modelled), 6-DOF fit
with a per-encode complex gain, coarse-to-fine in |k| (translation phase wraps otherwise: the simulation needed it).
Uncertainty from the Fisher matrix. Bootstrapping: the first recon needs a reference, so use the longest stationary cluster.

Reconstruction (stage 2, the user's idea made precise):
- (A) Shared-image joint model, known W_s: x ← x − μ Σ_s W_sᵀ A_sᴴ(A_s W_s x − y_s), denoiser on x (reference pose). For
  rigid W_s the operator is the k-space rotation + phase + warped maps of §2; the normal operator needs one kernel set per
  (pose cluster × frame), built from that cluster's rotations (total data volume unchanged).
- (B) Consensus: per segment s, gradient + denoiser steps on x_s, then x̄ = argmin Σ_s ‖x_s − W_s x̄‖²_{wₛ} and x_s ← W_s x̄
  (mixed with x_s), with W_s re-registered from the denoised x_s every few outer iterations. This is the user's
  two-image scheme and the estimation vehicle for a non-rigid W_s; it converges to (A) when W_s is right.
- Pose refinement inside (A): Gauss–Newton on the 6 pose parameters per cluster against the full data (derivatives from
  §2), every few PnP iterations; cheap relative to an image update.
- Non-rigid residual (stage 3, only if χ² stays high): W_s = rigid ∘ B-spline diffeo, with the machinery already in
  `hasty-recon` (`moco.rs` warp_pull / adjoint, `motion.rs` B-spline field + Jacobian check); bin × cluster compose.
- Short clusters (tens of seconds) still help in (A) because they share the image; in (B) they would be too undersampled
  to register alone. This is the main reason to prefer (A) as the final model.

Experiments (pre-registered, one factor each, dev scale, with controls per the measure-before-claiming rule):
1. Simulator: `simphantom --pose-schedule` (rigid pose(t), 2 s smooth transition, applied to tissue and blood sources,
   velocities rotated, coils fixed) **and write the encode and tail ADC** (needed for everything here).
2. Detection: ROC over move size (0.2–10 mm, 0.1–5°) and time; metric = detection delay and false alarms in the
   no-move scan (with respiration, cardiac, flow).
3. Pose accuracy vs truth; noise-free and inverse-crime controls (pose from navigators computed with the true image vs
   the first-cluster reconstruction).
4. Reconstruction at a move at 50 % of 10 min (velocity RMSE in vessels, magnitude sharpness): ignore the move / discard
   the second half / oracle pose / nav pose (A) / consensus (B) / (A) + refinement.
5. Cardiac and flow-phase confounds in the encode lines (the `cardiac_motion.hv` field, ph0001 beats).

## 7. Sequence-side notes (no change needed now)

- Keep the encode, tail and rewinder ADC (user already wants it). Cost is data volume only (+≈ 28 % samples).
- The 7 encode curves all drive three axes at once, so each curve is a 3D line, not an axis projection; the Fisher
  matrices above show every encode contributes to all six pose parameters (the per-axis Fisher of the 7 encodes differs by ≈ 4×).
- A readout-only self-navigation (low-k sphere of the yarn) is slow: an 8 mm Nyquist image needs ≈ 4900 spokes ≈ 70 s of
  readouts (my estimate from radial sampling counts, not run). The encode lines give pose every 0.1 s; use the readout
  data only inside the data-consistency refinement.
- GE: the encode / tail ADC events are normal ADC events in the pge2 sequence; the per-ADC sample cap splits the long
  windows, as for the readout. The samples need their own k-path (nominal + played GIRF) in the operator.

## 8. Literature to check before relying on it (from memory, not looked up)

FID navigators (Kober et al., MRM 2011; Wallace et al., MRM 2019); orbital / spherical / cloverleaf navigators (Fu 1995;
Welch 2002; van der Kouwe 2006); PROPELLER (Pipe 1999); self-navigated 3D radial and PC-VIPR respiratory gating;
joint motion + image estimation with parallel imaging (Batchelor 2005 matrix formulation; SAMER / TAMER, Cordero-Grande
2016, Haskell 2019); motion-compensated 4D flow (Feng XD-GRASP, Rank 2017).

## 9. Decisions (user, 2026-11-15)

- Rigid and non-rigid both matter; treat the deformation as **non-rigid at least in parts**. The head is nearly rigid, the neck
  never is. So the model is: nav gives detection plus a low-parameter pose (rigid + a few articulated / skinned modes); the rest of
  Φ_s comes from data consistency and registration of the segment images (stage 3 is not optional for head + neck; the consensus
  scheme (B) of §6 is the estimation vehicle). Identifiability of the non-rigid modes from the 3 line navigators is an open
  experiment (plan_motion_sim.md §7.2).
- Dropping a couple of seconds around a move is accepted. Limitation accepted with it: frequent minor movement (moves closer than
  ≈ 10 s) is out of scope for this scheme; it needs motion-resolved bins.
- `simphantom` is **not** extended now: specification for another model in `research/plan_motion_sim.md`.
