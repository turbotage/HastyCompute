# Magnitude coupling across velocity encodes

2026-11-12. Question (user): within a frame, the encodes of a voxel should differ in phase only. Can a cheap pointwise
term that pulls the magnitudes of the encodes towards their mean (in the objective, or in the denoiser) be exploited?
How accurate is "equal magnitudes", in tissue and in vessels? How strong should the term be? What does it do to the
convergence? Scripts: `rustlib/tools/phantom/bias_truth.py --signal-frames` (truth),
`research/experiments/recon_diag/mag_var.py` (statistics). Numbers marked [measured] come from ph0001
(`sim_flow_10min`, 2 mm, 7 encodes, 20 cardiac frames); everything else is derived.

## 1. Why magnitudes differ between encodes

Signal of voxel q, frame f, encode e (first moment k_e = γ M1_e, rad per m/s):

  s_e = b · φ(k_e) + t,   φ(k) = E[e^{−i k·u}]   (characteristic function of the voxel's velocity distribution),

with b the moving (blood) magnetization and t the static part (tissue, wall, fat). The expectation runs over the
spins in the voxel, over the time window of the frame, and over the heartbeats pooled into the frame.

1. **Intravoxel velocity spread** (shear across the voxel, turbulence, the velocity change inside the frame window,
   beat-to-beat variation): for a Gaussian velocity distribution with covariance Σ,
   |φ(k_e)| = exp(−½ k_eᵀ Σ k_e). The magnitude ratio between encodes is exactly what generalised PC-MRI uses to
   measure the intravoxel velocity SD (IVSD) and the turbulent kinetic energy (Dyverfeldt et al. 2006; review in
   Dyverfeldt's thesis): **in blood the magnitude difference is signal, not noise**.
   - Our design has no zero-moment reference: every encode has |k_e|² = 5.2–7.4 (rad/(m/s))² with per-axis sign flips
     (k_0 = −1.571·(1,1,1); the others flip one axis to +1.571 or +0.524). So
     k_eᵀΣk_e − k_0ᵀΣk_0 involves both the variances (|k|² 5.2 vs 7.4) and the cross-covariances (sign flips):
     shear-induced velocity correlations make encodes with the same |k| differ.
   - Size: isotropic σ = 0.05 / 0.1 / 0.25 m/s gives |φ| between encodes differing by ≈ 0.3 / 1.1 / 6.9 % from the
     diagonal part alone. TKE 100 J/m³ (the upper end reported for healthy aortas) corresponds to σ ≈ 0.25 m/s per
     direction.
   - Laminar shear across a 2 mm voxel (Poiseuille, R = 10 mm, v_max = 1 m/s): Δv ≈ 2 v_max r Δx / R², σ = Δv/√12:
     0.06 m/s at mid-radius, 0.23 m/s next to the wall → ≈ 1.5 % and ≈ 20 % attenuation of k ≈ 3 rad/(m/s).
2. **Partial volume with static tissue**: |b φ(k_e) + t| interferes with the encode phase. For b ≈ t and phase
   differences of order 1 rad between encodes, the magnitudes differ by tens of per cent. This is the dominant effect at
   vessel walls, and it is unavoidable at 2 mm for coronaries (diameter 2–4 mm).
3. **Skewness** of the velocity distribution changes the phase, not the magnitude, to first order.
4. **Per-encode system effects** (eddy currents, GIRF, concomitant fields) act mainly on the phase; magnitude only through
   intravoxel phase gradients (small at 2 mm).
5. **Static tissue**: φ = 1 for every encode, so the magnitudes are equal (up to slow tissue motion, σ_v ≲ 0.05 m/s →
   < 0.3 %). This is where the constraint is essentially exact.
6. **Noise**: the magnitude of a noisy complex value is biased (Rician) at low SNR; a term acting on |x| sees this.

## 2. Measured on the sim truth

See §6.

## 3. What equal magnitudes can buy (estimation theory)

**Per voxel, nothing for the velocity.** For s = m e^{iθ} + n with circular Gaussian noise of variance σ², the Fisher
information in (m, θ) is diag(2/σ², 2m²/σ²): magnitude and phase are orthogonal. Knowing m (or that m is shared by the
encodes) does not change the Cramér–Rao bound of θ, hence not that of the velocity. In a well-determined voxel, a
magnitude term cannot improve the velocity precision.

**In the undersampled, coupled problem, it can.** The per-frame sets are strongly undersampled; errors are aliasing
patterns spread over many voxels and coupled through the data term. A shared magnitude removes, per voxel and frame,
E − 1 of the 2E real degrees of freedom (6 of 14). In the language of the PnP recovery theory
(`mathematics_of_paper/pnp_restricted`, Thm local): the model tangent space shrinks from T₀ to T₀′ ⊂ T₀, hence
- d′ < d (sample complexity and the √d noise term shrink),
- λ₋(T₀′) = min over a smaller set ≥ λ₋(T₀) (restricted curvature can only improve),
- Φ₀(T₀′) ≤ Φ₀(T₀) (coherence can only improve),
provided the truth satisfies the constraint. Otherwise the mismatch ε₀ grows by the part of the truth outside the
constrained model: the magnitude deviations of §1. The phase directions benefit only through the improved conditioning
(aliasing that is shared between magnitude and phase directions is cut).

**The bias of a hard constraint acts on magnitudes first.** The pointwise projection onto equal magnitudes (§4) keeps the
phases exactly. A mis-specified magnitude prior moves the velocity only indirectly: the corrected magnitudes leave a
data residual that the next gradient step spreads into the neighbourhood and into the phases. This second-order bias has
to be measured; it is largest where the truth violates the constraint most (walls, jets).

**Strength (MAP).** Model |x_e| = m̄ + δ_e with δ_e ~ N(0, (s_q m̄_q)²), s_q the relative magnitude spread of voxel q
(physics, §1–2). With whitened data and complex noise of variance σ² per sample, −log L = ‖Ex − y‖²/σ² = 2f/σ². MAP:

  min f(x) + (σ²/(4 s_q²)) Σ_q Σ_e (|x_{q,e}|/m̄_q − 1)²     (relative deviations; normalisation-equivariant).

Compared with the data curvature per voxel and set, 1/σ_img,q² (σ_img,q the per-frame image noise), the term dominates
when

  s_q · SNR_q < 1/√2,   SNR_q = m̄_q / σ_img,q.

Design rule: **couple the magnitudes where the physical spread is below the per-frame relative noise, and not where it
is above.** Static tissue (s ≈ 0): strong coupling, limited only by Rician bias at low SNR. Vessel interiors with laminar
flow (s ~ 1 %): strong coupling at per-frame SNR ≲ 70. Walls and disturbed flow (s ~ 10–50 %): weak or none. A weight
μ_q ∝ 1/(s_q² + 1/SNR_q²) interpolates; s_q can come from a vessel mask / distance map, or be estimated from the current
iterate (the TKE estimate itself: s ≈ ½|Δ(k_eᵀΣk_e)|).

## 4. Objective term or denoiser

**(a) Objective term.** R(x) = (μ_q/2) Σ_e (|x_e| − m̄)², m̄ = mean_e |x_e|.
- Gradient pointwise: ∂R/∂x̄_e = μ_q (|x_e| − m̄) x_e/|x_e| (the derivative through m̄ sums to zero). Cost: negligible.
- Not L-smooth: the Hessian of (|x| − c)² has the tangential eigenvalue 2(1 − c/|x|), negative and unbounded as
  |x| → 0. Use |x|_δ = (|x|² + δ²)^{1/2} with δ ≈ the image noise, or the polynomial variant
  (μ/4) Σ_e (|x_e|² − mean|x|²)² ≈ μ m̄² Σ (|x_e| − m̄)² (automatically SNR-weighted, smooth, locally Lipschitz).
- Nonconvex, semi-algebraic: gradient and proximal-gradient methods converge to critical points under the
  Kurdyka–Łojasiewicz property (Attouch–Bolte–Svaiter 2013); near the truth (|x| ≫ δ) it is locally convex in the
  magnitude-difference directions with curvature μ_q(1 − 1/E) and flat in the phase directions.
- **Conditioning.** Its curvature is pointwise in image space, i.e. flat in k-space: on the magnitude-difference subspace
  the per-set data curvature n(k) becomes n(k) + μ, so the condition number there drops from n_max/n_min to
  (n_max + μ)/(n_min + μ). It acts exactly where undersampled per-set data are weak (high k).
- **Clash with the circulant metric.** P ≈ N⁻¹ is large at high k. The step must satisfy α λ_max(P(N + μW)) ≤ 1, and
  λ_max(PμW) ≈ μ λ_max(P) ~ μ · 10³/n_max: a pointwise term stiffens the preconditioned problem by the metric's
  dynamic range. Either μ ≲ n_min (then it is weak exactly where it should help), or the metric must include the term
  (not circulant any more).

**(b) In the denoiser (proximal step).** prox_{αR} is closed form per voxel and frame: keep the phases,
  |u_e| = mean_e|z| + (|z_e| − mean_e|z|)/(1 + αμ_q).
- No step-size constraint, exact for any μ, cost negligible (one pass over the stack).
- Composed with the LLR prox (D = prox_mag ∘ D_LLR, or alternated), it is the forward–backward iteration of
  f + κR_LLR + R_mag in the Euclidean metric (convergent to critical points by KL for α < 1/L_f). In a non-scalar
  metric the fixed point is displaced (Thm `thm:displacement` of the PnP draft): its Jacobian is pointwise (keeps
  phases and the mean magnitude, shrinks magnitude differences by 1/(1 + αμ)), so it does not commute with a spatial
  circulant (leak), and its scalar part re-tunes μ (Remark `rem:strength`). The fixed-point-preserving acceleration
  (`prop:keepfp`) keeps the Euclidean limit.
- Low SNR: if one encode is dominated by noise, pulling its magnitude up to the mean keeps its noisy phase and amplifies
  it. Hence μ_q → 0 where SNR_q is low (the SNR-weighted rule of §3 does this).

**(c) Hard constraint = parametrisation** x_e = m e^{iθ_e}: the limit μ → ∞ of (b). The velocity model
(`research/velocity_model_sr.md`) restricts further, θ_e = θ₀ + k_e·v. The soft term is the continuation between the
complex model (μ = 0) and these.

**Recommendation.** (b): the prox in the denoiser chain, with an SNR- and spread-aware weight, Euclidean-consistent or
used with the fixed-point-preserving acceleration. (a) only with the polynomial form and a weak μ, or with P = I.

## 5. Literature

- Generalised PC-MRI / IVSD / TKE from the magnitude ratio: Dyverfeldt, Sigfridsson, Kvitting, Ebbers, MRM 56:850, 2006;
  Dyverfeldt et al. JMRI 28:655, 2008; Dyverfeldt et al. MRI 27:913, 2009 (limits); Petersson et al. MRM 2010
  (simulation); encoding design for mean and turbulent flow (PMC11349923, optimal sensitivity at Venc ≈ π σ).
- Shared magnitude across encodes in reconstruction: Tan et al. MRM 2017 (model-based real-time PC, ρ·e^{i v·V_k});
  ReVEAL (Rich et al. MRM 2016: conditional mixture prior coupling magnitude and phase across encodes depending on the
  presence of flow; ReVEAL4D enforces magnitude and phase similarity outside flow); neural fields for 2D cine PC
  (Arratia et al. 2025/26: common magnitude). Separate magnitude/phase regularisation and its nonconvexity: Zhao, Noll,
  Nielsen, Fessler, IEEE TMI 31:1713, 2012; NAPALM (Liu & Haldar).
- To our knowledge no published 4D flow method quantifies the bias of magnitude coupling inside vessels against the
  IVSD it removes [CHECK].

## 6. Measurements

### 6.1 Sim truth (per-frame ideal blood signal, `truth_pc_frames.npz`, 134 469 LBM-covered voxels × 7 × 20)

Relative magnitude spread s = std_e|C_e| / mean_e|C_e| per voxel-frame [measured]:

| group | median | p90 | p99 |
|---|---|---|---|
| interior (blood in every frame, depth ≥ 2), all frames | 0.57 % | 4.0 % | 18.8 % |
| speed < 0.2 m/s | 0.29 % | 1.4 % | 4.1 % |
| 0.2–0.5 m/s | 1.2 % | 4.1 % | 12 % |
| 0.5–1.0 m/s | 3.7 % | 12 % | 32 % |
| > 1.0 m/s | 4.3 % | 20 % | 61 % |
| wall, blood fraction 0.5–0.7, + static tissue (r_tb 0.47, phase 0) | 0.36 % | 4.8 % | 39 % |

- Per frame (interior, p90): systole (cardiac 1–6) 6–16 %, diastole (8–18) 1.3–4 %.
- Systematic encode pattern = dispersion: the three encodes with |k|² = 5.2 are on average 1.5 % brighter than the four
  with |k|² = 7.4 (mean ratios 1.006–1.012 vs 0.991–0.995).
- The truth includes intravoxel spread from the six moment-matched sub-sources per LBM cell, beat-to-beat variation over
  15 library beats, the frame window, spin history; not modelled here: tissue phase other than 0 (fat), coil-map
  variation inside a voxel.

### 6.2 Reconstruction error (same statistic where the truth spread is ≈ 0)

| reconstruction | static tissue median / p90 | interior lumen median / p90 |
|---|---|---|
| inverse crime, 2 mm, 8 it (`pyr_synth`) | 1.3 % / 3.4 % | 2.6 % / 5.5 % |
| real sim, 2 mm, ZMSHR1s κ 1, 20 it (`fix_2mm/llr_s`) | 13.7 % / 38.5 % | 15.4 % / 31.5 % |

The current reconstruction error across encodes is 10–30× the physical spread: by the rule s·SNR < 0.7 the coupling is
safe for ~90–97 % of blood voxel-frames (not for fast systolic jets and the worst wall voxels).

**Why the encodes are so inconsistent in the real reconstruction.** No rotation has all 7 encodes in one cardiac frame
(frame 42.6 ms, 7 encodes = 79 ms). Within a frame, encode e shares only 74 / 47 / 20 / 0 / 0 / 0 % of encode 0's
rotations (e = 1 … 6) [measured, readouts.npy]. The encodes of a frame have different undersampling, so different
aliasing, which does not cancel in the phase differences. PC-VIPR-style reconstructions bin each projection's encode
group together. Test: `recon_p2np --bin-group` (all encodes of a rotation in the frame of its middle readout).

### 6.3 Toy (`research/experiments/magnitude/mag_toy.py`)

2-D, 3 encodes of the ph0001 design along one axis, vessel with Poiseuille flow (v_max 0.8 m/s) and a turbulent quadrant
(σ_t 0.3 m/s), truth on a 4× finer grid (partial volume and shear dispersion physical), 700 shots per encode, variable
density k₀ = 4, 4 coils, joint Haar soft threshold across encodes, then the magnitude prox with γ; best λ per γ.
Velocity RMSE (m/s):

| γ | interior | jet | wall | tissue (truth 0) | magnitude error |
|---|---|---|---|---|---|
| shared shots, 0 | 0.0467 | 0.0635 | 0.1000 | 0.0358 | 0.102 |
| 0.1 | **0.0441** | 0.0632 | 0.1006 | 0.0291 | 0.099 |
| 1 | 0.0522 | 0.0720 | 0.1110 | **0.0260** | 0.142 |
| 100 | 0.0524 | 0.0742 | 0.1097 | 0.0280 | 0.143 |
| independent shots, 0 | 0.0464 | 0.0619 | 0.1063 | 0.0489 | 0.074 |
| 0.1 | **0.0448** | **0.0604** | **0.1018** | 0.0453 | **0.062** |
| 10 | 0.0574 | 0.0781 | 0.1146 | 0.0373 | 0.157 |

- Weak coupling (γ ≈ 0.1) helps everywhere a little: −3…−6 % in the vessel, −7…−19 % for the tissue velocity, −16 %
  magnitude error with independent shots.
- Strong coupling hurts, even in the interior where the truth spread is 0.06 %. Cause [measured]: at λ = 0.001, γ = 0
  converges (relative change 1e-14 after 20 000 it, interior RMSE 0.0471), γ = 100 stalls (relative change 5e-6 at 20 000
  it, RMSE drifting 0.0472 → 0.0489). A per-encode SNR weight and a gain clip (`--cap`) change nothing.
- Explanation: keeping the phase while pulling a magnitude below the mean up scales that encode by m̄/m_e > 1: the map
  is expansive in that encode's phase direction, and nothing contracts there in the data's null space. Convergence needs
  γ · max_e(m̄/m_e − 1) below the contraction margin of the rest of the iteration, which the truth's own spread and the
  noise set. Strong coupling is a projection onto a nonconvex set (a torus per voxel).
- Early iterates beat the fixed point (interior 0.041 at 500 it vs 0.047 converged): the toy's λ grid is coarse.

### 6.4 Running

`data/hasty/phantom/factory_arms/ph0001/pyr_mag/queue.sh` (2 mm, ZMSHR1-ms, κ 1, 15 it, start = the circulant full-model
warm start): A baseline, B `--bin-group`, C `--mag-couple 0.1`; scored with `research/experiments/recon_diag/frame_fit.py`
against the per-frame truth. Reference numbers on that yardstick (slope / corr / wrapped phase error rms):
fix_2mm/llr_s deep (≥ 5) 0.653 / 0.641 / 0.420 rad, wall (0–2) 0.416 / 0.284 / 0.646 rad; inverse crime deep 0.965 /
0.910 / 0.218 rad.

2026-11-13: the first pass of queue.sh was OOM-killed in iteration 1 of A and C (B could not run: `--bin-group` regroups
the readouts into other sets, so the file store key changes and `HASTY_KSTORE_REUSE_ONLY` refuses a new 44 GB store).
Cause [measured, journalctl]: the host-state ZMSHR1-M step held x, g, pg, x_prev, g_prev plus four full temporaries
(s, m, M⁻¹s, Mm): 9 × 8.7 GiB + 22 GiB setup > 84 GiB. Fixed in `hoststack::run_objective_host`: the metric step is
fused per set chunk (s, m overwrite x_prev, g_prev; M⁻¹s and Mm are chunk temporaries, recomputed in a second pass,
≈ 2 % of an iteration), 4 stacks. `tests/host_state.rs` now also covers circ / zmshr1m / zmshr1ms (host vs device
1.4–1.5e-4, the same as run-to-run noise and as before the change). Rerun: `pyr_mag/queue2.sh`; B uses an in-RAM
quantised store (`--kstore zstd1+q0.1`, +0.08 % noise variance) instead of a new file store.
