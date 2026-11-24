# Temporal pyramid: how coarse and fine knot solutions relate (theory)

2026-11-22. **Question (user).** Reconstruct with knots (6, 12) (respiratory × cardiac) to near convergence. Then
double every axis to (12, 24): each knot becomes two.
- Does the average of the two fine velocities equal the coarse one?
- Can we regularise the fine level by "fine frames must average to the coarse solution"?
- How do we keep the fine level from drifting away from the truth when each knot carries less data (undersampling)?

This note derives what can be said exactly. §7 checks every statement on a small problem where all matrices can be
computed exactly.

## 1. Setting and notation

- **Time.** A sample j has a time label t_j: cardiac phase, and respiratory position or bin.
- **Unknown.** An image-valued function of time. In the complex-image model it is one complex image per encode; in the
  velocity model it is magnitude, background phase and velocity per voxel.
- **Level ℓ.** A level uses K_ℓ knots per axis and periodic (cardiac) or open (respiratory) uniform B-splines
  B^ℓ_k(t) of degree p: 0 = frames, 1 = hat functions, 3 = cubic.

  x(r, t) = Σ_k c_k(r) B^ℓ_k(t),   coefficient vector c ∈ ℂ^{K_ℓ · n_vox}.

- **Nesting.** For uniform B-splines of degree p, V_K ⊂ V_{mK} for every integer m. Each coarse B-spline is an exact
  combination of fine ones (two-scale relation; dyadic case B^c_k = Σ_i s_{i−2k} B^f_i with s =
  binom(p+1, ·)/2^p: (1, 1) for p = 0, (½, 1, ½) for p = 1, (1, 4, 6, 4, 1)/8 for p = 3). In coefficients, x_c ∈ V_c
  is the fine vector S c_c, where S (K_f × K_c per voxel) is the subdivision matrix.
  - **Consequence:** 12 → 24 and 16 → 48 nest. 12 → 16 and 16 → 24 do NOT. The frame pyramid "12 → 16 → 24"
    planned in pipeline_target.md breaks every statement below, and its prolongation is already an approximation.
    Use integer refinement ratios.
- **Data.** y_j = a_j^H x(t_j) + n_j, where a_j is sample j's spatial encoding row (coil, k-space position, segment
  physics). Then

  y = 𝒜_ℓ c,  (𝒜_ℓ c)_j = Σ_k B^ℓ_k(t_j) a_j^H c_k,  𝒜_c = 𝒜_f S.

- **Normal operator.** Data part N_ℓ = 𝒜_ℓ^H Σ⁻¹ 𝒜_ℓ, with coil noise whitening Σ⁻¹. Its block structure is

  (N_ℓ)_{kk'} = Σ_j B_k(t_j) B_{k'}(t_j) a_j a_j^H.      (1)

  Hence N_c = S^H N_f S: the coarse normal operator is the Galerkin restriction of the fine one, exactly.

## 2. Theorem 1: the coarse solution is the energy projection of the fine one

**Assumptions.**
- The objective is quadratic, J_ℓ(c) = ½‖𝒜_ℓ c − y‖²_{Σ⁻¹} + ½ c^H Q_ℓ c.
- The regulariser is level-consistent: Q_c = S^H Q_f S. This holds for Q = 0, and for any quadratic prior evaluated
  on the time function x(t) itself, i.e. on the prolongated image.

Let H_f = N_f + Q_f (assumed invertible on V_f), and let c_f, c_c be the minimisers on V_f and V_c.

**Claim.**

  S c_c = Π c_f,  Π = S (S^H H_f S)⁻¹ S^H H_f,                                            (2)

i.e. x_c is the H_f-orthogonal projection of x_f onto V_c.

**Proof.**
1. The fine normal equations are H_f c_f = 𝒜_f^H Σ⁻¹ y.
2. Multiply by S^H: S^H H_f c_f = 𝒜_c^H Σ⁻¹ y = H_c c_c = S^H H_f S c_c.
3. So S^H H_f (c_f − S c_c) = 0. ∎

**Equivalent forms.**
- (a) Energy: x_c = argmin_{v ∈ V_c} ‖v − x_f‖_{H_f}.
- (b) Data space, for Q = 0: 𝒜_c c_c is the orthogonal projection of the fine prediction 𝒜_f c_f onto range(𝒜_c).
  The coarse model "sees" exactly the coarse part of the fine solution's predicted k-space, and nothing else.
- (c) Exact split: x_f = x_c + w with w ⊥_{H_f} V_c. Refinement never changes the coarse solution's own component.
  It adds an H-orthogonal detail.

The coarse solution is therefore an exact "average" of the fine one, but in the metric H_f of the problem. That is
neither the knot-value average nor the plain time average.

## 3. Corollary 2: when the average is a plain (weighted) time average

The metric H_f becomes a time-only weight when the encoding is time-stationary. If the encoding seen in any short time
window is the same spatial normal operator E scaled by a sample density w(t), then (1) gives N_f = G^w_f ⊗ E with
temporal Gram matrix (G^w)_{kk'} = ∫ B_k B_{k'} w dt. If also Q = 0 (or Q = G-structured ⊗ anything), then
Π = Π^{G^w} ⊗ I, and per voxel

  ∫ B^c_k(t) [x_f(r, t) − x_c(r, t)] w(t) dt = 0   for every coarse k.                     (3)

**Readings.**
- **Degree 0 (frames).** A coarse frame is the union of two fine frames, and (3) gives

  x_c = (w_1 x_{f,1} + w_2 x_{f,2}) / (w_1 + w_2),

  with w_i the data weight (sample count × encoding power) in fine frame i. "Two fine frames average to the coarse
  one" is exactly true, with data weights, per voxel. It is the plain mean only if both halves hold equal data.
- **Degree ≥ 1.** The *moments against each coarse B-spline* agree; the knot values do not. In coefficients,
  c_c = (G^w_c)⁻¹ S^T G^w_f c_f. The weighted L2 projection is local only up to an exponentially decaying
  inverse-mass tail, so it is not a 3- or 5-tap average.
- **Invariant on every level.** Constants lie in every V_ℓ (partition of unity), so the data-weighted time mean
  ∫ x_ℓ(r, t) w(t) dt is identical on all levels. For a flow model this means the data-weighted cycle-mean flow is
  exactly level-independent in the quadratic case. That makes it a free diagnostic: drift in the cycle mean across
  levels measures prior and nonlinearity effects, not resolution.

**Real trajectories are not time-stationary.**
- **Single coil, gridding picture.** N acts per spatial frequency k with that frequency's own temporal sampling
  density w(k, t). (3) then holds per k-space frequency with weight w(k, ·).
- **The k-space centre** is visited in every short window, so w(0, t) ≈ w(t): the low-resolution image averages
  exactly as in (3).
- **High k** is revisited only every T_rev(k), so w(k, ·) is spiky. There the coarse solution is a k-dependent
  weighted time average of the fine one, not a local mean.
- **Coil sensitivities** couple each k to its neighbours within the sensitivity kernel width, which smears this
  dependence.

## 4. Theorem 3: general (non-quadratic) objectives — Bregman projection

**Convex case.**
- Assumptions: J is convex and differentiable on V_f, and x_f minimises it there.
- Then ⟨∇J(x_f), v − x_f⟩ = 0 for all v ∈ V_f ⊃ V_c, so for v ∈ V_c

  J(v) − J(x_f) = D_J(v, x_f)   (the Bregman divergence)  ⇒  x_c = argmin_{v ∈ V_c} D_J(v, x_f).   (4)

- The coarse solution is the Bregman projection of the fine solution onto the coarse space.
- Non-smooth case: the same holds with a subgradient g ∈ ∂J(x_f) that annihilates V_f.

**Velocity model (non-convex).** Locally around x_f, D_J(v, x_f) = ½‖v − x_f‖²_{H(x_f)} + O(‖v − x_f‖³), with H
the Gauss–Newton Hessian. So x_c = Π^{H(x_f)} x_f up to second order in the coarse–fine gap.
- For the velocity block in the time-stationary picture,

  H_vv(r, t) ≈ w(t) |m(r, t)|² Σ_e γ² M1_e M1_e^T   (3 × 3 per voxel and time).

- The weighted-average statement (3) therefore holds for velocities with weight w(t) · |m(r, t)|² · (encoding
  matrix):
  - time points with little signal (inflow-saturated or dark blood phases) count less;
  - with a non-isotropic encoding set, the components mix through the 3 × 3 weight.

**Complex-image model, velocity decoded afterwards.** Theorem 1 holds for the complex images. The decoded velocity of
the coarse image is then the argument of a weighted complex mean. For two fine values a e^{iφ1}, b e^{iφ2}:

  arg(a e^{iφ1} + b e^{iφ2}) = (a φ1 + b φ2)/(a + b) + ρ(1 − ρ²) Δ³/24 + O(Δ⁵),   ρ = (a − b)/(a + b), Δ = φ1 − φ2.

- So the coarse phase is the magnitude-weighted mean of the fine phases, up to a cubic term that vanishes for equal
  magnitudes.
- Worst case: Δ = 0.94 rad (Δv = 0.3 VENC between the two fine frames) gives ≤ 0.013 rad ≈ 0.4 % of VENC.

**PnP / P²nP.** The fixed point is not a minimiser in general, so (4) does not apply. In the local affine model of the
P²nP paper (a symmetric denoiser Jacobian W acts as a quadratic prior Q ∝ (W⁻¹ − I)/α), Theorem 1 applies with that
Q, but only if Q is level-consistent (Q_c = S^H Q_f S). A denoiser applied to the coarse level's own frames, at its
own strength, is a different prior.
- Design rule: if nested consistency is wanted, the coarse level must evaluate the prior on the prolongated
  (fine-time) representation.
- With per-level λ (σ_ℓ² ∝ 1/samples, convergence_p2np.md §6), coarse and fine are different estimators. The gap
  between them is the prior's level dependence, not a resolution effect.

## 5. Theorem 4: what refinement adds, and why it can diverge

**Detail split.** Split V_f = S V_c ⊕ T W, for any complement (hierarchical basis, B-spline prewavelets): c_f = S a + T d.
The fine normal equations in block form are

  [H_cc H_cd; H_dc H_dd] [a; d] = [S^H b; T^H b],   H_cc = S^H H S, H_cd = S^H H T, …

Eliminating a gives

  d = Σ_d⁻¹ T^H 𝒜^H Σ⁻¹ r_c,   Σ_d = H_dd − H_dc H_cc⁻¹ H_cd   (Schur complement),
  a = c_c − H_cc⁻¹ H_cd d,                                                                   (5)

where r_c = y − 𝒜_c c_c is the residual of the coarse solution.

**What this says.**
- The fine level fits only the coarse residual, through the Schur complement of the detail.
- The coarse coefficients shift by −H_cc⁻¹ H_cd d. They do not shift in the H-orthogonal split (Theorem 1 (c)), but
  they do in any other split; knot values and plain averages move.

**Error against the truth x\*.** Let Π_ℓ be the H-projection onto V_ℓ. With unbiased noise of covariance σ²:

  E‖x_f − x\*‖²_H = ‖(I − Π_f) x\*‖²_H + σ² tr(H_f⁻¹ N_f H_f⁻¹ H)   (Q = 0: σ² dim V_f).

Refinement from c to f improves the truth error iff

  ‖(Π_f − Π_c) x\*‖²  >  σ² tr(Σ_d⁻¹ …)   (detail bias removed > detail variance added).       (6)

**Size of the two sides.**
- **Bias removed.** It is ≈ h_c^{2(p+1)} ‖∂_t^{p+1} x\*‖² − h_f^{2(p+1)}(…). For smooth cardiac motion it falls fast
  with the level.
- **Variance added.** In the time-stationary case Σ_d = Σ^t_d ⊗ E, and the temporal Schur part scales like the data
  per fine knot. So the variance per detail coefficient grows like K_f (benign: ×2 per dyadic level).
- **The catastrophic part comes from non-stationary encoding.** A detail function oscillates on the fine knot scale.
  At spatial frequency k it is constrained by ∫ |ψ_d(t)|² w(k, t) dt:
  - where the fine knot spacing Δt_f is longer than the revisit time T_rev(k), every detail interval sees data at k,
    and Σ_d is well conditioned there;
  - where Δt_f < T_rev(k), whole fine intervals contain no sample at k. The detail at k is then fixed only by coil
    encoding (g-factor) or not at all, so Σ_d has near-null directions and tr(Σ_d⁻¹) explodes.
- **That is the divergence the question worries about.** It is spatially selective: it happens at high spatial
  frequencies first, at the knot spacing where T_rev(k) = Δt_f.

## 5b. Theorem 5: the "averages must match the coarse solution" anchor works, but only with a data-independent average

(Corrected after the numerical check of §7. A first version of this note claimed the anchor cannot help; that was
wrong, because the near-null directions are not purely detail.)

**Setup.** Penalty μ ‖R c_f − c_c‖², where R is a restriction with R S = I: an "averaging" operator, i.e. coordinates
of a projection onto V_c. Take an L2-type split adapted to R: V_f = S V_c ⊕ T W with T spanning ker R. Write
c_f = R⁺ a + T d with R⁺ a right inverse of R, and H in these coordinates as blocks H_cc, H_cd, H_dd.

**Case A: R = the H-coordinates (S^H H S)⁻¹ S^H H.** The unregularised fine solution already satisfies R c_f = c_c
(Theorem 1), so the penalty is inert at every μ (§7: errors identical to no anchor).

**Case B: R data-independent**, e.g. the temporal L2 restriction (S^T G S)⁻¹ S^T G or local averaging. For μ → ∞ the
coarse coordinate is pinned, a = c_c, and the detail solves

  H_dd d = T^H (b − H R⁺ c_c)      (instead of the Schur complement Σ_d = H_dd − H_dc H_cc⁻¹ H_cd when a is free).   (7)

- **Variance.** Since Σ_d ⪯ H_dd, the detail's noise covariance σ² H_dd⁻¹ ⪯ σ² Σ_d⁻¹. The gain
  tr Σ_d⁻¹ / tr H_dd⁻¹ is large exactly when H_cd is large, i.e. when the near-null directions of H have a sizeable
  coarse (L2) component.
  - That is what undersampling produces. A near-null vector is a function concentrated on fine time intervals that
    have no samples at some k. A bump localised on one fine knot interval projects onto the coarse space with about
    half its norm, so it is not a detail function.
  - The anchor removes the coarse half of every such hole, using a coarse estimate that is well determined.
  - Added variance: the coarse estimate's own noise propagated through R⁺ − T H_dd⁻¹ H_dc R⁺, which is small because
    the coarse level is well conditioned.
- **Bias.** The data-independent coarse component of the fine solution is forced to the coarse solution, whose mean is
  the data-weighted (H) projection of the truth. The bias is therefore ≈ ‖(R_L2 − R_H^(c)) x\*‖: the mismatch between
  the plain and the data-weighted coarse averages of the true dynamics. It vanishes for temporally uniform sampling
  (Corollary 2: then R_H = R_L2 per voxel) and is second order where the true temporal detail is small.

**So the proposal is sound with one precise condition.** "Fine frames must average to the coarse solution" is a
variance reducer if the average is a fixed, data-independent temporal average. It must not be the data-weighted one,
because that one is automatically satisfied and does nothing. What it buys is decoupling: the fine level only
estimates the detail given a trusted coarse part. That replaces the ill-conditioned Schur complement by the detail
block itself.

**It does not regularise the detail itself.** A pure detail function on an uncovered interval (H_dd's own near-null
space) is still free. That part needs the detail penalty of §6 rule 2. The two act on complementary subspaces, and
the toy's best large-K result uses both.

## 6. What follows for the pyramid (derived rules)

1. **Nest exactly.** Use integer knot ratios (6 → 12 → 24, 4 → 8 → 16 …) and prolongate by the subdivision matrix S,
   which is lossless. 12 → 16 → 24 is not nested.
2. **Anchor the coarse part with a data-independent average** (Theorem 5 case B): add μ ‖R_L2 c_f − c_c‖² with the
   temporal L2 restriction R_L2 = (S^T G S)⁻¹ S^T G per voxel, a cheap K_c × K_f banded-inverse operator applied
   per voxel. Never use the data-weighted average (inert). μ of the order of the data term's scale; the toy is flat
   between μ = 1 and 100 × the mean diagonal of N.
3. **Parametrise the fine level as coarse + detail:** x_f = S c_c + T d, with T = B-spline prewavelets or the
   hierarchical complement. Regularise the detail as well:
   - a level- and frequency-dependent penalty Σ_ℓ λ_ℓ(k) ‖d_ℓ(k)‖²;
   - λ_ℓ(k) small where the fine knots are covered at k (Δt_ℓ ≥ T_rev(k)), and large where they are not.
   - This is the Bayes choice for a temporal smoothness prior: by the wavelet norm equivalence, Σ 2^{2ℓs} ‖d_ℓ‖²
     ≍ ‖x‖²_{H^s(t)}. The k-dependence is what plain temporal TV or L2 smoothness lacks.
   - In practice: temporal detail is kept at low spatial frequency, and at high spatial frequency it comes from the
     prior or the coil encoding only. This is the principled version of view sharing (KWIC-like).
4. **Accept a refinement only if it helps, measured on the data** (real-data rule).
   - Held-out k-space prediction error satisfies E‖y_ho − 𝒜_ho x‖² = σ² n_ho + ‖𝒜_ho(x − x\*)‖², so it tracks the
     truth error in the data norm.
   - Refine while the held-out error drops, and stop or raise λ_ℓ when it rises: an empirical test of (6).
5. **Predict the divergence before running the level.** The spectrum of Σ_d can be bounded cheaply: a few Lanczos
   steps on the detail Schur complement give its smallest eigenvalues, using the new estimator from
   temporal_operator.md §24 on the lower end, and the coverage of the fine knots per k-space shell gives T_rev(k).
   Together they predict at which knot spacing and spatial frequency the variance term of (6) takes over.
6. **Early stopping on the fine level is a regulariser too.** Starting from S c_c, gradient-type iterations reduce
   the detail error per eigenvalue λ of Σ_d by (1 − αλ)^n. Near-null (aliased) detail barely moves in n iterations:
   implicit Tikhonov with strength ≈ 1/(nα). That makes "few fine passes after a converged coarse level" safe, but
   the effect is iteration-count dependent; an explicit λ_ℓ(k) is reproducible.
7. **Diagnostics that come for free:**
   - the data-weighted cycle-mean image and flow (Corollary 2) is level-invariant in the quadratic case;
   - the H-projection of the fine solution should reproduce the coarse one, which tests the solver and the nesting;
   - the gap in either one measures prior and nonlinearity effects.

## 7. Numerical check

**Script.** `research/experiments/temporal_pyramid/toy.py`; results in `toy_results.txt` / `.json` next to it
(14 s, CPU).

**Toy.** Complex dynamic 8 × 8 image (static background plus a disc whose magnitude and phase vary with 6–8 cardiac
harmonics), 4 coils with smooth complex sensitivities, Cartesian ky lines (all kx) at sample-level cardiac times.
Normal operators are exact dense matrices, N = Σ_line G_line ⊗ E_line.

**T1. Theorem 1** (full encoding per readout and line-sampled; degree 0 / 1 / 3; K 6 → 12): N_c = S^H N_f S, and
S c_c = Π_H c_f, both to 1e-14 to 1e-15. Exact.

**T2. Corollary 2.**
- Full encoding per readout: the coarse-B-spline moments of x_f − x_c and the data-weighted cycle mean agree to 1e-15
  for p = 0 / 1 / 3. At p = 0 the coarse frame equals the sample-count-weighted mean of its two fine frames. In this
  toy the counts per half-frame were near equal, so the plain mean agrees too.
- Line-sampled, one coil: the per-voxel moment identity fails (0.18 at p = 0, 2.8 at p = 3). It holds per ky line
  with that line's own sample times (1e-9 / 1e-7, regularisation-limited), as derived.

**T3. Phase of a weighted complex mean** (a = 1, b = 0.6, Δ = 0.94 rad):
- exact 0.12632 rad, magnitude-weighted phase mean 0.11750 rad;
- predicted cubic term 0.00811, actual gap 0.00882 (the rest is higher order).

**T4. Refinement under undersampling.**
- **Setup:** variable-density line sampling (density ∝ (1 + |ky|)^−1.8), 480 readouts, 12 % held out, noise,
  cubic B-splines.
- **What the table shows:** truth error = relative L2 over (voxel, time on a 400-point grid); held-out = relative
  k-space residual on the held-out readouts.
- **Anchors:** "avg" = L2 anchor to the previous level's unregularised solution (μ in units of the mean diagonal of
  N), "avgH" = the same with the data-weighted R.
- **Detail penalties:** "detail" = temporal L2 detail penalty, uniform or weighted per ky line by
  (1 − coverage)/coverage.
- **Coverage:** the fraction of fine knot intervals that contain a sample of that line.

| K | none | avgH μ 100 | avg μ 1 | avg μ 100 | detail uniform 3 | detail k-adaptive 3 | avg 1 + detail 3 |
|---|---|---|---|---|---|---|---|
| 6 | 0.353 (ho 0.098) | – | – | – | – | – | – |
| 12 | 0.282 (0.077) | 0.282 | 0.277 | 0.279 | 0.275 | 0.277 | 0.276 |
| 24 | 0.198 (0.0223) | 0.198 | 0.153 (0.0226) | 0.159 | 0.130 (0.0215) | **0.129 (0.0212)** | 0.145 (0.0231) |
| 48 | **1.296** (0.0599) | 1.296 | 0.258 (0.0258) | 0.256 | 0.207 (0.0233) | 0.203 (0.0235) | 0.199 (0.0222) |

Mechanism (Theorem 5), detail-block noise with the coarse part free (Schur complement) vs pinned (H_dd):

| K | tr Σ_d⁻¹ | tr H_dd⁻¹ | λ_min(Σ_d) / λ_min(H_dd) | fine-interval coverage of the outermost / central ky |
|---|---|---|---|---|
| 12 | 1.5e3 | 1.2e3 | 0.71 | 0.75 / 1.0 |
| 24 | 1.3e4 | 6.6e3 | 0.26 | 0.50 / 1.0 |
| 48 | 1.0e6 | 8.8e4 | 0.071 | 0.31 / 1.0 |

**Reading.**
- **Divergence.** Unregularised refinement 24 → 48 diverges: 0.198 → 1.296. The variance explodes (tr Σ_d⁻¹ ×80)
  exactly where the outer ky lines cover only a third of the fine intervals.
- **The energy-metric anchor is inert** (identical to none, Theorem 5 case A).
- **The data-independent anchor** removes most of the divergence (1.296 → 0.26), through the 12× smaller detail
  variance (Theorem 5 case B). It also helps at K = 24 (0.198 → 0.153).
- **The detail penalty** is best at K = 24, where coverage is still ≥ ½ (0.129–0.130). The k-adaptive weight is only
  marginally better than uniform in this 8-line toy (4 coils fill much of the missing ky).
- **Anchor + detail penalty** is best at K = 48 (0.199), but still worse than K = 24. With this much data, 24 knots
  is the right resolution.
- **Held-out k-space picks it.** The held-out error is lowest at K = 24 with the detail penalty (0.0212), and it flags
  the unregularised K = 48 divergence (0.0599 vs 0.0223). So the data-only stopping rule (§6 rule 4) selects the
  truth-best level here.

## 8. Answers to the question

1. **Does the average of the two fine velocities equal the coarse one?**
   - Exactly, in the quadratic (complex-image, no or level-consistent prior) case, but in the metric of the
     problem: x_c is the H-projection of x_f (Theorem 1).
   - It becomes a per-voxel, data-weighted time average only for time-stationary encoding (Corollary 2). For real
     trajectories it holds per k-space frequency with that frequency's own sampling times: the k-space centre
     averages like the plain mean, high k does not.
   - For velocities the weights are additionally |m|² and the encoding matrix (Theorem 3).
   - Decoding the coarse complex image gives the magnitude-weighted phase mean, up to a cubic term ≤ 0.4 % of VENC
     for 0.3 VENC between the halves.
   - With PnP at level-dependent strength, the levels are different estimators.
2. **Can the fine level be regularised by "averages equal the previous solution"?** Yes, and it works (§7 T4), with
   one condition: the average must be a fixed, data-independent temporal restriction. The data-weighted one is
   automatically satisfied and inert. The anchor works by decoupling: the fine level estimates only the detail
   around a trusted coarse part, replacing the Schur complement by the better-conditioned detail block. Its bias is
   the difference between plain and data-weighted coarse averages of the true dynamics.
3. **How to avoid diverging as knots increase?**
   - The divergence is in the detail variance. It starts at the spatial frequencies whose revisit time exceeds the
     fine knot spacing (Theorem 4, (6)).
   - Against it: anchor the coarse part (rule 2), penalise the detail with a weight that grows where fine intervals
     go unsampled (rule 3), and refine only while the held-out k-space error drops (rule 4).
   - Use integer knot ratios. 12 → 16 → 24 breaks the nesting every statement relies on.

## 9. Open (theory and next experiments)

- **Real trajectory.** Measure the coverage function cov(k, Δt) of the yarn readout at the planned knot spacings.
  This predicts where (6) flips, before any recon.
- **PnP.** Extend Theorem 5 to the P²nP fixed point with the anchor inside the data step, and check that the
  denoiser does not undo the anchor.
- **Nonlinear velocity model.** Theorem 3's local statement needs a bound on the second-order term in terms of the
  coarse–fine gap; check it on the dev phantom (cycle-mean flow invariance, Corollary 2, as the measurable).
- **μ and λ_ℓ(k) from the data.** Choose them per level from the held-out error, or from the noise level
  σ_ℓ² ∝ 1/samples.

## 10. Leave-one-out pooled averages + voxelwise unmixing (user's idea, 2026-11-22)

**Scheme.**
1. Split the data into N bins.
2. For each m, solve the pooled one-image problem over all bins but m: x̂_{−m}.
3. Recover the N frames voxelwise, by inverting the scalar mixing x̂_{−m} = Σ_{j≠m} w_j x_j / (W − w_m), with w_j the
   data weight of bin j.
4. Each pooled solve has ≈ all the data, so it is well conditioned and converges fast. It is also Toeplitz-friendly:
   the cost is independent of the sample count.

### 10.1 Exact algebra
The pooled solves satisfy N_{−m} x̂_{−m} = b − b_m (b_m = A_m^H y_m, b = Σ b_m). So the N averages are an invertible
re-encoding of exactly the per-frame back-projections:
- no information is gained or lost;
- recovering the frames exactly needs N_k⁻¹, the per-frame normal operator, i.e. the per-frame problem itself.

**Condition for voxelwise unmixing to be exact:** the mixing weights N_S⁻¹ N_j are scalars, i.e. N_j = w_j N̄ for every
bin. That is time-stationary encoding (Corollary 2).
- In the toy with full encoding per readout, LOO + unmixing equals per-frame least squares to machine precision,
  noise included.
- The mixing matrix has condition number N − 1, but in that case the unmixed estimate is algebraically the per-frame
  estimate (same variance).

### 10.2 What it computes in general (first order, and in the toy)
LOO + unmixing ≈ the pooled mean plus a mean-operator deconvolution of each frame's residual back-projection:

  x_k ≈ x̄ + (w̄ N̄)⁻¹ A_k^H (y_k − A_k x̄),   x̄ = pooled solution, w̄ N̄ = N_tot / N.            (8)

**Toy (line-sampled, 4 coils), truth error of LOO + unmix vs (8):**

| N | LOO + unmix | (8) |
|---|---|---|
| 6 | 0.081 | 0.076 |
| 12 | 0.106 | 0.098 |
| 24 | 0.166 | 0.159 |

**Reading of (8):** it is *one step of preconditioned Richardson for the per-frame problem*, started at the pooled
mean, with the inverse mean normal operator as preconditioner.
- It needs one pooled solve plus N solves with the same mean operator (multiple right-hand sides, or the circulant
  approximation).
- The user's form needs N solves with N different operators.
- Same estimator to first order, so (8) is the cheaper way to compute the idea.

### 10.3 Bias
For (8), noise-free, exactly:

  x̂_k − x_k = ((w̄ N̄)⁻¹ N_k − I)(x_k − x̄)   (first order: (w̄ N̄)⁻¹ Δ_k (x_k − x̄), Δ_k = N_k − w̄ N̄).      (9)

The bias is the frame's relative PSF deviation acting on the frame's temporal deviation. Three consequences:
- **Static tissue is unbiased** (x_k = x̄).
- **Dynamics are unbiased wherever the frame samples like the mean.** For radial/yarn-type readouts that means the
  k-space centre, i.e. the low-resolution dynamics.
- **Where the frame did not sample** (N_k δ = 0) the estimate falls back to the mean. Temporal detail there is lost
  (view sharing), and the bias equals minus the temporal detail.
- The bias grows with N because the per-frame deviation grows. Median ‖(w̄ N̄)⁻¹ N_k − I‖: 0.58 / 0.93 / 1.00 at
  N = 6 / 12 / 24.
- Noise-free truth error: 0.081 → 0.106 → 0.166.
- The first-order formula explains 80 % of the bias at N = 24 (residual 0.034 of 0.166). At small N the per-frame
  operators' own unrecoverable part dominates, and the formula explains less.

### 10.4 Variance
- Time-stationary: identical to per-frame least squares.
- Otherwise: much lower, because unsampled directions default to the mean instead of exploding. At N = 24 with
  noise, per-frame LS has truth error 0.86 and LOO + unmix 0.19.
- So it is a biased, low-variance estimator. It is Theorem 5's anchor with the coarse space = constants in time,
  and with H_dd replaced by the mean operator.

### 10.5 Use
- **It is an excellent initialiser.** The initialisation never changes the minimiser (convergence_p2np.md §4), so the
  bias does not matter as a start, and the cost is ≈ one pooled solve plus one gradient per set.
- **Iterating (8)** (more Richardson steps) moves it toward per-frame least squares: less bias, more variance. The
  number of steps is a regularisation parameter; combine it with the prior or the detail penalty of §6 for a final
  estimator.
- **As a pyramid level** it is K = 1 → N with the anchor built in. Intermediate levels (K = 1 → 6 → 12 → 24 with (8)
  generalised to coarse-space means) keep the bias of each step to the detail between consecutive levels.
- **Toeplitz.** The pooled solve needs one set of per-segment kernels (the coil-free kernel_l structure of the normal
  operator). The residual back-projection A_k^H(y_k − A_k x̄) is computed once per frame by the NUFFT, so no
  per-frame kernels are needed.

**Script:** `research/experiments/temporal_pyramid/loo.py` (results in `loo_results.txt`).

## 11. Coverage of the real trajectory (yarn, ph0001 10-min sim)

**Script.** `research/experiments/temporal_pyramid/coverage.py`; tables in `coverage_cell4.txt` / `coverage_cell2.txt`.

**Definition.** For every k-space cell, coverage is the fraction of (encode, knot interval) pairs that contain at
least one ADC sample of that encode in the cell.
- A knot interval is one respiratory amplitude bin (equal counts) × one cardiac interval.
- Cells are 4/FOV or 2/FOV per axis. 4/FOV is roughly the reach of coil encoding (smooth sensitivities), so a cell hit
  at 4/FOV can be filled by the coils; 2/FOV asks for coverage without that help.
- Shells are |k|/k_max, normalised per axis.
- The scan has 7592 readouts per encode.

**Mean coverage, 4/FOV cells** (in brackets: fraction of cells below 0.5):

| resp × cardiac | intervals / encode | |k| < 0.1 | 0.2–0.35 | 0.5–0.7 | 0.85–1 |
|---|---|---|---|---|---|
| 1 × 24 | 24 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 × 48 | 48 | 1.00 | 0.98 | 0.95 | 0.95 |
| 6 × 12 | 72 | 1.00 | 0.91 | 0.87 | 0.86 |
| 4 × 24 | 96 | 1.00 | 0.84 | 0.78 | 0.77 |
| 4 × 32 | 128 | 0.99 | 0.74 | 0.68 | 0.67 |
| 6 × 24 | 144 | 0.99 | 0.70 | 0.64 (0.04) | 0.63 (0.07) |
| 4 × 48 | 192 | 0.97 | 0.59 | 0.53 (0.23) | 0.52 (0.27) |
| 12 × 24 | 288 | 0.92 | 0.45 (0.95) | 0.40 (1.00) | 0.39 (1.00) |

With 2/FOV cells the coverage is roughly the 4/FOV value at 2.5–4× the interval count. For example 6 × 12 drops to
0.43–0.36 beyond 0.2 k_max, and 1 × 48 to 0.5.

**Reading.**
- **Coverage depends almost only on the number of intervals per encode,** about 1 − exp(−c · 7592 / intervals), not
  on how they split between respiratory and cardiac.
  - Respiratory bins cost the same as cardiac knots.
  - The user's example (6, 12) → (12, 24) is the step where the outer shells lose coverage at the coil-kernel scale:
    0.86 → 0.39, with every cell below ½.
- **The k-space centre stays covered at every layout.** The low-resolution dynamics are determined up to 288
  intervals per encode, as Theorem 4 predicts.
- **Where coverage < 1, (8) / LOO is biased and refinement diverges.** That is the shells beyond ≈ 0.2 k_max at
  ≥ 100 intervals per encode. This sets the data-driven ceiling on knots × bins for a 10-min scan: about 72–100
  intervals per encode at full coverage of the coil-kernel cells.
- **Interval coverage is a conservative proxy.** A cubic B-spline spans 4 intervals, so its coefficient is determined
  with more data. But the detail between neighbouring knots is not, and that detail is what refinement adds.
- **Consequence for the coronary target** (temporal_rank.md: ~32 SVD or ~48 cubic cardiac frames): 1 × 48 or 2 × 32
  is coverable, but × 4 respiratory bins is not without the detail penalty and the anchor. Resolving respiration by
  motion correction (a pose per readout, motion_navigators.md) rather than by bins saves the factor.

## 12. Implementation and dev results (2026-11-22)

### 12.1 What was built (frame mode only)
- **`--init warm|zero|meanres[:STEPS][:circ|pcgN][,..]`, `--resid-steps n`** (`hasty-recon/src/meanres.rs`). meanres =
  (8): start at the pooled `--warm-full` solution x̄ (required), one gradient pass gives every frame's residual
  back-projection r_k = A_kᴴ(y_k − A_k x̄), then δ_k ≈ N̄_e⁻¹ r_k with N̄_e = (1/n_e) Σ_{s∈e} N_s:
  - `circ`: the per-encode circulant metric M_e as the approximate inverse; its unknown scale is fitted per encode by
    an exact line search of the frames' own objectives along d = M r (one more gradient pass, which also gives the next
    residual, so each further step costs one pass);
  - `pcgN`: N PCG iterations on N̄_e for all frames at once (one right-hand side per frame), preconditioned by M_e;
    each iteration is one grouped pass with one group per frame (= n_e ordinary passes).
  - STEPS > 1: more Richardson steps x_k ← x_k + N̄⁻¹ A_kᴴ(y_k − A_k x_k). Stoppable at step boundaries.
- **`--anchor none|mean:μ|<coarse run dir>:μ[,..]`** (`hasty-recon/src/tanchor.rs`, an `Objective` wrapper like RED):
  f_A = ½ μ d̄ Σ_a n_a ‖R_a x − c_a‖², R_a = the plain mean over the fine frames of one encode inside coarse frame a
  (the temporal L2 restriction for nested degree-0 frames; never the data-weighted one). `mean`: c = x̄, one group per
  encode. A coarse run dir: its newest recon, nested frames only (integer ratio, `nested_frame`).
  - **Unit of μ**: the anchor's curvature on the group-constant mode, in units of the mean diagonal d̄ = tr N_t / n of
    one frame's normal operator (Hutchinson, `P2np::mean_diag`). μ = 1: as stiff on the mean mode as one frame's data
    term is on average. Same unit as the toy of §7 (whose penalty μ‖Rc − c_c‖² has curvature μ on the mean mode).
  - **Step safety**: λ_max(N + A) ≤ λ_max(N) + μ d̄; with the metric λ_max(M(N + A)) ≤ λ_max(M N) + μ d̄ λ_max(M)
    (Weyl in the M⁻¹ inner product), λ_max(M) by 30 power steps on M alone (+5 %, `P2np::metric_lmax`). The Lanczos
    λ_max of N or M N is unchanged; the bound adds the anchor's share.
  - Memory: targets on the host (one image per group) + their device copies for device-resident state; host state
    works in place on the host stack. Secant groups = union of the inner ones and the anchor groups.
  - Not in the knot / velocity objective (vobj): there R must act on the knot coefficients of a nonlinear model; left
    open.
- `--report-detail`: the lumen velocity error split into time mean and temporal detail (v_f − v̄ vs t_f − t̄).
- Every init × anchor combination is one more run after one setup (tags `_mr1c`, `_mr2p3`, `_zero`, `_anc_mean1`).
  Defaults unchanged (no flag → identical runs and tags).

### 12.2 Dev check
**Script.** `research/experiments/temporal_pyramid/run_dev.sh` (logs `dev_run1.log`, `dev_run2.log`, tables
`dev_results.md` via `parse_dev.py`). 14 + 6 min GPU, one setup each (40 s with the tune cache).
- **Data:** ph0001 sim_flow_10min, 128 × 72 × 60 grid (grid-scale 0.5), k-space cropped to the grid, 8 virtual
  coils, 20 cardiac frames × 7 encodes = 140 sets.
- **Operator:** full physics (B0/ratemap segments L = 6, concomitant degree-4 fit, GNL).
- **Unknowns:** one complex image per (frame, encode).
- **Pooled solution x̄:** 3 plain PCGLS + 8 pooled PCG iterations through the full model per encode (`--warm-full 8`,
  per-encode circulant).
- **Algorithm:** P²nP with ZMSHR1-M (circulant metric), LLR κ = 1, 12 iterations.
- **What changed:** only the start (warm = x̄ in every frame, zero, meanres variants) and the anchor (none, mean:μ).
- **Score:** lumen (truth fraction > 0.9, 6165 voxels) velocity RMSE against the truth, split into the time-mean error
  and the temporal-detail error ((v_f − v̄) − (t_f − t̄)); run 2 also gives the median and 90th percentile of |v − t|.

**Caveat first: the dev case's velocity score is dominated by a large error that is not temporal.**
- The trivial zero image scores 0.427 m/s RMSE (= the truth's RMS: mean 0.25, detail 0.35; median 0.14 m/s).
- The pooled start scores 1.69 m/s (time-mean error 1.66, median 0.81, p90 3.1 m/s). Every recon after 12 iterations
  stays at p90 ≈ 2–3 m/s, i.e. ≳ 10 % of voxel-frames look like phase wraps (VENC-scale errors).
- The pooled PCG is far from converged after 8 iterations (relative residual 0.18).
- So RMSE differences below ~0.3 m/s are not meaningful here. The median is the more robust number.
- This was not caused by the new code: same picture in research/experiments/lanczos/F_lz.log, 1.8 m/s at the start.

**(i) Start only (iteration 0).** Data term f = ½ Σ‖E x − y‖²; cost in gradient-pass equivalents.

| start | f | RMSE (mean / detail) | median / p90 | cost |
|---|---|---|---|---|
| zero | 3.88e14 | 0.427 (0.250 / 0.347) | 0.141 / 0.61 | – |
| pooled x̄ (warm) | 1.030e14 | 1.693 (1.657 / 0.347) | 0.810 / 3.12 | 8 pooled PCG |
| meanres 1 step, circ | 9.63e13 | 1.718 (1.439 / 0.938) | 0.849 / 3.13 | + 2 passes, 1.8 s |
| meanres 2 steps, circ | 9.21e13 | 1.717 (1.338 / 1.075) | – | + 3 passes, 3.6 s |
| meanres 1 step, pcg3 | 7.45e13 | 2.074 (0.670 / 1.963) | 1.510 / 3.46 | ≈ 40 pass-equivalents, 25 s |
| meanres 2 steps, pcg3 | 5.31e13 | 2.107 (0.673 / 1.997) | – | ≈ 80 pass-equivalents, 50 s |

- The circ line-search scales are 0.066–0.074 at step 1 and 0.042–0.047 at step 2 (per encode).
- The pcg3 inner residual 2-norm grows (0.97 → 1.37 → 1.61 relative), while f falls 28 % per step. So the energy
  norm converges, but 3 iterations are far from N̄⁻¹.
- The frame mean of (8) moves by exactly the pooled Newton correction N_tot⁻¹(b − N_tot x̄). That is zero only for a
  converged x̄. Here ‖R x − x̄‖/‖x̄‖ = 0.04 (circ) and 0.27 (pcg3): most of pcg3's time-mean gain (1.66 → 0.67) is
  finishing the unconverged pooled solve.
- Detail error grows with the inverse's strength (0.35 → 0.94 → 1.96). This is the variance of §10.4 plus wraps.

**(ii) 12 P²nP iterations from each start** (no anchor):

| start | it 3 RMSE | it 6 | it 12 | it 12 mean / detail | it 12 median / p90 | it 12 f |
|---|---|---|---|---|---|---|
| zero | 1.82 | 1.92 | 1.71 | 0.51 / 1.63 | 0.840 / 3.13 | 8.87e13 |
| pooled x̄ | 1.75 | 1.90 | 1.59 | 0.71 / 1.42 | 0.684 / 2.99 | 5.58e13 |
| meanres 1 circ | 1.65 | 1.64 | 1.35 | 0.44 / 1.28 | 0.479 / 2.59 | 5.34e13 |
| meanres 2 circ | 1.60 | 1.43 | 1.40 | 0.41 / 1.34 | – | 5.88e13 |
| meanres 1 pcg3 | 1.76 | 1.48 | **1.16** | 0.42 / 1.09 | **0.360 / 2.05** | 5.58e13 |
| meanres 2 pcg3 | 2.12 | 1.94 | 2.12 | 0.63 / 2.02 | – | 5.34e13 |

- **One meanres step is the best start.** At 12 iterations: median 0.36 (pcg3) and 0.48 (circ) vs 0.68 (pooled) and
  0.84 (zero) m/s; RMSE 1.16 / 1.35 vs 1.59 / 1.71.
- Two pcg3 steps are worse (2.12): the second step adds detail variance that 12 iterations with the prior do not
  remove. This is the bias–variance trade of §10.5, and it argues for 1 step plus the prior.
- The cheap circ variant gets most of the gain at ≈ 1/14 of pcg3's cost.

**(iii) Anchor to the pooled mean** (`mean:μ`, μ in units of d̄ = 1.79e4; λ_max(M) = 1.87e-6, so μ d̄ λ_max(M) =
0.034 μ against λ_max(M N) = 51.3):

| start | anchor | it 12 RMSE (mean / detail) | it 12 median / p90 |
|---|---|---|---|
| pooled x̄ | none / μ 1 / μ 10 / μ 100 | 1.59 / 1.59 / 1.58 / 1.59 (mean 0.71 → 0.79 at μ 100) | 0.684 → 0.695 (μ 100) |
| zero | none / μ 1 / μ 10 / μ 100 | 1.71 / 1.72 / 1.79 / 1.79 | 0.840 → 0.975 (μ 100) |
| meanres 1 circ | none / μ 1 / μ 10 / μ 100 | 1.35 / 1.35 / 1.33 / 1.47 | 0.479 → 0.589 (μ 100) |
| meanres 2 circ | none / μ 1 / μ 10 | 1.40 / 1.64 / 1.20 | – |
| meanres 1 pcg3 | none / μ 1 / μ 10 / μ 100 | 1.16 / 1.16 / 1.20 / 1.51 | 0.360 → 0.637 (μ 100) |

- μ ≤ 10 is nearly inert in 12 iterations (λ_max grows by ≤ 0.7 %).
- μ = 100 (λ_max 51.3 → 54.7) pulls the frame means toward x̄. Mid-run it lowers the detail error (pcg3: 1.59 → 1.47
  at it 3), but by it 12 it makes the result worse: the time-mean error goes 0.42 → 0.80.
- **Reading:** the anchor transmits its target's bias, as Theorem 5 says (bias = the coarse solution's error). Here
  the target is an 8-iteration pooled solve whose time-mean velocity is worse than the free recon's. So this dev
  check does not test the anchor's purpose: variance reduction at undersampled fine levels, with a *trusted*
  coarse level. It does confirm the mechanics: a stable step with the Weyl bound, and an effect that scales with μ.

### 12.3 Implications and open problems
- **Pooled level first.** Both features rest on x̄ being the pooled *solution*. At 8 PCG iterations it is not
  (residual 0.18). Before any anchor experiment, converge the pooled / coarse level, or anchor to a converged coarse
  P²nP run (`--anchor <run dir>:μ`, implemented but untested).
- **Initialiser:** use one meanres step (circ is nearly free; pcg3 better and still < 1 min at dev scale). Do not use
  2+ steps without a detail penalty.
- **Fix the dev score.** The p90 ≈ 3 m/s / median 0.7–0.8 m/s error, present in the pooled start too, has to be
  understood before the dev case can rank temporal methods. Check a plain-NUFFT recon and the decoder's alias search on
  the lumen voxels outside the support (450 of the 6165). Without that, use the median and data-space measures.
- **Anchor test that matches the theory:** a nested pair (10 → 20 frames, or 20 → 40 at full grid) with a converged
  coarse run, where the fine level is undersampled (§11: > 100 intervals per encode). Score it by held-out k-space
  (§6 rule 4), not only by the lumen velocity.
- **Not done:** the anchor in the knot / velocity objective (vobj.rs: R must act on the nonlinear model's knot
  coefficients); meanres in knot mode (coarse-space mean = constant in time on the knot coefficients); the
  k-dependent detail penalty of §6 rule 3.

## 13. A validated velocity score, the cause of the dev case's 0.8 m/s, re-ranked starts, the anchor test (2026-11-22)

**Scripts.** `research/experiments/temporal_pyramid/run_score.sh` (stages smoke / diag / k4 / anchor4),
`run_score_pool.sh` (pooled inverse crime), `pc_native.py` (unpacks the truth), `pcerr.py` (per voxel-frame errors),
`parse_score.py` → `score_results.txt`. Run directories: `data/hasty/phantom/factory_arms/ph0001/tpyr_score/` (scored
recon stacks deleted). GPU ≈ 61 min.

**Common setup.** Data: ph0001 sim_flow_10min, 128 × 72 × 60 grid (4.0 × 4.0 × 3.8 mm), k-space cropped to the grid,
8 virtual coils (68 % of the coil energy), 20 cardiac frames × 7 encodes, one respiratory bin. Operator: full physics
(6 ratemap/B0 segments, concomitant degree 4, GNL). Algorithm: P²nP, ZMSHR1-M (per-encode circulant metric), LLR κ 1.
Pooled start x̄: 60 pooled PCG iterations per encode with the circulant (relative residual 0.02–0.03; §12 used 8,
residual 0.18). Every run holds out 1/8 of the readouts (below). `--keep 16` is the §12 dev case: 2858 solve
readouts, 9–33 per (frame, encode) set; `--keep 4`: 11545, 4× more.

### 13.1 The score (`--score-pc DIR[:COVMIN]`, `hasty_recon::pcscore`)
- **Truth:** the sim's per-frame phase-contrast signal C_{e,f} (truth_pc_frames.npz: LBM voxels, spin history and
  intravoxel velocity spread included), not truth_vel.
- **Resampling: truth → recon grid.** A recon voxel's truth is the complex mean of C over the voxel's box at its true
  (GNL-inverted) position (the `sample_log_box` sub-positions, trilinear on the 2 mm grid). That is the signal-weighted
  partial-volume average the voxel measures.
- **Voxels:** only those where every native voxel in that mean is in the truth with coverage ≥ 0.99 in every frame
  (blood all through the cycle). 974 voxels at this grid; all 974 are in the legacy lumen set (6165 voxels,
  point-sampled lumen fraction > 0.9), 949 in the support.
- **Velocity:** the recon's own decoder (ML + alias search, as reported), applied to both sides. The recon also gets
  the GNL J⁻¹; the truth does not (its phases use the nominal moments). Recon frame f ↔ truth frame f mod 20; with
  `--card-frames K` the truth is averaged over the nested frames.
- **Reported** every report iteration (and `pcscore_<tag>.csv`): RMSE (and relative), time-mean RMSE, temporal-detail
  RMSE, |Δv| median / p90, share with |Δv| > 1 m/s (meta venc = the low VENC), the wrapped phase error of
  arg(x_e x̄_0) against arg(C_e C̄_0) (rad, encodes 1–6; decoder-free), and the same with the hierarchical decoder.
- **The legacy all-lumen print is unchanged.** It is not wrong as code. Its weaknesses are the voxel set (partly
  tissue during the cycle) and truth_vel. Its gap to the PC truth on the 974 voxels is small (below).

**Oracle checks** (printed at setup):

| as the "recon" | RMSE m/s (rel) | time mean / detail | median / p90 | phase rad |
|---|---|---|---|---|
| truth frames | 0.000 (0) | 0 / 0 | 0 / 0 | 0 |
| complex time average of the truth | 0.378 (0.89) | 0.129 / 0.355 | 0.137 / 0.545 | 0.519 |
| zero image | 0.425 (1.00) | 0.234 / 0.355 | 0.131 / 0.698 | 0.574 |
| truth_vel sampled (legacy truth) | 0.148 (0.35) | 0.022 / 0.147 | 0.043 / 0.216 | – |

- The time average's detail error equals the truth's detail RMS (0.355), as it should.
- Its time-mean error (0.129) is not 0: the velocity decoded from the mean signal is not the mean velocity
  (signal weighting, nonlinear decode).
- Zero has relative error 1 by construction. A recon is worth something only below rel 1 / RMSE 0.425.

**Data-space score (`--holdout N[:SEED]`, `--holdout-eval N[:SEED]`, `--cc-load`).**
- Every N-th readout of each (encode, frame) set is held out of the solve, from a hashed offset, so every set keeps
  some. The solve run writes `coil_basis.txt`.
- The eval run keeps only those readouts and uses the same coil basis:
  `--init-from <recon> --init warm,zero --iters 1`. f at it 1 gives ½‖E x − y_h‖² and ½‖y_h‖².
- Reported: √(f(x)/f(0)).
- `--init zero` now overrides `--init-from` (before, it was silently replaced by it).

### 13.2 What produced the 0.8 m/s: the dev case is too undersampled, not the score, noise or physics
At keep 16, the pc score of the converged pooled start x̄ is median 1.76 m/s, 69 % of voxel-frames > 1 m/s, phase
error 1.59 rad. That is close to uniform random phase (1.81). Every 12-iteration recon stays at 1.5–1.7 median,
except the zero start (0.94).

Controls, all scored at it 0 of the pooled start (`pc score` at it 0) or after P²nP:

| data | operator | start / solver | median m/s | phase rad | RMSE |
|---|---|---|---|---|---|
| keep 16 measured (noisy) | own | pooled x̄, 2 PCG iterations (smoke) | 0.696 | 1.26 | 1.63 |
| keep 16 measured (noisy) | own | pooled x̄, 60 PCG iterations | 1.756 | 1.59 | 2.23 |
| keep 16 noise-free measured (`HASTY_NOISE_SIGMA=0`) | own | pooled x̄, 60 | 1.743 | 1.59 | 2.22 |
| keep 16 inverse crime, noise-free (x★ = PD + truth blood) | own | pooled from zero, 60 | 1.757 | 1.59 | 2.21 |
| keep 4 measured | own | pooled x̄, 60 | 0.643 | 1.25 | 1.57 |
| keep 4 inverse crime, noise-free | own | pooled from zero, 60 | 0.674 | 1.25 | 1.58 |
| keep 16 inverse crime, noise-free | own | P²nP from zero, it 12 / 51 | 0.474 / 1.254 | 1.10 / 1.45 | 1.29 / 1.93 |
| keep 4 inverse crime, noise-free | own | P²nP from zero, it 12 / 27 / 30 | 0.391 / 0.209 / 0.388 | 1.01 / 0.76 / 1.03 | 1.16 / 0.73 / 1.23 |

Reading:
- **Not noise.** Noise-free measured data give the same numbers as noisy data (1.743 vs 1.756). The data term at it
  51 is 8.4e12 noise-free vs 9.7e12 noisy.
- **Not model / physics error.** The inverse crime has exact data for the recon's own operator, with a realistic
  object (PD + blood, blood |C| ≈ 0.23 × the body PD). It reproduces the measured numbers to 2 digits at both keeps.
  Breathing, ratemap, coil compression and GNL mismatch therefore contribute nothing visible.
- **Not grid mismatch.** The oracle checks pass, and the inverse crime builds x★ and scores it through the same
  mapping.
- **Not the decoder.** The ML decoder's alias search (±3 m/s design range) roughly doubles p90 and the RMSE against
  the hierarchical decoder (pooled keep 16: hierarchical median 1.24, RMSE 1.43). But the decoder-free phase error is
  already 1.59 rad, so it only magnifies random phases.
- **Not lack of convergence, in the usual sense.** More pooled iterations make it worse: median 0.70 at 2 iterations
  → 1.76 at 60, on measured data. Noise-free crime P²nP from zero also gets worse from it 12 to 51 while f falls.
- **The cause: the problem is badly conditioned at this sampling.** At 1/16 of the readouts (≈ 410 per encode pooled,
  9–33 per set), the error that remains at a 2–3 % normal-equation residual is large. It sits in directions the data
  barely constrain, and it swamps the blood signal (a few % of the image energy) with aliased static tissue.
- At keep 4 the pooled solve is still poor (0.64–0.67 median, against 0.137 for the time-averaged truth). The
  per-frame inverse crime gets to 0.21 median at best and is non-monotone. That is the P²nP step behaviour: RMSE 0.73
  at it 27, 1.23 at it 30.
- Supporting image check (warm.npy, keep 16): its magnitude correlates with the truth PD at 0.02 (warm_full 0.10)
  away from the FOV border, and the border rows are the brightest. There is no anatomy to speak of.

**Consequence:** at keep 16 this dev case cannot rank temporal methods by velocity: every recon is worse than the
zero image. It needs ≥ 4× the data (keep ≤ 4), or fewer unknowns (fewer frames, a coarser grid or a velocity
parametrisation), before velocity differences mean anything. The §12 tables (legacy score, keep 16) measured noise in
this sense. Their "meanres helps" ordering of medians is not supported by the validated score at keep 16 (below).

### 13.3 Re-ranked starts (same P²nP, 12 iterations; pc score)

| start | keep 16: it 0 median / RMSE | keep 16: it 12 median / RMSE | keep 4: it 0 | keep 4: it 12 |
|---|---|---|---|---|
| zero | 0.131 / 0.425 | **0.943** / **1.78** | 0.132 / 0.425 | 0.424 / **1.235** |
| pooled x̄ (60 PCG) | 1.756 / 2.23 | 1.715 / 2.21 | 0.643 / 1.565 | 0.519 / 1.434 |
| meanres 1 step, circ | 1.750 / 2.23 | 1.596 / 2.13 | 0.644 / 1.560 | **0.394** / 1.253 |
| meanres 1 step, pcg3 | 1.761 / 2.23 | 1.703 / 2.19 | 0.660 / 1.566 | 0.631 / 1.558 |
| meanres 2 steps, circ | 1.752 / 2.23 | 1.514 / 2.09 | 0.644 / 1.562 | 0.609 / 1.517 |

- With a converged x̄, meanres barely changes the start (it 0 equal to x̄ within 0.01). (8) adds the residual
  deconvolution of an x̄ that is itself mostly error.
- After 12 iterations at keep 4: zero and meanres-1-circ are the best two (zero best RMSE, meanres best median). pcg3
  and 2 steps are worse than plain x̄.
- **At 51 iterations** (keep 4, meanres 1 circ, no anchor): median 0.379 → 0.387 and RMSE 1.225 → 1.296 from it 12 to
  51 (detail 0.96 → 1.11). It slowly diverges in velocity while the training f falls (7.40e13 → 6.64e13).
- The zero start's 51-iteration trend is only available as the noise-free crime: 0.47 → 1.25 at keep 16, 0.39 → 0.21
  → 0.39 at keep 4 (it 12, 27, 30).
- **No start beats the zero image** (RMSE 0.425) at either keep. The ranking is among recons that are all
  velocity-useless at this scale.

### 13.4 Anchor test (Theorem 5 case B, nested frames, keep 4)
- **Coarse level:** `--card-frames 5` (each coarse frame pools 4 consecutive fine cardiac frames), pooled start, 30
  P²nP iterations. Its own pc score is against the truth averaged over the nested frames (zero image 0.398 RMSE):
  median 0.475 at it 12, 0.516 at it 30, time-mean error 1.21. **The coarse level is not trustworthy.**
- **Fine level:** 20 frames, start meanres 1 circ, 51 iterations. Run with no anchor and with `--anchor <coarse>:μ`,
  μ = 100 and 1000 in units of d̄ (d̄ = 6.73e4; λ_max(M N) 48.6 → 52.3 at μ 100).

| anchor | pc it 12 median / RMSE | pc it 51 median / RMSE (mean / detail) | held-out √(f(x)/f(0)) | training f it 51 |
|---|---|---|---|---|
| none | 0.379 / 1.225 | 0.387 / 1.296 (0.67 / 1.11) | 0.2858 | 6.64e13 |
| coarse, μ 100 | 0.513 / 1.459 | 0.548 / 1.495 (1.09 / 1.03) | **0.2599** | 7.72e13 |
| coarse, μ 1000 | 0.535 / 1.484 | 0.576 / 1.540 (1.03 / 1.15) | 0.2629 | 7.42e13 |

- **Data space:** the anchor lowers the held-out residual by 9 % (0.286 → 0.260) while it raises the training
  residual. It reduces overfitting of the 20-frame level, the variance reduction Theorem 5 promises. 20 frames at keep
  4 is undersampled enough to show it: the unanchored level fits the training data better and the held-out data
  worse.
- **Velocity:** the anchor makes the blood velocities worse (median 0.39 → 0.55). Its time-mean error goes to the
  coarse level's (≈ 1.1). The anchor carries its target's bias (Theorem 5), and this target's blood phases are poor.
- The two scores disagree because the held-out residual is dominated by the static tissue's energy, the velocity
  score by the blood phase (a few % of the energy).
- μ 1000 is no better than μ 100 in either score.
- Not run: a coarse level at a sampling where it is trustworthy for velocity. None of the levels here is: even the
  keep-4 noise-free pooled solve has 0.67 median. 40 frames was not tried (it would make the fine level worse, not
  the test more informative).

### 13.5 Open problems
- **Make the dev case well posed first.** Candidates: keep ≤ 2 (memory: the kept data ≈ 0.7 GiB at keep 4), fewer
  cardiac frames, or the velocity / knot parametrisation (`--vmodel knots`). Check with the pooled noise-free inverse
  crime: the pc score of x̄ must approach the time-averaged truth (0.14 median) before any start or anchor comparison.
- Why the pooled per-encode solve is so poor at keep 4 (1650 readouts per encode, 8 coils). The §11 coverage at ≈ 18
  intervals' worth of readouts per encode predicts uncovered outer shells, but a k-resolved check (error spectrum of
  x̄ − x★ in the crime) is the direct test. Likewise the bright FOV-border rows of the plain warm image.
- The P²nP velocity score is non-monotone over iterations: keep-4 crime RMSE 0.73 at it 27, 1.23 at it 30.
- The held-out eval of a `--card-frames` run needs the same `--card-frames` (the coarse eval failed: 5 recon frames
  vs 20 sets). Untested.
- The legacy all-lumen yardstick stays for comparability. Use `--score-pc` (plus `--holdout`) for decisions.

**New switches** (all default off; no change to existing runs): `--score-pc DIR[:COVMIN]`, `--synth-x pc` (with
`--synthetic` and staged data, e.g. `--plan host`), `--holdout N[:SEED]`, `--holdout-eval N[:SEED]`,
`--cc-load FILE`, `--card-frames K`, `--no-vel-save 1`. Fixes: `--init zero` with `--init-from`; the staged synthetic
path with `--warm 0`.

## 14. The pooled start had no anatomy: an inverse-crime bug and a harmful pooled preconditioner (2026-11-22)
Details, numbers and scripts: **research/recon_diag_2026-11-22.md**. In short:
- **Bug (fixed):** with staged data (`--plan host`) the grouped gradient of pooled PCG subtracted the MEASURED store
  data from E x, so every §13 "inverse crime" pooled solve (`--synthetic --synth-x pc --warm-full`) solved the
  measured problem. That is why crime and measured numbers agreed to 2 digits in §13.2. Regression test
  `grouped_sets::pooled_pcg_synthetic_ignores_staged_data`.
- **Cause of "no anatomy":** the per-encode circulant metric as the pooled PCG preconditioner. On a static noise-free
  crime it makes the image error grow (relative 1.07 → 2.42 → 2.95 at it 1 / 30 / 60, keep 16; also at keep 4 and 2),
  pushing 35 % of the image energy into unsampled k-space corners and leaving the low frequencies unreached. Plain CG
  converges (0.31 at 30 iterations, PD correlation 0.87; 0.17 / 0.93 at 60, keep 2). New `--pool-metric none`.
- **Consequence:** the §12 / §13 pooled starts (warm, meanres, anchor `mean`) were images without anatomy; their
  rankings are void. Use the solvable dev case `research/experiments/pooled_diag/run_dev_solvable.sh` (keep 2,
  plain-CG pooled start): its pooled image scores the time-averaged-truth oracle (median 0.126 vs 0.137 m/s).
