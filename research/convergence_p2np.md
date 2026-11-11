# Convergence of P²nP on centre-out 3D 4D flow: why it is slow, what is reachable, estimator-preserving acceleration

2026-11-10. Context: the 2 mm ph0001 runs after the recon audit (rustlib/PLAN.md, "Recon audit"). Numbers marked
*measured* come from those runs or from the scan's trajectory. Toy numbers come from
`research/experiments/convergence/{rates_toy,lambda_sweep}.py`: a 64×64 phantom, 4 coils, Haar-ℓ1 prior, k-density
centre/median ≈ 600.

## 0. Summary

1. **The cost is set by FFTs, so iterations are the budget.**
   - *Measured:* one gradient pass over 280 sets takes 480 s. Per set that is about 1.7 s, of which gridding is
     0.09 s; the rest is 2·L·C = 1364 FFTs (L = 22 segments, C = 31 coils, 8.2 M-point grid).
   - *Estimated, full scale:* 512×512×384 with 20 cardiac × 10 respiratory frames costs about 15 h per gradient pass
     on one 4090 (§1). The fine level can afford only tens of passes even on 8 GPUs.
2. **Slow from the first iteration, and the reason is the trajectory.**
   - *Measured:* the centre-out 3D yarn has sample density ∝ 1/k² near the centre. Density spread: 386 with 25
     radial bins, 1.2·10⁴ with 100, 3.8·10⁵ with 400; at k-voxel resolution (≈ 57 bins), 10³–10⁴.
   - λ_max(H) is set by the oversampled centre, while ≈ 90 % of the k-radius has nearly flat density 10⁻³–10⁻⁴ of it.
     Any 1/λ_max-scaled step therefore advances the bulk of k-space by ≈ 10⁻³ of its misfit per iteration.
   - *Measured:* ZMSHR1's scalar is capped by the same curvature (τ ≤ BB2; τ_t ≤ 9/λ_max), and its rank-one term fixes
     one direction per set and iteration.
   - The time-pooled warm start has already removed the fast components. The solver starts in the slow tail, so there
     is no early big drop.
   - A 2D interleaved spiral (P²nP's setting) has density ∝ 1/k, ratio ~N/2. The 3D centre-out ratio is roughly its
     square, which is why P²nP-D was fast there and is slow here.
3. **The PnP estimator is not algorithm-invariant as implemented.**
   - The LLR threshold is θ = κσ̂ (from the iterate), independent of the step. The fixed point solves
     0 ∈ ∇f + (θ/(α τ_t)) ∂R: the effective regularisation weight scales with 1/step and differs per set (τ_t spreads
     16×, *measured*) and per iteration.
   - In-tangent shrinkage is amplified by (αΠ₀PHΠ₀)⁻¹ (paper, Prop. 6.1(4)). *Measured:* the lumen velocity slope at
     iteration 10 is 0.41, with a temporal-deviation correlation of only 0.2–0.25.
   - *Toy:* the fixed-strength PnP fixed point moves by 16–18 % between α = 1/L and α = 0.5/L.
4. **The regularisation weight, not the weighting, decides accuracy.**
   - *Toy, λ tuned per estimator:* the unweighted (BLUE data term) and density-weighted minimisers reach the same error
     to truth (0.089). Their optimal λ differ by about 300× in λ_max-normalised units, because λ_max is
     centre-dominated.
   - *Measured:* density compensation would cost only 4 % SNR on this trajectory (η = 0.93, variance ×1.08). Only the
     few central samples get small weights.
   - Consequence: κ = 1 is very likely orders of magnitude too strong for the bulk of k-space. Expect the optimum well
     below 0.25; extend the sweep to 0.01–0.1.
5. **Estimator-preserving acceleration exists; ranked in §5.**
   - (i) LLR as a proper proximal map, with the threshold tied to the step, so that the estimator is the minimiser of
     f + λR_LLR whatever the algorithm.
   - (ii) FISTA momentum: √κ instead of κ. *Toy:* 82 vs 1001 normal-operator applications to 10 %; 301 to 1 %, where
     ISTA never gets there.
   - (iii) ADMM / HQS with the data step solved per set by PCG. Any preconditioner (circulant, DCW-like) is then
     allowed, because an exact inner solve does not change the estimator. *Toy, accurate λ:* 288 applications to 1 %
     and 504 to 0.1 %, against 1091 / — for FISTA. **The best scheme found.**
   - (iv) A spatial × temporal pyramid, where coarse levels are well conditioned and cheap.
   - (v) A density-compensated adjoint A^H W y, least-squares scaled, as a one-pass initialisation. It changes only the
     path, not the estimator.

## 1. Cost model

Per gradient pass, sets × (gridding + 2·L·C FFTs on the oversampled grid):
- *Measured* at 2 mm: 280 × (0.09 s + ≈ 1.6 s) ≈ 480 s.
- FFT cost ∝ (number of images) · L · C · n log n. Gridding ∝ samples · L · C · w³ and is independent of the frame
  count.
- Full scale: 512×512×384 (σ = 1.25: 640×640×480 = 1.97·10⁸ points, 24× the 2 mm grid; ×1.2 for log n) and
  20 × 10 frames × 7 encodes = 1400 sets (5×). The FFT part is ≈ 0.8 · 480 s · 29 · 5 ≈ 5.6·10⁴ s ≈ 15 h per pass on
  one GPU. Gridding stays ≈ 1–2 min.
- A level that pools frames into G groups costs G/sets of the FFT part. A level on a grid coarsened by s per axis
  costs s⁻³ of it, and only the samples inside its band (≈ 15–25 % at s = 2).

## 2. Spectrum of the data step

Normalised so that α λ_max = 1. A component of the error at k-space radius |k| has curvature ∝ ρ(k), the local sample
density (coil sensitivity is flat after the Σ|S|² = 1 normalisation). Per step with P = τI it contracts by
1 − τ α ρ(k)/ρ_max.

- With ρ(k)/ρ_max ≈ 10⁻³–10⁻⁴ over the bulk and τ ≤ 9 (*measured*), the bulk contracts by ≤ 1 − 10⁻² per iteration:
  ~10²–10³ iterations to converge, against 10 or 20 run.
- ZMSHR1 adds one exact secant direction per set and iteration. It does not change the rate on the remaining
  spectrum; BB-type methods are R-linear with an O(κ) rate.

## 3. Rates of the candidate schemes (local, on the tangent space T)

Write κ_T for the restricted condition number of PH on the tangent space of the prior.

| scheme | iterations to ε (order) | estimator |
|---|---|---|
| PnP-ISTA / P²nP (scalar or ZMSHR1) | κ_T log(1/ε) | fixed point depends on α and P if the denoiser strength is fixed |
| FISTA / heavy ball (prox scaled with the step) | √κ_T log(1/ε) | minimiser of f + λR (unchanged) |
| PnP-ADMM / HQS, data step by PCG with preconditioner M | outer O(1)–O(√κ_ρ); inner √κ(M(H + ρI)) | ADMM: minimiser of f + λR; preconditioner M irrelevant to it |
| FISTA on the density-weighted data term | ≈ κ(A^H W A) = O(1–10) | different estimator (weighted LS; variance ×1/η) |
| spatial × temporal pyramid + any of the above | fine level starts close; few fine passes | that of the fine-level scheme |

Toy, measured in normal-operator applications (cost units), distance to the exact minimiser at λ = 5·10⁻⁴:

| method | to 10⁻¹ | to 10⁻² | to 10⁻³ |
|---|---|---|---|
| ISTA | 1001 | — | — |
| FISTA | 82 | 301 | 1025 |
| ADMM-PCG, ρ = 0.02, 5 inner | 120 | 750 | — |
| ADMM-PCG, ρ = 0.05, 5 inner | 306 | — | — |

At the accurate λ = 10⁻⁵ (best error to truth; `out_8000_lam1e-5_rho`), the reference is converged to < 10⁻⁵:

| method | to 10⁻¹ | to 10⁻² | to 10⁻³ |
|---|---|---|---|
| ISTA | — (0.34 after 1500) | — | — |
| FISTA | 415 | 1091 | — |
| ADMM-PCG, ρ = 3·10⁻³, 3 inner | 264 | 932 | — |
| ADMM-PCG, ρ = 10⁻³, 3 inner | 200 | 548 | 1100 |
| ADMM-PCG, ρ = 3·10⁻⁴, 3 inner | 304 | 696 | 1172 |
| **ADMM-PCG, ρ = 3·10⁻⁴, 8 inner** | **153** | **288** | **504** |

- **Estimator.** ADMM-PCG keeps it: the circulant preconditioner acts only inside the exact data-step solves.
- **Speed.** It reaches 1 % about 4× faster than FISTA and at least 5× faster than ISTA, which never gets there within 1500.
- **Accuracy.** After 1500 applications ISTA's error to the truth is 0.37, against 0.089 for the converged estimator.
  That is the regime of our 10–20-iteration runs.
- **Why this regime is harder.** It is the accurate (weakly regularised) one: the ill-conditioned bulk is no longer
  hidden by strong shrinkage.
- **Expected in 3D.** The toy's density ratio (≈ 600) is far below the 3D yarn's 10³–10⁴, so preconditioned inner
  solves should gain more in 3D than here.

## 4. What the estimator is, and which changes keep it

- **Variational form.** F(x) = ½‖Ax − y‖² + λR(x).
  - Proximal-gradient methods with prox_{αλR}, and their accelerated and variable-metric versions with the prox taken
    in the metric, converge to argmin F. Step, momentum and preconditioner change only the path.
  - ADMM with exact (or convergent inexact) data-step solves also converges to argmin F, whatever preconditioner the
    inner CG uses.
- **Fixed-strength PnP** (our LLR with θ = κσ̂). The fixed point x = D_θ(x − αP∇f(x)) is argmin of f + (θ/α)R_P, with
  R_P the regulariser in P's metric.
  - It depends on α and P. With per-set P_t = τ_t I and a prior coupling the sets (an LLR block spans all frames and
    encodes), each set effectively gets its own regularisation weight θ/(ατ_t).
  - In the local affine model (paper, Prop. 6.1) the in-tangent bias is (αG_P)⁻¹Π₀ε, i.e. the shrinkage times κ_T.
- **DCW in the data term** changes the estimator to weighted least squares: unbiased, variance ×1/η. Here η = 0.93
  (*measured*), so the variance penalty is small. But the λ that is optimal for it is ≈ 300× larger in normalised units
  (*toy*), so λ cannot be carried over between the two.
- **Initialisation** (A^H W y, warm start, a pyramid level) never changes argmin F. With fixed-strength PnP it changes
  which fixed point is reached only where the denoiser is non-convex.

## 5. Recommendations (ranked)

1. **LLR as a proximal map** (no new estimator, just consistent):
   - θ_t = λ · (α τ_t), the step actually taken.
   - For a block whose columns come from sets with different τ_t, the exact prox in the blockwise-scalar metric is a
     column-weighted SVT. Either use one step per block (the sets an LLR block couples: all of them, so a global τ
     for the prox) or use ADMM (point 3).
   - Cycle spinning makes R depend on the iteration. Use a fixed set of shifts averaged per iteration, or a fixed
     shift per outer cycle, before adding momentum.
2. **ADMM with per-set PCG for the data step** (best in the toy, estimator-preserving):
   - The data step (H_t + ρI)x_t = A_t^H y_t + ρ(z_t − u_t) is separable over sets, so it parallelises (multi-GPU
     C3).
   - Precondition the inner CG by the per-encode circulant (already built, host-resident): DCW-like conditioning
     without touching the estimator.
   - The LLR prox acts on z with threshold λ/ρ (exact).
   - Cost per outer iteration = inner iterations × one forward + adjoint per set. *Toy at the accurate λ:* ρ ≈ 30 λ,
     8 inner, 288 applications to 1 % (FISTA 1091, ISTA ≫ 1500). ρ must scale with λ: ρ tuned for a 50× larger λ
     stalls.
   - With the grouped-set operator, a coarse temporal level costs groups/sets of the FFTs per inner iteration.
3. **FISTA momentum** on the consistent prox: the cheap fallback, √κ_T instead of κ_T. Keep a restart rule
   (function-value or gradient restart, O'Donoghue–Candès 2015). It needs a fixed set of LLR shifts.
4. **Pyramid:**
   - Temporal: grouped sets, so the FFT cost ∝ groups.
   - Spatial: a coarse grid and only the central samples. It is well posed because of the 10³–10⁴ central
     oversampling.
   - Run the coarse levels to convergence and do few fine passes. Coarse levels also give the air map early.
5. **Initialisation by the density-compensated adjoint** (A^H W y, scaled by the least-squares factor ⟨Ax, y⟩/‖Ax‖²):
   one adjoint pass, a quality comparable to gridding with density compensation. It replaces the 96-min pooled
   warm start as the starting point of the coarse level.
6. **κ / λ:** sweep down to 0.01–0.1 (queued: 0.05 and 0.25 against the κ = 1 run). With a consistent prox, λ becomes
   comparable across algorithms and resolutions.

## 6. Open theory

- Local rate of accelerated PnP with a non-symmetric denoiser Jacobian. Momentum needs a real, positive spectrum of
  J_D(I − αPH) on T; the paper's Assumption 4.1 gives this for a symmetric S_k.
- ADMM-PnP with a learned denoiser: fixed point in terms of the denoiser strength σ and ρ; the estimator as a function
  of (σ, ρ) only.
- Optimal λ scaling across pyramid levels: λ per level from the noise level of that level's data (σ_ℓ² ∝ 1/samples),
  so that the coarse solutions are consistent with the fine estimator.
