# The prior inside the step: RED for our P²nP / ZMSHR1 (2026-11-13)

User question (2026-11-13): in P²nP, x ← D(x − αP∇f), the iterate moves only through the data gradient, and the
denoiser keeps it "on track". Can the prior's knowledge go into the descent step itself? Short form in
`mathematics_of_paper/reconstruction_operator_section.tex` (subsubsection "The prior inside the step"). This note
gives the details and an implementation plan.

## 1. Iteration

    g_k   = ∇f(x_k) + λ (x_k − D(x_k))                       (RED residual; one denoiser call per gradient)
    x_k+1 = x_k − α P_k g_k                                   (no denoiser after the step)
    P_k   = ZMSHR1(M; s_k = x_k − x_k−1, u_k = α (g_k − g_k−1))   (secant pairs from the TOTAL gradient)

What changes against P²nP:
- The quasi-Newton secants see the prior's curvature: P_k approximates the inverse Hessian of f + λR, not of f
  alone.
- Every iterate is a gradient step on one objective. The denoiser is no longer a projection after the step, so its
  "pull" is a proper step direction scaled by the metric.
- The user's example: for the static-tissue lock (D = Π, the linear projection onto the temporal mean of static
  voxels), x − D(x) = (I − Π)x = ∇ ½‖(I − Π)x‖². That is a quadratic penalty. A Newton step on f + λ/2‖(I − Π)x‖²
  goes to its minimiser in one step; λ → ∞ gives the projection. P²nP gets this part right already by projecting.
  RED makes the same true for the parts of D that are not projections.

## 2. When the residual is a gradient (the fixed points then do not depend on the metric)

1. **Proximal denoisers.** D = prox_{θR} with R convex gives x − D(x) = θ ∇M_{θR}(x), where M is the Moreau
   envelope: convex, with a 1/θ-Lipschitz gradient. RED is then gradient / quasi-Newton descent on the smooth convex
   objective f + λθ M_{θR}.
   - Our LLR with *unshifted* blocks is the prox of a sum of block nuclear norms, so this applies exactly.
   - With a *random shift per iteration*, each shift is a prox of its own shifted block nuclear norm, and the
     expected residual is the gradient of the shift-averaged envelope. RED with a random shift is therefore a
     stochastic gradient of a fixed smooth convex objective. Cycle spinning makes this exact.
2. **MMSE denoisers.** A denoiser trained with squared error at noise σ satisfies Tweedie / Miyasawa:
   D(x) = x + σ²∇log p_σ(x). Then x − D(x) = −σ²∇log p_σ(x), with a symmetric Jacobian (the Hessian of the
   log-density), and RED descends on f − λσ² log p_σ: MAP with the Gaussian-smoothed prior (Kadkhodaie–Simoncelli).
   Our learned denoiser is trained with an MSE loss on 32³ × 12-frame patches, so it is approximately this, up to:
   - the noise-level adaptation;
   - the scale-equivariant normalisation;
   - the tiling.
3. **Otherwise** no R with ∇R = x − D(x) exists when the Jacobian is asymmetric (Reehorst–Schniter, Thm 1). They
   measured ‖J − Jᵀ‖²_F / ‖J‖²_F = 0.017 for DnCNN and 1.2 for BM3D; RED is then a fixed-point equation
   (score-matching interpretation).
   - A *gradient-step denoiser* D = I − ∇g_θ (Hurault–Leclaire–Papadakis, ICLR 2022) is a learned denoiser with an
     exactly conservative residual and no loss in denoising quality. That is the clean route if we retrain.

Consequence for the PnP paper (`pnp_restricted/paper.tex`, prop:estimator / prop:metricfp): there the P²nP fixed
point depends on the preconditioner P. For a RED gradient that is an exact gradient, the stationary points of
f + λR do not depend on P. P then changes only the speed, which is the property we want from a preconditioner.

## 3. Strength and step size

- Matching P²nP's fixed point: P²nP with prox_{θR} and P = τI is stationary for f + (θ/(ατ)) R (prop:metricfp).
  RED with the envelope matches it with λθ ≈ θ/(ατ), i.e. λ ≈ 1/(ατ). This gives a starting value, then sweep.
- Lipschitz: x − prox has Lipschitz constant ≤ 1, so the λ term adds ≤ λ to the curvature. The data term's
  λ_max(N) = 2.5·10⁹ (2 mm, full model), so in the circulant metric the prior's weight must be expressed in the same
  units: λ_RED ≈ c · λ_max(MN)/(per-set scale). [CHECK when implementing: work in units of the metric, as α and τ
  already do.]

## 4. Our denoiser components under RED

| component | D | residual x − D(x) | RED objective term |
|---|---|---|---|
| static-tissue lock | Π (temporal mean on static voxels) | (I − Π)x | λ/2 ‖(I − Π)x‖² (quadratic, exact) |
| cardiac low-pass (`--prior tlp`, K) | BBᴴ per voxel, encode and bin | (I − BBᴴ)x | λ/2 ‖(I − BBᴴ)x‖²: soft band limit, i.e. a weighted harmonic decay with weight 0 inside the band and λ outside; generalises to w_k |ĉ_k|² |
| LLR (unshifted / random shift) | SVT at θ | θ∇M_{θR} | λθ M_{θR} (smooth convex) |
| learned (MMSE) | network | ≈ −σ²∇log p_σ | ≈ −λσ² log p_σ (check Jacobian symmetry) |

The quadratic rows can go exactly into the metric if wanted (M⁻¹ + λ(I − Π), still cheap); the user prefers not to
build them into the metric (options 1 and 2 of 2026-11-13). Under RED they are simply gradient terms, and the
quasi-Newton secants take up their curvature.

## 5. Implementation (built 2026-11-13)

Built as a wrapper objective, not inside the loops: `hasty_recon::red::Red { inner, denoise, lambda }` implements
`Objective` with `gradient_host` / `gradient` = inner gradient + λ(x − D(x)) (one scratch host stack; the device path
downloads x once more). `run_objective_host` and `run_objective` run unchanged with an identity denoiser, so ZMSHR1 sees
the total gradient automatically and checkpoints / resume are untouched.

- recon_p2np: `--prior-mode pnp|red`, `--red-rho ρ` (default 1), `--red-shifts n` (LLR: fixed shift seeds 0..n,
  averaged; the per-iteration random shift is off in RED mode because it makes the gradient stochastic). Tag `_red{ρ}`.
- λ = ρ/(α μ), μ = `red::metric_gain` (Hutchinson tr(M)/n on a fixed ±1 vector over four sets; 1 without a metric).
  Deterministic, so a resumed run gets the same λ. With zmshr1s (no metric) λ = ρ·λ_max(N).
- Warning printed for κσ̂ LLR without `--llr-rank` (σ̂ re-estimated per call: the objective moves).
- Toy test `tests/red.rs` (soft threshold = prox of θ|·|, closed-form stationary point): none / zmshr1 / zmshr1s /
  circ / zmshr1ms and the device loop all reach it to rel ≤ 1.7e-7; zmshr1 in ≤ 50 iterations vs 300 for plain
  steps; P²nP with the same denoiser stops 95 % away (its fixed point depends on α).
- Found on the way: global (not per-set) ZMSHR1 returned NaN when the secant vanishes (f32 stagnation); now guarded
  like the per-set scalars.
- **Secant groups.** A prior in the gradient couples sets: tlp couples the frames of one (bin, encode), LLR couples
  everything. The per-set ZMSHR1 (zmshr1s / zmshr1ms) assumes H = blockdiag H_t, and on coupled sets it diverges
  (seen with the temporal subspace: NaN by it 40). `Objective::secant_groups` gives a group id per set, and the
  coupled sets share one (a, τ, c). recon_p2np derives the groups from the active denoiser parts by union of
  partitions:
  - tlp → (bin, encode);
  - LLR → all sets;
  - learned → bin;
  - DFW / magnitude coupling → frame;
  - static lock → encode.
- **Temporal subspace** (`hasty_recon::tsub`, `--tsub fourier:K | bspline:M[:P]`): x = T c as a second wrapper.
  RED composes inside it, giving the gradient Tᴴ(∇f + λ(x − D(x))).

Original plan (for reference):
- `--prior-mode pnp|red` (default pnp) and `--red-lambda λ` (default from §3).
- `run_objective_host`:
  - after `gradient_host(x, g)`: copy x → d (one extra host stack, 8.7 GiB at 2 mm; or chunk it where D is local:
    static lock, tlp);
  - `denoise(k, d)`;
  - g += λ(x − d);
  - skip the `denoise(k, x)` after the step.
  - The ZMSHR1 pair (s, u) uses the total g automatically.
  - Report f + λR where R is known (quadratic terms, Moreau envelope via ½‖x − D(x)‖²/θ + R(D(x))); otherwise f and
    ‖x − D(x)‖.
- Tests:
  - a toy where D = prox of ‖·‖₁: RED must reach the minimiser of f + λθ·Huber, independent of P (identity /
    circulant / ZMSHR1);
  - the quadratic Π case against the closed form.
- Measure the learned denoiser's Jacobian asymmetry with JVP/VJP on a few patches (Reehorst–Schniter metric) before
  relying on case 2.

## 6. Open

- Our LLR uses κσ̂ with σ̂ from the iterate itself (it changes every iteration). A fixed θ is needed for a fixed
  objective; freeze σ̂ after a few iterations.
- With the temporal subspace x = Tc (the velocity model's B-splines), the RED gradient is Tᴴ(∇f + λ(Tc − D(Tc))).
  The denoiser still works on frames; the chain rule is just Tᴴ.
- Convergence theory for ZMSHR1 on f + λR with a nonconvex learned R: the restricted analysis of the paper should
  carry over with the denoiser Jacobian replaced by I − λ ∂²R. [research]

Sources:
- [Romano, Elad, Milanfar, RED (SIAM J. Imaging Sci. 2017)](https://arxiv.org/abs/1611.02862)
- [Reehorst & Schniter, RED: clarifications and new interpretations (IEEE TCI 2019), arXiv 1806.02296](https://arxiv.org/pdf/1806.02296)
- [Cohen, Elad, Milanfar, RED-PRO, arXiv 2008.00226](https://ar5iv.labs.arxiv.org/html/2008.00226)
- [Hurault, Leclaire, Papadakis, Gradient Step Denoiser for convergent PnP (ICLR 2022)](https://openreview.net/pdf?id=fPhKeld3Okz), [code](https://github.com/samuro95/GSPnP)
- [Hurault et al., Proximal denoiser for convergent PnP with nonconvex regularization (ICML 2022)](https://proceedings.mlr.press/v162/hurault22a/hurault22a.pdf)
- [Overview: from image denoisers to regularizing inverse problems, arXiv 2509.03475](https://arxiv.org/pdf/2509.03475)
