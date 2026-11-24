# A fast preconditioner that does not change the estimator (2026-11-22)

**User question (2026-11-22).** Preconditioning is essential for fast solves on the large real problems. We use it
instead of DCW because DCW changes the estimator and a preconditioner should not. Yet the per-encode circulant metric
makes the pooled image error *grow* (recon_diag_2026-11-22.md §2: rel error 1.07 → 2.42 in 30 iterations, while plain
CG reaches 0.31). Why can a preconditioner do that? And what fast-to-apply preconditioner keeps the estimator? The
dynamic (ZMSHR1) preconditioner was cheap; did the research show it was not enough?

Labels: **[D]** derived here; **[M]** measured (source given); **[H]** hypothesis, to be measured (§5).

## 0. Short answers

1. **Dynamic rank-one alone was not enough [M, D].** In convergence_strategy.md §0.1 (Theorem K), every method whose
   iterates lie in x₀ + 𝒦_k(PN, P r₀) has a larger N-norm error than PCG in the same metric P. That class includes
   GD, BB, ZMSHR1, Nesterov, Anderson and L-BFGS with H₀ = cP. On the toy, reaching 1 % took: ZMSHR1 in the identity
   metric 150 iterations, PCG in the identity metric 85, PCG in the circulant metric 19. Without density correction
   the yarn spectrum spans ~10⁶. A rank-one update fixes one direction per iteration, so the metric has to carry the
   density. Dynamic terms are still useful *on top of* a good metric (§3, P3).
2. **A preconditioner leaves the estimator unchanged only if the objective has a unique minimiser and we converge
   [D, §1].** Our pooled least-squares problem has exact null directions (unsampled k-space corners: the ball is 52 %
   of the grid's modes, recon_diag §1) and near-null directions. There PCG from x₀ converges to the solution closest
   to x₀ *in the preconditioner's geometry*. Early stopping then makes P an implicit regulariser. Our circulant has
   gain 1/ε = 1000 relative to the k-space centre on 80 % of the modes [M], so in those directions it barely
   penalises energy.
3. **The circulant also does not match the operator [M, H].** It is built from the nominal sample positions with
   unit weights (`precond.rs`: `build_kernels` with `plain()` and `wt = 1`, convolved with the coil power and scaled
   by D = Σ_c|S_c|²). The ratemap/B0 and concomitant phase over the readout are left out. λ_max(M⁻¹N) is 51 with
   full physics vs 9.8 plain [M]. [H] Off-resonance spreads the energy of the dense k-space centre into regions where
   the nominal density, and so M, sits at the floor.
4. **What to build (§3).**
   - Make the objective strictly convex, with explicit curvature μ on the (near-)null space. Then set the
     preconditioner floor to *that* curvature, not to an arbitrary ε.
   - Build the circulant symbol from the actual full-physics operator (Chan's optimal circulant, estimated by
     probing N).
   - Add a cheap dynamic part: deflate the few outlier eigen-directions with Ritz vectors recycled from earlier
     solves. This generalises the ZMSHR1 secant. All three cost about one FFT pair of a single image per application.

## 1. When does a preconditioner change the estimator? [D]

Setting. f(x) = ½‖Ex − y‖² + ½ xᴴ R x with N = EᴴE, R ⪰ 0 a quadratic (or locally quadratic) prior curvature,
b = Eᴴy, Q = N + R. PCG with a fixed SPD operator P (≈ Q⁻¹, applied as z = P r) from x₀.

**Proposition 1 (invariance).** If Q ≻ 0 on the space of unknowns, then for every SPD P the PCG iterates converge to
x̂ = Q⁻¹b. The limit does not depend on P or x₀. (Standard. In exact arithmetic PCG is CG on P^{1/2} Q P^{1/2}, which
is SPD.)

**Proposition 2 (singular Q: the limit depends on P).** Let 𝒩 = null(Q) ≠ {0} and b ∈ range(Q). Then
x_k ∈ x₀ + P·range(Q) for all k, and the PCG limit is

    x_∞ = argmin { ‖x − x₀‖_{P⁻¹} : Q x = b }.

*Proof.* The Krylov space 𝒦_k(PQ, P r₀) ⊂ P·range(Q) because r₀ = b − Qx₀ ∈ range(Q). Write x_∞ − x₀ = P Q w. For
any n ∈ 𝒩: ⟨x_∞ − x₀, n⟩_{P⁻¹} = ⟨Q w, n⟩ = ⟨w, Q n⟩ = 0. So x_∞ − x₀ is P⁻¹-orthogonal to 𝒩, which is the
optimality condition of the minimum-P⁻¹-norm problem. ∎

- Plain CG (P = I) from x₀ = 0 gives the minimum-ℓ2-norm solution: no energy in 𝒩.
- P = C⁻¹ (diagonal 1/(λ_k + ε) in k-space) maps range(Q) into itself only if 𝒩 is spanned by Fourier modes.
  - Plain single-coil model: 𝒩 = the unsampled modes (corners), and P keeps them empty.
  - With coils, the image-space factor D^{-1/2} and the physics, 𝒩 is not k-aligned. P then mixes range directions
    into 𝒩, and the P⁻¹-norm (weight λ_k + ε) is cheapest exactly where λ_k ≈ 0. So the limit carries energy in 𝒩.
    Measured: 35 % of the image energy outside the sampled ball after 30 iterations [M].

**Proposition 3 (near-null directions and early stopping).** Let Q have eigenpairs (σ_j, v_j) with some σ_j tiny but
nonzero.
- PCG minimises the Q-norm error over the Krylov space, so ‖e_k‖_Q falls monotonically. The ℓ2 error does not have
  to.
- A component of e along a direction with Rayleigh quotient σ ≈ 0 costs almost nothing in the Q-norm. Its size after
  k steps is set by how P weights it (Proposition 2 with "≈").
- Hence the measured ℓ2 error can grow (1.07 → 2.95) while the data residual falls.
- Early-stopped CG is a spectral filter in the ℓ2 geometry (Landweber-like: near-null directions stay near x₀).
  Early-stopped PCG is the same filter in the P⁻¹ geometry. With P ≈ Q⁻¹ that filter is nearly flat: we reach the
  *unregularised* LS solution, including its ill-posed directions, quickly.

**Consequence.** "The preconditioner does not change the estimator" is a statement about a *strictly convex*
objective solved *to convergence*. Our pooled problem relied on plain CG's implicit regularisation (min-norm plus
early stopping) without saying so. A good preconditioner removes that implicit regularisation, so the explicit
estimator has to carry it.

Two clean ways out, both estimator-level choices (made once, independent of the preconditioner):
- (a) **Restrict the unknowns to the identifiable space.** Band-limit to the sampled ball, so the corners are
  unknowns no more. Support-mask the object (only from data: the prescription box and coil support, real_data_only).
- (b) **Add explicit curvature μ on what remains weak.** A ridge μ‖x‖², the prior's curvature, or the temporal
  anchor of temporal_pyramid.md Theorem 5. With Q = N + R ≻ 0, Proposition 1 holds, and *any* SPD P is safe given
  convergence.

**Rule for the preconditioner floor.** The floor ε (currently 1e-3 λ_max, a convergence knob) should be the symbol of
R, i.e. P ≈ (C + R̂)⁻¹. That is item 2 of convergence_strategy.md: the metric inverts the sum of data and prior
curvatures. Its gain in weak directions is then 1/μ by construction, and μ is a modelling choice that the held-out
k-space score can tune.

**With a denoiser (P²nP / PnP).** The fixed point of a fixed-strength PnP iteration depends on the metric
(convergence_p2np.md, algo_speed.md §"fixed-θ"). Two versions keep the estimator:
- RED / Moreau-envelope objectives solved by a metric method (convergence_strategy.md §0.4).
- ADMM / HQS with exact data-step solves, where P lives only inside the inner PCG.

Propositions 1–3 apply to the inner data step. Its quadratic is N + ρI, which is strictly convex, so with ρ > 0 the
inner estimator is P-invariant automatically. That is a structural argument for ADMM-PCG on our problem.

## 2. Why the current circulant mismatches the operator [M, H]

- Built from nominal k-space positions, unit weights, coil power folded in (λ ⊛ |Ŝ|²), D = Σ_c |S_c|²,
  ε = 1e-3 λ_max. 80 % of the modes sit at the floor on the dev grid [M, `HASTY_PRECOND_DIAG`]: the yarn density
  falls ~1/k², so the outer sampled shells *and* the corners are floored.
- With full physics, λ_max(M⁻¹N) = 51 (vs 9.8 plain) [M, Lanczos]. Ratemap + concomitant without GNL gives the same
  failure (2.40), GNL alone does not (0.97) [M].
- [H] **Smearing.** A voxel with off-resonance Δf sees its samples effectively moved along the readout. A small
  fraction of the centre's density (10³× the edge) moved outward exceeds the floor there by ~50×, which matches 51.
  Test: the k-space and image-space location of the top Ritz vectors of M⁻¹N, and λ_max(M⁻¹N) as a function of field
  scale (the phantom field tails reach ±750 Hz, phantom_field_tails.md).
- [H] **D^{-1/2} outside the coil support.** D = Σ|S_c|² is small in air and at the edges of the coil maps. D^{-1/2}
  amplifies there, which makes the mixing into 𝒩 of Proposition 2 worse. Test: air energy with D floored vs not.
- The plain model also loses (0.92 vs 0.20 after 30 iterations) [M]. So the null-space effect of §1 exists even
  without the physics mismatch.
- Tension with recon_diag_2026-11-12.md: there the circulant gave a sharp PSF and slope 0.96, and its absence a broad
  PSF. That was a plain model with all frames pooled (~16× the samples per encode of keep 16), judged by PSF and deep
  lumen slope, not by image error. Both can be true: the circulant speeds up the well-sampled part and pollutes the
  weak part. §5 E1 measures the two separately.

## 3. Candidates (all SPD, fixed or safeguarded, cost ≈ one single-image FFT pair per application)

For scale: one 256³ one-set gradient is ~391 ms, of which FFT is 169 ms (10 coils × 10 segments). One single-image
FFT pair on the same grid is a few ms, i.e. ≲ 2 % of a gradient.

- **P0. Explicit estimator + matched floor** (§1 (a)+(b)).
  - Band-limit to the sampled ball or support-mask the object, plus ridge μ.
  - P = D^{-1/2}(C + μ̂)⁻¹D^{-1/2} with the floor set from μ, not ε. D floored at a fraction of its body median.
  - Cost: none.
- **P1. Physics-aware Chan circulant.**
  - λ(k) = (F Q Fᴴ)_kk / n, the optimal circulant of the *actual* operator: segments, ratemap, concomitant, GNL,
    coils.
  - Estimated by diagonal probing: λ̂ = mean_j conj(ξ_j) ⊙ F Q Fᴴ ξ_j, with ξ_j random unit phases in k. For a nearly
    shift-invariant Q the off-diagonals are small, so the variance is low. Then smooth the symbol (it is smooth by
    construction).
  - J probes cost J pooled normal-operator applications, once per encode. They are reused by all frames, knots and
    iterations (stored symbol: 1 grid, host).
  - Optional exact route: the segment-pair Toeplitz kernels of the normal operator (normal_op_coil_factorization.md)
    folded to the N grid.
- **P2. Image-space diagonal from the real operator.** D = diag(Q) by the same probing in image space, including the
  ratemap decay energy per voxel and the prior curvature (Jacobi part of D^{-1/2} C⁻¹ D^{-1/2}). Cost: none per
  application.
- **P3. Deflation with recycled Ritz vectors** (the dynamic part, generalising ZMSHR1).
  - PCG's α, β give Lanczos tridiagonals for free. Take r Ritz pairs (θ_i, u_i) of PQ: the outliers, top and
    bottom.
  - Limited-memory preconditioner (Gratton–Sartenaer–Tshimanga 2011): P' = (I − U Uᴴ Q) P (I − Q U Uᴴ) + U Θ⁻¹ Uᴴ
    (P-orthonormal U). Simpler spectral variant: P' = P + U(Θ⁻¹ − I)Uᴴ.
  - *Correction (2026-11-22, [D], §6.2):* U must be P⁻¹-orthonormal (the Ritz vectors of PQ are), and Θ⁻¹ − I maps θ_i
    to 1, which is right only if the bulk of PQ sits at 1. Implemented: P' = P + U(Θ_c Θ⁻¹ − I)Uᴴ, θ_c = the nearest
    edge of the range of the Ritz values not deflated. The Ritz problem must be solved over the *complex* span of the
    Lanczos vectors, so that u and i·u are deflated together.
  - SPD, so §1 holds. Apply cost: 2r inner products + axpys on one image (r ≤ 32).
  - Memory: r images. At 1 mm this goes on the host or at a coarse level.
  - Harvested from the pooled solve, reused by every frame solve and outer iteration. Can be refreshed every few
    outer iterations with a safeguard (keep P' fixed within one inner solve).
- **P4. Spatially blended circulants** (only if P1 leaves spatial outliers).
  - P = Σ_j W_j^{1/2} C_j⁻¹ W_j^{1/2}, with W_j a partition of unity over field bins (ratemap value) or regions.
  - J FFT pairs per application, J ≤ 4.
- **Not candidates:**
  - polynomial preconditioners in Q (one Q application per degree: too slow, user 2026-11-22);
  - per-set spatial circulants (convergence_strategy.md §0.3);
  - DCW in the data term.

## 4. What "good" means (pre-registered)

- **Q1 (estimator).** With P0's explicit estimator, all preconditioners reach the same image when run to convergence
  (relative difference ≤ 1e-3). This checks Proposition 1 in practice.
- **Q2 (speed).** Iterations to 1 % and to 0.1 % of that converged image, and to the oracle-best image error vs x★.
  Measured on the pooled static and dynamic inverse crime and on per-frame P²nP.
- **Q3 (spectrum).** λ_max, the λ distribution and the effective condition number of PQ on range(Q), via Lanczos.
  Target κ ≲ 20, vs the current 51/ε-limited value.
- **Q4 (cost).** ms per application and memory, at dev scale and extrapolated to 256³ / 1 mm.

**Decision.** The default preconditioner becomes the cheapest variant that meets Q1 and wins Q2 over plain CG by at
least 3× in iterations to 1 %. If none does, keep plain CG for the pooled solve and report why.

## 5. Experiments (dev scale, noise-free inverse crime first)

Dev case: `research/experiments/pooled_diag/run_dev_solvable.sh`, keep 2 = 26 674 readouts, 162–214 per (frame,
encode) set, 8 virtual coils, 128 × 72 × 60, kcrop. Static (`--synth-x pd`) and dynamic (`--synth-x pc`) objects.

- **E1. Diagnose.** For the current circulant (full and plain), take Lanczos top/bottom Ritz vectors of PQ. Report
  where their energy lies in k (shell, inside/outside the ball) and in image space (air, body, high-field voxels).
  Also give λ_max vs field scale (0, 0.5, 1). Decides [H] of §2.
- **E2. Estimator.** P0 with μ ∈ {1e-4, 1e-3, 1e-2}·λ̄ on the body. Converge plain CG and the circulant to 1e-6
  relative residual and compare images (Q1).
- **E3. Speed.** P0, P0+P1, P0+P1+P2, P0+P1+P3 (r = 8, 32), plain CG, the current circulant. Pooled static and
  dynamic; then per-frame P²nP (LLR κ 1, 60 iterations) from the pooled start, scored with `--score-pc`
  (pcscore.rs).
- **E4. Physics sensitivity.** The best variant at field scale 0.5 and 1, and with a realistic field (if available).

## 6. Results

### 6.0a Update (2026-11-22 evening): built, tested, E1 measured; e2 running, e3/pnp/e2b/e4 not run

- **Build and tests.**
  - Compiled first time.
  - CPU unit tests 8/8 pass. Two failed at first, and the deflation one was a real bug: unconverged interior "ghost"
    Ritz values (θ 89.8, residual 0.17, between the eigenvalues 88 and 119) set the deflation target. Fixed: the
    target range now comes from the CONVERGED non-deflated Ritz values only.
  - GPU tests pass: `ppre_gpu` shows device metric = host reference with deflation, so the FFT sign conventions agree.
    `precond_host` passes.
- **E1 acceptance** (static object, keep 2, pooled encode 0, μ = 0) [M]: circulant 2.204 / plain CG 0.294 at 30
  iterations. The diagnosis had 2.19 / 0.29. One pooled iteration of one encode takes 0.45 s.
- **E1 at 40 iterations** [M], error vs x★ (all / static body / where x★ = 0):

| field scale | circ | none | p1 (J = 8, radius 2) | p1n | θ_max(circ) | θ_max(none) |
|---|---|---|---|---|---|---|
| 1 | 2.36 / 2.32 / 0.43 | 0.232 / 0.227 / 0.046 | 3.27 | 2.31 | 991 | 1.5e10 |
| 0.5 | 2.93 / 2.85 / 0.67 | 0.165 / 0.160 / 0.043 | 4.18 | 2.96 | 790 | 1.4e10 |
| 0 | 0.64 / 0.215 / 0.60 | 0.135 / 0.127 / 0.047 | 1.13 | 0.68 / 0.37 / 0.57 | 329 | 2.1e10 |

- **Where the top Ritz vectors of the circulant-preconditioned operator sit** (field scale 1):
  - k space: about 70 % of their energy at r ∈ [0.5, 1.2], 0.1 % at r < 1/8; 82–94 % inside the ball.
  - Image space: 0.6–6.5 % in the high-field 10 % of the body, i.e. under-represented. Air ≤ 0.2 % (the support
    covers all but 770 voxels).
  - No Ritz pair converged to a residual ≤ 1e-2 in 40 iterations, so the θ are lower bounds.
- **Symbol.** probed/nominal per k shell is 1.18 1.02 1.07 1.06 1.02 0.86 0.97 0.40 at field scale 1, and similar at
  scale 0. The probe noise is 0.96 per mode (J = 8, radius 2).
- **§6.3 verdict.**
  - Localised smearing: REFUTED. The outliers are not in the high-field voxels.
  - Air / D^{-1/2}: REFUTED.
  - The field does matter globally: θ_max 329 → 991 as the field scale goes 0 → 1, and without a field the body
    converges while the energy goes where x★ = 0 (Proposition 2).
  - The symbol prediction (< 1 at r < 1/8) is REFUTED. The physics changes the Chan symbol by ≲ ±20 %, within probe
    noise. Its effect is OFF-diagonal in k, which no circulant (P1) can represent.
  - Mechanism [D from these M]: the metric's gain 1/ε on the 91 % of modes at the floor, times the off-diagonal
    curvature that coils and field put into those mid-k modes, gives θ ≈ 1/ε. The fix has to bound the metric's
    dynamic range, i.e. the floor, not refine the symbol.
  - Caution for P0: the ridge floor μ/D at μ = 1e-3 λ̄ lies far BELOW ε λ_max (λ̄ ≪ λ_max), so it raises the dynamic
    range. Prediction: p0 at 1e-3 is slow and 1e-2 is better. e2 decides.

### 6.0c Semi-convergence at μ = 1e-3 (e2 restart, 2026-11-22) [M]

- Jacobi PCG on the μ = 1e-3 λ̄ ridge estimator, static inverse crime, keep 2, pooled encode 0:
  - after 700 iterations the residual is still 7.6e-5;
  - the error vs x★ reaches its minimum 0.0665 at iteration 314 and then rises to 0.204 at 700;
  - the rise is almost entirely in the voxels where x★ = 0 (0.198 of ‖x★‖).
- Plain CG (300 iterations: 0.075) and Jacobi (300: 0.069) are still in the descending phase.
- Consequences:
  - (i) With μ = 1e-3 the estimator x̂_μ itself carries substantial energy where x★ = 0, the weakly determined
    directions of Proposition 3. A ridge this small does not regularise them: μ must be larger (1e-2 was used for g3),
    or the support / band constraint of §1 (a) is needed.
  - (ii) "Converge to 1e-6 and compare" (E2 as planned) is infeasible at μ = 1e-3 within any affordable cap. Q1 could
    only be checked at larger μ.
  - (iii) The best image vs x★ is an early-stopped one, so the reported pooled images remain implicitly regularised
    by the iteration count. That is exactly what §1 warned about.

### 6.0b Partial e2 (2026-11-22, stopped by the user at 19:54: quiet time) [M]

The estimator here is static inverse crime, keep 2, pooled encode 0, ridge μ = 1e-3 λ̄ (λ̄ = 3.23e6, probe noise of
the image diagonal 0.05), no band. The table gives the state after the iteration cap.

| preconditioner | iterations | ‖r‖/‖r₀‖ | error vs x★ all / body / where x★ = 0 | PD corr |
|---|---|---|---|---|
| none (plain CG) | 150 | 2.1e-4 | **0.088** / 0.079 / 0.040 | 0.990 |
| p2 (probed Jacobi) | 150 | 1.8e-4 | **0.087** / 0.077 / 0.039 | 0.991 |
| circ (old metric, ε = 1e-3) | 150 | 5.1e-3 | 3.50 / 3.35 / 1.02 | −0.02 |
| p0 (nominal symbol, floor μ/D = 3.2e3 vs λ_max 5.8e9) | 150 | 9.9e-3 | 7.81 / 7.44 / 2.37 | 0.12 |
| p1n (probed ratio form, radius 6, noise 0.43) | 150 | 1.6e-2 | 7.27 | 0.21 |
| p0 + deflation 8 / 32 + 4 | 150 | 9.2e-3 / 1.1e-2 | 7.80 / 7.81 | 0.12 |
| p0 + deflation 16 ("ref") | 300 | 5.0e-3 | 8.28 | −0.02 |

What these numbers show:
- **With an explicit ridge, the Fourier metrics are still bad.** PCG in these metrics also makes the residual stall
  at ~1e-2, not only the ℓ2 error. Within the cap they never get near the common limit x̂_μ.
- **This is the predicted dynamic-range failure.** The floor matched to μ gives a metric range of 1.8e6, worse than
  ε = 1e-3. The ridge fixes the estimator (Proposition 1), not the conditioning.
- **Deflation with unconverged Ritz vectors did nothing.** No Ritz pair had a residual ≤ 1e-2.
- **Plain CG and Jacobi are the only working pooled solvers so far.** With the ridge they reach 0.087 in 150
  iterations (μ = 0, 40 iterations: 0.23).
- **The "ref" run is not a converged x̂.** The distance-to-x̂ columns of this partial run are void. The restart uses
  Jacobi to 1e-6 as the reference.

### 6.0 Status (2026-11-22): implemented, NOT compiled, NOT measured

Quiet mode was ordered mid-task (no GPU, no builds or tests). Everything below §6.2 is implementation and derivation
only. No number in this section comes from the dev case. The one **[M]** is a 40 × 40 toy (f64, single thread) that
checks the formulas. GPU minutes used: 0. The exact commands for the campaign are in §6.4.

### 6.1 What is implemented (all behind new switches, defaults unchanged, no old path removed)

- `hasty-recon/src/ppre.rs` (new, CPU, rustfft + rayon):
  - `Fft3`: the CPU 3-D FFT, with the conventions of `CirculantPrecond`.
  - `SymPre`: the circulant metric D^{-1/2} F⁻¹ diag(1/λ) F D^{-1/2}.
  - `project_band` / `ball_mask`: the band-limit B.
  - `probe_symbol` (P1): the Fourier diagonal of D^{-1/2} Q D^{-1/2}, from unit-phase probes.
  - `probe_diag` (P2): the image-space diagonal.
  - `smooth_symbol`: periodic box smoothing; the ratio form `p1n` = λ_nom · S(λ̂) / S(λ_nom).
  - `pcg_groups`: grouped PCG through a `PcgOps` trait, with `Harvest` (Lanczos data at no extra cost).
  - `ritz` (Rayleigh–Ritz over the complex span), `Deflation` (P3, plus its Woodbury inverse), `energy_where` (the
    E1 report), `PreSpec` (the spec parser).
  - Unit tests (CPU, tiny):
    - FFT vs DFT;
    - `SymPre` is Hermitian PD and its inverse is exact; the band composite is Hermitian and zero outside the ball;
    - probing is exact for a circulant;
    - **converged solution independent of the preconditioner** (I, Jacobi, circulant, deflated);
    - Ritz values vs a dense eigensolve; the deflated P' is Hermitian PD, lowers λ_max and cuts iterations;
      Woodbury is exact;
    - the spec parser.
- `precond.rs`:
  - `CirculantPrecond::from_factors(λ_eff, D^{-1/2})`: any symbol as the device metric.
  - `with_deflation(U, Θ, targets)`: device-resident U and V = M⁻¹U; deflation in `apply` / `apply_inv`; also
    host-resident via `to_host`.
  - GPU test `tests/ppre_gpu.rs`: device = host reference, with and without deflation; M'(M')⁻¹ = I. It catches an
    FFT-sign mismatch with an asymmetric symbol.
- `recon_p2np.rs`:
  - `--pp-study "name=SPEC:ITERS[:TOL];…"` with `--pp-groups`, `--pp-mu R[,R]`, `--pp-band`, `--pp-dfloor`,
    `--pp-probes`, `--pp-smooth`, `--pp-eps`, `--pp-harvest`, `--pp-tol`.
  - `--pp-pool SPEC`: the `--warm-full` start with SPEC. It writes the `warm_full_done_*` checkpoints, which the old
    block then loads.
  - `--pp-metric SPEC[+dT[,B]]`: replaces the zmshr1m per-encode circulants. It runs a device-vs-host check and
    mirrors the symbol automatically if the FFT signs differ.
  - `--field-scale S`: scales the operator's off-resonance field (diagnostic, for E1/E4).
  - SPEC = none | circ | p0 | p1 | p1n | p12 | p12n | p2, plus `+dT[,B]`.
  - Outputs: per run `pp_mu<μ>_<name>.csv` (per iteration: residual, ‖x − x̂‖/‖x̂‖, error vs x★ by class, PD
    correlation) and `pp_summary.txt`. Tables: `research/experiments/precond_estimator/summarize.py`.
- Estimator (P0) as implemented:
  - Q = B(N + μI)B on range(B), with B = I unless `--pp-band`; b = −B Σ∇f(0).
  - μ = R · λ̄, with λ̄ = the body mean of the probed image diagonal of the pooled N (per encode).
  - Symbol floor = μ / median_body(D), *added* to the probed symbol (the ridge's own symbol in the body).
  - D = max(Σ_c|S_c|², 0.1 · body median).
  - With μ = 0 the floor is ε λ_max, as before.

### 6.2 Derived while implementing [D]

1. **Ritz pairs of PQ from PCG at no cost, robust form.** Write PCG as r_{k+1} = r_k − α_k Q p_k, z = P r,
   p_{k+1} = z_{k+1} + β_k p_k.
   - Then P Q z_k = (z_k − z_{k+1})/α_k − β_{k−1}(z_{k−1} − z_k)/α_{k−1}, i.e. PQ Z_m = Z_{m+1} T̃ with T̃ bidiagonal
     in α, β.
   - With G_jk = z_jᴴ P⁻¹ z_k = z_jᴴ r_k (one inner product per stored vector and iteration), the projected matrix
     is H = Z_mᴴ Q Z_m = G_{m,m+1} T̃ (no Q applications).
   - The Ritz pairs solve H s = θ G s. Using G explicitly drops the Lanczos ghosts; the plain tridiagonal assumes
     orthogonality.
   - Exact residual: ‖PQu − θu‖²_{P⁻¹} = cᴴ G_{m+1} c, with c = T̃ s − θ[s; 0].
   - Toy check [M, 40 × 40 complex, f64]:
     - H matches Z_mᴴ Q Z_m to 6e-16;
     - after 25 iterations the top 4 Ritz values equal the eigenvalues of P^{1/2} Q P^{1/2} to all printed digits;
     - the residual estimate matches the direct one to 1 %;
     - deflating the top 4 takes κ from 1.4e4 to 110;
     - the Woodbury inverse is exact to 2e-14.
2. **Deflation target and the complex span.**
   - P' = P + U(Θ_c Θ⁻¹ − I)Uᴴ is Hermitian PD for any θ_i > 0 when U is P⁻¹-orthonormal. In whitened coordinates its
     eigenvalues are 1 and θ_c/θ_i.
   - A Ritz problem over the real span (Re⟨·,·⟩, as PCG uses) would deflate u but not i·u. Q is complex-linear, so
     i·u is an eigenvector with the same θ, and half of each outlier pair would remain. The complex span fixes this.
3. **Air under the matched floor.**
   - In air, N ≈ 0, Q ≈ μ and D = f · D_med. With floor μ/D_med, the eigenvalue of PQ there is ≈ f = `--pp-dfloor`
     (0.1).
   - This is a bounded lower outlier family. It is not the old metric's unbounded gain on (near-)null directions
     (Proposition 2: the old floor ε λ_max is unrelated to any curvature of the estimator).
   - If e2 shows the bottom Ritz vectors in air with θ ≈ f, raise f or deflate the bottom (`+dT,B`).
4. **Probe variance.** Var λ̂(k) = (1/J) Σ_{k'≠k} |Q̂_{kk'}|² (unit-phase probes; the estimate is exact for a
   circulant, unit test). The row energy off the diagonal comes from the coil-map spectra and the off-resonance
   coupling. Box smoothing over n_s ≈ 125 modes divides the variance by up to n_s. The log prints the split-half
   noise; acceptable if ≲ 0.1. The ratio form p1n keeps the nominal fine structure (the k = 0 spike) that smoothing
   would erase.
5. **Plain CG cannot reach the E2 tolerance at μ = 1e-3 λ̄ within the cap.**
   - κ(N + μ) ≥ λ_max(N)/μ = 10³ · (peak/mean of the symbol). The peak/mean is printed by e1 as the mean probed
     symbol per shell.
   - For the yarn density that is likely 10²–10³, so κ ≥ 1e5–1e6 and ≳ √κ ≈ 300–1000 CG iterations to 1e-6.
   - Pre-registered: in e2, "none" and probably "circ" hit the 200-iteration cap. For them Q1 is read as a monotone
     decrease of ‖x − x̂‖, not as ≤ 1e-3.
6. **Cost per application.**
   - P0/P1/p12: one FFT pair plus 4 real multiplies. The band-limit adds 2 FFT pairs. P3 adds 2r complex
     dots/axpys and needs r images.
   - At 256³ on the device: one FFT pair ≈ a few ms; P3 with r = 16 ≈ 32 passes over 134 MB ≈ 4–5 ms at ~1 TB/s
     (estimate). Both are ≲ 2 % of a 391 ms one-set gradient.
   - The probes are once per encode: J = 8 pooled passes per symbol (≈ 8 pooled iterations).
   - This implementation applies the pooled-PCG preconditioner on the CPU (study and `--pp-pool`). The ms printed
     there are CPU ms. The device ms are printed by `--pp-metric`.

### 6.3 Pre-registered predictions for E1 [H] (decided before any data)

- **Smearing (§2) confirmed if** all of the following hold:
  - with the current circulant (μ = 0, full physics) the top Ritz vectors of PQ have most of their k energy at
    r ∈ [0.5, 1.2] (the floored shells);
  - in image space they put > 10 % of their energy in the high-field 10 % of the body;
  - θ_max falls with field scale (1 → 0.5 → 0) towards the plain-model value (~10).
- **Refuted if** θ_max barely changes with the field scale (then the cause is the concomitant fields: scale 0 keeps
  them), **or** the top Ritz vectors sit in air (then the D^{-1/2} hypothesis of §2 is the mechanism, and P4 /
  a different D is needed).
- **Symbol:** probed/nominal per shell is < 1 at r < 1/8 and > 1 at r ∈ [0.5, 1] with full physics, and ≈ 1 at field
  scale 0.
- **Bottom Ritz** (μ = 0): corners (r > 1) and air, i.e. the null space of Proposition 2.

### 6.4 Commands (when the GPU is allowed again; one combined campaign, ≈ 70 GPU min estimated)

```
research/experiments/precond_estimator/run_campaign.sh build   # CPU, -j 4: recon_p2np + tests; ppre/lanczos unit tests
research/experiments/precond_estimator/run_campaign.sh test    # GPU: ppre_gpu, precond_host
research/experiments/precond_estimator/run_campaign.sh e1      # field scale 1, 0.5, 0: circ / none / p1 / p1n, μ = 0
research/experiments/precond_estimator/run_campaign.sh e2      # static, μ 1e-2 and 1e-3: all candidates (Q1, Q2, Q3, Q4)
research/experiments/precond_estimator/run_campaign.sh e3      # dynamic, μ 1e-3: short list
research/experiments/precond_estimator/run_campaign.sh pnp     # per-frame P²nP 60 it: (A) today's pipeline, (B) p1n
research/experiments/precond_estimator/run_campaign.sh e2b     # band-limited estimator (optional)
BEST=<from e2> research/experiments/precond_estimator/run_campaign.sh e4   # field scale 0.5
python research/experiments/precond_estimator/summarize.py data/hasty/phantom/factory_arms/ph0001/precond_est/{e1_fs1,e1_fs0.5,e1_fs0,e2_pd,e3_pc}
```

- Check the per-iteration time in the first e1 log. The budget assumes ≈ 0.5 s per pooled iteration of one encode
  (= 3.5 s / 7 encodes, §5 of the diagnosis). If it is slower, drop e2b / e4 first.
- Acceptance before reading anything: e1 at field scale 1 must reproduce the diagnosis.
  - The circ curve's error must grow (2.19 after 30 iterations, keep 2, static).
  - The none curve must reach ≈ 0.29.
  - The adjoint and `ppre_gpu` checks must pass. The device-vs-host line of `--pp-metric` must show ≤ 1e-4.
- Sampling level of every run: keep 2 = 26 674 readouts, 162–214 per (frame, encode) set, 0.27–0.35 samples per voxel
  per set per coil, 6.3 pooled per encode; 128 × 72 × 60 grid; 8 virtual coils; kcrop.

### 6.5 Decision (pending the campaign)

- **Rule (§4):** the default is the cheapest variant that meets Q1 and is ≥ 3× faster than plain CG in iterations to
  1 % of x̂.
- **Cost order:** p2 (Jacobi, no FFT) < p0 = p1 = p1n (1 FFT pair; the p1 variants also need 8 pooled probe passes
  once) < p12n (+8 passes) < +dR (+2R dots, R images).
- **Provisional, from theory only:** p1n with μ = 1e-3 λ̄ and the floor = μ, as the pooled default if it meets the
  rule. Deflation only if it cuts iterations to 1 % by a further ≥ 1.5×.
- **If no variant meets the rule:** plain CG stays for the pooled solve. Report why (κ_eff of the Ritz report vs the
  §6.2.5 bound).
- **Defaults until then:** unchanged (`--pool-metric circ`, zmshr1m = the old circulant). For yarn dev runs keep using
  `--pool-metric none --precond zmshr1` (recon_diag §6).

### 6.6 Open problems / limitations of this implementation

- Not compiled: expect a round of compile fixes, mainly in the binary block (`pp_run`) and the nalgebra complex
  Rayleigh–Ritz.
- The pooled-PCG preconditioner runs on the host (rustfft). Before 256³ / 1 mm, route `--pp-pool` through
  `CirculantPrecond::from_factors` on the device; the code for that exists.
- `--pp-pool` has no mid-solve checkpoint: a STOP restarts the pooled solve. This is cheap at dev scale and violates
  the resumable-runs rule at scale. Add raw checkpoints as in `pooled_pcg_host` before a large run.
- The P²nP metric deflation uses Ritz vectors of the *pooled* operator (20 frames, ridge, band if on). The per-set
  operator only has the same structure; P' stays SPD, but the deflation is approximate there.
- μ is set per encode from that encode's λ̄ (the encodes differ by ≲ %).
- P4 is not implemented. It is needed only if E1 shows localised outliers that P1 does not remove.
- Still open from recon_diag §4: `RcSets` over `SynthSets` subtracts the passed data (not used here).

## 7. Which structured classes can capture the curvature? (2026-11-22, theory + exact 2-D toy, no GPU)

User question: can the preconditioner class be more general, e.g. a sum of circulants, circulant + diagonal, banded, or
other sums of fast pieces? Toy: `research/experiments/precond_estimator/symbol_toy.py` (n = 32, exact dense
operators, single thread, results in `symbol_toy_results.txt`).

### 7.1 Algebra: what a "sum of fast pieces" can and cannot be [D]
- **Sums within one fast family add nothing.** Circulants are closed under sums and products (all are diagonal in F);
  so are diagonals. A sum of circulants is one circulant.
- **Useful classes mix image-space diagonals with Fourier diagonals.** The general form is Σ_j A_j C_j B_j, with A_j,
  B_j diagonal and C_j circulant (or Toeplitz), i.e. product-convolution, the "displacement rank" type. Every matrix is
  such a sum with ≤ n terms; the question is how few terms are needed.
- **Q itself has low rank in this sense, exactly.** Q = Σ_c S_cᴴ (Σ_{l,l'} B_lᴴ T_{ll'} B_{l'}) S_c: coils S_c and the
  stored SVD segment maps B_l are diagonals, T_{ll'} are Toeplitz kernels with the weights a_l*(t) a_{l'}(t). That is
  C·L² terms; it is the Toeplitz normal operator.
- **Q⁻¹ does not.**
  - Leading-order symbol calculus gives Q's symbol σ(x, k) = Σ_c |S_c(x)|² Σ_s w_s(x) K(k − k_s − t_s ∇φ(x)/2π).
  - Conjugating by a phase diagonal e^{iφ(x)} shifts the local symbol by ∇φ(x). This is how a sum of modulated
    circulants represents the off-resonance (and concomitant) k-shift exactly, through the stored maps.
  - The inverse symbol 1/σ has a "hole" at every local centre pass, and the hole MOVES with x by s(x) = Δt ∇f(x).
  - A family of translates h(k − s_j) at m resolvable shifts has separable rank ≈ m (nearly orthogonal columns). So any
    directly-applied P = Σ_j A_j C_j A_jᴴ that approximates Q⁻¹ needs ≈ (number of resolvable shifts) terms.
  - Toy: product-convolution on k-means clusters of ∇ω gets WORSE with J (J = 4: 159 → J = 16: 190 iterations).
- **Consequence.** There are two families.
  - (A) Directly applied approximate inverses: D·C·D, product-convolution, a k-space two-level block, deflation. These
    are cheap, but structurally they cannot follow a gradient-shifted centre.
  - (B) A cheap surrogate Q̃ of the *forward* class, Σ_c S_cᴴ (Σ_{l,l'} B_lᴴ C_{ll'} B_{l'}) S_c with circulants on the
    1× grid, inverted by a few inner iterations. This follows the shifts exactly through B_l and keeps the coils
    exactly.
    - Outer: flexible PCG on the true Q. Its limit is Q⁻¹b, so the estimator is unchanged given a strictly convex
      objective (§1).
    - A fixed-degree Chebyshev inner instead of CG makes P a fixed SPD operator, so plain PCG also works.

### 7.2 Toy results (2-D, n = 32, 96 out-and-back spokes, density ~k^-1.7, 4 coils, ridge μ = 1e-3 diag)
PCG iterations to ‖x − x̂‖/‖x̂‖ ≤ 1e-3 (1e-6 in brackets); κ = cond(PQ). The max local shift of the second centre pass
is 0 / 0.25 / 0.5 k_max.

| preconditioner | κ (shift 0) | its, shift 0 | shift 0.25 | shift 0.5 |
|---|---|---|---|---|
| none (plain CG) | 1.6e5 | 223 (>600) | 228 | 231 (>600) |
| D·C·D, ε 1e-3, p = 1 (current form) | 6.4e3 | 88 (193) | 96 | 115 (261) |
| D·C·D, bounded (ε 1e-2…1e-1, p 0.5–1) | 3e4–1.2e5 | 131–212 | 140–220 | 151–225 |
| U·DCD·Uᴴ, U = e^{iω t_ref}, t_ref = mid (free) | 6.4e3 | 88 | 88 | **89** |
| two-level k: dense block \|k\| ≤ 6 (113 modes) + DCD | 3.4e3 | 67 | 73 | 97 |
| product-convolution J = 4 / 16 | 6.4e3 | 88 | 159 / 190 | 156 / 192 |
| (D½ C D½ + μ)⁻¹ exact, no field (D-collapsed coils) | 804 | 76 | 104 | 149 |
| D-collapsed segment surrogate, exact inverse | 804 | 76 | 73 | 72 |
| **coil + segment surrogate (circulant), exact inverse** | **60** | **19 (44)** | **20** | **21 (47)** |
| same, Toeplitz instead of circulant (= Q up to segments) | 1 | 1 | 7 | 14 |
| **coil + segment surrogate, 10 inner PCG (DCD inner), flexible outer** | – | **26 (53)** | **25** | **30 (66)** |
| coil + segment surrogate, 30 inner | – | 20 | 22 | 24 |
| D-collapsed surrogate, 20 inner | – | 104 | 104 | 98 |

What the toy says [M, toy]:
- **The coil structure is the leading mismatch.** Folding coils into the scalar D (D·C·D, or even its exact
  inverse) leaves κ ≈ 800. Keeping the coil diagonals gives κ ≈ 60.
  - Symbol reason: the second-order terms of Σ_c S_cᴴ C S_c are ∝ |∇S|² ∇²_k λ, which is large where the density is
    sharply peaked (near the k-centre).
- **The field shift is the second mismatch.** D-collapsed κ goes 804 → 6180 as the shift grows to 0.5 k_max; the
  modulated surrogate stays flat.
- **The inexact surrogate keeps most of the gain:** 10 inner steps → 25–30 outer iterations (exact: 19–21). A poor
  surrogate (D-collapsed) inverted inexactly does not (~100).
- **Bounded-range D·C·D (the e5 family) is worse than ε = 1e-3 in the toy.** The toy's density range is only ~10²,
  so it does not reproduce the 3-D floor failure. The bounded family must be judged on the dev case, not here.
- **The toy does NOT reproduce the 3-D failure** (D·C·D beats plain CG here). The toy has a mild density range (10² vs
  10³–10⁴ in 3-D yarn) and mild shifts.

### 7.3 The real shifts are large [M, phantom field, light CPU]
- The yarn passes the k-centre twice: at the ADC start (TE 1.6 ms) and at the ADC end, Δt ≈ 7.9 ms later
  (`k_adc.npy`: low-|k| samples at the first and last ~0.1 ms).
- ph0001 field gradient in the body: |∇f| median 48 Hz/cm; 75/90/95/99 % at 101 / 273 / 636 / 1199 Hz/cm. The tails
  are unrealistic (phantom_field_tails).
- Shift of the second centre pass, s = Δt|∇f|, in units of k_max:

| grid | median | 75 % | 90 % | 95 % |
|---|---|---|---|---|
| 4 mm (dev, 128×72×60) | 0.31 | 0.64 | 1.7 | 4.0 |
| 2 mm (256×144×114) | 0.15 | 0.32 | 0.86 | 2.0 |
| 1 mm | 0.08 | 0.16 | 0.43 | 1.0 |

- So at dev scale a quarter of the body has its second centre pass beyond 0.64 k_max. That matches the E1 Ritz
  vectors, 70 % of whose k energy sits at r 0.5–1.2.
- **E1's "high-field voxels" test was the wrong criterion** (my §2 wording). The shift follows |∇f|, which peaks where
  the field *changes*, not where |f| is largest. The smearing hypothesis is therefore NOT refuted; it is untested.
- The dev grid exaggerates the effect 4× relative to 1 mm, and the rough phantom field exaggerates it further.
- **Correction (G1, 2026-11-22, measured; see §7.5):** the shift hypothesis was tested against the |∇f| shift map and
  is REFUTED for the dominant outliers.
  - Δt measured 7.975 ms. On the operator's own field the shift is median 0.21, 90 % 1.20; s > 0.3 covers 34 % of the
    grid.
  - The top-2 Ritz vectors of the circulant-preconditioned operator put only 0.03× and 0.12× their volume share into
    s > 0.3; the third puts 1.1×. The criterion was ≥ 2×.
  - The outliers sit where the field is SMOOTH. The statements above that the shift "matches the E1 Ritz vectors"
    and that the smearing hypothesis is "untested" are superseded.
  - The remaining candidate for the circulant's failure is the coil structure (§7.2) together with the density
    range, not the gradient shift. G2 (recentring), whose premise was the shift, was therefore not run.

### 7.4 Recommended design and GPU tests (pre-registered, run after quiet time)
1. **G1, retest the mechanism.** Do E1's Ritz-energy report against the shift map s(x) = 7.9 ms·|∇f(x)| instead of
   |f|.
   - Confirmed if the top Ritz vectors put ≥ 2× their volume share of energy in voxels with s > 0.3 k_max.
   - Cost: ~2 min.
2. **G2, free recentring.** U·P·Uᴴ with U = diag(e^{-z(x) t_ref}) from the STORED ratemap/segment maps (the B_l of the
   dominant segment, not an on-the-fly map). t_ref = the density-weighted mid time of the two centre passes. Cost: one
   diagonal multiply.
3. **G3, coil + segment surrogate preconditioner** (family B; the main candidate):
   - Q̃ = Σ_c S_cᴴ (Σ_{l,l' ≤ L_s} B_lᴴ C_{ll'} B_{l'}) S_c, with the stored SVD maps truncated to the top L_s = 2–4
     components. Per-k Hermitian L_s × L_s symbol Λ(k) = Chan symbols of the weights a_l*(t) a_{l'}(t), built once
     with `build_kernels` (weights), on the 1× grid.
   - Optionally fewer (compressed) coils than the gradient uses: the preconditioner need not be exact.
   - Inner: K = 5 / 10 PCG steps with the current DCD metric. Outer: flexible PCG on Q.
   - Cost per Q̃ application ≈ C·L_s FFT pairs on the 1× grid + one Λ read: ≈ 1/10 of a gradient at 256³ (C = 10,
     L_s = 3: ~26 ms FFT + ~6 ms Λ + diagonals, vs 391 ms per gradient). Memory: L_s(L_s+1)/2 grids.
   - Accept if wall time to 1e-3 of x̂ is ≥ 2× better than plain CG and Jacobi on the keep-2 dev case. The toy
     predicts ≈ 4× vs plain CG and ≈ 2× vs DCD.
4. **G4.** A two-level k-block and deflation only as add-ons on top of G3. Product-convolution is dropped (§7.1).
5. **Estimator.** All of G2–G4 are SPD (G3 via flexible outer PCG). With the explicit estimator of §1 (ridge/support)
   the converged image is preconditioner-independent; check as in E2 (Jacobi to 1e-6 as the reference).
6. **P²nP.** A variable inner-solved preconditioner breaks the fixed-point invariance of fixed-strength PnP
   (convergence_p2np.md). Use it in the exact data step of ADMM/HQS, or as a fixed Chebyshev-K polynomial (fixed
   SPD) if it goes inside the P²nP metric.

### 7.5 Implementation status (2026-11-22): G1–G4 implemented, NOT built, NOT measured

The build was refused twice by the session's permission check (cargo build, after quiet mode was lifted). Nothing below
is compiled or run. A source review fixed the obvious errors (borrows, closure moves, slice coercions).

**Code** (all behind new switches; defaults unchanged; no old path removed):
- `hasty-nufft`: `SegSegments::host_tables` → `SegHostTables`, host copies of the STORED diagonals and interpolator
  table. `interp` is the device's linear a_l(τ).
- `hasty-recon/src/surrogate.rs` (new):
  - `compress`: rank-L_s compression of the stored decomposition. It uses linear combinations of the stored maps and
    interpolators, weighted by coil power and sample count.
  - `recentre_diag` (G2): U = conj phase of Σ_l a_l(t_ref) B_l, unit modulus. The metric becomes U P Uᴴ.
  - `pair_weights` and `chan_symbol_dev`: `build_kernels` with complex `extra` weights, Fejér fold.
  - `chan_symbol_host`: direct Fejér sum, used by the tests.
  - `SurDev` (device): per coil, L_s forward FFTs, the per-k L_s × L_s product, L_s inverse FFTs; then `solve`
    (inner PCG = variable, or K-step Chebyshev = fixed SPD) and `bounds` (Lanczos + 5 %).
  - `SurHost`, `inner_pcg_host`, `cheb_host`: host references.
- `ppre.rs`:
  - `PcgCfg::flexible`: Polak–Ribière β, no harvest.
  - `mul_cx`.
  - SPEC grammar: `+u`, `sur<L>k<K>[n]`, `cheb<L>k<K>[n]`.
- `recon_p2np.rs`:
  - G1: the k-centre passes from the kept samples give Δt and t_ref; the shift map s = Δt|∇ω|/π uses the operator's
    ratemap field; the Ritz report prints the energy share in s > 0.3 against its volume share.
  - G2 / G3 in `--pp-study` and `--pp-pool`.
  - Surrogate conventions (k mirror, pair transpose) and scale are fixed by the operator check min ‖cÑx − Nx‖/‖Nx‖,
    which is data-free and printed.
  - The `--pp-study` summary adds outer iterations and wall time to 1e-2 / 1e-3 / 1e-4 of x̂, and setup time.
  - Flags: `--pp-tref MS`, `--pp-sur-coils N`, `--pp-cheb-ratio R` (default 30).
  - With `--pp-*`, the kept samples' coordinates and times are retained (≈ 0.5 GB at dev scale).
- G4: deflation on top of the fixed Chebyshev form (`cheb…+dT`). The dense low-|k| two-level block is NOT
  implemented (kept small, per the task).
- Tests:
  - CPU (`surrogate::tests`, tiny): pidx bijection; full-rank compression is exact; surrogate apply = dense
    definition, Hermitian PSD; unit-weight Chan symbol = Fejér density (mean = sample count); G2 diagonal is unit
    modulus and conj phase at a node; Chebyshev operator SPD; inner PCG solves; flexible PCG with an inexact
    surrogate inner reaches the plain-CG x̂ (estimator invariance).
  - `ppre::tests`: the spec parser.
  - GPU (`tests/surrogate_gpu.rs`): device Chan symbol = host Fejér (reports the convention); `SurDev::apply` =
    `SurHost::apply` for three conventions; device inner PCG / Chebyshev = host.

**Stages done:**
- build/test/e1 (§6.0a); partial e2 (§6.0b).
- 20:54: built and tested by the main session. CPU 15/15; `surrogate_gpu`: device Chan symbol = host Fejér 2e-6
  "as is"; `ppre_gpu` passes.
- **g1** (20:56–20:58), μ = 0, circulant, 40 iterations [M]:
  - Δt = 7.975 ms between the centre passes (1.601 / 9.576 ms), t_ref 6.865 ms.
  - Shift map in the body: median 0.21, 75 % 0.40, 90 % 1.20, 95 % 2.47; s > 0.3 covers 34.3 % of the grid.
  - Energy share of the top-3 Ritz vectors in s > 0.3, relative to that volume share (ratio):

    | operator | top 1 | top 2 | top 3 |
    |---|---|---|---|
    | circulant | 0.012 (0.03×) | 0.042 (0.12×) | 0.385 (1.12×) |
    | plain CG | 0.02× | 0.10× | 0.70× |

  - **G1 verdict: smearing by the field-gradient shift REFUTED.** The criterion was ≥ 2×. The dominant outliers
    AVOID the shifted voxels; they sit where the field is smooth. The mismatch the circulant cannot follow is not the
    centre shift. The remaining candidate is the coil structure (§7.2: the leading mismatch in the toy), which G3
    keeps.

- **e2 restart** (20:58–21:12, μ = 1e-3 block complete, stopped at the start of μ = 1e-2 to save budget) [M]:

| run | iterations | ‖r‖/‖r₀‖ | vs x★ end all / body / x★ = 0 | best all (it) | Ritz θ_max/θ_min |
|---|---|---|---|---|---|
| Jacobi p2 ("ref") | 700 | 7.6e-5 | 0.204 / 0.046 / 0.198 | 0.0665 (314) | 913 |
| plain CG | 300 | 1.0e-4 | 0.075 / 0.047 / 0.059 | 0.0748 (275) | 1065 |
| Jacobi p2 | 300 | 6.9e-5 | 0.069 / 0.047 / 0.051 | 0.0692 (300) | 1172 |
| circ (old) | 60 | 2.0e-2 | 2.62 | 1.06 (1) | 2284 |
| p0 | 60 | 2.7e-2 | 6.45 | 1.55 (1) | 2333 |
| p1n | 60 | 7.0e-2 | 6.76 | 2.94 (1) | 1797 |

  - At μ = 1e-3, NOTHING converges within the caps: Jacobi needs > 700 iterations to reach 1e-6.
  - Semi-convergence: past iteration ~300 the error vs x★ grows again, almost entirely where x★ = 0. The μ = 1e-3
    estimator itself puts energy there. A ridge this weak does not regularise the near-null directions enough
    (Proposition 3). The Jacobi iterate at 700 is therefore NOT a converged x̂.
  - g3 was changed accordingly: μ = 1e-2, the surrogate run as the reference, shorter caps.
- **g3** (21:12–21:19), μ = 1e-2, static, pooled encode 0 [M]. **Invalid as a test of G3: the surrogate does not
  match the full-physics operator.**
  - Setup check ‖(N − cÑ)x‖/‖Nx‖ = 0.988–0.989 for all four conventions, best scale c = 3.4e-3, i.e. Ñx is
    uncorrelated with Nx.
  - With this broken surrogate, after 150 / 100 iterations:

| run | iterations | ‖r‖/‖r₀‖ | vs x★ all | ms per P | ms per Q |
|---|---|---|---|---|---|
| sur3k10 (flexible) | 150 | 2.2e-3 | 1.43 | 25.2 | 453 |
| cheb3k10 (interval [0.50, 14.9]) | 100 | 3.0e-3 | 3.02 | 23.8 | 458 |
| sur3k5 | 100 | 7.1e-3 | 3.33 | 13.4 | 464 |
| plain CG | 150 | 3.9e-4 | **0.090** | 0.4 | 456 |
| Jacobi p2 | 150 | 2.7e-4 | **0.086** | 2.9 | 458 |

- **Diagnostic g3diag_plain** (21:20–21:22): the same check with `--phys none` (one segment, no field) gives
  error **0.147** and scale **1.03**. The coil and k-space conventions and the Chan symbol normalisation are correct.
  The fault is in the full-physics SEGMENT part: the stored maps / interpolators as used by `compress` /
  `pair_weights`.
  - Prime suspect: the time variable of a_l(τ). The kept samples' `tau` vs the table's (t0, dt) may differ by an
    offset or units. A clamp to the table ends would collapse the weights and explain c ≈ 3e-3.
  - Also possible: whether the operator applies conj(B_l), and the ΔC concomitant corrections the surrogate ignores.
  - **Needs a code fix + rebuild.** Print the sample τ range against [t0, t0 + (ntab − 1) dt] in `pp_build_sur`, then
    test conj(B) and conj(a) conventions in the operator check.
  - Add a GPU test of Ñ against the TRUE `SegSets` operator at small size. The current tests compare Ñ only with its
    own definition.
- **Surrogate debugging** (21:24–21:40, rebuilt twice), using the real-operator check ‖(N − cÑ)x‖/‖Nx‖ on random x,
  now over 8 conventions including conj(B):

| model (dev case) | L_s | error | best scale c |
|---|---|---|---|
| plain (`--phys none`) | 1 | **0.147** | 1.03 |
| plain + GNL | 1 | **0.133** | 1.03 |
| B0 segments, no GNL (`--phys b0 --nseg 6`) | 6 (no truncation) | 0.987 | 5.0e-3 |
| B0 + GNL | 3 / 6 | 0.986 / 0.986 | 4.6e-3 |
| full (conc deg 4 + GNL), operator coordinates | 3 / 4 / 6 | 0.988 / 0.988 / 0.988 | 3.4e-3 |

  Ruled out:
  - sample τ vs the table: identical ranges, 1.600–9.578 ms, 3990 nodes;
  - the coordinate shift: the operator's cropped coordinates are now used; b0 has no absorption anyway;
  - GNL;
  - truncation (L_s = 6 fails too);
  - the concomitant ΔC terms (b0 fails the same);
  - k-mirror / pair-transpose / conj(B) conventions;
  - the stored decomposition itself: mean |Σ_l B_l a_l(τ)| 0.978, distance from e^{−iωτ} 0.079, max |B| = max |a| = 1.

  The CPU toy against the EXACT dense E^H E (new test `surrogate_vs_exact_normal_operator`) gives 0.69 / 0.56 / 0.35 at
  L_s = 1 / 2 / 3 with c ≈ 1. So the method and the host pipeline work in principle. On the dev operator ‖Ñx‖ ≈ 28 ‖Nx‖
  as soon as L > 1: an unidentified dev-specific fault.

  **Next suspects** (not tested, no budget):
  - `compress` on the real, nearly collinear fixed-time maps. The six maps e^{−iωτ_l} are nearly equal where ω is
    small; that Gram is near-singular, with a 1e-12 floor, f32 maps of |B'| ~ 1e3 and |a'| ~ 1e-3.
  - `chan_symbol_dev` on 3.4 M samples with |k| up to π.
  - Test next: Ñ with L_s = 6 and NO compression (the stored B_l, a_l directly), and the compressed g against the
    stored g on the real tables.

  **Acceptance (≲ 0.3) NOT met**, so the g3 rerun was not done. G3 remains undecided.
- **Not run:** g2 (skipped: G1 refuted its premise), g4, e3, pnp. e5 (shortened) is running.
  - g2 and e5 need no fix and can run as they are.
  - g3, g4 and the surrogate in pnp need the fix.
- GPU used ≈ 60 min of the 75-min cap before e5.

**Restart, in this order** (from `/home/turbotage/Documents/GitHub/HastyCompute`;
`C=research/experiments/precond_estimator/run_campaign.sh`; wait for each unit with ONE blocking
`while systemctl --user is-active -q <unit>; do sleep 30; done`):

```
cd rustlib && nice cargo build --release -j 4 -p hasty-bench --bin recon_p2np \
  && nice cargo test --release -j 4 -p hasty-recon --lib -- ppre surrogate lanczos \
  && nice cargo test --release -j 4 -p hasty-recon --test surrogate_gpu --test ppre_gpu --no-run && cd ..
T=$(ls -t rustlib/target/release/deps/surrogate_gpu-* | grep -v '\.d$' | head -1)
flock /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock \
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 timeout 600 $T --test-threads 1 --nocapture
systemd-run --user --unit=pe_g1 --working-directory=$PWD $C g1      # ~3 min: G1 verdict (§7.4.1)
systemd-run --user --unit=pe_e2 --working-directory=$PWD $C e2      # ~12 min: Jacobi-to-1e-6 reference, μ 1e-3 / 1e-2
systemd-run --user --unit=pe_g3 --working-directory=$PWD $C g3      # ~15 min: the main candidate vs plain CG / Jacobi
systemd-run --user --unit=pe_g2 --working-directory=$PWD $C g2      # ~6 min
systemd-run --user --unit=pe_e5 --working-directory=$PWD $C e5      # ~12 min: bounded family (drop if over budget)
systemd-run --user --unit=pe_g4 --working-directory=$PWD $C g4      # only if budget remains
WIN=<winner> systemd-run --user --unit=pe_e3 --working-directory=$PWD -E WIN=<winner> $C e3
WIN=<winner> systemd-run --user --unit=pe_pnp --working-directory=$PWD -E WIN=<winner> $C pnp
python research/experiments/precond_estimator/summarize.py data/hasty/phantom/factory_arms/ph0001/precond_est/{g1_fs1,e2_pd,g3_pd,g2_pd}
```

**Acceptance (pre-registered §7.4):**
- G1 is confirmed if the top Ritz vectors have energy share in s > 0.3 ≥ 2× its volume share.
- G3 is accepted if its wall time to 1e-3 of x̂ is ≥ 2× better than BOTH plain CG and Jacobi p2.
  - p2 is the bar from the partial e2: it reached 0.087 in 150 iterations at 0.45 s each.
- Otherwise Jacobi stays the pooled default (§4).
