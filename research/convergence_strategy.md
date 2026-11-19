# Fewest iterations: algorithm, metric, per-set steps and their schedule (2026-11-17)

User question (2026-11-17): the gradient (A, Aᴴ) is now close to as fast as it gets, so what remains is to converge in
as few iterations as possible. Which algorithm and which preconditioning? Circulant plus a dynamic (quasi-Newton)
metric? Should every frame / knot / encode / respiratory knot have its own step length and preconditioner? How should
all of this change over the iterations?

Builds on: `convergence_p2np.md` (cost model, density spread, ADMM-PCG toy), `algo_speed.md` (FISTA, SVRG, step-tied
prox), `recon_diag_2026-11-12.md` (PSF, unpreconditioned paths), `box_recon.md` (R0–R2), `temporal_operator.md` §13,
§16–§19 (knot Gram, G⁻¹ ⊗ C⁻¹, held-out), `denoiser_artifact_theory.md` (where the prior acts), `red_prior.md`,
`wavelet_prior.md`, `stencil_weighted_update.md` §5–§6 (operator epochs), `memory_ledger.md`,
`mathematics_of_paper/pnp_restricted/paper.tex` (prop:estimator, thm:displacement, prop:strength, prop:keepfp).
Code read (read-only): `hasty-recon/src/{p2np,precond,cg,knotsolve,tsub,pyramid,red,vobj}.rs`, `recon_p2np.rs`.

Labels: **[D]** derived here; **[T]** toy of this note (`research/experiments/convergence/strategy_toy.py`, CPU,
numpy, results in `.../out_strategy/`); **[M]** measured in our runs (source given); **[A]** abstract / summary
checked this session; **[R]** recalled, not re-checked. No GPU was used; no rustlib code was changed.

---------------------------------------------------------------------------------------------------------------------

## 0. Answer

1. **On the quadratic part of the problem, CG in a fixed metric is optimal, and no dynamic metric can beat it [D, T].**
   - Theorem K (§2.4): every method whose iterates lie in x₀ + 𝒦_k(PQ, P r₀) has a Q-norm error ≥ that of PCG with
     the same P after the same number of gradients. This class contains gradient descent with any step rule, BB,
     ZMSHR1 in the metric P (our `zmshr1m`, global), Nesterov / heavy ball, Chebyshev, Anderson(m) and L-BFGS with
     H₀ = cP.
   - So "circulant + dynamic" ≤ "circulant + CG" on the data term plus any quadratic (or locally quadratic) prior.
     Toy, iterations (= gradients) to 1 % of the minimiser, coupled sets, density range 10³:

     | metric \ method | GD | Nesterov | ZMSHR1-M | PCG |
     |---|---|---|---|---|
     | identity | — (>300) | 164 | 150 | 85 |
     | shared circulant | 171 | 83 | 34 | 19 |
     | prior-aware, per-set scaled, K(ω, k) | 107 | 55 | 21 | **14** |

   - The circulant metric is the bigger lever (85 → 19 for PCG), the algorithm the second (34 → 19 within the
     circulant), and the prior-aware term the third (19 → 14).
2. **The metric must invert the sum of the data and prior curvatures, with a scalar per set [D, T].**
   - P = N^{-1/2} (I ⊗ C + Σ_prior)^{-1} N^{-1/2}: C = the shared circulant (one per trajectory, since the 7 encode
     densities are identical [M]), N = diag(n_t) the per-set sample weight (sample count × coil energy × basis Gram
     diagonal), Σ_prior the symbol of the quadratic or locally quadratic prior curvature (temporal coupling, knot Gram
     G, LLR/wavelet envelope curvature).
   - For knots it is G⁻¹ ⊗ C⁻¹ (`knotsolve.rs`) plus the prior term. With uniform periodic knots and a temporal-Fourier
     prior symbol everything is jointly diagonal in (ω, k): one FFT over knots and one over space.
   - Toy, 16× spread of set weights: PCG to 1 %: shared C 44, + per-set scalar 35, per-set circulants 37,
     prior-aware K(ω, k) **16**. Gradient descent with the shared C does not reach 10 % in 300.
3. **Per-set (per frame / knot / encode) step lengths: static scalars yes, dynamic per-set steps no [D, T, M].**
   - A per-set step is a block-diagonal scalar metric. The right scalars are known before the run (n_t), cost
     nothing, and belong in P.
   - Dynamic per-set steps (per-set BB / ZMSHR1 scalars, per-set CG) are exact and fast only for a separable Hessian
     (toy: per-set CG 17 vs global PCG 21). With any coupling (LLR, temporal basis, temporal prior, velocity model) they
     diverge (toy: NaN for per-set CG; production: the secant groups of `red.rs` then collapse them to one global
     scalar). All our target problems are coupled.
   - Per-set *spatial* circulants: no. The per-encode densities are identical [M]; a single set deviates from the
     pooled density by 0.07–2.4× beyond 0.2 Nyquist [M]. That is incoherent aliasing, which no circulant represents,
     and per-set circulants cost 133 MB per set at 2 mm (toy: per-set C is no better than shared C + n_t).
4. **Nonsmooth priors: do not push the metric into the prox; smooth the prior and run a Krylov-type method [D, T].**
   - The prox of wavelet ℓ1 or LLR in a circulant metric has no closed form. An inexact inner solve costs 2 wavelet
     transforms (or one SVT sweep) per inner step per set, i.e. tens of % of a gradient per inner step at 1 mm (§3.2).
   - Use the Moreau-envelope (= RED with a proximal denoiser, `red.rs`) objective F_γ = f + quadratic priors +
     env_γ(λR) with a fixed γ and fixed LLR shifts. Its minimiser is metric-independent and is within O(γ) of the ℓ1
     minimiser. Toy: γ = 1/(bulk curvature) moves it by 0.2 % with the same truth error (0.0359 vs 0.0357).
   - Solve it by **nonlinear PCG (Polak–Ribière+) in the metric of item 2 with a data-exact line search**: one gradient
     per iteration, 4 host stacks (the same as ZMSHR1), estimator unchanged by any metric. L-BFGS(2–3) is the
     alternative if memory allows.
   - Toy numbers (§6.2), gradients to 1 % / 0.1 % of the minimiser:

     | method | 1 % | 0.1 % |
     |---|---|---|
     | NCG, K(ω, k) metric | 14 | 22 |
     | L-BFGS(3), same metric | 13 | 22 |
     | NCG, shared C | 19 | 30 |
     | ZMSHR1-M (RED-like), shared C | 25–33 | 41–50 |
     | FISTA, exact prox | 107 | — |

   - γ = 0.1/bulk is as fast as γ = 1/bulk and lands 0.03 % from the exact ℓ1 minimiser, so the smoothing is
     essentially free here.
5. **ADMM: the user's scepticism is justified as a default, and is not needed [D, T, M].**
   - The ADMM-PCG speed of `convergence_p2np.md` came from putting the circulant into exact inner solves, not from ADMM
     itself, and it needs ρ tuned per λ (a 50× mistuned ρ stalls [M]).
   - A smoothed single-loop Krylov method keeps that advantage without the outer O(1/k) loop or ρ.
   - For exact nonsmooth priors, PDHG with a k-space (dual) preconditioner (Ong–Uecker–Lustig 2020 [A]) is the
     estimator-preserving alternative: DCW-like conditioning sits in the dual, so the estimator is untouched.
     Toy: 34 / 82 gradients to 1 % / 0.1 %, against 172–190 / — for scalar PDHG. That is 5× from the dual
     preconditioner, but still 2.4–3.7× behind NCG.
   - ADMM-PCG with the same K metric: 18 / 41 at its best ρ (0.3 × bulk, 3 inner), but 129 / — with ρ 10× off.
6. **Evolution over the iterations [D, T, M].**
   - The metric is right from iteration 0, since it needs no data. Do not "switch from circulant to quasi-Newton":
     the Krylov memory *is* the dynamic part, built on top of the fixed metric.
   - Restart the Krylov memory at events: an operator refit (stencil epochs), a change of LLR shift set, a γ / λ
     continuation stage, an update of the prior-aware metric term (the active set or rank, at iterations ≈ 5, 10, 20).
   - Unregularised runs semi-converge, and **the early-stopped estimator depends on the metric**. Toy: best truth error
     0.122 with the identity metric (implicit low-pass), 0.179 with the circulant, against 0.100 for the converged
     regularised estimator, reached in 11 PCG iterations.
   - R2's growing error is therefore expected behaviour of an unregularised preconditioned solver, not a solver
     defect. Add the prior, and stop by held-out k-space only in the unregularised case. Held-out picked iteration
     18 vs the truth's 15 [T].
7. **Velocity model [D].**
   - Its current metric (`vobj.rs`: pointwise Gauss–Newton blocks B⁻¹) has no k-space equalisation. That is the
     zmshr1s failure mode (broad PSF, velocity shrink [M, 2026-11-12]) in a new place.
   - Use the sandwich metric P_v = B^{-1/2} (C⁻¹ ⊗ I_p) B^{-1/2}, with one shared circulant (≈ 0.34 GB at 1 mm, not
     8 GB). It is exact up to the commutator of C with the phase field U, which is small except in fast lumens
     (§2.6).
   - Algorithm: NCG in P_v. It is single-loop and the closest to P²nP, with a Gauss–Newton line-search model that
     costs one forward per iteration. The arm to beat it is inexact Gauss–Newton–CG (5–10 inner).
8. **Ranked plan (§8):**
   - (1) K(ω, k) / G⁻¹⊗C⁻¹ metric with per-set scalars + PCG on the knot problem with quadratic priors;
   - (2) NCG on the smoothed wavelet + LLR objective;
   - (3) sandwich metric for the velocity model;
   - (4) PDHG with k-space preconditioning as the exact-ℓ1 arm;
   - (5) γ-continuation and restarts;
   - (6) SVRG coil subsets only with first-order methods.

   Expected [model]: 15–30 gradients to 1 % of the minimiser for (a), against ≥ 80–150 for ZMSHR1 / FISTA, with the
   same memory (4 stacks).

---------------------------------------------------------------------------------------------------------------------

## 1. Problem structure

- **Unknowns.** x = (x_t), t ∈ sets: frames × encodes, or knots κ = (respiratory knot, cardiac knot) × encodes; the
  velocity model has θ = (z, v, φ) per temporal node.
- **Objective.** F(x) = f(x) + R(x), f = ½ Σ_t ‖A_t x_t − y_t‖² (whitened, no DCW: feedback rule), R = wavelet ℓ1 /
  LLR nuclear / learned (RED or PnP) / velocity model priors.
- **Hessian of f.** H = blockdiag(H_t) for per-frame sets. H_t = Σ_c Σ_l S_cᴴ D_lᴴ T_t D_l S_c, where T_t = F_tᴴ F_t is
  Toeplitz with the set's sampling measure as its symbol.
  - With knots, H couples knots through the temporal weights: H_κκ′ = Σ_j w_κ(t_j) w_κ′(t_j) a_jᴴ a_j.
  - With the velocity model, H_GN = Σ_{e,f} J_{ef}ᴴ H_{ef} J_{ef}, J pointwise.
- **Cost unit.** One gradient = one A and one Aᴴ over all sets (152.5 s at 1 mm in R2, L = 10, 280 sets [M,
  box_recon.md]).
  - A coil-free circulant application costs 4.3 ms host-resident at 2 mm, against 1.7 s per set's gradient [M,
    recon_diag 11-12 §5]: about 1/400.
  - A 3-D db4 wavelet transform or an LLR sweep is *not* negligible at 1 mm (§3.2).

## 2. Spectra of the preconditioned operator, metric by metric

### 2.1 One set: the circulant metric

- **Symbol.** By the Szegő / GLT picture [R: Serra-Capizzano; Chan & Ng 1996, SIAM Rev. 38:427, doi
  10.1137/S0036144594276474], T_t has the symbol ρ_t(k), the (Fejér-smoothed) sampling density.
- **Metric.** `precond.rs` uses λ_t(k) = (ρ_t ⋆ P_c)(k), with P_c the coil power spectrum, floored at 10⁻³ λ_max, and
  M_t = F⁻¹ diag(1/λ_t) F. In the toy this is the exact diagonal of F H_t Fᴴ to 15 digits [T: `probe`].
- **Spectrum of M_t H_t.** A cluster around 1 for the part of H_t that the circulant represents (shift-invariant
  sampling × flat coil energy). The deviations come from three sources:
  - (i) **coil energy variation** Σ|S_c(x)|² across the FOV: a multiplicative spread, removed by the D^{-1/2}
    normalisation;
  - (ii) the **floor** (k-space corners, ≈ 0 % inside Nyquist [M]);
  - (iii) the **non-circulant sampling fluctuation** E_t below, which dominates.
- **Per-set fluctuation.** Write H_t = n_t(H̄ + E_t), with H̄ the pooled operator per unit weight and E_t the set's
  incoherent aliasing (E[E_t] = 0 over interleaved views). Then
  M̄ H_t ≈ n_t (I + M̄ E_t).
  - **Measured size:** a single set's density over the pooled one is 0.07–2.4 (p10–p90) beyond 0.2 Nyquist, at
    380 of 7592 rotations per set [M]. So ‖M̄E_t‖ = O(1) at high k: the per-set problem has a near-null cluster that
    *no* circulant removes. At R ≈ 50 per (frame, encode) the per-set κ is effectively infinite [M, box_recon R1].
  - **Consequence.** Per-set problems are made well posed only by coupling (pooling, a temporal basis, a prior). The
    coupled Hessian, not H_t, sets the convergence rate (§2.3). The circulant is right for the *mean* and wrong for
    the fluctuation by construction.

### 2.2 Many sets: shared vs per-set metric, and what "per-set step sizes" are

- **Shared metric without scale.** P = I ⊗ C̄⁻¹ on blockdiag(n_t(H̄ + E_t)). The spectrum is ⋃_t n_t σ(I + C̄⁻¹E_t),
  so κ(PH) ≈ (n_max/n_min) κ₀ [D].
  - Gradient descent with one step pays the full spread: iterations ∝ s κ₀, s = n_max/n_min.
  - The PCG bound pays √s, but on a union of scaled copies CG does better than the bound, because the polynomial can
    place roots per cluster [R: van der Sluis & van der Vorst 1986, Numer. Math. 48:543, doi 10.1007/BF01389450].
- **Per-set step sizes are exactly a block-diagonal scalar metric.** x_t ← x_t − α_t g_t is gradient descent in
  P = diag(α_t) ⊗ I. The ideal α_t is 1/(n_t λ_max(C̄⁻¹H̄)), known from the sample counts.
  - Block van der Sluis: equalising the block scales is within a factor (number of nonzero blocks per row) of the best
    block-diagonal scaling [R: van der Sluis 1969, Numer. Math. 14:14, doi 10.1007/BF02165096; block form CHECK].
  - Cheap estimates of n_t, in increasing cost:
    1. sample count × mean coil energy × the knot's basis weight energy G_κκ (free);
    2. λ_max(C_t⁻¹H_t) from per-set circulants (not affordable at 1 mm);
    3. a **block power iteration**: one global power iteration on the separable H with per-block Rayleigh quotients
       gives every L_t for the cost of one global estimate.
- **Measured spread.** In R2, 28–45 readouts per set (s = 1.6) [M, R2 log]. Bellows binning with unequal bin
  occupancy, or the end knots of a non-periodic respiratory basis (half weight), can reach 2–4. The 16× τ_t spread of
  ZMSHR1 [M, convergence_p2np §0.3] is BB dynamics, not density: τ ranged 0.48–4.9 per set at R2 iteration 20 [M].
- **Toy E1** (8 sets, coupled, density 10³), iterations to 1 % with PCG:

  | spread s | shared C | C + n_t | per-set C | K(ω, k) (n_t + prior) |
  |---|---|---|---|---|
  | 1.6 | 19 | 19 | 21 | 14 |
  | 16 | 44 | 35 | 37 | 16 |

  - GD with shared C at s = 16 does not reach 10 % in 300 iterations; with K(ω, k) it reaches 1 % in 110.
  - Per-set circulants buy nothing over shared C + n_t, as §2.1 predicts.
- **Coupling changes what the right scale is.** With Q = N^{1/2}(I ⊗ H̄)N^{1/2} + μ L ⊗ I (L the temporal
  Laplacian), scaling only the data part gives N^{-1/2}QN^{-1/2} = I ⊗ H̄ + μ N^{-1/2} L N^{-1/2}. The coupling is
  amplified by 1/n_min, so the low-weight sets become stiff in the prior.
  - Toy: GD with C + n_t at s = 16 needs 265 iterations to 10 %, GD with K(ω, k) 42.
  - Krylov methods are robust to this (35 vs 16), first-order methods are not.

### 2.3 Knots: the Kronecker metric and the temporal Gram

- **Approximation.** If the sample times are independent of the k-locations (interleaved views), then
  E[AᴴA] = G ⊗ H̄_s, with G the K × K temporal Gram of the basis under the sample times (`knotsolve.rs`). Then
  P = G⁻¹ ⊗ C⁻¹ gives P AᴴA = I + (G⁻¹ ⊗ C⁻¹)Δ, where Δ is the per-knot fluctuation (§2.1, now pooled over each
  knot's support: 4 intervals for cubic, so a smaller fluctuation than per frame).
- **What G does without its inverse [D].**
  - For uniform periodic knots G is circulant with symbol ĝ(ω) = Σ_m b_{2p+1}(m) e^{iωm} (the degree-2p+1 B-spline
    sampled at the integers: Euler–Frobenius).
  - Its minimum is at ω = π, the highest temporal frequency the knots carry. cond(G) = ĝ(0)/ĝ(π) = 18.5 for cubic
    and 113 for quintic, matching the measured values [M, temporal_operator §13].
  - Without G⁻¹, the cardiac dynamics (high ω) converge 18.5× (GD) or ≈ 4.3× (CG, √) slower than the temporal mean.
    This is exactly the "slope grows slowly" signature of §16.2.
  - R2 used an orthonormal Fourier tsub (λ_max(TᵀT) = 1.000 [M]), so this mechanism does not explain R2. It does
    apply to every spline-knot run.
- **Per-knot scale.** It is the diagonal of G (the sample weight energy of each knot), already inside G⁻¹. A
  non-periodic respiratory basis has end knots with about half weight, which G⁻¹ handles.
- **Prior term.**
  - A temporal-Fourier quadratic prior (cardiac low-pass, temporal Tikhonov) is jointly diagonal with a circulant G,
    so (ĝ(ω) λ(k) + μ ℓ(ω))⁻¹ is exact up to Δ.
  - A non-periodic respiratory basis with a periodic cardiac basis gives a 2-D (resp × cardiac) G: small (≤ 127²),
    inverted per k-bin, or factorised G_r ⊗ G_c when sampling is uncorrelated between the two phases (the cardiac and
    respiratory periods are incommensurate) [D, CHECK on the bellows data].

### 2.4 Quasi-Newton on top of a fixed metric: the Krylov bound (Theorem K)

**Theorem K [D].** Let Q ≻ 0, P ≻ 0 be fixed, F(x) = ½xᴴQx − Re⟨b, x⟩, r₀ = b − Qx₀. Consider any method whose
iterates satisfy x_k ∈ x₀ + 𝒦_k := x₀ + span{P r₀, (PQ) P r₀, …, (PQ)^{k−1} P r₀}. Then
‖x_k − x*‖_Q ≥ ‖x_k^PCG − x*‖_Q, where PCG is preconditioned CG with P from x₀.

*Proof.* In w = P^{-1/2}x the problem has Hessian Q̃ = P^{1/2}QP^{1/2}, and 𝒦_k maps to 𝒦_k(Q̃, r̃₀). PCG is CG on
Q̃, which minimises ‖w − w*‖_Q̃ over w₀ + 𝒦_k(Q̃, r̃₀) [R: Hestenes–Stiefel]. The Q̃-norm in w equals the Q-norm
in x. ∎

**Which methods are in the class [D].** Each uses one gradient per iteration and only linear combinations of past
P-scaled gradients.
- **Gradient descent and Richardson** with any step sequence (constant, BB1/BB2, Malitsky–Mishchenko, line search):
  x_{k+1} − x_k ∈ P g_k, and g_k = −r₀ + Q(x_k − x₀) ∈ r₀ + Q𝒦_k.
- **Heavy ball, Nesterov, FISTA (smooth part), Chebyshev:** linear combinations of the above.
- **ZMSHR1 in the metric P (global; `Precond::Zmshr1M`).** In the w coordinates P̃_k = τ_k I + ũũᴴ/⟨ũ, ṽ⟩ with
  ũ = s̃ − τṽ and ṽ = a s̃ + (1 − a)Q̃s̃. Both s̃ and ṽ lie in 𝒦_k, so P̃_k g̃_k = τ g̃_k + c ũ ∈ 𝒦_{k+1}. Induction.
  The θ₁/θ₂ safeguard only changes a, so it stays in the class.
- **Anderson(m)** on the preconditioned gradient map: equivalent to GMRES on linear problems [A: Walker & Ni 2011,
  SIAM J. Numer. Anal. 49:1715, doi 10.1137/10078356X], hence in 𝒦_k.
- **L-BFGS with H₀ = cP and exact line search:** it coincides with PCG on quadratics [R: Nazareth 1979, SIAM J.
  Numer. Anal. 16:794, doi 10.1137/0716059].

**Not in the class**, and therefore able to beat global PCG:
- per-set (block-separate) methods. Their space is ⊕_t 𝒦_k(P_tQ_tt, ·) ⊋ 𝒦_k, which is valid **only** when Q is
  block-diagonal;
- methods that change P with information outside the Krylov space (a better static metric).

**Toy check** (energy-norm error at iteration 20, E1, s = 1.6):

| metric | GD | Nesterov | ZMSHR1-M | PCG |
|---|---|---|---|---|
| identity | 2.6e-2 | 1.0e-2 | 7.4e-3 | 5.1e-3 |
| shared C | 3.9e-1 | 6.5e-2 | 1.4e-2 | 1.6e-3 |
| K(ω, k) | 2.4e-1 | 5.2e-2 | 2.4e-3 | 2.0e-4 |

PCG is lowest in every row, as the theorem requires. E2 (separable, s = 16) shows the exception: per-set CG reaches 1 %
in 17, per-set ZMSHR1-M in 26, global PCG with C + n_t in 21. Under coupling the same per-set CG produces NaN (E1, every
metric but I and K, where it stalls).

**What this means for "dynamic beats circulant" (the user's 2026-11-10 expectation).**
- Against *no* metric it does: ZMSHR1-M|I needs 150 iterations, PCG|I 85, ZMSHR1-M|C 34.
- Against a circulant *with CG* it cannot, on the quadratic part. The dynamic information that ZMSHR1 extracts (one
  secant direction per iteration) is a subset of what CG extracts (all past directions, optimally combined).
- What a dynamic metric adds legitimately is curvature that is *not* quadratic: the prior's active set or rank, the
  velocity Jacobian. In the Krylov family that is handled by nonlinear CG / L-BFGS (restarts, line search), not by a
  zero-memory BB step.

### 2.5 Prior coupling: the prior-aware metric

- **Quadratic priors (exact).** The Hessian is Q = H + Σ_prior. Temporal Tikhonov / cardiac low-pass / static lock:
  Σ_prior = μ L_t ⊗ I or μ(I − Π). The knot Gram is circulant in time, the data symbol circulant in space, so
  P = N^{-1/2}(ĝ(ω)λ(k) + μ ℓ(ω))⁻¹N^{-1/2} costs one FFT over sets and one over space [T: K(ω, k)].
- **Smoothed nonsmooth priors (local).**
  - The envelope of λ‖W·‖₁ has Hessian (1/γ) Wᴴ Π_Z W, with Π_Z the projector onto the thresholded (zeroed)
    coefficients. Haar / db4 subbands are octave bands, so the subband-averaged zero fraction z_j(k) gives an
    approximately circulant term Σ_j z_j/γ · B_j(k), with B_j the band indicator. Prior-aware metric:
    λ(k) + Σ_j z_j(k)/γ.
  - The LLR envelope has Hessian (1/γ)(I − 𝒥_SVT) blockwise ≈ (1/γ)(I − Ū_rŪ_rᴴ) in time, with Ū_r the dominant
    temporal subspace. A Fourier-like Ū_r makes it diagonal in ω: (ĝ(ω)λ(k) + ζ(ω)/γ)⁻¹, with ζ(ω) the discarded
    energy fraction per temporal frequency.
  - Both terms are static between active-set updates. They are cheap to re-estimate (one wavelet transform / one LLR
    sweep of the iterate), and a re-estimate is a restart event (§5).
- **Why it matters.** The data-only metric makes the prior *stiffer*: C⁻¹ is up to 10³ larger at high k, where the
  prior acts. Toy E1: the prior-aware K(ω, k) beats shared C by 1.4–2.7× for PCG and 1.6–3× for ZMSHR1, and it makes
  GD usable at s = 16.
- **What it does not do.** It does not change the estimator: for a convex smoothed F the minimiser is metric-free
  (`red_prior.md` §2, `denoiser_artifact_theory.md` Thm 1 applies only to PnP-type schemes).

### 2.6 Velocity model: the sandwich metric

- **Gauss–Newton Hessian.** H_GN = Σ_{e,f} J_{ef}ᴴ H_{ef} J_{ef}, with J pointwise (per voxel: z complex, v ∈ ℝ³,
  residual phase). The 7 encode densities are identical [M], so H_{ef} ≈ n_{ef} D^{1/2} C D^{1/2} with one C.
- **Current metric.** `vobj.rs` uses B = Σ n_{ef} J_{ef}ᴴ n̂ J_{ef} pointwise (3 × 3 for v): the k-space equalisation
  is missing. This is the same failure mode as `zmshr1s` (identity metric → broad effective PSF → velocity shrink:
  slope 0.41 at the wall, 0.71 deep [M, recon_diag 11-12 §4]).
- **Sandwich metric [D].** Take the pointwise polar decomposition J = U B^{1/2}, with B = JᴴJ (the per-voxel
  Gauss–Newton block, ⊗ n) and U a per-voxel isometry. U carries the encode phases e^{i(k_e·v + φ_e)} and the phase of
  z. With C acting per image channel,
  Jᴴ (C ⊗ I) J = B^{1/2} (C ⊗ I_p + Uᴴ[C, U]) B^{1/2}   (exact).
  - So P_v = B^{-1/2}(C⁻¹ ⊗ I_p)B^{-1/2} is similar to I + (C⁻¹ ⊗ I) Uᴴ[C, U]. The magnitude structure (|z| jumps at
    walls, partial volume) is handled exactly by B; only the spatial variation of the *phase field* U enters.
  - For a multiplier, ‖[C, U]‖ ≲ ‖∇U‖ × (first moment of C's kernel, the inverse-density PSF without its DC spike).
    Here ‖∇U‖ ≈ |k_e| ‖∇v‖ + ‖∇φ‖.
  - Background phase is smooth, so its error is small. In fast lumens it is not: at VENC 1.5 m/s and ≈ 1 m/s across
    3 voxels the phase moves ≈ 0.7 rad per voxel, so the error is O(1) there. The sandwich is then a good but not an
    exact metric, and the Krylov / NCG iteration absorbs the rest.
  - Without C (today) the error is the full density spread (10³), everywhere.
- **Cost and memory.** Two applications per metric step (B^{-1/2} pointwise, C⁻¹ per parameter image: p = 2 complex
  + 3 real FFT channels per node). One C for the scan: 5 N-grid arrays ≈ 0.34 GB at 1 mm, against the "≈ 8 GB" for
  per-set kernels noted in `recon_p2np.rs`.

## 3. Algorithms for min f + R

### 3.1 What must stay invariant

- **The estimator.**
  - For a convex variational F: any convergent method, any metric gives argmin F.
  - For PnP with a fixed-strength denoiser: the fixed point depends on (α, P) (paper prop:estimator,
    thm:displacement), and only prop:keepfp-type acceleration preserves the *reference* fixed point.
  - For RED with a conservative residual: the stationary points of f + λR are metric-free (`red_prior.md`).
- **The single loop.** The user prefers one gradient per iteration with a dynamic metric (P²nP family). Theorem K
  says the optimal member of that family on the quadratic part is (nonlinear) CG.

### 3.2 Proximal gradient / FISTA in a metric: when is the prox compatible?

Variable-metric forward–backward x⁺ = prox^P_{αR}(x − αP∇f) [R: Chouzenoux, Pesquet, Repetti 2014, JOTA 162:107,
doi 10.1007/s10957-013-0465-7; Combettes & Vũ 2014, Optimization 63:1289, doi 10.1080/02331934.2012.733883] needs the
prox in the metric P⁻¹. Cases:

| metric | wavelet ℓ1 (synthesis-orthonormal W) | LLR (blocks over all sets) | learned / RED |
|---|---|---|---|
| scalar | exact | exact | n/a (PnP) |
| diagonal in W (per subband × set) | **exact** (soft threshold with λ/d_j) | no | — |
| per-set scalars | exact (separable per set) | weighted nuclear norm: a different R (`algo_speed` §4) | — |
| circulant / Kronecker | inexact inner solve | inexact inner solve | — |
| ± rank one (ZMSHR1) on a diagonal D | 1-D root finding on prox_D [A: Becker, Fadili, Ochs 2019, SIAM J. Optim. 29:2445, doi 10.1137/18M1167152] | needs prox_D | — |

- **Subband-diagonal majoriser.** For wavelet ℓ1 the subband × set diagonal majoriser
  d_{t,j} ≥ Σ_{j′} ‖W_j H_t W_{j′}ᴴ‖ + 4μ (block Schur test) is the BARISTA idea [A: Muckley, Noll, Fessler 2015,
  IEEE TMI 34:578, doi 10.1109/TMI.2014.2363034]: an exact prox with per-band steps. It is limited by the
  Gershgorin overestimate of the inter-band coupling (Haar's poor frequency localisation). Toy: 101 / 254 gradients to 1 % / 0.1 %, against 107 / — for FISTA-I. The inter-band Gershgorin sums eat the gain
  (d_coarse ≈ 90 vs L = 72; d_finest ≈ 8.8).
- **Inexact circulant-metric prox.** Allowed by VMILA-type theory with verifiable inexactness [A: Bonettini, Loris,
  Porta, Prato 2016, SIAM J. Optim. 26:891, doi 10.1137/15M1019325]. For accelerated methods the errors must be
  summable with weight k [A: Schmidt, Le Roux, Bach 2011, arXiv:1109.2415].
  - Cost per inner step per set: two circulant applications (≈ 1/200 of the set's gradient) plus two wavelet
    transforms (3-D db4 on 8.5 M voxels: tens of ms [R, wavelet_prior §3]).
  - Over 280 sets, 5 inner steps cost ≈ 0.3–0.5 gradients. That is the price of exactness, paid every iteration.
  - Not recommended unless the smoothed route fails.

### 3.3 Smoothing + Krylov (recommended for classical priors)

- **Objective.** F_γ(x) = f(x) + q(x) + Σ_i e_γ(λ_i R_i)(x), with q the quadratic priors, e_γ the Moreau envelope
  (Huber for ℓ1, the singular-value Huber for LLR), fixed γ, fixed LLR shift set.
- **Estimator.**
  - ‖x_γ − x*‖ = O(γ) under quadratic growth of F near x* (O(√γ) in general).
  - |e_γ − R| ≤ γ L_R²/2 [R: Nesterov 2005, Math. Program. 103:127, doi 10.1007/s10107-004-0552-5; Beck & Teboulle
    2012, SIAM J. Optim. 22:557, doi 10.1137/100818327].
  - Toy: γ = 1/bulk → ‖x_γ − x*_ℓ1‖ = 0.2 % and identical truth error [T].
  - It is exactly RED with a proximal denoiser at θ = λγ and RED weight 1/γ (`red.rs`). The choice ρ = 1 there is
    γ ≈ α/μ_metric.
- **Method: nonlinear PCG (PR+) in the metric of §2.**
  d_k = −P g_k + β_k d_{k−1}, β = max(0, ⟨g_k − g_{k−1}, P g_k⟩/⟨g_{k−1}, P g_{k−1}⟩), restart when ⟨d, g⟩ ≥ 0, or by
  Powell's test |⟨g_k, P g_{k−1}⟩| ≥ 0.2 ⟨g_k, P g_k⟩.
- **Data-exact line search at no extra gradient [D].**
  - f is quadratic, so f(x + a d) = f(x) + a Re⟨∇f, d⟩ + ½a²‖A d‖², and ∇f(x + a d) = ∇f(x) + a Aᴴ(A d).
  - In CGLS form: one forward A d (gives ‖Ad‖² and the residual update r ← r − aAd), then one adjoint Aᴴ r: exactly
    one gradient pass per iteration, with the f32-safe explicit residual of `cg.rs`.
  - The prior part of φ′(a) is evaluated in the coefficient domain after transforming x and d once (wavelet: 3
    transforms per iteration in total; LLR: 1–2 SVT sweeps for a secant step on φ′).
  - Without the k-space residual (≈ 10 GB at 1 mm), use the image-space recurrence g ← g + a AᴴA d with an exact
    refresh every ≈ 10 iterations (1.1 gradients per iteration).
- **Memory.** x, g, d, P g_prev: 4 stacks. That is the same as ZMSHR1 (x, g, x_prev, g_prev). L-BFGS(m) needs
  3 + 2m stacks; m = 2 in f16 pairs is ≈ 5 stack-equivalents.
- **Convergence.** For strongly convex C² F_γ, PR+ with exact line search and restarts is n-step quadratically
  convergent and globally convergent [R: Hager & Zhang 2006, Pac. J. Optim. 2:35]. Near x_γ the active set is fixed,
  F_γ is locally quadratic and the iteration *is* PCG on ∇²F_γ, so Theorem K's optimality applies locally.
- **Continuation in γ** (NESTA [A: Becker, Bobin, Candès 2011, SIAM J. Imaging Sci. 4:1, doi 10.1137/090756855]):
  a large γ is well conditioned; decrease γ × 1/4 per stage with a warm start and restart. In the toy a fixed γ = 0.1/bulk needed no continuation (E3), so continuation is an experiment
  (P6), not a default.

### 3.4 ADMM with PCG inner solves (the user's scepticism)

- **The user's objection (2026-11-10):** ADMM has a poor convergence rate, the accelerated variants are not much
  better, and the point was that a dynamic preconditioner beats a circulant one. This note agrees on the first part,
  on the evidence below.
- **What our toy showed** (`convergence_p2np.md` §3): ADMM-PCG reached 1 % in 288 applications against 1091 for
  FISTA. But FISTA there ran in the *identity* metric, and ADMM-PCG ran its inner solves in the circulant metric. The
  gain came from the metric.
- **The outer loop** is Douglas–Rachford on the dual: O(1/k) in general, linear under regularity, with a rate set by
  ρ relative to the restricted spectrum [R: Giselsson & Boyd 2017, IEEE TAC 62:532, doi 10.1109/TAC.2016.2564160].
  [M]: ρ tuned for a 50× larger λ stalls.
- **Inexact inner solves** need increasing accuracy, i.e. growing inner counts.
- **Verdict.** Keep ADMM-PCG as a reference arm (it is the exact-ℓ1 solver with the circulant), not as the
  production loop. The smoothed Krylov method (§3.3) gets the metric without the outer loop. Toy (E3, same K preconditioner inside, gradients to 1 % / 0.1 % of the exact ℓ1 minimiser):

  | ρ / bulk | 3 inner | 8 inner |
  |---|---|---|
  | 0.3 | 18 / 41 | 19 / 65 |
  | 1 | 22 / 49 | 41 / 113 |
  | 3 | 46 / 127 | 129 / — |

  NCG on the smoothed objective: 14 / 22, with no ρ. ADMM's best is within 1.3–1.9× of NCG; its typical setting is
  2–6× slower.

### 3.5 Primal–dual with a k-space (dual) preconditioner

- **Formulation.** PDHG / Condat–Vũ with the data term dualised: u ∈ k-space, x⁺ = prox_{τR}(x − τ(Aᴴu + ∇q)),
  u⁺ = (u + Σ(A(2x⁺ − x) − y))/(1 + Σ) [R: Chambolle & Pock 2011, JMIV 40:120, doi 10.1007/s10851-010-0251-1;
  Condat 2013, JOTA 158:460, doi 10.1007/s10957-012-0245-9; Pock & Chambolle 2011, ICCV, doi
  10.1109/ICCV.2011.6126441].
- **The k-space preconditioner Σ ≈ 1/density** is DCW placed in the dual. The saddle point, hence the estimator, is
  unchanged; the prox is Euclidean (exact for ℓ1 and LLR).
  - Ong, Uecker, Lustig (IEEE TMI 39:1646, 2020, doi 10.1109/TMI.2019.2954121, arXiv:1902.09657) [A]: ℓ2-optimised
    single- and multi-channel k-space preconditioners; PDHG converges in about 10 iterations on 3-D UTE, where FISTA
    and plain PDHG are still blurred after 100.
  - The same group used it at our scale with stochastic gradients over frames (Extreme MRI, Ong et al. MRM 84:1763,
    2020, doi 10.1002/mrm.28235, arXiv:1909.13482) [A].
- **Cost.** One A and one Aᴴ per iteration (= 1 gradient), plus the prox.
- **Memory.** Dual u = data size (9.96 GB at 1 mm with 22 virtual coils, streamed per set like the KStore; a write
  per iteration, so host-resident, not SSD: feedback_ssd_writes), plus x and x_prev.
- **Rate.** O(1/k) ergodic in general; fast in practice when Σ makes ΣᵀAAᴴ near-identity on the sampled band.
- Toy (E5, single-channel Σ = c/w, wavelet prox exact; gradients to 10 % / 1 % / 0.1 %):

  | Σ | 10 % | 1 % | 0.1 % |
  |---|---|---|---|
  | 0.3 / w | 21 | 50 | 79 |
  | **1 / w** | 10 | **34** | 82 |
  | 3 / w | 9 | 73 | 188 |
  | scalar σ (best of 3) | 19–31 | 172–190 | — |

  The dual preconditioner gives the early "DCW-like" drop (10 % in 10 gradients). The O(1/k)-type tail remains, and
  the c balance matters 2×.

### 3.6 Learned priors: RED, PnP, and the fixed-point-preserving accelerator

- **RED with a conservative residual** (gradient-step denoiser, or an MMSE network with a near-symmetric Jacobian):
  F_RED = f + λR_θ is a smooth objective, so §3.3 applies unchanged (NCG with a backtracking line search on the
  RED objective when R_θ is available, or a Wolfe search on ∇ only).
- **PnP / non-conservative D.** There is no objective. Accelerate the reference fixed point with prop:keepfp,
  z⁺ = z − B_k F(z), B = (I − J) + βPJ, and wrap it in **Anderson(m)** on F [A: Walker & Ni 2011; Zhang, O'Donoghue,
  Boyd 2020, SIAM J. Optim. 30:3170, doi 10.1137/18M1232772 (safeguarded type-I AA), arXiv:1808.03971].
  - On a linearised problem Anderson = GMRES (§2.4), which is the best Krylov method for a non-symmetric fixed-point
    map. GMRES, not CG, is right here because J_D(I − αPH) is non-normal (paper: the ρ-version of Hong et al.'s
    Theorem 1 is false).
  - Cost per iteration: 1 gradient + 1 denoiser + 1 JVP; memory 2m stacks.

### 3.7 Inexact gradients: lagged, coil subsets, physics refits

- **Krylov relaxation.** In inexact Krylov methods the matvec error may *grow* as the residual falls,
  ε_k ≲ ε/‖r_k‖ [A: Simoncini & Szyld 2003, SIAM J. Sci. Comput. 25:454; R: van den Eshof & Sleijpen 2004, SIAM J.
  Matrix Anal. Appl. 26:125].
  - Early iterations need accurate operators; late ones tolerate cheap surrogates.
  - This favours `--lagged` (surrogate gradient) *late*, and exact physics *early*.
- **Stochastic coil subsets** break conjugacy (CG needs the same operator in every iteration).
  - With CG / NCG, use them only in the coarse or warm phase, or as a *preconditioner* (a 4-coil normal operator
    solved inexactly inside flexible CG [R: Notay 2000, SIAM J. Sci. Comput. 22:1444]).
  - With first-order / PDHG methods, SVRG anchoring is the right form [M, algo_speed §3: FISTA + SVRG 1.6–6× closer
    at equal cost; the anchor gives exact secants].
- **Physics refits** (stencil_weighted_update §5) change Q. Each accepted refit is an operator epoch: restart the
  Krylov memory (d ← −Pg, drop L-BFGS pairs), keep the metric. Its Proposition 4 (summable perturbations) keeps
  convergence.

### 3.8 Multilevel: pyramid as warm start, not as preconditioner

- **After prolongation**, the fine-level error is the band k > k_c plus temporal detail. That is the bulk of k-space,
  which the circulant metric already equalises.
  - So a warm start saves the few PCG iterations that fit the low band (psfcheck: a pooled circulant PCGLS reaches
    0.5 of a point's amplitude in 10 iterations [M]). It saves far more for unpreconditioned methods, which explains
    the historic value of the pyramid with zmshr1s.
- **A two-level preconditioner** (MG/OPT, IML-FISTA [R: Nash 2000; Lauga et al. 2024, arXiv:2304.13329]) pays off
  only if the smoother leaves low-frequency error, which a density-equalising metric does not.
- **Keep:**
  - the temporal pyramid as a start (pooled → coarse knots: B-splines nest exactly);
  - the spatial pyramid for the air map and the support;
  - λ scaled per level by σ_ℓ² ∝ 1/samples (`convergence_p2np` §6).

## 4. Per-block step sizes and metrics: rules

1. **Separable Hessian** (pure per-frame, no prior coupling): per-set CG (independent α_t, β_t) is optimal [T E2: 17
   vs 21]. Per-set ZMSHR1 is a cheaper approximation (26).
2. **Coupled Hessian** (LLR, temporal basis or prior, velocity model, RED with a coupling denoiser):
   - one global Krylov iteration with a block-diagonal-scaled metric;
   - per-set dynamic scalars are invalid: secant cross terms ⟨s_t, (Qs)_t⟩ can be negative (`red.rs` secant groups;
     toy NaN).
3. **Block scalars** come from static information (n_t, diag G). Rebuild them only when the operator changes.
4. **Adaptive steps** (BB, Malitsky–Mishchenko [A: ICML 2020, arXiv:1910.09529; prox version NeurIPS 2024]):
   valuable when no line search is cheap. Here the data-exact line search is free (§3.3), and Theorem K dominates
   them on quadratics. Use them only inside first-order arms (PDHG step balancing, FISTA).
5. **Priors coupling blocks and the prox.** Per-set metrics are incompatible with a coupled prox (LLR: weighted
   nuclear norm). In the smoothed / Krylov route there is no prox, so any SPD block metric is allowed. This
   decoupling of metric design from prox compatibility is the main structural argument for §3.3.
6. **Estimating the extreme eigenvalues.** λ_max(PQ) by 10–20 power iterations (lagged by one restart).
   λ_min of the restricted problem from the Lanczos coefficients of PCG itself: free, and it gives the convergence
   prediction and a stopping test.

## 5. Evolution over the iterations

- **PCG phases** [R: van der Sluis & van der Vorst 1986; Simoncini & Szyld 2005, SIAM Rev. 47:247]:
  - (i) fast removal of the components in large clusters: the mean, and the well-sampled band;
  - (ii) a linear phase at rate (√κ − 1)/(√κ + 1);
  - (iii) superlinear acceleration once the Ritz values capture the extremes (the near-null fluctuation cluster of
    §2.1, lifted by the prior).
  - The toy shows it: PCG|K reaches 10 % → 1 % → 0.1 % at 7 / 14 / 32 iterations.
- **Semi-convergence without a prior** [R: Hanke 1995, *Conjugate gradient type methods for ill-posed problems*;
  Engl, Hanke, Neubauer 1996]. CG and Landweber are regularisation methods with the iteration count as parameter.
  The metric decides *which* components enter first:
  - the circulant fits high k early, together with its noise and aliasing;
  - the identity filters them (Landweber's low-pass).
  - Toy E4 (model error + noise):

    | method | best truth error (it) | held-out min (it) → truth error there | at 200 |
    |---|---|---|---|
    | PCG, identity | 0.122 (22) | 26 → 0.123 | 0.297 |
    | GD, identity | 0.122 (200) | 200 → 0.122 | 0.122 |
    | PCG, circulant | 0.179 (15) | 18 → 0.182 | 1.24 |
    | ZMSHR1-M, circulant | 0.180 (20) | 25 → 0.186 | 0.521 |
    | GD, circulant | 0.179 (108) | 182 → 0.189 | 0.192 |
    | **regularised (μ = 1–3 × bulk), PCG K(ω,k), converged** | **0.100–0.102 (11)** | — | — |

  - Reading:
    - held-out k-space tracks the truth minimum to within 3–5 iterations;
    - the early-stopped estimator is metric-dependent (0.12 vs 0.18), just as the PnP fixed point is;
    - the regularised estimator is better than either and converges in 11 iterations.
  - This is R2: f fell 2.61e13 → 1.62e13 between iterations 10 and 40 (≈ 8–16× the noise floor
    ½ M σ²(1–2) ≈ 1–2e12 [M, R2 log; σ convention CHECK]), while the error grew. The fitted directions are
    aliasing / model error amplified by the metric, not converged signal. **Add the prior; do not tune iteration
    counts.**
- **When to change the metric.**
  - The circulant / Kronecker part is fixed from iteration 0.
  - The prior-aware term (§2.5) is updated at iterations ≈ 5, 10, 20, then frozen once the zero fraction / kept rank
    changes by < 5 %. Each update restarts NCG.
  - Do not add a ZMSHR1 rank-one term on top of NCG: the CG direction already carries the secant information
    (Theorem K).
- **Restart events** (each: d ← −P g, drop L-BFGS pairs, keep x):
  - an accepted physics refit (stencil epochs at {end of warm start, 10, 20, 40, 80}, at most 4);
  - a change of LLR shift set (fixed cycles only);
  - a γ / λ continuation stage;
  - a prior-aware metric update;
  - a Powell restart test failing;
  - ⟨d, g⟩ ≥ 0.
- **Stopping** (regularised):
  - relative stationarity ‖P^{1/2}∇F_γ‖/‖P^{1/2}∇F_γ(x₀)‖ < 10⁻² … 10⁻³;
  - held-out residual plateau (< 0.5 % change over 5 iterations) as a sanity check, not as the regulariser.
- **Resumes.** The user asked (2026-11-17) why a resumed quasi-Newton run is sensitive to 10⁻⁶ perturbations. BB-type
  step sequences are chaotic on quadratics with many distinct eigenvalues [R: Fletcher 2005, "On the Barzilai–Borwein
  method", in *Optimization and Control with Applications*, doi 10.1007/0-387-24255-4_10; Dai & Fletcher 2005]: the
  step depends discontinuously on the iterate history, so perturbations grow until the iteration is near convergence.
  NCG with an exact line search and restarts is far less sensitive. Store d with the checkpoint, or restart on resume:
  losing the direction costs ≈ 1–2 iterations.

## 6. Evidence

### 6.1 Our measured history (summary)

| finding | source |
|---|---|
| Density ∝ 1/k² near the centre, flat over ≈ 90 % of the radius; circulant dynamic range 10³ at 1 mm; λ_max(MN) = 6.02 | convergence_p2np §0; R2 log |
| A point's PSF: circulant PCGLS reaches 0.506 / 0.540 / 0.561 at 10 / 20 / 100 iterations, no preconditioner 0.001 / 0.005 / 0.246 | recon_diag 11-12 §3 |
| zmshr1s (identity metric) → broad PSF, velocity slope 0.41 at the wall; the circulant fixes the deep shrink (0.71 → 0.88) | recon_diag 11-12 §4, §6 |
| FISTA ≈ 10× ISTA at 30 gradients; ZMSHR1 + step-tied prox diverges (τ up to 9 > 2/L) | algo_speed §2.1, §4 |
| SVRG coil subsets: 1.6–6× closer to x* at equal cost (FISTA) | algo_speed §3 |
| ADMM-PCG 288 vs FISTA(I) 1091 applications to 1 % (accurate λ) | convergence_p2np §3 |
| Metric toy: shared circulant fastest at its own best λ; keepfp reaches the reference limit to 1.9e-4 | paper.tex (metric_fixedpoint v2/v3) |
| Knot CG with G⁻¹ ⊗ C⁻¹: still improving at 40 iterations (unregularised); held-out ranks the bases like the truth | temporal_operator §19.1 (§16.3's table disagrees on 6 vs 4 knots: CHECK which is current) |
| R2 (tsub fourier:4, zmshr1ms, no prior): error grows 1.77 → 2.32 rel over iterations 10–40; per-set τ 0.48–4.9 | box_recon R2, R2 log |
| Per-set density vs pooled: 1.00 (< 0.05 Nyq), 0.62–0.68 [p10 0.07, p90 2.4] beyond 0.2 Nyq; the 7 encode densities are identical | recon_diag 11-12 §5 |

### 6.2 Toys of this note (`strategy_toy.py`, out_strategy/)

Setup:
- 64² complex phantom with a pulsing, phase-varying vessel, 8 sets, 4 coils;
- Cartesian multiplicities w_t(k) ~ Poisson(n_t c ρ(k)), ρ = 1/(1 + k²/k₀²), density range 10³;
- 58 % of k unsampled per set, 7 % pooled;
- temporal coupling μ = bulk curvature (truth-optimal over 0.03–3 × bulk);
- references solved to 10⁻¹³ (ℓ1: ADMM and FISTA-8000 agree to 2e-13, prox-gradient residual 2e-15).

- **E1 / E2 / E4:** tables above (§0, §2.2, §2.4, §5).
- **E3** (Haar ℓ1 + temporal coupling, s = 1.6; λ truth-optimal over 10⁻⁵–10⁻³ × max|W b|, giving truth error
  0.0357; gradients to 1 % / 0.1 % of the respective minimiser):

  | method | target | 1 % | 0.1 % |
  |---|---|---|---|
  | FISTA-I, exact prox, restart | ℓ1 | 107 | — |
  | FISTA, subband × set diagonal majoriser | ℓ1 | 101 | 254 |
  | ADMM-PCG, K metric, best ρ (0.3 bulk, 3 inner) | ℓ1 | 18 | 41 |
  | ADMM-PCG, ρ = 3 bulk, 8 inner | ℓ1 | 129 | — |
  | PDHG, k-space Σ = 1/w (E5) | ℓ1 | 34 | 82 |
  | NCG, identity | smooth γ = 1/bulk | 57 | 107 |
  | NCG, shared C | smooth | 19 | 30 |
  | **NCG, K(ω, k)** | smooth | **14** | **22** |
  | L-BFGS(3), K(ω, k) | smooth | 13 | 22 |
  | ZMSHR1-M, shared C, α from data (RED-like) | smooth | 32 | 45 |
  | ZMSHR1-M, shared C, α from data + prior bound | smooth | 25 | 41 |

  - The smoothed minimiser lies 0.20 % (γ = 1/bulk) or 0.026 % (γ = 0.1/bulk) from x*_ℓ1, with truth errors 0.0359 /
    0.0357.
  - The γ = 0.1 runs need the same gradients (NCG|K 14 / 23).
  - The toy line search used ≈ 33 prior evaluations per iteration (bisection). A production version needs 2–3 secant
    steps on φ′, which is exact for a piecewise quadratic envelope once the active set settles.
- **E5:** table in §3.5.

Caveats:
- 2-D, Cartesian with Poisson multiplicities (real: 3-D non-Cartesian, R ≈ 50 per set, so the per-set fluctuation of
  §2.1 is larger);
- a quadratic temporal coupling in place of LLR;
- orthonormal Haar in place of db4 with cycle spinning;
- noise 1 %.

The orderings (which metric, which algorithm) are what transfers. The iteration counts are a lower bound for the 1 mm
problem.

## 7. Literature (beyond the papers cited inline)

- Fessler, *Optimization methods for MR image reconstruction*, IEEE SPM 2020, arXiv:1903.03510 [A]: review; CG with
  circulant preconditioners, majorisers, momentum (OGM), BARISTA, ADMM.
- Ramani & Fessler 2011, IEEE TMI 30:694, doi 10.1109/TMI.2010.2093536 [R]; Weller, Ramani, Fessler 2014, IEEE TMI
  33:351, doi 10.1109/TMI.2013.2285046 [R]: AL / ADMM with circulant inner solves, faster than MFISTA in 2-D SENSE.
  The evidence for ADMM-PCG, and for "the metric matters".
- Koolstra et al. 2019, MRM 81:670, doi 10.1002/mrm.27371 [R]: circulant preconditioner in split Bregman (Cartesian).
- Hong, Xu, Hu, Fessler 2024, P²nP, arXiv:2405.03854 [V, earlier session]; Hong, Hernandez-Garcia, Fessler,
  complex quasi-Newton proximal CS-MRI, arXiv:2303.02586 [R]: diagonal ± rank-one metric with the Becker–Fadili prox.
- Lee, Sun, Saunders 2014, proximal Newton, SIAM J. Optim. 24:1420, doi 10.1137/130921428 [R]: inner prox solves in a
  quasi-Newton metric (the exact version of "circulant + dynamic" with a prox; costly inner loops).
- Nocedal 1980, Math. Comp. 35:773 [R]; Liu & Nocedal 1989, Math. Program. 45:503 [R]: L-BFGS with H₀.
- O'Donoghue & Candès 2015, FoCM 15:715, doi 10.1007/s10208-013-9150-3 [R]: adaptive restart.
- Wright, Nowak, Figueiredo 2009 (SpaRSA), IEEE TSP 57:2479 [R]: non-monotone BB + prox safeguard (`algo_speed` §4).
- Johnson & Zhang 2013 (SVRG); Xiao & Zhang 2014 (prox-SVRG), SIAM J. Optim. 24:2057 [R]; Allen-Zhu 2017
  (Katyusha), arXiv:1603.05953 [R].
- Hale, Yin, Zhang 2008 (FPC continuation), SIAM J. Optim. 19:1107 [R].

## 8. Ranked plan with pre-registered experiments

Rules (feedback_experiment_discipline):
- For every run, write the question, the expected outcome and the decision first.
- Print the config fingerprint (solver, **metric present and which**, per-set scalars on/off, priors with γ / λ /
  shifts, iterations, held-out fraction).
- Change one factor at a time, at dev scale (≤ 10 min) first, then 2 mm to confirm, then the 1 mm box.
- Score by distance to the regularised minimiser (a long reference run), by held-out residual, and by frame_fit /
  sig_fit (practice baseline R1 alongside).

**P1. Krylov core on the knot problem with quadratic priors** (code: a PCG over knot stacks exists in
`knotsolve.rs` + `cg::pcgls_dyn`; add the per-set scalar and the prior symbol to the Kronecker metric).
- *Question:* to reach the reference objective ±1 %, how many gradients do (a) zmshr1ms (production), (b) PCG with
  G⁻¹⊗C⁻¹ and (c) PCG with N^{-1/2}(G⊗C + μ ℓ(ω))⁻¹N^{-1/2} need? Prior: cardiac temporal Tikhonov
  (quadratic, exact).
- *Expected:* (c) ≤ (b) < (a)/2 (toy: 14 / 19 / 34).
- *Decision:* if (c) ≤ ½ (a) → (c) becomes the core of every quadratic solve (warm starts, pyramid levels,
  model-acceptance refits); if (a) ≤ (c) → the toy's ordering fails on the real operator: measure the spectrum
  (Lanczos from PCG) before anything else.
- *Cost:* dev scale 3 arms × 40 gradients; memory: 4 stacks each.

**P2. Smoothed classical priors (wavelet db4 + LLR, fixed shifts, fixed γ) by NCG** (new solver loop: NCG / L-BFGS
in `hoststack.rs` next to `run_objective_host`; RED wrapper already gives ∇).
- *Question:* NCG(P from P1 + prior-aware term) vs zmshr1ms-RED vs FISTA (Euclidean, exact prox) on the same F_γ, in
  gradients to 1 % of x_γ; also the distance x_γ ↔ x*_ℓ1 at γ ∈ {1, 0.1}/bulk.
- *Expected:* NCG 1.8–2.4× fewer gradients than zmshr1ms-RED (toy: 14 vs 25–33 to 1 %, 22 vs 41–50 to 0.1 %);
  estimator gap < 0.3 % at γ = 0.1/bulk.
- *Decision:* NCG wins by ≥ 1.5× → production loop; otherwise keep zmshr1ms-RED and log the measured spectrum.

**P3. Per-set scalars and the prior-aware term, one at a time** inside the P2 winner.
- *Expected:* ≤ 10 % effect at s = 1.6 (R2-like occupancy), ≥ 20 % with 4 respiratory knots (end-knot weights).
- *Decision:* keep whatever is free (n_t, diag G) if not worse; the prior-aware update only if ≥ 10 % fewer
  gradients.

**P4. PDHG with a k-space preconditioner (exact ℓ1 / LLR)** against the P2 winner on the exact objective.
- *Expected:* 2–4× behind NCG (toy: 34 / 82 vs 14 / 22); memory +1 data-size dual (10 GB at 1 mm, host).
- *Decision:* adopt only if it beats NCG on the exact objective by ≥ 1.5× or if γ-smoothing shows an estimator gap
  > 1 % in P2.

**P5. Velocity model: sandwich metric.**
- *Question:* V2 benchmark (velocity_model_sr §V2): B⁻¹-ZMSHR1 vs P_v-ZMSHR1 vs NCG(P_v), gradients to a target
  data residual and the wall / deep velocity slope.
- *Expected:* P_v removes the wall shrink of the unequalised metric (as circ did for the complex model: deep slope
  0.71 → 0.88) and halves the gradients.
- *Decision:* P_v becomes the velocity metric; NCG vs Gauss–Newton–CG decided by gradients to the target.

**P6. Continuation and restarts.** γ stages ×1/4 (3 stages) vs a fixed final γ, in gradients to 1 % of the final
x_γ.
- The toy predicts no gain: γ = 0.1/bulk was as fast as 1/bulk.
- *Decision:* adopt only if ≥ 20 % fewer gradients; otherwise fix γ = 0.1/bulk (in metric units: ρ_RED ≈ 10).

**P7. SVRG coil subsets** only if P4 (PDHG) or a first-order arm wins. With NCG: only as a warm-phase accelerator
(first 5 iterations), tested once.

**P8. 1 mm box R3** with the winners:
- knots (4 linear respiratory × 10 cubic cardiac), wavelet + LLR smoothed, NCG, held-out 1/20;
- budget 40 gradients ≈ 1.7 h at 152 s;
- decision rule as in box_recon.md (beat R1 on frame_fit dynamics and the coronary ROIs).

**Memory ledger** (1 mm box, host 80 GiB cap, setup ≈ 16.6 GiB):

| state | stack | stacks (method) | GiB |
|---|---|---|---|
| per-frame 280 sets | 17.8 | 4 (ZMSHR1 / NCG / PCG) | 71 (does not fit with the setup) |
| knots 2 resp × 10 × 7 = 140 | 8.9 | 4 (NCG) / 7 (L-BFGS 2) | 36 / 62 |
| knots 4 × 10 × 7 = 280 | 17.8 | 4 (NCG) | 71: needs f16 for d and P g_prev, or NVMe tier (SSD writes: avoid) |
| velocity model, 10 nodes (z + v ≈ 20 B per voxel per node) | 1.6 | 4–7 | 6–11 |
| PDHG dual (k-space) | 9.96 | +1 | +10 |
| circulant metric (shared) | 0.34 | 1 | 0.34 |

## 9. Open

- The real restricted spectrum: run Lanczos from the PCG coefficients at dev scale (free) and compare κ with the toy.
- A tight block-van-der-Sluis statement for Kronecker plus prior metrics.
- Anderson + keepfp for the learned prior: rate theory with a non-normal J_D(I − αPH).
- The LLR prior-aware term: does a global temporal subspace Ū_r represent the per-block discarded subspaces well
  enough? Measure ⟨Ū_r, U_r,block⟩ on R-type iterates.

## Results P1–P2, cheap rerun (2026-11-17, `p12_k4g` / `p12_k16g`)

**Setup.** Legacy bench config: lin4 × cubic10, truth breathing coordinate, 8-coil subset, B0 only. Device priors (wavelet db4 L3 + LLR B8, smoothed γ = 0.1/bulk). Encodes 0–1, 30 gradients per arm, arms pcg / ncgI / ncgcap / admm10.

`--keep N` keeps 1/N of the readouts, so k16 has less data per knot than k4. Summary: `summarize.py k4g k16g`.

**Held-out k-space residual, best of each arm (encode 0 / encode 1):**

| arm | k4 | k16 |
|---|---|---|
| ncgI | **0.0571 / 0.0527** | **0.0919 / 0.0790** |
| ncgcap | 0.0592 / 0.0538 | 0.0974 / 0.0852 |
| admm10 | 0.0707 / 0.0675 | 0.0992 / 0.0899 |
| pcg | 0.0862 / 0.0789 | 0.1118 / 0.1032 |

At k16, pcg reaches its best at 6–11 gradients and then overfits.

**Truth velocity phase (encode 1 − 0) at r* 0.493, depth ≥ 5:** slope / corr of the best iterate. A slope of 1 means no velocity shrinkage.

| arm | k4 | k16 |
|---|---|---|
| ncgcap | **0.569 / 0.437** | **0.344 / 0.190** |
| admm10 | 0.373 / 0.262 | 0.227 / 0.085 |
| pcg | 0.249 / 0.083 | 0.151 / 0.040 |
| ncgI | 0.235 / 0.150 | 0.147 / 0.052 |

**Finding: the held-out residual and velocity accuracy rank the arms in opposite order.**
- ncgI wins the held-out residual but barely moves the velocity phase from the warm start (slope 0.15 → 0.24).
- The prior-aware metric (ncgcap) is about 3–4 % worse on held-out but recovers velocity 2.4× faster.
- After 30 gradients no arm is near converged in velocity (slope ≤ 0.57).

So the held-out k-space residual (one number over all samples) is dominated by the magnitude and static content. It is **not** a sufficient stopping or ranking criterion for the velocity-phase difference.

Consequences:
1. Rank solvers on a velocity-sensitive data metric. One candidate is the held-out residual of the encode-difference data, which is real-data computable.
2. The prior-aware metric's value is in the velocity directions, so it should not be dropped.
3. These results strengthen the case for solving on magnitude / velocity / phase directly (the end-goal parametrisation), where the slow direction is an explicit unknown.

Earlier statements in this document and in chat that "ncgI is best" hold only for the held-out residual.
