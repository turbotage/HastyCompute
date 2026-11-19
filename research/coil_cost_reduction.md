# Making the coil dimension cheap without losing the fixed point (2026-11-18)

Question: the data term uses ≈ 30 virtual coils and the operator cost is ≈ linear in coils (temporal_operator.md §22.3).
Can coil sketching, stochastic coil subsets (keep + draw, the user's "pyramid") or variance-reduced estimators cut that
cost while keeping (a) the full-coil fixed point and (b) the curvature methods (NCG, ZMSHR1, PCG) we rely on?

Scripts: `research/experiments/coil_sketch/` (`spectrum.py` → coil spectra of the phantom maps; `inflation.py` →
pointwise step-size inflation and variance factors; `toy.py` → the solver toy; `box_toy.py` → box compression;
`floor.py` → stochastic-PnP neighbourhood; `summarize.py` → cost-to-threshold table). Outputs in `out/` (small JSON
plus cached run arrays in `out/runs/`, ≈ 0.5 GB, deletable: they only make reruns resumable).

## 0. Answer

Question and decision rule (fixed before the solver toy ran): at an equal coil-application budget, does any of {coil sketching, keep + draw, SVRG, the
pyramid} reach the full-coil fixed point measurably faster than full-coil NCG with the circulant metric, without moving
that fixed point? Decision rule: implement only what gains ≥ 1.5× at 2–3 % of the fixed point.

**Verdict:**
1. **The big lever is the coil *ordering*, not stochastics.**
   - The whole-torso spectrum of this array is flat: 31 of 32 coils at 99.9 % by map energy, even 28 at 99 % (§2).
   - Inside the heart box, 18 coils hold 99.9 % of the energy (heart bbox; 22 for the heart + 15 mm box), and 11 hold
     99 %.
   - Compressing with the **box Gram** (a deterministic `--coil-gram box`) cuts every operator call by 30 → 18–22
     (1.4–1.7×). Toy, whole FOV reconstructed with 18 heart-ordered coils:
     - heart error vs truth 0.1302 vs 0.1287 with 30 coils (+1.2 %);
     - velocity vs truth 4.56 vs 4.42 cm/s;
     - distance to the 30-coil fixed point inside the heart 2.3 % (1.8 % with 22 coils).
   - The same cost with the 18 *globally* strongest coils is 3× worse (6.8 %, velocity 5.07 cm/s).
   - "Keep the 8–12 strongest" must use this ordering. The 12 globally strongest coils hold only 56 % of the heart's
     energy.
2. **Out-of-box signal cannot be dropped.** With 18 box-ordered coils the tissue outside the heart still carries 16–20×
   the heart's energy in the kept coils. ROVir (SIR-optimal beamforming) only halves that, and only at the price of
   in-box energy (e.g. 95 % kept at 16 coils). The full-FOV operator stays; it just runs on fewer coils.
3. **Coil sketching (Oscanoa et al., verified) does not pay with our solvers.**
   - As published (exact anchor gradient, sketched Hessian, inner solve, no outer safeguard) it **diverges** in every
     configuration we tried (keep-only, Rademacher, weighted subsets), with NCG inner solves on our un-DCW'd problem.
   - Made safe (outer exact line search on F, or our lagged-gradient τ rule with uninterrupted NCG), it is at parity
     to 1.2× with full NCG at 2–3 % of the fixed point. It is 1.3–2.4× *slower* early (at 10 %).
   - The restarts every epoch throw away the Krylov memory that NCG builds over ≈ 60 gradients. A cheaper inner
     operator does not buy that back when the conditioning is dominated by k-space density and the prior, not by the
     coils.
   - Below the 1.5× bar. Do not implement it now.
4. **Keep + draw (the existing CoilSampling, the user's pyramid) is a first-order tool.**
   - In the PnP form at a common α (needed so that all arms share the fixed point), keep 12 + draw 3 is 2.0× and
     keep 8 + draw 4 is 2.6× cheaper than full gradients to the same distance.
   - But the full-coil run could use a 1.8–2.1× larger α (measured λ_max inflation of those subsets). At its own α the
     sampled run is only ≈ 1.1–1.2× ahead (cost × inflation ≈ 0.9) [model].
   - The run then sits in a stochastic neighbourhood. Started at x*, keep 12 + draw 3 wanders to 1.2 % RMS (keep 12 +
     6: 0.6 %). That is ≈ 9 % of the estimation error of the PnP fixed point.
   - The annealing pyramid (keep 12, draw 5 → 10 → all) is 1.1–1.7× at equal α: it pays the full price exactly where
     first-order methods are still far from x*.
   - All of this is ≥ 5× behind full-coil NCG on the same data (§6), and ZMSHR1/NCG lose their secant pairs at every
     subset change.
5. **SVRG keeps the fixed point exactly.** Started at x*, it stays at 2.5e-5, the same as full gradients. At equal α it
   is 1.7× cheaper than full in the PnP form. It has the same α-inflation penalty and is first-order only. If coil
   subsets are used at all, this is the right estimator: it is a few lines on top of the existing lagged-gradient
   anchor.
6. **Recommended (§7):**
   - (a) `--coil-gram box` compression, with the keep-ordering from the box Gram.
   - (b) If coil sampling is used, use the SVRG form through the lagged anchor (`--lagged` + `--coil-sampling`),
     first-order phases only, with a norm-test annealing and a `full_last` sized by the contraction.
   - (c) No coil sketch for NCG/ZMSHR1/PCG.


---------------------------------------------------------------------------------------------------------------------

## 1. Setup and notation

- f(x) = Σ_c f_c(x), f_c = ½‖A_c x − y_c‖², A_c = F S_c: F the coil-free trajectory operator (segments, ratemap,
  knots), S_c the pointwise (whitened, compressed) map. N = Σ_c N_c, N_c = S_cᴴ Fᴴ F S_c.
- Cost: one gradient = Σ_c (forward + adjoint of coil c) = n_c coil applications. At 1 mm the coil batch is 1 and
  the sample side is per coil (§22.3), so the cost is linear in n_c. The circulant / knot metric (G ⊗ C, sandwiched by
  D^{1/2} = (Σ_c |s_c|²)^{1/2}) is coil-free and costs ≈ 1/400 of a gradient (convergence_strategy.md §1).
- A coil subset S with weights w_c (or a sketch: virtual maps ŝ_j = Σ_c Φ_jc s_c) gives Ñ_S = Σ_{c∈S} w_c N_c at
  cost |S|. Because the maps are pointwise and F is coil-free, a sketch is just another set of maps: Ñ_Φ =
  Σ_j ŝ_jᴴ Fᴴ F ŝ_j (Oscanoa et al. eq. 12). Everything below therefore plugs into the existing operator through a
  "which maps" switch (`SetOp::select_coils` today, a mixed-map variant for sketches).

## 2. The coil spectrum of the phantom (and of real cardiac arrays)

`spectrum.py`, true maps of `factory_arms/ph0001/sim_flow_10min` (32 coils, whitened with the sim's Ψ), body mask
ρ > 5 % max. "map" = Σ_r s sᴴ over the body; "signal" = Σ_r ρ² s sᴴ; "calib" = the Gram of the central 24³ k-space
cube of ρ s_c (what `--coil-gram calib` measures, noise-free); boxes from the box scan's prescription
(`sim_flow_10min_box1mm_b`): **heart** = heart + great-vessel bounding box, **box** = heart + 15 mm (the keep-box),
**fov** = box + 15 mm margin.

Virtual coils needed:

| Gram | 99 % | 99.5 % | 99.9 % | 99.95 % | 99.99 % | leading σ² fractions |
|---|---|---|---|---|---|---|
| whole FOV, map | 28 | 29 | 31 | 32 | 32 | 0.111 0.104 0.083 0.075 0.071 0.061 … |
| whole FOV, signal-weighted | 28 | 29 | 31 | 32 | 32 | 0.117 0.093 0.086 0.083 0.057 0.056 … |
| whole FOV, calib (noise-free) | 28 | 29 | 31 | 32 | 32 | 0.118 0.095 0.087 0.082 0.058 … |
| fov box | 21 | 23 | 27 | 28 | 31 | 0.256 0.159 0.122 0.116 0.070 … |
| box (heart + 15 mm) | 15 | 17 | 22 | 24 | 28 | 0.390 0.174 0.144 0.109 0.054 … |
| heart bbox | **11** | 13 | **18** | 20 | 24 | 0.495 0.211 0.115 0.074 0.042 … |

- The quoted splinerecon spectrum (0.32, 0.14, 0.086, …; `vknot/dev_k8/A_spline.log`) is the Gram of 200 k raw
  *k-space samples* (splinerecon's own compression, not the calibration cube). It is k-centre-weighted, hence more
  concentrated at the top; its flat tail (2.6e-4 per coil) is the noise floor, which is why 99.9 % of that Gram also
  needs ≈ 30 coils. The density-neutral Grams above are flat from the start: the factory-arms FOV (508 × 287 mm,
  arms included) is seen by every element of the array, so nothing compresses globally.
- **The global ordering is the wrong ordering for the heart.** The first n global virtual coils hold this fraction of
  the heart's signal energy: n = 8: 0.52, 12: 0.56, 16: 0.80, 20: 0.87, 24: 0.97, 30: 0.993. "Keep the 8–12
  strongest" with the global ordering keeps about half of the heart.
- Local loss (1 − ‖P_n s(r)‖²/‖s(r)‖², heart voxels) of the heart-box ordering: n = 11: median 3 %, p99 35 %;
  n = 18: median 0.3 %, p99 10 %, max 18 %. The 99.9 % box-energy criterion still leaves 10 % SNR² loss in the
  worst 1 % of heart voxels; n = 22 brings p99 to 4 %.

Literature on 32-channel cardiac arrays (all 2-D or ROI-based, all at looser criteria than 99.9 %):
- Buehrer et al., MRM 57:1131 (2007), "Array compression for MRI with large coil arrays": 2-D cine with a 32-element
  heart array, ROI (heart) SNR-optimal compression to **4 virtual coils with 0.3 % SNR loss** in the heart ROI (2 %
  at R = 2); 1 virtual coil: 13 %. The loss depends on the ROI size.
- Huang et al., MRI 26:133 (2008): PCA compression of 32-channel cardiac data, "dramatically reduced" time, no
  single count given.
- Zhang et al., MRM 69:571 (2013) (geometric coil compression, 3-D Cartesian): per-slice aligned compression to
  6–8 virtual coils of 32; a single 3-D-wide compression needs clearly more, because the maps vary in 3-D.
- Oscanoa et al. (below) compress 35–40-channel 3-D abdominal cones data to C = 20 virtual coils at 98 % energy.
- Practice: 32 → 8 (2-D), 32 → 16 default with ≤ 3 % SNR loss (vendor), 32 → 5 for CS perfusion (Compression²).

So the literature's "few coils" are 2-D slices or ROI criteria at 95–99 %. Our 3-D whole-torso volume at 99.9 % needs
31; the heart box at 99 % needs 11, at 99.9 % 18, in line with an ROI criterion on a 3-D region. Real thorax arrays
(anterior + posterior 16 + 16, no arms in the coil field) will compress somewhat better than this phantom globally,
but the heart is deep, so the box numbers are the relevant ones.

## 3. Literature check: coil sketching

Oscanoa JA, Ong F, Iyer SS, Li Z, Sandino CM, Ozturkler B, Ennis DB, Pilanci M, Vasanawala SS. "Coil sketching for
computationally efficient MR iterative reconstruction." Magn Reson Med 2024;91(2):784–802 (online 17 Oct 2023;
arXiv:2305.06482; ISMRM 2021 summa cum laude). Citation verified. Method (Algorithm 1, eqs. 8–13):

- It is the **Iterative Hessian Sketch** (Pilanci & Wainwright, JMLR 2016) on the coil dimension:
  x^{t+1} = argmin_x ½‖A_Sᵗ(x − xᵗ)‖² + ⟨x, Aᴴ(Axᵗ − y)⟩ + g(x), A_Sᵗ = Sᵗ A.
  - The gradient Aᴴ(Axᵗ − y) is exact (all coils), once per outer iteration.
  - Only the Hessian AᴴA is sketched.
- Structured sketch Sᵗ = S̃ᵗ ⊗ I_N, S̃ᵗ = blockdiag(I_V, S̃_S): the V highest-energy (PCA) virtual coils are kept,
  the C − V others are mixed into S random Rademacher (±1) or Gaussian combinations. It is redrawn every outer
  iteration. Rademacher had lower variance than Gaussian.
- Recommended V = Ĉ − 1, S = 1 (e.g. Ĉ = 4 of C = 20).
- **V = Ĉ, S = 0 (keep-only, plain aggressive compression of the Hessian) diverged.**
- Inner solver: the same as the baseline (FISTA for ℓ1-wavelets, PDHG for ℓ1-TV), with a fixed step from the first
  subproblem.
- Results, 3-D cones abdomen, C = 20, Ĉ = 4, with **density compensation W^{1/2}** in the data term:
  - ℓ1-wavelets (FISTA): 193 s → 114 s, i.e. **1.7×**; equal NRMSE / SSIM.
  - ℓ1-TV (PDHG with CG inner): 3499 s → 1249 s, i.e. **2.8×**. The gain here comes from PDHG's inner CG, which
    applies the (sketched) operator many times.
- Aggressive compression to Ĉ = 4 coils was fastest but shaded. AccProxSGD (coil-batch SGD) blurred, i.e. was under-converged.

What does not carry over:
- The DCW (we precondition instead: our metric P plays its role).
- The inner first-order solvers: we use NCG / ZMSHR1 / PCG.

What carries over:
- The structure: keep the strongest coils plus a few random combinations of the rest.
- The exact anchor gradient.
- Above all, the fact that keep-only diverges.

## 4. Theory

### 4.1 Coil sketching = our lagged gradient with a coil surrogate

`P2np::gradient_lagged` already implements ∇̃f(x) = ∇f(x_a) + Ñ (x − x_a) with an anchor x_a refreshed when
‖x − x_a‖ > τ‖x‖, and Ñ = the segment surrogate (implementation_results.md §5: 4.5–6.7× cheaper at equal error).
IHS is exactly this with Ñ = Ñ_Φ (a coil sketch) held for the anchor epoch. Consequences:

1. **Fixed point preserved for any sketch.** The model m_a(x) = f(x_a) + ⟨∇f(x_a), x − x_a⟩ + ½‖x − x_a‖²_Ñ
   satisfies ∇m_a(x_a) = ∇f(x_a). If x_a = x* (the full-coil fixed point of the P²nP / RED / smoothed-prior
   iteration), the inner iteration starts at a fixed point and stays there. Conversely, an inner fixed point that is
   also the anchor is a full fixed point. Unbiasedness of Ñ is *not* needed for this; it is needed only for the rate.
2. **Exact secant pairs.** Inside an epoch the model is fixed: m(x₂) − m(x₁) gradients differ by Ñ s + ΔR, so
   ZMSHR1 / NCG / L-BFGS curvature is exact for the model. This is what keep+draw with hold = 1 destroys
   (recon_realism.md §4: "nothing with ZMSHR1"). `curvature_pair_ok` already encodes "same anchor epoch".
3. **Rate.** For a quadratic (or locally quadratic) F with Hessian K = N + H_R and model Hessian K̃ = Ñ + H_R, the
   exact inner solve gives e⁺ = (I − K̃⁻¹K) e. With the generalized eigenvalues μ of (K, K̃):
   - contraction max |1 − μ|;
   - convergence iff K̃ ≻ K/2.
   - Keep-only (Ñ = N_keep ⪯ N) has μ ≥ 1 and diverges as soon as N_tail ⪰ N_keep + H_R in some direction. That
     direction exists wherever the dropped coils dominate locally: half of the heart with the global ordering (§2).
     This is Oscanoa's divergence for S = 0.
   - An unbiased sketch (E Ñ = N) concentrates μ around 1.
   - With K inner Krylov steps instead of an exact solve, e⁺ = (I − q_K(PK̃) P K) e. For small K this is K steps
     of the true problem plus an O(‖K − K̃‖) perturbation.
   - Early inner stopping is what makes the sketch safe in directions where μ ≫ 1: the aliasing directions that only
     the dropped coils unfold, where Ñ underestimates N.
4. **Speed-up bound.** An epoch costs n_c + K|S| coil applications. If an inner step makes the progress of a full NCG
   step, the speed-up is s(K) = K n_c / (n_c + K|S|), and s(K) ≤ n_c / |S|. For |S|/n_c = 0.43 (keep 12 + 1
   sketch of 30): s(5) = 1.6, s(10) = 1.9, s(∞) = 2.3. A combined segment + coil surrogate multiplies the ratios
   (cost ∝ |S| L_s / (n_c L)).
5. **No data access inside an epoch.** The inner gradient μ + Ñ(x − x_a) needs no k-space data. kstore streaming
   (disk / compressed tiers) is needed only at anchors.
6. **Metric.** The circulant / knot metric P = D^{-1/2}(G ⊗ C)⁻¹D^{-1/2} with D = Σ_c|s_c|² stays valid: E Ñ = N.
   A sketch-matched D̃ = Σ_{keep}|s_c|² + Σ_j |ŝ_j|² is free (pointwise) if wanted.

### 4.2 Stochastic keep + draw (the existing CoilSampling) and the user's pyramid

Estimator: ∇̃f = Σ_{c<k} ∇f_c + p Σ_{c∈G} ∇f_c, with G one of the p = ⌈(n_c − k)/d⌉ reshuffled groups.

- **Step-size inflation.** α must be stable for every subset: α ≤ 1/max_S λ_max(P Ñ_S). Under the pointwise
  approximation N ≈ D^{1/2} C D^{1/2}:
  λ_max(P Ñ_S)/λ_max(P N) ≈ η_S = max_r (D_keep(r) + p D_G(r))/D(r).
  Every element of a surface array dominates *somewhere* (its own skin patch), so with any ordering η_S ≈
  1 + (p − 1) × (largest local share of a tail coil).
  - `inflation.py` on the slice, global ordering, max over the body: keep 12 + draw 3: **2.55**; keep 12 + 6: 1.78;
    keep 8 + 4: 2.97; keep 0 + 6: 4.35.
  - Heart ordering, max over the body: keep 8 + 4 of 18: 2.10. Inside the heart it is only 1.34, but the
    out-of-box tissue is part of the reconstructed FOV, so the body maximum counts.
- **Cost per unit of progress (first-order).** The step shrinks by η_S, the cost by (k + d)/n_c, so a sampled
  iteration buys (k + d)/(n_c η_S) of a full one per coil application. That is 0.5 × 2.55 ≈ 1.3 > 1 for keep 12 +
  3: **slower than full gradients before any variance is counted.** It pays only if η_S × cost < 1, which needs
  locally weak tails everywhere in the FOV. A whole-torso array does not have that.
- **Variance and the stochastic-PnP neighbourhood.** At the full fixed point x*, the per-coil gradients
  g_c* = A_cᴴ r_c* contain the backprojected noise, E‖A_cᴴ n_c‖² = σ² M ‖s_c‖². They do not vanish, so
  Var(∇̃f(x*)) = p Σ_G ‖g_G*‖² − ‖g_tail*‖² ≈ (p − 1) σ² M E_tail
  (E_tail = Σ_{c≥k} ‖s_c‖², independent noise across white virtual coils).
  - For a map T(x) = D(x − αP∇f(x)) that contracts with factor ρ in the P⁻¹ norm and a nonexpansive denoiser D:
    limsup E‖x_k − x*‖_{P⁻¹} ≤ α √Var_P / (1 − ρ). The run lives in a neighbourhood whose radius scales with
    α √((p − 1) E_tail).
  - Relative to the noise in the full gradient, (p − 1)E_tail/E = 1.6 for keep 12 + 3 (global ordering, body), so the
    sampled gradient carries 1.6× the noise variance of the full one. Heart ordering, heart voxels: 0.1.
  - `full_last` ends at the exact fixed point only if the last full phase is long enough to contract the radius:
    ≈ log(radius/tol)/log(1/ρ) full iterations.
  - In PnP the fixed point depends on α. The full phase must keep the *sampled* α (`recon_p2np` uses one α for the
    whole run, which is correct).
- **Annealing = dynamic sample size.** Increasing the batch geometrically keeps linear convergence (Friedlander &
  Schmidt, SIAM J. Sci. Comput. 34:A1380, 2012; Byrd, Chin, Nocedal & Wu, Math. Program. 134:127, 2012). The
  practical rule is the **norm test**: enlarge d when
  V̂ = Σ_i ‖ĝ_i − ḡ‖²_P / (p − 1) > θ² ‖ḡ‖²_P (θ ≈ 0.5–1).
  - The p group estimates ĝ_i of one reshuffle epoch and their mean ḡ give V̂ for one accumulator stack plus
    scalars.
  - The test is passed early (large gradients) and fails near the fixed point, which is the user's pyramid ordering
    "cheap early, all coils late".
  - It is the *opposite* of the inexact-Krylov rule (convergence_strategy.md §3.7: accurate operators early, cheap
    late). The two agree once the cheap part is a control variate (lagged / SVRG / IHS), where the gradient error
    ‖(Ñ − N)(x − x_a)‖ shrinks with the step automatically.

### 4.3 Variance reduction: SVRG / SAGA

- **SVRG** (Johnson & Zhang, NeurIPS 2013): v = μ + Ñ_S(x − x_a), S redrawn every inner step, μ = ∇f(x_a).
  - It is unbiased, and Var(v) = E‖(Ñ_S − N)(x − x_a)‖² → 0, so the fixed point is exact.
  - The epoch costs n_c + m|S|.
  - But Ñ_S changes every step, so there is no conjugacy and no exact secant pair across steps: first-order only. The
    α penalty of §4.2 (η_S) is unchanged.
  - With S held over the epoch it *is* IHS (§4.1) with gradient-descent inner steps.
- **SAGA** (Defazio et al. 2014) keeps a stored gradient per coil (or group), 30 image stacks per set. That is out of
  the question at 1 mm (host RAM already holds 4–6 stacks).
- **Quasi-Newton.** Stochastic L-BFGS variants (Byrd, Hansen, Nocedal & Singer, SIAM J. Optim. 26:1008, 2016;
  Moritz, Nishihara & Jordan, AISTATS 2016) build curvature pairs from sub-sampled Hessian-vector products on a
  *separate, held* sample. For our quadratic data term that is again the IHS model.
  - So every route to "stochastic coils + exact curvature" leads to the same object: an exact anchor gradient plus a
    held, cheap curvature model.

## 5. Box-restricted coils (Q4)

Box energy from `spectrum.py` (ρ²-weighted map Grams, heart bbox in = 3.1 % of the body's signal energy, box in =
8.8 %):

| n coils | box-PCA: in-box kept | out-of-box energy carried / in-box | ROVir: in-box kept | ROVir: out / in |
|---|---|---|---|---|
| heart 8 | 0.979 | 8.5 | 0.728 | 4.3 |
| heart 12 | 0.994 | 12.4 | 0.876 | 6.6 |
| heart 16 | 0.998 | 16.0 | 0.947 | 9.8 |
| heart 20 | 0.9996 | 20.5 | 0.983 | 12.8 |
| box 16 | 0.994 | 5.4 | 0.933 | 3.2 |
| box 20 | 0.998 | 6.7 | 0.973 | 4.1 |
| box 24 | 0.9996 | 8.5 | 0.992 | 6.1 |

- **Box-PCA = compression with the Gram restricted to the box.** It keeps 99.9 % of the box energy with 18 (heart) /
  22 (box) coils.
  - The discarded coils take their in-box *and* out-of-box data with them. That is harmless: dropping data channels
    never biases the estimator, it only costs the dropped 0.1 % of box SNR².
  - The kept coils still see 16–20× (heart) or 5–7× (box) more signal outside the box than inside. A non-Cartesian
    undersampled acquisition aliases all of it into the box, so the out-of-box tissue must stay in the model
    (full-FOV operator, or an explicit out-of-box estimate subtracted from the data).
- **ROVir** (Kim et al. 2021) maximises in/out energy, but the heart is deep and every element sees the chest wall
  more than the heart. At equal coil count it halves the out-of-box energy and loses 5–25 % of the in-box energy.
  Useless here. It would matter for a superficial region or for discarding a very bright, far region (arms).
- **Local loss**, heart bbox ordering inside the heart: n = 18 median 0.3 %, p99 10 %, max 18 %; n = 22 p99 4 %. The
  99.9 % energy criterion is an average, so 22 is the safer count for coronary SNR.

Toy (`box_toy.py`): same slice, same raw 32-coil data and noise, compressed differently; whole FOV reconstructed; the
same λ; the reference is the 30-coil global fixed point. "ctrl" is that 30-coil problem solved like the others (500 NCG
iterations), i.e. the solver-error floor.

| compression | cost | heart: dist. to 30-coil x* | heart: vs truth | body: vs truth | velocity vs x*₃₀ [cm/s] | velocity vs truth [cm/s] |
|---|---|---|---|---|---|---|
| global 30 (reference) | 1.00 | 0 | 0.1287 | 0.0833 | 0 | 4.42 |
| ctrl (global 30, 500 its) | 1.00 | 2e-5 | 0.1287 | 0.0833 | 0.001 | 4.42 |
| global 18 (strongest globally) | 0.60 | 0.068 | 0.1482 | 0.0944 | 4.10 | 5.07 |
| heart 11 | 0.37 | 0.046 | 0.1354 | 0.1091 | 2.48 | 4.41 |
| **heart 18** | 0.60 | 0.023 | 0.1302 | 0.0912 | 1.31 | 4.56 |
| **heart 22** | 0.73 | 0.018 | 0.1296 | 0.0886 | 0.91 | 4.44 |

Velocity = signal-weighted RMS of the encode-difference phase over the heart lumen, in cm/s at venc 1.5 m/s.
- The heart-ordered compression moves the in-heart estimate by 2 % of itself and its truth error by ≤ 1.2 %. The body
  outside the box gets worse (0.083 → 0.091), as it should: it lost its own coils.
- The velocity changes vs truth (−0.01 … +0.14 cm/s) are within the single-noise-realisation spread. The same cost
  spent on the globally strongest coils costs 15 % heart error and 0.65 cm/s.
- This is a *different estimator* (fewer channels), not the same fixed point. Its fixed point differs from the
  30-coil one by less than the 30-coil one differs from the truth by a factor ≈ 6 in the heart.


## 6. Toy

`toy.py` (cached per arm and seed in `out/runs/`, resumable; `summarize.py` → `out/summary.json`).

**Problem.**
- An axial 2 mm slice of ph0001 through the heart (256 × 144 after padding).
- Data are simulated from the 32 raw whitened true maps with complex white noise (σ = 3 % of the RMS sample) and
  compressed with the global signal-weighted Gram to the production 30 virtual coils.
- Two encodes; encode 1 carries π v_z / venc of frame 14 (through-plane).
- 48 golden-angle full radial spokes per encode (different spokes per encode), i.e. ≈ 8× below radial Nyquist.
- Exact NDFT model with exact Toeplitz normal operator (checked to 1e-15).
- Support = body dilated by 3 voxels: without it the circulant metric excites the unsampled k-space corners outside
  the body (relative error 20).
- Smooth objective: data + λ Σ_detail env_γ(|Haar 4-level coefficient|), with γ = 0.1/bulk and λ = 124.8 from a truth
  sweep. The truth error of x* is 0.093.
- Metric: the circulant (T. Chan) metric sandwiched by D^{-1/2}, plus a prior floor κ = median symbol.
- PnP form: x ← soft(x − α P ∇f, τ), τ from a truth sweep. Its fixed point is a 6000-iteration run (change per 300
  iterations 6e-5).

**Cost.** One full gradient = 30 coil applications. The prior / metric costs are not charged. At 1 mm a wavelet / LLR
sweep costs tens of % of a gradient (convergence_strategy.md §3.2), which makes every scheme with more, cheaper
iterations *less* attractive than shown.

**Step-size inflation, measured** (data-only λ_max(P Ñ_S)/λ_max(P N), 8 draws, global ordering):

| keep k + draw d | cost | λ_max ratio | pointwise bound (`inflation.py`, body max) | cost × ratio |
|---|---|---|---|---|
| 0 + 6 | 0.20 | 3.76 | 4.35 | 0.75 |
| 4 + 4 | 0.27 | 3.90 | 6.22 | 1.04 |
| 8 + 4 | 0.40 | 2.10 | 2.97 | 0.84 |
| 12 + 3 | 0.50 | 1.80 | 2.55 | 0.90 |
| 12 + 5 | 0.57 | 1.38 | — | 0.78 |
| 12 + 6 | 0.60 | 1.35 | 1.78 | 0.81 |
| 16 + 4 | 0.67 | 1.27 | 2.10 | 0.85 |
| 20 + 5 | 0.83 | 1.01 | — | 0.84 |

- First-order progress per coil application relative to full gradients at *their* α is ≈ 1/(cost × ratio): 1.0–1.33×
  at best (keep 0 + 6, keep 12 + 5).
- With the smooth (Moreau) objective the step is limited by the prior curvature 1/γ for every subset (ratio 0.99), so
  first-order smooth arms are not informative and were dropped. The first-order comparison uses the PnP form below.

**Smooth objective: curvature methods.** Cost (full-gradient equivalents) to reach a relative distance to x*, mean of
2 seeds, both encodes (the pair counts when both reach it):

| arm | to 10 % | 5 % | 3 % | 2 % | dist. @60 | velocity vs x* @20 / 40 / 60 [cm/s] |
|---|---|---|---|---|---|---|
| **NCG, all coils** | **14.0** | **26.0** | 41.0 | ≈ 60 | 0.0197 | 3.90 / 1.45 / 0.58 |
| PGD, all coils | ∞ | ∞ | ∞ | ∞ | 0.89 | — |
| IHS (Oscanoa Alg. 1), Rademacher keep 3 + 1 | diverges | | | | 4e69 | |
| IHS, Rademacher keep 12 + 2, K = 3 / 5 / 10 | diverges | | | | 3e2 – 4e4 | |
| IHS, weighted subset keep 12 + 3 | diverges | | | | 2e3 | |
| IHS, keep-only 12 | diverges | | | | 1e28 | |
| IHS + outer line search, Rad. 12 + 2, K 5 | 21.7 | 33.3 | 58.3 | — | 0.0285 | 8.97 / 2.05 / 1.24 |
| sketch-preconditioned flexible NCG, Rad. 12 + 2, K 3 / 5 / 10 | 21.6 / 20.0 / 34.0 | 32.4 / 31.7 / 39.7 | 48.0 / 45.0 / 51.0 | — | 0.021 / 0.019 / 0.024 | 7.71 / 1.70 / 0.50 (K 5) |
| same, Rad. 3 + 1 / 8 + 2 / 16 + 2 / 20 + 2 (K 5) | 28.3 / 20.0 / 18.0 / 18.7 | 37.5 / 28.0 / 28.0 / 32.7 | 50.0 / **38.7** / 46.0 / 49.0 | — / **52.0** / — / — | 0.020 / **0.0155** / 0.021 / 0.024 | 8.04 / 1.27 / 0.42 (8 + 2) |
| same, weighted subset 12 + 3 | 19.2 | 29.8 | 45.5 | — | 0.0207 | 7.67 / 1.91 / 0.58 |
| same, keep k + circulant tail (k = 8 / 12 / 16) | 28.0 / 30.0 / 14.7 | 37.3 / 39.0 / 25.7 | 46.7 / 51.0 / **36.7** | 58.3 / — / **55.0** | 0.016 / 0.020 / 0.016 | 4.54 / 1.25 / 0.44 (k 16) |
| lagged NCG (uninterrupted, τ refresh), Rad. 12 + 2, τ 0.01 / 0.03 | 21.3 / 21.3 | 35.1 / 28.9 | 46.1 / 37.8 | 56.9 / 49.8 | 0.017 / 0.017 | 7.57 / 1.11 / 1.43 (τ 0.03) |
| lagged NCG, Rad. 8 + 2, τ 0.03 | 19.3 | 26.3 | **34.8** | **51.8** | 0.0186 | 6.23 / 2.21 / 1.03 |
| lagged NCG, keep 8 + circulant tail, τ 0.03 | 19.0 | 26.0 | 40.3 | — | 0.0218 | 5.87 / 1.75 / 1.22 |

Readings:
- **Plain IHS diverges**, as §4.1 (3) predicts for exact-ish inner solves: the sketch underestimates N by more than 2×
  in directions only the tail coils resolve.
  - Oscanoa's published configuration (keep 3, one Rademacher combination) diverges fastest. Their DCW'd problem and
    fixed-step FISTA inner solves (implicitly early-stopped) are not our setting.
- **Safeguarded versions converge to the exact x*.**
  - Outer line search with the next anchor gradient for free, flexible NCG, or the τ-refreshed lagged NCG.
  - The best gain is 1.06–1.18× at 3 % and ≈ 1.15–1.2× at 2 % (NCG reaches 2 % at ≈ 60). They are 0.4–0.8× (slower) at
    10 %.
  - The bound s(K) = K n_c/(n_c + K|S|) (§4.1) predicts 1.5–1.9×. The missing factor is the Krylov memory lost at
    every epoch restart.
- **Velocity follows the image distance.** No arm converges the velocity phase faster relative to its image error
  (all ≈ 0.4–1.4 cm/s at 60).

**PnP form (first-order, P²nP-like) at a common α** (the smallest subset-stable α, so that every arm has the same
fixed point; budget 120). Cost to reach a relative distance:

| arm | to 50 % | 30 % | 20 % | 10 % | dist. @120 | velocity vs x* @120 [cm/s] |
|---|---|---|---|---|---|---|
| full | 52.0 | 92.0 | — | — | 0.222 | 21.0 |
| keep 12 + draw 3 | 25.8 | 46.0 | 67.0 | 114 | 0.090 | 12.6 |
| keep 8 + draw 4 | 20.1 | 35.6 | 52.1 | 88.9 | 0.064 | 9.4 |
| keep 12 + draw 3, last 30 full | 25.8 | 46.0 | 67.0 | — | 0.109 | 14.0 |
| SVRG keep 12 + 3, m = 10 | 30.8 | 55.0 | 80.2 | — | 0.117 | 14.5 |
| pyramid keep 12: draw 5 → 10 (@20) → all (@40) | 30.5 | 66.2 | 109 | — | 0.178 | 18.3 |
| pyramid, fast (→ 10 @5, → all @10) | 45.0 | 85.5 | — | — | 0.210 | 20.3 |

Neighbourhood at the fixed point (`floor.py`: start at x*_PnP, 150 iterations, RMS of the relative distance over the
second half):

| estimator | keep 12 + 3 | keep 8 + 4 | keep 12 + 6 | SVRG 12 + 3 | full (control) |
|---|---|---|---|---|---|
| RMS distance from x* | 1.15e-2 | 1.22e-2 | 5.8e-3 | 2.5e-5 | 2.5e-5 |

Readings:
- Far from x* every cheap iteration counts. The sampled arms are cheaper per iteration by exactly their cost ratio
  because α is common, and nothing else matters there.
  - At its own α the full run would gain the inflation factor back (1.8× for 12 + 3, 2.1× for 8 + 4) [model].
  - So the real first-order gain is ≈ 1.1–1.2×.
- The annealing pyramid spends full-coil cost while still far from x*, which first-order methods are for most of a
  run (the PnP iteration is at 22 % after 120 gradients). In the PnP toy it is the worst sampled arm.
- The SGD neighbourhood (1.2 % RMS) is real but small compared to the estimator's own error (truth error ≈ 13 %).
  SVRG removes it completely (equal to the full-gradient control).
- The first-order PnP iteration is itself far behind NCG on the smooth objective (22 % after 120 gradients vs 2 %
  after 60). The coil question is second order next to the solver question (convergence_strategy.md §0).

**Caveats.**
- 2-D slice and one frame.
- No segments / ratemap, Haar instead of db4 / LLR / learned prior.
- A whole-torso FOV with arms (the flattest possible spectrum).
- Global coil ordering in the stochastic arms. A box ordering helps keep + draw only if the reconstructed FOV is the
  box, which it is not (out-of-box tissue is modelled).


## 7. Recommended implementation

Ordered by value; all within the existing hooks.

1. **`--coil-gram box[:MARGIN_MM]` (do this).**
   - Real-data recipe (no truth): the calibration-cube coil images (the same low-res images `--coil-gram calib` uses)
     restricted to the prescription box + margin (the box is part of the prescription, not a phantom label)
     → Gram → eigenvectors.
   - Keep the fewest coils with ≥ e of the *box* energy. Default e = 0.999; report the p99 local loss and allow
     `e=0.9995` (≈ 22).
   - The virtual coils are then energy-ordered *for the box*. This is what `CoilSampling.keep` must assume.
   - The full-FOV operator and data stay; only `CoilCompression.rows` changes (`coils.rs::CoilGram::add_with` over the
     masked low-res images).
   - Expected: 30 → 18–22 virtual coils, 1.4–1.7× on every operator call and on coil-dependent memory. Toy: ≤ 1.2 %
     heart truth-error change.
   - Pre-register on the box scan: held-out encode-difference residual and lumen velocity slope vs `calib`.
2. **SVRG form of coil sampling (only for first-order phases).**
   - In `P2np::gradient`, allow `cur.is_some()` together with `lagged`: the surrogate of `gradient_lagged` becomes
     the sampled coil groups. Compute Ñ_S(x − x_a) through `select_coils(groups)` with the group weights (as in
     `gradient_sampled`), times the segment surrogate if present.
   - The anchor refresh stays exact (all coils). `curvature_pair_ok` = same anchor epoch *and* same group set:
     `hold` ≥ the epoch length gives IHS, `hold` = 1 gives SVRG.
   - Two host stacks (x_a, ∇f(x_a)) already exist for `--lagged`.
   - The fixed point is then exact (toy: 2.5e-5, equal to full gradients) and `full_last` becomes unnecessary.
3. **Annealing / stopping (if 2 is used without anchors, or for `keep:draw` as today).**
   - Norm test per reshuffle epoch: V̂ = (Σ_i ‖ĝ_i‖²_P − p‖ḡ‖²_P)/(p − 1) from the p group estimates of an epoch.
     It needs one accumulator stack plus scalars.
   - When V̂ > θ²‖ḡ‖²_P (θ = 0.5), double `draw`.
   - Size `full_last` as ⌈log(r/tol)/log(1/ρ)⌉, with r from the measured neighbourhood (1e-2 relative here) and ρ the
     observed per-iteration contraction.
   - Keep the subset-stable α for the full phase (the PnP fixed point depends on α; `recon_p2np` already does this).
   - Defaults with the box ordering: keep ≈ 12, draw 5–6 (cost × inflation minimal, 0.78–0.81 in §6).
4. **Do not implement for NCG / ZMSHR1 / PCG:**
   - a coil-sketch surrogate (Rademacher mixed maps per anchor epoch: `SetOp::select_coils` would need a mixed-map
     variant);
   - plain IHS.

   The safeguarded variants gain ≤ 1.2× late and lose early, and plain IHS diverges. Revisit only if a run is
   coil-I/O-bound (kstore disk tier): the inner model needs no k-space data (§4.1.5). There the saving is I/O, not
   FLOPs.
5. **`recon_p2np --coil-sampling` documentation:**
   - with the global ordering "keep" is not "the heart's coils";
   - with ZMSHR1 / NCG the sampling discards the secant pairs (as measured in recon_realism.md §4);
   - the measured λ_max inflation (1.3–3.9) eats the cost saving.


## 8. References

- Oscanoa et al., MRM 91:784 (2024), arXiv:2305.06482: coil sketching (verified; §3).
- Pilanci & Wainwright, JMLR 17(53) (2016): iterative Hessian sketch.
- Kim, Cauley, Nayak, Leahy, Haldar, MRM 86:197 (2021): ROVir (region-optimized virtual coils).
- Buehrer, Pruessmann, Boesiger, Kozerke, MRM 57:1131 (2007): array compression, 32-channel cardiac, ROI-SNR-optimal.
- Huang, Vijayakumar, Duensing, Akao, MRI 26:133 (2008): PCA software channel compression.
- Zhang, Pauly, Vasanawala, Lustig, MRM 69:571 (2013): geometric-decomposition coil compression.
- Johnson & Zhang, NeurIPS 2013 (SVRG); Defazio, Bach, Lacoste-Julien, NeurIPS 2014 (SAGA).
- Friedlander & Schmidt, SIAM J. Sci. Comput. 34:A1380 (2012); Byrd, Chin, Nocedal, Wu, Math. Program. 134:127
  (2012): dynamic sample sizes and the norm test.
- Byrd, Hansen, Nocedal, Singer, SIAM J. Optim. 26:1008 (2016); Moritz, Nishihara, Jordan, AISTATS 2016: stochastic
  quasi-Newton with held curvature samples.
- Simoncini & Szyld, SIAM J. Sci. Comput. 25:454 (2003): inexact Krylov relaxation.
