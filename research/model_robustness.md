# Model robustness: when does estimated physics make the reconstruction worse? (2026-11-16)

Question (user): the forward operator A_θ keeps getting richer (B0 map, coil maps, GIRF, concomitant fields, fat,
respiratory warps, temporal bases). θ is estimated, and the estimate θ̂ is wrong by some amount. Can using A_θ̂ be worse
than ignoring the effect (θ = 0, static model)? Which designs come with guarantees, or at least with principled tests,
so that gains are not artefacts of how our phantoms were driven but hold on a real scanner?

Tags on references: **[V]** full text read (text extracted and read); **[S]** full text queried through the fetch tool's
summariser (not read line by line); **[A]** abstract / metadata only. DOIs were checked against Crossref.
Numbers marked *toy* come from small 1D numpy models written for this note (§1.6, §4.2). They are illustrations of the
mechanisms, not predictions for our scans. Scripts (scratchpad, not in the repo): `toy_b0.py`, `toy_b0b.py`,
`toy_b0c.py`, `toy_motion.py`.

## 0. Answer

1. **Yes, an estimated model can be worse than no model, but only past a computable break-even.** For regularised least
   squares, the model error e = (A_true − A_θ̂)x enters exactly like additive data noise. Unlike noise it scales with the
   signal, so it does not average down with scan time. The image-error bound is "well-specified error + κ‖e‖" (§1.1).
   "Estimate beats ignore" therefore reduces, up to conditioning, to ‖(A − A_θ̂)x‖ < ‖(A − A_0)x‖.
   This is a sufficient condition for a *better bound*, not for a better image. No a-priori guarantee of the image
   itself exists, because some model error is invisible (it lies in range(A_θ̂) and is absorbed into x̂).
2. **B0, per voxel.** Write δ = ω − ω̂ and let T be the readout length. If |δ| ≤ min(|ω|, 2.78/T), the readout-averaged
   phase-error energy of the estimate is at most that of ignoring the field, for every voxel (§1.2).
   - At T = 11 ms the cap 2.78/T corresponds to **40 Hz**.
   - With |ωT| ≤ π, the simpler pointwise-in-τ condition |δ| ≤ |ω| suffices.
   - In the 1D toy the plug-in map beat "ignore" up to an RMS map error of ≈ 25 Hz, about 90 % of the field's RMS.
   - Realistic map errors (respiratory-state mismatch ≈ 10–40 Hz in the heart at 3 T, extrapolated from 7 T; noise
     ≲ 2 Hz) sit mostly, but not entirely, inside that region.
3. **Shrinkage is the cheap insurance.** Use α·θ̂, with α chosen by held-out data consistency from a grid that contains
   α = 0 and α = 1. By construction the result is never worse than "ignore" or "plug-in" on the held-out data, up to a
   generalisation gap of order √(log|grid| / m_eff). In the toy this choice recovered most of the oracle accuracy
   whenever the map was mis-scaled.
   The **expected operator** E_θ[A_θ] (damping e^{−σ²τ²/2}), with or without its variance penalty, was *worse* than the
   plug-in in the toy. It is the MSE-optimal predictor of the *data*, not of the image: it shrinks amplitudes and discards
   late-τ (high-k) information. Keep it as a candidate in the selection, not as a default.
4. **The real-data guarantee is held-out k-space validation.**
   - Hold out whole readouts (TRs), stratified over resp × cardiac bin × encode.
   - Reconstruct on the rest and compare per-readout residuals between operator choices with a paired, block-level test.
   - Noise contributes the same expectation to both arms, so the comparison estimates the difference in noise-free
     prediction error, with no ground truth needed.
   - In the toy, the held-out decision "plug-in vs ignore" was right in 74 of 80 cases (4 of the 6 misses at break-even), and
     held-out error tracked image NRMSE (Spearman 0.81). Whether it tracks image error **must be checked in our own
     simulations before it is trusted** ("validate the validator").
5. **Residual structure tests catch model error that a χ² level hides.** Three checks, all on whitened data:
   - R(τ): residual energy against readout time. B0 and GIRF errors grow with τ.
   - R(ρ, φ): residual energy against respiratory and cardiac phase. Motion model errors show up here.
   - The top eigenvalue of the residual coil covariance against the Marchenko–Pastur edge. Model errors are coherent
     across coils; noise is not.
6. **For velocity, only encode-differential model errors matter to first order.** Errors common to all encodes (B0 blur,
   coil maps, common motion) cancel in the phase difference and bias v only through partial volume. GIRF / eddy terms of
   the encode lobes, encode-dependent concomitant phase, and motion that differs between encodes do not cancel. Rank the
   robustness effort by encode-differential impact.
7. **Moving edges: a basis in the respiratory coordinate needs a knot step of about one edge width of displacement.**
   - A linear blend between knots makes double edges.
   - In the toy, an edge of width 1 mm moving 15 mm needs about 17 knots for an L2 error of 0.03.
   - **Warp first, basis on the residual** needs 5 knots for 0.04, even with a 30 % error in the warp amplitude.
   - The hybrid class contains both pure models, so its best-approximation error is at most the smaller of the two.
     Selecting α and K on held-out data keeps that property in practice.
8. **Operator-cost caveat.** A warp that varies continuously with ρ breaks the shared-FFT knot operator
   (temporal_operator.md §0.1: the image-domain part must not depend on time within a knot's support).
   - Rigid and affine motion can be applied per sample in k-space, exactly and continuously in ρ.
   - Non-rigid warps must be piecewise constant per respiratory bin, with the B-spline residual inside the bin.
9. **Process rule.** Every physics component is a candidate. It enters a run only if it lowers paired held-out residual
   significantly against the run without it, and α is logged. Simulations must use a different grid, a different physics
   path and a different perturbation family from the reconstruction (§3e). Otherwise the "gain" is an inverse crime.

## 1. Theory

### 1.1 Least squares with a wrong operator

Data: y = A x + n, with whitened noise n ~ CN(0, I), true operator A and model B = A_θ̂. The estimator is
x̂_B = argmin_x ½‖Bx − y‖² + R(x), with R convex. Write the model error in data space as e_B := (A − B)x. Then

  y = B x + n + e_B,

so the reconstruction sees the true image under the *model* operator, plus noise, plus e_B.

**Lemma 1 (data-space stability).** Let x̂(y) be the minimiser for data y. For any y₁, y₂,
‖B(x̂₁ − x̂₂)‖² ≤ Re⟨y₁ − y₂, B(x̂₁ − x̂₂)⟩, hence ‖B(x̂₁ − x̂₂)‖ ≤ ‖y₁ − y₂‖.
*Proof.* Optimality gives B^H(y_i − Bx̂_i) ∈ ∂R(x̂_i). Monotonicity of ∂R gives
Re⟨B^H(y₁ − Bx̂₁) − B^H(y₂ − Bx̂₂), x̂₁ − x̂₂⟩ ≥ 0. Rearrange. ∎

Let x̃_B = x̂_B(Bx + n) be the reconstruction we would get if B were exact. Then ‖B(x̂_B − x̃_B)‖ ≤ ‖e_B‖. To pass from
data space to image space we need a restricted lower bound:
- if R is μ-strongly convex, ‖x̂_B − x̃_B‖ ≤ ‖B^H e_B‖ / μ;
- for linear Tikhonov, G_B = (B^H B + λI)^{-1} B^H, x̂_B − x = (G_B B − I)x + G_B n + G_B e_B and ‖G_B‖ ≤ 1/(2√λ);
- on a set where B satisfies a restricted isometry with constant c, ‖x̂_B − x̃_B‖ ≤ ‖e_B‖ / c.

Together:

  ‖x̂_B − x‖ ≤ ‖x̃_B − x‖ + κ_B ‖e_B‖.            (1)

The first term is the well-specified error. The second is the misspecification term.

**Comparison.** The candidates B = A_θ̂ and B = A_0 differ by per-sample unit-modulus phase factors (B0, concomitant
phase, rigid motion as a k-space phase ramp) or by small smooth image-domain maps. So x̃ and κ are nearly equal for the two,
and (1) favours θ̂ exactly when

  ‖(A − A_θ̂)x‖ < ‖(A − A_0)x‖.                    (2)

Two limits on (2):
- (1) is an upper bound. Model error that lies in range(B) is invisible: it gets absorbed into x̂ as a structured artefact,
  and the residual sees only its component orthogonal to the data fit. No a-priori theorem can therefore promise "x̂_θ̂
  is closer to x than x̂_0" for an arbitrary x. That is why §3 measures instead of proving.
- Noise is fixed while ‖e_B‖ ∝ ‖x‖. As SNR and scan time grow, the misspecification term dominates. A model error that
  is harmless at dev scale can dominate a full 10-minute scan. Our R metric (girf_sensitivity.md,
  R = ‖(A − Ã)x‖ / (σ√(MC))) is exactly ‖e_B‖ in noise units, and R ≈ 0.2 is the right tolerance scale.

**Worst-case and stochastic robust LS.**
- Worst case. For unstructured uncertainty ‖ΔA‖ ≤ ρ, min_x max_ΔA ‖(B + ΔA)x − y‖ = min_x ‖Bx − y‖ + ρ‖x‖, which is a
  ridge (El Ghaoui & Lebret 1997 [A]).
- Stochastic. For random θ, E‖A_θ x − y‖² = ‖Āx − y‖² + x^H Σ x, with Ā = E[A_θ] and Σ = E[A^H A] − Ā^H Ā.
- **Lemma 2.** For every fixed x, B = Ā minimises E‖(A_θ − B)x‖² over all operators B, uniformly in x. The plug-in
  A_{E θ} is not that minimiser (Jensen gap).

  Robust LS is therefore the right tool for *predicting data*. It is biased for *estimating x*: the Σ term is a
  Tikhonov term that shrinks amplitudes. The toy (§1.6) confirms this.
- The Bayesian approximation-error approach (Kaipio & Somersalo 2007 [A]; Arridge et al. 2006 [A]) instead folds the
  prior-averaged model error into the noise covariance, which is the principled form of "model error is noise".

### 1.2 B0: per-sample and per-readout conditions

The column factor at voxel r and readout time τ is e^{−iω(r)τ}. Per voxel, the error factors are

  using θ̂: |e^{−iδτ} − 1| = 2|sin(δτ/2)|;  ignoring: |e^{−iωτ} − 1| = 2|sin(ωτ/2)|,   with δ = ω − ω̂.

**Pointwise in τ.** |sin(x/2)| increases with dist(x, 2πℤ). So the estimate is no worse at τ iff
dist(δτ, 2πℤ) ≤ dist(ωτ, 2πℤ). For all τ ∈ [0, T] this holds if |δ| ≤ |ω| and |ω|T ≤ π. Beyond |ω|T = π the ignored
phase wraps back to zero at τ = 2π/|ω|, so pointwise domination fails at isolated τ. That is irrelevant in practice.

**Averaged over the readout** (uniform τ, which is approximately right for our yarn readout):
(1/T)∫₀ᵀ |e^{−iδτ} − 1|² dτ = 2 m(δT), with m(x) = 1 − sin(x)/x.
- m increases on [0, 4.493], to a maximum of 1.217.
- Beyond that point its infimum is 0.8716 (at x = 7.725).
- m(2.777) = 0.8716.

**Proposition 1.** If |δ| ≤ min(|ω|, 2.777/T), then m(|δ|T) ≤ m(|ω|T).
- *Proof.* If |ω|T ≤ 2.777, both arguments lie in the increasing range. Otherwise m(|δ|T) ≤ m(2.777) = 0.8716 ≤ m(x) for
  every x ≥ 2.777. ∎
- At T = 11 ms: the estimate is per-voxel safe if its error is at most the field itself *and* below 40 Hz.

Readout-end phase and damping at T = 11 ms:

| Δf (Hz) | phase at T (rad) | m(2πΔf·T) | e^{−(2πΔf T)²/2} |
|---|---|---|---|
| 5 | 0.35 | 0.020 | 0.94 |
| 10 | 0.69 | 0.078 | 0.79 |
| 20 | 1.38 | 0.289 | 0.39 |
| 50 | 3.46 | 1.089 | 0.003 |

So a 50 Hz off-resonance has wrapped by the readout end. Ignoring B0 is not an option for our readout lengths; the real
question is how large a map error remains safe.

**From voxels to data.** The data error is Σ_τ ‖F_{K_τ}[(e^{−iδτ} − 1) S x]‖², where K_τ is the set of k-samples
acquired at readout time τ.
- If K_τ were spread over all of k-space (full sampling per τ, or uniform random sampling in expectation), Parseval turns
  Proposition 1 into (2).
- For a centre-out readout, K_τ is a shell |k| ≈ k(τ). The late-τ error then counts only through the high-|k| content of
  (e^{−iδτ} − 1)Sx, which is concentrated where x has edges.
- So the per-voxel condition is a sufficient *design* rule, and the held-out residual by τ band (§3b) is the real test.

**Sizes of realistic errors.**
- Map noise: ≲ 2 Hz from the multi-echo train's CRLB (b0_r2s_model_based.md: B0 is overdetermined by about 10⁴).
- Respiratory-state mismatch:
  - At 7 T in the heart, ΔB0 between expiration and inspiration in shallow breathing reached up to −90 Hz, with a 35 Hz
    average, and was mostly a shift of the centre frequency (Dietrich et al. 2022 [A]).
  - Scaling by 3/7 gives roughly 15 Hz mean and 40 Hz maximum at 3 T. This is an extrapolation; Huang et al. 2025 [S]
    give respiratory variation at 3 T only in %.
- Map timing. Sutton et al. 2004 [V] note that a 1 Hz respiratory centre-frequency change between two TE images yields a
  ≈ 15 Hz field-map error (TE 30 ms, ΔTE 2 ms). The multi-echo map must therefore come from the same respiratory state,
  or be respiratory-resolved.
- **Consequence:** a single static map is at or past the 40 Hz break-even near the heart–lung interface. A
  respiratory-resolved map (or a per-bin constant offset, the dominant mode at 7 T) is what keeps us inside Proposition 1.

**Shrinkage α.** Use ω̂_α = αω̂ with ω̂ = ω + ε, ε independent of ω, E ε² = σ², E ω² = s² (or |ω| ≤ W for worst case).
- Then δ = (1 − α)ω − αε. For Gaussian δ ~ N(0, v(α)), E|e^{−iδτ} − 1|² = 2(1 − e^{−v(α)τ²/2}), which is monotone in
  v(α) = (1 − α)²s² + α²σ².
- So **α\* = s²/(s² + σ²) is optimal simultaneously for every τ**. With W² in place of s², it is the minimax choice
  over |ω| ≤ W.
- If ω̂ is already a posterior mean (a regularised map fit), then α\* ≈ 1 and shrinkage only guards against unmodelled
  error, for example a respiratory mismatch. That is why α should be *selected on data* (§3a), not computed from an
  assumed σ.

### 1.3 Motion warps

Model x_ρ = W_ρ x_ref, with true displacement u and estimate û.

- **Step edge.** Take an edge of contrast h across a surface Γ with normal n. For a planar edge, and to first order in
  curvature, ‖x(· − u) − x(· − û)‖² = h² ∫_Γ |(u − û)·n| dS.

  The static model has h² ∫_Γ |u·n| dS. This is L¹-like in the displacement: edges punish linearly, not quadratically.
  The estimate is no worse wherever |(u − û)·n| ≤ |u·n|. That holds for û = αu with α ∈ [0, 2], and for any û whose
  normal component lies between 0 and 2u·n. Unlike B0 there is no wrap; the condition is a ball condition on the normal
  component.
- **Smooth regions.** The error is ≈ ‖∇x·(u − û)‖ against ‖∇x·u‖, so the same ball condition applies along ∇x.
- **Interpolation cost of the warp.** Applying a sub-voxel shift s by linear interpolation has error
  |(1 − s) + s e^{2πiξ} − e^{2πiξs}|, against |e^{2πiξs} − 1| for ignoring the shift. At ξ = 0.4–0.5 cycles/voxel the
  two are nearly equal; at ξ = 0.5, s = 0.1, linear interpolation is *worse* (0.34 against 0.31).
  - Cubic B-spline interpolation (with prefilter) is 5–20× better below ξ = 0.25 but equally poor at Nyquist.
  - So sub-voxel non-rigid warps add little at the finest scale.
  - Rigid and affine motion applied in k-space (trajectory rotation and phase ramp; motion_navigators.md §2) has *no*
    interpolation error and is preferred wherever it is valid.
- **Model-class error.** Sliding at the pleura and chest wall is not a diffeomorphism. A smooth warp cannot represent it,
  and the residual must absorb it (§4).
- **Coil frame.** Coil maps belong to the coil frame. The anterior array moves with the chest, so neither the scanner
  frame nor the body frame is exact. This is the same class of error as §1.4.

### 1.4 Coil maps

- **Full sampling.** The coil-combined estimate is x̂ = (Σ_c Ŝ_c^* S_c / Σ_c |Ŝ_c|²) x, a smooth multiplicative bias that
  is common to all encodes.
- **Undersampling.** A map error ΔS also leaves an un-unfolded residue at the aliasing positions. That residue is not
  multiplicative and does not cancel.
- **"Ignore" here** means a lower-resolution or sum-of-squares model, so the comparison is between candidate maps, not
  against zero.
- **Existing robust variants.**
  - ESPIRiT's eigenvalue weighting already behaves like a shrinkage: maps are scaled down where the calibration
    eigenvalue is below 1, and a second map set covers FOV folding (Uecker et al. 2014 [A]).
  - Joint map-and-image estimation (NLINV, Uecker et al. 2008 [A]) is the coil analogue of §3d.

### 1.5 GIRF, trajectory and gauge

A trajectory error Δk(τ) changes sample j by ≈ ∇_k x̂(k_j)·Δk_j. Since |∇_k x̂| ≤ 2π ∫ |r||x(r)| dr, the phase-like
error is 2π|Δk|R_obj. The ball condition |Δk − Δk̂| ≤ |Δk| applies per sample.

**Degeneracies.** A linear B0 term ω = γ g·r is exactly a trajectory drift Δk(τ) = γgτ/2π, and a constant off-resonance
ω₀ is a global phase drift along the readout. Joint estimation of GIRF delays, B0 and centre frequency must therefore fix
a gauge:
- linear B0 terms come from the field map / shim, *or* from the trajectory, never from both;
- per-axis delays come from the scan data (girf_sensitivity.md: CRLB 0.004–0.008 ns).

Otherwise the joint fit wanders along the null direction, and "improvements" in one parameter are paid for in another.

### 1.6 Toy check of §1.1–1.2 (1D, off-resonance only; *toy*)

**Setup.**
- Object and field are simulated on a 4× finer grid than the reconstruction, with a sharper field peak near an
  "interface" (no inverse crime).
- 128 voxels of 1 mm; 32 readouts × 96 samples, each centre-out with a random speed profile; T = 10 ms. 24 readouts are
  used for reconstruction and 8 are held out.
- Field −36 … +69 Hz.
- Map error families that differ from the generator: global scale (1 + η); smooth random field of relative RMS η;
  misregistration by 10η mm; blurring by a 20η mm kernel.
- Quadratic-smoothness reconstruction (direct solve); 8–10 seeds.

| error (RMS) | ignore | plug-in | α by holdout | expected op. | expected + var |
|---|---|---|---|---|---|
| scale, 7 Hz | 0.113 | 0.060 | 0.053 (α 0.78) | 0.063 | 0.088 |
| scale, 14 Hz | 0.113 | 0.078 | 0.055 (α 0.66) | 0.088 | 0.204 |
| scale, 28 Hz | 0.113 | **0.125** | 0.052 (α 0.50) | 0.180 | 0.371 |
| smooth, 14 Hz | 0.113 | 0.081 | 0.077 | 0.091 | 0.205 |
| smooth, 28 Hz | 0.113 | **0.134** | 0.097 (α 0.34) | 0.189 | 0.371 |
| shift 10 mm, 16 Hz | 0.113 | 0.095 | 0.095 | 0.108 | — |
| blur, 11 Hz | 0.113 | 0.078 | 0.077 | 0.084 | — |

Values are image NRMSE; the oracle is 0.052.

**Findings.**
- Break-even near 25 Hz RMS.
- α selected by holdout was never materially worse than the better of ignore and plug-in. It was 0.003 worse at tiny
  errors, from selection noise.
- The expected operator lost everywhere. With the variance term, most of the loss is amplitude shrinkage: the best
  scalar refit lowers 0.208 to 0.102, still worse than the plug-in's 0.078.
- The paired held-out test chose correctly in 74 of 80 seeds. Mean z was +2.4 to +3.5 at errors of 7–14 Hz and
  −2.3 to −3.0 at 42 Hz. Of the 6 wrong calls, 4 were at the 28 Hz break-even level and 2 at 42 Hz (smooth family),
  with 8 held-out readouts only.

## 2. Literature

**Field-map errors and off-resonance correction.**
- Noll et al. 1991 [A], 10.1109/42.108599 (conjugate phase); Man et al. 1997 [A], 10.1002/mrm.1910370616 (autofocus
  without a map, i.e. selection by an image metric).
- Sutton, Noll & Fessler 2003 [A], 10.1109/TMI.2002.808360: time-segmented iterative reconstruction; more accurate than
  conjugate phase when the map is not smooth.
- Fessler et al. 2005 [A], 10.1109/TSP.2005.853152 (Toeplitz).
- **Sutton, Noll & Fessler 2004 [V], 10.1002/mrm.20079:**
  - joint estimation of image and field map from a spiral-in/out acquisition; field-map error about half that of the
    standard two-TE estimate;
  - the iterative reconstruction "may be limited by errors in the estimated field map";
  - convergence is slow from a zero-map initialisation, so they initialise with the standard map;
  - quadratic regularisation smooths map peaks at object edges (where our errors will be);
  - respiratory centre-frequency changes of 1 Hz produce ≈ 15 Hz errors in the two-TE map.
- Funai et al. 2008 [A], 10.1109/TMI.2008.923956 (regularised map estimation); Olafsson et al. 2008 [A],
  10.1109/TMI.2008.917247 (joint dynamic R2* + map).
- Wilm et al. 2011 [A], 10.1002/mrm.22767 (expanded encoding model with measured field dynamics).
- Respiratory B0: Van de Moortele et al. 2002 [A], 10.1002/mrm.10145 (brain); Dietrich et al. 2022 [A],
  10.1002/mrm.29144 (7 T heart); Huang et al. 2025 [S], 10.1002/mrm.70026 (3 T heart, motion-adaptive shim, variation
  given in %).
- **Gap.** I found no paper that maps field-map error in Hz to image error for model-based reconstruction. Proposition 1
  and the toy fill that gap only qualitatively.

**Motion-field errors and motion-compensated reconstruction.**
- General matrix model: Batchelor et al. 2005 [A], 10.1002/mrm.20656. GRICS (coupled reconstruction + motion-model
  fit, fixed-point multiresolution): Odille et al. 2008 [A], 10.1002/mrm.21623.
- Bin-wise non-rigid MC: Cruz et al. 2016 [A], 10.1002/mrm.25708. Soft-gating plus non-rigid: Cheng et al. 2015 [A],
  10.1002/jmri.24785.
- Consistency-weighted least squares: Johnson et al. 2012 [A], 10.1002/mrm.23144. Data inconsistency is *down-weighted*
  rather than modelled, which is the soft-gating analogue of shrinkage.
- iMoCo: Zhu et al. 2020 [A], 10.1002/mrm.27998 (motion fields registered from respiratory-resolved images).
- MR-MOTUS: Huttinga et al. 2020 [A], 10.1088/1361-6560/ab554a (motion fitted directly to k-space against a reference
  image).
- TAMER: Haskell et al. 2018 [A], 10.1109/TMI.2018.2791482 (data-consistency-driven motion search).
- Aligned SENSE: Cordero-Grande et al. 2016 [A], 10.1109/TCI.2016.2557069.
- **DC-CS: Lingala et al. 2015 [A], 10.1109/TMI.2014.2343953.** x_t = W_t(c_t) with a CS prior on the deformation-
  corrected residual. The abstract states it "does not require perfect compensation of motion", and the optimal λ was
  lower than for plain CS. This is the closest published form of the §4 hybrid.
- **XD-ORCCA: Correia et al. 2018 [A], 10.1002/mrm.27208.** Translational correction from iNAVs plus sparsity in the
  respiratory dimension. It beat XD-GRASP on coronary sharpness and is the coronary precedent for the hybrid.
- Motion-resolved alternatives: XD-GRASP (Feng et al. 2016 [A], 10.1002/mrm.25665); 5D whole-heart (Feng et al. 2018
  [A], 10.1002/mrm.26745; Di Sopra et al. 2019 [A], 10.1002/mrm.27898); **5D flow** (Ma et al. 2020 [A],
  10.1148/ryct.2020200219); low-rank / multitasking (Christodoulou et al. 2018 [A], 10.1038/s41551-018-0217-y; Ong et al.
  2020 [A], 10.1002/mrm.28235).
- **Gap.** I found no paper with a theorem on the bias from using an estimated motion field. The literature evaluates
  empirically: sharpness, vessel length, against gated references.

**GIRF and trajectory.**
- Vannesjo et al. 2013 [A], 10.1002/mrm.24263; Vannesjo et al. 2016 [A], 10.1002/mrm.25841.
- Trajectory auto-correction by joint estimation: Ianni & Grissom 2016 [A], 10.1002/mrm.25916.
- In the repo: girf_sensitivity.md (R metric, recon-side delay / resonance refit).

**Theory.**
- Perturbation of LS solutions: Stewart 1977 [A], 10.1137/1019104.
- Robust LS: El Ghaoui & Lebret 1997 [A], 10.1137/S0895479896298130.
- Perturbed operators in compressed sensing: Herman & Strohmer 2010 [A], 10.1109/JSTSP.2009.2039170 (arXiv:0907.2955).
  Recovery error is bounded by relative operator and data perturbations, with worst-case constants that degrade mildly.
- Sparse total least squares: Zhu, Leus & Giannakis 2011 [A], 10.1109/TSP.2011.2109956.
- **Cross-validation in CS: Ward 2009 [A], 10.1109/TIT.2009.2032712 (arXiv:0803.1845).** Reserving ≈ 4 log p
  measurements bounds the error of all p candidate reconstructions with high probability (Johnson–Lindenstrauss). This
  is the theoretical licence for §3a, though for random rows, not our structured ones.
- Variable projection: Golub & Pereyra 2003 [A], 10.1088/0266-5611/19/2/201. PALM convergence for alternating schemes
  under the Kurdyka–Łojasiewicz (KL) property: Bolte, Sabach & Teboulle 2014 [A], 10.1007/s10107-013-0701-9.
- Incidental-parameter inconsistency: Neyman & Scott 1948 [A], 10.2307/1914288.
- Residual whiteness for parameter choice: Almeida & Figueiredo 2013 [A], 10.1109/TIP.2013.2257810.
- Inverse crimes: Kaipio & Somersalo 2007 [A], 10.1016/j.cam.2005.09.027; Guerquin-Kern et al. 2012 [A],
  10.1109/TMI.2011.2174158 (analytic phantoms); Shimron et al. 2022 [A], 10.1073/pnas.2117203119 (hidden preprocessing
  inflates gains by up to 48 %).
- k-space splitting: Yaman et al. 2020 [A], 10.1002/mrm.28378 (SSDU).

## 3. Tests and designs that work on real data

### (a) Held-out data consistency: model selection without ground truth

**Split.** Hold out a set H of whole readouts (≈ 5 %), drawn as random TRs stratified over resp bin × cardiac bin ×
encode. Whole readouts matter for two reasons:
- along a readout, the samples share the same τ-dependent model error;
- single held-out samples are trivially predicted by their NUFFT neighbours. This is a leakage crime, as are the k ≈ 0
  samples of a centre-out readout. Report the residual with |k| < k_c excluded as well as included.

**Statistic.** For each candidate operator B, reconstruct x̂_B from the complement set and compute the whitened residual
r_B = y_H − B_H x̂_B. Since n_H is independent of x̂_B,

  E‖r_B‖² = ‖A_H x − B_H x̂_B‖² + |H|,

and the noise term is identical for every candidate. The difference between two candidates is therefore an unbiased
estimate of the difference in noise-free *prediction* error.

What this does and does not estimate:
- Prediction error is not image error. A_H x − B_H x̂ = B_H(x − x̂) + (A_H − B_H)x, so it combines the image error, seen
  through the operator, with the model error at the true image. Both are things we want small.
- The relation to image error holds for randomised, Parseval-like held-out sets (Ward 2009). For ours it must be
  verified in simulation (§3e). In the toy, Spearman ρ = 0.81.

**Decision rule.** Use per-readout paired differences d_i = |r_{0,i}|² − |r_{θ̂,i}|², summed per TR. Test with a block
bootstrap over TRs (or a sign test). Accept the component if the lower 95 % bound of mean d is > 0.

The selection guarantee: with a grid of G candidates (α values, K values, components) and m_eff independent held-out
blocks, Hoeffding plus a union bound gives

  PE(selected) ≤ min_g PE(g) + 2 v √(2 log(G/δ) / m_eff).

Here v bounds the per-block spread. Since α = 0 is always in the grid, the selected model is not worse than ignoring
the effect, up to that gap, with probability 1 − δ.

**Cost.** One extra reconstruction per candidate, at dev scale (§5). Selection is done once per protocol / subject.
The final reconstruction uses all data with the selected α (standard CV refit).

### (b) Residual structure

All on whitened data, in-sample and held out.
1. **χ² level** on H. ‖r‖²/|H| should be ≈ 1 + (image prediction error). In-sample values well below 1 mean overfitting
   (discrepancy principle, Morozov).
2. **R(τ)**: mean |r|² in readout-time bands. A correct B0 / GIRF gives a flat curve; errors make it grow ∝ m(δτ). This
   is the readout-time analogue of Proposition 1.
3. **R(ρ), R(φ_card), R(encode)**: motion model, binning and encode-specific errors. A rise at end-inspiration means the
   warp or map fails there.
4. **Coil coherence.** Take C_r = (1/M) Σ r r^H (C × C). Under the null it is ≈ I. Model error has coil structure
   (∝ S_c × something), so λ_max(C_r) > (1 + √(C/M))² (the Marchenko–Pastur edge) flags it. With M ~ 10⁷ the test is
   extremely sensitive; use it per τ band and per resp bin.
5. **Lag-1 autocorrelation along the readout** of r. Noise is white (after the ADC filter, measure its baseline once);
   smooth model error is not (Almeida & Figueiredo use residual whiteness for parameter choice).

### (c) Robust and partial use

- **α-curve.** PE_H(α) for α ∈ {0, 0.25, 0.5, 0.75, 1, 1.25}, per component. A minimum near 1 means the estimate is
  calibrated. A minimum at α < 1 means a biased or noisy estimate: use α\* and investigate. A flat curve means the data
  do not inform the component, so drop it (cost without benefit).
- **Spatially varying shrinkage**: α(r) = α₀ in voxels whose map has low confidence (fit residual, R2*, distance to the
  lung interface). The expected-operator damping e^{−σ(r)²τ²/2} is the alternative, but it lost in the toy with uniform σ.
  It may win only where σ(r) is large and localised; that is untested.
- **Soft weighting of inconsistent data** (Johnson 2012): down-weight readouts whose held-out residual stays high under
  every candidate (arrhythmic beats, coughs) rather than letting them pull θ.

### (d) Joint estimation (alternate x and θ)

**Properties.**
- min_{x,θ} ½‖A_θ x − y‖² + R(x) + Q(θ) is non-convex in θ. With KL-type regularity and proximal steps (PALM) the
  iterates converge to a critical point, not to the global minimiser.
- Profiling x out (variable projection) gives the same stationary points with better conditioning.

**Bias sources.**
- Joint MAP over (x, θ) is not the marginal MAP for θ. With an image of 10⁶–10⁷ unknowns, the incidental parameters
  (Neyman–Scott) can bias θ.
- The prior also pulls θ toward values that make x̂ "look" sparse; for motion this means toward over-sharpening.
- Gauge directions (§1.5) must be fixed.

**Validated alternation (what we can guarantee).**
- Start at θ = 0 or at the calibration θ̂.
- Accept an outer θ-update only if PE_H decreases significantly (§3a). Otherwise backtrack: halve the step, or shrink
  toward the previous θ.
- The held-out error is then monotone non-increasing by construction. With finitely many accepted steps, the §3a bound
  applies to the final θ.
- This is cross-validated early stopping. It does not fix bias inside the accepted region, but it cannot end worse than
  the start on held-out data.
- Low-dimensional θ (B0 on 2 cm splines, ≈ 10⁴ coefficients; poses ≈ 10²–10³) is far better determined than x
  (b0_r2s_model_based.md), so the incidental-parameter bias is expected to be small for B0 and poses. It is larger for
  dense non-rigid motion.

### (e) Monte-Carlo perturbation and simulation–reconstruction independence

**Perturbation curves.** In simulation, perturb θ̂ = θ_est + η·ε with error *families* that differ from the phantom's
generator:
- B0: global scale; smooth random field; respiratory-state mismatch; misregistration; low-resolution blur; localised
  error at the lung interface.
- Motion: amplitude miscalibration (bellows → mm); inhale/exhale hysteresis (invisible to bellows); slow drift; sliding;
  a rigid fit to non-rigid motion.
- GIRF: delay, resonance f₀ drift, missing cross terms.

Plot image / velocity error and PE_H against η.

**Requirements.**
1. Error is monotone in η.
2. The break-even η_c (plug-in error = ignore error) is at least 2× the realistic error level from calibration
   CRLB / QA.
3. PE_H ranks the conditions like the true error ("validate the validator"). Spearman ≥ 0.8 is a reasonable acceptance
   level; the toy reached 0.81.

**Independence.** Simulation and reconstruction must not share:
- **grid**: simulate at ≤ 0.5 mm or analytically, reconstruct at 1 mm;
- **physics code path**: the simulator's own B0 / concomitant / GIRF integration, not the reconstruction's segment
  basis;
- **parameter source**: the reconstruction uses θ estimated by the real pipeline (multi-echo map from simulated
  multi-echo data, motion from simulated navigators / bellows, GIRF from the simulated QA sequence), never θ_true;
- **motion model**: the phantom moves by CFD / LBM deformation including sliding, while the reconstruction warps are
  B-spline or affine;
- **noise and coil model**: measured noise covariance; coil maps from a different coil-simulation method than the
  reconstruction's ESPIRiT.

A noise-free, matched-operator control (inverse crime on purpose) gives the floor. The gap between that control and the
independent simulation is the honest model-error budget.

## 4. Temporal basis in the respiratory coordinate, and a robust hybrid

### 4.1 What is known

There are three families.
1. **Respiratory-resolved binning**, regularised along the respiratory dimension (XD-GRASP, 5D whole-heart, 5D flow).
   No motion model, so nothing can be "wrong"; the price is intra-bin blur and fewer samples per bin.
2. **Motion-compensated**: one reference image plus warps (GRICS, iMoCo, Cruz, MR-MOTUS). It uses all data, but every
   warp error becomes image error.
3. **Low-rank / subspace** along the motion dimension (multitasking, Extreme MRI). Moving edges are not low-rank, so the
   rank grows with displacement / edge width.

The hybrids that exist in the literature are warp + residual (DC-CS) and translation + respiratory sparsity
(XD-ORCCA, coronary). Their abstracts report that imperfect motion compensation still helps.

### 4.2 Interpolating a moving edge (*toy*)

An edge of width 1 mm moves 15 mm over ρ ∈ [0, 1] with a linear B-spline in ρ. The table gives the L2 error per state
(edge height × √mm); a static model has 2.84.

| knots K (step) | basis only | warp (1 − η)u + basis, η = 0.1 | η = 0.3 | η = 0.5 |
|---|---|---|---|---|
| 2 (15 mm) | 1.50 | 0.061 | 0.41 | 0.78 |
| 3 (7.5 mm) | 0.97 | 0.017 | 0.17 | 0.41 |
| 5 (3.75 mm) | 0.48 | 0.004 | 0.042 | 0.14 |
| 9 (1.9 mm) | 0.14 | 0.001 | 0.009 | 0.029 |
| 17 (0.94 mm) | 0.029 | 0.000 | 0.002 | 0.006 |

Findings:
- At K = 2 the blend at mid-cycle is a plateau of height 0.5: a double edge.
- The error is a function of (residual displacement per knot step) / (edge width). Basis-only with a 1.9 mm step (0.14)
  matches the hybrid with 0.3 × 7.5 = 2.25 mm residual per step (0.17).
- **Rule:** knot step × |residual motion fraction| ≲ edge width (≈ 1 voxel at 1 mm, i.e. the PSF). For a 15 mm
  diaphragm excursion this means about 16 knots without a warp, or about 5 knots with a warp that is right to 30 %.

### 4.3 Proposed hybrid and guarantee sketch

**Model.** x(ρ, φ) = W_{α û(ρ)} [ Σ_b β_b(ρ) c_b(φ) ], where:
- û comes from the navigators / bellows / registration;
- α ∈ [0, 1] is selected on held-out data;
- the residual basis has K knots in ρ (cardiac φ unchanged).

Nested classes: α = 0 gives the pure basis (respiratory-resolved), and K = 1 gives pure motion compensation.

**Claim 1 (approximation).** For any true family {x_ρ}, the best-approximation error of the hybrid class at (α, K) is
at most min(basis-only error at K, MC error at α). This follows directly from nestedness. With LS fitting on the same
data, the in-sample fit is ordered the same way.

**Claim 2 (selection).** Choosing (α, K) on held-out readouts from a finite grid (α ∈ {0, 0.5, 0.75, 1}, K ∈ {1, 3, 5,
9}) gives PE within 2v√(2 log(G/δ)/m_eff) of the best grid point (§3a). So the selected hybrid is, on predicted data,
not worse than either pure model up to the gap.

The variance cost of extra parameters is what held-out selection prices in. Neither claim says anything about image
error beyond §3a's validator caveat.

**Operator constraint** (temporal_operator.md §0.1 condition):
- Rigid / affine û: apply per sample in k-space (rotate k_j by R(ρ_j), phase e^{−2πik·t(ρ_j)}, coil maps warped once
  per pose cluster). This is exact, continuous in ρ, and keeps FFT sharing.
- Non-rigid û: piecewise constant per respiratory bin (image-domain warp per bin; cubic B-spline interpolation, §1.3),
  with the residual basis inside or across bins.
- Do not blend warped knot images, Σ_b β_b(ρ) W_{u(ρ_b)} c. That reintroduces double edges at the scale of the *true*
  displacement between knots.

**Velocity note.** The 7 encodes repeat every ≈ 99 ms, against a ≈ 4 s respiratory cycle, so respiratory motion is
nearly common to all encodes and cancels in the phase difference. What does not cancel is a different respiratory state
of an encode's data when bins are unbalanced. So bins should be encode-balanced, or encode-differential residuals should
be checked via R(encode).

## 5. Recommendations

### 5.1 Built into every reconstruction run (`--holdout` design; the code changes are not made here)

1. **Held-out set.** ≈ 5 % of TRs, seeded and stratified (resp × cardiac × encode), written to the run's `--out`
   directory with the config fingerprint. Report PE_H with and without |k| < k_c.
2. **Component ladder.** Each physics component P ∈ {B0 map, B0 per-bin offset, GIRF refit, concomitant, fat, coil-map
   variant, motion warp, temporal basis order} carries a flag `--α_P`. The ladder run evaluates PE_H for α_P ∈
   {0, 0.5, 1} at dev scale (2.5 mm, few iterations, warm start) with every other component at its accepted value. The
   rule from §3a decides. Log the α-curve, the paired-difference CI and the decision. Concomitant fields (θ known exactly
   from the waveform) are exempt from shrinkage but not from the check.
3. **Residual diagnostics in every run** (cheap, computed from the last gradient's residual): R(τ) in 8 bands, R(ρ),
   R(φ), R(encode), λ_max(C_r) / MP-edge per band, lag-1 autocorrelation. Write `residual_diag.json` plus figures.
   **Flag**: R(τ) slope > 10 % between first and last band; λ_max > 1.5 × edge; any resp bin with R > 1.2 × median.
4. **Joint updates** (B0 offsets per resp bin, GIRF delays, poses) only through validated alternation (§3d), with the
   gauge fixed (§1.5).
5. **Report velocity-relevant errors separately.** Encode-differential residuals (R(encode) spread), and in simulation
   ROI flow error, not only NRMSE.

### 5.2 Experiments (each pre-registered: question → decision rule)

- **E1 Validator validity** (simulation, dev scale).
  - Q: does PE_H rank operator choices like true image *and velocity* error, for B0 / motion / GIRF perturbation
    families that differ from the generator?
  - Decision: Spearman ≥ 0.8 for both metrics → adopt §3a as the acceptance gate. Otherwise fix the held-out design
    (block size, k_c, stratification) before any other decision.
- **E2 B0 break-even** (simulation, ph0001 box, independent multi-echo map).
  - Q: η_c for scale / smooth / respiratory-mismatch / misregistration errors at T = 9.5–11 ms.
  - Decision: if η_c < 2 × the realistic error (CRLB + respiratory mismatch, §1.2), build a respiratory-resolved or
    per-bin-offset B0 model before anything else. Otherwise keep the static map with α by holdout.
- **E3 Expected operator vs α** (simulation).
  - Q: does damping with spatially varying σ(r) (lung interface) beat α-shrinkage there?
  - Decision: adopt only if better in both PE_H and ROI velocity error. The uniform-σ toy says no.
- **E4 Hybrid motion** (simulation with sliding, hysteresis, bellows-scale errors).
  - Q: on PE_H, myocardium / diaphragm edge error and coronary ROI flow, compare binned (K = bins), MC (K = 1),
    hybrid (K = 3, 5) × α.
  - Decision: adopt the hybrid if it beats both pure models on PE_H *and* edge error at equal cost. If K = bins wins,
    motion estimates are not yet good enough (α → 0).
- **E5 Real-data α-curves** (first volunteer scans).
  - Q: where are the PE_H minima per component, and do the residual diagnostics flag anything?
  - Decision: any component with a minimum at α ≤ 0.5 or a flat curve is disabled and its estimator investigated. No
    component is turned on in the real pipeline on simulation evidence alone.
- **E6 Swap / sign tests** (real data, negative controls).
  - Q: does PE_H get worse with θ̂ from another session, with −θ̂, or with the opposite respiratory phase?
  - Decision: if not, the component is not informed by the data. Treat it as a regulariser and drop it from the physics
    claim.
