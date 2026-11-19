# Static-tissue temporal lock and divergence-free wavelets (2026-11-09)

Two priors that act after the main prior (LLR, db4 wavelet or the learned denoiser) in every P²nP iteration:
1. a temporal lock of the voxels that neither move nor carry flow, from a static / dynamic map estimated from the
   iterate itself;
2. divergence-free wavelet (DFW) shrinkage of the velocity.

Code:
- `hasty_recon::tlock` (map + lock);
- `hasty_recon::dfwavelet` (DFW);
- `hasty_recon::sepwt` (the separable wavelet engine shared with the db4 prior);
- flags in `recon_p2np` (`--static-lock`, `--dfw`).

## 1. Why

- **Intracranial:** nearly all tissue is static (brain pulsation is ~0.1 mm). Only vessels and CSF change over the
  cardiac cycle.
- **Thorax:** the heart, the great vessels and the tissue around them move with the heartbeat. The chest wall, liver
  and heart move with breathing. Spine, back muscles and arms (when in the FOV) do neither.
- A static voxel's image is the same in all F_c × F_r frames. Locking it gives F_c × F_r times the data per unknown.
- Every frame's undersampling artefact there is removed. Incoherent aliasing spreads over the FOV, so a cleaner
  static region also means less aliasing energy left for the dynamic region.
- This is the spatially-adaptive version of what k-t methods do with a temporal basis. k-t PCA / k-t SPARSE-SENSE
  keep few temporal components everywhere. Here static voxels keep exactly one (the temporal DC), dynamic voxels
  keep all of them, and the main prior (LLR / learned) handles the dynamic part.

## 2. Map (data-derived, no labels)

Notation:
- x[r, c, e]: respiratory bin r, cardiac frame c, encode e.
- b = G_σ ∗ x: a Gaussian low-pass, σ = `--lock-blur` (3 mm).

Why low-pass: per-frame incoherent aliasing is mostly high spatial frequency, while organ motion and flow survive
the low-pass. The k-space centre is densely sampled in every frame.

Variances, pooled over (r, e):
- cardiac variance: V_c(q) = mean_{r,e} Var_c b[r, ·, e](q);
- respiratory variance of the bin means: V_r(q) = mean_e Var_r (mean_c b)(q).

Noise / aliasing level σ²(q):
- second differences in cardiac phase: D(q) = mean |b_c − 2b_{c−1} + b_{c−2}|² / 6.
  - For frame-independent errors, D = σ².
  - For smooth motion D is small: a first harmonic over F = 20 frames gives (2π/20)⁴ / 6 ≈ 0.2 % of its variance.
- σ²(q) = medians of D over 24 mm blocks (`noise_block_mm`), trilinear between block centres. Moving voxels are a
  minority per block. This is local because coil-combined noise and aliasing are not uniform.
- Divided by the blur's exact white-noise gain per voxel g(q): edges repeat samples, so corners have higher gain.
  Without this the FOV corners scored ≈ 2 in the unit test.

Scores (≈ 1 for a static voxel; dof = n_e n_r (F_c − 1) complex samples for s_c):
- s_c = √(V_c / σ²);
- s_r = √(V_r / (σ² / F_c));
- s_v = |v̄| / σ_|v|: velocity of the temporal-mean images (decoded per voxel) against its noise. The noise is
  σ_|v| = √3 · √2 σ_φ / k_eff, with σ_φ = √(σ² / (2 F_c F_r)) / |m̄_0| and k_eff² = Σ_e |k_e|² / 3. Only counted
  above `--lock-vmin` (3 cm/s). Steady flow (veins) does not vary in time and must still not be locked.

Detection power. Undetectable is not static:
- On sim_test (10 s scan, ~73 readouts per frame), the first version called 99.7 % of the voxels static, heart
  included, because per-frame aliasing swamps the motion.
- Successive refreshes then swung 99.7 → 60.9 → 81.1 %.
- Locked voxels lose their variance, so a wrong lock can reinforce itself.
- Guard: conf = 1 where a relative variation (std / |m̄|) of `detect_ok` = 0.1 would have scored s_hi; 0 where not
  even `detect_max` = 0.3 would; linear in between.
  - Per frame for the cardiac lock, per bin mean (σ / √F_c) for the respiratory one.
  - Voxels without significant mean signal (|m̄| < 3σ / √(F_c F_r): air) keep conf = 1, since locking them costs
    nothing and removes their aliasing.
  - The log reports the "undetermined" fraction (signal, conf < 0.5).
- Unit test `undetectable_motion_is_not_locked_air_is`: under heavy noise, tissue w = 0.000 and air w = 1.000.

Static weights:
- w_c = ramp(max(s_c, s_v)) · conf_c and w_r = ramp(max(s_r, s_v)) · conf_r, with ramp = 1 below s_lo = 1.5 and 0
  above s_hi = 3 (`--lock-score`; `--lock-detect OK:MAX` for the detection power).
- Min filter over `--lock-dilate` (3 mm): the dynamic region grows by a safety margin.

Refresh: from iteration IT0, every EVERY iterations. The map is saved with its iteration (`static_map_<tag>.npy`
[x, y, z, 6] + `.json`), so a resumed run refreshes at the same iterations.

## 3. Lock (proximal map)

Orthogonal decomposition per voxel and encode: x_{r,c} = x̄ + (m_r − x̄) + (x_{r,c} − m_r), with m_r the bin mean.
Each deviation is scaled.

- **quadratic** (default): the prox of (μ w / 2)‖dev‖², i.e. dev ← dev / (1 + μ w). Firmly non-expansive, and
  composes with the P²nP fixed-point argument like any prox. μ = 20 ≈ a hard lock.
- **group** (`--lock-mode group`): the prox of μ w σ_x √dof ‖dev‖ (a group soft threshold over the frames), with σ_x
  the local noise level of the unfiltered images.
  - A static voxel goes exactly to its mean.
  - A voxel that moves keeps its motion minus the threshold, even where the map is wrong. This is the robust choice
    if the map turns out unreliable.
- The cardiac lock acts inside each respiratory bin with w_c; the respiratory lock acts on the bin means with w_r.
  Tissue that moves with breathing but not with the heart (liver dome, chest wall) is locked over the cardiac cycle
  only. A heart voxel without breathing variation keeps its cardiac motion and has its bins aligned.

Unit test (`tlock::tests`):
- 24³ phantom with 2 bins × 10 frames × 4 encodes, containing:
  - a cardiac block;
  - a block that moves with breathing only;
  - a vessel with steady flow at 0.4 m/s (VENC 1).
- Result:
  - background locked: 100 % over the cardiac cycle, 98 % over breathing;
  - every moving or flowing structure left free;
  - the breathing-only block locked over the cardiac cycle;
  - cardiac deviations unchanged after the lock (to 1e-5).

## 4. Divergence-free wavelets

Deriaz & Perrier (J. Turbulence 2006), Ong et al. (MRM 2015). Filters and linear combinations follow BART v0.4.04 (src/dfwavelet,
BSD). Our transform is periodic: critically sampled, with exact reconstruction for any even length.

- Component v_i: the CDF 3.1 bank along axis i and CDF 2.2 along the other axes, divided by the voxel size of axis i.
  The 1D pairs satisfy φ1′ = Δφ0 and ψ1′ = −4ψ0.
- Per level and detail subband ε, with H = {a : ε_a = 1}, Δ the backward difference inside the subband, and
  divergence d = −4 Σ_{a∈H} c_a + Σ_{a∉H} Δ_a c_a:

| \|H\| | df1, df2 | n ∝ d |
|---|---|---|
| 1 (pivot p, others q1, q2) | c_q1, c_q2 | c_p − ¼(Δc_q1 + Δc_q2) |
| 2 (p1 < p2, other q) | (c_p1 − c_p2)/2, c_q | (c_p1 + c_p2)/2 − ⅛Δc_q |
| 3 | (−2c_x + c_y + c_z)/3, (−c_x + 2c_y − c_z)/3 | (c_x + c_y + c_z)/3 |

- **Sign of the n row:** BART's table has +¼ / +⅛. With these filters, and in our periodic port of BART's indexing,
  only the minus sign separates exactly.
  - Test `divergence_free_field_has_no_n_coefficients`: a discrete curl field, a backward-difference curl of a smooth
    periodic potential.
  - With the minus sign, n/df ≈ 1e-7 in every subband and level (f32 rounding).
  - With +¼ it is ≈ 0.3–0.8.
  - The hand derivation from the synthesis filters agrees: Σ(hir1[m] − hir1[m−1]) · 2 = −4 hir0.
- **Shrinkage:**
  - Soft thresholds per level × subband × output, in units of the noise amplitude of that output: the std of the
    transform of unit white noise, estimated once per padded size.
  - Velocity noise scales as σ_v ∝ 1/|m|, so the shrinkage acts on u = c · w̄ / (s₀ · amp), where:
    - w̄ is the block-mean magnitude over the coefficient's support (a magnitude pyramid);
    - s₀ is the MAD level of u in the finest diagonal subband, over signal-bearing coefficients
      (w̄ ≥ 0.2 · p95).
  - Threshold rule:
    - SureShrink per subband (default `--dfw sure`; hybrid with the universal threshold for sparse subbands);
    - or fixed multiples (`--dfw fixed:TDF:TN`).
    - df and n get separate factors.
  - Scale-equivariant; cycle spinning per iteration.
- **Complex model:**
  - Decode v per frame (FlowDecoder, hierarchical unwrapping), shrink, then re-encode as a phase change of each encode:
    x_e ← x_e exp(i(φ_e(v′) − φ_e(v))).
  - Magnitude and background phase are untouched: this is the model-consistent change of v.
  - Mixing: x ← x + β(DFW(x) − x) (`--dfw-mix`).
- **Velocity model:** DFW on every v node, with weight |z̄|.
- **Open:** the image grid is GNL-apparent, so ∇· in apparent coordinates differs from the true divergence by the
  Jacobian (a small, smooth error); a Jacobian-weighted version is future work.

Tests:
- `sepwt::perfect_reconstruction_all_banks`: db4 and CDF, lanes 1 and 2, PR < 2e-5; db4 orthonormal.
- `dfwavelet::{perfect_reconstruction, divergence_free_field_has_no_n_coefficients,
  shrinkage_denoises_a_divergence_free_field_and_is_scale_equivariant}`.
  - The last one checks: SURE more than halves the error of a noisy curl field, and it is scale-equivariant to 1e-4.

## 5. Performance

Shared engine `sepwt`: f32, in-place Mallat layout, periodic.
- Each pass sees a line of n samples whose elements are contiguous vectors (x runs for the y and z passes; for the x
  pass the plane is transposed through scratch).
- The multiply-accumulate over the vector is AVX2+FMA, chosen at run time.
- The x and y passes of a level run per z plane (the plane stays in cache), the z pass per y row.
- Planes and rows spread over all threads, so a single image uses every core.

Benchmark (`tests/wavelet_bench.rs`, i9-13900K, 32 threads, 3 levels):

| case | new | previous (f64, one image per thread) |
|---|---|---|
| db4, 128×128×96 × 8 complex images | 0.045 s (0.28 Gvox/s) | 0.30 s (6.5×) |
| db4, one 128×128×96 image | 0.006 s | 0.32 s (54×) |
| db4, 240×200×160 × 2 | 0.072 s | 1.55 s (21×) |
| DFW, SURE + weights, 128×128×96 field | 0.029 s | — |
| DFW, 240×200×160 field | 0.11 s | — |

- The SURE searches of the 63 (level, subband, output) bands run in parallel, from ≤ 2^15 strided samples each.
- Still CPU and memory-bound (~20 GB/s effective). At 1 mm full stacks, 40 frames × 7 encodes × 36 M voxels are
  10 G voxel-images: about 50 s per denoise on the CPU.
- The next step is a GPU path inside the set streaming (each frame's images are on the device for the gradient
  anyway); the engine's line structure maps directly onto a kernel per pass.

## 6. Experiments

1. Code-path smoke on sim_test (2.5 mm, 20 frames, zero truth): runs, timings, maps.
2. factory_arms 2 mm 10-min scan, 2 respiratory bins, oracle settings, 20 iterations:
   - LLR;
   - LLR + lock (quad 20);
   - LLR + lock (group 1.5);
   - LLR + lock + DFW SURE.
   Metrics:
   - lumen velocity RMSE, per frame;
   - lumen voxels locked (must be ≈ 0);
   - static fractions;
   - visual: the static map against the anatomy.
3. Tissue-motion truth: compare the map with the phantom's displacement fields (motion/cardiac_motion.hv,
   resp_motion.hv) sampled on the recon grid — to implement.
4. Intracranial phantom (when available): expect > 90 % static.

## 7. Per-axis lock on respiratory × cardiac knots (2026-11-18)

Code: `hasty_recon::knotlock` (tests/knotlock.rs). The frame-stack lock (§2–§3) is unchanged.

**Setting.** A voxel's signal on the 2-D knot basis (temporal_operator.md §16, §21) is
x(t) = Σ_{k,b} ψ_k(r(t)) φ_b(c(t)) C[k, b]: respiratory knot k (linear hats at bellows quantiles, or bins), cardiac
knot b (periodic B-splines), image κ = k · mc + b. Both bases are partitions of unity. So "constant along an axis" means
"equal coefficients along that axis":
- static over the cardiac cycle (moves with breathing): C[k, b] = a_k (subspace S_c);
- static over breathing (moves with the heart): C[k, b] = d_b (S_r);
- fully static: C = m (S_0).

The subspaces are fixed by the basis. The metric decides the decomposition and therefore the prox.

**Which metric.**
- The data term's Hessian on knot stacks is ≈ G ⊗ AᴴA_static, with G_κκ′ = mean_j w_κ(j) w_κ′(j) the temporal Gram
  under the sample times (knotsolve §19). P²nP's forward step x − αP∇f is averaged in the P⁻¹ = G ⊗ C norm.
- The composition stays averaged only if the prior step is firmly non-expansive (FNE) in that same norm. A prox in
  the metric M, argmin R(u) + ½‖u − y‖²_M, is FNE in M. A Euclidean prox is FNE only in the Euclidean norm.
- In G, the orthogonal projection onto S_0 is the average weighted by each knot's **sample share** ω = G1/1ᵀG1 (the
  mean basis weight of the knot over the samples).
  - It is also the BLUE of the static value when the coefficient noise covariance is ∝ G⁻¹ (least-squares and
    knot-preconditioned iterates).
- The plain coefficient average is the special case of equal shares. Uniform periodic cardiac knots have equal
  shares: there the two coincide.
- With unequal shares, the plain average is an oblique projection in G, with G-operator norm √(ω_Eᵀ G_r⁻¹ ω_E) ≥ 1
  (Cauchy–Schwarz; = 1 only for equal shares).
- On sim_flow_10min (6 linear knots at the bellows quantiles, 16 cubic cardiac knots, from physio.npy):
  - respiratory shares 0.091 / 0.217 / 0.192 / 0.195 / 0.256 / 0.049; cardiac shares 0.0625 ± 1e-4;
  - plain-average breathing lock: G-operator norm **1.58**. It expands, so the plain average is wrong for breathing.
- Projections onto S_c and S_r in the full G do not commute when breathing and heart phase are sampled dependently.
  The four-piece decomposition below is then not orthogonal, and the additive prox has no closed form. The lock
  therefore uses the separable **G_s = G_r ⊗ G_c**:
  - G_r and G_c are the *exact* marginals of G: partial sums over the other axis, by partition of unity.
  - In G_s, P_c, P_r and P_0 are share-weighted averages along each axis, and they commute.
  - G and G_s differ by the dependence between breathing state and heart phase in the sampling. On sim_flow_10min:
    - ‖G − G_s‖_F / ‖G‖_F = 0.048;
    - √(λ_max / λ_min) of G_s⁻¹G = 1.12 with all encodes pooled, 1.36 for one encode's 7.6 k readouts. This is
      mostly finite-sample noise: 53 k readouts against 96² Gram entries.
  - A G_s-FNE map has G-Lipschitz constant ≤ that bound.
  - **Recommendation:** build the knot preconditioner from G_s (`KnotBasis::gs_inverse` → `KnotPrecond::new`). The
    solver metric is then G_s ⊗ C, in which the lock is exactly FNE (temporal part), at a cost of cond(G_s⁻¹G) ≤ 1.85
    in the preconditioned temporal operator.
- Spatial part: as in tlock, the lock acts per voxel and C is the circulant metric. Exactness would need C ≈ scalar,
  or spatially constant weights. This is unchanged from the frame lock.

**Decomposition and prox.** Per voxel (and per encode, or jointly over the three velocity components), with
c̄_c[k] = Σ_b ω_c[b] C[k, b], c̄_r[b] = Σ_k ω_r[k] C[k, b] and m = Σ ω_r ω_c C:
- the four pieces, G_s-orthogonal:
  - m;
  - R_k = c̄_c[k] − m: breathing only;
  - Cd_b = c̄_r[b] − m: heart only;
  - I = C − c̄_c − c̄_r + m: interaction.
- Their noise dimensions under a covariance model M are n(Π) = tr(G_s Π M). For M = G_s⁻¹ they are the plain
  dimensions: 1, kr − 1, mc − 1, (kr − 1)(mc − 1).
- **quadratic:** the prox of (μ/2)[w_c‖(I − P_c)x‖² + w_r‖(P_c − P_0)x‖² + w_x‖(I − P_c)(I − P_r)x‖²]_{G_s}:
  - R ÷ (1 + μ w_r);
  - Cd ÷ (1 + μ w_c);
  - I ÷ (1 + μ(w_c + w_x)).
- **group:** block soft thresholds in the G_s norm, τ = μ w s √n:
  - R with w_r;
  - Cd with w_c;
  - I with w_c + w_x.

  For velocities the groups span the three components.
- **Coupling.**
  - `Nested` (= tlock): w_x = 0, and in group mode the cardiac group is R⊥ = Cd + I, per bin when G_r is diagonal.
  - `Additive` (default): w_x locks the interaction, so "static over breathing" makes the whole cardiac curve
    identical in every breathing state, not only the bin means.
- Every map is a G_s-self-adjoint scaling with factors in [0, 1], or a block soft threshold on orthogonal pieces. Both
  are FNE in G_s.
- The mean m is never changed, so steady flow keeps its mean velocity.

**Map: the noise is coloured differently on knots.**
- The second difference of tlock assumes frame-independent errors. On B-spline coefficients the noise covariance is
  ≈ G⁻¹, which for 16 cubic cardiac knots is strongly high-pass (G_c has condition ≈ 18.5, §13 of
  temporal_operator.md).
- Evaluating at frame centres first would make it low-pass instead (the frames are rank K, smoothed). It costs F > K
  images and needs the same correction, so the map stays on the coefficients and normalises exactly:
  - score² = (piece energy in G_s, blurred) / (s² n(Π));
  - s² from the cardiac second differences D divided by Σ_terms (e_k ⊗ d_b)ᵀ M (e_k ⊗ d_b) (periodic wrap; tlock's
    no-wrap for the frame basis);
  - the detection-power "per-frame" noise is σ_t² = s² tr(G_s M) (the time-averaged noise of the evaluated signal);
  - the respiratory-mean detection uses tr(G_s M) / n(P_c) "frames";
  - air: |m̄| < 3 s √n(P_0);
  - the flow noise uses n(P_0).

  With G ∝ I these reduce exactly to tlock.
- **The colour model matters.** For lin6 × cubic16, Σ_terms dᵀMd is 5.2× larger under M = G_s⁻¹ than under white
  coefficients at equal piece noise. A wrong model scales the static score by about 2.2: it measured s ≈ 2.2 and
  w_c ≈ 0.5 on static velocity-model tissue whose pseudo-image noise is white.
- **`NoiseModel::Auto` (default for knots).** M = G_s⁻¹ + t·I.
  - t is fitted per coefficient set from ρ = median(interaction energy) / median(D). Both are pure noise in static
    voxels, the majority.
  - n(Π) is linear in M, so this is one scalar equation.
  - Measured in the tests: LS-coloured noise → t = 0 (check ratio 0.94–1.08); white noise → t ≈ 38 (blurred) / 200
    (raw), static scores 1.01–1.06.
  - Noise smoother than white (e.g. after a temporal low-rank prior) is outside the family: t clamps at White and
    the static scores come out inflated. That locks less, which is the safe side.
- Weights, as tlock:
  - w_c = ramp(max(s_c, s_v)) · conf;
  - w_r = ramp(max(s_r, s_v)) · conf_r;
  - w_x = ramp(max(s_r, s_int, s_v)) · min(conf_r, conf);
  - min filter over `dilate_mm`.
- s_v (flow of the mean images) still unlocks both axes. A flowing voxel is never locked, even when its flow does not
  change with breathing.

**Velocity model (z_κ, v_κ).**
- `estimate_vel` scores the pseudo-images z_κ exp(iφ_e(v_κ)). They are streamed per encode, so no 7 × K stack is
  held.
- `sigma_raw` is z's own noise level and `sigma_v` is v's (both block medians of their raw second differences, each
  with its own fitted t).
- `apply_z` is the image lock. `apply_v` locks the 3-vector's temporal deviations: constant velocity along locked axes,
  mean kept.
- The static-tissue anchor v ≈ 0 stays in vobj; the lock does not duplicate it.

**Tests** (tests/knotlock.rs, CPU, all pass):
- `metric_shares_and_noise_dimensions`: LS noise dimensions exact to 1e-8; plain-average G-norm 1.009 on a synthetic
  breathing distribution (1.58 on the real one); `gs_inverse` exact; a breathing-dependent heart-phase density
  raises nonsep 0.011 → 0.35 (metric bound 2.4).
- `prox_is_firmly_nonexpansive_in_the_gs_metric`: ‖Dx − Dy‖² ≤ ⟨Dx − Dy, x − y⟩ in G_s to 1e-5 relative, for every
  mode (quadratic / group × nested / additive × μ 0.7 / 20, random or binary weights), on lin6 × cubic16 and on bins.
- `fixed_points_and_group_threshold`:
  - a static voxel is unchanged (every mode);
  - a voxel constant over the cardiac cycle is unchanged under w_c alone;
  - a voxel constant over breathing is unchanged under w_r = w_x = 1 (additive);
  - a hard additive breathing lock makes a dynamic voxel constant along breathing;
  - group: a heart-only voxel's deviation drops by exactly τ, and the mean is unchanged.
- `map_classifies_static_cardiac_resp_dynamic_voxels`:
  - Setup: 24³, 6 × 16 knots, 4 encodes. Noise: LS-coloured, white plus spatially correlated "aliasing" with a ×3
    amplitude gradient over the FOV.
  - Background: w_c 0.995, w_r 0.984, w_x 0.980; median static s_c 0.95.
  - heart-only: (w_c, w_r, w_x) = (0, 1, 1);
  - breathing-only: (1, 0, ·);
  - both: (0, 0, 0);
  - cardiac curve modulated by breathing: w_x = 0 (s_int 7.5);
  - steady flow: (0, 0, 0) (s_v 345).
  - A hard lock freezes the background and removes the heart-only voxel's breathing spread (0.22 → 3e-7) while it
    keeps its cardiac curve.
- `detection_power_and_noise_model`:
  - Variation ≥ detect_ok (std/mean 0.1) is never locked, and 0.05 is detected too (s ≈ 10).
  - Static strips: w 1.00, s 1.01 (LS) / 1.06 (white, via Auto).
  - Under heavy noise (σ_t 0.5): tissue w 0 (undetermined 83 %), air w > 0.9.
- `frames_basis_reproduces_tlock` (tlock's 2 × 10 × 4 phantom, `KnotBasis::frames`, Nested):
  - lock identical to 5e-7 (quadratic μ 20 / 1e6; group μ 3 / 0.5);
  - s_flow and σ_raw/√K identical to 3e-7;
  - s_card within 0.25 %, s_resp within 9 % at static voxels. This comes from tlock's f32 Σx² − |Σx|²/n cancellation:
    where s > 3 they agree to 2e-4 / 3e-3;
  - weights within 0.009.
- `velocity_model_lock`:
  - static tissue w 1.0;
  - pulsatile and steady flow unlocked (s_v);
  - fitted t: z 2.2e3, v 1.1e2 (white, as injected);
  - v lock: mean velocity kept to 1.5e-8, static spread 0.078 → 5e-8, a breathing-locked pulsatile curve keeps its
    cardiac range (0.51 → 0.50) and loses its breathing spread (0.081 → 9e-8);
  - the z group lock makes static z exactly constant.

**Cost (host, rayon; measured 64³ × 96 knots on 10 threads, scaled to 2 mm: 4.17 M voxels × 96 knots × 7 encodes):**

| step | time | memory |
|---|---|---|
| lock, quadratic | ≈ 2.0 s (≈ 19 GB/s, memory-bound) | 2048-voxel chunk buffers |
| lock, group | ≈ 4.2 s | 2048-voxel chunk buffers |
| map (blur-dominated) | ≈ 23 s | one encode's K blurred images (3.2 GB) + ≈ 0.5 GB of per-voxel accumulators and means |
| velocity-model lock (z + v) | ≈ 1 s (estimate) | — |
| velocity-model map | ≈ map time + one sincos per voxel, knot and encode (estimate, not measured) | — |

The map runs only at refreshes. A GPU port is straightforward (per-voxel K-vector work), but it is not needed at this
cost.

**Open issues.**
- The full-G vs G_s mismatch (bound 1.12–1.36) remains unless the preconditioner uses `gs_inverse`. That needs a
  measurement of the preconditioner quality with G_s against G.
- The model check assumes a static majority (each ρ is a global median). In a thoracic box mostly made of moving
  tissue, inspect `model_check` / `noise_mix` in the log.
- Temporal low-rank priors (LLR across knots) make the noise smoother than white. Auto then clamps at White, which is
  conservative. An empirical per-piece calibration would be the next step if real iterates show static scores ≫ 1.
- The s_v rule unlocks flowing voxels on both axes. Locking flow over breathing (the v-curve the same in every
  breathing state) would need a flow-specific breathing criterion; not done.
- Group thresholds for v use v's own second-difference noise. Where |z| is small, v is noise and gets locked if the
  map says static. That is harmless (vobj's anchor handles v there), but it is not velocity-SNR weighted.

## Sources

- Deriaz & Perrier, Divergence-free and curl-free wavelets in 2D and 3D, application to turbulent flows, J. Turbulence 7 (2006); Orthogonal Helmholtz decomposition in arbitrary dimension using divergence-free and curl-free wavelets, ACHA 26 (2009)
- [Ong et al., Robust 4D flow denoising using divergence-free wavelet transform, MRM 2015](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.25176)
- [BART v0.4.04 src/dfwavelet (BSD)](https://github.com/mrirecon/bart/tree/v0.4.04/src/dfwavelet)
- Donoho & Johnstone, Adapting to unknown smoothness via wavelet shrinkage (SureShrink), JASA 90 (1995)
- Pedersen et al., k-t PCA: temporally constrained k-t BLAST reconstruction using principal component analysis, MRM 62 (2009)
- Otazo et al., Combination of compressed sensing and parallel imaging for highly accelerated first-pass cardiac perfusion MRI (k-t SPARSE-SENSE), MRM 64 (2010)
