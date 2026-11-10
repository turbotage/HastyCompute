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

## Sources

- Deriaz & Perrier, Divergence-free and curl-free wavelets in 2D and 3D, application to turbulent flows, J. Turbulence 7 (2006); Orthogonal Helmholtz decomposition in arbitrary dimension using divergence-free and curl-free wavelets, ACHA 26 (2009)
- [Ong et al., Robust 4D flow denoising using divergence-free wavelet transform, MRM 2015](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.25176)
- [BART v0.4.04 src/dfwavelet (BSD)](https://github.com/mrirecon/bart/tree/v0.4.04/src/dfwavelet)
- Donoho & Johnstone, Adapting to unknown smoothness via wavelet shrinkage (SureShrink), JASA 90 (1995)
- Pedersen et al., k-t PCA: temporally constrained k-t BLAST reconstruction using principal component analysis, MRM 62 (2009)
- Otazo et al., Combination of compressed sensing and parallel imaging for highly accelerated first-pass cardiac perfusion MRI (k-t SPARSE-SENSE), MRM 64 (2010)
