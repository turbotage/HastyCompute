# Model-based B0 / R2* / fat with motion: what to reconstruct and which priors (research note, 2026-11-06)

Context: the free-breathing multi-echo train (`hasty-seq` multiecho, research/plan_girf_b0_qfit.md WP3). First
results with per-echo images + LLR + voxel fits (`recon_b0`) showed the core problem: each echo image is
undersampled ≈ 65× at 2.5 mm in 30 s (≈ 16× at 5 mm), while all echoes *together* are close to Nyquist. Per-echo
images are the wrong unknowns. User direction (2026-11-06):
- B0 need not be an image during the reconstruction: a polynomial / piecewise-polynomial basis, reconstruct the
  coefficients;
- T2* (R2*) has far more structure, is interesting in itself → reconstruct at high resolution; it is spatially sparse;
- both change very little between cardiac and respiratory states (low information) — how to capture that change is
  the hard question;
- model-based: use all information (every lobe has its own direction and timing, …), dynamic preconditioning;
- respiratory *and* cardiac resolved; the scan may be 2–3 min (the high-resolution R2* map is worth it).

## 1. Information budget (2.5 min scan, the current 2 mm train)

- ≈ 6 200 TRs × 24 lobes × ≈ 320 samples ≈ 48 M samples per coil, × 12 virtual coils ≈ 0.6 G complex numbers.
- Unknowns at 1.5 mm over 400 × 400 × 160 mm (7.6 M voxels): ρ_w, ρ_f complex + R2* = 5 real per voxel ≈ 38 M.
  B0 as 2 cm cubic B-splines (+ shim harmonics) ≈ 10⁴; motion ≈ 10²–10³.
- So the data count is ≈ 30× the unknowns *if* the unknowns are tissue properties shared by all echoes and all
  motion states. What breaks it is only the parametrisation: per-echo images (50×) × per-state images (×20–40) make
  the problem 10³× underdetermined. Every prior below is a way to keep the unknowns at "one property map set".
- B0 at ~10⁴ coefficients is overdetermined by ~10⁴×: its precision is CRLB-limited, essentially free. R2* and ρ at
  high resolution are where priors matter (high k is sampled sparsely, but at all echo times).

## 2. Unknowns: tissue properties in a reference pose

| quantity | representation | why |
|---|---|---|
| ρ_w, ρ_f (complex) | images, high resolution | linear in the model → eliminated per iteration (VarPro / inner linear solve) |
| R2*_tissue | image, high resolution | intrinsic tissue property; moves with the tissue |
| B0 / ω | coefficients: (A) cubic B-splines (1–2 cm knots, refined near air) + spherical harmonics ≤ order 3, or (B) susceptibility χ on a coarse grid + harmonic background | smooth except at χ interfaces |
| motion | φ_s = respiratory ∘ cardiac, a few displacement modes | the "low-parameter diffeomorphism" |
| drift | global frequency δ(t), a few coefficients | heating / shim drift over minutes |

**B0 option B — susceptibility as the parameter.** ω(x) = γB0 (d ∗ χ)(x) + harmonic background, d the dipole kernel
(FFT). χ is piecewise constant (air / lung / tissue / fat / bone) → a TV prior, and air regions come from the
magnitude (no signal = air) — "as much information as possible". The decisive advantage is dynamics: χ moves with
the tissue, so the field of motion state s is d ∗ (χ_ref ∘ φ_s⁻¹): the lung-volume field change with breathing and
the heart-motion field change are *predicted*, not extra unknowns. Option A must add a low-order δω_s per state.
Caveat (inverse crime): the phantom's field is itself a dipole field of its χ map → the phantom must use a finer χ,
χ microstructure / anisotropy terms and a different background, and option A must be tested too.

**R2* physics.** R2*_eff(x, τ) = R2*_tissue(x) + the macroscopic intra-voxel dephasing of the field gradient across
the voxel (sinc of γ ∇B0 · Δx τ / 2 per axis, Yablonskiy), which the B0 model *predicts*. Near lungs this term
dominates; modelling it keeps R2*_tissue a clean, sparse tissue map and makes R2* resolution-consistent.

## 3. Motion: how the small change between states is captured

- No per-state images. Every sample is explained by the reference maps seen through the motion state at its own
  acquisition time t_n: x_s(q) = x_ref(φ_{s(t_n)}⁻¹(q)).
- Physiology (user, 2026-11-06): real scans record a finger pulse oximeter (PPG, cardiac; delayed by the pulse transit
  time) and respiratory bellows (abdominal excursion; lag, nonlinearity, drift). These are the default drivers;
  FID self-gating (`hasty_recon::selfgate`) only as a free extra (it would fail intracranially). The simulator writes
  realistic versions (`physio.npy`). Since bellows ≠ the thoracic motion state, the model maps the measured signals to
  the motion coefficients (r(t) = h(bellows, d/dt) with a few parameters, PPG phase + an estimated transit delay).
- Respiratory: x = X + Σ_k r_k(t) S_k(X), K = 1–2 (r and ṙ for hysteresis), r(t) from the bellows (or self-gating),
  S_k smooth B-spline fields (`hasty_recon::motion`), refined jointly with the maps.
- Cardiac: x = X + Σ_k θ_k(t_c) C_k(X), K ≈ 4–6 modes, θ_k periodic in cardiac phase (3–5 Fourier harmonics), ECG
  phase; initial C_k from cardiac-binned low-resolution reconstructions + registration + PCA.
- Composition φ = resp ∘ card (cardiac in the moving thorax frame). Warp operator: `hasty_recon::moco` (exact
  adjoint), applied per (state-chunk) of samples.
- What a diffeomorphism cannot express: blood signal changes with inflow / velocity (SPGR inflow, the flow-dependent
  RF-spoiling magnitude) and with cardiac phase → a low-rank residual on ρ restricted to blood (rank 2–3 over
  states); optionally χ of blood vs oxygenation (negligible here).
- Field per state: option B gives it from the warped χ; option A: warped spline field + δω_s (low order).

## 4. Forward model: every sample with its own physics

y_{c,n} = Σ_x S_c(x) [ρ_w + ρ_f c(τ_n)](φ_s⁻¹x) e^{−(R2*_eff + iω_s)(x) τ_n} e^{−i(k_n·u(x) + w_n·m(x))} + noise
- k_n: the played (GIRF) trajectory; τ_n: exact time since excitation (no echo binning); w_n: concomitant
  coefficients of that sample; u: GNL-warped position; s = s(t_n): motion state; whitened coils.
- Evaluation: echo windows (|τ − TE_e| ≤ T_lobe/2): e^{−zτ} = e^{−z TE_e} × a smooth function of δ = τ − TE_e;
  per window a small time segmentation in δ (L ≈ 3–6 terms incl. fat peaks; `hasty-nufft` SegNufft machinery with a
  complex rate) → forward cost ≈ L × (all samples) gridding + L × 50 FFTs. Stage 1 uses L = 1 (window-constant τ,
  as now) to get the parametrisation and priors right; then L > 1 for exactness.

## 5. Priors (what we know a priori)

- ρ: complex; magnitude piecewise smooth (TV / wavelet), phase smooth (receive phase; water and fat share it at
  t = 0) → separate magnitude / phase regularisation; later the learned denoiser (P²nP) on ρ.
- R2*_tissue: piecewise smooth with edges shared with ρ → structure-guided TV (directional TV aligned with ∇|ρ|) or
  joint TV over (|ρ_w|, |ρ_f|, R2*); R2* ≥ 0; sparse in a wavelet basis.
- B0: option A — knot spacing + small bending energy; option B — TV on χ, air mask from |ρ|, harmonic background.
- Motion: smoothness, invertibility (Jacobian > 0), periodicity of the cardiac part, few modes.
- Statistics: whitened complex data, the likelihood is the data term (no magnitude fits); priors as penalties or
  proximal / denoising steps — the same MLE-consistency rules as `hasty-qfit` (scaling is a reparametrisation).

## 6. Optimisation

- Variables in scaled units (amplitudes ~1, R2* / r2_max, ω coefficients / ω_max): the qfit lesson (damping and
  quasi-Newton memory in scaled variables, same solution).
- Outer: quasi-Newton on (R2*, B0 coefficients, motion) with ρ eliminated (VarPro: ρ(θ) = argmin, inner CG / P²nP
  with the circulant preconditioner and ZMSHR1 — the dynamic preconditioning — warm-started), or a joint L-BFGS
  with H₀ = circulant M on the ρ blocks and diagonal scaling elsewhere. Priors as smooth penalties (Huber-TV) first,
  proximal / denoiser steps later.
- Initialisation ladder: low resolution (k ≤ 1/(2·8 mm)), where every echo is well sampled → echo images → GPU
  profile-likelihood fits (`waterfat`, swaps and ω wraps resolved there) → spline / χ fit of ω → full resolution with
  the B0 model fixed-then-free. The smooth B0 model is what prevents swaps at high resolution.

## 7. Evaluation

- Truth from the simulator at the reference pose (field, R2*, fat fraction, water / fat PD: `simphantom` now writes
  them); CRLB per voxel (`hasty_qfit::stats`) as the efficiency reference.
- Ablations: per-echo LLR (current `recon_b0`), binned, model-based static, + motion, A vs B for B0, L = 1 vs exact τ.
- Metrics: ω error (Hz) by region (heart, lung boundary, liver dome), R2* error and edge-spread (resolution), FF,
  epicardial fat / coronary region (the coronary goal), motion field error vs the phantom's.

## 8. Sequence and phantom consequences

- Scan 2–3 min, free breathing, ECG; golden-means lobe directions give every (resp, cardiac) state uniform
  coverage. Because B0 is low-dimensional, the lobes can target the R2* resolution (1.5–2 mm) — re-run `echodesign`
  with the B0 parameters shared (its CRLB then changes).
- Development vs final (user, 2026-11-06): method research runs on fast development phantoms (`prep_sim --src-step 2`:
  8× fewer tissue sources, short scans; the six-peak fat spectrum is now evaluated per sample in the kernel, one source
  per fat voxel), final evaluations on the full phantom (exact direct sum, later richer physics such as LBM flow);
  32 coils throughout. The cost is the direct sum (sources × samples × coils), not the coil maps.
- Phantom: breathing- and heart-dependent field (recompute the dipole field of the warped χ at a few states,
  interpolate — finer than the reconstruction's χ to avoid the inverse crime); blood inflow; 16-coil option for
  2–3 min simulations (cost ∝ coils; the current 32-coil, 30 s simulation takes 1 h).

## 9. Stages

- S0 (done 2026-11-06): coil near-field fix, reference-pose truth maps, 30 s static development scan.
- S1: static model-based reconstruction, option A B0, window-constant τ (L = 1): unknowns ρ_w, ρ_f, R2*, spline
  coefficients; joint scaled L-BFGS with circulant H₀; Huber-TV priors; init from the low-resolution ladder.
- S2: exact τ (window segmentation, L > 1, complex-rate segments) and the concomitant / GIRF terms per sample.
- S3: option B (χ) and the macroscopic R2* term; structure-guided R2* prior; learned ρ prior.
- S4: motion: respiratory (selfgate + motion fit, refined), cardiac modes; 2–3 min free-breathing + ECG simulation.
- S5: maps → the 4D/5D-flow reconstruction; evaluation on the coronary region.

## Status

- S1 (2026-11-06): `mbq_b0` (hasty-bench): ρ_w, ρ_f, R2* per voxel + cubic B-spline ω (20 mm knots), scaled L-BFGS,
  Huber-TV, softplus R2*, ρ-only warm start, resumable; analytic gradient = finite differences to ≤ 1e-4.
  Full phantom, 30 s breath-hold, 5 mm: ω |error| median 13.7 Hz, RMSE 98 Hz (per-echo LLR baseline: RMSE 276 Hz,
  37 % of voxels > 150 Hz off); R2* median +15 s⁻¹ (|·| p50 23), FF |·| p50 0.11. 2.5 mm (initialised from 5 mm,
  150 iterations): ω the same, R2* / FF slightly worse (not converged; window-constant τ). The data term is ≈ 100×
  the noise floor → model error dominates (window-constant τ, arm outside the 40 cm FOV, GNL not modelled).
- S2 part: `hasty_recon::multiterm` (universal band-limited interpolators over |f| ≤ 1050 Hz, R2* ≤ 500: worst-case
  error 0.15 / 2.5e-2 / 3.4e-3 / 4.3e-4 for L = 4 / 5 / 6 / 7 nodes per window) + `MultiTermSets`; `mbq_b0 --terms L`.
- S4 part (respiratory): `hasty_recon::moco::MocoTermSets` (warped node images, adjoint-tested), `mbq_b0 --moco
  --resp bellows --levels 16` (bellows-driven continuous breathing state, quantised), motion field from the binned
  LLR reconstruction + `motionfit`. Cardiac motion not yet modelled.
- Development phantom ph0001 `sim_b0_dev` (2 mm sources, 50 cm in-plane FOV): 30 s static in ≈ 4 min.

