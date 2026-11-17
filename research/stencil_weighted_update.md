# Weighted chirp coefficients: E2 on the real model, the kernel change, and the update scheme (2026-11-17)

Scope:
- E2 of `stencil_kernel_optimality.md` §7, on the paper's full-model GNL setting;
- the record / kernel change for a general per-sample polynomial p(χ);
- how the image-dependent pieces (chirp coefficients, energy-weighted segment interpolators and bases) are refitted as
  the reconstruction improves, with guarantees.

Labels: [CPU] = computed for this note (Rust CPU emulation, f64, `research/experiments/stencil/e2_weighted/`);
[model] = prediction from measured costs; [R] = recalled, not re-checked. No GPU was used. No rustlib code was changed.

---------------------------------------------------------------------------------------------------------------------

## 0. Answer

1. **E2 fails its pre-registered rule (≥ 2× at 2.89 T).**
   - At 2.89 T and 1.5 T, order 2 is limited by the degree-2 fit residual ε_j at GNL-warped positions, not by Taylor
     truncation. The exact-e^{−iψ} floor equals Taylor (4.81e-3 vs 4.79e-3), and weighted LS gains only 1.03–1.05×.
   - The earlier note's 2–10× was measured on pure quadratic chirps, i.e. ε_j = 0.
2. **Where truncation dominates, LS removes nearly all of it** [CPU, white fwd / adj]:

   | field, order | Taylor | LS | gain (object) |
   |---|---|---|---|
   | 0.55 T, order 2 | 3.13e-2 / 6.97e-2 | 2.41e-2 / 4.57e-2 | at the 2.39e-2 floor |
   | 0.55 T, order 3 | 2.92e-2 / 7.33e-2 | 1.44e-2 / 3.37e-2 | 2.0× |
   | 1.5 T, order 3 | 2.95e-3 / 6.47e-3 | 2.12e-3 / 4.52e-3 | 1.3× |
   | 2.89 T, order 3 | 1.15e-3 / 2.19e-3 | 1.07e-3 / 1.92e-3 | 1.04× |

   - **Chebyshev / minimax is worse than Taylor** in the forward direction at every field, because it fits the FOV
     edges. Centred Taylor gains ≤ 1.1×.
   - Rule: energy-weighted LS on the full target (it also absorbs part of ε_j), with a ridge toward Taylor.
   - The product ψ·c would give 2.2× (1.5 T) and 3.2× (0.55 T), but it needs φ^{(5,6)}: a separate kernel project.
3. **Kernel change: one 16-byte side record per sample.**
   - It holds 8 × f16: the complex deviations from Taylor of p₀, q = −p₁, ρ₂ = −2p₂, s₃ = ip₃.
   - `stage_weights` combines its five existing partial sums with these complex coefficients (§3).
   - Why not the record itself:
     - φ0 is a phase only, and |p₀| − 1 reaches 5.7e-3;
     - the record has no free slot;
     - folding into ΔC would make every refit rewrite the tile tables.
   - The f16 round trip and every-8th-ADC coefficients leave E2 unchanged.
   - Cost [model]: ≤ 3 % kernel time, +0.75 GB (+8 %) host tables at 1 mm.
4. **The weight is |x̄|²Σ_c|S_c|²**, not coil-combined |x|² alone. Only the forward error at the true image moves the
   fixed point, because the adjoint is the exact transpose (Proposition 1).
   - For frame-shared parts (segments, interpolators, monomial fit) the time average mean_t|x_t|² is the *exact*
     weight. The warm start's |mean_t x_t|² falls short of it, by Jensen, in the vessels and moving walls.
   - Per-|k|-band weights measured 1–30 % worse. Rejected.
5. **Refits are safe and cheap, and they matter little.**
   - Wrong weights keep most of the gain. They lose to Taylor by 1–4 % only where the gain is marginal (2.89 T, and
     1.5 T at order 2); Proposition 3 bounds this.
   - Scheme (§5–6):
     - when: geometric checkpoints or a PE_H stall, at most 4 refits;
     - weight: the smoothed, blended iterate;
     - accept only with a data-free minimax gain test plus ΔR ≥ 0.02;
     - each acceptance is a new operator epoch: drop the secant pair, refresh the lagged anchor, restart CG.
   - Convergence is perturbed KM with finitely many operator changes. No monotone fixed surrogate exists: fitting θ
     to the data would overfit.
6. **What to rebuild:**
   - per-sample coefficients: < 1 s GPU, records untouched;
   - interpolators: < 0.1 s per encode via block Grams, which are linear in the weight;
   - the monomial fit: 2–4 % gain, but it rewrites the records, so once or never;
   - the segment basis: an exact cheap decision, at most one rebuild.
   - G4 (§8) tests whether one warm-start fit already suffices, which E2 suggests.

---------------------------------------------------------------------------------------------------------------------

## 1. E2 setting (the paper's GNL table, reproduced exactly)

**Source.** The paper's table is `seg_physics_model_error_budget_cpu_order3` in
`rustlib/crates/hasty-bench/tests/seg_physics_direct.rs`. Setting:
- flow_thorax sequence, ideal GIRF;
- 24 × 24 × 16 voxels of 12 mm, oblique, FOV centre (30, −40, 100) mm;
- synthetic GNL, B0 ±150 Hz + fat shell, 4 coils;
- 2 sets × 6 readouts × 665 ADC;
- white image on an ellipsoid, random y for the adjoint;
- L = 20, ADC-aligned table.

**E2.** A verbatim copy of that file plus one test, `e2_weighted_chirp_polynomial_cpu`
(`research/experiments/stencil/e2_weighted/`: standalone crate, `run_e2.sh`, outputs in `out/`). No rustlib file
changed.
- The Taylor rows reproduce the original printout exactly (`out/baseline_order3.txt`).
- The polynomial is applied exactly, so the window error and f32 (the GPU `impl` part, ≤ 1e-4) are excluded.

**Methods.** Per sample:
- Taylor;
- centred Taylor (at the weighted mean of ψ_j);
- Chebyshev interpolation on [min, max] ψ_j (≈ minimax);
- weighted LS (§3.4), with a uniform or energy weight and a "chirp" target e^{−iχ} or a "full" target (the exact
  deviation).

**Bases.** Degree-2 fit: 1, ψ, ψ². Degree-4 fit (order 3): 1, ψ', ψ'², c, optionally ψ'c.

**Images.**
- The white image, whose weight is ≈ support × coil energy.
- A smooth object (dim body 0.15 + a bright off-centre blob, σ 30 mm + a phase ramp), forward only. Its LS weight is
  one of:
  - oracle (its own energy);
  - box (support);
  - shifted (blob 24 mm off: a wrong-state warm start);
  - blend (0.7 shifted + 0.3 box).

## 2. E2 results [CPU]

Set 0 of 2; set 1 agrees within ±10 %. Relative errors against the f64 direct sum, written white fwd / adj · object fwd.
LS rows are fitted with the image's own energy weight (white: support × coil energy; object: its smoothed |x|² × coil
energy).

**Degree-2 fit (production order 2; order 1 for reference)**

| method | 2.89 T | 1.5 T | 0.55 T |
|---|---|---|---|
| order 1 Taylor | 5.54e-3 / 1.02e-2 · 2.65e-3 | 1.39e-2 / 2.47e-2 · 6.83e-3 | 7.91e-2 / 1.56e-1 · 4.26e-2 |
| order 1 LS energy | 5.12e-3 / 9.13e-3 · 2.42e-3 | 1.11e-2 / 1.94e-2 · 5.62e-3 | 5.01e-2 / 1.03e-1 · 2.85e-2 |
| **order 2 Taylor (production; the paper's row)** | 4.79e-3 / 8.33e-3 · 2.26e-3 | 8.92e-3 / 1.50e-2 · 4.51e-3 | 3.13e-2 / 6.97e-2 · 1.82e-2 |
| centred Taylor | 4.79e-3 / 8.35e-3 · 2.26e-3 | 8.94e-3 / 1.50e-2 · 4.51e-3 | 2.81e-2 / 6.11e-2 · 1.72e-2 |
| Chebyshev on the range | 4.84e-3 / 8.45e-3 · 2.27e-3 | 9.40e-3 / 1.56e-2 · 4.84e-3 | 4.00e-2 / 5.55e-2 · 2.52e-2 |
| LS energy, chirp target e^{−iψ} | 4.81e-3 / 8.40e-3 · 2.26e-3 | 9.10e-3 / 1.52e-2 · 4.51e-3 | 2.52e-2 / 4.68e-2 · 1.33e-2 |
| LS uniform, full target | 4.70e-3 / 8.00e-3 | 8.93e-3 / 1.46e-2 | 2.67e-2 / 4.48e-2 |
| **LS energy, full target** | 4.62e-3 / 7.92e-3 · 2.24e-3 | 8.63e-3 / 1.45e-2 · 4.42e-3 | 2.41e-2 / 4.57e-2 · 1.32e-2 |
| + energy-weighted monomial fit | 4.53e-3 / 7.87e-3 · 2.15e-3 | 8.55e-3 / 1.44e-2 · 4.26e-3 | 2.39e-2 / 4.55e-2 · 1.27e-2 |
| floor: exact e^{−iψ} (fit residual ε_j left) | 4.81e-3 / 8.42e-3 · 2.26e-3 | 9.07e-3 / 1.54e-2 · 4.52e-3 | 2.39e-2 / 4.43e-2 · 1.28e-2 |

**Degree-4 fit (order 3)**

| method | 2.89 T | 1.5 T | 0.55 T |
|---|---|---|---|
| **Taylor (built; the paper's row)** | 1.15e-3 / 2.19e-3 · 5.61e-4 | 2.95e-3 / 6.47e-3 · 1.36e-3 | 2.92e-2 / 7.33e-2 · 1.52e-2 |
| LS, c-term kept at −i (no s₃) | 1.13e-3 / 2.11e-3 · 5.59e-4 | 2.55e-3 / 5.44e-3 · 1.24e-3 | 1.75e-2 / 4.07e-2 · 9.34e-3 |
| **LS energy [1, ψ, ψ², c]** | 1.07e-3 / 1.92e-3 · 5.38e-4 | 2.12e-3 / 4.52e-3 · 1.03e-3 | 1.44e-2 / 3.37e-2 · 7.68e-3 |
| same, uniform weight | 1.10e-3 / 1.92e-3 | 2.28e-3 / 4.52e-3 | 1.76e-2 / 3.24e-2 |
| same, through the f16 side record | 1.07e-3 / 1.92e-3 | 2.12e-3 / 4.52e-3 | 1.44e-2 / 3.37e-2 |
| Taylor + ψc product (needs φ^{(5,6)}) | 9.89e-4 / 1.70e-3 · 5.05e-4 | 1.91e-3 / 4.48e-3 · 9.15e-4 | 2.74e-2 / 6.93e-2 · 1.35e-2 |
| LS energy [1, ψ, ψ², c, ψc] | 9.56e-4 / 1.56e-3 · 4.94e-4 | 1.33e-3 / 2.60e-3 · 6.62e-4 | 9.14e-3 / 2.18e-2 · 5.04e-3 |
| floor: exact e^{−i(ψ+c)} | 9.81e-4 / 1.67e-3 · 5.01e-4 | 1.35e-3 / 2.60e-3 · 6.55e-4 | 3.88e-3 / 8.32e-3 · 2.12e-3 |
| floor: exact deviation (segments, L = 20) | 8.26e-4 / 1.03e-3 · 4.39e-4 | 8.86e-4 / 1.18e-3 · 4.47e-4 | 2.71e-3 / 4.11e-3 · 1.54e-3 |

**Weight sensitivity** (object forward, LS fitted with: oracle | box | 24 mm shifted | 0.7 shifted + 0.3 box; Taylor
for comparison):

| | 2.89 T | 1.5 T | 0.55 T |
|---|---|---|---|
| order 1 | 2.42 / 2.51 / 2.49 / 2.47 e-3 (T 2.65) | 5.62 / 6.19 / 5.86 / 5.90 e-3 (T 6.83) | 2.85 / 3.22 / 2.92 / 2.97 e-2 (T 4.26) |
| order 2 | 2.24 / **2.35** / **2.28** / 2.25 e-3 (T 2.26) | 4.42 / **4.55** / **4.62** / 4.51 e-3 (T 4.51) | 1.32 / 1.33 / 1.37 / 1.35 e-2 (T 1.82) |
| order 3 | 5.38 / 5.43 / 5.41 / 5.39 e-4 (T 5.61) | 1.03 / 1.06 / 1.06 / 1.05 e-3 (T 1.36) | 7.68 / 8.17 / 7.79 / 7.87 e-3 (T 15.2) |

Bold marks the cases worse than Taylor.

Spectral-local weight: each sample uses the energy of the image band around |k_j| (4 bands). It is worse than the
global |x|² weight everywhere: order 2 2.28 vs 2.24e-3 (2.89 T), 4.48 vs 4.42e-3 (1.5 T), 1.44 vs 1.32e-2 (0.55 T);
order 3 1.11 vs 1.03e-3 (1.5 T), 9.96 vs 7.68e-3 (0.55 T). Rejected.

**Does the incoherent proxy predict the measured error?** The proxy is ε(θ; w) = Σ_j Σ_ρ w |E_j|²
(Proposition 2), against the object forward error, both relative to Taylor and written proxy / measured.

| | LS oracle | LS box | LS shifted |
|---|---|---|---|
| 2.89 T order 2 | 0.980 / 0.992 | 1.004 / 1.042 | 1.001 / 1.011 |
| 2.89 T order 3 | 0.848 / 0.959 | 0.862 / 0.967 | 0.862 / 0.965 |
| 1.5 T order 2 | 0.976 / 0.979 | 0.999 / 1.009 | 0.998 / 1.024 |
| 1.5 T order 3 | 0.728 / 0.755 | 0.744 / 0.777 | 0.743 / 0.781 |
| 0.55 T order 2 | 0.724 / 0.723 | 0.733 / 0.729 | 0.744 / 0.751 |
| 0.55 T order 3 | 0.501 / 0.506 | 0.519 / 0.538 | 0.511 / 0.513 |

- The proxy orders every candidate pair correctly.
- It is within 3 % where the stencil error dominates.
- It is optimistic in two cases:
  - where the segment floor dilutes the gain (2.89 T order 3);
  - in the marginal case (2.89 T, order 2, box weight): +0.4 % predicted against +4 % measured.
- Acceptance rules must therefore demand a real predicted gain.

Readings:
1. **E2 fails** (1.04× / 1.07× at 2.89 T). "Never worse than Taylor" holds with the right weight, and fails by 1–4 %
   with a wrong one where the gain is marginal.
2. The full-target LS goes *below* the exact-e^{−iψ} floor, because it absorbs ε_j ∩ span{1, ψ, ψ²}. The chirp-target
   LS cannot.
3. **Order 3 + LS** gains 1.4× (1.5 T) and 2× (0.55 T) with today's derivative tables. The remaining gap is ψc and χ³.
4. At 0.55 T the result (1.4e-2 / 3.4e-2) stays above 1e-2. The paper's "construction (b)" conclusion stands.

## 3. What the kernel applies, and the exact change

### 3.1 The structure that is available

In kernel-argument units (`points`: ∂/∂k_a = −g_a ∂/∂z_a, ΔC scaled by g_a g_b, off-diagonals doubled) write
D_j := ΔC_j : ∇∇ and c_j(i∂) for the order-3 polynomial (`points_hi`). Since χ(i∂_k) e^{−ik·ρ} = χ(ρ) e^{−ik·ρ} and
χ_j(i∂) = −D_j for the quadratic part, **any** polynomial

  p_j(χ) = p₀ + p₁ ψ + p₂ ψ² + p₃ c   (ψ = ρᵀΔC_jρ, c = c_j(ρ); order 1: p₂ = p₃ = 0; order 2: p₃ = 0)

is applied by the stencil Γ_j = [p₀ − p₁ D_j + p₂ D_j² + p₃ c_j(i∂)] φ(κ − ·)|_{κ = k'_j}. The derivative tables, the
monomial algebra and the support are those of today; only the four complex numbers change. Today's kernel is the point
p = (1, −i, −½, −i).

`stage_weights` / `stage_weights_hi` build, per (sample, tile row) and per x-derivative order a, four real partial sums
from the 1-D tables and the record:
- R₀[a]: the value product (a = 0 only);
- Q_D[a]: the ΔC second-derivative sum (today it goes into Q, i.e. coefficient +i);
- E[a]: the fourth-derivative sum with the −½ built into e = −½ d d (today into R, coefficient +1);
- H₃[a], H₄[a]: the degree-3 / degree-4 sums of the order-3 term (today R += H₃, Q −= H₄).

With q := −p₁, ρ₂ := −2p₂ and s₃ := i p₃ (Taylor: q = i, ρ₂ = 1, s₃ = 1), the general weight is

  R[a] = Re p₀·R₀ + Re q·Q_D + Re ρ₂·E + Re s₃·H₃ + Im s₃·H₄
  Q[a] = Im p₀·R₀ + Im q·Q_D + Im ρ₂·E + Im s₃·H₃ − Re s₃·H₄.

Setting (p₀, q, ρ₂, s₃) = (1, i, 1, 1) reproduces the current code term by term.

### 3.2 Why not fold the coefficients into the existing record

- arg p₀ could go into φ0, but |p₀| ≠ 1 needs an amplitude. ||p₀| − 1| ≤ 3e-4 / 1e-3 / 5.7e-3 at 2.89 / 1.5 / 0.55 T,
  which is comparable to the gain.
- The 12-float record is full: slot 11 is the frame fraction whenever frames > 1.
- Scaling ΔC by Im q leaves 4 floats, but it rewrites the record, so a refit would touch the tile tables.
- The multi-image kernels' per-sample weight w_k(j) could carry p₀ alone, but not q, ρ₂ or s₃.

### 3.3 Proposal: a 16-byte side record, deviations from Taylor in f16

**Format.** Per sample one `uint4` = 8 × f16: δp₀ = p₀ − 1, δq = q − i, δρ₂ = ρ₂ − 1, δs₃ = s₃ − 1 (complex). Zero means
Taylor, so a zero buffer is today's operator, and a define `LSPOLY` compiles the path out.
- *Storage:* a separate buffer like order 3's `uw` (`SegPoints::hi`), so a refit never touches `rec` or the tile lists.
  At the 1 mm box: 46.9 M samples → 0.75 GB host, +8 % on the 9.3 GiB tables.
- *Shared memory:* +16 B per staged sample (`float4 pc[CHUNK]`).
- *Arithmetic:* the two lines of §3.1 replace `R[a] += …` / `Q[a] = …`. That is ≈ +20 / +30 FMA per (sample, row)
  task at order 2 / 3, against ≈ 40 for the px loop and ~10³ MACs per sample-channel. [model]: ≤ 3 % kernel time.
- *Order 1* needs only δp₀, δq: 8 B.
- *Precision:* f16 deviations; measured unchanged (§2).
- *Window error:* |ρ₂| ≲ 2, so the ∂⁴ window term is at most 2× its negligible Taylor value.

**Not proposed: ψ·c.** It is the largest gain at 1.5 / 0.55 T, but it contains x⁶ and so needs φ^{(5,6)} (NK 5 → 7):
a worse ES clamp and more staging. That is a separate kernel project.

### 3.4 How the host computes (p₀, q, ρ₂, s₃)

Per sample j, on probe points ρ_i of the support with weights w_i (§4):

  min_p Σ_i w_i |Σ_b p_b β_b(ρ_i) − t_j(ρ_i)|² + ε Σ_i w_i ‖p − p_T‖²,
  β = (1, ψ_j, ψ_j², c_j),   t_j = exp(−i[Δ_j(ρ) − φ0_j − b_j·ρ]),

with Δ_j = Σ_q Δw_jq m_q(r(ρ)) the exact deviation at the GNL-warped positions ("full target"; it includes ε_j).
- The basis is real, so the system is a 4 × 4 real SPD solve with a complex right-hand side.
- ε = 1e-6 rad² in radian units. Without the ridge, samples with ΔC_j ≈ 0 gave |r| ~ 10⁵ (first run).
- The coefficients depend on the sample only through Δw_j ∈ ℝ⁴, which is smooth along a readout. Fitting every 8th ADC
  sample and interpolating linearly is exact to three digits.
- Then q = −p₁, ρ₂ = −2p₂, s₃ = ip₃.

## 4. The weight: what it must be, and which image it comes from

**Proposition 1 (only the forward error at the true image matters).** The adjoint is the exact transpose of the
implemented forward B_θ, so every solver minimises Φ_θ = ½‖B_θx − y‖² + R(x). With y = Ax̄ + n, that is the problem
for the data B_θx̄ + n + e_θ, e_θ := (A − B_θ)x̄.
- By `model_robustness.md` eq. (1), ‖x*_θ − x̃_θ‖ ≤ κ_θ‖e_θ‖. Since κ_θ and x̃_θ change only at O(‖B_θ − B_θ'‖), θ
  enters to first order through ‖e_θ‖ alone.
- The paper's adjoint column (random y) is an operator test, not a fixed-point term: A^H is never applied.

**Proposition 2 (the weight).** Let E_j := p_j(χ_j) − t_j and v_c = S_c·(segment part)·x̄. Then
e_cj = Σ_ρ v_c e^{−ik_j·ρ} E_j.
- *Incoherent limit.* When E_j varies slowly with j compared with the k-space sampling, Parseval gives
  ‖e‖² ≈ Σ_jΣ_ρ |x̄|²Σ_c|S_c|²|E_j|². So **w = |x̄|²Σ_c|S_c|²**: coil energy × magnitude². It equals the coil-combined
  |x|² only for maps with Σ|S_c|² = 1, and it is `--seg-weight energy` (`segment_basis.md` §2).
  - The §2 proxy check confirms the ranking it predicts.
- *Coherent limit* (minimise |⟨v e^{−ikρ}, E_j⟩| per sample). Rejected:
  - it needs A x_k, and there is no exact A at scale;
  - it overfits, with 4 complex unknowns against C rows;
  - it makes θ a non-smooth function of x.
- *Spectral-local* (per-|k|-band energy). Measured 2–30 % worse than the global energy, and rejected. A likely reason,
  not tested: 92 % of the samples are in the outer band, whose weight sees only edges.

**Shared vs per-frame.** ε is linear in w, so a component shared by frames t has error Σ_t ε(θ; w_t) = ε(θ; Σ_t w_t).
- *Shared components* are D_l, a_l(τ) and the monomial fit. For them **mean_t|x_t|² is the exact weight.** The warm
  start's |mean_t x_t|² is smaller by Jensen exactly where the frames differ: vessel phase, moving walls. Refit once
  the frames are resolved.
- *Per-sample coefficients* use their own set's w_t. Knot-major sets take the nearest knot image. This is free.

**Smoothing and noise.** Use G_1cm * |x_k|²: E_j is smooth on the polynomial scale, so finer detail in w cannot
change p_j. Noise adds a uniform floor, a built-in blend toward the box weight that Proposition 3 makes harmless.

## 5. Updating during the iterations: what is guaranteed

### 5.1 No fixed monotone surrogate exists

The alternation "x-steps on Φ_θ, then θ ← Fit(w(x))" minimises the physics proxy ε(θ; w) = Σw|E_θ|² in its θ-step, not
Φ_θ.
- Making it block-coordinate descent on Φ would fit the operator to the data. There are 4 complex dof per sample, and
  p₀ alone fits every y_j. So that is excluded.
- Φ_θ then jumps at a refit by up to ±‖(B_θ' − B_θ)x‖‖r‖.

The merit function is the true objective Φ_A together with the bound of Proposition 1.

### 5.2 Guarantees

**Proposition 3 (weight mismatch).** For θ_w fitted with weight w and the true weight w̄,
  ε(θ_w; w̄) ≤ ε(θ_T; w̄) + (sup|E_{θ_w}|² + sup|E_{θ_T}|²)‖w̄ − w‖₁ − [ε(θ_T; w) − ε(θ_w; w)].
*Proof.* ε(θ; ·) is linear, with |ε(θ; w̄) − ε(θ; w)| ≤ sup|E_θ|²‖w̄ − w‖₁. LS optimality gives ε(θ_w; w) ≤ ε(θ_T; w),
and the ridge only tightens this. ∎

So LS beats Taylor at the true weight whenever its gain at the estimated weight exceeds sup|E|²‖w̄ − w‖₁.

The additive form is sharp only at high field: sup|E_T| reaches |χ|³/6 ≈ 0.4 at the 0.55 T corners. The
multiplicative form is more useful. If w̄/w ∈ [1/(1 + δ'), 1 + δ] on the support (the blend floor β bounds δ'), then

  ε(θ_w; w̄)/ε(θ_T; w̄) ≤ (1 + δ)(1 + δ') · ε(θ_w; w)/ε(θ_T; w).

That is, LS wins at w̄ if its predicted relative gain beats the weight's multiplicative uncertainty. Both forms say the
same thing E2 shows: wrong weights cost only where the gain is marginal (1–4 % losses at 2.89 T and at 1.5 T order 2),
and every weight wins 1.3–2× at 0.55 T and at order 3.

**Proposition 4 (convergence with refits).** Let T_θ, the P²nP step or a CG cycle, be averaged (KM) with a nonempty
fixed-point set, uniformly on bounded sets. If Σ_k sup_{‖x‖≤ρ}‖T_{θ_{k+1}}x − T_{θ_k}x‖ < ∞, the iterates converge to
Fix T_{θ_∞}.
- This is perturbed KM with summable errors (Combettes 2001 [R]).
- It covers finitely many refits, and geometric schedules once w_k converges.
- If T is contractive (factor ρ), x ↦ T_{Θ(x)}x is a contraction when ρ + γℓ < 1. Here γ ~ sup|p − p'| ~ 1e-2 and ℓ,
  the Lipschitz constant of x ↦ θ, is bounded by the ridge. The fixed point x* = x*_{Θ(x*)} is then unique.
- *Inexact-gradient view:* ∇Φ_θ − ∇Φ_A = −B_θ^H(A − B_θ)x + (B_θ − A)^H r. The iteration reaches an O(κ ε_floor)
  neighbourhood of the Φ_A solution, and a refit moves the fixed point by at most κ‖e_θ − e_θ'‖ (Schmidt et al. 2011 [R]).

### 5.3 Consistency at a refit (an "operator epoch")

- **ZMSHR1:** drop the secant pair, reusing the existing lagged-epoch logic (`lag_hist`).
- **Lagged control variate:** refresh the anchor (x_a, ∇f(x_a), f(x_a)).
- **CG** (warm start, knot solves): restart from r = B'^H(y − B'x) − λx.
- **λ_max:** re-estimate with 3 warm power iterations and keep α < 2/λ'_max. The pointwise bound
  ‖B' − B‖ ≤ sup|E'| + sup|E| is loose: it reaches 0.4 at the 0.55 T corners.
- **Circulant M:** unchanged, since it does not move the fixed point.
- **Checkpoints:** store the epoch, the coarse w_ref and the coefficient hash, so that a resume rebuilds the same
  operator (`feedback_resumable_runs`).

### 5.4 Robustness while the image is poor

- **Blended weight:** w_k = (1 − β)·Ĝ|x_k|²Σ|S|² + β·1_boxΣ|S|² (normalised), β = max(0.3, 1/(1 + SNR_k)). Tissue
  that is still dark in x_k is never weighted zero.
- **Ridge toward Taylor** (1e-6 rad²): directions the support cannot determine keep their Taylor value.
- **Minimax acceptance (data-free):**
  - the refit must predict ≥ 5 % gain under w_k;
  - it must show no loss under w_prev and w_box.
  - Each ε is the LS objective on the probe set, so this costs microseconds per sample.
  - The strict "no loss" is needed because the proxy under-reports losses about 4× (§2).
- **Noise-unit gate:** skip the refit unless ΔR_pred ≥ 0.02 (`segment_basis.md` §2 metric). At 2.89 T it will usually
  skip.
- **Held-out gate:** compare PE_H(Taylor) with PE_H(LS) at the first refit and at the end (`model_robustness.md` §3a).
  This is a sanity check, not an α-selection: the coefficients are physics, not data fits.

## 6. The update algorithm

```
state: epoch e; θ = (side records δp per set, interpolators a_l(τ) per encode [, diagonals D_l]); reference weights w_ref
θ_0: δp ← Fit_p(blend(warm-start energy, box; β = 0.5)); segments as today (warm start, --seg-weight energy)
K  = {end of warm start, 10, 20, 40, 80, ...} (geometric), N_max = 4 accepted refits
for k = 0, 1, 2, ...
    x ← T_θ(x)                                            # unchanged P²nP step / CG iteration
    trigger = k ∈ K  or  (PE_H stalled: |ΔPE_H| / PE_H < 1 % over the last 10 iterations)
    if not trigger or accepted ≥ N_max: continue
    w_t ← blend_β(G_1cm * |x_t|² · Σ_c|S_c|²) per set t;   w̄_e ← Σ_{t ∈ e} w_t per encode
    if max_t ‖w_t − w_ref,t‖₁ / ‖w_ref,t‖₁ < 0.10: continue          # the weights did not move
    δp' ← Fit_p(w_t)                     # GPU, every 8th ADC sample, linear in between (§3.4)
    a'  ← Fit_a(w̄_e) from block Grams    # §7: no new diagonals
    for each component c ∈ {δp, a}:
        ok_c = ε(θ'_c; w_new) ≤ 0.95 ε(θ_c; w_new)                                  # a real predicted gain
               and  max_{w ∈ {w_ref, w_box}} ε(θ'_c; w) / ε(θ_c; w) ≤ 1.00             # minimax over plausible weights
               and  R_pred(θ_c) − R_pred(θ'_c) ≥ 0.02                              # worth it in noise units
    if any ok_c:
        install the accepted components; e += 1
        drop the ZMSHR1 secant pair; refresh the lagged anchor; restart CG; λ_max ← 3 warm power iterations
        w_ref ← w; accepted += 1; checkpoint (epoch, coarse w_ref, coefficient hash)
    if basis rule (§7) says D_l is stale: full segment rebuild at the next checkpoint (expensive, at most once)
end: PE_H(Taylor coefficients) vs PE_H(final) on the held-out readouts, logged; flag if the final is worse
```

Pre-registered constants (changed only by a GPU test, §8):
- blend floor β 0.3–0.5;
- ridge 1e-6 rad²;
- trigger at a 10 % weight move;
- predicted gain ≥ 5 % under the new weight, and no predicted loss under the others;
- ΔR ≥ 0.02;
- N_max = 4.

Finitely many refits give Prop. 4. The minimax rule and the ridge give Prop. 3's "not worse than the incumbent under
any plausible weight". PE_H is the data-level check.

## 7. Cost: what must be rebuilt, and how to make it incremental

| component | depends on the image via | rebuild today | incremental form | cost per refit [model] |
|---|---|---|---|---|
| per-sample δp (side record) | w_t (per set) | n/a (new) | recompute δp only; records and tile lists untouched | 46.9 M samples / 8 × 512 probes ≈ 3e9 sincos + 1e11 FMA: **< 1 s GPU**, ≈ 15 s CPU (10 cores); write 0.75 GB host |
| monomial fit (φ0, k-shift, ΔC) | w̄ (shared) | 1.2 s (step 1) | weighted 10 × 10 / 35 × 35 normal matrix; but **φ0 / k-shift / ΔC live in the record**, so a refit rewrites `rec` and the tile lists (the k-shift moves samples between tiles) | step 7 again ≈ 52 s CPU (ledger §4). Gain only 2–4 % (§2): **do once, at the first refit, or never** |
| interpolators a_l(τ) | w̄_e (shared per encode) | ≈ 20 s per encode (46 % of the build) | **block Grams**: G_b = Σ_{r∈b} D_r^H D_r (L × L) and β_b(τ) = Σ_{r∈b} D_r^H g_r(τ) (L × N_tab), once per encode; then G_w = Σ_b ω_b G_b, β_w = Σ_b ω_b β_b, one L × L solve per τ | B = 8³–12³ blocks: 64–216 MB per encode (×2 water + fat); refit **< 0.1 s** per encode |
| diagonals D_l (basis) | w̄_e | ≈ 7 s per encode + upload 0.75 GiB | decision exact and cheap from block Grams of the 96-time grid (the weighted Gram is linear in ω_b, so its eigen-tail under w̄ and the current basis's error under w̄ both follow); the rebuild itself is the projection | ≈ 300 s today, seconds after ledger §4 items 1–4. **At most once**, only if ΔR_pred ≥ 0.05; L fixed (NSEG is a kernel define) |

Notes:
- **The error of fixed D_l under a new weight** is Σ_τ(Σ_b ω_b‖g‖²_{b,τ} − β_wᴴG_w⁻¹β_w). That makes the §6 test free.
- **The cardinal form** is an L × L transform and is redone in microseconds.
- **Δw_j for the per-sample fit:** order 3 stores it in `uw`; order 2 recomputes it from the schedule.
- **Zero-memory alternative** (not measured): fit δp as a degree-3 polynomial in Δw, i.e. 35 coefficients per weight,
  and evaluate it in staging.

## 8. Pre-registered GPU test plan

Per `feedback_experiment_discipline`: one factor at a time, config fingerprint (σ, W, BETA, NSEG, CORR, LSPOLY, KV)
checked in the kernel define string, dev scale first, circ metric unchanged.

| id | question → decision | run | prediction (falsifiable) | accept if |
|---|---|---|---|---|
| G1 | Is the side-record kernel exact? → merge | `seg_multi seg_order3_matches_emulation` and `seg_full_physics_order3_matches_direct_sum`, with a zero side record and with E2's LS records | zero record ≡ today (≤ 1e-7 relative); LS: impl error within `impl_tol` of the E2 emulation; model columns equal E2's (≤ 5 %); adjoint identity ≤ 1e-5 | all three |
| G2 | Is it cheap? → default on/off | opbench matrix (1 mm static, 1 mm knot cubic 10, 2 mm 2-D knots), order 2 and 3, LSPOLY 0/1 | time +1–3 %; host tables +8 % | ≤ 5 % time |
| G3 | Does the operator gain reach the image? → default by field | dev-scale sims (2 mm, flow phantom, full model + GNL) at 0.55 / 1.5 / 2.89 T: Taylor vs LS, with the coefficients fitted once from the warm-start weight; noise-free inverse-crime control (truth from the GPU direct-sum simulator) + noisy | noise-free image NRMSE floor: 0.55 T ≥ 1.5× lower (order 3); 1.5 T ≥ 1.2×; 2.89 T unchanged within 5 %. With noise: PE_H difference significant at 0.55 T only | LS default at B0 ≤ 1.5 T and order 3; off (zero record) at 2.89 T unless R_pred gain ≥ 0.02 |
| G4 | Does refitting from the iterate matter? → refit schedule | G3 setting, 0.55 and 1.5 T: weights from (a) warm start, (b) iterate at k = 40, (c) oracle truth |x̄_t|²; same coefficients otherwise | ε under the true weight: (b) within 3 % of (c), (a) within 10 % (E2: box / shifted cost 1–6 %); image NRMSE (a) vs (b) differs < 2 % | if (a) is within 3 % of (b): one fit from the warm start suffices and the refit loop stays off by default |
| G5 | Interpolator refit by block Grams → implementation | 1 mm box build, one encode: block-Gram refit vs full LS table at the same weight | table ≤ 1e-6 relative to the full fit (B = 512; the weight piecewise-constant per block by construction); refit < 0.1 s | equality and time |
| G6 | Shared-component weight: warm start vs mean_t|x_t|² → basis rebuild | 1 mm box (R2 config), the realistic run's resolved frames: R_pred of the current basis and a rebuilt one under mean_t|x_t|² | ΔR_pred < 0.05 (the warm start already captures the static anatomy; the Jensen deficit is in the vessels, a few % of the energy) | rebuild only if ΔR_pred ≥ 0.05 |
| G7 | Convergence with epochs → solver rules | 2 mm, P²nP zmshr1ms, 4 refits vs none, same final coefficients | the final objective and image agree to the CG tolerance; no iteration with an objective increase beyond the predicted Φ jump | as predicted |

Run order: G1 → G2 → G3 (decides whether the rest matters) → G4 → G5 → G6 → G7.

## 9. Open and caveats

- **Dev scale.** 12 mm voxels, 4 coils. The phases are realistic, the object is not, and the weight sensitivity depends
  on how concentrated the energy is.
- **Emulation only.** Window error and f32 are excluded; G1 checks them.
- **The proxy** is validated on one object only.
- **Velocity.** Δw_j differs between encodes, so part of the stencil error is encode-differential. G3 must report
  velocity error.
- **Not measured:**
  - the Δw-polynomial model;
  - P = 3 in χ;
  - more spectral bands;
  - per-knot coefficients in knot-major sets.

## 10. Literature

- P. L. Combettes. Quasi-Fejérian analysis of some optimization algorithms. In *Inherently Parallel Algorithms in
  Feasibility and Optimization*, Elsevier 2001, 115–152 [R]. KM iteration with summable errors (Proposition 4).
- M. Schmidt, N. Le Roux, F. Bach. Convergence rates of inexact proximal-gradient methods for convex optimization.
  NeurIPS 2011 [R]. Error-floor and summable-error forms of the inexact-gradient view (§5.2).
- B. P. Sutton, D. C. Noll, J. A. Fessler. Fast, iterative image reconstruction for MRI in the presence of field
  inhomogeneities. IEEE TMI 22(2):178–188, 2003 [R]. LS time-segment interpolators, the a_l(τ) refitted here.
- Internal: `stencil_kernel_optimality.md` (Prop. 1–2, E2), `segment_basis.md` §2 (noise-referenced metric and energy
  weights), `model_robustness.md` §1.1 (fixed-point error = forward model error at x̄), §3a (held-out gate),
  `memory_ledger.md` §3–4 (table sizes, build timers).
