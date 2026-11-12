# Fewer or cheaper iterations at a fixed estimator: coil subsets, step-tied prox, momentum, starts, pyramid

2026-11-12. Algorithmic side only (the operator's kernels / FFTs are a separate work package). Builds on
`research/convergence_p2np.md` (cost model, spectrum, toy) and `research/recon_diag_2026-11-11.md`. Code:
`rustlib/.agents/algo.patch` (tree `rustlib/.agents/seq`); runs: `data/hasty/phantom/factory_arms/ph0001/algo_2026-11-12/`
(README there). Claims marked *measured* come from those runs or from the GPU benchmark below; *toy* = the synthetic
GPU benchmark; [CHECK] = not verified.

## 0. Summary

(filled in at the end of the runs — see §8)

## 1. What exists now (code)

All additive; defaults unchanged (a run without the new flags has the same tags, files and iterates as before).

| item | where | flag(s) |
|---|---|---|
| streamed coil-sampled host gradient (one `select_coils` per gradient: the union of the groups, residual rows scaled by w / √w) | `p2np.rs` `gradient_host_sampled` | `--coil-sampling` now also with `--state host` (any image size, 2 mm) |
| SVRG-anchored coil sampling, secants against the anchor | `p2np.rs` `with_svrg`, `SecantPolicy` | `--svrg M` |
| host solver with momentum (FISTA, generalised-gradient restart), step-aware prox hook, held secant reference, cost / exact-f reporting | `hoststack.rs` `run_objective_host_ext` (`run_objective_host` = the old loop through it) | `--momentum fista[:norestart]` |
| step-tied LLR threshold θ_k = μ σ̂_0 τ̄_k | `recon_p2np.rs` (`ProxRule`) | `--prox step[:μ]` |
| fixed LLR block grids (8 shifts {0, b/2}³, one per cycle), momentum restart at a change | `llr.rs` `LlrShift` | `--llr-shift fixed[:C]` |
| LLR with explicit threshold, regulariser value / kept rank reported; truncated SVT (subspace iteration) | `llr.rs` `apply_thr`, `top_eigs` | `--llr-trunc` |
| density-weighted start (radial shells, Nyquist-capped), exact line search per set | `dcw.rs`, `p2np.rs` `dcw_step_host` | `--init-dcw add|adj`, `--dcw-cap`, `--dcw-bins` |
| τ cap for ZMSHR1 | `hoststack.rs` `HostSolverOpts::tau_max` | `--tau-max T` |
| several solver configurations per process (one setup) | `recon_p2np.rs` `Variant` | `--variants "k=v,…;…"` |
| per-run solver metrics: cost, exact f, LLR θ / σ̂ / ‖·‖_* / rank, velocity RMSE (ML and hierarchical decoders), slope / correlation vs truth (all, per-frame deviations, time means) | `recon_p2np.rs` | `algo_<tag>.csv` |

Tests (GPU, < 0.5 GB device memory):
- `hasty-recon/tests/algo_host.rs`:
  - an epoch of streamed sampled gradients equals the full gradient (rel 1e-5, float32);
  - the streamed path equals the whole-stack sampled path (2e-5);
  - the SVRG anchor gradient equals the exact one (7e-6);
  - the epoch mean of the control variate equals ∇f(x) at x ≠ x_a (2.7e-5);
  - ⟨s, m⟩ > 0 for the anchor secant;
  - the DCW line search never increases f;
  - convergence per cost.
- `tests/algo_bench.rs` (ignored; the benchmark of §2–§5).
- `llr.rs`:
  - an explicit threshold is the exact prox of θ‖·‖_* (perturbation test);
  - fixed-shift cycles;
  - real-image full vs truncated SVT (rel 1e-7..3e-6).
- `dcw.rs`: 1/k² density → w ∝ k².
- Existing `host_state` and `coil_sampling` tests pass unchanged.

## 2. Synthetic benchmark (toy, GPU)

`tests/algo_bench.rs`: 48×48×32 grid, 12 sets (a dynamic "vessel" in a textured ellipsoid), 20 coils compressed to 16
energy-ordered virtual coils, 3D centre-out radial (240 readouts × 128 samples per set, golden-means directions
interleaved over the sets; density ∝ 1/k², spread (k_max/k_min)² = 1.6·10⁴), 4 off-resonance segments, noise 5 % of
the data rms.

- **One fixed convex objective.** F(x) = ½Σ‖E_t x_t − y_t‖² + λ Σ_b ‖X_b‖_*, LLR blocks of 8³ on one fixed grid,
  λ = μ σ̂_0/α.
- **Reference.** x* = FISTA after 400–500 iterations.
- **Cost unit.** One full gradient = 1; a sampled gradient counts its coils / all coils.
- **Start.** Zero unless stated.

### 2.1 Distance to x* after 30 gradient-equivalents (*toy*)

| method (μ = 0.5) | iterations | ‖x − x*‖/‖x*‖ | (F − F*)/F* |
|---|---|---|---|
| ISTA (α = 1/λ_max) | 30 | 0.151 | 2.8e-4 |
| FISTA, restart | 30 | 0.0141 | 1.1e-6 |
| FISTA, no restart | 30 | 0.0197 | 3.0e-6 |
| FISTA, coil sampling 4+4 of 16 (no variance reduction) | 58 | 0.0195 | 3.4e-5 |
| **FISTA, SVRG 4+4 of 16, period 4** | 46 | **0.0022** | **4.9e-8** |
| ZMSHR1 (global) + step-tied prox | 30 | 1.31 (diverges) | 5.7e-2 |
| ZMSHR1s (per set) + step-tied prox | 30 | 1.56 (diverges) | 9.1e-2 |
| ZMSHR1 + step-tied prox, τ ≤ 1.9 (μ = 0.1) | 30 | 0.31 | 8.1e-3 |
| ZMSHR1 + step-tied prox, τ ≤ 4 (μ = 0.1) | 30 | 0.14 | 2.8e-3 |
| FISTA, restart (μ = 0.1) | 30 | 0.046 | 5.3e-5 |
| FISTA, SVRG 4 (μ = 0.1) | 46 | 0.029 | 1.3e-5 |
| FISTA, restart (μ = 0.02) | 30 | 0.060 | 8.2e-5 |
| FISTA, SVRG 4 (μ = 0.02) | 46 | 0.025 | 2.2e-5 |

Reading:
- **Momentum.** About 10× closer than ISTA at equal cost. Restart helps at μ = 0.5 (0.014 vs 0.020) and is neutral
  at μ = 0.02.
- **SVRG-anchored coil subsets.** They make about the same progress per iteration as full gradients at half the
  cost: 1.6–6× closer than full FISTA. Plain sampling without the anchor stalls at its noise floor (0.02).
- **BB-type quasi-Newton steps and a step-tied prox are incompatible without a safeguard.**
  - ZMSHR1's τ (up to 9 measured on the real data; 1.5–2.4 here) leaves the proximal-gradient range γ < 2/L.
  - Capping τ < 2 restores convergence but leaves ZMSHR1 at ISTA speed.
  - See §4.

### 2.2 Distance to the truth (*toy*; the estimator question, not the solver question)

| μ | x* (FISTA 500) | FISTA 30 | ZMSHR1s fixed-θ PnP 30 | ZMSHR1s fixed-θ PnP 120 |
|---|---|---|---|---|
| 0.5 | 0.857 | 0.857 | 0.81 | — |
| 0.1 | 0.544 | 0.545 | **0.374** | 0.457 |
| 0.02 | 0.410 | 0.394 | 0.529 | 1.48 (diverges) |

- **Fixed-θ PnP has its own fixed point.** The fixed-threshold PnP iteration with ZMSHR1 (θ = λα whatever the step,
  the production rule's analogue) does not converge to x*. Its effective weight is λ/τ_t.
- **Its iterates semi-converge.** At μ = 0.1 they are closest to the truth after about 30 iterations, then drift. At
  weak priors they diverge.
- **Comparable best accuracy.** The best values are similar: 0.374 for PnP early-stopped at 30, 0.394 for FISTA
  converged at μ = 0.02. FISTA's is reached by a convergent method whose answer does not depend on the iteration
  count.
- **Why this matters for the real data.** At 2 mm the residual is still 6.5× the noise (*measured*,
  recon_diag §2), so the real runs are in the under-fitted regime. Early stopping regularises less there [CHECK].

## 3. Stochastic coil subsets

### 3.1 The existing scheme (`--coil-sampling keep:draw[:hold[:last[:seed]]]`)

The keep coils (leading virtual coils) are always used. The remaining nc − keep are partitioned at random into
p = ⌈(nc − keep)/draw⌉ groups per epoch (random reshuffling) and used in turn with weight w = p:
∇̃f = Σ_{c<keep} g_c + p Σ_{c∈D_b} g_c, with g_c = E_cᴴ(E_c x − y_c).

- **Unbiased.** E_b[p Σ_{D_b} g_c] = Σ_rest g_c, because each coil lies in exactly one group and b is uniform within
  an epoch. Also when the SVRG period and the epoch are not aligned: the marginal inclusion probability is 1/p.
  *Measured* in `algo_host.rs`: the epoch mean equals the full gradient to 1e-5.
- **Variance per draw.** Var = p Σ_b ‖G_b‖² − ‖Σ_b G_b‖², with G_b = Σ_{c∈D_b} g_c; about (p − 1) Σ_rest ‖g_c‖²
  for uncorrelated coil terms. At the solution the coil terms do not vanish individually: g_c(x*) = E_cᴴ r_c*,
  driven by noise and model error. Plain sampling therefore has a noise floor ∝ γ²(p − 1)Σ‖E_cᴴ r_c*‖². The last
  `full_last` iterations use all coils to end on the full fixed point.
- **Secants.** With different subsets in consecutive iterations,
  g̃_k − g̃_{k−1} = H̃_b s + (g̃_b − g̃_{b'})(x_{k−1}).
  The second term is the sampling noise of the gradient. It dominates H̃ s once s is small, so the ZMSHR1 pair is
  corrupted.
  - The existing scheme holds a subset for `hold` iterations: pairs inside a hold are exact for H̃_b. On a subset
    change it falls back to the plain metric step (τ = 1), so with hold 2 every second ZMSHR1 step is lost.
  - Before this change, sampling with `--state host` panicked: the whole-stack path needs two device stacks, which
    rules out 2 mm.

### 3.2 SVRG-anchored subsets (`--svrg M`, new)

- **Anchor.** Every M iterations (and on full-coil iterations of the schedule) the exact all-coil gradient
  g_a = ∇f(x_a) at the anchor.
- **In between, the data-free control variate:**
    ∇̃f(x) = g_a + Σ_G w_G N_G (x − x_a),
    f̃(x) = f(x_a) + Re⟨g_a, x − x_a⟩ + ½ Σ_G w_G ‖E_G(x − x_a)‖².
  - Unbiased for any x, since f is quadratic.
  - Variance ∝ ‖x − x_a‖², which vanishes at the fixed point: no noise floor, and no `full_last` needed.
  - Inner iterations read no data, so no staging.
- **Secants against the anchor.** g̃_k − g_a = Ñ_k (x_k − x_a) exactly. So (x_k − x_a, α(g̃_k − g_a)) is an exact
  pair for the iteration's sampled curvature every iteration, with no hold. At a new anchor,
  (x_a − x_a', g_a − g_a') is an exact pair for the full curvature.
  - Cost: the secant direction is the displacement since the anchor, not the last step. This changes the BB dynamics
    [CHECK on real data].
- **Implementation.** The solver's secant reference (x_prev, g_prev) is updated only at anchors
  (`SecantPolicy::Reference`).
- **Cost per period.** 1 + (M − 1)ρ, with ρ = (keep + draw)/nc.
  - Real data: keep 8 + draw 4 of 31 gives ρ = 0.39, p = 6, M = 7 (one reshuffling epoch per anchor), so 0.48 per
    iteration.
  - Keep 4 + draw 4 gives ρ = 0.26, p = 7, M = 8, so 0.35 per iteration.
- **Memory.** Two more host stacks (x_a, ∇f(x_a)): +17 GB at 2 mm.

### 3.3 Results

- *Toy* (§2.1): FISTA + SVRG is 1.6–6× closer to x* than full FISTA at equal cost.
  - With ZMSHR1s + step-tied prox both are unstable.
  - With fixed-θ ZMSHR1s, SVRG gives the same picture as without it (*toy*: 1.49 vs 1.37 to x*, which here measures
    the drift of a different fixed point).
- Real data, 8 mm: §8 (E2b, E3).

## 4. Step-tied ("consistent") LLR prox

- **The rule.** θ_k = λ γ_k with γ_k the step actually taken, here θ_k = μ σ̂_0 τ̄_k:
  - σ̂_0 is the PnP noise scale of the starting image, so μ ≈ κ at τ = 1;
  - τ̄_k is the geometric mean of the sets' scalar step factors.
- **When it is consistent.** For a single scalar step (no per-set τ, no rank-one term) the fixed point solves
  0 ∈ ∇f + λ∂R whatever the steps: the minimiser of F, the same for every solver.
- **Fixed shifts.** Cycle spinning changes R every iteration. `--llr-shift fixed:C` uses one of 8 fixed grids per C
  iterations, so within a cycle F is fixed. The momentum restarts at a change.
- **Per-set ZMSHR1 cannot be consistent with a block prox.** An LLR block couples all 280 sets.
  - With per-set τ_t the metric-prox of ‖X‖_* in the metric diag(1/(ατ_t)) has no closed form.
  - A correction to convergence_p2np.md §5.1 [CHECK]: the column-weighted SVT, Z = SVT_λ(V D^{−1/2}) D^{1/2}, is the
    exact prox of the *weighted* nuclear norm ‖X D^{−1/2}‖_* in that metric. It is not the prox of ‖X‖_*, so its
    estimator depends on the τ_t.
  - With one global θ (τ̄), each set's fixed-point weight is λτ̄/τ_t, and it changes every iteration with the BB
    dynamics.
- **BB-type steps with a step-tied prox diverge without a safeguard** (*toy*, §2.1, all μ, global and per set).
  - Proximal gradient needs γ < 2/L. ZMSHR1's τ = ρ(1 − sin ϑ) ≤ BB2 reaches 9 on the 8 mm data (*measured*:
    `zmshr1 per set` log lines) and 2.4 on the toy.
  - Without a prox, BB is R-linear on quadratics despite τ > 2: the amplified stiff components are corrected later.
    With an iteration-dependent shrinkage this has no guarantee, and the toy diverges.
  - A cap τ ≤ 1.9 converges, at ISTA speed.
  - The principled fix is a non-monotone acceptance test (SpaRSA, Wright–Nowak–Figueiredo 2009):
    - accept if F(x+) ≤ max_{j<M} F(x_{k−j}) − (σ/2γ)‖x+ − x_k‖², else shrink τ;
    - f(x+) comes free with the next gradient, and the rejected point's gradient still gives an exact secant pair;
    - not implemented.
- **PnP with a fixed θ (production).** It is stable on the 8 mm data (*measured*, pyr_2mm). Its fixed point has
  per-set weights θ/(ατ_t), changing with τ_t, so the estimator is defined by the algorithm (convergence_p2np.md §4).

## 5. Momentum (FISTA with restart)

- **Restart test without an extra stack.** `--momentum fista` uses the generalised-gradient restart
  (O'Donoghue–Candès 2015), restarting when Re⟨y_k − x_{k+1}, x_{k+1} − x_k⟩ > 0. y_k is the shifted secant reference,
  or stashed in the gradient buffer after the step. The solver then holds x, ∇f, x_k (+ the secant pair).
  - At 2 mm: FISTA on the plain step uses 3 host stacks (26 GB) instead of ZMSHR1s's 4.
  - FISTA + SVRG uses 5 (44 GB), about 75 GB RSS with the 27 GB setup (*measured* setup RSS of the current 2 mm run),
    under the 84 GB cap.
- *Toy*: about 10× closer than ISTA (§2.1).
- With fixed-θ PnP-ZMSHR1s, momentum changes little (*toy*): both follow their own fixed point.

## 6. Starting images

- **Density-weighted (DCW) start, LS-scaled per set.** x_t ← x_t + s_t E_tᴴ W r_t with the exact line search s_t
  (`dcw_step_host`): f_t never increases.
  - *Toy* (μ = 0.1), distances to x* and to the truth:

    | start | to x* | to the truth |
    |---|---|---|
    | zero | 1.0 | 1.0 |
    | DCW from zero (plain 1/ρ) | 18.5 | 3.4 |
    | DCW, density floored at Nyquist | 8.4 | 2.8 |
    | time mean (pooled PCG, 10 it) | 0.55 | **0.353** |
    | time mean + per-set DCW deviation | 1.44 | 1.12 |

  - **Why the DCW start is bad.** Per set the data are undersampled in the bulk (240 readouts against about 3600 for
    Nyquist at the edge). W ∝ k² up-weights exactly the shells whose aliasing the data barely constrain. The line
    search on f cannot see it: f drops 2 % while the image moves far. The Nyquist floor halves the damage but does not
    remove it.
  - The time mean is better than x* itself here: the pooled data are 12× denser.
  - On the real data: E4 (§8).
- **Recommendation (toy):** keep the time-averaged (pooled, physics-model) warm start per encode. Bring in temporal
  detail through coarser temporal groups (`--tpyramid`, E5) rather than per-frame density weighting. The DCW image is
  useful only as an initialisation of a level whose sampling is close to Nyquist [CHECK].

## 7. Cost of the LLR prior (a cheaper iteration, measured)

- **The prior is a large share of the iteration.**
  - At 8 mm one LLR call takes 15–20 s with 6 threads (225 blocks of 512 × 280), against 17 s for the gradient.
  - At 2 mm in tonight's run it took 155–174 s, against 303 s for the gradient: 35 % of the iteration (*measured*,
    pyr_2mm/L2_s1 log).
- **The cost is the eigensolver.** It is the nalgebra f64 Hermitian eigendecomposition of the 280 × 280 Gram matrix
  per block: about 0.5 s per block. LAPACK `zheevd` (OpenBLAS, 1 thread) takes 25 ms for the same matrix
  (*measured*, numpy).
  - A 20× faster prior is available through a LAPACK or faer dependency: about 170 s → 9 s per 2 mm iteration.
  - This is a dependency decision (faer is pure Rust and portable to aarch64), so it is not done here.
- **LLR barely shrinks.** At κ = 1 the PnP rule keeps 206 of 280 singular values per block (75 %, by construction of
  σ̂ = median of the lower-quarter singular value). At κ = 0.5 it keeps 239, at κ = 2 158 (*measured*, real 8 mm
  images, `llr::prox_tests::bench_trunc_real`).
  - So a truncated SVT (only the eigenpairs above θ²) gains nothing: 35–68 s vs 7–20 s full. It is kept as an option
    (`--llr-trunc`).
  - The prior acts as a nearly uniform shrinkage by σ̂ of most singular values. This is consistent with κ not
    mattering at 8 mm (recon_diag §2).

## 8. Real data (8 mm level of pyr_2mm, ph0001 10-min scan, oracle physics)

(results pending: E1 ZMSHR1s variants, E2 FISTA + step-tied prox, E2b FISTA + SVRG, E3 coil sampling with the
production solver, E4 DCW start, E5 temporal-pyramid start)

## 9. Recommended configuration and 2 mm estimate

(pending)

## 10. Open questions

- A non-monotone (SpaRSA-type) safeguard would make ZMSHR1 compatible with a step-tied prox: the preconditioned-LS
  family with a well-defined estimator.
- Accelerated variance reduction with guarantees (Katyusha-type negative momentum toward the anchor). Plain
  FISTA + SVRG works on the toy without it.
- λ calibration across pyramid levels (σ_ℓ² ∝ 1/samples), and λ vs PnP κ at equal effective strength (μ ≈ κ/τ̄).
- Is the per-frame estimator itself accurate enough? On the toy the pooled time mean is closer to the truth than the
  per-set minimiser at any μ tried. The prior choice (LLR rank ~75 %) may limit the temporal deviations as much as
  convergence does [CHECK].
