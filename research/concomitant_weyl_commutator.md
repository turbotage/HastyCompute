# Concomitant fields via the Weyl/commutator calculus of the normal operator

Companion to `concomitant_structure.md` and `RESEARCH_fast_normal_operator.md`. Scripts: `research/experiments/{identities_check,operator_frontier}.py`.
Sections 1–2 are exact (and the identities are checked numerically); Sections 3–5 are measured on tiny grids (8³ voxels, 960–3000 samples, random SO(3) rotations of a synthetic
twisty/radial prototype, phase rescaled to a target max|Φ|); treat the numbers as trends.

## 0. Summary

- The structure you remembered is real and exact. With `X` = "multiply by coordinate" and `T[g]` = Toeplitz (convolution) operator with kernel `g(Δ)`:
  **`[X_a, T[g]] = T[Δ_a g]`** (kernel × Δ, i.e. a *derivative of the k-space multiplier*). Quadratic phases conjugate Toeplitz operators into *sheared* ones, and the Weyl symbol of the
  normal operator is **exactly** a sheared copy of the sample cloud: no approximation, no ħ-corrections (quadratic Hamiltonians).
- First-order concomitant effect: **`N ≈ T⁰ + i Σ_q [m_q(X), T^q]`** — commutators of the four field multipliers `m_q ∈ {z², x²+y², xz, yz}` (diagonal, in registers) with four *base* Toeplitz operators `T^q` whose
  kernel weights are `w_j·w_{j,q}` (moments of the sample coefficients). Higher orders are nested commutators = polynomials in coordinates sandwiching Toeplitz operators with Δ-polynomial kernels. This is the "factors out nicely" structure.
- The "finite difference" has three exact realisations; **grid finite differences of ρ̂ are the bad one** (Sec. 2.3). The good ones: coordinate-multiply + FFT (Toeplitz form), or analytic derivative of the NUFFT interpolation kernel (sample form, needs no extra FFTs).
- Measured: the commutator (Weyl) expansion needs ~2× fewer FFT pairs than a Taylor expansion of `e^{−iΦ}` for the same operator accuracy, at every phase level tested. The SVD (Tucker) basis still needs fewer FFT pairs than either, but with `L(L+1)/2` kernels
  and a data-dependent basis. One-sided product-convolution (Escande–Weiss) is clearly worse here.
- Nothing found yet reduces FFT pairs below ~`L` (the SVD rank). The commutator form is the best *data-independent, hierarchical* representation I could find; its value is exactness, tiny setup cost and a cheap first-order mode.

## 1. Operator calculus (exact)

Notation: `(X_a ρ)(r) = r_a ρ(r)`; `P = −i∇` (translation generator); `T[g]ρ(r) = Σ_{r'} g(r−r')ρ(r')`; `Q_j = e^{−iΦ_j(X)}`; sample `j` contributes `w_j Q_j T[e^{ik_j·Δ}] Q_j^H` to the normal operator, so

```
N = Σ_j w_j  Q_j  T[e^{i k_j·Δ}]  Q_j^H,                       N(r,r') = Σ_j w_j e^{ik_j·(r−r')} e^{i(Φ_j(r) − Φ_j(r'))}
```

1. `[X_a, T[g]] = T[Δ_a g]` and so `ad_X^β T[g] = T[Δ^β g]` (checked: 6·10⁻¹⁴ in `identities_check.py`).
2. Quadratic `Φ_j = ½ rᵀM_j r + b_j·r`: `Φ_j(r) − Φ_j(r') = r̄ᵀ M_j Δ + b_j·Δ`, `r̄ = (r+r')/2` (exact; bilinear in the midpoint and the difference).
3. Hence the Weyl symbol of `N` is `σ(r̄,k) = Σ_j w_j δ(k − k_j − M_j r̄ − b_j)`. Equivalently `N = Σ_j w_j δ(Π_j − k_j)` with `Π_j = P + M_jX + b_j`, a *commuting* triple (`M_j` symmetric ⇒ `[Π_a,Π_b]=0`).
4. With `M_j = Σ_q w_{j,q} M_q`, `Φ_j = Σ_q w_{j,q} m_q`: `N = Σ_j w_j e^{−i Σ_q w_{j,q} m_q(X)} T_j e^{+i Σ_q w_{j,q} m_q(X)}` and expanding the exponential of an adjoint action,

```
N = T⁰ + i Σ_q [m_q(X), T^q]  − ½ Σ_{q,q'} [m_q,[m_{q'}, T^{qq'}]]  + …                 T^{μ} := T[ Σ_j w_j w_j^μ e^{ik_j·Δ} ]   (w^μ = Π_q w_q^{μ_q})
```

   (`identities_check.py` item 4: first-order residual 3·10⁻², first+second 1·10⁻³ at small phase.) Two ways to organise order `n`, trading kernels for FFTs:
   - *field form* (above): base kernels `T^μ`, `|μ| ≤ n` → `C(4+n,n)` kernels (not `L²`), but the sandwiching diagonals are products of the `m_q` (quartic and higher in coordinates) → FFT counts like the Taylor-of-Φ route;
   - *Δ-form* (Weyl, via `[X_a,T[g]] = T[Δ_a g]` and `Φ(r)−Φ(r') = r̄ᵀMΔ+b·Δ`): the sandwiching diagonals are **coordinate monomials of degree ≤ n only** (generated from the voxel index in registers), `C(n+3,3)` FFT pairs, but the kernels become Δ-weighted moment kernels (more of them: 7 at order 1).
5. Order-1 explicitly (`H_q` = Hessian of `m_q`): `i Σ_q [m_q, T^q] = (i/2) Σ_{q,a,b} (H_q)_{ab} ( X_a T[Δ_b 𝒢^q] + T[Δ_b 𝒢^q] X_a )`. Distinct Δ-weighted kernels: **7** (`Δ_z𝒢¹; Δ_x𝒢²,Δ_y𝒢²; Δ_x𝒢³,Δ_z𝒢³; Δ_y𝒢⁴,Δ_z𝒢⁴`).
   FFT cost with the Δ-form: 4 forward (`ρ, xρ, yρ, zρ`) + 4 inverse = **4 pairs** (base term fused in), or with the commutator-of-`m_q` form 5 pairs and only 4 kernels. Both are exact at first order.
6. Everything above also applies to smooth non-polynomial `ψ` (off-resonance `ω(r)t_j`, GNL) to first order with `∇ψ(r̄)` replacing `M r̄ + b` (remainder `O(Δ³∂³ψ)`): first-order off-resonance is `Σ_a ½{∂_aω(X), T[Δ_a 𝒢^t]}` with `𝒢^t` weights `w_j t_j` — 3 diagonals, 3 kernels.

## 2. Three realisations of "derivatives instead of new transforms"

**2.1 Toeplitz/image-domain form.** Kernels `Δ_b·g` are the base kernel multiplied by the coordinate in the *kernel* domain (before its FFT): one extra FFT per kernel at build time, no NUFFT. Or avoid them: `T[Δ_b g] = [X_b, T[g]]` — only coordinate multiplications (free) and the base kernels.

**2.2 Fourier-domain form.** In k-space `X_a → i∂_{k_a}`, `T[g] → ĝ(k)·`. The order-1 term is a first-order differential operator in `k` acting on `ρ̂`: `i Σ_{q,a,b}(H_q)_{ab}[ ∂_bK̂^q ∂_aρ̂ + ½ ∂_a∂_bK̂^q ρ̂ ]`, `K̂^q` = gridded `w_jw_{j,q}` density.
   This needs derivatives of the *rough* gridded density (exact: Δ-multiply the kernel) and of the *smooth* `ρ̂`.

**2.3 Why grid finite differences of `ρ̂` are a trap.** The image fills up to half of the `2N` padded box, so `ρ̂` is sampled at only 2× Nyquist: content near the FOV edge advances ~π/2 per k-cell. A 2nd-order central difference then has ~40% error; you would need a wide high-order stencil, at which point `FFT(X_aρ)` is both cheaper and exact.

**2.4 Sample-domain form (NUFFT).** `ŷ_j = F[e^{−iΦ_j}ρ](k_j) = ρ̂(k_j) + (i/2) Σ_q w_{j,q} (D_qρ̂)(k_j) + O(Φ²)`, `D_q = −∂ᵀH_q∂` (second derivatives, `F[m_qρ] = D_qρ̂`). Interpolating the derivative is *exact on the interpolant*:
   `∂ ρ̂(k_j) = Σ_u ∂φ(k_j−u) X_u` — **no extra FFT**, one multi-RHS interpolation with derivative-kernel weights. Higher orders need higher derivatives; accuracy of derivatives of an interpolant degrades with order, so this is a first/second-order tool.
   Design consequence: the ES kernel `exp(β(√(1−z²)−1))` has a √-singularity in its derivatives at the support edge; Kaiser–Bessel `I₀(β√(1−z²))` is an entire function of `z`. A NUFFT meant for derivative weights should use KB (or ES with derivative-safe handling) and expose `φ, φ', φ''`.

## 3. Lifted view: why the splits look as they do

`N(r,r') = Ĝ(Δ, Y)` with `Δ = r − r' ∈ ℝ³`, `Y = m(r) − m(r') ∈ ℝ⁴`, `Ĝ(Δ,Y) = Σ_j w_j e^{i(k_j·Δ + w_j·Y)}` — the 7-D Fourier transform of the sample measure in `(k, w)`.
It is band-limited in `Y` by the extent of the visited set `{w_j}` (a ≤3-D cone-like set when spokes are linear images of a base curve). Every method is an approximation of a band-limited function of `Y`:

| method | expansion in `Y` | FFT pairs | kernels |
|---|---|---|---|
| commutator/Weyl order `n` | Taylor at `Y=0` in the *pair* phase | `C(n+3,3)` | moments (Sec. 1.4–1.5) |
| Taylor of `e^{−iΦ}` (your multinomial, `F_α`) | Taylor at `m=0` of each *factor* | `C(4+P,P)` | `C(4+2P,2P)` (Hankel) |
| nodes (frequency segmentation / CP) | interpolation at nodes `w^{(l)}` | `L'` | `L'` |
| SVD (Tucker) | data-optimal basis of the factor | `L` | `L(L+1)/2` |

The rank is set by the Slepian number of (visited `w` set) × (range of `Y`) — roughly total pair-phase range `2Φ_max` per active direction, not by resolution or by the number of spokes.
Base-spoke structure only makes the visited set known analytically (so nodes/Taylor center can be placed without an SVD). It does not shrink it.

## 4. Measured operator accuracy vs FFT pairs (`operator_frontier.py`)

Relative error is `‖N − N_approx‖_F / ‖N − N⁰‖_F` (1 = ignoring the concomitant field). `n=8³`, K=960 (undersampled) unless stated.

| method (FFT pairs) | Φ=0.3 | Φ=1 | Φ=3 | Φ=1, fully-ish sampled (K=3000) | Φ=1 radial |
|---|---|---|---|---|---|
| Taylor-Φ P=1 (5) | 1.7e-1 | 5.8e-1 | 1.8 | 7.7e-1 | 4.4e-1 |
| Taylor-Φ P=2 (15) | 6.2e-3 | 7.6e-2 | 1.1 | 8.7e-2 | 6.1e-2 |
| Taylor-Φ P=3 (35) | 4.5e-4 | 1.6e-2 | 3.7e-1 | 1.9e-2 | 1.2e-2 |
| Taylor-Φ P=4 (70) | 1.6e-5 | 2.1e-3 | 1.8e-1 | 2.3e-3 | 1.4e-3 |
| **Weyl n=1 (4)** | 6.2e-2 | 2.1e-1 | 6.1e-1 | 2.1e-1 | 2.0e-1 |
| **Weyl n=2 (10)** | 3.5e-3 | 3.8e-2 | 3.4e-1 | 4.0e-2 | 3.6e-2 |
| **Weyl n=3 (20)** | 1.7e-4 | 6.3e-3 | 1.7e-1 | 6.6e-3 | 5.5e-3 |
| **Weyl n=4 (35)** | 7.2e-6 | 8.9e-4 | 7.3e-2 | 9.3e-4 | 7.3e-4 |
| Weyl n=6 (84) | 9e-9 | 1.3e-5 | 9.4e-3 | 1.3e-5 | 9e-6 |
| SVD-Tucker L=6 (6) | 4.4e-3 | 1.5e-2 | 4.6e-2 | 1.7e-2 | 3.8e-2 |
| SVD-Tucker L=8 (8) | 2.8e-3 | 9.3e-3 | 3.0e-2 | 1.1e-2 | 3.0e-2 |
| SVD-Tucker L=12 (12) | 5.9e-4 | 2.1e-3 | 9.8e-3 | 2.4e-3 | 4.7e-3 |
| prod-conv L=16 (16, 1 diagonal) | 1.9e-3 | 2.8e-2 | 2.0e-1 | 2.7e-2 | 4.3e-2 |
| prod-conv L=32 (32) | 4.5e-5 | 2.4e-3 | 6.9e-2 | 2.2e-3 | 4.4e-3 |

(‖N−N⁰‖/‖N‖ = 0.8%, 2.8%, 7.9%, 1.7%, 7.6% for the five columns: the concomitant correction itself is small relative to N at these phases.)

Reading:
1. **Weyl beats Taylor-Φ at equal FFT pairs by about 2×** (e.g. Φ=1: Weyl n=3 (20 pairs) 6e-3 vs Taylor P=3 (35 pairs) 1.6e-2; Weyl n=4 (35) 9e-4 vs Taylor P=4 (70) 2e-3). Cause: the expansion variable is the pair phase `Φ(r)−Φ(r')`, which vanishes on the diagonal and stays small where `N` has weight.
2. SVD-Tucker wins on FFT pairs (6 pairs ≈ Weyl n=3 at Φ=1, ≈ Weyl n=6 at Φ=3) — at the price of `L(L+1)/2` kernels, a dense data-dependent basis (`K×V` SVD) and per-voxel basis storage.
3. Product-convolution (one-sided) is 2–5× more terms than SVD-Tucker: its single diagonal is not worth the rank.
4. Radial vs twisty and under- vs fully-sampled change little at this scale; the dependence is on Φ. Weyl order needed grows roughly linearly with Φ_max (n=3 gives 6e-3 at Φ=1; n≈6 gives 9e-3 at Φ=3), consistent with the pair-phase bound `|x| ≤ 2Φ_max`.
5. Cheap modes: at Φ≈0.3 the first-order Weyl form (4 pairs, 6e-2 of the effect = 5·10⁻⁴ of N) may already be enough; n=2 (10 pairs) reaches ~3·10⁻³ of the effect.

Costs the table hides: Weyl order `n` also needs the *moment/Δ-weighted kernels* (order 1: 7 kernels; order 2 roughly 40–60; order 3 ~150) and a k-space stage contracting ~`C(n+3,3)` derivative fields per voxel — memory-bound if kernels are streamed, so
kernel count can dominate before FFT count does. Building them is cheap (same sample sweep, `T^μ` for all `μ` in one multi-weight spread pass) and rebuilding per frame is the right default.

## 5. What to do with this

1. **Library hooks it needs (general, nothing MRI-specific):** FFT pre/post-multiply by *coordinate monomials generated from the voxel index in registers*; multi-weight type-1 (several weight vectors, one sort/kernel evaluation) for the moment kernels; NUFFT kernels with analytic derivatives (KB/ES-smooth) and multi-RHS derivative weights.
2. **Prototype order (cheapest information first):** (a) fix the coefficients (see `concomitant_structure.md` §0.1); (b) implement first-order and second-order commutator forms on top of your existing Toeplitz machinery at small N and compare against `phi_lowrank` at equal FFT pairs on your real sequences;
   (c) test the sample-domain derivative-kernel first-order forward model (no extra FFTs) against the exact forward model; (d) only then decide whether SVD-Tucker, Weyl, or a hybrid (Weyl on top of an off-resonance SVD basis, which is product-structured: `exp(−iΦ_tot) = exp(−itz(r))·exp(−iΦ_conc)`) is worth building.
3. **Hybrid worth testing:** off-resonance has a good spatial/temporal split (`L_z` terms) and concomitant is small → use Weyl order 1–2 for the concomitant factor *inside* each off-resonance term: cost `L_z × C(n+3,3)` FFT pairs. Compare with SVD of the product (rank ~ `L_z·L_c`-ish).
4. **Not found / not claimed:** a method with FFT pairs independent of the phase range. The M5 stencil route (see the other notes) is the only candidate with `L`-independent FFT count; the derivative-kernel form of 2.4 is its low-order cousin.

## 6. Pointers (not verified in depth)

- Fessler et al., IEEE TSP 2005 (time-segmented Toeplitz; eq. 24 is the diagonal/CP form); Sutton, Noll, Fessler, IEEE TMI 2003 (min-max time segmentation).
- Escande & Weiss, "Approximation of integral operators using product-convolution expansions", JMIV 2017 (arXiv 1604.04374) — the general theory of `N ≈ Σ D_l T[h_l]`; wavelet/pseudodifferential-operator variants in SIAM J. Imaging Sci. 2015. Measured here as inferior to two-sided forms for this operator.
- Concomitant correction in spiral/non-Cartesian recon: King et al. "Rapid correction for concomitant gradient field effects in spiral scans"; fast conjugate-phase with polynomial approximation (US 8,094,907); MaxGIRF (higher-order model-based recon incl. concomitant terms).
- Exactness of metaplectic Egorov / Weyl calculus (Coburn–Hitrik–Sjöstrand, arXiv 1807.00922 and follow-ups) — the reason Sec. 1.3 has no error terms for quadratic phases.
