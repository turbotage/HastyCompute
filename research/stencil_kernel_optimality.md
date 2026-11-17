# Stencil and gridding-kernel optimality for the segmented physics NUFFT (2026-11-17)

Scope: the operator of `research/mathematics_of_paper/reconstruction_operator_section.tex` (§ phase split, time
segments, chirp-aware stencils) with the cost structure of `research/temporal_operator.md` §9–§17, as implemented in
`rustlib/crates/hasty-nufft/src/segnufft.cu` (ES window, σ = 1.25, W = 10 at 1 mm / W = 8 at 2 mm, derivative
corrections to order 4, 4³ tiles). Scripts and their raw outputs are in `research/experiments/stencil/`. Every
number marked [CPU] was computed for this note with numpy: 1-D sweeps, or 3-D direct sums at N = 32³ with 1500 samples.
That is dev scale, so the numbers are not GPU timings. [model] marks predictions from the measured cost structure.
[meas] marks numbers quoted from the two documents above.

"Operator error" is ‖y − y_direct‖/‖y_direct‖ against an f64 direct sum. It is reported for a white random image
(uniform energy, the worst case for FOV-edge errors) and for a smooth object (energy towards the centre).

---------------------------------------------------------------------------------------------------------------------

## 0. Answer

1. **The Taylor stencil is optimal only in a degenerate sense. Any degree-P polynomial in the chirp operator costs the
   same, so the polynomial should be chosen by the error measure.** The derivative stencil applies
   p(χ(i∂_k)) to the window, and on the plane wave the result is exactly p(χ(ρ))·(window interpolant). The whole
   truncation error is therefore E(χ(ρ)) with E(s) = e^{−is} − p(s), a function on the *range* of χ over the FOV
   (Prop. 1).
   - The worst-case optimum is the complex minimax polynomial on that range. Its error is ≈ 2(h/2)^{P+1}/(P+1)! for
     half-width h, against Taylor's a^{P+1}/(P+1)! with a = max|χ|. That is 2^P better for a symmetric range and
     2^{2P+1} = 32× better at P = 2 for a definite chirp (range [0, a]).
   - The operator-norm optimum for an image with energy density w is the **w-weighted least-squares polynomial**
     (Prop. 2). Taylor is the limit w → δ at χ = 0.
   - The coefficients are per-sample complex numbers computed on the host, so this costs nothing at run time.
   - Measured in 3-D [CPU, σ 1.25, W 8]:
     - |χ| ≤ 0.7: object error 1.7e-3 → 1.7e-4 (definite chirp), 3.4e-4 → 8.7e-5 (indefinite).
     - |χ| ≤ 1.4: 1.6e-3 → 8.2e-4.
     - So the P = 2 stencil now reaches ≈ 1–1.4 rad instead of ≈ 0.7 rad.
   - The LS/POD stencil (b) is the optimum over *all* stencils of its support. It wins only where no low-degree
     polynomial fits (|χ| ≳ 1.5 rad, or non-polynomial phase), and it costs a (W + 2S)³ support instead of W³.
   - **Segments vs stencil:** with the cost model of §1.4, a term goes into the stencil whenever it is polynomial
     (degree ≤ 2 exactly; 3–4 to first order) and its range fits the polynomial error budget.
     - In the sparse regime one extra segment costs ≈ 9 % of an image and the stencil ≈ 0–5 %, so the stencil wins
       even at ΔL = 1.
     - In the dense regime at order 0, turning the weights complex costs ≈ 40 % (equivalent to ΔL ≈ 4). Segments win
       there if the union basis grows by ≤ 3.
2. **The kernel is over-wide for a 1e-3 target.** At σ = 1.25 the 3-D plain error is [CPU, white / object]:
   - W = 10: 4e-6 / 5e-7;
   - W = 8: 4e-5 / 1e-5;
   - W = 7: 2.2e-4 / 7e-5 (ES), 1.2e-4 / 9e-5 (KB);
   - W = 6: 1.1e-3 / 3e-4 (ES), 5e-4 / 3e-4 (KB).

   With the order-2 chirp terms included (|χ| ≤ 0.7), W = 7 sits at the polynomial-truncation floor. The production
   W = 10 is a leftover of the ES derivative clamp argument.
   - Kaiser–Bessel (Beatty β) beats ES by 1.5–2× at equal W (≈ ½ tap). Its derivatives are entire, so no clamp is
     needed: unclamped ES gives e₄ ≈ 40 at W = 7, against KB's 5e-2.
   - Re-tuning ES's β does not close the gap.
   - Per-order least-squares (min-max-type) taps are 3–6× better on ∂² / ∂⁴ but no better on the value. They matter
     only if the derivative terms dominate.
   - **(σ, W) optimum** [model]:
     - 1 mm static (FFT-bound): σ = 1.125, W = 8–9 gives 1.15–1.18×.
     - 1 mm knots: σ = 1.25, W = 6–7 gives 1.3–1.4×.
     - 2 mm dense: σ = 1.25, W = 6 or σ = 1.5, W = 5–6 gives 1.3–1.5×.
     - σ = 2 always loses (FFT 2.4×).
3. **No kernel trick reduces C, and none reduces L below the SVD rank.**
   - Frequency segmentation (MFI) and time segmentation are both separable rank-L factorisations of e^{−iω(r)τ}, so
     Eckart–Young gives them the same lower bound (Prop. 4).
   - Coil maps as k-space kernels fail on compactness. Biot–Savart loops 2 cm from the box need K = 25 taps per axis
     on the σ-grid for 3e-3 [CPU]. The per-sample stencil then grows (W + K − 1)³ / W³ ≈ 90×, to save 22× in FFTs.
     Even at K = 13 (4 % map error) a 1 mm static image would take ≈ 1.3–1.6 s against 0.55 s now.
   - What does reduce L is what is already in place: the polynomial part of the field as exact k-shift plus chirp
     (`--b0-absorb`), with the remainder in the segments.
4. **No cheaper exact AᴴA for the time-resolved structure.**
   - The segment-pair Toeplitz form needs L(L + 1)/2 = 66 kernels: 36 GB at 1 mm, 2.4× the FFT bytes per channel of
     our σ = 1.25 pair, plus 0.8 s of kernel reads per pass. That is 1.7 s against our 0.55 s.
   - Compressing the L × L kernel matrix per k-point does not work [CPU, 2-D spiral]. Rank 4 of 9 still leaves
     3.5e-2 (white), because the 2N embedding's Dirichlet tails mix samples at different τ. A parametric rank-1 model
     fails completely.
   - Toeplitz wins only for **static dense sets** (2 mm, 30 M samples on one image): ≈ 0.25 s against ≈ 1.3 s [model],
     with 7–15 GB of kernels. Examples are warm starts, motion-estimation bins and per-encode time averages, but not
     the knot problem.
5. **Floor of the tile kernels.** Per sample, slot and channel the irreducible work is W³ complex MACs: 2 FMA with real
   weights, 4 FMA with complex ones.
   - The current dense 4³-tile loop multiplies (W + 3)³ points (W = 10: 2197, 45 % useful; W = 7: 999, 34 %).
   - 2 × 2 × 2 sub-box warps multiply ≈ 1.45 W³ (67–75 % useful).
   - Each sample is staged ((W + 3)/4)³ times (W = 10: 34; W = 7: 16).
   - At 2 mm the measured kernels (≈ 10 s) are ≈ 7× above the MAC floor at W = 8 (1.4 s at 70 % FMA peak). The factor
     splits as 2.6 (zero-weight points) × ≈ 2 (shared-memory operand bandwidth, latency) × 1.3 (staging).
   - Separable two-pass gridding does not lower the per-sample MAC count; only sharing across samples (Toeplitz) does.

**Ranked proposals** (expected gain from the measured cost structure; details in §6):

| # | proposal | regime gain [model] | accuracy | effort |
|---|---|---|---|---|
| 1 | W 10 → 7 at 1 mm, W 8 → 6–7 at 2 mm (ES now, KB-Horner next) | 1.08 static 1 mm; 1.32 knot 1 mm; 1.2–1.42 at 2 mm | 2e-4 → 5e-4 kernel error (3-D) | trivial (W is a define) + tests |
| 2 | energy-weighted LS chirp polynomial (per-sample complex coefficients) | none in time; extends the P = 2 range 0.7 → 1–1.4 rad, avoiding segments / LS stencils | 2–10× lower stencil error | low (host-side coefficients, complex e[] / c[]) |
| 3 | fp16 grid storage, fp32 arithmetic (FFT passes + tile-value loads) | 1 mm static ≈ 1.4–1.5×; 2 mm ≈ 1.1× | 4e-4 grid, 1.8e-4 sample-weighted [CPU] | medium |
| 4 | k-ball-pruned last forward / first inverse FFT pass | ≈ 0.8× FFT → 1.1× static 1 mm | exact | low–medium |
| 5 | KB kernel by piecewise-polynomial (Horner) tables, no clamp | half a tap (W 7 → 6 at 5e-4) | derivative errors bounded | low |
| 6 | point-lane / sub-box forward for complex weights (§17 item 2) | ≤ 1.9× on the forward MAC part (999 → 512 points at W = 7) | none | medium–high |
| 7 | σ as a planned parameter (1.125 for FFT-bound static sets) | 1.15–1.18 static 1 mm | with W = 8–9 | low (FFT sizes 2^a3^b5^c7^d) |
| 8 | segment-pair Toeplitz normal operator for static dense sets only | ≈ 5× on those sets | exact w/o chirp | medium; not for knots |

Rejected with numbers: coil kernels in k-space (§3.2), per-κ low-rank Toeplitz (§4.2), σ = 2 (§2.4), MFI as an L
reducer (§3.1), and the low-rank NUFFT of Ruiz-Antolín & Townsend (K³ FFTs in 3-D, §2.5).

---------------------------------------------------------------------------------------------------------------------

## 1. The chirp stencil: what it optimises

### 1.1 Exact factorisation of the error

Notation: grid-radian image coordinate ξ ∈ Ω_ξ = [−π/σ, π/σ]³ (ξ = 2πr/(σN)), sample k ∈ ℝ³ in fine-grid units,
window φ (separable), interpolant I_φ(k, ξ) = Σ_u φ(k − u) e^{−iu·ξ}. The chirp is χ(ξ) = ξᵀQξ (quadratic after
the split; Q = ΔC_j in grid units).

**Proposition 1 (error factorisation).** For any polynomial p,
  Γ_p := p(χ(i∂_k)) φ(k − ·) satisfies Σ_u Γ_p(u) e^{−iu·ξ} = p(χ(i∂_k)) I_φ(k, ξ).
With I_φ = φ̂(ξ) e^{−ik·ξ}(1 + ε_0) and ∂^α_k I_φ = φ̂(ξ)(−iξ)^α e^{−ik·ξ}(1 + ε_α),
  Σ_u Γ_p(u)e^{−iuξ} − φ̂ e^{−ikξ}e^{−iχ(ξ)} = φ̂ e^{−ikξ} [ (p(χ(ξ)) − e^{−iχ(ξ)}) + Σ_α γ_α(p,Q)(−iξ)^α ε_α(k, ξ) ].

*Proof.* ∂_k commutes with the finite sum over u. χ(i∂_k)e^{−ikξ} = χ(ξ)e^{−ikξ} because i∂_k e^{−ikξ} = ξ e^{−ikξ}.
Expand p(χ(i∂)) = Σ_α γ_α ∂^α and insert the window errors term by term. ∎

So the stencil error is the sum of two terms:
- **truncation** E_p(χ(ξ)) with E_p(s) = e^{−is} − p(s). It depends on ξ only through the value of χ, and it is the
  same for every kernel;
- **window** Σ γ_α(−iξ)^α ε_α. With e_n := ‖ε_n‖ normalised by (π/σ)^n (§2), it is bounded by
  e_0 + χ_max e_2 + ½χ_max² e_4 at P = 2.

The production Taylor stencil, eq. (dstencil) of the paper, is p_T(s) = 1 − is − s²/2.

### 1.2 Which p is optimal

Write R = χ(Ω) = [m, M] for the range of χ over the support, with centre c = (m + M)/2 and half-width h.

**Proposition 2 (optimal polynomial).**
(i) *Worst case.* sup_ξ|E_p(χ(ξ))| = sup_{s∈R}|e^{−is} − p(s)|. The minimax p over degree ≤ P is therefore the complex
Chebyshev approximation of e^{−is} on R. By Jacobi–Anger, e^{−i(c + h cos θ)} = e^{−ic} Σ_n (−i)^n ε_n J_n(h) cos nθ,
so truncating the Chebyshev series gives
  E*_P(R) ≤ 2 Σ_{n>P} |J_n(h)| = (2 (h/2)^{P+1}/(P+1)!) (1 + O(h²)).
This is within a factor ≤ 2 of the minimax error, since J_{P+1}(h)T_{P+1} equioscillates to leading order.
(ii) *Taylor.* sup|E_{p_T}| = a^{P+1}/(P+1)! (1 + O(a)), a = max(|m|, |M|) ≥ h. The ratio Taylor / minimax is
  ≥ 2^P (a/h)^{P+1}/2. For a symmetric range (a = h) that is 2^P = 4 at P = 2. For a definite range [0, a]
  (h = a/2) it is 2^{2P+1} = 32.
(iii) *Mean square.* If the image has energy density w(ξ) and the errors at different ξ are uncorrelated (white
residual), then E‖Ax − A_px‖² = ∫ |E_p(χ(ξ))|² w(ξ) dξ = ∫ |e^{−is} − p(s)|² dμ(s) with μ = χ_#(w dξ). The optimum
is the **least-squares polynomial in L²(μ)**, which is a (P + 1) × (P + 1) complex normal system per sample. Taylor is
the limit μ → δ_0: all energy where χ = 0, i.e. at the FOV centre for an isocentre-centred quadratic.

So Taylor is optimal exactly when the signal sits where the chirp vanishes. It is the order-(P + 1) contact solution
at s = 0, and it is never minimax for a finite range. **All three choices cost the same at run time.**
- p(X) = Σ c_n X^n with X = −Q:∇∇ uses the same derivative tables φ^{(0..4)} as now.
- Only the per-sample coefficients γ_jα change. They become complex for every order: today order 0 and order 4 are
  real and order 2 is imaginary. In `stage_weights`, R[a] and Q[a] each gain the missing parts, and `S.ee` / the
  c[6] row become complex: +21 floats per staged sample.
- μ needs a weight w. The smoothed magnitude² of the warm start or the prescription box fits the real-data rule
  (`feedback_real_data_only`). `--b0-absorb` already uses the same weight.

Measured [CPU] (`s3_chirp_poly.py`): pointwise E over a 33³ FOV cube. Values are rms / max, P = 2:

| chirp | max\|χ\| | Taylor | centred Taylor | Chebyshev | LS (uniform w) |
|---|---|---|---|---|---|
| definite z² | 0.7 | 1.9e-2 / 5.7e-2 | 4.0e-3 / 7.1e-3 | 1.3e-3 / 1.8e-3 | 1.3e-3 / 2.4e-3 |
| definite z² | 1.4 | 1.5e-1 / 4.4e-1 | 3.1e-2 / 5.7e-2 | 1.0e-2 / 1.4e-2 | 9.9e-3 / 1.9e-2 |
| indefinite (PSD−PSD) | 0.7 | 4.5e-3 / 5.7e-2 | 3.7e-3 / 1.9e-2 | 3.6e-3 / 4.6e-3 | 1.3e-3 / 2.1e-2 |
| indefinite | 1.4 | 3.6e-2 / 4.4e-1 | 2.8e-2 / 2.0e-1 | 3.4e-2 / 4.9e-2 | 1.3e-2 / 1.9e-1 |

The 32× definite ratio of Prop. 2(ii) appears exactly: max 5.7e-2 → 1.8e-3.

End to end in 3-D, i.e. the gridded type-2 transform with the stencil against the direct chirped sum
(`s2b_weighted_poly.py`, σ 1.25, ES W 8, 1500 samples with |χ_j| spread over [0.2, 1]·max, weights from a
5-voxel-smoothed |x|²):

| chirp | max\|χ\| | Taylor (white / object) | Chebyshev | energy-weighted LS |
|---|---|---|---|---|
| indefinite | 0.7 | 1.9e-3 / 3.4e-4 | 2.0e-3 / 1.2e-3 | **8.7e-4 / 8.7e-5** |
| definite | 0.7 | 5.6e-3 / 1.7e-3 | 6.7e-4 / 3.0e-4 | **6.9e-4 / 1.7e-4** |
| definite | 1.0 | 1.5e-2 / 4.0e-3 | 1.7e-3 / 1.2e-3 | **1.7e-3 / 4.4e-4** |
| indefinite | 1.4 | 1.5e-2 / 1.6e-3 | 1.4e-2 / 4.4e-3 | **6.4e-3 / 8.2e-4** |

Chebyshev is worse than Taylor on the centred object: it spends accuracy at the FOV corners where there is no signal.
The weighted LS is never worse than either. It is 2–10× better than Taylor, and it keeps the object error below
1e-3 up to 1.4 rad. The paper's "Taylor holds for |χ| ≲ 0.7" becomes ≈ 1–1.4 rad. That covers most of the 1.5 T row
and the lower end of the 0.55 T row (1–1.4 rad) of the GNL table without the LS stencils.

### 1.3 Relation to the least-squares / POD stencil (b)

(b) minimises over *every* Γ supported on a box of half-width S around k. The derivative stencil is the 35-dimensional
subspace span{∂^α φ(k − ·)}_{|α|≤4} of the W³-dimensional space with W³ support, so (b) is better at equal support.
Its costs:
1. The support is (W + 2S)³, because Γ = φ ⋆ ĝ. S = 3–4 at 1 rad (paper table) gives 13³–18³ = 2197–5832 taps
   against W³ = 343–1000.
2. Per sample: L_p sincos, an L_p × L_p solve and an L_p × (2S + 1)³ contraction. The first two are channel-free.

Per Prop. 1, for |χ| ≤ 1.4 the derivative stencil's error is the truncation E_p, and the weighted LS p brings that
below the 1e-3 target at no extra support. (b) is therefore needed only for (i) ranges beyond ≈ 1.5 rad (0.55 T, strong
GNL), (ii) non-polynomial residual phase (degree > 4, or GNL residual ε_j above budget), and (iii) the one-FFT-per-coil
operator without segments. An intermediate option is P = 3 (degree 6 in ∂, 84 coefficients). It is not worth it:
staging grows ≈ 2.4×, while the weighted LS at P = 2 already reaches 1.4 rad.

### 1.4 Segments vs stencil: the trade-off as a function of the cost model

With the cost model of the paper (eq. tcost) per set,
  T = L·C·(F_σ + P_c) + s·M·[ C·L·κ(W)·t_m(P) + n_t(W)·t_st(P) ] + T_fixed,
- F_σ is one channel FFT pair, P_c the pad/crop per channel, s the slots per sample;
- κ(W) = (W + 3)³ is the points multiplied per (sample, slot, channel), n_t(W) = ((W + 3)/4)³ the stagings per sample
  (§5);
- t_m(0) = 2 FMA (real weights), t_m(≥1) = 4 FMA; t_st(P) grows with NK = 1, 3, 5.

A phase term θ_j costs
- in the **segments**: ΔT_S = ΔL·[C(F_σ + P_c) + s M C κ t_m], with ΔL from the union rank at tolerance ε;
- in the **stencil**, which requires θ_j polynomial and E*_P(range θ_j) ≤ ε:
  ΔT_K = s M [n_t Δt_st + C L κ (t_m(P) − t_m(P_0))], where the last term is nonzero only if the weights were real
  before (P_0 = 0).

**Rule:** stencil iff it is feasible and ΔT_K < ΔT_S, i.e. ΔL > ΔL* := ΔT_K / (T/L). Measured [meas]:
- *1 mm static (C 22, L 11):* one segment ≈ 1/L of FFT + pad + kernels ≈ 51 ms (9 %). P = 2 vs P = 0 costs 31.7 vs
  30.1 ms (+5 %) on the small test. So ΔL* ≈ 0.5: always the stencil.
- *1 mm knots, order 2 already:* the weights are complex either way, and higher-order additions cost only staging.
  ΔL* < 1: the stencil.
- *2 mm dense, order 0 (b0 only):* the real-weight MAC was the largest single forward gain of §17. Going complex
  doubles the MAC instructions (MACs ≈ 50 % of the kernel) and NK 1 → 5 adds staging, ≈ +35–45 % of the pass. One
  segment is ≈ 10 %, so ΔL* ≈ 4. If the per-sample concomitant deviation Δw_j can be absorbed by ≤ 3 extra union
  segments, segments are cheaper at 2 mm. Otherwise use the stencil. This is a measurement to make, not to assume
  (E6 in §7).
- A term with several phase cycles over the readout (B0 beyond its polynomial part) can only go into segments.
  Its polynomial part goes to the k-shift (degree 1, exact at any size) or to the stencil (degree 2, range-limited).

---------------------------------------------------------------------------------------------------------------------

## 2. The gridding kernel at fixed accuracy

### 2.1 Error functional, including the derivative terms

By Poisson summation, for a sample at fractional offset f,
  S_n(f, ξ) := Σ_m φ^{(n)}(f − m) e^{i(f−m)ξ} = Σ_p (−i(ξ + 2πp))^n φ̂(ξ + 2πp) e^{−2πipf},
so
  e_n(f, ξ) = |S_n/φ̂(ξ) − (−iξ)^n| / (π/σ)^n ≤ Σ_{p≠0} |(ξ + 2πp)σ/π|^n |φ̂(ξ + 2πp)/φ̂(ξ)|.
The worst case is at ξ = ±π/σ. The first alias, at distance 2π − π/σ, is amplified by (2σ − 1)^n. Far aliases
(|p| ≥ 2) carry the tail of φ̂, which is set by the regularity of φ at its support edge, and they are amplified by
≈ (2σ|p|)^n. Two consequences:
- e_0 decays like e^{−π(W−1)√(1−1/σ)} for ES and KB (Barnett 2021; `kernel.rs` uses this rule).
- e_3 and e_4 are governed by the edge. ES's derivatives are √-singular at |z| = 1; KB = I_0(β√(1 − z²)) is an
  entire function of z on [−1, 1] (a power series in 1 − z²), with only a jump of 1/I_0(β) at the edge.

Measured 1-D [CPU] (`s1_kernel_error.py`, 48 offsets × 241 frequencies, rms). Full max and rms tables for σ ∈ {1.125,
1.25, 1.5, 2} and W 4–12 are in `s1_out.md` / `s1_kernel_error.csv`.

| σ | W | ES e0 / e2 / e4 | KB e0 / e2 / e4 | per-order LS taps e0 / e2 / e4 |
|---|---|---|---|---|
| 1.125 | 8 | 2.6e-4 / 1.5e-3 / 8.1e-3 | 1.8e-4 / 1.9e-3 / 1.2e-2 | 2.3e-4 / 4.3e-4 / 1.4e-3 |
| 1.25 | 6 | 4.5e-4 / 3.0e-3 / 2.0e-2 | 2.6e-4 / 3.2e-3 / 2.0e-2 | 4.0e-4 / 1.2e-3 / 8.8e-3 |
| 1.25 | 7 | 9.4e-5 / 8.3e-4 / 6.9e-3 | 6.3e-5 / 9.4e-4 / 7.6e-3 | 8.7e-5 / 3.1e-4 / 2.4e-3 |
| 1.25 | 8 | 2.4e-5 / 4.0e-4 / 3.9e-3 | 1.5e-5 / 2.6e-4 / 2.9e-3 | 1.8e-5 / 6.7e-5 / 6.9e-4 |
| 1.25 | 10 | 2.3e-6 / 4.1e-5 / 6.2e-4 | 9.5e-7 / 1.8e-5 / 3.5e-4 | 7.7e-7 / 2.9e-6 / 5.0e-5 |
| 1.5 | 5 | 3.8e-4 / 3.9e-3 / 7.6e-2 | 2.2e-4 / 5.3e-3 / 4.2e-2 | 3.4e-4 / 2.1e-3 / 2.4e-2 |
| 1.5 | 6 | 5.2e-5 / 1.1e-3 / 1.5e-2 | 3.5e-5 / 9.6e-4 / 1.3e-2 | 4.8e-5 / 3.0e-4 / 5.1e-3 |
| 2.0 | 5 | 7.4e-5 / 2.7e-3 / 9.0e-2 | 4.4e-5 / 2.8e-3 / 7.0e-2 | 6.6e-5 / 7.2e-4 / 1.8e-2 |

- The σ = 1.25, W = 10 ES row reproduces the paper's "value 2e-6, ∂² 3e-5".
- **KB is 1.5–2× better than ES at equal W** (rms e0). A β scan over 0.8–1.2× (`s1b_beta_scan.py`) improves ES by at most
  1.6× at W = 5 and not at all for W ≥ 7, while Beatty's KB rule is already at its optimum.
- **The ES derivative clamp matters a lot.** Unclamped, the max-norm e_4 at σ = 1.25 is 43 (W = 7) and 0.77 (W = 10).
  Clamped at |z| > 0.98 (production) it is 6.5e-2 / 7.0e-3. KB needs no clamp: 5.0e-2 / 3.5e-3. The clamp also costs
  e_1 accuracy (W = 10 max e_1 9.2e-5 clamped vs 2.7e-5 unclamped).
- **Per-order LS taps** (Fessler–Sutton type: for each offset and each derivative order, the W real taps that minimise
  the relative error with the ES deapodisation) equal ES / KB on the value. That is expected, since ES ≈ KB ≈ prolate
  are near-optimal for aliasing. They are 3–6× better on e_2 / e_4. This is the floor for a W-tap kernel, and it only
  matters if the derivative terms dominate the budget, which they do not at |χ| ≤ 0.7 (§2.2).

### 2.2 3-D, with the derivative stencil

`s2_3d_check.py`: chirped type-2 transform of a white and a smooth object image, N = 32³, 1500 random samples, f64
direct sum. Order-2 stencil with per-sample Q_j (|χ_j| spread over [0.2, 1] · max). Relative errors are white / object:

| σ | W | plain (χ = 0) ES | plain KB | indefinite χ ≤ 0.7, Taylor, ES | same, KB | floor (KB, σ 2, W 12) |
|---|---|---|---|---|---|---|
| 1.25 | 6 | 1.1e-3 / 3.3e-4 | 5.0e-4 / 3.3e-4 | 2.3e-3 / 4.3e-4 | 2.1e-3 / 6.3e-4 | 1.8e-3 / 1.2e-4 |
| 1.25 | 7 | 2.2e-4 / 7.4e-5 | 1.2e-4 / 9.0e-5 | 1.8e-3 / 2.2e-4 | 1.8e-3 / 3.0e-4 | |
| 1.25 | 8 | 3.8e-5 / 1.4e-5 | 2.6e-5 / 9.6e-6 | 1.8e-3 / 1.2e-4 | 1.8e-3 / 1.2e-4 | |
| 1.25 | 10 | 3.7e-6 / 5.4e-7 | 1.2e-6 / 3.1e-7 | 1.8e-3 / 1.2e-4 | 1.8e-3 / 1.2e-4 | |
| 1.5 | 5 | 7.4e-4 / 6.0e-4 | 4.0e-4 / 5.4e-4 | 2.6e-3 / 2.7e-3 | 2.4e-3 / 7.2e-4 | |
| 1.5 | 6 | 1.1e-4 / 7.0e-5 | 6.4e-5 / 7.8e-5 | 1.8e-3 / 9.8e-4 | 1.8e-3 / 2.1e-4 | |
| 2.0 | 4 | 1.4e-3 / 1.3e-3 | 7.6e-4 / 1.1e-3 | 1.5e-2 / 9.2e-3 | 7.2e-3 / 3.8e-3 | |
| 2.0 | 5 | 1.4e-4 / 2.0e-4 | 7.9e-5 / 1.6e-4 | 2.6e-3 / 7.1e-4 | 2.3e-3 / 7.5e-4 | |

Readings:
1. The 3-D white error is ≈ 2× the 1-D rms, as expected from ε_3D ≈ ε_x + ε_y + ε_z.
2. **From W = 7 at σ = 1.25 the chirped error equals the W = 12 floor.** At W ≥ 7 the derivative terms are not the
   bottleneck; the truncation E_p is, and §1.2 addresses that.
3. At W = 6 the ES derivative terms show (object 4.3e-4 against a 1.2e-4 floor); KB at W = 6 is similar.
4. σ = 1.5 with W = 5 ES shows the clamp penalty (2.7e-3 object); KB at 7.2e-4 does not.
5. Kernel budget: if the 1e-3 operator target is split as kernel ≤ 3e-4, segments ≤ 5e-4 and stencil ≤ 2e-4, the
   kernel needs W = 7 at σ = 1.25 (ES or KB) or W = 6 at σ = 1.5. With kernel ≤ 1e-3, KB W = 6 at σ = 1.25 suffices.
   W = 10 is 50–100× more accurate than needed.

### 2.3 Which kernels and evaluators

- **ES** (FINUFFT): cheap to evaluate (exp + sqrt), but its derivatives need the clamp. At equal W it is ≈ ½ tap
  worse than KB in our σ range.
- **KB** with Beatty's β = π√((W/σ)²(σ − ½)² − 0.8): the best closed form here, with bounded derivatives. Direct
  evaluation needs I_0 … I_4, which is too expensive per tap. Evaluate it as FINUFFT does, with a piecewise polynomial
  in the fractional offset (Horner, degree ≈ W + 2 per tap), and differentiate that polynomial for φ^{(1..4)}. The
  same tables can hold *any* kernel, including the per-order LS taps.
- **Min-max (Fessler–Sutton)** and **per-order LS taps**: optimal for a given deapodisation, but tabulated per offset.
  They equal KB on the value and are 3–6× better on ∂²/∂⁴. Worth it only if a regime appears where e_2/e_4 dominate,
  e.g. large χ at small W.
- **Small-σ kernels (Beatty 2005)**: their rule is the KB above; σ = 1.125–1.375 is their recommended range, which
  matches the cost optimum here.
- **Gaussian**: needs 1.5–2× the taps; it only pays if closed-form chirp stencils are needed
  (`concomitant_stencils_and_kernels.md` §3.4).

### 2.4 (σ, W) against the measured time model

`s8_cost_model.py`. The FFT scales as σ³ times the pruned-traffic factor (1/3)(h² + h + 1)(1 + h)/2 with h = 1/σ
(paper, pruned grid FFT), relative to σ = 1.25: 0.84 (1.125), 1.38 (1.5), 2.45 (2). Kernel time scales as (W + 3)³;
MACs and stagings both scale this way for 4³ tiles (§5). The fraction of per-tile work that does not depend on W is
taken as 0.5 / 0.25 / 0.1 for the three regimes. W(σ) is the 3-D requirement from §2.2.

| regime (measured base) | best at kernel ≤ 3e-4 | best at ≤ 1e-3 | σ = 2 |
|---|---|---|---|
| 1 mm static, 0.56 s (FFT 64 %) | σ 1.125, W 9: 0.49 s (1.15×) | σ 1.125, W 8: 0.48 s (1.18×) | 0.55× |
| 1 mm knot cubic 10, 10.2 s | σ 1.25, W 7: 7.7 s (1.32×) | σ 1.25, W 6: 7.2 s (1.42×) | 0.86× |
| 2 mm 2-D knots, 13.7 s (kernels 73 %) | σ 1.5, W 6: 10.8 s (1.27×); σ 1.25, W 7: 11.4 s (1.20×) | σ 1.5, W 5: 9.3 s (1.47×); σ 1.25, W 6: 9.6 s (1.42×) | 0.9–1.2× |

The prompt's intuition, "sparse → small σ, dense → small W", holds, but the gains are modest because the two terms
trade against each other. Most of the gain comes from W = 10 being over-specified: W 10 → 7 at fixed σ = 1.25 is
already 1.08 / 1.32 / 1.20× with no code beyond a define. σ should be planned per set, like the launch shapes. The FFT
sizes must stay 2^a3^b5^c7^d (σ = 1.125 at 1 mm: 224 × 216 × 256).

### 2.5 Not applicable

- **Ruiz-Antolín & Townsend's low-rank NUFFT** (σ = 1, the non-uniformity as a rank-K perturbation): needs K^d FFTs
  of the unpadded grid, i.e. ≈ 4–6 per axis at 1e-3, which is 64–216 in 3-D against 1.95. A 1-D idea.
- **Exact type-3 / butterfly**: the paper's butterfly measurement already shows the total basis count grows.

---------------------------------------------------------------------------------------------------------------------

## 3. Can a smarter kernel reduce L or C?

### 3.1 L: frequency segmentation, MFI, k-space B0 kernels

**Proposition 4 (no separable scheme beats the SVD).** Let K(r, τ) = e^{−iψ(r,τ)} on Ω_s × [0, T].
- Time segmentation is K ≈ Σ_ℓ D_ℓ(r)a_ℓ(τ).
- Frequency segmentation / MFI (Noll 1991, Man 1997) is K ≈ Σ_ℓ b_ℓ(ω(r))e^{−iω_ℓτ}, i.e. image-domain weights times
  per-sample demodulations.

Both are rank-L separable approximations, so by Eckart–Young both are bounded below by the truncated SVD of K, which
the code already uses (`--seg-basis svd*`). In an iterative operator both cost L channel FFTs. MFI's historical
advantage was computational, inside a one-pass conjugate-phase recon. ∎

Only non-separable structure can beat that rank:
1. *Polynomial field parts* as per-sample k-space terms. Degree 1 is an exact shift at any magnitude; degree 2 is the
   chirp, range-limited (§1); degree 3–4 is first order only. `--b0-absorb` does this, and §12 of temporal_operator.md
   measured the union rank dropping from 23–32 to 16–24 when degree ≤ 4 is moved out. The weighted LS p of §1.2 raises
   the admissible degree-2 range, so more of the field can leave the segments.
2. *A k-space B0 kernel* F[e^{−iωτ}x] = (e^{−iωτ})^ ⊛ x̂ is compact only for smooth fields. Its spectral half-width is
   ≈ τ·max|∇ω|·FOV/2π cells. Susceptibility near lung and heart (≈ 1 ppm/cm, 128 Hz/cm at 3 T) gives ≈ 20 cells at
   τ = 8 ms, which is not compact. This is the LS stencil (b) with a τ-dependent ĝ. It is viable only for the smooth
   residual, which is exactly what item 1 handles more cheaply.
3. *Sample-local bases* (butterfly): the total count rises (paper: 42 → 51–132).

So **L is set by the non-polynomial part of the field over the readout** (`shimming_direction`: only local shimming
lowers it). The kernel cannot help further.

### 3.2 C: coil maps as compact k-space kernels

y_c = Σ_ℓ a_ℓ Γ_j ⋆ (Ŝ_c ⊛ F[P_ℓx]) needs L FFTs instead of C·L. The price is a per-(sample, coil) stencil
Γ_j ⋆ Ŝ_c of support (W + K − 1)³, and the per-sample work becomes C·L·(W + K − 1)³ complex MACs plus building the
combined weights. Only S_c on the image box matters, so the best K is a *Fourier-extension* question: a trig
polynomial of period σ·FOV fitted on the box.

Measured [CPU] (`s5_coil_kspace.py`): 32 Biot–Savart loops (r = 40 mm, 4 rings × 8) on a 170 × 115 mm elliptic
cylinder, the 1 mm thorax box 198 × 189 × 228 mm (anterior/posterior coils ≈ 2 cm outside the box), LS fit over the
body inside the box. 30 virtual coils at 99.9 %, and they behave the same:

| K taps per axis (σ = 1.25) | 5 | 9 | 13 | 17 | 25 |
|---|---|---|---|---|---|
| energy-weighted rms error | 0.34 | 0.12 | 3.9e-2 | 1.7e-2 | 2.9e-3 |

(σ = 2 is no better. Its K = 25 row is an ill-conditioned extension, so it is not shown.)

Extrapolating the decay (×≈ 6 per 8 taps), ≤ 1e-3 needs K ≈ 29–31. Even at K = 13 (4 %), the 1 mm static cost [model] is:
- FFT 356 → 16 ms, pad / crop ≈ 57 → 5 ms;
- kernels 150 ms × (W + K − 1)³/(W + 3)³ × (L + R)/L. With W = 7, K = 13 that is 6859/1000 ≈ 6.9, and R ≈ 3–6 is the
  rank of Ŝ_c needed to build the weights. The result is ≈ 1.3–1.6 s, against 0.55 s now.

Dense regimes are worse still. **Rejected.** C is reduced only by compression (≥ 99.9 %) and stochastic coil subsets,
both done.

### 3.3 GNL / concomitant chirp in the kernel

That is the stencil itself (§1). Cost [meas]: free in the sparse regime (shared across C·L channels). At 2 mm,
switching on order ≥ 1 costs ≈ +35–45 % (§1.4). Order-3 GNL terms (degree 3–4 to first order) add NK = 5 staging but
no MACs, since the weights are already complex.

---------------------------------------------------------------------------------------------------------------------

## 4. The normal operator

### 4.1 Cost of the segment-pair Toeplitz form

  N = Σ_c S̄_c Σ_{ℓℓ'} D̄_ℓ T_{ℓℓ'} D_{ℓ'} S_c,  T̂_{ℓℓ'} = FFT_{2N}[Σ_j ā_ℓ(τ_j)a_{ℓ'}(τ_j)e^{ik_j·Δ}]  (no chirp).

Per application and per coil: L forward and L inverse FFTs on the 2N grid, and L² complex MACs per κ.
- **Memory at 1 mm:** the 2N grid is 396 × 378 × 456 = 68.3 M points, 0.546 GB per kernel (complex64). L = 11 gives
  66 kernels = 36 GB (18 GB in fp16). It does not fit 24 GB alongside anything.
- **FFT bytes:** per channel the 2N grid is 3.95× the σ = 1.25 grid. With pruning (h = ½: 0.44 vs 0.73) that is 2.4×
  per channel FFT → ≈ 840 ms for the 2·C·L channels, against 356 ms.
- **Kernel reads:** 36 GB per coil batch. Even at one read per coil (6 GB of channels per coil) this is
  22 × 36 GB ≈ 0.8 s. Total ≈ 1.7 s, against 0.55 s for our forward + adjoint. **The NUFFT pair wins at 1 mm.**
- **Dense 2 mm, static set** (all 30.3 M samples of an encode on one image, C = 8, L = 10): the 2N grid has 33.4 M
  points, giving 160 pruned channel FFTs ≈ 0.14 s. The 55 kernels take 14.7 GB (fp16: 7.3 GB), read about 4 × per pass
  ≈ 0.06 s, so ≈ 0.25 s in total. The NUFFT pair spends ≈ 1.3 s on per-sample work. **≈ 5× for static dense sets:**
  warm starts, per-encode time averages, the breathing-bin images of `--estimate-motion`. The library already has the
  fused pruned convolution (hasty-fft F4).
- **Knots:** the coefficient-space normal operator is banded in the knots, so per knot pair 2p + 1 diagonals × L² →
  hundreds of 2N-grid kernels per unit. The FFT count is the same as knot-major NUFFT (K images × 2CL) on 4× larger
  grids. **Never cheaper.** The Kronecker surrogate G ⊗ N_s (§19) costs K · N_s ≈ 60 × 0.25 s = 15 s per application
  at 2 mm, against the exact 13.7 s, so it is no good as a lagged control-variate surrogate either.

### 4.2 Compressing the L × L kernel matrix per k-point (measured: does not work)

The idea: τ is nearly a function of |k| along an outward trajectory, so T̂(κ) should be ≈ rank 1, a(τ(κ))a(τ(κ))ᴴ.
`s4_toeplitz_rank.py` (2-D, N = 64, 24-interleave spiral, 8 ms, 3.6 phase cycles, LS interpolators) gives:

| L | seg-Toeplitz vs exact | rank 1 | rank 2 | rank 3 | rank 4 | parametric rank 1 |
|---|---|---|---|---|---|---|
| 9 (white / object) | 9.5e-4 / 1.3e-3 | 0.58 / 0.14 | 0.11 / 6.9e-3 | 6.0e-2 / 5.8e-3 | 3.5e-2 / 2.5e-3 | 3.5 / 0.16 |
| 12 | 1.2e-4 / 1.2e-4 | 0.51 / 0.16 | 0.22 / 1.6e-2 | 7.0e-2 / 4.4e-3 | 3.8e-2 / 3.6e-3 | 2.7 / 0.18 |

The eigenvalue mass captured at rank 1 is only 60–80 % (median). The cause: T̂(κ) is the sample measure convolved with
the Dirichlet kernel of the lag box [−N, N), whose 1/κ tails mix samples from distant τ. Only one lag of the 2N
period is free, so the tails cannot be apodised away without a larger embedding. **Rejected.** The rank-R storage
(L·R per κ) is also no smaller than the pair kernels until R < (L + 1)/2.

### 4.3 Near-exact alternatives

- Fessler's histogram ("L-kernel") interpolators make T_{ℓℓ'} = 0 for ℓ ≠ ℓ'. They need a much larger L for 1e-3,
  because the histogram error is ∝ ΔfT/L.
- Banded B-spline interpolators give (p + 1)L kernels at a larger L. At our L this is ≈ the same 60–70 kernels.
- The Weyl/commutator forms (paper §weyl) handle the chirp with 4–35 FFT pairs, but they are coil-outer and segment-free.

For the iteration, the existing lagged control variate with a reduced-L surrogate remains the right "cheap N". Its
surrogate is the same NUFFT with L_s < L.

---------------------------------------------------------------------------------------------------------------------

## 5. Tile and work layout: the floor

Per sample, per slot, per channel, gather or scatter. `s7_tile_counts.py`, Monte Carlo over uniform offsets:

| W | stagings / sample (TT 4) | dense 4³ loop | 2 × 2 × 2 sub-box warps | exact box = W³ | useful: dense / sub-box |
|---|---|---|---|---|---|
| 5 | 8.0 | 512 | 216 | 125 | 0.24 / 0.58 |
| 6 | 11.4 | 728 | 343 | 216 | 0.30 / 0.63 |
| 7 | 15.6 | 999 | 512 | 343 | 0.34 / 0.67 |
| 8 | 20.8 | 1329 | 728 | 512 | 0.39 / 0.70 |
| 10 | 34.3 | 2197 | 1331 | 1000 | 0.46 / 0.75 |

The dense count is exactly (W − 1 + TT)³. TT = 8 cuts the stagings 3–4× but makes the dense loop 7–15 % useful, and it
needs a sub-box loop and more shared memory (it did not fit at L = 11, §15). The W = 8 sub-box count, 728 = 35 per
(sample, tile), matches §17's 36.

**Arithmetic floor.**
- *MACs:* s·C·L·W³ complex MACs. That is 2 FMA with real weights; with complex weights it is 4 FMA on the forward
  and, in the adjoint, 4 for z_c·w̄. The derivative coefficients add nothing per channel.
- *Operands:* the fused-multiply-add peak (≈ 41 T FMA/s on the 4090) needs ≥ 8 FMA per 8-byte shared-memory load.
  A complex MAC with a real weight and one tile-value load does 2. Without register reuse, the shared-memory
  bandwidth (≈ 41 TB/s) caps the MACs at ≈ ¼ of FMA peak. Point-lane layouts, with channels in registers and one
  weight load amortised over CPB·L channels, are needed to approach the FMA floor. The adjoint variant 1 does this;
  the forward does not yet (§17 item 2).
- *Staging:* once per sample, the floor is ≈ 3W·NK kernel values + L·s lerps + 1 sincos ≈ 10² FLOP, which is ≈ 1 %
  of the MACs. The measured 25 % is repetition (11–34 × per sample) and barrier latency. Double-buffering (§17 item 1)
  attacks the latency; fewer tiles per sample (smaller W, bigger tiles) attacks the repetition.
- *Grid traffic:* each touched grid point is read once per channel group. At 1 mm static that is the k-ball (≈ 27 %
  of the σ-grid) × 242 channels × 8 B ≈ 9 GB, ≈ 9 ms per direction, against FFT traffic ≈ 200 GB per direction.

**Where the time is against the floor** [model, 70 % FMA peak]:
- 2 mm dense, W = 8, order 0: the floor is 30.3 M × 8 × 80 × 512 × 2 FMA ≈ 2.0e13 per direction → 0.69 s, 1.4 s
  for both directions, against ≈ 10 s measured. The 7× splits as 2.6 (zero-weight points, 1329/512) × ≈ 2
  (shared-memory operand bound, latency) × 1.3 (staging).
- 1 mm knot cubic, W = 10, order 2: the floor is 1.8 s, against 6.1 s.
- 1 mm static: the floor is ≈ 10–20 ms, against 150 ms. That case is latency- and occupancy-bound with few samples
  per tile, and the FFT (356 ms, at bandwidth) dominates anyway.

**Separable / per-axis passes** (contract z, then y, then x) cost W³ + W² + W MACs per sample and channel. That equals
the dense contraction, because partial sums cannot be shared between samples with different offsets. They save only
the weight formation, which is already channel-amortised. Not a lever. FINUFFT's Horner kernel evaluation helps
staging, not MACs.

---------------------------------------------------------------------------------------------------------------------

## 6. Proposals in detail (ranked)

1. **Shrink W.** Production: W = 7 at σ = 1.25, 1 mm, order 2; W = 7 (3e-4 budget) or 6 (1e-3) at 2 mm.
   - Expected [model]: 1 mm static 0.56 → 0.52 s, 1 mm knot cubic 10.2 → 7.7 s, 2 mm 13.7 → 11.4 s (W 7) or 9.6 s
     (W 6, KB).
   - Accuracy [CPU 3-D]: kernel 2.2e-4 / 7e-5 (ES W 7). The chirped error equals the W = 12 floor.
   - Effort: W and BETA are compile-time defines; the tests need re-baselined tolerances (seg_* direct-sum bounds).
2. **Energy-weighted LS chirp polynomial** (Prop. 2(iii)).
   - Accuracy: 2–10× lower stencil error at the same cost; the P = 2 range goes from 0.7 to 1–1.4 rad. That defers the
     "LS stencils inside segments" item and keeps more of the field polynomial out of the segments (§3.1).
   - Effort: host-side per-sample (P + 1)-dimensional LS over a probe set of the FOV (≈ 9³ points, weighted by the
     smoothed warm-start |x|²); complex coefficient rows in `S.ee` / `rec`. The minimax (Chebyshev) variant is the
     fallback when no weight is available.
3. **fp16 grid storage with fp32 arithmetic** (FFT passes, tile-value loads, adjoint grid writes).
   - Accuracy [CPU, `s6_half_fft.py`, σ 1.25, 100³ grid]: 4.0e-4 over the grid, 3.1e-4 inside the k-ball, 1.8e-4
     with centre-dense sample weighting. bf16 gives 1.5–3e-3, too much.
   - Expected: FFT traffic ×0.5 → 1 mm static ≈ 0.56 → 0.38 s. The tile-value loads halve too, which helps the knot
     regimes ≈ 1.1×.
   - Catch: it uses up ≈ 20–40 % of the 1e-3 budget, so combine it with W = 7, not W = 6. Accuracy study at recon level
     first (held-out k-space residual).
4. **k-ball pruning of the last forward / first inverse FFT pass.** Only lines meeting the sample ball plus the W/2
   halo are needed: ≈ 55 % of (x, y) lines at σ = 1.25 in the z pass, and 84 % in the y pass. Saves ≈ 18–20 % of FFT
   traffic, exactly → 1 mm static ≈ 1.1×. The tile lists already know which tiles have samples.
5. **KB through piecewise-polynomial (Horner) tables.** About ½ tap: W = 6 KB ≈ W = 7 ES at 5e-4, with bounded
   derivatives and no clamp. The same evaluator can carry per-order designed taps later.
6. **Point-lane forward for complex weights** (§17 item 2), dense 999 → sub-box 512 points at W = 7. Up to 1.9× on the
   forward MACs at 1 mm; it is the remaining factor 2.6 → 1.45 of §5.
7. **σ per set.** σ = 1.125 with W = 8–9 for FFT-bound static sets (1.15–1.18×). It gains nothing for kernel-bound
   sets. Let the planner and autotune choose, as with TileCfg.
8. **Segment-pair Toeplitz normal operator for static dense sets only** (≈ 5× there; 7–15 GB of kernels). Not for
   knots or time-resolved sets.

Combined, 1 + 3 + 4 at 1 mm static: FFT 356 → ≈ 150 ms and kernels 150 → 110 ms, so ≈ 0.56 → 0.32 s (1.7×). At 2 mm,
1 + 6 together with the staging overlap of §17 should take the 13.7 s pass to ≈ 7–8 s [model]. Neither total has been
measured.

---------------------------------------------------------------------------------------------------------------------

## 7. What to try first (pre-registered)

Per `feedback_experiment_discipline`: one factor at a time, dev scale first, config fingerprint checked (σ, W, BETA,
NSEG, CORR, KV in the kernel define string; circ metric unchanged).

| id | question → decision | run | prediction (falsifiable) | accept if |
|---|---|---|---|---|
| E1 | Is W = 7 accurate and faster on GPU? → production W | opbench matrix (§11) at W ∈ {7, 8, 10}, σ 1.25, ES; seg_* direct-sum tests | time ×0.92 static 1 mm, ×0.76 knot cubic 1 mm, ×0.84 2 mm (W 8 → 7); operator error ≤ 3e-4 vs f64 | error ≤ 3e-4 and ≥ 1.15× on both knot regimes |
| E2 | Does the weighted LS polynomial beat Taylor on the real model? → coefficient rule | CPU emulation of the paper's full-model GNL table (2.89 / 1.5 / 0.55 T), same stencil code | ≥ 2× lower error at 2.89 T and 1.5 T; 0.55 T below 1e-2 | ≥ 2× at 2.89 T, no case worse than Taylor |
| E3 | KB-Horner at W = 6 vs ES W = 7 → kernel family | same tests as E1 | KB W 6 error ≤ 5e-4, time = ES W 6 | equal error to ES W 7 at W 6 time |
| E4 | fp16 grid: FFT gain and error → storage format | FFT microbench (180 × 180 × 50 × 16 as in the paper) + seg tests + one 2 mm recon with held-out residual | FFT ×0.55, operator error ≤ 4e-4, held-out residual within 1 % | as predicted, with no change in the PnP fixed point beyond noise |
| E5 | k-ball pruning → FFT plan | FFT microbench with the tile-occupancy mask | forward + inverse ×0.82, results equal to 1e-7 | ≥ 10 % on static 1 mm |
| E6 | Concomitant Δw_j: union segments or stencil at 2 mm? → physics placement | union rank of (τ, Δw) at ε = 1e-3 (`b0_union_rank.py`-style) + 2 mm time at order 0 vs 2 | ΔL ≤ 3 → segments; else stencil (§1.4 rule, ΔL* ≈ 4) | the measured ΔL vs the measured order-2 overhead |

Run order: E1 (no code) → E2 (CPU only) → E3 → E5 → E4 → E6.

---------------------------------------------------------------------------------------------------------------------

## 8. Literature

[V] = read in full this session. [A] = abstract / metadata checked this session. [R] = recalled, not re-checked this
session. No paper was read in full this session.

- A. H. Barnett, J. Magland, L. af Klinteberg. A parallel nonuniform FFT library based on an "exponential of
  semicircle" kernel. SIAM J. Sci. Comput. 41(5):C479–C504, 2019. doi:10.1137/18M120885X, arXiv:1808.06736 [A].
  ES kernel, Horner piecewise-polynomial evaluation, width rule.
- A. H. Barnett. Aliasing error of the exp(β√(1−z²)) kernel in the NUFFT. Appl. Comput. Harmon. Anal. 51:1–16, 2021.
  arXiv:2001.09405 [A]. Rate arbitrarily close to KB's.
- Y.-h. Shih, G. Wright, J. Andén, J. Blaschke, A. H. Barnett. cuFINUFFT: a load-balanced GPU library for
  general-purpose nonuniform FFTs. IPDPS-PDSEC 2021. arXiv:2102.08463 [A]. Point reordering, shared-memory blocked
  spreading: each point is staged once and padded bins are added atomically, the dual of our gather tiles.
- P. J. Beatty, D. G. Nishimura, J. M. Pauly. Rapid gridding reconstruction with a minimal oversampling ratio. IEEE TMI
  24(6):799–808, 2005. doi:10.1109/TMI.2005.848376 [A]. KB β rule for σ = 1.125–1.375.
- J. I. Jackson, C. H. Meyer, D. G. Nishimura, A. Macovski. Selection of a convolution function for Fourier inversion
  using gridding. IEEE TMI 10(3):473–478, 1991. doi:10.1109/42.97598 [A]. KB ≈ PSWF, sometimes better.
- J. A. Fessler, B. P. Sutton. Nonuniform fast Fourier transforms using min-max interpolation. IEEE TSP 51(2):560–574,
  2003. doi:10.1109/TSP.2002.807005 [R]. Per-offset optimal taps for a given scaling, i.e. the "ls-floor" of §2.1.
- D. Potts, M. Tasche. Uniform error estimates for nonequispaced fast Fourier transforms. Sampl. Theory Signal Process.
  Data Anal. 19:17, 2021. arXiv:1912.09746 [A]; and Continuous window functions for NFFT, arXiv:2010.06894 [A].
  Explicit KB / sinh-type bounds.
- D. Ruiz-Antolín, A. Townsend. A nonuniform fast Fourier transform based on low rank approximation. SIAM J. Sci.
  Comput. 40(1):A529–A547, 2018. doi:10.1137/17M1134822, arXiv:1701.04492 [A].
- B. P. Sutton, D. C. Noll, J. A. Fessler. Fast, iterative image reconstruction for MRI in the presence of field
  inhomogeneities. IEEE TMI 22(2):178–188, 2003. doi:10.1109/TMI.2002.808360 [R].
- J. A. Fessler, S. Lee, V. T. Olafsson, H. R. Shi, D. C. Noll. Toeplitz-based iterative image reconstruction for MRI
  with correction for magnetic field inhomogeneity. IEEE TSP 53(9):3393–3402, 2005. doi:10.1109/TSP.2005.853152 [R].
- L.-C. Man, J. M. Pauly, A. Macovski. Multifrequency interpolation for fast off-resonance correction. MRM
  37(5):785–792, 1997. doi:10.1002/mrm.1910370523 [A].
- D. C. Noll, C. H. Meyer, J. M. Pauly, D. G. Nishimura, A. Macovski. A homogeneity correction method for magnetic
  resonance imaging with time-varying gradients. IEEE TMI 10(4):629–637, 1991. doi:10.1109/42.108599 [R].
- B. J. Wilm, C. Barmet, K. P. Pruessmann. Fast higher-order MR image reconstruction using singular-vector separation.
  IEEE TMI 31(7):1396–1403, 2012. doi:10.1109/TMI.2012.2190991 [A]. The SVD-of-phase idea behind our segment basis.
- C. A. Baron, N. Dwork, J. M. Pauly, D. G. Nishimura. Rapid compressed sensing reconstruction of 3D non-Cartesian MRI.
  MRM 79(5):2685–2692, 2018. doi:10.1002/mrm.26928 [A]. Toeplitz beats NUFFT on GPU for dense 3-D sets without field
  terms, which is consistent with §4.1.

Internal: `research/concomitant_stencils_and_kernels.md` (§3.1 LS stencil widths, §3.4 kernel design),
`research/temporal_operator.md` §9–§17, `research/experiments/{stencil_width2,kernel_design,pair_rank}.py`.

**Open:**
- All accuracy numbers are dev scale (N = 32³ 3-D, 2-D Toeplitz). GPU f32 adds ≈ 1e-6.
- The coil model is idealised (no tissue loading, no cable or decoupling effects). Real maps are unlikely to be more
  compact, though.
- The cost model extrapolates from the measured ablations; E1 tests it directly.
