# A structured k-space normal operator for the knot-resolved 4D-flow model? (2026-11-18)

Question (user): naive Toeplitz needs L or L² kernels per knot / frame / encode, each 8× the image grid, i.e. several
TB. Could one 2×-oversampled FFT, one or two 2× kernels (density or "coordinates") and cheap local k-space work
(stencils, finite differences, small convolutions) apply AᴴA, with GNL, the rate map (B0 + R2* + fat) and the
concomitant fields, faster than today's gridding forward + adjoint at the target config, in realistic memory?

Continues `RESEARCH_fast_normal_operator.md` (Fessler O(L), regeneration, modulated rows),
`concomitant_weyl_commutator.md` (commutator calculus), `concomitant_stencils_and_kernels.md`,
`concomitant_segmented_first_order.md` (§5 already argued "NUFFT form for time-resolved sets") and
`temporal_operator.md` §13 (banded knot Gram), §16–§22 (2-D knots, 1 mm timings). Scripts and outputs:
`research/experiments/normal_op/` (CPU only; data = ph0001 `sim_flow_10min/scan`, the 2 mm yarn 4D-flow sim,
encode 0: 7592 readouts × 3990 ADC = 30.3 M samples, real k, cardiac phase and bellows per readout).

## 0. Verdict

1. **At the target config (1 mm, 96 knots = 6 linear resp × 16 cubic cardiac, 30 coils, L = 10, order 3) no
   Toeplitz-type normal operator beats the current gridding pair, and none fits in memory.**
   - The FFT count cannot drop below 2·C·L·K transforms of the 2N grid (§1.2). That grid has 4.1× the cells of our
     σ = 1.25 grid; with pruning the image side costs ≈ 3× (2.6–4.1×) today's.
   - Today the image side is 25 s of 145 s per encode pass on the 160 mm box and 651 s of 1260 s on the torso
     (`temporal_operator.md` §22.6). So the torso is lost on the image side alone (≈ 1950 s > 1260 s). On the box the
     ceiling is ≈ 145 / 75 ≈ 1.9×, reached only if the k-space part were free.
   - It is not free. The knot-pair multipliers do not compress (item 2); the smallest exact set is **704 knot-pair
     functions × 2 readout halves = 1408 real kernels per encode = 185 GB at the 1 mm box (4.3 TB torso)** with
     segments as pointwise weights, and 38 720 kernels (5.1 TB box) with exact segment pairs. Streaming or
     regenerating them per coil costs 220–290 s per encode pass on the box: **2–2.7× slower than today**.
2. **The knot Gram is not separable** (Q1, measured on the real acquisition order). T_kk′(u) ≈ Σ_r G⁽ʳ⁾_kk′ K_r(u) needs
   R ≈ 220–370 of the 704 independent pair functions for 1e-2…1e-4 (Frobenius, white image), 85–255 with a k⁻⁴
   image spectrum, and ≈ 500 for 1e-4 in the per-cell operator norm. This holds at every density from 1 mm (1.8 samples
   per 2N cell) to the 4 mm pyramid level (25 per cell). Cause: the incoherent time ordering makes every cell's knot
   matrix a random mixture of acquisition times. Its deviation from separable is ≈ √(κ/n_eff) with κ = 47, so 1e-4
   would need n_eff ≈ 5·10⁹ distinct readouts per cell. A cell sees at most 2 × 7592.
3. **Phase physics** (Q2):
   - Free in Toeplitz form: GNL (the image grid is already the apparent grid), the per-sample constant phase φ0 (cancels
     in AᴴA), and linear terms (k-shift of the kernel's samples).
   - The readout time is a function of k per readout half: the yarn goes out and back, |k(τ)| identical for all
     rotations to 4e-8. So segment-pair and fat weights can be pointwise k-space multipliers per half ("coordinate
     kernels"). But this is only accurate to 5e-3–1e-2 on the 2N grid, falling like 1/(guard band) (2-D test, §3.3), so
     it is not an exact 1e-4 operator.
   - The B0 field needs L = 15–16 segments for 1e-4 (14 for 1e-3) over the box and the torso; R2* adds ≤ 1. Removing its degree ≤ 4
     polynomial part does not lower L (15 → 15): the field is local susceptibility structure. Fessler's diagonal form
     needs L′ = 16–19.
   - The per-sample concomitant residual (0.42 rad max in the heart box, 1.15 rad on the torso FOV) is only ≈ 40 % a
     function of k. It needs its own moment kernels per knot pair (≥ 6 at first order) or stays per sample.
4. **Where a structured normal operator does win** (same cost model, §4):
   - the **static (K = 1) operator**: exact Tucker Toeplitz with 55 segment-pair kernels, 7 GB at the 1 mm box. It is
     ≈ 7–9× faster than the gridding pair there. On the torso only the approximate pointwise form fits (2 kernels, ≈ 24 GB on a 32 GB card, ≈ 2.5–3×); the exact one needs 166 GB;
   - **coarse pyramid levels** (≤ 0.5 M voxels, i.e. 2 mm / 4 mm box): 1–3× for the knot operator, kernels
     2.9–23 GB per encode.
   Both are the dense regime, which the literature (Baron 2018) also finds favourable for Toeplitz.
5. **Recommended prototype:** a per-encode Toeplitz surrogate N₀ = exact for B0 / R2* / fat / GNL / φ0 / k-shift (Tucker
   segment kernels), without ΔC, used for the static warm start and for coarse levels. Inside a defect-corrected
   iteration the exact gridding residual sets the fixed point, so no accuracy is traded. What it must measure first is
   in §7.

## 1. The operator in k-space

### 1.1 Exact form

Per encode, sample j (k-position k_j on the apparent GNL grid, readout time τ_j, knot weights β_k(t_j), per-sample
residual phase Φ_j(ρ) = φ0_j + b_j·ρ + ρᵀΔC_jρ + cubic/quartic terms), coil c, segments l:

    y_jc = Σ_k β_k(t_j) Σ_l a_l(τ_j) Σ_ρ e^{−i k_j·ρ − iΦ_j(ρ)} S_c(ρ) P_l(ρ) c_k(ρ)

    (AᴴA)_{kk′} = Σ_c S_cᴴ [ Σ_{l,l′} P_lᴴ 𝒯_{kk′,ll′} P_l′ ] S_c,
    𝒯_{kk′,ll′}(ρ, ρ′) = Σ_j β_k β_k′(t_j) ā_l a_l′(τ_j) e^{i k_j·(ρ−ρ′)} e^{i(Φ_j(ρ) − Φ_j(ρ′))}

The coil sum is outermost and the bracket is coil-free (memory note `normal_op_coil_factorization`). Without Φ_j, every
𝒯 is exactly a convolution, applied by the 2N embedding: crop · F⁻¹ · diag(T̂) · F · pad, with
T̂(u) = Σ_j w_j D(u − k_j) and D the 2N-periodic Dirichlet kernel (≈ sinc, main lobe 2 cells, 1/|v| tails).

Three ways to represent the middle operator, ordered by what they store:

| representation | stored | work per application | regime |
|---|---|---|---|
| per sample (today: gather + scatter) | per-sample tables, ≈ 70–110 B / sample | samples × W³ × s × coils × 2 | sparse k-space |
| per cell, Toeplitz | one multiplier per (knot pair, segment pair, cell) | cells × pairs × coils | dense k-space |
| per cell, separable / "coordinate" kernels | R kernels, or a few per-cell (time, weight) atoms | cells × R × coils | only if the per-cell knot matrix is low rank |

The question is whether the second or third row can beat the first at 1 mm.

### 1.2 Lower bound on the FFT count

The Toeplitz form diagonalises 𝒯 on the 2N grid. Its inputs P_l′ S_c c_k′ are distinct image functions for every
(c, l′, k′), so each needs its own forward FFT and each output its own inverse. The only escape is to move an
image-space factor into k-space as a convolution:
- **Coils.** S_c is a convolution with Ŝ_c of width w_s per axis: C · w_s³ complex MACs per cell. Already w_s = 7
  costs ≈ 340 MAC per cell per coil, ≈ 10× an FFT's ≈ 5 log₂ n flops per cell. Coil compression is already at
  ≥ 99.9 % SV energy, so the rank of Σ_c S̄_c(ρ)S_c(ρ′) is the coil count.
- **Segments.** e^{−iω(ρ)τ_l} with local susceptibility structure is broadband in k: no compact stencil. §3.2 shows
  that polynomial absorption does not lower L.
- **Knots.** They are the unknowns.

So **≥ 2·C·L·K FFTs of the (2N)³ grid**: the same count as today, on 4.1× the cells.

Pruning-aware ratio (element passes, zero-padded input, k-ball pruning 0.6–0.8 on the last pass):
- σ = 1.25 grid: 1.25 + 1.56 + 1.95·(0.6–0.8) ≈ 4.0–4.4 N³;
- 2N grid: 2 + 4 + 8·(0.6–0.8) ≈ 10.8–12.4 N³;
- **ratio ≈ 2.6–3.0**, plus one extra mixing pass over the 2N grid.

Used below: image side × 3 (range 2.6–4.1).

## 2. Q1: the knot Gram kernel and its separability

### 2.1 Exact structure (derivation)

For pair (k, k′), k = (r, c) (resp knot r, cardiac knot b_c):

    T_kk′(u) = Σ_j h_r h_r′(b_j) · B_c B_c′(φ_j) · D(u − k_j)        (b = bellows, φ = cardiac phase)

- Non-zero only for |r − r′| ≤ 1 and cyclic |c − c′| ≤ 3: each knot has 3 × 7 = 21 partners. That gives 1792 ordered
  and 944 unordered pairs.
- The pair functions h_r h_r′ · B_c B_c′ span a smaller space. Products of degree-p uniform B-splines are degree-2p
  splines with C^{p−1} joins:
  - cardiac (p = 3, M = 16, periodic): dimension M(p + 1) = 64, the same as the number of cardiac pairs;
  - respiratory (p = 1, 5 intervals): 2·5 + 1 = 11.
  The 2-D space has dimension 11 × 64 = **704**. Pairs ((r,c),(r′,c′)) and ((r,c′),(r′,c)) have the same function.
  **944 → 704 is a free, exact compression** (1.34×).
- The kernels are real-valued (real weights), but not symmetric in u, so there is no half-spectrum storage.

### 2.2 Measurement (`knot_gram_rank.py`)

Setup:
- encode 0 gridded onto a stratified cell subset of the 2N grid (all cells within 0.1·N of the centre, a random 2–3 %
  elsewhere, weighted);
- proxy kernel: trilinear;
- SVD of the [cells × 944] matrix;
- error metrics:
  - Frobenius relative error (Eckart–Young), uniform cell weight (white image) and weight (1 + (|u|/2)²)⁻² (k⁻⁴,
    phantom-like);
  - per-cell spectral norm of the 96 × 96 knot-matrix error on 4000 cells.

Trilinear smoothing is optimistic. It averages neighbouring samples, whereas the exact Dirichlet kernel is
energy-preserving and keeps every sample's own time, so the true ranks are at least these.

Proxies:
- **box1mm**: 160 mm box at 1 mm, k × 2 (a 1 mm yarn with the same samples per readout), 2N = 320;
- **box2mm**: native k, 2N = 160;
- **box4mm**: the 4 mm pyramid level, samples |k| ≤ 125 cyc/m, 2N = 80.

| proxy | samples / 2N cell: mean / centre shell | Frobenius white: R at 1e-1 / 1e-2 / 1e-3 / 1e-4 | k⁻⁴-weighted: R at 1e-2 / 1e-3 / 1e-4 | per-cell op-norm error, R = 64 / 256 / 512 (max over cells, rel. to max cell) | mean per-cell rank (1e-2 / 1e-4 of the cell) |
|---|---|---|---|---|---|
| box1mm | 1.8 / 58 | 17 / 216 / 322 / 364 | 4 / 85 / 233 | 0.37 / 5.5e-2 / 6e-5 | 6.2 / 7.1 |
| box2mm | 14 / 471 | 25 / 222 / 325 / 368 | 5 / 101 / 255 | 0.19 / 1.2e-2 / 3e-5 | 22 / 31 |
| box4mm | 25 / 2735 | 1 / 177 / 306 / 349 | 6 / 93 / 236 | 0.08 / 8.4e-3 / 1.5e-5 | 22 / 33 |

Frobenius error at fixed R (white / k⁻⁴), box1mm:
- R = 1: 0.12 / 0.015;
- R = 16: 0.10 / 5.3e-3;
- R = 64: 0.054 / 1.5e-3;
- R = 256: 5.2e-3 / 6.5e-5.

Per k-radius shell (box1mm, white, R at 1e-2 / 1e-4):
- centre 0–0.1 kmax: 76 / 336;
- every outer shell: ≈ 315 / 520.

### 2.3 Why: the fluctuation law (derivation + `knot_kappa.py`)

With incoherent ordering, cell u is fed by n_eff samples whose acquisition times are effectively independent draws
from the scan's time distribution. Then

    T(u) = ρ(u)·Ḡ + fluctuation,     ‖fluct‖_F / ‖ρ Ḡ‖_F ≈ √(κ / n_eff),     κ = E‖ββᵀ‖²_F / ‖E ββᵀ‖²_F − 1

where ββᵀ is the 96 × 96 knot matrix of one readout and Ḡ the scan's mean knot Gram.
- Measured κ = 46.6. Local B-splines put each readout on 8 of 96 knots, while Ḡ is spread over all of them.
- The rank-1 (separable) error falls only as n_eff^{−1/2}: 1e-2 needs n_eff ≈ 4.7·10⁵, 1e-4 needs 4.7·10⁹.
- A cell is crossed by at most 2 × 7592 readouts per encode (out and back), so even the k-space centre is bounded at
  ≈ 6 % for R = 1. This matches the centre shell (R = 1 already gives 1e-1).
- The fluctuation part is a random matrix with no preferred directions. Every added term R captures only ≈ 1/704 of
  it, hence the slow decay up to R ≈ 300–500.

**When is it separable?** Only if the time is (nearly) a function of k: sequential / segmented ordering, where each
k-region is acquired in one cardiac and respiratory window. Then T_kk′(u) ≈ β_kβ_k′(t(u))·ρ(u) is a coordinate kernel.
That ordering is exactly what golden-angle / super-Fibonacci incoherence avoids, and what the CS reconstruction needs.
So it is not an option for this sequence.

**"Coordinate" kernels for t.** A per-cell set of (time, weight) atoms would need ≥ 6–7 atoms at 1 mm and 22–33 at
coarse levels (the per-cell ranks in the table) with the optimistic trilinear proxy, more with the exact Dirichlet tails.
That is gridding to effective samples (GROG-like) with the same per-cell work as the gather. It saves nothing.

### 2.4 Memory (fp32 real kernels, per encode; ×7 encodes)

| grid | one kernel | 944 pairs × L(L+1)/2 = 55 | 704 × 55 (exact) | 704 × 2 halves (pointwise segments, §3.3) | separable R = 300 | static K = 1: 55 / 2 |
|---|---|---|---|---|---|---|
| box 160 mm, 1 mm (2N = 320³) | 131 MB | 6.8 TB | 5.1 TB | **185 GB** | 39 GB | 7.2 GB / 0.26 GB |
| torso 640 × 384 × 384 (2N = 1280 × 768²) | 3.0 GB | 157 TB | 117 TB | **4.3 TB** | 0.9 TB | 166 GB / 6 GB |
| box, 2 mm level (160³) | 16 MB | 0.85 TB | 0.64 TB | 23 GB | 4.9 GB | 0.9 GB |
| box, 4 mm level (80³) | 2 MB | 106 GB | 80 GB | 2.9 GB | 0.6 GB | 0.1 GB |

fp16 kernels would halve this, but fp16 rounding (4.9e-4 relative per entry) is an operator error of the same order,
so not below 1e-4 (§5).

## 3. Q2: the phase physics in the Toeplitz pair form

### 3.1 Which terms move to k-space (derivation)

Per sample the pair phase is Φ_j(ρ) − Φ_j(ρ′) plus the segment part ψ(ρ, τ_j) − ψ(ρ′, τ_j). The tools
(`concomitant_weyl_commutator.md` §1):
- [X_a, T[g]] = T[Δ_a g];
- in k-space X_a ↔ i∂_{u_a} and T[g] ↔ ĝ(u)·;
- for quadratic Φ_j, AᴴA_j = w_j δ(P + M_jX + b_j − k_j) exactly (sheared symbol).

| term | pair structure | Toeplitz realisation | cost |
|---|---|---|---|
| k_j·ρ with GNL | ρ is the apparent (warped) grid (`segphys.rs`): k·(ρ − ρ′) exact | plain embedding | **free** |
| φ0_j (constant per sample) | e^{iφ0}e^{−iφ0} = 1 | — | **free** (it remains in Aᴴy only) |
| b_j·ρ (linear concomitant / B0 terms) | b_j·(ρ − ρ′) = shift of k_j | kernel built at k′_j = k_j + b_j | **free** |
| ΔC_j (quadratic), deg 3–4 per sample | symbol shear k_j + M_jρ̄; first order (i/2)Σ_ab H_ab{X_a, T[Δ_b K^{ab}]} | moment kernels K^{ab} = Σ_j w_j (ΔC_j)_ab D(u − k_j) per knot pair; X_a as FFTs of x_aρ (+3 inputs) or k-space derivatives (9–11-tap stencils on the 2× grid, `concomitant_segmented_first_order.md` §2) | **×7 kernels per pair at first order, ≈ ×22 at second**; kernel-free only if ΔC is a function of k (it is not: §3.4) |
| ω(ρ)τ + w̄(τ)·m(ρ) (non-polynomial in ρ, depends on τ only) | e^{i(ψ(ρ)−ψ(ρ′))τ} | Tucker: L(L+1)/2 segment-pair kernels; Fessler CP: L′ kernels; pointwise per half: weights ā_la_l′(τ_h(\|u\|)) (§3.3) | L(L+1)/2 = 55 (L = 10) or 120 (L = 15) kernels per knot pair; or approximate pointwise |
| polynomial part of ω (shim-like, ≤ 4th order) | τ_j(ρ̄ᵀHΔ + b·Δ) + … | Weyl moments with τ-weighted kernels, pointwise per half | phase ≫ 1 rad: needs many orders; and it does not lower L (§3.2) |
| R2* | e^{−(R(ρ)+R(ρ′))τ}: a sum, not a difference | complex rate ω̃ = ω + iR inside the segment basis | changes L (§3.2) |
| fat (separate unknowns W, F) | per-sample complex scalar c(τ), spatially constant | weights 1, c, c̄, \|c\|² as pointwise multipliers per half, or 3 more Tucker kernel sets | ×2 unknowns (×4 knot-pair blocks) |

### 3.2 Segment count for the residual field (`field_rank.py`)

Setup:
- E[v, τ] = e^{−iω(v)(τ − τ_c)} over object voxels and the yarn ADC (1.6–9.58 ms, 7.98 ms);
- L(ε) = SVD rank at relative Frobenius error ε (the optimal segmentation);
- the polynomial fit is pd-weighted over the region;
- L′ = Fessler's diagonal (pair-difference) form;
- heart box = a 160 mm cube around the blood centroid; field = `field_ppm_recon` at 3 T.

| region | field 1–99 % | L at 1e-2 / 1e-3 / 1e-4: raw | − deg 1 | − deg 2 | − deg 4 | L′ (pair, raw) |
|---|---|---|---|---|---|---|
| torso FOV (2.2 M voxels) | −205 … +277 Hz | 10 / 14 / 15 | 10 / 14 / 15 | 10 / 13 / 15 | 10 / 14 / 15 | 12 / 15 / 17 |
| heart box 160 mm (0.43 M) | −257 … +277 Hz | 11 / 14 / 15 | 11 / 14 / 16 | 11 / 14 / 16 | 10 / 13 / 15 | 13 / 16 / 17 |

With R2* (`truth_r2s` as a stand-in for an estimated map; median 35 s⁻¹ on the torso and 314 s⁻¹ in the box, p99
≈ 1700 s⁻¹ (lung); each voxel's row normalised so that the error is relative per voxel):
- torso 10 / 14 / 16, box 11 / 15 / 16 (raw and after removing degree 4 alike);
- that is at most +1 segment over B0 alone.

Two independent voxel subsamples gave L(1e-4) = 15 and 16 for the box, so these counts carry a sampling noise of ±1.

Reading:
- **Polynomial absorption up to degree 4 does not lower L on this phantom.** The 1–99 % range is unchanged (box,
  degree 4: −295 … +234 Hz). This confirms the shimming note: only local shimming lowers L.
- Fessler's diagonal form needs L′ = L + 2…3 kernels for the same error (pair differences have twice the range).
- Note for the gridding operator too: with this 8 ms readout and field, **L = 10 sits at the 1e-2 level, and 1e-3 needs
  14** (the Frobenius metric over voxels and τ; the operator metric can differ). [CHECK against the driver's segment
  fit.]

### 3.3 Segment and fat weights as k-space "coordinate kernels" (`tau_conc_kspace.py`, `pointwise_weight_2d.py`)

**Readout geometry.**
- The yarn readout goes out to kmax at τ = 5.58 ms and comes back to the centre at 9.58 ms. |k(τ)| is identical for
  all rotations (spread 4e-8 of kmax): the base curve is rigidly rotated.
- Per half h, τ = τ_h(|k|) exactly, monotone; the back half is monotone except its last 2 samples. Each |k| is visited
  twice, up to 8 ms apart.

So per half every τ-only weight g(τ) is a function of k:

    T̂_{g}(u) = Σ_h Σ_{j∈h} g(τ_h(|k_j|)) D(u − k_j)  ≈  Σ_h g(τ_h(|u|)) · T̂_h(u)        (pointwise approximation)

With it the segment sum collapses before the knot mixing:
- V_{h,k′}(u) = Σ_l′ a_l′(τ_h(u)) [F P_l′ S_c c_k′](u), fused into the forward FFT output;
- W_{h,k} = Σ_k′ T̂^h_{kk′} V_{h,k′};
- the output for segment l is Σ_h ā_l(τ_h(u)) W_{h,k}.

So the segments cost 2 kernels per knot pair instead of 55. This is the user's "kernel representing coordinates", and
it is exact for the sample positions themselves.

**What it gets wrong.** The pointwise product is a circular convolution of the embedded kernel with ǧ (the transform
of g(τ_h(|u|))). That is wrong where ǧ reaches across the |Δ| = N edge of the embedding, and g has kinks (|u| at the
centre, the turn at kmax) that make ǧ decay slowly.

2-D test, exact dense reference: N = 40², 160 out-and-back spokes, g = e^{−i2πfτ}, the back half on a perturbed angle.
Relative operator-norm error against the guard band b (embedding 2N + 2b):

| f (phase cycles over the readout) | b = 0 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 1 | 5.1e-3 | 5.9e-4 | 3.9e-4 | 2.3e-4 | 1.2e-4 |
| 4 | 9.5e-3 | 3.7e-3 | 1.7e-3 | 9.7e-4 | 5.1e-4 |

(Harness check: the exact weighted kernel on 2N reproduces N to 1e-16.)
- The error falls ≈ 1/b. Reaching 1e-4 needs b ≳ N, i.e. a 3N-per-axis grid (3.4× the cells of 2N). That kills the
  speed argument.
- The 3-D cusp |u| is milder (ǧ ~ |Δ|⁻⁴ against |Δ|⁻³ in 2-D), so 3-D may be better [CHECK, not measured].
- So pointwise segment and fat weights are a **1e-3–1e-2 approximation**: fine for a surrogate or preconditioner,
  not for the exact operator. Exact segments need the Tucker kernels (55–120 per knot pair).

### 3.4 The concomitant residual is not a function of k

Setup:
- Δw_j = w_j − w̄(τ_j), the deviation of the 4 concomitant coefficients from the per-ADC mean over the encode's
  readouts;
- this is what the per-sample ΔC path carries today; the mean part is in the segments.

| | max phase | p99 | rms |
|---|---|---|---|
| total concomitant, heart box (160 mm, centred on the heart) | 1.51 rad | | |
| residual Δw, heart box | **0.42 rad** | 0.37 | 0.17 |
| residual Δw, torso FOV corners | **1.15 rad** | 1.06 | 0.47 |

- Fraction of Δw's variance explained by its k-cell mean (2 mm box cells, out / back separated, 4 samples per occupied
  cell): 0.55. A random relabelling of the cells gives 0.25 (finite-sample bias), so ≈ 40 % is a function of k and
  60 % is not.
- The yarn's rotation about the k direction is a free parameter at fixed k, and ΔC rotates with it.
- So in the Toeplitz form the ΔC terms need moment kernels per knot pair. From the Weyl table
  (`concomitant_weyl_commutator.md` §4), at Φ ≈ 0.4 rad order 1 leaves ≈ 1e-1 of the effect and order 2 ≈ 1e-2. The
  effect is ‖N − N₀‖/‖N‖ ≈ 1–3 % at these phases, so order 2 (≈ 22 moment kernels per pair, ≈ 10 FFT pairs per base
  pair) is needed for ≈ 1e-4.
- In the gridding operator the same physics is per sample and almost free (`concomitant_segmented_first_order.md` §7:
  o2 ≈ o0).

## 4. Q3: cost model per encode pass (forward + adjoint; 30 coils, L = 10, 96 knots, 30 M samples)

Today's numbers: `temporal_operator.md` §22.6 (I = image side, S = sample side; measured or extrapolated there).
Toeplitz image side I_T = 3 · I (2.6–4.1). The kernel cost is set by memory:
- the 1408-kernel set (pointwise segments) has to be streamed from host per coil batch (25 GB/s) or regenerated;
- a regeneration costs ≈ 1408 × (type-1 onto the 2N Δ-box: one FFT of (1.25·2N)³ + one of (2N)³) + a 36-weight spread
  of all samples;
- the ring buffer of the knot window (39 knots × 2 halves × 262 MB = 20 GB per coil at the box) forces cb = 1.

| case | today (I + S) | Toeplitz variant | Toeplitz estimate | memory needed | verdict |
|---|---|---|---|---|---|
| **box 1 mm, knots** | 25 + 120 = **145 s** | pointwise segments, 704 × 2 kernels streamed per coil | 75 (65–103) + 220 (stream) … 290 (regen) ≈ **300–390 s** | 185 GB kernels + 20 GB ring | **2–2.7× slower**; approximate (§3.3) |
| | | same, kernels resident (hypothetical ≥ 200 GB device) | 75 + ≈ 2 ≈ **80 s** | 205 GB on one device | ≤ 1.8× faster, still approximate |
| | | exact Tucker segments | — | 5.1 TB | impossible |
| **torso 1 mm, knots** | 651 + 612 = **1260 s** | any | ≥ 1950 s image side alone | 4.3 TB kernels; one 2N channel = 6 GB | **impossible / ≥ 1.5× slower** |
| box 1 mm, **static (K = 1)** warm start | 0.26 + 13.5 ≈ **13.8 s** | exact Tucker, 55 kernels | ≈ 0.8–1.1 (image) + 0.25 (kernel reads, 30 coils) + 0.3 (upload of the host-cached set; build ≈ 3 s once per encode: 1.7·10¹² spread MACs + 110 FFTs) ≈ **1.4–1.7 s** | 7.2 GB + 2.6 GB spectra | **≈ 7–9× faster** |
| torso 1 mm, static | 6.8 + 60 ≈ **67 s** | pointwise per half (2 kernels) / exact 55 | ≈ 20–28 s / — | 6 + 12 GB spectra / 166 GB | ≈ 2.5–3× (approximate; 32 GB card); exact impossible |
| box, **2 mm level**, knots | 3–5 + 17–63 = **20–68 s** (§22.7, extrapolated) | pointwise, 1408 kernels resident or host-cached | 9–20 + ≈ 1 (mix) + ≈ 1 (upload) ≈ **11–22 s** | 23 GB / encode (161 GB host for 7) | 1–3×, approximate |
| box, 4 mm level, knots | 1–2 + 2–32 = **3–34 s** | same | 3–6 + < 1 | 2.9 GB / encode | 1–6× |

Notes on the levers:
- **Coil compression** scales today's I and S and the Toeplitz image side by C alike. The Toeplitz kernel set is
  coil-free, so fewer coils make the kernel cost relatively larger. The verdict does not change.
- **Fewer segments** (local shim) shrink both image sides (∝ L) and today's sample side only partly (pass cost
  α + β·L, α dominant). This narrows the gap on the box but leaves the memory problem: the pointwise kernel set does
  not depend on L, and the exact set grows as L².
- **Samples per encode** (60 M at 2× samples per readout): today's S doubles, while the Toeplitz form only rebuilds
  kernels. This pushes the box toward Toeplitz on time (≈ 270 s today vs ≈ 300 s), but not on memory.
- The torso is image-side-bound today. The levers there stay the ones in §22.6: staging once per sample for all coil
  batches (≈ 2.5× on S) and maps-as-kernels / fewer segments / fewer knots on I.

## 5. Q4: accuracy budget of each approximation (target ≪ physics error, ~1e-4 relative)

| approximation | error law | measured here | to stay ≤ 1e-4 |
|---|---|---|---|
| separable knot Gram, rank R | ≈ √(κ/n_eff) until R ≈ dim, then the tail of a random spectrum | 1e-2 at R ≈ 220, 1e-4 at R ≈ 365 (Frobenius); op-norm 5.5e-2 at R = 256 (1 mm) | R ≈ 500 of 704: no compression worth having |
| 944 → 704 pair functions | exact | — | always use |
| pointwise segment / fat weights per readout half | ≈ 1/(guard band); f-dependent | 5e-3–1e-2 at 2N; 1.2e-4 / 5e-4 (f = 1 / 4) at 2N + 64 (2-D) | not reachable at reasonable grid size: use Tucker kernels |
| segments L (B0) | SVD tail; Fessler CP L′ = L + 2…3 | L = 15 (1e-4), 14 (1e-3) | L = 15 Tucker (120 kernels per pair) |
| ΔC moment expansion (Weyl order n) | ≈ (2Φ)^{n+1}/(n+1)! of the effect | Φ_res = 0.42 rad (box), 1.15 rad (torso) | order 2 for the box (≈ 22 kernels per pair); torso needs order ≥ 4 or segments |
| Dirichlet truncation / k-support compression of T̂ | the 1/v tails of D carry ≈ 3 % of the energy beyond ±6 cells per axis | — | keep the exact multiplier (type-1 onto the Δ-box, then FFT) |
| fp16 kernel storage | 4.9e-4 relative per entry → operator error of that order | — | fp32 (or fp16 + residual) |
| f32 FFTs on the 2N grid | ≈ 1e-7 | — | fine |

A surrogate used inside a defect-corrected / lagged-anchor iteration does not need 1e-4. The exact gridding residual
fixes the fixed point (memory note `pnp_theory_direction`: fixed-point-preserving acceleration). The surrogate only
sets the convergence factor ≈ ‖I − N₀⁻¹N‖ ≈ the omitted effect (ΔC: 1–3 %, pointwise segments: ≤ 1 %).

## 6. Q5: literature (verified as marked)

**[V]** full text read (in this or an earlier note) · **[A]** abstract / landing page only.

- **Fessler, Lee, Olafsson, Shi, Noll**, IEEE TSP 53(9) 2005, doi 10.1109/TSP.2005.853152 [V, earlier note]:
  Toeplitz with L² cross kernels for time-segmented field correction; the diagonal "L approach" segments
  e^{−i(ω_j−ω_k)t} with twice the bandwidth. Measured here: L′ = L + 2…3.
- **Sutton, Noll, Fessler**, IEEE TMI 22(2) 2003, doi 10.1109/TMI.2002.808360 [V, earlier]: time-segmented NUFFT =
  our SegNufft.
- **Baron, Dwork, Pauly, Nishimura**, "Rapid compressed sensing reconstruction of 3D non-Cartesian MRI", MRM
  79(5):2685–2692, 2018, doi 10.1002/mrm.26928 [A]:
  - Toeplitz needs "larger oversampling than … NUFFTs";
  - on GPU Toeplitz was faster "for all practical parameters" (static 3D radial / cones, no temporal basis, no
    segments);
  - matrix size reduced by estimating the image support (our box already is the support).
  The regime matches our static row in §4.
- **Mani, Jacob, Magnotta, Zhong**, MRM 74(4):1086–1094, 2015, doi 10.1002/mrm.25486 [A]: Toeplitz CG for multishot
  non-Cartesian diffusion with per-shot phase. The phase is per shot (few kernels), not per sample.
- **Gai et al.**, "More IMPATIENT: a gridding-accelerated Toeplitz-based strategy for non-Cartesian high-resolution 3D
  MRI on GPUs", J. Parallel Distrib. Comput. 2013 [A]: Toeplitz kernel built by gridding, field correction by time
  segmentation (L² kernels).
- **Chen, Chen, Xie, Li, Christodoulou**, ISBI 2022, arXiv 2205.01770 [A here; V in temporal_operator.md §8]:
  L × L subspace block per cell of the 2× grid, 2-D radial. This is our per-cell knot matrix, with L = 32 global (not
  local) temporal bases.
- **Maatman, Blumenthal, Scholand, Flassbeck, Uecker, Assländer**, "Memory-Efficient Iterative Subspace Reconstructions
  on GPUs for Non-Cartesian MRI", ISMRM workshop / annual meeting 2026 [A, title and impact statement only]: ≈ 8× lower
  Toeplitz-subspace memory, reportedly by PSF symmetry and compact k-support. Here: symmetry gives 944 → 704 (exact);
  compact support is the Dirichlet truncation (an approximation, §5).
- **Abraham et al.**, iGROG, arXiv 2310.10823 [V, earlier]: subspace + time segmentation "require L·K² Toeplitz PSFs";
  replaces the 2× embedding by GRAPPA gridding with the field per sample. This is the per-sample side again.
- **Lee, Ramasawmy, Lim, Campbell-Washburn, Nayak**, "MaxGIRF", MRM 2022, doi 10.1002/mrm.29232 [A]: concomitant +
  off-resonance as a higher-order encoding model with a low-rank (SVD) approximation and NUFFTs, no Toeplitz. Same
  family as our segments + per-sample ΔC.
- **Chan & Haldar**, "A New k-Space Model for Non-Cartesian Fourier Imaging", arXiv 2505.05647 [A]: a Fourier-domain
  basis model with lower cost and fewer wrap-around artefacts. Not checked whether it covers temporal bases or field
  terms; worth a read for the coarse-level operator.

Searched and not found: any non-Cartesian operator that combines a temporal (local or global) basis with
off-resonance segments in fewer than L·K²-type kernels, or that represents incoherently ordered time in a Toeplitz
kernel with fewer than the basis-pair count. This is consistent with §2.3: incoherence is exactly what prevents it.

## 7. Decision and prototype

**Not worth a prototype:** the knot-resolved Toeplitz operator at 1 mm (box or torso), the separable knot Gram,
"coordinate" kernels for cardiac / respiratory time, and k-space coil convolutions. Each is excluded by a measured
number above (§2.2, §2.4, §4).

**Worth a prototype: a per-encode Toeplitz surrogate N₀ for the static warm start and the coarse levels.**

Definition:
- exact Tucker segment-pair kernels (55 at L = 10);
- built from k′_j, so φ0 and the linear terms are exact;
- GNL free;
- fat as separate unknowns (4 blocks);
- no ΔC;
- for knots only at ≤ 2 mm levels: 704 functions × pointwise segment weights per readout half.

Use: static warm start, coarse pyramid levels, and inner iterations of a defect-corrected scheme. The exact gridding
operator gives the outer residual, so the final fixed point is unchanged.

Expected gain: 7–9× on the static operator at the 1 mm box; 1–3× per coarse-level knot pass. No 1 mm production gain.

What the prototype must measure first, cheapest first:
1. **Static exact Tucker Toeplitz vs the SegNufft normal operator** on the real tables (1 mm box proxy, CPU at
   small N then GPU), without ΔC:
   - ‖N_T − N_seg,o0‖/‖N_seg,o0‖ must be ≤ 1e-5 (pure implementation check);
   - then ‖N_T − N_seg,o3‖/‖N_seg,o3‖ = the omitted ΔC effect. Decision: if > 5 % the surrogate converges too slowly
     for defect correction.
2. **3-D accuracy of pointwise per-half segment weights** (this note's 2-D test predicts 5e-3–1e-2 at 2N) against the
   guard band. Decision: whether coarse-level knot surrogates can use 2 kernels per pair or need 55.
3. **Build and apply time on the GPU** at the box (2N = 320³): 55 kernels (type-1 onto the Δ-box + FFT, multi-weight
   spread), against the 13.8 s static gridding pass. Kill criterion: < 3× faster.
4. **Defect-corrected iteration count** on the box warm start (surrogate inner, exact outer), against plain
   preconditioned CG with the exact operator, at equal final residual on held-out k-space. Kill criterion: wall time
   not < 0.5× of plain.

Only if 1–4 pass: the coarse-level knot surrogate (704 × 2 kernels per encode at the 2 mm level, 23 GB, host-cached),
measured against the extrapolated 20–68 s per encode pass of §22.7.

## 8. Files

- `research/experiments/normal_op/common.py`: loaders (samples / sched_nominal per encode, knot weights from cardiac
  phase and bellows quantiles).
- `knot_gram_rank.py [box1mm|box2mm|box4mm] [tri|nearest] [frac]` → `out/knot_gram_*.json` (§2.2).
- `knot_kappa.py` (§2.3).
- `field_rank.py [degrees]` → `out/field_rank.json` (B0 and L′; its `L_b0_r2s` used unnormalised rows and is superseded) and `out/field_rank_-1,4.json` (R2* with per-voxel rows) (§3.2).
- `tau_conc_kspace.py` → `out/tau_conc_kspace.json` (§3.3, §3.4).
- `pointwise_weight_2d.py` → `out/pointwise_weight_2d.json` (§3.3).
- Run with `flock …/cpu.lock systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 rustlib/.venv/bin/python -I <script>`
  (≈ 0.5–4 min each).
