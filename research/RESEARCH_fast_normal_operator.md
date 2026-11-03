# Research note: a fast normal operator with off-resonance + concomitant fields + GNL, without L² kernels in memory

Status: hypotheses and back-of-envelope costs. Nothing here is measured. Items marked **[check]** need a
prototype or a literature check before anyone builds on them. This note exists to decide which
primitives the Rust FFT/NUFFT libraries must expose so these experiments are cheap to run.

Literature actually read for this note: Fessler et al., IEEE TSP 53(9) 2005 (text extracted; equations were
lost in extraction, so the structure below is reconstructed from the surrounding prose and my own derivation).

---

See also: `concomitant_structure.md` (base-spoke structure, ranks, probable coefficient bug) `concomitant_weyl_commutator.md` (exact commutator/Weyl calculus, measured operator accuracy vs FFT pairs) and `concomitant_stencils_and_kernels.md` (round 2: butterfly verdict, chirp-aware stencil NUFFT, kernel design, defect correction, regime table).

## 0. TL;DR

(Revised after your corrections: coils are looped *inside* each streamed kernel, typical sizes are N=320 / 512×512×320 / 512×512×256, i.e. padded 640³ / 1024×1024×640 / 1024×1024×512 — not 1024³.)

1. **The O(L) Toeplitz method exists: Fessler 2005, Sec. IV-B, eq. (24)**: `A'A ≈ Σ_l D_l^H T_l D_l` — diagonal-in-l (CP-like), `L'` kernels and `L'` FFT pairs, from time-segmenting the **pair** exponential
   `exp(i(ω(r_n) − ω(r_m)) t_j)` directly. Stated price: that pair term has twice the bandwidth, so `L' > L`. The L² version (IV-A) is what `make_normal_naive_off_fourier_toeplitz_embeddings` does.
   Your `P×R`/ALS compression is the general-phase optimum of this idea. **Action: measure `L'` vs `L(L+1)/2` on your phases.**
2. **With C coils inside each kernel, Toeplitz is FFT-bound, not PCIe-bound.** Per application: `2·L'·C` FFTs of `(2N)³`. Streaming one kernel (2 GB at 640³ ≈ 84 ms) hides behind `2C` FFTs (≈ 0.93 s). So the cost to attack is the **FFT count `L'·C` and the grid size**, and the *storage* (TB), not bandwidth.
3. **The storage problem is probably solvable by regeneration, at ~10% compute (Sec. 3.7).** One kernel costs ≈ 4 type-1 NUFFTs ≈ 0.1 s to build vs 0.93 s to apply to 32 coils. Build kernels per frame-solve (or even per application) and keep nothing.
4. **Modulated-row NUFFT (Sec. 3.1)** removes the factor `L'` from the FFT count but pays `C·K·taps`; with coils in the picture the realistic win is ~2–5× in the optimistic regime (not the 6–10× I first claimed, which ignored `C` on the sample side). Needs a prototype.
5. **Concomitant fields are quadratic in position only to leading order in `1/B0` and for ideal linear gradients** (Sec. 3.2). Real GNL coils give higher-order polynomial corrections. The local-shift formula is exact for degree ≤ 2 only; otherwise it has an `O(Δ³ ∂³ψ)` remainder.
6. **Rotated-prototype spokes (Sec. 3.6) give real structure**: trajectories, temporal bases, kernels and the concomitant waveform tensors are all generated from a tiny prototype table + per-spoke rotation. This makes kernel regeneration cheap and cuts the number of distinct temporal bases to one.

---

## 1. Model

Per sample `j` (time `t_j`, frame/encode implied), coil `c`, voxel `n`:

```
a_jn = exp( -i ( k_j·r_n + ψ_j(r_n) ) )                   A = [C_c ∘ a]
ψ_j(r) = ω(r) t_j                          (off-resonance)
       + Σ_q m_q(r) w_q(t_j)               (concomitant, m_q = quadratic monomials; w_q = waveform integrals)
GNL:  k_j·r  →  k_j·u(r)                   (coordinate warp; your q-grid)
```

Normal-operator entries (exact):

```
(A^H A)_{nm} = Σ_j w_j exp( i k_j·(r_n − r_m) ) · exp( i ( ψ_j(r_n) − ψ_j(r_m) ) )
```

**Lifted-Toeplitz identity (exact).** Define a feature vector `μ(r) = (r, m_1(r)…m_Q(r), ω(r))` and per-sample
`κ_j = (k_j, w_1(t_j)…w_Q(t_j), t_j)`. Then `(A^H A)_{nm} = T(μ_n − μ_m)` with `T(Δμ) = Σ_j w_j e^{i κ_j·Δμ}`.
So the operator is *exactly* Toeplitz in a `(3+Q+1)`-dimensional feature space. Every practical method is a way to
avoid gridding that high-dimensional space: tensor-product bins (exponential in Q), SVD/ALS low rank (your approach), or stencils (Sec. 3.1).

---

## 2. Cost model (per frame, per encode, per normal-operator application)

Notation: `N` image size per axis, `K` samples in the frame, `w` NUFFT kernel width, `σ` oversampling, `L` / `L'` interpolation terms (Tucker / CP), `C` coils.
Per-kernel loop order: kernel resident (streamed or rebuilt), **all `C` coils applied inside it**.

Measured cuFFT 3D C2C f32 on the RTX 4090 (`bench/baselines.md`): `640³` 14.6 ms · `1024×1024×512` 29.1 ms · `1024×1024×640` 36.7 ms · `1024³` 58.1 ms · `400³` 3.66 ms. All at the ~3-pass memory roofline.

| Method | FFTs per application | Sample-side work | Stored | Notes |
|---|---|---|---|---|
| M1 NUFFT-L | `2·L·C` of `(σN)³` | `2·L·C·K·w³` | trajectory, `c_l(t_j)` | O(L) but full interp/spread |
| M2 Toeplitz Tucker | `2·L·C` of `(2N)³` | none | `L(L+1)/2` kernels | cheap to apply per kernel (kernel read amortised over C), huge to store |
| M3 Toeplitz CP (Fessler IV-B) | `2·L'·C` of `(2N)³` | none | `L'` kernels | `L' > L`, linear storage |
| M4 lifted 4D Toeplitz (1 equispaced dim) | `2·C` of `(2N)³` + length-`2L` FFT along l | none | `2L−1` kernels | |
| M5 modulated-row NUFFT | `2·C` of `(σN)³` (σ may be < 2) | `2·C·K·taps` | factors only | wins when `K·taps ≪ L'·FFT(2N)` work |

Worked example, N=320 (padded 640³, 14.6 ms/FFT), `C=32`, `L'=10`:
- M3: `2·10·32 = 640` FFTs ≈ **9.3 s** per application. Kernel streaming (84 ms each, 10 kernels) is hidden. Peak GPU memory: one kernel + coil/field buffers.
- M5, σ=2: 64 FFTs ≈ 0.93 s; σ=1.25 (400³): 64 × 3.66 ms ≈ 0.23 s. Sample side: `2·C·K·taps`; at K=3·10⁶, taps=1000–3000 that is 2·10¹¹–6·10¹¹ MAC ≈ 1–3 s at an assumed ~2·10¹¹ MAC/s effective (atomics/gather) **[check]**. So M5 ≈ 1–4 s vs 9.3 s: **~2–5×**, break-even near taps ≈ 5000 at K=3·10⁶.
- Coil compression to ~8–12 virtual coils scales both methods by `C`, not the ratio.
- Crossover rule: M5 beats M3 while `K·taps ≲ (L' − σ³/8)·FFT(2N)-equivalent work`; high `K` (≳ few % sampling) or large `taps` flips it.

## 3. Structural ideas

### 3.1 NUFFT with modulated rows ("k-space stencil NUFFT")

Row `j` of the system matrix is `a_j(r) = C_c(r) · exp(-i(k_j·r + ψ_j(r)))`. NUFFT represents `a_j` on a fine grid as a *compact* stencil
(`ES kernel` around `k_j`) because the row's instantaneous frequency is `k_j` everywhere. For a modulated row, the instantaneous frequency is
`k_j + ∇ψ_j(r)` (+ coil-map gradient), which stays inside a small set around `k_j`. So the row still has a compact fine-grid stencil
`Γ_j(u)`, wider by the **frequency spread** of the modulation:

- Exact identity: multiplication by an `N`-periodic smooth image function `B(r)` = convolution of `x̂` with `b̂` (the DFT of `B`), and `b̂` is compact if `B` is smooth.
- Spread estimate for quadratic ψ: roughly `~1.3 · ψ_max` k-cells per axis for the phase **[check constant]**, plus ES width `w`, plus coil-map bandwidth (typically 3–6 cells).
- Representation: `Γ_j = Σ_l c_l(t_j) · (φ ⋆ b̂_l)` — i.e. **the L-term SVD output you already compute is exactly a rank-L factorisation of the `K × taps` stencil matrix.**
  Store `c_l(t_j)` (`K × L`) and the small tables; synthesise `Γ_j` on the fly in registers/shared memory, never store it.
- Normal operator: type-2 with stencil → multiply → type-1 with conjugate stencil (atomic scatter into *one* grid) → one FFT pair. `L`-independent FFT count.
- Needs `B_l` to be band-limited. Your current histogram/bin-indexed `B_l` (piecewise constant per bin) is not; evaluate `B_l` from the smooth polynomial features,
  or fit the SVD on a smooth basis.
- Coils: either keep `C_c` in the image domain (`C` FFT pairs, phase-only stencil) or fold `Ĉ_c` into the stencil (1 FFT pair, `C × taps` per sample; coil compression to ~8 virtual coils helps).
- Accuracy lever: truncation of `b̂_l`, `w`, `σ` (σ could go well below 2 since FFT cost ∝ σ³ and `K·taps` is not the bottleneck in this regime; kernel width grows). **[check]**

What the library needs: a NUFFT whose per-sample spread/interp weights come from a *user-supplied device function* (stencil generator) rather than
a hard-wired ES evaluation; multi-RHS (many `ntransf` sharing one weight evaluation); free `σ`.

### 3.2 Concomitant (quadratic) phases are exact local k-space shifts — to the order the quadratic model holds

When is the concomitant phase quadratic? Maxwell's equations give, for ideal **linear** gradients and to **leading order in 1/B0**,
`B_c = (1/2B0)[(G_x z − G_z x/2)² + (G_y z − G_z y/2)²]`, so `ψ(r,t) = (γ/2B0) Σ_{ab} W_{ab}(t) Q_{ab}(r)` with `W_{ab}(t) = ∫G_aG_b dt` and `Q_{ab}` quadratic. Not exact beyond that:
- next order in `1/B0` is relatively `~G·r/B0 ≈ 10⁻³` of `B_c` (rough: 40 mT/m · 0.25 m / 3 T) **[check magnitude]**;
- with real gradient coils (GNL) the concomitant field follows from the actual coil field, so it acquires cubic and higher terms of relative size ~ the GNL fraction (5–20% at the edge of FOV). Model those with extra polynomial features, or via the q-warp, and treat the local-shift identity below as first-order for them;
- eddy currents / gradient delays change `G(t)` itself (see Sec. 3.6) but not the quadratic structure.

For `ψ_j(r) = ½ rᵀ M_j r + b_j·r` (exactly quadratic; isocentre offset keeps it quadratic):
`ψ_j(r̄ + Δ/2) − ψ_j(r̄ − Δ/2) = (M_j r̄ + b_j)·Δ` **exactly**. Hence

```
H(r̄, Δ) = Σ_j w_j exp( i ( k_j + M_j r̄ + b_j ) · Δ )
```

- The normal operator is a *spatially varying convolution whose local PSF is the PSF of a sample set with k-positions shifted by `M_j r̄ + b_j`*.
- Degree-3+ terms (GNL corrections to `B_c`, non-linear ω): same form to first order, remainder `O(Δ³ ∂³ψ)`, which grows with the PSF range `|Δ|`. Off-resonance with smooth ω: shift `t_j ∇ω(r̄)`, remainder `O(Δ³ ∂³ω)`. GNL after linearisation: `k_j → J(r̄)^T k_j`; this one is **not** exact for long-range PSF components (cubic term × large |Δ| × large |k|) — keep your q-grid warp for GNL.
- Radial/spiral-like readouts have `M_j = W(t_j) · H_{n̂(s)}`: a scalar waveform times a spoke-direction matrix with 6 free numbers. The sample set is a low-dimensional manifold in `(κ)`; a model exploiting it (rather than generic SVD) could need far fewer terms. **[check, trajectory-specific]**
- Caveat: a purely *local* (patch/overlap-add) treatment drops long-range interactions. Undersampled frames have streak-like PSFs reaching across the FOV, with phase `M r̄·Δ` large at large `|Δ|`. Patch methods are good for compact-PSF (blur) problems, **not** here. Do not pursue unless near/far splitting is added.

### 3.3 Make the L² → L reduction systematic (CP instead of Tucker)

Write the pair phase `P_j(n,m) = exp(i(ψ_j(r_n) − ψ_j(r_m)))`. The L² form factors each side separately (Tucker: `L×L` core ⇒ `L(L+1)/2` kernels).
Fessler IV-B approximates `P` directly with `Σ_l d_l(n) conj(d_l(m)) · c̃_l(t_j)` (CP/diagonal ⇒ `L'` kernels). For
`ψ = ω(r)t` the `d_l = e^{-iω τ_l}`, `c̃_l = b_l(t)` time-interpolators designed over the **doubled** bandwidth. For your multi-waveform phases there is no single `t`, so the
analogue is an ALS/CP fit over the tensor `P_{n,m,j}` restricted to `d_l(n)·conj(d_l(m))` structure — which is what `als_refine` / `P×R` is heading toward.
Measure `L'` vs `L(L+1)/2` for your real phases; the memory win is `L(L+1)/2 / L'`.

### 3.4 Lifted Toeplitz with an equispaced extra dimension (M4)

If one dimension of the lift is handled with an *equispaced exponential family* (frequency segmentation with uniform ω-bins), the temporal factors are
`c_l(t) = e^{-iω_l t}` and `conj(c_l) c_{l'} = e^{i(ω_l − ω_{l'}) t}` depends only on `l − l'`. Then the normal operator is a 4D block-Toeplitz: pad along `l` to `2L`, FFT along `l`
(cheap), multiply by a single 4D kernel of size `(2N)³ × (2L−1)`. Fessler notes uniform bins only worked well for simple field maps (Lloyd–Max placement is better), so check how much non-uniform placement breaks the shift structure.
Tensor-product extension to `Q` concomitant dimensions is exponential in `Q` — probably worse than SVD/ALS unless `Q_effective` is 1–2.

### 3.5 Memory-side levers, independent of the algorithm

The kernel multiplier `K̂(u) = Σ_j w_j D(u − k_j)` (Dirichlet-smoothed gridded density) is:
- real-valued when weights are real and the kernel Hermitian ⇒ store half; f16/bf16/block-scaled storage ⇒ 2–4× more;
- concentrated near the trajectory for undersampled data: truncating it to a neighbourhood of the sample positions is *the same approximation as NUFFT gridding* (that is the bridge between M3 and M5);
- used in the same order as the FFT ⇒ expand in registers during the pointwise multiply (FFT convolution hook), bandwidth-bound anyway.

### 3.6 Rotated-prototype spokes: what is exploitable

Assumption (yours): every spoke is `R_s` applied to one prototype `k_0(τ)`, starting at the k-space centre. Then:

1. **Trajectory generation, not storage.** `k_{s}(τ) = R_s k_0(τ)`. With per-axis gradient delay/eddy response (LTI per axis), `[k_s(τ)]_a = Σ_c R_{ac,s} · κ_{ac}(τ)`, `κ_{ac} = h_a ⋆ k_{0,c}`: still *multilinear in the 9 entries of `R_s`* with a 3×3×T table. So imperfect (but calibrated, LTI) gradients keep the structure; only spoke-dependent non-LTI deviations break it. Coordinates `[3,K]` (and `[3,K]` per term for GNL/concomitant effective positions) can be generated in the kernel from `R_s` and the table.
2. **Concomitant waveform tensor rotates.** `W_s(τ) = R_s W_0(τ) R_sᵀ` (6 comps, = l=0 trace part ⊕ l=2 five-comp part under SO(3)). The B0 axis breaks the rotation symmetry of the *spatial* side `Q(r)`, so the phase is `φ_s(r,τ) = (γ/2B0) tr(W_0(τ) · R_sᵀ Q(r) R_s)`: the spoke enters only through `R_s`.
   For **radial** prototypes `W_0 = w(τ) ẑẑᵀ` with `w(τ) = ∫g²dt` (scalar, monotone): `φ_s(r,τ) = w(τ) · q_{n̂_s}(r)`, with `q_n̂(r) = (γ/2B0)((n_x z − n_z x/2)² + (n_y z − n_z y/2)²)` — **a product of one shared scalar time function and one direction-dependent spatial quadratic**. The time basis `c_l(τ)` is then a single table of length T for all spokes (not K×L), and the spoke dependence of the spatial basis is polynomial in `n̂_s` of degree `2l` at Taylor order `l` (spherical harmonics up to degree `2l`).
3. **PSF is a universal 2D table (radial case).** With `M_j = w(τ) H(n̂_s)` the normal-operator kernel is `H(r̄,Δ) = Σ_s F(n̂_s·Δ, Δᵀ H_s r̄)` with `F(α,β) = Σ_τ w_τ e^{i(a(τ)α + w(τ)β)}`: one small 2D table `F` plus S direction vectors describe the *entire* spatially varying kernel of any frame. (Exact within the quadratic model; `a(τ)` = radial k-position.) For non-radial prototypes the table gains dimensions (`W_0` has up to 6 components) but the idea holds.
4. **Frame kernels are sums of rotated copies of one function.** Without ψ, `T_frame(Δ) = Σ_{s∈frame} P(R_sᵀ Δ)` (ridge functions for radial). So any frame's `K̂` is a rotation-sum of the same sparse footprint — generated by spreading the prototype along `R_s` onto the grid (cost ~ `S · 2N · footprint`, ≪ grid size). This feeds directly into 3.7.
5. **What does NOT follow.** The normal operator of a *subset* of spokes has no continuous rotation symmetry (only full-sphere sampling would make it radial-kernel / diagonal in spherical harmonics), so there is no free spherical-harmonic diagonalisation for undersampled frames. A Radon/central-slice formulation `Σ_s R_sᵀ(h ⋆ R_s ρ)` costs `~S·N³` per term per coil; at S~10³ that is comparable to the FFT route (≈10¹⁴–10¹⁵ flops at L'=10, C=32), so no clear win except for very small S. **[check; speculative]**
6. **Imperfections.** Spoke-dependent deviations (non-LTI eddy currents, timing jitter, gradient heating drift) break the exact tables; model them as small residuals (extra stencil/phase terms) rather than discard the structure.

### 3.7 Rebuild kernels instead of storing them

Cost to **build** one Toeplitz kernel: `2^(d−1)` type-1 NUFFTs on `N^d` modes (current code) ≈ `4 × [FFT(2N)³ + K·w³ spread]`. At N=320, K=3·10⁶, w≈7: ≈ 4 × (14.6 ms + ~5 ms) ≈ 0.08–0.1 s.
Cost to **apply** that kernel to all coils: `2C` FFTs ≈ 0.93 s (C=32). So building is ~10% of *one* application, and amortises over every CG iteration of that frame.
- Process frame by frame; build the `L'` kernels for the frame (or build each kernel just before its coil loop and discard it). Peak memory ≈ one kernel + working fields instead of `frames × encodes × L' × kernel` (TB).
- If an outer algorithm revisits a frame with only ~3 CG iterations each time, rebuild per visit still costs only ~3–7% extra.
- A faster build exists for rotated-prototype data: spread the prototype footprint rotated by `R_s` (3.6.4), and share the `c_l(τ)` table. Needs an accuracy study of the truncated Dirichlet footprint vs the NUFFT-based build (slow 1/x decay is why NUFFT is used).
This may remove the memory problem outright at small compute cost; **please sanity-check against your streaming benchmarks** (why precompute everything — outer-loop revisits, frame coupling, or build cost that I'm underestimating?).

---

## 4. Proposed experiment sequence (cheapest first)

| E | Question | Method | Needs from libs |
|---|---|---|---|
| E0 | What `L`, `L'` (CP), `L(L+1)/2` do your actual ψ need for 1e-2…1e-4 operator error at realistic FOV/readout/field strengths? | Torch/C++ prototype on small N (64³–128³); error = ‖Nx − N_approx x‖/‖Nx‖ on random + phantom images | none (reuse `phi_lowrank`, `als_refine`) |
| E1 | CP (diagonal) Toeplitz vs current Tucker: kernel count × accuracy | extend `normal_toeplitz.cppm` prototype; ALS | none |
| E2 | M5 accuracy vs stencil support (b̂ truncation, `w`, `σ`) and its cost vs `K` | small prototype: generic gather/scatter with dense stencil from SVD factors; then optimise | FFT (any), generic NUFFT-with-stencil |
| E3 | Crossover `K/N³` between M3 and M5 | timings at N=128, 256, (384/512 multi-GPU) | benchmarked Rust FFT + NUFFT-with-stencil |
| E4 | Kernel compression (real/half/f16/truncated) vs accuracy and speed | storage-format experiments in convolution hook | FFT conv hook with packed-kernel loader |
| E5 | Distributed 512³: slab/pencil FFT scaling, kernel streaming across nodes | after the FFT track F3 | distributed FFT |

Decision gate after E0–E3: choose per regime (`K/N³`, `L'`, `C`) or implement the dispatcher that picks M3/M5 from a cost model.

---

## 5. Consequences for the library design (kept general, per your instruction)

1. **FFT**: generic batched/strided ND C2C with pluggable load/store functors (device-side snippets) and a "convolution" mode — no Toeplitz knowledge inside.
   Layout and plan interface must allow **slab/pencil distribution** from the start (global shape, local shape, process grid), even if v1 is single-GPU. Use 64-bit indices everywhere (`1024³` elements = 2³⁰ complex values = 8.6 GB; `2048³` would exceed `i32` element counts).
2. **NUFFT**: kernel/stencil evaluation is a *device-function parameter* (ES by default). `σ` free in `[1.1, 2]`. Multi-RHS with shared weight evaluation. Spread/interp-only exposed. Stream-ordered, plan reusable across `setpts`.
3. **Streaming**: first-class `Stream`/`Event`/pinned-host/pool in `hasty-cuda`; FFT convolution hook accepts a kernel *source* that may be refilled by a copy stream (double-buffered) — you already benchmarked that this hides the PCIe cost.
4. **Benchmarks first**: baselines vs cuFFT/VkFFT/cuFINUFFT precede any custom feature work (as you requested).
5. The M5 stencil generator, FFT hooks and packed-kernel loaders are all *users* of generic hooks; none of them belong in `hasty-fft`/`hasty-nufft` core.

## 6. Things I am unsure about

- Effective sample-side throughput assumed for M5 (2·10¹¹ MAC/s with atomics) is a guess; it decides the M3/M5 crossover.
- Whether the constant in "spread ≈ 1.3·ψ_max cells" is right, and whether `taps` stays ~10³ for your worst-case concomitant phase at 512³ FOV.
- Atomic throughput for ~10³–10⁴-tap spreads on a 1024³ grid; likely needs the SM (shared-memory subproblem) strategy with a stencil-aware bin size.
- Exact structure of Fessler eq. (24): the extracted text dropped the symbols. Re-derive from the paper before implementing.
- Whether piecewise-constant (bin-indexed) `B_l` can be replaced by smooth ones without raising `L`.
