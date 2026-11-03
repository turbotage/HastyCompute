# Concomitant fields inside the encoding operator: banded segmentation + first-order correction

Follow-up to `usernotes/concomitant_toeplitz_derivation.md` (user) and `concomitant_weyl_commutator.md` (round 2).
Numbers from `rustlib/crates/hasty-bench/src/bin/conc_study.rs` on the simulated spiral 4D-flow sequence of subject R4
(3 T, 80 mT/m / 200 T/m/s, 2 mm, 36 cm in-plane FOV, 56 mm sagittal-oblique slab, stack of spirals, 4.4 ms ADC,
FOV centre 3 cm from isocentre).

## 1. Relation to what was tried before

The user's note = the Weyl/commutator family of round 2 (pair phase ψ(r) − ψ(r′) = 2mᵀCd, Taylor in the pair phase,
{X_p, [X_q, S]} operators). Round 2 realised it with extra image-domain fields (≈ 4 FFT pairs at first order; measured
6.7× plain Toeplitz fused, 32× at second order) and one stored kernel per weight monomial. The two new ideas of the note:

1. **kernel-free weights**: S_pq = c_pq(−i∇)T, σ_pq = c_pq(ξ)·τ, valid if C_n ≈ c(k_n);
2. **FFT-free position multiplication**: X_p ↔ i∂_ξ as finite differences on the existing Fourier volume V = F(D_ℓ x).

The note's Fourier-domain formula F{i{X_p,[X_q,S]}v} = −i[(∂_pqσ)V + 2(∂_qσ)(∂_pV)] checks out (product rule; no
derivative of the output volume is needed).

## 2. Measurements

**Size of the concomitant phase** (scanner coordinates → FOV-centred logical ρ: φ = φ0 + b·ρ + ρᵀCρ):

| part | max over samples and FOV |
|---|---|
| constant φ0 | 0.08 rad |
| linear b·ρ (= k-space shift of ≤ 0.29 cells) | 0.96 rad |
| quadratic ρᵀCρ (FOV corners, ±18 cm in-plane) | 3.1 rad |

The linear part is absorbed exactly and for free by the trajectory (k′_n = k_n + b_n) and the constant by the sample
weights; the per-encode bipolar part is a constant C per encode (folded into the coil maps, `--conc-dc`). Only the
quadratic part needs operator work, and at 3 rad a global first/second-order expansion is far too inaccurate.

**Is C_n a compact function of k_n?** (the note's §12 question; per encode, worst phase error at the FOV corner)

| representation of the 6 entries of C_n | worst error | relative rms |
|---|---|---|
| constant | 0.97 rad | 3.2e-1 |
| polynomial in k, degree 2 (10 terms) | 0.066 rad | 2.7e-2 |
| polynomial in k, degree 4 (35 terms) | 0.061 rad | 8.2e-3 |
| polynomial in readout time τ, degree 6 | 0.18 rad | — |

Yes for stack-of-spirals with in-plane rotations (C depends on |k_xy| through the readout time, on kz through the blip,
on the encode through the bipolar): k is a better variable than τ, degree 2 suffices. (For readouts rotated in 3D the
plane normal is not a function of k; then c(k) needs several groups.)

**Finite differences on the 2×-oversampled Fourier volume** (optimal antisymmetric/symmetric stencils, content up to
half Nyquist; max relative error): 7 taps 6e-3 / 7e-4 (1st / 2nd derivative), 9 taps 1e-3 / 1e-4, 11 taps 2e-4 / 1e-5.
Usable, but 9–11-tap stencils per axis cost about one FFT pass each unless fused into the FFT/mixing passes. The
multiplier τ itself is *not* smooth (the kernel fills the whole 2N period), so ∂τ, ∂²τ cannot be finite-differenced:
they need derivative kernels (3 + 6 per segment, shared by all (p, q)) or the sample-domain form below.

**Low-rank (time-segmentation) rank of exp(−iψ)** (CUR, body voxels, reference encode):

| phase | L at 1e-2 | L at 2e-3 |
|---|---|---|
| B0 only | 8 | 9 |
| quadratic concomitant only | 4 | 6 |
| B0 + quadratic concomitant (joint) | 15 | 22 |

The joint rank is about the sum: treating both by one global segmentation is what is expensive (our current CUR
operator: L ≈ 20).

**Banded segmentation carrying both, + first-order correction** (the synthesis): segments banded in readout time,
diagonals D_ℓ = exp(−i[ω(ρ)τ_ℓ + ρᵀC̄(τ_ℓ)ρ]) (B0 and the τ-mean concomitant — free, D_ℓ is a pointwise multiply that
exists anyway), least-squares interpolators a_ℓn, and the within-band / non-τ residual ΔC_n = C_n − C̄(τ_n) handled by
the first-order factor (1 − iρᵀΔC_nρ):

| L | without correction | with first-order ΔC correction |
|---|---|---|
| 6 | 2.5e-2 | 2.0e-2 |
| 8 | 1.5e-2 | **3.2e-3** |
| 10 | 1.3e-2 | **8.9e-4** |
| 12 | 1.3e-2 | 4.3e-4 |

Without the correction the error saturates at ~1.3e-2 (the part of C_n that is not a function of τ). With it,
L = 8–10 segments (= the B0 rank) reach 1e-3: **concomitant fields cost no extra segments**, only the correction.

## 3. Design

**Encoding operator A (forward/adjoint — what the P²nP gradient needs).** Per segment ℓ: v_ℓ = D_ℓ(S_c x), one
oversampled FFT, then interpolation at the samples with the value *and* second-derivative weights:

ŷ_n = Σ_ℓ a_ℓn [ V_ℓ(k′_n) + i Σ_pq ΔC_n,pq ∂_p∂_q V_ℓ(k′_n) ],   (F{ρ_pρ_q v} = −∂_p∂_q V)

∂_p∂_q V(k) = Σ_u ∂_p∂_q φ(k − u) V_u is exact on the interpolant (round-2 note §2.4): **no extra FFT, no c(k)
representation needed** (ΔC_n is used per sample), only 6 extra weight sets in the gather/scatter. Requires a
spreading kernel with smooth derivatives (Kaiser–Bessel, or ES with derivative-safe handling) and accuracy of
interpolated second derivatives (to be measured). Adjoint = transpose (scatter with the same weights). Cost ≈ L FFT
pairs (L ≈ 8–10) + a gather/scatter ~2–3× heavier than plain; vs ≈ 20 NUFFT pairs for the joint CUR operator now.

**Normal operator H (Toeplitz surrogate).** H ≈ Σ_ℓ D_ℓ^H F⁻¹ Q_ℓ F D_ℓ with Q_ℓ = τ_ℓ + the note's first-order
correction built from σ_pq = c_pq(ξ)τ_ℓ, c = degree-2 polynomial in k (per encode). Needs ∂τ_ℓ, ∂²τ_ℓ (derivative
kernels: precomputable, fp16) and 9–11-tap finite differences of V_ℓ, ideally fused into the slowest-axis FFT pass
(`MixZPlan`) where whole z-lines are in shared memory. (Note: the time-segmented Toeplitz form with L kernels is itself
an approximation of the exact L² cross-kernel form; to be quantified alongside.)

**In the reconstruction**: A (sample-domain form) for exact gradients at anchors, H (Toeplitz form) as the cheap
surrogate in the lagged-correction scheme (`implementation_results.md` §5).

## 4. Next steps

1. Derivative-weight gather/scatter in `hasty-nufft` (value + ∂² weights, KB kernel), accuracy test vs direct sums.
2. Banded-segment encoding operator in `hasty-recon` (k′ shift, D_ℓ with B0 + C̄, ΔC first-order term), replacing the
   CUR operator; benchmark vs the CUR operator (L≈20) and the chirp NUFFT.
3. Toeplitz form with derivative kernels and fused finite differences; accuracy vs the exact normal operator.
4. Check representability for 3D-rotated readouts (plane normal not a function of k).

## 5. Round 2 of this study: tracking down the optimal operator

**Derivative interpolation from the NUFFT grid** (1-D study, relative error of value / 2nd derivative):

| σ (grid / volume vs σ=2) | W | value | ∂² |
|---|---|---|---|
| 2.0 (1.00) | 7 | 9e-7 | 8e-5 |
| 2.0 | 9 | 1e-8 | 1.4e-6 |
| 1.5 (0.42) | 10 | 6e-8 | 6e-5 |
| 1.25 (0.24) | 8 | 3e-5 | 3e-3 |
| 1.25 | 10 | 2e-6 | 3e-5 |

ES and Kaiser–Bessel behave the same. The correction term is ≤ ~0.2 rad, so ∂² accuracy 1e-3 is plenty: the
derivative form of the encoding operator is accurate even at σ = 1.25 (W = 10), i.e. on a grid with ¼ of the σ = 2
volume. → **zero-extra-FFT forward/adjoint with the concomitant correction is accurate.**

**Smooth B0 into the trajectory/quadratic terms** (linear → time-dependent k-shift, quadratic → added to C):
L = 8 + first order: 3.2e-3 → 2.7e-3 (deg 1), 3.4e-3 (deg 2). The post-shim B0 is local susceptibility structure, not
smooth: **the segment count is set by B0 alone** (≈ 8–10 at 1e-3) and the concomitant fields add none.

**Sparse (banded) interpolators** (each sample uses only its nearest segments in τ): L = 10 dense 8.9e-4 vs 4-sparse
4.3e-3; L = 16 4-sparse 7.8e-4. Sparsity would allow k-annulus output pruning and cheaper gathers but costs ~1.6× the
segments (FFTs) for the same accuracy → **dense interpolators**.

**Toeplitz vs NUFFT form for time-resolved 4D flow.** Per (frame, encode) set the Toeplitz form needs 2L−1 (banded) or
~L²/2 (dense) kernels *per set* (80 sets × L² × grid) — far beyond GPU memory — while the per-set sample count is
small (≈ 2·10⁵ vs 9·10⁵ voxels), so the gather/scatter the Toeplitz form saves is cheap anyway. Both forms need the same
L FFT pairs per coil. **For the time-resolved reconstruction the NUFFT form is the right operator.** The Toeplitz form
with the user's σ = c(ξ)τ correction (derivative kernels, fused finite differences) is only attractive for large-K,
few-set problems (time-averaged warm start, non-resolved scans) and is deferred.

### Proposed "segmented physics NUFFT" (hasty-nufft feature)

Per set and coil batch (coil-outermost is unnecessary here; the coil batch is the transform batch):

1. **fused multi-output pad pass**: read x once; for every segment ℓ write v_ℓ = S_c x · exp(−i[ω(ρ)τ_ℓ + ρᵀC̄_ℓρ]) / φ̂(ρ)
   into the σ = 1.25 grid (ω from a stored map, the quadratic from 6 coefficients, computed in registers; zero rows
   never touched) — the first FFT pass then runs on the L·C batch;
2. **batched FFT** of the L·C grids (remaining passes);
3. **multi-channel gather**: per sample the separable weights φ(t), ∂²-products are computed once and shared by all L
   segments (grids stored segment-interleaved), the per-sample kernel is w = φ − iΣ_pq ΔC_n,pq ∂_p∂_q φ (first-order
   chirp, no extra width), the sample value is Σ_u w_u Σ_ℓ a_ℓn G_ℓ,u at the shifted k′_n (linear concomitant/B0 terms);
4. adjoint = exact transpose (multi-channel scatter, batched FFT, fused multi-input crop pass Σ_ℓ conj(D_ℓ)·).

Per-sample table: k′ (3), ΔC (6), a_ℓ (L) — sequence + B0 dependent, computed once per scan. Accuracy budget: phase
model 9e-4 (L = 10) / 4e-4 (L = 12), interpolation 1e-5. Cost ≈ L FFT pairs at σ = 1.25 per coil ≈ 2.5 plain σ = 2
FFT pairs at L = 10, plus a gather ~L× heavier in flops but reading each neighbourhood once.

Cost lever left on the table: B0 segments are the whole bill. Options: shorter readouts (sequence design), and the
lagged-anchor scheme (cheap surrogate with fewer segments every iteration, exact operator at anchors).

## 6. Follow-up questions (2026-11-03)

**Does the concomitant error fall with L?** Partly. The segments interpolate everything that depends on readout time
τ: B0 *and* the τ-dependent (isotropic / rotation-averaged) concomitant part — both errors fall with L. The
rotation-dependent remainder ΔC_n is handled by the derivative correction, whose error (the dropped second-order term)
does not depend on L: it is the floor (≈ 4e-4 for the spiral here; L = 12 → 4.3e-4, L = 16 → 3.8e-4). Lower floor:
second-order derivative terms (4th derivatives of φ, 15 more weight products per sample, still no FFT).

**B0 segments vs readout length** (B0-only CUR rank, readout-time axis stretched about TE):

| readout | L at 1e-2 | 2e-3 | 1e-3 |
|---|---|---|---|
| 4.4 ms | 8 | 9 | 10 |
| 8.8 ms | 12 | 13 | 14 |
| 13.3 ms | 17 | 18 | 19 |
| 17.7 ms | 21 | 22 | 23 |

≈ one segment per extra ms of readout for this field map (1–99 % off-resonance −222…474 Hz): L ≈ 2·Δf·T_ro + 4.
L = 10 is enough for 4–5 ms readouts; long readouts pay linearly. Levers: shim/field-map quality (Δf), the cheap
surrogate (few segments) + exact anchors scheme, accuracy target (2e-3 instead of 1e-3 saves one segment).

**3D-rotated base trajectories (yarnball-like).** C_n = R_j M(s) R_jᵀ, M(s) = ∫₀ˢ g₀g₀ᵀ: the isotropic part (tr M/3)·I
is rotation-invariant → goes into the segment diagonals (with τ); the anisotropic part rotates with every readout →
ΔC_n (`yarn_study`, 80 mT/m / 200 T/m/s, phase at ±15 cm):

| base trajectory | anisotropic / isotropic | anisotropic phase at end of readout |
|---|---|---|
| planar spiral rotated in 3D | 0.71 | 0.71 rad |
| yarnball, no polar wobble (gradient stays ~tangential) | 0.69 | 0.50 rad |
| yarnball with polar wobble, 12.7 ms | 0.14–0.25 | 0.22 rad |
| same, 25 ms | 0.14–0.25 | 0.44 rad |

So the per-readout residual is set by how isotropically the base trajectory sweeps gradient directions (a sequence
design property: worth optimising for). First order suffices at ≲ 0.2 rad (error ~φ²/2 at the FOV edge, much less
over the body); 0.4–0.7 rad needs the second-order terms; beyond that the generalised per-sample chirped kernel (the
ChirpNufft stencil route, now only for the small anisotropic remainder → small stencil width). The library kernel
should therefore be "per-sample chirped interpolation kernel" with Taylor order selectable (1, 2) and a stencil mode.

**Architecture independence.** Nothing tuned to the 4090: kernels are generated at run time (NVRTC, disk cache keyed
by arch) with parameters (σ, W, L, coil batch, frame-interpolation order, grid precision, tile shape) chosen by a
planner from device attributes (SM count, shared memory, registers, free memory) and the problem (grid size, samples
per set, sample density per tile: per-sample gather vs shared-memory tiles), plus a short autotune on first use
cached on disk per (device, problem signature). Batches (segments × coils) are chunked to the free memory.

**Motion during long readouts and long encode trains.**
1. *Between encodes / within a frame*: bin every readout by its own acquisition time (done) and replace hard bins by
   temporal interpolation: x_e(t) = Σ_f B_f(t) x_{e,f} (periodic linear or cubic B-spline in cardiac phase, nodes =
   frames). In the operator each sample gathers from the 2–4 neighbouring frames' grids of its encode — the same
   multi-channel gather (channels = segments × frames), **no extra FFTs**; the adjoint scatters into them. Velocities
   are formed from x_e(t) and x_ref(t) at the same t, so the encode-train skew (7+ encodes) disappears from the model,
   and the bin width no longer blurs the dynamics (temporal resolution set by the node spacing + the prior).
2. *During the readout*: phase γ M1_ro(τ)·v(ρ) (+ ½γM2·a) — a displacement of moving spins along the readout,
   ≈ v·t_eff(τ) (up to ~1 cm at 1 m/s and 10 ms). It depends on the unknown velocity → include it as a lagged known
   warp: v̂ from the current iterate (refreshed with the anchors), applied per τ-segment as a sparse image-domain
   resampling of the moving (vessel) voxels inside D_ℓ — same segments as B0, no extra FFT; within-band residual
   ≤ v·Δτ (≈ 0.5 mm for 0.5 ms bands) is negligible. The part of M1 not parallel to M0 (curved readouts) must be
   quantified; if large it is a per-sample "velocity chirp" with non-polynomial spatial maps (needs segments).
3. Fully model-based velocity reconstruction (unknowns m, φ₀, v directly) handles all of this naturally but is
   non-convex; recommended only as a final refinement after the linear complex-image reconstruction.

## 7. Implementation status: `SegNufft` (hasty-nufft, 2026-11-03)

Implemented as designed in §5 with first- and second-order per-sample corrections, f32, any CUDA GPU
(`crates/hasty-nufft/src/segnufft.{rs,cu}`, tests `tests/segnufft.rs`, bench `segnufft_bench`).

**Accuracy** (16×14×10, 3 coils, B0 + τ-linear quadratic + per-sample ΔC vs an f64 direct sum): order 0 L = 4/6:
7.2e-2 / 3.8e-2; order 1 L = 6/8/10: 2.4e-3 / 1.7e-3 / 1.7e-3 (floor = dropped 2nd-order term); order 2 L = 6/8/10:
1.8e-3 / 8.5e-5 / 7.7e-5. Adjoint consistent to f32 rounding. ES derivative weights are clamped to zero for |z| > 0.98
(4th derivative is unbounded at the support edge; 46× → 1e-3 interpolation error).

**How it is organised** (what made it fast):
1. Fine grid channel-interleaved `[point][coil·L + l]`, one partial-axis FFT over all channels (`FftPlan::new_partial`),
   pad/crop fused over all channels of a voxel.
2. Exact tile → sample lists (CSR, built once per `set_points`): for every 4³ tile of the fine grid the samples whose W³
   support meets it; long lists cut into work items (≤ 1024 samples; split items scatter with atomics). One block per
   (item, coil group).
3. Per staged chunk of samples: one 48-byte record per sample (k′, τ, φ0, ΔC; the 15 second-order coefficients are
   formed in the kernel as e_α = −½Σ d_p d_q), prefetched into registers one chunk ahead; kernel factors on the tile;
   the weight of every (sample, tile point) is evaluated **once** in shared memory and reused by all coils and segments.
4. Adjoint: thread per (tile point, coil) with L accumulators, no atomics, skips 32-point slabs outside the sample's
   support box (warp-uniform). Forward: the tile's values in shared memory, PPF lanes per (sample, coil) split the tile
   points and reduce with shuffles, one atomic per (sample, coil, tile).
5. Launch shapes (coil-group sizes, lanes per sample, chunk) from a planner bounded by 48 KB static shared memory and
   ≤ 512 threads (fits every CUDA GPU), refined by `autotune()` on the operator's own points (disk-cached kernels, ≈ 2.5 s
   per configuration set once compiled); `set_tiles()` shares a tuned shape across sets.

**Performance** (RTX 4090, ms per call, all coils; plain NUFFT = hasty-nufft type 2 / type 1 with ntransf = coils):

| case | operator | forward | adjoint |
|---|---|---|---|
| 180×180×28, M = 2.2e5, 12 coils | plain NUFFT σ=2 W=6 / σ=1.25 W=10 | 6.0 / 7.1 | 24.7 / 89 |
| | first SegNufft (per-segment grids, atomics) L=8 o2 | 187 | 903 |
| | SegNufft L=1 o0 (autotuned) | 5.7 | 8.7 |
| | SegNufft L=8 o0 / o1 / o2 | 30.1 / 30.9 / 31.7 | 31.2 / 31.8 / 30.1 |
| | SegNufft L=10 o2 / L=12 o2 | 40.4 / 49.2 | 35.3 / 41.0 |
| 128³, M = 1e6, 8 coils | plain NUFFT σ=2 W=6 / σ=1.25 W=10 | 12.0 / 18.4 | 54 / 122 |
| | first SegNufft L=8 o2 | 535 | 1948 |
| | SegNufft L=1 o0 (autotuned) | 17.4 | 24.5 |
| | SegNufft L=8 o0 / o1 / o2 | 65.6 / 68.0 / 67.5 | 53.9 / 55.6 / 57.1 |
| | SegNufft L=10 o2 / L=12 o2 | 80.8 / 103.8 | 68.9 / 87.5 |

The second-order correction is now almost free (o2 ≈ o0). For 180×180×28 / L=8 o2 the forward is 13.7 ms tile kernel
+ 10.7 ms FFT + 4.2 ms pad, the adjoint 15.8 ms kernel + 10.7 FFT + 2.7 crop: **the FFT/pad/crop half is memory-bound**
(≈ 760 GB/s per axis pass over 96 interleaved channels).

**Measured dead ends** (kept out of the code): restricting the forward loop to the sample's support box inside the tile
(lanes of a warp desynchronise, the tile values stop being broadcast: slower); per-(sample, segment) vs per-sample
interpolator staging (neutral); larger tiles (more zero-weight MACs than staging saved).

**Remaining levers, in order of expected gain**
1. *Multi-channel pruned FFT with fused pad/crop* (`PadFft3` already prunes zero rows for one channel): the pad pass +
   first FFT pass fused (read x, S_c, P_l once, write only the image rows), Y/Z passes reading only nonzero rows, the
   mirror for the adjoint. Estimated traffic 10.4 → 5.4 GB per call here, i.e. ≈ 15 → 8 ms.
2. ~45 % of the tile MACs multiply zero weights (W = 10 support over 4³ tiles); worth revisiting with profiler counters
   (Nsight Compute needs `NVreg_RestrictProfilingToAdminUsers=0`, currently not permitted on this machine).
3. On data-centre GPUs (TF32 tensor throughput 8× FP32) the per-tile weight × tile-value product is a small complex GEMM
   (32 × 64 × 32): 3×TF32 split-precision MMA would keep f32 accuracy. Not useful on GeForce Ada (TF32 = FP32 rate).

**Temporal interpolation (next step)** fits the same kernels: a sample at cardiac time t between frames uses
coefficients a_l(τ)·B_f(t) over L segments × F frame slots (F = 2 linear, 4 cubic). Frame slots are separate grid
blocks (a ring buffer over frames: each frame FFT once per call), the adjoint accumulates into a frame's slot over the
two (four) intervals that touch it before its inverse FFT. Gather/scatter work scales with F; FFTs do not.

### 7.1 Temporal interpolation (implemented, 2026-11-03)

`SegNufftOpts { frames, bspline: 1 | 3 }`: input = F frame images, each sample carries its acquisition time t (frame
units, node f at t = f, periodic). Samples are grouped by frame interval (own tile lists); a launch processes one
interval with NF = 2 (linear) / 4 (cubic) grid slots and coefficients a_l(τ)·b_o(u). Slots are a ring: the forward
FFTs every frame once per coil batch (wrap-around frames once more); the adjoint accumulates a frame's slot over the
intervals that touch it and flushes it (IFFT + crop, accumulated in image space — the wrap-around frames flush twice,
exact by linearity). Shared memory is dynamic and sized per launch shape up to the device's opt-in limit.

Accuracy vs f64 direct sums (B0 + τ-quadratic + per-sample ΔC + φ0, L = 10, order 2): static 7.9e-5, linear (5 frames)
7.4e-5, cubic (6 frames) 7.8e-5; adjoint consistent for 1/2/5 frames linear and 4/7 frames cubic, also with split work
items and other tile shapes.

Cost (180×180×28, 12 coils, L = 8 order 2, 20 frames × 2.2e5 samples = 4.4e6 per encode): linear 0.98 / 0.93 s
(forward / adjoint), cubic 1.9 / 1.9 s, vs 20 static per-frame calls 0.62 / 0.59 s. The gather/scatter work scales with
the number of frames a sample touches (2 / 4), the FFTs do not. Linear is the default.

### 7.2 Integration into P²nP (in progress)

`hasty-recon`: `setop::{LinOp, SetOp, SenseSets, SegSets}` (CG and P²nP generic over the operator), `segphys`
(quadratic fit of the concomitant monomials over the support in apparent coordinates; per-sample φ0 / k-shift / ΔC from
the deviation of the sample's coefficients from the encode-wide mean per ADC sample; segments from B0 + that mean at the
true (GNL) voxel positions), per-frame block-diagonal circulant preconditioners for time-resolved sets. Driver:
`recon_p2np --model seg [--temporal linear] [--phys none|b0|full] [--gnl]`.

Noise calibration (`snrcheck`, one frame-encode set, 30 PCGLS iterations, no prior, 12 virtual coils): at the simulated
noise level the per-frame blood SNR is 35 (ideal data) / 30 (realistic: blood signal lost to unmodelled phase), tissue
35 / 43 — i.e. already in the realistic range; velocity noise ≈ 0.03 m/s. The 0.30 m/s plateau of the earlier P²nP runs
was therefore not noise-limited (model / convergence).
