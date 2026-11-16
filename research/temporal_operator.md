# Temporal model inside the operator: shared FFTs over time (2026-11-15, research + bench E2; part II 2026-11-16)

**Part II (2026-11-16, §9–§14) supersedes the cost numbers of §0.4 and §4:**
- **Kernel fixes.** The adjoint scatter and the pad / crop wrote 8 B per thread at a 352 B stride (≈ 200 GB/s). With
  coalesced write-back, a static image costs 547 ms instead of 815 ms (§9).
- **New knot-major temporal operator.** One launch and one grid slot per knot, so each knot is transformed, swept and
  stored once (§10). A cubic knot image now costs what a static image costs at equal samples. At the real 1 mm density:
  - **cubic, 10 knots per resp bin: 0.99× today's per-frame time;**
  - **linear, 10 knots: 0.72×.**
  The remaining cost is per sample (s slots × MACs, re-staging), not per image.
- **Benchmark matrix** over resolution, coils, segments, order and basis: research/experiments/opbench (§11).
- **Dynamic B0 / fat with shared knots** (§12). The algebra works: time-dependent physics goes into the per-sample
  coefficients a_l(τ, t). The cost is the rank of the union basis, measured on the box field map.
- **Interpolator choice and the k-space view** (§13). Multi-GPU (§14).

Question (user): the temporal subspace (`--tsub`, x = T c) saves memory, but the gradient still expands c into frames and
runs one NUFFT/NUFFTᴴ pair per frame, so it costs the same. Can the temporal model enter the gradient itself, so that one
(NUFFT, NUFFTᴴ) pass serves readouts from slightly different times? The analogy: a higher-order integrator (RK4 instead
of Euler) takes fewer steps for the same accuracy.

## 0. Answer

1. **Yes, exactly, under one condition.** The FFT is linear and time-invariant, so it commutes with the temporal
   synthesis: F[S_c P_l x(t)] = Σ_b β_b(t) F[S_c P_l c_b]. Each temporal basis function's image is FFT'd once per pass,
   and time enters only in the gather/scatter, as a per-sample weight β_b(t_j) at the sample's own acquisition time.
   - The condition: the image-domain part of the operator (coil maps, segment diagonals, fat maps) must not depend on
     time within a basis function's support.
   - R2 violates it with its per-frame fat maps (§2.2).
2. **The machinery already exists** in hasty-nufft:
   - `SegNufft::forward_group` / `adjoint_group` with time-resolved members and B-spline frame slots;
   - slots are reused across frame intervals, so each frame image is FFT'd once per coil batch, and each sample gathers
     from its p + 1 neighbouring slots at its own time (`SegTables::t`).
   - It is wired with **knots = recon frames** (`--temporal linear|cubic`), so it adds accuracy but saves no FFTs. It is
     also blocked with `--resp-bins`, `--circ-share encode`, `--warm-full` and `--tpyramid` in recon_p2np.
3. **The RK4 analogy holds for accuracy** (§3). Today's frames are piecewise constant in time: first order. Cubic
   B-splines at sample-level time are fourth order. With **10 cubic knots per cycle** (half today's 20 frames), the model
   error of the 1st / 2nd / 3rd cardiac harmonic is 2e-4 / 0.004 / 0.034, against 0.09 / 0.18 / 0.27 for 20 bins.
4. **Per-iteration speed, measured (E2, §4).**
   - At 1 mm the cost is **per image, not per sample**: 4× more samples per image adds 17 %, because k-space is ≈ 1 %
     filled and the tile kernels sweep the grid.
   - So pass cost ≈ (number of images FFT'd and swept) × (cost per image). Shared knots replace 40 bin images per encode
     by M knot images, but a knot image costs more:
     - **1.35× a static image for linear** (s = 2);
     - **2.0× for cubic** (s = 4).
     The adjoint scatter with s slots is the bottleneck.
   - Net, per encode, against today's 40 bins:

     | basis | relative cost | model error vs 20 bins |
     |---|---|---|
     | linear, 12 knots / bin | 0.85× (1.18× faster) | ≈ 2–8× smaller up to h = 3 |
     | cubic, 10 knots / bin | ≈ 1.05× (equal) | 20–400× smaller up to h = 3 |
     | cubic, 5 knots / bin | 0.52× (1.9× faster) | better at h ≤ 2 only |

   - **Not RK4's 4× by itself.** The kernel overhead per slot is the engineering target (§4: adjoint register pressure,
     slot-flush FFT overhead).
5. **The larger gains are likely elsewhere:**
   - no binning error, and no encode-train skew;
   - fewer, better-conditioned unknowns at R ≈ 50 per frame;
   - no frame buffer and half the state (§5).

## 1. Where one gradient's time goes (1 mm box, R0 attempt 4, from the log)

- **Per set: 0.78 s.** 219 s per gradient / 280 sets, with L = 11, 22 virtual coils and joint water + fat.
- Per iteration that is 280 frame images × 242 channels (22 coils × 11 segments) × 2 directions = **135 k channel
  FFTs** of a 250 × 240 × 288 fine grid, plus the tile kernels.
- Bench (E2, §4), static image at the same grid, coils and L: 818 ms at 167 k samples, the same as a real set.
  - Split from the autotune kernel times (forward / adjoint): **tile kernels 149 / 223 ms (42 %)**,
    FFT + pad / crop 289 / 227 ms (58 %).
  - **4× the samples (668 k) adds only 17 %.** k-space is ≈ 1 % filled at 1 mm (167 k samples on a 17 M-point grid,
    each touching 10³ grid points), so the tile kernels' cost is dominated by sweeping the tiles, not by per-sample
    multiply-adds.

**Structural point (corrected by measurement):**
- **Both the FFT and the tile-kernel cost scale with the number of images transformed.** The sample count matters
  little at this sparsity.
- A temporal model with fewer images lowers both. What it costs is the extra slots each sample touches (§4).

## 2. Commuting the temporal synthesis past the FFT

### 2.1 Operator

Model per encode e (and respiratory bin):
- x_e(r, t) = Σ_b β_b(t) c_{e,b}(r);
- M basis functions β_b: periodic B-splines of degree p in cardiac phase, or Fourier.

For sample j at cardiac time t_j, readout time τ_j and k-space position k_j, with coil c:

    forward   y_{j,c} = Σ_{b: β_b(t_j) ≠ 0} β_b(t_j) Σ_l a_l(τ_j) Ĝ_{c,l,b}(k_j),     Ĝ_{c,l,b} = F[S_c P_l c_{e,b}]
    adjoint   g_{e,b} = Σ_{c,l} P_l^* S_c^* Fᴴ [ Σ_j β_b(t_j) ā_l(τ_j) r_{j,c} δ_{k_j} ]

- **FFTs per encode and pass:** C · L · M, against C · L · F today.
- **Slots per sample:** s = p + 1 for B-splines (local support); s = M for Fourier (global support: every sample
  touches every coefficient grid).

This is exact for the model; nothing is approximated. The current `tsub` gradient is the same computation in the other
order: synthesise F frames first, then F per-set FFTs. Evaluate β at bin centres instead of t_j and the two agree to f32
rounding; that is the test of §7.

### 2.2 The time-invariance condition, and what breaks it today

The pad applies the image-domain factors (S_c, P_l, fat) before the FFT. For the FFT to be shared over a basis
function's support, these factors must be the same for every sample in that support.

- **Per-frame fat-fraction maps** (R2: `--fat-model joint` with per-frame maps): the pad applies
  P_l (1 + f_t (c_l − 1)), so it differs per frame.

  Fix: **water and fat as separate unknowns** on the same temporal basis (W_b, Fa_b). The pad forms S_c P_l (W_b + c_l Fa_b)
  per channel, which is the same FFT count; the state doubles. On real data the fat fraction is unknown anyway, so this is
  also the realistic model. The truth fat fraction becomes an initialisation or soft prior instead of a hard constraint.
- **Respiratory changes of B0 / segments:** keep respiratory bins as disjoint supports (per-bin diagonals), or move a
  low-order field change per sample into the tables (the `--b0-absorb` machinery already puts a linear field per sample).
- **Rigid motion:** per sample in k-space (rotation, phase), so it is fine. **Non-rigid motion** is a per-frame image
  warp and breaks sharing. Motion-compensated reconstruction (x_t = W_t x_ref) is a different model and needs no temporal
  basis for its static part.
- **Coil maps, GNL, concomitant terms:** time-invariant or per sample already. Fine.

### 2.3 Two ways to apply the weights

- **Sample-level** (what `SegNufft` frame slots do): β_b(t_j) at each sample's own time. Exact continuous-time model.
  Gather/scatter work per sample × s (multiply-adds per slot); kernel weights and sample staging are shared over slots.
- **Frame-level:** samples binned into fine time bins, with one weight per (bin, knot). The mixing can be fused into the
  tile load (forward) and tile write-back (adjoint), so the per-sample work is unchanged (g ≈ 1).

  But each fine bin is again piecewise constant in time, so it **keeps a first-order binning error** (§3: 0.045 at 40 bins,
  0.023 at 80 for the fundamental). It is a speed device only. Removing the binning error needs sample-level weights.

## 3. Accuracy: the RK4 analogy, computed

Relative L2 error of the best approximation of one cardiac harmonic e^{i2πht} over a cycle
(`scratchpad basis_err.py`; numbers exact for the bases, independent of MRI):

| basis (coefficients per cycle) | h = 1 | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| bins 20, piecewise constant (today's frames) | 0.091 | 0.180 | 0.268 | 0.353 | 0.513 | 0.654 | 0.771 |
| bins 40, piecewise constant | 0.045 | 0.091 | 0.136 | 0.180 | 0.268 | 0.353 | 0.435 |
| linear B-spline, 10 knots | 0.015 | 0.070 | 0.192 | 0.417 | 0.915 | 1.0 | 1.0 |
| linear B-spline, 20 knots | 0.004 | 0.015 | 0.037 | 0.070 | 0.192 | 0.417 | 0.712 |
| cubic B-spline, 8 knots | 2e-4 | 0.012 | 0.129 | 0.707 | 1.0 | 1.0 | 1.0 |
| **cubic B-spline, 10 knots** | **2e-4** | **0.004** | **0.034** | **0.194** | 0.981 | 1.0 | 1.0 |
| cubic B-spline, 12 knots | 1e-4 | 0.002 | 0.012 | 0.063 | 0.707 | 1.0 | 1.0 |
| cubic B-spline, 16 knots | 0 | 3e-4 | 0.003 | 0.012 | 0.129 | 0.707 | 0.992 |
| cubic B-spline, 20 knots | 0 | 0 | 0.001 | 0.004 | 0.034 | 0.194 | 0.707 |
| cubic 10 knots, frame-level at 40 / 80 bins | 0.045 / 0.023 | 0.091 / 0.046 | 0.140 / 0.076 | 0.262 / 0.213 | 0.98 | 1.0 | 1.0 |
| Fourier \|k\| ≤ 4 (9 coefficients; R2's `fourier:4`) | 0 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 |

Reading:
- Piecewise-constant bins are first order: error ≈ πh/(N√3).
- Cubic B-splines are fourth order up to about N/4, and every N-dimensional shift-invariant space is blind above
  h = N/2.
- Fourier |k| ≤ K is exact to K and blind above it, but has global support (gather × K, §4).
- R2 feeds 20 piecewise-constant bins to the operator, whatever its basis, so **R2 carries the 20-bin binning error**
  (0.09–0.35 for h = 1–4) on top of Fourier truncation.
- The binning error applies to the dynamic part of the signal only; static tissue (h = 0) is exact in every row.
- Sample-level time also removes the **encode-train skew**: 7 encodes acquired at different times within a frame are
  treated as simultaneous today.

[CHECK]
- The harmonic content needed for coronary and aortic curves: literature, and the phantom's flows at finer time than its
  20 truth frames.
- How the sim interpolates flow between beat-library frames.

## 4. Cost per iteration: measured (E2, 2026-11-15)

Bench: `segnufft_bench`, RTX 4090 at the 450 W power limit. Grid 198 × 189 × 228 (fine 250 × 240 × 288, σ 1.25, W 10),
22 coils, L = 11, order 2. Uniform-random k-space coordinates, random per-sample cardiac times. Periodic B-spline
frames via `HASTY_BENCH_FRAMES="F,deg"`; the fine-grid budget via `HASTY_BENCH_GRID_GB` (new), which sets the coil batch.
Logs: scratchpad `e2.log` / `e2b.log`. Time is forward + adjoint for all coils.

| case | coil batch | forward | adjoint | per image | vs static |
|---|---|---|---|---|---|
| static, 167 k samples | 4 | 376 | 442 | 818 ms | 1 |
| static, 420 k | 4 | 438 | 450 | 887 ms | 1 |
| static, 668 k | 4 | 491 | 465 | 956 ms | 1 |
| linear, 4 knots × 42 k | 2 | 2024 | 3043 | 1.27 s | 1.43 (wrap) |
| linear, 10 knots × 42 k | 2 | 4666 | 7448 | 1.21 s | 1.37 |
| linear, 10 knots × 42 k | 4 (12 GiB) | 4381 | 7536 | 1.19 s | 1.34 |
| cubic, 4 knots × 42 k | 1 | 2459 | 4168 | 1.66 s | 1.87 (wrap) |
| cubic, 10 knots × 42 k | 1 | 6912 | 10850 | 1.78 s | 2.00 |
| cubic, 10 knots × 42 k | 2 (12 GiB) | 6079 | 13012 | 1.91 s | 2.15 |
| cubic, 10 knots × 42 k | 3 (18 GiB) | 5866 | 14206 | 2.01 s | 2.26 |

Split per knot image against a static image. Kernel times are from autotune × coil batches; FFT = the rest.

| | forward kernels | adjoint kernels | forward FFT + pad | adjoint FFT + crop |
|---|---|---|---|---|
| static (420 k) | 149 ms | 223 ms | 289 ms | 227 ms |
| linear (cb 2) | × 1.17 | **× 1.9** | × 1.0 | × 1.4 |
| cubic (cb 1) | × 2.4 | **× 3.3** | × 1.13 | × 1.55 |

Reading:
- **Pass cost ∝ images.** 2.5× the samples per image (167 k → 420 k) costs +8 %.
- A knot image costs **1.35× (linear)** or **2.0× (cubic)** a static image. The cost is in the **adjoint scatter with
  s slots** (thread per tile point and coil with L·s accumulators: 44 complex for cubic). It *grows* with the coil batch
  (cubic adjoint 10.9 → 14.2 s from cb 1 to 3), so register pressure, not memory [CHECK nsys].
- The adjoint FFT side carries a 1.4–1.55× overhead too (slot flushes) [CHECK where].
- Uniform-random coordinates are not radial. But the static 167 k case reproduces the real per-set time (0.82 vs
  0.78 s), so the regime is right.

Per encode and pass, against today's 40 bin images (34 s):

| option | images / encode | relative time | model error h = 1 / 2 / 3 / 4 (§3) |
|---|---|---|---|
| today, 20 bins per resp bin | 40 | 1.00 | 0.091 / 0.180 / 0.268 / 0.353 |
| linear, 16 knots / bin | 32 | 1.13 | 0.006 / 0.025 / 0.061 / 0.120 |
| linear, 12 knots / bin | 24 | 0.85 | 0.011 / 0.046 / 0.120 / 0.254 |
| linear, 10 knots / bin | 20 | 0.71 | 0.015 / 0.070 / 0.192 / 0.417 |
| cubic, 10 knots / bin | 20 | 1.05 | 2e-4 / 0.004 / 0.034 / 0.194 |
| cubic, 8 knots / bin | 16 | 0.84 | 2e-4 / 0.012 / 0.129 / 0.707 |
| cubic, 5 knots / bin | 10 | 0.52 | 0.004 / 0.194 / 0.98 / 1.0 |
| Fourier \|k\| ≤ 4, sample-level | 18 | ≫ 1 (s = 9 slots) | 0 / 0 / 0 / 0, blind ≥ 5 |
| Toeplitz normal operator | — | not viable: L² × knot-pair kernels of 0.5 GiB each per encode | |

**Conclusion.** With today's kernels the shared-knot operator gives:
- **the same cost with 20–400× less temporal model error** (cubic, 10 knots per bin), or
- **≈ 1.2–1.4× speed with 3–8× less error** (linear, 10–12 knots per bin).

The RK4-type step reduction is real in accuracy per coefficient, but the per-slot kernel overhead eats most of it in
speed.

**Engineering targets:**
1. The s-slot adjoint, e.g. slots in two register passes over the same shared-memory weights, or fewer segments per
   pass.
2. The adjoint slot-flush FFT overhead.

If both reached the forward's ratio, cubic with 10 knots per bin would be ≈ 0.8×, and linear with 10 knots ≈ 0.6×.

Fourier is the wrong basis for this path despite its accuracy per coefficient: global support means every sample
touches every coefficient image.

## 5. Memory and loop order

The existing group path runs coil batch outer, frame intervals inner, with slots reused (`held` / `assign` in
`group_fwd_in`). Live on the device: s forward slots and s adjoint partial-sum slots of cb · L channels × 138 MB.

- **Cubic (s = 4) at cb 4: 4 × 6.1 GB per direction** does not fit a 24 GB card. At cb 1 it is 4 × 1.5 GB, but the
  per-sample weights are then shared over 11 channels instead of 44 [CHECK cost: part of g(s)]. The ledger
  (research/memory_ledger.md) must choose cb and s together.
- **Tables and data are restaged once per (interval, coil batch).** At cb 1 that is 22 passes over 9.3 GiB of tables per
  iteration: 205 GiB over PCIe (≈ 8 s, overlappable) plus zstd decode of 205 GiB on the CPU, which is too much.
  - On a 24 GB card: keep the tables device-resident (9.3 GiB, fits next to 6 GB of slots, 0.75 GiB of diagonals for
    one encode and 2.2 GiB of maps), or generate them on the GPU from per-readout parameters.
  - On ≥ 48 GB cards (cluster) none of this binds.
- **Alternative order, interval outer:** tables are staged once, and each knot is FFT'd (S + p)/S times for chunks of S
  intervals.
- **State:**
  - The gradient comes out in coefficient space directly, so the frame buffer and blocks of the current tsub
    (17.8 + 2 × 1.9 GiB at 1 mm) disappear.
  - With 10 knots per bin a coefficient stack is 20 images per encode = 8.9 GiB at 1 mm, so 4 stacks = 36 GiB.
  - Water + fat separate (§2.2) doubles that to 71 GiB, back at the host limit. Needs the half-precision secant pair
    (memory_ledger.md §3.1), or knots shared with fewer fat knots (fat moves with tissue, not with flow: [CHECK] whether
    5 fat knots per bin suffice).

## 6. Beyond one iteration (the RK4 analogy in iteration count)

- **Conditioning.** At R ≈ 50 per (frame, encode), each cubic knot is fitted from the data of its 4-interval support,
  consistently weighted: view sharing with a model, not with a heuristic window. Fewer, better-determined unknowns should
  cut iterations [CHECK; this is the R2 vs R1 question, extended].
- **Temporal pyramid.**
  - B-spline spaces nest under dyadic knot refinement, so prolongation from 5 to 10 to 20 knots per bin is exact.
  - Coarse temporal levels are cheap in FFTs but not in gridding (all samples). They pay off only together with the
    spatial pyramid (central samples, coarse grid; convergence_p2np.md §5 item 4).
- **Stochastic subsets.** The FFT and gridding both scale with C. Coil or set subsets per iteration (memory: recon
  realism direction) multiply with everything above.

## 7. Experiments (pre-registered; E2 run 2026-11-15)

E1 is a correctness test. E2 measures speed. E3 measures accuracy.

- **E1, exactness test.** Group-path gradient with β evaluated at bin centres equals the tsub gradient with the same
  B-spline basis, to f32 rounding. Small grid, any GPU; a `tests/` case next to `blocked_gradient_matches_full_stacks`.
- **E2, slot cost g(s)** at the 1 mm set size: forward / adjoint kernel time with s = 1, 2, 4 slots and cb = 1, 2, 4,
  plus the channel FFT time.
  - Decision: if cubic with 10 knots per bin predicts ≤ 0.8× today, build the driver path (E3). Otherwise use linear
    knots or the frame-level fused variant, and keep sample-level cubic for accuracy-only runs.
  - **Result (§4):** cubic with 10 knots per bin predicts **1.05×**, which fails the ≤ 0.8× rule; linear with 10–12
    knots per bin predicts 0.71–0.85×.

    So by the pre-registered rule: build the driver path with **linear knots first**, and fix the s-slot adjoint (§4
    engineering targets) before cubic is the default. The cubic arm of E3 still runs as the accuracy reference.
    (Frame-level fused weights are not needed for speed: the cost is per image, not per sample.)
- **E3, dev scale (2 mm, research/sim_convergence.md fast phantoms), equal iterations.** Arms:
  - (a) per-frame 20 bins;
  - (b) tsub fourier:4;
  - (c) cubic 10 knots per bin, sample-level, water + fat separate;
  - (d) the same with 5 knots.

  Metrics: frame_fit dynamics corr / slope, ROI curves, time per iteration, memory. Config fingerprint as usual.

  Question: does (c) beat (a) and (b) on the dynamics at ≤ today's time per iteration?
- **Driver work before E3:**
  - `--temporal` with a knot count independent of the data frames;
  - `--resp-bins` (knots per bin, disjoint supports);
  - a per-encode circulant metric for the coefficient space;
  - water + fat as separate unknowns.

## 8. Literature (survey 2026-11-15)

**[V]** read in full text · **[A]** abstract or snippet only · **[U]** from a citing source.

Two operator forms cover nearly everything:
- **(A) per-sample-weight NUFFT:** K FFTs + K·M interpolation per apply, the FFT count independent of F. This is §2's
  operator with global support.
- **(B) Toeplitz:** 2K FFTs on the 2× grid + K(K+1)/2 kernels.

Nothing found uses local support to keep the interpolation at (p+1)·M in a non-Cartesian MRI operator. Nor did any
work combine a temporal subspace with off-resonance segments in fewer than L·K² kernels.

**Subspace operators**
- **T2 shuffling** (Tamir et al., MRM 2017, doi 10.1002/mrm.26102; operator in Tamir et al., IEEE SPM 2020,
  arXiv 1906.11410 [V]): Cartesian. ΦᴴPΦ commutes with the FFT, giving one K × K block per k-space point, independent of
  the number of frames. BART `pics -B`.
- **PS-Sparse** (Zhao, Haldar, Christodoulou, Liang, IEEE TMI 2012, doi 10.1109/TMI.2012.2203921 [V]): Cartesian; the
  U-step decouples into L × L blocks.
- **Assländer et al.** (MRM 2018, doi 10.1002/mrm.26639 [V]): non-Cartesian MRF, **form (A) explicitly**. The gridding
  matrix is premultiplied by the basis (G̃ = G U_R): "R FFT operations … and one sparse matrix-vector multiplication";
  the gridding burden is "increased by a factor of R". MRFingerprintingRecon.jl.
- **Chen … Christodoulou** (ISBI 2022, doi 10.1109/isbi52829.2022.9761497 [V]): golden-angle radial 2D, L = 32.
  E_Φᴴ E_Φ is an L × L block-Toeplitz with one L × L matrix per cell of the 2× grid, inverted per cell for closed-form data
  consistency (1.6 GB vs 410 GB in frame space).

  → **Use for us:** a K × K block-circulant metric per k-point, as the coefficient-space version of our per-encode
  circulant metric (`--circ-share encode`). This is the preconditioner the knot path needs (§7, driver work).
- **iGROG** (Abraham et al., arXiv 2310.10823 [V]): subspace + time segmentation "will require L⋅K² Toeplitz PSFs".
  Replaces the 2× embedding with GRAPPA-kernel gridding to Cartesian, field absorbed per sample; 40 FISTA iterations
  16 → 4.3 min.
- **BART** `upper-triag-psf` / `compress-psf` [A]; Maatman et al. ISMRM 2026 (≈ 8× lower Toeplitz-subspace memory by
  PSF symmetry and compact support) [A/U]; Cao et al. MRM 2022 (3D MRF, K = 5 limited by time and RAM: 3–5 h,
  ≥ 200 GB) [V]; Scholand et al. MRM 2025 (rational golden angles: 100 frames need 13 PSFs) [V].
- **Extreme MRI** (Ong et al., MRM 2020, doi 10.1002/mrm.28235 [A]): keeps the per-frame cost but uses stochastic
  optimisation over frames.

**Local-support temporal bases**
- **List-mode PET**, Nichols, Qi, Asma, Leahy (IEEE TMI 2002, doi 10.1109/TMI.2002.1000263 [A]): per-voxel cubic
  B-spline rate functions fitted at each event's own arrival time; each event touches 4 coefficients. **The closest
  analogue of §2.3's sample-level weights.** Fully 4D follow-up: Li et al., IEEE TMI 2007 [A].
- **Levine & Hargreaves** (arXiv 1710.07837 [V]): C(r, c) β_l(t) "viewed as a subspace model with a predefined temporal
  basis"; 4 periodic cubic B-splines for 12 frames; Cartesian simulation.
- **Low-rank MR-MOTUS** (Huttinga et al., MRM 2021, doi 10.1002/mrm.28562 [A]): spatial and temporal B-splines for
  motion fields.

**Binning and view sharing (the zeroth-order baseline)**
- Tornado / HYPR (Barger MRM 2002; Mistretta MRM 2006 [A]): k-dependent windows. Edges get a "reduced temporal
  bandwidth" and "appear too early".
- PC-VIPR: retrospective gating by view sharing, "14 interpolated time frames per R-R" (Landgraf JMRI 2014 [A]).
- **Montalba et al.** (MRM 2018, doi 10.1002/mrm.26834 [A]): at 2 mm / 40 ms, peak flow −10.5 %, mean velocity −10 %,
  stroke volume −5 %. Flow metrics are "more sensitive to temporal than spatial" resolution.
- A boxcar bin of width Δ scales cardiac harmonic n by sinc(nΔ/T): 0.90 for the 5th harmonic at 20 frames, consistent
  with the 10 % peak losses above. This is the same first-order effect as §3's table.
- Shared velocity encoding (Lin et al., MRM 2012 [A]): peak-velocity error 8.8 → 5.6 % in simulation. This is the
  encode-train skew that sample-level time removes.

**Off-resonance time segmentation (the same mathematics on the readout-time axis)**
- Sutton, Noll, Fessler (IEEE TMI 2003, doi 10.1109/TMI.2002.808360): L NUFFTs per apply, i.e. form (A) with K → L.
  This is our SegNufft.
- Fessler et al. (IEEE TSP 2005, doi 10.1109/TSP.2005.853152 [V]):
  - Toeplitz with L² kernels (any approximation);
  - an **"L approach"** that segments the Gram's difference term e^{−i(ω_j − ω_k)t} directly: L kernels and L FFT pairs,
    but twice the bandwidth, so L grows (8 vs 6).

**4D flow:** all temporal-basis or low-rank 4D flow work found is Cartesian or binned (Sun / Zhao MRM 2023 partial
separability; Walheim JCMR 2019 LLR; Ma 2020 5D radial CS over bins; Pruitt ReVEAL4D 2021) [A]. **No non-Cartesian 4D
flow reconstruction with the temporal basis inside the operator was found.**

**A derivation (survey agent, not from the literature; checked by me).** With a Fourier basis the Toeplitz kernel
T_{kk'} depends only on k′ − k. So the normal operator is a single (3 + 1)-D Toeplitz convolution over (k-space ×
harmonic): pad to G × 2K, one 4-D FFT pair, one kernel. With segments by Fessler's L approach that is L·(2K − 1)
kernels. At 1 mm one 4-D kernel is 2K × 0.5 GiB ≈ 9 GiB, so this is not viable on 24 GB with L = 11. It is worth
remembering for coarse temporal levels or large-memory nodes.

**Gaps:**
- accuracy vs knot spacing in MRI;
- within-bin first-order / Taylor temporal models;
- a joint subspace × segments factorisation below L·K²;
- Kalman / state-space methods for PC-MRI.

# Part II (2026-11-16): kernels, knot-major operator, dynamic physics, multi-GPU

All timings: RTX 4090 at the 450 W limit. 1 mm box grid 198 × 189 × 228 (fine 250 × 240 × 288, σ 1.25, W 10), 22 virtual
coils, L = 11, ΔC order 2, uniform-random k-space; forward + adjoint for all coils unless stated.
Bench: `segnufft_bench`. Profiles: nsys timelines; ncu is not available (RmProfilingAdminOnly = 1).

## 9. Where a static image's time went, and the coalescing fix

nsys, one static image with 167 k samples:

| part | before | after | note |
|---|---|---|---|
| pad (S_c P_l x → 44-channel grid) | 98 ms | ≈ 27 ms | one thread per voxel stored 44 channels at a 352 B stride → staged in shared memory, contiguous runs |
| FFT, forward, 3 axis passes | 180 ms | 180 ms | ≈ 950 GB/s on the full pass: at bandwidth |
| gather (tile kernel) | 98 ms | 80–98 ms | tile-value loads ≈ 60 % of it |
| scatter (tile kernel) | 214 ms | ≈ 55 ms | **store-bound**, see the ablation below |
| inverse FFT | 176 ms | 176 ms | at bandwidth |
| crop | 50 ms | ≈ 30 ms | same fix as the pad |
| **total** | **815 ms** | **547–565 ms** | |

Ablation of the scatter, using compile-time switches `PROF_NOMAC`, `PROF_NOSTAGE` and `PROF_NOSTORE` in segnufft.cu:
- Removing the MACs and the staging changed nothing (440 → 437 ms).
- Removing the final store saved 180 ms of the kernel's 214 ms.

Each thread wrote its 11 channels (88 B) at a 352 B stride, which ran at ≈ 200 GB/s.

**Fix (`adj_store`):** the accumulators are parked slot by slot in the free stage area of shared memory (odd stride,
conflict-free) and written as contiguous runs of (coils in group × NSEG) channels per tile point.
- The scatter kernel per coil batch went 38.9 → 9.9 ms.
- The pad and crop use the same pattern (`pad_core` / `crop_core`, 32 voxels per 128-thread block), including the fat
  variants.
- The old kernels remain behind `HASTY_SEG_DEFS=SEG_STORE_DIRECT,SEG_PADCROP_DIRECT`, for comparison only.
- Tests (direct sums, adjointness, multi, fat union, pruned, streamed) pass. `seg_stream` sits at its 2e-5 bound with
  the old kernels as well.

**At high density (1.34 M samples on one static image, the per-knot load of cubic knots)** the kernels are:
- MAC ≈ 150 ms, staging ≈ 100 ms, list and tile-value loads ≈ 130 ms (forward), store ≈ 90 ms (adjoint).
- The MACs run at ≈ 60 % of FP32 FMA peak. That is why register blocking (several coils per adjoint thread, `acb`;
  several samples per forward lane, `fsb`; now TileCfg fields, explored by the autotune) never won.
- **About half of the MACs hit zero weights.** A sample's support covers 1000 of the 43 × 64 tile points it touches:
  TT = 4 tiles against a W = 10 support. This is now the largest kernel inefficiency (§15 item 3).

## 10. Knot-major temporal operator (`SegNufft::new_temporal(.., knot_major = true)`, segnufft_knot.cu)

The interval-major kernels (§5) run one launch per frame interval. Each launch touches the s = p + 1 slots of its
frames, so every slot is read (forward) or read-modified-written (adjoint) by s launches, and s slots of cb · L
channels must be resident: cb drops to 1–2 at 1 mm. That was the 1.35× / 2.0× of §4.

**Knot-major, one launch per knot b:**
- forward: pad + FFT of c_b into the one slot, then gather of every sample whose B-spline support contains b, weighted
  by β_b(t_j);
- adjoint: scatter of the same samples, one store per grid point, then IFFT + crop into c_b.

Each knot is transformed, swept and stored once, and one slot is resident, so cb = 4 at 1 mm.

**Tables.** One tile list per set, each tile's entries sorted by frame interval, entry = (interval << 24) | sample.
- A block filters its item to the knot's window (intervals b − KOFF − s + 1 .. b − KOFF, periodic). Since entries are
  interval-sorted, that is at most two contiguous runs, found with `__syncthreads_count` and compacted into the shared
  list.
- β_b(t_j) is folded into the staged per-sample coefficients.

**Correctness (tests/seg_knot.rs).** The knot-major operator equals the interval-major one to f32 rounding:
- forward 4–5e-6, adjoint 1–3e-5 relative per element;
- normalised adjointness 3e-8 – 2e-7 for both layouts;
- cases: linear and cubic; the minimum knot count (window = period); 40 k samples with split tile lists; orders 0–2.

**Speed**, forward + adjoint:

| case | interval-major | knot-major | per knot image |
|---|---|---|---|
| linear, 10 knots × 42 k samples | 7.66 s (cb 2) | **5.53 s** (cb 4) | 0.98× static |
| cubic, 10 knots × 42 k | 13.26 s (cb 1) | **5.70 s** (cb 4) | 1.01× static (was 2.0×) |
| cubic, 10 knots × 334 k (real density) | 17.2 s | **10.8 s** | |
| linear, 10 knots × 334 k | 9.6 s | **7.9 s** | |

At equal samples per knot, the per-image cost of the temporal basis is gone. At the real density each sample is
gathered and scattered s times (inherent to sample-level time) and staged s times (not inherent), so per-sample work
dominates (§11).

Scope: order ≤ 2, single-image forward / adjoint / residual. The multi-image, group, fat-map and warp paths return an
error on a knot-major engine. The interval-major path is unchanged and selectable; neither replaces the per-frame
operator.

## 11. Benchmark matrix (research/experiments/opbench/matrix.sh → results_2026-11-15.csv)

- **Unit:** one encode in one respiratory bin, i.e. what a temporal basis shares. It holds 3.34 M samples at 1 mm,
  scaled ∝ 1/res.
- **Per-frame:** 20 static images of unit / 20 samples (today's operator), time × 20.
- **knot / interval:** the B-spline operators over M knots.
- Autotuned launch shapes; `acb` / `fsb` off by default (`ACBFSB=1` enables them).

1 mm, 22 coils, L = 11, order 2 (unit time, forward + adjoint):

| operator | unit time | vs per-frame | model error h = 1 / 2 / 3 (§3, §13) |
|---|---|---|---|
| per-frame, 20 bins | 10.94 s | 1 | 0.091 / 0.18 / 0.27 |
| knot linear, M = 10 | 7.91 s | **0.72** | 0.015 / 0.070 / 0.19 |
| knot linear, M = 20 | 13.3 s | 1.22 | 0.004 / 0.015 / 0.037 |
| knot cubic, M = 8 | 9.78 s | **0.89** | 4e-4 / 0.012 / 0.13 |
| knot cubic, M = 10 | 10.78 s | **0.99** | 2e-4 / 0.004 / 0.034 |
| knot cubic, M = 20 | 15.9 s | 1.45 | ≈ 0 / ≈ 0 / 0.001 |
| interval linear, M = 10 | 9.59 s | 0.88 | as knot |
| interval cubic, M = 10 | 17.2 s | 1.57 | as knot |

Settings at 1 mm (per-frame / knot cubic 10 / knot linear 10):

| setting | per-frame | cubic 10 | linear 10 |
|---|---|---|---|
| L = 6 | 7.03 s | 7.26 s (1.03) | 5.18 s (0.74) |
| 12 coils | 5.59 s | 5.46 s (0.98) | 4.01 s (0.72) |
| order 0 | 10.92 s | 10.11 s (0.93) | 7.44 s (0.68) |

**§11b. Resolution** (unit time relative to per-frame; per-frame unit time in brackets):

| operator | 1 mm (10.9 s) | 1.5 mm (3.91 s) | 2 mm (2.00 s) |
|---|---|---|---|
| knot linear, M = 10 | **0.72** | **0.86** | 0.99 |
| knot linear, M = 20 | 1.22 | 1.35 | 1.42 |
| knot cubic, M = 8 | 0.89 | 1.17 | 1.50 |
| knot cubic, M = 10 | 0.99 | 1.26 | 1.74 |
| knot cubic, M = 20 | 1.45 | 1.73 | 1.98 |
| interval linear, M = 10 | 0.88 | 0.96 | 0.98 |
| interval cubic, M = 10 | 1.57 | 1.49 | **1.56 (beats knot-major)** |

Reading:
- **At 1 mm, cubic knots buy 20–400× less temporal model error at equal cost; linear knots save 28 %.** The ratio is
  stable across coils, segments and correction order.
- **Which formulation is fastest depends on the regime.** Lower resolution means more samples per fine-grid point
  (2 mm: coil batch 16, small grids, cheap FFTs), so the per-sample work dominates. The knot operator's s-fold
  per-sample work then costs more than the FFTs it saves.
  - At 2 mm, per-frame ≈ linear knots, and cubic is 1.5–1.7×.
  - At 2 mm, interval-major cubic beats knot-major: it stages each sample once for all s slots, and its s-slot traffic
    is cheap when the grids are small.
- Consequently all formulations stay: per-frame, tsub, interval-major and knot-major. The driver should choose by the
  measured cost model (this matrix), not by a fixed rule.
- The accuracy side does not depend on resolution, so at 2 mm the cubic basis is an accuracy purchase (1.5–1.7×), not
  a speed one.

## 12. Dynamic B0 and fat with shared knots: the algebra

Model per sample j (cardiac time t_j, readout time τ_j, respiratory coordinate ρ_j):

    y_j = Σ_b β_b(t_j) Σ_l a_l(τ_j, t_j, ρ_j) F[S_c P_l c_b](k_j)

**Rule.** Time-dependent physics must not enter the image-domain factors (P_l, S_c, fat maps) that sit before the
shared FFT. It can enter the **per-sample coefficients a_l**, which the kernels already apply per sample at no FFT
cost.
- The condition for exactness: the spatial basis P_l must span exp(−iω(r, t_j, ρ_j) τ_j) for **all** samples (a union
  basis).
- The price is the rank of that union, L(ε), and the FFT count is ∝ L.
- The temporal basis acts on the image; the segments act on the physics. They factor.

Measured on the box field map (field_ppm_recon, 11.44 ms readout, 3 T, 30 k object voxels; scratchpad
`b0_union_rank.py`). L at relative error ε = 1e-2 / 3e-3 / 1e-3:

| model | L |
|---|---|
| static field, water | 15 / 17 / 18 |
| static, water + fat **pad-combined** (W + c_l Fa before one FFT; rows (r, species) stacked) | 19 / 22 / 23 |
| static, water + fat with its own coefficients c(τ) a_l(τ) | 15 / 17 / 18, but two FFT sets |
| additive smooth change ±0.05 / ±0.1 / ±0.2 ppm (SI ramp + bump), union over the change | 23–32 / 25–36 / 29–42 |
| the same after moving the degree ≤ 4 polynomial part into per-sample k-space terms | 16–24 / 17–27 / 19–29 |
| field moving with tissue (Lagrangian shift up to 5 / 10 mm SI) | 106–134 / 113–141 |

**Consequences, by physics:**
1. **Fat.** Per-frame fat-fraction maps exist only because x is one species. With **water and fat as separate unknowns
   W_b, Fa_b on the same knots**, no fat map is needed and nothing is time-dependent. The pad forms
   S_c P_l (W_b + c_l Fa_b): the same FFT count as one image at the union L (+27 % over water-only, the cost today's
   joint fat model already pays). Fat moves with tissue, so it needs knots too, perhaps fewer [CHECK].
   - **Done (2026-11-16):** `SegNufft::forward_wf` / `forward_residual_wf` / `adjoint_wf`, x = [W frames | Fa frames],
     kernels `seg_pad_wf` / `seg_crop_wf` (`pad_core` / `crop_core` mode 2).
     - It works on per-frame, interval-major and knot-major engines.
     - tests/seg_wf.rs: equal to the fat-map path with W = (1 − f) x, Fa = f x, to 4e-6 forward and ≤ 1e-5 adjoint,
       static, cubic interval-major, and cubic / linear knot-major.
   - Driver work left: the doubled state, and the truth fat fraction as initialisation / prior instead of a constraint.
2. **Respiratory B0 with tissue motion** (Lagrangian, ≈ 100+ segments as a union) is not a union problem. Keep
   **respiratory bins as disjoint supports** with per-bin diagonals (today's approach). Each (bin, encode) unit has its
   own knots; the cardiac knot sharing is unaffected.
3. **Smooth susceptibility changes** (respiration or cardiac, ≤ 0.1 ppm) on top:
   - the linear part is a per-sample k-space shift (exact, the `--b0-absorb` machinery);
   - quadratic through ΔC (second order in the kernel);
   - degree 3–4 through the order-3 path (first order only: valid for phases ≲ 0.3 rad);
   - the residual by union segments: +2–9 at ±0.1 ppm.
   - The 2-D interpolator a_l(τ, ρ) needs a bilinear table or concatenated per-state tables (τ offset per state): a
     ≈ 10-line kernel change, not done.
4. **Cardiac-phase-dependent B0** (small): the same as 3, with ρ → t_j.
5. **Non-rigid motion** inside a basis function's support: not expressible by a_l. That is a motion model (warp), not
   a temporal basis.

## 13. Interpolator choice, conditioning and the k-space view

**k-space view.** A temporal B-spline basis at sample-level time is a (3 + 1)-D NUFFT whose fourth axis is not
transformed. The knots are the grid along time, and β_p is that axis's interpolation kernel (width s = p + 1).
- Everything spatial (gridding, ΔC terms, segments) is unchanged.
- The temporal axis adds a separable per-sample weight.
- The algebra commutes for any linear basis. What the basis decides is accuracy per image, slots per sample (per-sample
  cost) and conditioning.

Accuracy and conditioning (scratchpad `basis_err2.py`: best approximation of e^{i2πht}; cond = condition number of the
basis Gram under uniform sample times):

| basis (images, slots) | h = 1 | 2 | 3 | 4 | 5 | cond |
|---|---|---|---|---|---|---|
| linear, M = 10 (10, 2) | 1.5e-2 | 7.0e-2 | 0.19 | 0.42 | 0.71 | 3 |
| cubic, M = 8 (8, 4) | 4.4e-4 | 1.2e-2 | 0.13 | 0.71 | 0.99 | 18.5 |
| cubic, M = 10 (10, 4) | 1.7e-4 | 4.0e-3 | 3.4e-2 | 0.19 | 0.71 | 18.5 |
| cubic, M = 12 (12, 4) | 7.7e-5 | 1.7e-3 | 1.2e-2 | 6.3e-2 | 0.25 | 18.5 |
| quintic, M = 8 (8, 6) | 8.7e-6 | 1.4e-3 | 4.7e-2 | 0.71 | 1.0 | 113 |
| quintic, M = 10 (10, 6) | 2.0e-6 | 2.5e-4 | 6.2e-3 | 8.7e-2 | 0.71 | 113 |
| septic, M = 10 (10, 8) | 2.4e-8 | 1.5e-5 | 1.1e-3 | 3.9e-2 | 0.71 | 687 |
| KB-gridded Fourier \|k\| ≤ 4, σ 2, W 4 (18, 4) | 3.4e-4 | 2.9e-4 | 4.9e-4 | 6.0e-4 | 3.2e-3 | 43 |

Reading:
- **Degree buys accuracy per FFT.** Quintic with 10 knots is cubic-with-16 quality on 10 images. It costs slots, i.e.
  per-sample work (6 vs 4) and conditioning.
- Every M-dimensional shift-invariant space is blind above h = M/2.
- With knot-major kernels, a slot costs per-sample work only, so the choice is set by the density regime:
  - sparse (1 mm): higher degree is nearly free;
  - dense (lower resolution): fewer slots.
- **Conditioning fix.** For uniform periodic knots, the temporal Gram (Σ_j β_b β_b′ per voxel, sampling uniform in t)
  is circulant. Its inverse is one length-M FFT per voxel: an exact temporal preconditioner, composable with the
  spatial circulant metric. Chen et al. (ISBI 2022, §8) do the K × K block version per k-point. The condition numbers
  above (18–687) are the case for it before any degree > 3.
- **KB-gridded Fourier** (the Fourier temporal basis evaluated by a temporal NUFFT: 2K + 1 coefficients, FFT along t
  onto σ(2K + 1) grid images, KB kernel of width W_t). It is bandlimited-accurate with local support. Per image it
  beats cubic splines only at h ≥ 3, and it needs deapodisation and an oversampled time grid. B-splines are simpler and
  nest dyadically (the temporal pyramid, §6).
- **Normal operator in k-space.** With local support, AᴴA in coefficient space is banded in knots: 2p + 1 diagonals
  instead of M², so p + 1 distinct Toeplitz kernels per knot by symmetry. Still L² × that at 1 mm, so not viable on
  24 GB (§4), but this is the structure a coefficient-space circulant metric uses.

## 14. Multi-GPU layout

**Units of work** (§11): (respiratory bin, encode) pairs, 2 × 7 = 14 per gradient at the box config. Each holds M knot
images and its own tables and segments.
- Within a unit, coil batches are independent except for the adjoint's sum into the unit's knot images.
- Units are independent in the forward and in the adjoint (disjoint unknowns) under the per-encode complex model. Under
  the velocity model the encodes share unknowns: a sum over encodes.

**Plan:**
1. **Task = (unit, coil batch); static assignment by the cost model**, where unit time ≈ M · (FFT + sweep per image)
   + s · samples · (per-sample cost). The model comes from the matrix (§11); it is the ledger's job
   (research/memory_ledger.md) to also place tables and state per GPU.
   - Granularity: 14 units over 4 GPUs is 88 % balanced. Splitting units by coil batch (6 per unit at cb = 4) gives 84
     tasks, ≥ 97 %.
2. **Per GPU:** coil maps (1.5 GB at 1 mm, or only its coil subset when tasks are coil-split), the segment diagonals of
   its units, the knot images of its units, one grid slot (knot-major), and double-buffered table streaming from host
   or the KStore (kload).
   - No FFT is split across GPUs: one image fits one card.
3. **Reductions:**
   - coil-split units: the adjoint's partial knot images, M × nv × 8 B = 1.4 GB per unit at 1 mm (NVLink: ms; PCIe 4:
     ≈ 56 ms against ≈ 10 s of unit compute);
   - velocity model: the gradient over shared unknowns, ≈ 5 GB per gradient (≈ 0.2 s over PCIe).
   - Scalars (dot products, line search) through the host.
4. **Overlap:** table uploads on a second stream (`stream_prefetch`) and reductions after each unit's last coil batch,
   so communication hides behind the next task. Expected scaling is near-linear while units ≥ GPUs. The knot-major
   one-slot layout makes cb independent of s, so 48–80 GB cards (cluster) take all coils in one batch.
5. **Stochastic coil subsets** (memory: recon realism) reduce every term above proportionally and fit the same task
   model.

## 15. Next steps, ranked by expected gain (measured basis above)

1. **Water + fat as separate unknowns** (§12.1): the engine side is done (`forward_wf`). The driver state is left. It
   unlocks the knot operator for the fat model that R2 uses.
2. **Driver:** `--temporal` with knots ≠ frames on a knot-major engine, per (resp bin, encode) unit, merged cardiac sets;
   then E1 (bins ↔ knots exactness on the driver) and E3 (dev-scale accuracy).
3. **Wasted MACs (≈ 50 % zero weights)** in both tile kernels.
   - Measured (2026-11-16, `HASTY_SEG_TT`):
     - TT = 2 is slower: static 608 vs 548 ms, knot cubic 10 at real density 15.5 vs 10.8 s. Per-tile overheads and
       twice the table build win over the saved MACs.
     - TT = 8 does not fit 48 KB of shared memory at L = 11.
   - So the fix is structural: lanes assigned to the support box, or a separable two-pass gather for order 0–1.
     Expected ≤ 1.5× on the tile kernels, which now dominate the knot path.
4. **Staging once per sample for all s knots** while keeping one store per knot: a tile-major window that rotates s
   register accumulators as the interval-sorted list advances. It needs the knots' slots resident for the window only
   (s slots) and writes each tile once. Removes the s× re-staging (≈ 100 ms per 1.34 M samples).
5. **FFT bytes:** fuse the pad into the first forward axis pass and the crop into the last inverse pass (≈ 13 % of a
   static image). Half-precision grid storage with fp32 arithmetic (≈ 2× on FFT and tile traffic) needs an accuracy
   study first. **Coil maps as k-space kernels** would cut FFT channels from C · L to L (22×). The catch is the
   per-sample combined kernel with ΔC corrections (§9 analysis in the session notes); measure the maps' k-space
   compactness first.
6. Temporal circulant preconditioner (§13) before any degree > 3 basis.

## 16. Respiratory × cardiac (2-D) knots: engine and first test (2026-11-16)

**Engine.** `SegNufft::new_temporal2d(.., kr, rdeg)` + `points2d`, segnufft_knot.cu with `RDEG`.
- Knot (k, b) = respiratory knot k × cardiac knot b, image layout k · M + b (as respiratory bins today).
- Respiratory basis non-periodic: degree 0 = bins, degree 1 = linear hats. Cardiac basis periodic B-spline as before.
- Per sample: cardiac interval + fraction, respiratory interval + fraction (packed 16-bit fractions in the record).
- Tile entries sorted by key (ri · M + ic), entry = key << 25 | sample (≤ 33 M samples, ≤ 127 keys). A knot's window
  is ≤ 4 contiguous runs, bounds by binary search (no scan of the tile list per knot).
- One code path for bins and splines, so arms differ only in the basis.
- tests/seg_knot2d.rs: against a per-interval reference built from 1-D knot engines with host-side respiratory weights
  (bins, linear, split lists). Forward ≤ 1.1e-5, adjoint ≤ 2e-5 relative; normalised adjointness ≤ 1.1e-7.

**Why the respiratory knots get their own image, not a shared physics basis (§12).** Each knot carries the image of its
state. Between knots the *signal* (image and its off-resonance phase) is interpolated: bins are the zeroth-order version,
linear knots the first-order one, at no extra segment rank. The open risk is moving edges: interpolating between two
positions of an edge gives a double edge unless the displacement per knot step is ≲ an edge width
(research/model_robustness.md §§ on motion: in its 1-D toy a 15 mm move needed 17 knots, warp + residual basis 5).

**Test (pre-registered 2026-11-16).** 2 mm scan `sim_flow_10min` (254 × 144 × 114, 53 144 readouts × 3990 ADC, 7 encodes,
10 min).
- **Recon:** `splinerecon` per encode: CGLS with no regulariser, 20 iterations (saves at 5 / 10 / 20).
  - B0 segments (10, field map), true coil maps (strongest 8), no concomitant / GNL / fat model, the same for every arm.
  - Breathing coordinate = the sim's true state.
- **Arms:** respiratory 8 bins / 6 linear knots / 4 linear knots, each × 10 cubic cardiac knots (80 / 60 / 40 knot
  images per encode). The 8 × 20 cardiac-bin baseline (160 images per encode) does not fit the per-encode CG memory on
  24 GB.
- **Score:** frame_fit at three breathing states with their own truths (r = 0.07, 0.181, 0.493). Metrics: pooled
  depth 0–2 (vessel-edge band, where motion blur and double edges show), dynamics depth ≥ 2 / ≥ 5, ROI dynamics.
- **Question:** do linear respiratory knots (6 or 4) reach equal or lower error than 8 bins, at ≤ 1× the time per
  iteration?
- **Decision:**
  - if yes for 6 knots: the knot driver path gets respiratory splines;
  - if only for 6 and not for 4: knot count is motion-limited (the edge caveat), and the motion-compensated hybrid
    (warp + residual basis, model_robustness.md) is the next design;
  - if no: bins stay, diagnose (edge band vs deep lumen).
- **Caveats fixed in advance:**
  - The truth states cover only the lower half of the breathing range (median 0.48, max 1.4).
  - The recon ignores fat / GNL / concomitant terms (common to all arms).
  - Breathing is the true state, not bellows (bellows comes later, as a robustness arm).

### 16.1 Measured on the phantom (2026-11-16): how many knots the motion and the flow need

**Respiratory motion** (motion/resp_motion.hv: u = r · U(X), r ∈ [0, 1.4], 10.3 mm mean at the heart at r = 1):
- 4 cm sphere around the heart, at r = 1:
  - raw: |u| mean 10.3 mm, max 13.9 mm;
  - after a translation fit: 2.0 mm rms, 5.7 mm max;
  - **after an affine fit: 0.32 mm rms, 3.0 mm max.**
  - 6 cm sphere (more chest wall): affine leaves 1.8 mm rms.
- Heart-centre travel per knot step, knots at r quantiles:
  - 4 linear knots: 2.5 / 4.8 / 7.0 mm;
  - 6 linear knots: 0.9–5.5 mm;
  - 12 linear knots: ≤ 1.4 mm, except the last step (4.2 mm);
  - 8 bins: within-bin travel 0.4–4.6 mm.

  At 2 mm, 4–6 knots are at or past the double-edge limit; for 1 mm coronaries none of these suffice alone.
- **Consequence: the heart's breathing motion is nearly affine**, and an affine motion is exact per sample in k-space
  (phase ramp + k-coordinate transform + Jacobian). It costs no FFTs and keeps the knot operator. The design to test:
  per-sample affine heart motion A(r) = r · A₁ (12 parameters, estimated, shrunk and validated on held-out k-space;
  research/model_robustness.md), plus a few respiratory knots for the residual (≈ 0.3 mm rms).

**Cardiac content of the flow** (truth_vel_logical, 20 frames, deep lumen):
- Velocity dynamic energy per harmonic h = 1…10: 0.47 / 0.25 / 0.14 / 0.047 / 0.028 / 0.027 / 0.017 / 0.008 /
  0.006 / 0.003 (91 % in h ≤ 4).
  - The PC phase curves look broadband (66 % in h ≤ 4), but that is phase-wrap nonlinearity, not flow.
  - The truth itself is 20-frame binned, so the content above h ≈ 5 is underestimated.
- Model error of each basis on these velocity curves:

  | basis | images / cycle | error |
  |---|---|---|
  | 20 bins (today) | 20 | 0.234 |
  | cubic 10 | 10 | 0.276 |
  | cubic 12 | 12 | 0.220 |
  | cubic 16 | 16 | 0.121 |
  | cubic 20 | 20 | 0.054 |
  | cubic 24 | 24 | 0.018 |
  | quintic 16 | 16 | 0.116 |
  | linear 20 | 20 | 0.090 |

- **Correction of §0 / §11.** "Cubic 10 knots = 20–400× less error" holds per harmonic for h ≤ 3 only. For this
  phantom's spectrum:
  - cubic 10 is slightly *worse* than 20 bins;
  - equal error needs ≈ 12 cubic knots;
  - half the error needs 16;
  - 4.3× less error needs 20 (the same image count as today).

  With the 8-respiratory-bin density (§16 timings), knot images cost ≈ 0.55× a per-frame image at 1 mm. So:
  - cubic 16 ≈ 0.9× the time at half the cardiac model error;
  - cubic 20 ≈ 1.1× the time at a quarter of it.

  Faster *and* more accurate holds at 16 knots; at 20 it buys accuracy at about equal time.

### 16.2 Results (2026-11-16)

**Gradient-pass time** (one forward + adjoint, encode 0 = 30.3 M samples, 2 mm, 8 coils, 10 B0 segments, W 8, order 0,
`--time-only`, same 2-D engine; slots = respiratory × cardiac support):

| basis (resp × cardiac) | knot images per encode | slots / sample | pass time | vs today |
|---|---|---|---|---|
| **8 bins × 20 bins (today's per-frame layout)** | 160 | 1 | **13.8 s** (8 × 1.73 s, one bin timed with `--resp-keep`) | 1 |
| 8 bins × 10 cubic | 80 | 4 | 14.0 s | 1.01 |
| 6 linear × 10 cubic | 60 | 8 | 19.0 s | 1.38 |
| 4 linear × 10 cubic | 40 | 8 | 17.2 s | 1.25 |
| 4 linear × 20 bins | 80 | 2 | 10.9 s | 0.79 |

At 2 mm with 30 M samples per encode, **the cost is per sample × slots, not per image**, as the 1-D matrix predicted for
2 mm (§11b). Halving the images while doubling or quadrupling the slots per sample does not pay here. The 1 mm regime,
sparse per grid point, is where the per-image saving dominates. It is not yet measured in 2-D. The fix for the dense
regime is the per-sample work: the wasted zero-weight MACs (§15 item 3) and the per-slot re-staging (item 4).

**Accuracy: inconclusive. The test is solver-limited.**
- Unregularised CGLS from zero, 20 iterations: every arm reaches a pooled velocity-phase slope of only 0.03–0.04, and
  arms lin6 / lin4 score within ±0.002 of each other everywhere (scores.csv).
- Checks (encode 1 − 0, depth ≥ 5, 10 iterations):
  - B0 only: slope 0.045, corr 0.47;
  - + concomitant: 0.049 / 0.46;
  - + GNL: 0.049 / 0.46;
  - + warm start (static image, 10 preconditioned iterations): 0.163 / 0.27.
- So missing physics is not the cause. The iterate is far from converged (gradient norm 7–17 % of its start), and the
  arms' differences sit below the solver error.
- R2 (full driver, preconditioned, warm start) reached a pooled slope of 0.28 on the 1 mm box. This test needs the
  same:
  - a preconditioner for knot stacks (per-knot circulant scaled by the knot's sample weight energy);
  - classical regularisation (spatial wavelet + temporal LLR, as the user proposed);
  - iterations to convergence.

  Only then are basis differences measurable. Pre-registered decision: **no conclusion on bins vs splines from this
  run.**

The 8-bin CG arm needed 21.7 GiB against 21.3 GiB free. The new ledger refused it cleanly (no OOM). CGLS without a
preconditioner now skips its z vector (cg.rs), which saves 2.5 GiB, so it fits.

## 17. Dense-regime kernel work (2026-11-16/17)

**Starting point (§16.2).** 2 mm, 254 × 144 × 114, 2-D lin6 × cubic10 (8 slots per sample), 30.3 M samples per encode,
8 coils in 2 batches of 4, L = 10, W = 8, order 0: one forward + adjoint = 18.97 s. Ablation (`HASTY_SEG_DEFS`): MACs
≈ 9.4 s, staging ≈ 4.6 s, FFT + pad + crop ≈ 3.7 s, list / window / prefetch / store the rest.

**Two wastes in the MACs, counted before writing code.**
1. *Zero imaginary parts.* At order 0 the kernel weight w = φ has no imaginary part (the ΔC term is absent), but both
   kernels multiplied it as a complex number: 2 of the 4 FFMA of every complex MAC multiplied an exact zero.
2. *Zero-weight points.* Points per (sample, tile) that a warp multiplies, W = 8, TT = 4, sample centres uniform
   (box simulation, scratch boxsim.py; a sample's support box on a tile is ≤ 4 cells per axis):

   | loop | points of 64 |
   |---|---|
   | exact support box | 25 |
   | warps of 2 × 2 × 2 points (skip per warp) | 36 |
   | warps of 4 × 2 × 2 points | 44 |
   | warps of 4 × 4 × 2 points (the old adjoint's z-slab skip) | 53 |
   | dense (the old forward) | 64 |
   | forward: union of the boxes of a warp's 16 samples, lists box-sorted within a key, 4 key runs | 57 |
   | same, 4 samples per warp | 40 |

**What was built: kernel variants, `TileCfg::kv` (segnufft_knot.cu `KV`).** Variant 0 is the original pair, unchanged.
- **Variant 1, forward:** the same sample-lane layout and dense unrolled p loop; tile values read as float4 pairs
  (even L), and at order 0 real weights (2 FFMA per complex MAC, the same sums bit for bit).
- **Variant 1, adjoint:** each warp covers a spatial sub-box of the tile instead of 32 consecutive points: 2 × 2 × 2
  points × 4 coils when the coil group is a multiple of 4, else 4 × 2 × 2 × 2 or 4 × 4 × 2 × 1. A ballot over the
  chunk's samples gives each warp the samples whose support box meets its sub-box. The coil values are folded into the
  coefficients once per (sample, coil, segment): z = y a_l β, so the inner MAC is conj(w) z (order 0: real w, 2 FFMA).
- **Variant 1, staging:** the knot weight β and (adjoint) z are computed in the one pass over (sample, segment) that
  interpolates a_l, instead of two extra passes; at order 0 the weight rows are stored as float (half the shared
  memory of the largest staged array).
- **Variant 2 (opt-in, `HASTY_SEG_KV=2` + `HASTY_SEG_BOXSORT=1`):** adjoint as 1; forward loops only over the union of
  its warp's support boxes, with lanes ordered so that a warp holds few samples, on tables sorted by support-box class
  within each key (`box_sort`). It measured slower than variant 1 (below) and is kept for other regimes.

The autotune times variants 0 and 1 with every launch shape; `HASTY_SEG_KV=0` restores the old kernels everywhere
(TileCfg default and autotune). The interval-major and multi-image kernels are unchanged.

**Correctness.** tests/seg_knot_kv.rs: variants 1 and 2 against variant 0 for coil groups 1 / 2 / 4 / 8 (all sub-box
warp shapes), forward lanes 1 / 2 / 4, sample blocking 1 / 2, orders 0 / 1 / 2, L even and odd, 1-D cubic and 2-D
linear × cubic knots, split tile lists. Forward ≤ 5e-6, adjoint ≤ 4e-5 relative (z = y a association; the knot- vs
interval-major bound of seg_knot.rs), normalised adjointness ≤ 1.3e-7. All gate tests pass (list at the end).

**Results, 2 mm 2-D lin6 × cubic10** (`splinerecon --rknots 6 --rdeg 1 --time-only --encodes 0 --phys b0`, autotuned,
one forward + adjoint of encode 0, 30.3 M samples, 2 coil batches of 4):

| state | forward | adjoint | pass | tile kernels at knot 0, one batch (fwd / adj) |
|---|---|---|---|---|
| variant 0 (§16.2) | 9.73 s | 9.24 s | **18.97 s** | 42.1 / 43.7 ms |
| variant 1, first form (float4 + real weights; sub-box warps + z) | 7.53 s | 7.96 s | 15.49 s | 31.0 / 39.5 ms |
| union-box forward on box-sorted lists (now variant 2) | 9.61 s | 7.88 s | 17.48 s | 41.8 / 39.1 ms |
| **variant 1, final (+ fused staging pass, float weight rows)** | **7.23 s** | **6.44 s** | **13.66 s** (1.39×) | 29.5 / 30.6 ms |

The autotune picks variant 1 with the same shape as before (cpba 4, cpbf 4, ppf 2, chunk 32, no register blocking).

What the measurements say:
- The real-weight MAC is the largest single gain in the forward: the dense loop with half the FFMA beats skipping zero
  points. The union-box loop (variant 2) is *slower* although it multiplies ≈ 40 % fewer points (simulated): its loop bounds are
  per warp, so the p loop is no longer unrolled with its tile-value loads pipelined, and the lane order that keeps
  few samples per warp reads 8 distinct tile-value addresses per load instead of 2 (bank conflicts). The box sort
  costs ≈ 10 s of host time per 30 M samples. Both are opt-in now.
- The adjoint's sub-box warps cut its MACs ≈ 3× (53 → 36 points, 4 → 2 FFMA per complex MAC), but the adjoint went
  only 9.24 → 7.96 s until the staging was fused: the staging, not the MACs, is now the cost per sample. The fused
  pass and the float weight rows (adjoint shared memory 38 → 30 KB per block, so 3 blocks fit per SM instead of 2)
  gave the next 1.5 s.
- The uniform-random dev case (`segnufft_bench` with `HASTY_BENCH_2D=6,1`, half grid, 15 M samples) is not
  representative: its samples fill every tile, so the per-tile work (window, tile-value loads, stores) dominates and
  variant 1 gains only 17 % / 13 % there (5.35 / 4.80 → 4.42 / 4.18 s). Real trajectories put the samples in ≈ ¼ of
  the tiles (the k-space ball inside the σ-oversampled grid), with long lists, so per-sample work dominates.

1MM_RESULTS

**Not done, ranked (measured basis above).**
1. *Staging overlap.* Each chunk still runs stage → barrier → weights → barrier → MACs, so a block's staging latency
   (atab loads, sincos, exp) is hidden only by the other 1–2 blocks on the SM. A double-buffered stage (stage A of
   chunk k + 1 in the same barrier interval as the MACs of chunk k; ≈ 5 KB more shared memory) would overlap it.
2. *Point-lane forward for order ≥ 1* (lanes = (tile point, coil) with the tile values in registers, sub-box warps as
   in the adjoint, a shuffle reduction per sample): with complex weights it should cut the forward MACs ≈ 1.6×; at
   order 0 the dense real-weight loop is as cheap, so this only matters for 1 mm with ΔC.
3. *Staging once for several knots* needs several slots resident: at 2 mm the 3 GiB slot budget gives one slot of 4
   coils, and 2 slots of 2 coils stage each sample exactly as often (coil batches × knots is constant). It only pays
   where memory is free for more slot channels.

**Selecting the paths.** `HASTY_SEG_KV=0` (old kernels, default shape and autotune), `=1` (new), `=2` (union forward,
use with `HASTY_SEG_BOXSORT=1`); unset: TileCfg default variant 1 on knot-major engines with TT = 4, autotune times
0 and 1. `SegNufft::tile_shape(cpba, cpbf, ppf, acb, fsb, kv)` + `set_tiles` pick a shape explicitly. Static and
interval-major engines always use variant 0 (the static image path is unchanged).


## 18. Respiratory motion in the knot operator: design rules for real data, first steps (2026-11-16)

**Constraint (user):** nothing may depend on labels, segmentations or phantom truths that a real scan does not have.
Design rules:
1. **Motion is estimated from the scan's own data.** Use breathing-binned low-resolution reconstructions with the
   **bellows** surrogate (not the sim's true state), then the existing label-free smooth-field fit
   (`hasty_recon::motion::fit_shared`, I_b(x) ≈ I_0(x − r_b S(x)); its mask is the signal support of the data).
2. **The region is the prescribed reconstruction box** (what a technologist sets on a localizer), not anatomy. A robust
   (IRLS, Cauchy-weighted) affine fit inside the box takes the dominant coherent motion.
3. **The affine part is applied exactly per sample in k-space**, M_s = I + s A₁, d_s = s t₁:
   k ← M_sᵀ k, φ0 += k·d_s, data ÷ |det M_s|. This costs no FFTs, so it is compatible with the knot operator. Coil
   maps and field map stay in the reference frame, an approximation (research/model_robustness.md).
   - `hasty_recon::motion::affine_samples`. tests/affine_kspace.rs: a whole-voxel translation and a 90° rotation
     equal the transformed image to 3e-7 / 4.5e-7.
4. **The non-affine remainder goes to a few respiratory knots.**
5. **Acceptance only by held-out k-space validation with shrinkage α ∈ {0, ½, 1}** (model_robustness.md). α = 0 is
   the plain knot model, so the result is never worse on held-out data than not modelling the motion.
6. **The phantom's true motion field is used only as an explicitly labelled oracle arm or check**, never inside the
   method.

**Estimation mode** `splinerecon --estimate-motion NR:SCALE [--roi x0:x1,y0:y1,z0:z1]` (encode 0, NR bellows bins,
plain-SENSE preconditioned static images on a SCALE grid, the shared field, the robust affine fit; writes
`<out>_affine.json`).

First run on the 2 mm scan at 4 mm, 8 bins (≈ 30 s):
- the shared field explains 32 % of the bin-to-bin differences (relative residual 0.68);
- **min Jacobian −0.85 / −1.08: the fitted field folds**, so its regularisation is too weak for these images;
- affine residual ≈ 1.1 voxel rms (4 mm);
- translation at the largest bin, central box: 4.5 mm, against ≈ 10 mm of true heart motion.
  - The box includes chest wall.
  - Phantom and recon axes may differ (rot_presc), so this is a rough, not an exact, comparison.

So the estimate is not good enough yet. Next steps, in order:
1. the real prescription box (map the 1 mm box prescription onto the grid through the scan geometry, as the scanner
   does);
2. a field fit with an invertibility guard (stronger smoothness, a Jacobian floor) and finer bin images (2 mm, or
   more iterations);
3. the knot reconstruction with affine-corrected samples, α ∈ {0, ½, 1}, scored by held-out k-space and,
   separately, by frame_fit;
4. bellows vs true-state surrogate as a robustness arm (the bellows-to-motion relation has hysteresis on real
   scans).

Item 3 needs the converged, preconditioned, regularised knot solver (§16.2).
