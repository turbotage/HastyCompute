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

**Results, 1 mm (sparse regime), same session** (`segnufft_bench 198 189 228 …`, 22 coils in batches of 4, L = 11,
order 2, W = 10, autotuned; `HASTY_SEG_KV=0` for the old kernels):

| case | variant 0 | variant 1 (autotune's choice) |
|---|---|---|
| knot cubic 10, 3.34 M samples (forward + adjoint) | 5.16 + 5.93 = **11.09 s** | 5.13 + 5.06 = **10.19 s** (0.92×) |
| tile kernels at knot 0, one batch (fwd / adj) | 53.6 / 65.7 ms | 53.0 / 51.6 ms |
| static image, 167 k samples | 287 + 261 ms (interval-major path, variant 0 only) | unchanged: 287 + 261 ms |

- At order 2 the weights are complex, so the forward gains nothing from the real-weight MAC; its float4 loads are
  neutral (5.16 → 5.13 s). The adjoint gains 15 % from the sub-box warps and the fused staging.
- The 1-D cubic case has 4 slots per sample, not 8, and 0.08 samples per fine cell per knot. So per-sample work
  weighs less than at 2 mm, and the gain is smaller.
- The static path is untouched, as intended.
- **2-D lin6 × cubic10 at 1 mm** (synthetic: `HASTY_BENCH_2D=6,1`, 60 knot images × 445 k = 26.7 M uniform-random
  samples, 8 coils, order 2, fixed shape cpba 4 / cpbf 4 / ppf 2, no autotune). The 2 GiB slot budget gives a coil
  batch of 1, so every sample is staged 8 × 8 times per direction.

  | | forward | adjoint | pass |
  |---|---|---|---|
  | variant 0 | 80.2 s | 76.5 s | 156.7 s |
  | variant 1 | 79.3 s | 62.4 s | 141.7 s (0.90×) |

  A full 1 mm encode in 2-D is far from the 1-D unit (10.2 s for 3.34 M samples × 22 coils): ≈ 4.8 × per (sample, coil), from 8 vs 4 slots
  and a coil batch of 1 vs 4. At 1 mm the 2-D basis needs the coil batch (memory) and the per-sample staging more
  than MAC work. This is the case for item 1 below.

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

## 19. Knot-space solver: preconditioner and held-out validation (2026-11-17, code; test pending)

§16.2 showed the basis comparison is solver-limited: unpreconditioned CGLS from zero stalls at 7–17 % of the initial
gradient. `hasty_recon::knotsolve` adds two pieces.
- **Kronecker preconditioner** P = G⁻¹ ⊗ C⁻¹.
  - The normal operator on a knot stack is AᴴA = Σ_j w_κ(j) w_κ′(j) a_jᴴ a_j. When the k-space pattern near any
    time is statistically the same (interleaved views), AᴴA ≈ G ⊗ A_sᴴA_s, with G_κκ′ = mean_j w_κ(j) w_κ′(j)
    (K × K, the basis Gram under the actual sample times) and A_s the static operator.
  - C⁻¹ is the existing circulant metric, applied per knot image. G⁻¹ (host inverse, small ridge) mixes knots per
    voxel (`knot_mix` kernel).
  - This equalises sparsely sampled knots and the ill-conditioned high-degree bases (§13: cond 18.5 cubic, 113
    quintic).
  - `cg::Precond` trait + `pcgls_dyn`; the existing `pcgls*` are unchanged, and the circulant metric implements the
    trait.
- **Held-out k-space** (`heldout_mask`): a deterministic 1/n of the readouts (hash of the readout index, so every
  breathing / cardiac / encode stratum loses the same share) stays out of the fit.
  - Their residual is the ground-truth-free prediction error, used for early stopping, comparing bases, and accepting
    model components (motion α, B0, …; research/model_robustness.md E1).
  - On the phantom, E1 is checked first: does the held-out ranking agree with the frame_fit ranking?

### 16.3 Arms v2: preconditioned knot CG with held-out k-space (2026-11-17)

(Held-out column corrected 2026-11-17 from the run logs, `best at iteration 40`: lin6 0.0583 / 0.0573, lin4 0.0492 / 0.0486, bins8 0.0660 / 0.0659 per encode; the held-out ranking agrees with the truth ranking, as §19.1 states.)

**Setup.** `splinerecon`: warm start, then G⁻¹ ⊗ C⁻¹ preconditioned CGLS (§19), 40 iterations; 1/20 of the readouts
held out; encodes 0–1; scored by the encode 1 − 0 velocity-phase slope / corr against each breathing state's own
truth (r = 0.07 / 0.181 / 0.493); scripts in research/experiments/spline2d/arms_v2.sh, score_pair.py.

Scores at iteration 40 (range over the three states):

| arm (resp × cardiac 10 cubic) | depth ≥ 5 | dynamics depth ≥ 2 | depth 0–2 (edges) | held-out residual |
|---|---|---|---|---|
| 8 bins | 0.16–0.18 / 0.09–0.11 | 0.12–0.13 / 0.05–0.06 | 0.10–0.13 / 0.06–0.07 | 0.066 |
| 6 linear knots | 0.21–0.28 / 0.14–0.20 | 0.16–0.20 / 0.09–0.11 | 0.16–0.19 / 0.11–0.14 | 0.058 |
| 4 linear knots | **0.30–0.34 / 0.22–0.26** | **0.21–0.25 / 0.11–0.15** | **0.20–0.22 / 0.15–0.16** | **0.049** |

- **Decision (pre-registered §16):** linear respiratory knots beat bins at every state and in every band, edges
  included. The knot driver path gets respiratory splines.
- **4 knots beat 6 on the truth metrics.** At this undersampling and this iteration count, data per knot wins.
- **Model-robustness E1 fails in part.** Held-out k-space ranks bins last, as the truth does, but prefers 6 knots
  over 4, while the truth prefers 4.
  - Held-out error alone is therefore not yet a safe selector between nearby bases.
  - [CHECK] Hypothesis: the held-out residual rewards fitting magnitude / motion detail, while velocity phase rewards
    SNR per knot. Test it with converged runs (still improving at iteration 40) and with a regulariser.

### 19.1 Arms v2 result (2026-11-17)

Setup:
- preconditioned knot CG (G⁻¹ ⊗ C⁻¹), warm start, 40 iterations;
- held-out 1/20 of the readouts;
- encodes 0–1, 2 mm, B0 physics, true breathing coordinate;
- respiratory basis × 10 cubic cardiac knots.

Score: encode 1 − 0 velocity-phase slope / corr against each state's own truth, at the best held-out iterate (iteration 40 in every arm: still improving).

| respiratory basis | depth ≥ 5, r 0.07 / 0.181 / 0.493 | edge band depth 0–2, r 0.493 | dynamics, r 0.493 | held-out residual |
|---|---|---|---|---|
| 8 bins | 0.161/0.094, 0.179/0.106, 0.159/0.094 | 0.102/0.057 | 0.115/0.050 | 0.066 |
| 6 linear knots | 0.211/0.144, 0.273/0.197, 0.276/0.199 | 0.170/0.119 | 0.186/0.102 | 0.058 |
| 4 linear knots | 0.299/0.216, 0.282/0.198, 0.344/0.260 | 0.217/0.164 | 0.248/0.146 | 0.049 |

Reading:
- **Pre-registered answer: yes.** Linear respiratory knots beat 8 bins at every state, in the deep lumen, in the edge band and in the dynamics. Fewer knots win (4 > 6 > 8 bins): more data per unknown at R ≈ 50+.
- At 2 mm the edge band shows no double-edge penalty for 4 knots.
- **E1 passes here.** The held-out residual ranks the arms exactly as the truth scores do: ground-truth-free basis selection works on this case.
- All arms are still far from slope 1. The solver is unregularised and not converged, so the next step is the classical priors (spatial wavelet, temporal LLR) and running to the held-out minimum.
- Timing per pass (new kernels): lin6 13.7 s vs 8 × 20 per-frame 13.8 s (§17). The v2 runs had arm-dependent coil batches (memory plans), so their s/iteration is not a timing comparison.


## 20. FFT pruning, KB-Horner kernel, grid-size selection (2026-11-17)

Stencil-doc proposals 4, 5 and 7 (research/stencil_kernel_optimality.md §6), measured on the RTX 4090 with
`segnufft_bench` (no other compute process during the timed runs; nvidia-smi checked before each). Unless stated,
1 mm = 198 × 189 × 228, 22 coils in batches of 4, L = 11, order 2, W = 10, σ = 1.25, autotuned shapes.

**Answer.**
1. **k-support pruning of the grid FFTs: on by default, exact, 1.11× on a 1 mm static image with a ball / kooshball
   trajectory** (535 → 481 ms forward + adjoint), 1.06× with stack of spirals, 1.05× on the 1 mm knot unit and on
   the 2 mm dense 2-D unit (ball), 1.00× for samples filling the cube.
2. **KB-Horner window: implemented, opt-in (`HASTY_SEG_KERNEL=kb`), not the default.** On the operator (rms over
   samples) it is as accurate as ES or better (0.63× at W 8), but its *max-norm* derivative errors (e_2, e_4 at the
   band edge) exceed ES's at every W tested, so the "at least as accurate in value and derivatives" rule fails.
   Speed: equal in all three regimes (1 mm static, 1 mm knot cubic 10, 2 mm dense 2-D).
3. **Grid size: the smallest 2·3·5·7-smooth size is already (within 2 %) the fastest.** FFT time is ∝ volume for
   this generated FFT; powers of two are not faster. The measured choice (`HASTY_NUFFT_GRID=auto`) finds ≤ 1.2 % (2 mm
   dense) and nothing at 1 mm, so it stays opt-in. With a chosen grid larger than the default, β (ES and KB) is set
   for σ_eff = min_a nf_a / N_a, which makes the larger grid strictly more accurate (W 8: 3.4e-5 → 3.2e-6 at
   σ_eff 1.43).

### 20.1 k-support pruning of the last forward / first inverse passes

**Geometry, corrected.** σ refines the k grid, it does not extend it: samples with |k_a Δ_a| ≤ π span ±nf_a/2 cells,
i.e. the whole fine grid. The stencil doc's estimate (z-pass columns 55 %, y-pass 84 %) assumed a k extent of ±N/2 in
a σN grid and is wrong; a ball of radius ≈ π (+ W/2 halo) keeps 84 % of the (kx, ky) columns and ≈ 59 % of the cells
(ball / cube = 52 % + halo + 4³-tile rounding). A uniform cube keeps everything.

**What is pruned** (`hasty_fft::KSupport`, `PrunedChanFft3::new_ksupport`). The support is the union of the
nonempty TT³ tiles of every sample set an engine made or used (tile lists already exist; a set that extends it
re-plans the transforms before its next use, `SegNufft::kcover`). On top of the block pruning (zero image planes in,
image block out):
- forward y pass: lines (channel, kx) of the support's kx range, writing only the ky range of the support's columns
  at that kx; forward z pass: per ky row, only the (channel, kx) lines of the row's columns (runtime per-row line
  ranges, `AxisPlan::set_arange`), writing only that column's kz range (runtime per-line entry ranges,
  `AxisPlan::set_lrange`: the chord of a ball);
- inverse z and y passes: the same lines, reading only the same ranges (the rest of a scatter grid is zero).
The forward spectrum is exact on every cell the gather reads; the inverse equals the full one because everything it
skips is zero. Kernel cost: the per-line ranges are a predicate on each load / store (one table read per line).

**Exactness.** FFT level (`tests/chanfft.rs::ksupport_matches_full`, ball with per-column chords, cube, a 2 % support,
full): forward on the support **bit-for-bit** (0.0) against the full transform, inverse 1.1–1.7e-7 (f32 rounding,
the same as block pruning alone). Engine level (`tests/seg_pruned.rs::kpruned_grid_fft_matches_block_pruned`: stacks
of spirals and kooshballs, support cells 0.46–1.00, orders 0 / 2 / 3, 4 frames, the multi-image path, support growth):
k-pruned vs block-pruned within the run-to-run spread of the block-pruned path itself (f32 atomics: forward 5–14e-7,
adjoint 1.5–7e-5).

**FFT micro-benchmark** (288 × 240 × 250 × 16 channels, image 228 × 189 × 198, forward / inverse):

| support | z-pass columns | cells | forward | inverse |
|---|---|---|---|---|
| block only (before) | 1.00 | 1.00 | 10.80 ms | 10.61 ms |
| ball, columns only (first version, no per-line ranges) | 0.84 | — | operator 1.04× (535 → 514 ms, static 1 mm) | |
| **ball, columns + per-line ranges** | 0.81 | 0.55 | 8.85 ms (1.22×) | 8.84 ms (1.20×) |

**Operator, 1 mm static image, 167 k samples** (`HASTY_SEG_KPRUNE=0` vs default; `HASTY_BENCH_TRAJ`):

| trajectory | support cells | off: fwd + adj | on: fwd + adj | gain |
|---|---|---|---|---|
| cube (uniform in [−3.1, 3.1]³, the old bench) | 1.00 | 287.1 + 260.6 ms | 286.9 + 261.4 ms | 1.00× |
| ball (uniform in the ball) | 0.59 | 271.8 + 263.7 ms | 242.7 + 237.8 ms | **1.11×** |
| koosh (3-D radial, density ∝ 1/k²) | 0.58 | 273.4 + 323.8 ms | 242.9 + 296.9 ms | **1.11×** |
| sos (stack of spirals: disk × kz) | 0.85 | 279.4 + 261.8 ms | 262.7 + 246.6 ms | 1.06× |

**Other regimes, ball trajectory** (research/experiments/opbench/regimes_2026-11-17.sh: knot = `HASTY_BENCH_FRAMES=10,3
HASTY_BENCH_KNOT=1 … 334000 22`; dense = 2 mm 254 × 144 × 114, `HASTY_BENCH_2D=6,1 HASTY_BENCH_FRAMES=10,3`, 15 M
samples, 8 coils, L 10, order 0, W 8):

| unit | support cells | off | on | gain |
|---|---|---|---|---|
| 1 mm knot cubic 10 (3.34 M samples) | 0.61 | 4.95 + 5.04 s | 4.67 + 4.79 s | 1.055× |
| 2 mm dense 2-D lin6 × cubic10 | 0.63 | 5.11 + 5.05 s | 4.86 + 4.82 s | 1.05× |

The FFT is ≈ 60 % of a static image and ≈ 10–20 % of the knot / dense units, so the gains follow the FFT share. The
support is computed host-side from the tile lists in `points()` (≪ the tile-list build itself); a re-plan reuses the
cached kernels. Switches: `HASTY_SEG_KPRUNE=0` (env) or `SegNufft::set_kprune(false)`; `HASTY_SEG_KPRUNE_LOG=1` prints
each support. A per-set support (one plan per set) would not gain more: every realistic set with the W = 10 halo
covers the whole ball.

### 20.2 Kaiser–Bessel window by Horner polynomials

**Form** (`KbPoly`, kernel.rs). φ(z) = I₀(β√(1 − z²)) / I₀(β), Beatty β = π√((W/σ)²(σ − ½)² − 0.8). φ and its
derivatives up to order 4 are entire and even / odd, so each is z^(n mod 2) · P_n(2z² − 1): one Chebyshev fit per
order, converted to monomials, emitted as literal f32 Horner chains (`kb_eval`, no tables, no clamp). The tile
kernels' `es3` / `es5` call it under `KER_KB`; the deapodisation factors are KB's. Fit tolerance (relative to
max|φ⁽ⁿ⁾|) `HASTY_SEG_KB_TOL`, default 1e-8: degrees 13 / 13 / 14 / 14 / 15 at W 10 (12–13 at 1e-7, 11–13 at 1e-6);
f32 evaluation error ≤ 2.3e-7 (φ) … 8.5e-7 (φ⁽⁴⁾) of max.

**Accuracy, window level** (kernel.rs `kb_poly_interpolation_errors`; 1-D e_n of stencil §2.1, 48 offsets × 121
frequencies in |ξ| ≤ π/σ, σ 1.25; max / rms):

| W | window | e0 | e1 | e2 | e3 | e4 |
|---|---|---|---|---|---|---|
| 6 | ES (clamped, production) | 3.3e-3 / 5.1e-4 | 1.2e-2 / 1.0e-3 | 1.9e-2 / 2.9e-3 | 7.8e-2 / 8.7e-3 | 1.7e-1 / 2.4e-2 |
| 6 | KB f32 poly | 1.9e-3 / 2.7e-4 | 1.5e-2 / 8.7e-4 | 2.6e-2 / 3.5e-3 | 1.2e-1 / 1.1e-2 | 1.1e-1 / 2.1e-2 |
| 8 | ES | 1.2e-4 / 2.5e-5 | 6.2e-4 / 6.5e-5 | 3.4e-3 / 3.8e-4 | 1.0e-2 / 9.4e-4 | 3.2e-2 / 3.8e-3 |
| 8 | KB | 1.4e-4 / 1.6e-5 | 5.9e-4 / 4.8e-5 | 4.7e-3 / 3.2e-4 | 1.2e-2 / 1.1e-3 | 3.9e-2 / 3.3e-3 |
| 10 | ES | 1.3e-5 / 2.4e-6 | 8.0e-5 / 6.7e-6 | 4.6e-4 / 3.9e-5 | 1.3e-3 / 1.3e-4 | 6.8e-3 / 6.1e-4 |
| 10 | KB (tol 1e-8) | 1.2e-5 / 1.4e-6 | 3.3e-5 / 3.1e-6 | 5.6e-4 / 2.4e-5 | 1.0e-3 / 1.0e-4 | 7.5e-3 / 4.0e-4 |
| 10 | KB (tol 1e-6) | 1.6e-5 / 2.4e-6 | 3.4e-5 / 3.6e-6 | 5.6e-4 / 2.4e-5 | 1.0e-3 / 1.0e-4 | 7.5e-3 / 4.0e-4 |

KB wins the rms of e0, e1, e2, e4 (1.2–1.8×) but loses the max of e2 (1.2–1.4×) and e3 / e4 at W 6–8; the ES clamp
does not cost ES its max-norm lead here. At W 10 the f32 polynomial at tol 1e-6 brings e0 to ES's level (the exact KB
has 1.0e-6 rms); tol 1e-8 keeps the f32 polynomial within 1.4× of the exact KB on e0 and equal on e1–e4.

**Accuracy, operator level** (tests/seg_kernel.rs: 20 × 18 × 14, 3000 samples, 2 coils, against f64 direct sums;
forward / adjoint relative error):

| case | W 6 ES → KB | W 8 ES → KB | W 10 ES → KB |
|---|---|---|---|
| plain (order 0) | 3.19e-4 → 3.03e-4 | 3.47e-5 → 2.19e-5 | 9.1e-6 → 9.1e-6 (f32 floor) |
| order 1, χ 0.05 rad | 3.44e-4 → 3.18e-4 | 1.07e-4 → 1.04e-4 | 1.02e-4 → 1.02e-4 (Taylor floor) |
| order 2, χ 0.05 rad | 3.30e-4 → 3.11e-4 | 3.50e-5 → 2.21e-5 | 1.14e-5 → 1.20e-5 (f32 floor) |
| order 2, χ 0.4 rad | 7.40e-4 → 7.46e-4 | 5.85e-4 → 5.85e-4 | 5.85e-4 → 5.85e-4 (truncation) |

(adjoints within 10 % of these; tol 1e-6 gives the same table.) On the operator KB is never worse beyond the f32
floor; the max-norm derivative deficit does not show in rms-over-samples errors.

**Speed** (cube trajectory, same session, k-pruning on):

| regime | ES | KB tol 1e-8 | KB tol 1e-6 |
|---|---|---|---|
| 1 mm static, 167 k | 286.7 + 261.7 ms | 287.5 + 261.4 ms | — |
| 1 mm knot cubic 10, 3.34 M | 5.13 + 5.08 s | 5.13 + 5.02 s | 5.13 + 5.03 s |
| 2 mm dense 2-D, W 8, order 0, 15 M | 5.40 + 5.23 s | 5.39 + 5.21 s | — |

A first KB knot run measured 5.91 + 5.74 s; it was not reproducible (rerun 5.13 + 5.02 s; another process had used
the GPU during that batch), so the five Horner chains per (sample, tile cell, axis) of the order-2 staging cost no
more than ES's exp / rsqrt there (tol 1e-7: 5.13 + 5.03 s; the fit tolerance does not change the time).

**Decision.** ES stays the default. KB is selectable (`HASTY_SEG_KERNEL=kb`, or `hasty_nufft::set_thread_seg_kernel`
per thread) for experiments where rms matters more than the band-edge max, e.g. W 8 at 2 mm (operator 0.63×). The
proposal's "half a tap" does not hold in max norm on our grid of frequencies.

### 20.3 Fine-grid size selection

**FFT time per size** (`gridfft_bench nx ny nz nch 1.25 W 1.3 8`: block-pruned forward + inverse, one axis varied,
the others at the default; time relative to the default grid):

1 mm, 198 × 189 × 228, 44 channels (default 250 × 240 × 288, 59.4 ms):

| axis | sizes (σ) → relative time |
|---|---|
| x (198) | 250 (1.263) 1.000, 252 1.006, 256 1.035, 270 1.066, 280 1.108, 288 1.150, 294 1.178, 300 1.174 |
| y (189) | 240 (1.270) 1.000, 250 1.021, 252 1.027, 256 1.050, 270 1.072, 280 1.100, 288 1.118, 294 1.154 |
| z (228) | 288 (1.263) 1.000, 294 1.002, 300 1.010, 320 1.030, 324 1.044, 336 1.039, 350 1.053, 360 1.067 |

2 mm box, 99 × 95 × 114, 44 channels (default 126 × 120 × 144, 7.48 ms): x 128 1.025, 140 1.116, 144 1.133, 160 1.228;
y 126 1.065, 128 1.071, 140 1.095; z 150 1.005, 160 1.035, 162 1.003, 168 1.036, 180 1.060.

2 mm dense, 254 × 144 × 114, 80 channels, W 8 (default 320 × 180 × 144, 54.4 ms): x 324 **0.980**, 336 1.025, 350
1.046, 384 1.180; y 192 1.005, 200 1.024, 216 1.114; z 150 0.999, 160 1.028, 162 1.004, 180 1.052. Chosen
324 × 180 × 150: 53.75 ms (0.988×).

Reading: the generated mixed-radix FFT runs at bandwidth for every 7-smooth length, so time ∝ volume and the
smallest admissible size wins; powers of two (256, 128, 384) are not faster. The only gains are a few lengths whose
factorisation suits the line shape (320 → 324 = 4 · 81). The search is therefore opt-in: `HASTY_NUFFT_GRID=auto`
(per axis ≤ 4 candidates up to 1.15 × the default, separable, confirmed by a direct timing, cached per process and
image; 16 channels so that all engines of a process share one grid), `=n0,n1,n2` explicit, default `fixed`.

**β for the effective σ.** With a selected grid other than the default, the engine plans with σ_eff = min_a nf_a / N_a
(ES β = 0.97π(1 − 1/2σ_eff)W, KB Beatty β(σ_eff), deapodisation on the actual grid). Accuracy
(tests/seg_kernel.rs `larger_grid_with_effective_sigma_is_not_less_accurate`, 20 × 18 × 14, forward relative error):

| W | window | default 28 × 24 × 18 (β for σ 1.25) | 30 × 28 × 20 (σ_eff 1.43) | 36 × 30 × 24 (σ_eff 1.67) |
|---|---|---|---|---|
| 6 | ES / KB | 3.16e-4 / 3.00e-4 | 1.52e-4 / 7.29e-5 | 3.39e-5 / 2.13e-5 |
| 8 | ES / KB | 3.38e-5 / 2.12e-5 | 3.19e-6 / 2.50e-6 | 1.65e-6 / 1.39e-6 |

(order 2, χ 0.05 the same within 10 %). The default grid itself has σ_eff 1.26–1.29 but keeps β(1.25), so the
numerics of existing runs are unchanged. A cheap accuracy option the timings expose: on the z axis at 1 mm, 288 → 294
costs 0.2 % — but σ_eff is the minimum over axes, so raising accuracy needs all three axes (≈ +3–5 % FFT time for
σ_eff ≈ 1.29 at 1 mm).

### 20.4 Switches, tests

- `HASTY_SEG_KPRUNE` (default on), `HASTY_SEG_KPRUNE_LOG`; `SegNufft::set_kprune`, `grid_fft_ksupport`.
- `HASTY_SEG_KERNEL=es|kb`, `HASTY_SEG_KB_TOL`; `set_thread_seg_kernel`, `SegNufft::kernel_kind`.
- `HASTY_NUFFT_GRID=fixed|auto|n0,n1,n2`, `HASTY_NUFFT_GRID_LOG`; `set_thread_grid_rule`, `choose_grid`,
  `gridfft_bench`.
- `segnufft_bench`: `HASTY_BENCH_TRAJ=cube|ball|koosh|sos`; prints grid, window and support fractions.
- New tests: hasty-fft `chanfft::ksupport_matches_full` / `ksupport_timing`; hasty-nufft `seg_pruned::kpruned_grid_fft_matches_block_pruned`,
  `seg_kernel::{kb_window_at_least_as_accurate_as_es, larger_grid_with_effective_sigma_is_not_less_accurate}`, kernel.rs
  `kb_poly_fits_and_derivatives`, `kb_poly_interpolation_errors`, `cheb_mono_roundtrip`.
- Gate: `cargo test --release -j 4 -p hasty-fft -p hasty-nufft` — 82 passed, 0 failed, 3 ignored (seg_stream passed this run).

## 21. Target config (2026-11-17)

From now on, this is the config that experiments mean by "the reconstruction". Cheaper test-bench settings are allowed, but every run reports how it differs from this (the splinerecon header line `config: differs from target: …`). The driver defaults (`const TARGET` in splinerecon) follow this table.

| knob | target | why |
|---|---|---|
| respiratory basis | 6 linear knots at the bellows-amplitude quantiles | lin4 ≈ lin6 (±0.002), both beat 8 bins (§16.3); 6 for lower model error |
| cardiac basis | 16 periodic cubic knots | model error 0.121 vs 0.234 for 20 bins (§16 correction: cubic 12 ≈ 20 bins, 16 halves it) |
| breathing coordinate | bellows (`physio.npy` col 3) | real-data rule: the phantom truth (`--resp truth`) is an oracle only |
| coils | all 32, whitened, SV-compressed to ≥ 99.9 % energy (Gram of the whitened encode-0 data, ≤ 2·10⁵ samples) | recon realism; `--coils N` (truth-lumen subset) is legacy |
| physics | B0 segments + concomitant (per-encode mean in the segments, per-sample residual φ0 / k-shift / ΔC) + GNL (maps, B0, monomials at the warped positions) | the full forward model |
| concomitant fit under GNL | degree 4, degree-3/4 part per sample to first order in the kernel (correction order 3) | the quadratic fit at the warped positions leaves a floor (recon_p2np `--conc-fit deg4`: 4.8e-3 → 1.2e-3 forward) |
| kernel | ES, σ = 1.25, W 8 (2 mm) / W 10 | NUFFT error ≪ physics error |
| time segments | the count that reaches ≤ 1e-4 signal-weighted error after fitting the interpolants to the full ratemap (complex rate: off-resonance + T2 decay + fat spectrum), e.g. rule-based `--nseg auto`; not a fixed 10 | 2026-11-18: a first estimate (research/structured_normal_operator.md Q2: rank of the space–time matrix over object voxels, *not* signal-weighted) put the count at about 15–16 for 1e-4 without the decay and about 4 at 1e-2 with it. Fixed L = 10 may therefore be too few or plenty: [CHECK] with the signal-weighted fitted interpolants. Finding the interpolators efficiently (and rebuilding them during the recon) is the open problem |

The knot count is 6 × 16 = 96 knots with 5 × 16 = 80 interval keys, under the 2-D tables' limit of 128. The knot preconditioner allows K ≤ 128.

**Memory at 2 mm, full scan, 24 GB card (estimate).** Plain knot-preconditioned CGLS with everything on the device needs about 26 GiB, against 21 GiB free:

| item | size |
|---|---|
| 3 knot stacks (96 × 4.17 M voxels × 8 B) | ≈ 9 GiB |
| data r and q (≈ 16 virtual coils × 28.8 M samples, 2 vectors) | ≈ 6.9 GiB |
| sample tables | ≈ 4.5 GiB |
| knot preconditioner | ≈ 3.5 GiB |
| maps, workspace | ≈ 2.3 GiB |

The solver arms add 1–2 stacks plus about 6 GiB of device priors. The splinerecon ledger refuses such a plan rather than running out of memory. To run the target config:
- With `--keep 2` it fits.
- At full size it needs the streaming design of recon_p2np: host or compressed sample tables (KStore / staged SegPoints), the data staged per coil batch, and the image stacks resident.
- This is one more reason to bring the knot basis into recon_p2np (next step, together with the magnitude/velocity/phase unknowns and the per-axis static lock).

Legacy scripts pin their old config explicitly so that earlier results stay reproducible: convergence_p12/run_level.sh and spline2d/{run_arms, arms_v2, time_bases}.sh pass `--cknots 10 --resp truth --coils 8 --no-gnl --conc-fit quad`.

## 22. 1 mm scale: timing and memory (2026-11-18)

Goal: run the segmented NUFFT on 1 mm full-torso grids, measure what it costs, and use that to plan the box + pyramid
dev runs and the 8-GPU final recon. Setup: RTX 4090 (450 W limit), W 10, σ 1.25, ES window, ball trajectory
(`HASTY_BENCH_TRAJ=ball`), k-support pruning on, default tiles (`HASTY_BENCH_TUNE=0`), `segnufft_bench`. Logs are in the
session scratchpad (`b_*.txt`).

### 22.1 The 1 mm launch failure (fixed)

`segnufft_bench 640 384 384 …` (fine grid 800 × 480 × 480) failed in `op.forward` with
`Fft(Driver(CUDA_ERROR_INVALID_VALUE))`. Two things combined:
- `PrunedChanFft3::new` refused a block with more than 65535 x lines (384 · 384 = 147 456). `GridFft::new` then
  silently fell back to the unpruned `FftPlan::new_partial`.
- That fallback's x pass is strided with 480 · 480 = 230 400 line groups. `AxisPlan` put the line-group count straight
  into grid y, whose limit is 65535, so the launch failed.

Indexing inside the kernels was already 64-bit (`long` line and element offsets, `size_t` in pad / crop / tile kernels).
Only the launch shape was 32-bit and limited.

**Fix (hasty-fft):**
- `plan::launch_grid` folds the line groups into grid (y, z): b = z · gridDim.y + y, and the padding blocks return
  (block-uniform, before any `__syncthreads`). The same change is in the generated strided kernels (plain, a-ranged and
  Bluestein).
- Grid x is checked against 2³¹ − 1.
- The 65535 refusal in `PrunedChanFft3` is removed.

The pruned transform now builds at 1 mm (`pruned FFT true` in the bench line).

**Also (hasty-nufft):** `SegNufft` released the plain plan's one-channel scratch grid only at drop. It now calls
`NufftPlan::release_grid` right after construction. That grid is 1.47 GB at 1 mm and was never used. Type 1 / type 2 on a
released plan return an error.

**Tests** (`hasty-fft/tests/large_grid.rs`):
- `launch_grid_folds_line_groups`: CPU unit test of the launch shape, including the 1 mm torso x pass (40 channels,
  480 · 480 groups) and the limits.
- `strided_pass_many_groups`: GPU, more than 65535 line groups with padding blocks, mixed radix and Bluestein, against
  rustfft. Error 6e-8 and 1.2e-7.
- `pruned_many_x_lines`: pruned 263 × 263 × 4 × 2 channels with a 260 × 260 block (67 600 x lines) and the full
  partial-axis plan, against rustfft. Error 2.4e-7.
- `torso_1mm_grid_point_source` (`--ignored`, 3 GB): the real 800 × 480 × 480 × 2 pruned transform of point sources
  against the analytic plane wave. Max error 1.5e-7.

The last three fail on the old code: launch error, or construction refused. All hasty-fft tests pass. So do the
hasty-nufft tests segnufft, seg_knot, seg_knot2d, seg_knot_hi, seg_multi, seg_stream, accuracy and seg_pruned.
`seg_pruned::kpruned_grid_fft_matches_block_pruned` failed once in six runs: its threshold is 5× the run-to-run atomic
spread. These sizes do not reach the code this change touched.

**Bench (`segnufft_bench`):**
- A memory plan before each case: the estimated bytes per component, printed and compared with free memory. Cases that
  do not fit are skipped (`HASTY_BENCH_FORCE=1` runs them anyway). Allocation errors skip the case instead of aborting.
- The measured pool growth of each stage is printed.
- Order 3 uses `points_hi` / `points2d_hi` tables (`HASTY_BENCH_HI=0`: the old plain tables). Until now order 3 in the
  bench timed the order-2 operator.
- `HASTY_BENCH_REPS` sets the number of timed calls.

### 22.2 Measured times (ms per call, all coils)

Static engine unless noted. "cb" is the coil batch (`HASTY_BENCH_GRID_GB` 6, except where 12 is noted).

| grid (fine) | coils (cb) | L | order | M | forward | adjoint |
|---|---|---|---|---|---|---|
| 2 mm 254×144×114 (320×180×144) | 30 (9) | 10 | 2 | 2 M / 8 M | 917 / 3234 | 584 / 1883 |
| | | | 3 | 2 M / 8 M | 849 / 2767 | 677 / 2250 |
| box 160³ at 1 mm (200³) | 30 (10) | 10 | 2 | 2 M / 8 M | 591 / 2056 | 543 / 1792 |
| | | | 3 | 2 M / 8 M / 16 M / 24 M | 516 / 1776 / 3783 / 5766 | 593 / 1996 / 3912 / 5860 |
| box, 2-D knots 2 lin × 4 cubic (8 images, s = 8) | 30 (10) | 10 | 2 | 2 M / 8 M total | 4253 / 14686 | 3969 / 12634 |
| | | | 3 | 2 M / 8 M total | 5238 / 18022 | 4665 / 15381 |
| torso 640×384×384 at 0.94 mm (800×480×480) | 2 (1) | 4 | 2 | 0.2 M / 8 M / 30 M | 115 / 390 / 1160 | 111 / 408 / 1242 |
| | | | 3 | 0.2 M / 8 M / 30 M | 116 / 593 / 2071 | 114 / 627 / 2139 |
| | 2 (1) | 6 | 2 | 0.2 M / 8 M | 159 / 480 | 152 / 497 |
| | | | 3 | 8 M | 687 | 688 |
| | 2 (2, 12 GiB) | 4 | 2 | 0.2 M / 8 M | 87 / 212 | 88 / 211 |
| | | | 3 | 8 M | 284 | 291 |
| | 1 (1) | 8 / 4 | 2 | 0.2 M | 98 / 57 | 92 / 56 |
| torso, knot-major cubic 4 knots (s = 4) | 1 (1) | 4 | 3 | 2 M / 8 M total | 437 / 1207 | 471 / 1323 |

- L 8 with 2 coils and L ≥ 10 with any coil count do not fit at the torso (§22.3); the memory plan skipped them.
- Repeat runs agree within 0.2 %.
- Autotune keeps the default tiles at the torso, yet the order-3 static forward runs at 443 ms instead of 592 after it.
  The adjoint is unchanged. The multi-kernel path's configuration is probably involved [CHECK]. On the box 2-D knot
  engine at order 3 (2 M), autotune (451 s) picked other tiles and changed nothing useful: forward 5225 vs 5238 ms,
  adjoint 5053 vs 4665 ms. The projections below use the untuned numbers.

### 22.3 Split: image side and sample side

Image side means FFT + pad / crop per (coil, segment, image), per direction. At the torso it comes from the 0.2 M
intercept, where the sample side is ≈ 3–7 %. At the box and at 2 mm it comes from the 2 M / 8 M linear fit. Sample side
is per 10⁶ samples per coil, per static image.

| grid | cb | image side, fwd / adj (ms per coil · segment) | sample side, fwd / adj (ms per 10⁶ · coil), order 2 | order 3 |
|---|---|---|---|---|
| 2 mm full FOV | 9 | 0.48 / 0.50 | 12.9 / 7.2 | 10.7 / 8.7 |
| box 160³, 1 mm | 10 | 0.34 / 0.42 (order 3: 0.32 / 0.42; 2-D knots: 0.41 / 0.46) | 8.1 / 6.9 | 7.0 / 7.8 (at 24 M: 7.9 / 8.0) |
| torso, nch = 4 (cb 1, L 4) | 1 | 13.5 / 12.9 | 17.5 / 19.0 | 32.8 / 34.0 |
| torso, nch = 6 (cb 1, L 6) | 1 | 12.7 / 12.0 | 20.5 / 22.1 | 33.4 / 34.0 |
| torso, nch = 8 (cb 1, L 8) | 1 | 11.7 / 11.0 | — | — |
| torso, nch = 8 (cb 2, L 4) | 2 | 10.5 / 10.5 | 8.0 / 7.9 | 12.5 / 12.9 |
| torso knot-major (cb 1, L 4) | 1 | 11.3 / 11.7 | | ×4.2 / ×4.3 static (s = 4) |

Knot multiplier on the sample side, against static at the same grid:
- box 2-D (s = 8): order 2 ×7.1 / ×6.9, order 3 ×10.1 / ×7.6;
- torso 1-D cubic (s = 4): order 3 ×4.2 / ×4.3.

So a sample costs ≈ 1.0–1.3 × s static gathers. The knot image side equals the static image side to within 1.2×
(§10 confirmed at 1 mm).

Reading:
- **Image side ∝ fine-grid cells**: 0.33–0.42 ms per (coil, segment) at 8 M cells, ≈ 11 ms at 184 M cells (23×).
  That is ≈ 1.47 GB of complex grid per channel: one pad write plus ≈ 2 pruned sweeps, near the 4090's bandwidth.
  Small channel counts lose 20 %: the x pass coalesces over the channels only, so nch = 4 → 32 B runs.
- **The sample side is dominated by per-pass cost, not per-coil cost.** A pass (one coil batch over all samples) costs
  ≈ 25–31 ms per 10⁶ samples (forward, order 3) at the torso whether it carries 1 or 2 coils. At equal M (8 M), a coil
  costs 12.5 at cb 2 instead of 30.7 at cb 1. At the box (cb 10, 100 channels) a coil costs 7–8. At 1 mm the coil batch is forced to 1, so the sample side
  per coil is ≈ 4× the box's.
- **Order 3 costs ≈ 2× order 2 per sample at cb 1** (torso) but ≈ 1× at cb 10 (box). The degree-3/4 terms are
  per-sample, per-pass work.
- **The 2 mm forward is 1.6× the box's per sample at equal cells.** The grid is anisotropic (320×180×144) and the tiles
  are the defaults. Not investigated.

### 22.4 Memory per component

| component | formula | torso 640×384×384 (nv 94.4 M, fine 184 M) | box 160³ at 1 mm (nv 4.1 M, fine 8 M) |
|---|---|---|---|
| fine grid, one channel | ge · 8 B | 1.47 GB | 64 MB |
| grid slot, knot-major (one slot) | cb · L · ge · 8 B | **14.7 GB** at cb 1, L 10 | 6.4 GB at cb 10 (measured 6.0 GiB) |
| plain plan scratch | ge · 8 B | 1.47 GB → 0 (released now) | 64 MB → 0 |
| segment diagonals | L · nv · 8 B | **7.6 GB** (L 10) | 0.33 GB (measured 0.31 GiB) |
| coil maps | nc · nv · 8 B | **22.6 GB** (30 coils) | 0.98 GB (measured 0.91 GiB) |
| one knot image | nv · 8 B | 0.76 GB | 33 MB |
| knot stack, 96 knots | K · nv · 8 B | **72.5 GB** per vector | 3.1 GB |
| sample tables, 30 M samples | ≈ 200 B per sample at build peak (measured 5.7 / 6.2 GiB at order 2 / 3), ≈ 70–110 B resident | 3–6 GB per encode | same |
| data y, 30 M × 30 coils | M · nc · 8 B | 7.2 GB per encode | same |
| forward coil-batch y | cb · M · 8 B | 0.24 GB | 2.4 GB |

- **Torso, target config, on one card:** slot 14.7 + diagonals 7.6 + maps 22.6 + 2 knot images 1.5 + tables 3.3 +
  y 7.2 ≈ 57 GB, against ≈ 22 GB free on the 4090 and 32 GB on a 5090. The CG vectors (3–4 knot stacks = 220–290 GB)
  come on top, more than 8 × 32 GB.
- **Torso, measured:** L 4 with 2 coils and 30 M samples peaks at 17.1 GiB.
- **Box:** fits one 4090 with the data streamed per encode. Slot 6.4 at cb 10 (or 2.6 at cb 4), diagonals and maps
  1.3, 3 knot stacks 9.3, table and data chunks ≈ 3: ≈ 20 GB.

### 22.5 Options for the 1 mm torso memory (operator unchanged; expected cost)

1. **Segment batching inside a coil** (Lb of the L segments per slot pass): slot Lb · 1.47 GB.
   - The image side is unchanged: same (coil, segment) FFT count, plus one extra image / map read per extra batch,
     ≈ 1 ms.
   - The sample side is × L / Lb passes. A pass costs α + β · Lb, with α ≈ 56 and β ≈ 1.9 ms per 10⁶ samples per coil
     (fwd + adj, from torso L 4 / L 6 at order 3).
   - Lb = 5 → slot 7.4 GB and 1.75× the sample side (≈ +460 s per encode pass at 30 M samples, §22.6).
   - Use it only if the slot alone does not fit.
2. **Diagonals on the fly** from the field: store ω(v) (4 B) plus the GNL-warped position (12 B), 1.5 GB (or 0.4 GB
   without warp), instead of 7.6 GB.
   - Fixed-time diagonals (unit modulus, `build_segments`): one sincos per (voxel, segment) in pad / crop, < 1 ms per
     coil pass, and it removes the 7.6 GB per coil-direction read (≈ 7 % of the image side). Net faster.
   - SVD / complex diagonals (`build_segments_svd`): D_l(v) = Σ_i g(v, τ_i) V[i, l] costs ns · (sincos + L cmul) per
     voxel per coil pass, ≈ 0.5 TFLOP at ns 64: ≈ +10 ms per coil-direction, +9 % image side [estimate]. A Chebyshev
     fit in (ω, concomitant monomials) with a few terms would land in between.
   - **Recommended:** it frees 6–7 GB at no time cost for fixed-time diagonals.
3. **Host-resident diagonals:** 0 GB on the device, but 7.6 GB over PCIe per coil pass per direction.
   - That is 0.30 s at 25 GB/s (PCIe 4) or 0.15 s (PCIe 5), against 0.11 s of image compute per coil-direction:
     1.4–2.7× slower. It hides only at cb ≥ 3, impossible at 1 mm.
   - A segment-major loop (upload P_l once for all coils) needs every coil's grid for one segment resident, which is
     the same memory problem.
   - **Reject for the torso; fine for the box** (0.33 GB anyway).
4. **Coil maps** (22.6 GB), any of:
   - stream from host per coil pass: 0.76 GB at 25 GB/s = 30 ms against 110 ms compute, hidden by double buffering,
     1.5 GB on the device;
   - coil-split across GPUs: 4 coils per GPU, 3 GB;
   - store the smooth maps at 2 mm and interpolate in pad / crop: 2.8 GB.
   - Maps as k-space kernels (§15.5) would also cut FFT channels from C · L to L. This is the only lever on the image
     side, which is half the torso time.
5. **Knot stacks:** host-resident (pinned). The knot-major operator needs only the current knot image and its adjoint
   accumulator on the device: 2 × 0.76 GB per knot launch, 60 ms at 25 GB/s against ≈ 6.8 s of compute per knot.
   The node needs ≥ 512 GB RAM for 3–4 stacks, or bf16 storage (accuracy study first).
6. **Tables and data:** the existing streamed path (kload / Stager, per coil batch).

With 2 + 4 + 5 + 6, a torso task holds slot 14.7 + diagonals ≤ 1.5 + 2 map buffers 1.5 + 2 knot images 1.5 + chunks
≈ 2: ≈ 21 GB. That fits a 5090 (32 GB), and barely the 4090. With option 1 at Lb = 5 it is ≈ 14 GB, at a cost in time.

### 22.6 Projected per-encode gradient pass (forward + adjoint), target config

The target: 6 × 16 = 96 knots (s = 2 · 4 = 8 knots per sample), 30 virtual coils, L 10, order 3.

Model: T = K · I + M · S.
- Box (cb 10): I = 259 ms per knot image (measured, 2-D knot engine, 30 coils, L 10). S = s-multiplied static sample
  cost at the dense end (24 M: 236 / 239 ms per 10⁶ for 30 coils) × knot multipliers 10.1 / 7.6 → 4.2 s per 10⁶ samples.
  The purely measured 2 M–8 M knot slope gives 3.9 s.
- Torso (cb 1): I = 300 (coil, segment) × ≈ 22.7 ms (fwd + adj) = 6.8 s per knot image. Measured at nch 8
  (static L 8: 11.7 + 11.0; knot L 4: 11.3 + 11.7), applied to nch 10. S = 30 coils × (40.9 · 9.3 + 37.0 · 8.1) ms = 20.4 s per 10⁶ samples. Here 40.9 / 37.0 are the static
  per-coil fwd / adj costs at L 10, extrapolated from L 4 / L 6. 9.3 / 8.1 are the knot multipliers for s = 8, the mean
  of the torso s = 4 rate per knot and the box 2-D rate.

| case | image side | sample side, 30 M samples | total, 30 M samples | total, 60 M (2× samples per readout) |
|---|---|---|---|---|
| (a) torso 1 mm, one 4090 (if memory allowed, §22.5) | 96 × 6.8 s = 651 s | 612 s | **≈ 1260 s (21 min)** | ≈ 1880 s (31 min) |
| (b) box 160 mm at 1 mm, one 4090 | 96 × 0.26 s = 25 s | 117–127 s | **≈ 145 s** | ≈ 270 s |
| 2 mm full FOV (caller's reference model: 27 s + 3.3 s per 10⁶) | 27 s | 99 s | 126 s | 225 s |

What was measured vs extrapolated:
- Measured: the image side per knot image at the box; the torso image side per (coil, segment) at nch ≤ 8; static and
  knot sample costs up to 24 M samples (box) and 30 M (torso, L 4).
- Extrapolated: L 10 at the torso (no L ≥ 10 case fits), 30 coils at the torso (measured 1–2, linear in coils at
  cb 1: the passes are per coil), s = 8 at the torso, 30 M on the box knot engine (slope from 2–8 M; the static
  per-sample cost rises 13 % from 8 M to 24 M, which is included).

**8 × RTX 5090, per gradient (4 encodes):** T_grad = 4 · T_pass / (8 · 1.4).

> **Correction (2026-11-18): the scan has 7 encodes, not 4.**
> - The rule is T_grad = 7 · T_pass / (8 · 1.4) = 0.625 · T_pass, so every per-gradient number in this section is 7/4 too low.
> - Corrected, 8× 5090:
>   - torso 1 mm: 790 s per gradient at 30 M samples and 1175 s at 60 M (100 iterations: about 22 h / 33 h);
>   - 160 mm box at 1 mm: 91 s per gradient at 30 M and 169 s at 60 M (100 iterations: about 2.5 h / 4.7 h).
> - Box on one 4090: about 1015 s per gradient at 30 M.
> - The pyramid table's per-gradient column (one 4090) is also ×7/4.

Assumptions: 1.4× per GPU, and ideal splitting of (encode, knot batch) or (encode, coil) tasks across GPUs. That is
≥ 120 tasks per gradient, so balance ≥ 97 % (§14). Communication is small next to the compute:
- partial y sums within an encode: 7.2 GB, ≈ 0.3 s at 50 GB/s;
- coil-split partial knot stacks: ≤ 72 GB, ≈ 1.5–3 s;
- against ≈ 450 s of compute.

No FFT is split across GPUs.

| case | per gradient | 100 iterations |
|---|---|---|
| torso 1 mm, 30 M per encode | **≈ 450 s (7.5 min)** | ≈ 12.5 h |
| torso 1 mm, 60 M per encode | ≈ 670 s (11 min) | ≈ 19 h |
| box 160 mm, 30 M / 60 M | 52 s / 97 s | 1.4 h / 2.7 h |
| box on one 4090, 30 M | 580 s (9.7 min) | 16 h |

The torso is 2–3× over the 5–6 h budget (memory: recon_compute_budget). Where the time goes:
- half is the image side, which only maps-as-kernels, fewer segments (shimming) or fewer knots reduce;
- the rest is the per-pass sample cost at cb 1, which a staged-once-per-sample-for-all-coil-batches kernel would cut
  ≈ 2.5×, judging by the cb 2 vs cb 1 measurement.

### 22.7 Dev runs: box size and pyramid levels

- **Box 160 mm at 1 mm costs about what 2 mm full FOV costs** (4.1 M vs 4.17 M voxels, same fine-grid cells). The box
  time is 83 % sample side, which does not depend on the box size. A 128 mm box saves only ≈ 9 %, so choose the box for
  the anatomy: heart plus ≥ 20 mm margin for motion, so 160 mm.
- The box needs the out-of-box signal removed from the data: a coarse full-FOV estimate, forward-modelled once per
  encode and subtracted. That costs one coarse full-FOV forward per encode (not measured).
- **Pyramid** (box 160 mm; per level the image side ÷ 8 by cells, the sample side × the fraction of samples inside the
  level's k-ball; extrapolated):

| level | grid | image side (96 knots) | sample fraction: uniform ball / radial (∝ k⁻²) | per encode pass | per gradient (4 encodes, one 4090) |
|---|---|---|---|---|---|
| 4 mm | 40³ | ≈ 1–2 s | 1/64 / 1/4 | ≈ 3–34 s | 0.2–2.3 min |
| 2 mm | 80³ | ≈ 3–5 s | 1/8 / 1/2 | ≈ 20–68 s | 1.3–4.5 min |
| 1 mm | 160³ | 25 s | 1 | ≈ 145 s | 9.7 min |

**Recommendation:**
- Box 160 mm, pyramid 4 → 2 → 1 mm, with k-ball sample subsets per level and stochastic 8-of-30 coil subsets on the
  coarse levels (× 0.27 on both sides).
- Most iterations at 2 mm; finish with 10–20 iterations at 1 mm (≈ 1.6–3.2 h on one 4090, or minutes on the cluster).
- With a centre-dense trajectory the coarse levels keep half the samples. Then coil subsets and temporal subsets
  (fewer heartbeats) are the dev levers, not resolution.
- The coarse levels may also need fewer segments: the samples inside a small k-ball come early in centre-out readouts,
  so the τ range is shorter. Not measured.
- Keep W 10 / σ 1.25 at every level (NUFFT accuracy rule).

## 23. FFT-side traffic (2026-11-19)

Goal: cut the DRAM traffic around the grid FFTs of the segmented NUFFT. On the knot path that part was ≈ 60 % of a
gradient and bandwidth-bound (fft_axis at 85–93 % of DRAM bandwidth, seg_pad 85 %, seg_crop 95 %). Every change is
exact or f32-rounding-level against the previous path and sits behind an env switch; the previous paths stay selectable.

**Setup.** RTX 4090, opbench, 2 mm thorax: image 225 × 225 × 112, fine grid 288 × 288 × 140, 10 virtual coils, L 10,
order 2, coil batch 5, so one grid slot holds 50 channels (4.6 GB).
- Knot case: `opbench --models seg-full --nseg 10 --vcoils 10 --order 2 --temporal cubic --knot --frames 20 --scan-min
  0.44 --sets 1 --reps 3 --seq flow`. Cubic knot-major engine, 20 knots, 1.33 M samples (one encode, all frames). Per
  gradient: 40 grid loads (pad + forward FFT) and 40 flushes (inverse FFT + crop).
- Static case: the same without `--temporal` / `--knot` and `--frames 7`. One frame, 4.3 M samples, 2 loads and 2
  flushes per gradient. It is tile-kernel bound (kernels 278 / 408 ms of 333 / 462 ms).

Logs are in `research/experiments/fft_traffic/` (`t1_*` task 1, `t4_*` / `t6_*` / `t7_*` tasks 2 and 3, `t8_*` the final
defaults, `tests_*` the correctness tests); nsys / ncu reports stay in the session scratchpad.

### 23.1 opbench's FFT column

The old column timed separate unpruned `FftPlan::new_partial` plans: 1251 ms per direction on the knot case, unchanged by
pruning. It now times the engine's own transforms: `SegNufft::grid_fft_run` (forward / inverse of one coil batch on the
engine's slot, the plan it uses for the current k-support). It multiplies by the transform count of one call, which the
engine counts (`fft_calls`). opbench prints what it timed, for example:

`FFT column: k-support pruned grid FFT (50 ch, grid [288, 288, 140], z-pass columns 0.849, kx / ky / kz ranges [1.000,
1.000, 1.000], cells 0.641, x pass Some((16, 36)) lines × threads, whole-grid passes, pad / crop separate kernels) —
forward 19.61 ms × 40, inverse 19.43 ms × 40 per call`

Before the changes below: 784 / 777 ms per gradient direction on the knot case (48 % of the gradient, matching nsys).
With fused pad / crop the hooked first and last passes include the pad / crop work, so the column then times the plain
passes and says so.

### 23.2 Bytes per grid load and flush (knot case, one coil batch of 50 channels, model)

The pruned transform reads and writes only the image block and the k-support. The kx / ky / kz ranges are full here, the
z-pass keeps 85 % of the columns and 64 % of the cells. At 400 B per cell (50 channels × 8 B):

| pass | forward read / write (GB) | inverse read / write (GB) |
|---|---|---|
| pad: x, 5 coil maps, 10 segment diagonals → block | 0.73 / 2.27 | — |
| x pass (225 × 112 lines, 225 ↔ 288 entries) | 2.27 / 2.90 | 2.90 / 2.27 |
| y pass (288 × 112 lines, 225 ↔ 288) | 2.90 / 3.72 | 3.72 / 2.90 |
| z pass (85 % of 288² columns, 112 ↔ kz range) | 3.15 / 2.98 | 2.98 / 3.15 |
| crop: block + coil maps + segment diagonals → x (+=) | — | 3.00 / 0.05 |
| **total** | **20.9** | **21.0** |
| fused pad / crop (first pass reads 0.73 GB of image inputs, last pass writes x by atomics) | 16.4 | 16.5 |
| two-pass L2 slabs (the x → y intermediate stays in L2: −2 × 2.90) | 15.1 | 15.2 |
| both | 10.6 | 10.7 |
| full-slot memset per adjoint knot × batch when tile lists are split (old path) | — | + 4.65, and read-modify-write instead of stores |

ncu confirms the byte counts. The fused forward x pass reads 0.726 GB and writes 2.87 GB; the fused inverse x pass reads
3.63 GB and writes 0.047 GB. The plain x pass reads 2.27 GB and writes 2.87 GB.

### 23.3 Task 1: zero only the split tiles (`HASTY_SEG_ZERO=tiles|full`, default tiles)

Old: when any tile list is split (above `max_item` = 1024 samples, which the dense k-space centre always is), the adjoint
zeroed the whole slot and every item added, unsplit tiles with a read-modify-write. That is 4.65 GB of memset plus a read
of every stored cell, per knot and coil batch.

New:
- `tile_lists` marks the first part of every split tile with `split = 2`. All kernels test `split` for truth only, so
  nothing else changes. `Interval::nz` is the prefix of the heaviest-first items that holds every marked item.
- A small kernel `seg_zero_split` zeroes the cells of those tiles (all channels).
- The scatter then runs in store mode. Unsplit tiles have a single owner and store; split tiles add atomically onto the
  zeroed cells, as before.
- Covered paths: knot-major (`adjoint_knot`, `adjoint_source`), static / interval-major (`adjoint`, `adjoint_group`,
  `adjoint_source`). In grouped calls the first member stores and later members add.
- The multi-image kernels (order 3 interval-major, fat) keep their full `mzero`.

Exactness: identical up to the order of the f32 atomics on split tiles. Test `seg_fft_traffic::split_tile_zeroing_matches_full_slot`
(static, interval-major, knot; all sets split; adjoint, group, source) gives tiles vs full 2–5e-6 against full vs full
2–5e-6.

| knot case (ms) | forward | adjoint | gradient |
|---|---|---|---|
| before (full) | 1459.0 | 1765.0 | 3222.9 |
| tiles | 1459.7 | 1500.6 | 2959.8 |

Static case: 333.3 / 462.5 / 793.7 → 333.7 / 459.8 / 791.4. The adjoint gains 264 ms on the knot case, more than the
memsets' 192 ms, because the unsplit tiles now store instead of read-modify-write.

### 23.4 Task 2: pad fused into the first forward pass, crop into the last inverse pass (`HASTY_SEG_FUSE_PADCROP=1|gather|0`, default 1)

**hasty-fft (generic).**
- `IoHook` (codegen) holds caller CUDA code fused into a strided axis pass:
  - `load`: an expression replacing `in[GIN(e)]` in the first stage;
  - `prologue`: block code before the first stage, which may stage inputs into the second shared buffer (free until
    stage 1);
  - `epilogue`: the last stage writes its outputs to shared memory and this code consumes them, with the free buffer as
    `spare`;
  - extra kernel parameters, passed with `AxisPlan::launch_ext` / `KArg`.
- `PrunedChanFft3::set_hooks` / `forward_hooked` / `inverse_hooked` build and run the hooked x passes. The z offset
  `hz0` is appended per launch, for the slabs of §23.5.

**hasty-nufft (`segnufft/fused.rs`).**
- Pad hook: channel la = c · NSEG + l of grid (x, y, z) = S_c P_l x / φ̂. The prologue stages the line's x / φ̂ row, the
  coil rows its 16-channel tile touches and the 10 segment rows (coalesced). The first stage reads them from shared memory.
- Crop epilogue: the coil and segment rows are staged into the spare buffer. Two threads per x entry each sum 8 of the
  tile's channels g · conj(S_c P_l) in a rotated order (no shared-memory bank conflicts), reduce with one group shuffle,
  and one f32 atomic add per voxel and channel tile goes into x. That is the crop's x += Σ; the coil batches still
  accumulate.
- `HASTY_SEG_FUSE_PADCROP=gather` keeps the first version: per-element global loads, 16-lane shuffles.
- Used on the plain single-image paths with the pruned transforms: knot, static, interval-major, grouped and source
  calls. Not used for water + fat, the warp, or the multi-image kernels; those keep `seg_pad` / `seg_crop`.

Exactness:
- The fused pad's grid equals the separate pad's to 1.3e-7 (FMA contraction; `debug_fused_slot`).
- The forward differs by 3.5–4e-6. The forward itself amplifies one-ulp input changes to that level: x moved by one ulp
  gives 4.2e-6.
- The adjoint matches within the run-to-run atomics spread (≤ 2e-6).
- Adjointness of the fused pair equals the separate pair's (≈ 1e-7).
- Tests: `seg_fft_traffic::fused_pad_crop_matches_separate_kernels`. It covers two x shapes (8 × 9 and 8 × 10 threads),
  a short last coil batch, k-support and block pruning, static, interval-major and knot; staged and gather variants.

ncu, x pass alone at ncu clocks (no slabs):

| x pass | time | DRAM read / write | DRAM % of peak |
|---|---|---|---|
| plain forward / inverse | 5.61 / 5.59 ms | 2.27 / 2.87 GB, 2.90 / 2.24 GB | 93 % |
| fused forward, gather / staged | 7.03 / 6.43 ms | 0.73 / 3.11, 0.73 / 2.87 GB | 56 / 57 % |
| fused inverse, gather (16-lane shuffles) / staged (two threads per entry, final) | 6.88 / 5.44 ms | 3.63 / 0.05 GB | 54 / 69 % |

The fused passes move the bytes as planned but are latency-bound. The x passes run one block per SM (73.7 KB of shared
memory for 16 lines × 288 × 2 buffers), so the staging barrier and the reductions are not overlapped by other blocks. The
fused pass still replaces pad + x pass (3.5 + 5.6 ms) or x pass + crop (5.6 + 3.3 ms).

| knot case (ms), with tiles | forward | adjoint | gradient |
|---|---|---|---|
| separate pad / crop | 1459.7 | 1500.6 | 2959.8 |
| fused, gather | 1350.8 | 1386.3 | 2737.1 |
| fused, staged rows, 16-lane shuffle crop | 1327.1 | 1395.2 | 2721.2 |
| fused, staged (final) | 1326.8 | 1344.1 | 2669.5 |

Static case: 333.7 / 459.8 / 791.4 → 328.9 / 452.2 / 778.2.

### 23.5 Task 3: two-pass FFT through L2 (`HASTY_SEG_FFT_SLAB=auto|0|<planes>`, default auto)

`PrunedChanFft3::set_slab(dz)` runs the forward x and y passes slab by slab of dz z planes of the image block: x then y
per slab, then the z pass over the whole grid. The inverse runs the z pass, then y then x per slab. The y pass then reads
the x pass's output from L2 instead of DRAM, and the dirty lines are overwritten before eviction. The same kernels run on
the same lines, launched per slab with an offset pointer and a line-group override (`AxisPlan::launch_lines`; uniform
a-ranges and group-independent line tables only), so results are bit-identical
(`chanfft::slab_passes_equal_whole_grid`, `seg_fft_traffic::fft_slabs_match_whole_grid`).

Slab size:
- `auto` (default) takes as many planes as fit `HASTY_SEG_FFT_L2FRAC` (default 0.5) of the device's L2
  (`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`). If even one plane does not fit there, it uses one plane while that fits
  `HASTY_SEG_FFT_L2MAX` (default 0.85) of the L2, else slabs are off. `0` turns slabs off; an integer sets the planes.
- One plane here (all channels) is 33.2 MB. On the 4090 (72 MB) auto picks 1 plane, the 5090 (96 MB) 1, the H100
  (50 MB) and A100 (40 MB) 1 through the 0.85 rule (not measured on those cards).
- No L2 persistence control was needed: plain caching keeps one plane.

FFT alone (opbench FFT column, ms per transform):

| | forward | inverse |
|---|---|---|
| whole-grid passes | 19.56 | 19.40 |
| slabs of 1 plane (33 MB) | 14.74 | 15.81 |
| slabs of 2 planes (66 MB) | 15.41 | 15.94 |

In the operator (nsys, 1-plane slabs, separate pad / crop): forward x + y 7.55 ms per load and inverse y + x 8.66 ms per
flush, against 12.4 ms whole-grid.

| knot case (ms), with tiles | forward | adjoint | gradient |
|---|---|---|---|
| no slabs, separate pad / crop | 1459.7 | 1500.6 | 2959.8 |
| slabs 1, separate pad / crop | 1265.4 | 1356.0 | 2620.8 |
| slabs 1, fused gather | 1227.4 | 1358.9 | 2586.5 |
| slabs 2, fused gather | 1228.0 | 1354.5 | 2583.7 |
| slabs 1, fused staged (final) | **1203.3** | **1311.5** | **2514.0** |

Static case: slabs 1 separate 325.5 / 452.7 / 775.2; slabs 1 fused staged 322.6 / 451.2 / 770.4.

### 23.6 Summary and switches

Final check with the new defaults against all three switches set back (`t8_*`, same binary):

| ms | knot forward | knot adjoint | knot gradient | static forward | static adjoint | static gradient |
|---|---|---|---|---|---|---|
| old paths (`ZERO=full FUSE_PADCROP=0 FFT_SLAB=0`) | 1459.3 | 1763.8 | 3224.3 | 333.9 | 462.3 | 795.1 |
| defaults | 1201.8 | 1310.6 | 2511.7 | 322.1 | 450.5 | 771.4 |

Knot case gradient −22 %. Static case −3 %: it is tile-kernel bound.

Regression run with the new defaults (hasty-fft and hasty-nufft GPU test binaries):
- All pass except three.
- `seg_joint_chunks` (Φ CGEMM vs kernel forward 7.4e-5 > 1e-5) fails identically with all three switches off, so it is
  not caused by these changes.
- `seg_fat_union_gpu` fails one run in a few with its adjoint identity at 1.1e-5 against a 1e-5 bound. Its fat path
  does not use these changes; the old paths reach 9.7e-6 on the same test.
- `seg_pruned::pruned_grid_fft_and_fused_residual` and `seg_warp_gpu::warp_adjoint_gpu` compared the fused plain path
  against separate-kernel paths. They now pin the separate kernels for the comparison, and both pass:
  - seg_pruned: pruned vs full forward 1.5e-5 against a 1e-5 bound, from the fused pad's rounding;
  - seg_warp_gpu: the fused plain operator's adjoint mismatch (7.5e-5) was smaller than the separate one's (4.2e-4),
    which tightened the warped bound.

| switch | values (default first) | old path |
|---|---|---|
| `HASTY_SEG_ZERO` | `tiles` / `full` | `full` (also `SegNufft::zero_tiles`) |
| `HASTY_SEG_FUSE_PADCROP` | `1` (staged) / `gather` / `0` | `0` (also `SegNufft::fuse_padcrop`) |
| `HASTY_SEG_FFT_SLAB` | `auto` / `0` / planes; `HASTY_SEG_FFT_L2FRAC` 0.5, `HASTY_SEG_FFT_L2MAX` 0.85 | `0` (also `SegNufft::set_fft_slab`) |
| `HASTY_SEG_FFT_STORE` (§23.7) | `f32` / `bfp16` … `bfp24` | `f32` (also `SegNufft::set_fft_store`) |
| `HASTY_SEG_FFT_BFP_LAYOUT` (§23.7) | `inplace` / `dense` | — (only with a BFP store) |

Notes:
- **New adjoint kernels.** `zero_tiles` assumes the scatter stores every unsplit tile (every tile has at least one item,
  empty ones a zero-count item) and adds only split items atomically. A new adjoint variant that accumulates everywhere
  must zero its slot itself, or run with `HASTY_SEG_ZERO=full`.
- The fused crop adds into x with atomics. The adjoint was already order-nondeterministic through the split tiles.

Next, ranked by expected gain:
1. **The fused x passes are latency-bound** (55–69 % of DRAM bandwidth, one block per SM). The floor for the fused
   forward x pass is ≈ 3.9 ms against ≈ 5.8 ms now. Options: prefetch the next line's staged rows (software pipelining),
   or an in-place stage order (one shared buffer: two blocks per SM).
2. **Channel sub-slabs for 1 mm grids.** A 1 mm torso plane is 480 × 480 × 50 × 8 B = 92 MB, more than any target
   card's L2. `auto` then turns slabs off. A y-pass layout with lines (channel tile) × (kx, z) would allow slabs of
   (plane, channel tile).
3. **The z pass (7.1 ms per transform) is now the largest single pass.** Fusing z into the tile kernels (gather /
   scatter over kz in shared memory) is the remaining large DRAM round trip.

### 23.7 BFP y ↔ z intermediates (`HASTY_SEG_FFT_STORE`, 2026-11-20)

The y → z and z → y intermediates can be stored as block floating point, in place in the grid slot. Only the
exponent bytes are extra: 29 MB at the 288 × 288 × 140 grid with 50 channels. The x ↔ y intermediates stay f32 in the
slabs. The default is off, and that path is bit-identical to the old one.
- Model: 21.3 → 17.95 GB per grid load + flush (bfp20 … 24 in place, 3 of 4 sectors), about 0.97 s saved per 2-D-knot
  gradient (192 pairs); keep threshold 0.58 s.
- On ph0001 every error of bfp24 is at the operator's own f32 one-ulp floor.
- Timing not measured yet.
- Details, accuracy tables and tests: research/bfp_storage.md §8.1; handoff for the timing campaign:
  research/experiments/bfp_fft/HANDOFF.md.

## 24. Step size by Lanczos (2026-11-22)

**What it is.** The step α = 1 / λ̂ of P²nP (frame mode: λ_max of the normal operator N, or of M N with the circulant
metric M; knot / velocity mode: λ_max of M H with the Gauss–Newton Hessian H) was estimated by a plain power iteration
from a fixed start (frame mode 12 iterations on set 0 and the middle set, for N *and* M N even when only one is used;
velocity mode `--lam-iters` 12 per stage, + 3 × 8 with coil sampling). No safety margin, and the power iteration
always underestimates. Now `--lam-method lanczos[:K]|power` (default `lanczos` = 8 steps; `power` = the old path,
unchanged). Code: `hasty-recon/src/lanczos.rs` (driver, tridiagonal Ritz solve, dense CPU test),
`P2np::step_norm_with` / `step_norm_sampled_with` (frame), `VelProblem::lambda_lanczos` / `lambda_lanczos_host`
(velocity), recon_p2np (switch, `--lam-warm`, `--lam-compare`), opbench (`--lam-method` in the setup projection).

**Method.** M N is self-adjoint in the M⁻¹ inner product, so Lanczos runs in that inner product without ever applying
M⁻¹: carry q_j (M⁻¹-orthonormal) and p_j = M⁻¹ q_j; per step w = N q_j, α_j = Re q_jᴴ w, s = w − α_j p_j − β_j p_{j−1},
r = M s, β_{j+1} = √(Re sᴴ r), q_{j+1} = r / β_{j+1}, p_{j+1} = s / β_{j+1} — one N and one M per step, four set-sized
vectors (three without the metric; the power iteration used three). The velocity objective uses the equivalent form
r = M w − α q_j − β q_{j−1} with M⁻¹ only inside the norm (as its power iteration did): device memory stays at the power
iteration's two parameter stacks + chunk scratch, q_{j−1} goes to host memory (one stack down and up per step). No
reorthogonalisation (K ≤ 10, only the top Ritz value). Output: the top Ritz value θ of the K × K tridiagonal and the
residual bound b = β_{K+1} |s_K| of its Ritz pair; α = 1 / (θ + b). θ ≤ λ_max always; b brackets *an* eigenvalue, which
is λ_max once the top Ritz value has separated (true in every case below). Checked on dense 40 × 40 problems against
eig, plain and metric, p- and q-form identical (`cargo test -p hasty-recon --lib lanczos`).
Unused λ: λ_max(N) is computed only when a non-metric precondition is in `--precond`, λ_max(M N) only for a metric one.
Warm start (velocity): with `--lam-warm on|auto` the K Lanczos vectors are kept on the host and the next stage starts
from this stage's top Ritz vector (auto: only when (K + 1) stacks fit min(HASTY_LAM_WARM_GB = 4, ¼ of MemAvailable) —
not at the full shape, where a stack is several GiB).

**Validation** (`research/experiments/lanczos/`, logs + script): data ph0001 sim_flow_10min, grid-scale 0.5
(128 × 72 × 60), `--keep 16`, 8 virtual coils, L = 6 segments, GNL + concomitant degree 4, warm 3. Relative error
against 100 power iterations (reference), applications of the operator in brackets:

| operator | power 12 [12] | Lanczos 4: θ / θ+b [4] | Lanczos 6 | Lanczos 8 |
|---|---|---|---|---|
| frame N (set 0 + middle) | −1.3e-2 | −2.8e-1 / −1.0e-1 | −2.6e-2 / +8.4e-2 | −4.9e-4 / **+1.6e-2** |
| frame M N | −1.9e-3 | −7.1e-2 / +9.1e-2 | −2.7e-3 / +2.8e-2 | −8.3e-5 / **+5.9e-3** |
| velocity M H, stage 0 (ref + high VENC) | −2.4e-2 | −8.8e-2 / +4.0e-2 | −2.8e-2 / +4.2e-2 | −7.1e-3 / **+1.8e-2** |
| velocity M H, stage 1 (all encodes), cold | −2.0e-2 | −8.1e-2 / +3.4e-2 | −2.1e-2 / +6.0e-2 | −9.0e-4 / **+1.5e-2** |
| stage 1, warm from stage 0's Ritz vector | | −8.7e-3 / +3.6e-2 | −5.5e-4 / +1.3e-2 | |

(The power-12 column for M N in frame mode is the old ‖M N v‖ growth estimate, not a Rayleigh quotient.)
Read: the old step was 0.2–2.4 % too large (unguarded); Lanczos 8 gives an upper estimate 0.6–1.8 % above the
reference with ⅔ of the operator applications per λ (velocity) and needs only the λ that is used (frame: 16 instead of
48 set applications with one precondition, 32 instead of 48 with a metric and a non-metric one). Lanczos 6 is cheaper but its bound is loose
(+3–8 %: a step that much smaller), hence the default 8.

**Short recons** (3–4 iterations, same data; Lanczos 6, power 12): frame mode ZMSHR1-M f = 2.7231e14 vs 2.7234e14 at
iteration 3, ZMSHR1 2.0443e14 vs 2.0444e14, velocity RMSE equal to 3 digits; velocity knots (4 × 8 knots, VENC
continuation 2 + 2, ZMSHR1-M): λ 4.757e7 / 4.430e7 vs power 4.465e7 / 4.288e7, f after 4 iterations 4.1838e13 vs
4.1795e13, velocity RMSE 0.695 vs 0.689 m/s (the 3–6 % smaller step; the initial states already differ by 0.003 m/s
between the two runs). Stable, no change in behaviour beyond the step.
λ time at this scale: frame 0.1 s vs 0.2 s (negligible here, segment build dominates); velocity stage 0 11.6 s vs
21.7 s, stage 1 19.6 s vs 36.5 s (6 vs 12 Hessian-vector products).

**Projection.** opbench's setup term was "15 power iterations on two sets" (645 s in the p17/p18 velocity-knot
projection, arch_targets.md); recon_p2np actually ran 2 × 12 on two sets (N and M N). With Lanczos 8 on M N only:
16 set normal operators instead of 30 projected (→ ≈ 344 s) or 48 run (≈ 1032 s at the same per-set cost).
Velocity stages: 8 instead of 12 Hessian-vector products per stage (one Hv ≈ one gradient).

**Open.** `knotncg::power_lmax` (splinerecon, finite-difference prior Hessian) and `cg::power_norm` (opval) still use
the power iteration. The residual bound could be tightened (gap-based Kato–Temple bound) but needs an upper bound on
λ_2, which Lanczos does not give. Warm start at the full velocity shape would need the basis in BFP host storage or a
second pass; not done. The host-state velocity path (`lambda_lanczos_host`) needs one more host stack than the power
iteration (three instead of two) — inside the planner's 4-stack host state for f32 stores, not for BFP stores.
