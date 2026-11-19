# Sample-set compression for the box operator (2026-11-18)

Question: can the k-space sample set be compressed exactly enough (normal operator and Aᴴy to ≤ 1e-4) that the sample
side of the operator gets much cheaper? At the 160 mm box at 1 mm the sample side is ≈ 83 % of a pass (§22 of
temporal_operator.md: ≈ 120 s of ≈ 145 s per encode pass; 96 knots, 30 coils, L 10, order 3).

Scripts and outputs: research/experiments/sample_compress/ (CPU, numpy; `out/*.json`). Data: encode 0 of
factory_arms/ph0001/sim_flow_10min (7592 readouts × 3990 ADC samples per encode, dwell 2 µs, 7.98 ms per readout).

## 0. Answer

1. **Exact compression exists, but only inside a readout.** Within one readout the knot weights are (almost) shared
   and τ, k, the ratemap phase and the concomitant phase all vary smoothly. The readout data then lie in a subspace
   whose dimension is the readout's time–bandwidth product for the ROI.
   - The compression is an interpolative decomposition along time: a subset S of the readout's own samples, plus a
     sparse local interpolator E (32 taps).
   - The data term becomes ‖Ax − y‖² = (A_S x)ᴴ W (A_S x) − 2 Re (A_S x)ᴴ b_S + ‖y‖², with W = EᴴE (banded) and
     b_S = Eᴴ y.
   - The virtual samples are ordinary samples: the existing SegNufft runs on S unchanged. The new parts are a banded
     W between forward and adjoint, and the precombined data b_S.
2. **Accuracy** (encode-0 readouts; direct-sum reference; signal-weighted ratemap and worst-case band-edge sources):
   - per readout, at os 1.35 and q 32: worst source ≤ 4e-5 signal-weighted and ≤ 6e-5 band-edge;
   - normal operator on the dense reference: ‖AᴴA − A_SᴴWA_S‖₂ / ‖AᴴA‖₂ = 1.6e-5;
   - Aᴴy: 3e-6; the objective up to its constant: 3e-6;
   - noise covariance of b_S = σ²W, measured to sampling error.
   It is not DCW: the estimator is the unweighted LS estimator, unchanged (§1.3).
3. **How much it saves depends on how oversampled the readout is for the box, not on k-space density.** Whole encode,
   ≤ 1e-4 worst source:

   | data / ROI | cube 160 mm | ball Ø 160 mm | bound (ε-rank 1e-4), cube / ball |
   |---|---|---|---|
   | native 2 mm data, box at 2 mm (30 M) | **1.86×** | **2.58×** | 2.93 / 3.90 |
   | 1 mm proxy (k × 2, same 30 M samples) | 1.08× | 1.43× | 1.51 / 2.03 |
   | 1 mm, dwell 1 µs (60 M samples) | **1.90×** | **2.66×** | — |
   | 128 mm box, 1 mm proxy | 1.22× | — | 1.88 |
   | full FOV, 2 mm | ≈ 1.0 (Landau bound 1.26) | | |

   At the 1 mm proxy the readout already samples the 160 mm cube's band at Nyquist, so there is little to remove.
   A 1 mm protocol with a 1 µs dwell (60 M samples, the §22.6 "2× samples per readout" column) compresses back to
   23–32 M: **the 60 M acquisition costs what 30 M costs today.**
4. **Dense-core hybrid (direction 2): not worth building.**
   - After the intra-readout compression the inner shell (r < 0.1 k_max) holds 3.5–6 % of the samples.
   - Those samples come from ≈ 9–20 different readouts per occupied cell, each with its own knot weights, so they are
     linearly independent rows: exact merging across readouts does not reduce them.
   - Exact pre-gridding per (segment, knot) class is impossible: y_j is a sum over classes, and the per-class
     least-squares metric is a dense class-coupled matrix.
   - Upper bound on the gain: ≤ 6 % of the sample side.
5. **Per-readout knot weights (direction 3): reject.**
   - Over one readout the cardiac coordinate moves 0.15 knot spacings: Δβ max 0.050, rms 0.027.
   - β at the readout midpoint instead of per sample: operator error 8.5 % (worst case), 6.4e-4 on a signal-like
     image. Neither meets 1e-4.
   - It saves no multiply-adds: the gather/scatter cost is s × W³ × channels per sample either way. β is already
     folded into the staged coefficients (§10, §17).
   - The respiratory weight is per readout already (bellows per TR).
6. **Held-out validation (direction 4): exact.**
   - Held-out masks are whole readouts, and the compression is per readout, so the two commute.
   - Store ‖y_r,c‖² per readout and coil. The held-out residual Σ_r [(A_S x)ᴴ W (A_S x) − 2 Re (A_S x)ᴴ b_S + ‖y‖²]
     is then the exact original residual.
7. **Recommended prototype (§6):**
   - ROI = a ball of diameter 160 mm, data demodulated to the box centre;
   - **one** S and one banded W for the whole scan, because all readouts are rotations of one base readout and a
     ball's band is rotation-invariant (measured: the prototype E on other readouts is as accurate as their own);
   - os 1.35, q 32;
   - first on the native 2 mm data (2.6× fewer samples), then decide the 1 mm dwell.

   Expected box pass (§22.6 unit costs):
   - 1 mm proxy 30 M: 151 → 113 s (ball);
   - 1 mm with 60 M: 277 → 120 s (ball) / 158 s (cube);
   - 2 mm box: 129 → 52 s (ball).

## 1. Derivation

### 1.1 What the data term sees

Write the encode's forward operator readout by readout: (A x)_r ∈ C^n, n = 3990 samples. For any linear map
E_r: C^{|S_r|} → C^n with A_r ≈ E_r A_{r,S} (A_{r,S} = the rows of A_r at a subset S_r of its samples),
- ‖A_r x − y_r‖² = ‖E_r A_{r,S} x − y_r‖²
- = (A_{r,S} x)ᴴ W_r (A_{r,S} x) − 2 Re (A_{r,S} x)ᴴ b_r + ‖y_r‖², with W_r = E_rᴴ E_r and b_r = E_rᴴ y_r.

The gradient is A_Sᴴ (W A_S x − b) and the normal operator is A_Sᴴ W A_S. The error is first order in the
interpolation error δ_r = ‖A_r − E_r A_{r,S}‖ / ‖A_r‖:
- the normal operator moves by ≤ 2δ + δ²;
- Aᴴy by δ on y's range component.

The y-component outside range(E_r) is lost. That component is noise, plus out-of-band signal, which the model cannot
fit either. Its energy is in the constant ‖y‖² − (terms).

Two remarks on the form:
- Never form W⁻¹. W is singular-ish, with condition numbers up to 1e5 measured, because E has redundant freedom
  out of band. b = Eᴴ y needs no inverse.
- Orthogonal virtual samples (Q = range basis, ‖Ax − y‖² = ‖Qᴴ A x − Qᴴ y‖² + const) reach the ε-rank, but each
  Qᴴ A row is a combination of all samples of the readout. They cannot be evaluated by the NUFFT without evaluating
  all samples, so they save nothing. The subset form (interpolative decomposition) keeps physical rows.

### 1.2 Why a readout is compressible: the range of A_r

A_r acts on box-supported knot images through coil maps, segment diagonals and the ratemap. The signal of a source
at position r with complex rate z (one complex rate covering B0, T2 decay and the fat spectrum) is

  v(t_j) = β_b(u_j) · exp(−i k_j·r − z τ_j − i c_j·mono(r)).

Coil maps multiply the image (S_c x stays box-supported). The segment interpolators a_l(τ) P_l fit e^{−zτ}. The
respiratory weight is per readout. None of them leaves the span of these columns.

Instantaneous frequency of the source: −(k̇·r + Im z + ċ·mono(r))/2π. Over the ROI the band is
- cube: B(t) = 2 Σ_a |k̇_a| h_a + B_rate;
- ball: B(t) = 2 |k̇| R + B_rate.

The signal space has dimension ≈ ∫ B dt / 2π (Landau–Pollak–Slepian). That is the readout's path length in box
Nyquist cells: L1 for the cube, L2 for the ball.

B_rate is the ratemap band, signal-weighted inside the box: 1e-4 / 1 − 1e-4 weighted quantiles of the water field
plus the six fat peaks, weighted by the truth PD. Result: −875 … +474 Hz. That is < 1 % of B(t) ≈ 190 kHz, so it is
irrelevant to the count. The decay R (signal-weighted mean 319 s⁻¹, 20 % of the signal above 1000 s⁻¹) is in the
sources.

Measured on all encode-0 readouts (`traj_stats.py`, `eps_rank.py`), n / ∫B dt:

| ROI | n / Landau | ε-rank at 1e-4 (white sources) | n / ε-rank |
|---|---|---|---|
| full FOV 508 × 287 × 228 mm, 2 mm | 1.26 | — | — |
| 160 mm cube, 2 mm | 2.67 | 1364 | 2.93 |
| 160 mm ball, 2 mm | 4.00 | 1023 | 3.90 |
| 160 mm cube, 1 mm proxy | 1.34 | 2646 | 1.51 |
| 160 mm ball, 1 mm proxy | 2.01 | 1961 | 2.03 |
| 128 mm cube, 1 mm proxy | 1.68 | 2117 | 1.88 |

Readings:
- The readout runs at constant |k̇|, so for the ball n / Landau is the same (4.00) for every readout.
- The ε-rank is below the cube's Landau count because the corners of the cube band carry little measure.
- The 1 mm proxy's fastest samples have 1.07 degrees of freedom per sample for the cube: the k × 2 proxy readout is
  marginally under Nyquist even for the box. A real 1 mm protocol needs the shorter dwell anyway.

### 1.3 Not DCW, and the noise

DCW replaces ‖Ax − y‖² by ‖D^{1/2}(Ax − y)‖² with D ≈ 1/density: a different estimator with inflated noise. Here
the objective is the original one up to an x-independent constant (measured |f − f̃| / f ≤ 3.5e-6), so the
estimator, its noise and its bias are the original LS estimator's.

W is not a density weight. It is the Gram of the interpolator, the metric the original sample space induces on the
subset; on the diagonal it is ≈ the number of samples a virtual sample stands for, i.e. a multiplier, the opposite of
DCW.

Noise: for whitened data (y = Ax + n, cov n = σ² I) the precombined data have cov b = σ² EᴴE = σ² W. The weighted
residual W^{1/2}(A_S x − W⁻¹ b) has white noise, and that is exactly the metric the objective uses. Measured in
`dense_check.py` with 3000 noise draws:
- diag(cov b) / diag(W) = 1 ± 0.0186, against 0.0183 expected from sampling;
- adjacent normalised correlations within 0.0085 of W's.

Coil whitening, coil compression and stochastic coil subsets mix or pick coils. E mixes time within a readout. They
commute.

## 2. Construction (`sc_common.py`)

Per readout:
1. Band B(t) as above (finite differences of the stored k; × 2 at the 1 mm proxy).
2. Subset S, uniform in warped time s(t) = ∫ B dt / 2π at os samples per unit, with two safeguards:
   - at most 3 original samples between kept ones;
   - all of the first and last 64 samples kept. These are the gradient ramps at k ≈ 0, where the instantaneous band
     estimate is unreliable. Without this rule the worst-source error is 10× higher, and the errors sit in rows
     0–45 and 3930–3990.
3. E: each original sample j is LS-fitted from the q subset samples nearest to it, on a training sketch of 1800
   point sources. Half are uniform in the ROI, half on the band edge (cube faces and edges, ball surface, ratemap band
   ends). Uniform-only training leaves corner sources 10× worse: 1e-3 instead of 6e-5 at os 1.5.
4. W = EᴴE is banded (q − 1 off-diagonals); b = Eᴴ y.

Error measures, on fresh sources and readouts not used for training:
- **signal-weighted:** ratemap components drawn ∝ |amplitude| (truth water / fat PD, R2* and field, six-peak fat),
  jittered inside the voxel;
- **band edge:** the worst case of the model class.

Both are reported as relative Frobenius and as the worst single source.

## 3. Results

### 3.1 Per readout (`id_readouts.py`; 6 readouts at 2 mm cube, 1–3 elsewhere; worst over readouts)

Rows: reduction n / |S|, then err = worst source (signal-weighted / band edge), then normal = ‖VᴴV − V_SᴴWV_S‖₂ /
‖VᴴV‖₂ on the test sources (signal / edge).

| case | os 1.25, q 32 | os 1.35, q 32 | os 1.5, q 32 | os 2.0, q 32 |
|---|---|---|---|---|
| 160 cube, 2 mm | 1.99× | **1.86×** | 1.69× | 1.29× |
| err | 2.6e-4 / 3.9e-4 | **4.0e-5 / 5.8e-5** | 3.1e-5 / 3.8e-5 | 1.6e-5 / 2.5e-5 |
| normal | 2.6e-5 / 4.1e-5 | 8.7e-6 / 1.1e-5 | 5.6e-6 / 6.5e-6 | 1.7e-6 / 1.3e-6 |
| 160 ball, 2 mm | 2.70× | **2.58×** | 2.39× | 1.88× |
| err | 1.7e-4 / 3.3e-4 | **3.7e-5 / 1.4e-4** | 3.2e-5 / 9.4e-5 | 1.6e-5 / 4.8e-5 |
| 160 cube, 1 mm proxy | 1.12× | 1.08× | 1.05× | 1.02× |
| err | 4.8e-4 / 1.2e-3 | 8.8e-5 / 2.3e-4 | 3.3e-5 / 4.3e-5 | 1.3e-5 / 1.5e-5 |
| 160 ball, 1 mm proxy | 1.54× | 1.43× | 1.29× | 1.07× |
| err | 2.3e-4 / 4.8e-4 | 4.8e-5 / 1.3e-4 | 3.3e-5 / 7.8e-5 | 1.6e-5 / 2.4e-5 |
| 128 cube, 1 mm proxy | 1.30× | 1.22× | 1.14× | 1.04× |
| err | 2.7e-4 / 5.2e-4 | 5.2e-5 / 7.8e-5 | 3.6e-5 / 4.4e-5 | 2.3e-5 / 2.1e-5 |
| 160 cube, 1 mm, dwell 1 µs | **2.07×** | 1.93× | 1.75× | 1.34× |
| err | **5.9e-5 / 9.3e-5** | 3.5e-5 / 4.5e-5 | 2.2e-5 / 2.8e-5 | 4.5e-6 / 8.0e-6 |
| 160 ball, 1 mm, dwell 1 µs | **2.79×** | 2.66× | 2.47× | 1.92× |
| err | **7.4e-5 / 8.7e-5** | 2.3e-5 / 3.7e-5 | 1.7e-5 / 2.5e-5 | 5.0e-6 / 1.0e-5 |

- q 24 is 2–4× worse than q 32 at os ≤ 1.35. q 48 gains nothing.
- The 1 µs-dwell case interpolates the stored trajectory (cubic spline in the ADC index). The real 1 µs waveform on
  the 4 µs gradient raster is less smooth [CHECK on a simulated 1 µs schedule].
- The ball's band-edge error at os 1.35 (1.4e-4) is slightly above 1e-4; os 1.5 meets it at 2.39×.
- The achieved reduction is 64–75 % of the ε-rank bound. The gap is the local interpolator's oversampling (os) and
  the ramp safeguards. A global ID would close it, but needs a dense |S|² W per readout (15 MB).

### 3.2 Dense reference (`dense_check.py`)

Setup:
- 4 encode-0 readouts, native scale;
- 16 cardiac knot images on a 10³ grid spanning the 160 mm cube, corners included, with each voxel's truth ratemap
  rate (R2* + i·field) and the per-sample concomitant phase;
- direct sums, no NUFFT;
- E trained on independent continuous sources.

| os, q | reduction | ‖AᴴA − A_SᴴWA_S‖₂ / ‖AᴴA‖₂ | same on a signal-like image | Aᴴy | objective |
|---|---|---|---|---|---|
| 1.25, 32 | 1.98× | 6.0e-5 | 1.7e-5 | 1.1e-5 | 3.3e-6 |
| **1.35, 32** | **1.85×** | **1.6e-5** | **4.6e-6** | **3.3e-6** | **3.5e-6** |
| 1.5, 32 | 1.68× | 7.9e-6 | 3.0e-6 | 2.0e-6 | 8.5e-7 |
| 2.0, 24 | 1.28× | 2.5e-6 | 1.1e-6 | 7.9e-7 | 7.6e-7 |

Notes:
- The signal-like image is the truth PD with a 50 % cardiac modulation on the lumen voxels; y is that image's data
  plus 2 % white noise.
- Spectral norms are by power iteration on the active knots.
- The spectral error is ≈ 2× the per-readout Gram error of §3.1, as §1.1 predicts.

### 3.3 One interpolator for the scan (`proto_check.py`)

All readouts are rotations of one base readout, and |k̇| is rotation-invariant. For the ball the band, S and (after
demodulating the data to the box centre, y ← y e^{+i k·c + i c_j·mono(c)}) the signal class are the same for every
readout. Only the concomitant cross terms and the cardiac weights differ, and both lie inside the design band.

Test: E fitted on one readout only, applied to 4 others, ball, os 1.35, q 32.
- The subset is identical.
- Signal-weighted worst source: 2.9–3.2e-5 with the prototype E, against 2.9–3.3e-5 with each readout's own E.
- Band edge: 5.0–7.4e-5.

So the whole scan needs one S (≈ 1550 indices) and one banded W (≈ 1550 × 32 complex, 0.4 MB), not 4 GB of per-readout
W.

For the cube the band depends on the readout direction. Either:
- per-readout W, Hermitian band 32 complex64 ≈ 256 B per virtual sample: ≈ 4 GB per encode at 16 M virtual
  samples, host-resident and streamed at 0.16 s per encode pass; or
- a shared design for the circumscribed ball, which costs a factor √3 in band: ≈ 1.5× at 2 mm [estimate from the
  Landau scaling, not run].

### 3.4 Whole encode and what is left in the core (`encode_reduction.py`, every 4th readout of encode 0)

Whole-encode reduction n / |S|:

| case | os 1.25 | os 1.35 | os 1.5 | os 2.0 |
|---|---|---|---|---|
| 160 cube, 2 mm | 1.99 | 1.86 | 1.69 | 1.29 |
| 160 ball, 2 mm | 2.70 | 2.58 | 2.39 | 1.88 |
| 160 cube, 1 mm proxy | 1.11 | 1.08 | 1.05 | 1.02 |
| 160 ball, 1 mm proxy | 1.54 | 1.43 | 1.29 | 1.07 |
| 128 cube, 1 mm proxy | 1.30 | 1.22 | 1.14 | 1.04 |
| 160 cube, 1 mm, dwell 1 µs | 2.04 | 1.90 | 1.73 | 1.32 |
| 160 ball, 1 mm, dwell 1 µs | 2.79 | 2.66 | 2.47 | 1.92 |

Density after compression, os 1.35, shells of the 2N grid as in knot_gram_box1mm_tri.json:

| shell | share of samples before → after, 2 mm cube | 1 mm proxy cube | samples per readout visit after (2 mm / 1 mm) | readouts per occupied cell after (2 mm / 1 mm) |
|---|---|---|---|---|
| 0–0.1 | 3.3 % → 5.9 % | 3.3 % → 3.5 % | 5.5 / 3.1 | ≈ 20 / ≈ 9 |
| 0.1–0.3 | 6.2 % → 3.9 % | 6.2 % → 3.5 % | 1.3 / 1.1 | ≈ 5 / ≈ 4 |
| 0.3–0.6 | 22 % → 18 % | 22 % → 20 % | 1.1 / 1.1 | ≈ 5 / ≈ 4 |
| 0.6–1.0 | 68 % → 72 % | 68 % → 73 % | 1.1 / 1.0 | ≈ 5 / ≈ 4 |

(Readouts per occupied cell are counted on every 4th readout and scaled by 4.)

- The "58 samples per cell" of the inner shell is mostly one readout dwelling: before compression, 5.6 samples per
  readout visit at 2 mm, 3.1 at 1 mm.
- The ramp safeguard keeps the core uncompressed (reduction 1.03 in r < 0.1), so the core's share grows. Within one
  box Nyquist cell of k = 0 sit 1.7 % of all samples (68 per readout).

## 4. Direction 2: dense-core hybrid

- **Exact per-class pre-gridding does not exist.**
  - The data y_j = Σ_{l,b} φ_{j,lb} X_{lb}(k_j) sum over (segment, knot) classes, so the data cannot be split per
    class.
  - The exact LS reduction onto Cartesian per-class values X_{lb}(k_g) has a metric (C ⊗ Φ)ᴴ(C ⊗ Φ): dense over
    classes (≈ 96 × L) and over the q³ interpolation neighbourhood. That is the class-pair Toeplitz problem of the knot
    Gram study (233–364 knot pairs at 1e-4, times L² segment pairs), restricted to a region.
  - The only exact grid form is that Toeplitz route, already studied and rejected for memory.
- **Merging across readouts is not exact either.**
  - In a core cell after compression ≈ 9–20 readouts contribute, each with its own (cardiac, respiratory) weight
    vector in the 96-dimensional knot space.
  - The rows are linearly independent, so the LS rank equals the row count. Merging needs near-identical physics, to
    ≤ 1e-4, and continuous cardiac times never coincide that closely.
  - The only cell where readouts outnumber the physics rank (96 knots × ≈ 2 τ classes) is the one around k = 0, with
    1.7 % of the samples.
- **GROG-type data gridding** shifts samples with coil GRAPPA kernels. Its error is ≈ 1e-2, far from 1e-4.
- **Upper bound on any core treatment:** the inner shell's share of the sample side, ≤ 6 % after compression and
  3.3 % before. A second operator path for that is not justified.

## 5. Direction 3: per-readout knot weights

Measured:
- **Cardiac:** one readout spans 7.98 ms / RR 853 ms = 0.0094 cycles, 0.15 knot spacings at 16 knots. Over one
  readout Δβ max is 0.050 and rms 0.027 (dense_check).
- **Respiratory:** the bellows is recorded once per TR, so the respiratory weight is per readout by construction. Its
  change over one readout is 0.020 rms and 0.075 max in knot units. Physically this is ≈ 0.1 mm of breathing motion.
- **β at the readout midpoint instead of per sample:**
  - worst-case operator error 8.5 %, with independent knot images;
  - 6.4e-4 on the signal-like image.
  Both are above the 1e-4 target. Per-sample cardiac time stays.

Work: the knot-major kernel's per-sample cost is s × W³ × (cb · L) multiply-adds (§17: MACs 9.4 s, staging 4.6 s of
19 s). With per-readout β:
- the MAC count is unchanged, since every sample still needs its s slots;
- β, which is already folded into the staged coefficients, would be computed once per readout instead of per
  (sample, knot): a fraction of the 25 % staging share;
- the per-sample time saves 4 B per sample of tables.

No material gain. The real per-readout structure is the one direction 1 uses: smoothness along the readout, not
constancy.

## 6. Recommended prototype

1. **Offline, once per scan and trajectory model** (CPU, seconds; extends sc_common.py):
   - ROI: ball, diameter = box edge (160 mm), centred on the box. Box centre and ratemap band from the data:
     prescription box, and the ratemap estimate's signal-weighted quantiles plus a margin for drift (the band is < 1 %
     of B(t), so a 2× margin is free).
   - S from the base readout, os 1.35 (os 1.5 if the band-edge worst case must be < 1e-4), ≤ 3 samples apart, first
     and last 64 samples kept.
   - E: local q = 32 taps fitted on a 1800-source sketch, half on the band edge. W = EᴴE, banded, shared by all
     readouts.
   - Gate: dense_check.py-type direct-sum test, normal-operator error < 1e-4.
2. **Data path** (kload / KStore):
   - demodulate y_r by the box centre;
   - b_r,c = Eᴴ y_r,c per readout and coil (q MACs per original sample, once);
   - store b (1.9–2.7× smaller than y) and ‖y_r,c‖²;
   - build the tables (k, τ, concomitant, cardiac time, bellows) on S only. Table memory and build time shrink by the
     same factor.
3. **Operator:**
   - SegNufft on S, unchanged;
   - one new step between forward and adjoint: r = W (A_S x) − b, banded, per readout and coil;
   - the objective via the formula of §1.1.
   Cost of the W step: 63 complex MACs per virtual sample and coil, ≈ 0.2 TFLOP per encode pass at 16 M virtual
   samples, ≈ 0.1 s, against ≈ 50–130 s of sample side.
4. **Held-out readouts:** excluded from the fit set. Their residual is evaluated with the same formula; it is exact
   with the stored ‖y‖².
5. **First test (dev scale):**
   - native 2 mm data, 160 mm box at 2 mm (or the 4 mm pyramid level: 2.1–2.3× there), compressed against
     uncompressed;
   - same iterates to ≤ 1e-4 relative after a fixed number of iterations;
   - held-out residual curves identical up to the constant;
   - sample-side time ÷ ≈ 2.6 (ball) or 1.9 (cube).
   One factor only: no other knob changes.

**Expected pass times** (box 160 mm, target config: 96 knots, 30 coils, L 10, order 3; T = K · I + M′ · S with
K · I = 25 s and S = 4.2 s per 10⁶ samples from §22.6; W step negligible):

| case | M per encode | today | cube | ball |
|---|---|---|---|---|
| 1 mm, 30 M (k × 2 proxy) | 30 M | 151 s | 142 s (1.08×) | **113 s** (1.43×) |
| 1 mm, 60 M (dwell 1 µs) | 60 M | 277 s | 158 s (1.90×) | **120 s** (2.66×) |
| box at 2 mm (native data) [image side ≈ 25 / 8 s, estimate] | 30 M | ≈ 129 s | 71 s | **52 s** |

8 × 5090, per gradient = 0.625 × T_pass (7 encodes):
- 1 mm, 60 M, ball: ≈ 75 s per gradient, ≈ 2.1 h per 100 iterations, against 4.7 h uncompressed (§22.6 correction);
- the 30 M proxy, ball: ≈ 71 s per gradient, against 91 s.

## 7. Pitfalls

- **ROI band design:**
  - The cube's band depends on the readout direction (L1 norm of k̇), so either per-readout W or the circumscribed
    ball.
  - Rigid-motion correction rotates and shifts k: the ball band is invariant, and a translation is a per-sample
    phase.
  - A GIRF-modelled trajectory breaks the exact rotation symmetry slightly: add band margin, or recompute E per
    readout class.
  - Out-of-ROI signal must be subtracted before compression, as the box recon already requires. Whatever lies out of
    band goes into the constant.
- **Ramps:** without the end-sample rule the worst error is 10× higher. The band estimate from finite differences
  fails where k barely moves.
- **Training sketch:** must include band-edge sources. Uniform training gives 1e-3 at the cube corners.
- **Never form W⁻¹**, nor "whitened virtual samples" by W^{−1/2}: cond(W) reaches 1e5. Use b = Eᴴy and apply W
  forward.
- **Weights inside a readout** (sample masks, IRLS or robust weights per sample, a recon window): apply windows
  before compressing.
  - Per-readout scalars (noise level, held-out flags, coil subsets) commute.
  - Per-sample weights that vary within a readout break exactness unless they are smooth and included in E's
    training.
- **Preconditioner:** P²nP's circulant preconditioner should keep being built from the original coordinates, with unit
  weights, once. The subset alone would mis-model the PSF density by the local reduction factor. Weighting the subset
  by diag(W) is not a substitute for W in the data term either: in an early run (uniform training, no ramp rule)
  diag(W) alone gave normal-operator errors of 0.15 to > 1. Diagonal weights reproduce the Gram exactly only as
  quadrature weights for products of two in-band signals, which needs os ≥ 2 (≤ 1.3× reduction).
- **Segment basis rebuilds:** the virtual samples are original samples (their τ, concomitant terms and cardiac time
  come from the tables), so a_l(τ) rebuilds need nothing new. E depends on the physical band only, not on the ratemap
  estimate or L, so it needs no rebuild while the ratemap estimate stays inside the design band.
- **NUFFT error budget:** compression adds ≈ 1.6e-5 to the normal operator (dense test). That is comparable to a
  W 10 NUFFT and below the segment-model target (1e-4). Keep os ≥ 1.35 so it stays ≪ the physics error.
- **1 µs dwell:** measured on an interpolated trajectory. Re-measure on a schedule simulated at 1 µs before deciding the
  protocol.

## 8. Files

- research/experiments/sample_compress/sc_common.py: band, ratemap sources (signal-weighted), sketches, selection,
  local E.
- traj_stats.py: per-readout Landau counts and density origin (`out/traj_stats_s2.json`).
- id_readouts.py: per-readout ID grid over os, q and ROI (`out/id_readouts.json`).
- eps_rank.py: SVD bounds (`out/eps_rank.json`).
- dense_check.py: direct-sum normal operator, Aᴴy, objective, noise covariance and knot-midpoint test
  (`out/dense_check_g10_r4.json`).
- proto_check.py: one E for all readouts (`out/proto_check.json`).
- encode_reduction.py: whole-encode reduction and post-compression density (`out/encode_reduction_s4.json`).

## 9. Implementation and measured gains (2026-11-18)

### 9.1 What was built

**Library: `hasty_recon::rcompress`** (rustlib/crates/hasty-recon/src/rcompress.rs, kernels rcompress.cu).

- *Design* (`Design::build`, per readout, CPU f64): band B(t) = support width of the ROI along k̇ + rate band;
  subset S uniform in warped time at `os` per Nyquist unit, ≤ 3 samples apart, first/last 64 samples kept, and
  16 samples on each side of a τ gap (a readout cut by a k-space crop; without this rule the pieces' ends are
  extrapolated one-sided: gate 4e-3 at the 2 mm grid on the 1 mm scan, 3.4e-5 with ≥ 8).
- *Interpolator E*: q = 32 taps, fitted against the covariance K(i, j) = E[v_i v̄_j] of the study's source measure
  (half uniform in ROI × band, half on its edge: ROI boundary, band ends; decay uniform in [0, R_max]) in closed
  form. That is the P → ∞ limit of the 1800-source training sketch of sc_common.py. Rows sharing a window share one
  Cholesky factor, so a 3990-sample readout costs ≈ 5 ms (box: 475 readouts in 2.3 s on 32 threads).
- *Data-term form*: the banded Gram W = EᴴE is factored as W = RᴴR (banded Cholesky, R upper, q − 1 superdiagonals).
  The solvers then see an ordinary LS problem:
  ‖Ax − y‖² = ‖R A_S x − ỹ‖² + (‖y‖² − ‖ỹ‖²), ỹ = R⁻ᴴ Eᴴ y.
  This is the W form of §1.1 with the factor applied on both sides. ỹ = Qᴴy for E = QR (the coordinates of y's
  projection on range E, ‖ỹ‖ ≤ ‖y‖), computed by f64 forward substitution with cond(R) = cond(W)^½. W⁻¹ is never
  formed and there are no W^{−1/2} virtual samples (§7).
  - Why R and not W: every solver that forms residuals (CGLS with the explicit residual, P²nP, the velocity
    model's data term, held-out residuals) works unchanged. The operator is R A_S, its adjoint A_Sᴴ Rᴴ, the data are
    ỹ, and the constant is kept per readout and coil (`SetComp::cnorm`, with the original energy `SetComp::ynorm`
    for relative held-out residuals).
  - Noise in ỹ is white (cov = σ² R⁻ᴴ W R⁻¹ = σ² I), so the coil whitening, coil compression and stochastic coil
    subsets commute.
- *Shared / per-readout designs*:
  - Ball ROI: one design per τ grid, valid for every readout that is a rotation of it. Each readout is checked by a
    Kabsch fit, with residual 3e-7 on both scans; a readout failing the check (e.g. a GIRF-played trajectory) gets
    its own design automatically.
  - Box ROI: one design per distinct readout (the 7 encodes of a rotation share it), kept as S plus an f32 R
    (256 B per kept sample). `Compressor::compress_set` streams in chunks: each readout's interpolator is built,
    used for its data and dropped.
- *ROI and band from the prescription and the operator, never from truth*:
  - ROI = the reconstruction grid (box, per readout) or the ball enclosing it (one design). `ball:OS:R_MM` gives a
    smaller ball; the unknowns outside it are then excluded by a voxel mask in the operator wrapper (`roi_mask`), so
    the problem stays exact.
  - Band = the coil-energy-weighted 1e-4 quantiles of the operator's ratemap (field incl. `--rate`, R2*, the fat
    peaks when a fat model is on), widened 2× (≥ ±200 Hz), decay ≥ 300 1/s.
- *Gate* (`gate`, `gate_set`): direct sums on the readout's own samples, with fresh sources (half on the band edge),
  the readout's concomitant phase and a cardiac knot envelope. Reported: worst source ‖v − E v_S‖/‖v‖ and the
  normal-operator error on the sources' span, ‖VᴴV − (RV_S)ᴴ(RV_S)‖₂/‖VᴴV‖₂.
- *GPU*: `RcDev` (all designs' R in one f32 table, per set readout offsets) and the in-place banded multiply
  `rc_apply` (one block per readout, chunked so it reads only values not yet overwritten). Wrappers:
  - `RcSets: SetOp`, the outermost wrapper (voxel-basis weights, fat spectrum and coil subsets act inside, on the
    kept samples). Forward / adjoint / group / multi pass through; one data-size scratch for Rᴴ.
  - `RcLin: LinOp`, for splinerecon and the tests.

**Flags.** `--rcompress off|auto[:OS]|ball[:OS[:R_MM]]|box[:OS]`, default off, in splinerecon and recon_p2np.
- `auto` estimates the reduction of both ROIs from the selection rule on set 0 (cheap: no interpolator). It takes
  the better one when ≥ 1.2×, then runs the gate and raises os by 0.15 until the gate's normal-operator error is
  ≤ 1e-4 (off if the reduction then drops below 1.2×).
- Printed: band, estimated and achieved reduction, gate, R table size and the data constant.
- recon_p2np (run.json `estimates.rcompress`, `rcompress_gate`):
  - The compression runs after the warm start and the circulant metric, which are built from the original samples:
    the metric needs the original density (§7).
  - It runs before the segment operator: the store, device or pinned data are replaced by ỹ, the records cropped to
    S, and `crop_sel` composed with the selection. The segment tables are built on the whole records and cropped,
    exactly as for `--kcrop`, and kcrop and rcompress compose.
  - Edits: option parsing, one block before the operator stack, one wrapper after the synthetic block. Needs
    `--model seg`, the streamed data path and no `--synthetic`.
- splinerecon: per encode the fit and held-out parts are compressed separately (whole readouts). The circulant and
  knot preconditioners use the uncompressed fit samples. The held-out residual is
  (‖R A_S x − ỹ‖² + Σ cnorm) / Σ ynorm, the exact original. `--arms` is refused with `--rcompress`.

**Bench.** `rcompress_bench` (hasty-bench):
- design, reduction, Landau bound and gate on a real trajectory for any grid, ROI, k-scale (`--kscale`, the 1 mm
  proxy) and dwell (`--upsample F`: Catmull–Rom along the ADC index);
- on the GPU, forward + adjoint per encode pass on all samples against the kept samples plus the R step (static,
  1-D knot-major or `--knots2d 6,16` = the target 96-knot engine);
- the accuracy of both against a W 12 / σ 2 reference (`RC_BENCH_REF`).

Scripts: research/experiments/sample_compress/survey_cpu.sh, survey_gpu.sh, driver_check.sh; results in
`out/impl/{cpu,gpu}_*.json`.

**Tests.**
- `rcompress` unit tests (CPU):
  - kept rows exact, ‖Ev‖ = ‖Rv‖, the objective identity;
  - the direct-sum gate of dense_check.py on a synthetic readout, a 7³ voxel grid with voxel rates, concomitant
    phase and 4 cardiac knots, reduction 2.58×: normal operator 2.6e-7, gradient 9.5e-8, objective 3.8e-9;
  - rotation sharing of ball designs;
  - flag parsing.
- `tests/rcompress.rs` (GPU, real data, §9.3).

### 9.2 Achieved reduction on the real trajectories (CPU: selection, designs, gate)

**Setup** (`survey_cpu.sh`):
- os 1.35, q 32, band ±1.5 kHz, decay ≤ 300 1/s;
- encode 0, every 4th–32nd readout;
- "readout FOV" = 2π / (|k̇| · dwell) at the median sample: 0.60 m for the 2 mm protocol (sim_flow_10min, 3990
  samples, 2 µs) and 0.47 m for the 1 mm protocol (flow_thorax_v2_1mm, box1mm_b, 5720 samples, 2 µs);
- "1 mm proxy" = the 2 mm readout with k × 2 (readout FOV 0.30 m).

Columns:
- Landau = n / ∫B dt / 2π, the readout's oversampling for the ROI;
- gate = worst source / normal operator on the sources (4 readouts).
- A ball smaller than the grid's enclosing ball means the unknowns are masked to it; so does the ellipsoid inscribed
  in the grid box.

| grid | data / readout | ROI | Landau | reduction | gate |
|---|---|---|---|---|---|
| full FOV 2 mm 254×144×114 | 2 mm protocol | box 510×290×230 (per readout) | 1.26 | **1.08×** | 1.1e-4 / 1.6e-5 |
| | | enclosing ball r 315 | 1.02 | 1.02× | 3.5e-5 / 4.3e-6 |
| | | inscribed ellipsoid (body-like support) | — | **1.32×** | 1.1e-4 / 6.7e-6 |
| box 160 mm at 2 mm (80³) | 2 mm protocol | box (per readout) | 2.63 | **1.83×** | 5.2e-5 / 6.9e-6 |
| | | enclosing ball r 140 | 2.28 | 1.61× | 3.8e-5 / 5.0e-6 |
| | | ball r 80 (masked) | 3.95 | 2.56× | 3.5e-5 / 3.5e-6 |
| box 160 mm at 1 mm (160³) | 1 mm protocol | box | 1.99 | **1.41×** | 2.7e-5 / 3.5e-6 |
| | | enclosing ball r 139 | 1.72 | 1.24× | 1.9e-5 / 2.4e-6 |
| | | ball r 80 (masked) | 2.98 | 2.06× | 1.5e-5 / 1.4e-6 |
| | 1 mm proxy | box / enclosing ball / ball r 80 | 1.33 / 1.15 / 2.00 | 1.08× / 1.04× / 1.42× | ≤ 9.8e-5 / 1.9e-5 |
| box scan's own FOV 198×189×228 at 1 mm | 1 mm protocol | box / enclosing ball | 1.56 / 1.35 | 1.16× / 1.07× | 3.1e-5 / 3.9e-6 |
| same scan, 2 mm grid (k-cropped to the 2 mm band) | 1 mm protocol | box (FOV) / box 160 | 1.96 / 2.49 | 1.37× / 1.64× | 3.4e-5 / 6.8e-6 |
| full FOV 1 mm 508×287×228 | 1 mm protocol | box / inscribed ellipsoid | 0.93 | 1.03× / 1.12× | 1.3e-4 / 1.3e-5 |
| torso 640×384×384 at 0.94 mm | 1 mm protocol | box / enclosing ball / ellipsoid | 0.72 / 0.61 / — | 1.01× / 1.00× / 1.04× | ≤ 6.5e-5 / 5.5e-6 |
| | 1 mm proxy | box | 0.49 | 1.00× | 5.8e-4 / 5.8e-5 |
| head 320³ at 1 mm | 1 mm protocol | box / enclosing ball / ellipsoid | 0.99 / 0.87 / — | 1.02× / 1.01× / 1.12× | ≤ 5.4e-5 / 4.2e-6 |
| | | ball r 160 / ball r 100 (head support, masked) | 1.50 / 2.39 | 1.12× / **1.69×** | ≤ 4.0e-5 / 3.2e-6 |
| | 1 mm proxy | box / ball r 100 | 0.67 / 1.60 | 1.00× / 1.16× | 1.2e-4 / 6.7e-6 |
| head 384³ at 1 mm | 1 mm protocol | box / ball r 100 | 0.83 / 2.39 | 1.01× / 1.69× | ≤ 2.2e-5 / 2.9e-6 |

Readings:
- **The reduction is a property of readout FOV ÷ ROI width along k̇, not of the grid or k-space density.**
  - Above Landau ≈ 2 the achieved reduction is ≈ 0.65–0.75 of the Landau bound, as in §3.1.
  - Below ≈ 1.4 the os 1.35 rule and the 64 end samples leave ≤ 1.1×.
- **The readouts are designed at about Nyquist for their full FOV, so full-FOV grids gain nothing**:
  - 2 mm full FOV 1.08× (Landau 1.26);
  - 1 mm full FOV 1.03×, and the readout is under Nyquist for the box (Landau 0.93);
  - torso 1.00–1.01× (0.72);
  - head 320³ / 384³ box 1.01–1.02× (0.99 / 0.83).
  - The 1 mm proxy is less representative still: k × 2 at the same dwell halves the readout FOV, to 0.30 m. A real
    1 mm protocol needs a readout FOV ≥ the object, which is what the v2 readout has (0.47 m).
- **Where it gains**:
  - ROIs narrower than the readout FOV: the heart boxes (1.4–1.8× per-readout box, 1.24–1.6× with the shared
    enclosing ball);
  - support-restricted unknowns: masked balls 2.1–2.6×; a head occupying a Ø 200 mm ball inside a 320³ or 384³ grid,
    1.69×;
  - a body-like ellipsoid at full FOV, 1.32× at 2 mm. That needs a data-derived support: every voxel outside it must
    be zero. The printed "dropped energy ÷ noise expectation" is the check.
- The gate (normal-operator error on the source span) is ≤ 2e-5 everywhere. The worst single source reaches
  1.1–1.3e-4 only where Landau ≈ 1, at band-edge corners; there the reduction is ≈ 1 and `auto` is off. One case
  differs: the 1 mm torso proxy under Nyquist (0.49) reaches 5.8e-4 / 5.8e-5, still ≤ 1e-4 on the operator.
  `auto` raises os until the operator error is ≤ 1e-4.

**Protocol change: faster ADC (dwell / F, `--upsample F`; cubic interpolation of the stored records, so the real
waveform on its 4 µs raster is slightly less smooth, see §7).** The cost that matters is the compressed sample count
against today's Nyquist acquisition: F × M / reduction ÷ M.

| case | dwell / 2: reduction | samples vs today | dwell / 3: reduction | samples vs today |
|---|---|---|---|---|
| full FOV 2 mm, box | 1.78× | **1.13×** | 2.50× | 1.20× |
| full FOV 2 mm, enclosing ball | 1.47× | 1.36× | | |
| full FOV 2 mm, inscribed ellipsoid | 2.41× | 0.83× | | |
| full FOV 1 mm, box | 1.36× | 1.47× | | |
| torso 1 mm, box | 1.14× | 1.76× | | |
| head 320³, box | 1.44× | 1.39× | 2.11× | 1.42× |
| head 320³, ball r 100 (support) | 2.90× | 0.69× | | |
| box 160 at 1 mm, box | 2.63× | 0.76× | | |
| box 160 at 1 mm, ball r 80 | 2.92× | 0.68× | | |

**Answer for full FOV.** Per-readout compression cannot make a Nyquist-sampled full-FOV reconstruction cheaper.
- It removes only the readout's oversampling relative to the ROI, and at full FOV there is none; at os 1.35 the
  compressed count stays ≥ the Nyquist count.
- A faster ADC (dwell / 2 or / 3) does not help either: the compressed count is 1.13–1.20× today's at 2 mm and
  1.4–1.8× at 1 mm.
- The trajectory's sample count is already at its information limit for that ROI.
- The two levers that do gain at full FOV:
  1. **A support-restricted ROI** (body ellipsoid or a data-derived convex hull per readout; unknowns outside masked):
     1.32× today, 0.83× of today's samples with dwell / 2. That is a modelling decision. It is exact only if the data
     have no signal outside, which the dropped-energy check verifies on the data.
  2. **A sample budget spent elsewhere**: compression makes the 1 µs-dwell 1 mm box protocol cost what 30 M costs
     (§0 item 3, box 160 at 1 mm with dwell / 2: 0.68–0.76× of today's samples).

### 9.3 GPU: pass time and accuracy (measured so far)

**First measurement** (static operator, 8 or 4 coils, L 10, order 2, every 8th readout of encode 0: 3.8 M samples;
2 mm box 80³ on the 2 mm protocol, enclosing ball, reduction 1.605×):

| case | all samples fwd + adj | kept fwd + adj | R step fwd + adj | pass speed-up |
|---|---|---|---|---|
| 8 coils, W 10 / σ 1.25 | 193.9 + 175.1 = 368.9 ms | 122.8 + 107.7 = 230.5 ms | 2.08 + 2.19 ms | **1.57×** |
| 4 coils, W 10 / σ 1.25 | 95.6 + 96.3 = 191.9 ms | 65.0 + 60.6 = 125.6 ms | 1.02 + 1.09 ms | 1.50× |
| 4 coils, W 12 / σ 2 | 149.7 + 151.9 = 301.6 ms | 94.8 + 96.7 = 191.5 ms | 1.03 + 1.08 ms | 1.56× |

- The R step costs ≈ 1 % of the kept pass. Table builds shrink by the same factor (3.5 s → 2.1 s), and so does the
  device memory of the tables.
- The pass speed-up is ≈ the reduction where the sample side dominates (the 80³ grid's image side is small).

**Accuracy, and a finding about the NUFFT.**
- Compressed against all samples, *both* at W 10 / σ 1.25:
  - normal operator 2.0–2.7e-4;
  - gradient at 0.6 x_true 1.6–2.5e-4;
  - objective (with the data constant) 5e-9.
- At W 12 / σ 2 the same comparison gives normal operator 1.2e-6, gradient 1.5e-6, objective 4e-8.
- So the 2e-4 is the W 10 / σ 1.25 NUFFT's own error on a white (full-band) test image. The two sample sets sample
  that error differently. The compression itself contributes ≈ 1e-6, consistent with the gate (Gram 5e-6).
- Consequence for the accuracy rule: on full-band images the production kernel is at ≈ 1e-4 relative on the normal
  operator. The bench now compares both operators with a W 12 / σ 2 reference (`accuracy against the reference`);
  those runs are queued (§9.5).

### 9.4 Open issues

- **Box mode memory.** The R tables take 256 B per kept sample, all resident on the device (`RcDev`): 4.2 GB per
  encode set at the 2 mm box with all 7592 readouts, 0.3 GB at the 1 mm scan's 1/4 readouts. No per-set streaming
  from the host yet; the shared ball design needs ≤ 2 MB. If the R step's L2 misses matter at that size, store R per
  rotation class.
- **Data constant and the drivers' printed objectives.** recon_p2np's printed data-term values / "relative weighted
  error" are of ‖R A_S x − ỹ‖ (offset by the constant), not ‖Ax − y‖. Held-out residuals (splinerecon) are corrected
  exactly; P²nP / vmodel logs are not.
- **Masked ROIs.**
  - `ball:OS:R_MM` and `ellipsoid` restrict the unknowns (mask in `RcSets` / `RcLin`). The multi-image paths
    (`forward_multi` / `adjoint_multi`: shared water + fat gridding) refuse a mask.
  - Whether the data have signal outside the ROI is checked only by the dropped-energy ÷ noise ratio, printed and
    not enforced.
  - A data-derived support ROI (convex hull per readout) is not implemented; the ellipsoid is the stand-in.
- **The gate omits the per-sample concomitant residual tables and GNL.** The bench timing uses zero residual tables.
  The real-data test and the driver checks use the real operator.
- **The `auto` threshold** (1.2×) uses the sample-count reduction. At 1 mm grids with a large image side (torso:
  half the pass) the pass speed-up is smaller than the reduction.
- **Rotation sharing breaks under a GIRF-played trajectory.** It falls back to one design per readout, automatically,
  with the cost and memory of box mode.
- **1 µs-dwell numbers come from interpolated records** (§7). The interpolation itself is not measured.
- **recon_p2np** needs the streamed data path, `--model seg` and no `--synthetic`. The warm start and the circulant
  metric run on the original samples (by placement). `--rcompress` with `--vmodel` is untested end to end; the hooks
  are generic SetOp.
- **The bench and the test take the GPU lock only around their GPU parts** (`--gpu-lock`, `RC_GPU_LOCK`). The driver
  checks hold it through their CPU setup.

### 9.5 Status / resume (2026-11-18 15:00)

**Done and verified:**
- library, wrappers, flags (splinerecon, recon_p2np), bench; everything builds;
- `rcompress` unit tests pass (`cargo test --release -p hasty-recon --lib rcompress`): synthetic direct-sum gate
  2.6e-7;
- CPU survey complete (§9.2; `out/impl/cpu_*.json`, `survey_cpu.sh`);
- first GPU timing and the NUFFT-error finding (§9.3).

**Not yet verified:**
- the real-data CGLS test (tests/rcompress.rs; box1mm_b at 2 mm, reduction 1.353×, 287 + 67 readouts);
- the GPU survey (accuracy vs reference; full-encode timing with the 96-knot engine);
- the driver end-to-end checks.

**Queued:** unit `rc-gpu-queue` (`research/experiments/sample_compress/run_queue.sh <gpu.lock>`). It runs snapshot
binaries in rustlib/target/rcompress_snap/ and is resumable: finished parts are skipped on restart. Log:
`logs/queue.log`.
1. `logs/test_rcompress.log`: the real-data test, at W 10 / σ 1.25 and W 12 / σ 2. Expected "test result: ok";
   the asserts are ≤ 1e-4 on objective, gradient, 8 CGLS iterates and the held-out curve. At W 10 the NUFFT's own
   error (§9.3) may exceed 1e-4 on the iterates; then the W 12 line is the compression's number.
2. `logs/gpu_<case>.log`, `out/impl/gpu_<case>.json`:
   - accuracy against the reference: acc_box2_box, acc_box2_ball, acc_box1_box, acc_box1_ball80;
   - timing, `--knots2d 6,16`, L 10, order 3, 16 coils: box2_{box,ball,ball80}, box1_{box,ball,ball80},
     box1_fov_box, box1p_{ball80,box}.
   - Fill the gain table in §9.2 / §9.3 from `speedup` and `vs_ref`.
3. `out/drivers/{spline,p2np}_{off,box}.log`:
   - splinerecon, held-out curves and lumen values;
   - recon_p2np, per-iteration metrics.
   - Compare off vs box: equal to ≤ 1e-4 expected (up to the W 10 NUFFT error).

**Resume:**
```
cd /home/turbotage/Documents/GitHub/HastyCompute
systemctl --user status rc-gpu-queue          # or re-launch (resumes):
systemd-run --user --unit=rc-gpu-queue --collect -p MemoryMax=38G -p MemorySwapMax=0 --working-directory=$PWD \
  research/experiments/sample_compress/run_queue.sh /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
# rebuild the snapshot after code changes:
#   (cd rustlib && cargo build --release -j 4 -p hasty-bench --bin rcompress_bench --bin splinerecon --bin recon_p2np \
#    && cargo test --release -j 4 -p hasty-recon --test rcompress --no-run \
#    && cp target/release/{rcompress_bench,splinerecon,recon_p2np} target/rcompress_snap/ \
#    && cp target/release/deps/rcompress-<hash> target/rcompress_snap/rcompress_test)
```

**Remaining after the queue:**
- fill the GPU table (pass speed-ups per grid);
- target-config projection with §22.6's unit costs. Expected, not measured: box 160 at 1 mm on the 1 mm protocol,
  32 M samples per encode, 159 s → ≈ 121 s per pass with box (1.41×), ≈ 91 s with ball r 80 masked (2.06×);
- decide whether box designs need host streaming for the 2 mm box at full scale.
