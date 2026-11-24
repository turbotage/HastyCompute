# Coil-sensitivity estimation for the 4D-flow pipeline (2026-11-23)

Question: why the ESPIRiT maps of `recon_p2np --smaps espirit` sit so far from the true maps (logged 1 − |⟨ŝ, s⟩| =
0.326 on ph0001, sim_flow_10min, dev grid 128×72×60), what the literature does, and which method our pipeline should use.
The user added three requirements: maps resolved over the respiratory cycle, image signal kept out of the maps, and
implementation started behind switches.

All numbers come from CPU scripts in `research/experiments/smaps/`. The truth is used only as a diagnostic yardstick and
to make the inverse-crime oracle data, never inside a method.

## 0. Summary

- **Most of the 0.326 is not map error.** The metric averages over 552 950 of the 552 960 grid voxels, which is the whole
  FOV, because the true maps are nonzero in air. Every voxel ESPIRiT crops counts as error 1. Those cropped voxels make
  up 0.246 of the 0.326. Over the voxels that do have a map, the mean is 0.106, and that is dominated by air near the
  FOV edge.
  - On the run's saved slice (z = 20, truth-tissue regions), the mean error is 0.0051 in tissue and 0.0107 deep in the
    body.
  - In the lungs (low signal), 27 % of voxels have **no map at all**, so the image there is forced to zero.
- **Our ESPIRiT has a floor of 1–4·10⁻³ in tissue.** This was measured with inverse-crime oracles: exact Cartesian
  calibration of s·ρ, using our settings (24³ region, kernel 6, crop 0.8).
  - Three things barely matter: the ratemap phase and decay over the out-half calibration times (1.61–1.94 ms), a
    larger calibration region (32³), and the z faces of the FOV.
  - **Mixing the return half into the calibration matters a lot.** It adds samples at 9.2–9.56 ms, as every pre-2026-11-23
    run did, because the calibration used every ADC sample in the box. In the oracle this multiplies tissue error by
    10–40 and lung error by about 30.
  - The other agent's `--calib-adc out` (the default since 2026-11-23) removes that mixing. It is not measured yet.
- **Recommended method, ranked by accuracy per compute:**
  1. ESPIRiT on out-half calibration (done). Add a smooth B-spline refit that fills the cropped holes and removes
     sub-knot structure (`--smaps-fit`, implemented).
  2. One map set per respiratory bin with a common phase gauge (`--smaps-resp`, implemented). Later, a linear-in-bellows
     parametrisation of the bins.
  3. A few Gauss–Newton steps on the B-spline coefficients against the pooled per-bin images. This is cheap alternating
     minimisation, about 10–20 pooled-gradient equivalents.
  4. Full NLINV/ENLIVE-style joint estimation only if step 3 still leaves a measurable held-out gap. It costs about 100×
     more.
- **The later ADC samples should not go into map calibration by averaging.** Each echo can contribute its own
  calibration *Gram* (a sum of Grams, never a sum of k-space). Their best uses are:
  - ratemap refinement: an 8 ms lever arm between the two k-centre passes;
  - a free model-consistency check: predict the return half from the out-half reconstruction;
  - bulk-motion and respiratory navigators.
  None of this touches the out-half gradient.

## 1. Our ESPIRiT use: code facts and defects

Code: `rustlib/crates/hasty-recon/src/espirit.rs`, called from `rustlib/crates/hasty-bench/src/bin/recon_p2np.rs`
(calibration block "calibration region …", ESPIRiT block "coil maps: ESPIRiT …").

| item | what the code does |
|---|---|
| calibration data | every set of encode 0, all 80 frames (4 resp × 20 cardiac), whitened, all 32 physical coils; samples with \|k_a\| < 12/FOV_a (box of 24 points per axis on the reconstruction FOV) |
| ADC samples | before 2026-11-23: **every ADC sample in the box**, i.e. the out half (τ = 1.61–1.94 ms) and the return half (τ = 9.22–9.56 ms) about 50/50 (1 494 199 samples). Now `--calib-adc out`: out half only |
| non-Cartesian → calibration | `calib_from_noncart`: CG on ‖Fρ_c − y_c‖² + 10⁻³·m‖ρ_c‖² on a 24³ grid of the same FOV, **30 iterations, no density weighting**, then a centred FFT. A converged least-squares fit is unbiased; a truncated one is a common k-space filter (§1.3) |
| compression | the calibration is compressed with the solve's coil-compression rows (31 of 32 virtual coils) |
| ESPIRiT | kernel 6³, σ > 0.02 σ_max (≈ 580–640 kernels), W(r) on a coarse half grid, trilinear to the output, unit norm Σ\|s\|² = 1, phase aligned to a virtual reference coil |
| crop | maps = 0 where the interpolated top eigenvalue < 0.8 |
| afterwards | maps zeroed outside the dilated support (`--coil-mask` auto = support for estimated maps) |
| metric | 1 − \|⟨ŝ, s/‖s‖⟩\| averaged over voxels with true-map energy > 0.05·median (552 950 voxels = the whole FOV), zero maps included |

### 1.1 What the 0.326 is made of (`region_error.py`, `compare_run.py`)

On the slice saved by the run (`precond_compare/zmshr1/smaps_axial_z20`, full-ADC calibration):

| region (z = 20; truth-derived, diagnostic) | voxels | zero map | mean 1 − \|⟨ŝ,s⟩\| over nonzero |
|---|---|---|---|
| tissue | 4621 | 3 % | 0.0051 (p90 0.011) |
| tissue ≤ 20 mm from the skin | 1535 | 0 % | 0.0036 |
| tissue > 50 mm deep | 1185 | 13 % | 0.0107 |
| lungs / low-signal holes | 1448 | **27 %** | 0.0187 |
| air (outside the dilated support) | 870 | 100 % | — |

- Whole volume: zero maps contribute 135 993/552 950 = 0.246 of the logged 0.326.
- The metric is phase-invariant per voxel, so phase conventions do not inflate it. It does count air and cropped voxels
  as total failures.
- `recon_p2np` now also prints the error over the object support. In the new line, zero maps count as 1, and a second
  number is the mean over the nonzero maps.

### 1.2 Oracle controls: one factor at a time (`oracle.py`, `run_oracles.sh`, `results/oracle_*.json`)

The calibration is the exact centred-FFT crop of s_c·ρ, where s are the true whitened and compressed maps and ρ is the
truth PD·e^{−R·1.6 ms}. Regions are recomputed per slice (tissue = signal > 5 % of the median; lungs = filled body minus
tissue). The table gives mean 1 − |⟨ŝ,s⟩| over voxels that keep a map (eig > 0.8) at z = 20; values at z = 2, 40 and 57
are in the json files.

| arm | factor | tissue | skin ≤ 20 mm | deep > 50 mm | lungs | lungs cropped |
|---|---|---|---|---|---|---|
| A0 | our settings, exact calibration | 0.0015 | 0.0013 | 0.0037 | 0.0071 | 6 % |
| D_out | + ratemap (field + decay) over the out-half calibration times | 0.0015 | 0.0013 | 0.0037 | 0.0071 | 6 % |
| D_mix | + return half mixed in, count-weighted per \|k\| (the pre-11-23 calibration) | **0.060** | 0.014 | **0.19** | **0.30** | 1 % |
| C32 | 32³ calibration instead of 24³ | 0.0017 | — | — | — | — |

- Over all four slices, A0 tissue is 1.1–3.7·10⁻³.
- Without the crop, cropped voxels are about 0.01–0.03 (the "no-crop" column in the json files).
- **Off-resonance during calibration does not matter.** The field over the object is 62 Hz median and 304 Hz at the 99th
  percentile; over 0.33 ms that is ≤ 0.6 rad, and D_out = A0.
- **Echo mixing is catastrophic.** At the same k, the two halves see images that differ by e^{−z·7.6 ms}, and their
  share of the samples varies with |k|. Least squares then fits a k-dependent image, which ESPIRiT reads as coil
  structure.
- The real run (0.0051 / 0.0107 / 0.0187) lies between A0 and D_mix. The crude mixing emulation (one count-weighted mix
  per |k|) overestimates the real damage.
- What is left after the out-half fix has to be measured on the GPU; see §7. Plausible residual causes, none measured:
  - noise: noise_rel 0.02 and encode 0 only;
  - the truncated calibration CG (§1.3);
  - the 0.8 crop.

### 1.3 Defects, ranked

1. **Echo mixing in the calibration.** Fixed (`--calib-adc out`, default) but not yet measured. Expected effect: from
   about 3–10× the floor down to the floor.
2. **The eig_min 0.8 crop inside the body.** Lungs and parts of the mediastinum get zero maps (27 % / 13 % at z = 20).
   Every image voxel there is then forced to zero, including pulmonary vessels near the hilum.
   - In the oracle, the uncropped eigenvector there is about 3–5× worse than in tissue.
   - Remedy: fill from the surroundings with a smooth fit (`--smaps-fit`) instead of zeroing.
3. **The reported metric** (whole FOV, zero = 1). Fixed: a support-region line is now printed and stored in
   `run.json → estimates.smaps_support`.
4. **The truncated calibration least squares** (30 CG iterations, no density weighting).
   - Over the 24³ box the sample density of the yarn trajectory varies strongly. CG then under-resolves the box edge, a
     smooth k-filter H(k) common to all coils. A non-constant H breaks the ESPIRiT window relation.
   - Not measured yet. The check is `used` samples vs CG residual, or a 100-iteration rerun.
   - On density compensation: **calibration gridding is a separate estimator from the reconstruction.** A density
     weighting, a k-space preconditioner (Ong et al. 2020) or simply more CG iterations all converge to the same
     least-squares calibration. That is acceptable: the project's no-DCW rule protects the reconstruction's data term
     and noise model, not a pre-processing estimate. Preconditioned CG is the cleanest, since its fixed point is
     unchanged.
5. **Encode 0 only** (1/7 of the data).
   - Calibration from all 7 encodes is consistent: the maps are common, and any per-voxel image phase (velocity) is the
     same for all coils.
   - Use a **sum of per-encode calibration Grams AᴴA**, not a sum of k-space. Summing k-space is also consistent but
     lets velocity phases cancel the signal in vessels.
   - Gain: about 7× calibration SNR. Cost: 7 calibration CGs (seconds each).
6. **Non-issues measured here:**
   - the ratemap over the out-half window (D_out);
   - calibration size (C32 does not improve);
   - FOV wrap at the z faces (the slices at z = 2 and z = 57 are no worse than interior ones, because in the sim the
     object ends at the FOV).
   - Real scans will have signal outside the z-FOV. Non-Cartesian data then allow a calibration grid on 1.25–1.5× FOV
     (finer k spacing, same k extent). This is a later item.

### 1.4 How accurate the maps must be

- With known images, the fraction of a voxel's coil signal not explained by ŝ is 1 − |⟨ŝ,s⟩|² ≈ 2(1 − |⟨ŝ,s⟩|).
- The A0 floor of about 2·10⁻³ therefore leaves about 4·10⁻³ of the energy, i.e. a 6 % amplitude misfit per voxel and
  coil. That is far above the 10⁻⁴ NUFFT budget and comparable to the per-frame noise.
- Map error is therefore a first-order term in the error budget, and ESPIRiT alone will not reach the truth-map
  reconstruction. A refinement against the data is justified: §5, method (b).

## 2. Literature, with links

**ESPIRiT.** Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. MRM 71:990–1001 (2014),
[doi:10.1002/mrm.24751](https://doi.org/10.1002/mrm.24751).
- Calibration-matrix null space → a pointwise eigenproblem. Eigenvalue ≈ 1 marks signal; several eigenvalue-1 maps
  appear when the object exceeds the FOV ("soft SENSE"). The crop threshold (0.8–0.95) exists for those cases.
- Non-Cartesian use, as in BART/SigPy practice: grid or least-squares invert a low-resolution centre (e.g. `nufft -i`
  onto 24³), then run `ecalib` as if Cartesian. Lustig & Pauly's SPIRiT (MRM 64:457, 2010,
  [doi:10.1002/mrm.22428](https://doi.org/10.1002/mrm.22428)) introduced non-Cartesian calibration from gridded
  centres.
- Automatic parameter choice (kernel, threshold, crop) by SURE: Iyer S, Ong F, Setsompop K, Doneva M, Lustig M. MRM
  84:3423–3437 (2020), [doi:10.1002/mrm.28386](https://doi.org/10.1002/mrm.28386).
- Absolute phase via virtual conjugate coils: Uecker M, Lustig M. MRM 77:1201–1207 (2017),
  [doi:10.1002/mrm.26191](https://doi.org/10.1002/mrm.26191).
- Review of non-Cartesian parallel imaging: Wright KL et al., JMRI 40:1022 (2014),
  [doi:10.1002/jmri.24521](https://doi.org/10.1002/jmri.24521).

**Walsh adaptive combine / Inati.**
- Walsh DO, Gmitro AF, Marcellin MW. MRM 43:682–690 (2000),
  [doi:10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G](https://doi.org/10.1002/(SICI)1522-2594(200005)43:5%3C682::AID-MRM10%3E3.0.CO;2-G).
  Per-voxel dominant eigenvector of the local coil-image correlation matrix. It is an image-domain cousin of ESPIRiT:
  low-resolution coil images → local covariance → eigenvector. The block size sets the trade-off between smoothness and
  image leakage.
- Inati SJ, Hansen MS, Kellman P. A solution to the phase problem in adaptive coil combination, ISMRM 2013 p. 2672, and
  a fast iterative version (power iterations plus smoothing), ISMRM 2014 p. 4407. These are the Gadgetron/ISMRMRD
  defaults.
- Cheap (seconds); the standard for time-averaged dynamic data (below).

**JSENSE.** Ying L, Sheng J. MRM 57:1196–1202 (2007), [doi:10.1002/mrm.21245](https://doi.org/10.1002/mrm.21245).
- Maps = a low-order polynomial times an initial estimate. Alternates image (CG-SENSE) and polynomial coefficients
  (Gauss–Newton on a few dozen unknowns).
- This is our method (b), with polynomials instead of B-splines.
- Extended to spiral and non-Cartesian data by the same group.

**NLINV.**
- Uecker M, Hohage T, Block KT, Frahm J. MRM 60:674–682 (2008),
  [doi:10.1002/mrm.21691](https://doi.org/10.1002/mrm.21691).
  - Joint (ρ, c) by iteratively regularised Gauss–Newton (IRGNM) on the bilinear model y = F(c·ρ).
  - Maps are parametrised in k-space with a Sobolev weight W = (1 + a|k|²)^{l/2} (a = 220, l = 32 in the paper), so
    their high frequencies are expensive. That is the device that keeps image content out of the maps.
  - The regularisation decreases geometrically over the Gauss–Newton steps.
  - After convergence: ρ ← ρ·‖c‖, c ← c/‖c‖ (the gauge fix).
- Real-time radial: Uecker M, Zhang S, Frahm J. MRM 63:1456–1462 (2010),
  [doi:10.1002/mrm.22453](https://doi.org/10.1002/mrm.22453).
  - Each frame is initialised from the previous one and regularised towards it (temporal coupling of maps and image).
  - Uecker M, Zhang S, Voit D, Karaus A, Merboldt KD, Frahm J. NMR Biomed 23:986 (2010),
    [doi:10.1002/nbm.1585](https://doi.org/10.1002/nbm.1585).
- Variational penalties in NLINV: Knoll F, Clason C, Bredies K, Uecker M, Stollberger R. MRM 67:34–41 (2012),
  [doi:10.1002/mrm.22964](https://doi.org/10.1002/mrm.22964).

**ENLIVE.** Holme HCM, Rosenzweig S, Ong F, Wilke RN, Lustig M, Uecker M. Sci Rep 9:3034 (2019),
[doi:10.1038/s41598-019-39888-7](https://doi.org/10.1038/s41598-019-39888-7) (code: github.com/mrirecon/enlive).
- NLINV with several (ρ_j, c_j) pairs, i.e. a low-rank relaxation. It reproduces ESPIRiT's multi-map robustness without
  a calibration region.
- Shown on radial cardiac FLASH.

**Calibrationless low-rank methods.**
- SAKE: Shin PJ et al. MRM 72:959–970 (2014), [doi:10.1002/mrm.24997](https://doi.org/10.1002/mrm.24997).
- LORAKS: Haldar JP. IEEE TMI 33:668–681 (2014), [doi:10.1109/TMI.2013.2293974](https://doi.org/10.1109/TMI.2013.2293974).
- PRUNO: Zhang J, Liu C, Moseley ME. MRM 66:1241–1253 (2011),
  [doi:10.1002/mrm.22899](https://doi.org/10.1002/mrm.22899).
- These complete the multi-coil block-Hankel matrix (low rank) instead of estimating maps.
- Cartesian-centric and heavy: an SVD per iteration of a matrix the size of ESPIRiT's.
- Not useful for us: we have a fully sampled centre (thousands of readouts pass k = 0), so calibration data is not
  scarce.

**Joint estimation with motion.**
- Bammer R, Aksoy M, Liu C. MRM 57:90–102 (2007), [doi:10.1002/mrm.21106](https://doi.org/10.1002/mrm.21106).
  Augmented SENSE with rigid motion; the maps stay in the scanner frame while the object moves.
- Cordero-Grande L et al. IEEE TCI 2:266–280 (2016), [doi:10.1109/TCI.2016.2557069](https://doi.org/10.1109/TCI.2016.2557069),
  and MRM 79:1365 (2018), [doi:10.1002/mrm.26796](https://doi.org/10.1002/mrm.26796).
  Aligned SENSE: motion estimated jointly; maps fixed (scanner frame) from a separate or pooled estimate.
- The consensus: coil maps are a property of the coil (scanner frame for table coils), modulated by the body. Body
  motion changes them smoothly, and anterior arrays move with the chest.

**Deep-learning sensitivity estimation.**
- E2E-VarNet: Sriram A, Zbontar J, Murrell T, et al. MICCAI 2020, [arXiv:2004.06688](https://arxiv.org/abs/2004.06688).
  A U-Net refines ESPIRiT-like maps from the ACS, trained end-to-end with the reconstruction.
- Deep J-Sense: Arvinte M, Vishwanath S, Tewfik AH, Tamir JI. MICCAI 2021,
  [arXiv:2103.02087](https://arxiv.org/abs/2103.02087).
  Unrolled alternating map and image updates (JSENSE with learned denoisers on both).
- Both are Cartesian 2D (fastMRI) and need training pairs. Of no interest for us until the denoiser pipeline has map
  augmentation.

**Maps from time-averaged dynamic data.**
- TSENSE: Kellman P, Epstein FH, McVeigh ER. MRM 45:846–852 (2001),
  [doi:10.1002/mrm.1113](https://doi.org/10.1002/mrm.1113).
  Interleaved undersampling; the time average is fully sampled and gives the maps.
- TGRAPPA: Breuer FA et al. MRM 53:981–985 (2005), [doi:10.1002/mrm.20430](https://doi.org/10.1002/mrm.20430).
  The same idea for GRAPPA kernels.
- GRASP: Feng L et al. MRM 72:707–717 (2014), [doi:10.1002/mrm.24980](https://doi.org/10.1002/mrm.24980).
  Maps from the average of all radial spokes (adaptive combine).
- XD-GRASP: Feng L et al. MRM 75:775–788 (2016), [doi:10.1002/mrm.25665](https://doi.org/10.1002/mrm.25665).
  Maps from the time average of all respiratory and cardiac states (pooled).
- The free-running 5D-flow papers follow the same practice: Ma LE et al., Radiol Cardiothorac Imaging 2:e200219 (2020),
  [doi:10.1148/ryct.2020200219](https://doi.org/10.1148/ryct.2020200219).
- Pooling is the norm and is consistent when the maps do not change. The averaged object blurs, but x_c = s_c·ρ̄ still
  holds. It is not consistent when the maps move with breathing (§4).

**Smooth parametrisations.**
- Polynomial: JSENSE; the original SENSE (Pruessmann et al., MRM 42:952, 1999) fitted local polynomials to coil/body
  ratios.
- Sobolev: NLINV.
- Regularised smooth fit with a finite-difference (roughness) penalty: Allison MJ, Ramani S, Fessler JA. IEEE TMI
  32:556–564 (2013), [doi:10.1109/TMI.2012.2229711](https://doi.org/10.1109/TMI.2012.2229711).
  ADMM for the quadratic smoothing problem, used to fill and denoise map estimates. This is exactly our `--smaps-fit`
  problem, with B-splines in place of a voxel grid with a difference penalty.
- Keeling SL, Bammer R. A variational approach to MR coil sensitivity estimation. Appl Math Comput 158:359 (2004).

**Cost, in our units.** A pooled gradient (all sets) is about 14 s at dev scale; a per-frame gradient over all 560 sets
is 12–17 s.

| method | cost |
|---|---|
| ESPIRiT | 174 s CPU per call today (eigensolver 158 s: randomized subspace iteration doubling 256 → 1024); ~60 s with a start block, implemented as `EspiritOpts.block0` |
| Walsh / Inati | seconds |
| JSENSE-type refinement (b) | a few pooled gradients per Gauss–Newton step |
| NLINV on our 3D non-Cartesian data | each IRGNM step has an inner CG of 10–30 normal-operator applications on the joint (ρ, c) unknowns: ≈ 10 steps × 20 = 200 pooled-gradient equivalents ≈ 45 min at dev scale for pooled images, ×4 per bin; far more if per frame |
| ENLIVE | ×(number of map sets) |
| SAKE / LORAKS | a large SVD per iteration; impractical in 3D |

## 3. Keeping image signal out of the maps

Where image content gets in, and the remedy for each:

1. **ESPIRiT's eigen-structure.**
   - With exact, consistent calibration, the eigenvalue-1 eigenvector of W(r) is s(r) regardless of ρ. Content enters
     only through:
     - (i) a finite calibration region: kernels estimated from a low-resolution ρ, so the null space is "learnt" on
       blurred content;
     - (ii) low-eigenvalue voxels (λ < 1: no signal, so the eigenvector is a mix of neighbourhood and noise);
     - (iii) inconsistent calibration data (echo mixing, motion within the pooled data, a truncated LS filter).
   - The unit-norm gauge Σ|s|² = 1 prevents magnitude leakage. Only the *relative* coil profile can carry content.
   - Remedies:
     - (iii) consistent calibration (§1.3);
     - (ii) weight by λ and fill (B-spline fit);
     - (i) a larger calibration region does not help here (C32).
2. **Smoothness / low-dimensional bases.**
   - Represent maps in a basis that cannot carry structure finer than the coil physics. On ph0001 (Biot–Savart loops ≥
     12 mm off the skin), `bspline_rep.py` gives the true-map representation error 1 − |⟨B a, s⟩| per knot spacing:

     | h (mm) | object mean | within 20 mm of the skin | deep > 50 mm | coefficients per coil |
     |---|---|---|---|---|
     | 16 | 1.0·10⁻⁴ | 2.9·10⁻⁴ | 2.6·10⁻⁶ | 13 230 |
     | 24 | 4.2·10⁻⁴ | 1.1·10⁻³ | 2.5·10⁻⁵ | 4 875 |
     | 32 | 1.0·10⁻³ | 2.5·10⁻³ | 1.3·10⁻⁴ | 2 508 |
     | 48 | 3.9·10⁻³ | 8.9·10⁻³ | 7.6·10⁻⁴ | 1 008 |

   - **16 mm knots** keep the basis error 10× below the ESPIRiT floor near the skin. 32 mm is already as bad as
     ESPIRiT there.
   - A spatially varying knot density (finer near the coils) could halve the coefficient count later.
3. **Normalisation and the scaling ambiguity.**
   - S x = (S g)(x/g) for any nonzero g(r). Fix g once and identically for every bin and iteration:
     - |g| by Σ_c|s_c|² = 1 (ESPIRiT; `smapfit::normalise_rss`). The image then carries the RSS coil profile, a
       smooth shading.
     - arg g by alignment to a reference map set: `smapfit::align_phase` makes ⟨s_ref, s⟩ real and positive.
   - With per-bin maps, a free gauge would put respiratory-correlated phase and shading into the images and hurt
     temporal regularisation across bins. A common gauge prevents that.
   - Velocity phase cannot enter the maps as long as all 7 encodes share the maps: any map phase cancels in the encode
     differences. Keep it that way: never estimate maps per encode.
4. **NLINV / ENLIVE regularisers.**
   - A Sobolev weight on the map k-space ((1 + a|k|²)^{l/2}) is the continuous analogue of our B-spline basis plus
     second-difference penalty.
   - The IRGNM regularisation α_n → 0 is decreased with care: too small and image edges leak into c; too large and the
     maps are over-smoothed (Uecker 2008 §Theory).
   - In our Gauss–Newton (b), the same role is played by the knot spacing (hard) and the penalty λ (soft); do not drive
     λ → 0.
5. **Time-averaged data within each respiratory bin.**
   - Average over the cardiac frames, all encodes, out half only. Cardiac-resolved maps are not recommended (they would
     absorb blood inflow and wall motion).
6. **Constraints across bins.**
   - Maps move with the anatomy as a smooth deformation of a smooth field, plus the coil's own displacement:
     posterior table coils static, anterior array on the chest.
   - Practical parametrisation: a_b = a_0 + r_b·a_1, where r_b is the bin's mean bellows amplitude. That is linear in
     the respiratory state, two coefficient sets instead of four, and the bins regularise each other.
   - The more physical option is a_b = a_0 ∘ φ_b^{-1} with φ_b the respiratory motion field (when moco exists). It is
     not needed for 4 bins.
7. **Tests for leakage** (all from data except the diagnostic truth comparison):
   - **Edge correlation:** the normalised correlation between |δS| (estimate minus its smooth fit, or per-bin minus
     pooled) and |∇ρ̄|, the gradient magnitude of the pooled image, over the support. Leakage shows as positive
     correlation.
   - **Dynamic correlation:** maps from two cardiac halves of one resp bin (systole vs diastole sets) must agree. Their
     difference correlated with |ρ_sys − ρ_dia| flags dynamic leakage. `--smaps-resp` machinery restricted to cardiac
     groups; not implemented.
   - **Cross-echo agreement:** maps calibrated from the return half alone must equal those from the out half. The
     images differ by e^{−z·7.6 ms} (different content, same maps), so disagreement localises leakage.
   - **Static-map sim:** in ph0001 the true maps are static, so any per-bin difference printed by `--smaps-resp`
     ("vs the pooled maps") is pure estimation error or leakage.
   - **Held-out k-space (acceptance):**
     - leakage overfits, so the in-sample residual falls while the held-out residual (`--holdout-res`) does not;
     - compare arms at equal iterations;
     - the truth-map arm is the ceiling.

## 4. Respiratory-resolved maps

- The user requires maps per respiratory bin or a smooth respiratory parametrisation. `--smaps-resp` implements the
  per-bin version:
  - ESPIRiT from the encode-0 out-half samples of that bin (all its cardiac frames);
  - phase gauge aligned to the pooled maps;
  - the operator swaps bin r's maps in for every set of bin r (`setop::PerSetMaps`), also in the held-out residual.
- The warm start, preconditioners and support keep the pooled maps. That is fine as a start; the fixed point is
  unchanged.
- **The sim cannot show the gain.** `simphantom` uses static Biot–Savart loops (`hasty_phantom::coils`, one
  smaps_true). On ph0001 the expected result is no gain and a small loss from 4× less calibration data per bin; that
  loss is what the validation measures.
- To test the real benefit, the simulator must move the anterior loops with the chest wall per respiratory state. It
  already has the respiratory state per readout; this needs only a loop-position offset along AP from the resp signal.
  Proposed as a separate phantom work package.
- Scale: one map set at 1 mm is about 8 GB (31 coils). Four bins of dense maps do not fit next to the solve. The
  B-spline coefficients do (13 k × 31 × 8 B ≈ 3 MB per bin). The production path must therefore expand coefficients on
  the device per set, or per bin run, instead of storing dense per-bin maps. The current `PerSetMaps` holds dense maps;
  fine at dev scale (4 × 0.14 GB).

## 5. Methods for our pipeline, ranked by expected accuracy per compute

**(a) ESPIRiT on consistent pooled calibration, per respiratory bin, plus a smooth refit.**
- Calibration from the out half only (done).
- Sum of per-encode Grams (7× data; to implement).
- Crop only outside the support; low-eigenvalue voxels filled by the weighted B-spline fit with E0 = 0.5 (done).
- Ratemap phase removal is unnecessary over 0.33 ms (D_out = A0).
- Expected: tissue error at the A0 floor (≈ 2·10⁻³) and no zero-map voxels inside the body.
- Cost: ESPIRiT 60–170 s CPU per bin; the fit about 10–30 s CPU per map set.
- No new GPU work beyond the calibration CGs.

**(b) Gauss–Newton refinement of the B-spline coefficients against the pooled per-bin images.**
- Model per bin b: y_t = F_t(S(a_b)·x̄_{b,e}), with x̄_{b,e} the pooled image of encode e in bin b (or the current
  per-frame images).
- The map gradient is ∇_{a_c} = Bᵀ[Σ_t conj(x_t) ⊙ F_tᴴ r_{t,c}], a *per-coil* adjoint image before the coil
  combination. The engine computes those internally; it needs one new entry point that returns the per-coil adjoint
  images, or a "maps = unknowns" mode of the existing kernels.
- Linear in a for fixed x, so each Gauss–Newton step is a linear least-squares problem in about 13 k × 31 unknowns per
  bin. Solve it with CG (3–10 inner steps, each one forward plus one per-coil adjoint over the bin's sets).
- Then refit x by a few pooled CG steps. Alternate 2–3 times.
- Gauge: renormalise and phase-align after each map step. Penalty: the B-spline second difference (λ ~ 10⁻³, fixed).
- Expected accuracy: JSENSE/NLINV literature reports clear gains over calibration-only maps at higher acceleration, and
  JSENSE ≈ NLINV when the polynomial order is adequate. Here the target is the A0 floor and below: an oracle test with
  known x̄ is cheap to add.
- Cost per bin: 1/4 of the pooled set, so ≈ 3.5 s per operator pass. 3 outer × (5 inner + 5 image) passes ≈ 30 passes
  ≈ 2 min for all bins, about 10 pooled-gradient equivalents.
- Interaction with the estimator: maps from data only; acceptance by held-out k-space at equal solve iterations.
- Risk: the bilinear ambiguity. The fixed gauge and the knot spacing handle it.

**(c) Full NLINV-style joint estimation of per-frame images and maps.**
- Expected gain over (b): small when the maps are smooth and the images are well determined (true here: 10 min of data,
  pooled per bin).
- Cost: ≈ 200 pooled-gradient equivalents for pooled images, or per-frame gradients if the images are per frame. That
  is 45 min+ at dev scale and hours at 1 mm.
- Only if (b) leaves a held-out gap to the truth-map arm.

## 6. The unused later ADC samples

The return half (k-centre again at ≈ 9.58 ms), the spoiler part and the encoder ADC part:

| use | model needed | consistent with an out-half gradient? |
|---|---|---|
| **map calibration by averaging into the same k-space** | — | **No.** The oracle shows a 10–40× worse map: the images differ by e^{−z·7.6 ms} and least squares mixes them k-dependently |
| map calibration as a **second calibration Gram** (G = G_out + G_ret, separate grids) | none beyond a separate least-squares calibration per echo; the null-space relation holds per image | yes (calibration only). Gain: more calibration SNR. Low priority: the return half has lower signal (ratemap decay; lungs and fat) and SNR is not the bottleneck (A0 floor without noise) |
| **cross-echo map check (leakage test)** | as above, maps compared not combined | yes |
| **ratemap / B0 refinement (two k-centre passes ≈ dual echo)** | low-resolution images at τ₁ ≈ 1.6 ms and τ₂ ≈ 9.6 ms, ratio e^{−z(τ₂−τ₁)}, z = decay + iΔω; plus the return-half trajectory with GIRF, concomitant fields and eddy terms. Δτ = 8 ms wraps at ±62.5 Hz; the field reaches 304 Hz (p99), so unwrap with the out-half segment model or the B0 scan as prior. Better: fit z by Gauss–Newton through the existing segmented operator on the pooled low-resolution data of both halves | yes: a separate estimation step whose output (ratemap) the out-half operator consumes |
| **motion / respiratory navigators** (return centre pass, encoder ADC) | k ≈ 0 samples per TR are the low-resolution projections of the current image; rigid shifts appear as linear phase across the centre samples; the static ratemap phase at τ₂ cancels in TR-to-TR differences | yes (`motion_navigators.md`; the sim lacks the nav ADC and a pose schedule) |
| **data-consistency check** | predict the return half from the out-half reconstruction with the full model (ratemap over 8 ms, GIRF, concomitant fields) and report its residual next to the held-out one | yes: a physics-model test that out-half held-out k-space cannot provide (it shares the out-half model errors) |
| **ratemap decay estimation** | the magnitude ratio of the two centre-pass images per low-resolution voxel; the field term as above | yes |

## 7. Implemented (2026-11-23, behind switches, defaults unchanged)

| change | where |
|---|---|
| new module `hasty_recon::smapfit` | tensor cubic B-spline basis (separable eval/adjoint), weighted LS fit with second-difference penalty (CG per coil), `normalise_rss`, `align_phase`, `eig_weights`, `map_disagreement`; 3 CPU unit tests |
| `EspiritOpts.block0` | start block of the subspace iteration (None = old behaviour) |
| `SetOp::swap_maps` (default false) | implemented for `SegSets` (refused with a surrogate engine or a coil subset) |
| `setop::PerSetMaps` | wrapper giving per-set coil maps: lazy swap, group calls split by map |
| `recon_p2np --smaps-fit bspline:H_MM[:LAMBDA[:E0]]` | ESPIRiT run with eig_min = E0 (default 0.5), then the weighted fit (weights ((λ−E0)/(1−E0))² on the dilated support), normalisation on the map mask; per-bin maps fitted too |
| `recon_p2np --smaps-resp` | per-bin calibration (encode 0, out half), ESPIRiT per bin (block0 from the pooled count), phase-aligned to the pooled maps, per-bin error and per-bin-vs-pooled printed; `PerSetMaps` around the segment operator; held-out residual swaps per set; slice `smaps_resp_axial_z*.npy`; ESPIRiT eigenvalue slice saved |
| reporting | map error over the support printed for every estimated-map run (`estimates.smaps_support` in run.json) |

**Build and test status:**
- **Not compiled.** The cargo commands (`cargo build --target-dir target/alt …`, even `cargo check`) were refused by
  the session's permission classifier.
- Code review was done by reading. One borrow problem was found and fixed: `flag()` must come before `take` is defined.
- The fit algorithm was checked through a numpy port of `fit_maps` on the Rust unit test's case
  (`smapfit_port.py`), and its assertions pass:
  - error outside the hole 1.3·10⁻⁷, inside 4.6·10⁻⁷;
  - the 3 % texture's squared error falls from 8.7·10⁻⁴ to 3.8·10⁻⁷.
- To build and test:

  ```
  cd rustlib && nice cargo build --release -j 4 --target-dir target/alt -p hasty-bench --bin recon_p2np
  cd rustlib && nice cargo test --release -j 4 --target-dir target/alt -p hasty-recon --lib smapfit espirit
  ```

**GPU validation** (not started; after the campaign):
- Command: `systemd-run --user --unit=smaps_val --collect bash research/experiments/smaps/run_validation.sh all`
- Analysis: `python research/experiments/smaps/analyze_validation.py`
- Arms, all with the dev config, out-half calibration, own warm start, zmshr1, 40 iterations, held-out 25:

| arm | change vs `esp` | question (pre-registered) | decision |
|---|---|---|---|
| smoke | resp + fit16, 2 iterations | do the paths run (per-bin ESPIRiT, fit, PerSetMaps, held-out swap)? | stop if not exit 0 |
| esp | — (out-half ESPIRiT) | how much did `--calib-adc out` remove? Support error vs the A0 floor (≈ 2·10⁻³) | support error > 5·10⁻³ → measure calibration CG convergence next (defect 4) |
| fit16 | `--smaps-fit bspline:16` | holes filled (zero count inside the support → 0) and held-out ≤ esp? | adopt for estimated maps if held-out ≤ esp at equal iterations |
| resp | `--smaps-resp` | cost of 4× less calibration data per bin on a static-map sim (expect ≤ 0.5 % held-out loss) | keep for real data if the loss is ≤ 0.5 %; else go straight to the linear-in-bellows parametrisation |
| respfit16 | both | does the fit remove the per-bin noise (per-bin vs pooled disagreement drops)? | default candidate for real data |
| truth | `--smaps truth` | ceiling for the held-out residual | gap esp → truth sizes the need for method (b) |

## 8. Plan, ranked

1. **Measure the out-half calibration plus fit** (arms esp, fit16, truth).
   - Expected: support error from ≈ 5·10⁻³ to ≈ 2·10⁻³; zero-map voxels in the body 3 % → 0; held-out residual closer
     to truth.
   - Cost: about 2 GPU hours at dev scale.
   - Measured by: the support-error line and the held-out residual at equal iterations.
2. **Gram-sum calibration over the 7 encodes** (and optionally the return half as a separate Gram), plus a converged
   calibration LS (preconditioned CG or more iterations).
   - Expected: removes noise and truncation from the calibration.
   - Cost: 7 small calibration CGs (seconds); ESPIRiT unchanged.
   - Measured by: support error and held-out residual vs arm 1. An oracle-noise control first (add noise_rel 0.02 to
     `oracle.py`) tells whether noise is limiting at all.
3. **Gauss–Newton map refinement (b) per respiratory bin on B-spline coefficients, with a common gauge.**
   - Needs a per-coil adjoint entry point in `SegNufft`.
   - Expected: below the ESPIRiT floor, towards the truth-map residual.
   - Cost: about 10 pooled-gradient equivalents.
   - Measured by: held-out residual (acceptance), edge/dynamic leakage correlations, truth error (diagnostic).
4. **Respiratory parametrisation a_b = a_0 + r_b·a_1, with on-device coefficient expansion per set**, so per-bin maps
   scale to 1 mm.
   - Requires the sim to move the anterior coils with the chest to show a gain.
5. **Return-half uses:** a model-consistency residual (cheap; reuses the held-out machinery on the return-half samples),
   then two-pass ratemap refinement, then navigators.

## Files

- `research/experiments/smaps/`:
  - `common.py`: true maps on the recon grid (GNL, whitening, compression); cache in the session scratchpad
  - `region_error.py`, `fig_region.py`, `region_err_z20.png`: metric decomposition on the run's slice
  - `compare_run.py`: run slice on truth regions
  - `espirit_py.py`, `oracle.py`, `run_oracles.sh`, `results/oracle_*.json`: oracle controls
  - `bspline_rep.py`, `results/bspline_rep.txt`: basis representation error
  - `smapfit_port.py`: numpy check of the Rust fit
  - `run_validation.sh`, `analyze_validation.py`: GPU validation (not run)
