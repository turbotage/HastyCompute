# A spatial wavelet prior next to the learned denoiser (2026-11-09)

Motivation: the 1 mm design undersamples each frame by about 19× (20 cardiac × 2 respiratory frames from a 10-min
scan). Spatial undersampling with an incoherent trajectory is the compressed-sensing regime, and its theory is better
understood than temporal-only priors. A spatial sparsity term per frame is the classical complement to the learned,
spatio-temporal denoiser.

## 1. Where to apply it

| domain | pro | con | literature |
|---|---|---|---|
| complex encode images, per frame | convex, standard CS (L1-SPIRiT-type) | encodes differ in phase by k_e·v, and smooth background phase reduces the sparsity of complex images; magnitude shrinkage near the noise floor biases phase differences, hence velocity, in low-signal vessels | Zibetti & De Pierro; DT-CWT separate priors |
| magnitude + phase separately | sparsity where the image is sparse (magnitude); smoothness for phase | non-convex; needs a phase model | Zhao, Noll, Nielsen, Fessler TMI 2012; NAPALM (Haldar) |
| velocity field (velocity model: v explicit) | regularises the quantity we report; divergence-free wavelets encode incompressibility (strict ∇·v = 0 is wrong at walls and in partial-volume voxels, so use a soft split) | needs the velocity model, a vector wavelet | Ong et al. MRM 2015 (DFW + SureShrink + cycle spinning); ISMRM 2014 (DFW inside CS, phase cycle spinning) |
| z of the velocity model (common complex image) | holds all common phase, so encode phase differences are untouched by the prior | — | follows from our parametrisation |

Recommendation:
- Complex-encode model (now): wavelet shrinkage per frame × encode, the phase kept per coefficient, thresholds scaled
  by the MAD noise level, mixed with the learned denoiser.
- Velocity model (target): wavelet sparsity on z (complex, per node) and divergence-free wavelets on v (per node), with
  separate thresholds for the divergence-free and the non-divergence-free coefficients.

## 2. How it combines with the learned denoiser in P²nP

The P²nP step is x ← D(x − α P ∇f). Options for two priors:
- **Convex combination** D_β = (1 − β) D_learned + β W (implemented, `--wavelet-mix β`). Averaged non-expansive maps
  stay averaged, so the fixed-point convergence argument of P²nP carries over. The wavelet acts as a safety net out of
  the training distribution (real anatomy, noise and artifacts the phantoms lack).
- Composition D ∘ W or W ∘ D is also non-expansive, but its fixed point mixes the priors in an uncontrolled way. Not
  used.
- Splitting (Davis–Yin): the wavelet ℓ1 term gets its own proximal step and weight. It is exact for a hybrid objective
  but costs a third operator per iteration. Later, if the mix proves useful.
- Scale: the learned denoiser is normalization-equivariant, D(μx) = μ D(x). The wavelet threshold is t × the MAD noise
  level per image, hence equivariant too. Mixing keeps the property.
- Diagnostic: |D_learned(x) − W(x)| maps where the learned prior disagrees with plain sparsity, a hallucination
  indicator on real data.

## 3. Implemented (`hasty_recon::wavelet`, recon_p2np)

- 3D Daubechies-4, `levels` levels, reflection padding, cycle spinning per iteration.
- Complex soft-threshold by magnitude; τ = t · σ̂, with σ̂ from the finest diagonal subband (median / 1.1774 for a
  complex Gaussian); the approximation band is kept.
- Tests: perfect reconstruction, scale equivariance W(μx) = μW(x), denoising of a smooth image.
- `recon_p2np --prior wavelet --wavelet THR[:LEVELS]` (the wavelet alone; κ mixes as for LLR), and
  `--learned … --wavelet-mix β`.
- 2026-11-09: on the shared engine `hasty_recon::sepwt` (f32, AVX2/FMA, all threads per image): 6.5× faster on a
  stack, 54× on a single image than the first f64 version. Divergence-free wavelets on the velocity (complex model:
  decode → shrink → re-encode; velocity model: v nodes) and the static-tissue lock: research/static_lock.md.

## 4. Experiments (in order)

1. 2 mm scan, complex model, 2 respiratory bins: LLR vs wavelet t ∈ {0.7, 1.0, 1.5} vs none. Velocity RMSE in the
   lumen, flow rate through planes, wall bias, wrapped fraction.
2. 1 mm development scan (R ≈ 19 per frame): the same, plus the learned prior (7-point data) with β ∈ {0, 0.2, 0.4}.
3. Velocity model: z wavelets + divergence-free wavelets on v (DFW implemented: `--dfw sure`, research/static_lock.md),
   against the complex model.

## Sources

- [Ong et al., Robust 4D flow denoising using divergence-free wavelet transform, MRM 2015](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.25176)
- [Ong et al., Compressed sensing 4D flow reconstruction using divergence-free wavelet transform, ISMRM 2014](https://archive.ismrm.org/2014/0326.html)
- [Ong et al., Improved visualization and quantification of 4D flow data using divergence-free wavelets (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3559289/)
- [Zhao, Noll, Nielsen, Fessler, Separate magnitude and phase regularization via compressed sensing, TMI 2012](https://pubmed.ncbi.nlm.nih.gov/22552571/)
- [Zibetti & De Pierro, Improving compressive sensing in MRI with separate magnitude and phase priors](https://link.springer.com/article/10.1007/s11045-016-0383-6)
- [MRI reconstruction with separate magnitude and phase priors based on the dual-tree complex wavelet transform (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9076340/)
- [Haldar, NAPALM](https://mr.usc.edu/documents/0141.pdf)
- [Divergence-free phase contrast MRI (EPFL)](https://infoscience.epfl.ch/server/api/core/bitstreams/507ce4b0-2281-4012-8ede-50e53804970a/content)
- [Magnitude-difference regularization for phase-contrast flow (PubMed)](https://pubmed.ncbi.nlm.nih.gov/31805336/)
- [Ahmad et al., Plug-and-play methods for MRI, IEEE SPM 2020](https://engineering.purdue.edu/~bouman/publications/orig-pdf/SPM-2020.pdf)
- [Hong, Xu, Hu, Fessler, Provable preconditioned plug-and-play for CS-MRI](https://arxiv.org/html/2405.03854)
- [Terris et al., Equivariant plug-and-play image reconstruction, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Terris_Equivariant_Plug-and-Play_Image_Reconstruction_CVPR_2024_paper.pdf)
- [A complex quasi-Newton proximal method for CS-MRI (wavelet vs deep PnP comparison)](https://arxiv.org/pdf/2303.02586)
