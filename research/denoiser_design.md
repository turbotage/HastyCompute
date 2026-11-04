# An equivariant complex-domain 4D-flow denoiser for P²nP

Status: research note (2026-11-03). Goal: the plug-and-play prior D inside P²nP-ISTA for time-resolved 4D/5D flow.
It acts on the **complex multi-encode image stack**, not on velocity maps: it must denoise magnitude and velocity
together, be usable inside a provably convergent preconditioned PnP iteration, and exploit the geometry of the
problem (translations, rotations/reflections, cardiac periodicity, receiver gain/phase, background phase).

## 0. Recommendation in one paragraph

*Revised in §10:* architectural (not canonicalised) scale equivariance, no complex shift, fixed 7-point
dual-VENC layout on encode-aligned grids, patch-based training with interior loss.

A bias-free, factorised 4-D (3-D space × cyclic cardiac time) U-Net whose convolutions are **equivariant to the
octahedral group O_h acting on the voxel grid and, through the induced signed permutation, on the velocity-encoding
axes**. It is wrapped in an exact **normalisation canonicaliser** (complex shift Δ, positive scale µ, optionally global
phase) so that P²nP's dynamic-preconditioner assumption holds by construction. It is fed **gauge-relative features**
(reference image plus reference-relative encode images), which removes the smooth background phase from the velocity
channels without any explicit unwrapping. It outputs clean complex encode images through a light physical
parameterisation (magnitude, background phase, velocity, intravoxel dephasing) plus a small free complex residual.
Training: complex-MSE MMSE denoising on CFD-based ground truth (RACLETTE first, own CFD later) with noise drawn both
white and shaped by our actual operator, a velocity loss in the lumen and a soft divergence penalty. Lipschitz
control via spectral normalisation of the linear layers. Inference from Rust through an exported plain-PyTorch graph
(escnn → `export()` → TorchScript / AOTInductor → tch-rs).

## 1. What P²nP requires from D

From Hong, Xu, Hu, Fessler, *Provable Preconditioned Plug-and-Play Approach for Compressed Sensing MRI Reconstruction*
(IEEE TCI 2024, [arXiv 2405.03854](https://arxiv.org/html/2405.03854v1)):

* **Assumption 1 (both cases):** D_σ: Cᴺ → Cᴺ is Lipschitz with constant 1 + ε. Convergence of the fixed-preconditioner
  iteration if (1+ε)·ρ(I − αPAᴴA) < 1; not non-expansiveness.
* **Dynamic preconditioner (ZMSHR1, our case):** additionally a **normalisation-equivariant** denoiser,
  D(µx + Δ𝟏) = µD(x) + Δ𝟏 for real µ > 0 and complex Δ, plus a bounded gradient. Reason: the effective noise level of
  the proximal step changes with the preconditioner scale η_k every iteration; for an MMSE denoiser
  (1/µ)D_σ(µx) = D_{σ/µ}(x), so normalisation equivariance removes the need to re-tune σ per iteration.
* They built it with the construction of Herbreteau, Moebel, Kervrann, *Normalization-Equivariant Neural Networks
  with Application to Image Denoising* (NeurIPS 2023, [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/12143893d9d37c3569dda800b95cabd9-Paper-Conference.pdf),
  [code](https://github.com/sherbret/normalization_equivariant_nn/)): affine-constrained convolutions (weights summing
  to one) and channel-wise sort pooling instead of ReLU. Bias-free ReLU networks are already positively homogeneous
  (Mohan et al. 2020); the shift needs the affine constraint.

Consequences for us:
1. Exact normalisation equivariance is required, not approximate (augmentation is not enough).
2. Lipschitz control matters: spectral normalisation of linear layers, or a Jacobian-norm penalty during training.
3. The denoiser sees x_k − αP∇f(x_k): approximately Gaussian, but **coloured by P** and carrying residual aliasing.
   Training should include such noise (§5).

## 2. What the denoiser acts on, and its symmetries

The stack is x ∈ C^{N_vox × F × E}: F cardiac frames (later × respiratory bins), E velocity encodes (4 for referenced or
Hadamard four-point, 7 for RACLETTE's dual-VENC, 13 for icosahedral). Per voxel r and frame t the physics is

  x_e(r,t) ≈ m(r,t) · e^{iθ(r)} · exp(i k_eᵀv(r,t)) · exp(−½ k_eᵀ Σ(r,t) k_e) + n_e(r,t),

* m: magnitude (proton density, T1/inflow, coil-bias absorbed by normalised maps);
* θ: background phase (B0, coil phase, eddy currents); smooth, and common to all encodes;
* v: voxel velocity; k_e: encoding first moment (|k| = π/VENC per axis);
* Σ: intravoxel velocity covariance (Reynolds stress); makes stenotic jets dark in the encoded images;
* n: complex Gaussian noise (after whitening), possibly spatially varying.

Symmetries, and how to exploit each:

| transformation | physical? | exploit by |
|---|---|---|
| global gain & phase x → αx, α ∈ C* | yes (receiver) | canonicalisation (exact) |
| complex shift x → x + Δ𝟏 | no, but P²nP's theorem needs it | canonicalisation (exact) |
| per-voxel phase x_e(r) → e^{iφ(r)} x_e(r), common to e | yes: background phase θ is a U(1) gauge field | gauge-relative features (§3) |
| spatial translation | yes | convolutions |
| cardiac-time translation (periodic) | yes, up to the ECG trigger | cyclic convolutions in t |
| 90° grid rotations and reflections (O_h, 48 elements) acting jointly on space and on v | yes, and exact on voxel grids | group-equivariant convolutions (§4) |
| continuous rotations SO(3) | approximately: oblique slabs, anisotropic voxels, slab-limited FOV | steerable filters optional; small-angle augmentation |
| time reversal; spatial scaling | no | (condition on voxel size if it varies) |

**Induced action on the encodes.** A grid symmetry g ∈ O_h moves voxels (r → gr) and rotates velocities (v → gv). For
axis-aligned encodings (referenced, Cartesian dual-VENC), g is a signed permutation of the three axes.
- A permutation permutes the encodes.
- A sign flip v_a → −v_a conjugates the encode relative to the reference: x_a x̄₀ → conj(x_a x̄₀).

So the group acts **linearly** on the gauge-relative representation c_a = x_a x̄₀/|x₀|:
- Re c_a transforms in the 3-D permutation representation (even under flips);
- Im c_a transforms in the vector (signed-permutation) representation;
- x₀ is a scalar.

For the tetrahedral four-point (balanced Hadamard) encoding the relevant group is T_d ⊂ O_h, which permutes the four
encodes. For icosahedral encodings, decode to a velocity vector first (§3c), after which O_h acts on v as a vector.
escnn ([Cesa, Lang, Weiler, ICLR 2022](https://maurice-weiler.gitlab.io/publication/2022_01_escnn/)) builds steerable CNNs for any
subgroup of O(3), including the octahedral group, with scalar/vector/regular field types and an `export()` to plain
PyTorch.

Why O_h rather than continuous SO(3)? On a voxel grid only O_h acts exactly. Continuous steerable kernels are only
approximately equivariant after discretisation. Anisotropic voxels (2.0 × 2.0 × 2.3 mm in RACLETTE) break even 90°
symmetry about in-plane axes. In that case, use the subgroup that preserves the voxel shape (D_4h about the slab normal)
or resample to isotropic grids.

## 3. Representation

(a) **Raw complex encodes** (FlowMRI-Net's choice, complex-valued convolutions over encodes; [Jacobs et al. 2024/25](https://arxiv.org/pdf/2410.08856)).
Simple and phase-equivariant if the network is bias-free with modulus-only nonlinearities ([Singhal et al., co-domain
symmetry for complex-valued deep learning](https://arxiv.org/pdf/2112.01525)). But the background phase θ(r) has to be
learned away by every filter, and the O_h action on raw encodes is not linear (sign flips act by conjugation relative
to the reference).

(b) **Gauge-relative features** (recommended input). With a noise-regularised reference
x̂₀ = x₀/√(|x₀|² + κσ̂²):
  input = (x₀, c_a = x_a · conj(x̂₀))  for a = 1..E−1.
- c_a no longer carries θ, so it is invariant under the per-voxel phase gauge.
- c_a has homogeneity degree 1 in scale (like x).
- Its phase is k_aᵀv; its magnitude ≈ m·exp(−½k_aᵀΣk_a).
- O_h acts linearly (§2).
- No unwrapping is needed: the network sees e^{iφ}, never φ.
- x₀ keeps θ, which is smooth; the magnitude/phase of x₀ is learnt like an ordinary complex image.

(c) **Decoded physical parameters** (m, θ, v, Σ) per voxel. This is the strongest structure, but the decode is nonlinear,
wraps phases above VENC and is ill-defined where m is small. Use it **for the output parameterisation and the losses**,
not as the input.

**Output.** Predict (m′, θ′, v′, Σ′) and re-encode
x′_e = m′ e^{iθ′} exp(ik_eᵀv′) exp(−½k_eᵀΣ′k_e), plus a small free complex residual r_e (penalised).
- The structured part forces the denoised stack to be a physically consistent multi-encode set: all encodes share
  magnitude and background phase, and differ only by velocity and dephasing. That consistency is a strong prior
  that LLR and generic complex networks lack.
- The residual absorbs what the model misses: partial-volume voxels at the wall, where the phase of a voxel sum is not
  k·(mean velocity), and non-Gaussian intravoxel spectra.
- v′ is an O_h vector field, m′ and θ′ scalar fields, Σ′ a symmetric-tensor field (decomposes into irreps 0 ⊕ 2).
  The architecture outputs the matching field types.

## 4. Architecture

* **Canonicaliser (exact normalisation equivariance).** With the global complex mean Δ̂ (over all voxels, frames and
  encodes) and the RMS deviation µ̂:
  D(x) = µ̂ · N((x − Δ̂𝟏)/µ̂) + Δ̂𝟏,
  so D(µx + Δ𝟏) = µD(x) + Δ𝟏 holds exactly for any inner network N. Optionally also canonicalise the global phase, using
  the phase of a fixed weighted sum of x₀, for exact C*-equivariance.
  - This is simpler than Herbreteau's architectural constraint and keeps ordinary nonlinearities available inside N.
  - Its Lipschitz constant is bounded by N's away from µ̂ → 0, and µ̂ never approaches 0 for images.
  - Check empirically against their constrained-convolution variant.
* **Backbone N.** DRUNet-like U-Net (the denoiser P²nP itself used), 3–4 scales, residual blocks, **bias-free**.
  Factorised convolutions: 3×3×3 in space and 3 cyclic in cardiac time; later ×3 in respiratory phase for 5D.
  Spatial down-sampling only, since F ≈ 20–25 frames is too few to pyramid. O_h equivariance via escnn R3Conv with
  field types:
  - input: x₀ (2 scalars), Re c (permutation representation = trivial ⊕ E), Im c (vector);
  - hidden: mixtures of irreps with gated nonlinearities (vector × sigmoid(scalar));
  - output: scalars m′, θ′ as (cos, sin) of x₀, vector v′, tensor Σ′, residual in the input types.
  Cost of equivariance: weight sharing cuts parameters ~|G|×, but feature maps in regular representations are |G| wide.
  Use irrep-based steerable layers with modest multiplicities to keep memory near a plain U-Net of equal width.
* **Noise level.** Blind through the canonicaliser plus training over a wide σ range (bias-free networks generalise
  across noise levels). Optionally feed a σ̂-map (e.g. from the preconditioner's diagonal) as an extra scalar field
  for spatially varying noise ([non-blind ℂDnCNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC11605166/) uses a g-factor map
  the same way).
* **Lipschitz control** for Assumption 1: spectral normalisation per layer (product bound) and a Jacobian-spectral-norm
  penalty on samples; measure (1+ε) by power iteration on the trained network.
* **Patching.** Volumes of ~120 × 100 × 24 voxels × 20 frames × 4 encodes fit a 24 GB GPU for inference in fp16 with
  modest widths. Training on patches (e.g. 48³ × all frames), inference with overlapping tiles. Translation equivariance
  makes tiling exact up to boundary effects.

## 5. Training

* **Ground truth:** clean complex multi-encode stacks from CFD (RACLETTE: 139 + 75 subjects with in-vivo backgrounds,
  coil maps and CFD truth; our own CFD pipeline later). Pipeline in `research/cfd_training_data.md`. Targets are the
  band-limited projection that matches the reconstruction's PSF (`cfd_training_data.md` §5), not box averages. Note that
  RACLETTE backgrounds move and carry nonzero tissue phase (≈ 4 cm/s equivalent), so "static tissue = zero velocity"
  is not a valid assumption anywhere in the network or the losses.
* **Noise model:** a mixture of
  1. white complex Gaussian at σ spanning the SNR range (blood SNR 5–60);
  2. spatially varying σ from coil geometry (g-factor-like maps);
  3. **operator-shaped noise**: P Aᴴ n for k-space noise n, with our actual trajectory, coils and preconditioner, plus
     undersampling aliasing of the truth (x − αPAᴴ(Ax − y) evaluated at perturbed x).
  Item 3 is what the denoiser actually sees inside P²nP. Training only on item 1 is the classical PnP choice and often
  works, but at R ≈ 4.5 per frame our earlier runs showed that iterates carry strong structured aliasing.
* **Losses:**
  - complex MSE (the MMSE objective under Gaussian noise; scale-consistent with the canonicaliser);
  - velocity error in the lumen, weighted by magnitude;
  - soft divergence penalty ‖∇·v′‖² inside the lumen. Soft rather than hard because partial-volume and voxel-averaged
    velocities are not exactly divergence-free. Hard constructions exist: curl of a vector potential;
    [DAF-FlowNet](https://www.researchgate.net/publication/403429734); divergence-free wavelets.
  - small weights on the residual r_e and on the Lipschitz penalty.
* **Augmentation:** O_h is built in. Add small continuous rotations (±15°) and spatial resampling for robustness to
  oblique slabs, random global gain/phase (redundant with the canonicaliser, cheap), random smooth background phase,
  random VENC choices (k_e as input metadata), and frame-count/heart-rate variation.
* **Splits** by background patient (RACLETTE has 33 distinct backgrounds in its 4D set) to avoid leakage.

## 6. "Respecting the data"

* Inside P²nP data consistency comes from the gradient step; the denoiser must be a good MMSE prior and must not
  invent structure. Equivariance shrinks the hypothesis space, so fewer ways to hallucinate. The structured output
  keeps the encodes mutually consistent; it is not a free image-to-image map.
* Bias check on the validation case that already exists (`opval` on RACLETTE n001): the denoiser at the noise floor
  should leave a clean consistent stack (nearly) unchanged, D(x_true) ≈ x_true. Measure the velocity bias of
  D(x_true + n) averaged over many noise draws.
* Later, for real data without ground truth: self-supervised fine-tuning with equivariant imaging
  ([Chen, Tachella, Davies 2021](https://github.com/edongdongchen/EI); robust EI with SURE, CVPR 2022) using the same O_h
  group and our operator. Or SSDU-style splitting, as FlowMRI-Net does.

## 7. Integration with the Rust stack

Train in PyTorch (rustlib/.venv, torch 2.13). escnn models convert to plain PyTorch modules (`export()`), then to
TorchScript or AOTInductor. Load from Rust with tch-rs (CModule). The canonicaliser and the gauge-feature computation
are cheap elementwise ops: keep them in Rust/CUDA next to the P²nP loop and let the network see canonical inputs
only. LLR stays as the classical baseline prior.

## 8. Ablation ladder (each against the previous)

1. LLR (current) → 2. bias-free complex DRUNet on raw encodes → 3. + canonicaliser → 4. + gauge-relative features →
5. + O_h equivariance → 6. + structured output → 7. + divergence loss → 8. + operator-shaped noise.

Metrics:
- velocity RMSE and bias in the lumen;
- flow-rate curves;
- peak velocity;
- magnitude NRMSE;
- P²nP convergence (iterations to plateau, stability over long runs);
- the measured Lipschitz constant.

## 9. Open questions

* Is complex shift-equivariance (Δ𝟏) actually needed in practice? It's non-physical, and the canonicaliser makes it
  free, but it shifts the denoiser's input statistics. P²nP's theorem uses it.
* How much do continuous-rotation steerable filters gain over O_h + augmentation on oblique slabs?
* 5D: respiratory bins as a second cyclic axis, or as a separate low-rank factor?
* Whether to train the denoiser jointly with the preconditioner (an unrolled P²nP, like FlowVN/FlowMRI-Net) once the
  plug-and-play version works. The convergence guarantees then no longer apply.

References are linked inline. Related project notes: `research/synthetic_4dflow_pipeline.md`,
`research/cfd_training_data.md`, `research/concomitant_segmented_first_order.md` §7.

## 10. Revisions after review (2026-11-03, supersede §0, §3, §4 where they differ)

**What P²nP actually needs.** The P²nP convergence proofs (fixed and dynamic preconditioner, Appendix C) use only the
Lipschitz bound (1+ε) and a bounded gradient. Normalisation equivariance only justifies running one fixed denoiser
while the effective noise level changes with η_k: the MMSE scaling argument (1/µ)D_σ(µx) = D_{σ/µ}(x). The complex
shift Δ𝟏 is used nowhere. **Target: exact positive-scale equivariance (plus global phase). Drop the shift.**

**Why the shift must go.** Exact local-gauge equivariance (D(e^{iφ}x) = e^{iφ}D(x) for every per-voxel phase field φ)
plus exact shift equivariance would force D to be the identity plus a constant. Proof: gauge plus shift gives
equivariance to adding Δe^{iφ(r)} for any φ. Composing two such shifts gives an arbitrary complex field u, so
D(y+u) = D(y) + u for all u. The shift is unphysical; the gauge (background phase) is physical. The local gauge itself
is only approximate (the smooth background phase *should* be denoised), so the network gets gauge-relative features,
not a hard constraint.

**Symmetry group and how each factor is built in.**
G = (ℝ₊ × U(1)) × (ℤ³ ⋊ O_h) × ℤ_F. Value-space and domain actions commute, so G is a direct product and each factor
can be built into the layers independently:
- **Translations, cyclic time:** convolutions; circular in t.
- **O_h:** regular-representation group convolutions. O_h acts on regular feature channels by permutation.
- **ℝ₊:** bias-free layers and positively homogeneous nonlinearities. Pointwise ReLU commutes with permutations *and*
  with positive scaling, so a bias-free regular G-CNN with ReLU is exactly O_h × ℝ₊-equivariant with no further
  machinery. This is the main interplay, and it favours regular over steerable/irrep features: irrep fields need norm
  or gated nonlinearities, and a sigmoid gate breaks homogeneity unless its argument has degree 0 (a ratio of features).
- **No normalisation layers with learned shifts.** Conditioning on VENC_hi, spacing h and the noise map goes in
  multiplicatively: FiLM-style γ(·) of degree 0, no additive β. A σ̂ input must be estimated equivariantly (degree 1).
- **Global U(1):** exact canonicalisation by the phase of a fixed weighted sum of x_ref. This is cheap, and the phase
  is undefined only for an all-zero image. The c_a features are already invariant.

**Canonicaliser versus architecture.** For a degree-1 homogeneous N, Euler's identity J_N(z)z = N(z) cancels the
Jacobian term that a scale canonicaliser adds, so a scale canonicaliser is redundant. A canonicaliser built from
global statistics also differs between training patches and full volumes. Architectural equivariance is local, so
patch-trained weights apply unchanged to whole volumes. **Choose architectural equivariance.**

**Encoding: fixed 7-point dual-VENC layout** [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi], with encodes along the
logical (grid) axes.
- RACLETTE: VENC_lo = 1.0 m/s in every subject; VENC_hi varies from 1.0 to 7.0 m/s, so VENC_hi is a conditioning input.
- RACLETTE encodes along scanner xyz, about 20° oblique to its grid. The training patches are therefore resampled on
  the fly onto encode-aligned isotropic grids: B-spline, exact at the nodes; verified in
  `rustlib/tools/denoiser/test_dataset.py`.
- On that grid O_h acts exactly on the channels: permutations permute the (lo, hi) pairs, and a flip v_a → −v_a
  conjugates c_a (x_a → conj(x_a)·x_ref/conj(x_ref)). Noise statistics are preserved exactly.
- Real channel types:
  - x_ref (Re, Im): 2 scalars;
  - Re c_{a,v}: permutation representation, 2 copies;
  - Im c_{a,v}: vector (signed-permutation) representation, 2 copies;
  - 14 real channels in total.
- hasty-mrsim needs a `SevenPoint` encoding variant (it has Referenced and Hadamard 4-point today).

**Patch-based training and inference (4DFlowNet-style).**
- **Training:** P = 32³ at h ∈ [1.8, 2.4] mm, T = 12-frame cyclic windows, lumen-biased centres (70 %), a validity
  mask (inside the acquired slab) and the loss only on the interior (4 voxels, 1 frame margin). Every volume yields
  thousands of distinct patches × 48 symmetries × windows.
- **Inference:** whole-volume evaluation. A RACLETTE-size stack of 120×110×28 voxels × 25 frames fits on the GPU for a
  modest-width net. Otherwise **overlap-tile with crop**: tiles overlap by at least the receptive-field radius, and
  only the tile centres are kept. That is identical to whole-volume evaluation: no seams, no blending, and the
  Lipschitz constant is that of N.
- **No partition-of-unity blending.** It hides seams, but it is not equal to the full-volume map, and its Lipschitz
  bound can grow by up to √K (K = overlaps per voxel). Inside P²nP, also no random tile shifts per iteration: D must
  stay one fixed map.
- **Time:** circular convolution over the full cycle at inference.
- **Receptive field target:** ≈ 32–40 voxels spatially (3 U-Net levels) and ≈ 7–9 frames.

**Implementation (2026-11-03).** The layers are in plain PyTorch (`rustlib/tools/denoiser/oh.py`), with no
escnn and no custom kernels. All field types are monomial representations, so a layer is a signed gather from a small
parameter tensor followed by a standard conv3d. This keeps the model portable: `torch.compile`, TorchScript,
`torch.export` and AOTInductor all work, and are tested. Inference from Rust goes through a small libtorch C ABI shim
(`crates/hasty-denoise-torch`) that loads TorchScript or AOTInductor packages. tch-rs is pinned to libtorch 2.13 and
TorchScript only. Status and tests: `rustlib/PLAN.md` §19.

**Libraries and efficiency (survey).**
- escnn: kernels are expanded into ordinary Conv3d weights and `export()` gives plain PyTorch, so there is no inference
  overhead. The project is lightly maintained.
- NVIDIA cuEquivariance: tensor-product kernels for point-cloud/graph models (MACE, NequIP, Allegro, DiffDock). It has
  no voxel-grid convolution, so it does not apply here.
- e3nn: has an SO(3)-steerable voxel convolution, heavier.
- For O_h regular group convolutions the cost is intrinsic, not a library question. A regular field is 48 channels,
  and a group convolution is a standard Conv3d with tied weights: each g ∈ O_h permutes the 27 entries of a 3×3×3
  stencil exactly. Plan: an own small PyTorch layer that precomputes the 48 stencil permutations, gathers the full
  weight each forward and lets cuDNN do the convolution. It exports as an expanded Conv3d and later ports directly to
  Rust/NVRTC.
- Memory control: the rotation subgroup O (24 elements) for part of the layers, quotient representations, or few
  regular fields per layer. At equal channel width the FLOPs equal those of a plain CNN; the weights are 48× fewer.
