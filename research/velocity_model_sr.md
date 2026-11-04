# Velocity-parametrised 4D/5D flow reconstruction and finer reconstruction grids — research plan

Status 2026-11-04: plan only. Builds on `rustlib/crates/hasty-recon` (P²nP with ZMSHR1, SegNufft with temporal
B-splines, `flow::FlowDecoder`), the 7-point denoiser (`research/denoiser_design.md`) and the synthetic pipeline
(`research/synthetic_4dflow_pipeline.md`).

**Two threads:**
- **(A)** Reconstruct the physical quantities instead of 4 / 7 / 13 complex encode images:
  - complex reference z;
  - velocity v, in a temporal basis;
  - a low-order residual phase per encode.
  Solve with the same P²nP algorithm and dynamic preconditioner, and a learned prior in (|z|, v) space.
- **(B)** Reconstruct on a finer grid than the acquisition. The scale is a flag (anisotropic, non-integer), with
  cost-aware strategies (coarse-to-fine, ROI grids, exact voxel basis functions).

**Why together:**
- The velocity model cuts unknowns per voxel about 6×. Example: K_z = 6 and K_v = 12 nodes give 48 reals =
  192 B per voxel, against 1120 B for a complex 7-point × 20-frame stack.
- That memory is what finer grids need.
  - At R4 size, s = 2 becomes feasible: 5.6 GB instead of 32.5 GB of P²nP state.
  - At 320³ the complex stack doesn't fit even at s = 1 (37 GB per stack). The parameter stack (6.3 GB) does fit,
    with state offload.

---

## A. The velocity model

### A1. Forward model

Per encode e, coil c, readout sample n at absolute time t_n with cardiac phase τ_n:

    y_{e,c,n} = Σ_r S_c(r) · x_e(r, τ_n) · Φ_phys(r, n) · e^{−i 2π k_n·r} + noise
    x_e(r, τ) = z(r, τ) · e^{i φ_e(r)} · e^{−i γ M1_e · v(r, τ)}

| unknown | meaning | temporal model | size per voxel |
|---|---|---|---|
| z(r, τ) ∈ ℂ | reference signal: magnitude × common phase (coil / B0 at TE / anything shared by all encodes) | spline, K_z nodes (K_z ≤ K_v; magnitude is smooth except inflow) | 2 K_z |
| v(r, τ) ∈ ℝ³ | velocity, logical axes | periodic B-spline (or Fourier), K_v nodes | 3 K_v |
| φ_e(r) = Σ_p a_{e,p} P_p(r), e ≥ 1 | residual phase not explained by v: eddy currents, imperfect Maxwell / B0 / GNL correction, gradient delays | static per encode (the same bipolar every TR) | global: (E − 1) × #P |
| (later) Σ(r, τ) ⪰ 0 | intravoxel velocity covariance: dephasing \|x_e\| ∝ exp(−½ γ² M1_eᵀ Σ M1_e). Turbulence (IVSD / TKE); needs ≥ 2 VENCs | spline | 3 (diagonal) or 6 |

- **Physics.** Φ_phys = everything SegNufft already models per sample: B0 segments, concomitant terms (k-shift,
  phase, ΔC kernel), GNL positions. The velocity model sits on top. x_e is the "magnetisation image" SegNufft
  consumes.
- **Why z is complex, not m ≥ 0 plus θ.**
  - The model is **linear in z for fixed (v, a)**: no positivity constraint and no θ wraps.
  - Any phase shared by all encodes may vary over the cycle (motion through B0 inhomogeneity); velocity information
    lives only in encode differences.
  - It allows variable projection: z is the exact linear-least-squares solution given v (option in A4).
- **Unknowns per voxel and node:** 5 reals (z, v), against 2E reals for the complex model (8 for 4-point, 14 for
  7-point, 26 for 13-point). With K ≤ F nodes, the total reduction is 5–10×.
- **Encode coupling.** All encodes are explained by one v. Dual-VENC unwrapping is no longer a per-voxel rounding
  after the reconstruction, which was the failure mode in G1 (0.18 % wrap outliers dominate its RMSE).

### A2. Temporal basis and evaluation frames

The operator never sees the basis directly.
1. Evaluate x_e at **F_op operator frames** (pointwise).
2. SegNufft's existing temporal path interpolates complex images between operator frames to each readout's τ_n
   (`SegNufftOpts { frames, bspline }`).

So there are three separate knobs:

| knob | sets | cost |
|---|---|---|
| K_v, K_z basis nodes | unknowns / conditioning / smoothness | memory, prior cost |
| F_op operator frames | accuracy of complex interpolation between frames: the phase change per interval γ M1·Δv should stay < ~π/4 at peak acceleration | FFTs ∝ E · F_op per iteration |
| basis type | periodic cubic B-spline (local support, interpolating, so the prior acts on node samples); Fourier (global, periodic); learned subspace (CFD waveform PCA) | — |

- Non-uniform knots, denser in systole, placed from the ECG and the warm-start pulsatility.
- Coarse-to-fine refinement in K (1 → 5 → 10 → 20). Each level warm-starts the next; the current time-averaged
  warm start is the K = 1 case.
- Gradients per iteration don't grow with K: one forward and one adjoint per set, as now. Iteration counts follow
  conditioning, which fewer unknowns improve.

### A3. Residual phase and identifiability

- **Basis.** Legendre polynomials over the FOV, total degree d ∈ {0, 1, 2, 3} (1 / 4 / 10 / 20 terms) per encode
  e ≥ 1. The reference encode's share is absorbed by z.
  Option: a linear term in absolute time (B0 drift is shared by all encodes, so it sits in z unless the encodes are
  interleaved unevenly).
- **The degeneracy.** A velocity field that is itself a low-order polynomial over the whole FOV is indistinguishable
  from φ_e. Two things break it:
  1. **Pulsatility.** φ_e is static, so it trades only against the *time-mean* of v. The pulsatile part of v is
     identifiable.
  2. **Static tissue.** v ≈ 0 in non-vessel tissue. This is the standard phase-offset correction (polynomial fit to
     static tissue; e.g. Walker et al. 1993, Lankhaar et al. 2005 — verify the references), done here jointly
     inside the reconstruction.
     Implemented as a soft penalty λ_s Σ_{r ∈ static} |v(r, τ)|². The static mask comes from low PC-MRA × low
     temporal variance of the warm-start phase; the learned prior also pushes v → 0 in tissue.
- **Initialisation.**
  1. Fit a_e to the warm-start encode phase differences in the static mask (pointwise least squares, a 10–20 term
     system).
  2. Then update jointly. There is no prior on a; optionally a Tikhonov term towards the init.
- **Simulator support needed** (`sim4d --residual-phase`): inject per-encode polynomial phase (degree ≤ 3, amplitude
  equivalent to 1–10 cm/s at the FOV edge, plus a small non-polynomial part to test the mismatch) so recovery
  can be measured.

### A4. Optimisation: P²nP in parameter space

Same algorithm, in the parameter vector θ = (z-nodes, v-nodes, a):

    θ_{k+1} = D(θ_k − α P_k ∇f(θ_k)),   f(θ) = ½ Σ_sets ‖E_t G_t(θ) − y_t‖²
    ∇f = Jᵀ A^H (A G(θ) − y)

G is the pointwise map to x_e at the operator frames; Jᵀ is its pointwise adjoint followed by the spline transpose.

1. **Gradient.**
   - Forward: build x_e frames from θ for each encode on the fly. Only one encode × F_op frames is transient;
     the complex stack is never stored.
   - Then SegNufft forward, the residual, and the adjoint, giving g_x.
   - Chain rule (all pointwise, w = conj(x_e) g_x):
     - ∂/∂z̄ = Σ_e e^{−iφ_e} e^{+iγM1_e·v} g_x;
     - ∂/∂v = Σ_e γ M1_e Im(w);
     - ∂/∂a_{e,p} = Σ_r P_p(r) Im(w).
   - Cost: the same NUFFT calls as the complex model at F_op frames, plus elementwise kernels.
   - **Gate:** finite-difference gradient checks in f64.
2. **Preconditioner = pointwise Gauss–Newton blocks + ZMSHR1** (the analogue of the existing `Zmshr1M`).
   - B(r, node) = J_rᴴ diag(n̂_r) J_r, a 5×5 block per voxel and node.
   - n̂_r is a diagonal estimate of the normal operator at r: the circulant preconditioner's DC times Σ_c |S_c|².
   - For v, the block is Σ_e γ² |z|² M1_e M1_eᵀ. So uncertainty ∝ 1/|z|² comes for free, and velocity in dark
     voxels doesn't drive the step.
   - P_k = τ_k B⁻¹ + rank-one, with the secant pair taken in B-whitened coordinates. This fixes the units between
     z (signal) and v (m/s).
   - The safeguards already in `p2np.rs` keep P_k ≻ 0 under negative curvature — exactly what the nonconvex f
     produces. Those safeguards: the convex-combination secant v = a s + (1 − a) m with bounds on sᵀv / sᵀs and
     vᵀv / sᵀv.
3. **Step size.**
   - α = 1 / λ_max(B^{−½} Jᴴ N J B^{−½}), by power iteration at the start and every R iterations, because J
     changes.
   - Monitor f; if it rises twice, re-estimate. Keep P²nP's fixed-α structure; no line search in the default.
4. **Nonconvexity.**
   - (i) **Initialisation:**
     1. time-averaged warm start (as now);
     2. pointwise fit of (z, v) with the hierarchical `FlowDecoder` (dual-VENC unwrapping is exact for a
        noise-free input);
     3. a few iterations of the complex-model P²nP;
     4. a pointwise spline fit to the frames.
   - (ii) **VENC continuation (graduated nonconvexity):** scale the low-VENC M1 from M1_hi to M1_lo over the first
     N_c iterations, or ramp the low-VENC encode weights from 0 to 1. Each stage starts inside the basin of the
     previous one.
   - (iii) **Optional wrap moves:** every N iterations, test v → v ± 2 VENC_lo per axis in voxels with large
     encode-phase residuals. The test is a pointwise surrogate on the current x_e − G(θ); accept if the data term
     drops.
   - (iv) **Optional variable projection:** for fixed (v, a), z solves a linear least-squares problem (a few CG
     steps with the existing operators). This leaves (v, a) as the only nonlinear unknowns. It is compatible with
     P²nP by treating the z-solve as part of the gradient map — evaluate against plain joint P²nP.
5. **Convergence.** PnP with smooth nonconvex f and a Lipschitz denoiser converges to fixed points for α < 1/L with
   uniformly bounded metrics. ZMSHR1's bounds supply that boundedness. Check the exact conditions in Hong et al.
   (P²nP) and Hurault et al. (gradient-step PnP, nonconvex) before claiming anything.

### A5. Prior in (|z|, v) space

- **Field types (O_h).** |z| is a scalar (trivial representation); v is a vector (signed permutations). Both are
  monomial, so `oh.py` supports them directly. The gauge (arg z) and the residual phase never enter the prior.
- **Scale.** Signal intensity scales |z| but not v. Feed the network p = |z|·v (magnitude-weighted velocity, which is
  also the well-determined quantity) together with |z|. A bias-free ReLU network on (|z|, p) is degree-1
  homogeneous; read out v = p / |z| with a safe division. That gives exact intensity equivariance for |z| and exact
  invariance for v, architecturally (your preference over canonicalisers).
- **Noise model for training.**
  1. Add complex noise to clean 7-point patches (as now).
  2. Map to (|z|, v) with the pointwise decoder.
  This reproduces the real input distribution: σ_v ∝ VENC/(SNR·|z|), and wrap outliers in dark voxels.
  An optional input is the per-voxel Fisher information from B, so the network knows the local uncertainty.
- **Physics option.** Divergence-free velocity in the lumen. Either a masked Helmholtz projection after D, or a
  network that predicts a vector potential (exactly divergence-free) inside the lumen mask. Evaluate both against
  near-wall error.
- **Time.** Nodes act as frames (interpolating spline). Training frame spacing must match K_v, so add temporal
  B-spline resampling of RACLETTE's 25 frames to arbitrary K, alongside the existing spatial resampling. The temporal
  receptive field is counted in nodes.
- **Data.** RACLETTE: fit (|z|, v) from the clean 7-point images. Our CFD pipeline: v directly, at any resolution
  (needed for B).
- **Baseline meanwhile.** The current complex 7-point denoiser, applied to G(θ) at the nodes and projected back by a
  pointwise fit. That is a mismatched but cheap comparison.

### A6. Later physics the velocity model unlocks

- **Readout motion.** With v explicit, the readout first moment adds a per-sample phase e^{−iγ m1_ro(t_n)·v(r)}.
  It has the same structure as B0 (a spatial "frequency" v(r) × a sample "time" m1_ro), so it can be segmented
  like B0: low-rank in (voxels × samples), with the segments refreshed when v changes. That removes the `noro`
  simplification; spiral readouts last 4.4 ms.
- **Turbulence.** The intravoxel covariance Σ (TKE) from multi-VENC and 13-point ICOSA data.

---

## B. Finer reconstruction grids

### B1. What it buys

1. **Discretisation error of the forward model.** The object is continuous (sharp vessel walls, 0.5 mm blood truth
   in our simulator); a 2 mm voxel grid is a model mismatch even with perfect data.
2. **Null-space filling.** Data constrains |k| < k_max; a prior trained at the finer h fills the rest. This is
   super-resolution inside the inverse problem, with data consistency — unlike post-hoc SR networks
   (4DFlowNet-style).
3. **Partial volume and intravoxel dephasing at walls.** Smaller voxels make the single-velocity-per-voxel
   assumption truer. This matters for flow rates near walls, and later for wall shear stress.

Separate from the grid: **the voxel basis function.** A rect voxel contributes ∏_a sinc(k_a Δ_a / 2) per sample,
and a cubic B-spline basis contributes sinc⁴. That is a per-sample k-space weight: diagonal, essentially free in
SegNufft. The current operator implicitly assumes Dirac voxels. Measure this effect at every grid scale, because it
may recover part of the benefit at s = 1.

### B2. The flag

    --grid-scale s | sx,sy,sz       (default 1; e.g. 1.25, 1.5, or 1.5,1.5,1 for in-plane only)
    --voxel-basis dirac|rect|bspline3
    --sr-start k                    (coarse-to-fine: native grid until iteration k, then upsample and continue)
    --roi x0:x1,y0:y1,z0:z1 --roi-scale s   (phase B4: fine grid only inside a box)

- **Size.** dims_a = the smooth size ≥ s_a · n_a. Prefer sizes 2^a 3^b 5^c 7^d. hasty-fft also handles radix 11 / 13
  and Bluestein, but smooth sizes are cheaper. The NUFFT grid then gets its own σ = 1.25 rounding.
- **Fixed physical extent.** The FOV is kept exactly; the voxel size is FOV / dims.
- **Recomputed at the new positions (all analytic or sampled per voxel):** physics grid (x_log, the GNL Jacobian, B0,
  concomitant monomials), coil maps (B-spline-resampled from the native estimate, or the simulator truth), lumen
  mask, truth sampling.
- **Code change:** `SimData::physics_grid_on(dims, res)`. The operators already take dims and voxel size.

### B3. Cost

Assumptions:
- complex64 images, f32 parameters;
- velocity model with K_z = 6, K_v = 12 → 48 reals = 192 B per voxel;
- complex 7-point × 20 frames → 1120 B per voxel;
- P²nP keeps about 4 stacks: x, g, and the secant pair.

R4 (180 × 180 × 28 = 0.91 M voxels), P²nP state (×4 stacks):

| s | voxels | FFT work vs s = 1 (N log N) | complex model | velocity model |
|---|---|---|---|---|
| 1 | 0.91 M | 1× | 4.1 GB | 0.70 GB |
| 1.25 | 1.77 M | ≈ 2.0× | 7.9 GB | 1.4 GB |
| 1.5 | 3.06 M | ≈ 3.7× | 13.7 GB | 2.4 GB |
| 2 | 7.26 M | ≈ 9.2× | 32.5 GB (does not fit 24 GB) | 5.6 GB |

A 320³ acquisition (32.8 M voxels), one stack:

| s | voxels | NUFFT grid per coil channel (σ = 1.25) | complex model | velocity model |
|---|---|---|---|---|
| 1 | 32.8 M | 0.51 GB | 36.7 GB | 6.3 GB |
| 1.25 | 64 M | 1.0 GB | 72 GB | 12.3 GB |
| 1.5 | 111 M | 1.7 GB | 124 GB | 21.2 GB |
| 2 | 262 M | 4.1 GB | 294 GB | 50 GB |

**Consequences:**
- **320³, s = 1.** The complex time-resolved stack doesn't fit at all. The velocity model fits only with care:
  - keep x and g on the GPU;
  - keep the ZMSHR1 secant pair in bf16 or in pinned host memory (≈ 6 GB over PCIe per iteration ≈ 0.3 s).
- **320³, s ≥ 1.25 over the full FOV.** Out for memory and FFT (≥ 2×, and 1–4 GB per channel grid, which forces the
  coil batch to 1–3). That is the job of ROI grids.
- The sensible levers are modest s (1.1–1.3), anisotropic s, coarse-to-fine, and ROI grids.
- The multi-channel pruned FFT with fused pad / crop (open item, PLAN §17) removes about half of the NUFFT time at
  any s.

### B4. Strategies, in order of implementation

1. **Anisotropic / non-integer s.**
   - Refine where the acquisition is coarsest or where vessel walls are oblique.
   - Stack-of-spirals has its kz extent from the partitions, so s_z and s_xy differ in what they buy.
   - Measure per axis.
2. **Coarse-to-fine (`--sr-start`).**
   - Most iterations run at native cost.
   - Upsample by B-spline interpolation (z, a) or a vector-field-aware method (v).
   - A few fine iterations with the fine-h prior.
3. **Exact voxel basis** (`--voxel-basis`): a free per-sample weight, tested at every s.
4. **ROI dual grid.** Native grid everywhere plus a fine grid inside a vessel bounding box (the aorta is about
   5–15 % of the FOV):
   - x = U_c x_c (outside the ROI) + U_f x_f (inside), with a smooth partition of unity across the box boundary;
   - two NUFFT plans summed in k-space;
   - the fine ROI at s = 2 costs about as much as the whole native grid.
   This is the route to SR at 320³.

### B5. Prior at finer h

- The architecture conditions on h and the patch sampler takes an h range, so SR is a matter of training data with
  fine truth.
- RACLETTE `7p_2Venc_Super` has Low (4–5 mm) / High (2–2.5 mm) levels, 50 cases. It supports an SR test of 4 mm
  acquisitions reconstructed at 2 mm, with a prior trained on the High level.
- Below 2 mm we need our CFD pipeline, or our simulator truth (0.5 mm blood sources, 1 mm tissue). The current
  RACLETTE-only model can't supply detail below 2 mm that it never saw.

---

## C. Milestones, experiments and gates

All runs are interruptible and resumable (`RunCtl`) and write arrays, figures and flow-viewer values
(`tools/flow/show_run.py`).

| # | step | experiments | gate / metric |
|---|---|---|---|
| S0 | simulator: `--residual-phase` (polynomial + non-polynomial part); a longer-scan R4 (3 min) to separate undersampling from model error | — | sims reproducible; truth phase stored |
| V0 | pointwise study on the warm start and the complex-model reconstructions | fit (z, v, a) per voxel; residual-phase recovery vs d; unwrap failures with continuation vs per-voxel rounding | residual-phase recovery error (rad, cm/s equivalent); wrap-outlier rate < 4-point baseline |
| V1 | velocity-model operator wrapper | G, Jᵀ, spline map, VENC continuation | finite-difference gradient ≤ 1e-6 (f64); limit test: K = F_op with no residual phase reproduces the complex-model data term |
| V2 | preconditioned gradient, no prior | none / B⁻¹ / B-ZMSHR1; K_v ∈ {5, 8, 12, 20}; F_op ∈ {20, 30, 40} | iterations to a target data residual vs the complex model; velocity RMSE at true positions |
| V3 | P²nP with simple priors | static-tissue penalty, spatial TV / LLR on (\|z\|, p); variable projection on / off | G-series: RMSE, median \|e\|, unwrap failures, plane flow error (`flow_plane` at fixed centerline planes), s / iteration |
| V4 | learned (\|z\|, p) denoiser | O_h scalar + vector fields, bias-free homogeneous; divergence-free variants; temporal resampling | val NMSE / v RMSE on RACLETTE; then V3 metrics on R4 (cross-simulator) |
| V5 | full physics | B0 + concomitant + GNL (`--phys full --gnl`); readout-motion segments | the same metrics on non-ideal sims (R4 full, `noro` off) |
| B0 | grid flag + basis | `--grid-scale` incl. anisotropic, `--voxel-basis`; `physics_grid_on` | `opval` model error vs dense truth for s ∈ {1, 1.1, 1.25, 1.5, 2} × basis |
| B1 | SR reconstruction | coarse-to-fine vs full fine; fine-h prior (RACLETTE Super 4 → 2 mm; R4 2 → 1.33 mm with simulator truth) | near-wall velocity error, plane flow error, cost (s / iteration, GB) |
| B2 | ROI dual grid | box from PC-MRA; blend width | equals full-fine accuracy inside the ROI at ≤ 1.5× native cost |

**Order:**
1. S0 → V0 → V1 → V2. These decide whether the model is worth it.
2. In parallel, B0, cheaply on the CPU / `opval`.
3. Then V3 + B1.
4. V4 once the complex denoiser training infrastructure is reused for (|z|, p).

**Decisions to settle by experiment, not up front:**
- B-spline vs Fourier vs learned temporal basis;
- uniform vs systole-dense knots;
- residual-phase degree;
- variable projection or joint updates;
- whether Σ (TKE) is identifiable at our SNR.

## D. Risks and open questions

- **Basins of attraction.** Low-SNR, high-R voxels may stay wrapped despite continuation.
  Mitigations: the prior, wrap moves, and a better initialisation from the complex model.
- **Model mismatch between encodes.** Intravoxel dephasing differs per VENC; the common-z model ignores it until Σ
  is added. Finer grids reduce it.
- **Residual phase vs time-mean velocity in small FOVs with little static tissue.** Expect higher a-uncertainty;
  report it from B.
- **Dynamic preconditioner on a changing J.** The secant pairs mix curvature from different J. The ZMSHR1 bounds keep
  steps safe; watch for stalls. If needed, reset the pair after VENC-continuation stages.
- **FFT cost dominates any s > 1 at clinical matrix sizes.** The pruned FFT and ROI grids are prerequisites for
  320³.
- **Interface with your own sequence design.** This framework also scores encodings and trajectories:
  - 4-point Hadamard vs 7-point dual-VENC vs 13-point ICOSA, scored by the information they give about (v, Σ) under
    the model — Fisher information from B;
  - knot placement vs temporal sampling patterns.

## References (to verify before citing)

- Hong, Xu, Hu, Fessler — provable preconditioned plug-and-play (P²nP), ZMSHR1.
- Hurault, Leclaire, Papadakis — gradient-step denoisers / nonconvex PnP convergence.
- Liang (partial separability); Zhao et al. — low-rank temporal subspaces.
- Tan et al. — model-based reconstruction for real-time phase-contrast flow MRI.
- Walker et al. 1993; Lankhaar et al. 2005 — background phase-offset correction by polynomial fits to static tissue.
- Dyverfeldt et al. 2006 — intravoxel velocity standard deviation / turbulence from generalised phase contrast.
- Ferdian et al. 2020 — 4DFlowNet (post-hoc SR, CFD-trained) as the SR baseline.
