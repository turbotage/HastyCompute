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

---

## E. At-knots mode on the 2-D knot operator (2026-11-18, implemented)

`recon_p2np --vmodel knots [--knots2d RK:RDEG,CK:CDEG]` (default 6:1,16:3, the target of temporal_operator.md §21).

**Model.** Operator frames = the K = RK·CK knots of the respiratory × cardiac basis (knot k = r·CK + c, the layout of
`SegNufft::new_temporal2d`), identity weights: x_{e,k} = z_k e^{i(φ_e − k_e·v_k)}. One set per encode holds all K knot
images; the knot-major operator interpolates them to every sample's (breathing, cardiac) time.
- `VelSpec::at_knots` / `VelSpec::knots = Some([kr, mc])`: nf = kz = kv = K; frame f reads node f only (GPU kernels
  called with one node pointer and weight 1; `eval_knot` / `vjp_knot` / `jvp_knot` on single node chunks). The
  per-frame model is unchanged (`knots: None`).
- Metric (`vobj`): the Gauss–Newton blocks of knot k carry ω_k = G_kk / mean (diagonal of the temporal Gram of the
  basis under the sample times; ph0001 --keep 8: 0.14–1.97 × mean, Gram condition 315–419).
- State: device (run_objective_with), or host (`run_objective_host`; `gradient_host` / `hvp_host` / `lambda_max_host`
  stream θ knot by knot through six node slots: the device holds one encode's knot stack, its residual and data, never
  a parameter stack). Chosen by `vknot::plan` (the knot ledger) before the solver; refusal with the table otherwise.
- Breathing coordinate: bellows amplitude (physio.npy col 3) at the readouts used, linear knots at its quantiles;
  `--resp truth` = oracle. `--keep N`: the splinerecon readout subset.
- Priors: `--vkappa` (LLR on z and |z̄|·v), `--dfw` (v knots), the per-axis knot lock `--static-lock MU[:IT0[:EVERY]]
  [--lock-coupling additive|nested]` (`hasty_recon::knotlock`, map from IT0 every EVERY, saved for resume), the
  static anchor `--vanchor`. `--coil-sampling keep:draw[:hold[:last[:seed]]]` (α from the largest λ over 3 draws).
- Pyramid: `--grid-scale` / `--kcrop` / `--vcoils` / `--nseg` are per level; `--init-from <knot run dir>` resamples the
  previous level's knot parameters (vparams_<tag>.hv + vknots_<tag>.json; z band-limited with the image scaling, v
  rescaled as a velocity). No dynamic-box-with-static-outside mode exists in recon_p2np (the box runs are box scans).

**Caveat: complex interpolation between knots** (`research/experiments/vknot/interp_caveat.py`; phantom truth used for
this evaluation only). ph0001 lumen (truth fraction > 0.9, 54 372 voxels), truth velocities band-limited from the 20
frames, sim encodes k_e; velocity along each low-VENC encode difference e − 0 (VENC 1.0 m/s), errors relative to the
dynamic velocity RMS (§16.1's normalisation):

| cardiac basis | adjacent-knot phase step max / p99 (rad) | basis only | VM, truth knots | **VM, fitted** (its interpolant readout) | complex knots | 20 bins |
|---|---|---|---|---|---|---|
| 16 cubic | π / 2.87 | 0.10 | 0.55 | **0.147** (0.221) | 0.122 | 0.221 |
| 16 linear | π / 2.36 | 0.13 | 0.22 | **0.150** (0.156) | 0.139 | 0.221 |
| 20 / 24 / 32 cubic | π / 2.60, 2.15, 1.39 | 0.034 / 0.014 / 0.003 | 0.34 / 0.19 / 0.065 | — | 0.090 / 0.064 / 0.030 | 0.221 |

- "Truth knots" = v_b from a least-squares velocity fit, "fitted" = (z_b, v_b) fitted to the true per-encode signal (noise
  free, 600 lumen voxels): the model's own optimum. The phase steps reach π (6–8 % of steps > π/2): the per-encode
  phases k_e·v change by up to a wrap between knots in systole, far past A2's π/4 rule.
- At its optimum the velocity model is close to complex knot images (0.147 vs 0.122) and better than 20 bins (0.22),
  but its knot velocities are not the velocity at the knots: their interpolant reads 0.221 with cubic knots (the fit
  bends v_b to undo the cubic phasor averaging). Linear cardiac knots remove most of that (0.156) at equal knot count.
- Readout: score from the interpolated complex model (score_pair / --eval files), not from Σ_b B_b v_b, until
  the knot count or degree is chosen.

**Memory plan** (`vknot::plan`, sizes from `research/experiments/vknot/ledger_estimate.py`; 24 GB card, K = 96):

| case | leanest plan (host state, data staged, tables streamed): device | host state | verdict |
|---|---|---|---|
| 2 mm full scan, 31 coils | 37 GiB (operator 6.6, knot stack 3.0, residual + staged set 14.0, table slots 11.8) | 32 GiB | refused: one encode's data (30 M samples × 31 coils) and tables do not fit; needs coil-batch / readout-chunk streaming inside the knot-major engine (no group calls there) |
| 2 mm, --keep 8, 31 coils | 14.4 GiB | 32 GiB | fits (smoke at 4 mm ran; dev run at 12 coils) |
| 1 mm heart box, 22 coils | 22.8 GiB | 63 GiB | marginal on the device, over a 60 G host cap: half-precision secant pair (memory_ledger.md §3.1) or 4 × 16 knots (20.7 / 43 GiB) |

**Cost.** The knot-major operator's FFT work is K · L · coils per encode and pass: 4 mm smoke (31 coils, L 10, 0.86 M
samples per encode) 48.6 s per Hessian-vector product (7 encodes); 2 mm --keep 8, 12 coils: 18.8 s per knot CGLS
iteration per encode (splinerecon). The autotune compiles a module per tile candidate (140–340 s at coil batch 16).

**Convergence** (tests/vknot.rs, 12×10×8, 2 × 4 knots, noise-free): the diagonal metric leaves the knot coupling of
the cubic basis (Gram condition 37 for 4 cubic knots; 315–419 for 6 × 16 on ph0001) to ZMSHR1: velocity error ×1/10 in
80 iterations, ×1/100 in 250. A G⁻¹-mixing metric over the knots of a voxel needs a metric API over chunk groups.

## F. Parameter-spline mode: (m, ψ, v) in the knot basis, operator frames (2026-11-18, implemented)

`recon_p2np --vmodel knots` now means this mode (`--vop frames:F[:DEG]`, default `frames:48:3`); the at-knots model of
§E stays selectable with `--vop knots`.

**Model.** Parameters per knot k = r · mc + c of the respiratory × cardiac basis (6 linear × 16 cubic, the target):
magnitude m_k, zero-moment phase ψ_k, velocity v_k (5 reals, the same stack size as §E), plus the per-encode Legendre
residual phase. Along the cardiac axis the physical parameters are splines; along breathing the rows are mixed
complex-linearly (a breathing state between two knots is the partial-volume mixture of the two knot states):

    x_e(a, τ) = Σ_r β_r(a) m_r(τ) e^{i(ψ_r(τ) + φ_e − k'_e·v_r(τ))},  (m, ψ, v)_r(τ) = Σ_c B_c(τ) (m, ψ, v)_{r,c}

- Encodes are centred, k'_e = k_e − k̄ (stored ψ' = ψ0 − k̄·v): the Gauss–Newton blocks of ψ and v decouple exactly
  (Σ_e k'_e = 0), so the per-chunk metric (m: E ω_k c; ψ: E ω_k c m²; v: ω_k c m² Σ k'k'ᵀ) is the block diagonal of the
  true one. Priors and the lock act on the physical ψ0 = ψ' + k̄·v.
- m is a signed real (no log / positivity: the background has m ≈ 0, where a log diverges; a sign flip is the gauge ψ + π).
- Respiratory mixing complex-linear, measured on the truth (predict state 0.181 from 0.07 and 0.493, lumen, cov ≥ 0.99):
  complex-linear 1.42e-2, parameter-linear 1.41e-2 relative (global normalisation): the same physics error, and the
  operator then represents the respiratory direction exactly.

**Operator frames** (`vknot::OpFrames`, research/experiments/vknot/frames_caveat.py → frames_caveat.json; truth for
evaluation only: the 16-knot least-squares fit of the truth velocity and magnitude in the lumen, 6000 voxels, 960 times
per cycle). Error of the operator's signal against the model's, global normalisation (all voxels' signal energy, the
segment rule's), and the decoded encode-difference velocity against the model's own, relative to the dynamic velocity
RMS (the basis error of 16 cubic knots is 0.10 on that scale; the model's physics error against the truth signal is
4.3e-2 global). Cost per encode pass at 2 mm, 30 coils, 30 M samples (temporal_operator.md §22.6: 0.28 s per operator
image + 12.4 s per sample weight); images per respiratory knot, before the respiratory grouping below:

| option | images / resp. knot | weights / sample | signal error (global) | velocity error (rel.) | cost / encode pass |
|---|---|---|---|---|---|
| §E at-knots (complex knot values, no prefilter) | 16 | 8 | — (interp_caveat: velocity 0.55) | 0.55 | 126 s |
| linear frames F 32 / 64 / 128 | F | 4 | 2.1e-2 / 5.4e-3 / 1.4e-3 | 0.051 / 0.011 / 0.0027 | 103 / 157 / 265 s |
| **cubic-spline frames (prefiltered)** F 24 / 32 / 48 / 64 / 96 | F | 8 | 1.8e-2 / 8.3e-3 / **1.4e-3** / 4.6e-4 / 7.8e-5 | 0.061 / 0.030 / **0.0065** / 0.0017 / 0.0002 | 140 / 153 / **180** / 207 / 260 s |
| cubic L2 projection F 48 / 96 | F | 8 | 1.2e-3 / 6.0e-5 | 0.0059 / 0.0002 | as cubic |
| Hermite (value + derivative) F 32 / 48 / 64 | 2F | 8 | 2.2e-3 / 3.9e-4 / 1.7e-4 | 0.0070 / 0.0012 / 0.0005 | 207 / 260 / 314 s |
| cubic on a warped cardiac time (frames dense in systole) F 32 / 64 | F | 8 | 6.7e-3 / 4.6e-4 | 0.029 / 0.0019 | as cubic |

- Cubic-spline interpolation through the frame values (coefficients = the (1/6, 2/3, 1/6)-circulant inverse, a
  pointwise-in-space filter along the frames, its own adjoint) converges as F⁻⁴·³ and beats Hermite at equal image
  count (Hermite needs new operator kernels as well); L2 projection gains 1.2×, a global time warp nothing: not built.
  The per-sample first-order correction of (c) is the Hermite derivative image under another name (the in-between
  phase ω(r)Δt is not low-order in space, unlike the concomitant terms): no cheaper.
- The 1e-4 rule needs F ≈ 96 (7.8e-5). Default F = 48: 1.4e-3 global, 3 % of the model's physics error, 6.5 % of its
  basis error in velocity, comparable to the dev config's segment error (L = 10: 1.38e-3). `--vop frames:96` for the
  rule. The run prints the operator-frame error of its own model (`operator-frame approximation error …`, lumen
  subsample, first and last iterate).
- **Respiratory groups.** The 2-D tables hold < 128 (interval × cardiac knot) keys, so the respiratory knots are split
  into groups of kre knots with (kre − 1) · F < 128: one set per (encode, group), t = e · ng + g; the last group is
  shifted back (it repeats one knot row). 6 knots: F ≤ 25 one group (6 rows), F 26–42 two groups of 4 (8 rows), F 43–63
  three of 3 (9 rows), F 64–127 five of 2 (10 rows). F 48: 9 × 48 = 432 operator images per encode, 4.5× the 96 of §E.
  The extra rows are the price of the table key width; widening the key (the knot tables' u32 entry holds 7 key bits)
  would bring F 48 to 6 × 48 = 288.

**Gradient** (vmodel.cu `vp_*`): per respiratory row, the frame evaluation (grid voxel × frame, ≤ 4 cardiac knots),
its linearisation, and the vector-Jacobian product as one thread per (voxel, cardiac knot) over the ≤ 13 frames that
read the knot (race-free; the residual-phase sum by atomics); the cardiac filter in place on the set image (and on the
adjoint). Host-resident state streams one respiratory row (mc knots) into a θ slot and a d / gradient slot (2 × 1.33 GB
at 2 mm). Tests: vmodel `gpu_matches_cpu_splines` (f64 reference: gradient against central differences 1e-7, GPU
against reference 3e-6, filter dense and truncated), tests/vspline.rs (real 2-D operator, 3 respiratory knots in 2
forced groups: gradient against differences 1.8e-5, streamed against device 1.4e-7, Hessian symmetric, λ_max equal,
recovery 0.069 → 1e-5 m/s on device and host, the complex knot CGLS).

**Robust start** (`--vinit cg:N[:λ]`, default 15; replaces the time-averaged warm decode, 25 % wrapped at --keep 8):
complex knot images in the parameter basis per encode by CGLS through the same operator (the frames are linear in
them), their values at the knot centres (the robust static residual phase removed) through the hierarchical /
dual-VENC decoder per knot, m and ψ0 from the zero-moment value (ψ0 unwrapped along the knots around the mean phasor),
then the cardiac circulant inverse to coefficients. The CGLS needs three knot stacks on the device: run it at a coarse
pyramid level (`--grid-scale 0.5 --kcrop`) and `--init-from` that run at 2 mm (knot_init converts complex ↔ polar runs).

**Priors on by default** (each can be switched off): `--vkappa 1` (LLR on z = m e^{iψ0} and on |m̄|·v), `--dfw sure`
(divergence-free wavelets on v), `--vmwav 1:3` (db4 shrinkage of every magnitude knot, MAD threshold), `--vpoly 1` +
`--vanchor 1,auto` (residual phase with the static-tissue anchor), `--static-lock 20:10:5` (knotlock on m, ψ0 and v:
map from the pseudo-images, m and ψ0 locked as real sets with their own noise levels, `knotlock::apply_real`), VENC
continuation `--vcont 10`.

### F.1 Status / resume (2026-11-18)

- **Chosen:** cubic-spline-interpolated operator frames, F = 48 per respiratory knot (`--vop frames:48`), rows mixed
  complex-linearly along breathing. Why: 1.4e-3 global signal error and 0.0065 of the dynamic velocity (vs 0.10 basis
  error, 4.3e-2 physics error) at 180 s per encode pass (full scan, 30 coils) against 126 s for the §E at-knots model
  whose interpolation leaves 0.55; Hermite costs 2 images per frame for less, linear frames need 4× the frames, L2
  projection / time warps gain ≤ 1.2×. F = 96 meets the 1e-4 rule at 260 s (×2.1 the image side at 6 → 10 rows).
- **Implemented and tested:** see §F (vknot::OpFrames + frame tables, vmodel `vp_*` kernels + f64 reference, vobj spline
  passes on device / row-streamed host state, complex knot CGLS start, knotlock `real_noise` / `apply_real`, recon_p2np
  flags, init conversion polar ↔ complex for --init-from, in-run operator-frame error). Tests: `cargo test --release -p
  hasty-recon --lib vknot:: vmodel::` and `--test vspline` pass; the old at-knots path (`--vop knots`, tests/vknot.rs)
  is unchanged in behaviour.
- **Priors on by default:** LLR κ 1 on m e^{iψ0} and |m̄|·v, divergence-free wavelets (sure) on v, db4 magnitude
  shrinkage (1:3), residual phase degree 1 + static anchor (1, auto), knot lock 20:10:5 (m, ψ0, v), VENC continuation 10.
- **Pre-registered** (research/experiments/vknot/dev_run2.sh): does the velocity model with priors recover the encode
  1 − 0 velocity phase at least as well as complex knots with their own wavelet + LLR prior (ncgcap), --keep 8?
  Decision: B2 ≥ A2 (slope and corr, deep lumen and dynamics, ≥ 2 of 3 breathing states) → velocity model is the path;
  B2 < A2 only in the dynamics → operator frames / basis / v-prior strength; B2 < A2 everywhere → init or solver.
- **Results so far** (score_pair, encode 1 − 0 slope / corr, it 40; deep lumen = depth ≥ 5, dyn = dynamics depth ≥ 2):

  | arm | r 0.07 deep / dyn | r 0.181 deep / dyn | r 0.493 deep / dyn |
  |---|---|---|---|
  | A complex knots, CGLS 40, no prior | 0.106/0.024, 0.044/0.010 | 0.171/0.045, 0.078/0.019 | 0.159/0.046, 0.069/0.017 |
  | A2 complex knots, ncgcap (wavelet + LLR), 41 gradients | 0.375/0.227, 0.242/0.100 | 0.433/0.299, 0.270/0.127 | 0.402/0.276, 0.218/0.097 |
  | B2 velocity model | queued | queued | queued |

- **Queued** (one systemd unit, arms serialised on the GPU lock): `vknot_dev2_B2` runs
  research/experiments/vknot/dev_run2_chain.sh → B2a (log data/hasty/phantom/factory_arms/ph0001/vknot/dev_k8/B2a.log,
  out …/dev_k8/B2a: 4 mm grid-scale 0.5 + kcrop, --keep 8, 12 virtual coils, nseg 10, 40 iterations, --vinit cg:15,
  λ 6 power iterations) → B2b (…/B2b.log, …/B2b: 2 mm, --init-from B2a, 20 iterations, λ 4, --eval at the three
  breathing states) → score (…/dev_k8/scores2.txt). Binary snapshot …/dev_k8/recon_p2np_vspline. Config vs target
  (temporal_operator.md §21): basis 6:1,16:3 bellows = target; coils 12 fixed (82.8–90.9 % energy, target ≥ 99.9 %);
  --keep 8 (1/8 of the readouts); nseg 10 fixed (segment error 1.38e-3, target rule 1e-4); W 10, physics full + GNL +
  deg4 = target; B2a additionally a 4 mm pyramid level.
- **Pause / resume:** `touch …/dev_k8/B2a/STOP` (or B2b/STOP); resume = rerun the unit:
  `systemd-run --user --unit=vknot_dev2_B2 --collect -p MemoryMax=60G -p MemorySwapMax=0 --working-directory=$PWD
  research/experiments/vknot/dev_run2_chain.sh <gpu.lock>` (remove the STOP file first; logs are appended).
- **Score:** `research/experiments/vknot/dev_run2.sh score` (score_pair on A, A2, B2b + the in-driver RMSE lines).
- **Open:** the end-to-end driver path ran only to the model construction in a smoke test (a K ≤ 64 assertion fixed;
  the 8 mm smoke `vknot_smoke` was re-queued behind the GPU lock); B2a is the first full run. Cost of F 48 at 2 mm is
  ~4.5× the §E image side; widening the 2-D table key would remove the respiratory-group duplication (9 → 6 rows).

### F.2 Smoke debugging (2026-11-18)

8 mm smoke (grid-scale 0.25 + kcrop, --keep 8, 8 virtual coils, nseg 6, --vcont 2, --vinit cg:3, --static-lock
20:2:1, 4 iterations; scratchpad vdbg/run.sh). The `--vop` default is now **frames:16** (cubic; coordinator decision:
the 1e-4 rule is the NUFFT's only, and 16 frames give 4.4e-2 operator-frame error, the size of the model's physics
error 4.3e-2). New in-driver diagnostics, each iteration: `score split` (lumen RMSE in the upper half of the model
magnitude inside the support, its per-axis slope, the unwrapped components, the voxels outside the support) and
`step diag` (on ≤ 20 000 support voxels × all knots: encode phase max_e |k'_e·Δv| of the gradient step and of the
prior/lock, share of |v| beyond the box, phase per operator frame between cardiac knots). `HASTY_ZMSHR1_LOG=1` logs the
ZMSHR1 scalars (τ, a, rank-one c) at every iteration instead of every 10th.

**1. NaN objective: reporting only.** p2np/hoststack call the report with f = NaN for a stage's initial state (no
gradient evaluated yet); f in every later line is the data term + anchor of the iterate *before* the reported one.
No NaN in the state or gradient (every later f finite, state finite). "anchor λ 0" in stage 0 is `--vanchor …,auto`
switching the anchor off until the pulsatility refinement after stage 0. The line now reads `f(previous iterate) =
— (initial state, not evaluated)` and the stage-0 anchor line says it is off.

**2. Velocity RMSE > 1 at the start: noise outliers of the per-knot decode, conventions verified.**
- Conventions: the per-voxel start chain is now `vinit::SplineStart` (the driver calls it) with a CPU test
  `vinit::tests::spline_start_recovers_model_coefficients` on model-consistent values of the actual coprime-7 design
  (VENC 1.0 / 1.5, unaliased ±3, |v| to 2.2 m/s, residual phase per encode): knot-centre values, decoder sign / VENC,
  zero-moment phase ψ0 = arg z_ref + k_ref·v, centring ψ' = ψ0 − k̄·v and the circulant inverse give back the model's
  coefficients to 1e-4 and x_e = m e^{i(ψ' + φ_e − k'_e·v)} reproduces the encode values. No sign or VENC error.
- Cause: the knot values are noise-dominated. The warm image alone (4 PCGLS, --keep 8, 8 mm) has a circular std of the
  encode phase differences > 1 rad in half the support (deciles of |x_0|: 2.1 rad lowest … 0.19 rad highest); the per-knot
  CGLS images carry 1/96 of the data each. The decoder's alias search over ±3 m/s then returns lattice outliers
  (wrapped 5 %, |v| > 3 m/s in 3.8 % of the voxel-knots), which alone exceed the zero predictor (rel 1.43).
- Not a fixable accuracy floor at 8 mm: the upper half of the lumen by model magnitude has slope 0.015–0.02 in every
  variant (start and after iterations): the 8 mm lumen voxels carry almost no velocity phase (most likely partial
  volume of 2–5 mm vessels in 8 mm voxels; not measured against the truth fraction). At this level the RMSE checks
  stability, not accuracy.
- Fix (`--vinit-snr`, default `design` = the sequence's SNR of a 1e-3 unwrap failure, meta
  `snr_unwrap_1e-3` = 3.28): a knot is decoded only where its SNR (mean encode magnitude / per-voxel noise from the
  dispersion of the encode magnitudes, median over the knots) reaches the gate; else the voxel's time-mean decode
  (noise σ/√K); else 0. Decoded values and coefficients projected onto the box (below). Smoke: 60 % of the support
  voxel-knots decoded per knot, 40 % from the time mean; start rel 1.430 → 1.363, wrapped 4.98 → 3.64 %, |v| > 3 m/s
  3.8 → 0 %, operator-frame error of the start 1.08e-1 → 3.9e-2 (p99.9 1.65 → 0.75).

**3. Blow-up at iteration 4: an SR1 extrapolation in the first quasi-Newton step after the VENC switch.**
- The step, not the prior: at the second iteration of stage 1 the ZMSHR1-M rank-one coefficient is c ≈ −7.2 (τ 0.74;
  stage 0: c −0.65), i.e. the step extrapolates ~7× along the previous step. The gradient step's encode phase exceeds
  1 rad in 54 % and π in 21 % of the voxel-knots (max 282 rad); the prior/lock only follows (p99 25 rad after a 41 rad
  step). Unbounded v then lands on aliases and the operator frames cannot represent it (op-frame error 2.9e-1, p99.9
  1.75; velocity rel 3.35).
- Lock / continuation: neither causes it. `--static-lock off`: the same c (−8.1) and the same jump. `--vcont 0` (all
  encodes from the start): c −2.7 and −1.5 at its 2–3, gradient step > 1 rad in 18–24 % of the voxel-knots (max 276
  rad), op-frame error 1.87e-1 at it 4: the overshoot comes with the low-VENC encodes (k' up to 2.39 rad/(m/s));
  continuation only concentrates it at the switch. The SR1 model of a cost that is periodic in k'·v has negative curvature beyond ±π/2: SR1
  needs a trust region (Nocedal & Wright §6.2), the P²nP driver had none.
- Fixes: `--vtrust RHO` (default 1 rad; `Objective::limit_step[_host]`, used by both solvers, VelProblem only): per
  (voxel, knot) the step's velocity 3-vector is scaled to max_e |k'_e·Δv| ≤ ρ (the region where the Gauss–Newton
  phase linearisation holds, cos error 4 % at 1 rad), and spline mode clips |Δψ'| ≤ ρ; `--vbox V` (default design =
  unaliased ±3 m/s): projection of the v coefficients after the prior (indicator of the design range; the B-spline
  convex hull keeps the curve inside). The secant pair stays the actual x − x_prev. CPU test
  `vobj::tests::trust_shrink_scales_per_voxel`. The old paths keep their behaviour unless `--vtrust` is given
  (`--vop knots`: no box, no trust region by default).

| smoke, frames:16 | it 0 rel / wrapped | it 4 rel | op-frame error it 0 → last (p99.9) | max step phase | f at it 6 → it 7 |
|---|---|---|---|---|---|
| before (no gate / box / trust region) | 1.430 / 4.98 % | 3.348 | 1.08e-1 → 2.87e-1 (1.75) | 282 rad | — |
| after (defaults), 4 iterations | 1.363 / 3.64 % | 1.204 | 3.86e-2 → 1.76e-1 (1.43) | 1 rad (47 % clipped at it 3) | — |
| after (defaults), 8 iterations | 1.363 / 3.64 % | 1.210 | 3.86e-2 → 1.32e-1 (1.00) | 1 rad; > 1 rad share 21 → 0.4 % | 1.43e14 → 1.40e14 |
| --vtrust off (gate + box only) | 1.364 / 3.64 % | 1.433 | 3.92e-2 → 2.44e-1 (1.61) | 359 rad | — |
| --vtrust off --static-lock off | 1.369 / 3.69 % | 1.495 | 3.94e-2 → 1.33e-1 (1.48) | 305 rad | — |
| --vtrust off --vcont 0 | 1.369 / 3.69 % | 1.349 | 3.87e-2 → 1.87e-1 (1.60) | 276 rad | — |

(The first smoke at frames:48: rel 1.437 → 3.016, op-frame 3.45e-2 → 1.27e-1.) After the fixes the 8-iteration run
decreases f monotonically within stage 1 (2.60e14 → 1.40e14) and the velocity RMSE to rel 1.133 (it 8); the share of
clipped voxel-knots decays to 0.4 %.

**Open.** The operator-frame error still grows over the iterations (3.9e-2 → 1.3e-1 at 8 mm, F 16): the knot
velocities are temporally white at this SNR (cardiac spectrum flat; B2a at 4 mm after 40 iterations too: 5 % of the
knot-to-knot phase steps > π), and one frame per knot cannot follow > 1 rad per knot interval. A temporal regulariser
on v (or the planned 12 → 16 → 24 frame pyramid) is the lever; the trust region bounds only the step. B2a (4 mm)
ran without these fixes.
