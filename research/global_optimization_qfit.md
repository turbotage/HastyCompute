# Global optimisation for `hasty-qfit`: IVIM, SHGO and what else (research note, 2026-11-28)

Question (user): research IVIM estimation and global optimisation (SHGO and newer), compare with what `hasty-qfit`
already has, say what must be added, and which qMRI sequences/methods benefit.

Status: research only. Nothing built or run. Claims are tagged **[checked]** (read in this session, source given),
**[code]** (read in the repo today) or **[mine]** (my reasoning/estimate, not measured).

## 1. Literature

### 1.1 SHGO

Endres, Sandrock, Focke, *A simplicial homology algorithm for Lipschitz optimisation*, J. Glob. Optim. 72 (2018)
181–217 [checked: [paper](https://repository.up.ac.za/bitstream/2263/64460/1/Endres_Simplicial_2018.pdf),
[docs](https://shgo.readthedocs.io/en/latest/docs/README.html)].

- Samples the box, builds a simplicial complex on the samples, evaluates $f$ at the vertices, and takes the vertices
  with no lower neighbour in the 1-skeleton as the *minimiser pool*. Each pool member (or a subset) starts a local
  solver. Sperner's lemma / homology groups are the argument that the number of locally convex sub-domains found is
  stable as the sampling is refined.
- Output is **all** local minima found, not one. This is the feature that matters for IVIM.
- Convergence guarantee: for Lipschitz $f$ with the default *simplicial* sampling (finite time). The Sobol option is
  faster in their benchmarks but loses the guarantee.
- Practical limit about 10 dimensions, high RAM beyond (author, scipy-dev thread). Constraints are supported.
- `scipy.optimize.shgo` is the reference implementation; `rustlib/.venv` has scipy 1.18 [code], so a CPU oracle exists.

### 1.2 IVIM + SHGO is already published

- Fadnavis et al., *Fitting IVIM with variable projection and simplicial optimization*,
  [arXiv:1910.00095](https://arxiv.org/abs/1910.00095) (DIPY `fit_method='VarPro'`, extends MIX of Farooq 2016):
  VarPro removes the linear fraction, SH or DE searches the exponents, then NLLS on all parameters. Reported: SH maps
  ≈ DE maps, SH faster (≈2×) [checked, abstract-level via search; I could not read the PDF body].
- Fadnavis et al., *Bifurcated Topological Optimization for IVIM* (TopoPro), Front. Neurosci. 2021,
  [PMC8714828](https://pmc.ncbi.nlm.nih.gov/articles/PMC8714828/) [checked]: VarPro level 1 (eliminate $f$), SHGO on
  the reduced $(D, D^*)$ functional to get all minima, level 2 refinement on the full functional. Returns several
  candidates per voxel and *selects* with penalty functions (e.g. penalise large $D^*$ at small $f$). Reported ≈10×
  lower RMSE for $D^*$ and $f$ on a simulated Shepp–Logan IVIM phantom (SNR 2–50); real data: two brain sets;
  runtime not reported. Example bimodal voxel at $f=0.349$: one valley $D\approx0.005$, another $D^*\approx0.04$ mm²/s.
- **Caveat [mine]:** the penalty is a prior, not a global-optimisation result. The 10× comes from selection plus
  prior on a phantom; it does not mean the global MLE is that good. At low SNR the global MLE itself has a large
  variance in $D^*$ (flat valley). Global search guarantees reliability of the *estimator*, accuracy needs a declared
  prior or spatial coupling. This matches the repo rule "never change the statistics": a penalty must enter as an
  explicit log-prior with a stated meaning.

### 1.3 Other IVIM estimators (context)

- Six-algorithm comparison, pancreas (Gurney-Champion 2018, PLoS ONE, [link](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0194590)):
  Bayesian best on precision, not significantly better than a fixed-$D^*$ segmented fit [checked via search].
- Brain, flow-compensated + non-compensated (Jalnefjord 2024, MRM): Bayesian with spatial prior best for
  repeatability/accuracy; deep learning noise-robust but biased [checked via search].
- Kaandorp et al. 2026, JMRI [link](https://onlinelibrary.wiley.com/doi/full/10.1002/jmri.70301): Bayesian > LSQ/segmented,
  spatially-aware transformer networks lower error again; IVIM-NET_optim biased in $D^*$ on synthetic data [checked via search].
- Deep-ensemble mixture-density IVIM for uncertainty ([arXiv:2508.04588](https://arxiv.org/pdf/2508.04588v1)),
  motivated by prior-induced bias [checked via search].
- Take-away [mine]: the field's accuracy gains came from priors (Bayesian, spatial, learned), not from better
  optimisers. A global optimiser is the foundation (find all modes, know if the fit is ambiguous); a spatial/Bayesian
  stage on top of the *candidate set* is where accuracy is gained.
- Not verified here: OSIPI TF2.4 IVIM challenge results (search returned nothing). Look before benchmarking.

### 1.4 GPU / global optimisation tools

- DIPY's VarPro = DE on exponents + convex NNLS + NLLS [checked]. MDT (Harms 2017, NeuroImage 155) ran NODDI etc.
  on GPU and found derivative-free Powell more accurate/robust than LM there [checked via search, details unread];
  it is the closest published "millions of small problems on a GPU" system.
- GPU-native population methods (EvoX; "GPU DE with gradient polishing", SSRN; ChiSao arXiv:2606.26164;
  "Beyond speedups" arXiv:2601.18446) target **one** large black-box problem and parallelise the population
  [checked via search]. None targets $10^6$ tiny independent problems, which is the regime `hasty-qfit` is built for.
  So no off-the-shelf competitor; build, don't adopt.
- From memory, not re-checked: MLSL (Rinnooy Kan–Timmer) = multistart + clustering to avoid repeated local solves;
  TikTak (Arnoud–Guvenen–Kleineberg 2019) = Sobol screening then local solves from the best, then re-mixing;
  DIRECT/DIRECT-L; basin hopping; CMA-ES (collapses at $d\gtrsim8$ in the ChiSao benchmark [checked via search]);
  AMICO (Daducci 2015, NODDI as dictionary + convex fit) = the same profile idea as below.

## 2. What the repo already has (all [code])

| piece | where | global-optimisation meaning |
|---|---|---|
| per-thread LM, full Hessian, SE99/GMW81, active-set bounds, $S$-scaling, statistical stop | `lm.rs`, `gpu.rs` (`qfit_lm`) | the *local* solver every hybrid needs; unusually good (second order, bounded, statistically meaningful stop) |
| multi-start on device-resident data | `GpuData`, `fit_on`, `fit_dev` | multistart infrastructure exists; starts come from the caller |
| custom objective hook | `GpuFit::with_source` | any `objective`/`derivs` device code (VarPro, Rician NLL, …) |
| shared-node profile likelihood: per node orthonormal basis $Q$ of the linear part, $R=\lVert y\rVert^2-\lVert Q^H y\rVert^2$, top-$K$ local minima over $\omega$ of $\min_{R_2^*}R$, LM start from each | `waterfat.rs` (`Profile`, `wf_profile`, `wf_select`) | a 1-D **lattice SHGO / VarPro / dictionary match** already exists, GPU-resident, specific to water–fat |
| Fisher information, CRLB | `stats.rs` | identifiability diagnostic: tells which voxels/designs need global treatment |
| symbolic DAG with exact gradient and Hessian, CSE | `expr.rs`, `model.rs` | gives, for free, reduced-gradient (VarPro/Kaufman) derivatives, change of variables, interval/Lipschitz bounds |
| CPU reference path (`fit_cpu`) | `fit.rs` | test oracle |

Observations [code]:

1. The IVIM test (`tests/qfit.rs::ivim`) is **noise-free** and starts at one fixed point. It shows 1/2000 GMW81 swaps
   ($D\leftrightarrow D^*$), 0 for SE99. It says nothing about noisy, flat-valley behaviour, which is the real IVIM
   problem. No noisy-IVIM or basin-error benchmark exists.
2. `wf_profile` looks for local minima only along $\omega$ and takes the min over $R_2^*$ inside. Fine for water–fat
   (periodic in $\omega$); IVIM needs minima over a 2-D $(D,D^*)$ lattice.
3. The linear amplitudes in the profile are unconstrained (complex). IVIM needs non-negative amplitudes.
4. There is no generic candidate-set object, no deduplication of LM end points, no ambiguity measure.
5. Constraints are boxes only ($D^*>D$ must be a reparametrisation or a lattice mask).
6. Objective is weighted least squares. Magnitude DWI is Rician; fine at SNR ≳ 3–5, biased below. The repo rule is
   "complex data, not magnitude" (valid for the k-space pipeline; stock DWI is magnitude).

## 3. Design

### 3.1 Central idea: shared-topology lattice SHGO on the profile likelihood

SHGO's expensive parts (building the complex, hash tables, recursion) are hostile to per-voxel GPU threads. But every
voxel shares the same box and the same sample points; only the $f$ values differ. So the simplicial complex is
**static and shared**; per voxel one needs only (a) the profile values on the nodes, (b) a local-minimum test on the
fixed graph. That is exactly the structure `wf_profile` already uses in 1-D.

For a separable model $y\approx\Phi(\theta)\,c$ with linear $c$ (amplitudes) and nonlinear $\theta$ ($d$ of them):

$$R(\theta)=\min_{c\in\mathcal C}\lVert W^{1/2}(y-\Phi(\theta)c)\rVert^2 .$$

For unconstrained $c$, $R=\lVert y\rVert_W^2-\lVert Q(\theta)^H W^{1/2}y\rVert^2$ with $Q$ shared by all voxels, so
evaluating all nodes for all voxels is a **GEMM** ($Q$ stacked: nodes·atoms × samples, times data: samples × voxels).
For $c\ge0$: if the unconstrained amplitudes are non-negative that is the answer, else the optimum lies on a face
(single-atom fits, closed form $\langle\phi_k,y\rangle_+^2/\lVert\phi_k\rVert^2$); for ≤3 atoms this is a few flops.
Minimisers of $R$ over $\theta$ are the same $\theta$ as the full MLE (profile likelihood), so no statistics change.

Steps:

1. Node set: tensor lattice over the box (or masked by constraints), Kuhn/Freudenthal triangulation neighbours
   ($2(2^d-1)$ per node: 6, 14, 30 for $d=2,3,4$) = SHGO's 1-skeleton on a lattice. Spacing from the model's own
   scale (as `Grid::for_echoes`), not a free knob; for a certified spacing see 3.4.
2. Per voxel: $R$ at nodes, local minima = nodes with no lower neighbour (tie-break by index), keep top-$K$.
3. Each candidate: amplitudes by closed-form/NNLS, then existing LM on the full parameter vector (`fit_dev`), best
   $f$ wins, but **all** distinct end points are retained with $f$ and Hessian eigenvalues.
4. Output: `Candidates { params[K][np][n], f[K][n], gap = f_2nd − f_best, hess_min_eig, basin_id }` plus the
   profile on request.

Cost [mine, unmeasured]: IVIM, $64^2$ nodes × 13 b-values × ~8 flop ≈ $4\times10^5$ flop/voxel → $4\times10^{11}$ for
$10^6$ voxels, tens of ms at realistic GEMM throughput; the LM polish (23 ms for $10^6$ voxels already, [code]/plan doc)
is then the same order. $d=3$ at 48³ nodes ≈ 1e5 nodes → ~0.3–1 s per $10^6$ voxels; $d=4$ at ~20⁴ similar. Beyond
$d=4$ use hierarchical refinement (3.3). Output tensor nodes × voxels must be fused/tiled (memory ledger rule).

### 3.2 Nested/partially separable problems

Multi-scan problems (T1 series with shared B1, Gd time points, multi-contrast) are $\sum_t g_t(\theta_t,\beta)$ with
shared $\beta$. Profile over $\beta$ lattice, and for each $\beta$ node solve each $\theta_t$ separately (1-D each):
cost adds, not multiplies. This turns an apparently 6-D global problem into a 1-D lattice × small 1-D problems.

### 3.3 When it is not separable or $d>4$

- Hierarchical per-thread refinement: coarse shared lattice → top-$K$ cells → local sub-lattice or Sobol points per
  voxel → LM. (TikTak/SHGO-Sobol flavour; Q no longer shared, small Gram solve in-thread for ≤3 atoms.)
- Multistart + deduplication of end points (MLSL-style) as the generic fallback; needs a Sobol/Halton generator and an
  end-point clustering kernel.
- DE/CMA-ES per thread: low priority. Reasons: only competitive when no structure is available; CMA-ES degrades from
  $d\approx8$ [checked via search]; the repo's models are almost all separable.
- Not planned: PSO and learned fits as optimisers. A network could supply the *prior* later.

### 3.4 Unique to this repo: certified grids

The expression DAG gives $\nabla f$, $\nabla^2 f$ symbolically; interval evaluation over a box gives a Hessian bound
$\lVert\nabla^2R\rVert\le H_{\max}$ (αBB-style) and a Lipschitz constant. Then (i) lattice spacing guaranteeing that
no basin of width above a stated size is missed, (ii) design-time proof that a protocol has no spurious basin above a
tolerance. Research-grade, design-time only, not per voxel. Optional.

### 3.5 Outputs that matter more than the optimiser

- **Ambiguity flag**: $\Delta f=f_2-f_1$ in units of $\hat\sigma^2$ (a voxel is ambiguous when two distinct basins
  differ by less than a few $\sigma^2$). Spatial regularisation (ICM/graph cut over candidates) uses it, as the
  waterfat docs already anticipate.
- **Grid posterior**: $p(\theta\mid y)\propto e^{-R(\theta)/2\sigma^2}\pi(\theta)$ on the lattice gives mean/credible
  intervals and a proper Bayesian IVIM for free. Note: profile ≠ marginal; marginalising flat-prior amplitudes adds a
  factor $\det(A^HWA)^{-1/2}$. Document which is reported.
- **Model order**: mono vs bi vs tri-exponential by BIC/likelihood ratio on the same lattice machinery.
- Priors enter only as declared $\log\pi$ terms (the TopoPro penalty becomes one, with a stated meaning).

### 3.6 Noise models: complex (own data) and magnitude (external data)

Both are required (user, 2026-11-28). Both enter as a per-sample **loss** $\varphi(A_i(\theta);y_i,\sigma)$ instead of
the current hard-wired $\tfrac12 w r^2$; the solver needs only $\varphi'$, $\varphi''$ in the model output $A_i$:
$g=\sum\varphi'\,\partial_\theta A$, $H=\sum[\varphi''\,\partial A\,\partial A^T+\varphi'\,\nabla^2A]$ (the existing
residual form is the special case $\varphi=\tfrac12 w(A-y)^2$).

- **Complex, Gaussian** (own sequences): current path, unchanged.
- **Magnitude, Rician** (external IVIM/T1): $\varphi=\dfrac{A^2}{2\sigma^2}-\log I_0\!\left(\dfrac{yA}{\sigma^2}\right)$,
  $\varphi'=\dfrac{A}{\sigma^2}-\dfrac{y}{\sigma^2}\rho(z)$, $z=yA/\sigma^2$, $\rho=I_1/I_0$;
  $\varphi''=\sigma^{-2}-\sigma^{-4}y^2\,(1-\rho/z-\rho^2)$. $\varphi''$ goes negative at low SNR: the loss is non-convex in
  $A$, which is what modified Cholesky (SE99) is for. Needs a stable $\log I_0$ and $\rho$ in f32 (rational
  approximations; test against f64/scipy). Multi-coil sum-of-squares data: non-central $\chi$ with effective coil
  count $L$, same machinery with $L$ as a problem input. After parallel-imaging reconstruction neither holds exactly:
  $\sigma$ is spatially varying; take it as a per-voxel input, never a global constant.
- **Rule from the existing scaling doc, extended:** per-voxel RMS normalisation (as in `waterfat.rs`) keeps the MLE for
  Gaussian only if $\sigma$ is scaled by the same $c$. For Rician, $\sigma\to\sigma/c$ must go with it.
- **Profile vs marginal over a nuisance phase** [mine, algebra]: $\min_\phi|y-Ae^{i\phi}|^2=(|y|-A)^2$, so profiling a
  per-measurement phase gives least squares on magnitudes (biased at low SNR, incidental-parameter problem), whereas
  marginalising $\phi$ with a uniform prior gives exactly the Rician. For the own complex data prefer a smooth low-order
  phase map per preparation (few parameters, already planned) over per-sample phases; for external magnitude data use
  Rician.
- **Global stage for magnitude data**: the closed-form amplitude elimination is Gaussian-only. Plan: Gaussian profile
  (optionally on $y^2-2\sigma^2$ moment-corrected data, Koay–Basser style, from memory) to screen nodes, keep a larger $K$,
  Rician LM polish, **rank by the Rician NLL**. Risk: a Rician basin missed by the Gaussian screen. Measured by the
  wrong-basin metric against a dense Rician oracle (benchmark below), not assumed.
- Magnitude **inversion recovery** (T1 from external data) has the polarity-loss ambiguity $|A(1-2e^{-TI/T_1})|$:
  multimodal by construction, a good global-search test case for the same machinery.
- `stats.rs`: Rician Fisher information is below the Gaussian one at low SNR (factor depends on $A/\sigma$); the CRLB
  function needs a per-sample information factor to be honest for magnitude data.

### 3.7 Spatial priors: splitting around the per-voxel solve

Target: $\min_\theta\sum_v f_v(\theta_v)+R(\theta)$ with $R$ acting on whole parameter volumes (TV, wavelets, the
learned patch denoiser, a smooth low-order B1 model, sparse-deviation priors near implants). The scheme the user
described is half-quadratic splitting / ADMM, not ISTA (ISTA takes a gradient step on the data term; here the
data term has a cheap exact local solve, so use its proximal map):

$$\theta_v^{k+1}=\arg\min_\theta\; f_v(\theta)+\tfrac{\rho}{2}\lVert\theta-a_v^k\rVert_\Lambda^2,\quad a^k=z^k-u^k,$$
$$z^{k+1}=\mathrm{prox}_{R/\rho}\,(\theta^{k+1}+u^k)\ \ (\text{denoiser}),\qquad u^{k+1}=u^k+\theta^{k+1}-z^{k+1}.$$

$u\equiv0$ is HQS (increase $\rho$ over iterations); $u$ kept is scaled ADMM. The per-voxel problem is the same in both:
**an anchor $a_v$ and a precision $\rho\Lambda$ added to the local objective.** This is the whole interface to `qfit`.
Properties:

- **Fits the existing solver.** Gradient gets $+\rho\Lambda(\theta-a)$, Hessian $+\rho\Lambda$, Marquardt diagonal
  $+\rho\Lambda$; $a$ and $\rho\Lambda$ are data/arguments, so no NVRTC recompile per outer iteration. Cheapest prototype:
  extra residual rows $\sqrt{\rho\Lambda_j}(\theta_j-a_j)$ in the symbolic model with $a_j$ as `fixed` inputs. Clean version: a
  first-class anchor in `lm.rs`/`gpu.rs` so the statistical stop uses the data part only. $\Lambda=\mathrm{diag}(S^{-2})$
  in the solver's scale $S$ keeps $\rho$ dimensionless and consistent with the scaling doc. Priors on transformed
  parameters ($\log D^*$, logit $f$) are fine (bijection).
- **The global stage sees the anchor.** At lattice nodes add $\tfrac{\rho}{2}\lVert\theta-a_v\rVert^2_\Lambda$ to the profile:
  the search stays global *including* the prior, so ambiguous voxels are re-decided by their neighbourhood each
  outer iteration (the spatial selection TopoPro does with hand-made penalties). Large $\rho$ makes the local problem
  unimodal, so later iterations are warm-started LM only; HQS with growing $\rho$ is also a continuation method
  against local minima.
- **Elimination interacts with priors:** a parameter eliminated by VarPro (an amplitude, or $f$ if expressed through
  amplitudes) cannot carry an exact closed-form anchor. Either keep it as a lattice axis (IVIM: $(f,D,D^*)$ lattice,
  $S_0$ linear: one extra axis ≈ ×16 nodes) or apply its prior only in the LM stage. Choose the eliminated set by which
  parameters need spatial priors.
- **Separation of concerns:** `qfit` stays voxel-batched with no geometry. The outer loop and the `VolumePrior`
  (`prox(&mut volume, ρ, mask)`) live a level up (recon side), where the denoisers and the P²nP/RED machinery
  already are ([[pnp-theory-direction]] notes: convergence needs care for non-convex $f_v$ and non-expansive/bounded
  denoisers; check before claiming).
- **Candidate handling:** iteration 0 with $\rho=0$ yields the candidate set and ambiguity map; later iterations
  recompute the profile with the anchor (tens of ms to ~1 s per $10^6$ voxels [mine], no need to store the node×voxel
  table) rather than caching candidates.
- Same loop serves the Gd-T1 case: $B_1$ prior = projection onto a smooth low-order space plus robust sparse
  deviations; $R_1$ across time points shares it (the nested profile of 3.2 per outer iteration).

## 4. What to add (ordered)

| # | item | why | size |
|---|---|---|---|
| P0 | generic separable profile (`global.rs`): node table, atoms symbolic from `Model`, NNLS/LS amplitude solve, nD Kuhn minima, top-$K$, LM start generation; generalises `wf_profile` (water–fat becomes an instance) | IVIM, T1-IR, multi-T2, resonance fits | main work |
| P0 | `Candidates` + end-point dedupe + ambiguity gap + Hessian eigenvalues out of `fit_dev` | selection and spatial stage need it | small |
| P0 | benchmark/oracle harness (below) | no noisy-IVIM test exists | small |
| P1 | grid posterior / profile intervals output | uncertainty without MCMC | small |
| P1 | nested profile (shared $\beta$) | Gd T1 series, multi-scan | medium |
| P1 | reduced (VarPro) LM in $\theta$-space, Kaufman Jacobian from the DAG | better conditioned polish, TopoPro level 1 | medium |
| P2 | Sobol + hierarchical refinement, MLSL clustering | $d>4$, non-separable | medium |
| P2 | linear/nonlinear inequality constraints (masks on lattice; reparametrisation docs: bijection keeps MLE, changes box/scale) | $D^*>D$, ordering | small |
| P0 | per-sample loss abstraction ($\varphi,\varphi',\varphi''$) replacing hard-wired least squares; Rician and non-central $\chi$ losses, stable $\log I_0$/$I_1/I_0$ in f32 (device + CPU), per-voxel $\sigma$ (and $L$) inputs, Rician-aware CRLB | external magnitude IVIM/T1 is a hard requirement (user) | medium |
| P0 | first-class anchor term ($a_v$, $\rho\Lambda$) in `lm.rs`/`gpu.rs`, also added to lattice profile nodes | interface for any outer spatial prior (3.7); tiny change | small |
| P1 | outer HQS/ADMM driver + `VolumePrior` trait (above `qfit`, recon side), first priors: TV/wavelet, smooth low-order B1 projection; later the learned denoiser | spatial priors, user wants room for them | medium |
| P3 | interval/αBB certified lattice spacing | design-time proofs | research |
| — | per-thread DE/CMA-ES/PSO | no structure case only | skip |

## 5. Pre-registered benchmark (before any GPU run)

Question: does profile-lattice + LM reduce wrong-basin rate and $D^*$ error versus the practice baselines at equal
or lower cost, and does the ambiguity flag identify the failures?

- Truth: IVIM brain-like $(f,D,D^*)$ grid incl. low $f$ (0.03–0.1, brain), $D\in[0.5,1.2]\times10^{-3}$,
  $D^*/D\in[3,30]$; b-set = the planned sequence's b-values (13-value set only as the existing test); SNR 10–100 at
  $b=0$; Gaussian and Rician noise; complex noise-free and inverse-crime controls first.
- Oracle: dense brute-force profile on a fine grid, plus `scipy.optimize.shgo` / `differential_evolution` from
  `rustlib/.venv` on a subsample (CPU) to cross-check basin lists.
- Baselines (practice): fixed-start LM, 10-start LM, segmented (threshold 200), fixed-$D^*$ segmented, DIPY VarPro
  (DE), Bayesian with flat prior.
- Metrics: wrong-basin fraction (against oracle global minimum of the *noisy* $f$), bias/RMSE vs the true
  parameters, ratio to the CRLB where Fisher is well conditioned (`stats.rs`), ambiguity-flag ROC against wrong-basin,
  time per $10^6$ voxels. One factor at a time.
- Decision: adopt P0 if wrong-basin rate falls by a stated factor at ≤2× the LM-only time; otherwise report why.

## 6. Which qMRI problems benefit

Criterion [mine]: separable with ≤3–4 nonlinear parameters (lattice profile is exact and cheap), known multimodality,
or flat low-SNR likelihoods. Ordered by fit to this repo's planned work.

**In the current plan**

1. **IVIM / flow-compensated vs non-compensated IVIM** (brain, `t1_ivim_dti_sequences.md` §3.3): nonlinear
   $(D, D^*, v_d)$ or $(D,D_b,\ldots)$, linear fractions → 2–3-D lattice. The textbook case.
2. **T1 with free inversion efficiency / B1 / T2 (EPG dictionary)**, intrathecal Gd: nonlinear $(R_1,\eta\ \text{or}\ B_1,T_2)$,
   linear $M_0$. The planned subspace-then-voxel fit is a dictionary match = profile lattice; the Gd time series with a
   shared B1 is the nested case (3.2). CRLB already showed fixed-flip IR is singular and CSF T2 couples to $R_1$;
   global/ambiguity output tells where the fit is trustworthy.
3. **GIRF resonance fitting** (`girf_qa_sequence.md`): resonance frequency $f_0$ (needed to ≤0.2 Hz) and damping are
   nonlinear, amplitudes linear; classic multimodal sum-of-resonances. Resonance survey then targeted pass = coarse
   lattice then refinement. Strong fit and in scope.
4. **B0 / R2\* / water–fat**: already done in 1-D over $\omega$; extend to joint $(\omega,R_2^*)$ minima and to fat
   spectrum nuisance. Becomes the first instance of the generic code.
5. **4D flow velocity encode aliasing** (per-voxel $v$ vector, complex amplitude linear): periodic multimodality like
   $\omega$ wrap. Only relevant to initialisation/dual-VENC checks; the model-based recon may already avoid it. Not
   verified, check before investing.

**Would benefit if ever in scope** (the user said not to stray into general qMRI, listed for completeness)

6. Multi-component T2 / myelin water (separable, ill-conditioned, EPG stimulated echoes).
7. NODDI / SANDI / multi-compartment diffusion (known multimodal/degenerate: Jelescu 2016, Harms 2017 from memory;
   AMICO-style convexification = this profile idea).
8. DCE-MRI (Tofts, 2CXM), ASL multi-PLD kinetic model (ATT nonlinear, CBF linear): 1–2 nonlinear parameters.
9. qMT (two-pool) and CEST multi-pool Lorentzian fits: strongly multimodal, partially separable.
10. MR fingerprinting: dictionary match *is* the profile lattice; polish off-grid with LM (what MRF usually lacks).

**Poor fit for this machinery**: non-separable high-dimensional problems (joint motion/diffeomorphism parameters,
coil-map estimation, joint recon): there use warm starts from the voxel-wise global stage, not a global optimiser.

## 7. Decisions (user, 2026-11-28) and work packages

1. Data: own sequences are complex (Gaussian path); the library must also fit external IVIM/T1 **magnitude** data
   (Rician path, 3.6). Both required.
2. Per-voxel fit first. Leave room for whole-volume priors via an outer loop: local solve → proximal/denoiser on
   parameter volumes → anchor as soft per-voxel objective → local solve (3.7). Not built now; the anchor interface and
   the loss abstraction are the room.
3. Consumers: **both** IVIM and Gd-T1 voxel fit.

Proposed order (nothing started):

- **WP-A** (shared core): generic separable profile + nD Kuhn minima + NNLS amplitudes + candidate set (P0), anchor
  term, loss abstraction with Rician. CPU reference first, then GPU; `waterfat` reimplemented as an instance, its tests
  must still pass bit-for-bit where deterministic.
- **WP-B** (IVIM): noisy complex + Rician benchmark of section 5, oracle comparison, baselines; ballistic FC/NC model.
- **WP-C** (Gd-T1): EPG-dictionary lattice ($R_1$, $T_2$, $B_1$/$\eta$), nested profile over shared $B_1$ across the Gd
  series; needs the EPG module's per-voxel evaluation as lattice atoms (dictionary generation is outside `qfit`).
- **WP-D** (priors): HQS/ADMM driver and first `VolumePrior`s.
- Per repo discipline: implement WP-A first, audit (degenerate-equivalence check on production-shaped data, independent
  review) before any GPU run, then one combined measurement campaign.

## Sources

- Endres, Sandrock, Focke 2018: <https://repository.up.ac.za/bitstream/2263/64460/1/Endres_Simplicial_2018.pdf>
- SHGO docs: <https://shgo.readthedocs.io/en/latest/docs/README.html>
- scipy-dev thread on SHGO limits: <https://mail.python.org/archives/list/scipy-dev@python.org/message/HIF7Y6SZDLNOHMPQ32KRHOGBIOJ7IVRE/>
- Fadnavis et al. VarPro + SH: <https://arxiv.org/abs/1910.00095>; TopoPro: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8714828/>
- DIPY IVIM: <https://docs.dipy.org/dev/examples_built/reconstruction/reconst_ivim.html>
- Gurney-Champion 2018: <https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0194590>
- Jalnefjord 2024: <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30042>
- Kaandorp 2026: <https://onlinelibrary.wiley.com/doi/full/10.1002/jmri.70301>
- IVIM mixture-density ensembles: <https://arxiv.org/pdf/2508.04588v1>
- Harms 2017 (MDT): <https://www.researchgate.net/publication/316549101_Robust_and_fast_nonlinear_optimization_of_diffusion_MRI_microstructure_models>
- GPU global optimisation: <https://arxiv.org/html/2606.26164>, <https://arxiv.org/pdf/2301.12457>, <https://arxiv.org/pdf/2601.18446>
