# Denoisers for PSF artifacts without bias; flow priors in P²nP/RED; equivariance; super-resolution (2026-11-17)

Theory note. Builds on: `p2np_theory.md` (Theorem B, Lemma A), `learned_prior_cs_theory.md` (Theorem L, Prop. G),
`mathematics_of_paper/pnp_restricted/paper.tex` (prop:estimator, thm:displacement, prop:strength, prop:keepfp),
`red_prior.md`, `convergence_p2np.md`, `denoiser_design.md` §10, `wavelet_prior.md`, `model_robustness.md` §3,
`temporal_operator.md` §13/§16–§19, `velocity_model_sr.md` §B, `learned_prior_cs_literature.md` (cited as [Survey §x]).
Toys: scratchpad `split_toy4.py`, `colour_toy.py` (numpy, CPU, seconds). They are *toys*: 1-D, n = 256 / 4096.
Labels: **[D]** derived here, **[V]** source read in full (or the cited sections), **[A]** abstract only,
**[V, survey]** read in full in an earlier session (see the survey).

## 0. Answer

1. **Where a prior acts is set by the scheme, not only by the denoiser (Theorem 1).** At every fixed point,
   $H(\hat x-x_0)=A^*n+W c$ holds, with $c$ the denoiser's correction. The weight is $W=\rho I$ for RED or HQS, and
   $W=(\alpha P)^{-1}$ for PnP with metric $P$. On the $H$-eigendirection $\lambda$ the correction enters the error
   with weight $\omega(\lambda)$:
   - RED / HQS: $\omega=\rho/\lambda$;
   - PnP: $\omega=1/(\alpha\lambda p(\lambda))$ for $P=p(H)$.
   - So RED/HQS and PnP with $P=I$ are *complementary*: the prior is suppressed in well-determined directions by
     $\lambda_{\rm small}/\lambda$.
   - A density-compensating (circulant) metric, $p\approx1/\lambda$, makes $\omega$ **flat**: the prior pulls the
     well-determined band as hard as the null space.
   - Toy, each scheme at its own best strength: bias² in the well-determined band is
     - PnP-DCW-metric $6.4\cdot10^{-3}$;
     - PnP-$I$ $1.6\cdot10^{-3}$;
     - RED = HQS = hard split $2.7\cdot10^{-4}$, which is the Monte-Carlo floor of the unbiased BLUE.
   - The DCW-metric limit is worse in that band than no prior at all (BLUE).
2. **"Act only where the data don't determine" already exists in our stack.**
   - The HQS / ADMM data step is $x=W_\rho x_{LS}+(I-W_\rho)D(z)$ with $W_\rho=H(H+\rho I)^{-1}$, a soft range/null
     split.
   - As $\rho\to0$ it becomes DDNM's hard split $A^+y+(I-A^+A)D(z)$ (Prop. 2).
   - Its fixed points are exactly the RED stationary points with $\lambda=\rho$. The toy reproduces them to all printed
     digits.
   - DDNM/DDRM need an SVD of $A$, which we don't have, and the hard split amplifies noise by $\kappa$. **RED (built)
     and ADMM-PnP are the implementable "null-space denoisers" for our operator.**
3. **Metric-aware denoising is a theorem (Theorem 3).** Let the denoiser be MMSE for noise $\mathcal N(0,\sigma^2P)$.
   - With metric $P$ and step $\alpha$, the PnP stationarity reads $\nabla f=(\sigma^2/\alpha)\nabla\log p_{\sigma^2P}$:
     $P$ drops out of the data/prior balance.
   - For Gaussian priors the PnP fixed point is then *exactly* independent of $P$. A white-noise denoiser under a
     metric $P$ instead regularises with weight $P^{-1}\approx H$ (the bias of item 1).
   - The coloured-noise MMSE residual is conservative in the $\Sigma^{-1}$ metric ($J\Sigma$ symmetric).
   - So ZMSHR1 (≈ scalar) only needs scale equivariance, as Hong et al. use. A circulant metric needs a denoiser
     trained on circulant-coloured noise, or RED.
4. **Artifact-aware training: three options.**
   - (a) *Training on solver iterates* gives a de-aliasing reconstruction network, not a denoiser:
     - Tweedie, Jacobian symmetry and the RED objective are lost;
     - it is valid only at the iterate distribution it was trained on.
   - (b) *A-structured Gaussian noise* (covariance from $PHP$ / the Toeplitz PSF) keeps Tweedie in the metric. This is
     the recommended "artifact-aware" training.
   - (c) *Noise injection* (PnP-Flow, annealing) does **not** hide a structured artifact unless $\sigma\gtrsim$ the
     artifact's norm over the receptive field (Lemma 4, Pinsker; tight by the Neyman–Pearson test). What it does is
     lower the denoiser's effective dimension.
5. **Flow matching and score models.**
   - An independent-coupling flow-matching velocity *is* a σ-family of MMSE denoisers: $D_t=x+(1-t)v$, and in
     variance-exploding (VE) coordinates $D_t(x)=D^{VE}_{\sigma}(x/t)$ with $\sigma=(1-t)/t$ (Lemma 5).
   - Its exact velocity is a gradient field, so RED's symmetry condition holds at the optimum. A potential
     parametrisation $v=\nabla\Psi$ makes it hold for the network.
   - What flows add beyond a σ-conditioned denoiser:
     - continuation in σ;
     - ODE likelihoods (an OOD flag, weakly calibrated);
     - posterior samples (UQ).
   - PnP-Flow's convergence (Prop. 4 of the paper) is only Cauchy-ness from summable steps. It is not
     data-consistent and not suited as the final estimator in our $\kappa\sim10^3$ regime. Use it as a
     cold-start/annealing stage, then fixed-σ RED.
   - Caveat: with minibatch-OT couplings, $D_t$ is *not* the Gaussian MMSE denoiser.
6. **Equivariance (Theorem 6, Props. 7–9).**
   - Equivariant velocity plus invariant source gives an equivariant flow map, an invariant density and an invariant
     log-likelihood ODE. This holds for any group of diffeomorphisms with $v\circ g=Dg\,v$; density invariance needs
     $|\det|=1$. The optimal FM field is automatically equivariant for linear actions and invariant couplings.
   - What the group acts on:
     - O_h × global U(1) acts orthogonally on $(z,v)$ (velocity model) and on the gauge features;
     - on raw 7-point encodes the axis flips act by $x_{\rm ref}$-dependent reflections. These are fibred isometries
       that still preserve the Gaussian.
   - **Scale has no invariant probability** (Prop. 7). The fix is a Haar (log-uniform) prior on the gain only, with
     a proper prior on shape and SNR. The MMSE is then *exactly* scale-equivariant (Thm 8, the Pitman estimator), and
     our σ²-normalised loss with log-uniform σ trains exactly this.
   - For VE flows, scale acts as a shift in $\log\sigma$ (Prop. 9). Schedules $\sigma_k\propto m(y)$, with $m$ a
     degree-1 statistic, make the whole sampler equivariant.
7. **Super-resolution.**
   - *As a starting guess* it is harmless only *at convergence* with a unique or contractive fixed point. With our
     tens of passes, an SR guess survives exactly in the slow, poorly-determined directions where hallucinations
     live. Run two inits: their difference map is a label-free hallucination detector.
   - *As a final output*, use one of:
     - SR inside the inverse problem (fine grid);
     - one HQS consistency step;
     - band restriction: the network may change only $|k|>k_{\max}$ and temporal content above what the knots
       resolve.
   - No k-space test can see null-space content (Prop. 10). UQ must come from posterior samples plus a *real-data*
     conformal calibration of a band-extrapolation task (§7.3).
8. **Build first (§8).**
   - Diagnostics (correction spectral profile by Lanczos, Jacobian asymmetry, init-sensitivity, held-out PE).
   - RED (metric-free fixed point) with the circulant metric used only for speed (prop:keepfp / ADMM-PCG).
   - Then denoiser v2:
     - σ-conditioned and jointly homogeneous;
     - optionally a gradient-step (potential) network;
     - trained on log-uniform σ with a white + metric-coloured + Toeplitz-PSF noise mixture;
     - on RACLETTE + ph0001–0004 + diffeo ph0101–0103, with ph9001 held out.
     - Trained as a flow-matching family, so annealing, likelihood and sampling come for free.

## 1. Setting: what an "artifact" is in our iteration

- $y=Ax_0+n$, whitened $n\sim\mathcal{CN}(0,\sigma_n^2I)$, $H=A^*A$, $f=\frac12\|Ax-y\|^2$.
- Spectral projector $Q_\varepsilon=\mathbf 1_{[\varepsilon,\infty)}(H)$ onto the well-determined band; $Q_\varepsilon^\perp$
  holds $N(A)$ and the poorly-conditioned part.
- The PnP input is $z_k=x_k-\alpha P\nabla f(x_k)$, so with $e_k=x_k-x_0$:
$$z_k-x_0=\underbrace{(I-\alpha PH)e_k}_{\text{PSF artifact}}+\underbrace{\alpha PA^*n}_{\text{coloured noise}} .$$
- The "effective PSF" is $\alpha PH$ (spatially varying, nonlocal sidelobes: spiral aliasing, coil g-factor). Its
  artifact $(I-\alpha PH)e_k$ is *signal-dependent*: it is $e_k$ pushed through the slow part of the data step.
- On $Q_\varepsilon$ it decays like $(1-\alpha\lambda)^k$; on $N(A)$ it does not decay at all.
- For per-frame 1 mm data ($R\approx19$, 32 coils) $A_t$ is formally injective, with about 1.7 complex samples per
  voxel, but $\kappa$ is huge. "Null space" therefore always means the low end of the spectrum, not $N(A)$ exactly.
- Two things need to be kept apart:
  - **artifact removal** (getting $e_k$ out of the slow directions) is the solver's job, given a prior that tells it
    what the slow directions should contain;
  - **bias** is what the prior does to directions the data determine.
- The CS view: recovery happens because the prior's tangent $T_0$ is transversal to the poorly-determined subspace.
  This is the restricted conditioning $\lambda_-$ of Theorem L, and then the prior *must* act on $Q_\varepsilon^\perp$.
  "Unbiased" can only mean no bias on $Q_\varepsilon$ beyond noise. On $Q_\varepsilon^\perp$ the bias equals the prior
  mismatch $\varepsilon_0=\|D(x_0)-x_0\|$ amplified by the restricted conditioning (Theorem L,
  prop:estimator item 4).

## 2. Where the prior acts: the bias identity

**Theorem 1 (fixed-point bias identity) [D].** Let $\hat x$ be a fixed point of one of:
- (RED / HQS) $\nabla f(\hat x)=\rho\,c$ with $c:=D(\hat x)-\hat x$;
- (PnP, fixed $P\succ0$) $\hat x=D(\hat z)$, $\hat z=\hat x-\alpha P\nabla f(\hat x)$, with $c:=D(\hat z)-\hat z$.

Then with $e=\hat x-x_0$ and $W=\rho I$, respectively $W=(\alpha P)^{-1}$:
$$He=A^*n+Wc. \tag{2.1}$$
Consequently:
1. $Wc\in\mathrm{range}(H)$. For RED, the correction has **no $N(A)$ component** at a fixed point: $D$ leaves the
   null-space content of $\hat x$ where it is. The null-space content is selected by the prior's self-consistency,
   which is the CS mechanism.
2. For every $\varepsilon>0$: $Q_\varepsilon e=Q_\varepsilon H^+A^*n+Q_\varepsilon H^+Wc$. Hence the deviation from the
   band-BLUE $\hat x_{LS,\varepsilon}=Q_\varepsilon H^+A^*y$ is $Q_\varepsilon H^+Wc$, and the bias is
   $\mathbb E\,Q_\varepsilon e=Q_\varepsilon H^+W\,\mathbb Ec$.
3. If $P=p(H)$ commutes with $H$, then $\|Q_\varepsilon H^+W\|=\sup_{\lambda\ge\varepsilon}\omega(\lambda)$, with
   **prior-weight profile** $\omega(\lambda)=\rho/\lambda$ (RED) or $1/(\alpha\lambda p(\lambda))$ (PnP), and
$$\|\mathbb E\,Q_\varepsilon e\|\le\sup_{\lambda\ge\varepsilon}\omega(\lambda)\;\|\mathbb E\,Q_\varepsilon c\|.$$

*Proof.*
- RED: $\nabla f(\hat x)=H\hat x-A^*y=He-A^*n=\rho c$.
- PnP: $\hat x-\hat z=\alpha P\nabla f(\hat x)$ and $\hat x=D(\hat z)$, so $\nabla f(\hat x)=(\alpha P)^{-1}c$.
- Item 1: $He-A^*n\in\mathrm{range}(A^*)=\mathrm{range}(H)$.
- Item 2: apply $Q_\varepsilon H^+$. On $\mathrm{range}\,Q_\varepsilon$, $H^+H=I$. Take expectations; $n$ has mean 0.
- Item 3: spectral calculus. ∎

**Reading.**
- *Complementarity is the shape of $\omega$*, after normalising the strength to the undetermined band (Prop. strength
  of the paper: the scalar part of a metric is a change of strength).
- RED, HQS and PnP with $P=I$ have $\omega\propto1/\lambda$. The well-determined directions receive the correction
  attenuated by $\lambda_{\min}/\lambda$: "the prior works in the basis the data don't see".
- PnP with an (ideal) density-compensating metric $p(\lambda)=1/\lambda$ has $\omega\equiv1/\alpha$: no attenuation.
  It is the same phenomenon as prop:estimator (oblique estimator) and `convergence_p2np.md` §4 (regulariser
  $R_P$ in $P$'s metric), now stated per spectral band.
- Our circulant metric is only approximately $p(H)$ (H is not circulant), so item 3 holds up to the commutator
  $\|[P,H]\|$ (cf. the leak $\|\Pi P\Pi^\perp\|$ in cor:displacement).
- Data consistency: (2.1) gives $\|Q_\varepsilon A^*(A\hat x-y)\|=\|Q_\varepsilon Wc\|$. A fixed point is consistent in
  the well-determined band exactly to the extent that the weighted correction has no energy there. That is the
  measurable quantity of §4 (T1).

**Toy check** (`split_toy4.py`).
- Setup:
  - 1-D, 10 spikes, Fourier sampling with Poisson repeats $\propto|k|^{-1.2}$ (no DCW; $w_{\max}=65$, 42 % of the
    frequencies sampled), $\sigma=0.5$;
  - denoiser = soft threshold (prox of $\ell_1$);
  - well-determined band $w\ge8$ (13 frequencies, 6 % of $\|x_0\|^2$);
  - each scheme at its own best strength (grids, both $\lambda$ and $\theta$ for RED/HQS);
  - 8 noise draws, so the bias² Monte-Carlo floor is ≈ var/8 ≈ $3\cdot10^{-4}$.

| scheme | rel. error | band bias² | band var | undetermined bias² |
|---|---|---|---|---|
| PnP, $P=I$, $\alpha=1/L$ | **0.169** | $1.6\cdot10^{-3}$ | $1.3\cdot10^{-3}$ | 0.110 |
| PnP, DCW metric $P=1/w$ | 0.248 | $\mathbf{6.4\cdot10^{-3}}$ | $2.1\cdot10^{-3}$ | 0.154 |
| RED ($\lambda=3.1,\theta=0.1$) | 0.193 | $2.7\cdot10^{-4}$ | $2.1\cdot10^{-3}$ | 0.106 |
| HQS soft split ($\rho=3.1$) | 0.194 | $2.7\cdot10^{-4}$ | $2.1\cdot10^{-3}$ | 0.106 |
| hard split (band := BLUE, PnP-$I$ elsewhere) | 0.171 | $2.6\cdot10^{-4}$ | $2.6\cdot10^{-3}$ | 0.097 |
| BLUE on band only | — | $2.6\cdot10^{-4}$ (floor) | $2.6\cdot10^{-3}$ | — |

Reading:
- RED and HQS coincide, as Prop. 2 predicts.
- The DCW-metric limit has 25× the floor bias in the band: its band MSE $8.5\cdot10^{-3}$ is worse than the
  unregularised BLUE's $2.6\cdot10^{-3}$.
- PnP-$I$ trades a small band bias for variance and is best overall here.
- The hard split is unbiased in the band and nearly as good overall.
- This is a toy: an $\ell_1$ prior is not our learned denoiser, and the band split is exact only because $H$ is
  diagonal in Fourier.

**Proposition 2 (HQS = soft DDNM = RED) [D].** The HQS data step $x^+=\arg\min_x f(x)+\frac\rho2\|x-D(z)\|^2$ is:
1. $x^+=W_\rho\,x_{LS}+(I-W_\rho)D(z)$ with $W_\rho=H(H+\rho I)^{-1}$, where $x_{LS}=H^+A^*y$ (any LS solution;
   $W_\rho H^+A^*y=(H+\rho I)^{-1}A^*y$);
2. as $\rho\downarrow0$: $x^+\to A^+y+(I-A^+A)D(z)$, DDNM's range/null rectification (Wang, Yu, Zhang 2023);
3. its fixed points ($z=x$) are exactly the RED stationary points $\nabla f(x)+\rho(x-D(x))=0$;
4. they do not depend on the preconditioner of the inner solve (`convergence_p2np.md` §4).

*Proof.*
- 1: the normal equations $(H+\rho I)x^+=A^*y+\rho D(z)$, and $\rho(H+\rho I)^{-1}=I-W_\rho$.
- 2: $(H+\rho I)^{-1}A^*\to A^+$ and $\rho(H+\rho I)^{-1}\to\Pi_{N(H)}=I-A^+A$.
- 3: set $x^+=z=x$ in the normal equations. ∎

Consequences:
- DDNM's noise problem (amplifying $A^+$ by $\kappa$; DDNM+ rescales per singular value, which needs the SVD) is
  exactly what a finite $\rho$ fixes.
- For us the computable objects are:
  - RED, built (`hasty_recon::red`);
  - ADMM-PnP with an exact per-set PCG data step (`convergence_p2np.md` §5(2), circulant-preconditioned inner CG, the
    estimator is unchanged);
  - the hard split only with an approximate spectral projector: Chebyshev $p(PH)$, or the Kronecker
    $G^{-1}\otimes C^{-1}$ eigenbasis (§7.2).

**Theorem 3 (metric-matched denoising) [D].** Let $D_\Sigma(z)=\mathbb E[x\mid x+\xi=z]$, $\xi\sim\mathcal N(0,\Sigma)$,
$x\sim p$, and $p_\Sigma=p*\mathcal N(0,\Sigma)$.
1. (coloured Tweedie) $D_\Sigma(z)=z+\Sigma\nabla\log p_\Sigma(z)$ and $J_{D_\Sigma}=\mathrm{Cov}[x\mid z]\,\Sigma^{-1}$.
   So $J\Sigma$ is symmetric PSD: $D_\Sigma$ is self-adjoint in the $\Sigma^{-1}$ inner product, and
   $\Sigma^{-1}(z-D_\Sigma(z))=\nabla(-\log p_\Sigma)$ is conservative.
2. (RED in a metric) $g=\nabla f+\lambda\Sigma^{-1}(x-D_\Sigma(x))=\nabla(f-\lambda\log p_\Sigma)$. With preconditioner
   $P=\Sigma$ the step $x-\alpha Pg=x-\alpha P\nabla f-\alpha\lambda(x-D_\Sigma(x))$ uses the plain residual. Its
   stationary points do not depend on any preconditioner.
3. (PnP in a metric) With $\Sigma=\sigma^2P$, PnP's fixed points satisfy
   $\nabla f(\hat x)=(\sigma^2/\alpha)\nabla\log p_{\sigma^2P}(\hat z)$. $P$ no longer weights the prior; it enters only
   through the smoothing $p_{\sigma^2P}$, which is $O(\sigma^2)$, and through the evaluation point $\hat z$.
4. (Gaussian prior, exact) For $p=\mathcal N(0,C)$:
   - PnP with metric $P$ and the matched $D=C(C+\sigma^2P)^{-1}$ has the unique fixed point
     $(H+\tfrac{\sigma^2}{\alpha}C^{-1})^{-1}A^*y$, **independent of $P$**;
   - with the white $D=C(C+\sigma^2I)^{-1}$ the fixed point is $(H+\tfrac{\sigma^2}{\alpha}P^{-1}C^{-1})^{-1}A^*y$:
     the prior weighted by $P^{-1}$.

*Proof.*
- 1: $\nabla p_\Sigma(z)=\int\Sigma^{-1}(x-z)\mathcal N(z;x,\Sigma)p(x)dx$ gives the first identity. Differentiating
  again, $\nabla^2\log p_\Sigma=\Sigma^{-1}\mathrm{Cov}[x|z]\Sigma^{-1}-\Sigma^{-1}$, so
  $J=I+\Sigma\nabla^2\log p_\Sigma=\mathrm{Cov}[x|z]\Sigma^{-1}$.
- 2: immediate.
- 3: $c=\hat x-\hat z=\sigma^2P\nabla\log p(\hat z)$ and $c=\alpha P\nabla f(\hat x)$.
- 4: $x=C(C+\sigma^2P)^{-1}(x-\alpha P(Hx-A^*y))\iff\sigma^2PC^{-1}x=-\alpha P(Hx-A^*y)$, and $P$ cancels. The white
  case is the same computation without the cancellation. ∎

Reading:
- Hong et al.'s normalisation-equivariance argument is the scalar case $P_k=\tau_kI$: the matched noise level is
  $\sigma\sqrt{\tau_k}$, delivered by scale equivariance.
- ZMSHR1's rank-one term is a rank-one mismatch.
- A circulant or Kronecker metric is a *non-scalar* $P$. There, a white denoiser produces the flat $\omega$ of
  Theorem 1, so either train on noise coloured like $P$ or use RED.
- The equivalent statement in the paper's language: the matched denoiser makes the tangent compatible with the
  metric, so the leak $\Pi P\Pi^\perp$ acts on a correction that is itself $P$-shaped.

**Colour toy** (`colour_toy.py`, Gaussian stationary prior with spectrum $1/(1+(k/8)^2)$, error variance rising with
$|k|$ at a fixed total energy):

| spread of error spectrum | coloured MMSE | white D, σ² = mean error | white D, best σ | noise completion to max |
|---|---|---|---|---|
| 1 | 0.216 | 0.216 | 0.216 | 0.216 |
| 10 | 0.104 | 0.135 | 0.104 | 0.311 |
| 100 | 0.046 | 0.115 | 0.049 | 0.330 |
| 1000 | 0.036 | 0.113 | 0.041 | 0.331 |

Reading:
- A white denoiser at a *tuned* σ is within 13 % of the coloured MMSE here, because the error sits where the prior
  has little energy. At the energy-matched σ it is 2.5–3× worse.
- Noise completion, i.e. adding $\mathcal N(0,\sigma_{\max}^2I-\Sigma)$ to make the error white (the DDRM/DDNM+ device),
  is useless at our spreads (×10³ density).
- Consequence: the strength must be tuned per metric (prop:strength). The *shape* argument of Theorems 1/3 is about
  bias location, which tuning cannot fix.

## 3. Artifact-aware training: what is legitimate

**3.1 Training on solver iterates.** Let $D^\star(z)=\mathbb E[x_0\mid z_K=z]$ for the iterate distribution at
iteration $K$.
- The error $z_K-x_0$ depends on $x_0$ (aliasing of the image itself). $D^\star$ is then the posterior mean of a
  *linear inverse problem* $z=Bx_0+w$, not a Gaussian denoiser.
- From a zero start with $P$, $B=I-(I-\alpha PH)^K$, and $w$ is coloured noise.
- In a linear-Gaussian model diagonal in $H$'s eigenbasis (illustration, [D]): $D^\star_\lambda=c_\lambda b_\lambda/(c_\lambda b_\lambda^2+s_\lambda)$
  with $b_\lambda=1-(1-\alpha\lambda p)^K$.
  - $b\approx1$ on well-determined directions: a mild Wiener filter.
  - $b\approx K\alpha\lambda p$ on poorly-determined ones: an amplification.
  - So the learned correction *does* concentrate in the complement, as wanted.
- But:
  1. Tweedie fails, $J$ is not symmetric, and the RED gradient interpretation and Theorems 1/3 are lost (it is not a
     regulariser);
  2. it is tied to $K$, $P$ and the warm start. Inside PnP the iterates leave the training distribution (closed-loop
     shift). Training on the iterates of the scheme *with $D$ itself* is a fixed-point problem in training (DAgger /
     DEQ, Gilton et al. 2021 [V, survey]), with full solves per sample. That is not affordable at $10^8$ voxels.
  3. Gottschling et al.'s detail-transfer theorem [V, survey §10.2] applies in full: an accurate, stable de-aliasing
     map still hallucinates detail that lies in $N(A)$ of the training set.
- Use iterate residuals as a **validation distribution** and for a small fine-tuning share, not as the training
  definition.

**3.2 A-structured Gaussian perturbations (recommended).** Train the MMSE denoiser on $z=x_0+\xi$ with
$\xi\sim\mathcal N(0,\sigma^2\Sigma_A)$, where $\Sigma_A$ is stationary on a patch, so it is available on 32³ patches:
- $\Sigma_A=C$ (circulant metric, matched per Theorem 3);
- $\Sigma_A=P_cH_TP_c$ (back-projected noise through the Toeplitz embedding of $H$, which we have);
- a power of the data step, $(I-\alpha P_cH_T)^{2j}$, $j$ random: Gaussian with the *spectrum* of the PSF artifact,
  but independent of $x_0$.

This keeps Theorem 3 (conservative residual in $\Sigma^{-1}$), the patch locality and the equivariances:
$\Sigma_A$ commutes with O_h only if the PSF does. A spiral PSF is roughly isotropic, so augment it with the
48 rotated copies rather than require exactness.

What it does not capture:
- the non-Gaussian, signal-locked structure of aliasing;
- PSF sidelobes beyond the patch: periodic wrap on a 32³ patch is a known approximation.

**3.3 Noise injection does not hide structured artifacts.**

**Lemma 4 [D].** Let $D$ have a receptive field $\Omega$ and $z=x_0+e$ with $e$ deterministic. Then
$$\mathrm{TV}\big(\mathcal L(z_\Omega+\sigma\varepsilon),\,\mathcal L(x_{0,\Omega}+\sigma\varepsilon)\big)\le\frac{\|e_\Omega\|}{2\sigma},$$
by Pinsker with $\mathrm{KL}=\|e_\Omega\|^2/2\sigma^2$. The Neyman–Pearson test between the two laws has power set by
$\|e_\Omega\|/\sigma$. So a network *can* detect, and therefore treat specially, the artifact whenever
$\sigma\lesssim\|e_\Omega\|=\sqrt{|\Omega|}\,\mathrm{rms}(e)$.
- For $|\Omega|\approx32^3\cdot12$ that is $\sim600\times$ the per-voxel artifact level. Injected noise of
  practical size does not make the artifact "look Gaussian".
- What injection at level σ *does*: it selects $D_\sigma$, whose Jacobian is a projector of effective dimension
  $d(\sigma)\propto1/\sigma$ (Mohan et al. [V, survey §9.4]). The artifact passes through as $\bar J_\sigma e$: its
  off-tangent part is removed, and its in-tangent (plausible-looking) part survives.
- Annealing σ↓ (PnP-Flow, DiffPIR, continuation) trades artifact removal against signal removal. It cannot remove
  what the prior finds plausible. That part must be removed by the data term, i.e. by convergence on the slow
  directions.

**3.4 Training mixture (recommendation).** Per patch:
- 50 % white noise;
- 30 % metric-coloured noise ($C$);
- 20 % Toeplitz-PSF-shaped noise (3.2);
- σ log-uniform over the case-level SNR range (Thm 8);
- all at the case-level σ (the training lesson of 2026-11-04).

Add iterate residuals from dev-scale solves on ph0001–0004 only as a validation set. If the validation loss on
iterates stays ≥ 2× the Gaussian-noise loss at equal energy, add them as a 10 % fine-tuning share and accept the
loss of the RED interpretation explicitly (E3, §8).

## 4. Bias tests (labels never inside the method; truth only in explicitly oracle arms)

| # | test | what it measures | cost | label-free? |
|---|---|---|---|---|
| T1 | **spectral profile of the correction**: Lanczos on $H$ (or $PH$) started at $c=D(\cdot)-\cdot$ at the limit; Gauss quadrature gives $\mu_c(\lambda)=\sum_i|\langle c,u_i\rangle|^2\delta_{\lambda_i}$ | fraction of $\|Wc\|^2$ above ε; the bias weight of Thm 1 | 20–30 normal ops | yes |
| T2 | $\chi^2$ of the band residual: $\|Q_\varepsilon A^*(A\hat x-y)\|$ against the noise-only Monte-Carlo, using T1's quadrature | data consistency in the determined band | T1 + noise draws | yes |
| T3 | **held-out k-space** PE (whole readouts, stratified, $|k|<k_c$ excluded too; `model_robustness.md` §3a) vs prior strength and α-shrink $D_\alpha=(1-\alpha)I+\alpha D$ | prediction error; a biased prior raises it | one recon per arm | yes |
| T4 | Jacobian asymmetry $\|J-J^*\|_F^2/\|J\|_F^2$ (Hutchinson, JVP/VJP); also $\|J\Sigma-\Sigma J^*\|$ for coloured training | RED / Theorem 3 validity | ~20 JVPs | yes |
| T5 | init-sensitivity map $|x_K(\text{init}_1)-x_K(\text{init}_2)|$ and its T1 profile | non-data-determined content (§7.1) | two recons | yes |
| T6 | prior mismatch at the truth: $\varepsilon_0=\|D(x_0)-x_0\|$, $\|A(D(x_0)-x_0)\|^2$ vs $m\sigma^2$ | is the truth a near-fixed point; consistent mismatch? | 1 denoise | oracle |
| T7 | **noise-free, inverse-crime-free** control: hasty-mrsim (0.5 mm sources, continuous object, full physics) → recon operator on the coarse grid; $\sigma=0$; fixed point vs band-limited truth (`cfd_training_data.md` §5) | pure prior bias plus model error, no noise | 1 recon | oracle |
| T8 | Monte-Carlo bias: $K$ noise draws on the same k-space, split by $Q_\varepsilon$ (circulant approximation of the band) | bias² vs variance per band, as in the toy | $K$ recons, dev scale | oracle (truth) |

Two facts govern interpretation:
- **Prop. 10 [D, trivial].** If $x-x'\in N(A)$, the laws of $y$ under $x$ and $x'$ coincide. Every test based on
  $y$ (T2, T3) has power = size against null-space hallucinations. Those are visible only through priors (T5, the
  diversity of posterior samples) or oracles (T6–T8).
- Held-out PE is unbiased for *prediction* error only (`model_robustness.md` §3a). E1 failed in part on 2026-11-17
  (§16.3) and passed on the next run (§19.1), so treat PE rankings between close arms as provisional.

## 5. Flow-matching and diffusion priors in our scheme

**Lemma 5 (one network, a family of MMSE denoisers) [D; standard].** Let $X_t=tX_1+(1-t)X_0$,
$X_0\sim\mathcal N(0,I)$ independent of $X_1\sim p_{\rm data}$, and $v^\star(x,t)=\mathbb E[X_1-X_0\mid X_t=x]$
(the flow-matching optimum). Then:
1. $D_t(x):=\mathbb E[X_1|X_t=x]=x+(1-t)v^\star(x,t)$;
2. $v^\star=\big(x+(1-t)\nabla\log p_t(x)\big)/t$;
3. $D_t(x)=D^{VE}_{\sigma}(x/t)$ with $\sigma=(1-t)/t$, where $D^{VE}_\sigma$ is the Gaussian MMSE denoiser at
   level σ. Equivalently $\mathrm{SNR}=t^2/(1-t)^2$: only the SNR matters (Kingma et al. 2021 [A]);
4. $J_{D_t}=t^{-1}\sigma^{-2}\mathrm{Cov}[X_1|X_t]$ is symmetric PSD, and
   $v^\star=\nabla_x\Psi_t$ with $\Psi_t=(\tfrac12|x|^2+(1-t)\log p_t)/t$ is a gradient field.

*Proof.*
- 1: $X_0=(X_t-tX_1)/(1-t)$ inside the conditional expectation.
- 2: Tweedie for $X_t$: $\nabla\log p_t=-\mathbb E[X_0|X_t]/(1-t)$. Then $\mathbb E[X_1|x]=(x+(1-t)^2\nabla\log p_t)/t$,
  and subtract $\mathbb E[X_0|x]$.
- 3: $X_t/t=X_1+\sigma X_0$.
- 4: from 3 and the Miyasawa second-order identity [Survey §9.1]. ∎

Caveats:
- PnP-Flow trains with **minibatch-OT couplings** (Martin et al. [V]). Then $X_0$ and $X_1$ are dependent, and
  $D_t=\mathbb E[X_1|X_t]$ is no longer the Gaussian MMSE denoiser of item 3. Yet PnP-Flow applies it after injecting
  *independent* noise.
  - For exact (Brenier) OT, $v^\star$ is still a gradient (Benamou–Brenier), but item 3 fails.
  - For our use (a denoiser inside RED), train with the **independent coupling**.
- What flows add over a σ-conditioned denoiser (DRUNet-style) is not the denoiser; that is the same object:
  1. a principled continuation path in σ;
  2. the ODE: exact log-likelihood via $\frac{d}{dt}\log p_t(x_t)=-\nabla\cdot v$ (Hutchinson divergence), per patch;
  3. samples.
- "Better calibrated priors": the likelihoods of deep generative models are known to rank OOD inputs badly
  (Nalisnick et al. 2019 [A]). Use the patch likelihood as a *relative* flag (phantom vs real, per tissue class),
  not as calibrated evidence.

**5.1 The algorithms in our notation** (VE coordinates, metric $P$; $\nabla f$ costs a full normal-operator pass):

| method | iteration | guarantee | fit for us |
|---|---|---|---|
| PnP-Flow (Martin et al. ICLR 2025 [V]) | $z_n=x_n-\gamma_n\nabla F(x_n)$; $x_{n+1}=D_{t_n}((1-t_n)\varepsilon+t_nz_n)$, $t_n\uparrow1$, $\gamma_n=(1-t_n)^a$ | Prop. 4: if $\sum(1-t_n)<\infty$, $\gamma_n=1-t_n$ and iterates are bounded, then they converge (increments are summable). The limit is *not* characterised; $D_1=\mathrm{Id}$ | summable steps with $\kappa\gtrsim10^3$ ⇒ underfit data. Use as an annealed cold start only |
| annealed RED [D] | RED with $D_{\sigma_k}$, $\sigma_k\downarrow\sigma_\star$, then fixed $\sigma_\star$ | after the freeze: stationary points of $f-\lambda\sigma_\star^2\log p_{\sigma_\star}$ (Theorem 3), independent of $P$ | **recommended** (continuation handles nonconvexity; the final estimator is fixed) |
| RED-diff (Mardani et al. ICLR 2024 [A]) | descend $f(\mu)+\sum_tw_t\,\mathrm{KL}(\mathcal N(a_t\mu,\sigma_t^2)\Vert p_t)$; gradient $\sum_t w_t(a_t/\sigma_t)\mathbb E_\varepsilon\epsilon_\theta(a_t\mu+\sigma_t\varepsilon)$ [D, exact score] | conservative for an exact score: RED with a σ-averaged prior | possible; one network call per sampled $t$; adds stochastic gradients (breaks ZMSHR1 secants: average over fixed draws, as `--red-shifts`) |
| DPS (Chung et al. 2023 [V, survey]) | reverse SDE + $\nabla_{x_t}\log p(y\mid D_t(x_t))$ | Jensen-gap bound (constant suspect, [Survey §11.6]); not data-consistent | backprop through the net × normal op per step; too costly at scale |
| DDRM/DDNM (Kawar et al. 2022 [A]; Wang et al. 2023 [V, method]) | range/null split in the SVD of $A$ | exact consistency for noiseless data | no SVD; use Prop. 2 (HQS) |
| PnP-DM / split Gibbs (Wu et al. 2024 [V, survey]) | alternate a Gaussian data step $\propto\exp(-f-\|x-z\|^2/2\rho^2)$ (CG + perturbation) and a denoising-posterior sample | average-Fisher-information bound only | UQ at dev scale; the data step is our PCG |

**5.2 Convergence conditions vs flow denoisers.**
- P²nP's proofs need $(1+\epsilon)$-Lipschitz $D$ and are vacuous in CS (`p2np_theory.md` §0).
- Theorem B / L need a secant Jacobian close to a projector (leakage η).
- RED needs conservativeness.
- A flow/score family at a *fixed* σ satisfies Lemma 5.4 at the optimum, so it fits RED and the Theorem-L picture
  (symmetric $J$, eigenvalues in $[0,1]$ when the posterior covariance $\preceq\sigma^2I$, e.g. a log-concave prior
  [Survey §9.7]).
- No Lipschitz bound below $M^2/\sigma^2$ exists in general [Survey §9.7], so global contraction results do not
  apply. Only the local/restricted ones do, as before.
- To make conservativeness exact for the *network*, parametrise $D_\sigma=I-\nabla g_\theta(\cdot,\sigma)$
  (gradient-step denoiser, Hurault et al. 2022 [V, survey]); in flow form, $v_\theta=\nabla_x\Psi_\theta$. Cost: one
  extra backward pass per call, and the equivariance of $g_\theta$ must be invariance (scalar output, degree 2 under
  scale, so that $\nabla g$ has degree 1).
- With the time-varying denoiser of PnP-Flow, fixed-point theory does not apply at all. The algorithm is a
  stochastic continuation. Its data consistency must come from a final fixed-σ RED/HQS stage. Then Theorems 1–3 hold
  again.

**5.3 Preconditioner and noise-level coupling.**
- The annealing schedule should be tied to the metric (Theorem 3): with ZMSHR1's scalar $\tau_k$, the matched level
  is $\sigma\propto\sqrt{\tau_k}\,\sigma_{\rm noise}$. Scale equivariance (§6) makes that free.
- With the circulant metric, use the coloured-trained network (3.2) at level σ, or keep the circulant metric for
  speed only via prop:keepfp ($z_{k+1}=z_k-B_kF(z_k)$, limit = reference limit) or ADMM-PCG. In both cases the
  estimator is the RED / $P=I$ one.

## 6. Equivariance

**Theorem 6 (equivariant flows) [D; classical, cf. Köhler, Klein, Noé 2020 [A]].** Let $G$ act on $\mathbb R^n$ by
$C^1$ diffeomorphisms. Let $v(x,t)$ be locally Lipschitz in $x$ with a global flow $\Phi_{s,t}$, and assume
$v(gx,t)=Dg(x)\,v(x,t)$ for all $g,x,t$ ($g$-related). Then:
1. $\Phi_{s,t}\circ g=g\circ\Phi_{s,t}$;
2. if $g_\#\mu_0=\mu_0$, then $g_\#\mu_t=\mu_t$ for $\mu_t=(\Phi_{0,t})_\#\mu_0$;
3. for linear $g$, densities satisfy $p_t(gx)|\det g|=p_t(x)$, so they are invariant when $|\det g|=1$, and
   $(\nabla\!\cdot v)(gx,t)=(\nabla\!\cdot v)(x,t)$: the log-likelihood ODE is equivariant term by term;
4. (the optimum is equivariant) If $g$ is linear (or affine), and $(gX_0,gX_1)\overset d=(X_0,X_1)$ (invariant
   source, invariant data, invariant coupling), then $v^\star(gx,t)=g\,v^\star(x,t)$.

*Proof.*
1. $t\mapsto g\Phi_{s,t}(x)$ solves $\dot y=Dg\,v(\Phi_{s,t}x)=v(g\Phi_{s,t}x)$ with $y(s)=gx$. By uniqueness it equals
   $\Phi_{s,t}(gx)$.
2. $g_\#\mu_t=g_\#\Phi_\#\mu_0=\Phi_\#g_\#\mu_0=\mu_t$.
3. Change of variables; $J_v(gx)=gJ_v(x)g^{-1}$, so the traces agree.
4. $gX_t=tgX_1+(1-t)gX_0\overset d=X_t$ jointly with $g(X_1-X_0)$, so
   $\mathbb E[X_1-X_0\mid X_t=gx]=g\,\mathbb E[X_1-X_0\mid X_t=x]$. ∎

So equivariance is a property of the Bayes-optimal field, and architecture only enforces it. An equivariant
architecture cannot cost accuracy at the optimum.

**What the compact part is for us.**
- In the **velocity model** $(z,v)$ (the end goal): O_h acts by voxel permutation × (scalar on $z$, signed
  permutation on $v$). Global U(1) acts by $z\mapsto e^{i\phi}z$. Both are orthogonal on $\mathbb R^{2N_z+3N_v}$, so the
  Gaussian source is invariant and Theorem 6 applies with linear $g$.
- On **raw 7-point encodes**, a flip $v_a\to-v_a$ acts by $x_a\mapsto e^{2i\arg x_{\rm ref}}\overline{x_a}$
  (`denoiser_design.md` §10). This is a *fibred* isometry, $(x_{\rm ref},x_a)\mapsto(x_{\rm ref},R(x_{\rm ref})x_a)$ with $R$
  orthogonal. It is nonlinear, but:
  - $|\det Dg|=1$ (block-triangular, orthogonal diagonal);
  - it preserves $\mathcal{CN}(0,I)\otimes\mathcal{CN}(0,I)$, since conditionally on $x_{\rm ref}$ the law of $x_a$ is
    rotation invariant.
  - Hence items 1–2 hold, provided $v$ is $g$-*related* ($Dg$ includes $\partial R/\partial x_{\rm ref}$).
  - An O_h-equivariant CNN on gauge features $c_a=x_a\bar{\hat x}_0$ is equivariant in that coordinate system.
    The flow is best defined there, or in $(z,v)$, rather than on raw encodes.
- The **local gauge** $U(1)^N$ (per-voxel background phase) is compact, so invariant laws exist. But it is not a
  symmetry of the data: background phase is smooth. Exact local-gauge plus shift equivariance forces $D=I+$ const
  (§10 of the design note). Keep it as a feature choice, not a group.

**Proposition 7 (no scale-invariant source) [D].** The only probability measure on $\mathbb R^n$ invariant under all
dilations $x\mapsto\mu x$, $\mu>0$, is $\delta_0$.

*Proof.* $F(r)=\Pr(\|X\|\le r)$ satisfies $F(\mu r)=F(r)$ for all $\mu,r>0$, so it is constant on $(0,\infty)$.
$F(\infty)=1$ forces $F\equiv1$, so $\Pr(X=0)=\lim_{r\downarrow0}F(r)=1$. ∎

So Theorem 6.2 can never give a scale-invariant generated law. The non-compact, non-measure-preserving factor must
be handled differently.

**Theorem 8 (exactly scale-equivariant MMSE: Haar gain, proper shape) [D; the Pitman construction, Eaton 1989 /
Berger 1985 §6.6].** Model the stack as
$$z=\gamma w,\qquad x=\gamma\xi,\qquad w=\xi+\sigma_0\varepsilon,$$
- gain $\gamma>0$ with the improper Haar prior $d\gamma/\gamma$ (complex gain: $\times$ uniform U(1));
- $(\xi,w)$ with a *proper* joint law: shape, SNR, its own relative noise level $\sigma_0$; density $q$ of $w$.

Then for $z\ne0$:
- the generalised posterior mean $D(z)=\mathbb E[x\mid z]$ exists whenever $\int_0^\infty u^{n-1}q(u\hat z)du<\infty$
  (true for a.e. direction, since $\int q=1$);
- $D(\mu z)=\mu D(z)$ for all $\mu>0$;
- $D$ is the best equivariant estimator under the scale-invariant loss $\|D(z)-x\|^2/\gamma^2$.

*Proof.*
- $p(\gamma\mid z)\propto\gamma^{-n-1}q(z/\gamma)$ and $D(z)=\int\gamma\,\mathbb E[\xi\mid w=z/\gamma]\,\gamma^{-n-1}q(z/\gamma)d\gamma\big/\int\gamma^{-n-1}q(z/\gamma)d\gamma$.
- Substituting $z\to\mu z$, $\gamma=\mu\gamma'$ leaves $d\gamma/\gamma$ invariant. The factor $\mu^{-n}$ cancels and the
  numerator gains $\mu$.
- Finiteness: substitute $u=\|z\|/\gamma$.
- Optimality: the standard result that the best equivariant estimator is generalised Bayes under the right Haar
  measure (the group is abelian). ∎

Reading:
- A **bias-free (degree-1) network trained with the σ²-normalised loss, σ log-uniform over a wide range, at
  case-level σ**, is a least-squares fit of exactly this $D$. Log-uniform σ is the truncated Haar measure. This is the
  formal content of "bias-free nets generalise across noise levels" and of our training choices.
- Velocity-encoding phases and $v$ are degree 0 under real scale, and the global phase is compact.
- In the velocity model, scale acts as $\mathrm{diag}(\mu,1)$ on $(z,v)$: $z$ has degree 1, $v$ degree 0, so
  $\det=\mu^{N_z}$.
- The non-blind version $D(\mu z,\mu\sigma)=\mu D(z,\sigma)$ is the same construction conditioned on
  $\sigma=\gamma\sigma_0$.

**Proposition 9 (scale acts on flows as a time shift) [D].**
1. **VE flow** $\frac{dz}{d\sigma}=(z-D(z,\sigma))/\sigma$. If $D$ is jointly homogeneous, the field has degree 0 in
   $(z,\sigma)$, and $w(\sigma):=\mu z(\sigma/\mu)$ solves the same ODE. So $\Phi_{\mu\sigma_1\to\mu\sigma_0}(\mu z)=\mu\Phi_{\sigma_1\to\sigma_0}(z)$:
   dilation is translation in $\log\sigma$. The source $\mathcal N(0,\sigma_{\max}^2I)$ maps to the source at
   $\mu\sigma_{\max}$, consistent with Prop. 7 (invariance only as $\sigma_{\max}\to\infty$).
2. **Linear interpolant** $X_t=tX_1+(1-t)X_0$. Scaling the data by μ gives
   $t\mu X_1+(1-t)X_0=(t\mu+1-t)\big(t'X_1+(1-t')X_0\big)$ with $t'=t\mu/(t\mu+1-t)$: a time reparametrisation plus an
   amplitude. A network built in VE/SNR coordinates (Lemma 5.3) with joint homogeneity is exactly equivariant. A raw
   $v_\theta(x,t)$ cannot be, by Prop. 7.
3. **Equivariant schedules.** In any sampler or annealed RED, set every level $\sigma_k=\kappa_k\,m(y)$ with $m$ a
   degree-1 statistic, e.g. $\hat\sigma$ from the finest wavelet band of $A^*y$ (as `--wavelet` does), or the RMS of
   $x_{\rm ref}$. Each step is then equivariant, and $\mathcal L(\mathrm{out}(\mu y))=\mu_\#\mathcal L(\mathrm{out}(y))$
   (the noise ε is unscaled because it enters as $\sigma_k\varepsilon$).

*Proof.*
- 1: $\dot w(\sigma)=z'(\sigma/\mu)=V(z(\sigma/\mu),\sigma/\mu)=V(\mu z(\sigma/\mu),\sigma)$.
- 2: algebra.
- 3: induction over the steps; the data step is linear and commutes with scale. ∎

Answer to the user's question:
- The scale group does break the invariant-source requirement (Prop. 7).
- It is handled by quotienting the gain with a Haar nuisance, which gives exact equivariance of the MMSE (Thm 8),
  and, for generation or likelihood, by:
  - (a) modelling the normalised object $x/m(x)$ with a source invariant under the compact part only, the likelihood
    being exact for the shape;
  - or (b) equivariant VE schedules (Prop. 9).
- The likelihood of $x$ itself is then improper. Likelihood *ratios* between candidates at the same $m$ remain
  meaningful.

## 7. Super-resolution after or around the reconstruction

**7.1 As a starting guess.**

**Proposition SR1 [D].**
1. If $f+\lambda R$ is strictly convex, the limit does not depend on the start: RED with an envelope or prox
   regulariser (LLR unshifted, wavelet, the quadratic locks).
2. If the PnP/RED map is a contraction on a tube (Theorem L, radius $r$), every start inside the tube gives the same
   limit.
3. **After $K$ iterations** the start's error survives as $\prod_k T'_k(x_0-\hat x)$. On a direction with local rate
   $r_i=\rho(J_D(I-\alpha PH))|_i$:
   - plausible (in-tangent) structure in a poorly-determined direction has $r_i\approx1-\alpha\lambda_i\approx1$;
   - implausible structure has $r_i\approx0$.

So a hallucination that the prior finds plausible, placed in the slow directions, is neither removed by finitely many
iterations nor, if it is a near-fixed point of $D$, by the limit. With tens of passes at 1 mm, item 3 is the
operative regime.

*Test (T5):* reconstruct from {plain warm start, SR start}. The difference map plus its T1 spectral profile is a
label-free map of what the data do not decide. Report it with every SR-initialised run.

**7.2 As a final output: consistent designs and what is identifiable.**
- **D1, SR inside the inverse problem** (`velocity_model_sr.md` §B: fine grid, ROI dual grid, `--voxel-basis`, prior
  trained at the fine $h$). The data step keeps consistency, and Theorems 1/3 govern the bias. Preferred.
- **D2, one consistency step:** $x_{\rm out}=\arg\min\|Ax-y\|^2+\rho\|x-N(x_{\rm rec})\|^2=W_\rho x_{LS}+(I-W_\rho)N(x_{\rm rec})$
  (Prop. 2). One PCG solve. As $\rho$ decreases the output approaches DDNM-exact consistency, with $\kappa$-amplified
  noise, so pick ρ by T3.
- **D3, band restriction:** $x_{\rm out}=x_{\rm rec}+(I-B)\,(N(x_{\rm rec})-x_{\rm rec})$.
  - $B$ = spatial band $|k_a|\le k_{\max,a}$ ⊗ temporal band (what the knots resolve).
  - In-band content is untouched by construction. Out-of-band content is pure prior and must be shown as such
    (overlay mask, separate channel).
- **Spatial identifiability.** On a fine grid, components with $|k|>k_{\max}$ lie in $N(A)$ up to the voxel-basis
  leakage. Inside the band, conditioning is set by density, coils and g-factor (the ε-split).
- **Temporal identifiability** (`temporal_operator.md` §13, §19):
  - every readout has its own time, so a knot basis with $M$ cardiac knots is identified if the Kronecker
    approximation $H\approx G\otimes H_s$ is well conditioned;
  - its eigenvalues are $\lambda_i(G)\lambda_j(H_s)$, so high temporal harmonics × high spatial frequencies are the
    poorly-determined corner;
  - harmonics $h>M/2$ are not representable at all.
- Consequences for temporal SR:
  1. "Temporal SR beyond the knots" is pure prior unless the data are refit with more knots. Refitting is cheap:
     cubic 20 ≈ 1.1× today's cost with ¼ of the model error (§16.1).
  2. The *computable* soft split for the temporal axis is $W_{ij}=\lambda_i(G)h_j/(\lambda_i(G)h_j+\rho)$ in the
     $G$-eigenbasis × circulant spectrum, from the pieces of the existing $G^{-1}\otimes C^{-1}$ preconditioner. It
     tells where in the (harmonic, $|k|$) plane a temporal prior may act without bias.
  3. ECG-gating jitter and RR variability blur the *data* itself. Content faster than the jitter is model error, not
     null space, and no SR can recover it consistently.

**7.3 Uncertainty and hallucination tests that work without labels.**
- **Posterior samples** (split Gibbs with our PCG data step, §5.1), at dev scale. Pixelwise spread and posterior PCs
  (Manor–Michaeli via JVP [V, survey §9.1]).
  - Calibration is not guaranteed: score error, and the hardness results [Survey §10.5].
- **Split conformal on phantoms.** Score $s_i=\max_{\rm ROI}|x_{\rm SR}-x_{\rm true}|/u_i$ ($u$ = posterior std) on
  calibration phantoms. The $\lceil(n+1)(1-\alpha)\rceil$-th score gives marginal coverage on *exchangeable* new
  phantoms (Angelopoulos et al. 2022 [A]; task-driven intervals, Wen et al. 2024 [A]).
  - It does not transfer to patients: exchangeability fails.
- **Band-extrapolation conformal on real data** (new, [D]): it uses only measured data.
  - Reconstruct from $|k|\le k_1<k_{\max}$, apply the SR map to predict the band $(k_1,k_{\max}]$, and score against
    the *measured* data in that band. Use the whitened held-out residual per patch, normalised by the predicted
    uncertainty.
  - Over real patients / patches the scores are exchangeable, so the quantile gives distribution-free coverage *for
    the band-extrapolation task*.
  - Transfer to the true task ($k>k_{\max}$) assumes approximate scale similarity of the band transition. Check that
    assumption on phantoms, where both are available.
  - The same works in time: refit with fewer knots and predict the measured higher-knot content.
- **Held-out k-space:** the SR output must not raise held-out PE versus $x_{\rm rec}$ (paired block bootstrap). This
  is necessary but blind to $N(A)$ by Prop. 10.

## 8. Recommendation and pre-registered experiments

**Build order.**
1. **Diagnostics module** (CPU code now; runs at dev scale when the GPU is free):
   - T1 Lanczos spectral profile of a vector under $H$ / $PH$ (Gauss quadrature, 20–30 normal ops);
   - T4 Jacobian asymmetry via the plugin's JVP/VJP;
   - T5 init-sensitivity;
   - T3 held-out PE (`heldout_mask` exists).
2. **Estimator choice:** RED (or ADMM-PCG) with the existing learned denoiser as the default prior mode. The circulant
   metric is used only through prop:keepfp or the inner PCG. Theorems 1/3 and the toy say a white denoiser under a
   circulant PnP metric puts bias into the well-determined band.
3. **Denoiser v2** (`rustlib/tools/denoiser`):
   - σ-conditioned, jointly homogeneous ($D(\mu z,\mu\sigma)=\mu D(z,\sigma)$; FiLM of degree 0), O_h × global U(1);
   - trained as an independent-coupling flow-matching family, i.e. σ log-uniform, the Thm 8 loss;
   - noise mixture of §3.4;
   - optional potential parametrisation $D=I-\nabla g_\theta$ for exact RED;
   - data: RACLETTE + ph0001–0004 + ph0101–0103 diffeo variants with tissue variation (weight per the phantom plan);
     ph9001 never in training or validation.
   - In the velocity model the same network form acts on $(z,v)$: $z$ degree 1, $v$ degree 0.
4. **Flow uses:**
   - annealed RED (equivariant schedule, Prop. 9) for cold starts, then frozen σ;
   - patch ODE likelihood as a relative OOD flag on real data;
   - posterior samples for UQ at dev scale.
5. **SR:** D1 first (fine-grid inverse problem); D3 for any post-hoc network. Never D-free post-hoc SR as a
   quantitative output.

**Experiments** (each: question → metric → decision; dev scale first; config fingerprint checked; one factor at a
time; oracle arms labelled; practice baseline = LLR / wavelet).

| # | question | arms | metric | decision rule |
|---|---|---|---|---|
| E1 | Does the PnP circulant metric bias the well-determined band on our operator (Theorem 1 at scale)? | 2 mm ph0001, same learned D: PnP-circ, PnP-ZMSHR1, RED; each at its best strength (prop:strength sweep) | T1 band fraction of $Wc$; T8 band bias² (6 draws, circulant band approx.); lumen velocity slope; T3 | if RED's band bias² ≤ ½ PnP-circ's at equal or better velocity RMSE → RED becomes the default; if not, Theorem 1's commuting approximation fails at scale → measure $\|[P,H]\|$ |
| E2 | Is the current denoiser conservative enough for RED? | trained 7p_v2 on 20 patches at 3 σ | T4 asymmetry | < 0.05 → keep; 0.05–0.3 → RED usable with a measured objective gap; > 0.3 → train the gradient-step variant |
| E3 | Does operator-shaped training noise help inside the solver? | v2 white-only vs v2 §3.4 mixture (one factor) | held-out ph9001: lumen v RMSE, plane flow; validation loss on iterate residuals; T3 | adopt the mixture if v RMSE −10 % with no T3 degradation; add iterate fine-tuning only if the iterate-residual loss stays ≥ 2× the Gaussian loss |
| E4 | Is the hard split worth building? | RED vs hard split with the Kronecker/circulant band ($w\ge$ quantile) | E1 metrics | build only if band bias² drops ≥ 2× *and* total error is not worse (toy: equal error, 0 band bias) |
| E5 | Does σ-annealing (flow family) improve cold-start basins? | fixed-σ RED vs annealed→frozen RED, same final σ | final objective value, v RMSE, T5 between the two | adopt if v RMSE improves at a ≤ 1.3× pass count; if T5 shows differences only in the $Q_\varepsilon^\perp$ band, report them as prior-determined |
| E6 | SR hallucination control | 2 mm recon at $s=1.5$ (D1) vs post-hoc 4DFlowNet-like + D3 | phantom: near-wall v error, plane flow (oracle); real-data proxy: band-extrapolation conformal coverage at α = 0.1; T3 | a post-hoc SR is shippable only if coverage ≥ 0.85 on held-out cases *and* T3 is not worse; otherwise SR stays in-loop only |

Controls for all: a noise-free, inverse-crime-free run (T7: continuous 0.5 mm source → coarse recon operator, σ = 0)
to separate prior bias from noise; the scale-check $D(\mu z)=\mu D(z)$ to float precision; secant groups per
`red_prior.md` §5.

## 9. Sources

- Martin, Gagneux, Delon, … *PnP-Flow: Plug-and-play image restoration with flow matching*, ICLR 2025,
  [arXiv:2410.02423](https://arxiv.org/abs/2410.02423) [V: algorithm, Props. 1 and 4, experiments, limitations].
- Wang, Yu, Zhang, *Zero-shot image restoration using denoising diffusion null-space model (DDNM)*, ICLR 2023,
  [arXiv:2212.00490](https://arxiv.org/abs/2212.00490) [V: §3 method, DDNM+ noise handling].
- Kawar, Elad, Ermon, Song, *Denoising diffusion restoration models (DDRM)*, NeurIPS 2022,
  [arXiv:2201.11793](https://arxiv.org/abs/2201.11793) [A].
- Chen, Davies, *Deep decomposition learning for inverse imaging problems*, ECCV 2020,
  [arXiv:1911.11028](https://arxiv.org/abs/1911.11028), DOI 10.1007/978-3-030-58604-1_31 [A].
- Schwab, Antholzer, Haltmeier, *Deep null space learning for inverse problems*,
  [arXiv:1806.06137](https://arxiv.org/abs/1806.06137) [V, survey §7.11].
- Mardani, Song, Kautz, Vahdat, *A variational perspective on solving inverse problems with diffusion models
  (RED-diff)*, ICLR 2024, [arXiv:2305.04391](https://arxiv.org/abs/2305.04391) [A].
- Chung, Kim, Mccann, Klasky, Ye, *Diffusion posterior sampling (DPS)*, ICLR 2023,
  [arXiv:2209.14687](https://arxiv.org/abs/2209.14687) [V, survey §11.6].
- Wu, Sun, Chen, Zhang, Yue, Bouman, *PnP-DM*, [arXiv:2405.18782](https://arxiv.org/abs/2405.18782) [V, survey §11.3].
- Pokle, Muckley, Chen, Karrer, *Training-free linear image inverses via flows*, TMLR 2024,
  [arXiv:2310.04432](https://arxiv.org/abs/2310.04432) [A].
- Zhang et al., *Flow priors for linear inverse problems via iterative corrupted trajectory matching*, NeurIPS 2024,
  [arXiv:2405.18816](https://arxiv.org/abs/2405.18816) [A].
- Lipman et al., *Flow matching for generative modeling*, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) [A];
  Liu, Gong, Liu, *Rectified flow*, [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) [A]; Albergo, Boffi,
  Vanden-Eijnden, *Stochastic interpolants*, [arXiv:2303.08797](https://arxiv.org/abs/2303.08797) [A].
- Song et al., *Score-based generative modeling through SDEs*, [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) [A];
  Karras et al., *EDM*, [arXiv:2206.00364](https://arxiv.org/abs/2206.00364) [A]; Kingma et al., *Variational diffusion
  models*, [arXiv:2107.00630](https://arxiv.org/abs/2107.00630) [A].
- Efron, *Tweedie's formula and selection bias*, JASA 2011, DOI 10.1198/jasa.2011.tm11181 [A]; Kadkhodaie, Guth,
  Simoncelli, Mallat 2024, [arXiv:2310.02557](https://arxiv.org/abs/2310.02557) [V, survey §9.3]; Manor & Michaeli 2024
  [V, survey §9.1].
- Mohan et al., *Bias-free CNNs*, [arXiv:1906.05478](https://arxiv.org/abs/1906.05478) [V, survey §9.4]; Herbreteau et
  al., *Normalization-equivariant networks*, [arXiv:2306.05037](https://arxiv.org/abs/2306.05037) [V, survey §9.6].
- Romano, Elad, Milanfar, RED, [arXiv:1611.02862](https://arxiv.org/abs/1611.02862); Reehorst & Schniter,
  [arXiv:1806.02296](https://arxiv.org/abs/1806.02296); Hurault, Leclaire, Papadakis, gradient-step denoiser, ICLR 2022
  [V, survey §7.4].
- Hong, Xu, Hu, Fessler, *Provable preconditioned PnP (P²nP)*, IEEE TCI 2024,
  [arXiv:2405.03854](https://arxiv.org/abs/2405.03854) [V, `p2np_theory.md`].
- Gilton, Ongie, Willett, *Deep equilibrium architectures for inverse problems*,
  [arXiv:2102.07944](https://arxiv.org/abs/2102.07944) [V, survey §7.2].
- Gottschling, Antun, Hansen, Adcock, *The troublesome kernel*, SIAM Review 2025,
  [arXiv:2001.01258](https://arxiv.org/abs/2001.01258) [V, survey §10.2]; Bhadra et al., *On hallucinations in
  tomographic image reconstruction*, [arXiv:2012.00646](https://arxiv.org/abs/2012.00646) [V, survey §10.3]; Gupta et al.
  2024, [arXiv:2402.12727](https://arxiv.org/abs/2402.12727) [V, survey §10.5].
- Köhler, Klein, Noé, *Equivariant flows: exact likelihood generative learning for symmetric densities*, ICML 2020,
  [arXiv:2006.02425](https://arxiv.org/abs/2006.02425) [A; the theorem is proved here independently]; Klein, Krämer,
  Noé, *Equivariant flow matching*, NeurIPS 2023, [arXiv:2306.15030](https://arxiv.org/abs/2306.15030) [A].
- Chen, Tachella, Davies, *Equivariant imaging*, [arXiv:2103.14756](https://arxiv.org/abs/2103.14756) [A]; Terris et
  al., *Equivariant PnP*, CVPR 2024 [A].
- Eaton, *Group Invariance Applications in Statistics*, IMS 1989; Berger, *Statistical Decision Theory and Bayesian
  Analysis*, 2nd ed. 1985, §6.6 (best equivariant = generalised Bayes under right Haar measure) [textbook, from
  memory; check the section number].
- Nalisnick et al., *Do deep generative models know what they don't know?*, ICLR 2019,
  [arXiv:1810.09136](https://arxiv.org/abs/1810.09136) [A].
- Angelopoulos et al., *Image-to-image regression with distribution-free UQ*, ICML 2022,
  [arXiv:2202.05265](https://arxiv.org/abs/2202.05265) [A]; Wen, Ahmad, Schniter, *Task-driven UQ in inverse problems via
  conformal prediction*, ECCV 2024, [arXiv:2405.18527](https://arxiv.org/abs/2405.18527) [A].
- Ferdian et al., *4DFlowNet*, Front. Phys. 2020, DOI 10.3389/fphy.2020.00138 [A; the post-hoc SR baseline].
