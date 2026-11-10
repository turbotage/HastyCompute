# Compressed-sensing recovery with a learned denoiser: tangent-space theory

2026-11-09.

**Companion notes:**
- `p2np_theory.md`: Theorem B and the ZMSHR1 spectrum (Lemma A).
- `learned_prior_cs_literature.md`: the survey, cited below as **[Survey §x]**.

**Goal.** Statements of the form "if the pair (denoiser $D$, sampling $A$) satisfies X, then PnP recovers $x_0$ within … with probability ≥ …", for our iteration
$$x_{k+1}=D\big(x_k-\alpha P_k A^H(Ax_k-y)\big),\qquad y=Ax_0+n .$$

## 0. Summary

1. **Why templates that treat the denoiser as a Lipschitz map fail, and what replaces them.**
   - Theorems that treat $D$ as a Lipschitz map and ask for a RIP on the model set's secants (Liu 2021, Traonmilin, Joundi, Wolf) need the denoiser to be within $O(r/n)$ of a nonexpansive projection when undersampling is strong [Survey §13.2].
   - The only workable structure is the **tangent-space template**:
     - the denoiser's (secant) Jacobian is a rank-$d$ projector plus leakage $\eta$;
     - the data step is controlled only *between tangent spaces*.
   - This is Theorem B of the P²nP note. It is restated here in the sharper *pairwise* form, which is what the probability arguments need.
2. **Theorem L (local, non-uniform, probabilistic): the main result.** Fix the truth $x_0$ and let $T_0$ be the denoiser's $d$-dimensional tangent space there. Then:
   - if the number of i.i.d. shots (interleaves × all coils) satisfies $N_b\gtrsim\frac{\Phi_0}{\lambda_-}\log\frac{d}{p}$ (block coherence over restricted energy, i.e. total samples ≳ coherence × $d\log d$);
   - and the Jacobian leakage, Jacobian Lipschitzness and initial error are small enough (explicit tube condition);
   - then with probability $\ge1-p$, PnP-ISTA converges linearly to within
     $$\frac{\sigma\sqrt{d\lambda_+/N_b}}{\lambda_-}\ +\ \text{(leakage × full noise)}\ +\ \text{mismatch }\|D(x_0)-x_0\|,$$
     each divided by a restricted contraction margin.
   - The moving basis does not matter locally: within a tube of radius ≪ (Jacobian-Lipschitz)$^{-1}$, every secant Jacobian is close to the *truth's* tangent projector. A RIP is needed on $T_0$ only, a single $d$-dimensional subspace.
   - Sample complexity ∝ the local Jacobian rank. This matches the GMM-CS phase transition, which sits at the maximal local rank, not the union size [Survey §8.11].
3. **Theorem U (uniform, moving basis).** RIP holds simultaneously for all pairs from a family $\mathcal T$ of tangent spaces once
   $$N_b\gtrsim\frac{\Phi_{\mathcal T}}{\lambda_-t^2}\big(\log 2d+2\log\mathcal N(\mathcal T,\epsilon)+\log\tfrac1p\big),$$
   with $\log\mathcal N\approx k\log(\cdot)$ for a $k$-parameter family. This is the probabilistic half of the "moving sparsifying basis" problem. The other half, that a trained $D$ has this Jacobian structure far from the truth, is the open gap.
4. **Preconditioning changes the estimator, not only the path (Proposition G).**
   - In PnP the fixed point depends on $P$. Locally it equals the best linear unbiased estimator on the tangent space **iff $T_0$ is an invariant subspace of $P$**.
   - Consequences:
     - $P=I$ is optimal;
     - a density (circulant) preconditioner costs variance (Gauss–Markov excess);
     - ZMSHR1 is near-optimal because its rank-one direction lies in the tangent space;
     - a *tangent-restricted* preconditioner gives both one-step contraction ($\delta=0$) and the optimal estimator.
   - This is a rigorous local version of our no-DCW rule.
5. **Leakage tolerance scales with the restricted conditioning.**
   - Contraction needs $\delta+\eta(2+\delta)<1$. Without density compensation the spiral's restricted condition number $\kappa_{T_0}$ is large, $\delta\approx1-1/\kappa_{T_0}$, and the denoiser's leakage must satisfy $\eta\lesssim1/(3\kappa_{T_0})$.
   - With a good preconditioner the tolerance is $\approx1/3$.
   - So preconditioning matters for the *validity* of recovery, not just speed, because leakage accumulates over the many slow iterations.

## 1. Setting and notation

- **Shots.** $N_b$ shots (interleaf × readout × all $L$ coils, whitened), $\omega_1,\dots,\omega_{N_b}$ i.i.d. $\sim\nu$; the i.i.d. random-rotation model for golden-angle spirals.
  - Each shot has a block operator $B(\omega):\mathbb C^N\to\mathbb C^{L n_{ro}}$.
  - $H=A^HA=\sum_jB(\omega_j)^HB(\omega_j)$, $\bar H=\mathbb E_\omega B^HB$, so $\mathbb EH=N_b\bar H$. Deterministic shots (e.g. a fully sampled centre) are free to add [Survey §4.4, Def. 3.9].
- **Tangent at the truth.** $\Pi_0$ is an orthogonal projector of rank $d$ onto $T_0$ (the "tangent at the truth"). It is a property of $D$ and $x_0$ only, not of the sampling.
- **Restricted expected spectrum:** $\lambda_-=\lambda_{\min}(\Pi_0\bar H\Pi_0|_{T_0})$, $\lambda_+=\lambda_{\max}(\Pi_0\bar H\Pi_0|_{T_0})$.
- **Block coherence (variation):** $\Phi_0=\operatorname{ess\,sup}_\omega\|B(\omega)\Pi_0\|^2$.
  - Since $\mathbb E\|B\Pi_0\|_F^2=\mathrm{tr}\,\Pi_0\bar H\Pi_0\ge d\lambda_-$, we have $\Phi_0\ge d\lambda_-/\min(d,Ln_{ro})$.
  - Write $\Phi_0=\kappa_{\rm coh}\,d\lambda_-/(Ln_{ro})$ when $d>Ln_{ro}$; $\kappa_{\rm coh}\ge1$, and $=1$ for perfectly incoherent shots.
- **Noise:** whitened, $n\sim\mathcal{CN}(0,\sigma^2I)$.
- **Data step:** $M=I-\alpha PH$, $m=\|M\|$. For $P=I$, $\alpha\le2/\|H\|$: $m=1$.
- **Secant Jacobian:** $\bar J(z)=\int_0^1J_D(x_0+t(z-x_0))\,dt$, so $D(z)-D(x_0)=\bar J(z)(z-x_0)$.

## 2. Deterministic core (pairwise form of Theorem B)

**Theorem B′.** Suppose along the iteration $\bar J(z_k)=\Pi_k+E_k$ with $\Pi_k$ orthogonal projectors of rank $\le d$ and $\|E_k\|\le\eta_k$. Let:
- $\delta=\sup_k\|\Pi_kM_k\Pi_{k-1}\|$ (pairwise, consecutive tangent spaces);
- $m=\sup\|M_k\|$;
- $w_k=\alpha P_kA^Hn$, $W=\sup\|\Pi_kw_k\|$, $W'=\sup\|w_k\|$;
- $\varepsilon_0=\|D(x_0)-x_0\|$.

With $\eta=\sup\eta_k$:
$$\|e_{k+1}\|\le(\delta+\eta m)\|e_k\|+\eta m(\delta+m)\|e_{k-1}\|+W+(1+\delta+m)(\eta W'+\varepsilon_0),$$
so the conclusion of Theorem B holds: linear convergence when $\delta+\eta m(1+\delta+m)<1$, with the stated limsup.

*Proof.* As for Theorem B. The only change: the step bounding $\|\Pi_kM_ka_k\|$, with $a_k\in\mathrm{range}\,\Pi_{k-1}$, uses $\|\Pi_kM_k\Pi_{k-1}\|$ directly instead of passing through the union $U_k$. Since $\|\Pi_kM\Pi_{k-1}\|\le\|\Pi_{U}M\Pi_{U}\|$, this version is never worse. ∎

**Why pairwise.** $(T,T')\mapsto\|\Pi_TM\Pi_{T'}\|$ is Lipschitz in the projector distance:
$$\big|\|\Pi_TM\Pi_{T'}\|-\|\Pi_{\tilde T}M\Pi_{\tilde T'}\|\big|\le m\big(\|\Pi_T-\Pi_{\tilde T}\|+\|\Pi_{T'}-\Pi_{\tilde T'}\|\big).$$
Covering arguments therefore work. Sums of subspaces at small angles are ill-conditioned and would not.

## 3. Theorem L (local, non-uniform)

**Assumptions.**
- **(D1) Tangent structure at the truth.** $D\in C^1$ on $\bar B(x_0,R)$, and $\|J_D(x)-\Pi_0\|\le\eta_0+L_J\|x-x_0\|$ there.
  - $\eta_0=\|J_D(x_0)-\Pi_0\|$ is the leakage at the truth: Jacobian eigenvalues not near 0 or 1, plus non-symmetry.
  - Consequently $\|\bar J(z)-\Pi_0\|\le\eta_0+\tfrac{L_J}2\|z-x_0\|$.
- **(D2) Mismatch.** $\varepsilon_0=\|D(x_0)-x_0\|$. For a positively homogeneous $D$, $D(x_0)=J_D(x_0)x_0$, so (D1) and small $\varepsilon_0$ both say "$x_0\in T_0$, approximately".
- **(S) Sampling.** i.i.d. shots as in §1, and $P=I$, $\alpha=1/\|H\|$. General $P$: Remark L3.

**Theorem L.** Let $t\in(0,1)$, $p\in(0,1)$ and
$$N_b\ \ge\ \frac{2\,\Phi_0}{t^2\lambda_-}\log\frac{2d}{p}.$$
Then with probability $\ge1-p$ over the shots:

**(a) Restricted conditioning.** $\lambda_{\min}(\Pi_0H\Pi_0|_{T_0})\ge(1-t)N_b\lambda_-$, hence
$$\delta=\|\Pi_0(I-\alpha H)\Pi_0\|\le1-\frac{(1-t)N_b\lambda_-}{\|H\|}=:\delta_0 .$$

**(b) Recovery.** Define $\eta(r)=\eta_0+\tfrac{L_J}2(r+W')$ and $F(\eta)=W+(2+\delta_0)(\eta W'+\varepsilon_0)$. Suppose some $r\le R-W'$ satisfies the **tube condition**
$$\delta_0+\eta(r)(2+\delta_0)<1,\qquad F(\eta(r))\le\big(1-\delta_0-\eta(r)(2+\delta_0)\big)\,r .$$
Then every initialisation with $\|x_0'-x_0\|,\|x_1'-x_0\|\le r$ stays in the tube, converges linearly, and
$$\limsup_k\|x_k-x_0\|\le\frac{W+(2+\delta_0)\big(\eta_\infty W'+\varepsilon_0\big)}{1-\delta_0-\eta_\infty(2+\delta_0)}.$$
Here $\eta_\infty=\eta_0+\tfrac{L_J}2(\limsup\|e_k\|+W')$; it is self-consistent and decreases with the error.

**(c) Noise terms.** With additional probability $\ge1-e^{-u}$:
$$W=\alpha\|\Pi_0A^Hn\|\le\alpha\sigma\Big(\sqrt{d(1+t)N_b\lambda_+}+\sqrt{2(1+t)N_b\lambda_+u}\Big),$$
$$W'=\alpha\|A^Hn\|\le\alpha\sigma\Big(\sqrt{\mathrm{tr}H}+\sqrt{2\|H\|u}\Big),$$
where the bound on $W$ uses the upper Chernoff event (also within probability $1-p$, constants adjusted).

**Reading.** With $\|H\|\approx N_b\lambda_{\max}(\bar H)$ and $\eta\to0$:
$$\limsup\|x_k-x_0\|\lesssim\frac{\sigma}{\sqrt{N_b}}\cdot\frac{\sqrt{d\,\lambda_+}}{(1-t)\lambda_-}+\frac{(2+\delta_0)\lambda_{\max}(\bar H)}{(1-t)\lambda_-}\,\varepsilon_0 .$$
- The first term is the classical CS noise term: $\sqrt d$, decaying like $1/\sqrt{N_b}$, times restricted conditioning.
- If $d>Ln_{ro}$, the sample condition reads **total samples $N_bLn_{ro}\gtrsim\kappa_{\rm coh}\,d\log(2d/p)/t^2$**.
- If shots are rich ($Ln_{ro}\ge d$), it reads **number of shots $\gtrsim(\Phi_0/\lambda_-)\log(2d/p)$**.

*Proof.*
1. **(a)** $X_j=\Pi_0B(\omega_j)^HB(\omega_j)\Pi_0|_{T_0}$ are independent PSD $d\times d$ matrices with $\lambda_{\max}(X_j)\le\Phi_0$ and $\lambda_{\min}(\sum\mathbb EX_j)=N_b\lambda_-$. Matrix Chernoff (Tropp 2012, Thm 1.1) gives $\Pr[\lambda_{\min}(\sum X_j)\le(1-t)N_b\lambda_-]\le d\,e^{-t^2N_b\lambda_-/(2\Phi_0)}\le p/2$. The upper tail likewise, with exponent $t^2N_b\lambda_+/(3\Phi_0)$, which is smaller only by a constant. On $T_0$, $\Pi_0(I-\alpha H)\Pi_0$ has eigenvalues in $[1-\alpha\lambda_{\max},\,1-\alpha\lambda_{\min}]\subset[0,\delta_0]$, since $\alpha\le1/\|H\|$.
2. **(b)** Apply Theorem B′ with $\Pi_k\equiv\Pi_0$, so the pairwise $\delta$ is the single-subspace $\delta$, and $m=1$. Leakage $\eta_k\le\eta_0+\tfrac{L_J}2\|z_k-x_0\|$, with $\|z_k-x_0\|\le\|e_k\|+W'$.
   - Induction: if $\|e_j\|\le r$ for $j\le k$, then $\eta_j\le\eta(r)$ and $\|e_{k+1}\|\le(c_1+c_2)r+F\le r$ by the tube condition. Also $z_j\in\bar B(x_0,R)$, so (D1) applies.
   - Then the linear recursion of Theorem B′ gives the limsup.
3. **(c)** $\Pi_0A^Hn\sim\mathcal{CN}(0,\sigma^2\Pi_0H\Pi_0)$. Gaussian concentration of the norm (Lipschitz constant $\|\Pi_0H\Pi_0\|^{1/2}$) and $\mathrm{tr}\,\Pi_0H\Pi_0\le d\,\lambda_{\max}$. ∎

**Remark L1 (the moving basis does not matter locally).**
- Inside the tube, every secant Jacobian is within $\eta(r)$ of the *fixed* projector $\Pi_0$. The iterates' local bases all collapse onto the truth's tangent space up to curvature × radius ($L_J r$).
- The RIP is needed on **one** $d$-dimensional subspace, and the guarantee is non-uniform: for this $x_0$, with high probability over the shots.
- This parallels "support known ⇒ $s$ measurements" in sparse CS, and the local lower bound at the truth of Golbabaee–Davies [Survey §8.9].
- The price is the tube condition: $r\lesssim(1-\delta_0)/(L_J(2+\delta_0))$. Recovery needs initialisation within that radius, which is the theoretical role of the warm start.

**Remark L2 (leakage tolerance and the role of preconditioning).** The tube condition needs $\eta(2+\delta_0)<1-\delta_0$.
- For non-density-compensated spiral, $1-\delta_0=(1-t)\lambda_-/\lambda_{\max}(\bar H)\approx1/\kappa_{T_0}$, the restricted condition number of the expected normal operator. Since $\lambda_{\max}(\bar H)$ is dominated by k-space-centre dwell, this is large.
- Hence $\eta\lesssim1/(3\kappa_{T_0})$: a denoiser whose Jacobian deviates from a projector by more than $\sim1/(3\kappa)$ can make the error stall or drift during the ~$\kappa$ slow iterations.
- A preconditioner with $\alpha P\bar H\approx I$ on $T_0$ raises the tolerance to $\approx1/3$.

**Remark L3 (general $P$).** Replace $\delta_0$ by $\delta(P)=\|\Pi_0(I-\alpha PH)\Pi_0\|$ and $m$ by $\|I-\alpha PH\|$ (for dynamic $P_k$, use Lemma A of the P²nP note), and the noise by $\alpha\|\Pi_0PA^Hn\|$. Proposition G below shows what $P$ does to the *limit*.

## 4. Proposition G: preconditioning changes the PnP estimator

**Setting.** Idealised local model: $D(z)=\Pi_0z+c$ with $\Pi_0c=0$, $x_0\in c+T_0$ (so $\eta=\varepsilon_0=0$), and a fixed $P\succ0$.

**Fixed points.** The fixed points $\hat x_P$ of $x=D(x-\alpha PA^H(Ax-y))$ satisfy $\hat x_P\in c+T_0$ and
$$\Pi_0P\big(H\hat x_P-A^Hy\big)=0\quad\Longrightarrow\quad\hat x_P-x_0=\big(\Pi_0PH\Pi_0\big)^{-1}_{T_0}\Pi_0PA^Hn=:K_Pn .$$

**Gauss–Markov.** $K_PA\Pi_0=\Pi_0$: unbiased on $T_0$. Hence
$$\mathrm{Cov}(\hat x_P)=\sigma^2K_PK_P^H\ \succeq\ \sigma^2(\Pi_0H\Pi_0)^{-1}_{T_0},$$
with equality iff $K_P=(\Pi_0H\Pi_0)^{-1}\Pi_0A^H$.
- **Sufficient:** $\Pi_0P\Pi_0^\perp=0$ ($T_0$ an invariant subspace of $P$). Then $\Pi_0P=Q$ on $T_0$, and $K_P=(Q\,\Pi_0H\Pi_0)^{-1}Q\,\Pi_0A^H$ is the BLUE.
- **Generically necessary:** when $\mathrm{range}(A^H)$ is not confined to a subspace that $\Pi_0P\Pi_0^\perp$ annihilates.

**Consequences.**
- $P=I$ (PnP-ISTA) and any scalar $P$ give the BLUE on the tangent space.
- **Density / circulant preconditioners** do not commute with a signal-adaptive $T_0$. Their PnP limit is an *oblique* restricted least-squares estimator with excess variance $\sigma^2(K_PK_P^H-(\Pi_0H\Pi_0)^{-1})$.
  - Unlike in convex optimisation, the preconditioner here changes the answer, not only the path.
  - Density compensation *in the data fit* is the same phenomenon (weighted LS with weights ≠ noise precision). This makes our no-DCW rule a theorem in the local model.
- **ZMSHR1** ($P_k=\tau_kI+\lambda_r\hat u\hat u^H$, Lemma A) leaves $T_0$ invariant iff $\hat u\in T_0$ or $\hat u\perp T_0$.
  - Its rank-one direction is built from the secant $s_k=e_k-e_{k-1}$, which lies in $T_0$ up to leakage. So ZMSHR1 is near-optimal: excess variance $O(\|\Pi_0^\perp\hat u\|)$.
- **Tangent-restricted preconditioner** $P_T=\Pi_0(\Pi_0H\Pi_0)^{-1}\Pi_0+\tau\Pi_0^\perp$:
  - $T_0$ is invariant, so the limit is the BLUE;
  - $\Pi_0P_TH\Pi_0=\Pi_0$, so $\delta(P_T)=0$ at $\alpha=1$: one-step contraction on $T_0$.
  - It is the optimal local preconditioner in both senses. It is buildable from a few Lanczos steps on $J_D$ ($\Pi_0$) and CG on $\Pi_0H\Pi_0$ (dimension $d$).

## 5. Theorem U (uniform over a tangent family: the moving basis)

**Setting.**
- **Family.** $\mathcal T$ is a family of $d$-dimensional subspaces with covering numbers $\mathcal N(\mathcal T,\epsilon)$ in projector norm. Example: $\{T(x):x\in\mathcal X\}$ for a $k$-dimensional signal set $\mathcal X$ with $x\mapsto\Pi_{T(x)}$ being $L_T$-Lipschitz, giving $\log\mathcal N\le k\log(3L_T\mathrm{diam}\mathcal X/\epsilon)$.
- **Quantities.**
  - Coherence $\Phi_{\mathcal T}=\operatorname{ess\,sup}_\omega\sup_{T\in\mathcal T}\|B(\omega)\Pi_T\|^2$.
  - $\lambda_{\pm}$ are inf / sup over $T\in\mathcal T$ of the restricted expected spectrum.
  - Expected pairwise term $\bar\delta_{\mathcal T}=\sup_{T,T'}\|\Pi_T(I-\alpha N_b\bar H)\Pi_{T'}\|$: the deterministic design or density bias. It is 0 for isotropic $\alpha N_b\bar H=I$.

**Theorem U.** With $P=I$, if
$$N_b\gtrsim\frac{\Phi_{\mathcal T}\,(1+\lambda_+/\lambda_-)}{\lambda_-\,t^2}\Big(\log2d+2\log\mathcal N\big(\mathcal T,\tfrac{t}{4}\cdot\tfrac{(1-t)\lambda_-}{\lambda_{\max}(\bar H)}\big)+\log\tfrac1p\Big),$$
then with probability $\ge1-p$:
$$\sup_{T,T'\in\mathcal T}\|\Pi_T(I-\alpha H)\Pi_{T'}\|\le\bar\delta_{\mathcal T}+\big(\text{relative deviation }t\text{ in units of }\alpha N_b\lambda_+\big).$$
Hence Theorem B′ applies to *any* iteration whose secant Jacobians stay within $\eta$ of projectors onto members of $\mathcal T$, wherever in $\mathcal T$ the iterates wander. This is uniform recovery.

*Proof.*
1. For a fixed pair, $\Pi_T(\alpha H-\alpha\mathbb EH)\Pi_{T'}=\sum_j(Y_j-\mathbb EY_j)$ with $Y_j=\alpha\Pi_TB_j^HB_j\Pi_{T'}$ (rectangular, $d\times d$). We have $\|Y_j\|\le\alpha\Phi_{\mathcal T}$ and variance proxy $\le\alpha^2N_b\Phi_{\mathcal T}\lambda_+$. Matrix Bernstein (Tropp 2012, Thm 1.6) then gives deviation $\le\tau$ with probability $\ge1-2d\exp\!\big(-\tfrac{\tau^2/2}{\alpha^2N_b\Phi_{\mathcal T}\lambda_++\alpha\Phi_{\mathcal T}\tau/3}\big)$.
2. Take a union bound over an $\epsilon$-net of pairs ($\mathcal N^2$ of them), and use the Lipschitz property of §2 with $m=1$ to pass from the net to all pairs ($+2\epsilon$).
3. Choose $\tau$ and $\epsilon$ proportional to the margin $1-\bar\delta_{\mathcal T}$. ∎

**What remains open (the honest gap).**
- Theorem U is the *probabilistic* half of the moving-basis problem; it is a standard union-of-subspaces / manifold-RIP argument, with Adcock-style block coherence and our non-Cartesian multi-coil shots.
- The *deterministic* half is unproven: that a trained $D$, *far* from the truth, has secant Jacobians within $\eta$ of tangent projectors.
  - Along a long segment, $\bar J$ averages the projectors of all tangent spaces crossed. For a curved model set that is $\eta\sim\|z-x_0\|/\text{reach}$, which brings back the tube of Theorem L.
  - Exceptions:
    - (i) piecewise-linear denoisers whose pieces' Jacobians are projectors and whose model set is a union of flat cones (bias-free and normalization-equivariant CNNs are piecewise linear [Survey §9.4, §9.6], but their piece-Jacobians are only approximately projectors);
    - (ii) a curvature-corrected analysis in the style of Wei et al. [Survey §8.10], with $\|(I-\Pi_{T(x_k)})(x_k-x_0)\|\le\|x_k-x_0\|^2/\text{reach}$. This is quadratic, hence harmless locally and the right tool for a global result. Not done here.
  - Global convergence from a cold start (the "support identification" phase) is the remaining theoretical problem. Theorem L covers the final, accuracy-determining phase.

## 6. What to measure (when GPU work is allowed again), and the predictions

On phantom truth $x_0$ (ph0001, 2 mm, one frame), with our denoiser:

| quantity | method | prediction / use |
|---|---|---|
| $d$ | Hutchinson $\mathrm{tr}J_D(x_0)$ and $\|J_D\|_F^2$ (~20 JVPs) | sample complexity ∝ $d$; Mohan et al. observe $d\propto1/\sigma$ |
| $\Pi_0$, $\eta_0$ | block Lanczos on $\tfrac12(J+J^H)$ (JVP + VJP), top-$d$ eigenspace, residual spectrum, $\|J-J^H\|$ | leakage tolerance $\eta_0\lesssim1/(3\kappa_{T_0})$ |
| $\lambda_\pm$, $\kappa_{T_0}$ | Lanczos on $\Pi_0H\Pi_0$ | contraction margin $1-\delta_0$; P vs no-P |
| $\Phi_0$, $\kappa_{\rm coh}$ | per-shot $\|B(\omega)\Pi_0\|^2$ over all shots | predicted shots needed vs actual |
| $L_J$ | finite differences of $J_D$ along $e_k$ | tube radius → required warm-start accuracy |
| $\varepsilon_0$ | $\|D(x_0)-x_0\|$ | mismatch floor |
| Prop. G | limits with $P=I$ vs circulant vs ZMSHR1, same data, noise-only error | excess variance of oblique estimators |

**Falsifiable predictions.**
1. The error floor vs acceleration should turn up when total samples fall below $\sim\kappa_{\rm coh}d\log d$.
2. With a fixed number of iterations, preconditioning lowers the error *only* while the leakage condition is violated without it.
3. A circulant preconditioner's limit has higher noise than $P=I$'s limit (same denoiser, same data, iterated to convergence).
4. The tangent-restricted preconditioner reaches $P=I$'s limit in a few iterations.

## 7. Relation to the literature

- **Theorem L** is the denoiser analogue of "oracle-support least squares" and of Golbabaee–Davies' local lower bound.
  - New elements: the Jacobian-projector + leakage + Jacobian-Lipschitz formulation; the multi-coil non-Cartesian block-Chernoff sample condition; the explicit tube.
  - Sample complexity ∝ local rank: cf. Renna et al. (GMM) [Survey §8.11]. The survey found no denoiser-based result of this kind for Fourier sampling [Survey §14, gap 1].
- **Theorem B′** is template (iii) [Survey §13.1]. The survey lists its "sharper version" of Proposition L as unproven [Survey §13.3]; this note supplies it, at the level of the stated assumptions.
- **Theorem U** is a standard union-of-subspaces argument (Blumensath–Davies; Adcock–Cardenas–Dexter block variation) specialised to tangent families and pairwise constants.
- **Proposition G** is elementary (Gauss–Markov). We have not found it stated for PnP preconditioning; the closest is Joundi et al.'s rate-vs-$\|Le\|$ trade-off [Survey §6.3], which bounds rather than characterises the estimator.
- **Not covered:**
  - the denoiser satisfying (D1) is an assumption with empirical support (approximately symmetric, projector-like Jacobians [Survey §9.3–9.5]), not a theorem;
  - global convergence;
  - momentum;
  - phantom-to-patient prior shift [Survey §14, gap 7].
