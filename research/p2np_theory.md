# Tighter analysis of preconditioned plug-and-play (P²nP)

2026-11-09. Reference: T. Hong, X. Xu, J. Hu, J. A. Fessler, *Provable Preconditioned Plug-and-Play Approach for Compressed
Sensing MRI Reconstruction*, IEEE TCI 10 (2024) 1476–1488, arXiv:2405.03854 (text in
`data/hasty/external/papers_p2np/`). Below, "the paper" means this reference.

Notation:
- $f(x)=\tfrac12\|Ax-y\|^2$, $H=A^HA\succeq0$, $L=\|H\|$, step $\alpha$;
- denoiser $D$; iteration $x_{k+1}=D(x_k-\alpha P_k\nabla f(x_k))$; data step $M_k=I-\alpha P_kH$;
- $\|\cdot\|$ is the Euclidean norm and the induced operator norm.

## 0. Summary

1. **Both theorems of the paper are vacuous in compressed sensing.**
   - They need $q=(1+\epsilon)\,\rho(I-\alpha P H)<1$ with $\epsilon>0$.
   - If $A$ has a nontrivial null space, $I-\alpha PH$ has eigenvalue $1$ on it, so $q\ge 1+\epsilon>1$.
   - If $A$ is injective but ill-conditioned (multi-coil spiral), $q<1$ requires $\epsilon\lesssim\alpha\lambda_{\min}(PH)$, i.e. an essentially nonexpansive denoiser.
   - The proofs bound $D$ and the data step separately. Contraction in CS only comes from their *product*: the prior removes exactly the directions the data cannot see.
2. **Exact spectrum of the ZMSHR1 preconditioner (§2, new, verified numerically).**
   - $P_k$ has eigenvalue $\rho_k(1-\sin\vartheta_k)$ with multiplicity $N-1$, and $\rho_k(1+\sin\vartheta_k)$ once.
   - Here $\rho_k=\|s\|^2/\langle s,v\rangle$ is the BB1 step and $\vartheta_k=\angle(s_k,v_k)$.
   - The two eigenvalues have arithmetic mean BB1, geometric mean $\|s\|/\|v\|$ and harmonic mean BB2 $=\langle s,v\rangle/\|v\|^2$.
   - Consequences:
     - the paper's bound $P_k\preceq\frac{\delta+1}{\delta\theta_1}$ improves to $P_k\preceq\frac{1+\sqrt{1-\theta_1/\theta_2}}{\theta_1}<\frac2{\theta_1}$;
     - the safeguard $\delta$ can never fire unless $\delta\ge\tfrac12\sqrt{\theta_1/\theta_2}$, so with the paper's parameters ($\delta=10^{-8}$, $\theta_1=2\cdot10^{-6}$, $\theta_2=200$) it is never active.
     - The constant of the stability bound (their Theorem 2) shrinks from $\approx\frac{1}{\delta\theta_1}=5\cdot10^{13}$ to $\le\frac{2}{\theta_1}=10^{6}$, a factor $5\cdot10^{7}$ ($\approx\frac1{2\delta}$).
     - The bound's dependence on $\delta$ was an artefact of the proof.
3. **Tightened stability theorem (§3) under the paper's own assumptions.**
   - Weyl instead of the triangle inequality for $\|P_k-P_*\|$.
   - The initial gradient bound $R$ is replaced by $\limsup\|\nabla f(x_k)\|$; the paper's Fig. 7b shows the gradient decaying to ~0.
4. **The right objects (§4).**
   - The exact local rate is $\rho\big(J_D(z_*)\,(I-\alpha P_*H)\big)$ (Ostrowski), not $(1+\epsilon)\rho(I-\alpha P_*H)$.
   - A **restricted contraction theorem** works when the denoiser's (secant) Jacobian is a rank-$d$ tangent projector plus leakage $\eta$, and the data step satisfies a RIP-type bound $\delta$ on unions of two consecutive tangent spaces:
     - PnP converges linearly whenever $\delta+\eta m(1+\delta+m)<1$, with $m=\sup\|M_k\|$ ($m=1$ for plain PnP-ISTA);
     - the iterates approach the **truth** within $\frac{W+(1+\delta+m)(\eta W'+\varepsilon_0)}{1-\delta-\eta m(1+\delta+m)}$, where $W$ is the noise back-projected onto a $d$-dimensional tangent space and $\varepsilon_0=\|D(x_0)-x_0\|$ is the model mismatch.
     - This holds in the undersampled regime, with dynamic preconditioners, and needs no fixed point $x_*$.
5. **Why tight bounds via $(\delta,\theta_1,\theta_2)$ hurt in practice (§5).**
   - Capping $\|P_k\|$ through $\theta_1$ dilutes the secant pair ($a>0$), so $P_k\to$ scaled identity and convergence slows.
   - The restricted analysis shows the cap is unnecessary: whenever the secant step $s_k$ lies in a tangent union with RIP constant $\delta_U$, the a-search accepts $a=0$ and $P_k\preceq\frac{2}{1-\delta_U}$, *independent of $\theta_1$*.
   - Large eigenvalues of $P_k$ only occur along directions outside the tangent spaces, which the denoiser's Jacobian removes.
   - The design target for a preconditioner is therefore the **restricted** spectrum $\|\Pi_U(I-\alpha PH)\Pi_U\|$, not $\rho(I-\alpha PH)$. ZMSHR1 learns curvature exactly along $s_k\in U_k$, which explains why P²nP-D works much better than its bound says.

## 1. The paper's results (verbatim content)

- **Algorithm 1 (fixed $P\succ0$):** $x_{k+1}=D_\sigma(x_k-\alpha P\nabla f(x_k))$.
- **Assumption 1:** $\|D_\sigma(x)-D_\sigma(y)\|\le(1+\epsilon)\|x-y\|$ for all $x,y$, with $\epsilon>0$.
- **Theorem 1:** if $r_0=(1+\epsilon)\rho(I-\alpha PH)<1$, the iterates converge to a fixed point with rate $\le r_0$.
  - Proof: $\|x_{k+1}-x_*\|\le(1+\epsilon)\|I-\alpha PH\|\,\|x_k-x_*\|$.
  - The statement uses $\rho$, but the proof uses $\|\cdot\|$. These agree only if $PH$ is normal, e.g. $P=p(H)$ polynomial (their F-1, F-Cheb). For a general $P\succ0$, $\|I-\alpha PH\|$ can exceed $\rho$.
- **Algorithm 2 (ZMSHR1).** Given $s=x_k-x_{k-1}$ and $m=\nabla f(x_k)-\nabla f(x_{k-1})=Hs$:
  1. Find the smallest $a\in[0,1]$ such that $v=as+(1-a)m$ satisfies $\theta_1\le\frac{\langle s,v\rangle}{\langle s,s\rangle}$ and $\frac{\langle v,v\rangle}{\langle s,v\rangle}\le\theta_2$.
  2. Set $\tau=\frac{\langle s,s\rangle}{\langle s,v\rangle}-\sqrt{\big(\frac{\langle s,s\rangle}{\langle s,v\rangle}\big)^2-\frac{\langle s,s\rangle}{\langle v,v\rangle}}$ and $u=s-\tau v$.
  3. If $\langle u,v\rangle\le\delta\|u\|\|v\|$, set $u=0$.
  4. $P_k=\tau I+\frac{uu^H}{\langle u,v\rangle}$.
- **Lemma 1:** $\frac1{2\theta_2}<\tau_k\le\frac1{\theta_1}$ and $\frac1{2\theta_2}I\preceq P_k\preceq\frac{\delta+1}{\delta\theta_1}I$.
- **Theorem 2.** Let $0\prec P_*\preceq\lambda_*I$, $q=(1+\epsilon)\rho(I-\alpha P_*H)<1$, $\|\nabla f(x_k)\|\le R$, and $x_*=D(x_*-\alpha P_*\nabla f(x_*))$. Then
  $$\|x_{k+1}-x_*\|\le q^k\|x_1-x_*\|+\frac{(1+\epsilon)\alpha(\delta+1+\delta\theta_1\lambda_*)}{\delta\theta_1(1-q)}R.$$
  - Proof: $\|(P_m-P_*)\nabla f(x_m)\|\le\beta R$ with $\beta=\max_m\|P_m\|+\|P_*\|$.
- **Their remark:** "the algorithm may be very slow if we tune the parameters to enforce a small error bound … the parameters which perform well in experiments may result in an extremely high upper bound, suggesting that further analysis could lead to tighter results."

## 2. Exact spectrum of ZMSHR1

**Lemma A.** Let $s,v\in\mathbb C^N$ with $\langle s,v\rangle>0$, and define:
- $\rho=\frac{\|s\|^2}{\langle s,v\rangle}$;
- $\cos\vartheta=\frac{\langle s,v\rangle}{\|s\|\|v\|}\in(0,1]$;
- $\tau$ and $u$ as in Algorithm 2.

Then:
1. $\tau=\rho(1-\sin\vartheta)=\frac{\langle s,v\rangle}{\|v\|^2(1+\sin\vartheta)}$;
2. $\langle u,v\rangle=\langle s,v\rangle\frac{\sin\vartheta}{1+\sin\vartheta}\ge0$;
3. $\cos\angle(u,v)=\frac{\cos\vartheta}{\sqrt{2(1+\sin\vartheta)}}$;
4. without the safeguard, $P=\tau I+\frac{uu^H}{\langle u,v\rangle}$ satisfies $Pv=s$ and has eigenvalues $\rho(1-\sin\vartheta)$ ($N-1$ times) and $\rho(1+\sin\vartheta)$ (eigenvector $u$).

*Proof.* Write $r_1=\langle s,v\rangle/\|v\|^2$, $r_2=\|s\|^2/\|v\|^2$ (so $\rho=r_2/r_1$) and $t=r_1/\rho=\cos^2\vartheta$.

**Item 1.** Then $\tau=\rho-\sqrt{\rho^2-\rho r_1}=\rho(1-\sqrt{1-t})=\rho(1-\sin\vartheta)$. Using $1-\sin\vartheta=\cos^2\vartheta/(1+\sin\vartheta)$ gives the second form.

**Item 4, secant.** $Pv=\tau v+u\frac{\langle u,v\rangle}{\langle u,v\rangle}=\tau v+s-\tau v=s$.

**Item 4, eigenvalues.** $P$ is $\tau I$ on $u^\perp$; let $\lambda=\tau+\|u\|^2/\langle u,v\rangle$ be its eigenvalue on $u$. With $\hat v=v/\|v\|$ and weight $w=|\langle\hat u,\hat v\rangle|^2$, the secant gives
$$\langle\hat v,P\hat v\rangle=r_1,\qquad\|P\hat v\|^2=r_2.$$
Hence $\tau(1-w)+\lambda w=r_1$ and $\tau^2(1-w)+\lambda^2w=r_2$. Eliminating $w$:
$$\lambda=\tau+\frac{(r_2-r_1^2)+(r_1-\tau)^2}{r_1-\tau}.$$
Substituting $r_1=\rho t$, $r_2-r_1^2=\rho^2t(1-t)$ and $r_1-\tau=\rho c(1-c)$ with $c=\sin\vartheta$ gives
$$\lambda=\tau+\rho c\,\frac{1-c^2+(1-c)^2}{1-c}=\tau+2\rho c=\rho(1+c).$$

**Items 2 and 3** follow from $\|u\|^2=2\tau\big(\|s\|^2\|v\|^2-\langle s,v\rangle^2\big)/\langle s,v\rangle$, a consequence of $\tau$ solving $\tau^2-2\rho\tau+r_2=0$. ∎

Numerical check: random complex $s$, singular or ill-conditioned $H$, secant steps in the null space; 2000 trials. The maximum relative eigenvalue error is $2\cdot10^{-9}$ and the $\cos\angle(u,v)$ error $6\cdot10^{-13}$ (`zmshr1_spec.py`, session scratch).

**Corollaries.**
- **Mean structure.** The two eigenvalues have arithmetic mean $\frac{\|s\|^2}{\langle s,v\rangle}$ (BB1), harmonic mean $\frac{\langle s,v\rangle}{\|v\|^2}$ (BB2) and geometric mean $\frac{\|s\|}{\|v\|}$.
  - ZMSHR1 takes the conservative step $\tau\le$ BB2 on $u^\perp$ and the long step $\ge$ BB1 along $u$.
- **Tight bounds.** The a-search guarantees $\cos^2\vartheta=\frac{\langle s,v\rangle}{\|s\|^2}\cdot\frac{\langle s,v\rangle}{\|v\|^2}\ge\frac{\theta_1}{\theta_2}$, so $\sin\vartheta\le\sqrt{1-\theta_1/\theta_2}$. Therefore
  $$\frac{1}{\theta_2\,(1+\sqrt{1-\theta_1/\theta_2})}\;\le\;\lambda_{\min}(P_k)=\frac{\langle s,v\rangle}{\|v\|^2(1+\sin\vartheta)},\qquad
  \lambda_{\max}(P_k)=\rho(1+\sin\vartheta)\le\frac{1+\sqrt{1-\theta_1/\theta_2}}{\theta_1}<\frac{2}{\theta_1}.$$
  - Condition number: $\kappa(P_k)=\frac{(1+\sin\vartheta)^2}{\cos^2\vartheta}\le\frac{4\theta_2}{\theta_1}$.
  - No $\delta$ anywhere.
- **The safeguard is inert.** By item 3, $\cos\angle(u,v)\ge\frac{\cos\vartheta}{2}\ge\frac12\sqrt{\theta_1/\theta_2}$, so the test $\langle u,v\rangle\le\delta\|u\|\|v\|$ can only fire if $\delta\ge\frac12\sqrt{\theta_1/\theta_2}$, which is $5\cdot10^{-5}$ for the paper's parameters. If the a-search finds no admissible $a$ and falls back to $a=1$, then $v=s$, $\vartheta=0$, $u=0$ and $P=I$.

## 3. Tightened stability (the paper's Theorem 2, same assumptions)

**Theorem 2′.** Under the assumptions of Theorem 2, with
$$\beta:=\sup_k\|P_k-P_*\|\le\max\big(\lambda_{\max}(P_k)-\lambda_{\min}(P_*),\ \lambda_{\max}(P_*)-\lambda_{\min}(P_k)\big)\le\max\Big(\frac{2}{\theta_1},\lambda_*\Big)$$
(Weyl: the eigenvalues of $P-Q$ lie in $[\lambda_{\min}(P)-\lambda_{\max}(Q),\ \lambda_{\max}(P)-\lambda_{\min}(Q)]$), we have
$$\|x_{k+1}-x_*\|\le q^k\|x_1-x_*\|+(1+\epsilon)\alpha\sum_{j=1}^{k}q^{k-j}\|(P_j-P_*)\nabla f(x_j)\|,$$
$$\limsup_k\|x_k-x_*\|\le\frac{(1+\epsilon)\,\alpha\,\beta}{1-q}\ \limsup_k\|\nabla f(x_k)\|.$$

The limsup step is the discrete Grönwall argument $e_{k+1}\le qe_k+c_k\Rightarrow\limsup e\le\limsup c/(1-q)$.

| constant multiplying $(1+\epsilon)\alpha/(1-q)$ | paper | here |
|---|---|---|
| preconditioner spread $\beta$ | $\frac{\delta+1}{\delta\theta_1}+\lambda_*\approx5\cdot10^{13}$ | $\max(2/\theta_1,\lambda_*)=10^6$ |
| gradient factor | $R=\|\nabla f(x_1)\|$ (the maximum) | $\limsup\|\nabla f(x_k)\|$ (→ ~0 in their Fig. 7b) |

**Refinements.**
- **Gradient at the fixed point.** Splitting $\nabla f(x_k)=g_*+He_k$ with $g_*=\nabla f(x_*)$ and $e_k=x_k-x_*$ gives
  $$e_{k+1}\le\big(q+(1+\epsilon)\alpha\beta L\big)e_k+(1+\epsilon)\alpha\beta\|g_*\|.$$
  The radius then scales with the gradient *at the fixed point*, the data inconsistency the prior enforces, instead of $R$. This needs $q+(1+\epsilon)\alpha\beta L<1$.
- **Rank-one part.** For $P_*=\tau_*I$ the rank-one part of $P_j-P_*$ acts on the gradient only through $|\langle u_j,\nabla f(x_j)\rangle|=|\langle Au_j,Ax_j-y\rangle|$:
  $$\|(P_j-\tau_*I)\nabla f(x_j)\|\le|\tau_j-\tau_*|\,\|\nabla f(x_j)\|+2\rho_j\sin\vartheta_j\,\|A\hat u_j\|\,\|Ax_j-y\|.$$
  - A large eigenvalue along a near-null direction ($\|A\hat u\|\approx0$) costs nothing.
  - Worst-case operator norms cannot see this.

All of §3 still needs $q<1$, i.e. it inherits the vacuity of item 1 in §0. §4 removes that.

## 4. The right objects: Jacobian-coupled and restricted contraction

**Local rate (exact).** Let $T(x)=D(x-\alpha P_*\nabla f(x))$ and assume $D$ is $C^1$ near $z_*=x_*-\alpha P_*\nabla f(x_*)$. Then
$$T'(x_*)=J_D(z_*)\,(I-\alpha P_*H),$$
and by Ostrowski's theorem $x_*$ attracts iterates with asymptotic rate $\rho\big(J_D(z_*)(I-\alpha P_*H)\big)$ whenever that is $<1$.
- The paper's $r_0$ is an upper bound for it, through $\rho(JM)\le\|J\|\|M\|$, and only when $\|M\|=\rho(M)$.
- In CS, $\rho(M)=1$ but $\rho(JM)<1$ is typical: a trained denoiser's Jacobian is close to a symmetric projector onto a low-dimensional, signal-adaptive subspace (Kadkhodaie–Simoncelli), and annihilates the null-space directions on which $M$ is the identity.
- Measurable: power iteration with JVPs of $D$.

**Setting for the global result.**
- $y=Ax_0+n$, $e_k=x_k-x_0$ (error to the **truth**).
- $w_k=\alpha P_kA^Hn$ (back-projected noise), $\varepsilon_0=\|D(x_0)-x_0\|$ (prior mismatch: the truth is not a fixed point of $D$).
- Secant Jacobian $\bar J_k=\int_0^1J_D(x_0+t(z_k-x_0))\,dt$, so that $D(z_k)-D(x_0)=\bar J_k(z_k-x_0)$ and $z_k-x_0=M_ke_k+w_k$.

**Assumption J (tangent structure along the path).** $\bar J_k=\Pi_k+E_k$, with $\Pi_k$ an orthogonal projector of rank $\le d$ and $\|E_k\|\le\eta$.
- $\eta$ collects the off-tangent leakage of the denoiser plus curvature × error radius: averaging nearby tangent projectors along the segment.

**Assumption R (restricted data contraction).** With $U_k=\mathrm{range}\,\Pi_k+\mathrm{range}\,\Pi_{k-1}$ ($\dim\le2d$):
$$\|\Pi_{U_k}(I-\alpha P_kH)\Pi_{U_k}\|\le\delta,\qquad\|I-\alpha P_kH\|\le m.$$
- For $P_k=I$ and $\alpha\le2/L$: $m=1$, and $\delta$ is a RIP constant of $\sqrt\alpha A$ on unions of two tangent spaces.
- For dynamic $P_k$, Lemma A bounds $m\le\sqrt{\kappa(P_k)}\max(1,\alpha L\lambda_{\max}(P_k)-1)$ ($PH$ is self-adjoint in the $P^{-1}$ inner product).

**Theorem B (restricted contraction, recovery of the truth).** Under J and R, with $W=\sup_k\|\Pi_kw_k\|$ and $W'=\sup_k\|w_k\|$:
$$\|e_{k+1}\|\le c_1\|e_k\|+c_2\|e_{k-1}\|+F,\qquad c_1=\delta+\eta m,\quad c_2=\eta m(\delta+m),\quad F=W+(1+\delta+m)(\eta W'+\varepsilon_0).$$
If $c_1+c_2=\delta+\eta m(1+\delta+m)<1$, the error contracts linearly with rate $r=\tfrac12\big(c_1+\sqrt{c_1^2+4c_2}\big)<1$, and
$$\limsup_k\|x_k-x_0\|\le\frac{W+(1+\delta+m)(\eta W'+\varepsilon_0)}{1-\delta-\eta m(1+\delta+m)}.$$

*Proof.*
1. $e_k=\Pi_{k-1}(M_{k-1}e_{k-1}+w_{k-1})+[E_{k-1}(M_{k-1}e_{k-1}+w_{k-1})+D(x_0)-x_0]=:a_k+b_k$, with $a_k\in\mathrm{range}\,\Pi_{k-1}$ and $\|b_k\|\le\eta(m\|e_{k-1}\|+W')+\varepsilon_0$.
2. Since $\mathrm{range}\,\Pi_k\subset U_k\ni a_k$: $\|\Pi_kM_ka_k\|=\|\Pi_k\Pi_{U_k}M_k\Pi_{U_k}a_k\|\le\delta\|a_k\|\le\delta(\|e_k\|+\|b_k\|)$.
3. Then
$$\|e_{k+1}\|\le\|\Pi_kM_ka_k\|+\|\Pi_kM_kb_k\|+\|\Pi_kw_k\|+\|E_k(M_ke_k+w_k)\|+\varepsilon_0\le(\delta+\eta m)\|e_k\|+(\delta+m)\|b_k\|+W+\eta W'+\varepsilon_0.$$
   Insert the bound on $\|b_k\|$.
4. A nonnegative two-term recursion converges iff $c_1+c_2<1$; its dominant root is $r$. ∎

**Reading.**
- Plain PnP-ISTA ($m=1$) with an ideal tangent-projector denoiser ($\eta=0$) needs only $\delta<1$, a restricted (RIP-type) condition on unions of $2d$-dimensional tangent spaces. That is attainable with undersampled Fourier data, unlike $q<1$.
- The radius is then $\frac{W+(2+\delta)\varepsilon_0}{1-\delta}$.
- For white noise of level $\sigma$, $W\approx\alpha\sigma\sqrt{d\,(1+\delta)}$: noise folds in only through the $d$ tangent dimensions, as in classical CS.
- With leakage: $\delta+\eta(2+\delta)<1$, i.e. roughly $\eta<\frac{1-\delta}{3}$.
- No fixed point $x_*$, no global Lipschitz constant, and dynamic preconditioners are allowed.
- Theorem B is the bridge to the compressed-sensing question (separate note): the "moving sparsifying basis" is the family of tangent spaces $\{\mathrm{range}\,\Pi_k\}$. Assumption R needs RIP *uniformly over that family*, a union-of-subspaces / manifold RIP. That holds w.h.p. once the number of (variable-density) Fourier samples exceeds roughly coherence² × $2d$ × log(covering number of the tangent family)/$\delta^2$.

## 5. Why tight coefficient bounds slow P²nP down, and why they are unnecessary

**Proposition C (restricted spectral bound of ZMSHR1).** Suppose the secant step $s_k=x_k-x_{k-1}=e_k-e_{k-1}$ lies in a subspace $U$ with $(1-\delta_U)\|w\|^2\le\|Aw\|^2/L\le\|w\|^2$ for $w\in U$ (normalise $L=1$). If $\theta_1\le1-\delta_U$ and $\|Hs\|^2/\langle s,Hs\rangle\le\theta_2$ (always true for $\theta_2\ge L/(1-\delta_U)$), then:
- the a-search accepts $a=0$ (pure secant $v=Hs$);
- $\rho_k=\|s\|^2/\langle s,Hs\rangle\le\frac1{1-\delta_U}$;
- by Lemma A, $P_k\preceq\frac{2}{1-\delta_U}I$, **independent of $\theta_1$ and $\delta$**.
- With leakage $s=s_U+s_\perp$, $\|s_\perp\|\le\xi\|s\|$: $\rho_k\le\big(\sqrt{1-\delta_U}-\xi\big)^{-2}$ as long as positive.

**Consequences.**
- In the regime where P²nP actually converges (Theorem B), iterate differences live in tangent unions. There the preconditioner is automatically well bounded.
- The large eigenvalues allowed by a tiny $\theta_1$ ($2\cdot10^{-6}$) only arise for secant steps with large null-space content. Those are exactly the directions the denoiser's Jacobian removes ($\eta$), and the gradient has no component there.
- Forcing a small *worst-case* bound means raising $\theta_1$ (or $\delta$ above $\tfrac12\sqrt{\theta_1/\theta_2}$). That makes the a-search mix $s$ into $v$ ($a>0$) or drops the rank-one term, so $P_k\to\rho I$ and the method loses its curvature adaptation. Hence the slower practice the paper reports.
- **Design target.** The quantity to make small is the restricted contraction $\delta(P)=\|\Pi_U(I-\alpha PH)\Pi_U\|$ on tangent unions, not $\rho(I-\alpha PH)$, which is $\ge1$ in CS.
  - ZMSHR1 enforces $P_kHs_k=s_k$ along $s_k\in U_k$, so it shrinks $\delta(P_k)$ along one tangent direction per iteration.
  - This is why P²nP-D outperforms its worst-case analysis.
  - Natural extension: **tangent-restricted preconditioning**, for example a few Lanczos steps on $J_D$ to get $U$, then $P=\Pi_U(\Pi_UH\Pi_U)^{-1}\Pi_U+\tau(I-\Pi_U)$, or L-SR1 memory restricted to $U$.
  - Relation to our pipeline: without density compensation the spiral $H$ spans ~$10^6$. A circulant metric equalises the *density*, which mostly fixes $\delta(P)$ on low-dimensional $U$ too. Plain ZMSHR1 fixes one direction per iteration.

## 6. Measurable quantities (for checking the assumptions on our denoiser and trajectory)

| quantity | how to estimate | cost |
|---|---|---|
| local rate $\rho(J_D(z_*)M)$ | power iteration on $w\mapsto J_D(M w)$ (JVP) | ~20 JVP + normal ops |
| tangent dimension $d$ | $\mathrm{tr}\,J_D$ by Hutchinson (= SURE divergence) | ~10 JVP |
| leakage $\eta$ | spectrum of $J_D$ beyond the top $d$ (Lanczos on symmetrised $J$) | ~2d JVP |
| restricted $\delta$ | $\|\Pi_U(I-\alpha PH)\Pi_U\|$ for $U$ = top-$d$ Lanczos spaces at two consecutive iterates | ~2d normal ops |
| mismatch $\varepsilon_0$ | $\|D(x_0)-x_0\|$ on phantom truth | 1 denoise |

## 7. Open points

- **Theorem B for the secant Jacobian.** Assumption J must hold for $\bar J_k$ averaged along segments; bound $\eta$ by Jacobian Lipschitzness × error radius.
- **Uniform RIP over the tangent family** (probability statement): metric entropy of $\{\mathrm{range}\,J_D(x)\}$ over the relevant region, and coherence with the (variable-density) Fourier atoms. This is the compressed-sensing theory for learned priors — next note, with the literature survey in `learned_prior_cs_literature.md`.
- **Dynamic preconditioner inside Theorem B.** Bound $\delta(P_k)$ and $m_k$ jointly using Lemma A + Proposition C. Likely $\delta(P_k)\le\delta(\tau_kI)$ minus a rank-one improvement along $s_k$.
- **Momentum / FISTA.** The paper observes degradation after "too many" iterations. Momentum breaks the one-step tangent structure ($e_k$ no longer in a single tangent space); a three-term recursion is needed.
