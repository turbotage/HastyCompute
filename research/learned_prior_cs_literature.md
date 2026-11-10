# Recovery theory for learned priors in compressed sensing: what is proven

*Literature survey, 2026-11-09. Theory only. Written for the HastyCompute PnP pipeline.*

**Scope.** Our iteration is $x_{k+1}=D\big(x_k-\alpha P\nabla f(x_k)\big)$ with $f(x)=\tfrac12\|Ax-y\|^2$:
- $A$ is multi-coil, non-Cartesian (spiral) Fourier sampling;
- $P$ is a preconditioner;
- $D$ is a learned normalization-equivariant CNN denoiser, sometimes mixed with a wavelet soft-threshold.

The question is which *verifiable* conditions on the pair $(D,A)$ imply that $x_0$ is recovered. Recovery can mean exactly, or within $C\|e\|+C'\cdot(\text{model mismatch})$, with some probability over the sampling pattern.

**Local copies.** Every paper cited with a directory name has its PDF and `pdftotext -layout` text under `data/hasty/external/papers_learned_cs/<dir>/` (`paper.pdf`, `paper.txt`).

---

## 0. Legend

**Verification markers:**
- **[V]**: the statement was read in the paper's full text during this survey, and the theorem number is given. Most papers were read by parallel reading passes that quoted the theorem text.
- **[V, rechecked]**: I also re-read the theorem line by line while writing this file. This covers Berk 2022, Liu 2021, Traonmilin 2024, Joundi 2025, Leong 2025 and Wolf 2026.
- **[V~]**: read, but pdftotext garbled a formula. Check the constant against the PDF before quoting it.
- **[A]**: abstract or summary only. **[S]**: taken from a secondary source that restates the result.
- **[D]**: our own derivation or observation, not stated in the cited paper. These are claims to be checked. The proofs given are short.

**Coverage line.** Each entry ends with:

**Coverage — F** (subsampled Fourier/MRI) · **N** (noise) · **M** (model mismatch, i.e. truth not in the model set) · **Alg** (an algorithm used in practice, as opposed to an idealised estimator such as an exact minimiser over the range).

---

## 1. The target statement and the conditions in use

**Measurement model.** $y=Ax_0+e$ with $A:\mathbb C^n\to\mathbb C^{M}$, $A=[F_\Omega S_1;\dots;F_\Omega S_C]$. Here $F_\Omega$ is a NUFFT on trajectory $\Omega$ and $S_c$ are the coil maps. Let $r=\operatorname{rank}A\le\min(n,\,C\,|\Omega|)$.

With 32 coils (or about 12–20 after SV compression) and moderate undersampling, $A$ may be **injective but ill-conditioned**: the null space is trivial and tiny singular values cause g-factor noise amplification. At high per-frame acceleration $A$ has a true null space. Both regimes matter below.

**The statement we want:**
> If $(D,A)$ satisfy X, then $\limsup_k\|x_k-x_0\|\le C_1\|e\|+C_2\,d(x_0,\Sigma_D)+C_3\,\eta_D$ with probability $\ge 1-\varepsilon$ over $\Omega$.

Here $\Sigma_D$ is the model set implied by $D$ (for example $\mathrm{Fix}(D)$), and $\eta_D$ measures how far $D$ is from an ideal projection onto it.

**Conditions that recur below** (always for a model set $\Sigma$ and its secant set $\Sigma-\Sigma$):

| name | definition | used by |
|---|---|---|
| S-REC$(S,\gamma,\delta)$ | $\Vert A(x_1-x_2)\Vert \ge\gamma\Vert x_1-x_2\Vert -\delta$ for all $x_i\in S$ | Bora; Liu (with $\delta=0$, squared) |
| REC / bi-Lipschitz / RIP on secants | $a\Vert v\Vert ^2\le\Vert Av\Vert ^2\le b\Vert v\Vert ^2$ for $v\in\Sigma-\Sigma$ | Raj; Blumensath; Golbabaee–Davies; Berk |
| full-norm RIC | $\delta_\Sigma(B)=\sup_{v\in\Sigma-\Sigma}\Vert (I-B)v\Vert /\Vert v\Vert $ with $B=\mu LA$ | Traonmilin; Joundi; Leong |
| restricted $\beta$-Lipschitz map | $\Vert \Pi(z)-x\Vert \le\beta\Vert z-x\Vert $ for all $z$ and $x\in\Sigma$ | Traonmilin; Joundi; Leong |
| coherence | $\sup_{x\in\Delta(T)\cap S^{n-1}}\Vert Ux\Vert _\infty\le\alpha$; local $\alpha_j=\sup\vert \langle u_j,x\rangle\vert $ | Berk; Plan et al. |
| variation / Christoffel | $\Phi(V;\mathcal A)=\operatorname{ess\,sup}\sup_{v\in V}\Vert A v\Vert ^2$; $K(F)(\theta)=\sup_{f\in F}\Vert L(\theta)f\Vert ^2/\Vert f\Vert ^2$ | Adcock–Cardenas–Dexter; CS4ML |
| rNSP (robust null-space property) | $\Vert P_Ix\Vert \le\frac{\rho}{\sqrt\xi}\Vert P_{I^c}x\Vert _{\ell^1_w}+\gamma\Vert Ax\Vert $ | Colbrook et al.; Gottschling et al. |

---

## 2. The results that matter most

| # | Result | Sampling allowed | Model set / method | Conclusion | Marker |
|---|---|---|---|---|---|
| 1 | Bora et al. 2017, Lemma 4.3 | any $A$; deterministic given S-REC | range of $G$; idealised minimiser | $\Vert \hat x-x_0\Vert \le(\frac4\gamma+1)d(x_0,S)+\frac1\gamma(2\Vert e\Vert +\epsilon+\delta)$ | [V] |
| 2 | Berk et al. 2022, Thm II.1 | subsampled unitary (uniform Bernoulli) | range of a ReLU generator | $m\gtrsim\alpha^2 n\,k d\log$ gives uniform recovery; $\alpha\ge\sqrt{k/n}$ forces $m\gtrsim k^2 d$ | [V, rechecked] |
| 3 | Berk et al. 2023, Thm 2.1; Adcock–Cardenas–Dexter, Cor 6.2–6.7 | variable density $p_j\propto\alpha_j^2$; blocks; preconditioned least squares | ReLU / Lipschitz-generator range | $m\gtrsim\Vert \alpha\Vert _2^2\,kd\log(n/k)$ | [V] |
| 4 | Liu et al. 2021, Thm 1–2 | any $A$ with S-REC on $\mathrm{Im}(D)$ | PnP-PGM | linear convergence to a ball around $x_0$. **[D]: vacuous for $m<n$ when $\mathrm{Lip}(I-D)<1$; Lemma 6 is incorrect as stated** | [V, rechecked] |
| 5 | Joundi–Traonmilin–Aujol 2025, Thm 3.1 | any $A$, any back-projection $L$ (e.g. $PA^H$) | approximate projection $\Pi+R$ | explicit stability, robustness and projection constants if $\delta\beta<1$ | [V, rechecked] |
| 6 | Leong–Traonmilin 2025, Thm 1–2 | any $A$ with RIC on $\Sigma-\Sigma$ | exact MMSE denoisers with $\sigma_n\to0$ | exact recovery (noiseless) | [V, rechecked] |
| 7 | Wolf–Fadili–Guo–He 2026, Thm 3.2 / 3.11 | any $A$; "physics-aware" $B$ | empirical-MMSE or NN denoiser | linear convergence up to noise, approximation and memorisation terms; needs conditioning of $BA$ on a cone | [V, rechecked] |
| 8 | Blumensath 2011, Thm 2; Golbabaee–Davies 2018, Thm 1 | any $A$, bi-Lipschitz on secants | (inexact) projected gradient | explicit noise and mismatch terms | [V] |
| 9 | Hegde–Indyk–Schmidt 2015 | model-RIP | approximate head and tail projections | tail-only approximation can fail; head+tail works | [V] |
| 10 | Wei–Cai–Chan–Leung 2016, Thm 2.1 | RIP on tangent spaces | Riemannian gradient on low-rank matrices | the only "moving basis" proof found: tangent RIP + curvature | [V] |
| 11 | Jalal et al. 2021 (MRI), Thm 3.4 | any forward operator | exact posterior sampling | within factor 2 of *any* algorithm | [V] |
| 12 | Gottschling et al. 2025, Thm 4.1 / 4.5; kersize Prop 4.1 | any $A$ | any reconstruction map | accuracy–stability trade-off; stable only if injective on the model set | [V] |
| 13 | Colbrook–Antun–Hansen 2022, Thm 4 | Cartesian multilevel Fourier + Haar | unrolled network (FIRENET) | uniform, finite-noise truth bound | [V] |
| 14 | Kadkhodaie et al. 2024; Manor–Michaeli 2024 | — | MMSE denoiser | $\nabla D^\ast =\mathrm{Cov}[x\mid y]/\sigma^2$ and $\mathrm{MSE}^\ast =\sigma^2\,\mathbb E\,\mathrm{tr}\,J$ | [V] |

---

## 3. CS with generative priors (Gaussian-type $A$)

### 3.1 Bora, Jalal, Price, Dimakis 2017, "Compressed sensing using generative models" [V] (`bora2017`)
https://arxiv.org/abs/1703.03208

**Setting.** $y=Ax_0+\eta$ with $x_0$ and $\eta$ arbitrary. The model set is $S=G(B^k(r))$ or $G(\mathbb R^k)$. The estimator is any $\hat x\in S$ with $\|y-A\hat x\|\le\min_{x\in S}\|y-Ax\|+\epsilon$.

**Def. 1 (S-REC$(S,\gamma,\delta)$).** $\|A(x_1-x_2)\|\ge\gamma\|x_1-x_2\|-\delta$ for all $x_1,x_2\in S$.

**Lemma 4.1.** Let $G$ be $L$-Lipschitz and $A_{ij}\sim N(0,1/m)$. If $m=\Omega\big(\frac{k}{\alpha^2}\log\frac{Lr}{\delta}\big)$, then S-REC$(G(B^k(r)),1-\alpha,\delta)$ holds with probability $1-e^{-\Omega(\alpha^2m)}$.

**Lemma 4.2.** For a $d$-layer net with $\le c$ nodes per layer and 2-piece piecewise-linear activations, $m=\Omega(\alpha^{-2}kd\log c)$ gives S-REC$(G(\mathbb R^k),1-\alpha,0)$.

**Lemma 4.3 (deterministic).** Assume (i) S-REC holds, (ii) $\|Ax\|\le2\|x\|$ for the single fixed vector $\bar x-x_0$, and (iii) $\epsilon$-optimality. Then
$$\|\hat x-x_0\|\le(\tfrac4\gamma+1)\min_{x\in S}\|x_0-x\|+\tfrac1\gamma(2\|\eta\|+\epsilon+\delta).$$

**Thm 1.1 (ReLU, $m=O(kd\log n)$).** $\|G(\hat z)-x_0\|\le 6\min_z\|G(z)-x_0\|+3\|\eta\|+2\epsilon$, with probability $1-e^{-\Omega(m)}$.

**Thm 1.2 (Lipschitz).** $m=O(k\log\frac{Lr}\delta)$, same bound $+2\delta$.

The guarantee is non-uniform: $x_0$ is fixed and the probability is over $A$. The proofs need only the distributional JL property.

**Coverage — F:** no (only Lemma 4.3 is $A$-agnostic). **N:** yes. **M:** yes. **Alg:** idealised.

**For us.** Lemma 4.3 transfers verbatim to any $A$. Lemmas 4.1/4.2 do not, because spiral sampling has no JL property.

### 3.2 Hand & Voroninski 2017/2018, "Global guarantees for enforcing deep generative priors by empirical risk" [V] (`hand2017`)
https://arxiv.org/abs/1705.07576

**Setting.** $G(x)=\mathrm{relu}(W_d\cdots\mathrm{relu}(W_1x))$ with no biases; noiseless; objective $\tfrac12\|AG(x)-AG(x_0)\|^2$.

**Def. 2 (WDC).** $\big\|\sum_i 1_{w_i\cdot x>0}1_{w_i\cdot y>0}w_iw_i^T-Q_{x,y}\big\|\le\epsilon$, where $Q_{x,y}=\frac{\pi-\theta_0}{2\pi}I+\frac{\sin\theta_0}{2\pi}M_{\hat x\leftrightarrow\hat y}$.

**Def. 3 (RRIC).** An isometry condition on pairs of secants of $\mathrm{range}(G)$, up to $\epsilon$.

**Thm 4 (deterministic).** If $K_1d^8\epsilon^{1/4}\le1$, the WDC holds for all layers and the RRIC holds, then a descent direction exists everywhere outside small neighbourhoods of $x_0$ and of $-\rho_dx_0$.

**Prop. 6.** Gaussian weights, layers expansive with $n_i\ge cn_{i-1}\log n_{i-1}$, and $m>cdk\log\prod n_i$ imply WDC + RRIC with high probability.

**Coverage — F:** no. **N:** no. **M:** no. **Alg:** landscape only.

**For us.** The proof needs random expansive weights. It does not transfer to trained CNNs.

### 3.3 Huang, Hand, Heckel, Voroninski 2018, "A provably convergent scheme for compressive sensing under random generative priors" [V] (`huang2018`)
https://arxiv.org/abs/1812.04176

**Thm 3.1.** Subgradient descent in latent space with a sign-flip step. Assume WDC/RRIC with $\epsilon\le K_1/d^{90}$ and $\|e\|\le K_2\|x_*\|d^{-42}2^{-d/2}$. Then after $N\le K_4f(x_0)2^d/(d^4\epsilon\|x_*\|)$ steps, convergence is linear with rate $C=1-\frac{\nu}{2^d}\frac78$ to an $O(2^{d/2}\|e\|)$ ball.

**Coverage — F:** no. **N:** small noise. **M:** no. **Alg:** yes.

**For us.** Same random-weight limitation as 3.2.

### 3.4 Shah & Hegde 2018, "Solving linear inverse problems using GAN priors: an algorithm with provable guarantees" [V] (`shah2018`)
https://arxiv.org/abs/1802.08406

**Setting.** PGD with an **exact** projection $P_G(w)=G(\arg\min_z\|w-G(z)\|)$; noiseless; $x^\ast \in\mathrm{range}(G)$.

**Assumption.** Squared S-REC $\|A(x_1-x_2)\|^2\ge\gamma\|x_1-x_2\|^2-\delta$, and $\|Av\|\le\rho\|v\|$ for all $v$, with $\rho^2\le\gamma$.

**Thm 2.2.** Convergence to $x^\ast $.

**What the proof actually gives.** The residual recursion $\psi_{t+1}\le(\frac1{\eta\gamma}-1)\psi_t+\frac{\delta}{\eta\gamma}$ with the $\delta$ term dropped, so convergence is only to an $O(\delta)$ level.

**Coverage — F:** no. **N:** no. **M:** no. **Alg:** structure of PGD, but with an exact projection.

**For us.** The reading pass noted that $\rho^2\le\gamma$, with $\rho$ a *global* operator-norm bound, is essentially unsatisfiable for $m<n$. Raj et al. (3.5) repair this.

### 3.5 Raj, Li, Bresler 2019, "GAN-based projector for faster recovery with convergence guarantees in linear inverse problems" (ICCV) [V] (`raj2019`)
https://arxiv.org/abs/1902.09698

**Setting.** $x_{t+1}=G(G^\dagger(x_t-\eta A^T(Ax_t-y)))$, where $G^\dagger$ is a learned encoder.

**Def. 1 (REC$(S,\alpha,\beta)$).** $\alpha\|v\|^2\le\|Av\|^2\le\beta\|v\|^2$ on secants of $S$.

**Def. 2 ($\delta$-approximate projector, for all $x$).** $\|x-G(G^\dagger(x))\|^2\le\min_z\|x-G(z)\|^2+\delta$.

**Thm 1.** If $\beta/\alpha<2$ and $\eta=1/\beta$, then
$$f(x_n)\le(\tfrac\beta\alpha-1)^nf(x_0)+\frac{\beta\delta}{2-\beta/\alpha},\qquad\|x^\ast -x_\infty\|^2\le\frac{\delta}{2\alpha/\beta-1}.$$

**Coverage — F:** no (deterministic in $A$). **N:** no. **M:** no. **Alg:** yes, with a learned projector.

**For us.** This is the cleanest "learned approximate projector" template. Two conditions are unproven for spiral sampling: REC with ratio $<2$, and $\delta$-projection over *all* of $\mathbb R^n$.

### 3.6 Peng, Jalali, Yuan 2020 (auto-encoders) and Beygi, Jalali, Maleki, Mitra 2017 (compression-based PGD) [V] (`peng2020`, `beygi2017`)
https://arxiv.org/abs/1901.05045 · https://arxiv.org/abs/1704.01992

**Peng Thm 1.** An idealised estimator over an $L$-Lipschitz decoder with distortion $\delta$, Gaussian $A$, Gaussian noise. The error is $\lesssim\sqrt{L\sigma}(k/m)^{1/4}\delta^{\cdots}+\sigma\delta^{-2/\eta}\sqrt{k\ln(1/\delta)/m}$.

**Peng Thm 2.** PGD with an *exact* projection gives per-step contraction $0.9+\eta$ up to $\gamma_1\delta$ and noise terms.

**Beygi Thm 2.** Codebook $|C_r|\le2^r$, $m\ge80r(1+\epsilon)$. Then
$$\|x^{k+1}-\tilde x\|\le0.9\|x^k-\tilde x\|+2(2+\sqrt{n/m})^2\delta+\frac{\sigma_z}{\sigma_a}\sqrt{8(1+\epsilon)r/m}.$$

**Beygi Thm 3 (inexact projector, $\sup_x\|g(f(x))-P_C(x)\|\le\xi$).** The same bound with an additive $\xi/\sqrt n$ in normalised form. The authors leave Fourier theory explicitly open.

**Coverage — F:** no. **N:** yes. **M:** yes. **Alg:** yes. Beygi Thm 3 even covers an inexact projector.

**For us.** A uniform additive inexactness $\xi$ is the cleanest model of "$D$ is almost a projection". See §9.8 for why this is problematic for scale-equivariant $D$.

### 3.7 Lower bounds [V] (`kamath2019`, `liu2019`)
- **Kamath, Karmalkar, Price 2020, Thm 2.2** (https://arxiv.org/abs/1912.02938). For every $L,r,\delta$ with $\delta\le Lr/4$ there is an $L$-Lipschitz $G$ such that *any* (adaptive) $A$ and algorithm achieving Bora's guarantee with probability $3/4$ needs $m=\Omega(\min(k\log(Lr/\delta),n))$. Thm 2.3: a 2-layer net contains all $k$-sparse vectors in its range.
- **Liu & Scarlett 2020, Thms 3–4, 7** (https://arxiv.org/abs/1908.10744). Lower bounds $m=\Omega(k\log\frac{Lr}{\sqrt\alpha})$ for any $A$ with $\|A\|_F^2=C_An$ and noise $N(0,\frac\alpha m I)$. ReLU-depth versions also given.

**Coverage — F:** yes, in the sense that the lower bounds hold for every $A$. Positive guarantees: none.

### 3.8 Jalal, Karmalkar, Dimakis, Price 2021, "Instance-optimal compressed sensing via posterior sampling" (ICML) [V] (`jalal2021`)
https://arxiv.org/abs/2106.11438

**Def. 1.2.** $\mathrm{Cov}_{\eta,\delta}(R)=\min\{k:R(\cup_{i\le k}B(x_i,\eta))\ge1-\delta\}$.

**Thm 3.4 (upper bound, mismatch).** Gaussian $A_{ij}\sim N(0,1/m)$ and noise $N(0,\frac{\sigma^2}mI)$, with $W_p(R,P)\le\varepsilon$, $\sigma\gtrsim\varepsilon/\delta^{1/p}$ and $m\ge O(\min(\log\mathrm{Cov}_{\eta,\delta}(R),\log\mathrm{Cov}_{\eta,\delta}(P)))$. Then posterior sampling with respect to $P$ gives
$$\Pr[\|x^\ast -\hat x\|\ge c\eta+c\sigma]\le2\delta+2e^{-\Omega(m)}.$$

**Thm 4.1 (lower bound, any $A$).**
$$m\ge\frac{0.15}{\log(1+mr^2\|A\|_\infty^2/\sigma^2)}\big(\log\mathrm{Cov}_{3\eta,4\delta}(R)+\log6\delta-O(1)\big).$$
In the proof, $\|A\|_\infty$ acts as the maximum row norm.

**Coverage — F:** only the lower bound. **N:** yes. **M:** yes (Gaussian $A$ only). **Alg:** idealised (exact posterior sampling).

### 3.9 Heckel & Soltanolkotabi 2020, "Compressive sensing with un-trained neural networks: gradient descent finds the smoothest approximation" (ICML) [V] (`heckelsolt2020`)
https://arxiv.org/abs/2005.03991

**Thm 2–3.** Gaussian $A$ only; two-layer convolutional generator, overparameterised, gradient descent from random initialisation, noiseless. The result is
$$\|G(C_\infty)-x^\ast \|^2\le C\Big(\sum_i\sigma_i^{-2}\langle w_i,x^\ast \rangle^2\Big)\Big(\sum_{i>2m/3}\sigma_i^2\Big)+\xi^2\|x^\ast \|^2,$$
where $(w_i,\sigma_i)$ is the generator Jacobian's (trigonometric) singular system.

**Coverage — F:** experiments only. **N:** no. **M:** approximately. **Alg:** yes.

**For us.** This is an example of "recovery is governed by the Jacobian spectrum of the network at hand", for an untrained network.

---

## 4. Nonlinear / generative model classes with subsampled Fourier

### 4.1 Berk, Brugiapaglia, Joshi, Plan, Scott, Yılmaz 2022, "A coherence parameter characterizing generative compressed sensing with Fourier measurements" [V, rechecked] (`berk2022_coherence`)
https://arxiv.org/abs/2207.09340

**Def. I.1 (subsampled isometry).** $\theta_i\stackrel{iid}\sim\mathrm{Ber}(m/n)$. The rows of $A$ are $\sqrt{n/m}\,U_{j}$ for the selected $j$, with $U$ unitary (e.g. the DFT). Then $\mathbb E A^\ast A=I$.

**Def. I.2.** A $(k,d,n)$-generative network $G(z)=W^{(d)}\sigma(\cdots\sigma(W^{(1)}z))$ with ReLU, no biases, and $k\le k_i$. Biased networks are handled by augmenting to $k+1$.

**Defs. I.3–I.4, II.1.**
- $\|x\|_U:=\|Ux\|_\infty$.
- $T$ is $\alpha$-coherent if $\sup_{x\in T\cap S^{n-1}}\|x\|_U\le\alpha$.
- $\Delta(C)=\bigcup_i\mathrm{span}(C_i)$ for a union of convex cones $C=\bigcup C_i$.

**Thm II.1.** Let $\mathcal G=R(G)-R(G)$. If $\Delta(\mathcal G)$ is $\alpha$-coherent and
$$m\gtrsim\alpha^2n\Big(2k\sum_{i=1}^{d-1}\log\frac{2ek_i}{k}+\log\frac{4k}{\varepsilon}\Big),$$
then with probability $\ge1-\varepsilon$, **uniformly** over all $x_0\in\mathbb R^n$ and $\eta$, every $\hat x$ with $\|A\hat x-b\|\le\min_{x\in R(G)}\|Ax-b\|+\hat\varepsilon$ satisfies
$$\|\hat x-x_0\|\le\|x^\perp\|+3\|Ax^\perp\|+3\|\eta\|+\tfrac32\hat\varepsilon,\qquad x^\perp=x_0-\Pi_{R(G)}x_0 .$$
In general $\|Ax^\perp\|\le\sqrt{n/m}\|x^\perp\|$ (Remark II.2).

**Lemma II.4.** For a $k$-dimensional, $\alpha$-coherent subspace $L$:
$$\Pr\big[\sup_{x\in L\cap S^{n-1}}|\|Ax\|^2-1|\ge\delta\big]\le2k\exp\big(-C\delta^2m/(\alpha^2n)\big).$$

**Prop. III.1 (bottleneck).** Every $k$-dimensional subspace has coherence $\ge\sqrt{k/n}$, and this is tight. So $m\gtrsim k\cdot kd\log(\cdot)$: quadratic in $k$.

**Thm III.2.** With an i.i.d. Gaussian last layer (other layers arbitrary), with probability $\ge1-2e^{-\gamma^2}$:
$$\alpha\lesssim\sqrt{k/n}+\sqrt{\log n/n}+\Big(\tfrac kn\sum_i\log\tfrac{2ek_i}{k}\Big)^{1/2}+\gamma/\sqrt n.$$
The sum's placement inside the square root is from a garbled extraction [V~].

**Numerics (not theory).** A coherence proxy computed from the QR of the last layer, and a regulariser that trains low-coherence generators so that fewer measurements suffice.

**Coverage — F:** yes, uniform Cartesian on-grid sampling, real $x$. **N:** yes (deterministic $\eta$). **M:** yes. **Alg:** idealised minimiser over the range.

**For us.** This is the first Fourier theorem for a nonlinear learned model class. The relevant quantity is the alignment between the model's *local subspaces* $\Delta(\mathcal G)$ and the Fourier rows. It requires a finite union of cones, i.e. an explicit ReLU range.

### 4.2 Berk, Brugiapaglia, Plan, Scott, Sheng, Yılmaz 2023, "Model-adapted Fourier sampling for generative compressed sensing" [V] (`berk2023_adapted`)
https://arxiv.org/abs/2310.04984

**Setting.**
- $b=\frac1{\sqrt m}SFx_0+\eta$, where $S$ samples rows i.i.d. *with replacement*, $\Pr(s_i=e_j)=p_j$.
- Local coherence $\alpha_j=\sup_{x\in\Delta(T)\cap S^{n-1}}|\langle f_j,x\rangle|$, with $T=R(G)-R(G)$.
- The estimator minimises a **preconditioned** fit $\|\frac1{\sqrt m}SDF\hat x-\tilde Db\|$, with $D=\mathrm{diag}(1/\sqrt{p_j})$.

**Thm 2.1.** With $p_j=\alpha_j^2/\|\alpha\|_2^2$ and $m\ge C\|\alpha\|_2^2(kd\log\frac nk+\log\frac2\varepsilon)$, with probability $\ge1-\varepsilon$, uniformly:
$$\|\hat x-x_0\|\le\|x^\perp\|+\tfrac3{\sqrt m}\|SDFx^\perp\|+3\|\tilde D\eta\|+\tfrac32\hat\varepsilon .$$
For a general $p$, $\|\alpha\|_2^2$ is replaced by $\mu_T(F,p)^2=\max_j\alpha_j^2/p_j$ (Thm A3.1), and $p\propto\alpha^2$ is optimal (Lemma A4.4).

The authors note that their numerics drop the preconditioner $\tilde D$ and use channel-wise block sampling. Both departures are outside the theory.

**Coverage — F:** yes, variable density, Cartesian, with replacement. **N:** yes, but weighted by $\tilde D$, so low-probability samples amplify noise. **M:** yes. **Alg:** idealised.

**For us.** $\alpha_j$ can be estimated empirically from secants (differences of model outputs). That makes it a *measurable* sampling-design quantity. The theory needs the density-weighted fit, which is in tension with our no-DCW rule; see §14, gap 6.

### 4.3 Plan, Scott, Sheng, Yılmaz 2025, "Denoising guarantees for optimized sampling schemes in compressed sensing"; Plan, Scott, Yılmaz 2026, "Partially deterministic sampling for CS with denoising guarantees" [V] (`berk2025_denoising`, `partdet_2026`)
https://arxiv.org/abs/2504.01046 · https://arxiv.org/abs/2604.04802

**2025, Thm 2.7.** $Q-Q\subseteq$ a union of $M$ subspaces of dimension $\le\ell$; same optimised density and preconditioner; Gaussian noise $\eta=\sigma g/\sqrt m$. If $m\gtrsim\|\alpha\|_2^2(\log\ell+\log M+\log\frac1\delta)$, then with probability $\ge1-\delta$, uniformly:
$$\|\hat x-x_0\|\le9\frac{\sigma}{\sqrt m}\|\alpha\|_2\min\Big(\sqrt{\tfrac5{4\delta}},\tfrac{1}{\sqrt n\min\alpha}\sqrt{\ell+\log M+\log\tfrac{20}\delta}\Big)+\|x^\perp\|+6\|SDFx^\perp\|+\tfrac32\sqrt\varepsilon .$$
The noise term decays like $1/\sqrt m$. ReLU generators are covered (§3 of the paper).

**2026, Thm 2.8.** Bernoulli selectors with weights $w_j=\min(m\alpha_j^2/L^2(\alpha,m),1)$. Saturated rows are sampled *deterministically*, which gives a principled version of "fully sample the k-space centre". The constant $\|\alpha\|_2^2$ improves to $L^2(\alpha,m)\le\|\alpha\|_2^2$.

**Coverage — F:** yes, Cartesian. **N:** Gaussian, with a $1/\sqrt m$ rate. **M:** yes. **Alg:** idealised.

### 4.4 Adcock, Cardenas, Dexter 2023/2026, "A unified framework for learning with nonlinear model classes from arbitrary linear samples" [V] (`adcock2024_unified`)
https://arxiv.org/abs/2311.14886. The text read is **v2 (Mar 2026)**, which extends the ICML 2024 version (that version had union-of-subspaces models only).

**Setting.**
- $b_i=A_i(x)+e_i$ with independent random operators $A_i\sim\mathcal A_i$ into Hilbert spaces $Y_i$. Measurements may be vector-valued, and the distributions may differ per $i$.
- Nondegeneracy: $\alpha\|x\|^2\le\frac1m\sum\mathbb E\|A_ix\|^2\le\beta\|x\|^2$.
- Model class: any $U$ inside a finite-dimensional space.
- Estimator: a $\gamma$-minimiser of empirical least squares over $U$, then truncated.

**Def. 3.3 (variation).** $\Phi(V;\mathcal A)$ is the smallest $\Phi$ with $\|A v\|^2\le\Phi$ for all $v\in V$, almost surely.
- Def. 3.9: only *nonconstant* (random) samples count, so deterministically sampled k-space costs nothing.
- Special cases: coherence, leverage scores, Christoffel functions.

**Thm 4.1 / Cor. 4.3 (general, local).** The sample condition is an entropy integral of $U-x^\ast $, with an arbitrary anchor $x^\ast $. The error is
$$\mathbb E\|x-\check x\|^2\lesssim\tfrac\beta\alpha\big(\|x-x^\ast \|^2+\inf_U\|x-u\|^2\big)+\theta^2\epsilon+\gamma^2/\alpha+\|e\|^2/\alpha .$$
The bound is **in expectation** for a fixed $x$. The supporting nondegeneracy (RIP-like) event holds with probability $\ge1-\epsilon$.

**Cor. 4.6.** If $\Delta U\subseteq V_1\cup\dots\cup V_d$ with $\dim V_i\le n$, then e.g. $m\gtrsim\alpha^{-1}\Phi(S(\Delta^2U))[n+\log(2d/\epsilon)]$ suffices.

**Thm 6.1 (Lipschitz, positively homogeneous generators).** $F:\mathbb R^k\to\mathbb R^N$, $L$-Lipschitz with $F(tz)=tF(z)$. Then
$$m\gtrsim\alpha^{-1}\Phi\,k\,[\log(1+L\eta/(\sqrt\alpha\chi\iota_x))+\log\tfrac2\epsilon]$$
gives error $\lesssim\frac\beta\alpha\chi^2\|x\|^2+\dots$, where $\chi$ is the relative distance of $x$ to the model.

**Cor. 6.2–6.7 (subsampled unitary).**
- Density $\propto$ squared local coherence: $m\gtrsim\|\sigma\|_1k\log(\cdot)$.
- Bernoulli sampling works too.
- **Block sampling (Cor. 6.7)** with block local coherence $\sigma_i=\sup_{\|v\|=1}\sum_{j\in P_i}|u_j^\ast v|^2/r_j$, overlaps handled by multiplicities $r_j$.

**Multi-coil.** Ex. 2.8 notes that the multimodal setup "includes parallel MRI". No coil theorem is given.

**[D] Multi-coil specialisation (reading-pass derivation, not in the paper).** For shared trajectories, take $Y=\mathbb C^{C\cdot|P_i|}$ and $A(x)=\pi_i^{-1/2}\big((F S_cx)_{j\in P_i}\big)_c$. Then $\alpha=\min_j\sum_c|S_c(j)|^2$ and $\beta=\max_j\sum_c|S_c(j)|^2$. The variation becomes a coil-aware block coherence $\sup_v\sum_c\sum_{j\in P_i}|(FS_cv)_j|^2/\pi_i$.

**Coverage — F:** yes, on-grid, variable density, blocks. **N:** yes. **M:** yes. **Alg:** idealised.

**For us.** This is the most general host for a future theorem.
- $U=\mathrm{Fix}(D)$ is formally admissible.
- The open work is bounding the variation and the covering numbers of $\mathrm{Fix}(D)-x^\ast $.
- Thm 6.1 needs a Lipschitz, homogeneous *parametrisation* of $U$, which a denoiser does not supply.

### 4.5 Adcock, Cardenas, Dexter 2023, "CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions" (NeurIPS) [V] (`cs4ml2023`)
https://arxiv.org/abs/2306.00945

**Def. 2.6.** $K(F)(\theta)=\sup_{f\in F}\|L(\theta)f\|^2/\|f\|^2$ and $\kappa(F)=\int K(F)\,d\rho$.

**Thm 2.7.** If $F-F=\bigcup_{j\le d}F_j$ with $\dim F_j\le n$, sampling from $\mu^\ast \propto K(F-F)\,d\rho$ with $m\asymp\alpha^{-1}\kappa\log(2nd/\delta)$ gives
$$\mathbb E\|f^\ast -T_1(\hat f)\|^2\lesssim\tfrac\beta\alpha\inf_F\|f^\ast -f\|^2+\delta .$$

**Thm 4.8.** Explicit constants: $3(1+\frac{4\beta}{(1-\epsilon)\alpha})\inf\|f^\ast -f\|^2+4\sigma^2\delta+\frac{12}{(1-\epsilon)\alpha}N+\dots$.

**App. C (MRI with generative models).** Cartesian lines are treated as blocks. Radial and spiral blocks are mentioned as possible but not worked out. The authors state that the theory "does not completely apply" to their brain generator. The parallel-MRI formulation (C.7) uses *independent masks per coil*, which is not physical.

**Coverage — F:** yes, Cartesian blocks. **N:** yes. **M:** yes. **Alg:** idealised.

### 4.6 DeLise & Dexter 2026, "Active learning for conditional generative CS" [V] (`cond_gen_2026`)
https://arxiv.org/abs/2605.05435

**Setting.** Prompt-conditioned generators; weighted DFT $\frac1{\sqrt m}W_\Omega^{1/2}P_\Omega F$.

**Thm 3.6.** For an $L$-Lipschitz generator on $B_2^k(R)$, $m\gtrsim\tau^{-2}\Lambda\,k\log(\cdot)$ gives S-REC.

**Thm 3.7.** $\|\hat f-f^\ast \|\le(1+2\sqrt{\Lambda}/\gamma)\inf\|f^\ast -f\|+(2\|e\|+\omega+q)/\gamma$.

The Lipschitz and ReLU properties of the Stable Diffusion generator used in the experiments are assumed, not verified.

### 4.7 Jalal, Arvinte, Daras, Price, Dimakis, Tamir 2021, "Robust compressed sensing MRI with deep generative priors" (NeurIPS) [V] (`jalal2021_mri`)
https://arxiv.org/abs/2108.01368

**Thm 3.3 (Gaussian $A$ only).** Posterior sampling is robust to a $(\delta,\alpha)$-$W_\infty$ prior mismatch.

**Thm 3.4 (any forward operator, any metric).** If *some* algorithm achieves $d(x^\ast ,x')\le\varepsilon$ with probability $1-\delta$, then exact posterior sampling $\hat x\sim\mu(\cdot\mid y)$ achieves $d(x^\ast ,\hat x)\le2\varepsilon$ with probability $\ge1-2\delta$. The proof is a triangle inequality plus conditional independence. It requires the **exact posterior of the true prior**.

**What the authors say about Fourier.** Robustness to distribution shift under Fourier measurements "is not true for general distributions". Their example: two priors on Fourier-sparse signals that Fourier sampling cannot distinguish.

**Coverage — F:** only via Thm 3.4, a relative statement with no sample complexity. **N:** yes. **M:** Gaussian $A$ only. **Alg:** idealised; the Langevin dynamics actually run are not analysed.

**[D] Elementary.** If the prior is correct, then $\mathbb E\|\hat x_{PS}-x_0\|^2=2\,\mathrm{MMSE}$ for any $A$.

---

## 5. Classical structured-Fourier CS (variable density, multilevel, blocks)

These results are for sparse models, but they are the only rigorous treatment of **variable-density** and **trajectory/block** structure.

### 5.1 Adcock, Hansen, Poon, Roman 2017, "Breaking the coherence barrier" [V] (`adcock2017_breaking`)
https://arxiv.org/abs/1302.0561

**Definitions.** Multilevel random sampling (uniform within each level); sparsity in levels $(s_k)$; local coherences $\mu_{N,M}(k,l)$; relative sparsities $S_k$.

**Thm 4.4 (QCBP $\ell^1$).** Per-level conditions (4.4)–(4.5), with $m_k\gtrsim(N_k-N_{k-1})\log(\epsilon^{-1})\big(\sum_l\mu_{N,M}(k,l)s_l\big)\log N$. With probability $\ge1-s\epsilon$:
$$\|\xi-x\|\le C\big(\tilde\delta(1+L\sqrt s)+\sigma_{s,M}(f)\big).$$

**Thm 6.2 (Fourier–wavelets, 1D, continuous FT sampled on a grid).** Per-level counts decay with the distance between levels. In total, $m\gtrsim s\log(\epsilon^{-1})\log\tilde N$.

**Coverage — F:** yes, multilevel. **N:** yes. **M:** yes. **Alg:** convex program.

### 5.2 Krahmer & Ward 2014, "Stable and robust sampling strategies for compressive imaging" [V] (`krahmer2014_stable`)
https://arxiv.org/abs/1210.2380

**Setting.** Inverse-square density $\eta(k)\propto\min(C,1/|k|^2)$; weighted noise constraint $\|\rho\circ(F_\Omega g-y)\|\le\varepsilon\sqrt m$ with $\rho_j=\eta^{-1/2}$.

**Thm 3.1 (TV).** If $m\gtrsim s\log^3 s\log^5N$, then *uniformly* $\|f-f^\#\|\lesssim\|\nabla f-(\nabla f)_s\|_1/\sqrt s+\varepsilon$.

**Thm 5.2.** Local coherence $\kappa_j$ plus density $\nu_j=\kappa_j^2/\|\kappa\|_2^2$ gives RIP of the **preconditioned** matrix.

**Thm 6.2.** Fourier–Haar local coherence $\le\min(1,18\pi/\max(|k_1|,|k_2|))$.

**Coverage — F:** yes, Cartesian. **N:** weighted noise. **M:** yes. **Alg:** convex program.

### 5.3 Bigot, Boyer, Weiss 2016 (block sampling) and Boyer, Bigot, Weiss 2019 (structured acquisition) [V] (`bigot2016_block`, `boyer2019_structured`)
https://arxiv.org/abs/1305.4446 · https://arxiv.org/abs/1505.01619

**Setting.** $A=\frac1{\sqrt m}[B_1;\dots;B_m]$, i.i.d. random blocks with $\mathbb E B^\ast B=I$.

**Bigot et al., Thm 3.1.** Noiseless basis pursuit succeeds if $m\ge c\,\gamma(S)\log(4n)\log(12/\varepsilon)$ with $c=3\cdot534$.

**Bigot et al., Prop. 4.5 (negative).** For 2D Fourier with horizontal-line blocks, fewer than $\min(2s,\sqrt n)$ lines cannot identify all $s$-sparse vectors. Without structured sparsity, blocks cost as much as single samples.

**Boyer et al., Thm 3.3.** Structured sparsity gives $m\ge73\,\Gamma(S,\pi)\ln(64s)(\ln\frac{9n}\varepsilon+\ln\ln(64s))$. Cor. 4.10 covers Fourier∘Haar with lines.

**Coverage — F:** yes, line blocks. **N:** no. **M:** no. **Alg:** convex program.

**For us.** A spiral interleaf is a block. The penalty for correlated samples is governed by intra- and inter-block coherence, and only a structured prior removes it.

### 5.4 Chauffert, Ciuciu, Kahn, Weiss 2014, "Variable density sampling with continuous trajectories" [V] (`chauffert2014_vds`)
https://arxiv.org/abs/1311.6039

**What is proven.**
- Prop. 3.1: Markov-chain trajectories on a grid need $m\gtrsim\epsilon(P)^{-1}K^2s^2\log(\cdot)$, quadratic in $s$.
- Thm 4.2: a TSP curve's occupation measure converges to $p^{(d-1)/d}$.

**No CS guarantee exists for TSP or continuous curves.** Remark 13 calls that part "mostly heuristic".

### 5.5 Oymak, Recht, Soltanolkotabi 2018, "Sharp time–data tradeoffs for linear inverse problems" [V] (`oymak2018timedata`)
https://arxiv.org/abs/1507.04793

**Thm 1.2.** PGD onto $\{f\le f(x)\}$, possibly nonconvex, with rate $\kappa_f\rho$. Here $\rho=\sup_{u,v\in C_f(x)\cap B}u^\ast (I-\mu A^\ast A)v$ over the **descent cone at the truth**.

**Thm 2.10.** For subsampled orthonormal matrices times **random signs**, $m\gtrsim(\log n)^4m_0$ with $m_0\approx\omega(C_f(x)\cap B)^2$ gives linear convergence.

MRI has no random signs, so Thm 2.10 does not apply. Still, it is the only structured-matrix PGD rate in this survey, and the complexity is a *local* cone width at the truth.

---

## 6. PnP / RED with a bound to the TRUTH

### 6.1 Liu, Asif, Wohlberg, Kamilov 2021, "Recovery analysis for plug-and-play priors using the restricted eigenvalue condition" (NeurIPS) [V, rechecked] (`liu2021_pnp_rec`)
https://arxiv.org/abs/2106.03668

**Setting.** $y=Ax^\ast +e$ with $A\in\mathbb R^{m\times n}$. PnP-PGM: $x_k=D(x_{k-1}-\gamma\nabla g(x_{k-1}))$ with $g=\frac12\|y-Ax\|^2$.

**Assumption 1.** $R=I-D$ satisfies $\|R(x)\|\le\delta$ and $\|R(x)-R(z)\|\le\alpha\|x-z\|$ for all $x,z$.

**Assumption 2 (S-REC over $\mathrm{Im}(D)$).** $\|Ax-Az\|^2\ge\mu\|x-z\|^2$ for all $x,z\in\mathrm{Im}(D)$.

**Thm 1 (noiseless, $x^\ast \in\mathrm{Zer}(R)=\mathrm{Fix}(D)$, $x^0\in\mathrm{Im}(D)$).** $\|x^t-x^\ast \|\le c^t\|x^0-x^\ast \|$, where $c=(1+\alpha)\max\{|1-\gamma\mu|,|1-\gamma\lambda|\}$ and $\lambda=\lambda_{\max}(A^TA)$. The condition $c<1$ is achievable iff $\alpha<2\mu/(\lambda-\mu)$.

**Thm 2 (general $x^\ast $, noise).** $\|x^t-x^\ast \|\le c^t\|x^0-x^\ast \|+\varepsilon\frac{1-c^t}{1-c}$, with
$$\varepsilon=(1+c)\Big[(1+2\sqrt{\lambda/\mu})\,\|x^\ast -\mathrm{proj}_{\mathrm{Zer}(R)}(x^\ast )\|+\tfrac2{\sqrt\mu}\|e\|+\delta(1+1/\alpha)\Big]$$
(square-root placement from a garbled extraction) [V~].

**Thm 3.** If $D$ is nonexpansive and $\mathrm{Zer}(\nabla g)\cap\mathrm{Zer}(R)\ne\emptyset$, then $\mathrm{Fix}(\text{PnP})=\mathrm{Zer}(\text{RED})$.

**Empirical (Fig. 1, Fig. 6).**
- Measured $\mathrm{Lip}(R)$: max $0.415$ for the denoiser and $0.164$ for the artifact remover.
- Sampled S-REC ratios on $\mathrm{Im}(D)$: min $0.093$ for radial MRI at 10%, and $0.479$ at 50%.

**Coverage — F:** formally any $A$. **N:** yes. **M:** yes, via $\mathrm{dist}(x^\ast ,\mathrm{Fix}D)$. **Alg:** yes, exactly PnP-PGM.

**[D] Observation 1: vacuity for underdetermined $A$.**
- If $\mathrm{Lip}(R)=\alpha<1$, then $D=I-R$ is a bijection of $\mathbb R^n$: for any $z$, the map $x\mapsto z+R(x)$ is a contraction with a unique fixed point solving $D(x)=z$.
- So $\mathrm{Im}(D)=\mathbb R^n$, and Assumption 2 then requires $\lambda_{\min}(A^TA)\ge\mu>0$, which is impossible for $m<n$.
- Moreover, $c<1$ forces $\alpha<2\mu/(\lambda-\mu)$, and this is $<1$ whenever $\lambda/\mu>3$.

Consequences:
- In the strongly undersampled regime the theorem's hypotheses cannot hold simultaneously.
- In particular they cannot hold in the authors' own radial-MRI experiment, where $\alpha\approx0.16$–$0.42<1$ and $\lambda/\mu\approx10$.
- The sampled $\mu=0.093$ is an artefact of sampling $\mathrm{Im}(D)$ only near natural images.
- The theorem is non-vacuous only when $D$ is *projection-like* ($\mathrm{Lip}(I-D)\ge1$, so $D$ is non-surjective) **and** the restricted condition number $\lambda/\mu$ is below 3.

**[D] Observation 2: Lemma 6 is incorrect as stated.**
- Lemma 6 (supplement) claims that $\mu$-restricted strong convexity on a set $X$ gives $\|(I-\gamma\nabla g)(x)-(I-\gamma\nabla g)(z)\|\le\max\{|1-\gamma\mu|,|1-\gamma\lambda|\}\|x-z\|$ on $X$.
- That holds for *global* strong convexity, not for restricted strong convexity, because $A^TA$ maps secants of $X$ out of $X$.
- Counterexample: $A=[\sqrt\lambda,0]$, $X=\mathrm{span}\{(1,1)/\sqrt2\}$. Then $\mu=\lambda/2$, and at $\gamma=1/\lambda$ the claimed factor is $1/2$ but the true factor is $1/\sqrt2$.
- The correct restricted bound is $\|(I-\gamma A^TA)v\|^2\le\big(1-\gamma(2-\gamma\lambda)\mu\big)\|v\|^2$, using $\|A^TAv\|^2\le\lambda\|Av\|^2$. So $c$ should read $(1+\alpha)\sqrt{1-\gamma(2-\gamma\lambda)\mu}\ \ge(1+\alpha)\sqrt{1-\mu/\lambda}$.
- With this correction, $c<1$ needs restricted condition number $\lambda/\mu<4/3$ when $\alpha\ge1$, which is even more restrictive.

### 6.2 Traonmilin, Aujol, Guennec 2024/2026, "Towards optimal algorithms for the recovery of low-dimensional models with linear rates" [V, rechecked] (`traonmilin2024`)
https://arxiv.org/abs/2410.06607 (v4, Sep 2026)

**Def. 1.1 (full-norm RIC).** $\|(I-B)(x_1-x_2)\|\le\delta\|x_1-x_2\|$ for $x_i\in\Sigma$. Lemma 1.1: $\delta_\Sigma(A^HA)<1$ implies the usual RIP.

**Def. 2.1 (restricted $\beta$-Lipschitz).** As in §1. Facts:
- $\beta\ge1$;
- $\beta=1$ for the orthogonal projection onto a closed convex set;
- $\beta\le2$ for the orthogonal projection onto a proximinal set;
- $\beta\le\sqrt{(3+\sqrt5)/2}\approx1.618$ for hard thresholding;
- $\beta<\infty$ implies idempotence.

**Thm 2.1.** For $x_{n+1}=\Pi(x_n)-\mu A^H(A\Pi(x_n)-y)$ with $\hat x\in\Sigma$:
$$\|x_n-\hat x\|\le(\delta\beta)^n\|x_0-\hat x\|,\qquad\delta=\delta_\Sigma(\mu A^HA).$$
The proof uses only $x=\hat x$ in the $\beta$ property, so a version "restricted Lipschitz at the truth" suffices [D].

The paper reports empirical checks that general-purpose denoisers satisfy the restricted Lipschitz property, at sampled points only.

**Coverage — F:** any $A$, with no sampling theorem. **N:** no (see 6.3). **M:** no. **Alg:** yes.

### 6.3 Joundi, Traonmilin, Aujol 2025, "From sparse recovery to plug-and-play priors, understanding trade-offs for stable recovery with generalized projected gradient descent" [V, rechecked] (`aujol2025gpgd`)
https://arxiv.org/abs/2512.07397

**Setting.**
- Iteration: $x_{n+1}=\Pi'(x_n)-\mu L(A\Pi'(x_n)-y)$, where $L$ is **any** linear back-projection.
- Approximate projection: $\Pi'=\Pi_\Sigma+R$ with $\|R(x)\|\le\eta$ for all $x$.
- Correspondence with our iteration: put $u_n=\Pi'(x_n)$, the denoiser output. Then $u_{n+1}=D(u_n-\alpha PA^H(Au_n-y))$, which is **our iteration with $L=PA^H$**.

**Thm 3.1.** Let $\delta=\delta_\Sigma(\mu LA)$ and $\beta=\beta_\Sigma(\Pi_\Sigma)$, and assume $\delta\beta<1$. Then
$$\|x_n-\Pi_\Sigma\hat x\|\le(\delta\beta)^n\|x_0-\Pi_\Sigma\hat x\|+\tfrac{\mu}{1-\delta\beta}\|Le\|+\tfrac{\|\mu LA\|}{1-\delta\beta}\,d(\hat x,\Sigma)+\tfrac{\|I-\mu LA\|}{1-\delta\beta}\,\eta,$$
with $d(\hat x,\Sigma)=\|\Pi_\Sigma\hat x-\hat x\|$. Bound (11) gives the analogous statement for $\|x_n-\hat x\|$, with $C'_{rob}=1+\|\mu LA\|/(1-\delta\beta)$.

**Coverage — F:** any $A$, with no sampling theorem. **N:** yes. **M:** yes. **Alg:** yes. A preconditioner is allowed through $L$.

**For us.** This is the closest existing theorem to our iteration, including the preconditioner. The trade-off is explicit:
- a better-conditioned $\mu LA$ lowers $\delta$;
- but it raises $\|Le\|$, which is the g-factor / DCW noise penalty.

See §13 for the limits [D].

### 6.4 Leong & Traonmilin 2025, "A recovery theory for diffusion priors: deterministic analysis of the implicit prior algorithm" [V, rechecked] (`leong2025recovery`)
https://arxiv.org/abs/2509.20511

**Setting.** Noiseless $y=A\hat x$. The iteration is $x_{n+1}=P^n(x_n)-\mu A^T(AP^n(x_n)-y)$ with $P^n=D_{\sigma_n}$ the **exact MMSE denoiser** and $\sigma_n\to0$ (the deterministic Kadkhodaie–Simoncelli algorithm).

**Thm 1.** If $\Pi_\Sigma$ is restricted $\beta$-Lipschitz, $\delta\beta<1$ (full-norm RIC of $\mu A^TA$), and $u_n=\|P^n(x_n)-\Pi_\Sigma(x_n)\|\to0$, then
$$\|x_n-\hat x\|\le C\big((\delta\beta)^{n/2}+\max_{l\in[n/2,n]}\|P^l(x_l)-\Pi_\Sigma(x_l)\|\big).$$

**Prop. 1 / Cor. 3.** For the uniform prior on a compact convex set of dimension $s$: $\|P^n-\Pi^\perp_\Sigma\|_{op}=O(\sigma_n\sqrt{\log(1/\sigma_n)})$.

**Thm 2.** Low-rank Gaussian mixture on a union of subspaces: global convergence with error $O(e^{-c/\sigma_l^2}+\sigma_l^2)$.

Sample complexity is delegated to RIP results for random measurements.

**Coverage — F:** no. **N:** no. **M:** only through the projection-error term. **Alg:** yes, idealised denoiser.

### 6.5 Wolf, Fadili, Guo, He 2026, "Physics matters in PnP: recovery guarantees with the MMSE and NN denoisers" [V, rechecked] (`physmatters2026`; duplicate copy in `physicspnp2026`)
https://arxiv.org/abs/2607.29525

**Setting.** $x_{k+1}=\hat D(x_k+\gamma B(y-Ax_k))$ with a general linear $B$. $\hat D$ is the **empirical MMSE denoiser**, built from $N$ samples, for Gaussian noise $\Upsilon\sim N(0,\Sigma_\Upsilon)$. $\Sigma_\Upsilon$ may be degenerate and non-diagonal.

**Assumptions.**
- (A-2) Compact image set with $\|x\|\le M$.
- (A-4) $\mathrm{co}(\mathrm{supp}\,\mu_X)\subset V=\mathrm{Im}\,\Sigma_\Upsilon$.
- (A-5) $\mathrm{Im}\,B\subset V$.
- (A-6) $BA$ is symmetric PSD.
- (A-7) $\ker(BA)\cap T_{\mathrm{co}(\mathrm{supp}\mu_X)}(x)=\{0\}$, using the tangent cone of the convex hull.

**Thm 3.2.** Let $\kappa(BA)=\lambda_{\max}(BA)/\lambda_{\min}(BA;T)$ on that cone and $\gamma=(\kappa\lambda_{\max})^{-1}$. Assume $M^2/\lambda_{\min}(\tilde\Sigma_\Upsilon)\le1$, or $\kappa<(1-\lambda_{\min}(\tilde\Sigma_\Upsilon)^2/M^4)^{-1/2}$. Then
$$\|x_k-x\|\le q^{k+1}\|x_0-x\|+\frac{1-q^k}{1-q}\Big(\frac{M^2}{\lambda_{\min}}\gamma\|B\|\|e\|+2M(N-1)e^{-\epsilon/2\lambda_{\max}}+\sqrt{\tfrac{\lambda_{\max}}{\lambda_{\min}}}\,\mathrm{dist}(x,\mathcal D_X^N)\Big),$$
with $q=\frac{M^2}{\lambda_{\min}}\sqrt{1-\kappa^{-2}}$. The last term is a **memorisation** term, which becomes $(\log N/N)^{1/r}$ with $r=\dim V$.

**Thm 3.11 (NN denoiser with Lipschitz constant $L$).** The same structure, with $q=L\sqrt{1-\kappa(BA)^{-2}}$ and the condition $L\le1$ or $\kappa\le(1-L^{-2})^{-1/2}$.

**Remark 3.3.** The denoiser need not be nonexpansive. $B$ should make the injected noise $\gamma BE$ match the training noise $\Upsilon$ (a "physics-aware" denoiser).

**Coverage — F:** any $A$, with no sampling theorem. **N:** yes. **M:** yes (NN version). **Alg:** yes.

**[D] Caveats.**
- With $\lambda_{\min}(\tilde\Sigma_\Upsilon)=\sigma^2\ll M^2$, Thm 3.2's condition becomes $\kappa<1+\sigma^4/(2M^4)$. That is essentially vacuous for realistic images.
- For a *curved* support, the tangent cone of the convex hull at $x$ contains a half-space of $V$. (A-7) then reduces to **injectivity of $BA$ on $V$**: the null space is assumed away, not handled.
- (A-7) is non-trivial only at "extreme" points with pointed tangent cones, as with $\ell^1$ / atomic-norm descent cones.

### 6.6 Metzler, Maleki, Baraniuk 2016, "From denoising to compressed sensing" (D-AMP) [V] (`metzler2016_damp`)
https://arxiv.org/abs/1406.4175

**Defs. 1–3.** A denoiser is *proper of level $\kappa$* if $\sup_{x_o\in C}\mathbb E\|D_\sigma(x_o+\sigma\epsilon)-x_o\|^2/n\le\kappa\sigma^2$ for all $\sigma$.

**Prop. 1 (conditional on state evolution, "Finding 1", which is empirical in the paper).** With Gaussian $A$, D-AMP succeeds if $m/n>\kappa$.

**Prop. 2.** Noise sensitivity $\le\frac{\kappa\sigma_w^2+B}{1-\kappa/\delta}$.

**D-IT.** No theorem for plain D-IT, which is our iteration. The paper shows D-IT does *not* follow state evolution.

**[S] Berthier, Montanari, Nguyen.** Rigorous state evolution for Lipschitz non-separable denoisers and Gaussian $A$ (https://arxiv.org/abs/1708.03950).

**[D] Link to effective dimension.** For an MMSE denoiser, $\mathrm{MSE}=\sigma^2\,\mathbb E\,\mathrm{tr}J$ (§9.1). So $\kappa=\sup_\sigma\mathbb E\,\mathrm{tr}J_D/n$ is an *effective-dimension fraction*, and Prop. 1 reads "$m/n>$ sup effective-dimension fraction". This is the cleanest "samples $\gtrsim$ denoiser dimension" statement found, but only for Gaussian $A$ with AMP.

**Coverage — F:** no. **N:** yes. **M:** no. **Alg:** D-AMP (not D-IT).

### 6.7 Fletcher, Rangan, Sarkar, Schniter 2018, "Plug-in estimation in high-dimensional linear inverse problems: a rigorous analysis" (VAMP) [V] (`fletcher2018vamp`)
https://arxiv.org/abs/1806.10466

**Thm 1.** Assume $A=USV^T$ with **Haar-distributed $V$** (right-rotationally invariant) and a uniformly Lipschitz, "convergent" plug-in denoiser. Then the errors are asymptotically Gaussian and the MSE is exactly predicted by state evolution.

**Coverage — F:** no; deterministic subsampled Fourier is not rotationally invariant. **N:** yes. **M:** yes (any denoiser in the class). **Alg:** yes (VAMP).

### 6.8 Colbrook, Antun, Hansen 2022, "The difficulty of computing stable and accurate neural networks" (PNAS) — FIRENETs; Neyra-Nesterenko & Adcock 2023, NESTANets [V] (`colbrook2022_firenet`, `neyra2023_nestanet`)
https://arxiv.org/abs/2101.08286 · https://arxiv.org/abs/2203.00804

**Thm 3 (FIRENET).** Given $A$ with the weighted rNSP in levels, an *explicitly constructed* (unrolled, restarted primal–dual) network $\phi_n$ with $O(n)$ layers satisfies
$$\|\phi_n(y)-x\|\lesssim\delta+e^{-n}\quad\text{whenever }\sigma_{s,M}(x)_{\ell^1_w}+\|Ax-y\|\lesssim\delta .$$

**Thm 4.** Fourier/Walsh with Haar, multilevel random sampling, and $m_k\gtrsim\kappa M_F(s,k)L$. With probability $\ge1-P$, the rNSP holds with $\rho=1/2,\gamma=\sqrt2$. This gives a **uniform** truth bound $\lesssim\kappa^{1/4}\delta$ at depth $\lesssim\log(1/\delta)$.

**NESTANet, Thm 1.** The analysis-sparse analogue. It needs $AA^\ast =\nu I$, which fails for NUFFT and SENSE.

**Coverage — F:** yes, Cartesian, single coil. **N:** yes. **M:** yes ($\sigma_{s,M}$). **Alg:** yes, unrolled; not trained.

**For us.** These are the only finite-noise, uniform truth bounds for Fourier in the "network" world. They are relevant to the **wavelet soft-threshold component** of our prior.

### 6.9 Shoushtari, Liu, Kamilov 2022 / Shoushtari, Liu, Chandler, Asif, Kamilov 2023 — prior mismatch [V] (`shoushtari2022_mismatch`, `shoushtari2023_admm`)
https://arxiv.org/abs/2207.13200 · https://arxiv.org/abs/2310.00133

**2022, Thm 1 (SD-RED).** The ideal denoiser $D$ is $\lambda$-Lipschitz with $\lambda<1$; the trained one satisfies $\|\hat D-D\|\le\sigma\varepsilon$; small step size. Then
$$\|x_t-x^\ast \|\le\eta^tR_0+\tau\sigma\varepsilon\gamma/(1-\eta),$$
where $x^\ast $ is the **ideal-denoiser** solution, not the truth. The contraction of $D$ supplies the strong monotonicity that $g$ lacks.

**2023, Thm 1 (PnP-ADMM, nonconvex).** Stationarity $\min_k\|\nabla f(x^k)\|^2\le A_1/t+A_2\bar\varepsilon_t$, with $\bar\varepsilon_t$ the averaged mismatch.

**Coverage — F:** experiments (2022: subsampled Fourier). **N:** no. **M:** yes, in the denoiser. **Alg:** SD-RED / ADMM, not PnP-PGM.

---

## 7. Convergence-only and regularization-theory results (no bound to the truth)

### 7.1 Ryu, Liu, Wang, Chen, Wang, Yin 2019, "Plug-and-play methods provably converge with properly trained denoisers" (ICML) [V] (`ryu2019`)
https://arxiv.org/abs/1905.05406

**Assumption (A).** $\mathrm{Lip}(H_\sigma-I)\le\varepsilon$.

**Thm 1 (PnP-FBS).** With $f$ $\mu$-strongly convex: $\mathrm{Lip}(T)\le\max\{|1-\alpha\mu|,|1-\alpha L|\}(1+\varepsilon)$, which is $<1$ if $\varepsilon<2\mu/(L-\mu)$.

**Thm 2 / Cor. 3.** DRS and ADMM analogues.

**Real spectral normalisation.** A power iteration on the actual convolution operator.

**Quotes from the paper.** Strong convexity "does exclude … compressed sensing"; the theory "says nothing about the recovery performance".

**Coverage — F:** no ($\mu=0$). **N:** through $f$. **M:** no. **Alg:** yes.

### 7.2 Gilton, Ongie, Willett 2021, "Deep equilibrium architectures for inverse problems in imaging" [V] (`gilton2021_deq`)
https://arxiv.org/abs/2102.07944

- **Thm 1 (DE-Grad, $f_\theta=x+\eta A^T(y-Ax)-\eta R_\theta(x)$).** Contraction iff $\varepsilon<1+\mu$. Works with $\mu=0$ if $\varepsilon<1$.
- **Thms 2–3 (DE-Prox = our form; DE-ADMM).** These need $\mu>0$. The authors note this is "violated in … undersampled MRI".
- No error bound.

**Coverage — F:** DE-Grad only. **N:** no. **M:** no. **Alg:** yes.

### 7.3 Cohen, Elad, Milanfar 2021, "Regularization by denoising via fixed-point projection (RED-PRO)" [V] (`cohen2021_redpro`)
https://arxiv.org/abs/2008.00226

- **Thm 3.8.** For a $d$-demicontractive $D$, $\mathrm{Fix}(D)$ is closed and **convex**.
- **Thm 4.2.** $\min\ell$ over $\mathrm{Fix}(D)$ is then a convex problem.
- **Thms 4.3–4.5.** Hybrid steepest descent $x_{k+1}=D_\alpha(x_k-\mu_k\nabla\ell)$ with $\alpha<(1-d)/2$ converges to a solution. Thm 4.5 covers $\mathrm{Fix}(D)\cap\arg\min\ell=\emptyset$ (feasibility only).
- The authors: whether a given denoiser is demicontractive "remains an open challenge".

**Coverage — F:** yes (no strong convexity). **N:** no. **M:** no. **Alg:** yes, relaxed $D$.

**[D] For us.** A demicontractive $D$ has a convex fixed-point set. That is a strong restriction for a nonconvex image class.

### 7.4 Hurault, Leclaire, Papadakis 2022, "Gradient step denoiser" (ICLR) and "Proximal denoiser" (ICML) [V] (`hurault2022_gsd`, `hurault2022_prox`)
https://arxiv.org/abs/2110.03220 · https://arxiv.org/abs/2201.13256

- **GS denoiser.** $D=I-\nabla g_\sigma$ with $g_\sigma=\frac12\|x-N_\sigma(x)\|^2$. Thm 1: monotone decrease, $O(1/k)$ residual, stationary cluster points. Thm 2: KL property gives iterate convergence.
- **Prop. 3.1 (Prox paper).** If $\mathrm{Lip}(\nabla g_\sigma)=L<1$, then $D=\mathrm{Prox}_{\phi_\sigma}$ for an explicit, nonconvex $\phi_\sigma$.
- **Thm 4.1 (PnP-PGD, our iteration with $P=I$).** $f$ smooth and possibly nonconvex, $\lambda L_f<1$. Then $F=\lambda f+\phi_\sigma$ decreases, cluster points are stationary, and KL gives convergence.
- **In practice.** $L<1$ is enforced by a Jacobian-norm penalty and checked only on samples.

**Coverage — F:** yes. **N:** no. **M:** no. **Alg:** yes, if $D$ is conservative.

**For us.** The most applicable *convergence* theorem. It needs $D=I-\nabla g$, i.e. a symmetric Jacobian.

### 7.5 Xu, Sun, Liu, Wohlberg, Kamilov 2020, "Provable convergence of PnP priors with MMSE denoisers"; Pritchard & Parhi 2025 [V] (`xu2020_mmse`, `pnpmmse2025`)
https://arxiv.org/abs/2005.07685 · https://arxiv.org/abs/2510.27211

- **Xu, Thm 1.** For the *exact* MMSE denoiser and $\gamma\le1/L$, PnP-ISTA decreases $g+h_{\rm mmse}$ and $\|\nabla f\|\to0$. $h_{\rm mmse}$ is convex iff $p_y$ is log-concave.
- **Pritchard–Parhi, Thms III.2 / III.4.** The MMSE regulariser is an upper Moreau envelope and is 1-weakly convex. PnP proximal gradient with $L<1$ gives monotone decrease and stationary cluster points.

**Coverage — F:** yes. **N:** no. **M:** no (exact MMSE). **Alg:** yes.

### 7.6 Gribonval 2011; Gribonval & Nikolova 2020 [V] (`gribonval2011_map`, `gribonval2020_prox`)
https://hal.inria.fr/inria-00486840 · https://arxiv.org/abs/1807.04014

- **Gribonval, Thm II.2.** If the prior is not supported on a lower-dimensional affine space, the MMSE denoiser equals $\mathrm{prox}_{\phi_{\rm MMSE}}$ for an explicit $C^\infty$ penalty. Its output is the unique stationary point of $\frac12\|y-x\|^2+\phi$.
- **Gribonval, Prop. III.2.** There exist 1-D priors whose MMSE denoiser is **expansive**.
- **Gribonval–Nikolova, Thm 1.** $f$ is a prox iff $f\in\partial\psi$ with $\psi$ convex.
- **Gribonval–Nikolova, Thm 2 ($C^1$).** $f$ is a prox iff $Df(y)$ is symmetric PSD everywhere.
- **Gribonval–Nikolova, Prop. 1.** The penalty is convex iff $f$ is also nonexpansive.

**For us.** A CNN with a non-symmetric Jacobian is the prox of *no* function. PnP-PGD with it then minimises no objective.

### 7.7 Pesquet, Repetti, Terris, Wiaux 2021; Chan, Wang, Elgendy 2017; Cohen, Blau, Freedman, Rivlin 2021 (`pesquet2021_mmo`, `chan2016_pnpadmm`)
https://arxiv.org/abs/2012.13247 · https://arxiv.org/abs/1605.01710 · [NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/97108695bd93b6be52fa0334874c8722-Abstract.html)

- **Pesquet et al., Prop. 3.1** [V]. If $\tilde J$ is firmly nonexpansive, $f$ convex and smooth, and $\gamma<2/\mu$, then $x_{n+1}=\tilde J(x_n-\gamma\nabla f(x_n))$ converges to a zero of $\nabla f+\gamma^{-1}(\tilde J^{-1}-I)$. This is exactly our iteration with $P=I$ and an underdetermined $A$ allowed. The solution set depends on $\gamma$. There is no objective unless the monotone operator is cyclically monotone.
- **Chan et al., Thm 1** [V]. A bounded denoiser plus continuation $\rho_k\to\infty$ gives convergence to an uncharacterised point.
- **Cohen et al. 2021** [A]. Gradient-of-scalar-network denoisers give convergent RED/PnP via backtracking.

### 7.8 Fang, Buchanan, Sulam 2024, "What's in a prior? Learned proximal networks for inverse problems" (ICLR) [V] (`fang2024_lpn`)
https://arxiv.org/abs/2310.14344

- **Prop. 3.1.** $f_\theta=\nabla\psi_\theta$ with $\psi_\theta$ an ICNN is the exact prox of some (possibly nonconvex) $R_\theta$.
- **Thm 3.2 (proximal matching).** In the population limit $\gamma\to0$ over all measurable $f$, the minimiser is the MAP denoiser $\mathrm{prox}_{-\sigma^2\log p}$.
- **Thms 4.1 / B.2.** PnP-ADMM and PnP-PGD converge to critical points of $\frac12\|y-Ax\|^2+\rho R_\theta$ (KL / o-minimal).
- No recovery bound.

**Coverage — F:** yes. **N:** no. **M:** no. **Alg:** yes, with an LPN as $D$.

### 7.9 Goujon, Neumayer, Unser: CRR-NN 2023 and WCRR-NN 2024 [V] (`goujon2023_crr`, `goujon2024_wcrr`)
https://arxiv.org/abs/2211.12461 · https://arxiv.org/abs/2308.10542

- **CRR, Prop. III.2.** For convex ridge regularisers, $\|Hx_1-Hx_2\|\le\|y_1-y_2\|$. Stability holds **only in the range of $H$**; the null-space component is uncontrolled.
- **WCRR, Thm 4.3.** A 1-weakly-convex ridge regulariser with KL and bounded iterates converges to a critical point. Rem. 4.6 (convexity, hence a global minimum) needs $\lambda_{\min}(H^TH)\ge\lambda$, which fails for undersampled MRI.

**Coverage — F:** experiments. **N:** no. **M:** no. **Alg:** yes.

### 7.10 Hong, Xu, Hu, Fessler 2024, P²nP (our preconditioner) [V] (`ppnp_mri2024`, also `papers_p2np/`)
https://arxiv.org/abs/2405.03854

**Thm 1.** If $D_\sigma$ is $(1+\epsilon)$-Lipschitz and $(1+\epsilon)\rho(I-\alpha PA^HA)<1$, the iterates converge linearly to a fixed point.

**[D] (reading pass).** If $A$ has a null space, then $\rho(I-\alpha PA^HA)\ge1$ for every $P$. So the theorem needs $A^HA\succ0$, which is plausible only with enough coils. It is a fixed-point result, not a recovery result.

### 7.11 Regularization theory (noise $\to0$) [V] (`ebner2022_pnpconv`, `hauptmann2024_linpnp`, `li2020_nett`, `schwab2019_nullspace`, `mukherjee2020_acr`, `lunz2018_ar`)

**Ebner & Haltmeier 2022, "PnP is a convergent regularization method"** (https://arxiv.org/abs/2212.06881).
- Admissible family (B1)–(B4): contractions converging to $I$.
- Thm 3.10: stability with constant $\frac{\gamma\|A\|\mathrm{Lip}\Phi}{1-\mathrm{Lip}\Phi}$.
- Thm 3.14: as $\delta\to0$ with an existential $\lambda_k$, limits solve $Ax=y$.
- **Thm 3.15 (which solution).** $Ax^\ast =y$ and $H(x^\ast )\in\ker(A)^\perp$, where $H=\lim(\Phi_\lambda^{-1}-I)/\lambda$.
- [D] (B1) is impossible for a shift-equivariant (normalization-equivariant) denoiser, since $J\mathbf 1=\mathbf 1$ forces $\mathrm{Lip}\ge1$.

**Hauptmann, Mukherjee, Schönlieb, Sherry 2024** (https://arxiv.org/abs/2307.09441).
- Thm 5: for linear, symmetric, contractive denoisers with spectral filters, PnP-FBS converges to $\arg\min\frac12\|Ax-y\|^2+J_\tau$, and as $\delta\to0$ to the $J^\ast $-minimising solution.
- Sec. 4: input scaling cancels for linear denoisers. [D] The same holds for any positively 1-homogeneous denoiser, so a separate strength knob is needed.

**NETT (Li, Schwab, Antholzer, Haltmeier 2020)** (https://arxiv.org/abs/1803.00092).
- Thm 2.6: convergence to **R-minimising** solutions.
- Thm 3.1 / Prop. 3.3: rate $B_R(x_{\alpha,\delta},x^+)=O(\delta)$ under the source condition $R'(x^+)\in\mathrm{Ran}(F^\ast )$.
- The authors note that the R-minimising solution need not be the desired one for out-of-class data.

**Null-space networks (Schwab, Antholzer, Haltmeier 2019)** (https://arxiv.org/abs/1806.06137).
- Thm 2.8: $\sup\|R_{\alpha}(y^\delta)-x\|\le c\,\delta^{2\mu/(2\mu+1)}\rho^{1/(2\mu+1)}$, **only for $x\in(I+\Phi)(\mathrm{ran}(A^\ast A)^\mu B_\rho)$**.
- So the network chooses the null-space component, and the truth is recovered only if it lies in the network's range.

**ACR (Mukherjee et al.)** (https://arxiv.org/abs/2008.02839).
- Prop. 2: stability $\|\hat x_\alpha(y^\delta)-\hat x_\alpha(y)\|\le\|A\|\delta/(\alpha\rho_0)$.
- Prop. 3: convergence to the $R$-minimising solution.

**Adversarial regularizers (Lunz, Öktem, Schönlieb)** (https://arxiv.org/abs/1805.11572).
- Thm 1: a distributional statement (one gradient step decreases $W_1$ to the true distribution).
- Not a per-image bound.

**Common message.** Every regularization-theoretic result picks the element of $\{Ax=y\}$ preferred by the regulariser or denoiser. It equals the truth only if the truth happens to be that element. This is the formal version of "the network decides the null space".

---

## 8. Manifold, union-of-subspaces, model-based CS, inexact projections, tangent-space RIP

### 8.1 Baraniuk & Wakin 2009, "Random projections of smooth manifolds" [S via Wakin 2010]; Wakin 2010 [V] (`wakin2010`)
FoCM 9:51–77, doi:10.1007/s10208-007-9011-z · https://arxiv.org/abs/1002.1247

**Embedding (restated in Wakin 2010 as Thm 3).** $\mathcal M$ is a compact $K$-dimensional submanifold with condition number $1/\tau$ (reach $\tau$), volume $V$ and covering regularity $R$. A random orthoprojector with
$$M=O\Big(\frac{K\log(NVR\tau^{-1}\epsilon^{-1})\log(1/\rho)}{\epsilon^2}\Big)$$
gives a $(1\pm\epsilon)$ isometry on all secants with probability $\ge1-\rho$. The explicit constants are large.

**Wakin 2010, Thm 5.** For the nearest point in measurement space,
$$\|x-\hat x\|\le(1+0.25\epsilon)\|x-x^\ast \|+(2+0.32\epsilon)\|\eta\|+\frac{\epsilon^2\tau}{936N}.$$

**Coverage — F:** no. **N:** yes. **M:** yes. **Alg:** idealised.

### 8.2 Eftekhari & Wakin 2015, "New analysis of manifold embeddings and signal recovery from compressive measurements" [V] (`eftekhari2015`)
https://arxiv.org/abs/1306.4748

**Thm 2.** Gaussian $\Phi$, $0<\epsilon\le1/3$, and
$$M\ge18\epsilon^{-2}\max\big(24K+2K\log\frac{\sqrt K}{\tau\epsilon^2}+\log(2V^2),\log\frac8\rho\big)$$
give a $(1\pm\epsilon)$ embedding. There is no dependence on the ambient dimension $N$.

**Thm 3 (any $\Phi$ satisfying the embedding).** $\|x-\hat x\|\le(1+2\epsilon)(2\sigma_{\max}(\Phi)+1)\|x-x^\ast \|+(2+4\epsilon)\|n\|$. Prop. 1 shows the $\sqrt{N/M}$ factor is unavoidable for uniform bounds.

**Thm 4 (non-uniform).** $\|x-\hat x\|\le\min\big((1+3\epsilon)\|x-x^\ast \|+\frac{\epsilon\tau}{40},\dots\big)+(2+4\epsilon)\|n\|$.

**Coverage — F:** no. **N:** yes. **M:** yes, explicit. **Alg:** idealised.

**For us.** The reach $\tau$ of the model set enters only additively ($\epsilon\tau/40$).

### 8.3 Iwen & Maggioni 2013, "Approximation of points on low-dimensional manifolds via random linear projections" [V] (`iwen2013`)
https://arxiv.org/abs/1204.3337

**Setting.** A multiscale piecewise-affine model (GMRA, local $d$-dimensional charts $P_{j,k}$). The algorithm picks a chart from the data, $k'=\arg\min_k\|Mx-Mc_{j,k}\|$, then solves least squares in that chart.

**Thm 2.** Isometry only on a finite set of centres and on the local subspaces gives $\|x-A_Mx\|<100.3\|x-x_{opt}\|+O(2^{-j})$.

**Props. 1–2.** $m=O(d\log(d/(\delta\cdot\mathrm{reach}))+\log V)$, non-uniform, or with $\log D$ for the uniform version.

**Coverage — F:** no (JL). **N:** no. **M:** yes. **Alg:** yes, non-iterative.

**For us.** This is the explicit "data-selected local basis" result, with a local-RIP-style requirement.

### 8.4 Blumensath 2011, "Sampling and reconstructing signals from a union of linear subspaces" [V] (`blumensath2011`)
https://arxiv.org/abs/0911.3514 (IEEE TIT 57(7))

**Def. 1.** Bi-Lipschitz on the model set $A$ (any proximal set; footnote 1).

**Thm 2.** $x^{n+1}=P_A(x^n+\mu\Phi^\ast (y-\Phi x^n))$, the **exact nearest-point projection**. If $\beta\le1/\mu<1.5\alpha$, then after $n^\star$ iterations
$$\|x-x^{n^\star}\|\le(c^{0.5}+\delta)\|e_A\|+\|x_A-x\|,\qquad e_A=\Phi(x-x_A)+e .$$
In the proof, $c\le4/(3\alpha-2/\mu)$.

**Thm 1.** A matching lower bound.

**$\epsilon$-approximate projection [S, Hegde–Indyk–Schmidt §1.3].** In the journal version, an additive $\epsilon$-projection yields $O(\epsilon)$-neighbourhood recovery. **Not found in arXiv v2**; constants not seen.

**Coverage — F:** any $A$ satisfying bi-Lipschitz. **N:** yes. **M:** yes. **Alg:** yes, exact projection.

### 8.5 Blumensath & Davies 2009, "Sampling theorems for signals from the union of finite-dimensional linear subspaces" [V] (`blumensath2009`)
IEEE TIT 55(4):1872–1882

**Thm 3.3 (corrected preprint).** Sub-Gaussian $\Phi$ satisfies RIP on a union of $L$ subspaces of dimension $k$ if
$$M\ge\frac1{c(\delta/6)}\big(\ln2L+k\ln\frac{36}{\delta}+t\big),$$
with probability $1-e^{-t}$.

**Cor. 3.6.** $\log L$ dependence is necessary.

### 8.6 Baraniuk, Cevher, Duarte, Hegde 2010, "Model-based compressive sensing" [V] (`baraniuk2010mbcs`)
https://arxiv.org/abs/0808.3572

**Definitions.** Model-RIP (Def. 3); nested approximation property (Def. 5); restricted amplification property (RAmP, Def. 7).

**Thm 4.** Model-CoSaMP with $\delta_{\mathcal M_K^4}\le0.1$ gives $\|x-\hat x_i\|\le2^{-i}\|x\|+15\|n\|$.

**Thm 5 (model-compressible).** Adds $35C_s|x|_{\mathcal M_s}K^{-s}(1+\ln\lceil N/K\rceil)$. Model-IHT versions are in App. C.

**Coverage — F:** generic RIP. **N:** yes. **M:** via compressibility. **Alg:** yes, exact model projection.

### 8.7 Hegde, Indyk, Schmidt 2015, "Approximation algorithms for model-based compressive sensing" [V] (`hegde2015approx`)
https://arxiv.org/abs/1406.1579

**Defs. 7–8.**
- Head oracle: $\|x_{H(x)}\|\ge c_H\|x_\Omega\|$.
- Tail oracle: $\|x-x_{T(x)}\|\le c_T\|x-x_\Omega\|$.

**§3 (negative).** A tail-only approximate oracle ($c_T>1$) inside model-IHT can return 0 forever.

**Thm 11 (AM-IHT).** $\|r^{i+1}\|\le\alpha\|r^i\|+\beta\|e\|$ with $\alpha=(1+c_T)[\delta+\sqrt{1-\alpha_0^2}]$ and $\alpha_0=c_H(1-\delta)-\delta$. Convergence needs $(1+c_T)\sqrt{1-c_H^2}<1$.

**Cor. 19 (boosting).** Works for any constants, at the price of a larger RIP.

**For us.** "Close to a projection" in the tail sense is provably insufficient. A head-type step is also needed, one that captures the gradient's energy in the model's local directions.

### 8.8 Shah & Chandrasekaran 2011 (MIP); Hegde & Baraniuk 2012 (SPIN) [V] (`shah2011mip`, `hegde2012incoh`)
people.lids.mit.edu/pari/manifold.pdf · https://arxiv.org/abs/1202.1595

**Shah–Chandrasekaran.**
- Thm 2: manifold-RIP with $\delta<1/3$; PGD onto $\mathcal M$ gives $\psi(x_{k+1})\le\frac{2\delta}{1-\delta}\psi(x_k)$.
- Thm 4: with an $\epsilon_P$-approximate projection ($\|z-y\|^2\le\|P_{\mathcal M}(y)-y\|^2+\epsilon_P$), the error floor is $\epsilon_P'=\frac{(1+\delta)(1-\delta)}{1-3\delta}\epsilon_P$.
- Thm 5: tracking a moving target.
- The proofs of Thms 2–4 are omitted in the paper.

**Hegde–Baraniuk, Thms 2–3.** Two $\epsilon$-incoherent manifolds (signal = $a+b$) with $\delta<\frac{1-11\epsilon}{3+7\epsilon}$. Approximate projections add $\frac{1+\delta}{1-\alpha}\gamma$.

**For us.** A natural template for "anatomy + flow" or "static + dynamic" splits.

### 8.9 Golbabaee & Davies 2018, "Inexact gradient projection and fast data driven compressed sensing" [V] (`golbabaee2018inexact`)
https://arxiv.org/abs/1706.00092

**Main assumption.** A uniform upper bound $\beta$, plus a **local lower bound at the truth only**: $\|A(x-x^\ast )\|^2\ge\alpha_{x^\ast }\|x-x^\ast \|^2$ for $x\in C$.

**Thm 1 (fixed-precision projection, $\|u-x\|^2\le\inf\|u'-x\|^2+\nu_p^2$).** If $\beta<2\alpha_{x^\ast }$, then
$$\|x^k-x^\ast \|\le\rho^k\big(\|x^\ast \|+\textstyle\sum\rho^{-i}e^i\big)+\frac{2\sqrt\beta}{\alpha_{x^\ast }(1-\rho)}w .$$

**Thm 2 ($(1+\epsilon)$-relative projection).** Requires $\sqrt{2\epsilon+\epsilon^2}\le\delta\sqrt{\alpha_{x^\ast }}/\|A\|$, i.e. a tiny $\epsilon$.

**For us.** An additive-error projection is forgiving. A relative-error projection is fragile, scaling with $\|A\|$.

### 8.10 Wei, Cai, Chan, Leung 2016, "Guarantees of Riemannian optimization for low rank matrix recovery" [V] (`wei2016riem`)
https://arxiv.org/abs/1511.01562

**Setting.** $W_l=X_l+\alpha_lP_{S_l}(G_l)$ and $X_{l+1}=H_r(W_l)$, where $S_l$ is the **tangent space at the current iterate**. This is the moving basis.

**Lemma 4.1 (curvature).** $\|(I-P_{S_l})X\|_F\le\|X_l-X\|_2\|X_l-X\|_F/\sigma_{\min}(X)$.

**Thm 2.1.** If $\gamma=\frac{4R_{2r}+2R_{3r}}{1-R_{2r}}+\frac{4R_{2r}}{\sigma_{\min}}\|X\|_F<1$, then $\|X_l-X\|_F\le\gamma^l\|X_0-X\|_F$.

**Coverage — F:** generic RIC; entry sampling not covered. **N:** no. **M:** no. **Alg:** yes.

**For us.** This is the only proof found in which the basis genuinely moves with the iterate. Its ingredients are:
- RIP on sums of tangent spaces;
- a curvature bound (an effective reach);
- a retraction that is a near-projection.

These are exactly the ingredients missing for $\mathrm{Fix}(D)$.

### 8.11 Gaussian-mixture CS [V] (`renna2014`, `yu2011sapiro`, `chen2010mfa`)
https://arxiv.org/abs/1307.0861 · https://arxiv.org/abs/1010.4314 · PMC3721352

**Renna, Calderbank, Carin, Rodrigues 2014, Thms 2–3.** Gaussian $\Phi$, GMM prior with component ranks $s_k$. Then $\lim_{\sigma\to0}\mathrm{MMSE}=0$ if $\ell>s_{\max}$, and only if $\ell\ge s_{\max}$. The phase transition is set by the **maximum local rank**, not the size of the union.

**Yu & Sapiro, Thms 2–5.** Instance optimality in expectation via a null-space property, with "linear RIP" in the PCA basis.

**Chen et al. 2010, Thms 1–2.** Mixture-of-factor-analysers classification bounds and block-sparse RIP.

**For us.** This is the best theoretical support for "sample complexity ≈ local Jacobian rank of the denoiser", since an MMSE denoiser on a GMM is a soft local projection. The support is for Gaussian $\Phi$ only.

---

## 9. Denoiser geometry: candidate measurable assumptions

### 9.1 Miyasawa / Tweedie, first and second order [V] (`kadkhodaie2024`, `manor2024`, `meng2021`)

For $y=x+\sigma z$ with $z\sim N(0,I)$:
- $D^\ast (y)=\mathbb E[x\mid y]=y+\sigma^2\nabla\log p_\sigma(y)$.
- $\nabla D^\ast (y)=I+\sigma^2\nabla^2\log p_\sigma(y)=\sigma^{-2}\mathrm{Cov}[x\mid y]$, which is symmetric PSD (Kadkhodaie et al. 2024, eqs. (8)–(9), App. D).
- Manor & Michaeli 2024, Thm 2, gives all higher posterior moments, e.g. $\mu_3=\sigma^2\partial\mu_2/\partial y$, plus Algorithm 1 (posterior principal components via JVP subspace iteration).
- Meng et al. 2021, eq. (18): $\mathrm{Cov}[x\mid\tilde x]=\sigma^4\nabla^2\log q_\sigma+\sigma^2I$.

**Exact MSE identity (Kadkhodaie 2024, eq. (10)).** $\mathrm{MSE}(D^\ast ,\sigma^2)=\mathbb E\,\mathrm{tr}\,\mathrm{Cov}[x\mid y]=\sigma^2\,\mathbb E\,\mathrm{tr}\nabla D^\ast (y)$. This is total, not per-pixel, error.

**SURE (any weakly differentiable $D$, eq. (7)).** $\mathrm{MSE}(D)=\mathbb E[2\sigma^2\mathrm{tr}\nabla D(y)+\|y-D(y)\|^2]-n\sigma^2$.

### 9.2 Kadkhodaie & Simoncelli 2021, "Solving linear inverse problems using the prior implicit in a denoiser" [V] (`kadkhodaie2021`)
https://arxiv.org/abs/2007.13640

- The only thing proven is the Miyasawa relation.
- The coarse-to-fine stochastic ascent (Alg. 1) and the constrained version (Alg. 2, projection onto $\{x:M^Tx=x_c\}$) come with **no** convergence or recovery theorem. Their convergence is asserted and shown empirically.
- The blind noise estimate $\hat\sigma=\|D(y)-y\|/\sqrt n$ is empirical.

### 9.3 Kadkhodaie, Guth, Simoncelli, Mallat 2024, "Generalization in diffusion models arises from geometry-adaptive harmonic representations" (ICLR) [V] (`kadkhodaie2024`)
https://arxiv.org/abs/2310.02557

**Proven.**
- The identities in 9.1.
- Oracle shrinkage in a fixed basis: $\lambda_k=\frac{\langle x,e_k\rangle^2}{\langle x,e_k\rangle^2+\sigma^2}$, with error $\sim M\sigma^2+\|x-x_M\|^2$. Decay $\langle x,e_k\rangle^2\sim k^{-(\alpha+1)}$ gives error $\sim\sigma^{2\alpha/(\alpha+1)}$.
- For bias-free nets, $D(y)=\nabla D(y)\,y$, giving the expansion $\sum_k\lambda_k(y)\langle y,e_k(y)\rangle e_k(y)$. Symmetry and non-negativity of the Jacobian are **assumed** for networks, and proven only for $D^\ast $.

**Not defined formally.** GAHB: verbatim, "we do not provide a formal mathematical definition". The term "effective dimension" is not used; $\mathrm{tr}\nabla D$ plays that role.

**Empirical.** Near-optimal PSNR slopes on $C^\alpha$ images. The number of non-negligible eigenvalues grows as $\sigma$ decreases. There is a memorisation-to-generalisation transition between $10^3$ and $10^5$ training images.

### 9.4 Mohan, Kadkhodaie, Simoncelli, Fernandez-Granda 2020, "Robust and interpretable blind image denoising via bias-free CNNs" (ICLR) [V] (`mohan2020`)
https://arxiv.org/abs/1906.05478

**Lemma 1.** A ReLU network with no additive constants (skip connections allowed) satisfies $D(\alpha y)=\alpha D(y)$ for $\alpha\ge0$. Hence $D(y)=A_yy$ (a.e.), with $A_y$ the Jacobian on the current activation region.

**Definition.** Effective dimension $d:=\sum_is_i^2=\|A_y\|_F^2=\mathbb E\|A_yn\|^2/\sigma^2$. This is the Frobenius norm, **not** the trace.

**Empirical.**
- $A_y$ is approximately symmetric and acts as an approximate projection.
- **$d\propto1/\sigma$.**
- The subspaces are nested across noise levels.

### 9.5 Stanczuk, Batzolis, Deveney, Schönlieb 2024; Pidstrigach 2022 [V] (`stanczuk2024`, `pidstrigach2022`)
https://arxiv.org/abs/2212.12611 · https://arxiv.org/abs/2206.01018

**Stanczuk et al., Thm 5.1 / D.1.** For data on a compact embedded manifold with a density bounded below, and a point in the tubular neighbourhood, the score aligns with the normal direction as $t\to0$, and the tangential/normal ratio $\to0$. The dimension estimator (SVD of score vectors) is heuristic.

**Pidstrigach, Thms 1–2.** SGM samples stay on the data manifold. Bounded score error implies memorisation.

**[D] For us.** $D(y)-y=\sigma^2s(y)$. So for small $\sigma$ the residuals lie in the normal space, and the eigenvalues of $J_D$ near 1 span the tangent space.

### 9.6 Herbreteau, Moebel, Kervrann 2023, "Normalization-equivariant neural networks with application to image denoising" (NeurIPS) [V] (`herbreteau2023`)
https://arxiv.org/abs/2306.05037

- **Def.** $f(\lambda x+\mu\mathbf 1)=\lambda f(x)+\mu\mathbf 1$ for $\lambda>0$, $\mu\in\mathbb R$.
- **Lemma 2.** Preserved by composition, concatenation and affine combinations $(1-t)f+tg$.
- **Prop. 2.** No element-wise NE nonlinearity exists except the identity.
- **Prop. 3.** For their architecture (affine convolutions, sort pooling, affine skips), $f$ is continuous piecewise-linear with $f(y)=A_yy$ and $A_y\mathbf 1=\mathbf 1$ on each piece.

### 9.7 Is the MMSE denoiser nonexpansive?

**[D, standard] Log-concave prior.** The posterior potential $V=-\log p+\|x-y\|^2/(2\sigma^2)$ has $\nabla^2V\succeq\sigma^{-2}I$. Brascamp–Lieb then gives $\mathrm{Cov}[x\mid y]\preceq\sigma^2I$. So $0\preceq J^\ast \preceq I$, and $D^\ast =\nabla\psi$ with $\psi$ convex and $1$-smooth, which makes $D^\ast $ firmly nonexpansive (Baillon–Haddad). Primary sources were not read; Gribonval & Machart (NeurIPS 2013) is cited for "MMSE penalty convex iff $p_Y$ log-concave" [S].

**Non-log-concave priors.** $J^\ast $ can have eigenvalues $>1$ (bimodal posteriors, points near the medial axis). Gribonval Prop. III.2 [V] gives an expansive example.

**Compact support (Wolf et al. 2026, Prop. 4.10) [V~].** With $\|x\|\le M$, $D^\ast $ is $M^2/\sigma^2$-Lipschitz and $(\sigma^2/M^2)$-cocoercive. This bound is useless as $\sigma\to0$.

### 9.8 [D] Consequences for our denoiser

| property of our $D$ | consequence |
|---|---|
| shift equivariance, $J\mathbf 1=\mathbf 1$ | $\Vert J_D\Vert \ge1$ and $\mathrm{Lip}(D)\ge1$. Rules out every *contraction* hypothesis (Ryu with $\mu=0$, Ebner–Haltmeier (B1), Shoushtari $\lambda<1$). Harmless for $A$ itself as long as the DC component is sampled. |
| positive homogeneity (bias-free or NE) | $\mathrm{Fix}(D)$ and $\mathrm{Im}(D)$ are cones. A uniform *additive* projection error $\sup_x\Vert D(x)-\Pi_\Sigma(x)\Vert \le\eta$ (Joundi, Beygi Thm 3, Raj) is **impossible** unless $D=\Pi_\Sigma$: both maps are homogeneous for a cone $\Sigma$, so the error scales with $\Vert x\Vert $. Such results must be applied on the bounded set the iterates visit, or in relative form (Golbabaee–Davies Thm 2), which is much more fragile. |
| non-symmetric Jacobian (generic CNN) | not a prox (Gribonval–Nikolova Thm 2); no objective; Hurault, Xu and Fang do not apply. |
| MMSE-like training | not idempotent ($D\circ D\ne D$, it shrinks again). So $D$ is not a projection onto $\mathrm{Fix}(D)$, and $\beta_{\mathrm{Fix}D}(D)<\infty$ fails (Traonmilin: finite $\beta$ implies idempotence). Only a soft-projection analysis fits (§13). |
| mixing with wavelet soft-threshold | soft-thresholding the detail bands preserves shift equivariance (constants live in the scaling band) but breaks scale equivariance unless the threshold scales with the input. Convex combinations of nonexpansive maps stay nonexpansive. |

---

## 10. Negative and stability results

### 10.1 Antun, Renna, Poon, Adcock, Hansen 2020, "On instabilities of deep learning in image reconstruction and the potential costs of AI" (PNAS) [V] (`antun2020`)
https://arxiv.org/abs/1902.05300

**No theorem; everything is empirical** (arXiv v1 read; the PNAS page returned 403). Six networks were tested.
- Tiny perturbations, found by gradient ascent on $\frac12\|f(y+Ar)-p(x)\|^2-\frac\lambda2\|r\|^2$, cause severe artefacts.
- Small structures are lost.
- More samples can make results worse.

The attack doubles as an estimator of the local Lipschitz constant of a whole reconstruction pipeline.

### 10.2 Gottschling, Antun, Hansen, Adcock 2025, "The troublesome kernel" (SIAM Review); Gottschling, Campodonico, Antun, Hansen, kersize [V] (`gottschling2025`, `gottschling2023kersize`)
https://arxiv.org/abs/2001.01258 (v4) · https://arxiv.org/abs/2311.16898

**Notation.** $L_\epsilon(\Psi,y)=\sup_{0<|||z-y|||\le\epsilon}\|\Psi(z)-\Psi(y)\|/|||z-y|||$.

**Thm 4.1 (hallucination by detail transfer).** Let $|||Ax_{Det}|||\le\delta$, and let $\Psi$ be $L$-Lipschitz with $\|\Psi(A(x+x_{Det}))-(x+x_{Det})\|\le\delta$. Then for every $e\in B(0,\delta)$, $\Psi(Ax+e)=x+x_{Det}+z$ with $\|z\|\le(1+2L)\delta$. Stability does not prevent this.

**Thm 4.3 (no free lunch).** Accuracy on a training set $T$ implies in-distribution hallucinations for families $\mathcal M_1\supset T$.

**Thm 4.5 (accuracy–stability trade-off).** Suppose $\|\Psi(Ax)-x\|<\eta$, $\|\Psi(Ax')-x''\|<\eta$ and $|||A(x-x')|||\le\eta$. Then on a ball around $y=Ax$:
$$L_\epsilon(\Psi,\tilde y)\ge(\|x-x''\|-2\eta)/\eta .$$

**Thm 4.6 / 4.7.** These instabilities occur with positive probability under random noise. For $A=P_\Omega U$, an explicit Gaussian probability bound is given.

**Thm 4.10.** Lipschitz maps with $\sigma_{\min}(A)L<1$ cannot be optimal on certain model classes.

**Sec. 6.3.** Enforcing data consistency $A\Psi(y)=y$, or training with several sampling patterns, "neither … helps prevent" these mechanisms.

**App. B (positive).** If $AH^\ast $ has the rNSP, QCBP recovery forces $\|x-x''\|\le(C_1+C_2+1)\eta$, and the trade-off is void.

**Companion paper.**
- Def. 3.2: $\mathrm{kersize}=\sup_y\mathrm{diam}(F_y)$.
- Thm 3.4: $\mathrm{kersize}/2\le c_{opt}\le\mathrm{kersize}$.
- Prop. 4.1: under an rNSP relative to $\mathcal M_1$, $A$ is injective on $\mathcal M_1$ iff $(\mathcal M_1-\mathcal M_1)\cap N(A)=\{0\}$, and then $\mathrm{kersize}\le D_2\eta$.

"More samples can hurt" is **not** a theorem in any version; it is cited empirically.

**For us.**
- The PnP limit $\Psi(y)$ is a reconstruction map, so all of this applies to it.
- Our denoiser alone decides the $\ker A$ component (or the near-kernel component, $|||Ax_{Det}|||\le\delta$, in the injective multi-coil case).
- Stability is possible only if the implied model set contains no pair whose difference is near the kernel.

### 10.3 Bhadra, Kelkar, Brooks, Anastasio 2021, "On hallucinations in tomographic image reconstruction" (IEEE TMI) [V] (`bhadra2021`)
https://arxiv.org/abs/2012.00646

**Definitions only (no theorems).**
- Measurement-space hallucination map: $\hat\theta_{meas}-\hat\theta_{tp}$, computable without ground truth.
- Null-space hallucination map: $\hat\theta_{null}-\theta_{null}$, which needs ground truth.

Both are defined via the truncated pseudo-inverse $H^+_P$. For multi-coil spiral, early-stopped CG approximates $H^+_P$ [D].

### 10.4 Colbrook, Antun, Hansen 2022 (negative part) [V]
**Thms 1–2.** Stable and accurate networks exist, but in general cannot be computed or trained to arbitrary accuracy.

### 10.5 Computational hardness of posterior sampling [V] (`gupta2024`, `brunahan2024`)
**Gupta, Jalal, Parulekar, Price, Xun 2024, Thm 1.8** (https://arxiv.org/abs/2402.12727). If one-way functions exist, then for $m>d^{0.01}$ there is a well-modelled prior (polynomial-size score networks, accurate scores) and a linear measurement model on which posterior sampling needs super-polynomial time.
- The hard instance uses **pure coordinate subsampling**, $A=[0\ I]$.
- **[D, reading pass]** Cartesian subsampled Fourier is unitarily equivalent to coordinate subsampling (apply the hard prior in the Fourier domain). So worst-case intractability transfers to Cartesian MRI.

**Bruna & Han 2024, Prop. 4** (https://arxiv.org/abs/2407.00745). Ising-type hardness. There is an intermediate-SNR band where sampling is hard (§11.2).

### 10.6 Other negative facts collected above
- Hegde–Indyk–Schmidt §3: tail-only approximate projections can fail (§8.7).
- Jalal et al. (MRI): robustness to prior mismatch under Fourier measurements "is not true for general distributions" (§4.7).
- Bigot–Boyer–Weiss Prop. 4.5: line blocks cost as much as single samples without structure (§5.3).
- Berk et al. Prop. III.1: coherence $\ge\sqrt{k/n}$ gives a quadratic-in-$k$ bottleneck (§4.1).
- Chung et al. (DPS) Thm 1: the constant in the Jensen-gap bound appears wrong (§11.6).

---

## 11. Diffusion / score priors, 2022–2026

### 11.1 Xu & Chi 2024, "Provably robust score-based diffusion posterior sampling for plug-and-play image reconstruction" (NeurIPS) [V] (`xuchi2024`)
https://arxiv.org/abs/2403.17042

**Setting.** Alternate (i) a proximal consistency sampler, which for Gaussian data is a Gaussian with mean $(A^T\Sigma^{-1}A+\eta_k^{-2}I)^{-1}(A^T\Sigma^{-1}y+\eta_k^{-2}\hat x_k)$, and (ii) a sample from the *denoising posterior*.

**Thm 1.** Exact samplers plus annealing $\eta\downarrow0$ give convergence to the posterior in distribution.

**Thm 2.** With fixed $\eta$ and per-step TV errors,
$$\mathrm{TV}(p_{\hat x_K},\pi_\eta)\lesssim\varepsilon_{acc}\sqrt{\chi^2(\cdot)}+\frac{1}{1-\lambda}(\varepsilon_{DDS}+\varepsilon_{PCS})\log\frac1{\varepsilon_{acc}} .$$
$\lambda$ exists but may be exponentially close to 1.

**Coverage — F:** experiments only. **N:** yes. **M:** score error, as a black box. **Alg:** yes. No truth bound.

### 11.2 Bruna & Han 2024, "Posterior sampling with denoising oracles via tilted transport" [V] (`brunahan2024`)
https://arxiv.org/abs/2407.00745

**Thm 5 / Cor. 6.** An exact reduction: sample a boosted tilted posterior $\nu_T$, then run the unconditional reverse SDE.

**Prop. 10.** $\nu_T$ is strongly log-concave if $\chi_{\|Q\|}(\pi)<\|Q\|^{-1}\frac{\kappa}{\kappa-1}$. When $Q=\sigma^{-2}A^TA$ is degenerate, the condition is $\chi_{\|Q\|}(\pi)<\|Q\|^{-1}$.

**Cor. 17 (Gaussian mixtures).** Sampling is easy at low and high SNR and hard in between. The hard band vanishes as $\kappa(A)\to1$.

**Prop. 16.** KL stability in terms of the score error *along posterior paths*.

Continuous-time only ("not (yet) algorithmic").

### 11.3 Sun, Wu, Chen, Feng, Bouman 2023 (PMC); Wu, Sun, Chen, Zhang, Yue, Bouman 2024 (PnP-DM) [V] (`sun2023`, `wu2024pnpdm`)
https://arxiv.org/abs/2310.10835 · https://arxiv.org/abs/2405.18782

**PMC-RED / PnP, Thms 1–4.** Assumptions: Lipschitz likelihood gradient, Lipschitz prior score, Lipschitz learned score, and a uniform score error $\varepsilon_\sigma$. Conclusion:
$$\frac1{N\gamma}\int\mathrm{FI}(\nu_t\|\pi)\le\frac{4\mathrm{KL}}{N\gamma}+24nL^2\gamma+24C^2\sigma^2+24\varepsilon_\sigma^2 .$$
This is Fisher-information stationarity only. The full-dimensional smooth-prior assumption excludes manifold-supported priors.

**PnP-DM, Thm 3.1.** An analogous average-FI bound for split Gibbs with EDM priors. The target equals the posterior only as $\rho\to0$.

### 11.4 Laumont, De Bortoli, Almansa, Delon, Durmus, Pereyra 2022 (PnP-ULA) and 2023 (PnP-SGD) [V] (`laumont2022`, `laumont2023`)
https://arxiv.org/abs/2103.04715 · https://arxiv.org/abs/2201.06133

**PnP-ULA.** Assumptions:
- H2: $\mathrm{Lip}(I-D_\varepsilon)\le L$ and $\|D_\varepsilon-D_\varepsilon^\ast \|\le M_R$ on $B(0,R)$, where $D^\ast _\varepsilon$ is the MMSE denoiser.
- H3: strong log-concavity of the likelihood, or a compact convex set if $m\le0$.

Conclusions:
- Prop. 5: geometric ergodicity if $2\lambda(L_y+L/\varepsilon-\min(m,0))\le1$.
- Prop. 6–7: bias $\lesssim\delta^{1/2}+M_R+e^{-R}+\dots$.
- Prop. 4: training excess risk $\eta$ gives $M_R\le C\eta^{1/(2d+2)}$.

**PnP-SGD, Prop. 3.** $\limsup d(X_k,S)\le C\,M(R)^{r}$, where $S$ is the stationary set.

**Coverage — F:** no. **N:** yes. **M:** yes, linearly via $M_R$. **Alg:** yes.

**For us.** This is the Bayesian analogue of our iteration (denoiser residual used as a score). It gives no recovery bound.

### 11.5 Rout, Raoof, Daras, Caramanis, Dimakis, Shakkottai 2023 [V] (`rout2023`)
https://arxiv.org/abs/2307.00619

**Thms 3.4 / 3.8.** For a *linear* generative model $x=Sw$ with $S^TS=I$ and $(AS)^T(AS)\succ0$, noiseless, DPS and PSLD recover exactly. This is a sanity check (a determined system in latent space).

### 11.6 Chung et al. 2022: MCG and DPS [V] (`chung2022mcg`, `chung2022dps`)
https://arxiv.org/abs/2206.00941 · https://arxiv.org/abs/2209.14687

**MCG.** Assumes a locally *linear* manifold. Prop. 2: the denoiser is a local orthogonal projection, $J^2=J=J^T$. Thm 1: the measurement gradient lies in the tangent space. Geometric statements only.

**DPS, Thm 1 (Jensen gap).** $\le\frac{d}{\sqrt{2\pi\sigma^2}}e^{-1/2\sigma^2}\|\nabla A\|m_1$.

**[D, reading pass] The DPS constant appears wrong.** Lemma 2's $\sup|\phi'|$ for $N(\mu,\sigma^2)$ should be $e^{-1/2}/(\sqrt{2\pi}\sigma^2)$, and Lemma 3 drops the $(2\pi\sigma^2)^{-d/2}$ normalisation. So the $\sigma$-dependence of the bound is unreliable.

### 11.7 Montanari & Wu 2023, "Posterior sampling in high dimension via diffusion processes" [V] (`montanari2023`)
https://arxiv.org/abs/2304.11449

**Thm 6.** Gaussian design $X$ with i.i.d. $N(0,1/p)$ rows and $n/p\to\delta\ge\delta_0(\pi_\Theta)$, product prior with bounded support. Then AMP-based diffusion sampling achieves $W_{2,p}\le\xi$ with probability $1-o(1)$. Not image-like.

### 11.8 Others (2025–2026)
- **Karan, Shah, Chen 2025 (ReGuidance)**, https://arxiv.org/abs/2506.10955 [V]. Hypercube Gaussian mixture with *inpainting*: the output is within $\mathrm{poly}(\sigma,e^{-T})$ of $\{Ax=y\}$.
- **Chang et al. 2025**, https://arxiv.org/abs/2512.08022 [V statement, constants not extracted]. An $\mathbb E W_2^2$ posterior bound under semi-log-concavity.
- **Ma, Wu, Liu 2026**, https://arxiv.org/abs/2610.11834 [A]. One-bit CS posterior sampling with covering-number sample complexity and $W_\infty$-mismatch robustness.
- **Pesme et al. 2025**, https://arxiv.org/abs/2507.15397 [A]. A denoiser-based scheme converges to the prox for log-concave $p$.
- **Gavaskar, Athalye, Chaudhury 2022**, https://arxiv.org/abs/2207.13031 [S]. Exact and robust PnP-CS recovery with *linear* (kernel) denoisers.

**No result found** gives sample-complexity guarantees for a *denoiser* (rather than generator) prior with Fourier sampling.

---

## 12. What the two surveys add

**Scarlett, Heckel, Rodrigues, Hand, Eldar 2022/23, "Theoretical perspectives on deep learning methods in inverse problems"** [V] (`scarlett2022`), https://arxiv.org/abs/2206.14373.
- Covers generative priors (S-REC, lower bounds, Hand–Voroninski), untrained networks (deep decoder, Heckel–Soltanolkotabi), unfolding (LISTA rates, generalisation bounds), and PnP/RED pointers only.
- It lists **structured (subsampled-Fourier) measurements as open**, citing only Berk et al.

Additional items it points to, beyond the ones above:
- deep decoder: https://arxiv.org/abs/1810.03982;
- rate-optimal denoising with expansive generators: https://arxiv.org/abs/1805.08855;
- median-of-means robust CS: https://arxiv.org/abs/2006.09461;
- generalised Lasso with generative priors: https://arxiv.org/abs/2006.12415;
- phase retrieval with generative priors: https://arxiv.org/abs/1807.04261, https://arxiv.org/abs/2106.15358;
- NP-hardness of inverting generators: https://arxiv.org/abs/1906.07437;
- Langevin-based recovery: https://arxiv.org/abs/2102.12643, https://arxiv.org/abs/2206.09104;
- non-expansive generators: https://arxiv.org/abs/2204.13599;
- LISTA: https://arxiv.org/abs/1808.10038;
- generalisation of unfolded networks: https://arxiv.org/abs/2010.15658.

**Mukherjee, Hauptmann, Öktem, Pereyra, Schönlieb 2023, "Learned reconstruction methods with convergence guarantees"** [V] (`mukherjee2023`), https://arxiv.org/abs/2206.05431.
- Organised by notion of convergence: stability, fixed-point, objective, convergent regularisation.
- Covers AR/ACR, NETT, null-space networks, DEQ/unrolling, PnP (including linear denoisers, Nair–Gavaskar–Chaudhury https://arxiv.org/abs/2104.10348), Bayesian PnP-ULA, and plug-in VAMP.
- Extra pointers:
  - Jagatap & Hegde (PGD with untrained-network priors under S-REC): https://arxiv.org/abs/1906.08763;
  - Reehorst & Schniter (RED): https://arxiv.org/abs/1806.02296;
  - Holden–Pereyra–Zygalakis: https://arxiv.org/abs/2103.10182;
  - Alaifari–Alberti–Gauksson (localized adversarial artifacts in CS-MRI): https://arxiv.org/abs/2206.05289.
- Neither survey contains a recovery theorem for PnP with a learned denoiser under structured Fourier sampling.

---

## 13. Synthesis: what can be stated today for $(D,A)$

### 13.1 Three proof templates and what each demands of $D$

| template | key step | condition on $A$ | condition on $D$ | examples |
|---|---|---|---|---|
| **(i) optimality-based** | $D(z)$ is an ($\epsilon$-)nearest point of $\Sigma$ to $z$, and $D(z)\in\Sigma$ | RIP / bi-Lipschitz *ratio* on $\Sigma-\Sigma$ (e.g. $b/a<1.5$ or $<2$) | idempotent, nearest-point up to additive $\epsilon$ | Blumensath; Raj; Shah–Hegde; Golbabaee–Davies; MIP; SPIN; Beygi |
| **(ii) Lipschitz-based** | $\Vert \Pi(z)-x_0\Vert \le\beta\Vert z-x_0\Vert $ composed with $\Vert (I-\mu LA)v\Vert \le\delta\Vert v\Vert $ | **full-norm** RIC $\delta$ on $\Sigma-\Sigma$, with $\delta\beta<1$ | maps into $\Sigma$; restricted Lipschitz $\beta$; approximation error $\eta$ | Traonmilin; Joundi; Leong; Liu; Wolf |
| **(iii) tangent-space** | the error is propagated by $P_{T}(I-\mu A^HA)P_{T'}$ plus curvature | RIP on sums of local tangent spaces | acts like a retraction / projection onto the local tangent space; curvature (reach) bounded | Wei et al. (low-rank); IHT-style; Iwen–Maggioni |

### 13.2 [D] Template (ii) is provably restrictive at high undersampling

Let $B=\mu LA$, with $\mathrm{rank}\,B\le r=\mathrm{rank}A$. Since $Bv\in\mathrm{range}(B)$, every unit $v$ satisfies $\|(I-B)v\|\ge\|P_{\mathrm{range}(B)^\perp}v\|$. Hence
$$\delta_\Sigma(\mu LA)\ \ge\ \sup_{v\in\Sigma-\Sigma}\ \mathrm{dist}\big(v/\|v\|,\ \mathrm{range}(L)\big).$$
If $\Sigma-\Sigma$ contains a unit-norm tight frame of $\mathbb C^n$, then averaging $\|P_Rv\|^2$ over the frame gives $\delta\ge\sqrt{1-r/n}$ for **every** $L$ and $\mu$. More generally, $\delta^2\ge1-$ (the energy fraction of the secant distribution captured by the best $r$-dimensional subspace).

So $\delta\beta<1$ forces $\beta<(1-r/n)^{-1/2}\approx1+\frac{r}{2n}$.
- **Orthogonal projections onto convex sets** ($\beta=1$) pass.
- **Hard thresholding** ($\beta\approx1.618$) needs $r/n>0.62$.
- **A learned denoiser of a nonconvex image class**, whose global $\beta$ is presumably between 1 and 2, needs $\beta$ within $O(r/n)$ of 1.

Template (i) avoids this because nearest-point optimality removes the out-of-model component. Template (iii) avoids it because the tangent projection removes it. Liu et al.'s Thm 1–2 fall in template (ii); see §6.1 for why they are vacuous at $m<n$.

**Multi-coil.** If $r=n$ (enough coils, $A$ injective), the bound gives nothing. $\delta$ is then controlled by the conditioning of $LA$ on secants. A preconditioner $L=PA^H$ with $PA^HA\approx I$ lowers $\delta$, but raises $\frac{\mu}{1-\delta\beta}\|PA^He\|$, which is the g-factor noise. Joundi Thm 3.1 is the clean statement of this trade-off.

### 13.3 [D] A local, deterministic statement that holds for our iteration as it stands

**Proposition L (elementary; Banach fixed point).** Let $T(x)=D(x-\alpha PA^H(Ax-y))$ with $y=Ax_0+e$, and $D\in C^1$. Let $B=\bar B(x_0,\rho)$ and $G(x)=x-\alpha PA^H(Ax-y)$. Define
$$q=\sup_{\xi\in G(B)}\big\|J_D(\xi)\,(I-\alpha PA^HA)\big\|,\qquad t=\|D(x_0+\alpha PA^He)-x_0\| .$$
If $q<1$ and $t\le(1-q)\rho$, then:
- $T$ maps $B$ into itself and has a unique fixed point $x^\ast \in B$;
- PnP converges linearly from any $x_0'\in B$, with $\|x_k-x^\ast \|\le q^k\|x_0'-x^\ast \|$;
- the fixed point satisfies
$$\|x^\ast -x_0\|\le\frac{t}{1-q}\le\frac{\|D(x_0)-x_0\|+\sup_B\|J_D\|\,\alpha\|PA^He\|}{1-q}.$$

*Proof.*
1. By the mean value theorem along segments, $T(x)-T(x')=\int_0^1J_D(\cdot)\,dt\,(I-\alpha PA^HA)(x-x')$, and $G(B)$ is convex. So $T$ is $q$-Lipschitz on $B$.
2. For $x\in B$: $\|T(x)-x_0\|\le q\rho+t\le\rho$, so $T(B)\subseteq B$.
3. Banach gives the unique fixed point and linear convergence.
4. $\|x^\ast -x_0\|\le q\|x^\ast -x_0\|+t$. $\square$

**Interpretation.**
- $q<1$ requires $J_D$ to be **strictly contractive on directions that $I-\alpha PA^HA$ does not shrink**. These are the null-space and near-null-space directions of $A$, *evaluated at the aliased inputs the denoiser actually sees* ($\xi\in G(B)$).
- At the same time, $\|D(x_0)-x_0\|$ small requires $J_D$ to pass $x_0$'s own local (tangent) directions.
- If those directions overlap the near-null space of $A$, both cannot hold. This is the Gottschling et al. trade-off, in linearised form.
- In linearised form the hallucination is $(I-J(I-\alpha PA^HA))^{-1}(D(x_0)-x_0)$. Its null-space components are amplified by $1/(1-\lambda_k)$, where $\lambda_k$ is the Jacobian eigenvalue in that direction.

**Sharper versions.** The ball-based $q$ is crude: it includes cross-terms $J_D\,PA^HA\,P_{T^\perp}$ for directions the iterates never take. Restricting $x-x'$ to differences of denoiser outputs, as Liu does with $\mathrm{Im}(D)$, and assuming $J_D\approx P_{T}$ (template (iii)) would replace $q$ by $\|P_T(I-\alpha PA^HA)P_{T'}\|+\|J_D-P_T\|+\text{curvature}$. The first term is a tangent-space RIP constant. That version needs the assumptions of §14, gap 2, and is not proven for denoisers.

**Measurable quantities.** In the P-weighted norm $\|v\|_{P^{-1}}$, $I-\alpha PA^HA$ is self-adjoint with norm $\le1$ for $\alpha\le2/\lambda_{\max}(P^{1/2}A^HAP^{1/2})$, which is the natural metric to use.
- $q$: power iteration on $v\mapsto J_D(\xi)(I-\alpha PA^HA)v$ (JVP + VJP) at sampled $\xi$, such as the actual iterates and the aliased inputs of phantoms. This gives lower estimates of the sup. A certificate would need a Lipschitz bound on $J_D$, i.e. second-order information.
- $\|D(x_0)-x_0\|$: on phantoms with ground truth. This is "truth distance to $\mathrm{Fix}(D)$".
- Noise term: by Monte Carlo over coil-whitened noise.

### 13.4 What a probabilistic theorem would need

The randomness lives in the sampling pattern $\Omega$, e.g. random spiral rotations or interleaf subsets. The Adcock–Cardenas–Dexter framework (§4.4) is the natural host:
- vector-valued block measurements: one interleaf times all coils;
- arbitrary per-sample distributions;
- deterministic samples free of charge, e.g. the k-space centre.

**What it would require:**
- (a) A model class $U$ with computable covering numbers and *variation* (coil-aware block coherence). Candidates: $\mathrm{Fix}(D)\cap$ bounded set, or the union of local tangent spaces $\bigcup_x\mathrm{range}\,J_D(x)$ truncated at eigenvalue $\ge\tau$.
- (b) A fitting step that is close to the empirical least-squares minimiser over $U$, or a template-(iii) analysis.
- (c) The density weighting $1/\pi_i$ inside the fit. In Joundi's framework this can sit in the back-projection $L=A^HW$. Its noise cost is $\|Le\|$, and that is exactly the trade-off behind our no-DCW rule.

---

## 14. Gap analysis

What is missing for a statement of the form "if $(D,A)$ satisfy X, then PnP recovers $x_0$ within … with probability …"?

1. **No sampling theorem for structured non-Cartesian multi-coil Fourier on a learned model set.** *(Top gap.)*
   - Rigorous Fourier results exist only for explicit ReLU-generator ranges (finite unions of cones; Berk 2022/2023, Plan et al. 2025/26, Adcock et al.), sparse or wavelet models (AHPR, Krahmer–Ward, Colbrook), and Cartesian on-grid sampling with replacement or Bernoulli.
   - They all use **density-weighted** least squares and **idealised** estimators.
   - Not covered:
     - off-grid trajectories (Chauffert et al.: no CS theorem for continuous curves);
     - correlated interleaf blocks (only Cartesian blocks are covered);
     - shared-trajectory coil sensitivities (only a formulation with independent per-coil masks exists);
     - complex-valued images and phase;
     - the unweighted fit we actually use.
   - Even for Gaussian $A$, no result gives sample complexity in terms of a *denoiser* (as opposed to a generator). The closest is D-AMP's conditional "$m/n>\kappa$" (§6.6).

2. **No theorem that a trained denoiser satisfies the hypotheses of any recovery template.** *(Top gap.)*
   - **(i)** needs $D$ idempotent and $\epsilon$-nearest-point. MMSE-trained denoisers are neither (§9.8).
   - **(ii)** needs a restricted Lipschitz $\beta$ within $O(r/n)$ of 1 under strong undersampling (§13.2), plus a uniform additive approximation error. That error is impossible for scale-equivariant $D$ (§9.8).
   - **(iii)**, the only template that truly handles a moving basis, is proven only for low-rank matrices (Wei et al.). It would need, for the denoiser:
     - $J_D(\xi)\approx P_{T(x_0)}$ for inputs $\xi$ at the aliasing level;
     - $x_0$ (approximately) in $T(x_0)$;
     - a curvature/reach bound for $\mathrm{Fix}(D)$;
     - a tangent-space RIP of $A$.
   - The empirical facts that would motivate (iii) are not theorems: approximate symmetry and projection-likeness of $J_D$, and $d(\sigma)=\|J_D\|_F^2\propto1/\sigma$ with nested subspaces (Mohan; Kadkhodaie et al.).
   - **[D] Consequence.** As the PnP error shrinks, the denoiser's effective dimension grows. Recovery should therefore stall where $d(\sigma_{\rm eff})\cdot\log(\cdot)\cdot(\text{coherence})$ exceeds the measurement budget. This is a heuristic error-floor prediction worth testing, not a theorem.
   - The **implicit model set is uncharacterised**: whether $\mathrm{Fix}(D)$ is closed, low-dimensional, of positive reach, or convex (it is convex for demicontractive $D$, RED-PRO Thm 3.8), and how it relates to the training distribution. Wolf et al. quantify only memorisation: $\mathrm{dist}(x,\text{training set})$.

3. **Verifiability.** *(Top gap.)*
   - Every condition is a supremum over an infinite set: S-REC on $\mathrm{Im}(D)$, RIC on $\Sigma-\Sigma$, $\beta$, $\eta$, $\kappa(BA)$ on cones, and the $q$ of Proposition L.
   - Sampling gives only *one-sided* estimates. Liu et al.'s sampled $\mu=0.093$ for radial MRI is contradicted by the surjectivity argument (§6.1).
   - No paper certifies any of these for a trained CNN. The only certified-by-construction cases are architectural:
     - ICNN-gradient (LPN) gives an exact prox;
     - $D=I-\nabla g$ with a $\mathrm{Lip}<1$ penalty, which is verified only on samples;
     - firmly-nonexpansive training, also verified only on samples.
   - Certification would need Lipschitz or second-order bounds on $J_D$, which are not available for deep CNNs at useful tightness.

4. **Contraction-type convergence theory does not fit our denoiser.**
   - Normalization equivariance gives $J\mathbf 1=\mathbf 1$, so $\mathrm{Lip}\ge1$. This rules out Ryu with $\mu=0$, the Ebner–Haltmeier admissible families, Shoushtari $\lambda<1$, and the contraction arguments in DEQ.
   - A non-symmetric Jacobian rules out every prox or objective interpretation (Hurault, Xu, Fang, Pritchard–Parhi).
   - Hence even *convergence* of our exact iteration is unproven for underdetermined $A$. P²nP Thm 1 needs $A^HA\succ0$.

5. **Null-space selection is the denoiser's choice, and no theorem says it chooses the truth.**
   - Regularization theory proves convergence to the regulariser-preferred solution (Ebner–Haltmeier Thm 3.15, NETT, ACR, null-space networks).
   - The troublesome kernel proves that accuracy on near-kernel-separated pairs forces instability or hallucination (Thm 4.1, 4.5), and that data consistency does not help.
   - The only positive results tie stability to injectivity of $A$ on the model set, $(\mathcal M-\mathcal M)\cap N(A)=\{0\}$ with an rNSP-type margin. That is exactly the unverified condition of gaps 1–3.

6. **Noise vs density compensation.**
   - Variable-density theorems need the $1/p_j$ weighting to make the operator isotropic. Their noise terms carry the weights: $\|\tilde D\eta\|$ (Berk) and the weighted constraint (Krahmer–Ward, who note uniform noise works better empirically).
   - No theorem covers the unweighted fit with image-domain preconditioning that we use.
   - Joundi Thm 3.1 shows this is a genuine rate-vs-noise trade-off through $L$, not just a missing proof.

7. **Model mismatch beyond distance terms.**
   - Mismatch enters every bound as $d(x_0,\Sigma)$, possibly amplified by $\|A\|$ or $\sqrt{n/m}$ (Bora/Berk $\|Ax^\perp\|$, Joundi $C_{rob}$).
   - Distributional robustness is proven only for Gaussian $A$ (Jalal). Jalal et al. note it is false in general for Fourier.
   - Phantom-to-patient shift therefore has no theoretical cover.

8. **Posterior sampling / diffusion.**
   - Guarantees are either consistency-type (Xu–Chi, Laumont), stationarity in Fisher information (PMC, PnP-DM), or restricted to special priors (Rout, Montanari–Wu, ReGuidance).
   - Worst-case posterior sampling is cryptographically hard, even for coordinate subsampling (Gupta et al.).
   - The factor-2 optimality (Jalal Thm 3.4) holds for any $A$, but only for the *exact, correct* posterior.

**Top 3 gaps:** 1, 2, 3.

**Most promising route [D].** A template-(iii) theorem for "denoiser as soft local projection":
- Inputs: Proposition L with $J_D\approx$ tangent projection, plus a coil-aware block-coherence RIP of $A$ on unions of tangent spaces in the Adcock–Cardenas–Dexter framework.
- Measured quantities: (Hutchinson) $\mathrm{tr}J_D$ and $\|J_D\|_F^2$ as $k$; the angles between the top-$k$ eigenspace of $J_D$ and $\ker A$ / the sampled k-space; $\|D(x_0)-x_0\|$; and the local contraction $q$ on actual iterates.

---

## Sources

All arXiv links are abstract pages. Local copies are under `data/hasty/external/papers_learned_cs/<dir>/`. Empty or duplicate directories: `efron2011` and `gribonval2013` (downloads failed); `physicspnp2026` duplicates `physmatters2026`.

**Generative priors (Gaussian-type)**
- Bora, Jalal, Price, Dimakis 2017 — https://arxiv.org/abs/1703.03208
- Hand & Voroninski 2017 — https://arxiv.org/abs/1705.07576
- Huang, Hand, Heckel, Voroninski 2018 — https://arxiv.org/abs/1812.04176
- Shah & Hegde 2018 — https://arxiv.org/abs/1802.08406
- Raj, Li, Bresler 2019 — https://arxiv.org/abs/1902.09698
- Peng, Jalali, Yuan 2020 — https://arxiv.org/abs/1901.05045
- Beygi, Jalali, Maleki, Mitra 2017 — https://arxiv.org/abs/1704.01992
- Kamath, Karmalkar, Price 2020 — https://arxiv.org/abs/1912.02938
- Liu & Scarlett 2020 — https://arxiv.org/abs/1908.10744
- Jalal, Karmalkar, Dimakis, Price 2021 — https://arxiv.org/abs/2106.11438
- Heckel & Soltanolkotabi 2020 — https://arxiv.org/abs/2005.03991

**Fourier / variable density / structured sampling**
- Berk, Brugiapaglia, Joshi, Plan, Scott, Yılmaz 2022 — https://arxiv.org/abs/2207.09340
- Berk, Brugiapaglia, Plan, Scott, Sheng, Yılmaz 2023 — https://arxiv.org/abs/2310.04984
- Plan, Scott, Sheng, Yılmaz 2025 — https://arxiv.org/abs/2504.01046
- Plan, Scott, Yılmaz 2026 — https://arxiv.org/abs/2604.04802
- Adcock, Cardenas, Dexter 2023/2026 — https://arxiv.org/abs/2311.14886
- Adcock, Cardenas, Dexter 2023 (CS4ML) — https://arxiv.org/abs/2306.00945
- DeLise & Dexter 2026 — https://arxiv.org/abs/2605.05435
- Jalal, Arvinte, Daras, Price, Dimakis, Tamir 2021 — https://arxiv.org/abs/2108.01368
- Adcock, Hansen, Poon, Roman 2017 — https://arxiv.org/abs/1302.0561
- Krahmer & Ward 2014 — https://arxiv.org/abs/1210.2380
- Bigot, Boyer, Weiss 2016 — https://arxiv.org/abs/1305.4446
- Boyer, Bigot, Weiss 2019 — https://arxiv.org/abs/1505.01619
- Chauffert, Ciuciu, Kahn, Weiss 2014 — https://arxiv.org/abs/1311.6039
- Oymak, Recht, Soltanolkotabi 2018 — https://arxiv.org/abs/1507.04793

**PnP / RED / DEQ / learned regularisers**
- Liu, Asif, Wohlberg, Kamilov 2021 — https://arxiv.org/abs/2106.03668
- Traonmilin, Aujol, Guennec 2024/2026 — https://arxiv.org/abs/2410.06607
- Joundi, Traonmilin, Aujol 2025 — https://arxiv.org/abs/2512.07397
- Leong & Traonmilin 2025 — https://arxiv.org/abs/2509.20511
- Wolf, Fadili, Guo, He 2026 — https://arxiv.org/abs/2607.29525
- Metzler, Maleki, Baraniuk 2016 — https://arxiv.org/abs/1406.4175
- Berthier, Montanari, Nguyen 2020 [S] — https://arxiv.org/abs/1708.03950
- Fletcher, Rangan, Sarkar, Schniter 2018 — https://arxiv.org/abs/1806.10466
- Colbrook, Antun, Hansen 2022 — https://arxiv.org/abs/2101.08286
- Neyra-Nesterenko & Adcock 2023 — https://arxiv.org/abs/2203.00804
- Shoushtari, Liu, Kamilov 2022 — https://arxiv.org/abs/2207.13200
- Shoushtari, Liu, Chandler, Asif, Kamilov 2023 — https://arxiv.org/abs/2310.00133
- Ryu, Liu, Wang, Chen, Wang, Yin 2019 — https://arxiv.org/abs/1905.05406
- Gilton, Ongie, Willett 2021 — https://arxiv.org/abs/2102.07944
- Cohen, Elad, Milanfar 2021 (RED-PRO) — https://arxiv.org/abs/2008.00226
- Hurault, Leclaire, Papadakis 2022 (GS) — https://arxiv.org/abs/2110.03220
- Hurault, Leclaire, Papadakis 2022 (Prox) — https://arxiv.org/abs/2201.13256
- Xu, Sun, Liu, Wohlberg, Kamilov 2020 — https://arxiv.org/abs/2005.07685
- Pritchard & Parhi 2025 — https://arxiv.org/abs/2510.27211
- Gribonval 2011 — https://hal.inria.fr/inria-00486840
- Gribonval & Nikolova 2020 — https://arxiv.org/abs/1807.04014
- Pesquet, Repetti, Terris, Wiaux 2021 — https://arxiv.org/abs/2012.13247
- Chan, Wang, Elgendy 2017 — https://arxiv.org/abs/1605.01710
- Cohen, Blau, Freedman, Rivlin 2021 [A] — https://proceedings.neurips.cc/paper_files/paper/2021/hash/97108695bd93b6be52fa0334874c8722-Abstract.html
- Fang, Buchanan, Sulam 2024 — https://arxiv.org/abs/2310.14344
- Goujon, Neumayer, Bohra, Unser 2023 — https://arxiv.org/abs/2211.12461
- Goujon, Neumayer, Unser 2024 — https://arxiv.org/abs/2308.10542
- Hong, Xu, Hu, Fessler 2024 (P²nP) — https://arxiv.org/abs/2405.03854
- Ebner & Haltmeier 2022 — https://arxiv.org/abs/2212.06881
- Hauptmann, Mukherjee, Schönlieb, Sherry 2024 — https://arxiv.org/abs/2307.09441
- Li, Schwab, Antholzer, Haltmeier 2020 (NETT) — https://arxiv.org/abs/1803.00092
- Schwab, Antholzer, Haltmeier 2019 — https://arxiv.org/abs/1806.06137
- Mukherjee, Dittmer, Shumaylov, Lunz, Öktem, Schönlieb (ACR) — https://arxiv.org/abs/2008.02839
- Lunz, Öktem, Schönlieb 2018 — https://arxiv.org/abs/1805.11572
- Gavaskar, Athalye, Chaudhury 2022 [S] — https://arxiv.org/abs/2207.13031
- Nair, Gavaskar, Chaudhury 2021 — https://arxiv.org/abs/2104.10348

**Manifold / UoS / model-based**
- Baraniuk & Wakin 2009 [S] — https://doi.org/10.1007/s10208-007-9011-z
- Wakin 2010 — https://arxiv.org/abs/1002.1247
- Eftekhari & Wakin 2015 — https://arxiv.org/abs/1306.4748
- Iwen & Maggioni 2013 — https://arxiv.org/abs/1204.3337
- Blumensath 2011 — https://arxiv.org/abs/0911.3514
- Blumensath & Davies 2009 — IEEE TIT 55(4):1872–1882, https://doi.org/10.1109/TIT.2009.2013003
- Baraniuk, Cevher, Duarte, Hegde 2010 — https://arxiv.org/abs/0808.3572
- Hegde, Indyk, Schmidt 2015 — https://arxiv.org/abs/1406.1579
- Shah & Chandrasekaran 2011 — http://people.lids.mit.edu/pari/manifold.pdf
- Hegde & Baraniuk 2012 — https://arxiv.org/abs/1202.1595
- Golbabaee & Davies 2018 — https://arxiv.org/abs/1706.00092
- Wei, Cai, Chan, Leung 2016 — https://arxiv.org/abs/1511.01562
- Renna, Calderbank, Carin, Rodrigues 2014 — https://arxiv.org/abs/1307.0861
- Yu & Sapiro 2011 — https://arxiv.org/abs/1010.4314
- Chen, Silva, Paisley, Wang, Dunson, Carin 2010 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3721352/

**Denoiser geometry**
- Kadkhodaie & Simoncelli 2021 — https://arxiv.org/abs/2007.13640
- Kadkhodaie, Guth, Simoncelli, Mallat 2024 — https://arxiv.org/abs/2310.02557
- Mohan, Kadkhodaie, Simoncelli, Fernandez-Granda 2020 — https://arxiv.org/abs/1906.05478
- Manor & Michaeli 2024 — https://arxiv.org/abs/2309.13598
- Meng, Song, Li, Ermon 2021 — https://arxiv.org/abs/2111.04726
- Efron 2011 [S] — https://doi.org/10.1198/jasa.2011.tm11181
- Stanczuk, Batzolis, Deveney, Schönlieb 2024 — https://arxiv.org/abs/2212.12611
- Pidstrigach 2022 — https://arxiv.org/abs/2206.01018
- Herbreteau, Moebel, Kervrann 2023 — https://arxiv.org/abs/2306.05037

**Negative / stability**
- Antun, Renna, Poon, Adcock, Hansen 2020 — https://arxiv.org/abs/1902.05300
- Gottschling, Antun, Hansen, Adcock 2025 — https://arxiv.org/abs/2001.01258
- Gottschling, Campodonico, Antun, Hansen (kersize) — https://arxiv.org/abs/2311.16898
- Bhadra, Kelkar, Brooks, Anastasio 2021 — https://arxiv.org/abs/2012.00646
- Gupta, Jalal, Parulekar, Price, Xun 2024 — https://arxiv.org/abs/2402.12727

**Diffusion / score priors**
- Xu & Chi 2024 — https://arxiv.org/abs/2403.17042
- Bruna & Han 2024 — https://arxiv.org/abs/2407.00745
- Sun, Wu, Chen, Feng, Bouman 2023 — https://arxiv.org/abs/2310.10835
- Wu, Sun, Chen, Zhang, Yue, Bouman 2024 — https://arxiv.org/abs/2405.18782
- Laumont et al. 2022 (PnP-ULA) — https://arxiv.org/abs/2103.04715
- Laumont et al. 2023 (PnP-SGD) — https://arxiv.org/abs/2201.06133
- Rout et al. 2023 — https://arxiv.org/abs/2307.00619
- Chung et al. 2022 (MCG) — https://arxiv.org/abs/2206.00941
- Chung et al. 2023 (DPS) — https://arxiv.org/abs/2209.14687
- Montanari & Wu 2023 — https://arxiv.org/abs/2304.11449
- Karan, Shah, Chen 2025 — https://arxiv.org/abs/2506.10955
- Chang et al. 2025 — https://arxiv.org/abs/2512.08022
- Ma, Wu, Liu 2026 [A] — https://arxiv.org/abs/2610.11834
- Pesme et al. 2025 [A] — https://arxiv.org/abs/2507.15397

**Surveys**
- Scarlett, Heckel, Rodrigues, Hand, Eldar 2022/23 — https://arxiv.org/abs/2206.14373
- Mukherjee, Hauptmann, Öktem, Pereyra, Schönlieb 2023 — https://arxiv.org/abs/2206.05431
