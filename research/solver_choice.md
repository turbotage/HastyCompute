# A reproducible estimator: which solver, and what must be fixed first (2026-11-24)

User question (2026-11-24): two identical gated runs ended at pc RMSE 0.262 and 0.173 m/s (held-out 0.0482 vs
0.0476). Research what is likely the best solver, so the reconstruction is a well-defined estimator before any more
comparisons or cluster runs. Builds on `convergence_strategy.md` (2026-11-17, toys and P1–P2), the
`pnp_restricted` paper (prop:estimator, thm:displacement, prop:keepfp) and a literature review (this session).
Labels: **[code]** read in the repository; **[paper]** stated in a cited paper; **[ours]** our paper or earlier note;
**[D]** derived here; **[M]** measured. No runs were made for this note.

## 1. Why production P²nP is not an estimator [code, ours]

The production loop (`hasty-recon/src/p2np.rs`) is
$$x_{k+1} = D_k\big(x_k - P_k\,\alpha\,\nabla f(x_k)\big),\qquad f(x)=\tfrac12\textstyle\sum_t\|E_t x_t-y_t\|^2 .$$
Four parts change along the trajectory, each on its own enough to make the limit history-dependent:

| part | what the code does | consequence |
|---|---|---|
| metric $P_k$ | ZMSHR1 rank-1 from the last secant pair, rebuilt every iteration | fixed points of $x=D(x-\alpha P\nabla f(x))$ move with $P$ (thm:displacement); $P_k$ depends on the trajectory |
| denoiser $D_k$ | LLR with a new block shift per call (`llr.rs`, cycle spinning, seeded by $k$) | a different operator each iteration: no single fixed point |
| LLR threshold | $\kappa\hat\sigma(x)$, $\hat\sigma$ re-estimated from the iterate per call | $D$ is not the prox of a fixed function: there is no objective |
| air mask | re-estimated every 2 iterations (`--air auto`) | the constraint set changes discretely |

Measured consequence [M]: fp-noise differences (warm-start CG residuals at 1e-3) grow from it ~15; at it 55 the data
term differs by 4 % and the pc RMSE by 35 %, held-out by 1.3 %. The velocity-relevant directions are the ones the data
barely constrain (P1–P2 of `convergence_strategy.md`: after 30 gradients no arm had a velocity slope above 0.57).

## 2. Requirements

- **R1, a fixed objective.** $F(x)=f(x)+\lambda R(x)+\tfrac{\varepsilon}{2}\|x\|^2$ with $R$ fixed for the whole solve:
  - a fixed LLR threshold (`thr_abs`; e.g. $\kappa\hat\sigma$ computed once from the warm start, then frozen);
  - a fixed block structure: one partition, or the *proximal average* over a fixed set of $S$ shifts —
    $\frac1S\sum_s \mathrm{prox}_{g_s}$ is itself the prox of a convex function (the proximal average; Bauschke,
    Goebel, Lucet, Wang 2008), so averaging $S$ shifted SVTs in every call is a fixed prox, at $S\times$ the cost;
  - the air mask estimated once (from the warm start) and frozen;
  - a tiny Tikhonov term $\varepsilon$ (e.g. $10^{-4}$ of the bulk data curvature): $f$ has a large null space and
    nuclear norms are not strictly convex, so without it $\arg\min F$ can be a set, and exact solvers could still
    land at different points [D].
- **R2, a limit independent of the preconditioner**, so k-space equalisation (the circulant, the largest measured
  speed lever) can be used without changing the estimator.
- **R3, a stopping rule that sees velocity.** The total held-out residual ranked solvers opposite to velocity accuracy
  (P1–P2). Stop on the fixed-point residual and on the held-out residual of the encode-difference data, or by
  distance to a long reference solve.
- **R4, a path to learned denoisers**, whose PnP equilibrium must also be well defined.

## 3. Candidates

### 3.1 ADMM with a warm-started inner PCG [paper, D]
$$x^{k+1}\approx\arg\min_x f(x)+\tfrac\rho2\|x-z^k+u^k\|^2\ (\text{$m$ PCG steps}),\quad z^{k+1}=\mathrm{prox}_{\lambda R/\rho}(x^{k+1}+u^k),\quad u^{k+1}=u^k+x^{k+1}-z^{k+1}.$$
- **Fixed point**: for an exact prox it does not depend on $\rho$; it does not depend on the inner preconditioner nor on
  $m\ge1$ either: warm-started CG returns its input exactly when the input already solves the inner system, so the
  fixed-point set is that of exact ADMM [D, literature review]. Convergence with a fixed small $m$ is empirical (Ahmad
  et al. 2020, cardiac cine: 1–4 CG steps, 10–50 outer iterations); guarantees need summable or relative inner errors
  (Eckstein & Bertsekas 1992; Eckstein & Yao 2018).
- **$\rho$**: residual balancing (Wohlberg 2017) or spectral/BB penalty (Xu, Figueiredo, Goldstein 2017), then frozen
  (bounded adaptivity). With a learned denoiser $\rho$ becomes part of the estimator and must stay fixed (Ahmad 2020).
- **Cost**: $m$ normal-operator applications + 1 prox per outer iteration; the circulant goes into the inner PCG where it
  helps without displacing anything.
- **Ours**: `--prior-mode admm` exists (inner PCG, host stacks). The earlier ADMM runs (2026-11-23) did not converge
  (x–z gap growing), but they ran with the adaptive, shifted LLR and the moving air mask, i.e. without R1.
- **Toy** [ours]: 18 / 41 gradients to 1 % / 0.1 % of the minimiser at its best $\rho$, 129 / — with $\rho$ 10× off.

### 3.2 Smoothed objective + nonlinear PCG (the 2026-11-17 recommendation) [ours]
- $F_\gamma = f + \mathrm{env}_\gamma(\lambda R)$ (Moreau envelope, gradient from one prox: `red.rs`), Polak–Ribière+ in the
  fixed circulant (+ prior-aware) metric with a data-exact line search.
- Minimiser metric-independent and within $O(\gamma)$ of the nonsmooth one (toy: 0.03 % at $\gamma=0.1/$bulk).
- Toy: 14 / 22 gradients to 1 % / 0.1 % (best of all), one prox per iteration, 4 stacks.
- Not implemented for frame or temporal sets (`knotncg.rs` exists for knots). Real P1–P2: fastest on held-out,
  slow in velocity with the identity metric, better with the prior-aware metric (not converged at 30).

### 3.3 Metric-consistent proximal quasi-Newton (keep ZMSHR1, take the prox in the SR1 metric) [paper]
- Becker & Fadili 2012; Becker, Fadili, Ochs 2019: for $V=D\pm uu^{*}$ the prox in $V$ is the prox in $D$ at a
  shifted point, with a scalar root found by bisection or semismooth Newton (2–5 prox calls). Limit $=\arg\min F$.
- Applied to non-Cartesian CS-MRI by Hong, Hernandez-Garcia, Fessler (IEEE TCI 2024), faster than APM, PD, ADMM.
- Limitation for us: the base metric $D$ must be diagonal (scalar) for the prox in $D$ to be the usual SVT, so the
  circulant k-space equalisation is lost; 2–5 LLR sweeps per iteration.

### 3.4 PDHG with a k-space (dual) preconditioner [paper]
- Ong, Uecker, Lustig (IEEE TMI 2020); Pock & Chambolle (2011): the preconditioner acts in the dual step only; fixed
  points are the KKT points of the original problem for any positive diagonal dual preconditioner.
- ≈ 10 iterations in 3-D UTE; no LLR or learned prior tested; dual variable = one k-space data copy (10 GB at 1 mm),
  E and Eᴴ separately. Toy [ours]: 34 / 82, 2.4–3.7× behind 3.2.

### 3.5 Fixed-point-preserving accelerations (add-ons)
- FISTA / POGM with restart (O'Donoghue & Candès 2015; Lin & Fessler 2019) and safeguarded Anderson / SuperMann
  (Zhang, O'Donoghue, Boyd 2020; Themelis & Patrinos 2019) keep the fixed points of the Euclidean map
  $T(x)=D(x-\alpha\nabla f(x))$; our prop:keepfp gives the metric-based version for learned denoisers
  ($z_{k+1}=z_k-B_kF(z_k)$, $F=I-T$, any invertible $B_k$).

## 4. Recommendation [D]

1. **First make the objective fixed (R1).** No solver can be reproducible with a moving metric, a moving denoiser, an
   iterate-dependent threshold and a moving mask. This is solver-independent code work:
   frozen threshold, proximal average over a fixed shift set (or one partition), frozen air mask, $\varepsilon$-Tikhonov.
2. **Reference solver: ADMM–PCG** (3.1).
   - Exact LLR prox, no smoothing bias.
   - The circulant used where it is harmless (inner PCG).
   - Fixed point independent of preconditioner and inner iteration count.
   - Same equilibrium family as PnP-ISTA for learned denoisers later (R4).
   - Code exists.
   Use it to produce the reference estimate per configuration (long, converged).
3. **Production-speed candidate: smoothed NCG** (3.2), judged by its distance to the ADMM reference, not by pc at a
   fixed iteration.
4. **Not now**: metric-consistent QN (loses the circulant), PDHG (memory, untested with LLR); keep as later arms.

## 5. How we would verify it (before any comparison of priors or operators)

- **V1 reproducibility**: the same configuration from two perturbed starts (e.g. warm starts differing by fp noise)
  converges to images whose velocity differs by less than a tolerance set in advance (e.g. 0.005 m/s pc RMSE between
  the two runs), not only in f.
- **V2 solver equivalence**: ADMM–PCG and smoothed NCG ($\gamma\to0$) reach the same estimate within the tolerance.
- **V3 stopping**: the fixed-point residual and the encode-difference held-out residual flatten where the distance to
  the reference flattens.
- Only after V1–V3: compare priors and operator changes, one factor at a time, at convergence.

## 6. Open decisions

- The fixed threshold: $\kappa\hat\sigma$ from the warm start, or chosen on held-out k-space once per configuration.
- Shift set: one partition (block artefacts) vs a proximal average over $S$ shifts ($S\times$ LLR cost; $S=2$–$4$).
- The air mask: keep it (frozen) or replace it by the support constraint only.
- $\varepsilon$: relative to the bulk curvature; check it changes the estimate by much less than the tolerance.

## References

- Becker, S., Fadili, J. (2012). A quasi-Newton proximal splitting method. NeurIPS.
- Becker, S., Fadili, J., Ochs, P. (2019). On quasi-Newton forward–backward splitting: proximal calculus and
  convergence. SIAM J. Optim. 29(4), 2445. arXiv:1801.08691.
- Hong, T., Hernandez-Garcia, L., Fessler, J. A. (2024). A complex quasi-Newton proximal method for image
  reconstruction in compressed sensing MRI. IEEE TCI 10, 372. arXiv:2303.02586.
- Hong, T., Xu, X., Hu, J., Fessler, J. A. (2024). Provable preconditioned plug-and-play approach for compressed
  sensing MRI reconstruction. IEEE TCI 10, 1476. arXiv:2405.03854.
- Ong, F., Uecker, M., Lustig, M. (2020). Accelerating non-Cartesian MRI reconstruction convergence using k-space
  preconditioning. IEEE TMI 39(5), 1646. arXiv:1902.09657.
- Pock, T., Chambolle, A. (2011). Diagonal preconditioning for first order primal-dual algorithms. ICCV.
- Ahmad, R. et al. (2020). Plug-and-play methods for magnetic resonance imaging. IEEE SPM. arXiv:1903.08616.
- Eckstein, J., Bertsekas, D. (1992). On the Douglas–Rachford splitting method and the proximal point algorithm.
  Math. Prog. 55. Eckstein, J., Yao, W. (2018). Relative-error approximate versions of Douglas–Rachford splitting.
  Math. Prog. 170.
- Wohlberg, B. (2017). ADMM penalty parameter selection by residual balancing. arXiv:1704.06209.
- Xu, Z., Figueiredo, M., Goldstein, T. (2017). Adaptive ADMM with spectral penalty parameter selection. AISTATS.
- Bauschke, H., Goebel, R., Lucet, Y., Wang, X. (2008). The proximal average: basic theory. SIAM J. Optim. 19(2).
- O'Donoghue, B., Candès, E. (2015). Adaptive restart for accelerated gradient schemes. FoCM 15.
- Lin, C. Y., Fessler, J. A. (2019). Efficient dynamic parallel MRI reconstruction for the low-rank plus sparse model.
  IEEE TCI 5, 17.
- Zhang, J., O'Donoghue, B., Boyd, S. (2020). Globally convergent type-I Anderson acceleration for nonsmooth
  fixed-point iterations. SIAM J. Optim. 30. Themelis, A., Patrinos, P. (2019). SuperMann. IEEE TAC.
- Iyer, S. et al. (2024). Polynomial preconditioning for accelerated MRI reconstruction. SIIMS 17. arXiv:2204.10252.
