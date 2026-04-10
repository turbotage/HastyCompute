# ✅ Final Compression Framework for MRI Reconstruction

This document summarizes the **compression framework** for incorporating **off-resonance** and **gradient nonlinearities** into MRI reconstruction while preserving the **Toeplitz structure** of the normal operator. The formulation emphasizes the **identity–residual decomposition** and a **low-rank separable approximation**.

---

## ✅ 1. Signal Model

The MRI signal for coil ( k ) at time ( t_j ) is defined as:

$$
s_k(t_j) = \sum_{i}
W(t_j,\mathbf{x}_i),
c_k(\mathbf{x}_i),
\rho(\mathbf{x}_i),
e^{-i \mathbf{k}(t_j) \cdot \mathbf{x}_i},
$$

where:

* $ \rho(\mathbf{x}) $: object magnetization,
* $ c_k(\mathbf{x})  $: coil sensitivity,
* $ \mathbf{k}(t_j)  $: k-space trajectory,
* $ W(t,\mathbf{x})  $: multiplicative encoding term capturing **off-resonance**, **relaxation**, and **gradient nonlinearities**.

---

## ✅ 2. Definition of the ( W ) Matrix

The modulation term is defined as:

$$
W(t_j,\mathbf{x}) =
\exp\left[-z(\mathbf{x}) t_j\right]
\prod_{\beta \in {x,y,z}}
\exp\left[
-i\gamma \sum_{l,m}
\left(\alpha_{\beta}^{lm} * G_\beta\right)(t_j)
Y_{lm}(\mathbf{x})
\right],
$$

where:

* $z(\mathbf{x}) = R_2^*(\mathbf{x}) + i\Delta\omega(\mathbf{x})$,
* $Y_{lm}(\mathbf{x})$: spherical harmonics describing spatial gradient deviations,
* $(\alpha_{\beta}^{lm} * G_\beta)(t)$: time-dependent coefficients obtained from the **Gradient Impulse Response Function (GIRF)** or system calibration,
* $ \gamma $: gyromagnetic ratio.

This formulation captures both **temporal** and **spatial** dependencies in a multiplicative encoding.

---

## ✅ 3. Forward Operator

Define the encoding matrix:

$$
A_{ij} = W(t_i,\mathbf{x}_j),
e^{-i \mathbf{k}(t_i)\cdot \mathbf{x}_j},
$$

so that the forward model becomes:

$$
\mathbf{s} = A\boldsymbol{\rho}.
$$

---

## ✅ 4. Normal Operator and Toeplitz Structure

The normal operator is given by:

$$
\begin{aligned}
(A^{H}A)*{ij}
&= \sum*{k} A^{*}*{k i} A*{k j} \
&= \sum_{k}
W^{*}(t_k,\mathbf{x}_i),
W(t_k,\mathbf{x}_j),
e^{-i \mathbf{k}(t_k)\cdot (\mathbf{x}_j - \mathbf{x}_i)}.
\end{aligned}
$$

If the term $ W^{*}(t_k,\mathbf{x}_i)W(t_k,\mathbf{x}_j) $ can be expressed in a **separable form**, the exponential depends only on the spatial difference $\mathbf{x}_j - \mathbf{x}_i$, enabling a **Toeplitz structure** that can be efficiently applied using FFTs.

---

## ✅ 5. Identity–Residual Decomposition of ( W )

To ensure that the approximation is never worse than ignoring corrections, decompose $W$ as:

$$
W(t,\mathbf{x}) = 1 + \widehat{W}(t,\mathbf{x}),
$$

where:

* The **identity term** represents the uncorrected Fourier encoding.
* $\widehat{W}(t,\mathbf{x})$ captures deviations due to off-resonance and gradient nonlinearities.

---

## ✅ 6. Low-Rank Separable Approximation of the Residual

Approximate the residual using a rank-( R ) separable expansion:

$$
\widehat{W}(t,\mathbf{x})
\approx
\sum_{r=1}^{R}
\Gamma_r(t),\Omega_r(\mathbf{x}),
$$

so that

$$
W(t,\mathbf{x})
\approx
1 + \sum_{r=1}^{R}
\Gamma_r(t),\Omega_r(\mathbf{x}).
$$

This approximation can be obtained using **randomized SVD**, **Lanczos**, or other matrix-free low-rank methods requiring only operator evaluations.

---

## ✅ 7. Expansion of ( W^{*}W )

Using the residual decomposition:

$$
\begin{aligned}
W^{*}(t,\mathbf{x}_i) W(t,\mathbf{x}*j)
&=
\left(1 + \sum*{r} \Gamma_r^{*}(t)\Omega_r^{*}(\mathbf{x}*i)\right)
\left(1 + \sum*{s} \Gamma_s(t)\Omega_s(\mathbf{x}_j)\right) \
&=
1

* \sum_{r} \Gamma_r^{*}(t)\Omega_r^{*}(\mathbf{x}_i)
* \sum_{s} \Gamma_s(t)\Omega_s(\mathbf{x}_j) \
  &\quad
* \sum_{r,s} \Gamma_r^{*}(t)\Gamma_s(t)
  \Omega_r^{*}(\mathbf{x}_i)\Omega_s(\mathbf{x}_j).
  \end{aligned}
  $$

To maintain a compact representation, approximate this expression with a **symmetric separable form**:

$$
W^{*}(t,\mathbf{x}_i) W(t,\mathbf{x}*j)
\approx
\sum*{p=0}^{P}
A_p^{*}(\mathbf{x}_i),
B_p(t),
A_p(\mathbf{x}_j),
$$

where:

* $A_0(\mathbf{x}) = 1$, $B_0(t) = 1$ (identity term),
* The remaining $P$ terms approximate the residual contributions,
* Typically, $P \ll R^2$.

This compression can be obtained via an **eigendecomposition or SVD** of the temporal Gram matrix:

$$
C_{rs}(t) = \Gamma_r^{*}(t)\Gamma_s(t).
$$

---

## ✅ 8. Toeplitz Representation of the Normal Operator

Substituting the separable form into the normal operator:

$$
\begin{aligned}
(A^{H}A)*{ij}
&\approx
\sum*{p=0}^{P}
A_p^{*}(\mathbf{x}*i)
\left[
\sum*{k} B_p(t_k)
e^{-i \mathbf{k}(t_k)\cdot (\mathbf{x}_j - \mathbf{x}_i)}
\right]
A_p(\mathbf{x}_j).
\end{aligned}
$$

Define the **Toeplitz kernel**:

$$
T_p(\Delta \mathbf{x}) =
\sum_{k} B_p(t_k)
e^{-i \mathbf{k}(t_k)\cdot \Delta \mathbf{x}},
\qquad
\Delta \mathbf{x} = \mathbf{x}_j - \mathbf{x}_i.
$$

The normal operator can then be written compactly as:

$$
A^{H}A \approx \sum_{p=0}^{P} A_p^{H}, T_p, A_p,
$$

where:

* $A_p = \operatorname{diag}(A_p(\mathbf{x}))$ are **diagonal matrices**,
* $T_p$ are **Toeplitz matrices**, each efficiently applied using **two FFTs** via circulant embedding.

---

## ✅ 9. Computational Advantages

| Component | Operation                          | Cost                        |
| --------- | ---------------------------------- | --------------------------- |
|   $A_p$   | Diagonal multiplication            |  $\mathcal{O}(N)$           |
|   $T_p$   | Toeplitz convolution               |  $\mathcal{O}(N \log N)$    |
| **Total** | $\sum_{p=0}^{P} A_p^{H} T_p A_p$   |  $\mathcal{O}(P N \log N)$  |

This approach avoids:

* Storing the full matrix $W(t,\mathbf{x})$,
* Performing $R^2$ NUFFT operations,
* Explicitly forming the large normal operator.

---

## ✅ 10. Interpretation

### 🔹 Identity Term

* Corresponds to the standard MRI normal operator without corrections.
* Guarantees that using $P = 0$ reproduces the conventional reconstruction.
* Ensures that additional terms only **improve** the model.

### 🔹 Residual Terms

* Capture spatially varying off-resonance and gradient nonlinearities.
* Require only a **small number of additional Toeplitz convolutions**.

### 🔹 Applicability

* Works for **non-Cartesian trajectories** (radial, spiral, yarnball, etc.).
* Compatible with **compressed sensing** and **iterative reconstruction**.
* Particularly beneficial for **low-field MRI**, where such corrections are significant.

---

## ✅ 11. Summary of the Final Compression Scheme

1. **Define the modulation**
   $$
   W(t,\mathbf{x}) =
   \exp[-z(\mathbf{x})t]
   \prod_{\beta \in {x,y,z}}
   \exp\left[
   -i\gamma \sum_{l,m}
   (\alpha_{\beta}^{lm} * G_\beta)(t)
   Y_{lm}(\mathbf{x})
   \right].
   $$

2. **Residual decomposition**
   $$
   W(t,\mathbf{x}) =
   1 + \sum_{r=1}^{R} \Gamma_r(t)\Omega_r(\mathbf{x}).
   $$

3. **Separable approximation of ( W^{*}W )**
   $$
   W^{*}(t,\mathbf{x}_i) W(t,\mathbf{x}*j)
   \approx
   \sum*{p=0}^{P} A_p^{*}(\mathbf{x}_i) B_p(t) A_p(\mathbf{x}_j).
   $$

4. **Toeplitz normal operator**
   $$
   A^{H}A \approx \sum_{p=0}^{P} A_p^{H} T_p A_p,
   \qquad
   T_p(\Delta \mathbf{x}) =
   \sum_{k} B_p(t_k) e^{-i\mathbf{k}(t_k)\cdot \Delta \mathbf{x}}.
   $$

5. **Efficient computation**

   * Each $T_p$ is applied using two FFTs.
   * Only $P \ll R^2$ terms are required.
   * The identity term ensures baseline accuracy.

---

## ✅ Final Insight

This framework provides a **principled and computationally efficient** method for incorporating **off-resonance** and **gradient nonlinearities** into MRI reconstruction while preserving the **Toeplitz structure** essential for fast iterative solvers. The **identity–residual decomposition** guarantees that even aggressive compression does not degrade the reconstruction relative to the uncorrected model, making the approach both **robust and scalable**.

