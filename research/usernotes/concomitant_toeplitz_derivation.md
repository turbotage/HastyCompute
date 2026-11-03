# Concomitant-field correction inside an existing Toeplitz MRI normal operator

## Goal

We want to correct higher-order concomitant-field effects **without**

- adding extra FFTs,
- adding extra NUFFTs,
- storing additional \(1024^3\) Toeplitz kernels.

Assume gradient nonlinearity and the easy concomitant contributions have already been handled by coordinate transforms, and that off-resonance / \(T_2^*\) is handled by a time-segmented Toeplitz normal operator

\[
H_0
=
\sum_{\ell=1}^{L}
D_\ell^H T_\ell D_\ell,
\]

where \(T_\ell\) is Toeplitz and \(D_\ell\) is the spatial time-segmentation factor.

The aim is to incorporate the remaining quadratic concomitant phase using only cheap local operations around the existing FFT–multiply–IFFT path.

---

## 1. Normal-operator kernel with concomitant phase

For one time-segmentation term \(\ell\), define the baseline Toeplitz kernel

\[
T_\ell(d)
=
\sum_n a_{\ell n} e^{i k_n^T d},
\qquad
d=r-r',
\]

where \(a_{\ell n}\) contains the sequence weights and temporal interpolation coefficient.

Let the residual concomitant phase at sample \(n\) be quadratic:

\[
\psi_n(r)=r^T C_n r,
\]

with symmetric \(C_n\).

The corresponding normal kernel is

\[
K_\ell(r,r')
=
\sum_n
a_{\ell n}
e^{i k_n^T(r-r')}
e^{i[\psi_n(r)-\psi_n(r')]}.
\]

Introduce

\[
d=r-r',
\qquad
m=\frac{r+r'}{2}.
\]

Because \(C_n\) is symmetric,

\[
\boxed{
\psi_n(r)-\psi_n(r')
=
2m^T C_n d.
}
\]

Hence

\[
K_\ell(m,d)
=
\sum_n
a_{\ell n}
e^{i k_n^T d}
e^{i\,2m^T C_n d}.
\]

This is the central structural identity.

---

## 2. First-order concomitant expansion

For sufficiently small residual concomitant phase,

\[
e^{i\,2m^T C_n d}
=
1+i\,2m^T C_n d+O(\phi_c^2).
\]

Therefore

\[
K_\ell
=
T_\ell(d)
+
2i\sum_{p,q}
m_p d_q
S_{\ell,pq}(d)
+
O(\phi_c^2),
\]

where

\[
\boxed{
S_{\ell,pq}(d)
=
\sum_n
a_{\ell n}
(C_n)_{pq}
e^{i k_n^T d}.
}
\]

The only new sequence-dependent object is thus \(S_{\ell,pq}\).

---

## 3. Generate \(S_{\ell,pq}\) from the existing Toeplitz kernel

Suppose the concomitant coefficient can be represented along the acquired trajectory as

\[
(C_n)_{pq}
\approx
c_{pq}(k_n),
\]

with a low-order polynomial or other compact function \(c_{pq}\).

If

\[
c_{pq}(k)
=
\sum_\alpha
c_{pq,\alpha}k^\alpha,
\]

then using

\[
(-i\partial_d)^\alpha e^{ik^T d}
=
k^\alpha e^{ik^T d},
\]

we obtain

\[
\boxed{
S_{\ell,pq}(d)
=
c_{pq}(-i\nabla_d)\,T_\ell(d).
}
\]

Therefore no additional Toeplitz kernel needs to be stored.

---

## 4. Commutator / anticommutator representation

Let \(X_p\) denote multiplication by coordinate \(r_p\).

For a convolution operator \(S\),

\[
[X_q,S]
=
X_qS-SX_q
\]

has kernel

\[
(r_q-r_q')S(r-r')
=
d_q S(d).
\]

Likewise,

\[
\frac12\{X_p,S\}
=
\frac12(X_pS+SX_p)
\]

introduces the midpoint coordinate

\[
m_p=\frac{r_p+r_p'}{2}.
\]

Hence the first-order concomitant correction can be written as

\[
\boxed{
\Delta H_\ell
=
i
\sum_{p,q}
\{X_p,[X_q,S_{\ell,pq}]\}.
}
\]

Thus

\[
\boxed{
H
\approx
\sum_\ell
D_\ell^H
\left[
T_\ell
+
i\sum_{p,q}
\{X_p,[X_q,S_{\ell,pq}]\}
\right]
D_\ell.
}
\]

This is a first-order pseudodifferential representation of the concomitant correction.

---

## 5. Fourier-domain form: no additional FFTs

Let

\[
\tau_\ell(\xi)
=
\mathcal F\{T_\ell\}
\]

be the stored Toeplitz FFT multiplier.

Since

\[
S_{\ell,pq}
=
c_{pq}(-i\nabla)T_\ell,
\]

its Fourier multiplier is

\[
\boxed{
\sigma_{\ell,pq}(\xi)
=
c_{pq}(\xi)\tau_\ell(\xi).
}
\]

No new 3D kernel needs to be stored.

Under Fourier transform,

\[
X_p
\leftrightarrow
i\partial_{\xi_p}.
\]

For

\[
V_\ell(\xi)
=
\mathcal F\{D_\ell x\},
\]

the first-order commutator correction satisfies

\[
\boxed{
\mathcal F
\left\{
i\{X_p,[X_q,S]\}v
\right\}
=
-i
\left[
(\partial_{pq}\sigma)V
+
2(\partial_q\sigma)(\partial_pV)
\right].
}
\]

Therefore the corrected Fourier-domain volume can be formed as

\[
\boxed{
Y_\ell
=
\tau_\ell V_\ell
-
i\sum_{p,q}
\left[
(\partial_{pq}\sigma_{\ell,pq})V_\ell
+
2(\partial_q\sigma_{\ell,pq})
(\partial_pV_\ell)
\right].
}
\]

Then perform the same single inverse FFT:

\[
u_\ell=\mathcal F^{-1}Y_\ell.
\]

The outer accumulation remains

\[
y
=
\sum_\ell
D_\ell^H u_\ell.
\]

---

## 6. Computational consequence

The ordinary Toeplitz application for segment \(\ell\) is

\[
v_\ell=D_\ell x,
\]

\[
V_\ell=Fv_\ell,
\]

\[
Y_\ell=\tau_\ell V_\ell,
\]

\[
u_\ell=F^{-1}Y_\ell.
\]

With first-order concomitant correction:

1. compute \(V_\ell\) with the existing FFT;
2. compute a small number of finite differences of \(V_\ell\);
3. compute finite differences of \(\tau_\ell\) or of
   \[
   \sigma_{\ell,pq}=c_{pq}\tau_\ell;
   \]
4. form the corrected \(Y_\ell\) pointwise;
5. use the same existing IFFT.

Therefore

\[
\boxed{
\text{FFT count remains }2L.
}
\]

The concomitant correction adds only local finite-difference and pointwise arithmetic.

There are

- no extra FFTs,
- no NUFFTs,
- no extra \(1024^3\) Toeplitz kernels.

---

## 7. Derivatives of the concomitant-weighted multiplier

Since

\[
\sigma_{\ell,pq}=c_{pq}\tau_\ell,
\]

\[
\partial_q\sigma
=
(\partial_q c_{pq})\tau_\ell
+
c_{pq}\partial_q\tau_\ell,
\]

and

\[
\partial_{pq}\sigma
=
(\partial_{pq}c)\tau_\ell
+
(\partial_pc)(\partial_q\tau_\ell)
+
(\partial_qc)(\partial_p\tau_\ell)
+
c\,\partial_{pq}\tau_\ell.
\]

If \(c_{pq}(k)\) is a low-degree polynomial or Chebyshev expansion, its derivatives are analytic and cheap.

Derivatives of the stored Toeplitz multiplier \(\tau_\ell\) can be obtained on-the-fly using finite-difference stencils.

---

## 8. Can all finite differences be moved after the sum over \(\ell\)?

Not completely.

A key term is

\[
(\partial_q\sigma_{\ell,pq})
(\partial_pV_\ell),
\qquad
V_\ell=F(D_\ell x).
\]

Because the \(D_\ell\) differ across time-segmentation terms, there is generally no exact identity that combines all

\[
\partial_pV_\ell
\]

into one derivative after the \(\ell\)-sum.

So the strongest possible claim is **not**

> one set of finite differences after summing all segments.

Instead, the useful result is:

\[
\boxed{
\text{no additional FFTs are required;}
}
\]

the necessary finite differences operate on FFT arrays that already exist.

Only the three first derivatives

\[
\partial_xV_\ell,\quad
\partial_yV_\ell,\quad
\partial_zV_\ell
\]

need to be computed per segment at first order, and they can be reused for all \(p,q\) channels.

---

## 9. Second-order extension

Expanding one order further,

\[
e^{i\Phi_n}
=
1+i\Phi_n-\frac12\Phi_n^2+\cdots,
\]

with

\[
\Phi_n=2m^TC_nd,
\]

produces terms of the form

\[
m_pm_s d_qd_t
\]

weighted by

\[
(C_n)_{pq}(C_n)_{st}.
\]

These correspond to nested commutator / anticommutator operators such as

\[
\boxed{
\{X_p,\{X_s,[X_q,[X_t,S]]\}\}.
}
\]

If the products

\[
(C_n)_{pq}(C_n)_{st}
\]

can also be represented compactly as functions of \(k_n\), then second-order correction still needs

\[
\boxed{
\text{the same FFT/IFFT count}
}
\]

plus first- and second-order finite differences of the already available Fourier-domain volume.

The arithmetic grows, but transform count and large-kernel storage do not.

---

## 10. Canonical rotated 3D trajectories

For a family of rotated 3D readouts, write

\[
g_j(s)=R_jg_0(s)
\]

for the ideal gradient waveform.

Define the six canonical accumulated gradient moments

\[
M_\mu(s)
=
\int_0^s
g_{0,p}(u)g_{0,q}(u)\,du.
\]

Then the concomitant curvature for spoke/readout \(j\) can be written as

\[
\boxed{
C_j(s)
=
\sum_\mu
M_\mu(s)\,C_\mu(R_j).
}
\]

Thus all readouts share the same six canonical functions \(M_\mu(s)\); only the small rotation-dependent matrices change.

The key feasibility question is whether the resulting coefficients can be represented compactly along the global trajectory:

\[
\boxed{
(C_n)_{pq}
\approx
c_{pq}(k_n)
}
\]

or more generally

\[
(C_n)_{pq}
\approx
\sum_{h=1}^{H}
\gamma_{pq,h}\,c_h(k_n),
\qquad H\ll \text{number of readouts}.
\]

If the \(c_h(k)\) are low-order polynomials / Chebyshev functions, then they become low-order differential operators on the existing Toeplitz PSF.

---

## 11. Proposed computational architecture

The desired normal operator is

\[
H
\approx
\sum_{\ell=1}^{L}
D_\ell^H
F^{-1}
\mathcal Q_\ell
F
D_\ell,
\]

where \(\mathcal Q_\ell\) consists of

- the ordinary Toeplitz multiplication by \(\tau_\ell\),
- concomitant finite-difference corrections built from
  \[
  \tau_\ell,\quad
  \nabla\tau_\ell,\quad
  \nabla^2\tau_\ell,
  \]
- derivatives of the already computed Fourier volume
  \[
  V_\ell=F(D_\ell x).
  \]

The intended cost is therefore

\[
\boxed{
L\text{ existing Toeplitz FFT/IFFT pairs}
+
O(1)\text{ local finite-difference arithmetic per segment}.
}
\]

Storage remains

\[
\boxed{
L\text{ large Toeplitz kernels}
}
\]

with no additional concomitant \(1024^3\) kernels.

---

## 12. Main research question

The operator algebra is tractable.

The remaining decisive question is empirical / analytical:

\[
\boxed{
\text{How compactly can the quadratic concomitant coefficients }
(C_n)_{pq}
\text{ be represented as functions of the sampled }k_n?
}
\]

For a canonical 3D trajectory, compute

\[
k_0(s)
=
\gamma\int_0^s g_0(u)\,du
\]

and

\[
M_\mu(s)
=
\int_0^s g_{0,p}(u)g_{0,q}(u)\,du.
\]

Then determine the minimum degree / basis size required for

\[
M_\mu(s)
\approx
p_\mu(k_0(s)).
\]

If this representation is compact, the higher-order concomitant correction can potentially be inserted into the existing Toeplitz reconstruction with

\[
\boxed{
\textbf{zero additional FFTs and zero additional giant kernels.}
}
