//! Pooled preconditioners that keep the estimator (research/precond_estimator.md §3, P0–P3).
//!
//! Everything here is host-side (CPU FFTs through rustfft, rayon over lines) and independent of the GPU operator: the
//! caller supplies the normal operator through [`PcgOps`] (the recon binary: the pooled grouped gradient), so the
//! arithmetic is unit-tested on dense toys (tests below).
//!
//! - **Estimator (P0).** Q = N + μ I on the unknowns, optionally band-limited to the sampled ball (B = F⁻¹ 1_ball F:
//!   the solve is B Q B x = B b on range(B)). Strictly convex, so every SPD preconditioner reaches the same x̂
//!   (Proposition 1 of the note). μ is an estimator choice, set once, not a convergence knob.
//! - **Circulant metrics** [`SymPre`]: P = D^{-1/2} F⁻¹ diag(1/λ) F D^{-1/2}, λ the symbol of D^{-1/2} Q D^{-1/2}.
//!   P0: the nominal (no-physics) Chan symbol scaled to the probed mean, + μ̂. P1: the symbol of the ACTUAL operator by
//!   diagonal probing ([`probe_symbol`]), smoothed, + μ̂. P2: D from the probed image-space diagonal ([`probe_diag`]).
//!   Same conventions as `precond::CirculantPrecond` (forward FFT unnormalised, inverse 1/n, λ on the N grid in FFT
//!   order, x fastest), so a symbol built here is a `CirculantPrecond::from_factors` metric on the device.
//! - **Deflation (P3)** [`Deflation`]: P' = P + U (Θ_c Θ⁻¹ − I) Uᴴ with U the P⁻¹-orthonormal Ritz vectors of P Q,
//!   harvested for free from a PCG run ([`Harvest`], [`ritz`]): the Ritz value θ_i is moved to the edge of the
//!   remaining Ritz range. SPD for every θ_i > 0 (U orthonormal in P⁻¹ by construction), so it keeps the estimator.
//!
//! Inner products of the PCG are Re⟨a, b⟩ (Q, P complex-linear Hermitian); the Ritz problem is solved over the
//! COMPLEX span of the Lanczos vectors (so u and i·u are deflated together).

use std::sync::Arc;

use hasty_cuda::C32;
use nalgebra::{Complex, DMatrix};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

use crate::hoststack::HostStack;
use crate::Error;

const PAR: usize = 1 << 15;

fn as_cx(a: &mut [C32]) -> &mut [Complex32] {
    // SAFETY: C32 and Complex32 are both #[repr(C)] { re: f32, im: f32 }
    unsafe { std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut Complex32, a.len()) }
}

/// Σ conj(a) b (f64 accumulation)
pub fn cdot(a: &[C32], b: &[C32]) -> Complex64 {
    a.par_chunks(PAR)
        .zip(b.par_chunks(PAR))
        .map(|(x, y)| x.iter().zip(y).fold(Complex64::new(0.0, 0.0), |s, (p, q)| s + Complex64::new(p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64, p.re as f64 * q.im as f64 - p.im as f64 * q.re as f64)))
        .reduce(|| Complex64::new(0.0, 0.0), |a, b| a + b)
}

/// Re⟨a, b⟩ (f64 accumulation)
pub fn rdot(a: &[C32], b: &[C32]) -> f64 {
    a.par_chunks(PAR).zip(b.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
}

/// y += c x (complex c)
pub fn caxpy(c: Complex64, x: &[C32], y: &mut [C32]) {
    let (cr, ci) = (c.re as f32, c.im as f32);
    y.par_chunks_mut(PAR).zip(x.par_chunks(PAR)).for_each(|(yc, xc)| {
        for (yi, xi) in yc.iter_mut().zip(xc) {
            yi.re += cr * xi.re - ci * xi.im;
            yi.im += cr * xi.im + ci * xi.re;
        }
    });
}

/// a ⊙= d (real d)
pub fn mul_real(a: &mut [C32], d: &[f32]) {
    a.par_chunks_mut(PAR).zip(d.par_chunks(PAR)).for_each(|(ac, dc)| {
        for (ai, &di) in ac.iter_mut().zip(dc) {
            ai.re *= di;
            ai.im *= di;
        }
    });
}

/// a ⊙= d or a ⊙= conj(d) (complex diagonal)
pub fn mul_cx(a: &mut [C32], d: &[C32], conj: bool) {
    a.par_chunks_mut(PAR).zip(d.par_chunks(PAR)).for_each(|(ac, dc)| {
        for (ai, di) in ac.iter_mut().zip(dc) {
            let di_im = if conj { -di.im } else { di.im };
            *ai = C32::new(ai.re * di.re - ai.im * di_im, ai.re * di_im + ai.im * di.re);
        }
    });
}

/// CPU 3-D FFT on an x-fastest grid: forward unnormalised Σ_x a_x e^{−i k·x}, inverse with 1/n (the conventions of
/// `precond::CirculantPrecond`; rustfft's forward sign)
pub struct Fft3 {
    pub dims: [usize; 3],
    fwd: [Arc<dyn Fft<f32>>; 3],
    inv: [Arc<dyn Fft<f32>>; 3],
}

impl Fft3 {
    pub fn new(dims: [usize; 3]) -> Self {
        let mut pl = FftPlanner::<f32>::new();
        let fwd = [0, 1, 2].map(|a| pl.plan_fft_forward(dims[a]));
        let inv = [0, 1, 2].map(|a| pl.plan_fft_inverse(dims[a]));
        Self { dims, fwd, inv }
    }
    pub fn nv(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
    fn axis(&self, a: &mut [C32], ax: usize, inverse: bool) {
        let [nx, ny, nz] = self.dims;
        if self.dims[ax] == 1 {
            return;
        }
        let f = if inverse { &self.inv[ax] } else { &self.fwd[ax] };
        let a = as_cx(a);
        match ax {
            0 => a.par_chunks_mut(nx).for_each(|l| f.process(l)),
            1 => a.par_chunks_mut(nx * ny).for_each(|pl| {
                let mut buf = vec![Complex32::new(0.0, 0.0); ny];
                for i in 0..nx {
                    for j in 0..ny {
                        buf[j] = pl[i + nx * j];
                    }
                    f.process(&mut buf);
                    for j in 0..ny {
                        pl[i + nx * j] = buf[j];
                    }
                }
            }),
            _ => {
                let nxy = nx * ny;
                let mut t = vec![Complex32::new(0.0, 0.0); a.len()];
                {
                    let ar: &[Complex32] = a;
                    t.par_chunks_mut(nz).enumerate().for_each(|(q, l)| {
                        for k in 0..nz {
                            l[k] = ar[q + nxy * k];
                        }
                        f.process(l);
                    });
                }
                let tr: &[Complex32] = &t;
                a.par_chunks_mut(nxy).enumerate().for_each(|(k, pl)| {
                    for q in 0..nxy {
                        pl[q] = tr[q * nz + k];
                    }
                });
            }
        }
    }
    /// in place, unnormalised
    pub fn forward(&self, a: &mut [C32]) {
        assert_eq!(a.len(), self.nv());
        for ax in 0..3 {
            self.axis(a, ax, false);
        }
    }
    /// in place, with 1/n (inverse ∘ forward = identity)
    pub fn inverse(&self, a: &mut [C32]) {
        assert_eq!(a.len(), self.nv());
        for ax in 0..3 {
            self.axis(a, ax, true);
        }
        let s = 1.0 / self.nv() as f32;
        a.par_chunks_mut(PAR).for_each(|c| c.iter_mut().for_each(|v| { v.re *= s; v.im *= s; }));
    }
}

/// signed FFT index of i on an axis of n points (i ≤ n/2: i, else i − n), as `precond.rs`
fn signed(i: usize, n: usize) -> f64 {
    if i <= n / 2 { i as f64 } else { i as f64 - n as f64 }
}

/// normalised 3-D k radius r = |k| / k_Nyquist (per axis) of FFT-order mode q (r ≤ 1: inside the ball the k-crop keeps)
pub fn kradius(dims: [usize; 3], q: usize) -> f64 {
    let i = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
    (0..3).map(|a| if dims[a] > 1 { (signed(i[a], dims[a]) / (dims[a] as f64 / 2.0)).powi(2) } else { 0.0 }).sum::<f64>().sqrt()
}

/// the sampled ball r ≤ 1 (FFT order): the modes a `--kcrop` reconstruction has data for
pub fn ball_mask(dims: [usize; 3]) -> Vec<bool> {
    (0..dims[0] * dims[1] * dims[2]).map(|q| kradius(dims, q) <= 1.0).collect()
}

/// the symbol mirrored k → −k (for a transform whose forward sign is the other one)
pub fn mirror_symbol(lam: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let m = |i: usize, n: usize| (n - i) % n;
    (0..nx * ny * nz).map(|q| { let (i, j, k) = (q % nx, (q / nx) % ny, q / (nx * ny)); lam[m(i, nx) + nx * (m(j, ny) + ny * m(k, nz))] }).collect()
}

/// B v: v band-limited to `mask` (FFT order) in place (one FFT pair)
pub fn project_band(fft: &Fft3, mask: &[bool], v: &mut [C32]) {
    fft.forward(v);
    v.par_chunks_mut(PAR).zip(mask.par_chunks(PAR)).for_each(|(c, m)| {
        for (vi, &keep) in c.iter_mut().zip(m) {
            if !keep {
                *vi = C32::new(0.0, 0.0);
            }
        }
    });
    fft.inverse(v);
}

/// box filter of radius r along each axis, `passes` times, in place; `periodic`: wrap (k-space in FFT order), else
/// clamp at the borders (image space)
pub fn smooth_box(v: &mut [f32], dims: [usize; 3], r: usize, passes: usize, periodic: bool) {
    if r == 0 {
        return;
    }
    let nv = v.len();
    for _ in 0..passes {
        for ax in 0..3 {
            let n = dims[ax];
            if n == 1 {
                continue;
            }
            let stride = [1, dims[0], dims[0] * dims[1]][ax];
            let src = v.to_vec();
            v.par_chunks_mut(PAR).enumerate().for_each(|(ci, c)| {
                for (o, out) in c.iter_mut().enumerate() {
                    let q = ci * PAR + o;
                    let i = (q / stride) % n;
                    let base = q - i * stride;
                    let (mut s, mut w) = (0.0f64, 0.0f64);
                    for d in -(r as i64)..=(r as i64) {
                        let j = i as i64 + d;
                        let j = if periodic { j.rem_euclid(n as i64) } else if j < 0 || j >= n as i64 { continue } else { j };
                        s += src[base + j as usize * stride] as f64;
                        w += 1.0;
                    }
                    *out = (s / w) as f32;
                }
            });
            debug_assert_eq!(v.len(), nv);
        }
    }
}

/// median of d over the voxels of `mask` (0 for an empty mask)
pub fn median_on(d: &[f32], mask: &[bool]) -> f32 {
    let mut v: Vec<f32> = d.iter().zip(mask).filter(|(_, &m)| m).map(|(&x, _)| x).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// D floored at `frac` × its median over `body`
pub fn floor_at_median(d: &[f32], body: &[bool], frac: f64) -> Vec<f32> {
    let f = (frac * median_on(d, body) as f64) as f32;
    d.iter().map(|&x| x.max(f).max(f32::MIN_POSITIVE)).collect()
}

/// deterministic unit-modulus random phases e^{iφ}, φ uniform (xorshift64*; seed per probe and group)
pub fn phases(n: usize, seed: u64) -> Vec<C32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    if s == 0 {
        s = 1;
    }
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let ph = (s >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
            C32::new(ph.cos() as f32, ph.sin() as f32)
        })
        .collect()
}

/// The caller's operators for [`pcg_groups`]: Q on every group image at once (one pooled pass) and the
/// preconditioner of one group.
pub trait PcgOps {
    /// out_g = Q v_g for every group g (chunks of `v` / `out`)
    fn q(&mut self, v: &HostStack, out: &mut HostStack) -> Result<(), Error>;
    /// z = P_g r
    fn p(&mut self, g: usize, r: &[C32], z: &mut [C32]) -> Result<(), Error>;
}

/// Circulant metric z = D^{-1/2} F⁻¹ diag(1/λ) F D^{-1/2} r on the host (no band; see [`apply_composite`]).
pub struct SymPre {
    pub fft: Arc<Fft3>,
    /// λ_eff > 0 per Fourier mode (FFT order): the symbol of D^{-1/2} Q D^{-1/2}
    pub lam: Vec<f32>,
    pinv: Vec<f32>,
    /// D^{-1/2} per voxel (None: D = I)
    pub dinv: Option<Vec<f32>>,
}

impl SymPre {
    pub fn new(fft: Arc<Fft3>, lam: Vec<f32>, dinv: Option<Vec<f32>>) -> Self {
        assert_eq!(lam.len(), fft.nv());
        assert!(lam.iter().all(|&l| l > 0.0 && l.is_finite()), "SymPre: the symbol must be positive (floor it)");
        let pinv = lam.iter().map(|&l| 1.0 / l).collect();
        Self { fft, lam, pinv, dinv }
    }
    /// z = P r
    pub fn apply(&self, r: &[C32], z: &mut [C32]) {
        z.copy_from_slice(r);
        if let Some(d) = &self.dinv {
            mul_real(z, d);
        }
        self.fft.forward(z);
        mul_real(z, &self.pinv);
        self.fft.inverse(z);
        if let Some(d) = &self.dinv {
            mul_real(z, d);
        }
    }
    /// z = P⁻¹ r = D^{1/2} F⁻¹ diag(λ) F D^{1/2} r
    pub fn apply_inv(&self, r: &[C32], z: &mut [C32]) {
        z.copy_from_slice(r);
        let ds: Option<Vec<f32>> = self.dinv.as_ref().map(|d| d.iter().map(|&x| 1.0 / x).collect());
        if let Some(d) = &ds {
            mul_real(z, d);
        }
        self.fft.forward(z);
        mul_real(z, &self.lam);
        self.fft.inverse(z);
        if let Some(d) = &ds {
            mul_real(z, d);
        }
    }
}

/// Deflation P' = P + Σ_i d_i u_i u_iᴴ, d_i = target_i / θ_i − 1 (u_i P⁻¹-orthonormal Ritz vectors of P Q). SPD
/// whenever P is and every θ_i > 0: in the P^{-1/2}-whitened space P' is I + Û diag(d) Ûᴴ, eigenvalues 1 and
/// target_i / θ_i.
#[derive(Clone, Debug, Default)]
pub struct Deflation {
    pub u: Vec<Vec<C32>>,
    pub theta: Vec<f64>,
    pub target: Vec<f64>,
}

impl Deflation {
    pub fn d(&self, i: usize) -> f64 {
        self.target[i] / self.theta[i] - 1.0
    }
    pub fn rank(&self) -> usize {
        self.u.len()
    }
    /// z += Σ_i d_i u_i (u_iᴴ r)
    pub fn add(&self, r: &[C32], z: &mut [C32]) {
        for (i, u) in self.u.iter().enumerate() {
            let c = cdot(u, r) * self.d(i);
            caxpy(c, u, z);
        }
    }
    /// the inverse: (P')⁻¹ = P⁻¹ + Σ_i (θ_i / target_i − 1) v_i v_iᴴ with v_i = P⁻¹ u_i (Woodbury; `v` given):
    /// z += Σ_i e_i v_i (v_iᴴ r)
    pub fn add_inv(&self, v: &[Vec<C32>], r: &[C32], z: &mut [C32]) {
        for (i, vi) in v.iter().enumerate() {
            let c = cdot(vi, r) * (self.theta[i] / self.target[i] - 1.0);
            caxpy(c, vi, z);
        }
    }
    /// pick from a Ritz set: the `top` largest and `bot` smallest Ritz values whose relative residual is ≤ `max_res`,
    /// each moved to the nearest edge of the range of the converged Ritz values NOT picked. None if nothing qualifies.
    pub fn select(rs: &RitzSet, top: usize, bot: usize, max_res: f64) -> Option<(Vec<usize>, Vec<f64>)> {
        let n = rs.theta.len();
        if n < 3 {
            return None;
        }
        // Ritz values ascending (rs is ascending)
        let ok: Vec<usize> = (0..n).filter(|&i| rs.res[i] <= max_res && rs.theta[i] > 0.0).collect();
        let mut pick: Vec<usize> = ok.iter().rev().take(top).copied().collect();
        let low: Vec<usize> = ok.iter().take(bot).copied().filter(|i| !pick.contains(i)).collect();
        pick.extend(low);
        if pick.is_empty() {
            return None;
        }
        // the target range from the CONVERGED Ritz values not picked: unconverged interior Ritz values (ghosts between
        // eigenvalues, residual ≫ max_res) are no spectrum edge (unit test: a ghost at 90 between 88 and 119)
        let mut rest: Vec<f64> = (0..n).filter(|i| !pick.contains(i) && rs.theta[*i] > 0.0 && rs.res[*i] <= max_res).map(|i| rs.theta[i]).collect();
        if rest.is_empty() {
            rest = (0..n).filter(|i| !pick.contains(i) && rs.theta[*i] > 0.0).map(|i| rs.theta[i]).collect();
        }
        if rest.is_empty() {
            return None;
        }
        let (lo, hi) = (rest.iter().cloned().fold(f64::MAX, f64::min), rest.iter().cloned().fold(0.0, f64::max));
        let tg = pick.iter().map(|&i| rs.theta[i].clamp(lo, hi)).collect();
        Some((pick, tg))
    }
}

/// z = B P_base B r + deflation (B = band-limit, both optional). `base(r, z)` is the preconditioner core. With a band
/// the composite is SPD on range(B) (and zero on its complement), so PCG on B Q B stays in range(B).
pub fn apply_composite(band: Option<(&Fft3, &[bool])>, defl: Option<&Deflation>, r: &[C32], z: &mut [C32], base: &mut dyn FnMut(&[C32], &mut [C32]) -> Result<(), Error>) -> Result<(), Error> {
    match band {
        Some((fft, m)) => {
            let mut t = r.to_vec();
            project_band(fft, m, &mut t);
            base(&t, z)?;
            project_band(fft, m, z);
        }
        None => base(r, z)?,
    }
    if let Some(d) = defl {
        d.add(r, z);
    }
    Ok(())
}

/// Lanczos data of one group's PCG run, kept for the Ritz pairs of P Q ([`ritz`]): the preconditioned residuals z_j
/// (up to `cap`), the Gram entries G_jk = z_jᴴ P⁻¹ z_k = z_jᴴ r_k and the CG scalars. No extra operator applications.
#[derive(Clone, Debug, Default)]
pub struct Harvest {
    pub cap: usize,
    pub z: Vec<Vec<C32>>,
    /// g[k][j] = z_jᴴ r_k, j ≤ k (k up to cap: the last row uses r_cap against the stored z)
    g: Vec<Vec<Complex64>>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
}

impl Harvest {
    pub fn new(cap: usize) -> Self {
        Self { cap, ..Default::default() }
    }
    fn closed(&self) -> bool {
        self.g.len() > self.cap
    }
    /// the k-th residual pair (r_k, z_k = P r_k), k = number observed so far
    pub fn observe(&mut self, r: &[C32], z: &[C32]) {
        if self.closed() {
            return;
        }
        let mut row: Vec<Complex64> = self.z.iter().map(|zj| cdot(zj, r)).collect();
        row.push(cdot(z, r));
        self.g.push(row);
        if self.z.len() < self.cap {
            self.z.push(z.to_vec());
        }
    }
    pub fn step(&mut self, alpha: f64, beta: f64) {
        if self.alpha.len() < self.cap {
            self.alpha.push(alpha);
            self.beta.push(beta);
        }
    }
    /// basis size m usable for Rayleigh–Ritz (needs z_0..z_{m−1}, rows 0..=m, α_0..α_{m−1})
    pub fn m(&self) -> usize {
        self.alpha.len().min(self.z.len()).min(self.g.len().saturating_sub(1))
    }
    pub fn bytes(&self) -> usize {
        self.z.iter().map(|v| v.len() * 8).sum()
    }
}

/// Ritz pairs of P Q from a [`Harvest`] (ascending θ): values, relative residuals ‖PQu − θu‖_{P⁻¹} / θ (exact from the
/// CG recurrences: PQ Z_m = Z_{m+1} T̃), and the coefficient vectors (u = Σ_j c_j z_j, uᴴ P⁻¹ u = 1).
#[derive(Clone, Debug, Default)]
pub struct RitzSet {
    pub theta: Vec<f64>,
    pub res: Vec<f64>,
    pub coef: Vec<Vec<Complex64>>,
    pub m: usize,
}

impl RitzSet {
    /// effective condition number θ_max / θ_min over the Ritz values with relative residual ≤ `max_res`
    pub fn kappa(&self, max_res: f64) -> f64 {
        let v: Vec<f64> = self.theta.iter().zip(&self.res).filter(|(_, &r)| r <= max_res).map(|(&t, _)| t).filter(|&t| t > 0.0).collect();
        if v.len() < 2 {
            return f64::NAN;
        }
        v.iter().cloned().fold(0.0, f64::max) / v.iter().cloned().fold(f64::MAX, f64::min)
    }
    /// the Ritz vector i
    pub fn vector(&self, h: &Harvest, i: usize) -> Vec<C32> {
        let nv = h.z[0].len();
        let mut u = vec![C32::new(0.0, 0.0); nv];
        for (j, c) in self.coef[i].iter().enumerate() {
            caxpy(*c, &h.z[j], &mut u);
        }
        u
    }
    /// the deflation of the selected pairs ([`Deflation::select`])
    pub fn deflation(&self, h: &Harvest, top: usize, bot: usize, max_res: f64) -> Option<Deflation> {
        let (pick, tg) = Deflation::select(self, top, bot, max_res)?;
        Some(Deflation { u: pick.iter().map(|&i| self.vector(h, i)).collect(), theta: pick.iter().map(|&i| self.theta[i]).collect(), target: tg })
    }
}

/// Rayleigh–Ritz over the complex span of the stored z_j (robust to the loss of orthogonality: the Gram matrix is used
/// explicitly and its near-null directions, the Lanczos "ghosts", are dropped)
pub fn ritz(h: &Harvest) -> RitzSet {
    let m = h.m();
    if m == 0 {
        return RitzSet::default();
    }
    let cx = |z: Complex64| Complex::new(z.re, z.im);
    // G (m+1)×(m+1), Hermitian
    let g1 = DMatrix::<Complex<f64>>::from_fn(m + 1, m + 1, |j, k| if j <= k { cx(h.g[k][j]) } else { cx(h.g[j][k]).conj() });
    // T̃ (m+1)×m: PQ z_k = (z_k − z_{k+1})/α_k − β_{k−1} (z_{k−1} − z_k)/α_{k−1}
    let mut tt = DMatrix::<Complex<f64>>::zeros(m + 1, m);
    for k in 0..m {
        let a = h.alpha[k];
        let mut d = 1.0 / a;
        tt[(k + 1, k)] = Complex::new(-1.0 / a, 0.0);
        if k > 0 {
            let (ap, bp) = (h.alpha[k - 1], h.beta[k - 1]);
            d += bp / ap;
            tt[(k - 1, k)] = Complex::new(-bp / ap, 0.0);
        }
        tt[(k, k)] = Complex::new(d, 0.0);
    }
    // scale the basis to unit P⁻¹-norm (the residuals fall by orders of magnitude)
    let s: Vec<f64> = (0..=m).map(|j| 1.0 / g1[(j, j)].re.max(1e-300).sqrt()).collect();
    let ds = |n: usize| DMatrix::<Complex<f64>>::from_fn(n, n, |i, j| if i == j { Complex::new(s[i], 0.0) } else { Complex::new(0.0, 0.0) });
    let (dm, dm1) = (ds(m), ds(m + 1));
    let gn1 = &dm1 * &g1 * &dm1;
    // H = Z_mᴴ Q Z_m = Z_mᴴ P⁻¹ Z_{m+1} T̃ = G[0..m, :] T̃; normalised: D_m H D_m
    let hm = g1.rows(0, m).into_owned() * &tt;
    let mut hn = &dm * hm * &dm;
    hn = (&hn + hn.adjoint()) * Complex::new(0.5, 0.0);
    let tn = {
        // T̂ = D_{m+1}⁻¹ T̃ D_m
        let inv1 = DMatrix::<Complex<f64>>::from_fn(m + 1, m + 1, |i, j| if i == j { Complex::new(1.0 / s[i], 0.0) } else { Complex::new(0.0, 0.0) });
        inv1 * &tt * &dm
    };
    let gm = gn1.view((0, 0), (m, m)).into_owned();
    let ge = nalgebra::SymmetricEigen::new(gm);
    let wmax = ge.eigenvalues.iter().cloned().fold(0.0, f64::max);
    let keep: Vec<usize> = (0..m).filter(|&i| ge.eigenvalues[i] > 1e-9 * wmax).collect();
    let r = keep.len();
    // W = V_keep Λ^{-1/2}: Wᴴ G W = I
    let w = DMatrix::<Complex<f64>>::from_fn(m, r, |i, j| ge.eigenvectors[(i, keep[j])] * Complex::new(1.0 / ge.eigenvalues[keep[j]].sqrt(), 0.0));
    let c = w.adjoint() * &hn * &w;
    let c = (&c + c.adjoint()) * Complex::new(0.5, 0.0);
    let ce = nalgebra::SymmetricEigen::new(c);
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| ce.eigenvalues[a].total_cmp(&ce.eigenvalues[b]));
    let mut out = RitzSet { m, ..Default::default() };
    for &i in &order {
        let th = ce.eigenvalues[i];
        let sn = &w * ce.eigenvectors.column(i);
        // residual in the normalised basis: T̂ s − θ [s; 0], norm² = cᴴ G̃_{m+1} c
        let mut cvec = &tn * &sn;
        for j in 0..m {
            cvec[j] -= sn[j] * Complex::new(th, 0.0);
        }
        let rn2 = (cvec.adjoint() * &gn1 * &cvec)[(0, 0)].re;
        out.theta.push(th);
        out.res.push(rn2.max(0.0).sqrt() / th.abs().max(1e-300));
        out.coef.push((0..m).map(|j| Complex64::new(sn[j].re * s[j], sn[j].im * s[j])).collect());
    }
    out
}

/// PCG controls of [`pcg_groups`]
#[derive(Clone, Debug)]
pub struct PcgCfg {
    pub iters: usize,
    /// stop a group at ‖r‖ / ‖r₀‖ ≤ tol (0: run all iterations)
    pub tol: f64,
    /// Lanczos vectors kept per group for the Ritz pairs (0: none)
    pub harvest: usize,
    /// flexible PCG (Polak–Ribière β = (rᴴ_{k+1} z_{k+1} − rᴴ_{k+1} z_k) / rᴴ_k z_k) for a variable preconditioner
    /// (inner iterations, research/precond_estimator.md §7 G3); the Lanczos harvest is then meaningless (not kept)
    pub flexible: bool,
}

/// result of [`pcg_groups`]
#[derive(Clone, Debug, Default)]
pub struct PcgOut {
    pub iters: usize,
    /// final ‖r‖ / ‖r₀‖ per group
    pub rel: Vec<f64>,
    /// iteration at which each group met the tolerance
    pub converged: Vec<Option<usize>>,
    pub harvest: Vec<Harvest>,
}

/// Preconditioned CG on Q x_g = b_g, independently per group (own scalars), all groups through one `ops.q` per
/// iteration. `x` is the initial guess (updated). `cb(k, x, rel)` after every iteration k (1-based); return false
/// to stop. Inner products Re⟨·,·⟩.
pub fn pcg_groups(ops: &mut dyn PcgOps, b: &HostStack, x: &mut HostStack, cfg: &PcgCfg, cb: &mut dyn FnMut(usize, &HostStack, &[f64]) -> Result<bool, Error>) -> Result<PcgOut, Error> {
    let ng = b.nsets();
    let lens = b.lens.clone();
    let mut r = HostStack::zeros(&lens);
    let mut z = HostStack::zeros(&lens);
    let mut p = HostStack::zeros(&lens);
    let mut q = HostStack::zeros(&lens);
    if x.data.iter().any(|v| v.re != 0.0 || v.im != 0.0) {
        ops.q(x, &mut q)?;
        r.copy_from(b);
        HostStack::axpby(-1.0, &q, 1.0, &mut r);
    } else {
        r.copy_from(b);
    }
    let mut hv: Vec<Harvest> = (0..ng).map(|_| Harvest::new(cfg.harvest)).collect();
    let mut rz = vec![0.0f64; ng];
    let mut r0 = vec![0.0f64; ng];
    let mut rel = vec![1.0f64; ng];
    let mut done: Vec<Option<usize>> = vec![None; ng];
    let mut active = vec![true; ng];
    for g in 0..ng {
        let (o, l) = (z.off[g], z.lens[g]);
        ops.p(g, r.chunk(g), &mut z.data[o..o + l])?;
        p.chunk_mut(g).copy_from_slice(z.chunk(g));
        rz[g] = rdot(r.chunk(g), z.chunk(g));
        r0[g] = rdot(r.chunk(g), r.chunk(g)).sqrt();
        if cfg.harvest > 0 && !cfg.flexible {
            hv[g].observe(r.chunk(g), z.chunk(g));
        }
        if r0[g] == 0.0 {
            active[g] = false;
            done[g] = Some(0);
            rel[g] = 0.0;
        }
    }
    let mut k = 0;
    while k < cfg.iters && active.iter().any(|&a| a) {
        ops.q(&p, &mut q)?;
        for g in 0..ng {
            if !active[g] {
                continue;
            }
            let pq = rdot(p.chunk(g), q.chunk(g));
            if !(pq > 0.0) {
                active[g] = false;
                continue;
            }
            let a = rz[g] / pq;
            caxpy(Complex64::new(a, 0.0), p.chunk(g), x.chunk_mut(g));
            caxpy(Complex64::new(-a, 0.0), q.chunk(g), r.chunk_mut(g));
            // flexible: rᴴ_{k+1} z_k before z is overwritten
            let rz_old = if cfg.flexible { rdot(r.chunk(g), z.chunk(g)) } else { 0.0 };
            let (o, l) = (z.off[g], z.lens[g]);
            ops.p(g, r.chunk(g), &mut z.data[o..o + l])?;
            let rzn = rdot(r.chunk(g), z.chunk(g));
            let be = (rzn - rz_old) / rz[g];
            rz[g] = rzn;
            // p = z + β p
            {
                let (zc, pc) = (z.chunk(g), &mut p.data[o..o + l]);
                pc.par_chunks_mut(PAR).zip(zc.par_chunks(PAR)).for_each(|(pp, zz)| {
                    for (pi, zi) in pp.iter_mut().zip(zz) {
                        pi.re = zi.re + be as f32 * pi.re;
                        pi.im = zi.im + be as f32 * pi.im;
                    }
                });
            }
            if cfg.harvest > 0 && !cfg.flexible {
                hv[g].step(a, be);
                hv[g].observe(r.chunk(g), z.chunk(g));
            }
            rel[g] = rdot(r.chunk(g), r.chunk(g)).sqrt() / r0[g];
            if cfg.tol > 0.0 && rel[g] <= cfg.tol {
                active[g] = false;
                done[g] = Some(k + 1);
            }
        }
        k += 1;
        if !cb(k, x, &rel)? {
            break;
        }
    }
    Ok(PcgOut { iters: k, rel, converged: done, harvest: hv })
}

/// Fourier-diagonal probing of D^{-1/2} Q D^{-1/2} per group (P1): λ̂(k) = Re conj(ξ_k) [F D^{-1/2} Q D^{-1/2} F⁻¹ ξ](k)
/// with unit-modulus random phases ξ (exact for a circulant; for Q = circulant + E the error is the off-diagonal
/// energy of the row, averaged over the probes). `q` applies Q to one image per group (one pooled pass per probe).
/// Returns per group the means over the even and over the odd probes (their difference measures the probe noise).
pub fn probe_symbol(fft: &Fft3, dinv: &[Option<&[f32]>], nprobe: usize, seed: u64, q: &mut dyn FnMut(&HostStack, &mut HostStack) -> Result<(), Error>) -> Result<Vec<[Vec<f32>; 2]>, Error> {
    let ng = dinv.len();
    let nv = fft.nv();
    let lens = vec![nv; ng];
    let mut acc: Vec<[Vec<f64>; 2]> = (0..ng).map(|_| [vec![0.0; nv], vec![0.0; nv]]).collect();
    let mut cnt = [0usize; 2];
    for j in 0..nprobe {
        let mut x = HostStack::zeros(&lens);
        let xi: Vec<Vec<C32>> = (0..ng).map(|g| phases(nv, seed ^ ((j as u64) << 20) ^ (g as u64).wrapping_mul(0x5851_F42D))).collect();
        for g in 0..ng {
            let c = x.chunk_mut(g);
            c.copy_from_slice(&xi[g]);
            fft.inverse(c);
            if let Some(d) = dinv[g] {
                mul_real(c, d);
            }
        }
        let mut y = HostStack::zeros(&lens);
        q(&x, &mut y)?;
        for g in 0..ng {
            let c = y.chunk_mut(g);
            if let Some(d) = dinv[g] {
                mul_real(c, d);
            }
            fft.forward(c);
            let a = &mut acc[g][j % 2];
            a.par_chunks_mut(PAR).zip(c.par_chunks(PAR)).zip(xi[g].par_chunks(PAR)).for_each(|((ac, cc), xc)| {
                for ((ai, ci), xj) in ac.iter_mut().zip(cc).zip(xc) {
                    *ai += xj.re as f64 * ci.re as f64 + xj.im as f64 * ci.im as f64;
                }
            });
        }
        cnt[j % 2] += 1;
    }
    Ok(acc.into_iter().map(|[a, b]| [a.iter().map(|&v| (v / cnt[0].max(1) as f64) as f32).collect(), b.iter().map(|&v| (v / cnt[1].max(1) as f64) as f32).collect()]).collect())
}

/// Image-space diagonal probing per group (P2): d̂(x) = Re conj(η_x) (Q η)(x), unit-modulus random η. Same return
/// convention as [`probe_symbol`].
pub fn probe_diag(ng: usize, nv: usize, nprobe: usize, seed: u64, q: &mut dyn FnMut(&HostStack, &mut HostStack) -> Result<(), Error>) -> Result<Vec<[Vec<f32>; 2]>, Error> {
    let lens = vec![nv; ng];
    let mut acc: Vec<[Vec<f64>; 2]> = (0..ng).map(|_| [vec![0.0; nv], vec![0.0; nv]]).collect();
    let mut cnt = [0usize; 2];
    for j in 0..nprobe {
        let mut x = HostStack::zeros(&lens);
        for g in 0..ng {
            x.chunk_mut(g).copy_from_slice(&phases(nv, seed ^ 0xA5A5 ^ ((j as u64) << 20) ^ (g as u64).wrapping_mul(0x5851_F42D)));
        }
        let mut y = HostStack::zeros(&lens);
        q(&x, &mut y)?;
        for g in 0..ng {
            let (xc, yc) = (x.chunk(g), y.chunk(g));
            let a = &mut acc[g][j % 2];
            a.par_chunks_mut(PAR).zip(xc.par_chunks(PAR)).zip(yc.par_chunks(PAR)).for_each(|((ac, xx), yy)| {
                for ((ai, xi), yi) in ac.iter_mut().zip(xx).zip(yy) {
                    *ai += xi.re as f64 * yi.re as f64 + xi.im as f64 * yi.im as f64;
                }
            });
        }
        cnt[j % 2] += 1;
    }
    Ok(acc.into_iter().map(|[a, b]| [a.iter().map(|&v| (v / cnt[0].max(1) as f64) as f32).collect(), b.iter().map(|&v| (v / cnt[1].max(1) as f64) as f32).collect()]).collect())
}

/// The P1 symbol from the two probe halves: smoothed (periodic box, radius `rad`, 2 passes), clamped at 0. With a
/// nominal symbol, the ratio form λ_nom · S(λ_probe) / S(λ_nom) (keeps the nominal fine structure, e.g. the k = 0 peak,
/// and takes the physics correction from the probes). Returns (symbol, median split-half relative difference over the
/// modes where the symbol is ≥ 1e-3 of its max: the probe noise after smoothing).
pub fn smooth_symbol(halves: &[Vec<f32>; 2], nominal: Option<&[f32]>, dims: [usize; 3], rad: usize) -> (Vec<f32>, f64) {
    let sm = |v: &[f32]| {
        let mut w = v.to_vec();
        smooth_box(&mut w, dims, rad, 2, true);
        w
    };
    let (a, b) = (sm(&halves[0]), sm(&halves[1]));
    let mean: Vec<f32> = a.iter().zip(&b).map(|(x, y)| 0.5 * (x + y)).collect();
    let lam: Vec<f32> = match nominal {
        Some(nom) => {
            let sn = sm(nom);
            nom.iter().zip(&sn).zip(&mean).map(|((&n, &s), &m)| if s > 0.0 { (n * m / s).max(0.0) } else { m.max(0.0) }).collect()
        }
        None => mean.iter().map(|&m| m.max(0.0)).collect(),
    };
    let lmax = lam.iter().cloned().fold(0.0f32, f32::max);
    let mut rd: Vec<f64> = a.iter().zip(&b).zip(&mean).filter(|(_, &m)| m >= 1e-3 * lmax && m > 0.0).map(|((&x, &y), &m)| ((x - y) as f64 / m as f64).abs()).collect();
    rd.sort_by(|p, q| p.total_cmp(q));
    let med = if rd.is_empty() { f64::NAN } else { rd[rd.len() / 2] };
    (lam, med)
}

/// where a vector's energy sits: fractions of ‖F u‖² per shell of r = |k| / k_Nyquist (edges 0, 1/8, 1/4, 1/2, 3/4,
/// 0.9, 1, 1.2, ∞), inside the ball (r ≤ 1), and of ‖u‖² in each image mask
pub fn energy_where(fft: &Fft3, u: &[C32], masks: &[(&str, &[bool])]) -> String {
    const EDGES: [f64; 8] = [0.125, 0.25, 0.5, 0.75, 0.9, 1.0, 1.2, f64::INFINITY];
    let mut k = u.to_vec();
    fft.forward(&mut k);
    let dims = fft.dims;
    let mut sh = [0.0f64; 8];
    let mut tot = 0.0;
    for (q, v) in k.iter().enumerate() {
        let e = v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64;
        let r = kradius(dims, q);
        let i = EDGES.iter().position(|&x| r <= x).unwrap_or(7);
        sh[i] += e;
        tot += e;
    }
    let inside: f64 = sh[..6].iter().sum::<f64>() / tot.max(1e-300);
    let it: f64 = u.iter().map(|v| v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64).sum();
    let img: Vec<String> = masks.iter().map(|(n, m)| format!("{n} {:.3}", u.iter().zip(m.iter()).filter(|(_, &b)| b).map(|(v, _)| v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64).sum::<f64>() / it.max(1e-300))).collect();
    format!("k shells {} | in ball {inside:.3} | image {}", sh.iter().map(|v| format!("{:.3}", v / tot.max(1e-300))).collect::<Vec<_>>().join(" "), img.join(", "))
}

/// Preconditioner spec of the pooled solve: `base[+dT[,B]]`, base ∈ none (plain CG), circ (the existing per-encode
/// circulant), p0 (nominal Chan symbol, matched floor), p1 / p1n (probed symbol; n = ratio form with the nominal
/// symbol), p12 / p12n (p1 with D from the probed image diagonal), p2 (the probed image diagonal alone: Jacobi);
/// +dT deflates the T largest Ritz pairs, +dT,B also the B smallest (from the harvest of the same base's run).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreSpec {
    pub base: String,
    pub top: usize,
    pub bot: usize,
    /// "+u": the recentred metric U P Uᴴ (G2, `surrogate::recentre_diag`)
    pub recentre: bool,
}

impl PreSpec {
    pub fn parse(s: &str) -> Option<Self> {
        let recentre = s.contains("+u");
        let s0 = s.replacen("+u", "", 1);
        let s = s0.as_str();
        let (base, d) = match s.split_once("+d") {
            Some((b, d)) => (b, Some(d)),
            None => (s, None),
        };
        let pe_ok = base.strip_prefix("pe").and_then(|r| r.split_once('_')).is_some_and(|(e, q)| e.parse::<f64>().is_ok_and(|v| v > 0.0) && q.parse::<f64>().is_ok_and(|v| v > 0.0));
        let sur_ok = Self::parse_sur(base).is_some();
        if !matches!(base, "none" | "circ" | "p0" | "p1" | "p1n" | "p12" | "p12n" | "p2") && !pe_ok && !sur_ok {
            return None;
        }
        let (top, bot) = match d {
            None => (0, 0),
            Some(d) => match d.split_once(',') {
                Some((t, b)) => (t.parse().ok()?, b.parse().ok()?),
                None => (d.parse().ok()?, 0),
            },
        };
        Some(Self { base: base.to_string(), top, bot, recentre })
    }
    /// `sur<L>k<K>[n]` / `cheb<L>k<K>[n]`: the coil + segment surrogate with L_s = L maps, K inner PCG (flexible
    /// outer) or K-step Chebyshev (fixed) inner steps; n = no inner metric (else the existing DCD circulant)
    fn parse_sur(b: &str) -> Option<(bool, usize, usize, bool)> {
        let (cheb, r) = if let Some(r) = b.strip_prefix("cheb") { (true, r) } else { (false, b.strip_prefix("sur")?) };
        let (r, metric) = match r.strip_suffix('n') { Some(r) => (r, false), None => (r, true) };
        let (l, k) = r.split_once('k')?;
        let (l, k): (usize, usize) = (l.parse().ok()?, k.parse().ok()?);
        (l >= 1 && k >= 1).then_some((cheb, l, k, metric))
    }
    /// the surrogate base: (Chebyshev?, L_s, K, inner metric?)
    pub fn sur(&self) -> Option<(bool, usize, usize, bool)> {
        Self::parse_sur(&self.base)
    }
    /// a variable preconditioner (inner PCG): the outer solve must be flexible
    pub fn flexible(&self) -> bool {
        self.sur().is_some_and(|s| !s.0)
    }
    pub fn deflates(&self) -> bool {
        self.top + self.bot > 0
    }
    /// probed symbol needed
    pub fn probed(&self) -> bool {
        self.base.starts_with("p1") || self.base == "p0"
    }
    /// probed image diagonal needed
    pub fn diag(&self) -> bool {
        self.base.starts_with("p12") || self.base == "p2"
    }
    /// the bounded-dynamic-range circulant `pe<ε>_<p>`: P = D^{-1/2} F⁻¹ (λ_nom + ε λ_max)^{-p} F D^{-1/2} (nominal
    /// symbol, no probes): (ε, p)
    pub fn power(&self) -> Option<(f64, f64)> {
        let (e, q) = self.base.strip_prefix("pe")?.split_once('_')?;
        Some((e.parse().ok()?, q.parse().ok()?))
    }
    /// a host circulant (`SymPre`) base
    pub fn sym(&self) -> bool {
        self.probed() || self.diag() || self.power().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<C32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                let mut u = || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
                };
                C32::new(u(), u())
            })
            .collect()
    }
    fn nrm(a: &[C32]) -> f64 {
        rdot(a, a).sqrt()
    }
    fn diff(a: &[C32], b: &[C32]) -> f64 {
        let d: Vec<C32> = a.iter().zip(b).map(|(x, y)| C32::new(x.re - y.re, x.im - y.im)).collect();
        nrm(&d) / nrm(b).max(1e-300)
    }

    /// dense Hermitian PD Q = U diag(ev) Uᴴ (complex, f64) with a spread spectrum and a few large outliers
    struct Dense {
        n: usize,
        q: Vec<Complex64>,
    }
    impl Dense {
        fn new(n: usize, seed: u64) -> Self {
            let a = DMatrix::<Complex<f64>>::from_fn(n, n, |i, j| {
                let v = rnd(1, seed ^ ((i * n + j) as u64 + 1) * 0x9E37)[0];
                Complex::new(v.re as f64, v.im as f64)
            });
            let qr = a.qr();
            let u = qr.q();
            // bulk 0.1 … 1, four outliers 30 … 120 (κ ≈ 1200: f32 PCG still resolves 1e-4 differences)
            let ev: Vec<f64> = (0..n).map(|i| if i + 4 >= n { 30.0 * (i + 5 - n) as f64 } else { 0.1 * (10f64).powf(i as f64 / (n - 5) as f64) }).collect();
            let d = DMatrix::<Complex<f64>>::from_fn(n, n, |i, j| if i == j { Complex::new(ev[i], 0.0) } else { Complex::new(0.0, 0.0) });
            let m = &u * d * u.adjoint();
            Self { n, q: (0..n * n).map(|k| { let z = m[(k / n, k % n)]; Complex64::new(z.re, z.im) }).collect() }
        }
        fn mul(&self, x: &[C32], y: &mut [C32]) {
            for i in 0..self.n {
                let s: Complex64 = (0..self.n).map(|j| self.q[i * self.n + j] * Complex64::new(x[j].re as f64, x[j].im as f64)).sum();
                y[i] = C32::new(s.re as f32, s.im as f32);
            }
        }
        fn mat(&self) -> DMatrix<Complex<f64>> {
            DMatrix::from_fn(self.n, self.n, |i, j| { let z = self.q[i * self.n + j]; Complex::new(z.re, z.im) })
        }
    }

    enum Pk {
        Id,
        Diag(Vec<f32>),
        Sym(SymPre),
    }
    struct Ops<'a> {
        q: &'a Dense,
        pk: &'a Pk,
        defl: Option<&'a Deflation>,
    }
    impl PcgOps for Ops<'_> {
        fn q(&mut self, v: &HostStack, out: &mut HostStack) -> Result<(), Error> {
            for g in 0..v.nsets() {
                let (o, l) = (out.off[g], out.lens[g]);
                self.q.mul(v.chunk(g), &mut out.data[o..o + l]);
            }
            Ok(())
        }
        fn p(&mut self, _g: usize, r: &[C32], z: &mut [C32]) -> Result<(), Error> {
            let pk = self.pk;
            apply_composite(None, self.defl, r, z, &mut |a, b| {
                match pk {
                    Pk::Id => b.copy_from_slice(a),
                    Pk::Diag(d) => {
                        b.copy_from_slice(a);
                        mul_real(b, d);
                    }
                    Pk::Sym(s) => s.apply(a, b),
                }
                Ok(())
            })
        }
    }

    fn solve(q: &Dense, pk: &Pk, defl: Option<&Deflation>, b: &[C32], iters: usize, harvest: usize) -> (Vec<C32>, PcgOut) {
        let bs = HostStack { data: b.to_vec(), off: vec![0], lens: vec![b.len()] };
        let mut x = HostStack::zeros(&[b.len()]);
        let mut ops = Ops { q, pk, defl };
        let out = pcg_groups(&mut ops, &bs, &mut x, &PcgCfg { iters, tol: 1e-6, harvest, flexible: false }, &mut |_, _, _| Ok(true)).unwrap();
        (x.data, out)
    }

    #[test]
    fn fft3_matches_dft() {
        let dims = [4usize, 3, 5];
        let n = 60;
        let a = rnd(n, 3);
        let f = Fft3::new(dims);
        let mut b = a.clone();
        f.forward(&mut b);
        for q in [0usize, 7, 23, 59] {
            let (k0, k1, k2) = (q % 4, (q / 4) % 3, q / 12);
            let mut s = Complex64::new(0.0, 0.0);
            for p in 0..n {
                let (x0, x1, x2) = (p % 4, (p / 4) % 3, p / 12);
                let ph = -std::f64::consts::TAU * ((k0 * x0) as f64 / 4.0 + (k1 * x1) as f64 / 3.0 + (k2 * x2) as f64 / 5.0);
                s += Complex64::new(a[p].re as f64, a[p].im as f64) * Complex64::from_polar(1.0, ph);
            }
            assert!((s - Complex64::new(b[q].re as f64, b[q].im as f64)).norm() < 1e-4, "mode {q}");
        }
        f.inverse(&mut b);
        assert!(diff(&b, &a) < 1e-6);
    }

    #[test]
    fn sympre_spd_and_inverse() {
        let dims = [6usize, 4, 5];
        let n = 120;
        let f = Arc::new(Fft3::new(dims));
        let lam: Vec<f32> = (0..n).map(|q| 0.05 + (1.0 - kradius(dims, q) as f32).max(0.0) * 10.0).collect();
        let dinv: Vec<f32> = (0..n).map(|q| 0.5 + (q % 7) as f32 * 0.1).collect();
        let p = SymPre::new(f.clone(), lam, Some(dinv));
        let (a, b) = (rnd(n, 11), rnd(n, 12));
        let (mut pa, mut pb) = (vec![C32::default(); n], vec![C32::default(); n]);
        p.apply(&a, &mut pa);
        p.apply(&b, &mut pb);
        // Hermitian: ⟨a, P b⟩ = ⟨P a, b⟩
        let (l, r) = (cdot(&a, &pb), cdot(&pa, &b));
        assert!((l - r).norm() < 1e-5 * l.norm().max(1.0));
        assert!(rdot(&a, &pa) > 0.0);
        let mut back = vec![C32::default(); n];
        p.apply_inv(&pa, &mut back);
        assert!(diff(&back, &a) < 1e-5);
        // band composite: zero outside the ball, Hermitian
        let mask = ball_mask(dims);
        let (mut ca, mut cb) = (vec![C32::default(); n], vec![C32::default(); n]);
        apply_composite(Some((&*f, &mask[..])), None, &a, &mut ca, &mut |x, y| { p.apply(x, y); Ok(()) }).unwrap();
        apply_composite(Some((&*f, &mask[..])), None, &b, &mut cb, &mut |x, y| { p.apply(x, y); Ok(()) }).unwrap();
        assert!((cdot(&a, &cb) - cdot(&ca, &b)).norm() < 1e-5);
        let mut k = ca.clone();
        f.forward(&mut k);
        let out: f64 = k.iter().zip(&mask).filter(|(_, &m)| !m).map(|(v, _)| (v.re * v.re + v.im * v.im) as f64).sum();
        assert!(out < 1e-8 * rdot(&k, &k));
    }

    #[test]
    fn probe_symbol_exact_for_circulant() {
        let dims = [8usize, 4, 3];
        let n = 96;
        let f = Fft3::new(dims);
        let lam: Vec<f32> = (0..n).map(|q| 1.0 + (q % 5) as f32).collect();
        let mut q = |v: &HostStack, out: &mut HostStack| -> Result<(), Error> {
            out.copy_from(v);
            for g in 0..v.nsets() {
                let c = out.chunk_mut(g);
                f.forward(c);
                mul_real(c, &lam);
                f.inverse(c);
            }
            Ok(())
        };
        let h = probe_symbol(&f, &[None], 2, 5, &mut q).unwrap();
        for half in &h[0] {
            assert!(half.iter().zip(&lam).all(|(a, b)| (a - b).abs() < 1e-4 * b), "probe of a circulant is exact");
        }
        let (s, noise) = smooth_symbol(&h[0], None, dims, 0);
        assert!(s.iter().zip(&lam).all(|(a, b)| (a - b).abs() < 1e-4 * b) && noise < 1e-4);
        // the image diagonal of a circulant is its mean symbol
        let d = probe_diag(1, n, 2, 9, &mut q).unwrap();
        let mean = lam.iter().sum::<f32>() / n as f32;
        let dm = d[0][0].iter().sum::<f32>() / n as f32;
        // (one probe per half: the voxel mean of the estimate fluctuates by ~10 %; this checks the normalisation)
        assert!((dm - mean).abs() < 0.4 * mean, "{dm} vs {mean}");
    }

    #[test]
    fn converged_solution_independent_of_preconditioner() {
        let n = 48;
        let q = Dense::new(n, 21);
        let b = rnd(n, 22);
        let f = Arc::new(Fft3::new([4, 4, 3]));
        let lam: Vec<f32> = (0..n).map(|k| 0.2 + (k % 9) as f32).collect();
        let pks = [Pk::Id, Pk::Diag((0..n).map(|i| 0.3 + (i % 5) as f32 * 0.4).collect()), Pk::Sym(SymPre::new(f, lam, Some((0..n).map(|i| 0.7 + (i % 3) as f32 * 0.2).collect())))];
        let xs: Vec<Vec<C32>> = pks.iter().map(|pk| solve(&q, pk, None, &b, 400, 0).0).collect();
        for x in &xs[1..] {
            assert!(diff(x, &xs[0]) < 1e-3, "{}", diff(x, &xs[0]));
        }
        // and it solves Q x = b
        let mut qx = vec![C32::default(); n];
        q.mul(&xs[0], &mut qx);
        assert!(diff(&qx, &b) < 1e-4, "{}", diff(&qx, &b));
        // with a deflation (harvested from the diagonal run) the same solution
        let (_, out) = solve(&q, &pks[1], None, &b, 30, 30);
        let rs = ritz(&out.harvest[0]);
        let d = rs.deflation(&out.harvest[0], 4, 1, 1e-2).expect("converged Ritz pairs");
        let (xd, _) = solve(&q, &pks[1], Some(&d), &b, 400, 0);
        assert!(diff(&xd, &xs[0]) < 1e-3);
    }

    #[test]
    fn ritz_pairs_and_deflation() {
        let n = 48;
        let q = Dense::new(n, 31);
        let b = rnd(n, 32);
        let dg: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4) as f32 * 0.3).collect();
        let pk = Pk::Diag(dg.clone());
        let (_, out) = solve(&q, &pk, None, &b, 25, 25);
        let h = &out.harvest[0];
        let rs = ritz(h);
        // reference: eigenvalues of P^{1/2} Q P^{1/2}
        let ph = DMatrix::<Complex<f64>>::from_fn(n, n, |i, j| if i == j { Complex::new((dg[i] as f64).sqrt(), 0.0) } else { Complex::new(0.0, 0.0) });
        let a = &ph * q.mat() * &ph;
        let mut ev: Vec<f64> = nalgebra::SymmetricEigen::new(a).eigenvalues.iter().cloned().collect();
        ev.sort_by(|x, y| x.total_cmp(y));
        let top = *rs.theta.last().unwrap();
        assert!((top / ev[n - 1] - 1.0).abs() < 1e-4, "top Ritz {top} vs {}", ev[n - 1]);
        assert!(*rs.res.last().unwrap() < 1e-3);
        // P⁻¹-normalised Ritz vector with a small true residual
        let i = rs.theta.len() - 1;
        let u = rs.vector(h, i);
        let pinv_u: Vec<C32> = u.iter().zip(&dg).map(|(v, &d)| C32::new(v.re / d, v.im / d)).collect();
        assert!((rdot(&u, &pinv_u) - 1.0).abs() < 1e-3);
        let mut qu = vec![C32::default(); n];
        q.mul(&u, &mut qu);
        let pqu: Vec<C32> = qu.iter().zip(&dg).map(|(v, &d)| C32::new(v.re * d, v.im * d)).collect();
        let res: Vec<C32> = pqu.iter().zip(&u).map(|(a, b)| C32::new(a.re - top as f32 * b.re, a.im - top as f32 * b.im)).collect();
        assert!(nrm(&res) / (top * nrm(&u)) < 1e-2);
        // deflated preconditioner: Hermitian, PD, top eigenvalues moved down, inverse by Woodbury
        let d = rs.deflation(h, 4, 0, 1e-2).unwrap();
        let mut pm = DMatrix::<Complex<f64>>::zeros(n, n);
        for j in 0..n {
            let mut e = vec![C32::default(); n];
            e[j] = C32::new(1.0, 0.0);
            let mut z = vec![C32::default(); n];
            Ops { q: &q, pk: &pk, defl: Some(&d) }.p(0, &e, &mut z).unwrap();
            for i in 0..n {
                pm[(i, j)] = Complex::new(z[i].re as f64, z[i].im as f64);
            }
        }
        assert!((&pm - pm.adjoint()).norm() < 1e-5 * pm.norm());
        let pe = nalgebra::SymmetricEigen::new((&pm + pm.adjoint()) * Complex::new(0.5, 0.0));
        assert!(pe.eigenvalues.iter().all(|&v| v > 0.0), "deflated preconditioner not PD");
        let l = pe.eigenvectors.clone() * DMatrix::from_fn(n, n, |i, j| if i == j { Complex::new(pe.eigenvalues[i].sqrt(), 0.0) } else { Complex::new(0.0, 0.0) }) * pe.eigenvectors.adjoint();
        let mut e2: Vec<f64> = nalgebra::SymmetricEigen::new(&l * q.mat() * &l).eigenvalues.iter().cloned().collect();
        e2.sort_by(|x, y| x.total_cmp(y));
        assert!(e2[n - 1] < 0.5 * ev[n - 1], "deflation did not lower λ_max: {} vs {}", e2[n - 1], ev[n - 1]);
        // inverse: v = P⁻¹ u; (P + U D Uᴴ)(P⁻¹ r + Σ e v vᴴ r) = r
        let v: Vec<Vec<C32>> = d.u.iter().map(|u| u.iter().zip(&dg).map(|(x, &g)| C32::new(x.re / g, x.im / g)).collect()).collect();
        let r = rnd(n, 33);
        let mut w: Vec<C32> = r.iter().zip(&dg).map(|(x, &g)| C32::new(x.re / g, x.im / g)).collect();
        d.add_inv(&v, &r, &mut w);
        let mut back = vec![C32::default(); n];
        Ops { q: &q, pk: &pk, defl: Some(&d) }.p(0, &w, &mut back).unwrap();
        // (f32 vectors, deflation factors θ_c/θ ≈ 1/100: ~1e-4 rounding)
        assert!(diff(&back, &r) < 1e-3, "{}", diff(&back, &r));
        // fewer iterations with the deflation
        let (_, o1) = solve(&q, &pk, None, &b, 400, 0);
        let (_, o2) = solve(&q, &pk, Some(&d), &b, 400, 0);
        assert!(o2.iters < o1.iters, "deflated {} vs {}", o2.iters, o1.iters);
    }

    #[test]
    fn spec_parse() {
        assert_eq!(PreSpec::parse("p1+d8"), Some(PreSpec { base: "p1".into(), top: 8, bot: 0, recentre: false }));
        assert_eq!(PreSpec::parse("p12n+d16,4"), Some(PreSpec { base: "p12n".into(), top: 16, bot: 4, recentre: false }));
        assert_eq!(PreSpec::parse("circ"), Some(PreSpec { base: "circ".into(), top: 0, bot: 0, recentre: false }));
        assert_eq!(PreSpec::parse("circ+u").map(|p| (p.base, p.recentre)), Some(("circ".to_string(), true)));
        let s = PreSpec::parse("sur3k10").unwrap();
        assert_eq!((s.sur(), s.flexible()), (Some((false, 3, 10, true)), true));
        let c = PreSpec::parse("cheb2k8n+d8").unwrap();
        assert_eq!((c.sur(), c.flexible(), c.top), (Some((true, 2, 8, false)), false, 8));
        assert!(PreSpec::parse("surk3").is_none() && PreSpec::parse("sur0k3").is_none());
        assert!(PreSpec::parse("bogus").is_none() && PreSpec::parse("p1+dx").is_none());
        let pe = PreSpec::parse("pe3e-2_0.75+d8").unwrap();
        assert_eq!((pe.power(), pe.top, pe.sym()), (Some((3e-2, 0.75)), 8, true));
        assert!(PreSpec::parse("pe_1").is_none() && PreSpec::parse("pe1e-2").is_none());
        assert_eq!(mirror_symbol(&mirror_symbol(&(0..24).map(|v| v as f32).collect::<Vec<_>>(), [4, 3, 2]), [4, 3, 2]), (0..24).map(|v| v as f32).collect::<Vec<_>>());
    }
}
