//! Smooth coil maps: a tensor-product cubic B-spline fit of estimated coil sensitivities (research/smaps_estimation.md).
//!
//! The estimated maps m_c(r) (e.g. ESPIRiT, unit norm per voxel and phase-aligned to a virtual reference coil) are
//! replaced by s_c = B a_c, with B the cubic B-spline basis of knot spacing h (metres) on the reconstruction grid and
//!     a_c = argmin_a  Σ_r w(r) |(B a)(r) − m_c(r)|²  +  λ̄ Σ_axes ‖D²_axis a‖²  +  μ̄ ‖a‖²,
//! a weighted least-squares fit with a second-difference (thin-plate-like) penalty on the coefficient grid. w is a
//! confidence (ESPIRiT eigenvalue ramp × object support): where w = 0 (low-signal holes such as the lungs, the masked
//! rim) the penalty extends the maps smoothly from their surroundings; the basis cannot carry structure finer than
//! ≈ h, so image content that leaked into the estimate above that scale is removed. λ̄ = λ · d̄ with d̄ the mean
//! diagonal of BᵀWB over the coefficients the data touch (λ is dimensionless); μ̄ = 1e-6 d̄ keeps the system definite.
//! One normal operator serves every coil (the weights do not depend on the coil): CG per coil, in parallel.
//!
//! After the fit, [`normalise_rss`] restores the ESPIRiT gauge Σ_c |s_c|² = 1 on a mask (the scaling ambiguity
//! S x = (S g)(x / g) fixed as ESPIRiT does) and [`align_phase`] removes a per-voxel phase relative to reference maps
//! (the same gauge for maps of different respiratory bins).

use hasty_cuda::C32;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

/// one axis of the basis: per voxel the first coefficient index and the four cubic weights
#[derive(Clone, Debug)]
struct Axis {
    k: usize,
    tab: Vec<(usize, [f32; 4])>,
}

impl Axis {
    fn new(n: usize, vox: f64, h: f64) -> Self {
        assert!(n > 0 && vox > 0.0 && h > 0.0);
        if n == 1 {
            // a singleton axis: one constant coefficient
            return Self { k: 1, tab: vec![(0, [1.0, 0.0, 0.0, 0.0])] };
        }
        let len = (n - 1) as f64 * vox;
        let span = (len / h).floor() as usize;
        let k = span + 4;
        let tab = (0..n)
            .map(|i| {
                let u = (i as f64 * vox / h).min(span as f64 + 1.0 - 1e-9);
                let j = (u.floor() as usize).min(span);
                let t = u - j as f64;
                let w = [
                    (1.0 - t).powi(3) / 6.0,
                    (3.0 * t.powi(3) - 6.0 * t * t + 4.0) / 6.0,
                    (-3.0 * t.powi(3) + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
                    t.powi(3) / 6.0,
                ];
                (j, w.map(|v| v as f32))
            })
            .collect();
        Self { k, tab }
    }
}

/// Tensor-product cubic B-spline basis on a grid (x fastest), coefficients [kx, ky, kz] (x fastest)
#[derive(Clone, Debug)]
pub struct Bspline3 {
    pub dims: [usize; 3],
    ax: [Axis; 3],
}

impl Bspline3 {
    /// `vox`, `h`: voxel size and knot spacing (metres) per axis
    pub fn new(dims: [usize; 3], vox: [f64; 3], h: f64) -> Self {
        Self { dims, ax: [0, 1, 2].map(|a| Axis::new(dims[a], vox[a], h)) }
    }

    pub fn kdims(&self) -> [usize; 3] {
        [self.ax[0].k, self.ax[1].k, self.ax[2].k]
    }

    pub fn ncoef(&self) -> usize {
        self.ax.iter().map(|a| a.k).product()
    }

    pub fn nv(&self) -> usize {
        self.dims.iter().product()
    }

    /// image = B coef (separable: x, then y, then z)
    pub fn eval(&self, coef: &[Complex32]) -> Vec<Complex32> {
        let [kx, ky, kz] = self.kdims();
        let [nx, ny, nz] = self.dims;
        assert_eq!(coef.len(), kx * ky * kz);
        let z0 = Complex32::new(0.0, 0.0);
        // along x: [nx, ky, kz]
        let mut t1 = vec![z0; nx * ky * kz];
        t1.par_chunks_mut(nx).enumerate().for_each(|(row, o)| {
            let c = &coef[row * kx..(row + 1) * kx];
            for (i, oi) in o.iter_mut().enumerate() {
                let (j, w) = self.ax[0].tab[i];
                *oi = (0..4).filter(|&d| j + d < kx).map(|d| c[j + d] * w[d]).sum();
            }
        });
        // along y: [nx, ny, kz]
        let mut t2 = vec![z0; nx * ny * kz];
        t2.par_chunks_mut(nx * ny).enumerate().for_each(|(zk, o)| {
            let src = &t1[zk * nx * ky..(zk + 1) * nx * ky];
            for y in 0..ny {
                let (j, w) = self.ax[1].tab[y];
                for d in 0..4 {
                    if j + d >= ky {
                        continue;
                    }
                    let s = &src[(j + d) * nx..(j + d + 1) * nx];
                    for x in 0..nx {
                        o[y * nx + x] += s[x] * w[d];
                    }
                }
            }
        });
        drop(t1);
        // along z: [nx, ny, nz]
        let nxy = nx * ny;
        let mut out = vec![z0; nxy * nz];
        out.par_chunks_mut(nxy).enumerate().for_each(|(z, o)| {
            let (j, w) = self.ax[2].tab[z];
            for d in 0..4 {
                if j + d >= kz {
                    continue;
                }
                let s = &t2[(j + d) * nxy..(j + d + 1) * nxy];
                for q in 0..nxy {
                    o[q] += s[q] * w[d];
                }
            }
        });
        out
    }

    /// coef = Bᵀ image (real weights: the adjoint is the transpose)
    pub fn adjoint(&self, img: &[Complex32]) -> Vec<Complex32> {
        let [kx, ky, kz] = self.kdims();
        let [nx, ny, nz] = self.dims;
        assert_eq!(img.len(), nx * ny * nz);
        let z0 = Complex32::new(0.0, 0.0);
        let nxy = nx * ny;
        // along z: [nx, ny, kz]
        let mut t2 = vec![z0; nxy * kz];
        t2.par_chunks_mut(nxy).enumerate().for_each(|(zk, o)| {
            for z in 0..nz {
                let (j, w) = self.ax[2].tab[z];
                if zk < j || zk >= j + 4 {
                    continue;
                }
                let wd = w[zk - j];
                let s = &img[z * nxy..(z + 1) * nxy];
                for q in 0..nxy {
                    o[q] += s[q] * wd;
                }
            }
        });
        // along y: [nx, ky, kz]
        let mut t1 = vec![z0; nx * ky * kz];
        t1.par_chunks_mut(nx * ky).enumerate().for_each(|(zk, o)| {
            let src = &t2[zk * nxy..(zk + 1) * nxy];
            for y in 0..ny {
                let (j, w) = self.ax[1].tab[y];
                for d in 0..4 {
                    if j + d >= ky {
                        continue;
                    }
                    let oo = &mut o[(j + d) * nx..(j + d + 1) * nx];
                    for x in 0..nx {
                        oo[x] += src[y * nx + x] * w[d];
                    }
                }
            }
        });
        drop(t2);
        // along x: [kx, ky, kz]
        let mut out = vec![z0; kx * ky * kz];
        out.par_chunks_mut(kx).enumerate().for_each(|(row, o)| {
            let s = &t1[row * nx..(row + 1) * nx];
            for (i, si) in s.iter().enumerate() {
                let (j, w) = self.ax[0].tab[i];
                for d in 0..4 {
                    if j + d < kx {
                        o[j + d] += *si * w[d];
                    }
                }
            }
        });
        out
    }

    /// diag(Bᵀ W B) per coefficient: (B∘B)ᵀ w (separable squares of the weights)
    pub fn diag_normal(&self, w: &[f32]) -> Vec<f64> {
        let sq = Self { dims: self.dims, ax: self.ax.clone().map(|a| Axis { k: a.k, tab: a.tab.iter().map(|(j, w)| (*j, w.map(|v| v * v))).collect() }) };
        let img: Vec<Complex32> = w.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        sq.adjoint(&img).iter().map(|z| z.re as f64).collect()
    }

    /// Σ_axes (D²_a)ᵀ D²_a c on the coefficient grid (second differences, interior rows only)
    fn reg_apply(&self, c: &[Complex32]) -> Vec<Complex32> {
        let k = self.kdims();
        let mut out = vec![Complex32::new(0.0, 0.0); c.len()];
        let stride = [1, k[0], k[0] * k[1]];
        for a in 0..3 {
            if k[a] < 3 {
                continue;
            }
            let s = stride[a];
            for q in 0..c.len() {
                let i = (q / s) % k[a];
                if i == 0 || i + 1 >= k[a] {
                    continue;
                }
                let d = c[q - s] - c[q] * 2.0 + c[q + s];
                out[q - s] += d;
                out[q] -= d * 2.0;
                out[q + s] += d;
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct FitOpts {
    /// knot spacing (metres)
    pub h: f64,
    /// dimensionless second-difference penalty (relative to the mean diagonal of BᵀWB)
    pub lambda: f64,
    pub iters: usize,
    pub tol: f64,
}

impl Default for FitOpts {
    fn default() -> Self {
        // 16 mm: the ph0001 true maps are reproduced to 1 − |⟨ŝ, s⟩| ≈ 3e-4 within 20 mm of the skin (32 mm: 2.5e-3,
        // as large as ESPIRiT's own error there; research/experiments/smaps/results/bspline_rep.txt)
        Self { h: 0.016, lambda: 1e-3, iters: 200, tol: 1e-6 }
    }
}

pub struct FitReport {
    pub ncoef: usize,
    pub kdims: [usize; 3],
    /// CG iterations (max over coils) and final relative residual (max over coils)
    pub iters: usize,
    pub rel_res: f64,
    /// Σ w |B a − m|² / Σ w |m|² over all coils (how much of the input the smooth basis does not reproduce)
    pub misfit: f64,
}

fn dotc(a: &[Complex32], b: &[Complex32]) -> f64 {
    a.par_iter().zip(b).map(|(x, y)| (x.re as f64) * (y.re as f64) + (x.im as f64) * (y.im as f64)).sum()
}

/// Smooth weighted fit of coil-major maps (nc × nv); returns the fitted maps (nc × nv, B a_c everywhere) and a report.
pub fn fit_maps(b: &Bspline3, maps: &[C32], nc: usize, w: &[f32], o: &FitOpts) -> (Vec<C32>, FitReport) {
    let nv = b.nv();
    assert_eq!(maps.len(), nc * nv, "fit_maps: nc × nv maps");
    assert_eq!(w.len(), nv, "fit_maps: one weight per voxel");
    let diag = b.diag_normal(w);
    let touched: Vec<f64> = diag.iter().copied().filter(|&d| d > 0.0).collect();
    let dbar = if touched.is_empty() { 1.0 } else { touched.iter().sum::<f64>() / touched.len() as f64 };
    let (lam, mu) = ((o.lambda * dbar) as f32, (1e-6 * dbar) as f32);
    let normal = |c: &[Complex32]| -> Vec<Complex32> {
        let img = b.eval(c);
        let wi: Vec<Complex32> = img.par_iter().zip(w).map(|(z, &v)| z * v).collect();
        let mut n = b.adjoint(&wi);
        let r = b.reg_apply(c);
        n.par_iter_mut().zip(r.par_iter().zip(c)).for_each(|(ni, (ri, ci))| *ni += ri * lam + ci * mu);
        n
    };
    let mut out = vec![C32::new(0.0, 0.0); nc * nv];
    let (mut it_max, mut res_max, mut mis_num, mut mis_den) = (0usize, 0f64, 0f64, 0f64);
    // coils one after another (each operator application is itself parallel)
    for c in 0..nc {
        let m: Vec<Complex32> = maps[c * nv..(c + 1) * nv].iter().zip(w).map(|(z, &v)| Complex32::new(z.re, z.im) * v).collect();
        let rhs = b.adjoint(&m);
        let r0 = dotc(&rhs, &rhs);
        let mut x = vec![Complex32::new(0.0, 0.0); rhs.len()];
        let mut it = 0;
        let mut rel = 0.0;
        if r0 > 0.0 {
            let mut r = rhs.clone();
            let mut p = r.clone();
            let mut rr = r0;
            while it < o.iters {
                let ap = normal(&p);
                let alpha = (rr / dotc(&p, &ap).max(1e-300)) as f32;
                x.par_iter_mut().zip(&p).for_each(|(xi, pi)| *xi += pi * alpha);
                r.par_iter_mut().zip(&ap).for_each(|(ri, ai)| *ri -= ai * alpha);
                let rn = dotc(&r, &r);
                it += 1;
                rel = (rn / r0).sqrt();
                if rel < o.tol {
                    break;
                }
                let beta = (rn / rr) as f32;
                p.par_iter_mut().zip(&r).for_each(|(pi, ri)| *pi = ri + *pi * beta);
                rr = rn;
            }
        }
        let fitc = b.eval(&x);
        for q in 0..nv {
            let (f, mq) = (fitc[q], maps[c * nv + q]);
            let d = Complex64::new((f.re - mq.re) as f64, (f.im - mq.im) as f64);
            mis_num += w[q] as f64 * d.norm_sqr();
            mis_den += w[q] as f64 * ((mq.re as f64).powi(2) + (mq.im as f64).powi(2));
            out[c * nv + q] = C32::new(f.re, f.im);
        }
        it_max = it_max.max(it);
        res_max = res_max.max(rel);
    }
    (out, FitReport { ncoef: b.ncoef(), kdims: b.kdims(), iters: it_max, rel_res: res_max, misfit: mis_num / mis_den.max(1e-300) })
}

/// Σ_c |s_c|² = 1 where `mask` (and the norm is positive), 0 elsewhere (the ESPIRiT gauge)
pub fn normalise_rss(maps: &mut [C32], nc: usize, mask: &[bool]) {
    let nv = mask.len();
    assert_eq!(maps.len(), nc * nv);
    for q in 0..nv {
        let n = (0..nc).map(|c| (maps[c * nv + q].re as f64).powi(2) + (maps[c * nv + q].im as f64).powi(2)).sum::<f64>().sqrt();
        let s = if mask[q] && n > 0.0 { (1.0 / n) as f32 } else { 0.0 };
        for c in 0..nc {
            maps[c * nv + q].re *= s;
            maps[c * nv + q].im *= s;
        }
    }
}

/// Per-voxel phase gauge relative to reference maps: s ← s · conj(φ)/|φ|, φ = ⟨s_ref, s⟩ = Σ_c conj(s_ref,c) s_c (so
/// that ⟨s_ref, s⟩ is real and ≥ 0). Voxels with φ = 0 are left unchanged. Maps of different respiratory bins aligned to
/// one reference share the image phase convention (no phase jumps between bins from the gauge).
pub fn align_phase(maps: &mut [C32], reference: &[C32], nc: usize) {
    assert_eq!(maps.len(), reference.len());
    let nv = maps.len() / nc;
    for q in 0..nv {
        let phi: Complex64 = (0..nc).map(|c| {
            let (r, s) = (reference[c * nv + q], maps[c * nv + q]);
            Complex64::new(r.re as f64, -r.im as f64) * Complex64::new(s.re as f64, s.im as f64)
        }).sum();
        let n = phi.norm();
        if n > 0.0 {
            let u = phi.conj() / n;
            for c in 0..nc {
                let s = Complex64::new(maps[c * nv + q].re as f64, maps[c * nv + q].im as f64) * u;
                maps[c * nv + q] = C32::new(s.re as f32, s.im as f32);
            }
        }
    }
}

/// Fit weights from an ESPIRiT eigenvalue map: w = (clamp((λ − e0) / (1 − e0), 0, 1))² on `mask`, 0 elsewhere
pub fn eig_weights(eig: &[f32], mask: &[bool], e0: f64) -> Vec<f32> {
    assert_eq!(eig.len(), mask.len());
    eig.iter().zip(mask).map(|(&l, &m)| if m { (((l as f64 - e0) / (1.0 - e0)).clamp(0.0, 1.0)).powi(2) as f32 } else { 0.0 }).collect()
}

/// Map agreement 1 − |⟨ŝ, s⟩| / (‖ŝ‖ ‖s‖) per voxel (phase- and scale-invariant; 1 where either map is zero)
pub fn map_disagreement(a: &[C32], b: &[C32], nc: usize) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    let nv = a.len() / nc;
    (0..nv)
        .into_par_iter()
        .map(|q| {
            let (mut ip, mut na, mut nb) = (Complex64::new(0.0, 0.0), 0.0, 0.0);
            for c in 0..nc {
                let (x, y) = (Complex64::new(a[c * nv + q].re as f64, a[c * nv + q].im as f64), Complex64::new(b[c * nv + q].re as f64, b[c * nv + q].im as f64));
                ip += x.conj() * y;
                na += x.norm_sqr();
                nb += y.norm_sqr();
            }
            if na > 0.0 && nb > 0.0 { 1.0 - ip.norm() / (na * nb).sqrt() } else { 1.0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// smooth complex maps (Gaussian magnitude, linear phase), coil-major
    fn smooth_maps(dims: [usize; 3], nc: usize) -> Vec<C32> {
        let nv = dims.iter().product::<usize>();
        let mut s = vec![C32::new(0.0, 0.0); nc * nv];
        for c in 0..nc {
            let th = 2.0 * std::f64::consts::PI * c as f64 / nc as f64;
            let (cx, cy) = (0.45 * th.cos(), 0.45 * th.sin());
            for q in 0..nv {
                let p = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
                let u = [0, 1, 2].map(|a| (p[a] as f64 - (dims[a] / 2) as f64) / dims[a] as f64);
                let r2 = (u[0] - cx).powi(2) + (u[1] - cy).powi(2) + 0.5 * u[2].powi(2);
                let z = Complex64::from_polar((-r2 / 0.2).exp(), 1.2 * (u[0] * th.sin() - u[1] * th.cos()) + 0.4 * c as f64);
                s[c * nv + q] = C32::new(z.re as f32, z.im as f32);
            }
        }
        s
    }

    #[test]
    fn adjoint_is_the_transpose() {
        let b = Bspline3::new([13, 9, 7], [0.004, 0.004, 0.005], 0.011);
        let n = b.ncoef();
        let c: Vec<Complex32> = (0..n).map(|i| Complex32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos())).collect();
        let v: Vec<Complex32> = (0..b.nv()).map(|i| Complex32::new((i as f32 * 0.21).cos(), (i as f32 * 0.07).sin())).collect();
        let bc = b.eval(&c);
        let btv = b.adjoint(&v);
        let lhs: Complex64 = bc.iter().zip(&v).map(|(x, y)| Complex64::new(x.re as f64, -x.im as f64) * Complex64::new(y.re as f64, y.im as f64)).sum();
        let rhs: Complex64 = c.iter().zip(&btv).map(|(x, y)| Complex64::new(x.re as f64, -x.im as f64) * Complex64::new(y.re as f64, y.im as f64)).sum();
        assert!((lhs - rhs).norm() < 1e-4 * lhs.norm().max(1.0), "{lhs} vs {rhs}");
        // partition of unity: constant coefficients give a constant image
        let one = vec![Complex32::new(1.0, 0.0); n];
        assert!(b.eval(&one).iter().all(|z| (z.re - 1.0).abs() < 1e-5 && z.im.abs() < 1e-6));
    }

    /// a hole without data (w = 0) is filled from its surroundings; a high-frequency texture added to the input (image
    /// content leaking into the estimate) is removed; the fit of smooth maps is accurate
    #[test]
    fn fills_holes_and_removes_fine_structure() {
        let dims = [40usize, 36, 24];
        let vox = [0.005, 0.005, 0.006];
        let nc = 6;
        let nv = dims.iter().product::<usize>();
        let s = smooth_maps(dims, nc);
        let mut m = s.clone();
        let mut w = vec![1f32; nv];
        let mut hole = vec![false; nv];
        for q in 0..nv {
            let p = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            // texture: a 2-voxel checkerboard modulation of 3 % (leakage-like, above the basis' resolution)
            let tex = if (p[0] / 2 + p[1] / 2 + p[2] / 2) % 2 == 0 { 1.03 } else { 0.97 };
            for c in 0..nc {
                m[c * nv + q].re *= tex;
                m[c * nv + q].im *= tex;
            }
            // a ball hole of radius 6 voxels off-centre: no data there
            let d2 = (p[0] as f64 - 14.0).powi(2) + (p[1] as f64 - 18.0).powi(2) + (p[2] as f64 - 12.0).powi(2);
            if d2 < 36.0 {
                w[q] = 0.0;
                hole[q] = true;
                for c in 0..nc {
                    m[c * nv + q] = C32::new(0.0, 0.0);
                }
            }
        }
        let b = Bspline3::new(dims, vox, 0.03);
        let (fit, rep) = fit_maps(&b, &m, nc, &w, &FitOpts { h: 0.03, lambda: 1e-3, iters: 400, tol: 1e-7 });
        eprintln!("fit: {} coefficients {:?}, {} CG iterations, rel. residual {:.1e}, misfit {:.2e}", rep.ncoef, rep.kdims, rep.iters, rep.rel_res, rep.misfit);
        let e = map_disagreement(&fit, &s, nc);
        let e_in: Vec<f64> = (0..nv).filter(|&q| hole[q]).map(|q| e[q]).collect();
        let e_out: Vec<f64> = (0..nv).filter(|&q| !hole[q]).map(|q| e[q]).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let e_tex: Vec<f64> = map_disagreement(&m, &s, nc).into_iter().enumerate().filter(|(q, _)| !hole[*q]).map(|(_, v)| v).collect();
        eprintln!("1 − |⟨ŝ, s⟩|: input (textured, outside the hole) {:.2e}; fit outside the hole {:.2e}, inside the hole {:.2e}", mean(&e_tex), mean(&e_out), mean(&e_in));
        // the 3 % texture is common to the coils: |<,>| sees none of it (it is a per-voxel scale); the magnitude does
        let rel_mag = |a: &[C32]| -> f64 {
            (0..nv).filter(|&q| !hole[q]).map(|q| (0..nc).map(|c| { let (x, y) = (a[c * nv + q], s[c * nv + q]); ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2) }).sum::<f64>()).sum::<f64>()
                / (0..nc * nv).map(|i| (s[i].re as f64).powi(2) + (s[i].im as f64).powi(2)).sum::<f64>()
        };
        let (mi, mf) = (rel_mag(&m), rel_mag(&fit));
        eprintln!("relative squared map error outside the hole: input {mi:.2e}, fit {mf:.2e}");
        assert!(mf < 0.1 * mi, "the fit should remove the fine texture: {mf:.2e} vs {mi:.2e}");
        assert!(mean(&e_out) < 2e-3, "fit outside the hole {:.2e}", mean(&e_out));
        assert!(mean(&e_in) < 2e-2, "hole fill {:.2e}", mean(&e_in));
    }

    #[test]
    fn gauge_helpers() {
        let dims = [8usize, 6, 4];
        let nc = 3;
        let nv = 192;
        let s = smooth_maps(dims, nc);
        // a per-voxel complex gauge g(r): normalise_rss + align_phase to the reference recover s/‖s‖
        let mut g = s.clone();
        for q in 0..nv {
            let z = Complex32::from_polar(0.5 + q as f32 / nv as f32, q as f32 * 0.3);
            for c in 0..nc {
                let v = Complex32::new(g[c * nv + q].re, g[c * nv + q].im) * z;
                g[c * nv + q] = C32::new(v.re, v.im);
            }
        }
        let mask = vec![true; nv];
        let mut r = s.clone();
        normalise_rss(&mut r, nc, &mask);
        normalise_rss(&mut g, nc, &mask);
        align_phase(&mut g, &r, nc);
        let d = g.iter().zip(&r).map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)) as f64).fold(0.0, f64::max);
        assert!(d < 1e-10, "gauge not recovered: {d:.2e}");
        let w = eig_weights(&[0.4, 0.75, 1.0], &[true, true, false], 0.5);
        assert!(w[0] == 0.0 && (w[1] - 0.25).abs() < 1e-6 && w[2] == 0.0);
    }
}
