//! Coil + segment circulant surrogate of the normal operator (research/precond_estimator.md §7, G2/G3).
//!
//! Q = Σ_c S_cᴴ (Σ_{l,l'} B_lᴴ T_{ll'} B_{l'}) S_c exactly (stored SVD segment maps B_l, Toeplitz kernels T_{ll'} of the
//! sample weights a_l*(τ) a_{l'}(τ)). The surrogate keeps the coil diagonals S_c and (a rank-L_s compression of) the
//! stored maps exactly and replaces each Toeplitz T_{kk'} by its Chan (Fejér-folded) circulant C_{kk'} on the 1× grid:
//!
//!   Q̃ = Σ_c S_cᴴ (Σ_{k,k' < L_s} B'_kᴴ C_{kk'} B'_{k'}) S_c + μ I,   C_{kk'} = F⁻¹ diag(λ_{kk'}) F.
//!
//! The per-k matrix Λ(k) = [λ_{kk'}(k)] is Hermitian PSD (each Chan symbol is a Fejér average of the sample weights
//! conj(a'_k) a'_{k'}, i.e. of rank-one PSD matrices a' a'ᴴ), so Q̃ ⪰ μ I. It follows the off-resonance k-shift exactly
//! through B'_k (the stored maps, no on-the-fly diagonal) and keeps the coils, which a D·C·D metric folds away.
//!
//! - Host math (tested on CPU): [`compress`] (rank-L_s compression of the stored decomposition g ≈ Σ_l B_l a_l(τ)),
//!   [`recentre_diag`] (G2), [`pair_weights`], [`chan_symbol_host`] (direct Fejér sum, tests), [`SurHost`] (reference
//!   apply), [`inner_pcg_host`] / [`cheb_host`] (the two preconditioner forms).
//! - Device: [`chan_symbol_dev`] (`build_kernels` with complex `extra` weights, folded as `CirculantPrecond::new`),
//!   [`SurDev`] (apply: per coil L_s forward FFTs, the per-k L_s × L_s product, L_s inverse FFTs), its inner PCG (a
//!   variable preconditioner: the outer solve is flexible PCG) and fixed-degree Chebyshev (a fixed SPD operator), with
//!   an upper spectral bound by Lanczos ([`crate::lanczos`]).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_fft::{Direction, FftPlan, Norm};
use hasty_nufft::SegHostTables;
use hasty_toeplitz::{build_kernels, KernelBuildOpts, KernelKey, Samples};
use nalgebra::{Complex, DMatrix};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::gpu::Kernels;
use crate::ppre::{cdot, rdot, Fft3};
use crate::precond::CirculantPrecond;
use crate::Error;

/// index of the pair (k, k'), k ≤ k' < ls, in the upper-triangle storage
pub fn pidx(k: usize, kp: usize, ls: usize) -> usize {
    debug_assert!(k <= kp && kp < ls);
    k * ls - k * (k + 1) / 2 + kp
}

fn cx(z: Complex64) -> Complex<f64> {
    Complex::new(z.re, z.im)
}

/// The rank-L_s compression of the stored decomposition g(v, τ) ≈ Σ_l B_l(v) a_l(τ) (exact linear combinations of the
/// stored maps and interpolators, built once): B'_k = Σ_l B_l M_lk, a'_k(τ) = Σ_l a_l(τ) N_lk, the best rank-L_s
/// approximation of B Aᵀ in the inner products weighted by `voxel_w` (e.g. the coil power) and by the sample count per
/// table node (`taus`: the samples' times). Returns (B'_k as nv-vectors, N (L × L_s, column-major [l + L k]), the
/// relative weighted energy NOT kept, Σ_{k ≥ L_s} σ_k² / Σ σ_k²).
pub fn compress(tabs: &SegHostTables, nv: usize, ls: usize, voxel_w: &[f32], taus: &[f64]) -> (Vec<Vec<C32>>, Vec<Complex64>, f64) {
    let l = tabs.nseg;
    let ls = ls.min(l).max(1);
    // Gram of the maps over voxels (weighted) and of the interpolators over the samples (count per table node)
    let mut gb = DMatrix::<Complex<f64>>::zeros(l, l);
    for v in 0..nv {
        let w = voxel_w[v].max(0.0) as f64;
        if w == 0.0 {
            continue;
        }
        for i in 0..l {
            let bi = tabs.segph[i * nv + v];
            let bi = Complex::new(bi.re as f64, bi.im as f64);
            for j in 0..l {
                let bj = tabs.segph[j * nv + v];
                gb[(i, j)] += bi.conj() * Complex::new(bj.re as f64, bj.im as f64) * w;
            }
        }
    }
    let mut cnt = vec![0f64; tabs.ntab];
    for &t in taus {
        let u = ((t - tabs.t0) / tabs.dt).round().clamp(0.0, (tabs.ntab - 1) as f64) as usize;
        cnt[u] += 1.0;
    }
    let mut ga = DMatrix::<Complex<f64>>::zeros(l, l);
    for (i, &c) in cnt.iter().enumerate() {
        if c == 0.0 {
            continue;
        }
        for p in 0..l {
            let ap = tabs.atab[i * l + p];
            let ap = Complex::new(ap.re as f64, ap.im as f64);
            for q in 0..l {
                let aq = tabs.atab[i * l + q];
                ga[(p, q)] += ap.conj() * Complex::new(aq.re as f64, aq.im as f64) * c;
            }
        }
    }
    // Hermitian square roots and their inverses (eigen; tiny eigenvalues floored)
    let sq = |g: &DMatrix<Complex<f64>>| -> (DMatrix<Complex<f64>>, DMatrix<Complex<f64>>) {
        let gh = (g + g.adjoint()) * Complex::new(0.5, 0.0);
        let e = nalgebra::SymmetricEigen::new(gh);
        let emax = e.eigenvalues.iter().cloned().fold(0.0, f64::max).max(1e-300);
        let d = |p: f64| DMatrix::<Complex<f64>>::from_fn(l, l, |i, j| if i == j { Complex::new(e.eigenvalues[i].max(1e-12 * emax).powf(p), 0.0) } else { Complex::new(0.0, 0.0) });
        (&e.eigenvectors * d(0.5) * e.eigenvectors.adjoint(), &e.eigenvectors * d(-0.5) * e.eigenvectors.adjoint())
    };
    let (rb, rbi) = sq(&gb);
    let (ra, rai) = sq(&ga);
    // B Aᵀ = (B Rb⁻¹)(Rb Raᵀ)(Ra⁻ᵀ Aᵀ); with Hermitian roots Raᵀ = conj(Ra). SVD of X = Rb conj(Ra)
    let x = &rb * ra.map(|z| z.conj());
    let svd = x.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut order: Vec<usize> = (0..svd.singular_values.len()).collect();
    order.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
    let tot: f64 = svd.singular_values.iter().map(|s| s * s).sum();
    let kept: f64 = order.iter().take(ls).map(|&i| svd.singular_values[i].powi(2)).sum();
    // M = Rb⁻¹ U_s Σ_s (maps), N = conj(Ra⁻¹) conj(V_s) (interpolators): B' a'ᵀ = B Rb⁻¹ U Σ Vᴴ Ra⁻ᵀ Aᵀ
    let m = DMatrix::<Complex<f64>>::from_fn(l, ls, |i, k| {
        let c = order[k];
        (0..l).map(|j| rbi[(i, j)] * u[(j, c)]).sum::<Complex<f64>>() * Complex::new(svd.singular_values[c], 0.0)
    });
    // Vᴴ rows: v_t[(c, j)] = conj(V[j, c]); N[i, k] = Σ_j conj(rai[i, j]) conj(V[j, c]) = Σ_j conj(rai[i, j]) v_t[(c, j)]
    let n = DMatrix::<Complex<f64>>::from_fn(l, ls, |i, k| {
        let c = order[k];
        (0..l).map(|j| rai[(i, j)].conj() * vt[(c, j)]).sum::<Complex<f64>>()
    });
    let maps: Vec<Vec<C32>> = (0..ls)
        .map(|k| {
            (0..nv)
                .into_par_iter()
                .map(|v| {
                    let z: Complex<f64> = (0..l).map(|i| { let b = tabs.segph[i * nv + v]; Complex::new(b.re as f64, b.im as f64) * m[(i, k)] }).sum();
                    C32::new(z.re as f32, z.im as f32)
                })
                .collect()
        })
        .collect();
    let nmix: Vec<Complex64> = (0..ls).flat_map(|k| (0..l).map(move |i| (i, k))).map(|(i, k)| Complex64::new(n[(i, k)].re, n[(i, k)].im)).collect();
    (maps, nmix, ((tot - kept) / tot.max(1e-300)).max(0.0))
}

/// a'_k(τ) = Σ_l a_l(τ) N_lk for k < ls (`nmix` column-major [l + L k], from [`compress`])
pub fn interp_mixed(tabs: &SegHostTables, nmix: &[Complex64], ls: usize, tau: f64, a: &mut [Complex64], out: &mut [Complex64]) {
    tabs.interp(tau, a);
    let l = tabs.nseg;
    for k in 0..ls {
        out[k] = (0..l).map(|i| a[i] * nmix[i + l * k]).sum();
    }
}

/// per-sample weights conj(a'_k(τ_i)) a'_{k'}(τ_i) for every pair k ≤ k' (upper-triangle order, [`pidx`])
pub fn pair_weights(tabs: &SegHostTables, nmix: &[Complex64], ls: usize, taus: &[f64]) -> Vec<Vec<C32>> {
    let np = ls * (ls + 1) / 2;
    let per: Vec<Vec<C32>> = taus
        .par_iter()
        .map_init(
            || (vec![Complex64::new(0.0, 0.0); tabs.nseg], vec![Complex64::new(0.0, 0.0); ls]),
            |(a, o), &t| {
                interp_mixed(tabs, nmix, ls, t, a, o);
                let mut w = Vec::with_capacity(np);
                for k in 0..ls {
                    for kp in k..ls {
                        let z = o[k].conj() * o[kp];
                        w.push(C32::new(z.re as f32, z.im as f32));
                    }
                }
                w
            },
        )
        .collect();
    (0..np).map(|p| per.iter().map(|w| w[p]).collect()).collect()
}

/// G2: the recentring diagonal U(v) = conj(phase(g(v, t_ref))) with g(v, t_ref) = Σ_l B_l(v) a_l(t_ref) from the
/// STORED maps (unit modulus; 1 where g = 0). The recentred metric is P' = U P Uᴴ (the field-free operator is
/// conjugated by the phase the data carry at t_ref: A ≈ A₀ diag(g(t_ref)), so Q⁻¹ ≈ conj(g) Q₀⁻¹ g).
pub fn recentre_diag(tabs: &SegHostTables, nv: usize, t_ref: f64) -> Vec<C32> {
    let mut a = vec![Complex64::new(0.0, 0.0); tabs.nseg];
    tabs.interp(t_ref, &mut a);
    (0..nv)
        .into_par_iter()
        .map(|v| {
            let g: Complex64 = (0..tabs.nseg).map(|l| { let b = tabs.segph[l * nv + v]; Complex64::new(b.re as f64, b.im as f64) * a[l] }).sum();
            let n = g.norm();
            if n > 0.0 && n.is_finite() { C32::new((g.re / n) as f32, (-g.im / n) as f32) } else { C32::new(1.0, 0.0) }
        })
        .collect()
}

/// Chan symbol by the direct Fejér sum (tests; O(M · nv)): λ(m) = Σ_i w_i Π_a F_{N_a}(k_{i,a} − 2π m_a / N_a), F_N(θ) =
/// sin²(Nθ/2) / (N sin²(θ/2)), m in FFT order (x fastest). `mirror`: k_i → −k_i (the other NUFFT sign).
pub fn chan_symbol_host(dims: [usize; 3], coords: [&[f32]; 3], w: &[C32], mirror: bool) -> Vec<C32> {
    let fej = |n: usize, th: f64| -> f64 {
        let s = (0.5 * th).sin();
        if s.abs() < 1e-9 { n as f64 } else { (0.5 * n as f64 * th).sin().powi(2) / (n as f64 * s * s) }
    };
    let nv = dims[0] * dims[1] * dims[2];
    let sg = if mirror { -1.0 } else { 1.0 };
    (0..nv)
        .into_par_iter()
        .map(|q| {
            let m = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..w.len() {
                let mut f = 1.0;
                for a in 0..3 {
                    f *= fej(dims[a], sg * coords[a][i] as f64 - std::f64::consts::TAU * m[a] as f64 / dims[a] as f64);
                }
                acc += Complex64::new(w[i].re as f64, w[i].im as f64) * f;
            }
            C32::new(acc.re as f32, acc.im as f32)
        })
        .collect()
}

/// Chan symbol on the device: `build_kernels` with the complex per-sample weights `w` (the `extra` weight set) on the
/// 2N grid, back to lags, triangle-windowed and folded onto N, forward FFT (the steps of `CirculantPrecond::new`,
/// complex here). Same conventions as `CirculantPrecond` (forward FFT unnormalised).
pub fn chan_symbol_dev(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3], coords: [&[f32]; 3], w: &[C32]) -> Result<Vec<C32>, Error> {
    let nv = dims[0] * dims[1] * dims[2];
    let m = coords[0].len();
    assert_eq!(w.len(), m);
    let sdims = [dims[2], dims[1], dims[0]];
    let pad = [2 * dims[0], 2 * dims[1], 2 * dims[2]];
    let dc: Vec<CudaSlice<f32>> = coords.iter().map(|c| st.clone_htod(*c)).collect::<Result<_, _>>()?;
    let wt = st.clone_htod(&vec![1f32; m])?;
    let zero = st.alloc_zeros::<f32>(m)?;
    let extra: Vec<CudaSlice<C32>> = vec![st.clone_htod(w)?];
    let samples = Samples::<f32> { m, coords: [&dc[0], &dc[1], &dc[2]], wt: &wt, w: [&zero, &zero, &zero, &zero], extra: &extra };
    let key = KernelKey { mu: [0; 4], gamma: [0; 3], extra: Some(0) };
    let mut k = build_kernels(dev, st, sdims, &samples, &[key], &KernelBuildOpts { tol: 1e-5, sigma: 2.0 })?.remove(0);
    let inv2 = FftPlan::<f32>::new(dev, &[pad[2], pad[1], pad[0]], Direction::Inverse, Norm::Inverse)?;
    inv2.execute_inplace(st, &mut k)?;
    let h = st.clone_dtoh(&k)?;
    drop(k);
    let mut c = vec![C32::new(0.0, 0.0); nv];
    let lag = |i: usize, n2: usize| -> i64 { if i < n2 / 2 { i as i64 } else { i as i64 - n2 as i64 } };
    for kz in 0..pad[2] {
        let lz = lag(kz, pad[2]);
        let wz = 1.0 - lz.unsigned_abs() as f64 / dims[2] as f64;
        if wz <= 0.0 {
            continue;
        }
        for ky in 0..pad[1] {
            let ly = lag(ky, pad[1]);
            let wy = 1.0 - ly.unsigned_abs() as f64 / dims[1] as f64;
            if wy <= 0.0 {
                continue;
            }
            for kx in 0..pad[0] {
                let lx = lag(kx, pad[0]);
                let wx = 1.0 - lx.unsigned_abs() as f64 / dims[0] as f64;
                if wx <= 0.0 {
                    continue;
                }
                let wgt = (wx * wy * wz) as f32;
                let v = h[kx + pad[0] * (ky + pad[1] * kz)];
                let q = lx.rem_euclid(dims[0] as i64) as usize + dims[0] * (ly.rem_euclid(dims[1] as i64) as usize + dims[1] * lz.rem_euclid(dims[2] as i64) as usize);
                c[q].re += wgt * v.re;
                c[q].im += wgt * v.im;
            }
        }
    }
    let fwd = FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?;
    let mut d = st.clone_htod(&c)?;
    fwd.execute_inplace(st, &mut d)?;
    Ok(st.clone_dtoh(&d)?)
}

/// a complex grid mirrored k → −k (FFT order)
pub fn mirror_cx(lam: &[C32], dims: [usize; 3]) -> Vec<C32> {
    let [nx, ny, nz] = dims;
    let m = |i: usize, n: usize| (n - i) % n;
    (0..nx * ny * nz).map(|q| { let (i, j, k) = (q % nx, (q / nx) % ny, q / (nx * ny)); lam[m(i, nx) + nx * (m(j, ny) + ny * m(k, nz))] }).collect()
}

/// conventions of Λ found by the operator check (`SurDev::calibrate` in the binary): mirror k → −k, use the
/// transposed pair (λ_{kk'} ↔ conj λ_{k'k}), real scale
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurConv {
    pub mirror: bool,
    pub transpose: bool,
    /// the maps enter conjugated (B ↔ conj B: the operator's phase sign convention)
    pub conj_maps: bool,
    pub scale: f64,
}

impl Default for SurConv {
    fn default() -> Self {
        Self { mirror: false, transpose: false, conj_maps: false, scale: 1.0 }
    }
}

/// (pair index, conjugate?) of Λ_{kk'} under the transpose convention
fn lam_of(k: usize, kp: usize, ls: usize, transpose: bool) -> (usize, bool) {
    if k == kp {
        (pidx(k, k, ls), false)
    } else if k < kp {
        (pidx(k, kp, ls), transpose)
    } else {
        (pidx(kp, k, ls), !transpose)
    }
}

/// Host reference of the surrogate (tests; small grids): y = scale Σ_c S_cᴴ Σ_{kk'} B_kᴴ C_{kk'} B_{k'} S_c x + μ x
pub struct SurHost {
    pub fft: Arc<Fft3>,
    pub ls: usize,
    pub maps: Vec<Vec<C32>>,
    pub lam: Vec<Vec<C32>>,
    pub smaps: Vec<Vec<C32>>,
    pub mu: f64,
    pub conv: SurConv,
}

fn mulc(a: &[C32], b: &[C32], conj_b: bool) -> Vec<C32> {
    a.iter().zip(b).map(|(x, y)| { let yi = if conj_b { -y.im } else { y.im }; C32::new(x.re * y.re - x.im * yi, x.re * yi + x.im * y.re) }).collect()
}

impl SurHost {
    pub fn apply(&self, x: &[C32], y: &mut [C32]) {
        let nv = x.len();
        let lam: Vec<Vec<C32>> = if self.conv.mirror { self.lam.iter().map(|l| mirror_cx(l, self.fft.dims)).collect() } else { self.lam.clone() };
        y.iter_mut().for_each(|v| *v = C32::new(0.0, 0.0));
        for s in &self.smaps {
            let t1 = mulc(x, s, false);
            let cm = self.conv.conj_maps;
            let u: Vec<Vec<C32>> = self.maps.iter().map(|b| { let mut u = mulc(&t1, b, cm); self.fft.forward(&mut u); u }).collect();
            let mut t2 = vec![C32::new(0.0, 0.0); nv];
            for k in 0..self.ls {
                let mut v = vec![C32::new(0.0, 0.0); nv];
                for kp in 0..self.ls {
                    let (p, cj) = lam_of(k, kp, self.ls, self.conv.transpose);
                    let w = mulc(&u[kp], &lam[p], cj);
                    v.iter_mut().zip(&w).for_each(|(a, b)| { a.re += b.re; a.im += b.im; });
                }
                self.fft.inverse(&mut v);
                let w = mulc(&v, &self.maps[k], !cm);
                t2.iter_mut().zip(&w).for_each(|(a, b)| { a.re += b.re; a.im += b.im; });
            }
            let w = mulc(&t2, s, true);
            y.iter_mut().zip(&w).for_each(|(a, b)| { a.re += b.re; a.im += b.im; });
        }
        let (sc, mu) = (self.conv.scale as f32, self.mu as f32);
        y.iter_mut().zip(x).for_each(|(a, b)| { a.re = sc * a.re + mu * b.re; a.im = sc * a.im + mu * b.im; });
    }
}

/// K steps of PCG on A z = r from z = 0 with metric M (host reference): a nonlinear, variable map r → z (the outer
/// solve must be flexible)
pub fn inner_pcg_host(a: &mut dyn FnMut(&[C32], &mut [C32]), m: &mut dyn FnMut(&[C32], &mut [C32]), r: &[C32], z: &mut [C32], k: usize) {
    let n = r.len();
    z.iter_mut().for_each(|v| *v = C32::new(0.0, 0.0));
    let mut rr = r.to_vec();
    let mut s = vec![C32::new(0.0, 0.0); n];
    m(&rr, &mut s);
    let mut p = s.clone();
    let mut rs = rdot(&rr, &s);
    let mut q = vec![C32::new(0.0, 0.0); n];
    for it in 0..k {
        a(&p, &mut q);
        let pq = rdot(&p, &q);
        if !(pq > 0.0) || rs == 0.0 {
            break;
        }
        let al = (rs / pq) as f32;
        z.iter_mut().zip(&p).for_each(|(zi, pi)| { zi.re += al * pi.re; zi.im += al * pi.im; });
        rr.iter_mut().zip(&q).for_each(|(ri, qi)| { ri.re -= al * qi.re; ri.im -= al * qi.im; });
        if it + 1 == k {
            break;
        }
        m(&rr, &mut s);
        let rsn = rdot(&rr, &s);
        let be = (rsn / rs) as f32;
        rs = rsn;
        p.iter_mut().zip(&s).for_each(|(pi, si)| { pi.re = si.re + be * pi.re; pi.im = si.im + be * pi.im; });
    }
}

/// K-step preconditioned Chebyshev iteration on A z = r from z = 0 with the spectral interval [lo, hi] of M A (host
/// reference): z = q_K(MA) M r, a FIXED linear operator; symmetric, and positive definite whenever hi ≥ λ_max(MA) and
/// 0 < lo < hi (the residual polynomial lies in (0, 1) on (0, hi]). K − 1 applications of A, K of M.
pub fn cheb_host(a: &mut dyn FnMut(&[C32], &mut [C32]), m: &mut dyn FnMut(&[C32], &mut [C32]), r: &[C32], z: &mut [C32], k: usize, lo: f64, hi: f64) {
    let n = r.len();
    let (th, de) = (0.5 * (hi + lo), 0.5 * (hi - lo));
    let sig = th / de;
    let mut rho = 1.0 / sig;
    z.iter_mut().for_each(|v| *v = C32::new(0.0, 0.0));
    let mut res = r.to_vec();
    let mut mr = vec![C32::new(0.0, 0.0); n];
    m(&res, &mut mr);
    let mut d: Vec<C32> = mr.iter().map(|v| C32::new(v.re / th as f32, v.im / th as f32)).collect();
    let mut ad = vec![C32::new(0.0, 0.0); n];
    for it in 0..k {
        z.iter_mut().zip(&d).for_each(|(zi, di)| { zi.re += di.re; zi.im += di.im; });
        if it + 1 == k {
            break;
        }
        a(&d, &mut ad);
        res.iter_mut().zip(&ad).for_each(|(ri, ai)| { ri.re -= ai.re; ri.im -= ai.im; });
        let rn = 1.0 / (2.0 * sig - rho);
        m(&res, &mut mr);
        let (c1, c2) = ((rn * rho) as f32, (2.0 * rn / de) as f32);
        d.iter_mut().zip(&mr).for_each(|(di, mi)| { di.re = c1 * di.re + c2 * mi.re; di.im = c1 * di.im + c2 * mi.im; });
        rho = rn;
    }
}

/// inner solver of the surrogate preconditioner
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inner {
    /// K PCG steps (variable: flexible outer PCG)
    Pcg(usize),
    /// K-step Chebyshev on [hi / ratio, hi] (fixed SPD)
    Cheb(usize),
}

struct Ws {
    rr: CudaSlice<C32>,
    s: CudaSlice<C32>,
    p: CudaSlice<C32>,
    q: CudaSlice<C32>,
}

/// Device surrogate of one encode (G3)
pub struct SurDev {
    pub ls: usize,
    pub nv: usize,
    pub dims: [usize; 3],
    maps: Vec<CudaSlice<C32>>,
    lam: Vec<CudaSlice<C32>>,
    /// host copy of the symbols as built (re-uploaded on a convention change)
    pub lam_host: Vec<Vec<C32>>,
    smaps: Vec<CudaSlice<C32>>,
    pub mu: f64,
    pub conv: SurConv,
    /// Chebyshev interval [lo, hi] of M_in Q̃ (set by [`SurDev::bounds`])
    pub cheb: Option<(f64, f64)>,
    fwd: FftPlan<f32>,
    inv: FftPlan<f32>,
    u: Vec<CudaSlice<C32>>,
    v: Vec<CudaSlice<C32>>,
    t1: CudaSlice<C32>,
    t2: CudaSlice<C32>,
    ws: Option<Ws>,
    kern: Arc<Kernels>,
    /// Q̃ applications so far (cost accounting)
    pub applies: usize,
}

impl SurDev {
    /// `maps`: B'_k (ls × nv), `lam`: the pair symbols (upper triangle, [`pidx`]), `smaps`: the coils used
    pub fn new(dev: &Device, st: &Arc<CudaStream>, kern: Arc<Kernels>, dims: [usize; 3], maps: &[Vec<C32>], lam: Vec<Vec<C32>>, smaps: &[&[C32]], mu: f64) -> Result<Self, Error> {
        let nv = dims[0] * dims[1] * dims[2];
        let ls = maps.len();
        assert_eq!(lam.len(), ls * (ls + 1) / 2);
        let sdims = [dims[2], dims[1], dims[0]];
        let up = |v: &[C32]| st.clone_htod(v);
        Ok(Self {
            ls,
            nv,
            dims,
            maps: maps.iter().map(|m| up(&m[..])).collect::<Result<_, _>>()?,
            lam: lam.iter().map(|m| up(&m[..])).collect::<Result<_, _>>()?,
            lam_host: lam,
            smaps: smaps.iter().map(|m| up(*m)).collect::<Result<_, _>>()?,
            mu,
            conv: SurConv::default(),
            cheb: None,
            fwd: FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?,
            inv: FftPlan::<f32>::new(dev, &sdims, Direction::Inverse, Norm::Inverse)?,
            u: (0..ls).map(|_| st.alloc_zeros(nv)).collect::<Result<_, _>>()?,
            v: (0..ls).map(|_| st.alloc_zeros(nv)).collect::<Result<_, _>>()?,
            t1: st.alloc_zeros(nv)?,
            t2: st.alloc_zeros(nv)?,
            ws: Some(Ws { rr: st.alloc_zeros(nv)?, s: st.alloc_zeros(nv)?, p: st.alloc_zeros(nv)?, q: st.alloc_zeros(nv)? }),
            kern,
            applies: 0,
        })
    }
    pub fn coils(&self) -> usize {
        self.smaps.len()
    }
    pub fn device_bytes(&self) -> usize {
        (self.maps.len() + self.lam.len() + self.smaps.len() + 2 * self.ls + 6) * self.nv * 8
    }
    /// set the conventions (mirror / transpose re-upload Λ; scale is applied in `apply`)
    pub fn set_conv(&mut self, st: &Arc<CudaStream>, conv: SurConv) -> Result<(), Error> {
        if conv.mirror != self.conv.mirror {
            for (d, h) in self.lam.iter_mut().zip(&self.lam_host) {
                let v = if conv.mirror { mirror_cx(h, self.dims) } else { h.clone() };
                st.memcpy_htod(&v, d)?;
            }
        }
        self.conv = conv;
        Ok(())
    }
    /// y = Q̃ x
    pub fn apply(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (nv, ls) = (self.nv, self.ls);
        let k = self.kern.clone();
        st.memset_zeros(y)?;
        for c in 0..self.smaps.len() {
            k.sample_weight(st, x, &self.smaps[c], &mut self.t1, nv, 1, false, false)?;
            for l in 0..ls {
                k.sample_weight(st, &self.t1, &self.maps[l], &mut self.u[l], nv, 1, self.conv.conj_maps, false)?;
                self.fwd.execute_inplace(st, &mut self.u[l])?;
            }
            for l in 0..ls {
                for lp in 0..ls {
                    let (pi, cj) = lam_of(l, lp, ls, self.conv.transpose);
                    k.sample_weight(st, &self.u[lp], &self.lam[pi], &mut self.v[l], nv, 1, cj, lp > 0)?;
                }
                self.inv.execute_inplace(st, &mut self.v[l])?;
            }
            for l in 0..ls {
                k.sample_weight(st, &self.v[l], &self.maps[l], &mut self.t2, nv, 1, !self.conv.conj_maps, l > 0)?;
            }
            k.sample_weight(st, &self.t2, &self.smaps[c], y, nv, 1, true, true)?;
        }
        if self.conv.scale != 1.0 || self.mu != 0.0 {
            // y = scale y + μ x
            k.axpby(st, Complex64::new(self.mu, 0.0), x, Complex64::new(self.conv.scale, 0.0), y)?;
        }
        self.applies += 1;
        Ok(())
    }
    fn metric(&self, st: &Arc<CudaStream>, m: &mut Option<&mut CirculantPrecond>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        match m {
            Some(p) => p.apply(st, r, z),
            None => Ok(st.memcpy_dtod(r, z)?),
        }
    }
    fn rd(&self, st: &Arc<CudaStream>, a: &CudaSlice<C32>, b: &CudaSlice<C32>) -> Result<f64, Error> {
        Ok(self.kern.dot(st, a, b)?.re)
    }
    /// z ≈ Q̃⁻¹ r by the inner solver, metric `m` (None: identity)
    pub fn solve(&mut self, st: &Arc<CudaStream>, mut m: Option<&mut CirculantPrecond>, inner: Inner, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut w = self.ws.take().expect("SurDev workspace");
        let k = self.kern.clone();
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        st.memset_zeros(z)?;
        st.memcpy_dtod(r, &mut w.rr)?;
        let res = (|| -> Result<(), Error> {
            match inner {
                Inner::Pcg(kk) => {
                    self.metric(st, &mut m, &w.rr, &mut w.s)?;
                    st.memcpy_dtod(&w.s, &mut w.p)?;
                    let mut rs = self.rd(st, &w.rr, &w.s)?;
                    for it in 0..kk {
                        self.apply(st, &w.p, &mut w.q)?;
                        let pq = self.rd(st, &w.p, &w.q)?;
                        if !(pq > 0.0) || rs == 0.0 {
                            break;
                        }
                        let al = rs / pq;
                        k.axpby(st, Complex64::new(al, 0.0), &w.p, one, z)?;
                        k.axpby(st, Complex64::new(-al, 0.0), &w.q, one, &mut w.rr)?;
                        if it + 1 == kk {
                            break;
                        }
                        self.metric(st, &mut m, &w.rr, &mut w.s)?;
                        let rsn = self.rd(st, &w.rr, &w.s)?;
                        let be = rsn / rs;
                        rs = rsn;
                        k.axpby(st, one, &w.s, Complex64::new(be, 0.0), &mut w.p)?;
                    }
                }
                Inner::Cheb(kk) => {
                    let (lo, hi) = self.cheb.expect("SurDev::solve Cheb: call bounds() first");
                    let (th, de) = (0.5 * (hi + lo), 0.5 * (hi - lo));
                    let sig = th / de;
                    let mut rho = 1.0 / sig;
                    // d (in p) = M r / θ
                    self.metric(st, &mut m, &w.rr, &mut w.s)?;
                    k.axpby(st, Complex64::new(1.0 / th, 0.0), &w.s, zero, &mut w.p)?;
                    for it in 0..kk {
                        k.axpby(st, one, &w.p, one, z)?;
                        if it + 1 == kk {
                            break;
                        }
                        self.apply(st, &w.p, &mut w.q)?;
                        k.axpby(st, Complex64::new(-1.0, 0.0), &w.q, one, &mut w.rr)?;
                        let rn = 1.0 / (2.0 * sig - rho);
                        self.metric(st, &mut m, &w.rr, &mut w.s)?;
                        k.axpby(st, Complex64::new(2.0 * rn / de, 0.0), &w.s, Complex64::new(rn * rho, 0.0), &mut w.p)?;
                        rho = rn;
                    }
                }
            }
            Ok(())
        })();
        self.ws = Some(w);
        res
    }
    /// the Chebyshev interval of M Q̃: hi = 1.05 × (Lanczos θ + residual bound), `steps` steps; lo = hi / ratio
    pub fn bounds(&mut self, st: &Arc<CudaStream>, m: Option<&mut CirculantPrecond>, steps: usize, ratio: f64) -> Result<(f64, f64), Error> {
        let est = {
            let mut sp = LzSpace { s: self, m, st: st.clone(), p: None, pp: None, q: None, w: None };
            crate::lanczos::lanczos(&mut sp, steps)?
        };
        let hi = 1.05 * est.hat();
        let lo = hi / ratio.max(1.0 + 1e-6);
        self.cheb = Some((lo, hi));
        Ok((lo, hi))
    }
}

/// p-form Lanczos space of M Q̃ on the device (`crate::lanczos` module doc)
struct LzSpace<'a, 'b> {
    s: &'a mut SurDev,
    m: Option<&'b mut CirculantPrecond>,
    st: Arc<CudaStream>,
    p: Option<CudaSlice<C32>>,
    pp: Option<CudaSlice<C32>>,
    q: Option<CudaSlice<C32>>,
    w: Option<CudaSlice<C32>>,
}

impl crate::lanczos::LanczosSpace for LzSpace<'_, '_> {
    type E = Error;
    fn start(&mut self) -> Result<f64, Error> {
        let nv = self.s.nv;
        let p0 = self.st.clone_htod(&crate::ppre::phases(nv, 0x1A2C))?;
        let mut q0 = self.st.alloc_zeros::<C32>(nv)?;
        self.s.metric(&self.st, &mut self.m, &p0, &mut q0)?;
        let n2 = self.s.rd(&self.st, &p0, &q0)?;
        self.pp = Some(self.st.alloc_zeros(nv)?);
        self.w = Some(self.st.alloc_zeros(nv)?);
        self.p = Some(p0);
        self.q = Some(q0);
        Ok(n2)
    }
    fn scale(&mut self, c: f64) -> Result<(), Error> {
        let zc = Complex64::new(0.0, 0.0);
        let k = self.s.kern.clone();
        let (p, q) = (self.p.as_mut().unwrap(), self.q.as_mut().unwrap());
        let pc = p.clone();
        k.axpby(&self.st, zc, &pc, Complex64::new(c, 0.0), p)?;
        let qc = q.clone();
        k.axpby(&self.st, zc, &qc, Complex64::new(c, 0.0), q)?;
        Ok(())
    }
    fn apply(&mut self, ) -> Result<f64, Error> {
        let (q, w) = (self.q.as_ref().unwrap(), self.w.as_mut().unwrap());
        self.s.apply(&self.st, q, w)?;
        self.s.rd(&self.st, q, w)
    }
    fn advance(&mut self, alpha: f64, beta: f64) -> Result<f64, Error> {
        let k = self.s.kern.clone();
        let one = Complex64::new(1.0, 0.0);
        // s = w − α p − β p_prev (into w)
        {
            let (w, p, pp) = (self.w.as_mut().unwrap(), self.p.as_ref().unwrap(), self.pp.as_ref().unwrap());
            k.axpby(&self.st, Complex64::new(-alpha, 0.0), p, one, w)?;
            k.axpby(&self.st, Complex64::new(-beta, 0.0), pp, one, w)?;
        }
        // p_prev ← p, p ← s, q ← M s; w is reused as the next product buffer (the old p_prev)
        let s_vec = self.w.take().unwrap();
        let old_pp = self.pp.take().unwrap();
        let old_p = self.p.take().unwrap();
        let mut q = self.q.take().unwrap();
        self.s.metric(&self.st, &mut self.m, &s_vec, &mut q)?;
        let n2 = self.s.rd(&self.st, &s_vec, &q)?;
        self.pp = Some(old_p);
        self.p = Some(s_vec);
        self.q = Some(q);
        self.w = Some(old_pp);
        Ok(n2)
    }
}

/// the pair symbols of one encode on the device (G3 setup): the pooled samples (coords, times) of the encode's sets,
/// the stored tables, the compression to `ls` maps. Returns (maps B'_k, symbols Λ (upper triangle), kept-energy loss).
#[allow(clippy::too_many_arguments)]
pub fn build_symbols(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3], tabs: &SegHostTables, coords: [&[f32]; 3], taus: &[f64], voxel_w: &[f32], ls: usize) -> Result<(Vec<Vec<C32>>, Vec<Vec<C32>>, f64), Error> {
    let nv = dims[0] * dims[1] * dims[2];
    let (maps, nmix, loss) = compress(tabs, nv, ls, voxel_w, taus);
    let ls = maps.len();
    let w = pair_weights(tabs, &nmix, ls, taus);
    let lam = w.iter().map(|wp| chan_symbol_dev(dev, st, dims, coords, wp)).collect::<Result<Vec<_>, _>>()?;
    Ok((maps, lam, loss))
}

/// Hermitian check helper for tests: max |⟨a, Q̃b⟩ − ⟨Q̃a, b⟩| / (‖a‖ ‖Q̃b‖)
pub fn herm_err(apply: &mut dyn FnMut(&[C32], &mut [C32]), a: &[C32], b: &[C32]) -> f64 {
    let (mut qa, mut qb) = (vec![C32::new(0.0, 0.0); a.len()], vec![C32::new(0.0, 0.0); b.len()]);
    apply(a, &mut qa);
    apply(b, &mut qb);
    (cdot(a, &qb) - cdot(&qa, b)).norm() / (rdot(a, a).sqrt() * rdot(&qb, &qb).sqrt()).max(1e-300)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hoststack::HostStack;
    use crate::ppre::{pcg_groups, PcgCfg, PcgOps};

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
    fn rel(a: &[C32], b: &[C32]) -> f64 {
        let d: Vec<C32> = a.iter().zip(b).map(|(x, y)| C32::new(x.re - y.re, x.im - y.im)).collect();
        rdot(&d, &d).sqrt() / rdot(b, b).sqrt().max(1e-300)
    }
    /// a tiny problem: grid 6 × 4 × 3, 2 coils, 3 stored segments with a field, M samples on a spiral-ish set
    struct Toy {
        dims: [usize; 3],
        nv: usize,
        smaps: Vec<Vec<C32>>,
        tabs: SegHostTables,
        coords: [Vec<f32>; 3],
        taus: Vec<f64>,
    }
    fn toy() -> Toy {
        let dims = [6usize, 4, 3];
        let nv = 72;
        let smaps: Vec<Vec<C32>> = (0..2).map(|c| (0..nv).map(|v| C32::new(0.6 + 0.3 * ((v + 3 * c) as f32 * 0.37).sin(), 0.2 * ((v * (c + 1)) as f32 * 0.21).cos())).collect()).collect();
        // field ω(v) and stored segments: fixed-time diagonals e^{-iω τ_l}, linear-hat interpolators on a table
        let om: Vec<f64> = (0..nv).map(|v| 300.0 * ((v % 6) as f64 - 2.5) + 80.0 * ((v / 6) % 4) as f64).collect();
        let (t0, t1, l, ntab) = (1e-3, 9e-3, 3usize, 41usize);
        let tl: Vec<f64> = (0..l).map(|i| t0 + (t1 - t0) * i as f64 / (l - 1) as f64).collect();
        let (tlr, omr) = (&tl, &om);
        let segph: Vec<C32> = (0..l).flat_map(|i| omr.iter().map(move |&w| { let p = -w * tlr[i]; C32::new(p.cos() as f32, p.sin() as f32) }).collect::<Vec<_>>()).collect();
        let dt = (t1 - t0) / (ntab - 1) as f64;
        let mut atab = vec![C32::new(0.0, 0.0); ntab * l];
        for i in 0..ntab {
            let t = t0 + i as f64 * dt;
            let u = (t - t0) / (tl[1] - tl[0]);
            let j = (u.floor() as usize).min(l - 2);
            let fr = (u - j as f64) as f32;
            atab[i * l + j] = C32::new(1.0 - fr, 0.0);
            atab[i * l + j + 1] = C32::new(fr, 0.0);
        }
        let tabs = SegHostTables { segph, atab, nseg: l, ntab, t0, dt };
        let m = 400;
        let mut coords = [vec![0f32; m], vec![0f32; m], vec![0f32; m]];
        let mut taus = vec![0f64; m];
        for i in 0..m {
            let s = i as f64 / m as f64;
            let r = 2.8 * (1.0 - (2.0 * s - 1.0).abs());
            let a = 37.0 * s;
            coords[0][i] = (r * a.cos()) as f32;
            coords[1][i] = (r * a.sin()) as f32;
            coords[2][i] = (0.9 * (13.0 * s).sin()) as f32;
            taus[i] = t0 + (t1 - t0) * s;
        }
        Toy { dims, nv, smaps, tabs, coords, taus }
    }
    /// dense surrogate from the definition: Σ_c S_cᴴ Σ_{kk'} B_kᴴ C_{kk'} B_k' S_c with C = F⁻¹ diag(λ) F by columns
    fn dense_apply(t: &Toy, maps: &[Vec<C32>], lam: &[Vec<C32>], x: &[C32]) -> Vec<C32> {
        let fft = Fft3::new(t.dims);
        let ls = maps.len();
        let mut y = vec![C32::new(0.0, 0.0); t.nv];
        for s in &t.smaps {
            for k in 0..ls {
                for kp in 0..ls {
                    let lkk: Vec<C32> = if k <= kp { lam[pidx(k, kp, ls)].clone() } else { lam[pidx(kp, k, ls)].iter().map(|z| C32::new(z.re, -z.im)).collect() };
                    let mut u = mulc(&mulc(x, s, false), &maps[kp], false);
                    fft.forward(&mut u);
                    let mut w = mulc(&u, &lkk, false);
                    fft.inverse(&mut w);
                    let o = mulc(&mulc(&w, &maps[k], true), s, true);
                    y.iter_mut().zip(&o).for_each(|(a, b)| { a.re += b.re; a.im += b.im; });
                }
            }
        }
        y
    }
    fn host_sur(t: &Toy, ls: usize, mu: f64) -> (SurHost, Vec<Vec<C32>>, Vec<Vec<C32>>) {
        let dw: Vec<f32> = (0..t.nv).map(|v| t.smaps.iter().map(|s| s[v].re * s[v].re + s[v].im * s[v].im).sum()).collect();
        let (maps, nmix, _) = compress(&t.tabs, t.nv, ls, &dw, &t.taus);
        let w = pair_weights(&t.tabs, &nmix, maps.len(), &t.taus);
        let c = [&t.coords[0][..], &t.coords[1][..], &t.coords[2][..]];
        let lam: Vec<Vec<C32>> = w.iter().map(|wp| chan_symbol_host(t.dims, c, wp, false)).collect();
        let sh = SurHost { fft: Arc::new(Fft3::new(t.dims)), ls: maps.len(), maps: maps.clone(), lam: lam.clone(), smaps: t.smaps.clone(), mu, conv: SurConv::default() };
        (sh, maps, lam)
    }

    /// the surrogate against the EXACT normal operator E^H E of the toy (dense, from e^{-ik·u} and the stored
    /// decomposition g(v, τ) = Σ_l B_l(v) a_l(τ)), not against its own definition: L_s = 1 (plain-like), 2, 3
    #[test]
    fn surrogate_vs_exact_normal_operator() {
        let t = toy();
        let (nv, m) = (t.nv, t.taus.len());
        let dims = t.dims;
        let cen = |i: usize, n: usize| i as f64 - (n / 2) as f64;
        let pos: Vec<[f64; 3]> = (0..nv).map(|q| [cen(q % dims[0], dims[0]), cen((q / dims[0]) % dims[1], dims[1]), cen(q / (dims[0] * dims[1]), dims[2])]).collect();
        let mut a = vec![Complex64::new(0.0, 0.0); t.tabs.nseg];
        // E_c[i, v] = e^{-i k_i·u_v} g(v, τ_i) S_c(v)
        let exact = |x: &[C32]| -> Vec<C32> {
            let mut y = vec![Complex64::new(0.0, 0.0); nv];
            let mut a = vec![Complex64::new(0.0, 0.0); t.tabs.nseg];
            for s in &t.smaps {
                for i in 0..m {
                    t.tabs.interp(t.taus[i], &mut a);
                    let g = |v: usize| -> Complex64 { (0..t.tabs.nseg).map(|l| { let b = t.tabs.segph[l * nv + v]; Complex64::new(b.re as f64, b.im as f64) * a[l] }).sum() };
                    let e = |v: usize| -> Complex64 {
                        let ph = -(t.coords[0][i] as f64 * pos[v][0] + t.coords[1][i] as f64 * pos[v][1] + t.coords[2][i] as f64 * pos[v][2]);
                        Complex64::from_polar(1.0, ph) * g(v) * Complex64::new(s[v].re as f64, s[v].im as f64)
                    };
                    let row: Vec<Complex64> = (0..nv).map(e).collect();
                    let yi: Complex64 = (0..nv).map(|v| row[v] * Complex64::new(x[v].re as f64, x[v].im as f64)).sum();
                    for v in 0..nv {
                        y[v] += row[v].conj() * yi;
                    }
                }
            }
            y.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
        };
        let _ = &mut a;
        let x = rnd(nv, 21);
        let nx = exact(&x);
        for ls in [1usize, 2, 3] {
            let (sh, _, _) = host_sur(&t, ls, 0.0);
            let mut best = (f64::MAX, 0.0, SurConv::default());
            for (mirror, transpose, conj_maps) in [(false, false, false), (true, false, false), (false, true, false), (true, true, false), (false, false, true), (true, false, true), (false, true, true), (true, true, true)] {
                let mut s2 = SurHost { conv: SurConv { mirror, transpose, conj_maps, scale: 1.0 }, fft: sh.fft.clone(), ls: sh.ls, maps: sh.maps.clone(), lam: sh.lam.clone(), smaps: sh.smaps.clone(), mu: 0.0 };
                let mut y = vec![C32::new(0.0, 0.0); nv];
                s2.apply(&x, &mut y);
                let c = rdot(&y, &nx) / rdot(&y, &y).max(1e-300);
                let d: Vec<C32> = y.iter().zip(&nx).map(|(p, q)| C32::new(c as f32 * p.re - q.re, c as f32 * p.im - q.im)).collect();
                let err = rdot(&d, &d).sqrt() / rdot(&nx, &nx).sqrt();
                if err < best.0 {
                    s2.conv.scale = c;
                    best = (err, c, s2.conv);
                }
            }
            eprintln!("L_s {ls}: ‖(N − cÑ)x‖/‖Nx‖ = {:.3}, c {:.3e}, {:?}", best.0, best.1, best.2);
        }
    }

    #[test]
    fn pidx_is_a_bijection() {
        for ls in 1..5 {
            let mut seen = vec![false; ls * (ls + 1) / 2];
            for k in 0..ls {
                for kp in k..ls {
                    assert!(!seen[pidx(k, kp, ls)]);
                    seen[pidx(k, kp, ls)] = true;
                }
            }
            assert!(seen.iter().all(|&b| b));
        }
    }

    #[test]
    fn compression_full_rank_is_exact() {
        let t = toy();
        let dw = vec![1f32; t.nv];
        let (maps, nmix, loss) = compress(&t.tabs, t.nv, 3, &dw, &t.taus);
        assert!(loss < 1e-9, "{loss}");
        let mut a = vec![Complex64::new(0.0, 0.0); 3];
        let mut am = vec![Complex64::new(0.0, 0.0); 3];
        for &tau in &[1.3e-3, 4.4e-3, 8.7e-3] {
            interp_mixed(&t.tabs, &nmix, 3, tau, &mut a, &mut am);
            for v in [0usize, 17, 40, 71] {
                let g: Complex64 = (0..3).map(|l| { let b = t.tabs.segph[l * t.nv + v]; Complex64::new(b.re as f64, b.im as f64) * a[l] }).sum();
                let g2: Complex64 = (0..3).map(|k| { let b = maps[k][v]; Complex64::new(b.re as f64, b.im as f64) * am[k] }).sum();
                assert!((g - g2).norm() < 1e-4, "{g} vs {g2}");
            }
        }
        // truncation keeps the dominant part
        let (_, _, l2) = compress(&t.tabs, t.nv, 2, &dw, &t.taus);
        assert!(l2 > 0.0 && l2 < 0.9, "{l2}");
    }

    #[test]
    fn surrogate_apply_matches_dense_and_is_hermitian_psd() {
        let t = toy();
        let (sh, maps, lam) = host_sur(&t, 2, 0.0);
        let x = rnd(t.nv, 3);
        let mut y = vec![C32::new(0.0, 0.0); t.nv];
        sh.apply(&x, &mut y);
        let yd = dense_apply(&t, &maps, &lam, &x);
        assert!(rel(&y, &yd) < 1e-5, "{}", rel(&y, &yd));
        let mut ap = |a: &[C32], b: &mut [C32]| sh.apply(a, b);
        let (a, b) = (rnd(t.nv, 4), rnd(t.nv, 5));
        assert!(herm_err(&mut ap, &a, &b) < 1e-5);
        // PSD: dense eigenvalues ≥ −1e-5 λ_max
        let n = t.nv;
        let mut qm = DMatrix::<Complex<f64>>::zeros(n, n);
        for j in 0..n {
            let mut e = vec![C32::new(0.0, 0.0); n];
            e[j] = C32::new(1.0, 0.0);
            let mut col = vec![C32::new(0.0, 0.0); n];
            sh.apply(&e, &mut col);
            for i in 0..n {
                qm[(i, j)] = Complex::new(col[i].re as f64, col[i].im as f64);
            }
        }
        let ev = nalgebra::SymmetricEigen::new((&qm + qm.adjoint()) * Complex::new(0.5, 0.0)).eigenvalues;
        let (mn, mx) = (ev.iter().cloned().fold(f64::MAX, f64::min), ev.iter().cloned().fold(0.0, f64::max));
        assert!(mn > -1e-5 * mx, "Q̃ not PSD: λ_min {mn} λ_max {mx}");
    }

    #[test]
    fn chan_symbol_of_unit_weights_is_a_fejer_density() {
        let t = toy();
        let c = [&t.coords[0][..], &t.coords[1][..], &t.coords[2][..]];
        let lam = chan_symbol_host(t.dims, c, &vec![C32::new(1.0, 0.0); t.taus.len()], false);
        let mean = lam.iter().map(|z| z.re as f64).sum::<f64>() / t.nv as f64;
        // Σ_m F(θ − 2πm/N) = N per axis → the mean symbol over modes is M (sample count)
        assert!((mean - t.taus.len() as f64).abs() < 1e-3 * t.taus.len() as f64, "{mean}");
        assert!(lam.iter().all(|z| z.re >= -1e-3 && z.im.abs() < 1e-3));
    }

    #[test]
    fn recentre_diag_is_unit_modulus() {
        let t = toy();
        let u = recentre_diag(&t.tabs, t.nv, 5e-3);
        assert!(u.iter().all(|z| ((z.re * z.re + z.im * z.im).sqrt() - 1.0).abs() < 1e-5));
        // at a node time it is conj of the stored diagonal's phase: e^{+iω τ}
        let u0 = recentre_diag(&t.tabs, t.nv, 1e-3);
        let b = t.tabs.segph[7];
        assert!((u0[7].re - b.re).abs() < 1e-4 && (u0[7].im + b.im).abs() < 1e-4);
    }

    #[test]
    fn chebyshev_operator_is_spd_and_pcg_inner_converges() {
        let t = toy();
        let (sh, _, _) = host_sur(&t, 2, 1e-2);
        let n = t.nv;
        // λ_max of Q̃ by dense eigen (for the bound) and the Chebyshev operator by columns
        let mut qm = DMatrix::<Complex<f64>>::zeros(n, n);
        let mut cm = DMatrix::<Complex<f64>>::zeros(n, n);
        for j in 0..n {
            let mut e = vec![C32::new(0.0, 0.0); n];
            e[j] = C32::new(1.0, 0.0);
            let mut col = vec![C32::new(0.0, 0.0); n];
            sh.apply(&e, &mut col);
            for i in 0..n {
                qm[(i, j)] = Complex::new(col[i].re as f64, col[i].im as f64);
            }
        }
        let lmax = nalgebra::SymmetricEigen::new((&qm + qm.adjoint()) * Complex::new(0.5, 0.0)).eigenvalues.iter().cloned().fold(0.0, f64::max);
        for j in 0..n {
            let mut e = vec![C32::new(0.0, 0.0); n];
            e[j] = C32::new(1.0, 0.0);
            let mut z = vec![C32::new(0.0, 0.0); n];
            cheb_host(&mut |a, b| sh.apply(a, b), &mut |a, b| b.copy_from_slice(a), &e, &mut z, 8, 1.05 * lmax / 50.0, 1.05 * lmax);
            for i in 0..n {
                cm[(i, j)] = Complex::new(z[i].re as f64, z[i].im as f64);
            }
        }
        assert!((&cm - cm.adjoint()).norm() < 1e-4 * cm.norm(), "Chebyshev operator not Hermitian");
        let ev = nalgebra::SymmetricEigen::new((&cm + cm.adjoint()) * Complex::new(0.5, 0.0)).eigenvalues;
        assert!(ev.iter().all(|&v| v > 0.0), "Chebyshev operator not PD: min {}", ev.iter().cloned().fold(f64::MAX, f64::min));
        // inner PCG with many steps solves Q̃ z = r
        let r = rnd(n, 9);
        let mut z = vec![C32::new(0.0, 0.0); n];
        inner_pcg_host(&mut |a, b| sh.apply(a, b), &mut |a, b| b.copy_from_slice(a), &r, &mut z, 200);
        let mut qz = vec![C32::new(0.0, 0.0); n];
        sh.apply(&z, &mut qz);
        assert!(rel(&qz, &r) < 1e-3, "{}", rel(&qz, &r));
    }

    /// flexible outer PCG on a "true" Q (the Toeplitz-like dense operator: surrogate with a perturbed symbol + field)
    /// with the inexact inner (4 PCG steps on the surrogate) reaches the same x̂ as plain CG
    #[test]
    fn flexible_pcg_with_inexact_inner_keeps_the_estimator() {
        let t = toy();
        let (sur, maps, lam) = host_sur(&t, 2, 1e-1);
        // "true" operator: full 3-segment surrogate with a perturbed symbol (stands for the true Toeplitz Q) + ridge
        let (full, _, _) = host_sur(&t, 3, 1e-1);
        let _ = (&maps, &lam);
        struct Ops<'a> {
            q: &'a SurHost,
            p: Option<&'a SurHost>,
            k: usize,
        }
        impl PcgOps for Ops<'_> {
            fn q(&mut self, v: &HostStack, out: &mut HostStack) -> Result<(), Error> {
                let (o, l) = (out.off[0], out.lens[0]);
                let mut y = vec![C32::new(0.0, 0.0); l];
                self.q.apply(v.chunk(0), &mut y);
                // a non-circulant perturbation: + 0.3 diag(1 + v mod 5) (keeps Q ≻ 0)
                for (i, yi) in y.iter_mut().enumerate() {
                    let d = 0.3 * (1.0 + (i % 5) as f32);
                    yi.re += d * v.chunk(0)[i].re;
                    yi.im += d * v.chunk(0)[i].im;
                }
                out.data[o..o + l].copy_from_slice(&y);
                Ok(())
            }
            fn p(&mut self, _g: usize, r: &[C32], z: &mut [C32]) -> Result<(), Error> {
                match self.p {
                    None => z.copy_from_slice(r),
                    Some(s) => inner_pcg_host(&mut |a, b| s.apply(a, b), &mut |a, b| b.copy_from_slice(a), r, z, self.k),
                }
                Ok(())
            }
        }
        let n = t.nv;
        let b = HostStack { data: rnd(n, 11), off: vec![0], lens: vec![n] };
        let run = |p: Option<&SurHost>, flexible: bool, iters: usize| {
            let mut x = HostStack::zeros(&[n]);
            let mut ops = Ops { q: &full, p, k: 4 };
            let o = pcg_groups(&mut ops, &b, &mut x, &PcgCfg { iters, tol: 1e-7, harvest: 0, flexible }, &mut |_, _, _| Ok(true)).unwrap();
            (x.data, o.iters)
        };
        let (x0, i0) = run(None, false, 2000);
        let (x1, i1) = run(Some(&sur), true, 2000);
        assert!(rel(&x1, &x0) < 1e-4, "flexible inner-surrogate solution differs: {}", rel(&x1, &x0));
        eprintln!("plain CG {i0} iterations, flexible surrogate-preconditioned {i1}");
        assert!(i1 <= i0, "surrogate preconditioning did not help: {i1} vs {i0}");
    }
}
