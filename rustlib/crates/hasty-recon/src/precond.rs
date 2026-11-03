//! Unbiased preconditioning of the normal equations (no density compensation in the data term).
//!
//! The coil-free normal operator F^H F is Toeplitz (kernel h on lags −N…N−1, from `hasty_toeplitz::build_kernels`
//! with unit sample weights). Its optimal circulant approximation on the N grid (T. Chan) is the triangle-windowed
//! kernel folded onto N: c(n mod N) += Π_a (1 − |n_a|/N_a) h(n). Its spectrum λ = F_N c is a Fejér-smoothed k-space
//! sampling density, hence ≥ 0. With the coil scaling D = Σ_c |S_c|²:  M = D^{-1/2} F_N⁻¹ diag(1/(λ + ε)) F_N D^{-1/2}
//! ≈ (E^H E)⁻¹. Inside PCG (or as the fixed preconditioner of P²nP) it changes convergence, not the solution.
//! (The spectrum of the 2N circulant *embedding* is not usable: it is indefinite.)

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_fft::{Direction, FftPlan, Norm};
use num_complex::Complex32;
use hasty_toeplitz::{build_kernels, plain, KernelBuildOpts, Samples};

use crate::gpu::Kernels;
use crate::Error;

pub struct CirculantPrecond {
    fwd: FftPlan<f32>,
    inv: FftPlan<f32>,
    /// 1/(λ + ε) on the N grid (real, stored complex)
    pinv: CudaSlice<C32>,
    /// λ + ε (for M⁻¹)
    lam_eps: CudaSlice<C32>,
    /// D^{1/2}
    dsq: CudaSlice<C32>,
    /// D^{-1/2}
    dinv: CudaSlice<C32>,
    tmp: CudaSlice<C32>,
    kern: Arc<Kernels>,
    pub nv: usize,
    /// λ_max / λ_min after flooring (condition number the preconditioner targets)
    pub dyn_range: f64,
    /// λ on the N grid (FFT order, x fastest), for diagnostics
    pub lambda: Vec<f32>,
}

impl CirculantPrecond {
    /// `dims`: x fastest; `coords`: radians per voxel; `smaps`: coil-major (nc × nv); `eps_rel`: floor of λ relative to its max
    pub fn new(dev: &Device, st: &Arc<CudaStream>, kern: Arc<Kernels>, dims: [usize; 3], coords: [&[f32]; 3], smaps: &[C32], nc: usize, eps_rel: f64) -> Result<Self, Error> {
        let nv = dims[0] * dims[1] * dims[2];
        let m = coords[0].len();
        let sdims = [dims[2], dims[1], dims[0]];
        let pad = [2 * dims[0], 2 * dims[1], 2 * dims[2]];
        let dc: Vec<CudaSlice<f32>> = coords.iter().map(|c| st.clone_htod(*c)).collect::<Result<_, _>>()?;
        let wt = st.clone_htod(&vec![1f32; m])?;
        let zero = st.alloc_zeros::<f32>(m)?;
        let extra: Vec<CudaSlice<C32>> = vec![];
        let samples = Samples::<f32> { m, coords: [&dc[0], &dc[1], &dc[2]], wt: &wt, w: [&zero, &zero, &zero, &zero], extra: &extra };
        let mut k = build_kernels(dev, st, sdims, &samples, &plain().kernels, &KernelBuildOpts { tol: 1e-5, sigma: 2.0 })?.remove(0);
        // kernel lags h(n) on the 2N grid (FFT order)
        let inv2 = FftPlan::<f32>::new(dev, &[pad[2], pad[1], pad[0]], Direction::Inverse, Norm::Inverse)?;
        inv2.execute_inplace(st, &mut k)?;
        let h = st.clone_dtoh(&k)?;
        drop(k);
        // triangle window + fold onto the N grid
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
                    let w = (wx * wy * wz) as f32;
                    let v = h[kx + pad[0] * (ky + pad[1] * kz)];
                    let (fx, fy, fz) = (lx.rem_euclid(dims[0] as i64) as usize, ly.rem_euclid(dims[1] as i64) as usize, lz.rem_euclid(dims[2] as i64) as usize);
                    let q = fx + dims[0] * (fy + dims[1] * fz);
                    c[q].re += w * v.re;
                    c[q].im += w * v.im;
                }
            }
        }
        let fwd = FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?;
        let inv = FftPlan::<f32>::new(dev, &sdims, Direction::Inverse, Norm::Inverse)?;
        let mut dlam = st.clone_htod(&c)?;
        fwd.execute_inplace(st, &mut dlam)?;
        let lam = st.clone_dtoh(&dlam)?;
        // Fourier-diagonal of N = Σ_c S_c^H T S_c: ⟨e_k, N e_k⟩ = (ρ ⊛ P)(k), ρ = circulant density spectrum, P = coil-map
        // power spectrum Σ_c |F S_c|² / nv. Coil weighting is then inside λ and D is set to identity below. (Inverting ρ alone
        // over-amplifies the modes next to the spiral's k = 0 spike, which the coil maps smear onto them.)
        let use_conv = std::env::var("HASTY_PRECOND_MODE").map(|v| v != "density").unwrap_or(true);
        let mut lam = lam;
        if use_conv {
            let mut pw = vec![0f32; nv];
            let mut buf = st.alloc_zeros::<C32>(nv)?;
            for c in 0..nc {
                st.memcpy_htod(&smaps[c * nv..(c + 1) * nv], &mut buf)?;
                fwd.execute_inplace(st, &mut buf)?;
                let h = st.clone_dtoh(&buf)?;
                for q in 0..nv {
                    pw[q] += (h[q].re * h[q].re + h[q].im * h[q].im) / (nv as f32 * nv as f32);
                }
            }
            // circular convolution λ ⊛ P via FFTs (both real, Hermitian-symmetric)
            let la: Vec<C32> = lam.iter().map(|z| C32::new(z.re, 0.0)).collect();
            let pa: Vec<C32> = pw.iter().map(|&v| C32::new(v, 0.0)).collect();
            let (mut dl, mut dp) = (st.clone_htod(&la)?, st.clone_htod(&pa)?);
            inv.execute_inplace(st, &mut dl)?;
            inv.execute_inplace(st, &mut dp)?;
            let (hl, hp) = (st.clone_dtoh(&dl)?, st.clone_dtoh(&dp)?);
            // conv theorem with our normalisations: IFFT(a)·IFFT(b)·nv → FFT → a⊛b
            let prod: Vec<C32> = hl.iter().zip(&hp).map(|(x, y)| {
                let z = Complex32::new(x.re, x.im) * Complex32::new(y.re, y.im) * nv as f32;
                C32::new(z.re, z.im)
            }).collect();
            let mut dc = st.clone_htod(&prod)?;
            fwd.execute_inplace(st, &mut dc)?;
            lam = st.clone_dtoh(&dc)?;
        }
        // Unsampled in-plane k-space corners (outside the trajectory's disc, |k_xy| > N/2): no data there, the diagonal is
        // only coil leakage. Floor them at the spectrum level of the sampled outer ring so M does not amplify them.
        if std::env::var("HASTY_PRECOND_NOCORNER").is_err() {
            let kr = |q: usize| -> f64 {
                let (i, j) = (q % dims[0], (q / dims[0]) % dims[1]);
                let fi = if i <= dims[0] / 2 { i as f64 } else { i as f64 - dims[0] as f64 } / (dims[0] as f64 / 2.0);
                let fj = if j <= dims[1] / 2 { j as f64 } else { j as f64 - dims[1] as f64 } / (dims[1] as f64 / 2.0);
                (fi * fi + fj * fj).sqrt()
            };
            let mut ring: Vec<f32> = (0..nv).filter(|&q| { let r = kr(q); r > 0.85 && r < 0.95 }).map(|q| lam[q].re).collect();
            if !ring.is_empty() {
                ring.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let edge = ring[ring.len() / 2];
                for q in 0..nv {
                    if kr(q) > 0.95 {
                        lam[q].re = lam[q].re.max(edge);
                    }
                }
            }
        }
        let lmax = lam.iter().map(|z| z.re).fold(0f32, f32::max);
        let eps_rel = std::env::var("HASTY_PRECOND_EPS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(eps_rel);
        let eps = (eps_rel * lmax as f64) as f32;
        let lmin = lam.iter().map(|z| z.re.max(eps)).fold(f32::MAX, f32::min);
        let pinv: Vec<C32> = lam.iter().map(|z| C32::new(1.0 / z.re.max(eps), 0.0)).collect();
        let mut d = vec![C32::new(0.0, 0.0); nv];
        let mut dmax = 0f32;
        for q in 0..nv {
            let s: f32 = (0..nc).map(|c| smaps[c * nv + q].re.powi(2) + smaps[c * nv + q].im.powi(2)).sum();
            d[q].re = s;
            dmax = dmax.max(s);
        }
        // with the coil-convolved spectrum λ already carries the mean coil power D̄: N ≈ D^{1/2} C D^{1/2} with C = λ / D̄
        let dbar = d.iter().map(|v| v.re as f64).sum::<f64>() / nv as f64;
        for v in d.iter_mut() {
            v.re = 1.0 / (v.re + 1e-3 * dmax).sqrt();
        }
        let no_d = std::env::var("HASTY_PRECOND_NOD").is_ok();
        if use_conv && no_d {
            for v in d.iter_mut() {
                v.re = 1.0;
            }
        }
        let (pinv, lam_scale) = if use_conv && no_d {
            (pinv, 1.0)
        } else if use_conv {
            (pinv.iter().map(|z| C32::new(z.re * dbar as f32, 0.0)).collect::<Vec<_>>(), 1.0 / dbar as f32)
        } else {
            (pinv, 1.0)
        };
        let lam_eps: Vec<C32> = lam.iter().map(|z| C32::new(z.re.max(eps) * lam_scale, 0.0)).collect();
        let dsq: Vec<C32> = d.iter().map(|v| C32::new(1.0 / v.re, 0.0)).collect();
        Ok(Self { fwd, inv, lam_eps: st.clone_htod(&lam_eps)?, dsq: st.clone_htod(&dsq)?, pinv: st.clone_htod(&pinv)?, dinv: st.clone_htod(&d)?, tmp: st.alloc_zeros(nv)?, kern, nv, dyn_range: (lmax / lmin) as f64, lambda: lam.iter().map(|z| z.re).collect() })
    }

    /// z = M r
    pub fn apply(&mut self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.kern.mul_real(st, r, &self.dinv, &mut self.tmp)?;
        self.fwd.execute_inplace(st, &mut self.tmp)?;
        self.kern.mul_real(st, &self.tmp, &self.pinv, z)?;
        self.inv.execute_inplace(st, z)?;
        self.kern.mul_real(st, z, &self.dinv, &mut self.tmp)?;
        st.memcpy_dtod(&self.tmp, z)?;
        Ok(())
    }

    /// z = M⁻¹ r = D^{1/2} F⁻¹ diag(λ + ε) F D^{1/2} r (the circulant approximation of E^H E itself)
    pub fn apply_inv(&mut self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.kern.mul_real(st, r, &self.dsq, &mut self.tmp)?;
        self.fwd.execute_inplace(st, &mut self.tmp)?;
        self.kern.mul_real(st, &self.tmp, &self.lam_eps, z)?;
        self.inv.execute_inplace(st, z)?;
        self.kern.mul_real(st, z, &self.dsq, &mut self.tmp)?;
        st.memcpy_dtod(&self.tmp, z)?;
        Ok(())
    }
}
