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
    parts: Parts,
    kern: Arc<Kernels>,
    pub nv: usize,
    /// λ_max / λ_min after flooring (condition number the preconditioner targets)
    pub dyn_range: f64,
    /// λ on the N grid (FFT order, x fastest), for diagnostics
    pub lambda: Vec<f32>,
    /// optional deflation (research/precond_estimator.md P3): M' = M + Σ_i d_i u_i u_iᴴ, None = the metric alone
    defl: Option<DevDefl>,
}

/// device-resident deflation of a circulant metric: u_i (M⁻¹-orthonormal Ritz vectors of M Q), v_i = M⁻¹ u_i,
/// d_i = target_i / θ_i − 1 (apply) and e_i = θ_i / target_i − 1 (inverse, Woodbury)
struct DevDefl {
    u: Vec<CudaSlice<C32>>,
    v: Vec<CudaSlice<C32>>,
    d: Vec<f64>,
    e: Vec<f64>,
}

enum Parts {
    /// everything on the device (FFT plans and a temporary per preconditioner)
    Device {
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
    },
    /// the four N-grid arrays in host memory, uploaded per application into a workspace shared by all
    /// preconditioners of the same grid ([`CircShared`]): no device memory per set
    Host { pinv: Vec<C32>, lam_eps: Vec<C32>, dsq: Vec<C32>, dinv: Vec<C32>, ws: Arc<std::sync::Mutex<CircShared>> },
}

/// Device workspace shared by host-resident circulant preconditioners of one grid: FFT plans, a temporary and two
/// upload slots (stream-ordered: an application's uploads follow the previous application's kernels on the stream).
pub struct CircShared {
    fwd: FftPlan<f32>,
    inv: FftPlan<f32>,
    tmp: CudaSlice<C32>,
    a: CudaSlice<C32>,
    b: CudaSlice<C32>,
}

impl CircShared {
    pub fn new(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3]) -> Result<Arc<std::sync::Mutex<Self>>, Error> {
        let nv = dims[0] * dims[1] * dims[2];
        let sdims = [dims[2], dims[1], dims[0]];
        Ok(Arc::new(std::sync::Mutex::new(Self {
            fwd: FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?,
            inv: FftPlan::<f32>::new(dev, &sdims, Direction::Inverse, Norm::Inverse)?,
            tmp: st.alloc_zeros(nv)?,
            a: st.alloc_zeros(nv)?,
            b: st.alloc_zeros(nv)?,
        })))
    }
}

/// T. Chan's circulant spectrum of the coil-free Toeplitz normal operator of the samples `coords` (radians per voxel, unit
/// weights) on the `dims` grid (x fastest): the triangle-windowed kernel folded onto N, then F_N (real part, FFT order).
/// A Fejér-smoothed sample density: Σ_k λ(k) = nv · h(0) = nv · m · h₁ with h₁ the kernel of one sample at lag 0, so
/// λ / (nv h₁) is the count of samples per Nyquist cell (exactly 1 at the cell of one on-grid sample).
fn chan_lambda(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3], coords: [&[f32]; 3], fwd: &FftPlan<f32>) -> Result<Vec<f32>, Error> {
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
    let mut dlam = st.clone_htod(&c)?;
    fwd.execute_inplace(st, &mut dlam)?;
    Ok(st.clone_dtoh(&dlam)?.iter().map(|z| z.re).collect())
}

/// λ ⊛ P, P = Σ_c |F S_c|² / nv² the coil-map power spectrum (Σ_k P = mean_n Σ_c |S_c|²): the Fourier diagonal
/// ⟨e_k, Σ_c S_cᴴ C S_c e_k⟩ of the coil-weighted circulant (up to the mirror k → −k of P)
fn coil_conv(st: &Arc<CudaStream>, fwd: &FftPlan<f32>, inv: &FftPlan<f32>, lam: &[f32], smaps: &[C32], nc: usize) -> Result<Vec<f32>, Error> {
    let nv = lam.len();
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
    let la: Vec<C32> = lam.iter().map(|&z| C32::new(z, 0.0)).collect();
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
    Ok(st.clone_dtoh(&dc)?.iter().map(|z| z.re).collect())
}

/// [`chan_lambda`] with its own FFT plan: the unfloored Chan spectrum of `coords` (FFT order, x fastest; the
/// samples-per-cell scale is Σ_k λ / (nv m), see there)
pub fn chan_spectrum(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3], coords: [&[f32]; 3]) -> Result<Vec<f32>, Error> {
    let fwd = FftPlan::<f32>::new(dev, &[dims[2], dims[1], dims[0]], Direction::Forward, Norm::None)?;
    chan_lambda(dev, st, dims, coords, &fwd)
}

/// [`coil_conv`] with its own FFT plans: λ ⊛ P for the coil maps `smaps` (coil-major, nc × nv), unfloored
pub fn coil_convolved(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3], lam: &[f32], smaps: &[C32], nc: usize) -> Result<Vec<f32>, Error> {
    let sdims = [dims[2], dims[1], dims[0]];
    let fwd = FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?;
    let inv = FftPlan::<f32>::new(dev, &sdims, Direction::Inverse, Norm::Inverse)?;
    coil_conv(st, &fwd, &inv, lam, smaps, nc)
}

impl CirculantPrecond {
    /// `dims`: x fastest; `coords`: radians per voxel; `smaps`: coil-major (nc × nv); `eps_rel`: floor of λ relative to its max
    pub fn new(dev: &Device, st: &Arc<CudaStream>, kern: Arc<Kernels>, dims: [usize; 3], coords: [&[f32]; 3], smaps: &[C32], nc: usize, eps_rel: f64) -> Result<Self, Error> {
        let nv = dims[0] * dims[1] * dims[2];
        let sdims = [dims[2], dims[1], dims[0]];
        let fwd = FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?;
        let inv = FftPlan::<f32>::new(dev, &sdims, Direction::Inverse, Norm::Inverse)?;
        let lam = chan_lambda(dev, st, dims, coords, &fwd)?;
        // Fourier-diagonal of N = Σ_c S_c^H T S_c: ⟨e_k, N e_k⟩ = (ρ ⊛ P)(k), ρ = circulant density spectrum, P = coil-map
        // power spectrum Σ_c |F S_c|² / nv. Coil weighting is then inside λ and D is set to identity below. (Inverting ρ alone
        // over-amplifies the modes next to the spiral's k = 0 spike, which the coil maps smear onto them.)
        let use_conv = std::env::var("HASTY_PRECOND_MODE").map(|v| v != "density").unwrap_or(true);
        let lam: Vec<f32> = if use_conv { coil_conv(st, &fwd, &inv, &lam, smaps, nc)? } else { lam };
        let mut lam: Vec<C32> = lam.iter().map(|&v| C32::new(v, 0.0)).collect();
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
        if std::env::var("HASTY_PRECOND_DIAG").is_ok() {
            let clamped = lam.iter().filter(|z| z.re < eps).count();
            eprintln!("  circulant: {:.1} % of the {nv} modes at the floor ε = {eps_rel:.0e} λ_max (M's gain 1/ε there)", 100.0 * clamped as f64 / nv as f64);
        }
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
        Ok(Self { parts: Parts::Device { fwd, inv, lam_eps: st.clone_htod(&lam_eps)?, dsq: st.clone_htod(&dsq)?, pinv: st.clone_htod(&pinv)?, dinv: st.clone_htod(&d)?, tmp: st.alloc_zeros(nv)? }, kern, nv, dyn_range: (lmax / lmin) as f64, lambda: lam.iter().map(|z| z.re).collect(), defl: None })
    }

    /// A circulant metric from given factors (research/precond_estimator.md P0–P2, `crate::ppre`): M = D^{-1/2} F⁻¹
    /// diag(1/λ) F D^{-1/2} with `lam` = λ_eff > 0 per Fourier mode (FFT order, x fastest: the symbol of D^{-1/2} Q
    /// D^{-1/2}, floor included) and `dinv` = D^{-1/2} per voxel. Same conventions as [`new`](Self::new) (`factors()`
    /// returns them back); `lambda` holds `lam`.
    pub fn from_factors(dev: &Device, st: &Arc<CudaStream>, kern: Arc<Kernels>, dims: [usize; 3], lam: &[f32], dinv: &[f32]) -> Result<Self, Error> {
        let nv = dims[0] * dims[1] * dims[2];
        assert!(lam.len() == nv && dinv.len() == nv, "from_factors: {} / {} values for {nv} voxels", lam.len(), dinv.len());
        assert!(lam.iter().all(|&l| l > 0.0 && l.is_finite()) && dinv.iter().all(|&d| d > 0.0 && d.is_finite()), "from_factors: λ and D^(-1/2) must be positive");
        let sdims = [dims[2], dims[1], dims[0]];
        let fwd = FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?;
        let inv = FftPlan::<f32>::new(dev, &sdims, Direction::Inverse, Norm::Inverse)?;
        let c = |v: &mut dyn Iterator<Item = f32>| -> Vec<C32> { v.map(|x| C32::new(x, 0.0)).collect() };
        let pinv = c(&mut lam.iter().map(|&l| 1.0 / l));
        let lam_eps = c(&mut lam.iter().copied());
        let dsq = c(&mut dinv.iter().map(|&d| 1.0 / d));
        let dv = c(&mut dinv.iter().copied());
        let lmax = lam.iter().cloned().fold(0f32, f32::max);
        let lmin = lam.iter().cloned().fold(f32::MAX, f32::min);
        Ok(Self { parts: Parts::Device { fwd, inv, pinv: st.clone_htod(&pinv)?, lam_eps: st.clone_htod(&lam_eps)?, dsq: st.clone_htod(&dsq)?, dinv: st.clone_htod(&dv)?, tmp: st.alloc_zeros(nv)? }, kern, nv, dyn_range: (lmax / lmin) as f64, lambda: lam.to_vec(), defl: None })
    }

    /// Replace the symbol of a metric (of [`from_factors`](Self::from_factors) or [`new`](Self::new)) in place: M = D^{-1/2}
    /// F⁻¹ diag(1/λ) F D^{-1/2} with the new `lam` = λ_eff > 0 (FFT order), D unchanged, no new FFT plans (ADMM's
    /// per-set clip metric, `crate::admm`). Uploads two N-grid arrays. Drops an attached deflation (it was built for the
    /// old symbol).
    pub fn set_symbol(&mut self, st: &Arc<CudaStream>, lam: &[f32]) -> Result<(), Error> {
        assert_eq!(lam.len(), self.nv, "set_symbol: {} values for {} voxels", lam.len(), self.nv);
        assert!(lam.iter().all(|&l| l > 0.0 && l.is_finite()), "set_symbol: λ must be positive");
        let pv: Vec<C32> = lam.iter().map(|&l| C32::new(1.0 / l, 0.0)).collect();
        let lv: Vec<C32> = lam.iter().map(|&l| C32::new(l, 0.0)).collect();
        match &mut self.parts {
            Parts::Device { pinv, lam_eps, .. } => {
                st.memcpy_htod(&pv, pinv)?;
                st.memcpy_htod(&lv, lam_eps)?;
            }
            Parts::Host { pinv, lam_eps, .. } => {
                *pinv = pv;
                *lam_eps = lv;
            }
        }
        let lmax = lam.iter().cloned().fold(0f32, f32::max);
        let lmin = lam.iter().cloned().fold(f32::MAX, f32::min);
        self.dyn_range = (lmax / lmin) as f64;
        self.lambda.clear();
        self.lambda.extend_from_slice(lam);
        self.defl = None;
        Ok(())
    }

    /// attach a deflation (P3): `u` the M⁻¹-orthonormal Ritz vectors of M Q (host), `theta` their Ritz values,
    /// `target` where each is moved (`crate::ppre::Deflation`). v_i = M⁻¹ u_i is formed here with the metric itself
    /// (call before attaching another deflation). Device memory: 2 images per vector.
    pub fn with_deflation(mut self, st: &Arc<CudaStream>, u: &[Vec<C32>], theta: &[f64], target: &[f64]) -> Result<Self, Error> {
        assert!(u.len() == theta.len() && u.len() == target.len());
        assert!(theta.iter().all(|&t| t > 0.0) && target.iter().all(|&t| t > 0.0), "deflation: positive Ritz values and targets");
        self.defl = None;
        let mut du = Vec::with_capacity(u.len());
        let mut dvv = Vec::with_capacity(u.len());
        for ui in u {
            assert_eq!(ui.len(), self.nv);
            let d = st.clone_htod(ui)?;
            let mut v = st.alloc_zeros::<C32>(self.nv)?;
            self.apply_inv(st, &d, &mut v)?;
            du.push(d);
            dvv.push(v);
        }
        let d: Vec<f64> = theta.iter().zip(target).map(|(&t, &g)| g / t - 1.0).collect();
        let e: Vec<f64> = theta.iter().zip(target).map(|(&t, &g)| t / g - 1.0).collect();
        self.defl = Some(DevDefl { u: du, v: dvv, d, e });
        Ok(self)
    }

    /// rank of the attached deflation (0: none)
    pub fn deflation_rank(&self) -> usize {
        self.defl.as_ref().map_or(0, |d| d.u.len())
    }

    /// z += Σ_i c_i w_i (w_iᴴ r) for the deflation's (u, d) (apply) or (v, e) (inverse)
    fn defl_add(&self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        if let Some(df) = &self.defl {
            let (w, c) = if inverse { (&df.v, &df.e) } else { (&df.u, &df.d) };
            let one = num_complex::Complex64::new(1.0, 0.0);
            for (wi, &ci) in w.iter().zip(c) {
                let a = self.kern.dot(st, wi, r)? * ci;
                self.kern.axpby(st, a, wi, one, z)?;
            }
        }
        Ok(())
    }

    /// move the N-grid arrays to host memory and use the shared workspace `ws` (same grid) from now on: frees the
    /// device memory of this preconditioner (four arrays, a temporary, two FFT plans); each application then uploads
    /// two arrays (2 × 8 nv bytes)
    pub fn to_host(self, st: &Arc<CudaStream>, ws: &Arc<std::sync::Mutex<CircShared>>) -> Result<Self, Error> {
        let parts = match self.parts {
            Parts::Device { pinv, lam_eps, dsq, dinv, .. } => Parts::Host { pinv: st.clone_dtoh(&pinv)?, lam_eps: st.clone_dtoh(&lam_eps)?, dsq: st.clone_dtoh(&dsq)?, dinv: st.clone_dtoh(&dinv)?, ws: ws.clone() },
            h => h,
        };
        Ok(Self { parts, ..self })
    }

    /// the metric's factors on the host: (λ_eff, D^{-1/2}) with M⁻¹ = D^{1/2} F⁻¹ diag(λ_eff) F D^{1/2} (λ_eff = the
    /// floored, coil-normalised spectrum, FFT order, x fastest): what composite metrics (`knotncg::PriorMetric`) build on
    pub fn factors(&self, st: &Arc<CudaStream>) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let (lam, dinv) = match &self.parts {
            Parts::Device { lam_eps, dinv, .. } => (st.clone_dtoh(lam_eps)?, st.clone_dtoh(dinv)?),
            Parts::Host { lam_eps, dinv, .. } => (lam_eps.clone(), dinv.clone()),
        };
        Ok((lam.iter().map(|z| z.re).collect(), dinv.iter().map(|z| z.re).collect()))
    }

    pub fn on_host(&self) -> bool {
        matches!(self.parts, Parts::Host { .. })
    }

    /// z = M r
    pub fn apply(&mut self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        match &mut self.parts {
            Parts::Device { fwd, inv, pinv, dinv, tmp, .. } => {
                self.kern.mul_real(st, r, dinv, tmp)?;
                fwd.execute_inplace(st, tmp)?;
                self.kern.mul_real(st, tmp, pinv, z)?;
                inv.execute_inplace(st, z)?;
                self.kern.mul_real(st, z, dinv, tmp)?;
                st.memcpy_dtod(tmp, z)?;
            }
            Parts::Host { pinv, dinv, ws, .. } => {
                let mut g = ws.lock().unwrap();
                let w = &mut *g;
                st.memcpy_htod(dinv.as_slice(), &mut w.a)?;
                st.memcpy_htod(pinv.as_slice(), &mut w.b)?;
                self.kern.mul_real(st, r, &w.a, &mut w.tmp)?;
                w.fwd.execute_inplace(st, &mut w.tmp)?;
                self.kern.mul_real(st, &w.tmp, &w.b, z)?;
                w.inv.execute_inplace(st, z)?;
                self.kern.mul_real(st, z, &w.a, &mut w.tmp)?;
                st.memcpy_dtod(&w.tmp, z)?;
            }
        }
        self.defl_add(st, r, z, false)
    }

    /// z = M⁻¹ r = D^{1/2} F⁻¹ diag(λ + ε) F D^{1/2} r (the circulant approximation of E^H E itself)
    pub fn apply_inv(&mut self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        match &mut self.parts {
            Parts::Device { fwd, inv, lam_eps, dsq, tmp, .. } => {
                self.kern.mul_real(st, r, dsq, tmp)?;
                fwd.execute_inplace(st, tmp)?;
                self.kern.mul_real(st, tmp, lam_eps, z)?;
                inv.execute_inplace(st, z)?;
                self.kern.mul_real(st, z, dsq, tmp)?;
                st.memcpy_dtod(tmp, z)?;
            }
            Parts::Host { lam_eps, dsq, ws, .. } => {
                let mut g = ws.lock().unwrap();
                let w = &mut *g;
                st.memcpy_htod(dsq.as_slice(), &mut w.a)?;
                st.memcpy_htod(lam_eps.as_slice(), &mut w.b)?;
                self.kern.mul_real(st, r, &w.a, &mut w.tmp)?;
                w.fwd.execute_inplace(st, &mut w.tmp)?;
                self.kern.mul_real(st, &w.tmp, &w.b, z)?;
                w.inv.execute_inplace(st, z)?;
                self.kern.mul_real(st, z, &w.a, &mut w.tmp)?;
                st.memcpy_dtod(&w.tmp, z)?;
            }
        }
        self.defl_add(st, r, z, true)
    }
}

