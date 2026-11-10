//! f64 CPU references for physics-aware multi-coil encoding operators (segmented / temporal NUFFTs).
//!
//! One generic direct sum serves the exact physics (factor e^{−iΦ(j, q)}) and CPU emulations of an approximate
//! operator (the factor the approximation uses), so a test can split an operator's error into the approximation's own
//! error (emulation vs exact, computed here in f64) and the implementation error (GPU vs emulation, e.g. NUFFT
//! interpolation):
//!
//!   y_c(j)  = Σ_f b_f(j) Σ_q e(j, q) S_c(q) x_f(q)                     (forward)
//!   x̂_f(q) = Σ_j b_f(j) conj(e(j, q)) Σ_c conj(S_c(q)) y_c(j)          (adjoint, exact transpose)
//!
//! Data layout as in hasty-nufft: coil-major y (nc × m), frame-major x (frames × nv), coil-major maps (nc × nv).
//! Work is split over sample chunks on all cores (std threads, no extra dependency).

use crate::C64;

/// Periodic uniform B-spline weight of frame node `f` (node at t = f, period `frames`) at time `t` (frame units):
/// degree 1 = centred hat, degree 3 = cubic B-spline. One frame: weight 1.
pub fn bspline_weight(frames: usize, degree: u32, f: usize, t: f64) -> f64 {
    if frames == 1 {
        return 1.0;
    }
    let n = frames as f64;
    let mut d = (t - f as f64).rem_euclid(n);
    if d > 0.5 * n {
        d -= n;
    }
    let a = d.abs();
    match degree {
        1 => (1.0 - a).max(0.0),
        3 if a <= 1.0 => 2.0 / 3.0 - a * a + 0.5 * a * a * a,
        3 if a <= 2.0 => (2.0 - a).powi(3) / 6.0,
        3 => 0.0,
        d => panic!("B-spline degree {d}: 1 or 3"),
    }
}

/// The operator's shape and coil maps.
pub struct DirectSum<'a> {
    pub nv: usize,
    pub nc: usize,
    pub m: usize,
    pub frames: usize,
    /// coil-major sensitivities (nc × nv)
    pub smaps: &'a [C64],
    /// voxels summed over (others contribute nothing and get a zero adjoint)
    pub voxels: &'a [usize],
    /// per-sample frame weights, `frames` per sample (None: one frame)
    pub bw: Option<&'a [f64]>,
}

impl DirectSum<'_> {
    /// (A x, Aᴴ y) for the encoding factor `factor(j, q)` (sample j, voxel q), see the module docs.
    pub fn apply(&self, factor: &(dyn Fn(usize, usize) -> C64 + Sync), x: &[C64], y: &[C64]) -> (Vec<C64>, Vec<C64>) {
        let (nv, nc, m, nf) = (self.nv, self.nc, self.m, self.frames);
        assert!(x.len() >= nf * nv && y.len() >= nc * m && self.smaps.len() >= nc * nv);
        assert!(self.bw.is_none_or(|b| b.len() >= nf * m) && (nf == 1 || self.bw.is_some()));
        let nt = std::thread::available_parallelism().map_or(4, |n| n.get()).min(64).max(1);
        let chunk = m.div_ceil(nt).max(1);
        let parts: Vec<(usize, Vec<C64>, Vec<C64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..m)
                .step_by(chunk)
                .map(|j0| {
                    sc.spawn(move || {
                        let j1 = (j0 + chunk).min(m);
                        let mut yo = vec![C64::new(0.0, 0.0); nc * (j1 - j0)];
                        let mut xo = vec![C64::new(0.0, 0.0); nf * nv];
                        let mut b = vec![1.0f64; nf];
                        let mut yj = vec![C64::new(0.0, 0.0); nc];
                        for j in j0..j1 {
                            if let Some(bw) = self.bw {
                                b.copy_from_slice(&bw[j * nf..(j + 1) * nf]);
                            }
                            for c in 0..nc {
                                yj[c] = y[c * m + j];
                            }
                            for &q in self.voxels {
                                let e = factor(j, q);
                                let xv: C64 = (0..nf).filter(|&f| b[f] != 0.0).map(|f| x[f * nv + q] * b[f]).sum();
                                let mut s = C64::new(0.0, 0.0);
                                for c in 0..nc {
                                    let sm = self.smaps[c * nv + q];
                                    yo[c * (j1 - j0) + j - j0] += sm * e * xv;
                                    s += sm.conj() * yj[c];
                                }
                                let s = s * e.conj();
                                for f in 0..nf {
                                    if b[f] != 0.0 {
                                        xo[f * nv + q] += s * b[f];
                                    }
                                }
                            }
                        }
                        (j0, yo, xo)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut yout = vec![C64::new(0.0, 0.0); nc * m];
        let mut xout = vec![C64::new(0.0, 0.0); nf * nv];
        for (j0, yo, xo) in parts {
            let len = yo.len() / nc.max(1);
            for c in 0..nc {
                yout[c * m + j0..c * m + j0 + len].copy_from_slice(&yo[c * len..(c + 1) * len]);
            }
            for (a, b) in xout.iter_mut().zip(&xo) {
                *a += b;
            }
        }
        (yout, xout)
    }
}

/// Relative L2 error ‖a − b‖ / ‖b‖ restricted to the indices in `idx` (None: all).
pub fn rel_on(a: &[C64], b: &[C64], idx: Option<&[usize]>) -> f64 {
    let (mut n, mut d) = (0.0, 0.0);
    let mut acc = |i: usize| {
        n += (a[i] - b[i]).norm_sqr();
        d += b[i].norm_sqr();
    };
    match idx {
        Some(ix) => ix.iter().for_each(|&i| acc(i)),
        None => (0..a.len().min(b.len())).for_each(&mut acc),
    }
    (n / d.max(f64::MIN_POSITIVE)).sqrt()
}

/// Indices of every frame's copy of the voxels `vox` in a frame-major stack of `frames` × `nv`.
pub fn frame_indices(vox: &[usize], nv: usize, frames: usize) -> Vec<usize> {
    (0..frames).flat_map(|f| vox.iter().map(move |&q| f * nv + q)).collect()
}

/// Linear interpolation of a segment-interpolator table `atab` ([ntab][nseg]) at τ exactly as the SegNufft kernels do
/// (f32 index arithmetic, clamped to the table, last cell open by 1e-3).
pub fn interp_table(atab: &[C64], nseg: usize, t0: f64, dt: f64, tau: f64) -> Vec<C64> {
    let ntab = atab.len() / nseg;
    let mut u = ((tau as f32 - t0 as f32) / dt as f32).max(0.0).min((ntab - 1) as f32 - 1e-3);
    if !u.is_finite() {
        u = 0.0;
    }
    let i = u as usize;
    let fr = (u - i as f32) as f64;
    (0..nseg).map(|l| atab[i * nseg + l] + (atab[(i + 1) * nseg + l] - atab[i * nseg + l]) * fr).collect()
}

/// Truncated Taylor series of e^{−iψ} (order 0, 1, 2): what the per-sample ΔC kernel correction implements.
pub fn taylor_expm(psi: f64, order: u32) -> C64 {
    match order {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(1.0, -psi),
        _ => C64::new(1.0 - 0.5 * psi * psi, -psi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_sum_is_adjoint_and_temporal_weights_partition_unity() {
        let (nv, nc, m, nf) = (37usize, 3usize, 53usize, 5usize);
        let mut r = crate::Rng::new(9);
        let smaps = r.complex_vec(nc * nv);
        let x = r.complex_vec(nf * nv);
        let y = r.complex_vec(nc * m);
        let ph: Vec<f64> = (0..m * nv).map(|_| 7.0 * r.uniform()).collect();
        let t: Vec<f64> = (0..m).map(|_| (r.uniform() + 1.0) * 0.5 * nf as f64).collect();
        for deg in [1u32, 3] {
            let bw: Vec<f64> = (0..m).flat_map(|j| (0..nf).map(move |f| (j, f))).map(|(j, f)| bspline_weight(nf, deg, f, t[j])).collect();
            for j in 0..m {
                let s: f64 = bw[j * nf..(j + 1) * nf].iter().sum();
                assert!((s - 1.0).abs() < 1e-12, "B-spline weights sum {s}");
            }
            let vox: Vec<usize> = (0..nv).collect();
            let ds = DirectSum { nv, nc, m, frames: nf, smaps: &smaps, voxels: &vox, bw: Some(&bw) };
            let f = |j: usize, q: usize| C64::from_polar(1.0, -ph[j * nv + q]);
            let (ax, ahy) = ds.apply(&f, &x, &y);
            let l: C64 = y.iter().zip(&ax).map(|(a, b)| a.conj() * b).sum();
            let rr: C64 = ahy.iter().zip(&x).map(|(a, b)| a.conj() * b).sum();
            assert!((l - rr).norm() < 1e-12 * l.norm(), "degree {deg}: {l} vs {rr}");
            // explicit forward entry
            let (j, c) = (17, 2);
            let mut want = C64::new(0.0, 0.0);
            for q in 0..nv {
                let xv: C64 = (0..nf).map(|f| x[f * nv + q] * bw[j * nf + f]).sum();
                want += f(j, q) * smaps[c * nv + q] * xv;
            }
            assert!((ax[c * m + j] - want).norm() < 1e-12 * want.norm());
        }
    }
}
