//! Initialisation of the velocity model from a time-averaged (warm-start) reconstruction
//! (`research/velocity_model_sr.md` §A3 step 1):
//! - residual phase: per encode e ≥ 1, the polynomial φ_e fitted to the encode phase difference d_e = arg(x_e x̄_0)
//!   over the object, by iteratively reweighted least squares (Tukey biweight on the wrapped residual, signal
//!   weight |x_0||x_e|). In static tissue d_e = φ_e; vessels and noise are outliers and get weight 0.
//! - static-tissue mask: voxels that every encode's fit accepts (robust weight > ½) with enough signal; it anchors
//!   v ≈ 0 against the residual phase (the degeneracy of §A3) without simulator truth.
//! - velocity: the hierarchical decoder on the phase-corrected images.

use num_complex::Complex64 as C;

use crate::flow::FlowDecoder;
use crate::vmodel::{poly_at, poly_exps};

pub struct StaticFit {
    /// a[(e − 1)·np + p] (rad), the order of `vmodel::poly_exps`
    pub a: Vec<f64>,
    /// 1 = static tissue
    pub mask: Vec<f32>,
    /// robust residual scale (rad) per encode e ≥ 1
    pub sigma: Vec<f64>,
}

fn wrap(x: f64) -> f64 {
    let t = std::f64::consts::TAU;
    x - t * ((x + std::f64::consts::PI) / t).floor()
}

fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    // Gaussian elimination with partial pivoting (n ≤ 20), small ridge for safety
    let tr: f64 = (0..n).map(|i| a[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        a[i * n + i] += 1e-12 * tr.max(1e-300);
    }
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().partial_cmp(&a[j * n + c].abs()).unwrap()).unwrap();
        if p != c {
            for k in 0..n {
                a.swap(c * n + k, p * n + k);
            }
            b.swap(c, p);
        }
        let d = a[c * n + c];
        for i in c + 1..n {
            let f = a[i * n + c] / d;
            for k in c..n {
                a[i * n + k] -= f * a[c * n + k];
            }
            b[i] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| a[i * n + k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i * n + i];
    }
    x
}

/// Robust static-tissue fit of the residual phase. `x[e][r]`: time-averaged image of encode e; `object`: voxels to
/// consider (the support); `deg`: polynomial degree; `min_signal`: fraction of the 95th percentile of |x_0| below
/// which voxels are ignored.
pub fn fit_static_phase(x: &[Vec<C>], object: &[bool], dims: [usize; 3], deg: usize, min_signal: f64) -> StaticFit {
    let ex = poly_exps(deg);
    let np = ex.len();
    let ne = x.len();
    let nv = object.len();
    let mut mags: Vec<f64> = (0..nv).filter(|&r| object[r]).map(|r| x[0][r].norm()).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = mags.get(mags.len() * 95 / 100).copied().unwrap_or(1.0);
    let vox: Vec<usize> = (0..nv).filter(|&r| object[r] && x[0][r].norm() > min_signal * p95).collect();
    let pm: Vec<Vec<f64>> = vox.iter().map(|&r| ex.iter().map(|e| poly_at(e, r, dims)).collect()).collect();
    let mut a = vec![0.0; (ne - 1) * np];
    let mut sigma = vec![0.0; ne - 1];
    let mut keep = vec![true; vox.len()];
    for e in 1..ne {
        let d: Vec<f64> = vox.iter().map(|&r| (x[e][r] * x[0][r].conj()).arg()).collect();
        let sw: Vec<f64> = vox.iter().map(|&r| x[e][r].norm() * x[0][r].norm()).collect();
        let mut rw = vec![1.0; vox.len()];
        let mut coef = vec![0.0; np];
        for it in 0..25 {
            let (mut m, mut b) = (vec![0.0; np * np], vec![0.0; np]);
            for (i, p) in pm.iter().enumerate() {
                let w = sw[i] * rw[i];
                if w == 0.0 {
                    continue;
                }
                // unwrap the data against the current model so the fit is linear
                let fit: f64 = (0..np).map(|q| coef[q] * p[q]).sum();
                let di = fit + wrap(d[i] - fit);
                for u in 0..np {
                    b[u] += w * p[u] * di;
                    for v in 0..np {
                        m[u * np + v] += w * p[u] * p[v];
                    }
                }
            }
            coef = solve(m, b, np);
            // residuals, robust scale (MAD over signal-weighted voxels) and Tukey weights
            let res: Vec<f64> = pm.iter().zip(&d).map(|(p, &di)| wrap(di - (0..np).map(|q| coef[q] * p[q]).sum::<f64>())).collect();
            let mut ar: Vec<f64> = res.iter().map(|v| v.abs()).collect();
            ar.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let s = (1.4826 * ar[ar.len() / 2]).max(1e-6);
            let c = 4.685 * s;
            for (w, r) in rw.iter_mut().zip(&res) {
                let u = r / c;
                *w = if u.abs() < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
            }
            sigma[e - 1] = s;
            let _ = it;
        }
        for (i, k) in keep.iter_mut().enumerate() {
            *k &= rw[i] > 0.5;
        }
        a[(e - 1) * np..e * np].copy_from_slice(&coef);
    }
    let mut mask = vec![0f32; nv];
    for (i, &r) in vox.iter().enumerate() {
        if keep[i] {
            mask[r] = 1.0;
        }
    }
    StaticFit { a, mask, sigma }
}

/// images with the fitted residual phase removed: x_e · e^{−iφ_e}
pub fn remove_phase(x: &[Vec<C>], a: &[f64], dims: [usize; 3], deg: usize) -> Vec<Vec<C>> {
    let ex = poly_exps(deg);
    let np = ex.len();
    x.iter()
        .enumerate()
        .map(|(e, xe)| {
            if e == 0 {
                return xe.clone();
            }
            xe.iter().enumerate().map(|(r, &z)| z * C::from_polar(1.0, -(0..np).map(|p| a[(e - 1) * np + p] * poly_at(&ex[p], r, dims)).sum::<f64>())).collect()
        })
        .collect()
}

/// velocity per voxel by the hierarchical decoder (0 outside `object`)
pub fn decode(x: &[Vec<C>], object: &[bool], dec: &FlowDecoder) -> Vec<[f64; 3]> {
    (0..object.len()).map(|r| if object[r] { dec.velocity(&x.iter().map(|xe| xe[r]).collect::<Vec<_>>()) } else { [0.0; 3] }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_fit_recovers_phase_and_rejects_the_vessel() {
        // 40×30×12 object; static tissue with a degree-1 residual phase per encode; a "vessel" tube with flow
        // (phase differences far from the polynomial) and noise; the fit must recover the coefficients and exclude
        // the vessel from the static mask
        let dims = [40usize, 30, 12];
        let nv = dims.iter().product();
        let (lo, hi) = (std::f64::consts::PI / 1.0, std::f64::consts::PI / 3.0);
        let mut kenc = vec![[0.0; 3]];
        for ax in 0..3 {
            for k in [lo, hi] {
                let mut v = [0.0; 3];
                v[ax] = k;
                kenc.push(v);
            }
        }
        let ex = poly_exps(1);
        let a_true: Vec<f64> = (0..6 * ex.len()).map(|i| 0.5 * (((i * 7919) % 13) as f64 / 6.0 - 1.0)).collect();
        let mut s = 0x1234u64;
        let mut g = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let vessel = |r: usize| {
            let (i, j) = ((r % 40) as f64, ((r / 40) % 30) as f64);
            (i - 20.0).hypot(j - 15.0) < 4.0
        };
        let mut x = vec![vec![C::new(0.0, 0.0); nv]; 7];
        for r in 0..nv {
            let m = 1.0 + 0.3 * g();
            let v = if vessel(r) { [0.2, -0.7, 0.9] } else { [0.0; 3] };
            for e in 0..7 {
                let phi: f64 = if e == 0 { 0.0 } else { (0..ex.len()).map(|p| a_true[(e - 1) * ex.len() + p] * poly_at(&ex[p], r, dims)).sum() };
                let ph = phi - (kenc[e][0] * v[0] + kenc[e][1] * v[1] + kenc[e][2] * v[2]) + 0.4;
                x[e][r] = C::from_polar(m, ph) + C::new(0.03 * g(), 0.03 * g());
            }
        }
        let object = vec![true; nv];
        let fit = fit_static_phase(&x, &object, dims, 1, 0.05);
        let err = fit.a.iter().zip(&a_true).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / (a_true.len() as f64).sqrt();
        assert!(err < 5e-3, "coefficient RMS error {err:e}");
        let in_vessel = (0..nv).filter(|&r| vessel(r) && fit.mask[r] > 0.0).count();
        let static_kept = (0..nv).filter(|&r| !vessel(r) && fit.mask[r] > 0.0).count() as f64 / (0..nv).filter(|&r| !vessel(r)).count() as f64;
        assert!(in_vessel == 0, "{in_vessel} vessel voxels in the static mask");
        assert!(static_kept > 0.9, "only {static_kept:.2} of the static tissue kept");
        // decoded velocity after phase removal: the vessel speed, static ≈ 0
        let dec = FlowDecoder::new(&kenc.iter().map(|k| k.map(|v| v / hasty_mrsim::GAMMA)).collect::<Vec<_>>());
        let v = decode(&remove_phase(&x, &fit.a, dims, 1), &object, &dec);
        let r0 = 15 * 40 + 20; // vessel centre, slice 0
        assert!((v[r0][1] + 0.7).abs() < 0.05 && (v[r0][2] - 0.9).abs() < 0.05, "{:?}", v[r0]);
        assert!(v[5][0].abs() < 0.02, "static voxel velocity {:?}", v[5]);
    }
}
