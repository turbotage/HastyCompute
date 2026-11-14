//! Cardiac low-pass prior: per voxel, encode and respiratory bin, the series over the cardiac frames is projected onto
//! the cyclic harmonics |k| ≤ K (x ← x + κ (P x − x); κ = 1: the projection, P²nP then solves the least-squares
//! problem constrained to band-limited cardiac cycles).
//!
//! Why (research/recon_diag_2026-11-13.md §10): the per-frame reconstruction error of this protocol is mostly
//! incoherent across cardiac frames, but lies in the same few temporal components per 8³ block as the dynamics, so
//! LLR cannot separate the two (one soft step: corr 0.37 → 0.46 with the slope 0.46 → 0.37), while a fixed low-pass in
//! cardiac time keeps the slope (|k| ≤ 4: corr 0.37 → 0.57, slope 0.46 → 0.43). The ph0001 truth needs |k| ≤ 6–8 for
//! 0.12–0.08 rad (of 0.40 rad of dynamics).
//!
//! Stack layout: one image of `nv` voxels per set, set t = frame · ne + encode, frame = r · nf_card + c.

use hasty_cuda::C32;
use rayon::prelude::*;

/// the projection onto the cyclic harmonics min(k, n − k) ≤ kmax over n frames: real, symmetric, circulant (row-major)
pub fn lowpass_matrix(n: usize, kmax: usize) -> Vec<f32> {
    let keep: Vec<usize> = (0..n).filter(|&k| k.min(n - k) <= kmax).collect();
    let mut p = vec![0f32; n * n];
    for a in 0..n {
        for b in 0..n {
            let d = a as f64 - b as f64;
            let s: f64 = keep.iter().map(|&k| (2.0 * std::f64::consts::PI * k as f64 * d / n as f64).cos()).sum();
            p[a * n + b] = (s / n as f64) as f32;
        }
    }
    p
}

/// in place on the stack (layout above): x ← x + κ (P x − x) over the cardiac frames of every (respiratory bin, encode)
pub fn cardiac_lowpass(x: &mut [C32], nv: usize, ne: usize, nf_card: usize, nr: usize, kmax: usize, kappa: f32) {
    let n = nf_card;
    assert_eq!(x.len(), nv * ne * n * nr, "cardiac_lowpass: stack size");
    if n < 2 || kmax * 2 >= n {
        return;
    }
    let p = lowpass_matrix(n, kmax);
    const CH: usize = 4096;
    let xp = x.as_mut_ptr() as usize;
    for r in 0..nr {
        for e in 0..ne {
            let base: Vec<usize> = (0..n).map(|c| ((r * n + c) * ne + e) * nv).collect();
            (0..nv.div_ceil(CH)).into_par_iter().for_each(|ci| {
                let (q0, q1) = (ci * CH, ((ci + 1) * CH).min(nv));
                let len = q1 - q0;
                let d = xp as *mut C32;
                // SAFETY: every task owns the voxels q0..q1 of the n sets of this (r, e); the sets are disjoint
                let mut buf = vec![C32::new(0.0, 0.0); n * len];
                for c in 0..n {
                    for j in 0..len {
                        buf[c * len + j] = unsafe { *d.add(base[c] + q0 + j) };
                    }
                }
                for c in 0..n {
                    let row = &p[c * n..(c + 1) * n];
                    for j in 0..len {
                        let (mut re, mut im) = (0f32, 0f32);
                        for (c2, &w) in row.iter().enumerate() {
                            let v = buf[c2 * len + j];
                            re += w * v.re;
                            im += w * v.im;
                        }
                        let v = buf[c * len + j];
                        unsafe { *d.add(base[c] + q0 + j) = C32::new(v.re + kappa * (re - v.re), v.im + kappa * (im - v.im)) };
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_low_harmonics_removes_high() {
        let (nv, ne, n, nr, kmax) = (5usize, 2usize, 20usize, 2usize, 4usize);
        let tau = 2.0 * std::f64::consts::PI;
        // per voxel: a mean, harmonic 3 (kept) and harmonic 7 (removed), different per set index
        let mut x = vec![C32::new(0.0, 0.0); nv * ne * n * nr];
        let mut want = x.clone();
        for r in 0..nr {
            for c in 0..n {
                for e in 0..ne {
                    for q in 0..nv {
                        let ph = tau * c as f64 / n as f64;
                        let lo = (1.0 + q as f64) * (3.0 * ph + e as f64).cos() + 0.5 * r as f64;
                        let hi = 0.7 * (7.0 * ph + q as f64).sin();
                        let i = ((r * n + c) * ne + e) * nv + q;
                        x[i] = C32::new((lo + hi) as f32, (0.3 * lo) as f32);
                        want[i] = C32::new(lo as f32, (0.3 * lo) as f32);
                    }
                }
            }
        }
        let mut y = x.clone();
        cardiac_lowpass(&mut y, nv, ne, n, nr, kmax, 1.0);
        let err = y.iter().zip(&want).map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)) as f64).sum::<f64>().sqrt();
        assert!(err < 1e-4, "projection error {err}");
        // κ = 0.5: halfway
        let mut z = x.clone();
        cardiac_lowpass(&mut z, nv, ne, n, nr, kmax, 0.5);
        let err2 = z.iter().zip(x.iter().zip(&want)).map(|(a, (o, w))| ((a.re - 0.5 * (o.re + w.re)).powi(2) + (a.im - 0.5 * (o.im + w.im)).powi(2)) as f64).sum::<f64>().sqrt();
        assert!(err2 < 1e-4, "κ 0.5: {err2}");
        // a projection: idempotent
        let mut yy = y.clone();
        cardiac_lowpass(&mut yy, nv, ne, n, nr, kmax, 1.0);
        let d = yy.iter().zip(&y).map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)) as f64).sum::<f64>().sqrt();
        assert!(d < 1e-4, "not idempotent: {d}");
    }
}
