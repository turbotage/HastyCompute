//! Figures of merit of a base spoke and its rotation set.
//!
//! Information per spoke: N rotated copies of a curve that crosses the shell r with path length q(r) per unit radius
//! leave L(r) = N q / (4π r²) of curve length per unit volume; Nyquist for a FOV D (curves ≤ 1/D apart, samples along
//! the curve ≤ 1/D apart) needs L ≥ D² w(r). Hence the spokes needed: N* = max_r D² w(r) 4π r² / q(r), and the scan
//! time for one Nyquist-complete set ∝ N*·TR, the main objective (`research/sequence_design.md` §2). N* is the ideal
//! (perfectly interleaved) count; [`coverage`] measures how a concrete rotation set fills the Nyquist cells.

use crate::wave::Grad;
use crate::{norm, Mat3, Vec3, GAMMA};

#[derive(Clone, Debug)]
pub struct Density {
    /// bin centres, cycles/m
    pub r: Vec<f64>,
    /// curve length per unit volume of one spoke (rotation average), (cycles/m)⁻²
    pub l1: Vec<f64>,
    /// spokes needed for Nyquist per bin
    pub need: Vec<f64>,
    pub n_star: f64,
    pub r_star: f64,
}

/// rotation-averaged curve density of the k samples `k` (dense, cycles/m) for FOV `fov` and relative density `w(r)`
pub fn density(k: &[Vec3], kmax: f64, fov: f64, w: impl Fn(f64) -> f64, nbins: usize) -> Density {
    let h = kmax / nbins as f64;
    let mut len = vec![0.0; nbins];
    for s in k.windows(2) {
        let mid = [0, 1, 2].map(|q| 0.5 * (s[0][q] + s[1][q]));
        let r = norm(&mid);
        let b = (r / h) as usize;
        if b < nbins {
            len[b] += norm(&crate::sub(&s[1], &s[0]));
        }
    }
    let mut d = Density { r: vec![], l1: vec![], need: vec![], n_star: 0.0, r_star: 0.0 };
    for b in 0..nbins {
        let (r0, r1) = (b as f64 * h, (b + 1) as f64 * h);
        let vol = 4.0 / 3.0 * std::f64::consts::PI * (r1.powi(3) - r0.powi(3));
        let rc = 0.5 * (r0 + r1);
        let l1 = len[b] / vol;
        let need = if l1 > 0.0 { fov * fov * w(rc) / l1 } else { f64::INFINITY };
        if need > d.n_star {
            d.n_star = need;
            d.r_star = rc;
        }
        d.r.push(rc);
        d.l1.push(l1);
        d.need.push(need);
    }
    d
}

/// fraction of empty Nyquist cells (spacing 1/fov) per radial shell when the rotated copies of `k` are acquired
#[derive(Clone, Debug)]
pub struct Coverage {
    pub r: Vec<f64>,
    pub empty: Vec<f64>,
    /// mean hits per cell
    pub mean_hits: Vec<f64>,
}

pub fn coverage(k: &[Vec3], rots: &[Mat3], kmax: f64, fov: f64, nshell: usize) -> Coverage {
    let m = (2.0 * kmax * fov).ceil() as usize + 1;
    let cell = 1.0 / fov;
    let idx = |x: f64| ((x + kmax) / cell).floor() as isize;
    let mut hits = vec![0u16; m * m * m];
    // one deposit per cell per spoke (a spoke loitering in a cell counts once)
    let mut last = vec![u32::MAX; m * m * m];
    for (s, r) in rots.iter().enumerate() {
        for p in k {
            let q = crate::mat_vec(r, p);
            let (i, j, l) = (idx(q[0]), idx(q[1]), idx(q[2]));
            if i < 0 || j < 0 || l < 0 || i as usize >= m || j as usize >= m || l as usize >= m {
                continue;
            }
            let c = (i as usize * m + j as usize) * m + l as usize;
            if last[c] != s as u32 {
                last[c] = s as u32;
                hits[c] = hits[c].saturating_add(1);
            }
        }
    }
    let h = kmax / nshell as f64;
    let (mut tot, mut emp, mut sum) = (vec![0usize; nshell], vec![0usize; nshell], vec![0.0; nshell]);
    for i in 0..m {
        for j in 0..m {
            for l in 0..m {
                let c = [i, j, l].map(|x| -kmax + (x as f64 + 0.5) * cell);
                let b = (norm(&c) / h) as usize;
                if b >= nshell {
                    continue;
                }
                let v = hits[(i * m + j) * m + l];
                tot[b] += 1;
                sum[b] += v as f64;
                if v == 0 {
                    emp[b] += 1;
                }
            }
        }
    }
    let r = (0..nshell).map(|b| (b as f64 + 0.5) * h).collect();
    let empty = (0..nshell).map(|b| if tot[b] > 0 { emp[b] as f64 / tot[b] as f64 } else { 0.0 }).collect();
    let mean_hits = (0..nshell).map(|b| if tot[b] > 0 { sum[b] / tot[b] as f64 } else { 0.0 }).collect();
    Coverage { r, empty, mean_hits }
}

/// anisotropy of the concomitant tensor P = ∫G Gᵀ: ‖P − tr(P)/3 I‖_F / (tr(P)/√3); 0 = isotropic (the concomitant
/// phase is then the same for every rotation), √2 for a straight line, 1/√2 for an isotropic planar curve
pub fn concomitant_anisotropy(p: &[f64; 6]) -> f64 {
    let tr = p[0] + p[1] + p[2];
    if tr <= 0.0 {
        return 0.0;
    }
    let a = [p[0] - tr / 3.0, p[1] - tr / 3.0, p[2] - tr / 3.0];
    let an = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + 2.0 * (p[3] * p[3] + p[4] * p[4] + p[5] * p[5])).sqrt();
    an / (tr / 3f64.sqrt())
}

/// concomitant phase scale (rad) of P at distance `x` (m) from isocentre: γ/(2 B0) · tr(P)/3 · x² (isotropic part)
pub fn concomitant_phase(p: &[f64; 6], b0: f64, x: f64) -> f64 {
    GAMMA / (2.0 * b0) * (p[0] + p[1] + p[2]) / 3.0 * x * x
}

/// readout first moment where the trajectory first reaches each radius fraction: phase per m/s (rad·s/m)
pub fn readout_m1(w: &Grad, t_ref: f64, kmax: f64, fracs: &[f64]) -> Vec<f64> {
    let m = w.moments(t_ref);
    let k = m.k();
    fracs
        .iter()
        .map(|f| {
            let i = k.iter().position(|p| norm(p) >= f * kmax).unwrap_or(k.len() - 1);
            GAMMA * norm(&m.m1[i])
        })
        .collect()
}

/// largest k error (in Nyquist cells 1/fov) over the first `n` raster points
pub fn max_k_error(err: &[Vec3], n: usize, fov: f64) -> f64 {
    err.iter().take(n).map(|e| norm(e) * fov).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_spokes_need_pi_m_squared() {
        // center-out radial: N* = 4π kmax² D² = π M² (M = 2 kmax D)
        let (kmax, fov) = (100.0, 0.2);
        let k: Vec<Vec3> = (0..=100_000).map(|i| [0.0, 0.0, kmax * i as f64 / 100_000.0]).collect();
        let d = density(&k, kmax, fov, |_| 1.0, 50);
        let m = 2.0 * kmax * fov;
        let expect = std::f64::consts::PI * m * m;
        // the outermost bin is centred just inside kmax
        assert!((d.n_star - expect).abs() / expect < 0.03, "{} vs {expect}", d.n_star);
    }

    #[test]
    fn anisotropy_limits() {
        assert!(concomitant_anisotropy(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) < 1e-12);
        let line = concomitant_anisotropy(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let plane = concomitant_anisotropy(&[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        assert!((line - 2f64.sqrt()).abs() < 1e-12, "{line}");
        assert!((plane - 1.0 / 2f64.sqrt()).abs() < 1e-12, "{plane}");
    }
}
