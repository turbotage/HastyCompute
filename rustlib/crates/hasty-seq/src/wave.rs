//! Gradient waveforms: three axes, piecewise linear between the raster points `t0 + i·dt` (zero outside the span),
//! with exact cumulative moments.
//!
//! On each raster segment G is linear, so G·(t − t_ref)ⁿ (n ≤ 2) and G Gᵀ are polynomials of degree ≤ 3 and Simpson's
//! rule is exact: the moments below have no discretisation error.

use crate::system::System;
use crate::{mat_vec, norm, Mat3, Vec3, GAMMA_BAR};

#[derive(Clone, Debug)]
pub struct Grad {
    pub dt: f64,
    pub t0: f64,
    pub g: Vec<Vec3>,
}

/// Cumulative moments at the raster points, time origin `t_ref`: M0 = ∫G, M1 = ∫G (t − t_ref), M2 = ∫G (t − t_ref)²,
/// and P = ∫G Gᵀ (xx, yy, zz, xy, xz, yz; the concomitant second-order tensor).
#[derive(Clone, Debug)]
pub struct Moments {
    pub t: Vec<f64>,
    pub m0: Vec<Vec3>,
    pub m1: Vec<Vec3>,
    pub m2: Vec<Vec3>,
    pub p: Vec<[f64; 6]>,
}

impl Moments {
    pub fn last(&self) -> (Vec3, Vec3, Vec3, [f64; 6]) {
        let n = self.t.len() - 1;
        (self.m0[n], self.m1[n], self.m2[n], self.p[n])
    }
    /// k-space trajectory (cycles/m) at the raster points
    pub fn k(&self) -> Vec<Vec3> {
        self.m0.iter().map(|m| m.map(|x| GAMMA_BAR * x)).collect()
    }
}

/// Limit check result: largest per-axis and vector amplitude / slew, and the fraction of the limit.
#[derive(Clone, Debug)]
pub struct LimitReport {
    pub g_axis: f64,
    pub g_vec: f64,
    pub s_axis: f64,
    pub s_vec: f64,
}

impl LimitReport {
    pub fn ok_axis(&self, sys: &System) -> bool {
        self.g_axis <= sys.gmax * (1.0 + 1e-9) && self.s_axis <= sys.smax * (1.0 + 1e-6)
    }
    pub fn ok_vec(&self, sys: &System) -> bool {
        self.g_vec <= sys.gmax * (1.0 + 1e-9) && self.s_vec <= sys.smax * (1.0 + 1e-6)
    }
}

impl Grad {
    pub fn zeros(n: usize, dt: f64, t0: f64) -> Self {
        Self { dt, t0, g: vec![[0.0; 3]; n] }
    }
    /// from one axis' samples
    pub fn from_axis(axis: usize, s: &[f64], dt: f64, t0: f64) -> Self {
        let mut g = vec![[0.0; 3]; s.len()];
        for (i, v) in s.iter().enumerate() {
            g[i][axis] = *v;
        }
        Self { dt, t0, g }
    }
    /// from three axes' samples (padded with zeros to the longest)
    pub fn from_axes(x: &[f64], y: &[f64], z: &[f64], dt: f64, t0: f64) -> Self {
        let n = x.len().max(y.len()).max(z.len());
        let at = |s: &[f64], i: usize| s.get(i).copied().unwrap_or(0.0);
        Self { dt, t0, g: (0..n).map(|i| [at(x, i), at(y, i), at(z, i)]).collect() }
    }
    pub fn len(&self) -> usize {
        self.g.len()
    }
    pub fn is_empty(&self) -> bool {
        self.g.is_empty()
    }
    pub fn duration(&self) -> f64 {
        self.g.len().saturating_sub(1) as f64 * self.dt
    }
    pub fn t_end(&self) -> f64 {
        self.t0 + self.duration()
    }
    /// G at an arbitrary time (linear interpolation, zero outside)
    pub fn at(&self, t: f64) -> Vec3 {
        let f = (t - self.t0) / self.dt;
        if f < 0.0 || self.g.is_empty() || f > (self.g.len() - 1) as f64 {
            return [0.0; 3];
        }
        let i = (f.floor() as usize).min(self.g.len().saturating_sub(2));
        let a = f - i as f64;
        if i + 1 >= self.g.len() {
            return self.g[i];
        }
        [0, 1, 2].map(|q| self.g[i][q] * (1.0 - a) + self.g[i + 1][q] * a)
    }
    /// this waveform followed by `other` (same raster); a shared zero end / start sample is merged
    pub fn then(&self, other: &Grad) -> Grad {
        assert!((self.dt - other.dt).abs() < 1e-15, "raster mismatch");
        let mut g = self.g.clone();
        let mut o = other.g.as_slice();
        if let (Some(a), Some(b)) = (g.last(), o.first()) {
            if norm(a) == 0.0 && norm(b) == 0.0 {
                o = &o[1..];
            }
        }
        g.extend_from_slice(o);
        Grad { dt: self.dt, t0: self.t0, g }
    }
    /// G → R G (logical → physical)
    pub fn rotated(&self, r: &Mat3) -> Grad {
        Grad { dt: self.dt, t0: self.t0, g: self.g.iter().map(|v| mat_vec(r, v)).collect() }
    }
    pub fn scaled(&self, s: f64) -> Grad {
        Grad { dt: self.dt, t0: self.t0, g: self.g.iter().map(|v| v.map(|x| x * s)).collect() }
    }
    /// samples of the slew rate on each raster segment (constant there), T/m/s
    pub fn slew(&self) -> Vec<Vec3> {
        self.g.windows(2).map(|w| [0, 1, 2].map(|q| (w[1][q] - w[0][q]) / self.dt)).collect()
    }
    pub fn limits(&self) -> LimitReport {
        let ga = self.g.iter().flat_map(|v| v.iter().map(|x| x.abs())).fold(0.0, f64::max);
        let gv = self.g.iter().map(norm).fold(0.0, f64::max);
        let s = self.slew();
        let sa = s.iter().flat_map(|v| v.iter().map(|x| x.abs())).fold(0.0, f64::max);
        let sv = s.iter().map(norm).fold(0.0, f64::max);
        LimitReport { g_axis: ga, g_vec: gv, s_axis: sa, s_vec: sv }
    }
    /// exact cumulative moments with time origin `t_ref`
    pub fn moments(&self, t_ref: f64) -> Moments {
        let n = self.g.len();
        let mut m = Moments { t: Vec::with_capacity(n), m0: Vec::with_capacity(n), m1: Vec::with_capacity(n), m2: Vec::with_capacity(n), p: Vec::with_capacity(n) };
        let (mut m0, mut m1, mut m2, mut p) = ([0.0; 3], [0.0; 3], [0.0; 3], [0.0; 6]);
        for i in 0..n {
            let t = self.t0 + i as f64 * self.dt;
            if i > 0 {
                let (ga, gb) = (self.g[i - 1], self.g[i]);
                let gm = [0, 1, 2].map(|q| 0.5 * (ga[q] + gb[q]));
                let (ta, tb) = (t - self.dt - t_ref, t - t_ref);
                let tm = 0.5 * (ta + tb);
                let w = self.dt / 6.0;
                for q in 0..3 {
                    m0[q] += w * (ga[q] + 4.0 * gm[q] + gb[q]);
                    m1[q] += w * (ga[q] * ta + 4.0 * gm[q] * tm + gb[q] * tb);
                    m2[q] += w * (ga[q] * ta * ta + 4.0 * gm[q] * tm * tm + gb[q] * tb * tb);
                }
                for (k, (a, b)) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                    p[k] += w * (ga[a] * ga[b] + 4.0 * gm[a] * gm[b] + gb[a] * gb[b]);
                }
            }
            m.t.push(t);
            m.m0.push(m0);
            m.m1.push(m1);
            m.m2.push(m2);
            m.p.push(p);
        }
        m
    }
}

/// minimum-time trapezoid (one axis) with `area` (T·s/m) on raster `dt`: samples incl. the zero end points
pub fn trapezoid(area: f64, gmax: f64, smax: f64, dt: f64) -> Vec<f64> {
    if area == 0.0 {
        return vec![0.0];
    }
    let a = area.abs();
    let mut best: Option<(usize, usize)> = None;
    for nr in 1..1_000_000usize {
        let r = nr as f64 * dt;
        let gcap = gmax.min(smax * r);
        let f = (a / gcap - r).max(0.0);
        let nf = (f / dt - 1e-9).ceil().max(0.0) as usize;
        if best.map_or(true, |b| 2 * nr + nf < 2 * b.0 + b.1) {
            best = Some((nr, nf));
        }
        if smax * r >= gmax && best.map_or(false, |b| 2 * nr > 2 * b.0 + b.1) {
            break;
        }
    }
    let (nr, nf) = best.unwrap();
    let g = a / ((nr + nf) as f64 * dt);
    trap_shape(nr, nf, nr, g * area.signum())
}

/// trapezoid samples: `r1` ramp-up segments, `f` flat segments, `r2` ramp-down segments, amplitude `g`
pub fn trap_shape(r1: usize, f: usize, r2: usize, g: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(r1 + f + r2 + 1);
    for i in 0..=r1 {
        v.push(g * i as f64 / r1 as f64);
    }
    for _ in 0..f {
        v.push(g);
    }
    for i in (0..r2).rev() {
        v.push(g * i as f64 / r2 as f64);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoid_moments_closed_form() {
        // trapezoid ramp r, flat f, amplitude g starting at t = 0: M0 = g (r + f), M1 = M0 · (centre) = M0 (r + f/2 + r/2)…
        let dt = 10e-6;
        let s = trap_shape(10, 30, 10, 0.02);
        let w = Grad::from_axis(0, &s, dt, 0.0);
        let m = w.moments(0.0);
        let (m0, m1, m2, p) = m.last();
        let (r, f, g) = (10.0 * dt, 30.0 * dt, 0.02);
        let area = g * (r + f);
        assert!((m0[0] - area).abs() < 1e-15);
        let centre = r + f / 2.0;
        assert!((m1[0] - area * centre).abs() < 1e-15, "{} vs {}", m1[0], area * centre);
        // M2 about the centre: ∫ G (t − c)² for a symmetric trapezoid = g [ f³/12 + 2 ∫_0^r (s/r)(f/2 + r − s)² ds ]
        let m2c = {
            let w2 = Grad::from_axis(0, &s, dt, 0.0).moments(centre);
            w2.last().2[0]
        };
        let ramp: f64 = {
            // ∫_0^r (s/r) (f/2 + s')² with s' = r − s
            let n = 200000;
            (0..n).map(|i| {
                let s = (i as f64 + 0.5) / n as f64 * r;
                (s / r) * (f / 2.0 + r - s).powi(2) * r / n as f64
            }).sum()
        };
        let expect = g * (f.powi(3) / 12.0 + 2.0 * ramp);
        assert!((m2c - expect).abs() / expect < 1e-6, "{m2c} vs {expect}");
        assert!(m2[0] > 0.0);
        // P_xx = ∫ G² = g² (f + 2r/3)
        assert!((p[0] - g * g * (f + 2.0 * r / 3.0)).abs() < 1e-18);
    }

    #[test]
    fn min_time_trapezoid_respects_limits() {
        let dt = 10e-6;
        for area in [1e-6, 1e-5, 5e-5, 2e-4] {
            let s = trapezoid(area, 0.08, 200.0, dt);
            let w = Grad::from_axis(0, &s, dt, 0.0);
            let l = w.limits();
            assert!(l.g_axis <= 0.08 + 1e-12 && l.s_axis <= 200.0 * (1.0 + 1e-9), "{l:?}");
            assert!((w.moments(0.0).last().0[0] - area).abs() < 1e-14);
        }
    }

    #[test]
    fn rotation_preserves_norms_and_rotates_moments() {
        let dt = 10e-6;
        let w = Grad::from_axes(&trap_shape(5, 10, 5, 0.01), &trap_shape(4, 12, 4, -0.02), &[], dt, 0.0);
        let r = crate::axis_angle(&[1.0, 2.0, 0.5], 0.7);
        let wr = w.rotated(&r);
        let (a, b) = (w.moments(0.0).last().0, wr.moments(0.0).last().0);
        let ra = mat_vec(&r, &a);
        for q in 0..3 {
            assert!((ra[q] - b[q]).abs() < 1e-15);
        }
        assert!((w.limits().g_vec - wr.limits().g_vec).abs() < 1e-15);
    }
}
