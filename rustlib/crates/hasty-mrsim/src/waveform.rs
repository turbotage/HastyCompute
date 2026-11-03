//! Gradient waveforms (piecewise linear on a uniform raster) and their exact moments.

use crate::{GradLimits, Mat3, Vec3};

/// Gradient waveform, three logical axes, sampled at `t0 + i·dt`, linear in between, zero outside.
#[derive(Clone, Debug)]
pub struct Waveform {
    pub dt: f64,
    pub t0: f64,
    pub g: Vec<Vec3>,
}

/// Cumulative moments of a waveform with the time origin at t = 0 (the excitation), evaluated at the raster points.
/// `p` holds ∫ G Gᵀ dt in the order xx, yy, zz, xy, xz, yz.
#[derive(Clone, Debug)]
pub struct Moments {
    pub t: Vec<f64>,
    pub g: Vec<Vec3>,
    pub m0: Vec<Vec3>,
    pub m1: Vec<Vec3>,
    pub m2: Vec<Vec3>,
    pub p: Vec<[f64; 6]>,
}

/// Moments at one instant.
#[derive(Clone, Copy, Debug, Default)]
pub struct MomentSample {
    pub m0: Vec3,
    pub m1: Vec3,
    pub m2: Vec3,
    pub p: [f64; 6],
}

impl MomentSample {
    pub fn add(&self, o: &MomentSample) -> MomentSample {
        let mut r = *self;
        for a in 0..3 {
            r.m0[a] += o.m0[a];
            r.m1[a] += o.m1[a];
            r.m2[a] += o.m2[a];
        }
        for q in 0..6 {
            r.p[q] += o.p[q];
        }
        r
    }
    /// Moments of the waveform rotated by `r` (G → R G).
    pub fn rotate(&self, r: &Mat3) -> MomentSample {
        let pm = sym_from6(&self.p);
        let rp = crate::mat_mul(&crate::mat_mul(r, &pm), &crate::transpose(r));
        MomentSample { m0: crate::mat_vec(r, &self.m0), m1: crate::mat_vec(r, &self.m1), m2: crate::mat_vec(r, &self.m2), p: sym_to6(&rp) }
    }
}

pub fn sym_from6(p: &[f64; 6]) -> Mat3 {
    [[p[0], p[3], p[4]], [p[3], p[1], p[5]], [p[4], p[5], p[2]]]
}
pub fn sym_to6(m: &Mat3) -> [f64; 6] {
    [m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]]
}

impl Waveform {
    pub fn zeros(n: usize, dt: f64, t0: f64) -> Self {
        Self { dt, t0, g: vec![[0.0; 3]; n] }
    }
    /// Per-axis sample vectors (padded with zeros to a common length).
    pub fn from_axes(axes: [&[f64]; 3], dt: f64, t0: f64) -> Self {
        let n = axes.iter().map(|a| a.len()).max().unwrap_or(0);
        let g = (0..n).map(|i| [0, 1, 2].map(|a| axes[a].get(i).copied().unwrap_or(0.0))).collect();
        Self { dt, t0, g }
    }
    pub fn duration(&self) -> f64 {
        self.g.len().saturating_sub(1) as f64 * self.dt
    }
    pub fn t_end(&self) -> f64 {
        self.t0 + self.duration()
    }
    pub fn max_amplitude(&self) -> f64 {
        self.g.iter().map(|g| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()).fold(0.0, f64::max)
    }
    pub fn max_slew(&self) -> f64 {
        self.g
            .windows(2)
            .map(|w| ((0..3).map(|a| (w[1][a] - w[0][a]).powi(2)).sum::<f64>()).sqrt() / self.dt)
            .fold(0.0, f64::max)
    }
    pub fn scale(&mut self, f: f64) {
        for g in &mut self.g {
            for a in g.iter_mut() {
                *a *= f;
            }
        }
    }

    /// Exact cumulative moments (piecewise-linear integrals), time origin 0.
    pub fn moments(&self) -> Moments {
        let n = self.g.len();
        let h = self.dt;
        let mut m = Moments { t: Vec::with_capacity(n), g: self.g.clone(), m0: Vec::with_capacity(n), m1: Vec::with_capacity(n), m2: Vec::with_capacity(n), p: Vec::with_capacity(n) };
        let mut c = MomentSample::default();
        for i in 0..n {
            if i > 0 {
                let ta = self.t0 + (i - 1) as f64 * h;
                c = c.add(&segment(ta, &self.g[i - 1], &self.g[i], h));
            }
            m.t.push(self.t0 + i as f64 * h);
            m.m0.push(c.m0);
            m.m1.push(c.m1);
            m.m2.push(c.m2);
            m.p.push(c.p);
        }
        m
    }
}

/// Moment increments of a linear gradient segment from `ga` at `ta` to `gb` at `ta + h`.
fn segment(ta: f64, ga: &Vec3, gb: &Vec3, h: f64) -> MomentSample {
    let mut c = MomentSample::default();
    for a in 0..3 {
        let d = (gb[a] - ga[a]) / h;
        let a0 = ga[a];
        c.m0[a] = a0 * h + d * h * h / 2.0;
        c.m1[a] = a0 * ta * h + a0 * h * h / 2.0 + d * ta * h * h / 2.0 + d * h * h * h / 3.0;
        c.m2[a] = a0 * (ta * ta * h + ta * h * h + h * h * h / 3.0) + d * (ta * ta * h * h / 2.0 + 2.0 * ta * h * h * h / 3.0 + h.powi(4) / 4.0);
    }
    let idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    for (q, &(i1, i2)) in idx.iter().enumerate() {
        let (a1, a2) = (ga[i1], ga[i2]);
        let (d1, d2) = ((gb[i1] - ga[i1]) / h, (gb[i2] - ga[i2]) / h);
        c.p[q] = a1 * a2 * h + (a1 * d2 + a2 * d1) * h * h / 2.0 + d1 * d2 * h * h * h / 3.0;
    }
    c
}

impl Moments {
    pub fn last(&self) -> MomentSample {
        let i = self.t.len() - 1;
        MomentSample { m0: self.m0[i], m1: self.m1[i], m2: self.m2[i], p: self.p[i] }
    }
    /// Exact moments at any time (the cumulative value at the preceding raster point plus the exact partial segment).
    /// Clamped to the waveform span.
    pub fn at(&self, t: f64) -> MomentSample {
        let n = self.t.len();
        let dt = self.t[1] - self.t[0];
        let x = ((t - self.t[0]) / dt).clamp(0.0, (n - 1) as f64);
        let i = (x.floor() as usize).min(n - 2);
        let f = x - i as f64;
        let base = MomentSample { m0: self.m0[i], m1: self.m1[i], m2: self.m2[i], p: self.p[i] };
        if f <= 0.0 {
            return base;
        }
        let gb = [0, 1, 2].map(|a| self.g[i][a] * (1.0 - f) + self.g[i + 1][a] * f);
        base.add(&segment(self.t[i], &self.g[i], &gb, f * dt))
    }
}

/// Minimum-time trapezoid with the given area (T·s/m) on a raster `dt`; returns samples including the zero end points.
pub fn trapezoid(area: f64, lim: &GradLimits, dt: f64) -> Vec<f64> {
    if area == 0.0 {
        return vec![];
    }
    let a = area.abs();
    let mut best: Option<(usize, usize, f64)> = None; // (ramp samples, flat samples, amplitude)
    for nr in 1..100000usize {
        let r = nr as f64 * dt;
        let gcap = lim.gmax.min(lim.smax * r);
        let f = (a / gcap - r).max(0.0);
        let nf = (f / dt - 1e-9).ceil().max(0.0) as usize;
        let g = a / (r + nf as f64 * dt);
        let total = 2 * nr + nf;
        if best.map_or(true, |b| total < 2 * b.0 + b.1) {
            best = Some((nr, nf, g));
        }
        if lim.smax * r > lim.gmax && best.map_or(false, |b| 2 * nr > 2 * b.0 + b.1) {
            break;
        }
    }
    let (nr, nf, g) = best.unwrap();
    shape(nr, nf, g * area.signum())
}

fn shape(nr: usize, nf: usize, g: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(2 * nr + nf + 1);
    for i in 0..=nr {
        v.push(g * i as f64 / nr as f64);
    }
    for _ in 0..nf {
        v.push(g);
    }
    for i in (0..nr).rev() {
        v.push(g * i as f64 / nr as f64);
    }
    v
}

/// Minimum-time zero-M0 bipolar (two equal trapezoids of opposite sign) with first moment `m1` (T·s²/m), on raster `dt`.
/// For a zero-M0 waveform M1 does not depend on where it is played.
pub fn bipolar(m1: f64, lim: &GradLimits, dt: f64) -> Vec<f64> {
    if m1 == 0.0 {
        return vec![];
    }
    let a = m1.abs();
    let mut best: Option<(usize, usize, f64)> = None;
    for nr in 1..100000usize {
        let r = nr as f64 * dt;
        let gcap = lim.gmax.min(lim.smax * r);
        // |M1| = g (r+f)(2r+f) >= a  -> smallest f
        let disc = (r * r / 4.0 + a / gcap).sqrt();
        let f = (disc - 1.5 * r).max(0.0);
        let nf = (f / dt - 1e-9).ceil().max(0.0) as usize;
        let ff = nf as f64 * dt;
        let g = a / ((r + ff) * (2.0 * r + ff));
        let total = 2 * nr + nf;
        if best.map_or(true, |b| total < 2 * b.0 + b.1) {
            best = Some((nr, nf, g));
        }
        if lim.smax * r > lim.gmax && best.map_or(false, |b| 2 * nr > 2 * b.0 + b.1) {
            break;
        }
    }
    let (nr, nf, g) = best.unwrap();
    // positive-first lobe gives M1 = -g (r+f)(2r+f)
    let s = -m1.signum();
    let mut v = shape(nr, nf, s * g);
    let lobe2 = shape(nr, nf, -s * g);
    v.pop();
    v.extend(lobe2);
    v
}
