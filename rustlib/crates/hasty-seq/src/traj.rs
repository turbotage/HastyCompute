//! Readout design: time-optimal traversal of a k-space curve, the base-spoke family, and rotation sets.
//!
//! Time-optimal traversal (Lustig, Kim, Pauly, IEEE TMI 2008): with arc length s, speed v = ds/dt and curvature κ,
//! k̈ = v̇ T + κ v² N, so ‖G‖ ≤ G_max and ‖Ġ‖ ≤ S_max become v ≤ γ̄ G_max, v ≤ √(γ̄ S/κ) and
//! v̇² + κ² v⁴ ≤ (γ̄ S)², solved by a forward and a backward sweep in s. These are *vector* limits: the base readout then
//! satisfies the per-axis limits under every rotation. Extras over the classical algorithm: a speed cap (ADC bandwidth:
//! along-curve sample spacing v·dwell ≤ 1/FOV) and a slew cap that varies along the curve (PNS shaping).
//!
//! Base spoke ("yarn"): k(ℓ) = r(ℓ) d(σ(ℓ)) with the radius law fixed by the target density. A curve family crossing the
//! shell r with path length q(r) per unit radius leaves N q / (4π r²) of curve length per unit volume when N randomly
//! rotated copies are acquired; Nyquist for a FOV D needs D² (curves ≤ 1/D apart), so
//!   q(r) ≥ f(r) = 4π r² D² w(r) / N      (w = relative density, 1 = uniform; q = √(1 + f²) here),
//! and the direction d(σ) moves on the unit sphere by dσ/dr = √(q² − 1)/r along a path of geodesic curvature
//! k_g(σ) = k_g0 + A sin(ω σ): A = 0, k_g0 = 0 is a great circle (a rotated planar spiral, lowest curvature), k_g0 ≠ 0
//! circles an axis (cone-like), A ≠ 0 meanders (yarnball-like, sweeps the sphere: isotropic concomitant terms).

use crate::system::System;
use crate::wave::{trapezoid, Grad};
use crate::{cross, dot, mat_mul, norm, Mat3, Vec3, GAMMA_BAR};

#[derive(Clone, Debug)]
pub struct TopOpts {
    /// speed at the curve end (None = free)
    pub v_end: Option<f64>,
    /// speed cap, cycles/m/s (e.g. 1 / (FOV · dwell))
    pub vmax: Option<f64>,
    /// slew-limit factors (≤ 1) at uniformly spaced arc-length fractions [0, 1], linearly interpolated
    pub slew_cap: Option<Vec<f64>>,
    /// design margin on both limits (discretisation headroom)
    pub margin: f64,
}

impl Default for TopOpts {
    fn default() -> Self {
        Self { v_end: None, vmax: None, slew_cap: None, margin: 0.995 }
    }
}

/// a traversal: gradient on the raster (G = 0 at t = 0), and the arc-length fraction reached at each raster point
#[derive(Clone, Debug)]
pub struct Traversal {
    pub grad: Grad,
    pub frac: Vec<f64>,
    pub length: f64,
}

fn interp(table: &[f64], x: f64) -> f64 {
    if table.len() == 1 {
        return table[0];
    }
    let f = x.clamp(0.0, 1.0) * (table.len() - 1) as f64;
    let i = (f.floor() as usize).min(table.len() - 2);
    let a = f - i as f64;
    table[i] * (1.0 - a) + table[i + 1] * a
}

/// Menger curvature at the interior vertices of a polyline (end points copy their neighbours)
pub fn curvature(c: &[Vec3]) -> Vec<f64> {
    let n = c.len();
    let mut k = vec![0.0; n];
    for i in 1..n - 1 {
        let (u, w) = (crate::sub(&c[i], &c[i - 1]), crate::sub(&c[i + 1], &c[i - 1]));
        let (a, b, cc) = (norm(&u), norm(&crate::sub(&c[i + 1], &c[i])), norm(&w));
        let den = a * b * cc;
        k[i] = if den > 0.0 { 2.0 * norm(&cross(&u, &w)) / den } else { 0.0 };
    }
    if n > 2 {
        k[0] = k[1];
        k[n - 1] = k[n - 2];
    }
    k
}

/// traverse `curve` (cycles/m, dense, starting at rest) as fast as ‖G‖ ≤ gmax, ‖Ġ‖ ≤ smax (and the options) allow
pub fn time_optimal(curve: &[Vec3], gmax: f64, smax: f64, dt: f64, o: &TopOpts) -> Traversal {
    assert!(curve.len() >= 3);
    let n_in = curve.len();
    let mut s_in = vec![0.0; n_in];
    for i in 1..n_in {
        s_in[i] = s_in[i - 1] + norm(&crate::sub(&curve[i], &curve[i - 1]));
    }
    let total = s_in[n_in - 1];
    let kap_in = curvature(curve);
    let tan_in: Vec<Vec3> = (0..n_in)
        .map(|i| {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(n_in - 1));
            let d = crate::sub(&curve[b], &curve[a]);
            let l = norm(&d);
            if l > 0.0 {
                crate::scale(&d, 1.0 / l)
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect();
    let vg = GAMMA_BAR * gmax * o.margin;
    let vcap = o.vmax.map_or(vg, |v| v.min(vg));
    // arc-length grid: ~16 points per raster step at full speed
    let ns = ((total / (vcap * dt / 16.0)).ceil() as usize).clamp(1000, 4_000_000);
    let ds = total / (ns - 1) as f64;
    let mut kap = vec![0.0; ns];
    let mut tan = vec![[0.0; 3]; ns];
    let mut j = 0;
    for i in 0..ns {
        let si = i as f64 * ds;
        while j + 2 < n_in && s_in[j + 1] < si {
            j += 1;
        }
        let f = if s_in[j + 1] > s_in[j] { ((si - s_in[j]) / (s_in[j + 1] - s_in[j])).clamp(0.0, 1.0) } else { 0.0 };
        kap[i] = kap_in[j] * (1.0 - f) + kap_in[j + 1] * f;
        let t = [0, 1, 2].map(|q| tan_in[j][q] * (1.0 - f) + tan_in[j + 1][q] * f);
        let l = norm(&t);
        tan[i] = crate::scale(&t, 1.0 / l);
    }
    let gs: Vec<f64> = (0..ns).map(|i| GAMMA_BAR * smax * o.margin * o.slew_cap.as_ref().map_or(1.0, |c| interp(c, i as f64 / (ns - 1) as f64))).collect();
    let vlim: Vec<f64> = (0..ns).map(|i| if kap[i] > 0.0 { vcap.min((gs[i] / kap[i]).sqrt()) } else { vcap }).collect();
    let accel = |v: f64, k: f64, g: f64| (g * g - (k * v * v).powi(2)).max(0.0).sqrt();
    let mut v = vec![0.0; ns];
    for i in 0..ns - 1 {
        let a0 = accel(v[i], kap[i], gs[i]);
        let vm = (v[i] * v[i] + ds * a0).sqrt().min(0.5 * (vlim[i] + vlim[i + 1]));
        let a1 = accel(vm, 0.5 * (kap[i] + kap[i + 1]), 0.5 * (gs[i] + gs[i + 1]));
        v[i + 1] = (v[i] * v[i] + 2.0 * ds * a1).sqrt().min(vlim[i + 1]);
    }
    if let Some(ve) = o.v_end {
        v[ns - 1] = v[ns - 1].min(ve);
    }
    for i in (0..ns - 1).rev() {
        let a1 = accel(v[i + 1], kap[i + 1], gs[i + 1]);
        let vm = (v[i + 1] * v[i + 1] + ds * a1).sqrt();
        let a0 = accel(vm, 0.5 * (kap[i] + kap[i + 1]), 0.5 * (gs[i] + gs[i + 1]));
        v[i] = v[i].min((v[i + 1] * v[i + 1] + 2.0 * ds * a0).sqrt());
    }
    let mut t = vec![0.0; ns];
    for i in 0..ns - 1 {
        let vv = 0.5 * (v[i] + v[i + 1]);
        t[i + 1] = t[i] + if vv > 0.0 { ds / vv } else { 0.0 };
    }
    let tend = t[ns - 1];
    let nt = (tend / dt - 1e-9).ceil() as usize + 1;
    let mut g = Vec::with_capacity(nt);
    let mut frac = Vec::with_capacity(nt);
    let mut j = 0;
    for k in 0..nt {
        let tk = k as f64 * dt;
        if tk >= tend {
            g.push(crate::scale(&tan[ns - 1], v[ns - 1] / GAMMA_BAR));
            frac.push(1.0);
            continue;
        }
        while j + 2 < ns && t[j + 1] < tk {
            j += 1;
        }
        let f = if t[j + 1] > t[j] { ((tk - t[j]) / (t[j + 1] - t[j])).clamp(0.0, 1.0) } else { 0.0 };
        let vv = v[j] * (1.0 - f) + v[j + 1] * f;
        let tt = [0, 1, 2].map(|q| tan[j][q] * (1.0 - f) + tan[j + 1][q] * f);
        let l = norm(&tt);
        g.push(crate::scale(&tt, vv / (GAMMA_BAR * l)));
        frac.push((j as f64 + f) * ds / total);
    }
    g[0] = [0.0; 3];
    Traversal { grad: Grad { dt, t0: 0.0, g }, frac, length: total }
}

/// vector ramp to zero at the slew limit, appended
pub fn ramp_down(w: &mut Grad, smax: f64) {
    let gl = *w.g.last().unwrap();
    let a = norm(&gl);
    if a == 0.0 {
        return;
    }
    let n = ((a / smax) / w.dt - 1e-9).ceil() as usize;
    for i in 1..=n {
        w.g.push(crate::scale(&gl, 1.0 - i as f64 / n as f64));
    }
}

/// minimum-time single-direction trapezoid with zeroth moment `area` (T·s/m vector): vector and per-axis limits coincide
pub fn vector_trapezoid(area: &Vec3, gmax: f64, smax: f64, dt: f64) -> Grad {
    let a = norm(area);
    if a == 0.0 {
        return Grad::zeros(1, dt, 0.0);
    }
    let s = trapezoid(a, gmax, smax, dt);
    let u = crate::scale(area, 1.0 / a);
    Grad { dt, t0: 0.0, g: s.iter().map(|x| crate::scale(&u, *x)).collect() }
}

/// a base spoke: the readout (ADC window) in the base frame, t = 0 at the readout start (the echo for center-out);
/// ramp-down, rewinder and spoiler are one per-rotation [`tail`] (see `tr`)
#[derive(Clone, Debug)]
pub struct Spoke {
    /// readout gradient, samples 0..=i_adc (G(0) = 0, G(i_adc) generally ≠ 0)
    pub grad: Grad,
    /// ADC window [0, t_adc)
    pub t_adc: f64,
    pub i_adc: usize,
    pub length: f64,
    /// arc-length fraction at the raster points of the ADC window
    pub frac: Vec<f64>,
    /// zeroth moment at the ADC end (base frame)
    pub m0_end: Vec3,
}

impl Spoke {
    /// time-optimal readout of `curve` with vector limits `sys` (gmax, smax)
    pub fn design(curve: &[Vec3], sys: &System, o: &TopOpts) -> Spoke {
        let tr = time_optimal(curve, sys.gmax, sys.smax, sys.grad_raster, o);
        let i_adc = tr.grad.len() - 1;
        let t_adc = tr.grad.duration();
        let m0_end = tr.grad.moments(0.0).last().0;
        Spoke { grad: tr.grad, t_adc, i_adc, length: tr.length, frac: tr.frac, m0_end }
    }
    /// end exactly at zeroth moment `target` (T·s/m): the raster traversal of a curve ending at rest misses its end
    /// point by a small discretisation residual, which would rotate with the spoke and make the TR's net moment jitter;
    /// a minimal vector trapezoid closes it (part of the readout, a few raster points)
    pub fn close_to(&mut self, target: &Vec3, gmax: f64, smax: f64) {
        assert!(norm(&self.g_end()) == 0.0, "close_to needs a readout ending at rest");
        let d = crate::sub(target, &self.m0_end);
        if norm(&d) == 0.0 {
            return;
        }
        let blip = vector_trapezoid(&d, gmax, smax, self.grad.dt);
        self.grad.g.extend_from_slice(&blip.g[1..]);
        self.frac.extend(std::iter::repeat(1.0).take(blip.len() - 1));
        self.i_adc = self.grad.len() - 1;
        self.t_adc = self.grad.duration();
        self.m0_end = self.grad.moments(0.0).last().0;
    }
    /// actual k (cycles/m) at `m`-times upsampled raster points over the ADC window
    pub fn k_adc(&self, m: usize) -> Vec<Vec3> {
        upsample(&self.grad, m).moments(0.0).k()
    }
    pub fn g_end(&self) -> Vec3 {
        self.grad.g[self.i_adc]
    }
}

/// Shortest tail from a readout end (gradient `g0`, zeroth moment `m0`) to G = 0 with zeroth moment `target`, under
/// vector limits: a linear ramp-down of g0 at slew α·smax overlapped with a single-direction trapezoid (slew (1−α)·smax,
/// amplitude β·gmax) carrying the remaining area; searched over (α, β) plus the sequential ramp-then-trapezoid, every
/// candidate checked against the limits. Sample 0 is g0.
pub fn tail(g0: &Vec3, m0: &Vec3, target: &Vec3, gmax: f64, smax: f64, dt: f64) -> Grad {
    let a0 = norm(g0);
    let ok = |w: &Grad| {
        let l = w.limits();
        l.g_vec <= gmax * (1.0 + 1e-9) && l.s_vec <= smax * (1.0 + 1e-9)
    };
    let build = |alpha: f64, beta: f64, sequential: bool| -> Grad {
        let nr = if a0 > 0.0 { ((a0 / (alpha * smax)) / dt - 1e-9).ceil() as usize } else { 0 };
        let m_ramp = crate::scale(g0, nr as f64 * dt / 2.0);
        let rest = crate::sub(&crate::sub(target, m0), &m_ramp);
        let ar = norm(&rest);
        let trap = if ar > 0.0 { trapezoid(ar, beta * gmax, if sequential { smax } else { (1.0 - alpha) * smax }, dt) } else { vec![0.0] };
        let u = if ar > 0.0 { crate::scale(&rest, 1.0 / ar) } else { [0.0; 3] };
        let off = if sequential { nr } else { 0 };
        let n = (nr + 1).max(off + trap.len());
        let g = (0..n)
            .map(|i| {
                let r = if i <= nr && nr > 0 { crate::scale(g0, 1.0 - i as f64 / nr as f64) } else if nr == 0 && i == 0 { *g0 } else { [0.0; 3] };
                let tv = if i >= off && i - off < trap.len() { trap[i - off] } else { 0.0 };
                [0, 1, 2].map(|q| r[q] + u[q] * tv)
            })
            .collect();
        Grad { dt, t0: 0.0, g }
    };
    let mut best = build(1.0, 1.0, true);
    assert!(ok(&best), "sequential tail violates the limits");
    if a0 > 0.0 {
        for ia in 1..20 {
            let alpha = ia as f64 / 20.0;
            for beta in [1.0, 0.85, 0.7, 0.55, 0.4] {
                let w = build(alpha, beta, false);
                if w.len() < best.len() && ok(&w) {
                    best = w;
                }
            }
        }
    }
    best
}

/// piecewise-linear waveform on an `m`-times finer raster (exact)
pub fn upsample(w: &Grad, m: usize) -> Grad {
    let mut g = Vec::with_capacity((w.len() - 1) * m + 1);
    for i in 0..w.len() - 1 {
        for j in 0..m {
            let a = j as f64 / m as f64;
            g.push([0, 1, 2].map(|q| w.g[i][q] * (1.0 - a) + w.g[i + 1][q] * a));
        }
    }
    g.push(*w.g.last().unwrap());
    Grad { dt: w.dt / m as f64, t0: w.t0, g }
}

/// base-spoke family (see the module doc)
#[derive(Clone, Debug)]
pub struct Yarn {
    /// cycles/m
    pub kmax: f64,
    /// m
    pub fov: f64,
    /// spokes giving Nyquist density
    pub n_design: f64,
    /// relative density at kmax (w(r) = 1 + (edge − 1) (r/kmax)²)
    pub edge_density: f64,
    /// constant geodesic curvature of the direction path
    pub kg0: f64,
    /// meander amplitude and frequency (per radian of direction-path arc)
    pub wobble: f64,
    pub wobble_freq: f64,
}

impl Yarn {
    pub fn w(&self, r: f64) -> f64 {
        1.0 + (self.edge_density - 1.0) * (r / self.kmax).powi(2)
    }
    /// path length per unit radius: q = √(1 + f²), f = 4π r² D² w / N (a smooth max(1, f): tan α = f for the angle α
    /// to the radial direction, so the curvature stays finite where the winding starts; slightly denser than needed
    /// around f ≈ 1, exact for f ≫ 1)
    pub fn q(&self, r: f64) -> f64 {
        let f = 4.0 * std::f64::consts::PI * r * r * self.fov * self.fov * self.w(r) / self.n_design;
        (1.0 + f * f).sqrt()
    }
    /// curve length ∫₀^kmax q dr
    pub fn length(&self) -> f64 {
        let n = 4000;
        let h = self.kmax / n as f64;
        (0..n).map(|i| self.q((i as f64 + 0.5) * h) * h).sum()
    }
    fn kg(&self, sigma: f64) -> f64 {
        self.kg0 + self.wobble * (self.wobble_freq * sigma).sin()
    }
    /// dense curve (≈ `npts` points, uniform in arc length), starting at 0 along +z, first heading +x
    pub fn curve(&self, npts: usize) -> Vec<Vec3> {
        let h = self.length() / npts as f64;
        let (mut d, mut t, mut nn) = ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let (mut r, mut sigma) = (0.0f64, 0.0f64);
        let mut out = Vec::with_capacity(npts + 2);
        out.push([0.0; 3]);
        let mut it = 0usize;
        while r < self.kmax {
            let q0 = self.q(r);
            let rm = r + 0.5 * h / q0;
            let qm = self.q(rm);
            let mut dr = h / qm;
            let mut hh = h;
            if r + dr > self.kmax {
                dr = self.kmax - r;
                hh = dr * qm;
            }
            let rmid = r + 0.5 * dr;
            let dsig = if qm > 1.0 { hh * (qm * qm - 1.0).sqrt() / (qm * rmid) } else { 0.0 };
            if dsig > 0.0 {
                // frame rotation about Ω = k_g d + n by |Ω| dσ
                let kg = self.kg(sigma + 0.5 * dsig);
                let om = [0, 1, 2].map(|i| kg * d[i] + nn[i]);
                let rot = crate::axis_angle(&om, norm(&om) * dsig);
                d = crate::mat_vec(&rot, &d);
                t = crate::mat_vec(&rot, &t);
                nn = crate::mat_vec(&rot, &nn);
                it += 1;
                if it % 256 == 0 {
                    // re-orthonormalise
                    let l = norm(&d);
                    d = crate::scale(&d, 1.0 / l);
                    let tp = crate::sub(&t, &crate::scale(&d, dot(&t, &d)));
                    t = crate::scale(&tp, 1.0 / norm(&tp));
                    nn = cross(&d, &t);
                }
            }
            r += dr;
            sigma += dsig;
            out.push(crate::scale(&d, r));
        }
        out
    }
    /// Out-and-back curve: the centre-out curve to kmax, then back in along the same curve rotated by π about its end
    /// direction and reversed, down to radius `ret`·kmax (0 = back to the centre). The rotation keeps the tangential
    /// velocity at the turn-around, only the small radial component (≈ 1/q) flips; that kink is Gaussian-smoothed over
    /// a length ≈ kink angle · kmax (adds curvature ≈ 1/kmax, the shell's own). Every point stays inside the kmax ball
    /// (smoothing takes convex combinations). The return leg is a congruent copy, so each shell it crosses gets twice
    /// the coverage per spoke: N spokes of the out-and-back curve designed with `n_design` = 2N are Nyquist for ret = 0
    /// (see [`Yarn::n_eff`]).
    pub fn curve_return(&self, npts: usize, ret: f64) -> Vec<Vec3> {
        let out = self.curve(npts);
        let m = out.len();
        let end = out[m - 1];
        let rot = crate::axis_angle(&end, std::f64::consts::PI);
        let r_stop = ret.clamp(0.0, 1.0) * self.kmax;
        let mut c = out.clone();
        for p in out[..m - 1].iter().rev() {
            if norm(p) < r_stop {
                break;
            }
            c.push(crate::mat_vec(&rot, p));
        }
        // smooth the turn-around kink at index m-1
        let j = m - 1;
        if j >= 8 && j + 8 < c.len() {
            // mean point spacing (the last out step is clipped at kmax and can be tiny); chords over a few points
            let h = self.length() / (m - 1) as f64;
            let u = crate::sub(&c[j], &c[j - 6]);
            let w = crate::sub(&c[j + 6], &c[j]);
            let kink = (dot(&u, &w) / (norm(&u) * norm(&w))).clamp(-1.0, 1.0).acos();
            let sig = ((kink * self.kmax / 2.5) / h).clamp(3.0, 0.01 * m as f64);
            let half = (4.0 * sig).ceil() as usize;
            let lo = j.saturating_sub(half);
            let hi = (j + half).min(c.len() - 1);
            let src = c.clone();
            for i in lo..=hi {
                let (mut acc, mut wsum) = ([0.0; 3], 0.0);
                let a = i.saturating_sub(half).max(0);
                let b = (i + half).min(src.len() - 1);
                for (k, p) in src.iter().enumerate().take(b + 1).skip(a) {
                    let wt = (-0.5 * ((k as f64 - i as f64) / sig).powi(2)).exp();
                    acc = [0, 1, 2].map(|q| acc[q] + wt * p[q]);
                    wsum += wt;
                }
                c[i] = crate::scale(&acc, 1.0 / wsum);
            }
        }
        c
    }
    /// spokes needed for Nyquist with the out-and-back curve of [`Yarn::curve_return`]: shells above `ret`·kmax are
    /// crossed twice per spoke, those below once
    pub fn n_eff(&self, ret: f64) -> f64 {
        let r_stop = ret.clamp(0.0, 1.0) * self.kmax;
        let mut n = 0.5 * self.n_design;
        let steps = 400;
        for i in 1..=steps {
            let r = r_stop * i as f64 / steps as f64;
            let a = 4.0 * std::f64::consts::PI * r * r * self.fov * self.fov * self.w(r);
            n = n.max(a / self.q(r));
        }
        n
    }
    /// total turning of the direction path, rad
    pub fn sigma_total(&self) -> f64 {
        let n = 4000;
        let h = self.kmax / n as f64;
        (0..n)
            .map(|i| {
                let r = (i as f64 + 0.5) * h;
                let q = self.q(r);
                if q > 1.0 { (q * q - 1.0).sqrt() / r * h } else { 0.0 }
            })
            .sum()
    }
}

/// rotation-set constructions
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotSet {
    /// Fibonacci-sphere directions × in-plane angles 2π·frac(i·0.7548776662) (ZYZ)
    FibonacciGolden,
    /// super-Fibonacci low-discrepancy set on SO(3) (Alexa, CVPR 2022)
    SuperFibonacci,
    /// multidimensional golden means (cos θ, φ, ψ), temporally incoherent in acquisition order
    GoldenMeans,
}

fn rz(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}
fn ry(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}
/// R = Rz(φ) Ry(θ) Rz(ψ): base +z goes to the direction (θ, φ), ψ spins the spoke about it
pub fn zyz(phi: f64, theta: f64, psi: f64) -> Mat3 {
    mat_mul(&rz(phi), &mat_mul(&ry(theta), &rz(psi)))
}

fn frac(x: f64) -> f64 {
    x - x.floor()
}

/// largest root of x^{d+1} = x^d + 1 (the d-dimensional golden ratio of Anderson's multidimensional golden means)
fn golden_root(d: i32) -> f64 {
    let mut x = 1.5f64;
    for _ in 0..100 {
        let f = x.powi(d + 1) - x.powi(d) - 1.0;
        let df = (d + 1) as f64 * x.powi(d) - d as f64 * x.powi(d - 1);
        x -= f / df;
    }
    x
}

pub fn rotations(kind: RotSet, n: usize) -> Vec<Mat3> {
    let tau = 2.0 * std::f64::consts::PI;
    match kind {
        RotSet::FibonacciGolden => {
            let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
            (0..n)
                .map(|i| {
                    let z = 1.0 - (2.0 * i as f64 + 1.0) / n as f64;
                    zyz(i as f64 * ga, z.acos(), tau * frac(i as f64 * 0.754_877_666_246_692_8))
                })
                .collect()
        }
        RotSet::SuperFibonacci => {
            let phi = 2f64.sqrt();
            let psi = 1.533_751_168_755_204_3;
            (0..n)
                .map(|i| {
                    let s = i as f64 + 0.5;
                    let t = s / n as f64;
                    let (a, b) = (tau * s / phi, tau * s / psi);
                    let (r, rr) = (t.sqrt(), (1.0 - t).sqrt());
                    crate::quat_to_mat([r * a.sin(), r * a.cos(), rr * b.sin(), rr * b.cos()])
                })
                .collect()
        }
        RotSet::GoldenMeans => {
            let l = golden_root(3);
            let (g1, g2, g3) = (1.0 / l, 1.0 / (l * l), 1.0 / (l * l * l));
            (0..n).map(|i| {
                let i = i as f64;
                zyz(tau * frac(i * g2), (2.0 * frac(i * g1) - 1.0).acos(), tau * frac(i * g3))
            }).collect()
        }
    }
}

/// indices sorted by frac(i·g): consecutive positions are far apart in i
pub fn irrational_order(n: usize, g: f64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| frac(a as f64 * g).partial_cmp(&frac(b as f64 * g)).unwrap());
    idx
}

/// acquisition order of a rotation set, so that every contiguous run of spokes (a cardiac / respiratory bin) is spread
/// over SO(3). The sorting constant must be unrelated to the set's own generators: sorting the Fibonacci×golden set by
/// frac(i/Φ) would pick one azimuth wedge (its azimuth is −frac(i/Φ)·2π), so that set is sorted by frac(i·√2); the
/// super-Fibonacci set (generators √2, 1.5338) by frac(i/Φ); golden means are incoherent in their natural order.
pub fn acquisition_order(kind: RotSet, n: usize) -> Vec<usize> {
    match kind {
        RotSet::FibonacciGolden => irrational_order(n, 2f64.sqrt()),
        RotSet::SuperFibonacci => irrational_order(n, 2.0 / (1.0 + 5f64.sqrt())),
        RotSet::GoldenMeans => (0..n).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_is_bang_bang_and_respects_limits() {
        let sys = System::prisma_xr();
        let curve: Vec<Vec3> = (0..=1000).map(|i| [0.0, 0.0, i as f64 * 0.25]).collect(); // 250 cycles/m radial
        let tr = time_optimal(&curve, sys.gmax, sys.smax, sys.grad_raster, &TopOpts::default());
        let l = tr.grad.limits();
        assert!(l.ok_vec(&sys), "{l:?}");
        let k = tr.grad.moments(0.0).k();
        let kend = k.last().unwrap()[2];
        assert!((kend - 250.0).abs() < 0.6 * GAMMA_BAR * sys.gmax * sys.grad_raster, "{kend}");
        // analytic: ramp to 80 mT/m in 0.4 ms covers γ̄ G² / (2 S) = 681 cycles/m > 250, so a triangle-ish ramp at S
        let t_expect = (2.0 * 250.0 / (GAMMA_BAR * sys.smax)).sqrt();
        assert!((tr.grad.duration() - t_expect).abs() < 2.0 * sys.grad_raster, "{} vs {t_expect}", tr.grad.duration());
    }

    #[test]
    fn yarn_has_the_designed_density_and_respects_vector_limits() {
        let sys = System::prisma_xr();
        let y = Yarn { kmax: 250.0, fov: 0.3, n_design: 4000.0, edge_density: 1.0, kg0: 0.0, wobble: 1.0, wobble_freq: 2.0 };
        let c = y.curve(200_000);
        // radius ends at kmax, length matches the radius law
        assert!((norm(c.last().unwrap()) - 250.0).abs() < 1e-6);
        let len: f64 = c.windows(2).map(|w| norm(&crate::sub(&w[1], &w[0]))).sum();
        assert!((len - y.length()).abs() / y.length() < 2e-3, "{len} vs {}", y.length());
        let sp = Spoke::design(&c, &sys, &TopOpts::default());
        let l = sp.grad.limits();
        assert!(l.ok_vec(&sys), "{l:?}");
        // a tail to an arbitrary moment lands on it, within the limits, and is not longer than ramp-then-trapezoid
        let target = [3e-5, -1e-5, 2e-5];
        let t = tail(&sp.g_end(), &sp.m0_end, &target, sys.gmax, sys.smax, sys.grad_raster);
        assert!(norm(&crate::sub(&t.g[0], &sp.g_end())) == 0.0);
        let mut g = sp.grad.g.clone();
        g.extend_from_slice(&t.g[1..]);
        let full = Grad { dt: t.dt, t0: 0.0, g };
        let m0 = full.moments(0.0).last().0;
        assert!(norm(&crate::sub(&m0, &target)) < 1e-12, "{m0:?}");
        assert!(full.limits().ok_vec(&sys), "{:?}", full.limits());
        assert!(norm(full.g.last().unwrap()) == 0.0);
        // rotated: per-axis limits hold
        for r in rotations(RotSet::SuperFibonacci, 50) {
            assert!(sp.grad.rotated(&r).limits().ok_axis(&sys));
        }
    }

    #[test]
    fn rotation_sets_are_orthonormal_and_spread() {
        for kind in [RotSet::FibonacciGolden, RotSet::SuperFibonacci, RotSet::GoldenMeans] {
            let rs = rotations(kind, 2000);
            let mut mean = [0.0; 3];
            for r in &rs {
                let rt = crate::transpose(r);
                let p = mat_mul(r, &rt);
                for i in 0..3 {
                    for j in 0..3 {
                        assert!((p[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
                    }
                }
                let e = crate::mat_vec(r, &[0.0, 0.0, 1.0]);
                for q in 0..3 {
                    mean[q] += e[q] / rs.len() as f64;
                }
            }
            assert!(norm(&mean) < 0.02, "{kind:?}: mean direction {mean:?}");
        }
    }
}

#[cfg(test)]
mod return_tests {
    use super::*;

    #[test]
    fn curve_return_inside_and_closed() {
        let y = Yarn { kmax: 250.0, fov: 0.32, n_design: 2000.0, edge_density: 1.0, kg0: 0.0, wobble: 0.0, wobble_freq: 2.0 };
        let c = y.curve_return(60_000, 0.0);
        let out = y.curve(60_000);
        let rmax = c.iter().map(norm).fold(0.0, f64::max);
        let len: f64 = c.windows(2).map(|w| norm(&crate::sub(&w[1], &w[0]))).sum();
        println!("out pts {} total pts {} rmax {rmax} end {:?} length {len} (out {})", out.len(), c.len(), c.last().unwrap(), y.length());
        assert!(rmax <= y.kmax * (1.0 + 1e-12));
        assert!(norm(c.last().unwrap()) < 1e-9);
        assert!((len / (2.0 * y.length()) - 1.0).abs() < 0.01);
    }
}
