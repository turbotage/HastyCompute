//! Time-optimal gradient design for an arbitrary k-space curve under |G| ≤ Gmax, |dG/dt| ≤ Smax
//! (Lustig, Kim, Pauly, IEEE TMI 2008), and spiral curves.
//!
//! Works for any curve (spirals, cones, twisted/curly readouts), which is why it is used instead of closed-form spiral designs.

use crate::waveform::Waveform;
use crate::{GradLimits, Vec3, GAMMA_BAR};

/// A designed readout: gradient waveform (logical axes, starting at t = 0) and the time span of the ADC window.
#[derive(Clone, Debug)]
pub struct Readout {
    pub wave: Waveform,
    /// ADC window [start, end) relative to the waveform start
    pub adc: (f64, f64),
}

fn norm(v: &Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Traverse `curve` (k-space points, cycles/m, dense) as fast as the limits allow, starting at rest and ending at speed
/// `v_end` (`None` = unconstrained). Returns the gradient waveform on raster `dt` (G = 0 at t = 0).
pub fn time_optimal(curve: &[Vec3], lim: &GradLimits, dt: f64, v_end: Option<f64>) -> Waveform {
    // arc length
    let mut s = vec![0.0];
    for w in curve.windows(2) {
        let d = [w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]];
        s.push(s.last().unwrap() + norm(&d));
    }
    let total = *s.last().unwrap();
    // resample uniformly in arc length
    let ns = (curve.len() * 2).max(20000);
    let ds = total / (ns - 1) as f64;
    let mut c = Vec::with_capacity(ns);
    let mut j = 0;
    for i in 0..ns {
        let si = i as f64 * ds;
        while j + 2 < s.len() && s[j + 1] < si {
            j += 1;
        }
        let f = if s[j + 1] > s[j] { ((si - s[j]) / (s[j + 1] - s[j])).clamp(0.0, 1.0) } else { 0.0 };
        c.push([0, 1, 2].map(|a| curve[j][a] * (1.0 - f) + curve[j + 1][a] * f));
    }
    // curvature
    let kappa: Vec<f64> = (0..ns)
        .map(|i| {
            let (a, b) = (i.saturating_sub(1).min(ns - 3), i.saturating_sub(1).min(ns - 3) + 2);
            let m = (a + b) / 2;
            let dd = [0, 1, 2].map(|q| (c[b][q] - 2.0 * c[m][q] + c[a][q]) / (ds * ds));
            norm(&dd)
        })
        .collect();
    let gs = GAMMA_BAR * lim.smax; // max |d²k/dt²|, cycles/m/s²
    let vg = GAMMA_BAR * lim.gmax; // max |dk/dt|
    let vmax: Vec<f64> = kappa.iter().map(|&k| if k > 0.0 { vg.min((gs / k).sqrt()) } else { vg }).collect();
    let accel = |v: f64, k: f64| (gs * gs - (k * v * v).powi(2)).max(0.0).sqrt();
    let mut v = vec![0.0; ns];
    for i in 0..ns - 1 {
        // two half-steps (midpoint) for the tangential acceleration bound
        let a0 = accel(v[i], kappa[i]);
        let vm = (v[i] * v[i] + ds * a0).sqrt().min(vmax[i]);
        let a1 = accel(vm, 0.5 * (kappa[i] + kappa[i + 1]));
        v[i + 1] = (v[i] * v[i] + 2.0 * ds * a1).sqrt().min(vmax[i + 1]);
    }
    if let Some(ve) = v_end {
        v[ns - 1] = v[ns - 1].min(ve);
    }
    for i in (0..ns - 1).rev() {
        let a1 = accel(v[i + 1], kappa[i + 1]);
        let vm = (v[i + 1] * v[i + 1] + ds * a1).sqrt();
        let a0 = accel(vm, 0.5 * (kappa[i] + kappa[i + 1]));
        v[i] = v[i].min((v[i + 1] * v[i + 1] + 2.0 * ds * a0).sqrt());
    }
    // time along the curve
    let mut t = vec![0.0; ns];
    for i in 0..ns - 1 {
        let vv = 0.5 * (v[i] + v[i + 1]);
        t[i + 1] = t[i] + if vv > 0.0 { ds / vv } else { 0.0 };
    }
    let tend = t[ns - 1];
    let nt = (tend / dt).ceil() as usize + 1;
    // gradient at raster points: G = v · tangent / γ̄
    let mut g = Vec::with_capacity(nt);
    let mut j = 0;
    for k in 0..nt {
        let tk = k as f64 * dt;
        if tk >= tend {
            let i = ns - 2;
            let tan = [0, 1, 2].map(|q| (c[i + 1][q] - c[i][q]) / ds);
            g.push([0, 1, 2].map(|q| v[ns - 1] * tan[q] / GAMMA_BAR));
            continue;
        }
        while j + 2 < ns && t[j + 1] < tk {
            j += 1;
        }
        let f = if t[j + 1] > t[j] { ((tk - t[j]) / (t[j + 1] - t[j])).clamp(0.0, 1.0) } else { 0.0 };
        let vv = v[j] * (1.0 - f) + v[j + 1] * f;
        let tan = [0, 1, 2].map(|q| (c[j + 1][q] - c[j][q]) / ds);
        g.push([0, 1, 2].map(|q| vv * tan[q] / GAMMA_BAR));
    }
    g[0] = [0.0; 3];
    Waveform { dt, t0: 0.0, g }
}

/// Append a linear ramp to zero at the slew limit.
pub fn ramp_down(w: &mut Waveform, lim: &GradLimits) {
    let gl = *w.g.last().unwrap();
    let a = norm(&gl);
    if a == 0.0 {
        return;
    }
    let n = ((a / lim.smax) / w.dt).ceil() as usize;
    for i in 1..=n {
        let f = 1.0 - i as f64 / n as f64;
        w.g.push([gl[0] * f, gl[1] * f, gl[2] * f]);
    }
}

/// Variable-density spiral-out curve in the logical xy plane: dr/dθ = n_int / (2π FOV(r)), FOV linear in r from `fov0` at
/// the centre to `fov1` at `kmax` (cycles/m).
pub fn spiral_curve(kmax: f64, n_int: usize, fov0: f64, fov1: f64) -> Vec<Vec3> {
    let mut pts = vec![[0.0, 0.0, 0.0]];
    let (mut r, mut th) = (0.0f64, 0.0f64);
    while r < kmax {
        let fov = fov0 + (fov1 - fov0) * (r / kmax);
        let drdth = n_int as f64 / (2.0 * std::f64::consts::PI * fov);
        // step length |dk| = |dk/dθ| dθ ≈ kmax / 20000, and at most 0.05 rad
        let dth = ((kmax / 20000.0) / drdth.hypot(r)).min(0.05);
        r += drdth * dth;
        th += dth;
        pts.push([r.min(kmax) * th.cos(), r.min(kmax) * th.sin(), 0.0]);
    }
    pts
}

/// Spiral-out readout: time-optimal traversal of [`spiral_curve`], ADC over the spiral, then a ramp-down.
pub fn spiral_readout(kmax: f64, n_int: usize, fov0: f64, fov1: f64, lim: &GradLimits, dt: f64) -> Readout {
    let curve = spiral_curve(kmax, n_int, fov0, fov1);
    let mut w = time_optimal(&curve, lim, dt, None);
    let t_adc = w.duration();
    ramp_down(&mut w, lim);
    Readout { wave: w, adc: (0.0, t_adc) }
}
