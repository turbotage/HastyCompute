//! Flow-based vessel measurements: wall shear stress on a vessel mesh and pulse wave velocity along a centerline.
//! Both take a flow value (`flow.rs`: velocity [F,3,Z,Y,X] in m/s, spacing in m, dt in s) and a geometry that was built from the same voxel grid
//! (so a mesh in `voxel_mm` space, or in RAS space together with the grid's affine).

use std::sync::Arc;

use hasty_value::Value;
use serde_json::json;

use crate::commands::Registry;
use crate::flow::Flow;
use crate::mesh::{affine_opt, apply, invert, mesh_in};
use crate::Error;

type V3 = [f64; 3];

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

type Map3 = Box<dyn Fn(V3) -> V3>;

/// (positions: mm in mesh / line space → flow voxel coordinates, directions: unit vector in mesh space → unit vector along the flow grid's axes,
/// the frame the velocity components are expressed in)
fn to_voxel(space: &str, flow: &Flow, affine: Option<[f64; 16]>, who: &str) -> Result<(Map3, Map3), Error> {
    let sp_mm = flow.spacing.map(|s| s * 1e3);
    let unit = |v: V3| { let l = dot(v, v).sqrt(); if l > 0.0 { [v[0] / l, v[1] / l, v[2] / l] } else { v } };
    match space {
        "ras" => {
            let a = affine.as_ref().and_then(invert).ok_or_else(|| Error::Invalid(format!("{who}: the geometry is in RAS space: options.affine of the flow grid required")))?;
            let sp = flow.spacing;
            let dir = move |n: V3| {
                let dv = [a[0] * n[0] + a[1] * n[1] + a[2] * n[2], a[4] * n[0] + a[5] * n[1] + a[6] * n[2], a[8] * n[0] + a[9] * n[1] + a[10] * n[2]];
                let l = [dv[0] * sp[0], dv[1] * sp[1], dv[2] * sp[2]];
                let n2 = dot(l, l).sqrt();
                if n2 > 0.0 { [l[0] / n2, l[1] / n2, l[2] / n2] } else { l }
            };
            Ok((Box::new(move |p| apply(&a, p)), Box::new(dir)))
        }
        "voxel" => Ok((Box::new(|p| p), Box::new(move |n| unit(n)))),
        _ => Ok((Box::new(move |p| [p[0] / sp_mm[0], p[1] / sp_mm[1], p[2] / sp_mm[2]]), Box::new(move |n| unit(n)))),
    }
}

fn flow_in(v: Option<&Arc<Value>>, who: &str) -> Result<Flow, Error> {
    Flow::from_value(v.ok_or_else(|| Error::Invalid(format!("{who}: the flow value is missing")))?)
}


// ── pulse wave velocity from waveforms ───────────────────────────────────────────────────────────────────────────────────

/// periodic Catmull–Rom interpolation of a cyclic waveform, `up` points per sample
fn upsample(w: &[f64], up: usize) -> Vec<f64> {
    let n = w.len();
    let at = |i: i64| w[i.rem_euclid(n as i64) as usize];
    (0..n * up).map(|k| {
        let (i, t) = ((k / up) as i64, (k % up) as f64 / up as f64);
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t)
    }).collect()
}

/// (foot, half-maximum, steepest upstroke) times in seconds of the dominant upstroke of an upsampled cyclic waveform; None for a flat waveform
fn landmarks(x: &[f64], period: f64) -> Option<[f64; 3]> {
    let m = x.len();
    let dtu = period / m as f64;
    let (lo, hi) = (x.iter().cloned().fold(f64::INFINITY, f64::min), x.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    if hi - lo < 1e-12 * (1.0 + hi.abs()) {
        return None;
    }
    let d = |i: usize| (x[(i + 1) % m] - x[(i + m - 1) % m]) / (2.0 * dtu);
    let imax = (0..m).max_by(|&a, &b| d(a).total_cmp(&d(b)))?;
    let slope = d(imax);
    if slope <= 0.0 {
        return None;
    }
    // the derivative of a piecewise-cubic interpolant jumps from knot to knot at coarse sampling: take the centroid of the steepest part (≥ 80 % of the maximum)
    // as the time of the steepest upstroke
    let (mut num, mut den) = (0.0, 0.0);
    for k in -(m as i64) / 4..=(m as i64) / 4 {
        let i = (imax as i64 + k).rem_euclid(m as i64) as usize;
        let v = d(i);
        if v >= 0.8 * slope {
            num += (imax as f64 + k as f64) * dtu * v;
            den += v;
        }
    }
    let t_max = if den > 0.0 { num / den } else { imax as f64 * dtu };
    // crossing times of a level on the rising limb nearest the steepest point (interpolated linearly between the upsampled points)
    let cross = |level: f64| -> Option<f64> {
        let mut best: Option<(f64, f64)> = None;
        for i in 0..m {
            let (a, b) = (x[i], x[(i + 1) % m]);
            if a < level && b >= level {
                let t = (i as f64 + (level - a) / (b - a)) * dtu;
                let dist = ((t - t_max + 1.5 * period).rem_euclid(period) - 0.5 * period).abs();
                if best.map_or(true, |(bd, _)| dist < bd) {
                    best = Some((dist, t));
                }
            }
        }
        // unwrapped around the steepest point, so that the 20 %, 50 % and 80 % times are ordered even when the upstroke straddles the start of the cycle
        best.map(|b| t_max + wrap(b.1 - t_max, period))
    };
    // foot: the line through the 20 % and 80 % points of the upstroke meets the baseline (robust at coarse sampling, where a tangent at one sample is not)
    let (t20, t80) = (cross(lo + 0.2 * (hi - lo))?, cross(lo + 0.8 * (hi - lo))?);
    let foot = if t80 > t20 { t20 - 0.2 * (t80 - t20) / 0.6 } else { t_max - (x[imax] - lo) / slope };
    Some([foot, cross(lo + 0.5 * (hi - lo))?, t_max])
}

fn wrap(t: f64, period: f64) -> f64 {
    (t + 0.5 * period).rem_euclid(period) - 0.5 * period
}

pub struct MethodResult {
    pub name: &'static str,
    pub pwv: f64,
    pub ci: [f64; 2],
    pub r2: f64,
    pub slope: f64,
    pub delays_ms: Vec<f64>,
}

pub struct PwvResult {
    pub pwv: f64,
    pub ci: [f64; 2],
    pub r2: f64,
    pub slope: f64,
    pub method: String,
    pub delays_ms: Vec<f64>,
    pub methods: Vec<MethodResult>,
    pub warnings: Vec<String>,
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() { f64::NAN } else if v.len() % 2 == 1 { v[v.len() / 2] } else { 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]) }
}

/// Theil–Sen slope of y on x, with a bootstrap interval over the points (deterministic resampling) and the OLS r²
fn robust_slope(x: &[f64], y: &[f64]) -> (f64, [f64; 2], f64) {
    let ts = |idx: &[usize]| -> f64 {
        let mut sl = Vec::new();
        for a in 0..idx.len() {
            for b in a + 1..idx.len() {
                let dx = x[idx[b]] - x[idx[a]];
                if dx.abs() > 1e-12 {
                    sl.push((y[idx[b]] - y[idx[a]]) / dx);
                }
            }
        }
        median(&mut sl)
    };
    let all: Vec<usize> = (0..x.len()).collect();
    let slope = ts(&all);
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut boot = Vec::new();
    for _ in 0..200 {
        let idx: Vec<usize> = (0..x.len()).map(|_| { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((state >> 33) as usize) % x.len() }).collect();
        let s = ts(&idx);
        if s.is_finite() {
            boot.push(s);
        }
    }
    boot.sort_by(|a, b| a.total_cmp(b));
    let q = |f: f64| if boot.is_empty() { f64::NAN } else { boot[((boot.len() - 1) as f64 * f) as usize] };
    let (xm, ym) = (x.iter().sum::<f64>() / x.len() as f64, y.iter().sum::<f64>() / x.len() as f64);
    let (sxx, sxy, syy) = x.iter().zip(y).fold((0.0, 0.0, 0.0), |(a, b, c), (&u, &v)| (a + (u - xm).powi(2), b + (u - xm) * (v - ym), c + (v - ym).powi(2)));
    // r² of the robust line (so an outlier does not hide behind a good OLS fit): 1 − residual² / total²
    let icpt = { let mut r: Vec<f64> = x.iter().zip(y).map(|(&u, &v)| v - slope * u).collect(); median(&mut r) };
    let ssr: f64 = x.iter().zip(y).map(|(&u, &v)| (v - slope * u - icpt).powi(2)).sum();
    let r2 = if syy > 0.0 { (1.0 - ssr / syy).max(0.0) } else { 0.0 };
    let _ = (sxx, sxy);
    (slope, [q(0.05), q(0.95)], r2)
}

fn to_pwv(slope: f64, ci_slope: [f64; 2]) -> (f64, [f64; 2]) {
    // slope in s per mm → m/s
    let v = |s: f64| if s.abs() > 1e-12 { 1e-3 / s.abs() } else { f64::INFINITY };
    let (a, b) = (v(ci_slope[0]), v(ci_slope[1]));
    (v(slope), [a.min(b), a.max(b)])
}


/// Forward / backward decomposition of the flow waveforms at several cross-sections: per harmonic k, X_k(s_i) = F_k e^{−iθ} + B_k e^{+iθ}, θ = dir·2πk·s_i/(T·c)
/// (a wave travelling towards increasing s is the "forward" one when dir = +1), solved by complex least squares over the sections. `s_m` in metres, `c` m/s.
/// None (with a warning) when the cross-sections span too little of the wavelength for the two waves to be told apart.
pub fn separate_waves(waves: &[Vec<f64>], s_m: &[f64], c: f64, dir: f64, period: f64, warnings: &mut Vec<String>) -> Option<serde_json::Value> {
    let (n, nf) = (waves.len(), waves[0].len());
    let kmax = 6.min(nf / 2 - 1).max(1);
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut harmonics = Vec::new();
    let (mut fwd_t, mut bwd_t) = (vec![0.0; nf], vec![0.0; nf]);
    let mean0 = waves[0].iter().sum::<f64>() / nf as f64;
    let (mut sf, mut sb, mut resid, mut total) = (0.0, 0.0, 0.0, 0.0);
    let mut worst_cond: f64 = 1.0;
    for k in 1..=kmax {
        let w = two_pi * k as f64 / period;
        // DFT of each section at harmonic k
        let x: Vec<(f64, f64)> = waves.iter().map(|q| {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, &v) in q.iter().enumerate() {
                let a = two_pi * (k * i) as f64 / nf as f64;
                re += v * a.cos();
                im -= v * a.sin();
            }
            (2.0 * re / nf as f64, 2.0 * im / nf as f64)
        }).collect();
        let th: Vec<f64> = s_m.iter().map(|&s| dir * w * (s - s_m[0]) / c).collect();
        // normal equations of [e^{−iθ} e^{+iθ}] [F B]ᵀ = X
        let cplx = |a: f64| (a.cos(), a.sin());
        let (mut a12, mut b1, mut b2) = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0));
        for i in 0..n {
            let e = cplx(th[i]);                              // e^{+iθ}
            a12 = (a12.0 + (e.0 * e.0 - e.1 * e.1), a12.1 + 2.0 * e.0 * e.1);      // Σ φ1* φ2 = Σ e^{+iθ} e^{+iθ}
            let (xr, xi) = x[i];
            b1 = (b1.0 + e.0 * xr - e.1 * xi, b1.1 + e.0 * xi + e.1 * xr);            // Σ φ1* X = Σ e^{+iθ} X
            b2 = (b2.0 + e.0 * xr + e.1 * xi, b2.1 + e.0 * xi - e.1 * xr);            // Σ φ2* X = Σ e^{−iθ} X
        }
        // matrix [[n, a12], [conj(a12), n]]; det = n² − |a12|²
        let det = (n * n) as f64 - (a12.0 * a12.0 + a12.1 * a12.1);
        let cond = det / (n * n) as f64;
        worst_cond = worst_cond.min(cond);
        if cond < 0.02 {
            continue;
        }
        let nn = n as f64;
        let f = ((nn * b1.0 - (a12.0 * b2.0 - a12.1 * b2.1)) / det, (nn * b1.1 - (a12.0 * b2.1 + a12.1 * b2.0)) / det);
        let b = ((nn * b2.0 - (a12.0 * b1.0 + a12.1 * b1.1)) / det, (nn * b2.1 - (a12.0 * b1.1 - a12.1 * b1.0)) / det);
        let (fa, ba) = (f.0.hypot(f.1), b.0.hypot(b.1));
        sf += fa * fa;
        sb += ba * ba;
        for i in 0..n {
            let e = cplx(th[i]);
            let m = (f.0 * e.0 + f.1 * e.1 + b.0 * e.0 - b.1 * e.1, -f.0 * e.1 + f.1 * e.0 + b.0 * e.1 + b.1 * e.0);       // F e^{−iθ} + B e^{+iθ}
            resid += (x[i].0 - m.0).powi(2) + (x[i].1 - m.1).powi(2);
            total += x[i].0 * x[i].0 + x[i].1 * x[i].1;
        }
        for t in 0..nf {
            let a = two_pi * (k * t) as f64 / nf as f64;
            let (c_, s_) = (a.cos(), a.sin());
            fwd_t[t] += f.0 * c_ - f.1 * s_;
            bwd_t[t] += b.0 * c_ - b.1 * s_;
        }
        harmonics.push(json!({"k": k, "freq_hz": k as f64 / period, "forward_amp": fa, "backward_amp": ba, "ratio": if fa > 1e-12 { ba / fa } else { f64::NAN }}));
    }
    if harmonics.is_empty() {
        warnings.push(format!("wave separation: the cross-sections span too little of a wavelength (conditioning {worst_cond:.3}); use a longer segment, more distant sections, or higher harmonics"));
        return None;
    }
    if worst_cond < 0.1 {
        warnings.push(format!("wave separation: poorly conditioned for the lowest harmonics ({worst_cond:.3}): only the better conditioned ones are used"));
    }
    let (rf, rb) = (sf.sqrt(), sb.sqrt());
    Some(json!({"harmonics": harmonics, "reflection_index": rb / (rf + rb).max(1e-12), "backward_to_forward_rms": if rf > 1e-12 { rb / rf } else { f64::NAN },
        "forward": fwd_t.iter().map(|x| x + mean0).collect::<Vec<_>>(), "backward": bwd_t, "fit_residual": (resid / total.max(1e-30)).sqrt(), "c_m_s": c,
        "note": "forward = the wave of the main propagation direction; waveforms are at the first cross-section; steady flow (k = 0) is not separable; this is flow-only wave separation (no pressure is available from 4D flow)"}))
}

/// waveforms (one per cross-section, cyclic, `dt` s per sample) at arc lengths `s_mm` → pulse wave velocity by several delay estimators
pub fn pwv_from_waveforms(waves: &[Vec<f64>], s_mm: &[f64], dt: f64, want: &str) -> Result<PwvResult, Error> {
    let nf = waves[0].len();
    let period = nf as f64 * dt;
    let up = 16usize;
    let ups: Vec<Vec<f64>> = waves.iter().map(|w| upsample(w, up)).collect();
    let mut warnings = Vec::new();
    let mut methods = Vec::new();
    let mut add = |name: &'static str, delays: Option<Vec<f64>>, methods: &mut Vec<MethodResult>, warnings: &mut Vec<String>| {
        let Some(d) = delays else { warnings.push(format!("{name}: no usable upstroke in some waveform")); return };
        let (slope, ci_s, r2) = robust_slope(s_mm, &d);
        let (pwv, ci) = to_pwv(slope, ci_s);
        methods.push(MethodResult { name, pwv, ci, r2, slope, delays_ms: d.iter().map(|x| x * 1e3).collect() });
    };
    // landmark methods
    let lms: Vec<Option<[f64; 3]>> = ups.iter().map(|u| landmarks(u, period)).collect();
    for (k, name) in ["foot", "halfmax", "maxslope"].into_iter().enumerate() {
        let d = if lms.iter().all(|l| l.is_some()) {
            let t0 = lms[0].unwrap()[k];
            Some(lms.iter().map(|l| wrap(l.unwrap()[k] - t0, period)).collect::<Vec<f64>>())
        } else { None };
        add(name, d, &mut methods, &mut warnings);
    }
    // circular cross-correlation of the mean-removed waveforms against the first
    let m = nf * up;
    let centred: Vec<Vec<f64>> = ups.iter().map(|u| { let mu = u.iter().sum::<f64>() / m as f64; u.iter().map(|x| x - mu).collect() }).collect();
    let xc: Vec<f64> = centred.iter().map(|w| {
        let (mut best, mut lag) = (f64::NEG_INFINITY, 0i64);
        for sft in -(m as i64 / 2)..(m as i64 / 2) {
            let c: f64 = (0..m).map(|k| centred[0][k] * w[((k as i64 + sft).rem_euclid(m as i64)) as usize]).sum();
            if c > best { best = c; lag = sft; }
        }
        lag as f64 / up as f64 * dt         // w(t) ≈ w0(t − lag)
    }).collect();
    add("xcorr", Some(xc), &mut methods, &mut warnings);
    // phase of the first two harmonics, amplitude weighted (group delay)
    let dft = |w: &Vec<f64>, k: usize| -> (f64, f64) {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &x) in w.iter().enumerate() {
            let a = 2.0 * std::f64::consts::PI * (k * i) as f64 / nf as f64;
            re += x * a.cos();
            im -= x * a.sin();
        }
        (re, im)
    };
    let kmax = 2.min(nf / 2 - 1).max(1);
    let ph: Vec<f64> = waves.iter().map(|w| {
        let r0 = waves[0].clone();
        let (mut num, mut den) = (0.0, 0.0);
        for k in 1..=kmax {
            let (a, b) = (dft(w, k), dft(&r0, k));
            let (re, im) = (a.0 * b.0 + a.1 * b.1, a.1 * b.0 - a.0 * b.1);      // X_k · conj(X0_k)
            let amp = (a.0.hypot(a.1) * b.0.hypot(b.1)).sqrt();
            let dphi = im.atan2(re);
            // x(t − τ) ↔ X e^{−iωτ}: φ = −2π k τ / T
            num += amp * (-dphi * period / (2.0 * std::f64::consts::PI * k as f64));
            den += amp;
        }
        if den > 0.0 { num / den } else { 0.0 }
    }).collect();
    add("phase", Some(ph), &mut methods, &mut warnings);
    methods.retain(|m| m.pwv.is_finite());
    if methods.is_empty() {
        return inv("centerline_pwv: no delay could be estimated (flat waveforms or cross-sections at the same place)");
    }
    let chosen: &MethodResult = if want == "consensus" {
        let good: Vec<&MethodResult> = methods.iter().filter(|m| m.r2 > 0.8).collect();
        if good.is_empty() {
            warnings.push("no method fits the delays well (r² ≤ 0.8): the PWV is unreliable".into());
            &methods[0]
        } else { good[0] }
    } else {
        methods.iter().find(|m| m.name == want).ok_or_else(|| Error::Invalid(format!("centerline_pwv: unknown or unusable method {want:?}")))?
    };
    let (pwv, ci, r2, slope, delays) = if want == "consensus" {
        let good: Vec<&MethodResult> = methods.iter().filter(|m| m.r2 > 0.8).collect();
        if good.is_empty() { (chosen.pwv, chosen.ci, chosen.r2, chosen.slope, chosen.delays_ms.clone()) } else {
            let mut v: Vec<f64> = good.iter().map(|m| m.pwv).collect();
            let mut lo: Vec<f64> = good.iter().map(|m| m.ci[0]).collect();
            let mut hi: Vec<f64> = good.iter().map(|m| m.ci[1]).collect();
            let mut r: Vec<f64> = good.iter().map(|m| m.r2).collect();
            let sl = good.iter().min_by(|a, b| (a.pwv - median(&mut v.clone())).abs().total_cmp(&(b.pwv - median(&mut v.clone())).abs())).unwrap();
            (median(&mut v), [median(&mut lo), median(&mut hi)], median(&mut r), sl.slope, sl.delays_ms.clone())
        }
    } else { (chosen.pwv, chosen.ci, chosen.r2, chosen.slope, chosen.delays_ms.clone()) };
    for m in &methods {
        if (m.pwv - pwv).abs() > 0.3 * pwv {
            warnings.push(format!("{} gives {:.1} m/s, {:.0} % from the consensus (the median ignores it)", m.name, m.pwv, 100.0 * (m.pwv - pwv).abs() / pwv));
        }
    }
    let total_delay = delays.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - delays.iter().cloned().fold(f64::INFINITY, f64::min);
    if total_delay < 0.25 * dt * 1e3 {
        warnings.push(format!("the whole delay ({total_delay:.1} ms) is below a quarter of the frame interval ({:.0} ms): use a longer segment or more cross-sections", dt * 1e3));
    }
    Ok(PwvResult { pwv, ci, r2, slope, method: want.to_string(), delays_ms: delays, methods, warnings })
}

pub fn register(r: &mut Registry) {
    r.register(
        "mesh_wss",
        "wall shear stress on a vessel mesh from a flow value: inputs [mesh, flow]; options {mu (3.5e-3 Pa·s), depth_mm (distance of the first sample inside the wall; default 1 voxel), order (default 3; 2: two depths and a quadratic profile through the surface, 1: one depth, linear, 3: three depths, the wall is found where the fitted velocity vanishes), affine (4×4 voxel→mm of the flow grid, needed when the mesh is in RAS space)}; the tangential velocity at depths d and 2d along the inward normal gives τ = μ (4 v₁ − v₂) / (2 d) (order 2) or μ v₁ / d (order 1); output the mesh with per-vertex scalars tawss (time-averaged magnitude, Pa), peak_wss (Pa), osi (oscillatory shear index, 0 … 0.5) and wss_t<k> is not stored; message {mean_tawss_pa, mean_peak_pa, mean_osi, valid_vertices}",
        |_, o, inp| {
            let mut m = mesh_in("mesh_wss", inp, 0)?;
            let flow = flow_in(inp.get(1), "mesh_wss")?;
            let affine = affine_opt(o)?.or(m.affine);
            let (vox, dirmap) = to_voxel(&m.space, &flow, affine, "mesh_wss")?;
            let mu = o["mu"].as_f64().unwrap_or(3.5e-3);
            let h_m = flow.spacing.iter().cloned().fold(f64::INFINITY, f64::min);        // metres
            let d = o["depth_mm"].as_f64().map(|x| x * 1e-3).unwrap_or(h_m);
            let order = o["order"].as_u64().unwrap_or(3).clamp(1, 3);
            let order2 = order == 2;
            let normals = m.normals();
            let nv = m.verts.len();
            let (mut tawss, mut peak, mut osi) = (vec![f32::NAN; nv], vec![f32::NAN; nv], vec![f32::NAN; nv]);
            let mut valid = 0usize;
            for v in 0..nv {
                let p_mm = [m.verts[v][0] as f64, m.verts[v][1] as f64, m.verts[v][2] as f64];
                let n_mm = [normals[v][0] as f64, normals[v][1] as f64, normals[v][2] as f64];
                let n_g = dirmap(n_mm);      // the same direction along the flow grid's axes
                // sample points inside the wall: p − k d n, with distances in metres (positions in mm)
                let at = |k: f64| vox([p_mm[0] - k * d * 1e3 * n_mm[0], p_mm[1] - k * d * 1e3 * n_mm[1], p_mm[2] - k * d * 1e3 * n_mm[2]]);
                let (p1, p2, p3) = (at(1.0), at(2.0), at(3.0));
                let mut taus: Vec<V3> = Vec::with_capacity(flow.nf);
                let mut ok = true;
                for f in 0..flow.nf {
                    let Some(v1) = flow.sample(p1, f as f64) else { ok = false; break };
                    let tang = |u: V3| { let un = dot(u, n_g); [u[0] - un * n_g[0], u[1] - un * n_g[1], u[2] - un * n_g[2]] };
                    let (t1, t2, t3) = (tang(v1), flow.sample(p2, f as f64).map(tang), flow.sample(p3, f as f64).map(tang));
                    let a = match (order, t2, t3) {
                        (3, Some(t2), Some(t3)) => {
                            // v(x) = c0 + a x + b x² through the samples at x = d, 2d, 3d along the dominant tangential direction; the wall is where v = 0
                            // (the nearest root to the mesh surface), so a wall that the mesh misses by a fraction of a voxel is found, not assumed
                            let l1 = dot(t1, t1).sqrt();
                            let e = if l1 > 1e-12 { [t1[0] / l1, t1[1] / l1, t1[2] / l1] } else { [0.0; 3] };
                            let (s1, s2, s3) = (l1, dot(t2, e), dot(t3, e));
                            let b = (s3 - 2.0 * s2 + s1) / (2.0 * d * d);
                            let a1 = (s2 - s1) / d - b * 3.0 * d;
                            let c0 = s1 - a1 * d - b * d * d;
                            let disc = a1 * a1 - 4.0 * b * c0;
                            let x0 = if b.abs() < 1e-12 { if a1.abs() > 1e-12 { -c0 / a1 } else { 0.0 } } else if disc >= 0.0 {
                                let (r1, r2) = ((-a1 + disc.sqrt()) / (2.0 * b), (-a1 - disc.sqrt()) / (2.0 * b));
                                if r1.abs() < r2.abs() { r1 } else { r2 }
                            } else { 0.0 };
                            let x0 = x0.clamp(-1.5 * d, 1.5 * d);
                            let g = a1 + 2.0 * b * x0;
                            [e[0] * g, e[1] * g, e[2] * g]
                        }
                        (2 | 3, Some(t2), _) => [0, 1, 2].map(|k| (4.0 * t1[k] - t2[k]) / (2.0 * d)),     // order 2, or order 3 where the third sample falls outside the lumen
                        _ => [0, 1, 2].map(|k| t1[k] / d),
                    };
                    taus.push([mu * a[0], mu * a[1], mu * a[2]]);
                }
                if !ok || taus.is_empty() {
                    continue;
                }
                let mags: Vec<f64> = taus.iter().map(|t| dot(*t, *t).sqrt()).collect();
                let mean_mag = mags.iter().sum::<f64>() / mags.len() as f64;
                let mean_vec = taus.iter().fold([0.0; 3], |s, t| [s[0] + t[0], s[1] + t[1], s[2] + t[2]]).map(|x| x / taus.len() as f64);
                tawss[v] = mean_mag as f32;
                peak[v] = mags.iter().cloned().fold(0.0, f64::max) as f32;
                osi[v] = if mean_mag > 1e-12 { (0.5 * (1.0 - dot(mean_vec, mean_vec).sqrt() / mean_mag)) as f32 } else { 0.0 };
                valid += 1;
            }
            let mean = |x: &Vec<f32>| { let f: Vec<f64> = x.iter().filter(|v| v.is_finite()).map(|&v| v as f64).collect(); f.iter().sum::<f64>() / f.len().max(1) as f64 };
            let stats = json!({"mean_tawss_pa": mean(&tawss), "mean_peak_pa": mean(&peak), "mean_osi": mean(&osi), "valid_vertices": valid, "vertices": nv, "depth_mm": d * 1e3, "mu": mu});
            for (name, vals) in [("tawss", tawss), ("peak_wss", peak), ("osi", osi)] {
                m.scalars.retain(|(k, _)| k != name);
                m.scalars.push((name.to_string(), vals));
            }
            Ok((stats.to_string(), vec![m.to_value()]))
        },
    );
    r.register(
        "centerline_pwv",
        "pulse wave velocity along a vessel from the flow waveforms at cross-sections of a centerline: inputs [lines (centerline_from_mask), flow]; options {branch (default: the longest), all (false: every branch long enough), points (8 cross-sections), margin (0.1: fraction of the branch skipped at each end), signal: flow | velocity (waveform: through-plane flow in ml/s, or its mean velocity), radius_factor (1.6 × lumen radius, at least 3 mm: size of the sampling disc), quantile (restrict the disc to the lumen: PC-MRA region above this quantile), affine (flow grid, needed for RAS lines), method: consensus (default: median of the methods that fit well) | foot | halfmax | maxslope | xcorr | phase}; every waveform is interpolated periodically with cubic splines (×16); methods: foot = tangent at the steepest upstroke meets the baseline, halfmax = time to 50 % of the upstroke, maxslope = time of the steepest upstroke, xcorr = lag of the maximum circular cross-correlation, phase = amplitude-weighted phase of the first two harmonics; PWV = 1 / slope of delay against arc length (Theil–Sen, 5–95 % bootstrap interval over the cross-sections, OLS r²); separate (true): forward / backward decomposition of the flow waveforms (needs cross-sections that span a good part of a wavelength; option pwv_m_s fixes the wave speed instead of the fitted one; message `separation` {harmonics: [{k, forward_amp, backward_amp, ratio}], reflection_index = rms backward / (rms forward + rms backward), forward, backward, fit_residual}); output the lines with pwv_waveforms [P,F], pwv_arclength [P] (mm), pwv_delay_ms [P] added; message {pwv_m_s, ci_m_s, r2, method, methods: {name: {pwv_m_s, ci_m_s, r2, delay_ms}}, branch, arclength_mm, flow, dt_s, direction, warnings, branches: [..] when all}",
        |_, o, inp| {
            let lv = inp.first().ok_or_else(|| Error::Invalid("centerline_pwv: input 0 must be a lines value".into()))?;
            let flow = flow_in(inp.get(1), "centerline_pwv")?;
            let tf = |k: &str| -> Option<Vec<f64>> { lv.get(k).and_then(|t| t.as_tensor()).map(|t| (0..t.numel()).map(|i| t.get_f64(i).0).collect()) };
            let (Some(pts), Some(off), Some(rad), Some(arc)) = (tf("points"), tf("offsets"), tf("radius"), tf("arclength")) else { return inv("centerline_pwv: the lines need points, offsets, radius and arclength (use centerline_from_mask)") };
            let space = match lv.get("space") { Some(Value::Str(s)) => s.clone(), _ => "voxel_mm".into() };
            let own_aff: Option<[f64; 16]> = lv.get("affine").and_then(|t| t.as_tensor()).filter(|t| t.numel() == 16).map(|t| std::array::from_fn(|i| t.get_f64(i).0));
            let (vox, _) = to_voxel(&space, &flow, affine_opt(o)?.or(own_aff), "centerline_pwv")?;
            let p = |i: usize| [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]];
            let nb = off.len() - 1;
            let np = o["points"].as_u64().unwrap_or(8).clamp(3, 64) as usize;
            let margin = o["margin"].as_f64().unwrap_or(0.1).clamp(0.0, 0.4);
            let signal_velocity = o["signal"].as_str() == Some("velocity");
            let want = o["method"].as_str().unwrap_or("consensus").to_string();
            let long_enough = |b: usize| { let n = (off[b + 1] - off[b]) as usize; n as f64 * (1.0 - 2.0 * margin) > np as f64 };
            let branches: Vec<usize> = if o["all"].as_bool().unwrap_or(false) {
                (0..nb).filter(|&b| long_enough(b)).collect()
            } else {
                vec![o["branch"].as_u64().map(|b| b as usize).unwrap_or_else(|| (0..nb).max_by_key(|&b| (off[b + 1] - off[b]) as usize).unwrap_or(0))]
            };
            if branches.is_empty() || branches.iter().any(|&b| b >= nb) {
                return inv("centerline_pwv: no such branch, or no branch long enough for that many cross-sections");
            }
            let mut results = Vec::new();
            let mut first: Option<(Vec<usize>, Vec<Vec<f64>>, Vec<f64>)> = None;
            for &branch in &branches {
                let (b0, b1) = (off[branch] as usize, off[branch + 1] as usize);
                let (i0, i1) = (b0 + ((b1 - b0) as f64 * margin) as usize, b1 - 1 - ((b1 - b0) as f64 * margin) as usize);
                if i1 <= i0 + np {
                    return inv(format!("centerline_pwv: branch {branch} is too short for {np} cross-sections"));
                }
                let idx: Vec<usize> = (0..np).map(|k| i0 + (k * (i1 - i0)) / (np - 1)).collect();
                let mut waves: Vec<Vec<f64>> = Vec::new();
                for &i in &idx {
                    let (j0, j1) = (i.saturating_sub(3).max(b0), (i + 3).min(b1 - 1));
                    let c = vox(p(i));
                    // plane normal: the chord of the line in voxel space scaled by the voxel size (a physical direction along the grid axes, also for RAS lines)
                    let (cv1, cv0) = (vox(p(j1)), vox(p(j0)));
                    let nrm = [(cv1[0] - cv0[0]) * flow.spacing[0], (cv1[1] - cv0[1]) * flow.spacing[1], (cv1[2] - cv0[2]) * flow.spacing[2]];
                    let rad_m = (o["radius_factor"].as_f64().unwrap_or(1.6) * rad[i] * 1e-3).max(3e-3);
                    let (j, _, _) = flow.plane(c, nrm, rad_m, 32, o["quantile"].as_f64())?;
                    let key = if signal_velocity { "mean_velocity_m_s" } else { "flow_ml_s" };
                    let q: Vec<f64> = j[key].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()).unwrap_or_default();
                    waves.push(if q.iter().sum::<f64>() < 0.0 { q.iter().map(|x| -x).collect() } else { q });
                }
                let s_mm: Vec<f64> = idx.iter().map(|&i| arc[i]).collect();
                let res = pwv_from_waveforms(&waves, &s_mm, flow.dt, &want)?;
                if first.is_none() {
                    first = Some((idx.clone(), waves.clone(), res.delays_ms.clone()));
                }
                results.push((branch, s_mm, res));
            }
            let (branch, s_mm, res) = &results[0];
            let methods: serde_json::Map<String, serde_json::Value> = res.methods.iter().map(|m| (m.name.to_string(), json!({"pwv_m_s": m.pwv, "ci_m_s": m.ci, "r2": m.r2, "delay_ms": m.delays_ms}))).collect();
            let mut msg = json!({"pwv_m_s": res.pwv, "ci_m_s": res.ci, "r2": res.r2, "method": res.method, "methods": methods, "branch": branch, "arclength_mm": s_mm,
                "flow": if signal_velocity { "mean velocity" } else { "flow rate" }, "dt_s": flow.dt, "waveforms": first.as_ref().map(|f| f.1.clone()), "warnings": res.warnings,
                "direction": if res.slope >= 0.0 { "the wave travels towards increasing arc length (away from the root)" } else { "the wave travels towards the root" }});
            if o["separate"].as_bool().unwrap_or(true) {
                if let Some((_, w, _)) = &first {
                    let c = o["pwv_m_s"].as_f64().unwrap_or(res.pwv);
                    let mut warn = Vec::new();
                    let s_m: Vec<f64> = s_mm.iter().map(|x| x * 1e-3).collect();
                    if let Some(sep) = separate_waves(w, &s_m, c, if res.slope >= 0.0 { 1.0 } else { -1.0 }, flow.dt * flow.nf as f64, &mut warn) {
                        msg["separation"] = sep;
                    }
                    if let Some(a) = msg["warnings"].as_array_mut() {
                        a.extend(warn.into_iter().map(serde_json::Value::String));
                    }
                }
            }
            if results.len() > 1 {
                msg["branches"] = json!(results.iter().map(|(b, s, r)| json!({"branch": b, "pwv_m_s": r.pwv, "ci_m_s": r.ci, "r2": r.r2, "length_mm": s.last().unwrap() - s[0], "warnings": r.warnings})).collect::<Vec<_>>());
            }
            // the waveforms and delays travel with the lines, for the viewer's waveform plot
            let mut d = match &**lv { Value::Dict(d) => d.clone(), _ => return inv("centerline_pwv: input 0 is not a dict") };
            d.retain(|(k, _)| !k.starts_with("pwv_"));
            if let Some((_, w, dl)) = &first {
                let flat: Vec<f32> = w.iter().flatten().map(|&x| x as f32).collect();
                d.push(("pwv_waveforms".into(), Value::Tensor(hasty_value::Tensor::from_slice(&[w.len(), flow.nf], &flat).unwrap())));
                d.push(("pwv_arclength".into(), Value::Tensor(hasty_value::Tensor::from_slice(&[s_mm.len()], &s_mm.iter().map(|&x| x as f32).collect::<Vec<_>>()).unwrap())));
                d.push(("pwv_delay_ms".into(), Value::Tensor(hasty_value::Tensor::from_slice(&[dl.len()], &dl.iter().map(|&x| x as f32).collect::<Vec<_>>()).unwrap())));
                d.push(("pwv_dt_s".into(), Value::Float(flow.dt)));
                if let Some(sep) = msg.get("separation").filter(|x| x.is_object()) {
                    for (key, name) in [("forward", "pwv_forward"), ("backward", "pwv_backward")] {
                        let v: Vec<f32> = sep[key].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default();
                        d.push((name.into(), Value::Tensor(hasty_value::Tensor::from_slice(&[v.len()], &v).unwrap())));
                    }
                    d.push(("pwv_reflection_index".into(), Value::Float(sep["reflection_index"].as_f64().unwrap_or(0.0))));
                }
            }
            Ok((msg.to_string(), vec![Value::Dict(d)]))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Ctx;
    use crate::mesh::Mesh;
    use hasty_value::Tensor;
    use serde_json::{json, Value as Json};
    use std::collections::HashMap;

    fn tube_mask(nx: usize, ny: usize, nz: usize, r: f64) -> Vec<u8> {
        (0..nx * ny * nz).map(|q| {
            let (i, j) = ((q % nx) as f64, ((q / nx) % ny) as f64);
            (((i - (nx as f64 - 1.0) / 2.0).powi(2) + (j - (ny as f64 - 1.0) / 2.0).powi(2)).sqrt() <= r) as u8
        }).collect()
    }

    fn flow_value(reg: &Registry, ctx: &Ctx, vel: Vec<f32>, shape: &[usize], dt: f64) -> Arc<Value> {
        let t = Arc::new(Value::Tensor(Tensor::from_slice(shape, &vel).unwrap()));
        let o = format!(r#"{{"spacing":[0.001,0.001,0.001],"dt":{dt}}}"#);
        Arc::new(reg.run("flow_make", ctx, &o, &[t]).unwrap().1.remove(0))
    }

    fn read_npy(path: &std::path::Path) -> (Vec<usize>, Vec<u8>) {
        let b = std::fs::read(path).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        let head = String::from_utf8_lossy(&b[10..10 + hl]).to_string();
        let sh = head.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
        let shape: Vec<usize> = sh.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (shape, b[10 + hl..].to_vec())
    }

    /// Validation against the LBM phantom's own wall shear stress: `python -I rustlib/tools/flow/lbm_crop.py <lbm run dir> <dir>` then
    /// `HASTY_LBM_CROP=<dir> cargo test -p hasty-server --lib --release lbm_wall_shear -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn lbm_wall_shear_against_the_simulation() {
        let Some(dir) = std::env::var_os("HASTY_LBM_CROP").map(std::path::PathBuf::from) else { return };
        let meta: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let (sv, bv) = read_npy(&dir.join("vel.npy"));
        let (sm, bm) = read_npy(&dir.join("mask.npy"));
        let (st, bt) = read_npy(&dir.join("tau.npy"));
        let (nf, n) = (sv[0], sv[2]);
        assert_eq!((sm.len(), st.len()), (3, 3));
        let vel: Vec<f32> = bv.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let tau: Vec<f32> = bt.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let sp = meta["spacing_m"].as_f64().unwrap();
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&sv, &vel).unwrap()));
        let flow = Arc::new(reg.run("flow_make", &ctx, &json!({"spacing": [sp, sp, sp], "dt": meta["dt"]}).to_string(), &[t]).unwrap().1.remove(0));
        let mask = Arc::new(Value::Tensor(Tensor::from_slice(&sm, &bm).unwrap()));
        let (_, mesh) = reg.run("mesh_from_mask", &ctx, &json!({"spacing_mm": [sp * 1e3, sp * 1e3, sp * 1e3], "keep": "all", "blur": 1, "smooth": 3}).to_string(), &[mask]).unwrap();
        eprintln!("{nf} frames, {n}³ cells of {} mm, mesh {} vertices", sp * 1e3, Mesh::from_value(&mesh[0]).unwrap().verts.len());
        for order in [2u64, 3] {
            let (msg, out) = reg.run("mesh_wss", &ctx, &json!({"order": order}).to_string(), &[Arc::new(mesh[0].clone()), flow.clone()]).unwrap();
            let m = Mesh::from_value(&out[0]).unwrap();
            let ta = &m.scalars.iter().find(|(k, _)| k == "tawss").unwrap().1;
            // average my values per LBM wall cell (nearest wall cell within 1.5 cells of the vertex) and compare cell by cell
            let h = sp * 1e3;
            let mut cell: HashMap<usize, (f64, usize)> = HashMap::new();
            for (i, v) in m.verts.iter().enumerate() {
                if !ta[i].is_finite() { continue; }
                let c = [v[0] as f64 / h, v[1] as f64 / h, v[2] as f64 / h];
                let mut best: Option<(f64, usize)> = None;
                for dz in -2i64..=2 { for dy in -2i64..=2 { for dx in -2i64..=2 {
                    let (x, y, z) = (c[0].round() as i64 + dx, c[1].round() as i64 + dy, c[2].round() as i64 + dz);
                    if x < 0 || y < 0 || z < 0 || x >= n as i64 || y >= n as i64 || z >= n as i64 { continue; }
                    let q = (z as usize * n + y as usize) * n + x as usize;
                    if tau[q] > 0.0 {
                        let d = ((x as f64 - c[0]).powi(2) + (y as f64 - c[1]).powi(2) + (z as f64 - c[2]).powi(2)).sqrt();
                        if d <= 1.5 && best.map_or(true, |(bd, _)| d < bd) { best = Some((d, q)); }
                    }
                } } }
                if let Some((_, q)) = best {
                    let e = cell.entry(q).or_insert((0.0, 0));
                    e.0 += ta[i] as f64;
                    e.1 += 1;
                }
            }
            let pairs: Vec<(f64, f64)> = cell.iter().filter(|(_, (_, c))| *c >= 2).map(|(&q, &(s, c))| (s / c as f64, tau[q] as f64)).collect();
            let (mx, my) = (pairs.iter().map(|p| p.0).sum::<f64>() / pairs.len() as f64, pairs.iter().map(|p| p.1).sum::<f64>() / pairs.len() as f64);
            let (sxx, syy, sxy) = pairs.iter().fold((0.0, 0.0, 0.0), |(a, b, c), p| (a + (p.0 - mx).powi(2), b + (p.1 - my).powi(2), c + (p.0 - mx) * (p.1 - my)));
            let mut ratio: Vec<f64> = pairs.iter().filter(|p| p.1 > 1e-6).map(|p| p.0 / p.1).collect();
            ratio.sort_by(|a, b| a.total_cmp(b));
            eprintln!("order {order}: {} wall cells compared; mean mine {mx:.3} Pa, mean LBM {my:.3} Pa, ratio of means {:.2}, median cell ratio {:.2}, Pearson r {:.3}; {msg}",
                pairs.len(), mx / my, ratio[ratio.len() / 2], sxy / (sxx * syy).sqrt());
        }
    }

    #[test]
    fn forward_and_backward_waves_are_separated() {
        // a long tube (260 mm) carrying a forward pulse and a 40 % reflected pulse travelling the other way (three harmonics), c = 2 m/s, period 0.4 s: the wavelength is 0.8 m
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let (period, c, rfl) = (0.4, 2.0, 0.4);
        let harm = |t: f64| { let w = 2.0 * std::f64::consts::PI / period; (w * t).sin() + 0.5 * (2.0 * w * t + 0.7).sin() + 0.25 * (3.0 * w * t + 1.9).sin() };
        // shape(t − z/c) forward, rfl · shape(t + z/c) backward: the second argument needs the position, so build the field by hand
        let (nx, ny, nz, nf) = (24usize, 24usize, 260usize, 40usize);
        let r = 6.0;
        let dt = period / nf as f64;
        let nvx = nx * ny * nz;
        let mut vel = vec![0f32; nf * 3 * nvx];
        for f in 0..nf {
            for q in 0..nvx {
                let (i, j, k) = ((q % nx) as f64 - 11.5, ((q / nx) % ny) as f64 - 11.5, (q / (nx * ny)) as f64);
                let r2 = (i * i + j * j) / (r * r);
                if r2 < 1.0 {
                    let (t, z) = (f as f64 * dt, k * 1e-3);
                    vel[(f * 3 + 2) * nvx + q] = ((1.0 - r2) * (1.5 + harm(t - z / c) + rfl * harm(t + z / c))) as f32;
                }
            }
        }
        let flow = flow_value(&reg, &ctx, vel, &[nf, 3, nz, ny, nx], dt);
        let flow2 = flow.clone();
        let mask = Arc::new(Value::Tensor(Tensor::from_slice(&[nz, ny, nx], &tube_mask(nx, ny, nz, r)).unwrap()));
        let (_, cl) = reg.run("centerline_from_mask", &ctx, r#"{"spacing_mm":[1,1,1],"coordinates":"voxel"}"#, &[mask]).unwrap();
        let (msg, _) = reg.run("centerline_pwv", &ctx, &json!({"points": 10, "pwv_m_s": c, "margin": 0.05}).to_string(), &[Arc::new(cl[0].clone()), flow]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        let sep = &j["separation"];
        assert!(sep.is_object(), "{msg}");
        let hs = sep["harmonics"].as_array().unwrap();
        for h in hs.iter().take(3) {
            let (k, ratio) = (h["k"].as_u64().unwrap(), h["ratio"].as_f64().unwrap());
            eprintln!("harmonic {k}: backward / forward {ratio:.3} (true {rfl})");
            assert!((ratio - rfl).abs() < 0.08, "harmonic {k}: {ratio} vs {rfl}");
        }
        let ri = sep["reflection_index"].as_f64().unwrap();
        assert!((ri - rfl / (1.0 + rfl)).abs() < 0.05, "reflection index {ri} vs {}", rfl / (1.0 + rfl));
        assert!(sep["fit_residual"].as_f64().unwrap() < 0.05, "{}", sep["fit_residual"]);
        // with the wave speed taken from the fit of the delays (which the reflection biases): the separation degrades gracefully
        let (msg2, _) = reg.run("centerline_pwv", &ctx, &json!({"points": 10, "margin": 0.05}).to_string(), &[Arc::new(cl[0].clone()), flow2]).unwrap();
        let j2: Json = serde_json::from_str(&msg2).unwrap();
        eprintln!("without the wave speed given: PWV {:.2} m/s (true {c}), reflection index {:.3} (true {:.3})", j2["pwv_m_s"].as_f64().unwrap(), j2["separation"]["reflection_index"].as_f64().unwrap_or(f64::NAN), rfl / (1.0 + rfl));
        assert!((j2["separation"]["reflection_index"].as_f64().unwrap() - rfl / (1.0 + rfl)).abs() < 0.15);
    }

    #[test]
    fn landmarks_move_with_the_waveform() {
        let (nf, period) = (40usize, 0.2);
        let sine = |t: f64| 1.0 + 0.5 * (2.0 * std::f64::consts::PI * t / period).sin();
        let w0: Vec<f64> = (0..nf).map(|i| sine(i as f64 * period / nf as f64)).collect();
        let l0 = landmarks(&upsample(&w0, 16), period).unwrap();
        for k in 1..8 {
            let shift = k as f64 * 0.0034;
            let w: Vec<f64> = (0..nf).map(|i| sine(i as f64 * period / nf as f64 - shift)).collect();
            let l = landmarks(&upsample(&w, 16), period).unwrap();
            for m in 0..3 {
                assert!((wrap(l[m] - l0[m], period) - shift).abs() < 2e-4, "landmark {m} shift {} vs {shift} (l0 {l0:?} l {l:?})", wrap(l[m] - l0[m], period));
            }
        }
    }


    #[test]
    fn wall_shear_in_the_legs_of_a_bifurcation() {
        // Y tree (64³ voxels of 1 mm): trunk R = 5 along z, arms R = 3.5; every voxel carries the Poiseuille profile of the leg it belongs to (unit peak velocity)
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 64usize;
        let legs: [([f64; 3], [f64; 3], f64); 3] = [([32.0, 32.0, 4.0], [32.0, 32.0, 32.0], 5.0), ([32.0, 32.0, 32.0], [12.0, 20.0, 58.0], 3.5), ([32.0, 32.0, 32.0], [52.0, 44.0, 58.0], 3.5)];
        let nvx = n * n * n;
        let mut vel = vec![0f32; 2 * 3 * nvx];
        let mut mask = vec![0u8; nvx];
        for q in 0..nvx {
            let p = [(q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64];
            let mut best: Option<(f64, [f64; 3])> = None;
            for (a, b, r) in legs {
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / l2).clamp(0.0, 1.0);
                let e = [a[0] + t * d[0] - p[0], a[1] + t * d[1] - p[1], a[2] + t * d[2] - p[2]];
                let rr = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]) / (r * r);
                if rr <= 1.0 && best.map_or(true, |(b0, _)| rr < b0) {
                    let l = l2.sqrt();
                    best = Some((rr, [d[0] / l * (1.0 - rr), d[1] / l * (1.0 - rr), d[2] / l * (1.0 - rr)]));
                }
            }
            if let Some((_, v)) = best {
                mask[q] = 1;
                for f in 0..2 {
                    for c in 0..3 {
                        vel[(f * 3 + c) * nvx + q] = v[c] as f32;
                    }
                }
            }
        }
        let flow = flow_value(&reg, &ctx, vel, &[2, 3, n, n, n], 0.04);
        let m = Arc::new(Value::Tensor(Tensor::from_slice(&[n, n, n], &mask).unwrap()));
        let (_, mesh) = reg.run("mesh_from_mask", &ctx, r#"{"spacing_mm":[1,1,1],"blur":1,"smooth":4}"#, &[m]).unwrap();
        let (_, out) = reg.run("mesh_wss", &ctx, "{}", &[Arc::new(mesh[0].clone()), flow]).unwrap();
        let mm = Mesh::from_value(&out[0]).unwrap();
        let ta = &mm.scalars.iter().find(|(k, _)| k == "tawss").unwrap().1;
        let mu = 3.5e-3;
        for (li, (a, b, r)) in legs.iter().enumerate() {
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let sel: Vec<f64> = (0..mm.verts.len()).filter(|&i| {
                let p = [mm.verts[i][0] as f64, mm.verts[i][1] as f64, mm.verts[i][2] as f64];
                let t = ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / l2;
                let e = [a[0] + t * d[0] - p[0], a[1] + t * d[1] - p[1], a[2] + t * d[2] - p[2]];
                let rad = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
                (0.35..0.75).contains(&t) && (rad - r).abs() < 1.0 && ta[i].is_finite()
            }).map(|i| ta[i] as f64).collect();
            let mean = sel.iter().sum::<f64>() / sel.len().max(1) as f64;
            let want = mu * 2.0 / (r * 1e-3);
            eprintln!("leg {li}: WSS {mean:.3} Pa vs analytic {want:.3} ({} vertices)", sel.len());
            assert!(sel.len() > 100 && (mean - want).abs() / want < 0.2, "leg {li}: {mean} vs {want} ({} vertices)", sel.len());
        }
    }

    #[test]
    fn poiseuille_wall_shear_matches_the_analytic_value() {
        // tube along z, radius R = 8 voxels of 1 mm, v_z = V0 (1 − r²/R²): τ = μ · 2 V0 / R (= 0.875 Pa for V0 = 1 m/s, μ = 3.5e-3)
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let (nx, ny, nz, nf) = (40usize, 40usize, 40usize, 2usize);
        let r = 8.0;
        let nvx = nx * ny * nz;
        let mut vel = vec![0f32; nf * 3 * nvx];
        for f in 0..nf {
            for q in 0..nvx {
                let (i, j) = ((q % nx) as f64 - 19.5, ((q / nx) % ny) as f64 - 19.5);
                let r2 = (i * i + j * j) / (r * r);
                if r2 < 1.0 {
                    vel[(f * 3 + 2) * nvx + q] = (1.0 - r2) as f32;
                }
            }
        }
        let flow = flow_value(&reg, &ctx, vel, &[nf, 3, nz, ny, nx], 0.04);
        let flow3 = flow.clone();
        let mask = Arc::new(Value::Tensor(Tensor::from_slice(&[nz, ny, nx], &tube_mask(nx, ny, nz, r)).unwrap()));
        let (_, mesh) = reg.run("mesh_from_mask", &ctx, r#"{"spacing_mm":[1,1,1],"keep":"all","blur":2,"smooth":4}"#, &[mask]).unwrap();
        let (msg, out) = reg.run("mesh_wss", &ctx, "{}", &[Arc::new(mesh[0].clone()), flow]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        // only the side wall of the tube (the mesh ends are closed caps at z = 0 / 39): look at vertices in the middle third
        let m = Mesh::from_value(&out[0]).unwrap();
        let ta = &m.scalars.iter().find(|(k, _)| k == "tawss").unwrap().1;
        let mid: Vec<f64> = (0..m.verts.len()).filter(|&i| m.verts[i][2] > 12.0 && m.verts[i][2] < 27.0 && ta[i].is_finite()).map(|i| ta[i] as f64).collect();
        let mean = mid.iter().sum::<f64>() / mid.len() as f64;
        assert!(mid.len() > 200, "{} wall vertices", mid.len());
        eprintln!("WSS mean {mean:.3} Pa (analytic 0.875), {} wall vertices", mid.len());
        assert!((mean - 0.875).abs() / 0.875 < 0.2, "mean WSS {mean} Pa vs 0.875 ({j})");
        // three-depth extrapolation to the zero-velocity wall: less sensitive to where the iso-surface sits
        let (_, out3) = reg.run("mesh_wss", &ctx, r#"{"order":3}"#, &[Arc::new(mesh[0].clone()), flow3]).unwrap();
        let m3 = Mesh::from_value(&out3[0]).unwrap();
        let ta3 = &m3.scalars.iter().find(|(k, _)| k == "tawss").unwrap().1;
        let mid3: Vec<f64> = (0..m3.verts.len()).filter(|&i| m3.verts[i][2] > 12.0 && m3.verts[i][2] < 27.0 && ta3[i].is_finite()).map(|i| ta3[i] as f64).collect();
        let mean3 = mid3.iter().sum::<f64>() / mid3.len() as f64;
        eprintln!("WSS order 2: {mean:.3} Pa, order 3: {mean3:.3} Pa (analytic 0.875)");
        assert!((mean3 - 0.875).abs() <= (mean - 0.875).abs() + 0.01, "order 3 should not be worse: {mean3} vs {mean}");
        let osi = &m.scalars.iter().find(|(k, _)| k == "osi").unwrap().1;
        assert!(osi.iter().filter(|x| x.is_finite()).all(|&x| x < 0.05), "steady flow has no oscillation");
    }

    /// tube along z (radius 6 voxels of 1 mm) carrying the waveform `shape(t − z/c)`, sampled `nf` times per period; returns (flow value, mask)
    fn wave_tube(reg: &Registry, ctx: &Ctx, nz: usize, c: f64, nf: usize, period: f64, shape: &dyn Fn(f64) -> f64, noise: f64) -> (Arc<Value>, Arc<Value>) {
        let (nx, ny, r) = (24usize, 24usize, 6.0);
        let dt = period / nf as f64;
        let nvx = nx * ny * nz;
        let mut vel = vec![0f32; nf * 3 * nvx];
        let mut state = 12345u64;
        let mut gauss = || { let mut u = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((state >> 11) as f64 + 1.0) / (1u64 << 53) as f64 }; (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos() };
        for f in 0..nf {
            for q in 0..nvx {
                let (i, j, k) = ((q % nx) as f64 - 11.5, ((q / nx) % ny) as f64 - 11.5, (q / (nx * ny)) as f64);
                let r2 = (i * i + j * j) / (r * r);
                if r2 < 1.0 {
                    vel[(f * 3 + 2) * nvx + q] = ((1.0 - r2) * shape(f as f64 * dt - k * 1e-3 / c) + if noise > 0.0 { noise * gauss() } else { 0.0 }) as f32;
                }
            }
        }
        let flow = flow_value(reg, ctx, vel, &[nf, 3, nz, ny, nx], dt);
        let mask = Arc::new(Value::Tensor(Tensor::from_slice(&[nz, ny, nx], &tube_mask(nx, ny, nz, r)).unwrap()));
        (flow, mask)
    }

    fn pwv_of(reg: &Registry, ctx: &Ctx, flow: Arc<Value>, mask: Arc<Value>, opts: &str) -> Json {
        let (_, cl) = reg.run("centerline_from_mask", ctx, r#"{"spacing_mm":[1,1,1],"coordinates":"voxel"}"#, &[mask]).unwrap();
        let (msg, _) = reg.run("centerline_pwv", ctx, opts, &[Arc::new(cl[0].clone()), flow]).unwrap();
        serde_json::from_str(&msg).unwrap()
    }

    #[test]
    fn pulse_wave_velocity_of_a_travelling_wave() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let (period, c) = (0.2, 3.0);
        let sine = move |t: f64| 1.0 + 0.5 * (2.0 * std::f64::consts::PI * t / period).sin();
        let (flow, mask) = wave_tube(&reg, &ctx, 80, c, 40, period, &sine, 0.0);
        let j = pwv_of(&reg, &ctx, flow, mask, r#"{"points":8}"#);
        let pwv = j["pwv_m_s"].as_f64().unwrap();
        eprintln!("sine wave: consensus {pwv:.2} m/s, ci {:?}, methods {}", j["ci_m_s"], j["methods"].as_object().unwrap().iter().map(|(k, v)| format!("{k} {:.2}", v["pwv_m_s"].as_f64().unwrap())).collect::<Vec<_>>().join(", "));
        eprintln!("foot delays {}\nhalf delays {}\nxcorr delays {}", j["methods"]["foot"]["delay_ms"], j["methods"]["halfmax"]["delay_ms"], j["methods"]["xcorr"]["delay_ms"]);
        assert!((pwv - c).abs() / c < 0.1, "PWV {pwv} m/s vs {c}");
        assert!(j["r2"].as_f64().unwrap() > 0.95);
        // every estimator sees the same wave
        for (name, m) in j["methods"].as_object().unwrap() {
            let v = m["pwv_m_s"].as_f64().unwrap();
            assert!((v - c).abs() / c < 0.15, "{name}: {v} m/s vs {c}");
        }
    }

    #[test]
    fn pulse_wave_velocity_with_realistic_pulse_coarse_frames_and_noise() {
        // a systolic pulse with a dicrotic bump (period 1 s), 40 ms frames, c = 5 m/s: the whole delay along the vessel (≈ 14 ms) is a third of a frame
        let pulse = |t: f64| {
            let g = |t: f64, t0: f64, w: f64| (-(((t - t0 + 1.5).rem_euclid(1.0) - 0.5) / w).powi(2)).exp();
            0.3 + 1.2 * g(t, 0.25, 0.06) + 0.35 * g(t, 0.55, 0.05)
        };
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        for (c, nf, noise, tol) in [(5.0, 25usize, 0.0, 0.2), (4.0, 50, 0.04, 0.2)] {
            let (flow, mask) = wave_tube(&reg, &ctx, 100, c, nf, 1.0, &pulse, noise);
            let j = pwv_of(&reg, &ctx, flow, mask, r#"{"points":10}"#);
            let pwv = j["pwv_m_s"].as_f64().unwrap();
            eprintln!("pulse c = {c}, {nf} frames, noise {noise}: consensus {pwv:.2} m/s ci {:?} r2 {:.3}; {}; warnings {}", j["ci_m_s"], j["r2"].as_f64().unwrap(),
                j["methods"].as_object().unwrap().iter().map(|(k, v)| format!("{k} {:.2}", v["pwv_m_s"].as_f64().unwrap())).collect::<Vec<_>>().join(", "), j["warnings"]);
            assert!((pwv - c).abs() / c < tol, "PWV {pwv} m/s vs {c}: {j}");
            let ci = j["ci_m_s"].as_array().unwrap();
            assert!(ci[0].as_f64().unwrap() < ci[1].as_f64().unwrap() + 1e-9);
        }
    }
}
