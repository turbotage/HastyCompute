//! GIRF-robust velocity-encoding blocks (WP1 step 3, research/plan_girf_b0_qfit.md).
//!
//! What the gradient chain does to an encode block g (ending at T with G = 0) that matters for phase contrast is the
//! k offset it leaves behind during the readout, which differs per encode. For every first-order part of the chain
//! it is one linear functional of g,
//!
//!   E(τ) = ∫ g(s) e^{−(T−s)/τ} ds:
//!
//! a low-pass with time constant τ leaves Δm0(t) = −E(τ) e^{−(t−T)/τ}; an eddy term (a, τ) (`girf::Response`
//! `Parametric`, driven by dG/dt) leaves −a E(τ) e^{−(t−T)/τ}; a B0 eddy term (k, τ) leaves the phase γ k E(τ)
//! e^{−(t−T)/τ}; a delay d leaves ≈ −g(T⁻) d²/(2 dt), the small-τ limit. For τ ≫ the block, E(τ) → −M1_T/τ (M0 = 0),
//! which no shape can change: long eddy currents give the classic per-encode linear background phase ∝ ΔM1 / τ that
//! the reconstruction has to model (played moments, residual-phase polynomial). Short and medium τ (≲ the block
//! length) are a matter of shape, and so is the excitation of mechanical resonances (|Ĝ(f)| near f0, ringing for
//! ≈ 1/(2π ζ f0) ≈ 5 ms into the readout).
//!
//! Design (per axis, per encode, fixed block length n raster segments): the interior raster samples x (G = gmax·x,
//! piecewise linear, G = 0 at both ends) of
//!
//!   min  Σ_τ (a_τ γ̄ E_τ(x))² + Σ_f (gain γ̄)² (|Ĝ_f(x)|² + δf² |dĜ_f/df(x)|²) / n_f + λ ‖D² x‖²
//!   s.t. M0(x) = M0, M1(x) = M1 (exact), |x_i| ≤ 1, |x_{i+1} − x_i| ≤ smax dt / gmax,
//!
//! a convex QP, solved by ADMM (OSQP-type splitting, dense: n ≲ 400) and polished to exact moments. All encodes and
//! axes share one length (every TR has the same timing).

use crate::encode::{scheme_m1, EncodeDesign, EncodeSpec};
use crate::girf::Girf;
use crate::system::System;
use crate::wave::Grad;
use crate::{Vec3, GAMMA_BAR};

#[derive(Clone, Debug)]
pub struct RobustOpts {
    /// penalised sensitivities (τ, a): the term a·γ̄·E(τ) in cycles/m
    pub taus: Vec<(f64, f64)>,
    /// resonance band (f_lo, f_hi, gain): penalise gain·γ̄·|Ĝ(f)| (cycles/m) on `n_band` frequencies in the band
    pub band: Option<(f64, f64, f64)>,
    pub n_band: usize,
    /// resonance-frequency uncertainty δf (Hz) of the model the reconstruction will use: penalise also
    /// gain·γ̄·δf·|dĜ/df| over the band (what is left after modelling a resonance whose f0 / width is off scales
    /// with the spectrum's slope there, not with |Ĝ|)
    pub band_df: f64,
    /// per-axis band weights (f, w), used instead of `band` when non-empty: the term w·γ̄·|Ĝ(f)| (e.g. from a measured
    /// GIRF's resonances, `with_resonances`)
    pub band_table: [Vec<(f64, f64)>; 3],
    /// jerk regularisation, relative to the sensitivity terms
    pub jerk: f64,
    /// fraction of the limits the solver may use (room for the exact-moment polish)
    pub margin: f64,
    /// zero raster segments kept at the block end (a gap before the readout)
    pub gap: usize,
    pub max_iter: usize,
}

impl Default for RobustOpts {
    /// a broad prior until a GIRF is measured: the amplifier low-pass family (τ 4–16 µs, a = 1) and eddy currents
    /// (τ 30 µs – 1 ms, a = 2·10⁻³), resonances 0.5–2 kHz (weight 3·10⁻³ ≈ peak height × ζ of a 10 % / ζ 0.03 peak; 20 Hz grid, finer than a resonance
    /// width 2ζf0 ≈ 50–80 Hz, so the band cannot be met only at the grid frequencies)
    fn default() -> Self {
        let mut taus = vec![(4e-6, 1.0), (8e-6, 1.0), (16e-6, 1.0)];
        for t in [30e-6, 60e-6, 120e-6, 250e-6, 500e-6, 1e-3] {
            taus.push((t, 2e-3));
        }
        Self { taus, band: Some((500.0, 2000.0, 3e-3)), n_band: 76, band_df: 15.0, band_table: [vec![], vec![], vec![]], jerk: 1e-4, margin: 0.995, gap: 0, max_iter: 40_000 }
    }
}

impl RobustOpts {
    /// scale each sensitivity by how much of it survives into the readout's weighted window: W(τ) = Σ w(t) e^{−t/τ} /
    /// Σ w(t) (w over the ADC raster, t from the echo) — a short transient matters only where the readout weight is
    pub fn persist(mut self, w: &[f64], dt: f64) -> Self {
        let sw: f64 = w.iter().sum();
        for (tau, a) in self.taus.iter_mut() {
            let s: f64 = w.iter().enumerate().map(|(i, wi)| wi * (-(i as f64) * dt / *tau).exp()).sum();
            *a *= s / sw;
        }
        self
    }
}

impl RobustOpts {
    /// the band term of `axis` as (f, w) pairs
    pub fn band_list(&self, axis: usize) -> Vec<(f64, f64)> {
        if !self.band_table[axis].is_empty() {
            return self.band_table[axis].clone();
        }
        match self.band {
            Some((lo, hi, gain)) => {
                let nb = self.n_band.max(1);
                (0..nb).map(|j| (lo + (hi - lo) * j as f64 / (nb - 1).max(1) as f64, gain / (nb as f64).sqrt())).collect()
            }
            None => vec![],
        }
    }
    /// band weights from the resonances of `girf` (a measured / known chain): per axis, w(f) = 2 |H(f) − H_base(f)|
    /// Δf / (π f) on a grid `step` Hz over [lo, hi] (H_base = the response without its `Resonant` layers): a damped
    /// peak of height g and half width ζf0 leaves |Δk| ≈ γ̄ · 2gζ · |Ĝ(f0)| = γ̄ |Ĝ(f0)| (2/(π f0)) ∫|ΔH| df
    pub fn with_resonances(mut self, girf: &Girf, lo: f64, hi: f64, step: f64) -> Self {
        fn strip(r: &crate::girf::Response) -> crate::girf::Response {
            match r {
                crate::girf::Response::Resonant { base, .. } => strip(base),
                other => other.clone(),
            }
        }
        for a in 0..3 {
            let base = strip(&girf.axis[a]);
            let n = ((hi - lo) / step).round() as usize + 1;
            self.band_table[a] = (0..n)
                .map(|j| {
                    let f = lo + step * j as f64;
                    (f, 2.0 * (girf.axis[a].at(f) - base.at(f)).norm() * step / (std::f64::consts::PI * f))
                })
                .collect();
        }
        self
    }
}

/// E(τ) of the samples `g` (piecewise linear on raster `dt`, G = 0 outside), at the end of the samples
pub fn sensitivity(g: &[f64], dt: f64, tau: f64) -> f64 {
    let n = g.len() - 1;
    let c = hat_exp(dt, tau);
    g.iter().enumerate().map(|(i, v)| v * c * (-((n - i) as f64) * dt / tau).exp()).sum()
}

/// ∫ hat_dt(u) e^{u/τ} du = (τ²/dt)(2 cosh(dt/τ) − 2)
fn hat_exp(dt: f64, tau: f64) -> f64 {
    let r = dt / tau;
    if r < 1e-3 {
        dt * (1.0 + r * r / 12.0)
    } else {
        tau * tau / dt * (2.0 * r.cosh() - 2.0)
    }
}

/// |Ĝ(f)| of the samples (T·s/m)
pub fn spectrum(g: &[f64], dt: f64, f: f64) -> f64 {
    let sinc = |x: f64| if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
    let h = dt * sinc(f * dt).powi(2);
    let (mut re, mut im) = (0.0, 0.0);
    for (i, v) in g.iter().enumerate() {
        let ph = -2.0 * std::f64::consts::PI * f * i as f64 * dt;
        re += v * ph.cos();
        im += v * ph.sin();
    }
    h * (re * re + im * im).sqrt()
}

/// |dĜ/df| of the samples (T·s²/m), time origin at the end of the samples
pub fn spectrum_slope(g: &[f64], dt: f64, f: f64) -> f64 {
    let sinc = |x: f64| if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
    let h = dt * sinc(f * dt).powi(2);
    let n = g.len() - 1;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, v) in g.iter().enumerate() {
        let t = (i as f64 - n as f64) * dt;
        let ph = -2.0 * std::f64::consts::PI * f * i as f64 * dt;
        // −i2πt e^{iph}
        re += v * 2.0 * std::f64::consts::PI * t * ph.sin();
        im += -v * 2.0 * std::f64::consts::PI * t * ph.cos();
    }
    h * (re * re + im * im).sqrt()
}

/// the QP's sensitivity value Σ (a γ̄ E)² + band terms of `axis`, cycles²/m² (without the jerk term)
pub fn objective(g: &[f64], dt: f64, o: &RobustOpts, axis: usize) -> f64 {
    let mut s: f64 = o.taus.iter().map(|&(tau, a)| (a * GAMMA_BAR * sensitivity(g, dt, tau)).powi(2)).sum();
    for (f, w) in o.band_list(axis) {
        s += (w * GAMMA_BAR * spectrum(g, dt, f)).powi(2);
        if o.band_df > 0.0 {
            s += (w * GAMMA_BAR * o.band_df * spectrum_slope(g, dt, f)).powi(2);
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------------------------
// small dense linear algebra

struct Chol {
    n: usize,
    l: Vec<f64>,
}

impl Chol {
    fn new(a: &[f64], n: usize) -> Option<Self> {
        let mut l = vec![0.0; n * n];
        for j in 0..n {
            let mut d = a[j * n + j];
            for k in 0..j {
                d -= l[j * n + k] * l[j * n + k];
            }
            if d <= 0.0 {
                return None;
            }
            let d = d.sqrt();
            l[j * n + j] = d;
            for i in j + 1..n {
                let mut s = a[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / d;
            }
        }
        Some(Self { n, l })
    }
    fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= self.l[i * n + k] * b[k];
            }
            b[i] = s / self.l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for k in i + 1..n {
                s -= self.l[k * n + i] * b[k];
            }
            b[i] = s / self.l[i * n + i];
        }
    }
}

/// result of one axis design
#[derive(Clone, Debug)]
pub struct AxisDesign {
    /// raster samples 0..=n (G = 0 at both ends), T/m
    pub g: Vec<f64>,
    /// sensitivity objective (cycles²/m²) and the ADMM residuals at exit
    pub objective: f64,
    pub r_prim: f64,
    pub iters: usize,
}

/// GIRF-robust block with `n` raster segments for zeroth moment `m0` and first moment `m1` about the block start,
/// under per-axis limits `gmax`, `smax`. None if infeasible at this length.
pub fn axis_qp(m0: f64, m1: f64, n: usize, gmax: f64, smax: f64, dt: f64, o: &RobustOpts, axis: usize) -> Option<AxisDesign> {
    if m0 == 0.0 && m1 == 0.0 {
        return Some(AxisDesign { g: vec![0.0; n + 1], objective: 0.0, r_prim: 0.0, iters: 0 });
    }
    let gap = o.gap.min(n.saturating_sub(3));
    // free samples 1..=m (m = n − 1 − gap); sample i ↔ variable i − 1
    let m = n - 1 - gap;
    if m < 2 {
        return None;
    }
    let smax_x = o.margin * smax * dt / gmax;
    let gx = o.margin;
    // equality rows (normalised): M0 = gmax dt Σ x, M1 = gmax dt² Σ i x
    let mut a0: Vec<f64> = vec![1.0; m];
    let mut a1: Vec<f64> = (1..=m).map(|i| i as f64).collect();
    let mut b0 = m0 / (gmax * dt);
    let mut b1 = m1 / (gmax * dt * dt);
    let (n0, n1) = (a0.iter().map(|v| v * v).sum::<f64>().sqrt(), a1.iter().map(|v| v * v).sum::<f64>().sqrt());
    a0.iter_mut().for_each(|v| *v /= n0);
    a1.iter_mut().for_each(|v| *v /= n1);
    b0 /= n0;
    b1 /= n1;
    // objective rows (cycles/m per unit x)
    let mut rows: Vec<Vec<f64>> = vec![];
    for &(tau, a) in &o.taus {
        let c = hat_exp(dt, tau);
        rows.push((1..=m).map(|i| a * GAMMA_BAR * gmax * c * (-((n - i) as f64) * dt / tau).exp()).collect());
    }
    for (f, wf) in o.band_list(axis) {
        let sinc = |x: f64| if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
        let h = wf * GAMMA_BAR * gmax * dt * sinc(f * dt).powi(2);
        rows.push((1..=m).map(|i| h * (2.0 * std::f64::consts::PI * f * i as f64 * dt).cos()).collect());
        rows.push((1..=m).map(|i| -h * (2.0 * std::f64::consts::PI * f * i as f64 * dt).sin()).collect());
        if o.band_df > 0.0 {
            // dĜ/df = FT{−i2π (t − T) g}: time origin at the block end (the readout's reference)
            let hd = h * o.band_df * 2.0 * std::f64::consts::PI;
            rows.push((1..=m).map(|i| hd * (i as f64 - n as f64) * dt * (2.0 * std::f64::consts::PI * f * i as f64 * dt).sin()).collect());
            rows.push((1..=m).map(|i| hd * (i as f64 - n as f64) * dt * (2.0 * std::f64::consts::PI * f * i as f64 * dt).cos()).collect());
        }
    }
    let mut p = vec![0.0; m * m];
    for r in &rows {
        for i in 0..m {
            if r[i] == 0.0 {
                continue;
            }
            for j in 0..m {
                p[i * m + j] += 2.0 * r[i] * r[j];
            }
        }
    }
    let tr_e: f64 = (0..m).map(|i| p[i * m + i]).sum::<f64>().max(1e-300);
    // jerk: second differences of the full sample vector (zeros at both ends and in the gap), D2ᵀD2 has the
    // pentadiagonal stencil [1, −4, 6, −4, 1]
    let tr_j = 6.0 * m as f64;
    let wj = o.jerk * tr_e / tr_j;
    for i in 0..m {
        for (d, c) in [(0i64, 6.0), (1, -4.0), (2, 1.0)] {
            for s in [-1i64, 1] {
                if d == 0 && s == 1 {
                    continue;
                }
                let j = i as i64 + s * d;
                if j >= 0 && (j as usize) < m {
                    p[i * m + j as usize] += 2.0 * wj * c;
                }
            }
        }
    }
    // overall scale: mean diagonal 1
    let sc = m as f64 / (0..m).map(|i| p[i * m + i]).sum::<f64>();
    p.iter_mut().for_each(|v| *v *= sc);
    // constraints C x: [a0; a1; I (m); D1 (m + 1)]  with D1 x = differences of [0, x, 0]
    let nc = 2 + m + (m + 1);
    let lo: Vec<f64> = [vec![b0, b1], vec![-gx; m], vec![-smax_x; m + 1]].concat();
    let hi: Vec<f64> = [vec![b0, b1], vec![gx; m], vec![smax_x; m + 1]].concat();
    let cx = |x: &[f64], out: &mut [f64]| {
        out[0] = a0.iter().zip(x).map(|(a, b)| a * b).sum();
        out[1] = a1.iter().zip(x).map(|(a, b)| a * b).sum();
        out[2..2 + m].copy_from_slice(x);
        for i in 0..=m {
            let xi = if i < m { x[i] } else { 0.0 };
            let xp = if i > 0 { x[i - 1] } else { 0.0 };
            out[2 + m + i] = xi - xp;
        }
    };
    let ctv = |v: &[f64], out: &mut [f64]| {
        for i in 0..m {
            out[i] = a0[i] * v[0] + a1[i] * v[1] + v[2 + i] + v[2 + m + i] - v[2 + m + i + 1];
        }
    };
    let sigma = 1e-6;
    let mut rho = 0.1;
    let eq_scale = 1e3;
    let rho_of = |k: usize, rho: f64| if k < 2 { rho * eq_scale } else { rho };
    let factor = |rho: f64| -> Chol {
        let mut k = p.clone();
        for i in 0..m {
            k[i * m + i] += sigma + rho; // identity rows
            // D1ᵀD1: tridiagonal [−1, 2, −1]
            k[i * m + i] += 2.0 * rho;
            if i + 1 < m {
                k[i * m + i + 1] -= rho;
                k[(i + 1) * m + i] -= rho;
            }
            for j in 0..m {
                k[i * m + j] += rho * eq_scale * (a0[i] * a0[j] + a1[i] * a1[j]);
            }
        }
        Chol::new(&k, m).expect("ADMM matrix")
    };
    let mut ch = factor(rho);
    let mut x = vec![0.0; m];
    let mut z = vec![0.0; nc];
    let mut y = vec![0.0; nc];
    let mut cxv = vec![0.0; nc];
    let mut rhs = vec![0.0; m];
    let mut tmp = vec![0.0; nc];
    let mut px = vec![0.0; m];
    let alpha = 1.6;
    let (mut r_prim, mut iters) = (f64::INFINITY, 0);
    for it in 0..o.max_iter {
        iters = it + 1;
        for k in 0..nc {
            tmp[k] = rho_of(k, rho) * z[k] - y[k];
        }
        ctv(&tmp, &mut rhs);
        for i in 0..m {
            rhs[i] += sigma * x[i];
        }
        ch.solve(&mut rhs);
        let xt = rhs.clone();
        cx(&xt, &mut cxv);
        for i in 0..m {
            x[i] = alpha * xt[i] + (1.0 - alpha) * x[i];
        }
        for k in 0..nc {
            let zr = alpha * cxv[k] + (1.0 - alpha) * z[k];
            let zn = (zr + y[k] / rho_of(k, rho)).clamp(lo[k], hi[k]);
            y[k] += rho_of(k, rho) * (zr - zn);
            z[k] = zn;
        }
        if it % 50 == 49 {
            cx(&x, &mut cxv);
            r_prim = (0..nc).map(|k| (cxv[k] - z[k]).abs()).fold(0.0, f64::max);
            for i in 0..m {
                px[i] = (0..m).map(|j| p[i * m + j] * x[j]).sum();
            }
            ctv(&y, &mut rhs);
            let r_dual = (0..m).map(|i| (px[i] + rhs[i]).abs()).fold(0.0, f64::max);
            if r_prim < 1e-9 && r_dual < 1e-7 {
                break;
            }
            if it % 400 == 399 {
                let ratio = (r_prim / cxv.iter().chain(&z).map(|v| v.abs()).fold(1e-12, f64::max)) / (r_dual / px.iter().map(|v| v.abs()).fold(1e-12, f64::max)).max(1e-30);
                let f = ratio.sqrt().clamp(0.2, 5.0);
                if !(0.5..=2.0).contains(&f) {
                    rho = (rho * f).clamp(1e-6, 1e6);
                    ch = factor(rho);
                }
            }
        }
    }
    // polish: exact moments (minimum-norm correction in the equality rows), then the hard limits
    let (s00, s01, s11) = (a0.iter().map(|v| v * v).sum::<f64>(), a0.iter().zip(&a1).map(|(a, b)| a * b).sum::<f64>(), a1.iter().map(|v| v * v).sum::<f64>());
    for _ in 0..2 {
        let e0 = a0.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>() - b0;
        let e1 = a1.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>() - b1;
        let det = s00 * s11 - s01 * s01;
        let (l0, l1) = ((s11 * e0 - s01 * e1) / det, (s00 * e1 - s01 * e0) / det);
        for i in 0..m {
            x[i] -= l0 * a0[i] + l1 * a1[i];
        }
    }
    let mut g = vec![0.0; n + 1];
    for i in 0..m {
        g[i + 1] = gmax * x[i];
    }
    let gm = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let sm = g.windows(2).map(|w| (w[1] - w[0]).abs() / dt).fold(0.0, f64::max);
    if gm > gmax * (1.0 + 1e-9) || sm > smax * (1.0 + 1e-6) {
        return None;
    }
    let obj = objective(&g, dt, o, axis);
    Some(AxisDesign { g, objective: obj, r_prim, iters })
}

/// all encodes of `spec` as GIRF-robust blocks of `n` raster segments (per-axis limits of `sys`); None if any
/// (encode, axis) is infeasible at this length. Identical (M0, M1) targets are solved once.
pub fn design(spec: &EncodeSpec, sys: &System, n: usize, o: &RobustOpts) -> Option<(EncodeDesign, Vec<[f64; 3]>)> {
    let enc = scheme_m1(spec.scheme, spec.venc, spec.balanced);
    let local = |e: &[f64; 3], a: usize| e[a] + spec.m1_offset[a] - spec.m0[a] * (spec.t_start - spec.t_ref);
    let dt = sys.grad_raster;
    // identical (axis class, M0, M1) targets are solved once; axes differ only through per-axis band tables
    let per_axis = o.band_table.iter().any(|t| !t.is_empty());
    let cls = |a: usize| if per_axis { a } else { 0 };
    let mut targets: Vec<(usize, f64, f64)> = vec![];
    for e in &enc {
        for a in 0..3 {
            let t = (cls(a), spec.m0[a], local(e, a));
            if !targets.iter().any(|q| *q == t) {
                targets.push(t);
            }
        }
    }
    let sols: Vec<Option<AxisDesign>> = crate::par_map(&targets, |&(a, m0, m1)| axis_qp(m0, m1, n, sys.gmax, sys.smax, dt, o, a));
    if sols.iter().any(|s| s.is_none()) {
        return None;
    }
    let get = |a: usize, m0: f64, m1: f64| -> &AxisDesign { sols[targets.iter().position(|q| *q == (cls(a), m0, m1)).unwrap()].as_ref().unwrap() };
    let mut out = EncodeDesign { waves: vec![], m0: vec![], m1: vec![], m2: vec![], duration: n as f64 * dt, shapes: vec![] };
    let mut objs = vec![];
    for e in &enc {
        let ax: Vec<&AxisDesign> = (0..3).map(|a| get(a, spec.m0[a], local(e, a))).collect();
        let w = Grad::from_axes(&ax[0].g, &ax[1].g, &ax[2].g, dt, spec.t_start);
        let (m0, m1, m2, _) = w.moments(spec.t_ref).last();
        out.waves.push(w);
        out.m0.push(m0);
        out.m1.push(m1);
        out.m2.push(m2);
        objs.push([ax[0].objective, ax[1].objective, ax[2].objective]);
    }
    Some((out, objs))
}

/// robust blocks `extra` raster segments longer than the two-lobe block of `spec` at the same limits (lengthened
/// until feasible), memoised: the TR builder calls this for every rotation set / derating step it is asked for
pub fn design_like(spec: &EncodeSpec, sys: &System, o: &RobustOpts, extra: i64) -> EncodeDesign {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, EncodeDesign>>> = std::sync::OnceLock::new();
    let key = format!("{spec:?}|{sys:?}|{o:?}|{extra}");
    let cache = CACHE.get_or_init(Default::default);
    if let Some(d) = cache.lock().unwrap().get(&key) {
        return d.clone();
    }
    let n0 = crate::encode::design(spec, sys).waves[0].len() as i64 - 1;
    let mut n = (n0 + extra).max(4) as usize;
    let d = loop {
        if let Some((d, _)) = design(spec, sys, n, o) {
            break d;
        }
        n += 1;
        assert!(n as i64 <= n0 + extra + 1000, "no feasible robust encode near {n0} + {extra} segments");
    };
    cache.lock().unwrap().insert(key, d.clone());
    d
}

/// shortest length (raster segments, ≥ `n_from`) at which `design` is feasible
pub fn min_len(spec: &EncodeSpec, sys: &System, n_from: usize, o: &RobustOpts) -> usize {
    let mut n = n_from;
    while design(spec, sys, n, o).is_none() {
        n += 2;
        assert!(n < n_from + 2000, "no feasible robust encode");
    }
    n
}

// ---------------------------------------------------------------------------------------------------------------
// what an encode block leaves in the readout

/// k error (cycles/m) and B0-term phase (rad) that `girf` leaves after the block `w` (physical axes, ending at the echo
/// with G = 0), at `times` after the block end; with `model`, what is left after modelling the chain by `model` (the
/// reconstruction uses the model's played trajectory)
pub fn played_offsets(girf: &Girf, model: Option<&Girf>, w: &Grad, times: &[f64]) -> (Vec<Vec3>, Vec<f64>) {
    let dt = w.dt;
    let t_end = w.t_end();
    let tmax = times.iter().copied().fold(0.0, f64::max);
    let pad = (tmax / dt).ceil() as usize + 4;
    let mut g = w.g.clone();
    g.extend(std::iter::repeat([0.0; 3]).take(pad));
    let wp = Grad { g, ..w.clone() };
    let (p, b0) = girf.apply(&wp);
    let (refg, refb0): (Grad, Vec<f64>) = match model {
        Some(m) => m.apply(&wp),
        None => {
            let mut g = vec![[0.0; 3]; Girf::PRE];
            g.extend_from_slice(&wp.g);
            g.resize(p.len(), [0.0; 3]);
            (Grad { g, t0: p.t0, dt }, vec![0.0; p.len()])
        }
    };
    let nn = p.len().min(refg.len());
    let diff = Grad { dt, t0: p.t0, g: (0..nn).map(|i| [0, 1, 2].map(|a| p.g[i][a] - refg.g[i][a])).collect() };
    let abs: Vec<f64> = times.iter().map(|t| t_end + t).collect();
    let dk: Vec<Vec3> = diff.moments_at(diff.t0, &abs).into_iter().map(|m| m.0.map(|v| GAMMA_BAR * v)).collect();
    let mut ph = vec![0.0; nn];
    for i in 1..nn {
        let (a, b) = (b0[i - 1] - refb0.get(i - 1).copied().unwrap_or(0.0), b0[i] - refb0.get(i).copied().unwrap_or(0.0));
        ph[i] = ph[i - 1] + crate::GAMMA * 0.5 * (a + b) * dt;
    }
    let phi = abs
        .iter()
        .map(|&t| {
            let f = ((t - p.t0) / dt).clamp(0.0, (nn - 1) as f64 - 1e-9);
            let i = f.floor() as usize;
            ph[i] + (ph[(i + 1).min(nn - 1)] - ph[i]) * (f - i as f64)
        })
        .collect();
    (dk, phi)
}

/// least-squares weight of a readout sample at |k| for fitting a linear phase to a uniform ball of radius `r`:
/// |dρ̂/dk|², ρ̂(u) = 3 (sin u − u cos u)/u³, u = 2π|k|r (normalised to max 1 over `k_abs`)
pub fn ball_weights(k_abs: &[f64], r: f64) -> Vec<f64> {
    let d = |u: f64| -> f64 {
        if u < 1e-2 {
            -u / 5.0 + u * u * u / 70.0
        } else {
            3.0 * ((u * u - 3.0) * u.sin() + 3.0 * u * u.cos()) / u.powi(4)
        }
    };
    let w: Vec<f64> = k_abs.iter().map(|k| d(2.0 * std::f64::consts::PI * k * r).powi(2)).collect();
    let m = w.iter().copied().fold(0.0, f64::max).max(1e-300);
    w.into_iter().map(|v| v / m).collect()
}

/// resonance exposure of a (rotating) readout: sqrt(Σ_f (w_f γ̄ |Ĝ(f)|)²) in cycles/m, Ĝ the vector spectrum of `g`
/// (|R Ĝ| = |Ĝ| for every rotation, so the measure is the same for all readouts of a rotation set). With
/// `RobustOpts::with_resonances` weights this approximates the k error the resonances leave (γ̄ · 2gζ · |Ĝ(f0)| per
/// peak), with the generic band a relative score.
pub fn band_exposure(g: &Grad, band: &[(f64, f64)]) -> f64 {
    let dt = g.dt;
    let sinc = |x: f64| if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
    let mut s = 0.0;
    for &(f, w) in band {
        let h = dt * sinc(f * dt).powi(2);
        let mut acc = 0.0;
        for a in 0..3 {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, v) in g.g.iter().enumerate() {
                let ph = -2.0 * std::f64::consts::PI * f * i as f64 * dt;
                re += v[a] * ph.cos();
                im += v[a] * ph.sin();
            }
            acc += re * re + im * im;
        }
        s += (w * GAMMA_BAR * h).powi(2) * acc;
    }
    s.sqrt()
}

/// band weights averaged over the three physical axes (in quadrature): what a rotating readout sees
pub fn rotation_band(o: &RobustOpts) -> Vec<(f64, f64)> {
    let b: Vec<Vec<(f64, f64)>> = (0..3).map(|a| o.band_list(a)).collect();
    (0..b[0].len()).map(|j| (b[0][j].0, ((b[0][j].1.powi(2) + b[1].get(j).map_or(0.0, |x| x.1).powi(2) + b[2].get(j).map_or(0.0, |x| x.1).powi(2)) / 3.0).sqrt())).collect()
}

/// least-squares weights over the ADC window of a base readout `spoke` (from the echo, k = 0 at t = 0) on a `dtf`
/// grid: (times, body scale (12 cm ball), vessel scale (2 mm ball)), each normalised to max 1
pub fn readout_weights(spoke: &Grad, dtf: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let times: Vec<f64> = (0..(spoke.duration() / dtf) as usize).map(|i| i as f64 * dtf).collect();
    let kabs: Vec<f64> = spoke.moments_at(spoke.t0, &times.iter().map(|t| spoke.t0 + t).collect::<Vec<_>>()).iter().map(|m| GAMMA_BAR * crate::norm(&m.0)).collect();
    let (b, v) = (ball_weights(&kabs, 0.12), ball_weights(&kabs, 0.002));
    (times, b, v)
}

/// `RobustOpts::persist` weights: both scales, each with total weight 1
pub fn persist_weights(body: &[f64], vessel: &[f64]) -> Vec<f64> {
    let (sb, sv) = (body.iter().sum::<f64>(), vessel.iter().sum::<f64>());
    body.iter().zip(vessel).map(|(a, b)| a / sb + b / sv).collect()
}

/// weighted means of the k error and phase over the readout (the per-encode linear + constant background phase a
/// least-squares reconstruction sees)
pub fn weighted_offset(dk: &[Vec3], phi: &[f64], w: &[f64]) -> (Vec3, f64) {
    let sw: f64 = w.iter().sum();
    let mut c = [0.0; 3];
    let mut p = 0.0;
    for i in 0..w.len() {
        for a in 0..3 {
            c[a] += w[i] * dk[i][a] / sw;
        }
        p += w[i] * phi[i] / sw;
    }
    (c, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{delta_m1, two_lobe_min_time};
    use crate::girf::Response;

    #[test]
    fn sensitivity_matches_the_played_k_offset() {
        // a bipolar through an eddy term (a, τ): the k offset just after the block is −a E(τ)
        let sys = System::prisma_xr();
        let s = two_lobe_min_time(0.0, delta_m1(1.0), &System { smax: 60.0, ..sys.clone() });
        let g = s.samples();
        let (a, tau) = (0.003, 1e-3);
        let girf = Girf { axis: [Response::Parametric { delay: 0.0, tau_lp: 0.0, eddy: vec![(a, tau)] }, Response::Ideal, Response::Ideal], tail: 0.0, ..Girf::ideal() };
        let w = Grad::from_axis(0, &g, sys.grad_raster, 0.0);
        let (p, _) = girf.apply(&w);
        let mp = p.moments(p.t0).m0[Girf::PRE + g.len() - 1][0];
        let want = -a * sensitivity(&g, sys.grad_raster, tau);
        assert!((mp - want).abs() < 1e-3 * want.abs(), "{mp:e} vs {want:e}");
    }

    #[test]
    fn robust_block_has_exact_moments_limits_and_lower_sensitivity() {
        let sys = System { smax: 60.0, ..System::prisma_xr() };
        let dt = sys.grad_raster;
        let m1 = delta_m1(1.0);
        let base = two_lobe_min_time(0.0, m1, &sys);
        let n0 = base.segments();
        let o = RobustOpts::default();
        let g0 = base.samples();
        let f0 = objective(&g0, dt, &o, 0);
        let r = axis_qp(0.0, m1, n0 + 20, sys.gmax, sys.smax, dt, &o, 0).expect("feasible");
        let w = Grad::from_axis(0, &r.g, dt, 0.0);
        let (mm0, mm1, _, _) = w.moments(0.0).last();
        assert!(mm0[0].abs() < 1e-12 * m1 / (n0 as f64 * dt) && (mm1[0] - m1).abs() < 1e-10 * m1, "{mm0:?} {mm1:?}");
        assert!(w.limits().ok_axis(&sys), "{:?}", w.limits());
        eprintln!("two-lobe {n0} seg: obj {f0:.3e}; robust {} seg: obj {:.3e} ({} it, r_prim {:.1e})", n0 + 20, r.objective, r.iters, r.r_prim);
        assert!(r.objective < 0.5 * f0, "{} vs {f0}", r.objective);
    }
}
