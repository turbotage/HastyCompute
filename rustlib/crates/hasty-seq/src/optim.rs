//! Parametric design search: the base-spoke shape that gives the shortest Nyquist-complete scan (N*·TR) for a fixed
//! readout budget, under G_max, S_max and PNS of the played sequence.
//!
//! For a shape (k_g0, meander amplitude A, meander frequency ω):
//!  1. the spoke count N is the smallest whose time-optimal readout fits the ADC budget (bisection on log N; fewer
//!     spokes = longer curves);
//!  2. PNS by part-wise slew derating: the TR is split into encode / readout / tail; while the worst SAFE
//!     value over the rotation set exceeds the target, the slew factor of the part holding the maximum is scaled by
//!     target / max (then N is re-fitted, since a slower readout covers less);
//!  3. objective N·TR·(1 + λ·anisotropy).
//! The shape is searched by a coarse grid and Nelder–Mead.

use crate::encode::Scheme;
use crate::metrics::concomitant_anisotropy;
use crate::pns::SafeHw;
use crate::system::System;
use crate::tr::{pns_sequence, SeqPns, Spoil, Tr, TrSpec};
use crate::traj::{acquisition_order, rotations, RotSet, Spoke, TopOpts, Yarn};
use crate::Mat3;

#[derive(Clone, Debug)]
pub struct Problem {
    pub sys: System,
    pub hw: SafeHw,
    /// PNS limit, % of the SAFE threshold
    pub pns_target: f64,
    pub kmax: f64,
    pub fov: f64,
    pub edge_density: f64,
    /// ADC window budget, s
    pub t_adc: f64,
    /// speed cap (ADC bandwidth), cycles/m/s
    pub vmax: Option<f64>,
    pub rf_dur: f64,
    pub scheme: Scheme,
    pub venc: f64,
    pub balanced: bool,
    pub spoil: Spoil,
    /// rotations used for the PNS check (acquisition order)
    pub rot_kind: RotSet,
    pub n_rot_pns: usize,
    pub aniso_weight: f64,
    pub curve_pts: usize,
    /// out-and-back readout returning to `ret`·kmax and ending at rest ([`Yarn::curve_return`]); the per-rotation tail
    /// is then a straight move to the spoil moment, inside kmax when |spoil| ≤ kmax/γ̄. None = centre-out readout
    /// ending at kmax with a ramp-down / rewinder / spoiler tail.
    pub ret: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub kg0: f64,
    pub wobble: f64,
    pub wobble_freq: f64,
}

#[derive(Clone, Debug)]
pub struct Eval {
    pub shape: Shape,
    /// yarn design parameter (per leg)
    pub n_design: f64,
    /// spokes per Nyquist-complete set (= n_design for centre-out, ≈ n_design / 2 out-and-back)
    pub n_eff: f64,
    /// slew factors: encode, readout, tail
    pub slew: [f64; 3],
    pub tr: Tr,
    pub pns: SeqPns,
    pub feasible: bool,
    pub aniso: f64,
    pub objective: f64,
}

impl Problem {
    pub fn yarn(&self, s: &Shape, n: f64) -> Yarn {
        Yarn { kmax: self.kmax, fov: self.fov, n_design: n, edge_density: self.edge_density, kg0: s.kg0, wobble: s.wobble, wobble_freq: s.wobble_freq }
    }
    pub fn spoke(&self, s: &Shape, n: f64, slew_ro: f64) -> Spoke {
        let sys = System { smax: self.sys.smax * slew_ro, ..self.sys.clone() };
        let y = self.yarn(s, n);
        match self.ret {
            None => Spoke::design(&y.curve(self.curve_pts), &sys, &TopOpts { vmax: self.vmax, ..TopOpts::default() }),
            Some(r) => {
                let c = y.curve_return(self.curve_pts, r);
                let mut sp = Spoke::design(&c, &sys, &TopOpts { vmax: self.vmax, v_end: Some(0.0), ..TopOpts::default() });
                let before = crate::norm(&sp.m0_end) * crate::GAMMA_BAR;
                sp.close_to(&crate::scale(c.last().unwrap(), 1.0 / crate::GAMMA_BAR), sys.gmax, sys.smax);
                if std::env::var("HASTY_SEQ_DEBUG").is_ok() {
                    eprintln!("readout end residual {before:.3e} /m closed to {:.3e} /m", crate::norm(&crate::sub(&crate::scale(&sp.m0_end, crate::GAMMA_BAR), c.last().unwrap())));
                }
                sp
            }
        }
    }
    pub fn n_eff(&self, s: &Shape, n: f64) -> f64 {
        match self.ret {
            None => n,
            Some(r) => self.yarn(s, n).n_eff(r),
        }
    }
    /// N for which the readout just fits the budget (None if even radial spokes do not fit)
    pub fn fit_budget(&self, s: &Shape, slew_ro: f64) -> Option<(f64, Spoke)> {
        self.fit_budget_near(s, slew_ro, None)
    }
    /// as [`Problem::fit_budget`], bracketing around a previous solution `hint` when given
    pub fn fit_budget_near(&self, s: &Shape, slew_ro: f64, hint: Option<f64>) -> Option<(f64, Spoke)> {
        // radial (q ≈ 1 everywhere) bound
        let n_rad = 4.0 * std::f64::consts::PI * self.kmax.powi(2) * self.fov.powi(2) * self.edge_density.max(1.0);
        let (mut lo, mut hi) = (n_rad / 1e4, n_rad * 100.0);
        if let Some(h) = hint {
            let (a, b) = (h / 1.5, h * 1.5);
            if self.spoke(s, a, slew_ro).t_adc > self.t_adc && b < hi {
                lo = a;
                if self.spoke(s, b, slew_ro).t_adc <= self.t_adc {
                    hi = b;
                }
            }
        }
        let sp_hi = self.spoke(s, hi, slew_ro);
        if sp_hi.t_adc > self.t_adc {
            return None;
        }
        let mut best = (hi, sp_hi);
        for _ in 0..22 {
            let mid = (lo * hi).sqrt();
            let sp = self.spoke(s, mid, slew_ro);
            if sp.t_adc <= self.t_adc {
                hi = mid;
                best = (mid, sp);
            } else {
                lo = mid;
            }
            if hi / lo < 1.002 {
                break;
            }
        }
        Some(best)
    }
    pub fn tr_spec(&self, slew: &[f64; 3]) -> TrSpec {
        TrSpec { sys: self.sys.clone(), enc_slew: slew[0], tail_slew: slew[2], rf_dur: self.rf_dur, scheme: self.scheme, venc: self.venc, balanced: self.balanced, spoil: self.spoil }
    }
    pub fn pns_rotations(&self) -> Vec<Mat3> {
        let r = rotations(self.rot_kind, self.n_rot_pns);
        acquisition_order(self.rot_kind, r.len()).into_iter().map(|i| r[i]).collect()
    }

    /// N fit + part-wise PNS derating for one shape
    pub fn evaluate(&self, s: &Shape, rots: &[Mat3]) -> Option<Eval> {
        let mut slew = [1.0f64; 3];
        let mut last: Option<Eval> = None;
        let mut hint = None;
        for _ in 0..24 {
            let (n, sp) = self.fit_budget_near(s, slew[1], hint)?;
            hint = Some(n);
            let p_aniso = sp.grad.moments(0.0).p[sp.i_adc];
            let tr = Tr::build(&self.tr_spec(&slew), sp, rots);
            let pns = pns_sequence(&tr, rots, &self.hw);
            let aniso = concomitant_anisotropy(&p_aniso);
            let feasible = pns.max <= self.pns_target;
            let n_eff = self.n_eff(s, n);
            let objective = n_eff * tr.tr() * (1.0 + self.aniso_weight * aniso);
            let e = Eval { shape: *s, n_design: n, n_eff, slew, tr, pns: pns.clone(), feasible, aniso, objective };
            if feasible {
                return Some(e);
            }
            // derate every part above the target at once (a maximum in the quiet RF part comes from the previous
            // TR's tail); parts couple through the SAFE memory, so iterate
            let mut m = [pns.part_max[1], pns.part_max[2], pns.part_max[3].max(pns.part_max[0])];
            for k in 0..3 {
                m[k] /= self.pns_target;
                if m[k] > 1.0 {
                    slew[k] *= (1.0 / m[k]).min(0.995);
                }
            }
            last = Some(e);
            if slew.iter().any(|&x| x < 0.05) {
                break;
            }
        }
        last.map(|mut e| {
            e.objective = f64::INFINITY;
            e
        })
    }
}

/// Nelder–Mead minimisation (standard coefficients); returns the best point and value
pub fn nelder_mead(f: &mut dyn FnMut(&[f64]) -> f64, x0: &[f64], step: &[f64], iters: usize, tol: f64) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut pts: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] += step[i];
        pts.push(p);
    }
    let mut val: Vec<f64> = pts.iter().map(|p| f(p)).collect();
    for _ in 0..iters {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| val[a].partial_cmp(&val[b]).unwrap_or(std::cmp::Ordering::Equal));
        pts = idx.iter().map(|&i| pts[i].clone()).collect();
        val = idx.iter().map(|&i| val[i]).collect();
        if (val[n] - val[0]).abs() <= tol * val[0].abs().max(1e-30) {
            break;
        }
        let c: Vec<f64> = (0..n).map(|j| pts[..n].iter().map(|p| p[j]).sum::<f64>() / n as f64).collect();
        let at = |a: f64| -> Vec<f64> { (0..n).map(|j| c[j] + a * (pts[n][j] - c[j])).collect() };
        let xr = at(-1.0);
        let fr = f(&xr);
        if fr < val[0] {
            let xe = at(-2.0);
            let fe = f(&xe);
            if fe < fr {
                pts[n] = xe;
                val[n] = fe;
            } else {
                pts[n] = xr;
                val[n] = fr;
            }
        } else if fr < val[n - 1] {
            pts[n] = xr;
            val[n] = fr;
        } else {
            let (xc, fc) = if fr < val[n] {
                let x = at(-0.5);
                let v = f(&x);
                (x, v)
            } else {
                let x = at(0.5);
                let v = f(&x);
                (x, v)
            };
            if fc < val[n].min(fr) {
                pts[n] = xc;
                val[n] = fc;
            } else {
                for i in 1..=n {
                    pts[i] = (0..n).map(|j| pts[0][j] + 0.5 * (pts[i][j] - pts[0][j])).collect();
                    val[i] = f(&pts[i]);
                }
            }
        }
    }
    let b = (0..=n).min_by(|&a, &b| val[a].partial_cmp(&val[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    (pts[b].clone(), val[b])
}

/// optimisation variables ↔ shape: (k_g0, A, ln ω), A clamped ≥ 0, ω in [0.2, 30]
pub fn shape_of(x: &[f64]) -> Shape {
    Shape { kg0: x[0].clamp(-4.0, 4.0), wobble: x[1].clamp(0.0, 6.0), wobble_freq: x[2].exp().clamp(0.2, 30.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nelder_mead_rosenbrock() {
        let mut f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let (x, v) = nelder_mead(&mut f, &[-1.2, 1.0], &[0.5, 0.5], 2000, 1e-14);
        assert!(v < 1e-8 && (x[0] - 1.0).abs() < 1e-3, "{x:?} {v}");
    }
}

#[cfg(test)]
mod profile {
    use super::*;
    use crate::encode::DualLayout;
    use std::time::Instant;

    /// cargo test --release -p hasty-seq profile_evaluate -- --ignored --nocapture
    #[test]
    #[ignore]
    fn profile_evaluate() {
        let res = 2e-3;
        let p = Problem {
            sys: System::prisma_xr(),
            hw: SafeHw::example(),
            pns_target: 80.0,
            kmax: 0.5 / res,
            fov: 0.32,
            edge_density: 1.0,
            t_adc: 8e-3,
            vmax: None,
            rf_dur: 100e-6,
            scheme: Scheme::Coprime { p: 2, layout: DualLayout::Seven },
            venc: 1.0,
            balanced: true,
            spoil: Spoil::Physical([0.0, 0.0, 2.0 / (crate::GAMMA_BAR * res)]),
            rot_kind: RotSet::SuperFibonacci,
            n_rot_pns: 400,
            aniso_weight: 0.0,
            curve_pts: 60_000,
            ret: std::env::var("RET").ok().map(|x| x.parse().unwrap()),
        };
        let p = if p.ret.is_some() { Problem { spoil: Spoil::Physical([0.0, 0.0, 0.5 / (crate::GAMMA_BAR * res)]), ..p } } else { p };
        let s = Shape { kg0: 0.0, wobble: 0.0, wobble_freq: 2.0 };
        let rots = p.pns_rotations();
        let t = Instant::now();
        let y = p.yarn(&s, 2000.0);
        let c = y.curve(p.curve_pts);
        println!("curve {:.3} s ({} pts)", t.elapsed().as_secs_f64(), c.len());
        let t = Instant::now();
        let sp = p.spoke(&s, 2000.0, 0.5);
        println!("one time_optimal {:.3} s (length {:.0}, t_adc {:.2} ms)", t.elapsed().as_secs_f64(), sp.length, sp.t_adc * 1e3);
        let t = Instant::now();
        let (n, sp) = p.fit_budget(&s, 0.5).unwrap();
        println!("fit_budget {:.3} s (N {n:.0})", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let tr = Tr::build(&p.tr_spec(&[0.3, 0.5, 0.5]), sp, &rots);
        println!("Tr::build (tails over {} rotations) {:.3} s", rots.len(), t.elapsed().as_secs_f64());
        let t = Instant::now();
        let ps = pns_sequence(&tr, &rots, &p.hw);
        println!("pns_sequence {:.3} s (max {:.1})", t.elapsed().as_secs_f64(), ps.max);
        let t = Instant::now();
        let e = p.evaluate(&s, &rots).unwrap();
        println!("evaluate {:.3} s (slew {:?}, N {:.0} (design {:.0}), TR {:.2} ms, N·TR {:.2} s, PNS {:.1} {:?})", t.elapsed().as_secs_f64(), e.slew, e.n_eff, e.n_design, e.tr.tr() * 1e3, e.objective, e.pns.max, e.pns.part);
    }
}
