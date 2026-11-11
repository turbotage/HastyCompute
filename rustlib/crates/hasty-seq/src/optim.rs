//! Parametric design search: the base-spoke shape that gives the shortest Nyquist-complete scan (N*·TR) for a fixed
//! readout budget, under G_max, S_max and PNS of the played sequence.
//!
//! For a shape (k_g0, meander amplitude A, meander frequency ω):
//!  1. the spoke count N is the smallest whose time-optimal readout fits the ADC budget (bisection on log N; fewer
//!     spokes = longer curves);
//!  2. PNS by part-wise slew derating: the TR is split into encode / readout / tail; while the worst SAFE
//!     value over the rotation set exceeds the target, the slew factor of the part holding the maximum is scaled by
//!     target / max (then N is re-fitted, since a slower readout covers less);
//!  3. objective N·TR·(1 + λ·anisotropy)·(1 + μ·resonance exposure of the readout, Nyquist cells).
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
    /// resonance exposure of the readout (`robust::band_exposure` on `res_band`, in Nyquist cells × FOV): objective
    /// factor (1 + res_weight · exposure); 0 = off
    pub res_weight: f64,
    pub res_band: Vec<(f64, f64)>,
    pub curve_pts: usize,
    /// out-and-back readout returning to `ret`·kmax and ending at rest ([`Yarn::curve_return`]); the per-rotation tail
    /// is then a straight move to the spoil moment, inside kmax when |spoil| ≤ kmax/γ̄. None = centre-out readout
    /// ending at kmax with a ramp-down / rewinder / spoiler tail.
    pub ret: Option<f64>,
    /// readout geometry beyond `ret` (research/readout_redesign.md); [`Leg::Default`] = what `ret` selects
    pub leg: Leg,
}

/// Readout geometry (research/readout_redesign.md). Every variant other than `Default` ends at rest at k = 0 (so the
/// per-encode flow-compensated tail, `Spoil::FlowComp`, applies unchanged) and marks its reconstruction samples in
/// [`Spoke::recon`]; prephasers and rewinders are navigator ADC. The ADC budget `t_adc` then bounds the span of the
/// reconstruction samples ([`Spoke::t_recon`]), not the whole ADC window.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Leg {
    /// centre-out (`ret` None) or the symmetric out-and-back yarn (`ret` Some)
    #[default]
    Default,
    /// centre-out yarn to kmax ending at rest, then a straight rewind to k = 0 (navigator): every reconstruction sample
    /// on the way out, the low-k (high-energy) samples at the shortest times after the echo
    Rewind,
    /// as `Rewind`, but the rewinder also nulls the readout's first moment ([`Spoke::rewind_m1`]): the TR's net M1 is
    /// then the same for every rotation (not only every encode)
    RewindM1,
    /// asymmetric out-and-back: the return leg is the yarn designed for `back` × the out leg's spokes (fewer turns:
    /// shorter, coarser), sampled down to `ret`·kmax, then the straight rewind (ret > 0); `null_m1`: the rewind also
    /// nulls the readout's first moment ([`Spoke::rewind_m1`])
    Asym { back: f64, ret: f64, null_m1: bool },
    /// two centre-out legs in one readout, each followed by its rewind, the second turned by π about x
    Double,
    /// in-out: a prephaser to the reflected end point, in through k = 0 (the echo) and out again, then the rewind
    InOut,
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
    /// resonance exposure of the readout, Nyquist cells (0 when no band is set)
    pub res: f64,
    pub objective: f64,
}

impl Problem {
    pub fn yarn(&self, s: &Shape, n: f64) -> Yarn {
        Yarn { kmax: self.kmax, fov: self.fov, n_design: n, edge_density: self.edge_density, kg0: s.kg0, wobble: s.wobble, wobble_freq: s.wobble_freq }
    }
    pub fn spoke(&self, s: &Shape, n: f64, slew_ro: f64) -> Spoke {
        self.spoke_rw(s, n, slew_ro, slew_ro)
    }
    /// as [`Problem::spoke`], the final rewinder of a [`Leg`] readout at its own slew factor `slew_rw` (it counts as
    /// the tail for PNS derating, `Tr::part`); default readouts ignore it
    pub fn spoke_rw(&self, s: &Shape, n: f64, slew_ro: f64, slew_rw: f64) -> Spoke {
        let sys = System { smax: self.sys.smax * slew_ro, ..self.sys.clone() };
        let y = self.yarn(s, n);
        if self.leg != Leg::Default {
            let sys_rw = System { smax: self.sys.smax * slew_rw, ..self.sys.clone() };
            return self.spoke_leg(&y, &sys, Some(&sys_rw));
        }
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
    /// readouts of the [`Leg`] variants (vector limits `sys`; the final rewinder at `rw`, left out when None: the
    /// ADC budget only bounds the reconstruction span, so the N fit designs it once at the end)
    fn spoke_leg(&self, y: &Yarn, sys: &System, rw: Option<&System>) -> Spoke {
        let at_rest = TopOpts { vmax: self.vmax, v_end: Some(0.0), ..TopOpts::default() };
        let gb = crate::GAMMA_BAR;
        match self.leg {
            Leg::Default => unreachable!(),
            Leg::Rewind => {
                let c = y.curve(self.curve_pts);
                let mut sp = Spoke::design(&c, sys, &at_rest);
                if let Some(rw) = rw {
                    sp.rewind(rw.gmax, rw.smax);
                }
                sp
            }
            Leg::RewindM1 => {
                let c = y.curve(self.curve_pts);
                let mut sp = Spoke::design(&c, sys, &at_rest);
                if let Some(rw) = rw {
                    sp.rewind_m1(rw.gmax, rw.smax);
                }
                sp
            }
            Leg::Asym { back, ret, null_m1 } => {
                let yb = Yarn { n_design: y.n_design * back, ..y.clone() };
                let c = y.curve_asym(&yb, self.curve_pts, ret);
                let mut sp = Spoke::design(&c, sys, &at_rest);
                if null_m1 || ret > 0.0 {
                    if let Some(rw) = rw {
                        if null_m1 {
                            sp.rewind_m1(rw.gmax, rw.smax);
                        } else {
                            sp.rewind(rw.gmax, rw.smax);
                        }
                    }
                } else {
                    sp.close_to(&crate::scale(c.last().unwrap(), 1.0 / gb), sys.gmax, sys.smax);
                }
                sp
            }
            Leg::Double => {
                let c = y.curve(self.curve_pts);
                let mut sp = Spoke::design(&c, sys, &at_rest);
                let leg_g = sp.grad.g.clone();
                let leg_len = sp.length;
                sp.rewind(sys.gmax, sys.smax);
                let q = crate::axis_angle(&[1.0, 0.0, 0.0], std::f64::consts::PI);
                let second: Vec<crate::Vec3> = leg_g.iter().map(|g| crate::mat_vec(&q, g)).collect();
                sp.push(&second, true, leg_len);
                if let Some(rw) = rw {
                    sp.rewind(rw.gmax, rw.smax);
                }
                sp
            }
            Leg::InOut => {
                let (c, centre) = y.curve_in_out(self.curve_pts);
                let ro = Spoke::design(&c, sys, &at_rest);
                // prephaser from the encode's k = 0 to the curve start, then the in-out readout, then the rewind
                let pre = crate::traj::vector_trapezoid(&crate::scale(&c[0], 1.0 / gb), sys.gmax, sys.smax, sys.grad_raster);
                let mut sp = Spoke { grad: pre.clone(), t_adc: pre.duration(), i_adc: pre.len() - 1, length: 0.0, frac: vec![0.0; pre.len()], m0_end: pre.moments(0.0).last().0, recon: vec![], i_echo: 0 };
                let i0 = sp.i_adc;
                sp.push(&ro.grad.g, true, ro.length);
                // the echo: the raster point closest to the curve's centre (arc-length fraction)
                let fc = centre as f64 / (c.len() - 1) as f64;
                let j = ro.frac.iter().position(|&f| f >= fc).unwrap_or(0);
                sp.i_echo = i0 + j;
                if let Some(rw) = rw {
                    sp.rewind(rw.gmax, rw.smax);
                }
                sp
            }
        }
    }
    /// the time the ADC budget bounds: the whole window (default readouts), the reconstruction span ([`Leg`] variants)
    pub fn budget_time(&self, sp: &Spoke) -> f64 {
        if self.leg == Leg::Default { sp.t_adc } else { sp.t_recon() }
    }
    pub fn n_eff(&self, s: &Shape, n: f64) -> f64 {
        match self.leg {
            Leg::Default => {}
            Leg::Rewind | Leg::RewindM1 => return n,
            Leg::Asym { back, ret, .. } => {
                let y = self.yarn(s, n);
                return y.n_eff_asym(&Yarn { n_design: n * back, ..y.clone() }, ret);
            }
            Leg::Double | Leg::InOut => return 0.5 * n,
        }
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
        self.fit_budget_rw(s, slew_ro, slew_ro, hint)
    }
    /// as [`Problem::fit_budget_near`] with the final rewinder of a [`Leg`] readout at slew factor `slew_rw`
    pub fn fit_budget_rw(&self, s: &Shape, slew_ro: f64, slew_rw: f64, hint: Option<f64>) -> Option<(f64, Spoke)> {
        if self.leg != Leg::Default {
            // bisect on the readout without its final rewinder (outside the budgeted span), then add it once
            let sys = System { smax: self.sys.smax * slew_ro, ..self.sys.clone() };
            let part = |n: f64| self.spoke_leg(&self.yarn(s, n), &sys, None);
            let (n, _) = self.bisect_budget(&part, hint)?;
            return Some((n, self.spoke_rw(s, n, slew_ro, slew_rw)));
        }
        self.fit_budget_default(s, slew_ro, slew_rw, hint)
    }
    /// the N bisection of [`Problem::fit_budget_near`] over any readout constructor
    fn bisect_budget(&self, spoke: &dyn Fn(f64) -> Spoke, hint: Option<f64>) -> Option<(f64, Spoke)> {
        let n_rad = 4.0 * std::f64::consts::PI * self.kmax.powi(2) * self.fov.powi(2) * self.edge_density.max(1.0);
        let (mut lo, mut hi) = (n_rad / 1e4, n_rad * 100.0);
        if let Some(h) = hint {
            let (a, b) = (h / 1.5, h * 1.5);
            if self.budget_time(&spoke(a)) > self.t_adc && b < hi {
                lo = a;
                if self.budget_time(&spoke(b)) <= self.t_adc {
                    hi = b;
                }
            }
        }
        let sp_hi = spoke(hi);
        if self.budget_time(&sp_hi) > self.t_adc {
            return None;
        }
        let mut best = (hi, sp_hi);
        for _ in 0..22 {
            let mid = (lo * hi).sqrt();
            let sp = spoke(mid);
            if self.budget_time(&sp) <= self.t_adc {
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
    fn fit_budget_default(&self, s: &Shape, slew_ro: f64, slew_rw: f64, hint: Option<f64>) -> Option<(f64, Spoke)> {
        // radial (q ≈ 1 everywhere) bound
        let n_rad = 4.0 * std::f64::consts::PI * self.kmax.powi(2) * self.fov.powi(2) * self.edge_density.max(1.0);
        let (mut lo, mut hi) = (n_rad / 1e4, n_rad * 100.0);
        if let Some(h) = hint {
            let (a, b) = (h / 1.5, h * 1.5);
            if self.budget_time(&self.spoke_rw(s, a, slew_ro, slew_rw)) > self.t_adc && b < hi {
                lo = a;
                if self.budget_time(&self.spoke_rw(s, b, slew_ro, slew_rw)) <= self.t_adc {
                    hi = b;
                }
            }
        }
        let sp_hi = self.spoke_rw(s, hi, slew_ro, slew_rw);
        if self.budget_time(&sp_hi) > self.t_adc {
            return None;
        }
        let mut best = (hi, sp_hi);
        for _ in 0..22 {
            let mid = (lo * hi).sqrt();
            let sp = self.spoke_rw(s, mid, slew_ro, slew_rw);
            if self.budget_time(&sp) <= self.t_adc {
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
        TrSpec { sys: self.sys.clone(), enc_slew: slew[0], tail_slew: slew[2], rf_dur: self.rf_dur, scheme: self.scheme, venc: self.venc, balanced: self.balanced, spoil: self.spoil, encode_shape: Default::default(), tail_shape: Default::default() }
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
            // leg readouts: the final rewinder is derated with the tail (it counts as the tail for PNS, `Tr::part`)
            let (n, sp) = if self.leg == Leg::Default { self.fit_budget_near(s, slew[1], hint)? } else { self.fit_budget_rw(s, slew[1], slew[2], hint)? };
            hint = Some(n);
            let p_aniso = sp.grad.moments(0.0).p[sp.i_adc];
            let tr = Tr::build(&self.tr_spec(&slew), sp, rots);
            let pns = pns_sequence(&tr, rots, &self.hw);
            let aniso = concomitant_anisotropy(&p_aniso);
            let feasible = pns.max <= self.pns_target;
            let n_eff = self.n_eff(s, n);
            let res = if self.res_band.is_empty() { 0.0 } else { crate::robust::band_exposure(&tr.spoke.grad, &self.res_band) * self.fov };
            let objective = n_eff * tr.tr() * (1.0 + self.aniso_weight * aniso) * (1.0 + self.res_weight * res);
            let e = Eval { shape: *s, n_design: n, n_eff, slew, tr, pns: pns.clone(), feasible, aniso, res, objective };
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
            res_weight: 0.0,
            res_band: vec![],
            curve_pts: 60_000,
            ret: std::env::var("RET").ok().map(|x| x.parse().unwrap()),
            leg: Leg::Default,
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

#[cfg(test)]
mod leg_tests {
    use super::*;
    use crate::encode::DualLayout;
    use crate::{norm, sub, GAMMA, GAMMA_BAR};

    pub(super) fn problem_for_study(leg: Leg) -> Problem {
        Problem { curve_pts: 60_000, ..problem(leg) }
    }
    fn problem(leg: Leg) -> Problem {
        let res = 2e-3;
        Problem {
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
            spoil: Spoil::FlowComp([0.0, 0.0, 2.0 / (GAMMA_BAR * res)]),
            rot_kind: RotSet::SuperFibonacci,
            n_rot_pns: 400,
            aniso_weight: 0.0,
            res_weight: 0.0,
            res_band: vec![],
            curve_pts: 30_000,
            ret: Some(0.0),
            leg,
        }
    }
    const SHAPE: Shape = Shape { kg0: 0.0, wobble: 0.0, wobble_freq: 2.0 };

    /// every leg ends at rest at k = 0, keeps the reconstruction samples inside kmax, respects the vector limits
    #[test]
    fn legs_end_at_rest_inside_kmax() {
        for leg in [Leg::Rewind, Leg::RewindM1, Leg::Asym { back: 4.0, ret: 0.5, null_m1: false }, Leg::Asym { back: 4.0, ret: 0.5, null_m1: true }, Leg::Asym { back: 2.0, ret: 0.0, null_m1: false }, Leg::Double, Leg::InOut] {
            let p = problem(leg);
            let sp = p.spoke(&SHAPE, 2000.0, 0.45);
            // hardware vector limits; the derated (PNS) slew cap within 1.5 % (the raster traversal overshoots its cap by
            // ~1 % where the curvature changes fast, as the default out-and-back design does: 89.98 vs 89.4 T/m/s; PNS is
            // checked on the actual waveform)
            let l = sp.grad.limits();
            assert!(l.ok_vec(&p.sys), "{leg:?}: {l:?}");
            assert!(l.s_vec <= p.sys.smax * 0.45 * 1.015, "{leg:?}: {l:?}");
            assert!(norm(&sp.g_end()) == 0.0, "{leg:?}");
            assert!(norm(&sp.m0_end) * GAMMA_BAR < 1e-6, "{leg:?}: end k {:.3e}", norm(&sp.m0_end) * GAMMA_BAR);
            let k = sp.grad.moments(0.0).k();
            let kin = k.iter().enumerate().filter(|(i, _)| sp.is_recon(*i)).map(|(_, q)| norm(q)).fold(0.0, f64::max);
            assert!(kin <= p.kmax * (1.0 + 1e-3), "{leg:?}: recon |k| {kin}");
            assert!(sp.t_recon() <= sp.t_adc + 1e-12 && sp.t_recon_samples() <= sp.t_recon() + 1e-12);
            if matches!(leg, Leg::Rewind | Leg::RewindM1) {
                // the reconstruction part is the out leg only, starting at the echo
                assert_eq!(sp.recon.len(), 1);
                assert_eq!(sp.recon[0][0], 0);
            }
            if leg == Leg::Double {
                assert_eq!(sp.recon.len(), 2);
            }
            if leg == Leg::InOut {
                assert!(norm(&k[sp.i_echo]) < 2.0, "in-out echo at |k| {}", norm(&k[sp.i_echo]));
            }
            let m1 = GAMMA * norm(&sp.m1_end());
            if matches!(leg, Leg::RewindM1 | Leg::Asym { null_m1: true, .. }) {
                assert!(m1 < 1e-6, "M1-nulled rewind left γ|M1| {m1}");
            }
            println!("{leg:?}: ADC {:.2} ms, recon span {:.2} ms, end γ|M1| {m1:.3} rad/(m/s), n_eff {:.0}", sp.t_adc * 1e3, sp.t_recon() * 1e3, p.n_eff(&SHAPE, 2000.0));
        }
    }

    /// the asymmetric construction with back = 1 is the symmetric out-and-back curve
    #[test]
    fn asym_with_equal_legs_is_curve_return() {
        let p = problem(Leg::Default);
        let y = p.yarn(&SHAPE, 2000.0);
        let a = y.curve_return(20_000, 0.0);
        let b = y.curve_asym(&y, 20_000, 0.0);
        // curve_asym drops ≤ a few near-duplicate points at the junction: same out leg, same return leg (aligned from
        // the end), away from the smoothed turn
        assert!(a.len().abs_diff(b.len()) <= 4, "{} vs {}", a.len(), b.len());
        let m = y.curve(20_000).len();
        let d_out = a[..m - 400].iter().zip(&b[..m - 400]).map(|(x, z)| norm(&sub(x, z))).fold(0.0, f64::max);
        let d_ret = a.iter().rev().zip(b.iter().rev()).take(a.len() - m - 400).map(|(x, z)| norm(&sub(x, z))).fold(0.0, f64::max);
        assert!(d_out < 1e-9 * y.kmax && d_ret < 1e-6 * y.kmax, "max deviation out {d_out}, return {d_ret}");
        // n_eff takes N = n_design / 2 (q ≈ f); n_eff_asym the exact max over r of 4πr²D²w / (2q)
        assert!((y.n_eff(0.0) - y.n_eff_asym(&y, 0.0)).abs() < 1e-3 * y.n_design, "{} vs {}", y.n_eff(0.0), y.n_eff_asym(&y, 0.0));
    }

    /// FlowComp TR with the rewind legs: constant physical M0 and equal TR M1 across encodes, for any rotation; with the
    /// M1-nulled rewind also equal across rotations
    #[test]
    fn flowcomp_with_rewind_legs() {
        for leg in [Leg::Rewind, Leg::RewindM1] {
            let p = problem(leg);
            let sp = p.spoke(&SHAPE, 2000.0, 0.45);
            let rots = rotations(RotSet::SuperFibonacci, 16);
            let tr = Tr::build(&p.tr_spec(&[0.3, 0.45, 0.4]), sp, &rots);
            let Spoil::FlowComp(spoil) = p.spoil else { unreachable!() };
            let mut m1_rot: Vec<crate::Vec3> = vec![];
            for r in &rots {
                let ends: Vec<_> = (0..tr.n_enc()).map(|e| tr.physical(e, r).moments(tr.t_rf).last()).collect();
                for (m0, m1, _, _) in &ends {
                    assert!(norm(&sub(m0, &spoil)) < 1e-6 * norm(&spoil));
                    assert!(GAMMA * norm(&sub(m1, &ends[0].1)) < 1e-6, "{leg:?}: encode M1 spread");
                }
                m1_rot.push(ends[0].1);
            }
            let spread = m1_rot.iter().map(|m| GAMMA * norm(&sub(m, &m1_rot[0]))).fold(0.0, f64::max);
            println!("{leg:?}: TR {:.3} ms, net-M1 spread across rotations {spread:.3e} rad/(m/s)", tr.tr() * 1e3);
            if leg == Leg::RewindM1 {
                assert!(spread < 1e-5, "rotation spread {spread}");
            }
        }
    }
}

#[cfg(test)]
mod merged_tail_study {
    use super::*;
    use crate::encode::{design, EncodeSpec};
    use crate::{mat_vec, norm, GAMMA};

    /// Study (research/readout_redesign.md §7): instead of an M1-nulled rewinder, the per-encode flow-compensated tail
    /// could absorb the plain rewinder's rotated first moment R·m1 (a per-rotation block). Longest such block over
    /// rotations vs the rotation-independent block and the extra rewinder time.
    ///   cargo test --release -p hasty-seq merged_tail_study -- --ignored --nocapture
    #[test]
    #[ignore]
    fn merged_tail_length() {
        let p = super::leg_tests::problem_for_study(Leg::Rewind);
        let shape = Shape { kg0: -0.05, wobble: 0.0833, wobble_freq: 11.37 };
        let (_, sp) = p.fit_budget_rw(&shape, 0.444, 0.373, None).unwrap();
        let (_, spm) = Problem { leg: Leg::RewindM1, ..p.clone() }.fit_budget_rw(&shape, 0.444, 0.373, None).unwrap();
        let rots = rotations(RotSet::SuperFibonacci, 64);
        let spec = p.tr_spec(&[0.275, 0.444, 0.373]);
        let tr = Tr::build(&spec, sp.clone(), &rots);
        let m1r = sp.m1_end();
        let Spoil::FlowComp(spoil) = p.spoil else { unreachable!() };
        let tail_sys = System { smax: p.sys.smax * 0.373, ..p.sys.clone() };
        let t_s = tr.i_tail as f64 * tr.dt;
        // the rotation-independent block's M1 offset convention (Tr::build): half = 0.5 × block duration
        let half = 0.5 * tr.n_tail as f64 * tr.dt;
        let off = [0, 1, 2].map(|a| spoil[a] * (t_s - tr.t_rf + half));
        let mut longest = 0usize;
        for r in &rots {
            let rm = mat_vec(r, &m1r);
            let es = EncodeSpec { m0: spoil, m1_offset: [0, 1, 2].map(|a| off[a] - rm[a]), scheme: spec.scheme, venc: -spec.venc, balanced: spec.balanced, t_start: t_s, t_ref: tr.t_rf };
            longest = longest.max(design(&es, &tail_sys).waves[0].len() - 1);
        }
        println!(
            "plain rewinder {:.2} ms, M1-nulled rewinder {:.2} ms; flow-comp block {:.2} ms (rotation-independent) vs {:.2} ms (absorbing R·m1, γ|m1| {:.2} rad/(m/s), longest of {} rotations)",
            (sp.i_adc - sp.recon[0][1]) as f64 * tr.dt * 1e3, (spm.i_adc - spm.recon[0][1]) as f64 * tr.dt * 1e3, tr.n_tail as f64 * tr.dt * 1e3, longest as f64 * tr.dt * 1e3, GAMMA * norm(&m1r), rots.len()
        );
    }
}
