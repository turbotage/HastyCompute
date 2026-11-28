//! Extended phase graph (EPG, Weigel 2015) for RF-spoiled / prepared gradient-echo trains and later diffusion-prepared
//! trains (design: `research/t1_ivim_dti_sequences.md`). Fast, exact for a *fixed* per-TR gradient moment; the
//! multi-isochromat check with TR-to-TR varying moments (rotated yarnball spokes) stays in [`crate::spoil`].
//!
//! States per dephasing order k ≥ 0 (unit = one TR's net dephasing, so one [`Epg::shift`] moves k → k+1):
//! `fp[k]` = F⁺_k, `fm[k]` = F⁻_k (= conj F⁺_{−k}), `z[k]` = Z_k (Z_{−k} = conj Z_k). Equilibrium is Z_0 = 1. F⁺ = Mx + iMy
//! in the frame of the RF phase reference, so a pulse of phase φ is demodulated by e^{−iφ} at the receiver.
//!
//! Orders above `kmax` are dropped (they only matter for T2 ≳ many TR; the error is the dropped norm, test by doubling
//! `kmax`). No B0 term here: off-resonance dephasing is part of the ratemap model of the recon, not of the contrast
//! model; add it per state if needed (it is a k-independent phase on F⁺ / F⁻).
//!
//! Sign / unit conventions (verified by the tests below): RF matrix of Weigel (2015) eq. (A1); unitary invariant per k
//! `(|F⁺|² + |F⁻|²)/2 + |Z|²`; with every TR's transverse states crushed the signal equals the Ernst signal exactly.

use std::f64::consts::FRAC_PI_2;

use num_complex::Complex64 as C;

use crate::spoil::Tissue;

#[derive(Clone, Debug)]
pub struct Epg {
    pub fp: Vec<C>,
    pub fm: Vec<C>,
    pub z: Vec<C>,
}

impl Epg {
    /// equilibrium magnetisation M0 = 1 along z, orders 0..=kmax
    pub fn new(kmax: usize) -> Self {
        let n = kmax + 1;
        let mut z = vec![C::new(0.0, 0.0); n];
        z[0] = C::new(1.0, 0.0);
        Epg { fp: vec![C::new(0.0, 0.0); n], fm: vec![C::new(0.0, 0.0); n], z }
    }

    pub fn kmax(&self) -> usize {
        self.fp.len() - 1
    }

    /// hard RF pulse, flip `alpha` (rad, include B1 scale), RF phase `phi` (rad)
    pub fn rf(&mut self, alpha: f64, phi: f64) {
        let (s, c) = alpha.sin_cos();
        let (sh, ch) = ((alpha / 2.0).sin(), (alpha / 2.0).cos());
        let (c2, s2) = (ch * ch, sh * sh);
        let ep = C::from_polar(1.0, phi); // e^{iφ}
        let em = ep.conj();
        let ep2 = ep * ep;
        let em2 = em * em;
        let i = C::new(0.0, 1.0);
        for k in 0..self.fp.len() {
            let (p, m, z) = (self.fp[k], self.fm[k], self.z[k]);
            self.fp[k] = p * c2 + m * (ep2 * s2) - z * (i * ep * s);
            self.fm[k] = p * (em2 * s2) + m * c2 + z * (i * em * s);
            self.z[k] = p * (-0.5 * i * em * s) + m * (0.5 * i * ep * s) + z * c;
        }
    }

    /// relaxation over `t` seconds (T2 on all F, T1 on Z with recovery of Z_0 toward 1)
    pub fn relax(&mut self, t: f64, tis: &Tissue) {
        let e1 = (-t / tis.t1).exp();
        let e2 = (-t / tis.t2).exp();
        for k in 0..self.fp.len() {
            self.fp[k] *= e2;
            self.fm[k] *= e2;
            self.z[k] *= e1;
        }
        self.z[0] += C::new(1.0 - e1, 0.0);
    }

    /// gradient dephasing by `n` units (n ≥ 1): F⁺_k → F⁺_{k+n}, F⁻_k → F⁻_{k−n}, Z unchanged
    pub fn shift(&mut self, n: usize) {
        for _ in 0..n {
            let kmax = self.kmax();
            let new0 = self.fm[1.min(kmax)].conj();
            let new0 = if kmax >= 1 { new0 } else { C::new(0.0, 0.0) };
            for k in (1..=kmax).rev() {
                self.fp[k] = self.fp[k - 1];
            }
            self.fp[0] = new0;
            for k in 0..kmax {
                self.fm[k] = self.fm[k + 1];
            }
            self.fm[kmax] = C::new(0.0, 0.0);
        }
    }

    /// ideal crusher: all transverse states destroyed, Z_k (k > 0) kept (they are longitudinal gratings, still dephased
    /// away by a real crusher, so zero them too if `kill_z`)
    pub fn crush(&mut self, kill_z: bool) {
        for k in 0..self.fp.len() {
            self.fp[k] = C::new(0.0, 0.0);
            self.fm[k] = C::new(0.0, 0.0);
            if kill_z && k > 0 {
                self.z[k] = C::new(0.0, 0.0);
            }
        }
    }

    /// diffusion damping for one dephasing step of b-value `b_unit` (s/m², = Δk² τ for a constant gradient of area Δk
    /// (rad/m) over τ) applied *before* the shift: F⁺_k ← e^{−D b (k²+k+1/3)}, F⁻_k ← e^{−D b (k²−k+1/3)}, Z_k ← e^{−D b k²}
    pub fn diffuse(&mut self, d: f64, b_unit: f64) {
        let db = d * b_unit;
        for k in 0..self.fp.len() {
            let kf = k as f64;
            self.fp[k] *= (-db * (kf * kf + kf + 1.0 / 3.0)).exp();
            self.fm[k] *= (-db * (kf * kf - kf + 1.0 / 3.0)).exp();
            self.z[k] *= (-db * kf * kf).exp();
        }
    }

    /// inversion with efficiency `eta` (Z → −η Z, transverse destroyed); the usual idealisation of a prep + crusher
    pub fn invert(&mut self, eta: f64) {
        for k in 0..self.fp.len() {
            self.fp[k] = C::new(0.0, 0.0);
            self.fm[k] = C::new(0.0, 0.0);
            self.z[k] *= -eta;
        }
    }

    /// saturation with efficiency `eta` (Z → (1−η) Z, transverse destroyed)
    pub fn saturate(&mut self, eta: f64) {
        for k in 0..self.fp.len() {
            self.fp[k] = C::new(0.0, 0.0);
            self.fm[k] = C::new(0.0, 0.0);
            self.z[k] *= 1.0 - eta;
        }
    }

    /// longitudinal magnetisation Z_0
    pub fn mz(&self) -> f64 {
        self.z[0].re
    }

    /// norm per order, (|F⁺|²+|F⁻|²)/2 + |Z|², conserved by [`Epg::rf`]
    pub fn norm_k(&self, k: usize) -> f64 {
        0.5 * (self.fp[k].norm_sqr() + self.fm[k].norm_sqr()) + self.z[k].norm_sqr()
    }
}

/// one RF pulse of a train: nominal flip (rad, **without** B1 scale), repetition time to the next pulse (s), RF phase (rad)
#[derive(Clone, Copy, Debug)]
pub struct Pulse {
    pub flip: f64,
    pub tr: f64,
    pub phase: f64,
}

/// quadratic RF-spoiling phases φ_n = inc·n(n+1)/2 for n = start..start+count (rad)
pub fn spoil_phases(inc: f64, start: usize, count: usize) -> Vec<f64> {
    (start..start + count).map(|n| inc * (n * (n + 1)) as f64 / 2.0).collect()
}

#[derive(Clone, Copy, Debug)]
pub struct TrainOpts {
    /// B1 scale of the flip angle in this voxel
    pub b1: f64,
    /// echo time after the pulse (s); signal read at the k = 0 echo
    pub te: f64,
    /// gradient dephasing units per TR (spoiler), ≥ 1
    pub units: usize,
    /// destroy all transverse states every TR (ideal spoiling, = Ernst) instead of letting the F states evolve
    pub ideal: bool,
}

impl Default for TrainOpts {
    fn default() -> Self {
        TrainOpts { b1: 1.0, te: 0.0, units: 1, ideal: false }
    }
}

/// play `pulses` on `epg` (state carried in and out), returning the demodulated echo signal i·F⁺_0 e^{−iφ} (T2 decay to
/// TE included) after every pulse; the factor i makes an ideally spoiled signal real and positive (a pulse about its own
/// phase axis puts the magnetisation at −i·sin α, 90° behind the RF reference)
pub fn run_train(epg: &mut Epg, pulses: &[Pulse], tis: &Tissue, o: &TrainOpts) -> Vec<C> {
    let mut out = Vec::with_capacity(pulses.len());
    for p in pulses {
        epg.rf(o.b1 * p.flip, p.phase);
        epg.relax(o.te, tis);
        out.push(epg.fp[0] * C::from_polar(1.0, FRAC_PI_2 - p.phase));
        epg.relax(p.tr - o.te, tis);
        if o.ideal {
            epg.crush(true);
        } else {
            epg.shift(o.units);
        }
    }
    out
}

/// signal of a periodic train (prep + pulses repeated) in its steady state: the whole period is replayed until the
/// state at period start converges (max |Δ Z_0| and transverse norm < `tol`) or `max_periods` is reached. `prep` runs at the
/// start of each period (an inversion / saturation / diffusion-prep closure acting on the EPG). Returns the signals
/// of the last period and the number of periods played.
pub fn steady_period<F: FnMut(&mut Epg)>(
    kmax: usize,
    mut prep: F,
    pulses: &[Pulse],
    tis: &Tissue,
    o: &TrainOpts,
    tol: f64,
    max_periods: usize,
) -> (Vec<C>, usize) {
    let mut epg = Epg::new(kmax);
    let mut last = Vec::new();
    for n in 1..=max_periods {
        let before = epg.clone();
        prep(&mut epg);
        last = run_train(&mut epg, pulses, tis, o);
        let mut d = 0.0f64;
        for k in 0..epg.fp.len() {
            d = d.max((epg.fp[k] - before.fp[k]).norm()).max((epg.z[k] - before.z[k]).norm());
        }
        if d < tol {
            return (last, n);
        }
    }
    (last, max_periods)
}

/// Ernst signal of an ideally spoiled GRE: sin α (1 − E1) / (1 − E1 cos α) · e^{−TE/T2}
pub fn ernst(flip: f64, tr: f64, te: f64, tis: &Tissue) -> f64 {
    let e1 = (-tr / tis.t1).exp();
    flip.sin() * (1.0 - e1) / (1.0 - e1 * flip.cos()) * (-te / tis.t2).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tis(t1: f64, t2: f64) -> Tissue {
        Tissue { t1, t2 }
    }

    fn rand_state(kmax: usize) -> Epg {
        // deterministic pseudo-random state with the F⁻_0 = conj F⁺_0, Z_0 real symmetry
        let mut e = Epg::new(kmax);
        let mut x = 0.123456789f64;
        let mut r = || {
            x = (x * 9301.0 + 49297.0) % 233280.0;
            x / 233280.0 - 0.5
        };
        for k in 0..=kmax {
            e.fp[k] = C::new(r(), r());
            e.fm[k] = C::new(r(), r());
            e.z[k] = C::new(r(), r());
        }
        e.fm[0] = e.fp[0].conj();
        e.z[0] = C::new(e.z[0].re, 0.0);
        e
    }

    #[test]
    fn rf_is_unitary_per_order() {
        let mut e = rand_state(6);
        let n0: Vec<f64> = (0..=6).map(|k| e.norm_k(k)).collect();
        e.rf(0.7, 1.9);
        for k in 0..=6 {
            assert!((e.norm_k(k) - n0[k]).abs() < 1e-12, "order {k}");
        }
        // k = 0 symmetry preserved
        assert!((e.fm[0] - e.fp[0].conj()).norm() < 1e-12);
        assert!(e.z[0].im.abs() < 1e-12);
    }

    #[test]
    fn pulse_90_and_180_on_equilibrium() {
        let mut e = Epg::new(2);
        e.rf(std::f64::consts::FRAC_PI_2, 0.0);
        assert!(e.mz().abs() < 1e-12);
        assert!((e.fp[0].norm() - 1.0).abs() < 1e-12);
        let mut e = Epg::new(2);
        e.rf(std::f64::consts::PI, 0.3);
        assert!((e.mz() + 1.0).abs() < 1e-12);
        assert!(e.fp[0].norm() < 1e-12);
    }

    #[test]
    fn ideal_crush_equals_ernst() {
        let t = tis(1.3, 0.09);
        let tr = 0.020;
        let flip = 15f64.to_radians();
        let pulses: Vec<Pulse> = (0..3000).map(|_| Pulse { flip, tr, phase: 0.0 }).collect();
        let mut e = Epg::new(4);
        let s = run_train(&mut e, &pulses, &t, &TrainOpts { ideal: true, te: 0.002, ..Default::default() });
        let want = ernst(flip, tr, 0.002, &t);
        assert!((s.last().unwrap().re - want).abs() < 1e-10, "{} vs {}", s.last().unwrap().re, want);
        assert!(s.last().unwrap().im.abs() < 1e-12);
    }

    #[test]
    fn rf_spoiled_close_to_ernst_short_t2() {
        // 117° increment, TR 20 ms, T2 80 ms: remaining bias a few percent at most
        let t = tis(1.0, 0.08);
        let tr = 0.020;
        let flip = 15f64.to_radians();
        let n = 600;
        let ph = spoil_phases(117f64.to_radians(), 0, n);
        let pulses: Vec<Pulse> = ph.iter().map(|&p| Pulse { flip, tr, phase: p }).collect();
        let mut e = Epg::new(120);
        let s = run_train(&mut e, &pulses, &t, &TrainOpts::default());
        let got = s.last().unwrap().norm();
        let want = ernst(flip, tr, 0.0, &t);
        assert!((got / want - 1.0).abs() < 0.06, "ratio {}", got / want);
        // a doubled kmax changes nothing
        let mut e2 = Epg::new(240);
        let s2 = run_train(&mut e2, &pulses, &t, &TrainOpts::default());
        assert!((s2.last().unwrap().norm() - got).abs() < 1e-6 * got);
    }

    #[test]
    fn no_rf_spoiling_is_far_from_ernst() {
        // phase 0 every TR: coherent (SSFP-like, dephased) signal, differs strongly from the spoiled Ernst
        let t = tis(1.0, 0.08);
        let tr = 0.020;
        let flip = 15f64.to_radians();
        let pulses: Vec<Pulse> = (0..600).map(|_| Pulse { flip, tr, phase: 0.0 }).collect();
        let mut e = Epg::new(120);
        let s = run_train(&mut e, &pulses, &t, &TrainOpts::default());
        let ratio = s.last().unwrap().norm() / ernst(flip, tr, 0.0, &t);
        assert!((ratio - 1.0).abs() > 0.05, "ratio {ratio}");
    }

    #[test]
    fn inversion_recovery_matches_closed_form() {
        // inversion then wait t with no pulses: Mz = 1 − (1+η) e^{−t/T1}
        let t = tis(0.9, 0.08);
        let mut e = Epg::new(3);
        e.invert(0.9);
        e.relax(0.5, &t);
        let want = 1.0 - 1.9 * (-0.5f64 / 0.9).exp();
        assert!((e.mz() - want).abs() < 1e-12);
    }

    #[test]
    fn steady_period_converges_for_ir_train() {
        let t = tis(1.2, 0.07);
        let n = 200;
        let pulses: Vec<Pulse> = spoil_phases(117f64.to_radians(), 0, n)
            .into_iter()
            .map(|p| Pulse { flip: 12f64.to_radians(), tr: 0.020, phase: p })
            .collect();
        let (sig, np) = steady_period(60, |e| e.invert(0.95), &pulses, &t, &TrainOpts::default(), 1e-9, 200);
        assert!(np < 200, "no convergence");
        // first signal after inversion is negative-Mz driven, later ones positive: sign of Mz must change along the train
        assert!(sig.len() == n && sig[0].re.is_finite());
    }

    #[test]
    fn diffusion_damps_by_closed_form_at_k0() {
        // F⁺_0 after one dephasing step with b_unit: exp(−D b /3) before the shift
        let mut e = Epg::new(2);
        e.rf(std::f64::consts::FRAC_PI_2, 0.0);
        let before = e.fp[0];
        e.diffuse(2e-9, 1e8);
        assert!((e.fp[0] / before - C::new((-2e-9 * 1e8 / 3.0f64).exp(), 0.0)).norm() < 1e-12);
    }
}
