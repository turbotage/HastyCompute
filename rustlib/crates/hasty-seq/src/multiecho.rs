//! Short-readout multi-echo train for B0 / R2* / fat mapping (WP3 in `research/plan_girf_b0_qfit.md`).
//!
//! One TR: [hard RF][K out-and-back lobes, each leaving and re-entering k = 0 at rest][constant spoiler]. Lobe j plays
//! G(t) = s_j d_j g(t − t_j) with one scalar waveform g (positive lobe of area A = kmax/γ̄, then the negative one), so
//! k(t) goes 0 → s_j d_j kmax → 0 and every lobe boundary is a k = 0 crossing (an echo) with G = 0 there: the direction
//! may change freely between lobes without a slew penalty. The ADC runs through the whole train.
//!
//! Echo e (crossing e = 0..=K, at TE_e = TE₁ + e·T_lobe) owns the samples within T_lobe/2 of it: the returning half of
//! lobe e−1 and the outgoing half of lobe e, i.e. a time-symmetric echo whose highest k is reached at TE_e ± T_lobe/2.
//!
//! Flow: an out-and-back lobe has M0 = 0 but M1 = −A·T_lobe/2 along s_j d_j (integration by parts: M1 = −∫M0 dt). With
//! [`Polarity::Pairs`] (lobe pairs (d, −d) along one diameter, a new diameter per pair) every even crossing has M1 = 0
//! (first-order flow compensated) and the odd crossings carry the lobe's M1 along d; with an even K the whole train
//! has M1 = 0, so the spoiler block (and RF spoiling) sees the same moments for every direction and velocity.
//! `Same` (all lobes along d: M1 grows linearly with e, which a voxel fit cannot tell from ω) and `Independent` (a new
//! direction per lobe: a random walk of M1) are kept for comparison only.

use crate::pns::{pns, SafeHw};
use crate::system::System;
use crate::wave::{trapezoid, Grad};
use crate::{scale, Vec3, GAMMA_BAR};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LobeShape {
    /// time-optimal: back-to-back trapezoids (+A, −A)
    Trapezoid,
    /// one sine period G_p sin(2π t / T): less high-frequency content (gentler on the GIRF), ~10 % longer
    Sine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    Pairs,
    Same,
    Independent,
}

#[derive(Clone, Debug)]
pub struct EchoSpec {
    /// nominal resolution (m): kmax = 1 / (2 res)
    pub res: f64,
    /// number of lobes K (crossings 0..=K); even for `Pairs`
    pub n_lobe: usize,
    pub shape: LobeShape,
    pub polarity: Polarity,
    /// slew and amplitude fractions of the system limits used by the lobes (vector limits: rotated)
    pub slew_frac: f64,
    pub gmax_frac: f64,
    /// hard-pulse duration, s
    pub rf_dur: f64,
    /// TE shifts (s) cycled over TRs (delay before the train; TR fixed by the largest)
    pub te_shifts: Vec<f64>,
    /// spoiler moment in cycles per voxel along physical z
    pub spoil_cycles: f64,
}

impl EchoSpec {
    pub fn new(res: f64, n_lobe: usize) -> Self {
        Self { res, n_lobe, shape: LobeShape::Trapezoid, polarity: Polarity::Pairs, slew_frac: 1.0, gmax_frac: 1.0, rf_dur: 100e-6, te_shifts: vec![0.0], spoil_cycles: 2.0 }
    }
}

#[derive(Clone, Debug)]
pub struct EchoTrain {
    pub spec: EchoSpec,
    pub dt: f64,
    pub kmax: f64,
    /// scalar lobe waveform (T/m) on the raster, zero at both ends
    pub lobe: Vec<f64>,
    pub t_lobe: f64,
    /// first crossing after the RF centre (readout gradient start, no TE shift), s
    pub te1: f64,
    /// spoiler on physical z (same slew / amplitude fractions as the lobes), starts after the last lobe
    pub spoiler: Grad,
    /// RF centre → RF centre, s
    pub tr: f64,
    /// |M1| of one lobe, T·s²/m
    pub m1_lobe: f64,
}

/// out-and-back scalar lobe for k-excursion `kmax` under amplitude / slew limits (exact area on the raster)
pub fn lobe(kmax: f64, gmax: f64, smax: f64, dt: f64, shape: LobeShape) -> Vec<f64> {
    let area = kmax / GAMMA_BAR;
    match shape {
        LobeShape::Trapezoid => {
            let p = trapezoid(area, gmax, smax, dt);
            let mut v = p.clone();
            v.extend(p[1..].iter().map(|x| -x));
            v
        }
        LobeShape::Sine => {
            // slew-limited: T = π √(2A / S); amplitude-limited: T = π A / G
            let t = (std::f64::consts::PI * (2.0 * area / smax).sqrt()).max(std::f64::consts::PI * area / gmax);
            let mut n = (t / dt - 1e-9).ceil() as usize;
            n += n % 2;
            let s: Vec<f64> = (0..=n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin()).collect();
            // piecewise-linear area of the positive half = dt Σ interior samples (zero end points)
            let a_pl: f64 = s[1..n / 2].iter().sum::<f64>() * dt;
            let mut v: Vec<f64> = s.iter().map(|x| x * area / a_pl).collect();
            v[0] = 0.0;
            v[n / 2] = 0.0;
            v[n] = 0.0;
            v
        }
    }
}

impl EchoTrain {
    pub fn design(spec: &EchoSpec, sys: &System) -> EchoTrain {
        assert!(spec.polarity != Polarity::Pairs || spec.n_lobe % 2 == 0, "Pairs needs an even number of lobes");
        let dt = sys.grad_raster;
        let kmax = 0.5 / spec.res;
        let lobe = lobe(kmax, sys.gmax * spec.gmax_frac, sys.smax * spec.slew_frac, dt, spec.shape);
        let t_lobe = (lobe.len() - 1) as f64 * dt;
        let te1 = sys.ceil_raster(0.5 * spec.rf_dur + sys.rf_ringdown.max(sys.adc_dead));
        let m0 = spec.spoil_cycles / spec.res / GAMMA_BAR;
        let sp = trapezoid(m0, sys.gmax * spec.gmax_frac, sys.smax * spec.slew_frac, dt);
        let spoiler = Grad::from_axis(2, &sp, dt, 0.0);
        let shift = spec.te_shifts.iter().cloned().fold(0.0, f64::max);
        let tr = sys.ceil_raster(sys.rf_dead + 0.5 * spec.rf_dur + te1 + shift + spec.n_lobe as f64 * t_lobe + spoiler.duration());
        // M1 of one lobe = −∫ M0 dt over it
        let mut m0c = 0.0;
        let mut m1 = 0.0;
        for w in lobe.windows(2) {
            let m0n = m0c + 0.5 * (w[0] + w[1]) * dt;
            m1 -= 0.5 * (m0c + m0n) * dt;
            m0c = m0n;
        }
        EchoTrain { spec: spec.clone(), dt, kmax, lobe, t_lobe, te1, spoiler, tr, m1_lobe: m1.abs() }
    }

    pub fn n_echo(&self) -> usize {
        self.spec.n_lobe + 1
    }

    /// echo (crossing) times after the RF centre for TE shift `shift`
    pub fn te(&self, shift: f64) -> Vec<f64> {
        (0..self.n_echo()).map(|e| self.te1 + shift + e as f64 * self.t_lobe).collect()
    }

    /// ADC time owned by each echo (half a lobe on either side)
    pub fn adc_time(&self) -> Vec<f64> {
        (0..self.n_echo()).map(|e| if e == 0 || e == self.spec.n_lobe { 0.5 * self.t_lobe } else { self.t_lobe }).collect()
    }

    /// signed lobe directions of one TR from its pair/lobe directions `dirs` (one per pair for `Pairs`, one per TR for
    /// `Same`, one per lobe for `Independent`)
    pub fn lobe_dirs(&self, dirs: &[Vec3]) -> Vec<Vec3> {
        let k = self.spec.n_lobe;
        match self.spec.polarity {
            Polarity::Pairs => (0..k).map(|j| scale(&dirs[j / 2], if j % 2 == 0 { 1.0 } else { -1.0 })).collect(),
            Polarity::Same => vec![dirs[0]; k],
            Polarity::Independent => dirs[..k].to_vec(),
        }
    }

    /// number of directions one TR consumes
    pub fn dirs_per_tr(&self) -> usize {
        match self.spec.polarity {
            Polarity::Pairs => self.spec.n_lobe / 2,
            Polarity::Same => 1,
            Polarity::Independent => self.spec.n_lobe,
        }
    }

    /// first moment (T·s²/m) at each crossing, accumulated from the train start
    pub fn m1_at_echoes(&self, dirs: &[Vec3]) -> Vec<Vec3> {
        let ld = self.lobe_dirs(dirs);
        let mut acc = [0.0; 3];
        let mut out = vec![acc];
        for d in &ld {
            acc = [0, 1, 2].map(|q| acc[q] - self.m1_lobe * d[q]);
            out.push(acc);
        }
        out
    }

    /// the readout train (physical axes), t0 = the first crossing
    pub fn readout(&self, dirs: &[Vec3], t0: f64) -> Grad {
        let mut g = vec![[0.0; 3]];
        for d in self.lobe_dirs(dirs) {
            g.extend(self.lobe[1..].iter().map(|x| scale(&d, *x)));
        }
        Grad { dt: self.dt, t0, g }
    }

    /// one whole TR on the raster, t0 = RF start (gradients zero during RF), with TE shift `shift`
    pub fn tr_waveform(&self, dirs: &[Vec3], shift: f64, sys: &System) -> Grad {
        let n = (self.tr / self.dt).round() as usize + 1;
        let mut g = vec![[0.0; 3]; n];
        let t_rf = sys.rf_dead + 0.5 * self.spec.rf_dur; // RF centre after the TR start
        let i0 = ((t_rf + self.te1 + shift) / self.dt).round() as usize;
        let ro = self.readout(dirs, 0.0);
        for (i, v) in ro.g.iter().enumerate() {
            g[i0 + i] = *v;
        }
        let i1 = i0 + ro.len() - 1;
        for (i, v) in self.spoiler.g.iter().enumerate() {
            g[(i1 + i).min(n - 1)] = *v;
        }
        Grad { dt: self.dt, t0: 0.0, g }
    }
}

/// 2D multidimensional golden-means directions (Chan et al., MRM 2009) for i = i0..i0+n:
/// (cos θ, φ) = (2 frac(i φ₁) − 1, 2π frac(i φ₂)), φ₁ = 0.4656, φ₂ = 0.6823; any contiguous run is spread over the sphere
pub fn golden_dirs(i0: usize, n: usize) -> Vec<Vec3> {
    let (g1, g2) = (0.465_571_231_876_768_0, 0.682_327_803_828_019_3);
    (i0..i0 + n)
        .map(|i| {
            let z = 2.0 * (i as f64 * g1).fract() - 1.0;
            let ph = 2.0 * std::f64::consts::PI * (i as f64 * g2).fract();
            let r = (1.0 - z * z).max(0.0).sqrt();
            [r * ph.cos(), r * ph.sin(), z]
        })
        .collect()
}

/// PNS of `n_tr` consecutive TRs (golden-means directions, TE shifts cycled): max of the combined SAFE value over
/// the last half (steady state), %; and the same with every lobe on one physical axis (worst case per axis)
pub fn train_pns(tr: &EchoTrain, sys: &System, hw: &SafeHw, n_tr: usize) -> (f64, [f64; 3]) {
    let dpt = tr.dirs_per_tr();
    let shifts = &tr.spec.te_shifts;
    let concat = |dirs_of: &dyn Fn(usize) -> Vec<Vec3>| -> f64 {
        let mut g: Vec<Vec3> = Vec::new();
        for t in 0..n_tr {
            let w = tr.tr_waveform(&dirs_of(t), shifts[t % shifts.len()], sys);
            if g.is_empty() {
                g = w.g;
            } else {
                g.extend_from_slice(&w.g[1..]);
            }
        }
        let p = pns(&Grad { dt: tr.dt, t0: 0.0, g }, hw);
        let half = p.pad + p.norm.len().saturating_sub(p.pad) / 2;
        p.norm[half.min(p.norm.len() - 1)..].iter().cloned().fold(0.0, f64::max)
    };
    let real = concat(&|t| golden_dirs(t * dpt, dpt.max(1)));
    let axis = [0, 1, 2].map(|a| {
        let mut e = [0.0; 3];
        e[a] = 1.0;
        concat(&|_| vec![e; dpt.max(1)])
    });
    (real, axis)
}

/// largest slew fraction (bisection on [lo, 1]) whose train PNS over the golden-means directions stays ≤ `limit` %
pub fn fit_slew_to_pns(spec: &EchoSpec, sys: &System, hw: &SafeHw, limit: f64, n_tr: usize) -> EchoTrain {
    let at = |f: f64| {
        let mut s = spec.clone();
        s.slew_frac = f;
        let t = EchoTrain::design(&s, sys);
        let p = train_pns(&t, sys, hw, n_tr).0;
        (t, p)
    };
    let (t1, p1) = at(1.0);
    if p1 <= limit {
        return t1;
    }
    let (mut lo, mut hi) = (0.05, 1.0);
    for _ in 0..12 {
        let m = 0.5 * (lo + hi);
        if at(m).1 <= limit {
            lo = m;
        } else {
            hi = m;
        }
    }
    at(lo).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lobes_return_to_k0_with_exact_excursion_and_limits() {
        let sys = System::prisma_xr();
        for shape in [LobeShape::Trapezoid, LobeShape::Sine] {
            let mut spec = EchoSpec::new(2e-3, 6);
            spec.shape = shape;
            let tr = EchoTrain::design(&spec, &sys);
            let dirs = golden_dirs(0, tr.dirs_per_tr());
            let ro = tr.readout(&dirs, 0.0);
            let lim = ro.limits();
            assert!(lim.ok_vec(&sys), "{shape:?}: {lim:?}");
            let k = ro.moments(0.0).k();
            let nl = tr.lobe.len() - 1;
            for e in 0..=spec.n_lobe {
                let kn = crate::norm(&k[e * nl]);
                assert!(kn < 1e-9, "{shape:?}: crossing {e} |k| = {kn}");
            }
            // the turn-around point of each lobe is at kmax
            let kmid = crate::norm(&k[nl / 2]);
            assert!((kmid - tr.kmax).abs() < 1e-9 * tr.kmax, "{shape:?}: {kmid} vs {}", tr.kmax);
            assert!(k.iter().all(|v| crate::norm(v) <= tr.kmax * (1.0 + 1e-9)), "{shape:?}: k beyond kmax");
        }
    }

    #[test]
    fn pairs_are_flow_compensated_at_even_echoes() {
        let sys = System::prisma_xr();
        let tr = EchoTrain::design(&EchoSpec::new(2e-3, 8), &sys);
        let dirs = golden_dirs(5, tr.dirs_per_tr());
        let ro = tr.readout(&dirs, 0.0);
        let m = ro.moments(0.0);
        let nl = tr.lobe.len() - 1;
        let pred = tr.m1_at_echoes(&dirs);
        for e in 0..tr.n_echo() {
            let m1 = m.m1[e * nl];
            let err = crate::norm(&crate::sub(&m1, &pred[e]));
            assert!(err < 1e-6 * tr.m1_lobe, "echo {e}: {m1:?} vs {:?}", pred[e]);
            if e % 2 == 0 {
                assert!(crate::norm(&m1) < 1e-6 * tr.m1_lobe, "even echo {e} not compensated: {m1:?}");
            } else {
                assert!((crate::norm(&m1) - tr.m1_lobe).abs() < 1e-6 * tr.m1_lobe);
            }
        }
    }
}
