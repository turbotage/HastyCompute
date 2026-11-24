//! CSF inflow (spin-history) signal simulator for a multi-slice GRE-EPI fMRI acquisition.
//!
//! Reference: Söderström et al., "Quantifying Cardiac, Respiratory, and Low Frequency Components of CSF Motion From fMRI
//! Inflow Effects", MRM 2026. The signal of the bottom slice of the fMRI stack, where CSF flows through it along the slice
//! normal, is a magnitude effect of the spin history: fresh (unsaturated) CSF enters the slice, saturated CSF leaves it.
//!
//! Model (this module is *independent* of the k-space direct-sum simulator; pure CPU, no GPU, no k-space):
//! * 1D plug flow along z (the slice normal), z = 0 at the centre of the bottom slice, +z towards the head. Displacement
//!   ζ(t) = ∫ v dt moves every element, v(t) a sum of cardiac / respiratory / slow components ([`Component`], [`Flow`]).
//! * The magnetisation M_z(label) of a continuum of elements (label = position at t = 0, Lagrangian) is advanced event by
//!   event: T1 recovery and Gaussian dispersion for the time since the last event, then the slice-selective excitation of
//!   that slice: M_z → M_z cos α(z − z_slice), α from the Bloch profile of a real sinc pulse ([`SlabPulse::profile`]), so
//!   slice gaps, slice-profile tails and cross-talk between neighbouring slices (interleaved order) come out of the model.
//! * Perfect spoiling: the transverse magnetisation is destroyed at every excitation.
//! * Signal of the bottom slice at excitation n: S_n = e^{−TE/T2} ∫ sin α(z − z_0) M_z^{pre}(z) dz.
//! * An optional cross-section profile A(z) (funnel) makes the plug move by volume flux, v(z) = Q/A(z). Diffusion then
//!   uses the reference-area width (exact only for A = const).
//!
//! Units SI: m, s, rad. Diffusion/dispersion coefficient in m²/s (6 cm²/min = 1e-5 m²/s).
//!
//! [`lagrangian_signal`] is an independent Monte-Carlo implementation (random-walk dispersion, per-element positions) used
//! as a cross-check of the continuum engine. [`fit`] holds a minimal estimator in the spirit of the paper's Eq. 2.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::physio::Ecg;
use crate::rf::{SlabProfile, SlabPulse};

// ---------------------------------------------------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------------------------------------------------

/// xorshift64* generator (deterministic from a seed)
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1)
    }
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
    }
}

/// Frequency an oscillation of `f` Hz is aliased to when sampled at `fs` Hz.
pub fn alias_freq(f: f64, fs: f64) -> f64 {
    (f - (f / fs).round() * fs).abs()
}

/// Amplitude of the sinusoid at `f` Hz in the (possibly non-uniformly sampled) series s(t), mean removed first.
/// Equals A for s = A cos(2π f t + φ) over an integer number of periods.
pub fn tone_amplitude(t: &[f64], s: &[f64], f: f64) -> f64 {
    let n = s.len() as f64;
    let m = s.iter().sum::<f64>() / n;
    let (mut re, mut im) = (0.0, 0.0);
    for (&ti, &si) in t.iter().zip(s) {
        let ph = 2.0 * PI * f * ti;
        re += (si - m) * ph.cos();
        im += (si - m) * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / n
}

/// Gaussian noise of standard deviation `sigma` added in place.
pub fn add_noise(s: &mut [f64], sigma: f64, seed: u64) {
    let mut r = Rng::new(seed);
    for v in s {
        *v += sigma * r.normal();
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// acquisition, medium
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SliceOrder {
    Sequential,
    /// slices 1, 3, 5, … then 2, 4, 6, … (1-based), the usual fMRI interleave
    Interleaved,
}

impl SliceOrder {
    /// excitation position (0-based, within the TR) of slice `s` of `n`
    pub fn position(&self, s: usize, n: usize) -> usize {
        match self {
            SliceOrder::Sequential => s,
            SliceOrder::Interleaved => {
                let n_even = (n + 1) / 2;
                if s % 2 == 0 { s / 2 } else { n_even + s / 2 }
            }
        }
    }
}

/// Acquisition: the multi-slice stack of which only the first `n_sim` slices (counted from the bottom) are simulated;
/// the timing uses all `n_slices`.
#[derive(Clone, Debug)]
pub struct Acq {
    pub tr: f64,
    pub te: f64,
    /// nominal flip angle, rad
    pub flip: f64,
    pub thickness: f64,
    pub gap: f64,
    pub n_slices: usize,
    pub n_sim: usize,
    pub order: SliceOrder,
    /// TRs before the first recorded one (the scanner's dummy scans, plus settling)
    pub n_dummy: usize,
    pub n_scan: usize,
    pub pulse_duration: f64,
    pub pulse_tbw: f64,
    /// profile support on each side of a slice centre, in slice thicknesses
    pub extent_factor: f64,
    /// label grid spacing, m
    pub dz: f64,
}

impl Acq {
    /// The in-vivo fMRI of the paper: TR 2 s, TE 30 ms, 80°, 37 interleaved slices of 3.4 mm with a 0.5 mm gap, 6 min.
    pub fn paper() -> Self {
        Self {
            tr: 2.0,
            te: 0.030,
            flip: 80f64.to_radians(),
            thickness: 3.4e-3,
            gap: 0.5e-3,
            n_slices: 37,
            n_sim: 6,
            order: SliceOrder::Interleaved,
            n_dummy: 10,
            n_scan: 180,
            pulse_duration: 3e-3,
            pulse_tbw: 4.0,
            extent_factor: 2.5,
            dz: 80e-6,
        }
    }
    pub fn slice_centre(&self, s: usize) -> f64 {
        s as f64 * (self.thickness + self.gap)
    }
    pub fn extent(&self) -> f64 {
        self.extent_factor * self.thickness
    }
    pub fn n_tr(&self) -> usize {
        self.n_dummy + self.n_scan
    }
    pub fn t_end(&self) -> f64 {
        self.n_tr() as f64 * self.tr
    }
}

#[derive(Clone, Debug)]
pub struct Medium {
    pub t1: f64,
    pub t2: f64,
    /// dispersion coefficient, m²/s
    pub diffusion: f64,
}

impl Medium {
    /// CSF of the paper: T1 4.738 s (cohort mean), T2 1.6 s, D = 6 cm²/min
    pub fn csf() -> Self {
        Self { t1: 4.738, t2: 1.6, diffusion: 6e-4 / 60.0 }
    }
}

/// Relative cross-section A(z)/A(0), piecewise linear through `points` (z increasing, must contain z = 0 in its range).
#[derive(Clone, Debug)]
pub struct Area {
    z0: f64,
    h: f64,
    /// V(z) = ∫_0^z A dz on the dense grid
    v: Vec<f64>,
    pts: Vec<(f64, f64)>,
}

impl Area {
    pub fn new(points: &[(f64, f64)]) -> Self {
        assert!(points.len() >= 2 && points.windows(2).all(|w| w[1].0 > w[0].0), "area table must be increasing in z");
        assert!(points[0].0 <= 0.0 && points[points.len() - 1].0 >= 0.0, "area table must contain z = 0");
        let (z0, z1) = (points[0].0, points[points.len() - 1].0);
        let h = 10e-6;
        let n = ((z1 - z0) / h).ceil() as usize + 1;
        let mut a = Self { z0, h, v: vec![0.0; n], pts: points.to_vec() };
        let mut acc = 0.0;
        let mut v = vec![0.0; n];
        for i in 1..n {
            let (za, zb) = (z0 + (i - 1) as f64 * h, z0 + i as f64 * h);
            acc += 0.5 * (a.a_at(za) + a.a_at(zb)) * h;
            v[i] = acc;
        }
        let i0 = ((0.0 - z0) / h).round() as usize;
        let shift = v[i0.min(n - 1)];
        a.v = v.iter().map(|x| x - shift).collect();
        a
    }
    pub fn a_at(&self, z: f64) -> f64 {
        let p = &self.pts;
        if z <= p[0].0 {
            return p[0].1;
        }
        if z >= p[p.len() - 1].0 {
            return p[p.len() - 1].1;
        }
        let i = p.partition_point(|q| q.0 <= z) - 1;
        let f = (z - p[i].0) / (p[i + 1].0 - p[i].0);
        p[i].1 * (1.0 - f) + p[i + 1].1 * f
    }
    pub fn v_of_z(&self, z: f64) -> f64 {
        let zl = self.z0 + (self.v.len() - 1) as f64 * self.h;
        if z <= self.z0 {
            return self.v[0] + self.a_at(self.z0) * (z - self.z0);
        }
        if z >= zl {
            return self.v[self.v.len() - 1] + self.a_at(zl) * (z - zl);
        }
        let f = (z - self.z0) / self.h;
        let i = f as usize;
        let a = f - i as f64;
        self.v[i] * (1.0 - a) + self.v[i + 1] * a
    }
    pub fn z_of_v(&self, v: f64) -> f64 {
        let n = self.v.len();
        let zl = self.z0 + (n - 1) as f64 * self.h;
        if v <= self.v[0] {
            return self.z0 + (v - self.v[0]) / self.a_at(self.z0);
        }
        if v >= self.v[n - 1] {
            return zl + (v - self.v[n - 1]) / self.a_at(zl);
        }
        let i = self.v.partition_point(|&x| x <= v) - 1;
        let f = (v - self.v[i]) / (self.v[i + 1] - self.v[i]);
        self.z0 + (i as f64 + f) * self.h
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// flow
// ---------------------------------------------------------------------------------------------------------------------

/// Plug displacement ζ(t) (m, positive towards +z) and velocity v(t) on a uniform time grid starting at t = 0.
#[derive(Clone, Debug)]
pub struct Flow {
    pub dt: f64,
    pub zeta: Vec<f64>,
    pub vel: Vec<f64>,
}

impl Flow {
    /// ζ = ∫ v dt (trapezoid on the grid), ζ(0) = 0.
    pub fn from_vel(t_end: f64, dt: f64, f: impl Fn(f64) -> f64) -> Self {
        let n = (t_end / dt).ceil() as usize + 2;
        let vel: Vec<f64> = (0..n).map(|k| f(k as f64 * dt)).collect();
        let mut zeta = vec![0.0; n];
        for k in 1..n {
            zeta[k] = zeta[k - 1] + 0.5 * (vel[k - 1] + vel[k]) * dt;
        }
        Self { dt, zeta, vel }
    }
    /// from a displacement function; velocity by central differences
    pub fn from_zeta(t_end: f64, dt: f64, g: impl Fn(f64) -> f64) -> Self {
        let n = (t_end / dt).ceil() as usize + 2;
        let zeta: Vec<f64> = (0..n).map(|k| g(k as f64 * dt)).collect();
        let vel = (0..n).map(|k| (zeta[(k + 1).min(n - 1)] - zeta[k.saturating_sub(1)]) / (((k + 1).min(n - 1) - k.saturating_sub(1)) as f64 * dt)).collect();
        Self { dt, zeta, vel }
    }
    pub fn t_end(&self) -> f64 {
        (self.zeta.len() - 1) as f64 * self.dt
    }
    fn lerp(&self, a: &[f64], t: f64) -> f64 {
        let f = (t / self.dt).clamp(0.0, (a.len() - 1) as f64);
        let i = (f as usize).min(a.len() - 2);
        let w = f - i as f64;
        a[i] * (1.0 - w) + a[i + 1] * w
    }
    pub fn zeta_at(&self, t: f64) -> f64 {
        self.lerp(&self.zeta, t)
    }
    pub fn v_at(&self, t: f64) -> f64 {
        self.lerp(&self.vel, t)
    }
    pub fn zeta_range(&self) -> (f64, f64) {
        (self.zeta.iter().cloned().fold(f64::MAX, f64::min), self.zeta.iter().cloned().fold(f64::MIN, f64::max))
    }
}

/// A monotonically increasing phase φ(t) (rad) on a uniform grid.
#[derive(Clone, Debug)]
pub struct PhaseClock {
    pub dt: f64,
    pub ph: Vec<f64>,
}

impl PhaseClock {
    pub fn at(&self, t: f64) -> f64 {
        let f = (t / self.dt).clamp(0.0, (self.ph.len() - 1) as f64);
        let i = (f as usize).min(self.ph.len() - 2);
        let w = f - i as f64;
        self.ph[i] * (1.0 - w) + self.ph[i + 1] * w
    }
    /// φ'(t) by a central difference over ±`h` seconds
    pub fn rate(&self, t: f64, h: f64) -> f64 {
        (self.at(t + h) - self.at(t - h)) / (2.0 * h)
    }
    /// Cardiac clock φ = 2π (beat + fraction of the RR interval) from the R peaks of `ecg` (peaks must cover [0, t_end]).
    pub fn cardiac(ecg: &Ecg, t_end: f64, dt: f64) -> Self {
        let n = (t_end / dt).ceil() as usize + 2;
        let mut ph = Vec::with_capacity(n);
        for k in 0..n {
            let c = ecg.at(k as f64 * dt);
            ph.push(2.0 * PI * (c.beat as f64 + c.phase));
        }
        Self { dt, ph }
    }
    /// Quasi-periodic clock of mean frequency `f0` Hz whose instantaneous frequency is f0·exp(σ x(t)), x an
    /// Ornstein–Uhlenbeck process of unit variance and correlation time `tau` s (respiration, slow oscillation).
    pub fn wandering(f0: f64, sigma: f64, tau: f64, t_end: f64, dt: f64, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = (t_end / dt).ceil() as usize + 2;
        let rho = (-dt / tau).exp();
        let mut x = rng.normal();
        let mut ph = Vec::with_capacity(n);
        let mut p = 2.0 * PI * rng.uniform();
        for _ in 0..n {
            ph.push(p);
            p += 2.0 * PI * f0 * (sigma * x).exp() * dt;
            x = rho * x + (1.0 - rho * rho).sqrt() * rng.normal();
        }
        Self { dt, ph }
    }
    /// the same clock seen `tau` seconds late (a timing error of the recorded physiology): φ'(t) = φ(t − tau)
    pub fn shifted(&self, tau: f64) -> Self {
        Self { dt: self.dt, ph: (0..self.ph.len()).map(|k| self.at(k as f64 * self.dt - tau)).collect() }
    }
    /// constant frequency clock
    pub fn constant(f: f64, phase0: f64, t_end: f64, dt: f64) -> Self {
        let n = (t_end / dt).ceil() as usize + 2;
        Self { dt, ph: (0..n).map(|k| phase0 + 2.0 * PI * f * k as f64 * dt).collect() }
    }
}

/// One velocity component v_k cos(φ_k(t) + δ_k) (+ optional higher harmonics j+2: a_j cos((j+2)φ_k + ψ_j), which make
/// the "true" waveform non-sinusoidal).
#[derive(Clone, Debug)]
pub struct Component {
    pub clock: PhaseClock,
    /// velocity amplitude of the fundamental, m/s
    pub amp: f64,
    pub delta: f64,
    pub harmonics: Vec<(f64, f64)>,
}

impl Component {
    pub fn new(clock: PhaseClock, amp: f64, delta: f64) -> Self {
        Self { clock, amp, delta, harmonics: vec![] }
    }
    pub fn v(&self, t: f64) -> f64 {
        let p = self.clock.at(t);
        let mut s = (p + self.delta).cos();
        for (j, &(a, psi)) in self.harmonics.iter().enumerate() {
            s += a * (((j + 2) as f64) * p + psi).cos();
        }
        self.amp * s
    }
}

pub fn flow_from_components(comps: &[Component], t_end: f64, dt: f64) -> Flow {
    Flow::from_vel(t_end, dt, |t| comps.iter().map(|c| c.v(t)).sum())
}

// ---------------------------------------------------------------------------------------------------------------------
// the continuum spin-history engine
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Output {
    /// time of each recorded bottom-slice excitation, s (absolute, scan start of the first dummy TR = 0)
    pub t: Vec<f64>,
    /// raw CSF signal e^{−TE/T2} ∫ sin α M_z dz, m
    pub raw: Vec<f64>,
    /// true plug displacement ζ and velocity v at those times
    pub zeta: Vec<f64>,
    pub vel: Vec<f64>,
}

impl Output {
    /// signal normalised by its mean. `static_ratio` adds a constant (static-tissue) partial-volume signal equal to that
    /// multiple of the CSF mean, which reduces the relative modulation.
    pub fn normalized(&self, static_ratio: f64) -> Vec<f64> {
        let m = self.raw.iter().sum::<f64>() / self.raw.len() as f64;
        self.raw.iter().map(|&s| (s + static_ratio * m) / (m * (1.0 + static_ratio))).collect()
    }
}

pub struct InflowSim {
    pub acq: Acq,
    pub med: Medium,
    pub area: Option<Area>,
    prof: SlabProfile,
}

impl InflowSim {
    pub fn new(acq: Acq, med: Medium) -> Self {
        assert!(acq.n_sim >= 1 && acq.n_sim <= acq.n_slices);
        let pulse = SlabPulse::new(acq.pulse_duration, acq.pulse_tbw, acq.flip, acq.thickness);
        let ext = acq.extent();
        let nz = (2.0 * ext / 20e-6).ceil() as usize | 1;
        let prof = pulse.profile(ext, nz, 0.0);
        Self { acq, med, area: None, prof }
    }
    pub fn with_area(mut self, a: Area) -> Self {
        self.area = Some(a);
        self
    }
    pub fn profile(&self) -> &SlabProfile {
        &self.prof
    }
    fn v_of_z(&self, z: f64) -> f64 {
        match &self.area {
            Some(a) => a.v_of_z(z),
            None => z,
        }
    }
    fn z_of_v(&self, v: f64) -> f64 {
        match &self.area {
            Some(a) => a.z_of_v(v),
            None => v,
        }
    }
    fn a_at(&self, z: f64) -> f64 {
        match &self.area {
            Some(a) => a.a_at(z),
            None => 1.0,
        }
    }

    /// Gaussian kernel (normalised) of variance 2 D dt on the label grid, truncated at 4σ
    fn kernel(&self, dt: f64) -> Vec<f64> {
        let sig = (2.0 * self.med.diffusion * dt).sqrt() / self.acq.dz;
        if sig < 0.05 {
            return vec![1.0];
        }
        let hw = (4.0 * sig).ceil() as isize;
        let mut k: Vec<f64> = (-hw..=hw).map(|i| (-0.5 * (i as f64 / sig).powi(2)).exp()).collect();
        let s: f64 = k.iter().sum();
        k.iter_mut().for_each(|x| *x /= s);
        k
    }

    pub fn run(&self, flow: &Flow) -> Output {
        let acq = &self.acq;
        assert!(flow.t_end() >= acq.t_end(), "flow shorter than the acquisition ({} < {})", flow.t_end(), acq.t_end());
        let ext = acq.extent();
        let (zmin, zmax) = flow.zeta_range();
        let lo_lab = -ext - 2e-3;
        let hi_lab = acq.slice_centre(acq.n_sim - 1) + ext + 2e-3;
        let pad = 5.0 * (2.0 * self.med.diffusion * acq.tr).sqrt();
        let v_lo = self.v_of_z(lo_lab) - zmax - pad;
        let v_hi = self.v_of_z(hi_lab) - zmin + pad;
        let dv = acq.dz;
        let n = ((v_hi - v_lo) / dv).ceil() as usize + 1;

        let mut mz = vec![1.0f64; n];
        let mut tmp = vec![0.0f64; n];
        let mut kcache: HashMap<u64, Vec<f64>> = HashMap::new();

        let mut slice_at: Vec<Option<usize>> = vec![None; acq.n_slices];
        for s in 0..acq.n_sim {
            slice_at[acq.order.position(s, acq.n_slices)] = Some(s);
        }
        let dt_ev = acq.tr / acq.n_slices as f64;
        let te_fac = (-acq.te / self.med.t2).exp();

        let mut out = Output { t: vec![], raw: vec![], zeta: vec![], vel: vec![] };
        let mut t_last = 0.0;
        for tr_i in 0..acq.n_tr() {
            for pos in 0..acq.n_slices {
                let Some(s) = slice_at[pos] else { continue };
                let t = tr_i as f64 * acq.tr + pos as f64 * dt_ev;
                let dt = t - t_last;
                t_last = t;
                if dt > 0.0 {
                    let e = (-dt / self.med.t1).exp();
                    mz.iter_mut().for_each(|m| *m = 1.0 + (*m - 1.0) * e);
                    if self.med.diffusion > 0.0 {
                        let key = (dt * 1e9).round() as u64;
                        let k = kcache.entry(key).or_insert_with(|| self.kernel(dt));
                        if k.len() > 1 {
                            let hw = (k.len() / 2) as isize;
                            for i in 0..n as isize {
                                let mut acc = 0.0;
                                for (j, w) in k.iter().enumerate() {
                                    let idx = (i + j as isize - hw).clamp(0, n as isize - 1) as usize;
                                    acc += w * mz[idx];
                                }
                                tmp[i as usize] = acc;
                            }
                            std::mem::swap(&mut mz, &mut tmp);
                        }
                    }
                }
                let zeta = flow.zeta_at(t);
                let zc = acq.slice_centre(s);
                let record = s == 0 && tr_i >= acq.n_dummy;
                let mut sig = 0.0;
                for i in 0..n {
                    let z = self.z_of_v(v_lo + i as f64 * dv + zeta);
                    let u = z - zc;
                    if u.abs() >= ext {
                        continue;
                    }
                    let a = self.prof.alpha_at(u);
                    if a <= 0.0 {
                        continue;
                    }
                    if record {
                        sig += a.sin() * mz[i] * dv / self.a_at(z);
                    }
                    mz[i] *= a.cos();
                }
                if record {
                    out.t.push(t);
                    out.raw.push(sig * te_fac);
                    out.zeta.push(zeta);
                    out.vel.push(flow.v_at(t));
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// independent Monte-Carlo cross-check
// ---------------------------------------------------------------------------------------------------------------------

/// Per-element random-walk implementation of the same physics (constant area): `n_el` elements, uniformly placed over
/// the span the flow can reach, each with its own position (plug displacement + Brownian steps of variance 2 D dt) and
/// its own M_z. Returns the raw signal of the bottom slice (same units as [`Output::raw`]).
pub fn lagrangian_signal(acq: &Acq, med: &Medium, flow: &Flow, n_el: usize, seed: u64) -> Vec<f64> {
    lagrangian_signal_with(acq, med, flow, n_el, seed, None)
}

/// Velocity multipliers f_i of `n` elements spread uniformly over a circular cross-section for the power-law profile
/// u = (1 − ρ^m)(m + 2)/m (m = 2: laminar/Poiseuille, large m: blunt), normalised to mean 1 so that the mean velocity is
/// the prescribed one. Element i then moves by f_i ζ(t).
pub fn radial_factors(m: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| (1.0 - rng.uniform().powf(m / 2.0)) * (m + 2.0) / m).collect()
}

/// [`lagrangian_signal`] with a per-element velocity multiplier (`fac`, see [`radial_factors`]); the in-plane signal is
/// the mean over all elements, i.e. over the whole cross-section.
pub fn lagrangian_signal_with(acq: &Acq, med: &Medium, flow: &Flow, n_el: usize, seed: u64, fac: Option<&[f64]>) -> Vec<f64> {
    let pulse = SlabPulse::new(acq.pulse_duration, acq.pulse_tbw, acq.flip, acq.thickness);
    let ext = acq.extent();
    let prof = pulse.profile(ext, (2.0 * ext / 20e-6).ceil() as usize | 1, 0.0);
    let (zmin, zmax) = flow.zeta_range();
    let fmax = fac.map_or(1.0, |f| f.iter().cloned().fold(1.0, f64::max));
    let pad = 6.0 * (2.0 * med.diffusion * acq.tr).sqrt() + fmax * (zmax - zmin);
    // periodic lab domain [lo, hi): an element leaving one end re-enters at the other fully relaxed (it has been far from
    // every slice), which keeps the element density uniform over the whole scan
    let lo = -ext - 2e-3 - pad;
    let hi = acq.slice_centre(acq.n_sim - 1) + ext + 2e-3 + pad;
    let len = hi - lo;
    let mut rng = Rng::new(seed);
    // z0: initial lab position; b: accumulated Brownian displacement (plus wrap shifts)
    let z0: Vec<f64> = (0..n_el).map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n_el as f64).collect();
    let mut b = vec![0.0f64; n_el];
    let mut mz = vec![1.0f64; n_el];
    let mut tl = vec![0.0f64; n_el]; // last relaxation time per element
    let mut slice_at: Vec<Option<usize>> = vec![None; acq.n_slices];
    for s in 0..acq.n_sim {
        slice_at[acq.order.position(s, acq.n_slices)] = Some(s);
    }
    let dt_ev = acq.tr / acq.n_slices as f64;
    let dz_el = (hi - lo) / n_el as f64;
    let te_fac = (-acq.te / med.t2).exp();
    let mut t_last = 0.0;
    let mut out = vec![];
    for tr_i in 0..acq.n_tr() {
        for pos in 0..acq.n_slices {
            let Some(s) = slice_at[pos] else { continue };
            let t = tr_i as f64 * acq.tr + pos as f64 * dt_ev;
            let dt = t - t_last;
            t_last = t;
            let step = (2.0 * med.diffusion * dt).sqrt();
            let zeta = flow.zeta_at(t);
            let zc = acq.slice_centre(s);
            let record = s == 0 && tr_i >= acq.n_dummy;
            let mut sig = 0.0;
            for i in 0..n_el {
                if step > 0.0 {
                    b[i] += step * rng.normal();
                }
                let mut z = z0[i] + zeta * fac.map_or(1.0, |f| f[i]) + b[i];
                while z < lo || z >= hi {
                    let sh = if z < lo { len } else { -len };
                    b[i] += sh;
                    z += sh;
                    mz[i] = 1.0;
                    tl[i] = t;
                }
                let u = z - zc;
                if u.abs() >= ext {
                    continue;
                }
                let a = prof.alpha_at(u);
                if a <= 0.0 {
                    continue;
                }
                let e = (-(t - tl[i]) / med.t1).exp();
                mz[i] = 1.0 + (mz[i] - 1.0) * e;
                tl[i] = t;
                if record {
                    sig += a.sin() * mz[i] * dz_el;
                }
                mz[i] *= a.cos();
            }
            if record {
                out.push(sig * te_fac);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------
// estimator (the paper's Eq. 2, with Levenberg–Marquardt instead of particle swarm)
// ---------------------------------------------------------------------------------------------------------------------

pub mod fit {
    use super::*;

    /// Quasi-stationary model of the paper: ζ(t) = Σ_k v_k sin(φ_k + δ_k) / φ_k'(t), with p = [v_0.., δ_0..] (v in m/s).
    pub fn quasi_flow(clocks: &[PhaseClock], p: &[f64], t_end: f64, dt: f64) -> Flow {
        let nc = clocks.len();
        Flow::from_zeta(t_end, dt, |t| {
            (0..nc)
                .map(|k| {
                    let w = clocks[k].rate(t, 0.1).max(0.1);
                    p[k] * (clocks[k].at(t) + p[nc + k]).sin() / w
                })
                .sum()
        })
    }

    /// Dense Levenberg–Marquardt with forward finite differences. Returns (params, sum of squares).
    pub fn lm(f: &mut dyn FnMut(&[f64]) -> Vec<f64>, p0: &[f64], steps: &[f64], iters: usize) -> (Vec<f64>, f64) {
        let np = p0.len();
        let mut p = p0.to_vec();
        let mut r = f(&p);
        let mut cost: f64 = r.iter().map(|x| x * x).sum();
        let mut lambda = 1e-2;
        for _ in 0..iters {
            let mut jac = vec![vec![0.0; r.len()]; np];
            for j in 0..np {
                let mut q = p.clone();
                q[j] += steps[j];
                let rq = f(&q);
                for i in 0..r.len() {
                    jac[j][i] = (rq[i] - r[i]) / steps[j];
                }
            }
            let mut a = vec![vec![0.0; np]; np];
            let mut g = vec![0.0; np];
            for i in 0..np {
                for j in 0..np {
                    a[i][j] = jac[i].iter().zip(&jac[j]).map(|(x, y)| x * y).sum();
                }
                g[i] = jac[i].iter().zip(&r).map(|(x, y)| x * y).sum();
            }
            let mut improved = false;
            for _ in 0..6 {
                let mut m = a.clone();
                for i in 0..np {
                    m[i][i] += lambda * (a[i][i] + 1e-12);
                }
                let Some(dx) = solve(&m, &g.iter().map(|x| -x).collect::<Vec<_>>()) else {
                    lambda *= 10.0;
                    continue;
                };
                let q: Vec<f64> = p.iter().zip(&dx).map(|(a, b)| a + b).collect();
                let rq = f(&q);
                let cq: f64 = rq.iter().map(|x| x * x).sum();
                if cq < cost {
                    let rel = (cost - cq) / cost.max(1e-300);
                    p = q;
                    r = rq;
                    cost = cq;
                    lambda = (lambda * 0.3).max(1e-6);
                    improved = rel > 1e-6;
                    break;
                }
                lambda *= 5.0;
            }
            if !improved {
                break;
            }
        }
        (p, cost)
    }

    fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
        let n = b.len();
        let mut m: Vec<Vec<f64>> = (0..n).map(|i| { let mut r = a[i].clone(); r.push(b[i]); r }).collect();
        for c in 0..n {
            let piv = (c..n).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap())?;
            if m[piv][c].abs() < 1e-300 {
                return None;
            }
            m.swap(c, piv);
            for r in c + 1..n {
                let f = m[r][c] / m[c][c];
                for k in c..=n {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = (m[i][n] - (i + 1..n).map(|k| m[i][k] * x[k]).sum::<f64>()) / m[i][i];
        }
        Some(x)
    }

    #[derive(Clone, Debug)]
    pub struct FitResult {
        /// velocity amplitudes, m/s (sign absorbed so that v ≥ 0)
        pub v: Vec<f64>,
        pub delta: Vec<f64>,
        pub r2: f64,
        pub model: Vec<f64>,
    }

    /// Joint LM refinement only, from the start point `p0` = [v_0.., δ_0..] (v in m/s). Used to ask how a fit moves when
    /// the estimator's model or the data are perturbed, starting from the unperturbed solution.
    pub fn fit_from(sim: &InflowSim, clocks: &[PhaseClock], data: &[f64], p0: &[f64], lm_iters: usize) -> FitResult {
        let nc = clocks.len();
        let t_end = sim.acq.t_end();
        let model = |p: &[f64]| sim.run(&quasi_flow(clocks, p, t_end, 2e-3)).normalized(0.0);
        let steps: Vec<f64> = (0..2 * nc).map(|i| if i < nc { 0.05e-3 } else { 0.02 }).collect();
        let mut f = |x: &[f64]| -> Vec<f64> { model(x).iter().zip(data).map(|(a, b)| a - b).collect() };
        let (x, _) = lm(&mut f, p0, &steps, lm_iters);
        let mut v: Vec<f64> = x[..nc].to_vec();
        let mut delta: Vec<f64> = x[nc..].to_vec();
        for k in 0..nc {
            if v[k] < 0.0 {
                v[k] = -v[k];
                delta[k] += PI;
            }
            delta[k] = delta[k].rem_euclid(2.0 * PI);
        }
        let m = model(&x);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let ss_res: f64 = m.iter().zip(data).map(|(a, b)| (a - b).powi(2)).sum();
        let ss_tot: f64 = data.iter().map(|b| (b - mean).powi(2)).sum();
        FitResult { v, delta, r2: 1.0 - ss_res / ss_tot, model: m }
    }

    /// Fit amplitudes and phase shifts of every clock to the normalised signal `data` (one value per recorded TR of the
    /// simulator's bottom slice). Staged: component 0 alone from four phase starts, then component 1 added, …, then a joint
    /// refinement. `vmax` bounds nothing but sets the finite-difference scale (m/s).
    pub fn fit_quasi(sim: &InflowSim, clocks: &[PhaseClock], data: &[f64], v_start: &[f64], lm_iters: usize, log: &mut dyn FnMut(&str)) -> FitResult {
        let nc = clocks.len();
        let dt = 2e-3;
        let t_end = sim.acq.t_end();
        let model = |p: &[f64]| -> Vec<f64> {
            let flow = quasi_flow(clocks, p, t_end, dt);
            sim.run(&flow).normalized(0.0)
        };
        let mut p = vec![0.0; 2 * nc];
        for stage in 0..nc {
            // free parameters: v_0..=v_stage and δ_0..=δ_stage; earlier ones start from their fitted values
            let free: Vec<usize> = (0..=stage).map(|k| k).chain((0..=stage).map(|k| nc + k)).collect();
            let mut best: Option<(Vec<f64>, f64)> = None;
            let starts: Vec<f64> = if stage == 0 { vec![0.0, PI / 2.0, PI, 1.5 * PI] } else { vec![0.0, PI / 2.0, PI, 1.5 * PI] };
            for d0 in starts {
                let mut q = p.clone();
                q[stage] = v_start[stage];
                q[nc + stage] = d0;
                let q0: Vec<f64> = free.iter().map(|&i| q[i]).collect();
                let steps: Vec<f64> = free.iter().map(|&i| if i < nc { 0.05e-3 } else { 0.02 }).collect();
                let mut f = |x: &[f64]| -> Vec<f64> {
                    let mut full = q.clone();
                    for (j, &i) in free.iter().enumerate() {
                        full[i] = x[j];
                    }
                    let m = model(&full);
                    m.iter().zip(data).map(|(a, b)| a - b).collect()
                };
                let (x, c) = lm(&mut f, &q0, &steps, lm_iters);
                if best.as_ref().map_or(true, |b| c < b.1) {
                    let mut full = q.clone();
                    for (j, &i) in free.iter().enumerate() {
                        full[i] = x[j];
                    }
                    best = Some((full, c));
                }
            }
            let (b, c) = best.unwrap();
            p = b;
            log(&format!("stage {stage}: cost {c:.4e}, v {:?} mm/s", (0..nc).map(|k| p[k] * 1e3).collect::<Vec<_>>()));
        }
        // joint refinement
        let steps: Vec<f64> = (0..2 * nc).map(|i| if i < nc { 0.05e-3 } else { 0.02 }).collect();
        let mut f = |x: &[f64]| -> Vec<f64> { model(x).iter().zip(data).map(|(a, b)| a - b).collect() };
        let (x, _) = lm(&mut f, &p, &steps, lm_iters);
        let mut v: Vec<f64> = x[..nc].to_vec();
        let mut delta: Vec<f64> = x[nc..].to_vec();
        for k in 0..nc {
            if v[k] < 0.0 {
                v[k] = -v[k];
                delta[k] += PI;
            }
            delta[k] = delta[k].rem_euclid(2.0 * PI);
        }
        let m = model(&x);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let ss_res: f64 = m.iter().zip(data).map(|(a, b)| (a - b).powi(2)).sum();
        let ss_tot: f64 = data.iter().map(|b| (b - mean).powi(2)).sum();
        FitResult { v, delta, r2: 1.0 - ss_res / ss_tot, model: m }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// tests (small configurations: they must run in debug builds)
// ---------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Acq {
        let mut a = Acq::paper();
        a.dz = 50e-6;
        a.n_dummy = 12;
        a.n_scan = 3;
        a
    }

    fn zero_flow(acq: &Acq) -> Flow {
        Flow::from_vel(acq.t_end() + 1.0, 0.01, |_| 0.0)
    }

    #[test]
    fn alias_of_cardiac_at_tr2() {
        assert!((alias_freq(0.93, 0.5) - 0.07).abs() < 1e-12);
        assert!((alias_freq(0.23, 0.5) - 0.23).abs() < 1e-12);
    }

    #[test]
    fn static_single_slice_matches_closed_form() {
        let mut acq = small();
        acq.n_slices = 1;
        acq.n_sim = 1;
        let mut med = Medium::csf();
        med.diffusion = 0.0;
        let sim = InflowSim::new(acq.clone(), med.clone());
        let o = sim.run(&zero_flow(&acq));
        let e1 = (-acq.tr / med.t1).exp();
        let pr = sim.profile();
        let mut want = 0.0;
        let mut z = -acq.extent();
        while z < acq.extent() {
            let a = pr.alpha_at(z);
            want += a.sin() * (1.0 - e1) / (1.0 - a.cos() * e1) * acq.dz;
            z += acq.dz;
        }
        want *= (-acq.te / med.t2).exp();
        let got = o.raw[1];
        eprintln!("static single slice: sim {got:.6e}, closed form {want:.6e}");
        assert!((got / want - 1.0).abs() < 2e-3, "{got} vs {want}");
    }

    #[test]
    fn fast_flow_saturates_to_fresh_signal() {
        let mut acq = small();
        acq.n_slices = 1;
        acq.n_sim = 1;
        acq.n_dummy = 1;
        acq.n_scan = 3;
        let mut med = Medium::csf();
        med.diffusion = 0.0;
        let sim = InflowSim::new(acq.clone(), med.clone());
        let v = 0.05; // 50 mm/s: 100 mm per TR, element is fresh at every excitation
        let o = sim.run(&Flow::from_vel(acq.t_end() + 1.0, 0.01, |_| v));
        let pr = sim.profile();
        let mut fresh = 0.0;
        let mut z = -acq.extent();
        while z < acq.extent() {
            fresh += pr.alpha_at(z).sin() * acq.dz;
            z += acq.dz;
        }
        fresh *= (-acq.te / med.t2).exp();
        let stat = sim.run(&zero_flow(&acq)).raw[1];
        eprintln!("fresh {fresh:.6e} flowing {:.6e} static {stat:.6e}", o.raw[1]);
        assert!((o.raw[1] / fresh - 1.0).abs() < 1e-3);
        assert!(o.raw[1] > stat);
    }

    #[test]
    fn neighbour_slices_saturate_the_bottom_slice() {
        let mut single = small();
        single.n_slices = 1;
        single.n_sim = 1;
        let mut multi = small();
        multi.n_slices = 3;
        multi.n_sim = 3;
        multi.order = SliceOrder::Sequential;
        let mut med = Medium::csf();
        med.diffusion = 0.0;
        let s1 = InflowSim::new(single.clone(), med.clone()).run(&zero_flow(&single)).raw[1];
        let s3 = InflowSim::new(multi.clone(), med).run(&zero_flow(&multi)).raw[1];
        eprintln!("bottom slice alone {s1:.6e}, with neighbours {s3:.6e}");
        assert!(s3 < s1);
    }

    #[test]
    fn continuum_matches_monte_carlo_with_dispersion() {
        let mut acq = small();
        acq.n_sim = 4;
        acq.n_dummy = 6;
        acq.n_scan = 6;
        acq.dz = 100e-6;
        let med = Medium::csf();
        let comps = vec![Component::new(PhaseClock::constant(0.93, 0.3, acq.t_end() + 1.0, 0.01), 0.012, 0.4)];
        let flow = flow_from_components(&comps, acq.t_end() + 1.0, 2e-3);
        let sim = InflowSim::new(acq.clone(), med.clone());
        let a = sim.run(&flow).raw;
        let b = lagrangian_signal(&acq, &med, &flow, 600_000, 7);
        let ma = a.iter().sum::<f64>() / a.len() as f64;
        let err = a.iter().zip(&b).map(|(x, y)| ((x - y) / ma).abs()).fold(0.0, f64::max);
        eprintln!("continuum vs Monte Carlo max relative difference {err:.4}");
        assert!(err < 0.03, "{err}");
    }
}
