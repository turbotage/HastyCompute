//! GIRF QA measurement protocol (research/girf_qa_sequence.md, WP2 of research/plan_girf_b0_qfit.md): thin-slice
//! (Duyn 1998) and phase-encoded thin-slice (Rahmer 2019) frames, test inputs, heating trains with the 4D-flow TR,
//! and the direct measurement of the played v2 readout, as Pulseq blocks for GE / pge2.
//!
//! One *frame* = one pge2 segment (TRID): [RF (Hann sinc, TBW 4) + slice select on axis a, polarity η]
//! [rephaser (+ in-plane phase-encode blips)] [test input σ·level·u(t) on axis a + ADC chunk 1] [ADC chunks]
//! [spoiler] [wait]. Inputs, PE amplitudes, σ, η, slice positions (RF frequency) and levels are amplitude / frequency
//! changes of one segment, as pge2 allows; a zero amplitude is replaced by a tiny one (pge2 rule).
//! Ordering: within a sweep the 12 slices are interleaved over the axes (each slice recovers for 11 frames); the test
//! polarity follows the pattern + − − + over consecutive frames, shifted by one per sweep, so the leakage of the
//! previous frame's tail cancels in the ± difference and every slice sees both polarities twice per 4 sweeps; η
//! alternates every 4 sweeps (removes the first-order B0 warp of the slices, girf_sensitivity.md §5.5).
//! Timing model: pge2 adds `seg_gap` (117 µs) after every segment instance; the as-played timeline used for the
//! thermal power covariate and the simulation includes it.

use crate::pns::{pns, SafeHw};
use crate::pnsge::{pns_ge, GeCoil};
use crate::pulseq::{sinc_pulse, trap_area, AdcEv, Block, GradEv, Label, PulseqSystem, Sequence};
use crate::wave::Grad;
use crate::{Mat3, Vec3, GAMMA};

#[derive(Clone, Debug)]
pub struct QaConfig {
    pub sys: PulseqSystem,
    /// amplitude cap of the test inputs, T/m
    pub g_cap: f64,
    /// slice positions along the slice axis (m)
    pub slices: Vec<f64>,
    pub thk: f64,
    /// slice thickness of the v2-validation frames (larger k range: encodes reach 3600 rad/m)
    pub thk_v: f64,
    pub tbw: f64,
    pub t_rf: f64,
    pub flip: f64,
    pub dwell: f64,
    /// samples per ADC event [CHECK the pge2 / GE maximum]
    pub adc_chunk: usize,
    pub t_adc_long: f64,
    pub t_adc_short: f64,
    /// chirp: f0, f1, sweep duration; slew fraction
    pub chirp: (f64, f64, f64),
    pub s_chirp: f64,
    pub tri_ramps: Vec<f64>,
    pub s_tri: f64,
    pub s_tri_long: f64,
    /// |k| of every input ≤ klim_frac · k_lim (k_lim: slice-profile FT down to sinc(0.4))
    pub klim_frac: f64,
    /// in-plane phase encoding: n × n steps, FOV (m)
    pub pe_n: usize,
    pub pe_fov: f64,
    pub pns_limit: f64,
    pub safe: SafeHw,
    pub ge: GeCoil,
    /// dead time the interpreter adds after each segment instance
    pub seg_gap: f64,
    /// frame period padding after the last ADC sample (spoiler + settle)
    pub frame_pad: f64,
}

impl Default for QaConfig {
    fn default() -> Self {
        let mut sys = PulseqSystem::ge_pge2(3.0);
        sys.gmax = 70e-3; // TOPPE lists HRMW at 70 mT/m; inputs stay far below either way [CHECK]
        Self {
            sys,
            g_cap: 45e-3,
            slices: vec![-26.25e-3, -8.75e-3, 8.75e-3, 26.25e-3],
            thk: 1.5e-3,
            thk_v: 1.0e-3,
            tbw: 4.0,
            t_rf: 3e-3,
            flip: 90f64.to_radians(),
            dwell: 4e-6,
            adc_chunk: 12500,
            t_adc_long: 0.2,
            t_adc_short: 0.1,
            chirp: (100.0, 20e3, 40e-3),
            s_chirp: 0.5,
            tri_ramps: vec![60e-6, 100e-6, 160e-6, 240e-6],
            s_tri: 0.9,
            s_tri_long: 0.55,
            klim_frac: 0.8,
            pe_n: 5,
            pe_fov: 0.2,
            pns_limit: 80.0,
            safe: SafeHw::example(),
            ge: GeCoil::hrmw(),
            seg_gap: 117e-6,
            frame_pad: 4e-3,
        }
    }
}

impl QaConfig {
    pub fn klim(&self, thk: f64) -> f64 {
        0.4 * 2.0 * std::f64::consts::PI / thk
    }
    pub fn g_ss(&self, thk: f64) -> f64 {
        self.tbw / (self.t_rf * self.sys.gamma * thk)
    }
    fn ceil_r(&self, t: f64) -> f64 {
        let r = self.sys.grad_raster;
        (t / r - 1e-9).ceil() * r
    }
}

/// a test input: corner samples on the gradient raster from its block start (T/m, unit level)
#[derive(Clone, Debug)]
pub struct Input {
    pub name: String,
    pub g: Vec<f64>,
}

/// PNS (%) of a single-axis waveform on axis `a`: (SAFE example hw, GE coil model)
pub fn pns_pair(cfg: &QaConfig, g: &[f64], a: usize) -> (f64, f64) {
    let w = Grad::from_axis(a, g, cfg.sys.grad_raster, 0.0);
    let pad = (5e-3 / cfg.sys.grad_raster) as usize;
    (pns(&w, &cfg.safe).max, pns_ge(&w, &cfg.ge, pad).2)
}

fn worst_pns(cfg: &QaConfig, g: &[f64]) -> f64 {
    (0..3).map(|a| {
        let (s, e) = pns_pair(cfg, g, a);
        s.max(e)
    }).fold(0.0, f64::max)
}

/// log chirp f0 → f1 over T (cosine phase, k symmetric), amplitude min(g_cap, slew, k-limit), 2-cycle ramp-in, 1 ms
/// ramp-out, starting `t_on` after the block start (thinslice.py `Protocol.inputs`, on this raster)
pub fn chirp(cfg: &QaConfig, s_frac: f64, thk: f64, t_on: f64) -> Vec<f64> {
    let dt = cfg.sys.grad_raster;
    let (f0, f1, tt) = cfg.chirp;
    let kl = cfg.klim_frac * cfg.klim(thk);
    let n = ((t_on + tt + 0.5e-3) / dt).round() as usize + 1;
    let l = (f1 / f0).ln();
    (0..n)
        .map(|i| {
            let t = i as f64 * dt - t_on;
            if !(0.0..=tt).contains(&t) {
                return 0.0;
            }
            let f = f0 * (t * l / tt).exp();
            let ph = 2.0 * std::f64::consts::PI * f0 * tt / l * ((t * l / tt).exp() - 1.0);
            let a = cfg.g_cap.min(s_frac * cfg.sys.smax / (2.0 * std::f64::consts::PI * f.max(1.0))).min(2.0 * std::f64::consts::PI * f.max(1.0) * kl / GAMMA);
            let w = ((t * f0 / 2.0).clamp(0.0, 1.0) * ((tt - t) / 1e-3).clamp(0.0, 1.0) * 0.5 * std::f64::consts::PI).sin().powi(2);
            a * w * ph.cos()
        })
        .collect()
}

/// prephased triangle: slow trapezoid of −area/2 at 0.5 ms, the triangle (ramp `tr`, slew s_frac·S_max) at 3 ms
pub fn triangle(cfg: &QaConfig, tr: f64, s_frac: f64, thk: f64) -> Vec<f64> {
    let dt = cfg.sys.grad_raster;
    let kl = cfg.klim_frac * cfg.klim(thk);
    let gp = (s_frac * cfg.sys.smax * tr).min(cfg.g_cap).min(2.0 * kl / (GAMMA * tr));
    let nr = (tr / dt).round() as usize;
    let area = gp * nr as f64 * dt;
    let s = 0.3 * cfg.sys.smax;
    let mut gpp = (10e-3f64).min((s * area / 2.0).sqrt());
    let nrr = ((gpp / s / dt) - 1e-9).ceil().max(1.0) as usize;
    gpp = gpp.min(s * nrr as f64 * dt);
    let nf = ((area / 2.0 / gpp - nrr as f64 * dt) / dt).round().max(0.0) as usize;
    let mut pre: Vec<f64> = (0..=nrr).map(|i| gpp * i as f64 / nrr as f64).collect();
    pre.extend(std::iter::repeat(gpp).take(nf));
    pre.extend((0..=nrr).rev().map(|i| gpp * i as f64 / nrr as f64));
    let pa: f64 = pre.windows(2).map(|w| 0.5 * (w[0] + w[1]) * dt).sum();
    for x in pre.iter_mut() {
        *x *= (area / 2.0) / pa;
    }
    let i0 = (0.5e-3 / dt).round() as usize;
    let i1 = (3e-3 / dt).round() as usize;
    let n = i1 + 2 * nr + 1 + (0.5e-3 / dt) as usize;
    let mut g = vec![0.0; n];
    for (j, v) in pre.iter().enumerate() {
        g[i0 + j] -= v;
    }
    for j in 0..=2 * nr {
        g[i1 + j] += gp * (1.0 - (j as f64 - nr as f64).abs() / nr as f64);
    }
    g
}

/// targeted input around a resonance: linear chirp fc·(1 ± 6ζ) (at least ±15 Hz) over `t_sweep` with a sin² envelope,
/// amplitude from the k-limit and slew (pass 2 of the resonance survey)
pub fn narrow_chirp(cfg: &QaConfig, fc: f64, zeta: f64, t_sweep: f64, s_frac: f64, thk: f64, t_on: f64) -> Vec<f64> {
    let dt = cfg.sys.grad_raster;
    let kl = cfg.klim_frac * cfg.klim(thk);
    let half = (6.0 * zeta * fc).max(15.0);
    let (fa, fb) = (fc - half, fc + half);
    let a = cfg.g_cap.min(s_frac * cfg.sys.smax / (2.0 * std::f64::consts::PI * fb)).min(2.0 * std::f64::consts::PI * fa * kl / GAMMA);
    let n = ((t_on + t_sweep + 0.5e-3) / dt).round() as usize + 1;
    (0..n)
        .map(|i| {
            let t = i as f64 * dt - t_on;
            if !(0.0..=t_sweep).contains(&t) {
                return 0.0;
            }
            let ph = 2.0 * std::f64::consts::PI * (fa * t + 0.5 * (fb - fa) / t_sweep * t * t);
            let w = (std::f64::consts::PI * t / t_sweep).sin().powi(2);
            a * w * ph.sin()
        })
        .collect()
}

/// tone burst at f (Hann envelope, duration t_b), k-limited amplitude
pub fn tone_burst(cfg: &QaConfig, f: f64, t_b: f64, s_frac: f64, thk: f64, t_on: f64) -> Vec<f64> {
    let dt = cfg.sys.grad_raster;
    let kl = cfg.klim_frac * cfg.klim(thk);
    let a = cfg.g_cap.min(s_frac * cfg.sys.smax / (2.0 * std::f64::consts::PI * f)).min(2.0 * std::f64::consts::PI * f * kl / GAMMA);
    let n = ((t_on + t_b + 0.5e-3) / dt).round() as usize + 1;
    (0..n)
        .map(|i| {
            let t = i as f64 * dt - t_on;
            if !(0.0..=t_b).contains(&t) {
                return 0.0;
            }
            a * (std::f64::consts::PI * t / t_b).sin().powi(2) * (2.0 * std::f64::consts::PI * f * t).sin()
        })
        .collect()
}

/// scale an input generator's slew fraction down (×0.93 steps) until both PNS models are ≤ the limit on every axis;
/// returns the waveform and the fraction used
pub fn pns_fit(cfg: &QaConfig, s0: f64, make: impl Fn(f64) -> Vec<f64>) -> (Vec<f64>, f64) {
    let mut s = s0;
    loop {
        let g = make(s);
        if worst_pns(cfg, &g) <= cfg.pns_limit || s < 0.05 {
            return (g, s);
        }
        s *= 0.93;
    }
}

/// the survey inputs (chirp + 4 triangles), each PNS-fitted
pub fn survey_inputs(cfg: &QaConfig, thk: f64) -> Vec<(Input, f64)> {
    let mut v = vec![];
    let (g, s) = pns_fit(cfg, cfg.s_chirp, |s| chirp(cfg, s, thk, 1e-3));
    v.push((Input { name: "chirp".into(), g }, s));
    for &tr in &cfg.tri_ramps {
        let s0 = if tr <= 100e-6 { cfg.s_tri } else { cfg.s_tri_long };
        let (g, s) = pns_fit(cfg, s0, |s| triangle(cfg, tr, s, thk));
        v.push((Input { name: format!("tri{}", (tr * 1e6).round() as i64), g }, s));
    }
    v
}

/// what one frame is
#[derive(Clone, Debug)]
pub struct FrameSpec {
    pub kind: u8,
    pub trid: i64,
    pub axis: usize,
    pub slice: usize,
    pub pos: f64,
    pub eta: f64,
    pub sigma: f64,
    pub level: f64,
    pub input: usize,
    /// in-plane PE steps (rad/m on the two other axes, in increasing axis order); None = no PE events
    pub pe: Option<[f64; 2]>,
    pub thk: f64,
    pub t_adc: f64,
    pub avg: usize,
    /// v2 readout index (V frames)
    pub readout: i64,
}

/// timing of one built frame (relative to the frame start)
#[derive(Clone, Debug, Default)]
pub struct FrameTiming {
    pub t_exc: f64,
    pub t_in0: f64,
    pub adc0: Vec<f64>,
    pub nadc: Vec<usize>,
    pub duration: f64,
}

/// max PE area (T·s/m) for the configured grid
pub fn pe_kmax(cfg: &QaConfig) -> f64 {
    let dk = 2.0 * std::f64::consts::PI / cfg.pe_fov;
    dk * (cfg.pe_n as f64 - 1.0) / 2.0
}

pub fn pe_steps(cfg: &QaConfig) -> Vec<[f64; 2]> {
    let dk = 2.0 * std::f64::consts::PI / cfg.pe_fov;
    let h = (cfg.pe_n as f64 - 1.0) / 2.0;
    let mut v = vec![];
    for i in 0..cfg.pe_n {
        for j in 0..cfg.pe_n {
            v.push([(i as f64 - h) * dk, (j as f64 - h) * dk]);
        }
    }
    v
}

const EPS_AMP: f64 = 1e-6;

fn nz(x: f64) -> f64 {
    if x == 0.0 {
        EPS_AMP
    } else {
        x
    }
}

/// build the pre-input blocks (RF + slice select, rephaser + PE) for slice axis `a`; returns the blocks, the RF
/// centre from the start, and the time at the end
pub fn excitation_blocks(cfg: &QaConfig, a: usize, pos: f64, eta: f64, thk: f64, pe: Option<[f64; 2]>, trid: i64) -> (Vec<Block>, f64, f64) {
    let s = &cfg.sys;
    let gss = cfg.g_ss(thk);
    let rise = cfg.ceil_r((gss / (0.25 * s.smax)).max(s.rf_dead));
    let flat = cfg.ceil_r(cfg.t_rf);
    let freq = s.gamma * eta * gss * pos;
    let (rf, _) = sinc_pulse(s, cfg.flip, cfg.t_rf, cfg.tbw, thk, rise, freq, 0.0);
    let mut b1 = Block { rf: Some(rf), labels: vec![Label::set("TRID", trid)], ..Default::default() };
    b1.g[a] = Some(GradEv::Trap { amp: eta * gss, rise, flat, fall: rise, delay: 0.0 });
    let d1 = 2.0 * rise + flat;
    let t_exc = rise + 0.5 * cfg.t_rf;
    // rephaser: −η G_ss (t_rf/2 + rise/2), fixed timing for both η (amplitude sign only)
    let area = gss * (0.5 * cfg.t_rf + 0.5 * rise);
    let mut b2 = Block::default();
    let reph = trap_area(s, area, 30e-3, 0.25 * s.smax, 0.0);
    b2.g[a] = Some(reph.scaled(-eta));
    if let Some(k) = pe {
        let kmax = pe_kmax(cfg);
        let tmpl = trap_area(s, kmax / GAMMA, 20e-3, 0.25 * s.smax, 0.0);
        let others: Vec<usize> = (0..3).filter(|&b| b != a).collect();
        for (j, &b) in others.iter().enumerate() {
            b2.g[b] = Some(tmpl.scaled(nz(k[j] / kmax)));
        }
    }
    let mut d2 = 0.0f64;
    for g in b2.g.iter().flatten() {
        d2 = d2.max(g.duration(s.grad_raster));
    }
    let d2 = cfg.ceil_r(d2);
    b2.duration = Some(d2);
    (vec![Block { duration: Some(d1), ..b1 }, b2], t_exc, d1 + d2)
}

/// the smallest ADC delay ≥ the dead time, on the RF raster (pypulseq checks ADC delays there; pge2 allows 1 µs), that
/// puts the samples (delay + (i + ½)·dwell) on the gradient raster, so sample i of every chunk coincides with a corner
/// sample of the input (no half-raster ambiguity in the estimator's input / output alignment); if no such delay
/// exists within 100 µs (2 µs dwell: odd µs needed), the smallest RF-raster delay ≥ the dead time
pub fn adc_delay_on_raster(cfg: &QaConfig) -> f64 {
    let r = cfg.sys.grad_raster;
    let rr = cfg.sys.rf_raster;
    let d0 = ((cfg.sys.adc_dead / rr) - 1e-9).ceil() * rr;
    let mut d = d0;
    while d < d0 + 100e-6 {
        let x = (d + 0.5 * cfg.dwell) / r;
        if (x - x.round()).abs() < 1e-6 {
            return d;
        }
        d += rr;
    }
    d0
}

/// ADC chunks for a window of `t_adc` starting in the first block; the input (if any) is played in the first block
fn adc_blocks(cfg: &QaConfig, a: usize, input: Option<&[f64]>, t_adc: f64) -> (Vec<Block>, Vec<f64>, Vec<usize>) {
    let s = &cfg.sys;
    let total = (t_adc / cfg.dwell).round() as usize;
    let mut left = total;
    let mut out = vec![];
    let (mut t, mut adc0, mut nadc) = (0.0, vec![], vec![]);
    let mut first = true;
    while left > 0 {
        let n = left.min(cfg.adc_chunk);
        let adc = AdcEv { num: n, dwell: cfg.dwell, delay: adc_delay_on_raster(cfg), freq: 0.0, phase: 0.0 };
        let mut b = Block { adc: Some(adc.clone()), ..Default::default() };
        let mut d = adc.duration() + s.adc_dead;
        if first {
            if let Some(g) = input {
                b.g[a] = Some(GradEv::Pl { g: g.to_vec(), delay: 0.0 });
                d = d.max((g.len() - 1) as f64 * s.grad_raster);
            }
        }
        let d = cfg.ceil_r(d);
        b.duration = Some(d);
        adc0.push(t + adc_delay_on_raster(cfg) + 0.5 * cfg.dwell);
        nadc.push(n);
        out.push(b);
        t += d;
        left -= n;
        first = false;
    }
    (out, adc0, nadc)
}

/// spoiler (4 cycles across the slice on every axis) + wait
fn spoil_block(cfg: &QaConfig, thk: f64) -> Block {
    let s = &cfg.sys;
    let area = 4.0 / (s.gamma * thk);
    // staggered by one ramp per axis, low slew: the spoiler must not set the frame's PNS
    let tr = trap_area(s, area, 20e-3, 0.15 * s.smax, 0.0);
    let rise = if let GradEv::Trap { rise, .. } = tr { rise } else { 0.0 };
    let mut b = Block::default();
    for a in 0..3 {
        if let GradEv::Trap { amp, rise, flat, fall, .. } = tr {
            b.g[a] = Some(GradEv::Trap { amp, rise, flat, fall, delay: a as f64 * rise });
        }
    }
    let _ = rise;
    let d = cfg.ceil_r(b.g.iter().flatten().map(|g| g.duration(s.grad_raster)).fold(0.0, f64::max) + cfg.frame_pad);
    b.duration = Some(d);
    b
}

/// one thin-slice frame (S / X / L / R / D kinds): blocks and timing
pub fn frame(cfg: &QaConfig, f: &FrameSpec, input: &[f64]) -> (Vec<Block>, FrameTiming) {
    let (mut bl, t_exc, t_end) = excitation_blocks(cfg, f.axis, f.pos, f.eta, f.thk, f.pe, f.trid);
    let g: Vec<f64> = input.iter().map(|x| x * nz(f.sigma * f.level)).collect();
    let (ab, adc0, nadc) = adc_blocks(cfg, f.axis, Some(&g), f.t_adc);
    let t_in0 = t_end - t_exc;
    let adc0: Vec<f64> = adc0.iter().map(|x| x + t_end - t_exc).collect();
    bl.extend(ab);
    bl.push(spoil_block(cfg, f.thk));
    let dur: f64 = bl.iter().map(|b| b.duration.unwrap()).sum();
    (bl, FrameTiming { t_exc, t_in0, adc0, nadc, duration: dur })
}

// ───────────────────────────────────────────────────────────── the 4D-flow TR on the GE raster

/// the v2 flow TR split into pge2 segments on the GIRF raster: per encode the encode part (RF slot + encode, physical),
/// the base readout (logical, rotated per TR), the per-encode tail (physical)
#[derive(Clone, Debug)]
pub struct FlowTr {
    pub dt: f64,
    pub enc: Vec<Vec<Vec3>>,
    pub readout: Vec<Vec3>,
    pub tail: Vec<Vec3>,
    pub tails: Vec<Vec<Vec3>>,
    /// readout start after the encode-segment start, ADC (reconstruction + navigator) duration from the readout start
    pub t_ro: f64,
    pub t_adc: f64,
    pub te: f64,
}

/// resample a piecewise-linear waveform (raster `dt0`, from its start) onto raster `dt` from the same start, padded
/// with zeros to a whole number of samples
pub fn resample_pl(g: &[Vec3], dt0: f64, dt: f64) -> Vec<Vec3> {
    let t1 = (g.len() - 1) as f64 * dt0;
    let n = (t1 / dt - 1e-9).ceil() as usize + 1;
    (0..n)
        .map(|j| {
            let t = j as f64 * dt / dt0;
            if t >= (g.len() - 1) as f64 {
                return g[g.len() - 1];
            }
            let i = t.floor() as usize;
            let a = t - i as f64;
            [0, 1, 2].map(|q| g[i][q] * (1.0 - a) + g[i + 1][q] * a)
        })
        .collect()
}

impl FlowTr {
    /// from the seqdesign arrays: tr_grad [n_enc][n+1][3] on `dt0`, readout samples i_ro..=i_tail, ADC i_adc samples
    pub fn from_design(tr_grad: &[Vec<Vec3>], dt0: f64, i_ro: usize, i_tail: usize, i_adc: usize, te: f64, dt: f64) -> Self {
        let enc: Vec<Vec<Vec3>> = tr_grad.iter().map(|g| resample_pl(&g[..=i_ro], dt0, dt)).collect();
        let readout = resample_pl(&tr_grad[0][i_ro..=i_tail], dt0, dt);
        let tails: Vec<Vec<Vec3>> = tr_grad.iter().map(|g| resample_pl(&g[i_tail..], dt0, dt)).collect();
        Self { dt, t_ro: (enc[0].len() - 1) as f64 * dt, t_adc: i_adc as f64 * dt0, te, tail: tails[0].clone(), tails, enc, readout }
    }
    /// blocks of one heating / history TR: 3 segments (pge2 rotates whole segments: the readout carries the rotation)
    pub fn tr_blocks(&self, e: usize, rot: &Mat3, sigma: f64, trid0: i64) -> Vec<Block> {
        let ev = |w: &[Vec3], a: usize| -> Option<GradEv> {
            let g: Vec<f64> = w.iter().map(|v| v[a] * sigma).collect();
            if g.iter().all(|x| *x == 0.0) {
                return None;
            }
            Some(GradEv::Pl { g, delay: 0.0 })
        };
        let dur = |w: &[Vec3]| (w.len() - 1) as f64 * self.dt;
        let mut b1 = Block { duration: Some(dur(&self.enc[e])), labels: vec![Label::set("TRID", trid0 + e as i64)], ..Default::default() };
        let mut b2 = Block { duration: Some(dur(&self.readout)), labels: vec![Label::set("TRID", trid0 + 7)], rotation: Some(mat_to_quat(rot)), ..Default::default() };
        let mut b3 = Block { duration: Some(dur(&self.tails[e])), labels: vec![Label::set("TRID", trid0 + 8 + e as i64)], ..Default::default() };
        for a in 0..3 {
            b1.g[a] = ev(&self.enc[e], a);
            b2.g[a] = Some(GradEv::Pl { g: self.readout.iter().map(|v| v[a] * nz(sigma)).collect(), delay: 0.0 });
            b3.g[a] = ev(&self.tails[e], a);
        }
        vec![b1, b2, b3]
    }
    /// physical waveform of one TR as played (segments + the interpreter's gaps), on dt
    pub fn tr_played(&self, e: usize, rot: &Mat3, sigma: f64, gap: usize) -> Vec<Vec3> {
        let mut v: Vec<Vec3> = self.enc[e].iter().map(|g| g.map(|x| x * sigma)).collect();
        v.extend(std::iter::repeat([0.0; 3]).take(gap));
        v.extend(self.readout.iter().map(|g| crate::mat_vec(rot, g).map(|x| x * sigma)));
        v.extend(std::iter::repeat([0.0; 3]).take(gap));
        v.extend(self.tails[e].iter().map(|g| g.map(|x| x * sigma)));
        v.extend(std::iter::repeat([0.0; 3]).take(gap));
        v
    }
}

/// unit quaternion (w, x, y, z) of a rotation matrix (Shepperd)
pub fn mat_to_quat(m: &Mat3) -> [f64; 4] {
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    };
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let s = if q[0] < 0.0 { -1.0 } else { 1.0 };
    q.map(|x| s * x / n)
}

/// a v2-validation frame: thin slice on axis a, then the measured TR's axis-a components only (encode e + rotated
/// readout incl. the rewinder / navigator), ADC over encode + readout; the other axes would dephase the slice
pub fn v_frame(cfg: &QaConfig, f: &FrameSpec, ftr: &FlowTr, e: usize, rot: &Mat3) -> (Vec<Block>, FrameTiming, Vec<f64>) {
    let (mut bl, t_exc, t_end) = excitation_blocks(cfg, f.axis, f.pos, f.eta, f.thk, None, f.trid);
    let a = f.axis;
    let mut g: Vec<f64> = ftr.enc[e].iter().map(|v| v[a]).collect();
    // the RF slot of the flow TR (before its encode) is replaced by the thin-slice excitation: start at the encode
    let i_enc = g.iter().position(|x| *x != 0.0).unwrap_or(0).saturating_sub(1);
    g.drain(..i_enc);
    g.pop();
    g.extend(ftr.readout.iter().map(|v| crate::mat_vec(rot, v)[a]));
    let lead = (0.2e-3 / cfg.sys.grad_raster) as usize;
    let mut gin = vec![0.0; lead];
    gin.extend(g.iter());
    let t_adc = (gin.len() - 1) as f64 * cfg.sys.grad_raster;
    let gs: Vec<f64> = gin.iter().map(|x| x * nz(f.sigma)).collect();
    let (ab, adc0, nadc) = adc_blocks(cfg, a, Some(&gs), t_adc - cfg.sys.adc_dead - cfg.dwell);
    let t_in0 = t_end - t_exc;
    let adc0: Vec<f64> = adc0.iter().map(|x| x + t_end - t_exc).collect();
    bl.extend(ab);
    bl.push(spoil_block(cfg, f.thk));
    let dur: f64 = bl.iter().map(|b| b.duration.unwrap()).sum();
    (bl, FrameTiming { t_exc, t_in0, adc0, nadc, duration: dur }, gin)
}

/// a heater TR (`t_tr`) for loads above the flow scan's: per axis a bipolar pair of trapezoids at low slew, the axes
/// staggered by one ramp so their ramps never coincide; zero at both ends (pge2 segment rule)
pub fn heater_tr(cfg: &QaConfig, amp: f64, slew: f64, t_tr: f64) -> Vec<Vec3> {
    let dt = cfg.sys.grad_raster;
    let n = (t_tr / dt).round() as usize;
    let nr = ((amp / slew) / dt).ceil() as usize;
    let f = (n.saturating_sub(4 * nr + 2 * nr + 2)) / 2;
    let lobe = |j: usize| -> f64 {
        // trapezoid of ramps nr and flat f starting at 0
        let l = 2 * nr + f;
        if j >= l {
            return 0.0;
        }
        let up = j as f64 / nr as f64;
        let down = (l - j) as f64 / nr as f64;
        amp * up.min(down).min(1.0)
    };
    (0..=n)
        .map(|i| [0, 1, 2].map(|a| {
            let o = 1 + a * nr;
            if i < o {
                return 0.0;
            }
            let j = i - o;
            let l = 2 * nr + f;
            if j < l { lobe(j) } else if j < 2 * l { -lobe(j - l) } else { 0.0 }
        }))
        .collect()
}

/// mean Σ_a G_a² (T²/m²) of a waveform
pub fn mean_g2(g: &[Vec3]) -> f64 {
    g.iter().map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sum::<f64>() / g.len().max(1) as f64
}

/// sweep order of one (input, 4-sweep group): (slice-unit index, σ) with the + − − + pattern shifted per sweep
pub fn sweep_order(n_units: usize) -> Vec<(usize, f64)> {
    let pat = [1.0, -1.0, -1.0, 1.0];
    let mut v = vec![];
    for m in 0..4 {
        for n in 0..n_units {
            v.push((n, pat[(n + m) % 4]));
        }
    }
    v
}

/// slice units: (axis, slice index) interleaved axis-fastest
pub fn slice_units(cfg: &QaConfig, slices: &[usize]) -> Vec<(usize, usize)> {
    let mut v = vec![];
    for &s in slices {
        for a in 0..3 {
            v.push((a, s));
        }
    }
    let _ = cfg;
    v
}

/// append blocks to a sequence and the as-played physical waveform power: returns (energy Σ∫G², duration incl. gaps)
pub fn energy(blocks: &[Block], sys: &PulseqSystem) -> (f64, f64) {
    let mut s = Sequence::new(sys.clone());
    for b in blocks {
        s.add(b.clone());
    }
    let g = s.grad_raster();
    (g.g.iter().map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sum::<f64>() * g.dt, s.duration())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pulseq::SeqFile;

    #[test]
    fn inputs_respect_limits_and_pns() {
        let cfg = QaConfig::default();
        let inp = survey_inputs(&cfg, cfg.thk);
        assert_eq!(inp.len(), 5);
        let kl = cfg.klim(cfg.thk);
        for (i, s) in &inp {
            let w = Grad::from_axis(0, &i.g, cfg.sys.grad_raster, 0.0);
            let l = w.limits();
            assert!(l.g_axis <= cfg.g_cap * (1.0 + 1e-9) && l.s_axis <= cfg.sys.smax * (1.0 + 1e-6), "{}: {l:?}", i.name);
            let kmax = w.moments(0.0).k().iter().map(|k| (k[0] * 2.0 * std::f64::consts::PI).abs()).fold(0.0, f64::max);
            assert!(kmax <= kl * 1.0001, "{}: k {kmax} > {kl}", i.name);
            assert!(worst_pns(&cfg, &i.g) <= cfg.pns_limit + 1e-9, "{} PNS (s {s})", i.name);
            assert_eq!(i.g[0], 0.0);
            assert_eq!(*i.g.last().unwrap(), 0.0);
        }
    }

    #[test]
    fn frame_is_valid_pulseq_and_round_trips() {
        let cfg = QaConfig::default();
        let inp = survey_inputs(&cfg, cfg.thk);
        let mut seq = Sequence::new(cfg.sys.clone());
        for (k, (a, s)) in sweep_order(12).iter().take(8).enumerate() {
            let (ax, sl) = slice_units(&cfg, &[0, 1, 2, 3])[*a];
            let f = FrameSpec { kind: 1, trid: 1 + ax as i64, axis: ax, slice: sl, pos: cfg.slices[sl], eta: if k % 2 == 0 { 1.0 } else { -1.0 }, sigma: *s, level: 1.0, input: 0, pe: if k > 4 { Some([31.4, -62.8]) } else { None }, thk: cfg.thk, t_adc: 0.06, avg: 0, readout: -1 };
            let (bl, tm) = frame(&cfg, &f, &inp[0].0.g);
            assert!(tm.t_in0 > 0.0 && tm.adc0[0] > tm.t_in0);
            for b in bl {
                seq.add(b);
            }
        }
        let errs = seq.check();
        assert!(errs.is_empty(), "{errs:?}");
        let f = SeqFile::parse(&seq.to_text()).unwrap();
        assert!(f.signature_ok);
        let (a, b) = (seq.grad_raster(), f.grad_raster(cfg.sys.gamma));
        let e = a.g.iter().zip(&b.g).flat_map(|(p, q)| (0..3).map(move |i| (p[i] - q[i]).abs())).fold(0.0, f64::max);
        assert!(e < 1e-8, "{e}");
    }

    #[test]
    fn quaternion_round_trip() {
        for (ax, ang) in [([1.0, 0.0, 0.0], 0.3), ([0.2, -1.0, 0.5], 2.9), ([0.0, 0.0, 1.0], 3.14159), ([1.0, 1.0, 1.0], -1.0)] {
            let m = crate::axis_angle(&ax, ang);
            let q = mat_to_quat(&m);
            let m2 = crate::quat_to_mat(q);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((m[i][j] - m2[i][j]).abs() < 1e-12);
                }
            }
        }
    }
}
