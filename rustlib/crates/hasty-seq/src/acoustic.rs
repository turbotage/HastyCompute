//! Acoustic noise simulator (plan: `research/acoustic_sim.md`).
//!
//! gradient waveform G_a(t) (physical axes) → Lorentz force on the gradient coil (∝ B0·G_a) → pressure at the ear
//! p(t) = Σ_a h_a ∗ G_a, h_a = a sum of 2nd-order band-pass resonances (+ optional hearing protection).
//!
//! The default model ([`AcousticModel::toy_3t`]) is a **toy**: resonance frequencies / Q follow the broad literature
//! picture of a 3 T whole-body gradient coil (mechanical modes in the 0.3 – 3 kHz range, a few of them per axis,
//! Q ≈ 15 – 30), and the gains give roughly 100 – 120 dB(A) at the ear for strong readout trains. It is not a
//! measurement of any scanner: relative spectra (which harmonic sits on a resonance) are the usable output, the
//! absolute level is only indicative until a chirp-gradient measurement calibrates it.
//!
//! Pipeline per window ([`simulate`]): the piecewise-linear gradient (exact at the gradient raster) is padded to a
//! whole number of ms, transformed, multiplied by the response (with the sinc² of the linear interpolation), band
//! limited to 24 kHz and transformed back at 48 kHz in one step (the resampling is the truncation of the spectrum).
//! The window's ring-out is handled by a zero-guard after its end; a lead-in of earlier TRs carries the ringing that
//! was already present at the window start.

use std::path::Path;

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use serde_json::{json, Value};

use crate::pulseq::SeqFile;
use crate::wave::Grad;
use crate::GAMMA_BAR;

/// reference pressure, Pa
pub const P_REF: f64 = 20e-6;
/// audio output rate
pub const FS_AUDIO: f64 = 48_000.0;
const FS_MS: usize = 48;
/// octave band centres of the TR signature
pub const BANDS: [f64; 7] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
/// signature length: octave bands + total dB(A)
pub const NSIG: usize = 8;

// ───────────────────────────────────────────────────────────────────────────── model

/// one mechanical / acoustic resonance: `peak` is the pressure amplitude per T/m at `f0` (Pa / (T/m)) at `b0_ref`
#[derive(Clone, Debug)]
pub struct Mode {
    pub f0: f64,
    pub q: f64,
    pub peak: f64,
}

#[derive(Clone, Debug)]
pub struct AcousticModel {
    pub name: String,
    pub b0_ref: f64,
    pub b0: f64,
    pub axes: [Vec<Mode>; 3],
    /// hearing protection attenuation (frequency Hz, dB), log-f interpolation
    pub protection: Vec<(f64, f64)>,
    pub use_protection: bool,
    /// per-axis arrival time offset (s): path / timing difference between the axes' responses (Wu et al. 2013 need
    /// it to combine axes)
    pub delay: [f64; 3],
    /// reverberation (RT60 s, reverberant-to-direct level dB): the noise outlasts the gradients by 300–500 ms
    /// (Ravicz et al. 2000: coil resonance + room); None = off
    pub reverb: Option<(f64, f64)>,
}

impl AcousticModel {
    /// Literature-informed toy 3 T whole-body gradient coil (module doc; not a measurement). Resonances: clinical 3 T
    /// systems have the main modes at ≈ 1.2 – 1.9 kHz (x ≈ 1.7 kHz and z ≈ 1.3 kHz on one research 3 T system, Dillinger
    /// 2022; ≈ 1.4 / 1.9 / 2.4 kHz on a high-performance coil, x ≈ y ≠ z), a ≈ 0.7 kHz component at switching
    /// transients (Ravicz 2000), most energy at 0.5 – 3 kHz (Wu 2013). Gains: the measured transfer functions of Wu
    /// 2013 are plotted re 0.1 Pa/(mT/m) = 100 Pa/(T/m); peaks here ≈ 10 – 20 dB above that, valleys not deeper than
    /// ≈ 15 dB below (broad low-Q floor modes).
    pub fn toy_3t() -> Self {
        let m = |f0, q, peak| Mode { f0, q, peak };
        let floor = || vec![m(900.0, 1.3, 110.0), m(1800.0, 1.5, 90.0)];
        let mut ax = [
            vec![m(720.0, 15.0, 250.0), m(1230.0, 20.0, 500.0), m(1725.0, 25.0, 900.0), m(2400.0, 30.0, 350.0)],
            vec![m(740.0, 15.0, 250.0), m(1280.0, 20.0, 500.0), m(1700.0, 24.0, 800.0), m(2400.0, 30.0, 350.0)],
            vec![m(700.0, 15.0, 200.0), m(1275.0, 22.0, 900.0), m(2000.0, 28.0, 450.0), m(2900.0, 32.0, 250.0)],
        ];
        for a in ax.iter_mut() {
            a.extend(floor());
        }
        Self {
            name: "toy 3T whole-body gradient, literature-informed (uncalibrated)".into(),
            b0_ref: 3.0,
            b0: 3.0,
            axes: ax,
            // earplugs + muffs, typical combined attenuation (rough; plugs alone 10 – 30 dB)
            protection: vec![(125.0, 20.0), (250.0, 25.0), (500.0, 28.0), (1000.0, 30.0), (2000.0, 35.0), (4000.0, 42.0), (8000.0, 42.0)],
            use_protection: false,
            delay: [0.0; 3],
            reverb: None,
        }
    }

    fn prot_db(&self, f: f64) -> f64 {
        let t = &self.protection;
        if t.is_empty() {
            return 0.0;
        }
        if f <= t[0].0 {
            return t[0].1;
        }
        for w in t.windows(2) {
            if f <= w[1].0 {
                let x = (f / w[0].0).ln() / (w[1].0 / w[0].0).ln();
                return w[0].1 + x * (w[1].1 - w[0].1);
            }
        }
        t[t.len() - 1].1
    }

    /// pressure per unit gradient (Pa per T/m) of axis `a` at frequency `f`
    pub fn h(&self, a: usize, f: f64) -> Complex64 {
        let mut s = Complex64::new(0.0, 0.0);
        for m in &self.axes[a] {
            let u = f / m.f0;
            let jq = Complex64::new(0.0, u / m.q);
            s += m.peak * jq / (Complex64::new(1.0 - u * u, 0.0) + jq);
        }
        s *= self.b0 / self.b0_ref;
        if self.delay[a] != 0.0 {
            s *= Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f * self.delay[a]);
        }
        if self.use_protection {
            s *= 10f64.powf(-self.prot_db(f) / 20.0);
        }
        s
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("model json: {e}"))?;
        let mut m = Self::toy_3t();
        m.name = v["name"].as_str().unwrap_or("custom").to_string();
        m.b0_ref = v["b0_ref"].as_f64().ok_or("model: b0_ref missing")?;
        m.b0 = v["b0"].as_f64().unwrap_or(m.b0_ref);
        let ax = v["axes"].as_array().filter(|a| a.len() == 3).ok_or("model: axes must be an array of 3 mode lists")?;
        for (a, l) in ax.iter().enumerate() {
            m.axes[a] = l
                .as_array()
                .ok_or("model: axis modes")?
                .iter()
                .map(|e| Ok(Mode { f0: e["f0"].as_f64().ok_or("mode f0")?, q: e["q"].as_f64().ok_or("mode q")?, peak: e["peak"].as_f64().ok_or("mode peak")? }))
                .collect::<Result<_, &str>>()?;
        }
        if let Some(d) = v["delay_s"].as_array().filter(|d| d.len() == 3) {
            for a in 0..3 {
                m.delay[a] = d[a].as_f64().unwrap_or(0.0);
            }
        }
        m.reverb = match (v["reverb"]["rt60"].as_f64(), v["reverb"]["level_db"].as_f64()) {
            (Some(t), Some(l)) => Some((t, l)),
            _ => None,
        };
        if let Some(p) = v["protection"].as_array() {
            m.protection = p.iter().filter_map(|e| Some((e["f"].as_f64()?, e["db"].as_f64()?))).collect();
        }
        Ok(m)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name, "b0_ref": self.b0_ref, "b0": self.b0,
            "note": "peak = pressure amplitude per T/m at f0 (Pa/(T/m)) at b0_ref; h = sum of band-pass modes, scaled b0/b0_ref",
            "axes": self.axes.iter().map(|l| l.iter().map(|m| json!({"f0": m.f0, "q": m.q, "peak": m.peak})).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "protection": self.protection.iter().map(|(f, d)| json!({"f": f, "db": d})).collect::<Vec<_>>(),
            "delay_s": self.delay,
            "reverb": self.reverb.map(|(t, l)| json!({"rt60": t, "level_db": l})),
        })
    }
}

/// IEC 61672 A-weighting magnitude (linear, 1 at 1 kHz)
pub fn a_weight(f: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let f2 = f * f;
    let r = 12194.0f64.powi(2) * f2 * f2 / ((f2 + 20.6f64.powi(2)) * ((f2 + 107.7f64.powi(2)) * (f2 + 737.9f64.powi(2))).sqrt() * (f2 + 12194.0f64.powi(2)));
    r * 1.2589
}

fn sinc2(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let s = (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x);
        s * s
    }
}

/// smallest 2·3·5-smooth integer ≥ n
fn smooth(n: usize) -> usize {
    let mut m = n.max(1);
    loop {
        let mut r = m;
        for p in [2, 3, 5] {
            while r % p == 0 {
                r /= p;
            }
        }
        if r == 1 {
            return m;
        }
        m += 1;
    }
}

/// dB SPL of a mean-square pressure (Pa²)
pub fn db_spl(ms: f64) -> f64 {
    10.0 * (ms.max(1e-300) / (P_REF * P_REF)).log10()
}

// ───────────────────────────────────────────────────────────────────────────── simulation of one window

pub struct Sim {
    pub fs: f64,
    /// time of sample 0 (s, sequence time) = the gradient's `t0`
    pub t0: f64,
    /// total pressure (Pa), per-axis pressure, and the A-weighted total
    pub p: Vec<f64>,
    pub axis: [Vec<f64>; 3],
    pub pa: Vec<f64>,
}

/// pressure at the ear for the gradient `g` (physical axes, raster `g.dt` must divide 1 ms). `guard` (s) of zeros
/// are appended before the transform so the ring-out of the end does not wrap onto the start.
pub fn simulate(model: &AcousticModel, g: &Grad, guard: f64) -> Result<Sim, String> {
    let guard = model.reverb.map_or(guard, |(rt, _)| guard.max(rt * 1.2));
    let k = (1e-3 / g.dt).round() as usize;
    if k == 0 || (k as f64 * g.dt - 1e-3).abs() > 1e-9 {
        return Err(format!("gradient raster {} s does not divide 1 ms", g.dt));
    }
    let dur = g.len().saturating_sub(1) as f64 * g.dt;
    if dur > 120.0 {
        return Err(format!("window of {dur:.0} s too long for one transform (limit 120 s): use fewer TRs"));
    }
    let m = smooth(((dur + guard) * 1e3).ceil() as usize);
    let (nin, nout) = (m * k, m * FS_MS);
    let t_total = m as f64 * 1e-3;
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(nin);
    let inv = planner.plan_fft_inverse(nout);
    let nb = nout / 2;
    // reverberation: random-phase exponentially decaying impulse response (unit energy), added at `level` dB re the
    // direct sound (negative = below it); one fixed sequence, so runs are repeatable
    let rev: Option<Vec<Complex64>> = model.reverb.map(|(rt60, level)| {
        let n_ir = ((rt60 * FS_AUDIO * 1.2) as usize).min(nout);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut ir: Vec<f64> = (0..n_ir)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                u * (-6.9078 * i as f64 / (rt60 * FS_AUDIO)).exp()
            })
            .collect();
        let e = ir.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        ir.iter_mut().for_each(|v| *v /= e);
        let mut r: Vec<Complex64> = (0..nout).map(|i| Complex64::new(ir.get(i).copied().unwrap_or(0.0), 0.0)).collect();
        planner.plan_fft_forward(nout).process(&mut r);
        let w = 10f64.powf(level / 20.0);
        r.iter().map(|c| Complex64::new(1.0, 0.0) + c * w).collect()
    });
    let mut axis: [Vec<f64>; 3] = Default::default();
    let mut ya_sum = vec![Complex64::new(0.0, 0.0); nout];
    let scale = nout as f64 / nin as f64;
    for a in 0..3 {
        let mut x: Vec<Complex64> = (0..nin).map(|i| Complex64::new(g.g.get(i).map_or(0.0, |v| v[a]), 0.0)).collect();
        fwd.process(&mut x);
        let mut y = vec![Complex64::new(0.0, 0.0); nout];
        for j in 1..nb {
            let f = j as f64 / t_total;
            let taper = if f < 20e3 { 1.0 } else if f < 24e3 { 0.5 * (1.0 + (std::f64::consts::PI * (f - 20e3) / 4e3).cos()) } else { 0.0 };
            let mut v = x[j] * model.h(a, f) * (taper * sinc2(f * g.dt) * scale);
            if let Some(r) = &rev {
                v *= r[j];
            }
            y[j] = v;
            y[nout - j] = v.conj();
            let va = v * a_weight(f);
            ya_sum[j] += va;
            ya_sum[nout - j] += va.conj();
        }
        inv.process(&mut y);
        axis[a] = y.iter().map(|c| c.re / nout as f64).collect();
    }
    inv.process(&mut ya_sum);
    let pa: Vec<f64> = ya_sum.iter().map(|c| c.re / nout as f64).collect();
    let n_valid = ((dur * FS_AUDIO).ceil() as usize + 1).min(nout);
    let cut = |v: &Vec<f64>| v[..n_valid].to_vec();
    let axis = [cut(&axis[0]), cut(&axis[1]), cut(&axis[2])];
    let p: Vec<f64> = (0..n_valid).map(|i| axis[0][i] + axis[1][i] + axis[2][i]).collect();
    Ok(Sim { fs: FS_AUDIO, t0: g.t0, p, axis, pa: cut(&pa) })
}

// ───────────────────────────────────────────────────────────────────────────── level metrics

/// mean square (Pa²)
pub fn mean_sq(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64
    }
}

/// maximum of the exponentially (τ = 125 ms, "fast") averaged mean square, dB
pub fn l_fast_max(x: &[f64], fs: f64) -> f64 {
    let a = (-1.0 / (0.125 * fs)).exp();
    let (mut s, mut mx) = (0.0f64, 0.0f64);
    for v in x {
        s = a * s + (1.0 - a) * v * v;
        mx = mx.max(s);
    }
    db_spl(mx)
}

/// one-third-octave band levels (centre Hz, dB SPL) from the window's spectrum (rectangular window, Parseval)
pub fn third_octave(p: &[f64], fs: f64) -> Vec<(f64, f64)> {
    let n = p.len();
    if n < 16 {
        return vec![];
    }
    let mut x: Vec<Complex64> = p.iter().map(|v| Complex64::new(*v, 0.0)).collect();
    FftPlanner::<f64>::new().plan_fft_forward(n).process(&mut x);
    let df = fs / n as f64;
    let mut out = vec![];
    for nb in -17..=13 {
        let fc = 1000.0 * 10f64.powf(nb as f64 / 10.0);
        let (lo, hi) = (fc / 2f64.powf(1.0 / 6.0), fc * 2f64.powf(1.0 / 6.0));
        let mut ms = 0.0;
        for j in 1..n / 2 {
            let f = j as f64 * df;
            if f >= lo && f < hi {
                ms += 2.0 * x[j].norm_sqr() / (n as f64 * n as f64);
            }
        }
        out.push((fc, db_spl(ms)));
    }
    out
}

/// strongest spectral lines of the window (Hann window): level, what it is on (resonance / TR harmonic), which axis
#[derive(Clone, Debug)]
pub struct Line {
    pub f: f64,
    pub db_spl: f64,
    pub db_a: f64,
    pub axis: usize,
    /// resonance nearest in half-bandwidths: (axis, f0, Q, offset = (f − f0)/(f0/2Q))
    pub res: (usize, f64, f64, f64),
    /// f · TR when the TR is constant (integer = harmonic of the repetition rate)
    pub harm: Option<f64>,
}

pub fn lines(model: &AcousticModel, sim: &Sim, from: usize, to: usize, tr_period: Option<f64>, max: usize) -> Vec<Line> {
    let n = to - from;
    if n < 64 {
        return vec![];
    }
    let fs = sim.fs;
    let w: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()).collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let spec = |x: &[f64]| -> Vec<f64> {
        let mut v: Vec<Complex64> = x[from..to].iter().zip(&w).map(|(a, b)| Complex64::new(a * b, 0.0)).collect();
        fft.process(&mut v);
        v[..n / 2].iter().map(|c| 4.0 * c.norm() / n as f64 / std::f64::consts::SQRT_2).collect()
    };
    let tot = spec(&sim.p);
    let ax: Vec<Vec<f64>> = (0..3).map(|a| spec(&sim.axis[a])).collect();
    let mx = tot.iter().cloned().fold(0.0, f64::max);
    let mut cand: Vec<usize> = (2..tot.len() - 1).filter(|&j| tot[j] > tot[j - 1] && tot[j] >= tot[j + 1] && tot[j] > mx * 0.01).collect();
    cand.sort_by(|a, b| tot[*b].partial_cmp(&tot[*a]).unwrap());
    let mut chosen: Vec<usize> = vec![];
    for j in cand {
        if chosen.iter().all(|c| (*c as isize - j as isize).abs() > 4) {
            chosen.push(j);
            if chosen.len() >= max {
                break;
            }
        }
    }
    chosen
        .into_iter()
        .map(|j| {
            let f = j as f64 * fs / n as f64;
            let db = db_spl(tot[j] * tot[j]);
            let axis = (0..3).max_by(|a, b| ax[*a][j].partial_cmp(&ax[*b][j]).unwrap()).unwrap();
            let mut res = (0, 0.0, 1.0, f64::INFINITY);
            for a in 0..3 {
                for m in model.axes[a].iter().filter(|m| m.q >= 5.0) {
                    let off = (f - m.f0) / (m.f0 / (2.0 * m.q));
                    if off.abs() < res.3.abs() {
                        res = (a, m.f0, m.q, off);
                    }
                }
            }
            Line { f, db_spl: db, db_a: db + 20.0 * a_weight(f).log10(), axis, res, harm: tr_period.map(|t| f * t) }
        })
        .collect()
}

// ───────────────────────────────────────────────────────────────────────────── sequence → TRs

#[derive(Clone, Debug)]
pub struct Tr {
    pub b0: usize,
    pub b1: usize,
    pub t0: f64,
    pub t1: f64,
}

/// split into TRs at every `rf_per_tr`-th RF block (1: each excitation; 2: spin echo with its refocusing pulse);
/// `tr_ms` is the fallback period for sequences without RF
pub fn tr_split(sf: &SeqFile, tr_ms: Option<f64>, rf_per_tr: usize) -> Result<Vec<Tr>, String> {
    let mut t = 0.0;
    let starts: Vec<f64> = sf.blocks.iter().map(|b| { let s = t; t += b.0; s }).collect();
    let total = t;
    let n = sf.blocks.len();
    let rf: Vec<usize> = (0..n).filter(|&i| sf.blocks[i].1[0] != 0).step_by(rf_per_tr.max(1)).collect();
    let mut cuts: Vec<usize> = if rf.len() >= 2 {
        rf
    } else if let Some(ms) = tr_ms {
        let per = ms * 1e-3;
        let mut c = vec![];
        let mut last = -1i64;
        for i in 0..n {
            let k = (starts[i] / per + 1e-9).floor() as i64;
            if k != last {
                c.push(i);
                last = k;
            }
        }
        c
    } else {
        return Err("fewer than 2 RF blocks: cannot find TRs, give --tr-ms".into());
    };
    cuts.dedup();
    let mut trs = vec![];
    for (i, &b) in cuts.iter().enumerate() {
        let e = cuts.get(i + 1).copied().unwrap_or(n);
        trs.push(Tr { b0: b, b1: e, t0: starts[b], t1: if e < n { starts[e] } else { total } });
    }
    Ok(trs)
}

/// gradients (physical axes, rotations applied) of blocks `b0..b1`, `t0` = start time of block `b0`
pub fn grad_range(sf: &mut SeqFile, b0: usize, b1: usize, t_start: f64) -> Grad {
    let all = std::mem::take(&mut sf.blocks);
    sf.blocks = all[b0..b1].to_vec();
    let mut g = sf.grad_raster(GAMMA_BAR);
    sf.blocks = all;
    g.t0 = t_start;
    g
}

// ───────────────────────────────────────────────────────────────────────────── TR signatures and events

/// octave-band dB SPL (7) + dB(A) total of the sound of this TR repeated periodically
pub fn tr_signature(model: &AcousticModel, g: &Grad, planner: &mut FftPlanner<f64>) -> [f64; NSIG] {
    let n = g.len().saturating_sub(1);
    let floor = 0.0;
    if n < 8 {
        return [floor; NSIG];
    }
    let fft: std::sync::Arc<dyn Fft<f64>> = planner.plan_fft_forward(n);
    let mut spec: Vec<Vec<Complex64>> = vec![];
    for a in 0..3 {
        let mut x: Vec<Complex64> = (0..n).map(|i| Complex64::new(g.g[i][a], 0.0)).collect();
        fft.process(&mut x);
        spec.push(x);
    }
    let mut ms = [0.0f64; NSIG];
    for j in 1..n / 2 {
        let f = j as f64 / (n as f64 * g.dt);
        let s: Complex64 = (0..3).map(|a| spec[a][j] * model.h(a, f)).sum();
        let e = 2.0 * s.norm_sqr() / (n as f64 * n as f64);
        let b = BANDS.iter().position(|c| f < c * std::f64::consts::SQRT_2).unwrap_or(BANDS.len() - 1);
        ms[b] += e;
        let aw = a_weight(f);
        ms[NSIG - 1] += e * aw * aw;
    }
    ms.map(|v| db_spl(v).max(floor))
}

#[derive(Clone, Debug)]
pub struct Event {
    pub tr: usize,
    pub kind: &'static str,
    pub score: f64,
    pub la_before: f64,
    pub la_after: f64,
    /// signature column that changed most (0..6 octave band, 7 total dB(A))
    pub col: usize,
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn col_median(sig: &[[f64; NSIG]], lo: usize, hi: usize) -> [f64; NSIG] {
    let mut o = [0.0; NSIG];
    for c in 0..NSIG {
        let mut v: Vec<f64> = sig[lo..hi].iter().map(|r| r[c]).collect();
        o[c] = median(&mut v);
    }
    o
}

/// sound changes: jump of the median signature over the `w` TRs after vs before each TR (largest band change, dB,
/// among bands within 25 dB of the loudest band); returns the start, the loudest stretch and up to `max` − 2 change
/// points above `thr_db`, ordered in time
pub fn detect_events(sig: &[[f64; NSIG]], w: usize, thr_db: f64, max: usize) -> Vec<Event> {
    let n = sig.len();
    let mut ev: Vec<Event> = vec![];
    if n == 0 {
        return ev;
    }
    let w = w.max(2);
    let mut score = vec![0.0; n];
    let mut info = vec![(0.0, 0.0, 0usize); n];
    if n >= 2 * w {
        for k in w..=(n - w) {
            let (b, a) = (col_median(sig, k - w, k), col_median(sig, k, k + w));
            let top = (0..7).map(|c| b[c].max(a[c])).fold(0.0, f64::max);
            let (mut s, mut col) = (0.0, 7);
            for c in 0..NSIG {
                if c == 7 || b[c].max(a[c]) >= top - 25.0 {
                    let d = (a[c] - b[c]).abs();
                    if d > s {
                        s = d;
                        col = c;
                    }
                }
            }
            score[k] = s;
            info[k] = (b[7], a[7], col);
        }
    }
    let mut order: Vec<usize> = (0..n).filter(|&k| score[k] >= thr_db).collect();
    order.sort_by(|a, b| score[*b].partial_cmp(&score[*a]).unwrap());
    let mut acc: Vec<usize> = vec![];
    for k in order {
        // a step gives a plateau of equal scores (the median windows straddle it): put the event at its middle
        let (mut lo, mut hi) = (k, k);
        while lo > 0 && score[lo - 1] >= 0.98 * score[k] {
            lo -= 1;
        }
        while hi + 1 < n && score[hi + 1] >= 0.98 * score[k] {
            hi += 1;
        }
        let k = (lo + hi) / 2;
        if acc.iter().all(|c| (*c as isize - k as isize).unsigned_abs() > w) {
            acc.push(k);
        }
    }
    acc.truncate(max.saturating_sub(2));
    for k in acc {
        ev.push(Event { tr: k, kind: "change", score: score[k], la_before: info[k].0, la_after: info[k].1, col: info[k].2 });
    }
    // loudest stretch
    let hw = (w / 2).max(1);
    let la = |k: usize| col_median(sig, k.saturating_sub(hw), (k + hw).min(n))[7];
    let loud = (0..n).step_by(hw.max(1)).max_by(|a, b| la(*a).partial_cmp(&la(*b)).unwrap()).unwrap_or(0);
    let near = |ev: &Vec<Event>, k: usize| ev.iter().any(|e| (e.tr as isize - k as isize).unsigned_abs() <= w);
    if !near(&ev, loud) && max >= 2 {
        ev.push(Event { tr: loud, kind: "loudest", score: 0.0, la_before: la(loud), la_after: la(loud), col: 7 });
    }
    if !near(&ev, 0) || max == 1 {
        ev.push(Event { tr: 0, kind: "start", score: 0.0, la_before: la(0), la_after: la(0), col: 7 });
    }
    ev.sort_by_key(|e| e.tr);
    ev
}

// ───────────────────────────────────────────────────────────────────────────── files

/// 16-bit PCM mono WAV
pub fn write_wav(path: &Path, x: &[f64], fs: f64, gain: f64) -> std::io::Result<()> {
    let data: Vec<u8> = x.iter().flat_map(|v| (((v * gain).clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect();
    let mut b = Vec::with_capacity(44 + data.len());
    b.extend(b"RIFF");
    b.extend(((36 + data.len()) as u32).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend((fs as u32).to_le_bytes());
    b.extend(((fs as u32) * 2).to_le_bytes());
    b.extend(2u16.to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend((data.len() as u32).to_le_bytes());
    b.extend(data);
    std::fs::write(path, b)
}

/// minimal .npy (float32, C order)
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    let sh = if shape.len() == 1 { format!("({},)", shape[0]) } else { format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")) };
    let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {sh}, }}");
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut b = vec![0x93u8];
    b.extend(b"NUMPY");
    b.extend([1u8, 0]);
    b.extend((h.len() as u16).to_le_bytes());
    b.extend(h.as_bytes());
    b.extend(data.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(path, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_mode(f0: f64, q: f64, peak: f64) -> AcousticModel {
        let mut m = AcousticModel::toy_3t();
        m.axes = [vec![Mode { f0, q, peak }], vec![], vec![]];
        m
    }

    fn sine(f: f64, amp: f64, dur: f64, dt: f64, axis: usize) -> Grad {
        let n = (dur / dt).round() as usize + 1;
        Grad { dt, t0: 0.0, g: (0..n).map(|i| { let mut v = [0.0; 3]; v[axis] = amp * (2.0 * std::f64::consts::PI * f * i as f64 * dt).sin(); v }).collect() }
    }

    #[test]
    fn sine_on_resonance_gives_expected_pressure() {
        // 1 kHz, 10 mT/m, peak 100 Pa/(T/m) → 1 Pa amplitude (× sinc² of the linear interpolation), rms 0.707 Pa
        let m = one_mode(1000.0, 10.0, 100.0);
        let s = simulate(&m, &sine(1000.0, 0.01, 0.5, 10e-6, 0), 0.1).unwrap();
        let mid = &s.p[s.p.len() / 4..3 * s.p.len() / 4];
        let rms = mean_sq(mid).sqrt();
        let expect = 1.0 / 2f64.sqrt() * sinc2(1000.0 * 10e-6);
        assert!((rms / expect - 1.0).abs() < 0.01, "rms {rms} expect {expect}");
        // axis y / z carry nothing
        assert!(mean_sq(&s.axis[1]) < 1e-20);
    }

    #[test]
    fn linear_in_gradient_and_b0() {
        let m = one_mode(1000.0, 10.0, 100.0);
        let g = sine(900.0, 0.01, 0.2, 10e-6, 0);
        let p1 = simulate(&m, &g, 0.1).unwrap().p;
        let mut g2 = g.clone();
        g2.g.iter_mut().for_each(|v| v[0] *= 2.0);
        let p2 = simulate(&m, &g2, 0.1).unwrap().p;
        let mut m2 = m.clone();
        m2.b0 *= 2.0;
        let p3 = simulate(&m2, &g, 0.1).unwrap().p;
        let e = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - 2.0 * y).abs()).fold(0.0, f64::max) / b.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(e(&p2, &p1) < 1e-9 && e(&p3, &p1) < 1e-9);
    }

    #[test]
    fn half_bandwidth_is_minus_3db() {
        let m = one_mode(1000.0, 20.0, 1.0);
        let pk = m.h(0, 1000.0).norm();
        let h = m.h(0, 1000.0 * (1.0 + 1.0 / 40.0)).norm();
        assert!((pk - 1.0).abs() < 1e-12);
        assert!((h / pk - 1.0 / 2f64.sqrt()).abs() < 0.02, "{}", h / pk);
    }

    #[test]
    fn protection_attenuates_and_json_roundtrips() {
        let mut m = AcousticModel::toy_3t();
        let h0 = m.h(0, 1000.0).norm();
        m.use_protection = true;
        let h1 = m.h(0, 1000.0).norm();
        assert!((20.0 * (h0 / h1).log10() - 30.0).abs() < 1e-9);
        let m2 = AcousticModel::from_json(&m.to_json().to_string()).unwrap();
        assert_eq!(m2.axes[2].len(), AcousticModel::toy_3t().axes[2].len());
        assert!((m2.h(1, 700.0) - AcousticModel::toy_3t().h(1, 700.0)).norm() < 1e-12);
    }

    #[test]
    fn axis_delay_is_a_pure_time_shift() {
        let mut m = one_mode(1000.0, 10.0, 100.0);
        let g = sine(900.0, 0.01, 0.3, 10e-6, 0);
        let p0 = simulate(&m, &g, 0.1).unwrap().p;
        m.delay[0] = 0.5e-3; // 24 audio samples
        let p1 = simulate(&m, &g, 0.1).unwrap().p;
        let err = (100..p0.len() - 200).map(|i| (p1[i + 24] - p0[i]).abs()).fold(0.0, f64::max);
        assert!(err < 2e-3 * p0.iter().fold(0.0f64, |a, b| a.max(b.abs())), "{err}");
    }

    #[test]
    fn reverb_adds_a_tail_at_the_requested_level() {
        let mut m = one_mode(1000.0, 10.0, 100.0);
        // a 20 ms burst, then silence: with RT60 0.4 s and -10 dB the tail carries 10 % of the energy
        let mut g = sine(1000.0, 0.01, 0.02, 10e-6, 0);
        g.g.resize(60_000, [0.0; 3]);
        let dry = simulate(&m, &g, 0.1).unwrap().p;
        m.reverb = Some((0.4, -10.0));
        let wet = simulate(&m, &g, 0.1).unwrap().p;
        let (e_dry, e_wet) = (dry.iter().map(|v| v * v).sum::<f64>(), wet.iter().map(|v| v * v).sum::<f64>());
        assert!((e_wet / e_dry - 1.1).abs() < 0.04, "{}", e_wet / e_dry);
        // the wet signal still sounds 0.3 s after the dry one has died away
        let k = (0.35 * FS_AUDIO) as usize;
        assert!(mean_sq(&wet[k..k + 2000]) > 100.0 * mean_sq(&dry[k..k + 2000]).max(1e-30));
    }

    #[test]
    fn a_weighting_is_unity_at_1khz() {
        assert!((a_weight(1000.0) - 1.0).abs() < 2e-3);
        assert!((20.0 * a_weight(100.0).log10() + 19.1).abs() < 0.1);
    }

    #[test]
    fn wav_header_and_npy_header() {
        let d = std::env::temp_dir().join(format!("hasty_acoustic_test_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        write_wav(&d.join("a.wav"), &[0.0, 0.5, -0.5, 1.5], 48000.0, 1.0).unwrap();
        let b = std::fs::read(d.join("a.wav")).unwrap();
        assert_eq!(&b[..4], b"RIFF");
        assert_eq!(b.len(), 44 + 8);
        assert_eq!(i16::from_le_bytes([b[46], b[47]]), 16384);
        assert_eq!(i16::from_le_bytes([b[48], b[49]]), -16384);
        assert_eq!(i16::from_le_bytes([b[50], b[51]]), 32767);
        write_npy_f32(&d.join("a.npy"), &[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let n = std::fs::read(d.join("a.npy")).unwrap();
        assert_eq!((n.len() - 16) % 64, 0, "npy header must be 64-aligned");
        assert_eq!(f32::from_le_bytes(n[n.len() - 4..].try_into().unwrap()), 4.0);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn events_find_a_step() {
        let mut sig = vec![[60.0; NSIG]; 200];
        for r in sig.iter_mut().skip(100) {
            r[3] = 75.0;
            r[7] = 80.0;
        }
        let ev = detect_events(&sig, 16, 3.0, 6);
        assert!(ev.iter().any(|e| e.kind == "change" && (e.tr as i64 - 100).abs() <= 2), "{ev:?}");
        assert!(ev.iter().any(|e| e.kind == "start"));
    }

    #[test]
    fn smooth_sizes() {
        assert_eq!(smooth(7), 8);
        assert_eq!(smooth(121), 125);
        assert_eq!(smooth(250), 250);
    }
}

