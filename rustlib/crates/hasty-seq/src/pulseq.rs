//! Pulseq `.seq` writer (open file format for MR sequences, https://pulseq.github.io/) and a reader for checks.
//!
//! Format 1.5.0 by default (the layout of pypulseq 1.5 / MATLAB Pulseq 1.5: RF with `center` and `use`, arbitrary
//! gradients with `first` / `last`, ADC with ppm offsets and a phase-shape id), or 1.4.1 ([`Sequence::version14`]).
//! Sections: `[VERSION] [DEFINITIONS] [BLOCKS] [RF] [GRADIENTS] [TRAP] [ADC] [EXTENSIONS]` + `extension LABELSET /
//! LABELINC / ROTATIONS` + `[SHAPES]` + `[SIGNATURE]` (md5 of the file above it, as pypulseq).
//!
//! Units at the API are SI: T/m, s, Hz (RF amplitude = γ̄B1 in Hz). The file stores gradients in Hz/m (× `gamma`),
//! times in µs (block durations in units of `BlockDurationRaster`, ADC dwell in ns).
//!
//! Gradient events:
//! - [`GradEv::Trap`]: trapezoid (rise / flat / fall on the gradient raster).
//! - [`GradEv::Pl`]: a piecewise-linear waveform through corner samples g_i at `delay + i·raster` (what
//!   [`crate::wave::Grad`] means by a raster waveform), written as an arbitrary gradient with a time shape 0, 1, …, n−1
//!   (an "extended trapezoid"); first = g_0, last = g_{n−1}. No half-raster ambiguity: the file states the times.
//! - [`GradEv::Arb`]: Pulseq's default arbitrary gradient (samples at the raster *centres* (i + ½)·raster, time shape 0).
//!
//! Every event and shape is de-duplicated by its exact text representation (two events that print the same are one
//! library entry). Shapes are run-length compressed on the derivative exactly as pypulseq's `compress_shape` (quantum
//! 1e-7 of the normalised shape) when that is shorter, else stored verbatim with 9 significant digits.
//!
//! Rotations: `Block::rotation` (unit quaternion w, x, y, z; physical = R · logical) is written as the `ROTATIONS`
//! extension of Pulseq 1.5.1 ("id RotQuat0 RotQuatX RotQuatY RotQuatZ") [CHECK the exact layout against the
//! MATLAB 1.5.1 writer; pypulseq 1.5.0 cannot read it]. With `expand_rotations` the rotated gradients are baked into
//! per-axis waveforms instead (readable by every reader, more shapes).
//!
//! GE / pge2 notes (HarmonizedMRI design rules, https://github.com/HarmonizedMRI/SequenceExamples-GE): gradient and
//! block raster 4 µs, RF raster 2 µs, ADC dwell a multiple of 2 µs; segments are delimited by `TRID` labels and only
//! amplitudes / phases / frequencies may change between instances of a segment; gradients start and end at 0 in every
//! segment; amplitudes must not be exactly 0 (use a tiny value); the interpreter inserts 117 µs at the end of every
//! segment; RF dead 72 µs / ringdown 54 µs, ADC dead 40 µs. [`Sequence::check`] enforces the raster and dead-time rules.

use std::collections::HashMap;
use std::path::Path;

use num_complex::Complex64;

use crate::wave::Grad;
use crate::{quat_to_mat, Vec3};

/// Pulseq's γ̄ (Hz/T) for T/m ↔ Hz/m (pypulseq default 42.576 MHz/T)
pub const PULSEQ_GAMMA: f64 = 42.576e6;

#[derive(Clone, Debug)]
pub struct PulseqSystem {
    pub grad_raster: f64,
    pub rf_raster: f64,
    pub adc_raster: f64,
    pub block_raster: f64,
    /// Hz/T used for the T/m → Hz/m conversion
    pub gamma: f64,
    pub b0: f64,
    /// limits for [`Sequence::check`] (per axis, physical)
    pub gmax: f64,
    pub smax: f64,
    pub rf_dead: f64,
    pub rf_ringdown: f64,
    pub adc_dead: f64,
}

impl PulseqSystem {
    /// GE via pge2 (design rules above); gmax / smax of a SIGNA Premier (80 mT/m, 200 T/m/s; TOPPE's header lists the
    /// HRMW coil at 70 mT/m [CHECK]); RF dead / ringdown with margin over 72 / 54 µs
    pub fn ge_pge2(b0: f64) -> Self {
        Self { grad_raster: 4e-6, rf_raster: 2e-6, adc_raster: 2e-6, block_raster: 4e-6, gamma: PULSEQ_GAMMA, b0, gmax: 80e-3, smax: 200.0, rf_dead: 100e-6, rf_ringdown: 60e-6, adc_dead: 40e-6 }
    }
    /// Siemens defaults (10 µs gradient / block raster, 1 µs RF, 0.1 µs ADC)
    pub fn siemens(b0: f64) -> Self {
        Self { grad_raster: 10e-6, rf_raster: 1e-6, adc_raster: 0.1e-6, block_raster: 10e-6, gamma: PULSEQ_GAMMA, b0, gmax: 80e-3, smax: 200.0, rf_dead: 100e-6, rf_ringdown: 30e-6, adc_dead: 10e-6 }
    }
}

#[derive(Clone, Debug)]
pub enum GradEv {
    /// amplitude T/m, times s
    Trap { amp: f64, rise: f64, flat: f64, fall: f64, delay: f64 },
    /// corner samples (T/m) at delay + i·raster, piecewise linear between them
    Pl { g: Vec<f64>, delay: f64 },
    /// raster-centre samples (T/m) at delay + (i + ½)·raster; first / last = values at the shape's start / end
    Arb { g: Vec<f64>, first: f64, last: f64, delay: f64 },
}

impl GradEv {
    pub fn duration(&self, raster: f64) -> f64 {
        match self {
            GradEv::Trap { rise, flat, fall, delay, .. } => delay + rise + flat + fall,
            GradEv::Pl { g, delay } => delay + (g.len().saturating_sub(1)) as f64 * raster,
            GradEv::Arb { g, delay, .. } => delay + g.len() as f64 * raster,
        }
    }
    /// the same event × s
    pub fn scaled(&self, s: f64) -> GradEv {
        match self {
            GradEv::Trap { amp, rise, flat, fall, delay } => GradEv::Trap { amp: amp * s, rise: *rise, flat: *flat, fall: *fall, delay: *delay },
            GradEv::Pl { g, delay } => GradEv::Pl { g: g.iter().map(|x| x * s).collect(), delay: *delay },
            GradEv::Arb { g, first, last, delay } => GradEv::Arb { g: g.iter().map(|x| x * s).collect(), first: first * s, last: last * s, delay: *delay },
        }
    }
    /// breakpoints (t, g) of the piecewise-linear waveform, t from the block start
    pub fn points(&self, raster: f64) -> Vec<(f64, f64)> {
        match self {
            GradEv::Trap { amp, rise, flat, fall, delay } => {
                let mut v = vec![(*delay, 0.0), (delay + rise, *amp)];
                if *flat > 0.0 {
                    v.push((delay + rise + flat, *amp));
                }
                v.push((delay + rise + flat + fall, 0.0));
                v
            }
            GradEv::Pl { g, delay } => g.iter().enumerate().map(|(i, &x)| (delay + i as f64 * raster, x)).collect(),
            GradEv::Arb { g, first, last, delay } => {
                let mut v = vec![(*delay, *first)];
                v.extend(g.iter().enumerate().map(|(i, &x)| (delay + (i as f64 + 0.5) * raster, x)));
                v.push((delay + g.len() as f64 * raster, *last));
                v
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct RfEv {
    /// complex B1 waveform in Hz (γ̄B1)
    pub signal: Vec<Complex64>,
    /// sample times from the event start (s); None = (i + ½)·rf_raster
    pub t: Option<Vec<f64>>,
    /// time of the pulse centre from the event start (s)
    pub center: f64,
    pub delay: f64,
    pub freq: f64,
    pub phase: f64,
    /// 'e' excitation, 'r' refocusing, 'i' inversion, 's' saturation, 'p' preparation, 'o' other, 'u' undefined
    pub use_: char,
}

impl RfEv {
    pub fn duration(&self, raster: f64) -> f64 {
        match &self.t {
            None => self.delay + self.signal.len() as f64 * raster,
            Some(t) => self.delay + t.last().copied().unwrap_or(0.0) + 0.5 * raster,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdcEv {
    pub num: usize,
    pub dwell: f64,
    pub delay: f64,
    pub freq: f64,
    pub phase: f64,
}

impl AdcEv {
    pub fn duration(&self) -> f64 {
        self.delay + self.num as f64 * self.dwell
    }
    /// sample times from the block start (Pulseq: sample i at delay + (i + ½)·dwell)
    pub fn times(&self) -> Vec<f64> {
        (0..self.num).map(|i| self.delay + (i as f64 + 0.5) * self.dwell).collect()
    }
}

#[derive(Clone, Debug)]
pub struct Label {
    /// true: LABELSET, false: LABELINC
    pub set: bool,
    pub name: String,
    pub value: i64,
}

impl Label {
    pub fn set(name: &str, value: i64) -> Self {
        Self { set: true, name: name.into(), value }
    }
    pub fn inc(name: &str, value: i64) -> Self {
        Self { set: false, name: name.into(), value }
    }
}

/// one block; `duration` None = the longest event, rounded up to the block raster
#[derive(Clone, Debug, Default)]
pub struct Block {
    pub duration: Option<f64>,
    pub rf: Option<RfEv>,
    pub g: [Option<GradEv>; 3],
    pub adc: Option<AdcEv>,
    pub labels: Vec<Label>,
    /// unit quaternion (w, x, y, z): the block's gradients are logical, physical = R · logical
    pub rotation: Option<[f64; 4]>,
}

impl Block {
    pub fn delay(d: f64) -> Self {
        Self { duration: Some(d), ..Default::default() }
    }
}

/// a de-duplicating library: text line (without id) → id (1-based), in insertion order
#[derive(Clone, Debug, Default)]
struct Lib {
    map: HashMap<String, usize>,
    rows: Vec<String>,
}

impl Lib {
    fn id(&mut self, line: String) -> usize {
        if let Some(&i) = self.map.get(&line) {
            return i;
        }
        self.rows.push(line.clone());
        let i = self.rows.len();
        self.map.insert(line, i);
        i
    }
}

/// `%.{sig}g` as C / Python print it (shortest of fixed / exponent notation, trailing zeros removed)
pub fn fmt_g(x: f64, sig: usize) -> String {
    if x == 0.0 || !x.is_finite() {
        return if x.is_finite() { "0".into() } else { format!("{x}") };
    }
    let s = format!("{:.*e}", sig - 1, x);
    let (mant, exp) = s.split_once('e').unwrap();
    let e: i32 = exp.parse().unwrap();
    if e < -4 || e >= sig as i32 {
        let m = if mant.contains('.') { mant.trim_end_matches('0').trim_end_matches('.') } else { mant };
        format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
    } else {
        let dec = (sig as i32 - 1 - e).max(0) as usize;
        let f = format!("{:.*}", dec, x);
        if f.contains('.') {
            f.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            f
        }
    }
}

/// pypulseq `compress_shape`: run-length code of the quantised derivative (values v, v, n−2 for runs), or the samples
/// themselves when that is not shorter (and always for ≤ 4 samples)
pub fn compress_shape(w: &[f64]) -> Vec<f64> {
    if w.len() <= 4 {
        return w.to_vec();
    }
    let q = 1e-7;
    let sc: Vec<f64> = w.iter().map(|x| x / q).collect();
    let mut datq = Vec::with_capacity(w.len());
    datq.push(sc[0].round());
    for i in 1..sc.len() {
        datq.push((sc[i] - sc[i - 1]).round());
    }
    // error correction: the cumulative sum of the coded derivative stays within ½ quantum of the samples
    let mut cs = 0.0;
    let mut qerr_r = Vec::with_capacity(w.len());
    for i in 0..sc.len() {
        cs += datq[i];
        qerr_r.push((sc[i] - cs).round());
    }
    let mut datd = datq.clone();
    for i in 1..datd.len() {
        datd[i] += qerr_r[i] - qerr_r[i - 1];
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < datd.len() {
        let mut j = i + 1;
        while j < datd.len() && datd[j] == datd[i] {
            j += 1;
        }
        let v = datd[i] * q;
        let len = j - i;
        if len > 1 {
            out.extend_from_slice(&[v, v, (len - 2) as f64]);
        } else {
            out.push(v);
        }
        i = j;
    }
    if out.len() < w.len() {
        out
    } else {
        w.to_vec()
    }
}

/// pypulseq `decompress_shape`
pub fn decompress_shape(data: &[f64], num: usize) -> Vec<f64> {
    if data.len() == num {
        return data.to_vec();
    }
    let mut d = Vec::with_capacity(num);
    let mut i = 0;
    while i < data.len() {
        if i + 2 < data.len() && data[i] == data[i + 1] {
            let rep = data[i + 2].round() as usize + 2;
            for _ in 0..rep {
                d.push(data[i]);
            }
            i += 3;
        } else {
            d.push(data[i]);
            i += 1;
        }
    }
    let mut s = 0.0;
    d.iter()
        .map(|x| {
            s += x;
            s
        })
        .collect()
}

/// MD5 (RFC 1321) hex digest
pub fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let (mut a0, mut b0, mut c0, mut d0) = (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_le_bytes());
    for chunk in msg.chunks(64) {
        let m: Vec<u32> = (0..16).map(|i| u32::from_le_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]])).collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f2 = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f2.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    [a0, b0, c0, d0].iter().flat_map(|x| x.to_le_bytes()).map(|b| format!("{b:02x}")).collect()
}

/// decoded gradient library entry (waveform reconstruction without keeping the blocks)
#[derive(Clone, Debug)]
enum GDec {
    Trap { amp: f64, rise: f64, flat: f64, fall: f64, delay: f64 },
    Pl { amp: f64, shape: usize, delay: f64 },
    Arb { amp: f64, first: f64, last: f64, shape: usize, delay: f64 },
}

/// shapes de-duplicated by the exact bits of the normalised samples; text rendered once per shape
#[derive(Clone, Debug, Default)]
struct ShapeLib {
    map: HashMap<Vec<u64>, usize>,
    ramp: HashMap<usize, usize>,
    data: Vec<Vec<f64>>,
    text: Vec<String>,
}

impl ShapeLib {
    fn id(&mut self, w: &[f64]) -> usize {
        let key: Vec<u64> = w.iter().map(|x| x.to_bits()).collect();
        if let Some(&i) = self.map.get(&key) {
            return i;
        }
        let c = compress_shape(w);
        let mut txt = format!("num_samples {}\n", w.len());
        for x in &c {
            txt.push_str(&fmt_g(*x, 9));
            txt.push('\n');
        }
        self.data.push(decompress_shape(&c, w.len()));
        self.text.push(txt);
        let i = self.data.len();
        self.map.insert(key, i);
        i
    }
    /// the time shape 0, 1, …, n−1
    fn ramp(&mut self, n: usize) -> usize {
        if let Some(&i) = self.ramp.get(&n) {
            return i;
        }
        let i = self.id(&(0..n).map(|j| j as f64).collect::<Vec<_>>());
        self.ramp.insert(n, i);
        i
    }
}

/// a sequence being built: every block is encoded into de-duplicated event / shape libraries when it is added (so a
/// long heating train costs one row per block), with decoded copies for [`Sequence::grad_raster`] / `adc_times`.
/// Set `version` and `expand_rotations` before adding blocks.
#[derive(Clone, Debug)]
pub struct Sequence {
    pub sys: PulseqSystem,
    /// (major, minor, revision) of the written format: (1, 5, 0) or (1, 4, 1)
    pub version: (u32, u32, u32),
    /// extra [DEFINITIONS] (the rasters are always written)
    pub defs: Vec<(String, String)>,
    /// bake rotations into per-axis waveforms instead of writing the ROTATIONS extension
    pub expand_rotations: bool,
    /// duration of each block (s, on the block raster)
    pub durations: Vec<f64>,
    rows: Vec<[usize; 6]>,
    rot_of: Vec<Option<[f64; 4]>>,
    shapes: ShapeLib,
    rf: Lib,
    grads: Lib,
    gdec: Vec<GDec>,
    adc: Lib,
    adec: Vec<AdcEv>,
    lset: Lib,
    linc: Lib,
    rots: Lib,
    ext: Lib,
    problems: Vec<String>,
}

const EXT_LABELSET: usize = 1;
const EXT_LABELINC: usize = 2;
const EXT_ROTATIONS: usize = 3;

impl Sequence {
    pub fn new(sys: PulseqSystem) -> Self {
        Self { sys, version: (1, 5, 0), defs: vec![], expand_rotations: false, durations: vec![], rows: vec![], rot_of: vec![], shapes: ShapeLib::default(), rf: Lib::default(), grads: Lib::default(), gdec: vec![], adc: Lib::default(), adec: vec![], lset: Lib::default(), linc: Lib::default(), rots: Lib::default(), ext: Lib::default(), problems: vec![] }
    }
    pub fn version14(mut self) -> Self {
        assert!(self.rows.is_empty(), "set the version before adding blocks");
        self.version = (1, 4, 1);
        self
    }
    pub fn def(&mut self, key: &str, value: &str) {
        self.defs.push((key.into(), value.into()));
    }
    pub fn duration(&self) -> f64 {
        self.durations.iter().sum()
    }
    pub fn n_blocks(&self) -> usize {
        self.rows.len()
    }
    pub fn n_shapes(&self) -> usize {
        self.shapes.data.len()
    }

    fn check_block(&mut self, i: usize, b: &Block, dur: f64) {
        if self.problems.len() > 200 {
            return;
        }
        let s = &self.sys;
        let on = |x: f64, r: f64| ((x / r) - (x / r).round()).abs() < 1e-6;
        let mut err = vec![];
        if !on(dur, s.block_raster) {
            err.push(format!("block {i}: duration {dur} not on the block raster"));
        }
        for (a, g) in b.g.iter().enumerate() {
            let Some(g) = g else { continue };
            let (delay, times) = match g {
                GradEv::Trap { delay, rise, flat, fall, amp } => {
                    if amp.abs() > s.gmax * (1.0 + 1e-9) || (*rise > 0.0 && amp.abs() / rise > s.smax * (1.0 + 1e-6)) || (*fall > 0.0 && amp.abs() / fall > s.smax * (1.0 + 1e-6)) {
                        err.push(format!("block {i} axis {a}: trapezoid over the limits"));
                    }
                    (*delay, vec![*rise, *flat, *fall])
                }
                GradEv::Pl { delay, g } | GradEv::Arb { delay, g, .. } => {
                    let gm = g.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                    let sm = g.windows(2).fold(0.0f64, |m, w| m.max((w[1] - w[0]).abs() / s.grad_raster));
                    if gm > s.gmax * (1.0 + 1e-9) || sm > s.smax * (1.0 + 1e-6) {
                        err.push(format!("block {i} axis {a}: waveform over the limits (|g| {gm:.4}, slew {sm:.1})"));
                    }
                    (*delay, vec![])
                }
            };
            if !on(delay, s.grad_raster) || times.iter().any(|&t| !on(t, s.grad_raster)) {
                err.push(format!("block {i} axis {a}: gradient times not on the gradient raster"));
            }
            if g.duration(s.grad_raster) > dur + 1e-12 {
                err.push(format!("block {i} axis {a}: gradient longer than the block"));
            }
        }
        if let Some(rf) = &b.rf {
            if rf.delay + 1e-12 < s.rf_dead || !on(rf.delay, s.rf_raster) {
                err.push(format!("block {i}: RF delay {} < dead time or off the RF raster", rf.delay));
            }
            if rf.duration(s.rf_raster) + s.rf_ringdown > dur + 1e-12 {
                err.push(format!("block {i}: RF ringdown beyond the block"));
            }
        }
        if let Some(adc) = &b.adc {
            if adc.delay + 1e-12 < s.adc_dead || !on(adc.dwell, s.adc_raster) || !on(adc.delay * 1e6, 1.0) {
                err.push(format!("block {i}: ADC delay {} < dead time, dwell {} off the ADC raster or delay not integer µs", adc.delay, adc.dwell));
            }
            if adc.duration() + s.adc_dead > dur + 1e-12 {
                err.push(format!("block {i}: ADC (+ dead time) longer than the block"));
            }
        }
        self.problems.extend(err);
    }

    /// append a block (encoded at once); returns its duration
    pub fn add(&mut self, b: Block) -> f64 {
        let s = self.sys.clone();
        let mut d: f64 = 0.0;
        for g in b.g.iter().flatten() {
            d = d.max(g.duration(s.grad_raster));
        }
        if let Some(rf) = &b.rf {
            d = d.max(rf.duration(s.rf_raster) + s.rf_ringdown);
        }
        if let Some(adc) = &b.adc {
            // Pulseq convention (pypulseq check_timing): the ADC dead time also follows the last sample
            d = d.max(adc.duration() + s.adc_dead);
        }
        let dur = match b.duration {
            Some(x) => {
                assert!(x + 1e-12 >= d, "block duration {x} shorter than its events {d}");
                x
            }
            None => d,
        };
        let dur = (dur / s.block_raster - 1e-6).ceil().max(0.0) * s.block_raster;
        let i = self.rows.len();
        self.check_block(i, &b, dur);
        let v15 = self.version.1 >= 5;
        let us = |x: f64| fmt_g((x * 1e6 * 1e3).round() / 1e3, 9);
        let bake = self.expand_rotations && b.rotation.is_some();
        let gev: [Option<GradEv>; 3] = if bake {
            let m = quat_to_mat(b.rotation.unwrap());
            let pts = block_points_of(&b, s.grad_raster);
            [0, 1, 2].map(|a| {
                let used = (0..3).any(|l| b.g[l].is_some() && m[a][l].abs() > 1e-15);
                if !used {
                    return None;
                }
                let t1 = pts[a].last().map(|p| p.0).unwrap_or(0.0);
                let n = (t1 / s.grad_raster).round() as usize;
                Some(GradEv::Pl { g: (0..=n).map(|j| interp_pts(&pts[a], j as f64 * s.grad_raster)).collect(), delay: 0.0 })
            })
        } else {
            b.g.clone()
        };
        let mut gid = [0usize; 3];
        for a in 0..3 {
            let Some(g) = &gev[a] else { continue };
            let (line, dec) = match g {
                GradEv::Trap { amp, rise, flat, fall, delay } => (format!("t {} {} {} {} {}", fmt_g(amp * s.gamma, 9), us(*rise), us(*flat), us(*fall), us(*delay)), GDec::Trap { amp: *amp, rise: *rise, flat: *flat, fall: *fall, delay: *delay }),
                GradEv::Pl { g, delay } => {
                    let amp = signed_max(g);
                    let w: Vec<f64> = g.iter().map(|x| if amp != 0.0 { x / amp } else { *x }).collect();
                    let sid = self.shapes.id(&w);
                    let tid = self.shapes.ramp(g.len());
                    let (first, last) = (g[0] * s.gamma, g[g.len() - 1] * s.gamma);
                    let line = if v15 {
                        format!("g {} {} {} {} {} {}", fmt_g(amp * s.gamma, 9), fmt_g(first, 9), fmt_g(last, 9), sid, tid, us(*delay))
                    } else {
                        format!("g {} {} {} {}", fmt_g(amp * s.gamma, 9), sid, tid, us(*delay))
                    };
                    (line, GDec::Pl { amp, shape: sid, delay: *delay })
                }
                GradEv::Arb { g, first, last, delay } => {
                    let amp = signed_max(g);
                    let w: Vec<f64> = g.iter().map(|x| if amp != 0.0 { x / amp } else { *x }).collect();
                    let sid = self.shapes.id(&w);
                    let line = if v15 {
                        format!("g {} {} {} {} 0 {}", fmt_g(amp * s.gamma, 9), fmt_g(first * s.gamma, 9), fmt_g(last * s.gamma, 9), sid, us(*delay))
                    } else {
                        format!("g {} {} 0 {}", fmt_g(amp * s.gamma, 9), sid, us(*delay))
                    };
                    (line, GDec::Arb { amp, first: *first, last: *last, shape: sid, delay: *delay })
                }
            };
            let id = self.grads.id(line);
            if id > self.gdec.len() {
                self.gdec.push(dec);
            }
            gid[a] = id;
        }
        let rid = match &b.rf {
            None => 0,
            Some(r) => {
                let amp = r.signal.iter().fold(0.0f64, |m, c| m.max(c.norm()));
                let mag: Vec<f64> = r.signal.iter().map(|c| if amp > 0.0 { c.norm() / amp } else { 0.0 }).collect();
                let ph: Vec<f64> = r
                    .signal
                    .iter()
                    .map(|c| {
                        let mut p = c.arg();
                        if p < 0.0 {
                            p += 2.0 * std::f64::consts::PI;
                        }
                        p / (2.0 * std::f64::consts::PI)
                    })
                    .collect();
                let mid = self.shapes.id(&mag);
                let pid = self.shapes.id(&ph);
                let tid = match &r.t {
                    None => 0,
                    Some(t) => self.shapes.id(&t.iter().map(|x| x / s.rf_raster).collect::<Vec<_>>()),
                };
                if v15 {
                    self.rf.id(format!("{} {} {} {} {} {} 0 0 {} {} {}", fmt_g(amp, 9), mid, pid, tid, us(r.center), us(r.delay), fmt_g(r.freq, 9), fmt_g(r.phase, 9), r.use_))
                } else {
                    self.rf.id(format!("{} {} {} {} {} {} {}", fmt_g(amp, 9), mid, pid, tid, us(r.delay), fmt_g(r.freq, 9), fmt_g(r.phase, 9)))
                }
            }
        };
        let aid = match &b.adc {
            None => 0,
            Some(a) => {
                let dwell_ns = (a.dwell * 1e9).round() as i64;
                let delay_us = (a.delay * 1e6).round() as i64;
                let id = if v15 {
                    self.adc.id(format!("{} {} {} 0 0 {} {} 0", a.num, dwell_ns, delay_us, fmt_g(a.freq, 9), fmt_g(a.phase, 9)))
                } else {
                    self.adc.id(format!("{} {} {} {} {}", a.num, dwell_ns, delay_us, fmt_g(a.freq, 9), fmt_g(a.phase, 9)))
                };
                if id > self.adec.len() {
                    self.adec.push(AdcEv { num: a.num, dwell: dwell_ns as f64 * 1e-9, delay: delay_us as f64 * 1e-6, freq: a.freq, phase: a.phase });
                }
                id
            }
        };
        let mut items: Vec<(usize, usize)> = vec![];
        for l in &b.labels {
            if l.set {
                items.push((EXT_LABELSET, self.lset.id(format!("{} {}", l.value, l.name))));
            } else {
                items.push((EXT_LABELINC, self.linc.id(format!("{} {}", l.value, l.name))));
            }
        }
        if let (Some(q), false) = (b.rotation, bake) {
            items.push((EXT_ROTATIONS, self.rots.id(format!("{} {} {} {}", fmt_g(q[0], 9), fmt_g(q[1], 9), fmt_g(q[2], 9), fmt_g(q[3], 9)))));
        }
        let mut next = 0usize;
        for &(ty, rf_) in items.iter().rev() {
            next = self.ext.id(format!("{ty} {rf_} {next}"));
        }
        self.rows.push([rid, gid[0], gid[1], gid[2], aid, next]);
        self.rot_of.push(if bake { None } else { b.rotation });
        self.durations.push(dur);
        dur
    }

    /// raster / dead-time / limit problems found while adding (empty = ok; capped at ~200)
    pub fn check(&self) -> Vec<String> {
        self.problems.clone()
    }

    /// physical-axis breakpoints of block `i` (decoded events, rotation applied)
    fn block_points(&self, i: usize) -> [Vec<(f64, f64)>; 3] {
        let r = self.sys.grad_raster;
        let row = self.rows[i];
        let logical: [Vec<(f64, f64)>; 3] = [0, 1, 2].map(|a| {
            let id = row[1 + a];
            if id == 0 {
                return vec![];
            }
            match &self.gdec[id - 1] {
                GDec::Trap { amp, rise, flat, fall, delay } => GradEv::Trap { amp: *amp, rise: *rise, flat: *flat, fall: *fall, delay: *delay }.points(r),
                GDec::Pl { amp, shape, delay } => self.shapes.data[shape - 1].iter().enumerate().map(|(j, &x)| (delay + j as f64 * r, x * amp)).collect(),
                GDec::Arb { amp, first, last, shape, delay } => {
                    let w = &self.shapes.data[shape - 1];
                    let mut v = vec![(*delay, *first)];
                    v.extend(w.iter().enumerate().map(|(j, &x)| (delay + (j as f64 + 0.5) * r, x * amp)));
                    v.push((delay + w.len() as f64 * r, *last));
                    v
                }
            }
        });
        match self.rot_of[i] {
            None => logical,
            Some(q) => rotate_points(&logical, q),
        }
    }

    /// the whole sequence's physical gradients as corner samples on the gradient raster (piecewise linear), from t = 0
    pub fn grad_raster(&self) -> Grad {
        let r = self.sys.grad_raster;
        let total = self.duration();
        let n = (total / r).round() as usize + 1;
        let mut g = vec![[0.0; 3]; n];
        let mut t0 = 0.0;
        for i in 0..self.rows.len() {
            let pts = self.block_points(i);
            let i0 = (t0 / r).round() as usize;
            let nb = (self.durations[i] / r).round() as usize;
            for a in 0..3 {
                if pts[a].is_empty() {
                    continue;
                }
                for j in 0..=nb {
                    let v = interp_pts(&pts[a], j as f64 * r);
                    if j == 0 && i0 > 0 && v == 0.0 {
                        continue;
                    }
                    g[i0 + j][a] = v;
                }
            }
            t0 += self.durations[i];
        }
        Grad { dt: r, t0: 0.0, g }
    }

    /// absolute ADC sample times (s from the sequence start), one vector per ADC event in order
    pub fn adc_times(&self) -> Vec<Vec<f64>> {
        let mut out = vec![];
        let mut t0 = 0.0;
        for (i, row) in self.rows.iter().enumerate() {
            if row[4] != 0 {
                out.push(self.adec[row[4] - 1].times().iter().map(|t| t + t0).collect());
            }
            t0 += self.durations[i];
        }
        out
    }

    /// block start times (s)
    pub fn block_starts(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.rows.len());
        let mut t = 0.0;
        for d in &self.durations {
            v.push(t);
            t += d;
        }
        v
    }

    /// render the file text (with signature)
    pub fn to_text(&self) -> String {
        let s = &self.sys;
        let v15 = self.version.1 >= 5;
        let mut o = String::with_capacity(64 * self.rows.len() + self.shapes.text.iter().map(|t| t.len()).sum::<usize>() + 4096);
        o.push_str("# Pulseq sequence file\n# Created by hasty-seq (pulseq.rs)\n\n");
        o.push_str(&format!("[VERSION]\nmajor {}\nminor {}\nrevision {}\n\n", self.version.0, self.version.1, self.version.2));
        let mut defs: Vec<(String, String)> = vec![
            ("AdcRasterTime".into(), fmt_g(s.adc_raster, 9)),
            ("BlockDurationRaster".into(), fmt_g(s.block_raster, 9)),
            ("GradientRasterTime".into(), fmt_g(s.grad_raster, 9)),
            ("RadiofrequencyRasterTime".into(), fmt_g(s.rf_raster, 9)),
            ("TotalDuration".into(), fmt_g(self.duration(), 9)),
        ];
        if v15 {
            defs.push(("B0".into(), fmt_g(s.b0, 9)));
        }
        for (k, v) in &self.defs {
            defs.retain(|(kk, _)| kk != k);
            defs.push((k.clone(), v.clone()));
        }
        defs.sort_by(|a, b| a.0.cmp(&b.0));
        o.push_str("[DEFINITIONS]\n");
        for (k, v) in &defs {
            o.push_str(&format!("{k} {v} \n"));
        }
        o.push('\n');
        o.push_str("# Format of blocks:\n# NUM DUR RF  GX  GY  GZ  ADC  EXT\n[BLOCKS]\n");
        let w = self.rows.len().to_string().len();
        for (i, r) in self.rows.iter().enumerate() {
            let dur = (self.durations[i] / s.block_raster).round() as u64;
            o.push_str(&format!("{:>w$} {} {:>3} {:>3} {:>3} {:>3} {:>2} {:>2}\n", i + 1, dur, r[0], r[1], r[2], r[3], r[4], r[5]));
        }
        o.push('\n');
        let sect = |o: &mut String, head: &str, lib: &Lib| {
            if lib.rows.is_empty() {
                return;
            }
            o.push_str(head);
            for (i, r) in lib.rows.iter().enumerate() {
                o.push_str(&format!("{} {r}\n", i + 1));
            }
            o.push('\n');
        };
        if v15 {
            sect(&mut o, "# Format of RF events:\n# id ampl. mag_id phase_id time_shape_id center delay freqPPM phasePPM freq phase use\n# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   Hz   rad  ..\n[RF]\n", &self.rf);
        } else {
            sect(&mut o, "# Format of RF events:\n# id amplitude mag_id phase_id time_shape_id delay freq phase\n# ..        Hz   ....     ....          ....    us   Hz   rad\n[RF]\n", &self.rf);
        }
        let arb: Vec<(usize, &String)> = self.grads.rows.iter().enumerate().filter(|(_, r)| r.starts_with('g')).map(|(i, r)| (i + 1, r)).collect();
        let trap: Vec<(usize, &String)> = self.grads.rows.iter().enumerate().filter(|(_, r)| r.starts_with('t')).map(|(i, r)| (i + 1, r)).collect();
        if !arb.is_empty() {
            o.push_str("# Format of arbitrary gradients:\n#   time_shape_id of 0 means default timing (stepping with grad_raster starting at 1/2 of grad_raster)\n");
            o.push_str(if v15 { "# id amplitude first last amp_shape_id time_shape_id delay\n# ..      Hz/m  Hz/m Hz/m        ..         ..          us\n" } else { "# id amplitude amp_shape_id time_shape_id delay\n# ..      Hz/m       ..         ..          us\n" });
            o.push_str("[GRADIENTS]\n");
            for (i, r) in arb {
                o.push_str(&format!("{i} {}\n", &r[2..]));
            }
            o.push('\n');
        }
        if !trap.is_empty() {
            o.push_str("# Format of trapezoid gradients:\n# id amplitude rise flat fall delay\n# ..      Hz/m   us   us   us    us\n[TRAP]\n");
            for (i, r) in trap {
                o.push_str(&format!("{i} {}\n", &r[2..]));
            }
            o.push('\n');
        }
        if v15 {
            sect(&mut o, "# Format of ADC events:\n# id num dwell delay freqPPM phasePPM freq phase phase_id\n# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n[ADC]\n", &self.adc);
        } else {
            sect(&mut o, "# Format of ADC events:\n# id num dwell delay freq phase\n# ..  ..    ns    us   Hz   rad\n[ADC]\n", &self.adc);
        }
        sect(&mut o, "# Format of extension lists:\n# id type ref next_id\n# next_id of 0 terminates the list\n# Extension list is followed by extension specifications\n[EXTENSIONS]\n", &self.ext);
        sect(&mut o, &format!("# Extension specification for setting labels:\n# id set labelstring\nextension LABELSET {EXT_LABELSET}\n"), &self.lset);
        sect(&mut o, &format!("# Extension specification for setting labels:\n# id set labelstring\nextension LABELINC {EXT_LABELINC}\n"), &self.linc);
        sect(&mut o, &format!("# Extension specification for rotation events:\n# id RotQuat0 RotQuatX RotQuatY RotQuatZ\nextension ROTATIONS {EXT_ROTATIONS}\n"), &self.rots);
        if !self.shapes.text.is_empty() {
            o.push_str("# Sequence Shapes\n[SHAPES]\n\n");
            for (i, r) in self.shapes.text.iter().enumerate() {
                o.push_str(&format!("shape_id {}\n{r}\n", i + 1));
            }
        }
        let md5 = md5_hex(o.as_bytes());
        o.push_str("\n[SIGNATURE]\n# This is the hash of the Pulseq file, calculated right before the [SIGNATURE] section was added\n# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be stripped away for recalculating/verification)\n");
        o.push_str(&format!("Type md5\nHash {md5}\n"));
        o
    }

    pub fn write(&self, path: &Path) -> std::io::Result<String> {
        let t = self.to_text();
        std::fs::write(path, &t)?;
        Ok(t)
    }
}

/// breakpoints of a block's (logical) events, rotation applied
fn block_points_of(b: &Block, r: f64) -> [Vec<(f64, f64)>; 3] {
    let logical: [Vec<(f64, f64)>; 3] = [0, 1, 2].map(|a| b.g[a].as_ref().map(|g| g.points(r)).unwrap_or_default());
    match b.rotation {
        None => logical,
        Some(q) => rotate_points(&logical, q),
    }
}

/// rotate per-axis breakpoints: every logical axis sampled on the union of breakpoints (exact for PL events)
fn rotate_points(logical: &[Vec<(f64, f64)>; 3], q: [f64; 4]) -> [Vec<(f64, f64)>; 3] {
    let m = quat_to_mat(q);
    let mut ts: Vec<f64> = logical.iter().flat_map(|v| v.iter().map(|p| p.0)).collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    [0, 1, 2].map(|o| ts.iter().map(|&t| (t, (0..3).map(|l| m[o][l] * interp_pts(&logical[l], t)).sum())).collect())
}

fn signed_max(g: &[f64]) -> f64 {
    let (mut m, mut v) = (0.0f64, 0.0);
    for &x in g {
        if x.abs() > m {
            m = x.abs();
            v = x;
        }
    }
    v
}

/// piecewise-linear interpolation of breakpoints (zero outside)
fn interp_pts(v: &[(f64, f64)], t: f64) -> f64 {
    if v.is_empty() || t < v[0].0 - 1e-12 || t > v[v.len() - 1].0 + 1e-12 {
        return 0.0;
    }
    let j = v.partition_point(|p| p.0 <= t + 1e-12);
    if j == 0 {
        return v[0].1;
    }
    if j >= v.len() {
        return v[v.len() - 1].1;
    }
    let (a, b) = (v[j - 1], v[j]);
    if b.0 - a.0 < 1e-15 {
        return b.1;
    }
    a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0)
}

// ───────────────────────────────────────────────────────────── reader (for round-trip checks)

/// a parsed .seq file: definitions, per-block durations and events, decompressed shapes
#[derive(Clone, Debug, Default)]
pub struct SeqFile {
    pub version: (u32, u32, u32),
    pub defs: HashMap<String, String>,
    /// per block: duration (s), rf, gx, gy, gz, adc, ext ids
    pub blocks: Vec<(f64, [usize; 6])>,
    pub rf: HashMap<usize, Vec<f64>>,
    pub grad: HashMap<usize, (char, Vec<f64>)>,
    pub adc: HashMap<usize, Vec<f64>>,
    pub ext: HashMap<usize, [usize; 3]>,
    pub ext_types: HashMap<usize, String>,
    pub ext_data: HashMap<(usize, usize), Vec<String>>,
    pub shapes: HashMap<usize, Vec<f64>>,
    pub signature_ok: bool,
}

impl SeqFile {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut f = SeqFile::default();
        let body_end = text.find("\n[SIGNATURE]").unwrap_or(text.len());
        let mut sect = String::new();
        let mut shape: Option<(usize, usize, Vec<f64>)> = None;
        let mut ext_cur: Option<usize> = None;
        for raw in text.lines() {
            let line = raw.trim();
            if line.starts_with('#') {
                continue;
            }
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') {
                if let Some((id, n, d)) = shape.take() {
                    f.shapes.insert(id, decompress_shape(&d, n));
                }
                sect = line.to_string();
                ext_cur = None;
                continue;
            }
            if let Some(r) = line.strip_prefix("extension ") {
                let mut it = r.split_whitespace();
                let name = it.next().unwrap_or("").to_string();
                let id: usize = it.next().and_then(|x| x.parse().ok()).ok_or("bad extension line")?;
                f.ext_types.insert(id, name);
                ext_cur = Some(id);
                sect = "ext".into();
                continue;
            }
            let tok: Vec<&str> = line.split_whitespace().collect();
            let num = |i: usize| -> Result<f64, String> { tok.get(i).ok_or(format!("short line {line}"))?.parse::<f64>().map_err(|e| format!("{e}: {line}")) };
            match sect.as_str() {
                "[VERSION]" => {
                    let v: u32 = tok[1].parse().map_err(|_| "version")?;
                    match tok[0] {
                        "major" => f.version.0 = v,
                        "minor" => f.version.1 = v,
                        "revision" => f.version.2 = v,
                        _ => {}
                    }
                }
                "[DEFINITIONS]" => {
                    f.defs.insert(tok[0].to_string(), tok[1..].join(" "));
                }
                "[BLOCKS]" => {
                    let br: f64 = f.defs.get("BlockDurationRaster").and_then(|x| x.parse().ok()).unwrap_or(10e-6);
                    let ids = [2, 3, 4, 5, 6, 7].map(|i| tok[i].parse::<usize>().unwrap_or(0));
                    f.blocks.push((num(1)? * br, ids));
                }
                "[RF]" => {
                    f.rf.insert(num(0)? as usize, tok[1..].iter().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect());
                }
                "[GRADIENTS]" => {
                    f.grad.insert(num(0)? as usize, ('g', tok[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect()));
                }
                "[TRAP]" => {
                    f.grad.insert(num(0)? as usize, ('t', tok[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect()));
                }
                "[ADC]" => {
                    f.adc.insert(num(0)? as usize, tok[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect());
                }
                "[EXTENSIONS]" => {
                    f.ext.insert(num(0)? as usize, [num(1)? as usize, num(2)? as usize, num(3)? as usize]);
                }
                "ext" => {
                    let e = ext_cur.unwrap();
                    f.ext_data.insert((e, num(0)? as usize), tok[1..].iter().map(|s| s.to_string()).collect());
                }
                "[SHAPES]" => {
                    if tok[0] == "shape_id" {
                        if let Some((id, n, d)) = shape.take() {
                            f.shapes.insert(id, decompress_shape(&d, n));
                        }
                        shape = Some((num(1)? as usize, 0, vec![]));
                    } else if tok[0] == "num_samples" {
                        shape.as_mut().unwrap().1 = num(1)? as usize;
                    } else if let Some(s) = shape.as_mut() {
                        s.2.push(num(0)?);
                    }
                }
                "[SIGNATURE]" => {
                    if tok[0] == "Hash" {
                        f.signature_ok = md5_hex(text[..body_end].as_bytes()) == tok[1];
                    }
                }
                _ => {}
            }
        }
        if let Some((id, n, d)) = shape.take() {
            f.shapes.insert(id, decompress_shape(&d, n));
        }
        Ok(f)
    }

    fn def_f(&self, k: &str, d: f64) -> f64 {
        self.defs.get(k).and_then(|x| x.split_whitespace().next()?.parse().ok()).unwrap_or(d)
    }

    /// physical gradients (T/m with the given γ̄) as corner samples on the gradient raster, from t = 0
    pub fn grad_raster(&self, gamma: f64) -> Grad {
        let r = self.def_f("GradientRasterTime", 10e-6);
        let v15 = self.version.1 >= 5;
        let total: f64 = self.blocks.iter().map(|b| b.0).sum();
        let n = (total / r).round() as usize + 1;
        let mut out = vec![[0.0; 3]; n];
        let mut t0 = 0.0;
        for (dur, ids) in &self.blocks {
            let mut pts: [Vec<(f64, f64)>; 3] = Default::default();
            for a in 0..3 {
                let id = ids[1 + a];
                if id == 0 {
                    continue;
                }
                let (ty, d) = &self.grad[&id];
                pts[a] = if *ty == 't' {
                    let (amp, rise, flat, fall, delay) = (d[0] / gamma, d[1] * 1e-6, d[2] * 1e-6, d[3] * 1e-6, d[4] * 1e-6);
                    GradEv::Trap { amp, rise, flat, fall, delay }.points(r)
                } else {
                    let (amp, first, last, sid, tid, delay) = if v15 { (d[0], d[1], d[2], d[3] as usize, d[4] as i64, d[5] * 1e-6) } else { (d[0], f64::NAN, f64::NAN, d[1] as usize, d[2] as i64, d[3] * 1e-6) };
                    let w: Vec<f64> = self.shapes[&sid].iter().map(|x| x * amp / gamma).collect();
                    if tid > 0 {
                        let ts = &self.shapes[&(tid as usize)];
                        ts.iter().zip(&w).map(|(t, g)| (delay + t * r, *g)).collect()
                    } else {
                        let (f0, l0) = if v15 { (first / gamma, last / gamma) } else { (0.0, 0.0) };
                        let mut v = vec![(delay, f0)];
                        v.extend(w.iter().enumerate().map(|(i, &g)| (delay + (i as f64 + 0.5) * r, g)));
                        v.push((delay + w.len() as f64 * r, l0));
                        v
                    }
                };
            }
            // rotation extension
            let mut rot: Option<[f64; 4]> = None;
            let mut e = ids[5];
            while e != 0 {
                let [ty, rf_, next] = self.ext[&e];
                if self.ext_types.get(&ty).map(|s| s.as_str()) == Some("ROTATIONS") {
                    let d = &self.ext_data[&(ty, rf_)];
                    let q: Vec<f64> = d.iter().map(|x| x.parse().unwrap()).collect();
                    rot = Some([q[0], q[1], q[2], q[3]]);
                }
                e = next;
            }
            let i0 = (t0 / r).round() as usize;
            let nb = (dur / r).round() as usize;
            let m = rot.map(quat_to_mat);
            for j in 0..=nb {
                let lg: Vec3 = [0, 1, 2].map(|a| interp_pts(&pts[a], j as f64 * r));
                let pg = match m {
                    None => lg,
                    Some(m) => crate::mat_vec(&m, &lg),
                };
                for a in 0..3 {
                    if (pts[a].is_empty() && m.is_none()) || (j == 0 && i0 > 0 && pg[a] == 0.0) {
                        continue;
                    }
                    out[i0 + j][a] = pg[a];
                }
            }
            t0 += dur;
        }
        Grad { dt: r, t0: 0.0, g: out }
    }

    /// absolute ADC sample times (s), one vector per ADC event
    pub fn adc_times(&self) -> Vec<Vec<f64>> {
        let mut out = vec![];
        let mut t0 = 0.0;
        for (dur, ids) in &self.blocks {
            if ids[4] != 0 {
                let d = &self.adc[&ids[4]];
                let (num, dwell, delay) = (d[0] as usize, d[1] * 1e-9, d[2] * 1e-6);
                out.push((0..num).map(|i| t0 + delay + (i as f64 + 0.5) * dwell).collect());
            }
            t0 += dur;
        }
        out
    }

    /// labels set in each block (name → value), in block order
    pub fn labels(&self) -> Vec<Vec<(String, i64, bool)>> {
        self.blocks
            .iter()
            .map(|(_, ids)| {
                let mut v = vec![];
                let mut e = ids[5];
                while e != 0 {
                    let [ty, rf_, next] = self.ext[&e];
                    let name = self.ext_types.get(&ty).cloned().unwrap_or_default();
                    if name == "LABELSET" || name == "LABELINC" {
                        let d = &self.ext_data[&(ty, rf_)];
                        v.push((d[1].clone(), d[0].parse().unwrap_or(0), name == "LABELSET"));
                    }
                    e = next;
                }
                v
            })
            .collect()
    }
}

/// a thin-slice excitation: Hann-windowed sinc of time-bandwidth `tbw`, duration `t`, flip `flip` (rad), on the RF
/// raster; returns the RF event (delay `delay`, frequency `freq`) and the slice-select amplitude (T/m) for `thk`
pub fn sinc_pulse(sys: &PulseqSystem, flip: f64, t: f64, tbw: f64, thk: f64, delay: f64, freq: f64, phase: f64) -> (RfEv, f64) {
    let n = (t / sys.rf_raster).round() as usize;
    let bw = tbw / t;
    let tt: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) * sys.rf_raster - 0.5 * t).collect();
    let sinc = |x: f64| if x.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
    let w: Vec<f64> = tt.iter().map(|&x| sinc(bw * x) * 0.5 * (1.0 + (2.0 * std::f64::consts::PI * x / t).cos())).collect();
    let area: f64 = w.iter().sum::<f64>() * sys.rf_raster;
    let amp = flip / (2.0 * std::f64::consts::PI * area);
    let g = bw / (sys.gamma * thk);
    (RfEv { signal: w.iter().map(|&x| Complex64::new(x * amp, 0.0)).collect(), t: None, center: 0.5 * t, delay, freq, phase, use_: 'e' }, g)
}

/// a hard (block) pulse of duration `t` and flip `flip`
pub fn block_pulse(sys: &PulseqSystem, flip: f64, t: f64, delay: f64, use_: char) -> RfEv {
    let n = (t / sys.rf_raster).round() as usize;
    let amp = flip / (2.0 * std::f64::consts::PI * t);
    RfEv { signal: vec![Complex64::new(amp, 0.0); n], t: None, center: 0.5 * t, delay, freq: 0.0, phase: 0.0, use_ }
}

/// trapezoid of `area` (T·s/m) at `amp_max` / `slew`, times rounded up to the gradient raster (area kept exactly by
/// lowering the amplitude)
pub fn trap_area(sys: &PulseqSystem, area: f64, amp_max: f64, slew: f64, delay: f64) -> GradEv {
    let r = sys.grad_raster;
    let a = area.abs();
    let ceil = |x: f64| (x / r - 1e-9).ceil().max(1.0) * r;
    let gp = (a * slew).sqrt().min(amp_max);
    let rise = ceil(gp / slew);
    let flat = ((a / gp - rise) / r - 1e-9).ceil().max(0.0) * r;
    let amp = a / (rise + flat);
    GradEv::Trap { amp: amp * area.signum(), rise, flat, fall: rise, delay }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md5_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(md5_hex(b"The quick brown fox jumps over the lazy dog"), "9e107d9d372bb6826bd81d3542a419d6");
        let long = vec![b'a'; 1000];
        assert_eq!(md5_hex(&long), "cabe45dcc9ae5b66ba86600cca6b8ba8");
    }

    #[test]
    fn g_format_like_c() {
        assert_eq!(fmt_g(0.0, 9), "0");
        assert_eq!(fmt_g(1.0, 9), "1");
        assert_eq!(fmt_g(4e-6, 9), "4e-06");
        assert_eq!(fmt_g(1e-5, 9), "1e-05");
        assert_eq!(fmt_g(0.0001, 9), "0.0001");
        assert_eq!(fmt_g(123456.789, 9), "123456.789");
        assert_eq!(fmt_g(-2.5, 6), "-2.5");
        assert_eq!(fmt_g(1234567890.0, 9), "1.23456789e+09");
        assert_eq!(fmt_g(0.1 + 0.2, 9), "0.3");
    }

    #[test]
    fn shape_compression_round_trip() {
        let ramp: Vec<f64> = (0..200).map(|i| i as f64 * 0.005).collect();
        let c = compress_shape(&ramp);
        assert!(c.len() < 10, "{c:?}");
        let d = decompress_shape(&c, ramp.len());
        for (a, b) in ramp.iter().zip(&d) {
            assert!((a - b).abs() < 1e-7, "{a} {b}");
        }
        let chirp: Vec<f64> = (0..500).map(|i| (0.01 * (i * i) as f64).sin()).collect();
        let c2 = compress_shape(&chirp);
        assert_eq!(c2.len(), chirp.len()); // stored verbatim
        let times: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let ct = compress_shape(&times);
        assert_eq!(ct, vec![0.0, 1.0, 1.0, 997.0]);
        assert_eq!(decompress_shape(&ct, 1000), times);
    }

    fn demo(sys: &PulseqSystem, v14: bool, expand: bool) -> Sequence {
        let mut s = Sequence::new(sys.clone());
        if v14 {
            s = s.version14();
        }
        s.expand_rotations = expand;
        let (rf, gss) = sinc_pulse(sys, 0.5 * std::f64::consts::PI, 3e-3, 4.0, 1.5e-3, sys.rf_dead, 1234.5, 0.0);
        let r = sys.grad_raster;
        let ramp = (gss / 100.0 / r).ceil() * r;
        let flat = rf.duration(sys.rf_raster) - sys.rf_dead + sys.rf_ringdown;
        let flat = (flat / r).ceil() * r;
        s.add(Block { rf: Some(RfEv { delay: sys.rf_dead + ramp, ..rf }), g: [None, None, Some(GradEv::Trap { amp: gss, rise: ramp, flat, fall: ramp, delay: sys.rf_dead })], labels: vec![Label::set("TRID", 1)], ..Default::default() });
        let nc = (8e-3 / r).round() as usize;
        let chirp: Vec<f64> = (0..nc).map(|i| { let t = i as f64 * r; 0.01 * (2.0 * std::f64::consts::PI * (200.0 + 1e5 * t) * t).sin() * (std::f64::consts::PI * i as f64 / (nc - 1) as f64).sin() }).collect();
        s.add(Block { g: [Some(GradEv::Pl { g: chirp, delay: 0.0 }), Some(trap_area(sys, 2e-6, 0.03, 120.0, 40e-6)), None], adc: Some(AdcEv { num: 1900, dwell: 4e-6, delay: 40e-6, freq: 0.0, phase: 0.3 }), labels: vec![Label::inc("LIN", 1)], ..Default::default() });
        let arb: Vec<f64> = (0..300).map(|i| 0.01 * (i as f64 / 299.0 * std::f64::consts::PI).sin()).collect();
        s.add(Block { g: [None, Some(GradEv::Arb { g: arb, first: 0.0, last: 0.0, delay: 2.0 * r }), None], ..Default::default() });
        let pl: Vec<f64> = (0..101).map(|i| 0.015 * (1.0 - ((i as f64 - 50.0) / 50.0).abs())).collect();
        s.add(Block { g: [Some(GradEv::Pl { g: pl.clone(), delay: 0.0 }), Some(GradEv::Pl { g: pl.iter().map(|x| -0.5 * x).collect(), delay: 0.0 }), None], rotation: Some([0.9, 0.1, -0.3, 0.2]), ..Default::default() });
        s.add(Block::delay(1e-3));
        s
    }

    #[test]
    fn write_read_round_trip() {
        for sys in [PulseqSystem::ge_pge2(3.0), PulseqSystem::siemens(2.89)] {
            for v14 in [false, true] {
                for expand in [false, true] {
                    let s = demo(&sys, v14, expand);
                    let errs = s.check();
                    assert!(errs.is_empty(), "{errs:?}");
                    let txt = s.to_text();
                    let f = SeqFile::parse(&txt).unwrap();
                    assert!(f.signature_ok, "signature");
                    assert_eq!(f.blocks.len(), s.n_blocks());
                    for (a, b) in f.blocks.iter().zip(&s.durations) {
                        assert!((a.0 - b).abs() < 1e-12);
                    }
                    let gw = s.grad_raster();
                    let gr = f.grad_raster(sys.gamma);
                    assert_eq!(gw.len(), gr.len());
                    let gm = gw.g.iter().flat_map(|v| v.iter()).fold(0.0f64, |m, x| m.max(x.abs()));
                    let e = gw.g.iter().zip(&gr.g).flat_map(|(a, b)| (0..3).map(move |q| (a[q] - b[q]).abs())).fold(0.0f64, f64::max);
                    // the Arb block's raster-corner values are interpolations of centre samples: identical in both
                    assert!(e < 2e-7 * gm, "v14 {v14} expand {expand}: max |Δg| {e} of {gm}");
                    let (ta, tb) = (s.adc_times(), f.adc_times());
                    assert_eq!(ta.len(), tb.len());
                    for (x, y) in ta.iter().zip(&tb) {
                        assert!(x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-12));
                    }
                    let l = f.labels();
                    assert_eq!(l[0], vec![("TRID".to_string(), 1, true)]);
                    assert_eq!(l[1], vec![("LIN".to_string(), 1, false)]);
                }
            }
        }
    }

    #[test]
    fn dedup_and_trap_area() {
        let sys = PulseqSystem::ge_pge2(3.0);
        let mut s = Sequence::new(sys.clone());
        for _ in 0..5 {
            s.add(Block { g: [Some(trap_area(&sys, 1e-5, 0.04, 150.0, 0.0)), None, None], ..Default::default() });
        }
        let t = s.to_text();
        let f = SeqFile::parse(&t).unwrap();
        assert_eq!(f.grad.len(), 1);
        if let GradEv::Trap { amp, rise, flat, .. } = trap_area(&sys, 1e-5, 0.04, 150.0, 0.0) {
            assert!(((rise + flat) * amp - 1e-5).abs() < 1e-15);
            assert!(amp / rise <= 150.0 + 1e-9);
        }
    }
}
