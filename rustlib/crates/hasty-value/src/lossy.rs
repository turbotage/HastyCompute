//! Controlled-loss storage for floating-point tensors. Data-agnostic: the spec knows dtype and values only (no axes,
//! coils, readouts); the *scope* of a scale is the chunk (chunk shape comes from the [`crate::format::ChunkPlan`]) or
//! the whole tensor.
//!
//! ```text
//!   u = T(x)                       transfer: Linear | Asinh(k)  (u = k·asinh(x/k): linear below k, logarithmic above)
//!   q = round(u / step)            step from the loss spec (below), scale statistics over the scope
//!   z = zigzag(q)                  small magnitudes → small unsigned numbers (high byte planes become zero)
//!   stored as 1, 2 or 4 byte words → the usual block pipeline (byte shuffle + lz4 / zstd)
//! ```
//!
//! | loss | step | guarantee (in `u`-space) |
//! |---|---|---|
//! | `Abs(Δ)` | Δ | \|err\| ≤ Δ/2 |
//! | `RelMax(B)` | max\|u\| / (2^(B−1) − 1) | B-bit signed range of the scope |
//! | `RelRms(ε)` | ε·√12·rms(u) | rms err ≈ ε·rms(u) |
//! | `Mantissa(m)` | — | keeps m mantissa bits, pointwise relative error ≤ 2^−(m+1); the result is still an ordinary float tensor (readable by every decoder, no extra format) |
//!
//! A chunk whose spec cannot be met (non-finite values, > 32 bits needed) is stored losslessly; the chunk header says so.
//! Quantised tensors are chunked tensors with tag 10 (`FORMAT.md`).

use rayon::prelude::*;

use crate::{DType, Error, Result, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transfer {
    Linear,
    /// `u = k·asinh(x/k)`, k > 0: absolute resolution below `k`, relative resolution above
    Asinh(f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    Lossless,
    /// absolute step in transfer space
    Abs(f64),
    /// signed integers of this many bits (2..=32) spanning ±max of the scope
    RelMax(u32),
    /// quantisation error rms = this fraction of the scope's rms
    RelRms(f64),
    /// keep this many mantissa bits (≤ 23 for float32 / complex64, ≤ 52 for float64 / complex128)
    Mantissa(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// scale statistics per chunk (local dynamic range)
    Chunk,
    /// one scale for the whole tensor
    Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quant {
    pub loss: Loss,
    pub transfer: Transfer,
    pub scope: Scope,
}

/// loss spec for a tensor at a slice path (as [`crate::format::ChunkPlan`]); None = lossless
pub type LossPlan<'a> = &'a dyn Fn(&str, &crate::Tensor) -> Option<Quant>;

impl Quant {
    pub fn new(loss: Loss) -> Self {
        Self { loss, transfer: Transfer::Linear, scope: Scope::Chunk }
    }
    pub fn transfer(mut self, t: Transfer) -> Self {
        self.transfer = t;
        self
    }
    pub fn scope(mut self, s: Scope) -> Self {
        self.scope = s;
        self
    }

    /// integer-quantising loss (tag 10 chunks)
    pub(crate) fn is_int(&self) -> bool {
        matches!(self.loss, Loss::Abs(_) | Loss::RelMax(_) | Loss::RelRms(_))
    }

    pub fn validate(&self, dtype: DType) -> Result<()> {
        if matches!(self.loss, Loss::Lossless) {
            return Ok(());
        }
        let Some((is64, _)) = scalar_kind(dtype) else {
            return Err(Error::Type(format!("lossy storage needs a float32/64 or complex dtype, not {}", dtype.name())));
        };
        let bad = |m: &str| Err(Error::Type(format!("loss spec: {m}")));
        match self.loss {
            Loss::Abs(d) if !(d.is_finite() && d > 0.0) => return bad("Abs step must be > 0"),
            Loss::RelMax(b) if !(2..=32).contains(&b) => return bad("RelMax bits must be 2..=32"),
            Loss::RelRms(e) if !(e.is_finite() && e > 0.0) => return bad("RelRms must be > 0"),
            Loss::Mantissa(m) if m > if is64 { 52 } else { 23 } => return bad("too many mantissa bits for dtype"),
            _ => {}
        }
        if let Transfer::Asinh(k) = self.transfer {
            if !(k.is_finite() && k > 0.0) {
                return bad("Asinh knee must be > 0");
            }
        }
        if matches!(self.loss, Loss::Mantissa(_)) && self.transfer != Transfer::Linear {
            return bad("Mantissa has no transfer");
        }
        Ok(())
    }
}

/// (is f64, scalars per element) for float dtypes
pub(crate) fn scalar_kind(d: DType) -> Option<(bool, usize)> {
    match d {
        DType::F32 => Some((false, 1)),
        DType::C64 => Some((false, 2)),
        DType::F64 => Some((true, 1)),
        DType::C128 => Some((true, 2)),
        _ => None,
    }
}

/// scalar access for the two float widths
trait Sc {
    const SZ: usize;
    fn rd(b: &[u8]) -> f64;
    fn wr(v: f64, b: &mut [u8]);
}

struct S32;
struct S64;

impl Sc for S32 {
    const SZ: usize = 4;
    #[inline(always)]
    fn rd(b: &[u8]) -> f64 {
        f32::from_le_bytes(b.try_into().unwrap()) as f64
    }
    #[inline(always)]
    fn wr(v: f64, b: &mut [u8]) {
        b.copy_from_slice(&(v as f32).to_le_bytes());
    }
}

impl Sc for S64 {
    const SZ: usize = 8;
    #[inline(always)]
    fn rd(b: &[u8]) -> f64 {
        f64::from_le_bytes(b.try_into().unwrap())
    }
    #[inline(always)]
    fn wr(v: f64, b: &mut [u8]) {
        b.copy_from_slice(&v.to_le_bytes());
    }
}

/// transfer function with the variant fixed at compile time (the hot loops are monomorphised on it)
#[inline(always)]
fn fwd<const LIN: bool>(knee: f64, x: f64) -> f64 {
    if LIN {
        x
    } else {
        knee * (x / knee).asinh()
    }
}

#[inline(always)]
fn inv<const LIN: bool>(knee: f64, u: f64) -> f64 {
    if LIN {
        u
    } else {
        knee * (u / knee).sinh()
    }
}

/// round half away from zero without a libm call (baseline x86_64 has no roundsd)
#[inline(always)]
fn rnd(v: f64) -> i64 {
    (v + 0.5f64.copysign(v)) as i64
}

fn knee_of(t: Transfer) -> f64 {
    match t {
        Transfer::Linear => 1.0,
        Transfer::Asinh(k) => k,
    }
}

/// scale statistics of u = T(x) over a scope
#[derive(Clone, Copy, Debug)]
pub(crate) struct Stats {
    max: f64,
    ss: f64,
    n: usize,
    finite: bool,
}

impl Stats {
    const ZERO: Stats = Stats { max: 0.0, ss: 0.0, n: 0, finite: true };
    fn merge(a: Stats, b: Stats) -> Stats {
        Stats { max: a.max.max(b.max), ss: a.ss + b.ss, n: a.n + b.n, finite: a.finite && b.finite }
    }
}

fn stats_g<S: Sc, const LIN: bool>(raw: &[u8], knee: f64) -> Stats {
    let (mut mx, mut ss, mut fin) = (0.0f64, 0.0f64, true);
    for c in raw.chunks_exact(S::SZ) {
        let u = fwd::<LIN>(knee, S::rd(c));
        fin &= u.is_finite();
        mx = mx.max(u.abs());
        ss += u * u;
    }
    Stats { max: mx, ss, n: raw.len() / S::SZ, finite: fin }
}

fn stats_of(t: Transfer, dtype: DType, raw: &[u8]) -> Stats {
    let (k, is64) = (knee_of(t), scalar_kind(dtype).unwrap().0);
    match (is64, t == Transfer::Linear) {
        (false, true) => stats_g::<S32, true>(raw, k),
        (false, false) => stats_g::<S32, false>(raw, k),
        (true, true) => stats_g::<S64, true>(raw, k),
        (true, false) => stats_g::<S64, false>(raw, k),
    }
}

/// statistics of a whole tensor (parallel; for [`Scope::Tensor`])
pub(crate) fn tensor_stats(q: &Quant, t: &Tensor) -> Stats {
    let es = t.dtype().size();
    let step = (1usize << 20) / es * es;
    t.bytes().par_chunks(step.max(es)).map(|c| stats_of(q.transfer, t.dtype(), c)).reduce(|| Stats::ZERO, Stats::merge)
}

/// chunk prefix: qw u8 (0 = chunk stored losslessly), transfer u8 (0 linear, 1 asinh), pad u16, step f64, knee f64
pub(crate) const PREFIX: usize = 20;

#[inline(always)]
fn zigzag(q: i64) -> u64 {
    ((q << 1) ^ (q >> 63)) as u64
}

#[inline(always)]
fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

fn quant_loop<S: Sc, const LIN: bool, const QW: usize>(raw: &[u8], out: &mut [u8], knee: f64, inv_step: f64) {
    for (x, o) in raw.chunks_exact(S::SZ).zip(out.chunks_exact_mut(QW)) {
        let z = zigzag(rnd(fwd::<LIN>(knee, S::rd(x)) * inv_step));
        o.copy_from_slice(&z.to_le_bytes()[..QW]);
    }
}

fn quant_dispatch<S: Sc>(lin: bool, qw: usize, raw: &[u8], out: &mut [u8], knee: f64, inv_step: f64) {
    match (lin, qw) {
        (true, 1) => quant_loop::<S, true, 1>(raw, out, knee, inv_step),
        (true, 2) => quant_loop::<S, true, 2>(raw, out, knee, inv_step),
        (true, _) => quant_loop::<S, true, 4>(raw, out, knee, inv_step),
        (false, 1) => quant_loop::<S, false, 1>(raw, out, knee, inv_step),
        (false, 2) => quant_loop::<S, false, 2>(raw, out, knee, inv_step),
        (false, _) => quant_loop::<S, false, 4>(raw, out, knee, inv_step),
    }
}

/// Quantise one chunk's raw bytes: returns (prefix + payload, shuffle word); the payload is blocked by the caller.
pub(crate) fn encode_chunk(q: &Quant, dtype: DType, raw: &[u8], global: Option<&Stats>) -> (Vec<u8>, usize) {
    let fallback = || {
        let mut v = vec![0u8; PREFIX];
        v.extend_from_slice(raw);
        (v, dtype.word())
    };
    let is64 = scalar_kind(dtype).unwrap().0;
    let loc = stats_of(q.transfer, dtype, raw);
    if !loc.finite || loc.n == 0 {
        return fallback();
    }
    let st = global.copied().unwrap_or(loc);
    let step = match q.loss {
        Loss::Abs(d) => d,
        Loss::RelMax(b) => {
            if st.max > 0.0 {
                st.max / ((1u64 << (b - 1)) - 1) as f64
            } else {
                1.0
            }
        }
        Loss::RelRms(e) => {
            let rms = (st.ss / st.n.max(1) as f64).sqrt();
            if rms > 0.0 {
                e * 12f64.sqrt() * rms
            } else {
                1.0
            }
        }
        _ => return fallback(),
    };
    if !(step.is_finite() && step > 0.0) {
        return fallback();
    }
    // largest |q| of this chunk bounds the zigzag word; beyond 32 bits the spec cannot be met
    let qmax = loc.max / step;
    if !(qmax < 2.0e9) {
        return fallback();
    }
    // |q| ≤ round(qmax) (+ slack for u·(1/step) vs u/step), zigzag(q) ≤ 2|q|
    let zbound = 2 * (qmax + 0.5 + 1e-6) as u64;
    let qw = if zbound < 1 << 8 {
        1
    } else if zbound < 1 << 16 {
        2
    } else {
        4
    };
    let (tf, knee) = match q.transfer {
        Transfer::Linear => (0u8, 0.0),
        Transfer::Asinh(k) => (1u8, k),
    };
    let mut out = vec![0u8; PREFIX + loc.n * qw];
    out[..4].copy_from_slice(&[qw as u8, tf, 0, 0]);
    out[4..12].copy_from_slice(&step.to_le_bytes());
    out[12..20].copy_from_slice(&knee.to_le_bytes());
    let lin = q.transfer == Transfer::Linear;
    if is64 {
        quant_dispatch::<S64>(lin, qw, raw, &mut out[PREFIX..], knee_of(q.transfer), 1.0 / step);
    } else {
        quant_dispatch::<S32>(lin, qw, raw, &mut out[PREFIX..], knee_of(q.transfer), 1.0 / step);
    }
    (out, qw)
}

fn dequant_loop<S: Sc, const LIN: bool, const QW: usize>(qb: &[u8], out: &mut [u8], knee: f64, step: f64) {
    for (c, o) in qb.chunks_exact(QW).zip(out.chunks_exact_mut(S::SZ)) {
        let mut w = [0u8; 8];
        w[..QW].copy_from_slice(c);
        S::wr(inv::<LIN>(knee, unzigzag(u64::from_le_bytes(w)) as f64 * step), o);
    }
}

fn dequant_dispatch<S: Sc>(lin: bool, qw: usize, qb: &[u8], out: &mut [u8], knee: f64, step: f64) {
    match (lin, qw) {
        (true, 1) => dequant_loop::<S, true, 1>(qb, out, knee, step),
        (true, 2) => dequant_loop::<S, true, 2>(qb, out, knee, step),
        (true, _) => dequant_loop::<S, true, 4>(qb, out, knee, step),
        (false, 1) => dequant_loop::<S, false, 1>(qb, out, knee, step),
        (false, 2) => dequant_loop::<S, false, 2>(qb, out, knee, step),
        (false, _) => dequant_loop::<S, false, 4>(qb, out, knee, step),
    }
}

/// Decode one quantised chunk (`data` = prefix + blocks) of `numel` elements to raw dtype bytes.
pub(crate) fn decode_chunk(dtype: DType, numel: usize, data: &[u8], bs: usize) -> Result<Vec<u8>> {
    let (is64, comps) = scalar_kind(dtype).ok_or_else(|| Error::Format(format!("quantised chunk of dtype {}", dtype.name())))?;
    if data.len() < PREFIX {
        return Err(Error::Format("short quantised chunk".into()));
    }
    let qw = data[0] as usize;
    if qw == 0 {
        return crate::format::decode_blocks(&data[PREFIX..], numel * dtype.size(), bs);
    }
    if !matches!(qw, 1 | 2 | 4) {
        return Err(Error::Format(format!("quantised word size {qw}")));
    }
    let lin = match data[1] {
        0 => true,
        1 => false,
        t => return Err(Error::Format(format!("unknown transfer {t}"))),
    };
    let step = f64::from_le_bytes(data[4..12].try_into().unwrap());
    let knee = f64::from_le_bytes(data[12..20].try_into().unwrap());
    if !lin && !(knee.is_finite() && knee > 0.0) {
        return Err(Error::Format("bad asinh knee".into()));
    }
    let n = numel * comps;
    let qb = crate::format::decode_blocks(&data[PREFIX..], n * qw, bs)?;
    let mut out = vec![0u8; n * dtype.size() / comps];
    if is64 {
        dequant_dispatch::<S64>(lin, qw, &qb, &mut out, knee, step);
    } else {
        dequant_dispatch::<S32>(lin, qw, &qb, &mut out, knee, step);
    }
    Ok(out)
}

/// Round the mantissas of a float buffer to `m` bits (round to nearest even; inf / NaN untouched).
pub(crate) fn round_mantissa(dtype: DType, m: u32, raw: &[u8]) -> Vec<u8> {
    let mut out = raw.to_vec();
    if scalar_kind(dtype).unwrap().0 {
        let drop = 52 - m;
        if drop > 0 {
            out.par_chunks_mut(8 << 14).for_each(|c| {
                for w in c.chunks_exact_mut(8) {
                    let b = u64::from_le_bytes((&*w).try_into().unwrap());
                    if (b >> 52) & 0x7ff != 0x7ff {
                        let r = (b + (1u64 << (drop - 1)) - 1 + ((b >> drop) & 1)) & !((1u64 << drop) - 1);
                        w.copy_from_slice(&r.to_le_bytes());
                    }
                }
            });
        }
    } else {
        let drop = 23 - m;
        if drop > 0 {
            out.par_chunks_mut(4 << 14).for_each(|c| {
                for w in c.chunks_exact_mut(4) {
                    let b = u32::from_le_bytes((&*w).try_into().unwrap());
                    if (b >> 23) & 0xff != 0xff {
                        let r = (b + (1u32 << (drop - 1)) - 1 + ((b >> drop) & 1)) & !((1u32 << drop) - 1);
                        w.copy_from_slice(&r.to_le_bytes());
                    }
                }
            });
        }
    }
    out
}
