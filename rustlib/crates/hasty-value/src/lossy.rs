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

fn load(d: DType, b: &[u8]) -> Vec<f64> {
    if scalar_kind(d).unwrap().0 {
        b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()
    }
}

fn store(d: DType, x: &[f64]) -> Vec<u8> {
    if scalar_kind(d).unwrap().0 {
        x.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        x.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect()
    }
}

fn fwd(t: Transfer, x: f64) -> f64 {
    match t {
        Transfer::Linear => x,
        Transfer::Asinh(k) => k * (x / k).asinh(),
    }
}

fn inv(t: Transfer, u: f64) -> f64 {
    match t {
        Transfer::Linear => u,
        Transfer::Asinh(k) => k * (u / k).sinh(),
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

fn stats_of(t: Transfer, x: &[f64]) -> Stats {
    let mut s = Stats::ZERO;
    for &v in x {
        let u = fwd(t, v);
        if !u.is_finite() {
            s.finite = false;
            continue;
        }
        s.max = s.max.max(u.abs());
        s.ss += u * u;
    }
    s.n = x.len();
    s
}

/// statistics of a whole tensor (parallel; for [`Scope::Tensor`])
pub(crate) fn tensor_stats(q: &Quant, t: &Tensor) -> Stats {
    let es = t.dtype().size();
    let step = (1usize << 20) / es * es;
    t.bytes().par_chunks(step.max(es)).map(|c| stats_of(q.transfer, &load(t.dtype(), c))).reduce(|| Stats::ZERO, Stats::merge)
}

/// chunk prefix: qw u8 (0 = chunk stored losslessly), transfer u8 (0 linear, 1 asinh), pad u16, step f64, knee f64
pub(crate) const PREFIX: usize = 20;

fn zigzag(q: i64) -> u64 {
    ((q << 1) ^ (q >> 63)) as u64
}

fn unzigzag(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

/// Quantise one chunk's raw bytes: returns (prefix + payload, shuffle word); the payload is blocked by the caller.
pub(crate) fn encode_chunk(q: &Quant, dtype: DType, raw: &[u8], global: Option<&Stats>) -> (Vec<u8>, usize) {
    let fallback = || {
        let mut v = vec![0u8; PREFIX];
        v.extend_from_slice(raw);
        (v, dtype.word())
    };
    let x = load(dtype, raw);
    let st = global.copied().unwrap_or_else(|| stats_of(q.transfer, &x));
    if !st.finite || x.is_empty() {
        return fallback();
    }
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
    let z: Vec<u64> = x.iter().map(|&v| zigzag((fwd(q.transfer, v) / step).round() as i64)).collect();
    let zmax = z.iter().copied().max().unwrap_or(0);
    // |u/step| beyond i64 would have saturated in the cast: treat as unmeetable
    if zmax > u32::MAX as u64 || x.iter().any(|&v| (fwd(q.transfer, v) / step).abs() >= 4e9) {
        return fallback();
    }
    let qw = if zmax < 1 << 8 {
        1
    } else if zmax < 1 << 16 {
        2
    } else {
        4
    };
    let (tf, knee) = match q.transfer {
        Transfer::Linear => (0u8, 0.0),
        Transfer::Asinh(k) => (1u8, k),
    };
    let mut out = Vec::with_capacity(PREFIX + z.len() * qw);
    out.extend_from_slice(&[qw as u8, tf, 0, 0]);
    out.extend_from_slice(&step.to_le_bytes());
    out.extend_from_slice(&knee.to_le_bytes());
    for v in z {
        out.extend_from_slice(&v.to_le_bytes()[..qw]);
    }
    (out, qw)
}

/// Decode one quantised chunk (`data` = prefix + blocks) of `numel` elements to raw dtype bytes.
pub(crate) fn decode_chunk(dtype: DType, numel: usize, data: &[u8], bs: usize) -> Result<Vec<u8>> {
    let (_, comps) = scalar_kind(dtype).ok_or_else(|| Error::Format(format!("quantised chunk of dtype {}", dtype.name())))?;
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
    let transfer = match data[1] {
        0 => Transfer::Linear,
        1 => Transfer::Asinh(f64::from_le_bytes(data[12..20].try_into().unwrap())),
        t => return Err(Error::Format(format!("unknown transfer {t}"))),
    };
    let step = f64::from_le_bytes(data[4..12].try_into().unwrap());
    let n = numel * comps;
    let qb = crate::format::decode_blocks(&data[PREFIX..], n * qw, bs)?;
    let x: Vec<f64> = qb
        .chunks_exact(qw)
        .map(|c| {
            let mut w = [0u8; 8];
            w[..qw].copy_from_slice(c);
            inv(transfer, unzigzag(u64::from_le_bytes(w)) as f64 * step)
        })
        .collect();
    Ok(store(dtype, &x))
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
