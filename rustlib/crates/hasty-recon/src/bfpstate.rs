//! Block-floating-point (BFP) host solver state (research/bfp_storage.md §3.7, §6, §7 step 5).
//!
//! [`run_objective_host_bfp`] is the P²nP / ZMSHR1 loop of [`crate::hoststack::run_objective_host`] with the gradient
//! g, the previous gradient g_prev and the secant step held as BFP stacks (one [`hasty_bfp::BfpBuf`] per set) and the
//! iterate x kept as the f32 master copy (a BFP iterate stalls at ≈ 3 · 2^−b, §3.7). Switch: [`StateStore`]
//! (`HASTY_STATE_STORE` / `--state-store f32 | bfpN | bfpN:M`); with `f32` the old loop runs, untouched.
//!
//! - Every pass works on chunks of 64 Ki complex values (4096 BFP blocks), decoded into per-thread scratch; no BFP
//!   stack is ever copied to f32 as a whole. Set-sized f32 temporaries exist only where the objective's metric needs a
//!   whole set (three instead of the f32 loop's two).
//! - The gradient arrives set by set through a [`GradSink`] ([`Objective::gradient_host_sink`]): host encode of an f32
//!   download, or the device-packed bytes as they are (`hasty_bfp::xfer`, half the PCIe bytes at bfp16). Objectives
//!   without a sink path ([`Objective::has_gradient_sink`] false: the velocity model, the temporal subspace) keep g
//!   and g_prev f32 ([`StateStore::effective`]); then only the secant record is BFP.
//! - The α-scaling of g is applied on the fly (α g = α · decode(G)), so G holds the gradient exactly as delivered.
//! - **Secant step stored as a difference.** A BFP copy of x_prev would make s = x − x_prev wrong by 2^−b ‖x‖, which
//!   near convergence exceeds ‖s‖ itself. So when the iteration has no post-step denoiser (RED, no prior), the step
//!   writes s = x_new − x_old (exact f64 difference of the f32 iterates, then rounded: relative error 2^−b of s). Whether
//!   the denoiser changed x is checked by a fingerprint of all bits of x before and after it (two read passes of x,
//!   ≈ 1 % of a gradient at 1 mm); the first iteration stores x_prev, and the difference form is used from the second
//!   iteration on while the denoiser leaves x untouched. If it ever changes x (P²nP, or a prior switched on late), the
//!   stored difference is invalid: the next iteration takes the first-step branch (as at k = 0) and x_prev is stored
//!   from then on (error 2^−b‖x‖ in s).
//! - **Width by prior** (research/bfp_storage.md §9, measured): with no post-step denoiser (RED, no prior) g → 0 at
//!   the stationary point and s is a difference record, so bfp16 (even bfp12) leaves the fixed point at the f32
//!   floor. With P²nP (a denoiser after the step) g does not vanish and s is x − x_prev of non-vanishing iterates:
//!   bfp16 stalls ZMSHR1 at ≈ 2 · 2^−16 of x, bfp24 stays within 2.4e-7. A warning is printed below 20 bits.
//! - g_prev is stored in the gradient's width: the pair's m = α (decode G − decode G_prev) is the exact difference of
//!   the gradients the solver used; G and G_prev swap after the step (no copy).
//! - Checkpoints (`state_<name>.hsb`): the BFP stacks bit-exact (their own bytes); x quantised at the checkpoint
//!   policy of the f32 loop while the record is a difference, bit-exact while it is x_prev (s = x − x_prev would
//!   otherwise carry x's checkpoint noise, 1e-6 of rms(x)). A run never resumes across store kinds (refused).

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::CudaStream;
use hasty_bfp::{Bfp, BfpBuf, BfpView, BfpViewMut};
use hasty_cuda::C32;
use hasty_value::Value;
use rayon::prelude::*;

use crate::checkpoint::{Outcome, RunCtl};
use crate::hoststack::HostStack;
use crate::p2np::{zmshr1_rank_one_grouped, zmshr1_scalars_grouped, Objective, Precond};
use crate::Error;

/// complex values per parallel work item (4096 BFP blocks)
pub const PAR: usize = 1 << 16;

/// Storage of the host solver state. `grad`: the gradient and the previous gradient; `pair`: the secant step
/// (difference form) or x_prev. None = f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct StateStore {
    pub grad: Option<Bfp>,
    pub pair: Option<Bfp>,
}

impl StateStore {
    pub const F32: Self = Self { grad: None, pair: None };

    pub fn is_f32(&self) -> bool {
        self.grad.is_none() && self.pair.is_none()
    }

    /// `f32` | `bfp` (= bfp16) | `bfpN` (gradients and pair at N bits) | `G:P` with G, P each `f32` or `bfpN`
    pub fn parse(s: &str) -> Result<Self, String> {
        let w = |p: &str| -> Result<Option<Bfp>, String> {
            match p {
                "f32" => Ok(None),
                "bfp" => Ok(Some(Bfp::new(16))),
                _ => {
                    let b: u32 = p.strip_prefix("bfp").and_then(|v| v.parse().ok()).ok_or_else(|| format!("state store '{s}': expected f32 | bfp | bfpN | G:P"))?;
                    if !(8..=24).contains(&b) {
                        return Err(format!("state store '{s}': BFP width must be in 8..=24"));
                    }
                    Ok(Some(Bfp::new(b)))
                }
            }
        };
        match s.split_once(':') {
            None => {
                let f = w(s)?;
                Ok(Self { grad: f, pair: f })
            }
            Some((a, b)) => Ok(Self { grad: w(a)?, pair: w(b)? }),
        }
    }

    /// `HASTY_STATE_STORE` (unset or empty: f32)
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("HASTY_STATE_STORE") {
            Ok(v) if !v.is_empty() => Self::parse(&v),
            _ => Ok(Self::F32),
        }
    }

    pub fn describe(&self) -> String {
        let n = |f: Option<Bfp>| f.map_or("f32".to_string(), |b| format!("bfp{}", b.bits()));
        if self.is_f32() {
            "f32".into()
        } else {
            format!("x f32 (master), g and g_prev {}, secant step {}", n(self.grad), n(self.pair))
        }
    }

    /// the store a run actually uses: without a gradient sink path ([`Objective::has_gradient_sink`]) the gradient
    /// and g_prev stay f32 (m = α(g − g_prev) must difference two values of the same precision), only the secant
    /// record is BFP
    pub fn effective(self, grad_sink: bool) -> Self {
        if grad_sink {
            self
        } else {
            Self { grad: None, ..self }
        }
    }

    /// host bytes of one stack of the given sets in format `fmt`
    pub fn stack_bytes(fmt: Option<Bfp>, lens: &[usize]) -> usize {
        match fmt {
            None => 8 * lens.iter().sum::<usize>(),
            Some(f) => lens.iter().map(|&l| BfpBuf::bytes_for(f, 2 * l)).sum(),
        }
    }

    /// Host bytes of the solver state for stacks of sets `lens`: x (f32), g, and with a quasi-Newton preconditioner
    /// g_prev and the secant step; `metric`: the set-sized f32 temporaries of the metric steps (two in the f32 loop,
    /// three with BFP stacks).
    pub fn state_bytes(&self, lens: &[usize], quasi: bool, metric: bool) -> usize {
        let maxl = lens.iter().copied().max().unwrap_or(0);
        let mut b = 8 * lens.iter().sum::<usize>() + Self::stack_bytes(self.grad, lens);
        if quasi {
            b += Self::stack_bytes(self.grad, lens) + Self::stack_bytes(self.pair, lens);
        }
        if metric {
            b += if self.is_f32() { 2 } else { 3 } * 8 * maxl;
        }
        b
    }
}

/// One state stack: f32 ([`HostStack`]) or one BFP buffer per set (2 · len reals each).
#[derive(Clone, Debug)]
pub enum Buf {
    F(HostStack),
    B(Vec<BfpBuf>),
}

/// read view of one chunk
pub enum Rd<'a> {
    F(&'a [C32]),
    B(BfpView<'a>),
}

/// write view of one chunk
pub enum Wr<'a> {
    F(&'a mut [C32]),
    B(BfpViewMut<'a>),
}

impl<'a> Rd<'a> {
    /// the chunk's values (a BFP chunk decoded into `buf`)
    pub fn get<'s>(&'s self, buf: &'s mut Vec<C32>) -> &'s [C32] {
        match self {
            Rd::F(s) => s,
            Rd::B(v) => {
                let n = v.len / 2;
                if buf.len() < n {
                    buf.resize(n, C32::new(0.0, 0.0));
                }
                v.decode_c(&mut buf[..n]);
                &buf[..n]
            }
        }
    }
}

impl<'a> Wr<'a> {
    pub fn get<'s>(&'s self, buf: &'s mut Vec<C32>) -> &'s [C32] {
        match self {
            Wr::F(s) => s,
            Wr::B(v) => {
                let n = v.len / 2;
                if buf.len() < n {
                    buf.resize(n, C32::new(0.0, 0.0));
                }
                v.decode_c(&mut buf[..n]);
                &buf[..n]
            }
        }
    }
    pub fn put(&mut self, v: &[C32]) {
        match self {
            Wr::F(s) => s.copy_from_slice(v),
            Wr::B(w) => w.encode_c(v),
        }
    }
}

impl Buf {
    pub fn zeros(fmt: Option<Bfp>, lens: &[usize]) -> Self {
        match fmt {
            None => Buf::F(HostStack::zeros(lens)),
            Some(f) => Buf::B(lens.iter().map(|&l| BfpBuf::zeros_c(f, l)).collect()),
        }
    }

    pub fn fmt(&self) -> Option<Bfp> {
        match self {
            Buf::F(_) => None,
            Buf::B(v) => v.first().map(|b| b.fmt),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Buf::F(h) => h.bytes(),
            Buf::B(v) => v.iter().map(|b| b.bytes()).sum(),
        }
    }

    pub fn set_len(&self, t: usize) -> usize {
        match self {
            Buf::F(h) => h.lens[t],
            Buf::B(v) => v[t].len / 2,
        }
    }

    /// set t into `out` (decoded or copied, parallel)
    pub fn read_set(&self, t: usize, out: &mut [C32]) {
        match self {
            Buf::F(h) => out.par_chunks_mut(PAR).zip(h.chunk(t).par_chunks(PAR)).for_each(|(a, b)| a.copy_from_slice(b)),
            Buf::B(v) => v[t].decode_into_c(out),
        }
    }

    /// set t from `v` (encoded or copied, parallel)
    pub fn write_set(&mut self, t: usize, v: &[C32]) {
        match self {
            Buf::F(h) => h.chunk_mut(t).par_chunks_mut(PAR).zip(v.par_chunks(PAR)).for_each(|(a, b)| a.copy_from_slice(b)),
            Buf::B(s) => s[t].encode_from_c(v),
        }
    }

    /// read view of chunk c (PAR complex values) of set t
    pub fn rd_at(&self, t: usize, c: usize) -> Rd<'_> {
        let n = self.set_len(t);
        let (c0, c1) = (c * PAR, ((c + 1) * PAR).min(n));
        match self {
            Buf::F(h) => Rd::F(&h.chunk(t)[c0..c1]),
            Buf::B(v) => Rd::B(v[t].view(2 * c0, 2 * (c1 - c0))),
        }
    }

    /// write views of all chunks of set t (in chunk order)
    pub fn wr_all(&mut self, t: usize) -> Vec<Wr<'_>> {
        match self {
            Buf::F(h) => h.chunk_mut(t).chunks_mut(PAR).map(Wr::F).collect(),
            Buf::B(v) => v[t].par_views_mut(2 * PAR).map(Wr::B).collect(),
        }
    }

    /// whether `other` has the same storage (so the two can be swapped instead of copied)
    fn same_kind(&self, other: &Buf) -> bool {
        self.fmt() == other.fmt() && std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// Destination of a gradient delivered set by set ([`Objective::gradient_host_sink`]).
pub trait GradSink: Send {
    /// width the sink stores; when Some, a device may hand over packed bytes ([`put_packed`](Self::put_packed))
    fn packed(&self) -> Option<Bfp> {
        None
    }
    /// set t's gradient (complex values, the set's whole length)
    fn put(&mut self, t: usize, g: &[C32]) -> Result<(), Error>;
    /// set t's gradient packed in the [`packed`](Self::packed) width, one row of 2 · len reals (`hasty_bfp::xfer`)
    fn put_packed(&mut self, _t: usize, _mant: &[u32], _ex: &[u8]) -> Result<(), Error> {
        Err(Error::Format("gradient sink: packed input not supported".into()))
    }
    /// accumulation (objectives whose chunks collect contributions from several sets, e.g. the velocity model's
    /// knots over the encodes): read back the values last stored for chunk t into `out`; Ok(false) when this sink
    /// cannot read back (the objective then declines its sink path)
    fn get(&mut self, _t: usize, _out: &mut [C32]) -> Result<bool, Error> {
        Ok(false)
    }
    /// a partial sum of chunk t (read back by [`get`](Self::get) and completed by a later [`put`](Self::put)); a BFP
    /// sink stores it at its width, so a chunk completed after n partial sums carries ≈ √n roundings of 2^−b. Default:
    /// as `put`.
    fn put_partial(&mut self, t: usize, g: &[C32]) -> Result<(), Error> {
        self.put(t, g)
    }
}

impl GradSink for Buf {
    fn packed(&self) -> Option<Bfp> {
        self.fmt()
    }
    fn put(&mut self, t: usize, g: &[C32]) -> Result<(), Error> {
        if g.len() != self.set_len(t) {
            return Err(Error::Format(format!("gradient sink: set {t} has {} values, expected {}", g.len(), self.set_len(t))));
        }
        self.write_set(t, g);
        Ok(())
    }
    fn put_packed(&mut self, t: usize, mant: &[u32], ex: &[u8]) -> Result<(), Error> {
        match self {
            Buf::B(v) => {
                v[t].copy_packed(mant, ex);
                Ok(())
            }
            Buf::F(_) => Err(Error::Format("gradient sink: packed input into an f32 stack".into())),
        }
    }
    fn get(&mut self, t: usize, out: &mut [C32]) -> Result<bool, Error> {
        if out.len() != self.set_len(t) {
            return Err(Error::Format(format!("gradient sink: set {t} has {} values, read into {}", self.set_len(t), out.len())));
        }
        self.read_set(t, out);
        Ok(true)
    }
}

/// [`GradSink`] into the chunks of an f32 [`HostStack`] (what `SetStreamer::for_each` writes)
pub struct StackSink<'a>(pub &'a mut HostStack);

impl GradSink for StackSink<'_> {
    fn put(&mut self, t: usize, g: &[C32]) -> Result<(), Error> {
        self.0.chunk_mut(t).copy_from_slice(g);
        Ok(())
    }
    fn get(&mut self, t: usize, out: &mut [C32]) -> Result<bool, Error> {
        out.copy_from_slice(self.0.chunk(t));
        Ok(true)
    }
}

/// a gradient-sink adapter that scales nothing but records ‖g‖² per set (the RED wrapper's statistics)
pub struct SumSink<'a> {
    pub inner: &'a mut dyn GradSink,
    pub sq: f64,
    pub hook: Option<&'a (dyn Fn(usize, &mut [C32]) + Sync)>,
}

impl GradSink for SumSink<'_> {
    fn packed(&self) -> Option<Bfp> {
        // the hook modifies the values: packed bytes cannot pass through it
        if self.hook.is_some() {
            None
        } else {
            self.inner.packed()
        }
    }
    fn put(&mut self, t: usize, g: &[C32]) -> Result<(), Error> {
        self.sq += g.par_iter().map(|v| v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64).sum::<f64>();
        match self.hook {
            Some(h) => {
                let mut v = g.to_vec();
                h(t, &mut v);
                self.inner.put(t, &v)
            }
            None => self.inner.put(t, g),
        }
    }
    fn get(&mut self, t: usize, out: &mut [C32]) -> Result<bool, Error> {
        self.inner.get(t, out)
    }
    /// partial sums pass to the inner sink untouched: the statistic and the hook see the completed chunk only
    fn put_partial(&mut self, t: usize, g: &[C32]) -> Result<(), Error> {
        self.inner.put_partial(t, g)
    }
    fn put_packed(&mut self, t: usize, mant: &[u32], ex: &[u8]) -> Result<(), Error> {
        // ‖g‖² from the decoded values (zero-padded tail values add nothing)
        if let Some(f) = self.inner.packed() {
            let nb = ex.len().min(mant.len() / f.bits() as usize);
            let mut v = vec![0f32; nb * hasty_bfp::VALS];
            f.decode(mant, ex, &mut v);
            self.sq += v.par_iter().map(|&a| a as f64 * a as f64).sum::<f64>();
        }
        self.inner.put_packed(t, mant, ex)
    }
}

thread_local! {
    static SCRATCH: RefCell<[Vec<C32>; 8]> = RefCell::new(Default::default());
}

fn with_sc<R>(f: impl FnOnce(&mut [Vec<C32>; 8]) -> R) -> R {
    SCRATCH.with(|s| f(&mut s.borrow_mut()))
}

/// order-sensitive hash of every bit of `x` (parallel)
pub fn fingerprint(x: &[C32]) -> u64 {
    x.par_chunks(PAR)
        .enumerate()
        .map(|(c, ch)| {
            let mut h = 0xcbf2_9ce4_8422_2325u64 ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for v in ch {
                h = (h ^ (v.re.to_bits() as u64 | (v.im.to_bits() as u64) << 32)).wrapping_mul(0x0000_0100_0000_01b3);
            }
            let mut z = h;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z ^ (z >> 27)
        })
        .reduce(|| 0u64, |a, b| a.wrapping_add(b))
}

#[inline]
fn sc(v: C32, a: f32) -> C32 {
    C32::new(v.re * a, v.im * a)
}

/// x_t ← x_t − step chunk by chunk, writing the secant record of the step into `xp` (x_prev ← x_old, or the
/// difference x_new − x_old when `diff`). `step(c, i0, xchunk_old, sc, out)` fills the step of chunk c.
fn update_with_pair(x: &mut [C32], mut xp: Option<(&mut Buf, usize, bool)>, step: &(dyn Fn(usize, &[C32], &mut [Vec<C32>; 8], &mut Vec<C32>) + Sync)) {
    let xs: Vec<&mut [C32]> = x.chunks_mut(PAR).collect();
    match xp.as_mut() {
        None => xs.into_par_iter().enumerate().for_each(|(c, xc)| {
            with_sc(|scr| {
                let mut out = std::mem::take(&mut scr[3]);
                step(c, xc, scr, &mut out);
                for (v, s) in xc.iter_mut().zip(&out) {
                    *v = C32::new(v.re - s.re, v.im - s.im);
                }
                scr[3] = out;
            })
        }),
        Some((buf, t, diff)) => {
            let diff = *diff;
            let ws = buf.wr_all(*t);
            xs.into_par_iter().zip(ws.into_par_iter()).enumerate().for_each(|(c, (xc, mut w))| {
                with_sc(|scr| {
                    let mut out = std::mem::take(&mut scr[3]);
                    step(c, xc, scr, &mut out);
                    if diff {
                        // s = x_new − x_old of the f32 iterates (exact in f64), then the update
                        let mut d = std::mem::take(&mut scr[2]);
                        d.resize(xc.len(), C32::new(0.0, 0.0));
                        for ((v, s), dv) in xc.iter_mut().zip(&out).zip(d.iter_mut()) {
                            let nv = C32::new(v.re - s.re, v.im - s.im);
                            *dv = C32::new((nv.re as f64 - v.re as f64) as f32, (nv.im as f64 - v.im as f64) as f32);
                            *v = nv;
                        }
                        w.put(&d[..xc.len()]);
                        scr[2] = d;
                    } else {
                        w.put(xc);
                        for (v, s) in xc.iter_mut().zip(&out) {
                            *v = C32::new(v.re - s.re, v.im - s.im);
                        }
                    }
                    scr[3] = out;
                })
            });
        }
    }
}

/// chunks of PAR complex values in a set of `n`
fn chunks_of(n: usize) -> usize {
    n.div_ceil(PAR)
}

const CK_EXT: &str = "hsb";

fn ck_path(c: &RunCtl, name: &str) -> std::path::PathBuf {
    c.dir().join(format!("state_{name}.{CK_EXT}"))
}

fn as_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain old data (C32, u32), no padding
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Checkpoint of the BFP state: meta json, x per set (f32, the checkpoint policy of the f32 loop), each further stack
/// per set either as complex64 (f32 stacks, lossless) or as its BFP bytes (mantissa words + exponent bytes).
fn save_bfp(c: &RunCtl, name: &str, meta: &serde_json::Value, x: &HostStack, x_lossless: bool, stacks: &[(&str, &Buf)]) -> Result<(), Error> {
    use hasty_value::store::{self, Policy, Save, ValueWriter, CHECKPOINT_REL};
    let path = ck_path(c, name);
    let n_ent = 1 + x.nsets() + stacks.len() * 2 * x.nsets();
    let mut w = ValueWriter::create(&path, n_ent)?;
    w.put("meta", &Value::Str(meta.to_string()))?;
    let lossless = Save::default().policy(Policy::Lossless);
    let save = if x_lossless { lossless.clone() } else { Save::default().policy(Policy::Uniform { rel: CHECKPOINT_REL }) };
    for t in 0..x.nsets() {
        let v = x.chunk(t);
        let (q, _) = store::resolve_policy(save.policy, hasty_value::DType::C64, Some(&[as_bytes(v)]))?;
        w.begin_tensor(&format!("x.{t}"), hasty_value::DType::C64, &[v.len()], &save, q)?;
        w.push(as_bytes(v))?;
        w.end_tensor()?;
    }
    for (key, b) in stacks {
        for t in 0..x.nsets() {
            match b {
                Buf::F(h) => {
                    let v = h.chunk(t);
                    w.begin_tensor(&format!("{key}.{t}.f"), hasty_value::DType::C64, &[v.len()], &lossless, None)?;
                    w.push(as_bytes(v))?;
                    w.end_tensor()?;
                    w.put(&format!("{key}.{t}.e"), &Value::Str(String::new()))?;
                }
                Buf::B(v) => {
                    let m = as_bytes(&v[t].mant);
                    w.begin_tensor(&format!("{key}.{t}.m"), hasty_value::DType::U8, &[m.len()], &lossless, None)?;
                    w.push(m)?;
                    w.end_tensor()?;
                    w.begin_tensor(&format!("{key}.{t}.e"), hasty_value::DType::U8, &[v[t].ex.len()], &lossless, None)?;
                    w.push(&v[t].ex)?;
                    w.end_tensor()?;
                }
            }
        }
    }
    w.finish()?;
    Ok(())
}

/// Load a checkpoint of [`save_bfp`]; `check` sees the meta before anything is read into the stacks.
fn load_bfp(c: &RunCtl, name: &str, x: &mut HostStack, stacks: &mut [(&str, &mut Buf)], check: &dyn Fn(&serde_json::Value) -> Result<(), Error>) -> Result<Option<serde_json::Value>, Error> {
    use hasty_value::lazy::{Lazy, LazyFile};
    let path = ck_path(c, name);
    if !path.exists() {
        return Ok(None);
    }
    let bad = |m: String| Error::Format(format!("{}: {m}", path.display()));
    let lf = LazyFile::open(&path)?;
    let meta: serde_json::Value = match lf.root.get("meta") {
        Some(Lazy::Value(Value::Str(s))) => serde_json::from_str(s).map_err(|e| bad(e.to_string()))?,
        _ => return Err(bad("missing meta".into())),
    };
    check(&meta)?;
    let read = |key: &str, n: usize, dt: hasty_value::DType| -> Result<Vec<u8>, Error> {
        let tr = lf.root.get(key).and_then(|l| l.tensor()).ok_or_else(|| bad(format!("missing {key}")))?;
        if tr.shape != [n] || tr.dtype != dt {
            return Err(bad(format!("{key}: shape / dtype differ from the run's")));
        }
        let part = hasty_value::store::io_pool().install(|| lf.read_box(tr, &[0], &[n]))?;
        Ok(part.bytes().to_vec())
    };
    for t in 0..x.nsets() {
        let n = x.lens[t];
        let b = read(&format!("x.{t}"), n, hasty_value::DType::C64)?;
        // SAFETY: n complex64 values
        x.chunk_mut(t).copy_from_slice(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const C32, n) });
    }
    for (key, buf) in stacks.iter_mut() {
        for t in 0..x.nsets() {
            match buf {
                Buf::F(h) => {
                    let n = h.lens[t];
                    let b = read(&format!("{key}.{t}.f"), n, hasty_value::DType::C64)?;
                    h.chunk_mut(t).copy_from_slice(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const C32, n) });
                }
                Buf::B(v) => {
                    let nm = v[t].mant.len();
                    let m = read(&format!("{key}.{t}.m"), 4 * nm, hasty_value::DType::U8)?;
                    for (d, s) in v[t].mant.iter_mut().zip(m.chunks_exact(4)) {
                        *d = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                    }
                    let ne = v[t].ex.len();
                    let e = read(&format!("{key}.{t}.e"), ne, hasty_value::DType::U8)?;
                    v[t].ex.copy_from_slice(&e);
                }
            }
        }
    }
    Ok(Some(meta))
}

/// [`crate::hoststack::run_objective_host`] with the state in the given [`StateStore`] (see the module docs). Same
/// arguments, same updates; the f32 loop is used for [`StateStore::F32`].
#[allow(clippy::too_many_arguments)]
pub fn run_objective_host_bfp(
    obj: &mut dyn Objective,
    st: &Arc<CudaStream>,
    x: &mut HostStack,
    precond: Precond,
    mut alpha: f64,
    iters: usize,
    denoise: &mut dyn FnMut(usize, &mut Vec<C32>),
    report: &mut dyn FnMut(usize, &HostStack, f64),
    ctl: Option<(&RunCtl, &str)>,
    store: StateStore,
) -> Result<Outcome, Error> {
    let lens = x.lens.clone();
    let nt = lens.len();
    let quasi = precond.quasi();
    let metric = precond.metric();
    let per_set = precond.per_set();
    let groups = obj.secant_groups();
    let maxl = lens.iter().copied().max().unwrap_or(0);
    // set-sized temporaries of the metric steps: A, B, T
    let (mut ta, mut tb, mut tt) = if metric { (vec![C32::new(0.0, 0.0); maxl], vec![C32::new(0.0, 0.0); maxl], vec![C32::new(0.0, 0.0); maxl]) } else { (Vec::new(), Vec::new(), Vec::new()) };
    // g and g_prev in the gradient width, f32 when the objective cannot deliver the gradient set by set
    let req = store;
    let store = store.effective(obj.has_gradient_sink());
    if req.grad.is_some() && store.grad.is_none() {
        eprintln!("  host state: the objective has no gradient sink path: g and g_prev kept f32, the secant record {}", store.pair.map_or("f32".into(), |b| format!("bfp{}", b.bits())));
    }
    let mut g = Buf::zeros(store.grad, &lens);
    let mut sink_ok = store.grad.is_some();
    let mut gp = if quasi { Buf::zeros(store.grad, &lens) } else { Buf::F(HostStack::default()) };
    let mut xp = if quasi { Buf::zeros(store.pair, &lens) } else { Buf::F(HostStack::default()) };
    let pair_bfp = quasi && store.pair.is_some();
    // secant record: xp holds x_prev (false) or the difference x − x_prev (true); diff_ok: the denoiser left x
    // unchanged in every iteration so far (None: not checked yet); pair_invalid: the stored record does not match x
    let (mut xp_is_diff, mut diff_ok, mut pair_invalid): (bool, Option<bool>, bool) = (false, None, false);
    let mut k0 = 0;
    if let Some((c, name)) = ctl {
        if c.dir().join(format!("state_{name}.hsk")).exists() || c.dir().join(format!("state_{name}.hv")).exists() {
            return Err(Error::Format(format!("{name}: a checkpoint of the f32 state exists; resume with --state-store f32 (HASTY_STATE_STORE unset), or remove it")));
        }
        let mut sl: Vec<(&str, &mut Buf)> = if quasi { vec![("xp", &mut xp), ("gp", &mut gp)] } else { vec![] };
        let check = |m: &serde_json::Value| -> Result<(), Error> {
            let fmt_ok = m["grad"].as_u64().map(|v| v as u32) == store.grad.map(|b| b.bits()) && m["pair"].as_u64().map(|v| v as u32) == store.pair.map(|b| b.bits()) && m["precond"] == format!("{precond:?}");
            if fmt_ok {
                Ok(())
            } else {
                Err(Error::Format(format!("{name}: checkpoint written with state store {} / {}, this run uses {} / {precond:?}", m["store"], m["precond"], store.describe())))
            }
        };
        if let Some(m) = load_bfp(c, name, x, &mut sl, &check)? {
            k0 = m["k"].as_u64().ok_or_else(|| Error::Format("checkpoint: k".into()))? as usize;
            alpha = m["alpha"].as_f64().ok_or_else(|| Error::Format("checkpoint: alpha".into()))?;
            xp_is_diff = m["xp_is_diff"].as_bool().unwrap_or(false);
            diff_ok = m["diff_ok"].as_bool();
            pair_invalid = m["pair_invalid"].as_bool().unwrap_or(false);
            eprintln!("  resuming {name} at iteration {k0} / {iters} (host-resident state, {})", store.describe());
        }
    }
    eprintln!("  host state: {} ({:.2} GiB: x {:.2}, g {:.2}{})", store.describe(), (x.bytes() + g.bytes() + gp.bytes() + xp.bytes() + 8 * (ta.len() + tb.len() + tt.len())) as f64 / (1u64 << 30) as f64, x.bytes() as f64 / (1u64 << 30) as f64, g.bytes() as f64 / (1u64 << 30) as f64, if quasi { format!(", g_prev {:.2}, secant step {:.2}", gp.bytes() as f64 / (1u64 << 30) as f64, xp.bytes() as f64 / (1u64 << 30) as f64) } else { String::new() });
    if k0 == 0 {
        report(usize::MAX, x, f64::NAN);
    }
    let save = |c: &RunCtl, name: &str, next: usize, x: &HostStack, xp: &Buf, gp: &Buf, xp_is_diff: bool, diff_ok: Option<bool>, pair_invalid: bool| -> Result<(), Error> {
        let meta = serde_json::json!({ "k": next, "precond": format!("{precond:?}"), "alpha": alpha, "store": store.describe(), "grad": store.grad.map(|b| b.bits()), "pair": store.pair.map(|b| b.bits()), "xp_is_diff": xp_is_diff, "diff_ok": diff_ok, "pair_invalid": pair_invalid });
        let stacks: Vec<(&str, &Buf)> = if quasi { vec![("xp", xp), ("gp", gp)] } else { vec![] };
        save_bfp(c, name, &meta, x, quasi && !xp_is_diff, &stacks)
    };
    let a32 = |al: f64| al as f32;
    for k in k0..iters {
        let mut tm = crate::p2np::IterTimer::start();
        obj.begin_iteration(k, iters);
        // ── gradient into g (sink: set by set, packed by the device when it can) ─────────────────────────────────
        let res = if sink_ok {
            match obj.gradient_host_sink(st, x, &mut g) {
                Ok(Some(f)) => Ok(f),
                Ok(None) => {
                    eprintln!("  host state: the objective declined its gradient sink: g and g_prev kept f32 from here on");
                    sink_ok = false;
                    // g_prev to f32 too (m differences two values of one precision); g's BFP stack freed first
                    drop(std::mem::replace(&mut g, Buf::F(HostStack::default())));
                    if quasi {
                        let mut h = HostStack::zeros(&lens);
                        for t in 0..nt {
                            gp.read_set(t, h.chunk_mut(t));
                        }
                        gp = Buf::F(h);
                    }
                    g = Buf::F(HostStack::zeros(&lens));
                    let Buf::F(h) = &mut g else { unreachable!() };
                    obj.gradient_host(st, x, h)
                }
                Err(e) => Err(e),
            }
        } else {
            match &mut g {
                Buf::F(h) => obj.gradient_host(st, x, h),
                Buf::B(_) => unreachable!("BFP gradient without a sink"),
            }
        };
        let f = match res {
            Ok(f) => f,
            Err(e) => {
                if let (Some((c, name)), true) = (ctl, k > k0) {
                    eprintln!("  gradient failed at iteration {k} ({e}): saving a checkpoint to resume from iteration {k}");
                    if let Err(e2) = save(c, name, k, x, &xp, &gp, xp_is_diff, diff_ok, pair_invalid) {
                        eprintln!("  checkpoint failed too: {e2}");
                    }
                }
                return Err(e);
            }
        };
        tm.lap(st, 0);
        let al = a32(alpha);
        // the record this step writes: the difference form while the denoiser has left x untouched
        let write_diff = pair_bfp && diff_ok == Some(true);
        let first = k == 0 || !obj.curvature_pair_ok() || pair_invalid;
        // α g of chunk c of set t into scratch slot 0
        let gsc = |t: usize, c: usize, scr: &mut [Vec<C32>; 8], out: &mut Vec<C32>| {
            let r = g.rd_at(t, c);
            let gv = r.get(&mut scr[0]);
            out.clear();
            out.extend(gv.iter().map(|&v| sc(v, al)));
        };
        match precond {
            Precond::None => {
                for t in 0..nt {
                    update_with_pair(x.chunk_mut(t), None, &|c, _, scr, out| gsc(t, c, scr, out));
                }
            }
            Precond::Circulant => {
                for t in 0..nt {
                    let l = lens[t];
                    g.read_set(t, &mut ta[..l]);
                    ta[..l].par_iter_mut().for_each(|v| *v = sc(*v, al));
                    obj.metric_host(st, t, &ta[..l], &mut tt[..l], false)?;
                    let tref = &tt[..l];
                    update_with_pair(x.chunk_mut(t), None, &|c, _, _, out| {
                        out.clear();
                        out.extend_from_slice(&tref[c * PAR..((c + 1) * PAR).min(l)]);
                    });
                }
            }
            _ if first => {
                // the first step of a quasi-Newton run (or after an invalid pair): the plain / metric step
                for t in 0..nt {
                    let l = lens[t];
                    if metric {
                        g.read_set(t, &mut ta[..l]);
                        ta[..l].par_iter_mut().for_each(|v| *v = sc(*v, al));
                        obj.metric_host(st, t, &ta[..l], &mut tt[..l], false)?;
                        let tref = &tt[..l];
                        update_with_pair(x.chunk_mut(t), Some((&mut xp, t, write_diff)), &|c, _, _, out| {
                            out.clear();
                            out.extend_from_slice(&tref[c * PAR..((c + 1) * PAR).min(l)]);
                        });
                    } else {
                        update_with_pair(x.chunk_mut(t), Some((&mut xp, t, write_diff)), &|c, _, scr, out| gsc(t, c, scr, out));
                    }
                }
                if pair_invalid {
                    eprintln!("    it {k}: secant pair invalid (the denoiser changed x after a difference record): first-step update, x_prev stored from now on");
                }
                pair_invalid = false;
                xp_is_diff = write_diff;
            }
            Precond::Zmshr1 | Precond::Zmshr1Set => {
                // s from the record, m = α (g − g_prev), both per chunk
                let (gr, gpr, xpr) = (&g, &gp, &xp);
                let xd = xp_is_diff;
                let sm = |t: usize, c: usize, xc: &[C32], scr: &mut [Vec<C32>; 8], s: &mut Vec<C32>, m: &mut Vec<C32>| {
                    let [b0, b1, b2, ..] = scr;
                    let rx = xpr.rd_at(t, c);
                    let pv = rx.get(b0);
                    s.clear();
                    if xd {
                        s.extend_from_slice(pv);
                    } else {
                        s.extend(xc.iter().zip(pv).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)));
                    }
                    let (rg, rgp) = (gr.rd_at(t, c), gpr.rd_at(t, c));
                    let (gv, gpv) = (rg.get(b1), rgp.get(b2));
                    m.clear();
                    m.extend(gv.iter().zip(gpv).map(|(a, b)| C32::new(a.re * al - b.re * al, a.im * al - b.im * al)));
                };
                let xr = &*x;
                let d1: Vec<[f64; 3]> = (0..nt)
                    .map(|t| {
                        let xt = xr.chunk(t);
                        (0..chunks_of(lens[t]))
                            .into_par_iter()
                            .map(|c| {
                                with_sc(|scr| {
                                    let (mut s, mut m) = (std::mem::take(&mut scr[4]), std::mem::take(&mut scr[5]));
                                    sm(t, c, &xt[c * PAR..((c + 1) * PAR).min(lens[t])], scr, &mut s, &mut m);
                                    let mut a = [0.0f64; 3];
                                    for (p, q) in s.iter().zip(&m) {
                                        let (s0, s1, m0, m1) = (p.re as f64, p.im as f64, q.re as f64, q.im as f64);
                                        a[0] += s0 * s0 + s1 * s1;
                                        a[1] += s0 * m0 + s1 * m1;
                                        a[2] += m0 * m0 + m1 * m1;
                                    }
                                    (scr[4], scr[5]) = (s, m);
                                    a
                                })
                            })
                            .reduce(|| [0.0; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                    })
                    .collect();
                let at = zmshr1_scalars_grouped(&d1, per_set, groups.as_deref());
                let wz = |a: f64, tau: f64, s: C32, m: C32| -> ([f64; 2], [f64; 2]) {
                    let z = [a * s.re as f64 + (1.0 - a) * m.re as f64, a * s.im as f64 + (1.0 - a) * m.im as f64];
                    ([s.re as f64 - tau * z[0], s.im as f64 - tau * z[1]], z)
                };
                let d2: Vec<[f64; 4]> = (0..nt)
                    .map(|t| {
                        let (a, tau) = at[t];
                        let xt = xr.chunk(t);
                        (0..chunks_of(lens[t]))
                            .into_par_iter()
                            .map(|c| {
                                with_sc(|scr| {
                                    let (mut s, mut m) = (std::mem::take(&mut scr[4]), std::mem::take(&mut scr[5]));
                                    sm(t, c, &xt[c * PAR..((c + 1) * PAR).min(lens[t])], scr, &mut s, &mut m);
                                    let rg = gr.rd_at(t, c);
                                    let gv = rg.get(&mut scr[3]);
                                    let mut acc = [0.0f64; 4];
                                    for i in 0..s.len() {
                                        let (w, z) = wz(a, tau, s[i], m[i]);
                                        let gs = sc(gv[i], al);
                                        acc[0] += w[0] * z[0] + w[1] * z[1];
                                        acc[1] += w[0] * w[0] + w[1] * w[1];
                                        acc[2] += z[0] * z[0] + z[1] * z[1];
                                        acc[3] += w[0] * gs.re as f64 + w[1] * gs.im as f64;
                                    }
                                    (scr[4], scr[5]) = (s, m);
                                    acc
                                })
                            })
                            .reduce(|| [0.0; 4], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
                    })
                    .collect();
                let ct = zmshr1_rank_one_grouped(&d2, per_set, groups.as_deref());
                // step = τ α g + c w; the record is read before it is overwritten (same chunk, same task)
                for t in 0..nt {
                    let (a, tau) = at[t];
                    let cc = ct[t];
                    let l = lens[t];
                    let xs: Vec<&mut [C32]> = x.chunk_mut(t).chunks_mut(PAR).collect();
                    let ws = xp.wr_all(t);
                    xs.into_par_iter().zip(ws.into_par_iter()).enumerate().for_each(|(c, (xc, mut w))| {
                        with_sc(|scr| {
                            let [b0, b1, b2, b3, b4, ..] = scr;
                            let pv = w.get(b0);
                            let (rg, rgp) = (gr.rd_at(t, c), gpr.rd_at(t, c));
                            let (gv, gpv) = (rg.get(b1), rgp.get(b2));
                            b3.clear();
                            for i in 0..xc.len() {
                                let s = if xd { pv[i] } else { C32::new(xc[i].re - pv[i].re, xc[i].im - pv[i].im) };
                                let m = C32::new(gv[i].re * al - gpv[i].re * al, gv[i].im * al - gpv[i].im * al);
                                let (wv, _) = wz(a, tau, s, m);
                                let gs = sc(gv[i], al);
                                b3.push(C32::new((tau * gs.re as f64 + cc * wv[0]) as f32, (tau * gs.im as f64 + cc * wv[1]) as f32));
                            }
                            // record, then the update
                            let rec = b4;
                            rec.clear();
                            for (v, s) in xc.iter_mut().zip(b3.iter()) {
                                let nv = C32::new(v.re - s.re, v.im - s.im);
                                rec.push(if write_diff { C32::new((nv.re as f64 - v.re as f64) as f32, (nv.im as f64 - v.im as f64) as f32) } else { *v });
                                *v = nv;
                            }
                            w.put(rec);
                        })
                    });
                    let _ = l;
                }
                if per_set && k % 10 == 0 {
                    let (lo, hi) = at.iter().fold((f64::INFINITY, 0.0f64), |(l, h), p| (l.min(p.1), h.max(p.1)));
                    eprintln!("    zmshr1 per set it {k}: tau {lo:.3e}..{hi:.3e}, a > 0 in {}/{nt} sets, rank-one in {}/{nt}", at.iter().filter(|p| p.0 > 0.0).count(), ct.iter().filter(|&&c| c != 0.0).count());
                }
                xp_is_diff = write_diff;
            }
            Precond::Zmshr1M | Precond::Zmshr1MSet => {
                // s_t and m_t into the set temporaries A, B; the metric into T (three temporaries)
                let fill_s = |xp: &Buf, xt: &[C32], t: usize, out: &mut [C32]| {
                    xp.read_set(t, out);
                    if !xp_is_diff {
                        out.par_chunks_mut(PAR).zip(xt.par_chunks(PAR)).for_each(|(o, xv)| o.iter_mut().zip(xv).for_each(|(o, x)| *o = C32::new(x.re - o.re, x.im - o.im)));
                    }
                };
                let fill_m = |g: &Buf, gp: &Buf, t: usize, out: &mut [C32]| {
                    let l = out.len();
                    out.par_chunks_mut(PAR).enumerate().for_each(|(c, o)| {
                        with_sc(|scr| {
                            let [b0, b1, ..] = scr;
                            let (rg, rgp) = (g.rd_at(t, c), gp.rd_at(t, c));
                            let (gv, gpv) = (rg.get(b0), rgp.get(b1));
                            for (i, ov) in o.iter_mut().enumerate() {
                                *ov = C32::new(gv[i].re * al - gpv[i].re * al, gv[i].im * al - gpv[i].im * al);
                            }
                        })
                    });
                    let _ = l;
                };
                let rdot = |a: &[C32], b: &[C32]| -> f64 { a.par_chunks(PAR).zip(b.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum() };
                let mut d1: Vec<[f64; 3]> = Vec::with_capacity(nt);
                for t in 0..nt {
                    let l = lens[t];
                    fill_s(&xp, x.chunk(t), t, &mut ta[..l]);
                    obj.metric_host(st, t, &ta[..l], &mut tt[..l], true)?;
                    let d0 = rdot(&ta[..l], &tt[..l]);
                    fill_m(&g, &gp, t, &mut tb[..l]);
                    let dsm = rdot(&ta[..l], &tb[..l]);
                    obj.metric_host(st, t, &tb[..l], &mut tt[..l], false)?;
                    d1.push([d0, dsm, rdot(&tb[..l], &tt[..l])]);
                }
                let at = zmshr1_scalars_grouped(&d1, per_set, groups.as_deref());
                // w = (1 − τa) s − τ(1−a) M m → into the record buffer (it is rewritten by the step), z = (1−a) m + a M⁻¹s
                let mut d2: Vec<[f64; 4]> = Vec::with_capacity(nt);
                for t in 0..nt {
                    let (l, (a, tau)) = (lens[t], at[t]);
                    fill_s(&xp, x.chunk(t), t, &mut ta[..l]);
                    fill_m(&g, &gp, t, &mut tb[..l]);
                    obj.metric_host(st, t, &tb[..l], &mut tt[..l], false)?;
                    let (c1, c2) = ((1.0 - tau * a) as f32, (-tau * (1.0 - a)) as f32);
                    {
                        let (sa, mm) = (&ta[..l], &tt[..l]);
                        let ws = xp.wr_all(t);
                        ws.into_par_iter().enumerate().for_each(|(c, mut w)| {
                            let (c0, c1e) = (c * PAR, ((c + 1) * PAR).min(l));
                            let v: Vec<C32> = sa[c0..c1e].iter().zip(&mm[c0..c1e]).map(|(s, q)| C32::new(c2 * q.re + c1 * s.re, c2 * q.im + c1 * s.im)).collect();
                            w.put(&v);
                        });
                    }
                    obj.metric_host(st, t, &ta[..l], &mut tt[..l], true)?;
                    let (za, zb) = (a as f32, (1.0 - a) as f32);
                    tb[..l].par_iter_mut().zip(tt[..l].par_iter()).for_each(|(m, si)| *m = C32::new(za * si.re + zb * m.re, za * si.im + zb * m.im));
                    // dots with w as stored (the step reads the same values)
                    let zr = &tb[..l];
                    let (xpr, gr) = (&xp, &g);
                    let acc = (0..chunks_of(l))
                        .into_par_iter()
                        .map(|c| {
                            with_sc(|scr| {
                                let [b0, b1, ..] = scr;
                                let (rw, rg) = (xpr.rd_at(t, c), gr.rd_at(t, c));
                                let (wv, gv) = (rw.get(b0), rg.get(b1));
                                let z = &zr[c * PAR..((c + 1) * PAR).min(l)];
                                let mut a = [0.0f64; 4];
                                for i in 0..z.len() {
                                    let (w0, w1, z0, z1) = (wv[i].re as f64, wv[i].im as f64, z[i].re as f64, z[i].im as f64);
                                    let gs = sc(gv[i], al);
                                    a[0] += w0 * z0 + w1 * z1;
                                    a[1] += w0 * w0 + w1 * w1;
                                    a[2] += z0 * z0 + z1 * z1;
                                    a[3] += w0 * gs.re as f64 + w1 * gs.im as f64;
                                }
                                a
                            })
                        })
                        .reduce(|| [0.0; 4], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]);
                    d2.push(acc);
                }
                let ct = zmshr1_rank_one_grouped(&d2, per_set, groups.as_deref());
                if k % 10 == 0 {
                    let (lo, hi) = at.iter().fold((f64::INFINITY, 0.0f64), |(l, h), p| (l.min(p.1), h.max(p.1)));
                    eprintln!("    zmshr1-M{} it {k}: a {:.3} tau {lo:.3e}..{hi:.3e}, rank-one in {}/{nt}", if per_set { " per set" } else { "" }, at[0].0, ct.iter().filter(|&&c| c != 0.0).count());
                }
                // step = τ M α g + c w (w in the record buffer), fused with the record write and the update
                for t in 0..nt {
                    let (l, (_, tau), cc) = (lens[t], at[t], ct[t]);
                    g.read_set(t, &mut ta[..l]);
                    ta[..l].par_iter_mut().for_each(|v| *v = sc(*v, al));
                    obj.metric_host(st, t, &ta[..l], &mut tt[..l], false)?;
                    let mg = &tt[..l];
                    let xs: Vec<&mut [C32]> = x.chunk_mut(t).chunks_mut(PAR).collect();
                    let ws = xp.wr_all(t);
                    xs.into_par_iter().zip(ws.into_par_iter()).enumerate().for_each(|(c, (xc, mut w))| {
                        with_sc(|scr| {
                            let [b0, b1, b2, ..] = scr;
                            b1.clear();
                            b1.extend_from_slice(w.get(b0));
                            let wv = &*b1;
                            let m = &mg[c * PAR..((c + 1) * PAR).min(l)];
                            let (tf, cf) = (tau as f32, cc as f32);
                            let rec = b2;
                            rec.clear();
                            for i in 0..xc.len() {
                                // as metric_step_fused: c ≠ 0: c w + τ M g; else τ M g
                                let s = if cc != 0.0 { C32::new(cf * wv[i].re + tf * m[i].re, cf * wv[i].im + tf * m[i].im) } else if tau != 1.0 { C32::new(tf * m[i].re, tf * m[i].im) } else { m[i] };
                                let nv = C32::new(xc[i].re - s.re, xc[i].im - s.im);
                                rec.push(if write_diff { C32::new((nv.re as f64 - xc[i].re as f64) as f32, (nv.im as f64 - xc[i].im as f64) as f32) } else { xc[i] });
                                xc[i] = nv;
                            }
                            w.put(rec);
                        })
                    });
                }
                xp_is_diff = write_diff;
            }
        }
        // the objective's trust region on the step: step = x_prev − x, shrunk; x moves by the shrinkage only, so
        // unshrunk elements keep x exactly. With an x_prev record the step it sees (and a shrunk element's new value)
        // carries the record's rounding, 2^−b |x| (velocity model + denoiser + BFP record only); the difference
        // record follows the shrinkage. Set-sized temporaries here (rec, s0, s): velocity runs, not in the ledger.
        if obj.has_step_limit() {
            if quasi {
                let mut n = 0usize;
                for t in 0..nt {
                    let l = lens[t];
                    let mut rec = vec![C32::new(0.0, 0.0); l];
                    xp.read_set(t, &mut rec);
                    let s0: Vec<C32> = if xp_is_diff { rec.iter().map(|d| C32::new(-d.re, -d.im)).collect() } else { rec.iter().zip(x.chunk(t)).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect() };
                    let mut s = s0.clone();
                    let nt_ = obj.limit_step_host(t, &mut s);
                    if nt_ > 0 {
                        n += nt_;
                        for ((xo, a), b) in x.chunk_mut(t).iter_mut().zip(&s0).zip(&s) {
                            *xo = C32::new(xo.re + (a.re - b.re), xo.im + (a.im - b.im));
                        }
                        if xp_is_diff {
                            let d: Vec<C32> = s.iter().map(|v| C32::new(-v.re, -v.im)).collect();
                            xp.write_set(t, &d);
                        }
                    }
                }
                if n > 0 {
                    eprintln!("    step trust region (it {k}): {n} elements shrunk");
                }
            } else if k == k0 {
                eprintln!("    step trust region: needs a quasi-Newton preconditioner on host stacks (the previous iterate); not applied");
            }
        }
        // the pair's gradient: g_prev ← g (a swap when both have the same storage)
        if quasi {
            if gp.same_kind(&g) {
                std::mem::swap(&mut gp, &mut g);
            } else {
                for t in 0..nt {
                    let mut v = vec![C32::new(0.0, 0.0); lens[t]];
                    g.read_set(t, &mut v);
                    gp.write_set(t, &v);
                }
            }
        }
        tm.lap(st, 1);
        // the denoiser, and whether it changed x (only while the difference record is possible)
        let check = pair_bfp && diff_ok != Some(false);
        let h0 = check.then(|| fingerprint(&x.data));
        denoise(k, &mut x.data);
        if let Some(h0) = h0 {
            let same = fingerprint(&x.data) == h0;
            if !same {
                if xp_is_diff {
                    pair_invalid = true;
                }
                if diff_ok != Some(false) {
                    eprintln!("    it {k}: the denoiser changes x: secant step stored as x_prev (bfp{}; s error 2^-b ‖x‖)", store.pair.unwrap().bits());
                    let low = [store.grad, store.pair].iter().flatten().map(|b| b.bits()).min().unwrap_or(32);
                    if low < 20 {
                        eprintln!("    WARNING: P²nP (a denoiser after the step) with a {low}-bit state: g does not vanish at the fixed point and s is a difference of non-vanishing iterates; measured fixed-point bias ≈ 2 · 2^-b with ZMSHR1 (research/bfp_storage.md §9). Use --state-store bfp24 for P²nP, bfp16 for RED / no prior.");
                    }
                }
                diff_ok = Some(false);
            } else if diff_ok.is_none() {
                diff_ok = Some(true);
                eprintln!("    it {k}: no post-step denoiser: secant step stored as the difference x − x_prev from the next iteration on");
            }
        }
        tm.lap(st, 2);
        report(k, x, f);
        tm.lap(st, 3);
        tm.log(k, ctl);
        if let Some((c, name)) = ctl {
            let next = k + 1;
            if c.stop_requested() && next < iters {
                save(c, name, next, x, &xp, &gp, xp_is_diff, diff_ok, pair_invalid)?;
                return Ok(Outcome::Stopped(next));
            }
            if c.due(next) && next < iters {
                save(c, name, next, x, &xp, &gp, xp_is_diff, diff_ok, pair_invalid)?;
            }
        }
    }
    if let Some((c, name)) = ctl {
        let _ = std::fs::remove_file(ck_path(c, name));
    }
    Ok(Outcome::Finished)
}

/// `state_<name>.hsb` exists (a checkpoint of [`run_objective_host_bfp`])
pub fn has_checkpoint(c: &RunCtl, name: &str) -> bool {
    ck_path(c, name).exists()
}
