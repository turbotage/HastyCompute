//! Array files with a precision policy: one call to save an array (or a value tree) as a compressed `.hv` file, one
//! call to load either a `.hv` or a legacy `.npy` file. Policy and measurements: `research/io_policy.md`.
//!
//! ```text
//!   store::save_bytes(path, DType::C64, &[x, y, z, e, f], Order::F, bytes, &Save::image())?   // → path.hv
//!   let (t, order) = store::load_array(path)?                                                   // .hv or .npy
//! ```
//!
//! **Single-array files.** The root is a dict `{"array": tensor, "order": "C" | "F", "loss": "<json>"}`. A
//! Fortran-ordered array (hasty's x-fastest volumes) of shape (X, Y, Z, …) is stored as the C-order tensor
//! (…, Z, Y, X) with the same bytes, exactly as [`crate::npy::read`] returns a Fortran `.npy`; `order` says so and the
//! Python reader (`hastycompute.store.load_any`) returns the same array `np.load` returned for the `.npy`. `loss`
//! records the policy and the resolved quantiser step (self-describing error bound).
//!
//! **Policies** ([`Policy`]) act on float32 / float64 / complex tensors; integer, bool, float16 / bfloat16 tensors
//! and float tensors that hold only small integers (masks, labels stored as float) are always stored losslessly.
//! Non-finite values (NaN backgrounds) are excluded from the scale statistics; chunks containing them are stored
//! losslessly.
//!
//! **Streaming, bounded memory.** [`TensorWriter`] writes a chunked tensor while its bytes arrive (C order, any
//! piece sizes): chunks are encoded in parallel groups (≤ 256 MiB raw in flight) while a writer thread puts the previous
//! group on disk; the chunk index is reserved up front and filled in at the end. Files are written to `<path>.tmp`
//! and renamed when complete.
//!
//! **Threads.** Encoding and decoding run on a dedicated pool of `HASTY_IO_THREADS` threads (default
//! min(available cores, 12)); file reads are positional (`pread`) per chunk, issued from the pool, so reads of many
//! chunks overlap.
//!
//! **Environment.**
//! - `HASTY_IO=npy` makes the savers write plain `.npy` files (lossless) instead of `.hv` (compatibility);
//! - `HASTY_IO_LOSSLESS=1` keeps the `.hv` format but disables every quantiser (inverse-crime checks, debugging).

use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rayon::prelude::*;

use crate::format::{self, MAGIC, TAG_CHUNKED, TAG_DICT, TAG_QCHUNKED, VERSION};
use crate::lazy::{Lazy, LazyFile, TensorRef};
use crate::lossy::{self, Loss, Quant, Scope, Transfer};
use crate::{Codec, DType, Error, Options, Result, Tensor, Value};

/// default relative precision of images and reconstruction outputs (quantisation noise rms / data rms)
pub const IMAGE_REL: f64 = 1e-4;
/// k-space: quantiser step as a fraction of the thermal noise std per real component (`kspace_step`)
pub const KSPACE_NOISE_FRAC: f64 = 1.0 / 8.0;
/// k-space: quantisation noise rms relative to the data rms, for noise-free (clean) data (`kspace_step`)
pub const KSPACE_REL: f64 = 1e-4;
/// solver checkpoints: quantisation noise rms / rms of each stored vector (its own scale; research/io_policy.md §2)
pub const CHECKPOINT_REL: f64 = 1e-6;
/// checkpoint tensors with fewer elements are stored losslessly (parameter vectors of mixed units)
pub const CHECKPOINT_MIN_LOSSY: usize = 1 << 16;
/// raw bytes per chunk of the automatic chunking
pub const AUTO_CHUNK_BYTES: usize = 4 << 20;
/// raw bytes encoded per parallel group (bounds the compressed bytes in flight)
const GROUP_BYTES: usize = 256 << 20;

/// Memory order of the array as the caller sees it (`F`: first index fastest, hasty's x-fastest volumes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    C,
    F,
}

impl Order {
    pub fn is_f(self) -> bool {
        self == Order::F
    }
    fn tag(self) -> &'static str {
        match self {
            Order::C => "C",
            Order::F => "F",
        }
    }
}

/// Precision policy for float / complex data (see the module docs; numbers in `research/io_policy.md`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Policy {
    /// bit-exact
    Lossless,
    /// real data: uniform step rel·√12·rms (rms over the finite scalars of the tensor): quantisation noise rms =
    /// rel·rms, pointwise |err| ≤ step/2 = 1.73·rel·rms. Complex data (images whose phase carries the signal:
    /// velocity encodes, coil maps): pointwise-relative mantissa rounding to m = ⌈log2(1/rel)⌉ − 1 bits, |err| ≤
    /// 2^−(m+1)·|component| ≤ rel·|component| — a uniform step would cost low-magnitude voxels their phase
    /// (measured: 0.66 rad worst phase-difference error in a lumen series at rel 1e-4, 2e-4 rad with the mantissa).
    /// Images, reconstructions, maps.
    Image { rel: f64 },
    /// uniform step rel·√12·rms for any float dtype (complex: re and im with one scale): quantisation noise rms =
    /// rel·rms of the tensor (its own scale). Solver checkpoints.
    Uniform { rel: f64 },
    /// asinh transfer with knee k = floor·rms: pointwise |err| ≲ rel·√(k² + x²), i.e. relative precision `rel` above
    /// the knee, absolute rel·k below. High-dynamic-range data where small values matter relatively.
    Relative { rel: f64, floor: f64 },
    /// uniform absolute step: |err| ≤ step/2 (k-space data: [`kspace_step`], ADC-like)
    Abs(f64),
    /// per chunk: signed integers of this many bits spanning ± the chunk's max |x| (|err| ≤ max / (2^bits − 2))
    Bits(u32),
    /// keep m mantissa bits: pointwise relative error ≤ 2^−(m+1) (ordinary float tensor in the file)
    Mantissa(u32),
}

impl Policy {
    pub fn image() -> Self {
        Policy::Image { rel: IMAGE_REL }
    }
}

/// Chunk shape of a saved tensor (C-order axes).
#[derive(Clone, Debug, PartialEq)]
pub enum Chunking {
    /// contiguous slabs of about [`AUTO_CHUNK_BYTES`]
    Auto,
    /// contiguous slabs of about this many raw bytes
    Bytes(usize),
    /// this many entries of the first (slowest, C-order) axis per chunk (k-space: 1 readout)
    Rows(usize),
    /// explicit C-order chunk shape (e.g. one field of one readout, `[1, n_adc, 1]`, for per-field scales)
    Shape(Vec<usize>),
}

/// How to save.
#[derive(Clone, Debug)]
pub struct Save {
    pub policy: Policy,
    pub codec: Codec,
    pub chunk: Chunking,
    /// delta filter on quantised integer words (smooth data: trajectories, smooth maps)
    pub delta: bool,
}

impl Default for Save {
    fn default() -> Self {
        Self { policy: Policy::Lossless, codec: Codec::Zstd(1), chunk: Chunking::Auto, delta: false }
    }
}

impl Save {
    pub fn lossless() -> Self {
        Self::default()
    }
    /// [`Policy::Image`] at [`IMAGE_REL`]
    pub fn image() -> Self {
        Self { policy: Policy::image(), ..Self::default() }
    }
    pub fn policy(mut self, p: Policy) -> Self {
        self.policy = p;
        self
    }
    pub fn chunk(mut self, c: Chunking) -> Self {
        self.chunk = c;
        self
    }
    pub fn codec(mut self, c: Codec) -> Self {
        self.codec = c;
        self
    }
    pub fn delta(mut self, d: bool) -> Self {
        self.delta = d;
        self
    }
    pub fn options(&self) -> Options {
        Options { codec: self.codec, delta: self.delta, ..Options::default() }
    }
}

/// k-space quantiser step (uniform, ADC-like) for complex data with thermal noise std `sigma` per complex sample
/// (σ/√2 per real component) and data rms `rms` per complex sample (noise-free data):
/// step = min(KSPACE_NOISE_FRAC · σ/√2, KSPACE_REL · √12 · rms/√2). The first term keeps the quantisation noise
/// (step²/12) at 0.13 % of the thermal noise variance; the second keeps noise-free controls at 1e-4 relative rms.
/// `sigma` ≤ 0 (unknown / noise-free sim) uses the second term alone.
pub fn kspace_step(sigma: f64, rms: f64) -> f64 {
    let a = KSPACE_REL * 12f64.sqrt() * rms / 2f64.sqrt();
    if sigma > 0.0 {
        (KSPACE_NOISE_FRAC * sigma / 2f64.sqrt()).min(a)
    } else {
        a
    }
}

/// The file format new files are written in (`HASTY_IO`, default `.hv`).
pub fn write_hv() -> bool {
    !matches!(std::env::var("HASTY_IO").as_deref(), Ok("npy"))
}

/// `HASTY_IO_LOSSLESS=1`: every saver stores bit-exact data
pub fn force_lossless() -> bool {
    matches!(std::env::var("HASTY_IO_LOSSLESS").as_deref(), Ok("1") | Ok("true") | Ok("yes"))
}

/// the path a saver writes for `path` (its extension replaced by `.hv`, or `.npy` with `HASTY_IO=npy`)
pub fn out_path(path: &Path) -> PathBuf {
    path.with_extension(if write_hv() { "hv" } else { "npy" })
}

/// after `written` (a new version of the array) exists: remove the sibling with the other extension, an older
/// version that `resolve` would otherwise find first (`.npy` is preferred when both exist)
fn drop_stale(written: &Path) {
    let alt = match written.extension().and_then(|e| e.to_str()) {
        Some("hv") => written.with_extension("npy"),
        Some("npy") => written.with_extension("hv"),
        _ => return,
    };
    if alt != written && alt.exists() {
        let _ = std::fs::remove_file(&alt);
    }
}

/// `path` if it exists, else the same name with the other extension (`.npy` ↔ `.hv`)
pub fn resolve(path: &Path) -> Option<PathBuf> {
    if path.exists() {
        return Some(path.to_path_buf());
    }
    let alt = match path.extension().and_then(|e| e.to_str()) {
        Some("npy") => path.with_extension("hv"),
        Some("hv") => path.with_extension("npy"),
        _ => return None,
    };
    alt.exists().then_some(alt)
}

/// the array exists as `.npy` or `.hv`
pub fn exists_any(path: &Path) -> bool {
    resolve(path).is_some()
}

/// the I/O thread pool (`HASTY_IO_THREADS`, default min(cores, 12))
pub fn io_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("HASTY_IO_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12));
        rayon::ThreadPoolBuilder::new().num_threads(n.max(1)).thread_name(|i| format!("hasty-io-{i}")).build().expect("I/O thread pool")
    })
}

// ── statistics and policy resolution ──

/// scale statistics over the finite scalars
#[derive(Clone, Copy, Debug, Default)]
struct Scan {
    ss: f64,
    max: f64,
    n: usize,
    nonfinite: usize,
    integral: bool,
}

impl Scan {
    fn merge(a: Scan, b: Scan) -> Scan {
        Scan { ss: a.ss + b.ss, max: a.max.max(b.max), n: a.n + b.n, nonfinite: a.nonfinite + b.nonfinite, integral: a.integral && b.integral }
    }
    fn rms(&self) -> f64 {
        (self.ss / self.n.max(1) as f64).sqrt()
    }
}

fn scan_part(dtype: DType, b: &[u8]) -> Scan {
    let mut s = Scan { integral: true, ..Scan::default() };
    let mut f = |x: f64| {
        if x.is_finite() {
            s.ss += x * x;
            s.max = s.max.max(x.abs());
            s.n += 1;
            s.integral &= x == x.round() && x.abs() < 16_777_216.0;
        } else {
            s.nonfinite += 1;
        }
    };
    if lossy::scalar_kind(dtype).unwrap().0 {
        b.chunks_exact(8).for_each(|c| f(f64::from_le_bytes(c.try_into().unwrap())));
    } else {
        b.chunks_exact(4).for_each(|c| f(f32::from_le_bytes(c.try_into().unwrap()) as f64));
    }
    s
}

fn scan(dtype: DType, bytes: &[u8]) -> Scan {
    let es = if lossy::scalar_kind(dtype).unwrap().0 { 8 } else { 4 };
    bytes.par_chunks((1 << 20) / es * es).map(|c| scan_part(dtype, c)).reduce(|| Scan { integral: true, ..Scan::default() }, Scan::merge)
}

/// the quantiser for `policy` on these data (None: lossless) and the record stored in the file's "loss" entry
pub fn resolve_policy(policy: Policy, dtype: DType, data: Option<&[&[u8]]>) -> Result<(Option<Quant>, String)> {
    let policy = if force_lossless() { Policy::Lossless } else { policy };
    let lossless = |why: &str| Ok((None, serde_json::json!({"policy": "lossless", "why": why}).to_string()));
    if policy == Policy::Lossless {
        return lossless("requested");
    }
    if lossy::scalar_kind(dtype).is_none() {
        return lossless("not a float32/64 or complex dtype");
    }
    let needs_scan = matches!(policy, Policy::Image { .. } | Policy::Uniform { .. } | Policy::Relative { .. });
    let st = match (data, needs_scan) {
        (Some(parts), _) => Some(parts.iter().map(|d| scan(dtype, d)).fold(Scan { integral: true, ..Scan::default() }, Scan::merge)),
        (None, true) => return Err(Error::Type("image / relative policies need the data (streaming writers take Abs / Bits / Mantissa)".into())),
        (None, false) => None,
    };
    if let Some(s) = &st {
        if s.n == 0 || s.max == 0.0 {
            return lossless("no nonzero finite values");
        }
        if s.integral {
            return lossless("integer-valued (mask / labels)");
        }
    }
    let rec = |q: &Quant, extra: serde_json::Value| -> String {
        let mut j = serde_json::json!({"policy": format!("{policy:?}"), "loss": format!("{:?}", q.loss), "transfer": format!("{:?}", q.transfer)});
        if let (Some(o), Some(e)) = (j.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        j.to_string()
    };
    let (q, extra) = match policy {
        Policy::Lossless => unreachable!(),
        Policy::Image { rel } if dtype.is_complex() => {
            let m = ((1.0 / rel).log2().ceil() as u32).saturating_sub(1).clamp(1, if lossy::scalar_kind(dtype).unwrap().0 { 52 } else { 23 });
            (Quant::new(Loss::Mantissa(m)), serde_json::json!({"mantissa_bits": m, "max_rel_err": 0.5f64.powi(m as i32 + 1), "why": "complex: pointwise relative (phase)"}))
        }
        Policy::Image { rel } | Policy::Uniform { rel } => {
            let s = st.unwrap();
            let step = rel * 12f64.sqrt() * s.rms();
            (Quant::new(Loss::Abs(step)), serde_json::json!({"rms": s.rms(), "step": step, "max_abs_err": step / 2.0}))
        }
        Policy::Relative { rel, floor } => {
            let s = st.unwrap();
            let k = (floor * s.rms()).max(f64::MIN_POSITIVE);
            let step = 2.0 * rel * k;
            (Quant::new(Loss::Abs(step)).transfer(Transfer::Asinh(k)), serde_json::json!({"rms": s.rms(), "knee": k, "step": step}))
        }
        Policy::Abs(step) => (Quant::new(Loss::Abs(step)), serde_json::json!({"step": step, "max_abs_err": step / 2.0})),
        Policy::Bits(b) => (Quant::new(Loss::RelMax(b)).scope(Scope::Chunk), serde_json::json!({"bits": b})),
        Policy::Mantissa(m) => (Quant::new(Loss::Mantissa(m)), serde_json::json!({"mantissa_bits": m, "max_rel_err": 0.5f64.powi(m as i32 + 1)})),
    };
    q.validate(dtype)?;
    Ok((Some(q), rec(&q, extra)))
}

/// chunk shape (C-order) of `shape` for a chunking: contiguous slabs [1, …, 1, c, full, …]
pub fn chunk_shape(shape: &[usize], es: usize, c: &Chunking) -> Vec<usize> {
    let nd = shape.len();
    if nd == 0 {
        return vec![];
    }
    let target = match c {
        Chunking::Rows(r) => {
            let mut ch = shape.to_vec();
            ch[0] = (*r).clamp(1, shape[0].max(1));
            return ch;
        }
        Chunking::Shape(c) => return c.iter().zip(shape).map(|(&c, &n)| c.clamp(1, n.max(1))).collect(),
        Chunking::Auto => AUTO_CHUNK_BYTES,
        Chunking::Bytes(b) => *b,
    };
    let mut ch = vec![1usize; nd];
    let mut inner = es;
    for d in (0..nd).rev() {
        let n = shape[d].max(1);
        if inner * n <= target {
            ch[d] = n;
            inner *= n;
        } else {
            ch[d] = (target / inner).clamp(1, n);
            break;
        }
    }
    ch
}

// ── building blocks (size / error estimation, custom containers) ──

/// Encode the C-order bytes of one chunk exactly as the writers do (quantised if `quant` is an integer quantiser;
/// Mantissa rounds first). The result is the chunk's bytes in the file.
pub fn encode_chunk(raw: &[u8], dtype: DType, quant: Option<&Quant>, o: &Options) -> Vec<u8> {
    match quant {
        Some(Quant { loss: Loss::Mantissa(m), .. }) => format::encode_chunk_raw(&lossy::round_mantissa(dtype, *m, raw), dtype, None, None, o),
        Some(q) if q.is_int() => format::encode_chunk_raw(raw, dtype, Some(q), None, o),
        _ => format::encode_chunk_raw(raw, dtype, None, None, o),
    }
}

/// Decode a chunk made by [`encode_chunk`] (`numel` elements; `quant` as given to the encoder)
pub fn decode_chunk(bytes: &[u8], dtype: DType, numel: usize, quant: Option<&Quant>, o: &Options) -> Result<Vec<u8>> {
    format::decode_chunk_data(dtype, quant.is_some_and(|q| q.is_int()), &[numel], bytes, o.block_bytes())
}

// ── streaming writers ──

/// The tensor a [`ValueWriter`] is writing.
struct Cur {
    dtype: DType,
    shape: Vec<usize>,
    chunk: Vec<usize>,
    /// contiguous chunks ([1, …, 1, c, full …]): the axis of the partial extent, elements per entry of that axis
    axis: usize,
    inner: usize,
    grid_axis: usize,
    /// otherwise the unit of streaming is a slab of chunk[0] rows holding `per_unit` chunks
    slab: bool,
    grid: Vec<usize>,
    per_unit: usize,
    nunits: usize,
    nchunks: usize,
    total: usize,
    quant: Option<Quant>,
    opts: Options,
    index: Vec<(u64, u64)>,
    index_pos: u64,
    data_len: u64,
    next: usize,
    pending: Vec<u8>,
}

impl Cur {
    /// byte range of streaming unit `u` (a chunk, or a slab of chunk[0] rows) in the C-order data
    fn unit_range(&self, u: usize) -> (usize, usize) {
        let es = self.dtype.size();
        if self.slab {
            let row = self.shape[1..].iter().product::<usize>() * es;
            let c0 = self.chunk[0];
            return (u * c0 * row, c0.min(self.shape[0] - u * c0) * row);
        }
        let (o, j) = (u / self.grid_axis, u % self.grid_axis);
        let c = self.chunk[self.axis];
        let start = (o * self.shape[self.axis] + j * c) * self.inner * es;
        let len = c.min(self.shape[self.axis] - j * c) * self.inner * es;
        (start, len)
    }

    fn consumed(&self) -> usize {
        if self.next >= self.nunits {
            self.total
        } else {
            self.unit_range(self.next).0 + self.pending.len()
        }
    }
}

/// Writes a dict-rooted value tree to a `.hv` file while big tensors stream in (bounded memory): `begin_tensor` /
/// `push` / `end_tensor` for chunked tensors (encoded in parallel groups of ≤ 256 MiB raw while a writer thread puts
/// the previous group on disk; the chunk index is reserved and filled in at `finish`), `put` for small values. The
/// file is `<path>.tmp` until [`finish`](Self::finish) renames it; a writer dropped unfinished removes it.
pub struct ValueWriter {
    path: PathBuf,
    tmp: PathBuf,
    /// bytes handed to the writer thread so far (= file position)
    pos: u64,
    /// dict entries still to write
    left: usize,
    cur: Option<Cur>,
    /// (file offset, bytes) written at `finish` (chunk indices)
    patches: Vec<(u64, Vec<u8>)>,
    tx: Option<std::sync::mpsc::SyncSender<Vec<Vec<u8>>>>,
    io: Option<std::thread::JoinHandle<std::io::Result<File>>>,
}

impl ValueWriter {
    /// a file whose root is a dict of `entries` entries
    pub fn create(path: &Path, entries: usize) -> Result<Self> {
        let tmp = PathBuf::from(format!("{}.tmp", path.display()));
        let f = File::create(&tmp)?;
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<Vec<u8>>>(2);
        let io = std::thread::spawn(move || -> std::io::Result<File> {
            let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
            for group in rx {
                for c in group {
                    w.write_all(&c)?;
                }
            }
            w.flush()?;
            w.into_inner().map_err(|e| e.into_error())
        });
        let mut w = Self { path: path.to_path_buf(), tmp, pos: 0, left: entries, cur: None, patches: vec![], tx: Some(tx), io: Some(io) };
        let mut h = Vec::with_capacity(17);
        h.extend_from_slice(&MAGIC);
        h.extend_from_slice(&[VERSION, 0, 0, 0]);
        h.push(TAG_DICT);
        h.extend_from_slice(&(entries as u64).to_le_bytes());
        w.send(vec![h])?;
        Ok(w)
    }

    fn send(&mut self, bufs: Vec<Vec<u8>>) -> Result<()> {
        self.pos += bufs.iter().map(|b| b.len() as u64).sum::<u64>();
        self.tx.as_ref().unwrap().send(bufs).map_err(|_| Error::Io(std::io::Error::other("writer thread ended")))
    }

    fn key(&mut self, k: &str) -> Result<()> {
        if self.cur.is_some() {
            return Err(Error::Format("ValueWriter: a tensor is still open".into()));
        }
        if self.left == 0 {
            return Err(Error::Format("ValueWriter: more entries than declared".into()));
        }
        self.left -= 1;
        let mut b = (k.len() as u32).to_le_bytes().to_vec();
        b.extend_from_slice(k.as_bytes());
        self.send(vec![b])
    }

    /// a small value (encoded in memory, lossless)
    pub fn put(&mut self, key: &str, v: &Value) -> Result<()> {
        self.key(key)?;
        let mut b = format::encode_to_vec(v, &Options::zstd(1))?;
        b.drain(..crate::format::STREAM_HEADER);
        self.send(vec![b])
    }

    /// start the chunked tensor `key` of C-order `shape` (stored shape), quantiser `quant` (None: lossless)
    pub fn begin_tensor(&mut self, key: &str, dtype: DType, shape: &[usize], save: &Save, quant: Option<Quant>) -> Result<()> {
        let nd = shape.len();
        if nd == 0 || nd > 255 {
            return Err(Error::Format(format!("ValueWriter: {nd}-dimensional tensor (1..=255)")));
        }
        self.key(key)?;
        let cshape = shape.to_vec();
        let es = dtype.size();
        let chunk = chunk_shape(&cshape, es, &save.chunk);
        // contiguous chunks [1, …, 1, c, full …] stream chunk by chunk; any other shape by slabs of chunk[0] rows
        let axis = (0..nd).rev().find(|&d| chunk[d] < cshape[d]).unwrap_or(0);
        let slab = (0..axis).any(|d| chunk[d] != 1);
        let inner: usize = cshape[axis + 1..].iter().product();
        let grid: Vec<usize> = cshape.iter().zip(&chunk).map(|(&n, &c)| n.div_ceil(c)).collect();
        let total = cshape.iter().product::<usize>() * es;
        let nchunks = if total == 0 { 0 } else { grid.iter().product() };
        let per_unit = if slab { grid[1..].iter().product() } else { 1 };
        let nunits = if total == 0 { 0 } else if slab { grid[0] } else { nchunks };
        let opts = save.options();
        let mut h = Vec::with_capacity(16 + 16 * nd);
        let tag = if quant.as_ref().is_some_and(|q| q.is_int()) { TAG_QCHUNKED } else { TAG_CHUNKED };
        h.extend_from_slice(&[tag, dtype as u8, nd as u8, 0]);
        h.extend_from_slice(&(opts.block_bytes() as u32).to_le_bytes());
        for &d in cshape.iter().chain(&chunk) {
            h.extend_from_slice(&(d as u64).to_le_bytes());
        }
        h.extend_from_slice(&(nchunks as u64).to_le_bytes());
        let index_pos = self.pos + h.len() as u64;
        self.send(vec![h, vec![0u8; 16 * nchunks]])?;
        self.cur = Some(Cur { dtype, grid_axis: grid[axis], slab, grid, per_unit, nunits, shape: cshape, chunk, axis, inner, nchunks, total, quant, opts, index: Vec::with_capacity(nchunks), index_pos, data_len: 0, next: 0, pending: Vec::new() });
        Ok(())
    }

    /// encode the complete units at the start of `src` (which begins at unit `next`); returns bytes used
    fn encode_run(&mut self, src: &[u8]) -> Result<usize> {
        let c = self.cur.as_ref().unwrap();
        if c.next >= c.nunits {
            return Ok(0);
        }
        let base = c.unit_range(c.next).0;
        let mut used = 0usize;
        loop {
            let c = self.cur.as_ref().unwrap();
            // the group: complete units from `next` within GROUP_BYTES (at least one); jobs (unit offset, len, chunk m)
            let mut jobs = vec![];
            let mut end = used;
            let mut u = c.next;
            while u < c.nunits {
                let (s, l) = c.unit_range(u);
                let rel = s - base + l;
                if rel > src.len() || (u > c.next && rel - used > GROUP_BYTES) {
                    break;
                }
                jobs.extend((0..c.per_unit).map(|m| (s - base, l, m)));
                end = rel;
                u += 1;
            }
            if jobs.is_empty() {
                return Ok(used);
            }
            let (dtype, quant, opts, es) = (c.dtype, c.quant, c.opts, c.dtype.size());
            let (slab, shape, chunk, grid) = (c.slab, &c.shape, &c.chunk, &c.grid);
            let enc = |&(off, len, m): &(usize, usize, usize)| -> Vec<u8> {
                let unit = &src[off..off + len];
                let boxed;
                let raw: &[u8] = if slab {
                    // chunk m of the slab: its box in the slab of shape [rows, shape[1..]]
                    let row = shape[1..].iter().product::<usize>() * es;
                    let mut sshape = shape.clone();
                    sshape[0] = len / row;
                    let (ilo, iext) = format::chunk_box(m, &grid[1..], &chunk[1..], &shape[1..]);
                    let lo: Vec<usize> = std::iter::once(0).chain(ilo).collect();
                    let ext: Vec<usize> = std::iter::once(sshape[0]).chain(iext).collect();
                    let mut b = vec![0u8; ext.iter().product::<usize>() * es];
                    format::copy_sub(unit, &sshape, &lo, &mut b, &ext, &vec![0; ext.len()], &ext, es);
                    boxed = b;
                    &boxed
                } else {
                    unit
                };
                encode_chunk(raw, dtype, quant.as_ref(), &opts)
            };
            let encoded: Vec<Vec<u8>> = io_pool().install(|| jobs.par_iter().map(enc).collect());
            let c = self.cur.as_mut().unwrap();
            for e in &encoded {
                c.index.push((c.data_len, e.len() as u64));
                c.data_len += e.len() as u64;
            }
            c.next = u;
            used = end;
            self.send(encoded)?;
        }
    }

    /// append the next bytes of the open tensor (C order, any piece sizes)
    pub fn push(&mut self, mut data: &[u8]) -> Result<()> {
        let c = self.cur.as_ref().ok_or_else(|| Error::Format("ValueWriter: no tensor open".into()))?;
        if c.consumed() + data.len() > c.total {
            return Err(Error::Format(format!("{}: more data than the tensor holds", self.path.display())));
        }
        if !c.pending.is_empty() {
            let unit = c.unit_range(c.next).1;
            let c = self.cur.as_mut().unwrap();
            let take = (unit - c.pending.len()).min(data.len());
            c.pending.extend_from_slice(&data[..take]);
            data = &data[take..];
            if c.pending.len() < unit {
                return Ok(());
            }
            let p = std::mem::take(&mut c.pending);
            self.encode_run(&p)?;
        }
        if data.is_empty() {
            return Ok(());
        }
        let used = self.encode_run(data)?;
        self.cur.as_mut().unwrap().pending.extend_from_slice(&data[used..]);
        Ok(())
    }

    /// close the open tensor (all its bytes pushed)
    pub fn end_tensor(&mut self) -> Result<()> {
        let c = self.cur.take().ok_or_else(|| Error::Format("ValueWriter: no tensor open".into()))?;
        if c.next != c.nunits || !c.pending.is_empty() || c.index.len() != c.nchunks {
            return Err(Error::Format(format!("{}: {} of {} bytes written", self.path.display(), c.consumed(), c.total)));
        }
        let idx: Vec<u8> = c.index.iter().flat_map(|&(o, l)| [o.to_le_bytes(), l.to_le_bytes()].concat()).collect();
        self.patches.push((c.index_pos, idx));
        Ok(())
    }

    /// complete the file (every declared entry written): indices, rename. Returns (path, file bytes).
    pub fn finish(mut self) -> Result<(PathBuf, u64)> {
        if self.cur.is_some() {
            self.end_tensor()?;
        }
        if self.left != 0 {
            return Err(Error::Format(format!("{}: {} dict entries missing", self.path.display(), self.left)));
        }
        drop(self.tx.take());
        let f = self.io.take().unwrap().join().map_err(|_| Error::Io(std::io::Error::other("writer thread panicked")))??;
        for (off, b) in &self.patches {
            f.write_all_at(b, *off)?;
        }
        drop(f);
        std::fs::rename(&self.tmp, &self.path)?;
        Ok((self.path.clone(), self.pos))
    }
}

impl Drop for ValueWriter {
    fn drop(&mut self) {
        // an unfinished writer leaves no partial file under the real name; remove the temporary
        if let Some(io) = self.io.take() {
            drop(self.tx.take());
            let _ = io.join();
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// Writes one chunked tensor (a single-array `.hv` file) from its C-order bytes as they arrive.
pub struct TensorWriter {
    w: ValueWriter,
    order: Order,
    loss: String,
}

impl TensorWriter {
    /// Start `path` (written as `path.tmp` until [`finish`](Self::finish)). `shape` as the caller sees the array (for
    /// `Order::F` the stored C-order shape is reversed). `data` (all bytes, optional) lets image / relative policies
    /// compute their scale; streaming writers pass None and an Abs / Bits / Mantissa / Lossless policy.
    pub fn create(path: &Path, dtype: DType, shape: &[usize], order: Order, save: &Save, data: Option<&[u8]>) -> Result<Self> {
        let parts: Option<Vec<&[u8]>> = data.map(|d| vec![d]);
        let (quant, loss) = resolve_policy(save.policy, dtype, parts.as_deref())?;
        Self::create_with(path, dtype, shape, order, save, quant, loss)
    }

    fn create_with(path: &Path, dtype: DType, shape: &[usize], order: Order, save: &Save, quant: Option<Quant>, loss: String) -> Result<Self> {
        let cshape: Vec<usize> = if order.is_f() { shape.iter().rev().copied().collect() } else { shape.to_vec() };
        let mut w = ValueWriter::create(path, 3)?;
        w.begin_tensor("array", dtype, &cshape, save, quant)?;
        Ok(Self { w, order, loss })
    }

    /// append the next bytes of the tensor (C order of the stored shape = the caller's memory order)
    pub fn push(&mut self, data: &[u8]) -> Result<()> {
        self.w.push(data)
    }

    /// complete the file (all data pushed): index, order / loss entries, rename. Returns (path, file bytes).
    pub fn finish(mut self) -> Result<(PathBuf, u64)> {
        self.w.end_tensor()?;
        self.w.put("order", &Value::Str(self.order.tag().into()))?;
        let loss = std::mem::take(&mut self.loss);
        self.w.put("loss", &Value::Str(loss))?;
        self.w.finish()
    }
}

// ── one-call savers ──

/// Save an array given as raw little-endian bytes in `order` (no copy). `path`'s extension is replaced (`.hv`, or
/// `.npy` with `HASTY_IO=npy`); returns the path written.
pub fn save_bytes(path: &Path, dtype: DType, shape: &[usize], order: Order, bytes: &[u8], save: &Save) -> Result<PathBuf> {
    save_parts(path, dtype, shape, order, &[bytes], save)
}

/// [`save_bytes`] of an array given as consecutive byte parts (no assembly in memory; the policy's scale is computed
/// over all parts first).
pub fn save_parts(path: &Path, dtype: DType, shape: &[usize], order: Order, parts: &[&[u8]], save: &Save) -> Result<PathBuf> {
    let n: usize = shape.iter().product::<usize>() * dtype.size();
    let got: usize = parts.iter().map(|p| p.len()).sum();
    if n != got {
        return Err(Error::Format(format!("{}: {got} bytes for shape {shape:?} of {}", path.display(), dtype.name())));
    }
    let out = out_path(path);
    if !write_hv() {
        let mut w = std::io::BufWriter::with_capacity(8 << 20, File::create(&out)?);
        w.write_all(&crate::npy::header_bytes(dtype, shape, order.is_f())?)?;
        for p in parts {
            w.write_all(p)?;
        }
        w.flush()?;
        drop_stale(&out);
        return Ok(out);
    }
    let (quant, loss) = io_pool().install(|| resolve_policy(save.policy, dtype, Some(parts)))?;
    let mut w = TensorWriter::create_with(&out, dtype, shape, order, save, quant, loss)?;
    for p in parts {
        w.push(p)?;
    }
    let p = w.finish()?.0;
    drop_stale(&p);
    Ok(p)
}

/// [`save_bytes`] of a C-order tensor
pub fn save_tensor(path: &Path, t: &Tensor, save: &Save) -> Result<PathBuf> {
    save_bytes(path, t.dtype(), t.shape(), Order::C, t.bytes(), save)
}

/// [`save_bytes`] of a typed slice
pub fn save_slice<T: crate::Element>(path: &Path, shape: &[usize], order: Order, v: &[T], save: &Save) -> Result<PathBuf> {
    // SAFETY: Element types are plain data without padding
    let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    save_bytes(path, T::DTYPE, shape, order, b, save)
}

/// Encode a value tree to `w`: every float tensor with at least `min_lossy` elements gets `save.policy` resolved on
/// its own data (its own scale), smaller ones and non-float tensors stay lossless; tensors above
/// [`AUTO_CHUNK_BYTES`] are chunked (parallel encode / decode, random access). Compressed chunks of one tensor are
/// assembled in memory before they are written.
pub fn write_value(w: &mut (dyn Write + Send), v: &Value, save: &Save, min_lossy: usize) -> Result<()> {
    let o = save.options();
    let chunk = save.chunk.clone();
    let policy = save.policy;
    let plan = |_: &str, t: &Tensor| (t.nbytes() > AUTO_CHUNK_BYTES).then(|| chunk_shape(t.shape(), t.dtype().size(), &chunk));
    let loss = |_: &str, t: &Tensor| (t.numel() >= min_lossy).then(|| resolve_policy(policy, t.dtype(), Some(&[t.bytes()])).ok().and_then(|(q, _)| q)).flatten();
    io_pool().install(|| format::encode_lossy(v, &o, Some(&plan), &loss, &mut |c| Ok(w.write_all(&c)?)))
}

/// Save a value tree (see [`write_value`]; `min_lossy` 0). Always `.hv` (the extension is not changed); written as
/// `<path>.tmp` and renamed.
pub fn save_value(path: &Path, v: &Value, save: &Save) -> Result<()> {
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    {
        let mut w = std::io::BufWriter::with_capacity(8 << 20, File::create(&tmp)?);
        write_value(&mut w, v, save, 0)?;
        w.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ── loaders ──

fn not_found(path: &Path) -> Error {
    Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, format!("{} (nor its .npy / .hv sibling)", path.display())))
}

fn is_hv(path: &Path) -> Result<bool> {
    let mut m = [0u8; 4];
    let f = File::open(path)?;
    let n = f.read_at(&mut m, 0)?;
    Ok(n == 4 && m == MAGIC)
}

/// the array of a single-array `.hv` file: (tensor ref, order)
fn array_ref(lf: &LazyFile, path: &Path) -> Result<(TensorRef, Order)> {
    match &lf.root {
        Lazy::Tensor(t) => Ok((t.clone(), Order::C)),
        Lazy::Dict(_) => {
            let t = lf.root.get("array").and_then(|l| l.tensor()).ok_or_else(|| Error::Format(format!("{}: no \"array\" tensor (not a single-array file)", path.display())))?;
            let order = match lf.root.get("order") {
                Some(Lazy::Value(Value::Str(s))) if s == "F" => Order::F,
                _ => Order::C,
            };
            Ok((t.clone(), order))
        }
        _ => Err(Error::Format(format!("{}: not a single-array file", path.display()))),
    }
}

/// Load a single array from `.hv` or `.npy` (`path` or its sibling, see [`resolve`]): the C-order tensor (for
/// `Order::F` the shape is reversed: (…, Z, Y, X)) and the caller's order — as [`crate::npy::read`].
pub fn load_array(path: &Path) -> Result<(Tensor, Order)> {
    let p = resolve(path).ok_or_else(|| not_found(path))?;
    if !is_hv(&p)? {
        let (t, f) = crate::npy::read(&p)?;
        return Ok((t, if f { Order::F } else { Order::C }));
    }
    let lf = LazyFile::open(&p)?;
    let (t, order) = array_ref(&lf, &p)?;
    Ok((io_pool().install(|| lf.read(&t))?, order))
}

/// Load a value tree (`.hv`); a `.npy` file loads as its tensor.
pub fn load_value(path: &Path) -> Result<Value> {
    let p = resolve(path).ok_or_else(|| not_found(path))?;
    if !is_hv(&p)? {
        return Ok(Value::Tensor(crate::npy::read(&p)?.0));
    }
    let lf = LazyFile::open(&p)?;
    io_pool().install(|| lf.value_of(&lf.root))
}

/// Random access to the leading-axis rows of an array file (`.npy` by positional reads, chunked `.hv` by decoding the
/// overlapping chunks): the k-space data and per-sample records of a sim, read readout by readout.
pub struct RowFile {
    pub path: PathBuf,
    pub dtype: DType,
    /// C-order shape (as stored)
    pub shape: Vec<usize>,
    pub order: Order,
    inner: RowInner,
}

enum RowInner {
    Npy { file: File, offset: u64 },
    Hv { lf: LazyFile, t: TensorRef },
}

impl RowFile {
    pub fn open(path: &Path) -> Result<Self> {
        let p = resolve(path).ok_or_else(|| not_found(path))?;
        if is_hv(&p)? {
            let lf = LazyFile::open(&p)?;
            let (t, order) = array_ref(&lf, &p)?;
            return Ok(Self { path: p, dtype: t.dtype, shape: t.shape.clone(), order, inner: RowInner::Hv { lf, t } });
        }
        let (h, offset) = crate::npy::header(&p)?;
        let file = File::open(&p)?;
        let (dtype, mut shape, fortran) = h;
        if fortran {
            shape.reverse();
        }
        Ok(Self { path: p, dtype, shape, order: if fortran { Order::F } else { Order::C }, inner: RowInner::Npy { file, offset } })
    }

    pub fn is_hv(&self) -> bool {
        matches!(self.inner, RowInner::Hv { .. })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// raw bytes of the elements [start, start + out.len() / dtype size) in stored (C) order
    pub fn read_elems(&self, start: usize, out: &mut [u8]) -> Result<()> {
        let es = self.dtype.size();
        let n = out.len() / es;
        if out.len() % es != 0 || start + n > self.numel() {
            return Err(Error::Slice(format!("{}: elements {start}..{} of {}", self.path.display(), start + n, self.numel())));
        }
        match &self.inner {
            RowInner::Npy { file, offset } => Ok(file.read_exact_at(out, offset + (start * es) as u64)?),
            RowInner::Hv { lf, t } => {
                if n == 0 {
                    return Ok(());
                }
                // the smallest box [fixed prefix, range on the first axis where start and end differ, full rest]: the
                // requested elements are one contiguous run in it (only the chunks it overlaps are decoded)
                let nd = self.shape.len();
                let idx = |mut e: usize| -> Vec<usize> {
                    let mut v = vec![0usize; nd];
                    for d in (0..nd).rev() {
                        v[d] = e % self.shape[d];
                        e /= self.shape[d];
                    }
                    v
                };
                let (a, b) = (idx(start), idx(start + n - 1));
                let d0 = (0..nd).find(|&d| a[d] != b[d]).unwrap_or(nd - 1);
                let mut lo = vec![0usize; nd];
                let mut hi = self.shape.clone();
                for d in 0..d0 {
                    lo[d] = a[d];
                    hi[d] = a[d] + 1;
                }
                lo[d0] = a[d0];
                hi[d0] = b[d0] + 1;
                let off = a[d0 + 1..].iter().zip(&self.shape[d0 + 1..]).fold(0usize, |acc, (&i, &s)| acc * s + i) * es;
                let b = lf.read_box(t, &lo, &hi)?;
                out.copy_from_slice(&b.bytes()[off..off + out.len()]);
                Ok(())
            }
        }
    }
}
