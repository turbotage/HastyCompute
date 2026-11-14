//! Streaming, block-compressed value format (specification: `FORMAT.md`).
//!
//! The encoder emits byte chunks as it goes (tensor blocks are compressed in parallel groups, memory stays bounded);
//! the [`Decoder`] accepts arbitrary chunk boundaries and decodes every tensor block as soon as it is complete — the
//! compressed size of each block is in its own header, so nothing has to be known in advance.
//!
//! Chunked tensors (tag 9, [`encode_with`]): the tensor is cut into N-D chunks, each compressed independently and
//! located by an index, so files can be read box by box ([`crate::lazy::LazyFile`]); the stream decoder assembles
//! them into an ordinary tensor once their data is complete.

use std::io::{Read, Write};

use rayon::prelude::*;

use crate::codec::{decode_block, encode_block, BlockHeader, Options};
use crate::integrity::{Hasher, Integrity};
use crate::lossy::{self, Loss, LossPlan, Quant, Scope};
use crate::{Buffer, DType, Error, Result, Tensor, Value};

pub const MAGIC: [u8; 4] = *b"HSTV";
pub const VERSION: u8 = 1;
pub const STREAM_HEADER: usize = 8;

pub(crate) const TAG_NONE: u8 = 0;
pub(crate) const TAG_BOOL: u8 = 1;
pub(crate) const TAG_INT: u8 = 2;
pub(crate) const TAG_FLOAT: u8 = 3;
pub(crate) const TAG_STR: u8 = 4;
pub(crate) const TAG_LIST: u8 = 5;
pub(crate) const TAG_TUPLE: u8 = 6;
pub(crate) const TAG_DICT: u8 = 7;
pub(crate) const TAG_TENSOR: u8 = 8;
pub(crate) const TAG_CHUNKED: u8 = 9;
/// chunked tensor whose chunks are quantised (see `lossy`)
pub(crate) const TAG_QCHUNKED: u8 = 10;

/// chunk shape for a tensor at a slice path (`["x"]`, `[0]["y"]`, … ; "" for the root), or None for a plain tensor
pub type ChunkPlan<'a> = &'a dyn Fn(&str, &Tensor) -> Option<Vec<usize>>;
/// encoder output chunk size
const FLUSH: usize = 1 << 20;

struct Enc<'a> {
    o: &'a Options,
    buf: Vec<u8>,
    sink: &'a mut dyn FnMut(Vec<u8>) -> Result<()>,
    plan: Option<ChunkPlan<'a>>,
    loss: Option<LossPlan<'a>>,
    path: String,
    hasher: Option<Hasher>,
}

/// C-order strides (elements)
pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// copy the box of extent `ext` at `slo` in `src` (C-order, shape `sshape`) to `dlo` in `dst` (shape `dshape`)
pub(crate) fn copy_sub(src: &[u8], sshape: &[usize], slo: &[usize], dst: &mut [u8], dshape: &[usize], dlo: &[usize], ext: &[usize], es: usize) {
    // SAFETY: `dst` is a unique slice of the stated length
    unsafe { copy_sub_ptr(src, sshape, slo, dst.as_mut_ptr(), dst.len(), dshape, dlo, ext, es) }
}

/// Destination shared between threads that write disjoint boxes (parallel chunk assembly).
#[derive(Clone, Copy)]
pub(crate) struct Shared(*mut u8, usize);
impl Shared {
    pub fn new(b: &mut [u8]) -> Self {
        Self(b.as_mut_ptr(), b.len())
    }
    pub fn ptr(&self) -> *mut u8 {
        self.0
    }
    pub fn len(&self) -> usize {
        self.1
    }
}
// SAFETY: users only write disjoint boxes through it, while the owner keeps the buffer alive and untouched
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// [`copy_sub`] into a raw destination (`dst_len` bytes). Bounds are checked; the caller guarantees that no other
/// thread touches the bytes of the box being written.
///
/// # Safety
/// `dst` must be valid for writes of `dst_len` bytes.
pub(crate) unsafe fn copy_sub_ptr(src: &[u8], sshape: &[usize], slo: &[usize], dst: *mut u8, dst_len: usize, dshape: &[usize], dlo: &[usize], ext: &[usize], es: usize) {
    let nd = ext.len();
    let cp = |dbase: usize, sbase: usize, row: usize| {
        let (d, s) = (dbase * es, sbase * es);
        assert!(d + row <= dst_len && s + row <= src.len(), "copy_sub out of range");
        std::ptr::copy_nonoverlapping(src.as_ptr().add(s), dst.add(d), row);
    };
    if ext.iter().any(|&e| e == 0) {
        return;
    }
    let (ss, ds) = (strides(sshape), strides(dshape));
    let mut sbase: usize = (0..nd).map(|d| slo[d] * ss[d]).sum();
    let mut dbase: usize = (0..nd).map(|d| dlo[d] * ds[d]).sum();
    // merge trailing dimensions that are complete in source, destination and box into one contiguous run
    let mut k = nd;
    let mut run = 1usize;
    while k > 0 && ext[k - 1] == sshape[k - 1] && ext[k - 1] == dshape[k - 1] {
        run *= ext[k - 1];
        k -= 1;
    }
    if k > 0 {
        run *= ext[k - 1];
        k -= 1;
    }
    let row = run * es;
    // remaining outer dimensions (extent-1 ones only shift the base)
    let dims: Vec<usize> = (0..k).filter(|&d| ext[d] > 1).collect();
    if dims.is_empty() {
        cp(dbase, sbase, row);
        return;
    }
    let mut idx = vec![0usize; dims.len()];
    loop {
        cp(dbase, sbase, row);
        let mut j = dims.len();
        loop {
            if j == 0 {
                return;
            }
            j -= 1;
            let d = dims[j];
            idx[j] += 1;
            sbase += ss[d];
            dbase += ds[d];
            if idx[j] < ext[d] {
                break;
            }
            sbase -= ss[d] * ext[d];
            dbase -= ds[d] * ext[d];
            idx[j] = 0;
        }
    }
}

/// box (origin, extent) of chunk `ci` (C-order over `grid`)
pub(crate) fn chunk_box(ci: usize, grid: &[usize], chunk: &[usize], shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let nd = grid.len();
    let mut lo = vec![0usize; nd];
    let mut r = ci;
    for d in (0..nd).rev() {
        lo[d] = (r % grid[d]) * chunk[d];
        r /= grid[d];
    }
    let ext = (0..nd).map(|d| chunk[d].min(shape[d] - lo[d])).collect();
    (lo, ext)
}

/// decode the consecutive blocks of `bytes` into `raw_len` bytes (block size `bs`)
pub(crate) fn decode_blocks(bytes: &[u8], raw_len: usize, bs: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; raw_len];
    let (mut p, mut off) = (0usize, 0usize);
    while off < raw_len {
        let h = BlockHeader::parse(&bytes[p..]).ok_or_else(|| Error::Format("truncated block header".into()))?;
        let hl = h.len();
        let expect = bs.min(raw_len - off);
        if h.raw_len as usize != expect || bytes.len() < p + hl + h.comp_len as usize {
            return Err(Error::Format("chunk block mismatch".into()));
        }
        decode_block(&h, &bytes[p + hl..p + hl + h.comp_len as usize], &mut out[off..off + expect])?;
        p += hl + h.comp_len as usize;
        off += expect;
    }
    if p != bytes.len() {
        return Err(Error::Format("trailing bytes in chunk".into()));
    }
    Ok(out)
}

/// decode the data of one chunk with box extent `ext` (plain blocks, or prefix + blocks if quantised)
pub(crate) fn decode_chunk_data(dtype: DType, quant: bool, ext: &[usize], bytes: &[u8], bs: usize) -> Result<Vec<u8>> {
    let numel: usize = ext.iter().product();
    if quant {
        lossy::decode_chunk(dtype, numel, bytes, bs)
    } else {
        decode_blocks(bytes, numel * dtype.size(), bs)
    }
}

impl Enc<'_> {
    fn flush(&mut self, force: bool) -> Result<()> {
        if self.buf.len() >= FLUSH || (force && !self.buf.is_empty()) {
            let b = std::mem::replace(&mut self.buf, Vec::with_capacity(FLUSH + (FLUSH >> 2)));
            if let Some(h) = &mut self.hasher {
                h.update(&b);
            }
            (self.sink)(b)?;
        }
        Ok(())
    }

    fn len_u64(&mut self, n: usize) {
        self.buf.extend_from_slice(&(n as u64).to_le_bytes());
    }

    fn value(&mut self, v: &Value) -> Result<()> {
        let base = self.path.len();
        match v {
            Value::None => self.buf.push(TAG_NONE),
            Value::Bool(b) => self.buf.extend_from_slice(&[TAG_BOOL, *b as u8]),
            Value::Int(i) => {
                self.buf.push(TAG_INT);
                self.buf.extend_from_slice(&i.to_le_bytes());
            }
            Value::Float(f) => {
                self.buf.push(TAG_FLOAT);
                self.buf.extend_from_slice(&f.to_le_bytes());
            }
            Value::Str(s) => {
                self.buf.push(TAG_STR);
                self.len_u64(s.len());
                self.buf.extend_from_slice(s.as_bytes());
            }
            Value::List(items) | Value::Tuple(items) => {
                self.buf.push(if matches!(v, Value::List(_)) { TAG_LIST } else { TAG_TUPLE });
                self.len_u64(items.len());
                for (i, it) in items.iter().enumerate() {
                    self.path.push_str(&format!("[{i}]"));
                    self.value(it)?;
                    self.path.truncate(base);
                    self.flush(false)?;
                }
            }
            Value::Dict(items) => {
                self.buf.push(TAG_DICT);
                self.len_u64(items.len());
                for (k, it) in items {
                    self.buf.extend_from_slice(&(k.len() as u32).to_le_bytes());
                    self.buf.extend_from_slice(k.as_bytes());
                    self.path.push_str(&format!("[\"{k}\"]"));
                    self.value(it)?;
                    self.path.truncate(base);
                    self.flush(false)?;
                }
            }
            Value::Tensor(t) => {
                let q = self.loss.and_then(|l| l(&self.path, t)).filter(|q| q.loss != Loss::Lossless);
                if let Some(q) = &q {
                    q.validate(t.dtype())?;
                }
                let chunk = self.plan.and_then(|p| p(&self.path, t));
                match q {
                    Some(q) if q.is_int() => self.chunked(t, &chunk.unwrap_or_else(|| t.shape().to_vec()), Some(&q))?,
                    Some(Quant { loss: Loss::Mantissa(m), .. }) => {
                        let r = Tensor::new(t.dtype(), t.shape().to_vec(), Buffer::from_bytes(&lossy::round_mantissa(t.dtype(), m, t.bytes())))?;
                        match chunk {
                            Some(c) => self.chunked(&r, &c, None)?,
                            None => self.tensor(&r)?,
                        }
                    }
                    _ => match chunk {
                        Some(c) => self.chunked(t, &c, None)?,
                        None => self.tensor(t)?,
                    },
                }
            }
        }
        Ok(())
    }

    fn tensor(&mut self, t: &Tensor) -> Result<()> {
        if t.shape().len() > 255 {
            return Err(Error::Format("more than 255 dimensions".into()));
        }
        let bs = self.o.block_bytes();
        self.buf.extend_from_slice(&[TAG_TENSOR, t.dtype() as u8, t.shape().len() as u8, 0]);
        self.buf.extend_from_slice(&(bs as u32).to_le_bytes());
        for &d in t.shape() {
            self.len_u64(d);
        }
        let data = t.bytes();
        let nblocks = data.len().div_ceil(bs);
        let (word, delta_ok) = (t.dtype().word(), t.dtype().is_integer());
        let group = if self.o.parallel { (rayon::current_num_threads() * 2).clamp(1, 64) } else { 1 };
        let o = *self.o;
        let enc = |b: usize| {
            let raw = &data[b * bs..((b + 1) * bs).min(data.len())];
            let mut out = Vec::with_capacity(raw.len() / 2 + 32);
            encode_block(raw, word, delta_ok, &o, &mut out);
            out
        };
        let mut g = 0;
        while g < nblocks {
            let e = (g + group).min(nblocks);
            let encoded: Vec<Vec<u8>> = if group > 1 && e - g > 1 { (g..e).into_par_iter().map(enc).collect() } else { (g..e).map(enc).collect() };
            for blk in encoded {
                self.buf.extend_from_slice(&blk);
                self.flush(false)?;
            }
            g = e;
        }
        Ok(())
    }
}

impl Enc<'_> {
    fn chunked(&mut self, t: &Tensor, chunk: &[usize], quant: Option<&Quant>) -> Result<()> {
        let shape = t.shape();
        let nd = shape.len();
        if chunk.len() != nd || nd > 255 {
            return Err(Error::Format(format!("chunk shape {chunk:?} for tensor shape {shape:?}")));
        }
        let chunk: Vec<usize> = chunk.iter().zip(shape).map(|(&c, &n)| c.clamp(1, n.max(1))).collect();
        let grid: Vec<usize> = shape.iter().zip(&chunk).map(|(&n, &c)| n.div_ceil(c)).collect();
        let nchunks: usize = if shape.iter().any(|&n| n == 0) { 0 } else { grid.iter().product() };
        let (es, bs, word, delta_ok, o) = (t.dtype().size(), self.o.block_bytes(), t.dtype().word(), t.dtype().is_integer(), *self.o);
        let global = quant.filter(|q| q.scope == Scope::Tensor).map(|q| lossy::tensor_stats(q, t));
        let enc = |ci: usize| -> Vec<u8> {
            let (lo, ext) = chunk_box(ci, &grid, &chunk, shape);
            let mut raw = vec![0u8; ext.iter().product::<usize>() * es];
            copy_sub(t.bytes(), shape, &lo, &mut raw, &ext, &vec![0; nd], &ext, es);
            let mut out = Vec::with_capacity(raw.len() / 2 + 32);
            match quant {
                Some(q) => {
                    let (payload, qw) = lossy::encode_chunk(q, t.dtype(), &raw, global.as_ref());
                    out.extend_from_slice(&payload[..lossy::PREFIX]);
                    for b in payload[lossy::PREFIX..].chunks(bs) {
                        encode_block(b, qw, false, &o, &mut out);
                    }
                }
                None => {
                    for b in raw.chunks(bs) {
                        encode_block(b, word, delta_ok, &o, &mut out);
                    }
                }
            }
            out
        };
        let encoded: Vec<Vec<u8>> = if self.o.parallel { (0..nchunks).into_par_iter().map(enc).collect() } else { (0..nchunks).map(enc).collect() };
        self.buf.extend_from_slice(&[if quant.is_some() { TAG_QCHUNKED } else { TAG_CHUNKED }, t.dtype() as u8, nd as u8, 0]);
        self.buf.extend_from_slice(&(bs as u32).to_le_bytes());
        for &d in shape.iter().chain(&chunk) {
            self.len_u64(d);
        }
        self.len_u64(nchunks);
        let mut off = 0u64;
        for c in &encoded {
            self.buf.extend_from_slice(&off.to_le_bytes());
            self.buf.extend_from_slice(&(c.len() as u64).to_le_bytes());
            off += c.len() as u64;
        }
        for c in encoded {
            self.buf.extend_from_slice(&c);
            self.flush(false)?;
        }
        Ok(())
    }
}

/// Encode `value`, handing out chunks of roughly 1 MiB to `sink`.
pub fn encode(value: &Value, o: &Options, sink: &mut dyn FnMut(Vec<u8>) -> Result<()>) -> Result<()> {
    encode_plan(value, o, None, sink)
}

/// [`encode`], storing the tensors that `plan` selects as chunked tensors (random-access files).
pub fn encode_with(value: &Value, o: &Options, plan: ChunkPlan, sink: &mut dyn FnMut(Vec<u8>) -> Result<()>) -> Result<()> {
    encode_plan(value, o, Some(plan), sink)
}

/// [`encode_with`] with controlled-loss storage: tensors for which `loss` returns a spec are quantised (see
/// [`crate::lossy`]); integer quantisation makes them chunked tensors (tag 10; one chunk if `plan` gives no shape).
pub fn encode_lossy(value: &Value, o: &Options, plan: Option<ChunkPlan>, loss: LossPlan, sink: &mut dyn FnMut(Vec<u8>) -> Result<()>) -> Result<()> {
    encode_all(value, o, plan, Some(loss), sink)
}

fn encode_plan(value: &Value, o: &Options, plan: Option<ChunkPlan>, sink: &mut dyn FnMut(Vec<u8>) -> Result<()>) -> Result<()> {
    encode_all(value, o, plan, None, sink)
}

fn encode_all(value: &Value, o: &Options, plan: Option<ChunkPlan>, loss: Option<LossPlan>, sink: &mut dyn FnMut(Vec<u8>) -> Result<()>) -> Result<()> {
    let mut e = Enc { o, buf: Vec::with_capacity(FLUSH + (FLUSH >> 2)), sink, plan, loss, path: String::new(), hasher: Hasher::new(o.integrity) };
    e.buf.extend_from_slice(&MAGIC);
    e.buf.extend_from_slice(&[VERSION, o.integrity.code(), 0, 0]);
    e.value(value)?;
    e.flush(true)?;
    if let Some(h) = e.hasher.take() {
        (e.sink)(h.finalize())?;
    }
    Ok(())
}

pub fn encode_to_vec(value: &Value, o: &Options) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    encode(value, o, &mut |c| {
        out.extend_from_slice(&c);
        Ok(())
    })?;
    Ok(out)
}

pub fn write(value: &Value, o: &Options, w: &mut impl Write) -> Result<()> {
    encode(value, o, &mut |c| Ok(w.write_all(&c)?))
}

enum Frame {
    Seq { items: Vec<Value>, left: u64, tuple: bool },
    Dict { items: Vec<(String, Value)>, left: u64, key: Option<String> },
}

struct ChunkedIn {
    dtype: DType,
    shape: Vec<usize>,
    chunk: Vec<usize>,
    grid: Vec<usize>,
    index: Vec<(u64, u64)>,
    block_size: usize,
    quant: bool,
}

struct TensorIn {
    dtype: DType,
    shape: Vec<usize>,
    data: Buffer,
    off: usize,
    blocks_left: u64,
    block_size: usize,
}

/// Incremental decoder: `push` chunks of any size, then `finish`.
pub struct Decoder {
    buf: Vec<u8>,
    pos: usize,
    header: bool,
    stack: Vec<Frame>,
    tensor: Option<TensorIn>,
    chunked: Option<ChunkedIn>,
    done: Option<Value>,
    parallel: bool,
    /// integrity digest still to be verified (stream header flags), the hashed prefix of `buf`
    hasher: Option<Hasher>,
    algo: Integrity,
    hpos: usize,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

impl Decoder {
    pub fn new() -> Self {
        Self { buf: Vec::new(), pos: 0, header: false, stack: Vec::new(), tensor: None, chunked: None, done: None, parallel: true, hasher: None, algo: Integrity::None, hpos: 0 }
    }

    /// decode blocks on the calling thread only
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }

    /// the value is complete (and its integrity digest, if the stream carries one, verified)
    pub fn is_done(&self) -> bool {
        self.done.is_some() && self.hasher.is_none()
    }

    /// bytes buffered but not yet consumed
    pub fn pending(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn push(&mut self, data: &[u8]) -> Result<()> {
        if self.done.is_some() && self.hasher.is_none() {
            return if data.is_empty() { Ok(()) } else { Err(Error::Format("data after the end of the value".into())) };
        }
        if self.buf.len() == self.pos {
            // nothing pending: decode straight from the caller's bytes and buffer only what is left over
            self.buf.clear();
            self.pos = 0;
            self.hpos = 0;
            self.advance(data)?;
            self.verify(data)?;
            self.buf.extend_from_slice(&data[self.pos..]);
            self.pos = 0;
            self.hpos = 0;
            return Ok(());
        }
        self.buf.extend_from_slice(data);
        let buf = std::mem::take(&mut self.buf);
        let r = self.advance(&buf).and_then(|_| self.verify(&buf));
        self.buf = buf;
        r?;
        if self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
            self.hpos = 0;
        } else if self.pos > (1 << 23) || self.pos > self.buf.len() / 2 {
            self.buf.drain(..self.pos);
            self.pos = 0;
            self.hpos = 0;
        }
        Ok(())
    }

    /// feed the consumed bytes to the hasher; once the value is complete, check the trailing digest
    fn verify(&mut self, src: &[u8]) -> Result<()> {
        let Some(h) = &mut self.hasher else { return Ok(()) };
        h.update(&src[self.hpos..self.pos]);
        self.hpos = self.pos;
        let n = self.algo.digest_len();
        if self.done.is_some() && src.len() - self.pos >= n {
            let d = self.hasher.take().unwrap().finalize();
            if d[..] != src[self.pos..self.pos + n] {
                return Err(Error::Codec("integrity digest mismatch".into()));
            }
            self.pos += n;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Value> {
        if self.done.is_some() && self.hasher.is_some() {
            return Err(Error::Format("truncated stream: integrity digest missing".into()));
        }
        if self.pending() != 0 && self.done.is_some() {
            return Err(Error::Format(format!("{} trailing bytes", self.pending())));
        }
        self.done.ok_or_else(|| Error::Format("truncated stream".into()))
    }

    fn complete(&mut self, mut v: Value) {
        loop {
            match self.stack.last_mut() {
                None => {
                    self.done = Some(v);
                    return;
                }
                Some(Frame::Seq { items, left, .. }) => {
                    items.push(v);
                    *left -= 1;
                    if *left > 0 {
                        return;
                    }
                }
                Some(Frame::Dict { items, left, key }) => {
                    items.push((key.take().expect("dict key"), v));
                    *left -= 1;
                    if *left > 0 {
                        return;
                    }
                }
            }
            v = match self.stack.pop().unwrap() {
                Frame::Seq { items, tuple: true, .. } => Value::Tuple(items),
                Frame::Seq { items, .. } => Value::List(items),
                Frame::Dict { items, .. } => Value::Dict(items),
            };
        }
    }

    fn advance(&mut self, src: &[u8]) -> Result<()> {
        if !self.header {
            let a = &src[self.pos..];
            if a.len() < STREAM_HEADER {
                return Ok(());
            }
            if a[..4] != MAGIC {
                return Err(Error::Format("not a hasty value stream (bad magic)".into()));
            }
            if a[4] != VERSION {
                return Err(Error::Format(format!("unsupported format version {}", a[4])));
            }
            self.algo = Integrity::from_code(a[5])?;
            self.hasher = Hasher::new(self.algo);
            self.pos += STREAM_HEADER;
            self.header = true;
        }
        while self.done.is_none() {
            if self.chunked.is_some() {
                if !self.chunked_data(src)? {
                    break;
                }
                continue;
            }
            if self.tensor.is_some() {
                if !self.tensor_blocks(src)? {
                    break;
                }
                continue;
            }
            if let Some(Frame::Dict { key: k @ None, .. }) = self.stack.last_mut() {
                let a = &src[self.pos..];
                if a.len() < 4 {
                    break;
                }
                let n = u32_at(a, 0) as usize;
                if a.len() < 4 + n {
                    break;
                }
                *k = Some(String::from_utf8(a[4..4 + n].to_vec()).map_err(|_| Error::Format("dict key is not utf-8".into()))?);
                self.pos += 4 + n;
                continue;
            }
            let a = &src[self.pos..];
            if a.is_empty() {
                break;
            }
            let tag = a[0];
            match tag {
                TAG_NONE => {
                    self.pos += 1;
                    self.complete(Value::None);
                }
                TAG_BOOL => {
                    if a.len() < 2 {
                        break;
                    }
                    let b = a[1] != 0;
                    self.pos += 2;
                    self.complete(Value::Bool(b));
                }
                TAG_INT | TAG_FLOAT => {
                    if a.len() < 9 {
                        break;
                    }
                    let x = u64_at(a, 1);
                    self.pos += 9;
                    self.complete(if tag == TAG_INT { Value::Int(x as i64) } else { Value::Float(f64::from_bits(x)) });
                }
                TAG_STR => {
                    if a.len() < 9 {
                        break;
                    }
                    let n = u64_at(a, 1) as usize;
                    if a.len() < 9 + n {
                        break;
                    }
                    let s = String::from_utf8(a[9..9 + n].to_vec()).map_err(|_| Error::Format("string is not utf-8".into()))?;
                    self.pos += 9 + n;
                    self.complete(Value::Str(s));
                }
                TAG_LIST | TAG_TUPLE | TAG_DICT => {
                    if a.len() < 9 {
                        break;
                    }
                    let n = u64_at(a, 1);
                    self.pos += 9;
                    if n == 0 {
                        self.complete(match tag {
                            TAG_LIST => Value::List(vec![]),
                            TAG_TUPLE => Value::Tuple(vec![]),
                            _ => Value::Dict(vec![]),
                        });
                    } else if tag == TAG_DICT {
                        self.stack.push(Frame::Dict { items: Vec::new(), left: n, key: None });
                    } else {
                        self.stack.push(Frame::Seq { items: Vec::new(), left: n, tuple: tag == TAG_TUPLE });
                    }
                }
                TAG_TENSOR => {
                    if a.len() < 8 {
                        break;
                    }
                    let dtype = DType::from_code(a[1])?;
                    let ndim = a[2] as usize;
                    let block_size = u32_at(a, 4) as usize;
                    if a.len() < 8 + 8 * ndim {
                        break;
                    }
                    let shape: Vec<usize> = (0..ndim).map(|d| u64_at(a, 8 + 8 * d) as usize).collect();
                    let n = shape.iter().try_fold(1usize, |p, &d| p.checked_mul(d)).and_then(|n| n.checked_mul(dtype.size())).ok_or_else(|| Error::Format("tensor size overflow".into()))?;
                    if block_size == 0 && n > 0 {
                        return Err(Error::Format("zero block size".into()));
                    }
                    self.pos += 8 + 8 * ndim;
                    let blocks = if n == 0 { 0 } else { n.div_ceil(block_size) as u64 };
                    let t = TensorIn { dtype, shape, data: Buffer::zeroed(n), off: 0, blocks_left: blocks, block_size };
                    if blocks == 0 {
                        self.complete(Value::Tensor(Tensor::new(t.dtype, t.shape, t.data)?));
                    } else {
                        self.tensor = Some(t);
                    }
                }
                TAG_CHUNKED | TAG_QCHUNKED => {
                    if a.len() < 8 {
                        break;
                    }
                    let dtype = DType::from_code(a[1])?;
                    let nd = a[2] as usize;
                    let block_size = u32_at(a, 4) as usize;
                    let head = 8 + 16 * nd + 8;
                    if a.len() < head {
                        break;
                    }
                    let shape: Vec<usize> = (0..nd).map(|d| u64_at(a, 8 + 8 * d) as usize).collect();
                    let chunk: Vec<usize> = (0..nd).map(|d| u64_at(a, 8 + 8 * (nd + d)) as usize).collect();
                    let n = u64_at(a, 8 + 16 * nd) as usize;
                    if a.len() < head + 16 * n {
                        break;
                    }
                    let index: Vec<(u64, u64)> = (0..n).map(|i| (u64_at(a, head + 16 * i), u64_at(a, head + 16 * i + 8))).collect();
                    if chunk.iter().any(|&c| c == 0) || (block_size == 0 && n > 0) {
                        return Err(Error::Format("bad chunked tensor header".into()));
                    }
                    let grid: Vec<usize> = shape.iter().zip(&chunk).map(|(&s, &c)| s.div_ceil(c)).collect();
                    let expect: usize = if shape.iter().any(|&s| s == 0) { 0 } else { grid.iter().product() };
                    if n != expect {
                        return Err(Error::Format(format!("{n} chunks for grid {grid:?}")));
                    }
                    self.pos += head + 16 * n;
                    self.chunked = Some(ChunkedIn { dtype, shape, chunk, grid, index, block_size, quant: tag == TAG_QCHUNKED });
                }
                t => return Err(Error::Format(format!("unknown value tag {t}"))),
            }
        }
        Ok(())
    }

    /// chunked tensor: once all chunk data is buffered, decode the chunks (in parallel) and assemble the tensor
    fn chunked_data(&mut self, src: &[u8]) -> Result<bool> {
        let c = self.chunked.as_ref().unwrap();
        let total: u64 = c.index.iter().map(|x| x.1).sum();
        let a = &src[self.pos..];
        if (a.len() as u64) < total {
            return Ok(false);
        }
        let es = c.dtype.size();
        let dec = |ci: usize| -> Result<(usize, Vec<u8>)> {
            let (off, len) = c.index[ci];
            let (_, ext) = chunk_box(ci, &c.grid, &c.chunk, &c.shape);
            if off + len > total {
                return Err(Error::Format("chunk index out of range".into()));
            }
            Ok((ci, decode_chunk_data(c.dtype, c.quant, &ext, &a[off as usize..(off + len) as usize], c.block_size)?))
        };
        let mut t = Tensor::zeros(c.dtype, c.shape.clone());
        let nd = c.shape.len();
        let out = Shared::new(t.bytes_mut());
        let job = |ci: usize| -> Result<()> {
            let (ci, raw) = dec(ci)?;
            let (lo, ext) = chunk_box(ci, &c.grid, &c.chunk, &c.shape);
            // SAFETY: chunks cover disjoint boxes of `t`, which outlives the job
            unsafe { copy_sub_ptr(&raw, &ext, &vec![0; nd], out.ptr(), out.len(), &c.shape, &lo, &ext, es) };
            Ok(())
        };
        if self.parallel {
            (0..c.index.len()).into_par_iter().try_for_each(job)?;
        } else {
            (0..c.index.len()).try_for_each(job)?;
        }
        self.pos += total as usize;
        self.chunked = None;
        self.complete(Value::Tensor(t));
        Ok(true)
    }

    /// decode every complete block in the buffer; Ok(true) when the tensor is complete
    fn tensor_blocks(&mut self, src: &[u8]) -> Result<bool> {
        let t = self.tensor.as_mut().unwrap();
        let a = &src[self.pos..];
        let total = t.data.len();
        let mut jobs: Vec<(BlockHeader, usize)> = Vec::new(); // header, payload start in `a`
        let (mut p, mut off, mut left) = (0usize, t.off, t.blocks_left);
        while left > 0 {
            let Some(h) = BlockHeader::parse(&a[p..]) else { break };
            let hl = h.len();
            if a.len() < p + hl + h.comp_len as usize {
                break;
            }
            let expect = t.block_size.min(total - off);
            if h.raw_len as usize != expect {
                return Err(Error::Format(format!("block raw length {} (expected {expect})", h.raw_len)));
            }
            jobs.push((h, p + hl));
            p += hl + h.comp_len as usize;
            off += expect;
            left -= 1;
        }
        if jobs.is_empty() {
            return Ok(false);
        }
        let out = &mut t.data.as_bytes_mut()[t.off..off];
        let mut slices: Vec<&mut [u8]> = Vec::with_capacity(jobs.len());
        let mut rest = out;
        for (h, _) in &jobs {
            let (s, r) = rest.split_at_mut(h.raw_len as usize);
            slices.push(s);
            rest = r;
        }
        let run = |((h, ps), s): (&(BlockHeader, usize), &mut &mut [u8])| decode_block(h, &a[*ps..*ps + h.comp_len as usize], s);
        if self.parallel && jobs.len() > 1 {
            jobs.par_iter().zip(slices.par_iter_mut()).try_for_each(run)?;
        } else {
            jobs.iter().zip(slices.iter_mut()).try_for_each(run)?;
        }
        t.off = off;
        t.blocks_left = left;
        self.pos += p;
        if left == 0 {
            let t = self.tensor.take().unwrap();
            self.complete(Value::Tensor(Tensor::new(t.dtype, t.shape, t.data)?));
            return Ok(true);
        }
        Ok(false)
    }
}

pub fn decode(bytes: &[u8]) -> Result<Value> {
    let mut d = Decoder::new();
    d.push(bytes)?;
    d.finish()
}

pub fn read(r: &mut impl Read) -> Result<Value> {
    let mut d = Decoder::new();
    let mut chunk = vec![0u8; 8 << 20];
    loop {
        let n = r.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        d.push(&chunk[..n])?;
    }
    d.finish()
}
