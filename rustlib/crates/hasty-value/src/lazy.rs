//! Random-access reading of `.hv` files: [`LazyFile::open`] parses the tree without reading tensor data; tensors are
//! read whole ([`LazyFile::read`]) or by box ([`LazyFile::read_box`]). For chunked tensors (tag 9) a box read decodes
//! only the overlapping chunks, in parallel (positional reads, the file is not held locked).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use rayon::prelude::*;

use crate::codec::BlockHeader;
use crate::integrity::{Hasher, Integrity};
use crate::format::{chunk_box, copy_sub, copy_sub_ptr, Shared, decode_blocks, decode_chunk_data, MAGIC, STREAM_HEADER, TAG_BOOL, TAG_CHUNKED, TAG_QCHUNKED, TAG_DICT, TAG_FLOAT, TAG_INT, TAG_LIST, TAG_NONE, TAG_STR, TAG_TENSOR, TAG_TUPLE, VERSION};
use crate::{DType, Error, Result, Tensor, Value};

#[derive(Clone, Debug)]
enum Layout {
    /// blocks from `offset` to `end`
    Plain { offset: u64, end: u64 },
    Chunked { chunk: Vec<usize>, grid: Vec<usize>, index: Vec<(u64, u64)>, data: u64, quant: bool },
}

/// A tensor in a file (not read yet).
#[derive(Clone, Debug)]
pub struct TensorRef {
    pub dtype: DType,
    pub shape: Vec<usize>,
    block_size: usize,
    layout: Layout,
}

impl TensorRef {
    pub fn chunk_shape(&self) -> Option<&[usize]> {
        match &self.layout {
            Layout::Chunked { chunk, .. } => Some(chunk),
            Layout::Plain { .. } => None,
        }
    }
}

/// The value tree of a file with tensors as [`TensorRef`]s.
#[derive(Clone, Debug)]
pub enum Lazy {
    Value(Value),
    Tensor(TensorRef),
    List(Vec<Lazy>),
    Tuple(Vec<Lazy>),
    Dict(Vec<(String, Lazy)>),
}

impl Lazy {
    pub fn get(&self, key: &str) -> Option<&Lazy> {
        match self {
            Lazy::Dict(d) => d.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn tensor(&self) -> Option<&TensorRef> {
        match self {
            Lazy::Tensor(t) => Some(t),
            _ => None,
        }
    }
}

pub struct LazyFile {
    file: File,
    integrity: Integrity,
    pub root: Lazy,
}

struct Cur<'a> {
    f: &'a File,
    pos: u64,
}

impl Cur<'_> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        self.f.read_exact_at(&mut b, self.pos)?;
        self.pos += n as u64;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self, n: usize) -> Result<String> {
        String::from_utf8(self.bytes(n)?).map_err(|_| Error::Format("string is not utf-8".into()))
    }

    fn value(&mut self) -> Result<Lazy> {
        let tag = self.u8()?;
        Ok(match tag {
            TAG_NONE => Lazy::Value(Value::None),
            TAG_BOOL => Lazy::Value(Value::Bool(self.u8()? != 0)),
            TAG_INT => Lazy::Value(Value::Int(self.u64()? as i64)),
            TAG_FLOAT => Lazy::Value(Value::Float(f64::from_bits(self.u64()?))),
            TAG_STR => {
                let n = self.u64()? as usize;
                Lazy::Value(Value::Str(self.string(n)?))
            }
            TAG_LIST | TAG_TUPLE => {
                let n = self.u64()?;
                let items = (0..n).map(|_| self.value()).collect::<Result<Vec<_>>>()?;
                if tag == TAG_LIST {
                    Lazy::List(items)
                } else {
                    Lazy::Tuple(items)
                }
            }
            TAG_DICT => {
                let n = self.u64()?;
                let mut items = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let k = self.u32()? as usize;
                    let key = self.string(k)?;
                    items.push((key, self.value()?));
                }
                Lazy::Dict(items)
            }
            TAG_TENSOR | TAG_CHUNKED | TAG_QCHUNKED => {
                let h = self.bytes(7)?;
                let dtype = DType::from_code(h[0])?;
                let nd = h[1] as usize;
                let block_size = u32::from_le_bytes(h[3..7].try_into().unwrap()) as usize;
                let shape = (0..nd).map(|_| self.u64().map(|x| x as usize)).collect::<Result<Vec<_>>>()?;
                if tag == TAG_TENSOR {
                    let nbytes: usize = shape.iter().product::<usize>() * dtype.size();
                    let offset = self.pos;
                    let mut left = if nbytes == 0 { 0 } else { nbytes.div_ceil(block_size) };
                    while left > 0 {
                        let mut hb = self.bytes(crate::codec::BLOCK_HEADER)?;
                        if hb[3] & crate::codec::FLAG_CRC != 0 {
                            hb.extend(self.bytes(4)?);
                        }
                        let bh = BlockHeader::parse(&hb).ok_or_else(|| Error::Format("block header".into()))?;
                        self.pos += bh.comp_len as u64;
                        left -= 1;
                    }
                    Lazy::Tensor(TensorRef { dtype, shape, block_size, layout: Layout::Plain { offset, end: self.pos } })
                } else {
                    let chunk = (0..nd).map(|_| self.u64().map(|x| x as usize)).collect::<Result<Vec<_>>>()?;
                    let n = self.u64()? as usize;
                    let ix = self.bytes(16 * n)?;
                    let index: Vec<(u64, u64)> = (0..n).map(|i| (u64::from_le_bytes(ix[16 * i..16 * i + 8].try_into().unwrap()), u64::from_le_bytes(ix[16 * i + 8..16 * i + 16].try_into().unwrap()))).collect();
                    let grid: Vec<usize> = shape.iter().zip(&chunk).map(|(&s, &c)| s.div_ceil(c.max(1))).collect();
                    let data = self.pos;
                    self.pos += index.iter().map(|x| x.1).sum::<u64>();
                    Lazy::Tensor(TensorRef { dtype, shape, block_size, layout: Layout::Chunked { chunk, grid, index, data, quant: tag == TAG_QCHUNKED } })
                }
            }
            t => return Err(Error::Format(format!("unknown value tag {t}"))),
        })
    }
}

impl LazyFile {
    /// Check the whole file against its integrity digest (one sequential read of the file; `open` and the tensor
    /// reads never do this). Ok(false): the file carries no digest. Err: mismatch or truncation.
    pub fn verify(&self) -> Result<bool> {
        let Some(mut h) = Hasher::new(self.integrity) else { return Ok(false) };
        let (len, n) = (self.file.metadata()?.len(), self.integrity.digest_len() as u64);
        if len < STREAM_HEADER as u64 + n {
            return Err(Error::Format("truncated file: integrity digest missing".into()));
        }
        let end = len - n;
        let mut buf = vec![0u8; 4 << 20];
        let mut p = 0u64;
        while p < end {
            let m = ((end - p) as usize).min(buf.len());
            self.file.read_exact_at(&mut buf[..m], p)?;
            h.update(&buf[..m]);
            p += m as u64;
        }
        let mut want = vec![0u8; n as usize];
        self.file.read_exact_at(&mut want, end)?;
        if h.finalize() != want {
            return Err(Error::Codec("integrity digest mismatch".into()));
        }
        Ok(true)
    }

    pub fn integrity(&self) -> Integrity {
        self.integrity
    }

    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut c = Cur { f: &file, pos: 0 };
        let h = c.bytes(STREAM_HEADER)?;
        if h[..4] != MAGIC || h[4] != VERSION {
            return Err(Error::Format(format!("{}: not a hasty value file (v{VERSION})", path.display())));
        }
        let integrity = Integrity::from_code(h[5])?;
        let root = c.value()?;
        Ok(Self { file, integrity, root })
    }

    /// the whole tensor
    pub fn read(&self, t: &TensorRef) -> Result<Tensor> {
        match &t.layout {
            Layout::Plain { offset, end } => {
                let mut b = vec![0u8; (end - offset) as usize];
                self.file.read_exact_at(&mut b, *offset)?;
                let n: usize = t.shape.iter().product::<usize>() * t.dtype.size();
                Tensor::from_bytes(t.dtype, t.shape.clone(), &decode_blocks(&b, n, t.block_size)?)
            }
            Layout::Chunked { .. } => {
                let hi = t.shape.clone();
                self.read_box(t, &vec![0; hi.len()], &hi)
            }
        }
    }

    /// the box [lo, hi) of a tensor (per dimension), as a C-order tensor of shape hi − lo
    pub fn read_box(&self, t: &TensorRef, lo: &[usize], hi: &[usize]) -> Result<Tensor> {
        let nd = t.shape.len();
        if lo.len() != nd || hi.len() != nd || (0..nd).any(|d| lo[d] > hi[d] || hi[d] > t.shape[d]) {
            return Err(Error::Slice(format!("box {lo:?}..{hi:?} outside {:?}", t.shape)));
        }
        let ext: Vec<usize> = (0..nd).map(|d| hi[d] - lo[d]).collect();
        let es = t.dtype.size();
        let mut out = Tensor::zeros(t.dtype, ext.clone());
        if ext.iter().any(|&e| e == 0) {
            return Ok(out);
        }
        match &t.layout {
            Layout::Plain { .. } => {
                let full = self.read(t)?;
                copy_sub(full.bytes(), &t.shape, lo, out.bytes_mut(), &ext, &vec![0; nd], &ext, es);
            }
            Layout::Chunked { chunk, grid, index, data, quant } => {
                // chunks overlapping the box, C-order
                let ranges: Vec<(usize, usize)> = (0..nd).map(|d| (lo[d] / chunk[d], (hi[d] - 1) / chunk[d] + 1)).collect();
                let mut ids = vec![];
                let mut g: Vec<usize> = ranges.iter().map(|r| r.0).collect();
                'outer: loop {
                    ids.push(g.iter().zip(grid).fold(0usize, |a, (&x, &n)| a * n + x));
                    let mut d = nd;
                    loop {
                        if d == 0 {
                            break 'outer;
                        }
                        d -= 1;
                        g[d] += 1;
                        if g[d] < ranges[d].1 {
                            break;
                        }
                        g[d] = ranges[d].0;
                    }
                }
                let dst = Shared::new(out.bytes_mut());
                ids.par_iter().try_for_each(|&ci| -> Result<()> {
                    let (off, len) = index[ci];
                    let mut b = vec![0u8; len as usize];
                    self.file.read_exact_at(&mut b, data + off)?;
                    let (clo, cext) = chunk_box(ci, grid, chunk, &t.shape);
                    let raw = decode_chunk_data(t.dtype, *quant, &cext, &b, t.block_size)?;
                    let a: Vec<usize> = (0..nd).map(|d| clo[d].max(lo[d])).collect();
                    let e: Vec<usize> = (0..nd).map(|d| (clo[d] + cext[d]).min(hi[d])).collect();
                    let iext: Vec<usize> = (0..nd).map(|d| e[d] - a[d]).collect();
                    let src_lo: Vec<usize> = (0..nd).map(|d| a[d] - clo[d]).collect();
                    let dst_lo: Vec<usize> = (0..nd).map(|d| a[d] - lo[d]).collect();
                    // SAFETY: chunks write disjoint boxes of `out`, which outlives the loop
                    unsafe { copy_sub_ptr(&raw, &cext, &src_lo, dst.ptr(), dst.len(), &ext, &dst_lo, &iext, es) };
                    Ok(())
                })?;
            }
        }
        Ok(out)
    }

    /// materialise a subtree
    pub fn value_of(&self, l: &Lazy) -> Result<Value> {
        Ok(match l {
            Lazy::Value(v) => v.clone(),
            Lazy::Tensor(t) => Value::Tensor(self.read(t)?),
            Lazy::List(v) => Value::List(v.iter().map(|x| self.value_of(x)).collect::<Result<_>>()?),
            Lazy::Tuple(v) => Value::Tuple(v.iter().map(|x| self.value_of(x)).collect::<Result<_>>()?),
            Lazy::Dict(d) => Value::Dict(d.iter().map(|(k, x)| Ok((k.clone(), self.value_of(x)?))).collect::<Result<_>>()?),
        })
    }
}
