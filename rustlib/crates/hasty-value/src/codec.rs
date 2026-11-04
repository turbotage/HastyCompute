//! Block compression: raw bytes → [delta] → [byte shuffle] → codec. Every block is independent (parallel encode /
//! decode, streaming), carries its own header and falls back to stored bytes when compression does not pay.

use crate::{Error, Result};

pub(crate) const CODEC_NONE: u8 = 0;
pub(crate) const CODEC_LZ4: u8 = 1;
pub(crate) const CODEC_ZSTD: u8 = 2;
pub(crate) const FILTER_SHUFFLE: u8 = 1;
pub(crate) const FILTER_DELTA: u8 = 2;
pub(crate) const FLAG_CRC: u8 = 1;
/// block header: codec u8, filter u8, word u8, flags u8, raw_len u32, comp_len u32 (+ crc32 u32 if FLAG_CRC)
pub(crate) const BLOCK_HEADER: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    None,
    /// LZ4 block format: ~GB/s per core both ways; decodable in the browser
    Lz4,
    /// zstd frame at the given level (1–22; 3 is a good default for files)
    Zstd(i32),
}

impl Codec {
    pub fn parse(name: &str, level: i32) -> Result<Self> {
        match name {
            "" | "lz4" => Ok(Codec::Lz4),
            "zstd" => Ok(Codec::Zstd(if level == 0 { 3 } else { level })),
            "none" => Ok(Codec::None),
            _ => Err(Error::Codec(format!("unknown codec {name}"))),
        }
    }
}

/// Encoding options.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub codec: Codec,
    /// raw bytes per block (rounded down to a multiple of 16)
    pub block_size: usize,
    /// byte-shuffle multi-byte words (by real component for complex types)
    pub shuffle: bool,
    /// integer delta before shuffle (integer dtypes only; smooth quantised images)
    pub delta: bool,
    /// CRC32 of every compressed payload
    pub checksum: bool,
    /// compress blocks in parallel (rayon)
    pub parallel: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { codec: Codec::Lz4, block_size: 1 << 20, shuffle: true, delta: false, checksum: false, parallel: true }
    }
}

impl Options {
    pub fn none() -> Self {
        Self { codec: Codec::None, shuffle: false, ..Self::default() }
    }

    pub fn zstd(level: i32) -> Self {
        Self { codec: Codec::Zstd(level), ..Self::default() }
    }

    pub(crate) fn block_bytes(&self) -> usize {
        (self.block_size.max(16) / 16) * 16
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockHeader {
    pub codec: u8,
    pub filter: u8,
    pub word: u8,
    pub flags: u8,
    pub raw_len: u32,
    pub comp_len: u32,
    pub crc: u32,
}

impl BlockHeader {
    pub fn len(&self) -> usize {
        BLOCK_HEADER + if self.flags & FLAG_CRC != 0 { 4 } else { 0 }
    }

    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.len() < BLOCK_HEADER {
            return None;
        }
        let flags = b[3];
        let crc = if flags & FLAG_CRC != 0 {
            if b.len() < BLOCK_HEADER + 4 {
                return None;
            }
            u32::from_le_bytes(b[12..16].try_into().unwrap())
        } else {
            0
        };
        Some(Self {
            codec: b[0],
            filter: b[1],
            word: b[2],
            flags,
            raw_len: u32::from_le_bytes(b[4..8].try_into().unwrap()),
            comp_len: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            crc,
        })
    }

    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&[self.codec, self.filter, self.word, self.flags]);
        out.extend_from_slice(&self.raw_len.to_le_bytes());
        out.extend_from_slice(&self.comp_len.to_le_bytes());
        if self.flags & FLAG_CRC != 0 {
            out.extend_from_slice(&self.crc.to_le_bytes());
        }
    }
}

/// byte shuffle: word i, byte b -> position b·n + i (n = number of whole words; a ragged tail is copied as is)
pub(crate) fn shuffle(src: &[u8], word: usize, dst: &mut [u8]) {
    let n = src.len() / word;
    match word {
        2 => shuffle_w::<2>(&src[..2 * n], &mut dst[..2 * n]),
        4 => shuffle_w::<4>(&src[..4 * n], &mut dst[..4 * n]),
        8 => shuffle_w::<8>(&src[..8 * n], &mut dst[..8 * n]),
        _ => {
            for b in 0..word {
                let d = &mut dst[b * n..(b + 1) * n];
                for (i, x) in d.iter_mut().enumerate() {
                    *x = src[i * word + b];
                }
            }
        }
    }
    dst[n * word..].copy_from_slice(&src[n * word..]);
}

pub(crate) fn unshuffle(src: &[u8], word: usize, dst: &mut [u8]) {
    let n = src.len() / word;
    match word {
        2 => unshuffle_w::<2>(&src[..2 * n], &mut dst[..2 * n]),
        4 => unshuffle_w::<4>(&src[..4 * n], &mut dst[..4 * n]),
        8 => unshuffle_w::<8>(&src[..8 * n], &mut dst[..8 * n]),
        _ => {
            for b in 0..word {
                let s = &src[b * n..(b + 1) * n];
                for (i, &x) in s.iter().enumerate() {
                    dst[i * word + b] = x;
                }
            }
        }
    }
    dst[n * word..].copy_from_slice(&src[n * word..]);
}

fn shuffle_w<const W: usize>(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / W;
    let planes: [&mut [u8]; W] = {
        let mut rest = dst;
        std::array::from_fn(|_| {
            let (a, r) = std::mem::take(&mut rest).split_at_mut(n);
            rest = r;
            a
        })
    };
    for (i, w) in src.chunks_exact(W).enumerate() {
        for b in 0..W {
            planes[b][i] = w[b];
        }
    }
}

fn unshuffle_w<const W: usize>(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / W;
    let planes: [&[u8]; W] = std::array::from_fn(|b| &src[b * n..(b + 1) * n]);
    for (i, w) in dst.chunks_exact_mut(W).enumerate() {
        for b in 0..W {
            w[b] = planes[b][i];
        }
    }
}

macro_rules! delta_impl {
    ($buf:expr, $t:ty, $enc:expr) => {{
        let w = std::mem::size_of::<$t>();
        let n = $buf.len() / w;
        let mut prev: $t = 0;
        for i in 0..n {
            let x = <$t>::from_le_bytes($buf[i * w..(i + 1) * w].try_into().unwrap());
            let y = if $enc { x.wrapping_sub(prev) } else { x.wrapping_add(prev) };
            prev = if $enc { x } else { y };
            $buf[i * w..(i + 1) * w].copy_from_slice(&y.to_le_bytes());
        }
    }};
}

/// wrapping difference of consecutive little-endian words (in place)
pub(crate) fn delta(buf: &mut [u8], word: usize, encode: bool) {
    match word {
        1 => delta_impl!(buf, u8, encode),
        2 => delta_impl!(buf, u16, encode),
        4 => delta_impl!(buf, u32, encode),
        8 => delta_impl!(buf, u64, encode),
        _ => {}
    }
}

/// Encode one block (header + payload appended to `out`). `word` = shuffle word size, `delta` only for integers.
pub(crate) fn encode_block(raw: &[u8], word: usize, delta_ok: bool, o: &Options, out: &mut Vec<u8>) {
    if o.codec == Codec::None {
        return finish(raw, raw, 0, word, o, out);
    }
    let mut filter = 0u8;
    let mut work = std::borrow::Cow::Borrowed(raw);
    if o.delta && delta_ok && matches!(word, 1 | 2 | 4 | 8) {
        let mut w = raw.to_vec();
        delta(&mut w, word, true);
        work = std::borrow::Cow::Owned(w);
        filter |= FILTER_DELTA;
    }
    if o.shuffle && word > 1 {
        let mut sh = vec![0u8; raw.len()];
        shuffle(&work, word, &mut sh);
        work = std::borrow::Cow::Owned(sh);
        filter |= FILTER_SHUFFLE;
    }
    finish(raw, &work, filter, word, o, out)
}

fn finish(raw: &[u8], filtered: &[u8], filter: u8, word: usize, o: &Options, out: &mut Vec<u8>) {
    let comp: Option<(u8, Vec<u8>)> = match o.codec {
        Codec::None => None,
        Codec::Lz4 => Some((CODEC_LZ4, lz4_flex::block::compress(filtered))),
        Codec::Zstd(level) => zstd::bulk::compress(filtered, level).ok().map(|c| (CODEC_ZSTD, c)),
    };
    let (codec, filter, payload): (u8, u8, &[u8]) = match &comp {
        Some((c, p)) if p.len() < raw.len() => (*c, filter, &p[..]),
        _ => (CODEC_NONE, 0, raw), // incompressible (or no codec): store the raw bytes, no filters
    };
    let flags = if o.checksum { FLAG_CRC } else { 0 };
    let h = BlockHeader { codec, filter, word: word as u8, flags, raw_len: raw.len() as u32, comp_len: payload.len() as u32, crc: if o.checksum { crc32fast::hash(payload) } else { 0 } };
    h.write(out);
    out.extend_from_slice(payload);
}

/// Decode one block payload into `out` (length `h.raw_len`).
pub(crate) fn decode_block(h: &BlockHeader, payload: &[u8], out: &mut [u8]) -> Result<()> {
    if out.len() != h.raw_len as usize || payload.len() != h.comp_len as usize {
        return Err(Error::Format("block length mismatch".into()));
    }
    if h.flags & FLAG_CRC != 0 && crc32fast::hash(payload) != h.crc {
        return Err(Error::Codec("block checksum mismatch".into()));
    }
    let filtered = h.filter != 0;
    let mut tmp = if filtered { vec![0u8; out.len()] } else { Vec::new() };
    let dst: &mut [u8] = if filtered { &mut tmp[..] } else { &mut *out };
    match h.codec {
        CODEC_NONE => dst.copy_from_slice(payload),
        CODEC_LZ4 => {
            let n = lz4_flex::block::decompress_into(payload, dst).map_err(|e| Error::Codec(format!("lz4: {e}")))?;
            if n != dst.len() {
                return Err(Error::Codec(format!("lz4: {n} of {} bytes", dst.len())));
            }
        }
        CODEC_ZSTD => {
            let n = zstd::bulk::decompress_to_buffer(payload, dst).map_err(|e| Error::Codec(format!("zstd: {e}")))?;
            if n != dst.len() {
                return Err(Error::Codec(format!("zstd: {n} of {} bytes", dst.len())));
            }
        }
        c => return Err(Error::Format(format!("unknown codec {c}"))),
    }
    if filtered {
        let word = h.word as usize;
        let work = &tmp;
        if h.filter & FILTER_SHUFFLE != 0 {
            unshuffle(work, word, out);
        } else {
            out.copy_from_slice(work);
        }
        if h.filter & FILTER_DELTA != 0 {
            delta(out, word, false);
        }
    }
    Ok(())
}
