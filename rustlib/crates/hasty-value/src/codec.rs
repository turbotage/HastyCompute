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
    /// end-to-end stream digest (trailer); `None` costs nothing
    pub integrity: crate::Integrity,
}

impl Default for Options {
    fn default() -> Self {
        Self { codec: Codec::Lz4, block_size: 1 << 20, shuffle: true, delta: false, checksum: false, parallel: true, integrity: crate::Integrity::None }
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
#[doc(hidden)]
pub fn shuffle(src: &[u8], word: usize, dst: &mut [u8]) {
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

#[doc(hidden)]
pub fn unshuffle(src: &[u8], word: usize, dst: &mut [u8]) {
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
    let done = simd::shuffle(W, src, dst, n);
    let planes: [&mut [u8]; W] = {
        let mut rest = dst;
        std::array::from_fn(|_| {
            let (a, r) = std::mem::take(&mut rest).split_at_mut(n);
            rest = r;
            a
        })
    };
    for (i, w) in src.chunks_exact(W).enumerate().skip(done) {
        for b in 0..W {
            planes[b][i] = w[b];
        }
    }
}

fn unshuffle_w<const W: usize>(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / W;
    let done = simd::unshuffle(W, src, dst, n);
    let planes: [&[u8]; W] = std::array::from_fn(|b| &src[b * n..(b + 1) * n]);
    for (i, w) in dst.chunks_exact_mut(W).enumerate().skip(done) {
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

thread_local! {
    /// per-thread scratch (filter output, compressor output / decode staging): reused across blocks, so the memory
    /// stays warm in cache and no block pays for a fresh allocation and its page faults
    static SCRATCH: std::cell::RefCell<[Vec<u8>; 3]> = const { std::cell::RefCell::new([Vec::new(), Vec::new(), Vec::new()]) };
}

/// Encode one block (header + payload appended to `out`). `word` = shuffle word size, `delta` only for integers.
pub(crate) fn encode_block(raw: &[u8], word: usize, delta_ok: bool, o: &Options, out: &mut Vec<u8>) {
    if o.codec == Codec::None {
        return finish(raw, raw, 0, word, o, out, &mut Vec::new());
    }
    SCRATCH.with(|s| {
        let [a, b, c] = &mut *s.borrow_mut();
        let mut filter = 0u8;
        let mut cur: &[u8] = raw;
        if o.delta && delta_ok && matches!(word, 1 | 2 | 4 | 8) {
            a.clear();
            a.extend_from_slice(raw);
            delta(a, word, true);
            cur = &a[..];
            filter |= FILTER_DELTA;
        }
        if o.shuffle && word > 1 {
            b.resize(raw.len(), 0);
            shuffle(cur, word, &mut b[..]);
            cur = &b[..];
            filter |= FILTER_SHUFFLE;
        }
        finish(raw, cur, filter, word, o, out, c)
    })
}

fn finish(raw: &[u8], filtered: &[u8], filter: u8, word: usize, o: &Options, out: &mut Vec<u8>, scratch: &mut Vec<u8>) {
    let comp: Option<(u8, usize)> = match o.codec {
        Codec::None => None,
        Codec::Lz4 => {
            scratch.resize(lz4_flex::block::get_maximum_output_size(filtered.len()), 0);
            lz4_flex::block::compress_into(filtered, scratch).ok().map(|n| (CODEC_LZ4, n))
        }
        Codec::Zstd(level) => {
            scratch.resize(zstd::zstd_safe::compress_bound(filtered.len()), 0);
            zstd::bulk::compress_to_buffer(filtered, scratch, level).ok().map(|n| (CODEC_ZSTD, n))
        }
    };
    let (codec, filter, payload): (u8, u8, &[u8]) = match comp {
        Some((c, n)) if n < raw.len() => (c, filter, &scratch[..n]),
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
    if h.filter == 0 {
        return decompress(h.codec, payload, out);
    }
    SCRATCH.with(|s| {
        let tmp = &mut s.borrow_mut()[0];
        tmp.resize(out.len(), 0);
        decompress(h.codec, payload, &mut tmp[..])?;
        let word = h.word as usize;
        if h.filter & FILTER_SHUFFLE != 0 {
            unshuffle(tmp, word, out);
        } else {
            out.copy_from_slice(tmp);
        }
        if h.filter & FILTER_DELTA != 0 {
            delta(out, word, false);
        }
        Ok(())
    })
}

fn decompress(codec: u8, payload: &[u8], dst: &mut [u8]) -> Result<()> {
    match codec {
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
    Ok(())
}

/// Compress one self-describing block (header + payload): the public form of the format's block encoder, for
/// containers outside `.hv` (e.g. `hasty_recon::kstore`). `word` is the byte-shuffle word size (4 for f32 / complex
/// f32 components, 1 = no shuffle); `delta` applies the integer delta filter (integer data only).
pub fn compress_block(raw: &[u8], word: usize, delta: bool, o: &Options) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 2 + BLOCK_HEADER + 4);
    encode_block(raw, word, delta, o, &mut out);
    out
}

/// Raw length of a block made by [`compress_block`] (from its header).
pub fn block_raw_len(block: &[u8]) -> Result<usize> {
    BlockHeader::parse(block).map(|h| h.raw_len as usize).ok_or_else(|| Error::Format("short block".into()))
}

/// Decompress one block made by [`compress_block`] into `out` (exactly its raw length).
pub fn decompress_block(block: &[u8], out: &mut [u8]) -> Result<()> {
    let h = BlockHeader::parse(block).ok_or_else(|| Error::Format("short block".into()))?;
    let n = h.len();
    if block.len() < n + h.comp_len as usize {
        return Err(Error::Format("truncated block".into()));
    }
    decode_block(&h, &block[n..n + h.comp_len as usize], out)
}

/// SSE2 (always present on x86_64) byte transposes for 2- and 4-byte words; other targets use the scalar loops.
/// Each returns the number of leading words handled (a multiple of 16), the caller finishes the rest.
mod simd {
    #[cfg(target_arch = "x86_64")]
    pub fn shuffle(w: usize, src: &[u8], dst: &mut [u8], n: usize) -> usize {
        use std::arch::x86_64::*;
        let blocks = n / 16;
        // SAFETY: SSE2 is baseline on x86_64; every load / store below stays inside src[..w*n] / dst[..w*n]
        unsafe {
            match w {
                4 => {
                    // 4 sixteen-word blocks per round, so every plane receives one full 64-byte line
                    let mut i = 0;
                    while i + 4 <= blocks {
                        let mut out = [[_mm_setzero_si128(); 4]; 4]; // [plane][block]
                        for j in 0..4 {
                            let p = src.as_ptr().add((i + j) * 64) as *const __m128i;
                            let (a0, a1, a2, a3) = (_mm_loadu_si128(p), _mm_loadu_si128(p.add(1)), _mm_loadu_si128(p.add(2)), _mm_loadu_si128(p.add(3)));
                            let (l0, h0) = (_mm_unpacklo_epi8(a0, a1), _mm_unpackhi_epi8(a0, a1));
                            let (l1, h1) = (_mm_unpacklo_epi8(a2, a3), _mm_unpackhi_epi8(a2, a3));
                            let (m0, m1) = (_mm_unpacklo_epi8(l0, h0), _mm_unpackhi_epi8(l0, h0));
                            let (n0, n1) = (_mm_unpacklo_epi8(l1, h1), _mm_unpackhi_epi8(l1, h1));
                            let (p01, p23) = (_mm_unpacklo_epi8(m0, m1), _mm_unpackhi_epi8(m0, m1));
                            let (q01, q23) = (_mm_unpacklo_epi8(n0, n1), _mm_unpackhi_epi8(n0, n1));
                            out[0][j] = _mm_unpacklo_epi64(p01, q01);
                            out[1][j] = _mm_unpackhi_epi64(p01, q01);
                            out[2][j] = _mm_unpacklo_epi64(p23, q23);
                            out[3][j] = _mm_unpackhi_epi64(p23, q23);
                        }
                        for (b, o) in out.iter().enumerate() {
                            let d = dst.as_mut_ptr().add(i * 16 + b * n) as *mut __m128i;
                            for (j, v) in o.iter().enumerate() {
                                _mm_storeu_si128(d.add(j), *v);
                            }
                        }
                        i += 4;
                    }
                    while i < blocks {
                        let p = src.as_ptr().add(i * 64) as *const __m128i;
                        let (a0, a1, a2, a3) = (_mm_loadu_si128(p), _mm_loadu_si128(p.add(1)), _mm_loadu_si128(p.add(2)), _mm_loadu_si128(p.add(3)));
                        let (l0, h0) = (_mm_unpacklo_epi8(a0, a1), _mm_unpackhi_epi8(a0, a1));
                        let (l1, h1) = (_mm_unpacklo_epi8(a2, a3), _mm_unpackhi_epi8(a2, a3));
                        let (m0, m1) = (_mm_unpacklo_epi8(l0, h0), _mm_unpackhi_epi8(l0, h0));
                        let (n0, n1) = (_mm_unpacklo_epi8(l1, h1), _mm_unpackhi_epi8(l1, h1));
                        let (p01, p23) = (_mm_unpacklo_epi8(m0, m1), _mm_unpackhi_epi8(m0, m1));
                        let (q01, q23) = (_mm_unpacklo_epi8(n0, n1), _mm_unpackhi_epi8(n0, n1));
                        let d = dst.as_mut_ptr().add(i * 16);
                        _mm_storeu_si128(d as *mut __m128i, _mm_unpacklo_epi64(p01, q01));
                        _mm_storeu_si128(d.add(n) as *mut __m128i, _mm_unpackhi_epi64(p01, q01));
                        _mm_storeu_si128(d.add(2 * n) as *mut __m128i, _mm_unpacklo_epi64(p23, q23));
                        _mm_storeu_si128(d.add(3 * n) as *mut __m128i, _mm_unpackhi_epi64(p23, q23));
                        i += 1;
                    }
                }
                2 => {
                    let mask = _mm_set1_epi16(0x00ff);
                    for i in 0..blocks {
                        let p = src.as_ptr().add(i * 32) as *const __m128i;
                        let (a0, a1) = (_mm_loadu_si128(p), _mm_loadu_si128(p.add(1)));
                        let lo = _mm_packus_epi16(_mm_and_si128(a0, mask), _mm_and_si128(a1, mask));
                        let hi = _mm_packus_epi16(_mm_srli_epi16(a0, 8), _mm_srli_epi16(a1, 8));
                        let d = dst.as_mut_ptr().add(i * 16);
                        _mm_storeu_si128(d as *mut __m128i, lo);
                        _mm_storeu_si128(d.add(n) as *mut __m128i, hi);
                    }
                }
                _ => return 0,
            }
        }
        blocks * 16
    }

    #[cfg(target_arch = "x86_64")]
    pub fn unshuffle(w: usize, src: &[u8], dst: &mut [u8], n: usize) -> usize {
        use std::arch::x86_64::*;
        let blocks = n / 16;
        // SAFETY: as in `shuffle`
        unsafe {
            match w {
                4 => {
                    for i in 0..blocks {
                        let s = src.as_ptr().add(i * 16);
                        let ld = |b: usize| _mm_loadu_si128(s.add(b * n) as *const __m128i);
                        let (p0, p1, p2, p3) = (ld(0), ld(1), ld(2), ld(3));
                        let (u01l, u01h) = (_mm_unpacklo_epi8(p0, p1), _mm_unpackhi_epi8(p0, p1));
                        let (u23l, u23h) = (_mm_unpacklo_epi8(p2, p3), _mm_unpackhi_epi8(p2, p3));
                        let d = dst.as_mut_ptr().add(i * 64) as *mut __m128i;
                        _mm_storeu_si128(d, _mm_unpacklo_epi16(u01l, u23l));
                        _mm_storeu_si128(d.add(1), _mm_unpackhi_epi16(u01l, u23l));
                        _mm_storeu_si128(d.add(2), _mm_unpacklo_epi16(u01h, u23h));
                        _mm_storeu_si128(d.add(3), _mm_unpackhi_epi16(u01h, u23h));
                    }
                }
                2 => {
                    for i in 0..blocks {
                        let s = src.as_ptr().add(i * 16);
                        let (p0, p1) = (_mm_loadu_si128(s as *const __m128i), _mm_loadu_si128(s.add(n) as *const __m128i));
                        let d = dst.as_mut_ptr().add(i * 32) as *mut __m128i;
                        _mm_storeu_si128(d, _mm_unpacklo_epi8(p0, p1));
                        _mm_storeu_si128(d.add(1), _mm_unpackhi_epi8(p0, p1));
                    }
                }
                _ => return 0,
            }
        }
        blocks * 16
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn shuffle(_: usize, _: &[u8], _: &mut [u8], _: usize) -> usize {
        0
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn unshuffle(_: usize, _: &[u8], _: &mut [u8], _: usize) -> usize {
        0
    }
}
