//! Block-floating-point (BFP) storage of f32 / complex f32 buffers (research/bfp_storage.md).
//!
//! A block is 32 consecutive reals (16 complex values, re/im interleaved) of a buffer, aligned to 32 reals in the
//! buffer's flat index. Its values are stored as `bits`-bit two's complement mantissas q_i sharing one exponent
//! byte: x_i ≈ q_i · 2^(e − bits + 1), every |x_i| < 2^e. Value i of a block occupies bits [i·bits, i·bits + bits)
//! of the block's bit stream, which is `bits` little-endian 32-bit words; the exponent bytes are a separate array.
//! So a block of 32 values takes exactly `bits` words for any `bits` (6.0625 / 5.0625 / 4.0625 bytes per complex
//! value at 24 / 20 / 16 bits including the exponent byte).
//!
//! This is the host (CPU) codec, bit-for-bit identical to the device codec in `bfp.cuh` ([`BFP_CUH`]), for host RAM,
//! files and PCIe transfers of data the GPU reads or writes packed.
//!
//! * Exponent: the smallest e in [bits − 127, 128] with max|x| < 2^e and round(max|x| · 2^(bits−1−e)) ≤ 2^(bits−1) − 1,
//!   so no value saturates and |x − x̂| ≤ 2^(e − bits) for every value of the block ([`Bfp::error_bound`]). Decoded
//!   values of finite inputs are always finite: only values within half a step of 2^128 saturate (error < one step).
//! * Rounding: to nearest, ties to even, of the exact power-of-two scaling x · 2^(bits−1−e).
//! * Non-finite values: a block holding an Inf or NaN gets the exponent byte [`ENAN`] and decodes as all-NaN.
//! * Subnormal inputs are quantised like any other value (the lower exponent limit keeps the scale factors normal);
//!   tiny blocks may decode to subnormal floats, so device kernels using the codec must not flush subnormals to zero.

use hasty_cuda::C32;
use rayon::prelude::*;

/// CUDA header with the device codec (NVRTC: prepend to the kernel source).
pub const BFP_CUH: &str = include_str!("bfp.cuh");
/// reals per block
pub const VALS: usize = 32;
/// exponent byte = e + EBIAS
pub const EBIAS: i32 = 125;
/// exponent byte of a non-finite block
pub const ENAN: u8 = 255;
/// blocks per parallel work item of [`Bfp::encode`] / [`Bfp::decode`] (a multiple of 4, so chunk boundaries fall on
/// whole exponent words)
pub const CHUNK_BLOCKS: usize = 4096;

/// Round to nearest, ties to even, of |v| < 2^51: exact through the default rounding of the f64 addition (a libm
/// `roundeven` call on baseline x86-64 otherwise). Applied to exact power-of-two scalings of f32 values, so it gives
/// the f32 `round_ties_even` / the device's `__float2int_rn` result bit for bit.
#[inline(always)]
fn rne(v: f64) -> f64 {
    const M: f64 = 6_755_399_441_055_744.0; // 1.5 · 2^52
    (v + M) - M
}

#[inline]
fn pow2(k: i32) -> f32 {
    debug_assert!((-126..=127).contains(&k));
    f32::from_bits(((k + 127) as u32) << 23)
}

/// A BFP format: mantissa bits per real, 8 ..= 24.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bfp {
    bits: u32,
}

impl Bfp {
    pub const fn new(bits: u32) -> Self {
        assert!(bits >= 8 && bits <= 24, "BFP mantissa bits must be in 8..=24");
        Self { bits }
    }

    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// largest mantissa magnitude 2^(bits−1) − 1
    pub const fn lim(self) -> i32 {
        (1 << (self.bits - 1)) - 1
    }

    /// blocks holding `n` reals
    pub const fn blocks(n: usize) -> usize {
        n.div_ceil(VALS)
    }

    /// mantissa words for `n` reals
    pub const fn mant_words(self, n: usize) -> usize {
        Self::blocks(n) * self.bits as usize
    }

    /// stored bytes for `n` reals (mantissa words + one exponent byte per block)
    pub const fn bytes(self, n: usize) -> usize {
        self.mant_words(n) * 4 + Self::blocks(n)
    }

    /// stored bytes per complex value (16 per block)
    pub fn bytes_per_complex(self) -> f64 {
        (4.0 * self.bits as f64 + 1.0) / 16.0
    }

    /// values per thread run (fills whole words): 32 / gcd(bits, 32)
    pub const fn run_values(self) -> usize {
        32 / gcd(self.bits as usize, 32)
    }

    /// words per thread run: run_values · bits / 32
    pub const fn run_words(self) -> usize {
        self.run_values() * self.bits as usize / 32
    }

    /// exponent byte of a block whose max |x| has the float bits `mbits` (sign cleared)
    pub fn estore(self, mbits: u32) -> u8 {
        if mbits >= 0x7f80_0000 {
            return ENAN;
        }
        let b = self.bits as i32;
        let ee = (mbits >> 23) as i32;
        let mut e = if ee == 0 { -126 } else { ee - 126 };
        e = e.max(b - 127);
        if e < 128 && rne((f32::from_bits(mbits) * pow2(b - 1 - e)) as f64) as i32 > self.lim() {
            e += 1;
        }
        (e + EBIAS) as u8
    }

    /// exponent e of a stored byte (not for [`ENAN`])
    pub fn exponent(es: u8) -> i32 {
        es as i32 - EBIAS
    }

    /// quantisation step 2^(e − bits + 1) of a block with exponent byte `es`
    pub fn step(self, es: u8) -> f32 {
        pow2(Self::exponent(es) - self.bits as i32 + 1)
    }

    /// per-value error bound of a block: |x − x̂| ≤ 2^(e − bits) (half a step)
    pub fn error_bound(self, es: u8) -> f64 {
        (2.0f64).powi(Self::exponent(es) - self.bits as i32)
    }

    #[inline]
    pub fn quant(self, x: f32, es: u8) -> i32 {
        let b = self.bits as i32;
        let q = rne((x * pow2(b - 1 - Self::exponent(es))) as f64) as i32;
        q.clamp(-self.lim(), self.lim())
    }

    #[inline]
    pub fn dequant(self, q: i32, es: u8) -> f32 {
        if es == ENAN {
            return f32::NAN;
        }
        q as f32 * pow2(Self::exponent(es) - self.bits as i32 + 1)
    }

    /// Encode one block: `x` holds up to 32 reals (missing values are zeros), `mant` exactly `bits` words.
    /// Returns the exponent byte.
    pub fn encode_block(self, x: &[f32], mant: &mut [u32]) -> u8 {
        debug_assert!(x.len() <= VALS && mant.len() == self.bits as usize);
        let m = x.iter().map(|v| v.to_bits() & 0x7fff_ffff).max().unwrap_or(0);
        let es = self.estore(m);
        mant.fill(0);
        if es == ENAN {
            return es;
        }
        let b = self.bits;
        let mask = (1u64 << b) - 1;
        // the scaling 2^(bits−1−e) is exact, so the f64 product equals the f32 product of `quant` wherever it can
        // round to a non-zero integer
        let sc = pow2(b as i32 - 1 - Self::exponent(es)) as f64;
        let lim = self.lim() as f64;
        if b == 16 {
            // two values per word, no straddling (the general loop gives the same words)
            for (k, w) in mant.iter_mut().enumerate() {
                let q0 = x.get(2 * k).map_or(0, |&v| rne(v as f64 * sc).clamp(-lim, lim) as i32);
                let q1 = x.get(2 * k + 1).map_or(0, |&v| rne(v as f64 * sc).clamp(-lim, lim) as i32);
                *w = (q0 as u32 & 0xffff) | (q1 as u32) << 16;
            }
            return es;
        }
        // values streamed into a 64-bit accumulator, one word out per 32 bits
        let (mut acc, mut nb, mut k) = (0u64, 0u32, 0usize);
        for &v in x {
            let q = rne(v as f64 * sc).clamp(-lim, lim) as i64;
            acc |= (q as u64 & mask) << nb;
            nb += b;
            if nb >= 32 {
                mant[k] = acc as u32;
                k += 1;
                acc >>= 32;
                nb -= 32;
            }
        }
        if nb > 0 {
            mant[k] = acc as u32;
        }
        es
    }

    /// Mantissa integer of value i of a block.
    #[inline]
    pub fn mantissa(self, mant: &[u32], i: usize) -> i32 {
        let b = self.bits as usize;
        let bit = i * b;
        let (k, s) = (bit >> 5, bit & 31);
        let lo = mant[k] as u64;
        let hi = if s + b > 32 { mant[k + 1] as u64 } else { 0 };
        let u = (((hi << 32) | lo) >> s) as u32;
        ((u << (32 - b)) as i32) >> (32 - b)
    }

    /// Decode one block into `out` (up to 32 reals).
    pub fn decode_block(self, mant: &[u32], es: u8, out: &mut [f32]) {
        debug_assert!(out.len() <= VALS && mant.len() == self.bits as usize);
        if es == ENAN {
            out.fill(f32::NAN);
            return;
        }
        let b = self.bits;
        let st = pow2(Self::exponent(es) - b as i32 + 1);
        if b == 16 {
            for (i, o) in out.iter_mut().enumerate() {
                let w = mant[i >> 1];
                *o = ((if i & 1 == 0 { w } else { w >> 16 }) as u16 as i16) as f32 * st;
            }
            return;
        }
        let mask = (1u64 << b) - 1;
        let sh = 32 - b;
        let (mut acc, mut nb, mut k) = (0u64, 0u32, 0usize);
        for o in out.iter_mut() {
            if nb < b {
                acc |= (mant[k] as u64) << nb;
                k += 1;
                nb += 32;
            }
            let u = (acc & mask) as u32;
            acc >>= b;
            nb -= b;
            *o = (((u << sh) as i32) >> sh) as f32 * st;
        }
    }

    /// Encode `x` on the calling thread: block k of `x` into words `mant[k·bits ..]` and byte `ex[k]` (the tail block
    /// zero-padded). The parallel [`encode`](Self::encode) runs this loop per chunk, so both write the same bits.
    pub fn encode_serial(self, x: &[f32], mant: &mut [u32], ex: &mut [u8]) {
        let nb = Self::blocks(x.len());
        assert!(mant.len() >= nb * self.bits as usize && ex.len() >= nb, "bfp encode: output too small");
        let b = self.bits as usize;
        for (k, (mb, eb)) in mant[..nb * b].chunks_mut(b).zip(ex[..nb].iter_mut()).enumerate() {
            let i0 = k * VALS;
            *eb = self.encode_block(&x[i0..(i0 + VALS).min(x.len())], mb);
        }
    }

    /// Decode on the calling thread (`out.len()` reals); see [`encode_serial`](Self::encode_serial).
    pub fn decode_serial(self, mant: &[u32], ex: &[u8], out: &mut [f32]) {
        let nb = Self::blocks(out.len());
        assert!(mant.len() >= nb * self.bits as usize && ex.len() >= nb, "bfp decode: input too small");
        let b = self.bits as usize;
        for (k, ob) in out.chunks_mut(VALS).enumerate() {
            self.decode_block(&mant[k * b..(k + 1) * b], ex[k], ob);
        }
    }

    /// Encode `x` (any length; the tail block is zero-padded) into `mant` ([`mant_words`](Self::mant_words)) and
    /// `ex` ([`blocks`](Self::blocks) bytes). Parallel over chunks of [`CHUNK_BLOCKS`] blocks.
    pub fn encode(self, x: &[f32], mant: &mut [u32], ex: &mut [u8]) {
        let nb = Self::blocks(x.len());
        assert!(mant.len() >= nb * self.bits as usize && ex.len() >= nb, "bfp encode: output too small");
        let b = self.bits as usize;
        mant[..nb * b].par_chunks_mut(CHUNK_BLOCKS * b).zip(ex[..nb].par_chunks_mut(CHUNK_BLOCKS)).enumerate().for_each(|(c, (m, e))| {
            let i0 = c * CHUNK_BLOCKS * VALS;
            self.encode_serial(&x[i0..(i0 + CHUNK_BLOCKS * VALS).min(x.len())], m, e);
        });
    }

    /// Decode into `out` (its length selects how many reals are decoded). Parallel over chunks of blocks.
    pub fn decode(self, mant: &[u32], ex: &[u8], out: &mut [f32]) {
        let nb = Self::blocks(out.len());
        assert!(mant.len() >= nb * self.bits as usize && ex.len() >= nb, "bfp decode: input too small");
        let b = self.bits as usize;
        out.par_chunks_mut(VALS * CHUNK_BLOCKS).enumerate().for_each(|(c, o)| {
            let k0 = c * CHUNK_BLOCKS;
            self.decode_serial(&mant[k0 * b..], &ex[k0..], o);
        });
    }

    /// quantisation step of a block whose max |x| is `max_abs` (finite)
    pub fn block_step(self, max_abs: f32) -> f64 {
        let es = self.estore(max_abs.abs().to_bits());
        if es == ENAN {
            return f64::NAN;
        }
        (2.0f64).powi(Self::exponent(es) - self.bits as i32 + 1)
    }

    /// The smallest width in 8 ..= 24 whose step for a block with max |x| = `max_abs` is at most `step_max` (the
    /// k-space rule of research/bfp_storage.md §3.6: step ≤ q σ, i.e. b ≥ log2(peak / σ) + 2 + log2(1 / q)); None
    /// when even 24 bits are too coarse. A buffer's worst block is the one holding its peak, so the rule applied to
    /// the buffer's max |x| bounds every block's step.
    pub fn bits_for_step(max_abs: f32, step_max: f64) -> Option<u32> {
        (8..=24).find(|&b| Bfp::new(b).block_step(max_abs) <= step_max)
    }

    pub fn encode_c(self, x: &[C32], mant: &mut [u32], ex: &mut [u8]) {
        self.encode(as_reals(x), mant, ex)
    }

    pub fn decode_c(self, mant: &[u32], ex: &[u8], out: &mut [C32]) {
        self.decode(mant, ex, as_reals_mut(out))
    }

    /// encode then decode (the stored values as the consumer sees them)
    pub fn round_trip(self, x: &[f32]) -> Vec<f32> {
        let buf = BfpBuf::encode(self, x);
        buf.decode()
    }
}

const fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

pub fn as_reals(x: &[C32]) -> &[f32] {
    // C32 is repr(C) { re: f32, im: f32 }
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len() * 2) }
}

pub fn as_reals_mut(x: &mut [C32]) -> &mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut f32, x.len() * 2) }
}

/// An owned BFP buffer of `len` reals: mantissa words of all blocks, one exponent byte per block (padded to whole
/// 32-bit words). Values are reached through f32 views of block-aligned ranges, decoded on demand
/// ([`view`](Self::view), [`par_views`](Self::par_views)); no f32 copy of the whole buffer is ever needed.
#[derive(Clone, Debug)]
pub struct BfpBuf {
    pub fmt: Bfp,
    pub len: usize,
    pub mant: Vec<u32>,
    pub ex: Vec<u8>,
}

/// A read view of reals [r0, r0 + len) of a [`BfpBuf`] (r0 a multiple of 32: whole blocks).
#[derive(Clone, Copy, Debug)]
pub struct BfpView<'a> {
    pub fmt: Bfp,
    pub r0: usize,
    pub len: usize,
    pub mant: &'a [u32],
    pub ex: &'a [u8],
}

/// A write view of reals [r0, r0 + len) of a [`BfpBuf`].
#[derive(Debug)]
pub struct BfpViewMut<'a> {
    pub fmt: Bfp,
    pub r0: usize,
    pub len: usize,
    pub mant: &'a mut [u32],
    pub ex: &'a mut [u8],
}

impl BfpView<'_> {
    /// decode all `len` reals into `out[..len]` (serial: views are the unit of parallel work)
    pub fn decode(&self, out: &mut [f32]) {
        self.fmt.decode_serial(self.mant, self.ex, &mut out[..self.len]);
    }
    /// decode as complex values (`len` even)
    pub fn decode_c(&self, out: &mut [C32]) {
        debug_assert!(self.len % 2 == 0);
        self.decode(as_reals_mut(&mut out[..self.len / 2]));
    }
    /// real value i of the view
    pub fn get(&self, i: usize) -> f32 {
        let b = self.fmt.bits as usize;
        let k = i / VALS;
        self.fmt.dequant(self.fmt.mantissa(&self.mant[k * b..(k + 1) * b], i % VALS), self.ex[k])
    }
}

impl BfpViewMut<'_> {
    /// encode `x` (exactly `len` reals; the tail block of the buffer is zero-padded)
    pub fn encode(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.len, "bfp view encode: {} reals for a view of {}", x.len(), self.len);
        self.fmt.encode_serial(x, self.mant, self.ex);
    }
    pub fn encode_c(&mut self, x: &[C32]) {
        self.encode(as_reals(x));
    }
    pub fn decode(&self, out: &mut [f32]) {
        self.fmt.decode_serial(self.mant, self.ex, &mut out[..self.len]);
    }
    pub fn decode_c(&self, out: &mut [C32]) {
        debug_assert!(self.len % 2 == 0);
        self.decode(as_reals_mut(&mut out[..self.len / 2]));
    }
    pub fn as_view(&self) -> BfpView<'_> {
        BfpView { fmt: self.fmt, r0: self.r0, len: self.len, mant: self.mant, ex: self.ex }
    }
}

impl BfpBuf {
    pub fn encode(fmt: Bfp, x: &[f32]) -> Self {
        let mut mant = vec![0u32; fmt.mant_words(x.len())];
        // exponent array padded to whole 32-bit words (4 blocks per word)
        let mut ex = vec![0u8; Bfp::blocks(x.len()).div_ceil(4) * 4];
        fmt.encode(x, &mut mant, &mut ex);
        Self { fmt, len: x.len(), mant, ex }
    }

    /// [`encode`](Self::encode) on the calling thread (same bits)
    pub fn encode_serial(fmt: Bfp, x: &[f32]) -> Self {
        let mut mant = vec![0u32; fmt.mant_words(x.len())];
        let mut ex = vec![0u8; Bfp::blocks(x.len()).div_ceil(4) * 4];
        fmt.encode_serial(x, &mut mant, &mut ex);
        Self { fmt, len: x.len(), mant, ex }
    }

    /// `len` reals, all zero (every block with the exponent byte the encoder gives a zero block)
    pub fn zeros(fmt: Bfp, len: usize) -> Self {
        let nb = Bfp::blocks(len);
        let mut ex = vec![0u8; nb.div_ceil(4) * 4];
        ex[..nb].fill(fmt.estore(0));
        Self { fmt, len, mant: vec![0u32; fmt.mant_words(len)], ex }
    }

    /// zeros of `n` complex values
    pub fn zeros_c(fmt: Bfp, n: usize) -> Self {
        Self::zeros(fmt, 2 * n)
    }

    /// blocks of the buffer
    pub fn nblocks(&self) -> usize {
        Bfp::blocks(self.len)
    }

    /// stored bytes of a buffer of `len` reals as [`encode`](Self::encode) allocates it (exponent bytes padded to
    /// whole words); [`Bfp::bytes`] without the padding
    pub fn bytes_for(fmt: Bfp, len: usize) -> usize {
        fmt.mant_words(len) * 4 + Bfp::blocks(len).div_ceil(4) * 4
    }

    fn range(&self, r0: usize, n: usize) -> (usize, usize, usize) {
        assert!(r0 % VALS == 0, "bfp view: start {r0} is not a multiple of {VALS} reals");
        assert!(r0 + n <= self.len, "bfp view: [{r0}, {}) outside a buffer of {}", r0 + n, self.len);
        let (k0, nb) = (r0 / VALS, Bfp::blocks(n));
        (k0, nb, self.fmt.bits as usize)
    }

    /// read view of reals [r0, r0 + n)
    pub fn view(&self, r0: usize, n: usize) -> BfpView<'_> {
        let (k0, nb, b) = self.range(r0, n);
        BfpView { fmt: self.fmt, r0, len: n, mant: &self.mant[k0 * b..(k0 + nb) * b], ex: &self.ex[k0..k0 + nb] }
    }

    /// write view of reals [r0, r0 + n); `n` must reach the end of the buffer or be a multiple of 32 (a view never
    /// shares a block with another)
    pub fn view_mut(&mut self, r0: usize, n: usize) -> BfpViewMut<'_> {
        let (k0, nb, b) = self.range(r0, n);
        assert!(n % VALS == 0 || r0 + n == self.len, "bfp write view: [{r0}, {}) ends inside a block", r0 + n);
        BfpViewMut { fmt: self.fmt, r0, len: n, mant: &mut self.mant[k0 * b..(k0 + nb) * b], ex: &mut self.ex[k0..k0 + nb] }
    }

    /// Read views of consecutive ranges of `chunk` reals (a multiple of 32; the last one shorter), in parallel.
    pub fn par_views(&self, chunk: usize) -> impl IndexedParallelIterator<Item = BfpView<'_>> {
        assert!(chunk > 0 && chunk % VALS == 0, "bfp views: chunk {chunk} is not a multiple of {VALS}");
        let (fmt, len, b, cb) = (self.fmt, self.len, self.fmt.bits as usize, chunk / VALS);
        let nb = self.nblocks();
        self.mant[..nb * b].par_chunks(cb * b).zip(self.ex[..nb].par_chunks(cb)).enumerate().map(move |(c, (m, e))| {
            let r0 = c * chunk;
            BfpView { fmt, r0, len: chunk.min(len - r0), mant: m, ex: e }
        })
    }

    /// Write views of consecutive ranges of `chunk` reals, in parallel (see [`par_views`](Self::par_views)).
    pub fn par_views_mut(&mut self, chunk: usize) -> impl IndexedParallelIterator<Item = BfpViewMut<'_>> {
        assert!(chunk > 0 && chunk % VALS == 0, "bfp views: chunk {chunk} is not a multiple of {VALS}");
        let (fmt, len, b, cb) = (self.fmt, self.len, self.fmt.bits as usize, chunk / VALS);
        let nb = self.nblocks();
        self.mant[..nb * b].par_chunks_mut(cb * b).zip(self.ex[..nb].par_chunks_mut(cb)).enumerate().map(move |(c, (m, e))| {
            let r0 = c * chunk;
            BfpViewMut { fmt, r0, len: chunk.min(len - r0), mant: m, ex: e }
        })
    }

    /// re-encode the whole buffer from `x` (same length), in parallel
    pub fn encode_from(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.len);
        self.fmt.encode(x, &mut self.mant, &mut self.ex);
    }

    pub fn encode_from_c(&mut self, x: &[C32]) {
        self.encode_from(as_reals(x));
    }

    /// decode the whole buffer into `out` (same length), in parallel
    pub fn decode_into(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.len);
        self.fmt.decode(&self.mant, &self.ex, out);
    }

    pub fn decode_into_c(&self, out: &mut [C32]) {
        self.decode_into(as_reals_mut(out));
    }

    /// copy packed words and exponent bytes produced elsewhere (the device pack kernel) into this buffer
    pub fn copy_packed(&mut self, mant: &[u32], ex: &[u8]) {
        let (nw, nb) = (self.fmt.mant_words(self.len), self.nblocks());
        self.mant[..nw].copy_from_slice(&mant[..nw]);
        self.ex[..nb].copy_from_slice(&ex[..nb]);
    }

    pub fn encode_c(fmt: Bfp, x: &[C32]) -> Self {
        Self::encode(fmt, as_reals(x))
    }

    pub fn decode(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.len];
        self.fmt.decode(&self.mant, &self.ex, &mut out);
        out
    }

    pub fn decode_c(&self) -> Vec<C32> {
        assert!(self.len % 2 == 0);
        let mut out = vec![C32::new(0.0, 0.0); self.len / 2];
        self.fmt.decode(&self.mant, &self.ex, as_reals_mut(&mut out));
        out
    }

    /// stored bytes (mantissas + exponent bytes)
    pub fn bytes(&self) -> usize {
        self.mant.len() * 4 + self.ex.len()
    }
}

pub mod gpu;
pub mod xfer;
