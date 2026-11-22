//! CUDA C++ source generation for one FFT "axis pass": a shared-memory Stockham autosort kernel that can optionally
//!   * read only the first `n_in` entries of every line (the rest are implicit zeros: zero-padded input, never read),
//!   * write only the first `n_out` entries (cropped output),
//!   * multiply loaded values by diagonal arrays (`pre`) and stored values by diagonal arrays (`post`), optionally accumulating into the output,
//!   * fuse "forward transform -> pointwise multiply by a kernel array -> inverse transform" in a single kernel (`conv`).
//!
//! Line addressing is fully runtime: a line is `(a, b)`; element `n` of the line is at `a*la + b*lb + n*sn` (separately for input
//! and output). If the transformed axis is contiguous (`sn == 1`) one thread group handles one line; otherwise a thread block handles a tile
//! of `lpb` consecutive `a` values (which must be the stride-1 dimension) so global accesses stay coalesced.

use std::fmt::Write;

use crate::radix::emit_butterfly;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiagOp {
    /// v *= d
    Mult,
    /// v *= conj(d)
    MultConj,
}

/// Storage format of the k-space multiplier `kern` used by fused convolution passes (smaller = less DRAM/PCIe traffic).
/// All formats are multiplied by the runtime scalar `kscale` after decoding (use it to bring fp16 data into range).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelFormat {
    /// interleaved complex in the transform precision
    Complex,
    /// real scalars in the transform precision (kernel known to be real, e.g. real symmetric weights)
    Real,
    /// complex as two IEEE binary16 packed in a u32 (re in the low half)
    ComplexHalf,
    /// complex as two bfloat16 packed in a u32 (re in the low half)
    ComplexBf16,
}

/// One term `coef * xi_x^px * xi_y^py * xi_z^pz` of a real polynomial diagonal; `xi_a = (idx_a - (N_a-1)/2) * 2/N_a` are the
/// FOV-normalised voxel coordinates. Evaluated in registers from the element position (needs `vox` in the pass config).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolyTerm {
    pub coef: f64,
    pub p: [u8; 3],
}

/// Caller-supplied CUDA code fused into an axis pass (strided, non-`arange` passes of more than one radix stage):
///   * `load`: an expression of type `cx` that replaces `in[GIN(e)]` in the first stage (the pass generates its input,
///     e.g. a NUFFT pad from the image, coil maps and segment diagonals; `in` is never read). In scope: `la` (line
///     index along the stride-1 tile dimension), `lb1`, `lb2` (the line-group sub-indices after the [`LineSplit`] gaps,
///     i.e. grid coordinates), `e` (entry along the line, inside the N_IN / N_IN_TAIL ranges).
///   * `epilogue`: the last stage writes its scaled outputs into shared memory instead of `out` (entry e of tile line
///     k at `fin[e * LPB + k]`, only entries inside N_OUT / N_OUT_TAIL) and this code runs after a barrier, once per
///     block (e.g. a NUFFT crop reducing the block's lines into an image). In scope: `fin`, `spare` (the other shared
///     buffer, N · LPB values, free; the same as `fin` when `FFT_INPLACE` is 1), `tid`, `LA0` (first line of the
///     tile), `la_cnt`, `lb1`, `lb2`, N, N_OUT, N_OUT_TAIL, LPB, TPL, FFT_INPLACE.
///
/// `params` (", type name, …") are appended to the kernel parameters (passed with [`crate::AxisPlan::launch_ext`]),
/// `prelude` is emitted before the kernel.
///
/// [`LineSplit`]: crate::LineSplit
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IoHook {
    pub params: String,
    pub prelude: String,
    pub load: Option<String>,
    pub epilogue: Option<String>,
    /// code run by the whole block before the first stage (with `load`), followed by a barrier: e.g. staging the
    /// block's inputs with coalesced loads into `stage` (the second shared buffer, N · LPB values, free until stage
    /// 1; only when `FFT_INPLACE` is 0). In scope as for `load`, plus `tid` and `LA0`.
    pub prologue: Option<String>,
}

/// Storage of a DRAM-resident buffer between two axis passes: plain complex in the transform precision, or block
/// floating point (`hasty_bfp`, research/bfp_storage.md) with the given mantissa bits (16 ..= 24).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Store {
    #[default]
    F32,
    Bfp(u32),
}

impl Store {
    /// `f32`, `bfp<b>` (b in 16 ..= 24)
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" | "" => Some(Store::F32),
            _ => s.strip_prefix("bfp").and_then(|b| b.parse::<u32>().ok()).filter(|b| (16..=24).contains(b)).map(Store::Bfp),
        }
    }
}

impl std::fmt::Display for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Store::F32 => write!(f, "f32"),
            Store::Bfp(b) => write!(f, "bfp{b}"),
        }
    }
}

/// Block-floating-point input or output of a strided axis pass of 16-line tiles ([`AxisCfg::bfp_in`] /
/// [`AxisCfg::bfp_out`]). Entry e of the tile's 16 lines is one 16-complex block of the BFP buffer, at block index
/// `lb · b_lb + se · b_sn + LT0 / 16` (`lb` the raw line-group index of the launch, before any [`crate::LineSplit`]
/// gap; `LT0` the tile's first line, a multiple of 16 in the flat line index; `b_sn`, `b_lb` runtime block strides).
/// `se` is the entry e itself, or with `compact` its index in the static entry range (the pass's N_IN / N_IN_TAIL
/// entries for an input, N_OUT / N_OUT_TAIL for an output, packed without the gap). Mantissas: `bits` 32-bit words
/// per block; exponents: one byte per block in a separate array. An exponent byte of 0 (never produced by the codec)
/// marks a block that was not written: it decodes as zeros.
///
/// `inplace`: the mantissas live in the pass's own f32 buffer instead of a separate array, in the cells of the block's
/// own 16 lines at that entry (the first 4 · bits of their 128 bytes; the tile's lines must be contiguous, stride-1
/// `la`, and every tile whole: la_cnt a multiple of 16). The exponents stay in the separate array. A pass then reads and
/// writes only its own tile's cells, so an in-place transform needs no second buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BfpIo {
    pub bits: u32,
    pub compact: bool,
    pub inplace: bool,
    /// in-place output only: after the first stage has read the tile, drop the tile's 128-byte cell runs of the
    /// entries the pass will write from the L2 without writing them back (`discard.global.L2`, sm_80+; a no-op
    /// below). The pass then rewrites only the first 4 · bits bytes of each run, so the run's last sector, dirty in L2
    /// from the producing pass (the x pass of a two-pass L2 slab), is no longer written back to DRAM: an in-place
    /// BFP output then costs ⌈4 · bits / 32⌉ of 4 sectors in DRAM also behind an L2 slab. The dropped sector holds
    /// undefined bytes afterwards (no reader looks at it).
    pub discard: bool,
}

impl BfpIo {
    /// values per thread run: a multiple of 32 / gcd(bits, 32) of at least 8 (runs fill whole words)
    pub fn run_values(&self) -> usize {
        let g = gcd(self.bits as usize, 32);
        (32 / g).max(8)
    }
    pub fn run_words(&self) -> usize {
        self.run_values() * self.bits as usize / 32
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// BFP store path of the axis passes (research/bfp_storage.md §10): env HASTY_FFT_BFP_STORE = epi (default) | fused.
/// fused: the last stage takes each block's exponent by a half-warp max in registers and writes the quantised values to unpadded
/// shared-memory rows; one barrier; then one thread per mantissa word, consecutive lanes = consecutive words of a
/// block). epi (the epilogue of §8.1 / sample_kernels §20.2.1): the last stage writes floats, a phase re-reads them
/// per run for the exponent and the quantisation, a second barrier, then vector pieces of 4 words. Bit-identical
/// output. Measured (RTX 4090, 320³ grid, 24 channels): fused −6 % on a whole-grid y pass (DRAM-bound) but +2 % on
/// the inverse z pass and +21 % on the L2-slab y pass (issue-bound: the half-warp shuffles and the per-word
/// assembly add instructions), so epi stays the default.
pub fn bfp_store_fused() -> bool {
    std::env::var("HASTY_FFT_BFP_STORE").is_ok_and(|v| v == "fused")
}

/// HASTY_FFT_BFP_STORE = pack: as epi, but the per-run phase packs its quantised run into whole words in registers
/// (`bfp_run_pack`) and writes the words into the entry's own shared row (after the run group's max shuffle, every
/// run of the block has been read), so the coalesced phase is a plain copy of words, without the per-word assembly
/// of `bfp_block_word_q`. Bit-identical output.
pub fn bfp_store_pack() -> bool {
    std::env::var("HASTY_FFT_BFP_STORE").is_ok_and(|v| v == "pack")
}

/// BFP load path of the axis passes: env HASTY_FFT_BFP_LOAD = run (default: each thread loads its run's words straight
/// from global memory; L1 merges the scattered 4-byte loads, DRAM sees whole sectors) | vec (the block's mantissa words
/// are first copied into the first shared buffer with coalesced vector loads, one barrier, then every thread unpacks its
/// run from shared memory; passes whose stages run in one shared buffer keep the scattered loads). Bit-identical
/// output. Measured (RTX 4090, 320³ grid, 24 channels): vec 4 % slower on the z pass (the extra barrier and shared
/// round trip of a one-block-per-SM pass cost more than the L1 wavefronts it saves), research/bfp_storage.md §10.
pub fn bfp_load_vec() -> bool {
    std::env::var("HASTY_FFT_BFP_LOAD").is_ok_and(|v| v == "vec")
}

#[derive(Clone, Debug)]
pub struct AxisCfg {
    pub n: usize,
    pub radices: Vec<usize>,
    pub contiguous: bool,
    /// lines per block (contiguous) or tile width in lines (strided)
    pub lpb: usize,
    /// threads per line
    pub tpl: usize,
    pub ctype: &'static str,
    /// direction of the (first) transform; with `conv` the second one has the opposite direction
    pub forward: bool,
    pub scale: f64,
    pub n_in: usize,
    pub n_out: usize,
    /// also read the last `n_in_tail` entries of every line (wrapped zero padding: data at [0, n_in) ∪ [n − n_in_tail, n))
    pub n_in_tail: usize,
    /// also write the last `n_out_tail` entries (wrapped crop)
    pub n_out_tail: usize,
    /// forward -> multiply by `kern` -> inverse inside one kernel
    pub conv: bool,
    pub pre: Vec<DiagOp>,
    pub post: Vec<DiagOp>,
    pub accumulate: bool,
    pub kfmt: KernelFormat,
    /// real polynomial diagonals applied after `pre` arrays on load / after `post` arrays on store (X-axis passes of a rank-3 pipeline)
    pub poly_pre: bool,
    pub poly_post: bool,
    /// voxel grid `[Nx, Ny, Nz]` the polynomial coordinates refer to (x = position along the pass axis, y = `la`, z = `lb`)
    pub vox: [usize; 3],
    /// strided passes only: a runtime table `arange[2 lb] = head, arange[2 lb + 1] = tail` per line group `lb` restricts
    /// the `a` index of that group to `[0, head) ∪ [la_cnt − tail, la_cnt)` (k-support pruning: the lines of a row that
    /// meet no sample are neither read nor written). Adds a trailing `const int* arange` kernel parameter.
    pub arange: bool,
    /// `arange` passes only: a runtime table `lrange[2 q], lrange[2 q + 1]` = (head, tail) per line, q = lb · lr_stride +
    /// la / lr_div, restricts the entries read (`lrange_in`) or written (`lrange_out`) on that line to
    /// `[0, head) ∪ [N − tail, N)`, within the static N_IN / N_OUT ranges (k-support pruning inside a line: the chord of a
    /// ball column). Adds trailing `const int* lrange, long lr_div, long lr_stride` parameters.
    pub lrange_in: bool,
    pub lrange_out: bool,
    /// fused input generation / output reduction ([`IoHook`]); None: plain loads and stores
    pub hook: Option<IoHook>,
    /// read the input from a block-floating-point buffer ([`BfpIo`]; strided passes of 16-line tiles and at least two
    /// stages, no `pre` diagonals): a prologue decodes the tile's blocks into the second shared buffer, the first stage
    /// reads them there. Adds trailing `const unsigned* bin_m, const unsigned char* bin_e, long bin_sn, long bin_lb`.
    pub bfp_in: Option<BfpIo>,
    /// write the output to a block-floating-point buffer (no `post` diagonals, no accumulation): the last stage writes
    /// to shared memory and an epilogue packs whole 16-line blocks (per-thread runs, exponent by a group max). Lines
    /// outside the pass's line range are stored as zeros. With `lrange_out` a block is written when any line of the
    /// tile writes that entry. Lines whose channel `la mod bch_mod` is at or past `bch_act` (runtime; e.g. the unused
    /// channels of a short coil batch, which hold stale data) count as outside the range. Adds trailing
    /// `unsigned* bout_m, unsigned char* bout_e, long bout_sn, long bout_lb, long bch_mod, long bch_act`.
    pub bfp_out: Option<BfpIo>,
    /// BFP passes: no padding complex per shared-memory entry (set by the planner when the padded buffers do not fit
    /// the device's shared memory; the codec's shared-memory runs then see more bank conflicts, results are identical)
    pub bfp_nopad: bool,
}

impl AxisCfg {
    pub fn simple(
        n: usize,
        radices: Vec<usize>,
        contiguous: bool,
        lpb: usize,
        tpl: usize,
        ctype: &'static str,
        forward: bool,
        scale: f64,
    ) -> Self {
        Self {
            n,
            radices,
            contiguous,
            lpb,
            tpl,
            ctype,
            forward,
            scale,
            n_in: n,
            n_out: n,
            n_in_tail: 0,
            n_out_tail: 0,
            conv: false,
            pre: vec![],
            post: vec![],
            accumulate: false,
            kfmt: KernelFormat::Complex,
            poly_pre: false,
            poly_post: false,
            vox: [1, 1, 1],
            arange: false,
            lrange_in: false,
            lrange_out: false,
            hook: None,
            bfp_in: None,
            bfp_out: None,
            bfp_nopad: false,
        }
    }
    /// whether the pass reads or writes a BFP buffer
    pub fn bfp(&self) -> bool {
        self.bfp_in.is_some() || self.bfp_out.is_some()
    }
    /// shared-memory row stride (complex values per entry) of the stage buffers: LPB, or LPB + 1 with BFP I/O (the
    /// per-thread runs of the codec then read and write shared memory with at most 2-way bank conflicts)
    pub fn smem_stride(&self) -> usize {
        // the fused BFP store (bfp_store_fused) reads shared memory per word of whole blocks: no padding wanted
        let pad = self.bfp_in.is_some() || (self.bfp_out.is_some() && !bfp_store_fused());
        self.lpb + (pad && !self.bfp_nopad) as usize
    }
    pub fn block_threads(&self) -> usize {
        self.lpb * self.tpl
    }
    pub fn passes(&self) -> usize {
        self.radices.len() * if self.conv { 2 } else { 1 }
    }
    /// All radices divide the largest one: every thread then owns exactly `rmax` values per pass, so passes can run in place
    /// in a single shared buffer (read all -> barrier -> write all -> barrier).
    pub fn inplace(&self) -> bool {
        let rmax = *self.radices.iter().max().unwrap();
        self.radices.iter().all(|r| rmax % r == 0) && self.tpl * rmax == self.n
    }
    /// bytes of dynamic shared memory (only needed when more than one pass)
    pub fn smem_bytes(&self, real_bytes: usize) -> usize {
        if self.passes() > 1 {
            (if self.inplace() { 1 } else { 2 }) * self.n * self.smem_stride() * 2 * real_bytes
        } else {
            0
        }
    }
}

pub(crate) fn poly_expr(table: &str, nterms: &str, x: &str, y: &str, z: &str, vox: [usize; 3]) -> String {
    format!(
        "poly_eval_({table}, {nterms}, ((real)({x}) - (real){:.17e}) * (real){:.17e}, ((real)({y}) - (real){:.17e}) * (real){:.17e}, ((real)({z}) - (real){:.17e}) * (real){:.17e})",
        (vox[0] as f64 - 1.0) / 2.0,
        2.0 / vox[0] as f64,
        (vox[1] as f64 - 1.0) / 2.0,
        2.0 / vox[1] as f64,
        (vox[2] as f64 - 1.0) / 2.0,
        2.0 / vox[2] as f64
    )
}

struct Pass {
    r: usize,
    ns: usize,
    sign: i32,
    /// last pass of the forward half of a `conv` kernel: multiply by the kernel array before storing
    fuse_kernel: bool,
}

pub(crate) fn sfx(sign: i32) -> &'static str {
    if sign < 0 {
        "m"
    } else {
        "p"
    }
}

/// Type definitions and complex/polynomial/half-precision helpers shared by all generated axis kernels.
pub(crate) fn prelude(ctype: &str) -> String {
    let mut s = String::new();
    writeln!(s, "typedef {ctype} real;").unwrap();
    writeln!(s, "struct cx {{ real x, y; }};").unwrap();
    writeln!(s, "#define RL(v) ((real)(v))").unwrap();
    writeln!(
        s,
        r#"__device__ __forceinline__ cx make_cx(real x, real y) {{ cx r; r.x = x; r.y = y; return r; }}
__device__ __forceinline__ cx cadd(cx a, cx b) {{ return make_cx(a.x + b.x, a.y + b.y); }}
__device__ __forceinline__ cx csub(cx a, cx b) {{ return make_cx(a.x - b.x, a.y - b.y); }}
__device__ __forceinline__ cx cscale(cx a, real c) {{ return make_cx(a.x * c, a.y * c); }}
__device__ __forceinline__ cx cmul(cx a, cx b) {{ return make_cx(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }}
__device__ __forceinline__ cx cmulc(cx a, cx b) {{ return make_cx(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y); }}
__device__ __forceinline__ real ipow_(real x, int p) {{ real r = (real)1; for (int i = 0; i < p; ++i) r *= x; return r; }}
__device__ __forceinline__ real poly_eval_(const real* t, int n, real cx_, real cy_, real cz_) {{ real acc = (real)0; for (int i = 0; i < n; ++i) acc += t[4 * i] * ipow_(cx_, (int)t[4 * i + 1]) * ipow_(cy_, (int)t[4 * i + 2]) * ipow_(cz_, (int)t[4 * i + 3]); return acc; }}
__device__ __forceinline__ real h2r(unsigned int h) {{ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)h)); return (real)f; }}
__device__ __forceinline__ real b2r(unsigned int h) {{ return (real)__uint_as_float(h << 16); }}"#
    )
    .unwrap();
    s
}

/// Parameter type of the k-space multiplier `kern` for a kernel format.
pub(crate) fn kern_type(kfmt: KernelFormat) -> &'static str {
    match kfmt {
        KernelFormat::Complex => "const cx*",
        KernelFormat::Real => "const real*",
        KernelFormat::ComplexHalf | KernelFormat::ComplexBf16 => "const unsigned int*",
    }
}

/// Expression for `kern` at element address `GIN(e)` scaled by `kscale`, multiplied into `v` (fused convolution).
pub(crate) fn kern_mul(kfmt: KernelFormat, v: &str) -> String {
    match kfmt {
        KernelFormat::Complex => format!("cmul({v}, cscale(kern[GIN(e)], kscale))"),
        KernelFormat::Real => format!("cscale({v}, kern[GIN(e)] * kscale)"),
        KernelFormat::ComplexHalf => format!("cmul({v}, make_cx(h2r(kern[GIN(e)] & 0xffffu) * kscale, h2r(kern[GIN(e)] >> 16) * kscale))"),
        KernelFormat::ComplexBf16 => format!("cmul({v}, make_cx(b2r(kern[GIN(e)] & 0xffffu) * kscale, b2r(kern[GIN(e)] >> 16) * kscale))"),
    }
}

pub fn generate(cfg: &AxisCfg) -> String {
    let n = cfg.n;
    let sign1 = if cfg.forward { -1 } else { 1 };
    let mut passes = Vec::new();
    for half in 0..(if cfg.conv { 2 } else { 1 }) {
        let sign = if half == 0 { sign1 } else { -sign1 };
        let mut ns = 1usize;
        for (i, &r) in cfg.radices.iter().enumerate() {
            passes.push(Pass {
                r,
                ns,
                sign,
                fuse_kernel: cfg.conv && half == 0 && i == cfg.radices.len() - 1,
            });
            ns *= r;
        }
    }
    let p = passes.len();
    let mut s = prelude(cfg.ctype);
    let (bin, bout) = (cfg.bfp_in, cfg.bfp_out);
    if cfg.bfp() {
        assert!(
            !cfg.contiguous && !cfg.conv && cfg.hook.is_none() && p > 1 && cfg.lpb == 16 && cfg.ctype == "float",
            "BFP I/O: strided f32 passes of 16-line tiles and at least two stages, no conv / hooks"
        );
        assert!(bin.is_none() || (cfg.pre.is_empty() && !cfg.poly_pre), "BFP input: no pre diagonals");
        assert!(bout.is_none() || (cfg.post.is_empty() && !cfg.poly_post && !cfg.accumulate), "BFP output: no post diagonals / accumulation");
        assert!([bin, bout].iter().flatten().all(|b| (8..=24).contains(&b.bits)), "BFP mantissa bits in 8 ..= 24");
        // the axis kernels compile with --use_fast_math (flush to zero): blocks whose values are below 2^-126 quantise
        // and decode with flushed subnormals (an absolute error below 2^-126; the CPU codec keeps the subnormals)
        s.push_str(hasty_bfp::BFP_CUH);
        s.push('\n');
    }
    let kty = match cfg.kfmt {
        KernelFormat::Complex => "const cx*",
        KernelFormat::Real => "const real*",
        KernelFormat::ComplexHalf | KernelFormat::ComplexBf16 => "const unsigned int*",
    };
    let mut done: Vec<(usize, i32)> = vec![];
    for ps in &passes {
        if !done.contains(&(ps.r, ps.sign)) {
            s.push_str(&emit_butterfly(ps.r, ps.sign));
            done.push((ps.r, ps.sign));
        }
    }
    writeln!(
        s,
        "#define N {n}\n#define N_IN {}\n#define N_OUT {}\n#define N_IN_TAIL {}\n#define N_OUT_TAIL {}\n#define LPB {}\n#define TPL {}",
        cfg.n_in, cfg.n_out, cfg.n_in_tail, cfg.n_out_tail, cfg.lpb, cfg.tpl
    )
    .unwrap();
    if cfg.bfp() {
        // stage-buffer row stride (one padding complex per entry: the codec's per-thread runs then hit at most 2-way
        // bank conflicts), and per BFP side: bits, run values / words, runs per block, compact entry index, static
        // entries
        writeln!(s, "#define LPS {}", cfg.smem_stride()).unwrap();
        for (tag, io, ns) in [("BI", bin, "(N_IN + N_IN_TAIL)"), ("BO", bout, "(N_OUT + N_OUT_TAIL)")] {
            if let Some(b) = io {
                writeln!(
                    s,
                    "#define {tag}_B {}\n#define {tag}_V {}\n#define {tag}_W {}\n#define {tag}_R {}\n#define {tag}_C {}\n#define {tag}_NS {ns}",
                    b.bits,
                    b.run_values(),
                    b.run_words(),
                    32 / b.run_values(),
                    b.compact as u8
                )
                .unwrap();
                // vector width (words) of the coalesced block pieces (store side: every lane writes BO_VW
                // consecutive words of one block; a block's B words are contiguous, consecutive lanes take consecutive
                // pieces, so a warp's store covers whole sectors instead of BO_W scattered words per lane)
                let vw = if b.bits % 4 == 0 { 4 } else if b.bits % 2 == 0 { 2 } else { 1 };
                writeln!(s, "#define {tag}_VW {vw}\n#define {tag}_PW ({tag}_B / {tag}_VW)").unwrap();
                if tag == "BO" {
                    writeln!(s, "#define BO_PACK {}", bfp_store_pack() as u8).unwrap();
                }
                if tag == "BI" {
                    // vector-load path: the block's words staged in shared memory at an odd stride (bank spread of
                    // the per-thread run reads)
                    writeln!(s, "#define BI_RS {}", b.bits | 1).unwrap();
                }
                // the block's mantissa words: the separate array, or (in place) the f32 cells of the tile's lines
                let m = match (tag, b.inplace) {
                    ("BI", false) => "(bin_m + blk * BI_B)".to_string(),
                    ("BI", true) => "((const unsigned*)(in + (LT0 * in_la + lb1 * in_lb + lb2 * in_lb2 + (long)e * in_sn)))".to_string(),
                    (_, false) => "(bout_m + blk * BO_B)".to_string(),
                    (_, true) => "((unsigned*)(out + (LT0 * out_la + lb1 * out_lb + lb2 * out_lb2 + (long)e * out_sn)))".to_string(),
                };
                writeln!(s, "#define {tag}_MANT {m}").unwrap();
            }
        }
    }
    let sidx_stride = if cfg.bfp() { "LPS" } else { "LPB" };
    let hook = cfg.hook.clone().unwrap_or_default();
    let (hload, hepi) = (hook.load.as_deref(), hook.epilogue.as_deref());
    assert!(cfg.hook.is_none() || (!cfg.contiguous && !cfg.arange && !cfg.conv && p > 1), "IoHook: strided, non-arange, non-conv passes of more than one stage");
    if cfg.hook.is_some() {
        // for the hook code: whether the stages run in one shared buffer (no spare buffer for staging)
        writeln!(s, "#define FFT_INPLACE {}", (cfg.inplace() && p > 1) as u8).unwrap();
    }
    s.push_str(&hook.prelude);
    s.push('\n');
    writeln!(
        s,
        "extern \"C\" __global__ void __launch_bounds__({}) fft_axis(const cx* __restrict__ in, cx* out, const cx* __restrict__ tw_m, const cx* __restrict__ tw_p, {kty} __restrict__ kern, real kscale,\n  const cx* __restrict__ p0, const cx* __restrict__ p1, const cx* __restrict__ p2, const cx* __restrict__ p3,\n  const cx* __restrict__ q0, const cx* __restrict__ q1, const cx* __restrict__ q2, const cx* __restrict__ q3,\n  const real* __restrict__ ppre, int nppre, const real* __restrict__ ppost, int nppost,\n  long la_cnt, long lb_cnt, long in_la, long in_lb, long in_sn, long out_la, long out_lb, long out_sn,\n  long lb1_cnt, long lb1_hi, long lb1_gap, long lb2_hi, long lb2_gap, long in_lb2, long out_lb2{}) {{",
        cfg.block_threads(),
        if cfg.arange { ", const int* __restrict__ arange" } else { "" }.to_string()
            + if cfg.lrange_in || cfg.lrange_out { ", const int* __restrict__ lrange, long lr_div, long lr_stride" } else { "" }
            + if bin.is_some() { ", const unsigned* __restrict__ bin_m, const unsigned char* __restrict__ bin_e, long bin_sn, long bin_lb" } else { "" }
            + if bout.is_some() { ", unsigned* __restrict__ bout_m, unsigned char* __restrict__ bout_e, long bout_sn, long bout_lb, long bch_mod, long bch_act" } else { "" }
            + &hook.params
    )
    .unwrap();
    writeln!(s, "  extern __shared__ cx smem[];").unwrap();
    if cfg.contiguous {
        writeln!(
            s,
            r#"  const int tid = threadIdx.x;
  const int li = tid / TPL, tl = tid % TPL;
  const long line = (long)blockIdx.x * LPB + li;
  const bool act = line < la_cnt * lb_cnt;
  const long la = line % la_cnt, lb = line / la_cnt;
  long lb1 = lb % lb1_cnt, lb2 = lb / lb1_cnt;
  if (lb1 >= lb1_hi) lb1 += lb1_gap;
  if (lb2 >= lb2_hi) lb2 += lb2_gap;
  const long ib = la * in_la + lb1 * in_lb + lb2 * in_lb2, ob_ = la * out_la + lb1 * out_lb + lb2 * out_lb2;
#define SIDX(n) (li * N + (n))"#
        )
        .unwrap();
    } else if cfg.arange && cfg.bfp() {
        // k-support pruning with BFP I/O: tiles of 16 lines aligned in the flat line index (one BFP block per entry),
        // covering the row's range [0, h) ∪ [la_cnt − t, la_cnt) rounded out to whole tiles; lines of a tile outside
        // the range (and past la_cnt) are inactive: they read nothing and store zeros into BFP blocks
        writeln!(
            s,
            r#"  const int tid = threadIdx.x;
  const int li = tid % LPB, tl = tid / LPB;
  const long lb = (long)blockIdx.z * gridDim.y + blockIdx.y;   // line groups fold into (y, z): > 65535 groups
  if (lb >= lb_cnt) return;
  const long ar_h = arange[2 * lb], ar_t = arange[2 * lb + 1];
  const long nt_ = (la_cnt + LPB - 1) / LPB, th_ = (ar_h + LPB - 1) / LPB;
  long ts_ = ar_t > 0 ? (la_cnt - ar_t) / LPB : nt_;
  if (ts_ < th_) ts_ = th_;
  if ((long)blockIdx.x >= th_ + (nt_ - ts_)) return;
  const long LT0 = ((long)blockIdx.x < th_ ? (long)blockIdx.x : ts_ + ((long)blockIdx.x - th_)) * LPB;
  const long la = LT0 + li;
  const bool act = la < la_cnt && (la < ar_h || la >= la_cnt - ar_t);
  long lb1 = lb % lb1_cnt, lb2 = lb / lb1_cnt;
  if (lb1 >= lb1_hi) lb1 += lb1_gap;
  if (lb2 >= lb2_hi) lb2 += lb2_gap;
  const long ib = la * in_la + lb1 * in_lb + lb2 * in_lb2, ob_ = la * out_la + lb1 * out_lb + lb2 * out_lb2;"#
        )
        .unwrap();
        writeln!(s, "#define SIDX(n) ((n) * {sidx_stride} + li)").unwrap();
        if cfg.lrange_in || cfg.lrange_out {
            writeln!(s, "  int lr_h = 0, lr_t = 0;
  if (act) {{ const long q = lb * lr_stride + la / lr_div; lr_h = lrange[2 * q]; lr_t = lrange[2 * q + 1]; }}").unwrap();
        }
    } else if cfg.arange {
        // k-support pruning: the row's `a` range is [0, h) ∪ [la_cnt − t, la_cnt); blocks past it leave (block-uniform)
        writeln!(
            s,
            r#"  const int tid = threadIdx.x;
  const int li = tid % LPB, tl = tid / LPB;
  const long lb = (long)blockIdx.z * gridDim.y + blockIdx.y;   // line groups fold into (y, z): > 65535 groups
  if (lb >= lb_cnt) return;
  const long ar_h = arange[2 * lb], ar_c = ar_h + arange[2 * lb + 1];
  if ((long)blockIdx.x * LPB >= ar_c) return;
  long la = (long)blockIdx.x * LPB + li;
  const bool act = la < ar_c;
  if (la >= ar_h) la += la_cnt - ar_c;
  long lb1 = lb % lb1_cnt, lb2 = lb / lb1_cnt;
  if (lb1 >= lb1_hi) lb1 += lb1_gap;
  if (lb2 >= lb2_hi) lb2 += lb2_gap;
  const long ib = la * in_la + lb1 * in_lb + lb2 * in_lb2, ob_ = la * out_la + lb1 * out_lb + lb2 * out_lb2;"#
        )
        .unwrap();
        writeln!(s, "#define SIDX(n) ((n) * {sidx_stride} + li)").unwrap();
        if cfg.lrange_in || cfg.lrange_out {
            writeln!(s, "  int lr_h = 0, lr_t = 0;
  if (act) {{ const long q = lb * lr_stride + la / lr_div; lr_h = lrange[2 * q]; lr_t = lrange[2 * q + 1]; }}").unwrap();
        }
    } else {
        writeln!(
            s,
            r#"  const int tid = threadIdx.x;
  const int li = tid % LPB, tl = tid / LPB;
  const long la = (long)blockIdx.x * LPB + li;
  const long lb = (long)blockIdx.z * gridDim.y + blockIdx.y;   // line groups fold into (y, z): > 65535 groups
  if (lb >= lb_cnt) return;   // block-uniform (padding blocks of the folded grid)
  const bool act = la < la_cnt;
  long lb1 = lb % lb1_cnt, lb2 = lb / lb1_cnt;
  if (lb1 >= lb1_hi) lb1 += lb1_gap;
  if (lb2 >= lb2_hi) lb2 += lb2_gap;
  const long ib = la * in_la + lb1 * in_lb + lb2 * in_lb2, ob_ = la * out_la + lb1 * out_lb + lb2 * out_lb2;"#
        )
        .unwrap();
        writeln!(s, "#define SIDX(n) ((n) * {sidx_stride} + li)").unwrap();
        if cfg.bfp() {
            writeln!(s, "  const long LT0 = (long)blockIdx.x * LPB;").unwrap();
        }
    }
    writeln!(
        s,
        "#define GIN(n) (ib + (long)(n) * in_sn)\n#define GOUT(n) (ob_ + (long)(n) * out_sn)"
    )
    .unwrap();
    let inplace = cfg.inplace() && p > 1;
    if p > 1 {
        if inplace {
            writeln!(s, "  cx* bufs[2] = {{ smem, smem }};").unwrap();
        } else {
            writeln!(s, "  cx* bufs[2] = {{ smem, smem + N * {sidx_stride} }};").unwrap();
        }
    }
    if cfg.hook.is_some() {
        writeln!(s, "  const long LA0 = (long)blockIdx.x * LPB;\n  (void)LA0;").unwrap();
    }
    if let Some(pro) = hook.prologue.as_deref() {
        // the second buffer is free until stage 1 writes it (not with FFT_INPLACE: the hook must not use `stage` then)
        writeln!(s, "  cx* const stage = bufs[1];\n  (void)stage;\n  {{\n{pro}\n  }}\n  __syncthreads();").unwrap();
    }
    // the tile's union of per-line entry ranges [0, uh) ∪ [N − ut, N) (a half warp holds the 16 lines of the tile:
    // tid = tl · 16 + li), the static range when the pass has no per-line ranges
    let tile_union = |s: &mut String, on: bool| {
        if on {
            writeln!(
                s,
                "    int uh = act ? lr_h : 0, ut = act ? lr_t : 0;\n    {{ const unsigned hm_ = 0xffffu << (threadIdx.x & 16);\n      #pragma unroll\n      for (int o = 8; o > 0; o >>= 1) {{ uh = max(uh, __shfl_xor_sync(hm_, uh, o)); ut = max(ut, __shfl_xor_sync(hm_, ut, o)); }} }}"
            )
            .unwrap();
        } else {
            writeln!(s, "    const int uh = N, ut = 0;").unwrap();
        }
    };
    let (store_fused, load_vec) = (bfp_store_fused(), bfp_load_vec() && !inplace);
    let store_pack = bfp_store_pack() && !store_fused;
    if bin.is_some() {
        // BFP input: decode the tile's blocks (entries of the static input range that some line of the tile reads)
        // into the free shared buffer, per-thread runs of BI_V values; unwritten blocks (exponent byte 0) are zeros
        writeln!(s, "  cx* const stage_b = bufs[1];\n  {{").unwrap();
        tile_union(&mut s, cfg.lrange_in);
        if load_vec {
            // phase A: the blocks' mantissa words into the first shared buffer (free until stage 1 writes it) as
            // coalesced vector pieces, BI_PW per block (consecutive lanes, consecutive pieces: whole sectors per warp
            // load); phase B below unpacks the runs from there
            writeln!(
                s,
                r#"    unsigned* const raw_ = (unsigned*)bufs[0];
    for (int t = tid; t < BI_NS * BI_PW; t += LPB * TPL) {{
      const int s_ = t / BI_PW, pc_ = t % BI_PW;
      const int e = s_ < N_IN ? s_ : N - N_IN_TAIL + (s_ - N_IN);
      if (!(e < uh || e >= N - ut)) continue;
      const long blk = lb * bin_lb + (long)(BI_C ? s_ : e) * bin_sn + LT0 / LPB;
      if (bin_e[blk] == 0) continue;
      const unsigned* src_ = BI_MANT + pc_ * BI_VW;
      unsigned* d_ = raw_ + s_ * BI_RS + pc_ * BI_VW;
#if BI_VW == 4
      if ((((unsigned long long)src_) & 15ull) == 0ull) {{ const uint4 v_ = *(const uint4*)src_; d_[0] = v_.x; d_[1] = v_.y; d_[2] = v_.z; d_[3] = v_.w; continue; }}
#elif BI_VW == 2
      if ((((unsigned long long)src_) & 7ull) == 0ull) {{ const uint2 v_ = *(const uint2*)src_; d_[0] = v_.x; d_[1] = v_.y; continue; }}
#endif
      #pragma unroll
      for (int k = 0; k < BI_VW; ++k) d_[k] = src_[k];
    }}
    __syncthreads();"#
            )
            .unwrap();
        }
        let wsrc = if load_vec { "raw_[s_ * BI_RS + j_ * BI_W + k]" } else { "BI_MANT[j_ * BI_W + k]" };
        writeln!(
            s,
            r#"    for (int r = tid; r < BI_NS * BI_R; r += LPB * TPL) {{
      const int s_ = r / BI_R, j_ = r % BI_R;
      const int e = s_ < N_IN ? s_ : N - N_IN_TAIL + (s_ - N_IN);
      if (!(e < uh || e >= N - ut)) continue;
      const long blk = lb * bin_lb + (long)(BI_C ? s_ : e) * bin_sn + LT0 / LPB;
      const int es = bin_e[blk];
      float* d_ = (float*)(stage_b + e * LPS) + j_ * BI_V;
      if (es == 0) {{
        #pragma unroll
        for (int i = 0; i < BI_V; ++i) d_[i] = 0.f;
      }} else {{
        unsigned w_[BI_W];
        #pragma unroll
        for (int k = 0; k < BI_W; ++k) w_[k] = {wsrc};
        int q_[BI_V];
        bfp_run_unpack<BI_B, BI_V>(w_, q_);
        #pragma unroll
        for (int i = 0; i < BI_V; ++i) d_[i] = bfp_dequant<BI_B>(q_[i], es);
      }}
    }}
  }}
  __syncthreads();"#
        )
        .unwrap();
    }
    if bout.is_some() {
        // the tile's output entry union, before the stages (the fused store writes the exponent bytes in the last
        // stage; the discard drops the written entries' cell runs after the first stage)
        writeln!(s, "  int uo_h, uo_t;\n  {{").unwrap();
        tile_union(&mut s, cfg.lrange_out);
        writeln!(s, "    uo_h = uh; uo_t = ut;\n  }}\n  (void)uo_h; (void)uo_t;\n  const unsigned hm_o = 0xffffu << (threadIdx.x & 16);\n  (void)hm_o;").unwrap();
    }
    let discard = bout.is_some_and(|b| b.inplace && b.discard);
    let rmax = *cfg.radices.iter().max().unwrap();
    let pre_ptr = ["p0", "p1", "p2", "p3"];
    let post_ptr = ["q0", "q1", "q2", "q3"];
    for (pi, ps) in passes.iter().enumerate() {
        let (r, ns) = (ps.r, ps.ns);
        let m = n / r;
        let first = pi == 0;
        let last = pi == p - 1;
        let sf = sfx(ps.sign);
        let twp = if ps.sign < 0 { "tw_m" } else { "tw_p" };
        let mult = if inplace { rmax / r } else { 1 };
        writeln!(
            s,
            "  // pass {pi}: radix {r}, Ns = {ns}, sign {}{}{}",
            ps.sign,
            if inplace { ", in place" } else { "" },
            if ps.fuse_kernel {
                ", fused kernel multiply"
            } else {
                ""
            }
        )
        .unwrap();
        writeln!(s, "  {{").unwrap();
        if !first {
            writeln!(s, "    const cx* src = bufs[{}];", (pi - 1) % 2).unwrap();
        }
        if !last {
            writeln!(s, "    cx* dst = bufs[{}];", pi % 2).unwrap();
        } else if hepi.is_some() || bout.is_some() {
            writeln!(s, "    cx* fin_ = bufs[{}];", pi % 2).unwrap();
        }
        let load = |s: &mut String, jj: &str, vv: &str| {
            writeln!(
                s,
                "      #pragma unroll\n      for (int t = 0; t < {r}; ++t) {{"
            )
            .unwrap();
            if first {
                writeln!(s, "        const int e = {jj} + t * {m};").unwrap();
                let lr = if cfg.lrange_in { " && (e < lr_h || e >= N - lr_t)" } else { "" };
                let src = if bin.is_some() { "stage_b[SIDX(e)]".to_string() } else { hload.map_or_else(|| "in[GIN(e)]".to_string(), |h| format!("({h})")) };
                if cfg.n_in_tail > 0 {
                    writeln!(s, "        cx x = make_cx(RL(0), RL(0));\n        if (act && (e < N_IN || e >= N - N_IN_TAIL){lr}) {{ x = {src};").unwrap();
                } else if cfg.n_in < n {
                    writeln!(s, "        cx x = make_cx(RL(0), RL(0));\n        if (act && e < N_IN{lr}) {{ x = {src};").unwrap();
                } else {
                    writeln!(s, "        cx x = make_cx(RL(0), RL(0));\n        if (act{lr}) {{ x = {src};").unwrap();
                }
                for (k, op) in cfg.pre.iter().enumerate() {
                    let f = if *op == DiagOp::Mult { "cmul" } else { "cmulc" };
                    writeln!(s, "          x = {f}(x, {}[GIN(e)]);", pre_ptr[k]).unwrap();
                }
                if cfg.poly_pre {
                    writeln!(
                        s,
                        "          x = cscale(x, {});",
                        poly_expr("ppre", "nppre", "e", "la", "lb", cfg.vox)
                    )
                    .unwrap();
                }
                writeln!(s, "        }}\n        {vv}[t] = x;").unwrap();
            } else {
                writeln!(s, "        {vv}[t] = src[SIDX({jj} + t * {m})];").unwrap();
            }
            writeln!(s, "      }}").unwrap();
        };
        let compute_store = |s: &mut String, jj: &str, vv: &str| {
            if ns > 1 {
                let tws = n / (ns * r);
                writeln!(s, "      {{ const int k = {jj} % {ns};").unwrap();
                writeln!(s, "      #pragma unroll\n      for (int t = 1; t < {r}; ++t) {vv}[t] = cmul({vv}[t], {twp}[t * k * {tws}]);").unwrap();
                writeln!(s, "      const int ob = ({jj} / {ns}) * {} + k;", ns * r).unwrap();
            } else {
                writeln!(s, "      {{ const int ob = {jj} * {r};").unwrap();
            }
            writeln!(s, "      bfly{r}{sf}({vv});").unwrap();
            writeln!(
                s,
                "      #pragma unroll\n      for (int u = 0; u < {r}; ++u) {{"
            )
            .unwrap();
            if last {
                writeln!(s, "        const int e = ob + u * {ns};").unwrap();
                let guard = if cfg.n_out_tail > 0 {
                    "act && (e < N_OUT || e >= N - N_OUT_TAIL)"
                } else if cfg.n_out < n {
                    "act && e < N_OUT"
                } else {
                    "act"
                };
                let lr = if cfg.lrange_out { " && (e < lr_h || e >= N - lr_t)" } else { "" };
                if bout.is_some() && store_fused {
                    // BFP output, fused: the half warp holds the block (its 16 lanes = the tile's 16 lines at entry
                    // e; inactive lines zeros): exponent by a half-warp max, values quantised into shared memory,
                    // exponent byte by lane 0; the mantissa words follow after the stages (coalesced pieces)
                    let g = if cfg.n_out_tail > 0 { "e < N_OUT || e >= N - N_OUT_TAIL" } else if cfg.n_out < n { "e < N_OUT" } else { "true" };
                    writeln!(s, "        if ({g}) {{ cx w = act ? {vv}[u] : make_cx(RL(0), RL(0));").unwrap();
                    if cfg.scale != 1.0 {
                        writeln!(s, "          w = cscale(w, RL({:.17e}));", cfg.scale).unwrap();
                    }
                    writeln!(
                        s,
                        r#"          unsigned m_ = max(bfp_abs_bits(w.x), bfp_abs_bits(w.y));
          #pragma unroll
          for (int o = 8; o > 0; o >>= 1) m_ = max(m_, __shfl_xor_sync(hm_o, m_, o));
          const int es_ = bfp_estore<BO_B>(m_);
          int* qd_ = (int*)(fin_ + SIDX(e));
          qd_[0] = es_ == BFP_ENAN ? 0 : bfp_quant<BO_B>(w.x, es_);
          qd_[1] = es_ == BFP_ENAN ? 0 : bfp_quant<BO_B>(w.y, es_);
          if (li == 0 && (e < uo_h || e >= N - uo_t)) bout_e[lb * bout_lb + (long)(BO_C ? (e < N_OUT ? e : N_OUT + e - (N - N_OUT_TAIL)) : e) * bout_sn + LT0 / LPB] = (unsigned char)es_; }}
      }} }}"#
                    )
                    .unwrap();
                    return;
                }
                if bout.is_some() {
                    // BFP output: every line of the tile writes the static range (inactive lines zeros); the
                    // epilogue packs whole blocks
                    let g = if cfg.n_out_tail > 0 { "e < N_OUT || e >= N - N_OUT_TAIL" } else if cfg.n_out < n { "e < N_OUT" } else { "true" };
                    writeln!(s, "        if ({g}) {{ cx w = {vv}[u];").unwrap();
                    if cfg.scale != 1.0 {
                        writeln!(s, "          w = cscale(w, RL({:.17e}));", cfg.scale).unwrap();
                    }
                    writeln!(s, "          fin_[SIDX(e)] = act ? w : make_cx(RL(0), RL(0)); }}\n      }} }}").unwrap();
                    return;
                }
                writeln!(s, "        if ({guard}{lr}) {{ cx w = {vv}[u];").unwrap();
                if cfg.scale != 1.0 {
                    writeln!(s, "          w = cscale(w, RL({:.17e}));", cfg.scale).unwrap();
                }
                for (k, op) in cfg.post.iter().enumerate() {
                    let f = if *op == DiagOp::Mult { "cmul" } else { "cmulc" };
                    writeln!(s, "          w = {f}(w, {}[GOUT(e)]);", post_ptr[k]).unwrap();
                }
                if cfg.poly_post {
                    writeln!(
                        s,
                        "          w = cscale(w, {});",
                        poly_expr("ppost", "nppost", "e", "la", "lb", cfg.vox)
                    )
                    .unwrap();
                }
                if hepi.is_some() {
                    writeln!(s, "          fin_[SIDX(e)] = w; }}").unwrap();
                } else if cfg.accumulate {
                    writeln!(s, "          out[GOUT(e)] = cadd(out[GOUT(e)], w); }}").unwrap();
                } else {
                    writeln!(s, "          out[GOUT(e)] = w; }}").unwrap();
                }
            } else if ps.fuse_kernel {
                writeln!(s, "        const int e = ob + u * {ns};").unwrap();
                let kexpr = match cfg.kfmt {
                    KernelFormat::Complex => "cscale(kern[GIN(e)], kscale)".to_string(),
                    KernelFormat::Real => "make_cx(kern[GIN(e)] * kscale, RL(0))".to_string(),
                    KernelFormat::ComplexHalf => "make_cx(h2r(kern[GIN(e)] & 0xffffu) * kscale, h2r(kern[GIN(e)] >> 16) * kscale)".to_string(),
                    KernelFormat::ComplexBf16 => "make_cx(b2r(kern[GIN(e)] & 0xffffu) * kscale, b2r(kern[GIN(e)] >> 16) * kscale)".to_string(),
                };
                let kmul = if cfg.kfmt == KernelFormat::Real {
                    format!("cscale({vv}[u], kern[GIN(e)] * kscale)")
                } else {
                    format!("cmul({vv}[u], {kexpr})")
                };
                writeln!(s, "        dst[SIDX(e)] = act ? {kmul} : {vv}[u];").unwrap();
            } else {
                writeln!(s, "        dst[SIDX(ob + u * {ns})] = {vv}[u];").unwrap();
            }
            writeln!(s, "      }} }}").unwrap();
        };
        if inplace {
            writeln!(s, "    cx vv[{mult}][{r}];").unwrap();
            writeln!(s, "    #pragma unroll\n    for (int mm = 0; mm < {mult}; ++mm) {{ const int jj = tl + mm * TPL;").unwrap();
            load(&mut s, "jj", "vv[mm]");
            writeln!(s, "    }}").unwrap();
            // (in place: the BFP input is staged in the one shared buffer the first stage then overwrites)
            if (!first && (!last || hepi.is_some() || bout.is_some())) || (first && bin.is_some()) {
                writeln!(s, "    __syncthreads();").unwrap();
            }
            writeln!(s, "    #pragma unroll\n    for (int mm = 0; mm < {mult}; ++mm) {{ const int jj = tl + mm * TPL;").unwrap();
            compute_store(&mut s, "jj", "vv[mm]");
            writeln!(s, "    }}").unwrap();
        } else {
            writeln!(
                s,
                "    for (int jj = tl; jj < {m}; jj += TPL) {{\n      cx v[{r}];"
            )
            .unwrap();
            load(&mut s, "jj", "v");
            compute_store(&mut s, "jj", "v");
            writeln!(s, "    }}").unwrap();
        }
        writeln!(s, "  }}").unwrap();
        if !last {
            writeln!(s, "  __syncthreads();").unwrap();
        }
        if first && !last && discard {
            // the tile has been read (stage 0 done): drop the cell runs of the entries this pass writes from L2
            // without write-back (only their first 4 · BO_B bytes are rewritten; see BfpIo::discard)
            writeln!(
                s,
                r#"#if __CUDA_ARCH__ >= 800
  for (int e = tid; e < N; e += LPB * TPL) {{
    if (!((e < N_OUT || e >= N - N_OUT_TAIL) && (e < uo_h || e >= N - uo_t))) continue;
    const unsigned long long a_ = (unsigned long long)(const void*)BO_MANT;
    if ((a_ & 127ull) == 0ull) asm volatile("discard.global.L2 [%0], 128;" :: "l"(a_) : "memory");
  }}
  __threadfence();
#endif"#
            )
            .unwrap();
        }
    }
    if let Some(epi) = hepi {
        writeln!(s, "  __syncthreads();\n  {{\n    const cx* fin = bufs[{}];\n    cx* const spare = bufs[{}];\n    (void)spare;\n{epi}\n  }}", (p - 1) % 2, p % 2).unwrap();
    }
    if bout.is_some() && store_fused {
        writeln!(s, "  __syncthreads();\n  {{\n    const cx* fin = bufs[{}];\n    const int uh = uo_h, ut = uo_t;", (p - 1) % 2).unwrap();
    }
    if bout.is_some() && !store_fused {
        // BFP output: per-thread runs of BO_V values (BO_R consecutive lanes per block, exponent by their group
        // max), whole words per run and one exponent byte per block; only entries some line of the tile writes
        writeln!(s, "  __syncthreads();\n  {{\n    const cx* fin = bufs[{}];", (p - 1) % 2).unwrap();
        tile_union(&mut s, cfg.lrange_out);
        writeln!(
            s,
            r#"    for (int r = tid; r < BO_NS * BO_R; r += LPB * TPL) {{
      const int s_ = r / BO_R, j_ = r % BO_R;
      const int e = s_ < N_OUT ? s_ : N - N_OUT_TAIL + (s_ - N_OUT);
      const float* x_ = (const float*)(fin + e * LPS) + j_ * BO_V;
      float v_[BO_V];
      unsigned m_ = 0u;
      #pragma unroll
      for (int i = 0; i < BO_V; ++i) {{ v_[i] = x_[i]; m_ = max(m_, bfp_abs_bits(v_[i])); }}
#if BO_R > 1
      {{ const unsigned gm_ = ((1u << BO_R) - 1u) << ((threadIdx.x & 31) & ~(BO_R - 1));
        #pragma unroll
        for (int o = BO_R / 2; o > 0; o >>= 1) m_ = max(m_, __shfl_xor_sync(gm_, m_, o)); }}
#endif
      const int es_ = bfp_estore<BO_B>(m_);
#if BO_PACK
      // packed: the run's words into the row's first BO_B words (all runs of the block were read before the max
      // shuffle above; one run per block when BO_R = 1)
      {{ int q_[BO_V];
        #pragma unroll
        for (int i = 0; i < BO_V; ++i) q_[i] = es_ == BFP_ENAN ? 0 : bfp_quant<BO_B>(v_[i], es_);
        unsigned w_[BO_W];
        bfp_run_pack<BO_B, BO_V>(q_, w_);
        unsigned* pw_ = (unsigned*)(fin + e * LPS) + j_ * BO_W;
        #pragma unroll
        for (int k = 0; k < BO_W; ++k) pw_[k] = w_[k]; }}
#else
      // the run's values quantised in place (the run's own shared slots: no other lane reads them in this phase)
      int* qw_ = (int*)(fin + e * LPS) + j_ * BO_V;
      #pragma unroll
      for (int i = 0; i < BO_V; ++i) qw_[i] = es_ == BFP_ENAN ? 0 : bfp_quant<BO_B>(v_[i], es_);
#endif
      if ((e < uh || e >= N - ut) && j_ == 0) {{
        const long blk = lb * bout_lb + (long)(BO_C ? s_ : e) * bout_sn + LT0 / LPB;
        bout_e[blk] = (unsigned char)es_;
      }}
    }}
    __syncthreads();"#
        )
        .unwrap();
    }
    if bout.is_some() && store_fused {
        // fused store, phase 2: one thread per mantissa word (consecutive lanes = consecutive words of a block: a warp
        // store covers whole sectors; the unpadded stage rows put a block's 32 values in 32 distinct banks)
        writeln!(
            s,
            r#"    for (int t = tid; t < BO_NS * BO_B; t += LPB * TPL) {{
      const int s_ = t / BO_B, k_ = t % BO_B;
      const int e = s_ < N_OUT ? s_ : N - N_OUT_TAIL + (s_ - N_OUT);
      if (!(e < uh || e >= N - ut)) continue;
      const long blk = lb * bout_lb + (long)(BO_C ? s_ : e) * bout_sn + LT0 / LPB;
      (void)blk;
      BO_MANT[k_] = bfp_block_word_q<BO_B>((const int*)(fin + e * LPS), k_);
    }}
  }}"#
        )
        .unwrap();
    }
    if bout.is_some() && !store_fused {
        writeln!(
            s,
            r#"    // phase 2 (after the barrier): the mantissas as coalesced vector pieces, BO_PW per block of BO_VW words, word k
    // assembled from the block's quantised values (bfp_block_word_q: the bfp_run_pack layout)
    for (int t = tid; t < BO_NS * BO_PW; t += LPB * TPL) {{
      const int s_ = t / BO_PW, pc_ = t % BO_PW;
      const int e = s_ < N_OUT ? s_ : N - N_OUT_TAIL + (s_ - N_OUT);
      if (!(e < uh || e >= N - ut)) continue;
      const long blk = lb * bout_lb + (long)(BO_C ? s_ : e) * bout_sn + LT0 / LPB;
      (void)blk;
      unsigned w_[BO_VW];
#if BO_PACK
      {{ const unsigned* pw_ = (const unsigned*)(fin + e * LPS) + pc_ * BO_VW;
        #pragma unroll
        for (int k = 0; k < BO_VW; ++k) w_[k] = pw_[k]; }}
#else
      const int* q_ = (const int*)(fin + e * LPS);
      #pragma unroll
      for (int k = 0; k < BO_VW; ++k) w_[k] = bfp_block_word_q<BO_B>(q_, pc_ * BO_VW + k);
#endif
      unsigned* d_ = BO_MANT + pc_ * BO_VW;
#if BO_VW == 4
      if ((((unsigned long long)d_) & 15ull) == 0ull) {{ *(uint4*)d_ = make_uint4(w_[0], w_[1], w_[2], w_[3]); continue; }}
#elif BO_VW == 2
      if ((((unsigned long long)d_) & 7ull) == 0ull) {{ *(uint2*)d_ = make_uint2(w_[0], w_[1]); continue; }}
#endif
      #pragma unroll
      for (int k = 0; k < BO_VW; ++k) d_[k] = w_[k];
    }}
  }}"#
        )
        .unwrap();
    }
    writeln!(s, "#undef GIN\n#undef GOUT\n#undef SIDX\n}}").unwrap();
    if bout.is_some() {
        // BFP output: lines whose channel (la mod bch_mod) is at or past bch_act (a short coil batch: stale channels)
        // are inactive, so they store zeros and cannot set a block's exponent
        let pat = "  const bool act = la < la_cnt";
        assert_eq!(s.matches(pat).count(), 1, "BFP output: exactly one act line to mask");
        let k = s.find(pat).unwrap();
        s.replace_range(k..k + "  const bool act = la < la_cnt".len(), "  const bool act = (la % bch_mod) < bch_act && la < la_cnt");
    }
    s
}
