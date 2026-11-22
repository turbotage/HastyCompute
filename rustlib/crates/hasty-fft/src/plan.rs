use std::sync::Arc;

use cudarc::driver::{
    sys::CUfunction_attribute_enum, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use hasty_cuda::{CompileSettings, Device, Real};

use crate::codegen::{generate, AxisCfg, BfpIo, DiagOp, IoHook, KernelFormat};
use crate::factor::factorize;

#[derive(Debug, thiserror::Error)]
pub enum FftError {
    #[error(transparent)]
    Cuda(#[from] hasty_cuda::Error),
    #[error("CUDA driver error: {0:?}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("unsupported FFT length {0}: needs prime factors in {{2,3,5,7,11,13}} (Bluestein not implemented)")]
    UnsupportedLength(usize),
    #[error("FFT length {n} needs {need} bytes of shared memory, device allows {have}")]
    SharedMemory { n: usize, need: usize, have: usize },
    #[error("buffer too small: need {need} elements, have {have}")]
    BufferSize { need: usize, have: usize },
    #[error("invalid argument: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Inverse,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Norm {
    None,
    /// divide by the total number of points (use on the inverse transform)
    Inverse,
    /// divide by sqrt(total number of points)
    Unitary,
}

/// Runtime line addressing of one axis pass. A line is `(a, b)`; element `n` is at `a*la + b*lb + n*sn`.
/// For a strided axis (`sn != 1`, `contiguous == false`) `a` must be the stride-1 dimension (it is the coalesced tile dimension).
#[derive(Clone, Copy, Debug, Default)]
pub struct LineAddr {
    pub la_cnt: usize,
    pub lb_cnt: usize,
    pub in_la: usize,
    pub in_lb: usize,
    pub in_sn: usize,
    pub out_la: usize,
    pub out_lb: usize,
    pub out_sn: usize,
}

/// Optional runtime split of the line index `b` into two sub-indices with gaps (pruned multi-dimensional passes: only
/// the lines whose other coordinates lie in a wrapped block are transformed). With `lb1_cnt > 0`:
/// `b1 = b % lb1_cnt`, `b2 = b / lb1_cnt`; `b1 += lb1_gap` if `b1 >= lb1_hi` (likewise `b2`); the line starts at
/// `a*la + b1*lb + b2*lb2` (input `in_lb`, `in_lb2`; output `out_lb`, `out_lb2`). `LineAddr::lb_cnt` is then the
/// number of selected lines `b`. All zero (default): no split.
#[derive(Clone, Copy, Debug, Default)]
pub struct LineSplit {
    pub lb1_cnt: usize,
    pub lb1_hi: usize,
    pub lb1_gap: usize,
    pub lb2_hi: usize,
    pub lb2_gap: usize,
    pub in_lb2: usize,
    pub out_lb2: usize,
}

#[derive(Clone, Debug)]
pub struct AxisSpec {
    pub n: usize,
    pub n_in: usize,
    pub n_out: usize,
    /// wrapped pruning: also read the last `n_in_tail` / write the last `n_out_tail` entries of each line (0 = none)
    pub n_in_tail: usize,
    pub n_out_tail: usize,
    pub dir: Direction,
    pub scale: f64,
    pub conv: bool,
    pub pre: Vec<DiagOp>,
    pub post: Vec<DiagOp>,
    pub accumulate: bool,
    pub contiguous: bool,
    pub addr: LineAddr,
    pub kfmt: KernelFormat,
    /// polynomial diagonals from a runtime term table (`LaunchPtrs::ppre/ppost`), see `PolyTerm`
    pub poly_pre: bool,
    pub poly_post: bool,
    pub vox: [usize; 3],
}

/// Raw device pointers for one launch (0 = unused).
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchPtrs {
    pub inp: u64,
    pub out: u64,
    pub kern: u64,
    /// multiplies the decoded kernel value (fused convolution passes only)
    pub kscale: f64,
    /// device tables of poly terms `[coef, px, py, pz]` in the transform precision (and term counts)
    pub ppre: u64,
    pub nppre: i32,
    pub ppost: u64,
    pub nppost: i32,
    pub pre: [u64; 4],
    pub post: [u64; 4],
    /// BFP input of a pass built with [`AxisPlan::from_spec_bfp`]: (mantissa words, exponent bytes), offset to the
    /// launch's first block
    pub bin: [u64; 2],
    /// BFP output: (mantissa words, exponent bytes)
    pub bout: [u64; 2],
    /// BFP output: lines with `la mod bch[0] ≥ bch[1]` are stored as zeros (unused channels); [0, 0]: every line
    pub bch: [i64; 2],
}

/// One extra kernel argument of a hooked pass ([`IoHook::params`], [`AxisPlan::launch_ext`]), in parameter order.
#[derive(Clone, Copy, Debug)]
pub enum KArg {
    /// a device pointer
    Ptr(u64),
    I32(i32),
    I64(i64),
    F32(f32),
}

/// Chirp and B̂ tables of a Bluestein pass (`bluestein.rs`).
struct BlueTables<R: Real> {
    ch: CudaSlice<R::Complex>,
    bh_f: CudaSlice<R::Complex>,
    bh_i: CudaSlice<R::Complex>,
}

/// One FFT pass over lines of a row-major array. See [`AxisSpec`] and `codegen.rs`. Lengths with a prime factor > 13
/// use a Bluestein (chirp-z) core of smooth length m ≥ 2n − 1 with the same load / store semantics.
pub struct AxisPlan<R: Real> {
    func: CudaFunction,
    tw_m: CudaSlice<R::Complex>,
    tw_p: CudaSlice<R::Complex>,
    blue: Option<BlueTables<R>>,
    pub cfg: AxisCfg,
    pub addr: LineAddr,
    grid: (u32, u32, u32),
    smem: u32,
    /// elements the plain (outer, n, inner) layout would occupy; used for buffer-size checks by the simple API
    simple_len: usize,
    /// line-index split ([`LineSplit`]; default none)
    pub split: LineSplit,
    /// per-row `a` ranges of a k-support-pruned pass ([`AxisPlan::from_spec_ranged`], [`AxisPlan::set_arange`])
    arange: Option<CudaSlice<i32>>,
    /// per-line entry ranges ([`AxisCfg::lrange_in`] / `lrange_out`, [`AxisPlan::set_lrange`]): table, div, stride
    lrange: Option<(CudaSlice<i32>, i64, i64)>,
    /// every line group has the same `a` range (set_arange): sub-range launches ([`AxisPlan::launch_lines`]) see the
    /// same table entries whatever their first group
    arange_uniform: bool,
    /// BFP block strides (input entry, input line group, output entry, output line group; [`crate::BfpIo`])
    pub bfp_strides: [i64; 4],
}

const DEFAULT_SMEM: usize = 48 * 1024;

fn choose_shape(
    n: usize,
    radices: &[usize],
    contiguous: bool,
    tile_cnt: usize,
    real_bytes: usize,
    smem_limit: usize,
) -> (usize, usize) {
    let rmax = *radices.iter().max().unwrap();
    let tpl = (n / rmax).max(1);
    let passes = radices.len();
    let inplace = radices.iter().all(|r| rmax % r == 0) && tpl * rmax == n;
    let per_line = if passes > 1 {
        (if inplace { 1 } else { 2 }) * n * 2 * real_bytes
    } else {
        0
    };
    let mut lpb = if contiguous { (256 / tpl).max(1) } else { 16 };
    if !contiguous {
        lpb = lpb.min(tile_cnt.next_power_of_two().max(1));
    }
    while per_line > 0 && lpb > 1 && lpb * per_line > smem_limit {
        lpb /= 2;
    }
    while lpb * tpl > 1024 && lpb > 1 {
        lpb /= 2;
    }
    (lpb, tpl)
}

/// grid y / z limit of a CUDA launch
pub const GRID_YZ_MAX: usize = 65535;
/// grid x limit of a CUDA launch
pub const GRID_X_MAX: usize = (1 << 31) - 1;

/// Launch grid of an axis pass with `la_cnt × lb_cnt` lines and `lpb` lines per block. Contiguous passes run all
/// lines over grid x. Strided passes tile `a` over grid x and run the line groups `b` over grid y, folded into
/// `(y, z)` when there are more than [`GRID_YZ_MAX`] of them (`b = z · gridDim.y + y`; the kernels return from the
/// `gy · gz − lb_cnt` padding blocks): a 1 mm torso fine grid has 480 · 480 x lines per channel.
pub fn launch_grid(contiguous: bool, la_cnt: usize, lb_cnt: usize, lpb: usize) -> Result<(u32, u32, u32), FftError> {
    let lpb = lpb.max(1);
    let gx = if contiguous { (la_cnt * lb_cnt).div_ceil(lpb) } else { la_cnt.div_ceil(lpb) }.max(1);
    if gx > GRID_X_MAX {
        return Err(FftError::Invalid(format!("axis pass: {gx} blocks exceed the grid x limit ({la_cnt} × {lb_cnt} lines, {lpb} per block)")));
    }
    if contiguous {
        return Ok((gx as u32, 1, 1));
    }
    let lb = lb_cnt.max(1);
    let gz = lb.div_ceil(GRID_YZ_MAX);
    let gy = lb.div_ceil(gz);
    if gz > GRID_YZ_MAX {
        return Err(FftError::Invalid(format!("axis pass: {lb} line groups exceed the grid y · z limit")));
    }
    Ok((gx as u32, gy as u32, gz as u32))
}

impl<R: Real> AxisPlan<R> {
    /// Launch grid of the pass (see [`launch_grid`]).
    pub fn grid(&self) -> (u32, u32, u32) {
        self.grid
    }

    /// Plain batched 1-D FFT of the axis in a `(outer, n, inner)` layout (`inner == 1` is the contiguous case).
    pub fn new(
        dev: &Device,
        n: usize,
        inner: usize,
        outer: usize,
        dir: Direction,
        scale: f64,
    ) -> Result<Self, FftError> {
        let (addr, contiguous) = if inner == 1 {
            (
                LineAddr {
                    la_cnt: outer,
                    lb_cnt: 1,
                    in_la: n,
                    in_lb: 0,
                    in_sn: 1,
                    out_la: n,
                    out_lb: 0,
                    out_sn: 1,
                },
                true,
            )
        } else {
            (
                LineAddr {
                    la_cnt: inner,
                    lb_cnt: outer,
                    in_la: 1,
                    in_lb: n * inner,
                    in_sn: inner,
                    out_la: 1,
                    out_lb: n * inner,
                    out_sn: inner,
                },
                false,
            )
        };
        let spec = AxisSpec {
            n,
            n_in: n,
            n_out: n,
            n_in_tail: 0,
            n_out_tail: 0,
            dir,
            scale,
            conv: false,
            pre: vec![],
            post: vec![],
            accumulate: false,
            contiguous,
            addr,
            kfmt: KernelFormat::Complex,
            poly_pre: false,
            poly_post: false,
            vox: [1, 1, 1],
        };
        let mut p = Self::from_spec(dev, &spec)?;
        p.simple_len = n * inner * outer;
        Ok(p)
    }

    pub fn from_spec(dev: &Device, spec: &AxisSpec) -> Result<Self, FftError> {
        Self::from_spec_impl(dev, spec, false, false, false, None, (None, None))
    }

    /// As [`from_spec`](Self::from_spec) (`ranged`: as [`from_spec_ranged`](Self::from_spec_ranged) /
    /// [`from_spec_ranged_lines`](Self::from_spec_ranged_lines) with `lin` / `lout`) with the input and / or the output
    /// in a block-floating-point buffer ([`BfpIo`], [`AxisCfg::bfp_in`] / [`AxisCfg::bfp_out`]): f32 strided passes of
    /// 16-line tiles and at least two radix stages, no diagonals or accumulation on the BFP side. Set the block strides
    /// with [`bfp_strides`](Self::bfp_strides) and pass the buffers in [`LaunchPtrs::bin`] / [`LaunchPtrs::bout`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_spec_bfp(dev: &Device, spec: &AxisSpec, ranged: bool, lin: bool, lout: bool, bin: Option<BfpIo>, bout: Option<BfpIo>) -> Result<Self, FftError> {
        let bad = |m: &str| Err(FftError::Invalid(format!("BFP pass (n = {}): {m}", spec.n)));
        if R::IS_F64 {
            return bad("f32 transforms only");
        }
        if spec.contiguous || spec.conv {
            return bad("strided, non-conv passes only");
        }
        if bin.is_some() && (!spec.pre.is_empty() || spec.poly_pre) {
            return bad("no pre diagonals on a BFP input");
        }
        if bout.is_some() && (!spec.post.is_empty() || spec.poly_post || spec.accumulate) {
            return bad("no post diagonals or accumulation on a BFP output");
        }
        if [bin, bout].iter().flatten().any(|b| !(8..=24).contains(&b.bits)) {
            return bad("mantissa bits outside 8 ..= 24");
        }
        if (lin || lout) && !ranged {
            return bad("per-line ranges need a ranged pass");
        }
        let a = &spec.addr;
        if (bin.is_some_and(|b| b.inplace) && a.in_la != 1) || (bout.is_some_and(|b| b.inplace) && a.out_la != 1) || ([bin, bout].iter().flatten().any(|b| b.inplace) && a.la_cnt % 16 != 0) {
            return bad("in-place BFP needs stride-1 lines in whole 16-line tiles");
        }
        Self::from_spec_impl(dev, spec, ranged, lin, lout, None, (bin, bout))
    }

    /// As [`from_spec`](Self::from_spec) with caller code fused into the pass ([`IoHook`]: generated input and / or a
    /// reducing epilogue; strided passes of a mixed-radix length with more than one stage). Launch with
    /// [`launch_ext`](Self::launch_ext) and the hook's extra arguments.
    pub fn from_spec_hooked(dev: &Device, spec: &AxisSpec, hook: IoHook) -> Result<Self, FftError> {
        if spec.contiguous || spec.conv {
            return Err(FftError::Invalid("IoHook: strided, non-conv passes only".into()));
        }
        Self::from_spec_impl(dev, spec, false, false, false, Some(hook), (None, None))
    }

    /// As [`from_spec`](Self::from_spec) for a strided pass whose line groups `lb` each transform only a wrapped range
    /// of `a` ([`AxisCfg::arange`]): set the ranges with [`set_arange`](Self::set_arange) before the first launch.
    pub fn from_spec_ranged(dev: &Device, spec: &AxisSpec) -> Result<Self, FftError> {
        if spec.contiguous {
            return Err(FftError::Invalid("a-range pruning needs a strided pass".into()));
        }
        Self::from_spec_impl(dev, spec, true, false, false, None, (None, None))
    }

    /// As [`from_spec_ranged`](Self::from_spec_ranged), additionally restricting the entries read (`lin`) or written
    /// (`lout`) per line ([`AxisCfg::lrange_in`]): set the table with [`set_lrange`](Self::set_lrange).
    pub fn from_spec_ranged_lines(dev: &Device, spec: &AxisSpec, lin: bool, lout: bool) -> Result<Self, FftError> {
        if spec.contiguous {
            return Err(FftError::Invalid("a-range pruning needs a strided pass".into()));
        }
        Self::from_spec_impl(dev, spec, true, lin, lout, None, (None, None))
    }

    /// Per-line entry ranges `(head, tail)` (≤ n), indexed q = lb · stride + la / div (see [`AxisCfg::lrange_in`]); the
    /// table must cover every q a launched line can reach.
    pub fn set_lrange(&mut self, stream: &Arc<CudaStream>, ranges: &[(usize, usize)], div: usize, stride: usize) -> Result<(), FftError> {
        let need = (self.addr.lb_cnt.max(1) - 1) * stride + (self.addr.la_cnt.max(1) - 1) / div.max(1) + 1;
        if !(self.cfg.lrange_in || self.cfg.lrange_out) || div == 0 || ranges.len() < need || ranges.iter().any(|&(h, t)| h + t > self.cfg.n) {
            return Err(FftError::Invalid(format!("set_lrange: {} ranges (need {need}) for lines of {} (line-ranged plan: {})", ranges.len(), self.cfg.n, self.cfg.lrange_in || self.cfg.lrange_out)));
        }
        let flat: Vec<i32> = ranges.iter().flat_map(|&(h, t)| [h as i32, t as i32]).collect();
        self.lrange = Some((stream.clone_htod(&flat)?, div as i64, stride as i64));
        Ok(())
    }

    /// Per line group `lb` (`addr.lb_cnt` entries): `(head, tail)` — the group transforms the lines
    /// `a ∈ [0, head) ∪ [la_cnt − tail, la_cnt)`; the grid is sized for the longest range.
    pub fn set_arange(&mut self, stream: &Arc<CudaStream>, ranges: &[(usize, usize)]) -> Result<(), FftError> {
        if !self.cfg.arange || ranges.len() != self.addr.lb_cnt || ranges.iter().any(|&(h, t)| h + t > self.addr.la_cnt) {
            return Err(FftError::Invalid(format!("set_arange: {} ranges for {} line groups of {} lines (ranged plan: {})", ranges.len(), self.addr.lb_cnt, self.addr.la_cnt, self.cfg.arange)));
        }
        let flat: Vec<i32> = ranges.iter().flat_map(|&(h, t)| [h as i32, t as i32]).collect();
        self.arange = Some(stream.clone_htod(&flat)?);
        self.arange_uniform = ranges.iter().all(|r| *r == ranges[0]);
        // BFP passes: tiles aligned in the flat line index, the range rounded out to whole tiles (codegen)
        let (lp, nt) = (self.cfg.lpb, self.addr.la_cnt.div_ceil(self.cfg.lpb));
        let tiles = |(h, t): (usize, usize)| if self.cfg.bfp() { let th = h.div_ceil(lp); let ts = if t > 0 { (self.addr.la_cnt - t) / lp } else { nt }; th + nt - ts.max(th) } else { (h + t).div_ceil(lp) };
        let longest = ranges.iter().map(|&r| tiles(r)).max().unwrap_or(0);
        self.grid.0 = u32::try_from(longest.max(1)).map_err(|_| FftError::Invalid(format!("set_arange: {longest} lines exceed the grid x limit")))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn from_spec_impl(dev: &Device, spec: &AxisSpec, arange: bool, lrange_in: bool, lrange_out: bool, hook: Option<IoHook>, (bfp_in, bfp_out): (Option<BfpIo>, Option<BfpIo>)) -> Result<Self, FftError> {
        let n = spec.n;
        if spec.n_in + spec.n_in_tail > n || spec.n_out + spec.n_out_tail > n || spec.n_in == 0 || spec.n_out == 0 {
            return Err(FftError::Invalid(format!(
                "n_in/n_out must be in 1..=n (n={n}, n_in={}, n_out={})",
                spec.n_in, spec.n_out
            )));
        }
        if spec.pre.len() > 4 || spec.post.len() > 4 {
            return Err(FftError::Invalid(
                "at most 4 pre and 4 post diagonal operands".into(),
            ));
        }
        let max_smem = dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize;
        let tile_cnt = spec.addr.la_cnt;
        // transform core: mixed radix of length n, or Bluestein of length m (two shared buffers of m per line)
        let (radices, m) = match factorize(n) {
            Some(r) => (r, n),
            None => {
                let (m, r) = crate::bluestein::conv_length(n);
                (r, m)
            }
        };
        let blue = m != n;
        let (lpb, tpl) = if blue {
            let rmax = *radices.iter().max().unwrap();
            let tpl = (m / rmax).clamp(1, 512);
            let per_line = 2 * m * 2 * R::BYTES;
            let mut lpb = if spec.contiguous { (256 / tpl).max(1) } else { 16usize.min(tile_cnt.next_power_of_two().max(1)) };
            while lpb > 1 && (lpb * per_line > max_smem || lpb * tpl > 1024) {
                lpb /= 2;
            }
            if per_line > max_smem {
                return Err(FftError::SharedMemory { n, need: per_line, have: max_smem });
            }
            (lpb, tpl)
        } else {
            choose_shape(n, &radices, spec.contiguous, tile_cnt, R::BYTES, max_smem)
        };
        let cfg = AxisCfg {
            n,
            radices,
            contiguous: spec.contiguous,
            lpb,
            tpl,
            ctype: R::CTYPE,
            forward: spec.dir == Direction::Forward,
            scale: spec.scale,
            n_in: spec.n_in,
            n_out: spec.n_out,
            n_in_tail: spec.n_in_tail,
            n_out_tail: spec.n_out_tail,
            conv: spec.conv,
            pre: spec.pre.clone(),
            post: spec.post.clone(),
            accumulate: spec.accumulate,
            kfmt: spec.kfmt,
            poly_pre: spec.poly_pre,
            poly_post: spec.poly_post,
            vox: spec.vox,
            arange,
            lrange_in,
            lrange_out,
            hook,
            bfp_in,
            bfp_out,
            bfp_nopad: false,
        };
        let mut cfg = cfg;
        let mut smem = if blue { 2 * m * lpb * 2 * R::BYTES } else { cfg.smem_bytes(R::BYTES) };
        if cfg.bfp() && !blue && smem > max_smem {
            // BFP passes pad each shared-memory entry by one complex (bank conflicts of the codec's runs); without the
            // padding the tile still fits wherever the f32 pass fits
            cfg.bfp_nopad = true;
            smem = cfg.smem_bytes(R::BYTES);
        }
        if smem > max_smem {
            return Err(FftError::SharedMemory {
                n,
                need: smem,
                have: max_smem,
            });
        }
        if blue && arange {
            return Err(FftError::Invalid(format!("a-range pruning is not implemented for the Bluestein length {n}")));
        }
        if cfg.hook.is_some() && (blue || cfg.passes() < 2) {
            return Err(FftError::Invalid(format!("IoHook: needs a mixed-radix length of more than one stage (n = {n})")));
        }
        if cfg.bfp() && (blue || cfg.passes() < 2 || cfg.lpb != 16) {
            return Err(FftError::Invalid(format!("BFP pass: needs a mixed-radix length of more than one stage and 16-line tiles (n = {n}, {} lines per tile)", cfg.lpb)));
        }
        let src = if blue { crate::bluestein::generate(&cfg, m, &cfg.radices) } else { generate(&cfg) };
        // the generated source fully determines the kernel; the name only labels cache files / dumps
        let name = format!(
            "fft_axis_n{n}_{}{}_{}_{}_i{}o{}p{}q{}{}",
            if cfg.forward { "f" } else { "i" },
            if cfg.conv { "c" } else { "" },
            R::CTYPE,
            if cfg.contiguous { "c" } else { "s" },
            spec.n_in,
            spec.n_out,
            spec.pre.len(),
            spec.post.len(),
            if spec.accumulate { "a" } else { "" }
        ) + &format!(
            "{:?}p{}q{}",
            spec.kfmt, spec.poly_pre as u8, spec.poly_post as u8
        ) + &if spec.n_in_tail + spec.n_out_tail > 0 { format!("_t{}u{}", spec.n_in_tail, spec.n_out_tail) } else { String::new() } + &if blue { format!("_blu{m}") } else { String::new() } + if arange { "_ar" } else { "" } + if lrange_in { "_lri" } else { "" } + if lrange_out { "_lro" } else { "" } + if cfg.hook.is_some() { "_hook" } else { "" } + &[("bi", bfp_in), ("bo", bfp_out)].iter().filter_map(|(t, b)| b.map(|b| format!("_{t}{}{}{}", b.bits, if b.compact { "c" } else { "" }, if b.inplace { "i" } else { "" }))).collect::<String>();
        let func = dev.load_function(
            &name,
            &src,
            "fft_axis",
            &CompileSettings {
                options: vec!["--use_fast_math".into()],
                ..Default::default()
            },
        )?;
        if smem > DEFAULT_SMEM {
            func.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )?;
        }
        // twiddles of the core length (n, or m for Bluestein)
        let table = |sg: f64| -> Vec<R::Complex> {
            (0..m)
                .map(|t| {
                    let a = sg * 2.0 * std::f64::consts::PI * (t as f64) / (m as f64);
                    R::complex(a.cos(), a.sin())
                })
                .collect()
        };
        let stream = dev.default_stream();
        let tw_m = stream.clone_htod(&table(-1.0))?;
        let tw_p = stream.clone_htod(&table(1.0))?;
        let blue = if blue {
            let (ch, bf, bi) = crate::bluestein::tables(n, m);
            let up = |v: &[(f64, f64)]| stream.clone_htod(&v.iter().map(|&(a, b)| R::complex(a, b)).collect::<Vec<_>>());
            Some(BlueTables { ch: up(&ch)?, bh_f: up(&bf)?, bh_i: up(&bi)? })
        } else {
            None
        };
        let a = spec.addr;
        let grid = launch_grid(spec.contiguous, a.la_cnt, a.lb_cnt, lpb)?;
        Ok(Self {
            func,
            tw_m,
            tw_p,
            blue,
            cfg,
            addr: a,
            grid,
            smem: smem as u32,
            simple_len: 0,
            split: LineSplit::default(),
            arange: None,
            lrange: None,
            arange_uniform: false,
            bfp_strides: [0; 4],
        })
    }

    /// Elements covered by the plain layout (only meaningful for plans built with [`AxisPlan::new`]).
    pub fn len(&self) -> usize {
        self.simple_len
    }

    pub fn launch(&self, stream: &Arc<CudaStream>, p: &LaunchPtrs) -> Result<(), FftError> {
        self.launch_ext(stream, p, &[])
    }

    /// [`launch`](Self::launch) with the extra arguments of a hooked pass ([`IoHook::params`], in order)
    pub fn launch_ext(&self, stream: &Arc<CudaStream>, p: &LaunchPtrs, extra: &[KArg]) -> Result<(), FftError> {
        self.launch_lines(stream, p, extra, None)
    }

    /// [`launch_ext`](Self::launch_ext) over a sub-range of the line groups: `sub = (lb_cnt, split)` replaces the
    /// plan's line-group count and split for this launch (the caller offsets `p.inp` / `p.out` to the first group;
    /// e.g. one slab of z planes of a pass). `arange` passes only with uniform ranges, and per-line tables
    /// (`lrange`) only when they do not depend on the group (stride 0).
    pub fn launch_lines(&self, stream: &Arc<CudaStream>, p: &LaunchPtrs, extra: &[KArg], sub: Option<(usize, LineSplit)>) -> Result<(), FftError> {
        if (self.cfg.hook.is_some() || !extra.is_empty()) && extra.len() != self.cfg.hook.as_ref().map_or(0, |h| h.params.matches(',').count()) {
            return Err(FftError::Invalid(format!("launch_ext: {} extra arguments for the hook's parameters", extra.len())));
        }
        if sub.is_some() && ((self.cfg.arange && !self.arange_uniform) || self.lrange.as_ref().is_some_and(|l| l.2 != 0)) {
            return Err(FftError::Invalid("launch_lines: per-group a-ranges or line tables".into()));
        }
        let (twm, _g1) = self.tw_m.device_ptr(stream);
        let (twp, _g2) = self.tw_p.device_ptr(stream);
        let a = &self.addr;
        let (la_cnt, lb_cnt) = (a.la_cnt as i64, sub.map_or(a.lb_cnt, |s| s.0) as i64);
        let split = sub.map_or(self.split, |s| s.1);
        let grid = match sub {
            Some((n, _)) => {
                let g = launch_grid(self.cfg.contiguous, a.la_cnt, n, self.cfg.lpb)?;
                // a-range passes: grid x from the (uniform) range
                if self.cfg.arange { (self.grid.0, g.1, g.2) } else { g }
            }
            None => self.grid,
        };
        let (ila, ilb, isn, ola, olb, osn) = (
            a.in_la as i64,
            a.in_lb as i64,
            a.in_sn as i64,
            a.out_la as i64,
            a.out_lb as i64,
            a.out_sn as i64,
        );
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: (self.cfg.block_threads() as u32, 1, 1),
            shared_mem_bytes: self.smem,
        };
        unsafe {
            let mut b = stream.launch_builder(&self.func);
            let (ks32, ks64) = (p.kscale as f32, p.kscale);
            b.arg(&p.inp).arg(&p.out).arg(&twm).arg(&twp).arg(&p.kern);
            if R::IS_F64 {
                b.arg(&ks64);
            } else {
                b.arg(&ks32);
            }
            for x in &p.pre {
                b.arg(x);
            }
            for x in &p.post {
                b.arg(x);
            }
            b.arg(&p.ppre).arg(&p.nppre).arg(&p.ppost).arg(&p.nppost);
            b.arg(&la_cnt)
                .arg(&lb_cnt)
                .arg(&ila)
                .arg(&ilb)
                .arg(&isn)
                .arg(&ola)
                .arg(&olb)
                .arg(&osn);
            let sp = &split;
            let sv: [i64; 7] = if sp.lb1_cnt > 0 {
                [sp.lb1_cnt as i64, sp.lb1_hi as i64, sp.lb1_gap as i64, sp.lb2_hi as i64, sp.lb2_gap as i64, sp.in_lb2 as i64, sp.out_lb2 as i64]
            } else {
                [lb_cnt.max(1), i64::MAX, 0, i64::MAX, 0, 0, 0]
            };
            for v in &sv {
                b.arg(v);
            }
            let arg = match (&self.arange, self.cfg.arange) {
                (Some(t), true) => Some(t.device_ptr(stream)),
                (None, true) => return Err(FftError::Invalid("a-range pass launched before set_arange".into())),
                _ => None,
            };
            if let Some((ap, _)) = &arg {
                b.arg(ap);
            }
            let lr = match (&self.lrange, self.cfg.lrange_in || self.cfg.lrange_out) {
                (Some((t, d, q)), true) => Some((t.device_ptr(stream), *d, *q)),
                (None, true) => return Err(FftError::Invalid("line-ranged pass launched before set_lrange".into())),
                _ => None,
            };
            if let Some(((lp, _), d, q)) = &lr {
                b.arg(lp).arg(d).arg(q);
            }
            let bs = &self.bfp_strides;
            let bch: (i64, i64) = if p.bch[0] > 0 { (p.bch[0], p.bch[1]) } else { (1, 1) };
            let bch_ref = (&bch.0, &bch.1);
            // block strides: per entry always, per line group when there is more than one group
            let unset = |e: i64, g: i64| e == 0 || (g == 0 && lb_cnt > 1);
            if (self.cfg.bfp_in.is_some() && unset(bs[0], bs[1])) || (self.cfg.bfp_out.is_some() && unset(bs[2], bs[3])) {
                return Err(FftError::Invalid("BFP pass launched without its block strides (bfp_strides)".into()));
            }
            if self.cfg.bfp_in.is_some() {
                if p.bin[0] == 0 || p.bin[1] == 0 {
                    return Err(FftError::Invalid("BFP-input pass launched without its buffer".into()));
                }
                b.arg(&p.bin[0]).arg(&p.bin[1]).arg(&bs[0]).arg(&bs[1]);
            }
            if self.cfg.bfp_out.is_some() {
                if p.bout[0] == 0 || p.bout[1] == 0 {
                    return Err(FftError::Invalid("BFP-output pass launched without its buffer".into()));
                }
                b.arg(&p.bout[0]).arg(&p.bout[1]).arg(&bs[2]).arg(&bs[3]);
                b.arg(bch_ref.0).arg(bch_ref.1);
            }
            for x in extra {
                match x {
                    KArg::Ptr(v) => b.arg(v),
                    KArg::I32(v) => b.arg(v),
                    KArg::I64(v) => b.arg(v),
                    KArg::F32(v) => b.arg(v),
                };
            }
            let guards = self.blue.as_ref().map(|t| (t.ch.device_ptr(stream), t.bh_f.device_ptr(stream), t.bh_i.device_ptr(stream)));
            if let Some(((ch, _), (bf, _), (bi, _))) = &guards {
                b.arg(ch).arg(bf).arg(bi);
            }
            b.launch(cfg)?;
        }
        Ok(())
    }

    /// Out-of-place execute (`input` and `output` may not alias).
    pub fn execute<I: DevicePtr<R::Complex>, O: DevicePtrMut<R::Complex>>(
        &self,
        stream: &Arc<CudaStream>,
        input: &I,
        output: &mut O,
    ) -> Result<(), FftError> {
        let need = self.len();
        for have in [input.len(), output.len()] {
            if have < need {
                return Err(FftError::BufferSize { need, have });
            }
        }
        let (ip, _gi) = input.device_ptr(stream);
        let (op, _go) = output.device_ptr_mut(stream);
        self.launch(
            stream,
            &LaunchPtrs {
                inp: ip,
                out: op,
                ..Default::default()
            },
        )
    }

    pub fn execute_inplace<B: DevicePtrMut<R::Complex>>(
        &self,
        stream: &Arc<CudaStream>,
        buf: &mut B,
    ) -> Result<(), FftError> {
        let need = self.len();
        if buf.len() < need {
            return Err(FftError::BufferSize {
                need,
                have: buf.len(),
            });
        }
        let (p, _g) = buf.device_ptr_mut(stream);
        self.launch(
            stream,
            &LaunchPtrs {
                inp: p,
                out: p,
                ..Default::default()
            },
        )
    }
}

/// N-dimensional C2C FFT over a row-major array (`dims[last]` contiguous): one [`AxisPlan`] per axis of length > 1.
pub struct FftPlan<R: Real> {
    pub dims: Vec<usize>,
    axes: Vec<AxisPlan<R>>, // execution order: last axis first
}

impl<R: Real> FftPlan<R> {
    pub fn new(dev: &Device, dims: &[usize], dir: Direction, norm: Norm) -> Result<Self, FftError> {
        let total: usize = dims.iter().product();
        let scale = match norm {
            Norm::None => 1.0,
            Norm::Inverse => 1.0 / total as f64,
            Norm::Unitary => 1.0 / (total as f64).sqrt(),
        };
        let nd = dims.len();
        let mut axes = Vec::new();
        let order: Vec<usize> = (0..nd).rev().filter(|&a| dims[a] > 1).collect();
        for (k, &a) in order.iter().enumerate() {
            let inner: usize = dims[a + 1..].iter().product();
            let outer: usize = dims[..a].iter().product();
            // the scale is applied once, on the last pass
            let sc = if k == order.len() - 1 { scale } else { 1.0 };
            axes.push(AxisPlan::new(dev, dims[a], inner, outer, dir, sc)?);
        }
        Ok(Self {
            dims: dims.to_vec(),
            axes,
        })
    }

    /// Like [`FftPlan::new`] but only the axes with `transform[a] == true` are transformed (e.g. a trailing channel axis
    /// of interleaved multi-channel data is left alone). The normalisation uses the transformed size.
    pub fn new_partial(dev: &Device, dims: &[usize], transform: &[bool], dir: Direction, norm: Norm) -> Result<Self, FftError> {
        assert_eq!(dims.len(), transform.len());
        let total: usize = dims.iter().zip(transform).filter(|(_, &t)| t).map(|(d, _)| *d).product();
        let scale = match norm {
            Norm::None => 1.0,
            Norm::Inverse => 1.0 / total as f64,
            Norm::Unitary => 1.0 / (total as f64).sqrt(),
        };
        let nd = dims.len();
        let order: Vec<usize> = (0..nd).rev().filter(|&a| dims[a] > 1 && transform[a]).collect();
        let mut axes = Vec::new();
        for (k, &a) in order.iter().enumerate() {
            let inner: usize = dims[a + 1..].iter().product();
            let outer: usize = dims[..a].iter().product();
            let sc = if k == order.len() - 1 { scale } else { 1.0 };
            axes.push(AxisPlan::new(dev, dims[a], inner, outer, dir, sc)?);
        }
        Ok(Self { dims: dims.to_vec(), axes })
    }

    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn execute<I: DevicePtr<R::Complex>, O: DevicePtrMut<R::Complex>>(
        &self,
        stream: &Arc<CudaStream>,
        input: &I,
        output: &mut O,
    ) -> Result<(), FftError> {
        let mut it = self.axes.iter();
        match it.next() {
            Some(first) => first.execute(stream, input, output)?,
            None => return Ok(()),
        }
        for ax in it {
            ax.execute_inplace(stream, output)?;
        }
        Ok(())
    }

    pub fn execute_inplace<B: DevicePtrMut<R::Complex>>(
        &self,
        stream: &Arc<CudaStream>,
        buf: &mut B,
    ) -> Result<(), FftError> {
        for ax in &self.axes {
            ax.execute_inplace(stream, buf)?;
        }
        Ok(())
    }
}
