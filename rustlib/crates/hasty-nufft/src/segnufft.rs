//! Segmented physics NUFFT: multi-coil encoding operator with off-resonance / time-dependent quadratic phase handled by
//! L time segments whose spatial diagonals P_l are applied in a fused pad pass, interpolators a_l(τ) tabulated in the
//! readout time, a per-sample first/second-order correction for the remaining quadratic phase ΔC_n folded into the
//! interpolation kernel (no extra FFTs), and optional temporal interpolation over cardiac frames (each sample sees the
//! B-spline combination of the neighbouring frame images at its own acquisition time; no extra FFTs).
//! See `research/concomitant_segmented_first_order.md`.
//!
//!   y_c(n) = e^{−iφ0_n} Σ_f b_f(t_n) Σ_l a_l(τ_n) F[S_c P_l x_f](k_n)  + ΔC_n derivative terms        (forward)
//!
//! The fine grid holds NSEG × (coil batch) channels per frame slot; the coil batch is sized to the memory budget, so the
//! same code runs on cards with very different memory. Kernels are generated at run time and cached by `hasty-cuda`.
//!
//! Additions (the paths above are unchanged):
//! * several images on the same samples ([`SegNufft::forward_multi`] / [`SegNufft::adjoint_multi`]):
//!   y = Σ_k w_k ⊙ A x_k and x_k = Aᴴ(w̄_k ⊙ y) with per-sample weights w_k, one tile pass over the K images' grids
//!   (kernels `segnufft_multi.cu`): the FFTs stay per image, the sample staging, kernel weights, interpolators and
//!   ΔC terms are shared; the K grids come out of the engine's grid memory (smaller coil batch) or, within
//!   `max_grid_bytes + multi_extra_bytes`, extra buffers;
//! * correction order 3 ([`SegNufft::points_hi`]): order 2 plus a per-sample degree-3/4 polynomial phase to first
//!   order (the higher-order part of the concomitant fit at GNL-warped positions), applied by the multi kernels.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use hasty_fft::{Direction, FftPlan, Norm, PrunedChanFft3};

use crate::plan::{MDims, NufftError, NufftOpts, NufftPlan};

pub struct SegNufftOpts {
    pub sigma: f64,
    pub width: usize,
    /// number of time segments
    pub nseg: usize,
    /// order of the per-sample quadratic-phase correction in the interpolation kernel: 0 none, 1 first, 2 second
    pub order: u32,
    /// number of (periodic) frames; 1 = static image
    pub frames: usize,
    /// temporal B-spline degree between frames: 1 linear, 3 cubic (ignored for one frame)
    pub bspline: u32,
    /// fine-grid memory budget (bytes) for all frame slots × NSEG × coil-batch channels
    pub max_grid_bytes: usize,
}

impl Default for SegNufftOpts {
    fn default() -> Self {
        Self { sigma: 1.25, width: 10, nseg: 8, order: 2, frames: 1, bspline: 1, max_grid_bytes: 4 << 30 }
    }
}

struct Funcs {
    pad: CudaFunction,
    crop: CudaFunction,
    /// r = −y (residual initialisation of `forward_residual`)
    neg: CudaFunction,
    fwd: CudaFunction,
    adj: CudaFunction,
}

/// Launch shape of the tile kernels: tile edge, coils per block (adjoint / forward), forward lanes per (sample, coil,
/// frame slot), staged samples per chunk and the longest sample list of one work item (longer tile lists are split).
#[derive(Clone, Copy, Debug)]
pub struct TileCfg {
    pub tt: usize,
    pub cpba: usize,
    pub cpbf: usize,
    pub ppf: usize,
    pub chunk: usize,
    pub max_item: usize,
}

/// what a shape is fitted for: segments, correction order, frame slots
#[derive(Clone, Copy, Debug)]
struct Shape {
    nseg: usize,
    order: u32,
    nf: usize,
}

impl TileCfg {
    /// Default shape within 48 KB of shared memory (any CUDA GPU) and ≤ 512-thread blocks; `autotune` refines it within
    /// `smax` (the device's opt-in limit). Env
    /// overrides for experiments: HASTY_SEG_TT, HASTY_SEG_CPBA, HASTY_SEG_CPBF, HASTY_SEG_PPF, HASTY_SEG_CHUNK,
    /// HASTY_SEG_MAXITEM.
    fn new(sh: Shape, cb: usize, smax: usize) -> Self {
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        let tt = env("HASTY_SEG_TT").unwrap_or(4);
        let ppf = env("HASTY_SEG_PPF").unwrap_or((2 / sh.nf).max(1));
        let max_item = env("HASTY_SEG_MAXITEM").unwrap_or(1024);
        // coil groups divide the batch (no idle coil threads); adjoint blocks ≤ 512 threads
        let divisors: Vec<usize> = (1..=cb).filter(|d| cb % d == 0).collect();
        let cpba = env("HASTY_SEG_CPBA").unwrap_or(*divisors.iter().filter(|&&d| d * sh.nf * tt * tt * tt <= 512).max().unwrap_or(&1)).clamp(1, cb);
        let fit = |cpbf: usize| Self::fit(sh, tt, cpba, cpbf, ppf, max_item, smax).map_or(0, |t| t.chunk);
        // forward: the divisor with the most (sample, coil) work per staged chunk; ties → more coils per block
        let cpbf = env("HASTY_SEG_CPBF").unwrap_or_else(|| *divisors.iter().filter(|&&d| fit(d) >= 8).max_by_key(|&&d| (d * fit(d), d)).unwrap_or(&1)).clamp(1, cb);
        let mut tc = Self::fit(sh, tt, cpba, cpbf, ppf, max_item, smax).unwrap_or_else(|| panic!("no tile shape fits ({sh:?}, tt {tt}, cpba {cpba}, cpbf {cpbf}, ppf {ppf})"));
        if let Some(c) = env("HASTY_SEG_CHUNK") {
            tc.chunk = c;
            assert!(tc.fits(sh, smax), "HASTY_SEG_CHUNK={c} does not fit: {tc:?}");
        }
        tc
    }

    /// The shape with the largest chunk that fits `smax` bytes of shared memory (forward blocks whole warps, chunk ≤ 32);
    /// None if nothing fits.
    fn fit(sh: Shape, tt: usize, cpba: usize, cpbf: usize, ppf: usize, max_item: usize, smax: usize) -> Option<Self> {
        let lanes = ppf * sh.nf;
        if !ppf.is_power_of_two() || lanes > 32 || tt.pow(3) * sh.nf * cpba > 1024 {
            return None;
        }
        let step = 32 / gcd(32, cpbf * lanes);
        let mut c = 32usize.min(512 / (cpbf * lanes)) / step * step;
        while c > 0 {
            let t = Self { tt, cpba, cpbf, ppf, chunk: c, max_item };
            if t.fits(sh, smax) {
                return Some(t);
            }
            c -= step;
        }
        None
    }

    fn fits(&self, sh: Shape, smax: usize) -> bool {
        self.smem(sh).iter().all(|&b| b <= smax) && (self.chunk * self.cpbf * self.ppf * sh.nf) % 32 == 0 && self.tt.pow(3) * sh.nf * self.cpba <= 1024
    }

    /// dynamic shared memory (bytes) of the [adjoint, forward] tile kernels (mirrors the kernels' layout)
    fn smem(&self, sh: Shape) -> [usize; 2] {
        let (tt3, c, nseg) = (self.tt.pow(3), self.chunk, sh.nseg);
        let nk = [1usize, 3, 5][sh.order.min(2) as usize];
        let up = |b: usize, a: usize| b.div_ceil(a) * a;
        // Stage: rec, ax, ee, box | aa, wt (8-aligned) | 16-aligned size
        let st = 48 * c + 4 * 3 * nk * self.tt * c + 4 * 15 * if sh.order >= 2 { c } else { 1 } + 4 * c;
        let st = up(up(st, 8) + 8 * sh.nf * nseg * c + 8 * (tt3 + self.ppf * sh.nf) * c, 16);
        let base = up(st + 4 * self.max_item, 16);
        let gn = sh.nf * self.cpbf * nseg;
        let gts = gn + (2 + 4 - gn % 4) % 4;
        [base + 8 * c * self.cpba, base + 8 * tt3 * gts]
    }
}

/// shared memory a block may use without opting in (every CUDA GPU); `SegNufft` raises it to the device's opt-in limit
const SMEM_DEFAULT: usize = 48 * 1024;

// launch shapes of the multi-image kernels (`segnufft_multi.cu`, `km` images per pass); the single-image shapes above
// are unchanged
impl TileCfg {
    /// dynamic shared memory (bytes) of the [adjoint, forward] multi-image kernels (mirrors their layout:
    /// Stage | list | StageM (weights, order-3 coefficients) | coil values / tile values)
    fn smem_multi(&self, sh: Shape, km: usize) -> [usize; 2] {
        let (tt3, c, nseg) = (self.tt.pow(3), self.chunk, sh.nseg);
        let nk = [1usize, 3, 5][sh.order.min(2) as usize];
        let up = |b: usize, a: usize| b.div_ceil(a) * a;
        let st = 48 * c + 4 * 3 * nk * self.tt * c + 4 * 15 * if sh.order >= 2 { c } else { 1 } + 4 * c;
        let st = up(up(st, 8) + 8 * sh.nf * nseg * c + 8 * (tt3 + self.ppf * sh.nf) * c, 16);
        let base = up(st + 4 * self.max_item, 16);
        let sm = up(8 * c * km + if sh.order >= 3 { 4 * NHI * c } else { 0 }, 16);
        let gn = km * sh.nf * self.cpbf * nseg;
        let gts = gn + (2 + 4 - gn % 4) % 4;
        [base + sm + 8 * c * self.cpba, base + sm + 8 * tt3 * gts]
    }

    fn fits_multi(&self, sh: Shape, km: usize, smax: usize) -> bool {
        self.smem_multi(sh, km).iter().all(|&b| b <= smax) && (self.chunk * self.cpbf * self.ppf * sh.nf) % 32 == 0 && self.tt.pow(3) * sh.nf * self.cpba * km <= 1024
    }

    /// the multi shape with the largest chunk that fits (as [`TileCfg::fit`])
    #[allow(clippy::too_many_arguments)]
    fn fit_multi(sh: Shape, km: usize, tt: usize, cpba: usize, cpbf: usize, ppf: usize, max_item: usize, smax: usize) -> Option<Self> {
        let lanes = ppf * sh.nf;
        if !ppf.is_power_of_two() || lanes > 32 || tt.pow(3) * sh.nf * cpba * km > 1024 {
            return None;
        }
        let step = 32 / gcd(32, cpbf * lanes);
        let mut c = 32usize.min(512 / (cpbf * lanes)) / step * step;
        while c > 0 {
            let t = Self { tt, cpba, cpbf, ppf, chunk: c, max_item };
            if t.fits_multi(sh, km, smax) {
                return Some(t);
            }
            c -= step;
        }
        None
    }

    /// default multi shape for `km` images and a coil batch of `cb` (heuristics of [`TileCfg::new`]; adjoint blocks
    /// ≤ 512 threads where possible); None if nothing fits `smax`
    fn new_multi(sh: Shape, km: usize, cb: usize, tt: usize, max_item: usize, smax: usize) -> Option<Self> {
        let ppf = (2 / sh.nf).max(1);
        let divisors: Vec<usize> = (1..=cb).filter(|d| cb % d == 0).collect();
        let cpba = *divisors.iter().filter(|&&d| d * sh.nf * tt.pow(3) * km <= 512).max().unwrap_or(&1);
        let fit = |cpbf: usize| Self::fit_multi(sh, km, tt, cpba, cpbf, ppf, max_item, smax).map_or(0, |t| t.chunk);
        let cpbf = *divisors.iter().filter(|&&d| fit(d) >= 8).max_by_key(|&&d| (d * fit(d), d)).unwrap_or(&1);
        Self::fit_multi(sh, km, tt, cpba, cpbf, ppf, max_item, smax)
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// images per pass of the multi-image kernels (their grid table holds 4 images × 4 frame slots)
pub const MULTI_KMAX: usize = 4;
/// degree-3/4 monomials of the order-3 correction: degree 3 (10) then degree 4 (15), x order outer, y order inner
pub const NHI: usize = 25;

/// exponents [a, b, c] of the order-3 monomials x^a y^b z^c in the kernels' order
pub fn hi_exponents() -> [[u32; 3]; NHI] {
    let mut e = [[0u32; 3]; NHI];
    let mut i = 0;
    for d in [3u32, 4] {
        for a in 0..=d {
            for b in 0..=d - a {
                e[i] = [a, b, d - a - b];
                i += 1;
            }
        }
    }
    e
}

/// Memory plan of the multi-image path for `k` images: (images per pass g ≤ min(k, 4), coils per batch, images held in
/// each engine frame slot). The engine slots hold `cb` coils × NSEG channels of one image (`per_coil` bytes per coil
/// over all frame slots); a pass needs g grids of `cbm` coils per frame slot, ⌊cb / cbm⌋ of them are views into an
/// engine slot, the rest are extra buffers. Feasible: engine slots + extra ≤ `budget` and `ok(g, cbm)` (a tile shape
/// fits); among those the most images × coils per pass, then the most images. g = 1, cbm = cb always fits the budget.
pub fn multi_plan(k: usize, cb: usize, nc: usize, per_coil: usize, budget: usize, ok: &dyn Fn(usize, usize) -> bool) -> (usize, usize, usize) {
    let cap = nc.min(16).max(1);
    let mut best: Option<(usize, usize, usize)> = None;
    for g in (1..=k.clamp(1, MULTI_KMAX)).rev() {
        for cbm in (1..=cap).rev() {
            let fit = (cb / cbm).min(g);
            let extra = (g - fit) * cbm * per_coil;
            if cb * per_coil + extra > budget || !ok(g, cbm) {
                continue;
            }
            if best.is_none_or(|(bg, bc, _)| g * cbm > bg * bc) {
                best = Some((g, cbm, fit));
            }
        }
    }
    best.unwrap_or_else(|| {
        let cbm = cb.clamp(1, cap);
        assert!(ok(1, cbm), "no multi-image tile shape fits");
        (1, cbm, 1)
    })
}

/// grid-slot pointers of one launch (frame offset o → slot), passed by value
#[repr(C)]
#[derive(Clone, Copy)]
struct Grids {
    p: [u64; 4],
}
unsafe impl DeviceRepr for Grids {}

/// multi-image grid-slot pointers, [image × 4 + frame offset] (`GridsM` in segnufft_multi.cu)
#[repr(C)]
#[derive(Clone, Copy)]
struct GridsM {
    p: [u64; 16],
}
unsafe impl DeviceRepr for GridsM {}

/// per-image sample-weight pointers (0: weight 1)
#[repr(C)]
#[derive(Clone, Copy)]
struct WPtrs {
    p: [u64; 4],
}
unsafe impl DeviceRepr for WPtrs {}

/// one frame interval: its work items and whether any tile list was split
#[derive(Clone, Copy, Debug, Default)]
struct Interval {
    first: usize,
    count: usize,
    split: bool,
}

/// Exact tile → sample lists: for every tt³ tile of the fine grid, the samples (sorted order `perm`) whose W³ support
/// meets it (`sc`: fine-grid coordinates in [0, nf) of all m samples, SoA 3 × m; `perm`: the samples to list, in
/// bin-sorted order). Returns the lists and the work items
/// [tile, first, count, split] heaviest first (every tile has at least one item, empty tiles a zero-count one).
pub fn tile_lists(sc: &[f32], m: usize, perm: &[u32], nf: [usize; 3], w: usize, tt: usize, max_item: usize) -> (Vec<u32>, Vec<[i32; 4]>, bool) {
    let nt = nf.map(|n| n.div_ceil(tt));
    let ntiles = nt[0] * nt[1] * nt[2];
    let axis = |s: f32, a: usize, out: &mut Vec<usize>| {
        out.clear();
        let i0 = (s - w as f32 * 0.5).ceil() as i64;
        for t in 0..w as i64 {
            let ti = (i0 + t).rem_euclid(nf[a] as i64) as usize / tt;
            if !out.contains(&ti) {
                out.push(ti);
            }
        }
    };
    let mut ax: [Vec<usize>; 3] = Default::default();
    let mut tiles_of = |j: usize, f: &mut dyn FnMut(usize)| {
        for a in 0..3 {
            axis(sc[a * m + j], a, &mut ax[a]);
        }
        for &z in &ax[2] {
            for &y in &ax[1] {
                for &x in &ax[0] {
                    f((z * nt[1] + y) * nt[0] + x);
                }
            }
        }
    };
    let mut cnt = vec![0usize; ntiles];
    for &j in perm {
        tiles_of(j as usize, &mut |t| cnt[t] += 1);
    }
    let mut off = vec![0usize; ntiles + 1];
    for t in 0..ntiles {
        off[t + 1] = off[t] + cnt[t];
    }
    let mut pos = off.clone();
    let mut tl = vec![0u32; off[ntiles]];
    for &j in perm {
        tiles_of(j as usize, &mut |t| {
            tl[pos[t]] = j;
            pos[t] += 1;
        });
    }
    let mut items = Vec::with_capacity(ntiles);
    let mut split = false;
    for t in 0..ntiles {
        let c = cnt[t];
        let parts = c.div_ceil(max_item).max(1);
        split |= parts > 1;
        for k in 0..parts {
            let first = off[t] + k * max_item;
            items.push([t as i32, first as i32, (c - k * max_item).min(max_item) as i32, (parts > 1) as i32]);
        }
    }
    items.sort_by_key(|it| std::cmp::Reverse(it[2]));
    (tl, items, split)
}

/// Per-sample-set data of a [`SegNufft`] (records, tile lists, frame intervals), built once by [`SegNufft::points`].
pub struct SegPoints {
    /// per-sample records, 12 floats: sc0, sc1, sc2, τ, φ0, ΔC (kernel-argument units, off-diagonals doubled), u
    rec: CudaSlice<f32>,
    tl: CudaSlice<u32>,
    items: CudaSlice<i32>,
    ivals: Vec<Interval>,
    pub m: usize,
    /// order 3: per-sample weights u (float4 per sample) and the degree-3/4 table in kernel units (4 × NHI)
    hi: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    /// host-resident form ([`SegPoints::to_host`]): the tables live in pinned host memory (the device buffers above
    /// are 1-element placeholders) and are uploaded into one of the engine's two device slots when the set is used
    host: Option<Box<PointsHost>>,
    /// identity of the tables (the slot cache key), unique per `points` call
    id: u64,
}

static POINTS_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// host copies of a set's tables, with the true lengths (the buffers hold at least one element): pinned, or packed
/// ([`SegPoints::pack`])
struct PointsHost {
    rec: HostBuf<f32>,
    tl: HostBuf<u32>,
    items: HostBuf<i32>,
    hi: Option<(HostBuf<f32>, HostBuf<f32>)>,
}

/// raw bytes per packed block
const PACK_BLOCK: usize = 1 << 20;

/// One host-resident table buffer: pinned memory (uploaded directly), or compressed blocks of [`PACK_BLOCK`] raw
/// bytes (hasty_value block codec, 4-byte shuffle) decoded in parallel into a pinned staging buffer on upload.
enum HostBuf<T> {
    Pinned(PinnedHostSlice<T>),
    Packed { len: usize, blocks: Vec<Vec<u8>> },
}

impl<T: DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy> HostBuf<T> {
    fn len(&self) -> usize {
        match self {
            HostBuf::Pinned(p) => p.len(),
            HostBuf::Packed { len, .. } => *len,
        }
    }
    fn bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    /// bytes held on the host
    fn stored(&self) -> usize {
        match self {
            HostBuf::Pinned(p) => p.len() * std::mem::size_of::<T>(),
            HostBuf::Packed { blocks, .. } => blocks.iter().map(|b| b.len()).sum(),
        }
    }
    fn pack(&mut self, o: &hasty_value::Options) -> Result<(), NufftError> {
        use rayon::prelude::*;
        if let HostBuf::Pinned(p) = self {
            let sl = p.as_slice()?;
            // SAFETY: plain 4-byte words viewed as bytes
            let raw = unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u8, std::mem::size_of_val(sl)) };
            let blocks = raw.par_chunks(PACK_BLOCK).map(|c| hasty_value::codec::compress_block(c, 4, false, o)).collect();
            *self = HostBuf::Packed { len: sl.len(), blocks };
        }
        Ok(())
    }
    /// decode into `out` (exactly `bytes()` long)
    fn unpack_into(&self, out: &mut [u8]) -> Result<(), NufftError> {
        use rayon::prelude::*;
        match self {
            HostBuf::Pinned(p) => {
                let sl = p.as_slice()?;
                out.copy_from_slice(unsafe { std::slice::from_raw_parts(sl.as_ptr() as *const u8, std::mem::size_of_val(sl)) });
                Ok(())
            }
            HostBuf::Packed { blocks, .. } => out
                .par_chunks_mut(PACK_BLOCK)
                .zip(blocks.par_iter())
                .try_for_each(|(o, b)| hasty_value::codec::decompress_block(b, o))
                .map_err(|e| NufftError::Invalid(format!("packed sample tables: {e}"))),
        }
    }
    fn to_vec(&self) -> Result<Vec<T>, NufftError> {
        let mut v: Vec<T> = vec![unsafe { std::mem::zeroed() }; self.len()];
        // SAFETY: T is a plain 4-byte word
        let b = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, self.bytes()) };
        self.unpack_into(b)?;
        Ok(v)
    }
    fn to_dev(&self, st: &Arc<CudaStream>) -> Result<CudaSlice<T>, NufftError> {
        match self {
            HostBuf::Pinned(p) => Ok(st.clone_htod(p)?),
            HostBuf::Packed { .. } => Ok(st.clone_htod(&self.to_vec()?)?),
        }
    }
    /// upload into `dst[..len]` on `on`: pinned directly; packed through `stage[off..]` (decoded now)
    fn upload(&self, on: &Arc<CudaStream>, dst: &mut CudaSlice<T>, stage: &mut [u8], off: usize) -> Result<usize, NufftError> {
        let n = self.len();
        match self {
            HostBuf::Pinned(p) => on.memcpy_htod(p, &mut dst.slice_mut(0..n))?,
            HostBuf::Packed { .. } => {
                let b = &mut stage[off..off + self.bytes()];
                self.unpack_into(b)?;
                // SAFETY: 4-byte aligned region of the pinned staging buffer, plain words; the copy is asynchronous
                // (pinned), the caller keeps the staging buffer untouched until its event completes
                let src = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, n) };
                on.memcpy_htod(src, &mut dst.slice_mut(0..n))?;
            }
        }
        Ok(self.bytes())
    }
}

/// pinned copy of a device buffer (at least one element; empty buffers keep one unused element)
fn pinned_copy<T: DeviceRepr>(ctx: &Arc<CudaContext>, st: &Arc<CudaStream>, d: &CudaSlice<T>) -> Result<PinnedHostSlice<T>, NufftError> {
    let mut h = unsafe { ctx.alloc_pinned_with_flags::<T>(d.len().max(1), 0)? };
    st.memcpy_dtoh(d, &mut h)?;
    Ok(h)
}

impl SegPoints {
    /// the tables are in pinned host memory (streamed per use)
    pub fn is_host(&self) -> bool {
        self.host.is_some()
    }

    /// The host-resident tables as (name, bytes) — records f32, tile lists u32, work items i32, order-3 data f32;
    /// all 4-byte words. For size / compression studies and host-side stores; None while the tables are on the device.
    pub fn host_parts(&self) -> Option<Vec<(&'static str, &[u8])>> {
        let h = self.host.as_ref()?;
        fn b<T: cudarc::driver::ValidAsZeroBits>(x: &HostBuf<T>) -> Option<&[u8]> {
            let HostBuf::Pinned(x) = x else { return None };
            let s = x.as_slice().expect("pinned slice");
            // SAFETY: plain 4-byte words viewed as bytes
            Some(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) })
        }
        let mut v = vec![("rec", b(&h.rec)?), ("tl", b(&h.tl)?), ("items", b(&h.items)?)];
        if let Some((u, t)) = &h.hi {
            v.push(("hi_u", b(u)?));
            v.push(("hi_t", b(t)?));
        }
        Some(v)
    }

    /// Compress host-resident tables ([`to_host`](Self::to_host) first) into blocks (`o`: codec; 4-byte shuffle);
    /// on use they are decoded in parallel into a pinned staging buffer and uploaded. The tables are trajectory data
    /// (coordinates, tile lists, work items) and compress well (zstd1 ≈ ×6). Returns (host bytes before, after).
    pub fn pack(&mut self, o: &hasty_value::Options) -> Result<(usize, usize), NufftError> {
        if self.host.is_none() {
            return Err(NufftError::Invalid("pack: tables are on the device (to_host first)".into()));
        }
        let before = self.host_stored_bytes();
        let h = self.host.as_mut().unwrap();
        h.rec.pack(o)?;
        h.tl.pack(o)?;
        h.items.pack(o)?;
        if let Some((u, t)) = h.hi.as_mut() {
            u.pack(o)?;
            t.pack(o)?;
        }
        Ok((before, self.host_stored_bytes()))
    }

    /// the host form is packed
    pub fn is_packed(&self) -> bool {
        self.host.as_ref().is_some_and(|h| matches!(h.rec, HostBuf::Packed { .. }))
    }

    /// bytes held on the host (pinned or packed; 0 on the device)
    pub fn host_stored_bytes(&self) -> usize {
        self.host.as_ref().map_or(0, |h| h.rec.stored() + h.tl.stored() + h.items.stored() + h.hi.as_ref().map_or(0, |(u, t)| u.stored() + t.stored()))
    }

    /// bytes of the set's sample tables (records, tile lists, work items, order-3 data), wherever they live
    pub fn table_bytes(&self) -> usize {
        match &self.host {
            Some(h) => h.rec.bytes() + h.tl.bytes() + h.items.bytes() + h.hi.as_ref().map_or(0, |(u, t)| u.bytes() + t.bytes()),
            None => self.device_bytes(),
        }
    }

    /// device bytes held by this object now (placeholders only when host-resident)
    pub fn device_bytes(&self) -> usize {
        4 * (self.rec.len() + self.tl.len() + self.items.len()) + self.hi.as_ref().map_or(0, |(u, t)| 4 * (u.len() + t.len()))
    }

    /// Move the tables to pinned host memory (freeing the device copies); the engine streams them into a device slot
    /// whenever the set is used. Returns the device bytes freed.
    pub fn to_host(&mut self, ctx: &Arc<CudaContext>, st: &Arc<CudaStream>) -> Result<usize, NufftError> {
        if self.host.is_some() {
            return Ok(0);
        }
        let before = self.device_bytes();
        let hi = match &self.hi {
            Some((u, t)) => Some((pinned_copy(ctx, st, u)?, pinned_copy(ctx, st, t)?)),
            None => None,
        };
        let hi = hi.map(|(u, t)| (HostBuf::Pinned(u), HostBuf::Pinned(t)));
        let h = PointsHost { rec: HostBuf::Pinned(pinned_copy(ctx, st, &self.rec)?), tl: HostBuf::Pinned(pinned_copy(ctx, st, &self.tl)?), items: HostBuf::Pinned(pinned_copy(ctx, st, &self.items)?), hi };
        st.synchronize()?;
        self.rec = st.alloc_zeros(1)?;
        self.tl = st.alloc_zeros(1)?;
        self.items = st.alloc_zeros(1)?;
        self.hi = None;
        self.host = Some(Box::new(h));
        Ok(before.saturating_sub(self.device_bytes()))
    }

    /// Move host-resident tables back to the device. Returns the device bytes allocated.
    pub fn to_device(&mut self, st: &Arc<CudaStream>) -> Result<usize, NufftError> {
        let Some(h) = self.host.take() else { return Ok(0) };
        self.rec = h.rec.to_dev(st)?;
        self.tl = h.tl.to_dev(st)?;
        self.items = h.items.to_dev(st)?;
        self.hi = match &h.hi {
            Some((u, t)) => Some((u.to_dev(st)?, t.to_dev(st)?)),
            None => None,
        };
        st.synchronize()?;
        Ok(self.device_bytes())
    }
}

/// Segment diagonals P_l and interpolator table a_l(τ), built by [`SegNufft::segments`] (e.g. one per encode).
pub struct SegSegments {
    segph: CudaSlice<C32>,
    atab: CudaSlice<C32>,
    tab: (usize, f32, f32),
    /// joint water + fat segments ([`SegNufft::segments_fat`]): the fat spectrum c(τ_l) at the segment nodes
    fatc: Option<CudaSlice<C32>>,
}

/// Fat-fraction maps of the joint water + fat segments (nmap × nv, in [0, 1]), [`SegNufft::fat_maps`]
pub struct SegFatMaps {
    ff: CudaSlice<f32>,
    pub nmap: usize,
}

/// Engine shared by all sample sets of one image grid: FFT plans, generated kernels, grid slots, coil maps.
pub struct SegNufft {
    pub plan: NufftPlan<f32>,
    pub nseg: usize,
    pub order: u32,
    pub nc: usize,
    /// frames and frame slots per sample (B-spline degree + 1, 1 for a static image)
    pub frames: usize,
    pub nf: usize,
    /// coils per batch (each grid slot holds nseg × cb channels)
    pub cb: usize,
    /// image voxel size (m), x fastest
    pub res: [f64; 3],
    /// launch shapes of the forward / adjoint tile kernels (same tile edge and item length: they share the tile lists)
    pub tiles_fwd: TileCfg,
    pub tiles_adj: TileCfg,
    dev: Device,
    /// opt-in shared memory per block of this device
    smax: usize,
    f: Funcs,
    /// frame slots, each a channel-interleaved fine grid [nf2][nf1][nf0][cb·nseg]
    slots: Vec<CudaSlice<C32>>,
    gfft: GridFft,
    smaps: Option<CudaSlice<C32>>,
    /// the grid budget of the options (the multi path plans within it)
    max_grid_bytes: usize,
    /// grid memory the multi-image path may use beyond `max_grid_bytes` (default 0); set before its first call
    pub multi_extra_bytes: usize,
    /// multi-image states, one per image count used
    multi: Vec<Multi>,
    /// device slots for host-resident sample tables ([`SegPoints::to_host`]), created on first use
    stream: Option<Streamer>,
    /// upload host-resident tables of the next set on a second stream while the current one computes
    /// ([`SegNufft::prefetch`], e.g. from `SegSets`); default false: uploads on the compute stream when a set is used
    pub stream_prefetch: bool,
    /// all coils' maps and count while a coil subset is selected ([`SegNufft::select_coils`])
    coils_all: Option<(CudaSlice<C32>, usize)>,
}

/// Pinned staging buffer of one table slot for packed tables, and the event of its last upload.
#[derive(Default)]
struct Staging {
    buf: Option<PinnedHostSlice<u8>>,
    ev: Option<CudaEvent>,
}

impl Staging {
    /// the buffer (≥ n bytes), once its previous upload has completed
    fn get(&mut self, on: &Arc<CudaStream>, n: usize) -> Result<&mut [u8], NufftError> {
        if let Some(ev) = &self.ev {
            ev.synchronize()?;
        }
        if self.buf.as_ref().is_none_or(|b| b.len() < n) {
            self.buf = Some(unsafe { on.context().alloc_pinned_with_flags::<u8>(n.max(1), 0)? });
        }
        Ok(&mut self.buf.as_mut().unwrap().as_mut_slice()?[..n])
    }
    fn record(&mut self, on: &Arc<CudaStream>) -> Result<(), NufftError> {
        if self.ev.is_none() {
            self.ev = Some(on.context().new_event(None)?);
        }
        self.ev.as_ref().unwrap().record(on)?;
        Ok(())
    }
    fn bytes(&self) -> usize {
        self.buf.as_ref().map_or(0, |b| b.len())
    }
}

/// Two device slots for streamed sample tables (the set in use and the next one), an upload stream for prefetching
/// and the events ordering the two streams: `ready[i]` (upload into slot i done, on the upload stream) and `free[i]`
/// (last use of slot i, on the compute stream).
struct Streamer {
    up: Option<Arc<CudaStream>>,
    /// pinned staging for packed tables, per slot
    staging: Vec<Staging>,
    slots: Vec<Option<SegPoints>>,
    owner: Vec<Option<u64>>,
    ready: Vec<Option<CudaEvent>>,
    free: Vec<Option<CudaEvent>>,
    last: usize,
    uploads: usize,
    upload_bytes: usize,
}

/// State of the multi-image path for calls with `k` images: plan, kernels of its coil batch, FFT plans if that batch
/// differs from the engine's, extra grids
struct Multi {
    k: usize,
    /// images per pass, coils per batch, images held in each engine frame slot (the others in `extra`)
    g: usize,
    cb: usize,
    fit: usize,
    tf: TileCfg,
    ta: TileCfg,
    pad: CudaFunction,
    crop: CudaFunction,
    /// pad / crop with a separate fat-fraction map (joint water + fat segments)
    pad_ff: CudaFunction,
    crop_ff: CudaFunction,
    fwd: CudaFunction,
    adj: CudaFunction,
    /// transforms for this pass's channel count (None: the engine's own)
    fft: Option<GridFft>,
    /// (g − fit) × nf grids of cb × NSEG channels, [image − fit][frame slot]
    extra: Vec<CudaSlice<C32>>,
}

/// The fine-grid transforms for one channel count: pruned to the wrapped image block (`hasty_fft::PrunedChanFft3`:
/// the forward never reads outside the block, so the grid needs no zeroing; the inverse writes only the block the crop
/// reads), or the full partial-axis plans (one channel, a length the pruned plan declines, or `HASTY_SEG_NOPRUNE=1`).
struct GridFft {
    pruned: Option<PrunedChanFft3<f32>>,
    full: Option<(FftPlan<f32>, FftPlan<f32>)>,
}

impl GridFft {
    fn new(dev: &Device, nf: [usize; 3], dims: [usize; 3], nch: usize) -> Result<Self, NufftError> {
        let off = std::env::var("HASTY_SEG_NOPRUNE").is_ok_and(|v| v != "0");
        // block per axis [z, y, x]: the pad kernels place image index a at (a − N/2) mod nf
        let keep = [2, 1, 0].map(|a| (dims[a] - dims[a] / 2, dims[a] / 2));
        let pruned = if off || nch < 2 { None } else { PrunedChanFft3::<f32>::new(dev, [nf[2], nf[1], nf[0]], nch, keep, 1.0).ok() };
        let full = if pruned.is_some() {
            None
        } else {
            let fd = [nf[2], nf[1], nf[0], nch];
            let tr = [true, true, true, false];
            Some((FftPlan::<f32>::new_partial(dev, &fd, &tr, Direction::Forward, Norm::None)?, FftPlan::<f32>::new_partial(dev, &fd, &tr, Direction::Inverse, Norm::None)?))
        };
        Ok(Self { pruned, full })
    }

    fn is_pruned(&self) -> bool {
        self.pruned.is_some()
    }


    /// zero the grid before the pad pass if the transform reads outside the image block
    fn prepare<B: DevicePtrMut<C32>>(&self, st: &Arc<CudaStream>, buf: &mut B) -> Result<(), NufftError> {
        if self.pruned.is_none() {
            st.memset_zeros(buf)?;
        }
        Ok(())
    }

    fn forward<B: DevicePtrMut<C32>>(&self, st: &Arc<CudaStream>, buf: &mut B) -> Result<(), NufftError> {
        match (&self.pruned, &self.full) {
            (Some(p), _) => p.forward(st, buf)?,
            (None, Some((f, _))) => f.execute_inplace(st, buf)?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn inverse<B: DevicePtrMut<C32>>(&self, st: &Arc<CudaStream>, buf: &mut B) -> Result<(), NufftError> {
        match (&self.pruned, &self.full) {
            (Some(p), _) => p.inverse(st, buf)?,
            (None, Some((_, i))) => i.execute_inplace(st, buf)?,
            _ => unreachable!(),
        }
        Ok(())
    }
}

fn lc(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(128).max(1), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

fn compile(dev: &Device, plan: &NufftPlan<f32>, sh: Shape, cb: usize, tc: &TileCfg) -> Result<Arc<CudaModule>, NufftError> {
    let defs = std::env::var("HASTY_SEG_DEFS").unwrap_or_default();
    let src = format!(
        "{}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NF {}\n#define NCB {}\n#define TT {}\n#define CPBA {}\n#define CPBF {}\n#define PPF {}\n#define CHUNK {}\n#define LISTMAX {}\n{}",
        defs.split(',').filter(|d| !d.is_empty()).map(|d| format!("#define {d}\n")).collect::<String>(),
        plan.ker.w,
        plan.ker.beta,
        sh.nseg,
        sh.order,
        sh.nf,
        cb,
        tc.tt,
        tc.cpba,
        tc.cpbf,
        tc.ppf,
        tc.chunk,
        tc.max_item,
        include_str!("segnufft.cu")
    );
    let name = format!(
        "segnufft{}_w{}_l{}_o{}_f{}_c{}_t{}_{}_{}_{}_{}_{}",
        defs.replace([',', ' '], "_"),
        plan.ker.w,
        sh.nseg,
        sh.order,
        sh.nf,
        cb,
        tc.tt,
        tc.cpba,
        tc.cpbf,
        tc.ppf,
        tc.chunk,
        tc.max_item
    );
    Ok(dev.load_module(&name, &src, &CompileSettings::default())?)
}

/// source and module name of the multi-image kernels: segnufft.cu (same defines as [`compile`]) + segnufft_multi.cu
fn multi_source(w: usize, beta: f64, sh: Shape, cb: usize, km: usize, tc: &TileCfg) -> (String, String) {
    let defs = std::env::var("HASTY_SEG_DEFS").unwrap_or_default();
    let src = format!(
        "{}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NF {}\n#define NCB {}\n#define TT {}\n#define CPBA {}\n#define CPBF {}\n#define PPF {}\n#define CHUNK {}\n#define LISTMAX {}\n#define KM {}\n{}\n{}",
        defs.split(',').filter(|d| !d.is_empty()).map(|d| format!("#define {d}\n")).collect::<String>(),
        w,
        beta,
        sh.nseg,
        sh.order,
        sh.nf,
        cb,
        tc.tt,
        tc.cpba,
        tc.cpbf,
        tc.ppf,
        tc.chunk,
        tc.max_item,
        km,
        include_str!("segnufft.cu"),
        include_str!("segnufft_multi.cu")
    );
    let name = format!(
        "segnufftm{}_w{}_l{}_o{}_f{}_c{}_k{}_t{}_{}_{}_{}_{}_{}",
        defs.replace([',', ' '], "_"),
        w,
        sh.nseg,
        sh.order,
        sh.nf,
        cb,
        km,
        tc.tt,
        tc.cpba,
        tc.cpbf,
        tc.ppf,
        tc.chunk,
        tc.max_item
    );
    (src, name)
}

fn compile_multi(dev: &Device, plan: &NufftPlan<f32>, sh: Shape, cb: usize, km: usize, tc: &TileCfg) -> Result<Arc<CudaModule>, NufftError> {
    let (src, name) = multi_source(plan.ker.w, plan.ker.beta, sh, cb, km, tc);
    Ok(dev.load_module(&name, &src, &CompileSettings::default())?)
}

/// load a tile kernel and allow it `smem` bytes of dynamic shared memory
fn tile_fn(module: &Arc<CudaModule>, name: &str, smem: usize) -> Result<CudaFunction, NufftError> {
    let f = module.load_function(name)?;
    f.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)?;
    Ok(f)
}

impl SegNufft {
    /// `dims`: image size, x fastest; `res`: voxel size in metres (the quadratic-phase units); `nc`: coils.
    pub fn new(dev: &Device, dims: [usize; 3], res: [f64; 3], nc: usize, opts: SegNufftOpts) -> Result<Self, NufftError> {
        assert!(opts.frames >= 1 && matches!(opts.bspline, 1 | 3));
        assert!(opts.order <= 3, "correction order {} (0, 1, 2 or 3)", opts.order);
        let nf = if opts.frames > 1 { opts.bspline as usize + 1 } else { 1 };
        assert!(opts.frames == 1 || opts.frames >= nf, "need at least {nf} frames for B-spline degree {}", opts.bspline);
        let plan = NufftPlan::<f32>::new(dev, &dims, NufftOpts { sigma: opts.sigma, width: Some(opts.width), ntransf: 1, max_grid_bytes: 1, ..Default::default() })?;
        let ge = plan.grid_elems();
        // coil batch: memory budget over all frame slots
        let per_coil = nf * opts.nseg * ge * std::mem::size_of::<C32>();
        let cb = (opts.max_grid_bytes / per_coil).clamp(1, nc.min(16));
        let sh = Shape { nseg: opts.nseg, order: opts.order, nf };
        let smax = (dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize).max(SMEM_DEFAULT);
        let tc = TileCfg::new(sh, cb, SMEM_DEFAULT);
        let module = compile(dev, &plan, sh, cb, &tc)?;
        let f = Funcs { pad: module.load_function("seg_pad")?, crop: module.load_function("seg_crop")?, neg: module.load_function("seg_neg")?, fwd: tile_fn(&module, "seg_forward_t", tc.smem(sh)[1])?, adj: tile_fn(&module, "seg_adjoint_t", tc.smem(sh)[0])? };
        let nch = cb * opts.nseg;
        let gfft = GridFft::new(dev, [plan.nf[0], plan.nf[1], plan.nf[2]], dims, nch)?;
        let stream = dev.default_stream();
        let slots = (0..nf).map(|_| stream.alloc_zeros::<C32>(nch * ge)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan,
            nseg: opts.nseg,
            order: opts.order,
            nc,
            frames: opts.frames,
            nf,
            cb,
            res,
            tiles_fwd: tc,
            tiles_adj: tc,
            dev: dev.clone(),
            smax,
            f,
            slots,
            gfft,
            smaps: None,
            max_grid_bytes: opts.max_grid_bytes,
            multi_extra_bytes: 0,
            multi: Vec::new(),
            stream: None,
            stream_prefetch: false,
            coils_all: None,
        })
    }

    fn shape(&self) -> Shape {
        Shape { nseg: self.nseg, order: self.order, nf: self.nf }
    }

    /// whether the fine-grid transforms are pruned to the image block ([`hasty_fft::PrunedChanFft3`])
    pub fn grid_fft_pruned(&self) -> bool {
        self.gfft.is_pruned()
    }

    /// voxels of one frame image
    pub fn nv(&self) -> usize {
        self.plan.nmodes()
    }

    /// coil sensitivities, coil-major (nc × nv), x fastest
    pub fn set_maps(&mut self, st: &Arc<CudaStream>, smaps: &[C32]) -> Result<(), NufftError> {
        assert_eq!(smaps.len(), self.nc * self.nv());
        self.smaps = Some(st.clone_htod(smaps)?);
        Ok(())
    }

    /// segment diagonals P_l (nseg × nv) and interpolator table a_l(τ) ([ntab][nseg], τ_i = t0 + i·dt)
    pub fn segments(&self, st: &Arc<CudaStream>, segph: &[C32], atab: &[C32], t0: f64, dt: f64) -> Result<SegSegments, NufftError> {
        assert_eq!(segph.len(), self.nseg * self.nv());
        assert_eq!(atab.len() % self.nseg, 0);
        Ok(SegSegments { segph: st.clone_htod(segph)?, atab: st.clone_htod(atab)?, tab: (atab.len() / self.nseg, t0 as f32, dt as f32), fatc: None })
    }

    /// Restrict the engine to a subset of its coils (stochastic coil sampling): the maps of `idx` are gathered (device
    /// copies) into a compact buffer and forward / adjoint then use idx.len() coils, the data coil-major in idx order.
    /// `None` restores all coils. Select `None` before [`set_maps`](Self::set_maps) or [`swap_smaps`](Self::swap_smaps).
    pub fn select_coils(&mut self, st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<(), NufftError> {
        let nv = self.nv();
        match idx {
            None => {
                if let Some((all, nc)) = self.coils_all.take() {
                    self.smaps = Some(all);
                    self.nc = nc;
                }
            }
            Some(idx) => {
                if self.coils_all.is_none() {
                    let all = self.smaps.take().expect("set_maps first");
                    self.coils_all = Some((all, self.nc));
                }
                let nca = self.coils_all.as_ref().unwrap().1;
                assert!(!idx.is_empty() && idx.iter().all(|&c| c < nca), "select_coils: coil index out of range");
                let mut buf = match self.smaps.take() {
                    Some(b) if b.len() == idx.len() * nv => b,
                    _ => st.alloc_zeros::<C32>(idx.len() * nv)?,
                };
                let all = &self.coils_all.as_ref().unwrap().0;
                for (i, &c) in idx.iter().enumerate() {
                    st.memcpy_dtod(&all.slice(c * nv..(c + 1) * nv), &mut buf.slice_mut(i * nv..(i + 1) * nv))?;
                }
                self.smaps = Some(buf);
                self.nc = idx.len();
            }
        }
        Ok(())
    }

    /// coils of the maps (all of them, also while a subset is selected)
    pub fn nc_all(&self) -> usize {
        self.coils_all.as_ref().map_or(self.nc, |a| a.1)
    }

    /// Exchange the coil maps with `s` (e.g. maps carrying a per-encode phase); call again to restore.
    pub fn swap_smaps(&mut self, s: &mut CudaSlice<C32>) {
        let m = self.smaps.as_mut().expect("set_maps first");
        assert_eq!(s.len(), m.len());
        std::mem::swap(m, s);
    }

    /// Samples: `coords` k·Δ (radians per voxel, x/y/z), readout time `tau` (s), constant phase `phi0` (rad), the
    /// residual quadratic phase coefficients `dc` = [xx, yy, zz, xy, xz, yz] (rad/m², ψ = ρᵀΔCρ, ρ in m), SoA (6 × m),
    /// and (with frames > 1) the acquisition time `t` in frame units, periodic in [0, frames): frame f's node is t = f.
    pub fn points(&mut self, st: &Arc<CudaStream>, coords: [&[f32]; 3], tau: &[f32], phi0: &[f32], dc: &[f32], t: Option<&[f32]>) -> Result<SegPoints, NufftError> {
        let m = coords[0].len();
        assert!(tau.len() == m && phi0.len() == m);
        assert!(self.frames == 1 || t.is_some_and(|t| t.len() == m), "frames > 1 need sample times");
        let dcd: Vec<CudaSlice<f32>> = coords.iter().map(|c| st.clone_htod(*c)).collect::<Result<_, _>>()?;
        self.plan.set_points(st, &[&dcd[0], &dcd[1], &dcd[2]], m)?;
        let p = self.plan.points().unwrap();
        let (sc, perm) = (st.clone_dtoh(&p.sc)?, st.clone_dtoh(&p.perm)?);
        // ΔC in kernel-argument units: ∂/∂k_a = −g_a ∂/∂z_a, g_a = res_a nf_a / (π W); off-diagonals ×2. The second-order
        // coefficients (−½ (ρᵀΔCρ)² over the 15 degree-4 monomials) are formed in the kernel from these six.
        let w = self.plan.ker.w as f64;
        let g: [f64; 3] = [0, 1, 2].map(|a| self.res[a] * self.plan.nf[a] as f64 / (std::f64::consts::PI * w));
        let fac = [g[0] * g[0], g[1] * g[1], g[2] * g[2], 2.0 * g[0] * g[1], 2.0 * g[0] * g[2], 2.0 * g[1] * g[2]];
        if self.order >= 1 {
            assert_eq!(dc.len(), 6 * m);
        }
        // frame interval of every sample
        let nfr = self.frames;
        let mut ival = vec![0u32; m];
        let mut rec = vec![0f32; 12 * m];
        for j in 0..m {
            let r = &mut rec[12 * j..12 * j + 12];
            r[..3].copy_from_slice(&[sc[j], sc[m + j], sc[2 * m + j]]);
            r[3] = tau[j];
            r[4] = phi0[j];
            if self.order >= 1 {
                for q in 0..6 {
                    r[5 + q] = (dc[q * m + j] as f64 * fac[q]) as f32;
                }
            }
            if let (Some(t), true) = (t, nfr > 1) {
                let tt = (t[j] as f64).rem_euclid(nfr as f64);
                let i = (tt.floor() as usize).min(nfr - 1);
                ival[j] = i as u32;
                r[11] = (tt - i as f64) as f32;
            }
        }
        // tile lists per interval (samples kept in bin-sorted order)
        let mut tl_all: Vec<u32> = Vec::new();
        let mut items_all: Vec<[i32; 4]> = Vec::new();
        let mut ivals = Vec::with_capacity(nfr);
        for i in 0..nfr as u32 {
            let pi: Vec<u32> = perm.iter().copied().filter(|&j| ival[j as usize] == i).collect();
            let (tl, mut items, split) = tile_lists(&sc, m, &pi, self.plan.nf, self.plan.ker.w, self.tiles_fwd.tt, self.tiles_fwd.max_item);
            let base = tl_all.len() as i32;
            items.iter_mut().for_each(|it| it[1] += base);
            ivals.push(Interval { first: items_all.len(), count: items.len(), split });
            tl_all.extend_from_slice(&tl);
            items_all.extend_from_slice(&items);
        }
        Ok(SegPoints { rec: st.clone_htod(&rec)?, tl: st.clone_htod(&tl_all)?, items: st.clone_htod(items_all.as_flattened())?, ivals, m, hi: None, host: None, id: POINTS_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) })
    }

    fn mdims(&self) -> MDims {
        MDims { n0: self.plan.modes[0] as i32, n1: self.plan.modes[1] as i32, n2: self.plan.modes[2] as i32, fftord: 0 }
    }

    /// frames read by interval i, by frame offset o
    fn frames_of(&self, i: usize) -> Vec<usize> {
        let off = -((self.nf as i64 - 2).max(0) / 2);
        (0..self.nf).map(|o| (i as i64 + off + o as i64).rem_euclid(self.frames as i64) as usize).collect()
    }

    /// slot i ← FFT(pad(S_c P_l x_f)) for the coil batch
    #[allow(clippy::too_many_arguments)]
    fn load_frame(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, slot: usize, x: &CudaSlice<C32>, f: usize, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let nv = self.nv();
        let (md, d, c0_i, ncb_i) = (self.mdims(), self.plan.dims(), c0 as i32, ncb as i32);
        self.gfft.prepare(st, &mut self.slots[slot])?;
        {
            let xf = x.slice(f * nv..(f + 1) * nv);
            let (gp, _g) = self.slots[slot].device_ptr_mut(st);
            unsafe {
                st.launch_builder(&self.f.pad)
                    .arg(&xf)
                    .arg(self.smaps.as_ref().unwrap())
                    .arg(&seg.segph)
                    .arg(&gp)
                    .arg(&self.plan.invphi[0])
                    .arg(&self.plan.invphi[1])
                    .arg(&self.plan.invphi[2])
                    .arg(&c0_i)
                    .arg(&ncb_i)
                    .arg(&md)
                    .arg(&d)
                    .launch(lc(nv))?;
            }
        }
        self.gfft.forward(st, &mut self.slots[slot])?;
        Ok(())
    }

    /// x_f += crop(IFFT(slot)) summed over the batch coils and segments
    #[allow(clippy::too_many_arguments)]
    fn flush_frame(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, slot: usize, x: &mut CudaSlice<C32>, f: usize, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let nv = self.nv();
        let (md, d, c0_i, ncb_i, acc) = (self.mdims(), self.plan.dims(), c0 as i32, ncb as i32, 1i32);
        self.gfft.inverse(st, &mut self.slots[slot])?;
        let (gp, _g) = self.slots[slot].device_ptr(st);
        let mut xf = x.slice_mut(f * nv..(f + 1) * nv);
        unsafe {
            st.launch_builder(&self.f.crop)
                .arg(&gp)
                .arg(self.smaps.as_ref().unwrap())
                .arg(&seg.segph)
                .arg(&mut xf)
                .arg(&self.plan.invphi[0])
                .arg(&self.plan.invphi[1])
                .arg(&self.plan.invphi[2])
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&acc)
                .arg(&md)
                .arg(&d)
                .launch(lc(nv))?;
        }
        Ok(())
    }

    /// Slot assignment for interval i: frames already held stay; missing ones take slots holding frames not needed.
    /// Returns (slot per frame offset, slots that were (re)assigned, evicted (slot, frame) pairs).
    fn assign(held: &mut [Option<usize>], need: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<(usize, usize)>) {
        let mut fresh = Vec::new();
        let mut evicted = Vec::new();
        let map = need
            .iter()
            .map(|&f| {
                if let Some(s) = held.iter().position(|&h| h == Some(f)) {
                    return s;
                }
                let s = held.iter().position(|h| h.is_none_or(|g| !need.contains(&g))).expect("frame slots");
                if let Some(g) = held[s] {
                    evicted.push((s, g));
                }
                held[s] = Some(f);
                fresh.push(s);
                s
            })
            .collect();
        (map, fresh, evicted)
    }

    fn grids(&self, st: &Arc<CudaStream>, map: &[usize]) -> Grids {
        let mut g = Grids { p: [0; 4] };
        for (o, &s) in map.iter().enumerate() {
            let (p, _guard) = self.slots[s].device_ptr(st);
            g.p[o] = p;
        }
        g
    }

    /// y (nc × m) = A x, x = frames × nv (frame-major)
    pub fn forward(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        self.forward_in(st, seg, pts, x, None, y)
    }

    /// r (nc × m) = A x − y: the gather accumulates onto −y instead of zero (fused residual: no separate pass over the
    /// data and no second buffer of the data size; equal to `forward` then subtract up to f32 summation order).
    pub fn forward_residual(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, x: &CudaSlice<C32>, y: &CudaSlice<C32>, r: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        assert!(y.len() >= self.nc * pts.m, "forward_residual: data of {} values, need {} × {}", y.len(), self.nc, pts.m);
        self.forward_in(st, seg, pts, x, Some(y), r)
    }

    /// y[rows] ← −init[rows] (or 0)
    fn init_rows(&self, st: &Arc<CudaStream>, init: Option<&CudaSlice<C32>>, y: &mut CudaSlice<C32>, lo: usize, hi: usize) -> Result<(), NufftError> {
        match init {
            None => st.memset_zeros(&mut y.slice_mut(lo..hi))?,
            Some(d) => {
                let n = (hi - lo) as u64;
                let src = d.slice(lo..hi);
                let mut dst = y.slice_mut(lo..hi);
                unsafe { st.launch_builder(&self.f.neg).arg(&src).arg(&mut dst).arg(&n).launch(lc(hi - lo))? };
            }
        }
        Ok(())
    }

    fn forward_in(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, x: &CudaSlice<C32>, init: Option<&CudaSlice<C32>>, y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.forward_in(st, seg, p, x, init, y));
        }
        if self.order >= 3 {
            // order 3 lives in the multi-image kernels
            let i = self.multi_index(st, 1)?;
            let mut ms = self.multi.swap_remove(i);
            let r = self.forward_multi_in(&mut ms, st, seg, None, pts, &[x], &[None], init, y);
            self.multi.push(ms);
            return r;
        }
        let (nv, m) = (self.nv(), pts.m);
        assert!(x.len() >= self.frames * nv && y.len() >= self.nc * m);
        let (fwd, tc) = (self.f.fwd.clone(), self.tiles_fwd);
        let mut c0 = 0;
        while c0 < self.nc {
            let ncb = self.cb.min(self.nc - c0);
            self.init_rows(st, init, y, c0 * m, (c0 + ncb) * m)?;
            let mut held: Vec<Option<usize>> = vec![None; self.nf];
            for i in 0..self.frames {
                let need = self.frames_of(i);
                let (map, fresh, _) = Self::assign(&mut held, &need);
                for s in fresh {
                    self.load_frame(st, seg, s, x, held[s].unwrap(), c0, ncb)?;
                }
                let g = self.grids(st, &map);
                self.launch_fwd(st, &fwd, &tc, seg, pts, i, g, y, c0, ncb)?;
            }
            c0 += ncb;
        }
        Ok(())
    }

    /// x (frames × nv) = A^H y
    pub fn adjoint(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.adjoint(st, seg, p, y, x));
        }
        if self.order >= 3 {
            return self.adjoint_multi(st, seg, pts, y, &[None], &mut [x]);
        }
        let (nv, m) = (self.nv(), pts.m);
        assert!(x.len() >= self.frames * nv && y.len() >= self.nc * m);
        st.memset_zeros(&mut x.slice_mut(0..self.frames * nv))?;
        let (adj, tc) = (self.f.adj.clone(), self.tiles_adj);
        let mut c0 = 0;
        while c0 < self.nc {
            let ncb = self.cb.min(self.nc - c0);
            let mut held: Vec<Option<usize>> = vec![None; self.nf];
            let mut stale = vec![false; self.nf];   // slot holds partial sums (add instead of store)
            for i in 0..self.frames {
                let need = self.frames_of(i);
                let (map, fresh, evicted) = Self::assign(&mut held, &need);
                for (s, g) in evicted {
                    self.flush_frame(st, seg, s, x, g, c0, ncb)?;
                }
                for &s in &fresh {
                    stale[s] = false;
                }
                let split = pts.ivals[i].split;
                let mut addmask = 0i32;
                for (o, &s) in map.iter().enumerate() {
                    if !stale[s] && split {
                        st.memset_zeros(&mut self.slots[s])?;
                        stale[s] = true;
                    }
                    if stale[s] {
                        addmask |= 1 << o;
                    }
                }
                let g = self.grids(st, &map);
                self.launch_adj(st, &adj, &tc, seg, pts, i, g, addmask, y, c0, ncb)?;
                map.iter().for_each(|&s| stale[s] = true);
            }
            for s in 0..self.nf {
                if let Some(g) = held[s] {
                    self.flush_frame(st, seg, s, x, g, c0, ncb)?;
                }
            }
            c0 += ncb;
        }
        Ok(())
    }

    /// tile gather of interval i for coils c0..c0+ncb: y[c0.., :] += A_tiles(grids)
    #[allow(clippy::too_many_arguments)]
    fn launch_fwd(&self, st: &Arc<CudaStream>, f: &CudaFunction, tc: &TileCfg, seg: &SegSegments, pts: &SegPoints, i: usize, g: Grids, y: &mut CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let iv = pts.ivals[i];
        if iv.count == 0 {
            return Ok(());
        }
        let (t0, dt, ntab, m_i, c0_i, ncb_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, self.plan.dims());
        let items = pts.items.slice(4 * iv.first..4 * (iv.first + iv.count));
        let (yp, _gy) = y.device_ptr_mut(st);
        unsafe {
            st.launch_builder(f)
                .arg(&pts.rec)
                .arg(&pts.tl)
                .arg(&items)
                .arg(&seg.atab)
                .arg(&ntab)
                .arg(&t0)
                .arg(&dt)
                .arg(&g)
                .arg(&yp)
                .arg(&m_i)
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&d)
                .launch(LaunchConfig { grid_dim: (iv.count as u32, ncb.div_ceil(tc.cpbf) as u32, 1), block_dim: ((tc.chunk * tc.cpbf * tc.ppf * self.nf) as u32, 1, 1), shared_mem_bytes: tc.smem(self.shape())[1] as u32 })?;
        }
        Ok(())
    }

    /// tile scatter of interval i for coils c0..c0+ncb into the grid slots (bit o of `addmask`: add instead of store)
    #[allow(clippy::too_many_arguments)]
    fn launch_adj(&self, st: &Arc<CudaStream>, f: &CudaFunction, tc: &TileCfg, seg: &SegSegments, pts: &SegPoints, i: usize, g: Grids, addmask: i32, y: &CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let iv = pts.ivals[i];
        let (t0, dt, ntab, m_i, c0_i, ncb_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, self.plan.dims());
        let items = pts.items.slice(4 * iv.first..4 * (iv.first + iv.count));
        unsafe {
            st.launch_builder(f)
                .arg(&pts.rec)
                .arg(&pts.tl)
                .arg(&items)
                .arg(&seg.atab)
                .arg(&ntab)
                .arg(&t0)
                .arg(&dt)
                .arg(y)
                .arg(&m_i)
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&g)
                .arg(&addmask)
                .arg(&d)
                .launch(LaunchConfig { grid_dim: (iv.count as u32, ncb.div_ceil(tc.cpba) as u32, 1), block_dim: ((tc.tt.pow(3) * self.nf * tc.cpba) as u32, 1, 1), shared_mem_bytes: tc.smem(self.shape())[0] as u32 })?;
        }
        Ok(())
    }

    /// Use the given launch shapes (e.g. tuned on another set with the same trajectory type). Tile edge and item length
    /// must match the current tile lists.
    pub fn set_tiles(&mut self, fwd: TileCfg, adj: TileCfg) -> Result<(), NufftError> {
        assert!(fwd.tt == self.tiles_fwd.tt && adj.tt == fwd.tt && fwd.max_item == self.tiles_fwd.max_item && adj.max_item == fwd.max_item);
        let sh = self.shape();
        assert!(fwd.fits(sh, self.smax) && adj.fits(sh, self.smax), "tile shapes do not fit {sh:?}");
        self.f.fwd = tile_fn(&compile(&self.dev, &self.plan, sh, self.cb, &fwd)?, "seg_forward_t", fwd.smem(sh)[1])?;
        self.f.adj = tile_fn(&compile(&self.dev, &self.plan, sh, self.cb, &adj)?, "seg_adjoint_t", adj.smem(sh)[0])?;
        self.tiles_fwd = fwd;
        self.tiles_adj = adj;
        Ok(())
    }

    /// Time the tile kernels on the given set (busiest interval) over candidate launch shapes (coil groups, lanes
    /// per sample, chunk) and keep the fastest for each direction. Kernels come from the disk cache after the first run;
    /// the operator is applied hundreds of times, so this pays for itself. Returns the chosen (forward, adjoint) kernel
    /// times in ms.
    pub fn autotune(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints) -> Result<(f32, f32), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.autotune(st, seg, p));
        }
        if self.order >= 3 {
            return self.autotune_multi(st, seg, pts, 1);
        }
        assert!(pts.m > 0, "autotune needs points");
        let sh = self.shape();
        let (tt, mi, cb) = (self.tiles_fwd.tt, self.tiles_fwd.max_item, self.cb);
        let ncb = cb.min(self.nc);
        let busiest = (0..pts.ivals.len()).max_by_key(|&i| pts.ivals[i].count).unwrap();
        let map: Vec<usize> = (0..self.nf).collect();
        let g = self.grids(st, &map);
        let groups = [1usize, 2, 3, 4, 6, 8, 12, 16];
        let mut ys = st.alloc_zeros::<C32>(cb * pts.m)?;
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ev = (self.dev.ctx.new_event(flags)?, self.dev.ctx.new_event(flags)?);
        let time = |run: &mut dyn FnMut() -> Result<(), NufftError>| -> Result<f32, NufftError> {
            run()?;
            let mut best = f32::INFINITY;
            for _ in 0..3 {
                ev.0.record(st)?;
                run()?;
                ev.1.record(st)?;
                ev.1.synchronize()?;
                best = best.min(ev.0.elapsed_ms(&ev.1)?);
            }
            Ok(best)
        };
        let mut best_f = (f32::INFINITY, self.tiles_fwd);
        for &cpbf in groups.iter().filter(|&&c| c <= cb) {
            for ppf in [1usize, 2, 4] {
                let Some(tc) = TileCfg::fit(sh, tt, self.tiles_adj.cpba, cpbf, ppf, mi, self.smax) else { continue };
                if tc.chunk < 8 {
                    continue;
                }
                let f = tile_fn(&compile(&self.dev, &self.plan, sh, cb, &tc)?, "seg_forward_t", tc.smem(sh)[1])?;
                let t = time(&mut || self.launch_fwd(st, &f, &tc, seg, pts, busiest, g, &mut ys, 0, ncb))?;
                if t < best_f.0 {
                    best_f = (t, tc);
                }
            }
        }
        let mut best_a = (f32::INFINITY, self.tiles_adj);
        for &cpba in groups.iter().filter(|&&c| c <= cb) {
            let Some(tc) = TileCfg::fit(sh, tt, cpba, best_f.1.cpbf, best_f.1.ppf, mi, self.smax) else { continue };
            let f = tile_fn(&compile(&self.dev, &self.plan, sh, cb, &tc)?, "seg_adjoint_t", tc.smem(sh)[0])?;
            let t = time(&mut || self.launch_adj(st, &f, &tc, seg, pts, busiest, g, 0, &ys, 0, ncb))?;
            if t < best_a.0 {
                best_a = (t, tc);
            }
        }
        self.set_tiles(best_f.1, best_a.1)?;
        Ok((best_f.0, best_a.0))
    }
}

/// Time segments for a phase ψ(voxel, τ) (e.g. ω(ρ)τ + ρᵀC̄(τ)ρ): diagonals P_l = exp(−iψ(·, τ_l)) at `nseg` uniformly
/// spaced τ_l in [t_min, t_max], and least-squares interpolators a(τ) on an `ntab`-point table (fitted over the voxel
/// subset `fit_voxels`, small ridge). Returns (P [nseg × nv], a [ntab × nseg], t0, dt).
pub fn build_segments(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    psi: &(dyn Fn(usize, f64) -> f64 + Sync),
) -> (Vec<C32>, Vec<C32>, f64, f64) {
    use nalgebra::DVector;
    use num_complex::Complex64;
    let tl: Vec<f64> = (0..nseg).map(|l| if nseg == 1 { 0.5 * (t_min + t_max) } else { t_min + (t_max - t_min) * l as f64 / (nseg - 1) as f64 }).collect();
    let segph = seg_diagonals(nv, &tl, &|v, t| {
        let (s, c) = (-psi(v, t)).sin_cos();
        Complex64::new(c, s)
    });
    let b = seg_basis(fit_voxels.len(), &tl, &|r, t| Complex64::from_polar(1.0, -psi(fit_voxels[r], t)));
    let atab = ls_interpolators(&b, nseg, t_min, t_max, ntab, &|t| DVector::<Complex64>::from_iterator(fit_voxels.len(), fit_voxels.iter().map(|&v| Complex64::from_polar(1.0, -psi(v, t)))));
    (segph, atab, t_min, (t_max - t_min) / (ntab - 1) as f64)
}

/// Time segments from the optimal rank-L separable approximation of the time function g(v, τ) over the fit voxels
/// (instead of diagonals at L fixed times, [`build_segments`] / [`build_segments_complex`]): with G[r, i] = g(v_r, τ_i)
/// on `ns` readout times, the top-L eigenvectors V of GᴴG (= the right singular vectors of G) give the temporal basis;
/// every voxel's diagonals are its projection D_l(v) = Σ_i g(v, τ_i) V[i, l] (least squares in that basis), and the
/// interpolators a(τ) are refitted by least squares at every table entry for these diagonals ([`build_segments`]'s
/// criterion). Since the fixed-time interpolators are already least-squares optimal for their diagonals, this changes
/// only the diagonals: it is the minimum of Σ_{r,i} |g − Σ_l D_l a_l|² over both, i.e. the fewest segments for a given
/// error. Diagonals are complex (not unit modulus); the operator's pad / crop take any complex diagonals.
/// Same output layout as [`build_segments`]: (diagonals [nseg × nv], table [ntab × nseg], t0, dt).
#[allow(clippy::too_many_arguments)]
pub fn build_segments_svd(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    ns: usize,
    g: &(dyn Fn(usize, f64) -> num_complex::Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, f64, f64) {
    build_segments_svd_weighted(nv, nseg, t_min, t_max, ntab, fit_voxels, None, None, ns, g)
}

/// [`build_segments_svd`] with weights: per fit voxel `voxel_w` (e.g. the expected signal energy |x(v)|² Σ_c |S_c(v)|²
/// from a warm start, the C++ `L2Energy` weighting) and per readout time `time_w(τ)` (e.g. the measured k-space data
/// energy at that ADC time: a centre-out readout carries most of its energy early, so the model error late in the
/// readout — where the accumulated phase and hence the rank is largest — matters less in the data norm). The basis
/// minimises Σ_r w_r Σ_i ω(τ_i) |g(v_r, τ_i) − Σ_l D_l(v_r) b_l(τ_i)|²: with G̃ = W^½ G Ω^½ and its leading right
/// singular vectors Ṽ, b_l(τ_i) = conj(Ṽ_il) / √ω_i and D_l(v) = Σ_i g(v, τ_i) √ω_i Ṽ_il (each voxel's ω-weighted
/// projection). The interpolators are then refitted per table entry (voxel-weighted least squares at that τ). None =
/// uniform.
#[allow(clippy::too_many_arguments)]
pub fn build_segments_svd_weighted(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    voxel_w: Option<&[f64]>,
    time_w: Option<&(dyn Fn(f64) -> f64 + Sync)>,
    ns: usize,
    g: &(dyn Fn(usize, f64) -> num_complex::Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, f64, f64) {
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex64;
    use rayon::prelude::*;
    if let Some(w) = voxel_w {
        assert_eq!(w.len(), fit_voxels.len(), "one weight per fit voxel");
    }
    let sw = |r: usize| voxel_w.map_or(1.0, |w| w[r].max(0.0).sqrt());
    let ns = ns.max(nseg + 1).max(2);
    let ts: Vec<f64> = (0..ns).map(|i| t_min + (t_max - t_min) * i as f64 / (ns - 1) as f64).collect();
    // √ω(τ_i), floored so no readout time drops out entirely
    let st_w: Vec<f64> = {
        let w: Vec<f64> = ts.iter().map(|&t| time_w.map_or(1.0, |f| f(t).max(0.0))).collect();
        let wmax = w.iter().cloned().fold(0.0f64, f64::max).max(1e-300);
        w.iter().map(|&x| x.max(1e-6 * wmax).sqrt()).collect()
    };
    // Gram GᴴG over the fit voxels (ns × ns, Hermitian), in parallel over voxel chunks
    let zero = || vec![Complex64::new(0.0, 0.0); ns * ns];
    let idx: Vec<usize> = (0..fit_voxels.len()).collect();
    let gram = idx
        .par_chunks(1024)
        .map(|chunk| {
            let mut acc = zero();
            let mut row = vec![Complex64::new(0.0, 0.0); ns];
            for &r in chunk {
                let (v, w) = (fit_voxels[r], sw(r));
                for (i, t) in ts.iter().enumerate() {
                    row[i] = g(v, *t) * (w * st_w[i]);
                }
                for i in 0..ns {
                    let ci = row[i].conj();
                    for j in i..ns {
                        acc[i * ns + j] += ci * row[j];
                    }
                }
            }
            acc
        })
        .reduce(zero, |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        });
    let mut c = DMatrix::<Complex64>::zeros(ns, ns);
    for i in 0..ns {
        for j in i..ns {
            c[(i, j)] = gram[i * ns + j];
            c[(j, i)] = gram[i * ns + j].conj();
        }
    }
    let eig = nalgebra::SymmetricEigen::new(c);
    let mut order: Vec<usize> = (0..ns).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
    // V[:, l] (ns × nseg): leading right singular vectors of G
    let v = DMatrix::<Complex64>::from_fn(ns, nseg, |i, l| eig.eigenvectors[(i, order[l])]);
    // diagonals of every voxel: D_l(v) = Σ_i g(v, τ_i) V[i, l]
    let mut segph = vec![C32::new(0.0, 0.0); nseg * nv];
    let rows: Vec<Vec<C32>> = (0..nv)
        .into_par_iter()
        .with_min_len(512)
        .map(|vx| {
            let gv: Vec<Complex64> = ts.iter().zip(&st_w).map(|(&t, &w)| g(vx, t) * w).collect();
            (0..nseg)
                .map(|l| {
                    let z: Complex64 = (0..ns).map(|i| gv[i] * v[(i, l)]).sum();
                    C32::new(z.re as f32, z.im as f32)
                })
                .collect()
        })
        .collect();
    for (vx, r) in rows.iter().enumerate() {
        for l in 0..nseg {
            segph[l * nv + vx] = r[l];
        }
    }
    // least-squares interpolators for these diagonals at every table entry (fit voxels)
    let b = DMatrix::<Complex64>::from_fn(fit_voxels.len(), nseg, |r, l| {
        let z = segph[l * nv + fit_voxels[r]];
        Complex64::new(z.re as f64, z.im as f64) * sw(r)
    });
    let atab = ls_interpolators(&b, nseg, t_min, t_max, ntab, &|t| DVector::<Complex64>::from_iterator(fit_voxels.len(), fit_voxels.iter().enumerate().map(|(r, &vx)| g(vx, t) * sw(r))));
    (segph, atab, t_min, (t_max - t_min) / (ntab - 1) as f64)
}

/// diagonals P_l(v) = f(v, τ_l), [nseg × nv] (parallel over l and voxel blocks)
fn seg_diagonals(nv: usize, tl: &[f64], f: &(dyn Fn(usize, f64) -> num_complex::Complex64 + Sync)) -> Vec<C32> {
    use rayon::prelude::*;
    let mut segph = vec![C32::new(0.0, 0.0); tl.len() * nv];
    if nv == 0 {
        return segph;
    }
    segph.par_chunks_mut(nv).zip(tl.par_iter()).for_each(|(row, &t)| {
        row.par_chunks_mut(4096).enumerate().for_each(|(b, blk)| {
            for (i, o) in blk.iter_mut().enumerate() {
                let z = f(b * 4096 + i, t);
                *o = C32::new(z.re as f32, z.im as f32);
            }
        })
    });
    segph
}

/// the fit basis B[r, l] = f(r, τ_l) (rows in parallel)
fn seg_basis(rows: usize, tl: &[f64], f: &(dyn Fn(usize, f64) -> num_complex::Complex64 + Sync)) -> nalgebra::DMatrix<num_complex::Complex64> {
    use rayon::prelude::*;
    let nseg = tl.len();
    // column-major: element (r, l) at l · rows + r
    let mut v = vec![num_complex::Complex64::new(0.0, 0.0); rows * nseg];
    if rows > 0 {
        v.par_chunks_mut(rows).zip(tl.par_iter()).for_each(|(col, &t)| col.iter_mut().enumerate().for_each(|(r, o)| *o = f(r, t)));
    }
    nalgebra::DMatrix::from_vec(rows, nseg, v)
}

// ───────────────────────────────────────────────────────────── multi-image path and correction order 3 (additions)

/// joint water + fat diagonals of a call: the maps, which map (None: map f for frame f) and the nodes' fat spectrum
#[derive(Clone, Copy)]
struct FatArg<'a> {
    maps: &'a SegFatMaps,
    map: Option<usize>,
    cl: &'a CudaSlice<C32>,
}

fn wptrs(st: &Arc<CudaStream>, ws: &[Option<&CudaSlice<C32>>]) -> WPtrs {
    let mut p = WPtrs { p: [0; 4] };
    for (k, w) in ws.iter().enumerate() {
        if let Some(w) = w {
            p.p[k] = w.device_ptr(st).0;
        }
    }
    p
}

impl SegNufft {
    /// grid bytes of one coil of one image over all frame slots
    fn per_coil_bytes(&self) -> usize {
        self.nf * self.nseg * self.plan.grid_elems() * std::mem::size_of::<C32>()
    }

    /// the multi-image state for `k` images (planned and built on first use)
    fn multi_index(&mut self, st: &Arc<CudaStream>, k: usize) -> Result<usize, NufftError> {
        if let Some(i) = self.multi.iter().position(|m| m.k == k) {
            return Ok(i);
        }
        let sh = self.shape();
        let (tt, mi, smax) = (self.tiles_fwd.tt, self.tiles_fwd.max_item, self.smax);
        let pc = self.per_coil_bytes();
        let budget = self.max_grid_bytes.max(self.cb * pc).saturating_add(self.multi_extra_bytes);
        let (g, cbm, fit) = multi_plan(k, self.cb, self.nc, pc, budget, &|g, c| TileCfg::new_multi(sh, g, c, tt, mi, smax).is_some());
        let tc = TileCfg::new_multi(sh, g, cbm, tt, mi, smax).expect("multi-image tile shape");
        let module = compile_multi(&self.dev, &self.plan, sh, cbm, g, &tc)?;
        let [sa, sf] = tc.smem_multi(sh, g);
        let (pad, crop) = (module.load_function("seg_pad")?, module.load_function("seg_crop")?);
        let (pad_ff, crop_ff) = (module.load_function("seg_pad_ff")?, module.load_function("seg_crop_ff")?);
        let (fwd, adj) = (tile_fn(&module, "seg_forward_mt", sf)?, tile_fn(&module, "seg_adjoint_mt", sa)?);
        let nch = cbm * self.nseg;
        let fft = if cbm == self.cb { None } else { Some(GridFft::new(&self.dev, [self.plan.nf[0], self.plan.nf[1], self.plan.nf[2]], [self.plan.modes[0], self.plan.modes[1], self.plan.modes[2]], nch)?) };
        let ge = self.plan.grid_elems();
        let extra = (0..(g - fit) * self.nf).map(|_| st.alloc_zeros::<C32>(nch * ge)).collect::<Result<Vec<_>, _>>()?;
        self.multi.push(Multi { k, g, cb: cbm, fit, tf: tc, ta: tc, pad, crop, pad_ff, crop_ff, fwd, adj, fft, extra });
        Ok(self.multi.len() - 1)
    }

    /// Plan of the multi-image path for `k` images: (images per tile pass, coils per batch, extra grid bytes beyond the
    /// engine's own slots). Builds the state (kernels, FFT plans, extra grids) if needed.
    pub fn multi_info(&mut self, st: &Arc<CudaStream>, k: usize) -> Result<(usize, usize, usize), NufftError> {
        let i = self.multi_index(st, k)?;
        let m = &self.multi[i];
        Ok((m.g, m.cb, m.extra.iter().map(|b| b.len() * std::mem::size_of::<C32>()).sum()))
    }

    /// drop the multi-image states (kernels, extra grids), e.g. to re-plan after changing `multi_extra_bytes`
    pub fn clear_multi(&mut self) {
        self.multi.clear();
    }

    /// (in the engine slots, buffer index, element offset) of image `kk` of a pass in frame slot `s`
    fn mloc(&self, ms: &Multi, kk: usize, s: usize) -> (bool, usize, usize) {
        let len = ms.cb * self.nseg * self.plan.grid_elems();
        if kk < ms.fit {
            (true, s, kk * len)
        } else {
            (false, (kk - ms.fit) * self.nf + s, 0)
        }
    }

    fn mgrids(&self, ms: &Multi, st: &Arc<CudaStream>, map: &[usize], nk: usize) -> GridsM {
        let mut g = GridsM { p: [0; 16] };
        for kk in 0..nk {
            for (o, &s) in map.iter().enumerate() {
                let (eng, b, off) = self.mloc(ms, kk, s);
                let p = if eng { self.slots[b].device_ptr(st).0 } else { ms.extra[b].device_ptr(st).0 };
                g.p[kk * self.nf + o] = p + (off * std::mem::size_of::<C32>()) as u64;
            }
        }
        g
    }

    fn hi_ptrs(&self, st: &Arc<CudaStream>, pts: &SegPoints) -> (u64, u64) {
        match (&pts.hi, self.order >= 3) {
            (Some((u, h)), true) => (u.device_ptr(st).0, h.device_ptr(st).0),
            _ => (0, 0),
        }
    }

    /// image kk's grid in frame slot s ← FFT(pad(S_c P_l x_f)) for the coil batch
    #[allow(clippy::too_many_arguments)]
    fn mload(&mut self, ms: &mut Multi, st: &Arc<CudaStream>, seg: &SegSegments, fat: Option<FatArg>, kk: usize, s: usize, x: &CudaSlice<C32>, f: usize, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let nv = self.nv();
        let len = ms.cb * self.nseg * self.plan.grid_elems();
        let (eng, b, off) = self.mloc(ms, kk, s);
        let (md, d, c0_i, ncb_i) = (self.mdims(), self.plan.dims(), c0 as i32, ncb as i32);
        let buf = if eng { &mut self.slots[b] } else { &mut ms.extra[b] };
        let mut view = buf.slice_mut(off..off + len);
        ms.fft.as_ref().unwrap_or(&self.gfft).prepare(st, &mut view)?;
        if let Some(fa) = fat {
            let xf = x.slice(f * nv..(f + 1) * nv);
            let mi = fa.map.unwrap_or(f);
            let ffv = fa.maps.ff.slice(mi * nv..(mi + 1) * nv);
            let (gp, _g) = view.device_ptr_mut(st);
            unsafe {
                st.launch_builder(&ms.pad_ff)
                    .arg(&xf)
                    .arg(self.smaps.as_ref().expect("set_maps first"))
                    .arg(&seg.segph)
                    .arg(&ffv)
                    .arg(fa.cl)
                    .arg(&gp)
                    .arg(&self.plan.invphi[0])
                    .arg(&self.plan.invphi[1])
                    .arg(&self.plan.invphi[2])
                    .arg(&c0_i)
                    .arg(&ncb_i)
                    .arg(&md)
                    .arg(&d)
                    .launch(lc(nv))?;
            }
        } else {
            let xf = x.slice(f * nv..(f + 1) * nv);
            let (gp, _g) = view.device_ptr_mut(st);
            unsafe {
                st.launch_builder(&ms.pad)
                    .arg(&xf)
                    .arg(self.smaps.as_ref().expect("set_maps first"))
                    .arg(&seg.segph)
                    .arg(&gp)
                    .arg(&self.plan.invphi[0])
                    .arg(&self.plan.invphi[1])
                    .arg(&self.plan.invphi[2])
                    .arg(&c0_i)
                    .arg(&ncb_i)
                    .arg(&md)
                    .arg(&d)
                    .launch(lc(nv))?;
            }
        }
        ms.fft.as_ref().unwrap_or(&self.gfft).forward(st, &mut view)?;
        Ok(())
    }

    /// x_f += crop(IFFT(image kk's grid in frame slot s)) over the batch coils and segments
    #[allow(clippy::too_many_arguments)]
    fn mflush(&mut self, ms: &mut Multi, st: &Arc<CudaStream>, seg: &SegSegments, fat: Option<FatArg>, kk: usize, s: usize, x: &mut CudaSlice<C32>, f: usize, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let nv = self.nv();
        let len = ms.cb * self.nseg * self.plan.grid_elems();
        let (eng, b, off) = self.mloc(ms, kk, s);
        let (md, d, c0_i, ncb_i, acc) = (self.mdims(), self.plan.dims(), c0 as i32, ncb as i32, 1i32);
        let buf = if eng { &mut self.slots[b] } else { &mut ms.extra[b] };
        let mut view = buf.slice_mut(off..off + len);
        ms.fft.as_ref().unwrap_or(&self.gfft).inverse(st, &mut view)?;
        let (gp, _g) = view.device_ptr(st);
        let mut xf = x.slice_mut(f * nv..(f + 1) * nv);
        if let Some(fa) = fat {
            let mi = fa.map.unwrap_or(f);
            let ffv = fa.maps.ff.slice(mi * nv..(mi + 1) * nv);
            unsafe {
                st.launch_builder(&ms.crop_ff)
                    .arg(&gp)
                    .arg(self.smaps.as_ref().expect("set_maps first"))
                    .arg(&seg.segph)
                    .arg(&ffv)
                    .arg(fa.cl)
                    .arg(&mut xf)
                    .arg(&self.plan.invphi[0])
                    .arg(&self.plan.invphi[1])
                    .arg(&self.plan.invphi[2])
                    .arg(&c0_i)
                    .arg(&ncb_i)
                    .arg(&acc)
                    .arg(&md)
                    .arg(&d)
                    .launch(lc(nv))?;
            }
            return Ok(());
        }
        unsafe {
            st.launch_builder(&ms.crop)
                .arg(&gp)
                .arg(self.smaps.as_ref().expect("set_maps first"))
                .arg(&seg.segph)
                .arg(&mut xf)
                .arg(&self.plan.invphi[0])
                .arg(&self.plan.invphi[1])
                .arg(&self.plan.invphi[2])
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&acc)
                .arg(&md)
                .arg(&d)
                .launch(lc(nv))?;
        }
        Ok(())
    }

    fn mzero(&mut self, ms: &mut Multi, st: &Arc<CudaStream>, kk: usize, s: usize) -> Result<(), NufftError> {
        let len = ms.cb * self.nseg * self.plan.grid_elems();
        let (eng, b, off) = self.mloc(ms, kk, s);
        let buf = if eng { &mut self.slots[b] } else { &mut ms.extra[b] };
        st.memset_zeros(&mut buf.slice_mut(off..off + len))?;
        Ok(())
    }

    /// multi tile gather of interval i (`km` images compiled, `nk` live) for coils c0..c0+ncb: y[c0.., :] += …
    #[allow(clippy::too_many_arguments)]
    fn mlaunch_fwd(&self, km: usize, f: &CudaFunction, tc: &TileCfg, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, i: usize, g: GridsM, wp: WPtrs, nk: usize, y: &mut CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let iv = pts.ivals[i];
        if iv.count == 0 {
            return Ok(());
        }
        let (t0, dt, ntab, m_i, c0_i, ncb_i, nk_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, nk as i32, self.plan.dims());
        let items = pts.items.slice(4 * iv.first..4 * (iv.first + iv.count));
        let (uw, ht) = self.hi_ptrs(st, pts);
        let (yp, _gy) = y.device_ptr_mut(st);
        unsafe {
            st.launch_builder(f)
                .arg(&pts.rec)
                .arg(&pts.tl)
                .arg(&items)
                .arg(&seg.atab)
                .arg(&ntab)
                .arg(&t0)
                .arg(&dt)
                .arg(&g)
                .arg(&wp)
                .arg(&nk_i)
                .arg(&uw)
                .arg(&ht)
                .arg(&yp)
                .arg(&m_i)
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&d)
                .launch(LaunchConfig { grid_dim: (iv.count as u32, ncb.div_ceil(tc.cpbf) as u32, 1), block_dim: ((tc.chunk * tc.cpbf * tc.ppf * self.nf) as u32, 1, 1), shared_mem_bytes: tc.smem_multi(self.shape(), km)[1] as u32 })?;
        }
        Ok(())
    }

    /// multi tile scatter of interval i into the images' grid slots (bit o of `addmask`: add instead of store)
    #[allow(clippy::too_many_arguments)]
    fn mlaunch_adj(&self, km: usize, f: &CudaFunction, tc: &TileCfg, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, i: usize, g: GridsM, addmask: i32, wp: WPtrs, nk: usize, y: &CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let iv = pts.ivals[i];
        let (t0, dt, ntab, m_i, c0_i, ncb_i, nk_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, nk as i32, self.plan.dims());
        let items = pts.items.slice(4 * iv.first..4 * (iv.first + iv.count));
        let (uw, ht) = self.hi_ptrs(st, pts);
        unsafe {
            st.launch_builder(f)
                .arg(&pts.rec)
                .arg(&pts.tl)
                .arg(&items)
                .arg(&seg.atab)
                .arg(&ntab)
                .arg(&t0)
                .arg(&dt)
                .arg(&wp)
                .arg(&nk_i)
                .arg(&uw)
                .arg(&ht)
                .arg(y)
                .arg(&m_i)
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&g)
                .arg(&addmask)
                .arg(&d)
                .launch(LaunchConfig { grid_dim: (iv.count as u32, ncb.div_ceil(tc.cpba) as u32, 1), block_dim: ((tc.tt.pow(3) * self.nf * tc.cpba * km) as u32, 1, 1), shared_mem_bytes: tc.smem_multi(self.shape(), km)[0] as u32 })?;
        }
        Ok(())
    }

    /// y (nc × m) = Σ_k w_k ⊙ A x_k: images `xs` (frames × nv each), per-sample weights `ws` (m values, the same for
    /// every coil; None = 1). One tile pass over the images' grids per (coil batch, frame interval, pass of ≤ 4
    /// images); the FFTs stay one per image. Equals Σ_k w_k ⊙ `forward`(x_k) in exact arithmetic (summation order
    /// differs in f32). With order 3 this is also the path of `forward`.
    pub fn forward_multi(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, xs: &[&CudaSlice<C32>], ws: &[Option<&CudaSlice<C32>>], y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.forward_multi(st, seg, p, xs, ws, y));
        }
        let k = xs.len();
        let (nv, m) = (self.nv(), pts.m);
        assert!(k >= 1 && ws.len() == k, "{k} images, {} weights", ws.len());
        assert!(xs.iter().all(|x| x.len() >= self.frames * nv) && y.len() >= self.nc * m);
        assert!(ws.iter().all(|w| w.is_none_or(|w| w.len() >= m)), "weights need one value per sample");
        let i = self.multi_index(st, k)?;
        let mut ms = self.multi.swap_remove(i);
        let r = self.forward_multi_in(&mut ms, st, seg, None, pts, xs, ws, None, y);
        self.multi.push(ms);
        r
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn forward_multi_in(&mut self, ms: &mut Multi, st: &Arc<CudaStream>, seg: &SegSegments, fat: Option<FatArg>, pts: &SegPoints, xs: &[&CudaSlice<C32>], ws: &[Option<&CudaSlice<C32>>], init: Option<&CudaSlice<C32>>, y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        let (k, m) = (xs.len(), pts.m);
        let (f, tc, km) = (ms.fwd.clone(), ms.tf, ms.g);
        let mut c0 = 0;
        while c0 < self.nc {
            let ncb = ms.cb.min(self.nc - c0);
            self.init_rows(st, init, y, c0 * m, (c0 + ncb) * m)?;
            for k0 in (0..k).step_by(km) {
                let nk = km.min(k - k0);
                let wp = wptrs(st, &ws[k0..k0 + nk]);
                let mut held: Vec<Option<usize>> = vec![None; self.nf];
                for i in 0..self.frames {
                    let need = self.frames_of(i);
                    let (map, fresh, _) = Self::assign(&mut held, &need);
                    for s in fresh {
                        for kk in 0..nk {
                            self.mload(ms, st, seg, fat, kk, s, xs[k0 + kk], held[s].unwrap(), c0, ncb)?;
                        }
                    }
                    let g = self.mgrids(ms, st, &map, nk);
                    self.mlaunch_fwd(km, &f, &tc, st, seg, pts, i, g, wp, nk, y, c0, ncb)?;
                }
            }
            c0 += ncb;
        }
        Ok(())
    }

    /// x_k (frames × nv each) = Aᴴ(w̄_k ⊙ y) for every image k (conjugated per-sample weights; None = 1), one tile
    /// scatter pass writing the images' grids. Equals `adjoint`(w̄_k ⊙ y) per image in exact arithmetic. With order 3
    /// this is also the path of `adjoint`.
    pub fn adjoint_multi(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, y: &CudaSlice<C32>, ws: &[Option<&CudaSlice<C32>>], xs: &mut [&mut CudaSlice<C32>]) -> Result<(), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.adjoint_multi(st, seg, p, y, ws, xs));
        }
        let k = xs.len();
        let (nv, m) = (self.nv(), pts.m);
        assert!(k >= 1 && ws.len() == k, "{k} images, {} weights", ws.len());
        assert!(xs.iter().all(|x| x.len() >= self.frames * nv) && y.len() >= self.nc * m);
        assert!(ws.iter().all(|w| w.is_none_or(|w| w.len() >= m)), "weights need one value per sample");
        for x in xs.iter_mut() {
            st.memset_zeros(&mut x.slice_mut(0..self.frames * nv))?;
        }
        let i = self.multi_index(st, k)?;
        let mut ms = self.multi.swap_remove(i);
        let r = self.adjoint_multi_in(&mut ms, st, seg, None, pts, y, ws, xs);
        self.multi.push(ms);
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn adjoint_multi_in(&mut self, ms: &mut Multi, st: &Arc<CudaStream>, seg: &SegSegments, fat: Option<FatArg>, pts: &SegPoints, y: &CudaSlice<C32>, ws: &[Option<&CudaSlice<C32>>], xs: &mut [&mut CudaSlice<C32>]) -> Result<(), NufftError> {
        let k = xs.len();
        let (f, tc, km) = (ms.adj.clone(), ms.ta, ms.g);
        let mut c0 = 0;
        while c0 < self.nc {
            let ncb = ms.cb.min(self.nc - c0);
            for k0 in (0..k).step_by(km) {
                let nk = km.min(k - k0);
                let wp = wptrs(st, &ws[k0..k0 + nk]);
                let mut held: Vec<Option<usize>> = vec![None; self.nf];
                let mut stale = vec![false; self.nf];
                for i in 0..self.frames {
                    let need = self.frames_of(i);
                    let (map, fresh, evicted) = Self::assign(&mut held, &need);
                    for (s, g) in evicted {
                        for kk in 0..nk {
                            self.mflush(ms, st, seg, fat, kk, s, &mut *xs[k0 + kk], g, c0, ncb)?;
                        }
                    }
                    for &s in &fresh {
                        stale[s] = false;
                    }
                    let split = pts.ivals[i].split;
                    let mut addmask = 0i32;
                    for (o, &s) in map.iter().enumerate() {
                        if !stale[s] && split {
                            for kk in 0..nk {
                                self.mzero(ms, st, kk, s)?;
                            }
                            stale[s] = true;
                        }
                        if stale[s] {
                            addmask |= 1 << o;
                        }
                    }
                    let g = self.mgrids(ms, st, &map, nk);
                    self.mlaunch_adj(km, &f, &tc, st, seg, pts, i, g, addmask, wp, nk, y, c0, ncb)?;
                    map.iter().for_each(|&s| stale[s] = true);
                }
                for s in 0..self.nf {
                    if let Some(g) = held[s] {
                        for kk in 0..nk {
                            self.mflush(ms, st, seg, fat, kk, s, &mut *xs[k0 + kk], g, c0, ncb)?;
                        }
                    }
                }
            }
            c0 += ncb;
        }
        Ok(())
    }

    /// [`SegNufft::points`] for correction order 3: additionally the per-sample degree-3/4 polynomial phase
    ///   c_j(ρ) = Σ_α (Σ_q u_qj h_qα) ρ^α   (ρ in m as ΔC; α: the [`NHI`] monomials of [`hi_exponents`]),
    /// `u` SoA (4 × m, e.g. the concomitant deviation Δw_j), `h` q-major (4 × [`NHI`], e.g. the degree-3/4 coefficients
    /// of the concomitant monomials' fit). The kernels apply 1 − iρᵀΔCρ − ½(ρᵀΔCρ)² − i c_j(ρ) (Taylor of
    /// e^{−i(ρᵀΔCρ + c_j)} without the products of the two parts).
    #[allow(clippy::too_many_arguments)]
    pub fn points_hi(&mut self, st: &Arc<CudaStream>, coords: [&[f32]; 3], tau: &[f32], phi0: &[f32], dc: &[f32], u: &[f32], h: &[f64], t: Option<&[f32]>) -> Result<SegPoints, NufftError> {
        assert!(self.order >= 3, "points_hi needs correction order 3 (engine order {})", self.order);
        let m = coords[0].len();
        assert!(u.len() == 4 * m && h.len() == 4 * NHI, "points_hi: u {} (4 × {m}), h {} (4 × {NHI})", u.len(), h.len());
        let mut pts = self.points(st, coords, tau, phi0, dc, t)?;
        let uw: Vec<f32> = (0..m).flat_map(|j| (0..4).map(move |q| u[q * m + j])).collect();
        // kernel-argument units: ρ^α ↔ Π_a g_a^{α_a} ∂^α (g_a = res_a nf_a / (π W), as ΔC in `points`)
        let w = self.plan.ker.w as f64;
        let g: [f64; 3] = [0, 1, 2].map(|a| self.res[a] * self.plan.nf[a] as f64 / (std::f64::consts::PI * w));
        let ex = hi_exponents();
        let ht: Vec<f32> = (0..4).flat_map(|q| (0..NHI).map(move |i| (h[q * NHI + i] * (0..3).map(|a| g[a].powi(ex[i][a] as i32)).product::<f64>()) as f32)).collect();
        pts.hi = Some((st.clone_htod(&uw)?, st.clone_htod(&ht)?));
        Ok(pts)
    }

    /// Time the multi-image tile kernels for `k` images on the given set (busiest interval) over candidate launch
    /// shapes and keep the fastest per direction (as [`SegNufft::autotune`] for the single-image kernels). Returns the
    /// chosen (forward, adjoint) kernel times in ms.
    pub fn autotune_multi(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, k: usize) -> Result<(f32, f32), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.autotune_multi(st, seg, p, k));
        }
        assert!(pts.m > 0, "autotune needs points");
        let i = self.multi_index(st, k)?;
        let mut ms = self.multi.swap_remove(i);
        let r = self.autotune_multi_in(&mut ms, st, seg, pts);
        self.multi.push(ms);
        r
    }

    /// Joint water + fat segments ([`build_segments_fat`]): water diagonals P_l (nseg × nv), the table a(τ) and the fat
    /// spectrum at the nodes c_l; used with fat-fraction maps by [`SegNufft::forward_fat`] / [`SegNufft::adjoint_fat`]
    /// (diagonals P_l (1 + ff (c_l − 1)), the maps kept separate so that one table per encode serves every frame's map).
    /// As [`SegNufft::segments`] otherwise (plain `forward` / `adjoint` see the water diagonals).
    pub fn segments_fat(&self, st: &Arc<CudaStream>, segph: &[C32], atab: &[C32], c_nodes: &[C32], t0: f64, dt: f64) -> Result<SegSegments, NufftError> {
        assert_eq!(c_nodes.len(), self.nseg);
        let mut s = self.segments(st, segph, atab, t0, dt)?;
        s.fatc = Some(st.clone_htod(c_nodes)?);
        Ok(s)
    }

    /// fat-fraction maps for [`SegNufft::forward_fat`]: `ff` = nmap × nv values (clamped to [0, 1]), e.g. one per
    /// cardiac frame
    pub fn fat_maps(&self, st: &Arc<CudaStream>, ff: &[f32]) -> Result<SegFatMaps, NufftError> {
        let nv = self.nv();
        assert!(!ff.is_empty() && ff.len() % nv == 0, "{} fat-fraction values for {nv} voxels", ff.len());
        let v: Vec<f32> = ff.iter().map(|f| f.clamp(0.0, 1.0)).collect();
        Ok(SegFatMaps { ff: st.clone_htod(&v)?, nmap: ff.len() / nv })
    }

    /// y = A x with the joint water + fat diagonals P_l (1 + ff (c_l − 1)) of `seg` ([`SegNufft::segments_fat`]):
    /// `map` Some(i) = map i of `maps` for every frame (e.g. a static (frame, encode) set), None = map f for frame f
    /// (a time-resolved set, `maps.nmap == frames`). One image, the engine's L segments (single-image kernels of the
    /// multi path, K = 1).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fat(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, maps: &SegFatMaps, map: Option<usize>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.forward_fat(st, seg, p, maps, map, x, y));
        }
        let (nv, m) = (self.nv(), pts.m);
        assert!(x.len() >= self.frames * nv && y.len() >= self.nc * m);
        let cl = seg.fatc.as_ref().expect("forward_fat needs segments_fat");
        assert!(map.map_or(maps.nmap == self.frames, |i| i < maps.nmap), "fat map {map:?} of {} (frames {})", maps.nmap, self.frames);
        let i = self.multi_index(st, 1)?;
        let mut ms = self.multi.swap_remove(i);
        let r = self.forward_multi_in(&mut ms, st, seg, Some(FatArg { maps, map, cl }), pts, &[x], &[None], None, y);
        self.multi.push(ms);
        r
    }

    /// x = Aᴴ y, the adjoint of [`SegNufft::forward_fat`]
    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_fat(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, maps: &SegFatMaps, map: Option<usize>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        if pts.is_host() {
            // host-resident tables: run on a device slot holding them
            return self.with_staged(st, pts, |e, p| e.adjoint_fat(st, seg, p, maps, map, y, x));
        }
        let (nv, m) = (self.nv(), pts.m);
        assert!(x.len() >= self.frames * nv && y.len() >= self.nc * m);
        let cl = seg.fatc.as_ref().expect("adjoint_fat needs segments_fat");
        assert!(map.map_or(maps.nmap == self.frames, |i| i < maps.nmap), "fat map {map:?} of {} (frames {})", maps.nmap, self.frames);
        st.memset_zeros(&mut x.slice_mut(0..self.frames * nv))?;
        let i = self.multi_index(st, 1)?;
        let mut ms = self.multi.swap_remove(i);
        let r = self.adjoint_multi_in(&mut ms, st, seg, Some(FatArg { maps, map, cl }), pts, y, &[None], &mut [x]);
        self.multi.push(ms);
        r
    }

    fn autotune_multi_in(&mut self, ms: &mut Multi, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints) -> Result<(f32, f32), NufftError> {
        let sh = self.shape();
        let (tt, mi, cb, km) = (ms.tf.tt, ms.tf.max_item, ms.cb, ms.g);
        let ncb = cb.min(self.nc);
        let busiest = (0..pts.ivals.len()).max_by_key(|&i| pts.ivals[i].count).unwrap();
        let map: Vec<usize> = (0..self.nf).collect();
        let g = self.mgrids(ms, st, &map, km);
        let wp = WPtrs { p: [0; 4] };
        let groups = [1usize, 2, 3, 4, 6, 8, 12, 16];
        let mut ys = st.alloc_zeros::<C32>(cb * pts.m)?;
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ev = (self.dev.ctx.new_event(flags)?, self.dev.ctx.new_event(flags)?);
        let time = |run: &mut dyn FnMut() -> Result<(), NufftError>| -> Result<f32, NufftError> {
            run()?;
            let mut best = f32::INFINITY;
            for _ in 0..3 {
                ev.0.record(st)?;
                run()?;
                ev.1.record(st)?;
                ev.1.synchronize()?;
                best = best.min(ev.0.elapsed_ms(&ev.1)?);
            }
            Ok(best)
        };
        let mut best_f = (f32::INFINITY, ms.tf);
        for &cpbf in groups.iter().filter(|&&c| c <= cb) {
            for ppf in [1usize, 2, 4] {
                let Some(tc) = TileCfg::fit_multi(sh, km, tt, ms.ta.cpba, cpbf, ppf, mi, self.smax) else { continue };
                if tc.chunk < 8 {
                    continue;
                }
                let f = tile_fn(&compile_multi(&self.dev, &self.plan, sh, cb, km, &tc)?, "seg_forward_mt", tc.smem_multi(sh, km)[1])?;
                let t = time(&mut || self.mlaunch_fwd(km, &f, &tc, st, seg, pts, busiest, g, wp, km, &mut ys, 0, ncb))?;
                if t < best_f.0 {
                    best_f = (t, tc);
                }
            }
        }
        let mut best_a = (f32::INFINITY, ms.ta);
        for &cpba in groups.iter().filter(|&&c| c <= cb) {
            let Some(tc) = TileCfg::fit_multi(sh, km, tt, cpba, best_f.1.cpbf, best_f.1.ppf, mi, self.smax) else { continue };
            let f = tile_fn(&compile_multi(&self.dev, &self.plan, sh, cb, km, &tc)?, "seg_adjoint_mt", tc.smem_multi(sh, km)[0])?;
            let t = time(&mut || self.mlaunch_adj(km, &f, &tc, st, seg, pts, busiest, g, 0, wp, km, &ys, 0, ncb))?;
            if t < best_a.0 {
                best_a = (t, tc);
            }
        }
        ms.fwd = tile_fn(&compile_multi(&self.dev, &self.plan, sh, cb, km, &best_f.1)?, "seg_forward_mt", best_f.1.smem_multi(sh, km)[1])?;
        ms.adj = tile_fn(&compile_multi(&self.dev, &self.plan, sh, cb, km, &best_a.1)?, "seg_adjoint_mt", best_a.1.smem_multi(sh, km)[0])?;
        ms.tf = best_f.1;
        ms.ta = best_a.1;
        Ok((best_f.0, best_a.0))
    }
}

#[cfg(test)]
mod multi_tests {
    use super::*;

    #[test]
    fn multi_plan_respects_budget_and_prefers_shared_passes() {
        let pc = 100usize;
        let ok = |_: usize, _: usize| true;
        // ample budget: two images × all 12 coils, one extra image of grids
        assert_eq!(multi_plan(2, 12, 12, pc, 24 * pc, &ok), (2, 12, 1));
        // budget = the engine's own slots: half the coils per batch, both images inside the engine slots
        assert_eq!(multi_plan(2, 12, 12, pc, 12 * pc, &ok), (2, 6, 2));
        // one coil already exceeds the budget (cb clamped to 1): no extra memory → one image per pass
        assert_eq!(multi_plan(2, 1, 12, pc, pc, &ok), (1, 1, 1));
        // … unless extra memory is allowed
        assert_eq!(multi_plan(2, 1, 12, pc, 2 * pc, &ok), (2, 1, 1));
        // more than four images: passes of ≤ 4
        assert_eq!(multi_plan(6, 16, 16, pc, 16 * pc, &ok), (4, 4, 4));
        // tile shapes that do not fit reduce the images per pass
        assert_eq!(multi_plan(3, 12, 12, pc, 12 * pc, &|g, _| g <= 2), (2, 6, 2));
        for (k, cb, nc, b) in [(2usize, 5usize, 7usize, 9usize), (3, 4, 4, 4), (4, 3, 9, 30), (1, 8, 8, 8)] {
            let (g, cbm, fit) = multi_plan(k, cb, nc, pc, b * pc, &ok);
            assert!(g >= 1 && g <= k.min(MULTI_KMAX) && cbm >= 1 && cbm <= nc.min(16) && fit == (cb / cbm).min(g));
            assert!(cb * pc + (g - fit) * cbm * pc <= (b * pc).max(cb * pc), "k {k} cb {cb}: plan ({g}, {cbm}, {fit}) over budget");
        }
    }

    #[test]
    fn multi_tile_shapes_fit_shared_memory() {
        for order in 0..=3u32 {
            for nf in [1usize, 2, 4] {
                for nseg in [1usize, 8, 16, 24] {
                    let sh = Shape { nseg, order, nf };
                    for km in 1..=MULTI_KMAX {
                        let tc = TileCfg::new_multi(sh, km, 4, 4, 1024, 99 * 1024);
                        if let Some(tc) = tc {
                            assert!(tc.fits_multi(sh, km, 99 * 1024) && tc.chunk >= 1);
                            // km = 1 needs at least what the single-image kernels need (plus the weights)
                            if km == 1 {
                                let s1 = tc.smem(sh);
                                let sm = tc.smem_multi(sh, 1);
                                assert!(sm[0] >= s1[0] && sm[1] >= s1[1]);
                            }
                        } else {
                            assert!(km > 1 || nseg * nf > 16, "order {order} nf {nf} L {nseg} km {km}: no tile shape");
                        }
                    }
                }
            }
        }
        assert_eq!(hi_exponents()[0], [0, 0, 3]);
        assert_eq!(hi_exponents()[9], [3, 0, 0]);
        assert_eq!(hi_exponents()[10], [0, 0, 4]);
        assert_eq!(hi_exponents()[24], [4, 0, 0]);
        assert!(hi_exponents().iter().all(|e| e.iter().sum::<u32>() >= 3));
    }

    /// The multi-image module source for every order and image count (dumped for an offline compile check with
    /// `HASTY_SEG_MULTI_SRC=dir`; nothing is compiled or run here).
    #[test]
    fn multi_sources_are_generated() {
        let dir = std::env::var("HASTY_SEG_MULTI_SRC").ok();
        for order in 0..=3u32 {
            for (nf, km) in [(1usize, 1usize), (1, 2), (2, 3), (4, 4)] {
                let sh = Shape { nseg: 8, order, nf };
                let tc = TileCfg::new_multi(sh, km, 4, 4, 1024, 99 * 1024).unwrap();
                let (src, name) = multi_source(10, 23.0, sh, 4, km, &tc);
                assert!(src.contains("seg_forward_mt") && src.contains("seg_adjoint_mt") && src.contains(&format!("#define KM {km}\n")));
                if let Some(d) = &dir {
                    std::fs::write(std::path::Path::new(d).join(format!("{name}.cu")), &src).unwrap();
                }
            }
        }
    }
}

/// [`build_segments`] for a complex per-voxel time function g(v, τ) instead of a phase (e.g. water + fat with a known
/// fat fraction folded in: g_v(τ) = [(1 − ff_v) + ff_v c(τ)] e^{−iψ(v, τ)}): diagonals P_l = g(·, τ_l) at `nseg`
/// uniformly spaced τ_l in [t_min, t_max] and least-squares interpolators a(τ) on an `ntab`-point table, fitted over
/// `fit_voxels` (small ridge). Returns (P [nseg × nv], a [ntab × nseg], t0, dt) for [`SegNufft::segments`].
pub fn build_segments_complex(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    g: &(dyn Fn(usize, f64) -> num_complex::Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, f64, f64) {
    use nalgebra::DVector;
    use num_complex::Complex64;
    let tl: Vec<f64> = (0..nseg).map(|l| if nseg == 1 { 0.5 * (t_min + t_max) } else { t_min + (t_max - t_min) * l as f64 / (nseg - 1) as f64 }).collect();
    let segph = seg_diagonals(nv, &tl, g);
    let b = seg_basis(fit_voxels.len(), &tl, &|r, t| g(fit_voxels[r], t));
    let atab = ls_interpolators(&b, nseg, t_min, t_max, ntab, &|t| DVector::<Complex64>::from_iterator(fit_voxels.len(), fit_voxels.iter().map(|&v| g(v, t))));
    (segph, atab, t_min, (t_max - t_min) / (ntab - 1) as f64)
}

/// least-squares interpolators: a(τ_i) = argmin ‖B a − y(τ_i)‖² (+ 1e-9 relative ridge) on the table
/// (table entries in parallel: each is independent, so the result does not depend on the thread count)
fn ls_interpolators(b: &nalgebra::DMatrix<num_complex::Complex64>, nseg: usize, t_min: f64, t_max: f64, ntab: usize, y: &(dyn Fn(f64) -> nalgebra::DVector<num_complex::Complex64> + Sync)) -> Vec<C32> {
    use num_complex::Complex64;
    use rayon::prelude::*;
    let bh = b.adjoint();
    let mut g = &bh * b;
    let tr = (0..nseg).map(|l| g[(l, l)].re).sum::<f64>() / nseg as f64;
    for l in 0..nseg {
        g[(l, l)] += Complex64::new(1e-9 * tr, 0.0);
    }
    let chol = g.cholesky().expect("segment Gram matrix");
    let dt = (t_max - t_min) / (ntab - 1) as f64;
    let mut atab = vec![C32::new(0.0, 0.0); ntab * nseg];
    atab.par_chunks_mut(nseg.max(1)).enumerate().for_each(|(i, row)| {
        let a = chol.solve(&(&bh * y(t_min + i as f64 * dt)));
        for l in 0..nseg {
            row[l] = C32::new(a[l].re as f32, a[l].im as f32);
        }
    });
    atab
}

/// Joint water + fat segments that do not depend on the fat fraction: water diagonals P_l = e^{−iψ(·, τ_l)} (as
/// [`build_segments`]), the fat spectrum at the nodes c_l = c(τ_l), and interpolators a(τ) fitted to the water AND the
/// fat time function of every fit voxel ({e^{−iψ(v, τ)}, c(τ) e^{−iψ(v, τ)}}). The joint diagonal of a voxel with fat
/// fraction f is P_l (1 + f (c_l − 1)); since its time function (1 − f) e^{−iψ} + f c e^{−iψ} is linear in f, the same
/// table serves every fat-fraction map (per frame, per set): fold one map into the diagonals with
/// [`fold_fat_fraction`] and use [`SegNufft::segments`], or keep the maps separate ([`SegNufft::segments_fat`],
/// [`SegNufft::forward_fat`]). Returns (P [nseg × nv], a [ntab × nseg], c_l [nseg], t0, dt).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn build_segments_fat(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    psi: &(dyn Fn(usize, f64) -> f64 + Sync),
    c: &(dyn Fn(f64) -> num_complex::Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, Vec<C32>, f64, f64) {
    use nalgebra::DVector;
    use num_complex::Complex64;
    let tl: Vec<f64> = (0..nseg).map(|l| if nseg == 1 { 0.5 * (t_min + t_max) } else { t_min + (t_max - t_min) * l as f64 / (nseg - 1) as f64 }).collect();
    let segph = seg_diagonals(nv, &tl, &|v, t| {
        let (s, co) = (-psi(v, t)).sin_cos();
        Complex64::new(co, s)
    });
    let cl: Vec<Complex64> = tl.iter().map(|&t| c(t)).collect();
    let nf = fit_voxels.len();
    let b = seg_basis(2 * nf, &tl, &|r, t| {
        let w = Complex64::from_polar(1.0, -psi(fit_voxels[r % nf], t));
        if r < nf { w } else { w * c(t) }
    });
    let atab = ls_interpolators(&b, nseg, t_min, t_max, ntab, &|t| {
        let ct = c(t);
        DVector::<Complex64>::from_iterator(2 * nf, (0..2 * nf).map(|r| {
            let w = Complex64::from_polar(1.0, -psi(fit_voxels[r % nf], t));
            if r < nf { w } else { w * ct }
        }))
    });
    (segph, atab, cl.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect(), t_min, (t_max - t_min) / (ntab - 1) as f64)
}

/// one fat-fraction map folded into water diagonals of [`build_segments_fat`]: P_l (1 + ff (c_l − 1)), ff clamped to [0, 1]
pub fn fold_fat_fraction(segph: &[C32], c_nodes: &[C32], ff: &[f32]) -> Vec<C32> {
    let (nseg, nv) = (c_nodes.len(), ff.len());
    assert_eq!(segph.len(), nseg * nv);
    let mut out = segph.to_vec();
    for l in 0..nseg {
        let c = c_nodes[l];
        for v in 0..nv {
            let f = ff[v].clamp(0.0, 1.0);
            let (p, fr, fi) = (segph[l * nv + v], 1.0 + f * (c.re - 1.0), f * c.im);
            out[l * nv + v] = C32::new(p.re * fr - p.im * fi, p.re * fi + p.im * fr);
        }
    }
    out
}

// ───────────────────────────────────────────────────────────── streamed sample tables (additions)

/// upload host-resident tables `src` into `slot` on stream `on` (grows the slot if needed; the caller makes sure
/// nothing still uses a slot that has to grow)
fn fill_slot(on: &Arc<CudaStream>, src: &SegPoints, slot: &mut Option<SegPoints>, stage: &mut Staging) -> Result<usize, NufftError> {
    let h = src.host.as_ref().expect("host-resident tables");
    let grow = |d: Option<&CudaSlice<f32>>, n: usize| d.is_none_or(|d| d.len() < n);
    let fits = slot.as_ref().is_some_and(|s| {
        s.rec.len() >= h.rec.len() && s.tl.len() >= h.tl.len() && s.items.len() >= h.items.len() && match &h.hi {
            Some((u, t)) => !grow(s.hi.as_ref().map(|x| &x.0), u.len()) && !grow(s.hi.as_ref().map(|x| &x.1), t.len()),
            None => true,
        }
    });
    if !fits {
        let old = slot.take();
        let cap = |d: Option<usize>, n: usize| d.unwrap_or(0).max(n);
        let hi = match &h.hi {
            Some((u, t)) => Some((on.alloc_zeros::<f32>(cap(old.as_ref().and_then(|o| o.hi.as_ref().map(|x| x.0.len())), u.len()))?, on.alloc_zeros::<f32>(cap(old.as_ref().and_then(|o| o.hi.as_ref().map(|x| x.1.len())), t.len()))?)),
            None => old.as_ref().and_then(|o| o.hi.as_ref()).map(|(u, t)| -> Result<_, NufftError> { Ok((on.alloc_zeros::<f32>(u.len())?, on.alloc_zeros::<f32>(t.len())?)) }).transpose()?,
        };
        *slot = Some(SegPoints {
            rec: on.alloc_zeros(cap(old.as_ref().map(|o| o.rec.len()), h.rec.len()))?,
            tl: on.alloc_zeros(cap(old.as_ref().map(|o| o.tl.len()), h.tl.len()))?,
            items: on.alloc_zeros(cap(old.as_ref().map(|o| o.items.len()), h.items.len()))?,
            ivals: vec![],
            m: 0,
            hi,
            host: None,
            id: 0,
        });
    }
    let s = slot.as_mut().unwrap();
    // packed tables decode into the slot's pinned staging buffer (after its previous upload completed)
    let need = h.rec.bytes() + h.tl.bytes() + h.items.bytes() + h.hi.as_ref().map_or(0, |(u, t)| u.bytes() + t.bytes());
    let packed = matches!(h.rec, HostBuf::Packed { .. });
    let buf: &mut [u8] = if packed { stage.get(on, need)? } else { &mut [] };
    let mut off = 0;
    let mut bytes = 0;
    bytes += h.rec.upload(on, &mut s.rec, buf, off)?;
    off += if packed { h.rec.bytes() } else { 0 };
    bytes += h.tl.upload(on, &mut s.tl, buf, off)?;
    off += if packed { h.tl.bytes() } else { 0 };
    bytes += h.items.upload(on, &mut s.items, buf, off)?;
    off += if packed { h.items.bytes() } else { 0 };
    if let (Some((u, t)), Some((du, dt))) = (&h.hi, s.hi.as_mut()) {
        bytes += u.upload(on, du, buf, off)?;
        off += if packed { u.bytes() } else { 0 };
        bytes += t.upload(on, dt, buf, off)?;
    }
    let _ = off;
    if packed {
        stage.record(on)?;
    }
    // one engine streams either order-3 tables or plain ones (a slot's order-3 buffers are never dropped in flight)
    assert!(h.hi.is_some() || s.hi.is_none(), "streamed tables: order-3 and plain sample tables mixed on one engine");
    s.ivals = src.ivals.clone();
    s.m = src.m;
    Ok(bytes)
}

impl SegNufft {
    fn streamer(&mut self) -> &mut Streamer {
        self.stream.get_or_insert_with(|| Streamer { up: None, staging: vec![Staging::default(), Staging::default()], slots: vec![None, None], owner: vec![None, None], ready: vec![None, None], free: vec![None, None], last: 0, uploads: 0, upload_bytes: 0 })
    }

    /// Size both table slots for the largest of `pts` up front (avoids growing them while streaming).
    pub fn stream_reserve(&mut self, st: &Arc<CudaStream>, pts: &[&SegPoints]) -> Result<(), NufftError> {
        let host: Vec<&SegPoints> = pts.iter().copied().filter(|p| p.is_host()).collect();
        if host.is_empty() {
            return Ok(());
        }
        st.synchronize()?;
        if let Some(up) = self.stream.as_ref().and_then(|s| s.up.clone()) {
            up.synchronize()?;
        }
        let big = |f: &dyn Fn(&PointsHost) -> usize| host.iter().map(|p| f(p.host.as_ref().unwrap())).max().unwrap_or(1).max(1);
        let (nr, nt, ni) = (big(&|h| h.rec.len()), big(&|h| h.tl.len()), big(&|h| h.items.len()));
        let hi = host.iter().any(|p| p.host.as_ref().unwrap().hi.is_some()).then(|| (big(&|h| h.hi.as_ref().map_or(1, |x| x.0.len())), big(&|h| h.hi.as_ref().map_or(1, |x| x.1.len()))));
        let s = self.streamer();
        for i in 0..s.slots.len() {
            s.slots[i] = Some(SegPoints {
                rec: st.alloc_zeros(nr)?,
                tl: st.alloc_zeros(nt)?,
                items: st.alloc_zeros(ni)?,
                ivals: vec![],
                m: 0,
                hi: hi.map(|(a, b)| -> Result<_, NufftError> { Ok((st.alloc_zeros::<f32>(a)?, st.alloc_zeros::<f32>(b)?)) }).transpose()?,
                host: None,
                id: 0,
            });
            s.owner[i] = None;
            s.ready[i] = None;
            s.free[i] = None;
        }
        Ok(())
    }

    /// free the table slots (after the tables went back to the device)
    pub fn stream_release(&mut self, st: &Arc<CudaStream>) -> Result<(), NufftError> {
        if let Some(s) = self.stream.take() {
            st.synchronize()?;
            if let Some(up) = &s.up {
                up.synchronize()?;
            }
        }
        Ok(())
    }

    /// pinned staging bytes for packed tables
    pub fn stream_staging_bytes(&self) -> usize {
        self.stream.as_ref().map_or(0, |s| s.staging.iter().map(|x| x.bytes()).sum())
    }

    /// device bytes of the table slots (0 before any host-resident set was used)
    pub fn stream_slot_bytes(&self) -> usize {
        self.stream.as_ref().map_or(0, |s| s.slots.iter().flatten().map(|p| p.device_bytes()).sum())
    }

    /// (uploads, bytes uploaded) of streamed tables so far
    pub fn stream_stats(&self) -> (usize, usize) {
        self.stream.as_ref().map_or((0, 0), |s| (s.uploads, s.upload_bytes))
    }

    /// a slot holding `pts` (host-resident) for the compute stream `st`: found (waiting for its prefetch) or
    /// uploaded now on `st` into the slot not used last
    fn stage(&mut self, st: &Arc<CudaStream>, pts: &SegPoints) -> Result<usize, NufftError> {
        let s = self.streamer();
        if let Some(i) = s.owner.iter().position(|o| *o == Some(pts.id)) {
            if let Some(ev) = &s.ready[i] {
                st.wait(ev)?;
            }
            s.last = i;
            return Ok(i);
        }
        let i = (s.last + 1) % s.slots.len();
        if let Some(ev) = &s.ready[i] {
            st.wait(ev)?;
        }
        if !slot_fits(&s.slots[i], pts) {
            // growing frees the old buffers: nothing may still use them
            st.synchronize()?;
            if let Some(up) = &s.up {
                up.synchronize()?;
            }
        }
        let b = fill_slot(st, pts, &mut s.slots[i], &mut s.staging[i])?;
        s.owner[i] = Some(pts.id);
        s.ready[i] = None;
        s.last = i;
        s.uploads += 1;
        s.upload_bytes += b;
        Ok(i)
    }

    /// run `f` with the device slot holding the host-resident tables `pts`
    fn with_staged<R>(&mut self, st: &Arc<CudaStream>, pts: &SegPoints, f: impl FnOnce(&mut Self, &SegPoints) -> Result<R, NufftError>) -> Result<R, NufftError> {
        let i = self.stage(st, pts)?;
        let slot = self.stream.as_mut().unwrap().slots[i].take().expect("table slot");
        let r = f(self, &slot);
        let prefetch = self.stream_prefetch;
        let ev = if prefetch { Some(self.dev.ctx.new_event(None)?) } else { None };
        let s = self.stream.as_mut().unwrap();
        s.slots[i] = Some(slot);
        if let Some(ev) = ev {
            // the slot's last use so far: a prefetch into it waits for this
            ev.record(st)?;
            s.free[i] = Some(ev);
        }
        r
    }

    /// With `stream_prefetch`: start uploading the host-resident tables `pts` (e.g. the next set) into the slot not
    /// used last, on a second stream, overlapping the current set's work. No-op for device-resident tables, when they
    /// are already in a slot, or without `stream_prefetch`.
    pub fn prefetch(&mut self, st: &Arc<CudaStream>, pts: &SegPoints) -> Result<(), NufftError> {
        if !self.stream_prefetch || !pts.is_host() {
            return Ok(());
        }
        let ctx = self.dev.ctx.clone();
        let s = self.streamer();
        if s.owner.contains(&Some(pts.id)) {
            return Ok(());
        }
        if s.up.is_none() {
            s.up = Some(ctx.new_stream()?);
        }
        let up = s.up.clone().unwrap();
        let j = (s.last + 1) % s.slots.len();
        if !slot_fits(&s.slots[j], pts) {
            st.synchronize()?;
            up.synchronize()?;
        }
        if let Some(ev) = &s.free[j] {
            up.wait(ev)?;
        }
        let b = fill_slot(&up, pts, &mut s.slots[j], &mut s.staging[j])?;
        let ev = ctx.new_event(None)?;
        ev.record(&up)?;
        s.ready[j] = Some(ev);
        s.owner[j] = Some(pts.id);
        s.uploads += 1;
        s.upload_bytes += b;
        Ok(())
    }
}

fn slot_fits(slot: &Option<SegPoints>, src: &SegPoints) -> bool {
    let h = src.host.as_ref().unwrap();
    slot.as_ref().is_some_and(|s| {
        s.rec.len() >= h.rec.len() && s.tl.len() >= h.tl.len() && s.items.len() >= h.items.len() && match (&h.hi, &s.hi) {
            (Some((u, t)), Some((du, dt))) => du.len() >= u.len() && dt.len() >= t.len(),
            (Some(_), None) => false,
            (None, _) => true,
        }
    })
}

#[cfg(test)]
mod seg_build_tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex64;

    /// the serial implementation before parallelisation (reference)
    fn serial(nv: usize, nseg: usize, t_min: f64, t_max: f64, ntab: usize, fit: &[usize], psi: &dyn Fn(usize, f64) -> f64) -> (Vec<C32>, Vec<C32>) {
        let tl: Vec<f64> = (0..nseg).map(|l| t_min + (t_max - t_min) * l as f64 / (nseg - 1) as f64).collect();
        let mut segph = vec![C32::new(0.0, 0.0); nseg * nv];
        for l in 0..nseg {
            for v in 0..nv {
                let (s, c) = (-psi(v, tl[l])).sin_cos();
                segph[l * nv + v] = C32::new(c as f32, s as f32);
            }
        }
        let b = DMatrix::<Complex64>::from_fn(fit.len(), nseg, |r, l| Complex64::from_polar(1.0, -psi(fit[r], tl[l])));
        let bh = b.adjoint();
        let mut g = &bh * &b;
        let tr = (0..nseg).map(|l| g[(l, l)].re).sum::<f64>() / nseg as f64;
        for l in 0..nseg {
            g[(l, l)] += Complex64::new(1e-9 * tr, 0.0);
        }
        let chol = g.cholesky().unwrap();
        let dt = (t_max - t_min) / (ntab - 1) as f64;
        let mut atab = vec![C32::new(0.0, 0.0); ntab * nseg];
        for i in 0..ntab {
            let t = t_min + i as f64 * dt;
            let y = DVector::<Complex64>::from_iterator(fit.len(), fit.iter().map(|&v| Complex64::from_polar(1.0, -psi(v, t))));
            let a = chol.solve(&(&bh * y));
            for l in 0..nseg {
                atab[i * nseg + l] = C32::new(a[l].re as f32, a[l].im as f32);
            }
        }
        (segph, atab)
    }

    #[test]
    fn parallel_segments_equal_serial() {
        let nv = 20000;
        let om: Vec<f64> = (0..nv).map(|v| 2.0 * std::f64::consts::PI * 300.0 * ((v as f64 * 0.37).sin() + 0.2 * (v as f64 * 0.011).cos())).collect();
        let psi = |v: usize, t: f64| om[v] * t + 40.0 * t * t * (v % 7) as f64;
        let fit: Vec<usize> = (0..nv).step_by(3).collect();
        let (p0, a0) = serial(nv, 9, 1e-4, 4e-3, 700, &fit, &psi);
        let (p1, a1, _, _) = build_segments(nv, 9, 1e-4, 4e-3, 700, &fit, &psi);
        assert!(p0.iter().zip(&p1).all(|(a, b)| a.re == b.re && a.im == b.im), "diagonals differ");
        assert!(a0.iter().zip(&a1).all(|(a, b)| a.re == b.re && a.im == b.im), "interpolators differ");
        // the complex builder with the same time function equals it too
        let g = |v: usize, t: f64| Complex64::from_polar(1.0, -psi(v, t));
        let (p2, a2, _, _) = build_segments_complex(nv, 9, 1e-4, 4e-3, 700, &fit, &g);
        let d = |x: &[C32], y: &[C32]| x.iter().zip(y).map(|(a, b)| ((a.re - b.re).abs()).max((a.im - b.im).abs())).fold(0.0f32, f32::max);
        assert!(d(&p0, &p2) < 1e-6 && d(&a0, &a2) < 1e-5, "complex builder: {} {}", d(&p0, &p2), d(&a0, &a2));
    }
}
