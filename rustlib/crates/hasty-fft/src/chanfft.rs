//! Pruned in-place 3-D FFT of channel-interleaved data `[n2][n1][n0][nch]` (channels fastest, e.g. the coil × segment
//! grids of a multi-channel NUFFT) whose image occupies a *wrapped* block per axis: indices `[0, hi_a) ∪ [n_a − lo_a, n_a)`
//! (the zero-padded NUFFT layout with the image centred on index 0). Everything outside is zero on the way in
//! (forward) or not needed on the way out (inverse), so
//!
//! forward  x-pass: lines (y, z) in the block only, reading the block's x entries; y-pass: planes z in the block,
//!          reading the block's y entries; z-pass: all lines, reading the block's z entries — nothing outside the block
//!          is ever read, so the buffer needs no zeroing before the image is written;
//! inverse  z-pass: all lines, writing the block's z entries; y-pass: planes z in the block, writing the block's y
//!          entries; x-pass: lines (y, z) in the block, writing the block's x entries — only the block is valid after.
//!
//! Fraction of the unpruned work per direction ≈ (h1 h2 + h2 + 1) / 3 · (1 + h) / 2 with h_a the block fraction of axis
//! a: 0.73 at oversampling 1.25, 0.44 at 2. Results equal the full transform on the block (tested to rounding).
//!
//! **k-support pruning** ([`PrunedChanFft3::new_ksupport`], [`KSupport`]): a NUFFT gathers from / scatters into only
//! the k-space cells its samples' kernel supports meet. With that set known (per kx row of the z pass, and its
//! projections on the three axes), the forward also skips the y-pass lines (kx) and z-pass columns (kx, ky) that meet
//! no sample and writes only the needed kx range, the ky range of each kx line and the kz range of each column (the
//! chord of a ball); the inverse skips the same lines and reads only the same ranges (the rest of a scatter grid is
//! zero). Exact on the support: the forward spectrum is correct on every
//! cell of the support (elsewhere undefined), the inverse equals the full one whenever the input is zero outside it.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use hasty_cuda::{Device, Real};

use crate::codegen::{BfpIo, IoHook, KernelFormat, Store};
use crate::plan::{AxisPlan, AxisSpec, Direction, FftError, KArg, LaunchPtrs, LineAddr, LineSplit};

/// The k-space cells a transform must produce (forward) or may read (inverse), as wrapped index ranges
/// `[0, head) ∪ [n − tail, n)` (the FFT output layout, k = 0 at index 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KSupport {
    /// per axis `[x, y, z]` = `[n0, n1, n2]` (x fastest): the projection of the support
    pub proj: [(usize, usize); 3],
    /// per ky index (n1 entries): the kx range of the support's columns (kx, ky); (0, 0) = no column in this row
    pub rows: Vec<(usize, usize)>,
    /// per kx index (n0 entries): the ky range of the support's columns at that kx (the y-pass entries written by the
    /// forward / read by the inverse); (0, 0) = none
    pub ycols: Vec<(usize, usize)>,
    /// per column `ky · n0 + kx`: the kz range of the support on that column (the z-pass entries written / read);
    /// (0, 0) = not in the support
    pub zcols: Vec<(usize, usize)>,
}

impl KSupport {
    /// Everything (no pruning) for a grid `n = [n0, n1, n2]` (x fastest).
    pub fn full(n: [usize; 3]) -> Self {
        Self { proj: n.map(|m| (m, 0)), rows: vec![(n[0], 0); n[1]], ycols: vec![(n[1], 0); n[0]], zcols: vec![(n[2], 0); n[0] * n[1]] }
    }

    /// Smallest wrapped range `[0, h) ∪ [n − t, n)` containing a set of indices given by `has(i)`, split at n / 2
    /// (h ≥ 1: index 0, the k-space centre, is always kept).
    pub fn hull(n: usize, has: impl Fn(usize) -> bool) -> (usize, usize) {
        let s = n / 2;
        let h = (0..s).rev().find(|&i| has(i)).map_or(1, |i| i + 1).max(1);
        let t = (s..n).find(|&i| has(i)).map_or(0, |i| n - i);
        (h, t)
    }

    /// From a column mask `xy(kx, ky)` and one kz mask for all columns (`n = [n0, n1, n2]`).
    pub fn from_masks(n: [usize; 3], xy: impl Fn(usize, usize) -> bool, z: impl Fn(usize) -> bool) -> Self {
        let zr = Self::hull(n[2], z);
        Self::from_columns(n, xy, |_, _| zr)
    }

    /// From a column mask `xy(kx, ky)` and the kz range `zr(kx, ky)` of every column in it (wrapped, as [`hull`](Self::hull)).
    pub fn from_columns(n: [usize; 3], xy: impl Fn(usize, usize) -> bool, zr: impl Fn(usize, usize) -> (usize, usize)) -> Self {
        let rows: Vec<(usize, usize)> = (0..n[1])
            .map(|ky| if (0..n[0]).any(|kx| xy(kx, ky)) { Self::hull(n[0], |kx| xy(kx, ky)) } else { (0, 0) })
            .collect();
        let ycols: Vec<(usize, usize)> = (0..n[0])
            .map(|kx| if (0..n[1]).any(|ky| xy(kx, ky)) { Self::hull(n[1], |ky| xy(kx, ky)) } else { (0, 0) })
            .collect();
        let zcols: Vec<(usize, usize)> = (0..n[0] * n[1]).map(|q| if xy(q % n[0], q / n[0]) { zr(q % n[0], q / n[0]) } else { (0, 0) }).collect();
        let mx = |v: &[(usize, usize)]| (v.iter().map(|r| r.0).max().unwrap_or(1).max(1), v.iter().map(|r| r.1).max().unwrap_or(0));
        let px = mx(&rows);
        let py = Self::hull(n[1], |ky| rows[ky] != (0, 0));
        let pz = mx(&zcols);
        Self { proj: [px, py, pz], rows, ycols, zcols }
    }

    /// fraction of the full z-pass columns, y-pass lines (kx) and per-axis output entries that the support keeps
    pub fn fractions(&self, n: [usize; 3]) -> (f64, [f64; 3]) {
        let cols: usize = self.rows.iter().map(|r| r.0 + r.1).sum();
        (cols as f64 / (n[0] * n[1]) as f64, [0, 1, 2].map(|a| (self.proj[a].0 + self.proj[a].1) as f64 / n[a] as f64))
    }

    /// fraction of the grid's cells in the support (Σ of the column kz ranges)
    pub fn cell_fraction(&self, n: [usize; 3]) -> f64 {
        self.zcols.iter().map(|r| r.0 + r.1).sum::<usize>() as f64 / (n[0] * n[1] * n[2]) as f64
    }

    fn check(&self, n: [usize; 3]) -> Result<(), FftError> {
        let within = |r: (usize, usize), p: (usize, usize)| r.0 <= p.0 && r.1 <= p.1;
        let ok = self.rows.len() == n[1]
            && self.ycols.len() == n[0]
            && self.zcols.len() == n[0] * n[1]
            && (0..3).all(|a| self.proj[a].0 >= 1 && self.proj[a].0 + self.proj[a].1 <= n[a])
            && self.rows.iter().all(|&r| within(r, self.proj[0]))
            && self.ycols.iter().all(|&r| r.0 + r.1 <= n[1])
            && self.zcols.iter().all(|&r| within(r, self.proj[2]));
        if ok { Ok(()) } else { Err(FftError::Invalid(format!("k support does not fit the grid {n:?} (rows {}, proj {:?})", self.rows.len(), self.proj))) }
    }
}

/// Block-floating-point storage of the DRAM-resident intermediates ([`PrunedChanFft3::set_store`]): the forward y
/// pass writes, and the z pass reads, a BFP buffer instead of the grid's own y → z intermediate; the inverse z pass
/// writes and the inverse y pass reads the z → y intermediate the same way. The x ↔ y intermediates stay in the grid
/// buffer (f32, in L2 with slabs).
///
/// Layout: planes zc (the image block's z planes, compact) × rows y (all n1) × lines of the plane (n0 · nch, padded to
/// `nt` whole 16-line tiles); block (zc, y, tile) = 16 complex values of 16 consecutive lines, at block index
/// (zc · n1 + y) · nt + tile. Mantissa words and exponent bytes as `hasty_bfp` (bfp.cuh). In place (default when a
/// plane's lines n0 · nch are whole tiles): the mantissas of block (z, y, tile) occupy the first 4 · b bytes of the
/// grid's own 128-byte cell run of those 16 lines at (z, y), and only the exponent bytes (1 per 16 complex) are extra.
struct BfpStage<R: Real> {
    bits: u32,
    /// forward y (BFP out), forward z (BFP in), inverse z (BFP out), inverse y (BFP in)
    fy: AxisPlan<R>,
    fz: AxisPlan<R>,
    iz: AxisPlan<R>,
    iy: AxisPlan<R>,
    mant: CudaSlice<u32>,
    ex: CudaSlice<u8>,
    /// 16-line tiles per row of a plane
    nt: usize,
    /// mantissas in the grid buffer itself (`BfpIo::inplace`; `mant` is then a 1-word placeholder)
    inplace: bool,
}

/// Drop dead cell runs from the L2 without writing them back (`discard.global.L2`, sm_80+; a no-op below): planes
/// k < gridDim.y at `base + k · pstride`, the byte ranges [lo0, hi0) and [lo1, hi1) of each plane, rounded inwards to
/// whole 128-byte runs (a run that is only partly inside a range is kept).
const DISCARD_SRC: &str = r#"
extern "C" __global__ void l2_discard(unsigned long long base, long long pstride, long long lo0, long long hi0, long long lo1, long long hi1) {
#if __CUDA_ARCH__ >= 800
  const unsigned long long pb = base + (unsigned long long)blockIdx.y * (unsigned long long)pstride;
  for (int r = 0; r < 2; ++r) {
    const unsigned long long a = (pb + (unsigned long long)(r ? lo1 : lo0) + 127ull) & ~127ull, b = (pb + (unsigned long long)(r ? hi1 : hi0)) & ~127ull;
    for (unsigned long long q = a + 128ull * (blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x); q < b; q += 128ull * gridDim.x * blockDim.x)
      asm volatile("discard.global.L2 [%0], 128;" :: "l"(q) : "memory");
  }
#endif
}
"#;

/// whether the hooked inverse of a two-pass L2 slab drops the dead y-pass output of each slab from the L2 after the
/// crop epilogue consumed it (env HASTY_FFT_SLAB_DISCARD = 1 default | 0)
fn slab_discard_default() -> bool {
    std::env::var("HASTY_FFT_SLAB_DISCARD").map_or(true, |v| v != "0")
}

/// pass spec of the pruned transforms: strided, no diagonals; `(n_in, n_in_tail)`, `(n_out, n_out_tail)`
fn axis_spec(len: usize, dir: Direction, scale: f64, addr: LineAddr, (n_in, n_in_tail): (usize, usize), (n_out, n_out_tail): (usize, usize)) -> AxisSpec {
    AxisSpec { n: len, n_in, n_out, n_in_tail, n_out_tail, dir, scale, conv: false, pre: vec![], post: vec![], accumulate: false, contiguous: false, addr, kfmt: KernelFormat::Complex, poly_pre: false, poly_post: false, vox: [1, 1, 1] }
}

pub struct PrunedChanFft3<R: Real> {
    /// `[n2, n1, n0]`
    pub n: [usize; 3],
    pub nch: usize,
    /// per axis `[2, 1, 0]`: (head, tail) of the wrapped block
    pub keep: [(usize, usize); 3],
    /// k-support pruning ([`new_ksupport`](Self::new_ksupport)); None: every k-space line
    pub ksup: Option<KSupport>,
    fwd: [AxisPlan<R>; 3],
    inv: [AxisPlan<R>; 3],
    /// specs of the x passes (forward first, inverse last) and their line split, for hooked variants
    xspec: [AxisSpec; 2],
    split_x: LineSplit,
    /// hooked x passes ([`set_hooks`](Self::set_hooks)): forward with generated input, inverse with an epilogue
    hooked: [Option<AxisPlan<R>>; 2],
    /// z planes per slab of the two-pass L2 path ([`set_slab`](Self::set_slab)); 0: whole-grid passes
    slab: usize,
    /// BFP intermediates ([`set_store`](Self::set_store)); None: f32 in place
    bfp: Option<BfpStage<R>>,
    /// channels holding data ([`set_active_channels`](Self::set_active_channels)); the BFP writers store the others
    /// as zeros
    act_ch: std::sync::atomic::AtomicUsize,
    /// L2 discard kernel of the hooked slab inverse (built with the hooks; None: off or unavailable)
    discard: Option<CudaFunction>,
    /// the forward z pass writing the grid itself as BFP ([`set_grid_store`](Self::set_grid_store)), None: f32 grid
    grid: Option<GridBfp<R>>,
    /// exponent array of the next forward call's BFP grid ([`set_grid_exponents`](Self::set_grid_exponents)); 0: f32
    grid_ex: std::sync::atomic::AtomicU64,
}

/// BFP grid output of the forward transform (research/bfp_storage.md §10): the last z pass stores block (z, y, tile) of
/// 16 lines (x, channel) at entry z in place (mantissas = the first 4 · bits bytes of the 16 lines' 128-byte cell run;
/// a plane's lines n0 · nch must be whole tiles), its exponent byte at index (z · n1 + y) · nt + tile = flat cell
/// index / 16 of the block's first line, in a caller array of n2 · n1 · nt bytes. Blocks the pass does not write (outside
/// a k support) keep exponent byte 0 (cleared every call): they decode as zeros.
struct GridBfp<R: Real> {
    bits: u32,
    fz: AxisPlan<R>,
}

impl<R: Real> PrunedChanFft3<R> {
    /// `n = [n2, n1, n0]`, `keep[a] = (hi, lo)` for the same axes: the block is `[0, hi) ∪ [n − lo, n)`. `scale_inv`
    /// multiplies the inverse (1 = unnormalised, as the forward).
    pub fn new(dev: &Device, n: [usize; 3], nch: usize, keep: [(usize, usize); 3], scale_inv: f64) -> Result<Self, FftError> {
        Self::new_ksupport(dev, n, nch, keep, scale_inv, None)
    }

    /// As [`new`](Self::new), additionally pruned to the k-space support `ks` (module docs): the forward is exact on the
    /// support only, the inverse expects input that is zero outside it. `ks` is in `[x, y, z]` axis order (`n` is
    /// `[n2, n1, n0]`).
    pub fn new_ksupport(dev: &Device, n: [usize; 3], nch: usize, keep: [(usize, usize); 3], scale_inv: f64, ks: Option<&KSupport>) -> Result<Self, FftError> {
        if let Some(k) = ks {
            k.check([n[2], n[1], n[0]])?;
        }
        for a in 0..3 {
            let (h, l) = keep[a];
            if h == 0 || h + l > n[a] {
                return Err(FftError::Invalid(format!("pruned block ({h}, {l}) does not fit axis length {}", n[a])));
            }
        }
        let [n2, n1, n0] = n;
        let cnt = |a: usize| keep[a].0 + keep[a].1;
        let gap = |a: usize| n[a] - cnt(a);
        let (sx, sy, sz) = (nch, n0 * nch, n1 * n0 * nch);
        // tiles run over `la` (stride 1: channels, or channels × x × … where contiguous); the line count `lb` runs over
        // grid (y, z)
        let addr = |la_cnt: usize, lb_cnt: usize, in_sn: usize, lb: usize| LineAddr { la_cnt, lb_cnt, in_la: 1, in_lb: lb, in_sn, out_la: 1, out_lb: lb, out_sn: in_sn };
        let spec = |len: usize, dir: Direction, scale: f64, ad: LineAddr, (n_in, n_in_tail): (usize, usize), (n_out, n_out_tail): (usize, usize)| AxisSpec {
            n: len,
            n_in,
            n_out,
            n_in_tail,
            n_out_tail,
            dir,
            scale,
            conv: false,
            pre: vec![],
            post: vec![],
            accumulate: false,
            contiguous: false,
            addr: ad,
            kfmt: KernelFormat::Complex,
            poly_pre: false,
            poly_post: false,
            vox: [1, 1, 1],
        };
        // lines of the x pass: channels × (y, z) in the block; of the y pass: (channels, x) × z in the block; of the
        // z pass: (channels, x, y)
        let split_x = LineSplit { lb1_cnt: cnt(1), lb1_hi: keep[1].0, lb1_gap: gap(1), lb2_hi: keep[0].0, lb2_gap: gap(0), in_lb2: sz, out_lb2: sz };
        let split_y = LineSplit { lb1_cnt: cnt(0), lb1_hi: keep[0].0, lb1_gap: gap(0), lb2_hi: usize::MAX / 2, lb2_gap: 0, in_lb2: 0, out_lb2: 0 };
        // the x pass has cnt(1) · cnt(0) line groups (384 · 384 at 1 mm full torso): more than 65535 fold into grid
        // (y, z) (`plan::launch_grid`)
        let full = |a: usize| (n[a], 0usize);
        use Direction::*;
        // ranged: 0 = all lines, 1 = per-group line ranges, 2 / 3 = also per-line entry ranges read / written
        let mk = |sp: AxisSpec, split: Option<LineSplit>, ranged: u8| -> Result<AxisPlan<R>, FftError> {
            let mut p = match ranged {
                0 => AxisPlan::from_spec(dev, &sp)?,
                1 => AxisPlan::from_spec_ranged(dev, &sp)?,
                _ => AxisPlan::from_spec_ranged_lines(dev, &sp, ranged == 2, ranged == 3)?,
            };
            if let Some(s) = split {
                p.split = s;
            }
            Ok(p)
        };
        let st = dev.default_stream();
        let xspec = [
            spec(n0, Forward, 1.0, addr(nch, cnt(1) * cnt(0), sx, sy), keep[2], ks.map_or(full(2), |k| k.proj[0])),
            spec(n0, Inverse, scale_inv, addr(nch, cnt(1) * cnt(0), sx, sy), ks.map_or(full(2), |k| k.proj[0]), keep[2]),
        ];
        let (fwd, inv) = match ks {
            None => (
                [
                    mk(spec(n0, Forward, 1.0, addr(nch, cnt(1) * cnt(0), sx, sy), keep[2], full(2)), Some(split_x), 0)?,
                    mk(spec(n1, Forward, 1.0, addr(n0 * nch, cnt(0), sy, sz), keep[1], full(1)), Some(split_y), 0)?,
                    mk(spec(n2, Forward, 1.0, addr(n1 * n0 * nch, 1, sz, 0), keep[0], full(0)), None, 0)?,
                ],
                [
                    mk(spec(n2, Inverse, 1.0, addr(n1 * n0 * nch, 1, sz, 0), full(0), keep[0]), None, 0)?,
                    mk(spec(n1, Inverse, 1.0, addr(n0 * nch, cnt(0), sy, sz), full(1), keep[1]), Some(split_y), 0)?,
                    mk(spec(n0, Inverse, scale_inv, addr(nch, cnt(1) * cnt(0), sx, sy), full(2), keep[2]), Some(split_x), 0)?,
                ],
            ),
            Some(k) => {
                // x pass: lines (y, z) in the block, writing / reading only the support's kx range; y pass: lines
                // (channel, kx) of the kx range × z in the block, writing / reading the ky range of each kx
                // (`ycols`); z pass: per ky row, the (channel, kx) lines of the row's columns, writing / reading the kz
                // range of each column (`zcols`; rows of no column launch idle blocks). Lines (channel, kx) are
                // la = kx · nch + channel in both passes, so the line tables are indexed q = lb · stride + la / nch.
                let [px, py, pz] = k.proj;
                let chr = |(h, t): (usize, usize)| (h * nch, t * nch);
                let mut f = [
                    mk(spec(n0, Forward, 1.0, addr(nch, cnt(1) * cnt(0), sx, sy), keep[2], px), Some(split_x), 0)?,
                    mk(spec(n1, Forward, 1.0, addr(n0 * nch, cnt(0), sy, sz), keep[1], py), Some(split_y), 3)?,
                    mk(spec(n2, Forward, 1.0, addr(n0 * nch, n1, sz, sy), keep[0], pz), None, 3)?,
                ];
                let mut i = [
                    mk(spec(n2, Inverse, 1.0, addr(n0 * nch, n1, sz, sy), pz, keep[0]), None, 2)?,
                    mk(spec(n1, Inverse, 1.0, addr(n0 * nch, cnt(0), sy, sz), py, keep[1]), Some(split_y), 2)?,
                    mk(spec(n0, Inverse, scale_inv, addr(nch, cnt(1) * cnt(0), sx, sy), px, keep[2]), Some(split_x), 0)?,
                ];
                let ylines = vec![chr(px); cnt(0)];
                let zrows: Vec<(usize, usize)> = k.rows.iter().map(|&r| chr(r)).collect();
                f[1].set_arange(&st, &ylines)?;
                f[2].set_arange(&st, &zrows)?;
                i[0].set_arange(&st, &zrows)?;
                i[1].set_arange(&st, &ylines)?;
                f[1].set_lrange(&st, &k.ycols, nch, 0)?;
                i[1].set_lrange(&st, &k.ycols, nch, 0)?;
                f[2].set_lrange(&st, &k.zcols, nch, n0)?;
                i[0].set_lrange(&st, &k.zcols, nch, n0)?;
                (f, i)
            }
        };
        Ok(Self { n, nch, keep, ksup: ks.cloned(), fwd, inv, xspec, split_x, hooked: [None, None], slab: 0, bfp: None, act_ch: std::sync::atomic::AtomicUsize::new(nch), discard: None, grid: None, grid_ex: std::sync::atomic::AtomicU64::new(0) })
    }

    /// bytes of one z plane of the grid (all channels): the working set per plane of the x + y passes
    pub fn plane_bytes(&self) -> usize {
        self.n[1] * self.n[2] * self.nch * 2 * R::BYTES
    }

    /// Two-pass L2 path: run the x and y passes slab by slab of `dz` z planes of the image block (forward: x then y
    /// per slab; inverse: y then x), so the x pass's output is read back by the y pass from L2 rather than DRAM when
    /// a slab (`dz` × [`plane_bytes`](Self::plane_bytes)) fits the L2 with room for the other traffic. 0: off (one
    /// launch per pass over the whole grid). Results are identical (the same kernels on the same lines).
    pub fn set_slab(&mut self, dz: usize) {
        self.slab = dz;
    }

    pub fn slab(&self) -> usize {
        self.slab
    }

    /// planes per slab that fit `frac` of the device's L2 (CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE); 0 if not even one
    pub fn l2_slab_planes(&self, dev: &Device, frac: f64) -> Result<usize, FftError> {
        let l2 = dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)? as f64;
        Ok((frac * l2 / self.plane_bytes() as f64).floor() as usize)
    }

    /// the slabs (first grid z plane, planes): chunks of `slab` planes of the block's head and tail z ranges
    fn slabs(&self) -> Vec<(usize, usize)> {
        let (h, t) = self.keep[0];
        let mut v = Vec::new();
        for (a, b) in [(0, h), (self.n[0] - t, self.n[0])] {
            let mut z = a;
            while z < b {
                let dz = self.slab.min(b - z);
                v.push((z, dz));
                z += dz;
            }
        }
        v
    }

    /// the forward passes; `extra`: hooked first pass with these arguments (+ the z offset `hz0`)
    fn fwd_impl<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B, extra: Option<&[KArg]>) -> Result<(), FftError> {
        // the BFP grid request applies to this call only, also when it fails below
        let gex = self.grid_ex.swap(0, std::sync::atomic::Ordering::Relaxed);
        if buf.len() < self.len() {
            return Err(FftError::Invalid(format!("buffer of {} elements, the pruned transform needs {}", buf.len(), self.len())));
        }
        let xp = match extra {
            Some(_) => self.hooked[0].as_ref().ok_or_else(|| FftError::Invalid("forward_hooked: no load hook set".into()))?,
            None => &self.fwd[0],
        };
        let ext = |z0: usize| -> Vec<KArg> { extra.map(|e| e.iter().copied().chain([KArg::I64(z0 as i64)]).collect()).unwrap_or_default() };
        let (p, _g) = buf.device_ptr_mut(stream);
        let lp = LaunchPtrs { inp: p, out: p, ..Default::default() };
        // BFP intermediate: the y pass writes it, the z pass reads it (unwritten blocks of a k-support plan read as
        // zeros: exponent bytes cleared first)
        let bs = self.bfp.as_ref();
        let (yp, zp) = bs.map_or((&self.fwd[1], &self.fwd[2]), |b| (&b.fy, &b.fz));
        let bg = bs.map(|b| (b.mant.device_ptr(stream), b.ex.device_ptr(stream)));
        let bptr = |z0: usize| -> [u64; 2] { match (&bg, bs) { (Some(((m, _), (e, _))), Some(b)) => { let o = (self.zc(z0) * self.n[1] * b.nt) as u64; [if b.inplace { e + o } else { m + o * b.bits as u64 * 4 }, e + o] } _ => [0, 0] } };
        self.clear_unwritten(stream)?;
        let bch = self.take_bch();
        // BFP grid output: the z pass of `grid` with the caller's exponent array (cleared first with a k support)
        let gz = if gex != 0 { self.grid.as_ref() } else { None };
        if let Some(g) = gz {
            if self.ksup.is_some() {
                unsafe { cudarc::driver::result::memset_d8_async(gex, 0, self.grid_exponent_bytes(), stream.cu_stream())? };
            }
            let _ = g.bits;
        }
        if self.slab == 0 {
            xp.launch_ext(stream, &lp, &ext(0))?;
            yp.launch(stream, &LaunchPtrs { bout: bptr(0), bch, ..lp })?;
        } else {
            for (z0, dz) in self.slabs() {
                let q = p + (z0 * self.n[1] * self.n[2] * self.nch * 2 * R::BYTES) as u64;
                let lq = LaunchPtrs { inp: q, out: q, ..Default::default() };
                xp.launch_lines(stream, &lq, &ext(z0), Some(self.sub_x(dz)))?;
                yp.launch_lines(stream, &LaunchPtrs { bout: bptr(z0), bch, ..lq }, &[], Some(Self::sub_y(dz)))?;
            }
        }
        match gz {
            Some(g) => g.fz.launch(stream, &LaunchPtrs { bin: bptr(0), bout: [gex, gex], bch, ..lp }),
            None => zp.launch(stream, &LaunchPtrs { bin: bptr(0), ..lp }),
        }
    }

    /// the inverse passes; `extra`: hooked last pass with these arguments (+ `hz0`)
    fn inv_impl<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B, extra: Option<&[KArg]>) -> Result<(), FftError> {
        if buf.len() < self.len() {
            return Err(FftError::Invalid(format!("buffer of {} elements, the pruned transform needs {}", buf.len(), self.len())));
        }
        let xp = match extra {
            Some(_) => self.hooked[1].as_ref().ok_or_else(|| FftError::Invalid("inverse_hooked: no epilogue hook set".into()))?,
            None => &self.inv[2],
        };
        let ext = |z0: usize| -> Vec<KArg> { extra.map(|e| e.iter().copied().chain([KArg::I64(z0 as i64)]).collect()).unwrap_or_default() };
        let (p, _g) = buf.device_ptr_mut(stream);
        let lp = LaunchPtrs { inp: p, out: p, ..Default::default() };
        // BFP intermediate: the z pass writes it, the y pass reads it
        let bs = self.bfp.as_ref();
        let (zp, yp) = bs.map_or((&self.inv[0], &self.inv[1]), |b| (&b.iz, &b.iy));
        let bg = bs.map(|b| (b.mant.device_ptr(stream), b.ex.device_ptr(stream)));
        let bptr = |z0: usize| -> [u64; 2] { match (&bg, bs) { (Some(((m, _), (e, _))), Some(b)) => { let o = (self.zc(z0) * self.n[1] * b.nt) as u64; [if b.inplace { e + o } else { m + o * b.bits as u64 * 4 }, e + o] } _ => [0, 0] } };
        self.clear_unwritten(stream)?;
        let bch = self.take_bch();
        zp.launch(stream, &LaunchPtrs { bout: bptr(0), bch, ..lp })?;
        if self.slab == 0 {
            yp.launch(stream, &LaunchPtrs { bin: bptr(0), ..lp })?;
            xp.launch_ext(stream, &lp, &ext(0))
        } else {
            // hooked: the y pass's output of a slab is dead once the crop epilogue has read it; dropping it from the
            // L2 saves its write-back (the image block's rows of the slab's planes; their contents are undefined
            // afterwards, which inverse_hooked allows)
            let dk = if extra.is_some() { self.discard.as_ref() } else { None };
            let pb = self.plane_bytes() as i64;
            let rowb = (self.n[2] * self.nch * 2 * R::BYTES) as i64;
            let (h1, t1) = self.keep[1];
            for (z0, dz) in self.slabs() {
                let q = p + (z0 * self.n[1] * self.n[2] * self.nch * 2 * R::BYTES) as u64;
                let lq = LaunchPtrs { inp: q, out: q, ..Default::default() };
                yp.launch_lines(stream, &LaunchPtrs { bin: bptr(z0), ..lq }, &[], Some(Self::sub_y(dz)))?;
                xp.launch_lines(stream, &lq, &ext(z0), Some(self.sub_x(dz)))?;
                if let Some(f) = dk {
                    let (lo0, hi0, lo1, hi1) = (0i64, h1 as i64 * rowb, (self.n[1] - t1) as i64 * rowb, self.n[1] as i64 * rowb);
                    let lines = ((h1 + t1) as i64 * rowb / 128).max(1) as u32;
                    let cfg = LaunchConfig { grid_dim: (lines.div_ceil(256).min(64), dz as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                    let mut b = stream.launch_builder(f);
                    b.arg(&q).arg(&pb).arg(&lo0).arg(&hi0).arg(&lo1).arg(&hi1);
                    unsafe { b.launch(cfg)? };
                }
            }
            Ok(())
        }
    }

    /// Channels 0 .. n of the buffer hold data for the next calls (a short coil batch; default all `nch`). Only the BFP
    /// intermediates use it: their writers store the other channels as zeros, so stale values there cannot set a
    /// block's exponent (a block of 16 lines mixes channels). The f32 passes transform every channel regardless.
    /// The count applies to the next forward or inverse call only (it is reset to `nch` after every call).
    pub fn set_active_channels(&self, n: usize) {
        self.act_ch.store(n.clamp(1, self.nch), std::sync::atomic::Ordering::Relaxed);
    }

    /// the channel mask of this call (read once per call); resets the count to all channels for the next call
    fn take_bch(&self) -> [i64; 2] {
        [self.nch as i64, self.act_ch.swap(self.nch, std::sync::atomic::Ordering::Relaxed) as i64]
    }

    /// compact z plane index of grid plane z of the image block
    fn zc(&self, z: usize) -> usize {
        let (h, t) = self.keep[0];
        if z < h { z } else { z - (self.n[0] - h - t) }
    }

    /// k-support plans with BFP intermediates: clear the exponent bytes, so that blocks the writing pass skips (tiles
    /// or entries outside the support) read as zeros
    fn clear_unwritten(&self, stream: &Arc<CudaStream>) -> Result<(), FftError> {
        if let (Some(b), Some(_)) = (&self.bfp, &self.ksup) {
            let (e, _g) = b.ex.device_ptr(stream);
            unsafe { cudarc::driver::result::memset_d8_async(e, 0, b.ex.len(), stream.cu_stream())? };
        }
        Ok(())
    }

    /// Storage of the y → z (forward) and z → y (inverse) intermediates: [`Store::F32`] (default: in place in the
    /// grid buffer) or [`Store::Bfp`]`(b)`, b in 16 ..= 24 (a separate buffer of [`store_bytes`](Self::store_bytes),
    /// research/bfp_storage.md §7 step 1). The x ↔ y intermediates stay f32 in the grid buffer. Kept over
    /// [`set_slab`](Self::set_slab). Errors for an invalid width (the plan keeps its previous storage) and for lengths
    /// the BFP passes do not cover: Bluestein or single-stage y / z lengths, or y / z tiles of fewer than 16 lines (shared memory: on a
    /// 99 KB-per-block device, two-buffer y / z lengths above 396). On an error the plan is left with f32 storage (a
    /// previous BFP buffer is freed before the new one is allocated) — also on an allocation error.
    ///
    /// Notes: the BFP buffer belongs to the plan, so calls of this plan on different streams must be ordered by the
    /// caller (f32 plans touch only the caller's buffer). A non-finite value makes its whole 16-line block NaN (15
    /// neighbouring lines of the tile), where the f32 path keeps it in its own line.
    /// whether BFP intermediates can live in the grid buffer itself (a plane's lines n0 · nch are whole 16-line
    /// tiles); otherwise [`set_store_opts`](Self::set_store_opts) needs the separate (dense) buffer
    pub fn bfp_inplace_possible(&self) -> bool {
        (self.n[2] * self.nch) % 16 == 0
    }

    pub fn set_store(&mut self, dev: &Device, store: Store) -> Result<(), FftError> {
        self.set_store_opts(dev, store, false)
    }

    /// [`set_store`](Self::set_store) with the layout chosen: `dense` = a separate buffer of 4 · b + 1 bytes per
    /// 16-complex block (fewer DRAM sectors: e.g. bfp22 88 + 1 B per block); otherwise in place in the grid buffer when
    /// a plane's lines n0 · nch are whole 16-line tiles (mantissas in the cells of the block's own lines: 4 · b bytes of
    /// their 128, i.e. ⌈4 b / 32⌉ DRAM sectors of 4; extra memory only the exponent bytes), else dense.
    pub fn set_store_opts(&mut self, dev: &Device, store: Store, dense: bool) -> Result<(), FftError> {
        let bits = match store {
            Store::F32 => {
                self.bfp = None;
                return self.regrid(dev);
            }
            Store::Bfp(b) if (16..=24).contains(&b) => b,
            Store::Bfp(b) => return Err(FftError::Invalid(format!("BFP intermediates: {b} mantissa bits (16 ..= 24)"))),
        };
        self.bfp = None;
        // (the BFP grid pass reads the y → z intermediate: rebuilt now for f32, and again below once it exists)
        self.regrid(dev)?;
        let n = self.n;
        let [n2, n1, n0] = n;
        let (nch, keep) = (self.nch, self.keep);
        let cnt = |a: usize| keep[a].0 + keep[a].1;
        let gap = |a: usize| n[a] - cnt(a);
        let (lp, sy, sz) = (n0 * nch, n0 * nch, n1 * n0 * nch);
        let nt = lp.div_ceil(16);
        let addr = |la_cnt: usize, lb_cnt: usize, in_sn: usize, lb: usize| LineAddr { la_cnt, lb_cnt, in_la: 1, in_lb: lb, in_sn, out_la: 1, out_lb: lb, out_sn: in_sn };
        let split_y = LineSplit { lb1_cnt: cnt(0), lb1_hi: keep[0].0, lb1_gap: gap(0), lb2_hi: usize::MAX / 2, lb2_gap: 0, in_lb2: 0, out_lb2: 0 };
        let full = |a: usize| (n[a], 0usize);
        let inplace = !dense && lp % 16 == 0;
        // in-place outputs may drop their cell runs from L2 before rewriting them (BfpIo::discard)
        // (default off: measured on the L2-slab y pass it removes the dirty last sector's write-back, −20 % of the
        // slab passes' DRAM writes, but the discards cost more issue time than the writes did, +1.1 ms per transform
        // at 320³ × 24 channels; env HASTY_FFT_BFP_DISCARD=1: on)
        let discard = inplace && std::env::var("HASTY_FFT_BFP_DISCARD").is_ok_and(|v| v == "1");
        let io = |compact: bool| Some(BfpIo { bits, compact, inplace, discard: discard && !compact });
        use Direction::*;
        // y passes: lines (channel, x) of the plane × the block's z planes, entries y (BFP entry index = y); z passes:
        // lines (channel, x) × rows y, entries z (BFP entry index = the compact block plane)
        let st = dev.default_stream();
        let (yr, zr) = match &self.ksup {
            None => ((n1, (0usize, 0usize)), (n2, (0usize, 0usize))),
            Some(k) => ((n1, k.proj[1]), (n2, k.proj[2])),
        };
        let ks = self.ksup.clone();
        let ranged = ks.is_some();
        let (py, pz) = if ranged { (yr.1, zr.1) } else { (full(1), full(0)) };
        let mut fy = AxisPlan::from_spec_bfp(dev, &axis_spec(n1, Forward, 1.0, addr(lp, cnt(0), sy, sz), keep[1], py), ranged, false, ranged, None, io(false))?;
        let mut fz = AxisPlan::from_spec_bfp(dev, &axis_spec(n2, Forward, 1.0, addr(lp, n1, sz, sy), keep[0], pz), ranged, false, ranged, io(true), None)?;
        let mut iz = AxisPlan::from_spec_bfp(dev, &axis_spec(n2, Inverse, 1.0, addr(lp, n1, sz, sy), pz, keep[0]), ranged, ranged, false, None, io(true))?;
        let mut iy = AxisPlan::from_spec_bfp(dev, &axis_spec(n1, Inverse, 1.0, addr(lp, cnt(0), sy, sz), py, keep[1]), ranged, ranged, false, io(false), None)?;
        fy.split = split_y;
        iy.split = split_y;
        let (nt_i, pl) = (nt as i64, (n1 * nt) as i64);
        fy.bfp_strides = [0, 0, nt_i, pl];
        iy.bfp_strides = [nt_i, pl, 0, 0];
        fz.bfp_strides = [pl, nt_i, 0, 0];
        iz.bfp_strides = [0, 0, pl, nt_i];
        if let Some(k) = &ks {
            let chr = |(h, t): (usize, usize)| (h * nch, t * nch);
            let ylines = vec![chr(k.proj[0]); cnt(0)];
            let zrows: Vec<(usize, usize)> = k.rows.iter().map(|&r| chr(r)).collect();
            fy.set_arange(&st, &ylines)?;
            iy.set_arange(&st, &ylines)?;
            fz.set_arange(&st, &zrows)?;
            iz.set_arange(&st, &zrows)?;
            fy.set_lrange(&st, &k.ycols, nch, 0)?;
            iy.set_lrange(&st, &k.ycols, nch, 0)?;
            fz.set_lrange(&st, &k.zcols, nch, n0)?;
            iz.set_lrange(&st, &k.zcols, nch, n0)?;
        }
        let blocks = cnt(0) * n1 * nt;
        let mant = st.alloc_zeros::<u32>(if inplace { 1 } else { blocks * bits as usize })?;
        let ex = st.alloc_zeros::<u8>(blocks)?;
        self.bfp = Some(BfpStage { bits, fy, fz, iz, iy, mant, ex, nt, inplace });
        if let Err(e) = self.regrid(dev) {
            // roll back to f32 intermediates (the documented state after an error), the grid pass rebuilt for them
            self.bfp = None;
            let _ = self.regrid(dev);
            return Err(e);
        }
        Ok(())
    }

    /// rebuild the BFP grid pass for the current y → z storage
    fn regrid(&mut self, dev: &Device) -> Result<(), FftError> {
        match self.grid.as_ref().map(|g| g.bits) {
            Some(b) => self.set_grid_store(dev, Store::Bfp(b)),
            None => Ok(()),
        }
    }

    /// storage of the y ↔ z intermediates ([`set_store`](Self::set_store))
    pub fn store(&self) -> Store {
        self.bfp.as_ref().map_or(Store::F32, |b| Store::Bfp(b.bits))
    }

    /// device bytes of the BFP intermediate buffer beyond the grid buffer (0 with f32 storage; in place: the exponent
    /// bytes only)
    pub fn store_bytes(&self) -> usize {
        self.bfp.as_ref().map_or(0, |b| if b.inplace { b.ex.len() } else { b.mant.len() * 4 + b.ex.len() })
    }

    /// whether the BFP intermediates live in the grid buffer ([`set_store_opts`](Self::set_store_opts))
    pub fn store_inplace(&self) -> bool {
        self.bfp.as_ref().is_some_and(|b| b.inplace)
    }

    /// Extra device bytes [`set_store_opts`](Self::set_store_opts) would allocate for a grid `n = [n2, n1, n0]` with
    /// `nch` channels and an image block of `nz` z planes (for memory budgets before the plan exists).
    pub fn store_bytes_for(store: Store, dense: bool, n: [usize; 3], nch: usize, nz: usize) -> usize {
        match store {
            Store::F32 => 0,
            Store::Bfp(b) => {
                let lp = n[2] * nch;
                let blocks = nz * n[1] * lp.div_ceil(16);
                if !dense && lp % 16 == 0 { blocks } else { blocks * (4 * b as usize + 1) }
            }
        }
    }

    /// Debug / test: the BFP intermediate's mantissa words and exponent bytes (layout of [`BfpStage`]: planes of the
    /// image block's z range × n1 rows × `tiles` 16-line tiles; returns `tiles`), after the last transform.
    #[doc(hidden)]
    pub fn debug_store_buffer(&self, stream: &Arc<CudaStream>) -> Result<Option<(Vec<u32>, Vec<u8>, usize)>, FftError> {
        match &self.bfp {
            Some(b) => Ok(Some((stream.clone_dtoh(&b.mant)?, stream.clone_dtoh(&b.ex)?, b.nt))),
            None => Ok(None),
        }
    }

    /// Debug / test: run some of the f32 in-place passes over the whole grid (`which`: indices into [x, y, z] forward
    /// or [z, y, x] inverse), e.g. to read the f32 intermediates the BFP buffer replaces.
    #[doc(hidden)]
    pub fn debug_f32_passes<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B, forward: bool, which: &[usize]) -> Result<(), FftError> {
        let (p, _g) = buf.device_ptr_mut(stream);
        let plans = if forward { &self.fwd } else { &self.inv };
        for &i in which {
            plans[i].launch(stream, &LaunchPtrs { inp: p, out: p, ..Default::default() })?;
        }
        Ok(())
    }

    /// Debug / test: fill the BFP intermediate with garbage (non-finite exponent bytes, random mantissas), so that
    /// any block a reading pass uses without a writer having written it shows up as NaN.
    #[doc(hidden)]
    pub fn debug_poison_store(&mut self, stream: &Arc<CudaStream>) -> Result<(), FftError> {
        if let Some(b) = self.bfp.as_mut() {
            let m: Vec<u32> = (0..b.mant.len() as u32).map(|i| i.wrapping_mul(2654435761)).collect();
            stream.memcpy_htod(&m, &mut b.mant)?;
            stream.memcpy_htod(&vec![255u8; b.ex.len()], &mut b.ex)?;
        }
        Ok(())
    }

    /// x-pass lines of one slab of dz planes (pointer at its first plane): the block's y lines × dz, no z gap
    fn sub_x(&self, dz: usize) -> (usize, LineSplit) {
        let sz = self.n[1] * self.n[2] * self.nch;
        let c1 = self.keep[1].0 + self.keep[1].1;
        (c1 * dz, LineSplit { lb1_cnt: c1, lb1_hi: self.keep[1].0, lb1_gap: self.n[1] - c1, lb2_hi: usize::MAX / 2, lb2_gap: 0, in_lb2: sz, out_lb2: sz })
    }

    /// y-pass line groups of one slab: its dz planes, no gap
    fn sub_y(dz: usize) -> (usize, LineSplit) {
        (dz, LineSplit { lb1_cnt: dz, lb1_hi: usize::MAX / 2, lb1_gap: 0, lb2_hi: usize::MAX / 2, lb2_gap: 0, in_lb2: 0, out_lb2: 0 })
    }

    /// Fuse caller code into the x passes ([`IoHook`]): `load` generates the input of the first forward pass (the
    /// buffer's image block is then never read: e.g. a NUFFT pad), `epilogue` consumes the output of the last inverse
    /// pass (the block is then never written: e.g. a NUFFT crop). In the hooks, `lb1` / `lb2 + hz0` are the grid y / z
    /// indices (`n[1]`, `n[0]` axes; the hooks' last parameter must be `long hz0`, the z offset of each launch, which
    /// the hooked calls append) and `e` the x index; `la` the channel. None leaves that direction without a hooked
    /// variant. Run them with [`forward_hooked`](Self::forward_hooked) / [`inverse_hooked`](Self::inverse_hooked).
    pub fn set_hooks(&mut self, dev: &Device, load: Option<IoHook>, epilogue: Option<IoHook>) -> Result<(), FftError> {
        let mk = |sp: &AxisSpec, h: IoHook| -> Result<AxisPlan<R>, FftError> {
            let mut p = AxisPlan::from_spec_hooked(dev, sp, h)?;
            p.split = self.split_x;
            Ok(p)
        };
        let f = load.map(|h| mk(&self.xspec[0], IoHook { epilogue: None, ..h })).transpose()?;
        let i = epilogue.map(|h| mk(&self.xspec[1], IoHook { load: None, prologue: None, ..h })).transpose()?;
        self.discard = if i.is_some() && slab_discard_default() {
            dev.load_function("l2_discard", DISCARD_SRC, "l2_discard", &hasty_cuda::CompileSettings::default()).ok()
        } else {
            None
        };
        self.hooked = [f, i];
        Ok(())
    }

    /// (lines per block, threads per line) of the x passes (the hooks' lane layout)
    pub fn x_shape(&self) -> (usize, usize) {
        (self.inv[2].cfg.lpb, self.inv[2].cfg.tpl)
    }

    /// whether hooked (forward, inverse) x passes are set
    pub fn hooks(&self) -> (bool, bool) {
        (self.hooked[0].is_some(), self.hooked[1].is_some())
    }

    /// in place: the hooked first pass generates the block (`extra`: the load hook's arguments without its last,
    /// `long hz0`, which this call appends: the z offset of each launch, absolute z = lb2 + hz0), then the y and z
    /// passes as [`forward`](Self::forward)
    pub fn forward_hooked<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B, extra: &[KArg]) -> Result<(), FftError> {
        self.fwd_impl(stream, buf, Some(extra))
    }

    /// in place: the z and y passes as [`inverse`](Self::inverse), then the hooked last pass hands its output to the
    /// epilogue (`extra`: its arguments without the trailing `long hz0`, appended here); the buffer's block is left as
    /// the y pass wrote it, except with two-pass L2 slabs and HASTY_FFT_SLAB_DISCARD on (default), where its rows are
    /// dropped from the L2 without write-back and hold undefined bytes afterwards
    pub fn inverse_hooked<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B, extra: &[KArg]) -> Result<(), FftError> {
        self.inv_impl(stream, buf, Some(extra))
    }

    pub fn len(&self) -> usize {
        self.n.iter().product::<usize>() * self.nch
    }

    fn run<B: DevicePtrMut<R::Complex>>(plans: &[AxisPlan<R>; 3], stream: &Arc<CudaStream>, buf: &mut B, need: usize) -> Result<(), FftError> {
        if buf.len() < need {
            return Err(FftError::Invalid(format!("buffer of {} elements, the pruned transform needs {need}", buf.len())));
        }
        let (p, _g) = buf.device_ptr_mut(stream);
        for ax in plans {
            ax.launch(stream, &LaunchPtrs { inp: p, out: p, ..Default::default() })?;
        }
        Ok(())
    }

    /// in place: block data in, full spectrum out (entries outside the block are never read)
    pub fn forward<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B) -> Result<(), FftError> {
        let g = self.grid.is_some() && self.grid_ex.load(std::sync::atomic::Ordering::Relaxed) != 0;
        if self.slab == 0 && self.bfp.is_none() && !g { Self::run(&self.fwd, stream, buf, self.len()) } else { self.fwd_impl(stream, buf, None) }
    }

    /// The forward z pass may store the grid itself as BFP (`Store::Bfp(b)`, b in 16 ..= 24; [`GridBfp`]): in place in
    /// the grid buffer, the exponent bytes in a caller array of [`grid_exponent_bytes`](Self::grid_exponent_bytes)
    /// passed per call with [`set_grid_exponents`](Self::set_grid_exponents) (a call without one writes the f32 grid).
    /// Needs a plane's lines n0 · nch to be whole 16-line tiles and a z length the BFP passes cover; errors otherwise
    /// (the grid store is then f32). Kept over [`set_store`](Self::set_store) (rebuilt with the y → z storage).
    pub fn set_grid_store(&mut self, dev: &Device, store: Store) -> Result<(), FftError> {
        self.grid = None;
        let bits = match store {
            Store::F32 => return Ok(()),
            Store::Bfp(b) if (16..=24).contains(&b) => b,
            Store::Bfp(b) => return Err(FftError::Invalid(format!("BFP grid: {b} mantissa bits (16 ..= 24)"))),
        };
        if !self.bfp_inplace_possible() {
            return Err(FftError::Invalid("BFP grid: a plane's lines n0 · nch are not whole 16-line tiles".into()));
        }
        let [n2, n1, n0] = self.n;
        let (nch, keep) = (self.nch, self.keep);
        let (lp, sy, sz) = (n0 * nch, n0 * nch, n1 * n0 * nch);
        let nt = lp / 16;
        let addr = LineAddr { la_cnt: lp, lb_cnt: n1, in_la: 1, in_lb: sy, in_sn: sz, out_la: 1, out_lb: sy, out_sn: sz };
        let ranged = self.ksup.is_some();
        let pz = self.ksup.as_ref().map_or((n2, 0), |k| k.proj[2]);
        let bin = self.bfp.as_ref().map(|b| BfpIo { bits: b.bits, compact: true, inplace: b.inplace, discard: false });
        let bout = BfpIo { bits, compact: false, inplace: true, discard: false };
        let mut fz = AxisPlan::from_spec_bfp(dev, &axis_spec(n2, Direction::Forward, 1.0, addr, keep[0], pz), ranged, false, ranged, bin, Some(bout))?;
        let pl = self.bfp.as_ref().map_or(0, |b| (n1 * b.nt) as i64);
        fz.bfp_strides = [pl, nt as i64, (n1 * nt) as i64, nt as i64];
        if let Some(k) = &self.ksup {
            let st = dev.default_stream();
            let zrows: Vec<(usize, usize)> = k.rows.iter().map(|&(h, t)| (h * nch, t * nch)).collect();
            fz.set_arange(&st, &zrows)?;
            fz.set_lrange(&st, &k.zcols, nch, n0)?;
        }
        self.grid = Some(GridBfp { bits, fz });
        Ok(())
    }

    /// storage of the forward grid output ([`set_grid_store`](Self::set_grid_store))
    pub fn grid_store(&self) -> Store {
        self.grid.as_ref().map_or(Store::F32, |g| Store::Bfp(g.bits))
    }

    /// exponent bytes of a BFP grid: one per 16-line block of every (z, y) row, n2 · n1 · n0 · nch / 16
    pub fn grid_exponent_bytes(&self) -> usize {
        self.len() / 16
    }

    /// The next forward call stores the grid as BFP with its exponent bytes at device address `ex`
    /// ([`grid_exponent_bytes`](Self::grid_exponent_bytes) bytes; 0: f32). Applies to that call only.
    pub fn set_grid_exponents(&self, ex: u64) {
        self.grid_ex.store(ex, std::sync::atomic::Ordering::Relaxed);
    }

    /// in place: full spectrum in, the block of the inverse transform out (the rest is left undefined)
    pub fn inverse<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B) -> Result<(), FftError> {
        if self.slab == 0 && self.bfp.is_none() { Self::run(&self.inv, stream, buf, self.len()) } else { self.inv_impl(stream, buf, None) }
    }
}
