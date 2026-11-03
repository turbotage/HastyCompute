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

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use hasty_fft::{Direction, FftPlan, Norm};

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

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// grid-slot pointers of one launch (frame offset o → slot), passed by value
#[repr(C)]
#[derive(Clone, Copy)]
struct Grids {
    p: [u64; 4],
}
unsafe impl DeviceRepr for Grids {}

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
}

/// Segment diagonals P_l and interpolator table a_l(τ), built by [`SegNufft::segments`] (e.g. one per encode).
pub struct SegSegments {
    segph: CudaSlice<C32>,
    atab: CudaSlice<C32>,
    tab: (usize, f32, f32),
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
    fft_fwd: FftPlan<f32>,
    fft_inv: FftPlan<f32>,
    smaps: Option<CudaSlice<C32>>,
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
        let f = Funcs { pad: module.load_function("seg_pad")?, crop: module.load_function("seg_crop")?, fwd: tile_fn(&module, "seg_forward_t", tc.smem(sh)[1])?, adj: tile_fn(&module, "seg_adjoint_t", tc.smem(sh)[0])? };
        let nch = cb * opts.nseg;
        let fd = [plan.nf[2], plan.nf[1], plan.nf[0], nch];
        let fft_fwd = FftPlan::<f32>::new_partial(dev, &fd, &[true, true, true, false], Direction::Forward, Norm::None)?;
        let fft_inv = FftPlan::<f32>::new_partial(dev, &fd, &[true, true, true, false], Direction::Inverse, Norm::None)?;
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
            fft_fwd,
            fft_inv,
            smaps: None,
        })
    }

    fn shape(&self) -> Shape {
        Shape { nseg: self.nseg, order: self.order, nf: self.nf }
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
        Ok(SegSegments { segph: st.clone_htod(segph)?, atab: st.clone_htod(atab)?, tab: (atab.len() / self.nseg, t0 as f32, dt as f32) })
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
        Ok(SegPoints { rec: st.clone_htod(&rec)?, tl: st.clone_htod(&tl_all)?, items: st.clone_htod(items_all.as_flattened())?, ivals, m })
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
        st.memset_zeros(&mut self.slots[slot])?;
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
        self.fft_fwd.execute_inplace(st, &mut self.slots[slot])?;
        Ok(())
    }

    /// x_f += crop(IFFT(slot)) summed over the batch coils and segments
    #[allow(clippy::too_many_arguments)]
    fn flush_frame(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, slot: usize, x: &mut CudaSlice<C32>, f: usize, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let nv = self.nv();
        let (md, d, c0_i, ncb_i, acc) = (self.mdims(), self.plan.dims(), c0 as i32, ncb as i32, 1i32);
        self.fft_inv.execute_inplace(st, &mut self.slots[slot])?;
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
        let (nv, m) = (self.nv(), pts.m);
        assert!(x.len() >= self.frames * nv && y.len() >= self.nc * m);
        let (fwd, tc) = (self.f.fwd.clone(), self.tiles_fwd);
        let mut c0 = 0;
        while c0 < self.nc {
            let ncb = self.cb.min(self.nc - c0);
            st.memset_zeros(&mut y.slice_mut(c0 * m..(c0 + ncb) * m))?;
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
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex64;
    let tl: Vec<f64> = (0..nseg).map(|l| if nseg == 1 { 0.5 * (t_min + t_max) } else { t_min + (t_max - t_min) * l as f64 / (nseg - 1) as f64 }).collect();
    let mut segph = vec![C32::new(0.0, 0.0); nseg * nv];
    for l in 0..nseg {
        for v in 0..nv {
            let (s, c) = (-psi(v, tl[l])).sin_cos();
            segph[l * nv + v] = C32::new(c as f32, s as f32);
        }
    }
    let b = DMatrix::<Complex64>::from_fn(fit_voxels.len(), nseg, |r, l| Complex64::from_polar(1.0, -psi(fit_voxels[r], tl[l])));
    let bh = b.adjoint();
    let mut g = &bh * &b;
    let tr = (0..nseg).map(|l| g[(l, l)].re).sum::<f64>() / nseg as f64;
    for l in 0..nseg {
        g[(l, l)] += Complex64::new(1e-9 * tr, 0.0);
    }
    let chol = g.cholesky().expect("segment Gram matrix");
    let dt = (t_max - t_min) / (ntab - 1) as f64;
    let mut atab = vec![C32::new(0.0, 0.0); ntab * nseg];
    for i in 0..ntab {
        let t = t_min + i as f64 * dt;
        let y = DVector::<Complex64>::from_iterator(fit_voxels.len(), fit_voxels.iter().map(|&v| Complex64::from_polar(1.0, -psi(v, t))));
        let a = chol.solve(&(&bh * y));
        for l in 0..nseg {
            atab[i * nseg + l] = C32::new(a[l].re as f32, a[l].im as f32);
        }
    }
    (segph, atab, t_min, dt)
}
