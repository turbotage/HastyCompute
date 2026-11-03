use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use hasty_cuda::{CompileSettings, Device, Real};
use hasty_fft::{Direction, FftPlan, Norm};

use crate::kernel::{next_smooth_even, EsKernel};

#[derive(Debug, thiserror::Error)]
pub enum NufftError {
    #[error(transparent)]
    Cuda(#[from] hasty_cuda::Error),
    #[error("CUDA driver error: {0:?}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    Fft(#[from] hasty_fft::FftError),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub nf0: i32,
    pub nf1: i32,
    pub nf2: i32,
    pub nb0: i32,
    pub nb1: i32,
    pub nb2: i32,
}
unsafe impl DeviceRepr for Dims {}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MDims {
    pub n0: i32,
    pub n1: i32,
    pub n2: i32,
    pub fftord: i32,
}
unsafe impl DeviceRepr for MDims {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpreadMethod {
    /// pick by density
    Auto,
    /// global-memory atomics, points in bin-sorted order
    Gm,
    /// shared-memory tile per (bin, chunk of points) subproblem
    Sm,
}

#[derive(Clone, Debug)]
pub struct NufftOpts {
    pub tol: f64,
    pub sigma: f64,
    /// overrides the width derived from `tol`
    pub width: Option<usize>,
    pub spread: SpreadMethod,
    /// number of transforms (coils) executed per call
    pub ntransf: usize,
    /// cap on the fine-grid scratch (bytes); transforms are processed in batches that fit
    pub max_grid_bytes: usize,
    /// mode arrays in FFT order (index = k mod N) instead of centred order
    pub fft_order: bool,
    /// extra footprint (cells) beyond the kernel width of custom spreaders (bin size selection)
    pub footprint_pad: usize,
}

impl Default for NufftOpts {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            sigma: 2.0,
            width: None,
            spread: SpreadMethod::Auto,
            ntransf: 1,
            max_grid_bytes: 3 << 30,
            fft_order: false,
            footprint_pad: 0,
        }
    }
}

pub(crate) struct Funcs {
    pub prep: CudaFunction,
    pub scan_local: CudaFunction,
    pub scan_sums: CudaFunction,
    pub scan_add: CudaFunction,
    pub scatter: CudaFunction,
    pub count_sub: CudaFunction,
    pub build_sub: CudaFunction,
    pub spread_gm: CudaFunction,
    pub spread_sm: CudaFunction,
    pub interp_gm: CudaFunction,
    pub pad: CudaFunction,
    pub crop: CudaFunction,
}

/// Device buffers describing the bin-sorted point set.
pub struct SortedPoints<R: Real> {
    pub m: usize,
    pub sc: CudaSlice<R>,
    pub perm: CudaSlice<u32>,
    pub hist: CudaSlice<u32>,
    pub offsets: CudaSlice<u32>,
    pub sub_bin: CudaSlice<u32>,
    pub sub_start: CudaSlice<u32>,
    pub sub_cnt: CudaSlice<u32>,
    pub nsub: usize,
}

pub struct NufftPlan<R: Real> {
    pub dev: Device,
    pub ndim: usize,
    /// modes per dim, index 0 = fastest (x)
    pub modes: [usize; 3],
    pub nf: [usize; 3],
    pub ker: EsKernel,
    pub opts: NufftOpts,
    pub(crate) f: Funcs,
    pub(crate) module: Arc<CudaModule>,
    pub(crate) invphi: [CudaSlice<R>; 3],
    pub(crate) bs: usize,
    pub(crate) maxsub: usize,
    pub(crate) grid: CudaSlice<R::Complex>,
    pub(crate) batch: usize,
    pub(crate) fft_fwd: FftPlan<R>,
    pub(crate) fft_inv: FftPlan<R>,
    pub(crate) pts: Option<SortedPoints<R>>,
    pub(crate) use_sm: bool,
    pub(crate) smem_tile: usize,
}

fn choose_bs(ndim: usize, w: usize, cx_bytes: usize, limit: usize) -> usize {
    let cands: &[usize] = match ndim {
        1 => &[256, 128, 64],
        2 => &[32, 16, 8, 4],
        _ => &[8, 4, 2],
    };
    for &b in cands {
        let t = (b + w).pow(ndim as u32) * cx_bytes;
        if t <= limit {
            return b;
        }
    }
    *cands.last().unwrap()
}

impl<R: Real> NufftPlan<R> {
    /// `modes`: number of Fourier modes per dimension, **fastest first** (`[Nx, Ny, Nz]`).
    pub fn new(dev: &Device, modes: &[usize], opts: NufftOpts) -> Result<Self, NufftError> {
        let ndim = modes.len();
        if !(1..=3).contains(&ndim) || modes.iter().any(|&n| n < 2) {
            return Err(NufftError::Invalid(
                "1..=3 dims with >= 2 modes each".into(),
            ));
        }
        let ker = match opts.width {
            Some(w) => EsKernel::with_width(w, opts.sigma),
            None => EsKernel::from_tol(opts.tol, opts.sigma),
        };
        let mut m3 = [1usize; 3];
        let mut nf = [1usize; 3];
        for a in 0..ndim {
            m3[a] = modes[a];
            nf[a] =
                next_smooth_even(((opts.sigma * modes[a] as f64).ceil() as usize).max(2 * ker.w));
        }
        let cx_bytes = 2 * R::BYTES;
        let max_smem = dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize;
        let bs = choose_bs(ndim, ker.w + opts.footprint_pad, cx_bytes, max_smem.min(96 * 1024));
        let smem_tile = (bs + ker.w).pow(ndim as u32) * cx_bytes; // plain SM spreader tile
        let src = format!(
            "#define REAL {}\n#define REAL_IS_DOUBLE {}\n#define DIM {}\n#define W {}\n#define BETA {:.17e}\n#define BS {}\n{}",
            R::CTYPE,
            R::IS_F64 as i32,
            ndim,
            ker.w,
            ker.beta,
            bs,
            include_str!("nufft.cu")
        );
        let name = format!("nufft_d{ndim}_w{}_{}_bs{bs}", ker.w, R::CTYPE);
        let module = dev.load_module(&name, &src, &CompileSettings::default())?;
        let get = |n: &str| module.load_function(n);
        let f = Funcs {
            prep: get("prep_points")?,
            scan_local: get("scan_local")?,
            scan_sums: get("scan_sums")?,
            scan_add: get("scan_add")?,
            scatter: get("scatter_perm")?,
            count_sub: get("count_subprobs")?,
            build_sub: get("build_subprobs")?,
            spread_gm: get("spread_gm")?,
            spread_sm: get("spread_sm")?,
            interp_gm: get("interp_gm")?,
            pad: get("pad_deconv")?,
            crop: get("crop_deconv")?,
        };
        if smem_tile > 48 * 1024 {
            f.spread_sm.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_tile as i32)?;
        }
        let stream = dev.default_stream();
        let mk_inv = |a: usize| -> Result<CudaSlice<R>, NufftError> {
            let v: Vec<f64> = if a < ndim {
                ker.deconv_factors(nf[a], m3[a])
            } else {
                vec![1.0]
            };
            let host: Vec<R> = v.iter().map(|&x| R::from_f64(x)).collect();
            Ok(stream.clone_htod(&host)?)
        };
        let invphi = [mk_inv(0)?, mk_inv(1)?, mk_inv(2)?];
        let grid_elems: usize = nf.iter().product();
        let batch = (opts.max_grid_bytes / (grid_elems * cx_bytes)).clamp(1, opts.ntransf.max(1));
        let grid = stream.alloc_zeros::<R::Complex>(batch * grid_elems)?;
        let fdims: Vec<usize> = (0..ndim).rev().map(|a| nf[a]).collect();
        let fft_fwd = FftPlan::<R>::new(dev, &fdims, Direction::Forward, Norm::None)?;
        let fft_inv = FftPlan::<R>::new(dev, &fdims, Direction::Inverse, Norm::None)?;
        Ok(Self {
            dev: dev.clone(),
            ndim,
            modes: m3,
            nf,
            ker,
            opts,
            f,
            module,
            invphi,
            bs,
            maxsub: 1024,
            grid,
            batch,
            fft_fwd,
            fft_inv,
            pts: None,
            use_sm: false,
            smem_tile,
        })
    }

    pub fn bs(&self) -> usize {
        self.bs
    }

    pub fn nmodes(&self) -> usize {
        self.modes.iter().product()
    }
    pub fn grid_elems(&self) -> usize {
        self.nf.iter().product()
    }
    pub fn points(&self) -> Option<&SortedPoints<R>> {
        self.pts.as_ref()
    }

    pub(crate) fn dims(&self) -> Dims {
        let nb = |a: usize| (self.nf[a].div_ceil(self.bs)) as i32;
        Dims {
            nf0: self.nf[0] as i32,
            nf1: self.nf[1] as i32,
            nf2: self.nf[2] as i32,
            nb0: nb(0),
            nb1: if self.ndim >= 2 { nb(1) } else { 1 },
            nb2: if self.ndim >= 3 { nb(2) } else { 1 },
        }
    }
    fn nbins(&self) -> usize {
        let d = self.dims();
        (d.nb0 * d.nb1 * d.nb2) as usize
    }

    /// exclusive prefix sum on the device
    fn scan(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<u32>,
        n: usize,
    ) -> Result<CudaSlice<u32>, NufftError> {
        let mut out = stream.alloc_zeros::<u32>(n.max(1))?;
        let nblocks = n.div_ceil(1024).max(1);
        let mut sums = stream.alloc_zeros::<u32>(nblocks)?;
        let n_i = n as i32;
        unsafe {
            stream
                .launch_builder(&self.f.scan_local)
                .arg(input)
                .arg(&mut out)
                .arg(&mut sums)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (nblocks as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            let nb_i = nblocks as i32;
            stream
                .launch_builder(&self.f.scan_sums)
                .arg(&mut sums)
                .arg(&nb_i)
                .launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                })?;
            stream
                .launch_builder(&self.f.scan_add)
                .arg(&mut out)
                .arg(&sums)
                .arg(&n_i)
                .launch(LaunchConfig {
                    grid_dim: (nblocks as u32, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(out)
    }

    /// Fold the points onto the fine grid and bin-sort them. `coords[a]` = coordinate a (a = 0 is x, the fastest dim) in radians per grid unit,
    /// any real value (periodic with period 2*pi).
    pub fn set_points(
        &mut self,
        stream: &Arc<CudaStream>,
        coords: &[&CudaSlice<R>],
        m: usize,
    ) -> Result<(), NufftError> {
        if coords.len() != self.ndim {
            return Err(NufftError::Invalid(format!(
                "expected {} coordinate arrays",
                self.ndim
            )));
        }
        let nbins = self.nbins();
        let mut sc = stream.alloc_zeros::<R>(self.ndim * m)?;
        let mut binid = stream.alloc_zeros::<u32>(m)?;
        let mut hist = stream.alloc_zeros::<u32>(nbins)?;
        let d = self.dims();
        let m_i = m as i32;
        let null = 0u64;
        unsafe {
            let mut b = stream.launch_builder(&self.f.prep);
            for a in 0..3 {
                if a < self.ndim {
                    b.arg(coords[a]);
                } else {
                    b.arg(&null);
                }
            }
            b.arg(&m_i)
                .arg(&mut sc)
                .arg(&mut binid)
                .arg(&mut hist)
                .arg(&d);
            b.launch(LaunchConfig::for_num_elems(m as u32))?;
        }
        let offsets = self.scan(stream, &hist, nbins)?;
        let mut counter = stream.alloc_zeros::<u32>(nbins)?;
        let mut perm = stream.alloc_zeros::<u32>(m)?;
        unsafe {
            stream
                .launch_builder(&self.f.scatter)
                .arg(&binid)
                .arg(&offsets)
                .arg(&mut counter)
                .arg(&mut perm)
                .arg(&m_i)
                .launch(LaunchConfig::for_num_elems(m as u32))?;
        }
        // subproblems for the shared-memory spreader
        let nb_i = nbins as i32;
        let maxsub = self.maxsub as i32;
        let mut nsub_arr = stream.alloc_zeros::<u32>(nbins)?;
        unsafe {
            stream
                .launch_builder(&self.f.count_sub)
                .arg(&hist)
                .arg(&mut nsub_arr)
                .arg(&nb_i)
                .arg(&maxsub)
                .launch(LaunchConfig::for_num_elems(nbins as u32))?;
        }
        let sub_off = self.scan(stream, &nsub_arr, nbins)?;
        let tail_off = stream.clone_dtoh(&sub_off.slice(nbins - 1..nbins))?[0] as usize;
        let tail_cnt = stream.clone_dtoh(&nsub_arr.slice(nbins - 1..nbins))?[0] as usize;
        let nsub = tail_off + tail_cnt;
        let mut sub_bin = stream.alloc_zeros::<u32>(nsub.max(1))?;
        let mut sub_start = stream.alloc_zeros::<u32>(nsub.max(1))?;
        let mut sub_cnt = stream.alloc_zeros::<u32>(nsub.max(1))?;
        unsafe {
            stream
                .launch_builder(&self.f.build_sub)
                .arg(&hist)
                .arg(&offsets)
                .arg(&sub_off)
                .arg(&mut sub_bin)
                .arg(&mut sub_start)
                .arg(&mut sub_cnt)
                .arg(&nb_i)
                .arg(&maxsub)
                .launch(LaunchConfig::for_num_elems(nbins as u32))?;
        }
        // method choice: average multiplicity of the stencil footprint per fine-grid cell
        let cells = self.grid_elems() as f64;
        let mult = m as f64 * (self.ker.w as f64).powi(self.ndim as i32) / cells;
        self.use_sm = match self.opts.spread {
            SpreadMethod::Gm => false,
            SpreadMethod::Sm => true,
            SpreadMethod::Auto => mult >= 2.0,
        };
        self.pts = Some(SortedPoints {
            m,
            sc,
            perm,
            hist,
            offsets,
            sub_bin,
            sub_start,
            sub_cnt,
            nsub,
        });
        Ok(())
    }

    fn pts_or_err(&self) -> Result<&SortedPoints<R>, NufftError> {
        self.pts
            .as_ref()
            .ok_or_else(|| NufftError::Invalid("set_points not called".into()))
    }

    /// Spread `ntr` weight vectors (`c`: `ntr * M`, transform-major) into `self.grid` (must be zeroed by the caller).
    pub(crate) fn spread_into_grid(
        &self,
        stream: &Arc<CudaStream>,
        c_ptr: u64,
        ntr: usize,
    ) -> Result<(), NufftError> {
        let p = self.pts_or_err()?;
        let d = self.dims();
        let (m_i, ntr_i) = (p.m as i32, ntr as i32);
        let gs = self.grid_elems() as u64;
        let (gp, _gg) = self.grid.device_ptr(stream);
        unsafe {
            if self.use_sm {
                let cfg = LaunchConfig {
                    grid_dim: (p.nsub as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: self.smem_tile as u32,
                };
                stream
                    .launch_builder(&self.f.spread_sm)
                    .arg(&p.sc)
                    .arg(&p.perm)
                    .arg(&c_ptr)
                    .arg(&gp)
                    .arg(&p.sub_bin)
                    .arg(&p.sub_start)
                    .arg(&p.sub_cnt)
                    .arg(&m_i)
                    .arg(&ntr_i)
                    .arg(&gs)
                    .arg(&d)
                    .launch(cfg)?;
            } else {
                stream
                    .launch_builder(&self.f.spread_gm)
                    .arg(&p.sc)
                    .arg(&p.perm)
                    .arg(&c_ptr)
                    .arg(&gp)
                    .arg(&m_i)
                    .arg(&ntr_i)
                    .arg(&gs)
                    .arg(&d)
                    .launch(LaunchConfig {
                        grid_dim: ((p.m as u32).div_ceil(128), 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        Ok(())
    }

    pub(crate) fn interp_from_grid(
        &self,
        stream: &Arc<CudaStream>,
        c_ptr: u64,
        ntr: usize,
    ) -> Result<(), NufftError> {
        let p = self.pts_or_err()?;
        let d = self.dims();
        let (m_i, ntr_i) = (p.m as i32, ntr as i32);
        let gs = self.grid_elems() as u64;
        let (gp, _gg) = self.grid.device_ptr(stream);
        unsafe {
            stream
                .launch_builder(&self.f.interp_gm)
                .arg(&p.sc)
                .arg(&p.perm)
                .arg(&gp)
                .arg(&c_ptr)
                .arg(&m_i)
                .arg(&ntr_i)
                .arg(&gs)
                .arg(&d)
                .launch(LaunchConfig {
                    grid_dim: ((p.m as u32).div_ceil(128), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    fn fft_batch(
        &mut self,
        stream: &Arc<CudaStream>,
        ntr: usize,
        sign: i32,
    ) -> Result<(), NufftError> {
        let ge = self.grid_elems();
        for t in 0..ntr {
            let mut view = self.grid.slice_mut(t * ge..(t + 1) * ge);
            if sign > 0 {
                self.fft_inv.execute_inplace(stream, &mut view)?;
            } else {
                self.fft_fwd.execute_inplace(stream, &mut view)?;
            }
        }
        Ok(())
    }

    fn mode_launch(
        &self,
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        a: u64,
        b: u64,
        ntr: usize,
    ) -> Result<(), NufftError> {
        let md = MDims {
            n0: self.modes[0] as i32,
            n1: self.modes[1] as i32,
            n2: self.modes[2] as i32,
            fftord: self.opts.fft_order as i32,
        };
        let d = self.dims();
        let ntr_i = ntr as i32;
        let gs = self.grid_elems() as u64;
        unsafe {
            stream
                .launch_builder(func)
                .arg(&a)
                .arg(&b)
                .arg(&self.invphi[0])
                .arg(&self.invphi[1])
                .arg(&self.invphi[2])
                .arg(&ntr_i)
                .arg(&gs)
                .arg(&md)
                .arg(&d)
                .launch(LaunchConfig::for_num_elems(self.nmodes() as u32))?;
        }
        Ok(())
    }

    /// Type 1 with a custom spreading step (used by the chirp-aware plan): `spread(plan, stream, c_ptr, ntr)` must accumulate into `plan.grid`.
    pub(crate) fn type1_with(
        &mut self,
        stream: &Arc<CudaStream>,
        c: &CudaSlice<R::Complex>,
        f: &mut CudaSlice<R::Complex>,
        sign: i32,
        spread: &dyn Fn(&NufftPlan<R>, &Arc<CudaStream>, u64, usize) -> Result<(), NufftError>,
    ) -> Result<(), NufftError> {
        let m = self.pts_or_err()?.m;
        let nm = self.nmodes();
        let nt = self.opts.ntransf;
        if c.len() < nt * m || f.len() < nt * nm {
            return Err(NufftError::Invalid("c/f too small for ntransf".into()));
        }
        let es = std::mem::size_of::<R::Complex>() as u64;
        let (cp, _g1) = c.device_ptr(stream);
        let (fp, _g2) = f.device_ptr_mut(stream);
        let mut t0 = 0;
        while t0 < nt {
            let ntr = self.batch.min(nt - t0);
            stream.memset_zeros(&mut self.grid)?;
            spread(self, stream, cp + t0 as u64 * m as u64 * es, ntr)?;
            self.fft_batch(stream, ntr, sign)?;
            let (gp, _gg) = self.grid.device_ptr(stream);
            self.mode_launch(
                stream,
                &self.f.crop,
                gp,
                fp + t0 as u64 * nm as u64 * es,
                ntr,
            )?;
            t0 += ntr;
        }
        Ok(())
    }

    pub(crate) fn type2_with(
        &mut self,
        stream: &Arc<CudaStream>,
        f: &CudaSlice<R::Complex>,
        c: &mut CudaSlice<R::Complex>,
        sign: i32,
        interp: &dyn Fn(&NufftPlan<R>, &Arc<CudaStream>, u64, usize) -> Result<(), NufftError>,
    ) -> Result<(), NufftError> {
        let m = self.pts_or_err()?.m;
        let nm = self.nmodes();
        let nt = self.opts.ntransf;
        if c.len() < nt * m || f.len() < nt * nm {
            return Err(NufftError::Invalid("c/f too small for ntransf".into()));
        }
        let es = std::mem::size_of::<R::Complex>() as u64;
        let (fp, _g1) = f.device_ptr(stream);
        let (cp, _g2) = c.device_ptr_mut(stream);
        let mut t0 = 0;
        while t0 < nt {
            let ntr = self.batch.min(nt - t0);
            stream.memset_zeros(&mut self.grid)?;
            {
                let (gp, _gg) = self.grid.device_ptr(stream);
                self.mode_launch(
                    stream,
                    &self.f.pad,
                    fp + t0 as u64 * nm as u64 * es,
                    gp,
                    ntr,
                )?;
            }
            self.fft_batch(stream, ntr, sign)?;
            interp(self, stream, cp + t0 as u64 * m as u64 * es, ntr)?;
            t0 += ntr;
        }
        Ok(())
    }

    /// Type 1: `f[k] = sum_j c_j exp(sign * i * k . x_j)`. `c`: `ntransf * M`, `f`: `ntransf * nmodes` (modes centred, x fastest).
    pub fn type1(
        &mut self,
        stream: &Arc<CudaStream>,
        c: &CudaSlice<R::Complex>,
        f: &mut CudaSlice<R::Complex>,
        sign: i32,
    ) -> Result<(), NufftError> {
        let m = self.pts_or_err()?.m;
        let nm = self.nmodes();
        let nt = self.opts.ntransf;
        if c.len() < nt * m || f.len() < nt * nm {
            return Err(NufftError::Invalid("c/f too small for ntransf".into()));
        }
        let es = std::mem::size_of::<R::Complex>() as u64;
        let (cp, _g1) = c.device_ptr(stream);
        let (fp, _g2) = f.device_ptr_mut(stream);
        let mut t0 = 0;
        while t0 < nt {
            let ntr = self.batch.min(nt - t0);
            stream.memset_zeros(&mut self.grid)?;
            self.spread_into_grid(stream, cp + t0 as u64 * m as u64 * es, ntr)?;
            self.fft_batch(stream, ntr, sign)?;
            let (gp, _gg) = self.grid.device_ptr(stream);
            self.mode_launch(
                stream,
                &self.f.crop,
                gp,
                fp + t0 as u64 * nm as u64 * es,
                ntr,
            )?;
            t0 += ntr;
        }
        Ok(())
    }

    /// Type 2: `c_j = sum_k f[k] exp(sign * i * k . x_j)`.
    pub fn type2(
        &mut self,
        stream: &Arc<CudaStream>,
        f: &CudaSlice<R::Complex>,
        c: &mut CudaSlice<R::Complex>,
        sign: i32,
    ) -> Result<(), NufftError> {
        let m = self.pts_or_err()?.m;
        let nm = self.nmodes();
        let nt = self.opts.ntransf;
        if c.len() < nt * m || f.len() < nt * nm {
            return Err(NufftError::Invalid("c/f too small for ntransf".into()));
        }
        let es = std::mem::size_of::<R::Complex>() as u64;
        let (fp, _g1) = f.device_ptr(stream);
        let (cp, _g2) = c.device_ptr_mut(stream);
        let mut t0 = 0;
        while t0 < nt {
            let ntr = self.batch.min(nt - t0);
            stream.memset_zeros(&mut self.grid)?;
            {
                let (gp, _gg) = self.grid.device_ptr(stream);
                self.mode_launch(
                    stream,
                    &self.f.pad,
                    fp + t0 as u64 * nm as u64 * es,
                    gp,
                    ntr,
                )?;
            }
            self.fft_batch(stream, ntr, sign)?;
            self.interp_from_grid(stream, cp + t0 as u64 * m as u64 * es, ntr)?;
            t0 += ntr;
        }
        Ok(())
    }
}
