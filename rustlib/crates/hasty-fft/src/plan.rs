use std::sync::Arc;

use cudarc::driver::{
    sys::CUfunction_attribute_enum, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use hasty_cuda::{CompileSettings, Device, Real};

use crate::codegen::{generate, AxisCfg, DiagOp, KernelFormat};
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

#[derive(Clone, Debug)]
pub struct AxisSpec {
    pub n: usize,
    pub n_in: usize,
    pub n_out: usize,
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
}

/// One FFT pass over lines of a row-major array. See [`AxisSpec`] and `codegen.rs`.
pub struct AxisPlan<R: Real> {
    func: CudaFunction,
    tw_m: CudaSlice<R::Complex>,
    tw_p: CudaSlice<R::Complex>,
    pub cfg: AxisCfg,
    pub addr: LineAddr,
    grid: (u32, u32, u32),
    smem: u32,
    /// elements the plain (outer, n, inner) layout would occupy; used for buffer-size checks by the simple API
    simple_len: usize,
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

impl<R: Real> AxisPlan<R> {
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
        let n = spec.n;
        if spec.n_in > n || spec.n_out > n || spec.n_in == 0 || spec.n_out == 0 {
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
        let radices = factorize(n).ok_or(FftError::UnsupportedLength(n))?;
        let max_smem = dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)? as usize;
        let tile_cnt = spec.addr.la_cnt;
        let (lpb, tpl) = choose_shape(n, &radices, spec.contiguous, tile_cnt, R::BYTES, max_smem);
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
            conv: spec.conv,
            pre: spec.pre.clone(),
            post: spec.post.clone(),
            accumulate: spec.accumulate,
            kfmt: spec.kfmt,
            poly_pre: spec.poly_pre,
            poly_post: spec.poly_post,
            vox: spec.vox,
        };
        let smem = cfg.smem_bytes(R::BYTES);
        if smem > max_smem {
            return Err(FftError::SharedMemory {
                n,
                need: smem,
                have: max_smem,
            });
        }
        let src = generate(&cfg);
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
        );
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
        let table = |sg: f64| -> Vec<R::Complex> {
            (0..n)
                .map(|m| {
                    let a = sg * 2.0 * std::f64::consts::PI * (m as f64) / (n as f64);
                    R::complex(a.cos(), a.sin())
                })
                .collect()
        };
        let stream = dev.default_stream();
        let tw_m = stream.clone_htod(&table(-1.0))?;
        let tw_p = stream.clone_htod(&table(1.0))?;
        let a = spec.addr;
        let grid = if spec.contiguous {
            ((a.la_cnt * a.lb_cnt).div_ceil(lpb) as u32, 1, 1)
        } else {
            (a.la_cnt.div_ceil(lpb) as u32, a.lb_cnt as u32, 1)
        };
        Ok(Self {
            func,
            tw_m,
            tw_p,
            cfg,
            addr: a,
            grid,
            smem: smem as u32,
            simple_len: 0,
        })
    }

    /// Elements covered by the plain layout (only meaningful for plans built with [`AxisPlan::new`]).
    pub fn len(&self) -> usize {
        self.simple_len
    }

    pub fn launch(&self, stream: &Arc<CudaStream>, p: &LaunchPtrs) -> Result<(), FftError> {
        let (twm, _g1) = self.tw_m.device_ptr(stream);
        let (twp, _g2) = self.tw_p.device_ptr(stream);
        let a = &self.addr;
        let (la_cnt, lb_cnt) = (a.la_cnt as i64, a.lb_cnt as i64);
        let (ila, ilb, isn, ola, olb, osn) = (
            a.in_la as i64,
            a.in_lb as i64,
            a.in_sn as i64,
            a.out_la as i64,
            a.out_lb as i64,
            a.out_sn as i64,
        );
        let cfg = LaunchConfig {
            grid_dim: self.grid,
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
