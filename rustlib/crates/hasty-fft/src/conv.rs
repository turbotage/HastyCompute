//! Fused, pruned zero-padded convolution:  `out (+)= post . crop( IFFT( K . FFT( pad( pre . x ) ) ) ) / prod(dims_pad)`.
//!
//! This is the generic building block behind Toeplitz multiplication (x of shape `dims`, padded to `dims_pad` = 2*dims, `K` = FFT of the
//! circulant-embedded kernel in natural FFT order), but it knows nothing about MRI: `pre`/`post` are lists of diagonal arrays (multiply or
//! conjugate-multiply) and `accumulate` selects `out += ...`.
//!
//! Passes for rank 3 (slowest axis first in `dims`; X = last/contiguous):
//!   1. FFT X   : lines (z<Zn, y<Yn), reads Xn inputs from `x` (applies `pre`), writes X2 into scratch
//!   2. FFT Y   : lines (z<Zn, x<X2), reads Yn, writes Y2, in place in scratch
//!   3. conv Z  : lines (y<Y2, x<X2): FFT Z (reads Zn) -> times K -> IFFT Z (writes Zn), in place in scratch
//!   4. IFFT Y  : lines (z<Zn, x<X2), reads Y2, writes Yn
//!   5. IFFT X  : lines (z<Zn, y<Yn), reads X2 from scratch, writes Xn to `out` (applies `scale`, `post`, `accumulate`)
//! Zero rows are never read nor written; the padded scratch only has to be allocated, never cleared.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use hasty_cuda::{Device, Real};

use crate::codegen::{DiagOp, KernelFormat};
use crate::plan::{AxisPlan, AxisSpec, Direction, FftError, LaunchPtrs, LineAddr};

pub struct ConvSpec {
    /// size of the (unpadded) input/output array, slowest axis first, 1..=3 axes
    pub dims: Vec<usize>,
    /// padded size per axis (>= dims; any length with prime factors in {2,3,5,7,11,13})
    pub dims_pad: Vec<usize>,
    pub pre: Vec<DiagOp>,
    pub post: Vec<DiagOp>,
    pub accumulate: bool,
    pub kernel_format: KernelFormat,
}

pub struct ConvPlan<R: Real> {
    pub spec_dims: Vec<usize>,
    pub dims_pad: Vec<usize>,
    passes: Vec<AxisPlan<R>>,
    npre: usize,
    npost: usize,
}

impl<R: Real> ConvPlan<R> {
    pub fn new(dev: &Device, spec: &ConvSpec) -> Result<Self, FftError> {
        let d = spec.dims.len();
        if d == 0 || d > 3 || spec.dims_pad.len() != d {
            return Err(FftError::Invalid(
                "ConvPlan supports rank 1..=3 with matching dims/dims_pad".into(),
            ));
        }
        for i in 0..d {
            if spec.dims_pad[i] < spec.dims[i] {
                return Err(FftError::Invalid("dims_pad must be >= dims".into()));
            }
        }
        let total_pad: usize = spec.dims_pad.iter().product();
        let scale = 1.0 / total_pad as f64;
        // row-major strides of the padded scratch and of the dense input/output
        let mut sp = vec![1usize; d];
        let mut sd = vec![1usize; d];
        for i in (0..d - 1).rev() {
            sp[i] = sp[i + 1] * spec.dims_pad[i + 1];
            sd[i] = sd[i + 1] * spec.dims[i + 1];
        }
        let (nn, np) = (&spec.dims, &spec.dims_pad);
        let mk = |n: usize,
                  n_in: usize,
                  n_out: usize,
                  dir,
                  sc,
                  conv,
                  pre: &[DiagOp],
                  post: &[DiagOp],
                  acc,
                  contiguous,
                  addr| AxisSpec {
            n,
            n_in,
            n_out,
            dir,
            scale: sc,
            conv,
            pre: pre.to_vec(),
            post: post.to_vec(),
            accumulate: acc,
            contiguous,
            addr,
            kfmt: if conv {
                spec.kernel_format
            } else {
                KernelFormat::Complex
            },
            poly_pre: false,
            poly_post: false,
            vox: [1, 1, 1],
        };
        let mut passes = Vec::new();
        use Direction::*;
        match d {
            1 => {
                // fused conv directly from x to out
                let addr = LineAddr {
                    la_cnt: 1,
                    lb_cnt: 1,
                    in_la: 0,
                    in_lb: 0,
                    in_sn: 1,
                    out_la: 0,
                    out_lb: 0,
                    out_sn: 1,
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[0],
                        nn[0],
                        nn[0],
                        Forward,
                        scale,
                        true,
                        &spec.pre,
                        &spec.post,
                        spec.accumulate,
                        true,
                        addr,
                    ),
                )?);
            }
            2 => {
                let (y, x) = (0, 1);
                // 1. FFT X: lines y<Yn  x -> scratch
                let a1 = LineAddr {
                    la_cnt: nn[y],
                    lb_cnt: 1,
                    in_la: sd[y],
                    in_lb: 0,
                    in_sn: 1,
                    out_la: sp[y],
                    out_lb: 0,
                    out_sn: 1,
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[x],
                        nn[x],
                        np[x],
                        Forward,
                        1.0,
                        false,
                        &spec.pre,
                        &[],
                        false,
                        true,
                        a1,
                    ),
                )?);
                // 2. conv along Y: lines x<X2 (tile dim), in place in scratch
                let a2 = LineAddr {
                    la_cnt: np[x],
                    lb_cnt: 1,
                    in_la: 1,
                    in_lb: 0,
                    in_sn: sp[y],
                    out_la: 1,
                    out_lb: 0,
                    out_sn: sp[y],
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[y],
                        nn[y],
                        nn[y],
                        Forward,
                        1.0,
                        true,
                        &[],
                        &[],
                        false,
                        false,
                        a2,
                    ),
                )?);
                // 3. IFFT X: lines y<Yn  scratch -> out
                let a3 = LineAddr {
                    la_cnt: nn[y],
                    lb_cnt: 1,
                    in_la: sp[y],
                    in_lb: 0,
                    in_sn: 1,
                    out_la: sd[y],
                    out_lb: 0,
                    out_sn: 1,
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[x],
                        np[x],
                        nn[x],
                        Inverse,
                        scale,
                        false,
                        &[],
                        &spec.post,
                        spec.accumulate,
                        true,
                        a3,
                    ),
                )?);
            }
            _ => {
                let (z, y, x) = (0, 1, 2);
                // 1. FFT X: lines (y<Yn fast, z<Zn)
                let a1 = LineAddr {
                    la_cnt: nn[y],
                    lb_cnt: nn[z],
                    in_la: sd[y],
                    in_lb: sd[z],
                    in_sn: 1,
                    out_la: sp[y],
                    out_lb: sp[z],
                    out_sn: 1,
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[x],
                        nn[x],
                        np[x],
                        Forward,
                        1.0,
                        false,
                        &spec.pre,
                        &[],
                        false,
                        true,
                        a1,
                    ),
                )?);
                // 2. FFT Y: lines (x<X2 tile, z<Zn)
                let a2 = LineAddr {
                    la_cnt: np[x],
                    lb_cnt: nn[z],
                    in_la: 1,
                    in_lb: sp[z],
                    in_sn: sp[y],
                    out_la: 1,
                    out_lb: sp[z],
                    out_sn: sp[y],
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[y],
                        nn[y],
                        np[y],
                        Forward,
                        1.0,
                        false,
                        &[],
                        &[],
                        false,
                        false,
                        a2,
                    ),
                )?);
                // 3. conv Z: lines (x<X2 tile, y<Y2)
                let a3 = LineAddr {
                    la_cnt: np[x],
                    lb_cnt: np[y],
                    in_la: 1,
                    in_lb: sp[y],
                    in_sn: sp[z],
                    out_la: 1,
                    out_lb: sp[y],
                    out_sn: sp[z],
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[z],
                        nn[z],
                        nn[z],
                        Forward,
                        1.0,
                        true,
                        &[],
                        &[],
                        false,
                        false,
                        a3,
                    ),
                )?);
                // 4. IFFT Y: lines (x<X2 tile, z<Zn), reads Y2 writes Yn
                let a4 = LineAddr {
                    la_cnt: np[x],
                    lb_cnt: nn[z],
                    in_la: 1,
                    in_lb: sp[z],
                    in_sn: sp[y],
                    out_la: 1,
                    out_lb: sp[z],
                    out_sn: sp[y],
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[y],
                        np[y],
                        nn[y],
                        Inverse,
                        1.0,
                        false,
                        &[],
                        &[],
                        false,
                        false,
                        a4,
                    ),
                )?);
                // 5. IFFT X: lines (y<Yn, z<Zn): scratch -> out
                let a5 = LineAddr {
                    la_cnt: nn[y],
                    lb_cnt: nn[z],
                    in_la: sp[y],
                    in_lb: sp[z],
                    in_sn: 1,
                    out_la: sd[y],
                    out_lb: sd[z],
                    out_sn: 1,
                };
                passes.push(AxisPlan::from_spec(
                    dev,
                    &mk(
                        np[x],
                        np[x],
                        nn[x],
                        Inverse,
                        scale,
                        false,
                        &[],
                        &spec.post,
                        spec.accumulate,
                        true,
                        a5,
                    ),
                )?);
            }
        }
        Ok(Self {
            spec_dims: spec.dims.clone(),
            dims_pad: spec.dims_pad.clone(),
            passes,
            npre: spec.pre.len(),
            npost: spec.post.len(),
        })
    }

    pub fn len_in(&self) -> usize {
        self.spec_dims.iter().product()
    }
    pub fn len_scratch(&self) -> usize {
        self.dims_pad.iter().product()
    }

    /// `x`: dense input (`dims`), `kernel`: `dims_pad` array (k-space multiplier), `scratch`: `dims_pad` elements (contents irrelevant),
    /// `out`: dense output (`dims`). `pre`/`post`: dense `dims`-shaped diagonals, in the order given at plan time.
    #[allow(clippy::too_many_arguments)]
    pub fn execute<KT, KS: DevicePtr<KT>>(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<R::Complex>,
        kernel: &KS,
        kscale: f64,
        scratch: &mut CudaSlice<R::Complex>,
        out: &mut CudaSlice<R::Complex>,
        pre: &[&CudaSlice<R::Complex>],
        post: &[&CudaSlice<R::Complex>],
    ) -> Result<(), FftError> {
        if pre.len() != self.npre || post.len() != self.npost {
            return Err(FftError::Invalid(format!(
                "plan has {} pre / {} post operands, got {} / {}",
                self.npre,
                self.npost,
                pre.len(),
                post.len()
            )));
        }
        let check = |have: usize, need: usize| {
            if have < need {
                Err(FftError::BufferSize { need, have })
            } else {
                Ok(())
            }
        };
        check(x.len(), self.len_in())?;
        check(out.len(), self.len_in())?;
        check(kernel.len(), self.len_scratch())?;
        if self.spec_dims.len() > 1 {
            check(scratch.len(), self.len_scratch())?;
        }
        for p in pre.iter().chain(post.iter()) {
            check(p.len(), self.len_in())?;
        }
        let (xp, _g0) = x.device_ptr(stream);
        let (kp, _g1) = kernel.device_ptr(stream);
        let (sp, _g2) = scratch.device_ptr_mut(stream);
        let (op, _g3) = out.device_ptr_mut(stream);
        let mut pre_p = [0u64; 4];
        let mut post_p = [0u64; 4];
        let mut guards = Vec::new();
        for (i, a) in pre.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            pre_p[i] = p;
            guards.push(g);
        }
        for (i, a) in post.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            post_p[i] = p;
            guards.push(g);
        }
        self.run_passes(stream, xp, kp, kscale, sp, op, pre_p, post_p, None)
    }

    /// Like `execute` but times every pass with events (synchronises after each pass; for profiling only).
    pub fn execute_profiled<KT, KS: DevicePtr<KT>>(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<R::Complex>,
        kernel: &KS,
        kscale: f64,
        scratch: &mut CudaSlice<R::Complex>,
        out: &mut CudaSlice<R::Complex>,
        pre: &[&CudaSlice<R::Complex>],
        post: &[&CudaSlice<R::Complex>],
    ) -> Result<Vec<f32>, FftError> {
        let (xp, _g0) = x.device_ptr(stream);
        let (kp, _g1) = kernel.device_ptr(stream);
        let (sp, _g2) = scratch.device_ptr_mut(stream);
        let (op, _g3) = out.device_ptr_mut(stream);
        let mut pre_p = [0u64; 4];
        let mut post_p = [0u64; 4];
        let mut guards = Vec::new();
        for (i, a) in pre.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            pre_p[i] = p;
            guards.push(g);
        }
        for (i, a) in post.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            post_p[i] = p;
            guards.push(g);
        }
        let mut out_t = Vec::new();
        self.run_passes(
            stream,
            xp,
            kp,
            kscale,
            sp,
            op,
            pre_p,
            post_p,
            Some(&mut out_t),
        )?;
        Ok(out_t)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_passes(
        &self,
        stream: &Arc<CudaStream>,
        xp: u64,
        kp: u64,
        kscale: f64,
        sp: u64,
        op: u64,
        pre_p: [u64; 4],
        post_p: [u64; 4],
        mut times: Option<&mut Vec<f32>>,
    ) -> Result<(), FftError> {
        let np = self.passes.len();
        let ctx = stream.context().clone();
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        for (i, pl) in self.passes.iter().enumerate() {
            let mut lp = LaunchPtrs {
                kern: kp,
                kscale,
                ..Default::default()
            };
            let (first, last) = (i == 0, i == np - 1);
            lp.inp = if first { xp } else { sp };
            lp.out = if last { op } else { sp };
            if np == 1 {
                lp.inp = xp;
                lp.out = op;
            }
            if first {
                lp.pre = pre_p;
            }
            if last {
                lp.post = post_p;
            }
            if let Some(t) = times.as_deref_mut() {
                let (a, b) = (ctx.new_event(flags)?, ctx.new_event(flags)?);
                a.record(stream)?;
                pl.launch(stream, &lp)?;
                b.record(stream)?;
                b.synchronize()?;
                t.push(a.elapsed_ms(&b)?);
            } else {
                pl.launch(stream, &lp)?;
            }
        }
        Ok(())
    }
}
