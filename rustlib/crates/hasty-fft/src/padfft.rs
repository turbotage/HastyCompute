//! Pruned zero-padded 3-D forward / inverse transforms as separate steps (so that a k-space operation other than "multiply by one kernel"
//! can sit in between, see `hasty-toeplitz`). Same pruning rules as [`crate::ConvPlan`]: zero rows are never read or written.
//!
//! forward : x (dims, dense) --pre--> X pass (reads Xn writes X2) -> Y pass -> Z pass            => scratch (pad dims, full spectrum)
//! inverse : scratch -> Z pass (reads Z2 writes Zn) -> Y pass (Y2 -> Yn) -> X pass (X2 -> Xn) --post/scale/accumulate--> out (dims, dense)

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use hasty_cuda::{Device, Real};

use crate::codegen::DiagOp;
use crate::plan::{AxisPlan, AxisSpec, Direction, FftError, LaunchPtrs, LineAddr};

#[derive(Clone, Debug, Default)]
pub struct PadFftOps {
    pub pre: Vec<DiagOp>,
    pub poly_pre: bool,
    pub post: Vec<DiagOp>,
    pub poly_post: bool,
    pub accumulate: bool,
}

/// Table of polynomial-diagonal terms on the device: rows `[coef, px, py, pz]` in the transform precision.
pub struct PolyTable<R: Real> {
    pub buf: CudaSlice<R>,
    pub n: usize,
}

impl<R: Real> PolyTable<R> {
    pub fn new(
        stream: &Arc<CudaStream>,
        terms: &[crate::codegen::PolyTerm],
    ) -> Result<Self, FftError> {
        let mut host: Vec<R> = Vec::with_capacity(4 * terms.len().max(1));
        for t in terms {
            host.push(R::from_f64(t.coef));
            host.extend(t.p.iter().map(|&p| R::from_f64(p as f64)));
        }
        if host.is_empty() {
            host.extend([R::from_f64(0.0); 4]);
        }
        Ok(Self {
            buf: stream.clone_htod(&host).map_err(FftError::Driver)?,
            n: terms.len(),
        })
    }
}

pub struct PadFft3<R: Real> {
    pub dims: [usize; 3],
    pub pad: [usize; 3],
    fwd: [AxisPlan<R>; 3],
    inv: [AxisPlan<R>; 3],
}

impl<R: Real> PadFft3<R> {
    /// `dims`/`pad`: `[Zn, Yn, Xn]` / `[Z2, Y2, X2]` (slowest first, X contiguous).
    pub fn new(
        dev: &Device,
        dims: [usize; 3],
        pad: [usize; 3],
        ops: &PadFftOps,
    ) -> Result<Self, FftError> {
        let (z, y, x) = (0usize, 1usize, 2usize);
        let sp = [pad[y] * pad[x], pad[x], 1usize];
        let sd = [dims[y] * dims[x], dims[x], 1usize];
        let scale = 1.0 / (pad[0] * pad[1] * pad[2]) as f64;
        let vox = [dims[x], dims[y], dims[z]];
        let mk = |n: usize,
                  n_in: usize,
                  n_out: usize,
                  dir,
                  sc,
                  contiguous,
                  addr: LineAddr,
                  pre: &[DiagOp],
                  poly_pre,
                  post: &[DiagOp],
                  poly_post,
                  acc| AxisSpec {
            n,
            n_in,
            n_out,
            dir,
            scale: sc,
            conv: false,
            pre: pre.to_vec(),
            post: post.to_vec(),
            accumulate: acc,
            contiguous,
            addr,
            kfmt: crate::codegen::KernelFormat::Complex,
            poly_pre,
            poly_post,
            vox,
        };
        use Direction::*;
        let f_x = LineAddr {
            la_cnt: dims[y],
            lb_cnt: dims[z],
            in_la: sd[1],
            in_lb: sd[0],
            in_sn: 1,
            out_la: sp[1],
            out_lb: sp[0],
            out_sn: 1,
        };
        let f_y = LineAddr {
            la_cnt: pad[x],
            lb_cnt: dims[z],
            in_la: 1,
            in_lb: sp[0],
            in_sn: sp[1],
            out_la: 1,
            out_lb: sp[0],
            out_sn: sp[1],
        };
        let f_z = LineAddr {
            la_cnt: pad[x],
            lb_cnt: pad[y],
            in_la: 1,
            in_lb: sp[1],
            in_sn: sp[0],
            out_la: 1,
            out_lb: sp[1],
            out_sn: sp[0],
        };
        let fwd = [
            AxisPlan::from_spec(
                dev,
                &mk(
                    pad[x],
                    dims[x],
                    pad[x],
                    Forward,
                    1.0,
                    true,
                    f_x,
                    &ops.pre,
                    ops.poly_pre,
                    &[],
                    false,
                    false,
                ),
            )?,
            AxisPlan::from_spec(
                dev,
                &mk(
                    pad[y],
                    dims[y],
                    pad[y],
                    Forward,
                    1.0,
                    false,
                    f_y,
                    &[],
                    false,
                    &[],
                    false,
                    false,
                ),
            )?,
            AxisPlan::from_spec(
                dev,
                &mk(
                    pad[z],
                    dims[z],
                    pad[z],
                    Forward,
                    1.0,
                    false,
                    f_z,
                    &[],
                    false,
                    &[],
                    false,
                    false,
                ),
            )?,
        ];
        let i_z = f_z;
        let i_y = f_y;
        let i_x = LineAddr {
            la_cnt: dims[y],
            lb_cnt: dims[z],
            in_la: sp[1],
            in_lb: sp[0],
            in_sn: 1,
            out_la: sd[1],
            out_lb: sd[0],
            out_sn: 1,
        };
        let inv = [
            AxisPlan::from_spec(
                dev,
                &mk(
                    pad[z],
                    pad[z],
                    dims[z],
                    Inverse,
                    1.0,
                    false,
                    i_z,
                    &[],
                    false,
                    &[],
                    false,
                    false,
                ),
            )?,
            AxisPlan::from_spec(
                dev,
                &mk(
                    pad[y],
                    pad[y],
                    dims[y],
                    Inverse,
                    1.0,
                    false,
                    i_y,
                    &[],
                    false,
                    &[],
                    false,
                    false,
                ),
            )?,
            AxisPlan::from_spec(
                dev,
                &mk(
                    pad[x],
                    pad[x],
                    dims[x],
                    Inverse,
                    scale,
                    true,
                    i_x,
                    &[],
                    false,
                    &ops.post,
                    ops.poly_post,
                    ops.accumulate,
                ),
            )?,
        ];
        Ok(Self {
            dims,
            pad,
            fwd,
            inv,
        })
    }

    pub fn len_in(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn len_pad(&self) -> usize {
        self.pad.iter().product()
    }

    /// `scratch` receives the full (padded) spectrum of `pre . x`.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<R::Complex>,
        scratch: &mut CudaSlice<R::Complex>,
        pre: &[&CudaSlice<R::Complex>],
        poly: Option<&PolyTable<R>>,
    ) -> Result<(), FftError> {
        let (xp, _g0) = x.device_ptr(stream);
        let (sp, _g1) = scratch.device_ptr_mut(stream);
        let mut guards = Vec::new();
        let mut pre_p = [0u64; 4];
        for (i, a) in pre.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            pre_p[i] = p;
            guards.push(g);
        }
        let (pp, np, _gp) = match poly {
            Some(t) => {
                let (p, g) = t.buf.device_ptr(stream);
                (p, t.n as i32, Some(g))
            }
            None => (0, 0, None),
        };
        self.fwd[0].launch(
            stream,
            &LaunchPtrs {
                inp: xp,
                out: sp,
                pre: pre_p,
                ppre: pp,
                nppre: np,
                ..Default::default()
            },
        )?;
        self.fwd[1].launch(
            stream,
            &LaunchPtrs {
                inp: sp,
                out: sp,
                ..Default::default()
            },
        )?;
        self.fwd[2].launch(
            stream,
            &LaunchPtrs {
                inp: sp,
                out: sp,
                ..Default::default()
            },
        )?;
        Ok(())
    }

    /// forward transform along X and Y only (the Z pass is done by the caller, e.g. [`crate::MixZPlan`])
    pub fn forward_xy(
        &self,
        stream: &Arc<CudaStream>,
        x: &CudaSlice<R::Complex>,
        scratch: &mut CudaSlice<R::Complex>,
        pre: &[&CudaSlice<R::Complex>],
        poly: Option<&PolyTable<R>>,
    ) -> Result<(), FftError> {
        let (xp, _g0) = x.device_ptr(stream);
        let (sp, _g1) = scratch.device_ptr_mut(stream);
        let mut guards = Vec::new();
        let mut pre_p = [0u64; 4];
        for (i, a) in pre.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            pre_p[i] = p;
            guards.push(g);
        }
        let (pp, np, _gp) = match poly {
            Some(t) => {
                let (p, g) = t.buf.device_ptr(stream);
                (p, t.n as i32, Some(g))
            }
            None => (0, 0, None),
        };
        self.fwd[0].launch(stream, &LaunchPtrs { inp: xp, out: sp, pre: pre_p, ppre: pp, nppre: np, ..Default::default() })?;
        self.fwd[1].launch(stream, &LaunchPtrs { inp: sp, out: sp, ..Default::default() })?;
        Ok(())
    }

    /// inverse transform along Y and X only (input: Z already inverted, `Zn` valid planes)
    pub fn inverse_yx(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut CudaSlice<R::Complex>,
        out: &mut CudaSlice<R::Complex>,
        post: &[&CudaSlice<R::Complex>],
        poly: Option<&PolyTable<R>>,
    ) -> Result<(), FftError> {
        let (sp, _g1) = scratch.device_ptr_mut(stream);
        let (op, _g2) = out.device_ptr_mut(stream);
        let mut guards = Vec::new();
        let mut post_p = [0u64; 4];
        for (i, a) in post.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            post_p[i] = p;
            guards.push(g);
        }
        let (pp, np, _gp) = match poly {
            Some(t) => {
                let (p, g) = t.buf.device_ptr(stream);
                (p, t.n as i32, Some(g))
            }
            None => (0, 0, None),
        };
        self.inv[1].launch(stream, &LaunchPtrs { inp: sp, out: sp, ..Default::default() })?;
        self.inv[2].launch(stream, &LaunchPtrs { inp: sp, out: op, post: post_p, ppost: pp, nppost: np, ..Default::default() })?;
        Ok(())
    }

    /// `out (+)= post . crop(IFFT(scratch)) / prod(pad)`; `scratch` is destroyed.
    pub fn inverse(
        &self,
        stream: &Arc<CudaStream>,
        scratch: &mut CudaSlice<R::Complex>,
        out: &mut CudaSlice<R::Complex>,
        post: &[&CudaSlice<R::Complex>],
        poly: Option<&PolyTable<R>>,
    ) -> Result<(), FftError> {
        let (sp, _g1) = scratch.device_ptr_mut(stream);
        let (op, _g2) = out.device_ptr_mut(stream);
        let mut guards = Vec::new();
        let mut post_p = [0u64; 4];
        for (i, a) in post.iter().enumerate() {
            let (p, g) = a.device_ptr(stream);
            post_p[i] = p;
            guards.push(g);
        }
        let (pp, np, _gp) = match poly {
            Some(t) => {
                let (p, g) = t.buf.device_ptr(stream);
                (p, t.n as i32, Some(g))
            }
            None => (0, 0, None),
        };
        self.inv[0].launch(
            stream,
            &LaunchPtrs {
                inp: sp,
                out: sp,
                ..Default::default()
            },
        )?;
        self.inv[1].launch(
            stream,
            &LaunchPtrs {
                inp: sp,
                out: sp,
                ..Default::default()
            },
        )?;
        self.inv[2].launch(
            stream,
            &LaunchPtrs {
                inp: sp,
                out: op,
                post: post_p,
                ppost: pp,
                nppost: np,
                ..Default::default()
            },
        )?;
        Ok(())
    }
}
