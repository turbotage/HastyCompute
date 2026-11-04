//! Operator abstractions for the solvers: one linear operator (`LinOp`, CG family) and a stack of sample sets on one
//! image grid (`SetOp`, P²nP), with implementations for the plain SENSE operator and the segmented physics NUFFT.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use hasty_nufft::{SegNufft, SegPoints, SegSegments};

use crate::gpu::Kernels;
use crate::op::SenseOp;
use crate::Error;

pub trait LinOp {
    /// image / data lengths (complex elements)
    fn len_x(&self) -> usize;
    fn len_y(&self) -> usize;
    fn kernels(&self) -> &Kernels;
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error>;
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error>;
    /// out = A^H A x
    fn normal(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut y = st.alloc_zeros::<C32>(self.len_y())?;
        self.forward(st, x, &mut y)?;
        self.adjoint(st, &y, out)
    }
}

/// Encoding operators of a stack of sample sets sharing one image grid: set t maps an image of `len_x(t)` elements
/// (one frame, or all frames of a time-resolved set) to `len_y(t)` data (coil-major).
pub trait SetOp {
    fn nsets(&self) -> usize;
    fn len_x(&self, t: usize) -> usize;
    fn len_y(&self, t: usize) -> usize;
    fn kernels(&self) -> &Kernels;
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error>;
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error>;
}

/// One set of a stack as a [`LinOp`].
pub struct SetView<'a> {
    pub op: &'a mut dyn SetOp,
    pub t: usize,
}

impl LinOp for SetView<'_> {
    fn len_x(&self) -> usize {
        self.op.len_x(self.t)
    }
    fn len_y(&self) -> usize {
        self.op.len_y(self.t)
    }
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.op.forward(st, self.t, x, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.op.adjoint(st, self.t, y, x)
    }
}

impl LinOp for SenseOp {
    fn len_x(&self) -> usize {
        self.nv
    }
    fn len_y(&self) -> usize {
        self.nc * self.m
    }
    fn kernels(&self) -> &Kernels {
        SenseOp::kernels(self)
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        SenseOp::forward(self, st, x, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        SenseOp::adjoint(self, st, y, x)
    }
    fn normal(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        SenseOp::normal(self, st, x, out)
    }
}

/// Plain SENSE sets: one `SenseOp` re-pointed to each set's coordinates, optional per-set coil maps (e.g. maps carrying
/// a per-encode phase).
pub struct SenseSets<'a> {
    pub op: &'a mut SenseOp,
    pub coords: Vec<[CudaSlice<f32>; 3]>,
    pub ms: Vec<usize>,
    pub maps: Vec<CudaSlice<C32>>,
    pub map_of_set: Vec<usize>,
}

impl SenseSets<'_> {
    fn with<R>(&mut self, st: &Arc<CudaStream>, t: usize, f: impl FnOnce(&mut SenseOp) -> Result<R, Error>) -> Result<R, Error> {
        let c = &self.coords[t];
        self.op.set_points(st, [&c[0], &c[1], &c[2]], self.ms[t])?;
        let mi = self.map_of_set.get(t).copied();
        if let Some(i) = mi {
            self.op.swap_smaps(&mut self.maps[i]);
        }
        let r = f(self.op);
        if let Some(i) = mi {
            self.op.swap_smaps(&mut self.maps[i]);
        }
        r
    }
}

impl SetOp for SenseSets<'_> {
    fn nsets(&self) -> usize {
        self.ms.len()
    }
    fn len_x(&self, _t: usize) -> usize {
        self.op.nv
    }
    fn len_y(&self, t: usize) -> usize {
        self.op.nc * self.ms[t]
    }
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.with(st, t, |op| op.forward(st, x, y))
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.with(st, t, |op| op.adjoint(st, y, x))
    }
}

/// Segmented physics NUFFT sets: one engine (grid slots, FFT plans, kernels, coil maps), per-set points, segment tables
/// shared by the sets of one encode.
pub struct SegSets {
    pub eng: SegNufft,
    pub pts: Vec<SegPoints>,
    pub segs: Vec<SegSegments>,
    pub seg_of_set: Vec<usize>,
    pub kern: Arc<Kernels>,
}

impl SetOp for SegSets {
    fn nsets(&self) -> usize {
        self.pts.len()
    }
    fn len_x(&self, _t: usize) -> usize {
        self.eng.nv() * self.eng.frames
    }
    fn len_y(&self, t: usize) -> usize {
        self.eng.nc * self.pts[t].m
    }
    fn kernels(&self) -> &Kernels {
        &self.kern
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        Ok(self.eng.forward(st, &self.segs[self.seg_of_set[t]], &self.pts[t], x, y)?)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        Ok(self.eng.adjoint(st, &self.segs[self.seg_of_set[t]], &self.pts[t], y, x)?)
    }
}

/// Per-sample weights around any stack of sets: E'_t = diag(w_t) E_t. Used for the voxel basis function: a rect
/// voxel of size Δ contributes Π_a sinc(k_a Δ_a / 2) at sample k (a cubic B-spline basis: the 4th power), which the
/// plain operators (Dirac voxels) omit (research/velocity_model_sr.md §B1).
pub struct WeightedSets<'a> {
    pub inner: &'a mut dyn SetOp,
    /// per set, one complex weight per sample
    pub w: Vec<CudaSlice<C32>>,
}

impl<'a> WeightedSets<'a> {
    pub fn new(st: &Arc<CudaStream>, inner: &'a mut dyn SetOp, w: Vec<Vec<C32>>) -> Result<Self, Error> {
        assert_eq!(w.len(), inner.nsets());
        let w = w.iter().map(|v| st.clone_htod(v)).collect::<Result<_, _>>()?;
        Ok(Self { inner, w })
    }

    /// voxel-basis weights for one set's coordinates (k·Δ, radians per voxel): Π_a sinc(c_a / 2)^power
    pub fn basis_weights(coords: [&[f32]; 3], power: i32) -> Vec<C32> {
        let sinc = |x: f64| if x.abs() < 1e-8 { 1.0 } else { x.sin() / x };
        (0..coords[0].len()).map(|n| C32::new((0..3).map(|a| sinc(coords[a][n] as f64 / 2.0)).product::<f64>().powi(power) as f32, 0.0)).collect()
    }
}

impl SetOp for WeightedSets<'_> {
    fn nsets(&self) -> usize {
        self.inner.nsets()
    }
    fn len_x(&self, t: usize) -> usize {
        self.inner.len_x(t)
    }
    fn len_y(&self, t: usize) -> usize {
        self.inner.len_y(t)
    }
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        // per-call scratch (stream-ordered allocation): no second copy of all data
        let mut tmp = st.alloc_zeros::<C32>(y.len())?;
        self.inner.forward(st, t, x, &mut tmp)?;
        let m = self.w[t].len();
        let nc = y.len() / m;
        self.inner.kernels().sample_weight(st, &tmp, &self.w[t], y, m, nc, false, false)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let m = self.w[t].len();
        let nc = y.len() / m;
        let mut tmp = st.alloc_zeros::<C32>(y.len())?;
        self.inner.kernels().sample_weight(st, y, &self.w[t], &mut tmp, m, nc, true, false)?;
        self.inner.adjoint(st, t, &tmp, x)
    }
}
