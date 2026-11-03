//! Multi-coil non-Cartesian encoding operator with an optional low-rank phase model:
//! E x = Σ_l diag(a_l) F (S ∘ (b_l ∘ x)), F = type-2 NUFFT (sign −1), E^H by type-1 (sign +1).
//! Plain SENSE is L = 0 (a = b = 1).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{NufftOpts, NufftPlan};

use crate::gpu::Kernels;
use crate::Error;

pub struct SenseOp {
    pub dims: [usize; 3],
    pub nv: usize,
    pub nc: usize,
    pub m: usize,
    nufft: NufftPlan<f32>,
    smaps: CudaSlice<C32>,
    seg_a: Vec<CudaSlice<C32>>,
    seg_b: Vec<CudaSlice<C32>>,
    img: CudaSlice<C32>,
    ksp: CudaSlice<C32>,
    ksp2: CudaSlice<C32>,
    kern: Arc<Kernels>,
}

impl SenseOp {
    /// `coords`: 3 arrays of k·Δ (radians per voxel); `smaps`: nc × nv (whitened); `seg`: per-term (a: m, b: nv)
    pub fn new(
        dev: &Device,
        stream: &Arc<CudaStream>,
        kern: Arc<Kernels>,
        dims: [usize; 3],
        coords: [&[f32]; 3],
        smaps: &[C32],
        nc: usize,
        seg: Vec<(Vec<C32>, Vec<C32>)>,
        tol: f64,
    ) -> Result<Self, Error> {
        let nv = dims[0] * dims[1] * dims[2];
        let m = coords[0].len();
        let mut nufft = NufftPlan::<f32>::new(dev, &dims, NufftOpts { tol, sigma: 2.0, ntransf: nc, ..Default::default() })?;
        let dc: Vec<CudaSlice<f32>> = coords.iter().map(|c| stream.clone_htod(*c)).collect::<Result<_, _>>()?;
        nufft.set_points(stream, &[&dc[0], &dc[1], &dc[2]], m)?;
        let mut seg_a = Vec::new();
        let mut seg_b = Vec::new();
        for (a, b) in &seg {
            seg_a.push(stream.clone_htod(a)?);
            seg_b.push(stream.clone_htod(b)?);
        }
        Ok(Self {
            dims,
            nv,
            nc,
            m,
            nufft,
            smaps: stream.clone_htod(smaps)?,
            seg_a,
            seg_b,
            img: stream.alloc_zeros(nc * nv)?,
            ksp: stream.alloc_zeros(nc * m)?,
            ksp2: stream.alloc_zeros(nc * m)?,
            kern,
        })
    }

    pub fn n_terms(&self) -> usize {
        self.seg_a.len().max(1)
    }

    /// y (nc × m) = E x
    pub fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (nv, nc, m) = (self.nv, self.nc, self.m);
        if self.seg_a.is_empty() {
            self.kern.coil_expand(st, x, None, &self.smaps, &mut self.img, nv, nc)?;
            self.nufft.type2(st, &self.img, y, -1)?;
            return Ok(());
        }
        for l in 0..self.seg_a.len() {
            self.kern.coil_expand(st, x, Some(&self.seg_b[l]), &self.smaps, &mut self.img, nv, nc)?;
            self.nufft.type2(st, &self.img, &mut self.ksp, -1)?;
            self.kern.sample_weight(st, &self.ksp, &self.seg_a[l], y, m, nc, false, l > 0)?;
        }
        Ok(())
    }

    /// x (nv) = E^H y
    pub fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (nv, nc, m) = (self.nv, self.nc, self.m);
        if self.seg_a.is_empty() {
            self.nufft.type1(st, y, &mut self.img, 1)?;
            self.kern.coil_reduce(st, &self.img, None, &self.smaps, x, nv, nc, false)?;
            return Ok(());
        }
        for l in 0..self.seg_a.len() {
            self.kern.sample_weight(st, y, &self.seg_a[l], &mut self.ksp, m, nc, true, false)?;
            self.nufft.type1(st, &self.ksp, &mut self.img, 1)?;
            self.kern.coil_reduce(st, &self.img, Some(&self.seg_b[l]), &self.smaps, x, nv, nc, l > 0)?;
        }
        Ok(())
    }

    /// out = E^H E x
    pub fn normal(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut tmp = std::mem::replace(&mut self.ksp2, st.alloc_zeros(0)?);
        self.forward(st, x, &mut tmp)?;
        self.adjoint(st, &tmp, out)?;
        self.ksp2 = tmp;
        Ok(())
    }

    /// Exchange the coil maps with `s` (e.g. maps carrying a per-encode phase); call again to restore.
    pub fn swap_smaps(&mut self, s: &mut CudaSlice<C32>) {
        assert_eq!(s.len(), self.smaps.len());
        std::mem::swap(&mut self.smaps, s);
    }

    /// Switch to another sample set (plain operators only: segmented terms are tied to one sample set).
    pub fn set_points(&mut self, st: &Arc<CudaStream>, coords: [&CudaSlice<f32>; 3], m: usize) -> Result<(), Error> {
        assert!(self.seg_a.is_empty(), "set_points on a segmented operator");
        self.nufft.set_points(st, &coords, m)?;
        if self.m != m {
            self.m = m;
            self.ksp = st.alloc_zeros(self.nc * m)?;
            self.ksp2 = st.alloc_zeros(self.nc * m)?;
        }
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kern
    }
}
