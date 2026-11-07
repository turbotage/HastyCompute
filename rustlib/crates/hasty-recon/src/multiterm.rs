//! Exact per-sample time inside short readout windows: a window's samples are taken at τ = TE + δ, |δ| ≤ h, and the
//! image changes with δ (off-resonance, R2*, fat peaks). With L nodes δ_l and interpolators a_l(δ),
//!
//!   x(q, TE + δ) ≈ Σ_l a_l(δ) x(q, TE + δ_l),   y_n = Σ_l a_l(δ_n) F[S x(·, TE + δ_l)](k_n),
//!
//! the forward model is exact up to the interpolation error for every signal whose components have complex rates in
//! the design set {r + i2πf : 0 ≤ r ≤ r_max, |f| ≤ f_max} ("universal" band-limited interpolators: least squares over
//! that set, independent of the voxel parameters, so the image unknowns at the nodes are just the model evaluated at
//! TE + δ_l). `MultiTermSets` is the corresponding stack of sample sets (L node images per set).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

use crate::gpu::Kernels;
use crate::op::SenseOp;
use crate::setop::SetOp;
use crate::Error;

/// band-limited interpolators on [−h, h]: Chebyshev nodes and a table of a_l(δ) (`ntab` points)
#[derive(Clone, Debug)]
pub struct BandInterp {
    pub h: f64,
    pub nodes: Vec<f64>,
    pub ntab: usize,
    /// [ntab][L]
    pub tab: Vec<Complex64>,
    /// max |e^{−zδ} − Σ a_l e^{−zδ_l}| over the design set and the table
    pub max_err: f64,
}

impl BandInterp {
    pub fn new(h: f64, l: usize, f_max: f64, r_max: f64, ntab: usize) -> Self {
        let nodes: Vec<f64> = (0..l).map(|k| if l == 1 { 0.0 } else { -h * ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * l) as f64).cos() }).collect();
        // design rates
        let nf = 121;
        let nr = 6;
        let rates: Vec<Complex64> = (0..nr)
            .flat_map(|ir| (0..nf).map(move |jf| Complex64::new(r_max * ir as f64 / (nr - 1).max(1) as f64, 2.0 * std::f64::consts::PI * f_max * (2.0 * jf as f64 / (nf - 1) as f64 - 1.0))))
            .collect();
        let b = DMatrix::<Complex64>::from_fn(rates.len(), l, |r, k| (-rates[r] * nodes[k]).exp());
        let bh = b.adjoint();
        let mut g = &bh * &b;
        let tr = (0..l).map(|k| g[(k, k)].re).sum::<f64>() / l as f64;
        for k in 0..l {
            g[(k, k)] += Complex64::new(1e-10 * tr, 0.0);
        }
        let chol = g.cholesky().expect("interpolator Gram");
        let mut tab = Vec::with_capacity(ntab * l);
        let mut max_err: f64 = 0.0;
        for i in 0..ntab {
            let d = -h + 2.0 * h * i as f64 / (ntab - 1) as f64;
            let y = DVector::<Complex64>::from_iterator(rates.len(), rates.iter().map(|z| (-z * d).exp()));
            let a = chol.solve(&(&bh * &y));
            let r = &b * &a - &y;
            max_err = max_err.max(r.iter().map(|c| c.norm()).fold(0.0, f64::max));
            tab.extend(a.iter().copied());
        }
        Self { h, nodes, ntab, tab, max_err }
    }
    pub fn l(&self) -> usize {
        self.nodes.len()
    }
    /// a_l(δ) by linear interpolation in the table
    pub fn at(&self, d: f64) -> Vec<Complex64> {
        let l = self.l();
        let f = ((d + self.h) / (2.0 * self.h) * (self.ntab - 1) as f64).clamp(0.0, (self.ntab - 1) as f64 - 1e-9);
        let i = f.floor() as usize;
        let w = f - i as f64;
        (0..l).map(|k| self.tab[i * l + k] * (1.0 - w) + self.tab[(i + 1).min(self.ntab - 1) * l + k] * w).collect()
    }
}

/// sample sets whose image is L node images back to back ([L][nv]); set t, term l has per-sample weights a_{t,l}
pub struct MultiTermSets<'a> {
    pub op: &'a mut SenseOp,
    pub coords: Vec<[CudaSlice<f32>; 3]>,
    pub ms: Vec<usize>,
    /// [t][l] per-sample weights (m_t)
    pub w: Vec<Vec<CudaSlice<C32>>>,
    pub l: usize,
    tmp_x: CudaSlice<C32>,
    tmp_y: CudaSlice<C32>,
}

impl<'a> MultiTermSets<'a> {
    pub fn new(st: &Arc<CudaStream>, op: &'a mut SenseOp, coords: Vec<[CudaSlice<f32>; 3]>, ms: Vec<usize>, w: Vec<Vec<CudaSlice<C32>>>) -> Result<Self, Error> {
        let l = w.first().map_or(1, |x| x.len());
        let mmax = ms.iter().copied().max().unwrap_or(1);
        let (nv, nc) = (op.nv, op.nc);
        Ok(Self { tmp_x: st.alloc_zeros(nv)?, tmp_y: st.alloc_zeros(nc * mmax)?, op, coords, ms, w, l })
    }
}

impl SetOp for MultiTermSets<'_> {
    fn nsets(&self) -> usize {
        self.ms.len()
    }
    fn len_x(&self, _t: usize) -> usize {
        self.l * self.op.nv
    }
    fn len_y(&self, t: usize) -> usize {
        self.op.nc * self.ms[t]
    }
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (nv, nc, m) = (self.op.nv, self.op.nc, self.ms[t]);
        let c = &self.coords[t];
        self.op.set_points(st, [&c[0], &c[1], &c[2]], m)?;
        for l in 0..self.l {
            st.memcpy_dtod(&x.slice(l * nv..(l + 1) * nv), &mut self.tmp_x)?;
            self.op.forward(st, &self.tmp_x, &mut self.tmp_y)?;
            self.op.kernels().sample_weight(st, &self.tmp_y, &self.w[t][l], y, m, nc, false, l > 0)?;
        }
        Ok(())
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (nv, nc, m) = (self.op.nv, self.op.nc, self.ms[t]);
        let c = &self.coords[t];
        self.op.set_points(st, [&c[0], &c[1], &c[2]], m)?;
        for l in 0..self.l {
            self.op.kernels().sample_weight(st, y, &self.w[t][l], &mut self.tmp_y, m, nc, true, false)?;
            self.op.adjoint(st, &self.tmp_y, &mut self.tmp_x)?;
            st.memcpy_dtod(&self.tmp_x, &mut x.slice_mut(l * nv..(l + 1) * nv))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_error_falls_with_the_number_of_nodes() {
        // the B0 train: half window 0.47 ms, |f| ≤ 1050 Hz (water ±500 Hz + fat peaks), R2* ≤ 500 /s
        let errs: Vec<f64> = (1..=8).map(|l| BandInterp::new(0.47e-3, l, 1050.0, 500.0, 101).max_err).collect();
        for (l, e) in errs.iter().enumerate() {
            eprintln!("L = {}: max error {e:.2e}", l + 1);
        }
        assert!(errs[5] < errs[3] && errs[3] < errs[2], "{errs:?}");
        assert!(errs[7] < 1e-3, "L = 8 error {}", errs[7]);
    }
}
