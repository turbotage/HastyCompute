//! Elementwise kernels (coil expand/reduce, per-sample weights, axpby, dot).

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use num_complex::Complex64;

use crate::Error;

pub struct Kernels {
    coil_expand: CudaFunction,
    coil_reduce: CudaFunction,
    sample_weight: CudaFunction,
    axpby: CudaFunction,
    dot_partial: CudaFunction,
    mul_real: CudaFunction,
    part: std::sync::Mutex<Option<CudaSlice<f64>>>,
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn ptr(s: Option<&CudaSlice<C32>>, stream: &Arc<CudaStream>) -> u64 {
    s.map(|v| v.device_ptr(stream).0).unwrap_or(0)
}

const NPART: usize = 1024;

impl Kernels {
    pub fn new(dev: &Device) -> Result<Self, Error> {
        let m = dev.load_module("recon_elementwise", include_str!("elementwise.cu"), &CompileSettings::default())?;
        Ok(Self {
            coil_expand: m.load_function("coil_expand")?,
            coil_reduce: m.load_function("coil_reduce")?,
            sample_weight: m.load_function("sample_weight")?,
            axpby: m.load_function("axpby")?,
            dot_partial: m.load_function("dot_partial")?,
            mul_real: m.load_function("mul_real")?,
            part: std::sync::Mutex::new(None),
        })
    }

    /// out[c][v] = s[c][v] · d[v] · x[v]
    pub fn coil_expand(&self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, d: Option<&CudaSlice<C32>>, s: &CudaSlice<C32>, out: &mut CudaSlice<C32>, nv: usize, nc: usize) -> Result<(), Error> {
        let (nv_i, nc_i, dp) = (nv as i64, nc as i32, ptr(d, st));
        unsafe { st.launch_builder(&self.coil_expand).arg(x).arg(&dp).arg(s).arg(out).arg(&nv_i).arg(&nc_i).launch(cfg(nv * nc))? };
        Ok(())
    }
    /// out[v] (+)= conj(d[v]) Σ_c conj(s[c][v]) y[c][v]
    pub fn coil_reduce(&self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, d: Option<&CudaSlice<C32>>, s: &CudaSlice<C32>, out: &mut CudaSlice<C32>, nv: usize, nc: usize, acc: bool) -> Result<(), Error> {
        let (nv_i, nc_i, dp, a) = (nv as i64, nc as i32, ptr(d, st), acc as i32);
        unsafe { st.launch_builder(&self.coil_reduce).arg(y).arg(&dp).arg(s).arg(out).arg(&nv_i).arg(&nc_i).arg(&a).launch(cfg(nv))? };
        Ok(())
    }
    /// y[c][j] (+)= w[j] z[c][j] (or conj(w))
    pub fn sample_weight(&self, st: &Arc<CudaStream>, z: &CudaSlice<C32>, w: &CudaSlice<C32>, y: &mut CudaSlice<C32>, m: usize, nc: usize, conj: bool, acc: bool) -> Result<(), Error> {
        let (m_i, nc_i, cj, a) = (m as i64, nc as i32, conj as i32, acc as i32);
        unsafe { st.launch_builder(&self.sample_weight).arg(z).arg(w).arg(y).arg(&m_i).arg(&nc_i).arg(&cj).arg(&a).launch(cfg(m * nc))? };
        Ok(())
    }
    /// y = a x + b y
    pub fn axpby(&self, st: &Arc<CudaStream>, a: Complex64, x: &CudaSlice<C32>, b: Complex64, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let n = y.len();
        let (av, bv, n_i) = (C32::new(a.re as f32, a.im as f32), C32::new(b.re as f32, b.im as f32), n as i64);
        unsafe { st.launch_builder(&self.axpby).arg(x).arg(y).arg(&av).arg(&bv).arg(&n_i).launch(cfg(n))? };
        Ok(())
    }
    /// out = Re(d) ∘ x
    pub fn mul_real(&self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, d: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        let n = x.len();
        let n_i = n as i64;
        unsafe { st.launch_builder(&self.mul_real).arg(x).arg(d).arg(out).arg(&n_i).launch(cfg(n))? };
        Ok(())
    }
    /// ⟨x, y⟩ = Σ conj(x) y (f64 accumulation)
    pub fn dot(&self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &CudaSlice<C32>) -> Result<Complex64, Error> {
        let n = x.len().min(y.len());
        let mut guard = self.part.lock().unwrap();
        if guard.is_none() {
            *guard = Some(st.alloc_zeros::<f64>(2 * NPART)?);
        }
        let part = guard.as_mut().unwrap();
        let n_i = n as i64;
        let c = LaunchConfig { grid_dim: (NPART as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { st.launch_builder(&self.dot_partial).arg(x).arg(y).arg(&mut *part).arg(&n_i).launch(c)? };
        let h = st.clone_dtoh(part)?;
        let (mut re, mut im) = (0.0, 0.0);
        for b in 0..NPART {
            re += h[2 * b];
            im += h[2 * b + 1];
        }
        Ok(Complex64::new(re, im))
    }
}
