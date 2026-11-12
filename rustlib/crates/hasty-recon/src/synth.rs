//! Model-consistent ("inverse crime") test data that are never stored: a [`SetOp`] wrapper whose residuals are those of
//! y_t = E_t x★_t − n_t,
//!
//!     r_t = E_t x − y_t = E_t (x − x★_t) + n_t,
//!
//! with x★_t a reference image supplied per set on the host and n_t complex Gaussian noise drawn on the device from a
//! counter-based hash of (seed, set, coil, sample): the same noise in every pass, for staged and resident data, and for
//! every coil subset (`select_coils`: the rows of the selected coils). The data argument of the residual methods is
//! ignored, so the solver can keep staging the measured data (their values are unused) and needs no second store.
//! Cost per residual: one host → device copy of x★_t and one elementwise pass; the operator work is unchanged.
//!
//! Use: separate solver / prior behaviour (this wrapper: exact model) from model error of the measured data (the same
//! run on the real data). Forward / adjoint (and their multi / group forms) are the inner operator's.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use num_complex::Complex64;

use crate::gpu::Kernels;
use crate::setop::SetOp;
use crate::Error;

const SRC: &str = r#"
struct __align__(8) cx { float x, y; };
__device__ __forceinline__ unsigned long long mix64(unsigned long long z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// r[j] += sigma (g1 + i g2), g1, g2 independent standard normals (Box-Muller on two 32-bit uniforms of the hash of
// key and j)
extern "C" __global__ void add_gauss(cx* __restrict__ r, long long m, unsigned long long key, float sigma)
{
    long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (j >= m) return;
    unsigned long long h = mix64(key ^ mix64((unsigned long long)j));
    float u1 = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
    float u2 = ((float)((h >> 8) & 0xFFFFFFull) + 0.5f) * (1.0f / 16777216.0f);
    float rad = sigma * sqrtf(-2.0f * logf(u1));
    float s, c;
    sincospif(2.0f * u2, &s, &c);
    r[j].x += rad * c;
    r[j].y += rad * s;
}
"#;

/// fills the reference image x★_t of set t (`len_x(t)` values)
pub type TruthFn<'a> = Box<dyn FnMut(usize, &mut [C32]) + 'a>;

pub struct SynthSets<'a> {
    inner: &'a mut dyn SetOp,
    truth: TruthFn<'a>,
    /// noise standard deviation per real component (E|n|² = 2 σ²)
    sigma: f32,
    seed: u64,
    ncoils: usize,
    coils: Option<Vec<usize>>,
    add_gauss: CudaFunction,
    host: Vec<C32>,
}

fn key(seed: u64, t: usize, c: usize) -> u64 {
    // splitmix64 of the (set, coil) counter, salted with the seed (host side; the device mixes in the sample index)
    let mut z = seed ^ ((t as u64) << 20 | c as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl<'a> SynthSets<'a> {
    /// `ncoils`: coils of the data (rows of `len_y`); `sigma`: per real component; `truth(t, out)` writes x★_t.
    pub fn new(dev: &Device, inner: &'a mut dyn SetOp, ncoils: usize, sigma: f64, seed: u64, truth: TruthFn<'a>) -> Result<Self, Error> {
        assert!(ncoils > 0);
        for t in 0..inner.nsets() {
            assert_eq!(inner.len_y(t) % ncoils, 0, "set {t}: data length not a multiple of the coil count");
        }
        let m = dev.load_module("recon_synth", SRC, &CompileSettings::default())?;
        Ok(Self { inner, truth, sigma: sigma as f32, seed, ncoils, coils: None, add_gauss: m.load_function("add_gauss")?, host: vec![] })
    }

    /// d = x − x★_t (device, `len_x(t)` values)
    fn diff(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>) -> Result<CudaSlice<C32>, Error> {
        let n = self.inner.len_x(t);
        assert_eq!(x.len(), n, "synthetic residual: image of set {t} has {} values, operator {n}", x.len());
        self.host.resize(n, C32::new(0.0, 0.0));
        (self.truth)(t, &mut self.host);
        let mut d = unsafe { st.alloc::<C32>(n)? };
        st.memcpy_htod(&self.host, &mut d)?;
        self.inner.kernels().axpby(st, Complex64::new(1.0, 0.0), x, Complex64::new(-1.0, 0.0), &mut d)?;
        Ok(d)
    }

    /// r += n_t on the rows of the selected coils
    fn add_noise(&self, st: &Arc<CudaStream>, t: usize, r: &mut CudaSlice<C32>) -> Result<(), Error> {
        if self.sigma == 0.0 {
            return Ok(());
        }
        let rows: Vec<usize> = match &self.coils {
            Some(c) => c.clone(),
            None => (0..self.ncoils).collect(),
        };
        assert_eq!(r.len() % rows.len(), 0);
        let m = r.len() / rows.len();
        for (i, &c) in rows.iter().enumerate() {
            let (k, m_i) = (key(self.seed, t, c), m as i64);
            let cfg = LaunchConfig { grid_dim: ((m as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut row = r.slice_mut(i * m..(i + 1) * m);
            unsafe { st.launch_builder(&self.add_gauss).arg(&mut row).arg(&m_i).arg(&k).arg(&self.sigma).launch(cfg)? };
        }
        Ok(())
    }
}

impl SetOp for SynthSets<'_> {
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
    fn select_coils(&mut self, st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<bool, Error> {
        let ok = self.inner.select_coils(st, idx)?;
        if ok {
            self.coils = idx.map(|v| v.to_vec());
        }
        Ok(ok)
    }
    fn use_surrogate(&mut self, on: bool) -> bool {
        self.inner.use_surrogate(on)
    }
    fn n_coils(&self) -> usize {
        self.ncoils
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.inner.forward(st, t, x, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.inner.adjoint(st, t, y, x)
    }
    /// r = E_t (x − x★_t) + n_t (the data `_y` are not read)
    fn forward_residual(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, _y: &CudaSlice<C32>, r: &mut CudaSlice<C32>) -> Result<(), Error> {
        let d = self.diff(st, t, x)?;
        self.inner.forward(st, t, &d, r)?;
        self.add_noise(st, t, r)
    }
    fn forward_multi(&mut self, st: &Arc<CudaStream>, t: usize, xs: &[&CudaSlice<C32>], ws: &[Option<&CudaSlice<C32>>], y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.inner.forward_multi(st, t, xs, ws, y)
    }
    fn adjoint_multi(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, ws: &[Option<&CudaSlice<C32>>], xs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.inner.adjoint_multi(st, t, y, ws, xs)
    }
    fn forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.inner.forward_group(st, members, x, ys)
    }
    /// per member: r_m = E_m (x − x★_m) + n_m (each member has its own reference, so no shared gridding)
    fn forward_residual_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert!(members.len() == ys.len() && members.len() == rs.len());
        for ((&t, y), r) in members.iter().zip(ys).zip(rs.iter_mut()) {
            self.forward_residual(st, t, x, y, r)?;
        }
        Ok(())
    }
    fn adjoint_group(&mut self, st: &Arc<CudaStream>, members: &[usize], ys: &[&CudaSlice<C32>], x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.inner.adjoint_group(st, members, ys, x)
    }
    fn table_bytes(&self) -> (usize, usize) {
        self.inner.table_bytes()
    }
    fn tables_on_host(&self) -> bool {
        self.inner.tables_on_host()
    }
    fn place_tables(&mut self, ctx: &Arc<cudarc::driver::CudaContext>, st: &Arc<CudaStream>, on_host: bool) -> Result<i64, Error> {
        self.inner.place_tables(ctx, st, on_host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = (c + 1) · x per coil c (a trivial operator with coil subsets)
    struct Toy {
        k: Kernels,
        n: usize,
        nc: usize,
        sel: Option<Vec<usize>>,
    }
    impl SetOp for Toy {
        fn nsets(&self) -> usize {
            2
        }
        fn len_x(&self, _t: usize) -> usize {
            self.n
        }
        fn len_y(&self, _t: usize) -> usize {
            self.n * self.sel.as_ref().map_or(self.nc, |s| s.len())
        }
        fn kernels(&self) -> &Kernels {
            &self.k
        }
        fn select_coils(&mut self, _st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<bool, Error> {
            self.sel = idx.map(|v| v.to_vec());
            Ok(true)
        }
        fn forward(&mut self, st: &Arc<CudaStream>, _t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
            let rows: Vec<usize> = self.sel.clone().unwrap_or((0..self.nc).collect());
            for (i, &c) in rows.iter().enumerate() {
                let mut yi = st.alloc_zeros::<C32>(self.n)?;
                self.k.axpby(st, Complex64::new(c as f64 + 1.0, 0.0), x, Complex64::new(0.0, 0.0), &mut yi)?;
                st.memcpy_dtod(&yi, &mut y.slice_mut(i * self.n..(i + 1) * self.n))?;
            }
            Ok(())
        }
        fn adjoint(&mut self, _st: &Arc<CudaStream>, _t: usize, _y: &CudaSlice<C32>, _x: &mut CudaSlice<C32>) -> Result<(), Error> {
            unimplemented!()
        }
    }

    #[test]
    fn residual_is_model_error_plus_reproducible_noise() {
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let (n, nc) = (1 << 16, 4);
        let mut toy = Toy { k: Kernels::new(&dev).unwrap(), n, nc, sel: None };
        let xt: Vec<C32> = (0..n).map(|q| C32::new((q % 7) as f32, -((q % 3) as f32))).collect();
        let xt2 = xt.clone();
        let sigma = 0.5;
        let mut s = SynthSets::new(&dev, &mut toy, nc, sigma, 7, Box::new(move |_t, out: &mut [C32]| out.copy_from_slice(&xt2))).unwrap();
        let dummy = st.alloc_zeros::<C32>(n * nc).unwrap();
        // at x = x★ the residual is the noise: mean 0, E|n|² = 2σ², reproducible
        let x = st.clone_htod(&xt).unwrap();
        let mut r = st.alloc_zeros::<C32>(n * nc).unwrap();
        s.forward_residual(&st, 1, &x, &dummy, &mut r).unwrap();
        let h = st.clone_dtoh(&r).unwrap();
        let (mut m2, mut mr, mut mi) = (0.0f64, 0.0f64, 0.0f64);
        for v in &h {
            m2 += (v.re * v.re + v.im * v.im) as f64;
            mr += v.re as f64;
            mi += v.im as f64;
        }
        let nn = h.len() as f64;
        assert!((m2 / nn / (2.0 * sigma * sigma) - 1.0).abs() < 0.02, "E|n|² / 2σ² = {}", m2 / nn / (2.0 * sigma * sigma));
        assert!((mr / nn).abs() < 0.01 && (mi / nn).abs() < 0.01);
        let mut r2 = st.alloc_zeros::<C32>(n * nc).unwrap();
        s.forward_residual(&st, 1, &x, &dummy, &mut r2).unwrap();
        assert_eq!(h, st.clone_dtoh(&r2).unwrap(), "the same noise in every pass");
        // another set: different noise
        s.forward_residual(&st, 0, &x, &dummy, &mut r2).unwrap();
        let h0 = st.clone_dtoh(&r2).unwrap();
        let c: f64 = h.iter().zip(&h0).map(|(a, b)| (a.re * b.re + a.im * b.im) as f64).sum::<f64>() / m2;
        assert!(c.abs() < 0.02, "sets 0 and 1 noise correlation {c}");
        // coil subset {3, 1}: the rows of coils 3 and 1 of the full residual
        assert!(s.select_coils(&st, Some(&[3, 1])).unwrap());
        let mut rs = st.alloc_zeros::<C32>(2 * n).unwrap();
        s.forward_residual(&st, 1, &x, &dummy, &mut rs).unwrap();
        let hs = st.clone_dtoh(&rs).unwrap();
        assert_eq!(&hs[..n], &h[3 * n..4 * n]);
        assert_eq!(&hs[n..], &h[n..2 * n]);
        s.select_coils(&st, None).unwrap();
        // model error: x = x★ + δ → r = E δ + n
        let mut xd = xt.clone();
        xd[5].re += 2.0;
        let x = st.clone_htod(&xd).unwrap();
        s.forward_residual(&st, 1, &x, &dummy, &mut r2).unwrap();
        let hd = st.clone_dtoh(&r2).unwrap();
        for c in 0..nc {
            let j = c * n + 5;
            assert!(((hd[j].re - h[j].re) - 2.0 * (c as f32 + 1.0)).abs() < 1e-4);
        }
    }
}
