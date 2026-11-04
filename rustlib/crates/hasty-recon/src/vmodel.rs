//! Velocity-parametrised 4D flow model (`research/velocity_model_sr.md` §A): per voxel, a complex reference z and a
//! velocity v, each in a periodic B-spline basis over the cardiac cycle, plus a global low-order residual phase per
//! encode (Legendre, total degree ≤ d). Operator frames f = 0..F (cardiac phase (f + ½)/F) are evaluated pointwise:
//!
//!   x_{e,f}(r) = z_f(r) · exp(i (φ_e(r) − k_e·v_f(r))),   z_f = Σ_k Wz[f,k] z_k,   v_f = Σ_k Wv[f,k] v_k,
//!   φ_0 = 0, φ_e(r) = Σ_p a_{e,p} P_p(r) (e ≥ 1),        k_e = γ M1_e (hasty-mrsim convention x_e ∝ e^{−iγM1·v})
//!
//! Parameters live in a [`Stack`] of complex chunks so the P²nP vector algebra applies unchanged (real inner product
//! of the packed reals): chunks 0..K_z = z_k (nv complex), K_z..K_z+K_v = v_k (3·nv reals component-major, packed in
//! pairs), last = a (packed reals). GPU kernels: `vmodel.cu`; an f64 CPU reference (`cpu`) backs the tests.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};

use crate::p2np::Stack;
use crate::Error;

/// centred cardinal B-spline of degree p (0..=3)
pub fn bspline(p: u32, x: f64) -> f64 {
    let a = x.abs();
    match p {
        0 => (a < 0.5) as u8 as f64,
        1 => (1.0 - a).max(0.0),
        2 => {
            if a < 0.5 {
                0.75 - a * a
            } else if a < 1.5 {
                0.5 * (1.5 - a).powi(2)
            } else {
                0.0
            }
        }
        3 => {
            if a < 1.0 {
                2.0 / 3.0 - a * a + 0.5 * a * a * a
            } else if a < 2.0 {
                (2.0 - a).powi(3) / 6.0
            } else {
                0.0
            }
        }
        _ => panic!("B-spline degree {p} (0..=3)"),
    }
}

/// periodic B-spline weights W[f][k] (row-major, nf × k): node k at cycle fraction (k + ½)/K, frame f at (f + ½)/F;
/// rows sum to 1. K = F with degree 0 or 1 is the identity.
pub fn bspline_weights(k: usize, nf: usize, degree: u32) -> Vec<f64> {
    let mut w = vec![0.0; nf * k];
    for f in 0..nf {
        let s = k as f64 * (f as f64 + 0.5) / nf as f64 - 0.5;
        for kk in 0..k {
            let mut acc = 0.0;
            for j in -3i64..=3 {
                acc += bspline(degree, s - kk as f64 - (j * k as i64) as f64);
            }
            w[f * k + kk] = acc;
        }
        let sum: f64 = w[f * k..(f + 1) * k].iter().sum();
        for kk in 0..k {
            w[f * k + kk] /= sum;
        }
    }
    w
}

/// Legendre exponent triples of total degree ≤ d (degree-ordered; (0,0,0) first)
pub fn poly_exps(d: usize) -> Vec<[i32; 3]> {
    assert!(d <= 3, "residual-phase degree ≤ 3");
    let mut v = vec![];
    for t in 0..=d as i32 {
        for a in (0..=t).rev() {
            for b in (0..=t - a).rev() {
                v.push([a, b, t - a - b]);
            }
        }
    }
    v
}

pub fn legendre(n: i32, u: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => u,
        2 => 0.5 * (3.0 * u * u - 1.0),
        3 => 0.5 * u * (5.0 * u * u - 3.0),
        _ => panic!("Legendre degree {n}"),
    }
}

/// P_p at voxel r (coordinates normalised to (−1, 1) at voxel centres)
pub fn poly_at(ex: &[i32; 3], r: usize, dims: [usize; 3]) -> f64 {
    let [nx, ny, nz] = dims;
    let (i, j, k) = (r % nx, (r / nx) % ny, r / (nx * ny));
    let u = |i: usize, n: usize| (2.0 * i as f64 + 1.0) / n as f64 - 1.0;
    legendre(ex[0], u(i, nx)) * legendre(ex[1], u(j, ny)) * legendre(ex[2], u(k, nz))
}

/// sizes of a velocity model
#[derive(Clone, Debug)]
pub struct VelSpec {
    pub dims: [usize; 3],
    /// k_e = γ M1_e per encode (rad per m/s)
    pub kenc: Vec<[f64; 3]>,
    /// operator frames
    pub nf: usize,
    pub kz: usize,
    pub deg_z: u32,
    pub kv: usize,
    pub deg_v: u32,
    /// residual-phase total degree (None: no residual phase)
    pub poly: Option<usize>,
}

impl VelSpec {
    pub fn nv(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn ne(&self) -> usize {
        self.kenc.len()
    }
    pub fn np(&self) -> usize {
        self.poly.map_or(0, |d| poly_exps(d).len())
    }
    pub fn na(&self) -> usize {
        (self.ne() - 1) * self.np()
    }
    /// chunk lengths of the parameter stack (complex elements)
    pub fn lens(&self) -> Vec<usize> {
        let nv = self.nv();
        let mut l = vec![nv; self.kz];
        l.extend(std::iter::repeat_n((3 * nv).div_ceil(2), self.kv));
        l.push(self.na().div_ceil(2).max(1));
        l
    }
    pub fn chunk_z(&self, k: usize) -> usize {
        k
    }
    pub fn chunk_v(&self, k: usize) -> usize {
        self.kz + k
    }
    pub fn chunk_a(&self) -> usize {
        self.kz + self.kv
    }
    pub fn wz(&self) -> Vec<f64> {
        bspline_weights(self.kz, self.nf, self.deg_z)
    }
    pub fn wv(&self) -> Vec<f64> {
        bspline_weights(self.kv, self.nf, self.deg_v)
    }
    /// bytes of one parameter stack
    pub fn stack_bytes(&self) -> usize {
        self.lens().iter().sum::<usize>() * 8
    }
}

/// f64 reference implementation (tests, initialisation).
pub mod cpu {
    use num_complex::Complex64 as C;

    use super::*;

    #[derive(Clone, Debug)]
    pub struct Theta {
        pub z: Vec<Vec<C>>,
        pub v: Vec<Vec<[f64; 3]>>,
        pub a: Vec<f64>,
    }

    impl Theta {
        pub fn zeros(s: &VelSpec) -> Self {
            let nv = s.nv();
            Self { z: vec![vec![C::new(0.0, 0.0); nv]; s.kz], v: vec![vec![[0.0; 3]; nv]; s.kv], a: vec![0.0; s.na()] }
        }
        /// real inner product over all parameters
        pub fn dot(&self, o: &Theta) -> f64 {
            let mut s = 0.0;
            for (a, b) in self.z.iter().zip(&o.z) {
                s += a.iter().zip(b).map(|(x, y)| x.re * y.re + x.im * y.im).sum::<f64>();
            }
            for (a, b) in self.v.iter().zip(&o.v) {
                s += a.iter().zip(b).map(|(x, y)| x[0] * y[0] + x[1] * y[1] + x[2] * y[2]).sum::<f64>();
            }
            s + self.a.iter().zip(&o.a).map(|(x, y)| x * y).sum::<f64>()
        }
        /// self + t·d
        pub fn add(&self, t: f64, d: &Theta) -> Theta {
            let mut o = self.clone();
            for (a, b) in o.z.iter_mut().zip(&d.z) {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += t * y;
                }
            }
            for (a, b) in o.v.iter_mut().zip(&d.v) {
                for (x, y) in a.iter_mut().zip(b) {
                    for c in 0..3 {
                        x[c] += t * y[c];
                    }
                }
            }
            for (x, y) in o.a.iter_mut().zip(&d.a) {
                *x += t * y;
            }
            o
        }
        /// pack into the GPU stack layout (host chunks)
        pub fn pack(&self, s: &VelSpec) -> Vec<Vec<C32>> {
            let nv = s.nv();
            let mut out: Vec<Vec<C32>> = self.z.iter().map(|z| z.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect()).collect();
            for v in &self.v {
                let mut f = vec![0f32; 2 * (3 * nv).div_ceil(2)];
                for r in 0..nv {
                    for c in 0..3 {
                        f[c * nv + r] = v[r][c] as f32;
                    }
                }
                out.push(f.chunks(2).map(|p| C32::new(p[0], p[1])).collect());
            }
            let mut f = vec![0f32; 2 * s.na().div_ceil(2).max(1)];
            for (i, &x) in self.a.iter().enumerate() {
                f[i] = x as f32;
            }
            out.push(f.chunks(2).map(|p| C32::new(p[0], p[1])).collect());
            out
        }
        pub fn unpack(s: &VelSpec, h: &[Vec<C32>]) -> Self {
            let nv = s.nv();
            let mut t = Theta::zeros(s);
            for k in 0..s.kz {
                t.z[k] = h[k].iter().map(|c| C::new(c.re as f64, c.im as f64)).collect();
            }
            for k in 0..s.kv {
                let f: Vec<f32> = h[s.kz + k].iter().flat_map(|c| [c.re, c.im]).collect();
                for r in 0..nv {
                    t.v[k][r] = [f[r] as f64, f[nv + r] as f64, f[2 * nv + r] as f64];
                }
            }
            let f: Vec<f32> = h[s.kz + s.kv].iter().flat_map(|c| [c.re, c.im]).collect();
            for i in 0..s.na() {
                t.a[i] = f[i] as f64;
            }
            t
        }
    }

    pub struct Model {
        pub s: VelSpec,
        wz: Vec<f64>,
        wv: Vec<f64>,
        ex: Vec<[i32; 3]>,
    }

    impl Model {
        pub fn new(s: VelSpec) -> Self {
            let (wz, wv) = (s.wz(), s.wv());
            let ex = s.poly.map(poly_exps).unwrap_or_default();
            Self { s, wz, wv, ex }
        }

        fn phi(&self, a: &[f64], e: usize, r: usize) -> f64 {
            if e == 0 {
                return 0.0;
            }
            let np = self.ex.len();
            (0..np).map(|p| a[(e - 1) * np + p] * poly_at(&self.ex[p], r, self.s.dims)).sum()
        }

        fn zf(&self, th: &Theta, f: usize, r: usize) -> C {
            (0..self.s.kz).map(|k| th.z[k][r] * self.wz[f * self.s.kz + k]).sum()
        }

        fn vf(&self, th: &Theta, f: usize, r: usize) -> [f64; 3] {
            let mut v = [0.0; 3];
            for k in 0..self.s.kv {
                let w = self.wv[f * self.s.kv + k];
                for c in 0..3 {
                    v[c] += w * th.v[k][r][c];
                }
            }
            v
        }

        fn phase(&self, th: &Theta, e: usize, f: usize, r: usize) -> f64 {
            let v = self.vf(th, f, r);
            let k = self.s.kenc[e];
            self.phi(&th.a, e, r) - (k[0] * v[0] + k[1] * v[1] + k[2] * v[2])
        }

        pub fn eval(&self, th: &Theta, e: usize, f: usize) -> Vec<C> {
            (0..self.s.nv()).map(|r| self.zf(th, f, r) * C::from_polar(1.0, self.phase(th, e, f, r))).collect()
        }

        /// grad += Jᴴ g for (e, f) (df = Re⟨g, dx⟩)
        pub fn vjp(&self, th: &Theta, e: usize, f: usize, g: &[C], grad: &mut Theta) {
            let np = self.ex.len();
            for r in 0..self.s.nv() {
                let u = C::from_polar(1.0, self.phase(th, e, f, r));
                let x = self.zf(th, f, r) * u;
                let t = u.conj() * g[r];
                let im = (g[r].conj() * x).im;
                for k in 0..self.s.kz {
                    grad.z[k][r] += t * self.wz[f * self.s.kz + k];
                }
                let ke = self.s.kenc[e];
                for k in 0..self.s.kv {
                    let w = self.wv[f * self.s.kv + k];
                    for c in 0..3 {
                        grad.v[k][r][c] += w * ke[c] * im;
                    }
                }
                if e > 0 {
                    for p in 0..np {
                        grad.a[(e - 1) * np + p] -= poly_at(&self.ex[p], r, self.s.dims) * im;
                    }
                }
            }
        }

        pub fn jvp(&self, th: &Theta, d: &Theta, e: usize, f: usize) -> Vec<C> {
            (0..self.s.nv())
                .map(|r| {
                    let u = C::from_polar(1.0, self.phase(th, e, f, r));
                    let x = self.zf(th, f, r) * u;
                    let dz = self.zf(d, f, r);
                    let dv = self.vf(d, f, r);
                    let k = self.s.kenc[e];
                    let dph = self.phi(&d.a, e, r) - (k[0] * dv[0] + k[1] * dv[1] + k[2] * dv[2]);
                    u * dz + x * C::new(0.0, dph)
                })
                .collect()
        }
    }
}

// ─── GPU ──────────────────────────────────────────────────────────────────────────────────────────────────────────

pub struct VelModel {
    pub s: VelSpec,
    pub wz: Vec<f64>,
    pub wv: Vec<f64>,
    wz_d: CudaSlice<f32>,
    wv_d: CudaSlice<f32>,
    ke_d: CudaSlice<f32>,
    ex_d: CudaSlice<i32>,
    f_eval: CudaFunction,
    f_vjp: CudaFunction,
    f_jvp: CudaFunction,
    f_poly: CudaFunction,
    f_sigma: CudaFunction,
    f_mz: CudaFunction,
    f_mv: CudaFunction,
    f_anchor: CudaFunction,
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

pub(crate) fn raw<T>(s: &CudaSlice<T>, st: &Arc<CudaStream>) -> u64 {
    s.device_ptr(st).0
}

const NB: usize = 256;

impl VelModel {
    pub fn new(dev: &Device, st: &Arc<CudaStream>, s: VelSpec) -> Result<Self, Error> {
        assert!(s.kv <= 64, "K_v ≤ 64");
        let m = dev.load_module("recon_vmodel", include_str!("vmodel.cu"), &CompileSettings::default())?;
        let (wz, wv) = (s.wz(), s.wv());
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let ke: Vec<f32> = s.kenc.iter().flat_map(|k| k.map(|x| x as f32)).collect();
        let mut ex: Vec<i32> = s.poly.map(poly_exps).unwrap_or_default().iter().flatten().copied().collect();
        if ex.is_empty() {
            ex.push(0);
        }
        Ok(Self {
            wz_d: st.clone_htod(&f32s(&wz))?,
            wv_d: st.clone_htod(&f32s(&wv))?,
            ke_d: st.clone_htod(&ke)?,
            ex_d: st.clone_htod(&ex)?,
            f_eval: m.load_function("vm_eval")?,
            f_vjp: m.load_function("vm_vjp")?,
            f_jvp: m.load_function("vm_jvp")?,
            f_poly: m.load_function("vm_poly_reduce")?,
            f_sigma: m.load_function("vm_sigma")?,
            f_mz: m.load_function("vm_metric_z")?,
            f_mv: m.load_function("vm_metric_v")?,
            f_anchor: m.load_function("vm_anchor")?,
            wz,
            wv,
            s,
        })
    }

    pub fn lens(&self) -> Vec<usize> {
        self.s.lens()
    }

    fn ptrs(&self, st: &Arc<CudaStream>, th: &Stack) -> Result<(CudaSlice<u64>, CudaSlice<u64>, u64), Error> {
        let z: Vec<u64> = (0..self.s.kz).map(|k| raw(&th.v[self.s.chunk_z(k)], st)).collect();
        let v: Vec<u64> = (0..self.s.kv).map(|k| raw(&th.v[self.s.chunk_v(k)], st)).collect();
        Ok((st.clone_htod(&z)?, st.clone_htod(&v)?, raw(&th.v[self.s.chunk_a()], st)))
    }

    fn dims_i(&self) -> (i32, i32, i32, i32) {
        let [nx, ny, nz] = self.s.dims;
        (nx as i32, ny as i32, nz as i32, self.s.np() as i32)
    }

    /// x = x_{e,f}(θ)
    pub fn eval(&self, st: &Arc<CudaStream>, th: &Stack, e: usize, f: usize, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (zp, vp, ap) = self.ptrs(st, th)?;
        let (nx, ny, nz, np) = self.dims_i();
        let (kz, kv, ei) = (self.s.kz as i32, self.s.kv as i32, e as i32);
        let wz = self.wz_d.slice(f * self.s.kz..(f + 1) * self.s.kz);
        let wv = self.wv_d.slice(f * self.s.kv..(f + 1) * self.s.kv);
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        unsafe {
            st.launch_builder(&self.f_eval).arg(&zp).arg(&wz).arg(&kz).arg(&vp).arg(&wv).arg(&kv).arg(&ap).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(x).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    /// grad += scale · Jᴴ g for (e, f); `sacc` (per-encode residual-phase accumulator, nv f32) += Im(conj(g) x)
    #[allow(clippy::too_many_arguments)]
    pub fn vjp(&self, st: &Arc<CudaStream>, th: &Stack, e: usize, f: usize, g: &CudaSlice<C32>, scale: f32, grad: &mut Stack, sacc: Option<&mut CudaSlice<f32>>) -> Result<(), Error> {
        let (zp, vp, ap) = self.ptrs(st, th)?;
        let gz: Vec<u64> = (0..self.s.kz).map(|k| raw(&grad.v[self.s.chunk_z(k)], st)).collect();
        let gv: Vec<u64> = (0..self.s.kv).map(|k| raw(&grad.v[self.s.chunk_v(k)], st)).collect();
        let (gz, gv) = (st.clone_htod(&gz)?, st.clone_htod(&gv)?);
        let sp: u64 = sacc.map(|s| raw(s, st)).unwrap_or(0);
        let (nx, ny, nz, np) = self.dims_i();
        let (kz, kv, ei) = (self.s.kz as i32, self.s.kv as i32, e as i32);
        let wz = self.wz_d.slice(f * self.s.kz..(f + 1) * self.s.kz);
        let wv = self.wv_d.slice(f * self.s.kv..(f + 1) * self.s.kv);
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        unsafe {
            st.launch_builder(&self.f_vjp).arg(&zp).arg(&wz).arg(&kz).arg(&vp).arg(&wv).arg(&kv).arg(&ap).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(g).arg(&scale).arg(&gz).arg(&gv).arg(&sp).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    /// dx = J d at θ for (e, f)
    pub fn jvp(&self, st: &Arc<CudaStream>, th: &Stack, d: &Stack, e: usize, f: usize, dx: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (zp, vp, ap) = self.ptrs(st, th)?;
        let (dzp, dvp, dap) = self.ptrs(st, d)?;
        let (nx, ny, nz, np) = self.dims_i();
        let (kz, kv, ei) = (self.s.kz as i32, self.s.kv as i32, e as i32);
        let wz = self.wz_d.slice(f * self.s.kz..(f + 1) * self.s.kz);
        let wv = self.wv_d.slice(f * self.s.kv..(f + 1) * self.s.kv);
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        unsafe {
            st.launch_builder(&self.f_jvp).arg(&zp).arg(&wz).arg(&kz).arg(&vp).arg(&wv).arg(&kv).arg(&ap).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(&dzp).arg(&dvp).arg(&dap).arg(dx).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    /// Σ_r P_p(r)^pw s[r] for every residual-phase polynomial p
    pub fn poly_reduce(&self, st: &Arc<CudaStream>, s: &CudaSlice<f32>, pw: i32) -> Result<Vec<f64>, Error> {
        let np = self.s.np();
        if np == 0 {
            return Ok(vec![]);
        }
        let mut part = st.alloc_zeros::<f64>(np * NB)?;
        let (nx, ny, nz, npi) = self.dims_i();
        let c = LaunchConfig { grid_dim: (NB as u32, np as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            st.launch_builder(&self.f_poly).arg(s).arg(&self.ex_d).arg(&npi).arg(&pw).arg(&nx).arg(&ny).arg(&nz).arg(&mut part).launch(c)?;
        }
        let h = st.clone_dtoh(&part)?;
        Ok((0..np).map(|p| h[p * NB..(p + 1) * NB].iter().sum()).collect())
    }

    /// Gauss–Newton weights: sig[k][r] = Σ_f Wv[f,k] |z_f(r)|², zz[r] = Σ_f |z_f(r)|²
    pub fn sigma(&self, st: &Arc<CudaStream>, th: &Stack, sig: &mut CudaSlice<f32>, zz: &mut CudaSlice<f32>) -> Result<(), Error> {
        let (zp, _, _) = self.ptrs(st, th)?;
        let (kz, kv, nf, nv) = (self.s.kz as i32, self.s.kv as i32, self.s.nf as i32, self.s.nv() as i64);
        unsafe {
            st.launch_builder(&self.f_sigma).arg(&zp).arg(&self.wz_d).arg(&kz).arg(&self.wv_d).arg(&kv).arg(&nf).arg(&nv).arg(sig).arg(zz).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    /// z-chunk metric: out = r / (wsum·c + damp) (inverse: ·)
    #[allow(clippy::too_many_arguments)]
    pub fn metric_z(&self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, out: &mut CudaSlice<C32>, c: Option<&CudaSlice<f32>>, wsum: f32, damp: f32, inverse: bool) -> Result<(), Error> {
        let cp = c.map(|c| raw(c, st)).unwrap_or(0);
        let (inv, nv) = (inverse as i32, self.s.nv() as i64);
        unsafe {
            st.launch_builder(&self.f_mz).arg(r).arg(out).arg(&cp).arg(&wsum).arg(&damp).arg(&inv).arg(&nv).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    /// static-tissue anchor on velocity chunk `v`: grad += λ w v; returns ½ λ Σ w |v|²
    pub fn anchor(&self, st: &Arc<CudaStream>, v: &CudaSlice<C32>, w: &CudaSlice<f32>, lam: f32, grad: &mut CudaSlice<C32>) -> Result<f64, Error> {
        let mut part = st.alloc_zeros::<f64>(NB)?;
        let nv = self.s.nv() as i64;
        let c = LaunchConfig { grid_dim: (NB as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            st.launch_builder(&self.f_anchor).arg(v).arg(w).arg(&lam).arg(grad).arg(&nv).arg(&mut part).launch(c)?;
        }
        Ok(st.clone_dtoh(&part)?.iter().sum())
    }

    /// v-chunk metric: out = M⁻¹ r / (c·sig + damp) (inverse: (c·sig + damp) M r); `sig` = row k of [`sigma`]
    #[allow(clippy::too_many_arguments)]
    pub fn metric_v(&self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, out: &mut CudaSlice<C32>, c: Option<&CudaSlice<f32>>, sig: u64, m: &CudaSlice<f32>, minv: &CudaSlice<f32>, damp: f32, inverse: bool, anchor: Option<(&CudaSlice<f32>, f32)>) -> Result<(), Error> {
        let cp = c.map(|c| raw(c, st)).unwrap_or(0);
        let (inv, nv) = (inverse as i32, self.s.nv() as i64);
        let (wp, aw) = anchor.map(|(w, a)| (raw(w, st), a)).unwrap_or((0, 0.0));
        unsafe {
            st.launch_builder(&self.f_mv).arg(r).arg(out).arg(&cp).arg(&sig).arg(m).arg(minv).arg(&damp).arg(&inv).arg(&nv).arg(&wp).arg(&aw).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64 as C;

    use super::cpu::{Model, Theta};
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn n(&mut self) -> f64 {
            2.0 * self.u() - 1.0
        }
    }

    fn spec() -> VelSpec {
        // dual-VENC-like encodes (VENC 1.0 / 2.5 m/s): k = π / VENC along each axis
        let (lo, hi) = (std::f64::consts::PI / 1.0, std::f64::consts::PI / 2.5);
        let mut kenc = vec![[0.0; 3]];
        for a in 0..3 {
            for k in [lo, hi] {
                let mut v = [0.0; 3];
                v[a] = k;
                kenc.push(v);
            }
        }
        VelSpec { dims: [7, 5, 3], kenc, nf: 6, kz: 3, deg_z: 1, kv: 4, deg_v: 3, poly: Some(2) }
    }

    fn random(s: &VelSpec, rng: &mut Rng, scale: f64) -> Theta {
        let mut t = Theta::zeros(s);
        for z in t.z.iter_mut() {
            for x in z.iter_mut() {
                *x = C::new(1.0 + 0.5 * rng.n(), 0.5 * rng.n()) * scale;
            }
        }
        for v in t.v.iter_mut() {
            for x in v.iter_mut() {
                *x = [0, 1, 2].map(|_| 0.8 * rng.n() * scale);
            }
        }
        for a in t.a.iter_mut() {
            *a = 0.3 * rng.n() * scale;
        }
        t
    }

    #[test]
    fn bases() {
        // partition of unity, identity for K = F (degree 0 and 1), polynomial count
        for (k, f, p) in [(5, 12, 3), (4, 20, 1), (7, 7, 2), (6, 6, 0), (6, 6, 1)] {
            let w = bspline_weights(k, f, p);
            for ff in 0..f {
                assert!((w[ff * k..(ff + 1) * k].iter().sum::<f64>() - 1.0).abs() < 1e-12);
            }
            if k == f && p <= 1 {
                for a in 0..k {
                    for b in 0..k {
                        assert!((w[a * k + b] - (a == b) as u8 as f64).abs() < 1e-12);
                    }
                }
            }
        }
        assert_eq!(poly_exps(0).len(), 1);
        assert_eq!(poly_exps(1).len(), 4);
        assert_eq!(poly_exps(2).len(), 10);
        assert_eq!(poly_exps(3).len(), 20);
    }

    #[test]
    fn cpu_gradient_matches_finite_differences() {
        let s = spec();
        let m = Model::new(s.clone());
        let mut rng = Rng(0x1234);
        let th = random(&s, &mut rng, 1.0);
        let d = random(&s, &mut rng, 1.0);
        let y: Vec<Vec<Vec<C>>> = (0..s.ne()).map(|_| (0..s.nf).map(|_| (0..s.nv()).map(|_| C::new(rng.n(), rng.n())).collect()).collect()).collect();
        let f = |t: &Theta| -> f64 {
            let mut acc = 0.0;
            for e in 0..s.ne() {
                for ff in 0..s.nf {
                    acc += 0.5 * m.eval(t, e, ff).iter().zip(&y[e][ff]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
                }
            }
            acc
        };
        let mut g = Theta::zeros(&s);
        for e in 0..s.ne() {
            for ff in 0..s.nf {
                let r: Vec<C> = m.eval(&th, e, ff).iter().zip(&y[e][ff]).map(|(a, b)| a - b).collect();
                m.vjp(&th, e, ff, &r, &mut g);
            }
        }
        let h = 1e-5;
        let fd = (f(&th.add(h, &d)) - f(&th.add(-h, &d))) / (2.0 * h);
        let an = g.dot(&d);
        assert!((fd - an).abs() < 1e-7 * an.abs().max(1.0), "directional derivative: fd {fd:.10e} vs vjp {an:.10e}");
        // JVP: (x(θ + h d) − x(θ − h d)) / 2h
        for e in [0, 3, 6] {
            let fdx: Vec<C> = m.eval(&th.add(h, &d), e, 2).iter().zip(m.eval(&th.add(-h, &d), e, 2)).map(|(a, b)| (a - b) / (2.0 * h)).collect();
            let jx = m.jvp(&th, &d, e, 2);
            let err = fdx.iter().zip(&jx).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-7, "jvp e {e}: {err:e}");
        }
        // adjoint consistency ⟨J d, g⟩ = ⟨d, Jᴴ g⟩
        let gx: Vec<C> = (0..s.nv()).map(|_| C::new(rng.n(), rng.n())).collect();
        let mut jt = Theta::zeros(&s);
        m.vjp(&th, 5, 4, &gx, &mut jt);
        let lhs: f64 = m.jvp(&th, &d, 5, 4).iter().zip(&gx).map(|(a, b)| (a.conj() * b).re).sum();
        assert!((lhs - jt.dot(&d)).abs() < 1e-10 * lhs.abs().max(1.0));
    }

    #[test]
    fn gpu_matches_cpu() {
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let s = spec();
        let cm = Model::new(s.clone());
        let gm = VelModel::new(&dev, &st, s.clone()).unwrap();
        let mut rng = Rng(0x9876);
        let th = random(&s, &mut rng, 1.0);
        let d = random(&s, &mut rng, 1.0);
        let up = |t: &Theta| Stack { v: t.pack(&s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
        let (tg, dg) = (up(&th), up(&d));
        let nv = s.nv();
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        let rel = |a: &[C32], b: &[C]| a.iter().zip(b).map(|(p, q)| (C::new(p.re as f64, p.im as f64) - q).norm()).fold(0.0, f64::max) / b.iter().map(|q| q.norm()).fold(0.0, f64::max);
        for (e, f) in [(0, 0), (2, 3), (6, 5)] {
            gm.eval(&st, &tg, e, f, &mut x).unwrap();
            let err = rel(&st.clone_dtoh(&x).unwrap(), &cm.eval(&th, e, f));
            assert!(err < 2e-6, "eval ({e},{f}): {err:e}");
            gm.jvp(&st, &tg, &dg, e, f, &mut x).unwrap();
            let err = rel(&st.clone_dtoh(&x).unwrap(), &cm.jvp(&th, &d, e, f));
            assert!(err < 2e-6, "jvp ({e},{f}): {err:e}");
        }
        // vjp over all (e, f) incl. the residual-phase reduction
        let gx: Vec<Vec<C>> = (0..s.ne() * s.nf).map(|_| (0..nv).map(|_| C::new(rng.n(), rng.n())).collect()).collect();
        let mut gc = Theta::zeros(&s);
        let mut gg = Stack { v: s.lens().iter().map(|&n| st.alloc_zeros::<C32>(n).unwrap()).collect() };
        let mut sacc: Vec<CudaSlice<f32>> = (0..s.ne()).map(|_| st.alloc_zeros::<f32>(nv).unwrap()).collect();
        for e in 0..s.ne() {
            for f in 0..s.nf {
                let g = &gx[e * s.nf + f];
                cm.vjp(&th, e, f, g, &mut gc);
                let gd = st.clone_htod(&g.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
                gm.vjp(&st, &tg, e, f, &gd, 1.0, &mut gg, Some(&mut sacc[e])).unwrap();
            }
        }
        let mut host: Vec<Vec<C32>> = gg.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect();
        // residual-phase gradient: −Σ_r P_p S_e
        let np = s.np();
        let mut af = vec![0f32; 2 * s.na().div_ceil(2)];
        for e in 1..s.ne() {
            let red = gm.poly_reduce(&st, &sacc[e], 1).unwrap();
            for p in 0..np {
                af[(e - 1) * np + p] = -red[p] as f32;
            }
        }
        host[s.chunk_a()] = af.chunks(2).map(|p| C32::new(p[0], p[1])).collect();
        let gt = Theta::unpack(&s, &host);
        let diff = gt.add(-1.0, &gc);
        let err = diff.dot(&diff).sqrt() / gc.dot(&gc).sqrt();
        assert!(err < 2e-6, "vjp: rel error {err:e}");
    }
}
