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

/// k × k identity (row-major)
pub fn identity_weights(k: usize) -> Vec<f64> {
    (0..k * k).map(|i| (i / k == i % k) as u8 as f64).collect()
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
    /// At-knots mode `[kr, mc]` (`VelSpec::at_knots`): the operator frames are the kr × mc knots of a respiratory ×
    /// cardiac basis (knot k = r · mc + c, the layout of `SegNufft::new_temporal2d`), nf = kz = kv = kr · mc, identity
    /// weights (frame f is node f). The operator interpolates the knot images to each sample's time. None: the
    /// per-frame model above (B-spline weights from the nodes to the operator frames).
    pub knots: Option<[usize; 2]>,
    /// Parameter-spline mode (`VelSpec::splines`, research/velocity_model_sr.md §F): the parameters are (m, ψ, v) per
    /// knot of the kr × mc basis (node chunk k = (m_k, ψ_k) packed as one complex, re = magnitude, im = phase), the
    /// operator sees `fop` cardiac frames per respiratory knot ([`crate::vknot::OpFrames`]):
    ///   x_e(r, f) = m · exp(i (ψ + φ_e − k'_e·v)), (m, ψ, v) = Σ_c w_c(f) (m, ψ, v)_{r,c},  k'_e = k_e − mean_e k_e
    /// (centred encodes: the Gauss–Newton blocks of ψ and v decouple; the zero-moment phase is ψ0 = ψ + k̄·v).
    pub op: Option<crate::vknot::OpFrames>,
}

impl VelSpec {
    /// The model at the knots of a 2-D (respiratory kr × cardiac mc) knot operator: z_k, v_k per knot k = r · mc + c,
    /// identity weights (one operator frame per knot)
    pub fn at_knots(dims: [usize; 3], kenc: Vec<[f64; 3]>, kr: usize, mc: usize, poly: Option<usize>) -> Self {
        let k = kr * mc;
        assert!(k >= 1, "at_knots: kr · mc ≥ 1");
        Self { dims, kenc, nf: k, kz: k, deg_z: 1, kv: k, deg_v: 1, poly, knots: Some([kr, mc]), op: None }
    }
    /// The parameter-spline model (§F): (m_k, ψ_k, v_k) per knot k = r · mc + c of the kr × mc basis, the operator frames
    /// of `of`; one set per (encode, respiratory group), t = e · ng + g
    pub fn splines(dims: [usize; 3], kenc: Vec<[f64; 3]>, of: crate::vknot::OpFrames, poly: Option<usize>) -> Self {
        let k = of.k();
        Self { dims, kenc, nf: k, kz: k, deg_z: 1, kv: k, deg_v: 1, poly, knots: Some([of.kr, of.mc]), op: Some(of) }
    }
    /// identity weights (at-knots mode): frame f uses node f only
    pub fn identity(&self) -> bool {
        self.knots.is_some() && self.op.is_none()
    }
    /// parameter-spline mode (§F)
    pub fn spline(&self) -> bool {
        self.op.is_some()
    }
    /// mean encode k̄ (spline mode's decoupling; zero otherwise)
    pub fn kbar(&self) -> [f64; 3] {
        if !self.spline() {
            return [0.0; 3];
        }
        let n = self.kenc.len() as f64;
        [0, 1, 2].map(|a| self.kenc.iter().map(|k| k[a]).sum::<f64>() / n)
    }
    /// the encodes as the model uses them: k_e, or (spline mode) the centred k_e − k̄
    pub fn kenc_model(&self) -> Vec<[f64; 3]> {
        let kb = self.kbar();
        self.kenc.iter().map(|k| [k[0] - kb[0], k[1] - kb[1], k[2] - kb[2]]).collect()
    }
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
        if self.identity() {
            return identity_weights(self.kz);
        }
        bspline_weights(self.kz, self.nf, self.deg_z)
    }
    pub fn wv(&self) -> Vec<f64> {
        if self.identity() {
            return identity_weights(self.kv);
        }
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
        /// spline mode: cardiac parameter weights per operator frame, the centred encodes
        fw: Vec<Vec<(usize, f64)>>,
        kem: Vec<[f64; 3]>,
    }

    impl Model {
        pub fn new(s: VelSpec) -> Self {
            let (wz, wv) = if s.spline() { (vec![], vec![]) } else { (s.wz(), s.wv()) };
            let ex = s.poly.map(poly_exps).unwrap_or_default();
            let fw = s.op.as_ref().map_or(vec![], |o| (0..o.fop).map(|f| o.frame_weights(f)).collect());
            let kem = s.kenc_model();
            Self { s, wz, wv, ex, fw, kem }
        }

        // ── spline mode (§F): z[k] = (m_k, ψ_k) as (re, im), k = r · mc + c; frame (r, f) of operator frame f

        fn sp_state(&self, th: &Theta, r: usize, f: usize, q: usize) -> (f64, f64, [f64; 3]) {
            let mc = self.s.op.as_ref().unwrap().mc;
            let (mut m, mut p, mut v) = (0.0, 0.0, [0.0; 3]);
            for &(c, w) in &self.fw[f] {
                let k = r * mc + c;
                m += w * th.z[k][q].re;
                p += w * th.z[k][q].im;
                for a in 0..3 {
                    v[a] += w * th.v[k][q][a];
                }
            }
            (m, p, v)
        }

        fn sp_phase(&self, th: &Theta, e: usize, p: f64, v: &[f64; 3], q: usize) -> f64 {
            let k = self.kem[e];
            p + self.phi(&th.a, e, q) - (k[0] * v[0] + k[1] * v[1] + k[2] * v[2])
        }

        /// spline mode: x_e at operator frame f of respiratory knot r
        pub fn eval_frame(&self, th: &Theta, e: usize, r: usize, f: usize) -> Vec<C> {
            (0..self.s.nv())
                .map(|q| {
                    let (m, p, v) = self.sp_state(th, r, f, q);
                    C::from_polar(1.0, self.sp_phase(th, e, p, &v, q)) * m
                })
                .collect()
        }

        /// spline mode: grad += Jᴴ g of frame (r, f) (df = Re⟨g, dx⟩)
        pub fn vjp_frame(&self, th: &Theta, e: usize, r: usize, f: usize, g: &[C], grad: &mut Theta) {
            let mc = self.s.op.as_ref().unwrap().mc;
            let np = self.ex.len();
            let ke = self.kem[e];
            for q in 0..self.s.nv() {
                let (m, p, v) = self.sp_state(th, r, f, q);
                let u = C::from_polar(1.0, self.sp_phase(th, e, p, &v, q));
                let x = u * m;
                let re = (u.conj() * g[q]).re;
                let im = (g[q].conj() * x).im;
                for &(c, w) in &self.fw[f] {
                    let k = r * mc + c;
                    grad.z[k][q] += C::new(w * re, -w * im);
                    for a in 0..3 {
                        grad.v[k][q][a] += w * ke[a] * im;
                    }
                }
                if e > 0 {
                    for pp in 0..np {
                        grad.a[(e - 1) * np + pp] -= poly_at(&self.ex[pp], q, self.s.dims) * im;
                    }
                }
            }
        }

        /// spline mode: J d at frame (r, f)
        pub fn jvp_frame(&self, th: &Theta, d: &Theta, e: usize, r: usize, f: usize) -> Vec<C> {
            let ke = self.kem[e];
            (0..self.s.nv())
                .map(|q| {
                    let (m, p, v) = self.sp_state(th, r, f, q);
                    let u = C::from_polar(1.0, self.sp_phase(th, e, p, &v, q));
                    let (dm, dp, dv) = self.sp_state(d, r, f, q);
                    let dph = dp + self.phi(&d.a, e, q) - (ke[0] * dv[0] + ke[1] * dv[1] + ke[2] * dv[2]);
                    u * dm + u * m * C::new(0.0, dph)
                })
                .collect()
        }

        fn phi(&self, a: &[f64], e: usize, r: usize) -> f64 {
            if e == 0 {
                return 0.0;
            }
            let np = self.ex.len();
            (0..np).map(|p| a[(e - 1) * np + p] * poly_at(&self.ex[p], r, self.s.dims)).sum()
        }

        fn zf(&self, th: &Theta, f: usize, r: usize) -> C {
            if self.s.identity() {
                return th.z[f][r];
            }
            (0..self.s.kz).map(|k| th.z[k][r] * self.wz[f * self.s.kz + k]).sum()
        }

        fn vf(&self, th: &Theta, f: usize, r: usize) -> [f64; 3] {
            if self.s.identity() {
                return th.v[f][r];
            }
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
                let ke = self.s.kenc[e];
                if self.s.identity() {
                    grad.z[f][r] += t;
                    for c in 0..3 {
                        grad.v[f][r][c] += ke[c] * im;
                    }
                } else {
                    for k in 0..self.s.kz {
                        grad.z[k][r] += t * self.wz[f * self.s.kz + k];
                    }
                    for k in 0..self.s.kv {
                        let w = self.wv[f * self.s.kv + k];
                        for c in 0..3 {
                            grad.v[k][r][c] += w * ke[c] * im;
                        }
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
    /// the single weight 1 of the at-knots mode (frame f = node f)
    one_d: CudaSlice<f32>,
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
    f_absq: CudaFunction,
    /// spline mode (§F): frame kernels and their tables (`vknot::frame_tables`), the cardiac interpolation filter
    sp: Option<SplineK>,
}

struct SplineK {
    f_eval: CudaFunction,
    f_jvp: CudaFunction,
    f_vjp: CudaFunction,
    f_pre: CudaFunction,
    f_mp: CudaFunction,
    f_cf: CudaFunction,
    f_cfa: CudaFunction,
    fi: CudaSlice<i32>,
    fw: CudaSlice<f32>,
    nn: i32,
    ki: CudaSlice<i32>,
    kw: CudaSlice<f32>,
    nkf: i32,
    prim: CudaSlice<i32>,
    pre: CudaSlice<f32>,
    /// prefilter half-width (taps |n| ≤ T); 0: identity (linear operator frames)
    pre_t: i32,
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

pub(crate) fn raw<T>(s: &CudaSlice<T>, st: &Arc<CudaStream>) -> u64 {
    s.device_ptr(st).0
}

const NB: usize = 256;

/// device pointers of one frame's model state: z nodes, v nodes (pointer arrays) and the residual-phase chunk
struct FramePtrs {
    z: CudaSlice<u64>,
    v: CudaSlice<u64>,
    a: u64,
}

impl VelModel {
    pub fn new(dev: &Device, st: &Arc<CudaStream>, s: VelSpec) -> Result<Self, Error> {
        assert!(s.identity() || s.spline() || s.kv <= 64, "K_v ≤ 64");
        let m = dev.load_module("recon_vmodel", include_str!("vmodel.cu"), &CompileSettings::default())?;
        let (wz, wv) = (s.wz(), s.wv());
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        // the encodes as the model uses them (spline mode: centred)
        let ke: Vec<f32> = s.kenc_model().iter().flat_map(|k| k.map(|x| x as f32)).collect();
        let mut ex: Vec<i32> = s.poly.map(poly_exps).unwrap_or_default().iter().flatten().copied().collect();
        if ex.is_empty() {
            ex.push(0);
        }
        // at-knots mode: the weight matrices are the identity and never read by the kernels (one weight 1 instead)
        let (wzd, wvd) = if s.identity() || s.spline() { (vec![1f32], vec![1f32]) } else { (f32s(&wz), f32s(&wv)) };
        let sp = match &s.op {
            None => None,
            Some(of) => {
                let (fi, fw, nn, ki, kw, nkf, prim) = crate::vknot::frame_tables(of);
                let pre = of.prefilter();
                Some(SplineK {
                    f_eval: m.load_function("vp_eval")?,
                    f_jvp: m.load_function("vp_jvp")?,
                    f_vjp: m.load_function("vp_vjp")?,
                    f_pre: m.load_function("vp_prefilter")?,
                    f_mp: m.load_function("vp_metric_mp")?,
                    f_cf: m.load_function("vp_cframes")?,
                    f_cfa: m.load_function("vp_cframes_adj")?,
                    fi: st.clone_htod(&fi)?,
                    fw: st.clone_htod(&fw)?,
                    nn: nn as i32,
                    ki: st.clone_htod(&ki)?,
                    kw: st.clone_htod(&kw)?,
                    nkf: nkf as i32,
                    prim: st.clone_htod(&prim)?,
                    pre: st.clone_htod(&pre.iter().map(|&x| x as f32).collect::<Vec<f32>>())?,
                    pre_t: if of.odeg == 1 { 0 } else { 12 },
                })
            }
        };
        Ok(Self {
            sp,
            wz_d: st.clone_htod(&wzd)?,
            wv_d: st.clone_htod(&wvd)?,
            one_d: st.clone_htod(&[1f32])?,
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
            f_absq: m.load_function("vm_absq")?,
            wz,
            wv,
            s,
        })
    }

    pub fn lens(&self) -> Vec<usize> {
        self.s.lens()
    }

    // ── spline mode (§F): one respiratory row at a time. `zrow[c]` / `vrow[c]`: device pointers of the (m, ψ) and v
    // chunks of cardiac knot c of the row; `out` / `g`: the row's fop frames (frame-major, nv each).

    fn sp(&self) -> &SplineK {
        self.sp.as_ref().expect("spline mode (VelSpec::splines)")
    }

    fn row_cfg(&self, ny: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: ((self.s.nv() as u32).div_ceil(256).max(1), ny as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    /// the row's fop frames of encode e into `out`
    pub fn eval_row(&self, st: &Arc<CudaStream>, zrow: &[u64], vrow: &[u64], a: u64, e: usize, out: u64) -> Result<(), Error> {
        let k = self.sp();
        let (zp, vp) = (st.clone_htod(zrow)?, st.clone_htod(vrow)?);
        let (nx, ny, nz, np) = self.dims_i();
        let ei = e as i32;
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        let fop = self.s.op.as_ref().unwrap().fop;
        unsafe {
            st.launch_builder(&k.f_eval).arg(&zp).arg(&vp).arg(&k.fi).arg(&k.fw).arg(&k.nn).arg(&a).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(&out).launch(self.row_cfg(fop))?;
        }
        Ok(())
    }

    /// the row's linearisation J d (d row pointers, residual-phase chunk `da`) into `out`
    #[allow(clippy::too_many_arguments)]
    pub fn jvp_row(&self, st: &Arc<CudaStream>, zrow: &[u64], vrow: &[u64], dzrow: &[u64], dvrow: &[u64], a: u64, da: u64, e: usize, out: u64) -> Result<(), Error> {
        let k = self.sp();
        let (zp, vp, dzp, dvp) = (st.clone_htod(zrow)?, st.clone_htod(vrow)?, st.clone_htod(dzrow)?, st.clone_htod(dvrow)?);
        let (nx, ny, nz, np) = self.dims_i();
        let ei = e as i32;
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        let fop = self.s.op.as_ref().unwrap().fop;
        unsafe {
            st.launch_builder(&k.f_jvp).arg(&zp).arg(&vp).arg(&dzp).arg(&dvp).arg(&k.fi).arg(&k.fw).arg(&k.nn).arg(&a).arg(&da).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(&out).launch(self.row_cfg(fop))?;
        }
        Ok(())
    }

    /// (gz, gv) row += scale · Jᴴ g over the row's frames; `sacc` (0: none) += Im(conj(g) x) per voxel
    #[allow(clippy::too_many_arguments)]
    pub fn vjp_row(&self, st: &Arc<CudaStream>, zrow: &[u64], vrow: &[u64], a: u64, e: usize, g: u64, scale: f32, gzrow: &[u64], gvrow: &[u64], sacc: u64) -> Result<(), Error> {
        let k = self.sp();
        let (zp, vp, gzp, gvp) = (st.clone_htod(zrow)?, st.clone_htod(vrow)?, st.clone_htod(gzrow)?, st.clone_htod(gvrow)?);
        let (nx, ny, nz, np) = self.dims_i();
        let ei = e as i32;
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        let mc = self.s.op.as_ref().unwrap().mc;
        unsafe {
            st.launch_builder(&k.f_vjp).arg(&zp).arg(&vp).arg(&k.fi).arg(&k.fw).arg(&k.nn).arg(&k.ki).arg(&k.kw).arg(&k.nkf).arg(&k.prim).arg(&a).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(&g).arg(&scale).arg(&gzp).arg(&gvp).arg(&sacc).launch(self.row_cfg(mc))?;
        }
        Ok(())
    }

    /// the cardiac interpolation filter in place on `rows` rows of fop frames at `x` (its own adjoint; no-op for linear
    /// operator frames)
    pub fn prefilter(&self, st: &Arc<CudaStream>, x: u64, rows: usize) -> Result<(), Error> {
        let k = self.sp();
        if k.pre_t == 0 || rows == 0 {
            return Ok(());
        }
        let f = self.s.op.as_ref().unwrap().fop as i32;
        let nv = self.s.nv() as i64;
        unsafe {
            st.launch_builder(&k.f_pre).arg(&x).arg(&f).arg(&k.pre).arg(&k.pre_t).arg(&nv).launch(self.row_cfg(rows))?;
        }
        Ok(())
    }

    /// complex knot images of one row (`crow[c]`: nv complex per cardiac knot) at the row's fop operator frames
    pub fn cframes_row(&self, st: &Arc<CudaStream>, crow: &[u64], out: u64) -> Result<(), Error> {
        let k = self.sp();
        let cp = st.clone_htod(crow)?;
        let nv = self.s.nv() as i64;
        let fop = self.s.op.as_ref().unwrap().fop;
        unsafe {
            st.launch_builder(&k.f_cf).arg(&cp).arg(&k.fi).arg(&k.fw).arg(&k.nn).arg(&nv).arg(&out).launch(self.row_cfg(fop))?;
        }
        Ok(())
    }

    /// its adjoint, accumulated into the row's knot images: C_c += Σ_f w_fc g_f
    pub fn cframes_adj_row(&self, st: &Arc<CudaStream>, g: u64, crow: &[u64]) -> Result<(), Error> {
        let k = self.sp();
        let cp = st.clone_htod(crow)?;
        let nv = self.s.nv() as i64;
        let mc = self.s.op.as_ref().unwrap().mc;
        unsafe {
            st.launch_builder(&k.f_cfa).arg(&g).arg(&k.ki).arg(&k.kw).arg(&k.nkf).arg(&nv).arg(&cp).launch(self.row_cfg(mc))?;
        }
        Ok(())
    }

    /// (m, ψ) chunk metric: re / (wm · c + dm), im / (we · (c · sig + dp)) (inverse: ·)
    #[allow(clippy::too_many_arguments)]
    pub fn metric_mp(&self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, out: &mut CudaSlice<C32>, c: Option<&CudaSlice<f32>>, sig: u64, wm: f32, dm: f32, we: f32, dp: f32, inverse: bool) -> Result<(), Error> {
        let k = self.sp();
        let cp = c.map(|c| raw(c, st)).unwrap_or(0);
        let (inv, nv) = (inverse as i32, self.s.nv() as i64);
        unsafe {
            st.launch_builder(&k.f_mp).arg(r).arg(out).arg(&cp).arg(&sig).arg(&wm).arg(&dm).arg(&we).arg(&dp).arg(&inv).arg(&nv).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    fn ptrs(&self, st: &Arc<CudaStream>, th: &Stack) -> Result<(CudaSlice<u64>, CudaSlice<u64>, u64), Error> {
        let z: Vec<u64> = (0..self.s.kz).map(|k| raw(&th.v[self.s.chunk_z(k)], st)).collect();
        let v: Vec<u64> = (0..self.s.kv).map(|k| raw(&th.v[self.s.chunk_v(k)], st)).collect();
        Ok((st.clone_htod(&z)?, st.clone_htod(&v)?, raw(&th.v[self.s.chunk_a()], st)))
    }

    /// the model state frame f reads: all nodes, or (at-knots mode) node f only
    fn frame_ptrs(&self, st: &Arc<CudaStream>, th: &Stack, f: usize) -> Result<FramePtrs, Error> {
        if self.s.identity() {
            return self.knot_ptrs(st, raw(&th.v[self.s.chunk_z(f)], st), raw(&th.v[self.s.chunk_v(f)], st), raw(&th.v[self.s.chunk_a()], st));
        }
        let (z, v, a) = self.ptrs(st, th)?;
        Ok(FramePtrs { z, v, a })
    }

    fn knot_ptrs(&self, st: &Arc<CudaStream>, z: u64, v: u64, a: u64) -> Result<FramePtrs, Error> {
        Ok(FramePtrs { z: st.clone_htod(&[z])?, v: st.clone_htod(&[v])?, a })
    }

    /// (wz row, kz, wv row, kv) of frame f as the kernels read them
    fn frame_weights(&self, f: usize) -> (cudarc::driver::CudaView<'_, f32>, i32, cudarc::driver::CudaView<'_, f32>, i32) {
        if self.s.identity() {
            return (self.one_d.slice(0..1), 1, self.one_d.slice(0..1), 1);
        }
        let (kz, kv) = (self.s.kz, self.s.kv);
        (self.wz_d.slice(f * kz..(f + 1) * kz), kz as i32, self.wv_d.slice(f * kv..(f + 1) * kv), kv as i32)
    }

    fn dims_i(&self) -> (i32, i32, i32, i32) {
        let [nx, ny, nz] = self.s.dims;
        (nx as i32, ny as i32, nz as i32, self.s.np() as i32)
    }

    fn launch_eval(&self, st: &Arc<CudaStream>, p: &FramePtrs, e: usize, f: usize, x: u64) -> Result<(), Error> {
        let (nx, ny, nz, np) = self.dims_i();
        let (wz, kz, wv, kv) = self.frame_weights(f);
        let ei = e as i32;
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        unsafe {
            st.launch_builder(&self.f_eval).arg(&p.z).arg(&wz).arg(&kz).arg(&p.v).arg(&wv).arg(&kv).arg(&p.a).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(&x).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_vjp(&self, st: &Arc<CudaStream>, p: &FramePtrs, e: usize, f: usize, g: u64, scale: f32, gz: &CudaSlice<u64>, gv: &CudaSlice<u64>, sacc: u64) -> Result<(), Error> {
        let (nx, ny, nz, np) = self.dims_i();
        let (wz, kz, wv, kv) = self.frame_weights(f);
        let ei = e as i32;
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        unsafe {
            st.launch_builder(&self.f_vjp).arg(&p.z).arg(&wz).arg(&kz).arg(&p.v).arg(&wv).arg(&kv).arg(&p.a).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(&g).arg(&scale).arg(gz).arg(gv).arg(&sacc).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    fn launch_jvp(&self, st: &Arc<CudaStream>, p: &FramePtrs, d: &FramePtrs, e: usize, f: usize, dx: u64) -> Result<(), Error> {
        let (nx, ny, nz, np) = self.dims_i();
        let (wz, kz, wv, kv) = self.frame_weights(f);
        let ei = e as i32;
        let ke = self.ke_d.slice(3 * e..3 * e + 3);
        unsafe {
            st.launch_builder(&self.f_jvp).arg(&p.z).arg(&wz).arg(&kz).arg(&p.v).arg(&wv).arg(&kv).arg(&p.a).arg(&self.ex_d).arg(&np).arg(&ei).arg(&ke).arg(&nx).arg(&ny).arg(&nz).arg(&d.z).arg(&d.v).arg(&d.a).arg(&dx).launch(cfg(self.s.nv()))?;
        }
        Ok(())
    }

    /// x = x_{e,f}(θ)
    pub fn eval(&self, st: &Arc<CudaStream>, th: &Stack, e: usize, f: usize, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let p = self.frame_ptrs(st, th, f)?;
        self.launch_eval(st, &p, e, f, raw(x, st))
    }

    /// grad += scale · Jᴴ g for (e, f); `sacc` (per-encode residual-phase accumulator, nv f32) += Im(conj(g) x)
    #[allow(clippy::too_many_arguments)]
    pub fn vjp(&self, st: &Arc<CudaStream>, th: &Stack, e: usize, f: usize, g: &CudaSlice<C32>, scale: f32, grad: &mut Stack, sacc: Option<&mut CudaSlice<f32>>) -> Result<(), Error> {
        let p = self.frame_ptrs(st, th, f)?;
        let (gz, gv): (Vec<u64>, Vec<u64>) = if self.s.identity() {
            (vec![raw(&grad.v[self.s.chunk_z(f)], st)], vec![raw(&grad.v[self.s.chunk_v(f)], st)])
        } else {
            ((0..self.s.kz).map(|k| raw(&grad.v[self.s.chunk_z(k)], st)).collect(), (0..self.s.kv).map(|k| raw(&grad.v[self.s.chunk_v(k)], st)).collect())
        };
        let (gz, gv) = (st.clone_htod(&gz)?, st.clone_htod(&gv)?);
        let sp: u64 = sacc.map(|s| raw(s, st)).unwrap_or(0);
        self.launch_vjp(st, &p, e, f, raw(g, st), scale, &gz, &gv, sp)
    }

    /// dx = J d at θ for (e, f)
    pub fn jvp(&self, st: &Arc<CudaStream>, th: &Stack, d: &Stack, e: usize, f: usize, dx: &mut CudaSlice<C32>) -> Result<(), Error> {
        let p = self.frame_ptrs(st, th, f)?;
        let dp = self.frame_ptrs(st, d, f)?;
        self.launch_jvp(st, &p, &dp, e, f, raw(dx, st))
    }

    // ── at-knots mode on single node chunks (θ streamed knot by knot from host memory, `vobj` host-resident state).
    // Pointers: z = node chunk (nv complex), v = node chunk ([3][nv] f32 packed in pairs), a = residual-phase chunk;
    // outputs / gradients by raw device pointer (e.g. the knot's part of a set image).

    /// x = x_{e,k} from node chunks (at-knots mode)
    pub fn eval_knot(&self, st: &Arc<CudaStream>, z: u64, v: u64, a: u64, e: usize, x: u64) -> Result<(), Error> {
        assert!(self.s.identity(), "eval_knot: at-knots mode only");
        let p = self.knot_ptrs(st, z, v, a)?;
        self.launch_eval(st, &p, e, 0, x)
    }

    /// (gz, gv) += scale · J_{e,k}ᴴ g (at-knots mode); `sacc` += Im(conj(g) x) (0: none)
    #[allow(clippy::too_many_arguments)]
    pub fn vjp_knot(&self, st: &Arc<CudaStream>, z: u64, v: u64, a: u64, e: usize, g: u64, scale: f32, gz: u64, gv: u64, sacc: u64) -> Result<(), Error> {
        assert!(self.s.identity(), "vjp_knot: at-knots mode only");
        let p = self.knot_ptrs(st, z, v, a)?;
        let (gzp, gvp) = (st.clone_htod(&[gz])?, st.clone_htod(&[gv])?);
        self.launch_vjp(st, &p, e, 0, g, scale, &gzp, &gvp, sacc)
    }

    /// dx = J_{e,k} (dz, dv, da) at (z, v, a) (at-knots mode)
    #[allow(clippy::too_many_arguments)]
    pub fn jvp_knot(&self, st: &Arc<CudaStream>, z: u64, v: u64, a: u64, dz: u64, dv: u64, da: u64, e: usize, dx: u64) -> Result<(), Error> {
        assert!(self.s.identity(), "jvp_knot: at-knots mode only");
        let p = self.knot_ptrs(st, z, v, a)?;
        let d = self.knot_ptrs(st, dz, dv, da)?;
        self.launch_jvp(st, &p, &d, e, 0, dx)
    }

    /// out = w |z|² (out ≠ 0), acc += w |z|² (acc ≠ 0), per voxel of a node chunk
    pub fn absq(&self, st: &Arc<CudaStream>, z: u64, w: f32, out: u64, acc: u64) -> Result<(), Error> {
        let nv = self.s.nv() as i64;
        unsafe {
            st.launch_builder(&self.f_absq).arg(&z).arg(&nv).arg(&w).arg(&out).arg(&acc).launch(cfg(self.s.nv()))?;
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

    /// Gauss–Newton weights: sig[k][r] = Σ_f Wv[f,k] |z_f(r)|², zz[r] = Σ_f |z_f(r)|² (at-knots mode: sig[k] = |z_k|²)
    pub fn sigma(&self, st: &Arc<CudaStream>, th: &Stack, sig: &mut CudaSlice<f32>, zz: &mut CudaSlice<f32>) -> Result<(), Error> {
        if self.s.identity() {
            st.memset_zeros(zz)?;
            let (nv, zp) = (self.s.nv(), raw(zz, st));
            let sp = raw(sig, st);
            for k in 0..self.s.kz {
                let out = if sig.len() >= self.s.kz * nv { sp + (k * nv * 4) as u64 } else { 0 };
                self.absq(st, raw(&th.v[self.s.chunk_z(k)], st), 1.0, out, zp)?;
            }
            return Ok(());
        }
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
        VelSpec { dims: [7, 5, 3], kenc, nf: 6, kz: 3, deg_z: 1, kv: 4, deg_v: 3, poly: Some(2), knots: None, op: None }
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


    /// at-knots spec: kr 2 × mc 3 knots, 7 dual-VENC-like encodes, residual phase degree 1
    fn knot_spec() -> VelSpec {
        let s = spec();
        VelSpec::at_knots([6, 5, 4], s.kenc, 2, 3, Some(1))
    }

    /// The at-knots forward model is x_{e,k} = z_k e^{i(φ_e − k_e·v_k)} (node k alone), and the gradient of the knot
    /// objective f(θ) = ½ Σ_e ‖A_e [x_{e,0..K−1}(θ)] − y_e‖² through an explicit f64 knot operator (per sample: knot
    /// weights w_k(j) of a respiratory-linear × cardiac-cubic basis times the DFT, the structure of the 2-D knot NUFFT)
    /// matches central finite differences.
    #[test]
    fn knots_cpu_forward_and_objective_gradient() {
        let s = knot_spec();
        let [kr, mc] = s.knots.unwrap();
        let (nv, ne, kk) = (s.nv(), s.ne(), s.kz);
        assert_eq!((s.nf, s.kz, s.kv), (kk, kk, kk));
        let m = Model::new(s.clone());
        let mut rng = Rng(0x4b4e4f54);
        let th = random(&s, &mut rng, 1.0);
        let d = random(&s, &mut rng, 1.0);
        // forward at the knots: node k alone
        let ex = poly_exps(1);
        for e in [0, 2, 5] {
            for k in [0, kk / 2, kk - 1] {
                let x = m.eval(&th, e, k);
                for r in [0, nv / 3, nv - 1] {
                    let phi: f64 = if e == 0 { 0.0 } else { (0..ex.len()).map(|p| th.a[(e - 1) * ex.len() + p] * poly_at(&ex[p], r, s.dims)).sum() };
                    let kv: f64 = (0..3).map(|c| s.kenc[e][c] * th.v[k][r][c]).sum();
                    let want = th.z[k][r] * C::from_polar(1.0, phi - kv);
                    assert!((x[r] - want).norm() < 1e-12, "at-knots forward (e {e}, k {k}, r {r})");
                }
            }
        }
        // explicit knot operator per encode: samples j with k-space position κ_j (rad / voxel) and knot weights
        let nsamp = 90;
        let pos: Vec<[f64; 3]> = (0..nv).map(|r| [(r % 6) as f64, ((r / 6) % 5) as f64, (r / 30) as f64]).collect();
        let ops: Vec<Vec<([f64; 3], Vec<(usize, f64)>)>> = (0..ne)
            .map(|_| {
                (0..nsamp)
                    .map(|_| {
                        let kap = [0, 1, 2].map(|_| 3.0 * rng.n());
                        // respiratory linear hat between knots 0 and 1, periodic cubic cardiac weights
                        let (u, tc) = (0.5 * (rng.n() + 1.0), 0.5 * (rng.n() + 1.0) * mc as f64);
                        let mut w = vec![];
                        for b in 0..mc {
                            let mut c = 0.0;
                            for j in -2i64..=2 {
                                c += bspline(3, tc - b as f64 - (j * mc as i64) as f64);
                            }
                            for (ri, wr) in [(0usize, 1.0 - u), (1usize, u)] {
                                if c * wr != 0.0 {
                                    w.push((ri * mc + b, c * wr));
                                }
                            }
                        }
                        let _ = kr;
                        (kap, w)
                    })
                    .collect()
            })
            .collect();
        let fwd = |e: usize, xs: &[Vec<C>]| -> Vec<C> {
            ops[e].iter().map(|(kap, w)| w.iter().map(|&(k, wk)| wk * (0..nv).map(|r| xs[k][r] * C::from_polar(1.0, -(kap[0] * pos[r][0] + kap[1] * pos[r][1] + kap[2] * pos[r][2]))).sum::<C>()).sum()).collect()
        };
        let adj = |e: usize, y: &[C]| -> Vec<Vec<C>> {
            let mut g = vec![vec![C::new(0.0, 0.0); nv]; kk];
            for (j, (kap, w)) in ops[e].iter().enumerate() {
                for &(k, wk) in w {
                    for r in 0..nv {
                        g[k][r] += wk * y[j] * C::from_polar(1.0, kap[0] * pos[r][0] + kap[1] * pos[r][1] + kap[2] * pos[r][2]);
                    }
                }
            }
            g
        };
        let y: Vec<Vec<C>> = (0..ne).map(|_| (0..nsamp).map(|_| C::new(rng.n(), rng.n())).collect()).collect();
        let images = |t: &Theta, e: usize| -> Vec<Vec<C>> { (0..kk).map(|k| m.eval(t, e, k)).collect() };
        let f = |t: &Theta| -> f64 { (0..ne).map(|e| 0.5 * fwd(e, &images(t, e)).iter().zip(&y[e]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>()).sum() };
        let mut g = Theta::zeros(&s);
        for e in 0..ne {
            let r: Vec<C> = fwd(e, &images(&th, e)).iter().zip(&y[e]).map(|(a, b)| a - b).collect();
            let gx = adj(e, &r);
            for k in 0..kk {
                m.vjp(&th, e, k, &gx[k], &mut g);
            }
        }
        let h = 1e-5;
        let fd = (f(&th.add(h, &d)) - f(&th.add(-h, &d))) / (2.0 * h);
        let an = g.dot(&d);
        eprintln!("at-knots objective: directional derivative fd {fd:.12e} vs analytic {an:.12e} (rel {:.1e})", (fd - an).abs() / an.abs());
        assert!((fd - an).abs() < 1e-7 * an.abs().max(1.0), "at-knots objective gradient: fd {fd:.10e} vs {an:.10e}");
        // JVP and adjoint consistency per (encode, knot)
        for (e, k) in [(0, 0), (3, kk - 1), (6, 2)] {
            let fdx: Vec<C> = m.eval(&th.add(h, &d), e, k).iter().zip(m.eval(&th.add(-h, &d), e, k)).map(|(a, b)| (a - b) / (2.0 * h)).collect();
            let err = fdx.iter().zip(&m.jvp(&th, &d, e, k)).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-7, "at-knots jvp (e {e}, k {k}): {err:e}");
            let gx: Vec<C> = (0..nv).map(|_| C::new(rng.n(), rng.n())).collect();
            let mut jt = Theta::zeros(&s);
            m.vjp(&th, e, k, &gx, &mut jt);
            let lhs: f64 = m.jvp(&th, &d, e, k).iter().zip(&gx).map(|(a, b)| (a.conj() * b).re).sum();
            assert!((lhs - jt.dot(&d)).abs() < 1e-10 * lhs.abs().max(1.0));
            // the vjp of knot k touches node k only
            for kk2 in (0..kk).filter(|&q| q != k) {
                assert!(jt.z[kk2].iter().all(|c| c.norm() == 0.0) && jt.v[kk2].iter().all(|v| v.iter().all(|&x| x == 0.0)));
            }
        }
    }

    /// GPU at-knots kernels (stack form and the single-chunk streaming form) against the f64 reference
    #[test]
    fn gpu_matches_cpu_at_knots() {
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let s = knot_spec();
        let kk = s.kz;
        let cm = Model::new(s.clone());
        let gm = VelModel::new(&dev, &st, s.clone()).unwrap();
        let mut rng = Rng(0x2468);
        let th = random(&s, &mut rng, 1.0);
        let d = random(&s, &mut rng, 1.0);
        let up = |t: &Theta| Stack { v: t.pack(&s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
        let (tg, dg) = (up(&th), up(&d));
        let nv = s.nv();
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        let rel = |a: &[C32], b: &[C]| a.iter().zip(b).map(|(p, q)| (C::new(p.re as f64, p.im as f64) - q).norm()).fold(0.0, f64::max) / b.iter().map(|q| q.norm()).fold(0.0, f64::max);
        for (e, f) in [(0, 0), (2, 3), (6, kk - 1)] {
            gm.eval(&st, &tg, e, f, &mut x).unwrap();
            assert!(rel(&st.clone_dtoh(&x).unwrap(), &cm.eval(&th, e, f)) < 2e-6, "eval ({e},{f})");
            // single-chunk form
            let (zc, vc, ac) = (raw(&tg.v[s.chunk_z(f)], &st), raw(&tg.v[s.chunk_v(f)], &st), raw(&tg.v[s.chunk_a()], &st));
            gm.eval_knot(&st, zc, vc, ac, e, raw(&x, &st)).unwrap();
            assert!(rel(&st.clone_dtoh(&x).unwrap(), &cm.eval(&th, e, f)) < 2e-6, "eval_knot ({e},{f})");
            gm.jvp(&st, &tg, &dg, e, f, &mut x).unwrap();
            assert!(rel(&st.clone_dtoh(&x).unwrap(), &cm.jvp(&th, &d, e, f)) < 2e-6, "jvp ({e},{f})");
            let (dz, dv, da) = (raw(&dg.v[s.chunk_z(f)], &st), raw(&dg.v[s.chunk_v(f)], &st), raw(&dg.v[s.chunk_a()], &st));
            gm.jvp_knot(&st, zc, vc, ac, dz, dv, da, e, raw(&x, &st)).unwrap();
            assert!(rel(&st.clone_dtoh(&x).unwrap(), &cm.jvp(&th, &d, e, f)) < 2e-6, "jvp_knot ({e},{f})");
        }
        // vjp over all (e, k): stack form into one gradient, chunk form into another
        let gx: Vec<Vec<C>> = (0..s.ne() * kk).map(|_| (0..nv).map(|_| C::new(rng.n(), rng.n())).collect()).collect();
        let mut gc = Theta::zeros(&s);
        let mut gg = Stack { v: s.lens().iter().map(|&n| st.alloc_zeros::<C32>(n).unwrap()).collect() };
        let mut g2 = Stack { v: s.lens().iter().map(|&n| st.alloc_zeros::<C32>(n).unwrap()).collect() };
        for e in 0..s.ne() {
            for f in 0..kk {
                let g = &gx[e * kk + f];
                cm.vjp(&th, e, f, g, &mut gc);
                let gd = st.clone_htod(&g.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
                gm.vjp(&st, &tg, e, f, &gd, 1.0, &mut gg, None).unwrap();
                let (zc, vc, ac) = (raw(&tg.v[s.chunk_z(f)], &st), raw(&tg.v[s.chunk_v(f)], &st), raw(&tg.v[s.chunk_a()], &st));
                gm.vjp_knot(&st, zc, vc, ac, e, raw(&gd, &st), 1.0, raw(&g2.v[s.chunk_z(f)], &st), raw(&g2.v[s.chunk_v(f)], &st), 0).unwrap();
            }
        }
        let mut gc_noa = gc.clone();
        gc_noa.a.iter_mut().for_each(|a| *a = 0.0);
        for gs in [&gg, &g2] {
            let host: Vec<Vec<C32>> = gs.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect();
            let gt = Theta::unpack(&s, &host);
            let diff = gt.add(-1.0, &gc_noa);
            let err = diff.dot(&diff).sqrt() / gc_noa.dot(&gc_noa).sqrt();
            assert!(err < 2e-6, "at-knots vjp: rel error {err:e}");
        }
        // Gauss–Newton weights: sig[k] = |z_k|², zz = Σ_k |z_k|²
        let mut sig = st.alloc_zeros::<f32>(kk * nv).unwrap();
        let mut zz = st.alloc_zeros::<f32>(nv).unwrap();
        gm.sigma(&st, &tg, &mut sig, &mut zz).unwrap();
        let (sh, zh) = (st.clone_dtoh(&sig).unwrap(), st.clone_dtoh(&zz).unwrap());
        for r in [0, nv / 2, nv - 1] {
            let want: f64 = (0..kk).map(|k| th.z[k][r].norm_sqr()).sum();
            assert!((zh[r] as f64 - want).abs() < 1e-5 * want, "zz");
            assert!((sh[3 * nv + r] as f64 - th.z[3][r].norm_sqr()).abs() < 1e-5 * want, "sig");
        }
    }

    /// spline mode: 3 × 5 knots (cardiac cubic), 12 linear-interpolated... cubic operator frames, degree-1 residual phase
    fn spline_spec() -> VelSpec {
        let s = spec();
        VelSpec::splines([6, 5, 4], s.kenc, crate::vknot::OpFrames::new(3, 5, 3, 12, 3), Some(1))
    }

    /// spline-mode kernels (row evaluation, linearisation, gradient, the cardiac interpolation filter) against the f64
    /// reference; the reference's gradient against central differences of ½ Σ |x − y|² over the frames
    #[test]
    fn gpu_matches_cpu_splines() {
        let s = spline_spec();
        let of = s.op.clone().unwrap();
        let (nv, mc, fop) = (s.nv(), of.mc, of.fop);
        let cm = Model::new(s.clone());
        let mut rng = Rng(0x5911);
        let mut th = random(&s, &mut rng, 1.0);
        let d = random(&s, &mut rng, 1.0);
        // magnitudes around 1, phases within ±1.5 rad (any reals are valid; this keeps the FD step meaningful)
        for z in th.z.iter_mut() {
            for x in z.iter_mut() {
                *x = C::new(1.0 + 0.4 * rng.n(), 1.5 * rng.n());
            }
        }
        // reference gradient vs central differences
        let y: Vec<Vec<C>> = (0..s.ne() * of.kr * fop).map(|_| (0..nv).map(|_| C::new(rng.n(), rng.n())).collect()).collect();
        let f = |t: &Theta| -> f64 {
            let mut acc = 0.0;
            for e in 0..s.ne() {
                for r in 0..of.kr {
                    for ff in 0..fop {
                        acc += 0.5 * cm.eval_frame(t, e, r, ff).iter().zip(&y[(e * of.kr + r) * fop + ff]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>();
                    }
                }
            }
            acc
        };
        let mut g = Theta::zeros(&s);
        for e in 0..s.ne() {
            for r in 0..of.kr {
                for ff in 0..fop {
                    let res: Vec<C> = cm.eval_frame(&th, e, r, ff).iter().zip(&y[(e * of.kr + r) * fop + ff]).map(|(a, b)| a - b).collect();
                    cm.vjp_frame(&th, e, r, ff, &res, &mut g);
                }
            }
        }
        let h = 1e-5;
        let fd = (f(&th.add(h, &d)) - f(&th.add(-h, &d))) / (2.0 * h);
        let an = g.dot(&d);
        assert!((fd - an).abs() < 1e-7 * an.abs().max(1.0), "spline reference gradient: fd {fd:.10e} vs {an:.10e}");
        for (e, r, ff) in [(0, 0, 0), (3, 1, 7), (6, 2, 11)] {
            let fdx: Vec<C> = cm.eval_frame(&th.add(h, &d), e, r, ff).iter().zip(cm.eval_frame(&th.add(-h, &d), e, r, ff)).map(|(a, b)| (a - b) / (2.0 * h)).collect();
            let err = fdx.iter().zip(&cm.jvp_frame(&th, &d, e, r, ff)).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(err < 1e-7, "spline jvp (e {e}, r {r}, f {ff}): {err:e}");
        }
        // GPU
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let gm = VelModel::new(&dev, &st, s.clone()).unwrap();
        let up = |t: &Theta| Stack { v: t.pack(&s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
        let (tg, dg) = (up(&th), up(&d));
        let rowp = |x: &Stack, r: usize| -> (Vec<u64>, Vec<u64>) { ((0..mc).map(|c| raw(&x.v[s.chunk_z(r * mc + c)], &st)).collect(), (0..mc).map(|c| raw(&x.v[s.chunk_v(r * mc + c)], &st)).collect()) };
        let mut out = st.alloc_zeros::<C32>(fop * nv).unwrap();
        let rel = |a: &[C32], b: &[C]| a.iter().zip(b).map(|(p, q)| (C::new(p.re as f64, p.im as f64) - q).norm()).fold(0.0, f64::max) / b.iter().map(|q| q.norm()).fold(0.0, f64::max);
        let (ap, dap) = (raw(&tg.v[s.chunk_a()], &st), raw(&dg.v[s.chunk_a()], &st));
        for (e, r) in [(0, 0), (2, 1), (6, 2)] {
            let (zr, vr) = rowp(&tg, r);
            gm.eval_row(&st, &zr, &vr, ap, e, raw(&out, &st)).unwrap();
            let h_out = st.clone_dtoh(&out).unwrap();
            let want: Vec<C> = (0..fop).flat_map(|ff| cm.eval_frame(&th, e, r, ff)).collect();
            assert!(rel(&h_out, &want) < 3e-6, "eval_row (e {e}, r {r}): {:e}", rel(&h_out, &want));
            let (dzr, dvr) = rowp(&dg, r);
            gm.jvp_row(&st, &zr, &vr, &dzr, &dvr, ap, dap, e, raw(&out, &st)).unwrap();
            let want: Vec<C> = (0..fop).flat_map(|ff| cm.jvp_frame(&th, &d, e, r, ff)).collect();
            let hj = st.clone_dtoh(&out).unwrap();
            assert!(rel(&hj, &want) < 3e-6, "jvp_row (e {e}, r {r}): {:e}", rel(&hj, &want));
        }
        // vjp of whole rows: every (e, r) with random frame cotangents, into one gradient (+ the residual-phase sums)
        let gx: Vec<Vec<C>> = (0..s.ne() * of.kr).map(|_| (0..fop * nv).map(|_| C::new(rng.n(), rng.n())).collect()).collect();
        let mut gc = Theta::zeros(&s);
        let mut gg = Stack { v: s.lens().iter().map(|&n| st.alloc_zeros::<C32>(n).unwrap()).collect() };
        let mut sacc: Vec<CudaSlice<f32>> = (0..s.ne()).map(|_| st.alloc_zeros::<f32>(nv).unwrap()).collect();
        for e in 0..s.ne() {
            for r in 0..of.kr {
                let gv = &gx[e * of.kr + r];
                for ff in 0..fop {
                    cm.vjp_frame(&th, e, r, ff, &gv[ff * nv..(ff + 1) * nv], &mut gc);
                }
                let gd = st.clone_htod(&gv.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
                let (zr, vr) = rowp(&tg, r);
                let (gzr, gvr) = rowp(&gg, r);
                gm.vjp_row(&st, &zr, &vr, ap, e, raw(&gd, &st), 1.0, &gzr, &gvr, raw(&sacc[e], &st)).unwrap();
            }
        }
        let mut host: Vec<Vec<C32>> = gg.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect();
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
        assert!(err < 3e-6, "vjp_row: rel error {err:e}");
        // prefilter: 2 rows of fop frames against the dense circulant
        let pre = of.prefilter();
        let xr: Vec<C> = (0..2 * fop * nv).map(|_| C::new(rng.n(), rng.n())).collect();
        let mut xd = st.clone_htod(&xr.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
        gm.prefilter(&st, raw(&xd, &st), 2).unwrap();
        let hx = st.clone_dtoh(&xd).unwrap();
        let mut want = vec![C::new(0.0, 0.0); 2 * fop * nv];
        for row in 0..2 {
            for ff in 0..fop {
                for q in 0..nv {
                    want[(row * fop + ff) * nv + q] = (0..fop).map(|n| xr[(row * fop + (ff + fop - n) % fop) * nv + q] * pre[n]).sum();
                }
            }
        }
        assert!(rel(&hx, &want) < 3e-6, "prefilter: {:e}", rel(&hx, &want));
        let _ = &mut xd;
        // the truncated filter (|n| ≤ 12 taps) at 40 frames against the dense circulant: ≤ 1e-6
        let s40 = VelSpec::splines([6, 5, 4], s.kenc.clone(), crate::vknot::OpFrames::new(3, 5, 3, 40, 3), Some(1));
        let gm40 = VelModel::new(&dev, &st, s40.clone()).unwrap();
        let p40 = s40.op.as_ref().unwrap().prefilter();
        let xr: Vec<C> = (0..40 * nv).map(|_| C::new(rng.n(), rng.n())).collect();
        let xd = st.clone_htod(&xr.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
        gm40.prefilter(&st, raw(&xd, &st), 1).unwrap();
        let hx = st.clone_dtoh(&xd).unwrap();
        let want: Vec<C> = (0..40 * nv).map(|i| { let (ff, q) = (i / nv, i % nv); (0..40).map(|n| xr[((ff + 40 - n) % 40) * nv + q] * p40[n]).sum() }).collect();
        assert!(rel(&hx, &want) < 2e-6, "truncated prefilter: {:e}", rel(&hx, &want));
    }
}
