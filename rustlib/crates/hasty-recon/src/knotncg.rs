//! Knot-space solvers with smoothed classical priors (research/convergence_strategy.md §2.5, §3.3, §8 P1–P2).
//!
//! **Prior-aware metric** ([`PriorMetric`]). On a knot stack x = (x_κ)_κ (K knot images of nv voxels) the data
//! Hessian is ≈ G ⊗ D^{1/2} C D^{1/2} (`knotsolve.rs`: G = temporal Gram, C = the circulant metric's spectrum, D = coil
//! energy). With G = U Λ_G Uᵀ (U = the basis' "temporal frequencies" ω under the actual sample times) and a prior whose
//! curvature is diagonal in (ω, k),
//!   P = D^{-1/2} (Uᵀ ⊗ F)ᴴ diag(1 / (λ_G(ω) c(k) + (σ_w(k) + σ_l(ω)) / D_s)) (Uᵀ ⊗ F) D^{-1/2}
//! with σ_w the wavelet envelope's curvature per spatial frequency (subband-averaged curvature fraction / γ), σ_l the
//! LLR envelope's (discarded energy fraction per ω / γ) and D_s the typical coil energy of the object (the sandwich
//! D^{-1/2} Σ D^{-1/2} ≈ Σ / D_s). The per-set (per-knot) scalars are the diagonal of G, inside Λ_G. With σ = 0 it is
//! exactly G⁻¹ ⊗ C⁻¹ (`knotsolve::KnotPrecond`). Static: built once per encode, no data-dependent update in the loop.
//!
//! **Smoothed priors** (fixed γ, fixed thresholds, fixed LLR block shift):
//! - [`WavHuber`]: Σ_i h_γ(|(W S x)_i|) over the detail coefficients of a zero-padded (S, an isometry) orthonormal
//!   3-D db4 transform W: the smoothing of λ‖W S x‖₁ with θ = λγ. Its gradient is exactly (x − SᴴWᴴ soft_θ(W S x)) / γ
//!   (= (x − prox(x)) / γ, the Moreau form). Along a line the coefficients are affine, so the line search evaluates it in
//!   the coefficient domain without transforms.
//! - [`LlrHuber`]: the Moreau envelope of λ Σ_b ‖C_b‖_* (blocks of a fixed partition, Casorati over the knot axis):
//!   gradient (x − SVT_θ(x)) / γ, value from the singular values ([`crate::llr::LlrStats`]).
//!
//! **Solvers** (one A and one Aᴴ per iteration each; `cb` after every iteration, return false to stop):
//! - [`ncg`]: nonlinear PCG, Polak–Ribière+, Powell restarts, in the metric P; data-exact line search (f is quadratic:
//!   ‖A d‖² from the forward that also updates the residual), wavelet part exact in the coefficient domain, LLR part by
//!   one or two secant evaluations;
//! - [`fista`]: accelerated gradient on the same smoothed objective in the same metric, step 1/L, gradient restart;
//! - [`zmshr1`]: ZMSHR1 in the metric P on the same objective (the RED form of `red.rs` with proximal denoisers);
//! - [`admm`]: scaled ADMM on the exact (unsmoothed) ℓ1 + nuclear-norm objective, splits u_w = W S x (coefficient
//!   domain: exact prox and exact x-subproblem since W S is an isometry) and u_l = x; x-update by `n_inner` CGLS steps
//!   in the metric P with the ridge 2ρ.
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{Device, C32};
use hasty_fft::{Direction, FftPlan, Norm};
use num_complex::Complex64;
use rayon::prelude::*;

use crate::gpu::Kernels;
use crate::llr::{Llr, LlrStats};
use crate::llr_gpu::LlrGpu;
use crate::sepwt::{forward, inverse, Bank};
use crate::setop::LinOp;
use crate::Error;

const SRC: &str = r#"
struct cx { float x, y; };
#define VB 16
// in place: z[a][v] ← Σ_b M[a][b] z[b][v] (real K × K, K ≤ 128)
extern "C" __global__ void knot_mix(cx* __restrict__ z, const float* __restrict__ mm, int k, unsigned long long nv) {
    __shared__ cx t[128 * VB];
    const unsigned long long v0 = (unsigned long long)blockIdx.x * VB;
    for (int q = threadIdx.x; q < k * VB; q += blockDim.x) {
        const int b = q / VB, v = q % VB;
        t[q] = v0 + v < nv ? z[(unsigned long long)b * nv + v0 + v] : cx{0.f, 0.f};
    }
    __syncthreads();
    for (int q = threadIdx.x; q < k * VB; q += blockDim.x) {
        const int a = q / VB, v = q % VB;
        if (v0 + v >= nv) continue;
        float zr = 0.f, zi = 0.f;
        for (int b = 0; b < k; ++b) {
            const float g = mm[a * k + b];
            const cx w = t[b * VB + v];
            zr = fmaf(g, w.x, zr); zi = fmaf(g, w.y, zi);
        }
        z[(unsigned long long)a * nv + v0 + v] = cx{zr, zi};
    }
}
// z[w][v] ×= den^{-1} (inv = 0) or den (inv = 1), den = lamg[w] c[v] + sl[w] + sw[v]
extern "C" __global__ void knot_den(cx* __restrict__ z, const float* __restrict__ lamg, const float* __restrict__ sl, const float* __restrict__ c, const float* __restrict__ sw, int k, unsigned long long nv, int inv) {
    const unsigned long long n = nv * (unsigned long long)k;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        const int w = (int)(i / nv);
        const unsigned long long v = i % nv;
        const float d = lamg[w] * c[v] + sl[w] + sw[v];
        const float s = inv ? d : 1.f / d;
        z[i].x *= s; z[i].y *= s;
    }
}
// z[a][v] ×= d[v] (inv = 0) or 1 / d[v] (inv = 1)
extern "C" __global__ void knot_scale(cx* __restrict__ z, const float* __restrict__ d, int k, unsigned long long nv, int inv) {
    const unsigned long long n = nv * (unsigned long long)k;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
        const float s = inv ? 1.f / d[i % nv] : d[i % nv];
        z[i].x *= s; z[i].y *= s;
    }
}
extern "C" __global__ void cscale(cx* __restrict__ z, float f, unsigned long long n) {
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { z[i].x *= f; z[i].y *= f; }
}
"#;

fn grid(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n.div_ceil(256)).min(65535 * 16) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn re(v: f64) -> Complex64 {
    Complex64::new(v, 0.0)
}

/// P = D^{-1/2}(G ⊗ C + Σ_prior)^{-1}D^{-1/2} on knot stacks [K][nv] (module docs)
pub struct PriorMetric {
    pub k: usize,
    pub nv: usize,
    pub dims: [usize; 3],
    /// eigenvalues of the (ridged) temporal Gram and its eigenvectors U (row-major U[b][ω])
    pub lam_g: Vec<f64>,
    pub u: Vec<f64>,
    /// the data symbol c(k) (FFT order) and the object's coil energy D_s (host copies)
    pub c: Vec<f32>,
    pub d_s: f64,
    /// the prior terms currently in the metric (image-space curvature): per ω and per spatial frequency (or empty)
    pub sl: Vec<f64>,
    pub sw: Vec<f32>,
    /// Some(s): the metric is s⁻¹ I instead (the identity arm; P⁻¹ = s I)
    pub ident: Option<f64>,
    ut_d: CudaSlice<f32>,
    u_d: CudaSlice<f32>,
    lamg_d: CudaSlice<f32>,
    sl_d: CudaSlice<f32>,
    c_d: CudaSlice<f32>,
    sw_d: CudaSlice<f32>,
    dinv_d: CudaSlice<f32>,
    fwd: FftPlan<f32>,
    inv: FftPlan<f32>,
    f_mix: CudaFunction,
    f_den: CudaFunction,
    f_scale: CudaFunction,
    f_cscale: CudaFunction,
}

impl PriorMetric {
    /// `gram` K × K (row-major, the basis Gram under the sample times), ridge `eps` × mean diagonal (as
    /// `knotsolve::gram_inverse`); `lam_eff`, `dinv` the circulant metric's factors ([`crate::precond::CirculantPrecond::factors`])
    pub fn new(dev: &Device, st: &Arc<CudaStream>, k: usize, dims: [usize; 3], gram: &[f64], eps: f64, lam_eff: Vec<f32>, dinv: Vec<f32>) -> Result<Self, Error> {
        assert!(k <= 128 && gram.len() == k * k);
        let nv: usize = dims.iter().product();
        assert_eq!(lam_eff.len(), nv);
        assert_eq!(dinv.len(), nv);
        let md = (0..k).map(|i| gram[i * k + i]).sum::<f64>() / k as f64;
        let m = nalgebra::DMatrix::<f64>::from_fn(k, k, |i, j| 0.5 * (gram[i * k + j] + gram[j * k + i]) + if i == j { eps * md } else { 0.0 });
        let e = m.symmetric_eigen();
        let lam_g: Vec<f64> = e.eigenvalues.iter().copied().collect();
        let u: Vec<f64> = (0..k * k).map(|q| e.eigenvectors[(q / k, q % k)]).collect();
        // mixing to ω: ẑ_ω = Σ_b U[b][ω] z_b (matrix Uᵀ); back: z_b = Σ_ω U[b][ω] ẑ_ω (matrix U)
        let ut: Vec<f32> = (0..k * k).map(|q| u[(q % k) * k + q / k] as f32).collect();
        let uu: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        // D_s: median coil energy over the object (D > 5 % of its max)
        let dv: Vec<f64> = dinv.iter().map(|&v| 1.0 / (v as f64 * v as f64).max(1e-30)).collect();
        let dmax = dv.iter().cloned().fold(0.0, f64::max);
        let mut obj: Vec<f64> = dv.iter().copied().filter(|&v| v > 0.05 * dmax).collect();
        obj.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d_s = obj.get(obj.len() / 2).copied().unwrap_or(1.0);
        let sdims = [dims[2], dims[1], dims[0]];
        let m = dev.load_module("knotncg", SRC, &hasty_cuda::CompileSettings::default())?;
        Ok(Self {
            k,
            nv,
            dims,
            ut_d: st.clone_htod(&ut)?,
            u_d: st.clone_htod(&uu)?,
            lamg_d: st.clone_htod(&lam_g.iter().map(|&v| v as f32).collect::<Vec<_>>())?,
            sl_d: st.alloc_zeros(k)?,
            c_d: st.clone_htod(&lam_eff)?,
            sw_d: st.alloc_zeros(nv)?,
            dinv_d: st.clone_htod(&dinv)?,
            lam_g,
            u,
            c: lam_eff,
            d_s,
            sl: vec![0.0; k],
            sw: vec![],
            ident: None,
            fwd: FftPlan::<f32>::new(dev, &sdims, Direction::Forward, Norm::None)?,
            inv: FftPlan::<f32>::new(dev, &sdims, Direction::Inverse, Norm::Inverse)?,
            f_mix: m.load_function("knot_mix")?,
            f_den: m.load_function("knot_den")?,
            f_scale: m.load_function("knot_scale")?,
            f_cscale: m.load_function("cscale")?,
        })
    }

    /// device bytes it holds (ledger row): 3 nv-arrays of f32 + the FFT plans' twiddles (small)
    pub fn bytes(k: usize, nv: usize) -> usize {
        3 * nv * 4 + 4 * k * k * 4 + (1 << 24)
    }

    /// prior curvature in the metric (image space): `sl` per temporal frequency ω (K values, a ridge is a constant
    /// vector), `sw` per spatial frequency (nv, FFT order) or None; both divided by D_s inside
    pub fn set_prior(&mut self, st: &Arc<CudaStream>, sl: &[f64], sw: Option<&[f32]>) -> Result<(), Error> {
        assert_eq!(sl.len(), self.k);
        let s = 1.0 / self.d_s;
        st.memcpy_htod(&sl.iter().map(|&v| (v * s) as f32).collect::<Vec<_>>(), &mut self.sl_d)?;
        match sw {
            Some(w) => {
                assert_eq!(w.len(), self.nv);
                st.memcpy_htod(&w.iter().map(|&v| (v as f64 * s) as f32).collect::<Vec<_>>(), &mut self.sw_d)?;
                self.sw = w.to_vec();
            }
            None => {
                st.memset_zeros(&mut self.sw_d)?;
                self.sw = vec![];
            }
        }
        self.sl = sl.to_vec();
        Ok(())
    }

    /// the bulk data curvature: D_s · mean λ_G · median c(k) (the flat high-k part of the data Hessian, where the
    /// prior acts; γ and the prior weights are set relative to it)
    pub fn bulk(&self) -> f64 {
        let mut c: Vec<f32> = self.c.clone();
        let mid = c.len() / 2;
        let cm = *c.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1 as f64;
        self.d_s * self.lam_g.iter().sum::<f64>() / self.k as f64 * cm
    }

    fn run(&self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        let (k, nv) = (self.k, self.nv);
        assert_eq!(r.len(), k * nv);
        st.memcpy_dtod(r, z)?;
        if let Some(sc) = self.ident {
            let f = (if inverse { sc } else { 1.0 / sc }) as f32;
            let n = z.len() as u64;
            let cfg = grid(z.len());
            unsafe { st.launch_builder(&self.f_cscale).arg(&mut *z).arg(&f).arg(&n).launch(cfg)? };
            return Ok(());
        }
        let (ki, nvu, iv) = (k as i32, nv as u64, inverse as i32);
        let cfg = grid(k * nv);
        // D^{-1/2} (metric) or D^{1/2} (inverse)
        unsafe { st.launch_builder(&self.f_scale).arg(&mut *z).arg(&self.dinv_d).arg(&ki).arg(&nvu).arg(&iv).launch(cfg)? };
        for q in 0..k {
            self.fwd.execute_inplace(st, &mut z.slice_mut(q * nv..(q + 1) * nv))?;
        }
        let mcfg = LaunchConfig { grid_dim: (nv.div_ceil(16) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            st.launch_builder(&self.f_mix).arg(&mut *z).arg(&self.ut_d).arg(&ki).arg(&nvu).launch(mcfg)?;
            st.launch_builder(&self.f_den).arg(&mut *z).arg(&self.lamg_d).arg(&self.sl_d).arg(&self.c_d).arg(&self.sw_d).arg(&ki).arg(&nvu).arg(&iv).launch(cfg)?;
            st.launch_builder(&self.f_mix).arg(&mut *z).arg(&self.u_d).arg(&ki).arg(&nvu).launch(mcfg)?;
        }
        for q in 0..k {
            self.inv.execute_inplace(st, &mut z.slice_mut(q * nv..(q + 1) * nv))?;
        }
        unsafe { st.launch_builder(&self.f_scale).arg(&mut *z).arg(&self.dinv_d).arg(&ki).arg(&nvu).arg(&iv).launch(cfg)? };
        Ok(())
    }

    /// z = P r
    pub fn apply_metric(&self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.run(st, r, z, false)
    }

    /// z = P⁻¹ r
    pub fn apply_inv(&self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.run(st, r, z, true)
    }
}

impl crate::cg::Precond for PriorMetric {
    fn apply(&mut self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.run(st, r, z, false)
    }
}

// ------------------------------------------------------------------------------------------------------------------
// host helpers

const PAR: usize = 1 << 15;

/// Re⟨a, b⟩ (f64)
pub fn hdot(a: &[C32], b: &[C32]) -> f64 {
    a.par_chunks(PAR).zip(b.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
}

/// y ← a x + b y
pub fn haxpby(a: f32, x: &[C32], b: f32, y: &mut [C32]) {
    y.par_chunks_mut(PAR).zip(x.par_chunks(PAR)).for_each(|(yc, xc)| {
        for (u, v) in yc.iter_mut().zip(xc) {
            *u = C32::new(a * v.re + b * u.re, a * v.im + b * u.im);
        }
    });
}

fn faxpby(a: f32, x: &[f32], b: f32, y: &mut [f32]) {
    y.par_chunks_mut(PAR).zip(x.par_chunks(PAR)).for_each(|(yc, xc)| {
        for (u, v) in yc.iter_mut().zip(xc) {
            *u = a * v + b * *u;
        }
    });
}

// ------------------------------------------------------------------------------------------------------------------
// smoothed priors

/// Daubechies-4 low-pass (as `wavelet.rs`)
const DB4: [f64; 8] = [
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
    0.6308807679295904,
    0.7148465705525415,
    0.23037781330885523,
];

/// Huber-smoothed analysis ℓ1 of a zero-padded orthonormal db4 transform, per knot image (module docs)
pub struct WavHuber {
    pub dims: [usize; 3],
    /// padded box (each axis the next multiple of 2^levels; zero padding)
    pub p: [usize; 3],
    pub levels: usize,
    /// θ = λγ and γ of the envelope
    pub theta: f64,
    pub gamma: f64,
    bank: Bank,
}

/// h_γ(|c|) with θ = λγ (value) and its gradient factor ψ(c) = c · w
#[inline]
fn huber(cr: f32, ci: f32, th: f32, ig: f32) -> (f32, f32) {
    let m = (cr * cr + ci * ci).sqrt();
    if m <= th {
        (0.5 * m * m * ig, ig)
    } else {
        ((th * m - 0.5 * th * th) * ig, th * ig / m)
    }
}

impl WavHuber {
    pub fn new(dims: [usize; 3], levels: usize, theta: f64, gamma: f64) -> Self {
        let m = 1usize << levels;
        Self { dims, p: dims.map(|n| n.div_ceil(m) * m), levels, theta, gamma, bank: Bank::orthonormal(&DB4) }
    }

    /// coefficients per image (f32 interleaved)
    pub fn np(&self) -> usize {
        self.p[0] * self.p[1] * self.p[2] * 2
    }

    /// c = W S x for the k images of `x` (k × nv) into `out` (k × np)
    pub fn coef(&self, x: &[C32], k: usize, out: &mut Vec<f32>) {
        let (d, p) = (self.dims, self.p);
        let nv = d[0] * d[1] * d[2];
        let np = self.np();
        out.resize(k * np, 0.0);
        for q in 0..k {
            let o = &mut out[q * np..(q + 1) * np];
            let img = &x[q * nv..(q + 1) * nv];
            o.par_chunks_mut(p[0] * 2).enumerate().for_each(|(row, line)| {
                let (y, z) = (row % p[1], row / p[1]);
                line.fill(0.0);
                if y < d[1] && z < d[2] {
                    let src = &img[d[0] * (y + d[1] * z)..d[0] * (y + d[1] * z) + d[0]];
                    for (x, v) in src.iter().enumerate() {
                        line[2 * x] = v.re;
                        line[2 * x + 1] = v.im;
                    }
                }
            });
            let b = &self.bank;
            forward(o, p, 2, self.levels, [b, b, b]);
        }
    }

    /// x = Sᴴ Wᴴ c (consumes `c`: transformed back in place), k images into `out` (k × nv)
    pub fn synth(&self, c: &mut [f32], k: usize, out: &mut [C32]) {
        let (d, p) = (self.dims, self.p);
        let nv = d[0] * d[1] * d[2];
        let np = self.np();
        for q in 0..k {
            let cb = &mut c[q * np..(q + 1) * np];
            let b = &self.bank;
            inverse(cb, p, 2, self.levels, [b, b, b]);
            let cb: &[f32] = cb;
            out[q * nv..(q + 1) * nv].par_chunks_mut(d[0]).enumerate().for_each(|(row, o)| {
                let (y, z) = (row % d[1], row / d[1]);
                let s = &cb[(p[0] * (y + p[1] * z)) * 2..];
                for (x, v) in o.iter_mut().enumerate() {
                    *v = C32::new(s[2 * x], s[2 * x + 1]);
                }
            });
        }
    }

    /// apply `f(plane slice, x0 of each row)` over the detail coefficients: rows of plane (knot, z), detail part of a
    /// row starts at x0 (the approximation block is the first p/2^levels of each axis)
    fn planes(&self) -> (usize, usize, [usize; 3]) {
        let p = self.p;
        (p[0] * p[1] * 2, p[2], p.map(|v| v >> self.levels))
    }

    /// Σ h(c) over the detail coefficients; with `psi`, ψ(c) written there (approximation band: 0)
    pub fn value_psi(&self, c: &[f32], psi: Option<&mut [f32]>) -> f64 {
        let (pl, p2, a) = self.planes();
        let (p0, p1) = (self.p[0], self.p[1]);
        let (th, ig) = (self.theta as f32, 1.0 / self.gamma as f32);
        let row = |z: usize, y: usize| if z < a[2] && y < a[1] { a[0] } else { 0 };
        match psi {
            Some(out) => c
                .par_chunks(pl)
                .zip(out.par_chunks_mut(pl))
                .enumerate()
                .map(|(i, (cp, op))| {
                    let z = i % p2;
                    let mut v = 0.0f64;
                    for y in 0..p1 {
                        let x0 = row(z, y);
                        let (cr, or) = (&cp[y * p0 * 2..(y + 1) * p0 * 2], &mut op[y * p0 * 2..(y + 1) * p0 * 2]);
                        or[..x0 * 2].fill(0.0);
                        for x in x0..p0 {
                            let (h, w) = huber(cr[2 * x], cr[2 * x + 1], th, ig);
                            v += h as f64;
                            or[2 * x] = w * cr[2 * x];
                            or[2 * x + 1] = w * cr[2 * x + 1];
                        }
                    }
                    v
                })
                .sum(),
            None => c
                .par_chunks(pl)
                .enumerate()
                .map(|(i, cp)| {
                    let z = i % p2;
                    let mut v = 0.0f64;
                    for y in 0..p1 {
                        let x0 = row(z, y);
                        let cr = &cp[y * p0 * 2..(y + 1) * p0 * 2];
                        for x in x0..p0 {
                            v += huber(cr[2 * x], cr[2 * x + 1], th, ig).0 as f64;
                        }
                    }
                    v
                })
                .sum(),
        }
    }

    /// along the line c(a) = cx + a cd: (Σ h(c(a)), d/da = Σ Re⟨ψ(c(a)), cd⟩)
    pub fn line(&self, cx: &[f32], cd: &[f32], a: f64) -> (f64, f64) {
        let (pl, p2, aa) = self.planes();
        let (p0, p1) = (self.p[0], self.p[1]);
        let (th, ig, af) = (self.theta as f32, 1.0 / self.gamma as f32, a as f32);
        cx.par_chunks(pl)
            .zip(cd.par_chunks(pl))
            .enumerate()
            .map(|(i, (xp, dp))| {
                let z = i % p2;
                let (mut v, mut g) = (0.0f64, 0.0f64);
                for y in 0..p1 {
                    let x0 = if z < aa[2] && y < aa[1] { aa[0] } else { 0 };
                    let o = y * p0 * 2;
                    for x in x0..p0 {
                        let (dr, di) = (dp[o + 2 * x], dp[o + 2 * x + 1]);
                        let (cr, ci) = (xp[o + 2 * x] + af * dr, xp[o + 2 * x + 1] + af * di);
                        let (h, w) = huber(cr, ci, th, ig);
                        v += h as f64;
                        g += (w * (cr * dr + ci * di)) as f64;
                    }
                }
                (v, g)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// soft threshold of the detail coefficients at `thr` (the exact ℓ1 prox in the coefficient domain)
    pub fn soft(&self, c: &mut [f32], thr: f64) {
        let (pl, p2, a) = self.planes();
        let (p0, p1) = (self.p[0], self.p[1]);
        let t = thr as f32;
        c.par_chunks_mut(pl).enumerate().for_each(|(i, cp)| {
            let z = i % p2;
            for y in 0..p1 {
                let x0 = if z < a[2] && y < a[1] { a[0] } else { 0 };
                let o = y * p0 * 2;
                for x in x0..p0 {
                    let (cr, ci) = (cp[o + 2 * x], cp[o + 2 * x + 1]);
                    let m = (cr * cr + ci * ci).sqrt();
                    let s = if m > t { 1.0 - t / m } else { 0.0 };
                    cp[o + 2 * x] = s * cr;
                    cp[o + 2 * x + 1] = s * ci;
                }
            }
        });
    }

    /// noise level from the finest diagonal subband: median |c| / 1.1774 (complex Gaussian), median over the k images
    pub fn sigma_finest(&self, c: &[f32], k: usize) -> f64 {
        let np = self.np();
        let h = self.p.map(|v| v / 2);
        let mut s: Vec<f64> = (0..k).map(|q| crate::wavelet::median_abs(&c[q * np..(q + 1) * np], self.p, 2, h, h) / 1.1774).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    }

    /// mean curvature factor (1 inside the quadratic zone, ½θ/|c| outside: radial 0, tangential θ/|c|) per detail
    /// subband (level 1 = finest, ε = 1..7), over the k images (strided samples)
    pub fn curvature_fractions(&self, c: &[f32], k: usize) -> Vec<(usize, [usize; 3], f64)> {
        let np = self.np();
        let th = self.theta as f32;
        let p = self.p;
        let mut out = vec![];
        for l in 1..=self.levels {
            for (eps, off, size) in crate::sepwt::subbands(p, l) {
                let n = size[0] * size[1] * size[2];
                let step = n.div_ceil(1 << 18).max(1);
                let (s, cnt): (f64, usize) = (0..k)
                    .into_par_iter()
                    .map(|q| {
                        let cb = &c[q * np..(q + 1) * np];
                        let mut s = 0.0f64;
                        let mut cnt = 0usize;
                        for i in (0..n).step_by(step) {
                            let (x, y, z) = (i % size[0], (i / size[0]) % size[1], i / (size[0] * size[1]));
                            let j = ((off[0] + x) + p[0] * ((off[1] + y) + p[1] * (off[2] + z))) * 2;
                            let m = (cb[j] * cb[j] + cb[j + 1] * cb[j + 1]).sqrt();
                            s += if m <= th { 1.0 } else { 0.5 * (th / m) as f64 };
                            cnt += 1;
                        }
                        (s, cnt)
                    })
                    .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
                out.push((l, eps, s / cnt.max(1) as f64));
            }
        }
        out
    }

    /// the envelope's curvature symbol per spatial frequency (FFT order on `dims`, x fastest): the curvature fraction of
    /// the octave subband holding k, / γ; 0 in the approximation band
    pub fn symbol(&self, fr: &[(usize, [usize; 3], f64)]) -> Vec<f32> {
        let d = self.dims;
        let lev = |i: usize, n: usize| -> usize {
            // octave of the normalised frequency f = min(i, n − i) / n ∈ [0, ½]: level j holds [2^-(j+1), 2^-j)
            let f = i.min(n - i) as f64 / n as f64;
            if f <= 0.0 { usize::MAX } else { (-(f.log2()) - 1.0).floor().max(0.0) as usize + 1 }
        };
        let ig = 1.0 / self.gamma;
        let mut out = vec![0f32; d[0] * d[1] * d[2]];
        out.par_chunks_mut(d[0]).enumerate().for_each(|(row, o)| {
            let (j, k) = (row % d[1], row / d[1]);
            let (ly, lz) = (lev(j, d[1]), lev(k, d[2]));
            for (i, v) in o.iter_mut().enumerate() {
                let l = [lev(i, d[0]), ly, lz];
                let jj = *l.iter().min().unwrap();
                if jj == usize::MAX || jj > self.levels {
                    continue;
                }
                let eps = l.map(|a| (a == jj) as usize);
                if let Some(f) = fr.iter().find(|(lv, e, _)| *lv == jj && *e == eps) {
                    *v = (f.2 * ig) as f32;
                }
            }
        });
        out
    }
}

/// CPU or GPU block-SVT engine of the LLR prior
pub enum LlrBackend {
    Cpu(Llr),
    Gpu(LlrGpu),
}

impl LlrBackend {
    pub fn llr(&mut self) -> &mut Llr {
        match self {
            LlrBackend::Cpu(l) => l,
            LlrBackend::Gpu(g) => &mut g.llr,
        }
    }
    pub fn apply(&mut self, x: &mut [C32], nt: usize) -> Result<(), Error> {
        match self {
            LlrBackend::Cpu(l) => {
                l.apply(x, nt);
                Ok(())
            }
            LlrBackend::Gpu(g) => g.apply(x, nt),
        }
    }
}

/// Moreau envelope of λ Σ_b ‖C_b‖_* with θ = λγ on a fixed block partition (iteration `shift` of the LLR shift sequence)
pub struct LlrHuber {
    pub be: LlrBackend,
    pub theta: f64,
    pub gamma: f64,
    pub shift: u64,
}

impl LlrHuber {
    /// singular-value soft threshold at `thr` in place (the exact prox of thr Σ‖·‖_*), the stats of that call
    pub fn svt(&mut self, z: &mut [C32], nt: usize, thr: f64) -> Result<LlrStats, Error> {
        let shift = self.shift;
        let l = self.be.llr();
        l.thr_abs = Some(thr);
        l.set_iteration(shift);
        self.be.apply(z, nt)?;
        Ok(self.be.llr().last.clone())
    }

    /// value at z and the gradient (z − SVT_θ(z)) / γ into `g`
    pub fn eval(&mut self, z: &[C32], g: &mut [C32], nt: usize) -> Result<f64, Error> {
        g.copy_from_slice(z);
        let s = self.svt(g, nt, self.theta)?;
        let ig = 1.0 / self.gamma as f32;
        g.par_chunks_mut(PAR).zip(z.par_chunks(PAR)).for_each(|(gc, zc)| {
            for (u, v) in gc.iter_mut().zip(zc) {
                *u = C32::new((v.re - u.re) * ig, (v.im - u.im) * ig);
            }
        });
        Ok(((hdot(z, z) - s.s2_above) * 0.5 + s.lin) / self.gamma)
    }
}

/// The smoothed priors of a knot stack (k images of nv voxels) and their evaluation counts
pub struct Priors {
    pub k: usize,
    pub nv: usize,
    pub wav: Option<WavHuber>,
    pub llr: Option<LlrHuber>,
    /// wavelet transforms (forward or inverse, of a whole stack) and LLR sweeps (one SVT of a whole stack)
    pub n_wt: usize,
    pub n_llr: usize,
    pub secs: f64,
    tmp: Vec<C32>,
    cbuf: Vec<f32>,
    pbuf: Vec<f32>,
}

impl Priors {
    pub fn new(k: usize, nv: usize, wav: Option<WavHuber>, llr: Option<LlrHuber>) -> Self {
        Self { k, nv, wav, llr, n_wt: 0, n_llr: 0, secs: 0.0, tmp: vec![], cbuf: vec![], pbuf: vec![] }
    }

    pub fn any(&self) -> bool {
        self.wav.is_some() || self.llr.is_some()
    }

    /// total prior gradient at x into g; returns the values (wavelet, LLR)
    pub fn grad(&mut self, x: &[C32], g: &mut [C32]) -> Result<(f64, f64), Error> {
        let t0 = std::time::Instant::now();
        let (k, n) = (self.k, x.len());
        let (mut vw, mut vl) = (0.0, 0.0);
        match &self.wav {
            Some(w) => {
                w.coef(x, k, &mut self.cbuf);
                let mut ps = std::mem::take(&mut self.pbuf);
                ps.resize(self.cbuf.len(), 0.0);
                vw = w.value_psi(&self.cbuf, Some(&mut ps));
                w.synth(&mut ps, k, g);
                self.pbuf = ps;
                self.n_wt += 2;
            }
            None => g.par_chunks_mut(PAR).for_each(|c| c.fill(C32::new(0.0, 0.0))),
        }
        if let Some(l) = self.llr.as_mut() {
            self.tmp.resize(n, C32::new(0.0, 0.0));
            vl = l.eval(x, &mut self.tmp, k)?;
            haxpby(1.0, &self.tmp, 1.0, g);
            self.n_llr += 1;
        }
        self.secs += t0.elapsed().as_secs_f64();
        Ok((vw, vl))
    }
}

// ------------------------------------------------------------------------------------------------------------------
// solvers

/// per-iteration numbers given to the callback
#[derive(Clone, Debug, Default)]
pub struct Info {
    /// 0 = after the initialisation (gradient at the start), then 1, 2, …
    pub iter: usize,
    /// ½‖A x − y‖² and the prior values (NaN where not evaluated)
    pub f: f64,
    pub pw: f64,
    pub pl: f64,
    pub step: f64,
    pub beta: f64,
    pub restart: bool,
    /// prior evaluations so far (wavelet stack transforms, LLR sweeps)
    pub n_wt: usize,
    pub n_llr: usize,
}

type Cb<'a> = &'a mut dyn FnMut(&Info, &CudaSlice<C32>) -> bool;

fn neg(kern: &Kernels, st: &Arc<CudaStream>, dummy: &CudaSlice<C32>, v: &mut CudaSlice<C32>) -> Result<(), Error> {
    kern.axpby(st, re(0.0), dummy, re(-1.0), v)
}

/// upload the host prior gradient `gh` into `tmp` and add it to `g`
fn add_host(kern: &Kernels, st: &Arc<CudaStream>, gh: &[C32], tmp: &mut CudaSlice<C32>, g: &mut CudaSlice<C32>) -> Result<(), Error> {
    st.memcpy_htod(gh, tmp)?;
    kern.axpby(st, re(1.0), tmp, re(1.0), g)
}

/// g = −Aᴴ r + ∇R(x) (x downloaded into `xh`, prior gradient via `gh` and the device buffer `tmp`); fills info.f / pw / pl
#[allow(clippy::too_many_arguments)]
fn full_grad(op: &mut impl LinOp, kern: &Kernels, st: &Arc<CudaStream>, pri: &mut Priors, x: &CudaSlice<C32>, r: &CudaSlice<C32>, g: &mut CudaSlice<C32>, tmp: &mut CudaSlice<C32>, xh: &mut [C32], gh: &mut [C32], info: &mut Info) -> Result<(), Error> {
    op.adjoint(st, r, g)?;
    kern.axpby(st, re(0.0), tmp, re(-1.0), g)?;
    info.f = 0.5 * kern.dot(st, r, r)?.re;
    if pri.any() {
        st.memcpy_dtoh(x, xh)?;
        let (vw, vl) = pri.grad(xh, gh)?;
        info.pw = vw;
        info.pl = vl;
        add_host(kern, st, gh, tmp, g)?;
    }
    info.n_wt = pri.n_wt;
    info.n_llr = pri.n_llr;
    Ok(())
}

/// Nonlinear PCG (PR+) on ½‖A x − y‖² + priors in the metric P. `r` = y − A x on entry (consumed).
#[allow(clippy::too_many_arguments)]
/// Returns the final residual y − A x (a continuation stage hands it to the next).
pub fn ncg(op: &mut impl LinOp, kern: &Kernels, st: &Arc<CudaStream>, metric: &PriorMetric, pri: &mut Priors, x: &mut CudaSlice<C32>, mut r: CudaSlice<C32>, cb: Cb<'_>) -> Result<CudaSlice<C32>, Error> {
    let n = x.len();
    let k = pri.k;
    let mut g = st.alloc_zeros::<C32>(n)?;
    let mut gp = st.alloc_zeros::<C32>(n)?;
    let mut d = st.alloc_zeros::<C32>(n)?;
    let mut z = st.alloc_zeros::<C32>(n)?;
    let mut q = st.alloc_zeros::<C32>(r.len())?;
    let zero = C32::new(0.0, 0.0);
    let mut xh = vec![zero; n];
    let mut dh = vec![zero; n];
    let has_l = pri.llr.is_some();
    let has_w = pri.wav.is_some();
    let mut glh = vec![zero; if has_l { n } else { 0 }];
    let mut glt = vec![zero; if has_l { n } else { 0 }];
    let mut wh = vec![zero; if has_l { n } else { 0 }];
    let mut gwh = vec![zero; if has_w { n } else { 0 }];
    let (mut cx, mut cd) = (Vec::<f32>::new(), Vec::<f32>::new());
    let mut info = Info { pw: f64::NAN, pl: f64::NAN, ..Default::default() };
    // gradient at x
    op.adjoint(st, &r, &mut g)?;
    neg(kern, st, &z, &mut g)?;
    st.memcpy_dtoh(x, &mut xh)?;
    let t0 = std::time::Instant::now();
    let mut sumh = vec![zero; n];
    if let Some(w) = &pri.wav {
        w.coef(&xh, k, &mut cx);
        cd.resize(cx.len(), 0.0);
        info.pw = w.value_psi(&cx, Some(&mut cd));
        w.synth(&mut cd, k, &mut gwh);
        pri.n_wt += 2;
        haxpby(1.0, &gwh, 1.0, &mut sumh);
    }
    if let Some(l) = pri.llr.as_mut() {
        info.pl = l.eval(&xh, &mut glh, k)?;
        pri.n_llr += 1;
        haxpby(1.0, &glh, 1.0, &mut sumh);
    }
    pri.secs += t0.elapsed().as_secs_f64();
    if pri.any() {
        add_host(kern, st, &sumh, &mut z, &mut g)?;
    }
    info.f = 0.5 * kern.dot(st, &r, &r)?.re;
    info.n_wt = pri.n_wt;
    info.n_llr = pri.n_llr;
    if !cb(&info, x) {
        return Ok(r);
    }
    metric.apply_metric(st, &g, &mut z)?;
    st.memcpy_dtod(&z, &mut d)?;
    neg(kern, st, &z, &mut d)?;
    let mut gz = kern.dot(st, &g, &z)?.re;
    st.memcpy_dtod(&g, &mut gp)?;
    // LLR curvature along d per ‖d‖² (secant of the previous iteration)
    let mut kap_l = 0.0f64;
    for it in 1.. {
        op.forward(st, &d, &mut q)?;
        let bq = kern.dot(st, &q, &q)?.re;
        let phi0 = kern.dot(st, &g, &d)?.re;
        st.memcpy_dtoh(&d, &mut dh)?;
        let t1 = std::time::Instant::now();
        let dd = hdot(&dh, &dh);
        let pl0 = if has_l { hdot(&glh, &dh) } else { 0.0 };
        let pw0 = if let Some(w) = &pri.wav {
            w.coef(&dh, k, &mut cd);
            pri.n_wt += 1;
            w.line(&cx, &cd, 0.0).1
        } else {
            0.0
        };
        let af0 = phi0 - pl0 - pw0;
        // φ'(a) of the model: data exact, wavelet exact, LLR linearised with curvature h
        let wav = pri.wav.as_ref();
        let solve = |h: f64| -> f64 {
            let fp = |a: f64| af0 + a * bq + pl0 + a * h + wav.map_or(0.0, |w| w.line(&cx, &cd, a).1);
            let mut hi = -phi0 / (bq + h).max(1e-300);
            if wav.is_none() || !(hi > 0.0) {
                return hi;
            }
            let (mut lo, mut flo, mut fhi) = (0.0, phi0, fp(hi));
            let mut guard = 0;
            while fhi < 0.0 && guard < 30 {
                lo = hi;
                flo = fhi;
                hi *= 2.0;
                fhi = fp(hi);
                guard += 1;
            }
            // Illinois regula falsi
            let mut side = 0;
            let mut a = hi;
            for _ in 0..40 {
                a = (lo * fhi - hi * flo) / (fhi - flo);
                let fa = fp(a);
                if fa.abs() <= 1e-4 * phi0.abs() || (hi - lo) <= 1e-5 * hi {
                    break;
                }
                if fa < 0.0 {
                    lo = a;
                    flo = fa;
                    if side == -1 {
                        fhi *= 0.5;
                    }
                    side = -1;
                } else {
                    hi = a;
                    fhi = fa;
                    if side == 1 {
                        flo *= 0.5;
                    }
                    side = 1;
                }
            }
            a
        };
        let mut a = solve(kap_l * dd);
        if !(a > 0.0) || !a.is_finite() {
            eprintln!("    ncg it {it}: no descent step (φ'(0) = {phi0:.3e}), stopping");
            return Ok(r);
        }
        if let Some(l) = pri.llr.as_mut() {
            // evaluate the LLR envelope at a, secant curvature, re-solve; a second evaluation if the step moved > 10 %
            wh.par_chunks_mut(PAR).zip(xh.par_chunks(PAR).zip(dh.par_chunks(PAR))).for_each(|(w, (xc, dc))| {
                for ((u, v), e) in w.iter_mut().zip(xc).zip(dc) {
                    *u = C32::new(v.re + a as f32 * e.re, v.im + a as f32 * e.im);
                }
            });
            info.pl = l.eval(&wh, &mut glt, k)?;
            pri.n_llr += 1;
            let pla = hdot(&glt, &dh);
            let h = ((pla - pl0) / a).max(0.0);
            kap_l = h / dd.max(1e-300);
            let a1 = solve(h);
            if a1 > 0.0 && a1.is_finite() && (a1 - a).abs() > 0.1 * a {
                a = a1;
                wh.par_chunks_mut(PAR).zip(xh.par_chunks(PAR).zip(dh.par_chunks(PAR))).for_each(|(w, (xc, dc))| {
                    for ((u, v), e) in w.iter_mut().zip(xc).zip(dc) {
                        *u = C32::new(v.re + a as f32 * e.re, v.im + a as f32 * e.im);
                    }
                });
                info.pl = l.eval(&wh, &mut glt, k)?;
                pri.n_llr += 1;
            }
            std::mem::swap(&mut glh, &mut glt);
        }
        pri.secs += t1.elapsed().as_secs_f64();
        // the step
        kern.axpby(st, re(a), &d, re(1.0), x)?;
        kern.axpby(st, re(-a), &q, re(1.0), &mut r)?;
        haxpby(a as f32, &dh, 1.0, &mut xh);
        op.adjoint(st, &r, &mut g)?;
        neg(kern, st, &z, &mut g)?;
        let t2 = std::time::Instant::now();
        if pri.any() {
            sumh.par_chunks_mut(PAR).for_each(|c| c.fill(zero));
        }
        if let Some(w) = &pri.wav {
            faxpby(a as f32, &cd, 1.0, &mut cx);
            info.pw = w.value_psi(&cx, Some(&mut cd));
            w.synth(&mut cd, k, &mut gwh);
            pri.n_wt += 1;
            haxpby(1.0, &gwh, 1.0, &mut sumh);
        }
        if has_l {
            haxpby(1.0, &glh, 1.0, &mut sumh);
        }
        pri.secs += t2.elapsed().as_secs_f64();
        if pri.any() {
            add_host(kern, st, &sumh, &mut z, &mut g)?;
        }
        info.iter = it;
        info.step = a;
        info.f = 0.5 * kern.dot(st, &r, &r)?.re;
        info.n_wt = pri.n_wt;
        info.n_llr = pri.n_llr;
        // direction
        metric.apply_metric(st, &g, &mut z)?;
        let gz_new = kern.dot(st, &g, &z)?.re;
        let gpz = kern.dot(st, &gp, &z)?.re;
        let mut beta = ((gz_new - gpz) / gz).max(0.0);
        let mut restart = false;
        if gpz.abs() >= 0.2 * gz_new {
            beta = 0.0;
            restart = true;
        }
        kern.axpby(st, re(-1.0), &z, re(beta), &mut d)?;
        if kern.dot(st, &d, &g)?.re >= 0.0 {
            st.memcpy_dtod(&z, &mut d)?;
            neg(kern, st, &z, &mut d)?;
            beta = 0.0;
            restart = true;
        }
        st.memcpy_dtod(&g, &mut gp)?;
        gz = gz_new;
        info.beta = beta;
        info.restart = restart;
        if !cb(&info, x) {
            break;
        }
    }
    Ok(r)
}

/// FISTA (accelerated gradient, step 1/L in the metric P, gradient restart) on the smoothed objective. `y` = the data
/// (kept); every iteration: one forward (the residual at the extrapolated point) and one adjoint.
#[allow(clippy::too_many_arguments)]
pub fn fista(op: &mut impl LinOp, kern: &Kernels, st: &Arc<CudaStream>, metric: &PriorMetric, pri: &mut Priors, x: &mut CudaSlice<C32>, y: CudaSlice<C32>, lip: f64, cb: Cb<'_>) -> Result<(), Error> {
    let n = x.len();
    let zero = C32::new(0.0, 0.0);
    let mut xo = st.alloc_zeros::<C32>(n)?;
    let mut yk = st.clone_dtod(x)?;
    let mut g = st.alloc_zeros::<C32>(n)?;
    let mut z = st.alloc_zeros::<C32>(n)?;
    let mut q = st.alloc_zeros::<C32>(y.len())?;
    let (mut yh, mut gh) = (vec![zero; n], vec![zero; n]);
    let mut info = Info { pw: f64::NAN, pl: f64::NAN, ..Default::default() };
    op.forward(st, &yk, &mut q)?;
    kern.axpby(st, re(1.0), &y, re(-1.0), &mut q)?;
    full_grad(op, kern, st, pri, &yk, &q, &mut g, &mut z, &mut yh, &mut gh, &mut info)?;
    if !cb(&info, x) {
        return Ok(());
    }
    let mut t = 1.0f64;
    for it in 1.. {
        metric.apply_metric(st, &g, &mut z)?;
        st.memcpy_dtod(x, &mut xo)?;
        st.memcpy_dtod(&yk, x)?;
        kern.axpby(st, re(-1.0 / lip), &z, re(1.0), x)?;
        // restart (O'Donoghue–Candès gradient scheme): ⟨∇F(y), x⁺ − x⟩ > 0
        let up = kern.dot(st, &g, x)?.re - kern.dot(st, &g, &xo)?.re;
        let restart = up > 0.0;
        let beta = if restart {
            t = 1.0;
            0.0
        } else {
            let tn = 0.5 * (1.0 + (1.0 + 4.0 * t * t).sqrt());
            let b = (t - 1.0) / tn;
            t = tn;
            b
        };
        st.memcpy_dtod(x, &mut yk)?;
        kern.axpby(st, re(-beta), &xo, re(1.0 + beta), &mut yk)?;
        op.forward(st, &yk, &mut q)?;
        kern.axpby(st, re(1.0), &y, re(-1.0), &mut q)?;
        full_grad(op, kern, st, pri, &yk, &q, &mut g, &mut z, &mut yh, &mut gh, &mut info)?;
        info.iter = it;
        info.step = 1.0 / lip;
        info.beta = beta;
        info.restart = restart;
        if !cb(&info, x) {
            break;
        }
    }
    Ok(())
}

/// ZMSHR1 in the metric P (global secant, Hong et al.; `p2np.rs` form) on the smoothed objective, step scale α.
/// `r` = y − A x on entry (consumed). Previous iterate / gradient on the host.
#[allow(clippy::too_many_arguments)]
pub fn zmshr1(op: &mut impl LinOp, kern: &Kernels, st: &Arc<CudaStream>, metric: &PriorMetric, pri: &mut Priors, x: &mut CudaSlice<C32>, mut r: CudaSlice<C32>, alpha: f64, cb: Cb<'_>) -> Result<(), Error> {
    let n = x.len();
    let zero = C32::new(0.0, 0.0);
    let mut g = st.alloc_zeros::<C32>(n)?;
    let mut s = st.alloc_zeros::<C32>(n)?;
    let mut m = st.alloc_zeros::<C32>(n)?;
    let mut wb = st.alloc_zeros::<C32>(n)?;
    let mut mb = st.alloc_zeros::<C32>(n)?;
    let mut q = st.alloc_zeros::<C32>(r.len())?;
    let mut xph = vec![zero; n];
    let mut gph = vec![zero; n];
    let mut xh = vec![zero; n];
    let mut gh = vec![zero; n];
    let mut info = Info { pw: f64::NAN, pl: f64::NAN, ..Default::default() };
    full_grad(op, kern, st, pri, x, &r, &mut g, &mut wb, &mut xh, &mut gh, &mut info)?;
    if !cb(&info, x) {
        return Ok(());
    }
    for it in 0.. {
        // α g (stored scaled, as p2np)
        kern.axpby(st, re(0.0), &s, re(alpha), &mut g)?;
        let (mut tau_l, mut c_l) = (1.0, 0.0);
        if it == 0 {
            metric.apply_metric(st, &g, &mut s)?;
        } else {
            st.memcpy_htod(&xph, &mut s)?;
            kern.axpby(st, re(1.0), x, re(-1.0), &mut s)?;
            st.memcpy_htod(&gph, &mut m)?;
            kern.axpby(st, re(1.0), &g, re(-1.0), &mut m)?;
            metric.apply_inv(st, &s, &mut wb)?;
            metric.apply_metric(st, &m, &mut mb)?;
            let d1 = [kern.dot(st, &s, &wb)?.re, kern.dot(st, &s, &m)?.re, kern.dot(st, &m, &mb)?.re];
            let (a, tau) = crate::p2np::zmshr1_scalars_guarded(d1[0], d1[1], d1[2]);
            // w = (1 − τa) s − τ(1−a) M m (into mb), z = (1−a) m + a M⁻¹ s (into wb)
            kern.axpby(st, re(1.0 - tau * a), &s, re(-tau * (1.0 - a)), &mut mb)?;
            kern.axpby(st, re(1.0 - a), &m, re(a), &mut wb)?;
            let d2 = [kern.dot(st, &mb, &wb)?.re, kern.dot(st, &mb, &mb)?.re, kern.dot(st, &wb, &wb)?.re, kern.dot(st, &mb, &g)?.re];
            let c = crate::p2np::zmshr1_rank_one_sets(&[d2], false)[0];
            metric.apply_metric(st, &g, &mut s)?;
            kern.axpby(st, re(c), &mb, re(tau), &mut s)?;
            tau_l = tau;
            c_l = c;
        }
        st.memcpy_dtoh(x, &mut xph)?;
        st.memcpy_dtoh(&g, &mut gph)?;
        kern.axpby(st, re(-1.0), &s, re(1.0), x)?;
        op.forward(st, &s, &mut q)?;
        kern.axpby(st, re(1.0), &q, re(1.0), &mut r)?;
        full_grad(op, kern, st, pri, x, &r, &mut g, &mut wb, &mut xh, &mut gh, &mut info)?;
        info.iter = it + 1;
        info.step = tau_l;
        info.beta = c_l;
        info.n_wt = pri.n_wt;
        info.n_llr = pri.n_llr;
        if !cb(&info, x) {
            break;
        }
    }
    Ok(())
}

/// Scaled ADMM on ½‖A x − y‖² + λ_w ‖W S x‖₁ + λ_l Σ_b ‖C_b(x)‖_* (exact, unsmoothed; the thresholds are the priors'
/// λ = θ/γ). `metric` must carry the ridge 2ρ (ρ per split) in its prior term. `r` = y − A x on entry.
#[allow(clippy::too_many_arguments)]
pub fn admm(op: &mut impl LinOp, kern: &Kernels, st: &Arc<CudaStream>, metric: &PriorMetric, pri: &mut Priors, x: &mut CudaSlice<C32>, mut r: CudaSlice<C32>, rho: f64, n_inner: usize, cb: Cb<'_>) -> Result<(), Error> {
    let n = x.len();
    let k = pri.k;
    let zero = C32::new(0.0, 0.0);
    let nsplit = pri.wav.is_some() as usize + pri.llr.is_some() as usize;
    assert!(nsplit > 0, "ADMM needs a prior");
    let lam = rho * nsplit as f64;
    let mut s = st.alloc_zeros::<C32>(n)?;
    let mut p = st.alloc_zeros::<C32>(n)?;
    let mut z = st.alloc_zeros::<C32>(n)?;
    let mut mc = st.alloc_zeros::<C32>(n)?;
    let mut q = st.alloc_zeros::<C32>(r.len())?;
    let mut xh = vec![zero; n];
    st.memcpy_dtoh(x, &mut xh)?;
    // splits: u_w = W S x, u_l = x; scaled duals 0; the centre m = x
    let lw = pri.wav.as_ref().map(|w| w.theta / w.gamma);
    let ll = pri.llr.as_ref().map(|l| l.theta / l.gamma);
    let (mut cx, mut uw, mut vw) = (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<f32>::new());
    if let Some(w) = &pri.wav {
        w.coef(&xh, k, &mut uw);
        vw = vec![0.0; uw.len()];
        pri.n_wt += 1;
    }
    let (mut ul, mut vl) = if pri.llr.is_some() { (xh.clone(), vec![zero; n]) } else { (vec![], vec![]) };
    let mut mh = xh.clone();
    let mut tmp = vec![zero; n];
    st.memcpy_htod(&mh, &mut mc)?;
    op.adjoint(st, &r, &mut s)?;
    kern.axpby(st, re(-lam), x, re(1.0), &mut s)?;
    kern.axpby(st, re(lam), &mc, re(1.0), &mut s)?;
    let mut info = Info { pw: f64::NAN, pl: f64::NAN, ..Default::default() };
    info.f = 0.5 * kern.dot(st, &r, &r)?.re;
    info.n_wt = pri.n_wt;
    if !cb(&info, x) {
        return Ok(());
    }
    let mut it = 0;
    loop {
        // x-update: n_inner CGLS steps on ½‖A x − y‖² + (λ/2)‖x − m‖² from the current x, metric P (ridge λ)
        metric.apply_metric(st, &s, &mut z)?;
        st.memcpy_dtod(&z, &mut p)?;
        let mut gam = kern.dot(st, &s, &z)?.re;
        for _ in 0..n_inner {
            op.forward(st, &p, &mut q)?;
            let qq = kern.dot(st, &q, &q)?.re + lam * kern.dot(st, &p, &p)?.re;
            let al = gam / qq;
            kern.axpby(st, re(al), &p, re(1.0), x)?;
            kern.axpby(st, re(-al), &q, re(1.0), &mut r)?;
            op.adjoint(st, &r, &mut s)?;
            kern.axpby(st, re(-lam), x, re(1.0), &mut s)?;
            kern.axpby(st, re(lam), &mc, re(1.0), &mut s)?;
            metric.apply_metric(st, &s, &mut z)?;
            let gn = kern.dot(st, &s, &z)?.re;
            kern.axpby(st, re(1.0), &z, re(gn / gam), &mut p)?;
            gam = gn;
            it += 1;
            info.iter = it;
            info.f = 0.5 * kern.dot(st, &r, &r)?.re;
            info.step = al;
            info.n_wt = pri.n_wt;
            info.n_llr = pri.n_llr;
            if !cb(&info, x) {
                return Ok(());
            }
        }
        // prox steps and duals
        let t0 = std::time::Instant::now();
        st.memcpy_dtoh(x, &mut xh)?;
        mh.par_chunks_mut(PAR).for_each(|c| c.fill(zero));
        if let Some(w) = &pri.wav {
            w.coef(&xh, k, &mut cx);
            // t = W S x + v → u = soft(t), v = t − u; centre part Sᴴ Wᴴ (u − v)
            faxpby(1.0, &vw, 1.0, &mut cx);
            uw.copy_from_slice(&cx);
            w.soft(&mut uw, lw.unwrap() / rho);
            vw.copy_from_slice(&cx);
            faxpby(-1.0, &uw, 1.0, &mut vw);
            cx.copy_from_slice(&uw);
            faxpby(-1.0, &vw, 1.0, &mut cx);
            w.synth(&mut cx, k, &mut tmp);
            pri.n_wt += 2;
            haxpby(1.0 / nsplit as f32, &tmp, 1.0, &mut mh);
        }
        if let Some(l) = pri.llr.as_mut() {
            // u = SVT(x + v), v ← v + x − u; centre part u − v
            ul.copy_from_slice(&xh);
            haxpby(1.0, &vl, 1.0, &mut ul);
            l.svt(&mut ul, k, ll.unwrap() / rho)?;
            pri.n_llr += 1;
            haxpby(1.0, &xh, 1.0, &mut vl);
            haxpby(-1.0, &ul, 1.0, &mut vl);
            tmp.copy_from_slice(&ul);
            haxpby(-1.0, &vl, 1.0, &mut tmp);
            haxpby(1.0 / nsplit as f32, &tmp, 1.0, &mut mh);
        }
        pri.secs += t0.elapsed().as_secs_f64();
        // s = Aᴴr − λ(x − m): swap the centre
        kern.axpby(st, re(-lam), &mc, re(1.0), &mut s)?;
        st.memcpy_htod(&mh, &mut mc)?;
        kern.axpby(st, re(lam), &mc, re(1.0), &mut s)?;
    }
}

/// λ_max(P ∇²F) at x by `iters` power iterations: data Hessian AᴴA exactly (one forward + adjoint each), prior
/// Hessian by a forward difference of the prior gradient (one prior evaluation each). Returns (λ, history).
#[allow(clippy::too_many_arguments)]
pub fn power_lmax(op: &mut impl LinOp, kern: &Kernels, st: &Arc<CudaStream>, metric: &PriorMetric, pri: &mut Priors, x: &CudaSlice<C32>, data_len: usize, iters: usize) -> Result<(f64, Vec<f64>), Error> {
    let n = x.len();
    let zero = C32::new(0.0, 0.0);
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let host: Vec<C32> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            C32::new(((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5, ((s >> 16 & 0xFFFFFF) as f32 / (1u64 << 24) as f32) - 0.5)
        })
        .collect();
    let mut v = st.clone_htod(&host)?;
    drop(host);
    let mut u = st.alloc_zeros::<C32>(n)?;
    let mut w = st.alloc_zeros::<C32>(n)?;
    let mut q = st.alloc_zeros::<C32>(data_len)?;
    let (mut xh, mut g0, mut g1, mut vh) = (vec![zero; n], vec![zero; n], vec![zero; n], vec![zero; n]);
    if pri.any() {
        st.memcpy_dtoh(x, &mut xh)?;
        pri.grad(&xh, &mut g0)?;
    }
    let xn = if pri.any() { hdot(&xh, &xh).sqrt() } else { 1.0 };
    let mut hist = vec![];
    let mut lam = 0.0;
    for _ in 0..iters {
        let vn = kern.dot(st, &v, &v)?.re.sqrt();
        kern.axpby(st, re(0.0), &u, re(1.0 / vn), &mut v)?;
        op.forward(st, &v, &mut q)?;
        op.adjoint(st, &q, &mut u)?;
        if pri.any() {
            // (∇R(x + εv) − ∇R(x)) / ε, ε = 10⁻³ ‖x‖ (v is unit)
            let eps = 1e-3 * xn.max(1e-30);
            st.memcpy_dtoh(&v, &mut vh)?;
            haxpby(1.0, &xh, eps as f32, &mut vh);
            pri.grad(&vh, &mut g1)?;
            haxpby(-1.0, &g0, 1.0, &mut g1);
            g1.par_chunks_mut(PAR).for_each(|c| c.iter_mut().for_each(|z| *z = C32::new(z.re / eps as f32, z.im / eps as f32)));
            add_host(kern, st, &g1, &mut w, &mut u)?;
        }
        metric.apply_metric(st, &u, &mut w)?;
        lam = kern.dot(st, &w, &u)?.re / kern.dot(st, &v, &u)?.re.max(1e-300);
        hist.push(lam);
        std::mem::swap(&mut v, &mut w);
    }
    Ok((lam, hist))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64, amp: f32) -> Vec<C32> {
        let mut s = seed;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        (0..n).map(|_| C32::new(amp * u(), amp * u())).collect()
    }

    /// smooth images plus noise (a realistic coefficient distribution: some below, some above θ)
    fn images(d: [usize; 3], k: usize) -> Vec<C32> {
        let nv = d[0] * d[1] * d[2];
        let noise = rnd(nv * k, 7, 0.4);
        (0..nv * k)
            .map(|i| {
                let (q, t) = (i % nv, i / nv);
                let (x, y, z) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
                let v = ((x as f32) * 0.3).cos() * ((y as f32) * 0.2 + t as f32 * 0.4).sin() + 0.3 * ((z as f32) * 0.5).cos();
                C32::new(v + noise[i].re, 0.5 * v + noise[i].im)
            })
            .collect()
    }

    #[test]
    fn wavelet_roundtrip_gradient_line() {
        let d = [18usize, 12, 10];
        let (k, nv) = (3usize, 18 * 12 * 10);
        let x = images(d, k);
        let w = WavHuber::new(d, 2, 0.15, 0.7);
        assert_eq!(w.p, [20, 12, 12]);
        // Sᴴ Wᴴ W S = I (zero padding is an isometry, W orthonormal)
        let mut c = vec![];
        w.coef(&x, k, &mut c);
        let mut back = vec![C32::new(0.0, 0.0); nv * k];
        let mut cc = c.clone();
        w.synth(&mut cc, k, &mut back);
        let err = back.iter().zip(&x).map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)) as f64).sum::<f64>().sqrt() / hdot(&x, &x).sqrt();
        assert!(err < 1e-5, "round trip {err}");
        // gradient Sᴴ Wᴴ ψ(W S x) against central differences of the value along random directions
        let mut psi = vec![0f32; c.len()];
        let v0 = w.value_psi(&c, Some(&mut psi));
        let mut g = vec![C32::new(0.0, 0.0); nv * k];
        w.synth(&mut psi, k, &mut g);
        for seed in [11u64, 12, 13] {
            let dir = rnd(nv * k, seed, 1.0);
            let h = 1e-2f32;
            let val = |s: f32| {
                let xs: Vec<C32> = x.iter().zip(&dir).map(|(a, b)| C32::new(a.re + s * b.re, a.im + s * b.im)).collect();
                let mut cs = vec![];
                w.coef(&xs, k, &mut cs);
                w.value_psi(&cs, None)
            };
            let fd = (val(h) - val(-h)) / (2.0 * h as f64);
            let an = hdot(&g, &dir);
            assert!((fd - an).abs() <= 5e-3 * an.abs().max(1.0), "wavelet gradient: fd {fd} vs {an}");
            // the line form: value and derivative at a match the transformed point
            let mut cd = vec![];
            w.coef(&dir, k, &mut cd);
            let (lv, ld) = w.line(&c, &cd, 0.0);
            assert!((lv - v0).abs() < 1e-4 * v0.abs());
            assert!((ld - an).abs() < 1e-3 * an.abs().max(1.0), "line derivative {ld} vs {an}");
            let (la, _) = w.line(&c, &cd, 0.3);
            assert!((la - val(0.3)).abs() < 1e-3 * la.abs(), "line value {la} vs {}", val(0.3));
        }
    }

    #[test]
    fn llr_envelope_gradient() {
        let d = [12usize, 12, 8];
        let (k, nv) = (6usize, 12 * 12 * 8);
        let x = images(d, k);
        let mut l = Llr::new(d, 4, 1.0, vec![true; nv]);
        l.zeta_basis = Some((0..k * k).map(|q| if q / k == q % k { 1.0 } else { 0.0 }).collect());
        let mut h = LlrHuber { be: LlrBackend::Cpu(l), theta: 1.5, gamma: 0.8, shift: 3 };
        let mut g = vec![C32::new(0.0, 0.0); nv * k];
        let v0 = h.eval(&x, &mut g, k).unwrap();
        assert!(v0.is_finite() && v0 > 0.0);
        let st = h.be.llr().last.clone();
        assert!(st.n_above > 0 && st.n_above < st.n_sv, "threshold should split the spectrum: {st:?}");
        // ζ in [0, 1], identity basis: discarded fraction per knot
        assert!(st.zeta.iter().all(|&z| (-1e-6..=1.0 + 1e-6).contains(&z)), "{:?}", st.zeta);
        let mut gt = vec![C32::new(0.0, 0.0); nv * k];
        for seed in [21u64, 22] {
            let dir = rnd(nv * k, seed, 1.0);
            let e = 1e-2f32;
            let mut val = |s: f32| {
                let xs: Vec<C32> = x.iter().zip(&dir).map(|(a, b)| C32::new(a.re + s * b.re, a.im + s * b.im)).collect();
                h.eval(&xs, &mut gt, k).unwrap()
            };
            let fd = (val(e) - val(-e)) / (2.0 * e as f64);
            let an = hdot(&g, &dir);
            assert!((fd - an).abs() <= 3e-3 * an.abs().max(1.0), "LLR envelope gradient: fd {fd} vs {an}");
        }
        // with the threshold at 0 the envelope is ‖x‖²·0 + …: gradient (x − x)/γ = 0
        h.theta = 0.0;
        let v = h.eval(&x, &mut g, k).unwrap();
        assert!(v.abs() < 1e-6 * hdot(&x, &x), "θ = 0: value {v}");
    }

    #[test]
    fn wavelet_symbol_bands() {
        let d = [16usize, 16, 16];
        let w = WavHuber::new(d, 2, 0.1, 1.0);
        // fractions: level 1 → 1, level 2 → 0.5
        let fr: Vec<(usize, [usize; 3], f64)> = (1..=2).flat_map(|l| crate::sepwt::subbands(w.p, l).into_iter().map(move |(e, _, _)| (l, e, if l == 1 { 1.0 } else { 0.5 }))).collect();
        let s = w.symbol(&fr);
        let at = |i: usize, j: usize, kk: usize| s[i + 16 * (j + 16 * kk)];
        assert_eq!(at(0, 0, 0), 0.0); // DC: approximation band
        assert_eq!(at(8, 0, 0), 1.0); // Nyquist along x: finest
        assert_eq!(at(5, 2, 1), 1.0); // f_x = 5/16 ∈ [1/4, 1/2): finest
        assert_eq!(at(3, 1, 0), 0.5); // f_x = 3/16 ∈ [1/8, 1/4): level 2
        assert_eq!(at(1, 1, 1), 0.0); // f < 1/8 on every axis: approximation (2 levels)
        assert_eq!(at(15, 0, 0), 0.0); // f = 1/16
    }
}
