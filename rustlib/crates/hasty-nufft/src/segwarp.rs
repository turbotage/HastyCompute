//! In-readout motion inside the time segments: a lagged known warp of the moving voxels
//! (`research/readout_physics_budget.md`).
//!
//! A spin moving with velocity v (acceleration a) during the readout is seen by sample j at r + d(τ_j),
//! d(τ) = v t1(τ) + a t2(τ), where t1 = (γM1·k)/|k|², t2 = (γM2/2·k)/|k|² are the parts of the readout's first and
//! (half) second moment about the echo parallel to M0 = k/γ (the perpendicular remainder is not a displacement and is
//! not modelled here). With the segment basis in **cardinal (interpolatory) form** — nodes τ_l, D'_l ≈ g(·, τ_l),
//! a'_l(τ_m) = δ_lm, obtained from any basis by QDEIM ([`cardinal_form`]) without changing the operator — the moving
//! voxels' segment image is resampled at the node displacement:
//!
//!   u_l = D'_l x + Δu_l,   Δu_l = (W_l − I)(D'_l x) on the moving set V,   W_l = push of x_v to r_v + d_v(τ_l),
//!
//! so y_j = Σ_l a'_l(τ_j) F_{k_j}[S_c u_l]: the same L segment channels and FFTs, plus a sparse correction scattered into
//! the segment's grid channel before the FFT (adjoint: gathered after the inverse FFT, transposed push). The push uses a
//! separable stencil ([`WarpStencil`]); the integer part of d is exact, the fraction goes through the stencil (whose
//! k-response falls off towards the grid's Nyquist edge: a half-voxel shift is not representable at |k_a Δ| = π).
//!
//! CPU reference here (tests, cost model); the GPU path is [`crate::SegNufft::set_warp`] (single-image path, order ≤ 2).

use hasty_cuda::C32;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[inline]
fn c64(z: C32) -> Complex64 {
    Complex64::new(z.re as f64, z.im as f64)
}
#[inline]
fn c32(z: Complex64) -> C32 {
    C32::new(z.re as f32, z.im as f32)
}

/// Separable push stencil for the fractional part of the displacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpStencil {
    /// 2 taps per axis (8 points)
    Trilinear,
    /// Keys a = −1/2 (Catmull–Rom), 4 taps per axis (64 points)
    Cubic,
    /// Lanczos a = 3, 6 taps per axis (216 points)
    Lanczos3,
}

impl WarpStencil {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "trilinear" => Some(Self::Trilinear),
            "cubic" => Some(Self::Cubic),
            "lanczos3" => Some(Self::Lanczos3),
            _ => None,
        }
    }
    /// taps per axis
    pub fn taps(&self) -> usize {
        match self {
            Self::Trilinear => 2,
            Self::Cubic => 4,
            Self::Lanczos3 => 6,
        }
    }
    /// offset of the first tap relative to floor(d)
    pub fn first(&self) -> i64 {
        1 - (self.taps() as i64) / 2
    }
    /// kernel value at distance s (voxels)
    pub fn kernel(&self, s: f64) -> f64 {
        let a = s.abs();
        match self {
            Self::Trilinear => (1.0 - a).max(0.0),
            Self::Cubic => {
                if a < 1.0 {
                    1.5 * a * a * a - 2.5 * a * a + 1.0
                } else if a < 2.0 {
                    -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0
                } else {
                    0.0
                }
            }
            Self::Lanczos3 => {
                if a < 1e-12 {
                    1.0
                } else if a < 3.0 {
                    let p = std::f64::consts::PI * a;
                    3.0 * p.sin() * (p / 3.0).sin() / (p * p)
                } else {
                    0.0
                }
            }
        }
    }
    /// weights of the taps floor(d) + first() + i for the fraction fr = d − floor(d) ∈ [0, 1); unit sum (Lanczos is
    /// renormalised so that a constant image is pushed exactly)
    pub fn weights(&self, fr: f64) -> [f64; 6] {
        let mut w = [0.0; 6];
        let f0 = self.first();
        let mut s = 0.0;
        for (i, wi) in w.iter_mut().enumerate().take(self.taps()) {
            *wi = self.kernel((f0 + i as i64) as f64 - fr);
            s += *wi;
        }
        if *self == Self::Lanczos3 {
            for wi in w.iter_mut().take(self.taps()) {
                *wi /= s;
            }
        }
        w
    }
    /// k-response of the push relative to the exact shift by fr: Σ_m w(m − fr) e^{−iθ(m − fr)} (θ = k_a Δ)
    pub fn response(&self, theta: f64, fr: f64) -> Complex64 {
        let w = self.weights(fr);
        let f0 = self.first();
        (0..self.taps()).map(|i| w[i] * Complex64::from_polar(1.0, -theta * ((f0 + i as i64) as f64 - fr))).sum()
    }
}

/// QDEIM cardinal form of a segment basis (segph: L × nv, atab: ntab × L entry-major): pivoted-QR node selection on the
/// interpolator table, T = atab[nodes] (L × L), atab' = atab T⁻¹, segph' = T segph. The operator is unchanged
/// (a'D' = aD up to f32 rounding); a'_l(τ_{nodes_m}) = δ_lm and D'_l ≈ g(·, τ_{nodes_l}). Returns (segph', atab',
/// table entries of the nodes ascending, condition number of T).
pub fn cardinal_form(segph: &[C32], atab: &[C32], nseg: usize, nv: usize) -> (Vec<C32>, Vec<C32>, Vec<usize>, f64) {
    assert_eq!(segph.len(), nseg * nv);
    assert_eq!(atab.len() % nseg, 0);
    let ntab = atab.len() / nseg;
    // columns = table entries (L-vectors); greedy column pivoting with Gram–Schmidt deflation (= pivoted QR of Aᴴ)
    let mut r: Vec<Vec<Complex64>> = (0..ntab).map(|i| (0..nseg).map(|l| c64(atab[i * nseg + l]).conj()).collect()).collect();
    let mut chosen = Vec::with_capacity(nseg);
    let mut used = vec![false; ntab];
    for _ in 0..nseg {
        let (j, _) = (0..ntab).filter(|&j| !used[j]).map(|j| (j, r[j].iter().map(|z| z.norm_sqr()).sum::<f64>())).fold((usize::MAX, -1.0), |a, b| if b.1 > a.1 { b } else { a });
        used[j] = true;
        chosen.push(j);
        let q: Vec<Complex64> = {
            let n = r[j].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
            r[j].iter().map(|z| z / n).collect()
        };
        for (k, col) in r.iter_mut().enumerate() {
            if used[k] {
                continue;
            }
            let p: Complex64 = q.iter().zip(col.iter()).map(|(a, b)| a.conj() * b).sum();
            for (c, qi) in col.iter_mut().zip(&q) {
                *c -= p * qi;
            }
        }
    }
    chosen.sort_unstable();
    let t = DMatrix::<Complex64>::from_fn(nseg, nseg, |m, l| c64(atab[chosen[m] * nseg + l]));
    let sv = t.clone().svd(false, false).singular_values;
    let cond = sv.max() / sv.min().max(1e-300);
    let ti = t.clone().try_inverse().expect("cardinal_form: singular node matrix");
    let mut atab2 = vec![C32::new(0.0, 0.0); atab.len()];
    for i in 0..ntab {
        for l in 0..nseg {
            let z: Complex64 = (0..nseg).map(|m| c64(atab[i * nseg + m]) * ti[(m, l)]).sum();
            atab2[i * nseg + l] = c32(z);
        }
    }
    let mut segph2 = vec![C32::new(0.0, 0.0); segph.len()];
    for l in 0..nseg {
        for v in 0..nv {
            let z: Complex64 = (0..nseg).map(|m| t[(l, m)] * c64(segph[m * nv + v])).sum();
            segph2[l * nv + v] = c32(z);
        }
    }
    (segph2, atab2, chosen, cond)
}

/// The moving set and its node displacements (one cardiac frame). Displacements in voxels.
#[derive(Clone, Debug)]
pub struct SegWarp {
    pub dims: [usize; 3],
    pub stencil: WarpStencil,
    /// image voxel index of every moving voxel
    pub moving: Vec<u32>,
    /// displacement per (segment, moving voxel), voxels: d_l(v) = v̂ t1_l + â t2_l
    pub disp: Vec<[f32; 3]>,
    pub nseg: usize,
    /// sorted unique image indices of every tap and moving voxel
    pub footprint: Vec<u32>,
    /// optional factor per (segment, moving voxel) applied to the pushed value only (nseg × nm): e.g. the phase of the
    /// field along the path, e^{−i ½ (∇ω·v̂)(τ_l − TE)²} (motion through the B0 gradient, [`SegWarp::path_field`])
    pub phase: Option<Vec<C32>>,
}

impl SegWarp {
    /// v_hat (m/s) and a_hat (m/s², optional) per moving voxel, node parallel moments t1_l (s), t2_l (s²), voxel size
    #[allow(clippy::too_many_arguments)]
    pub fn new(dims: [usize; 3], vox: [f64; 3], moving: Vec<u32>, v_hat: &[[f64; 3]], a_hat: Option<&[[f64; 3]]>, t1: &[f64], t2: &[f64], stencil: WarpStencil) -> Self {
        assert_eq!(moving.len(), v_hat.len());
        assert_eq!(t1.len(), t2.len());
        let nseg = t1.len();
        let nm = moving.len();
        let mut disp = vec![[0f32; 3]; nseg * nm];
        for l in 0..nseg {
            for i in 0..nm {
                for a in 0..3 {
                    let d = v_hat[i][a] * t1[l] + a_hat.map_or(0.0, |ah| ah[i][a]) * t2[l];
                    disp[l * nm + i][a] = (d / vox[a]) as f32;
                }
            }
        }
        let mut w = SegWarp { dims, stencil, moving, disp, nseg, footprint: vec![], phase: None };
        let mut fp: Vec<u32> = w.moving.clone();
        for l in 0..nseg {
            for i in 0..nm {
                w.for_taps(l, i, |q, _| fp.push(q));
            }
        }
        fp.sort_unstable();
        fp.dedup();
        w.footprint = fp;
        w
    }

    /// the field along the path of every moving voxel, linearised: phase −½ (g·v̂)(τ_l − TE)² at the nodes, g = ∇ω (rad/s/m)
    /// at the voxel (e.g. central differences of the B0 map), `t_node` = τ_l − TE (s)
    pub fn path_field(mut self, grad_w: &[[f64; 3]], v_hat: &[[f64; 3]], t_node: &[f64]) -> Self {
        let nm = self.moving.len();
        assert_eq!(grad_w.len(), nm);
        assert_eq!(t_node.len(), self.nseg);
        let mut ph = vec![C32::new(1.0, 0.0); self.nseg * nm];
        for l in 0..self.nseg {
            for i in 0..nm {
                let gv: f64 = (0..3).map(|a| grad_w[i][a] * v_hat[i][a]).sum();
                ph[l * nm + i] = c32(Complex64::from_polar(1.0, -0.5 * gv * t_node[l] * t_node[l]));
            }
        }
        self.phase = Some(ph);
        self
    }

    fn phase_of(&self, l: usize, i: usize) -> Complex64 {
        self.phase.as_ref().map_or(Complex64::new(1.0, 0.0), |p| c64(p[l * self.moving.len() + i]))
    }

    /// visit the in-image taps (image index, weight) of moving voxel i at segment l
    pub fn for_taps(&self, l: usize, i: usize, mut f: impl FnMut(u32, f64)) {
        let [nx, ny, nz] = self.dims;
        let q = self.moving[i] as usize;
        let c = [q % nx, (q / nx) % ny, q / (nx * ny)];
        let d = self.disp[l * self.moving.len() + i];
        let nt = self.stencil.taps();
        let f0 = self.stencil.first();
        let mut base = [0i64; 3];
        let mut w = [[0.0f64; 6]; 3];
        for a in 0..3 {
            let da = d[a] as f64;
            let fl = da.floor();
            base[a] = c[a] as i64 + fl as i64 + f0;
            w[a] = self.stencil.weights(da - fl);
        }
        for iz in 0..nt {
            let z = base[2] + iz as i64;
            if z < 0 || z >= nz as i64 {
                continue;
            }
            for iy in 0..nt {
                let y = base[1] + iy as i64;
                if y < 0 || y >= ny as i64 {
                    continue;
                }
                let wzy = w[2][iz] * w[1][iy];
                for ix in 0..nt {
                    let x = base[0] + ix as i64;
                    if x < 0 || x >= nx as i64 {
                        continue;
                    }
                    f((x as usize + nx * (y as usize + ny * z as usize)) as u32, wzy * w[0][ix]);
                }
            }
        }
    }

    fn fpos(&self, q: u32) -> usize {
        self.footprint.binary_search(&q).expect("tap outside the footprint")
    }

    /// Δu (nseg × |footprint|) for an image x (nv) and the cardinal segment diagonals segph' (nseg × nv)
    pub fn forward_cpu(&self, segph: &[C32], x: &[C32]) -> Vec<C32> {
        let nv = x.len();
        let nf = self.footprint.len();
        let mut du = vec![Complex64::new(0.0, 0.0); self.nseg * nf];
        for l in 0..self.nseg {
            for (i, &q) in self.moving.iter().enumerate() {
                let s = c64(segph[l * nv + q as usize]) * c64(x[q as usize]);
                du[l * nf + self.fpos(q)] -= s;
                let sp = s * self.phase_of(l, i);
                self.for_taps(l, i, |t, w| du[l * nf + self.fpos(t)] += sp * w);
            }
        }
        du.into_iter().map(c32).collect()
    }

    /// adjoint: x̄ (nv) += Σ_l conj(D'_l) (W_l − I)ᵀ r_l, r: nseg × |footprint|
    pub fn adjoint_cpu(&self, segph: &[C32], r: &[C32], xbar: &mut [C32]) {
        let nv = xbar.len();
        let nf = self.footprint.len();
        for l in 0..self.nseg {
            for (i, &q) in self.moving.iter().enumerate() {
                let mut push = Complex64::new(0.0, 0.0);
                self.for_taps(l, i, |t, w| push += c64(r[l * nf + self.fpos(t)]) * w);
                let acc = push * self.phase_of(l, i).conj() - c64(r[l * nf + self.fpos(q)]);
                xbar[q as usize] = c32(c64(xbar[q as usize]) + c64(segph[l * nv + q as usize]).conj() * acc);
            }
        }
    }

    /// stencil taps per moving voxel and segment (cost model: forward = adjoint ≈ nseg·|V|·taps complex MACs for the
    /// correction images + nseg·|footprint|·coils for the grid scatter / gather)
    pub fn macs(&self, ncoil: usize) -> (usize, usize) {
        let t = self.stencil.taps().pow(3);
        (self.nseg * self.moving.len() * t, self.nseg * self.footprint.len() * ncoil)
    }
}

// ───────────────────────────────────────────────────────────────────── GPU (SegNufft::set_warp)

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};

use crate::plan::{Dims, MDims, NufftError};

/// correction images Δu (nseg × |F|): build (forward), grid scatter (after the pad, before the FFT), grid gather (after
/// the inverse FFT), transposed push (adjoint). Layout and centring of the grid as `seg_pad` / `seg_crop`.
const WARP_SRC: &str = r#"
struct cx { float x, y; };
struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };
struct MDims { int N0, N1, N2, fftord; };
__device__ __forceinline__ cx cmul(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ __forceinline__ cx cmulc(cx a, cx b) { return cx{a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x}; }
__device__ __forceinline__ int wrapi(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }
__device__ float wkern(int taps, float s) {
  const float a = fabsf(s);
  if (taps == 2) return fmaxf(1.f - a, 0.f);
  if (taps == 4) return a < 1.f ? 1.5f * a * a * a - 2.5f * a * a + 1.f : (a < 2.f ? -0.5f * a * a * a + 2.5f * a * a - 4.f * a + 2.f : 0.f);
  if (a < 1e-6f) return 1.f;
  if (a >= 3.f) return 0.f;
  const float p = 3.14159265358979f * a;
  return 3.f * sinf(p) * sinf(p / 3.f) / (p * p);
}
__device__ void wweights(int taps, float fr, float* w) {
  const int f0 = 1 - taps / 2;
  float s = 0.f;
  for (int i = 0; i < taps; ++i) { w[i] = wkern(taps, (float)(f0 + i) - fr); s += w[i]; }
  if (taps == 6) for (int i = 0; i < taps; ++i) w[i] /= s;
}
// thread per (segment, moving voxel): du[l][fpos(tap)] += w s, du[l][fpos(v)] -= s, s = segph[l][v] x[v]
extern "C" __global__ void warp_build(const cx* __restrict__ x, const cx* __restrict__ segph, const unsigned* __restrict__ moving, const float* __restrict__ disp,
                                      const int* __restrict__ fpos, cx* __restrict__ du, const cx* __restrict__ ph, int has_ph, long long nv, int nm, int nf, int nseg, int taps, int nx, int ny, int nz) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)nm * nseg) return;
  const int l = (int)(t / nm), i = (int)(t % nm);
  const unsigned q = moving[i];
  const cx s0 = cmul(segph[(long long)l * nv + q], x[q]);
  cx* d = du + (long long)l * nf;
  const int fq = fpos[q];
  atomicAdd(&d[fq].x, -s0.x); atomicAdd(&d[fq].y, -s0.y);
  const cx s = has_ph ? cmul(ph[(long long)l * nm + i], s0) : s0;
  const int c[3] = {(int)(q % nx), (int)((q / nx) % ny), (int)(q / ((unsigned)nx * ny))};
  const float* dd = disp + ((long long)l * nm + i) * 3;
  int base[3]; float w[3][6];
  for (int a = 0; a < 3; ++a) { const float fl = floorf(dd[a]); base[a] = c[a] + (int)fl + 1 - taps / 2; wweights(taps, dd[a] - fl, w[a]); }
  for (int iz = 0; iz < taps; ++iz) {
    const int z = base[2] + iz; if (z < 0 || z >= nz) continue;
    for (int iy = 0; iy < taps; ++iy) {
      const int y = base[1] + iy; if (y < 0 || y >= ny) continue;
      const float wzy = w[2][iz] * w[1][iy];
      for (int ix = 0; ix < taps; ++ix) {
        const int xx = base[0] + ix; if (xx < 0 || xx >= nx) continue;
        const float ww = wzy * w[0][ix];
        const int f = fpos[xx + (long long)nx * (y + (long long)ny * z)];
        atomicAdd(&d[f].x, ww * s.x); atomicAdd(&d[f].y, ww * s.y);
      }
    }
  }
}
// thread per footprint point: grid[gi][c nseg + l] += fac S_c du[l]
extern "C" __global__ void warp_scatter(const cx* __restrict__ du, const unsigned* __restrict__ foot, const cx* __restrict__ smaps, cx* __restrict__ grid,
                                        const float* __restrict__ i0v, const float* __restrict__ i1v, const float* __restrict__ i2v,
                                        int c0, int ncb, int nch, int nseg, int nf, MDims m, Dims d) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nf) return;
  const size_t nv = (size_t)m.N0 * m.N1 * m.N2;
  const unsigned q = foot[t];
  const int a0 = q % m.N0, a1 = (q / m.N0) % m.N1, a2 = q / ((unsigned)m.N0 * m.N1);
  const float fac = i0v[a0] * i1v[a1] * i2v[a2];
  const size_t gi = ((size_t)wrapi(a2 - m.N2 / 2, d.nf2) * d.nf1 + wrapi(a1 - m.N1 / 2, d.nf1)) * d.nf0 + wrapi(a0 - m.N0 / 2, d.nf0);
  cx* g = grid + gi * nch;
  for (int c = 0; c < ncb; ++c) {
    cx s = smaps[(size_t)(c0 + c) * nv + q]; s.x *= fac; s.y *= fac;
    for (int l = 0; l < nseg; ++l) { const cx v = cmul(s, du[(size_t)l * nf + t]); g[c * nseg + l].x += v.x; g[c * nseg + l].y += v.y; }
  }
}
// thread per footprint point: du[l] = fac sum_c conj(S_c) grid[gi][c nseg + l]
extern "C" __global__ void warp_gather(const cx* __restrict__ grid, const unsigned* __restrict__ foot, const cx* __restrict__ smaps, cx* __restrict__ du,
                                       const float* __restrict__ i0v, const float* __restrict__ i1v, const float* __restrict__ i2v,
                                       int c0, int ncb, int nch, int nseg, int nf, int accumulate, MDims m, Dims d) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nf) return;
  const size_t nv = (size_t)m.N0 * m.N1 * m.N2;
  const unsigned q = foot[t];
  const int a0 = q % m.N0, a1 = (q / m.N0) % m.N1, a2 = q / ((unsigned)m.N0 * m.N1);
  const float fac = i0v[a0] * i1v[a1] * i2v[a2];
  const size_t gi = ((size_t)wrapi(a2 - m.N2 / 2, d.nf2) * d.nf1 + wrapi(a1 - m.N1 / 2, d.nf1)) * d.nf0 + wrapi(a0 - m.N0 / 2, d.nf0);
  const cx* g = grid + gi * nch;
  for (int l = 0; l < nseg; ++l) {
    cx acc = cx{0.f, 0.f};
    for (int c = 0; c < ncb; ++c) { const cx v = cmulc(smaps[(size_t)(c0 + c) * nv + q], g[c * nseg + l]); acc.x += v.x; acc.y += v.y; }
    acc.x *= fac; acc.y *= fac;
    if (accumulate) { const cx o = du[(size_t)l * nf + t]; acc.x += o.x; acc.y += o.y; }
    du[(size_t)l * nf + t] = acc;
  }
}
// thread per moving voxel: xbar[v] += sum_l conj(segph[l][v]) (sum_taps w du[l][tap] - du[l][v])
extern "C" __global__ void warp_apply_adj(const cx* __restrict__ du, const cx* __restrict__ segph, const unsigned* __restrict__ moving, const float* __restrict__ disp,
                                          const int* __restrict__ fpos, cx* __restrict__ xbar, const cx* __restrict__ ph, int has_ph, long long nv, int nm, int nf, int nseg, int taps, int nx, int ny, int nz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nm) return;
  const unsigned q = moving[i];
  const int c[3] = {(int)(q % nx), (int)((q / nx) % ny), (int)(q / ((unsigned)nx * ny))};
  cx tot = cx{0.f, 0.f};
  for (int l = 0; l < nseg; ++l) {
    const cx* d = du + (long long)l * nf;
    cx acc = cx{0.f, 0.f};
    const float* dd = disp + ((long long)l * nm + i) * 3;
    int base[3]; float w[3][6];
    for (int a = 0; a < 3; ++a) { const float fl = floorf(dd[a]); base[a] = c[a] + (int)fl + 1 - taps / 2; wweights(taps, dd[a] - fl, w[a]); }
    for (int iz = 0; iz < taps; ++iz) {
      const int z = base[2] + iz; if (z < 0 || z >= nz) continue;
      for (int iy = 0; iy < taps; ++iy) {
        const int y = base[1] + iy; if (y < 0 || y >= ny) continue;
        const float wzy = w[2][iz] * w[1][iy];
        for (int ix = 0; ix < taps; ++ix) {
          const int xx = base[0] + ix; if (xx < 0 || xx >= nx) continue;
          const float ww = wzy * w[0][ix];
          const cx v = d[fpos[xx + (long long)nx * (y + (long long)ny * z)]];
          acc.x += ww * v.x; acc.y += ww * v.y;
        }
      }
    }
    if (has_ph) acc = cmulc(ph[(long long)l * nm + i], acc);
    const cx dq = d[fpos[q]];
    acc.x -= dq.x; acc.y -= dq.y;
    const cx v = cmulc(segph[(long long)l * nv + q], acc);
    tot.x += v.x; tot.y += v.y;
  }
  xbar[q].x += tot.x; xbar[q].y += tot.y;
}
"#;

pub(crate) struct WarpFuncs {
    build: CudaFunction,
    scatter: CudaFunction,
    gather: CudaFunction,
    apply: CudaFunction,
}

impl WarpFuncs {
    pub(crate) fn new(dev: &Device) -> Result<Self, NufftError> {
        let m = dev.load_module("hasty_seg_warp_v3", WARP_SRC, &CompileSettings::default())?;
        Ok(Self { build: m.load_function("warp_build")?, scatter: m.load_function("warp_scatter")?, gather: m.load_function("warp_gather")?, apply: m.load_function("warp_apply_adj")? })
    }
}

/// one frame's warp on the device
pub(crate) struct WarpDev {
    nseg: usize,
    nm: usize,
    nf: usize,
    taps: i32,
    dims: [usize; 3],
    moving: CudaSlice<u32>,
    disp: CudaSlice<f32>,
    fpos: CudaSlice<i32>,
    foot: CudaSlice<u32>,
    du: CudaSlice<C32>,
    /// per (segment, moving voxel) factor of the pushed value (length 1 and unused without a phase)
    ph: CudaSlice<C32>,
    has_ph: i32,
}

fn lc1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(128).max(1), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 }
}

impl SegWarp {
    pub(crate) fn upload(&self, st: &Arc<CudaStream>) -> Result<WarpDev, NufftError> {
        let nv = self.dims.iter().product::<usize>();
        let mut fpos = vec![-1i32; nv];
        for (i, &q) in self.footprint.iter().enumerate() {
            fpos[q as usize] = i as i32;
        }
        let disp: Vec<f32> = self.disp.iter().flat_map(|d| d.iter().copied()).collect();
        Ok(WarpDev {
            nseg: self.nseg,
            nm: self.moving.len(),
            nf: self.footprint.len(),
            taps: self.stencil.taps() as i32,
            dims: self.dims,
            moving: st.clone_htod(&self.moving)?,
            disp: st.clone_htod(&disp)?,
            fpos: st.clone_htod(&fpos)?,
            foot: st.clone_htod(&self.footprint)?,
            du: st.alloc_zeros::<C32>(self.nseg * self.footprint.len().max(1))?,
            ph: match &self.phase {
                Some(p) => st.clone_htod(p)?,
                None => st.alloc_zeros::<C32>(1)?,
            },
            has_ph: self.phase.is_some() as i32,
        })
    }
}

impl WarpDev {
    /// Δu ← (W − I)(D' x) for the frame image x (nv) and the segment diagonals
    pub(crate) fn build<X: cudarc::driver::DevicePtr<C32>>(&mut self, f: &WarpFuncs, st: &Arc<CudaStream>, segph: &CudaSlice<C32>, x: &X) -> Result<(), NufftError> {
        st.memset_zeros(&mut self.du)?;
        if self.nm == 0 {
            return Ok(());
        }
        let nv = self.dims.iter().product::<usize>() as i64;
        let (nm, nf, nseg, taps, nx, ny, nz) = (self.nm as i32, self.nf as i32, self.nseg as i32, self.taps, self.dims[0] as i32, self.dims[1] as i32, self.dims[2] as i32);
        let (xp, _g) = x.device_ptr(st);
        unsafe {
            st.launch_builder(&f.build).arg(&xp).arg(segph).arg(&self.moving).arg(&self.disp).arg(&self.fpos).arg(&mut self.du).arg(&self.ph).arg(&self.has_ph).arg(&nv).arg(&nm).arg(&nf).arg(&nseg).arg(&taps).arg(&nx).arg(&ny).arg(&nz).launch(lc1(self.nm * self.nseg))?;
        }
        Ok(())
    }

    /// grid slot (nch channels per point) += S_c fac Δu for coils c0 .. c0 + ncb
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scatter(&self, f: &WarpFuncs, st: &Arc<CudaStream>, smaps: &CudaSlice<C32>, grid: &mut CudaSlice<C32>, invphi: [&CudaSlice<f32>; 3], c0: usize, ncb: usize, nch: usize, m: MDims, d: Dims) -> Result<(), NufftError> {
        if self.nm == 0 {
            return Ok(());
        }
        let (c0, ncb, nch, nseg, nf) = (c0 as i32, ncb as i32, nch as i32, self.nseg as i32, self.nf as i32);
        unsafe {
            st.launch_builder(&f.scatter).arg(&self.du).arg(&self.foot).arg(smaps).arg(grid).arg(invphi[0]).arg(invphi[1]).arg(invphi[2]).arg(&c0).arg(&ncb).arg(&nch).arg(&nseg).arg(&nf).arg(&m).arg(&d).launch(lc1(self.nf))?;
        }
        Ok(())
    }

    pub(crate) fn zero(&mut self, st: &Arc<CudaStream>) -> Result<(), NufftError> {
        st.memset_zeros(&mut self.du)?;
        Ok(())
    }

    /// Δū += gather of the grid slot for coils c0 .. c0 + ncb (adjoint of `scatter`; Δū zeroed by `zero`)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather(&mut self, f: &WarpFuncs, st: &Arc<CudaStream>, smaps: &CudaSlice<C32>, grid: &CudaSlice<C32>, invphi: [&CudaSlice<f32>; 3], c0: usize, ncb: usize, nch: usize, m: MDims, d: Dims) -> Result<(), NufftError> {
        if self.nm == 0 {
            return Ok(());
        }
        let (c0i, ncbi, nchi, nseg, nf, acc) = (c0 as i32, ncb as i32, nch as i32, self.nseg as i32, self.nf as i32, 1i32);
        unsafe {
            st.launch_builder(&f.gather).arg(grid).arg(&self.foot).arg(smaps).arg(&mut self.du).arg(invphi[0]).arg(invphi[1]).arg(invphi[2]).arg(&c0i).arg(&ncbi).arg(&nchi).arg(&nseg).arg(&nf).arg(&acc).arg(&m).arg(&d).launch(lc1(self.nf))?;
        }
        Ok(())
    }

    /// x̄ += (W − I)ᵀ conj(D') Δū (adjoint of `build`), after every gather of the adjoint
    pub(crate) fn apply_adjoint<X: cudarc::driver::DevicePtrMut<C32>>(&mut self, f: &WarpFuncs, st: &Arc<CudaStream>, segph: &CudaSlice<C32>, xbar: &mut X) -> Result<(), NufftError> {
        if self.nm == 0 {
            return Ok(());
        }
        let nv = self.dims.iter().product::<usize>() as i64;
        let (nm, nf, nseg, taps, nx, ny, nz) = (self.nm as i32, self.nf as i32, self.nseg as i32, self.taps, self.dims[0] as i32, self.dims[1] as i32, self.dims[2] as i32);
        let (xp, _g) = xbar.device_ptr_mut(st);
        unsafe {
            st.launch_builder(&f.apply).arg(&self.du).arg(segph).arg(&self.moving).arg(&self.disp).arg(&self.fpos).arg(&xp).arg(&self.ph).arg(&self.has_ph).arg(&nv).arg(&nm).arg(&nf).arg(&nseg).arg(&taps).arg(&nx).arg(&ny).arg(&nz).launch(lc1(self.nm))?;
        }
        Ok(())
    }
}
