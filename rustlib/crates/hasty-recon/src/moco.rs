//! Motion-compensated sample sets: one reference-pose image per set, seen by every breathing bin through its warp,
//!
//!   y_{t,b} = E_{t,b} W_b x_t,   (W_b x)(q) = x(q − d_b(q))   (trilinear pull-back, d_b = r_b S in voxels),
//!
//! so all bins' data inform the same image (the low-parameter diffeomorphism prior of the dynamic maps, see
//! `motion`). The adjoint W_bᵀ scatters with the same weights (exact adjoint, atomics). A set's data are the bins'
//! blocks back to back, each coil-major (nc × m_{t,b}).

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};

use crate::gpu::Kernels;
use crate::op::SenseOp;
use crate::setop::SetOp;
use crate::Error;

const SRC: &str = r#"
__device__ inline void corner(float p, int n, int& i0, float& f) {
    p = fminf(fmaxf(p, 0.f), (float)(n - 1));
    i0 = min((int)floorf(p), n - 2);
    f = p - i0;
}
extern "C" __global__ void warp_pull(const float2* __restrict__ x, const float* __restrict__ d, float2* __restrict__ out, int nx, int ny, int nz)
{
    const long long nv = (long long)nx * ny * nz;
    const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nv) return;
    const int i = q % nx, j = (q / nx) % ny, k = q / ((long long)nx * ny);
    int i0, j0, k0; float fx, fy, fz;
    corner(i - d[q], nx, i0, fx); corner(j - d[nv + q], ny, j0, fy); corner(k - d[2 * nv + q], nz, k0, fz);
    float2 acc = make_float2(0.f, 0.f);
    for (int c = 0; c < 8; c++) {
        const int di = c & 1, dj = (c >> 1) & 1, dk = c >> 2;
        const float w = (di ? fx : 1.f - fx) * (dj ? fy : 1.f - fy) * (dk ? fz : 1.f - fz);
        const float2 v = x[(i0 + di) + (long long)nx * ((j0 + dj) + (long long)ny * (k0 + dk))];
        acc.x += w * v.x; acc.y += w * v.y;
    }
    out[q] = acc;
}
extern "C" __global__ void warp_push(const float2* __restrict__ y, const float* __restrict__ d, float2* __restrict__ out, int nx, int ny, int nz)
{
    const long long nv = (long long)nx * ny * nz;
    const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nv) return;
    const int i = q % nx, j = (q / nx) % ny, k = q / ((long long)nx * ny);
    int i0, j0, k0; float fx, fy, fz;
    corner(i - d[q], nx, i0, fx); corner(j - d[nv + q], ny, j0, fy); corner(k - d[2 * nv + q], nz, k0, fz);
    const float2 v = y[q];
    for (int c = 0; c < 8; c++) {
        const int di = c & 1, dj = (c >> 1) & 1, dk = c >> 2;
        const float w = (di ? fx : 1.f - fx) * (dj ? fy : 1.f - fy) * (dk ? fz : 1.f - fz);
        float* o = (float*)&out[(i0 + di) + (long long)nx * ((j0 + dj) + (long long)ny * (k0 + dk))];
        atomicAdd(o, w * v.x); atomicAdd(o + 1, w * v.y);
    }
}
"#;

pub struct Warp {
    pull: CudaFunction,
    push: CudaFunction,
}

impl Warp {
    pub fn new(dev: &Device) -> Result<Self, Error> {
        let m = dev.load_module("recon_moco_warp_v2", SRC, &CompileSettings::default())?;
        Ok(Self { pull: m.load_function("warp_pull")?, push: m.load_function("warp_push")? })
    }
    fn cfg(nv: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: ((nv as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }
    /// out = W x (d: [3][nv] displacement in voxels)
    pub fn pull(&self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, d: &CudaSlice<f32>, out: &mut CudaSlice<C32>, dims: [usize; 3]) -> Result<(), Error> {
        let (nx, ny, nz) = (dims[0] as i32, dims[1] as i32, dims[2] as i32);
        unsafe { st.launch_builder(&self.pull).arg(x).arg(d).arg(out).arg(&nx).arg(&ny).arg(&nz).launch(Self::cfg(dims[0] * dims[1] * dims[2]))? };
        Ok(())
    }
    /// out += Wᵀ y
    pub fn push(&self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, d: &CudaSlice<f32>, out: &mut CudaSlice<C32>, dims: [usize; 3]) -> Result<(), Error> {
        let (nx, ny, nz) = (dims[0] as i32, dims[1] as i32, dims[2] as i32);
        unsafe { st.launch_builder(&self.push).arg(y).arg(d).arg(out).arg(&nx).arg(&ny).arg(&nz).launch(Self::cfg(dims[0] * dims[1] * dims[2]))? };
        Ok(())
    }
}

/// reference-pose sets observed through per-bin warps
pub struct MocoSets<'a> {
    pub op: &'a mut SenseOp,
    pub warp: Warp,
    /// [t][b] coordinates and sample counts
    pub coords: Vec<Vec<[CudaSlice<f32>; 3]>>,
    pub ms: Vec<Vec<usize>>,
    /// per bin displacement [3][nv] (voxels)
    pub disp: Vec<CudaSlice<f32>>,
    tmp_x: CudaSlice<C32>,
    tmp_x2: CudaSlice<C32>,
    tmp_y: CudaSlice<C32>,
}

impl<'a> MocoSets<'a> {
    pub fn new(st: &Arc<CudaStream>, dev: &Device, op: &'a mut SenseOp, coords: Vec<Vec<[CudaSlice<f32>; 3]>>, ms: Vec<Vec<usize>>, disp: Vec<CudaSlice<f32>>) -> Result<Self, Error> {
        let mmax = ms.iter().flatten().copied().max().unwrap_or(1);
        let (nv, nc) = (op.nv, op.nc);
        Ok(Self { warp: Warp::new(dev)?, tmp_x: st.alloc_zeros(nv)?, tmp_x2: st.alloc_zeros(nv)?, tmp_y: st.alloc_zeros(nc * mmax)?, op, coords, ms, disp })
    }
}

impl SetOp for MocoSets<'_> {
    fn nsets(&self) -> usize {
        self.ms.len()
    }
    fn len_x(&self, _t: usize) -> usize {
        self.op.nv
    }
    fn len_y(&self, t: usize) -> usize {
        self.op.nc * self.ms[t].iter().sum::<usize>()
    }
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let dims = self.op.dims;
        let nc = self.op.nc;
        let mut off = 0;
        for b in 0..self.ms[t].len() {
            let m = self.ms[t][b];
            if m == 0 {
                continue;
            }
            self.warp.pull(st, x, &self.disp[b], &mut self.tmp_x, dims)?;
            let c = &self.coords[t][b];
            self.op.set_points(st, [&c[0], &c[1], &c[2]], m)?;
            self.op.forward(st, &self.tmp_x, &mut self.tmp_y)?;
            st.memcpy_dtod(&self.tmp_y.slice(0..nc * m), &mut y.slice_mut(off..off + nc * m))?;
            off += nc * m;
        }
        Ok(())
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let dims = self.op.dims;
        let nc = self.op.nc;
        st.memset_zeros(x)?;
        let mut off = 0;
        for b in 0..self.ms[t].len() {
            let m = self.ms[t][b];
            if m == 0 {
                continue;
            }
            st.memcpy_dtod(&y.slice(off..off + nc * m), &mut self.tmp_y.slice_mut(0..nc * m))?;
            let c = &self.coords[t][b];
            self.op.set_points(st, [&c[0], &c[1], &c[2]], m)?;
            self.op.adjoint(st, &self.tmp_y, &mut self.tmp_x2)?;
            self.warp.push(st, &self.tmp_x2, &self.disp[b], x, dims)?;
            off += nc * m;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moco_term_sets_forward_and_adjoint_agree() {
        use crate::gpu::Kernels;
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let kern = Arc::new(Kernels::new(&dev).unwrap());
        let dims = [12usize, 10, 8];
        let nv = dims[0] * dims[1] * dims[2];
        let (nc, l, ng, nt) = (2usize, 3usize, 2usize, 2usize);
        let mut s = 11u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        let smaps: Vec<C32> = (0..nc * nv).map(|_| C32::new(1.0 + u(), u())).collect();
        let ms = vec![vec![70usize, 50], vec![40, 60]];
        let mk_c = |m: usize, u: &mut dyn FnMut() -> f32| -> [Vec<f32>; 3] { [0, 1, 2].map(|_| (0..m).map(|_| 6.0 * u()).collect()) };
        let hc: Vec<Vec<[Vec<f32>; 3]>> = (0..ng).map(|b| (0..nt).map(|t| mk_c(ms[b][t], &mut u)).collect()).collect();
        let mut op = SenseOp::new(&dev, &st, kern, dims, [&hc[0][0][0], &hc[0][0][1], &hc[0][0][2]], &smaps, nc, vec![], 1e-6).unwrap();
        let coords = hc.iter().map(|r| r.iter().map(|c| [0, 1, 2].map(|a| st.clone_htod(&c[a]).unwrap())).collect()).collect();
        let w = (0..ng).map(|b| (0..nt).map(|t| (0..l).map(|_| st.clone_htod(&(0..ms[b][t]).map(|_| C32::new(u(), u())).collect::<Vec<_>>()).unwrap()).collect()).collect()).collect();
        let disp = (0..ng).map(|_| st.clone_htod(&(0..3 * nv).map(|_| 2.0 * u()).collect::<Vec<_>>()).unwrap()).collect();
        let mut sets = MocoTermSets::new(&st, &dev, &mut op, coords, ms.clone(), w, disp).unwrap();
        for t in 0..nt {
            let x: Vec<C32> = (0..l * nv).map(|_| C32::new(u(), u())).collect();
            let ny = sets.len_y(t);
            let y: Vec<C32> = (0..ny).map(|_| C32::new(u(), u())).collect();
            let (dx, dy) = (st.clone_htod(&x).unwrap(), st.clone_htod(&y).unwrap());
            let mut ax = st.alloc_zeros::<C32>(ny).unwrap();
            let mut aty = st.alloc_zeros::<C32>(l * nv).unwrap();
            sets.forward(&st, t, &dx, &mut ax).unwrap();
            sets.adjoint(&st, t, &dy, &mut aty).unwrap();
            let (ax, aty) = (st.clone_dtoh(&ax).unwrap(), st.clone_dtoh(&aty).unwrap());
            let ip = |a: &[C32], b: &[C32]| a.iter().zip(b).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>();
            let (lh, rh) = (ip(&ax, &y), ip(&x, &aty));
            assert!((lh - rh).abs() < 1e-3 * lh.abs().max(1.0), "set {t}: <Ax, y> {lh} vs <x, Aᴴy> {rh}");
        }
    }

    #[test]
    fn push_is_the_adjoint_of_pull() {
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let w = Warp::new(&dev).unwrap();
        let dims = [17usize, 13, 9];
        let nv = dims[0] * dims[1] * dims[2];
        let mut s = 7u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };
        let x: Vec<C32> = (0..nv).map(|_| C32::new(u(), u())).collect();
        let y: Vec<C32> = (0..nv).map(|_| C32::new(u(), u())).collect();
        let d: Vec<f32> = (0..3 * nv).map(|_| 3.0 * u()).collect();
        let (dx, dy, dd) = (st.clone_htod(&x).unwrap(), st.clone_htod(&y).unwrap(), st.clone_htod(&d).unwrap());
        let mut wx = st.alloc_zeros::<C32>(nv).unwrap();
        let mut wty = st.alloc_zeros::<C32>(nv).unwrap();
        w.pull(&st, &dx, &dd, &mut wx, dims).unwrap();
        w.push(&st, &dy, &dd, &mut wty, dims).unwrap();
        let (wx, wty) = (st.clone_dtoh(&wx).unwrap(), st.clone_dtoh(&wty).unwrap());
        let ip = |a: &[C32], b: &[C32]| a.iter().zip(b).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>();
        let (l, r) = (ip(&wx, &y), ip(&x, &wty));
        assert!((l - r).abs() < 1e-4 * l.abs().max(1.0), "<Wx, y> {l} vs <x, Wᵀy> {r}");
        // zero displacement is the identity
        let dz = st.alloc_zeros::<f32>(3 * nv).unwrap();
        let mut id = st.alloc_zeros::<C32>(nv).unwrap();
        w.pull(&st, &dx, &dz, &mut id, dims).unwrap();
        let id = st.clone_dtoh(&id).unwrap();
        assert!(id.iter().zip(&x).all(|(a, b)| (a.re - b.re).abs() < 1e-6 && (a.im - b.im).abs() < 1e-6));
    }
}

/// reference-pose sets with node images ([L][nv] per set, see `multiterm`) observed by groups of readouts (motion
/// states), each through its warp: block (b, t) = Σ_l a_{b,t,l} ⊙ E_{b,t} W_b x_{t,l}. Data of set t: the groups'
/// blocks back to back, each coil-major.
pub struct MocoTermSets<'a> {
    pub op: &'a mut SenseOp,
    pub warp: Warp,
    /// [b][t] coordinates, sample counts and per-node weights [b][t][l]
    pub coords: Vec<Vec<[CudaSlice<f32>; 3]>>,
    pub ms: Vec<Vec<usize>>,
    pub w: Vec<Vec<Vec<CudaSlice<C32>>>>,
    /// per group displacement [3][nv] (voxels)
    pub disp: Vec<CudaSlice<f32>>,
    pub l: usize,
    pub nset: usize,
    tmp_x: CudaSlice<C32>,
    tmp_y: CudaSlice<C32>,
    tmp_y2: CudaSlice<C32>,
    acc: Vec<CudaSlice<C32>>,
}

impl<'a> MocoTermSets<'a> {
    pub fn new(st: &Arc<CudaStream>, dev: &Device, op: &'a mut SenseOp, coords: Vec<Vec<[CudaSlice<f32>; 3]>>, ms: Vec<Vec<usize>>, w: Vec<Vec<Vec<CudaSlice<C32>>>>, disp: Vec<CudaSlice<f32>>) -> Result<Self, Error> {
        let mmax = ms.iter().flatten().copied().max().unwrap_or(1);
        let (nv, nc) = (op.nv, op.nc);
        let l = w[0][0].len();
        let nset = ms[0].len();
        let acc = (0..l).map(|_| st.alloc_zeros(nv)).collect::<Result<_, _>>()?;
        Ok(Self { warp: Warp::new(dev)?, tmp_x: st.alloc_zeros(nv)?, tmp_y: st.alloc_zeros(nc * mmax)?, tmp_y2: st.alloc_zeros(nc * mmax)?, acc, op, coords, ms, w, disp, l, nset })
    }
}

impl SetOp for MocoTermSets<'_> {
    fn nsets(&self) -> usize {
        self.nset
    }
    fn len_x(&self, _t: usize) -> usize {
        self.l * self.op.nv
    }
    fn len_y(&self, t: usize) -> usize {
        self.op.nc * self.ms.iter().map(|m| m[t]).sum::<usize>()
    }
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (dims, nv, nc) = (self.op.dims, self.op.nv, self.op.nc);
        let mut off = 0;
        for b in 0..self.ms.len() {
            let m = self.ms[b][t];
            if m == 0 {
                continue;
            }
            let c = &self.coords[b][t];
            self.op.set_points(st, [&c[0], &c[1], &c[2]], m)?;
            for l in 0..self.l {
                let xl = x.slice(l * nv..(l + 1) * nv);
                st.memcpy_dtod(&xl, &mut self.acc[0])?; // acc[0] as scratch for the node image
                self.warp.pull(st, &self.acc[0], &self.disp[b], &mut self.tmp_x, dims)?;
                self.op.forward(st, &self.tmp_x, &mut self.tmp_y)?;
                self.op.kernels().sample_weight(st, &self.tmp_y, &self.w[b][t][l], &mut self.tmp_y2, m, nc, false, l > 0)?;
            }
            st.memcpy_dtod(&self.tmp_y2.slice(0..nc * m), &mut y.slice_mut(off..off + nc * m))?;
            off += nc * m;
        }
        Ok(())
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (dims, nv, nc) = (self.op.dims, self.op.nv, self.op.nc);
        for a in self.acc.iter_mut() {
            st.memset_zeros(a)?;
        }
        let mut off = 0;
        for b in 0..self.ms.len() {
            let m = self.ms[b][t];
            if m == 0 {
                continue;
            }
            st.memcpy_dtod(&y.slice(off..off + nc * m), &mut self.tmp_y2.slice_mut(0..nc * m))?;
            let c = &self.coords[b][t];
            self.op.set_points(st, [&c[0], &c[1], &c[2]], m)?;
            for l in 0..self.l {
                self.op.kernels().sample_weight(st, &self.tmp_y2, &self.w[b][t][l], &mut self.tmp_y, m, nc, true, false)?;
                self.op.adjoint(st, &self.tmp_y, &mut self.tmp_x)?;
                self.warp.push(st, &self.tmp_x, &self.disp[b], &mut self.acc[l], dims)?;
            }
            off += nc * m;
        }
        for l in 0..self.l {
            st.memcpy_dtod(&self.acc[l], &mut x.slice_mut(l * nv..(l + 1) * nv))?;
        }
        Ok(())
    }
}
