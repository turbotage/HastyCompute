//! GPU engine of the smoothed wavelet prior [`crate::knotncg::WavHuber`] (kernels: `wavgpu.cu`): the zero-padded
//! orthonormal db4 3-D transform of knot stacks and the Huber reductions over its detail coefficients, so that the
//! knot-space solvers keep x, the direction and the coefficient stacks on the device (`knotncg`, device mode).
//!
//! Same conventions as the host path (`sepwt.rs` + `WavHuber`): per image the box p = ceil(d / 2^L) 2^L per axis, zero
//! padding, x fastest, complex = two interleaved f32 lanes (a `C32` per coefficient), in-place Mallat layout, periodic
//! boundary. A transform level is one pass per axis (x: whole rows through shared memory; y, z: tiles of 32 consecutive
//! x times the whole line, so every global access is a coalesced row segment), ≈ 6 reads + writes of the box per
//! stack transform. Coefficient stacks are `CudaSlice<C32>` of k · npc (npc = p0 p1 p2), so the generic `Kernels`
//! (axpby, dot) apply to them.
//!
//! Reductions (values, line sums) accumulate per coefficient in f64 as the host does, through fixed per-block partials
//! and a second fixed-order pass (deterministic). [`WavGpu::line`] evaluates up to [`MAXM`] step values per read of the
//! two stacks.
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};

use crate::knotncg::WavHuber;
use crate::Error;

/// step values per line sweep
pub const MAXM: usize = 8;
/// reduction blocks per image
const NBX: usize = 64;
const NT: u32 = 256;
/// shared memory per block without opt-in
const SMEM: usize = 48 * 1024;
const C: u64 = std::mem::size_of::<C32>() as u64;

pub struct WavGpu {
    f_x: CudaFunction,
    f_yz: CudaFunction,
    f_pad: CudaFunction,
    f_crop: CudaFunction,
    f_psi: CudaFunction,
    f_line: CudaFunction,
    f_sum: CudaFunction,
    f_soft: CudaFunction,
    f_admm: CudaFunction,
    /// per-block partials (NBX · kmax · 2 MAXM) and the reduced values (2 MAXM)
    part: CudaSlice<f64>,
    out: CudaSlice<f64>,
    kmax: usize,
}

fn dp<T>(s: &CudaSlice<T>, st: &Arc<CudaStream>) -> u64 {
    s.device_ptr(st).0
}

fn dpm<T>(s: &mut CudaSlice<T>, st: &Arc<CudaStream>) -> u64 {
    s.device_ptr_mut(st).0
}

/// pointwise launch over nk images of n elements each (grid-stride along x, image along y)
fn pcfg(n: usize, nk: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(NT as usize).clamp(1, 512) as u32, nk as u32, 1), block_dim: (NT, 1, 1), shared_mem_bytes: 0 }
}

impl WavGpu {
    /// `kmax`: the most images a single reduction covers (the knot count of the stacks)
    pub fn new(dev: &Device, st: &Arc<CudaStream>, kmax: usize) -> Result<Self, Error> {
        let m = dev.load_module("recon_wavgpu", include_str!("wavgpu.cu"), &CompileSettings::default())?;
        let f = |n: &str| m.load_function(n);
        Ok(Self {
            f_x: f("wt_x")?,
            f_yz: f("wt_yz")?,
            f_pad: f("pad_zero")?,
            f_crop: f("crop_axpby")?,
            f_psi: f("huber_psi")?,
            f_line: f("huber_line")?,
            f_sum: f("sum_parts")?,
            f_soft: f("huber_soft")?,
            f_admm: f("admm_wav")?,
            part: st.alloc_zeros(NBX * kmax.max(1) * 2 * MAXM)?,
            out: st.alloc_zeros(2 * MAXM)?,
            kmax: kmax.max(1),
        })
    }

    /// complex coefficients per image (the padded box)
    pub fn npc(w: &WavHuber) -> usize {
        w.p.iter().product()
    }

    /// device bytes of the engine itself (partials; kernels and the module are small)
    pub fn bytes(kmax: usize) -> usize {
        (NBX * kmax.max(1) + 1) * 2 * MAXM * 8
    }

    fn geom(w: &WavHuber) -> (i32, i32, i32, i32, i32, i64) {
        let a = w.p.map(|v| v >> w.levels);
        (w.p[0] as i32, w.p[1] as i32, a[0] as i32, a[1] as i32, a[2] as i32, Self::npc(w) as i64)
    }

    // ---- raw building blocks (device addresses of the first image; nk consecutive images)

    /// box ← zero-padded images
    unsafe fn pad(&self, st: &Arc<CudaStream>, w: &WavHuber, img: u64, bx: u64, nk: usize) -> Result<(), Error> {
        let (d, p) = (w.dims, w.p);
        let (np, nv) = (Self::npc(w) as i64, (d[0] * d[1] * d[2]) as i64);
        let (d0, d1, d2, p0, p1) = (d[0] as i32, d[1] as i32, d[2] as i32, p[0] as i32, p[1] as i32);
        st.launch_builder(&self.f_pad).arg(&img).arg(&bx).arg(&d0).arg(&d1).arg(&d2).arg(&p0).arg(&p1).arg(&nv).arg(&np).launch(pcfg(np as usize, nk))?;
        Ok(())
    }

    /// out = a · crop(box) + b · out
    unsafe fn crop(&self, st: &Arc<CudaStream>, w: &WavHuber, bx: u64, out: u64, nk: usize, a: f32, b: f32) -> Result<(), Error> {
        let (d, p) = (w.dims, w.p);
        let (np, nv) = (Self::npc(w) as i64, (d[0] * d[1] * d[2]) as i64);
        let (d0, d1, p0, p1) = (d[0] as i32, d[1] as i32, p[0] as i32, p[1] as i32);
        st.launch_builder(&self.f_crop).arg(&bx).arg(&out).arg(&d0).arg(&d1).arg(&p0).arg(&p1).arg(&nv).arg(&np).arg(&a).arg(&b).launch(pcfg(nv as usize, nk))?;
        Ok(())
    }

    /// the forward (analysis) or inverse transform of nk boxes in place (`bx`: a device address of nk boxes)
    fn transform(&self, st: &Arc<CudaStream>, w: &WavHuber, bx: u64, nk: usize, inverse: bool) -> Result<(), Error> {
        let p = w.p;
        let np = Self::npc(w) as i64;
        let inv = inverse as i32;
        let level = |l: usize| -> Result<(), Error> {
            let s = p.map(|d| d >> l);
            assert!(s.iter().all(|&d| d >= 2 && d % 2 == 0), "box {p:?} not divisible by 2^{}", w.levels);
            let xpass = || -> Result<(), Error> {
                let lpb = (512 / s[0]).clamp(1, s[1]);
                let smem = lpb * s[0] * 8;
                assert!(smem <= SMEM, "x line of {} too long for shared memory", s[0]);
                let (p0, p1, s0, s1, lp) = (p[0] as i32, p[1] as i32, s[0] as i32, s[1] as i32, lpb as i32);
                let cfg = LaunchConfig { grid_dim: (s[1].div_ceil(lpb) as u32, s[2] as u32, nk as u32), block_dim: (NT, 1, 1), shared_mem_bytes: smem as u32 };
                unsafe { st.launch_builder(&self.f_x).arg(&bx).arg(&np).arg(&p0).arg(&p1).arg(&s0).arg(&s1).arg(&lp).arg(&inv).launch(cfg)? };
                Ok(())
            };
            // axis 1 (y: element stride p0, other axis z at p0 p1) or 2 (z: element stride p0 p1, other axis y at p0)
            let yzpass = |axis: usize| -> Result<(), Error> {
                let n = s[axis];
                let (es, os, nother) = if axis == 1 { (p[0], p[0] * p[1], s[2]) } else { (p[0] * p[1], p[0], s[1]) };
                let mut tx = 32usize;
                while n * tx * 8 > SMEM && tx > 1 {
                    tx /= 2;
                }
                assert!(n * tx * 8 <= SMEM, "line of {n} too long for shared memory");
                let (s0, ni, esi, osi, txi) = (s[0] as i32, n as i32, es as i64, os as i64, tx as i32);
                let cfg = LaunchConfig { grid_dim: (s[0].div_ceil(tx) as u32, nother as u32, nk as u32), block_dim: (NT, 1, 1), shared_mem_bytes: (n * tx * 8) as u32 };
                unsafe { st.launch_builder(&self.f_yz).arg(&bx).arg(&np).arg(&s0).arg(&ni).arg(&esi).arg(&osi).arg(&txi).arg(&inv).launch(cfg)? };
                Ok(())
            };
            if inverse {
                yzpass(2)?;
                yzpass(1)?;
                xpass()
            } else {
                xpass()?;
                yzpass(1)?;
                yzpass(2)
            }
        };
        if inverse {
            for l in (0..w.levels).rev() {
                level(l)?;
            }
        } else {
            for l in 0..w.levels {
                level(l)?;
            }
        }
        Ok(())
    }

    /// Σ h over nk images at c (and ψ into psi unless psi = 0), added to out[0] if `acc`
    unsafe fn psi(&self, st: &Arc<CudaStream>, w: &WavHuber, c: u64, psi: u64, nk: usize, acc: bool) -> Result<(), Error> {
        assert!(nk <= self.kmax);
        let (p0, p1, a0, a1, a2, np) = Self::geom(w);
        let (th, ig, wr) = (w.theta as f32, 1.0 / w.gamma as f32, (psi != 0) as i32);
        let cfg = LaunchConfig { grid_dim: (NBX as u32, nk as u32, 1), block_dim: (NT, 1, 1), shared_mem_bytes: 0 };
        let part = dp(&self.part, st);
        st.launch_builder(&self.f_psi).arg(&c).arg(&psi).arg(&part).arg(&p0).arg(&p1).arg(&a0).arg(&a1).arg(&a2).arg(&np).arg(&th).arg(&ig).arg(&wr).launch(cfg)?;
        self.sum(st, NBX * nk, 1, acc)
    }

    /// out[0..nval) (+)= Σ_b part[b · nval + j]
    unsafe fn sum(&self, st: &Arc<CudaStream>, nparts: usize, nval: usize, acc: bool) -> Result<(), Error> {
        let (np, nv, a) = (nparts as i32, nval as i32, acc as i32);
        let cfg = LaunchConfig { grid_dim: (nval as u32, 1, 1), block_dim: (NT, 1, 1), shared_mem_bytes: 0 };
        let out = dp(&self.out, st);
        st.launch_builder(&self.f_sum).arg(&self.part).arg(&np).arg(&nv).arg(&out).arg(&a).launch(cfg)?;
        Ok(())
    }

    fn read(&self, st: &Arc<CudaStream>, n: usize) -> Result<Vec<f64>, Error> {
        Ok(st.clone_dtoh(&self.out.slice(0..n))?)
    }

    // ---- stack-level API (k images: x k · nv, coefficient stacks k · npc)

    /// c = W S x
    pub fn coef(&self, st: &Arc<CudaStream>, w: &WavHuber, x: &CudaSlice<C32>, k: usize, c: &mut CudaSlice<C32>) -> Result<(), Error> {
        let nv = w.dims.iter().product::<usize>();
        assert!(x.len() >= k * nv && c.len() >= k * Self::npc(w));
        let (xp, cp) = (dp(x, st), dpm(c, st));
        unsafe {
            self.pad(st, w, xp, cp, k)?;
            self.transform(st, w, cp, k, false)
        }
    }

    /// out = a Sᴴ Wᴴ c + b out (c is consumed: transformed back in place)
    pub fn synth_axpby(&self, st: &Arc<CudaStream>, w: &WavHuber, c: &mut CudaSlice<C32>, k: usize, a: f32, out: &mut CudaSlice<C32>, b: f32) -> Result<(), Error> {
        let nv = w.dims.iter().product::<usize>();
        assert!(out.len() >= k * nv && c.len() >= k * Self::npc(w));
        let (cp, op) = (dpm(c, st), dpm(out, st));
        unsafe {
            self.transform(st, w, cp, k, true)?;
            self.crop(st, w, cp, op, k, a, b)
        }
    }

    /// Σ h(c) over the detail coefficients
    pub fn value(&self, st: &Arc<CudaStream>, w: &WavHuber, c: &CudaSlice<C32>, k: usize) -> Result<f64, Error> {
        assert!(c.len() >= k * Self::npc(w));
        unsafe { self.psi(st, w, dp(c, st), 0, k, false)? };
        Ok(self.read(st, 1)?[0])
    }

    /// Σ h(c) and ψ(c) into `psi` (approximation band 0)
    pub fn value_psi(&self, st: &Arc<CudaStream>, w: &WavHuber, c: &CudaSlice<C32>, psi: &mut CudaSlice<C32>, k: usize) -> Result<f64, Error> {
        assert!(c.len() >= k * Self::npc(w) && psi.len() >= k * Self::npc(w));
        unsafe { self.psi(st, w, dp(c, st), dpm(psi, st), k, false)? };
        Ok(self.read(st, 1)?[0])
    }

    /// Σ h(c), c ← ψ(c)
    pub fn value_psi_inplace(&self, st: &Arc<CudaStream>, w: &WavHuber, c: &mut CudaSlice<C32>, k: usize) -> Result<f64, Error> {
        assert!(c.len() >= k * Self::npc(w));
        let cp = dpm(c, st);
        unsafe { self.psi(st, w, cp, cp, k, false)? };
        Ok(self.read(st, 1)?[0])
    }

    /// along c(a) = cx + a cd: (Σ h(c(a)), Σ Re⟨ψ(c(a)), cd⟩) for every a (MAXM per sweep of the stacks)
    pub fn line(&self, st: &Arc<CudaStream>, w: &WavHuber, cx: &CudaSlice<C32>, cd: &CudaSlice<C32>, k: usize, a: &[f64]) -> Result<Vec<(f64, f64)>, Error> {
        assert!(k <= self.kmax && cx.len() >= k * Self::npc(w) && cd.len() >= k * Self::npc(w));
        let (p0, p1, a0, a1, a2, np) = Self::geom(w);
        let (th, ig) = (w.theta as f32, 1.0 / w.gamma as f32);
        let (xp, dpp) = (dp(cx, st), dp(cd, st));
        let mut res = Vec::with_capacity(a.len());
        for grp in a.chunks(MAXM) {
            let m = grp.len();
            let mut s = [0f32; MAXM];
            for (j, &v) in grp.iter().enumerate() {
                s[j] = v as f32;
            }
            let mi = m as i32;
            let cfg = LaunchConfig { grid_dim: (NBX as u32, k as u32, 1), block_dim: (NT, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                let part = dp(&self.part, st);
                st.launch_builder(&self.f_line)
                    .arg(&xp)
                    .arg(&dpp)
                    .arg(&part)
                    .arg(&p0)
                    .arg(&p1)
                    .arg(&a0)
                    .arg(&a1)
                    .arg(&a2)
                    .arg(&np)
                    .arg(&th)
                    .arg(&ig)
                    .arg(&mi)
                    .arg(&s[0])
                    .arg(&s[1])
                    .arg(&s[2])
                    .arg(&s[3])
                    .arg(&s[4])
                    .arg(&s[5])
                    .arg(&s[6])
                    .arg(&s[7])
                    .launch(cfg)?;
                self.sum(st, NBX * k, 2 * m, false)?;
            }
            let v = self.read(st, 2 * m)?;
            res.extend((0..m).map(|j| (v[2 * j], v[2 * j + 1])));
        }
        Ok(res)
    }

    /// soft threshold of the detail coefficients at `thr`
    pub fn soft(&self, st: &Arc<CudaStream>, w: &WavHuber, c: &mut CudaSlice<C32>, k: usize, thr: f64) -> Result<(), Error> {
        assert!(c.len() >= k * Self::npc(w));
        let (p0, p1, a0, a1, a2, np) = Self::geom(w);
        let (cp, t) = (dpm(c, st), thr as f32);
        unsafe { st.launch_builder(&self.f_soft).arg(&cp).arg(&p0).arg(&p1).arg(&a0).arg(&a1).arg(&a2).arg(&np).arg(&t).launch(pcfg(np as usize, k))? };
        Ok(())
    }

    /// value and gradient of the envelope at x through a scratch of kc = scr.len() / npc images: out = a ∇(x) + b out;
    /// returns Σ h(W S x)
    #[allow(clippy::too_many_arguments)]
    pub fn grad(&self, st: &Arc<CudaStream>, w: &WavHuber, x: &CudaSlice<C32>, k: usize, scr: &mut CudaSlice<C32>, out: &mut CudaSlice<C32>, a: f32, b: f32) -> Result<f64, Error> {
        let (nv, npc) = (w.dims.iter().product::<usize>() as u64, Self::npc(w));
        let kc = (scr.len() / npc).min(self.kmax);
        assert!(kc >= 1, "wavelet scratch smaller than one image");
        assert!(x.len() as u64 >= k as u64 * nv && out.len() as u64 >= k as u64 * nv);
        let (xp, sp, op) = (dp(x, st), dpm(scr, st), dpm(out, st));
        for q0 in (0..k).step_by(kc) {
            let nk = kc.min(k - q0);
            let off = q0 as u64 * nv * C;
            unsafe {
                self.pad(st, w, xp + off, sp, nk)?;
                self.transform(st, w, sp, nk, false)?;
                self.psi(st, w, sp, sp, nk, q0 > 0)?;
                self.transform(st, w, sp, nk, true)?;
                self.crop(st, w, sp, op + off, nk, a, b)?;
            }
        }
        Ok(self.read(st, 1)?[0])
    }

    /// The wavelet split of ADMM (`knotncg::admm`) through a scratch of kc images: t = W S x + v, u = soft_thr(t),
    /// v ← t − u, centre m ← m + s Sᴴ Wᴴ (u − v)
    #[allow(clippy::too_many_arguments)]
    pub fn admm(&self, st: &Arc<CudaStream>, w: &WavHuber, x: &CudaSlice<C32>, k: usize, scr: &mut CudaSlice<C32>, vw: &mut CudaSlice<C32>, thr: f64, m: &mut CudaSlice<C32>, s: f32) -> Result<(), Error> {
        let (nv, npc) = (w.dims.iter().product::<usize>() as u64, Self::npc(w));
        let kc = scr.len() / npc;
        assert!(kc >= 1 && vw.len() >= k * npc);
        let (p0, p1, a0, a1, a2, np) = Self::geom(w);
        let t = thr as f32;
        let (xp, sp, vp, mp) = (dp(x, st), dpm(scr, st), dpm(vw, st), dpm(m, st));
        for q0 in (0..k).step_by(kc) {
            let nk = kc.min(k - q0);
            let off = q0 as u64 * nv * C;
            let vo = vp + q0 as u64 * npc as u64 * C;
            unsafe {
                self.pad(st, w, xp + off, sp, nk)?;
                self.transform(st, w, sp, nk, false)?;
                st.launch_builder(&self.f_admm).arg(&sp).arg(&vo).arg(&p0).arg(&p1).arg(&a0).arg(&a1).arg(&a2).arg(&np).arg(&t).launch(pcfg(np as usize, nk))?;
                self.transform(st, w, sp, nk, true)?;
                self.crop(st, w, sp, mp + off, nk, s, 1.0)?;
            }
        }
        Ok(())
    }
}
