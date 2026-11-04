//! The velocity-model data term as a P²nP [`Objective`] (`research/velocity_model_sr.md` §A4):
//!   f(θ) = ½ Σ_t w_t ‖E_t x_{e,f}(θ) − y_t‖²,   set t = f·E + e (one operator frame of one encode per set)
//! with the gradient by the model's vector-Jacobian product, the pointwise Gauss–Newton metric M ≈ H⁻¹ (z: scalar per
//! voxel, v: 3×3 per voxel = (c·σ_k)⁻¹ (Σ_e k_e k_eᵀ)⁻¹, residual phase: diagonal) for `Precond::Circulant` /
//! `Precond::Zmshr1M`, and λ_max of (M) H by power iteration with Gauss–Newton Hessian-vector products.
//! Set weights w_t implement VENC continuation (low-VENC encodes ramped in).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use num_complex::Complex64;

use crate::gpu::Kernels;
use crate::p2np::{axpby, rdot, Objective, Stack};
use crate::setop::SetOp;
use crate::vmodel::{poly_exps, raw, VelModel};
use crate::Error;

/// k-space data of one set: resident, or in pinned host memory (staged per use)
pub enum YData {
    Dev(CudaSlice<C32>),
    Host(cudarc::driver::PinnedHostSlice<C32>),
}

impl YData {
    pub fn len(&self) -> usize {
        match self {
            YData::Dev(d) => d.len(),
            YData::Host(h) => h.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct VelProblem<'a> {
    pub op: &'a mut dyn SetOp,
    pub ys: Vec<YData>,
    pub model: VelModel,
    /// per-set data weights (VENC continuation; default 1)
    pub weights: Vec<f32>,
    /// recompute the Gauss–Newton metric every this many gradients (it depends on |z|)
    pub metric_every: usize,
    /// relative Levenberg damping of the metric
    pub damping: f64,
    kern: Arc<Kernels>,
    x: CudaSlice<C32>,
    g: CudaSlice<C32>,
    sacc: Vec<CudaSlice<f32>>,
    coil: Option<CudaSlice<f32>>,
    coil_host: Option<Vec<f32>>,
    sig: CudaSlice<f32>,
    zz: CudaSlice<f32>,
    m3: CudaSlice<f32>,
    m3inv: CudaSlice<f32>,
    wsum_z: Vec<f32>,
    damp_z: Vec<f32>,
    damp_v: Vec<f32>,
    diag_a: Vec<f64>,
    calls: usize,
    have_metric: bool,
    /// per-encode sets holding all frames (continuous time in the operator)
    pub temporal: bool,
    /// update z only: the v and residual-phase gradients (and Hessian rows) are zeroed (z warm-up)
    pub freeze_v: bool,
    /// static-tissue anchor: per-voxel weight and the relative strength μ (λ = μ × mean Gauss–Newton curvature of v)
    anchor: Option<(CudaSlice<f32>, f64)>,
    pub anchor_lam: f64,
    v_scale: f64,
    /// mean diagonal of E_tᴴE_t per unit coil energy (absolute scale of the data curvature; 0 = not estimated)
    pub nhat: f64,
}

fn inv3(m: &[f64; 9]) -> [f64; 9] {
    let d = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    [
        (m[4] * m[8] - m[5] * m[7]) / d,
        (m[2] * m[7] - m[1] * m[8]) / d,
        (m[1] * m[5] - m[2] * m[4]) / d,
        (m[5] * m[6] - m[3] * m[8]) / d,
        (m[0] * m[8] - m[2] * m[6]) / d,
        (m[2] * m[3] - m[0] * m[5]) / d,
        (m[3] * m[7] - m[4] * m[6]) / d,
        (m[1] * m[6] - m[0] * m[7]) / d,
        (m[0] * m[4] - m[1] * m[3]) / d,
    ]
}

impl<'a> VelProblem<'a> {
    /// `coil`: Σ_c |S_c(r)|² per voxel (None: 1); sets must be t = f·E + e with one nv image each
    pub fn new(st: &Arc<CudaStream>, kern: Arc<Kernels>, op: &'a mut dyn SetOp, ys: Vec<CudaSlice<C32>>, model: VelModel, coil: Option<Vec<f32>>) -> Result<Self, Error> {
        let ys: Vec<YData> = ys.into_iter().map(YData::Dev).collect();
        let s = &model.s;
        let (nv, ne) = (s.nv(), s.ne());
        // per-frame sets (t = f·E + e, one image) or per-encode temporal sets (t = e, all F frames: each sample at its
        // own cardiac phase inside the operator)
        let temporal = op.nsets() == ne && (0..ne).all(|t| op.len_x(t) == s.nf * nv) && s.nf > 1;
        assert!(temporal || (op.nsets() == s.nf * ne && (0..op.nsets()).all(|t| op.len_x(t) == nv)), "velocity problem: sets must be frames × encodes (one image each) or encodes (all frames each)");
        let ny = (0..op.nsets()).map(|t| op.len_y(t)).max().unwrap_or(0);
        let mut m = [0.0f64; 9];
        for k in &s.kenc {
            for i in 0..3 {
                for j in 0..3 {
                    m[3 * i + j] += k[i] * k[j];
                }
            }
        }
        let tr = m[0] + m[4] + m[8];
        for i in 0..3 {
            m[4 * i] += 1e-9 * tr;
        }
        let mi = inv3(&m);
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let wsum_z: Vec<f32> = (0..s.kz).map(|k| (ne as f64 * (0..s.nf).map(|f| model.wz[f * s.kz + k]).sum::<f64>()) as f32).collect();
        let _ = ny;
        Ok(Self {
            kern,
            x: st.alloc_zeros(nv)?,
            g: st.alloc_zeros(nv)?,
            sacc: (0..ne).map(|_| st.alloc_zeros::<f32>(nv)).collect::<Result<_, _>>()?,
            coil: coil.as_ref().map(|c| st.clone_htod(c)).transpose()?,
            coil_host: coil,
            sig: st.alloc_zeros(s.kv * nv)?,
            zz: st.alloc_zeros(nv)?,
            m3: st.clone_htod(&f32s(&m))?,
            m3inv: st.clone_htod(&f32s(&mi))?,
            wsum_z,
            damp_z: vec![0.0; s.kz],
            damp_v: vec![0.0; s.kv],
            diag_a: vec![1.0; s.np()],
            weights: vec![1.0; op.nsets()],
            metric_every: 5,
            damping: 1e-3,
            calls: 0,
            have_metric: false,
            temporal,
            freeze_v: false,
            anchor: None,
            anchor_lam: 0.0,
            v_scale: 1.0,
            nhat: 0.0,
            op,
            ys,
            model,
        })
    }

    /// penalise velocity in static tissue (weights `w` per voxel, e.g. a 0/1 mask): breaks the degeneracy between a
    /// low-order residual phase and the time-mean velocity (research §A3)
    pub fn set_anchor(&mut self, st: &Arc<CudaStream>, w: &[f32], mu: f64) -> Result<(), Error> {
        self.anchor = Some((st.clone_htod(w)?, mu));
        self.have_metric = false;
        Ok(())
    }

    /// remove the static-tissue anchor
    pub fn clear_anchor(&mut self) {
        self.anchor = None;
        self.anchor_lam = 0.0;
        self.have_metric = false;
    }

    /// zero the v and residual-phase chunks (frozen parameters)
    fn zero_frozen(&self, st: &Arc<CudaStream>, g: &mut Stack) -> Result<(), Error> {
        if self.freeze_v {
            let s = &self.model.s;
            for c in s.kz..s.kz + s.kv + 1 {
                st.memset_zeros(&mut g.v[c])?;
            }
        }
        Ok(())
    }

    /// forget the metric state (start of a new run from another θ)
    pub fn reset(&mut self) {
        self.have_metric = false;
        self.calls = 0;
    }

    /// bytes of all k-space data and of the largest set
    pub fn data_bytes(&self) -> (usize, usize) {
        (self.ys.iter().map(|y| y.len() * 8).sum(), self.ys.iter().map(|y| y.len() * 8).max().unwrap_or(0))
    }

    /// bytes of the problem's own device scratch (model images, accumulators, metric weights)
    pub fn scratch_bytes(&self) -> usize {
        let s = &self.model.s;
        let nv = s.nv();
        2 * nv * 8 + (s.ne() + s.kv + 1) * nv * 4
    }

    /// move the k-space data to pinned host memory (frees the device copies); staged per set in `gradient`
    pub fn data_to_host(&mut self, ctx: &Arc<cudarc::driver::CudaContext>, st: &Arc<CudaStream>) -> Result<usize, Error> {
        let mut freed = 0;
        for y in self.ys.iter_mut() {
            if let YData::Dev(d) = y {
                let mut h = unsafe { ctx.alloc_pinned_with_flags::<C32>(d.len().max(1), 0)? };
                st.memcpy_dtoh(d, &mut h)?;
                freed += d.len() * 8;
                *y = YData::Host(h);
            }
        }
        st.synchronize()?;
        Ok(freed)
    }

    /// encode and frames of set t
    fn set_frames(&self, t: usize) -> (usize, Vec<usize>) {
        if self.temporal {
            (t, (0..self.model.s.nf).collect())
        } else {
            (t % self.model.s.ne(), vec![t / self.model.s.ne()])
        }
    }

    /// the set's image (its frames, frame-major) from the model, or its linearisation J d
    fn set_image(&mut self, st: &Arc<CudaStream>, th: &Stack, d: Option<&Stack>, t: usize) -> Result<CudaSlice<C32>, Error> {
        let (e, fr) = self.set_frames(t);
        let nv = self.model.s.nv();
        if fr.len() == 1 {
            let mut x = st.alloc_zeros::<C32>(nv)?;
            match d {
                None => self.model.eval(st, th, e, fr[0], &mut x)?,
                Some(d) => self.model.jvp(st, th, d, e, fr[0], &mut x)?,
            }
            return Ok(x);
        }
        let mut x = st.alloc_zeros::<C32>(fr.len() * nv)?;
        for (i, &f) in fr.iter().enumerate() {
            match d {
                None => self.model.eval(st, th, e, f, &mut self.x)?,
                Some(d) => self.model.jvp(st, th, d, e, f, &mut self.x)?,
            }
            st.memcpy_dtod(&self.x, &mut x.slice_mut(i * nv..(i + 1) * nv))?;
        }
        Ok(x)
    }

    /// grad += w Jᴴ g for the set's frames (g frame-major like the set image)
    fn set_vjp(&mut self, st: &Arc<CudaStream>, th: &Stack, t: usize, g: &CudaSlice<C32>, w: f32, grad: &mut Stack) -> Result<(), Error> {
        let (e, fr) = self.set_frames(t);
        let nv = self.model.s.nv();
        for (i, &f) in fr.iter().enumerate() {
            st.memcpy_dtod(&g.slice(i * nv..(i + 1) * nv), &mut self.g)?;
            self.model.vjp(st, th, e, f, &self.g, w, grad, Some(&mut self.sacc[e]))?;
        }
        Ok(())
    }

    /// recompute the Gauss–Newton weights at θ
    pub fn update_metric(&mut self, st: &Arc<CudaStream>, th: &Stack) -> Result<(), Error> {
        let s = self.model.s.clone();
        let nv = s.nv();
        self.model.sigma(st, th, &mut self.sig, &mut self.zz)?;
        let zz = st.clone_dtoh(&self.zz)?;
        let czz: Vec<f32> = match &self.coil_host {
            Some(c) => zz.iter().zip(c).map(|(a, b)| a * b).collect(),
            None => zz.clone(),
        };
        let mean_czz = czz.iter().map(|&v| v as f64).sum::<f64>() / nv as f64;
        let mean_c = self.coil_host.as_ref().map_or(1.0, |c| c.iter().map(|&v| v as f64).sum::<f64>() / nv as f64);
        for k in 0..s.kz {
            self.damp_z[k] = (self.damping * self.wsum_z[k] as f64 * mean_c) as f32;
        }
        for k in 0..s.kv {
            let om: f64 = (0..s.nf).map(|f| self.model.wv[f * s.kv + k]).sum();
            self.damp_v[k] = (self.damping * mean_czz * om / s.nf as f64).max(1e-30) as f32;
        }
        // absolute scale of the data curvature: n̂ = ⟨u, EᴴE u⟩ / Σ c |u|² for a random probe u on the middle set
        // (≈ the mean diagonal of the normal operator per unit coil energy)
        if self.nhat == 0.0 {
            let t = self.op.nsets() / 2;
            let mut sgen = 0x5DEECE66Du64;
            let mut rnd = || { sgen ^= sgen << 13; sgen ^= sgen >> 7; sgen ^= sgen << 17; ((sgen >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0 };
            let nx = self.op.len_x(t);
            let uh: Vec<C32> = (0..nx).map(|_| C32::new(rnd() as f32, rnd() as f32)).collect();
            let u = st.clone_htod(&uh)?;
            let mut buf = st.alloc_zeros::<C32>(self.op.len_y(t))?;
            let mut gu = st.alloc_zeros::<C32>(nx)?;
            self.op.forward(st, t, &u, &mut buf)?;
            self.op.adjoint(st, t, &buf, &mut gu)?;
            let num = self.kern.dot(st, &u, &gu)?.re;
            let den: f64 = uh.iter().enumerate().map(|(q, c)| (c.re as f64 * c.re as f64 + c.im as f64 * c.im as f64) * self.coil_host.as_ref().map_or(1.0, |cc| cc[q % nv] as f64)).sum();
            self.nhat = (num / den.max(1e-30)).max(1e-30);
        }
        // mean Gauss–Newton curvature of one velocity node and component: n̂ c σ_k tr(Σ k kᵀ)/3, σ_k ≈ Σ_f |z_f|² / K_v
        let tr: f64 = s.kenc.iter().map(|k| k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sum();
        self.v_scale = self.nhat * mean_czz / s.kv as f64 * tr / 3.0;
        if let Some((_, mu)) = &self.anchor {
            self.anchor_lam = mu * self.v_scale;
        }
        if s.np() > 0 {
            let czz_d = st.clone_htod(&czz)?;
            let d = self.model.poly_reduce(st, &czz_d, 2)?;
            let mean = d.iter().sum::<f64>() / d.len() as f64;
            self.diag_a = d.iter().map(|&x| x + self.damping * mean).collect();
        }
        self.have_metric = true;
        Ok(())
    }

    fn zero(&self, st: &Arc<CudaStream>, g: &mut Stack) -> Result<(), Error> {
        for c in g.v.iter_mut() {
            st.memset_zeros(c)?;
        }
        Ok(())
    }

    /// grad's residual-phase chunk = −Σ_r P_p S_e (from the per-encode accumulators)
    fn finish_a(&mut self, st: &Arc<CudaStream>, grad: &mut Stack) -> Result<(), Error> {
        let s = &self.model.s;
        let np = s.np();
        if np == 0 {
            return Ok(());
        }
        let mut af = vec![0f32; 2 * s.na().div_ceil(2).max(1)];
        for e in 1..s.ne() {
            let red = self.model.poly_reduce(st, &self.sacc[e], 1)?;
            for p in 0..np {
                af[(e - 1) * np + p] = -red[p] as f32;
            }
        }
        let host: Vec<C32> = af.chunks(2).map(|p| C32::new(p[0], p[1])).collect();
        st.memcpy_htod(&host, &mut grad.v[s.chunk_a()])?;
        Ok(())
    }

    /// out = H d, the Gauss–Newton Hessian-vector product Σ_t w_t J_tᴴ E_tᴴ E_t J_t d at θ
    pub fn hvp(&mut self, st: &Arc<CudaStream>, th: &Stack, d: &Stack, out: &mut Stack) -> Result<(), Error> {
        self.zero(st, out)?;
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        for t in 0..self.op.nsets() {
            let dx = self.set_image(st, th, Some(d), t)?;
            // operators take exact-length slices: per-set buffers (stream-ordered allocation)
            let mut buf = st.alloc_zeros::<C32>(self.op.len_y(t))?;
            self.op.forward(st, t, &dx, &mut buf)?;
            let mut g = st.alloc_zeros::<C32>(dx.len())?;
            self.op.adjoint(st, t, &buf, &mut g)?;
            let w = self.weights[t];
            self.set_vjp(st, th, t, &g, w, out)?;
        }
        self.finish_a(st, out)?;
        if let Some((w, _)) = &self.anchor {
            for k in 0..self.model.s.kv {
                let c = self.model.s.chunk_v(k);
                self.model.anchor(st, &d.v[c], w, self.anchor_lam as f32, &mut out.v[c])?;
            }
        }
        self.zero_frozen(st, out)
    }

    /// λ_max of M H (`with_metric`) or of H, by power iteration in the M⁻¹ (or Euclidean) inner product. Uses two
    /// parameter stacks plus chunk scratch (the planner's device minimum, together with θ).
    pub fn lambda_max(&mut self, st: &Arc<CudaStream>, th: &Stack, with_metric: bool, iters: usize) -> Result<f64, Error> {
        if with_metric && !self.have_metric {
            self.update_metric(st, th)?;
        }
        let lens = self.model.lens();
        let mut v = Stack::zeros_lens(st, &lens)?;
        // deterministic start: θ's own pattern (excites z, v and a)
        v.copy_from(st, th)?;
        let mut hv = Stack::zeros_lens(st, &lens)?;
        let mut sc = crate::store::Scratch::default();
        let k = self.kern.clone();
        let mut lam = 0.0;
        for _ in 0..iters {
            // ‖v‖²_{M⁻¹}, chunk by chunk
            let mut n2 = 0.0;
            for t in 0..lens.len() {
                if with_metric {
                    let w = &mut sc.get(st, lens[t], 1)?[0];
                    self.metric(st, t, &v.v[t], w, true)?;
                    n2 += k.dot(st, &v.v[t], w)?.re;
                } else {
                    n2 += k.dot(st, &v.v[t], &v.v[t])?.re;
                }
            }
            for t in 0..lens.len() {
                k.axpby(st, Complex64::new(0.0, 0.0), &hv.v[t], Complex64::new(1.0 / n2.sqrt(), 0.0), &mut v.v[t])?;
            }
            self.hvp(st, th, &v, &mut hv)?;
            lam = rdot(&k, st, &v, &hv)?; // ⟨v, H v⟩ with ‖v‖_{M⁻¹} = 1
            for t in 0..lens.len() {
                if with_metric {
                    self.metric(st, t, &hv.v[t], &mut v.v[t], false)?;
                } else {
                    st.memcpy_dtod(&hv.v[t], &mut v.v[t])?;
                }
            }
        }
        let _ = axpby;
        Ok(lam)
    }
}

impl Objective for VelProblem<'_> {
    fn kernels(&self) -> &Kernels {
        &self.kern
    }

    fn lens(&self) -> Vec<usize> {
        self.model.lens()
    }

    fn gradient(&mut self, st: &Arc<CudaStream>, th: &Stack, grad: &mut Stack) -> Result<f64, Error> {
        if !self.have_metric || (self.metric_every > 0 && self.calls % self.metric_every == 0) {
            self.update_metric(st, th)?;
        }
        self.calls += 1;
        self.zero(st, grad)?;
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        let mut fval = 0.0;
        for t in 0..self.op.nsets() {
            let w = self.weights[t];
            if w == 0.0 {
                continue;
            }
            let x = self.set_image(st, th, None, t)?;
            let ny = self.op.len_y(t);
            let mut buf = st.alloc_zeros::<C32>(ny)?;
            self.op.forward(st, t, &x, &mut buf)?;
            let staged;
            let y: &CudaSlice<C32> = match &self.ys[t] {
                YData::Dev(d) => d,
                YData::Host(h) => {
                    let mut d = st.alloc_zeros::<C32>(h.len())?;
                    st.memcpy_htod(h, &mut d)?;
                    staged = d;
                    &staged
                }
            };
            self.kern.axpby(st, Complex64::new(-1.0, 0.0), y, Complex64::new(1.0, 0.0), &mut buf)?;
            fval += 0.5 * w as f64 * self.kern.dot(st, &buf, &buf)?.re;
            let mut g = st.alloc_zeros::<C32>(x.len())?;
            self.op.adjoint(st, t, &buf, &mut g)?;
            self.set_vjp(st, th, t, &g, w, grad)?;
        }
        self.finish_a(st, grad)?;
        if let Some((w, _)) = &self.anchor {
            for k in 0..self.model.s.kv {
                let c = self.model.s.chunk_v(k);
                fval += self.model.anchor(st, &th.v[c], w, self.anchor_lam as f32, &mut grad.v[c])?;
            }
        }
        self.zero_frozen(st, grad)?;
        Ok(fval)
    }

    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        let s = self.model.s.clone();
        if t < s.kz {
            return self.model.metric_z(st, r, z, self.coil.as_ref(), self.wsum_z[t], self.damp_z[t], inverse);
        }
        if t < s.kz + s.kv {
            let k = t - s.kz;
            let sig = raw(&self.sig, st) + (k * s.nv() * 4) as u64;
            // the metric is in units of the data curvature / n̂; the anchor's λ w I is trace-matched onto Σ k kᵀ
            let tr: f64 = s.kenc.iter().map(|k| k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sum();
            let anchor = self.anchor.as_ref().map(|(w, _)| (w, (self.anchor_lam / self.nhat.max(1e-30) * 3.0 / tr) as f32));
            return self.model.metric_v(st, r, z, self.coil.as_ref(), sig, &self.m3, &self.m3inv, self.damp_v[k], inverse, anchor);
        }
        // residual phase: diagonal (identical for every encode: |∂x/∂a_p|² = P_p² |z_f|²)
        let np = s.np();
        let h = st.clone_dtoh(r)?;
        let f: Vec<f32> = h.iter().flat_map(|c| [c.re, c.im]).collect();
        let mut o = vec![0f32; f.len()];
        for i in 0..s.na() {
            let d = self.diag_a[i % np.max(1)];
            o[i] = if inverse { (f[i] as f64 * d) as f32 } else { (f[i] as f64 / d) as f32 };
        }
        let host: Vec<C32> = o.chunks(2).map(|p| C32::new(p[0], p[1])).collect();
        st.memcpy_htod(&host, z)?;
        Ok(())
    }
}

/// number of residual-phase terms of degree d (re-export for drivers)
pub fn poly_terms(d: usize) -> usize {
    poly_exps(d).len()
}

#[cfg(test)]
mod tests {
    use hasty_cuda::Device;

    use super::*;
    use crate::op::SenseOp;
    use crate::p2np::{run_objective, Precond};
    use crate::setop::SenseSets;
    use crate::vmodel::cpu::Theta;
    use crate::vmodel::VelSpec;
    use num_complex::Complex64 as C;

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
        let (lo, hi) = (std::f64::consts::PI / 1.0, std::f64::consts::PI / 2.5);
        let mut kenc = vec![[0.0; 3]];
        for a in 0..3 {
            for k in [lo, hi] {
                let mut v = [0.0; 3];
                v[a] = k;
                kenc.push(v);
            }
        }
        VelSpec { dims: [12, 10, 6], kenc, nf: 4, kz: 2, deg_z: 1, kv: 4, deg_v: 1, poly: Some(1) }
    }

    /// smooth "vessel" truth: magnitude blob, velocity field within ±0.9 m/s, small residual phase
    fn truth(s: &VelSpec) -> Theta {
        let mut t = Theta::zeros(s);
        let [nx, ny, nz] = s.dims;
        for r in 0..s.nv() {
            let (i, j, k) = ((r % nx) as f64, ((r / nx) % ny) as f64, (r / (nx * ny)) as f64);
            let m = (-((i - 6.0).powi(2) / 18.0 + (j - 5.0).powi(2) / 12.0 + (k - 3.0).powi(2) / 6.0)).exp() + 0.2;
            for kk in 0..s.kz {
                t.z[kk][r] = C::from_polar(m * (1.0 + 0.1 * kk as f64), 0.3 * (i / nx as f64));
            }
            for kk in 0..s.kv {
                let ph = 2.0 * std::f64::consts::PI * kk as f64 / s.kv as f64;
                t.v[kk][r] = [0.6 * ph.sin() * (j / ny as f64), 0.4 * ph.cos(), 0.3 * (k / nz as f64) * (1.0 + ph.sin())];
            }
        }
        for (i, a) in t.a.iter_mut().enumerate() {
            *a = 0.05 * ((i % 4) as f64 - 1.5);
        }
        t
    }

    #[test]
    fn velocity_objective_gradient_hessian_and_solve() {
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let kern = Arc::new(Kernels::new(&dev).unwrap());
        let s = spec();
        let (nv, ne) = (s.nv(), s.ne());
        let mut rng = Rng(0x77);
        let nc = 3;
        let smaps: Vec<C32> = (0..nc * nv).map(|i| { let c = i / nv; let r = i % nv; let ph = 0.4 * c as f64 + 0.002 * r as f64; C32::new((0.6 * ph.cos()) as f32, (0.6 * ph.sin()) as f32) }).collect();
        let nsets = s.nf * ne;
        let m = 900;
        let coords: Vec<[Vec<f32>; 3]> = (0..nsets).map(|_| [0, 1, 2].map(|_| (0..m).map(|_| (std::f64::consts::PI * rng.n()) as f32).collect())).collect();
        let mut op = SenseOp::new(&dev, &st, kern.clone(), s.dims, [&coords[0][0], &coords[0][1], &coords[0][2]], &smaps, nc, vec![], 1e-5).unwrap();
        let dc: Vec<[CudaSlice<f32>; 3]> = coords.iter().map(|c| [0, 1, 2].map(|a| st.clone_htod(&c[a]).unwrap())).collect();
        let model = VelModel::new(&dev, &st, s.clone()).unwrap();
        let tr = truth(&s);
        let up = |t: &Theta| Stack { v: t.pack(&s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
        // data y_t = E_t x_t(θ_true)
        let tg = up(&tr);
        let mut sets = SenseSets { op: &mut op, coords: dc, ms: vec![m; nsets], maps: vec![], map_of_set: vec![] };
        let mut ys = vec![];
        {
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            for t in 0..nsets {
                model.eval(&st, &tg, t % ne, t / ne, &mut x).unwrap();
                let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
                sets.forward(&st, t, &x, &mut y).unwrap();
                ys.push(y);
            }
        }
        let mut pr = VelProblem::new(&st, kern.clone(), &mut sets, ys, model, None).unwrap();
        // start: truth with perturbed velocity (inside the basin: ≤ 0.15 m/s) and zero residual phase
        let mut start = tr.clone();
        for v in start.v.iter_mut() {
            for x in v.iter_mut() {
                for c in 0..3 {
                    x[c] += 0.15 * rng.n();
                }
            }
        }
        for a in start.a.iter_mut() {
            *a = 0.0;
        }
        let th = up(&start);
        let lens = s.lens();
        let mut g = Stack::zeros_lens(&st, &lens).unwrap();
        let f0 = pr.gradient(&st, &th, &mut g).unwrap();
        // directional derivative vs central differences (f32 pipeline: loose tolerance)
        let mut d = Theta::zeros(&s);
        for k in 0..s.kv {
            for r in 0..nv {
                d.v[k][r] = [rng.n(), rng.n(), rng.n()];
            }
        }
        for k in 0..s.kz {
            for r in 0..nv {
                d.z[k][r] = C::new(rng.n(), rng.n()) * 0.1;
            }
        }
        for a in d.a.iter_mut() {
            *a = rng.n();
        }
        let h = 2e-3;
        let fp = { let t = up(&start.add(h, &d)); pr.gradient(&st, &t, &mut Stack::zeros_lens(&st, &lens).unwrap()).unwrap() };
        let fm = { let t = up(&start.add(-h, &d)); pr.gradient(&st, &t, &mut Stack::zeros_lens(&st, &lens).unwrap()).unwrap() };
        let fd = (fp - fm) / (2.0 * h);
        let gh = Theta::unpack(&s, &g.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect::<Vec<_>>());
        let an = gh.dot(&d);
        assert!((fd - an).abs() < 2e-3 * an.abs(), "objective gradient: fd {fd:.6e} vs analytic {an:.6e} (f {f0:.3e})");
        // Gauss–Newton Hessian symmetry
        let u = up(&d);
        let w = { let mut e = d.clone(); for k in 0..s.kv { for r in 0..nv { e.v[k][r] = [rng.n(), 0.5 * rng.n(), rng.n()]; } } up(&e) };
        let (mut hu, mut hw) = (Stack::zeros_lens(&st, &lens).unwrap(), Stack::zeros_lens(&st, &lens).unwrap());
        pr.hvp(&st, &th, &u, &mut hu).unwrap();
        pr.hvp(&st, &th, &w, &mut hw).unwrap();
        let (a1, a2) = (rdot(&kern, &st, &w, &hu).unwrap(), rdot(&kern, &st, &u, &hw).unwrap());
        assert!((a1 - a2).abs() < 1e-4 * a1.abs().max(a2.abs()), "Hessian not symmetric: {a1:e} vs {a2:e}");
        // solve with the residual phase free: the data term converges, but the time-mean velocity that is linear in
        // space trades against the degree-1 residual phase (the identifiability issue of research §A3; in practice
        // the static-tissue anchor breaks it)
        let (lam, fs, e_free) = solve(&mut pr, &st, &kern, &s, &start, &tr, &up);
        eprintln!("residual phase free: f {:.3e} -> {:.3e}, velocity RMS error -> {e_free:.4} m/s (degenerate), λ_max(MH) {lam:.3e}", fs[1], fs.last().unwrap());
        assert!(fs.last().unwrap() < &(1e-3 * fs[1]));
        drop(pr);
        // identifiable model (no residual phase): exact recovery
        let s2 = VelSpec { poly: None, ..s.clone() };
        let model2 = VelModel::new(&dev, &st, s2.clone()).unwrap();
        let mut tr2 = tr.clone();
        tr2.a.clear();
        let up2 = |t: &Theta| Stack { v: t.pack(&s2).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
        let tg2 = up2(&tr2);
        let mut ys2 = vec![];
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        for t in 0..nsets {
            model2.eval(&st, &tg2, t % ne, t / ne, &mut x).unwrap();
            let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
            sets.forward(&st, t, &x, &mut y).unwrap();
            ys2.push(y);
        }
        let mut pr2 = VelProblem::new(&st, kern.clone(), &mut sets, ys2, model2, None).unwrap();
        let mut start2 = start.clone();
        start2.a.clear();
        let e0 = vrms(&s2, &start2, &tr2);
        let (_, fs2, e1) = solve(&mut pr2, &st, &kern, &s2, &start2, &tr2, &up2);
        eprintln!("identifiable: f {:.3e} -> {:.3e}, velocity RMS error {e0:.4} -> {e1:.4} m/s", fs2[1], fs2.last().unwrap());
        assert!(e1 < 0.02 * e0, "velocity not recovered: {e0:.4} -> {e1:.4}");
        // the same solve with the secant pair and the rank-one direction swapped out to pinned host memory (and w
        // recomputed in a third pass): identical iterates up to float rounding
        for (i, (prev_host, w_mode)) in [(true, crate::p2np::WMode::Host), (true, crate::p2np::WMode::Recompute), (false, crate::p2np::WMode::Recompute)].into_iter().enumerate() {
            if i == 1 {
                // from here on the k-space data is streamed from pinned host memory as well
                let freed = pr2.data_to_host(&dev.ctx, &st).unwrap();
                assert!(freed > 0);
            }
            let opts = crate::p2np::StateOpts { ctx: Some(dev.ctx.clone()), prev_on_host: prev_host, w_mode };
            let th = up2(&start2);
            let lam = pr2.lambda_max(&st, &th, true, 15).unwrap();
            let mut x = up2(&start2);
            let mut fs3 = vec![];
            crate::p2np::run_objective_with(&mut pr2, &st, &mut x, Precond::Zmshr1M, 1.0 / lam, 80, &mut |_, _| {}, &mut |_, _, f| fs3.push(f), None, &opts).unwrap();
            let xe = Theta::unpack(&s2, &x.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect::<Vec<_>>());
            let e3 = vrms(&s2, &xe, &tr2);
            eprintln!("state on host {prev_host}, w {w_mode:?}: velocity RMS error {e3:.2e} (device: {e1:.2e}), f {:.3e} vs {:.3e}", fs3.last().unwrap(), fs2.last().unwrap());
            assert!(e3 < 0.02 * e0, "host-placed state: {e3}");
            let r = (fs3[10] / fs2[10] - 1.0).abs();
            assert!(r < 1e-3, "iterate 10 differs: f {:.6e} vs {:.6e}", fs3[10], fs2[10]);
        }
    }

    const SIGMA: f32 = -1.0;
    const ORIGIN_SIGN: f64 = 1.0;
    /// NUFFT convention used by residual-phase injection: E_k(x · e^{i g·r}) = E_{k + σ g}(x) for σ = ?
    #[test]
    fn nufft_linear_phase_is_a_coordinate_shift() {
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let kern = Arc::new(Kernels::new(&dev).unwrap());
        let dims = [16usize, 12, 8];
        let nv = 16 * 12 * 8;
        let mut rng = Rng(5);
        let m = 2000;
        let c: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (2.5 * rng.n()) as f32).collect());
        let smaps = vec![C32::new(1.0, 0.0); nv];
        let g = [0.11f32, -0.07, 0.05];
        let x: Vec<C32> = (0..nv).map(|i| { let (a, b, k) = (i % 16, (i / 16) % 12, i / 192); let v = (-(((a as f32 - 8.0) / 4.0).powi(2) + ((b as f32 - 6.0) / 3.0).powi(2) + ((k as f32 - 4.0) / 2.0).powi(2))).exp(); C32::new(v, 0.3 * v) }).collect();
        let xm: Vec<C32> = (0..nv).map(|i| { let (a, b, k) = ((i % 16) as f32, ((i / 16) % 12) as f32, (i / 192) as f32); let ph = g[0] * a + g[1] * b + g[2] * k; let z = Complex64::new(x[i].re as f64, x[i].im as f64) * Complex64::from_polar(1.0, ph as f64); C32::new(z.re as f32, z.im as f32) }).collect();
        let fwd = |cc: [Vec<f32>; 3], img: &[C32]| -> Vec<C32> {
            let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&cc[0], &cc[1], &cc[2]], &smaps, 1, vec![], 1e-6).unwrap();
            let dx = st.clone_htod(img).unwrap();
            let mut y = st.alloc_zeros::<C32>(m).unwrap();
            crate::setop::LinOp::forward(&mut op, &st, &dx, &mut y).unwrap();
            st.clone_dtoh(&y).unwrap()
        };
        // per axis: E_k(x e^{i g r}) = e^{i g r0} E_{k + σ g}(x) (image origin r0)
        for ax in 0..3 {
            let mut gg = [0f32; 3];
            gg[ax] = g[ax];
            let xm: Vec<C32> = (0..nv).map(|i| { let p = [(i % 16) as f32, ((i / 16) % 12) as f32, (i / 192) as f32]; let ph = gg[0] * p[0] + gg[1] * p[1] + gg[2] * p[2]; let z = Complex64::new(x[i].re as f64, x[i].im as f64) * Complex64::from_polar(1.0, ph as f64); C32::new(z.re as f32, z.im as f32) }).collect();
            let y1 = fwd(c.clone(), &xm);
            for sg in [1.0f32, -1.0] {
                let cs: [Vec<f32>; 3] = [0, 1, 2].map(|a| c[a].iter().map(|&k| k + sg * gg[a]).collect());
                let y2 = fwd(cs, &x);
                let (mut num, mut den) = (Complex64::new(0.0, 0.0), 0.0);
                for (a, b) in y1.iter().zip(&y2) {
                    let (a, b) = (Complex64::new(a.re as f64, a.im as f64), Complex64::new(b.re as f64, b.im as f64));
                    num += b.conj() * a;
                    den += b.norm_sqr();
                }
                let cc = num / den;
                let res: f64 = y1.iter().zip(&y2).map(|(a, b)| (Complex64::new(a.re as f64, a.im as f64) - cc * Complex64::new(b.re as f64, b.im as f64)).norm_sqr()).sum::<f64>() / y1.iter().map(|a| (a.re as f64).powi(2) + (a.im as f64).powi(2)).sum::<f64>();
                eprintln!("axis {ax} σ {sg}: |c| {:.6} arg(c)/g {:.4} voxels, residual {:.2e}", cc.norm(), cc.arg() / gg[ax] as f64, res.sqrt());
                if sg == SIGMA {
                    assert!(res.sqrt() < 1e-4 && (cc.norm() - 1.0).abs() < 1e-4, "E_k(x e^(igr)) ≠ c E_(k+σg)(x)");
                    assert!((cc.arg() / gg[ax] as f64 - ORIGIN_SIGN * dims[ax] as f64 / 2.0).abs() < 1e-2, "image origin is not n/2");
                }
            }
        }
    }

    fn vrms(s: &VelSpec, a: &Theta, b: &Theta) -> f64 {
        let mut e = 0.0;
        for k in 0..s.kv {
            for r in 0..s.nv() {
                for c in 0..3 {
                    e += (a.v[k][r][c] - b.v[k][r][c]).powi(2);
                }
            }
        }
        (e / (s.kv * s.nv() * 3) as f64).sqrt()
    }

    /// P²nP without a prior, ZMSHR1 in the Gauss–Newton metric, α = 1/λ_max(MH): (λ, f history, final velocity error)
    fn solve(pr: &mut VelProblem, st: &Arc<CudaStream>, kern: &Arc<Kernels>, s: &VelSpec, start: &Theta, tr: &Theta, up: &dyn Fn(&Theta) -> Stack) -> (f64, Vec<f64>, f64) {
        let _ = kern;
        let th = up(start);
        let lam = pr.lambda_max(st, &th, true, 15).unwrap();
        assert!(lam.is_finite() && lam > 0.0);
        let mut x = up(start);
        let mut fs = vec![];
        run_objective(pr, st, &mut x, Precond::Zmshr1M, 1.0 / lam, 80, &mut |_, _| {}, &mut |_, _, f| fs.push(f), None).unwrap();
        let xe = Theta::unpack(s, &x.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect::<Vec<_>>());
        (lam, fs, vrms(s, &xe, tr))
    }
}
