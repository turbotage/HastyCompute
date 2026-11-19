//! The velocity-model data term as a P²nP [`Objective`] (`research/velocity_model_sr.md` §A4):
//!   f(θ) = ½ Σ_t w_t ‖E_t x_{e,f}(θ) − y_t‖²,   set t = f·E + e (one operator frame of one encode per set)
//! with the gradient by the model's vector-Jacobian product, the pointwise Gauss–Newton metric M ≈ H⁻¹ (z: scalar per
//! voxel, v: 3×3 per voxel = (c·σ_k)⁻¹ (Σ_e k_e k_eᵀ)⁻¹, residual phase: diagonal) for `Precond::Circulant` /
//! `Precond::Zmshr1M`, and λ_max of (M) H by power iteration with Gauss–Newton Hessian-vector products.
//! Set weights w_t implement VENC continuation (low-VENC encodes ramped in).
//!
//! At-knots mode (`VelSpec::at_knots`, research/velocity_model_sr.md + temporal_operator.md §16–§21): one set per encode
//! holding all K = kr · mc knot images, x_{e,k} = z_k e^{i(φ_e − k_e·v_k)}, the 2-D knot operator interpolating them to
//! each sample's (breathing, cardiac) time. The Gauss–Newton metric of knot k carries the knot's weight ω_k (diagonal
//! of the temporal Gram of the basis under the sample times, mean 1: sparsely sampled knots get less curvature). The
//! parameters may live in host memory ([`crate::hoststack::run_objective_host`]): `gradient_host` / `hvp_host` stream
//! θ knot by knot through a few node-sized device slots, so the device holds one encode's two knot stacks (set image
//! and its adjoint), the data of one set and the operator, never a parameter stack.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use num_complex::Complex64;

use rayon::prelude::*;

use crate::gpu::Kernels;
use crate::hoststack::HostStack;
use crate::p2np::{axpby, rdot, Objective, Stack};
use crate::setop::SetOp;
use crate::vmodel::{poly_exps, raw, VelModel};
use crate::Error;

/// k-space data of one set: resident, or in pinned host memory (staged per use)
pub enum YData {
    Dev(CudaSlice<C32>),
    Host(cudarc::driver::PinnedHostSlice<C32>),
    /// in the problem's host k-space store, staged per set ([`VelProblem::data_from_store`]); its length
    Staged(usize),
}

impl YData {
    pub fn len(&self) -> usize {
        match self {
            YData::Dev(d) => d.len(),
            YData::Host(h) => h.len(),
            YData::Staged(n) => *n,
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
    /// trust region of the velocity step (radians, research/velocity_model_sr.md §F.2): per (voxel, knot) the step's
    /// largest encode phase max_e |k'_e·Δv| is shrunk to ≤ ρ (the Gauss–Newton phase linearisation holds there);
    /// None = no limit
    pub vtrust: Option<f64>,
    /// static-tissue anchor: per-voxel weight and the relative strength μ (λ = μ × mean Gauss–Newton curvature of v)
    anchor: Option<(CudaSlice<f32>, f64)>,
    pub anchor_lam: f64,
    v_scale: f64,
    /// mean diagonal of E_tᴴE_t per unit coil energy (absolute scale of the data curvature; 0 = not estimated)
    pub nhat: f64,
    /// host store of the data (`YData::Staged` sets), staged per gradient with prefetch
    stage: Option<crate::kstage::Stager>,
    /// at-knots mode: relative curvature per knot ω_k (mean 1; default 1), [`VelProblem::set_knot_weights`]
    knot_w: Vec<f64>,
    /// at-knots mode: the Gauss–Newton weights ω_k |z_k|² (K × nv) on the host, uploaded per metric call into `sig1`
    sig_host: Vec<f32>,
    sig1: Option<CudaSlice<f32>>,
    /// host copy of the anchor weights (host-resident state)
    anchor_host: Vec<f32>,
    /// node-sized device slots of the streamed (host-resident) at-knots gradient: z, v, dz, dv, gz, gv
    slots: Option<KnotSlots>,
    /// spline mode, host-resident state: row slots (θ row, and d or gradient row)
    rows: Option<[RowSlot; 2]>,
    /// stochastic coil sampling of the data term (`p2np::CoilSampling`: the strongest `keep` coils, `draw` of the tail
    /// weighted to be unbiased, held for `hold` iterations, all coils for the last ones); None = all coils
    pub sampling: Option<crate::p2np::CoilSampling>,
    /// the coil groups and weights of the current iteration (None: all coils); set by `begin_iteration`, or by the
    /// caller for a sampled λ_max estimate
    pub cur_groups: Option<Vec<(Vec<usize>, f64)>>,
    cur_k: usize,
    iters_total: usize,
}

struct KnotSlots {
    z: CudaSlice<C32>,
    v: CudaSlice<C32>,
    dz: CudaSlice<C32>,
    dv: CudaSlice<C32>,
    gz: CudaSlice<C32>,
    gv: CudaSlice<C32>,
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
        let temporal = !s.spline() && op.nsets() == ne && (0..ne).all(|t| op.len_x(t) == s.nf * nv) && s.nf > 1;
        if let Some(of) = &s.op {
            assert!(op.nsets() == ne * of.ng() && (0..op.nsets()).all(|t| op.len_x(t) == of.set_frames() * nv), "spline velocity model: sets must be encodes × respiratory groups ({} × {}), {} operator frames each", ne, of.ng(), of.set_frames());
        } else {
            assert!(temporal || (op.nsets() == s.nf * ne && (0..op.nsets()).all(|t| op.len_x(t) == nv)), "velocity problem: sets must be frames × encodes (one image each) or encodes (all frames each)");
        }
        let ny = (0..op.nsets()).map(|t| op.len_y(t)).max().unwrap_or(0);
        let mut m = [0.0f64; 9];
        for k in &s.kenc_model() {
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
        let ident = s.identity() || s.spline();
        Ok(Self {
            kern,
            x: st.alloc_zeros(nv)?,
            g: st.alloc_zeros(nv)?,
            sacc: (0..ne).map(|_| st.alloc_zeros::<f32>(nv)).collect::<Result<_, _>>()?,
            coil: coil.as_ref().map(|c| st.clone_htod(c)).transpose()?,
            coil_host: coil,
            // at-knots mode: the per-knot weights live on the host (`sig_host`)
            sig: st.alloc_zeros(if ident { 1 } else { s.kv * nv })?,
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
            vtrust: None,
            temporal,
            freeze_v: false,
            anchor: None,
            anchor_lam: 0.0,
            v_scale: 1.0,
            nhat: 0.0,
            stage: None,
            knot_w: vec![1.0; s.kz],
            sig_host: vec![],
            sig1: None,
            anchor_host: vec![],
            slots: None,
            rows: None,
            sampling: None,
            cur_groups: None,
            cur_k: 0,
            iters_total: 0,
            op,
            ys,
            model,
        })
    }

    /// One set's data-term pass on its set image `x` (consumed: freed before the adjoints): r_g = E_{t,g} x − y_g per
    /// coil group g (all coils, or the sampled groups of `cur_groups`; `y` None: no data, r = E x for Hessian-vector
    /// products), f += w w_g ½‖r_g‖², then `vjp(self, Eᴴ_g r_g, w · w_g)` per group (the caller's Jᴴ accumulation).
    /// Returns the f contribution.
    fn set_pass(&mut self, st: &Arc<CudaStream>, t: usize, x: CudaSlice<C32>, y: Option<&CudaSlice<C32>>, w: f32, vjp: &mut dyn FnMut(&mut Self, &CudaSlice<C32>, f32) -> Result<(), Error>) -> Result<f64, Error> {
        let nx = x.len();
        let Some(groups) = self.cur_groups.clone() else {
            let mut buf = unsafe { st.alloc::<C32>(self.op.len_y(t))? };
            match y {
                Some(y) => self.op.forward_residual(st, t, &x, y, &mut buf)?,
                None => self.op.forward(st, t, &x, &mut buf)?,
            }
            drop(x);
            let f = if y.is_some() { 0.5 * w as f64 * self.kern.dot(st, &buf, &buf)?.re } else { 0.0 };
            let mut g = unsafe { st.alloc::<C32>(nx)? };
            self.op.adjoint(st, t, &buf, &mut g)?;
            drop(buf);
            vjp(self, &g, w)?;
            return Ok(f);
        };
        let nc = self.op.n_coils();
        assert!(nc > 0, "coil sampling: the operator does not report its coils");
        let m = self.op.len_y(t) / nc;
        let mut f = 0.0;
        let mut rs: Vec<CudaSlice<C32>> = Vec::with_capacity(groups.len());
        for (coils, wg) in &groups {
            assert!(self.op.select_coils(st, Some(coils))?, "coil sampling: operator without coil subsets");
            let mut r = unsafe { st.alloc::<C32>(coils.len() * m)? };
            match y {
                Some(y) => {
                    // y restricted to the group's coils (rows of the coil-major data)
                    let mut ysub = unsafe { st.alloc::<C32>(coils.len() * m)? };
                    for (i, &c) in coils.iter().enumerate() {
                        st.memcpy_dtod(&y.slice(c * m..(c + 1) * m), &mut ysub.slice_mut(i * m..(i + 1) * m))?;
                    }
                    self.op.forward_residual(st, t, &x, &ysub, &mut r)?;
                    f += wg * 0.5 * w as f64 * self.kern.dot(st, &r, &r)?.re;
                }
                None => self.op.forward(st, t, &x, &mut r)?,
            }
            rs.push(r);
        }
        drop(x);
        for ((coils, wg), r) in groups.iter().zip(rs) {
            self.op.select_coils(st, Some(coils))?;
            let mut g = unsafe { st.alloc::<C32>(nx)? };
            self.op.adjoint(st, t, &r, &mut g)?;
            drop(r);
            vjp(self, &g, (w as f64 * wg) as f32)?;
        }
        self.op.select_coils(st, None)?;
        Ok(f)
    }

    /// at-knots mode: relative curvature ω_k of every knot (e.g. the diagonal of the temporal Gram of the basis under
    /// the sample times); normalised to mean 1
    pub fn set_knot_weights(&mut self, w: &[f64]) {
        let s = &self.model.s;
        assert!((s.identity() || s.spline()) && w.len() == s.kz, "set_knot_weights: at-knots / spline mode, one weight per knot");
        let mean = w.iter().sum::<f64>() / w.len() as f64;
        assert!(mean > 0.0 && w.iter().all(|&x| x >= 0.0), "knot weights must be ≥ 0 with a positive mean");
        self.knot_w = w.iter().map(|&x| (x / mean).max(1e-3)).collect();
        let ne = s.ne() as f64;
        self.wsum_z = self.knot_w.iter().map(|&x| (ne * x) as f32).collect();
        self.have_metric = false;
    }

    pub fn knot_weights(&self) -> &[f64] {
        &self.knot_w
    }

    /// penalise velocity in static tissue (weights `w` per voxel, e.g. a 0/1 mask): breaks the degeneracy between a
    /// low-order residual phase and the time-mean velocity (research §A3)
    pub fn set_anchor(&mut self, st: &Arc<CudaStream>, w: &[f32], mu: f64) -> Result<(), Error> {
        self.anchor = Some((st.clone_htod(w)?, mu));
        self.anchor_host = w.to_vec();
        self.have_metric = false;
        Ok(())
    }

    /// remove the static-tissue anchor
    pub fn clear_anchor(&mut self) {
        self.anchor = None;
        self.anchor_host.clear();
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
        if s.spline() {
            // x, g, accumulators, coil energy, zz, sig1; the streamed mode's two row slots ((m, ψ) + v of mc knots)
            return 2 * nv * 8 + (s.ne() + 3) * nv * 4 + self.row_slot_bytes();
        }
        if s.identity() {
            // x, g node images, accumulators, coil energy, zz, sig1; the streamed mode's six node slots (≤ 12 B / voxel)
            return 2 * nv * 8 + (s.ne() + 3) * nv * 4 + 6 * (3 * nv).div_ceil(2) * 8;
        }
        2 * nv * 8 + (s.ne() + s.kv + 1) * nv * 4
    }

    /// Keep the k-space data in a host store (`kstore`), staged per set in `gradient` with prefetch; device / pinned
    /// copies are dropped. The store's sets must match the operator's.
    pub fn data_from_store(&mut self, stage: crate::kstage::Stager) {
        let k = stage.store();
        assert_eq!(k.n_sets(), self.op.nsets());
        self.ys = (0..k.n_sets()).map(|t| {
            assert_eq!(k.len(t) * k.nvc(), self.op.len_y(t), "k-space store set {t} does not match the operator");
            YData::Staged(k.len(t) * k.nvc())
        }).collect();
        self.stage = Some(stage);
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
        let zz = if s.identity() || s.spline() {
            self.knot_sigma(&mut |k| Ok(st.clone_dtoh(&th.v[s.chunk_z(k)])?))?
        } else {
            self.model.sigma(st, th, &mut self.sig, &mut self.zz)?;
            st.clone_dtoh(&self.zz)?
        };
        self.finish_metric(st, zz)
    }

    /// [`update_metric`](Self::update_metric) at a host-resident θ (at-knots mode: straight from the host chunks)
    pub fn update_metric_host(&mut self, st: &Arc<CudaStream>, th: &HostStack) -> Result<(), Error> {
        let s = self.model.s.clone();
        if !s.identity() && !s.spline() {
            let mut d = Stack::zeros_lens(st, &th.lens)?;
            for t in 0..th.lens.len() {
                st.memcpy_htod(th.chunk(t), &mut d.v[t])?;
            }
            return self.update_metric(st, &d);
        }
        let zz = self.knot_sigma(&mut |k| Ok(th.chunk(s.chunk_z(k)).to_vec()))?;
        self.finish_metric(st, zz)
    }

    /// at-knots mode: sig_host[k] = ω_k |z_k|², returns zz = Σ_k ω_k |z_k|² (spline mode: the magnitude m_k = Re z_k)
    fn knot_sigma(&mut self, z: &mut dyn FnMut(usize) -> Result<Vec<C32>, Error>) -> Result<Vec<f32>, Error> {
        let s = &self.model.s;
        let (nv, k) = (s.nv(), s.kz);
        let polar = s.spline();
        self.sig_host.resize(k * nv, 0.0);
        let mut zz = vec![0f32; nv];
        for kk in 0..k {
            let zk = z(kk)?;
            let w = self.knot_w[kk] as f32;
            let sg = &mut self.sig_host[kk * nv..(kk + 1) * nv];
            sg.par_iter_mut().zip(zk.par_iter()).zip(zz.par_iter_mut()).for_each(|((o, c), a)| {
                *o = w * (c.re * c.re + if polar { 0.0 } else { c.im * c.im });
                *a += *o;
            });
        }
        Ok(zz)
    }

    /// the metric weights from zz = Σ_f (ω_f) |z_f|² (host): damping, n̂, the anchor's λ, the residual-phase diagonal
    fn finish_metric(&mut self, st: &Arc<CudaStream>, zz: Vec<f32>) -> Result<(), Error> {
        let s = self.model.s.clone();
        let nv = s.nv();
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
            let om: f64 = if s.identity() || s.spline() { self.knot_w[k] } else { (0..s.nf).map(|f| self.model.wv[f * s.kv + k]).sum() };
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
            self.op.forward(st, t, &u, &mut buf)?;
            drop(u);
            // ⟨u, EᴴE u⟩ = ‖E u‖² (no second set-image buffer)
            let num = self.kern.dot(st, &buf, &buf)?.re;
            let den: f64 = uh.iter().enumerate().map(|(q, c)| (c.re as f64 * c.re as f64 + c.im as f64 * c.im as f64) * self.coil_host.as_ref().map_or(1.0, |cc| cc[q % nv] as f64)).sum();
            self.nhat = (num / den.max(1e-30)).max(1e-30);
        }
        // mean Gauss–Newton curvature of one velocity node and component: n̂ c σ_k tr(Σ k kᵀ)/3, σ_k ≈ Σ_f |z_f|² / K_v
        let tr: f64 = s.kenc_model().iter().map(|k| k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sum();
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
        if self.model.s.spline() {
            return self.hvp_spline(st, Src::Dev(th), Src::Dev(d), Dst::Dev(out));
        }
        self.zero(st, out)?;
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        for t in 0..self.op.nsets() {
            let dx = self.set_image(st, th, Some(d), t)?;
            let w = self.weights[t];
            self.set_pass(st, t, dx, None, w, &mut |me, g, wg| me.set_vjp(st, th, t, g, wg, out))?;
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

// ─── at-knots mode with host-resident parameters: θ streamed knot by knot ───────────────────────────────────────────

impl VelProblem<'_> {
    fn ensure_slots(&mut self, st: &Arc<CudaStream>) -> Result<(), Error> {
        if self.slots.is_none() {
            let nv = self.model.s.nv();
            let nvv = (3 * nv).div_ceil(2);
            self.slots = Some(KnotSlots { z: st.alloc_zeros(nv)?, v: st.alloc_zeros(nvv)?, dz: st.alloc_zeros(nv)?, dv: st.alloc_zeros(nvv)?, gz: st.alloc_zeros(nv)?, gv: st.alloc_zeros(nvv)? });
        }
        Ok(())
    }

    /// device bytes of the streamed mode's node slots
    pub fn slot_bytes(&self) -> usize {
        let nv = self.model.s.nv();
        3 * (nv + (3 * nv).div_ceil(2)) * 8
    }

    /// the set image of encode e (all K knots, knot-major) from host θ, or its linearisation J d (`d`)
    fn knot_image_host(&mut self, st: &Arc<CudaStream>, th: &HostStack, d: Option<(&HostStack, &CudaSlice<C32>)>, a: &CudaSlice<C32>, e: usize) -> Result<CudaSlice<C32>, Error> {
        self.ensure_slots(st)?;
        let s = self.model.s.clone();
        let (nv, k) = (s.nv(), s.kz);
        let mut xe = unsafe { st.alloc::<C32>(k * nv)? };
        let xp = raw(&xe, st);
        let ap = raw(a, st);
        for kk in 0..k {
            let sl = self.slots.as_mut().unwrap();
            st.memcpy_htod(th.chunk(s.chunk_z(kk)), &mut sl.z)?;
            st.memcpy_htod(th.chunk(s.chunk_v(kk)), &mut sl.v)?;
            let out = xp + (kk * nv * 8) as u64;
            match d {
                None => self.model.eval_knot(st, raw(&sl.z, st), raw(&sl.v, st), ap, e, out)?,
                Some((dh, da)) => {
                    st.memcpy_htod(dh.chunk(s.chunk_z(kk)), &mut sl.dz)?;
                    st.memcpy_htod(dh.chunk(s.chunk_v(kk)), &mut sl.dv)?;
                    self.model.jvp_knot(st, raw(&sl.z, st), raw(&sl.v, st), ap, raw(&sl.dz, st), raw(&sl.dv, st), raw(da, st), e, out)?;
                }
            }
        }
        let _ = &mut xe;
        Ok(xe)
    }

    /// grad += w Jᴴ g_e over all knots (g_e knot-major like the set image), θ and the gradient chunks streamed;
    /// `first`: the gradient chunks are still zero (not uploaded)
    #[allow(clippy::too_many_arguments)]
    fn knot_vjp_host(&mut self, st: &Arc<CudaStream>, th: &HostStack, a: &CudaSlice<C32>, e: usize, ge: &CudaSlice<C32>, w: f32, grad: &mut HostStack, first: bool) -> Result<(), Error> {
        self.ensure_slots(st)?;
        let s = self.model.s.clone();
        let (nv, k) = (s.nv(), s.kz);
        let (gp, ap) = (raw(ge, st), raw(a, st));
        let sp = raw(&self.sacc[e], st);
        for kk in 0..k {
            let sl = self.slots.as_mut().unwrap();
            st.memcpy_htod(th.chunk(s.chunk_z(kk)), &mut sl.z)?;
            st.memcpy_htod(th.chunk(s.chunk_v(kk)), &mut sl.v)?;
            if first {
                st.memset_zeros(&mut sl.gz)?;
                st.memset_zeros(&mut sl.gv)?;
            } else {
                st.memcpy_htod(grad.chunk(s.chunk_z(kk)), &mut sl.gz)?;
                st.memcpy_htod(grad.chunk(s.chunk_v(kk)), &mut sl.gv)?;
            }
            self.model.vjp_knot(st, raw(&sl.z, st), raw(&sl.v, st), ap, e, gp + (kk * nv * 8) as u64, w, raw(&sl.gz, st), raw(&sl.gv, st), sp)?;
            st.memcpy_dtoh(&sl.gz, grad.chunk_mut(s.chunk_z(kk)))?;
            st.memcpy_dtoh(&sl.gv, grad.chunk_mut(s.chunk_v(kk)))?;
        }
        Ok(())
    }

    /// host gradient chunk a = −Σ_r P_p S_e (from the per-encode accumulators)
    fn finish_a_host(&mut self, st: &Arc<CudaStream>, grad: &mut HostStack) -> Result<(), Error> {
        let s = self.model.s.clone();
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
        grad.chunk_mut(s.chunk_a()).copy_from_slice(&host);
        Ok(())
    }

    /// static-tissue anchor on host chunks: grad += λ w v (all v chunks), returns ½ λ Σ w |v|²
    fn anchor_host_term(&self, v: &HostStack, grad: &mut HostStack) -> f64 {
        if self.anchor.is_none() || self.anchor_host.is_empty() {
            return 0.0;
        }
        let s = &self.model.s;
        let nv = s.nv();
        let (lam, w) = (self.anchor_lam as f32, &self.anchor_host);
        let mut e = 0.0;
        for k in 0..s.kv {
            let c = s.chunk_v(k);
            let vf: Vec<f32> = v.chunk(c).iter().flat_map(|q| [q.re, q.im]).collect();
            let mut gf: Vec<f32> = grad.chunk(c).iter().flat_map(|q| [q.re, q.im]).collect();
            e += (0..3)
                .map(|a| {
                    gf[a * nv..(a + 1) * nv].par_iter_mut().zip(vf[a * nv..(a + 1) * nv].par_iter()).zip(w.par_iter()).map(|((g, &x), &wr)| {
                        if wr == 0.0 {
                            return 0.0;
                        }
                        *g += lam * wr * x;
                        0.5 * lam as f64 * wr as f64 * x as f64 * x as f64
                    }).sum::<f64>()
                })
                .sum::<f64>();
            for (i, q) in grad.chunk_mut(c).iter_mut().enumerate() {
                *q = C32::new(gf[2 * i], *gf.get(2 * i + 1).unwrap_or(&0.0));
            }
        }
        e
    }

    fn zero_frozen_host(&self, g: &mut HostStack) {
        if self.freeze_v {
            let s = &self.model.s;
            for c in s.kz..s.kz + s.kv + 1 {
                g.chunk_mut(c).iter_mut().for_each(|q| *q = C32::new(0.0, 0.0));
            }
        }
    }

    /// at-knots gradient on host-resident θ / g (see the module docs)
    fn gradient_knots_host(&mut self, st: &Arc<CudaStream>, th: &HostStack, grad: &mut HostStack) -> Result<f64, Error> {
        if !self.have_metric || (self.metric_every > 0 && self.calls % self.metric_every == 0) {
            self.update_metric_host(st, th)?;
        }
        self.calls += 1;
        grad.data.par_iter_mut().for_each(|q| *q = C32::new(0.0, 0.0));
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        let s = self.model.s.clone();
        let a = st.clone_htod(th.chunk(s.chunk_a()))?;
        let order: Vec<usize> = (0..self.op.nsets()).filter(|&t| self.weights[t] != 0.0).collect();
        let mut fval = 0.0;
        let mut first = true;
        let mut body = |me: &mut Self, t: usize, y: &CudaSlice<C32>, first: &mut bool| -> Result<(), Error> {
            let w = me.weights[t];
            let xe = me.knot_image_host(st, th, None, &a, t)?;
            fval += me.set_pass(st, t, xe, Some(y), w, &mut |me, ge, wg| {
                me.knot_vjp_host(st, th, &a, t, ge, wg, grad, *first)?;
                *first = false;
                Ok(())
            })?;
            Ok(())
        };
        if let Some(mut sg) = self.stage.take() {
            let staged: Vec<usize> = order.iter().copied().filter(|&t| matches!(self.ys[t], YData::Staged(_))).collect();
            let r = sg.for_each(st, &staged, None, &mut |t, y| body(self, t, y, &mut first));
            self.stage = Some(sg);
            r?;
        }
        for &t in &order {
            if matches!(self.ys[t], YData::Staged(_)) {
                continue;
            }
            // pinned host data: uploaded for this set; resident data: borrowed from `ys` (taken out while `body` runs)
            let y = match &self.ys[t] {
                YData::Host(hd) => {
                    let mut d = st.alloc_zeros::<C32>(hd.len())?;
                    st.memcpy_htod(hd, &mut d)?;
                    Some(d)
                }
                _ => None,
            };
            match y {
                Some(d) => body(self, t, &d, &mut first)?,
                None => {
                    // resident data: borrow through a raw clone-free path (the set stays in `ys`)
                    let ys = std::mem::take(&mut self.ys);
                    let r = if let YData::Dev(d) = &ys[t] { body(self, t, d, &mut first) } else { unreachable!() };
                    self.ys = ys;
                    r?;
                }
            }
        }
        self.finish_a_host(st, grad)?;
        fval += self.anchor_host_term(th, grad);
        self.zero_frozen_host(grad);
        Ok(fval)
    }

    /// out = H d on host-resident θ, d (Gauss–Newton Hessian-vector product, streamed; at-knots mode)
    pub fn hvp_host(&mut self, st: &Arc<CudaStream>, th: &HostStack, d: &HostStack, out: &mut HostStack) -> Result<(), Error> {
        if self.model.s.spline() {
            return self.hvp_spline(st, Src::Host(th), Src::Host(d), Dst::Host(out));
        }
        assert!(self.model.s.identity(), "hvp_host: at-knots mode");
        out.data.par_iter_mut().for_each(|q| *q = C32::new(0.0, 0.0));
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        let s = self.model.s.clone();
        let a = st.clone_htod(th.chunk(s.chunk_a()))?;
        let da = st.clone_htod(d.chunk(s.chunk_a()))?;
        let mut first = true;
        for t in 0..self.op.nsets() {
            let w = self.weights[t];
            if w == 0.0 {
                continue;
            }
            let dx = self.knot_image_host(st, th, Some((d, &da)), &a, t)?;
            self.set_pass(st, t, dx, None, w, &mut |me, g, wg| {
                me.knot_vjp_host(st, th, &a, t, g, wg, out, first)?;
                first = false;
                Ok(())
            })?;
        }
        self.finish_a_host(st, out)?;
        // the anchor's curvature: λ w d
        if self.anchor.is_some() && !self.anchor_host.is_empty() {
            let _ = self.anchor_host_term(d, out);
        }
        self.zero_frozen_host(out);
        Ok(())
    }

    /// λ_max of M H (`with_metric`) or H by power iteration on host stacks (two extra host stacks; at-knots mode)
    pub fn lambda_max_host(&mut self, st: &Arc<CudaStream>, th: &HostStack, with_metric: bool, iters: usize) -> Result<f64, Error> {
        if with_metric && !self.have_metric {
            self.update_metric_host(st, th)?;
        }
        let lens = th.lens.clone();
        let mut v = th.clone();
        let mut hv = HostStack::zeros(&lens);
        let maxl = lens.iter().copied().max().unwrap_or(0);
        let mut tmp = vec![C32::new(0.0, 0.0); maxl];
        let mut lam = 0.0;
        for _ in 0..iters {
            let mut n2 = 0.0;
            for t in 0..lens.len() {
                if with_metric {
                    let l = lens[t];
                    self.metric_host(st, t, v.chunk(t), &mut tmp[..l], true)?;
                    n2 += v.chunk(t).par_iter().zip(tmp[..l].par_iter()).map(|(a, b)| a.re as f64 * b.re as f64 + a.im as f64 * b.im as f64).sum::<f64>();
                } else {
                    n2 += v.chunk(t).par_iter().map(|a| a.re as f64 * a.re as f64 + a.im as f64 * a.im as f64).sum::<f64>();
                }
            }
            let sc = (1.0 / n2.max(1e-300).sqrt()) as f32;
            v.data.par_iter_mut().for_each(|q| *q = C32::new(q.re * sc, q.im * sc));
            self.hvp_host(st, th, &v, &mut hv)?;
            lam = HostStack::rdot(&v, &hv);
            for t in 0..lens.len() {
                if with_metric {
                    let l = lens[t];
                    self.metric_host(st, t, hv.chunk(t), &mut tmp[..l], false)?;
                    v.chunk_mut(t).copy_from_slice(&tmp[..l]);
                } else {
                    let h = hv.chunk(t).to_vec();
                    v.chunk_mut(t).copy_from_slice(&h);
                }
            }
        }
        Ok(lam)
    }
}

// ─── parameter-spline mode (research/velocity_model_sr.md §F): sets (encode, respiratory group), rows streamed ────────

/// θ / d for the spline-mode passes: device stack, or host stack streamed one respiratory row (mc knots) at a time
#[derive(Clone, Copy)]
enum Src<'s> {
    Dev(&'s Stack),
    Host(&'s HostStack),
}

/// the gradient / Hessian-vector output: device stack, or host stack (rows downloaded after each set's adjoint)
enum Dst<'s> {
    Dev(&'s mut Stack),
    Host(&'s mut HostStack),
}

/// one respiratory row on the device: the (m, ψ) chunks and v chunks of its mc cardiac knots
struct RowSlot {
    z: CudaSlice<C32>,
    v: CudaSlice<C32>,
}

impl VelProblem<'_> {
    /// device bytes of the two row slots of the host-streamed spline mode
    pub fn row_slot_bytes(&self) -> usize {
        let s = &self.model.s;
        let Some(of) = &s.op else { return 0 };
        let nv = s.nv();
        2 * of.mc * (nv + (3 * nv).div_ceil(2)) * 8
    }

    fn ensure_rows(&mut self, st: &Arc<CudaStream>) -> Result<(), Error> {
        if self.rows.is_none() {
            let s = &self.model.s;
            let mc = s.op.as_ref().unwrap().mc;
            let (nv, nvv) = (s.nv(), (3 * s.nv()).div_ceil(2));
            let mk = || -> Result<RowSlot, Error> { Ok(RowSlot { z: st.alloc_zeros(mc * nv)?, v: st.alloc_zeros(mc * nvv)? }) };
            self.rows = Some([mk()?, mk()?]);
        }
        Ok(())
    }

    /// device pointers of respiratory row r's chunks: straight into a device stack, or uploaded into row slot `slot`
    fn row_ptrs(&mut self, st: &Arc<CudaStream>, src: Src, r: usize, slot: usize) -> Result<(Vec<u64>, Vec<u64>), Error> {
        let s = self.model.s.clone();
        let mc = s.op.as_ref().unwrap().mc;
        match src {
            Src::Dev(x) => Ok(((0..mc).map(|c| raw(&x.v[s.chunk_z(r * mc + c)], st)).collect(), (0..mc).map(|c| raw(&x.v[s.chunk_v(r * mc + c)], st)).collect())),
            Src::Host(h) => {
                self.ensure_rows(st)?;
                let (nv, nvv) = (s.nv(), (3 * s.nv()).div_ceil(2));
                let sl = &mut self.rows.as_mut().unwrap()[slot];
                for c in 0..mc {
                    st.memcpy_htod(h.chunk(s.chunk_z(r * mc + c)), &mut sl.z.slice_mut(c * nv..(c + 1) * nv))?;
                    st.memcpy_htod(h.chunk(s.chunk_v(r * mc + c)), &mut sl.v.slice_mut(c * nvv..(c + 1) * nvv))?;
                }
                let (zb, vb) = (raw(&sl.z, st), raw(&sl.v, st));
                Ok(((0..mc).map(|c| zb + (c * nv * 8) as u64).collect(), (0..mc).map(|c| vb + (c * nvv * 8) as u64).collect()))
            }
        }
    }

    /// the set image of set t = (e, g): the model (or its linearisation J d, `d` = (d, its residual-phase chunk)) at the
    /// operator frames of the group's kre respiratory knots, cardiac-interpolation-filtered (the operator's coefficients)
    fn spline_image(&mut self, st: &Arc<CudaStream>, th: Src, d: Option<(Src, u64)>, a: u64, t: usize) -> Result<CudaSlice<C32>, Error> {
        let s = self.model.s.clone();
        let of = s.op.as_ref().unwrap();
        let (e, g) = of.set_eg(t);
        let (nv, fop, kre) = (s.nv(), of.fop, of.kre);
        let x = unsafe { st.alloc::<C32>(kre * fop * nv)? };
        let xp = raw(&x, st);
        for rl in 0..kre {
            let r = of.r0[g] + rl;
            let (zr, vr) = self.row_ptrs(st, th, r, 0)?;
            let out = xp + (rl * fop * nv * 8) as u64;
            match d {
                None => self.model.eval_row(st, &zr, &vr, a, e, out)?,
                Some((ds, da)) => {
                    let (dzr, dvr) = self.row_ptrs(st, ds, r, 1)?;
                    self.model.jvp_row(st, &zr, &vr, &dzr, &dvr, a, da, e, out)?;
                }
            }
        }
        self.model.prefilter(st, xp, kre)?;
        Ok(x)
    }

    /// grad += w Jᴴ (filter g): the adjoint of the set's operator image `g` (filtered in place: the filter is its own
    /// adjoint), then the row gradients; host output rows are uploaded (or zeroed on first touch), accumulated, downloaded
    #[allow(clippy::too_many_arguments)]
    fn spline_vjp(&mut self, st: &Arc<CudaStream>, th: Src, a: u64, t: usize, g: &CudaSlice<C32>, w: f32, grad: &mut Dst, touched: &mut [bool]) -> Result<(), Error> {
        let s = self.model.s.clone();
        let of = s.op.as_ref().unwrap();
        let (e, gi) = of.set_eg(t);
        let (nv, fop, kre, mc) = (s.nv(), of.fop, of.kre, of.mc);
        let nvv = (3 * nv).div_ceil(2);
        let gp = raw(g, st);
        self.model.prefilter(st, gp, kre)?;
        let sp = raw(&self.sacc[e], st);
        for rl in 0..kre {
            let r = of.r0[gi] + rl;
            let (zr, vr) = self.row_ptrs(st, th, r, 0)?;
            let gr = gp + (rl * fop * nv * 8) as u64;
            match grad {
                Dst::Dev(gd) => {
                    let gz: Vec<u64> = (0..mc).map(|c| raw(&gd.v[s.chunk_z(r * mc + c)], st)).collect();
                    let gv: Vec<u64> = (0..mc).map(|c| raw(&gd.v[s.chunk_v(r * mc + c)], st)).collect();
                    self.model.vjp_row(st, &zr, &vr, a, e, gr, w, &gz, &gv, sp)?;
                }
                Dst::Host(gh) => {
                    self.ensure_rows(st)?;
                    let sl = &mut self.rows.as_mut().unwrap()[1];
                    if touched[r] {
                        for c in 0..mc {
                            st.memcpy_htod(gh.chunk(s.chunk_z(r * mc + c)), &mut sl.z.slice_mut(c * nv..(c + 1) * nv))?;
                            st.memcpy_htod(gh.chunk(s.chunk_v(r * mc + c)), &mut sl.v.slice_mut(c * nvv..(c + 1) * nvv))?;
                        }
                    } else {
                        st.memset_zeros(&mut sl.z)?;
                        st.memset_zeros(&mut sl.v)?;
                    }
                    let (zb, vb) = (raw(&sl.z, st), raw(&sl.v, st));
                    let gz: Vec<u64> = (0..mc).map(|c| zb + (c * nv * 8) as u64).collect();
                    let gv: Vec<u64> = (0..mc).map(|c| vb + (c * nvv * 8) as u64).collect();
                    self.model.vjp_row(st, &zr, &vr, a, e, gr, w, &gz, &gv, sp)?;
                    let sl = &self.rows.as_ref().unwrap()[1];
                    for c in 0..mc {
                        st.memcpy_dtoh(&sl.z.slice(c * nv..(c + 1) * nv), gh.chunk_mut(s.chunk_z(r * mc + c)))?;
                        st.memcpy_dtoh(&sl.v.slice(c * nvv..(c + 1) * nvv), gh.chunk_mut(s.chunk_v(r * mc + c)))?;
                    }
                    touched[r] = true;
                }
            }
        }
        Ok(())
    }

    /// run `body(self, t, y_t)` over the sets `order` with their data on the device (staged from the store, uploaded
    /// from pinned memory, or resident)
    fn for_each_data(&mut self, st: &Arc<CudaStream>, order: &[usize], body: &mut dyn FnMut(&mut Self, usize, &CudaSlice<C32>) -> Result<(), Error>) -> Result<(), Error> {
        if let Some(mut sg) = self.stage.take() {
            let staged: Vec<usize> = order.iter().copied().filter(|&t| matches!(self.ys[t], YData::Staged(_))).collect();
            let r = sg.for_each(st, &staged, None, &mut |t, y| body(self, t, y));
            self.stage = Some(sg);
            r?;
        }
        for &t in order {
            match &self.ys[t] {
                YData::Staged(_) => continue,
                YData::Host(hd) => {
                    let mut d = st.alloc_zeros::<C32>(hd.len())?;
                    st.memcpy_htod(hd, &mut d)?;
                    body(self, t, &d)?;
                }
                YData::Dev(_) => {
                    let ys = std::mem::take(&mut self.ys);
                    let r = if let YData::Dev(d) = &ys[t] { body(self, t, d) } else { unreachable!() };
                    self.ys = ys;
                    r?;
                }
            }
        }
        Ok(())
    }

    /// the residual-phase chunk of a source on the device
    fn a_chunk(&self, st: &Arc<CudaStream>, x: Src) -> Result<(Option<CudaSlice<C32>>, u64), Error> {
        let ca = self.model.s.chunk_a();
        Ok(match x {
            Src::Dev(d) => (None, raw(&d.v[ca], st)),
            Src::Host(h) => {
                let b = st.clone_htod(h.chunk(ca))?;
                let p = raw(&b, st);
                (Some(b), p)
            }
        })
    }

    /// residual-phase gradient, the anchor (value and gradient, or its curvature for `d`), frozen parameters
    fn spline_finish(&mut self, st: &Arc<CudaStream>, v: Src, out: &mut Dst) -> Result<f64, Error> {
        let mut fval = 0.0;
        match out {
            Dst::Dev(gd) => {
                self.finish_a(st, gd)?;
                if let (Some((w, _)), Src::Dev(x)) = (&self.anchor, v) {
                    for k in 0..self.model.s.kv {
                        let c = self.model.s.chunk_v(k);
                        fval += self.model.anchor(st, &x.v[c], w, self.anchor_lam as f32, &mut gd.v[c])?;
                    }
                }
                self.zero_frozen(st, gd)?;
            }
            Dst::Host(gh) => {
                self.finish_a_host(st, gh)?;
                if let Src::Host(x) = v {
                    fval += self.anchor_host_term(x, gh);
                }
                self.zero_frozen_host(gh);
            }
        }
        Ok(fval)
    }

    fn zero_dst(&self, st: &Arc<CudaStream>, out: &mut Dst) -> Result<(), Error> {
        match out {
            Dst::Dev(g) => self.zero(st, g),
            Dst::Host(g) => {
                g.data.par_iter_mut().for_each(|q| *q = C32::new(0.0, 0.0));
                Ok(())
            }
        }
    }

    /// spline-mode data term and gradient
    fn gradient_spline(&mut self, st: &Arc<CudaStream>, th: Src, mut grad: Dst) -> Result<f64, Error> {
        if !self.have_metric || (self.metric_every > 0 && self.calls % self.metric_every == 0) {
            match th {
                Src::Dev(x) => self.update_metric(st, x)?,
                Src::Host(x) => self.update_metric_host(st, x)?,
            }
        }
        self.calls += 1;
        self.zero_dst(st, &mut grad)?;
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        let (_ab, a) = self.a_chunk(st, th)?;
        let order: Vec<usize> = (0..self.op.nsets()).filter(|&t| self.weights[t] != 0.0).collect();
        let mut touched = vec![false; self.model.s.op.as_ref().unwrap().kr];
        let mut fval = 0.0;
        self.for_each_data(st, &order, &mut |me, t, y| {
            let w = me.weights[t];
            let x = me.spline_image(st, th, None, a, t)?;
            fval += me.set_pass(st, t, x, Some(y), w, &mut |me, g, wg| me.spline_vjp(st, th, a, t, g, wg, &mut grad, &mut touched))?;
            Ok(())
        })?;
        fval += self.spline_finish(st, th, &mut grad)?;
        Ok(fval)
    }

    /// spline-mode Gauss–Newton Hessian-vector product out = H d (+ the anchor's curvature)
    fn hvp_spline(&mut self, st: &Arc<CudaStream>, th: Src, d: Src, mut out: Dst) -> Result<(), Error> {
        self.zero_dst(st, &mut out)?;
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        let (_ab, a) = self.a_chunk(st, th)?;
        let (_db, da) = self.a_chunk(st, d)?;
        let mut touched = vec![false; self.model.s.op.as_ref().unwrap().kr];
        for t in 0..self.op.nsets() {
            let w = self.weights[t];
            if w == 0.0 {
                continue;
            }
            let dx = self.spline_image(st, th, Some((d, da)), a, t)?;
            self.set_pass(st, t, dx, None, w, &mut |me, g, wg| me.spline_vjp(st, th, a, t, g, wg, &mut out, &mut touched))?;
        }
        let _ = self.spline_finish(st, d, &mut out)?;
        Ok(())
    }
}

impl VelProblem<'_> {
    /// forward of set t for complex parameter-basis knot images at device address `xb` (K · nv, knot-major): the rows'
    /// operator frames (linear in the knots), filtered, through the operator
    fn cf_forward(&mut self, st: &Arc<CudaStream>, t: usize, xb: u64, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let s = self.model.s.clone();
        let of = s.op.as_ref().unwrap();
        let (_, g) = of.set_eg(t);
        let (nv, fop, kre, mc) = (s.nv(), of.fop, of.kre, of.mc);
        let img = unsafe { st.alloc::<C32>(kre * fop * nv)? };
        let ip = raw(&img, st);
        for rl in 0..kre {
            let r = of.r0[g] + rl;
            let crow: Vec<u64> = (0..mc).map(|c| xb + (((r * mc + c) * nv) * 8) as u64).collect();
            self.model.cframes_row(st, &crow, ip + (rl * fop * nv * 8) as u64)?;
        }
        self.model.prefilter(st, ip, kre)?;
        self.op.forward(st, t, &img, y)
    }

    /// adjoint of [`cf_forward`](Self::cf_forward), accumulated into the knot images at `xb`
    fn cf_adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, xb: u64) -> Result<(), Error> {
        let s = self.model.s.clone();
        let of = s.op.as_ref().unwrap();
        let (_, g) = of.set_eg(t);
        let (nv, fop, kre, mc) = (s.nv(), of.fop, of.kre, of.mc);
        let mut img = unsafe { st.alloc::<C32>(kre * fop * nv)? };
        self.op.adjoint(st, t, y, &mut img)?;
        let ip = raw(&img, st);
        self.model.prefilter(st, ip, kre)?;
        for rl in 0..kre {
            let r = of.r0[g] + rl;
            let crow: Vec<u64> = (0..mc).map(|c| xb + (((r * mc + c) * nv) * 8) as u64).collect();
            self.model.cframes_adj_row(st, ip + (rl * fop * nv * 8) as u64, &crow)?;
        }
        Ok(())
    }

    /// device bytes of [`complex_knots`](Self::complex_knots) beyond the operator: three knot stacks (x, s, p), the
    /// residuals of the encode's sets and one forward buffer, one set image
    pub fn complex_knots_bytes(&self, e: usize) -> usize {
        let s = &self.model.s;
        let Some(of) = &s.op else { return 0 };
        let ys: usize = (0..of.ng()).map(|g| self.op.len_y(e * of.ng() + g)).sum::<usize>() + (0..of.ng()).map(|g| self.op.len_y(e * of.ng() + g)).max().unwrap_or(0);
        (3 * s.kz * s.nv() + ys + of.set_frames() * s.nv()) * 8
    }

    /// Spline mode, robust start (research/velocity_model_sr.md §F): complex knot images C of encode e in the parameter
    /// basis (K = kr · mc images, knot-major; the operator sees x(r, f) = Σ_c w_c(f) C_{r,c} at its frames), by `iters`
    /// CGLS iterations on Σ_g ‖A_g C − y_g‖² + λ‖C‖² over the encode's sets, started from `x0` (nv) in every knot.
    /// Returns (C on the host, ‖s‖/‖s₀‖ per iteration).
    pub fn complex_knots(&mut self, st: &Arc<CudaStream>, e: usize, x0: &[C32], iters: usize, lambda: f64) -> Result<(Vec<C32>, Vec<f64>), Error> {
        let s = self.model.s.clone();
        let of = s.op.clone().expect("complex_knots: spline mode");
        let (nv, k, ng) = (s.nv(), s.kz, of.ng());
        assert_eq!(x0.len(), nv);
        let saved_groups = self.cur_groups.take();
        let ts: Vec<usize> = (0..ng).map(|g| e * ng + g).collect();
        let kern = self.kern.clone();
        let one = Complex64::new(1.0, 0.0);
        let mut c = st.alloc_zeros::<C32>(k * nv)?;
        {
            let x0d = st.clone_htod(x0)?;
            for kk in 0..k {
                st.memcpy_dtod(&x0d, &mut c.slice_mut(kk * nv..(kk + 1) * nv))?;
            }
        }
        let cb = raw(&c, st);
        // residuals r_g = y_g − A_g C0
        let mut rs: Vec<Option<CudaSlice<C32>>> = (0..ng).map(|_| None).collect();
        self.for_each_data(st, &ts, &mut |me, t, y| {
            let mut b = st.alloc_zeros::<C32>(me.op.len_y(t))?;
            me.cf_forward(st, t, cb, &mut b)?;
            me.kern.axpby(st, one, y, Complex64::new(-1.0, 0.0), &mut b)?;
            rs[t % ng] = Some(b);
            Ok(())
        })?;
        let mut rs: Vec<CudaSlice<C32>> = rs.into_iter().map(|r| r.expect("complex_knots: every set's data")).collect();
        let mut sv = st.alloc_zeros::<C32>(k * nv)?;
        let normal_rhs = |me: &mut Self, rs: &[CudaSlice<C32>], c: &CudaSlice<C32>, sv: &mut CudaSlice<C32>| -> Result<f64, Error> {
            st.memset_zeros(sv)?;
            let sb = raw(sv, st);
            for (g, r) in rs.iter().enumerate() {
                me.cf_adjoint(st, e * ng + g, r, sb)?;
            }
            if lambda > 0.0 {
                me.kern.axpby(st, Complex64::new(-lambda, 0.0), c, one, sv)?;
            }
            Ok(me.kern.dot(st, sv, sv)?.re)
        };
        let mut gamma = normal_rhs(self, &rs, &c, &mut sv)?;
        let g0 = gamma.max(1e-300);
        let mut p = sv.clone();
        let mut hist = vec![];
        for _ in 0..iters {
            let pb = raw(&p, st);
            let mut qs = vec![];
            let mut qq = if lambda > 0.0 { lambda * kern.dot(st, &p, &p)?.re } else { 0.0 };
            for &t in &ts {
                let mut q = st.alloc_zeros::<C32>(self.op.len_y(t))?;
                self.cf_forward(st, t, pb, &mut q)?;
                qq += kern.dot(st, &q, &q)?.re;
                qs.push(q);
            }
            let alpha = gamma / qq.max(1e-300);
            kern.axpby(st, Complex64::new(alpha, 0.0), &p, one, &mut c)?;
            for (r, q) in rs.iter_mut().zip(&qs) {
                kern.axpby(st, Complex64::new(-alpha, 0.0), q, one, r)?;
            }
            drop(qs);
            let gn = normal_rhs(self, &rs, &c, &mut sv)?;
            let beta = gn / gamma.max(1e-300);
            kern.axpby(st, one, &sv, Complex64::new(beta, 0.0), &mut p)?;
            gamma = gn;
            hist.push((gn / g0).sqrt());
        }
        self.cur_groups = saved_groups;
        Ok((st.clone_dtoh(&c)?, hist))
    }
}

impl Objective for VelProblem<'_> {
    fn kernels(&self) -> &Kernels {
        &self.kern
    }

    fn has_step_limit(&self) -> bool {
        self.vtrust.is_some() && !self.freeze_v
    }

    /// the velocity trust region on a device chunk (v chunks only; through the host)
    fn limit_step(&mut self, st: &Arc<CudaStream>, t: usize, step: &mut CudaSlice<C32>) -> Result<usize, Error> {
        let s = &self.model.s;
        if self.vtrust.is_none() || (t < s.kz && !s.spline()) || t >= s.kz + s.kv {
            return Ok(0);
        }
        let mut h = st.clone_dtoh(step)?;
        let n = self.limit_step_host(t, &mut h);
        if n > 0 {
            st.memcpy_htod(&h, step)?;
        }
        Ok(n)
    }

    /// the velocity trust region: chunk t = v of one knot ([3][nv] f32 packed in pairs); per voxel the step's velocity
    /// 3-vector is scaled so that max_e |k'_e·Δv| ≤ ρ (direction kept); spline mode also |Δψ'| ≤ ρ on the (m, ψ')
    /// chunks; returns the number of elements shrunk
    fn limit_step_host(&mut self, t: usize, step: &mut [C32]) -> usize {
        let s = &self.model.s;
        let Some(rho) = self.vtrust else { return 0 };
        if t < s.kz {
            // spline mode: (m, ψ') per voxel; the zero-moment phase is as periodic as the velocity phase: |Δψ'| ≤ ρ
            if !s.spline() {
                return 0;
            }
            let r = rho as f32;
            return step.par_iter_mut().map(|c| if c.im.abs() > r { c.im = c.im.clamp(-r, r); 1usize } else { 0 }).sum();
        }
        if t >= s.kz + s.kv {
            return 0;
        }
        trust_shrink(step, s.nv(), &s.kenc_model(), rho)
    }

    /// coil sampling: the iteration's coil groups (a function of (seed, k): a resumed run draws the same)
    fn begin_iteration(&mut self, k: usize, iters: usize) {
        self.cur_k = k;
        self.iters_total = iters;
        if let Some(s) = &self.sampling {
            self.cur_groups = s.groups(self.op.n_coils(), k, iters);
        }
    }

    /// the secant pair is exact only between gradients of the same coil subset
    fn curvature_pair_ok(&self) -> bool {
        match &self.sampling {
            None => true,
            Some(s) => {
                let (nc, k) = (self.op.n_coils(), self.cur_k);
                k > 0 && s.block(nc, k, self.iters_total) == s.block(nc, k - 1, self.iters_total)
            }
        }
    }

    fn lens(&self) -> Vec<usize> {
        self.model.lens()
    }

    fn gradient(&mut self, st: &Arc<CudaStream>, th: &Stack, grad: &mut Stack) -> Result<f64, Error> {
        if self.model.s.spline() {
            return self.gradient_spline(st, Src::Dev(th), Dst::Dev(grad));
        }
        if !self.have_metric || (self.metric_every > 0 && self.calls % self.metric_every == 0) {
            self.update_metric(st, th)?;
        }
        self.calls += 1;
        self.zero(st, grad)?;
        for s in self.sacc.iter_mut() {
            st.memset_zeros(s)?;
        }
        let mut fval = 0.0;
        if let Some(mut sg) = self.stage.take() {
            // data from the host store: the next set is decoded while the GPU works on this one
            let order: Vec<usize> = (0..self.op.nsets()).filter(|&t| self.weights[t] != 0.0).collect();
            let r = sg.for_each(st, &order, None, &mut |t, y| {
                let w = self.weights[t];
                let x = self.set_image(st, th, None, t)?;
                fval += self.set_pass(st, t, x, Some(y), w, &mut |me, g, wg| me.set_vjp(st, th, t, g, wg, grad))?;
                Ok(())
            });
            self.stage = Some(sg);
            r?;
        }
        for t in 0..self.op.nsets() {
            if matches!(self.ys[t], YData::Staged(_)) {
                continue;
            }
            let w = self.weights[t];
            if w == 0.0 {
                continue;
            }
            let x = self.set_image(st, th, None, t)?;
            // pinned host data: uploaded for this set; resident data: borrowed from `ys` (taken out during the pass)
            let up = match &self.ys[t] {
                YData::Host(h) => {
                    let mut d = st.alloc_zeros::<C32>(h.len())?;
                    st.memcpy_htod(h, &mut d)?;
                    Some(d)
                }
                _ => None,
            };
            match up {
                Some(d) => fval += self.set_pass(st, t, x, Some(&d), w, &mut |me, g, wg| me.set_vjp(st, th, t, g, wg, grad))?,
                None => {
                    let ys = std::mem::take(&mut self.ys);
                    let r = if let YData::Dev(d) = &ys[t] { self.set_pass(st, t, x, Some(d), w, &mut |me, g, wg| me.set_vjp(st, th, t, g, wg, grad)) } else { unreachable!() };
                    self.ys = ys;
                    fval += r?;
                }
            }
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

    /// host-resident θ: at-knots mode streams θ knot by knot (see the module docs); otherwise the whole stack is
    /// uploaded (as the trait's default)
    fn gradient_host(&mut self, st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        if self.model.s.identity() {
            return self.gradient_knots_host(st, x, g);
        }
        if self.model.s.spline() {
            return self.gradient_spline(st, Src::Host(x), Dst::Host(g));
        }
        let lens = x.lens.clone();
        let mut xd = Stack::zeros_lens(st, &lens)?;
        for t in 0..lens.len() {
            st.memcpy_htod(x.chunk(t), &mut xd.v[t])?;
        }
        let mut gd = Stack::zeros_lens(st, &lens)?;
        let f = self.gradient(st, &xd, &mut gd)?;
        for t in 0..lens.len() {
            st.memcpy_dtoh(&gd.v[t], g.chunk_mut(t))?;
        }
        Ok(f)
    }

    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        let s = self.model.s.clone();
        if t < s.kz && s.spline() {
            // (m, ψ): magnitude curvature E ω_k c, phase curvature E (c ω_k m_k² + damping)
            let nv = s.nv();
            if self.sig1.is_none() {
                self.sig1 = Some(st.alloc_zeros::<f32>(nv)?);
            }
            let s1 = self.sig1.as_mut().unwrap();
            st.memcpy_htod(&self.sig_host[t * nv..(t + 1) * nv], s1)?;
            let sp = raw(s1, st);
            return self.model.metric_mp(st, r, z, self.coil.as_ref(), sp, self.wsum_z[t], self.damp_z[t], s.ne() as f32, self.damp_v[t], inverse);
        }
        if t < s.kz {
            return self.model.metric_z(st, r, z, self.coil.as_ref(), self.wsum_z[t], self.damp_z[t], inverse);
        }
        if t < s.kz + s.kv {
            let k = t - s.kz;
            let sig = if s.identity() || s.spline() {
                let nv = s.nv();
                if self.sig1.is_none() {
                    self.sig1 = Some(st.alloc_zeros::<f32>(nv)?);
                }
                let s1 = self.sig1.as_mut().unwrap();
                st.memcpy_htod(&self.sig_host[k * nv..(k + 1) * nv], s1)?;
                raw(s1, st)
            } else {
                raw(&self.sig, st) + (k * s.nv() * 4) as u64
            };
            // the metric is in units of the data curvature / n̂; the anchor's λ w I is trace-matched onto Σ k kᵀ
            let tr: f64 = s.kenc_model().iter().map(|k| k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sum();
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

    /// the velocity trust region (§F.2): a voxel's step is scaled to max_e |k_e·d| = ρ, direction kept; small steps
    /// and the packing padding are untouched (CPU only)
    #[test]
    fn trust_shrink_scales_per_voxel() {
        let nv = 5;
        let km = [[-0.75, -0.75, -0.75], [2.4, -0.75, -0.75], [1.35, -0.75, -0.75]];
        let d: Vec<[f32; 3]> = vec![[0.01, 0.0, 0.0], [1.0, 0.5, -0.2], [-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]];
        let mut f = vec![0f32; 3 * nv + 1];
        for (q, v) in d.iter().enumerate() {
            for a in 0..3 {
                f[a * nv + q] = v[a];
            }
        }
        f[3 * nv] = 7.0;
        let mut c: Vec<C32> = f.chunks(2).map(|p| C32::new(p[0], p[1])).collect();
        let ph = |v: [f32; 3]| km.iter().map(|k| (k[0] * v[0] as f64 + k[1] * v[1] as f64 + k[2] * v[2] as f64).abs()).fold(0.0, f64::max);
        let n = trust_shrink(&mut c, nv, &km, 1.0);
        let g: Vec<f32> = c.iter().flat_map(|z| [z.re, z.im]).collect();
        let out: Vec<[f32; 3]> = (0..nv).map(|q| [g[q], g[nv + q], g[2 * nv + q]]).collect();
        assert_eq!(n, d.iter().filter(|&&v| ph(v) > 1.0).count());
        for (a, b) in d.iter().zip(&out) {
            let (pa, pb) = (ph(*a), ph(*b));
            if pa > 1.0 {
                assert!((pb - 1.0).abs() < 1e-5, "shrunk to ρ: {pb}");
                let s = b[0] / a[0];
                assert!((0..3).all(|i| (b[i] - s * a[i]).abs() < 1e-6), "direction kept");
            } else {
                assert_eq!(a, b);
            }
        }
        assert_eq!(g[3 * nv], 7.0);
    }
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
        VelSpec { dims: [12, 10, 6], kenc, nf: 4, kz: 2, deg_z: 1, kv: 4, deg_v: 1, poly: Some(1), knots: None, op: None }
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

/// per-voxel velocity trust region on a packed v chunk ([3][nv] f32 in pairs of C32): the 3-vector d of voxel q is
/// scaled by min(1, ρ / max_e |k_e·d|); returns the number of voxels shrunk (research/velocity_model_sr.md §F.2)
pub fn trust_shrink(step: &mut [C32], nv: usize, km: &[[f64; 3]], rho: f64) -> usize {
    assert!(2 * step.len() >= 3 * nv, "packed v chunk");
    // SAFETY: C32 = num_complex::Complex<f32> is #[repr(C)] (re, im): the chunk is 2·len contiguous f32
    let f: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(step.as_mut_ptr() as *mut f32, 2 * step.len()) };
    let (fx, rest) = f.split_at_mut(nv);
    let (fy, rest) = rest.split_at_mut(nv);
    let fz = &mut rest[..nv];
    fx.par_iter_mut().zip(fy.par_iter_mut()).zip(fz.par_iter_mut()).map(|((x, y), z)| {
        let ph = km.iter().map(|k| (k[0] * *x as f64 + k[1] * *y as f64 + k[2] * *z as f64).abs()).fold(0.0, f64::max);
        if ph > rho {
            let c = (rho / ph) as f32;
            *x *= c;
            *y *= c;
            *z *= c;
            1usize
        } else {
            0
        }
    }).sum()
}
