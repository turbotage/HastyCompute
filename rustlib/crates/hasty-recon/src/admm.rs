//! ADMM with an exactly solved per-set data step (research/precond_sampling_theory.md §2–§5; recon_p2np
//! `--prior-mode admm`).
//!
//!   x_t ← argmin_x ½‖E_t x − y_t‖² + (ρ/2)‖x − (z_t − u_t)‖²   K PCG iterations on N_t + ρI, warm start x_t
//!   z   ← D(x + u)                                                the P²nP denoiser pipeline (LLR, wavelet, DFW, lock)
//!   u   ← u + x − z                                               (scaled dual)
//!
//! Why. P²nP, x ← D(x − αP∇f(x)) with a Euclidean denoiser D = prox_{λR}, has fixed points ∇f(x) + (λ/α)P⁻¹g = 0
//! (g ∈ ∂R(x)): the metric P reweights the prior, and a density-inverse P is density compensation. ADMM's fixed point is
//! the minimiser of f + λR whatever the inner preconditioner: the metric only changes how fast the data step converges.
//! The data term is block-diagonal over the sets, so the data step is one independent PCG per set (own α, β).
//!
//! Fixed point and ρ. At a fixed point x = z and u = −∇f(z)/ρ, so z = D(z − ∇f(z)/ρ): the fixed point of P²nP with
//! step 1/ρ, no metric, and the same D. For D = prox_{τR} with τ = λ/ρ it is the minimiser of f + λR, the same for
//! every ρ (test `fixed_point_invariant_to_rho`). The production thresholds are data-adaptive (LLR κσ̂, wavelet κ·MAD,
//! DFW SURE): σ̂ is re-estimated from D's input x + u = z − ∇f(z)/ρ. Where σ̂ is set by that data-gradient part (∝ 1/ρ),
//! τ = κσ̂ ∝ 1/ρ already, so the effective strength λ = ρτ = κ(ρσ̂) does not move with ρ: the adaptive threshold
//! carries the 1/ρ scale itself, and κ keeps its meaning (no κ rescaling with ρ). Where σ̂ is set by z's own spectrum the
//! invariance is approximate; compare ρ arms by their fixed points, not by κ.
//!
//! Units of ρ. ρ₁ = the curvature of one sample in one Nyquist cell, coil energy summed: per set
//! N_t ≈ D^{1/2} F⁻¹ diag(c · n_t(k)) F D^{1/2} (n_t = samples per cell, D = Σ_c |S_c|²), ρ₁ = c · D_body. For an
//! unnormalised NUFFT (A x(ω_j) = Σ_n x_n e^{∓iω_j·n}) the trace gives c = nv (times the mean physics attenuation
//! |e^{−(R+iω)τ}|² of the samples); recon_p2np measures c by a Hutchinson trace of N_t and checks it on plane waves.
//!
//! Inner metric (`clip`): P_t = D^{-1/2} F⁻¹ diag(1 / (max(μ̃_t(k), 1) + ρ/ρ₁)) F D^{-1/2}: density compensation only
//! where a set holds at least one expected sample per cell (§3), scalar elsewhere ([`clip_symbol`], [`ClipMetric`]).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use rayon::prelude::*;

use crate::checkpoint::{Outcome, RunCtl};
use crate::gpu::Kernels;
use crate::hoststack::{clear_raw, load_raw, save_raw, HostStack};
use crate::p2np::Objective;
use crate::precond::CirculantPrecond;
use crate::Error;

const PAR: usize = 1 << 16;

/// The inner metric of the data step
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmmMetric {
    /// plain CG on N_t + ρI
    None,
    /// D^{-1/2} F⁻¹ diag(1/(max(μ̃_t, 1) + ρ/ρ₁)) F D^{-1/2} ([`ClipMetric`])
    Clip,
}

impl AdmmMetric {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "clip" => Some(Self::Clip),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Clip => "clip",
        }
    }
}

/// ADMM controls
#[derive(Clone, Debug)]
pub struct AdmmCfg {
    /// outer iterations
    pub iters: usize,
    /// ρ at the start (absolute, the data term's units; a resumed run continues with its saved ρ)
    pub rho: f64,
    /// ρ₁, the curvature of one sample in one cell: the natural unit of ρ, of the clip symbol's ρ/ρ₁ and of the dual
    /// residual in the balancing rule
    pub rho1: f64,
    /// inner PCG iterations per set
    pub inner: usize,
    /// residual balancing (factor, ratio): ρ ← ρ·factor when ‖primal‖ > ratio·‖dual‖, ρ ← ρ/factor when ‖dual‖ >
    /// ratio·‖primal‖ (the dual residual in units of ρ₁: (ρ/ρ₁)‖z − z_prev‖), u rescaled by the inverse factor. None:
    /// fixed ρ.
    pub adapt: Option<(f64, f64)>,
    /// ρ stays within [rho1 · lo, rho1 · hi] under balancing
    pub rho_bounds: (f64, f64),
}

impl AdmmCfg {
    /// fixed ρ = `rho_rel` · ρ₁, `inner` iterations
    pub fn new(iters: usize, rho_rel: f64, rho1: f64, inner: usize) -> Self {
        Self { iters, rho: rho_rel * rho1, rho1, inner, adapt: None, rho_bounds: (1e-3, 1e3) }
    }
}

/// Numbers of one data step (all sets)
#[derive(Clone, Debug, Default)]
pub struct XStats {
    /// Σ_t ½‖E_t x_t − y_t‖² at the warm start and after the step (the latter by the exact PCG recurrence)
    pub f0: f64,
    pub f: f64,
    /// (Σ_t ‖r₀,t‖²)^½ and (Σ_t ‖r_K,t‖²)^½: residuals of the inner systems (N_t + ρI) x = b_t
    pub r0: f64,
    pub rk: f64,
    /// median and maximum over the sets of ‖r_K,t‖ / ‖r₀,t‖
    pub median: f64,
    pub worst: f64,
    /// inner iterations run (summed over the sets), and sets that stopped early (pᴴHp ≤ 0 or zero residual)
    pub iters: usize,
    pub early: usize,
    /// seconds: whole step, and the metric's symbol updates inside it
    pub secs: f64,
    pub metric_secs: f64,
    pub metric_updates: usize,
}

impl XStats {
    /// fold per-set results ([`InnerOut`], the set's f before) into the totals; call [`finish`](Self::finish) after
    /// the last set
    pub fn add(&mut self, f0: f64, o: &InnerOut, ratios: &mut Vec<f64>) {
        self.f0 += f0;
        self.f += o.f;
        self.r0 += o.r0 * o.r0;
        self.rk += o.rk * o.rk;
        self.iters += o.iters;
        self.early += o.early as usize;
        ratios.push(if o.r0 > 0.0 { o.rk / o.r0 } else { 0.0 });
    }
    pub fn finish(&mut self, mut ratios: Vec<f64>) {
        self.r0 = self.r0.sqrt();
        self.rk = self.rk.sqrt();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.median = ratios.get(ratios.len() / 2).copied().unwrap_or(0.0);
        self.worst = ratios.last().copied().unwrap_or(0.0);
    }
}

/// The data step of ADMM: x_t ← K PCG iterations on (N_t + ρI) x = E_tᴴ y_t + ρ (z_t − u_t), warm-started at x_t,
/// for every set t (independently: own α, β per set).
pub trait AdmmData {
    /// chunk lengths (complex elements)
    fn lens(&self) -> Vec<usize>;
    fn x_step(&mut self, x: &mut HostStack, z: &HostStack, u: &HostStack, rho: f64, inner: usize) -> Result<XStats, Error>;
}

/// Vector operations of one set's inner PCG ([`pcg_inner`]): host vectors (tests, references) or device slices.
pub trait SetPcgOps {
    type V;
    /// Re⟨a, b⟩
    fn dot(&mut self, a: &Self::V, b: &Self::V) -> Result<f64, Error>;
    /// y ← a x + b y (y is initialised)
    fn axpby(&mut self, a: f64, x: &Self::V, b: f64, y: &mut Self::V) -> Result<(), Error>;
    /// out ← N p = Eᴴ E p; returns ‖E p‖²
    fn normal(&mut self, p: &Self::V, out: &mut Self::V) -> Result<f64, Error>;
    /// z ← P r (the inner metric; identity: a copy)
    fn precond(&mut self, r: &Self::V, z: &mut Self::V) -> Result<(), Error>;
}

/// result of [`pcg_inner`] for one set
#[derive(Clone, Copy, Debug, Default)]
pub struct InnerOut {
    /// ‖r₀‖, ‖r_K‖
    pub r0: f64,
    pub rk: f64,
    /// f(x) after the iterations (exact recurrence)
    pub f: f64,
    pub iters: usize,
    /// stopped before K iterations (zero residual or pᴴ(N + ρI)p ≤ 0)
    pub early: bool,
}

/// `iters` PCG iterations on (N + ρI) x = b from the warm start x, given r = b − (N + ρI) x, the data gradient
/// g = ∇f(x) = N x − Eᴴy and f = f(x) = ½‖E x − y‖². Updates x, r, g in place and tracks f exactly:
/// f(x + αp) = f + α Re⟨p, g⟩ + ½α²‖E p‖², g ← g + α N p (one normal operator application per iteration, no data).
/// `w` = three work vectors (p, q, z) of the same length.
pub fn pcg_inner<O: SetPcgOps>(o: &mut O, rho: f64, iters: usize, x: &mut O::V, r: &mut O::V, g: &mut O::V, f: f64, w: &mut [O::V; 3]) -> Result<InnerOut, Error> {
    let [p, q, z] = w;
    let mut f = f;
    let r0 = o.dot(r, r)?.max(0.0).sqrt();
    let mut out = InnerOut { r0, rk: r0, f, iters: 0, early: false };
    if !(r0 > 0.0) || iters == 0 {
        out.early = iters > 0;
        return Ok(out);
    }
    o.precond(r, z)?;
    o.axpby(1.0, z, 0.0, p)?;
    let mut rz = o.dot(r, z)?;
    for k in 0..iters {
        if !(rz > 0.0) {
            out.early = true;
            break;
        }
        // q = N p (+ ρ p below); ‖E p‖² for the data-term recurrence
        let ep2 = o.normal(p, q)?;
        let pg = o.dot(p, g)?;
        // g += α N p needs N p before ρ p is added: g += α q now, scaled after α is known → keep N p in q, form H p
        // implicitly: pᴴHp = pᴴNp + ρ‖p‖²
        let pnp = o.dot(p, q)?;
        let pp = o.dot(p, p)?;
        let php = pnp + rho * pp;
        if !(php > 0.0) {
            out.early = true;
            break;
        }
        let a = rz / php;
        f += a * pg + 0.5 * a * a * ep2;
        // g ← g + α N p, x ← x + α p, r ← r − α (N p + ρ p)
        o.axpby(a, q, 1.0, g)?;
        o.axpby(a, p, 1.0, x)?;
        o.axpby(-a, q, 1.0, r)?;
        o.axpby(-a * rho, p, 1.0, r)?;
        out.iters = k + 1;
        if k + 1 == iters {
            break;
        }
        o.precond(r, z)?;
        let rz_new = o.dot(r, z)?;
        let b = rz_new / rz;
        rz = rz_new;
        o.axpby(1.0, z, b, p)?;
    }
    out.rk = o.dot(r, r)?.max(0.0).sqrt();
    out.f = f;
    Ok(out)
}

/// Clip symbol of set t: λ_eff(k) = max(share · μ(k), 1) + ρ/ρ₁ (FFT order), so the metric is
/// P = D^{-1/2} F⁻¹ diag(1/λ_eff) F D^{-1/2}, its whitened spectrum 1/(max(μ̃, 1) + ρ/ρ₁) ∈ (0, 1/(1 + ρ/ρ₁)].
/// `mu`: samples per Nyquist cell of the pooled sets, `share` = the set's fraction of those samples.
pub fn clip_symbol(mu: &[f32], share: f64, rho_rel: f64) -> Vec<f32> {
    let (s, r) = (share as f32, rho_rel as f32);
    mu.par_iter().map(|&m| (s * m).max(1.0) + r).collect()
}

/// samples per Nyquist cell from a coil-convolved Chan spectrum: μ = m λ_conv / (D̄ Σ_k λ_chan)
/// (`crate::precond::chan_spectrum`: Σ_k λ_chan = nv · m · h₁; `coil_convolved`: λ_conv = λ_chan ⊛ P, Σ P = D̄)
pub fn samples_per_cell(lam_conv: &[f32], sum_chan: f64, m: usize, dbar: f64) -> Vec<f32> {
    let s = (m as f64 / (dbar * sum_chan).max(1e-300)) as f32;
    lam_conv.par_iter().map(|&l| (l * s).max(0.0)).collect()
}

/// The `clip` inner metric on the device: one circulant (FFT plans, D^{-1/2}) whose symbol is replaced per set
/// (`CirculantPrecond::set_symbol`) from the per-encode samples-per-cell symbols and the set's sample share.
pub struct ClipMetric {
    /// samples per cell μ_e(k) of all sets of symbol group e pooled (coil-smoothed Chan symbol, FFT order)
    pub mu: Vec<Vec<f32>>,
    /// set → (group e, share m_t / m_e)
    pub of_set: Vec<(usize, f64)>,
    pub rho1: f64,
    pre: CirculantPrecond,
    cur: Option<(usize, u64, u64)>,
    pub updates: usize,
    pub secs: f64,
}

impl ClipMetric {
    /// `dinv` = D^{-1/2} per voxel (D = Σ_c |S_c|² floored)
    pub fn new(dev: &Device, st: &Arc<CudaStream>, kern: Arc<Kernels>, dims: [usize; 3], mu: Vec<Vec<f32>>, of_set: Vec<(usize, f64)>, rho1: f64, dinv: &[f32]) -> Result<Self, Error> {
        assert!(!mu.is_empty() && of_set.iter().all(|&(e, s)| e < mu.len() && s > 0.0));
        let lam0 = clip_symbol(&mu[0], of_set[0].1, 1.0);
        let pre = CirculantPrecond::from_factors(dev, st, kern, dims, &lam0, dinv)?;
        Ok(Self { mu, of_set, rho1, pre, cur: None, updates: 0, secs: 0.0 })
    }
    /// the metric of set t at ρ (symbol uploaded unless it is the current one)
    pub fn prepare(&mut self, st: &Arc<CudaStream>, t: usize, rho: f64) -> Result<&mut CirculantPrecond, Error> {
        let (e, s) = self.of_set[t];
        let key = (e, s.to_bits(), rho.to_bits());
        if self.cur != Some(key) {
            let t0 = Instant::now();
            let lam = clip_symbol(&self.mu[e], s, rho / self.rho1);
            self.pre.set_symbol(st, &lam)?;
            self.cur = Some(key);
            self.updates += 1;
            self.secs += t0.elapsed().as_secs_f64();
        }
        Ok(&mut self.pre)
    }
    /// set order that keeps the symbol: by group, then share
    pub fn order(&self) -> Vec<usize> {
        let mut o: Vec<usize> = (0..self.of_set.len()).collect();
        o.sort_by(|&a, &b| {
            let (ea, sa) = self.of_set[a];
            let (eb, sb) = self.of_set[b];
            ea.cmp(&eb).then(sa.partial_cmp(&sb).unwrap()).then(a.cmp(&b))
        });
        o
    }
    /// fraction of the cells with share · μ_e ≥ 1 at the mean share of group e's sets (all cells, and the cells with
    /// μ_e > 0)
    pub fn dense_fraction(&self, e: usize) -> (f64, f64) {
        let sh: Vec<f64> = self.of_set.iter().filter(|p| p.0 == e).map(|p| p.1).collect();
        let s = sh.iter().sum::<f64>() / sh.len().max(1) as f64;
        let mu = &self.mu[e];
        let dense = mu.iter().filter(|&&m| s * m as f64 >= 1.0).count() as f64;
        let occ = mu.iter().filter(|&&m| m > 1e-3).count().max(1) as f64;
        (dense / mu.len().max(1) as f64, dense / occ)
    }
}

/// [`AdmmData`] of a [`crate::p2np::P2np`]: the per-set device data step ([`crate::p2np::P2np::admm_x_step`])
pub struct AdmmP2np<'r, 'a> {
    pub p: &'r mut crate::p2np::P2np<'a>,
    pub st: Arc<CudaStream>,
    pub metric: Option<ClipMetric>,
}

impl AdmmData for AdmmP2np<'_, '_> {
    fn lens(&self) -> Vec<usize> {
        self.p.lens()
    }
    fn x_step(&mut self, x: &mut HostStack, z: &HostStack, u: &HostStack, rho: f64, inner: usize) -> Result<XStats, Error> {
        self.p.admm_x_step(&self.st, x, z, u, rho, inner, self.metric.as_mut())
    }
}

/// Reference data step through an objective's gradient only (N p = ∇f(p) − ∇f(0)), per-set PCG scalars over host
/// stacks: all sets advance together, one gradient pass per inner iteration. Holds ∇f(0) and five work stacks (tests
/// and small problems; the production path is [`AdmmP2np`], device-resident per set). `pre(t, r, z)`: z = P_t r.
pub struct GradXStep<'r> {
    pub obj: &'r mut dyn Objective,
    pub st: Arc<CudaStream>,
    pub pre: Option<Box<dyn FnMut(usize, &[C32], &mut [C32]) -> Result<(), Error> + 'r>>,
    g0: Option<HostStack>,
}

impl<'r> GradXStep<'r> {
    pub fn new(obj: &'r mut dyn Objective, st: Arc<CudaStream>) -> Self {
        Self { obj, st, pre: None, g0: None }
    }
}

fn rdot_chunk(a: &[C32], b: &[C32]) -> f64 {
    a.par_chunks(PAR).zip(b.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
}

fn axpby_chunk(a: f64, x: &[C32], b: f64, y: &mut [C32]) {
    let (a, b) = (a as f32, b as f32);
    y.par_chunks_mut(PAR).zip(x.par_chunks(PAR)).for_each(|(yc, xc)| {
        for (yi, xi) in yc.iter_mut().zip(xc) {
            *yi = C32::new(a * xi.re + b * yi.re, a * xi.im + b * yi.im);
        }
    });
}

impl<'r> AdmmData for GradXStep<'r> {
    fn lens(&self) -> Vec<usize> {
        self.obj.lens()
    }
    fn x_step(&mut self, x: &mut HostStack, z: &HostStack, u: &HostStack, rho: f64, inner: usize) -> Result<XStats, Error> {
        let t0 = Instant::now();
        let lens = x.lens.clone();
        let nt = lens.len();
        if self.g0.is_none() {
            let mut g0 = HostStack::zeros(&lens);
            self.obj.gradient_host(&self.st, &HostStack::zeros(&lens), &mut g0)?;
            self.g0 = Some(g0);
        }
        let g0 = self.g0.take().unwrap();
        let mut g = HostStack::zeros(&lens);
        let f0 = self.obj.gradient_host(&self.st, x, &mut g)?;
        // r = ρ (z − u − x) − g
        let mut r = HostStack::zeros(&lens);
        r.data.par_iter_mut().zip(z.data.par_iter().zip(u.data.par_iter())).zip(x.data.par_iter().zip(g.data.par_iter())).for_each(|((ri, (zi, ui)), (xi, gi))| {
            let rh = rho as f32;
            *ri = C32::new(rh * (zi.re - ui.re - xi.re) - gi.re, rh * (zi.im - ui.im - xi.im) - gi.im);
        });
        let (mut p, mut q, mut zz, mut gp) = (HostStack::zeros(&lens), HostStack::zeros(&lens), HostStack::zeros(&lens), HostStack::zeros(&lens));
        let precond = |pre: &mut Option<Box<dyn FnMut(usize, &[C32], &mut [C32]) -> Result<(), Error> + 'r>>, t: usize, r: &HostStack, z: &mut HostStack| -> Result<(), Error> {
            let (o, l) = (z.off[t], z.lens[t]);
            match pre.as_mut() {
                Some(f) => f(t, r.chunk(t), &mut z.data[o..o + l]),
                None => {
                    z.chunk_mut(t).copy_from_slice(r.chunk(t));
                    Ok(())
                }
            }
        };
        let mut rz = vec![0.0f64; nt];
        let mut r0 = vec![0.0f64; nt];
        let mut active = vec![true; nt];
        let mut its = vec![0usize; nt];
        let mut early = vec![false; nt];
        for t in 0..nt {
            precond(&mut self.pre, t, &r, &mut zz)?;
            p.chunk_mut(t).copy_from_slice(zz.chunk(t));
            rz[t] = rdot_chunk(r.chunk(t), zz.chunk(t));
            r0[t] = rdot_chunk(r.chunk(t), r.chunk(t)).sqrt();
            if !(r0[t] > 0.0) {
                active[t] = false;
                early[t] = inner > 0;
            }
        }
        let mut f = f0;
        for k in 0..inner {
            if !active.iter().any(|&a| a) {
                break;
            }
            // N p = ∇f(p) − ∇f(0)
            self.obj.gradient_host(&self.st, &p, &mut gp)?;
            HostStack::axpby(-1.0, &g0, 1.0, &mut gp);
            for t in 0..nt {
                if !active[t] {
                    continue;
                }
                let (pc, nq) = (p.chunk(t), gp.chunk(t));
                let pnp = rdot_chunk(pc, nq);
                let pp = rdot_chunk(pc, pc);
                let pg = rdot_chunk(pc, g.chunk(t));
                let php = pnp + rho * pp;
                if !(rz[t] > 0.0) || !(php > 0.0) {
                    active[t] = false;
                    early[t] = true;
                    continue;
                }
                let a = rz[t] / php;
                // ‖E p‖² = pᴴ N p
                f += a * pg + 0.5 * a * a * pnp;
                q.chunk_mut(t).copy_from_slice(nq);
                axpby_chunk(a, q.chunk(t), 1.0, g.chunk_mut(t));
                axpby_chunk(a, p.chunk(t), 1.0, x.chunk_mut(t));
                axpby_chunk(-a, q.chunk(t), 1.0, r.chunk_mut(t));
                axpby_chunk(-a * rho, p.chunk(t), 1.0, r.chunk_mut(t));
                its[t] = k + 1;
                if k + 1 < inner {
                    precond(&mut self.pre, t, &r, &mut zz)?;
                    let rzn = rdot_chunk(r.chunk(t), zz.chunk(t));
                    let b = rzn / rz[t];
                    rz[t] = rzn;
                    let (o, l) = (p.off[t], p.lens[t]);
                    axpby_chunk(1.0, zz.chunk(t), b, &mut p.data[o..o + l]);
                }
            }
        }
        self.g0 = Some(g0);
        let mut s = XStats { f0, f, ..Default::default() };
        let mut ratios = Vec::with_capacity(nt);
        for t in 0..nt {
            let rk = rdot_chunk(r.chunk(t), r.chunk(t)).sqrt();
            s.r0 += r0[t] * r0[t];
            s.rk += rk * rk;
            s.iters += its[t];
            s.early += early[t] as usize;
            ratios.push(if r0[t] > 0.0 { rk / r0[t] } else { 0.0 });
        }
        s.finish(ratios);
        s.secs = t0.elapsed().as_secs_f64();
        Ok(s)
    }
}

/// Numbers of one outer iteration (for the caller's report)
#[derive(Clone, Debug, Default)]
pub struct AdmmIter {
    pub k: usize,
    /// ρ of this iteration's data step, and ρ for the next one (balancing)
    pub rho: f64,
    pub rho_next: f64,
    /// ‖x − z‖ (primal), ρ‖z − z_prev‖ (dual), (ρ/ρ₁)‖z − z_prev‖ (dual in units of ρ₁: the balancing rule's)
    pub primal: f64,
    pub dual: f64,
    pub dual_nat: f64,
    /// ‖x − z‖ / max(‖x‖, ‖z‖), ‖z − z_prev‖ / ‖u‖
    pub primal_rel: f64,
    pub dual_rel: f64,
    pub x: XStats,
    /// seconds: data step, denoiser, the rest (x + u, residuals, u update; the report is timed by the caller)
    pub t_inner: f64,
    pub t_prior: f64,
    pub t_other: f64,
}

/// The ADMM loop on host stacks x, z, u (one work stack more inside). Start: the caller's x, z, u (x = z = the warm
/// start, u = 0); a checkpoint `state_<name>.hsk` (x, z, u, ρ, written on stop, and every `ctl.every` iterations when
/// set) is resumed instead. `report(k, z, x, info)` after every iteration (k = usize::MAX once before the first, info
/// None). Returns the outcome and the final ρ.
#[allow(clippy::too_many_arguments)]
pub fn run_admm(
    data: &mut dyn AdmmData,
    x: &mut HostStack,
    z: &mut HostStack,
    u: &mut HostStack,
    cfg: &AdmmCfg,
    denoise: &mut dyn FnMut(usize, &mut Vec<C32>),
    report: &mut dyn FnMut(usize, &HostStack, &HostStack, Option<&AdmmIter>),
    ctl: Option<(&RunCtl, &str)>,
) -> Result<(Outcome, f64), Error> {
    let lens = x.lens.clone();
    assert!(z.lens == lens && u.lens == lens, "run_admm: x, z, u must have the same layout");
    assert!(cfg.rho > 0.0 && cfg.rho1 > 0.0, "run_admm: ρ and ρ₁ must be positive");
    let mut rho = cfg.rho;
    let mut k0 = 0;
    if let Some((c, name)) = ctl {
        if let Some(m) = load_raw(c, name, &mut [&mut *x, &mut *z, &mut *u])? {
            k0 = m["k"].as_u64().ok_or_else(|| Error::Format("ADMM checkpoint: k".into()))? as usize;
            rho = m["rho"].as_f64().ok_or_else(|| Error::Format("ADMM checkpoint: rho".into()))?;
            eprintln!("  resuming ADMM {name} at outer iteration {k0} / {} (ρ = {:.4} ρ₁)", cfg.iters, rho / cfg.rho1);
        }
    }
    if k0 == 0 {
        report(usize::MAX, z, x, None);
    }
    let save = |c: &RunCtl, name: &str, next: usize, rho: f64, x: &HostStack, z: &HostStack, u: &HostStack| -> Result<(), Error> {
        let meta = serde_json::json!({ "k": next, "rho": rho, "rho1": cfg.rho1, "admm": true });
        save_raw(c, name, &meta, &[x, z, u], None)
    };
    let mut w = HostStack::zeros(&lens);
    for k in k0..cfg.iters {
        let t_a = Instant::now();
        let xs = match data.x_step(x, z, u, rho, cfg.inner) {
            Ok(s) => s,
            Err(e) => {
                // z, u are those of the start of iteration k; x is partly updated, still a valid warm start
                if let (Some((c, name)), true) = (ctl, k > k0) {
                    eprintln!("  ADMM data step failed at iteration {k} ({e}): saving a checkpoint to resume from iteration {k}");
                    if let Err(e2) = save(c, name, k, rho, x, z, u) {
                        eprintln!("  checkpoint failed too: {e2}");
                    }
                }
                return Err(e);
            }
        };
        let t_inner = t_a.elapsed().as_secs_f64();
        let t_b = Instant::now();
        // w = x + u
        w.data.par_chunks_mut(PAR).zip(x.data.par_chunks(PAR).zip(u.data.par_chunks(PAR))).for_each(|(wc, (xc, uc))| {
            for ((wi, xi), ui) in wc.iter_mut().zip(xc).zip(uc) {
                *wi = C32::new(xi.re + ui.re, xi.im + ui.im);
            }
        });
        let mut t_other = t_b.elapsed().as_secs_f64();
        let t_c = Instant::now();
        // z = prox_{λR/ρ}(x + u): the prox step of the fixed objective f + λR is 1/ρ (crate::p2np::prox_step)
        crate::p2np::set_prox_step(1.0 / rho);
        denoise(k, &mut w.data);
        let t_prior = t_c.elapsed().as_secs_f64();
        let t_d = Instant::now();
        // ‖z_new − z_prev‖², then z ← z_new (the work stack keeps z_prev, overwritten next iteration)
        let dz2: f64 = w.data.par_chunks(PAR).zip(z.data.par_chunks(PAR)).map(|(a, b)| a.iter().zip(b).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum::<f64>()).sum();
        std::mem::swap(z, &mut w);
        // u ← u + x − z, with ‖x − z‖², ‖x‖², ‖z‖², ‖u‖² (new u)
        let sums = u
            .data
            .par_chunks_mut(PAR)
            .zip(x.data.par_chunks(PAR).zip(z.data.par_chunks(PAR)))
            .map(|(uc, (xc, zc))| {
                let mut s = [0.0f64; 4];
                for ((ui, xi), zi) in uc.iter_mut().zip(xc).zip(zc) {
                    let d = C32::new(xi.re - zi.re, xi.im - zi.im);
                    *ui = C32::new(ui.re + d.re, ui.im + d.im);
                    s[0] += (d.re as f64).powi(2) + (d.im as f64).powi(2);
                    s[1] += (xi.re as f64).powi(2) + (xi.im as f64).powi(2);
                    s[2] += (zi.re as f64).powi(2) + (zi.im as f64).powi(2);
                    s[3] += (ui.re as f64).powi(2) + (ui.im as f64).powi(2);
                }
                s
            })
            .reduce(|| [0.0; 4], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]);
        let primal = sums[0].sqrt();
        let dz = dz2.sqrt();
        let mut info = AdmmIter {
            k,
            rho,
            rho_next: rho,
            primal,
            dual: rho * dz,
            dual_nat: rho / cfg.rho1 * dz,
            primal_rel: primal / sums[1].sqrt().max(sums[2].sqrt()).max(1e-300),
            dual_rel: dz / sums[3].sqrt().max(1e-300),
            x: xs,
            t_inner,
            t_prior,
            t_other: 0.0,
        };
        // residual balancing (Boyd et al. §3.4.1), dual in units of ρ₁; u is the scaled dual y/ρ: rescaled with ρ
        if let Some((fac, ratio)) = cfg.adapt {
            let (lo, hi) = (cfg.rho_bounds.0 * cfg.rho1, cfg.rho_bounds.1 * cfg.rho1);
            let new = if info.primal > ratio * info.dual_nat {
                (rho * fac).min(hi)
            } else if info.dual_nat > ratio * info.primal {
                (rho / fac).max(lo)
            } else {
                rho
            };
            if new != rho {
                let s = (rho / new) as f32;
                u.data.par_iter_mut().for_each(|v| *v = C32::new(v.re * s, v.im * s));
                rho = new;
            }
            info.rho_next = rho;
        }
        t_other += t_d.elapsed().as_secs_f64();
        info.t_other = t_other;
        report(k, z, x, Some(&info));
        if let Some((c, name)) = ctl {
            let next = k + 1;
            if c.stop_requested() && next < cfg.iters {
                save(c, name, next, rho, x, z, u)?;
                return Ok((Outcome::Stopped(next), rho));
            }
            if c.due(next) && next < cfg.iters {
                save(c, name, next, rho, x, z, u)?;
            }
        }
    }
    if let Some((c, name)) = ctl {
        c.clear(name);
        clear_raw(c, name);
    }
    Ok((Outcome::Finished, rho))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ppre::{Fft3, SymPre};
    use num_complex::Complex64;

    /// deterministic xorshift in [0, 1)
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (a, b) = (self.next().max(1e-300), self.next());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
        fn poisson(&mut self, mu: f64) -> usize {
            let (l, mut k, mut p) = ((-mu).exp(), 0usize, 1.0);
            loop {
                p *= self.next();
                if p <= l {
                    return k;
                }
                k += 1;
            }
        }
    }

    fn n2(z: C32) -> f32 {
        z.re * z.re + z.im * z.im
    }

    const DIMS: [usize; 3] = [8, 8, 6];
    const NC: usize = 3;

    /// Per-cell Fourier sets: E_t x = (√n_t(k) ⊙ F(S_c x))_c on the grid (unnormalised FFT), so
    /// N_t = Σ_c S_cᴴ F⁻¹ diag(nv · n_t) F S_c — the per-cell model of research/precond_sampling_theory.md §3 exactly,
    /// with c = nv (one sample in one cell, coil energy D). Sampling: Poisson counts of a radial density, per set.
    struct CellSets {
        fft: Arc<Fft3>,
        maps: Vec<Vec<C32>>,
        n: Vec<Vec<f32>>,
        /// data per set: y_c(k) of the sampled cells (zero elsewhere), coil-major
        y: Vec<Vec<C32>>,
        /// expected count per cell (the metric's μ̃) and D^{-1/2}
        mu: Vec<f32>,
        dinv: Vec<f32>,
        metric: AdmmMetric,
    }

    fn kfrac(q: usize) -> f64 {
        let i = [q % DIMS[0], (q / DIMS[0]) % DIMS[1], q / (DIMS[0] * DIMS[1])];
        let f: Vec<f64> = (0..3).map(|a| { let v = i[a] as f64; let n = DIMS[a] as f64; (if v <= n / 2.0 { v } else { v - n }) / (n / 2.0) }).collect();
        (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt()
    }

    impl CellSets {
        fn new(nt: usize, seed: u64, metric: AdmmMetric) -> (Self, Vec<Vec<C32>>) {
            let fft = Arc::new(Fft3::new(DIMS));
            let nv = fft.nv();
            let mut r = Rng(seed | 1);
            let maps: Vec<Vec<C32>> = (0..NC)
                .map(|c| {
                    (0..nv)
                        .map(|q| {
                            let x = (q % DIMS[0]) as f64 / DIMS[0] as f64 - 0.5;
                            let y = ((q / DIMS[0]) % DIMS[1]) as f64 / DIMS[1] as f64 - 0.5;
                            let th = std::f64::consts::TAU * c as f64 / NC as f64;
                            let a = (-((x - 0.4 * th.cos()).powi(2) + (y - 0.4 * th.sin()).powi(2)) / 0.4).exp();
                            let z = Complex64::from_polar(a, 0.7 * (x * th.sin() - y * th.cos()));
                            C32::new(z.re as f32, z.im as f32)
                        })
                        .collect()
                })
                .collect();
            let mu: Vec<f32> = (0..nv).map(|q| (40.0 * (-kfrac(q) * 6.0).exp() + 0.25) as f32).collect();
            let d: Vec<f32> = (0..nv).map(|q| (0..NC).map(|c| n2(maps[c][q])).sum::<f32>()).collect();
            let med = { let mut v = d.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
            let dinv: Vec<f32> = d.iter().map(|&v| 1.0 / v.max(0.1 * med).sqrt()).collect();
            // truth: a smooth object + a sparse part, different per set
            let truth: Vec<Vec<C32>> = (0..nt)
                .map(|t| {
                    (0..nv)
                        .map(|q| {
                            let x = (q % DIMS[0]) as f64 / DIMS[0] as f64 - 0.5;
                            let y = ((q / DIMS[0]) % DIMS[1]) as f64 / DIMS[1] as f64 - 0.5;
                            let s = if (q * 7 + t * 3) % 11 == 0 { 2.0 } else { 0.0 };
                            let a = if x * x + y * y < 0.15 { 1.0 + 0.3 * (6.0 * x + t as f64).sin() } else { 0.0 } + s;
                            let z = Complex64::from_polar(a, 0.5 * y + 0.2 * t as f64);
                            C32::new(z.re as f32, z.im as f32)
                        })
                        .collect()
                })
                .collect();
            let n: Vec<Vec<f32>> = (0..nt).map(|_| mu.iter().map(|&m| r.poisson(m as f64) as f32).collect()).collect();
            let mut s = Self { fft, maps, n, y: vec![], mu, dinv, metric };
            let y: Vec<Vec<C32>> = (0..nt)
                .map(|t| {
                    let mut yt = s.forward(t, &truth[t]);
                    for (j, v) in yt.iter_mut().enumerate() {
                        if s.n[t][j % nv] > 0.0 {
                            // noise of n samples averaged into one per-cell value of weight √n: variance per value 1
                            v.re += (3.0 * r.normal()) as f32;
                            v.im += (3.0 * r.normal()) as f32;
                        }
                    }
                    yt
                })
                .collect();
            s.y = y;
            (s, truth)
        }
        /// E_t x (coil-major, nv per coil): √n ⊙ F(S_c x)
        fn forward(&self, t: usize, x: &[C32]) -> Vec<C32> {
            let nv = self.fft.nv();
            let mut out = vec![C32::new(0.0, 0.0); NC * nv];
            for c in 0..NC {
                let o = &mut out[c * nv..(c + 1) * nv];
                for q in 0..nv {
                    let (a, b) = (self.maps[c][q], x[q]);
                    o[q] = C32::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
                }
                self.fft.forward(o);
                for q in 0..nv {
                    let s = self.n[t][q].sqrt();
                    o[q] = C32::new(o[q].re * s, o[q].im * s);
                }
            }
            out
        }
        /// E_tᴴ y: Σ_c conj(S_c) Fᴴ(√n ⊙ y_c), Fᴴ = nv F⁻¹
        fn adjoint(&self, t: usize, y: &[C32]) -> Vec<C32> {
            let nv = self.fft.nv();
            let mut out = vec![C32::new(0.0, 0.0); nv];
            let mut b = vec![C32::new(0.0, 0.0); nv];
            for c in 0..NC {
                for q in 0..nv {
                    let s = self.n[t][q].sqrt() * nv as f32;
                    b[q] = C32::new(y[c * nv + q].re * s, y[c * nv + q].im * s);
                }
                self.fft.inverse(&mut b);
                for q in 0..nv {
                    let (a, v) = (self.maps[c][q], b[q]);
                    out[q].re += a.re * v.re + a.im * v.im;
                    out[q].im += a.re * v.im - a.im * v.re;
                }
            }
            out
        }
        /// ∇f_t(x) = E_tᴴ(E_t x − y_t) and f_t(x)
        fn grad(&self, t: usize, x: &[C32]) -> (Vec<C32>, f64) {
            let mut r = self.forward(t, x);
            for (a, b) in r.iter_mut().zip(&self.y[t]) {
                *a = C32::new(a.re - b.re, a.im - b.im);
            }
            let f = 0.5 * r.iter().map(|v| n2(*v) as f64).sum::<f64>();
            (self.adjoint(t, &r), f)
        }
    }

    struct HostOps<'a> {
        s: &'a CellSets,
        t: usize,
        pre: Option<SymPre>,
    }

    impl SetPcgOps for HostOps<'_> {
        type V = Vec<C32>;
        fn dot(&mut self, a: &Vec<C32>, b: &Vec<C32>) -> Result<f64, Error> {
            Ok(rdot_chunk(a, b))
        }
        fn axpby(&mut self, a: f64, x: &Vec<C32>, b: f64, y: &mut Vec<C32>) -> Result<(), Error> {
            axpby_chunk(a, x, b, y);
            Ok(())
        }
        fn normal(&mut self, p: &Vec<C32>, out: &mut Vec<C32>) -> Result<f64, Error> {
            let e = self.s.forward(self.t, p);
            *out = self.s.adjoint(self.t, &e);
            Ok(e.iter().map(|v| n2(*v) as f64).sum())
        }
        fn precond(&mut self, r: &Vec<C32>, z: &mut Vec<C32>) -> Result<(), Error> {
            match &self.pre {
                Some(p) => p.apply(r, z),
                None => z.copy_from_slice(r),
            }
            Ok(())
        }
    }

    /// ρ₁ of the cell model: c · D_body with c = nv
    fn rho1(s: &CellSets) -> f64 {
        let nv = s.fft.nv();
        let d: Vec<f64> = (0..nv).map(|q| (0..NC).map(|c| n2(s.maps[c][q]) as f64).sum()).collect();
        let mut v = d.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        nv as f64 * v[v.len() / 2]
    }

    impl AdmmData for CellSets {
        fn lens(&self) -> Vec<usize> {
            vec![self.fft.nv(); self.n.len()]
        }
        fn x_step(&mut self, x: &mut HostStack, z: &HostStack, u: &HostStack, rho: f64, inner: usize) -> Result<XStats, Error> {
            let r1 = rho1(self);
            let mut s = XStats::default();
            let mut ratios = vec![];
            for t in 0..self.n.len() {
                let pre = (self.metric == AdmmMetric::Clip).then(|| SymPre::new(self.fft.clone(), clip_symbol(&self.mu, 1.0, rho / r1), Some(self.dinv.clone())));
                let mut ops = HostOps { s: self, t, pre };
                let mut xt = x.chunk(t).to_vec();
                let (mut g, f0) = self.grad(t, &xt);
                let mut r: Vec<C32> = (0..xt.len()).map(|q| {
                    let (zi, ui, xi, gi) = (z.chunk(t)[q], u.chunk(t)[q], xt[q], g[q]);
                    let rh = rho as f32;
                    C32::new(rh * (zi.re - ui.re - xi.re) - gi.re, rh * (zi.im - ui.im - xi.im) - gi.im)
                }).collect();
                let mut w = [vec![C32::new(0.0, 0.0); xt.len()], vec![C32::new(0.0, 0.0); xt.len()], vec![C32::new(0.0, 0.0); xt.len()]];
                let o = pcg_inner(&mut ops, rho, inner, &mut xt, &mut r, &mut g, f0, &mut w)?;
                x.chunk_mut(t).copy_from_slice(&xt);
                s.add(f0, &o, &mut ratios);
            }
            s.finish(ratios);
            Ok(s)
        }
    }

    /// prox of τR, R(x) = ‖x‖₁ + (ε/2)‖x‖² (pixelwise complex elastic net; strongly convex, so the minimiser of f + λR
    /// is unique and the solvers converge linearly): soft(v, τ) / (1 + τε); `te` = τε
    fn enet(v: &mut [C32], tau: f32, te: f32) {
        for z in v.iter_mut() {
            let m = n2(*z).sqrt();
            let s = if m > tau { (1.0 - tau / m) / (1.0 + te) } else { 0.0 };
            *z = C32::new(z.re * s, z.im * s);
        }
    }

    /// prior strength: λ = LAM ρ₁ (image units 1), strong convexity λε = EPS ρ₁
    const LAM: f64 = 0.05;
    const EPS: f64 = 0.3;

    /// ADMM on the cell sets with the prox of (λ/ρ)R; returns z and the per-iteration numbers
    fn admm(s: &mut CellSets, rho_rel: f64, inner: usize, iters: usize) -> (HostStack, Vec<AdmmIter>) {
        let r1 = rho1(s);
        let lens = s.lens();
        let mut x = HostStack::zeros(&lens);
        // start: the adjoint images scaled (any start; the fixed point must not depend on it)
        for t in 0..lens.len() {
            let a = s.adjoint(t, &s.y[t]);
            let sc = 1.0 / (r1 as f32 * 4.0);
            x.chunk_mut(t).iter_mut().zip(&a).for_each(|(v, b)| *v = C32::new(b.re * sc, b.im * sc));
        }
        let mut z = x.clone();
        let mut u = HostStack::zeros(&lens);
        let cfg = AdmmCfg::new(iters, rho_rel, r1, inner);
        // τ = λ / ρ, τε = EPS ρ₁ / ρ
        let (tau, te) = ((LAM * r1 / cfg.rho) as f32, (EPS * r1 / cfg.rho) as f32);
        let mut den = |_: usize, h: &mut Vec<C32>| enet(h, tau, te);
        let mut infos = vec![];
        let mut rep = |_: usize, _: &HostStack, _: &HostStack, i: Option<&AdmmIter>| {
            if let Some(i) = i {
                infos.push(i.clone());
            }
        };
        run_admm(s, &mut x, &mut z, &mut u, &cfg, &mut den, &mut rep, None).unwrap();
        (z, infos)
    }

    /// the minimiser of Σ_t f_t + λR(x_t) by FISTA (step 1/L, L = nv · max n · max D ≥ λ_max(N_t))
    fn fista(s: &CellSets, iters: usize) -> HostStack {
        let r1 = rho1(s);
        let nv = s.fft.nv();
        let dmax = (0..nv).map(|q| (0..NC).map(|c| n2(s.maps[c][q]) as f64).sum::<f64>()).fold(0.0, f64::max);
        let nmax = s.n.iter().flat_map(|v| v.iter()).fold(0f32, |a, &b| a.max(b)) as f64;
        // λ_max(N_t) ≤ nv · max n · max_q Σ_c |S_c|² (Cauchy–Schwarz over coils)
        let l = nv as f64 * nmax * dmax * 1.05;
        let lens = s.lens();
        let mut x = HostStack::zeros(&lens);
        let mut yv = x.clone();
        let mut tk = 1.0f64;
        for _ in 0..iters {
            let mut xn = yv.clone();
            for t in 0..lens.len() {
                let (g, _) = s.grad(t, yv.chunk(t));
                axpby_chunk(-1.0 / l, &g, 1.0, xn.chunk_mut(t));
                enet(xn.chunk_mut(t), (LAM * r1 / l) as f32, (EPS * r1 / l) as f32);
            }
            let tn = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk).sqrt());
            let b = (tk - 1.0) / tn;
            for i in 0..yv.data.len() {
                let (a, p) = (xn.data[i], x.data[i]);
                yv.data[i] = C32::new(a.re + b as f32 * (a.re - p.re), a.im + b as f32 * (a.im - p.im));
            }
            x = xn;
            tk = tn;
        }
        x
    }

    fn rel(a: &HostStack, b: &HostStack) -> f64 {
        let (mut d, mut n) = (0.0, 0.0);
        for (x, y) in a.data.iter().zip(&b.data) {
            d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
            n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
        }
        (d / n.max(1e-300)).sqrt()
    }

    /// (a) metric none vs clip: the same fixed point, the minimiser of the convex problem (FISTA); the clip metric
    /// reduces the inner residual faster
    #[test]
    fn metric_does_not_move_the_fixed_point() {
        let (mut sn, _) = CellSets::new(3, 11, AdmmMetric::None);
        let (mut sc, _) = CellSets::new(3, 11, AdmmMetric::Clip);
        let (zn, inn) = admm(&mut sn, 1.0, 3, 1500);
        let (zc, inc) = admm(&mut sc, 1.0, 3, 1500);
        let zf = fista(&sn, 20000);
        let (e_nc, e_nf, e_cf) = (rel(&zn, &zc), rel(&zn, &zf), rel(&zc, &zf));
        let red = |v: &[AdmmIter]| v[..5].iter().map(|i| i.x.rk / i.x.r0.max(1e-300)).sum::<f64>() / 5.0;
        let (ln, lc) = (inn.last().unwrap(), inc.last().unwrap());
        eprintln!("(a) fixed points: none vs clip {e_nc:.2e}, none vs FISTA {e_nf:.2e}, clip vs FISTA {e_cf:.2e}; at the end primal rel {:.1e} / {:.1e}, dual rel {:.1e} / {:.1e}; inner ‖r_K‖/‖r₀‖ (first 5 outer) none {:.2e} clip {:.2e}", ln.primal_rel, lc.primal_rel, ln.dual_rel, lc.dual_rel, red(&inn), red(&inc));
        assert!(e_nc < 1e-4, "metric none vs clip: {e_nc:.2e}");
        assert!(e_nf < 1e-4 && e_cf < 1e-4, "ADMM vs the minimiser: {e_nf:.2e} {e_cf:.2e}");
        assert!(red(&inc) < red(&inn), "the clip metric should reduce the inner residual faster");
    }

    /// (b) the fixed point does not depend on ρ when the prox threshold is λ/ρ
    #[test]
    fn fixed_point_invariant_to_rho() {
        let (s0, _) = CellSets::new(3, 11, AdmmMetric::Clip);
        let zf = fista(&s0, 20000);
        let zs: Vec<HostStack> = [0.3, 1.0, 4.0]
            .iter()
            .map(|&rr| {
                let (mut s, _) = CellSets::new(3, 11, AdmmMetric::Clip);
                admm(&mut s, rr, 3, 2500).0
            })
            .collect();
        let e: Vec<f64> = zs.iter().map(|z| rel(z, &zs[1])).collect();
        let ef: Vec<f64> = zs.iter().map(|z| rel(z, &zf)).collect();
        eprintln!("(b) fixed points over ρ ∈ {{0.3, 1, 4}} ρ₁ vs ρ = ρ₁: {:.2e}, {:.2e}; vs FISTA {:.2e} {:.2e} {:.2e}", e[0], e[2], ef[0], ef[1], ef[2]);
        assert!(e[0] < 1e-4 && e[2] < 1e-4, "ρ changed the fixed point: {:.2e} {:.2e}", e[0], e[2]);
        assert!(ef.iter().all(|&v| v < 1e-4), "ADMM vs the minimiser: {ef:?}");
    }

    /// residual balancing changes ρ (and rescales u) without moving the fixed point
    #[test]
    fn adaptive_rho_same_fixed_point() {
        let (mut s2, _) = CellSets::new(3, 11, AdmmMetric::Clip);
        let zf = fista(&s2, 20000);
        let r1 = rho1(&s2);
        let lens = s2.lens();
        let mut x = HostStack::zeros(&lens);
        let mut z = x.clone();
        let mut u = HostStack::zeros(&lens);
        let mut cfg = AdmmCfg::new(2000, 30.0, r1, 3);
        cfg.adapt = Some((2.0, 10.0));
        // the prox of the same λR at the current ρ (ρ read back from the loop's report)
        let rho = std::cell::Cell::new(cfg.rho);
        let mut den = |_: usize, h: &mut Vec<C32>| enet(h, (LAM * r1 / rho.get()) as f32, (EPS * r1 / rho.get()) as f32);
        let mut changes = 0;
        let mut rep = |_: usize, _: &HostStack, _: &HostStack, i: Option<&AdmmIter>| {
            if let Some(i) = i {
                if i.rho_next != i.rho {
                    changes += 1;
                }
                rho.set(i.rho_next);
            }
        };
        let (_, rho_end) = run_admm(&mut s2, &mut x, &mut z, &mut u, &cfg, &mut den, &mut rep, None).unwrap();
        let e = rel(&z, &zf);
        eprintln!("adaptive ρ: {changes} changes, ρ 30 → {:.3} ρ₁; vs the minimiser (FISTA) {e:.2e}", rho_end / r1);
        assert!(changes > 0, "residual balancing never changed ρ from 30 ρ₁");
        assert!(e < 1e-4, "adaptive ρ moved the fixed point: {e:.2e}");
    }

    /// (c) the clip symbol: λ_eff = max(sμ, 1) + ρ/ρ₁, the metric SPD with whitened spectrum 1/λ_eff
    #[test]
    fn clip_symbol_is_spd() {
        let fft = Arc::new(Fft3::new(DIMS));
        let nv = fft.nv();
        let mu: Vec<f32> = (0..nv).map(|q| (80.0 * (-kfrac(q) * 5.0).exp()) as f32).collect();
        let (share, rr) = (0.05, 0.7);
        let lam = clip_symbol(&mu, share, rr);
        for q in 0..nv {
            let want = (share as f32 * mu[q]).max(1.0) + rr as f32;
            assert!((lam[q] - want).abs() <= 1e-6 * want, "symbol at {q}: {} vs {want}", lam[q]);
            assert!(lam[q] >= 1.0 + rr as f32 - 1e-6);
        }
        // dense cells (sμ ≥ 1) are inverted, the sparse ones get the scalar 1/(1 + ρ/ρ₁)
        let dense = (0..nv).filter(|&q| share as f32 * mu[q] >= 1.0).count();
        assert!(dense > 0 && dense < nv, "test symbol should have both regions ({dense} of {nv})");
        let mut r = Rng(5);
        let dinv: Vec<f32> = (0..nv).map(|_| (0.5 + r.next()) as f32).collect();
        let p = SymPre::new(fft.clone(), lam.clone(), Some(dinv.clone()));
        let rv = |r: &mut Rng| -> Vec<C32> { (0..nv).map(|_| C32::new(r.normal() as f32, r.normal() as f32)).collect() };
        let cd = |a: &[C32], b: &[C32]| -> Complex64 { a.iter().zip(b).map(|(x, y)| Complex64::new(x.re as f64, -x.im as f64) * Complex64::new(y.re as f64, y.im as f64)).sum() };
        for _ in 0..5 {
            let (a, b) = (rv(&mut r), rv(&mut r));
            let (mut pa, mut pb) = (vec![C32::new(0.0, 0.0); nv], vec![C32::new(0.0, 0.0); nv]);
            p.apply(&a, &mut pa);
            p.apply(&b, &mut pb);
            // Hermitian: ⟨a, P b⟩ = ⟨P a, b⟩
            let (l, rr2) = (cd(&a, &pb), cd(&pa, &b));
            assert!((l - rr2).norm() <= 1e-4 * l.norm().max(1e-30), "not Hermitian: {l} vs {rr2}");
            // positive definite, Rayleigh quotient of the whitened operator inside [1/λ_max, 1/λ_min]
            let q = cd(&a, &pa).re;
            assert!(q > 0.0, "⟨a, P a⟩ = {q}");
            // P⁻¹ P a = a
            let mut back = vec![C32::new(0.0, 0.0); nv];
            p.apply_inv(&pa, &mut back);
            let e: f64 = back.iter().zip(&a).map(|(x, y)| ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2)).sum::<f64>().sqrt() / a.iter().map(|v| n2(*v) as f64).sum::<f64>().sqrt();
            assert!(e < 1e-4, "P⁻¹ P ≠ I: {e:.2e}");
        }
        // whitened spectrum: D^{1/2} P D^{1/2} e_k = e_k / λ_eff(k) for a plane wave e_k
        let k = (0..nv).max_by(|&a, &b| lam[a].partial_cmp(&lam[b]).unwrap()).unwrap();
        let mut ek = vec![C32::new(0.0, 0.0); nv];
        ek[k] = C32::new(1.0, 0.0);
        fft.inverse(&mut ek);
        let w: Vec<C32> = ek.iter().zip(&dinv).map(|(v, &d)| C32::new(v.re / d, v.im / d)).collect();
        let mut pw = vec![C32::new(0.0, 0.0); nv];
        p.apply(&w, &mut pw);
        for (v, &d) in pw.iter_mut().zip(&dinv) {
            *v = C32::new(v.re / d, v.im / d);
        }
        let ratio = cd(&ek, &pw).re / cd(&ek, &ek).re;
        assert!((ratio * lam[k] as f64 - 1.0).abs() < 1e-4, "whitened eigenvalue {ratio} vs 1/λ {}", 1.0 / lam[k]);
    }

    /// samples per cell from a Chan spectrum: one on-grid sample gives λ_chan = nv h₁ δ_k, so μ = 1 at its cell
    #[test]
    fn samples_per_cell_scale() {
        let nv = 64;
        let h1 = 3.5f64;
        let mut lam = vec![0f32; nv];
        lam[5] = (nv as f64 * h1) as f32;
        // coil power spectrum P with Σ P = D̄ = 2: λ_conv = λ ⊛ P (a delta convolved: P shifted, scaled by nv h₁)
        let dbar = 2.0;
        let conv: Vec<f32> = (0..nv).map(|q| if q == 5 { (nv as f64 * h1 * dbar) as f32 } else { 0.0 }).collect();
        let mu = samples_per_cell(&conv, lam.iter().map(|&v| v as f64).sum(), 1, dbar);
        assert!((mu[5] - 1.0).abs() < 1e-6, "μ at the sample's cell {}", mu[5]);
    }

    /// pcg_inner tracks f(x) exactly and solves the inner system (many iterations: to the solution)
    #[test]
    fn inner_pcg_recurrences() {
        let (s, _) = CellSets::new(1, 3, AdmmMetric::None);
        let nv = s.fft.nv();
        let r1 = rho1(&s);
        let rho = 0.5 * r1;
        let v: Vec<C32> = (0..nv).map(|q| C32::new((q % 5) as f32 * 0.1, 0.0)).collect();
        for metric in [false, true] {
            let mut x = vec![C32::new(0.0, 0.0); nv];
            let (mut g, f0) = s.grad(0, &x);
            let mut r: Vec<C32> = (0..nv).map(|q| C32::new(rho as f32 * v[q].re - g[q].re, rho as f32 * v[q].im - g[q].im)).collect();
            let pre = metric.then(|| SymPre::new(s.fft.clone(), clip_symbol(&s.mu, 1.0, rho / r1), Some(s.dinv.clone())));
            let mut ops = HostOps { s: &s, t: 0, pre };
            let mut w = [vec![C32::new(0.0, 0.0); nv], vec![C32::new(0.0, 0.0); nv], vec![C32::new(0.0, 0.0); nv]];
            let o = pcg_inner(&mut ops, rho, 200, &mut x, &mut r, &mut g, f0, &mut w).unwrap();
            let (g2, f2) = s.grad(0, &x);
            let ge: f64 = g.iter().zip(&g2).map(|(a, b)| ((a.re - b.re) as f64).powi(2) + ((a.im - b.im) as f64).powi(2)).sum::<f64>().sqrt() / g2.iter().map(|v| n2(*v) as f64).sum::<f64>().sqrt();
            eprintln!("inner PCG (metric {metric}): {} its, ‖r‖/‖r₀‖ {:.1e}, f tracked {:.6e} vs direct {f2:.6e}, ∇f tracked vs direct {ge:.1e}", o.iters, o.rk / o.r0, o.f);
            assert!(o.rk / o.r0 < 1e-4, "inner PCG did not converge: {:.1e}", o.rk / o.r0);
            assert!((o.f - f2).abs() <= 1e-3 * f2, "tracked f {} vs {f2}", o.f);
            assert!(ge < 1e-3, "tracked gradient {ge:.1e}");
        }
    }
}
