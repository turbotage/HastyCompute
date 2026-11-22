//! Regularisation by denoising (RED): the prior inside the gradient step instead of after it.
//!
//! P²nP iterates x ← D(x − α P ∇f(x)): the step moves x through the data gradient only, the denoiser corrects it
//! afterwards, and the quasi-Newton secant pairs see the curvature of f alone. RED uses the denoiser residual as a
//! gradient term,
//!
//!     g(x) = ∇f(x) + λ (x − D(x)),            x ← x − α P g(x)        (no denoiser after the step),
//!
//! so every iterate is a (quasi-Newton) gradient step on one objective and the ZMSHR1 secants see the prior's curvature.
//! The residual is an exact gradient for the parts of our D that matter (research/red_prior.md §2):
//! - a linear projection Π (static-tissue lock, cardiac low-pass, air zeroing): x − Πx = ∇ ½‖(I − Π)x‖²;
//! - hard truncation to a set S (rank-R LLR): x − P_S(x) = ∇ ½ dist²(x, S) (almost everywhere);
//! - a prox of a convex R at a fixed θ: x − prox_θR(x) = θ ∇(Moreau envelope);
//! - an MMSE denoiser: −σ² ∇log p_σ (Tweedie), as long as its Jacobian is symmetric.
//!
//! The stationary points of f + λR then do not depend on the preconditioner P (P²nP's do: pnp_restricted, prop:metricfp).
//!
//! [`Red`] wraps any [`Objective`]: the solver loops ([`crate::p2np::run_objective`], [`crate::hoststack::run_objective_host`])
//! run unchanged with an identity denoiser. Strength: λ = ρ / (α μ), μ = the mean gain of the metric (1 without one,
//! [`metric_gain`] otherwise): one step moves x by about τρ (D(x) − x), so ρ = 1 is P²nP's full denoising step and ρ is
//! comparable across resolutions and preconditioners. The gradient f reported is the data term's alone; the prior's
//! ½λ‖x − D(x)‖² (the whole penalty for projections) is in [`RedStats`].

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use num_complex::Complex64;
use rayon::prelude::*;

use crate::gpu::Kernels;
use crate::hoststack::HostStack;
use crate::p2np::{Objective, Stack};
use crate::Error;

const PAR: usize = 1 << 16;

/// Last RED gradient's numbers
#[derive(Clone, Copy, Debug, Default)]
pub struct RedStats {
    pub k: usize,
    /// f(x) of the data term
    pub f_data: f64,
    /// ½ λ ‖x − D(x)‖²
    pub prior_quad: f64,
    /// ‖x − D(x)‖ / ‖x‖
    pub r_rel: f64,
    /// ‖λ (x − D(x))‖ / ‖∇f(x)‖
    pub pull: f64,
    /// seconds in the denoiser
    pub secs: f64,
}

/// f + λR with ∇R = x − D(x) around an inner objective (see the module docs)
pub struct Red<'a> {
    pub inner: &'a mut dyn Objective,
    pub denoise: &'a mut dyn FnMut(usize, &mut Vec<C32>),
    pub lambda: f64,
    /// print the stats of iterations k < 3 and every `log_every`-th (0: never)
    pub log_every: usize,
    /// secant groups of the per-set preconditioners: the sets the denoiser couples (a temporal prior couples the
    /// frames of a voxel), see [`Objective::secant_groups`]; None: the inner objective's
    pub groups: Option<Vec<usize>>,
    pub last: RedStats,
    k: usize,
    xh: HostStack,
    d: HostStack,
}

impl<'a> Red<'a> {
    pub fn new(inner: &'a mut dyn Objective, denoise: &'a mut dyn FnMut(usize, &mut Vec<C32>), lambda: f64) -> Self {
        Self { inner, denoise, lambda, log_every: 5, groups: None, last: RedStats::default(), k: 0, xh: HostStack::default(), d: HostStack::default() }
    }

    /// d ← x − D(x) (d holds a copy of x on entry); g += λ d; returns the stats (f_data filled in by the caller)
    fn residual_into(&mut self, x: &HostStack) -> RedStats {
        let t0 = Instant::now();
        (self.denoise)(self.k, &mut self.d.data);
        let secs = t0.elapsed().as_secs_f64();
        self.d.data.par_chunks_mut(PAR).zip(x.data.par_chunks(PAR)).for_each(|(dc, xc)| {
            for (di, xi) in dc.iter_mut().zip(xc) {
                *di = C32::new(xi.re - di.re, xi.im - di.im);
            }
        });
        let rr = HostStack::rdot(&self.d, &self.d);
        let xx = HostStack::rdot(x, x);
        RedStats { k: self.k, f_data: 0.0, prior_quad: 0.5 * self.lambda * rr, r_rel: (rr / xx.max(1e-300)).sqrt(), pull: 0.0, secs }
    }

    fn log(&self) {
        let s = &self.last;
        if self.log_every > 0 && (s.k < 3 || s.k % self.log_every == 0) {
            eprintln!(
                "    RED it {}: λ {:.3e}, ‖x − D(x)‖/‖x‖ {:.3e}, ‖λ(x − D(x))‖/‖∇f‖ {:.3e}, f {:.4e} + ½λ‖x − D(x)‖² {:.4e}, denoiser {:.1}s",
                s.k, self.lambda, s.r_rel, s.pull, s.f_data, s.prior_quad, s.secs
            );
        }
    }
}

/// Re⟨a, b⟩ / ‖·‖² helper on slices (f64 accumulation)
fn rdot(a: &[C32], b: &[C32]) -> f64 {
    a.par_chunks(PAR).zip(b.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
}

/// Mean gain μ = Σ⟨z, M z⟩ / Σ‖z‖² of the objective's metric on a fixed pseudo-random ±1 vector over the sets `sets`
/// (a Hutchinson estimate of tr(M)/n: deterministic, so a resumed run gets the same λ)
pub fn metric_gain(obj: &mut dyn Objective, st: &Arc<CudaStream>, sets: &[usize]) -> Result<f64, Error> {
    let lens = obj.lens();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for &t in sets {
        let l = lens[t];
        let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let z: Vec<C32> = (0..l)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let re = if s & 1 == 0 { 1.0 } else { -1.0 };
                let im = if s & 2 == 0 { 1.0 } else { -1.0 };
                C32::new(re, im)
            })
            .collect();
        let mut mz = vec![C32::new(0.0, 0.0); l];
        obj.metric_host(st, t, &z, &mut mz, false)?;
        num += rdot(&z, &mz);
        den += rdot(&z, &z);
    }
    Ok(num / den.max(1e-300))
}

impl Objective for Red<'_> {
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn lens(&self) -> Vec<usize> {
        self.inner.lens()
    }
    fn begin_iteration(&mut self, k: usize, iters: usize) {
        self.k = k;
        self.inner.begin_iteration(k, iters);
    }
    fn curvature_pair_ok(&self) -> bool {
        self.inner.curvature_pair_ok()
    }
    fn secant_groups(&self) -> Option<Vec<usize>> {
        self.groups.clone().or_else(|| self.inner.secant_groups())
    }
    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        self.inner.metric(st, t, r, z, inverse)
    }
    fn metric_host(&mut self, st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inverse: bool) -> Result<(), Error> {
        self.inner.metric_host(st, t, r, z, inverse)
    }
    fn gradient_host(&mut self, st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        let f = self.inner.gradient_host(st, x, g)?;
        if self.d.lens != x.lens {
            self.d = HostStack::zeros(&x.lens);
        }
        self.d.copy_from(x);
        let mut s = self.residual_into(x);
        let gg = HostStack::rdot(g, g);
        HostStack::axpby(self.lambda, &self.d, 1.0, g);
        s.f_data = f;
        s.pull = self.lambda * (2.0 * s.prior_quad / self.lambda.max(1e-300)).sqrt() / gg.sqrt().max(1e-300);
        self.last = s;
        self.log();
        Ok(f)
    }
    fn has_gradient_sink(&self) -> bool {
        self.inner.has_gradient_sink()
    }
    /// d = x − D(x) first (it needs only x), then the inner gradient set by set with λ d_t added on its way into the
    /// sink (f32 downloads: the sum is encoded once, on the host)
    fn gradient_host_sink(&mut self, st: &Arc<CudaStream>, x: &HostStack, sink: &mut dyn crate::bfpstate::GradSink) -> Result<Option<f64>, Error> {
        if !self.inner.has_gradient_sink() {
            return Ok(None);
        }
        if self.d.lens != x.lens {
            self.d = HostStack::zeros(&x.lens);
        }
        self.d.copy_from(x);
        let mut s = self.residual_into(x);
        let (lam, d) = (self.lambda as f32, &self.d);
        let hook = move |t: usize, v: &mut [C32]| {
            v.par_chunks_mut(PAR).zip(d.chunk(t).par_chunks(PAR)).for_each(|(a, b)| {
                for (ai, bi) in a.iter_mut().zip(b) {
                    // as HostStack::axpby(λ, d, 1, g): λ d + 1 · g in f32
                    *ai = C32::new(lam * bi.re + 1.0 * ai.re, lam * bi.im + 1.0 * ai.im);
                }
            });
        };
        let mut ss = crate::bfpstate::SumSink { inner: sink, sq: 0.0, hook: Some(&hook) };
        let Some(f) = self.inner.gradient_host_sink(st, x, &mut ss)? else {
            // declined after offering (it should not): the caller falls back to gradient_host (the denoiser runs again)
            return Ok(None);
        };
        let gg = ss.sq;
        s.f_data = f;
        s.pull = self.lambda * (2.0 * s.prior_quad / self.lambda.max(1e-300)).sqrt() / gg.sqrt().max(1e-300);
        self.last = s;
        self.log();
        Ok(Some(f))
    }
    fn gradient(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error> {
        let f = self.inner.gradient(st, x, g)?;
        let lens = self.inner.lens();
        if self.xh.lens != lens {
            self.xh = HostStack::zeros(&lens);
            self.d = HostStack::zeros(&lens);
        }
        for t in 0..lens.len() {
            st.memcpy_dtoh(&x.v[t], self.xh.chunk_mut(t))?;
        }
        self.d.copy_from(&self.xh);
        let xh = std::mem::take(&mut self.xh);
        let mut s = self.residual_into(&xh);
        self.xh = xh;
        let mut gg = 0.0;
        let kern = self.inner.kernels();
        for t in 0..lens.len() {
            gg += kern.dot(st, &g.v[t], &g.v[t])?.re;
            let r = st.clone_htod(self.d.chunk(t))?;
            kern.axpby(st, Complex64::new(self.lambda, 0.0), &r, Complex64::new(1.0, 0.0), &mut g.v[t])?;
        }
        s.f_data = f;
        s.pull = self.lambda * (2.0 * s.prior_quad / self.lambda.max(1e-300)).sqrt() / gg.sqrt().max(1e-300);
        self.last = s;
        self.log();
        Ok(f)
    }
}
