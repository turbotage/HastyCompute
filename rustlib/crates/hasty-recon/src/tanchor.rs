//! L2 temporal anchor (research/temporal_pyramid.md Theorem 5 case B, §6 rule 2).
//!
//! The fine level's data-independent temporal average is pinned to a coarser solution:
//!
//!     f_A(x) = ½ μ Σ_a n_a ‖R_a x − c_a‖²,   R_a x = (1/n_a) Σ_{t∈a} x_t,
//!
//! a = an anchor group (the fine frames of one encode inside one coarse frame; for the pooled mean, all frames of an
//! encode), n_a its size, c_a the coarse image. For degree-0 frames nested into coarser degree-0 frames (or the pooled
//! mean, K_c = 1) the temporal L2 restriction (Sᵀ G S)⁻¹ Sᵀ G IS the plain mean over the group: G = diag(frame
//! widths) is uniform for equal frames. Never the data-weighted restriction: that one is satisfied by the free
//! solution already (Theorem 5 case A, inert).
//!
//! Gradient ∂f_A/∂x_t = μ (R_a x − c_a) for t ∈ a; Hessian μ (1/n_a) 1 1ᵀ ⊗ I per group: eigenvalue μ on the
//! group-constant mode, 0 on the detail. **Unit of μ** (what `--anchor ..:μ` gives, times the data scale): the
//! anchor's curvature on the group-constant mode. The command line takes μ relative to the mean diagonal d̄ of one
//! set's normal operator (tr N_t / n, `P2np::mean_diag`): μ_abs = μ · d̄, so μ = 1 makes the anchor as stiff on the
//! mean mode as one frame's data term is on average (the data term's curvature on the mean mode is the frame-average
//! Rayleigh quotient of N_t, ≈ d̄). Step safety: λ_max(N + A) ≤ λ_max(N) + μ_abs and λ_max(M (N + A)) ≤ λ_max(M N) +
//! μ_abs λ_max(M) (Weyl in the M⁻¹ inner product; `P2np::metric_lmax`).
//!
//! [`TAnchor`] wraps any [`Objective`] (P²nP data term, RED): the solver loops run unchanged. The anchor couples the
//! sets of a group, so the per-set ZMSHR1 secant groups are unioned with the anchor groups. Memory: the targets (one
//! image per group) on the host, plus — for device-resident solver state — their device copies and one image
//! temporary; host state (sets streamed) works on the host stack in place, no extra stack.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use num_complex::Complex64;
use rayon::prelude::*;

use crate::gpu::Kernels;
use crate::hoststack::HostStack;
use crate::p2np::{Objective, Stack};
use crate::Error;

const PAR: usize = 1 << 16;

/// Last anchor evaluation
#[derive(Clone, Copy, Debug, Default)]
pub struct AnchorStats {
    pub k: usize,
    /// data term f(x)
    pub f_data: f64,
    /// ½ μ Σ_a n_a ‖R_a x − c_a‖²
    pub f_anchor: f64,
    /// ‖R x − c‖ / ‖c‖ over all groups
    pub rel: f64,
}

/// f + f_A around an inner objective (see the module docs)
pub struct TAnchor<'a> {
    pub inner: &'a mut dyn Objective,
    /// set → anchor group
    pub group_of: Vec<usize>,
    /// c_a (one image per group, the set image length)
    pub target: Vec<Vec<C32>>,
    /// absolute weight μ_abs (see the module docs)
    pub mu: f64,
    /// print iterations k < 3 and every `log_every`-th (0: never)
    pub log_every: usize,
    pub last: AnchorStats,
    k: usize,
    tdev: Vec<CudaSlice<C32>>,
    tmp: Option<CudaSlice<C32>>,
}

impl<'a> TAnchor<'a> {
    pub fn new(inner: &'a mut dyn Objective, group_of: Vec<usize>, target: Vec<Vec<C32>>, mu: f64) -> Self {
        let lens = inner.lens();
        assert_eq!(group_of.len(), lens.len(), "temporal anchor: one group per set");
        assert!(group_of.iter().all(|&a| a < target.len()), "temporal anchor: group index without a target");
        assert!(group_of.iter().all(|&a| target[a].len() == lens[0]) && lens.iter().all(|&l| l == lens[0]), "temporal anchor: one image per set, the target's size");
        Self { inner, group_of, target, mu, log_every: 5, last: AnchorStats::default(), k: 0, tdev: vec![], tmp: None }
    }

    fn log(&self) {
        let s = &self.last;
        if self.log_every > 0 && (s.k < 3 || s.k % self.log_every == 0) {
            eprintln!("    anchor it {}: μ {:.3e}, ‖R x − c‖/‖c‖ {:.3e}, f {:.4e} + anchor {:.4e}", s.k, self.mu, s.rel, s.f_data, s.f_anchor);
        }
    }
}

/// The anchor on host stacks: returns (f_A, ‖R x − c‖/‖c‖) and adds ∇f_A to `g` when given (f64 group means)
pub fn anchor_host(x: &HostStack, group_of: &[usize], target: &[Vec<C32>], mu: f64, g: Option<&mut HostStack>) -> (f64, f64) {
    let na = target.len();
    let nv = target.first().map_or(0, |c| c.len());
    let mut fa = 0.0f64;
    let (mut e2, mut c2) = (0.0f64, 0.0f64);
    let mut ms: Vec<Vec<C32>> = Vec::with_capacity(na);
    for a in 0..na {
        let mem: Vec<usize> = (0..group_of.len()).filter(|&t| group_of[t] == a).collect();
        if mem.is_empty() {
            ms.push(vec![]);
            continue;
        }
        let mut acc = vec![[0.0f64; 2]; nv];
        for &t in &mem {
            acc.par_chunks_mut(PAR).zip(x.chunk(t).par_chunks(PAR)).for_each(|(a, b)| {
                for (ai, bi) in a.iter_mut().zip(b) {
                    ai[0] += bi.re as f64;
                    ai[1] += bi.im as f64;
                }
            });
        }
        let inv = 1.0 / mem.len() as f64;
        let m: Vec<C32> = acc.par_iter().zip(target[a].par_iter()).map(|(s, c)| C32::new((s[0] * inv - c.re as f64) as f32, (s[1] * inv - c.im as f64) as f32)).collect();
        let mm: f64 = m.par_iter().map(|v| v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64).sum();
        c2 += target[a].par_iter().map(|v| v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64).sum::<f64>();
        e2 += mm;
        fa += 0.5 * mu * mem.len() as f64 * mm;
        ms.push(m);
    }
    if let Some(g) = g {
        let m32 = mu as f32;
        for t in 0..group_of.len() {
            let m = &ms[group_of[t]];
            g.chunk_mut(t).par_chunks_mut(PAR).zip(m.par_chunks(PAR)).for_each(|(gc, mc)| {
                for (gi, mi) in gc.iter_mut().zip(mc) {
                    *gi = C32::new(gi.re + m32 * mi.re, gi.im + m32 * mi.im);
                }
            });
        }
    }
    (fa, (e2 / c2.max(1e-300)).sqrt())
}

/// union of two secant groupings (sets in one group of either share a group)
pub fn union_groups(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut p: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        let mut j = i;
        while p[j] != r {
            let nx = p[j];
            p[j] = r;
            j = nx;
        }
        r
    }
    for key in [a, b] {
        let mut first: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for t in 0..n {
            match first.get(&key[t]) {
                Some(&s) => {
                    let (ra, rb) = (find(&mut p, s), find(&mut p, t));
                    if ra != rb {
                        p[rb] = ra;
                    }
                }
                None => {
                    first.insert(key[t], t);
                }
            }
        }
    }
    let mut id: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    (0..n)
        .map(|t| {
            let r = find(&mut p, t);
            let k = id.len();
            *id.entry(r).or_insert(k)
        })
        .collect()
}

/// Anchor groups of a nested frame pyramid: fine frame `fr` = (respiratory bin b, cardiac phase c) of `nf_card`
/// phases per bin, coarse level with `nf_coarse` frames over the same bins (`resp_bins`); the coarse frame of a fine
/// one is (b, c / m), m = nf_card / (nf_coarse / resp_bins). Panics if the levels do not nest (integer ratios only,
/// research/temporal_pyramid.md §6 rule 1). `nf_coarse` = 1: the pooled mean over all frames.
pub fn nested_frame(fr: usize, nf_card: usize, resp_bins: usize, nf_coarse: usize) -> usize {
    if nf_coarse == 1 {
        return 0;
    }
    assert!(nf_coarse % resp_bins == 0, "temporal anchor: {nf_coarse} coarse frames over {resp_bins} respiratory bins do not nest");
    let kc = nf_coarse / resp_bins;
    assert!(kc > 0 && nf_card % kc == 0, "temporal anchor: {kc} coarse cardiac frames do not nest into {nf_card} (integer ratios only)");
    let m = nf_card / kc;
    (fr / nf_card) * kc + (fr % nf_card) / m
}

impl Objective for TAnchor<'_> {
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
        Some(match self.inner.secant_groups() {
            Some(g) => union_groups(&g, &self.group_of),
            None => self.group_of.clone(),
        })
    }
    fn limit_step(&mut self, st: &Arc<CudaStream>, t: usize, step: &mut CudaSlice<C32>) -> Result<usize, Error> {
        self.inner.limit_step(st, t, step)
    }
    fn limit_step_host(&mut self, t: usize, step: &mut [C32]) -> usize {
        self.inner.limit_step_host(t, step)
    }
    fn has_step_limit(&self) -> bool {
        self.inner.has_step_limit()
    }
    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        self.inner.metric(st, t, r, z, inverse)
    }
    fn metric_host(&mut self, st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inverse: bool) -> Result<(), Error> {
        self.inner.metric_host(st, t, r, z, inverse)
    }
    /// the inner (unanchored) grouped gradient: pooled solves are not anchored
    fn gradient_groups_host(&mut self, st: &Arc<CudaStream>, groups: &[Vec<usize>], x: &HostStack, g: &mut HostStack) -> Result<Option<f64>, Error> {
        self.inner.gradient_groups_host(st, groups, x, g)
    }
    fn gradient_host(&mut self, st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        let f = self.inner.gradient_host(st, x, g)?;
        let (fa, rel) = anchor_host(x, &self.group_of, &self.target, self.mu, Some(g));
        self.last = AnchorStats { k: self.k, f_data: f, f_anchor: fa, rel };
        self.log();
        Ok(f + fa)
    }
    fn gradient(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error> {
        let f = self.inner.gradient(st, x, g)?;
        if self.tdev.is_empty() {
            self.tdev = self.target.iter().map(|c| st.clone_htod(c)).collect::<Result<_, _>>()?;
            self.tmp = Some(st.alloc_zeros::<C32>(self.target[0].len())?);
        }
        let kern = self.inner.kernels();
        let (zero, one) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
        let tmp = self.tmp.as_mut().unwrap();
        let (mut fa, mut e2, mut c2) = (0.0f64, 0.0f64, 0.0f64);
        for a in 0..self.target.len() {
            let mem: Vec<usize> = (0..self.group_of.len()).filter(|&t| self.group_of[t] == a).collect();
            if mem.is_empty() {
                continue;
            }
            let inv = Complex64::new(1.0 / mem.len() as f64, 0.0);
            // tmp = mean − c (the first member overwrites: b = 0)
            kern.axpby(st, inv, &x.v[mem[0]], zero, tmp)?;
            for &t in &mem[1..] {
                kern.axpby(st, inv, &x.v[t], one, tmp)?;
            }
            kern.axpby(st, Complex64::new(-1.0, 0.0), &self.tdev[a], one, tmp)?;
            let mm = kern.dot(st, tmp, tmp)?.re;
            c2 += kern.dot(st, &self.tdev[a], &self.tdev[a])?.re;
            e2 += mm;
            fa += 0.5 * self.mu * mem.len() as f64 * mm;
            for &t in &mem {
                kern.axpby(st, Complex64::new(self.mu, 0.0), tmp, one, &mut g.v[t])?;
            }
        }
        self.last = AnchorStats { k: self.k, f_data: f, f_anchor: fa, rel: (e2 / c2.max(1e-300)).sqrt() };
        self.log();
        Ok(f + fa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(nt: usize, nv: usize, seed: u64) -> HostStack {
        let mut h = HostStack::zeros(&vec![nv; nt]);
        let mut s = seed | 1;
        for v in h.data.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *v = C32::new((s % 1000) as f32 / 500.0 - 1.0, ((s >> 20) % 1000) as f32 / 500.0 - 1.0);
        }
        h
    }

    /// gradient = finite difference of f_A (f64 reference), and the plain-mean restriction is the L2 one
    #[test]
    fn anchor_gradient_matches_fd() {
        let (nt, nv) = (6, 5);
        let group_of = vec![0, 1, 0, 1, 0, 1];
        let x = stack(nt, nv, 7);
        let tg: Vec<Vec<C32>> = (0..2).map(|a| stack(1, nv, 11 + a).data).collect();
        let mu = 0.7;
        let mut g = HostStack::zeros(&vec![nv; nt]);
        let (f0, _) = anchor_host(&x, &group_of, &tg, mu, Some(&mut g));
        // reference f in f64
        let fref = |x: &HostStack| -> f64 {
            let mut f = 0.0;
            for a in 0..2 {
                let mem: Vec<usize> = (0..nt).filter(|&t| group_of[t] == a).collect();
                for q in 0..nv {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for &t in &mem {
                        re += x.chunk(t)[q].re as f64;
                        im += x.chunk(t)[q].im as f64;
                    }
                    re = re / mem.len() as f64 - tg[a][q].re as f64;
                    im = im / mem.len() as f64 - tg[a][q].im as f64;
                    f += 0.5 * mu * mem.len() as f64 * (re * re + im * im);
                }
            }
            f
        };
        assert!((f0 - fref(&x)).abs() < 1e-5 * fref(&x));
        // directional derivative along a random direction: Re⟨g, d⟩
        let d = stack(nt, nv, 99);
        let h = 1e-2f32;
        let mut xp = x.clone();
        let mut xm = x.clone();
        HostStack::axpby(h as f64, &d, 1.0, &mut xp);
        HostStack::axpby(-(h as f64), &d, 1.0, &mut xm);
        let fd = (fref(&xp) - fref(&xm)) / (2.0 * h as f64);
        let an = HostStack::rdot(&g, &d);
        assert!((fd - an).abs() < 1e-3 * an.abs().max(1.0), "fd {fd} vs analytic {an}");
    }

    #[test]
    fn union_and_nesting() {
        assert_eq!(union_groups(&[0, 0, 1, 2], &[5, 6, 6, 7]), vec![0, 0, 0, 1]);
        // 20 cardiac frames into 10 / 5 / 1; two respiratory bins of 4 into 2 per bin
        assert_eq!((0..20).map(|f| nested_frame(f, 20, 1, 10)).collect::<Vec<_>>(), (0..20).map(|f| f / 2).collect::<Vec<_>>());
        assert_eq!(nested_frame(19, 20, 1, 5), 4);
        assert_eq!(nested_frame(13, 20, 1, 1), 0);
        assert_eq!((0..8).map(|f| nested_frame(f, 4, 2, 4)).collect::<Vec<_>>(), vec![0, 0, 1, 1, 2, 2, 3, 3]);
        assert!(std::panic::catch_unwind(|| nested_frame(0, 20, 1, 8)).is_err());
    }
}
