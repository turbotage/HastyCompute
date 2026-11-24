//! Mean + residual deconvolution initialiser (research/temporal_pyramid.md §10, eq. (8)):
//!
//!     x_k = x̄ + (N_tot/N)⁻¹ A_kᴴ (y_k − A_k x̄),   x̄ = the pooled solution of the encode, N = its number of frames,
//!
//! one step of preconditioned Richardson for each frame's own problem, started at the pooled mean, with the inverse
//! mean normal operator N̄_e = (1/n_e) Σ_{s∈e} N_s as the preconditioner; more steps (x_k ← x_k + N̄⁻¹ A_kᴴ(y_k − A_k
//! x_k)) move it toward per-frame least squares (less bias, more variance: §10.5). Frame mode only (one image per
//! set; the sets of group e = the frames of encode e).
//!
//! The inverse N̄_e⁻¹ r_k, for every frame k at once (multiple right-hand sides sharing the encode's operator):
//! - [`Inner::Pcg`]`(n)`: n PCG iterations on N̄_e per frame, preconditioned by the encode's fixed metric (circulant)
//!   when there is one. Each iteration is one grouped gradient pass with one group per frame (the group = all sets of
//!   the frame's encode: `Objective::gradient_groups_host`), i.e. ≈ n_e ordinary gradient passes.
//! - [`Inner::Circ`]: the circulant metric M_e alone as the approximate inverse, with its unknown scale fitted per
//!   encode by an exact line search of the frames' own objectives along d = M r (one extra gradient pass, which also
//!   yields the next residual: a step costs one pass). Without a metric: plain steepest descent (warned).
//!
//! Memory (host): x, ∇f and three (Circ) or five (Pcg) more set stacks; nothing on the device beyond the gradient's
//! two streaming slots. Stoppable between steps (`save_raw` of x at the step boundary, research/memory_ledger.md: one
//! stack written only on a STOP).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaStream;
use hasty_cuda::C32;
use rayon::prelude::*;

use crate::checkpoint::{Outcome, RunCtl};
use crate::hoststack::{clear_raw, load_raw, save_raw, HostStack};
use crate::p2np::Objective;
use crate::Error;

const PAR: usize = 1 << 16;

/// How N̄_e⁻¹ is applied
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inner {
    /// the per-encode circulant metric, scaled by a per-encode line search
    Circ,
    /// n PCG iterations on the mean normal operator (multiple right-hand sides)
    Pcg(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeanRes {
    /// Richardson steps (1 = eq. (8))
    pub steps: usize,
    pub inner: Inner,
}

impl MeanRes {
    /// "meanres[:STEPS][:circ|pcgN]" (defaults: `default_steps`, circ); None for anything else
    pub fn parse(s: &str, default_steps: usize) -> Option<Self> {
        let mut it = s.split(':');
        if it.next()? != "meanres" {
            return None;
        }
        let mut m = MeanRes { steps: default_steps, inner: Inner::Circ };
        for p in it {
            if let Ok(n) = p.parse::<usize>() {
                m.steps = n;
            } else if p == "circ" {
                m.inner = Inner::Circ;
            } else if let Some(n) = p.strip_prefix("pcg") {
                m.inner = Inner::Pcg(n.parse().ok().filter(|&n: &usize| n > 0)?);
            } else {
                return None;
            }
        }
        (m.steps > 0).then_some(m)
    }
    /// short name for run tags: mr<steps><c|p<n>>
    pub fn tag(&self) -> String {
        match self.inner {
            Inner::Circ => format!("mr{}c", self.steps),
            Inner::Pcg(n) => format!("mr{}p{n}", self.steps),
        }
    }
}

fn dot(a: &[C32], b: &[C32]) -> f64 {
    a.par_chunks(PAR).zip(b.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
}

/// y = α x + β y on slices
fn axpby(alpha: f64, x: &[C32], beta: f64, y: &mut [C32]) {
    let (al, be) = (alpha as f32, beta as f32);
    y.par_chunks_mut(PAR).zip(x.par_chunks(PAR)).for_each(|(yc, xc)| {
        for (yi, xi) in yc.iter_mut().zip(xc) {
            *yi = C32::new(al * xi.re + be * yi.re, al * xi.im + be * yi.im);
        }
    });
}

/// z_t = M_t r_t for every set (copy without a metric)
fn apply_metric(obj: &mut dyn Objective, st: &Arc<CudaStream>, metric: bool, r: &HostStack, z: &mut HostStack) -> Result<(), Error> {
    for t in 0..r.nsets() {
        if metric {
            let (o, l) = (z.off[t], z.lens[t]);
            obj.metric_host(st, t, r.chunk(t), &mut z.data[o..o + l], false)?;
        } else {
            z.chunk_mut(t).copy_from_slice(r.chunk(t));
        }
    }
    Ok(())
}

/// The initialiser (see the module docs): `group_of` = set → encode, `xbar[e]` = the pooled solution of encode e.
/// Returns the start stack (all sets) and Finished, or Stopped(step) after saving the state at a step boundary
/// (`ctl` = (run control, state name); rerun to resume).
pub fn meanres_init(obj: &mut dyn Objective, st: &Arc<CudaStream>, group_of: &[usize], xbar: &[Vec<C32>], spec: MeanRes, metric: bool, ctl: Option<(&RunCtl, &str)>) -> Result<(HostStack, Outcome), Error> {
    let lens = obj.lens();
    let nt = lens.len();
    let ne = xbar.len();
    let nv = xbar[0].len();
    assert_eq!(group_of.len(), nt, "meanres: one encode per set");
    assert!(lens.iter().all(|&l| l == nv), "meanres: one image per set (frame mode)");
    let members: Vec<Vec<usize>> = (0..ne).map(|e| (0..nt).filter(|&t| group_of[t] == e).collect()).collect();
    assert!(members.iter().all(|m| !m.is_empty()), "meanres: an encode without sets");
    let mut x = HostStack::zeros(&lens);
    let mut k0 = 0;
    if let Some((c, name)) = ctl {
        if let Some(m) = load_raw(c, name, &mut [&mut x])? {
            k0 = m["k"].as_u64().unwrap_or(0) as usize;
            eprintln!("  meanres: resuming at step {k0} / {}", spec.steps);
        }
    }
    if k0 == 0 {
        for t in 0..nt {
            x.chunk_mut(t).copy_from_slice(&xbar[group_of[t]]);
        }
    }
    if !metric {
        eprintln!("  meanres: no metric (circulant) in this run: the inner inverse is unpreconditioned");
    }
    let t0 = Instant::now();
    let mut g = HostStack::zeros(&lens);
    let mut f = obj.gradient_host(st, &x, &mut g)?;
    println!("  meanres start (step {k0}): f = {f:.4e} ({:.1}s)", t0.elapsed().as_secs_f64());
    // pcg: Σ_{s∈e} ∇f_s(0) per encode (once), the per-frame group lists
    let mut g0: Option<HostStack> = None;
    let fgroups: Vec<Vec<usize>> = (0..nt).map(|t| members[group_of[t]].clone()).collect();
    for k in k0..spec.steps {
        let ts = Instant::now();
        match spec.inner {
            Inner::Circ => {
                // d = M (−∇f); N d = ∇f(x + d) − ∇f(x); per encode s_e = ⟨r, d⟩ / ⟨d, N d⟩
                let mut r = g.clone();
                r.data.par_iter_mut().for_each(|v| *v = C32::new(-v.re, -v.im));
                let mut d = HostStack::zeros(&lens);
                apply_metric(obj, st, metric, &r, &mut d)?;
                drop(r);
                let mut xd = x.clone();
                HostStack::axpby(1.0, &d, 1.0, &mut xd);
                let mut nd = HostStack::zeros(&lens);
                obj.gradient_host(st, &xd, &mut nd)?;
                drop(xd);
                HostStack::axpby(-1.0, &g, 1.0, &mut nd);
                let mut scales = Vec::with_capacity(ne);
                let mut df = 0.0;
                for e in 0..ne {
                    let (mut rd, mut dnd) = (0.0, 0.0);
                    for &t in &members[e] {
                        rd -= dot(g.chunk(t), d.chunk(t));
                        dnd += dot(d.chunk(t), nd.chunk(t));
                    }
                    let s = if dnd > 0.0 { rd / dnd } else { 0.0 };
                    df += -s * rd + 0.5 * s * s * dnd;
                    for &t in &members[e] {
                        axpby(s, d.chunk(t), 1.0, x.chunk_mut(t));
                        axpby(s, nd.chunk(t), 1.0, g.chunk_mut(t));
                    }
                    scales.push(s);
                }
                f += df;
                println!("  meanres step {} (circulant, per-encode line search): f = {f:.4e}, scales {} ({:.1}s)", k + 1, scales.iter().map(|s| format!("{s:.3e}")).collect::<Vec<_>>().join(" "), ts.elapsed().as_secs_f64());
            }
            Inner::Pcg(n) => {
                if g0.is_none() {
                    let glens = vec![nv; ne];
                    let mut out = HostStack::zeros(&glens);
                    let z = HostStack::zeros(&glens);
                    obj.gradient_groups_host(st, &members, &z, &mut out)?.ok_or_else(|| Error::Format("meanres pcg: the objective has no grouped gradient (use :circ)".into()))?;
                    g0 = Some(out);
                }
                let g0r = g0.as_ref().unwrap();
                // N̄ p: one grouped pass, group t = the sets of t's encode, minus Σ ∇f(0), / n_e
                let nbar = |obj: &mut dyn Objective, p: &HostStack, q: &mut HostStack| -> Result<(), Error> {
                    obj.gradient_groups_host(st, &fgroups, p, q)?.ok_or_else(|| Error::Format("meanres pcg: no grouped gradient".into()))?;
                    for t in 0..nt {
                        let e = group_of[t];
                        let inv = 1.0 / members[e].len() as f64;
                        let gz = g0r.chunk(e);
                        q.chunk_mut(t).par_chunks_mut(PAR).zip(gz.par_chunks(PAR)).for_each(|(qc, zc)| {
                            for (qi, zi) in qc.iter_mut().zip(zc) {
                                *qi = C32::new(((qi.re as f64 - zi.re as f64) * inv) as f32, ((qi.im as f64 - zi.im as f64) * inv) as f32);
                            }
                        });
                    }
                    Ok(())
                };
                // PCG per frame: N̄ δ = r, δ₀ = 0
                let mut res = g.clone();
                res.data.par_iter_mut().for_each(|v| *v = C32::new(-v.re, -v.im));
                let mut delta = HostStack::zeros(&lens);
                let mut z = HostStack::zeros(&lens);
                apply_metric(obj, st, metric, &res, &mut z)?;
                let mut p = z.clone();
                let mut q = HostStack::zeros(&lens);
                let mut rz: Vec<f64> = (0..nt).map(|t| dot(res.chunk(t), z.chunk(t))).collect();
                let r0: Vec<f64> = (0..nt).map(|t| dot(res.chunk(t), res.chunk(t)).sqrt()).collect();
                let mut rel = vec![1.0f64; nt];
                for i in 0..n {
                    nbar(obj, &p, &mut q)?;
                    for t in 0..nt {
                        let pq = dot(p.chunk(t), q.chunk(t));
                        if !(pq > 0.0) || rz[t] == 0.0 {
                            continue;
                        }
                        let a = rz[t] / pq;
                        axpby(a, p.chunk(t), 1.0, delta.chunk_mut(t));
                        axpby(-a, q.chunk(t), 1.0, res.chunk_mut(t));
                    }
                    apply_metric(obj, st, metric, &res, &mut z)?;
                    for t in 0..nt {
                        let rzn = dot(res.chunk(t), z.chunk(t));
                        let b = if rz[t] != 0.0 { rzn / rz[t] } else { 0.0 };
                        rz[t] = rzn;
                        axpby(1.0, z.chunk(t), b, p.chunk_mut(t));
                        rel[t] = dot(res.chunk(t), res.chunk(t)).sqrt() / r0[t].max(1e-300);
                    }
                    let mut s = rel.clone();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    eprintln!("    meanres pcg it {}: ‖r‖/‖r₀‖ median {:.2e}, max {:.2e}", i + 1, s[nt / 2], s[nt - 1]);
                }
                drop((res, z, p, q));
                HostStack::axpby(1.0, &delta, 1.0, &mut x);
                drop(delta);
                f = obj.gradient_host(st, &x, &mut g)?;
                println!("  meanres step {} (PCG {n} on the mean normal operator): f = {f:.4e} ({:.1}s)", k + 1, ts.elapsed().as_secs_f64());
            }
        }
        if let Some((c, name)) = ctl {
            if c.stop_requested() && k + 1 < spec.steps {
                save_raw(c, name, &serde_json::json!({ "k": k + 1 }), &[&x], None)?;
                return Ok((x, Outcome::Stopped(k + 1)));
            }
        }
    }
    if let Some((c, name)) = ctl {
        clear_raw(c, name);
    }
    Ok((x, Outcome::Finished))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_specs() {
        assert_eq!(MeanRes::parse("meanres", 1), Some(MeanRes { steps: 1, inner: Inner::Circ }));
        assert_eq!(MeanRes::parse("meanres:2", 1), Some(MeanRes { steps: 2, inner: Inner::Circ }));
        assert_eq!(MeanRes::parse("meanres:pcg4", 3), Some(MeanRes { steps: 3, inner: Inner::Pcg(4) }));
        assert_eq!(MeanRes::parse("meanres:2:pcg3", 1), Some(MeanRes { steps: 2, inner: Inner::Pcg(3) }));
        assert_eq!(MeanRes::parse("meanres:circ:1", 2), Some(MeanRes { steps: 1, inner: Inner::Circ }));
        assert_eq!(MeanRes::parse("meanres:pcg0", 1), None);
        assert_eq!(MeanRes::parse("meanres:x", 1), None);
        assert_eq!(MeanRes::parse("warm", 1), None);
        assert_eq!(MeanRes::parse("meanres:2:pcg3", 1).unwrap().tag(), "mr2p3");
    }
}
