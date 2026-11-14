//! Host-resident solver state: P²nP for problems whose image stacks do not fit the GPU (high resolution, many
//! cardiac × respiratory frames).
//!
//! - [`HostStack`]: one contiguous host buffer per stack (all sets), offsets per set. The P²nP / ZMSHR1 arithmetic
//!   (a few axpby / dot passes) runs on the CPU with rayon: memory-bound, small next to the gradient.
//! - [`SetStreamer`]: the gradient streams set by set through two device slots. The next set's image is copied into
//!   pinned memory by a worker thread while the GPU computes the current set, then uploaded on a copy stream; the
//!   finished gradient slice comes back the same way. The device holds two image slots and two gradient slots, never a
//!   stack.
//! - [`run_objective_host`]: the P²nP loop of [`crate::p2np::run_objective_with`] on host stacks (same updates, same
//!   checkpoint keys), for any [`Objective`] through [`Objective::gradient_host`] / [`Objective::metric_host`].

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
use hasty_cuda::C32;
use hasty_value::{DType, Tensor, Value};
use rayon::prelude::*;

use crate::checkpoint::{get, get_usize, Outcome, RunCtl};
use crate::p2np::{Objective, Precond};
use crate::Error;

/// Host stack: all sets' chunks in one buffer.
#[derive(Clone, Debug, Default)]
pub struct HostStack {
    pub data: Vec<C32>,
    pub off: Vec<usize>,
    pub lens: Vec<usize>,
}

const PAR: usize = 1 << 16;

impl HostStack {
    pub fn zeros(lens: &[usize]) -> Self {
        let off: Vec<usize> = lens.iter().scan(0, |a, &n| { let o = *a; *a += n; Some(o) }).collect();
        Self { data: vec![C32::new(0.0, 0.0); lens.iter().sum()], off, lens: lens.to_vec() }
    }
    pub fn nsets(&self) -> usize {
        self.lens.len()
    }
    pub fn chunk(&self, t: usize) -> &[C32] {
        &self.data[self.off[t]..self.off[t] + self.lens[t]]
    }
    pub fn chunk_mut(&mut self, t: usize) -> &mut [C32] {
        &mut self.data[self.off[t]..self.off[t] + self.lens[t]]
    }
    pub fn bytes(&self) -> usize {
        self.data.len() * 8
    }
    pub fn copy_from(&mut self, o: &HostStack) {
        self.data.par_chunks_mut(PAR).zip(o.data.par_chunks(PAR)).for_each(|(a, b)| a.copy_from_slice(b));
    }
    /// b = α a + β b (real scalars)
    pub fn axpby(alpha: f64, a: &HostStack, beta: f64, b: &mut HostStack) {
        let (al, be) = (alpha as f32, beta as f32);
        b.data.par_chunks_mut(PAR).zip(a.data.par_chunks(PAR)).for_each(|(y, x)| {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = C32::new(al * xi.re + be * yi.re, al * xi.im + be * yi.im);
            }
        });
    }
    /// Re⟨a, b⟩ (f64 accumulation)
    pub fn rdot(a: &HostStack, b: &HostStack) -> f64 {
        a.data.par_chunks(PAR).zip(b.data.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
    }
    pub fn to_value(&self) -> Result<Value, Error> {
        Ok(Value::List(
            (0..self.nsets())
                .map(|t| {
                    let h = self.chunk(t);
                    // SAFETY: C32 is two f32, no padding
                    let b = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 8) };
                    Ok(Value::Tensor(Tensor::from_bytes(DType::C64, vec![h.len()], b)?))
                })
                .collect::<Result<_, Error>>()?,
        ))
    }
    pub fn load_value(&mut self, v: &Value) -> Result<(), Error> {
        let Value::List(items) = v else { return Err(Error::Format("checkpoint: expected a list".into())) };
        if items.len() != self.nsets() {
            return Err(Error::Format(format!("checkpoint: {} sets, expected {}", items.len(), self.nsets())));
        }
        for (t, it) in items.iter().enumerate() {
            let tn = it.as_tensor().ok_or_else(|| Error::Format("checkpoint: expected a tensor".into()))?;
            let n = self.lens[t];
            if tn.dtype() != DType::C64 || tn.numel() != n {
                return Err(Error::Format(format!("checkpoint: set {t}: {} values, expected {n} complex64", tn.numel())));
            }
            // SAFETY: the tensor buffer holds n complex64 = C32 values
            let h = unsafe { std::slice::from_raw_parts(tn.bytes().as_ptr() as *const C32, n) };
            self.chunk_mut(t).copy_from_slice(h);
        }
        Ok(())
    }
}

/// Two device slots for set images and two for set gradients, pinned staging, a copy stream. All sets must have the
/// same length (one frame image or all frames of a set).
pub struct SetStreamer {
    up: Arc<CudaStream>,
    dx: [CudaSlice<C32>; 2],
    dg: [CudaSlice<C32>; 2],
    hx: [PinnedHostSlice<C32>; 2],
    hg: [PinnedHostSlice<C32>; 2],
    ev_up: [CudaEvent; 2],
    ev_comp: [CudaEvent; 2],
    ev_down: [CudaEvent; 2],
    len: usize,
    /// sets streamed, bytes up / down
    pub stats: (usize, usize, usize),
}

impl SetStreamer {
    pub fn new(ctx: &Arc<CudaContext>, st: &Arc<CudaStream>, len: usize) -> Result<Self, Error> {
        let up = ctx.new_stream()?;
        let dev = || st.alloc_zeros::<C32>(len.max(1));
        let pin = || unsafe { ctx.alloc_pinned_with_flags::<C32>(len.max(1), 0) };
        let ev = || ctx.new_event(None);
        let s = Self {
            up,
            dx: [dev()?, dev()?],
            dg: [dev()?, dev()?],
            hx: [pin()?, pin()?],
            hg: [pin()?, pin()?],
            ev_up: [ev()?, ev()?],
            ev_comp: [ev()?, ev()?],
            ev_down: [ev()?, ev()?],
            len,
            stats: (0, 0, 0),
        };
        for e in s.ev_up.iter().chain(&s.ev_comp).chain(&s.ev_down) {
            e.record(st)?;
        }
        Ok(s)
    }

    pub fn device_bytes(&self) -> usize {
        4 * self.len * 8
    }

    /// For every set t in `order`: `f(t, x_t, g_t)` on the device (x_t uploaded from `x`, g_t written by f and
    /// downloaded into `g`). The next set's image is staged by a worker thread while f runs.
    pub fn for_each(&mut self, st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack, order: &[usize], f: &mut dyn FnMut(usize, &CudaSlice<C32>, &mut CudaSlice<C32>) -> Result<(), Error>) -> Result<(), Error> {
        if order.is_empty() {
            return Ok(());
        }
        assert!(order.iter().all(|&t| x.lens[t] == self.len && g.lens[t] == self.len), "SetStreamer: every set must have {} elements", self.len);
        let n = self.len;
        // prime: set order[0] into slot 0
        self.ev_up[0].synchronize()?;
        self.hx[0].as_mut_slice()?[..n].copy_from_slice(x.chunk(order[0]));
        self.up.wait(&self.ev_comp[0])?;
        self.up.memcpy_htod(&self.hx[0].as_slice()?[..n], &mut self.dx[0])?;
        self.ev_up[0].record(&self.up)?;
        for (k, &t) in order.iter().enumerate() {
            let (b, nb) = (k % 2, (k + 1) % 2);
            let next = order.get(k + 1).copied();
            let prev = if k > 0 { Some(order[k - 1]) } else { None };
            // compute set t (waits for its upload)
            st.wait(&self.ev_up[b])?;
            // worker: stage the next image into hx[nb] (its previous upload is done once ev_up[nb] completed) and
            // collect the previous gradient from hg[nb] (downloaded behind ev_down[nb]) while the GPU computes
            let (hx_a, hx_b) = self.hx.split_at_mut(1);
            let hx_nb: &mut PinnedHostSlice<C32> = if nb == 0 { &mut hx_a[0] } else { &mut hx_b[0] };
            let hg_nb: &PinnedHostSlice<C32> = &self.hg[nb];
            let (ev_up_nb, ev_down_nb) = (&self.ev_up[nb], &self.ev_down[nb]);
            let g_prev_dst: Option<&mut [C32]> = prev.map(|p| {
                let (o, l) = (g.off[p], g.lens[p]);
                // SAFETY-free split: a disjoint mutable chunk of g (the worker writes only this one)
                &mut g.data[o..o + l]
            });
            let (dx_a, dx_b) = (&self.dx[0], &self.dx[1]);
            let (dg_a, dg_b) = self.dg.split_at_mut(1);
            let dg_b_ref: &mut CudaSlice<C32> = if b == 0 { &mut dg_a[0] } else { &mut dg_b[0] };
            let dx_b_ref = if b == 0 { dx_a } else { dx_b };
            let res = std::thread::scope(|s| -> Result<(), Error> {
                let h = s.spawn(move || -> Result<(), Error> {
                    if let Some(tn) = next {
                        ev_up_nb.synchronize()?;
                        hx_nb.as_mut_slice()?[..n].copy_from_slice(x.chunk(tn));
                    }
                    if let Some(dst) = g_prev_dst {
                        ev_down_nb.synchronize()?;
                        dst.copy_from_slice(&hg_nb.as_slice()?[..n]);
                    }
                    Ok(())
                });
                let r = f(t, dx_b_ref, dg_b_ref);
                let w = h.join().expect("set streamer worker panicked");
                r.and(w)
            });
            res?;
            self.ev_comp[b].record(st)?;
            // download g_t; upload the next image once the slot's previous compute is done
            self.up.wait(&self.ev_comp[b])?;
            self.up.memcpy_dtoh(&self.dg[b], &mut self.hg[b].as_mut_slice()?[..n])?;
            self.ev_down[b].record(&self.up)?;
            if next.is_some() {
                self.up.wait(&self.ev_comp[nb])?;
                self.up.memcpy_htod(&self.hx[nb].as_slice()?[..n], &mut self.dx[nb])?;
                self.ev_up[nb].record(&self.up)?;
            }
            self.stats.0 += 1;
            self.stats.1 += n * 8;
            self.stats.2 += n * 8;
        }
        // the last gradient
        let (lb, lt) = ((order.len() - 1) % 2, *order.last().unwrap());
        self.ev_down[lb].synchronize()?;
        g.chunk_mut(lt).copy_from_slice(&self.hg[lb].as_slice()?[..n]);
        Ok(())
    }
}

use crate::p2np::{zmshr1_rank_one_grouped, zmshr1_scalars_grouped};

/// [`crate::p2np::run_objective_with`] on host stacks: x, ∇f, the step and the ZMSHR1 secant pair live in host memory;
/// the gradient streams set by set ([`Objective::gradient_host`]); the update arithmetic runs on the CPU.
#[allow(clippy::too_many_arguments)]
const CK_MAGIC: &[u8; 8] = b"HSTKCK01";

fn ck_path(c: &RunCtl, name: &str) -> std::path::PathBuf {
    c.dir().join(format!("state_{name}.hsk"))
}

/// Raw checkpoint of host stacks, streamed from the stacks themselves (no in-memory copy: the hasty-value form needs a
/// tensor copy of every set, i.e. one more stack per saved stack): magic, header length, JSON header (`meta`, the set
/// lengths of every stack), then the stacks' complex64 data in order. Written to a temporary file, synced, renamed.
pub fn save_raw(c: &RunCtl, name: &str, meta: &serde_json::Value, stacks: &[&HostStack]) -> Result<(), Error> {
    use std::io::Write;
    let path = ck_path(c, name);
    let tmp = c.dir().join(format!("state_{name}.hsk.tmp"));
    {
        let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp)?);
        let hdr = serde_json::json!({ "meta": meta, "lens": stacks.iter().map(|s| s.lens.clone()).collect::<Vec<_>>() }).to_string();
        w.write_all(CK_MAGIC)?;
        w.write_all(&(hdr.len() as u64).to_le_bytes())?;
        w.write_all(hdr.as_bytes())?;
        for s in stacks {
            // SAFETY: C32 is two f32, no padding
            let b = unsafe { std::slice::from_raw_parts(s.data.as_ptr() as *const u8, s.data.len() * 8) };
            for part in b.chunks(64 << 20) {
                w.write_all(part)?;
            }
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    // a hasty-value checkpoint of the same name is now stale
    c.clear(name);
    Ok(())
}

/// Load a raw checkpoint into `stacks` (lengths must match); returns its `meta`, None without a checkpoint.
pub fn load_raw(c: &RunCtl, name: &str, stacks: &mut [&mut HostStack]) -> Result<Option<serde_json::Value>, Error> {
    use std::io::Read;
    let path = ck_path(c, name);
    if !path.exists() {
        return Ok(None);
    }
    let mut r = std::io::BufReader::with_capacity(8 << 20, std::fs::File::open(&path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != CK_MAGIC {
        return Err(Error::Format(format!("{}: not a host-stack checkpoint", path.display())));
    }
    let mut n = [0u8; 8];
    r.read_exact(&mut n)?;
    let mut hdr = vec![0u8; u64::from_le_bytes(n) as usize];
    r.read_exact(&mut hdr)?;
    let h: serde_json::Value = serde_json::from_slice(&hdr).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    let lens: Vec<Vec<usize>> = serde_json::from_value(h["lens"].clone()).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    if lens.len() != stacks.len() || lens.iter().zip(stacks.iter()).any(|(l, s)| *l != s.lens) {
        return Err(Error::Format(format!("{}: stack layout differs from the run's", path.display())));
    }
    for s in stacks.iter_mut() {
        // SAFETY: as in save_raw
        let b = unsafe { std::slice::from_raw_parts_mut(s.data.as_mut_ptr() as *mut u8, s.data.len() * 8) };
        for part in b.chunks_mut(64 << 20) {
            r.read_exact(part)?;
        }
    }
    Ok(Some(h["meta"].clone()))
}

/// How [`pooled_pcg_host_with`] evaluates the group gradients Σ_{t∈g} ∇f_t(x_g).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PooledGrad {
    /// the objective's grouped gradient ([`Objective::gradient_groups_host`]) when it has one, else `Replicate`
    Auto,
    /// the grouped gradient (an error if the objective has none): each group image is used once by all of its sets
    /// (`SetOp::forward_group` / `adjoint_group`: for the segmented operator one grid FFT per segment and coil batch
    /// per group instead of per set); no full-size host stacks
    Grouped,
    /// copy every group image into each of its sets, one full-stack `gradient_host`, sum per group (two host stacks
    /// of all sets; every set pays its own transforms)
    Replicate,
}

impl PooledGrad {
    /// parse a command-line name: auto, grouped, replicate
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "auto" => PooledGrad::Auto,
            "grouped" => PooledGrad::Grouped,
            "replicate" => PooledGrad::Replicate,
            _ => return None,
        })
    }
}

/// Pooled preconditioned CG through the full objective (the physics-model warm start).
///
/// For every group g of sets (e.g. all frames of one encode) solve min_x Σ_{t∈g} f_t(x) for one image x_g shared by
/// the group's sets: CG on the normal equations Σ_{t∈g} N_t x = Σ_{t∈g} A_t^H y_t, independently per group (own CG
/// scalars), all groups advanced by one gradient pass over all sets per iteration. The products use only the
/// objective's gradient (affine in x): N p = ∇f(p) − ∇f(0). `metric`: precondition group g by the objective's metric
/// of its first set (with shared circulants, the group's circulant). `x[g]` is the initial guess, updated in place.
/// Resumable through raw checkpoints every `every` iterations (`state_<name>.hsk`). The group gradients come from
/// the objective's grouped gradient when it has one ([`PooledGrad::Auto`], [`pooled_pcg_host_with`]).
#[allow(clippy::too_many_arguments)]
pub fn pooled_pcg_host(
    obj: &mut dyn Objective,
    st: &Arc<CudaStream>,
    group_of: &[usize],
    x: &mut [Vec<C32>],
    iters: usize,
    metric: bool,
    ctl: Option<(&RunCtl, &str)>,
    every: usize,
) -> Result<Outcome, Error> {
    pooled_pcg_host_with(obj, st, group_of, x, iters, metric, ctl, every, PooledGrad::Auto)
}

/// [`pooled_pcg_host`] with the group-gradient evaluation chosen by `grad`. The iterates do not depend on it (up to
/// float32 summation order), so a checkpoint written under one mode resumes under the other.
#[allow(clippy::too_many_arguments)]
pub fn pooled_pcg_host_with(
    obj: &mut dyn Objective,
    st: &Arc<CudaStream>,
    group_of: &[usize],
    x: &mut [Vec<C32>],
    iters: usize,
    metric: bool,
    ctl: Option<(&RunCtl, &str)>,
    every: usize,
    grad: PooledGrad,
) -> Result<Outcome, Error> {
    let lens = obj.lens();
    let nt = lens.len();
    let ng = x.len();
    let nv = x[0].len();
    assert_eq!(group_of.len(), nt, "pooled PCG: one group per set");
    assert!(lens.iter().all(|&l| l == nv), "pooled PCG: every set must hold one image of the group's size");
    assert!(group_of.iter().all(|&g| g < ng));
    let rep: Vec<usize> = (0..ng).map(|g| group_of.iter().position(|&h| h == g).expect("pooled PCG: empty group")).collect();
    let glens = vec![nv; ng];
    let groups: Vec<Vec<usize>> = (0..ng).map(|g| (0..nt).filter(|&t| group_of[t] == g).collect()).collect();
    // replicate path: full stacks (all sets), allocated on first use
    let mut full: Option<(HostStack, HostStack)> = None;
    // per-group vectors as host stacks of ng chunks (raw checkpoints)
    let (mut xs, mut rs, mut ps, mut g0) = (HostStack::zeros(&glens), HostStack::zeros(&glens), HostStack::zeros(&glens), HostStack::zeros(&glens));
    for g in 0..ng {
        xs.chunk_mut(g).copy_from_slice(&x[g]);
    }
    let mut rz = vec![0.0f64; ng];
    let mut r0 = vec![0.0f64; ng];
    let mut k0 = 0;
    let replicate = |v: &HostStack, big: &mut HostStack| {
        for t in 0..nt {
            big.chunk_mut(t).copy_from_slice(v.chunk(group_of[t]));
        }
    };
    // Σ_{t∈g} G_t − sub_g into out (f64 accumulation per group)
    let group_sum = |grad: &HostStack, sub: Option<&HostStack>, out: &mut HostStack| {
        for g in 0..ng {
            let mut acc = vec![[0.0f64; 2]; nv];
            for t in (0..nt).filter(|&t| group_of[t] == g) {
                acc.par_chunks_mut(PAR).zip(grad.chunk(t).par_chunks(PAR)).for_each(|(a, b)| {
                    for (ai, bi) in a.iter_mut().zip(b) {
                        ai[0] += bi.re as f64;
                        ai[1] += bi.im as f64;
                    }
                });
            }
            let o = out.chunk_mut(g);
            match sub {
                Some(s) => {
                    let sg = s.chunk(g);
                    o.par_chunks_mut(PAR).zip(acc.par_chunks(PAR)).zip(sg.par_chunks(PAR)).for_each(|((oc, ac), sc)| {
                        for ((oi, ai), si) in oc.iter_mut().zip(ac).zip(sc) {
                            *oi = C32::new((ai[0] - si.re as f64) as f32, (ai[1] - si.im as f64) as f32);
                        }
                    })
                }
                None => o.par_chunks_mut(PAR).zip(acc.par_chunks(PAR)).for_each(|(oc, ac)| {
                    for (oi, ai) in oc.iter_mut().zip(ac) {
                        *oi = C32::new(ai[0] as f32, ai[1] as f32);
                    }
                }),
            }
        }
    };
    let precond = |obj: &mut dyn Objective, r: &HostStack, z: &mut HostStack| -> Result<(), Error> {
        for g in 0..ng {
            if metric {
                let (o, l) = (z.off[g], z.lens[g]);
                obj.metric_host(st, rep[g], r.chunk(g), &mut z.data[o..o + l], false)?;
            } else {
                z.chunk_mut(g).copy_from_slice(r.chunk(g));
            }
        }
        Ok(())
    };
    let gdot = |a: &HostStack, b: &HostStack, g: usize| -> f64 {
        a.chunk(g).par_chunks(PAR).zip(b.chunk(g).par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
    };
    let gaxpy = |alpha: f64, a: &HostStack, b: &mut HostStack, g: usize, beta: f64| {
        let (al, be) = (alpha as f32, beta as f32);
        b.chunk_mut(g).par_chunks_mut(PAR).zip(a.chunk(g).par_chunks(PAR)).for_each(|(y, x)| {
            for (yi, xi) in y.iter_mut().zip(x) {
                *yi = C32::new(al * xi.re + be * yi.re, al * xi.im + be * yi.im);
            }
        });
    };
    // out_g = Σ_{t∈g} ∇f_t(v_g) − sub_g: grouped, or replicated into the full stack and summed (f64)
    let mut group_grad = |obj: &mut dyn Objective, v: &HostStack, sub: Option<&HostStack>, out: &mut HostStack| -> Result<(), Error> {
        if grad != PooledGrad::Replicate {
            if obj.gradient_groups_host(st, &groups, v, out)?.is_some() {
                if let Some(s) = sub {
                    HostStack::axpby(-1.0, s, 1.0, out);
                }
                return Ok(());
            }
            if grad == PooledGrad::Grouped {
                return Err(Error::Format("pooled PCG: the objective has no grouped gradient (PooledGrad::Grouped)".into()));
            }
        }
        let (big, gradf) = full.get_or_insert_with(|| (HostStack::zeros(&lens), HostStack::zeros(&lens)));
        replicate(v, big);
        obj.gradient_host(st, big, gradf)?;
        group_sum(gradf, sub, out);
        Ok(())
    };
    let mut loaded = false;
    if let Some((c, name)) = ctl {
        if let Some(m) = load_raw(c, name, &mut [&mut xs, &mut rs, &mut ps, &mut g0])? {
            k0 = m["k"].as_u64().unwrap_or(0) as usize;
            rz = serde_json::from_value(m["rz"].clone()).map_err(|e| Error::Format(format!("pooled PCG checkpoint: {e}")))?;
            r0 = serde_json::from_value(m["r0"].clone()).map_err(|e| Error::Format(format!("pooled PCG checkpoint: {e}")))?;
            loaded = true;
            eprintln!("  resuming pooled PCG {name} at iteration {k0} / {iters}");
        }
    }
    if !loaded {
        // g0 = Σ ∇f(0) = −b (one pass); r = b − N x = −Σ ∇f(x) (a second pass unless x = 0)
        group_grad(obj, &HostStack::zeros(&glens), None, &mut g0)?;
        let x_zero = xs.data.iter().all(|v| v.re == 0.0 && v.im == 0.0);
        if x_zero {
            rs.copy_from(&g0);
        } else {
            group_grad(obj, &xs, None, &mut rs)?;
        }
        rs.data.par_chunks_mut(PAR).for_each(|c| c.iter_mut().for_each(|v| *v = C32::new(-v.re, -v.im)));
        let mut zs = HostStack::zeros(&glens);
        precond(obj, &rs, &mut zs)?;
        ps.copy_from(&zs);
        for g in 0..ng {
            rz[g] = gdot(&rs, &zs, g);
            r0[g] = gdot(&rs, &rs, g).sqrt();
        }
    }
    let mut zs = HostStack::zeros(&glens);
    let mut qs = HostStack::zeros(&glens);
    for k in k0..iters {
        // q = N p = Σ ∇f(p) − Σ ∇f(0)
        group_grad(obj, &ps, Some(&g0), &mut qs)?;
        for g in 0..ng {
            let pq = gdot(&ps, &qs, g);
            if !(pq > 0.0) || rz[g] == 0.0 {
                continue;
            }
            let a = rz[g] / pq;
            gaxpy(a, &ps, &mut xs, g, 1.0);
            gaxpy(-a, &qs, &mut rs, g, 1.0);
        }
        precond(obj, &rs, &mut zs)?;
        let mut rel = Vec::with_capacity(ng);
        for g in 0..ng {
            let rz_new = gdot(&rs, &zs, g);
            let b = if rz[g] != 0.0 { rz_new / rz[g] } else { 0.0 };
            rz[g] = rz_new;
            gaxpy(1.0, &zs, &mut ps, g, b);
            rel.push(gdot(&rs, &rs, g).sqrt() / r0[g].max(1e-300));
        }
        eprintln!("  pooled PCG it {}: ‖r‖/‖r₀‖ per group {}", k + 1, rel.iter().map(|v| format!("{v:.2e}")).collect::<Vec<_>>().join(" "));
        if let Some((c, name)) = ctl {
            let next = k + 1;
            let meta = serde_json::json!({ "k": next, "rz": rz, "r0": r0 });
            if c.stop_requested() && next < iters {
                save_raw(c, name, &meta, &[&xs, &rs, &ps, &g0])?;
                return Ok(Outcome::Stopped(next));
            }
            if every > 0 && next % every == 0 && next < iters {
                save_raw(c, name, &meta, &[&xs, &rs, &ps, &g0])?;
            }
        }
    }
    for g in 0..ng {
        x[g].copy_from_slice(xs.chunk(g));
    }
    if let Some((c, name)) = ctl {
        clear_raw(c, name);
    }
    Ok(Outcome::Finished)
}

/// Re⟨a, b⟩ of two slices (f64 accumulation)
fn rdot_slice(a: &[C32], b: &[C32]) -> f64 {
    a.par_chunks(PAR).zip(b.par_chunks(PAR)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>()).sum()
}

/// b = α a + β b on slices (real scalars, f32 arithmetic as [`HostStack::axpby`])
fn axpby_slice(alpha: f64, a: &[C32], beta: f64, b: &mut [C32]) {
    let (al, be) = (alpha as f32, beta as f32);
    b.par_chunks_mut(PAR).zip(a.par_chunks(PAR)).for_each(|(y, x)| {
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi = C32::new(al * xi.re + be * yi.re, al * xi.im + be * yi.im);
        }
    });
}

/// Metric step fused per set chunk t: step_t = τ_t M g_t + c_t w_t (w_t read from x_prev before it is overwritten;
/// `prev` = (x_prev, g_prev), None for the non-quasi circulant step), then x_prev_t ← x_t, g_prev_t ← g_t and
/// x_t ← x_t − step_t. `tc[t]` = (τ_t, c_t); `tmp` holds one chunk.
fn metric_step_fused(obj: &mut dyn Objective, st: &Arc<CudaStream>, x: &mut HostStack, g: &HostStack, mut prev: Option<(&mut HostStack, &mut HostStack)>, tc: &[(f64, f64)], tmp: &mut [C32]) -> Result<(), Error> {
    for t in 0..x.nsets() {
        let (l, (tau, c)) = (x.lens[t], tc[t]);
        let sp = &mut tmp[..l];
        obj.metric_host(st, t, g.chunk(t), sp, false)?;
        if tau != 1.0 || c != 0.0 {
            match prev.as_ref() {
                Some((xp, _)) if c != 0.0 => axpby_slice(c, xp.chunk(t), tau, sp),
                _ => sp.par_iter_mut().for_each(|v| *v = C32::new(tau as f32 * v.re, tau as f32 * v.im)),
            }
        }
        if let Some((xp, gp)) = prev.as_mut() {
            xp.chunk_mut(t).par_chunks_mut(PAR).zip(x.chunk(t).par_chunks(PAR)).for_each(|(a, b)| a.copy_from_slice(b));
            gp.chunk_mut(t).par_chunks_mut(PAR).zip(g.chunk(t).par_chunks(PAR)).for_each(|(a, b)| a.copy_from_slice(b));
        }
        axpby_slice(-1.0, sp, 1.0, x.chunk_mut(t));
    }
    Ok(())
}

pub fn clear_raw(c: &RunCtl, name: &str) {
    let _ = std::fs::remove_file(ck_path(c, name));
}

pub fn run_objective_host(
    obj: &mut dyn Objective,
    st: &Arc<CudaStream>,
    x: &mut HostStack,
    precond: Precond,
    mut alpha: f64,
    iters: usize,
    denoise: &mut dyn FnMut(usize, &mut Vec<C32>),
    report: &mut dyn FnMut(usize, &HostStack, f64),
    ctl: Option<(&RunCtl, &str)>,
) -> Result<Outcome, Error> {
    let lens = x.lens.clone();
    let nt = lens.len();
    let quasi = precond.quasi();
    // ZMSHR1 in the circulant metric (the plain circulant step has its own arm)
    let metric = quasi && precond.metric();
    let per_set = precond.per_set();
    let groups = obj.secant_groups();
    let mut g = HostStack::zeros(&lens);
    // metric steps are fused per set chunk (M applied chunk by chunk into two chunk-sized temporaries): no step stack
    // and no full-size s, m, M⁻¹s, Mm (the secant pair overwrites x_prev, g_prev in place, both dead after the step),
    // so the host state is x, g, x_prev, g_prev (4 stacks instead of 9 at the peak)
    let maxl = lens.iter().copied().max().unwrap_or(0);
    let (mut ta, mut tb) = if precond.metric() { (vec![C32::new(0.0, 0.0); maxl], vec![C32::new(0.0, 0.0); maxl]) } else { (Vec::new(), Vec::new()) };
    let (mut x_prev, mut g_prev) = if quasi { (HostStack::zeros(&lens), HostStack::zeros(&lens)) } else { (HostStack::default(), HostStack::default()) };
    let mut k0 = 0;
    let mut raw_loaded = false;
    if let Some((c, name)) = ctl {
        let mut st_list: Vec<&mut HostStack> = if quasi { vec![&mut *x, &mut x_prev, &mut g_prev] } else { vec![&mut *x] };
        if let Some(m) = load_raw(c, name, &mut st_list)? {
            k0 = m["k"].as_u64().ok_or_else(|| Error::Format("checkpoint: k".into()))? as usize;
            alpha = m["alpha"].as_f64().ok_or_else(|| Error::Format("checkpoint: alpha".into()))?;
            raw_loaded = true;
            eprintln!("  resuming {name} at iteration {k0} / {iters} (host-resident state, raw checkpoint)");
        }
    }
    if let (Some((c, name)), false) = (ctl, raw_loaded) {
        if let Some(v) = c.load(name)? {
            k0 = get_usize(&v, "k")?;
            alpha = crate::checkpoint::get_f64(&v, "alpha")?;
            x.load_value(get(&v, "x")?)?;
            if quasi {
                x_prev.load_value(get(&v, "x_prev")?)?;
                g_prev.load_value(get(&v, "g_prev")?)?;
            }
            eprintln!("  resuming {name} at iteration {k0} / {iters} (host-resident state)");
        }
    }
    if k0 == 0 {
        report(usize::MAX, x, f64::NAN);
    }
    // raw checkpoints streamed from the stacks (no copies; the hasty-value form is still read for older runs)
    let save = |c: &RunCtl, name: &str, next: usize, x: &HostStack, xp: &HostStack, gp: &HostStack| -> Result<(), Error> {
        let meta = serde_json::json!({ "k": next, "precond": format!("{precond:?}"), "alpha": alpha });
        let stacks: Vec<&HostStack> = if quasi { vec![x, xp, gp] } else { vec![x] };
        save_raw(c, name, &meta, &stacks)
    };
    for k in k0..iters {
        let mut tm = crate::p2np::IterTimer::start();
        obj.begin_iteration(k, iters);
        let f = match obj.gradient_host(st, x, &mut g) {
            Ok(f) => f,
            Err(e) => {
                // the state is consistent at the start of iteration k (the gradient only reads x): save it, so the
                // run resumes here instead of from the last checkpoint (no periodic checkpoints by default)
                if let (Some((c, name)), true) = (ctl, k > k0) {
                    eprintln!("  gradient failed at iteration {k} ({e}): saving a checkpoint to resume from iteration {k}");
                    if let Err(e2) = save(c, name, k, x, &x_prev, &g_prev) {
                        eprintln!("  checkpoint failed too: {e2}");
                    }
                }
                return Err(e);
            }
        };
        tm.lap(st, 0);
        // scaled gradient α g
        g.data.par_chunks_mut(PAR).for_each(|c| c.iter_mut().for_each(|v| *v = C32::new(v.re * alpha as f32, v.im * alpha as f32)));
        // step kinds: `plain` = the scaled gradient itself, `fused` = already applied (x, x_prev, g_prev updated)
        let (mut plain, mut fused) = (false, false);
        match precond {
            Precond::None => plain = true,
            Precond::Circulant => {
                metric_step_fused(obj, st, x, &g, None, &vec![(1.0, 0.0); nt], &mut ta)?;
                fused = true;
            }
            Precond::Zmshr1 | Precond::Zmshr1M | Precond::Zmshr1Set | Precond::Zmshr1MSet if k == 0 || !obj.curvature_pair_ok() => {
                if metric {
                    metric_step_fused(obj, st, x, &g, Some((&mut x_prev, &mut g_prev)), &vec![(1.0, 0.0); nt], &mut ta)?;
                    fused = true;
                } else {
                    plain = true
                }
            }
            Precond::Zmshr1 | Precond::Zmshr1Set => {
                // plain ZMSHR1 fused over the elements: s = x − x_prev, m = g − g_prev are never stored
                let el = |i: usize, x: &[C32], xp: &[C32], g: &[C32], gp: &[C32]| -> ([f64; 2], [f64; 2]) {
                    ([(x[i].re - xp[i].re) as f64, (x[i].im - xp[i].im) as f64], [(g[i].re - gp[i].re) as f64, (g[i].im - gp[i].im) as f64])
                };
                // element ranges: one per set (per-set ZMSHR1) or the whole stack (global)
                let ranges: Vec<std::ops::Range<usize>> = if per_set { (0..nt).map(|t| x.off[t]..x.off[t] + lens[t]).collect() } else { vec![0..x.data.len()] };
                let dots = |r: &std::ops::Range<usize>, f: &(dyn Fn([f64; 2], [f64; 2], [f64; 2]) -> [f64; 4] + Sync)| -> [f64; 4] {
                    x.data[r.clone()].par_chunks(PAR).zip(x_prev.data[r.clone()].par_chunks(PAR)).zip(g.data[r.clone()].par_chunks(PAR)).zip(g_prev.data[r.clone()].par_chunks(PAR))
                        .map(|(((xc, xpc), gc), gpc)| {
                            let mut acc = [0.0f64; 4];
                            for i in 0..xc.len() {
                                let (s, m) = el(i, xc, xpc, gc, gpc);
                                let v = f(s, m, [gc[i].re as f64, gc[i].im as f64]);
                                for j in 0..4 { acc[j] += v[j]; }
                            }
                            acc
                        })
                        .reduce(|| [0.0; 4], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
                };
                let d1: Vec<[f64; 3]> = ranges.iter().map(|r| {
                    let v = dots(r, &|s, m, _| [s[0] * s[0] + s[1] * s[1], s[0] * m[0] + s[1] * m[1], m[0] * m[0] + m[1] * m[1], 0.0]);
                    [v[0], v[1], v[2]]
                }).collect();
                let at = zmshr1_scalars_grouped(&d1, per_set, groups.as_deref());
                // z = a s + (1−a) m, w = s − τ z
                let wz = |a: f64, tau: f64, s: [f64; 2], m: [f64; 2]| -> ([f64; 2], [f64; 2]) {
                    let z = [a * s[0] + (1.0 - a) * m[0], a * s[1] + (1.0 - a) * m[1]];
                    ([s[0] - tau * z[0], s[1] - tau * z[1]], z)
                };
                let d2: Vec<[f64; 4]> = ranges.iter().zip(&at).map(|(r, &(a, tau))| {
                    dots(r, &|s, m, gv| { let (w, z) = wz(a, tau, s, m); [w[0] * z[0] + w[1] * z[1], w[0] * w[0] + w[1] * w[1], z[0] * z[0] + z[1] * z[1], w[0] * gv[0] + w[1] * gv[1]] })
                }).collect();
                let ct = zmshr1_rank_one_grouped(&d2, per_set, groups.as_deref());
                // step = τ g + c w, fused with the update: x_prev ← x, g_prev ← g, x ← x − step (no step stack)
                for ((r, &(a, tau)), &c) in ranges.iter().zip(&at).zip(&ct) {
                    x.data[r.clone()].par_chunks_mut(PAR).zip(x_prev.data[r.clone()].par_chunks_mut(PAR)).zip(g.data[r.clone()].par_chunks(PAR)).zip(g_prev.data[r.clone()].par_chunks_mut(PAR))
                        .for_each(|(((xc, xpc), gc), gpc)| {
                            for i in 0..xc.len() {
                                let (s, m) = el(i, xc, xpc, gc, gpc);
                                let (w, _) = wz(a, tau, s, m);
                                let step = C32::new((tau * gc[i].re as f64 + c * w[0]) as f32, (tau * gc[i].im as f64 + c * w[1]) as f32);
                                xpc[i] = xc[i];
                                gpc[i] = gc[i];
                                xc[i] = C32::new(xc[i].re - step.re, xc[i].im - step.im);
                            }
                        });
                }
                if per_set && k % 10 == 0 {
                    let (lo, hi) = at.iter().fold((f64::INFINITY, 0.0f64), |(l, h), p| (l.min(p.1), h.max(p.1)));
                    eprintln!("    zmshr1 per set it {k}: tau {lo:.3e}..{hi:.3e}, a > 0 in {}/{nt} sets, rank-one in {}/{nt}", at.iter().filter(|p| p.0 > 0.0).count(), ct.iter().filter(|&&c| c != 0.0).count());
                }
                fused = true;
            }
            Precond::Zmshr1M | Precond::Zmshr1MSet => {
                // with the metric, chunk by chunk: s = x − x_prev overwrites x_prev, m = g − g_prev overwrites g_prev
                // (both dead after the step); M⁻¹s and Mm only as chunk temporaries, recomputed in the second pass
                HostStack::axpby(1.0, x, -1.0, &mut x_prev);
                HostStack::axpby(1.0, &g, -1.0, &mut g_prev);
                let (s, m) = (&mut x_prev, &mut g_prev);
                let mut d1: Vec<[f64; 3]> = Vec::with_capacity(nt);
                for t in 0..nt {
                    let l = lens[t];
                    obj.metric_host(st, t, s.chunk(t), &mut ta[..l], true)?;
                    obj.metric_host(st, t, m.chunk(t), &mut tb[..l], false)?;
                    d1.push([rdot_slice(s.chunk(t), &ta[..l]), rdot_slice(s.chunk(t), m.chunk(t)), rdot_slice(m.chunk(t), &tb[..l])]);
                }
                let at = zmshr1_scalars_grouped(&d1, per_set, groups.as_deref());
                // w = (1 − τa) s − τ(1−a) Mm into s, z = (1−a) m + a M⁻¹s into m (per set)
                let mut d2: Vec<[f64; 4]> = Vec::with_capacity(nt);
                for t in 0..nt {
                    let (l, (a, tau)) = (lens[t], at[t]);
                    obj.metric_host(st, t, s.chunk(t), &mut ta[..l], true)?;
                    obj.metric_host(st, t, m.chunk(t), &mut tb[..l], false)?;
                    axpby_slice(-tau * (1.0 - a), &tb[..l], 1.0 - tau * a, s.chunk_mut(t));
                    axpby_slice(a, &ta[..l], 1.0 - a, m.chunk_mut(t));
                    let (w, z) = (s.chunk(t), m.chunk(t));
                    d2.push([rdot_slice(w, z), rdot_slice(w, w), rdot_slice(z, z), rdot_slice(w, g.chunk(t))]);
                }
                let ct = zmshr1_rank_one_grouped(&d2, per_set, groups.as_deref());
                if k % 10 == 0 {
                    let (lo, hi) = at.iter().fold((f64::INFINITY, 0.0f64), |(l, h), p| (l.min(p.1), h.max(p.1)));
                    eprintln!("    zmshr1-M{} it {k}: a {:.3} tau {lo:.3e}..{hi:.3e}, rank-one in {}/{nt}", if per_set { " per set" } else { "" }, at[0].0, ct.iter().filter(|&&c| c != 0.0).count());
                }
                // step = τ_t M g + c_t w (w in x_prev), fused with x_prev ← x, g_prev ← g, x ← x − step
                let tc: Vec<(f64, f64)> = at.iter().zip(&ct).map(|(&(_, tau), &c)| (tau, c)).collect();
                metric_step_fused(obj, st, x, &g, Some((&mut x_prev, &mut g_prev)), &tc, &mut ta)?;
                fused = true;
            }
        }
        // z = x − P α g (the secant pair shifted first), x = D(z)
        if plain {
            if quasi {
                x.data.par_chunks_mut(PAR).zip(x_prev.data.par_chunks_mut(PAR)).zip(g.data.par_chunks(PAR)).zip(g_prev.data.par_chunks_mut(PAR))
                    .for_each(|(((xc, xpc), gc), gpc)| {
                        for i in 0..xc.len() {
                            xpc[i] = xc[i];
                            gpc[i] = gc[i];
                            xc[i] = C32::new(xc[i].re - gc[i].re, xc[i].im - gc[i].im);
                        }
                    });
            } else {
                HostStack::axpby(-1.0, &g, 1.0, x);
            }
        }
        debug_assert!(plain || fused);
        tm.lap(st, 1);
        denoise(k, &mut x.data);
        tm.lap(st, 2);
        report(k, x, f);
        tm.lap(st, 3);
        tm.log(k, ctl);
        if let Some((c, name)) = ctl {
            let next = k + 1;
            if c.stop_requested() && next < iters {
                save(c, name, next, x, &x_prev, &g_prev)?;
                return Ok(Outcome::Stopped(next));
            }
            if c.due(next) && next < iters {
                save(c, name, next, x, &x_prev, &g_prev)?;
            }
        }
    }
    if let Some((c, name)) = ctl {
        c.clear(name);
        clear_raw(c, name);
    }
    Ok(Outcome::Finished)
}
