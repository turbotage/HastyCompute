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

use crate::p2np::zmshr1_scalars;

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
    let quasi = matches!(precond, Precond::Zmshr1 | Precond::Zmshr1M);
    let metric = precond == Precond::Zmshr1M;
    let mut g = HostStack::zeros(&lens);
    // the step as a stack only for the metric preconditioners; plain steps are fused into the update below
    let needs_pg = matches!(precond, Precond::Circulant | Precond::Zmshr1M);
    let mut pg = if needs_pg { HostStack::zeros(&lens) } else { HostStack::default() };
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
    // M (or M⁻¹) applied chunk by chunk
    let apply_metric = |obj: &mut dyn Objective, src: &HostStack, dst: &mut HostStack, inverse: bool| -> Result<(), Error> {
        for t in 0..nt {
            let (o, l) = (dst.off[t], dst.lens[t]);
            obj.metric_host(st, t, src.chunk(t), &mut dst.data[o..o + l], inverse)?;
        }
        Ok(())
    };
    for k in k0..iters {
        obj.begin_iteration(k, iters);
        let f = obj.gradient_host(st, x, &mut g)?;
        // scaled gradient α g
        g.data.par_chunks_mut(PAR).for_each(|c| c.iter_mut().for_each(|v| *v = C32::new(v.re * alpha as f32, v.im * alpha as f32)));
        // step kinds: `plain` = the scaled gradient itself, `fused` = already applied (x, x_prev, g_prev updated)
        let (mut plain, mut fused) = (false, false);
        match precond {
            Precond::None => plain = true,
            Precond::Circulant => apply_metric(obj, &g, &mut pg, false)?,
            Precond::Zmshr1 | Precond::Zmshr1M if k == 0 || !obj.curvature_pair_ok() => {
                if metric { apply_metric(obj, &g, &mut pg, false)? } else { plain = true }
            }
            Precond::Zmshr1 => {
                // plain ZMSHR1 fused over the elements: s = x − x_prev, m = g − g_prev are never stored
                let el = |i: usize, x: &[C32], xp: &[C32], g: &[C32], gp: &[C32]| -> ([f64; 2], [f64; 2]) {
                    ([(x[i].re - xp[i].re) as f64, (x[i].im - xp[i].im) as f64], [(g[i].re - gp[i].re) as f64, (g[i].im - gp[i].im) as f64])
                };
                let dots = |f: &(dyn Fn([f64; 2], [f64; 2], [f64; 2]) -> [f64; 4] + Sync)| -> [f64; 4] {
                    x.data.par_chunks(PAR).zip(x_prev.data.par_chunks(PAR)).zip(g.data.par_chunks(PAR)).zip(g_prev.data.par_chunks(PAR))
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
                let d1 = dots(&|s, m, _| [s[0] * s[0] + s[1] * s[1], s[0] * m[0] + s[1] * m[1], m[0] * m[0] + m[1] * m[1], 0.0]);
                let (a, tau) = zmshr1_scalars(d1[0], d1[1], d1[2]);
                // z = a s + (1−a) m, w = s − τ z
                let wz = move |s: [f64; 2], m: [f64; 2]| -> ([f64; 2], [f64; 2]) {
                    let z = [a * s[0] + (1.0 - a) * m[0], a * s[1] + (1.0 - a) * m[1]];
                    ([s[0] - tau * z[0], s[1] - tau * z[1]], z)
                };
                let d2 = dots(&|s, m, gv| { let (w, z) = wz(s, m); [w[0] * z[0] + w[1] * z[1], w[0] * w[0] + w[1] * w[1], z[0] * z[0] + z[1] * z[1], w[0] * gv[0] + w[1] * gv[1]] });
                let (uv, ww, zz, wg) = (d2[0], d2[1], d2[2], d2[3]);
                let c = if uv > 1e-8 * ww.sqrt() * zz.sqrt() { wg / uv } else { 0.0 };
                // step = τ g + c w, fused with the update: x_prev ← x, g_prev ← g, x ← x − step (no step stack)
                x.data.par_chunks_mut(PAR).zip(x_prev.data.par_chunks_mut(PAR)).zip(g.data.par_chunks(PAR)).zip(g_prev.data.par_chunks_mut(PAR))
                    .for_each(|(((xc, xpc), gc), gpc)| {
                        for i in 0..xc.len() {
                            let (s, m) = el(i, xc, xpc, gc, gpc);
                            let (w, _) = wz(s, m);
                            let step = C32::new((tau * gc[i].re as f64 + c * w[0]) as f32, (tau * gc[i].im as f64 + c * w[1]) as f32);
                            xpc[i] = xc[i];
                            gpc[i] = gc[i];
                            xc[i] = C32::new(xc[i].re - step.re, xc[i].im - step.im);
                        }
                    });
                fused = true;
            }
            Precond::Zmshr1M => {
                // with the metric: s, m and M⁻¹ s, M m as host temporaries (the metric is applied chunk by chunk)
                let mut s = x.clone();
                HostStack::axpby(-1.0, &x_prev, 1.0, &mut s);
                let mut m = g.clone();
                HostStack::axpby(-1.0, &g_prev, 1.0, &mut m);
                let (ms, mm_) = {
                    let (mut a, mut b) = (HostStack::zeros(&lens), HostStack::zeros(&lens));
                    apply_metric(obj, &s, &mut a, true)?;
                    apply_metric(obj, &m, &mut b, false)?;
                    (a, b)
                };
                let ss = HostStack::rdot(&s, &ms);
                let sm = HostStack::rdot(&s, &m);
                let mm = HostStack::rdot(&m, &mm_);
                let (a, tau) = zmshr1_scalars(ss, sm, mm);
                // w = (1 − τa) s − τ(1−a) Mm, z = (1−a) m + a M⁻¹s
                let mut w = s;
                HostStack::axpby(-tau * (1.0 - a), &mm_, 1.0 - tau * a, &mut w);
                let mut z = m;
                HostStack::axpby(a, &ms, 1.0 - a, &mut z);
                let (uv, ww, zz, wg) = (HostStack::rdot(&w, &z), HostStack::rdot(&w, &w), HostStack::rdot(&z, &z), HostStack::rdot(&w, &g));
                // step = τ M g (+ (⟨w, g⟩ / ⟨w, z⟩) w)
                apply_metric(obj, &g, &mut pg, false)?;
                HostStack::axpby(0.0, &g, tau, &mut pg);
                if uv > 1e-8 * ww.sqrt() * zz.sqrt() {
                    HostStack::axpby(wg / uv, &w, 1.0, &mut pg);
                }
                if k % 10 == 0 {
                    eprintln!("    zmshr1-M it {k}: a {a:.3} tau {tau:.3e}");
                }
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
        } else if !fused {
            if quasi {
                x_prev.copy_from(x);
                std::mem::swap(&mut g_prev, &mut g);
            }
            HostStack::axpby(-1.0, &pg, 1.0, x);
        }
        denoise(k, &mut x.data);
        report(k, x, f);
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
