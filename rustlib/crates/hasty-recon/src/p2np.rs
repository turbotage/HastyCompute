//! Preconditioned plug-and-play ISTA (P²nP; Hong, Xu, Hu, Fessler) for a stack of sample sets sharing one image grid:
//! x_t (t = frame × encode, or encode with all frames for time-resolved operators), f(x) = ½ Σ_t ‖E_t x_t − y_t‖², E_t
//! any [`SetOp`] (plain SENSE NUFFT or the segmented physics NUFFT).
//!   x_{k+1} = D(x_k − P_k α ∇f(x_k)),  ∇f_t = E_t^H (E_t x_t − y_t)   (no density compensation)
//! Preconditioners: none (PnP-ISTA), fixed circulant M_t (`CirculantPrecond`), dynamic ZMSHR1 (zero-memory symmetric
//! rank-1 from consecutive iterates; one gradient per iteration). α = 1/λ_max (of N, or of M N for the circulant).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use num_complex::Complex64;

use crate::checkpoint::{get, get_usize, stack_value, value_stack, Outcome, RunCtl};
use crate::gpu::Kernels;
use hasty_value::Value;
use crate::precond::CirculantPrecond;
use crate::setop::SetOp;
use crate::Error;

pub struct SetGpu {
    pub coords: [CudaSlice<f32>; 3],
    /// whitened, coil-compressed data, coil-major
    pub y: CudaSlice<C32>,
    pub m: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precond {
    None,
    Circulant,
    Zmshr1,
    /// ZMSHR1 in the metric of the circulant M: P_k = τ_k M + rank one (secant on M^{-1/2}-transformed variables)
    Zmshr1M,
}

pub struct Stack {
    pub v: Vec<CudaSlice<C32>>,
}

impl Stack {
    pub fn zeros(st: &Arc<CudaStream>, nt: usize, nv: usize) -> Result<Self, Error> {
        Ok(Self { v: (0..nt).map(|_| st.alloc_zeros(nv)).collect::<Result<_, _>>()? })
    }
    /// zero stack with per-set lengths
    pub fn zeros_lens(st: &Arc<CudaStream>, lens: &[usize]) -> Result<Self, Error> {
        Ok(Self { v: lens.iter().map(|&n| st.alloc_zeros(n)).collect::<Result<_, _>>()? })
    }
    pub fn copy_from(&mut self, st: &Arc<CudaStream>, o: &Stack) -> Result<(), Error> {
        for (a, b) in self.v.iter_mut().zip(&o.v) {
            st.memcpy_dtod(b, a)?;
        }
        Ok(())
    }
}

pub struct P2np<'a> {
    pub op: &'a mut dyn SetOp,
    /// per-set data (whitened, coil-compressed, coil-major)
    pub ys: Vec<&'a CudaSlice<C32>>,
    /// per-set fixed preconditioners: one circulant per frame block of the set's image (block-diagonal for
    /// time-resolved sets), empty if unused
    pub circ: Vec<Vec<CirculantPrecond>>,
    /// per-set residual buffers
    res: Vec<CudaSlice<C32>>,
    /// frame-block scratch (block-diagonal preconditioners)
    blk: Option<(CudaSlice<C32>, CudaSlice<C32>)>,
}

/// Re⟨a, b⟩ over all chunks (the real inner product of the stacked parameters)
pub fn rdot(k: &Kernels, st: &Arc<CudaStream>, a: &Stack, b: &Stack) -> Result<f64, Error> {
    let mut s = 0.0;
    for (x, y) in a.v.iter().zip(&b.v) {
        s += k.dot(st, x, y)?.re;
    }
    Ok(s)
}
/// b = α a + β b
pub fn axpby(k: &Kernels, st: &Arc<CudaStream>, alpha: f64, a: &Stack, beta: f64, b: &mut Stack) -> Result<(), Error> {
    for (x, y) in a.v.iter().zip(b.v.iter_mut()) {
        k.axpby(st, Complex64::new(alpha, 0.0), x, Complex64::new(beta, 0.0), y)?;
    }
    Ok(())
}

/// A smooth objective on a stack of complex chunks (images, or real parameters packed in pairs), for the P²nP loop.
pub trait Objective {
    fn kernels(&self) -> &Kernels;
    /// chunk lengths (complex elements)
    fn lens(&self) -> Vec<usize>;
    /// g = ∇f(x) (real gradient: df = Re⟨g, dx⟩); returns f(x)
    fn gradient(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error>;
    /// z = M_t r (`inverse`: M_t⁻¹ r) for the fixed metric M (≈ inverse Hessian) of chunk t; needed by
    /// [`Precond::Circulant`] and [`Precond::Zmshr1M`]
    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error>;
}

impl Objective for P2np<'_> {
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn lens(&self) -> Vec<usize> {
        P2np::lens(self)
    }
    fn gradient(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error> {
        P2np::gradient(self, st, x, g)
    }
    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        self.precond(st, t, r, z, inverse)
    }
}

impl<'a> P2np<'a> {
    pub fn new(op: &'a mut dyn SetOp, ys: Vec<&'a CudaSlice<C32>>, circ: Vec<Vec<CirculantPrecond>>, st: &Arc<CudaStream>) -> Result<Self, Error> {
        assert_eq!(ys.len(), op.nsets());
        assert!(circ.is_empty() || circ.len() == op.nsets());
        let res = (0..op.nsets()).map(|t| st.alloc_zeros(op.len_y(t))).collect::<Result<_, _>>()?;
        let blk = match circ.iter().position(|c| c.len() > 1) {
            Some(t) => {
                let n = op.len_x(t) / circ[t].len();
                Some((st.alloc_zeros(n)?, st.alloc_zeros(n)?))
            }
            None => None,
        };
        Ok(Self { op, ys, circ, res, blk })
    }

    /// z = M_t r (or M_t⁻¹ r), M_t block-diagonal over the set's frames
    fn precond(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        let nb = self.circ[t].len();
        if nb == 1 {
            let c = &mut self.circ[t][0];
            return if inverse { c.apply_inv(st, r, z) } else { c.apply(st, r, z) };
        }
        let n = r.len() / nb;
        let (a, b) = self.blk.as_mut().unwrap();
        for (i, c) in self.circ[t].iter_mut().enumerate() {
            st.memcpy_dtod(&r.slice(i * n..(i + 1) * n), a)?;
            if inverse { c.apply_inv(st, a, b)? } else { c.apply(st, a, b)? }
            st.memcpy_dtod(b, &mut z.slice_mut(i * n..(i + 1) * n))?;
        }
        Ok(())
    }

    /// per-set image lengths
    pub fn lens(&self) -> Vec<usize> {
        (0..self.op.nsets()).map(|t| self.op.len_x(t)).collect()
    }

    /// g = ∇f(x); returns f(x)
    pub fn gradient(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error> {
        let mut f = 0.0;
        for t in 0..self.ys.len() {
            let buf = &mut self.res[t];
            self.op.forward(st, t, &x.v[t], buf)?;
            self.op.kernels().axpby(st, Complex64::new(-1.0, 0.0), self.ys[t], Complex64::new(1.0, 0.0), buf)?;
            f += 0.5 * self.op.kernels().dot(st, buf, buf)?.re;
            self.op.adjoint(st, t, buf, &mut g.v[t])?;
        }
        Ok(f)
    }

    /// λ_max of N (or of M N) by power iteration on set 0 and the middle set
    pub fn step_norm(&mut self, st: &Arc<CudaStream>, with_circ: bool, iters: usize) -> Result<f64, Error> {
        let mut best: f64 = 0.0;
        let idx: Vec<usize> = { let mut v = vec![0, self.ys.len() / 2]; v.dedup(); v };
        for &t in &idx {
            let nx = self.op.len_x(t);
            let host: Vec<C32> = (0..nx).map(|i| C32::new(1.0 + 0.1 * (i % 7) as f32, 0.05 * (i % 5) as f32)).collect();
            let mut v = st.clone_htod(&host)?;
            let mut w = st.alloc_zeros::<C32>(nx)?;
            let mut u = st.alloc_zeros::<C32>(nx)?;
            let mut lam = 0.0;
            for _ in 0..iters {
                let n = self.op.kernels().dot(st, &v, &v)?.re.sqrt();
                self.op.kernels().axpby(st, Complex64::new(0.0, 0.0), &w, Complex64::new(1.0 / n, 0.0), &mut v)?;
                self.op.forward(st, t, &v, &mut self.res[t])?;
                self.op.adjoint(st, t, &self.res[t], &mut w)?;
                if with_circ {
                    self.precond(st, t, &w, &mut u, false)?;
                    std::mem::swap(&mut w, &mut u);
                    // Rayleigh quotient of M N in the M⁻¹ inner product ≈ ‖M N v‖ growth
                    lam = self.op.kernels().dot(st, &w, &w)?.re.sqrt();
                } else {
                    lam = self.op.kernels().dot(st, &v, &w)?.re;
                }
                std::mem::swap(&mut v, &mut w);
            }
            best = best.max(lam);
            if with_circ && t == 0 {
                if let Ok(path) = std::env::var("HASTY_DUMP_EIGVEC") {
                    let h = st.clone_dtoh(&v)?;
                    let bytes: Vec<u8> = h.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat()).collect();
                    std::fs::write(path, bytes)?;
                }
            }
        }
        Ok(best)
    }

    /// Run `iters` iterations from `x` (in place). `denoise(x_host_stack)` is applied to the host copy; `report(k, x)`
    /// is called after every iteration.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        st: &Arc<CudaStream>,
        x: &mut Stack,
        precond: Precond,
        alpha: f64,
        iters: usize,
        denoise: &mut dyn FnMut(&mut Vec<C32>),
        report: &mut dyn FnMut(usize, &Stack, f64),
    ) -> Result<(), Error> {
        self.run_from(st, x, precond, alpha, iters, &mut |_, h| denoise(h), report, None).map(|_| ())
    }

    /// Resumable [`run`](Self::run) (`denoise(k, x)` gets the iteration, so randomised priors such as LLR's cycle
    /// spinning can be a function of k and repeat exactly after a resume): with `ctl = (run control, state name)` the
    /// loop continues from a saved state
    /// (`k`, `x`, `x_prev`, `g_prev`; the latter two carry the ZMSHR1 secant pair), checkpoints every `ctl.every`
    /// iterations and stops cleanly on request, returning [`Outcome::Stopped`]. `x` is overwritten by a loaded state.
    #[allow(clippy::too_many_arguments)]
    pub fn run_from(
        &mut self,
        st: &Arc<CudaStream>,
        x: &mut Stack,
        precond: Precond,
        alpha: f64,
        iters: usize,
        denoise: &mut dyn FnMut(usize, &mut Vec<C32>),
        report: &mut dyn FnMut(usize, &Stack, f64),
        ctl: Option<(&RunCtl, &str)>,
    ) -> Result<Outcome, Error> {
        run_objective(self, st, x, precond, alpha, iters, denoise, report, ctl)
    }
}

/// Where the P²nP state lives (chosen by the memory planner, `plan.rs`): x, ∇f and the preconditioned step are always
/// on the device; the previous iterate / gradient (the ZMSHR1 secant pair) and the rank-one direction may be swapped
/// out to pinned host memory and streamed chunk by chunk.
#[derive(Clone)]
pub struct StateOpts {
    /// context for pinned host allocations (needed when anything is on the host)
    pub ctx: Option<Arc<cudarc::driver::CudaContext>>,
    pub prev_on_host: bool,
    pub w_mode: WMode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WMode {
    /// keep the rank-one direction w on the device (one stack)
    Device,
    /// keep it in pinned host memory
    Host,
    /// recompute it in a third pass (re-reads the secant pair)
    Recompute,
}

impl Default for StateOpts {
    fn default() -> Self {
        Self { ctx: None, prev_on_host: false, w_mode: WMode::Device }
    }
}

/// The P²nP loop on any [`Objective`] (see [`P2np::run_from`]): x_{k+1} = D(x_k − P_k α ∇f(x_k)), all state on the
/// device ([`run_objective_with`] for planner-placed state).
#[allow(clippy::too_many_arguments)]
pub fn run_objective(
    obj: &mut dyn Objective,
    st: &Arc<CudaStream>,
    x: &mut Stack,
    precond: Precond,
    alpha: f64,
    iters: usize,
    denoise: &mut dyn FnMut(usize, &mut Vec<C32>),
    report: &mut dyn FnMut(usize, &Stack, f64),
    ctl: Option<(&RunCtl, &str)>,
) -> Result<Outcome, Error> {
    run_objective_with(obj, st, x, precond, alpha, iters, denoise, report, ctl, &StateOpts::default())
}

/// ZMSHR1 secant scalars (Hong et al.): convex combination v = a s + (1−a) m with bounded curvature, then τ
fn zmshr1_scalars(ss: f64, sm: f64, mm: f64) -> (f64, f64) {
    let (th1, th2) = (2e-6, 200.0);
    let mut a = 1.0;
    for i in 0..=2000 {
        let aa = i as f64 / 2000.0;
        let s_v = aa * ss + (1.0 - aa) * sm;
        let vv = aa * aa * ss + 2.0 * aa * (1.0 - aa) * sm + (1.0 - aa) * (1.0 - aa) * mm;
        if s_v > 0.0 && s_v / ss >= th1 && vv / s_v <= th2 {
            a = aa;
            break;
        }
    }
    let s_v = a * ss + (1.0 - a) * sm;
    let vv = a * a * ss + 2.0 * a * (1.0 - a) * sm + (1.0 - a) * (1.0 - a) * mm;
    let tau = ss / s_v - ((ss / s_v).powi(2) - ss / vv).max(0.0).sqrt();
    (a, tau)
}

/// [`run_objective`] with planner-placed state. The ZMSHR1 / ZMSHR1-M step is evaluated chunk by chunk in two or three
/// passes (secant inner products; rank-one direction; its contribution), so besides x, g and the step only chunk-sized
/// scratch is needed on the device.
#[allow(clippy::too_many_arguments)]
pub fn run_objective_with(
    obj: &mut dyn Objective,
    st: &Arc<CudaStream>,
    x: &mut Stack,
    precond: Precond,
    mut alpha: f64,
    iters: usize,
    denoise: &mut dyn FnMut(usize, &mut Vec<C32>),
    report: &mut dyn FnMut(usize, &Stack, f64),
    ctl: Option<(&RunCtl, &str)>,
    opts: &StateOpts,
) -> Result<Outcome, Error> {
    use crate::store::{Scratch, Store};
    let nt = x.v.len();
    let lens = obj.lens();
    let off: Vec<usize> = lens.iter().scan(0, |a, &n| { let o = *a; *a += n; Some(o) }).collect();
    let quasi = matches!(precond, Precond::Zmshr1 | Precond::Zmshr1M);
    let metric = precond == Precond::Zmshr1M;
    let ctx = || opts.ctx.clone().ok_or_else(|| Error::Format("StateOpts: host placement needs a context".into()));
    let mut g = Stack::zeros_lens(st, &lens)?;
    let mut pg = Stack::zeros_lens(st, &lens)?;
    let mk_store = |host: bool| -> Result<Store, Error> {
        if host { Store::zeros(&ctx()?, st, &lens, true) } else { Store::zeros(&st.context().clone(), st, &lens, false) }
    };
    let mut x_prev = if quasi { mk_store(opts.prev_on_host)? } else { Store { chunks: vec![], on_host: false } };
    let mut g_prev = if quasi { mk_store(opts.prev_on_host)? } else { Store { chunks: vec![], on_host: false } };
    let mut w_store = match (quasi, opts.w_mode) {
        (true, WMode::Device) => Some(mk_store(false)?),
        (true, WMode::Host) => Some(mk_store(true)?),
        _ => None,
    };
    let mut scratch = Scratch::default();
    let mut host = vec![C32::new(0.0, 0.0); lens.iter().sum()];
    let mut k0 = 0;
    if let Some((c, name)) = ctl {
        if let Some(v) = c.load(name)? {
            k0 = get_usize(&v, "k")?;
            // the step size of the interrupted run (λ_max estimates differ in the last digits between runs)
            alpha = crate::checkpoint::get_f64(&v, "alpha")?;
            *x = value_stack(st, get(&v, "x")?, &lens)?;
            if quasi {
                x_prev.load_value(st, get(&v, "x_prev")?)?;
                g_prev.load_value(st, get(&v, "g_prev")?)?;
            }
            eprintln!("  resuming {name} at iteration {k0} / {iters}");
        }
    }
    if k0 == 0 {
        report(usize::MAX, x, f64::NAN);
    }
    let save = |c: &RunCtl, name: &str, next: usize, x: &Stack, xp: &Store, gp: &Store| -> Result<(), Error> {
        let (xpv, gpv) = if quasi { (xp.to_value(st)?, gp.to_value(st)?) } else { (Value::List(vec![]), Value::List(vec![])) };
        c.save(name, &Value::Dict(vec![("k".into(), Value::Int(next as i64)), ("precond".into(), Value::Str(format!("{precond:?}"))), ("alpha".into(), Value::Float(alpha)), ("x".into(), stack_value(st, x)?), ("x_prev".into(), xpv), ("g_prev".into(), gpv)]))
    };
    let one = Complex64::new(1.0, 0.0);
    let re = |v: f64| Complex64::new(v, 0.0);
    for k in k0..iters {
        let f = obj.gradient(st, x, &mut g)?;
        // scaled gradient α g
        axpby(obj.kernels(), st, 0.0, x, alpha, &mut g)?;
        match precond {
            Precond::None => pg.copy_from(st, &g)?,
            Precond::Circulant => {
                for t in 0..nt {
                    obj.metric(st, t, &g.v[t], &mut pg.v[t], false)?;
                }
            }
            Precond::Zmshr1 | Precond::Zmshr1M if k == 0 => {
                for t in 0..nt {
                    if metric { obj.metric(st, t, &g.v[t], &mut pg.v[t], false)? } else { st.memcpy_dtod(&g.v[t], &mut pg.v[t])? }
                }
            }
            Precond::Zmshr1 | Precond::Zmshr1M => {
                // chunk t: s = x − x_prev, m = g − g_prev, ms = M⁻¹ s, mm = M m (M = I for plain ZMSHR1); buffers
                // [s, m, ms, mm, stage_a, stage_b] of the chunk's length
                let secant = |obj: &mut dyn Objective, sc: &mut Vec<CudaSlice<C32>>, t: usize, xp: &Store, gp: &Store| -> Result<(), Error> {
                    let (head, tail) = sc.split_at_mut(4);
                    let (sa, sb) = tail.split_at_mut(1);
                    let xpt = xp.view(st, t, &mut sa[0])?;
                    let gpt = gp.view(st, t, &mut sb[0])?;
                    st.memcpy_dtod(&x.v[t], &mut head[0])?;
                    obj.kernels().axpby(st, re(-1.0), xpt, one, &mut head[0])?;
                    st.memcpy_dtod(&g.v[t], &mut head[1])?;
                    obj.kernels().axpby(st, re(-1.0), gpt, one, &mut head[1])?;
                    if metric {
                        let (a, b) = head.split_at_mut(2);
                        obj.metric(st, t, &a[0], &mut b[0], true)?;
                        obj.metric(st, t, &a[1], &mut b[1], false)?;
                    }
                    Ok(())
                };
                // pass 1: secant inner products
                let (mut ss, mut sm, mut mm) = (0.0, 0.0, 0.0);
                for t in 0..nt {
                    let sc = scratch.get(st, lens[t], 6)?;
                    secant(obj, sc, t, &x_prev, &g_prev)?;
                    let (s_, m_, ms, mmv) = if metric { (&sc[0], &sc[1], &sc[2], &sc[3]) } else { (&sc[0], &sc[1], &sc[0], &sc[1]) };
                    ss += obj.kernels().dot(st, s_, ms)?.re;
                    sm += obj.kernels().dot(st, s_, m_)?.re;
                    mm += obj.kernels().dot(st, m_, mmv)?.re;
                }
                let (a, tau) = zmshr1_scalars(ss, sm, mm);
                // pass 2: w = (1 − τa) s − τ(1−a) mm, z = (1−a) m + a ms; uv, norms, ⟨w, g⟩; step = τ M g
                let (mut uv, mut ww, mut zz, mut wg) = (0.0, 0.0, 0.0, 0.0);
                for t in 0..nt {
                    let sc = scratch.get(st, lens[t], 6)?;
                    secant(obj, sc, t, &x_prev, &g_prev)?;
                    let (w_, rest) = sc.split_at_mut(1);
                    let (z_, rest) = rest.split_at_mut(1);
                    if metric {
                        obj.kernels().axpby(st, re(-tau * (1.0 - a)), &rest[1], re(1.0 - tau * a), &mut w_[0])?;
                        obj.kernels().axpby(st, re(a), &rest[0], re(1.0 - a), &mut z_[0])?;
                    } else {
                        // plain: w = s − τ (a s + (1−a) m) = (1 − τa) s − τ(1−a) m;  z = a s + (1−a) m
                        obj.kernels().axpby(st, re(a), &w_[0], re(1.0 - a), &mut z_[0])?;
                        obj.kernels().axpby(st, re(-tau), &z_[0], one, &mut w_[0])?;
                    }
                    uv += obj.kernels().dot(st, &w_[0], &z_[0])?.re;
                    ww += obj.kernels().dot(st, &w_[0], &w_[0])?.re;
                    zz += obj.kernels().dot(st, &z_[0], &z_[0])?.re;
                    wg += obj.kernels().dot(st, &w_[0], &g.v[t])?.re;
                    if let Some(ws) = w_store.as_mut() {
                        ws.put(st, t, &w_[0])?;
                    }
                    if metric { obj.metric(st, t, &g.v[t], &mut pg.v[t], false)? } else { st.memcpy_dtod(&g.v[t], &mut pg.v[t])? }
                    obj.kernels().axpby(st, re(0.0), &g.v[t], re(tau), &mut pg.v[t])?;
                }
                let ok = uv > 1e-8 * ww.sqrt() * zz.sqrt();
                if ok {
                    // pass 3: step += (⟨w, g⟩ / ⟨w, z⟩) w
                    let c = wg / uv;
                    for t in 0..nt {
                        let sc = scratch.get(st, lens[t], 6)?;
                        if let Some(ws) = w_store.as_ref() {
                            let w = ws.view(st, t, &mut sc[0])?;
                            obj.kernels().axpby(st, re(c), w, one, &mut pg.v[t])?;
                        } else {
                            secant(obj, sc, t, &x_prev, &g_prev)?;
                            let (w_, rest) = sc.split_at_mut(1);
                            if metric {
                                obj.kernels().axpby(st, re(-tau * (1.0 - a)), &rest[2], re(1.0 - tau * a), &mut w_[0])?;
                            } else {
                                obj.kernels().axpby(st, re(-tau * (1.0 - a)), &rest[0], re(1.0 - tau * a), &mut w_[0])?;
                            }
                            obj.kernels().axpby(st, re(c), &w_[0], one, &mut pg.v[t])?;
                        }
                    }
                }
                if metric && k % 10 == 0 {
                    eprintln!("    zmshr1-M it {k}: a {a:.3} tau {tau:.3e} uv>thr {ok}");
                }
            }
        }
        if quasi {
            for t in 0..nt {
                x_prev.put(st, t, &x.v[t])?;
                g_prev.put(st, t, &g.v[t])?;
            }
        }
        // z = x − P α g, x = D(z)
        axpby(obj.kernels(), st, -1.0, &pg, 1.0, x)?;
        for t in 0..nt {
            let h = st.clone_dtoh(&x.v[t])?;
            host[off[t]..off[t] + lens[t]].copy_from_slice(&h);
        }
        denoise(k, &mut host);
        for t in 0..nt {
            st.memcpy_htod(&host[off[t]..off[t] + lens[t]], &mut x.v[t])?;
        }
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
    }
    Ok(Outcome::Finished)
}
