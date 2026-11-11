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
    /// ZMSHR1 per set (block-diagonal): every set t (frame × encode, or parameter image × frame) has its own secant pair
    /// (s_t, m_t), mixing a_t, scale τ_t and rank-one term. The data term is separable over the sets (H = blockdiag H_t),
    /// so the block-diagonal quasi-Newton matrix is the natural structure: one step scale per set instead of one for the
    /// whole stack (limited by the stiffest set), and one curvature direction per set and iteration instead of one.
    Zmshr1Set,
    /// [`Precond::Zmshr1Set`] in the circulant metric M (per set: P_t = τ_t M_t + rank one)
    Zmshr1MSet,
}

impl Precond {
    /// quasi-Newton (keeps a secant pair)
    pub fn quasi(self) -> bool {
        matches!(self, Precond::Zmshr1 | Precond::Zmshr1M | Precond::Zmshr1Set | Precond::Zmshr1MSet)
    }
    /// uses the circulant metric M
    pub fn metric(self) -> bool {
        matches!(self, Precond::Circulant | Precond::Zmshr1M | Precond::Zmshr1MSet)
    }
    /// one secant pair and scalar set per set (block-diagonal) instead of one for the whole stack
    pub fn per_set(self) -> bool {
        matches!(self, Precond::Zmshr1Set | Precond::Zmshr1MSet)
    }
    /// parse a command-line name: none, circ, zmshr1, zmshr1m, zmshr1s (per set), zmshr1ms (per set, metric)
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "none" => Precond::None,
            "circ" => Precond::Circulant,
            "zmshr1" => Precond::Zmshr1,
            "zmshr1m" => Precond::Zmshr1M,
            "zmshr1s" => Precond::Zmshr1Set,
            "zmshr1ms" => Precond::Zmshr1MSet,
            _ => return None,
        })
    }
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
    /// set → index into `circ` when preconditioners are shared (e.g. one per encode, built from the pooled sampling of
    /// that encode's frames); empty = one per set
    pub circ_map: Vec<usize>,
    /// per-set residual buffers
    res: Vec<CudaSlice<C32>>,
    /// frame-block scratch (block-diagonal preconditioners)
    blk: Option<(CudaSlice<C32>, CudaSlice<C32>)>,
    /// stochastic coil sampling of the data term (None: all coils, as before; see [`CoilSampling`])
    pub sampling: Option<CoilSampling>,
    /// the coil groups of the current iteration (None: all coils) and the iteration
    cur: Option<Vec<(Vec<usize>, f64)>>,
    cur_k: usize,
    iters_total: usize,
    /// samples per set (data length / coils, with all coils)
    m_set: Vec<usize>,
    /// host-resident data (`kstore`), staged per set instead of `ys` (empty then); see [`P2np::from_store`]
    stage: Option<crate::kstage::Stager>,
    /// lagged gradient with the operator's surrogate ([`P2np::with_lagged`]): refresh threshold τ
    lagged: Option<f64>,
    /// anchor of the lagged gradient: (x_a, ∇f(x_a), f(x_a))
    anchor: Option<(Stack, Stack, f64)>,
    /// gradients evaluated (exact, surrogate) since construction
    pub lag_stats: (usize, usize),
    /// (anchor epoch, exact) of the previous and the current lagged gradient: a secant pair is exact only within one
    /// anchor (both gradients from the same quadratic model) or between two exact gradients
    lag_hist: [(u64, bool); 2],
    /// set streaming of host-resident stacks ([`Objective::gradient_host`])
    streamer: Option<crate::hoststack::SetStreamer>,
    /// device bytes the grouped gradient may hold in member residuals at once ([`P2np::gradient_groups_host`]; larger
    /// groups are split into sub-groups, each paying its own grid transforms); 0 (default) = half the free device
    /// memory at the call
    pub group_bytes: usize,
}

/// Stochastic coil sampling for the data term of [`P2np`]: each gradient uses a subset of the coils and stays
/// unbiased (inclusion weights): ∇̃f = Σ_{c < keep} E_cᴴ r_c + w Σ_{c ∈ D} E_cᴴ r_c, D = `draw` of the remaining
/// nc − keep coils, w = 1 / P(c ∈ D). With random reshuffling the remaining coils are partitioned at random into
/// p = ⌈(nc − keep) / draw⌉ groups used in turn (w = p; every coil once per epoch, the epoch mean is the full
/// gradient); otherwise independent draws (w = (nc − keep) / draw).
/// - keep: with an energy-ordered compression (`coils::CoilCompression`) the leading virtual coils carry most of the
///   signal; sampling them would make the variance and the step-size inflation (λ_max of the sampled normal operator,
///   [`P2np::step_norm_sampled`]) large, so they are always used.
/// - hold ≥ 2: a subset is kept for `hold` iterations, so every second gradient pair shares its subset and the ZMSHR1
///   secant pair is exact (for a quadratic data term m = Ñ s); on a new subset the step falls back to the metric step.
/// - full_last: the last iterations use all coils, so the iteration ends at the fixed point of the full problem.
/// The groups are a function of (seed, k): a resumed run draws the same subsets.
#[derive(Clone, Debug)]
pub struct CoilSampling {
    pub keep: usize,
    pub draw: usize,
    pub hold: usize,
    pub full_last: usize,
    pub reshuffle: bool,
    pub seed: u64,
}

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl CoilSampling {
    /// subset block of iteration k (None: all coils)
    pub fn block(&self, nc: usize, k: usize, iters: usize) -> Option<usize> {
        let rest = nc.saturating_sub(self.keep);
        if self.draw == 0 || self.draw >= rest || k + self.full_last >= iters {
            return None;
        }
        Some(k / self.hold.max(1))
    }

    /// coil groups and weights of iteration k of `iters` (None: all coils)
    pub fn groups(&self, nc: usize, k: usize, iters: usize) -> Option<Vec<(Vec<usize>, f64)>> {
        let b = self.block(nc, k, iters)?;
        let rest: Vec<usize> = (self.keep.min(nc)..nc).collect();
        let n = rest.len();
        let perm = |key: u64| -> Vec<usize> {
            let mut v = rest.clone();
            let mut h = mix64(self.seed ^ key.wrapping_mul(0x9E3779B97F4A7C15));
            for i in (1..n).rev() {
                h = mix64(h.wrapping_add(i as u64));
                v.swap(i, (h % (i as u64 + 1)) as usize);
            }
            v
        };
        let (drawn, w) = if self.reshuffle {
            let p = n.div_ceil(self.draw);
            let (e, i) = (b / p, b % p);
            let v = perm(e as u64);
            (v[i * self.draw..((i + 1) * self.draw).min(n)].to_vec(), p as f64)
        } else {
            (perm(b as u64 | (1 << 63))[..self.draw].to_vec(), n as f64 / self.draw as f64)
        };
        let mut g = Vec::new();
        if self.keep > 0 {
            g.push(((0..self.keep.min(nc)).collect(), 1.0));
        }
        g.push((drawn, w));
        Some(g)
    }

    /// fraction of the coil applications of a full gradient that iteration k costs
    pub fn cost(&self, nc: usize, k: usize, iters: usize) -> f64 {
        self.groups(nc, k, iters).map_or(1.0, |g| g.iter().map(|(c, _)| c.len()).sum::<usize>() as f64 / nc as f64)
    }
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
    /// called at the start of iteration k of `iters` (stochastic objectives choose their sample); default: nothing
    fn begin_iteration(&mut self, _k: usize, _iters: usize) {}
    /// whether this iteration's gradient and the previous one come from the same (sampled) objective, i.e. the ZMSHR1
    /// secant pair is exact; default true
    fn curvature_pair_ok(&self) -> bool {
        true
    }
    /// [`gradient`](Self::gradient) on host-resident stacks ([`crate::hoststack`]). Default: upload the whole stack,
    /// compute, download (needs the device memory of two stacks); objectives that stream set by set override it.
    fn gradient_host(&mut self, st: &Arc<CudaStream>, x: &crate::hoststack::HostStack, g: &mut crate::hoststack::HostStack) -> Result<f64, Error> {
        let lens = x.lens.clone();
        let mut xd = Stack::zeros_lens(st, &lens)?;
        for t in 0..lens.len() {
            st.memcpy_htod(x.chunk(t), &mut xd.v[t])?;
        }
        let mut gd = Stack::zeros_lens(st, &lens)?;
        let f = self.gradient(st, &xd, &mut gd)?;
        for t in 0..lens.len() {
            let h = st.clone_dtoh(&gd.v[t])?;
            g.chunk_mut(t).copy_from_slice(&h);
        }
        Ok(f)
    }
    /// [`metric`](Self::metric) of chunk t on host slices (uploaded, applied, downloaded)
    fn metric_host(&mut self, st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inverse: bool) -> Result<(), Error> {
        let rd = st.clone_htod(r)?;
        let mut zd = st.alloc_zeros::<C32>(z.len())?;
        self.metric(st, t, &rd, &mut zd, inverse)?;
        st.memcpy_dtoh(&zd, z)?;
        Ok(())
    }
    /// Grouped gradient on host stacks: chunk g of `x` is the image shared by the chunks (sets) `groups[g]`;
    /// g_g = Σ_{t ∈ G_g} ∇f_t(x_g), G_g = `groups[g]` (the gradient of f(x_t = x_g) with respect to x_g). Returns Some(f), or None
    /// when the objective has no grouped form (callers then replicate the group images into a full stack and sum
    /// [`gradient_host`](Self::gradient_host); same result). Default: None.
    fn gradient_groups_host(&mut self, _st: &Arc<CudaStream>, _groups: &[Vec<usize>], _x: &crate::hoststack::HostStack, _g: &mut crate::hoststack::HostStack) -> Result<Option<f64>, Error> {
        Ok(None)
    }
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
    /// streamed: each set's image goes up and its gradient comes down through two device slots while the GPU works on
    /// the neighbouring set (the device never holds a stack); data resident or staged from the store. Coil-sampled and
    /// lagged gradients take the default (whole-stack) path.
    fn gradient_host(&mut self, st: &Arc<CudaStream>, x: &crate::hoststack::HostStack, g: &mut crate::hoststack::HostStack) -> Result<f64, Error> {
        let uniform = x.lens.iter().all(|&l| l == x.lens[0]);
        if self.cur.is_some() || self.lagged.is_some() || !uniform {
            let lens = x.lens.clone();
            let mut xd = Stack::zeros_lens(st, &lens)?;
            for t in 0..lens.len() {
                st.memcpy_htod(x.chunk(t), &mut xd.v[t])?;
            }
            let mut gd = Stack::zeros_lens(st, &lens)?;
            let f = P2np::gradient(self, st, &xd, &mut gd)?;
            for t in 0..lens.len() {
                let h = st.clone_dtoh(&gd.v[t])?;
                g.chunk_mut(t).copy_from_slice(&h);
            }
            return Ok(f);
        }
        let mut sr = match self.streamer.take() {
            Some(s) => s,
            None => crate::hoststack::SetStreamer::new(&st.context().clone(), st, x.lens[0])?,
        };
        let mut stage = self.stage.take();
        let order: Vec<usize> = (0..self.op.nsets()).collect();
        let mut f = 0.0;
        let op = &mut *self.op;
        let ys = &self.ys;
        let r = sr.for_each(st, x, g, &order, &mut |t, xd, gd| {
            let staged;
            let y: &CudaSlice<C32> = match stage.as_mut() {
                Some(sg) => {
                    staged = sg.upload(st, t, None)?;
                    &staged
                }
                None => ys[t],
            };
            let mut r = unsafe { st.alloc::<C32>(op.len_y(t))? };
            op.forward_residual(st, t, xd, y, &mut r)?;
            f += 0.5 * op.kernels().dot(st, &r, &r)?.re;
            op.adjoint(st, t, &r, gd)?;
            hasty_cuda::pace::pace(st);
            Ok(())
        });
        self.stage = stage;
        self.streamer = Some(sr);
        r?;
        Ok(f)
    }
    /// streamed per group (`SetStreamer` over the group images), data resident or staged per member from the store:
    /// g_g = Σ_{t∈g} E_tᴴ(E_t x_g − y_t) through `SetOp::forward_group` / `adjoint_group` (`SegSets`: one grid FFT per
    /// segment and coil batch for each group instead of one per set). None with coil sampling or lagged gradients
    /// (their per-set paths stay in charge).
    fn gradient_groups_host(&mut self, st: &Arc<CudaStream>, groups: &[Vec<usize>], x: &crate::hoststack::HostStack, g: &mut crate::hoststack::HostStack) -> Result<Option<f64>, Error> {
        if self.cur.is_some() || self.lagged.is_some() {
            return Ok(None);
        }
        P2np::gradient_groups_host(self, st, groups, x, g).map(Some)
    }
    fn begin_iteration(&mut self, k: usize, iters: usize) {
        self.cur_k = k;
        self.iters_total = iters;
        self.cur = match &self.sampling {
            Some(s) => s.groups(self.op.n_coils(), k, iters),
            None => None,
        };
    }
    fn curvature_pair_ok(&self) -> bool {
        if self.lagged.is_some() && self.cur.is_none() {
            let [p, c] = self.lag_hist;
            if p.0 != c.0 && !(p.1 && c.1) {
                return false;
            }
        }
        match &self.sampling {
            None => true,
            Some(s) => {
                let nc = self.op.n_coils();
                let k = self.cur_k;
                k > 0 && s.block(nc, k, self.iters_total) == s.block(nc, k - 1, self.iters_total)
            }
        }
    }
}

impl<'a> P2np<'a> {
    pub fn new(op: &'a mut dyn SetOp, ys: Vec<&'a CudaSlice<C32>>, circ: Vec<Vec<CirculantPrecond>>, st: &Arc<CudaStream>) -> Result<Self, Error> {
        assert_eq!(ys.len(), op.nsets());
        // one per set, or fewer shared ones (then `with_circ_map` assigns them)
        assert!(circ.is_empty() || circ.len() <= op.nsets());
        // no persistent residual buffers: each set's residual is formed in a stream-ordered temporary by the fused
        // `SetOp::forward_residual` (one set in flight instead of a second copy of all data)
        let res = vec![];
        let blk = match circ.iter().position(|c| c.len() > 1) {
            Some(t) => {
                let n = op.len_x(t) / circ[t].len();
                Some((st.alloc_zeros(n)?, st.alloc_zeros(n)?))
            }
            None => None,
        };
        Ok(Self { op, ys, circ, res, blk, sampling: None, cur: None, cur_k: 0, iters_total: 0, m_set: vec![], stage: None, lagged: None, anchor: None, lag_stats: (0, 0), lag_hist: [(0, false); 2], streamer: None, circ_map: vec![], group_bytes: 0 })
    }

    /// Data on the host: each gradient stages y_t from the store set by set (the next set decoded while the GPU works
    /// on the current one), so the device holds no data and no per-set residuals — only the set in flight. Same
    /// iterates as [`new`](Self::new) with resident data (up to float32 summation order).
    pub fn from_store(op: &'a mut dyn SetOp, stage: crate::kstage::Stager, circ: Vec<Vec<CirculantPrecond>>, st: &Arc<CudaStream>) -> Result<Self, Error> {
        assert_eq!(stage.store().n_sets(), op.nsets());
        for t in 0..op.nsets() {
            assert_eq!(stage.store().len(t) * stage.store().nvc(), op.len_y(t), "k-space store set {t} does not match the operator");
        }
        let mut p = Self::with_res(op, vec![], circ, st, false)?;
        p.stage = Some(stage);
        Ok(p)
    }

    fn with_res(op: &'a mut dyn SetOp, ys: Vec<&'a CudaSlice<C32>>, circ: Vec<Vec<CirculantPrecond>>, st: &Arc<CudaStream>, res: bool) -> Result<Self, Error> {
        // one per set, or fewer shared ones (then `with_circ_map` assigns them)
        assert!(circ.is_empty() || circ.len() <= op.nsets());
        let res = if res { (0..op.nsets()).map(|t| st.alloc_zeros(op.len_y(t))).collect::<Result<_, _>>()? } else { vec![] };
        let blk = match circ.iter().position(|c| c.len() > 1) {
            Some(t) => {
                let n = op.len_x(t) / circ[t].len();
                Some((st.alloc_zeros(n)?, st.alloc_zeros(n)?))
            }
            None => None,
        };
        Ok(Self { op, ys, circ, res, blk, sampling: None, cur: None, cur_k: 0, iters_total: 0, m_set: vec![], stage: None, lagged: None, anchor: None, lag_stats: (0, 0), lag_hist: [(0, false); 2], streamer: None, circ_map: vec![], group_bytes: 0 })
    }

    /// set-streaming statistics of host-resident runs: (sets, bytes up, bytes down)
    pub fn streamer_stats(&self) -> Option<(usize, usize, usize)> {
        self.streamer.as_ref().map(|s| s.stats)
    }

    /// staging statistics of a store-backed problem
    pub fn stage_stats(&self) -> Option<crate::kstage::StageStats> {
        self.stage.as_ref().map(|s| s.stats)
    }

    /// F v → r and Fᴴ r → out for set t through the persistent residual buffer (resident data) or a temporary
    fn normal_full(&mut self, st: &Arc<CudaStream>, t: usize, v: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        if self.res.is_empty() {
            let mut r = st.alloc_zeros::<C32>(self.op.len_y(t))?;
            self.op.forward(st, t, v, &mut r)?;
            return self.op.adjoint(st, t, &r, out);
        }
        self.op.forward(st, t, v, &mut self.res[t])?;
        self.op.adjoint(st, t, &self.res[t], out)
    }

    /// enable stochastic coil sampling (the operator must support `SetOp::select_coils`)
    /// Lagged gradients with refresh threshold `tau` ([`P2np::gradient`]; ≈ 0.05–0.1): most iterations apply only
    /// the operator's surrogate (no data access). Needs an operator with a surrogate (`SegSets::with_surrogate`);
    /// coil-sampled iterations stay exact. Costs two more image stacks (the anchor and its gradient).
    pub fn with_lagged(mut self, tau: f64) -> Self {
        assert!(tau > 0.0, "lagged gradient: tau {tau} must be > 0");
        assert!(self.op.use_surrogate(false), "lagged gradient: the operator has no surrogate");
        self.lagged = Some(tau);
        self
    }

    pub fn with_sampling(mut self, s: CoilSampling) -> Self {
        let nc = self.op.n_coils();
        assert!(nc > 0, "coil sampling: the operator does not report its coils");
        self.m_set = (0..self.op.nsets()).map(|t| self.op.len_y(t) / nc).collect();
        self.sampling = Some(s);
        self
    }

    /// N v (or Σ_g w_g N_G v over coil groups) for set t
    fn normal_apply(&mut self, st: &Arc<CudaStream>, t: usize, v: &CudaSlice<C32>, out: &mut CudaSlice<C32>, groups: Option<&[(Vec<usize>, f64)]>) -> Result<(), Error> {
        let Some(groups) = groups else {
            return self.normal_full(st, t, v, out);
        };
        let m = self.m_set[t];
        let mut tmp = st.alloc_zeros::<C32>(out.len())?;
        for (gi, (coils, w)) in groups.iter().enumerate() {
            assert!(self.op.select_coils(st, Some(coils))?, "coil sampling: operator without coil subsets");
            let mut r = st.alloc_zeros::<C32>(coils.len() * m)?;
            self.op.forward(st, t, v, &mut r)?;
            if gi == 0 {
                self.op.adjoint(st, t, &r, out)?;
                self.op.kernels().axpby(st, Complex64::new(0.0, 0.0), v, Complex64::new(*w, 0.0), out)?;
            } else {
                self.op.adjoint(st, t, &r, &mut tmp)?;
                self.op.kernels().axpby(st, Complex64::new(*w, 0.0), &tmp, Complex64::new(1.0, 0.0), out)?;
            }
        }
        self.op.select_coils(st, None)?;
        Ok(())
    }

    /// Share circulant preconditioners between sets: `map[t]` is the index into `circ` of set t's preconditioner (e.g.
    /// the encode of the set, with one circulant per encode built from all of its frames' samples).
    pub fn with_circ_map(mut self, map: Vec<usize>) -> Self {
        assert_eq!(map.len(), self.op.nsets(), "circulant map: one entry per set");
        assert!(map.iter().all(|&i| i < self.circ.len()), "circulant map: index out of range");
        self.circ_map = map;
        self
    }

    /// [`with_circ_map`](Self::with_circ_map) for a non-empty map, unchanged otherwise
    pub fn with_circ_map_opt(self, map: Vec<usize>) -> Self {
        if map.is_empty() { self } else { self.with_circ_map(map) }
    }

    /// z = M_t r (or M_t⁻¹ r), M_t block-diagonal over the set's frames
    fn precond(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        assert!(self.circ_map.len() == self.op.nsets() || self.circ.len() == self.op.nsets(), "shared circulants need with_circ_map");
        let t = if self.circ_map.is_empty() { t } else { self.circ_map[t] };
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

    /// g = ∇f(x) (with coil sampling: the unbiased estimate of the current iteration); returns f(x) (its estimate)
    pub fn gradient(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error> {
        if let Some(groups) = self.cur.clone() {
            return self.gradient_sampled(st, x, g, &groups);
        }
        if let Some(tau) = self.lagged {
            return self.gradient_lagged(st, x, g, tau);
        }
        self.gradient_exact(st, x, g)
    }

    /// Lagged gradient (control variate): with an anchor x_a where the exact gradient is known,
    ///   ∇̃f(x) = Ñ (x − x_a) + ∇f(x_a),   f̃(x) = f(x_a) + Re⟨∇f(x_a), x − x_a⟩ + ½‖Ẽ (x − x_a)‖²,
    /// Ñ = Ẽᴴ Ẽ the operator's surrogate ([`SetOp::use_surrogate`]; it reads no data). The anchor is refreshed by an
    /// exact gradient when ‖x − x_a‖ > τ ‖x‖, so the error ‖(N − Ñ)(x − x_a)‖ ≤ ‖N − Ñ‖ τ ‖x‖ vanishes at the
    /// fixed point. research/mathematics_of_paper (§ fast gradients), research/implementation_results.md §5.
    fn gradient_lagged(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack, tau: f64) -> Result<f64, Error> {
        let cheap = match &self.anchor {
            Some((xa, _, _)) => {
                let k = self.op.kernels();
                let (xx, aa, xa_) = (rdot(k, st, x, x)?, rdot(k, st, xa, xa)?, rdot(k, st, x, xa)?);
                (xx - 2.0 * xa_ + aa).max(0.0) <= tau * tau * xx
            }
            None => false,
        };
        self.lag_hist[0] = self.lag_hist[1];
        let epoch = self.lag_hist[1].0 + u64::from(!cheap);
        self.lag_hist[1] = (epoch, !cheap);
        if !cheap {
            let f = self.gradient_exact(st, x, g)?;
            match &mut self.anchor {
                Some((xa, ga, fa)) => {
                    xa.copy_from(st, x)?;
                    ga.copy_from(st, g)?;
                    *fa = f;
                }
                None => {
                    let lens = self.lens();
                    let (mut xa, mut ga) = (Stack::zeros_lens(st, &lens)?, Stack::zeros_lens(st, &lens)?);
                    xa.copy_from(st, x)?;
                    ga.copy_from(st, g)?;
                    self.anchor = Some((xa, ga, f));
                }
            }
            self.lag_stats.0 += 1;
            return Ok(f);
        }
        assert!(self.op.use_surrogate(true), "lagged gradient: the operator has no surrogate");
        let one = Complex64::new(1.0, 0.0);
        let op = &mut *self.op;
        let (xa, ga, fa) = self.anchor.as_ref().unwrap();
        let r = (|| -> Result<f64, Error> {
            let mut f = *fa;
            for t in 0..op.nsets() {
                let mut d = st.alloc_zeros::<C32>(x.v[t].len())?;
                st.memcpy_dtod(&x.v[t], &mut d)?;
                op.kernels().axpby(st, Complex64::new(-1.0, 0.0), &xa.v[t], one, &mut d)?;
                f += op.kernels().dot(st, &ga.v[t], &d)?.re;
                let mut buf = st.alloc_zeros::<C32>(op.len_y(t))?;
                op.forward(st, t, &d, &mut buf)?;
                f += 0.5 * op.kernels().dot(st, &buf, &buf)?.re;
                op.adjoint(st, t, &buf, &mut g.v[t])?;
                op.kernels().axpby(st, one, &ga.v[t], one, &mut g.v[t])?;
                hasty_cuda::pace::pace(st);
            }
            Ok(f)
        })();
        op.use_surrogate(false);
        self.lag_stats.1 += 1;
        r
    }

    /// The exact grouped gradient (all coils) on host stacks: chunk g of `x` / `g` is the image of group `groups[g]`
    /// (sets sharing it), g_g = Σ_{t∈g} E_tᴴ(E_t x_g − y_t); returns f = ½ Σ_g Σ_{t∈g} ‖E_t x_g − y_t‖². Per group:
    /// residuals of all members by one `SetOp::forward_group` (resident data: fused, `forward_residual_group`; staged
    /// data: the forward first, then each member's y_t subtracted as it is decoded, so the device holds one copy of the
    /// group's data), then one `adjoint_group`. A group whose residuals exceed [`group_bytes`](Self::group_bytes) is
    /// split into consecutive sub-groups (results summed). Coil sampling and lagged gradients are ignored here
    /// (exact gradient).
    pub fn gradient_groups_host(&mut self, st: &Arc<CudaStream>, groups: &[Vec<usize>], x: &crate::hoststack::HostStack, g: &mut crate::hoststack::HostStack) -> Result<f64, Error> {
        let ng = groups.len();
        assert!(ng > 0 && x.nsets() == ng && g.nsets() == ng, "grouped gradient: {ng} groups, {} / {} chunks", x.nsets(), g.nsets());
        let n = x.lens[0];
        assert!(x.lens.iter().chain(&g.lens).all(|&l| l == n), "grouped gradient: every group image must have {n} elements");
        for (gi, m) in groups.iter().enumerate() {
            assert!(!m.is_empty(), "grouped gradient: group {gi} is empty");
            assert!(m.iter().all(|&t| t < self.op.nsets() && self.op.len_x(t) == n), "grouped gradient: group {gi} has a set outside the stack or of another image size");
        }
        // member residual budget per sub-group
        let budget = if self.group_bytes > 0 {
            self.group_bytes
        } else {
            st.context().mem_get_info().map(|(free, _)| free / 2).unwrap_or(1 << 30)
        };
        let mut sr = match self.streamer.take() {
            Some(s) if s.device_bytes() == 4 * n * 8 => s,
            _ => crate::hoststack::SetStreamer::new(&st.context().clone(), st, n)?,
        };
        let mut stage = self.stage.take();
        let order: Vec<usize> = (0..ng).collect();
        let mut f = 0.0;
        let op = &mut *self.op;
        let ys = &self.ys;
        let one = Complex64::new(1.0, 0.0);
        let r = sr.for_each(st, x, g, &order, &mut |gi, xd, gd| {
            // consecutive sub-groups within the residual budget (at least one member each)
            let mem = &groups[gi];
            let mut subs: Vec<&[usize]> = Vec::new();
            let (mut a, mut bytes) = (0, 0usize);
            for (j, &t) in mem.iter().enumerate() {
                let b = op.len_y(t) * 8;
                if j > a && bytes + b > budget {
                    subs.push(&mem[a..j]);
                    a = j;
                    bytes = 0;
                }
                bytes += b;
            }
            subs.push(&mem[a..]);
            let mut tmp: Option<CudaSlice<C32>> = None;
            for (k, sub) in subs.iter().enumerate() {
                // every entry is written by the group forward: no zeroing
                let mut rs: Vec<CudaSlice<C32>> = sub.iter().map(|&t| unsafe { st.alloc::<C32>(op.len_y(t)) }).collect::<Result<_, _>>()?;
                match stage.as_mut() {
                    Some(sg) => {
                        {
                            let mut refs: Vec<&mut CudaSlice<C32>> = rs.iter_mut().collect();
                            op.forward_group(st, sub, xd, &mut refs)?;
                        }
                        let pos: std::collections::HashMap<usize, usize> = sub.iter().enumerate().map(|(j, &t)| (t, j)).collect();
                        sg.for_each(st, sub, None, &mut |t, y| {
                            let r = &mut rs[pos[&t]];
                            op.kernels().axpby(st, Complex64::new(-1.0, 0.0), y, one, r)?;
                            f += 0.5 * op.kernels().dot(st, r, r)?.re;
                            Ok(())
                        })?;
                    }
                    None => {
                        let d: Vec<&CudaSlice<C32>> = sub.iter().map(|&t| ys[t]).collect();
                        {
                            let mut refs: Vec<&mut CudaSlice<C32>> = rs.iter_mut().collect();
                            op.forward_residual_group(st, sub, xd, &d, &mut refs)?;
                        }
                        for r in &rs {
                            f += 0.5 * op.kernels().dot(st, r, r)?.re;
                        }
                    }
                }
                let refs: Vec<&CudaSlice<C32>> = rs.iter().collect();
                if k == 0 {
                    op.adjoint_group(st, sub, &refs, gd)?;
                } else {
                    if tmp.is_none() {
                        tmp = Some(st.alloc_zeros::<C32>(n)?);
                    }
                    let tb = tmp.as_mut().unwrap();
                    op.adjoint_group(st, sub, &refs, tb)?;
                    op.kernels().axpby(st, one, tb, one, gd)?;
                }
                hasty_cuda::pace::pace(st);
            }
            Ok(())
        });
        self.stage = stage;
        self.streamer = Some(sr);
        r?;
        Ok(f)
    }

    /// the exact gradient (all coils), resident or staged data
    fn gradient_exact(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error> {
        if let Some(mut sg) = self.stage.take() {
            let order: Vec<usize> = (0..self.op.nsets()).collect();
            let mut f = 0.0;
            let op = &mut *self.op;
            let r = sg.for_each(st, &order, None, &mut |t, y| {
                // every entry is written by the fused residual: no zeroing
                let mut buf = unsafe { st.alloc::<C32>(op.len_y(t))? };
                op.forward_residual(st, t, &x.v[t], y, &mut buf)?;
                f += 0.5 * op.kernels().dot(st, &buf, &buf)?.re;
                op.adjoint(st, t, &buf, &mut g.v[t])?;
                hasty_cuda::pace::pace(st);
                Ok(())
            });
            self.stage = Some(sg);
            r?;
            return Ok(f);
        }
        let mut f = 0.0;
        for t in 0..self.ys.len() {
            let buf = &mut unsafe { st.alloc::<C32>(self.op.len_y(t))? };
            self.op.forward_residual(st, t, &x.v[t], self.ys[t], buf)?;
            f += 0.5 * self.op.kernels().dot(st, buf, buf)?.re;
            self.op.adjoint(st, t, buf, &mut g.v[t])?;
            // GPU pacing point per set (HASTY_GPU_DUTY / HASTY_GPU_WATTS; a no-op unless configured)
            hasty_cuda::pace::pace(st);
        }
        Ok(f)
    }

    fn gradient_sampled(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack, groups: &[(Vec<usize>, f64)]) -> Result<f64, Error> {
        let mut f = 0.0;
        let one = Complex64::new(1.0, 0.0);
        if let Some(mut sg) = self.stage.take() {
            // only the group's coil rows leave the host
            let order: Vec<usize> = (0..self.op.nsets()).collect();
            let mut res = Ok(());
            for (gi, (coils, w)) in groups.iter().enumerate() {
                assert!(self.op.select_coils(st, Some(coils))?, "coil sampling: operator without coil subsets");
                let op = &mut *self.op;
                let m_set = &self.m_set;
                res = sg.for_each(st, &order, Some(coils), &mut |t, ysub| {
                    let mut r = unsafe { st.alloc::<C32>(coils.len() * m_set[t])? };
                    op.forward_residual(st, t, &x.v[t], ysub, &mut r)?;
                    f += w * 0.5 * op.kernels().dot(st, &r, &r)?.re;
                    if gi == 0 {
                        op.adjoint(st, t, &r, &mut g.v[t])?;
                        if *w != 1.0 {
                            op.kernels().axpby(st, Complex64::new(0.0, 0.0), &x.v[t], Complex64::new(*w, 0.0), &mut g.v[t])?;
                        }
                    } else {
                        let mut tmp = st.alloc_zeros::<C32>(g.v[t].len())?;
                        op.adjoint(st, t, &r, &mut tmp)?;
                        op.kernels().axpby(st, Complex64::new(*w, 0.0), &tmp, one, &mut g.v[t])?;
                    }
                    hasty_cuda::pace::pace(st);
                    Ok(())
                });
                if res.is_err() {
                    break;
                }
            }
            self.stage = Some(sg);
            self.op.select_coils(st, None)?;
            res?;
            return Ok(f);
        }
        for (gi, (coils, w)) in groups.iter().enumerate() {
            assert!(self.op.select_coils(st, Some(coils))?, "coil sampling: operator without coil subsets");
            for t in 0..self.ys.len() {
                let m = self.m_set[t];
                // y restricted to the group's coils (rows of the coil-major data), then r = E_G x − y_G
                let mut ysub = unsafe { st.alloc::<C32>(coils.len() * m)? };
                for (i, &c) in coils.iter().enumerate() {
                    st.memcpy_dtod(&self.ys[t].slice(c * m..(c + 1) * m), &mut ysub.slice_mut(i * m..(i + 1) * m))?;
                }
                let mut r = unsafe { st.alloc::<C32>(coils.len() * m)? };
                self.op.forward_residual(st, t, &x.v[t], &ysub, &mut r)?;
                drop(ysub);
                f += w * 0.5 * self.op.kernels().dot(st, &r, &r)?.re;
                if gi == 0 {
                    self.op.adjoint(st, t, &r, &mut g.v[t])?;
                    if *w != 1.0 {
                        self.op.kernels().axpby(st, Complex64::new(0.0, 0.0), &x.v[t], Complex64::new(*w, 0.0), &mut g.v[t])?;
                    }
                } else {
                    let mut tmp = st.alloc_zeros::<C32>(g.v[t].len())?;
                    self.op.adjoint(st, t, &r, &mut tmp)?;
                    self.op.kernels().axpby(st, Complex64::new(*w, 0.0), &tmp, one, &mut g.v[t])?;
                }
                hasty_cuda::pace::pace(st);
            }
        }
        self.op.select_coils(st, None)?;
        Ok(f)
    }

    /// λ_max of the sampled normal operator Σ_g w_g N_G (or of M Σ_g w_g N_G) by power iteration on set 0, maximised
    /// over `draws` subsets of the sampling (the step must be stable for every subset: α = 1 / max(λ_full, this))
    pub fn step_norm_sampled(&mut self, st: &Arc<CudaStream>, with_circ: bool, iters: usize, draws: usize) -> Result<f64, Error> {
        let s = self.sampling.clone().expect("step_norm_sampled without sampling");
        let nc = self.op.n_coils();
        let mut best: f64 = 0.0;
        for d in 0..draws {
            // iteration d · hold of a long run: draws d = 0, 1, … are distinct subsets
            let Some(groups) = s.groups(nc, d * s.hold.max(1), usize::MAX) else { continue };
            let t = 0;
            let nx = self.op.len_x(t);
            let host: Vec<C32> = (0..nx).map(|i| C32::new(1.0 + 0.1 * (i % 7) as f32, 0.05 * (i % 5) as f32)).collect();
            let mut v = st.clone_htod(&host)?;
            let mut w = st.alloc_zeros::<C32>(nx)?;
            let mut u = st.alloc_zeros::<C32>(nx)?;
            let mut lam = 0.0;
            for _ in 0..iters {
                let n = self.op.kernels().dot(st, &v, &v)?.re.sqrt();
                self.op.kernels().axpby(st, Complex64::new(0.0, 0.0), &w, Complex64::new(1.0 / n, 0.0), &mut v)?;
                self.normal_apply(st, t, &v, &mut w, Some(&groups))?;
                if with_circ {
                    self.precond(st, t, &w, &mut u, false)?;
                    std::mem::swap(&mut w, &mut u);
                    lam = self.op.kernels().dot(st, &w, &w)?.re.sqrt();
                } else {
                    lam = self.op.kernels().dot(st, &v, &w)?.re;
                }
                std::mem::swap(&mut v, &mut w);
            }
            best = best.max(lam);
        }
        Ok(best)
    }

    /// λ_max of N (or of M N) by power iteration on set 0 and the middle set
    pub fn step_norm(&mut self, st: &Arc<CudaStream>, with_circ: bool, iters: usize) -> Result<f64, Error> {
        let mut best: f64 = 0.0;
        let idx: Vec<usize> = { let mut v = vec![0, self.op.nsets() / 2]; v.dedup(); v };
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
                self.normal_full(st, t, &v, &mut w)?;
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
        // a new run (or a resumed one) starts with an exact gradient
        self.anchor = None;
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
pub fn zmshr1_scalars(ss: f64, sm: f64, mm: f64) -> (f64, f64) {
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

/// [`zmshr1_scalars`] guarded for one set of a per-set preconditioner: a set whose secant vanishes (unchanged iterate,
/// or no finite curvature information) gets the plain scaled-gradient step (a = 1, τ = 1, no rank-one term)
pub fn zmshr1_scalars_guarded(ss: f64, sm: f64, mm: f64) -> (f64, f64) {
    if !(ss > 0.0) || !ss.is_finite() || !sm.is_finite() || !mm.is_finite() {
        return (1.0, 1.0);
    }
    let (a, tau) = zmshr1_scalars(ss, sm, mm);
    if tau.is_finite() && tau > 0.0 { (a, tau) } else { (1.0, 1.0) }
}

/// The (a, τ) of every set: per set for a per-set preconditioner, else the global pair from the summed inner products
pub fn zmshr1_scalars_sets(d: &[[f64; 3]], per_set: bool) -> Vec<(f64, f64)> {
    if per_set {
        d.iter().map(|v| zmshr1_scalars_guarded(v[0], v[1], v[2])).collect()
    } else {
        let s = d.iter().fold([0.0; 3], |a, v| [a[0] + v[0], a[1] + v[1], a[2] + v[2]]);
        vec![zmshr1_scalars(s[0], s[1], s[2]); d.len()]
    }
}

/// The rank-one coefficients c_t = ⟨w, g⟩ / ⟨w, z⟩ of every set from the per-set sums [uv, ww, zz, wg] (per set, or
/// from the summed values for the global preconditioner), 0 where the safeguard ⟨w, z⟩ ≤ 10⁻⁸‖w‖‖z‖ fires
pub fn zmshr1_rank_one_sets(d: &[[f64; 4]], per_set: bool) -> Vec<f64> {
    let c = |v: &[f64; 4]| if v[0] > 1e-8 * v[1].sqrt() * v[2].sqrt() { v[3] / v[0] } else { 0.0 };
    if per_set {
        d.iter().map(c).collect()
    } else {
        let s = d.iter().fold([0.0; 4], |a, v| [a[0] + v[0], a[1] + v[1], a[2] + v[2], a[3] + v[3]]);
        vec![c(&s); d.len()]
    }
}

/// Wall time of the phases of one solver iteration (gradient = data term on the GPU(s), step = preconditioned update,
/// prior = denoiser / regulariser incl. host↔device copies, report = metrics). Logged per iteration and appended to
/// `<run dir>/timing_<name>.csv`: it shows which phase limits scaling to more GPUs.
pub(crate) struct IterTimer {
    t: std::time::Instant,
    pub(crate) secs: [f64; 4],
}

impl IterTimer {
    pub(crate) const PHASES: [&'static str; 4] = ["gradient", "step", "prior", "report"];
    pub(crate) fn start() -> Self {
        Self { t: std::time::Instant::now(), secs: [0.0; 4] }
    }
    /// close phase `i` (after synchronising `st`, so asynchronous GPU work is attributed to it)
    pub(crate) fn lap(&mut self, st: &Arc<CudaStream>, i: usize) {
        let _ = st.synchronize();
        let now = std::time::Instant::now();
        self.secs[i] += (now - self.t).as_secs_f64();
        self.t = now;
    }
    pub(crate) fn log(&self, k: usize, ctl: Option<(&RunCtl, &str)>) {
        let s = &self.secs;
        eprintln!("    time it {k}: gradient {:.1}s, step {:.1}s, prior {:.1}s, report {:.1}s", s[0], s[1], s[2], s[3]);
        if let Some((c, name)) = ctl {
            use std::io::Write;
            let path = c.dir().join(format!("timing_{name}.csv"));
            let new = !path.exists();
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                if new {
                    let _ = writeln!(f, "iter,{}", Self::PHASES.join(","));
                }
                let _ = writeln!(f, "{k},{:.3},{:.3},{:.3},{:.3}", s[0], s[1], s[2], s[3]);
            }
        }
    }
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
    let quasi = precond.quasi();
    // ZMSHR1 in the circulant metric (the plain circulant step has its own arm)
    let metric = quasi && precond.metric();
    let per_set = precond.per_set();
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
        let mut tm = IterTimer::start();
        obj.begin_iteration(k, iters);
        let f = obj.gradient(st, x, &mut g)?;
        tm.lap(st, 0);
        // scaled gradient α g
        axpby(obj.kernels(), st, 0.0, x, alpha, &mut g)?;
        match precond {
            Precond::None => pg.copy_from(st, &g)?,
            Precond::Circulant => {
                for t in 0..nt {
                    obj.metric(st, t, &g.v[t], &mut pg.v[t], false)?;
                }
            }
            // first iteration, or a new coil subset (stochastic objectives): no exact secant pair → the metric step
            Precond::Zmshr1 | Precond::Zmshr1M | Precond::Zmshr1Set | Precond::Zmshr1MSet if k == 0 || !obj.curvature_pair_ok() => {
                for t in 0..nt {
                    if metric { obj.metric(st, t, &g.v[t], &mut pg.v[t], false)? } else { st.memcpy_dtod(&g.v[t], &mut pg.v[t])? }
                }
            }
            Precond::Zmshr1 | Precond::Zmshr1M | Precond::Zmshr1Set | Precond::Zmshr1MSet => {
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
                // pass 1: secant inner products per set [⟨s, M⁻¹s⟩, ⟨s, m⟩, ⟨m, M m⟩]
                let mut d1 = vec![[0.0f64; 3]; nt];
                for t in 0..nt {
                    let sc = scratch.get(st, lens[t], 6)?;
                    secant(obj, sc, t, &x_prev, &g_prev)?;
                    let (s_, m_, ms, mmv) = if metric { (&sc[0], &sc[1], &sc[2], &sc[3]) } else { (&sc[0], &sc[1], &sc[0], &sc[1]) };
                    d1[t] = [obj.kernels().dot(st, s_, ms)?.re, obj.kernels().dot(st, s_, m_)?.re, obj.kernels().dot(st, m_, mmv)?.re];
                }
                // (a, τ) per set, or the global pair from the summed products (identical for every set)
                let at = zmshr1_scalars_sets(&d1, per_set);
                // pass 2: w = (1 − τa) s − τ(1−a) mm, z = (1−a) m + a ms; uv, norms, ⟨w, g⟩ per set; step = τ M g
                let mut d2 = vec![[0.0f64; 4]; nt];
                for t in 0..nt {
                    let (a, tau) = at[t];
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
                    d2[t] = [
                        obj.kernels().dot(st, &w_[0], &z_[0])?.re,
                        obj.kernels().dot(st, &w_[0], &w_[0])?.re,
                        obj.kernels().dot(st, &z_[0], &z_[0])?.re,
                        obj.kernels().dot(st, &w_[0], &g.v[t])?.re,
                    ];
                    if let Some(ws) = w_store.as_mut() {
                        ws.put(st, t, &w_[0])?;
                    }
                    if metric { obj.metric(st, t, &g.v[t], &mut pg.v[t], false)? } else { st.memcpy_dtod(&g.v[t], &mut pg.v[t])? }
                    obj.kernels().axpby(st, re(0.0), &g.v[t], re(tau), &mut pg.v[t])?;
                }
                // rank-one coefficients c_t = ⟨w, g⟩ / ⟨w, z⟩ (0 where the safeguard fires)
                let ct = zmshr1_rank_one_sets(&d2, per_set);
                let ok = ct.iter().any(|&c| c != 0.0);
                if ok {
                    // pass 3: step += c_t w
                    for t in 0..nt {
                        let (a, tau) = at[t];
                        let c = ct[t];
                        if c == 0.0 {
                            continue;
                        }
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
                if k % 10 == 0 && (metric || per_set) {
                    let taus: Vec<f64> = at.iter().map(|p| p.1).collect();
                    let (lo, hi) = taus.iter().fold((f64::INFINITY, 0.0f64), |(l, h), &v| (l.min(v), h.max(v)));
                    let mixed = at.iter().filter(|p| p.0 > 0.0).count();
                    eprintln!("    zmshr1{} it {k}: tau {lo:.3e}..{hi:.3e}, a > 0 in {mixed}/{nt} sets, rank-one in {}/{nt}", if metric { "-M" } else { "" }, ct.iter().filter(|&&c| c != 0.0).count());
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
        tm.lap(st, 1);
        for t in 0..nt {
            let h = st.clone_dtoh(&x.v[t])?;
            host[off[t]..off[t] + lens[t]].copy_from_slice(&h);
        }
        denoise(k, &mut host);
        for t in 0..nt {
            st.memcpy_htod(&host[off[t]..off[t] + lens[t]], &mut x.v[t])?;
        }
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
    }
    Ok(Outcome::Finished)
}
