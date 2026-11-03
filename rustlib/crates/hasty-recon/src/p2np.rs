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

use crate::gpu::Kernels;
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

fn rdot(op: &dyn SetOp, st: &Arc<CudaStream>, a: &Stack, b: &Stack) -> Result<f64, Error> {
    let mut s = 0.0;
    for (x, y) in a.v.iter().zip(&b.v) {
        s += op.kernels().dot(st, x, y)?.re;
    }
    Ok(s)
}
/// b = α a + β b
fn axpby(k: &Kernels, st: &Arc<CudaStream>, alpha: f64, a: &Stack, beta: f64, b: &mut Stack) -> Result<(), Error> {
    for (x, y) in a.v.iter().zip(b.v.iter_mut()) {
        k.axpby(st, Complex64::new(alpha, 0.0), x, Complex64::new(beta, 0.0), y)?;
    }
    Ok(())
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
        let nt = x.v.len();
        let lens = self.lens();
        let off: Vec<usize> = lens.iter().scan(0, |a, &n| { let o = *a; *a += n; Some(o) }).collect();
        let mut g = Stack::zeros_lens(st, &lens)?;
        let mut pg = Stack::zeros_lens(st, &lens)?;
        let mut x_prev = Stack::zeros_lens(st, &lens)?;
        let mut g_prev = Stack::zeros_lens(st, &lens)?;
        let mut sv = Stack::zeros_lens(st, &lens)?;
        let mut mv = Stack::zeros_lens(st, &lens)?;
        let mut host = vec![C32::new(0.0, 0.0); lens.iter().sum()];
        let need_m = precond == Precond::Zmshr1M;
        let mk = || if need_m { Stack::zeros_lens(st, &lens) } else { Stack::zeros(st, 0, 0) };
        let (mut minv_s, mut mm_v, mut mg, mut wv) = (mk()?, mk()?, mk()?, mk()?);
        report(usize::MAX, x, f64::NAN);
        for k in 0..iters {
            let f = self.gradient(st, x, &mut g)?;
            // scaled gradient α g
            axpby(self.op.kernels(), st, 0.0, x, alpha, &mut g)?;
            match precond {
                Precond::None => pg.copy_from(st, &g)?,
                Precond::Circulant => {
                    for t in 0..nt {
                        self.precond(st, t, &g.v[t], &mut pg.v[t], false)?;
                    }
                }
                Precond::Zmshr1M => {}
                Precond::Zmshr1 => {
                    if k == 0 {
                        pg.copy_from(st, &g)?;
                    } else {
                        // s = x − x_prev, m = g − g_prev
                        sv.copy_from(st, x)?;
                        axpby(self.op.kernels(), st, -1.0, &x_prev, 1.0, &mut sv)?;
                        mv.copy_from(st, &g)?;
                        axpby(self.op.kernels(), st, -1.0, &g_prev, 1.0, &mut mv)?;
                        let ss = rdot(&*self.op, st, &sv, &sv)?;
                        let sm = rdot(&*self.op, st, &sv, &mv)?;
                        let mm = rdot(&*self.op, st, &mv, &mv)?;
                        let (th1, th2, delta) = (2e-6, 200.0, 1e-8);
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
                        // v = a s + (1−a) m (into mv)
                        axpby(self.op.kernels(), st, a, &sv, 1.0 - a, &mut mv)?;
                        let s_v = rdot(&*self.op, st, &sv, &mv)?;
                        let vv = rdot(&*self.op, st, &mv, &mv)?;
                        let tau = ss / s_v - ((ss / s_v).powi(2) - ss / vv).max(0.0).sqrt();
                        // u = s − τ v (into sv)
                        axpby(self.op.kernels(), st, -tau, &mv, 1.0, &mut sv)?;
                        let uv = rdot(&*self.op, st, &sv, &mv)?;
                        let un = rdot(&*self.op, st, &sv, &sv)?.sqrt();
                        let vn = vv.sqrt();
                        pg.copy_from(st, &g)?;
                        axpby(self.op.kernels(), st, 0.0, &g, tau, &mut pg)?;
                        if uv > delta * un * vn {
                            let c = rdot(&*self.op, st, &sv, &g)? / uv;
                            axpby(self.op.kernels(), st, c, &sv, 1.0, &mut pg)?;
                        }
                    }
                }
            }
            if precond == Precond::Zmshr1M {
                for t in 0..nt {
                    self.precond(st, t, &g.v[t], &mut mg.v[t], false)?;
                }
                if k == 0 {
                    pg.copy_from(st, &mg)?;
                } else {
                    sv.copy_from(st, x)?;
                    axpby(self.op.kernels(), st, -1.0, &x_prev, 1.0, &mut sv)?;
                    mv.copy_from(st, &g)?;
                    axpby(self.op.kernels(), st, -1.0, &g_prev, 1.0, &mut mv)?;
                    for t in 0..nt {
                        self.precond(st, t, &sv.v[t], &mut minv_s.v[t], true)?;
                        self.precond(st, t, &mv.v[t], &mut mm_v.v[t], false)?;
                    }
                    // inner products of the transformed pair s̃ = M^{-1/2}s, m̃ = M^{1/2}m
                    let ss = rdot(&*self.op, st, &sv, &minv_s)?;
                    let sm = rdot(&*self.op, st, &sv, &mv)?;
                    let mm = rdot(&*self.op, st, &mv, &mm_v)?;
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
                    // w = M^{1/2} ũ = s − τ (a s + (1−a) M m);  z = M^{-1/2} ṽ = a M⁻¹s + (1−a) m (into minv_s)
                    wv.copy_from(st, &sv)?;
                    axpby(self.op.kernels(), st, -tau * a, &sv, 1.0, &mut wv)?;
                    axpby(self.op.kernels(), st, -tau * (1.0 - a), &mm_v, 1.0, &mut wv)?;
                    axpby(self.op.kernels(), st, 1.0 - a, &mv, a, &mut minv_s)?;
                    let uv = rdot(&*self.op, st, &wv, &minv_s)?;
                    pg.copy_from(st, &mg)?;
                    axpby(self.op.kernels(), st, 0.0, &mg, tau, &mut pg)?;
                    let wn = rdot(&*self.op, st, &wv, &wv)?.sqrt();
                    let zn = rdot(&*self.op, st, &minv_s, &minv_s)?.sqrt();
                    if uv > 1e-8 * wn * zn {
                        let c = rdot(&*self.op, st, &wv, &g)? / uv;
                        axpby(self.op.kernels(), st, c, &wv, 1.0, &mut pg)?;
                    }
                    if k % 10 == 0 {
                        eprintln!("    zmshr1-M it {k}: a {a:.3} tau {tau:.3e} uv>thr {}", uv > 1e-8 * wn * zn);
                    }
                }
            }
            x_prev.copy_from(st, x)?;
            g_prev.copy_from(st, &g)?;
            // z = x − P α g, x = D(z)
            axpby(self.op.kernels(), st, -1.0, &pg, 1.0, x)?;
            for t in 0..nt {
                let h = st.clone_dtoh(&x.v[t])?;
                host[off[t]..off[t] + lens[t]].copy_from_slice(&h);
            }
            denoise(&mut host);
            for t in 0..nt {
                st.memcpy_htod(&host[off[t]..off[t] + lens[t]], &mut x.v[t])?;
            }
            report(k, x, f);
        }
        Ok(())
    }
}
