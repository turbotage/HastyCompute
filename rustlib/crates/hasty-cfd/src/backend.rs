//! Sparse tile topology, the backend interface and the CPU backend.
//!
//! Both backends run the same per-node C core (`kernels/lbm_core.h`); the CPU one calls it through FFI over
//! tile-aligned node ranges on rayon threads (a tile's accumulator cells are only touched by one thread; diagnostics
//! are reduced from per-chunk buffers).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::params::{GeomParams, MotionParams, SpinParams, StepParams, ACC_STRIDE, Q, TILE3};
use crate::Result;

/// active 8³ tiles and their neighbour table (27 per tile, −1 = not active)
#[derive(Clone, Debug)]
pub struct Topology {
    pub ntiles: usize,
    /// tile coordinates (x, y, z) in tile units, 3 per tile
    pub txyz: Vec<i32>,
    pub tnbr: Vec<i32>,
}

impl Topology {
    pub fn new(tiles: &[[i32; 3]]) -> Self {
        let map: HashMap<[i32; 3], i32> = tiles.iter().enumerate().map(|(i, t)| (*t, i as i32)).collect();
        let mut tnbr = vec![-1i32; tiles.len() * 27];
        for (i, t) in tiles.iter().enumerate() {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let k = ((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)) as usize;
                        tnbr[i * 27 + k] = *map.get(&[t[0] + dx, t[1] + dy, t[2] + dz]).unwrap_or(&-1);
                    }
                }
            }
        }
        Self { ntiles: tiles.len(), txyz: tiles.iter().flat_map(|t| t.iter().copied()).collect(), tnbr }
    }
    pub fn nodes(&self) -> usize {
        self.ntiles * TILE3
    }
    /// lattice coordinates of node n
    pub fn coords(&self, n: usize) -> [i32; 3] {
        let t = n >> 9;
        let l = (n & 511) as i32;
        [self.txyz[3 * t] * 8 + (l & 7), self.txyz[3 * t + 1] * 8 + ((l >> 3) & 7), self.txyz[3 * t + 2] * 8 + (l >> 6)]
    }
    /// node index of lattice coordinates (None if the tile is not active)
    pub fn node_of(&self, g: [i32; 3]) -> Option<usize> {
        let t = [g[0].div_euclid(8), g[1].div_euclid(8), g[2].div_euclid(8)];
        let ti = (0..self.ntiles).find(|&i| self.txyz[3 * i] == t[0] && self.txyz[3 * i + 1] == t[1] && self.txyz[3 * i + 2] == t[2])?;
        let l = ((g[2].rem_euclid(8) * 8 + g[1].rem_euclid(8)) * 8 + g[0].rem_euclid(8)) as usize;
        Some(ti * TILE3 + l)
    }
}

/// per-node fields for inspection (tests, debugging)
#[derive(Clone, Debug, Default)]
pub struct Fields {
    pub flags: Vec<u8>,
    /// ρ, ux, uy, uz (SoA, lattice units) of the latest step
    pub ru: Vec<f32>,
    pub phi: Vec<f32>,
    pub um: Vec<f32>,
    pub bsol: Vec<f32>,
    pub bcid: Vec<u8>,
}

pub trait Backend {
    fn nodes(&self) -> usize;
    /// U(t), ∂U/∂t(t) on the motion grid from the Catmull-Rom samples
    fn motion(&mut self, mp: &MotionParams) -> Result<()>;
    /// geometry update: inverse motion, distance, valves, cuts, wall links; with `refill` the uncovered nodes get
    /// equilibrium populations (at start-up: all of them); then the fluid flags are committed
    fn geometry(&mut self, gp: &GeomParams, valves: &[f32], cuts: &[f32], refill: bool) -> Result<()>;
    /// one stream + collide step (swaps the population and macroscopic buffers)
    fn step(&mut self, sp: &StepParams) -> Result<()>;
    /// add the latest velocity to the output accumulator (current phase bin)
    fn accumulate(&mut self, sp: &StepParams) -> Result<()>;
    /// the accumulator ([tile][cell][ACC_STRIDE]) and zero it
    fn take_acc(&mut self) -> Result<Vec<f32>>;
    /// diagnostics [4·ncuts + 2·nprobes + 2·nreg + 4 + 2·ncuts] (see `node_diag`); `light`: cuts and probes only (the
    /// regions and the global slots zero)
    fn diag(&mut self, ncuts: usize, nprobes: usize, nreg: usize, light: bool) -> Result<Vec<f32>>;
    /// populations (current source) and macroscopic fields (latest), for checkpoints
    fn save(&mut self) -> Result<(Vec<f32>, Vec<f32>)>;
    fn restore(&mut self, f: &[f32], ru: &[f32]) -> Result<()>;
    fn fields(&mut self) -> Result<Fields>;
    /// spin history: the α(z) table, one advection / saturation / recovery step of the scalar, its state
    fn set_spin_table(&mut self, atab: &[f32]) -> Result<()>;
    fn scalar(&mut self, sp: &SpinParams) -> Result<()>;
    fn save_scalar(&mut self) -> Result<Vec<f32>>;
    fn restore_scalar(&mut self, s: &[f32]) -> Result<()>;
    /// GPU pacing point (hasty_cuda::pace: HASTY_GPU_DUTY / HASTY_GPU_WATTS); a no-op unless configured, and on the CPU
    fn pace(&self) {}
}

// ── CPU backend ─────────────────────────────────────────────────────────────────────────────────────────────────────

extern "C" {
    fn cpu_motion(mp: *const MotionParams, k0: i64, k1: i64, uall: *const u16, ut: *mut f32, vt: *mut f32);
    fn cpu_geom1(p: *const GeomParams, n0: i64, n1: i64, n: i64, txyz: *const i32, ut: *const f32, vt: *const f32, sdf: *const f32, valves: *const f32, cuts: *const f32, xref: *mut f32, phi: *mut f32, um: *mut f32, bsol: *mut f32, bcid: *mut u8, flags: *mut u8, spg: *mut f32, pid: *mut u8, rmap: *const u8, rid: *mut u8);
    fn cpu_geom2(n0: i64, n1: i64, n: i64, tnbr: *const i32, phi: *const f32, links: *mut u32, qlink: *mut u8, flags: *mut u8);
    fn cpu_refill(n0: i64, n1: i64, n: i64, tnbr: *const i32, flags: *const u8, um: *const f32, f: *mut f32, ru: *mut f32);
    fn cpu_commit(n0: i64, n1: i64, flags: *mut u8);
    fn cpu_step(p: *const StepParams, n0: i64, n1: i64, n: i64, tnbr: *const i32, flags: *const u8, links: *const u32, qlink: *const u8, um: *const f32, bsol: *const f32, bcid: *const u8, cuts: *const f32, src: *const f32, dst: *mut f32, ru_prev: *const f32, ru_cur: *mut f32, spg: *const f32);
    fn cpu_accum(p: *const StepParams, n0: i64, n1: i64, n: i64, flags: *const u8, bsol: *const f32, ru: *const f32, acc: *mut f32, tnbr: *const i32, phi: *const f32, spg: *const f32, f: *const f32, sc: *const f32);
    fn cpu_scalar(sp: *const SpinParams, pass: i32, n0: i64, n1: i64, n: i64, txyz: *const i32, tnbr: *const i32, flags: *const u8, bcid: *const u8, cuts: *const f32, ru: *const f32, atab: *const f32, s_old: *const f32, aux: *const f32, s_new: *mut f32);
    fn cpu_srefill(n0: i64, n1: i64, tnbr: *const i32, flags: *const u8, s: *mut f32);
    fn cpu_diag(n0: i64, n1: i64, n: i64, ncuts: i32, nprobes: i32, nreg: i32, mode: i32, flags: *const u8, bcid: *const u8, pid: *const u8, rid: *const u8, cuts: *const f32, ru: *const f32, sc: *const f32, diag: *mut f32);
}

/// a raw pointer that may cross threads (each thread writes a disjoint node range)
#[derive(Clone, Copy)]
struct P<T>(*mut T);
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}
impl<T> P<T> {
    /// the pointer (a method, so closures capture the whole Send wrapper, not the raw field)
    fn p(self) -> *mut T {
        self.0
    }
    fn m(v: &mut [T]) -> Self {
        P(v.as_mut_ptr())
    }
    fn c(v: &[T]) -> Self {
        P(v.as_ptr() as *mut T)
    }
}

pub struct CpuBackend {
    pub topo: Topology,
    n: usize,
    f: [Vec<f32>; 2],
    ru: [Vec<f32>; 2],
    cur: usize,
    xref: Vec<f32>,
    phi: Vec<f32>,
    um: Vec<f32>,
    bsol: Vec<f32>,
    bcid: Vec<u8>,
    pid: Vec<u8>,
    rid: Vec<u8>,
    rmap: Vec<u8>,
    flags: Vec<u8>,
    links: Vec<u32>,
    qlink: Vec<u8>,
    spg: Vec<f32>,
    uall: Vec<u16>,
    mvox: usize,
    ut: Vec<f32>,
    vt: Vec<f32>,
    sdf: Vec<f32>,
    valves: Vec<f32>,
    cuts: Vec<f32>,
    acc: Vec<f32>,
    /// spin scalar (two buffers) and the α(z) table
    sc: [Vec<f32>; 2],
    scur: usize,
    stmp: [Vec<f32>; 2],
    atab: Vec<f32>,
    /// nodes per parallel chunk (whole tiles)
    chunk: usize,
}

impl CpuBackend {
    /// `uall`: the motion samples (f16 bits [T][3][Z][Y][X]), `sdf`: the reference distance [Z][Y][X], `cells`: output
    /// cells per tile
    pub fn new(topo: Topology, uall: Vec<u16>, mvox: usize, sdf: Vec<f32>, rmap: Vec<u8>, cells: usize) -> Self {
        let n = topo.nodes();
        let acc = vec![0.0; topo.ntiles * cells * ACC_STRIDE];
        Self {
            n,
            f: [vec![0.0; Q * n], vec![0.0; Q * n]],
            ru: [vec![0.0; 4 * n], vec![0.0; 4 * n]],
            cur: 0,
            xref: vec![0.0; 3 * n],
            phi: vec![-1.0; n],
            um: vec![0.0; 3 * n],
            bsol: vec![0.0; n],
            bcid: vec![0; n],
            pid: vec![0; n],
            rid: vec![0; n],
            rmap: if rmap.is_empty() { vec![0] } else { rmap },
            flags: vec![0; n],
            links: vec![0; n],
            qlink: vec![0; (Q - 1) * n],
            spg: vec![0.0; n],
            uall,
            mvox,
            ut: vec![0.0; 3 * mvox],
            vt: vec![0.0; 3 * mvox],
            sdf,
            valves: vec![0.0; 1],
            cuts: vec![0.0; 1],
            acc,
            sc: [vec![1.0; n], vec![1.0; n]],
            scur: 0,
            stmp: [vec![1.0; n], vec![1.0; n]],
            atab: vec![0.0; 2],
            chunk: 8 * TILE3,
            topo,
        }
    }

    fn ranges(&self) -> Vec<(i64, i64)> {
        (0..self.n).step_by(self.chunk).map(|a| (a as i64, (a + self.chunk).min(self.n) as i64)).collect()
    }
}

impl Backend for CpuBackend {
    fn nodes(&self) -> usize {
        self.n
    }

    fn motion(&mut self, mp: &MotionParams) -> Result<()> {
        let (ua, ut, vt) = (P::c(&self.uall), P::m(&mut self.ut), P::m(&mut self.vt));
        let m = self.mvox;
        (0..m).step_by(65536).collect::<Vec<_>>().into_par_iter().for_each(|k0| unsafe {
            cpu_motion(mp, k0 as i64, (k0 + 65536).min(m) as i64, ua.p(), ut.p(), vt.p());
        });
        Ok(())
    }

    fn geometry(&mut self, gp: &GeomParams, valves: &[f32], cuts: &[f32], refill: bool) -> Result<()> {
        self.valves = if valves.is_empty() { vec![0.0] } else { valves.to_vec() };
        self.cuts = if cuts.is_empty() { vec![0.0] } else { cuts.to_vec() };
        let n = self.n as i64;
        let r = self.ranges();
        let (tx, tn) = (P::c(&self.topo.txyz), P::c(&self.topo.tnbr));
        let (ut, vt, sdf, va, cu) = (P::c(&self.ut), P::c(&self.vt), P::c(&self.sdf), P::c(&self.valves), P::c(&self.cuts));
        let (xr, ph, um, bs, bc, fl, sg, pi) = (P::m(&mut self.xref), P::m(&mut self.phi), P::m(&mut self.um), P::m(&mut self.bsol), P::m(&mut self.bcid), P::m(&mut self.flags), P::m(&mut self.spg), P::m(&mut self.pid));
        let (rm, ri) = (P::c(&self.rmap), P::m(&mut self.rid));
        r.par_iter().for_each(|&(a, b)| unsafe { cpu_geom1(gp, a, b, n, tx.p(), ut.p(), vt.p(), sdf.p(), va.p(), cu.p(), xr.p(), ph.p(), um.p(), bs.p(), bc.p(), fl.p(), sg.p(), pi.p(), rm.p(), ri.p()) });
        let (li, ql) = (P::m(&mut self.links), P::m(&mut self.qlink));
        r.par_iter().for_each(|&(a, b)| unsafe { cpu_geom2(a, b, n, tn.p(), ph.p(), li.p(), ql.p(), fl.p()) });
        if refill {
            let (f, ru) = (P::m(&mut self.f[self.cur]), P::m(&mut self.ru[self.cur]));
            r.par_iter().for_each(|&(a, b)| unsafe { cpu_refill(a, b, n, tn.p(), fl.p(), um.p(), f.p(), ru.p()) });
            let sc = P::m(&mut self.sc[self.scur]);
            r.par_iter().for_each(|&(a, b)| unsafe { cpu_srefill(a, b, tn.p(), fl.p(), sc.p()) });
        }
        r.par_iter().for_each(|&(a, b)| unsafe { cpu_commit(a, b, fl.p()) });
        Ok(())
    }

    fn step(&mut self, sp: &StepParams) -> Result<()> {
        let n = self.n as i64;
        let r = self.ranges();
        let (c, o) = (self.cur, 1 - self.cur);
        let (src, dst) = if c == 0 {
            let (a, b) = self.f.split_at_mut(1);
            (P::c(&a[0]), P::m(&mut b[0]))
        } else {
            let (a, b) = self.f.split_at_mut(1);
            (P::c(&b[0]), P::m(&mut a[0]))
        };
        let (rp, rc) = if c == 0 {
            let (a, b) = self.ru.split_at_mut(1);
            (P::c(&a[0]), P::m(&mut b[0]))
        } else {
            let (a, b) = self.ru.split_at_mut(1);
            (P::c(&b[0]), P::m(&mut a[0]))
        };
        let (tn, fl, li, ql, um, bs, bc, cu) = (P::c(&self.topo.tnbr), P::c(&self.flags), P::c(&self.links), P::c(&self.qlink), P::c(&self.um), P::c(&self.bsol), P::c(&self.bcid), P::c(&self.cuts));
        let sg = P::c(&self.spg);
        r.par_iter().for_each(|&(a, b)| unsafe { cpu_step(sp, a, b, n, tn.p(), fl.p(), li.p(), ql.p(), um.p(), bs.p(), bc.p(), cu.p(), src.p(), dst.p(), rp.p(), rc.p(), sg.p()) });
        self.cur = o;
        Ok(())
    }

    fn accumulate(&mut self, sp: &StepParams) -> Result<()> {
        let n = self.n as i64;
        let (fl, bs, ru, acc) = (P::c(&self.flags), P::c(&self.bsol), P::c(&self.ru[self.cur]), P::m(&mut self.acc));
        let (tn, ph, sg, f, sc) = (P::c(&self.topo.tnbr), P::c(&self.phi), P::c(&self.spg), P::c(&self.f[self.cur]), P::c(&self.sc[self.scur]));
        self.ranges().par_iter().for_each(|&(a, b)| unsafe { cpu_accum(sp, a, b, n, fl.p(), bs.p(), ru.p(), acc.p(), tn.p(), ph.p(), sg.p(), f.p(), sc.p()) });
        Ok(())
    }

    fn take_acc(&mut self) -> Result<Vec<f32>> {
        let out = self.acc.clone();
        self.acc.iter_mut().for_each(|x| *x = 0.0);
        Ok(out)
    }

    fn diag(&mut self, ncuts: usize, nprobes: usize, nreg: usize, light: bool) -> Result<Vec<f32>> {
        let n = self.n as i64;
        let len = 4 * ncuts + 2 * nprobes + 2 * nreg + 4 + 2 * ncuts;
        let (fl, bc, pi, ri, cu, ru, sc) = (P::c(&self.flags), P::c(&self.bcid), P::c(&self.pid), P::c(&self.rid), P::c(&self.cuts), P::c(&self.ru[self.cur]), P::c(&self.sc[self.scur]));
        let parts: Vec<Vec<f32>> = self
            .ranges()
            .par_iter()
            .map(|&(a, b)| {
                let mut d = vec![0f32; len];
                unsafe { cpu_diag(a, b, n, ncuts as i32, nprobes as i32, nreg as i32, light as i32, fl.p(), bc.p(), pi.p(), ri.p(), cu.p(), ru.p(), sc.p(), d.as_mut_ptr()) };
                d
            })
            .collect();
        let mut d = vec![0f32; len];
        for p in parts {
            for k in 0..len {
                if k == 4 * ncuts + 2 * nprobes + 2 * nreg + 2 {
                    d[k] = d[k].max(p[k]);
                } else {
                    d[k] += p[k];
                }
            }
        }
        Ok(d)
    }

    fn save(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok((self.f[self.cur].clone(), self.ru[self.cur].clone()))
    }

    fn restore(&mut self, f: &[f32], ru: &[f32]) -> Result<()> {
        self.f[self.cur].copy_from_slice(f);
        self.ru[self.cur].copy_from_slice(ru);
        Ok(())
    }

    fn fields(&mut self) -> Result<Fields> {
        Ok(Fields { flags: self.flags.clone(), ru: self.ru[self.cur].clone(), phi: self.phi.clone(), um: self.um.clone(), bsol: self.bsol.clone(), bcid: self.bcid.clone() })
    }

    fn set_spin_table(&mut self, atab: &[f32]) -> Result<()> {
        self.atab = if atab.len() < 2 { vec![0.0; 2] } else { atab.to_vec() };
        Ok(())
    }

    fn scalar(&mut self, sp: &SpinParams) -> Result<()> {
        let n = self.n as i64;
        let (tx, tn, fl, bc, cu, ru, at) = (P::c(&self.topo.txyz), P::c(&self.topo.tnbr), P::c(&self.flags), P::c(&self.bcid), P::c(&self.cuts), P::c(&self.ru[self.cur]), P::c(&self.atab));
        let (so, sn) = if self.scur == 0 {
            let (a, b) = self.sc.split_at_mut(1);
            (P::c(&a[0]), P::m(&mut b[0]))
        } else {
            let (a, b) = self.sc.split_at_mut(1);
            (P::c(&b[0]), P::m(&mut a[0]))
        };
        let (t0, t1) = {
            let (a, b) = self.stmp.split_at_mut(1);
            (P::m(&mut a[0]), P::m(&mut b[0]))
        };
        let r = self.ranges();
        // BFECC: ŝ → t0, s_c → t1, s' → sn
        r.par_iter().for_each(|&(a, b)| unsafe { cpu_scalar(sp, 0, a, b, n, tx.p(), tn.p(), fl.p(), bc.p(), cu.p(), ru.p(), at.p(), so.p(), so.p(), t0.p()) });
        r.par_iter().for_each(|&(a, b)| unsafe { cpu_scalar(sp, 1, a, b, n, tx.p(), tn.p(), fl.p(), bc.p(), cu.p(), ru.p(), at.p(), so.p(), t0.p(), t1.p()) });
        r.par_iter().for_each(|&(a, b)| unsafe { cpu_scalar(sp, 2, a, b, n, tx.p(), tn.p(), fl.p(), bc.p(), cu.p(), ru.p(), at.p(), so.p(), t1.p(), sn.p()) });
        self.scur = 1 - self.scur;
        Ok(())
    }

    fn save_scalar(&mut self) -> Result<Vec<f32>> {
        Ok(self.sc[self.scur].clone())
    }

    fn restore_scalar(&mut self, s: &[f32]) -> Result<()> {
        self.sc[self.scur].copy_from_slice(s);
        Ok(())
    }
}
