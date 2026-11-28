//! The driver: units, active tiles, valve / cut kinematics through the prescribed motion, RCR outlets, phase-binned
//! accumulation of the Eulerian velocity, diagnostics, checkpoints.
//!
//! Units: dt = u_lattice · dx / v_max, rounded so that a cardiac period is a whole number of steps; ν_lattice = ν dt /
//! dx², τ0 = 3 ν_lattice + ½ (≈ 0.5 + 10⁻³ at 0.5 mm: LES regime, WALE on top). Pressure p ↔ density ρ = 1 + 3 p dt² /
//! (ρ_blood dx²).
//!
//! Time loop: every `geom_every` steps the motion field, valves and cuts are re-evaluated at the current time and the
//! geometry pass runs (walls move ≲ 0.01 dx per step, so the update interval is a few steps); every `diag_every` steps
//! the diagnostics (mass, max |u|, cut fluxes) update the Windkessel outlets and are logged (`log.jsonl`); from cycle
//! `warmup` on, every `acc_every` steps the velocity is added to the accumulator of the current phase bin, which is
//! flushed into `flow_bins.f32` ([bin][tile][cell][Σw, Σwu (3), Σwuuᵀ (6)], m/s) when the bin changes; checkpoints
//! every `ckpt_every` steps and on a stop request (resumable).

use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use serde_json::json;

use crate::backend::{Backend, CpuBackend, Topology};
use crate::case::{cross, dot, normalize, Case, CutKind, ValveKind};
use crate::params::{GeomParams, SpinParams, StepParams, ACC_STRIDE, CUT_STRIDE, Q, VALVE_STRIDE};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Units {
    pub dx: f64,
    pub dt: f64,
    pub period: f64,
    pub steps_per_cycle: u64,
    /// m/s → lattice velocity
    pub vscale: f64,
    pub nu_lat: f64,
    pub tau0: f64,
    pub rho: f64,
}

impl Units {
    pub fn new(case: &Case) -> Self {
        let dx = case.grid.dx;
        let period = case.motion.period;
        let dt0 = case.u_lattice * dx / case.v_max;
        let spc = (period / dt0).ceil().max(1.0) as u64;
        let dt = period / spc as f64;
        let nu_lat = case.nu * dt / (dx * dx);
        Self { dx, dt, period, steps_per_cycle: spc, vscale: dt / dx, nu_lat, tau0: 3.0 * nu_lat + 0.5, rho: case.rho }
    }
    /// the units of refinement level l (dx, dt / 2^l: the lattice velocity is the same, ν_lattice and τ0 − ½ double per
    /// level)
    pub fn level(&self, l: usize) -> Self {
        let k = (1u64 << l) as f64;
        let nu_lat = self.nu_lat * k;
        Self { dx: self.dx / k, dt: self.dt / k, period: self.period, steps_per_cycle: self.steps_per_cycle << l, vscale: self.vscale, nu_lat, tau0: 3.0 * nu_lat + 0.5, rho: self.rho }
    }
    pub fn p_to_drho(&self, p: f64) -> f64 {
        3.0 * p * self.dt * self.dt / (self.rho * self.dx * self.dx)
    }
    pub fn drho_to_p(&self, d: f64) -> f64 {
        d * self.rho * self.dx * self.dx / (3.0 * self.dt * self.dt)
    }
}

/// tiles that hold blood at some time of the cycle (tile centre within a margin of the moving domain at `nsamp` times)
pub fn active_tiles(case: &Case, nsamp: usize) -> Vec<[i32; 3]> {
    let g = &case.grid;
    let nt = g.dims.map(|d| d.div_ceil(8) as i32);
    let margin = (3f64.sqrt() * 4.0 + 4.0) * g.dx * 1.5;
    let cand: Vec<[i32; 3]> = (0..nt[2]).flat_map(|z| (0..nt[1]).flat_map(move |y| (0..nt[0]).map(move |x| [x, y, z]))).collect();
    let period = case.motion.period;
    cand.into_par_iter()
        .filter(|t| {
            let c = [0, 1, 2].map(|a| g.origin[a] + (t[a] as f64 * 8.0 + 3.5) * g.dx);
            (0..nsamp).any(|s| {
                let tt = s as f64 * period / nsamp as f64;
                let x0 = case.motion.inverse(&c, tt, 6);
                case.sdf.at(&x0, -1.0) > -margin
            })
        })
        .collect()
}

/// precomputed leaflet in units of the hinge radius: hinge point, hinge direction d (= m0 × n), inward m0, polygon (a, b)
#[derive(Clone, Debug)]
struct Leaflet {
    pa: [f64; 2],
    d: [f64; 2],
    m0: [f64; 2],
    poly: Vec<[f64; 2]>,
}

fn prep_leaflet(p: &[[f64; 2]]) -> Leaflet {
    let pa = p[0];
    let ch = [p[1][0] - p[0][0], p[1][1] - p[0][1]];
    let l = (ch[0] * ch[0] + ch[1] * ch[1]).sqrt();
    let ch = [ch[0] / l, ch[1] / l];
    let cen = [p.iter().map(|v| v[0]).sum::<f64>() / p.len() as f64, p.iter().map(|v| v[1]).sum::<f64>() / p.len() as f64];
    let mut m0 = [-ch[1], ch[0]];
    if (cen[0] - pa[0]) * m0[0] + (cen[1] - pa[1]) * m0[1] < 0.0 {
        m0 = [-m0[0], -m0[1]];
    }
    // d3 = m0 × n with n = e1 × e2 (local z): (m0.y, −m0.x)
    let d = [m0[1], -m0[0]];
    let mut poly: Vec<[f64; 2]> = p.iter().map(|v| [(v[0] - pa[0]) * d[0] + (v[1] - pa[1]) * d[1], (v[0] - pa[0]) * m0[0] + (v[1] - pa[1]) * m0[1]]).collect();
    let area: f64 = (0..poly.len()).map(|k| { let (a, b) = (poly[k], poly[(k + 1) % poly.len()]); a[0] * b[1] - b[0] * a[1] }).sum();
    if area < 0.0 {
        poly.reverse();
    }
    Leaflet { pa, d, m0, poly }
}

/// a finer refinement level (refine.rs): its backend, topology, units and interface statistics
pub struct Level<B: Backend> {
    pub l: usize,
    pub be: B,
    pub topo: Topology,
    pub u: Units,
    pub ghosts: usize,
    pub rim: usize,
    pub cells: usize,
}

pub struct Solver<B: Backend> {
    /// finer refinement levels 1, 2, … (empty: one uniform level, the behaviour before grid refinement)
    pub fine: Vec<Level<B>>,
    /// node updates per coarse step over all levels (Σ fluid nodes × 2^l; the MLUPS of a refined run)
    node_updates: f64,
    /// wall mass: the region bit mask whose wall operations are mass-neutral (RunSpec::wall_mass; 0: off)
    pub wmask: u32,
    /// time of the last wall-mass correction (the pump target integrates −dV/dt from it)
    wm_last_t: Option<f64>,
    /// wall pump gain per region (1 at start), updated every cycle so that each side's venous inflow (exact link flux,
    /// the chambers' throughput) matches the model's: [left (pulmonary venous), right (systemic venous)] accumulators
    pub pump_gain: Vec<f64>,
    q_link: Vec<f64>,
    pg_meas: [f64; 2],
    pg_want: [f64; 2],
    pg_t: Option<f64>,
    pg_cycle: i64,
    pub case: Case,
    pub u: Units,
    pub topo: Topology,
    pub be: B,
    pub step: u64,
    pub cells: usize,
    leaflets: Vec<Vec<Leaflet>>,
    wk_pc: Vec<f64>,
    q_out: Vec<f64>,
    /// measured section of each cut (the first interior node layer in front of the plane), m²; π r² until the first
    /// diagnostics
    area: Vec<f64>,
    /// inlet flux controller: the plug velocity is κ Q / section, κ adjusted so that the delivered flux matches Q
    /// (an equilibrium velocity slab loses some of its nominal flux to the stub walls)
    kappa: Vec<f64>,
    /// share trim of pressure cuts (Pa) and the cycle-scale moving average of their inflow (m³/s)
    p_trim: Vec<f64>,
    q_ema: Vec<f64>,
    /// spin history: per cut the cycle-scale moving averages of the flux-layer Σ u_n s and Σ u_n (the outflow's
    /// flux-weighted s is their ratio), and the entering s of each cut (cut slot 19)
    s_flux: Vec<[f64; 2]>,
    pub s_in: Vec<f64>,
    last_diag_t: Option<f64>,
    /// inertial inlets: the flow added to q(t) (m³/s) and the last measured cut pressure (Pa)
    dq: Vec<f64>,
    p_cut: Vec<f64>,
    /// pressure-triggered valves: dynamic opening, its rate (per s), filtered forward gradient (Pa), probe pressures
    o_dyn: Vec<f64>,
    o_rate: Vec<f64>,
    dp_f: Vec<f64>,
    p_probe: Vec<[f64; 2]>,
    /// scheduled opening times per valve (rising through 0.05)
    open_edges: Vec<Vec<f64>>,
    /// (cycle, length) of the beat file at the checkpoint being resumed
    beat_resume: Option<(u64, u64)>,
    comp: Compartments,
}

/// what a run reports
#[derive(Clone, Debug)]
pub struct Report {
    pub steps: u64,
    pub seconds: f64,
    pub mlups: f64,
    pub last_diag: serde_json::Value,
    pub stopped: bool,
}

impl Solver<CpuBackend> {
    pub fn new_cpu(case: Case) -> Result<Self> {
        Self::with_backend(case, |topo, uall, mvox, sdf, rmap, cells| Ok(CpuBackend::new(topo, uall, mvox, sdf, rmap, cells)))
    }
}

impl Solver<crate::gpu::GpuBackend> {
    pub fn new_gpu(case: Case, dev: &hasty_cuda::Device) -> Result<Self> {
        Self::with_backend(case, |topo, uall, mvox, sdf, rmap, cells| crate::gpu::GpuBackend::new(dev, topo, &uall, mvox, &sdf, &rmap, cells))
    }
}

/// compartments of the region graph: regions joined by free contacts (`allowed`, e.g. a vein stub and its atrium);
/// valves join two compartments; a compartment with a cut is never sealed
#[derive(Clone, Debug, Default)]
struct Compartments {
    /// per valve: (upstream, downstream) compartment, or None without a region graph
    sides: Vec<Option<(usize, usize)>>,
    has_cut: Vec<bool>,
    /// valves touching each compartment
    valves_of: Vec<Vec<usize>>,
}

impl Compartments {
    fn new(case: &Case) -> Self {
        let Some(r) = &case.regions else { return Self { sides: vec![None; case.valves.len()], ..Default::default() } };
        let ids: Vec<u16> = r.names.keys().copied().collect();
        let mut parent: std::collections::BTreeMap<u16, u16> = ids.iter().map(|&i| (i, i)).collect();
        fn find(p: &mut std::collections::BTreeMap<u16, u16>, x: u16) -> u16 {
            let mut x = x;
            while p[&x] != x {
                let up = p[&p[&x]];
                p.insert(x, up);
                x = up;
            }
            x
        }
        for &(a, b) in &r.allowed {
            if parent.contains_key(&a) && parent.contains_key(&b) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent.insert(ra, rb);
            }
        }
        let roots: Vec<u16> = {
            let mut v: Vec<u16> = ids.iter().map(|&i| find(&mut parent, i)).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        let comp_of = |p: &mut std::collections::BTreeMap<u16, u16>, id: u16| roots.iter().position(|&x| x == find(p, id));
        let mut has_cut = vec![false; roots.len()];
        for c in &case.cuts {
            if let Some((&id, _)) = r.names.iter().find(|(_, n)| **n == c.name) {
                if let Some(k) = comp_of(&mut parent, id) {
                    has_cut[k] = true;
                }
            }
        }
        let mut valves_of = vec![vec![]; roots.len()];
        let sides = case
            .valves
            .iter()
            .enumerate()
            .map(|(vi, v)| {
                let &(u, d) = r.valves.get(&v.name)?;
                let (cu, cd) = (comp_of(&mut parent, u)?, comp_of(&mut parent, d)?);
                valves_of[cu].push(vi);
                valves_of[cd].push(vi);
                Some((cu, cd))
            })
            .collect();
        Self { sides, has_cut, valves_of }
    }

    /// Some(true) while valve `vi` has a sealed compartment on one side at time t (by the schedule), None without
    /// a region graph
    fn sealed_side(&self, vi: usize, valves: &[crate::case::ValveSpec], t: f64) -> Option<bool> {
        let (u, d) = (*self.sides.get(vi)?)?;
        let sealed = |c: usize| !self.has_cut[c] && self.valves_of[c].iter().all(|&w| valves[w].opening.at(t) < 0.05);
        Some(sealed(u) || sealed(d))
    }
}

impl<B: Backend> Solver<B> {
    pub fn with_backend(case: Case, make: impl FnOnce(Topology, Vec<u16>, usize, Vec<f32>, Vec<u8>, usize) -> Result<B>) -> Result<Self> {
        for v in &case.valves {
            if let ValveKind::Leaflets { polys } = &v.kind {
                if polys.len() > crate::params::MAX_LEAFLETS || polys.iter().any(|p| p.len() < 3 || p.len() > 4) {
                    return Err(Error::Case(format!("valve {}: ≤ {} leaflet strips of 3–4 vertices", v.name, crate::params::MAX_LEAFLETS)));
                }
            }
        }
        let u = Units::new(&case);
        let tiles = active_tiles(&case, 24);
        if tiles.is_empty() {
            return Err(Error::Case("no active tiles: the domain is empty".into()));
        }
        let topo = Topology::new(&tiles);
        let f = 8 / case.run.out_factor;
        let cells = f * f * f;
        let leaflets = case.valves.iter().map(|v| if let ValveKind::Leaflets { polys } = &v.kind { polys.iter().map(|p| prep_leaflet(p)).collect() } else { vec![] }).collect();
        let rmap = case.regions.as_ref().map_or(vec![], |r| r.map.clone());
        let mut be = make(topo.clone(), case.motion.u.clone(), case.motion.voxels(), case.sdf.data.clone(), rmap, cells)?;
        if let Some(sp) = &case.spin {
            be.set_spin_table(&sp.alpha)?;
        }
        if let Some(tr) = &case.transpiration {
            be.set_transpiration(&tr.group, &tr.weight)?;
        }
        // tissue-frame valve planes (reference pose) for the exact valve flux; the lateral reach is the hinge circle +
        // 2 lattice units, the upstream depth 2 lattice units (only links next to the plane cross it)
        let vp: Vec<f32> = case
            .valves
            .iter()
            .flat_map(|v| {
                let on = v.tissue;
                [v.center[0], v.center[1], v.center[2], v.normal[0], v.normal[1], v.normal[2], if on { v.radius + 2.0 * case.grid.dx } else { 0.0 }, case.grid.dx].map(|x| x as f32)
            })
            .collect();
        be.set_vplanes(&vp)?;
        // wall mass (RunSpec::wall_mass): the chambers' regions (blood_*), or the whole domain without a region map; the
        // vessel trees keep their wall transpiration, which is a deliberate wall mass flux
        let wmask: u32 = if case.run.wall_pump && !case.wall_pump.is_empty() {
            case.wall_pump.iter().filter(|(id, _)| (*id as usize) < crate::params::MAX_REGIONS).fold(0u32, |m, (id, _)| m | (1u32 << *id))
        } else if !case.run.wall_mass {
            0
        } else if let Some(r) = &case.regions {
            r.names.iter().filter(|(id, nm)| (**id as usize) < crate::params::MAX_REGIONS && nm.starts_with("blood_")).fold(0u32, |m, (id, _)| m | (1u32 << *id))
        } else {
            1
        };
        let pump = case.run.wall_pump && !case.wall_pump.is_empty();
        be.set_wall_mass(wmask, !pump);
        // grid refinement: the finer levels share the case buffers of level 0
        let mut fine = vec![];
        if !case.refine.is_empty() {
            let b = crate::refine::build(&case, &topo, 24)?;
            be.set_roles(&b.roles0)?;
            for lb in b.levels {
                let l = lb.level;
                let mut fb = be.new_level(lb.topo.clone())?;
                fb.set_roles(&lb.roles)?;
                fb.set_interface(&lb.iface)?;
                fb.set_wale_scale(&crate::refine::wale_ramp(&lb.topo, &lb.roles, case.run.refine_wale_ramp))?;
                if let Some(sp) = &case.spin {
                    fb.set_spin_table(&sp.alpha)?;
                }
                let dxl = case.grid.dx / (1u64 << l) as f64;
                let vpl: Vec<f32> = case
                    .valves
                    .iter()
                    .flat_map(|v| [v.center[0], v.center[1], v.center[2], v.normal[0], v.normal[1], v.normal[2], if v.tissue { v.radius + 2.0 * dxl } else { 0.0 }, dxl].map(|x| x as f32))
                    .collect();
                fb.set_vplanes(&vpl)?;
                fb.set_wall_mass(wmask, !pump);
                let ul = u.level(l);
                fine.push(Level { l, be: fb, topo: lb.topo, u: ul, ghosts: lb.iface.glist.len(), rim: lb.iface.rim.len(), cells: lb.cells });
                eprintln!(
                    "refinement level {l}: {} coarse cells refined (dropped: {} nesting margin, {} never blood), {} tiles = {:.2} M nodes, {} ghosts, {} rim nodes",
                    lb.cells,
                    lb.dropped_nest,
                    lb.dropped_inactive,
                    fine.last().map_or(0, |x: &Level<B>| x.topo.ntiles),
                    fine.last().map_or(0, |x: &Level<B>| x.topo.nodes()) as f64 / 1e6,
                    lb.iface.glist.len(),
                    lb.iface.rim.len()
                );
            }
        }
        let comp = Compartments::new(&case);
        let nc = case.cuts.len();
        let nv = case.valves.len();
        let open_edges = case.valves.iter().map(|v| v.opening.rising(0.05)).collect();
        let area = case.cuts.iter().map(|c| std::f64::consts::PI * c.radius * c.radius).collect();
        let mut s = Self { fine, node_updates: 0.0, case, u, topo, be, step: 0, cells, leaflets, wk_pc: vec![0.0; nc], q_out: vec![0.0; nc], area, kappa: vec![1.0; nc], p_trim: vec![0.0; nc], q_ema: vec![f64::NAN; nc], s_flux: vec![[0.0; 2]; nc], s_in: vec![1.0; nc], last_diag_t: None, dq: vec![0.0; nc], p_cut: vec![0.0; nc], o_dyn: vec![0.0; nv], o_rate: vec![0.0; nv], dp_f: vec![0.0; nv], p_probe: vec![[0.0; 2]; nv], open_edges, comp, beat_resume: None, wmask, wm_last_t: None, pump_gain: vec![1.0; crate::params::MAX_REGIONS], q_link: vec![0.0; nc], pg_meas: [0.0; 2], pg_want: [0.0; 2], pg_t: None, pg_cycle: -1 };
        s.update_s_in();
        Ok(s)
    }

    /// lab thickness (lattice units) of a cut's flux layer, one reference unit thick: 1 / |F⁻ᵀ n| with F = ∂Φ_t/∂X at the
    /// cut centre (central differences of the motion over 2 mm)
    fn layer_thickness(&self, ci: usize) -> f64 {
        let c = &self.case.cuts[ci];
        let m = &self.case.motion;
        let t = self.time();
        let e = 1e-3;
        let mut f = [[0.0; 3]; 3];
        for j in 0..3 {
            let mut a = c.center;
            let mut b = c.center;
            a[j] += e;
            b[j] -= e;
            let (pa, pb) = (m.forward(&a, t), m.forward(&b, t));
            for i in 0..3 {
                f[i][j] = (pa[i] - pb[i]) / (2.0 * e);
            }
        }
        let det = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1]) - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0]) + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
        if det.abs() < 1e-6 {
            return 1.0;
        }
        // F⁻ᵀ n = cof(F) n / det
        let cof = |i: usize, j: usize| {
            let (i1, i2, j1, j2) = ((i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3);
            f[i1][j1] * f[i2][j2] - f[i1][j2] * f[i2][j1]
        };
        let v: Vec<f64> = (0..3).map(|i| (0..3).map(|j| cof(i, j) * c.normal[j]).sum::<f64>() / det).collect();
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if l > 1e-6 { (1.0 / l).clamp(0.3, 3.0) } else { 1.0 }
    }

    /// [cycle, byte length] of the beat file being written (the checkpoint's resume point)
    fn beat_state(&self, out: &Path) -> serde_json::Value {
        let spc = self.u.steps_per_cycle;
        let cyc = self.step / spc;
        let p = out.join("beats").join(format!("beat_{cyc:04}.bin"));
        json!([cyc, std::fs::metadata(&p).map_or(0, |m| m.len())])
    }

    /// the spin scalar's step parameters (None without a spin history)
    pub fn spin_params(&self) -> Option<SpinParams> {
        self.spin_params_at(0)
    }

    /// the spin scalar's step parameters of refinement level l
    pub fn spin_params_at(&self, l: usize) -> Option<SpinParams> {
        let s = self.case.spin.as_ref()?;
        let ul = self.u.level(l);
        let o = crate::refine::level_origin(self.case.grid.origin, self.u.dx, l);
        Some(SpinParams {
            dt_tr: (ul.dt / s.tr) as f32,
            e1_step: (-ul.dt / s.t1).exp() as f32,
            o: [0, 1, 2].map(|a| (o[a] - s.center[a]) as f32),
            nrm: s.normal.map(|x| x as f32),
            dx: ul.dx as f32,
            ntab: s.alpha.len() as i32,
            az0: s.z0 as f32,
            adz: s.dz as f32,
            enabled: 1,
        })
    }

    /// region slots of the diagnostics (0 without a region map)
    fn nreg(&self) -> usize {
        if self.case.regions.is_some() { crate::params::MAX_REGIONS } else { 0 }
    }

    pub fn time(&self) -> f64 {
        self.step as f64 * self.u.dt
    }

    /// lattice coordinates of a world position on refinement level l
    fn lat_at(&self, l: usize, x: &[f64; 3]) -> [f64; 3] {
        let o = crate::refine::level_origin(self.case.grid.origin, self.u.dx, l);
        let dxl = self.u.dx / (1u64 << l) as f64;
        [0, 1, 2].map(|a| (x[a] - o[a]) / dxl)
    }

    /// the valve records at time t
    pub fn valve_records(&self, t: f64) -> Vec<f32> {
        self.valve_records_at(t, 0)
    }

    /// the valve records at time t in the lattice units of refinement level l
    pub fn valve_records_at(&self, t: f64, l: usize) -> Vec<f32> {
        let ul = self.u.level(l);
        let m = &self.case.motion;
        let mut out = vec![];
        for (vi, v) in self.case.valves.iter().enumerate() {
            let n0 = v.normal;
            let e1r = normalize({
                let p = dot(&v.e1, &n0);
                [v.e1[0] - p * n0[0], v.e1[1] - p * n0[1], v.e1[2] - p * n0[2]]
            });
            let e2r = cross(&n0, &e1r);
            // the plane fitted to the moved annulus: centre and normal from 16 annulus points (least squares), mean radius
            let fitted = {
                let pts: Vec<[f64; 3]> = (0..16)
                    .map(|k| {
                        let a = std::f64::consts::TAU * k as f64 / 16.0;
                        m.forward(&[0, 1, 2].map(|i| v.center[i] + v.radius * (a.cos() * e1r[i] + a.sin() * e2r[i])), t)
                    })
                    .collect();
                let c: [f64; 3] = [0, 1, 2].map(|i| pts.iter().map(|p| p[i]).sum::<f64>() / 16.0);
                // normal: the sum of the cross products of consecutive radius vectors (area vector of the polygon)
                let mut nv = [0.0; 3];
                for k in 0..16 {
                    let (a, b) = (&pts[k], &pts[(k + 1) % 16]);
                    let cr = cross(&[a[0] - c[0], a[1] - c[1], a[2] - c[2]], &[b[0] - c[0], b[1] - c[1], b[2] - c[2]]);
                    for i in 0..3 {
                        nv[i] += cr[i];
                    }
                }
                let nv = normalize(nv);
                let r = pts.iter().map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt()).sum::<f64>() / 16.0;
                (c, nv, r)
            };
            // tissue frame: the reference-pose disk (the kernel evaluates it at each node's reference position and adds
            // the tissue velocity); rigid: the moved centre, axes from the moved radius vectors, the centre's velocity
            let (c, e1, e2, n, r, uv) = if v.tissue {
                (v.center, e1r, e2r, n0, v.radius, [0.0; 3])
            } else {
                let c = m.forward(&v.center, t);
                let a1 = m.forward(&[0, 1, 2].map(|i| v.center[i] + v.radius * e1r[i]), t);
                let a2 = m.forward(&[0, 1, 2].map(|i| v.center[i] + v.radius * e2r[i]), t);
                let d1 = [a1[0] - c[0], a1[1] - c[1], a1[2] - c[2]];
                let d2 = [a2[0] - c[0], a2[1] - c[1], a2[2] - c[2]];
                let r = 0.5 * (dot(&d1, &d1).sqrt() + dot(&d2, &d2).sqrt());
                let e1 = normalize(d1);
                let p = dot(&d2, &e1);
                let e2 = normalize([d2[0] - p * e1[0], d2[1] - p * e1[1], d2[2] - p * e1[2]]);
                let n = cross(&e1, &e2);
                (c, e1, e2, n, r, m.velocity(&v.center, t).map(|x| x * self.u.vscale))
            };
            let os = v.opening.at(t).clamp(0.0, 1.0);
            let (o, dodt) = if v.trigger.enabled && self.o_dyn[vi] > os { (self.o_dyn[vi], self.o_rate[vi]) } else { (os, v.opening.deriv(t, 1e-3)) };
            let cl = self.lat_at(l, &c);
            let rl = r / ul.dx;
            let mut rec = [0f32; VALVE_STRIDE];
            for a in 0..3 {
                rec[a] = cl[a] as f32;
                rec[3 + a] = n[a] as f32;
                rec[6 + a] = e1[a] as f32;
                rec[9 + a] = e2[a] as f32;
                rec[19 + a] = uv[a] as f32;
            }
            rec[12] = rl as f32;
            rec[13] = (rl + v.ring / ul.dx) as f32;
            rec[14] = (v.thickness / ul.dx) as f32;
            rec[16] = o as f32;
            rec[17] = (v.theta_open * o) as f32;
            // the leaflets' own rotation enters the wall velocity only × push (default 0: quasi-static leaflets). A
            // prescribed swing pumps its swept volume through the valve: in a sealed tube a 30 ms opening built −7 % /
            // +9 % density across it, in the heart +6 % in the RV and a 6 m/s backflow at tricuspid opening
            rec[18] = if o > 0.0 && o < 1.0 { (v.push * v.theta_open * dodt * ul.dt) as f32 } else { 0.0 };
            rec[25] = (2.3 * rl + v.ring / ul.dx + v.thickness / ul.dx + 3.0) as f32;
            rec[26] = v.tissue as i32 as f32;
            rec[38] = v.membrane as i32 as f32;
            if let Some(hl) = v.hole {
                for i in 0..4 {
                    rec[34 + i] = hl[i] as f32;
                }
            }
            if v.tissue {
                let (fc, fnrm, fr) = fitted;
                let fl = self.lat_at(l, &fc);
                for a in 0..3 {
                    rec[27 + a] = fl[a] as f32;
                    rec[30 + a] = fnrm[a] as f32;
                }
                rec[33] = ((fr + 3e-3) / ul.dx) as f32;
                // the reach must cover the rigid disk around the reference centre too
                rec[25] = rec[25].max((2.3 * fr / ul.dx + 12.0) as f32);
            }
            match &v.kind {
                ValveKind::Orifice { hole } => {
                    rec[15] = 0.0;
                    rec[23] = hole[0] as f32;
                    rec[24] = hole[1] as f32;
                }
                ValveKind::Leaflets { .. } => {
                    rec[15] = 1.0;
                    let ls = &self.leaflets[vi];
                    rec[22] = ls.len() as f32;
                    for (k, l) in ls.iter().enumerate() {
                        let b = 64 + 16 * k;
                        rec[b] = (l.pa[0] * rl) as f32;
                        rec[b + 1] = (l.pa[1] * rl) as f32;
                        rec[b + 2] = l.d[0] as f32;
                        rec[b + 3] = l.d[1] as f32;
                        rec[b + 4] = l.m0[0] as f32;
                        rec[b + 5] = l.m0[1] as f32;
                        rec[b + 6] = l.poly.len() as f32;
                        for (j, q) in l.poly.iter().enumerate() {
                            rec[b + 8 + 2 * j] = (q[0] * rl) as f32;
                            rec[b + 9 + 2 * j] = (q[1] * rl) as f32;
                        }
                    }
                }
            }
            out.extend_from_slice(&rec);
        }
        out
    }

    /// the cut records at time t (uses the Windkessel state and the last measured outflow)
    pub fn cut_records(&self, t: f64) -> Vec<f32> {
        self.cut_records_at(t, 0)
    }

    /// the cut records in the lattice units of refinement level l
    pub fn cut_records_at(&self, t: f64, l: usize) -> Vec<f32> {
        let ul = self.u.level(l);
        let m = &self.case.motion;
        let mut out = vec![];
        for (ci, c) in self.case.cuts.iter().enumerate() {
            let p = m.forward(&c.center, t);
            let q = m.forward(&[0, 1, 2].map(|i| c.center[i] + c.radius * c.normal[i]), t);
            let n = normalize([q[0] - p[0], q[1] - p[1], q[2] - p[2]]);
            // lateral scale of the radius: a direction perpendicular to the normal
            let a = if c.normal[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let e = normalize(cross(&c.normal, &a));
            let s = m.forward(&[0, 1, 2].map(|i| c.center[i] + c.radius * e[i]), t);
            let r = dot(&[s[0] - p[0], s[1] - p[1], s[2] - p[2]], &[s[0] - p[0], s[1] - p[1], s[2] - p[2]]).sqrt();
            let mv = m.velocity(&c.center, t).map(|x| x * self.u.vscale);
            let pl = self.lat_at(l, &p);
            let mut rec = [0f32; CUT_STRIDE];
            for k in 0..3 {
                rec[k] = pl[k] as f32;
                rec[3 + k] = n[k] as f32;
            }
            rec[6] = (r / ul.dx) as f32;
            rec[7] = (c.slab / ul.dx) as f32;
            rec[13] = (c.sponge / ul.dx) as f32;
            // the sponge's added τ keeps its physical viscosity on every level
            rec[14] = (c.sponge_tau * (1u64 << l) as f64) as f32;
            for k in 0..3 {
                rec[16 + k] = mv[k] as f32;
            }
            let cref = self.lat_at(l, &c.center);
            for k in 0..3 {
                rec[24 + k] = cref[k] as f32;
                rec[27 + k] = c.normal[k] as f32;
            }
            rec[30] = (c.radius / ul.dx) as f32;
            // the entering relative M_z (spin history; 1 = fresh)
            rec[19] = self.s_in[ci] as f32;
            match &c.kind {
                CutKind::Inlet { q } => {
                    // plug over the measured section (the slab's node layer, not π r²: the discrete lumen differs)
                    let um = self.kappa[ci] * (q.at(t) + self.dq[ci]) / self.area[ci].max(1e-9) * self.u.vscale;
                    rec[8] = 0.0;
                    rec[9] = 1.0;
                    for k in 0..3 {
                        rec[10 + k] = (mv[k] - n[k] * um) as f32;
                    }
                }
                CutKind::Pressure { p } => {
                    rec[8] = 1.0;
                    rec[9] = (1.0 + self.u.p_to_drho(*p + self.p_trim[ci])) as f32;
                }
                CutKind::Windkessel { r1, .. } => {
                    rec[8] = 1.0;
                    rec[9] = (1.0 + self.u.p_to_drho(r1 * self.q_out[ci] + self.wk_pc[ci])) as f32;
                }
            }
            out.extend_from_slice(&rec);
        }
        out
    }

    pub fn geom_params(&self, first: bool) -> GeomParams {
        self.geom_params_at(first, 0)
    }

    /// the topology of refinement level l
    pub fn level_topo(&self, l: usize) -> &Topology {
        if l == 0 { &self.topo } else { &self.fine[l - 1].topo }
    }

    pub fn geom_params_at(&self, first: bool, l: usize) -> GeomParams {
        let (g, s, m) = (&self.case.grid, &self.case.sdf, &self.case.motion);
        let o = crate::refine::level_origin(g.origin, g.dx, l);
        GeomParams {
            o: o.map(|x| x as f32),
            dx: (g.dx / (1u64 << l) as f64) as f32,
            mo: m.origin.map(|x| x as f32),
            mh: m.h as f32,
            mn: m.dims.map(|x| x as i32),
            so: s.origin.map(|x| x as f32),
            sh: s.h as f32,
            sn: s.dims.map(|x| x as i32),
            vscale: self.u.vscale as f32,
            iters: if first { 10 } else { self.case.run.inv_iters as i32 },
            first: first as i32,
            nvalves: self.case.valves.len() as i32,
            ncuts: self.case.cuts.len() as i32,
            ntiles: self.level_topo(l).ntiles as i32,
            regions: self.case.regions.is_some() as i32,
            ntrans: self.case.transpiration.as_ref().map_or(0, |tr| tr.groups.iter().map(|g| g.0 as i32).max().unwrap_or(0)),
            tband: self.case.transpiration.as_ref().map_or(0.0, |tr| tr.band as f32),
            tq: self.transpiration_q(self.time()).map(|q| (q * self.u.vscale) as f32),
            vq: self.case.run.valve_q as i32,
        }
    }

    /// storage flux q_w(t) (m³/s) per transpiration group, index = id − 1 (zeros without transpiration)
    pub fn transpiration_q(&self, t: f64) -> [f64; 8] {
        let mut q = [0.0; 8];
        if let Some(tr) = &self.case.transpiration {
            for (id, _, tab) in &tr.groups {
                q[*id as usize - 1] = tab.at(t);
            }
        }
        q
    }

    pub fn step_params(&self) -> StepParams {
        self.step_params_at(0)
    }

    pub fn step_params_at(&self, l: usize) -> StepParams {
        StepParams {
            ntiles: self.level_topo(l).ntiles as i32,
            tau0: self.u.level(l).tau0 as f32,
            cwale: self.case.run.cwale as f32,
            uclip: self.case.run.uclip as f32,
            ncuts: self.case.cuts.len() as i32,
            fct: self.case.run.out_factor as i32,
            vel_out: (self.u.dx / self.u.dt) as f32,
            // the bulk relaxation keeps its physical viscosity on every level (τ_b − ½ doubles per level, as τ0 − ½;
            // the unscaled interface copy of the trace is then consistent like the shear part's)
            tau_b: if self.case.run.tau_bulk > 0.5 { ((self.case.run.tau_bulk - 0.5) * (1u64 << l) as f64 + 0.5) as f32 } else { self.case.run.tau_bulk as f32 },
            stress_out: (self.case.rho * (self.u.dx / self.u.dt).powi(2)) as f32,
            wmask: self.wmask as i32,
            lshift: l as i32,
            // the direct fine → coarse rescale of the ghosts' packets: unstable at τ ≲ 0.52 (lbm_core.h node_coalesce),
            // experiments only (HASTY_CFD_RESCALE_GHOST)
            rghost: if l > 0 && std::env::var("HASTY_CFD_RESCALE_GHOST").is_ok() { 1.0 / self.rescale(l - 1) } else { 1.0 },
        }
    }

    /// the Dupuis–Chopard non-equilibrium factor from level l to level l + 1 (kernels/lbm_core.h "grid refinement"):
    /// the post-collision part scales as (τ − 1) dt, so fine = coarse × (τ_f − 1) / (2 (τ_c − 1)) (RunSpec::refine_neq
    /// "dc" and "layers" use it; the default copy does not)
    pub fn rescale(&self, l: usize) -> f32 {
        let (tc, tf) = (self.u.level(l).tau0, self.u.level(l + 1).tau0);
        ((tf - 1.0) / (2.0 * (tc - 1.0))) as f32
    }

    /// the coarse → fine non-equilibrium scales [first fine step, second] of RunSpec::refine_neq
    pub fn explode_scales(&self, l: usize) -> [f32; 2] {
        let (tc, tf) = (self.u.level(l).tau0, self.u.level(l + 1).tau0);
        match self.case.run.refine_neq.as_str() {
            "layers" => [((tf - 1.0) / (2.0 * (tc - 1.0))) as f32, ((tf - 2.0) / (2.0 * (tc - 1.0))) as f32],
            "dc" => [self.rescale(l); 2],
            _ => [1.0, 1.0],
        }
    }

    /// fine → coarse ("dc" only): the post-collision stress the fine packets lack on level l, Π*_c − Π*_f = ρ kfd S
    /// (S the strain rate in level-l lattice units), kfd = −c_s² (2 (τ_c − 1) − (τ_f − 1)); 0 otherwise
    pub fn stress_gap(&self, l: usize) -> f32 {
        if self.case.run.refine_neq != "dc" || std::env::var("HASTY_CFD_RESCALE_GHOST").is_ok() || std::env::var("HASTY_CFD_NO_FD").is_ok() {
            return 0.0;
        }
        let (tc, tf) = (self.u.level(l).tau0, self.u.level(l + 1).tau0);
        (-(2.0 * (tc - 1.0) - (tf - 1.0)) / 3.0) as f32
    }

    /// coarse level l and the next finer level l + 1
    fn pair(&mut self, l: usize) -> (&mut B, &mut B) {
        if l == 0 {
            (&mut self.be, &mut self.fine[0].be)
        } else {
            let (a, b) = self.fine.split_at_mut(l);
            (&mut a[l - 1].be, &mut b[0].be)
        }
    }

    /// the geometry update of a refined case: the motion field, then pass 1 of every level, the ghosts masked by their
    /// coarse owners, pass 2 of every level; the wall mass (RunSpec::wall_mass / wall_pump) summed over the levels by
    /// volume (8^−l per node) and corrected as one density per region
    fn update_geometry_levels(&mut self, first: bool, refill: bool) -> Result<()> {
        let t = self.time();
        let mp = self.case.motion.params(t);
        self.be.motion(&mp)?;
        for lv in &mut self.fine {
            lv.be.motion(&mp)?;
        }
        if first {
            self.wm_last_t = Some(t);
        }
        let nl = self.fine.len();
        let gps: Vec<GeomParams> = (0..=nl).map(|l| self.geom_params_at(first, l)).collect();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..=nl).map(|l| (self.valve_records_at(t, l), self.cut_records_at(t, l))).collect();
        self.be.geometry1(&gps[0], &recs[0].0, &recs[0].1)?;
        for (k, lv) in self.fine.iter_mut().enumerate() {
            lv.be.geometry1(&gps[k + 1], &recs[k + 1].0, &recs[k + 1].1)?;
        }
        for l in 0..nl {
            let (c, f) = self.pair(l);
            f.ghost_mask(c)?;
        }
        self.be.geometry2(&gps[0], refill)?;
        for (k, lv) in self.fine.iter_mut().enumerate() {
            lv.be.geometry2(&gps[k + 1], refill)?;
        }
        if self.wmask != 0 && refill && !first {
            let mr = crate::params::MAX_REGIONS;
            let pump = self.case.run.wall_pump && !self.case.wall_pump.is_empty();
            let target = if pump {
                let mut tg = vec![0f32; mr];
                let t1 = self.time();
                let t0 = self.wm_last_t.unwrap_or(t1);
                for (id, tab) in &self.case.wall_pump {
                    let dv = -tab.at(0.5 * (t0 + t1)) * (t1 - t0) / self.u.dx.powi(3) * self.pump_gain[*id as usize];
                    tg[*id as usize] = dv as f32;
                }
                Some(tg)
            } else {
                None
            };
            self.wm_last_t = Some(self.time());
            let mut sums = self.be.wall_mass_sums(false)?;
            for lv in &mut self.fine {
                let w = 0.125f32.powi(lv.l as i32);
                let e = lv.be.wall_mass_sums(false)?;
                for k in 0..2 * mr {
                    sums[k] += w * e[k];
                }
            }
            let corr: Vec<f32> = (0..mr).map(|r| if sums[mr + r] > 0.0 && (self.wmask >> r) & 1 == 1 { (target.as_ref().map_or(0.0, |t| t[r]) - sums[r]) / sums[mr + r] } else { 0.0 }).collect();
            self.be.wall_mass_correct(&corr)?;
            for lv in &mut self.fine {
                lv.be.wall_mass_correct(&corr)?;
            }
        }
        Ok(())
    }

    /// one coarse step of every level: level l explodes into level l + 1, which takes two steps (recursively), then
    /// restricts and coalesces back, then level l steps (kernels/lbm_core.h "grid refinement"); without refinement the
    /// level-0 step and its scalar, as before
    fn step_level(&mut self, l: usize, sp: &[StepParams], spp: &[Option<SpinParams>]) -> Result<()> {
        if l < self.fine.len() {
            let r = self.explode_scales(l);
            {
                let (c, f) = self.pair(l);
                f.explode(c, r, std::env::var("HASTY_CFD_EXPLODE_CONST").is_err())?;
            }
            for k in 0..2 {
                self.fine[l].be.iflux(k == 0)?;
                self.step_level(l + 1, sp, spp)?;
            }
            let kfd = self.stress_gap(l);
            let (c, f) = self.pair(l);
            f.restrict_to(c)?;
            f.coalesce_to(c, kfd)?;
        }
        let b = if l == 0 { &mut self.be } else { &mut self.fine[l - 1].be };
        b.step(&sp[l])?;
        b.pace();
        if let Some(ss) = &spp[l] {
            b.scalar(ss)?;
        }
        Ok(())
    }

    /// the step and spin parameters of every level
    fn level_params(&self) -> (Vec<StepParams>, Vec<Option<SpinParams>>) {
        let nl = self.fine.len();
        ((0..=nl).map(|l| self.step_params_at(l)).collect(), (0..=nl).map(|l| self.spin_params_at(l)).collect())
    }

    /// the latest velocity of every level into the level-0 output accumulator
    pub fn accumulate_all(&mut self) -> Result<()> {
        let (sp, _) = self.level_params();
        self.be.accumulate(&sp[0])?;
        for lv in &self.fine {
            lv.be.accumulate_to(&mut self.be, &sp[lv.l])?;
        }
        Ok(())
    }

    /// the diagnostics of every level in level-0 units: per level l the area-type slots (cut fluxes and layers, the
    /// spin flux, the link mass fluxes: lattice mass per step) × 4^−l, the volume-type ones (probes, regions, density
    /// sums, node counts) × 8^−l, max |u| the maximum, refills summed
    fn diag_all(&mut self, nc: usize, np: usize, nr: usize, light: bool) -> Result<Vec<f32>> {
        let mut d = self.be.diag(nc, np, nr, light)?;
        let kg = 4 * nc + 2 * np + 2 * nr;
        if !light {
            self.node_updates = d[kg + 1] as f64;
        }
        for lv in &mut self.fine {
            let e = lv.be.diag(nc, np, nr, light)?;
            let (s2, s3) = (0.25f32.powi(lv.l as i32), 0.125f32.powi(lv.l as i32));
            if !light {
                self.node_updates += e[kg + 1] as f64 * (1u64 << lv.l) as f64;
            }
            for k in 0..d.len().min(e.len()) {
                let sc = if k < 4 * nc {
                    if k % 4 == 2 { s3 } else { s2 }
                } else if k <= kg + 1 {
                    s3
                } else if k == kg + 2 {
                    d[k] = d[k].max(e[k]);
                    continue;
                } else if k == kg + 3 {
                    1.0
                } else {
                    s2
                };
                d[k] += sc * e[k];
            }
        }
        Ok(d)
    }

    /// motion, valves and cuts at the current time, then the geometry pass
    pub fn update_geometry(&mut self, first: bool, refill: bool) -> Result<()> {
        if !self.fine.is_empty() {
            return self.update_geometry_levels(first, refill);
        }
        let t = self.time();
        self.be.motion(&self.case.motion.params(t))?;
        let gp = self.geom_params(first);
        if first {
            self.wm_last_t = Some(t);
        }
        let (v, c) = (self.valve_records(t), self.cut_records(t));
        // diagnostics (HASTY_CFD_WM_DEBUG): the bounce-back share of the wall mass, before this update's cover / refill
        let wm_dbg = self.wmask != 0 && refill && !first && std::env::var("HASTY_CFD_WM_DEBUG").is_ok() && (self.step < 12 || self.step % 200 == 0);
        let bb = if wm_dbg { Some(self.be.wall_mass_peek()?) } else { None };
        self.be.geometry(&gp, &v, &c, refill)?;
        // the wall mass of the steps since the last update and of this update's cover / refill (not at start-up, where
        // every node is filled)
        if self.wmask != 0 && refill && !first {
            // HASTY_CFD_WM_NOAPPLY (diagnostics): measure the wall mass, apply nothing
            let sums = if std::env::var("HASTY_CFD_WM_NOAPPLY").is_ok() {
                let s_ = self.be.wall_mass_peek()?;
                self.be.wall_mass_clear()?;
                s_
            } else {
                // pump mode: the target per chamber is the model's −dV/dt over the steps since the last update (lattice mass)
                let pump = self.case.run.wall_pump && !self.case.wall_pump.is_empty();
                let target = if pump {
                    let mut tg = vec![0f32; crate::params::MAX_REGIONS];
                    let t1 = self.time();
                    let t0 = self.wm_last_t.unwrap_or(t1);
                    for (id, tab) in &self.case.wall_pump {
                        let dv = -tab.at(0.5 * (t0 + t1)) * (t1 - t0) / self.u.dx.powi(3) * self.pump_gain[*id as usize];
                        tg[*id as usize] = dv as f32;
                    }
                    Some(tg)
                } else {
                    None
                };
                self.wm_last_t = Some(self.time());
                if let (Some(tg), true) = (&target, bb.is_some()) {
                    let v: Vec<String> = (0..crate::params::MAX_REGIONS).filter(|r| (self.wmask >> r) & 1 == 1).map(|r| format!("r{r} target {:.3e}", tg[r])).collect();
                    eprintln!("wall pump step {}: {}", self.step, v.join(", "));
                }
                self.be.wall_mass_apply(target.as_deref())?
            };
            if let Some(bb) = bb {
                let mr = crate::params::MAX_REGIONS;
                let v: Vec<String> = (0..mr).filter(|r| (self.wmask >> r) & 1 == 1).map(|r| format!("r{r}: Σ {:.3e} (bounce-back {:.3e}, cover/refill {:.3e}) over {}", sums[r], bb[r], sums[r] - bb[r], sums[mr + r])).collect();
                eprintln!("wall mass step {}: {}", self.step, v.join(", "));
            }
        }
        Ok(())
    }

    /// start-up: geometry from a cold inverse, all fluid nodes at equilibrium with the wall velocity
    pub fn init(&mut self) -> Result<()> {
        self.update_geometry(true, true)?;
        self.sync_hidden()
    }

    /// ρ, u (and s) of the coarse hidden nodes next to owned ones from their fine cells, finest pair first (start-up,
    /// resume, after populations are set from outside)
    pub fn sync_hidden(&mut self) -> Result<()> {
        for l in (0..self.fine.len()).rev() {
            let (c, f) = self.pair(l);
            f.restrict_to(c)?;
        }
        Ok(())
    }

    /// `n` steps with geometry updates (no accumulation, no logging): tests and warm-up
    pub fn advance(&mut self, n: u64) -> Result<()> {
        let (sp, spp) = self.level_params();
        for _ in 0..n {
            if self.step % self.case.run.geom_every as u64 == 0 && self.step > 0 {
                self.update_geometry(false, true)?;
            }
            self.step_level(0, &sp, &spp)?;
            self.step += 1;
            if self.step % self.case.run.fb_every as u64 == 0 {
                self.feedback(self.case.run.fb_every as u64)?;
            }
        }
        Ok(())
    }

    /// diagnostics: mean density deviation, max speed (m/s), per-cut outward flux (m³/s) and node count
    pub fn diagnostics(&mut self) -> Result<serde_json::Value> {
        let nc = self.case.cuts.len();
        let np = 2 * self.case.valves.len();
        let nr = self.nreg();
        let d = self.diag_all(nc, np, nr, false)?;
        let k = 4 * nc + 2 * np + 2 * nr;
        let cnt = d[k + 1].max(1.0) as f64;
        let mut cuts = vec![];
        for (ci, c) in self.case.cuts.iter().enumerate() {
            // flux through the first interior layer in front of the plane: Σ (u − u_cut)·n × dx²; that layer's section
            let th = self.layer_thickness(ci);
            let q = d[4 * ci] as f64 * (self.u.dx / self.u.dt) * self.u.dx * self.u.dx / th;
            if d[4 * ci + 1] > 0.0 {
                self.area[ci] = d[4 * ci + 1] as f64 * self.u.dx * self.u.dx / th;
                self.p_cut[ci] = self.u.drho_to_p(d[4 * ci + 3] as f64 / d[4 * ci + 1] as f64);
            }
            self.q_out[ci] = q;
            // the exact mass through the cut's slab interface (D3Q19 link populations, node_diag tail), as a volume flux
            let ql = d[k + 4 + 2 * nc + ci] as f64 * self.u.dx.powi(3) / self.u.dt;
            self.q_link[ci] = ql;
            if let CutKind::Inlet { q: tab } = &c.kind {
                let want = tab.at(self.time()) + self.dq[ci];
                let qm = if self.case.run.kappa_link { ql } else { q };
                // damped multiplicative correction while the inflow is clearly non-zero
                if want.abs() > 1e-7 && (-qm) * want > 0.0 {
                    self.kappa[ci] = (self.kappa[ci] * (want / -qm).powf(0.5)).clamp(0.7, 1.6);
                }
            }
            let target = match (&c.kind, &c.q_ref) {
                (CutKind::Inlet { q: tab }, _) => Some(-tab.at(self.time())),
                (_, Some(tab)) => Some(tab.at(self.time())),
                _ => None,
            };
            cuts.push(json!({"name": c.name, "q_out_ml_s": q * 1e6, "q_link_ml_s": ql * 1e6, "target_ml_s": target.map(|x| x * 1e6), "nodes": d[4 * ci + 2], "p_cut_mmHg": self.p_cut[ci] / 133.322, "dq_ml_s": if c.inertance > 0.0 { Some(self.dq[ci] * 1e6) } else { None }, "section_cm2": self.area[ci] * 1e4, "kappa": self.kappa[ci], "s_in": if self.case.spin.is_some() { Some(self.s_in[ci]) } else { None }, "s_out": if self.case.spin.is_some() && self.s_flux[ci][1].abs() > 1e-12 { Some(self.s_flux[ci][0] / self.s_flux[ci][1]) } else { None },
                             "p_mmHg": match c.kind { CutKind::Windkessel { r1, .. } => Some((r1 * q + self.wk_pc[ci]) / 133.322), _ => None }}));
        }
        self.share_trim();
        self.pump_gain_update();
        for (ci, c) in cuts.iter_mut().enumerate() {
            if self.case.cuts[ci].share.is_some() {
                c["p_trim_pa"] = json!(self.p_trim[ci]);
            }
        }
        let vmax = d[k + 2] as f64 * self.u.dx / self.u.dt;
        let t = self.time();
        let valves: Vec<serde_json::Value> = self
            .case
            .valves
            .iter()
            .enumerate()
            .map(|(vi, v)| {
                let p = [0, 1].map(|j| {
                    let (s_, c_) = (d[4 * nc + 4 * vi + 2 * j] as f64, d[4 * nc + 4 * vi + 2 * j + 1] as f64);
                    if c_ >= 1.0 { Some(self.u.drho_to_p(s_ / c_) / 133.322) } else { None }
                });
                let os = v.opening.at(t).clamp(0.0, 1.0);
                // the exact link flux through the material valve plane (node_diag; downstream positive)
                let ql = d.get(k + 4 + 3 * nc + vi).map(|x| *x as f64 * self.u.dx.powi(3) / self.u.dt * 1e6);
                json!({"name": v.name, "q_link_ml_s": ql, "p_up_mmHg": p[0], "p_down_mmHg": p[1], "dp_filtered_mmHg": self.dp_f[vi] / 133.322,
                       "open_schedule": os, "open": if v.trigger.enabled { os.max(self.o_dyn[vi]) } else { os }})
            })
            .collect();
        let mut regions = serde_json::Map::new();
        if let Some(r) = &self.case.regions {
            for (&id, name) in &r.names {
                let i = id as usize;
                if i < nr {
                    let (c_, s_) = (d[4 * nc + 2 * np + 2 * i] as f64, d[4 * nc + 2 * np + 2 * i + 1] as f64);
                    regions.insert(name.clone(), json!({"id": id, "volume_ml": c_ * self.u.dx.powi(3) * 1e6, "drho": if c_ > 0.0 { s_ / c_ } else { 0.0 }}));
                }
            }
        }
        // the wall transpiration's prescribed storage flux per group (mL/s, outward = stored): with it the pressure
        // outlets' q_out should match their q_ref targets up to the residual
        let transp: serde_json::Map<String, serde_json::Value> = self.case.transpiration.as_ref().map_or(Default::default(), |tr| {
            tr.groups.iter().map(|(_, name, tab)| (name.clone(), json!(tab.at(t) * 1e6))).collect()
        });
        Ok(json!({"step": self.step, "t": t, "drho_mean": d[k] as f64 / cnt, "fluid_nodes": d[k + 1], "node_updates": if self.fine.is_empty() { None } else { Some(self.node_updates) }, "u_max_lattice": d[k + 2], "v_max_m_s": vmax, "cuts": cuts, "valves": valves, "regions": regions, "transpiration_ml_s": transp, "pump_gain": self.case.regions.as_ref().map(|r| r.names.iter().filter(|(k, n)| n.starts_with("blood_") && (**k as usize) < self.pump_gain.len()).map(|(k, n)| (n.clone(), json!(self.pump_gain[*k as usize]))).collect::<serde_json::Map<_, _>>())}))
    }

    /// wall pump gain (RunSpec::wall_pump): per side, the venous inlets' exact inflow (link flux) against their model
    /// targets over each cycle; the inertial inlets deliver what the chambers draw, so the ratio is the chambers'
    /// throughput error (the 0.5 mm heart over-pumped 9 % with the bounce-back corrected to −dV/dt alone: another
    /// resolution-dependent mass source remains). The side's chambers' pump targets × (want / measured)^0.7 per cycle,
    /// from the second cycle on, within 0.7…1.5
    fn pump_gain_update(&mut self) {
        if !(self.case.run.wall_pump && !self.case.wall_pump.is_empty()) {
            return;
        }
        let t = self.time();
        let dt = match self.pg_t.replace(t) {
            Some(t0) if t > t0 => t - t0,
            _ => return,
        };
        for (ci, c) in self.case.cuts.iter().enumerate() {
            if let (Some((g, _)), CutKind::Inlet { q: tab }) = (&c.share, &c.kind) {
                let side = match g.as_str() {
                    "pulmonary_venous" => 0,
                    "systemic_venous" => 1,
                    _ => continue,
                };
                self.pg_meas[side] += -self.q_link[ci] * dt;
                self.pg_want[side] += tab.at(t) * dt;
            }
        }
        let cyc = (t / self.u.period).floor() as i64;
        if cyc > self.pg_cycle {
            if self.pg_cycle >= 1 {
                let names: Vec<(u16, String)> = self.case.regions.as_ref().map_or(vec![], |r| r.names.iter().map(|(k, v)| (*k, v.clone())).collect());
                for side in 0..2 {
                    if self.pg_meas[side] > 1e-9 && self.pg_want[side] > 1e-9 {
                        let f = (self.pg_want[side] / self.pg_meas[side]).powf(0.7);
                        let chambers: [&str; 2] = if side == 0 { ["blood_LA", "blood_LV"] } else { ["blood_RA", "blood_RV"] };
                        for (id, nm) in &names {
                            if chambers.contains(&nm.as_str()) && (*id as usize) < self.pump_gain.len() {
                                self.pump_gain[*id as usize] = (self.pump_gain[*id as usize] * f).clamp(0.7, 1.5);
                            }
                        }
                    }
                }
            }
            self.pg_meas = [0.0; 2];
            self.pg_want = [0.0; 2];
            self.pg_cycle = cyc;
        }
    }

    /// pressure cuts with a share: inflow moving average over about a cycle, then an integral trim of each cut's
    /// pressure towards its fraction of the group's inflow. The gain is scaled by the cut's dynamic pressure
    /// ½ρū² (a relative flow error e needs ≈ −2·½ρū²·e: Bernoulli), half of that per cycle; zero mean per group,
    /// |trim| ≤ 5 mmHg, from half a cycle on (the average needs a start)
    fn share_trim(&mut self) {
        let t = self.time();
        let dtd = match self.last_diag_t.replace(t) {
            Some(t0) if t > t0 => t - t0,
            _ => return,
        };
        let a = (dtd / self.u.period).min(1.0);
        let mut groups: std::collections::BTreeMap<String, Vec<(usize, f64)>> = Default::default();
        for (ci, c) in self.case.cuts.iter().enumerate() {
            if let (Some((g, f)), CutKind::Pressure { .. }) = (&c.share, &c.kind) {
                let qin = -self.q_out[ci];
                self.q_ema[ci] = if self.q_ema[ci].is_finite() { (1.0 - a) * self.q_ema[ci] + a * qin } else { qin };
                groups.entry(g.clone()).or_default().push((ci, *f));
            }
        }
        if t < 0.5 * self.u.period {
            return;
        }
        let rho = self.case.rho;
        for (_, m) in groups {
            let tot: f64 = m.iter().map(|&(ci, _)| self.q_ema[ci]).sum();
            let fsum: f64 = m.iter().map(|&(_, f)| f).sum::<f64>().max(1e-12);
            if tot.abs() < 1e-8 {
                continue;
            }
            for &(ci, f) in &m {
                let want = tot * f / fsum;
                let ubar = want.abs() / (std::f64::consts::PI * self.case.cuts[ci].radius.powi(2));
                let pdyn = (0.5 * rho * ubar * ubar).max(5.0);
                let e = (self.q_ema[ci] - want) / want.abs().max(1e-9);
                self.p_trim[ci] -= pdyn * e.clamp(-1.0, 1.0) * a;
            }
            let mean = m.iter().map(|&(ci, _)| self.p_trim[ci]).sum::<f64>() / m.len() as f64;
            for &(ci, _) in &m {
                self.p_trim[ci] = (self.p_trim[ci] - mean).clamp(-666.0, 666.0);
            }
        }
    }

    /// the feedback every `fb_every` steps (light diagnostics): inertial inlets and pressure-triggered valves
    pub fn feedback(&mut self, steps: u64) -> Result<()> {
        let inl = self.case.cuts.iter().any(|c| c.inertance > 0.0);
        let val = self.case.valves.iter().any(|v| v.trigger.enabled);
        let spin = self.case.spin.is_some();
        if !inl && !val && !spin {
            return Ok(());
        }
        let (nc, np) = (self.case.cuts.len(), 2 * self.case.valves.len());
        let d = self.diag_all(nc, np, self.nreg(), true)?;
        let dt = steps as f64 * self.u.dt;
        if spin {
            let k = 4 * nc + 2 * np + 2 * self.nreg() + 4;
            self.spin_feedback(&d[k..k + 2 * nc], dt);
        }
        if val {
            self.valve_feedback(&d[4 * nc..4 * nc + 2 * np], dt);
        }
        if inl {
            self.inlet_feedback(&d, dt);
        }
        Ok(())
    }

    /// the entering s of the cuts (spin history): fixed `SpinSpec::s_in`, or along the cut's upstream path, starting
    /// fresh or from the cycle-averaged flux-weighted outflow s of its `from` cuts
    pub fn update_s_in(&mut self) {
        let Some(sp) = &self.case.spin else { return };
        let names: std::collections::HashMap<&str, usize> = self.case.cuts.iter().enumerate().map(|(i, c)| (c.name.as_str(), i)).collect();
        for (ci, c) in self.case.cuts.iter().enumerate() {
            self.s_in[ci] = match &c.upstream {
                None => sp.s_in.get(ci).copied().unwrap_or(1.0),
                Some(u) => {
                    let (mut a, mut b) = (0.0, 0.0);
                    for f in &u.from {
                        if let Some(&j) = names.get(f.as_str()) {
                            a += self.s_flux[j][0];
                            b += self.s_flux[j][1];
                        }
                    }
                    let s0 = if u.from.is_empty() || b.abs() < 1e-12 { 1.0 } else { (a / b).clamp(0.0, 1.0) };
                    sp.along_path(s0, u)
                }
            };
        }
    }

    /// cycle-averaged flux-layer Σ u_n s and Σ u_n of a cut (spin history; lattice units)
    pub fn s_flux_of(&self, ci: usize) -> [f64; 2] {
        self.s_flux[ci]
    }

    fn spin_feedback(&mut self, d: &[f32], dt: f64) {
        // moving averages over one cycle (the outflow s is pulsatile; its flux-weighted cycle mean is what enters a
        // long upstream path)
        let a = (dt / (self.u.steps_per_cycle as f64 * self.u.dt)).min(1.0);
        for ci in 0..self.case.cuts.len() {
            for j in 0..2 {
                self.s_flux[ci][j] += (d[2 * ci + j] as f64 - self.s_flux[ci][j]) * a;
            }
        }
        self.update_s_in();
    }

    /// probe pressures → filtered forward gradient → dynamic opening (see `case::ValveTrigger`)
    fn valve_feedback(&mut self, pr: &[f32], dt: f64) {
        let t = self.time();
        for (vi, v) in self.case.valves.iter().enumerate() {
            let p = [0, 1].map(|j| if pr[4 * vi + 2 * j + 1] >= 1.0 { self.u.drho_to_p(pr[4 * vi + 2 * j] as f64 / pr[4 * vi + 2 * j + 1] as f64) } else { f64::NAN });
            if !(p[0].is_finite() && p[1].is_finite()) {
                continue;
            }
            self.p_probe[vi] = p;
            let tr = &v.trigger;
            if !tr.enabled {
                continue;
            }
            self.dp_f[vi] += (p[0] - p[1] - self.dp_f[vi]) * (dt / tr.tau_filter).min(1.0);
            // hysteresis: a closed valve needs Δp > p_open to start; an opened one keeps opening while any forward
            // gradient pushes flow through it; it closes on a reverse gradient (flow reversal), else holds
            let (o, os) = (self.o_dyn[vi], v.opening.at(t).clamp(0.0, 1.0));
            let dp = self.dp_f[vi];
            // the trigger acts only while one of the valve's two compartments is sealed (no open valve, no cut: the
            // isovolumic phases, from the region graph), or without a region graph within `window` before a
            // scheduled opening
            let allowed = match self.comp.sealed_side(vi, &self.case.valves, t) {
                Some(s) => s,
                None => self.open_edges[vi].iter().map(|&te| (te - t).rem_euclid(v.opening.period)).fold(f64::INFINITY, f64::min) <= tr.window,
            };
            let rate = if os < 0.05 && !allowed {
                -o / tr.tau_close
            } else if dp > tr.p_open {
                (1.0 - o) / tr.tau_open
            } else if dp > 0.0 && o.max(os) > 0.05 {
                (1.0 - o) / tr.tau_open * dp / tr.p_open
            } else if dp < -tr.p_close {
                -o / tr.tau_close
            } else {
                0.0
            };
            self.o_dyn[vi] = (o + rate * dt).clamp(0.0, 1.0);
            self.o_rate[vi] = rate;
        }
    }

    /// inertial inlets: dΔQ/dt = −p_cut / L − ΔQ / relax (grouped per atrium: area-mean pressure, damping R)
    fn inlet_feedback(&mut self, d: &[f32], dt: f64) {
        let mut groups: std::collections::BTreeMap<String, Vec<(usize, f64)>> = Default::default();
        for (ci, c) in self.case.cuts.iter().enumerate() {
            if c.inertance <= 0.0 || !matches!(c.kind, CutKind::Inlet { .. }) {
                continue;
            }
            if d[4 * ci + 1] >= 1.0 {
                self.p_cut[ci] = self.u.drho_to_p(d[4 * ci + 3] as f64 / d[4 * ci + 1] as f64);
            }
            match &c.share {
                Some((g, f)) => groups.entry(g.clone()).or_default().push((ci, *f)),
                None => {
                    let p = self.p_cut[ci];
                    self.dq[ci] = (self.dq[ci] + dt * (-p / c.inertance - self.dq[ci] / c.relax.max(1e-3))).clamp(-1e-3, 1e-3);
                }
            }
        }
        // grouped: one ΔQ (stored as the group total on every member, × fraction when applied)
        for (_, m) in groups {
            let c0 = &self.case.cuts[m[0].0];
            let (mut pa, mut aa) = (0.0, 0.0);
            for &(ci, _) in &m {
                let a = (d[4 * ci + 1] as f64).max(0.0);
                pa += a * self.p_cut[ci];
                aa += a;
            }
            let p = if aa > 0.0 { pa / aa } else { 0.0 };
            let fsum: f64 = m.iter().map(|&(_, f)| f).sum::<f64>().max(1e-12);
            let tot = self.dq[m[0].0] / (m[0].1 / fsum).max(1e-12);
            let tot = (tot + dt * ((-p - c0.resistance * tot) / c0.inertance - tot / c0.relax.max(1e-3))).clamp(-2e-3, 2e-3);
            for &(ci, f) in &m {
                self.dq[ci] = tot * f / fsum;
            }
        }
    }

    fn windkessel(&mut self, dt: f64) {
        for (ci, c) in self.case.cuts.iter().enumerate() {
            if let CutKind::Windkessel { c: cap, r2, .. } = c.kind {
                self.wk_pc[ci] += dt * (self.q_out[ci] - self.wk_pc[ci] / r2) / cap;
            }
        }
    }

    fn checkpoint(&mut self, out: &Path, counts: &[f64]) -> Result<()> {
        let (f, ru) = self.be.save()?;
        let hdr = json!({"step": self.step, "wk_pc": self.wk_pc, "q_out": self.q_out, "kappa": self.kappa, "area": self.area, "p_trim": self.p_trim, "dq": self.dq, "o_dyn": self.o_dyn, "dp_f": self.dp_f, "beat": self.beat_state(out), "q_ema": self.q_ema.iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect::<Vec<_>>(), "s_flux": self.s_flux, "spin": self.case.spin.as_ref().map(|s| s.protocol.clone()), "counts": counts, "nodes": self.be.nodes(), "q": Q, "levels": self.fine.iter().map(|l| l.be.nodes()).collect::<Vec<_>>()}).to_string();
        let tmp = out.join("ckpt.tmp");
        {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            w.write_all(&(hdr.len() as u64).to_le_bytes())?;
            w.write_all(hdr.as_bytes())?;
            for x in f.iter().chain(&ru) {
                w.write_all(&x.to_le_bytes())?;
            }
            if self.case.spin.is_some() {
                for x in self.be.save_scalar()? {
                    w.write_all(&x.to_le_bytes())?;
                }
            }
            // refinement levels after level 0, each as level 0 (populations, ρ u, the scalar)
            for lv in &mut self.fine {
                let (f, ru) = lv.be.save()?;
                for x in f.iter().chain(&ru) {
                    w.write_all(&x.to_le_bytes())?;
                }
                if self.case.spin.is_some() {
                    for x in lv.be.save_scalar()? {
                        w.write_all(&x.to_le_bytes())?;
                    }
                }
            }
            w.flush()?;
        }
        std::fs::rename(tmp, out.join("ckpt.bin"))?;
        Ok(())
    }

    fn resume(&mut self, out: &Path) -> Result<Option<Vec<f64>>> {
        let p = out.join("ckpt.bin");
        if !p.exists() {
            return Ok(None);
        }
        let mut r = std::io::BufReader::new(std::fs::File::open(&p)?);
        let mut b8 = [0u8; 8];
        r.read_exact(&mut b8)?;
        let mut hdr = vec![0u8; u64::from_le_bytes(b8) as usize];
        r.read_exact(&mut hdr)?;
        let h: serde_json::Value = serde_json::from_slice(&hdr).map_err(|e| Error::Case(e.to_string()))?;
        let n = self.be.nodes();
        if h["nodes"].as_u64() != Some(n as u64) {
            return Err(Error::Case("checkpoint of another topology".into()));
        }
        let lvn: Vec<u64> = h["levels"].as_array().map_or(vec![], |a| a.iter().filter_map(|x| x.as_u64()).collect());
        if lvn != self.fine.iter().map(|l| l.be.nodes() as u64).collect::<Vec<_>>() {
            return Err(Error::Case(format!("checkpoint of other refinement levels ({lvn:?} nodes)")));
        }
        // the spin scalar follows the populations only when the run that wrote it had one; a protocol change would
        // silently mix two spin histories
        let ck_spin = h.get("spin").cloned().unwrap_or(serde_json::Value::Null);
        let my_spin = self.case.spin.as_ref().map_or(serde_json::Value::Null, |s| s.protocol.clone());
        if ck_spin.is_null() != my_spin.is_null() || (!my_spin.is_null() && ck_spin != my_spin) {
            return Err(Error::Case(format!("checkpoint spin protocol {ck_spin} differs from this run's {my_spin} (use another out dir)")));
        }
        let mut buf = vec![0u8; (Q + 4) * n * 4];
        r.read_exact(&mut buf)?;
        let v: Vec<f32> = buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        self.step = h["step"].as_u64().unwrap();
        self.wk_pc = h["wk_pc"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        self.q_out = h["q_out"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        if let (Some(k), Some(a)) = (h["kappa"].as_array(), h["area"].as_array()) {
            self.kappa = k.iter().map(|x| x.as_f64().unwrap()).collect();
            self.area = a.iter().map(|x| x.as_f64().unwrap()).collect();
        }
        if let Some(b) = h["beat"].as_array() {
            self.beat_resume = Some((b[0].as_u64().unwrap_or(0), b[1].as_u64().unwrap_or(0)));
        }
        if let Some(x) = h["dq"].as_array() {
            self.dq = x.iter().map(|v| v.as_f64().unwrap()).collect();
        }
        if let (Some(o), Some(p)) = (h["o_dyn"].as_array(), h["dp_f"].as_array()) {
            self.o_dyn = o.iter().map(|v| v.as_f64().unwrap()).collect();
            self.dp_f = p.iter().map(|v| v.as_f64().unwrap()).collect();
        }
        if let (Some(p), Some(q)) = (h["p_trim"].as_array(), h["q_ema"].as_array()) {
            self.p_trim = p.iter().map(|x| x.as_f64().unwrap()).collect();
            self.q_ema = q.iter().map(|x| x.as_f64().unwrap()).collect();
        }
        if let Some(x) = h["s_flux"].as_array() {
            self.s_flux = x.iter().map(|v| [0, 1].map(|j| v[j].as_f64().unwrap_or(0.0))).collect();
            self.update_s_in();
        }
        // flags of the current geometry first (no refill), then the saved populations
        self.update_geometry(true, false)?;
        self.be.restore(&v[..Q * n], &v[Q * n..])?;
        if self.case.spin.is_some() {
            let mut sb = vec![0u8; n * 4];
            if r.read_exact(&mut sb).is_ok() {
                let s: Vec<f32> = sb.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                self.be.restore_scalar(&s)?;
            }
        }
        let spin = self.case.spin.is_some();
        for lv in &mut self.fine {
            let nl = lv.be.nodes();
            let mut buf = vec![0u8; (Q + 4) * nl * 4];
            r.read_exact(&mut buf)?;
            let v: Vec<f32> = buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            lv.be.restore(&v[..Q * nl], &v[Q * nl..])?;
            if spin {
                let mut sb = vec![0u8; nl * 4];
                r.read_exact(&mut sb)?;
                let s: Vec<f32> = sb.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                lv.be.restore_scalar(&s)?;
            }
        }
        self.sync_hidden()?;
        Ok(Some(h["counts"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()))
    }

    fn flush_bin(&mut self, file: &std::fs::File, bin: usize, samples: f64, counts: &mut [f64], beat: Option<(&Path, u64)>) -> Result<()> {
        let acc = self.be.take_acc()?;
        if let Some((dir, cycle)) = beat {
            self.write_beat_bin(dir, cycle, bin, samples, &acc)?;
        }
        let slab = acc.len();
        let off = (bin * slab * 4) as u64;
        let mut buf = vec![0u8; slab * 4];
        file.read_exact_at(&mut buf, off)?;
        for (k, a) in acc.iter().enumerate() {
            let v = f32::from_le_bytes([buf[4 * k], buf[4 * k + 1], buf[4 * k + 2], buf[4 * k + 3]]) + a;
            buf[4 * k..4 * k + 4].copy_from_slice(&v.to_le_bytes());
        }
        file.write_all_at(&buf, off)?;
        counts[bin] += samples;
        Ok(())
    }

    /// one cycle's bin of the beat library: per occupied cell (Σw > 0) the coverage w (fluid fraction over the bin's
    /// samples), the mean velocity, the covariance (xx yy zz xy xz yz), the third central moments (xxx yyy zzz xxy xxz
    /// xyy yyz xzz yzz xyz), the mean wall shear stress vector (0 off the wall) and the spin scalar's mean and standard
    /// deviation and cov(s, u), as f16 [n][28] (m/s, m²/s², m³/s³, Pa, 1, m/s); appended to `<dir>/beat_<cycle>.bin` as a block
    /// [bin u32][n u32][samples f32][cell u32 × n][f16 × 28n]
    fn write_beat_bin(&self, dir: &Path, cycle: u64, bin: usize, samples: f64, acc: &[f32]) -> Result<()> {
        use crate::motion::f32_to_f16;
        let f3 = (self.case.run.out_factor as f64).powi(3);
        let ncell = self.topo.ntiles * self.cells;
        let mut idx: Vec<u32> = vec![];
        let mut vals: Vec<u16> = vec![];
        for c in 0..ncell {
            let a = &acc[c * ACC_STRIDE..(c + 1) * ACC_STRIDE];
            let s0 = a[0] as f64;
            if s0 <= 0.0 {
                continue;
            }
            let m = [a[1] as f64 / s0, a[2] as f64 / s0, a[3] as f64 / s0];
            // E[u_i u_j] by (i, j) from slots 4–9 (xx yy zz xy xz yz)
            let e2 = |i: usize, j: usize| -> f64 {
                let k = match (i.min(j), i.max(j)) {
                    (0, 0) => 4,
                    (1, 1) => 5,
                    (2, 2) => 6,
                    (0, 1) => 7,
                    (0, 2) => 8,
                    _ => 9,
                };
                a[k] as f64 / s0
            };
            let cov = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].map(|(i, j)| e2(i, j) - m[i] * m[j]);
            // slots 15–24: xxx yyy zzz xxy xxz xyy yyz xzz yzz xyz
            let tri = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2), (0, 1, 1), (1, 1, 2), (0, 2, 2), (1, 2, 2), (0, 1, 2)];
            let mut rec = [0f32; 28];
            rec[0] = (s0 / (samples.max(1.0) * f3)) as f32;
            for i in 0..3 {
                rec[1 + i] = m[i] as f32;
            }
            for k in 0..6 {
                rec[4 + k] = cov[k] as f32;
            }
            for (k, &(i, j, l)) in tri.iter().enumerate() {
                let e3 = a[15 + k] as f64 / s0;
                rec[10 + k] = (e3 - m[i] * e2(j, l) - m[j] * e2(i, l) - m[l] * e2(i, j) + 2.0 * m[i] * m[j] * m[l]) as f32;
            }
            if a[10] > 0.0 {
                for i in 0..3 {
                    rec[20 + i] = a[11 + i] / a[10];
                }
            }
            // spin scalar: mean and standard deviation of s = M_z / M0 (1 without a spin history)
            let (sm, s2) = (a[25] as f64 / s0, a[26] as f64 / s0);
            rec[23] = sm as f32;
            rec[24] = (s2 - sm * sm).max(0.0).sqrt() as f32;
            // cov(s, u): with it a sub-source's s follows its velocity (s ≈ E[s] + cov(s,u)ᵀ Σ⁻¹ (u − μ))
            for i in 0..3 {
                rec[25 + i] = (a[27 + i] as f64 / s0 - sm * m[i]) as f32;
            }
            idx.push(c as u32);
            vals.extend(rec.iter().map(|&x| f32_to_f16(x)));
        }
        let p = dir.join(format!("beat_{cycle:04}.bin"));
        let mut w = std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(&p)?);
        w.write_all(&(bin as u32).to_le_bytes())?;
        w.write_all(&(idx.len() as u32).to_le_bytes())?;
        w.write_all(&(samples as f32).to_le_bytes())?;
        for i in &idx {
            w.write_all(&i.to_le_bytes())?;
        }
        for v in &vals {
            w.write_all(&v.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    fn write_meta(&self, out: &Path, counts: &[f64], report: Option<&Report>) -> Result<()> {
        let c = &self.case;
        let meta = json!({
            "format": "flow_bins.f32: f32 [bins][tiles][cells][acc_stride] = Σw, Σw u (3), Σw u uᵀ (xx yy zz xy xz yz), m/s; then at wall nodes Σw_wall, Σ τ_w (3, Pa), Σ |τ_w| (Pa); cell (cz, cy, cx) of a tile covers out_factor³ nodes",
            "acc_stride": ACC_STRIDE,
            "bins": c.run.bins, "period": self.u.period, "cycles": c.run.cycles, "warmup": c.run.warmup, "counts": counts,
            "tiles": self.topo.ntiles, "tile_xyz": self.topo.txyz, "cells_per_tile": self.cells, "out_factor": c.run.out_factor,
            "origin": c.grid.origin, "dx": c.grid.dx, "node_position": "origin + g dx (g = tile·8 + local)",
            "cell_centre": "origin + (tile·8 + (c + ½)·out_factor − ½) dx",
            "units": {"dt": self.u.dt, "steps_per_cycle": self.u.steps_per_cycle, "tau0": self.u.tau0, "nu_lattice": self.u.nu_lat, "vscale": self.u.vscale},
            "valves": c.valves.iter().map(|v| v.name.clone()).collect::<Vec<_>>(), "cuts": c.cuts.iter().map(|x| x.name.clone()).collect::<Vec<_>>(),
            "report": report.map(|r| json!({"steps": r.steps, "seconds": r.seconds, "mlups": r.mlups, "stopped": r.stopped, "last_diag": r.last_diag})),
            "refinement": self.fine.iter().map(|l| json!({"level": l.l, "dx": l.u.dx, "dt": l.u.dt, "tau0": l.u.tau0, "tiles": l.topo.ntiles, "nodes": l.topo.nodes(), "refined_cells": l.cells, "ghosts": l.ghosts, "rim": l.rim, "neq": c.run.refine_neq})).collect::<Vec<_>>(),
        });
        std::fs::write(out.join("flow_bins.json"), serde_json::to_string_pretty(&meta).unwrap())?;
        Ok(())
    }

    /// the full run into `out` (resumes from `out/ckpt.bin`); `stop` is polled every step
    pub fn run(&mut self, out: &Path, stop: &dyn Fn() -> bool, verbose: bool) -> Result<Report> {
        std::fs::create_dir_all(out)?;
        let run = self.case.run.clone();
        let spc = self.u.steps_per_cycle;
        let total = run.cycles as u64 * spc;
        let acc_from = run.warmup as u64 * spc;
        let slab = self.topo.ntiles * self.cells * ACC_STRIDE;
        let fpath = out.join("flow_bins.f32");
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&fpath)?;
        let want = (run.bins * slab * 4) as u64;
        let mut counts = match self.resume(out)? {
            Some(c) => c,
            None => {
                file.set_len(0)?;
                self.init()?;
                vec![0.0; run.bins]
            }
        };
        file.set_len(want)?;
        self.write_meta(out, &counts, None)?;
        let mut log = std::fs::OpenOptions::new().create(true).append(true).open(out.join("log.jsonl"))?;
        let (sp, spp) = self.level_params();
        let bin_of = |s: u64| (((s % spc) as f64 / spc as f64) * run.bins as f64) as usize % run.bins;
        // the beat library (run.beats): one file per accumulated cycle; checkpoints and stops wait for a bin boundary
        // so no bin is split, and a resumed run cuts the current beat file back to its checkpointed length
        let beats_dir = out.join("beats");
        if run.beats {
            std::fs::create_dir_all(&beats_dir)?;
            if let Some((cyc, len)) = self.beat_resume.take() {
                let p = beats_dir.join(format!("beat_{cyc:04}.bin"));
                if p.exists() {
                    std::fs::OpenOptions::new().write(true).open(&p)?.set_len(len)?;
                }
                for e in std::fs::read_dir(&beats_dir)?.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if let Some(c) = n.strip_prefix("beat_").and_then(|x| x.strip_suffix(".bin")).and_then(|x| x.parse::<u64>().ok()) {
                        if c > cyc {
                            std::fs::remove_file(e.path())?;
                        }
                    }
                }
            }
            std::fs::write(beats_dir.join("index.json"), serde_json::to_string_pretty(&json!({
                "format": "beat_<cycle>.bin: blocks [bin u32][n u32][samples f32][cell u32 × n][f16 × nf n] (nf = len(fields)); cell = tile · cells_per_tile + c (layout of flow_bins.json)",
                "fields": ["w", "ux", "uy", "uz", "cxx", "cyy", "czz", "cxy", "cxz", "cyz", "m_xxx", "m_yyy", "m_zzz", "m_xxy", "m_xxz", "m_xyy", "m_yyz", "m_xzz", "m_yzz", "m_xyz", "wss_x", "wss_y", "wss_z", "mz", "mz_std", "c_su_x", "c_su_y", "c_su_z"],
                "spin": self.case.spin.as_ref().map(|s| json!({"tr": s.tr, "t1": s.t1, "center": s.center, "normal": s.normal, "protocol": s.protocol, "s_in_start": self.s_in})),
                "units": "w: fluid fraction of the cell over the bin's samples; velocity moments central (m/s, m²/s², m³/s³); wss Pa",
                "bins": run.bins, "period": self.u.period, "warmup": run.warmup, "cycles": run.cycles,
                "note": "the moments are over the cell's out_factor³ nodes and the bin's time samples of that one cycle",
            })).unwrap())?;
        }
        let beat_of = |cyc: u64| if run.beats { Some((beats_dir.as_path(), cyc)) } else { None };
        let mut cur_bin: Option<(usize, f64, u64)> = None;
        let mut prev_bin = bin_of(self.step);
        let mut pending_ckpt = false;
        let mut pending_stop = false;
        let t0 = Instant::now();
        let s0 = self.step;
        let mut fluid = 0f64;
        let mut last = json!({});
        let mut stopped = false;
        while self.step < total {
            if self.step % run.geom_every as u64 == 0 && self.step > s0 {
                self.update_geometry(false, true)?;
            }
            self.step_level(0, &sp, &spp)?;
            self.step += 1;
            if self.step % run.fb_every as u64 == 0 {
                self.feedback(run.fb_every as u64)?;
            }
            let b_now = bin_of(self.step);
            let boundary = b_now != prev_bin;
            prev_bin = b_now;
            if boundary {
                if let Some((cb, n, cyc)) = cur_bin.take() {
                    self.flush_bin(&file, cb, n, &mut counts, beat_of(cyc))?;
                }
                if pending_ckpt || pending_stop {
                    self.checkpoint(out, &counts)?;
                    self.write_meta(out, &counts, None)?;
                    pending_ckpt = false;
                    if pending_stop {
                        stopped = true;
                        break;
                    }
                }
            }
            if self.step >= acc_from && self.step % run.acc_every as u64 == 0 {
                self.accumulate_all()?;
                cur_bin = Some((b_now, cur_bin.map_or(0.0, |x| x.1) + 1.0, self.step / spc));
            }
            if self.step % run.diag_every as u64 == 0 || self.step == total {
                let d = self.diagnostics()?;
                self.windkessel(run.diag_every as f64 * self.u.dt);
                fluid = if self.fine.is_empty() { d["fluid_nodes"].as_f64().unwrap_or(0.0) } else { self.node_updates };
                let ul = d["u_max_lattice"].as_f64().unwrap_or(f64::NAN);
                writeln!(log, "{}", d)?;
                if verbose {
                    let cs: Vec<String> = d["cuts"].as_array().unwrap().iter().map(|c| format!("{} {:.1}", c["name"].as_str().unwrap_or(""), c["q_out_ml_s"].as_f64().unwrap_or(0.0))).collect();
                    println!(
                        "step {} (cycle {:.2}) Δρ̄ {:.2e} |u|max {:.3} ({:.2} m/s) | out mL/s: {} | {:.0} MLUPS",
                        self.step,
                        self.step as f64 / spc as f64,
                        d["drho_mean"].as_f64().unwrap_or(0.0),
                        ul,
                        d["v_max_m_s"].as_f64().unwrap_or(0.0),
                        cs.join(", "),
                        fluid * (self.step - s0) as f64 / t0.elapsed().as_secs_f64() / 1e6
                    );
                }
                last = d;
                if !ul.is_finite() || ul > 0.45 {
                    return Err(Error::Unstable(format!("|u| = {ul} (lattice) at step {}", self.step)));
                }
            }
            if stop() {
                pending_stop = true;
            }
            if run.ckpt_every > 0 && self.step % run.ckpt_every as u64 == 0 {
                pending_ckpt = true;
            }
        }
        if let Some((cb, n, cyc)) = cur_bin.take() {
            self.flush_bin(&file, cb, n, &mut counts, beat_of(cyc))?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let rep = Report { steps: self.step - s0, seconds: secs, mlups: fluid * (self.step - s0) as f64 / secs.max(1e-9) / 1e6, last_diag: last, stopped };
        if !stopped {
            self.checkpoint(out, &counts)?;
        }
        self.write_meta(out, &counts, Some(&rep))?;
        Ok(rep)
    }
}
