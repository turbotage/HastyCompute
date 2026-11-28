//! A cardiac LBM case: `<dir>/case.json` + the reference signed distance of the blood domain (`sdf.npy`, f32
//! [Z, Y, X], metres, > 0 in the blood, voxel centres at origin + (i + ½) h) + the motion (`cardiac_motion.hv`).
//! Written by `rustlib/tools/phantom/lbm_case.py` from a phantom (chambers + great-vessel stubs in the reference pose,
//! valves at the heart roots / AV interfaces, cuts at the stub ends with the network flows).
//!
//! case.json:
//! ```text
//! { "grid":   {"origin": [x, y, z], "dx": 5e-4, "dims": [nx, ny, nz]},          lattice box (node (0,0,0) at origin)
//!   "sdf":    {"file": "sdf.npy", "origin": [..], "h": 5e-4},
//!   "motion": {"file": "cardiac_motion.hv"} | {"still": true, "period": 0.857},
//!   "fluid":  {"nu": 3.3e-6, "rho": 1060},  "v_max": 1.5, "u_lattice": 0.08,
//!   "valves": [{"name", "kind": "leaflets" | "orifice", "center", "normal" (downstream), "e1", "radius",
//!               "ring", "thickness", "theta_open" (rad), "hole": [ax, ay] (orifice, × radius),
//!               "leaflets": "tri[:alpha0_deg,strips]" | "bi[:yc,strips_a,strips_p]" | [[[x, y], ..], ..] (units of radius, first two
//!               vertices = hinge chord), "opening": {"t": [..], "v": [..]} (0..1, periodic),
//!               "frame": "tissue" (default: deforms with the motion, reference pose) | "rigid", "push": 0 (fraction of
//!               the leaflet rotation velocity imposed on the fluid; 1 pumps the swept volume through the valve),
//!               "trigger": false (default) | true | {"p_open_mmHg": 5, "p_close_mmHg": 1, "tau_open": 0.015, "tau_close": 0.02,
//!               "tau_filter": 0.005, "window": 0.1} (pressure-triggered opening on top of the schedule, see ValveTrigger),
//!               "regurg_hole": [x, y, a, b] (optional: the closed valve's regurgitant orifice, valve-local units of R)}],
//!   "cuts":   [{"name", "center", "normal" (outward), "radius", "slab", "kind": "inlet" | "pressure" | "windkessel",
//!               "q": {"t", "v"} (inlet, m³/s into the domain), "p": Pa, "rcr": [R1, C, R2] (Pa·s/m³, m³/Pa),
//!               "sponge": m (default 6 mm for outlets, 0 for inlets), "sponge_tau": added τ at the plane (0.3),
//!               "q_ref": {"t", "v"} (expected outward flow, diagnostics), "share": {"group", "fraction"} (pressure cuts:
//!               trim the pressure so the cycle-mean inflow splits by fraction within the group), "inertance": Pa·s²/m³
//!               and "relax": s (inlets: upstream vein inertia, see CutSpec)}],
//!   "run":    {"cycles", "warmup", "bins", "out_factor", "acc_every", "geom_every", "cwale", "inv_iters", "uclip",
//!              "diag_every", "ckpt_every"} }
//! ```

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::motion::Motion;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Grid {
    pub origin: [f64; 3],
    pub dx: f64,
    pub dims: [usize; 3],
}

#[derive(Clone, Debug)]
pub struct Sdf {
    pub origin: [f64; 3],
    pub h: f64,
    /// x, y, z
    pub dims: [usize; 3],
    /// [z][y][x], metres, > 0 in the blood
    pub data: Vec<f32>,
}

impl Sdf {
    /// trilinear like the kernel's `tri_clamp` (`outside` beyond half a voxel outside)
    pub fn at(&self, x: &[f64; 3], outside: f64) -> f64 {
        let n = self.dims;
        let mut g = [0, 1, 2].map(|a| (x[a] - self.origin[a]) / self.h - 0.5);
        for a in 0..3 {
            if g[a] < -0.5 || g[a] > n[a] as f64 - 0.5 {
                return outside;
            }
            g[a] = g[a].clamp(0.0, (n[a] - 1) as f64);
        }
        let i0 = [0, 1, 2].map(|a| (g[a].floor() as usize).min(n[a].saturating_sub(2)));
        let f = [0, 1, 2].map(|a| g[a] - i0[a] as f64);
        let mut acc = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (x_, y_, z_) = ((i0[0] + dx).min(n[0] - 1), (i0[1] + dy).min(n[1] - 1), (i0[2] + dz).min(n[2] - 1));
                    let w = (if dx == 1 { f[0] } else { 1.0 - f[0] }) * (if dy == 1 { f[1] } else { 1.0 - f[1] }) * (if dz == 1 { f[2] } else { 1.0 - f[2] });
                    acc += w * self.data[(z_ * n[1] + y_) * n[0] + x_] as f64;
                }
            }
        }
        acc
    }
}

/// periodic table v(t)
#[derive(Clone, Debug)]
pub struct Table {
    pub t: Vec<f64>,
    pub v: Vec<f64>,
    pub period: f64,
}

impl Table {
    pub fn constant(v: f64, period: f64) -> Self {
        Self { t: vec![0.0], v: vec![v], period }
    }
    pub fn at(&self, t: f64) -> f64 {
        let n = self.t.len();
        if n == 1 {
            return self.v[0];
        }
        let tt = t.rem_euclid(self.period);
        let j = self.t.partition_point(|&x| x <= tt);
        let (ta, va, tb, vb) = if j == 0 {
            (self.t[n - 1] - self.period, self.v[n - 1], self.t[0], self.v[0])
        } else if j == n {
            (self.t[n - 1], self.v[n - 1], self.t[0] + self.period, self.v[0])
        } else {
            (self.t[j - 1], self.v[j - 1], self.t[j], self.v[j])
        };
        let a = if tb > ta { (tt - ta) / (tb - ta) } else { 0.0 };
        va * (1.0 - a) + vb * a
    }
    pub fn deriv(&self, t: f64, h: f64) -> f64 {
        (self.at(t + h) - self.at(t - h)) / (2.0 * h)
    }
    /// times in [0, period) where the table rises through `level` (sampled at 1 ms)
    pub fn rising(&self, level: f64) -> Vec<f64> {
        let n = ((self.period / 1e-3).ceil() as usize).max(8);
        let dt = self.period / n as f64;
        (0..n).filter(|&k| self.at(k as f64 * dt) < level && self.at((k + 1) as f64 * dt) >= level).map(|k| (k + 1) as f64 * dt % self.period).collect()
    }
}

#[derive(Clone, Debug)]
pub enum ValveKind {
    /// a plate over the annulus with an elliptical hole (semi-axes × radius at full opening)
    Orifice { hole: [f64; 2] },
    /// kinematic leaflets: closed-configuration polygons in valve-local 2D (units of radius), first two vertices =
    /// hinge chord; each rotates about its chord by θ = θ_open · opening
    Leaflets { polys: Vec<Vec<[f64; 2]>> },
}

#[derive(Clone, Debug)]
pub struct ValveSpec {
    pub name: String,
    pub kind: ValveKind,
    pub center: [f64; 3],
    /// downstream (direction of forward flow)
    pub normal: [f64; 3],
    pub e1: [f64; 3],
    pub radius: f64,
    /// annulus ring width beyond the hinge circle
    pub ring: f64,
    pub thickness: f64,
    pub theta_open: f64,
    pub opening: Table,
    /// "tissue" frame (default): the valve lives in the reference pose and deforms with the tissue (evaluated at each
    /// node's reference position, solid velocity = tissue velocity + leaflet rotation), so it stays sealed against the
    /// collar and walls built in the reference pose; "rigid": a rigid disk on the moved centre and axes (the annulus
    /// saddle / ellipse then opens gaps around a closed valve)
    pub tissue: bool,
    pub trigger: ValveTrigger,
    /// fraction of the leaflets' prescribed rotation velocity imposed on the fluid (0: quasi-static leaflets — the
    /// geometry moves, the wall velocity is the tissue's; real leaflets are passive and move with the flow)
    pub push: f64,
    /// regurgitant orifice of the closed valve (valve disease): an ellipse in valve-local 2D (units of the radius),
    /// centre (x, y) on e1 / e2 and semi-axes (a, b), kept open through the leaflets and the closed disks
    /// (JSON "regurg_hole": [x, y, a, b])
    pub hole: Option<[f64; 4]>,
    /// zero-thickness membrane leaflets (JSON "membrane": true): every lattice link crossing a leaflet or the closed
    /// disk is an interpolated bounce-back wall link, instead of partially saturated plates `thickness` thick (which
    /// act as sharp-edged orifices and push their own volume through the valve as they swing)
    pub membrane: bool,
}

/// pressure-triggered opening on top of the prescribed schedule (a 0D valve law, geometry-free): the opening is
/// max(schedule, o_p), with o_p driven by the filtered forward gradient Δp = p_up − p_down of the valve's two probes
/// — a closed valve starts opening (towards 1 over τ_open) once Δp > p_open, an open one keeps opening at a rate
/// ∝ Δp / p_open while any forward gradient remains, and closes over τ_close once Δp < −p_close (flow reversal);
/// otherwise it holds. A chamber the motion squeezes or dilates behind closed valves vents through the valve
/// physiology would open, instead of building tens of mmHg in the LBM's compliance and bursting when the schedule opens.
/// Only within `window` before a scheduled opening (the isovolumic phase): the LBM's pressures are gauge pressures
/// around the outlets' level, without the ventricle–atrium difference of systole, so elsewhere a "forward" gradient
/// held the AV valves half open through systole; outside the window a schedule-closed valve's o_p decays (τ_close)
#[derive(Clone, Debug)]
pub struct ValveTrigger {
    pub enabled: bool,
    /// Pa
    pub p_open: f64,
    pub p_close: f64,
    /// s
    pub tau_open: f64,
    pub tau_close: f64,
    pub tau_filter: f64,
    /// s before a scheduled opening in which the trigger may act
    pub window: f64,
}

impl Default for ValveTrigger {
    fn default() -> Self {
        // off by default: with quasi-static leaflets the isovolumic bursts were gone and the trigger only added
        // closed-valve leakage on ph0001 (2026-11-07); kept for geometries whose sealed chambers do change volume
        Self { enabled: false, p_open: 5.0 * 133.322, p_close: 1.0 * 133.322, tau_open: 0.015, tau_close: 0.02, tau_filter: 0.005, window: 0.1 }
    }
}

#[derive(Clone, Debug)]
pub enum CutKind {
    /// prescribed flow into the domain (m³/s), plug profile plus the cut's motion
    Inlet { q: Table },
    /// constant pressure (Pa, relative)
    Pressure { p: f64 },
    /// three-element Windkessel on the measured outflow
    Windkessel { r1: f64, c: f64, r2: f64 },
}

#[derive(Clone, Debug)]
pub struct CutSpec {
    pub name: String,
    pub center: [f64; 3],
    /// outward
    pub normal: [f64; 3],
    pub radius: f64,
    /// length of the equilibrium slab beyond the plane, m
    pub slab: f64,
    /// sponge before the plane (outlets): length (m) and added relaxation time at the plane
    pub sponge: f64,
    pub sponge_tau: f64,
    /// expected outward flow (m³/s; diagnostics only, e.g. the cardiac model's Q_AO at the aortic outlet)
    pub q_ref: Option<Table>,
    /// pressure cuts of one group (e.g. the pulmonary veins) whose cycle-mean inflows should split by these fractions:
    /// a slow integral trim of each cut's pressure (zero mean over the group) steers the split, the wall motion sets
    /// the total and the waveform
    pub share: Option<(String, f64)>,
    /// inlets: inertance of the upstream vein (Pa·s²/m³; 0: a plain velocity inlet). The inflow is the prescribed
    /// q(t) plus ΔQ with dΔQ/dt = −p_cut / L − ΔQ / relax: the wall motion, not q(t), decides the total (no pressure
    /// builds up behind closed valves), the vein's inertia keeps the waveform and the split near q(t)
    pub inertance: f64,
    pub relax: f64,
    /// inlets with a share group and an inertance: one correction ΔQ per group, driven by the group's area-mean cut
    /// pressure through the group inertance, damped by this resistance (Pa·s/m³), split by the share fractions (the
    /// veins of one atrium cannot slosh against each other)
    pub resistance: f64,
    /// the upstream path of an inlet (builder): the spin history integrates the slab's saturation along it
    pub upstream: Option<Upstream>,
    pub kind: CutKind,
}

/// where an inlet's blood was before it entered the domain: lab sample points from far upstream to the cut and the
/// transit time from each to the next (s); it starts fresh (s = 1), or, with `from`, with the flux-weighted outflow s
/// of those cuts (e.g. the pulmonary arteries for a pulmonary vein: the path then runs through the lung)
#[derive(Clone, Debug, Default)]
pub struct Upstream {
    pub points: Vec<[f64; 3]>,
    pub dt: Vec<f64>,
    pub from: Vec<String>,
}

/// spin history of the blood for a given protocol: TR, blood T1, the slab (centre, unit normal, the flip-angle
/// profile over the signed distance along the normal), the entering relative M_z of cuts without an upstream path
/// (1: fresh)
#[derive(Clone, Debug)]
pub struct SpinSpec {
    pub tr: f64,
    pub t1: f64,
    pub center: [f64; 3],
    pub normal: [f64; 3],
    pub z0: f64,
    pub dz: f64,
    pub alpha: Vec<f32>,
    pub s_in: Vec<f64>,
    /// the protocol it was made from (metadata for the beat library: flip, slab thickness, pulse, …)
    pub protocol: serde_json::Value,
}

impl SpinSpec {
    /// a slab along the lab z through the origin (tests)
    pub fn along_z(tr: f64, t1: f64, z0: f64, dz: f64, alpha: Vec<f32>, s_in: Vec<f64>) -> Self {
        Self { tr, t1, center: [0.0; 3], normal: [0.0, 0.0, 1.0], z0, dz, alpha, s_in, protocol: serde_json::Value::Null }
    }

    /// signed distance of a lab point along the slab normal from its centre
    pub fn dist(&self, p: &[f64; 3]) -> f64 {
        (0..3).map(|a| (p[a] - self.center[a]) * self.normal[a]).sum()
    }

    /// relative M_z after an upstream path entering with s0
    pub fn along_path(&self, s0: f64, u: &Upstream) -> f64 {
        let d: Vec<f64> = u.points.iter().map(|p| self.dist(p)).collect();
        self.along_from(s0, &d, &u.dt)
    }

    pub fn alpha_at(&self, z: f64) -> f64 {
        let f = (z - self.z0) / self.dz;
        if f < 0.0 || f > (self.alpha.len() - 1) as f64 {
            return 0.0;
        }
        let i = (f as usize).min(self.alpha.len() - 2);
        let a = f - i as f64;
        self.alpha[i] as f64 * (1.0 - a) + self.alpha[i + 1] as f64 * a
    }

    /// relative M_z after a path (z, transit) entering with s = 1: per segment ds/dt = −k s + (1 − s)/T1 with
    /// k = (1 − cos α(z)) / TR, solved exactly
    pub fn along(&self, z: &[f64], dt: &[f64]) -> f64 {
        self.along_from(1.0, z, dt)
    }

    pub fn along_from(&self, s0: f64, z: &[f64], dt: &[f64]) -> f64 {
        let mut s = s0;
        for (zi, ti) in z.iter().zip(dt) {
            let k = (1.0 - self.alpha_at(*zi).cos()) / self.tr;
            let r = k + 1.0 / self.t1;
            let ss = (1.0 / self.t1) / r;
            s = ss + (s - ss) * (-r * ti).exp();
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct RunSpec {
    pub cycles: usize,
    pub warmup: usize,
    pub bins: usize,
    pub out_factor: usize,
    pub acc_every: usize,
    pub geom_every: usize,
    pub cwale: f64,
    pub inv_iters: usize,
    pub uclip: f64,
    pub diag_every: usize,
    pub ckpt_every: usize,
    /// steps between the inertial-inlet updates (cut pressures → upstream flow)
    pub fb_every: usize,
    /// bulk relaxation time (acoustic damping; ≤ ½: none beyond the shear τ)
    pub tau_bulk: f64,
    /// write the per-cycle beat library (`<out>/beats/`)
    pub beats: bool,
    /// the inlet flux controller (κ) steers the exact link flux through the slab interface (node_diag; default) or,
    /// as before, the u-based first-layer flux, which reads 2–3 % low at a velocity inlet (JSON "kappa_flux":
    /// "link" | "layer")
    pub kappa_link: bool,
    /// a valve core's bounce-back wall where the solid fraction crosses ½ along the link instead of half-way (JSON
    /// "valve_q": true); the half-way wall advances in half-cell steps as the sheet moves
    pub valve_q: bool,
    /// wall mass (2026-11-26): the moving walls and valves change the chambers' volumes, not their mass (node_wallmass);
    /// JSON "wall_mass": true. Off: covering destroys a node's mass and the moving-wall bounce-back puts back ≈ 0.85–0.97
    /// of it, so the chambers pump ≈ 0.9 of their stroke
    pub wall_mass: bool,
    /// wall pump (2026-11-27): each chamber's moving-wall bounce-back corrected to the cardiac model's volume rate
    /// (Case::wall_pump; JSON run "wall_pump": true). The bounce-back is the chambers' pump (with a steady density a
    /// region's outflow equals it); in the heart it injected ≈ 0.89 of the model's stroke
    pub wall_pump: bool,
    /// grid refinement: the non-equilibrium transfer between levels (kernels/lbm_core.h "grid refinement"; JSON run
    /// "refine_neq"): "copy" (default: unscaled, consistent over a coarse step — Rohde 2006 / Chen 2006), "layers"
    /// (per fine step: × (τ_f − 1) / (2 (τ_c − 1)) for the near ghost layer, × (τ_f − 2) / (2 (τ_c − 1)) for the far
    /// one), "dc" (Dupuis–Chopard × (τ_f − 1) / (2 (τ_c − 1)) for all, a strain-rate stress correction fine → coarse:
    /// worse interface pressure steps, kept for comparison)
    pub refine_neq: String,
    /// grid refinement: the WALE filter width ramped from the coarser level's at the interface to the level's own over
    /// this many fine nodes (refine.rs wale_ramp; JSON run "refine_wale_ramp"; 0: none — ν_t then jumps 4× at the
    /// interface for the same physical gradient)
    pub refine_wale_ramp: usize,
}

impl Default for RunSpec {
    fn default() -> Self {
        Self { cycles: 4, warmup: 2, bins: 50, out_factor: 2, acc_every: 10, geom_every: 4, cwale: 0.5, inv_iters: 3, uclip: 0.3, diag_every: 500, ckpt_every: 20_000, fb_every: 50, tau_bulk: 2.0, beats: false, kappa_link: true, valve_q: false, wall_mass: false, wall_pump: false, refine_neq: "copy".into(), refine_wale_ramp: 0 }
    }
}

/// the builder's region map (reference pose, the sdf grid: chambers, vessel stubs, …) and its graph: which regions
/// may exchange fluid freely (`allowed`, e.g. a vein and its atrium), which two regions each valve separates
#[derive(Clone, Debug, Default)]
pub struct Regions {
    pub names: std::collections::BTreeMap<u16, String>,
    pub allowed: Vec<(u16, u16)>,
    pub valves: std::collections::BTreeMap<String, (u16, u16)>,
    /// region id per sdf voxel (ids ≥ 256 clamp to 0)
    pub map: Vec<u8>,
}

/// wall transpiration: the compliance of rigid vessel trees (research/flow_phantom.md Tier B4). In a band of `band`
/// lattice units (≥ 2; the last half unit tapers) on both sides of a tree's wall the wall velocity gains the outward
/// normal velocity q_g(t) w(X) / (da/dA)
/// (kernels/lbm_core.h node_geom1), so tree g stores q_g(t) = valve flow − Σ outlet targets − its motion volume change
/// and its pressure outlet carries only the residual. `group` (u8, 0 = none) and `weight` (m⁻², ∮ w dA = 1 per group)
/// on the sdf grid.
#[derive(Clone, Debug)]
pub struct Transpiration {
    pub band: f64,
    pub group: Vec<u8>,
    pub weight: Vec<f32>,
    /// (id ≥ 1, name, q_w(t) in m³/s, outward = stored)
    pub groups: Vec<(u8, String, Table)>,
}

/// `case.json` "transpiration": {"group": "transp_group.npy", "weight": "transp_weight.npy", "band": 2.0,
/// "groups": [{"id": 1, "name": .., "q": {"t": [..], "v": [..]}}]}; absent: None
fn load_transpiration(dir: &Path, r: &Value, len: usize, period: f64) -> Result<Option<Transpiration>> {
    if !r.is_object() {
        return Ok(None);
    }
    let rd = |k: &str| -> Result<hasty_value::Tensor> {
        let f = dir.join(r[k].as_str().ok_or_else(|| Error::Case(format!("transpiration.{k}: want a file name")))?);
        let (t, _) = hasty_value::npy::read(&f).map_err(|e| Error::Value(format!("{}: {e}", f.display())))?;
        Ok(t)
    };
    let group: Vec<u8> = rd("group")?.to_vec::<u8>().map_err(|e| Error::Value(e.to_string()))?;
    let weight: Vec<f32> = rd("weight")?.to_vec::<f32>().map_err(|e| Error::Value(e.to_string()))?;
    if group.len() != len || weight.len() != len {
        return Err(Error::Case(format!("transpiration: {} group / {} weight voxels, the sdf has {len}", group.len(), weight.len())));
    }
    let mut groups = vec![];
    for g in r["groups"].as_array().ok_or_else(|| Error::Case("transpiration.groups: want a list".into()))? {
        let id = g["id"].as_u64().filter(|&x| (1..=8).contains(&x)).ok_or_else(|| Error::Case("transpiration group id: 1..8".into()))? as u8;
        groups.push((id, g["name"].as_str().unwrap_or("").to_string(), table(&g["q"], period)?));
    }
    if let Some(&mx) = group.iter().max() {
        if mx as usize > groups.iter().map(|g| g.0 as usize).max().unwrap_or(0) {
            return Err(Error::Case(format!("transpiration: group map holds id {mx} without a waveform")));
        }
    }
    let band = r["band"].as_f64().unwrap_or(2.0);
    if band < 2.0 {
        return Err(Error::Case(format!("transpiration.band {band}: ≥ 2 lattice units (the wall links reach √2, the taper takes ½)")));
    }
    Ok(Some(Transpiration { band, group, weight, groups }))
}

/// grid refinement: the region of level `level` (≥ 1: dx / 2^level, dt / 2^level) as a union of boxes and spheres in
/// the lab frame (m); a level's region lies inside the next coarser level's (refine.rs keeps 3 coarse cells of margin).
/// JSON "refine": [{"level": 1, "boxes": [{"min": [x, y, z], "max": [x, y, z]}], "spheres": [{"center": [..],
/// "radius": r}]}, …]
#[derive(Clone, Debug, Default)]
pub struct RefineSpec {
    pub level: usize,
    pub boxes: Vec<([f64; 3], [f64; 3])>,
    pub spheres: Vec<([f64; 3], f64)>,
}

impl RefineSpec {
    pub fn contains(&self, x: &[f64; 3]) -> bool {
        self.boxes.iter().any(|(lo, hi)| (0..3).all(|a| x[a] >= lo[a] && x[a] <= hi[a]))
            || self.spheres.iter().any(|(c, r)| (0..3).map(|a| (x[a] - c[a]).powi(2)).sum::<f64>() <= r * r)
    }
}

fn load_refine(r: &Value) -> Result<Vec<RefineSpec>> {
    let Some(a) = r.as_array() else { return Ok(vec![]) };
    let mut out = vec![];
    for e in a {
        let level = e["level"].as_u64().unwrap_or(1) as usize;
        if !(1..=3).contains(&level) {
            return Err(Error::Case(format!("refine.level {level}: 1..3")));
        }
        let mut sp = RefineSpec { level, ..Default::default() };
        for b in e["boxes"].as_array().map(|x| x.as_slice()).unwrap_or(&[]) {
            sp.boxes.push((v3(b, "min")?, v3(b, "max")?));
        }
        for b in e["spheres"].as_array().map(|x| x.as_slice()).unwrap_or(&[]) {
            sp.spheres.push((v3(b, "center")?, num(b, "radius")?));
        }
        out.push(sp);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct Case {
    pub dir: PathBuf,
    pub grid: Grid,
    pub sdf: Sdf,
    pub motion: Motion,
    pub nu: f64,
    pub rho: f64,
    pub v_max: f64,
    pub u_lattice: f64,
    pub valves: Vec<ValveSpec>,
    pub cuts: Vec<CutSpec>,
    pub run: RunSpec,
    pub regions: Option<Regions>,
    /// the blood spin history (set by the run's protocol, e.g. `cardiac_lbm --spin`)
    pub spin: Option<SpinSpec>,
    /// wall transpiration of the vessel trees (None: rigid trees, the pressure outlets take the storage)
    pub transpiration: Option<Transpiration>,
    /// the chambers' volume rates of the cardiac model, (region id, dV/dt table in m³/s): with RunSpec::wall_pump each
    /// chamber's moving-wall bounce-back injects exactly −dV/dt (JSON "wall_pump": {region name: table})
    pub wall_pump: Vec<(u16, Table)>,
    /// grid refinement regions (empty: a single uniform level)
    pub refine: Vec<RefineSpec>,
}

/// `case.json` "regions": {"file": "region.npy" (i16 on the sdf grid), "names": {id: name}, "allowed": [[a, b], ..],
/// "valves": {valve: [upstream, downstream]}}; absent (older cases): None
fn load_regions(dir: &Path, r: &Value, len: usize) -> Result<Option<Regions>> {
    if !r.is_object() {
        return Ok(None);
    }
    let f = dir.join(r["file"].as_str().unwrap_or("region.npy"));
    if !f.exists() {
        return Ok(None);
    }
    let (t, _) = hasty_value::npy::read(&f).map_err(|e| Error::Value(e.to_string()))?;
    let raw: Vec<i16> = t.to_vec::<i16>().map_err(|e| Error::Value(e.to_string()))?;
    if raw.len() != len {
        return Err(Error::Case(format!("{}: {} voxels, the sdf has {len}", f.display(), raw.len())));
    }
    let map = raw.iter().map(|&x| if (0..256).contains(&x) { x as u8 } else { 0 }).collect();
    let names = r["names"].as_object().map_or(Default::default(), |o| o.iter().filter_map(|(k, v)| Some((k.parse::<u16>().ok()?, v.as_str()?.to_string()))).collect());
    let pair = |v: &Value| -> Option<(u16, u16)> { Some((v[0].as_u64()? as u16, v[1].as_u64()? as u16)) };
    let allowed = r["allowed"].as_array().map_or(vec![], |a| a.iter().filter_map(pair).collect());
    let valves = r["valves"].as_object().map_or(Default::default(), |o| o.iter().filter_map(|(k, v)| Some((k.clone(), pair(v)?))).collect());
    Ok(Some(Regions { names, allowed, valves, map }))
}

fn v3(v: &Value, k: &str) -> Result<[f64; 3]> {
    let a = v[k].as_array().ok_or_else(|| Error::Case(format!("{k}: want [x, y, z]")))?;
    if a.len() != 3 {
        return Err(Error::Case(format!("{k}: want 3 numbers")));
    }
    Ok([0, 1, 2].map(|i| a[i].as_f64().unwrap_or(f64::NAN)))
}
fn num(v: &Value, k: &str) -> Result<f64> {
    v[k].as_f64().ok_or_else(|| Error::Case(format!("missing number {k}")))
}
fn table(v: &Value, period: f64) -> Result<Table> {
    let t: Vec<f64> = v["t"].as_array().ok_or_else(|| Error::Case("table without t".into()))?.iter().map(|x| x.as_f64().unwrap()).collect();
    let y: Vec<f64> = v["v"].as_array().ok_or_else(|| Error::Case("table without v".into()))?.iter().map(|x| x.as_f64().unwrap()).collect();
    if t.len() != y.len() || t.is_empty() || t.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::Case("table: t, v of equal length, t increasing".into()));
    }
    Ok(Table { t, v: y, period })
}

pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    a.map(|x| x / n)
}
pub fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// leaflet presets in valve-local 2D, units of the hinge radius. A rigid leaflet hinged on the chord between its
/// commissures would cap the open orifice at the inscribed polygon (≈ 41 % of the annulus for three leaflets); each
/// anatomical leaflet is therefore a fan of narrow strips hinged on short chords along the annulus arc, which tile
/// the annulus when closed and leave nearly the whole lumen open (≈ the leaflets folding against the wall).
pub mod presets {
    fn arc(a: f64) -> [f64; 2] {
        [a.cos(), a.sin()]
    }
    /// three equal leaflets (aortic, pulmonary, tricuspid), commissures at α0 + 120° k, closed leaflets meet at the
    /// centre; `strips` per leaflet
    pub fn trileaflet(alpha0_deg: f64, strips: usize) -> Vec<Vec<[f64; 2]>> {
        let a0 = alpha0_deg.to_radians();
        let mut out = vec![];
        for k in 0..3 {
            let (a, b) = (a0 + k as f64 * 2.0 * std::f64::consts::PI / 3.0, a0 + (k + 1) as f64 * 2.0 * std::f64::consts::PI / 3.0);
            for j in 0..strips {
                let (s0, s1) = (a + (b - a) * j as f64 / strips as f64, a + (b - a) * (j + 1) as f64 / strips as f64);
                out.push(vec![arc(s0), arc(s1), [0.0, 0.0]]);
            }
        }
        out
    }
    /// two leaflets (mitral): coaptation along y = yc, commissures where it meets the annulus; the anterior leaflet
    /// (arc above the line) and the posterior one (below) as strips reaching the line
    pub fn bileaflet(yc: f64, strips_a: usize, strips_p: usize) -> Vec<Vec<[f64; 2]>> {
        let tc = yc.clamp(-0.9, 0.9).asin();
        let mut out = vec![];
        let mut fan = |a: f64, b: f64, n: usize| {
            for j in 0..n {
                let (s0, s1) = (a + (b - a) * j as f64 / n as f64, a + (b - a) * (j + 1) as f64 / n as f64);
                let (p0, p1) = (arc(s0), arc(s1));
                out.push(vec![p0, p1, [p1[0], yc], [p0[0], yc]]);
            }
        };
        fan(tc, std::f64::consts::PI - tc, strips_a);
        fan(std::f64::consts::PI - tc, 2.0 * std::f64::consts::PI + tc, strips_p);
        out
    }
}

fn leaflets(v: &Value) -> Result<Vec<Vec<[f64; 2]>>> {
    if let Some(s) = v.as_str() {
        let (kind, args) = s.split_once(':').unwrap_or((s, ""));
        let nums: Vec<f64> = args.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
        return match kind {
            "tri" => Ok(presets::trileaflet(nums.first().copied().unwrap_or(90.0), nums.get(1).map_or(4, |x| *x as usize))),
            "bi" => Ok(presets::bileaflet(nums.first().copied().unwrap_or(0.2), nums.get(1).map_or(5, |x| *x as usize), nums.get(2).map_or(7, |x| *x as usize))),
            k => Err(Error::Case(format!("leaflets {k}: tri | bi | polygons"))),
        };
    }
    let a = v.as_array().ok_or_else(|| Error::Case("leaflets: preset string or polygon list".into()))?;
    a.iter()
        .map(|p| {
            p.as_array().ok_or_else(|| Error::Case("leaflet polygon".into())).map(|vs| vs.iter().map(|q| [q[0].as_f64().unwrap(), q[1].as_f64().unwrap()]).collect())
        })
        .collect()
}

impl Case {
    pub fn load(dir: &Path) -> Result<Self> {
        let j: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("case.json"))?).map_err(|e| Error::Case(e.to_string()))?;
        let g = &j["grid"];
        let dims = g["dims"].as_array().ok_or_else(|| Error::Case("grid.dims".into()))?;
        let grid = Grid { origin: v3(g, "origin")?, dx: num(g, "dx")?, dims: [0, 1, 2].map(|i| dims[i].as_u64().unwrap() as usize) };
        let s = &j["sdf"];
        let (t, _) = hasty_value::npy::read(&dir.join(s["file"].as_str().unwrap_or("sdf.npy"))).map_err(|e| Error::Value(e.to_string()))?;
        let sh = t.shape().to_vec();
        if sh.len() != 3 {
            return Err(Error::Case(format!("sdf shape {sh:?}, want [Z, Y, X]")));
        }
        let data: Vec<f32> = t.to_vec::<f32>().map_err(|e| Error::Value(e.to_string()))?;
        let sdf = Sdf { origin: v3(s, "origin")?, h: num(s, "h")?, dims: [sh[2], sh[1], sh[0]], data };
        let sdf_len = sdf.data.len();
        let m = &j["motion"];
        let motion = if m["still"].as_bool() == Some(true) {
            Motion::still(grid.origin, 1e-2, num(m, "period")?)
        } else {
            let f = m["file"].as_str().ok_or_else(|| Error::Case("motion.file".into()))?;
            let p = if Path::new(f).is_absolute() { PathBuf::from(f) } else { dir.join(f) };
            Motion::load(&p)?
        };
        let period = motion.period;
        let fl = &j["fluid"];
        let valves = j["valves"]
            .as_array()
            .map(|a| a.as_slice())
            .unwrap_or(&[])
            .iter()
            .map(|v| {
                let kind = match v["kind"].as_str().unwrap_or("leaflets") {
                    "orifice" => ValveKind::Orifice { hole: v["hole"].as_array().map_or([0.9, 0.9], |h| [h[0].as_f64().unwrap(), h[1].as_f64().unwrap()]) },
                    _ => ValveKind::Leaflets { polys: leaflets(&v["leaflets"])? },
                };
                Ok(ValveSpec {
                    name: v["name"].as_str().unwrap_or("valve").into(),
                    kind,
                    center: v3(v, "center")?,
                    normal: normalize(v3(v, "normal")?),
                    e1: v3(v, "e1")?,
                    radius: num(v, "radius")?,
                    ring: v["ring"].as_f64().unwrap_or(2.0 * grid.dx),
                    thickness: v["thickness"].as_f64().unwrap_or(3.0 * grid.dx),
                    theta_open: v["theta_open"].as_f64().unwrap_or(75f64.to_radians()),
                    opening: table(&v["opening"], period)?,
                    tissue: v["frame"].as_str() != Some("rigid"),
                    push: v["push"].as_f64().unwrap_or(0.0),
                    hole: v["regurg_hole"].as_array().map(|h| [0, 1, 2, 3].map(|i| h.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0))),
                    membrane: v["membrane"].as_bool().unwrap_or(false),
                    trigger: {
                        let d = ValveTrigger::default();
                        let t = &v["trigger"];
                        if t.as_bool() == Some(false) || t.is_null() {
                            ValveTrigger { enabled: false, ..d }
                        } else if t.as_bool() == Some(true) {
                            ValveTrigger { enabled: true, ..d }
                        } else {
                            ValveTrigger {
                                enabled: true,
                                p_open: t["p_open_mmHg"].as_f64().map_or(d.p_open, |x| x * 133.322),
                                p_close: t["p_close_mmHg"].as_f64().map_or(d.p_close, |x| x * 133.322),
                                tau_open: t["tau_open"].as_f64().unwrap_or(d.tau_open),
                                tau_close: t["tau_close"].as_f64().unwrap_or(d.tau_close),
                                tau_filter: t["tau_filter"].as_f64().unwrap_or(d.tau_filter),
                                window: t["window"].as_f64().unwrap_or(d.window),
                            }
                        }
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cuts = j["cuts"]
            .as_array()
            .map(|a| a.as_slice())
            .unwrap_or(&[])
            .iter()
            .map(|c| {
                let kind = match c["kind"].as_str().unwrap_or("pressure") {
                    "inlet" => CutKind::Inlet { q: table(&c["q"], period)? },
                    "windkessel" => {
                        let r = c["rcr"].as_array().ok_or_else(|| Error::Case("windkessel needs rcr".into()))?;
                        CutKind::Windkessel { r1: r[0].as_f64().unwrap(), c: r[1].as_f64().unwrap(), r2: r[2].as_f64().unwrap() }
                    }
                    _ => CutKind::Pressure { p: c["p"].as_f64().unwrap_or(0.0) },
                };
                Ok(CutSpec {
                    name: c["name"].as_str().unwrap_or("cut").into(),
                    center: v3(c, "center")?,
                    normal: normalize(v3(c, "normal")?),
                    radius: num(c, "radius")?,
                    slab: c["slab"].as_f64().unwrap_or(3.0 * grid.dx),
                    sponge: c["sponge"].as_f64().unwrap_or(if matches!(kind, CutKind::Inlet { .. }) { 0.0 } else { 6e-3 }),
                    sponge_tau: c["sponge_tau"].as_f64().unwrap_or(0.3),
                    q_ref: if c["q_ref"].is_object() { Some(table(&c["q_ref"], period)?) } else { None },
                    inertance: c["inertance"].as_f64().unwrap_or(0.0),
                    relax: c["relax"].as_f64().unwrap_or(0.25),
                    resistance: c["resistance"].as_f64().unwrap_or(0.0),
                    upstream: c["upstream"].as_object().map(|u| Upstream {
                        points: u.get("points").and_then(|x| x.as_array()).map_or(vec![], |a| {
                            a.iter().filter_map(|p| p.as_array().filter(|q| q.len() == 3).map(|q| [0, 1, 2].map(|i| q[i].as_f64().unwrap_or(0.0)))).collect()
                        }),
                        dt: u.get("dt").and_then(|x| x.as_array()).map_or(vec![], |a| a.iter().filter_map(|x| x.as_f64()).collect()),
                        from: u.get("from").and_then(|x| x.as_array()).map_or(vec![], |a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()),
                    }),
                    share: c["share"].as_object().map(|o| (o.get("group").and_then(|g| g.as_str()).unwrap_or("").to_string(), o.get("fraction").and_then(|f| f.as_f64()).unwrap_or(0.0))),
                    kind,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let r = &j["run"];
        let d = RunSpec::default();
        let u = |k: &str, dv: usize| r[k].as_u64().map_or(dv, |x| x as usize);
        let f = |k: &str, dv: f64| r[k].as_f64().unwrap_or(dv);
        let run = RunSpec {
            cycles: u("cycles", d.cycles),
            warmup: u("warmup", d.warmup),
            bins: u("bins", d.bins),
            out_factor: u("out_factor", d.out_factor),
            acc_every: u("acc_every", d.acc_every),
            geom_every: u("geom_every", d.geom_every),
            cwale: f("cwale", d.cwale),
            inv_iters: u("inv_iters", d.inv_iters),
            uclip: f("uclip", d.uclip),
            diag_every: u("diag_every", d.diag_every),
            ckpt_every: u("ckpt_every", d.ckpt_every),
            fb_every: u("fb_every", d.fb_every).max(1),
            tau_bulk: f("tau_bulk", d.tau_bulk),
            beats: r["beats"].as_bool().unwrap_or(d.beats),
            kappa_link: r["kappa_flux"].as_str().map_or(d.kappa_link, |x| x != "layer"),
            valve_q: r["valve_q"].as_bool().unwrap_or(d.valve_q),
            wall_mass: r["wall_mass"].as_bool().unwrap_or(d.wall_mass),
            wall_pump: r["wall_pump"].as_bool().unwrap_or(d.wall_pump),
            refine_neq: r["refine_neq"].as_str().map_or(d.refine_neq.clone(), String::from),
            refine_wale_ramp: u("refine_wale_ramp", d.refine_wale_ramp),
        };
        if !["copy", "layers", "dc"].contains(&run.refine_neq.as_str()) {
            return Err(Error::Case(format!("run.refine_neq {:?}: copy, layers or dc", run.refine_neq)));
        }
        if ![1, 2, 4, 8].contains(&run.out_factor) {
            return Err(Error::Case("run.out_factor: 1, 2, 4 or 8".into()));
        }
        let transpiration = load_transpiration(dir, &j["transpiration"], sdf_len, motion.period)?;
        let regions = load_regions(dir, &j["regions"], sdf_len)?;
        let mut wall_pump = vec![];
        if let (Some(wp), Some(rg)) = (j["wall_pump"].as_object(), regions.as_ref()) {
            for (name, tab) in wp {
                if let Some((&id, _)) = rg.names.iter().find(|(_, n)| *n == name) {
                    wall_pump.push((id, table(tab, motion.period)?));
                }
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            grid,
            sdf,
            motion,
            nu: fl["nu"].as_f64().unwrap_or(3.3e-6),
            rho: fl["rho"].as_f64().unwrap_or(1060.0),
            v_max: j["v_max"].as_f64().unwrap_or(1.5),
            u_lattice: j["u_lattice"].as_f64().unwrap_or(0.08),
            valves,
            cuts,
            run,
            regions,
            spin: None,
            transpiration,
            wall_pump,
            refine: load_refine(&j["refine"])?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_table_and_presets() {
        let t = Table { t: vec![0.0, 0.25, 0.5], v: vec![0.0, 1.0, 0.0], period: 1.0 };
        assert!((t.at(0.125) - 0.5).abs() < 1e-12);
        assert!((t.at(0.75) - 0.0).abs() < 1e-12); // between 0.5 (0) and 1.0 (wraps to t = 0: 0)
        assert!((t.at(1.125) - 0.5).abs() < 1e-12);
        let t2 = Table { t: vec![0.1, 0.6], v: vec![1.0, 3.0], period: 1.0 };
        assert!((t2.at(0.05) - 1.2).abs() < 1e-12); // wrap segment −0.4 → 0.1 (3 → 1)
        assert!((t2.at(0.85) - 2.0).abs() < 1e-12);
        let area = |ps: &Vec<Vec<[f64; 2]>>| -> f64 {
            ps.iter().map(|p| (0..p.len()).map(|k| { let (a, b) = (p[k], p[(k + 1) % p.len()]); a[0] * b[1] - b[0] * a[1] }).sum::<f64>().abs() * 0.5).sum()
        };
        // closed strips tile the inscribed polygon of the annulus (12-gon: 3 sin 30° = 3.0 of π)
        let tri = presets::trileaflet(90.0, 4);
        assert_eq!(tri.len(), 12);
        assert!((area(&tri) - 6.0 * (std::f64::consts::PI / 6.0).sin()).abs() < 1e-12, "{}", area(&tri));
        let bi = presets::bileaflet(0.2, 5, 7);
        assert_eq!(bi.len(), 12);
        assert!(bi.iter().all(|p| p.len() == 4));
        assert!(area(&bi) > 0.95 * 2.9 && area(&bi) < std::f64::consts::PI, "{}", area(&bi));
    }
}
