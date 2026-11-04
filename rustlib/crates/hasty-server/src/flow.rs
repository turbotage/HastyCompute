//! 4D flow on the server: the flow value schema, derived quantities and particle tracing for the flow viewer
//! (`plotting_website/flow`). Everything is CPU (rayon) so it runs next to GPU work.
//!
//! **Flow value** (a dict; `viz::show_flow`, Python `show_flow`, or the `flow_make` command build one):
//! - `"kind": "flow4d"`
//! - `"vel"`: f32, hasty dims `[X, Y, Z, 3, F]` (C-order `[F, 3, Z, Y, X]`), m/s; component c is along array axis c
//!   (x fastest) — the layout `recon_p2np` writes (`vel_<tag>.npy`).
//! - `"mag"` (optional): f32 `[X, Y, Z, F]`, magnitude (anatomy); default: speed.
//! - `"spacing"`: `[dx, dy, dz]` m; `"dt"`: s per frame (cyclic: frame F wraps to 0); `"venc"` (optional), m/s.
//! - `"mask"` (optional): u8 `[X, Y, Z]`, vessel / lumen (seeding, stopping, plane segmentation).
//!
//! Points returned to the viewer are in voxel index coordinates (x, y, z), voxel centres at integers.

use std::sync::Arc;

use hasty_value::{DType, Tensor, Value};
use rayon::prelude::*;
use serde_json::{json, Value as Json};

use crate::commands::Registry;
use crate::Error;

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

pub struct Flow {
    pub dims: [usize; 3],
    pub nf: usize,
    pub spacing: [f64; 3],
    pub dt: f64,
    pub venc: Option<f64>,
    /// `[F, 3, Z, Y, X]`
    pub vel: Vec<f32>,
    /// `[F, Z, Y, X]`
    pub mag: Option<Vec<f32>>,
    /// `[Z, Y, X]`
    pub mask: Option<Vec<u8>>,
}

fn f32s(t: &Tensor) -> Vec<f32> {
    match t.dtype() {
        DType::F32 => t.to_vec::<f32>().unwrap(),
        _ => (0..t.numel()).map(|i| t.get_f64(i).0 as f32).collect(),
    }
}

fn floats(v: Option<&Value>) -> Option<Vec<f64>> {
    match v? {
        Value::List(x) | Value::Tuple(x) => x.iter().map(|e| match e {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }).collect(),
        Value::Tensor(t) => Some((0..t.numel()).map(|i| t.get_f64(i).0).collect()),
        _ => None,
    }
}

fn float(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        Value::Tensor(t) if t.numel() == 1 => Some(t.get_f64(0).0),
        _ => None,
    }
}

impl Flow {
    pub fn nv(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// from velocity `[F, 3, Z, Y, X]` (C-order; also `[3, Z, Y, X]` = one frame) and optional magnitude
    pub fn new(vel: &Tensor, mag: Option<&Tensor>, spacing: [f64; 3], dt: f64) -> Result<Self, Error> {
        let s = vel.shape();
        let (nf, z, y, x) = match s.len() {
            5 if s[1] == 3 => (s[0], s[2], s[3], s[4]),
            4 if s[0] == 3 => (1, s[1], s[2], s[3]),
            _ => return inv(format!("flow: vel must be hasty [X, Y, Z, 3, F] (C-order [F, 3, Z, Y, X]), got C-order {s:?}")),
        };
        let nv = x * y * z;
        let mag = match mag {
            Some(m) if m.numel() == nf * nv => Some(f32s(m)),
            Some(m) => return inv(format!("flow: mag has {} values, expected {} (hasty [X, Y, Z, F])", m.numel(), nf * nv)),
            None => None,
        };
        if spacing.iter().any(|&d| d <= 0.0) || dt <= 0.0 {
            return inv("flow: spacing and dt must be positive");
        }
        Ok(Self { dims: [x, y, z], nf, spacing, dt, venc: None, vel: f32s(vel), mag, mask: None })
    }

    pub fn from_value(v: &Value) -> Result<Self, Error> {
        let vel = v.get("vel").and_then(|t| t.as_tensor()).ok_or_else(|| Error::Invalid("flow value: no \"vel\" tensor".into()))?;
        let sp = floats(v.get("spacing")).filter(|s| s.len() == 3).ok_or_else(|| Error::Invalid("flow value: \"spacing\" [dx, dy, dz] (m) missing".into()))?;
        let dt = float(v.get("dt")).ok_or_else(|| Error::Invalid("flow value: \"dt\" (s per frame) missing".into()))?;
        let mut f = Self::new(vel, v.get("mag").and_then(|t| t.as_tensor()), [sp[0], sp[1], sp[2]], dt)?;
        f.venc = float(v.get("venc"));
        if let Some(m) = v.get("mask").and_then(|t| t.as_tensor()) {
            if m.numel() != f.nv() {
                return inv("flow value: mask must be [X, Y, Z]");
            }
            f.mask = Some((0..m.numel()).map(|i| (m.get_f64(i).0 != 0.0) as u8).collect());
        }
        Ok(f)
    }

    pub fn to_value(&self) -> Value {
        let [x, y, z] = self.dims;
        let mut d = vec![
            ("kind".to_string(), Value::Str("flow4d".into())),
            ("vel".to_string(), Value::Tensor(Tensor::from_slice(&[self.nf, 3, z, y, x], &self.vel).unwrap())),
        ];
        if let Some(m) = &self.mag {
            d.push(("mag".into(), Value::Tensor(Tensor::from_slice(&[self.nf, z, y, x], m).unwrap())));
        }
        d.push(("spacing".into(), Value::List(self.spacing.iter().map(|&s| Value::Float(s)).collect())));
        d.push(("dt".into(), Value::Float(self.dt)));
        if let Some(v) = self.venc {
            d.push(("venc".into(), Value::Float(v)));
        }
        if let Some(m) = &self.mask {
            d.push(("mask".into(), Value::Tensor(Tensor::from_slice(&[z, y, x], m).unwrap())));
        }
        Value::Dict(d)
    }

    #[inline]
    fn at(&self, f: usize, c: usize, i: usize, j: usize, k: usize) -> f64 {
        let [nx, ny, nz] = self.dims;
        self.vel[((f * 3 + c) * nz + k) * ny * nx + j * nx + i] as f64
    }

    /// velocity (m/s) at voxel coordinates `p`, frame time `t` (frames, cyclic): trilinear in space, linear in time;
    /// None outside the volume (voxel centres span [0, n − 1])
    pub fn sample(&self, p: [f64; 3], t: f64) -> Option<[f64; 3]> {
        let mut i0 = [0usize; 3];
        let mut w = [0f64; 3];
        for a in 0..3 {
            let n = self.dims[a];
            if !(p[a] >= 0.0 && p[a] <= (n - 1) as f64) {
                return None;
            }
            let fl = (p[a].floor() as usize).min(n.saturating_sub(2));
            i0[a] = fl;
            w[a] = if n == 1 { 0.0 } else { p[a] - fl as f64 };
        }
        let nf = self.nf as f64;
        let tc = t.rem_euclid(nf);
        let f0 = (tc.floor() as usize).min(self.nf - 1);
        let f1 = (f0 + 1) % self.nf;
        let wt = tc - f0 as f64;
        let mut v = [0f64; 3];
        let d = |a: usize| if self.dims[a] > 1 { 1 } else { 0 };
        for (f, wf) in [(f0, 1.0 - wt), (f1, wt)] {
            if wf == 0.0 {
                continue;
            }
            for c in 0..3 {
                let mut acc = 0.0;
                for dk in 0..=d(2) {
                    for dj in 0..=d(1) {
                        for di in 0..=d(0) {
                            let wv = (if di == 1 { w[0] } else { 1.0 - w[0] }) * (if dj == 1 { w[1] } else { 1.0 - w[1] }) * (if dk == 1 { w[2] } else { 1.0 - w[2] });
                            acc += wv * self.at(f, c, i0[0] + di, i0[1] + dj, i0[2] + dk);
                        }
                    }
                }
                v[c] += wf * acc;
            }
        }
        Some(v)
    }

    /// nearest-voxel value of a [Z, Y, X] field (None outside)
    fn nearest<T: Copy>(&self, field: &[T], p: [f64; 3]) -> Option<T> {
        let mut q = 0;
        let mut stride = 1;
        for a in 0..3 {
            let i = p[a].round();
            if i < 0.0 || i > (self.dims[a] - 1) as f64 {
                return None;
            }
            q += i as usize * stride;
            stride *= self.dims[a];
        }
        Some(field[q])
    }

    pub fn speed(&self, f: usize) -> Vec<f32> {
        let nv = self.nv();
        (0..nv).map(|q| (0..3).map(|c| self.vel[(f * 3 + c) * nv + q].powi(2)).sum::<f32>().sqrt()).collect()
    }

    /// PC-MRA `[Z, Y, X]`: `mag_speed` = time mean of mag · speed (default; speed alone without mag),
    /// `speed` = time-mean speed, `speed_max` = max over time
    pub fn pcmra(&self, method: &str) -> Vec<f32> {
        let nv = self.nv();
        let mut out = vec![0f32; nv];
        for f in 0..self.nf {
            let sp = self.speed(f);
            for q in 0..nv {
                let v = match (method, &self.mag) {
                    ("speed_max", _) => {
                        out[q] = out[q].max(sp[q]);
                        continue;
                    }
                    ("mag_speed", Some(m)) => m[f * nv + q].abs() * sp[q],
                    _ => sp[q],
                };
                out[q] += v / self.nf as f32;
            }
        }
        out
    }

    /// seeding / stopping region `[Z, Y, X]`: the mask, else PC-MRA above its `quantile`
    pub fn region(&self, quantile: f64) -> Vec<bool> {
        if let Some(m) = &self.mask {
            return m.iter().map(|&b| b != 0).collect();
        }
        let p = self.pcmra("mag_speed");
        let mut s: Vec<f32> = p.iter().copied().filter(|v| v.is_finite()).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = s.get(((s.len() as f64 - 1.0) * quantile.clamp(0.0, 1.0)) as usize).copied().unwrap_or(0.0);
        p.iter().map(|&v| v > thr).collect()
    }

    /// voxel-space velocity (voxels / s)
    fn vox(&self, v: [f64; 3]) -> [f64; 3] {
        [v[0] / self.spacing[0], v[1] / self.spacing[1], v[2] / self.spacing[2]]
    }
}

/// deterministic xorshift
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// orthonormal basis (u, w) of the plane ⟂ n (physical directions)
fn plane_basis(n: [f64; 3]) -> [[f64; 3]; 2] {
    let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let u = norm(cross(n, a));
    [u, cross(n, u)]
}

fn unit(n: [f64; 3]) -> Result<[f64; 3], Error> {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if !(l > 0.0) {
        return inv("normal must be nonzero");
    }
    Ok([n[0] / l, n[1] / l, n[2] / l])
}

/// Seed points (voxel coordinates).
pub enum Seeds {
    /// `n` random points in region voxels (jittered within the voxel)
    Region { n: usize, quantile: f64 },
    /// points on a disc (centre in voxel coordinates, physical normal, radius m), restricted to the region
    Plane { center: [f64; 3], normal: [f64; 3], radius: f64, n: usize, quantile: f64 },
    Points(Vec<[f64; 3]>),
}

impl Flow {
    pub fn seeds(&self, s: &Seeds, seed: u64) -> Result<Vec<[f64; 3]>, Error> {
        let mut rng = Rng(seed.max(1) ^ 0x9E37_79B9_7F4A_7C15);
        Ok(match s {
            Seeds::Points(p) => p.clone(),
            Seeds::Region { n, quantile } => {
                let r = self.region(*quantile);
                let idx: Vec<usize> = (0..r.len()).filter(|&q| r[q]).collect();
                if idx.is_empty() {
                    return inv("seeding region is empty (lower the quantile or give a mask)");
                }
                let [nx, ny, _] = self.dims;
                (0..*n)
                    .map(|_| {
                        let q = idx[((rng.u() * idx.len() as f64) as usize).min(idx.len() - 1)];
                        let c = [(q % nx) as f64, ((q / nx) % ny) as f64, (q / (nx * ny)) as f64];
                        [0, 1, 2].map(|a| (c[a] + rng.u() - 0.5).clamp(0.0, (self.dims[a] - 1) as f64))
                    })
                    .collect()
            }
            Seeds::Plane { center, normal, radius, n, quantile } => {
                let r = self.region(*quantile);
                let nrm = unit(*normal)?;
                let [u, w] = plane_basis(nrm);
                let mut out = vec![];
                let mut tries = 0;
                while out.len() < *n && tries < 50 * n {
                    tries += 1;
                    let (a, b) = (2.0 * rng.u() - 1.0, 2.0 * rng.u() - 1.0);
                    if a * a + b * b > 1.0 {
                        continue;
                    }
                    let p = [0, 1, 2].map(|k| center[k] + radius * (a * u[k] + b * w[k]) / self.spacing[k]);
                    if self.nearest(&r, p) == Some(true) {
                        out.push(p);
                    }
                }
                out
            }
        })
    }
}

/// Polylines: points (voxel coordinates), per-point speed (m/s) and time (s; pathlines), offsets (L + 1).
#[derive(Default)]
pub struct Lines {
    pub points: Vec<[f32; 3]>,
    pub speed: Vec<f32>,
    pub time: Vec<f32>,
    pub offsets: Vec<u32>,
}

impl Lines {
    fn push(&mut self, l: Vec<([f64; 3], f64, f64)>) {
        if l.len() < 2 {
            return;
        }
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        for (p, s, t) in l {
            self.points.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            self.speed.push(s as f32);
            self.time.push(t as f32);
        }
        self.offsets.push(self.points.len() as u32);
    }

    pub fn count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn to_value(&self) -> Value {
        let n = self.points.len();
        let flat: Vec<f32> = self.points.iter().flatten().copied().collect();
        let off = if self.offsets.is_empty() { vec![0u32] } else { self.offsets.clone() };
        Value::Dict(vec![
            ("kind".into(), Value::Str("lines".into())),
            ("points".into(), Value::Tensor(Tensor::from_slice(&[n, 3], &flat).unwrap())),
            ("speed".into(), Value::Tensor(Tensor::from_slice(&[n], &self.speed).unwrap())),
            ("time".into(), Value::Tensor(Tensor::from_slice(&[n], &self.time).unwrap())),
            ("offsets".into(), Value::Tensor(Tensor::from_slice(&[off.len()], &off).unwrap())),
        ])
    }
}

pub struct TraceOpts {
    /// streamlines: step in voxels (arc length); pathlines: substeps per frame interval
    pub step: f64,
    pub max_steps: usize,
    /// stop below this speed (m/s; streamlines)
    pub min_speed: f64,
    /// stop when leaving the region (`quantile` / mask), dilated by one voxel
    pub stop_outside: Option<f64>,
}

impl Flow {
    fn stop_region(&self, o: &TraceOpts) -> Option<Vec<bool>> {
        let q = o.stop_outside?;
        let r = self.region(q);
        // dilate by one voxel: lines may run along the wall
        let [nx, ny, nz] = self.dims;
        let mut d = r.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let q0 = i + nx * (j + ny * k);
                    if r[q0] {
                        continue;
                    }
                    let mut any = false;
                    for (di, dj, dk) in [(-1i64, 0i64, 0i64), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                        let (a, b, c) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                        if a >= 0 && b >= 0 && c >= 0 && (a as usize) < nx && (b as usize) < ny && (c as usize) < nz && r[a as usize + nx * (b as usize + ny * c as usize)] {
                            any = true;
                        }
                    }
                    d[q0] = any;
                }
            }
        }
        Some(d)
    }

    /// Instantaneous streamlines of frame `frame` (fractional: interpolated), traced both ways from each seed with RK4
    /// on the unit direction field, `step` voxels per step.
    pub fn streamlines(&self, frame: f64, seeds: &[[f64; 3]], o: &TraceOpts) -> Lines {
        let stop = self.stop_region(o);
        let inside = |p: [f64; 3]| stop.as_ref().is_none_or(|r| self.nearest(r, p) == Some(true));
        let dir = |p: [f64; 3]| -> Option<([f64; 3], f64)> {
            let v = self.sample(p, frame)?;
            let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if s < o.min_speed.max(1e-12) {
                return None;
            }
            let u = self.vox(v);
            let l = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            Some(([u[0] / l, u[1] / l, u[2] / l], s))
        };
        let trace = |p0: [f64; 3], sign: f64| -> Vec<([f64; 3], f64)> {
            let mut out = vec![];
            let mut p = p0;
            let h = o.step * sign;
            for _ in 0..o.max_steps {
                let Some((k1, s)) = dir(p) else { break };
                if !inside(p) {
                    break;
                }
                out.push((p, s));
                let add = |p: [f64; 3], k: [f64; 3], f: f64| [p[0] + f * k[0], p[1] + f * k[1], p[2] + f * k[2]];
                let Some((k2, _)) = dir(add(p, k1, h / 2.0)) else { break };
                let Some((k3, _)) = dir(add(p, k2, h / 2.0)) else { break };
                let Some((k4, _)) = dir(add(p, k3, h)) else { break };
                p = [0, 1, 2].map(|a| p[a] + h / 6.0 * (k1[a] + 2.0 * k2[a] + 2.0 * k3[a] + k4[a]));
            }
            out
        };
        let lines: Vec<Vec<([f64; 3], f64, f64)>> = seeds
            .par_iter()
            .map(|&s| {
                let mut back = trace(s, -1.0);
                back.reverse();
                let fwd = trace(s, 1.0);
                let skip = if back.is_empty() { 0 } else { 1 }; // the seed is in both
                back.into_iter().chain(fwd.into_iter().skip(skip)).map(|(p, s)| (p, s, frame * self.dt)).collect()
            })
            .collect();
        let mut out = Lines::default();
        for l in lines {
            out.push(l);
        }
        out
    }

    /// Pathlines: particles released at `t0` (frames) from each seed, advected with RK4 through space and cyclic
    /// time for `duration` frames, `step` = substeps per frame interval.
    pub fn pathlines(&self, t0: f64, duration: f64, seeds: &[[f64; 3]], o: &TraceOpts) -> Lines {
        let stop = self.stop_region(o);
        let inside = |p: [f64; 3]| stop.as_ref().is_none_or(|r| self.nearest(r, p) == Some(true));
        // equal steps of ≤ 1 / substeps frames that end exactly at t0 + duration
        let nsteps = ((duration * o.step.max(1.0)).ceil() as usize).clamp(1, o.max_steps);
        let h = duration / nsteps as f64; // frames per step
        let hs = h * self.dt; // seconds per step
        let vel = |p: [f64; 3], t: f64| self.sample(p, t).map(|v| (self.vox(v), (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()));
        let lines: Vec<Vec<([f64; 3], f64, f64)>> = seeds
            .par_iter()
            .map(|&s| {
                let mut out = vec![];
                let (mut p, mut t) = (s, t0);
                for _ in 0..=nsteps {
                    let Some((k1, sp)) = vel(p, t) else { break };
                    if !inside(p) {
                        break;
                    }
                    out.push((p, sp, t * self.dt));
                    let add = |p: [f64; 3], k: [f64; 3], f: f64| [p[0] + f * k[0], p[1] + f * k[1], p[2] + f * k[2]];
                    let Some((k2, _)) = vel(add(p, k1, hs / 2.0), t + h / 2.0) else { break };
                    let Some((k3, _)) = vel(add(p, k2, hs / 2.0), t + h / 2.0) else { break };
                    let Some((k4, _)) = vel(add(p, k3, hs), t + h) else { break };
                    p = [0, 1, 2].map(|a| p[a] + hs / 6.0 * (k1[a] + 2.0 * k2[a] + 2.0 * k3[a] + k4[a]));
                    t += h;
                }
                out
            })
            .collect();
        let mut out = Lines::default();
        for l in lines {
            out.push(l);
        }
        out
    }

    /// Flow through a disc: centre (voxel coordinates), normal (physical), radius (m), sampled on an n × n grid.
    /// Region inside the disc: the mask / PC-MRA region if `quantile` is given, else the whole disc.
    /// Returns (summary JSON, through-plane velocity [F, n, n] (0 outside), region [n, n]).
    pub fn plane(&self, center: [f64; 3], normal: [f64; 3], radius: f64, n: usize, quantile: Option<f64>) -> Result<(Json, Vec<f32>, Vec<u8>), Error> {
        let nrm = unit(normal)?;
        let [u, w] = plane_basis(nrm);
        let ds = 2.0 * radius / n as f64;
        let region = quantile.map(|q| self.region(q));
        let pts: Vec<Option<[f64; 3]>> = (0..n * n)
            .map(|i| {
                let (a, b) = ((i % n) as f64 + 0.5 - n as f64 / 2.0, (i / n) as f64 + 0.5 - n as f64 / 2.0);
                if (a * a + b * b) * ds * ds > radius * radius {
                    return None;
                }
                let p = [0, 1, 2].map(|k| center[k] + ds * (a * u[k] + b * w[k]) / self.spacing[k]);
                if let Some(r) = &region {
                    if self.nearest(r, p) != Some(true) {
                        return None;
                    }
                }
                self.sample(p, 0.0).map(|_| p)
            })
            .collect();
        let inreg: Vec<u8> = pts.iter().map(|p| p.is_some() as u8).collect();
        let na = inreg.iter().filter(|&&b| b == 1).count();
        let mut img = vec![0f32; self.nf * n * n];
        let (mut q, mut vmean, mut vpeak) = (vec![0f64; self.nf], vec![0f64; self.nf], vec![0f64; self.nf]);
        for f in 0..self.nf {
            for (i, p) in pts.iter().enumerate() {
                if let Some(p) = p {
                    let v = self.sample(*p, f as f64).unwrap();
                    let vn = v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2];
                    img[f * n * n + i] = vn as f32;
                    q[f] += vn * ds * ds;
                    if vn.abs() > vpeak[f].abs() {
                        vpeak[f] = vn;
                    }
                }
            }
            vmean[f] = if na > 0 { q[f] / (na as f64 * ds * ds) } else { 0.0 };
        }
        let (fwd, bwd): (f64, f64) = q.iter().fold((0.0, 0.0), |(a, b), &x| if x >= 0.0 { (a + x * self.dt, b) } else { (a, b - x * self.dt) });
        let ml = 1e6;
        let j = json!({
            "frames": self.nf, "dt_s": self.dt, "area_mm2": na as f64 * ds * ds * 1e6, "samples": na, "grid": n, "pixel_mm": ds * 1e3,
            "normal": nrm, "center": center, "u": u, "w": w,
            "layout": "image[f][iy][ix]: sample p = center + ds ((ix + 0.5 − n/2) u + (iy + 0.5 − n/2) w), physical directions",
            "flow_ml_s": q.iter().map(|x| x * ml).collect::<Vec<_>>(),
            "mean_velocity_m_s": vmean, "peak_velocity_m_s": vpeak,
            "forward_ml": fwd * ml, "backward_ml": bwd * ml, "net_ml": (fwd - bwd) * ml,
            "regurgitant_fraction": if fwd > 0.0 { bwd / fwd } else { 0.0 },
        });
        Ok((j, img, inreg))
    }
}

fn arr3(o: &Json, k: &str) -> Option<[f64; 3]> {
    let a = o[k].as_array()?;
    (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0)))
}

fn flow_in(name: &str, inp: &[Arc<Value>]) -> Result<Flow, Error> {
    Flow::from_value(inp.first().ok_or_else(|| Error::Invalid(format!("{name}: one flow value input")))?)
}

fn seeds_from(o: &Json, f: &Flow) -> Result<Vec<[f64; 3]>, Error> {
    let n = o["n"].as_u64().unwrap_or(500) as usize;
    let quantile = o["quantile"].as_f64().unwrap_or(0.97);
    let s = if let Some(p) = o["points"].as_array() {
        Seeds::Points(p.iter().filter_map(|x| x.as_array().filter(|a| a.len() == 3).map(|a| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0)))).collect())
    } else if let (Some(c), Some(nr)) = (arr3(o, "center"), arr3(o, "normal")) {
        Seeds::Plane { center: c, normal: nr, radius: o["radius_mm"].as_f64().unwrap_or(15.0) * 1e-3, n, quantile }
    } else {
        Seeds::Region { n, quantile }
    };
    f.seeds(&s, o["seed"].as_u64().unwrap_or(1))
}

fn stop_opt(o: &Json) -> Option<f64> {
    match &o["stop_outside"] {
        Json::Bool(false) => None,
        Json::Number(x) => x.as_f64(),
        _ => Some(o["quantile"].as_f64().unwrap_or(0.97) - 0.05).map(|q: f64| q.max(0.0)),
    }
}

pub fn register(r: &mut Registry) {
    r.register(
        "flow_make",
        "flow value from tensors: inputs [vel f32 hasty [X,Y,Z,3,F] (C [F,3,Z,Y,X]) or [X,Y,Z,3], optional mag [X,Y,Z,F], optional mask [X,Y,Z]]; options {spacing: [dx,dy,dz] m (default 2 mm), dt: s per frame (default 0.04), venc}",
        |_, o, inp| {
            let vel = inp.first().and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("flow_make: input 0 = vel tensor".into()))?;
            let sp = arr3(o, "spacing").unwrap_or([2e-3; 3]);
            let mut f = Flow::new(vel, inp.get(1).and_then(|v| v.as_tensor()), sp, o["dt"].as_f64().unwrap_or(0.04))?;
            f.venc = o["venc"].as_f64();
            if let Some(m) = inp.get(2).and_then(|v| v.as_tensor()) {
                f.mask = Some((0..m.numel()).map(|i| (m.get_f64(i).0 != 0.0) as u8).collect());
            }
            Ok((String::new(), vec![f.to_value()]))
        },
    );
    r.register("flow_info", "flow value summary (message JSON: dims, frames, spacing, dt, venc, max / p99 speed)", |_, _, inp| {
        let f = flow_in("flow_info", inp)?;
        let mut sp: Vec<f32> = (0..f.nf).flat_map(|t| f.speed(t)).collect();
        sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = sp.get(sp.len() * 99 / 100).copied().unwrap_or(0.0);
        // mean speed per frame in the vessel region (mask / PC-MRA top 3 %): the cardiac phase (peak systole = argmax)
        let reg = f.region(0.97);
        let nreg = reg.iter().filter(|&&b| b).count().max(1);
        let per_frame: Vec<f64> = (0..f.nf).map(|t| f.speed(t).iter().zip(&reg).filter(|(_, &r)| r).map(|(&v, _)| v as f64).sum::<f64>() / nreg as f64).collect();
        let peak = per_frame.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
        let j = json!({"dims": f.dims, "frames": f.nf, "spacing": f.spacing, "dt": f.dt, "venc": f.venc, "max_speed": sp.last(), "p99_speed": p99, "p999_speed": sp.get(sp.len() * 999 / 1000), "has_mag": f.mag.is_some(), "has_mask": f.mask.is_some(), "region_mean_speed": per_frame, "peak_frame": peak, "region_voxels": nreg});
        Ok((j.to_string(), vec![]))
    });
    r.register("flow_pcmra", "PC-MRA f32 hasty [X,Y,Z]: options {method: mag_speed (time mean of |mag|·speed, default) | speed (time mean) | speed_max}", |_, o, inp| {
        let f = flow_in("flow_pcmra", inp)?;
        let [x, y, z] = f.dims;
        let p = f.pcmra(o["method"].as_str().unwrap_or("mag_speed"));
        Ok((String::new(), vec![Value::Tensor(Tensor::from_slice(&[z, y, x], &p)?)]))
    });
    r.register(
        "flow_streamlines",
        "instantaneous streamlines (lines value: points [N,3] voxel coords, speed, time, offsets): options {frame (fractional ok), n (500), quantile (0.97: seeds in the PC-MRA region above this quantile, or the mask), center+normal+radius_mm (disc seeding) or points, step (0.5 voxel), max_steps (600), min_speed (0.02 m/s), stop_outside (quantile | false), seed}",
        |_, o, inp| {
            let f = flow_in("flow_streamlines", inp)?;
            let seeds = seeds_from(o, &f)?;
            let opts = TraceOpts { step: o["step"].as_f64().unwrap_or(0.5), max_steps: o["max_steps"].as_u64().unwrap_or(600) as usize, min_speed: o["min_speed"].as_f64().unwrap_or(0.02), stop_outside: stop_opt(o) };
            let l = f.streamlines(o["frame"].as_f64().unwrap_or(0.0), &seeds, &opts);
            Ok((json!({"lines": l.count(), "points": l.points.len(), "seeds": seeds.len()}).to_string(), vec![l.to_value()]))
        },
    );
    r.register(
        "flow_pathlines",
        "pathlines (lines value with per-point time in s): options {t0 (release frame, 0), releases (1: also release at t0 + k·F/releases), duration (frames, default one cycle), substeps (4 per frame), n, quantile, center+normal+radius_mm or points, stop_outside, seed}",
        |_, o, inp| {
            let f = flow_in("flow_pathlines", inp)?;
            let seeds = seeds_from(o, &f)?;
            let opts = TraceOpts { step: o["substeps"].as_f64().unwrap_or(4.0), max_steps: 100_000, min_speed: 0.0, stop_outside: stop_opt(o) };
            let t0 = o["t0"].as_f64().unwrap_or(0.0);
            let dur = o["duration"].as_f64().unwrap_or(f.nf as f64);
            let rel = o["releases"].as_u64().unwrap_or(1).max(1) as usize;
            let mut all = Lines::default();
            for r in 0..rel {
                let tr = t0 + r as f64 * f.nf as f64 / rel as f64;
                let l = f.pathlines(tr, dur, &seeds, &opts);
                for i in 0..l.count() {
                    let (a, b) = (l.offsets[i] as usize, l.offsets[i + 1] as usize);
                    all.push((a..b).map(|k| ([l.points[k][0] as f64, l.points[k][1] as f64, l.points[k][2] as f64], l.speed[k] as f64, l.time[k] as f64)).collect());
                }
            }
            Ok((json!({"lines": all.count(), "points": all.points.len(), "seeds": seeds.len(), "releases": rel, "cycle_s": f.nf as f64 * f.dt}).to_string(), vec![all.to_value()]))
        },
    );
    r.register(
        "flow_plane",
        "flow through an analysis disc: options {center [x,y,z] voxel coords, normal [3] (physical, +flow direction), radius_mm (15), grid (64), quantile (segment the lumen by the PC-MRA region; omit for the whole disc)}; message JSON (flow_ml_s per frame, mean / peak velocity, forward / backward / net ml, regurgitant fraction, area), outputs [through-plane velocity f32 [F, n, n], region u8 [n, n]]",
        |_, o, inp| {
            let f = flow_in("flow_plane", inp)?;
            let c = arr3(o, "center").ok_or_else(|| Error::Invalid("flow_plane: options.center".into()))?;
            let nr = arr3(o, "normal").ok_or_else(|| Error::Invalid("flow_plane: options.normal".into()))?;
            let n = o["grid"].as_u64().unwrap_or(64) as usize;
            let (j, img, reg) = f.plane(c, nr, o["radius_mm"].as_f64().unwrap_or(15.0) * 1e-3, n, o["quantile"].as_f64())?;
            Ok((j.to_string(), vec![Value::Tensor(Tensor::from_slice(&[f.nf, n, n], &img)?), Value::Tensor(Tensor::from_slice(&[n, n], &reg)?)]))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// field from a function of (physical position m, time s), dims, spacing
    fn field(dims: [usize; 3], nf: usize, h: f64, dt: f64, f: impl Fn([f64; 3], f64) -> [f64; 3]) -> Flow {
        let [nx, ny, nz] = dims;
        let nv = nx * ny * nz;
        let mut vel = vec![0f32; nf * 3 * nv];
        for t in 0..nf {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let v = f([i as f64 * h, j as f64 * h, k as f64 * h], t as f64 * dt);
                        for c in 0..3 {
                            vel[(t * 3 + c) * nv + i + nx * (j + ny * k)] = v[c] as f32;
                        }
                    }
                }
            }
        }
        let vt = Tensor::from_slice(&[nf, 3, nz, ny, nx], &vel).unwrap();
        Flow::new(&vt, None, [h; 3], dt).unwrap()
    }

    fn opts(step: f64) -> TraceOpts {
        TraceOpts { step, max_steps: 100_000, min_speed: 0.0, stop_outside: None }
    }

    #[test]
    fn uniform_flow_pathline_is_exact() {
        let f = field([40, 10, 10], 8, 2e-3, 0.05, |_, _| [0.1, 0.02, 0.0]);
        // 0.1 m/s = 50 voxels / s; one frame (50 ms) = 2.5 voxels in x, 0.5 in y
        let l = f.pathlines(0.0, 4.0, &[[5.0, 3.0, 4.0]], &opts(4.0));
        let last = *l.points.last().unwrap();
        assert!((last[0] - 15.0).abs() < 1e-4 && (last[1] - 5.0).abs() < 1e-4, "{last:?}");
        assert!((l.time.last().unwrap() - 0.2).abs() < 1e-6);
        assert!((l.speed[0] - (0.1f32.hypot(0.02))).abs() < 1e-6);
    }

    #[test]
    fn solid_body_rotation_closes_and_streamlines_are_circles() {
        // ω about the z axis through the centre (physical); period 2π/ω
        let (n, h) = (41usize, 1e-3);
        let c = 20.0 * h;
        let w = 2.0;
        let f = field([n, n, 3], 6, h, 0.1, |p, _| [-w * (p[1] - c), w * (p[0] - c), 0.0]);
        let r0 = 12.0;
        let seed = [20.0 + r0, 20.0, 1.0];
        // pathline over one period: back at the start
        let period_frames = 2.0 * std::f64::consts::PI / w / f.dt;
        let l = f.pathlines(0.0, period_frames, &[seed], &opts(20.0));
        let last = *l.points.last().unwrap();
        let err = ((last[0] as f64 - seed[0]).powi(2) + (last[1] as f64 - seed[1]).powi(2)).sqrt();
        assert!(err < 0.05, "closure error {err} voxels");
        // streamline: radius stays r0 (trilinear interpolation of a linear field is exact)
        let s = f.streamlines(0.0, &[seed], &TraceOpts { step: 0.25, max_steps: 200, min_speed: 0.0, stop_outside: None });
        let worst = s.points.iter().map(|p| ((p[0] as f64 - 20.0).hypot(p[1] as f64 - 20.0) - r0).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-3, "radius drift {worst}");
        assert!(s.count() == 1 && s.points.len() > 300);
    }

    #[test]
    fn cyclic_time_interpolation() {
        let f = field([4, 4, 4], 4, 1e-3, 0.1, |_, t| [t, 0.0, 0.0]); // frames 0.0, 0.1, 0.2, 0.3 m/s
        assert!((f.sample([1.0, 1.0, 1.0], 1.5).unwrap()[0] - 0.15).abs() < 1e-7);
        // between the last frame and frame 0 (wraps)
        assert!((f.sample([1.0, 1.0, 1.0], 3.5).unwrap()[0] - 0.15).abs() < 1e-7);
        assert!((f.sample([1.0, 1.0, 1.0], -0.5).unwrap()[0] - 0.15).abs() < 1e-7);
        assert!(f.sample([3.5, 1.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn poiseuille_flow_rate() {
        // pipe along z, radius R, v_z = vmax (1 − r²/R²) (1 + 0.5 sin 2πt/T): Q = π R² vmax / 2 · (…)
        let (n, h, rr, vmax) = (61usize, 0.5e-3, 12e-3, 1.0);
        let c = 30.0 * h;
        let nf = 10;
        let dt = 0.08;
        let f = field([n, n, 5], nf, h, dt, |p, t| {
            let r2 = (p[0] - c).powi(2) + (p[1] - c).powi(2);
            let s = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * t / (nf as f64 * dt)).sin();
            [0.0, 0.0, if r2 < rr * rr { vmax * (1.0 - r2 / (rr * rr)) * s } else { 0.0 }]
        });
        let (j, img, reg) = f.plane([30.0, 30.0, 2.0], [0.0, 0.0, 1.0], 14e-3, 128, None).unwrap();
        let q: Vec<f64> = j["flow_ml_s"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        for (t, &qt) in q.iter().enumerate() {
            let s = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * t as f64 / nf as f64).sin();
            let want = std::f64::consts::PI * rr * rr * vmax / 2.0 * s * 1e6;
            assert!((qt / want - 1.0).abs() < 0.01, "frame {t}: {qt:.3} vs {want:.3} ml/s");
        }
        // reversed normal: negative flow, all backward
        let (jr, _, _) = f.plane([30.0, 30.0, 2.0], [0.0, 0.0, -1.0], 14e-3, 128, None).unwrap();
        assert!(jr["net_ml"].as_f64().unwrap() < 0.0 && jr["forward_ml"].as_f64().unwrap() == 0.0);
        assert_eq!(img.len(), nf * 128 * 128);
        assert!(reg.iter().any(|&b| b == 1));
        let peak = j["peak_velocity_m_s"][0].as_f64().unwrap();
        assert!((peak - 1.0).abs() < 0.02, "peak {peak}");
    }

    #[test]
    fn region_seeding_and_value_round_trip() {
        let f = field([20, 20, 6], 3, 1e-3, 0.05, |p, _| [0.0, 0.0, if (p[0] - 0.01).hypot(p[1] - 0.01) < 4e-3 { 0.5 } else { 0.0 }]);
        let s = f.seeds(&Seeds::Region { n: 100, quantile: 0.8 }, 3).unwrap();
        assert_eq!(s.len(), 100);
        assert!(s.iter().all(|p| (p[0] - 10.0).hypot(p[1] - 10.0) < 5.5), "seeds outside the jet");
        let back = Flow::from_value(&f.to_value()).unwrap();
        assert_eq!(back.vel, f.vel);
        assert_eq!(back.dims, f.dims);
        // streamlines in the jet run along z and stop at the volume ends
        let l = f.streamlines(0.0, &s[..5], &TraceOpts { step: 0.5, max_steps: 1000, min_speed: 0.01, stop_outside: Some(0.75) });
        assert_eq!(l.count(), 5);
        for i in 0..5 {
            let (a, b) = (l.offsets[i] as usize, l.offsets[i + 1] as usize);
            assert!((l.points[b - 1][2] - l.points[a][2]).abs() > 4.0);
        }
    }
}
