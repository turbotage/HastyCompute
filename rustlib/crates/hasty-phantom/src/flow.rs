//! Flow fields. [`FlowField`] is what the source builder consumes (a CFD solution will implement it later); the M0
//! model [`CenterlineFlow`] puts a Womersley profile on every branch of the vessel centerline tree, with Murray-law flow
//! splits and pulse-wave delays.

use hasty_mrsim::Vec3;
use num_complex::Complex64;

use crate::subject::Centerline;
use crate::volume::Volume;

/// A time-periodic velocity field in the physical frame.
pub trait FlowField: Sync {
    /// cardiac period the field was generated for, s
    fn period(&self) -> f64;
    /// (velocity m/s, ∂v/∂t m/s²) at `x` for flow time `t` (zero outside the lumen)
    fn eval(&self, x: &Vec3, t: f64) -> (Vec3, Vec3);
    /// (v, ∂v/∂t, Dv/Dt) when the field knows its Lagrangian acceleration analytically; `None` → finite differences
    fn eval_lagrangian(&self, _x: &Vec3, _t: f64) -> Option<(Vec3, Vec3, Vec3)> {
        None
    }
}

/// Lagrangian acceleration Dv/Dt = ∂v/∂t + (v·∇)v, the convective term by central differences along v.
pub fn lagrangian_accel(f: &dyn FlowField, x: &Vec3, t: f64, h: f64) -> (Vec3, Vec3, Vec3) {
    if let Some(r) = f.eval_lagrangian(x, t) {
        return r;
    }
    let (v, dvdt) = f.eval(x, t);
    let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if s == 0.0 {
        return (v, dvdt, dvdt);
    }
    let d = [v[0] / s, v[1] / s, v[2] / s];
    let xp = [x[0] + h * d[0], x[1] + h * d[1], x[2] + h * d[2]];
    let xm = [x[0] - h * d[0], x[1] - h * d[1], x[2] - h * d[2]];
    let (vp, _) = f.eval(&xp, t);
    let (vm, _) = f.eval(&xm, t);
    let a = [0, 1, 2].map(|q| dvdt[q] + s * (vp[q] - vm[q]) / (2.0 * h));
    (v, dvdt, a)
}

// ---------------------------------------------------------------- Bessel J0, J1 of complex argument, overflow-safe
/// J_ν(z) = m · exp(e)
#[derive(Clone, Copy, Debug)]
struct Scaled {
    m: Complex64,
    e: Complex64,
}

fn bessel(nu: u32, z: Complex64) -> Scaled {
    if z.norm() < 20.0 {
        // power series
        let h = z * 0.5;
        let h2 = h * h;
        let mut term = if nu == 0 { Complex64::new(1.0, 0.0) } else { h };
        let mut sum = term;
        for k in 1..200 {
            term = -term * h2 / ((k * (k + nu as usize)) as f64);
            sum += term;
            if term.norm() < 1e-17 * sum.norm() {
                break;
            }
        }
        Scaled { m: sum, e: Complex64::new(0.0, 0.0) }
    } else {
        // Hankel asymptotics, written as m·exp(−iχ) with |m| bounded for Im z ≥ 0
        let mu = 4.0 * (nu * nu) as f64;
        let w = z * 8.0;
        let p = Complex64::new(1.0, 0.0) - (mu - 1.0) * (mu - 9.0) / (2.0 * w * w) + (mu - 1.0) * (mu - 9.0) * (mu - 25.0) * (mu - 49.0) / (24.0 * w * w * w * w);
        let q = (mu - 1.0) / w - (mu - 1.0) * (mu - 9.0) * (mu - 25.0) / (6.0 * w * w * w);
        let chi = z - (nu as f64 * 0.5 + 0.25) * std::f64::consts::PI;
        let i = Complex64::new(0.0, 1.0);
        let pre = (2.0 / (std::f64::consts::PI * z)).sqrt();
        let m = pre * 0.5 * ((p - i * q) + (p + i * q) * (2.0 * i * chi).exp());
        Scaled { m, e: -i * chi }
    }
}

fn ratio(a: Scaled, b: Scaled) -> Complex64 {
    a.m / b.m * (a.e - b.e).exp()
}

/// Womersley profile shape of harmonic with Womersley number `alpha` at normalised radius `r` (mean over the disc = 1).
pub fn womersley_shape(alpha: f64, r: f64) -> Complex64 {
    if alpha == 0.0 {
        return Complex64::new(2.0 * (1.0 - r * r), 0.0);
    }
    let lam = Complex64::from_polar(alpha, 0.75 * std::f64::consts::PI); // α i^{3/2}
    let j0 = bessel(0, lam);
    let j1 = bessel(1, lam);
    let num = Complex64::new(1.0, 0.0) - ratio(bessel(0, lam * r), j0);
    let den = Complex64::new(1.0, 0.0) - 2.0 * ratio(j1, j0) / lam;
    num / den
}

// ---------------------------------------------------------------- inflow waveform
/// Aortic-root volume flow: half-sine ejection, short retrograde lobe at valve closure, zero diastole.
#[derive(Clone, Copy, Debug)]
pub struct AorticWaveform {
    /// stroke volume, m³
    pub sv: f64,
    /// ejection duration, s
    pub t_sys: f64,
    /// retrograde lobe: relative amplitude and duration
    pub back: f64,
    pub t_back: f64,
    /// cardiac period, s
    pub period: f64,
}

impl AorticWaveform {
    pub fn healthy(period: f64) -> Self {
        Self { sv: 75e-6, t_sys: 0.30 * (period / 0.9).sqrt(), back: 0.08, t_back: 0.06, period }
    }
    pub fn q(&self, t: f64) -> f64 {
        let t = t.rem_euclid(self.period);
        let qp = self.sv * std::f64::consts::PI / (2.0 * (self.t_sys - self.back * self.t_back));
        if t < self.t_sys {
            qp * (std::f64::consts::PI * t / self.t_sys).sin()
        } else if t < self.t_sys + self.t_back {
            -self.back * qp * (std::f64::consts::PI * (t - self.t_sys) / self.t_back).sin()
        } else {
            0.0
        }
    }
    /// Q(t) = Re Σ_{n=0}^{nh} c_n e^{i n ω t}
    pub fn harmonics(&self, nh: usize) -> Vec<Complex64> {
        let m = 2048;
        let w = 2.0 * std::f64::consts::PI / self.period;
        (0..=nh)
            .map(|n| {
                let s: Complex64 = (0..m).map(|j| {
                    let t = j as f64 * self.period / m as f64;
                    self.q(t) * Complex64::from_polar(1.0, -(n as f64) * w * t)
                }).sum();
                s * if n == 0 { 1.0 / m as f64 } else { 2.0 / m as f64 }
            })
            .collect()
    }
}

// ---------------------------------------------------------------- centerline flow
struct Branch {
    fraction: f64,
    radius: f64,
    /// profile tables T_n(r) on NR points, n = 0..=nh
    table: Vec<Vec<Complex64>>,
}

const NR: usize = 257;

pub struct CenterlineFlow {
    pub cl: Centerline,
    pub period: f64,
    pub root: usize,
    branch_of_vertex: Vec<usize>,
    delay: Vec<f64>,
    tangent: Vec<Vec3>,
    /// dt̂/ds (curvature × principal normal), 1/m
    curv: Vec<Vec3>,
    adj: Vec<Vec<usize>>,
    branches: Vec<Branch>,
    coef: Vec<Complex64>,
    omega: f64,
    sdf: Volume<f32>,
    /// spatial hash of vertices (cell size)
    cell: f64,
    hash: std::collections::HashMap<(i64, i64, i64), Vec<usize>>,
    /// nearest centerline vertex for every SDF-grid voxel within 1 mm of the lumen (u32::MAX elsewhere)
    nearest: Vec<u32>,
}

impl CenterlineFlow {
    /// `root_hint`: point near the inlet (e.g. the heart centroid); `nu` kinematic viscosity (m²/s); `pwv` pulse-wave
    /// velocity (m/s) for the delay along the tree.
    pub fn new(cl: Centerline, sdf: Volume<f32>, root_hint: Vec3, wave: AorticWaveform, nh: usize, nu: f64, pwv: f64) -> Self {
        let nv = cl.vertices.len();
        let mut adj = vec![Vec::new(); nv];
        for e in &cl.edges {
            adj[e[0]].push(e[1]);
            adj[e[1]].push(e[0]);
        }
        let d2 = |a: &Vec3, b: &Vec3| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
        let root = (0..nv).filter(|&v| adj[v].len() == 1).min_by(|&a, &b| d2(&cl.vertices[a], &root_hint).partial_cmp(&d2(&cl.vertices[b], &root_hint)).unwrap()).unwrap();
        // BFS: parent, path length, branch decomposition (a new branch starts after every junction)
        let mut parent = vec![usize::MAX; nv];
        let mut dist = vec![0.0; nv];
        let mut branch_of_vertex = vec![usize::MAX; nv];
        let mut branch_parent: Vec<Option<usize>> = vec![None];
        let mut order = vec![root];
        branch_of_vertex[root] = 0;
        parent[root] = root;
        let mut head = 0;
        while head < order.len() {
            let v = order[head];
            head += 1;
            let children: Vec<usize> = adj[v].iter().copied().filter(|&w| parent[w] == usize::MAX).collect();
            let junction = children.len() > 1;
            for w in children {
                parent[w] = v;
                dist[w] = dist[v] + d2(&cl.vertices[v], &cl.vertices[w]).sqrt();
                branch_of_vertex[w] = if junction {
                    branch_parent.push(Some(branch_of_vertex[v]));
                    branch_parent.len() - 1
                } else {
                    branch_of_vertex[v]
                };
                order.push(w);
            }
        }
        let nb = branch_parent.len();
        // branch radius: median of interior vertex radii
        let mut radius = vec![0.0; nb];
        for b in 0..nb {
            let mut r: Vec<f64> = (0..nv).filter(|&v| branch_of_vertex[v] == b).map(|v| cl.radius[v]).collect();
            r.sort_by(|a, c| a.partial_cmp(c).unwrap());
            radius[b] = if r.is_empty() { 1e-3 } else { r[r.len() / 2].max(1e-3) };
        }
        // Flow splits. Main path: from the root follow the widest child at each junction. A side subtree leaving the main
        // path gets its Murray (r³) share, capped so that the mean velocity in its first branch at peak inflow is at most
        // V_CAP (radii of small segmented branches are unreliable); inside a side subtree Murray splits apply with the same
        // cap; the main path carries the rest.
        const V_CAP: f64 = 0.8;
        let q_peak = (0..1000).map(|i| wave.q(i as f64 * wave.period / 1000.0)).fold(0.0, f64::max);
        let cap = |b: usize| V_CAP * std::f64::consts::PI * radius[b] * radius[b] / q_peak;
        let kids_of = |p: usize| -> Vec<usize> { (1..nb).filter(|&c| branch_parent[c] == Some(p)).collect() };
        let mut main = vec![false; nb];
        let mut b = 0;
        main[0] = true;
        loop {
            let k = kids_of(b);
            if k.is_empty() {
                break;
            }
            b = *k.iter().max_by(|&&x, &&y| radius[x].partial_cmp(&radius[y]).unwrap()).unwrap();
            main[b] = true;
        }
        let mut fraction = vec![0.0; nb];
        fraction[0] = 1.0;
        for p in 0..nb {
            let kids = kids_of(p);
            if kids.is_empty() {
                continue;
            }
            let sum: f64 = kids.iter().map(|&c| radius[c].powi(3)).sum();
            let mut side_total = 0.0;
            for &c in &kids {
                if !main[c] {
                    fraction[c] = (fraction[p] * radius[c].powi(3) / sum).min(cap(c));
                    side_total += fraction[c];
                }
            }
            for &c in &kids {
                if main[c] {
                    fraction[c] = fraction[p] - side_total;
                }
            }
            if !main[p] {
                // inside a side subtree: renormalise to the parent's flow, keeping the caps where possible
                let tot: f64 = kids.iter().map(|&c| fraction[c]).sum();
                if tot < fraction[p] {
                    let free: Vec<usize> = kids.iter().copied().filter(|&c| fraction[c] < cap(c)).collect();
                    let room: f64 = free.iter().map(|&c| cap(c) - fraction[c]).sum();
                    let add = (fraction[p] - tot).min(room);
                    for &c in &free {
                        fraction[c] += add * (cap(c) - fraction[c]) / room.max(1e-30);
                    }
                }
            }
        }
        // smoothed tangents (±4 neighbours along the tree)
        let tangent: Vec<Vec3> = (0..nv)
            .map(|v| {
                let mut a = v;
                for _ in 0..4 {
                    a = parent[a];
                }
                let mut b = v;
                for _ in 0..4 {
                    if let Some(&c) = adj[b].iter().find(|&&c| parent[c] == b) {
                        b = c;
                    }
                }
                let t = [0, 1, 2].map(|q| cl.vertices[b][q] - cl.vertices[a][q]);
                let n = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt().max(1e-12);
                [t[0] / n, t[1] / n, t[2] / n]
            })
            .collect();
        // curvature vector dt̂/ds from the smoothed tangents ±4 vertices along the tree
        let curv: Vec<Vec3> = (0..nv)
            .map(|v| {
                let mut a = v;
                for _ in 0..4 {
                    a = parent[a];
                }
                let mut b = v;
                for _ in 0..4 {
                    if let Some(&c) = adj[b].iter().find(|&&c| parent[c] == b) {
                        b = c;
                    }
                }
                let ds = dist[b] - dist[a];
                if ds <= 0.0 {
                    return [0.0; 3];
                }
                [0, 1, 2].map(|q| (tangent[b][q] - tangent[a][q]) / ds)
            })
            .collect();
        let coef = wave.harmonics(nh);
        let omega = 2.0 * std::f64::consts::PI / wave.period;
        let branches = (0..nb)
            .map(|b| {
                let table = (0..=nh)
                    .map(|n| {
                        let alpha = radius[b] * (n as f64 * omega / nu).sqrt();
                        (0..NR).map(|i| womersley_shape(alpha, i as f64 / (NR - 1) as f64)).collect()
                    })
                    .collect();
                Branch { fraction: fraction[b], radius: radius[b], table }
            })
            .collect();
        let delay = dist.iter().map(|d| d / pwv).collect();
        let cell = 0.006;
        let mut hash: std::collections::HashMap<(i64, i64, i64), Vec<usize>> = std::collections::HashMap::new();
        for (v, p) in cl.vertices.iter().enumerate() {
            hash.entry(((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64)).or_default().push(v);
        }
        let mut me = Self { cl, period: wave.period, root, branch_of_vertex, delay, tangent, curv, adj, branches, coef, omega, sdf, cell, hash, nearest: Vec::new() };
        me.nearest = me.build_nearest();
        me
    }

    fn build_nearest(&self) -> Vec<u32> {
        let n = self.sdf.len();
        let mut out = vec![u32::MAX; n];
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|sc| {
            for (t, o) in out.chunks_mut(chunk).enumerate() {
                sc.spawn(move || {
                    let d = self.sdf.dims;
                    for (l, v) in o.iter_mut().enumerate() {
                        let q = t * chunk + l;
                        if self.sdf.data[q] < 1e-3 {
                            let x = self.sdf.to_phys([(q % d[0]) as f64, ((q / d[0]) % d[1]) as f64, (q / (d[0] * d[1])) as f64]);
                            if let Some(v0) = self.nearest_vertex(&x) {
                                *v = v0 as u32;
                            }
                        }
                    }
                });
            }
        });
        out
    }

    pub fn n_branches(&self) -> usize {
        self.branches.len()
    }
    /// vertices of branches carrying at least `min_fraction` of the inflow (the main aortic path), and their median radius
    pub fn main_path(&self, min_fraction: f64) -> (Vec<Vec3>, f64) {
        let mut pts = Vec::new();
        let mut r = Vec::new();
        for (v, &b) in self.branch_of_vertex.iter().enumerate() {
            if b != usize::MAX && self.branches[b].fraction >= min_fraction {
                pts.push(self.cl.vertices[v]);
                r.push(self.cl.radius[v]);
            }
        }
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if r.is_empty() { 0.0 } else { r[r.len() / 2] };
        (pts, med)
    }
    /// diagnostics at a point: (branch, fraction, branch radius, foot vertices v/w, s, rho, sdf, inscribed radius)
    pub fn probe(&self, x: &Vec3) -> Option<(usize, f64, f64, usize, usize, f64, f64, f64, f64)> {
        let sd = self.sdf.sample(x, 1.0) as f64;
        let (f, v, w, s) = self.foot(x)?;
        let rho = ((x[0] - f[0]).powi(2) + (x[1] - f[1]).powi(2) + (x[2] - f[2]).powi(2)).sqrt();
        let b = self.branch_of_vertex[v];
        Some((b, self.branches[b].fraction, self.branches[b].radius, v, w, s, rho, sd, self.cl.radius[v]))
    }
    pub fn branch_info(&self) -> Vec<(f64, f64)> {
        self.branches.iter().map(|b| (b.fraction, b.radius)).collect()
    }

    fn nearest_vertex(&self, x: &Vec3) -> Option<usize> {
        let c = ((x[0] / self.cell).floor() as i64, (x[1] / self.cell).floor() as i64, (x[2] / self.cell).floor() as i64);
        let mut best = (f64::MAX, None);
        for r in 1..6i64 {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if let Some(vs) = self.hash.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) {
                            for &v in vs {
                                let p = &self.cl.vertices[v];
                                let d = (p[0] - x[0]).powi(2) + (p[1] - x[1]).powi(2) + (p[2] - x[2]).powi(2);
                                if d < best.0 {
                                    best = (d, Some(v));
                                }
                            }
                        }
                    }
                }
            }
            if best.1.is_some() && best.0.sqrt() < (r as f64 - 1.0) * self.cell {
                break;
            }
        }
        best.1
    }

    /// foot point on the centerline (projection on the incident edges of the nearest vertex) and its vertex weights
    fn foot(&self, x: &Vec3) -> Option<(Vec3, usize, usize, f64)> {
        let g = self.sdf.to_voxel(x);
        let d = self.sdf.dims;
        let gi = [0, 1, 2].map(|a| g[a].round().clamp(0.0, (d[a] - 1) as f64) as usize);
        let nv = self.nearest[gi[0] + d[0] * (gi[1] + d[1] * gi[2])];
        let v = if nv != u32::MAX { nv as usize } else { self.nearest_vertex(x)? };
        let p = &self.cl.vertices[v];
        let mut best = (f64::MAX, *p, v, v, 0.0);
        for &w in &self.adj[v] {
            let q = &self.cl.vertices[w];
            let e = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
            let l2 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            let s = (((x[0] - p[0]) * e[0] + (x[1] - p[1]) * e[1] + (x[2] - p[2]) * e[2]) / l2).clamp(0.0, 1.0);
            let f = [p[0] + s * e[0], p[1] + s * e[1], p[2] + s * e[2]];
            let d = (f[0] - x[0]).powi(2) + (f[1] - x[1]).powi(2) + (f[2] - x[2]).powi(2);
            if d < best.0 {
                best = (d, f, v, w, s);
            }
        }
        Some((best.1, best.2, best.3, best.4))
    }
}

impl FlowField for CenterlineFlow {
    fn period(&self) -> f64 {
        self.period
    }
    /// Dv/Dt = ∂v/∂t + |v|² dt̂/ds (axial flow along the curved centerline; centripetal term)
    fn eval_lagrangian(&self, x: &Vec3, t: f64) -> Option<(Vec3, Vec3, Vec3)> {
        let (v, dvdt) = self.eval(x, t);
        let s2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if s2 == 0.0 {
            return Some((v, dvdt, dvdt));
        }
        let (_, vv, w, s) = self.foot(x)?;
        let k = [0, 1, 2].map(|q| self.curv[vv][q] * (1.0 - s) + self.curv[w][q] * s);
        Some((v, dvdt, [0, 1, 2].map(|q| dvdt[q] + s2 * k[q])))
    }
    fn eval(&self, x: &Vec3, t: f64) -> (Vec3, Vec3) {
        let z = ([0.0; 3], [0.0; 3]);
        let sd = self.sdf.sample(x, 1.0) as f64;
        if sd >= 0.0 {
            return z;
        }
        let Some((f, v, w, s)) = self.foot(x) else { return z };
        let rho = ((x[0] - f[0]).powi(2) + (x[1] - f[1]).powi(2) + (x[2] - f[2]).powi(2)).sqrt();
        let r_loc = rho - sd;
        let rn = (rho / r_loc).clamp(0.0, 1.0);
        let b = self.branch_of_vertex[v];
        let br = &self.branches[b];
        let delay = self.delay[v] * (1.0 - s) + self.delay[w] * s;
        let tan = [0, 1, 2].map(|q| self.tangent[v][q] * (1.0 - s) + self.tangent[w][q] * s);
        let tn = (tan[0] * tan[0] + tan[1] * tan[1] + tan[2] * tan[2]).sqrt().max(1e-12);
        let pos = rn * (NR - 1) as f64;
        let i0 = (pos.floor() as usize).min(NR - 2);
        let fr = pos - i0 as f64;
        // area from the inscribed radius at the foot point (robust at tips/junctions where r_loc degenerates)
        // and clamped around the branch median (skeleton radii shrink to ~0 at path endpoints)
        let r_v = (self.cl.radius[v] * (1.0 - s) + self.cl.radius[w] * s).clamp(0.6 * br.radius, 1.6 * br.radius);
        let area = std::f64::consts::PI * r_v * r_v;
        let tt = t - delay;
        let (mut u, mut du) = (0.0, 0.0);
        let e1 = Complex64::from_polar(1.0, self.omega * tt);
        let mut e = Complex64::new(1.0, 0.0);
        for (n, c) in self.coef.iter().enumerate() {
            let shape = br.table[n][i0] * (1.0 - fr) + br.table[n][i0 + 1] * fr;
            let val = c * shape * e;
            e *= e1;
            u += val.re;
            du += (val * Complex64::new(0.0, n as f64 * self.omega)).re;
        }
        let k = br.fraction / area;
        ([0, 1, 2].map(|q| k * u * tan[q] / tn), [0, 1, 2].map(|q| k * du * tan[q] / tn))
    }
}
