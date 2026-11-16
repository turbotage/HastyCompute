//! Mesh processing beyond the basics of `mesh.rs`: quadric decimation, plane cuts (with caps), centerlines of a mask (TEASAR) and
//! mesh sequences (one surface per frame of a 4-D mask, for moving walls).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use hasty_value::{Tensor, Value};
use ordered_float_lite::F;
use serde_json::{json, Value as Json};

use crate::commands::Registry;
use crate::mesh_fill::{boundary_loops, fill_loops, loops_from_edges};
use crate::mesh::{affine_opt, apply, blur, invert, largest_component, marching_tets, mask_to_mesh, mesh_in, nums, select, space_map, Mesh};
use crate::Error;

/// f64 with a total order (costs in the heaps are never NaN; NaN sorts last)
mod ordered_float_lite {
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct F(pub f64);
    impl Eq for F {}
    impl PartialOrd for F {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for F {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&o.0)
        }
    }
}

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

// ── quadric edge-collapse decimation (Garland & Heckbert) ─────────────────────────────────────────────────────────

type V3 = [f64; 3];
fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn plane_quadric(n: V3, d: f64, w: f64) -> [f64; 10] {
    [n[0] * n[0], n[0] * n[1], n[0] * n[2], n[0] * d, n[1] * n[1], n[1] * n[2], n[1] * d, n[2] * n[2], n[2] * d, d * d].map(|x| x * w)
}

fn q_err(q: &[f64; 10], p: V3) -> f64 {
    q[0] * p[0] * p[0] + 2.0 * q[1] * p[0] * p[1] + 2.0 * q[2] * p[0] * p[2] + 2.0 * q[3] * p[0] + q[4] * p[1] * p[1] + 2.0 * q[5] * p[1] * p[2] + 2.0 * q[6] * p[1] + q[7] * p[2] * p[2] + 2.0 * q[8] * p[2] + q[9]
}

impl Mesh {
    /// Reduce to about `target` triangles by edge collapses ordered by quadric error. The new vertex sits at the cheapest of the two ends and
    /// the midpoint. Collapses that would flip a neighbouring triangle, pinch the surface (link condition) or move an open border vertex are
    /// skipped, so closed surfaces stay closed and manifold. Scalars keep the value of the surviving vertex.
    pub fn decimate(&mut self, target: usize, protect: Option<&[bool]>, scalar_weight: f64) {
        let nv = self.verts.len();
        let mut pos: Vec<V3> = self.verts.iter().map(|v| [v[0] as f64, v[1] as f64, v[2] as f64]).collect();
        let mut faces = self.faces.clone();
        let mut alive = vec![true; faces.len()];
        let mut n_alive = faces.len();
        let mut vf: Vec<Vec<u32>> = vec![Vec::new(); nv];
        for (fi, f) in faces.iter().enumerate() {
            for &v in f {
                vf[v as usize].push(fi as u32);
            }
        }
        // quadrics from the face planes; border edges get a perpendicular penalty plane so open meshes keep their outline
        let mut q = vec![[0f64; 10]; nv];
        let mut edge_count: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        let area_normal = |p: &Vec<V3>, f: [u32; 3]| cross3(sub3(p[f[1] as usize], p[f[0] as usize]), sub3(p[f[2] as usize], p[f[0] as usize]));
        for (fi, f) in faces.iter().enumerate() {
            let c = area_normal(&pos, *f);
            let len = dot3(c, c).sqrt();
            if len > 0.0 {
                let n = [c[0] / len, c[1] / len, c[2] / len];
                let d = -dot3(n, pos[f[0] as usize]);
                let qf = plane_quadric(n, d, len * 0.5);
                for &v in f {
                    for k in 0..10 {
                        q[v as usize][k] += qf[k];
                    }
                }
            }
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                let e = edge_count.entry((a.min(b), a.max(b))).or_insert((0, fi as u32));
                e.0 += 1;
            }
        }
        let mut border = vec![false; nv];
        for (&(a, b), &(c, fi)) in &edge_count {
            if c == 1 {
                border[a as usize] = true;
                border[b as usize] = true;
                let f = faces[fi as usize];
                let c3 = area_normal(&pos, f);
                let len = dot3(c3, c3).sqrt();
                if len > 0.0 {
                    let n = [c3[0] / len, c3[1] / len, c3[2] / len];
                    let e = sub3(pos[b as usize], pos[a as usize]);
                    let mut pn = cross3(e, n);
                    let pl = dot3(pn, pn).sqrt();
                    if pl > 0.0 {
                        pn = [pn[0] / pl, pn[1] / pl, pn[2] / pl];
                        let qf = plane_quadric(pn, -dot3(pn, pos[a as usize]), 100.0 * dot3(e, e).sqrt());
                        for &v in &[a, b] {
                            for k in 0..10 {
                                q[v as usize][k] += qf[k];
                            }
                        }
                    }
                }
            }
        }
        // scalar term: every scalar normalised to its range and scaled to the mean edge length × weight (mm), so merging two vertices whose values differ by the
        // full range costs as much as moving a vertex by that distance: edges across a sharp scalar change are kept longer
        let mean_edge = { let (mut s, mut c) = (0.0, 0usize); for f in &self.faces { for k in 0..3 { let (p, q2) = (pos[f[k] as usize], pos[f[(k + 1) % 3] as usize]); s += dot3(sub3(p, q2), sub3(p, q2)).sqrt(); c += 1; } } if c > 0 { s / c as f64 } else { 0.0 } };
        let sn: Vec<Vec<f64>> = if scalar_weight > 0.0 {
            self.scalars.iter().map(|(_, s)| {
                let f: Vec<f64> = s.iter().filter(|x| x.is_finite()).map(|&x| x as f64).collect();
                let (lo, hi) = (f.iter().cloned().fold(f64::INFINITY, f64::min), f.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
                let k = if hi > lo { mean_edge * scalar_weight / (hi - lo) } else { 0.0 };
                s.iter().map(|&x| if x.is_finite() { (x as f64 - lo) * k } else { 0.0 }).collect()
            }).collect()
        } else { Vec::new() };
        let mut stamp = vec![0u32; nv];
        let mut dead = vec![false; nv];
        let cost = |q: &Vec<[f64; 10]>, pos: &Vec<V3>, a: usize, b: usize| -> (f64, V3) {
            let pen: f64 = sn.iter().map(|s| (s[a] - s[b]).powi(2)).sum();
            let qq: [f64; 10] = std::array::from_fn(|k| q[a][k] + q[b][k]);
            let mid = [(pos[a][0] + pos[b][0]) / 2.0, (pos[a][1] + pos[b][1]) / 2.0, (pos[a][2] + pos[b][2]) / 2.0];
            [pos[a], pos[b], mid].into_iter().map(|p| (q_err(&qq, p) + pen, p)).min_by(|x, y| x.0.total_cmp(&y.0)).unwrap()
        };
        type Entry = Reverse<(F, u32, u32, u32, u32)>;
        let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        for f in &faces {
            for k in 0..3 {
                let (a, b) = (f[k].min(f[(k + 1) % 3]), f[k].max(f[(k + 1) % 3]));
                if seen.insert((a, b)) {
                    heap.push(Reverse((F(cost(&q, &pos, a as usize, b as usize).0.max(0.0)), a, b, 0, 0)));
                }
            }
        }
        let neighbours = |vf: &Vec<Vec<u32>>, alive: &Vec<bool>, faces: &Vec<[u32; 3]>, v: u32| -> HashSet<u32> {
            let mut s = HashSet::new();
            for &fi in &vf[v as usize] {
                if alive[fi as usize] {
                    for &w in &faces[fi as usize] {
                        if w != v {
                            s.insert(w);
                        }
                    }
                }
            }
            s
        };
        while n_alive > target {
            let Some(Reverse((_, a, b, sa, sb))) = heap.pop() else { break };
            let (au, bu) = (a as usize, b as usize);
            if dead[au] || dead[bu] || stamp[au] != sa || stamp[bu] != sb {
                continue;
            }
            // protected vertices (extremes of a scalar) are never merged into another vertex: the protected end survives; two protected ends: no collapse
            let (pa, pb) = (protect.map_or(false, |p| p[au]), protect.map_or(false, |p| p[bu]));
            if pa && pb {
                continue;
            }
            let (a, b, au, bu) = if pb { (b, a, bu, au) } else { (a, b, au, bu) };
            // faces around the edge
            let fa: Vec<u32> = vf[au].iter().copied().filter(|&f| alive[f as usize]).collect();
            let fb: Vec<u32> = vf[bu].iter().copied().filter(|&f| alive[f as usize]).collect();
            let shared: Vec<u32> = fa.iter().copied().filter(|f| faces[*f as usize].contains(&b)).collect();
            if shared.is_empty() || shared.len() > 2 {
                continue;
            }
            if border[au] || border[bu] {
                continue;          // keep open outlines exactly
            }
            // link condition: the common neighbours of a and b are exactly the apexes of the shared faces
            let (na, nb) = (neighbours(&vf, &alive, &faces, a), neighbours(&vf, &alive, &faces, b));
            let common: HashSet<u32> = na.intersection(&nb).copied().collect();
            let apex: HashSet<u32> = shared.iter().flat_map(|&f| faces[f as usize].iter().copied().filter(|&v| v != a && v != b)).collect();
            if common != apex {
                continue;
            }
            let np = if pa || pb { pos[au] } else { cost(&q, &pos, au, bu).1 };
            // no flipped or degenerate triangles around the new vertex
            let mut ok = true;
            for &fi in fa.iter().chain(fb.iter()) {
                if shared.contains(&fi) {
                    continue;
                }
                let f = faces[fi as usize];
                let old = area_normal(&pos, f);
                let mut p = f.map(|v| pos[v as usize]);
                for k in 0..3 {
                    if f[k] == a || f[k] == b {
                        p[k] = np;
                    }
                }
                let new = cross3(sub3(p[1], p[0]), sub3(p[2], p[0]));
                let (lo, ln) = (dot3(old, old).sqrt(), dot3(new, new).sqrt());
                if ln < 1e-12 || lo < 1e-12 || dot3(old, new) / (lo * ln) < 0.2 {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            // collapse b into a
            for &fi in &shared {
                alive[fi as usize] = false;
                n_alive -= 1;
            }
            for &fi in &fb {
                if alive[fi as usize] {
                    for v in faces[fi as usize].iter_mut() {
                        if *v == b {
                            *v = a;
                        }
                    }
                    vf[au].push(fi);
                }
            }
            pos[au] = np;
            for k in 0..10 {
                q[au][k] += q[bu][k];
            }
            dead[bu] = true;
            stamp[au] += 1;
            for w in neighbours(&vf, &alive, &faces, a) {
                let (x, y) = (a.min(w), a.max(w));
                heap.push(Reverse((F(cost(&q, &pos, x as usize, y as usize).0.max(0.0)), x, y, stamp[x as usize], stamp[y as usize])));
            }
        }
        // compact
        let mut map = vec![u32::MAX; nv];
        let mut verts = Vec::new();
        let mut keep_idx = Vec::new();
        let mut out_faces = Vec::new();
        for (fi, f) in faces.iter().enumerate() {
            if !alive[fi] {
                continue;
            }
            let mut g = [0u32; 3];
            for k in 0..3 {
                let v = f[k] as usize;
                if map[v] == u32::MAX {
                    map[v] = verts.len() as u32;
                    verts.push([pos[v][0] as f32, pos[v][1] as f32, pos[v][2] as f32]);
                    keep_idx.push(v);
                }
                g[k] = map[v];
            }
            out_faces.push(g);
        }
        for (_, s) in &mut self.scalars {
            *s = keep_idx.iter().map(|&i| s[i]).collect();
        }
        self.verts = verts;
        self.faces = out_faces;
    }

    /// Keep the part of the surface on one side of a plane (triangles crossing it are split, scalars interpolated). With `cap`, the cut
    /// outline is closed by triangle fans (fine for convex-ish sections such as vessel cross-sections; sections with holes are not handled).
    /// vertices that are a strict local extreme of some scalar among their neighbours and lie in the top or bottom `frac` of its finite range (None without scalars)
    pub fn scalar_extremes(&self, frac: f64) -> Option<Vec<bool>> {
        if self.scalars.is_empty() {
            return None;
        }
        let nv = self.verts.len();
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); nv];
        for f in &self.faces {
            for k in 0..3 {
                adj[f[k] as usize].push(f[(k + 1) % 3]);
                adj[f[(k + 1) % 3] as usize].push(f[k]);
            }
        }
        let mut prot = vec![false; nv];
        for (_, s) in &self.scalars {
            let mut f: Vec<f32> = s.iter().copied().filter(|x| x.is_finite()).collect();
            if f.len() < 10 {
                continue;
            }
            f.sort_by(|a, b| a.total_cmp(b));
            let (lo, hi) = (f[((f.len() - 1) as f64 * frac) as usize], f[((f.len() - 1) as f64 * (1.0 - frac)) as usize]);
            for v in 0..nv {
                let x = s[v];
                if !x.is_finite() || adj[v].is_empty() {
                    continue;
                }
                let (mut gt, mut lt) = (true, true);
                for &u in &adj[v] {
                    let y = s[u as usize];
                    if y.is_finite() {
                        gt &= x > y;
                        lt &= x < y;
                    }
                }
                if (gt && x >= hi) || (lt && x <= lo) {
                    prot[v] = true;
                }
            }
        }
        Some(prot)
    }

    /// close boundary loops of at most `max_edges` edges (each in its own best-fit plane, by ear clipping); returns (filled, remaining open loops)
    pub fn fill_holes(&mut self, max_edges: usize, fair: bool, cotan: bool) -> (usize, usize) {
        let loops = boundary_loops(&self.faces);
        let (small, big): (Vec<_>, Vec<_>) = loops.into_iter().partition(|l| l.len() <= max_edges);
        let n = small.len();
        let from = self.faces.len();
        let add = fill_loops(&self.verts, &small, None);
        self.faces.extend(add);
        if fair && n > 0 {
            self.fair_patch(from, 120, cotan);
        }
        (n, big.len())
    }

    pub fn cut(&mut self, origin: V3, normal: V3, keep_positive: bool, cap: bool) {
        let nl = dot3(normal, normal).sqrt().max(1e-30);
        let n = [normal[0] / nl, normal[1] / nl, normal[2] / nl];
        let sgn = if keep_positive { 1.0 } else { -1.0 };
        let dist: Vec<f64> = self.verts.iter().map(|v| sgn * dot3(sub3([v[0] as f64, v[1] as f64, v[2] as f64], origin), n)).collect();
        let mut verts = self.verts.clone();
        let mut scalars = self.scalars.clone();
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();
        let mut cut_edges: Vec<(u32, u32)> = Vec::new();
        for f in &self.faces {
            let d = f.map(|v| dist[v as usize]);
            let inside = d.iter().filter(|&&x| x >= 0.0).count();
            if inside == 0 {
                continue;
            }
            if inside == 3 {
                faces.push(*f);
                continue;
            }
            let mut mid = |a: u32, b: u32, verts: &mut Vec<[f32; 3]>, scalars: &mut Vec<(String, Vec<f32>)>| -> u32 {
                let key = (a.min(b), a.max(b));
                *cache.entry(key).or_insert_with(|| {
                    let (da, db) = (dist[a as usize], dist[b as usize]);
                    let t = (da / (da - db)) as f32;
                    let (pa, pb) = (verts[a as usize], verts[b as usize]);
                    verts.push([0, 1, 2].map(|c| pa[c] + t * (pb[c] - pa[c])));
                    for (_, s) in scalars.iter_mut() {
                        let (x, y) = (s[a as usize], s[b as usize]);
                        s.push(if x.is_nan() || y.is_nan() { f32::NAN } else { x + t * (y - x) });
                    }
                    (verts.len() - 1) as u32
                })
            };
            // rotate so the vertex alone on its side comes first
            let lone_in = inside == 1;
            let k = (0..3).find(|&k| (d[k] >= 0.0) == lone_in).unwrap();
            let (p0, p1, p2) = (f[k], f[(k + 1) % 3], f[(k + 2) % 3]);
            if lone_in {
                let (a, b) = (mid(p0, p1, &mut verts, &mut scalars), mid(p0, p2, &mut verts, &mut scalars));
                faces.push([p0, a, b]);
                cut_edges.push((a, b));
            } else {
                // p0 outside, p1 and p2 inside
                let (a, b) = (mid(p0, p1, &mut verts, &mut scalars), mid(p0, p2, &mut verts, &mut scalars));
                faces.push([a, p1, p2]);
                faces.push([a, p2, b]);
                cut_edges.push((b, a));
            }
        }
        if cap && !cut_edges.is_empty() {
            // the cut outline (all loops lie in the plane): nested loops are holes (annular sections), the fill reverses the loop direction
            let loops = loops_from_edges(&cut_edges);
            faces.extend(fill_loops(&verts, &loops, Some(n)));
        }
        self.verts = verts;
        self.scalars = scalars;
        self.faces = faces;
        let mut used = vec![false; self.verts.len()];
        self.faces.iter().flatten().for_each(|&v| used[v as usize] = true);
        self.retain_used(&used);
    }

    /// drop vertices not marked in `used` (no face may refer to them) and renumber
    fn retain_used(&mut self, used: &[bool]) {
        let mut map = vec![u32::MAX; self.verts.len()];
        let mut nv = Vec::new();
        for (i, v) in self.verts.iter().enumerate() {
            if used[i] {
                map[i] = nv.len() as u32;
                nv.push(*v);
            }
        }
        for f in &mut self.faces {
            *f = f.map(|i| map[i as usize]);
        }
        for (_, s) in &mut self.scalars {
            *s = s.iter().enumerate().filter(|(i, _)| used[*i]).map(|(_, &x)| x).collect();
        }
        self.verts = nv;
    }
}

// ── centerlines (TEASAR) ─────────────────────────────────────────────────────────────────────────────────────────────

/// squared Euclidean distance (mm²) to the nearest 0 voxel, exact (Felzenszwalb–Huttenlocher), separable per axis with voxel size `h`
fn edt_sq(mask: &[u8], dims: [usize; 3], h: [f64; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let big = 1e20f64;
    let mut d: Vec<f64> = mask.iter().map(|&m| if m != 0 { big } else { 0.0 }).collect();
    let line = |d: &mut Vec<f64>, n: usize, stride: usize, base: usize, hh: f64, f: &mut Vec<f64>, v: &mut Vec<usize>, z: &mut Vec<f64>| {
        for i in 0..n {
            f[i] = d[base + i * stride];
        }
        let mut k = 0usize;
        v[0] = 0;
        z[0] = f64::NEG_INFINITY;
        z[1] = f64::INFINITY;
        for q in 1..n {
            let sq = |x: usize| (x as f64 * hh) * (x as f64 * hh);
            loop {
                let p = v[k];
                let s = ((f[q] + sq(q)) - (f[p] + sq(p))) / (2.0 * (q as f64 - p as f64) * hh);
                if s <= z[k] && k > 0 {
                    k -= 1;
                    continue;
                }
                if s <= z[k] {
                    v[0] = q;
                    break;
                }
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
        let mut k = 0usize;
        for q in 0..n {
            while z[k + 1] < q as f64 * hh {
                k += 1;
            }
            let dx = (q as f64 - v[k] as f64) * hh;
            d[base + q * stride] = dx * dx + f[v[k]];
        }
    };
    let m = nx.max(ny).max(nz);
    let (mut f, mut v, mut z) = (vec![0f64; m], vec![0usize; m], vec![0f64; m + 1]);
    for k in 0..nz {
        for j in 0..ny {
            line(&mut d, nx, 1, (k * ny + j) * nx, h[0], &mut f, &mut v, &mut z);
        }
    }
    for k in 0..nz {
        for i in 0..nx {
            line(&mut d, ny, nx, k * ny * nx + i, h[1], &mut f, &mut v, &mut z);
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            line(&mut d, nz, nx * ny, j * nx + i, h[2], &mut f, &mut v, &mut z);
        }
    }
    d.iter().map(|&x| x.min(1e18) as f32).collect()
}

struct Branch {
    pts: Vec<usize>,      // voxel indices from tip towards the root / junction
}

/// TEASAR skeleton of a binary volume in the cropped box `dims` (voxel size `h` mm): returns the branches (voxel indices into the box) and the
/// distance-to-wall (mm) per voxel
fn teasar(mask: &[u8], dims: [usize; 3], h: [f64; 3], root: Option<[usize; 3]>, scale: f64, const_mm: f64, min_len_mm: f64) -> (Vec<Branch>, Vec<f32>) {
    let [nx, ny, nz] = dims;
    let n = mask.len();
    let dt: Vec<f32> = edt_sq(&mask.iter().map(|&m| (m != 0) as u8).collect::<Vec<_>>(), dims, h).iter().map(|x| x.sqrt()).collect();
    let step: Vec<([i64; 3], f64)> = (-1..=1i64).flat_map(|dz| (-1..=1i64).flat_map(move |dy| (-1..=1i64).map(move |dx| [dx, dy, dz]))).filter(|d| *d != [0, 0, 0])
        .map(|d| (d, ((d[0] as f64 * h[0]).powi(2) + (d[1] as f64 * h[1]).powi(2) + (d[2] as f64 * h[2]).powi(2)).sqrt())).collect();
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    // penalty: cheap on the medial axis, expensive towards the wall. Normalised by the *local* inscribed radius (the largest distance-to-wall within a ball of
    // radius dt around the voxel) rather than the global maximum, so thin branches next to a thick trunk are held on their own axis too. Without this the
    // paths to two daughters leave the trunk side by side and separate far below the bifurcation.
    let rloc: Vec<f32> = (0..n).map(|q| {
        if mask[q] == 0 {
            return 0.0;
        }
        let r = dt[q] as f64 + h[0].min(h[1]).min(h[2]);
        let rv = [(r / h[0]).ceil().min(12.0) as i64, (r / h[1]).ceil().min(12.0) as i64, (r / h[2]).ceil().min(12.0) as i64];
        let (i, j, k) = ((q % nx) as i64, ((q / nx) % ny) as i64, (q / (nx * ny)) as i64);
        let mut m = dt[q];
        for c in (k - rv[2]).max(0)..=(k + rv[2]).min(nz as i64 - 1) {
            for b in (j - rv[1]).max(0)..=(j + rv[1]).min(ny as i64 - 1) {
                for a in (i - rv[0]).max(0)..=(i + rv[0]).min(nx as i64 - 1) {
                    let d2 = ((a - i) as f64 * h[0]).powi(2) + ((b - j) as f64 * h[1]).powi(2) + ((c - k) as f64 * h[2]).powi(2);
                    if d2 <= r * r {
                        let v = dt[idx(a as usize, b as usize, c as usize)];
                        if v > m {
                            m = v;
                        }
                    }
                }
            }
        }
        m
    }).collect();
    let pen_v: Vec<f32> = (0..n).map(|q| {
        let f = if rloc[q] > 0.0 { (1.0 - dt[q] as f64 / rloc[q] as f64).max(0.0) } else { 1.0 };
        (1.0 + 300.0 * f * f) as f32
    }).collect();
    let pen = |q: usize| pen_v[q] as f64;
    let dijkstra = |src: usize, parent: &mut Vec<u32>| -> Vec<f32> {
        let mut dist = vec![f32::INFINITY; n];
        let mut heap: BinaryHeap<Reverse<(F, u32)>> = BinaryHeap::new();
        dist[src] = 0.0;
        parent[src] = src as u32;
        heap.push(Reverse((F(0.0), src as u32)));
        while let Some(Reverse((F(dv), u))) = heap.pop() {
            let u = u as usize;
            if dv as f32 > dist[u] {
                continue;
            }
            let (i, j, k) = (u % nx, (u / nx) % ny, u / (nx * ny));
            for (d, len) in &step {
                let (a, b, c) = (i as i64 + d[0], j as i64 + d[1], k as i64 + d[2]);
                if a < 0 || b < 0 || c < 0 || a >= nx as i64 || b >= ny as i64 || c >= nz as i64 {
                    continue;
                }
                let w = idx(a as usize, b as usize, c as usize);
                if mask[w] == 0 {
                    continue;
                }
                let nd = (dv + len * pen(w)) as f32;
                if nd < dist[w] {
                    dist[w] = nd;
                    parent[w] = u as u32;
                    heap.push(Reverse((F(nd as f64), w as u32)));
                }
            }
        }
        dist
    };
    // extraction of the branches for one root: shortest-path tree, then repeatedly the farthest remaining tip back to the tree
    let extract = |root_q: usize| -> Vec<Branch> {
        let mut parent = vec![u32::MAX; n];
        let dist = dijkstra(root_q, &mut parent);
        let mut order: Vec<usize> = (0..n).filter(|&q| dist[q].is_finite()).collect();
        order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]));
        let mut valid: Vec<bool> = (0..n).map(|q| dist[q].is_finite()).collect();
        let mut on_tree = vec![false; n];
        on_tree[root_q] = true;
        let mut branches = Vec::new();
        for &tip in &order {
            if !valid[tip] || on_tree[tip] {
                continue;
            }
            let mut path = vec![tip];
            let mut q = tip;
            while !on_tree[q] {
                q = parent[q] as usize;
                path.push(q);
                if parent[q] == q as u32 {
                    break;
                }
            }
            // invalidate balls around the path voxels (radius = scale · distance to wall + const)
            for &p in &path {
                let r = scale * dt[p] as f64 + const_mm;
                let rv = [(r / h[0]).ceil() as i64, (r / h[1]).ceil() as i64, (r / h[2]).ceil() as i64];
                let (i, j, k) = ((p % nx) as i64, ((p / nx) % ny) as i64, (p / (nx * ny)) as i64);
                for c in (k - rv[2]).max(0)..=(k + rv[2]).min(nz as i64 - 1) {
                    for b in (j - rv[1]).max(0)..=(j + rv[1]).min(ny as i64 - 1) {
                        for a in (i - rv[0]).max(0)..=(i + rv[0]).min(nx as i64 - 1) {
                            let d2 = ((a - i) as f64 * h[0]).powi(2) + ((b - j) as f64 * h[1]).powi(2) + ((c - k) as f64 * h[2]).powi(2);
                            if d2 <= r * r {
                                valid[idx(a as usize, b as usize, c as usize)] = false;
                            }
                        }
                    }
                }
            }
            let len: f64 = path.windows(2).map(|w| {
                let (a, b) = (w[0], w[1]);
                (((a % nx) as f64 - (b % nx) as f64) * h[0]).hypot((((a / nx) % ny) as f64 - ((b / nx) % ny) as f64) * h[1]).hypot(((a / (nx * ny)) as f64 - (b / (nx * ny)) as f64) * h[2])
            }).sum();
            if len < min_len_mm {
                continue;
            }
            for &p in &path {
                on_tree[p] = true;
            }
            branches.push(Branch { pts: path });
        }
        branches
    };
    let nearest_mask = |r: [usize; 3]| -> usize {
        let q = idx(r[0].min(nx - 1), r[1].min(ny - 1), r[2].min(nz - 1));
        if mask[q] != 0 { q } else {
            (0..n).filter(|&q| mask[q] != 0).min_by_key(|&q| {
                let (i, j, k) = (q % nx, (q / nx) % ny, q / (nx * ny));
                (i as i64 - r[0] as i64).pow(2) + (j as i64 - r[1] as i64).pow(2) + (k as i64 - r[2] as i64).pow(2)
            }).unwrap_or(0)
        }
    };
    // root: the given voxel; else the tip of the thickest branch. A first pass roots the tree at the thickest point; the branch whose tip-side half is thickest
    // is the trunk (aorta, carotid stem...), and its far end becomes the root, so daughters leave the trunk at the bifurcation instead of being linked
    // to each other by a path that cuts the corner of the crotch.
    let root_q = match root {
        Some(r) => nearest_mask(r),
        None => {
            let start = (0..n).filter(|&q| mask[q] != 0).max_by(|&a, &b| dt[a].total_cmp(&dt[b])).unwrap_or(0);
            let first = extract(start);
            let score = |b: &Branch| -> f64 {
                let m = (b.pts.len() * 7 / 10).max(1);
                b.pts[..m].iter().map(|&q| dt[q] as f64).sum::<f64>() / m as f64
            };
            first.iter().max_by(|a, b| score(a).total_cmp(&score(b))).map(|b| b.pts[0]).unwrap_or(start)
        }
    };
    let mut branches = extract(root_q);
    // junction centring: a daughter ends where its shortest path left the parent, which may sit a little before the bifurcation; move the end to the point
    // of the parent path with the largest inscribed ball in the neighbourhood (the centre of the bifurcation)
    let mut owner: HashMap<usize, (usize, usize)> = HashMap::new();
    for bi in 0..branches.len() {
        if bi > 0 {
            let j = *branches[bi].pts.last().unwrap();
            if let Some(&(pb, _)) = owner.get(&j) {
                if pb != bi {
                    let rr = 1.5 * dt[j] as f64 + h[0].max(h[1]).max(h[2]);
                    let (ji, jj, jk) = ((j % nx) as f64, ((j / nx) % ny) as f64, (j / (nx * ny)) as f64);
                    let best = branches[pb].pts.iter().copied().filter(|&q| {
                        (((q % nx) as f64 - ji) * h[0]).hypot((((q / nx) % ny) as f64 - jj) * h[1]).hypot(((q / (nx * ny)) as f64 - jk) * h[2]) <= rr
                    }).max_by(|&a, &b| dt[a].total_cmp(&dt[b]));
                    if let Some(c) = best {
                        if dt[c] > dt[j] + 0.05 {
                            *branches[bi].pts.last_mut().unwrap() = c;
                        }
                    }
                }
            }
        }
        for (k, &q) in branches[bi].pts.iter().enumerate() {
            owner.entry(q).or_insert((bi, k));
        }
    }
    (branches, dt)
}


// ── voxelisation and booleans (resolution-limited, via masks) ────────────────────────────────────────────────────────

/// parity fill of a closed mesh on a grid whose voxel (i,j,k) has its centre at origin + h·(i,j,k) mm; scanlines along x. Voxels inside → 1.
pub fn voxelize(m: &Mesh, origin: V3, h: f64, dims: [usize; 3]) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let mut rows: Vec<Vec<f32>> = vec![Vec::new(); ny * nz];
    let (e1, e2) = (1.7e-4, 2.9e-4);                       // sample the scanlines a hair off the voxel centres: no exact edge hits
    for f in &m.faces {
        let q: [V3; 3] = f.map(|i| {
            let v = m.verts[i as usize];
            [(v[0] as f64 - origin[0]) / h, (v[1] as f64 - origin[1]) / h, (v[2] as f64 - origin[2]) / h]
        });
        let d = (q[1][1] - q[0][1]) * (q[2][2] - q[0][2]) - (q[2][1] - q[0][1]) * (q[1][2] - q[0][2]);
        if d.abs() < 1e-18 {
            continue;
        }
        let (ymin, ymax) = (q.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min), q.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max));
        let (zmin, zmax) = (q.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min), q.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max));
        let (j0, j1) = (((ymin - e1).ceil().max(0.0)) as usize, ((ymax - e1).floor().min(ny as f64 - 1.0)).max(-1.0) as i64);
        let (k0, k1) = (((zmin - e2).ceil().max(0.0)) as usize, ((zmax - e2).floor().min(nz as f64 - 1.0)).max(-1.0) as i64);
        for k in k0 as i64..=k1 {
            for j in j0 as i64..=j1 {
                let (y, z) = (j as f64 + e1, k as f64 + e2);
                let u = ((y - q[0][1]) * (q[2][2] - q[0][2]) - (q[2][1] - q[0][1]) * (z - q[0][2])) / d;
                let v = ((q[1][1] - q[0][1]) * (z - q[0][2]) - (y - q[0][1]) * (q[1][2] - q[0][2])) / d;
                if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                    rows[k as usize * ny + j as usize].push((q[0][0] + u * (q[1][0] - q[0][0]) + v * (q[2][0] - q[0][0])) as f32);
                }
            }
        }
    }
    let mut out = vec![0u8; nx * ny * nz];
    for (r, xs) in rows.iter_mut().enumerate() {
        xs.sort_by(|a, b| a.total_cmp(b));
        for pair in xs.chunks_exact(2) {
            let (a, b) = (pair[0].ceil().max(0.0) as usize, (pair[1].ceil().max(0.0) as usize).min(nx));
            for i in a..b {
                out[r * nx + i] = 1;
            }
        }
    }
    out
}

fn bbox_of(m: &Mesh) -> (V3, V3) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for v in &m.verts {
        for a in 0..3 {
            lo[a] = lo[a].min(v[a] as f64);
            hi[a] = hi[a].max(v[a] as f64);
        }
    }
    (lo, hi)
}

fn grid_for(lo: V3, hi: V3, voxel_mm: Option<f64>, margin: usize) -> Result<(V3, f64, [usize; 3]), Error> {
    let ext = [0, 1, 2].map(|a| (hi[a] - lo[a]).max(1e-6));
    let h = voxel_mm.filter(|&h| h > 0.0).unwrap_or_else(|| ext.iter().cloned().fold(0.0, f64::max) / 128.0);
    let origin = [0, 1, 2].map(|a| lo[a] - margin as f64 * h);
    let dims = [0, 1, 2].map(|a| (ext[a] / h).ceil() as usize + 2 * margin + 1);
    if dims.iter().product::<usize>() > 300_000_000 {
        return inv(format!("voxel grid {dims:?} is too large: raise options.voxel_mm"));
    }
    Ok((origin, h, dims))
}

fn grid_affine(origin: V3, h: f64) -> [f64; 16] {
    [h, 0.0, 0.0, origin[0], 0.0, h, 0.0, origin[1], 0.0, 0.0, h, origin[2], 0.0, 0.0, 0.0, 1.0]
}

// ── temporally coherent surface tracking ─────────────────────────────────────────────────────────────────────────────

/// trilinear sample of s ([Z,Y,X], dims [nx,ny,nz]) at voxel coordinates p, clamped to the grid
fn tri(s: &[f32], d: [usize; 3], p: V3) -> f64 {
    let [nx, ny, nz] = d;
    let c = [p[0].clamp(0.0, nx as f64 - 1.0), p[1].clamp(0.0, ny as f64 - 1.0), p[2].clamp(0.0, nz as f64 - 1.0)];
    let (i, j, k) = ((c[0] as usize).min(nx.saturating_sub(2)), (c[1] as usize).min(ny.saturating_sub(2)), (c[2] as usize).min(nz.saturating_sub(2)));
    let (fx, fy, fz) = (c[0] - i as f64, c[1] - j as f64, c[2] - k as f64);
    let at = |a: usize, b: usize, c2: usize| s[(c2.min(nz - 1) * ny + b.min(ny - 1)) * nx + a.min(nx - 1)] as f64;
    let l = |a: f64, b: f64, f: f64| a * (1.0 - f) + b * f;
    let z0 = l(l(at(i, j, k), at(i + 1, j, k), fx), l(at(i, j + 1, k), at(i + 1, j + 1, k), fx), fy);
    let z1 = l(l(at(i, j, k + 1), at(i + 1, j, k + 1), fx), l(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), fx), fy);
    l(z0, z1, fz)
}

/// One mesh topology, one vertex per surface point, for every frame of a 4-D mask: frame 0 is the iso-surface; later frames move the previous vertices
/// onto the 0.5 level of the blurred frame (Newton steps along the indicator gradient, at most one voxel each) with a little Laplacian relaxation, so
/// vertices keep their identity and slide as little as the data allows.
fn track_sequence(t: &Tensor, o: &Json) -> Result<(Vec<Mesh>, Vec<bool>, Vec<Option<Vec<u32>>>), Error> {
    let &[nt, nz, ny, nx] = t.shape() else { return inv(format!("meshseq_track: tensor must be [T, Z, Y, X], got {:?}", t.shape())) };
    let dims = [nx, ny, nz];
    let nvox = nx * ny * nz;
    let sels: Vec<Vec<u8>> = (0..nt).map(|f| select(t, f * nvox, nvox, o)).collect();
    if sels[0].iter().all(|&x| x == 0) {
        return inv("meshseq_track: frame 0 selects no voxel");
    }
    let (first, _) = largest_component(&sels[0], dims);
    let (mut lo, mut hi) = ([nx, ny, nz], [0usize; 3]);
    for (f, sel) in sels.iter().enumerate() {
        let s = if f == 0 { &first } else { sel };
        for (q, &c) in s.iter().enumerate() {
            if c != 0 {
                let p = [q % nx, (q / nx) % ny, q / (nx * ny)];
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
        }
    }
    let pad = 4i64;
    let org = [0, 1, 2].map(|a| lo[a] as i64 - pad);
    let cd = [0, 1, 2].map(|a| hi[a].saturating_sub(lo[a]) + 1 + 2 * pad as usize);
    let blur_n = o["blur"].as_u64().unwrap_or(1) as usize;
    let crops: Vec<Vec<f32>> = sels.iter().enumerate().map(|(f, sel)| {
        let s = if f == 0 { &first } else { sel };
        let mut crop = vec![0f32; cd[0] * cd[1] * cd[2]];
        for (q, &c) in s.iter().enumerate() {
            if c != 0 {
                let p = [(q % nx) as i64 - org[0], ((q / nx) % ny) as i64 - org[1], (q / (nx * ny)) as i64 - org[2]];
                crop[(p[2] as usize * cd[1] + p[1] as usize) * cd[0] + p[0] as usize] = 1.0;
            }
        }
        crop
    }).collect();
    let centroid = |c: &Vec<f32>| -> V3 {
        let (mut s, mut w) = ([0f64; 3], 0f64);
        for (q, &v) in c.iter().enumerate() {
            if v > 0.0 {
                s[0] += (q % cd[0]) as f64;
                s[1] += ((q / cd[0]) % cd[1]) as f64;
                s[2] += (q / (cd[0] * cd[1])) as f64;
                w += 1.0;
            }
        }
        if w > 0.0 { [s[0] / w, s[1] / w, s[2] / w] } else { [0.0; 3] }
    };
    let max_tris = o["max_triangles"].as_u64().unwrap_or(6_000_000) as usize;
    let steps = o["steps"].as_u64().unwrap_or(12) as usize;
    let relax = o["relax"].as_f64().unwrap_or(0.15);
    let (space, affine, spacing_mm, map) = space_map(o)?;
    let flip = affine.map(|a| a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8]) + a[2] * (a[4] * a[9] - a[5] * a[8]) < 0.0).unwrap_or(false);
    let iters = o["smooth"].as_u64().unwrap_or(6) as usize;
    // a fresh iso-surface of frame f: vertices (crop voxel coordinates), triangles, neighbour lists
    let fresh = |f: usize| -> Result<(Vec<V3>, Vec<[u32; 3]>, Vec<Vec<u32>>), Error> {
        let (pos, faces) = marching_tets(&blur(&crops[f], cd, blur_n), cd, 0.5, max_tris)?;
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); pos.len()];
        for t in &faces {
            for k in 0..3 {
                adj[t[k] as usize].push(t[(k + 1) % 3]);
                adj[t[(k + 1) % 3] as usize].push(t[k]);
            }
        }
        adj.iter_mut().for_each(|a| { a.sort_unstable(); a.dedup(); });
        Ok((pos.iter().map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect(), faces, adj))
    };
    let ncomp = |c: &Vec<f32>| -> usize {
        let m: Vec<u8> = c.iter().map(|&v| (v > 0.0) as u8).collect();
        let mut left = m.clone();
        let mut n = 0;
        while left.iter().any(|&x| x != 0) {
            let (comp, _) = largest_component(&left, cd);
            for (l, &c) in left.iter_mut().zip(&comp) {
                if c != 0 {
                    *l = 0;
                }
            }
            n += 1;
            if n > 64 {
                break;
            }
        }
        n
    };
    let counts: Vec<usize> = crops.iter().map(ncomp).collect();
    let (mut cur, mut faces, mut adj) = fresh(0)?;
    let mut out = Vec::with_capacity(nt);
    let mut links: Vec<bool> = Vec::new();
    let mut parents: Vec<Option<Vec<u32>>> = vec![None; nt];
    for f in 0..nt {
        if f > 0 {
            let mut linked = counts[f] == counts[f - 1];      // a split or a merge changes the topology: tracking cannot follow it
            if linked {
                // large motion: first carry the vertices along with the centroid shift, then snap in a coarse-to-fine sequence of blurred indicators
                let (c0, c1) = (centroid(&crops[f - 1]), centroid(&crops[f]));
                let shift = [c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]];
                for v in cur.iter_mut() {
                    for a in 0..3 {
                        v[a] += shift[a];
                    }
                }
                let mut fine = Vec::new();
                for (level, extra) in [(blur_n + 4, 4usize), (blur_n + 2, 2), (blur_n, 0)] {
                    let ind = blur(&crops[f], cd, level);
                    let clamp = 1.0 + extra as f64 / 2.0;
                    for _ in 0..steps {
                        for v in cur.iter_mut() {
                            let val = tri(&ind, cd, *v);
                            let g = [0, 1, 2].map(|a| {
                                let (mut p, mut m) = (*v, *v);
                                p[a] += 0.5;
                                m[a] -= 0.5;
                                tri(&ind, cd, p) - tri(&ind, cd, m)
                            });
                            let g2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                            if g2 < 1e-10 {
                                continue;
                            }
                            let k = (0.5 - val) / g2;
                            let mut mv = [g[0] * k, g[1] * k, g[2] * k];
                            let l = (mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]).sqrt();
                            if l > clamp {
                                mv = mv.map(|x| x * clamp / l);
                            }
                            for a in 0..3 {
                                v[a] += mv[a];
                            }
                        }
                        if relax > 0.0 {
                            let prev = cur.clone();
                            for (i, v) in cur.iter_mut().enumerate() {
                                if adj[i].is_empty() {
                                    continue;
                                }
                                let c = adj[i].len() as f64;
                                for a in 0..3 {
                                    let m = adj[i].iter().map(|&j| prev[j as usize][a]).sum::<f64>() / c;
                                    v[a] += relax * (m - prev[i][a]);
                                }
                            }
                        }
                    }
                    fine = ind;
                }
                // fit check: when many vertices still miss the 0.5 level (the object changed shape beyond what a fixed topology can follow), start a new run
                let bad = cur.iter().filter(|&&v| (tri(&fine, cd, v) - 0.5).abs() > 0.2).count();
                if bad as f64 > 0.05 * cur.len() as f64 {
                    linked = false;
                }
            }
            if !linked {
                // a new surface run: every new vertex remembers the nearest vertex of the previous frame, so a viewer can morph across the split / merge
                let prev = std::mem::take(&mut cur);
                let (new_cur, new_faces, new_adj) = fresh(f)?;
                let cell = 2.0;
                let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
                for (i, p) in prev.iter().enumerate() {
                    grid.entry(((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64)).or_default().push(i as u32);
                }
                let nearest = |q: V3| -> u32 {
                    let c = [0, 1, 2].map(|a| (q[a] / cell).floor() as i64);
                    for r in 1..=64i64 {
                        let mut best: Option<(f64, u32)> = None;
                        for i in c[0] - r..=c[0] + r {
                            for j in c[1] - r..=c[1] + r {
                                for k in c[2] - r..=c[2] + r {
                                    if (i - c[0]).abs().max((j - c[1]).abs()).max((k - c[2]).abs()) != r && r > 1 {
                                        continue;
                                    }
                                    for &pi in grid.get(&(i, j, k)).into_iter().flatten() {
                                        let p = prev[pi as usize];
                                        let d = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                                        if best.map_or(true, |(bd, _)| d < bd) {
                                            best = Some((d, pi));
                                        }
                                    }
                                }
                            }
                        }
                        if let Some((d, pi)) = best {
                            if d.sqrt() <= (r as f64 - 1.0).max(0.0) * cell + cell || r >= 3 {
                                return pi;
                            }
                        }
                    }
                    0
                };
                parents[f] = Some(new_cur.iter().map(|&q| nearest(q)).collect());
                (cur, faces, adj) = (new_cur, new_faces, new_adj);
            }
            links.push(linked);
        }
        let verts: Vec<[f32; 3]> = cur.iter().map(|p| {
            let w = map([p[0] + org[0] as f64, p[1] + org[1] as f64, p[2] + org[2] as f64]);
            [w[0] as f32, w[1] as f32, w[2] as f32]
        }).collect();
        let mut m = Mesh { verts, faces: faces.clone(), scalars: vec![], space: space.into(), affine, spacing_mm };
        if flip {
            m.faces.iter_mut().for_each(|f| f.swap(1, 2));
        }
        if iters > 0 {
            m.taubin(iters, 0.5, -0.53);
        }
        out.push(m);
    }
    Ok((out, links, parents))
}


// ── snapping the voxel boolean back onto the input surfaces ──────────────────────────────────────────────────────────

fn closest_on_tri(p: V3, a: V3, b: V3, c: V3) -> V3 {
    // Ericson, Real-Time Collision Detection
    let (ab, ac, ap) = (sub3(b, a), sub3(c, a), sub3(p, a));
    let (d1, d2) = (dot3(ab, ap), dot3(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 { return a; }
    let bp = sub3(p, b);
    let (d3, d4) = (dot3(ab, bp), dot3(ac, bp));
    if d3 >= 0.0 && d4 <= d3 { return b; }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { let v = d1 / (d1 - d3); return [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]]; }
    let cp = sub3(p, c);
    let (d5, d6) = (dot3(ab, cp), dot3(ac, cp));
    if d6 >= 0.0 && d5 <= d6 { return c; }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { let w = d2 / (d2 - d6); return [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]]; }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = sub3(c, b);
        return [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]];
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]
}

pub(crate) struct TriGrid<'a> {
    pub(crate) m: &'a Mesh,
    pub(crate) cell: f64,
    pub(crate) cells: HashMap<(i64, i64, i64), Vec<u32>>,
}

impl<'a> TriGrid<'a> {
    pub(crate) fn new(m: &'a Mesh, cell: f64) -> Self {
        let mut cells: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        for (fi, f) in m.faces.iter().enumerate() {
            let p = f.map(|i| m.verts[i as usize]);
            let lo = [0, 1, 2].map(|a| (p[0][a].min(p[1][a]).min(p[2][a]) as f64 / cell).floor() as i64);
            let hi = [0, 1, 2].map(|a| (p[0][a].max(p[1][a]).max(p[2][a]) as f64 / cell).floor() as i64);
            for i in lo[0]..=hi[0] {
                for j in lo[1]..=hi[1] {
                    for k in lo[2]..=hi[2] {
                        cells.entry((i, j, k)).or_default().push(fi as u32);
                    }
                }
            }
        }
        TriGrid { m, cell, cells }
    }

    /// closest surface point within `max_d` (None when nothing is that close)
    pub(crate) fn closest(&self, p: V3, max_d: f64) -> Option<(f64, V3)> {
        let r = (max_d / self.cell).ceil() as i64;
        let c = [0, 1, 2].map(|a| (p[a] / self.cell).floor() as i64);
        let mut best: Option<(f64, V3)> = None;
        let mut seen: HashSet<u32> = HashSet::new();
        for i in c[0] - r..=c[0] + r {
            for j in c[1] - r..=c[1] + r {
                for k in c[2] - r..=c[2] + r {
                    for &fi in self.cells.get(&(i, j, k)).into_iter().flatten() {
                        if !seen.insert(fi) {
                            continue;
                        }
                        let f = self.m.faces[fi as usize];
                        let [a, b, c3] = f.map(|v| { let x = self.m.verts[v as usize]; [x[0] as f64, x[1] as f64, x[2] as f64] });
                        let q = closest_on_tri(p, a, b, c3);
                        let d = dot3(sub3(p, q), sub3(p, q)).sqrt();
                        if d <= max_d && best.map_or(true, |(bd, _)| d < bd) {
                            best = Some((d, q));
                        }
                    }
                }
            }
        }
        best
    }
}

/// move every vertex of `m` onto the nearest point of the given surfaces when one lies within `max_d` mm
pub fn snap_to(m: &mut Mesh, sources: &[&Mesh], max_d: f64, skip: Option<&[bool]>) -> usize {
    let grids: Vec<TriGrid> = sources.iter().map(|s| TriGrid::new(s, (2.0 * max_d).max(1e-3))).collect();
    let mut moved = 0;
    for (vi, v) in m.verts.iter_mut().enumerate() {
        if skip.map_or(false, |s| s[vi]) {
            continue;
        }
        let p = [v[0] as f64, v[1] as f64, v[2] as f64];
        let best = grids.iter().filter_map(|g| g.closest(p, max_d)).min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((_, q)) = best {
            *v = [q[0] as f32, q[1] as f32, q[2] as f32];
            moved += 1;
        }
    }
    moved
}


// ── intersection curve of two meshes: crease reconstruction for the voxel boolean ─────────────────────────────────────────

pub(crate) fn seg_tri(p0: V3, p1: V3, a: V3, b: V3, c: V3) -> Option<V3> {
    // Möller–Trumbore on the segment p0→p1 (t in [0, 1])
    let d = sub3(p1, p0);
    let (e1, e2) = (sub3(b, a), sub3(c, a));
    let h = cross3(d, e2);
    let det = dot3(e1, h);
    if det.abs() < 1e-14 {
        return None;
    }
    let f = 1.0 / det;
    let s = sub3(p0, a);
    let u = f * dot3(s, h);
    if !(-1e-9..=1.0 + 1e-9).contains(&u) {
        return None;
    }
    let q = cross3(s, e1);
    let v = f * dot3(d, q);
    if v < -1e-9 || u + v > 1.0 + 1e-9 {
        return None;
    }
    let t = f * dot3(e2, q);
    if !(-1e-9..=1.0 + 1e-9).contains(&t) {
        return None;
    }
    Some([p0[0] + t * d[0], p0[1] + t * d[1], p0[2] + t * d[2]])
}

/// the segments where the surfaces of `a` and `b` cross (triangle pairs found through a grid over `b`)
pub fn intersection_segments(a: &Mesh, b: &Mesh, cell: f64) -> Vec<[V3; 2]> {
    let gb = TriGrid::new(b, cell);
    let p3 = |m: &Mesh, v: u32| { let x = m.verts[v as usize]; [x[0] as f64, x[1] as f64, x[2] as f64] };
    let mut out = Vec::new();
    for fa in &a.faces {
        let ta = fa.map(|v| p3(a, v));
        let lo = [0, 1, 2].map(|k| (ta[0][k].min(ta[1][k]).min(ta[2][k]) / cell).floor() as i64);
        let hi = [0, 1, 2].map(|k| (ta[0][k].max(ta[1][k]).max(ta[2][k]) / cell).floor() as i64);
        let mut seen: HashSet<u32> = HashSet::new();
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                for k in lo[2]..=hi[2] {
                    for &fi in gb.cells.get(&(i, j, k)).into_iter().flatten() {
                        if !seen.insert(fi) {
                            continue;
                        }
                        let tb = b.faces[fi as usize].map(|v| p3(b, v));
                        let mut pts: Vec<V3> = Vec::new();
                        for e in 0..3 {
                            if let Some(p) = seg_tri(tb[e], tb[(e + 1) % 3], ta[0], ta[1], ta[2]) { pts.push(p); }
                            if let Some(p) = seg_tri(ta[e], ta[(e + 1) % 3], tb[0], tb[1], tb[2]) { pts.push(p); }
                        }
                        if pts.len() < 2 {
                            continue;
                        }
                        let (mut best, mut pair) = (1e-12, None);
                        for x in 0..pts.len() {
                            for y in x + 1..pts.len() {
                                let d = dot3(sub3(pts[x], pts[y]), sub3(pts[x], pts[y]));
                                if d > best {
                                    best = d;
                                    pair = Some([pts[x], pts[y]]);
                                }
                            }
                        }
                        out.extend(pair);
                    }
                }
            }
        }
    }
    out
}

fn closest_on_segment(p: V3, a: V3, b: V3) -> V3 {
    let d = sub3(b, a);
    let l2 = dot3(d, d);
    let t = if l2 > 0.0 { (dot3(sub3(p, a), d) / l2).clamp(0.0, 1.0) } else { 0.0 };
    [a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]]
}

/// vertices that are about equally close to both surfaces (within `max_d`) lie on the crease: move them onto the intersection curve; returns which were moved
pub fn snap_crease(m: &mut Mesh, a: &Mesh, b: &Mesh, segs: &[[V3; 2]], h: f64) -> Vec<bool> {
    let mut moved = vec![false; m.verts.len()];
    if segs.is_empty() {
        return moved;
    }
    let cell = 2.0 * h;
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (si, s) in segs.iter().enumerate() {
        let lo = [0, 1, 2].map(|k| (s[0][k].min(s[1][k]) / cell).floor() as i64);
        let hi = [0, 1, 2].map(|k| (s[0][k].max(s[1][k]) / cell).floor() as i64);
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                for k in lo[2]..=hi[2] {
                    grid.entry((i, j, k)).or_default().push(si as u32);
                }
            }
        }
    }
    let (ga, gb) = (TriGrid::new(a, cell), TriGrid::new(b, cell));
    for (vi, v) in m.verts.iter_mut().enumerate() {
        let p = [v[0] as f64, v[1] as f64, v[2] as f64];
        let (Some((da, _)), Some((db, _))) = (ga.closest(p, 1.6 * h), gb.closest(p, 1.6 * h)) else { continue };
        if (da - db).abs() > 0.6 * h {
            continue;
        }
        let c = [0, 1, 2].map(|k| (p[k] / cell).floor() as i64);
        let mut best: Option<(f64, V3)> = None;
        for i in c[0] - 1..=c[0] + 1 {
            for j in c[1] - 1..=c[1] + 1 {
                for k in c[2] - 1..=c[2] + 1 {
                    for &si in grid.get(&(i, j, k)).into_iter().flatten() {
                        let q = closest_on_segment(p, segs[si as usize][0], segs[si as usize][1]);
                        let d = dot3(sub3(p, q), sub3(p, q)).sqrt();
                        if d <= 1.6 * h && best.map_or(true, |(bd, _)| d < bd) {
                            best = Some((d, q));
                        }
                    }
                }
            }
        }
        if let Some((_, q)) = best {
            *v = [q[0] as f32, q[1] as f32, q[2] as f32];
            moved[vi] = true;
        }
    }
    moved
}

// ── fairing of filled patches: refine the ear-clipped triangulation, then relax the new interior ────────────────────────

impl Mesh {
    /// refine the triangles `patch` (indices into `self.faces` from `from`), whose boundary is fixed, to about the mean boundary edge length and relax the
    /// interior vertices towards a smooth membrane
    pub fn fair_patch(&mut self, from: usize, relax_iters: usize, cotan: bool) {
        let patch: Vec<[u32; 3]> = self.faces[from..].to_vec();
        if patch.is_empty() {
            return;
        }
        self.faces.truncate(from);
        let mut edge_use: HashMap<(u32, u32), u32> = HashMap::new();
        for f in &patch {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *edge_use.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let boundary: HashSet<(u32, u32)> = edge_use.iter().filter(|(_, &c)| c == 1).map(|(&e, _)| e).collect();
        let fixed: HashSet<u32> = boundary.iter().flat_map(|&(a, b)| [a, b]).collect();
        let tgt = if boundary.is_empty() { 0.0 } else {
            boundary.iter().map(|&(a, b)| { let (p, q) = (self.verts[a as usize], self.verts[b as usize]); (((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)) as f64).sqrt() }).sum::<f64>() / boundary.len() as f64
        };
        let mut tris = patch;
        let len = |v: &Vec<[f32; 3]>, a: u32, b: u32| -> f64 { let (p, q) = (v[a as usize], v[b as usize]); (((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)) as f64).sqrt() };
        for _ in 0..6 {
            if tris.len() > 12 * boundary.len() + 500 {
                break;     // big openings (a whole cut face): keep the ear-clipped triangulation of the rest instead of exploding the triangle count
            }
            // split every interior edge longer than 1.5 × target at its midpoint, and subdivide the triangles around it
            let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
            for f in &tris {
                for k in 0..3 {
                    let (a, b) = (f[k], f[(k + 1) % 3]);
                    let e = (a.min(b), a.max(b));
                    if !boundary.contains(&e) && !mid.contains_key(&e) && len(&self.verts, a, b) > 1.5 * tgt {
                        let (p, q) = (self.verts[a as usize], self.verts[b as usize]);
                        self.verts.push([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0, (p[2] + q[2]) / 2.0]);
                        for (_, sc) in self.scalars.iter_mut() {
                            let (x, y) = (sc[a as usize], sc[b as usize]);
                            sc.push(if x.is_nan() || y.is_nan() { f32::NAN } else { (x + y) / 2.0 });
                        }
                        mid.insert(e, (self.verts.len() - 1) as u32);
                    }
                }
            }
            if mid.is_empty() {
                break;
            }
            let mut next = Vec::with_capacity(tris.len() * 2);
            for f in &tris {
                let m: [Option<u32>; 3] = [0, 1, 2].map(|k| { let (a, b) = (f[k], f[(k + 1) % 3]); mid.get(&(a.min(b), a.max(b))).copied() });
                let n = m.iter().flatten().count();
                match n {
                    0 => next.push(*f),
                    3 => {
                        let (m0, m1, m2) = (m[0].unwrap(), m[1].unwrap(), m[2].unwrap());
                        next.extend([[f[0], m0, m2], [m0, f[1], m1], [m2, m1, f[2]], [m0, m1, m2]]);
                    }
                    _ => {
                        // 1 or 2 split edges: rotate so edge 0 (f0→f1) is split
                        let r = (0..3).find(|&k| m[k].is_some() && (n == 1 || m[(k + 2) % 3].is_none() || m[(k + 1) % 3].is_some())).unwrap();
                        let (a, b, c) = (f[r], f[(r + 1) % 3], f[(r + 2) % 3]);
                        let (mab, mbc, mca) = (m[r], m[(r + 1) % 3], m[(r + 2) % 3]);
                        let mab = mab.unwrap();
                        match (mbc, mca) {
                            (None, None) => next.extend([[a, mab, c], [mab, b, c]]),
                            (Some(mbc), None) => next.extend([[a, mab, c], [mab, b, mbc], [mab, mbc, c]]),
                            (None, Some(mca)) => next.extend([[a, mab, mca], [mab, b, c], [mab, c, mca]]),
                            (Some(_), Some(_)) => unreachable!(),
                        }
                    }
                }
            }
            tris = next;
        }
        // relax the interior vertices (uniform Laplacian, boundary fixed)
        let nv = self.verts.len();
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); nv];
        for f in &tris {
            for k in 0..3 {
                adj[f[k] as usize].push(f[(k + 1) % 3]);
                adj[f[(k + 1) % 3] as usize].push(f[k]);
            }
        }
        adj.iter_mut().for_each(|a| { a.sort_unstable(); a.dedup(); });
        let interior: Vec<u32> = tris.iter().flatten().copied().collect::<HashSet<_>>().into_iter().filter(|v| !fixed.contains(v)).collect();
        // cotangent weights: the relaxed patch approaches a minimal surface whatever the triangulation (uniform weights bend it towards the refinement pattern)
        let mut cw: HashMap<(u32, u32), f64> = HashMap::new();
        if cotan {
            for f in &tris {
                let p = f.map(|v| { let x = self.verts[v as usize]; [x[0] as f64, x[1] as f64, x[2] as f64] });
                for k in 0..3 {
                    let (a, b, c) = (p[k], p[(k + 1) % 3], p[(k + 2) % 3]);
                    let (u, w) = (sub3(b, a), sub3(c, a));
                    let cr = cross3(u, w);
                    let area2 = dot3(cr, cr).sqrt();
                    if area2 > 1e-18 {
                        let (i, j) = (f[(k + 1) % 3], f[(k + 2) % 3]);
                        *cw.entry((i.min(j), i.max(j))).or_default() += 0.5 * dot3(u, w) / area2;
                    }
                }
            }
        }
        for _ in 0..relax_iters {
            let prev = self.verts.clone();
            for &v in &interior {
                let a = &adj[v as usize];
                if a.is_empty() {
                    continue;
                }
                if cotan {
                    let (mut sw, mut acc) = (0.0f64, [0f64; 3]);
                    for &u in a {
                        let w = cw.get(&(u.min(v), u.max(v))).copied().unwrap_or(0.0).max(1e-3);
                        sw += w;
                        for k in 0..3 {
                            acc[k] += w * prev[u as usize][k] as f64;
                        }
                    }
                    self.verts[v as usize] = [0, 1, 2].map(|k| (acc[k] / sw) as f32);
                } else {
                    let c = a.len() as f32;
                    self.verts[v as usize] = [0, 1, 2].map(|k| a.iter().map(|&u| prev[u as usize][k]).sum::<f32>() / c);
                }
            }
        }
        self.faces.extend(tris);
    }
}

// ── vessel measurements along a centerline ────────────────────────────────────────────────────────────────────────────

fn unit3(a: V3) -> V3 {
    let l = dot3(a, a).sqrt();
    if l > 1e-12 { [a[0] / l, a[1] / l, a[2] / l] } else { [0.0, 0.0, 1.0] }
}

fn angle_deg(a: V3, b: V3) -> f64 {
    dot3(unit3(a), unit3(b)).clamp(-1.0, 1.0).acos().to_degrees()
}

pub fn register(r: &mut Registry) {
    r.register(
        "mesh_decimate",
        "quadric edge-collapse decimation: inputs [mesh]; options {target_faces | ratio (0..1 of the current count), preserve_scalars (true): vertices that are local extremes of a scalar and within the top / bottom 1 % of its range are never merged away, scalar_weight (0.5): cost of merging vertices whose scalars differ, in units of the mean edge length per full scalar range; 0 ignores the scalars}; closed surfaces stay closed, open borders are kept; message {faces_before, faces_after}",
        |_, o, inp| {
            let mut m = mesh_in("mesh_decimate", inp, 0)?;
            let before = m.faces.len();
            let target = match (o["target_faces"].as_u64(), o["ratio"].as_f64()) {
                (Some(t), _) => t as usize,
                (None, Some(r)) if r > 0.0 && r <= 1.0 => (before as f64 * r) as usize,
                _ => return inv("mesh_decimate: options.target_faces or options.ratio (0..1] required"),
            };
            let prot = if o["preserve_scalars"].as_bool().unwrap_or(true) { m.scalar_extremes(0.01) } else { None };
            m.decimate(target.max(4), prot.as_deref(), o["scalar_weight"].as_f64().unwrap_or(0.5));
            Ok((json!({"faces_before": before, "faces_after": m.faces.len(), "vertices": m.verts.len()}).to_string(), vec![m.to_value()]))
        },
    );
    r.register(
        "mesh_cut",
        "keep one side of a plane: inputs [mesh]; options {origin [x,y,z] mm (default: bbox centre), normal [x,y,z] (default z), keep: positive | negative (default positive), cap: close the cut with fans (default false)}",
        |_, o, inp| {
            let mut m = mesh_in("mesh_cut", inp, 0)?;
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for v in &m.verts {
                for a in 0..3 {
                    lo[a] = lo[a].min(v[a] as f64);
                    hi[a] = hi[a].max(v[a] as f64);
                }
            }
            let vec3 = |k: &str| nums(o, k).filter(|v| v.len() == 3).map(|v| [v[0], v[1], v[2]]);
            let origin = vec3("origin").unwrap_or([0, 1, 2].map(|a| (lo[a] + hi[a]) / 2.0));
            let normal = vec3("normal").unwrap_or([0.0, 0.0, 1.0]);
            if dot3(normal, normal) == 0.0 {
                return inv("mesh_cut: normal must not be zero");
            }
            let keep = match o["keep"].as_str().unwrap_or("positive") {
                "positive" => true,
                "negative" => false,
                other => return inv(format!("mesh_cut: keep must be positive | negative, got {other:?}")),
            };
            let before = m.faces.len();
            m.cut(origin, normal, keep, o["cap"].as_bool().unwrap_or(false));
            if m.faces.is_empty() {
                return inv("mesh_cut: nothing is left on that side of the plane");
            }
            Ok((json!({"faces_before": before, "faces_after": m.faces.len(), "origin": origin}).to_string(), vec![m.to_value()]))
        },
    );
    r.register(
        "centerline_from_mask",
        "centerlines (TEASAR) of a vessel tree: inputs [tensor [Z,Y,X]]; options {label | threshold (0.5), spacing_mm, affine (output in RAS mm), coordinates: \"voxel\" for voxel indices (flow viewer centerlines), root [x,y,z] voxel (default: the tip farthest from the thickest point), scale (1.5) and const_mm (2) of the branch-removal balls, min_length_mm (4), smooth (2 iterations)}; output lines value {kind lines, points [N,3], offsets, radius [N] mm, arclength [N] mm from the root along the tree, branch [N] id, space}; message {branches, points, length_mm, max_radius_mm, branch_info: [{id, parent (-1: root branch), join_point (global point index on the parent), points, length_mm, mean / min / max / tip radius}]}",
        |_, o, inp| {
            let t = inp.first().and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("centerline_from_mask: input 0 must be a tensor [Z, Y, X]".into()))?;
            let &[nz, ny, nx] = t.shape() else { return inv(format!("centerline_from_mask: tensor must be [Z, Y, X], got {:?}", t.shape())) };
            let sel = select(t, 0, nx * ny * nz, o);
            if sel.iter().all(|&x| x == 0) {
                return inv("centerline_from_mask: no voxel selected (check label / threshold)");
            }
            let (comp, _) = largest_component(&sel, [nx, ny, nz]);
            // crop to the component's box (+1)
            let (mut lo, mut hi) = ([nx, ny, nz], [0usize; 3]);
            for (q, &c) in comp.iter().enumerate() {
                if c != 0 {
                    let p = [q % nx, (q / nx) % ny, q / (nx * ny)];
                    for a in 0..3 {
                        lo[a] = lo[a].min(p[a]);
                        hi[a] = hi[a].max(p[a]);
                    }
                }
            }
            let org = [0, 1, 2].map(|a| lo[a].saturating_sub(1));
            let cd = [0, 1, 2].map(|a| (hi[a] + 2).min([nx, ny, nz][a]) - org[a]);
            let mut crop = vec![0u8; cd[0] * cd[1] * cd[2]];
            for q in 0..comp.len() {
                if comp[q] != 0 {
                    let p = [q % nx - org[0], (q / nx) % ny - org[1], q / (nx * ny) - org[2]];
                    crop[(p[2] * cd[1] + p[1]) * cd[0] + p[0]] = 1;
                }
            }
            let h = nums(o, "spacing_mm").filter(|s| s.len() == 3 && s.iter().all(|&v| v > 0.0)).map(|s| [s[0], s[1], s[2]])
                .or_else(|| affine_opt(o).ok().flatten().map(|a| [0, 1, 2].map(|c| (a[c] * a[c] + a[4 + c] * a[4 + c] + a[8 + c] * a[8 + c]).sqrt())))
                .unwrap_or([1.0; 3]);
            let root = nums(o, "root").filter(|r| r.len() == 3).map(|r| [0, 1, 2].map(|a| (r[a].round().max(0.0) as usize).saturating_sub(org[a]).min(cd[a] - 1)));
            let (branches, dt) = teasar(&crop, cd, h, root, o["scale"].as_f64().unwrap_or(1.5), o["const_mm"].as_f64().unwrap_or(2.0), o["min_length_mm"].as_f64().unwrap_or(4.0));
            if branches.is_empty() {
                return inv("centerline_from_mask: no branch longer than min_length_mm");
            }
            let (space, _a, _s, map) = space_map(o)?;
            let voxel_out = o["coordinates"].as_str() == Some("voxel");
            let vox = |q: usize| [(q % cd[0] + org[0]) as f64, ((q / cd[0]) % cd[1] + org[1]) as f64, (q / (cd[0] * cd[1]) + org[2]) as f64];
            let mut pts: Vec<[f64; 3]> = Vec::new();
            let mut radius: Vec<f32> = Vec::new();
            let mut arclen: Vec<f32> = Vec::new();
            let mut bid: Vec<u32> = Vec::new();
            let mut offsets: Vec<u32> = vec![0];
            let (mut length, mut rmax) = (0.0f64, 0.0f32);
            let mut owner: HashMap<usize, (usize, usize)> = HashMap::new();   // voxel → (branch, point index in it)
            let mut base: Vec<usize> = Vec::new();
            let mut info: Vec<Json> = Vec::new();
            let (mut parent_of, mut join_of): (Vec<i32>, Vec<i32>) = (Vec::new(), Vec::new());
            for (bi, b) in branches.iter().enumerate() {
                let join = *b.pts.last().unwrap();
                let parent = owner.get(&join).copied().filter(|(pb, _)| *pb != bi);
                for (k, &q) in b.pts.iter().enumerate() {
                    owner.entry(q).or_insert((bi, k));
                }
                let mut line: Vec<[f64; 3]> = b.pts.iter().map(|&q| vox(q)).collect();
                let mut rad: Vec<f32> = b.pts.iter().map(|&q| dt[q]).collect();
                // the geodesic runs into the corner of the cap voxel at a tip: replace the part where the lumen radius is still collapsing by a straight continuation of the stable part
                let fix_end = |line: &mut Vec<[f64; 3]>, rad: &mut Vec<f32>| {
                    let n = line.len();
                    if n < 8 {
                        return;
                    }
                    // reference radius: the vessel next to the tip (not the whole branch, which may run on into a much thicker trunk)
                    let mut mid: Vec<f32> = rad[(n / 8).max(1)..(n / 3).max(n / 8 + 2).min(n)].to_vec();
                    mid.sort_by(|a, b| a.total_cmp(b));
                    let r_ref = mid[mid.len() / 2];
                    let Some(k) = (0..n - 4).find(|&i| rad[i] >= 0.95 * r_ref) else { return };
                    if k == 0 || k as f64 * h[0].min(h[1]).min(h[2]) > 4.0 * r_ref as f64 {
                        return;       // nothing collapsing at the end, or a collapse longer than any tip effect: leave the line alone
                    }
                    // direction of the stable part over a long baseline (a few points would amplify the voxel noise over the length of the extension)
                    let far = (k + 14).min(n - 1);
                    let d = unit3(sub3(line[k], line[far]));
                    let inside = |p: V3| -> bool {
                        let (x, y, z) = (p[0].round() as i64 - org[0] as i64, p[1].round() as i64 - org[1] as i64, p[2].round() as i64 - org[2] as i64);
                        x >= 0 && y >= 0 && z >= 0 && x < cd[0] as i64 && y < cd[1] as i64 && z < cd[2] as i64 && crop[(z as usize * cd[1] + y as usize) * cd[0] + x as usize] != 0
                    };
                    let mut ext = Vec::new();
                    let mut p = line[k];
                    for _ in 0..((3.0 * r_ref as f64 / h[0].min(h[1]).min(h[2])).ceil() as usize + 2) {
                        let q = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
                        if !inside(q) {
                            break;
                        }
                        ext.push(q);
                        p = q;
                    }
                    let rk = rad[k];
                    ext.reverse();
                    let mut nl = ext.clone();
                    nl.extend_from_slice(&line[k..]);
                    let mut nr = vec![rk; ext.len()];
                    nr.extend_from_slice(&rad[k..]);
                    *line = nl;
                    *rad = nr;
                };
                // centring: move each interior point to the centroid of the lumen cross-section ⟂ to the line (the discrete path wanders by about a voxel where
                // several voxels are equally far from the wall); not near the ends, where the cross-section is a cap or a junction
                let hh = [h[0], h[1], h[2]];
                let tip_end_is_root = parent.is_none();
                let recentre = |line: &mut Vec<[f64; 3]>, rad: &Vec<f32>| {
                    let n = line.len();
                    let src = line.clone();
                    let inside = |p: V3| -> bool {
                        let (x, y, z) = (p[0].round() as i64 - org[0] as i64, p[1].round() as i64 - org[1] as i64, p[2].round() as i64 - org[2] as i64);
                        x >= 0 && y >= 0 && z >= 0 && x < cd[0] as i64 && y < cd[1] as i64 && z < cd[2] as i64 && crop[(z as usize * cd[1] + y as usize) * cd[0] + x as usize] != 0
                    };
                    for i in 0..n {
                        // tips (the start of every line, and the end of the root branch) have a plain cap: recentre right up to the end; a junction end needs 2 radii
                        let junction_skip = (2.0 * rad[i] as f64 / hh[0].min(hh[1]).min(hh[2])).ceil() as usize + 1;
                        let skip_hi = if tip_end_is_root { 0 } else { junction_skip };
                        if i + skip_hi >= n {
                            continue;
                        }
                        // tangent: a short symmetric window inside the line, a longer one-sided window towards the interior at the ends (the last voxels hook around the cap)
                        let (a, b) = if i < 8 { (src[i], src[(i + 8).min(n - 1)]) } else if i + 8 >= n { (src[i.saturating_sub(8)], src[i]) } else { (src[i - 3], src[i + 3]) };
                        let t = unit3([(b[0] - a[0]) * hh[0], (b[1] - a[1]) * hh[1], (b[2] - a[2]) * hh[2]]);
                        let axis = if t[0].abs() < 0.8 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                        let u = unit3(cross3(t, axis));
                        let v = cross3(t, u);
                        let pm = [src[i][0] * hh[0], src[i][1] * hh[1], src[i][2] * hh[2]];
                        let ds = 0.5 * hh[0].min(hh[1]).min(hh[2]);
                        // window sized by the vessel next to this point, not by the collapsing radius of a tip voxel
                        let r_win = rad[i.saturating_sub(8)..(i + 9).min(n)].iter().cloned().fold(0.0f32, f32::max) as f64;
                        let m = ((2.5 * r_win) / ds).ceil() as i64;
                        let w = (2 * m + 1) as usize;
                        let mut grid = vec![false; w * w];
                        for ia in -m..=m {
                            for ic in -m..=m {
                                let q = [0, 1, 2].map(|k| pm[k] + (ia as f64 * u[k] + ic as f64 * v[k]) * ds);
                                grid[(ia + m) as usize * w + (ic + m) as usize] = inside([q[0] / hh[0], q[1] / hh[1], q[2] / hh[2]]);
                            }
                        }
                        let centre = m as usize * w + m as usize;
                        if !grid[centre] {
                            continue;
                        }
                        let mut seen = vec![false; w * w];
                        let (mut st, mut sa, mut sc, mut cnt) = (vec![centre], 0.0f64, 0.0f64, 0.0f64);
                        seen[centre] = true;
                        while let Some(q) = st.pop() {
                            sa += (q / w) as f64 - m as f64;
                            sc += (q % w) as f64 - m as f64;
                            cnt += 1.0;
                            let (a2, c2) = (q / w, q % w);
                            for (da, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                                let (na, nc) = (a2 as i64 + da, c2 as i64 + dc);
                                if na >= 0 && nc >= 0 && (na as usize) < w && (nc as usize) < w {
                                    let nq = na as usize * w + nc as usize;
                                    if grid[nq] && !seen[nq] {
                                        seen[nq] = true;
                                        st.push(nq);
                                    }
                                }
                            }
                        }
                        let (ca, cc) = (sa / cnt, sc / cnt);
                        // a cross-section that touches the border of the sampling grid is cut off (a junction): leave the point
                        let at_tip = i == 0 || (tip_end_is_root && i + 1 == n);       // the very end voxel of a tip can sit at the corner of the cap
                        if (ca.abs() + cc.abs()) * ds > if at_tip { 3.0 } else { 1.5 } * hh[0].max(hh[1]).max(hh[2]) {
                            continue;
                        }
                        let q = [0, 1, 2].map(|k| pm[k] + (ca * u[k] + cc * v[k]) * ds);
                        line[i] = [q[0] / hh[0], q[1] / hh[1], q[2] / hh[2]];
                    }
                };
                // interior first (the ends still hook around the cap and would bias the cross-sections), then cut the hooks, then the ends too
                recentre(&mut line, &rad);
                fix_end(&mut line, &mut rad);
                if parent.is_none() {
                    line.reverse();
                    rad.reverse();
                    fix_end(&mut line, &mut rad);
                    line.reverse();
                    rad.reverse();
                }
                recentre(&mut line, &rad);
                for _ in 0..o["smooth"].as_u64().unwrap_or(4) {          // light 3-tap smoothing of the interior, ends stay on the tree
                    let prev = line.clone();
                    for i in 1..line.len().saturating_sub(1) {
                        line[i] = [0, 1, 2].map(|c| 0.25 * prev[i - 1][c] + 0.5 * prev[i][c] + 0.25 * prev[i + 1][c]);
                    }
                }
                let mm: Vec<[f64; 3]> = line.iter().map(|&p| if voxel_out { p } else { map(p) }).collect();
                let phys: Vec<[f64; 3]> = line.iter().map(|&p| map(p)).collect();
                let seg: Vec<f64> = phys.windows(2).map(|w| ((w[0][0] - w[1][0]).powi(2) + (w[0][1] - w[1][1]).powi(2) + (w[0][2] - w[1][2]).powi(2)).sqrt()).collect();
                // arc length from the root along the tree: the branch ends at its junction, whose value is known from the parent
                let s_join = parent.map(|(pb, pk)| arclen[base[pb] + pk] as f64).unwrap_or(0.0);
                let mut s = vec![0f64; line.len()];
                s[line.len() - 1] = s_join;
                for i in (0..line.len() - 1).rev() {
                    s[i] = s[i + 1] + seg[i];
                }
                parent_of.push(parent.map(|(pb, _)| pb as i32).unwrap_or(-1));
                join_of.push(parent.map(|(pb, pk)| (base[pb] + pk) as i32).unwrap_or(-1));
                base.push(pts.len());
                let blen: f64 = seg.iter().sum();
                length += blen;
                rmax = rad.iter().cloned().fold(rmax, f32::max);
                info.push(json!({"id": bi, "parent": parent.map(|(pb, _)| pb as i64).unwrap_or(-1), "join_point": parent.map(|(pb, pk)| base[pb] + pk),
                    "points": line.len(), "length_mm": blen, "mean_radius_mm": rad.iter().map(|&x| x as f64).sum::<f64>() / rad.len() as f64,
                    "min_radius_mm": rad.iter().cloned().fold(f32::INFINITY, f32::min), "max_radius_mm": rad.iter().cloned().fold(0.0, f32::max), "tip_radius_mm": rad[0]}));
                pts.extend(mm);
                radius.extend(rad);
                arclen.extend(s.iter().map(|&x| x as f32));
                bid.extend(std::iter::repeat(bi as u32).take(line.len()));
                offsets.push(pts.len() as u32);
            }
            let flat: Vec<f32> = pts.iter().flatten().map(|&x| x as f32).collect();
            let n = pts.len();
            let v = Value::Dict(vec![
                ("kind".into(), Value::Str("lines".into())),
                ("points".into(), Value::Tensor(Tensor::from_slice(&[n, 3], &flat).unwrap())),
                ("offsets".into(), Value::Tensor(Tensor::from_slice(&[offsets.len()], &offsets).unwrap())),
                ("radius".into(), Value::Tensor(Tensor::from_slice(&[n], &radius).unwrap())),
                ("arclength".into(), Value::Tensor(Tensor::from_slice(&[n], &arclen).unwrap())),
                ("branch".into(), Value::Tensor(Tensor::from_slice(&[n], &bid).unwrap())),
                ("parent".into(), Value::Tensor(Tensor::from_slice(&[parent_of.len()], &parent_of).unwrap())),
                ("join_point".into(), Value::Tensor(Tensor::from_slice(&[join_of.len()], &join_of).unwrap())),
                ("space".into(), Value::Str(if voxel_out { "voxel".into() } else { space.into() })),
            ]);
            // remember the grid the line came from, so later measurements need no options
            let mut v = v;
            if let Value::Dict(d) = &mut v {
                if let Some(a) = affine_opt(o).ok().flatten() {
                    d.push(("affine".into(), Value::Tensor(Tensor::from_slice(&[16], &a).unwrap())));
                }
                if let Some(sp) = nums(o, "spacing_mm").filter(|s| s.len() == 3) {
                    d.push(("spacing_mm".into(), Value::Tensor(Tensor::from_slice(&[3], &sp).unwrap())));
                }
            }
            Ok((json!({"branches": branches.len(), "points": n, "length_mm": length, "max_radius_mm": rmax, "branch_info": info}).to_string(), vec![v]))
        },
    );
    r.register(
        "mesh_fill_holes",
        "close open boundary loops: inputs [mesh]; options {max_edges (80): loops with more edges stay open (cut-open vessel ends), fair (true): refine the patch to the boundary edge length and relax its interior vertices, cotan (true): with cotangent weights (a minimal-surface patch), else uniform}; message {filled, remaining, open_edges}",
        |_, o, inp| {
            let mut m = mesh_in("mesh_fill_holes", inp, 0)?;
            let (filled, remaining) = m.fill_holes(o["max_edges"].as_u64().unwrap_or(80) as usize, o["fair"].as_bool().unwrap_or(true), o["cotan"].as_bool().unwrap_or(true));
            Ok((json!({"filled": filled, "remaining": remaining, "open_edges": m.boundary_edges()}).to_string(), vec![m.to_value()]))
        },
    );
    r.register(
        "mesh_voxelize",
        "inside / outside mask of a closed mesh: inputs [mesh]; options {voxel_mm (default: longest side / 128), margin (2 voxels)}; output u8 [Z,Y,X]; message {spacing_mm, affine (4×4 voxel→mm rows), dims, voxels}",
        |_, o, inp| {
            let m = mesh_in("mesh_voxelize", inp, 0)?;
            let (lo, hi) = bbox_of(&m);
            let (origin, h, dims) = grid_for(lo, hi, o["voxel_mm"].as_f64(), o["margin"].as_u64().unwrap_or(2) as usize)?;
            let v = voxelize(&m, origin, h, dims);
            let n = v.iter().filter(|&&x| x != 0).count();
            let a = grid_affine(origin, h);
            let rows: Vec<Vec<f64>> = a.chunks(4).map(|r| r.to_vec()).collect();
            Ok((json!({"spacing_mm": [h, h, h], "affine": rows, "dims": dims, "voxels": n}).to_string(), vec![Value::Tensor(Tensor::from_slice(&[dims[2], dims[1], dims[0]], &v).unwrap())]))
        },
    );
    r.register(
        "mesh_boolean",
        "union / intersection / difference (A minus B) of two closed meshes, through a shared voxel grid (accurate to about one voxel, not exact): inputs [A, B]; options {op: union | intersection | difference, voxel_mm (default: longest side / 160), blur (1), smooth (6), refine (true): move the result vertices onto the nearest input surface within 1.5 voxels and those near both surfaces onto the exact intersection curve of the two meshes}; output mesh in RAS space",
        |_, o, inp| {
            let (a, b) = (mesh_in("mesh_boolean", inp, 0)?, mesh_in("mesh_boolean", inp, 1)?);
            let (la, ha) = bbox_of(&a);
            let (lb, hb) = bbox_of(&b);
            let (lo, hi) = ([0, 1, 2].map(|c| la[c].min(lb[c])), [0, 1, 2].map(|c| ha[c].max(hb[c])));
            let ext = [0, 1, 2].map(|c| hi[c] - lo[c]);
            let h = o["voxel_mm"].as_f64().filter(|&h| h > 0.0).unwrap_or_else(|| ext.iter().cloned().fold(0.0, f64::max) / 160.0);
            let (origin, h, dims) = grid_for(lo, hi, Some(h), 3)?;
            let (ma, mb) = (voxelize(&a, origin, h, dims), voxelize(&b, origin, h, dims));
            let op = o["op"].as_str().unwrap_or("union");
            let mask: Vec<u8> = match op {
                "union" => ma.iter().zip(&mb).map(|(&x, &y)| x | y).collect(),
                "intersection" => ma.iter().zip(&mb).map(|(&x, &y)| x & y).collect(),
                "difference" => ma.iter().zip(&mb).map(|(&x, &y)| x & (1 - y)).collect(),
                other => return inv(format!("mesh_boolean: op must be union | intersection | difference, got {other:?}")),
            };
            if mask.iter().all(|&x| x == 0) {
                return inv("mesh_boolean: the result is empty");
            }
            let opts = json!({"affine": grid_affine(origin, h).to_vec(), "keep": "all", "blur": o["blur"].as_u64().unwrap_or(1), "smooth": o["smooth"].as_u64().unwrap_or(6)});
            let (mut m, _) = mask_to_mesh(mask, dims, &opts)?;
            let (mut snapped, mut crease) = (0, 0);
            if o["refine"].as_bool().unwrap_or(true) {
                let segs = intersection_segments(&a, &b, 2.0 * h);
                let creased = snap_crease(&mut m, &a, &b, &segs, h);
                crease = creased.iter().filter(|&&c| c).count();
                snapped = snap_to(&mut m, &[&a, &b], 1.5 * h, Some(&creased)) + crease;
            }
            Ok((json!({"op": op, "voxel_mm": h, "snapped_vertices": snapped, "crease_vertices": crease, "vertices": m.verts.len(), "faces": m.faces.len(), "volume_mm3": m.volume()}).to_string(), vec![m.to_value()]))
        },
    );
    r.register(
        "meshseq_track",
        "temporally coherent surfaces of a 4-D mask: one topology for all frames, vertices follow the wall (frame 0 iso-surface, then Newton snapping onto each frame's 0.5 level): inputs [tensor [T,Z,Y,X]]; options as mesh_from_mask plus dt, steps (12), relax (0.15); output {kind mesh_seq, coherent (all frames linked), links u8 [T-1] (1: frame i+1 has the vertices of frame i; 0: new surface after a split / merge / a shape the topology cannot follow), frames, dt}",
        |_, o, inp| {
            let t = inp.first().and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("meshseq_track: input 0 must be a tensor [T, Z, Y, X]".into()))?;
            let (meshes, links, parents) = track_sequence(t, o)?;
            let area: Vec<f64> = meshes.iter().map(|m| m.area()).collect();
            let lk: Vec<u8> = links.iter().map(|&b| b as u8).collect();
            let v = Value::Dict(vec![
                ("kind".into(), Value::Str("mesh_seq".into())),
                ("coherent".into(), Value::Bool(links.iter().all(|&b| b))),
                ("links".into(), Value::Tensor(Tensor::from_slice(&[lk.len()], &lk).unwrap())),
                ("frames".into(), Value::List(meshes.iter().zip(&parents).map(|(m, par)| {
                    let mut d = m.to_value();
                    if let (Value::Dict(v), Some(p)) = (&mut d, par) {
                        v.push(("parent".into(), Value::Tensor(Tensor::from_slice(&[p.len()], p).unwrap())));      // index of the nearest vertex of the previous frame
                    }
                    d
                }).collect())),
                ("dt".into(), Value::Float(o["dt"].as_f64().unwrap_or(0.04))),
            ]);
            Ok((json!({"frames": meshes.len(), "vertices": meshes.iter().map(|m| m.verts.len()).collect::<Vec<_>>(), "links": links, "area_mm2": area}).to_string(), vec![v]))
        },
    );
    r.register(
        "centerline_measure",
        "cross-section area, tortuosity and bifurcation angles along a centerline: inputs [lines (from centerline_from_mask), mask tensor [Z,Y,X]]; options {label | threshold (0.5), spacing_mm, affine (of the mask; needed when the lines are in RAS), window (3 points for the tangent), angle_near_mm / angle_far_mm (the limb directions are chords between these distances from the junction; default 1.5 r + 2 and 4.5 r + 6)}; output lines value with area_mm2 and eq_diameter_mm per point; message {branches: [{id, parent, length_mm, chord_mm, tortuosity (a branch that bends at a junction is not straight: see segments), mean / min / max eq_diameter_mm, mean_area_mm2}], segments: [{branch, from_point, to_point, length_mm, core_length_mm, core_chord_mm, tortuosity (of the core, 1.5 radii in from each end), mean / min eq_diameter_mm, mean_area_mm2}] split where daughters join, junctions: [{daughter, parent, angle_between_daughters_deg, daughter_to_trunk_deg, parent_distal_to_trunk_deg}]}",
        |_, o, inp| {
            let lv = inp.first().ok_or_else(|| Error::Invalid("centerline_measure: input 0 must be a lines value".into()))?;
            let mt = inp.get(1).and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("centerline_measure: input 1 must be the mask tensor [Z, Y, X]".into()))?;
            let &[nz, ny, nx] = mt.shape() else { return inv(format!("centerline_measure: mask must be [Z, Y, X], got {:?}", mt.shape())) };
            let tensor_f = |k: &str| -> Option<Vec<f64>> { lv.get(k).and_then(|t| t.as_tensor()).map(|t| (0..t.numel()).map(|i| t.get_f64(i).0).collect()) };
            let pts_flat = tensor_f("points").ok_or_else(|| Error::Invalid("centerline_measure: lines value without points".into()))?;
            let (radius, arclen, branch, offsets) = (tensor_f("radius"), tensor_f("arclength"), tensor_f("branch"), tensor_f("offsets"));
            let (Some(radius), Some(arclen), Some(branch), Some(offsets)) = (radius, arclen, branch, offsets) else { return inv("centerline_measure: the lines need radius, arclength, branch and offsets (use centerline_from_mask)") };
            let (parent, join) = (tensor_f("parent").unwrap_or_default(), tensor_f("join_point").unwrap_or_default());
            let n = pts_flat.len() / 3;
            let space = match lv.get("space") { Some(Value::Str(s)) => s.clone(), _ => "voxel_mm".into() };
            let own = |k: &str, n: usize| -> Option<Vec<f64>> { lv.get(k).and_then(|t| t.as_tensor()).map(|t| (0..t.numel()).map(|i| t.get_f64(i).0).collect()).filter(|v: &Vec<f64>| v.len() == n) };
            let spacing = nums(o, "spacing_mm").or_else(|| own("spacing_mm", 3)).filter(|s| s.len() == 3 && s.iter().all(|&v| v > 0.0)).map(|s| [s[0], s[1], s[2]]);
            let affine = affine_opt(o)?.or_else(|| own("affine", 16).map(|a| std::array::from_fn(|i| a[i])));
            // points in mm, and the way back to voxel indices
            let mut p: Vec<V3> = (0..n).map(|i| [pts_flat[3 * i], pts_flat[3 * i + 1], pts_flat[3 * i + 2]]).collect();
            let to_vox: Box<dyn Fn(V3) -> V3> = match space.as_str() {
                "ras" => {
                    let inv_a = affine.as_ref().and_then(invert).ok_or_else(|| Error::Invalid("centerline_measure: lines are in RAS space: options.affine of the mask required".into()))?;
                    Box::new(move |q| apply(&inv_a, q))
                }
                "voxel" => {
                    let s = spacing.ok_or_else(|| Error::Invalid("centerline_measure: lines are in voxel indices: options.spacing_mm of the mask required".into()))?;
                    for q in p.iter_mut() { *q = [q[0] * s[0], q[1] * s[1], q[2] * s[2]]; }
                    Box::new(move |q| [q[0] / s[0], q[1] / s[1], q[2] / s[2]])
                }
                _ => {
                    let s = spacing.ok_or_else(|| Error::Invalid("centerline_measure: options.spacing_mm of the mask required".into()))?;
                    Box::new(move |q| [q[0] / s[0], q[1] / s[1], q[2] / s[2]])
                }
            };
            let vox_mm = spacing.map(|s| s[0].min(s[1]).min(s[2])).or_else(|| affine.map(|a| [0, 1, 2].map(|c| (a[c] * a[c] + a[4 + c] * a[4 + c] + a[8 + c] * a[8 + c]).sqrt()).iter().cloned().fold(f64::INFINITY, f64::min))).unwrap_or(1.0);
            let sel = select(mt, 0, nx * ny * nz, o);
            let inside = |q: V3| -> bool {
                let v = to_vox(q);
                let (i, j, k) = (v[0].round() as i64, v[1].round() as i64, v[2].round() as i64);
                i >= 0 && j >= 0 && k >= 0 && i < nx as i64 && j < ny as i64 && k < nz as i64 && sel[(k as usize * ny + j as usize) * nx + i as usize] != 0
            };
            let win = o["window"].as_u64().unwrap_or(3) as usize;
            let nb = offsets.len() - 1;
            let ds = vox_mm * 0.7;
            let (mut area, mut eqd) = (vec![0f32; n], vec![0f32; n]);
            let range_of = |b: usize| (offsets[b] as usize, offsets[b + 1] as usize);
            for b in 0..nb {
                let (b0, b1) = range_of(b);
                for i in b0..b1 {
                    let (j0, j1) = (i.saturating_sub(win).max(b0), (i + win).min(b1 - 1));
                    let t = unit3(sub3(p[j1], p[j0]));
                    let axis = if t[0].abs() < 0.8 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                    let u = unit3(cross3(t, axis));
                    let v = cross3(t, u);
                    let r = (2.0 * radius[i] + 2.0 * ds).min(60.0);
                    let m = (r / ds).ceil() as i64;
                    let w = (2 * m + 1) as usize;
                    let mut grid = vec![false; w * w];
                    for a in -m..=m {
                        for c in -m..=m {
                            let q = [p[i][0] + (a as f64 * u[0] + c as f64 * v[0]) * ds, p[i][1] + (a as f64 * u[1] + c as f64 * v[1]) * ds, p[i][2] + (a as f64 * u[2] + c as f64 * v[2]) * ds];
                            grid[(a + m) as usize * w + (c + m) as usize] = inside(q);
                        }
                    }
                    // the connected cross-section around the centre cell (the nearest inside cell when the centre itself is outside)
                    let centre = (m as usize) * w + m as usize;
                    let start = if grid[centre] { Some(centre) } else { (0..w * w).filter(|&q| grid[q]).min_by_key(|&q| { let (a, c) = ((q / w) as i64 - m, (q % w) as i64 - m); a * a + c * c }) };
                    let mut cnt = 0usize;
                    if let Some(s0) = start {
                        let mut seen = vec![false; w * w];
                        let mut st = vec![s0];
                        seen[s0] = true;
                        while let Some(q) = st.pop() {
                            cnt += 1;
                            let (a, c) = (q / w, q % w);
                            for (da, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                                let (na, nc) = (a as i64 + da, c as i64 + dc);
                                if na >= 0 && nc >= 0 && (na as usize) < w && (nc as usize) < w {
                                    let nq = na as usize * w + nc as usize;
                                    if grid[nq] && !seen[nq] {
                                        seen[nq] = true;
                                        st.push(nq);
                                    }
                                }
                            }
                        }
                    }
                    area[i] = (cnt as f64 * ds * ds) as f32;
                    eqd[i] = (2.0 * (cnt as f64 * ds * ds / std::f64::consts::PI).sqrt()) as f32;
                }
            }
            let mut branches = Vec::new();
            for b in 0..nb {
                let (b0, b1) = range_of(b);
                let length: f64 = (b0..b1 - 1).map(|i| dot3(sub3(p[i + 1], p[i]), sub3(p[i + 1], p[i])).sqrt()).sum();
                // measure from the tip's first points inward: the end voxels carry the wall (tip) and the junction (trunk) distortions, so trim 10 % at each end
                let (t0, t1) = (b0 + (b1 - b0) / 10, b1 - (b1 - b0) / 10);
                let core = &eqd[t0..t1.max(t0 + 1).min(b1)];
                let chord = dot3(sub3(p[b1 - 1], p[b0]), sub3(p[b1 - 1], p[b0])).sqrt();
                branches.push(json!({"id": b, "parent": parent.get(b).copied().unwrap_or(-1.0) as i64, "length_mm": length, "chord_mm": chord, "tortuosity": if chord > 0.0 { length / chord } else { 1.0 },
                    "mean_eq_diameter_mm": core.iter().map(|&x| x as f64).sum::<f64>() / core.len().max(1) as f64, "min_eq_diameter_mm": core.iter().cloned().fold(f32::INFINITY, f32::min),
                    "max_eq_diameter_mm": core.iter().cloned().fold(0.0, f32::max), "mean_area_mm2": area[t0..t1.max(t0 + 1).min(b1)].iter().map(|&x| x as f64).sum::<f64>() / core.len().max(1) as f64}));
            }
            // segments between junctions: a branch is split where daughters join it, so tortuosity and diameters are not mixed across a bifurcation
            let mut segments = Vec::new();
            for b in 0..nb {
                let (b0, b1) = range_of(b);
                let mut cuts: Vec<usize> = (0..nb).filter(|&d| parent.get(d).copied() == Some(b as f64)).filter_map(|d| join.get(d).map(|&j| j as usize)).filter(|&j| j > b0 && j + 1 < b1).collect();
                cuts.sort_unstable();
                let mut starts = vec![b0];
                starts.extend(cuts.iter().copied());
                let mut ends = cuts.clone();
                ends.push(b1 - 1);
                for (&s0, &s1) in starts.iter().zip(&ends) {
                    if s1 <= s0 + 1 {
                        continue;
                    }
                    let length_full: f64 = (s0..s1).map(|i| dot3(sub3(p[i + 1], p[i]), sub3(p[i + 1], p[i])).sqrt()).sum();
                    // the ends of a segment sit in a tip (cap of the lumen) or a junction: measure the core, from 1.5 radii in from each end (at most 30 % each side)
                    let walk_in = |from: usize, step: i64| -> usize {
                        let lim = 0.3 * length_full;
                        let (mut k, mut acc) = (from as i64, 0.0);
                        let reach = 1.5 * radius[from];
                        while (k + step) >= s0 as i64 && (k + step) <= s1 as i64 && acc < reach.min(lim) {
                            acc += dot3(sub3(p[(k + step) as usize], p[k as usize]), sub3(p[(k + step) as usize], p[k as usize])).sqrt();
                            k += step;
                        }
                        k as usize
                    };
                    let (t0, t1) = (walk_in(s0, 1), walk_in(s1, -1));
                    let (t0, t1) = if t1 > t0 + 1 { (t0, t1) } else { (s0, s1) };
                    let length: f64 = (t0..t1).map(|i| dot3(sub3(p[i + 1], p[i]), sub3(p[i + 1], p[i])).sqrt()).sum();
                    let chord = dot3(sub3(p[t1], p[t0]), sub3(p[t1], p[t0])).sqrt();
                    let core = &eqd[t0..=t1];
                    segments.push(json!({"branch": b, "from_point": s0, "to_point": s1, "length_mm": length, "core_chord_mm": chord, "tortuosity": if chord > 0.0 { length / chord } else { 1.0 },
                        "mean_eq_diameter_mm": core.iter().map(|&x| x as f64).sum::<f64>() / core.len().max(1) as f64, "min_eq_diameter_mm": core.iter().cloned().fold(f32::INFINITY, f32::min),
                        "mean_area_mm2": area[t0..=t1].iter().map(|&x| x as f64).sum::<f64>() / core.len().max(1) as f64}));
                }
            }
            // bifurcations: each daughter ends at a point of its parent; the directions of the three limbs are chords of the centerline away from the junction
            // (from 1.5 r + 2 mm to 4.5 r + 6 mm along each limb, r = lumen radius at the junction), which skips the rounded transition inside the junction
            let mut junctions = Vec::new();
            for b in 1..nb {
                let (Some(&pb), Some(&jp)) = (parent.get(b), join.get(b)) else { continue };
                if pb < 0.0 || jp < 0.0 {
                    continue;
                }
                let (pb, j) = (pb as usize, jp as usize);
                let (p0, p1) = range_of(pb);
                let rj = radius[j];
                let (near, far) = (o["angle_near_mm"].as_f64().unwrap_or(1.5 * rj + 2.0), o["angle_far_mm"].as_f64().unwrap_or(4.5 * rj + 6.0));
                // index along a limb starting at `from` (walking by `step` inside [lo, hi)) where the straight-line distance from the junction first reaches `d` mm
                let at_dist = |from: usize, step: i64, lo: usize, hi: usize, d: f64| -> usize {
                    let mut k = from as i64;
                    while k + step >= lo as i64 && k + step < hi as i64 && dot3(sub3(p[(k + step) as usize], p[from]), sub3(p[(k + step) as usize], p[from])).sqrt() < d {
                        k += step;
                    }
                    k as usize
                };
                let (d0, d1) = range_of(b);
                let limb = |from: usize, step: i64, lo: usize, hi: usize| unit3(sub3(p[at_dist(from, step, lo, hi, far)], p[at_dist(from, step, lo, hi, near)]));
                let daughter = limb(d1 - 1, -1, d0, d1);
                let distal = limb(j, -1, p0, p1);
                let proximal = limb(j, 1, p0, p1);
                let trunk = [-proximal[0], -proximal[1], -proximal[2]];
                junctions.push(json!({"daughter": b, "parent": pb, "angle_between_daughters_deg": angle_deg(daughter, distal), "daughter_to_trunk_deg": angle_deg(daughter, trunk), "parent_distal_to_trunk_deg": angle_deg(distal, trunk)}));
            }
            let mut d = match &**lv { Value::Dict(d) => d.clone(), _ => return inv("centerline_measure: input 0 is not a dict") };
            d.retain(|(k, _)| k != "area_mm2" && k != "eq_diameter_mm");
            d.push(("area_mm2".into(), Value::Tensor(Tensor::from_slice(&[n], &area).unwrap())));
            d.push(("eq_diameter_mm".into(), Value::Tensor(Tensor::from_slice(&[n], &eqd).unwrap())));
            let _ = arclen;
            let _ = branch;
            Ok((json!({"branches": branches, "segments": segments, "junctions": junctions}).to_string(), vec![Value::Dict(d)]))
        },
    );
    r.register(
        "meshseq_from_mask",
        "one surface per frame of a 4-D mask (moving walls): inputs [tensor [T,Z,Y,X]]; options as mesh_from_mask plus dt (s per frame); output {kind mesh_seq, frames: [mesh…], dt}; message {frames, vertices, area_mm2 per frame}",
        |_, o, inp| {
            let t = inp.first().and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("meshseq_from_mask: input 0 must be a tensor [T, Z, Y, X]".into()))?;
            let &[nt, nz, ny, nx] = t.shape() else { return inv(format!("meshseq_from_mask: tensor must be [T, Z, Y, X], got {:?}", t.shape())) };
            let nvox = nx * ny * nz;
            let mut frames = Vec::new();
            let mut area = Vec::new();
            for f in 0..nt {
                let sel = select(t, f * nvox, nvox, o);
                match mask_to_mesh(sel, [nx, ny, nz], o) {
                    Ok((m, _)) => {
                        area.push(m.area());
                        frames.push(m.to_value());
                    }
                    Err(e) => return inv(format!("meshseq_from_mask: frame {f}: {e}")),
                }
            }
            let v = Value::Dict(vec![
                ("kind".into(), Value::Str("mesh_seq".into())),
                ("frames".into(), Value::List(frames)),
                ("dt".into(), Value::Float(o["dt"].as_f64().unwrap_or(0.04))),
            ]);
            Ok((json!({"frames": nt, "area_mm2": area}).to_string(), vec![v]))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Ctx;
    use std::sync::Arc;

    fn sphere_mask(n: usize, r: f64) -> Arc<Value> {
        let c = (n as f64 - 1.0) / 2.0;
        let d: Vec<u8> = (0..n * n * n).map(|q| {
            let (i, j, k) = ((q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64);
            (((i - c).powi(2) + (j - c).powi(2) + (k - c).powi(2)).sqrt() <= r) as u8
        }).collect();
        Arc::new(Value::Tensor(Tensor::from_slice(&[n, n, n], &d).unwrap()))
    }

    fn sphere_mesh() -> (Registry, Ctx, Arc<Value>) {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let (_, out) = reg.run("mesh_from_mask", &ctx, r#"{"spacing_mm":[1,1,1],"blur":2,"smooth":4}"#, &[sphere_mask(48, 16.0)]).unwrap();
        (reg, ctx, Arc::new(out[0].clone()))
    }

    #[test]
    fn decimation_keeps_shape_and_closure() {
        let (reg, ctx, m) = sphere_mesh();
        let m0 = Mesh::from_value(&m).unwrap();
        let (msg, out) = reg.run("mesh_decimate", &ctx, r#"{"ratio":0.2}"#, &[m.clone()]).unwrap();
        let m1 = Mesh::from_value(&out[0]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        assert!(m1.faces.len() < m0.faces.len() / 3, "{} -> {}", m0.faces.len(), m1.faces.len());
        assert_eq!(j["faces_after"].as_u64().unwrap() as usize, m1.faces.len());
        assert_eq!(m1.boundary_edges(), 0, "still closed");
        assert_eq!(m1.components().1, 1);
        assert!((m1.area() - m0.area()).abs() / m0.area() < 0.05, "area {} vs {}", m1.area(), m0.area());
        assert!((m1.volume() - m0.volume()).abs() / m0.volume() < 0.06, "volume {} vs {}", m1.volume(), m0.volume());
        assert!(reg.run("mesh_decimate", &ctx, "{}", &[m]).is_err());
    }

    #[test]
    fn cut_halves_a_sphere_and_caps_it() {
        let (reg, ctx, m) = sphere_mesh();
        let m0 = Mesh::from_value(&m).unwrap();
        let c = m0.verts.iter().fold([0f64; 3], |s, v| [s[0] + v[0] as f64, s[1] + v[1] as f64, s[2] + v[2] as f64]);
        let c = [c[0] / m0.verts.len() as f64, c[1] / m0.verts.len() as f64, c[2] / m0.verts.len() as f64];
        let o = json!({"origin": c, "normal": [0, 0, 1], "cap": true}).to_string();
        let (_, out) = reg.run("mesh_cut", &ctx, &o, &[m.clone()]).unwrap();
        let h = Mesh::from_value(&out[0]).unwrap();
        assert_eq!(h.boundary_edges(), 0, "cap closes the cut");
        assert!(h.volume() > 0.0 && (h.volume() - m0.volume() / 2.0).abs() / m0.volume() < 0.04, "half volume {} of {}", h.volume(), m0.volume());
        let open = json!({"origin": c, "normal": [0, 0, 1]}).to_string();
        let (_, o2) = reg.run("mesh_cut", &ctx, &open, &[m]).unwrap();
        assert!(Mesh::from_value(&o2[0]).unwrap().boundary_edges() > 0);
    }

    #[test]
    fn teasar_finds_the_branches_of_a_y_tree() {
        let n = 64usize;
        let mut d = vec![0u8; n * n * n];
        let seg = |d: &mut Vec<u8>, a: [f64; 3], b: [f64; 3], r: f64| {
            let dv = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let l2 = dv.iter().map(|x| x * x).sum::<f64>();
            for q in 0..d.len() {
                let p = [(q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64];
                let t = (((p[0] - a[0]) * dv[0] + (p[1] - a[1]) * dv[1] + (p[2] - a[2]) * dv[2]) / l2).clamp(0.0, 1.0);
                let e = [a[0] + t * dv[0] - p[0], a[1] + t * dv[1] - p[1], a[2] + t * dv[2] - p[2]];
                if e.iter().map(|x| x * x).sum::<f64>() <= r * r {
                    d[q] = 1;
                }
            }
        };
        seg(&mut d, [32.0, 32.0, 4.0], [32.0, 32.0, 32.0], 5.0);
        seg(&mut d, [32.0, 32.0, 32.0], [12.0, 20.0, 58.0], 3.5);
        seg(&mut d, [32.0, 32.0, 32.0], [52.0, 44.0, 58.0], 3.5);
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let v = Arc::new(Value::Tensor(Tensor::from_slice(&[n, n, n], &d).unwrap()));
        let (msg, out) = reg.run("centerline_from_mask", &ctx, r#"{"spacing_mm":[1,1,1],"coordinates":"voxel"}"#, &[v]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        // three tips: one becomes the root, the other two give two branches (one of them runs through the junction up to the root)
        assert_eq!(j["branches"], 2, "{msg}");
        // tree length: stem 28 + two arms of 34.9 = 97.8 mm; voxel paths are a little longer than the straight lines
        let len = j["length_mm"].as_f64().unwrap();
        assert!(len > 90.0 && len < 125.0, "length {len}");
        let rmax = j["max_radius_mm"].as_f64().unwrap();
        assert!((rmax - 5.0).abs() < 0.8, "max radius {rmax}");
        // the points of the stem lie on its axis
        let pts = out[0].get("points").and_then(|p| p.as_tensor()).unwrap();
        let rad = out[0].get("radius").and_then(|p| p.as_tensor()).unwrap();
        let mut stem = 0;
        for i in 0..pts.shape()[0] {
            let (x, y, z) = (pts.get_f64(i * 3).0, pts.get_f64(i * 3 + 1).0, pts.get_f64(i * 3 + 2).0);
            if z > 8.0 && z < 28.0 && rad.get_f64(i).0 > 3.5 {
                // within ~1.5 voxels of the axis (the discrete path may cut a little off the exact middle)
                assert!((x - 32.0).abs() < 1.6 && (y - 32.0).abs() < 1.6, "stem point off axis: {x},{y},{z}");
                stem += 1;
            }
        }
        assert!(stem > 10, "stem points {stem}");
        // tree structure: one root branch, the other attaches to it; arc length counts from the root along the tree
        let info = j["branch_info"].as_array().unwrap();
        assert_eq!(info.len(), 2);
        assert_eq!(info[0]["parent"], -1);
        assert_eq!(info[1]["parent"], 0);
        let s = out[0].get("arclength").and_then(|p| p.as_tensor()).unwrap();
        let smax = (0..s.numel()).map(|i| s.get_f64(i).0).fold(0.0, f64::max);
        let longest_path = 28.0 + 34.9;       // root tip → junction → farthest tip (the root is one of the tips)
        assert!(smax > 50.0 && smax < 85.0, "max arclength {smax} (≈ {longest_path})");
        assert!((0..s.numel()).any(|i| s.get_f64(i).0 == 0.0), "the root has arclength 0");
        // the root is the thick end of the trunk (stem bottom), and the daughters meet at the true bifurcation, not at a corner-cutting link between the arms
        let at = |i: usize| [pts.get_f64(i * 3).0, pts.get_f64(i * 3 + 1).0, pts.get_f64(i * 3 + 2).0];
        let root_i = (0..s.numel()).find(|&i| s.get_f64(i).0 == 0.0).unwrap();
        assert!(at(root_i)[2] < 9.0, "root at the stem bottom, got {:?}", at(root_i));
        let jp = info[1]["join_point"].as_u64().unwrap() as usize;
        let jn = at(jp);
        let dj = ((jn[0] - 32.0).powi(2) + (jn[1] - 32.0).powi(2) + (jn[2] - 32.0).powi(2)).sqrt();
        assert!(dj < 4.0, "junction at {jn:?}, {dj:.1} voxels from the bifurcation");
        // arms are straight: points above the junction lie close to one of the two arm axes
        let dist_seg = |p: [f64; 3], a: [f64; 3], b: [f64; 3]| {
            let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1] + (p[2] - a[2]) * d[2]) / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).clamp(0.0, 1.0);
            ((a[0] + t * d[0] - p[0]).powi(2) + (a[1] + t * d[1] - p[1]).powi(2) + (a[2] + t * d[2] - p[2]).powi(2)).sqrt()
        };
        let mut arm_pts = 0;
        for i in 0..pts.shape()[0] {
            let p = at(i);
            if p[2] > 38.0 && p[2] < 54.0 {
                let d = dist_seg(p, [32.0, 32.0, 32.0], [12.0, 20.0, 58.0]).min(dist_seg(p, [32.0, 32.0, 32.0], [52.0, 44.0, 58.0]));
                assert!(d < 1.8, "arm point {p:?} is {d:.1} voxels from the arm axis");
                arm_pts += 1;
            }
        }
        assert!(arm_pts > 12, "arm points {arm_pts}");
    }


    fn ball(d: &mut [u8], n: usize, c: [f64; 3], r0: f64, r1: f64) {
        for q in 0..d.len() {
            let p = [(q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64];
            let r = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt();
            if r >= r0 && r <= r1 {
                d[q] = 1;
            }
        }
    }

    fn mesh_of(reg: &Registry, ctx: &Ctx, d: Vec<u8>, n: usize, extra: &str) -> Arc<Value> {
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[n, n, n], &d).unwrap()));
        let o = format!(r#"{{"spacing_mm":[1,1,1],"blur":2,"smooth":4{extra}}}"#);
        Arc::new(reg.run("mesh_from_mask", ctx, &o, &[t]).unwrap().1.remove(0))
    }

    /// timing on a large surface: `cargo test -p hasty-server --lib --release big_mesh -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn big_mesh_timing() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 200usize;
        let mut d = vec![0u8; n * n * n];
        ball(&mut d, n, [99.5; 3], 0.0, 90.0);
        let t0 = std::time::Instant::now();
        let m = mesh_of(&reg, &ctx, d, n, "");
        let faces = Mesh::from_value(&m).unwrap().faces.len();
        eprintln!("mask -> mesh: {faces} triangles in {:.1} s", t0.elapsed().as_secs_f64());
        let t1 = std::time::Instant::now();
        let (_, out) = reg.run("mesh_decimate", &ctx, r#"{"ratio":0.1}"#, &[m.clone()]).unwrap();
        eprintln!("decimate to 10 %: {} triangles in {:.1} s", Mesh::from_value(&out[0]).unwrap().faces.len(), t1.elapsed().as_secs_f64());
        let t2 = std::time::Instant::now();
        let o = r#"{"normal":[0,0,1],"cap":true}"#;
        let (_, c) = reg.run("mesh_cut", &ctx, o, &[m.clone()]).unwrap();
        eprintln!("cut with cap: {} triangles in {:.1} s", Mesh::from_value(&c[0]).unwrap().faces.len(), t2.elapsed().as_secs_f64());
        let t3 = std::time::Instant::now();
        let _ = reg.run("mesh_voxelize", &ctx, "{}", &[m]).unwrap();
        eprintln!("voxelize: {:.1} s", t3.elapsed().as_secs_f64());
    }


    fn y_tree(n: usize) -> Vec<u8> {
        let mut d = vec![0u8; n * n * n];
        let seg = |d: &mut Vec<u8>, a: [f64; 3], b: [f64; 3], r: f64| {
            let dv = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let l2 = dv.iter().map(|x| x * x).sum::<f64>();
            for q in 0..d.len() {
                let p = [(q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64];
                let t = (((p[0] - a[0]) * dv[0] + (p[1] - a[1]) * dv[1] + (p[2] - a[2]) * dv[2]) / l2).clamp(0.0, 1.0);
                let e = [a[0] + t * dv[0] - p[0], a[1] + t * dv[1] - p[1], a[2] + t * dv[2] - p[2]];
                if e.iter().map(|x| x * x).sum::<f64>() <= r * r {
                    d[q] = 1;
                }
            }
        };
        seg(&mut d, [32.0, 32.0, 4.0], [32.0, 32.0, 32.0], 5.0);
        seg(&mut d, [32.0, 32.0, 32.0], [12.0, 20.0, 58.0], 3.5);
        seg(&mut d, [32.0, 32.0, 32.0], [52.0, 44.0, 58.0], 3.5);
        d
    }

    #[test]
    fn measurements_on_the_y_tree() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 64usize;
        let mask = Arc::new(Value::Tensor(Tensor::from_slice(&[n, n, n], &y_tree(n)).unwrap()));
        let (_, cl) = reg.run("centerline_from_mask", &ctx, r#"{"spacing_mm":[1,1,1]}"#, &[mask.clone()]).unwrap();
        let lines = Arc::new(cl[0].clone());
        let (msg, out) = reg.run("centerline_measure", &ctx, r#"{"spacing_mm":[1,1,1]}"#, &[lines, mask]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        let br = j["branches"].as_array().unwrap();
        assert_eq!(br.len(), 2);
        // equivalent diameters: the discrete tubes of radius 5 (trunk) and 3.5 (arms); branch 0 runs arm → trunk, so look at the arm part through branch 1
        let arm = br[1]["mean_eq_diameter_mm"].as_f64().unwrap();
        assert!((arm - 7.0).abs() < 1.2, "arm equivalent diameter {arm}");
        let segs = j["segments"].as_array().unwrap();
        assert_eq!(segs.len(), 3, "trunk, arm A and arm B: {segs:?}");
        for sg in segs {
            let t = sg["tortuosity"].as_f64().unwrap();
            assert!(t > 0.99 && t < 1.12, "straight segment tortuosity {t}: {sg}");
        }
        // the trunk part: points of branch 0 near the stem
        let pts = out[0].get("points").and_then(|t| t.as_tensor()).unwrap();
        let eq = out[0].get("eq_diameter_mm").and_then(|t| t.as_tensor()).unwrap();
        let stem: Vec<f64> = (0..pts.shape()[0]).filter(|&i| pts.get_f64(i * 3 + 2).0 > 8.0 && pts.get_f64(i * 3 + 2).0 < 27.0).map(|i| eq.get_f64(i).0).collect();
        let m = stem.iter().sum::<f64>() / stem.len() as f64;
        assert!(stem.len() > 8 && (m - 10.0).abs() < 1.5, "trunk equivalent diameter {m} over {} points", stem.len());
        // arms leave the junction at ±(20,12,26): the angle between them is about 84°; each makes about 35° with the trunk axis
        let junction = &j["junctions"].as_array().unwrap()[0];
        let ab = junction["angle_between_daughters_deg"].as_f64().unwrap();
        assert!((ab - 84.0).abs() < 14.0, "angle between the daughters {ab}");
        let dt = junction["daughter_to_trunk_deg"].as_f64().unwrap();
        assert!((dt - 42.0).abs() < 14.0, "daughter to trunk {dt}");
    }

    #[test]
    fn fair_fill_adds_interior_vertices_and_stays_closed() {
        let (reg, ctx, m) = sphere_mesh();
        let full = Mesh::from_value(&m).unwrap();
        let mut open = full.clone();
        let c = open.verts[open.verts.len() / 3];
        open.faces.retain(|f| {
            let p = f.map(|i| open.verts[i as usize]);
            let m = [0, 1, 2].map(|a| (p[0][a] + p[1][a] + p[2][a]) / 3.0 - c[a]);
            m[0] * m[0] + m[1] * m[1] + m[2] * m[2] > 64.0           // an 8 mm hole
        });
        let ov = Arc::new(open.to_value());
        let (_, plain) = reg.run("mesh_fill_holes", &ctx, r#"{"fair":false,"max_edges":500}"#, &[ov.clone()]).unwrap();
        let (_, fair) = reg.run("mesh_fill_holes", &ctx, r#"{"max_edges":500}"#, &[ov]).unwrap();
        let (p, f) = (Mesh::from_value(&plain[0]).unwrap(), Mesh::from_value(&fair[0]).unwrap());
        assert_eq!(f.boundary_edges(), 0);
        assert!(f.verts.len() > p.verts.len() + 5, "interior vertices added: {} vs {}", f.verts.len(), p.verts.len());
        assert!((f.volume() - full.volume()).abs() / full.volume() < 0.03);
        // the patch triangles are no longer long slivers across the hole: longest edge of the added triangles
        let longest = |m: &Mesh, from: usize| m.faces[from..].iter().flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3]))).map(|(a, b)| { let (x, y) = (m.verts[a as usize], m.verts[b as usize]); (((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)) as f64).sqrt() }).fold(0.0, f64::max);
        assert!(longest(&f, open.faces.len()) < longest(&p, open.faces.len()), "fairing shortens the longest patch edge");
    }

    #[test]
    fn tracking_follows_large_motion() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 48usize;
        let mut d = Vec::new();
        let frames = [([14.0, 24.0, 24.0], 7.0), ([20.0, 24.0, 24.0], 8.0), ([27.0, 24.0, 24.0], 9.0), ([33.0, 25.0, 24.0], 8.0)];   // moves 6–7 voxels per frame
        for (c, r) in frames {
            let mut f = vec![0u8; n * n * n];
            ball(&mut f, n, c, 0.0, r);
            d.extend(f);
        }
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[4, n, n, n], &d).unwrap()));
        let (_, out) = reg.run("meshseq_track", &ctx, r#"{"spacing_mm":[1,1,1]}"#, &[t]).unwrap();
        let Some(Value::List(fr)) = out[0].get("frames") else { panic!("frames") };
        let ms: Vec<Mesh> = fr.iter().map(|v| Mesh::from_value(v).unwrap()).collect();
        for (m, (c, r)) in ms.iter().zip(frames) {
            let cen = m.verts.iter().fold([0f64; 3], |s, v| [s[0] + v[0] as f64, s[1] + v[1] as f64, s[2] + v[2] as f64]);
            let cen = [cen[0] / m.verts.len() as f64, cen[1] / m.verts.len() as f64, cen[2] / m.verts.len() as f64];
            assert!((cen[0] - c[0]).abs() < 0.8 && (cen[1] - c[1]).abs() < 0.8, "centroid {cen:?} vs {c:?}");
            let mr = m.verts.iter().map(|v| ((v[0] as f64 - c[0]).powi(2) + (v[1] as f64 - c[1]).powi(2) + (v[2] as f64 - c[2]).powi(2)).sqrt()).sum::<f64>() / m.verts.len() as f64;
            assert!((mr - r).abs() < 1.0, "radius {mr} vs {r}");
            assert_eq!(m.faces, ms[0].faces);
        }
    }

    #[test]
    fn decimation_keeps_the_extremes_of_a_scalar() {
        let (reg, ctx, m) = sphere_mesh();
        // a scalar with one narrow spike: without protection the spike vertex is merged into a flatter neighbour
        let mut mm = Mesh::from_value(&m).unwrap();
        let c = mm.verts[mm.verts.len() / 2];
        let s: Vec<f32> = mm.verts.iter().map(|v| { let d2 = (v[0] - c[0]).powi(2) + (v[1] - c[1]).powi(2) + (v[2] - c[2]).powi(2); 0.1 * v[2] + 5.0 * (-d2 / 0.5).exp() }).collect();
        let peak = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        mm.scalars.push(("s".into(), s));
        let mv = Arc::new(mm.to_value());
        let max_of = |v: &Value| Mesh::from_value(v).unwrap().scalars[0].1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let (_, kept) = reg.run("mesh_decimate", &ctx, r#"{"ratio":0.05}"#, &[mv.clone()]).unwrap();
        let (_, lost) = reg.run("mesh_decimate", &ctx, r#"{"ratio":0.05,"preserve_scalars":false}"#, &[mv]).unwrap();
        assert!(max_of(&kept[0]) >= peak - 1e-4, "preserved peak {} vs {peak}", max_of(&kept[0]));
        assert!(max_of(&lost[0]) < peak - 0.2, "unprotected decimation should flatten the spike ({} vs {peak})", max_of(&lost[0]));
        assert_eq!(Mesh::from_value(&kept[0]).unwrap().boundary_edges(), 0);
    }

    #[test]
    fn tracking_starts_a_new_surface_when_the_topology_changes() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 48usize;
        let mut d = Vec::new();
        // one ball; then it splits into two; the two grow; the two shrink
        let frames: [(Vec<([f64; 3], f64)>, ()); 4] = [
            (vec![([24.0, 24.0, 24.0], 8.0)], ()),
            (vec![([15.0, 24.0, 24.0], 5.0), ([33.0, 24.0, 24.0], 5.0)], ()),
            (vec![([15.0, 24.0, 24.0], 6.0), ([33.0, 24.0, 24.0], 6.0)], ()),
            (vec![([15.0, 24.0, 24.0], 5.0), ([33.0, 24.0, 24.0], 5.0)], ()),
        ];
        for (balls, _) in &frames {
            let mut f = vec![0u8; n * n * n];
            for (c, r) in balls {
                ball(&mut f, n, *c, 0.0, *r);
            }
            d.extend(f);
        }
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[4, n, n, n], &d).unwrap()));
        let (msg, out) = reg.run("meshseq_track", &ctx, r#"{"spacing_mm":[1,1,1],"keep":"all"}"#, &[t]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        assert_eq!(j["links"], json!([false, true, true]), "{msg}");
        assert!(matches!(out[0].get("coherent"), Some(Value::Bool(false))));
        let Some(Value::List(fr)) = out[0].get("frames") else { panic!("frames") };
        let ms: Vec<Mesh> = fr.iter().map(|v| Mesh::from_value(v).unwrap()).collect();
        assert_eq!(ms[0].components().1, 1);
        assert_eq!(ms[1].components().1, 2);
        // the split frame knows where each vertex came from: its left piece descends from the left side of the ball, the right piece from the right side
        let par = fr[1].get("parent").and_then(|t| t.as_tensor()).expect("parent map on the new run");
        assert_eq!(par.numel(), ms[1].verts.len());
        assert!(fr[0].get("parent").is_none() && fr[2].get("parent").is_none());
        let wrong = (0..ms[1].verts.len()).filter(|&i| { let from = ms[0].verts[par.get_f64(i).0 as usize]; (ms[1].verts[i][0] < 24.0) != (from[0] < 24.0) }).count();
        assert!((wrong as f64) < 0.02 * ms[1].verts.len() as f64, "{wrong} of {} vertices descend from the wrong side", ms[1].verts.len());
        assert_eq!(ms[1].faces, ms[2].faces);
        assert_eq!(ms[2].faces, ms[3].faces);
        assert!(ms[2].area() > ms[1].area() && ms[3].area() < ms[2].area());
    }

    #[test]
    fn cotangent_fairing_gives_a_smaller_patch_than_uniform_weights() {
        let (reg, ctx, m) = sphere_mesh();
        let mut open = Mesh::from_value(&m).unwrap();
        let c = open.verts[open.verts.len() / 3];
        open.faces.retain(|f| {
            let p = f.map(|i| open.verts[i as usize]);
            let m = [0, 1, 2].map(|a| (p[0][a] + p[1][a] + p[2][a]) / 3.0 - c[a]);
            m[0] * m[0] + m[1] * m[1] + m[2] * m[2] > 64.0
        });
        let ov = Arc::new(open.to_value());
        let base = open.faces.len();
        let patch_area = |m: &Mesh| -> f64 { m.faces[base..].iter().map(|f| { let [a, b, c] = f.map(|i| { let v = m.verts[i as usize]; [v[0] as f64, v[1] as f64, v[2] as f64] }); let w = cross3(sub3(b, a), sub3(c, a)); 0.5 * dot3(w, w).sqrt() }).sum() };
        let (_, cot) = reg.run("mesh_fill_holes", &ctx, r#"{"max_edges":500,"cotan":true}"#, &[ov.clone()]).unwrap();
        let (_, uni) = reg.run("mesh_fill_holes", &ctx, r#"{"max_edges":500,"cotan":false}"#, &[ov]).unwrap();
        let (mc, mu) = (Mesh::from_value(&cot[0]).unwrap(), Mesh::from_value(&uni[0]).unwrap());
        assert_eq!(mc.boundary_edges(), 0);
        // the cotangent relaxation is the discrete minimal-surface flow: its patch cannot be larger than the uniform one (up to the refinement difference)
        assert!(patch_area(&mc) <= patch_area(&mu) * 1.01, "patch areas: cotan {} uniform {}", patch_area(&mc), patch_area(&mu));
    }

    #[test]
    fn boolean_crease_vertices_lie_on_the_exact_intersection_curve() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 48usize;
        let (mut a, mut b) = (vec![0u8; n * n * n], vec![0u8; n * n * n]);
        ball(&mut a, n, [16.0, 24.0, 24.0], 0.0, 10.0);
        ball(&mut b, n, [24.0, 24.0, 24.0], 0.0, 10.0);
        let (ma, mb) = (mesh_of(&reg, &ctx, a, n, ""), mesh_of(&reg, &ctx, b, n, ""));
        let (msg, out) = reg.run("mesh_boolean", &ctx, r#"{"op":"intersection","voxel_mm":0.5}"#, &[ma.clone(), mb.clone()]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        assert!(j["crease_vertices"].as_u64().unwrap() > 40, "{msg}");
        let (am, bm) = (Mesh::from_value(&ma).unwrap(), Mesh::from_value(&mb).unwrap());
        let segs = intersection_segments(&am, &bm, 1.0);
        assert!(segs.len() > 50);
        let r = Mesh::from_value(&out[0]).unwrap();
        let on_curve = r.verts.iter().filter(|v| {
            let p = [v[0] as f64, v[1] as f64, v[2] as f64];
            segs.iter().any(|s| { let q = closest_on_segment(p, s[0], s[1]); dot3(sub3(p, q), sub3(p, q)).sqrt() < 1e-3 })
        }).count();
        assert!(on_curve > 40, "{on_curve} result vertices on the exact curve");
        assert_eq!(r.boundary_edges(), 0);
    }

    #[test]
    fn scalar_aware_decimation_keeps_a_sharp_scalar_edge_where_it_was() {
        let (reg, ctx, m) = sphere_mesh();
        let mut mm = Mesh::from_value(&m).unwrap();
        let s: Vec<f32> = mm.verts.iter().map(|v| if v[2] > 24.0 + 0.5 * (v[0] - 24.0).sin() { 1.0 } else { 0.0 }).collect();    // a wavy step across the sphere
        let orig = mm.verts.clone();
        let orig_s = s.clone();
        mm.scalars.push(("s".into(), s));
        let mv = Arc::new(mm.to_value());
        let mismatch = |opts: &str| -> f64 {
            let (_, out) = reg.run("mesh_decimate", &ctx, opts, &[mv.clone()]).unwrap();
            let d = Mesh::from_value(&out[0]).unwrap();
            let ds = &d.scalars[0].1;
            let bad = d.verts.iter().enumerate().filter(|(i, v)| {
                let near = (0..orig.len()).min_by(|&a, &b| {
                    let (da, db) = ((orig[a][0] - v[0]).powi(2) + (orig[a][1] - v[1]).powi(2) + (orig[a][2] - v[2]).powi(2), (orig[b][0] - v[0]).powi(2) + (orig[b][1] - v[1]).powi(2) + (orig[b][2] - v[2]).powi(2));
                    da.total_cmp(&db)
                }).unwrap();
                orig_s[near] != ds[*i]
            }).count();
            bad as f64 / d.verts.len() as f64
        };
        let (without, with) = (mismatch(r#"{"ratio":0.04,"scalar_weight":0,"preserve_scalars":false}"#), mismatch(r#"{"ratio":0.04,"scalar_weight":2,"preserve_scalars":false}"#));
        eprintln!("scalar mismatch without {without:.3}, with {with:.3}");
        assert!(with <= without, "scalar-aware cost should not misplace the edge more: {with} vs {without}");
    }

    #[test]
    fn centerline_of_a_straight_tube_stays_on_the_axis_up_to_the_ends() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let (nx, ny, nz) = (24usize, 24usize, 80usize);
        let d: Vec<u8> = (0..nx * ny * nz).map(|q| { let (i, j) = ((q % nx) as f64 - 11.5, ((q / nx) % ny) as f64 - 11.5); (i * i + j * j <= 36.0) as u8 }).collect();
        let mask = Arc::new(Value::Tensor(Tensor::from_slice(&[nz, ny, nx], &d).unwrap()));
        let (_, out) = reg.run("centerline_from_mask", &ctx, r#"{"spacing_mm":[1,1,1],"coordinates":"voxel"}"#, &[mask]).unwrap();
        let pts = out[0].get("points").and_then(|p| p.as_tensor()).unwrap();
        let n = pts.shape()[0];
        let off: Vec<(f64, f64, f64)> = (0..n).map(|i| (pts.get_f64(i * 3).0 - 11.5, pts.get_f64(i * 3 + 1).0 - 11.5, pts.get_f64(i * 3 + 2).0)).collect();
        let worst = off.iter().map(|&(x, y, _)| x.hypot(y)).fold(0.0, f64::max);
        let far: Vec<String> = off.iter().filter(|p| p.0.hypot(p.1) > 0.9).map(|p| format!("z{:.0}:{:.2}", p.2, p.0.hypot(p.1))).collect();
        let ends: Vec<String> = off.iter().take(5).chain(off.iter().skip(n - 5)).map(|p| format!("({:.2},{:.2},{:.1})", p.0, p.1, p.2)).collect();
        assert!(worst < 1.3, "worst distance from the axis {worst}: {} | ends {}", far.join(" "), ends.join(" "));
        let (zmin, zmax) = (off.iter().map(|p| p.2).fold(f64::INFINITY, f64::min), off.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max));
        assert!(zmin < 2.0 && zmax > 77.0, "the line should reach the ends of the tube: {zmin} .. {zmax}");
    }

    #[test]
    fn holes_are_filled() {
        let (reg, ctx, m) = sphere_mesh();
        let full = Mesh::from_value(&m).unwrap();
        let mut open = full.clone();
        let c = open.verts[open.verts.len() / 3];
        open.faces.retain(|f| {
            let p = f.map(|i| open.verts[i as usize]);
            let m = [0, 1, 2].map(|a| (p[0][a] + p[1][a] + p[2][a]) / 3.0 - c[a]);
            m[0] * m[0] + m[1] * m[1] + m[2] * m[2] > 16.0
        });
        assert!(open.boundary_edges() > 0);
        let (_, out) = reg.run("mesh_fill_holes", &ctx, "{}", &[Arc::new(open.to_value())]).unwrap();
        let f = Mesh::from_value(&out[0]).unwrap();
        assert_eq!(f.boundary_edges(), 0, "closed again");
        assert!((f.volume() - full.volume()).abs() / full.volume() < 0.03, "volume {} vs {}", f.volume(), full.volume());
        assert!((f.area() - full.area()).abs() / full.area() < 0.03);
    }

    #[test]
    fn cut_cap_handles_a_cavity() {
        // hollow ball: inner radius 8, outer 16 (a cavity: the outer and the inner surface); the capped half is an annular cap
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 44usize;
        let mut d = vec![0u8; n * n * n];
        ball(&mut d, n, [21.5; 3], 8.0, 16.0);
        let m = mesh_of(&reg, &ctx, d, n, r#","keep":"all""#);
        let m0 = Mesh::from_value(&m).unwrap();
        let o = json!({"origin": [21.5, 21.5, 21.5], "normal": [0, 0, 1], "cap": true}).to_string();
        let (_, out) = reg.run("mesh_cut", &ctx, &o, &[m]).unwrap();
        let h = Mesh::from_value(&out[0]).unwrap();
        assert_eq!(h.boundary_edges(), 0, "annular cap closes the cut");
        let want = 2.0 / 3.0 * std::f64::consts::PI * (16f64.powi(3) - 8f64.powi(3));
        assert!(h.volume() > 0.0 && (h.volume() - want).abs() / want < 0.06, "half-shell volume {} vs {want} (full {})", h.volume(), m0.volume());
    }

    #[test]
    fn boolean_volumes_match_the_lens_formula() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 48usize;
        let (mut a, mut b) = (vec![0u8; n * n * n], vec![0u8; n * n * n]);
        ball(&mut a, n, [16.0, 24.0, 24.0], 0.0, 10.0);
        ball(&mut b, n, [24.0, 24.0, 24.0], 0.0, 10.0);
        let (ma, mb) = (mesh_of(&reg, &ctx, a, n, ""), mesh_of(&reg, &ctx, b, n, ""));
        let sphere = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
        let lens = std::f64::consts::PI * (40.0 + 8.0) * (20.0 - 8.0f64).powi(2) / 12.0;
        let mut vols: HashMap<&str, f64> = HashMap::new();
        for (op, want) in [("union", 2.0 * sphere - lens), ("intersection", lens), ("difference", sphere - lens)] {
            let o = format!(r#"{{"op":"{op}","voxel_mm":0.5}}"#);
            let (_, out) = reg.run("mesh_boolean", &ctx, &o, &[ma.clone(), mb.clone()]).unwrap();
            let r = Mesh::from_value(&out[0]).unwrap();
            assert_eq!(r.boundary_edges(), 0, "{op}: closed");
            vols.insert(op, r.volume());
            assert!((r.volume() - want).abs() / want < 0.12, "{op}: volume {} vs {want}", r.volume());
        }
        // the identities hold whatever the shrinkage of the two input surfaces: |A∪B| + |A∩B| = |A| + |B|, |A∖B| + |A∩B| = |A|
        let (va, vb) = (Mesh::from_value(&ma).unwrap().volume(), Mesh::from_value(&mb).unwrap().volume());
        assert!((vols["union"] + vols["intersection"] - va - vb).abs() / (va + vb) < 0.03, "inclusion-exclusion {vols:?} {va} {vb}");
        assert!((vols["difference"] + vols["intersection"] - va).abs() / va < 0.03, "difference {vols:?} {va}");
        // refinement: the result's vertices lie on the input surfaces (within 1.5 voxels they are moved onto them)
        let (_, rb) = reg.run("mesh_boolean", &ctx, r#"{"op":"union","voxel_mm":0.5}"#, &[ma.clone(), mb.clone()]).unwrap();
        let (rm, am, bm) = (Mesh::from_value(&rb[0]).unwrap(), Mesh::from_value(&ma).unwrap(), Mesh::from_value(&mb).unwrap());
        let grids = [TriGrid::new(&am, 1.0), TriGrid::new(&bm, 1.0)];
        let on_surface = rm.verts.iter().filter(|v| grids.iter().filter_map(|g| g.closest([v[0] as f64, v[1] as f64, v[2] as f64], 0.05)).next().is_some()).count();
        assert!(on_surface as f64 > 0.9 * rm.verts.len() as f64, "{on_surface} of {} union vertices lie on an input surface", rm.verts.len());
        // voxelisation of one sphere
        let (msg, vox) = reg.run("mesh_voxelize", &ctx, r#"{"voxel_mm":0.5}"#, &[ma]).unwrap();
        let j: Json = serde_json::from_str(&msg).unwrap();
        let vol = j["voxels"].as_f64().unwrap() * 0.125;
        let own = va;
        assert!((vol - own).abs() / own < 0.03, "voxelised volume {vol} vs the mesh volume {own}");
        assert_eq!(vox[0].as_tensor().unwrap().shape().len(), 3);
    }

    #[test]
    fn tracked_sequence_keeps_one_topology_and_follows_the_wall() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 40usize;
        let mut d = Vec::new();
        let radii = [6.0, 8.0, 10.0, 7.0];
        for &r in &radii {
            let mut f = vec![0u8; n * n * n];
            ball(&mut f, n, [19.5; 3], 0.0, r);
            d.extend(f);
        }
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[4, n, n, n], &d).unwrap()));
        let (_, out) = reg.run("meshseq_track", &ctx, r#"{"spacing_mm":[1,1,1],"dt":0.05}"#, &[t]).unwrap();
        assert!(matches!(out[0].get("coherent"), Some(Value::Bool(true))));
        let Some(Value::List(frames)) = out[0].get("frames") else { panic!("frames") };
        let ms: Vec<Mesh> = frames.iter().map(|v| Mesh::from_value(v).unwrap()).collect();
        for (m, &r) in ms.iter().zip(&radii) {
            assert_eq!(m.verts.len(), ms[0].verts.len());
            assert_eq!(m.faces, ms[0].faces, "same triangles in every frame");
            assert_eq!(m.boundary_edges(), 0);
            let mean_r = m.verts.iter().map(|v| (((v[0] as f64 - 19.5).powi(2) + (v[1] as f64 - 19.5).powi(2) + (v[2] as f64 - 19.5).powi(2)).sqrt())).sum::<f64>() / m.verts.len() as f64;
            assert!((mean_r - r).abs() < 0.8, "frame radius {mean_r} vs {r}");
        }
        // vertices keep their identity: a vertex near +x stays near +x
        let v0 = ms[0].verts.iter().enumerate().max_by(|a, b| a.1[0].total_cmp(&b.1[0])).unwrap().0;
        assert!(ms.iter().all(|m| m.verts[v0][0] > 19.5 + 3.0 && (m.verts[v0][1] - 19.5).abs() < 3.5));
    }

    #[test]
    fn mesh_sequence_has_one_surface_per_frame() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let n = 32usize;
        let mut d = Vec::new();
        for f in 0..3 {
            let c = (n as f64 - 1.0) / 2.0;
            let r = 6.0 + 2.0 * f as f64;
            d.extend((0..n * n * n).map(|q| {
                let (i, j, k) = ((q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64);
                (((i - c).powi(2) + (j - c).powi(2) + (k - c).powi(2)).sqrt() <= r) as u8
            }));
        }
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[3, n, n, n], &d).unwrap()));
        let (msg, out) = reg.run("meshseq_from_mask", &ctx, r#"{"spacing_mm":[1,1,1],"dt":0.05}"#, &[t]).unwrap();
        let a: Vec<f64> = serde_json::from_str::<Json>(&msg).unwrap()["area_mm2"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert!(a.len() == 3 && a[0] < a[1] && a[1] < a[2], "{a:?}");
        match out[0].get("frames") {
            Some(Value::List(l)) => assert_eq!(l.len(), 3),
            _ => panic!("frames"),
        }
    }
}
