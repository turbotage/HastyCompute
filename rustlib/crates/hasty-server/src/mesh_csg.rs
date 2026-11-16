//! Exact (to floating point) boolean operations on closed triangle meshes by corefinement: the triangles that the other surface cuts are re-triangulated along the
//! intersection curve, every resulting triangle is classified inside / outside the other mesh by a ray test, and the pieces kept by the operation are welded.
//!
//! For meshes in general position (no coplanar overlapping faces, no vertex exactly on the other surface). Where that fails the result is not closed and the command
//! returns an error (use the voxel `mesh_boolean` then). Vertex scalars are dropped.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hasty_value::Value;
use serde_json::json;

use crate::commands::Registry;
use crate::mesh::{mesh_in, Mesh};
use crate::mesh_fill::{area2, inside_poly, triangulate};
use crate::mesh_ops::{seg_tri, TriGrid};
use crate::Error;

type V3 = [f64; 3];
type P2 = [f64; 2];

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn dist(a: V3, b: V3) -> f64 {
    dot(sub(a, b), sub(a, b)).sqrt()
}

fn p3(m: &Mesh, v: u32) -> V3 {
    let x = m.verts[v as usize];
    [x[0] as f64, x[1] as f64, x[2] as f64]
}

/// the segment where triangle `ta` of `a` meets triangle `tb` of `b` (None when they do not cross)
fn pair_segment(ta: [V3; 3], tb: [V3; 3]) -> Option<[V3; 2]> {
    let mut pts: Vec<V3> = Vec::new();
    for e in 0..3 {
        if let Some(p) = seg_tri(tb[e], tb[(e + 1) % 3], ta[0], ta[1], ta[2]) {
            pts.push(p);
        }
        if let Some(p) = seg_tri(ta[e], ta[(e + 1) % 3], tb[0], tb[1], tb[2]) {
            pts.push(p);
        }
    }
    if pts.len() < 2 {
        return None;
    }
    let (mut best, mut pair) = (0.0, None);
    for x in 0..pts.len() {
        for y in x + 1..pts.len() {
            let d = dot(sub(pts[x], pts[y]), sub(pts[x], pts[y]));
            if d > best {
                best = d;
                pair = Some([pts[x], pts[y]]);
            }
        }
    }
    pair.filter(|_| best > 1e-24)
}

/// all crossing triangle pairs (index in a, index in b, segment)
fn crossings(a: &Mesh, b: &Mesh, cell: f64) -> Vec<(u32, u32, [V3; 2])> {
    let gb = TriGrid::new(b, cell);
    let mut out = Vec::new();
    for (ia, fa) in a.faces.iter().enumerate() {
        let ta = fa.map(|v| p3(a, v));
        let lo = [0, 1, 2].map(|k| (ta[0][k].min(ta[1][k]).min(ta[2][k]) / cell).floor() as i64);
        let hi = [0, 1, 2].map(|k| (ta[0][k].max(ta[1][k]).max(ta[2][k]) / cell).floor() as i64);
        let mut seen: HashSet<u32> = HashSet::new();
        for i in lo[0]..=hi[0] {
            for j in lo[1]..=hi[1] {
                for k in lo[2]..=hi[2] {
                    for &fb in gb.cells.get(&(i, j, k)).into_iter().flatten() {
                        if seen.insert(fb) {
                            let tb = b.faces[fb as usize].map(|v| p3(b, v));
                            if let Some(s) = pair_segment(ta, tb) {
                                out.push((ia as u32, fb, s));
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// points merged within `eps`
struct Registry3 {
    eps: f64,
    cell: f64,
    map: HashMap<(i64, i64, i64), Vec<u32>>,
    pts: Vec<V3>,
}

impl Registry3 {
    fn new(eps: f64) -> Self {
        Registry3 { eps, cell: eps * 4.0, map: HashMap::new(), pts: Vec::new() }
    }
    fn key(&self, p: V3) -> (i64, i64, i64) {
        ((p[0] / self.cell).floor() as i64, (p[1] / self.cell).floor() as i64, (p[2] / self.cell).floor() as i64)
    }
    fn get(&mut self, p: V3) -> u32 {
        let k = self.key(p);
        for i in k.0 - 1..=k.0 + 1 {
            for j in k.1 - 1..=k.1 + 1 {
                for l in k.2 - 1..=k.2 + 1 {
                    for &id in self.map.get(&(i, j, l)).into_iter().flatten() {
                        if dist(self.pts[id as usize], p) <= self.eps {
                            return id;
                        }
                    }
                }
            }
        }
        self.pts.push(p);
        let id = (self.pts.len() - 1) as u32;
        self.map.entry(k).or_default().push(id);
        id
    }
}

/// Re-triangulate the triangles of `m` that the segments cut. Returns the vertex positions (original vertices first, then the new ones) and the triangles.
fn refine(m: &Mesh, segs: &HashMap<u32, Vec<[V3; 2]>>, eps: f64) -> Result<(Vec<V3>, Vec<[u32; 3]>, HashSet<(u32, u32)>), Error> {
    let nv = m.verts.len() as u32;
    let mut reg = Registry3::new(eps);
    let mut new_pos: Vec<V3> = Vec::new();            // positions of the new vertices; global id = nv + registry id
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut crease: HashSet<(u32, u32)> = HashSet::new();
    for (ti, f) in m.faces.iter().enumerate() {
        let Some(sg) = segs.get(&(ti as u32)) else {
            tris.push(*f);
            continue;
        };
        let corners = f.map(|v| p3(m, v));
        // nodes: the corners and the segment ends (global ids; an end that coincides with a corner is that corner)
        let mut ids: Vec<u32> = f.to_vec();
        let mut pos: Vec<V3> = corners.to_vec();
        let node_of = |p: V3, ids: &mut Vec<u32>, pos: &mut Vec<V3>, reg: &mut Registry3, new_pos: &mut Vec<V3>| -> usize {
            for (c, cp) in corners.iter().enumerate() {
                if dist(*cp, p) <= eps {
                    return c;
                }
            }
            let rid = reg.get(p);
            if rid as usize >= new_pos.len() {
                new_pos.push(p);
            }
            let gid = nv + rid;
            if let Some(i) = ids.iter().position(|&x| x == gid) {
                return i;
            }
            ids.push(gid);
            pos.push(p);
            ids.len() - 1
        };
        let mut constraints: Vec<(usize, usize)> = Vec::new();
        for s in sg {
            let (a, b) = (node_of(s[0], &mut ids, &mut pos, &mut reg, &mut new_pos), node_of(s[1], &mut ids, &mut pos, &mut reg, &mut new_pos));
            if a != b && !constraints.contains(&(a, b)) && !constraints.contains(&(b, a)) {
                constraints.push((a, b));
                let (ga, gb) = (ids[a], ids[b]);
                crease.insert((ga.min(gb), ga.max(gb)));
            }
        }
        // planar coordinates (counter-clockwise for the triangle as stored)
        let nrm = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
        let k = (0..3).max_by(|&a, &b| nrm[a].abs().total_cmp(&nrm[b].abs())).unwrap();
        let (iu, iv) = ((k + 1) % 3, (k + 2) % 3);
        let flip = if nrm[k] < 0.0 { -1.0 } else { 1.0 };
        let pts2: Vec<P2> = pos.iter().map(|p| [flip * p[iu], p[iv]]).collect();
        // the triangle's three edges, split at the nodes lying on them
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for e in 0..3 {
            let (a, b) = (e, (e + 1) % 3);
            let (pa, pb) = (corners[a], corners[b]);
            let d = sub(pb, pa);
            let l2 = dot(d, d);
            let mut on: Vec<(f64, usize)> = (3..ids.len()).filter_map(|n| {
                let t = dot(sub(pos[n], pa), d) / l2;
                let q = [pa[0] + t * d[0], pa[1] + t * d[1], pa[2] + t * d[2]];
                (t > 0.0 && t < 1.0 && dist(q, pos[n]) <= eps * 4.0).then_some((t, n))
            }).collect();
            on.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut prev = a;
            for (_, n) in on {
                edges.push((prev, n));
                prev = n;
            }
            edges.push((prev, b));
        }
        for &(a, b) in &constraints {
            // a constraint that lies along a triangle edge is already one of its pieces
            if !edges.contains(&(a, b)) && !edges.contains(&(b, a)) {
                edges.push((a, b));
            }
        }
        // planar faces: neighbours sorted by angle; a face is walked keeping it on the left
        let m_nodes = ids.len();
        let mut nb: Vec<Vec<usize>> = vec![Vec::new(); m_nodes];
        for &(a, b) in &edges {
            nb[a].push(b);
            nb[b].push(a);
        }
        for (n, l) in nb.iter_mut().enumerate() {
            l.sort_by(|&x, &y| {
                let (ax, ay) = ((pts2[x][1] - pts2[n][1]).atan2(pts2[x][0] - pts2[n][0]), (pts2[y][1] - pts2[n][1]).atan2(pts2[y][0] - pts2[n][0]));
                ax.total_cmp(&ay)
            });
        }
        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        let mut cycles: Vec<Vec<usize>> = Vec::new();
        for &(a, b) in &edges {
            for (u0, v0) in [(a, b), (b, a)] {
                if visited.contains(&(u0, v0)) {
                    continue;
                }
                let mut cyc = vec![u0];
                let (mut u, mut v) = (u0, v0);
                loop {
                    visited.insert((u, v));
                    let l = &nb[v];
                    let iu_ = l.iter().position(|&x| x == u).unwrap();
                    let w = l[(iu_ + l.len() - 1) % l.len()];
                    if v == u0 && w == v0 {
                        break;
                    }
                    cyc.push(v);
                    (u, v) = (v, w);
                    if cyc.len() > 4 * m_nodes + 8 {
                        return inv("corefinement: a face of the planar graph did not close (degenerate input)");
                    }
                    if visited.contains(&(u, v)) {
                        break;
                    }
                }
                cycles.push(cyc);
            }
        }
        // counter-clockwise cycles are faces; clockwise ones are the outer boundary (largest) or holes (the boundary of a separate piece of the curve)
        let areas: Vec<f64> = cycles.iter().map(|c| area2(&c.iter().map(|&n| pts2[n]).collect::<Vec<_>>())).collect();
        let outer_cw = (0..cycles.len()).filter(|&i| areas[i] < 0.0).min_by(|&x, &y| areas[x].total_cmp(&areas[y]));
        let faces: Vec<usize> = (0..cycles.len()).filter(|&i| areas[i] > 1e-18).collect();
        let holes: Vec<usize> = (0..cycles.len()).filter(|&i| areas[i] < 0.0 && Some(i) != outer_cw).collect();
        let mut face_holes: HashMap<usize, Vec<usize>> = HashMap::new();
        for &h in &holes {
            let probe = pts2[cycles[h][0]];
            let owner = faces.iter().copied().filter(|&f| inside_poly(probe, &cycles[f].iter().map(|&n| pts2[n]).collect::<Vec<_>>())).min_by(|&x, &y| areas[x].total_cmp(&areas[y]));
            if let Some(o) = owner {
                face_holes.entry(o).or_default().push(h);
            }
        }
        for &fcy in &faces {
            let hs = face_holes.get(&fcy).cloned().unwrap_or_default();
            // one point list for the polygon and its holes
            let mut idx: Vec<usize> = Vec::new();
            let mut pl: Vec<P2> = Vec::new();
            let mut nodes_of: Vec<usize> = Vec::new();
            let ring = |c: &Vec<usize>, idx: &mut Vec<usize>, pl: &mut Vec<P2>, nodes_of: &mut Vec<usize>| -> Vec<usize> {
                let r: Vec<usize> = c.iter().map(|&n| { pl.push(pts2[n]); nodes_of.push(n); pl.len() - 1 }).collect();
                idx.extend(&r);
                r
            };
            let outer = ring(&cycles[fcy], &mut idx, &mut pl, &mut nodes_of);
            let hole_rings: Vec<Vec<usize>> = hs.iter().map(|&h| {
                // holes are walked clockwise already
                ring(&cycles[h], &mut idx, &mut pl, &mut nodes_of)
            }).collect();
            for t in triangulate(&outer, &hole_rings, &pl) {
                let g = t.map(|i| ids[nodes_of[i]]);
                if g[0] != g[1] && g[1] != g[2] && g[0] != g[2] {
                    tris.push(g);
                }
            }
        }
    }
    let mut verts: Vec<V3> = (0..nv).map(|v| p3(m, v)).collect();
    verts.extend(new_pos);
    Ok((verts, tris, crease))
}

/// inside / outside of a closed mesh by the parity of the crossings of a ray along +x (the ray is offset by an irrational amount to avoid hitting edges)
struct Inside<'a> {
    m: &'a Mesh,
    cell: f64,
    grid: HashMap<(i64, i64), Vec<u32>>,
    off: [f64; 2],
}

impl<'a> Inside<'a> {
    fn new(m: &'a Mesh, scale: f64) -> Self {
        let tri_size = m.faces.iter().take(2000).map(|f| { let p = f.map(|v| p3(m, v)); dist(p[0], p[1]).max(dist(p[1], p[2])) }).sum::<f64>() / m.faces.len().clamp(1, 2000) as f64;
        let cell = (2.0 * tri_size).max(scale * 1e-6);
        let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
        for (fi, f) in m.faces.iter().enumerate() {
            let p = f.map(|v| p3(m, v));
            let (y0, y1) = (p.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min), p.iter().map(|q| q[1]).fold(f64::NEG_INFINITY, f64::max));
            let (z0, z1) = (p.iter().map(|q| q[2]).fold(f64::INFINITY, f64::min), p.iter().map(|q| q[2]).fold(f64::NEG_INFINITY, f64::max));
            for j in (y0 / cell).floor() as i64..=(y1 / cell).floor() as i64 {
                for k in (z0 / cell).floor() as i64..=(z1 / cell).floor() as i64 {
                    grid.entry((j, k)).or_default().push(fi as u32);
                }
            }
        }
        Inside { m, cell, grid, off: [scale * 1.2345678e-7, scale * 2.3456789e-7] }
    }

    fn contains(&self, c: V3) -> bool {
        let (y, z) = (c[1] + self.off[0], c[2] + self.off[1]);
        let mut n = 0;
        for &fi in self.grid.get(&((y / self.cell).floor() as i64, (z / self.cell).floor() as i64)).into_iter().flatten() {
            let q = self.m.faces[fi as usize].map(|v| p3(self.m, v));
            let d = (q[1][1] - q[0][1]) * (q[2][2] - q[0][2]) - (q[2][1] - q[0][1]) * (q[1][2] - q[0][2]);
            if d.abs() < 1e-18 {
                continue;
            }
            let u = ((y - q[0][1]) * (q[2][2] - q[0][2]) - (q[2][1] - q[0][1]) * (z - q[0][2])) / d;
            let v = ((q[1][1] - q[0][1]) * (z - q[0][2]) - (y - q[0][1]) * (q[1][2] - q[0][2])) / d;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                let x = q[0][0] + u * (q[1][0] - q[0][0]) + v * (q[2][0] - q[0][0]);
                if x > c[0] {
                    n += 1;
                }
            }
        }
        n % 2 == 1
    }
}

/// exact boolean of two closed meshes: op is union | intersection | difference (a minus b)
pub fn csg(a: &Mesh, b: &Mesh, op: &str) -> Result<Mesh, Error> {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for m in [a, b] {
        for v in &m.verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k] as f64);
                hi[k] = hi[k].max(v[k] as f64);
            }
        }
    }
    let scale = dist(lo, hi).max(1e-9);
    let eps = scale * 1e-9;
    let edge = a.faces.iter().take(500).map(|f| { let p = f.map(|v| p3(a, v)); dist(p[0], p[1]) }).sum::<f64>() / a.faces.len().clamp(1, 500) as f64;
    let cell = (2.0 * edge).max(scale * 1e-4);
    let cr = crossings(a, b, cell);
    let mut sa: HashMap<u32, Vec<[V3; 2]>> = HashMap::new();
    let mut sb: HashMap<u32, Vec<[V3; 2]>> = HashMap::new();
    for (ta, tb, s) in &cr {
        sa.entry(*ta).or_default().push(*s);
        sb.entry(*tb).or_default().push(*s);
    }
    let (va, ta, ca) = refine(a, &sa, eps)?;
    let (vb, tb, cb) = refine(b, &sb, eps)?;
    if std::env::var_os("CSG_DEBUG").is_some() {
        for (name, v, t) in [("A'", &va, &ta), ("B'", &vb, &tb)] {
            let mut w = Registry3::new(eps * 50.0);
            let faces: Vec<[u32; 3]> = t.iter().map(|f| f.map(|i| w.get(v[i as usize]))).collect();
            let verts: Vec<[f32; 3]> = w.pts.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
            let m = Mesh { verts, faces, scalars: vec![], space: "x".into(), affine: None, spacing_mm: None };
            eprintln!("{name}: {} triangles ({} originally), {} open edges, {} crossings", m.faces.len(), if name == "A'" { a.faces.len() } else { b.faces.len() }, m.boundary_edges(), cr.len());
        }
    }
    let (ia, ib) = (Inside::new(b, scale), Inside::new(a, scale));
    let centroid = |v: &Vec<V3>, t: &[u32; 3]| -> V3 { let (p, q, r) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]); [(p[0] + q[0] + r[0]) / 3.0, (p[1] + q[1] + r[1]) / 3.0, (p[2] + q[2] + r[2]) / 3.0] };
    let tri_pos = |v: &Vec<V3>, t: &[u32; 3]| -> [V3; 3] { [v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]] };
    // A surface patch cannot cross the other surface without a crease edge, so every patch (connected through non-crease edges) is on one side: classify it once, by
    // its biggest triangle (a tiny sliver next to the crease has its centroid within rounding of the other surface)
    let classify = |v: &Vec<V3>, t: &Vec<[u32; 3]>, crease: &HashSet<(u32, u32)>, inside: &Inside| -> Vec<bool> {
        let n = t.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut Vec<usize>, mut x: usize) -> usize { while p[x] != x { p[x] = p[p[x]]; x = p[x]; } x }
        let mut by_edge: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, f) in t.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                let e = (a.min(b), a.max(b));
                if crease.contains(&e) {
                    continue;
                }
                if let Some(&j) = by_edge.get(&e) {
                    let (ra, rb) = (find(&mut parent, i), find(&mut parent, j));
                    if ra != rb { parent[ra] = rb; }
                } else {
                    by_edge.insert(e, i);
                }
            }
        }
        let area = |f: &[u32; 3]| { let p = tri_pos(v, f); let c = cross(sub(p[1], p[0]), sub(p[2], p[0])); dot(c, c) };
        let mut best: HashMap<usize, usize> = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            let e = best.entry(r).or_insert(i);
            if area(&t[i]) > area(&t[*e]) { *e = i; }
        }
        let side: HashMap<usize, bool> = best.iter().map(|(&r, &i)| (r, inside.contains(centroid(v, &t[i])))).collect();
        (0..n).map(|i| { let r = find(&mut parent, i); side[&r] }).collect()
    };
    let (side_a, side_b) = (classify(&va, &ta, &ca, &ia), classify(&vb, &tb, &cb, &ib));
    let mut out: Vec<[V3; 3]> = Vec::new();
    for (t, &inside) in ta.iter().zip(&side_a) {
        let keep = match op { "union" => !inside, "intersection" => inside, "difference" => !inside, _ => return inv(format!("mesh_csg: op must be union | intersection | difference, got {op:?}")) };
        if keep {
            out.push(tri_pos(&va, t));
        }
    }
    for (t, &inside) in tb.iter().zip(&side_b) {
        let p = tri_pos(&vb, t);
        match op {
            "union" if !inside => out.push(p),
            "intersection" if inside => out.push(p),
            "difference" if inside => out.push([p[0], p[2], p[1]]),
            _ => {}
        }
    }
    // weld the pieces (the crease vertices were computed twice, once for each mesh)
    let mut reg = Registry3::new(eps * 50.0);
    let mut faces: Vec<[u32; 3]> = Vec::new();
    for t in &out {
        let g = t.map(|p| reg.get(p));
        if g[0] != g[1] && g[1] != g[2] && g[0] != g[2] {
            faces.push(g);
        }
    }
    let verts: Vec<[f32; 3]> = reg.pts.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    let mut m = Mesh { verts, faces, scalars: vec![], space: a.space.clone(), affine: a.affine, spacing_mm: a.spacing_mm };
    // vertices only used by dropped triangles
    let mut used = vec![false; m.verts.len()];
    m.faces.iter().flatten().for_each(|&v| used[v as usize] = true);
    m.retain(&used);
    if m.faces.is_empty() {
        return inv("mesh_csg: the result is empty");
    }
    if m.boundary_edges() > 0 {
        return inv(format!("mesh_csg: the result has {} open edges (coplanar faces or a vertex on the other surface?): use the voxel mesh_boolean instead", m.boundary_edges()));
    }
    Ok(m)
}

pub fn register(r: &mut Registry) {
    r.register(
        "mesh_csg",
        "exact boolean of two closed meshes by corefinement: inputs [A, B]; options {op: union | intersection | difference (A minus B)}; the cut triangles are re-triangulated along the intersection curve, so the crease is exact and the result is watertight when the meshes are in general position (no coplanar overlapping faces, no vertex on the other surface); otherwise an error (use mesh_boolean). Scalars are dropped; message {op, vertices, faces, volume_mm3, crossing_pairs}",
        |_, o, inp: &[Arc<Value>]| {
            let (a, b) = (mesh_in("mesh_csg", inp, 0)?, mesh_in("mesh_csg", inp, 1)?);
            let op = o["op"].as_str().unwrap_or("union");
            let m = csg(&a, &b, op)?;
            Ok((json!({"op": op, "vertices": m.verts.len(), "faces": m.faces.len(), "volume_mm3": m.volume(), "area_mm2": m.area()}).to_string(), vec![m.to_value()]))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Ctx;
    use hasty_value::Tensor;

    fn sphere(n: usize, c: [f64; 3], r: f64, reg: &Registry, ctx: &Ctx) -> Arc<Value> {
        let d: Vec<u8> = (0..n * n * n).map(|q| {
            let p = [(q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64];
            (((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() <= r) as u8
        }).collect();
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[n, n, n], &d).unwrap()));
        Arc::new(reg.run("mesh_from_mask", ctx, r#"{"spacing_mm":[1,1,1],"blur":2,"smooth":4}"#, &[t]).unwrap().1.remove(0))
    }

    #[test]
    fn exact_csg_of_two_spheres_closes_and_obeys_the_identities() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let (ma, mb) = (sphere(48, [16.0, 24.0, 24.0], 10.0, &reg, &ctx), sphere(48, [24.0, 24.0, 24.0], 10.0, &reg, &ctx));
        let (a, b) = (Mesh::from_value(&ma).unwrap(), Mesh::from_value(&mb).unwrap());
        let (va, vb) = (a.volume(), b.volume());
        let (u, i, d) = (csg(&a, &b, "union").unwrap(), csg(&a, &b, "intersection").unwrap(), csg(&a, &b, "difference").unwrap());
        for (n, m) in [("union", &u), ("intersection", &i), ("difference", &d)] {
            assert_eq!(m.boundary_edges(), 0, "{n} is watertight");
            assert!(m.volume() > 0.0, "{n} windings point outward");
        }
        // exact identities (to the precision of the volumes, not of a voxel grid)
        assert!((u.volume() + i.volume() - va - vb).abs() / (va + vb) < 1e-4, "|A∪B| + |A∩B| = |A| + |B|: {} vs {}", u.volume() + i.volume(), va + vb);
        assert!((d.volume() + i.volume() - va).abs() / va < 1e-4, "|A∖B| + |A∩B| = |A|");
        // the crease is exact: its vertices lie on both surfaces
        let ga = TriGrid::new(&a, 1.0);
        let gb = TriGrid::new(&b, 1.0);
        let on_both = i.verts.iter().filter(|v| { let p = [v[0] as f64, v[1] as f64, v[2] as f64]; ga.closest(p, 1e-3).is_some() && gb.closest(p, 1e-3).is_some() }).count();
        assert!(on_both > 40, "{on_both} intersection vertices on both input surfaces");
        // and it agrees with the voxel boolean up to the voxel size
        let (_, vox) = reg.run("mesh_boolean", &ctx, r#"{"op":"intersection","voxel_mm":0.5}"#, &[ma, mb]).unwrap();
        let vv = Mesh::from_value(&vox[0]).unwrap().volume();
        assert!((vv - i.volume()).abs() / i.volume() < 0.06, "voxel {vv} vs exact {}", i.volume());
    }

    #[test]
    fn csg_of_disjoint_and_nested_meshes() {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let big = Mesh::from_value(&sphere(48, [24.0, 24.0, 24.0], 14.0, &reg, &ctx)).unwrap();
        let small = Mesh::from_value(&sphere(48, [24.0, 24.0, 24.0], 6.0, &reg, &ctx)).unwrap();
        let (vb, vs) = (big.volume(), small.volume());
        assert!((csg(&big, &small, "union").unwrap().volume() - vb).abs() / vb < 1e-6);
        assert!((csg(&big, &small, "intersection").unwrap().volume() - vs).abs() / vs < 1e-6);
        let ring = csg(&big, &small, "difference").unwrap();       // a hollow ball: an outer and an inner (flipped) shell
        assert_eq!(ring.boundary_edges(), 0);
        assert!((ring.volume() - (vb - vs)).abs() / vb < 1e-6);
    }
}
