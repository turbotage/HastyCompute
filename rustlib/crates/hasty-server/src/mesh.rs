//! Triangle meshes on the server: iso-surfaces of masks (vessel trees, segmentations), smoothing, statistics and sampling
//! of volumes onto the surface, for the geometry viewer (`plotting_website/mesh`) and the node editor's Geometry nodes.
//!
//! **Mesh value** (a dict):
//! - `"kind": "mesh"`
//! - `"vertices"`: f32 `[N, 3]`, millimetres. `"space"`: `"ras"` (voxel → RAS through the input's affine), `"voxel_mm"` (voxel index × spacing) or
//!   `"voxel"` (no spacing known).
//! - `"faces"`: u32 `[M, 3]` vertex indices, counter-clockwise seen from outside (normals point out of the solid).
//! - `"normals"`: f32 `[N, 3]` unit vertex normals (area weighted).
//! - `"scalars"` (optional): dict name → f32 `[N]` per-vertex values (NaN = no data), e.g. from `mesh_sample`.
//! - `"affine"` (optional, with space `ras`): the 4×4 (row major, 16 numbers) the vertices went through; `"spacing_mm"` (optional) of the source grid.
//!
//! Iso-surfaces use Kuhn marching tetrahedra (6 tetrahedra per cell sharing one diagonal: small tables, consistent across cells),
//! the same scheme as the viewer's vessel tool (`common/surface.js`).

use std::collections::HashMap;

use hasty_value::{Tensor, Value};
use serde_json::{json, Value as Json};

use crate::commands::Registry;
use crate::Error;

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

#[derive(Clone, Default)]
pub struct Mesh {
    pub verts: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub scalars: Vec<(String, Vec<f32>)>,
    pub space: String,
    pub affine: Option<[f64; 16]>,
    pub spacing_mm: Option<[f64; 3]>,
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Mesh {
    pub fn from_value(v: &Value) -> Result<Mesh, Error> {
        if v.get("kind").and_then(|k| if let Value::Str(s) = k { Some(s.as_str()) } else { None }) != Some("mesh") {
            return inv("not a mesh value (dict with kind \"mesh\")");
        }
        let vt = v.get("vertices").and_then(|t| t.as_tensor()).ok_or_else(|| Error::Invalid("mesh: vertices missing".into()))?;
        let ft = v.get("faces").and_then(|t| t.as_tensor()).ok_or_else(|| Error::Invalid("mesh: faces missing".into()))?;
        if vt.shape().len() != 2 || vt.shape()[1] != 3 || ft.shape().len() != 2 || ft.shape()[1] != 3 {
            return inv("mesh: vertices must be [N, 3] and faces [M, 3]");
        }
        let verts: Vec<[f32; 3]> = (0..vt.shape()[0]).map(|i| [0, 1, 2].map(|c| vt.get_f64(i * 3 + c).0 as f32)).collect();
        let faces: Vec<[u32; 3]> = (0..ft.shape()[0]).map(|i| [0, 1, 2].map(|c| ft.get_f64(i * 3 + c).0 as u32)).collect();
        if faces.iter().flatten().any(|&i| i as usize >= verts.len()) {
            return inv("mesh: a face refers to a vertex that does not exist");
        }
        let mut scalars = Vec::new();
        if let Some(Value::Dict(d)) = v.get("scalars") {
            for (k, t) in d {
                if let Some(t) = t.as_tensor().filter(|t| t.numel() == verts.len()) {
                    scalars.push((k.clone(), (0..t.numel()).map(|i| t.get_f64(i).0 as f32).collect()));
                }
            }
        }
        let space = match v.get("space") {
            Some(Value::Str(s)) => s.clone(),
            _ => "voxel_mm".into(),
        };
        let nums = |k: &str| -> Option<Vec<f64>> { v.get(k).and_then(|t| t.as_tensor()).map(|t| (0..t.numel()).map(|i| t.get_f64(i).0).collect()) };
        let affine = nums("affine").filter(|a| a.len() == 16).map(|a| std::array::from_fn(|i| a[i]));
        let spacing_mm = nums("spacing_mm").filter(|a| a.len() == 3).map(|a| [a[0], a[1], a[2]]);
        Ok(Mesh { verts, faces, scalars, space, affine, spacing_mm })
    }

    pub fn to_value(&self) -> Value {
        let n = self.verts.len();
        let flat: Vec<f32> = self.verts.iter().flatten().copied().collect();
        let ff: Vec<u32> = self.faces.iter().flatten().copied().collect();
        let nf: Vec<f32> = self.normals().iter().flatten().copied().collect();
        let mut d = vec![
            ("kind".to_string(), Value::Str("mesh".into())),
            ("vertices".into(), Value::Tensor(Tensor::from_slice(&[n, 3], &flat).unwrap())),
            ("faces".into(), Value::Tensor(Tensor::from_slice(&[self.faces.len(), 3], &ff).unwrap())),
            ("normals".into(), Value::Tensor(Tensor::from_slice(&[n, 3], &nf).unwrap())),
            ("space".into(), Value::Str(self.space.clone())),
            ("unit".into(), Value::Str("mm".into())),
        ];
        if !self.scalars.is_empty() {
            d.push(("scalars".into(), Value::Dict(self.scalars.iter().map(|(k, s)| (k.clone(), Value::Tensor(Tensor::from_slice(&[n], s).unwrap()))).collect())));
        }
        if let Some(a) = self.affine {
            d.push(("affine".into(), Value::Tensor(Tensor::from_slice(&[16], &a).unwrap())));
        }
        if let Some(s) = self.spacing_mm {
            d.push(("spacing_mm".into(), Value::Tensor(Tensor::from_slice(&[3], &s).unwrap())));
        }
        Value::Dict(d)
    }

    /// area weighted unit vertex normals
    pub fn normals(&self) -> Vec<[f32; 3]> {
        let mut n = vec![[0f32; 3]; self.verts.len()];
        for f in &self.faces {
            let [a, b, c] = f.map(|i| self.verts[i as usize]);
            let w = cross(sub(b, a), sub(c, a));
            for &i in f {
                for k in 0..3 {
                    n[i as usize][k] += w[k];
                }
            }
        }
        for v in &mut n {
            let l = dot(*v, *v).sqrt();
            if l > 0.0 {
                v.iter_mut().for_each(|x| *x /= l);
            }
        }
        n
    }

    pub fn area(&self) -> f64 {
        self.faces.iter().map(|f| {
            let [a, b, c] = f.map(|i| self.verts[i as usize]);
            let w = cross(sub(b, a), sub(c, a));
            0.5 * (dot(w, w) as f64).sqrt()
        }).sum()
    }

    /// signed volume (divergence theorem); meaningful for closed meshes
    pub fn volume(&self) -> f64 {
        self.faces.iter().map(|f| {
            let [a, b, c] = f.map(|i| self.verts[i as usize]);
            dot(a, cross(b, c)) as f64 / 6.0
        }).sum()
    }

    /// edges used by exactly one face (0 for a closed surface)
    pub fn boundary_edges(&self) -> usize {
        let mut m: HashMap<(u32, u32), u32> = HashMap::new();
        for f in &self.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *m.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        m.values().filter(|&&c| c == 1).count()
    }

    /// connected components (by shared vertices): label per vertex and the count
    pub fn components(&self) -> (Vec<u32>, usize) {
        let mut p: Vec<u32> = (0..self.verts.len() as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for f in &self.faces {
            for k in 1..3 {
                let (a, b) = (find(&mut p, f[0]), find(&mut p, f[k]));
                if a != b {
                    p[a as usize] = b;
                }
            }
        }
        let mut ids: HashMap<u32, u32> = HashMap::new();
        let lab: Vec<u32> = (0..self.verts.len() as u32).map(|v| {
            let r = find(&mut p, v);
            let n = ids.len() as u32;
            *ids.entry(r).or_insert(n)
        }).collect();
        (lab, ids.len())
    }

    /// Taubin smoothing (λ|μ passes, little shrinkage): `iters` iterations of a positive then a negative Laplacian step
    pub fn taubin(&mut self, iters: usize, lambda: f32, mu: f32) {
        let n = self.verts.len();
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for f in &self.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                adj[a as usize].push(b);
                adj[b as usize].push(a);
            }
        }
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
        }
        let step = |v: &Vec<[f32; 3]>, w: f32| -> Vec<[f32; 3]> {
            (0..n).map(|i| {
                if adj[i].is_empty() {
                    return v[i];
                }
                let mut m = [0f32; 3];
                for &j in &adj[i] {
                    for k in 0..3 {
                        m[k] += v[j as usize][k];
                    }
                }
                let c = adj[i].len() as f32;
                [0, 1, 2].map(|k| v[i][k] + w * (m[k] / c - v[i][k]))
            }).collect()
        };
        for _ in 0..iters {
            self.verts = step(&self.verts, lambda);
            self.verts = step(&self.verts, mu);
        }
    }

    /// keep the vertices selected by `keep`, drop faces that lose a vertex, renumber
    pub fn retain(&mut self, keep: &[bool]) {
        let mut map = vec![u32::MAX; self.verts.len()];
        let mut nv = Vec::new();
        for (i, v) in self.verts.iter().enumerate() {
            if keep[i] {
                map[i] = nv.len() as u32;
                nv.push(*v);
            }
        }
        self.faces = self.faces.iter().filter(|f| f.iter().all(|&i| keep[i as usize])).map(|f| f.map(|i| map[i as usize])).collect();
        for (_, s) in &mut self.scalars {
            *s = s.iter().enumerate().filter(|(i, _)| keep[*i]).map(|(_, &x)| x).collect();
        }
        self.verts = nv;
    }
}

// ── iso-surface ───────────────────────────────────────────────────────────────────────────────────────────────

const CORNER: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const TETS: [[usize; 4]; 6] = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];

type Corner = (usize, f32, [usize; 3]);

/// vertex on the edge between two grid corners (shared between cells through `cache`)
fn edge_vertex(cache: &mut HashMap<u64, u32>, pos: &mut Vec<[f32; 3]>, iso: f32, a: Corner, b: Corner) -> u32 {
    let (a, b) = if a.0 > b.0 { (b, a) } else { (a, b) };
    let d = [b.2[0] as i64 - a.2[0] as i64, b.2[1] as i64 - a.2[1] as i64, b.2[2] as i64 - a.2[2] as i64];
    let key = a.0 as u64 * 32 + ((d[0] + 1) * 9 + (d[1] + 1) * 3 + (d[2] + 1)) as u64;   // each component in {-1, 0, 1}: 27 directions
    *cache.entry(key).or_insert_with(|| {
        let t = (iso - a.1) / (b.1 - a.1);
        pos.push([0, 1, 2].map(|c| a.2[c] as f32 + t * d[c] as f32));
        (pos.len() - 1) as u32
    })
}

/// add the triangle wound so its normal has a non-negative component along `dir` (inside → outside); degenerate ones are dropped
fn push_tri(faces: &mut Vec<[u32; 3]>, pos: &[[f32; 3]], i0: u32, i1: u32, i2: u32, dir: [f32; 3]) {
    if i0 == i1 || i1 == i2 || i0 == i2 {
        return;
    }
    let (a, b, c) = (pos[i0 as usize], pos[i1 as usize], pos[i2 as usize]);
    if dot(cross(sub(b, a), sub(c, a)), dir) >= 0.0 { faces.push([i0, i1, i2]) } else { faces.push([i0, i2, i1]) }
}

/// Iso-surface of `s` ([Z,Y,X], dims = [nx,ny,nz]) at `iso`, inside = s ≥ iso. Vertices in voxel index coordinates; triangles wound
/// so the geometric normal points from inside to outside. `max_tris` stops runaway surfaces with an error.
pub fn marching_tets(s: &[f32], dims: [usize; 3], iso: f32, max_tris: usize) -> Result<(Vec<[f32; 3]>, Vec<[u32; 3]>), Error> {
    let [nx, ny, nz] = dims;
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let mut cache: HashMap<u64, u32> = HashMap::new();
    let node = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx.saturating_sub(1) {
                let mut cn: [Corner; 8] = [(0usize, 0f32, [0usize; 3]); 8];
                let mut inside = 0;
                for (c, o) in CORNER.iter().enumerate() {
                    let (x, y, z) = (i + o[0], j + o[1], k + o[2]);
                    let n = node(x, y, z);
                    cn[c] = (n, s[n], [x, y, z]);
                    if s[n] >= iso {
                        inside += 1;
                    }
                }
                if inside == 0 || inside == 8 {
                    continue;
                }
                for t in &TETS {
                    let (mut ins, mut out) = (Vec::with_capacity(4), Vec::with_capacity(4));
                    for &c in t {
                        if cn[c].1 >= iso { ins.push(cn[c]) } else { out.push(cn[c]) }
                    }
                    if ins.is_empty() || out.is_empty() {
                        continue;
                    }
                    let mean = |l: &Vec<Corner>| [0, 1, 2].map(|a| l.iter().map(|p| p.2[a] as f32).sum::<f32>() / l.len() as f32);
                    let (mi, mo) = (mean(&ins), mean(&out));
                    let dir = sub(mo, mi);
                    let mut vert = |a: Corner, b: Corner| edge_vertex(&mut cache, &mut pos, iso, a, b);
                    if ins.len() == 1 {
                        let v = [vert(ins[0], out[0]), vert(ins[0], out[1]), vert(ins[0], out[2])];
                        push_tri(&mut faces, &pos, v[0], v[1], v[2], dir);
                    } else if out.len() == 1 {
                        let v = [vert(ins[0], out[0]), vert(ins[1], out[0]), vert(ins[2], out[0])];
                        push_tri(&mut faces, &pos, v[0], v[1], v[2], dir);
                    } else {
                        let q = [vert(ins[0], out[0]), vert(ins[0], out[1]), vert(ins[1], out[1]), vert(ins[1], out[0])];
                        push_tri(&mut faces, &pos, q[0], q[1], q[2], dir);
                        push_tri(&mut faces, &pos, q[0], q[2], q[3], dir);
                    }
                }
                if faces.len() > max_tris {
                    return inv(format!("surface exceeds {max_tris} triangles: raise the threshold, blur more, or crop / downsample the mask"));
                }
            }
        }
    }
    Ok((pos, faces))
}

/// 3-tap box blur along each axis, `passes` times, edges clamped
pub fn blur(src: &[f32], dims: [usize; 3], passes: usize) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut a = src.to_vec();
    let mut b = vec![0f32; a.len()];
    let line = |s: &[f32], d: &mut [f32], n: usize, stride: usize, count: usize, base: &dyn Fn(usize) -> usize| {
        for c in 0..count {
            let o = base(c);
            for i in 0..n {
                let m = o + i.saturating_sub(1) * stride;
                let p = o + (i + 1).min(n - 1) * stride;
                let q = o + i * stride;
                d[q] = (s[m] + s[q] + s[p]) / 3.0;
            }
        }
    };
    for _ in 0..passes {
        line(&a, &mut b, nx, 1, ny * nz, &|c| c * nx);
        std::mem::swap(&mut a, &mut b);
        line(&a, &mut b, ny, nx, nx * nz, &|c| (c % nx) + (c / nx) * nx * ny);
        std::mem::swap(&mut a, &mut b);
        line(&a, &mut b, nz, nx * ny, nx * ny, &|c| c);
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// largest 6-connected component of `mask` ([Z,Y,X] bytes); returns the component mask and its size
pub(crate) fn largest_component(mask: &[u8], dims: [usize; 3]) -> (Vec<u8>, usize) {
    let [nx, ny, nz] = dims;
    let mut label = vec![0u32; mask.len()];
    let (mut best, mut best_n, mut next) = (0u32, 0usize, 0u32);
    let mut stack = Vec::new();
    for s in 0..mask.len() {
        if mask[s] == 0 || label[s] != 0 {
            continue;
        }
        next += 1;
        label[s] = next;
        stack.push(s);
        let mut n = 0usize;
        while let Some(q) = stack.pop() {
            n += 1;
            let (i, j, k) = (q % nx, (q / nx) % ny, q / (nx * ny));
            let mut go = |r: usize| {
                if mask[r] != 0 && label[r] == 0 {
                    label[r] = next;
                    stack.push(r);
                }
            };
            if i > 0 { go(q - 1) }
            if i + 1 < nx { go(q + 1) }
            if j > 0 { go(q - nx) }
            if j + 1 < ny { go(q + nx) }
            if k > 0 { go(q - nx * ny) }
            if k + 1 < nz { go(q + nx * ny) }
        }
        if n > best_n {
            best_n = n;
            best = next;
        }
    }
    (label.iter().map(|&l| (l == best && best != 0) as u8).collect(), best_n)
}

pub(crate) fn nums(o: &Json, k: &str) -> Option<Vec<f64>> {
    o[k].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
}

pub(crate) fn apply(a: &[f64; 16], p: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| a[r * 4] * p[0] + a[r * 4 + 1] * p[1] + a[r * 4 + 2] * p[2] + a[r * 4 + 3])
}

pub(crate) fn invert(a: &[f64; 16]) -> Option<[f64; 16]> {
    // 3×3 block inverse + translation
    let m = |r: usize, c: usize| a[r * 4 + c];
    let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
    if det.abs() < 1e-12 {
        return None;
    }
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m(r0, c0) * m(r1, c1) - m(r0, c1) * m(r1, c0);
    let i = [
        c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det,
        -c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det,
        c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det,
    ];
    let t = [m(0, 3), m(1, 3), m(2, 3)];
    let mut o = [0f64; 16];
    for r in 0..3 {
        for cc in 0..3 {
            o[r * 4 + cc] = i[r * 3 + cc];
        }
        o[r * 4 + 3] = -(i[r * 3] * t[0] + i[r * 3 + 1] * t[1] + i[r * 3 + 2] * t[2]);
    }
    o[15] = 1.0;
    Some(o)
}

pub(crate) fn affine_opt(o: &Json) -> Result<Option<[f64; 16]>, Error> {
    // options.affine: 4×4 (nested) or 3×4 / flat 16
    let Some(a) = o.get("affine").filter(|a| !a.is_null()) else { return Ok(None) };
    let flat: Vec<f64> = match a.as_array() {
        Some(rows) if rows.iter().all(|r| r.is_array()) => rows.iter().flat_map(|r| r.as_array().unwrap().iter().filter_map(|x| x.as_f64())).collect(),
        Some(v) => v.iter().filter_map(|x| x.as_f64()).collect(),
        None => return inv("affine: 4×4 numbers"),
    };
    match flat.len() {
        16 => Ok(Some(std::array::from_fn(|i| flat[i]))),
        12 => {
            let mut m = [0f64; 16];
            m[..12].copy_from_slice(&flat);
            m[15] = 1.0;
            Ok(Some(m))
        }
        _ => inv("affine: 4×4 (or 3×4) numbers"),
    }
}

/// voxels of tensor `t` (starting at flat offset `off`, `n` voxels) selected by options.label (== value) or options.threshold (≥, default 0.5)
pub(crate) fn select(t: &Tensor, off: usize, n: usize, o: &Json) -> Vec<u8> {
    match o["label"].as_f64() {
        Some(l) => (0..n).map(|i| (t.get_f64(off + i).0 == l) as u8).collect(),
        None => {
            let thr = o["threshold"].as_f64().unwrap_or(0.5);
            (0..n).map(|i| (t.get_f64(off + i).0 >= thr) as u8).collect()
        }
    }
}

/// voxel index → output coordinates: RAS mm through options.affine, else index × options.spacing_mm, else the index itself
pub(crate) fn space_map(o: &Json) -> Result<(&'static str, Option<[f64; 16]>, Option<[f64; 3]>, Box<dyn Fn([f64; 3]) -> [f64; 3]>), Error> {
    let spacing = nums(o, "spacing_mm").filter(|s| s.len() == 3 && s.iter().all(|&v| v > 0.0)).map(|s| [s[0], s[1], s[2]]);
    let affine = affine_opt(o)?;
    Ok(match (affine, spacing) {
        (Some(a), sp) => ("ras", Some(a), sp, Box::new(move |p| apply(&a, p))),
        (None, Some(s)) => ("voxel_mm", None, Some(s), Box::new(move |p| [p[0] * s[0], p[1] * s[1], p[2] * s[2]])),
        _ => ("voxel", None, None, Box::new(|p| p)),
    })
}

/// the surface pipeline of mesh_from_mask on a selection (keep, crop, blur, marching tetrahedra, space, Taubin); returns the mesh and the
/// number of voxels dropped by `keep: largest`
pub(crate) fn mask_to_mesh(sel: Vec<u8>, dims: [usize; 3], o: &Json) -> Result<(Mesh, usize), Error> {
    let [nx, ny, nz] = dims;
    let total = sel.iter().filter(|&&x| x != 0).count();
    if total == 0 {
        return inv("no voxel selected (check label / threshold)");
    }
    let (comp, removed) = match o["keep"].as_str().unwrap_or("largest") {
        "all" => (sel, 0),
        "largest" => {
            let (c, n) = largest_component(&sel, dims);
            (c, total - n)
        }
        other => return inv(format!("keep must be largest | all, got {other:?}")),
    };
    // crop to the bounding box + 2 voxels so the surface closes at the box edge and the blur has room
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
    let org = [0, 1, 2].map(|a| lo[a] as i64 - 2);
    let cd = [0, 1, 2].map(|a| hi[a] - lo[a] + 5);
    let mut crop = vec![0f32; cd[0] * cd[1] * cd[2]];
    for q in 0..comp.len() {
        if comp[q] != 0 {
            let p = [(q % nx) as i64 - org[0], ((q / nx) % ny) as i64 - org[1], (q / (nx * ny)) as i64 - org[2]];
            crop[(p[2] as usize * cd[1] + p[1] as usize) * cd[0] + p[0] as usize] = 1.0;
        }
    }
    let ind = blur(&crop, cd, o["blur"].as_u64().unwrap_or(1) as usize);
    let (pos, faces) = marching_tets(&ind, cd, 0.5, o["max_triangles"].as_u64().unwrap_or(6_000_000) as usize)?;
    let (space, affine, spacing_mm, map) = space_map(o)?;
    let verts: Vec<[f32; 3]> = pos.iter().map(|p| {
        let v = map([0, 1, 2].map(|a| p[a] as f64 + org[a] as f64));
        [v[0] as f32, v[1] as f32, v[2] as f32]
    }).collect();
    let mut m = Mesh { verts, faces, scalars: vec![], space: space.into(), affine, spacing_mm };
    // a reflecting affine (negative determinant) flips the winding: keep normals outward
    if let Some(a) = &m.affine {
        let det = a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8]) + a[2] * (a[4] * a[9] - a[5] * a[8]);
        if det < 0.0 {
            m.faces.iter_mut().for_each(|f| f.swap(1, 2));
        }
    }
    let iters = o["smooth"].as_u64().unwrap_or(6) as usize;
    if iters > 0 {
        m.taubin(iters, 0.5, -0.53);
    }
    Ok((m, removed))
}

pub(crate) fn mesh_in(name: &str, inp: &[std::sync::Arc<Value>], i: usize) -> Result<Mesh, Error> {
    Mesh::from_value(inp.get(i).ok_or_else(|| Error::Invalid(format!("{name}: input {i} (mesh) missing")))?).map_err(|e| Error::Invalid(format!("{name}: input {i}: {e}")))
}

pub fn register(r: &mut Registry) {
    r.register(
        "mesh_from_mask",
        "iso-surface mesh of a mask / label volume: inputs [tensor [Z,Y,X]]; options {label: value to select (else threshold: voxels ≥ threshold, default 0.5), blur: box-blur passes before the surface (default 1), smooth: Taubin iterations (default 6), keep: largest | all (default largest), spacing_mm [sx,sy,sz], affine (4×4 voxel→RAS mm; then space ras), max_triangles (6e6)}; output mesh value (see hasty_server::mesh); message JSON {vertices, faces, area_mm2, volume_mm3, components_removed, space}",
        |_, o, inp| {
            let t = inp.first().and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("mesh_from_mask: input 0 must be a tensor [Z, Y, X]".into()))?;
            let &[nz, ny, nx] = t.shape() else { return inv(format!("mesh_from_mask: tensor must be [Z, Y, X], got {:?}", t.shape())) };
            let sel = select(t, 0, nx * ny * nz, o);
            let (m, removed) = mask_to_mesh(sel, [nx, ny, nz], o)?;
            let msg = json!({"vertices": m.verts.len(), "faces": m.faces.len(), "area_mm2": m.area(), "volume_mm3": m.volume(), "components_removed_voxels": removed, "space": m.space});
            Ok((msg.to_string(), vec![m.to_value()]))
        },
    );
    r.register("mesh_smooth", "Taubin smoothing: inputs [mesh]; options {iterations (10), lambda (0.5), mu (-0.53)}; output mesh", |_, o, inp| {
        let mut m = mesh_in("mesh_smooth", inp, 0)?;
        m.taubin(o["iterations"].as_u64().unwrap_or(10) as usize, o["lambda"].as_f64().unwrap_or(0.5) as f32, o["mu"].as_f64().unwrap_or(-0.53) as f32);
        Ok((String::new(), vec![m.to_value()]))
    });
    r.register(
        "mesh_largest",
        "keep only the largest connected component (by vertex count): inputs [mesh]; message {removed_components}",
        |_, _, inp| {
            let mut m = mesh_in("mesh_largest", inp, 0)?;
            let (lab, n) = m.components();
            let mut cnt = vec![0usize; n];
            lab.iter().for_each(|&l| cnt[l as usize] += 1);
            let best = cnt.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i as u32).unwrap_or(0);
            m.retain(&lab.iter().map(|&l| l == best).collect::<Vec<_>>());
            Ok((json!({"removed_components": n.saturating_sub(1)}).to_string(), vec![m.to_value()]))
        },
    );
    r.register("mesh_info", "mesh statistics (message JSON: vertices, faces, area_mm2, volume_mm3 (signed; closed meshes), boundary_edges, components, bbox_mm, space, scalars)", |_, _, inp| {
        let m = mesh_in("mesh_info", inp, 0)?;
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for v in &m.verts {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        let stats: serde_json::Map<String, Json> = m.scalars.iter().map(|(k, s)| {
            let f: Vec<f32> = s.iter().copied().filter(|x| x.is_finite()).collect();
            let (mn, mx) = f.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
            (k.clone(), json!({"min": mn, "max": mx, "mean": f.iter().map(|&x| x as f64).sum::<f64>() / f.len().max(1) as f64, "valid": f.len()}))
        }).collect();
        Ok((json!({"vertices": m.verts.len(), "faces": m.faces.len(), "area_mm2": m.area(), "volume_mm3": m.volume(), "boundary_edges": m.boundary_edges(),
            "components": m.components().1, "bbox_mm": [lo, hi], "space": m.space, "scalars": stats}).to_string(), vec![]))
    });
    r.register(
        "mesh_sample",
        "sample a volume onto the mesh as a per-vertex scalar (trilinear; NaN outside the volume): inputs [mesh, tensor [Z,Y,X]]; options {name (\"value\"), spacing_mm, affine (of the volume; needed when the mesh space is ras)}; output mesh with the scalar added",
        |_, o, inp| {
            let mut m = mesh_in("mesh_sample", inp, 0)?;
            let t = inp.get(1).and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("mesh_sample: input 1 must be a tensor [Z, Y, X]".into()))?;
            let &[nz, ny, nx] = t.shape() else { return inv(format!("mesh_sample: volume must be [Z, Y, X], got {:?}", t.shape())) };
            let sp = nums(o, "spacing_mm").filter(|s| s.len() == 3 && s.iter().all(|&v| v > 0.0)).or(m.spacing_mm.map(|s| s.to_vec()));
            let aff = affine_opt(o)?.or(m.affine);
            let to_idx: Box<dyn Fn([f64; 3]) -> [f64; 3]> = match m.space.as_str() {
                "ras" => {
                    let inv_a = aff.as_ref().and_then(invert).ok_or_else(|| Error::Invalid("mesh_sample: mesh is in RAS space: options.affine of the volume required".into()))?;
                    Box::new(move |p| apply(&inv_a, p))
                }
                "voxel_mm" => {
                    let s = sp.ok_or_else(|| Error::Invalid("mesh_sample: options.spacing_mm of the volume required".into()))?;
                    Box::new(move |p| [p[0] / s[0], p[1] / s[1], p[2] / s[2]])
                }
                _ => Box::new(|p| p),
            };
            let at = |i: usize, j: usize, k: usize| t.get_f64((k * ny + j) * nx + i).0;
            let vals: Vec<f32> = m.verts.iter().map(|v| {
                let p = to_idx([v[0] as f64, v[1] as f64, v[2] as f64]);
                if p[0] < -0.5 || p[1] < -0.5 || p[2] < -0.5 || p[0] > nx as f64 - 0.5 || p[1] > ny as f64 - 0.5 || p[2] > nz as f64 - 0.5 {
                    return f32::NAN;
                }
                let c = [p[0].clamp(0.0, nx as f64 - 1.0), p[1].clamp(0.0, ny as f64 - 1.0), p[2].clamp(0.0, nz as f64 - 1.0)];
                let (i, j, k) = ((c[0] as usize).min(nx.saturating_sub(2)), (c[1] as usize).min(ny.saturating_sub(2)), (c[2] as usize).min(nz.saturating_sub(2)));
                let (fx, fy, fz) = (c[0] - i as f64, c[1] - j as f64, c[2] - k as f64);
                let (i1, j1, k1) = ((i + 1).min(nx - 1), (j + 1).min(ny - 1), (k + 1).min(nz - 1));
                let l = |a: f64, b: f64, f: f64| a * (1.0 - f) + b * f;
                let z0 = l(l(at(i, j, k), at(i1, j, k), fx), l(at(i, j1, k), at(i1, j1, k), fx), fy);
                let z1 = l(l(at(i, j, k1), at(i1, j, k1), fx), l(at(i, j1, k1), at(i1, j1, k1), fx), fy);
                l(z0, z1, fz) as f32
            }).collect();
            let name = o["name"].as_str().unwrap_or("value").to_string();
            m.scalars.retain(|(k, _)| *k != name);
            m.scalars.push((name, vals));
            Ok((String::new(), vec![m.to_value()]))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn sphere(n: usize, r: f64) -> Value {
        let c = (n as f64 - 1.0) / 2.0;
        let d: Vec<u8> = (0..n * n * n).map(|q| {
            let (i, j, k) = ((q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64);
            (((i - c).powi(2) + (j - c).powi(2) + (k - c).powi(2)).sqrt() <= r) as u8
        }).collect();
        Value::Tensor(Tensor::from_slice(&[n, n, n], &d).unwrap())
    }

    #[test]
    fn sphere_surface_is_closed_outward_and_has_the_right_size() {
        let reg = Registry::base();
        let ctx = crate::commands::Ctx::new(Arc::new(crate::Bank::new()));
        let r = 12.0;
        let (msg, out) = reg.run("mesh_from_mask", &ctx, r#"{"spacing_mm":[1.0,1.0,1.0],"blur":2,"smooth":4}"#, &[Arc::new(sphere(40, r))]).unwrap();
        let m = Mesh::from_value(&out[0]).unwrap();
        assert_eq!(m.boundary_edges(), 0, "closed surface");
        assert_eq!(m.components().1, 1);
        let (a, v) = (m.area(), m.volume());
        let (a0, v0) = (4.0 * std::f64::consts::PI * r * r, 4.0 / 3.0 * std::f64::consts::PI * r.powi(3));
        assert!((a - a0).abs() / a0 < 0.08, "area {a} vs {a0}");
        assert!(v > 0.0 && (v - v0).abs() / v0 < 0.08, "volume {v} vs {v0} (positive = outward windings)");
        let j: Json = serde_json::from_str(&msg).unwrap();
        assert_eq!(j["space"], "voxel_mm");
        // sample the distance-like volume back onto the mesh: values near the iso level everywhere
        let dist: Vec<f32> = (0..40 * 40 * 40).map(|q| {
            let (i, j, k) = ((q % 40) as f64, ((q / 40) % 40) as f64, (q / 1600) as f64);
            (((i - 19.5).powi(2) + (j - 19.5).powi(2) + (k - 19.5).powi(2)).sqrt()) as f32
        }).collect();
        let vol = Arc::new(Value::Tensor(Tensor::from_slice(&[40, 40, 40], &dist).unwrap()));
        let (_, s) = reg.run("mesh_sample", &ctx, r#"{"name":"r","spacing_mm":[1,1,1]}"#, &[Arc::new(out[0].clone()), vol]).unwrap();
        let ms = Mesh::from_value(&s[0]).unwrap();
        let r_s = &ms.scalars.iter().find(|(k, _)| k == "r").unwrap().1;
        let mean = r_s.iter().sum::<f32>() / r_s.len() as f32;
        assert!((mean as f64 - r).abs() < 0.6, "mean sampled radius {mean}");
    }

    #[test]
    fn two_blobs_keep_largest_and_tubes_report_components() {
        let n = 30usize;
        let mut d = vec![0u8; n * n * n];
        for q in 0..d.len() {
            let (i, j, k) = (q % n, (q / n) % n, q / (n * n));
            if (3..10).contains(&i) && (3..10).contains(&j) && (3..10).contains(&k) { d[q] = 1; }       // 7³
            if (18..22).contains(&i) && (18..22).contains(&j) && (18..22).contains(&k) { d[q] = 1; }    // 4³
        }
        let reg = Registry::base();
        let ctx = crate::commands::Ctx::new(Arc::new(crate::Bank::new()));
        let t = Arc::new(Value::Tensor(Tensor::from_slice(&[n, n, n], &d).unwrap()));
        let (_, all) = reg.run("mesh_from_mask", &ctx, r#"{"keep":"all","smooth":0}"#, &[t.clone()]).unwrap();
        assert_eq!(Mesh::from_value(&all[0]).unwrap().components().1, 2);
        let (_, one) = reg.run("mesh_from_mask", &ctx, r#"{"smooth":0}"#, &[t]).unwrap();
        assert_eq!(Mesh::from_value(&one[0]).unwrap().components().1, 1);
        let (msg, _) = reg.run("mesh_info", &ctx, "{}", &[Arc::new(one[0].clone())]).unwrap();
        assert!(serde_json::from_str::<Json>(&msg).unwrap()["boundary_edges"] == 0);
    }

    #[test]
    fn affine_inverse_round_trips() {
        let a = [0.0, -1.2, 0.0, 10.0, 1.2, 0.0, 0.0, -5.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let p = [3.0, 4.0, 5.0];
        let q = apply(&invert(&a).unwrap(), apply(&a, p));
        assert!((0..3).all(|i| (p[i] - q[i]).abs() < 1e-9));
    }
}
