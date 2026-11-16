//! Closing boundary loops of a mesh: hole filling and the caps of plane cuts. Loops are projected to a plane, nested loops are treated
//! as holes of the loop around them (bridged into one polygon), and the polygon is triangulated by ear clipping. No vertices are added.

use std::collections::{HashMap, HashSet};

type P2 = [f64; 2];
type V3 = [f64; 3];

/// directed edges → closed loops (each vertex follows its outgoing edge; open chains are dropped)
pub fn loops_from_edges(edges: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in edges {
        next.entry(a).or_default().push(b);
    }
    let mut used: HashSet<(u32, u32)> = HashSet::new();
    let mut out = Vec::new();
    for &(a0, b0) in edges {
        if used.contains(&(a0, b0)) {
            continue;
        }
        let mut lp = vec![a0];
        let (mut a, mut b) = (a0, b0);
        let closed = loop {
            used.insert((a, b));
            if b == a0 {
                break true;
            }
            if lp.len() > edges.len() {
                break false;
            }
            lp.push(b);
            let Some(nb) = next.get(&b).and_then(|v| v.iter().copied().find(|&c| !used.contains(&(b, c)))) else { break false };
            a = b;
            b = nb;
        };
        if closed && lp.len() >= 3 {
            out.push(lp);
        }
    }
    out
}

/// boundary loops of a triangle list (edges used by one triangle only), directed as in that triangle
pub fn boundary_loops(faces: &[[u32; 3]]) -> Vec<Vec<u32>> {
    let mut have: HashSet<(u32, u32)> = HashSet::new();
    for f in faces {
        for k in 0..3 {
            have.insert((f[k], f[(k + 1) % 3]));
        }
    }
    let edges: Vec<(u32, u32)> = faces.iter().flat_map(|f| (0..3).map(move |k| (f[k], f[(k + 1) % 3]))).filter(|&(a, b)| !have.contains(&(b, a))).collect();
    loops_from_edges(&edges)
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 0.0 { [a[0] / l, a[1] / l, a[2] / l] } else { a }
}

/// Newell normal of a loop (its best-fit plane normal; length ∝ area)
fn newell(p: &[V3]) -> V3 {
    let mut n = [0.0; 3];
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

pub(crate) fn area2(p: &[P2]) -> f64 {
    (0..p.len()).map(|i| { let (a, b) = (p[i], p[(i + 1) % p.len()]); a[0] * b[1] - b[0] * a[1] }).sum::<f64>() / 2.0
}

pub(crate) fn inside_poly(pt: P2, poly: &[P2]) -> bool {
    let mut c = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > pt[1]) != (b[1] > pt[1]) && pt[0] < (b[0] - a[0]) * (pt[1] - a[1]) / (b[1] - a[1]) + a[0] {
            c = !c;
        }
        j = i;
    }
    c
}

fn cross2(o: P2, a: P2, b: P2) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// proper crossing of segments ab and cd (shared endpoints and touching do not count)
fn crosses(a: P2, b: P2, c: P2, d: P2) -> bool {
    let (d1, d2, d3, d4) = (cross2(a, b, c), cross2(a, b, d), cross2(c, d, a), cross2(c, d, b));
    ((d1 > 1e-12 && d2 < -1e-12) || (d1 < -1e-12 && d2 > 1e-12)) && ((d3 > 1e-12 && d4 < -1e-12) || (d3 < -1e-12 && d4 > 1e-12))
}

fn in_tri(p: P2, a: P2, b: P2, c: P2) -> bool {
    let (d1, d2, d3) = (cross2(a, b, p), cross2(b, c, p), cross2(c, a, p));
    d1 >= -1e-12 && d2 >= -1e-12 && d3 >= -1e-12
}

/// triangulate a polygon (CCW, vertex indices into `pts`) with holes (CW) by bridging the holes and ear clipping; returns index triples
pub(crate) fn triangulate(outer: &[usize], holes: &[Vec<usize>], pts: &[P2]) -> Vec<[usize; 3]> {
    let mut poly: Vec<usize> = outer.to_vec();
    let mut order: Vec<usize> = (0..holes.len()).collect();
    let max_x = |h: &Vec<usize>| h.iter().map(|&i| pts[i][0]).fold(f64::NEG_INFINITY, f64::max);
    order.sort_by(|&a, &b| max_x(&holes[b]).total_cmp(&max_x(&holes[a])));
    for &hi in &order {
        let h = &holes[hi];
        let m_pos = (0..h.len()).max_by(|&a, &b| pts[h[a]][0].total_cmp(&pts[h[b]][0])).unwrap();
        let m = h[m_pos];
        let mut cand: Vec<usize> = (0..poly.len()).collect();
        cand.sort_by(|&a, &b| {
            let (da, db) = ((pts[poly[a]][0] - pts[m][0]).powi(2) + (pts[poly[a]][1] - pts[m][1]).powi(2), (pts[poly[b]][0] - pts[m][0]).powi(2) + (pts[poly[b]][1] - pts[m][1]).powi(2));
            da.total_cmp(&db)
        });
        let others: Vec<&Vec<usize>> = holes.iter().enumerate().filter(|(i, _)| order.iter().position(|&o| o == *i).map(|p| p >= order.iter().position(|&o| o == hi).unwrap()).unwrap_or(false)).map(|(_, h)| h).collect();
        let visible = |v: usize| -> bool {
            let (a, b) = (pts[m], pts[poly[v]]);
            for i in 0..poly.len() {
                let (c, d) = (pts[poly[i]], pts[poly[(i + 1) % poly.len()]]);
                if crosses(a, b, c, d) { return false; }
            }
            for hh in &others {
                for i in 0..hh.len() {
                    if crosses(a, b, pts[hh[i]], pts[hh[(i + 1) % hh.len()]]) { return false; }
                }
            }
            true
        };
        let at = cand.iter().copied().find(|&v| visible(v)).unwrap_or(cand[0]);
        let mut seq: Vec<usize> = (0..h.len()).map(|k| h[(m_pos + k) % h.len()]).collect();
        seq.push(m);
        seq.push(poly[at]);
        let tail: Vec<usize> = poly.split_off(at + 1);
        poly.extend(seq);
        poly.extend(tail);
    }
    // ear clipping
    let mut tris = Vec::new();
    let mut ring: Vec<usize> = poly;
    let mut guard = 0usize;
    let limit = ring.len() * ring.len() + 100;
    while ring.len() > 3 && guard < limit {
        guard += 1;
        let n = ring.len();
        let mut clipped = false;
        for pass in 0..2 {
            for i in 0..n {
                let (a, b, c) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
                let (pa, pb, pc) = (pts[a], pts[b], pts[c]);
                let cr = cross2(pa, pb, pc);
                if cr < -1e-14 {
                    continue;      // reflex corner (collinear corners are allowed: a zero-area triangle keeps the edges matched)
                }
                if pass == 0 {
                    let blocked = ring.iter().any(|&q| {
                        let pq = pts[q];
                        q != a && q != b && q != c && pq != pa && pq != pb && pq != pc && in_tri(pq, pa, pb, pc)
                    });
                    if blocked {
                        continue;
                    }
                }
                tris.push([a, b, c]);
                ring.remove(i);
                clipped = true;
                break;
            }
            if clipped {
                break;
            }
        }
        if !clipped {
            // degenerate remainder (collinear): drop the vertex with the smallest |cross|
            let i = (0..n).min_by(|&x, &y| cross2(pts[ring[(x + n - 1) % n]], pts[ring[x]], pts[ring[(x + 1) % n]]).abs().total_cmp(&cross2(pts[ring[(y + n - 1) % n]], pts[ring[y]], pts[ring[(y + 1) % n]]).abs())).unwrap();
            tris.push([ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]]);
            ring.remove(i);
        }
    }
    // whatever is left (guard hit): fan, so the loop is always closed
    for k in 1..ring.len().saturating_sub(1) {
        tris.push([ring[0], ring[k], ring[k + 1]]);
    }
    tris
}

/// Fill `loops` (vertex ids, directed as the boundary edges of the existing triangles). With `plane` (a normal) all loops are taken as lying in
/// that plane and loops inside other loops become holes; without it each loop is filled in its own best-fit plane. The returned triangles close the
/// surface (their winding is the reverse of the loop direction).
pub fn fill_loops(verts: &[[f32; 3]], loops: &[Vec<u32>], plane: Option<V3>) -> Vec<[u32; 3]> {
    let p3 = |v: u32| [verts[v as usize][0] as f64, verts[v as usize][1] as f64, verts[v as usize][2] as f64];
    let mut out = Vec::new();
    let basis = |n: V3| -> (V3, V3) {
        let n = norm(n);
        let axis = if n[0].abs() < 0.6 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = norm(cross(n, axis));
        (u, cross(n, u))
    };
    // groups: (outer loop index, hole loop indices) and the shared / per-loop 2-D coordinates
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut coords: Vec<Vec<P2>> = Vec::new();
    match plane {
        Some(n) => {
            let (u, v) = basis(n);
            coords = loops.iter().map(|l| l.iter().map(|&i| { let p = p3(i); [dot(p, u), dot(p, v)] }).collect()).collect();
            let mut by_area: Vec<usize> = (0..loops.len()).collect();
            by_area.sort_by(|&a, &b| area2(&coords[b]).abs().total_cmp(&area2(&coords[a]).abs()));
            let mut depth = vec![0usize; loops.len()];
            let mut parent = vec![usize::MAX; loops.len()];
            for (k, &i) in by_area.iter().enumerate() {
                // innermost larger loop containing this one
                for &j in by_area[..k].iter().rev() {
                    if inside_poly(coords[i][0], &coords[j]) {
                        parent[i] = j;
                        depth[i] = depth[j] + 1;
                        break;
                    }
                }
            }
            for &i in &by_area {
                if depth[i] % 2 == 0 {
                    groups.push((i, (0..loops.len()).filter(|&h| parent[h] == i && depth[h] % 2 == 1).collect()));
                }
            }
        }
        None => {
            for l in loops {
                let pts: Vec<V3> = l.iter().map(|&i| p3(i)).collect();
                let (u, v) = basis(newell(&pts));
                coords.push(pts.iter().map(|p| [dot(*p, u), dot(*p, v)]).collect());
            }
            groups = (0..loops.len()).map(|i| (i, vec![])).collect();
        }
    }
    let directed: HashSet<(u32, u32)> = loops.iter().flat_map(|l| (0..l.len()).map(move |k| (l[k], l[(k + 1) % l.len()]))).collect();
    for (outer, holes) in groups {
        // one point list for the group: positions of loop vertices (a vertex id may appear in a bridge twice, so index by position)
        let mut ids: Vec<u32> = Vec::new();
        let mut pts: Vec<P2> = Vec::new();
        let ring = |li: usize, want_ccw: bool, ids: &mut Vec<u32>, pts: &mut Vec<P2>| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..loops[li].len()).collect();
            if (area2(&coords[li]) > 0.0) != want_ccw {
                idx.reverse();
            }
            idx.iter().map(|&k| { ids.push(loops[li][k]); pts.push(coords[li][k]); ids.len() - 1 }).collect()
        };
        let o = ring(outer, true, &mut ids, &mut pts);
        let hs: Vec<Vec<usize>> = holes.iter().map(|&h| ring(h, false, &mut ids, &mut pts)).collect();
        let tris = triangulate(&o, &hs, &pts);
        let mut tv: Vec<[u32; 3]> = tris.iter().map(|t| [ids[t[0]], ids[t[1]], ids[t[2]]]).collect();
        // winding: the fill must traverse every loop edge in the direction opposite to the loop
        let flip = tv.iter().find_map(|t| (0..3).find_map(|k| {
            let e = (t[k], t[(k + 1) % 3]);
            if directed.contains(&e) { Some(true) } else if directed.contains(&(e.1, e.0)) { Some(false) } else { None }
        })).unwrap_or(false);
        if flip {
            tv.iter_mut().for_each(|t| t.swap(1, 2));
        }
        out.extend(tv);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_with_a_square_hole_is_filled_without_the_hole() {
        // an annulus in the z = 0 plane: outer 4×4, hole 1×1 around the centre; loops directed like the boundary of a kept surface
        let verts: Vec<[f32; 3]> = vec![[0., 0., 0.], [4., 0., 0.], [4., 4., 0.], [0., 4., 0.], [1.5, 1.5, 0.], [2.5, 1.5, 0.], [2.5, 2.5, 0.], [1.5, 2.5, 0.]];
        let loops = vec![vec![0, 1, 2, 3], vec![4, 7, 6, 5]];
        let t = fill_loops(&verts, &loops, Some([0.0, 0.0, 1.0]));
        let area: f64 = t.iter().map(|f| {
            let [a, b, c] = f.map(|i| verts[i as usize]);
            (((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0).abs() as f64
        }).sum();
        assert!((area - 15.0).abs() < 1e-6, "annulus area {area}");
        // the fill uses every loop edge in the opposite direction
        let used: HashSet<(u32, u32)> = t.iter().flat_map(|f| (0..3).map(move |k| (f[k], f[(k + 1) % 3]))).collect();
        for l in &loops {
            for k in 0..l.len() {
                assert!(used.contains(&(l[(k + 1) % l.len()], l[k])), "edge {}→{} not closed", l[k], l[(k + 1) % l.len()]);
            }
        }
    }

    #[test]
    fn concave_loop_is_filled_by_ear_clipping() {
        // an L-shaped loop (area 3)
        let verts: Vec<[f32; 3]> = vec![[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [1., 1., 0.], [1., 2., 0.], [0., 2., 0.]];
        let t = fill_loops(&verts, &[vec![0, 1, 2, 3, 4, 5]], None);
        assert_eq!(t.len(), 4);
        let area: f64 = t.iter().map(|f| {
            let [a, b, c] = f.map(|i| verts[i as usize]);
            (((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0).abs() as f64
        }).sum();
        assert!((area - 3.0).abs() < 1e-9, "{area}");
    }
}
