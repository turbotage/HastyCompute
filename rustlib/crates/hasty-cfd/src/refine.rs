//! Grid refinement: node roles and interface tables of the finer levels (the scheme: kernels/lbm_core.h "grid
//! refinement").
//!
//! Level l has spacing dx / 2^l and its own sparse 8³-tile lattice; fine node 2g + {0, 1}³ lies in coarse cell g (cell
//! centred: the level origin moves by −dx_{l−1} / 4 per level). The refined region F_l is a set of level-(l−1) cells:
//! the cells whose centre lies in the case's `refine` shapes of level l, that are owned at level l − 1 and at least 3
//! cells from any node that is not (so the rim and its neighbours are plain owned nodes of level l − 1), and whose fine
//! tile can hold blood during the cycle. Then
//! - level l − 1: the cells of F_l are F_HIDDEN;
//! - the rim (cells next to F_l, 26-neighbourhood) stays owned at level l − 1; its 8 fine cells each are the ghosts;
//! - the fine tiles are the 4³-cell blocks touching F_l or the rim; their fine nodes outside both are unused
//!   (F_HIDDEN | F_GHOST: never fluid).

use std::collections::HashMap;

use crate::backend::Topology;
use crate::case::{Case, RefineSpec};
use crate::params::{F_GHOST, F_HIDDEN, F_IFACE, TILE3};
use crate::{Error, Result};

/// the interface tables of a fine level towards its coarser one (kernels/lbm_core.h node_explode … node_coalesce)
#[derive(Clone, Debug, Default)]
pub struct Interface {
    /// ghost fine node ids and their owner coarse node ids
    pub glist: Vec<i32>,
    pub gown: Vec<i32>,
    /// rim coarse node ids and, per rim node, the ghost indices of its 8 fine cells (−1: none)
    pub rim: Vec<i32>,
    pub csub: Vec<i32>,
    /// hidden coarse nodes next to owned ones and their 8 fine node ids (−1: none)
    pub hrim: Vec<i32>,
    pub hsub: Vec<i32>,
    /// per fine tile: the level-0 tile holding it (output accumulation; −1: none)
    pub atile: Vec<i32>,
}

/// WALE scale ramp of a level (× 1/32 per node, kernels/lbm_core.h node_step): Δ² of the coarser level (4) next to the
/// interface, falling linearly in Δ to this level's (1) at `ramp` nodes (26-neighbour layers) from the nearest ghost or
/// unused node; `ramp` 0: none (empty)
pub fn wale_ramp(topo: &Topology, roles: &[u8], ramp: usize) -> Vec<u8> {
    if ramp == 0 {
        return vec![];
    }
    let n = topo.nodes();
    let map = tile_map(topo);
    let mut d = vec![u16::MAX; n];
    let mut front: Vec<usize> = (0..n).filter(|&k| roles[k] & F_GHOST != 0).collect();
    for &k in &front {
        d[k] = 0;
    }
    for layer in 1..=ramp as u16 {
        let mut next = vec![];
        for &k in &front {
            let g = topo.coords(k);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(m) = node_at(&map, [g[0] + dx, g[1] + dy, g[2] + dz]) {
                            if d[m] == u16::MAX {
                                d[m] = layer;
                                next.push(m);
                            }
                        }
                    }
                }
            }
        }
        front = next;
    }
    d.iter()
        .map(|&x| {
            // Δ from 2 (layer 1, next to the ghosts) to 1 (layer ramp and beyond)
            let t = if x == u16::MAX { 1.0 } else { ((x as f64 - 1.0) / (ramp as f64 - 1.0).max(1.0)).clamp(0.0, 1.0) };
            let delta = 2.0 - t;
            (32.0 * delta * delta).round() as u8
        })
        .collect()
}

/// one finer level as built
#[derive(Clone, Debug)]
pub struct LevelBuild {
    pub level: usize,
    pub topo: Topology,
    /// static role bits per node (F_GHOST / unused; F_HIDDEN where a still finer level covers it)
    pub roles: Vec<u8>,
    pub iface: Interface,
    /// refined coarse cells, and the candidates dropped for the nesting margin / for tiles that never hold blood
    pub cells: usize,
    pub dropped_nest: usize,
    pub dropped_inactive: usize,
}

#[derive(Clone, Debug)]
pub struct Build {
    /// role bits of the level-0 nodes (F_HIDDEN)
    pub roles0: Vec<u8>,
    pub levels: Vec<LevelBuild>,
}

/// world position of node (0, 0, 0) of level l
pub fn level_origin(o0: [f64; 3], dx0: f64, l: usize) -> [f64; 3] {
    let s = 0.5 * dx0 * (1.0 - 0.5f64.powi(l as i32));
    [o0[0] - s, o0[1] - s, o0[2] - s]
}

pub fn tile_map(t: &Topology) -> HashMap<[i32; 3], i32> {
    (0..t.ntiles).map(|i| ([t.txyz[3 * i], t.txyz[3 * i + 1], t.txyz[3 * i + 2]], i as i32)).collect()
}

/// node index of lattice coordinates g through a tile map
pub fn node_at(map: &HashMap<[i32; 3], i32>, g: [i32; 3]) -> Option<usize> {
    let t = [g[0].div_euclid(8), g[1].div_euclid(8), g[2].div_euclid(8)];
    let ti = *map.get(&t)?;
    let l = ((g[2].rem_euclid(8) * 8 + g[1].rem_euclid(8)) * 8 + g[0].rem_euclid(8)) as usize;
    Some(ti as usize * TILE3 + l)
}

/// a dense box of cells [lo, lo + n)
struct BoxGrid {
    lo: [i32; 3],
    n: [i32; 3],
}

impl BoxGrid {
    fn len(&self) -> usize {
        (self.n[0] * self.n[1] * self.n[2]) as usize
    }
    fn idx(&self, g: [i32; 3]) -> Option<usize> {
        let r = [g[0] - self.lo[0], g[1] - self.lo[1], g[2] - self.lo[2]];
        if (0..3).any(|a| r[a] < 0 || r[a] >= self.n[a]) {
            return None;
        }
        Some(((r[2] * self.n[1] + r[1]) * self.n[0] + r[0]) as usize)
    }
    fn coord(&self, i: usize) -> [i32; 3] {
        let i = i as i32;
        [self.lo[0] + i % self.n[0], self.lo[1] + (i / self.n[0]) % self.n[1], self.lo[2] + i / (self.n[0] * self.n[1])]
    }
    /// Chebyshev dilation by r (separable running max)
    fn dilate(&self, m: &[bool], r: i32) -> Vec<bool> {
        let mut cur = m.to_vec();
        for a in 0..3 {
            let mut nxt = vec![false; cur.len()];
            for i in 0..cur.len() {
                if !cur[i] {
                    continue;
                }
                let g = self.coord(i);
                for d in -r..=r {
                    let mut h = g;
                    h[a] += d;
                    if let Some(j) = self.idx(h) {
                        nxt[j] = true;
                    }
                }
            }
            cur = nxt;
        }
        cur
    }
}

/// a fine tile can hold blood during the cycle: its centre within `margin` of the moving domain at one of `nsamp`
/// times (as solver::active_tiles)
fn block_active(case: &Case, c: [f64; 3], margin: f64, nsamp: usize) -> bool {
    let period = case.motion.period;
    (0..nsamp).any(|s| {
        let x0 = case.motion.inverse(&c, s as f64 * period / nsamp as f64, 6);
        case.sdf.at(&x0, -1.0) > -margin
    })
}

/// the levels of `case.refine` over the level-0 topology
pub fn build(case: &Case, topo0: &Topology, nsamp: usize) -> Result<Build> {
    let maxl = case.refine.iter().map(|r| r.level).max().unwrap_or(0);
    let (o0, dx0) = (case.grid.origin, case.grid.dx);
    let map0 = tile_map(topo0);
    let mut roles: Vec<Vec<u8>> = vec![vec![0u8; topo0.nodes()]];
    let mut topos: Vec<Topology> = vec![topo0.clone()];
    let mut maps: Vec<HashMap<[i32; 3], i32>> = vec![map0.clone()];
    let mut out: Vec<LevelBuild> = vec![];
    for l in 1..=maxl {
        let specs: Vec<&RefineSpec> = case.refine.iter().filter(|r| r.level == l).collect();
        if specs.is_empty() {
            return Err(Error::Case(format!("refine: level {l} has no region (levels must be consecutive)")));
        }
        let tc = &topos[l - 1];
        let rc = &roles[l - 1];
        let dxc = dx0 / (1u64 << (l - 1)) as f64;
        let oc = level_origin(o0, dx0, l - 1);
        let (dxf, of) = (dx0 / (1u64 << l) as f64, level_origin(o0, dx0, l));
        // candidate cells: owned coarse nodes whose centre lies in the region (tiles outside the shapes' bounding box
        // skipped: hundreds of spheres against tens of millions of nodes)
        let (blo, bhi) = specs.iter().flat_map(|s| s.boxes.iter().copied().chain(s.spheres.iter().map(|(c, r)| (c.map(|v| v - r), c.map(|v| v + r))))).fold(([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]), |(lo, hi), (a, b)| {
            ([0, 1, 2].map(|k| lo[k].min(a[k])), [0, 1, 2].map(|k| hi[k].max(b[k])))
        });
        let mut cand: Vec<[i32; 3]> = vec![];
        for t in 0..tc.ntiles {
            let t0 = [0, 1, 2].map(|a| oc[a] + (tc.txyz[3 * t + a] * 8) as f64 * dxc);
            if (0..3).any(|a| t0[a] + 7.0 * dxc < blo[a] || t0[a] > bhi[a]) {
                continue;
            }
            for k in 0..TILE3 {
                let n = t * TILE3 + k;
                if rc[n] != 0 {
                    continue;
                }
                let g = tc.coords(n);
                let x = [0, 1, 2].map(|a| oc[a] + g[a] as f64 * dxc);
                if specs.iter().any(|s| s.contains(&x)) {
                    cand.push(g);
                }
            }
        }
        if cand.is_empty() {
            return Err(Error::Case(format!("refine: the level-{l} region holds no owned level-{} node", l - 1)));
        }
        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        for g in &cand {
            for a in 0..3 {
                lo[a] = lo[a].min(g[a]);
                hi[a] = hi[a].max(g[a]);
            }
        }
        let bx = BoxGrid { lo: lo.map(|v| v - 8), n: [0, 1, 2].map(|a| hi[a] - lo[a] + 17) };
        // owned coarse nodes inside the box
        let mut owned = vec![false; bx.len()];
        let mut nidx = vec![-1i64; bx.len()];
        for t in 0..tc.ntiles {
            let tg = [tc.txyz[3 * t] * 8, tc.txyz[3 * t + 1] * 8, tc.txyz[3 * t + 2] * 8];
            if (0..3).any(|a| tg[a] + 7 < bx.lo[a] || tg[a] >= bx.lo[a] + bx.n[a]) {
                continue;
            }
            for k in 0..TILE3 {
                let n = t * TILE3 + k;
                if let Some(i) = bx.idx(tc.coords(n)) {
                    owned[i] = rc[n] == 0;
                    nidx[i] = n as i64;
                }
            }
        }
        let mut fset = vec![false; bx.len()];
        for g in &cand {
            fset[bx.idx(*g).unwrap()] = true;
        }
        // nesting margin: ≥ 3 cells from any node that is not owned (or does not exist)
        let notown: Vec<bool> = owned.iter().map(|&o| !o).collect();
        let near = bx.dilate(&notown, 2);
        let mut dropped_nest = 0;
        for i in 0..bx.len() {
            if fset[i] && near[i] {
                fset[i] = false;
                dropped_nest += 1;
            }
        }
        // fine tiles (4³-cell blocks) that never hold blood: their cells stay coarse
        let blk = |g: [i32; 3]| [g[0].div_euclid(4), g[1].div_euclid(4), g[2].div_euclid(4)];
        let margin = (3f64.sqrt() * 4.0 + 4.0) * dxc * 1.5;
        let mut active: HashMap<[i32; 3], bool> = HashMap::new();
        let mut dropped_inactive = 0;
        for i in 0..bx.len() {
            if !fset[i] {
                continue;
            }
            let k = blk(bx.coord(i));
            let a = *active.entry(k).or_insert_with(|| {
                let c = [0, 1, 2].map(|a| of[a] + (k[a] as f64 * 8.0 + 3.5) * dxf);
                block_active(case, c, margin, nsamp)
            });
            if !a {
                fset[i] = false;
                dropped_inactive += 1;
            }
        }
        let ncell = fset.iter().filter(|&&x| x).count();
        if ncell == 0 {
            return Err(Error::Case(format!("refine: level {l}: no cell left after the nesting margin ({dropped_nest} dropped) and the blood test ({dropped_inactive} dropped)")));
        }
        let dil = bx.dilate(&fset, 1);
        let rim: Vec<bool> = (0..bx.len()).map(|i| dil[i] && !fset[i]).collect();
        // fine tiles: blocks touching F or the rim
        let mut blocks: Vec<[i32; 3]> = {
            let mut v: Vec<[i32; 3]> = (0..bx.len()).filter(|&i| dil[i]).map(|i| blk(bx.coord(i))).collect();
            v.sort_unstable_by_key(|k| (k[2], k[1], k[0]));
            v.dedup();
            v
        };
        blocks.shrink_to_fit();
        let tf = Topology::new(&blocks);
        let mf = tile_map(&tf);
        let nf = tf.nodes();
        // roles of the fine nodes; the ghost list
        let mut rf = vec![0u8; nf];
        let mut gidx = vec![-1i32; nf];
        let mut it = Interface::default();
        for n in 0..nf {
            let g = tf.coords(n);
            let gc = [g[0].div_euclid(2), g[1].div_euclid(2), g[2].div_euclid(2)];
            let i = bx.idx(gc).ok_or_else(|| Error::Case("refine: a fine tile outside the build box".into()))?;
            if fset[i] {
                continue;
            }
            if rim[i] {
                rf[n] = F_GHOST;
                let own = nidx[i];
                if own < 0 || !owned[i] {
                    return Err(Error::Case(format!("refine: level {l}: a rim cell {gc:?} has no owned coarse node")));
                }
                gidx[n] = it.glist.len() as i32;
                it.glist.push(n as i32);
                it.gown.push(own as i32);
            } else {
                rf[n] = F_IFACE;
            }
        }
        let sub = |gc: [i32; 3], k: usize| -> Option<usize> { node_at(&mf, [2 * gc[0] + (k & 1) as i32, 2 * gc[1] + ((k >> 1) & 1) as i32, 2 * gc[2] + ((k >> 2) & 1) as i32]) };
        for i in 0..bx.len() {
            let gc = bx.coord(i);
            if rim[i] {
                it.rim.push(nidx[i] as i32);
                for k in 0..8 {
                    it.csub.push(sub(gc, k).map_or(-1, |q| gidx[q]));
                }
            }
            if fset[i] {
                // hidden next to an owned node
                let mut edge = false;
                'o: for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if let Some(j) = bx.idx([gc[0] + dx, gc[1] + dy, gc[2] + dz]) {
                                if !fset[j] {
                                    edge = true;
                                    break 'o;
                                }
                            }
                        }
                    }
                }
                if edge {
                    it.hrim.push(nidx[i] as i32);
                    for k in 0..8 {
                        it.hsub.push(sub(gc, k).map_or(-1, |q| q as i32));
                    }
                }
            }
        }
        // coarse nodes covered by this level
        let mut rcn = roles[l - 1].clone();
        for i in 0..bx.len() {
            if fset[i] {
                rcn[nidx[i] as usize] |= F_HIDDEN;
            }
        }
        roles[l - 1] = rcn;
        // output accumulation: the level-0 tile of each fine tile
        it.atile = (0..tf.ntiles)
            .map(|t| {
                let g0 = [0, 1, 2].map(|a| (tf.txyz[3 * t + a] * 8).div_euclid(1 << l));
                *map0.get(&[g0[0].div_euclid(8), g0[1].div_euclid(8), g0[2].div_euclid(8)]).unwrap_or(&-1)
            })
            .collect();
        out.push(LevelBuild { level: l, topo: tf.clone(), roles: vec![], iface: it, cells: ncell, dropped_nest, dropped_inactive });
        roles.push(rf);
        maps.push(mf);
        topos.push(tf);
    }
    for (k, lb) in out.iter_mut().enumerate() {
        lb.roles = roles[k + 1].clone();
    }
    Ok(Build { roles0: roles[0].clone(), levels: out })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins_nest_cell_centred() {
        // fine nodes 2g and 2g + 1 sit at ∓ dx/4 around coarse node g, on every level
        let (o0, dx) = ([0.0; 3], 1.0);
        let o1 = level_origin(o0, dx, 1);
        let o2 = level_origin(o0, dx, 2);
        assert!((o1[0] + 0.25).abs() < 1e-12 && (o2[0] + 0.375).abs() < 1e-12);
        for g in 0..4 {
            let xc = o0[0] + g as f64 * dx;
            assert!(((o1[0] + (2 * g) as f64 * 0.5) - (xc - 0.25)).abs() < 1e-12 && ((o1[0] + (2 * g + 1) as f64 * 0.5) - (xc + 0.25)).abs() < 1e-12);
            let x1 = o1[0] + g as f64 * 0.5;
            assert!(((o2[0] + (2 * g) as f64 * 0.25) - (x1 - 0.125)).abs() < 1e-12 && ((o2[0] + (2 * g + 1) as f64 * 0.25) - (x1 + 0.125)).abs() < 1e-12);
        }
    }
}
