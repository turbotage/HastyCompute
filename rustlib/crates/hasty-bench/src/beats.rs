//! The beat library of a cardiac LBM run (`cardiac_lbm --beats`, hasty-cfd): per cycle, phase bin and occupied output
//! cell the coverage, mean velocity, covariance, third central moments and wall shear stress (f16), read lazily and
//! cached per (beat, bin). Used by `simphantom --beats` to give every heartbeat of a simulated scan its own flow
//! realisation, with the intravoxel velocity distribution as moment-matched sub-sources.
//!
//! Positions are the LBM's lab frame at zero respiration (the cardiac frame of the phantom), as the Eulerian blood
//! frames of `prep_sim.py`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::Value;

/// f16 record fields of the current format (beats/index.json): w, mean (3), covariance (6), third central (10), wall
/// shear stress (3), spin history mz, mz_std, cov(s, u) (3); older libraries have 23 (no spin) or 25 (no cov(s, u))
pub const NF: usize = 28;
/// moments used per source: w, mean (3), covariance (6: xx yy zz xy xz yz), third central (10: xxx yyy zzz xxy xxz xyy
/// yyz xzz yzz xyz), the relative M_z E[s] (1 without a spin history) and cov(s, u) (3)
pub const NM: usize = 24;

fn f16(h: u16) -> f32 {
    hasty_cfd::motion::f16_to_f32(h)
}

/// one cycle's bin: occupied cells (ascending) and their records
pub struct BinData {
    pub cells: Vec<u32>,
    pub vals: Vec<f32>,
    pub samples: f32,
    /// fields per record
    pub nf: usize,
}

impl BinData {
    /// the record of a cell, if occupied
    pub fn get(&self, cell: u32) -> Option<&[f32]> {
        self.cells.binary_search(&cell).ok().map(|i| &self.vals[i * self.nf..(i + 1) * self.nf])
    }
}

pub struct BeatLibrary {
    pub dir: PathBuf,
    pub bins: usize,
    pub period: f64,
    /// cycle numbers of the beat files
    pub beats: Vec<u64>,
    /// fields per record (23, 25 or 28)
    pub nf: usize,
    /// the spin protocol of the run (index.json "spin"; Null without a spin history)
    pub spin: Value,
    origin: [f64; 3],
    /// cell spacing (out_factor × dx)
    pub h: f64,
    c1: i64,
    cpt: i64,
    tiles: HashMap<[i64; 3], i64>,
    cache: Mutex<HashMap<(usize, usize), Arc<BinData>>>,
    pub cache_cap: usize,
}

impl BeatLibrary {
    pub fn open(run: &Path) -> Result<Self, String> {
        let rd = |p: PathBuf| -> Result<Value, String> { serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string()) };
        let meta = rd(run.join("flow_bins.json"))?;
        let index = rd(run.join("beats").join("index.json"))?;
        let nf = index["fields"].as_array().map_or(0, |a| a.len());
        if ![23, 25, NF].contains(&nf) {
            return Err(format!("beat records of {nf} fields, want 23, 25 or {NF}"));
        }
        let mut beats: Vec<u64> = std::fs::read_dir(run.join("beats"))
            .map_err(|e| e.to_string())?
            .flatten()
            .filter_map(|e| e.file_name().to_string_lossy().strip_prefix("beat_").and_then(|x| x.strip_suffix(".bin")).and_then(|x| x.parse().ok()))
            .collect();
        beats.sort_unstable();
        if beats.is_empty() {
            return Err(format!("{}: no beat files", run.display()));
        }
        let fct = meta["out_factor"].as_u64().unwrap() as i64;
        let dx = meta["dx"].as_f64().unwrap();
        let o: Vec<f64> = meta["origin"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let txyz: Vec<i64> = meta["tile_xyz"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        let tiles = txyz.chunks(3).enumerate().map(|(i, t)| ([t[0], t[1], t[2]], i as i64)).collect();
        // centre of global cell (0, 0, 0): origin + (½ fct − ½) dx
        let c0 = [0, 1, 2].map(|a| o[a] + (0.5 * fct as f64 - 0.5) * dx);
        Ok(Self {
            dir: run.join("beats"),
            bins: index["bins"].as_u64().unwrap() as usize,
            period: index["period"].as_f64().unwrap(),
            beats,
            nf,
            spin: index["spin"].clone(),
            origin: c0,
            h: fct as f64 * dx,
            c1: 8 / fct,
            cpt: meta["cells_per_tile"].as_i64().unwrap(),
            tiles,
            cache: Mutex::new(HashMap::new()),
            cache_cap: 400,
        })
    }

    /// the output cell containing x (cell index tile · cells_per_tile + c), if its tile is active
    pub fn cell_of(&self, x: &[f64; 3]) -> Option<u32> {
        let g = [0, 1, 2].map(|a| ((x[a] - self.origin[a]) / self.h).round() as i64);
        let t = g.map(|v| v.div_euclid(self.c1));
        let l = g.map(|v| v.rem_euclid(self.c1));
        let ti = *self.tiles.get(&t)?;
        Some((ti * self.cpt + (l[2] * self.c1 + l[1]) * self.c1 + l[0]) as u32)
    }

    /// bin `b` of library beat `k` (an index into `beats`)
    pub fn bin(&self, k: usize, b: usize) -> Arc<BinData> {
        if let Some(d) = self.cache.lock().unwrap().get(&(k, b)) {
            return d.clone();
        }
        let d = Arc::new(self.load_bin(k, b));
        let mut c = self.cache.lock().unwrap();
        if c.len() >= self.cache_cap {
            c.clear();
        }
        c.insert((k, b), d.clone());
        d
    }

    fn load_bin(&self, k: usize, b: usize) -> BinData {
        // the whole file is small enough to scan (≈ 1 GB per beat at 0.5 mm / 1 mm cells): find the block of bin b
        let p = self.dir.join(format!("beat_{:04}.bin", self.beats[k]));
        let raw = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let u32at = |o: usize| u32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
        let mut o = 0;
        while o + 12 <= raw.len() {
            let (bb, n) = (u32at(o) as usize, u32at(o + 4) as usize);
            let samples = f32::from_le_bytes([raw[o + 8], raw[o + 9], raw[o + 10], raw[o + 11]]);
            let (ci, vi) = (o + 12, o + 12 + 4 * n);
            let end = vi + 2 * self.nf * n;
            if bb == b {
                let cells = (0..n).map(|i| u32at(ci + 4 * i)).collect();
                let vals = (0..self.nf * n).map(|i| f16(u16::from_le_bytes([raw[vi + 2 * i], raw[vi + 2 * i + 1]]))).collect();
                return BinData { cells, vals, samples, nf: self.nf };
            }
            o = end;
        }
        BinData { cells: vec![], vals: vec![], samples: 0.0, nf: self.nf }
    }

    /// moments [NM] at a cell, mixed in time between bins b0 and b1 with weight a (the mixture of the two bins'
    /// distributions: raw moments interpolate linearly); None when the cell is not covered (w < w_min) in either bin
    pub fn moments(d0: &BinData, d1: &BinData, a: f64, cell: u32, w_min: f32) -> Option<[f32; NM]> {
        let (r0, r1) = (d0.get(cell)?, d1.get(cell)?);
        if r0[0] < w_min || r1[0] < w_min {
            return None;
        }
        let raw = |r: &[f32]| -> ([f64; 3], [[f64; 3]; 3], [[[f64; 3]; 3]; 3]) {
            let m = [r[1] as f64, r[2] as f64, r[3] as f64];
            let mut c = [[0.0; 3]; 3];
            for (k, (i, j)) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].iter().enumerate() {
                c[*i][*j] = r[4 + k] as f64;
                c[*j][*i] = r[4 + k] as f64;
            }
            let mut t = [[[0.0; 3]; 3]; 3];
            for (k, &(i, j, l)) in TRI.iter().enumerate() {
                for (p, q, s) in perms(i, j, l) {
                    t[p][q][s] = r[10 + k] as f64;
                }
            }
            // raw E[uu], E[uuu] from the central ones
            let mut e2 = [[0.0; 3]; 3];
            let mut e3 = [[[0.0; 3]; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    e2[i][j] = c[i][j] + m[i] * m[j];
                    for l in 0..3 {
                        e3[i][j][l] = t[i][j][l] + m[i] * c[j][l] + m[j] * c[i][l] + m[l] * c[i][j] + m[i] * m[j] * m[l];
                    }
                }
            }
            (m, e2, e3)
        };
        let (m0, e20, e30) = raw(r0);
        let (m1, e21, e31) = raw(r1);
        let (wa, wb) = (1.0 - a, a);
        let m: [f64; 3] = [0, 1, 2].map(|i| wa * m0[i] + wb * m1[i]);
        let mut out = [0f32; NM];
        out[0] = (wa * r0[0] as f64 + wb * r1[0] as f64) as f32;
        for i in 0..3 {
            out[1 + i] = m[i] as f32;
        }
        let e2 = |i: usize, j: usize| wa * e20[i][j] + wb * e21[i][j];
        for (k, (i, j)) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].iter().enumerate() {
            out[4 + k] = (e2(*i, *j) - m[*i] * m[*j]) as f32;
        }
        for (k, &(i, j, l)) in TRI.iter().enumerate() {
            let e3 = wa * e30[i][j][l] + wb * e31[i][j][l];
            out[10 + k] = (e3 - m[i] * e2(j, l) - m[j] * e2(i, l) - m[l] * e2(i, j) + 2.0 * m[i] * m[j] * m[l]) as f32;
        }
        // spin history: E[s] and cov(s, u) of the mixture (raw E[s u] interpolates linearly)
        let sm = |r: &[f32]| if r.len() > 23 { r[23] as f64 } else { 1.0 };
        let su = |r: &[f32], i: usize| if r.len() > 25 { r[25 + i] as f64 + sm(r) * r[1 + i] as f64 } else { sm(r) * r[1 + i] as f64 };
        let s_mix = wa * sm(r0) + wb * sm(r1);
        out[20] = s_mix as f32;
        for i in 0..3 {
            out[21 + i] = (wa * su(r0, i) + wb * su(r1, i) - s_mix * m[i]) as f32;
        }
        Some(out)
    }
}

/// raw moments (E[u], E[uuᵀ], E[uuu], E[s], E[s u]) of a moment record (central form, see `moments`)
fn raw_of(r: &[f32; NM]) -> ([f64; 3], [[f64; 3]; 3], [[[f64; 3]; 3]; 3], f64, [f64; 3]) {
    let m = [r[1] as f64, r[2] as f64, r[3] as f64];
    let mut c = [[0.0; 3]; 3];
    for (k, (i, j)) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].iter().enumerate() {
        c[*i][*j] = r[4 + k] as f64;
        c[*j][*i] = r[4 + k] as f64;
    }
    let mut t = [[[0.0; 3]; 3]; 3];
    for (k, &(i, j, l)) in TRI.iter().enumerate() {
        for (p, q, s) in perms(i, j, l) {
            t[p][q][s] = r[10 + k] as f64;
        }
    }
    let mut e2 = [[0.0; 3]; 3];
    let mut e3 = [[[0.0; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            e2[i][j] = c[i][j] + m[i] * m[j];
            for l in 0..3 {
                e3[i][j][l] = t[i][j][l] + m[i] * c[j][l] + m[j] * c[i][l] + m[l] * c[i][j] + m[i] * m[j] * m[l];
            }
        }
    }
    let sm = r[20] as f64;
    (m, e2, e3, sm, [0, 1, 2].map(|i| r[21 + i] as f64 + sm * m[i]))
}

/// central-form record from raw moments (coverage w)
fn central_of(w: f64, m: [f64; 3], e2: [[f64; 3]; 3], e3: [[[f64; 3]; 3]; 3], sm: f64, esu: [f64; 3]) -> [f32; NM] {
    let mut out = [0f32; NM];
    out[0] = w as f32;
    for i in 0..3 {
        out[1 + i] = m[i] as f32;
    }
    for (k, (i, j)) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].iter().enumerate() {
        out[4 + k] = (e2[*i][*j] - m[*i] * m[*j]) as f32;
    }
    for (k, &(i, j, l)) in TRI.iter().enumerate() {
        out[10 + k] = (e3[i][j][l] - m[i] * e2[j][l] - m[j] * e2[i][l] - m[l] * e2[i][j] + 2.0 * m[i] * m[j] * m[l]) as f32;
    }
    out[20] = sm as f32;
    for i in 0..3 {
        out[21 + i] = (esu[i] - sm * m[i]) as f32;
    }
    out
}

/// the mixture of several cells' distributions (weights: blood volume of each part); the coverage of the result is
/// `coverage` (the caller's voxel fraction); None without weight
pub fn mix(parts: &[(f64, [f32; NM])], coverage: f64) -> Option<[f32; NM]> {
    let wt: f64 = parts.iter().map(|p| p.0).sum();
    if wt <= 0.0 {
        return None;
    }
    let (mut m, mut e2, mut e3, mut sm, mut esu) = ([0.0; 3], [[0.0; 3]; 3], [[[0.0; 3]; 3]; 3], 0.0, [0.0; 3]);
    for (w, r) in parts {
        let a = w / wt;
        let (m_, e2_, e3_, sm_, esu_) = raw_of(r);
        sm += a * sm_;
        for i in 0..3 {
            m[i] += a * m_[i];
            esu[i] += a * esu_[i];
            for j in 0..3 {
                e2[i][j] += a * e2_[i][j];
                for l in 0..3 {
                    e3[i][j][l] += a * e3_[i][j][l];
                }
            }
        }
    }
    Some(central_of(coverage, m, e2, e3, sm, esu))
}

/// a record of the velocity relative to a moving frame mapped to the lab: u = F u_rel + u_w (F = ∂x/∂X of the frame
/// motion at the cell, u_w its velocity): mean F μ + u_w, covariance F Σ Fᵀ, third central moments F⊗F⊗F, cov(s, u) F c
pub fn transform(r: &[f32; NM], f: &[[f64; 3]; 3], uw: &[f64; 3]) -> [f32; NM] {
    let mut out = *r;
    let mu = [r[1] as f64, r[2] as f64, r[3] as f64];
    for i in 0..3 {
        out[1 + i] = ((0..3).map(|j| f[i][j] * mu[j]).sum::<f64>() + uw[i]) as f32;
    }
    let mut c = [[0.0; 3]; 3];
    for (k, (i, j)) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].iter().enumerate() {
        c[*i][*j] = r[4 + k] as f64;
        c[*j][*i] = r[4 + k] as f64;
    }
    for (k, (i, j)) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)].iter().enumerate() {
        let mut v = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                v += f[*i][a] * c[a][b] * f[*j][b];
            }
        }
        out[4 + k] = v as f32;
    }
    let mut t = [[[0.0; 3]; 3]; 3];
    for (k, &(i, j, l)) in TRI.iter().enumerate() {
        for (p, q, s) in perms(i, j, l) {
            t[p][q][s] = r[10 + k] as f64;
        }
    }
    for (k, &(i, j, l)) in TRI.iter().enumerate() {
        let mut v = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                for d in 0..3 {
                    v += f[i][a] * f[j][b] * f[l][d] * t[a][b][d];
                }
            }
        }
        out[10 + k] = v as f32;
    }
    let cs = [r[21] as f64, r[22] as f64, r[23] as f64];
    for i in 0..3 {
        out[21 + i] = (0..3).map(|j| f[i][j] * cs[j]).sum::<f64>() as f32;
    }
    out
}

pub const TRI: [(usize, usize, usize); 10] = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2), (0, 1, 1), (1, 1, 2), (0, 2, 2), (1, 2, 2), (0, 1, 2)];

fn perms(i: usize, j: usize, l: usize) -> [(usize, usize, usize); 6] {
    [(i, j, l), (i, l, j), (j, i, l), (j, l, i), (l, i, j), (l, j, i)]
}

/// the library beat for each heartbeat of a scan: a seeded shuffle of the library, reshuffled when exhausted, never the
/// same beat twice in a row
pub fn beat_sequence(n_lib: usize, n_beats: usize, seed: u64) -> Vec<usize> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x632BE59BD9B4E019) | 1;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545F4914F6CDD1D) >> 11
    };
    let mut out: Vec<usize> = Vec::with_capacity(n_beats);
    while out.len() < n_beats {
        let mut p: Vec<usize> = (0..n_lib).collect();
        for i in (1..n_lib).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        if n_lib > 1 && out.last() == p.first() {
            p.swap(0, n_lib - 1);
        }
        out.extend(p);
    }
    out.truncate(n_beats);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixture_of_identical_bins_is_the_bin() {
        let mut v = vec![0f32; NF];
        v[0] = 1.0;
        v[1] = 0.3;
        v[2] = -0.1;
        v[3] = 0.05;
        v[4] = 0.01;
        v[5] = 0.02;
        v[6] = 0.005;
        v[7] = 0.001;
        v[10] = 1e-4;
        v[13] = -2e-4;
        v[23] = 0.4;
        v[25] = 0.02;
        v[27] = -0.01;
        let d = BinData { cells: vec![7], vals: v.clone(), samples: 1.0, nf: NF };
        let m = BeatLibrary::moments(&d, &d, 0.3, 7, 0.5).unwrap();
        for k in 0..20 {
            assert!((m[k] - v[k]).abs() < 1e-6, "{k}: {} vs {}", m[k], v[k]);
        }
        assert!((m[20] - 0.4).abs() < 1e-6 && (m[21] - 0.02).abs() < 1e-6 && m[22].abs() < 1e-6 && (m[23] + 0.01).abs() < 1e-6, "{:?}", &m[20..]);
        assert!(BeatLibrary::moments(&d, &d, 0.3, 8, 0.5).is_none());
    }

    #[test]
    fn mixture_of_two_deltas_has_their_spread() {
        // two cells' bins: point masses at u = 0 and u = 1 (x): the 50/50 mixture has mean ½, variance ¼, skew 0
        let mk = |u: f32| {
            let mut v = vec![0f32; NF];
            v[0] = 1.0;
            v[1] = u;
            // the fast one fresh (s = 1), the slow one saturated (s = 0.2)
            v[23] = 0.2 + 0.8 * u;
            BinData { cells: vec![0], vals: v, samples: 1.0, nf: NF }
        };
        let m = BeatLibrary::moments(&mk(0.0), &mk(1.0), 0.5, 0, 0.5).unwrap();
        assert!((m[1] - 0.5).abs() < 1e-6 && (m[4] - 0.25).abs() < 1e-6 && m[10].abs() < 1e-6, "{m:?}");
        // E[s] 0.6, cov(s, u_x) = E[s u] − E[s] E[u] = 0.5 − 0.3 = 0.2
        assert!((m[20] - 0.6).abs() < 1e-6 && (m[21] - 0.2).abs() < 1e-6, "{:?}", &m[20..]);
        let m = BeatLibrary::moments(&mk(0.0), &mk(1.0), 0.25, 0, 0.5).unwrap();
        // P(1) = ¼: mean ¼, var 3/16, third central p(1−p)(1−2p) = 3/32
        assert!((m[1] - 0.25).abs() < 1e-6 && (m[4] - 3.0 / 16.0).abs() < 1e-6 && (m[10] - 3.0 / 32.0).abs() < 1e-6, "{m:?}");
    }

    #[test]
    fn mixture_and_frame_transform() {
        // two point masses (x = 0 and x = 1, s 0.2 and 1) mixed 50/50 as two cells: mean ½, var ¼, cov(s, u) 0.2
        let mk = |u: f32, s: f32| {
            let mut v = [0f32; NM];
            v[0] = 1.0;
            v[1] = u;
            v[20] = s;
            v
        };
        let m = mix(&[(1.0, mk(0.0, 0.2)), (1.0, mk(1.0, 1.0))], 0.8).unwrap();
        assert!((m[0] - 0.8).abs() < 1e-6 && (m[1] - 0.5).abs() < 1e-6 && (m[4] - 0.25).abs() < 1e-6 && (m[20] - 0.6).abs() < 1e-6 && (m[21] - 0.2).abs() < 1e-6, "{m:?}");
        // a rotation by 90° about z and a frame velocity: x-moments go to y
        let f = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let mut r = m;
        r[10] = 0.03; // m_xxx
        let t = transform(&r, &f, &[0.0, 0.0, 0.1]);
        assert!((t[2] - 0.5).abs() < 1e-6 && t[1].abs() < 1e-6 && (t[3] - 0.1).abs() < 1e-6, "mean {:?}", &t[1..4]);
        assert!((t[5] - 0.25).abs() < 1e-6 && t[4].abs() < 1e-6, "cov {:?}", &t[4..10]);
        assert!((t[11] - 0.03).abs() < 1e-6 && t[10].abs() < 1e-6, "m3 {:?}", &t[10..20]);
        assert!((t[22] - 0.2).abs() < 1e-6 && t[21].abs() < 1e-6 && (t[20] - 0.6).abs() < 1e-6, "s {:?}", &t[20..]);
    }

    #[test]
    fn sequence_never_repeats_back_to_back() {
        let s = beat_sequence(5, 200, 3);
        assert!(s.windows(2).all(|w| w[0] != w[1]));
        assert!((0..5).all(|k| s.iter().filter(|&&x| x == k).count() == 40));
    }
}
