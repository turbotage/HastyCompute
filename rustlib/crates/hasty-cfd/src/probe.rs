//! Debug views of a running solver (any backend): the fastest fluid nodes with their surroundings (density, flags,
//! wall fraction, wall velocity, nearest valve and cut), speed percentiles, and sparse node snapshots for slicing in
//! Python (`rustlib/tools/phantom/lbm_report.py --snap`).

use std::io::Write;
use std::path::Path;

use serde_json::json;

use crate::backend::Backend;
use crate::params::{F_BC, F_FLUID, F_PSM, F_WALL};
use crate::solver::Solver;
use crate::Result;

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// the `k` fastest fluid nodes and speed percentiles (m/s), split by node class (bulk / next to a wall / PSM / cut slab)
pub fn hot_spots<B: Backend>(s: &mut Solver<B>, k: usize) -> Result<serde_json::Value> {
    let f = s.be.fields()?;
    let n = s.be.nodes();
    let sc = s.u.dx / s.u.dt;
    let t = s.time();
    let speed = |i: usize| ((f.ru[n + i].powi(2) + f.ru[2 * n + i].powi(2) + f.ru[3 * n + i].powi(2)).sqrt() as f64) * sc;
    let mut all: Vec<(f64, usize)> = (0..n).filter(|&i| f.flags[i] & F_FLUID != 0).map(|i| (speed(i), i)).collect();
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let pct = |v: &[(f64, usize)], q: f64| if v.is_empty() { f64::NAN } else { v[((v.len() as f64 * (1.0 - q)) as usize).min(v.len() - 1)].0 };
    let class = |i: usize| {
        let fl = f.flags[i];
        if fl & F_BC != 0 {
            "cut"
        } else if fl & F_PSM != 0 {
            "psm"
        } else if fl & F_WALL != 0 {
            "wall"
        } else {
            "bulk"
        }
    };
    let mut by_class = serde_json::Map::new();
    for c in ["bulk", "wall", "psm", "cut"] {
        let v: Vec<(f64, usize)> = all.iter().copied().filter(|&(_, i)| class(i) == c).collect();
        by_class.insert(c.into(), json!({"nodes": v.len(), "p99": pct(&v, 0.99), "p999": pct(&v, 0.999), "max": v.first().map_or(f64::NAN, |x| x.0)}));
    }
    let vc: Vec<(String, [f64; 3])> = s.case.valves.iter().map(|v| (v.name.clone(), s.case.motion.forward(&v.center, t))).collect();
    let cc: Vec<(String, [f64; 3])> = s.case.cuts.iter().map(|c| (c.name.clone(), s.case.motion.forward(&c.center, t))).collect();
    let near = |x: &[f64; 3], l: &[(String, [f64; 3])]| l.iter().map(|(nm, c)| (nm.clone(), dist(x, c))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(nm, d)| json!({"name": nm, "mm": d * 1e3}));
    let mut top = vec![];
    for &(sp, i) in all.iter().take(k) {
        let g = s.topo.coords(i);
        let x = [0, 1, 2].map(|a| s.case.grid.origin[a] + g[a] as f64 * s.u.dx);
        top.push(json!({
            "speed": sp, "u": [f.ru[n + i] as f64 * sc, f.ru[2 * n + i] as f64 * sc, f.ru[3 * n + i] as f64 * sc], "rho": f.ru[i], "class": class(i),
            "flags": f.flags[i], "phi_mm": f.phi[i] as f64 * 1e3, "B": f.bsol[i], "wall_speed": ((f.um[i].powi(2) + f.um[n + i].powi(2) + f.um[2 * n + i].powi(2)).sqrt() as f64) * sc,
            "g": g, "x_mm": x.map(|v| v * 1e3), "valve": near(&x, &vc), "cut": near(&x, &cc),
        }));
    }
    Ok(json!({"step": s.step, "t": t, "fluid": all.len(), "p99": pct(&all, 0.99), "p999": pct(&all, 0.999), "p9999": pct(&all, 0.9999), "classes": by_class, "top": top}))
}

/// sparse snapshot: `<dir>/snap_<step>.f32` = f32 [6][nodes] (ρ, u_x, u_y, u_z in m/s, B, flags) and a JSON header with
/// the tile layout (node g = tile·8 + (i % 8, i / 8 % 8, i / 64), position origin + g dx)
pub fn snapshot<B: Backend>(s: &mut Solver<B>, dir: &Path) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(dir)?;
    let f = s.be.fields()?;
    let n = s.be.nodes();
    let sc = (s.u.dx / s.u.dt) as f32;
    let p = dir.join(format!("snap_{:07}.f32", s.step));
    let mut w = std::io::BufWriter::new(std::fs::File::create(&p)?);
    for c in 0..4 {
        for i in 0..n {
            let v = if c == 0 { f.ru[i] } else { f.ru[c * n + i] * sc };
            w.write_all(&v.to_le_bytes())?;
        }
    }
    for i in 0..n {
        w.write_all(&f.bsol[i].to_le_bytes())?;
    }
    for i in 0..n {
        w.write_all(&(f.flags[i] as f32).to_le_bytes())?;
    }
    w.flush()?;
    let hdr = json!({"step": s.step, "t": s.time(), "nodes": n, "fields": ["rho", "ux", "uy", "uz", "B", "flags"], "tiles": s.topo.ntiles, "tile_xyz": s.topo.txyz,
                     "origin": s.case.grid.origin, "dx": s.u.dx, "node_order": "i = (z·8 + y)·8 + x within a tile, tile-major"});
    std::fs::write(p.with_extension("json"), hdr.to_string())?;
    Ok(p)
}
