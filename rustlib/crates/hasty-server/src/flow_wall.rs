//! Wall shell of the blood-pool surfaces (`flow_wall_shell`): the outer surface of the tissue wall that surrounds the blood
//! (myocardium, thick vessel walls), estimated from the magnitude alone, per frame, on the vertices of the blood-pool mesh.
//!
//! For every vertex the magnitude is sampled along the outward normal (profile). The wall is a segment of roughly constant,
//! darker level directly outside the blood; behind it the level changes (fat: brighter, lung: darker, the other chamber's
//! blood: brighter). A change-point fit finds the thickness T that maximises the two-sample t statistic between the wall
//! segment and a window behind it. Then:
//! - profiles are averaged over the 1-ring of the mesh first (noise), the fit uses each vertex's own thickness;
//! - the t statistic is the confidence; vertices whose wall is not darker than `max_wall_rel` × the blood level beside them
//!   (fat directly on a vessel, no resolved wall), as dark as air / lung (`dark_thr`), or thinner than `min_mm` get confidence 0;
//! - thickness and confidence are diffused over the mesh as numerator / denominator (normalised convolution), so a
//!   confident neighbourhood fills the gaps, and `support` (the diffused weight, 0…1) says how far from evidence a value is;
//! - for transitions where the blood-pool mesh keeps its vertices (`links`) the fields are also smoothed over time;
//! - outer surface = inner vertex + T · normal, same triangles, a few Taubin steps.
//!
//! Where the wall touches tissue of the same level (muscle, liver) there is no edge in the data: confidence stays low there
//! and the viewer hides it by `support`. Nothing here uses labels.

use std::sync::Arc;

use hasty_value::{Tensor, Value};
use rayon::prelude::*;
use serde_json::{json, Value as Json};

use crate::commands::Registry;
use crate::flow::Flow;
use crate::mesh::Mesh;
use crate::Error;

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

/// trilinear sample of frame `t` of the magnitude at voxel coordinates `p` (clamped to the volume)
fn mag_at(f: &Flow, mag: &[f32], t: usize, p: [f64; 3]) -> f64 {
    let [nx, ny, nz] = f.dims;
    let c = [p[0].clamp(0.0, nx as f64 - 1.0), p[1].clamp(0.0, ny as f64 - 1.0), p[2].clamp(0.0, nz as f64 - 1.0)];
    let (i, j, k) = ((c[0] as usize).min(nx.saturating_sub(2)), (c[1] as usize).min(ny.saturating_sub(2)), (c[2] as usize).min(nz.saturating_sub(2)));
    let (fx, fy, fz) = (c[0] - i as f64, c[1] - j as f64, c[2] - k as f64);
    let base = t * nx * ny * nz;
    let at = |a: usize, b: usize, c2: usize| mag[base + (c2.min(nz - 1) * ny + b.min(ny - 1)) * nx + a.min(nx - 1)] as f64;
    let l = |a: f64, b: f64, w: f64| a * (1.0 - w) + b * w;
    let z0 = l(l(at(i, j, k), at(i + 1, j, k), fx), l(at(i, j + 1, k), at(i + 1, j + 1, k), fx), fy);
    let z1 = l(l(at(i, j, k + 1), at(i + 1, j, k + 1), fx), l(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), fx), fy);
    l(z0, z1, fz)
}

fn adjacency(m: &Mesh) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); m.verts.len()];
    for t in &m.faces {
        for k in 0..3 {
            adj[t[k] as usize].push(t[(k + 1) % 3]);
            adj[t[(k + 1) % 3] as usize].push(t[k]);
        }
    }
    adj.iter_mut().for_each(|a| {
        a.sort_unstable();
        a.dedup();
    });
    adj
}

/// per-vertex result of the change-point fit
#[derive(Clone, Copy, Default)]
struct Fit {
    t_mm: f64,
    conf: f64,
    wall_rel: f64,
    out_rel: f64,
    wall_abs: f64,
    in_abs: f64,
    /// local diameter of the blood pool behind this vertex (mm; 0 where there is no confident edge)
    chord_mm: f64,
}

pub struct ShellOpts {
    pub max_mm: f64,
    pub window_mm: f64,
    pub max_wall_rel: f64,
    pub cap: f64,
    pub conf_min: f64,
    pub diffuse: usize,
    pub temporal: bool,
    pub noise_rel: f64,
    pub min_mm: f64,
    pub max_wall_cv: f64,
    pub dark_thr: f64,
    pub max_chord_mm: f64,
    pub min_chord_mm: f64,
}

impl ShellOpts {
    fn from(o: &Json, dark_thr: f64) -> Self {
        Self {
            max_mm: o["max_mm"].as_f64().unwrap_or(20.0),
            window_mm: o["window_mm"].as_f64().unwrap_or(6.0),
            max_wall_rel: o["max_wall_rel"].as_f64().unwrap_or(0.7),
            cap: o["conf_cap"].as_f64().unwrap_or(12.0),
            conf_min: o["conf_min"].as_f64().unwrap_or(4.0),
            diffuse: o["diffuse"].as_u64().unwrap_or(24) as usize,
            temporal: o["temporal"].as_bool().unwrap_or(true),
            noise_rel: o["noise_rel"].as_f64().unwrap_or(0.03),
            min_mm: o["min_mm"].as_f64().unwrap_or(3.0),
            max_wall_cv: o["max_wall_cv"].as_f64().unwrap_or(0.5),
            dark_thr: o["dark_thr"].as_f64().unwrap_or(dark_thr),
            max_chord_mm: o["max_chord_mm"].as_f64().unwrap_or(80.0),
            min_chord_mm: o["min_chord_mm"].as_f64().unwrap_or(25.0),
        }
    }
}

/// fits of one frame: profiles along the normals of `m`, 1-ring averaged, change-point per vertex
fn fit_frame(f: &Flow, mag: &[f32], t: usize, m: &Mesh, adj: &[Vec<u32>], so: &ShellOpts) -> Vec<Fit> {
    let sp_mm = f.spacing.map(|s| s * 1e3);
    let minsp = sp_mm.iter().cloned().fold(f64::INFINITY, f64::min);
    let step = 0.5 * minsp;
    let s_min = -2.0 * minsp;
    let k_of = |s: f64| ((s - s_min) / step).round() as isize;
    let kk = k_of(so.max_mm + so.window_mm) as usize + 1;
    let normals = m.normals();
    // profiles [vertex][k]
    let prof: Vec<Vec<f32>> = (0..m.verts.len())
        .into_par_iter()
        .map(|v| {
            let (p, n) = (m.verts[v], normals[v]);
            (0..kk)
                .map(|k| {
                    let s = s_min + k as f64 * step;
                    let q = [0, 1, 2].map(|a| (p[a] as f64 + s * n[a] as f64) / sp_mm[a]);
                    mag_at(f, mag, t, q) as f32
                })
                .collect()
        })
        .collect();
    // 1-ring average, twice
    let mut cur = prof;
    for _ in 0..2 {
        let prev = &cur;
        let next: Vec<Vec<f32>> = (0..prev.len())
            .into_par_iter()
            .map(|v| {
                if adj[v].is_empty() {
                    return prev[v].clone();
                }
                let w = 1.0 / (adj[v].len() as f32 + 1.0);
                (0..kk).map(|k| (prev[v][k] + adj[v].iter().map(|&j| prev[j as usize][k]).sum::<f32>()) * w).collect()
            })
            .collect();
        cur = next;
    }
    let ka = k_of(minsp).max(0) as usize; // the wall starts one voxel outside the blood surface (partial volume)
    let kin0 = k_of(-2.0 * minsp).max(0) as usize;
    let kin1 = k_of(-0.5 * minsp).max(kin0 as isize + 1) as usize;
    let nb = (so.window_mm / step).round() as usize;
    let kmax_t = k_of(so.max_mm) as usize;
    let mut fits: Vec<Fit> = cur.par_iter()
        .map(|p| {
            let m_in = p[kin0..=kin1].iter().map(|&x| x as f64).sum::<f64>() / (kin1 - kin0 + 1) as f64;
            if m_in <= 0.0 {
                return Fit::default();
            }
            // prefix sums
            let mut s1 = vec![0f64; kk + 1];
            let mut s2 = vec![0f64; kk + 1];
            for k in 0..kk {
                s1[k + 1] = s1[k] + p[k] as f64;
                s2[k + 1] = s2[k] + (p[k] as f64).powi(2);
            }
            let floor = so.noise_rel * m_in;
            let mut best = Fit::default();
            let mut best_kt = 0usize;
            for kt in (ka + 2)..=kmax_t.min(kk - nb.max(2)) {
                let (n1, n2) = ((kt - ka) as f64, nb.min(kk - kt) as f64);
                if n2 < 2.0 {
                    break;
                }
                let k2 = kt + n2 as usize;
                let (a1, a2) = (s1[kt] - s1[ka], s1[k2] - s1[kt]);
                let (mu1, mu2) = (a1 / n1, a2 / n2);
                let sse1 = (s2[kt] - s2[ka] - a1 * a1 / n1).max(0.0);
                if mu1 > 0.0 && (sse1 / (n1 - 1.0)).sqrt() > so.max_wall_cv * mu1 {
                    continue; // the wall segment is not a layer of roughly constant brightness
                }
                let sse = sse1 + (s2[k2] - s2[kt] - a2 * a2 / n2);
                let sd = (sse / (n1 + n2 - 2.0)).max(0.0).sqrt().max(floor);
                let tstat = (mu1 - mu2).abs() / (sd * (1.0 / n1 + 1.0 / n2).sqrt());
                if tstat > best.conf {
                    best_kt = kt;
                    best = Fit { t_mm: s_min + (kt as f64 - 0.5) * step, conf: tstat, wall_rel: mu1 / m_in, out_rel: mu2 / m_in, wall_abs: mu1, in_abs: m_in, chord_mm: 0.0 };
                }
            }
            if best.wall_rel > so.max_wall_rel {
                best.conf = 0.0; // not darker than the blood beside it: no resolved wall here
            }
            if best.wall_abs <= so.dark_thr {
                best.conf = 0.0; // as dark as air / lung: that is not a wall
            }
            if best.t_mm < so.min_mm || best_kt <= ka + 2 {
                best.conf = 0.0; // thinner than the data can resolve (vessel walls), or the edge sits at the first position the fit can place one
            }
            best
        })
        .collect();
    // local lumen diameter behind every confident vertex: how far inwards the magnitude stays near the blood level (halfway between
    // blood and wall), two samples in a row below it end the chord. A chamber is wide, a vessel is not.
    let nk = (so.max_chord_mm / minsp).ceil() as usize;
    let raw: Vec<f64> = (0..fits.len())
        .into_par_iter()
        .map(|v| {
            let ft = fits[v];
            if ft.conf < so.conf_min {
                return 0.0;
            }
            let level = 0.5 * (ft.in_abs + ft.wall_abs);
            let (p, n) = (m.verts[v], normals[v]);
            let val = |k: usize| {
                let s = -(k as f64) * minsp;
                mag_at(f, mag, t, [0, 1, 2].map(|a| (p[a] as f64 + s * n[a] as f64) / sp_mm[a]))
            };
            let mut below = 0;
            for k in 2..=nk {
                if val(k) < level {
                    below += 1;
                    if below >= 2 {
                        return (k - 1) as f64 * minsp;
                    }
                } else {
                    below = 0;
                }
            }
            so.max_chord_mm
        })
        .collect();
    for v in 0..fits.len() {
        if raw[v] > 0.0 {
            let mut c: Vec<f64> = std::iter::once(raw[v]).chain(adj[v].iter().map(|&j| raw[j as usize]).filter(|&x| x > 0.0)).collect();
            c.sort_by(|a, b| a.partial_cmp(b).unwrap());
            fits[v].chord_mm = c[c.len() / 2];
        }
    }
    fits
}

/// outer surfaces of the wall shells for the blood-pool meshes `meshes` (frame i of the flow value ↔ meshes[i])
pub fn wall_shell(f: &Flow, meshes: &[Mesh], links: &[bool], so: &ShellOpts) -> Result<(Vec<Mesh>, Json), Error> {
    let mag = f.mag.as_ref().ok_or_else(|| Error::Invalid("flow_wall_shell: the flow value needs a magnitude (mag)".into()))?;
    if meshes.len() != f.nf {
        return inv(format!("flow_wall_shell: {} mesh frames for {} flow frames", meshes.len(), f.nf));
    }
    let adjs: Vec<Vec<Vec<u32>>> = meshes.iter().map(adjacency).collect();
    let fits: Vec<Vec<Fit>> = meshes.iter().enumerate().map(|(t, m)| fit_frame(f, mag, t, m, &adjs[t], so)).collect();
    // numerator / denominator fields: weight = confidence (capped, 0…1), value = thickness
    let w0: Vec<Vec<f64>> = fits.iter().map(|fs| fs.iter().map(|x| ((x.conf - so.conf_min) / (so.cap - so.conf_min)).clamp(0.0, 1.0)).collect()).collect();
    let mut num: Vec<Vec<f64>> = fits.iter().zip(&w0).map(|(fs, w)| fs.iter().zip(w).map(|(x, w)| x.t_mm * w).collect()).collect();
    let mut den: Vec<Vec<f64>> = w0.clone();
    let diffuse = |v: &mut Vec<f64>, adj: &[Vec<u32>]| {
        for _ in 0..so.diffuse {
            let prev = v.clone();
            for (i, x) in v.iter_mut().enumerate() {
                if adj[i].is_empty() {
                    continue;
                }
                let m = adj[i].iter().map(|&j| prev[j as usize]).sum::<f64>() / adj[i].len() as f64;
                *x = 0.5 * prev[i] + 0.5 * m;
            }
        }
    };
    for t in 0..meshes.len() {
        diffuse(&mut num[t], &adjs[t]);
        diffuse(&mut den[t], &adjs[t]);
    }
    if so.temporal {
        let (n0, d0) = (num.clone(), den.clone());
        for t in 0..meshes.len() {
            for (nbr, ok) in [(t.wrapping_sub(1), t > 0 && links.get(t - 1) == Some(&true)), (t + 1, t + 1 < meshes.len() && links.get(t) == Some(&true))] {
                if ok {
                    for i in 0..num[t].len() {
                        num[t][i] += 0.5 * n0[nbr][i];
                        den[t][i] += 0.5 * d0[nbr][i];
                    }
                }
            }
        }
    }
    let mut out = Vec::with_capacity(meshes.len());
    let mut stats = Vec::new();
    for (t, m) in meshes.iter().enumerate() {
        let nrm = m.normals();
        let nv = m.verts.len();
        // support: the diffused weight relative to a typical well-supported neighbourhood (median over the vertices with a confident edge)
        let mut dc: Vec<f64> = (0..nv).filter(|&i| fits[t][i].conf >= so.conf_min).map(|i| den[t][i]).collect();
        dc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let dref = dc.get(dc.len() / 2).copied().unwrap_or(1.0).max(1e-9);
        let thick: Vec<f32> = (0..nv).map(|i| if den[t][i] > 1e-6 * dref { (num[t][i] / den[t][i]) as f32 } else { 0.0 }).collect();
        // chamber scale: the local lumen diameter (chord), averaged over the neighbourhood (a chamber has wide patches, a vessel does not);
        // support fades to 0 below min_chord_mm, so cuffs around vessels are hidden. The wall of a vessel is below the resolution anyway.
        let adj = &adjs[t];
        let mut cn: Vec<f64> = fits[t].iter().map(|x| x.chord_mm).collect();
        let mut cd: Vec<f64> = fits[t].iter().map(|x| (x.chord_mm > 0.0) as u8 as f64).collect();
        for v in [&mut cn, &mut cd] {
            for _ in 0..40 {
                let prev = v.clone();
                for (i, x) in v.iter_mut().enumerate() {
                    if !adj[i].is_empty() {
                        *x = 0.5 * prev[i] + 0.5 * adj[i].iter().map(|&j| prev[j as usize]).sum::<f64>() / adj[i].len() as f64;
                    }
                }
            }
        }
        let chord_s: Vec<f32> = (0..nv).map(|i| if cd[i] > 1e-6 { (cn[i] / cd[i]) as f32 } else { 0.0 }).collect();
        let lo = 0.8 * so.min_chord_mm;
        let support: Vec<f32> = (0..nv).map(|i| (den[t][i] / dref).min(1.0) as f32 * ((chord_s[i] as f64 - lo) / (so.min_chord_mm - lo).max(1e-9)).clamp(0.0, 1.0) as f32).collect();
        let verts: Vec<[f32; 3]> = (0..nv).map(|i| [0, 1, 2].map(|a| m.verts[i][a] + thick[i] * nrm[i][a])).collect();
        let mut o = Mesh { verts, faces: m.faces.clone(), scalars: vec![], space: m.space.clone(), affine: m.affine, spacing_mm: m.spacing_mm };
        o.taubin(3, 0.5, -0.53);
        o.scalars = vec![
            ("thickness_mm".into(), thick.clone()),
            ("confidence".into(), fits[t].iter().map(|x| x.conf as f32).collect()),
            ("support".into(), support.clone()),
            ("wall_rel".into(), fits[t].iter().map(|x| x.wall_rel as f32).collect()),
            ("out_rel".into(), fits[t].iter().map(|x| x.out_rel as f32).collect()),
            ("chord_mm".into(), chord_s.clone()),
        ];
        let good: Vec<f32> = (0..nv).filter(|&i| fits[t][i].conf >= 4.0).map(|i| thick[i]).collect();
        let mut g = good.clone();
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        stats.push(json!({"confident_vertices": good.len(), "vertices": nv, "median_thickness_mm": g.get(g.len() / 2), "p10": g.get(g.len() / 10), "p90": g.get(g.len() * 9 / 10)}));
        out.push(o);
    }
    Ok((out, json!({ "frames": stats })))
}

pub fn register(r: &mut Registry) {
    r.register(
        "flow_wall_shell",
        "outer surface of the wall around the blood pool (myocardium, thick vessel walls) from the magnitude alone: inputs [flow value with mag, mesh_seq from flow_bloodpool]; options {max_mm (20), window_mm (6: the level behind the wall is compared over this much), max_wall_rel (0.7: a wall must be darker than this × the blood level beside it), max_wall_cv (0.5: the wall segment must be a layer of roughly constant brightness: its standard deviation at most this × its mean), min_chord_mm (25: the wall is shown only where the lumen behind it is at least this wide, averaged over the neighbourhood: chambers, not vessels; support fades out below), max_chord_mm (80), min_mm (3: a thinner wall is below what the data resolves), dark_thr (default: the one flow_bloodpool found between air / lung and dark tissue; a wall must be brighter than this), conf_cap (12), conf_min (4: edges with a smaller t statistic carry no weight), diffuse (24 smoothing steps over the mesh), temporal (true), noise_rel (0.03)}; output mesh_seq like the input (same links, dt) whose vertices are the outer wall surface, scalars thickness_mm, confidence (t statistic of the edge, 0 = no edge), support (0…1: how close to evidence the diffused thickness is), wall_rel, out_rel (levels relative to the blood)",
        |_, o, inp: &[Arc<Value>]| {
            let f = Flow::from_value(inp.first().ok_or_else(|| Error::Invalid("flow_wall_shell: input 0 = flow value".into()))?)?;
            let seq = inp.get(1).ok_or_else(|| Error::Invalid("flow_wall_shell: input 1 = mesh_seq (flow_bloodpool)".into()))?;
            let Some(Value::List(fr)) = seq.get("frames") else { return inv("flow_wall_shell: input 1 is not a mesh_seq") };
            let meshes: Vec<Mesh> = fr.iter().map(Mesh::from_value).collect::<Result<_, _>>()?;
            let links: Vec<bool> = seq.get("links").and_then(|t| t.as_tensor()).map(|t| (0..t.numel()).map(|i| t.get_f64(i).0 != 0.0).collect()).unwrap_or_default();
            let air = match seq.get("dark_thr") { Some(Value::Float(a)) => *a, _ => 0.0 };
            let (shell, msg) = wall_shell(&f, &meshes, &links, &ShellOpts::from(o, air))?;
            let lk: Vec<u8> = links.iter().map(|&b| b as u8).collect();
            let v = Value::Dict(vec![
                ("kind".into(), Value::Str("mesh_seq".into())),
                ("coherent".into(), Value::Bool(links.iter().all(|&b| b))),
                ("links".into(), Value::Tensor(Tensor::from_slice(&[lk.len()], &lk)?)),
                ("frames".into(), Value::List(shell.iter().zip(fr).map(|(m, src)| {
                    let mut d = m.to_value();
                    if let (Value::Dict(dv), Some(p)) = (&mut d, src.get("parent")) {
                        dv.push(("parent".into(), p.clone()));
                    }
                    d
                }).collect())),
                ("dt".into(), Value::Float(f.dt)),
                ("source".into(), Value::Str("flow_wall_shell".into())),
            ]);
            Ok((msg.to_string(), vec![v]))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Ctx;

    /// ball of blood (radius 8 voxels, mag 1, moving) in a wall of 4 voxels (mag 0.35); behind it: fat (0.8) on the +x half,
    /// lung (0.05) on the −x half-space beyond x < centre, and in the y > centre + 0 quarter plain muscle at the wall level (no edge)
    fn shelled(n: usize, thick: f64) -> (Value, f64) {
        let c = (n as f64 - 1.0) / 2.0;
        let nv = n * n * n;
        let nf = 3;
        let r_in = 8.0;
        let mut mag = vec![0f32; nf * nv];
        let mut vel = vec![0f32; nf * 3 * nv];
        for t in 0..nf {
            for q in 0..nv {
                let (i, j, k) = ((q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64);
                let (dx, dy, dz) = (i - c, j - c, k - c);
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                let m = if r <= r_in { 1.0 } else if r <= r_in + thick { 0.35 } else if r > r_in + thick && r <= 22.0 {
                    if dy > 6.0 { 0.35 } else if dx >= 0.0 { 0.8 } else { 0.05 }
                } else { 0.0 };
                mag[t * nv + q] = m;
                if r <= r_in {
                    vel[(t * 3) * nv + q] = 0.6;
                }
            }
        }
        let v = Value::Dict(vec![
            ("kind".into(), Value::Str("flow4d".into())),
            ("vel".into(), Value::Tensor(Tensor::from_slice(&[nf, 3, n, n, n], &vel).unwrap())),
            ("mag".into(), Value::Tensor(Tensor::from_slice(&[nf, n, n, n], &mag).unwrap())),
            ("spacing".into(), Value::List(vec![Value::Float(2e-3); 3])),
            ("dt".into(), Value::Float(0.1)),
        ]);
        (v, thick * 2.0)
    }

    #[test]
    fn wall_thickness_from_the_magnitude_profile() {
        let (v, truth_mm) = shelled(56, 4.0);
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let flow = Arc::new(v);
        let (_, pool) = reg.run("flow_bloodpool", &ctx, "{}", &[flow.clone()]).unwrap();
        let pool = Arc::new(pool[0].clone());
        let (msg, out) = reg.run("flow_wall_shell", &ctx, "{}", &[flow, pool.clone()]).unwrap();
        eprintln!("{msg}");
        let Some(Value::List(fr)) = out[0].get("frames") else { panic!() };
        let Some(Value::List(src)) = pool.get("frames") else { panic!() };
        let m = Mesh::from_value(&fr[1]).unwrap();
        let inner = Mesh::from_value(&src[1]).unwrap();
        let sc = |k: &str| m.scalars.iter().find(|(n, _)| n == k).unwrap().1.clone();
        let (thick, conf, support) = (sc("thickness_mm"), sc("confidence"), sc("support"));
        // vertex classes by direction from the centre: fat side (+x, away from the muscle quarter), lung side (−x), muscle quarter (+y)
        let c = 27.5 * 2.0;
        let (mut fat, mut lung, mut muscle) = (vec![], vec![], vec![]);
        let (mut cf, mut cl, mut cm) = (vec![], vec![], vec![]);
        for (i, p) in inner.verts.iter().enumerate() {
            let d = [p[0] as f64 - c, p[1] as f64 - c, p[2] as f64 - c];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let u = [d[0] / r, d[1] / r, d[2] / r];
            if u[1] > 0.8 { muscle.push(thick[i]); cm.push(conf[i]); }
            else if u[0] > 0.6 && u[1].abs() < 0.5 { fat.push(thick[i]); cf.push(conf[i]); }
            else if u[0] < -0.6 && u[1].abs() < 0.5 { lung.push(thick[i]); cl.push(conf[i]); }
        }
        {
            let (wr, orl) = (sc("wall_rel"), sc("out_rel"));
            let i = (0..inner.verts.len()).max_by(|&a, &b| inner.verts[a][0].partial_cmp(&inner.verts[b][0]).unwrap()).unwrap();
            let nrm = inner.normals()[i];
            eprintln!("+x vertex {:?} normal {:?} thickness {} conf {} wall_rel {} out_rel {}", inner.verts[i], nrm, thick[i], conf[i], wr[i], orl[i]);
        }
        let med = |v: &mut Vec<f32>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] as f64 };
        let (mf, ml, mm) = (med(&mut fat), med(&mut lung), med(&mut muscle));
        eprintln!("thickness (truth {truth_mm} mm): fat side {mf:.2}, lung side {ml:.2}, muscle quarter {mm:.2}; confidence medians {:.1} {:.1} {:.1}", med(&mut cf), med(&mut cl), med(&mut cm));
        assert!((mf - truth_mm).abs() < 1.6, "fat side {mf}");
        assert!((ml - truth_mm).abs() < 1.6, "lung side {ml}");
        // no edge in the muscle quarter: low confidence there, thickness filled from the neighbours (around the truth, never wild)
        assert!(med(&mut cm.clone()) < 0.5 * med(&mut cf.clone()), "muscle quarter must have lower confidence");
        assert!(mm > 0.5 * truth_mm && mm < 1.6 * truth_mm, "muscle quarter filled from the neighbours: {mm}");
        assert!(support.iter().all(|&s| (0.0..=1.0).contains(&s)));
        assert_eq!(m.faces.len(), inner.faces.len());
    }

    #[test]
    fn a_wall_thinner_than_the_resolution_is_not_reported() {
        // 1-voxel (2 mm) wall: below min_mm = 3 → no confident vertex, the shell is hidden by support everywhere
        let (v, _) = shelled(56, 1.0);
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let flow = Arc::new(v);
        let (_, pool) = reg.run("flow_bloodpool", &ctx, "{}", &[flow.clone()]).unwrap();
        let (msg, out) = reg.run("flow_wall_shell", &ctx, "{}", &[flow, Arc::new(pool[0].clone())]).unwrap();
        // the clean sectors (straight towards the fat and towards the lung): a 2 mm wall is below min_mm / the first position an edge can be placed
        let Some(Value::List(fr)) = out[0].get("frames") else { panic!() };
        let Some(Value::List(src)) = pool[0].get("frames") else { panic!() };
        let (m, inner) = (Mesh::from_value(&fr[1]).unwrap(), Mesh::from_value(&src[1]).unwrap());
        let conf = m.scalars.iter().find(|(n, _)| n == "confidence").unwrap().1.clone();
        let c = 55.0f64;
        let mut n_sector = 0;
        for (i, p) in inner.verts.iter().enumerate() {
            let d = [p[0] as f64 - c, p[1] as f64 - c, p[2] as f64 - c];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if (d[1] / r).abs() < 0.25 && (d[0] / r).abs() > 0.9 {
                n_sector += 1;
                assert!(conf[i] < 4.0, "vertex {i} towards {:.2} has confidence {}", d[0] / r, conf[i]);
            }
        }
        assert!(n_sector > 40, "{n_sector}");
        let _ = msg;
    }

    #[test]
    fn a_vessel_gets_no_wall_shell_a_chamber_does() {
        // tube of radius 3 voxels (chord 12 mm) with a 4-voxel dark layer and fat behind: a clear edge, but not chamber scale
        let n = 56usize;
        let (c, nv, nf) = (27.5f64, 56 * 56 * 56, 3usize);
        let mut mag = vec![0f32; nf * nv];
        let mut vel = vec![0f32; nf * 3 * nv];
        for t in 0..nf {
            for q in 0..nv {
                let (i, j, k) = ((q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64);
                let r = ((j - c).powi(2) + (k - c).powi(2)).sqrt();
                let _ = i;
                mag[t * nv + q] = if r <= 3.0 { 1.0 } else if r <= 7.0 { 0.35 } else if r <= 20.0 { 0.8 } else { 0.0 };
                if r <= 3.0 {
                    vel[(t * 3) * nv + q] = 0.6;
                }
            }
        }
        let v = Value::Dict(vec![
            ("kind".into(), Value::Str("flow4d".into())),
            ("vel".into(), Value::Tensor(Tensor::from_slice(&[nf, 3, n, n, n], &vel).unwrap())),
            ("mag".into(), Value::Tensor(Tensor::from_slice(&[nf, n, n, n], &mag).unwrap())),
            ("spacing".into(), Value::List(vec![Value::Float(2e-3); 3])),
            ("dt".into(), Value::Float(0.1)),
        ]);
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let flow = Arc::new(v);
        let (_, pool) = reg.run("flow_bloodpool", &ctx, "{}", &[flow.clone()]).unwrap();
        let (flow2, pool2) = (flow.clone(), Arc::new(pool[0].clone()));
        let (_, out) = reg.run("flow_wall_shell", &ctx, "{}", &[flow, pool2.clone()]).unwrap();
        let Some(Value::List(fr)) = out[0].get("frames") else { panic!() };
        let m = Mesh::from_value(&fr[1]).unwrap();
        let sc = |k: &str| m.scalars.iter().find(|(n, _)| n == k).unwrap().1.clone();
        let (conf, support, chord) = (sc("confidence"), sc("support"), sc("chord_mm"));
        let edges = conf.iter().filter(|&&c| c >= 4.0).count();
        assert!(edges > m.verts.len() / 4, "the tube has a clear wall edge in the data ({edges} of {})", m.verts.len());
        assert!(support.iter().all(|&s| s < 0.5), "no vessel-scale shell may be shown, max support {}", support.iter().cloned().fold(0.0, f32::max));
        let mc = chord.iter().cloned().filter(|&c| c > 0.0).fold(0.0, f32::max);
        assert!(mc < 20.0, "tube chord ≈ 12 mm, got max {mc}");
        // the same data with the chamber-scale requirement switched off shows the shell
        let (_, out2) = reg.run("flow_wall_shell", &ctx, r#"{"min_chord_mm": 1}"#, &[flow2, pool2]).unwrap();
        let Some(Value::List(fr2)) = out2[0].get("frames") else { panic!() };
        let m2 = Mesh::from_value(&fr2[1]).unwrap();
        let sup2 = m2.scalars.iter().find(|(n, _)| n == "support").unwrap().1.clone();
        assert!(sup2.iter().filter(|&&s| s >= 0.5).count() > m2.verts.len() / 4, "without the chord requirement the tube's shell is shown");
    }
}
