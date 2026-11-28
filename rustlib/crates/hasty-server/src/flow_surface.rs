//! Moving blood-pool surfaces of a flow value, from the data alone (`vel`, `mag`, optional `mask`): no atlas, no labels.
//!
//! `flow_bloodpool` classifies every voxel of every frame as blood or not, cleans the result in space and time and hands
//! the 4-D indicator to the coherent surface tracker (`meshseq_track`: one topology for the whole cycle, vertices follow
//! the wall, `links` tells where a split / merge — a valve closing — starts a new surface run).
//!
//! **Classifier** (all thresholds come from the data; each can be overridden):
//! - dark | bright: Otsu on the magnitude histogram of all frames (`air_thr`: air, lung, muscle, myocardium | fat, blood); inside the
//!   dark class a second Otsu separates air and lung from dark tissue (`dark_thr`, used by the wall shell).
//! - "moving": the speed is weighted by min(1, magnitude / m_ref) (phase noise is inversely proportional to the magnitude),
//!   `speed_thr` = q50 + 3 (q50 − q10) of the speeds of the non-air voxels (mostly tissue: noise + slow motion;
//!   about 3.8 σ for Gaussian noise), at least 3 % of their 99.9th percentile (noise-free tissue has speed 0).
//! - where blood flows at all: voxels faster than `speed_thr` in at least max(2, 10 %) of the frames, minus specks, plus a
//!   reach of `reach_mm` (walls move, chambers change size). A global magnitude threshold cannot separate blood from fat and
//!   skin (both bright in a gradient echo image), so the magnitude is classified only inside this neighbourhood, where
//!   the competitors are wall and adjacent tissue: Otsu on its non-air voxels, but never below the 10th percentile of the
//!   magnitude of the voxels that demonstrably flow (fat next to a vessel can be nearly as bright as blood) (`mag_thr`).
//! - blood in frame f = (inside the neighbourhood and brighter than `mag_thr`) or (non-air and faster than `speed_thr` in
//!   frame f: dephased or saturated blood is dark but moves).
//! - without `mag`: the mask when the value has one, otherwise Otsu on the speed (a warning says so).
//!   Blood that never moves faster than the threshold (an occluded vessel) is not found.
//!
//! **Clean-up:** connected components (6-connected) smaller than `min_voxels` are dropped unless they have at least
//! `min_rescue` voxels and mostly lie where blood is present in `persist` of the frames (a small real vessel is there
//! every frame, noise is not); holes enclosed in blood and
//! smaller than `min_hole` voxels are filled. Larger enclosed holes (papillary muscles, trabeculae) are kept.
//!
//! The classifier is deliberately simple: it is the baseline that is scored on phantoms against the truth lumen
//! (`rustlib/tools/flow/score_bloodpool.py`) before anything smarter replaces it.

use std::sync::Arc;

use hasty_value::{Tensor, Value};
use serde_json::{json, Value as Json};

use crate::commands::Registry;
use crate::flow::Flow;
use crate::mesh_ops::track_sequence;
use crate::Error;

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

/// Otsu threshold (bin index: values `< t` are the lower class) of `hist`
fn otsu(hist: &[f64]) -> usize {
    let total: f64 = hist.iter().sum();
    let sum: f64 = hist.iter().enumerate().map(|(i, &h)| i as f64 * h).sum();
    let (mut w0, mut s0, mut best, mut bt) = (0.0, 0.0, -1.0, 0);
    for (i, &h) in hist.iter().enumerate() {
        w0 += h;
        s0 += i as f64 * h;
        let w1 = total - w0;
        if w0 <= 0.0 || w1 <= 0.0 {
            continue;
        }
        let (m0, m1) = (s0 / w0, (sum - s0) / w1);
        let var = w0 * w1 * (m0 - m1) * (m0 - m1);
        if var > best {
            best = var;
            bt = i + 1;
        }
    }
    bt
}

struct Hist {
    lo: f64,
    step: f64,
    bins: Vec<f64>,
}

impl Hist {
    fn new(lo: f64, hi: f64, n: usize) -> Self {
        Self { lo, step: (hi - lo).max(1e-12) / n as f64, bins: vec![0.0; n] }
    }
    fn add(&mut self, x: f64) {
        let i = (((x - self.lo) / self.step) as isize).clamp(0, self.bins.len() as isize - 1) as usize;
        self.bins[i] += 1.0;
    }
    fn value(&self, bin: usize) -> f64 {
        self.lo + bin as f64 * self.step
    }
    fn quantile(&self, q: f64) -> f64 {
        let total: f64 = self.bins.iter().sum();
        let mut c = 0.0;
        for (i, &h) in self.bins.iter().enumerate() {
            c += h;
            if c >= q * total {
                return self.value(i + 1);
            }
        }
        self.value(self.bins.len())
    }
}

/// 6-connected components of `m != 0` ([Z,Y,X] bytes): labels (0 = none, 1…) and sizes (index = label − 1)
fn components(m: &[u8], dims: [usize; 3]) -> (Vec<u32>, Vec<usize>) {
    let [nx, ny, nz] = dims;
    let mut lab = vec![0u32; m.len()];
    let mut sizes = Vec::new();
    let mut stack = Vec::new();
    for s in 0..m.len() {
        if m[s] == 0 || lab[s] != 0 {
            continue;
        }
        let id = sizes.len() as u32 + 1;
        let mut n = 0usize;
        lab[s] = id;
        stack.push(s);
        while let Some(q) = stack.pop() {
            n += 1;
            let (i, j, k) = (q % nx, (q / nx) % ny, q / (nx * ny));
            let mut visit = |p: usize| {
                if m[p] != 0 && lab[p] == 0 {
                    lab[p] = id;
                    stack.push(p);
                }
            };
            if i > 0 { visit(q - 1); }
            if i + 1 < nx { visit(q + 1); }
            if j > 0 { visit(q - nx); }
            if j + 1 < ny { visit(q + nx); }
            if k > 0 { visit(q - nx * ny); }
            if k + 1 < nz { visit(q + nx * ny); }
        }
        sizes.push(n);
    }
    (lab, sizes)
}

/// `m` dilated by `r` steps of the 6-neighbourhood
fn dilate(m: &[u8], dims: [usize; 3], r: usize) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let mut cur = m.to_vec();
    for _ in 0..r {
        let mut nxt = cur.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let q = (k * ny + j) * nx + i;
                    if cur[q] != 0 {
                        continue;
                    }
                    if (i > 0 && cur[q - 1] != 0) || (i + 1 < nx && cur[q + 1] != 0) || (j > 0 && cur[q - nx] != 0) || (j + 1 < ny && cur[q + nx] != 0) || (k > 0 && cur[q - nx * ny] != 0) || (k + 1 < nz && cur[q + nx * ny] != 0) {
                        nxt[q] = 1;
                    }
                }
            }
        }
        cur = nxt;
    }
    cur
}

/// the indicator [T, Z, Y, X] (u8) and a description of what was done
pub struct BloodPool {
    pub mask: Vec<u8>,
    pub info: Json,
}

pub fn blood_pool(f: &Flow, o: &Json) -> Result<BloodPool, Error> {
    let nv = f.nv();
    let nf = f.nf;
    let dims = f.dims;
    let mut warnings: Vec<String> = Vec::new();
    let speed = |t: usize, q: usize| -> f32 { (0..3).map(|c| f.vel[(t * 3 + c) * nv + q].powi(2)).sum::<f32>().sqrt() };
    let mut raw = vec![0u8; nf * nv];
    let (mut t1, mut t2, mut speed_thr) = (f64::NAN, f64::NAN, f64::NAN);
    let mut mag_parts: Option<(f64, f64)> = None;
    let mut dark_thr = f64::NAN;
    let mut m_ref = f64::NAN;
    let source;
    if let Some(mag) = &f.mag {
        source = "mag+vel";
        let stride = (mag.len() / 8_000_000).max(1);
        let hi = {
            let mut h = Hist::new(0.0, mag.iter().cloned().fold(0.0f32, f32::max).max(1e-12) as f64, 4096);
            mag.iter().step_by((mag.len() / 4_000_000).max(1)).for_each(|&x| h.add(x as f64));
            h.quantile(0.999)
        };
        // 1. air / lung: Otsu on the magnitude histogram of all frames
        let mut h = Hist::new(0.0, hi, 256);
        mag.iter().step_by(stride).for_each(|&x| h.add(x as f64));
        t1 = o["air_thr"].as_f64().unwrap_or_else(|| h.value(otsu(&h.bins)));
        // the dark class (below t1) is air / lung / some muscle: its own Otsu split separates air and lung from dark tissue (the wall shell needs it)
        let mut hd = Hist::new(0.0, t1.max(1e-12), 128);
        mag.iter().step_by(stride).filter(|&&x| (x as f64) < t1).for_each(|&x| hd.add(x as f64));
        dark_thr = hd.value(otsu(&hd.bins));
        // The phase noise of a voxel is inversely proportional to its magnitude (σ_v = venc / (π SNR)), so a raw speed threshold is dominated by dark tissue and
        // air-like voxels. Speed is therefore weighted by min(1, magnitude / m_ref), m_ref = the 95th percentile of the non-air magnitude (bright blood / fat):
        // the same noise level everywhere, in m/s of a bright voxel. `norm_speed: false` uses the plain speed.
        m_ref = {
            let mut hn = Hist::new(0.0, hi, 1024);
            mag.iter().step_by(stride).filter(|&&x| x as f64 > t1).for_each(|&x| hn.add(x as f64));
            hn.quantile(0.95).max(1e-12)
        };
        let norm = o["norm_speed"].as_bool().unwrap_or(true);
        let nspeed = |t: usize, q: usize| -> f32 { speed(t, q) * if norm { (mag[t * nv + q] as f64 / m_ref).min(1.0) as f32 } else { 1.0 } };
        // 2. what "moving" means: the speed distribution of the non-air voxels is mostly tissue (noise + slow motion), blood is a minor
        //    part, so the low quantiles describe the tissue. Noise-free tissue (a simulation) has speed 0: never below 3 % of the fast end.
        speed_thr = match o["speed_thr"].as_f64() {
            Some(s) => s,
            None => {
                let mut hs = Hist::new(0.0, 4.0, 4096);
                for t in 0..nf {
                    for q in (0..nv).step_by(stride) {
                        if mag[t * nv + q] as f64 > t1 {
                            hs.add(nspeed(t, q) as f64);
                        }
                    }
                }
                let (q50, q10) = (hs.quantile(0.5), hs.quantile(0.1));
                (q50 + 3.0 * (q50 - q10)).max(0.03 * hs.quantile(0.999))
            }
        };
        // 3. where blood flows at all: voxels that are fast in at least max(2, 10 %) of the frames, minus specks, then a reach around them
        //    (walls move, chambers change size) — the magnitude is only classified inside this neighbourhood, where the competitors
        //    are the wall and the surrounding tissue, not all fat and skin of the body
        let need = ((0.1 * nf as f64).ceil() as usize).max(2).min(nf) as u16;
        let mut fast = vec![0u16; nv];
        for t in 0..nf {
            for q in 0..nv {
                fast[q] += (mag[t * nv + q] as f64 > t1 && nspeed(t, q) as f64 >= speed_thr) as u16;
            }
        }
        let ever: Vec<u8> = fast.iter().map(|&c| (c >= need) as u8).collect();
        let min_comp = o["min_voxels"].as_u64().unwrap_or(200) as usize;
        let (lab, sizes) = components(&ever, dims);
        let ever: Vec<u8> = lab.iter().map(|&l| (l != 0 && sizes[l as usize - 1] >= min_comp) as u8).collect();
        let reach_mm = o["reach_mm"].as_f64().unwrap_or(6.0);
        let reach = ((reach_mm / (f.spacing.iter().cloned().fold(f64::INFINITY, f64::min) * 1e3)).round() as usize).max(1);
        let near = dilate(&ever, dims, reach);
        let n_near = near.iter().map(|&v| v as usize).sum::<usize>();
        if n_near == 0 {
            return inv("flow_bloodpool: no voxel moves faster than the speed threshold in at least two frames: lower options.speed_thr");
        }
        // 4. blood / wall: Otsu on the magnitude of the non-air voxels of that neighbourhood
        let mut h2 = Hist::new(0.0, hi, 256);
        for t in 0..nf {
            for q in 0..nv {
                let m = mag[t * nv + q] as f64;
                if near[q] != 0 && m > t1 && (t + q) % (stride.max(1)) == 0 {
                    h2.add(m);
                }
            }
        }
        // blood is not much darker than the voxels that demonstrably flow: fat next to a vessel can be nearly as bright as blood, and
        // Otsu alone would put the threshold between the wall and the fat. The 10th percentile of the magnitude of the fast voxels is a floor.
        let mut h3 = Hist::new(0.0, hi, 256);
        for t in 0..nf {
            for q in 0..nv {
                if ever[q] != 0 && (t + q) % stride.max(1) == 0 && nspeed(t, q) as f64 >= speed_thr {
                    h3.add(mag[t * nv + q] as f64);
                }
            }
        }
        let otsu_thr = h2.value(otsu(&h2.bins));
        let flow_floor = h3.quantile(0.10);
        t2 = o["mag_thr"].as_f64().unwrap_or_else(|| otsu_thr.max(flow_floor));
        mag_parts = Some((otsu_thr, flow_floor));
        for t in 0..nf {
            for q in 0..nv {
                let m = mag[t * nv + q] as f64;
                raw[t * nv + q] = ((near[q] != 0 && m >= t2) || (m > t1 && nspeed(t, q) as f64 >= speed_thr)) as u8;
            }
        }
        warnings.push(format!("blood is classified only within {reach_mm} mm of voxels that move faster than {speed_thr:.3} m/s in at least {need} frames ({n_near} voxels)"));
    } else if let Some(mask) = &f.mask {
        source = "mask";
        warnings.push("no magnitude: the (static) mask is used for every frame".into());
        for t in 0..nf {
            raw[t * nv..(t + 1) * nv].copy_from_slice(mask);
        }
    } else {
        source = "speed";
        warnings.push("no magnitude and no mask: blood = speed above its Otsu threshold (moving blood only)".into());
        let hi = (0..nf).flat_map(|t| (0..nv).map(move |q| (t, q))).step_by(97).map(|(t, q)| speed(t, q)).fold(0.0f32, f32::max).max(1e-6) as f64;
        let mut h = Hist::new(0.0, hi, 256);
        for t in 0..nf {
            for q in (0..nv).step_by(7) {
                h.add(speed(t, q) as f64);
            }
        }
        speed_thr = o["speed_thr"].as_f64().unwrap_or_else(|| h.value(otsu(&h.bins)));
        for t in 0..nf {
            for q in 0..nv {
                raw[t * nv + q] = (speed(t, q) as f64 >= speed_thr) as u8;
            }
        }
    }

    // temporal persistence: voxels that are blood in at least `persist` of the frames
    let persist = o["persist"].as_f64().unwrap_or(0.5).clamp(0.0, 1.0);
    let mut count = vec![0u16; nv];
    for t in 0..nf {
        for q in 0..nv {
            count[q] += raw[t * nv + q] as u16;
        }
    }
    let core: Vec<u8> = count.iter().map(|&c| (c as f64 >= (persist * nf as f64).max(1.0)) as u8).collect();
    let min_voxels = o["min_voxels"].as_u64().unwrap_or(200) as usize;
    let min_rescue = o["min_rescue"].as_u64().unwrap_or(8) as usize;
    let min_hole = o["min_hole"].as_u64().unwrap_or(27) as usize;
    let mut mask = vec![0u8; nf * nv];
    let mut voxels = Vec::with_capacity(nf);
    let mut ncomp = Vec::with_capacity(nf);
    for t in 0..nf {
        let fr = &raw[t * nv..(t + 1) * nv];
        let (lab, sizes) = components(fr, dims);
        let mut in_core = vec![0usize; sizes.len()];
        for (q, &l) in lab.iter().enumerate() {
            if l != 0 && core[q] != 0 {
                in_core[l as usize - 1] += 1;
            }
        }
        // big enough, or a resolved structure (≥ min_rescue voxels) that is mostly where blood is in most frames
        let keep: Vec<bool> = sizes.iter().zip(&in_core).map(|(&s, &c)| s >= min_voxels || (s >= min_rescue && 2 * c >= s)).collect();
        let out = &mut mask[t * nv..(t + 1) * nv];
        for (q, &l) in lab.iter().enumerate() {
            out[q] = (l != 0 && keep[l as usize - 1]) as u8;
        }
        // enclosed small holes
        let inv_m: Vec<u8> = out.iter().map(|&v| 1 - v).collect();
        let (bl, bs) = components(&inv_m, dims);
        let mut touches = vec![false; bs.len()];
        let [nx, ny, nz] = dims;
        for (q, &l) in bl.iter().enumerate() {
            if l != 0 {
                let (i, j, k) = (q % nx, (q / nx) % ny, q / (nx * ny));
                if i == 0 || j == 0 || k == 0 || i + 1 == nx || j + 1 == ny || k + 1 == nz {
                    touches[l as usize - 1] = true;
                }
            }
        }
        for (q, &l) in bl.iter().enumerate() {
            if l != 0 && !touches[l as usize - 1] && bs[l as usize - 1] < min_hole {
                out[q] = 1;
            }
        }
        voxels.push(out.iter().map(|&v| v as usize).sum::<usize>());
        if std::env::var("FLOW_SURFACE_DEBUG").is_ok() {
            let kept: Vec<(usize, usize)> = keep.iter().enumerate().filter(|(_, &k)| k).map(|(i, _)| (i, sizes[i])).collect();
            eprintln!("frame {t}: kept components (label, voxels) {kept:?}");
        }
        ncomp.push(keep.iter().filter(|&&k| k).count());
    }
    let vox_ml = f.spacing[0] * f.spacing[1] * f.spacing[2] * 1e6;
    let info = json!({
        "source": source, "air_thr": t1, "dark_thr": dark_thr, "m_ref": m_ref, "mag_thr": t2, "mag_thr_otsu": mag_parts.map(|p| p.0), "mag_floor_fast": mag_parts.map(|p| p.1), "speed_thr": speed_thr, "persist": persist, "min_voxels": min_voxels, "min_hole": min_hole,
        "voxels": voxels, "volume_ml": voxels.iter().map(|&v| v as f64 * vox_ml).collect::<Vec<_>>(), "components": ncomp, "warnings": warnings,
    });
    Ok(BloodPool { mask, info })
}

pub fn register(r: &mut Registry) {
    r.register(
        "flow_bloodpool",
        "moving blood-pool surfaces of a flow value from vel + mag (+ mask) alone: inputs [flow value]; options {air_thr, mag_thr, speed_thr (default: q50 + 3 (q50 − q10) of the non-air speeds, at least 3 % of their 99.9th percentile), reach_mm (6: blood is classified only this close to voxels that move fast in ≥ 2 frames), mag_thr (default: Otsu on the magnitude in that neighbourhood), persist (0.5), min_voxels (200), min_rescue (8), min_hole (27), blur (1), smooth (6), steps (12), relax (0.15), max_triangles, return_mask (also output the indicator u8 [F,Z,Y,X])}; output {kind mesh_seq (coherent where links = 1; vertices in mm, voxel centres at index × spacing = the flow viewer's world), links, frames, dt} [+ indicator]; message {thresholds, per-frame volume_ml / components, vertices, area_mm2, links}",
        |_, o, inp: &[Arc<Value>]| {
            let f = Flow::from_value(inp.first().ok_or_else(|| Error::Invalid("flow_bloodpool: one flow value input".into()))?)?;
            let bp = blood_pool(&f, o)?;
            let [x, y, z] = f.dims;
            let t = Tensor::from_slice(&[f.nf, z, y, x], &bp.mask)?;
            let sp = f.spacing.map(|s| s * 1e3);
            let mut opts = o.clone();
            let m = opts.as_object_mut().ok_or_else(|| Error::Invalid("flow_bloodpool: options must be an object".into()))?;
            m.insert("threshold".into(), json!(0.5));
            m.insert("spacing_mm".into(), json!(sp));
            m.insert("keep_first_all".into(), json!(true));
            m.insert("dt".into(), json!(f.dt));
            m.remove("label");
            let (meshes, links, parents) = track_sequence(&t, &opts)?;
            let lk: Vec<u8> = links.iter().map(|&b| b as u8).collect();
            let seq = Value::Dict(vec![
                ("kind".into(), Value::Str("mesh_seq".into())),
                ("coherent".into(), Value::Bool(links.iter().all(|&b| b))),
                ("links".into(), Value::Tensor(Tensor::from_slice(&[lk.len()], &lk)?)),
                ("frames".into(), Value::List(meshes.iter().zip(&parents).map(|(m, par)| {
                    let mut d = m.to_value();
                    if let (Value::Dict(v), Some(p)) = (&mut d, par) {
                        v.push(("parent".into(), Value::Tensor(Tensor::from_slice(&[p.len()], p).unwrap())));
                    }
                    d
                }).collect())),
                ("dt".into(), Value::Float(f.dt)),
                ("source".into(), Value::Str("flow_bloodpool".into())),
                ("dark_thr".into(), Value::Float(bp.info["dark_thr"].as_f64().unwrap_or(0.0))),
            ]);
            let mut msg = bp.info;
            msg["vertices"] = json!(meshes.iter().map(|m| m.verts.len()).collect::<Vec<_>>());
            msg["triangles"] = json!(meshes.iter().map(|m| m.faces.len()).collect::<Vec<_>>());
            msg["area_mm2"] = json!(meshes.iter().map(|m| m.area()).collect::<Vec<_>>());
            msg["surface_volume_ml"] = json!(meshes.iter().map(|m| m.volume() * 1e-3).collect::<Vec<_>>());
            msg["links"] = json!(links);
            let mut out = vec![seq];
            if o["return_mask"].as_bool().unwrap_or(false) {
                out.push(Value::Tensor(t));
            }
            Ok((msg.to_string(), out))
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Ctx;

    /// pulsating ball of blood (radius 7 → 12 voxels over the cycle) in a ring of tissue, air outside; magnitude noise, speckle
    /// noise in the air, blood moving, tissue and air with velocity noise only. Returns the flow value and the true radius per frame.
    fn pulsating(n: usize, nf: usize, noise: f64, speckle: bool) -> (Value, Vec<f64>) {
        let c = (n as f64 - 1.0) / 2.0;
        let nv = n * n * n;
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / (1u64 << 53) as f64
        };
        let mut gauss = move |s: f64, r: &mut dyn FnMut() -> f64| s * ((-2.0 * r().max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * r()).cos());
        let radius: Vec<f64> = (0..nf).map(|t| 9.5 + 2.5 * (2.0 * std::f64::consts::PI * t as f64 / nf as f64).sin()).collect();
        let mut vel = vec![0f32; nf * 3 * nv];
        let mut mag = vec![0f32; nf * nv];
        for t in 0..nf {
            for q in 0..nv {
                let (i, j, k) = ((q % n) as f64, ((q / n) % n) as f64, (q / (n * n)) as f64);
                let r = ((i - c).powi(2) + (j - c).powi(2) + (k - c).powi(2)).sqrt();
                let (m, v) = if r <= radius[t] { (1.0, 0.6) } else if r <= radius[t] + 4.0 { (0.45, 0.0) } else { (0.02, 0.0) };
                mag[t * nv + q] = (m + gauss(noise, &mut rnd)).max(0.0) as f32;
                if speckle && r > radius[t] + 8.0 && rnd() < 0.002 {
                    mag[t * nv + q] = 1.0;
                }
                for a in 0..3 {
                    vel[(t * 3 + a) * nv + q] = ((if a == 0 { v } else { 0.0 }) + gauss(0.02 * noise.min(1.0).max(0.2), &mut rnd)) as f32;
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
        (v, radius)
    }

    fn run(v: Value, opts: &str) -> (Json, Vec<crate::mesh::Mesh>) {
        let reg = Registry::base();
        let ctx = Ctx::new(Arc::new(crate::Bank::new()));
        let (msg, out) = reg.run("flow_bloodpool", &ctx, opts, &[Arc::new(v)]).unwrap();
        let seq = &out[0];
        let Some(Value::List(fr)) = seq.get("frames") else { panic!("frames") };
        (serde_json::from_str(&msg).unwrap(), fr.iter().map(|f| crate::mesh::Mesh::from_value(f).unwrap()).collect())
    }

    fn check(msg: &Json, meshes: &[crate::mesh::Mesh], radius: &[f64], tol: f64) {
        for (t, m) in meshes.iter().enumerate() {
            let truth = 4.0 / 3.0 * std::f64::consts::PI * (radius[t] * 2.0).powi(3); // mm³ (2 mm voxels)
            let got = m.volume();
            assert!((got / truth - 1.0).abs() < tol, "frame {t}: surface volume {got:.0} mm³ against {truth:.0} mm³ ({:.1} %)", 100.0 * (got / truth - 1.0));
        }
        eprintln!("thresholds air {} mag {} speed {} | voxels {} | components {} | surface ml {}", msg["air_thr"], msg["mag_thr"], msg["speed_thr"], msg["voxels"], msg["components"], msg["surface_volume_ml"]);
        assert!(msg["components"].as_array().unwrap().iter().all(|c| c == 1), "one pool per frame: {:?}", msg["components"]);
    }

    #[test]
    fn pulsating_ball_with_noise_and_speckle() {
        let (v, radius) = pulsating(40, 6, 0.08, true);
        let (msg, meshes) = run(v, r#"{}"#);
        assert_eq!(msg["source"], "mag+vel");
        check(&msg, &meshes, &radius, 0.07);
        // speckle removed, motion followed with one topology
        assert!(msg["links"].as_array().unwrap().iter().all(|l| l == true), "links {:?}", msg["links"]);
        let nv0 = meshes[0].verts.len();
        assert!(meshes.iter().all(|m| m.verts.len() == nv0), "coherent sequence keeps its vertices");
    }

    #[test]
    fn noise_free_control_is_tighter() {
        let (v, radius) = pulsating(40, 6, 0.0, false);
        let (msg, meshes) = run(v, r#"{}"#);
        check(&msg, &meshes, &radius, 0.07);
    }

    #[test]
    fn fast_dark_blood_is_found_by_its_speed() {
        // blood whose magnitude dropped to tissue level (saturation) is still blood when it moves: here the lower half of the ball
        let (v, radius) = pulsating(40, 6, 0.0, false);
        let Value::Dict(mut d) = v else { unreachable!() };
        let n = 40;
        for (k, val) in d.iter_mut() {
            if k == "mag" {
                let Value::Tensor(t) = val else { unreachable!() };
                let mut x: Vec<f32> = (0..t.numel()).map(|i| t.get_f64(i).0 as f32).collect();
                for (i, m) in x.iter_mut().enumerate() {
                    let q = i % (n * n * n);
                    if *m > 0.9 && q / (n * n) < 19 {
                        *m = 0.45;
                    }
                }
                *t = Tensor::from_slice(&[6, n, n, n], &x).unwrap();
            }
        }
        let (msg, meshes) = run(Value::Dict(d), r#"{}"#);
        check(&msg, &meshes, &radius, 0.07);
    }
}
