//! Air map: voxels without signal (external air, large airways) estimated from the iterate and projected to zero.
//!
//! The coil-map support is a thresholded, smoothed warm start; on a thorax it contains a third of air. Air carries no
//! signal, so x = 0 there is a constraint (a convex set: the projection is firmly non-expansive) that removes
//! unconstrained artefact modes. Marking tissue as air is much worse than missing air, so every step is conservative.
//!
//! Estimate, from the current iterate x[r, c, e] (respiratory bin r, cardiac frame c, encode e):
//! - reference magnitude R(q) = max_r √(mean_e |mean_c x_{r,c,e}(q)|²): the frame mean cancels incoherent aliasing,
//!   the RSS over encodes keeps flow-dephased voxels, the max over respiratory bins keeps every position the moving
//!   body surface takes (the map is applied to all frames, so its body must be the union over the motion states);
//! - body level L = median of R over the voxels above 10 % of its 99th percentile (scale-free: the map is invariant
//!   to x → c·x and to the normalisation of the coil maps);
//! - R_s = max over scales σ ∈ `blur_mm` of the Gaussian-smoothed R (low only if low at every scale);
//! - low set: R_s < `tau` · L (plus the voxels marked air at the previous refresh, which are zero in the iterate);
//! - boundary seeds: low voxels on the 12 edges of the box, grown within each of the 6 faces (2D, 4-connected): a face
//!   that cuts the body (head–foot faces of a thorax) only seeds outside the body's cross-section in that face, never
//!   its dark interior;
//! - growth in 3D (6-connected) from the seeds through the low set opened by a ball of `ball_mm` (erode, grow,
//!   dilate back inside the low set): a leak needs a hole of ≥ 2 `ball_mm` through the body wall;
//! - enclosure tests (`slices`): a voxel stays a candidate only if, in its slice of each of the three orientations,
//!   it is reachable from the slice border (seed voxels) through low voxels. Tubes (airways) are enclosed in the slices
//!   across them, lungs in the axial and sagittal slices: low-signal tissue behind a bright wall is never marked even
//!   where a bronchus would connect it in 3D;
//! - candidates C = grown ∪ previous candidates (monotone), minus protected voxels;
//! - weight w = clamp((d − `margin_mm`) / `ramp_mm`, 0, 1), d = Euclidean distance (mm) from the nearest non-candidate
//!   voxel (the FOV exterior counts as air): a safety margin around everything that is not certainly air;
//! - hysteresis over refreshes: a voxel once zeroed (w ≥ ½) is released only when the iterate puts consistent signal
//!   back (R smoothed at the finest scale over the zeroed region only, above `release` · L, after the gradient step
//!   from zero) and is then protected for the rest of the run, so no voxel flips back and forth.
//!
//! Apply: x ← (1 − w) x in every frame and encode.
//!
//! Seeds (`AirMap::resample`, e.g. from a converged coarse level of a resolution pyramid): a map on another grid of the
//! same field of view (voxel centres at (i − ⌊n/2⌋)·vox per axis, `SimData::set_recon_grid`), trilinear weights,
//! eroded by half the source voxel (its boundary is only known to that accuracy), used as the previous map.

use hasty_cuda::C32;
use rayon::prelude::*;

use crate::tlock::{cf, cf_mut, FrameLayout};

type Cf = num_complex::Complex<f32>;

#[derive(Clone, Debug)]
pub struct AirOpts {
    /// low set: R_s < tau · L
    pub tau: f64,
    /// release a zeroed voxel when R (finest scale, zeroed region only) > release · L
    pub release: f64,
    /// smoothing scales (mm); R_s = max over the scales
    pub blur_mm: Vec<f64>,
    /// opening radius of the growth (mm, 0 = plain growth)
    pub ball_mm: f64,
    /// safety margin (mm) between the zeroed region and anything not certainly air
    pub margin_mm: f64,
    /// soft transition of the weight beyond the margin (mm, 0 = step)
    pub ramp_mm: f64,
    /// 2D enclosure tests in the slices of all three orientations
    pub slices: bool,
}

impl Default for AirOpts {
    fn default() -> Self {
        Self { tau: 0.05, release: 0.15, blur_mm: vec![2.0], ball_mm: 4.0, margin_mm: 4.0, ramp_mm: 2.0, slices: true }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AirStats {
    /// voxels with w ≥ ½ (zeroed)
    pub marked: usize,
    /// candidates (before the margin)
    pub candidates: usize,
    /// voxels zeroed now that were not before
    pub added: usize,
    /// voxels released this refresh / protected in total
    pub released: usize,
    pub protected: usize,
    /// low voxels, boundary seeds
    pub low: usize,
    pub seeds: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AirMap {
    /// weight in [0, 1]: x ← (1 − w) x
    pub w: Vec<f32>,
    /// air candidates before the margin (monotone over refreshes)
    pub cand: Vec<bool>,
    /// released voxels: never air again
    pub protect: Vec<bool>,
    /// R_s / L of the estimate (diagnostic)
    pub rel: Vec<f32>,
    /// body level L of the estimate
    pub level: f64,
    /// iteration the map was estimated at
    pub iteration: usize,
    pub stats: AirStats,
}

impl AirMap {
    /// the four maps (w, candidate, protected, R_s / L), image-major
    pub fn stacked(&self) -> Vec<f32> {
        let b = |v: &[bool]| v.iter().map(|&x| x as u8 as f32).collect::<Vec<f32>>();
        [self.w.clone(), b(&self.cand), b(&self.protect), self.rel.clone()].concat()
    }

    pub fn from_stacked(v: &[f32], nv: usize, iteration: usize, level: f64) -> Self {
        assert!(v.len() >= 3 * nv, "air map: expected ≥ 3 maps of {nv} voxels");
        let g = |i: usize| &v[i * nv..(i + 1) * nv];
        let w = g(0).to_vec();
        let cand: Vec<bool> = g(1).iter().map(|&x| x > 0.5).collect();
        let protect: Vec<bool> = g(2).iter().map(|&x| x > 0.5).collect();
        let rel = if v.len() >= 4 * nv { g(3).to_vec() } else { vec![0.0; nv] };
        let mut m = Self { w, cand, protect, rel, level, iteration, stats: AirStats::default() };
        m.stats = AirStats { marked: m.marked(), candidates: m.cand.iter().filter(|&&b| b).count(), protected: m.protect.iter().filter(|&&b| b).count(), ..Default::default() };
        m
    }

    /// a map from an air mask or weights (e.g. from another estimator): candidates where w ≥ ½; call
    /// [`refresh_weights`](Self::refresh_weights) to put the margin of the current options around them
    pub fn from_weights(w: Vec<f32>, iteration: usize) -> Self {
        let nv = w.len();
        let w: Vec<f32> = w.into_iter().map(|x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }).collect();
        let cand = w.iter().map(|&x| x >= 0.5).collect();
        let mut m = Self { w, cand, protect: vec![false; nv], rel: vec![0.0; nv], level: f64::NAN, iteration, stats: AirStats::default() };
        m.update_counts();
        m
    }

    /// weights from the candidates with the margin and ramp of `opts` (voxel size in mm)
    pub fn refresh_weights(&mut self, d: [usize; 3], vox_mm: [f64; 3], opts: &AirOpts) {
        self.w = margin_weights(&self.cand, d, vox_mm, opts);
        self.update_counts();
    }

    fn update_counts(&mut self) {
        self.stats.marked = self.marked();
        self.stats.candidates = self.cand.iter().filter(|&&b| b).count();
        self.stats.protected = self.protect.iter().filter(|&&b| b).count();
    }

    pub fn nv(&self) -> usize {
        self.w.len()
    }

    /// voxels with w ≥ ½
    pub fn marked(&self) -> usize {
        self.w.iter().filter(|&&x| x >= 0.5).count()
    }

    /// The map on another grid of the same field of view (voxel centres at (i − ⌊n/2⌋)·vox per axis): the candidate
    /// indicator trilinear (≥ ½), eroded by half a source voxel per axis (the source boundary is known to ±½ voxel);
    /// protected = any protected source voxel among the 8 neighbours. The weights are the resampled ones, eroded
    /// likewise; [`refresh_weights`](Self::refresh_weights) puts the margin of the target grid's options around the
    /// candidates instead. `level` and `rel` are not carried over (the next estimate recomputes them).
    pub fn resample(&self, from_dims: [usize; 3], from_vox: [f64; 3], to_dims: [usize; 3], to_vox: [f64; 3]) -> AirMap {
        let fd = from_dims;
        assert_eq!(self.nv(), fd.iter().product::<usize>(), "air map: dims do not match");
        let nv = to_dims.iter().product::<usize>();
        let fidx = |i: [usize; 3]| i[0] + fd[0] * (i[1] + fd[1] * i[2]);
        let res: Vec<(f32, f32, bool)> = (0..nv)
            .into_par_iter()
            .map(|q| {
                let p = [q % to_dims[0], (q / to_dims[0]) % to_dims[1], q / (to_dims[0] * to_dims[1])];
                let (mut i0, mut f) = ([0usize; 3], [0f64; 3]);
                for a in 0..3 {
                    let x = (p[a] as f64 - (to_dims[a] / 2) as f64) * to_vox[a];
                    let u = (x / from_vox[a] + (fd[a] / 2) as f64).clamp(0.0, (fd[a] - 1) as f64);
                    i0[a] = (u.floor() as usize).min(fd[a].saturating_sub(2));
                    f[a] = if fd[a] > 1 { u - i0[a] as f64 } else { 0.0 };
                }
                let (mut aw, mut ac, mut pr) = (0f64, 0f64, false);
                for corner in 0..8 {
                    let mut wt = 1.0;
                    let mut idx = [0usize; 3];
                    for a in 0..3 {
                        let up = (corner >> a) & 1;
                        idx[a] = (i0[a] + up).min(fd[a] - 1);
                        wt *= if up == 1 { f[a] } else { 1.0 - f[a] };
                    }
                    if wt > 0.0 {
                        let i = fidx(idx);
                        aw += wt * self.w[i] as f64;
                        ac += wt * self.cand[i] as u8 as f64;
                        pr |= self.protect[i];
                    }
                }
                if pr { (0.0, 0.0, true) } else { (aw as f32, ac as f32, false) }
            })
            .collect();
        // erosion of the air by half a source voxel: min filter over that distance (per axis, in target voxels)
        let r = [0, 1, 2].map(|a| (0.5 * from_vox[a] / to_vox[a]).round() as usize);
        let w = min_filter(&res.iter().map(|t| t.0).collect::<Vec<_>>(), to_dims, r);
        let c = min_filter(&res.iter().map(|t| t.1).collect::<Vec<_>>(), to_dims, r);
        let mut m = AirMap { w, cand: c.iter().map(|&x| x >= 0.5).collect(), protect: res.iter().map(|t| t.2).collect(), rel: vec![0.0; nv], level: f64::NAN, iteration: self.iteration, stats: AirStats::default() };
        m.update_counts();
        m
    }
}

/// w = clamp((d − margin) / ramp, 0, 1) on the candidates, d = distance (mm) to the nearest non-candidate voxel
pub fn margin_weights(cand: &[bool], d: [usize; 3], vox_mm: [f64; 3], opts: &AirOpts) -> Vec<f32> {
    let dist = edt_mm(cand, d, vox_mm);
    let (mg, rp) = (opts.margin_mm as f32, opts.ramp_mm as f32);
    dist.par_iter()
        .zip(cand.par_iter())
        .map(|(&s, &c)| {
            if !c {
                0.0
            } else if rp > 0.0 {
                ((s - mg) / rp).clamp(0.0, 1.0)
            } else if s > mg {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// reference magnitude R(q) = max over respiratory bins of √(mean_e |mean_c x_{r,c,e}(q)|²)
pub fn reference(h: &[C32], lay: &FrameLayout) -> Vec<f32> {
    let h = cf(h);
    let (nv, nc, nr, ne) = (lay.nv(), lay.nc, lay.nr, lay.ne);
    let mut r = vec![0f32; nv];
    let mut mean = vec![Cf::new(0.0, 0.0); nv];
    let mut acc = vec![0f32; nv];
    let s = 1.0 / (nc * nc * ne) as f32;
    for b in 0..nr {
        acc.par_iter_mut().for_each(|a| *a = 0.0);
        for e in 0..ne {
            mean.par_iter_mut().for_each(|m| *m = Cf::new(0.0, 0.0));
            for c in 0..nc {
                let img = &h[lay.img(b, c, e) * nv..][..nv];
                mean.par_iter_mut().zip(img.par_iter()).for_each(|(m, &x)| *m += x);
            }
            acc.par_iter_mut().zip(mean.par_iter()).for_each(|(a, m)| *a += m.norm_sqr() * s);
        }
        r.par_iter_mut().zip(acc.par_iter()).for_each(|(r, &a)| *r = r.max(a.sqrt()));
    }
    r
}

/// robust body level: median of R over the voxels above 10 % of its 99th percentile (≤ 2²⁰ samples)
pub fn body_level(r: &[f32]) -> f64 {
    let step = r.len().div_ceil(1 << 20).max(1);
    let mut s: Vec<f32> = r.iter().step_by(step).copied().filter(|v| v.is_finite()).collect();
    if s.is_empty() {
        return 0.0;
    }
    let k = ((s.len() - 1) as f64 * 0.99) as usize;
    let q99 = *s.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1;
    let mut hi: Vec<f32> = s.into_iter().filter(|&v| v > 0.1 * q99).collect();
    if hi.is_empty() {
        return q99 as f64;
    }
    let m = hi.len() / 2;
    *hi.select_nth_unstable_by(m, |a, b| a.partial_cmp(b).unwrap()).1 as f64
}

/// the JSON sidecar of a saved map (`air_map_*.json`): what [`load_seed`] and the resume path read back
pub fn sidecar(map: &AirMap, opts: &AirOpts, dims: [usize; 3], vox_mm: [f64; 3]) -> serde_json::Value {
    let s = &map.stats;
    serde_json::json!({
        "iteration": map.iteration, "level": map.level, "dims": dims, "vox_mm": vox_mm,
        "maps": ["w", "candidate", "protected", "rel"], "opts": format!("{opts:?}"),
        "marked": s.marked, "candidates": s.candidates, "added": s.added, "released": s.released, "protected": s.protected, "low": s.low, "seeds": s.seeds,
    })
}

/// A seed map (`--air-init`) for the grid `dims` / `vox_mm` from an npy file (Fortran order, x fastest), any of
/// - an air map of this module (`air_map_*.npy`, (X,Y,Z,≥3) f32 with its `.json` sidecar): its candidates;
/// - an air mask or weights ((X,Y,Z) bool / u8, or f32 with values in [0, 1] of which ≥ 99 % are exactly 0 or 1):
///   candidates where ≥ ½;
/// - a magnitude ((X,Y,Z) f32, not a mask) or a complex image stack ((X,Y,Z[,E[,F]]) c8, encodes E, frames F;
///   e.g. a converged coarse level): the estimate with `opts` on its own grid;
///
/// on a grid of the same field of view (source voxel = vox_mm · dims / source dims per axis, unless the sidecar
/// gives `vox_mm`), resampled (candidates eroded by half a source voxel) and with the margin of `opts` around the
/// candidates. Returns the map and a description.
pub fn load_seed(path: &std::path::Path, dims: [usize; 3], vox_mm: [f64; 3], opts: &AirOpts) -> Result<(AirMap, String), crate::Error> {
    let a = crate::data::read_npy(path)?;
    let fmt = |m: &str| crate::Error::Format(format!("{}: {m}", path.display()));
    if a.shape.len() < 3 {
        return Err(fmt("need (X,Y,Z[,…]) dims"));
    }
    if !a.fortran && a.shape[..3].iter().filter(|&&n| n > 1).count() > 1 {
        return Err(fmt("need Fortran order (x fastest), as the reconstruction writes"));
    }
    let sd = [a.shape[0], a.shape[1], a.shape[2]];
    let snv = sd.iter().product::<usize>();
    let trailing: usize = a.shape[3..].iter().product();
    let side: Option<serde_json::Value> = std::fs::read_to_string(path.with_extension("json")).ok().and_then(|t| serde_json::from_str(&t).ok());
    let svox: [f64; 3] = side
        .as_ref()
        .and_then(|j| j["vox_mm"].as_array().map(|v| [0, 1, 2].map(|i| v[i].as_f64().unwrap_or(0.0))))
        .filter(|v| v.iter().all(|&x| x > 0.0))
        .unwrap_or([0, 1, 2].map(|i| vox_mm[i] * dims[i] as f64 / sd[i] as f64));
    let (src, what) = match a.descr.as_str() {
        "<f4" if trailing >= 3 && side.as_ref().is_some_and(|j| j["maps"].is_array()) => {
            let v = a.f32s();
            (AirMap::from_stacked(&v, snv, side.as_ref().unwrap()["iteration"].as_u64().unwrap_or(0) as usize, f64::NAN), "air map")
        }
        "|b1" | "|u1" if trailing == 1 => (AirMap::from_weights(a.bytes.iter().map(|&b| (b != 0) as u8 as f32).collect(), 0), "air mask"),
        "<f4" if trailing == 1 => {
            let v = a.f32s();
            let binary = v.iter().filter(|&&x| x == 0.0 || x == 1.0).count();
            if v.iter().all(|&x| (0.0..=1.0).contains(&x)) && binary as f64 >= 0.99 * v.len() as f64 {
                (AirMap::from_weights(v, 0), "air mask")
            } else {
                (estimate_from_reference(&v.iter().map(|x| x.abs()).collect::<Vec<_>>(), sd, svox, opts, None, 0), "magnitude volume (estimated)")
            }
        }
        "<c8" => {
            let h = a.c32s();
            let (ne, nf) = match a.shape.len() {
                3 => (1, 1),
                4 => (a.shape[3], 1),
                _ => (a.shape[3], a.shape[4..].iter().product()),
            };
            let lay = FrameLayout::new(sd, svox.map(|v| v * 1e-3), nf, 1, ne, false);
            let r = reference(&h, &lay);
            (estimate_from_reference(&r, sd, svox, opts, None, 0), "complex images (estimated)")
        }
        d => return Err(fmt(&format!("unsupported dtype {d} / shape {:?}", a.shape))),
    };
    let mut m = if sd == dims { src.clone() } else { src.resample(sd, svox, dims, vox_mm) };
    m.refresh_weights(dims, vox_mm, opts);
    let desc = format!("{what} {:?} at {:?} mm: {} candidates → {} zeroed on {:?}", sd, svox.map(|v| (v * 100.0).round() / 100.0), src.stats.candidates, m.stats.marked, dims);
    Ok((m, desc))
}

/// estimate the air map from the stack `h` (layout `lay`); `prior`: the previous map (monotone update, release test)
pub fn estimate(h: &[C32], lay: &FrameLayout, opts: &AirOpts, prior: Option<&AirMap>, iteration: usize) -> AirMap {
    let r = reference(h, lay);
    estimate_from_reference(&r, lay.dims, lay.vox.map(|v| v * 1e3), opts, prior, iteration)
}

/// the estimate from a reference magnitude volume R (x fastest), voxel size in mm
pub fn estimate_from_reference(r: &[f32], d: [usize; 3], vox_mm: [f64; 3], opts: &AirOpts, prior: Option<&AirMap>, iteration: usize) -> AirMap {
    let nv = d[0] * d[1] * d[2];
    assert_eq!(r.len(), nv, "air map: reference size");
    let level = body_level(r);
    let inv = if level > 0.0 { (1.0 / level) as f32 } else { 0.0 };
    // R_s / L at every scale: the max for the low set, the finest scale for the release test
    let mut scales: Vec<f64> = opts.blur_mm.iter().copied().filter(|s| *s >= 0.0).collect();
    if scales.is_empty() {
        scales.push(0.0);
    }
    scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut rel = vec![0f32; nv];
    for &s in &scales {
        let b = gauss_blur(r, d, [0, 1, 2].map(|a| s / vox_mm[a]));
        rel.par_iter_mut().zip(b.par_iter()).for_each(|(o, &v)| *o = o.max(v * inv));
    }
    let empty = AirMap::default();
    let pr = prior.filter(|p| p.nv() == nv).unwrap_or(&empty);
    let has_prior = pr.nv() == nv;
    // release: zeroed voxels where the iterate puts consistent signal back. The evidence is R smoothed at the finest
    // scale over the zeroed region only (normalised convolution): artefact outside the region, next to its edge, does
    // not count, signal re-injected inside it does
    let released: Vec<bool> = if has_prior {
        let zm: Vec<f32> = pr.w.par_iter().zip(pr.protect.par_iter()).map(|(&w, &p)| if w >= 0.5 && !p { 1.0 } else { 0.0 }).collect();
        let rz: Vec<f32> = r.par_iter().zip(zm.par_iter()).map(|(&a, &m)| a * m).collect();
        let sig = [0, 1, 2].map(|a| scales[0] / vox_mm[a]);
        let (num, den) = (gauss_blur(&rz, d, sig), gauss_blur(&zm, d, sig));
        let thr = (opts.release * level) as f32;
        (0..nv).into_par_iter().map(|q| zm[q] > 0.0 && num[q] > thr * den[q]).collect()
    } else {
        vec![false; nv]
    };
    let protect: Vec<bool> = (0..nv).into_par_iter().map(|q| released[q] || (has_prior && pr.protect[q])).collect();
    let prior_cand: Vec<bool> = (0..nv).into_par_iter().map(|q| has_prior && pr.cand[q] && !protect[q]).collect();
    let tau = opts.tau as f32;
    let low: Vec<bool> = (0..nv).into_par_iter().map(|q| !protect[q] && (rel[q] < tau || prior_cand[q])).collect();
    // boundary seeds: edge voxels grown within their faces
    let mut seed = vec![false; nv];
    for a in 0..3 {
        for k in [0, d[a] - 1] {
            let (pl, ids) = plane(d, a, k);
            let pass: Vec<bool> = ids.iter().map(|&q| low[q]).collect();
            let s0: Vec<bool> = (0..ids.len()).map(|i| pass[i] && on_border(pl, i)).collect();
            for (i, &v) in flood2(pl, &pass, &s0).iter().enumerate() {
                seed[ids[i]] |= v;
            }
            if d[a] == 1 {
                break;
            }
        }
    }
    // growth through the opened low set
    let grown = if opts.ball_mm > 0.0 {
        let dl = edt_mm(&low, d, vox_mm);
        let core: Vec<bool> = dl.par_iter().map(|&v| v > opts.ball_mm as f32).collect();
        let s: Vec<bool> = (0..nv).into_par_iter().map(|q| seed[q] && core[q]).collect();
        let g = flood3(d, &core, &s);
        let notg: Vec<bool> = g.par_iter().map(|&b| !b).collect();
        let dg = edt_mm(&notg, d, vox_mm);
        (0..nv).into_par_iter().map(|q| low[q] && dg[q] <= opts.ball_mm as f32).collect::<Vec<bool>>()
    } else {
        flood3(d, &low, &seed)
    };
    let mut grown = grown;
    if opts.slices {
        for a in 0..3 {
            let reach = slice_reach(d, a, &low, &seed);
            grown.par_iter_mut().zip(reach.par_iter()).for_each(|(g, &r)| *g &= r);
        }
    }
    let cand: Vec<bool> = (0..nv).into_par_iter().map(|q| !protect[q] && (grown[q] || prior_cand[q])).collect();
    let w = margin_weights(&cand, d, vox_mm, opts);
    let cnt = |v: &[bool]| v.iter().filter(|&&b| b).count();
    let marked = w.iter().filter(|&&x| x >= 0.5).count();
    let added = (0..nv).filter(|&q| w[q] >= 0.5 && !(has_prior && pr.w[q] >= 0.5)).count();
    let stats = AirStats { marked, candidates: cnt(&cand), added, released: cnt(&released), protected: cnt(&protect), low: cnt(&low), seeds: cnt(&seed) };
    AirMap { w, cand, protect, rel, level, iteration, stats }
}

/// the projection x ← (1 − w) x on every image of the stack (in place)
pub fn apply(h: &mut [C32], lay: &FrameLayout, map: &AirMap) {
    let nv = lay.nv();
    assert_eq!(map.nv(), nv, "air map: size");
    let idx: Vec<(u32, f32)> = map.w.iter().enumerate().filter(|(_, &w)| w > 0.0).map(|(q, &w)| (q as u32, 1.0 - w)).collect();
    if idx.is_empty() {
        return;
    }
    let h = cf_mut(h);
    h.par_chunks_mut(nv).for_each(|img| {
        for &(q, g) in &idx {
            img[q as usize] *= g;
        }
    });
}

/// separable Gaussian low-pass (σ per axis in voxels, replicated edges)
pub fn gauss_blur(v: &[f32], d: [usize; 3], sig: [f64; 3]) -> Vec<f32> {
    let mut a = v.to_vec();
    for ax in 0..3 {
        let k = gauss(sig[ax]);
        if k.len() == 1 {
            continue;
        }
        a = along(&a, d, ax, |line, out| {
            let n = line.len() as isize;
            let r = (k.len() / 2) as isize;
            for (i, o) in out.iter_mut().enumerate() {
                let mut s = 0f32;
                for (t, &g) in k.iter().enumerate() {
                    s += g * line[(i as isize + t as isize - r).clamp(0, n - 1) as usize];
                }
                *o = s;
            }
        });
    }
    a
}

fn gauss(s: f64) -> Vec<f32> {
    if s < 0.3 {
        return vec![1.0];
    }
    let r = (3.0 * s).ceil() as isize;
    let k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / s).powi(2)).exp()).collect();
    let t: f64 = k.iter().sum();
    k.iter().map(|v| (v / t) as f32).collect()
}

/// separable min filter (half-width r voxels per axis)
fn min_filter(v: &[f32], d: [usize; 3], r: [usize; 3]) -> Vec<f32> {
    let mut a = v.to_vec();
    for ax in 0..3 {
        if r[ax] == 0 {
            continue;
        }
        let ra = r[ax];
        a = along(&a, d, ax, |line, out| {
            let n = line.len();
            for (i, o) in out.iter_mut().enumerate() {
                *o = line[i.saturating_sub(ra)..=(i + ra).min(n - 1)].iter().copied().fold(f32::INFINITY, f32::min);
            }
        });
    }
    a
}

/// raw pointer shared by tasks that write disjoint elements
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// apply `f(line, out)` to every line along axis `ax` (lines processed in parallel), out of place
fn along<T: Copy + Default + Send + Sync>(v: &[T], d: [usize; 3], ax: usize, f: impl Fn(&[T], &mut [T]) + Sync) -> Vec<T> {
    let nv = d[0] * d[1] * d[2];
    let mut out = vec![T::default(); nv];
    let n = d[ax];
    let stride = [1, d[0], d[0] * d[1]][ax];
    let nlines = nv / n;
    // line l: the other two coordinates in order (x fastest)
    let base = |l: usize| -> usize {
        match ax {
            0 => l * d[0],
            1 => (l % d[0]) + (l / d[0]) * d[0] * d[1],
            _ => l,
        }
    };
    let p = SyncPtr(out.as_mut_ptr());
    (0..nlines).into_par_iter().with_min_len(64).for_each_init(
        || (vec![T::default(); n], vec![T::default(); n]),
        |(line, o), l| {
            let b = base(l);
            for i in 0..n {
                line[i] = v[b + i * stride];
            }
            f(line, o);
            // capture the whole wrapper (Send + Sync), not its raw-pointer field
            #[allow(clippy::redundant_locals)]
            let p = p;
            for i in 0..n {
                // SAFETY: lines are disjoint sets of voxels, every index < nv
                unsafe { *p.0.add(b + i * stride) = o[i] };
            }
        },
    );
    out
}

/// Euclidean distance (mm) from every voxel of `set` to the nearest voxel outside it (0 outside; the exterior of the
/// box counts as inside the set; ∞ if every voxel is in the set). Felzenszwalb–Huttenlocher, separable.
pub fn edt_mm(set: &[bool], d: [usize; 3], vox_mm: [f64; 3]) -> Vec<f32> {
    let mut g: Vec<f64> = set.par_iter().map(|&b| if b { f64::INFINITY } else { 0.0 }).collect();
    for ax in 0..3 {
        let s2 = vox_mm[ax] * vox_mm[ax];
        g = along(&g, d, ax, |line, out| dt1d(line, s2, out));
    }
    g.into_par_iter().map(|v| v.sqrt() as f32).collect()
}

/// 1D squared distance transform: out[p] = min_q f[q] + s2 (p − q)² (infinite f excluded)
fn dt1d(f: &[f64], s2: f64, out: &mut [f64]) {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0f64; n + 1];
    let mut k: isize = -1;
    for q in 0..n {
        if !f[q].is_finite() {
            continue;
        }
        loop {
            if k < 0 {
                k = 0;
                v[0] = q;
                z[0] = f64::NEG_INFINITY;
                z[1] = f64::INFINITY;
                break;
            }
            let p = v[k as usize];
            let (qf, pf) = (q as f64, p as f64);
            let s = ((f[q] + s2 * qf * qf) - (f[p] + s2 * pf * pf)) / (2.0 * s2 * (qf - pf));
            if s <= z[k as usize] {
                k -= 1;
                continue;
            }
            k += 1;
            v[k as usize] = q;
            z[k as usize] = s;
            z[k as usize + 1] = f64::INFINITY;
            break;
        }
    }
    if k < 0 {
        out.iter_mut().for_each(|o| *o = f64::INFINITY);
        return;
    }
    let mut j = 0usize;
    for (p, o) in out.iter_mut().enumerate() {
        while z[j + 1] < p as f64 {
            j += 1;
        }
        let dp = p as f64 - v[j] as f64;
        *o = s2 * dp * dp + f[v[j]];
    }
}

/// 3D growth (6-connected) from `seed` through `pass`
fn flood3(d: [usize; 3], pass: &[bool], seed: &[bool]) -> Vec<bool> {
    let nv = d[0] * d[1] * d[2];
    let mut out = vec![false; nv];
    let mut st: Vec<u32> = (0..nv).filter(|&q| seed[q] && pass[q]).map(|q| q as u32).collect();
    for &q in &st {
        out[q as usize] = true;
    }
    let (sx, sy) = (d[0], d[0] * d[1]);
    while let Some(q) = st.pop() {
        let q = q as usize;
        let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / sy);
        let mut push = |n: usize| {
            if pass[n] && !out[n] {
                out[n] = true;
                st.push(n as u32);
            }
        };
        if i > 0 { push(q - 1); }
        if i + 1 < d[0] { push(q + 1); }
        if j > 0 { push(q - sx); }
        if j + 1 < d[1] { push(q + sx); }
        if k > 0 { push(q - sy); }
        if k + 1 < d[2] { push(q + sy); }
    }
    out
}

/// slice k of axis a: its 2D shape (u, v = the other two axes in order) and the 3D index of every pixel (u fastest)
fn plane(d: [usize; 3], a: usize, k: usize) -> ([usize; 2], Vec<usize>) {
    let (ua, va) = match a {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let st = [1, d[0], d[0] * d[1]];
    let pl = [d[ua], d[va]];
    let ids = (0..pl[0] * pl[1]).map(|i| (i % pl[0]) * st[ua] + (i / pl[0]) * st[va] + k * st[a]).collect();
    (pl, ids)
}

fn on_border(pl: [usize; 2], i: usize) -> bool {
    let (u, v) = (i % pl[0], i / pl[0]);
    u == 0 || v == 0 || u + 1 == pl[0] || v + 1 == pl[1]
}

/// 2D growth (4-connected) from `seed` through `pass`
fn flood2(pl: [usize; 2], pass: &[bool], seed: &[bool]) -> Vec<bool> {
    let n = pl[0] * pl[1];
    let mut out = vec![false; n];
    let mut st: Vec<u32> = (0..n).filter(|&i| seed[i] && pass[i]).map(|i| i as u32).collect();
    for &i in &st {
        out[i as usize] = true;
    }
    while let Some(i) = st.pop() {
        let i = i as usize;
        let (u, v) = (i % pl[0], i / pl[0]);
        let mut push = |m: usize| {
            if pass[m] && !out[m] {
                out[m] = true;
                st.push(m as u32);
            }
        };
        if u > 0 { push(i - 1); }
        if u + 1 < pl[0] { push(i + 1); }
        if v > 0 { push(i - pl[0]); }
        if v + 1 < pl[1] { push(i + pl[0]); }
    }
    out
}

/// voxels reachable, within their slice across axis a, from the slice border's seed voxels through `pass`
fn slice_reach(d: [usize; 3], a: usize, pass: &[bool], seed: &[bool]) -> Vec<bool> {
    let nv = d[0] * d[1] * d[2];
    let mut out = vec![false; nv];
    let parts: Vec<(Vec<usize>, Vec<bool>)> = (0..d[a])
        .into_par_iter()
        .map(|k| {
            let (pl, ids) = plane(d, a, k);
            let ps: Vec<bool> = ids.iter().map(|&q| pass[q]).collect();
            let s0: Vec<bool> = (0..ids.len()).map(|i| ps[i] && seed[ids[i]] && on_border(pl, i)).collect();
            let r = flood2(pl, &ps, &s0);
            (ids, r)
        })
        .collect();
    for (ids, r) in parts {
        for (q, v) in ids.into_iter().zip(r) {
            out[q] = v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        fn g(&mut self) -> f64 {
            (-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos()
        }
    }

    /// 64 × 56 × 40 at 2 mm: an elliptic body (receive shading: 3× brighter at the "anterior coil" side), two
    /// enclosed lungs with 3 % of the tissue signal (darker than the air threshold, so only the topology protects them)
    /// and a bright vessel each, a dark trachea from the body top into the right lung (connected to the FOV boundary
    /// in 3D); coherent haze in the air (smooth random field, `haze` × the local tissue level next to the body,
    /// decaying over 8 mm); 2 respiratory bins (body shifted by one voxel) × 3 frames × 2 encodes, noise per frame
    struct Phantom {
        d: [usize; 3],
        h: Vec<C32>,
        lay: FrameLayout,
        body: Vec<bool>,
        lung: Vec<bool>,
        trachea: Vec<bool>,
        /// air at least `far` mm from the body (union over the bins)
        far_air: Vec<bool>,
    }

    fn phantom(haze: f64, seed: u64) -> Phantom {
        let d = [64usize, 56, 40];
        let nv = d.iter().product::<usize>();
        let (nc, nr, ne) = (3usize, 2usize, 2usize);
        let lay = FrameLayout::new(d, [2e-3; 3], nc, nr, ne, false);
        let mut rng = Rng(seed);
        let pos = |q: usize| [(q % d[0]) as f64, ((q / d[0]) % d[1]) as f64, (q / (d[0] * d[1])) as f64];
        let c = [31.5, 27.5, 19.5];
        // body: ellipse in x-y (semi-axes 22, 17 voxels), spanning z ∈ [4, 36)
        let ell = |p: [f64; 3], shift: f64, ax: f64, ay: f64| ((p[0] - c[0]) / ax).powi(2) + ((p[1] - c[1] - shift) / ay).powi(2) <= 1.0 && p[2] >= 4.0 && p[2] < 36.0;
        let lungc = [[20.0, 27.5, 19.5], [43.0, 27.5, 19.5]];
        let in_lung = |p: [f64; 3], shift: f64| lungc.iter().any(|l| ((p[0] - l[0]) / 8.0).powi(2) + ((p[1] - l[1] - shift) / 10.0).powi(2) + ((p[2] - l[2]) / 11.0).powi(2) <= 1.0);
        let in_vessel = |p: [f64; 3], shift: f64| lungc.iter().any(|l| (p[0] - l[0]).powi(2) + (p[1] - l[1] - shift).powi(2) <= 2.0 && (p[2] - l[2]).abs() < 9.0);
        // trachea: radius 3 voxels at (43, 27.5), from the top face (z = 39) down into the right lung
        let in_trachea = |p: [f64; 3], shift: f64| (p[0] - 43.0).powi(2) + (p[1] - 27.5 - shift).powi(2) <= 9.0 && p[2] >= 24.0;
        // smooth haze field: random coarse lattice, trilinear
        let lat: Vec<f64> = (0..9 * 8 * 7).map(|_| rng.g()).collect();
        let hz = |p: [f64; 3]| {
            let u = [p[0] / 8.0, p[1] / 8.0, p[2] / 6.67];
            let i0 = u.map(|x| x.floor() as usize);
            let f = [u[0] - i0[0] as f64, u[1] - i0[1] as f64, u[2] - i0[2] as f64];
            let mut s = 0.0;
            for cr in 0..8 {
                let (a, b, cc) = (cr & 1, (cr >> 1) & 1, (cr >> 2) & 1);
                let w = (if a == 1 { f[0] } else { 1.0 - f[0] }) * (if b == 1 { f[1] } else { 1.0 - f[1] }) * (if cc == 1 { f[2] } else { 1.0 - f[2] });
                s += w * lat[(i0[0] + a).min(8) + 9 * ((i0[1] + b).min(7) + 8 * (i0[2] + cc).min(6))];
            }
            s.abs() + 0.3
        };
        let mut body = vec![false; nv];
        let mut lung = vec![false; nv];
        let mut trachea = vec![false; nv];
        let mut h = vec![C32::new(0.0, 0.0); nv * nc * nr * ne];
        let mut dist_body = vec![f64::INFINITY; nv];
        for r in 0..nr {
            let sh = r as f64;
            let bset: Vec<bool> = (0..nv).map(|q| ell(pos(q), sh, 22.0, 17.0)).collect();
            let db = edt_mm(&bset.iter().map(|&b| !b).collect::<Vec<_>>(), d, [2.0; 3]);
            for q in 0..nv {
                let p = pos(q);
                dist_body[q] = dist_body[q].min(db[q] as f64);
                // receive shading: bright at y = 0 (anterior coil), 4× darker at the far side
                let shade = 0.25 + 0.75 * (1.0 - p[1] / d[1] as f64).powi(2);
                let (mut m, mut tis) = (0.0, false);
                if bset[q] {
                    m = 1.0;
                    tis = true;
                    // inner wall boundary: the lungs and trachea sit inside a 6 mm wall
                    if in_lung(p, sh) {
                        m = if in_vessel(p, sh) { 1.0 } else { 0.03 };
                    }
                    if in_trachea(p, sh) {
                        m = 0.0;
                        tis = false;
                    }
                }
                if r == 0 {
                    body[q] = tis;
                    lung[q] = bset[q] && in_lung(p, sh) && !in_trachea(p, sh);
                    trachea[q] = bset[q] && in_trachea(p, sh);
                } else {
                    body[q] |= tis;
                }
                let sig = m * shade;
                // haze: coherent over frames and encodes, in air only, decaying over ~20 mm from the body
                let air_haze = if !bset[q] { haze * shade * hz(p) * (-(db[q] as f64) / 8.0).exp() } else { 0.0 };
                for c in 0..nc {
                    for e in 0..ne {
                        let z = num_complex::Complex64::from_polar(sig + air_haze, 0.4 + 0.2 * e as f64);
                        h[lay.img(r, c, e) * nv + q] = C32::new((z.re + 0.01 * rng.g()) as f32, (z.im + 0.01 * rng.g()) as f32);
                    }
                }
            }
        }
        let far_air = (0..nv).map(|q| dist_body[q] > 10.0).collect();
        Phantom { d, h, lay, body, lung, trachea, far_air }
    }

    fn count(m: &AirMap, set: &[bool]) -> (usize, usize) {
        let n = set.iter().filter(|&&b| b).count();
        (m.w.iter().zip(set).filter(|(&w, &s)| s && w > 0.0).count(), n)
    }

    #[test]
    fn edt_matches_brute_force() {
        let d = [9usize, 7, 5];
        let nv = d.iter().product::<usize>();
        let mut rng = Rng(7);
        let set: Vec<bool> = (0..nv).map(|_| rng.u() < 0.85).collect();
        let vox = [1.0, 2.0, 3.0];
        let e = edt_mm(&set, d, vox);
        let p = |q: usize| [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
        for q in 0..nv {
            let want = if !set[q] { 0.0 } else { (0..nv).filter(|&o| !set[o]).map(|o| (0..3).map(|a| ((p(q)[a] as f64 - p(o)[a] as f64) * vox[a]).powi(2)).sum::<f64>().sqrt()).fold(f64::INFINITY, f64::min) };
            assert!((e[q] as f64 - want).abs() < 1e-4, "edt at {:?}: {} vs {}", p(q), e[q], want);
        }
    }

    #[test]
    fn air_outside_found_lung_and_airway_kept_monotone_refresh() {
        // weak haze (a tenth of the tissue level next to the body, decaying): most of the far air is found
        let ph = phantom(0.1, 0x5eed);
        let nv = ph.d.iter().product::<usize>();
        let opts = AirOpts::default();
        let m0 = estimate(&ph.h, &ph.lay, &opts, None, 0);
        let (fa, nb) = count(&m0, &ph.body);
        let (ln, nl) = count(&m0, &ph.lung);
        let (tr, nt) = count(&m0, &ph.trachea);
        let (far, nfar) = count(&m0, &ph.far_air);
        println!("weak haze: level {:.3}, low {}, seeds {}, candidates {}, zeroed {}; far air found {far}/{nfar}; tissue {fa}/{nb}, lung {ln}/{nl}, trachea {tr}/{nt}", m0.level, m0.stats.low, m0.stats.seeds, m0.stats.candidates, m0.stats.marked);
        assert_eq!(fa, 0, "tissue marked as air");
        assert_eq!(ln, 0, "lung marked as air");
        assert!(far as f64 > 0.7 * nfar as f64, "far air not found: {far} / {nfar}");
        // normalisation equivariance: scaling the images changes nothing
        let hs: Vec<C32> = ph.h.iter().map(|c| C32::new(c.re * 4.0, c.im * 4.0)).collect();
        let ms = estimate(&hs, &ph.lay, &opts, None, 0);
        assert_eq!(ms.w, m0.w, "map depends on the image scale");
        // refresh after the projection: monotone, nothing released (the zeroed voxels stay ≈ 0)
        let mut x = ph.h.clone();
        apply(&mut x, &ph.lay, &m0);
        let m1 = estimate(&x, &ph.lay, &opts, Some(&m0), 3);
        assert!((0..nv).all(|q| m1.w[q] >= m0.w[q] && (!m0.cand[q] || m1.cand[q])), "refresh not monotone");
        assert_eq!(m1.stats.released, 0);
        assert_eq!(count(&m1, &ph.body).0, 0, "tissue marked after refresh");
        // signal that reappears inside the zeroed region (a gradient step puts tissue back): released, protected,
        // and not marked again by later refreshes even when it is dark again
        let blob: Vec<bool> = (0..nv).map(|q| m1.w[q] >= 1.0 && { let p = [q % ph.d[0], (q / ph.d[0]) % ph.d[1], q / (ph.d[0] * ph.d[1])]; p[0] >= 2 && p[0] < 6 && p[1] >= 2 && p[1] < 6 && p[2] >= 16 && p[2] < 20 }).collect();
        assert!(blob.iter().filter(|&&b| b).count() > 20, "test blob not in the zeroed air");
        let mut y = x.clone();
        apply(&mut y, &ph.lay, &m1);
        for i in 0..y.len() / nv {
            for q in (0..nv).filter(|&q| blob[q]) {
                y[i * nv + q] = C32::new(0.8, 0.0);
            }
        }
        let m2 = estimate(&y, &ph.lay, &opts, Some(&m1), 6);
        assert!(m2.stats.released > 0 && (0..nv).filter(|&q| blob[q]).all(|q| m2.protect[q] && m2.w[q] == 0.0), "reappearing signal not released");
        let mut z = y.clone();
        for i in 0..z.len() / nv {
            for q in (0..nv).filter(|&q| blob[q]) {
                z[i * nv + q] = C32::new(0.0, 0.0);
            }
        }
        let m3 = estimate(&z, &ph.lay, &opts, Some(&m2), 9);
        assert!((0..nv).filter(|&q| blob[q]).all(|q| m3.w[q] == 0.0), "released voxels flipped back to air");
        println!("refreshes: zeroed {} → {} → {} → {}, released {}, protected {}", m0.stats.marked, m1.stats.marked, m2.stats.marked, m3.stats.marked, m2.stats.released, m3.stats.protected);
    }

    /// minimal Fortran-order npy writer (tests only)
    fn write_npy(path: &std::path::Path, descr: &str, shape: &[usize], data: &[u8]) {
        let shp = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ");
        let mut hdr = format!("{{'descr': '{descr}', 'fortran_order': True, 'shape': ({shp}), }}");
        while (10 + hdr.len() + 1) % 64 != 0 {
            hdr.push(' ');
        }
        hdr.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(hdr.len() as u16).to_le_bytes());
        out.extend_from_slice(hdr.as_bytes());
        out.extend_from_slice(data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn seed_files_round_trip_and_coarse_seeds() {
        let ph = phantom(0.1, 0x5eed);
        let d = ph.d;
        let opts = AirOpts::default();
        let m0 = estimate(&ph.h, &ph.lay, &opts, None, 4);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/airmap_test");
        std::fs::create_dir_all(&dir).unwrap();
        // 1. a saved map (+ sidecar) on the same grid: identical candidates and weights
        let f1 = dir.join("air_map_t.npy");
        write_npy(&f1, "<f4", &[d[0], d[1], d[2], 4], &m0.stacked().iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        std::fs::write(f1.with_extension("json"), serde_json::to_string(&sidecar(&m0, &opts, d, [2.0; 3])).unwrap()).unwrap();
        let (m1, desc) = load_seed(&f1, d, [2.0; 3], &opts).unwrap();
        println!("{desc}");
        assert!(m1.cand == m0.cand && m1.w == m0.w && m1.iteration == 4, "saved map not reproduced");
        // 2. a coarse mask (2× voxels): no tissue, most of the far air
        let dc = [d[0] / 2, d[1] / 2, d[2] / 2];
        let at = |p: [usize; 3]| p[0] + d[0] * (p[1] + d[1] * p[2]);
        let child = |q: usize, c: usize| { let p = [q % dc[0], (q / dc[0]) % dc[1], q / (dc[0] * dc[1])]; at([2 * p[0] + (c & 1), 2 * p[1] + ((c >> 1) & 1), 2 * p[2] + ((c >> 2) & 1)]) };
        let ncv = dc.iter().product::<usize>();
        let mask: Vec<f32> = (0..ncv).map(|q| (0..8).all(|c| m0.cand[child(q, c)]) as u8 as f32).collect();
        let f2 = dir.join("mask_coarse.npy");
        write_npy(&f2, "<f4", &dc, &mask.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let (m2, desc) = load_seed(&f2, d, [2.0; 3], &opts).unwrap();
        println!("{desc}");
        let (fa, _) = count(&m2, &ph.body);
        let (far, nfar) = count(&m2, &ph.far_air);
        assert!(fa == 0 && far as f64 > 0.7 * nfar as f64, "coarse mask seed: tissue {fa}, far air {far}/{nfar}");
        // 3. coarse complex images (2×2×2 means of every frame and encode): estimated on the coarse grid, resampled
        let nimg = ph.h.len() / at([0, 0, d[2]]);
        let nv = d.iter().product::<usize>();
        let mut hc = vec![0f32; 2 * ncv * nimg];
        for i in 0..nimg {
            for q in 0..ncv {
                let (mut re, mut im) = (0f32, 0f32);
                for c in 0..8 {
                    let v = ph.h[i * nv + child(q, c)];
                    re += v.re / 8.0;
                    im += v.im / 8.0;
                }
                hc[2 * (i * ncv + q)] = re;
                hc[2 * (i * ncv + q) + 1] = im;
            }
        }
        let f3 = dir.join("coarse_images.npy");
        // image order of the stack (frame-major, encodes inner) = (X,Y,Z,E,F) Fortran
        write_npy(&f3, "<c8", &[dc[0], dc[1], dc[2], ph.lay.ne, nimg / ph.lay.ne], &hc.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let (m3, desc) = load_seed(&f3, d, [2.0; 3], &opts).unwrap();
        println!("{desc}");
        let (fa, _) = count(&m3, &ph.body);
        let (ln, _) = count(&m3, &ph.lung);
        let (far, _) = count(&m3, &ph.far_air);
        assert!(fa + ln == 0 && far as f64 > 0.5 * nfar as f64, "coarse image seed: tissue {fa}, lung {ln}, far air {far}/{nfar}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn strong_haze_marks_nothing_wrong() {
        // haze as bright as the tissue next to the body: the estimator may find little, but never tissue or lung
        let ph = phantom(1.2, 0xbad5eed);
        let m = estimate(&ph.h, &ph.lay, &AirOpts::default(), None, 0);
        let (fa, _) = count(&m, &ph.body);
        let (ln, _) = count(&m, &ph.lung);
        let (far, nfar) = count(&m, &ph.far_air);
        println!("strong haze: zeroed {}, far air {far}/{nfar}, tissue {fa}, lung {ln}", m.stats.marked);
        assert_eq!(fa + ln, 0, "tissue or lung marked under strong haze");
        // a seed (e.g. from a converged coarse level) on a 2× coarser grid, resampled: kept where the iterate is dark,
        // eroded by half a coarse voxel, never in tissue
        let weak = phantom(0.1, 0x5eed);
        let fine = estimate(&weak.h, &weak.lay, &AirOpts::default(), None, 0);
        let dc = [32usize, 28, 20];
        let coarse_c: Vec<f32> = (0..dc.iter().product::<usize>())
            .map(|q| {
                let p = [q % dc[0], (q / dc[0]) % dc[1], q / (dc[0] * dc[1])];
                let all = (0..8).all(|c| {
                    let f = [2 * p[0] + (c & 1), 2 * p[1] + ((c >> 1) & 1), 2 * p[2] + ((c >> 2) & 1)];
                    fine.cand[f[0] + ph.d[0] * (f[1] + ph.d[1] * f[2])]
                });
                all as u8 as f32
            })
            .collect();
        let mut seed = AirMap::from_weights(coarse_c, 0).resample(dc, [4.0; 3], ph.d, [2.0; 3]);
        seed.refresh_weights(ph.d, [2.0; 3], &AirOpts::default());
        assert_eq!(count(&seed, &ph.body).0, 0, "resampled seed in tissue");
        // the seed is applied from the first iteration on, so the iterate the first estimate sees is zero there
        let mut x = ph.h.clone();
        apply(&mut x, &ph.lay, &seed);
        let ms = estimate(&x, &ph.lay, &AirOpts::default(), Some(&seed), 1);
        let (fa, _) = count(&ms, &ph.body);
        let (far2, _) = count(&ms, &ph.far_air);
        println!("seeded under strong haze: zeroed {} (seed {}), far air {far2}/{nfar}, released {}, tissue {fa}", ms.stats.marked, seed.stats.marked, ms.stats.released);
        assert_eq!(fa, 0);
        assert!(far2 > far, "the seed did not help");
    }
}
