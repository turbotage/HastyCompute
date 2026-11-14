//! Pre/post-processing around a [`Backend`].
//!
//! `patch3d`: single 3-D network applied to overlapping patches (nnU-Net style): resample the image onto the model
//! grid, normalise, tile with Gaussian-weighted blending, optional mirror test-time augmentation, argmax, resample
//! the labels back onto the input grid (nearest).

use crate::backend::{Backend, Tensor};
use crate::geom::{covering_geometry, crop_geometry, rescaled_geometry, resample, Interp, Volume};
use crate::spec::{ModelSpec, Norm};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct RunOptions {
    /// patch overlap in [0, 1); `None`: the model's own
    pub overlap: Option<f32>,
    /// average over mirrored copies (none + one per axis)
    pub tta: bool,
    /// also return the class probabilities (nclass × input voxels)
    pub probabilities: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { overlap: None, tta: false, probabilities: false }
    }
}

pub struct Segmentation {
    /// label per voxel of the input grid
    pub labels: Vec<u16>,
    /// `[class][voxel]` when requested
    pub probabilities: Option<Vec<Vec<f32>>>,
}

/// `done` of `total` steps finished; return `false` to cancel
pub type Progress<'a> = &'a mut dyn FnMut(usize, usize) -> bool;

/// `shape` = C-order `[z, y, x]` of `data` (needed by the 3-D filters)
pub fn normalize(data: &mut [f32], norm: &Norm, shape: [usize; 3]) {
    let positives = || -> Vec<f32> { data.iter().copied().filter(|&x| x > 0.0 && x.is_finite()).collect() };
    match norm {
        Norm::None => {}
        Norm::T1wPvs => {
            let out = crate::pvsnorm::t1w_pvs_norm(data, shape);
            data.copy_from_slice(&out);
        }
        Norm::T2wPvs => {
            let out = crate::pvsnorm::t2w_pvs_norm(data, shape);
            data.copy_from_slice(&out);
        }
        Norm::ZScore { nonzero_mask } => {
            let sel = |x: &f32| !*nonzero_mask || *x != 0.0;
            let n = data.iter().filter(|x| sel(x)).count().max(1) as f64;
            let mean = data.iter().filter(|x| sel(x)).map(|&x| x as f64).sum::<f64>() / n;
            let var = data.iter().filter(|x| sel(x)).map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
            let sd = var.sqrt().max(1e-8);
            for x in data.iter_mut() {
                if !*nonzero_mask || *x != 0.0 {
                    *x = ((*x as f64 - mean) / sd) as f32;
                }
            }
        }
        Norm::Ct { lo, hi, mean, std } => {
            let sd = std.max(1e-8);
            for x in data.iter_mut() {
                *x = (x.clamp(*lo, *hi) - mean) / sd;
            }
        }
        Norm::Percentile { lo, hi } => {
            let mut pos = positives();
            if pos.is_empty() {
                return;
            }
            pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f32| pos[(((pos.len() - 1) as f32) * p / 100.0).round() as usize];
            let (a, b) = (q(*lo), q(*hi));
            let s = (b - a).max(1e-8);
            for x in data.iter_mut() {
                *x = ((*x - a) / s).clamp(0.0, 1.0);
            }
        }
        Norm::ScaleMax { hi } => {
            let mut pos = positives();
            if pos.is_empty() {
                return;
            }
            pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let b = pos[(((pos.len() - 1) as f32) * hi / 100.0).round() as usize].max(1e-8);
            for x in data.iter_mut() {
                *x = (*x / b).clamp(0.0, 1.0);
            }
        }
    }
}

/// patch start positions along an axis of length `n` for patch `p` (nnU-Net: `ceil((n-p)/(p·step))+1` evenly spaced
/// starts from 0 to n-p, step = 1 - overlap)
pub fn starts(n: usize, p: usize, overlap: f32) -> Vec<usize> {
    if n <= p {
        return vec![0];
    }
    let target = (p as f64 * (1.0 - overlap as f64)).max(1.0);
    let steps = (((n - p) as f64 / target).ceil() as usize + 1).max(2);
    let actual = (n - p) as f64 / (steps - 1) as f64;
    (0..steps).map(|i| (actual * i as f64).round() as usize).collect()
}

/// separable Gaussian importance window (sigma = patch/8, as nnU-Net), floored so no voxel gets zero weight
fn window(patch: [usize; 3]) -> Vec<f32> {
    let g = |n: usize| -> Vec<f32> {
        let c = (n as f32 - 1.0) / 2.0;
        let s = (n as f32 / 8.0).max(1e-3);
        (0..n).map(|i| (-(i as f32 - c).powi(2) / (2.0 * s * s)).exp()).collect()
    };
    let (gx, gy, gz) = (g(patch[0]), g(patch[1]), g(patch[2]));
    let mut w = Vec::with_capacity(patch.iter().product());
    for z in 0..patch[2] {
        for y in 0..patch[1] {
            for x in 0..patch[0] {
                w.push((gx[x] * gy[y] * gz[z]).max(1e-4));
            }
        }
    }
    w
}

fn softmax_inplace(t: &mut [f32], k: usize, nv: usize) {
    for v in 0..nv {
        let m = (0..k).map(|c| t[c * nv + v]).fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for c in 0..k {
            let e = (t[c * nv + v] - m).exp();
            t[c * nv + v] = e;
            s += e;
        }
        for c in 0..k {
            t[c * nv + v] /= s;
        }
    }
}

/// mirror a `[pz, py, px]` block (one channel) along axes (bit 0 = x, 1 = y, 2 = z)
fn flip(block: &[f32], p: [usize; 3], mask: u8, out: &mut [f32]) {
    let (px, py, pz) = (p[0], p[1], p[2]);
    for z in 0..pz {
        let sz = if mask & 4 != 0 { pz - 1 - z } else { z };
        for y in 0..py {
            let sy = if mask & 2 != 0 { py - 1 - y } else { y };
            for x in 0..px {
                let sx = if mask & 1 != 0 { px - 1 - x } else { x };
                out[(z * py + y) * px + x] = block[(sz * py + sy) * px + sx];
            }
        }
    }
}

/// Tiling parameters of [`tile_probabilities`].
pub struct Tile<'a> {
    /// networks whose raw outputs are averaged (nnU-Net folds)
    pub roles: &'a [String],
    pub k: usize,
    /// `[x, y, z]`
    pub patch: [usize; 3],
    pub overlap: f32,
    /// average over the 8 mirrorings
    pub tta: bool,
    /// map the averaged raw outputs through a softmax
    pub softmax: bool,
}

/// Class probabilities `[k * nvoxels]` of model-grid channels (already normalised; each `nx·ny·nz`, x fastest) by
/// patch tiling.
pub fn tile_probabilities(backend: &mut dyn Backend, chans: &[Vec<f32>], dims: [usize; 3], t: &Tile, progress: Progress) -> Result<Vec<f32>> {
    let [nx, ny, nz] = dims;
    let (k, patch) = (t.k, t.patch);
    let nc = chans.len();
    if nc == 0 || chans.iter().any(|c| c.len() != nx * ny * nz) {
        return Err(Error::Geometry("tile_probabilities: channels must all have nx·ny·nz values".into()));
    }
    // pad to at least one patch; centred like nnU-Net's pad_nd_image: (diff / 2) before, the rest after
    let (px, py, pz) = (nx.max(patch[0]), ny.max(patch[1]), nz.max(patch[2]));
    let (ox, oy, oz) = ((px - nx) / 2, (py - ny) / 2, (pz - nz) / 2);
    let padded: Vec<Vec<f32>> = chans
        .iter()
        .map(|c| {
            let mut p = vec![0.0f32; px * py * pz];
            for z in 0..nz {
                for y in 0..ny {
                    let s = (z * ny + y) * nx;
                    let d = ((z + oz) * py + y + oy) * px + ox;
                    p[d..d + nx].copy_from_slice(&c[s..s + nx]);
                }
            }
            p
        })
        .collect();
    let nvp = px * py * pz;
    let mut acc = vec![0.0f32; k * nvp];
    let mut wsum = vec![0.0f32; nvp];
    let w = window(patch);
    let npv = patch[0] * patch[1] * patch[2];
    let (sx, sy, sz) = (starts(px, patch[0], t.overlap), starts(py, patch[1], t.overlap), starts(pz, patch[2], t.overlap));
    let flips: Vec<u8> = if t.tta { (0..8).collect() } else { vec![0] };
    let total = sx.len() * sy.len() * sz.len() * flips.len() * t.roles.len();
    let nrun = (flips.len() * t.roles.len()) as f32;
    let mut done = 0;
    let (mut block, mut fl) = (vec![0.0f32; npv], vec![0.0f32; npv]);
    let mut input = vec![0.0f32; nc * npv];
    for &z0 in &sz {
        for &y0 in &sy {
            for &x0 in &sx {
                let mut sum = vec![0.0f32; k * npv];
                for &m in &flips {
                    for (ci, pc) in padded.iter().enumerate() {
                        for z in 0..patch[2] {
                            for y in 0..patch[1] {
                                let s = ((z0 + z) * py + y0 + y) * px + x0;
                                let d = (z * patch[1] + y) * patch[0];
                                block[d..d + patch[0]].copy_from_slice(&pc[s..s + patch[0]]);
                            }
                        }
                        flip(&block, patch, m, &mut fl);
                        input[ci * npv..(ci + 1) * npv].copy_from_slice(&fl);
                    }
                    let tin = Tensor::new(vec![1, nc, patch[2], patch[1], patch[0]], input.clone());
                    for role in t.roles {
                        let out = backend.forward(role, &tin)?;
                        if out.data.len() != k * npv {
                            return Err(Error::Backend(format!("{}: output has {} values, expected {k} × {npv} (shape {:?})", backend.name(), out.data.len(), out.shape)));
                        }
                        for c in 0..k {
                            let (src, dst) = (&out.data[c * npv..(c + 1) * npv], &mut sum[c * npv..(c + 1) * npv]);
                            // undo the mirror (flips are involutions)
                            flip(src, patch, m, &mut fl);
                            for (d, s) in dst.iter_mut().zip(&fl) {
                                *d += s / nrun;
                            }
                        }
                        done += 1;
                        if !progress(done, total) {
                            return Err(Error::Cancelled);
                        }
                    }
                }
                for z in 0..patch[2] {
                    for y in 0..patch[1] {
                        let base = ((z0 + z) * py + y0 + y) * px + x0;
                        for x in 0..patch[0] {
                            let wi = w[(z * patch[1] + y) * patch[0] + x];
                            wsum[base + x] += wi;
                            for c in 0..k {
                                acc[c * nvp + base + x] += wi * sum[c * npv + (z * patch[1] + y) * patch[0] + x];
                            }
                        }
                    }
                }
            }
        }
    }
    // crop the padding, normalise by the weights; raw outputs (logits) are averaged and mapped once at the end
    let nv = nx * ny * nz;
    let mut out = vec![0.0f32; k * nv];
    for c in 0..k {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let p = ((z + oz) * py + y + oy) * px + x + ox;
                    out[c * nv + (z * ny + y) * nx + x] = acc[c * nvp + p] / wsum[p];
                }
            }
        }
    }
    if t.softmax {
        softmax_inplace(&mut out, k, nv);
    }
    Ok(out)
}

/// bounding box `[lo, hi)` (x, y, z) of the voxels that are non-zero in any channel; the full volume when there are none
fn nonzero_bbox(chans: &[&Volume]) -> ([usize; 3], [usize; 3]) {
    let [nx, ny, nz] = chans[0].geom.dims;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for z in 0..nz {
        for y in 0..ny {
            let r = (z * ny + y) * nx;
            let mut first = usize::MAX;
            let mut last = 0;
            for c in chans {
                let row = &c.data[r..r + nx];
                if let (Some(f), Some(l)) = (row.iter().position(|&a| a != 0.0), row.iter().rposition(|&a| a != 0.0)) {
                    first = first.min(f);
                    last = last.max(l);
                }
            }
            if first != usize::MAX {
                for (d, c) in [(0, first), (0, last), (1, y), (2, z)] {
                    lo[d] = lo[d].min(c);
                    hi[d] = hi[d].max(c + 1);
                }
            }
        }
    }
    if lo[0] == usize::MAX {
        ([0; 3], chans[0].geom.dims)
    } else {
        (lo, hi)
    }
}

fn crop_volume(v: &Volume, lo: [usize; 3], hi: [usize; 3]) -> Result<Volume> {
    let [nx, ny, _] = v.geom.dims;
    let g = crop_geometry(&v.geom, lo, hi);
    let mut d = Vec::with_capacity(g.numel());
    for z in lo[2]..hi[2] {
        for y in lo[1]..hi[1] {
            let r = (z * ny + y) * nx;
            d.extend_from_slice(&v.data[r + lo[0]..r + hi[0]]);
        }
    }
    Volume::new(g, d)
}

/// `patch3d` on co-registered input channels given on any grid, in nnU-Net's order: crop to the non-zero box,
/// resample to the model grid, normalise per channel, tile, argmax, resample back, paste into the input grid.
///
/// Model grid: `orientation: "native"` without `spacing_mm` runs on the stored grid as is; with `spacing_mm` on the
/// same axes resampled to that voxel size (what nnU-Net does); any other orientation resamples to an axis-aligned
/// grid of that orientation (and the input's own spacing if `spacing_mm` is absent).
pub fn segment_patch3d(backend: &mut dyn Backend, spec: &ModelSpec, images: &[Volume], opts: &RunOptions, progress: Progress) -> Result<Segmentation> {
    let patch = spec.patch.ok_or_else(|| Error::Spec(format!("{}: patch3d needs \"patch\"", spec.name)))?;
    let k = spec.num_classes();
    if k < 2 {
        return Err(Error::Spec(format!("{}: patch3d needs labels with ≥ 2 classes", spec.name)));
    }
    if images.len() != spec.in_channels {
        return Err(Error::Spec(format!("{}: {} input channels expected, got {}", spec.name, spec.in_channels, images.len())));
    }
    let g0 = &images[0].geom;
    if images.iter().any(|v| v.geom != *g0) {
        return Err(Error::Geometry("all input channels must share one grid".into()));
    }
    let refs: Vec<&Volume> = images.iter().collect();
    let (lo, hi) = if spec.crop_nonzero { nonzero_bbox(&refs) } else { ([0; 3], g0.dims) };
    let cropped: Vec<Volume> = images.iter().map(|v| crop_volume(v, lo, hi)).collect::<Result<_>>()?;
    let cg = cropped[0].geom.clone();
    let native = spec.orientation.eq_ignore_ascii_case("native");
    let grid = if native {
        match spec.spacing_mm {
            Some(s) => rescaled_geometry(&cg, s),
            None => cg.clone(),
        }
    } else {
        covering_geometry(&cg, spec.spacing_mm.unwrap_or_else(|| cg.spacing()), &spec.orientation)?
    };
    let same = grid == cg;
    let mut chans: Vec<Vec<f32>> = Vec::with_capacity(cropped.len());
    for (c, v) in cropped.iter().enumerate() {
        let mut d = if same { v.data.clone() } else { resample(v, &grid, Interp::Linear, 0.0)?.data };
        normalize(&mut d, spec.norm_for(c), [grid.dims[2], grid.dims[1], grid.dims[0]]);
        chans.push(d);
    }
    let overlap = opts.overlap.unwrap_or(spec.overlap);
    let tile = Tile { roles: &spec.ensemble, k, patch, overlap, tta: opts.tta, softmax: spec.softmax };
    let probs = tile_probabilities(backend, &chans, grid.dims, &tile, progress)?;
    let nv = grid.numel();
    let mut lab = vec![0.0f32; nv];
    for v in 0..nv {
        let (mut best, mut bc) = (f32::NEG_INFINITY, 0usize);
        for c in 0..k {
            let p = probs[c * nv + v];
            if p > best {
                best = p;
                bc = c;
            }
        }
        lab[v] = bc as f32;
    }
    // back onto the crop grid, then paste into the input grid (class 0 / probability 1 for class 0 outside the crop)
    let back = |data: Vec<f32>, interp: Interp| -> Result<Vec<f32>> {
        if same {
            Ok(data)
        } else {
            Ok(resample(&Volume::new(grid.clone(), data)?, &cg, interp, 0.0)?.data)
        }
    };
    let [gx, gy, _] = g0.dims;
    let cd = cg.dims;
    let ncv = cg.numel();
    let nfull = g0.numel();
    let paste = |crop: &[f32], fill: f32| -> Vec<f32> {
        if cd == g0.dims {
            return crop.to_vec();
        }
        let mut f = vec![fill; nfull];
        for z in 0..cd[2] {
            for y in 0..cd[1] {
                let dst = ((z + lo[2]) * gy + y + lo[1]) * gx + lo[0];
                let src = (z * cd[1] + y) * cd[0];
                f[dst..dst + cd[0]].copy_from_slice(&crop[src..src + cd[0]]);
            }
        }
        f
    };
    let labels = paste(&back(lab, Interp::Nearest)?, 0.0);
    let probabilities = if opts.probabilities {
        let mut all = Vec::with_capacity(k);
        for c in 0..k {
            let pc = back(probs[c * nv..(c + 1) * nv].to_vec(), Interp::Linear)?;
            debug_assert_eq!(pc.len(), ncv);
            all.push(paste(&pc, if c == 0 { 1.0 } else { 0.0 }));
        }
        Some(all)
    } else {
        None
    };
    Ok(Segmentation { labels: labels.iter().map(|&x| x as u16).collect(), probabilities })
}

/// Run the recipe named by the spec (`patch3d`, `slab2d_views`) on co-registered input channels.
pub fn segment(backend: &mut dyn Backend, spec: &ModelSpec, images: &[Volume], opts: &RunOptions, progress: Progress) -> Result<Segmentation> {
    match spec.recipe.as_str() {
        "patch3d" => segment_patch3d(backend, spec, images, opts, progress),
        "slab2d_views" => {
            if images.len() != 1 {
                return Err(Error::Spec(format!("{}: slab2d_views takes one input image", spec.name)));
            }
            crate::fastsurfer::segment_slab2d_views(backend, spec, &images[0], opts, progress)
        }
        r => Err(Error::Spec(format!("{}: unknown recipe {r:?}", spec.name))),
    }
}
