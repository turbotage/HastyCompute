//! `slab2d_views` recipe: FastSurfer (Deep-MI, Apache-2.0) inference at 1 mm, from the published pipeline
//! (`FastSurferCNN/{data_loader/conform.py, data_loader/dataset.py, data_loader/data_utils.py, inference.py}`, v2.6.1):
//!
//! 1. conform to a 256³ 1 mm LIA grid (linear interpolation), intensities rescaled to 0..255 with FreeSurfer's robust
//!    scale (statistics of the *original* data, 99.9th percentile of a 1000-bin histogram), uint8, then /255
//! 2. three 2-D networks (axial, coronal, sagittal), each fed 7 adjacent slices (edge-clamped) as channels
//! 3. raw logits summed with weights 0.4 / 0.4 / 0.2 (the sagittal net has 51 hemisphere-merged classes, gathered to
//!    the full 79 first), argmax, class index → FreeSurfer label id, nearest-neighbour back onto the input grid
//!
//! Not included: FastSurfer's cortex re-lateralisation and segmentation clean-up, and its native sub-mm modes (the
//! volume is always conformed to 1 mm). The logit sum is kept in half precision like the original (79 × 256³ values).

use rayon::prelude::*;

use crate::backend::{Backend, Tensor};
use crate::geom::{conform_geometry, resample, Interp, Volume};
use crate::recipe::{Progress, RunOptions, Segmentation};
use crate::spec::ModelSpec;
use crate::{Error, Result};

/// FreeSurfer `getscale(data, 0, 255, f_low = 0, f_high = 0.999)`: returns `(src_min, scale)`.
pub fn getscale(data: &[f32]) -> (f64, f64) {
    let (mn, mx) = data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let (mn, mx) = (mn as f64, mx as f64);
    let nonzero = data.iter().filter(|v| v.abs() >= 1e-15).count();
    let total = data.len();
    const BINS: usize = 1000;
    let mut hist = [0usize; BINS];
    let w = (mx - mn) / BINS as f64;
    for &v in data {
        let b = if w > 0.0 { (((v as f64 - mn) / w) as usize).min(BINS - 1) } else { 0 };
        hist[b] += 1;
    }
    let mut cum = vec![0usize; BINS + 1];
    for i in 0..BINS {
        cum[i + 1] = cum[i] + hist[i];
    }
    let edge = |i: isize| -> f64 {
        let n = (BINS + 1) as isize;
        let i = if i < 0 { i + n } else { i } as usize;
        mn + (mx - mn) * i as f64 / BINS as f64
    };
    let upper_cutoff = total - ((1.0 - 0.999f64) * nonzero as f64) as usize;
    let src_min = edge(0);
    let src_max = match cum.iter().position(|&c| c >= upper_cutoff) {
        Some(i) => edge(i as isize - 2),
        None => edge(-1),
    };
    let scale = if src_min == src_max { 1.0 } else { 255.0 / (src_max - src_min) };
    (src_min, scale)
}

// --- IEEE half precision for the logit accumulator ---------------------------------------------------------------
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e = e - 127 + 15;
    if e >= 31 {
        return s | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return s;
        }
        let m = (m | 0x80_0000) >> (1 - e);
        let r = (m + 0xfff + ((m >> 13) & 1)) >> 13;
        return s | r as u16;
    }
    let r = ((e as u32) << 10 | m >> 13) + ((m & 0x1fff) + 0xfff + ((m >> 13) & 1) >> 13);
    s | r as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 as u32) << 23 | (m2 & 0x3ff) << 13
        }
    } else if e == 31 {
        s | 0x7f80_0000 | m << 13
    } else {
        s | (e + 127 - 15) << 23 | m << 13
    };
    f32::from_bits(bits)
}

const N: usize = 256;

/// voxel index of `(i, j, k)` = (L, I, A) in the conformed volume (L fastest)
#[inline]
fn vox(i: usize, j: usize, k: usize) -> usize {
    (k * N + j) * N + i
}

#[derive(Clone, Copy, PartialEq)]
enum View {
    Axial,
    Coronal,
    Sagittal,
}

impl View {
    fn name(self) -> &'static str {
        match self {
            View::Axial => "axial",
            View::Coronal => "coronal",
            View::Sagittal => "sagittal",
        }
    }
    /// voxel of the 2-D pixel `(h, w)` of slice `s`
    #[inline]
    fn voxel(self, s: usize, h: usize, w: usize) -> usize {
        match self {
            View::Coronal => vox(h, w, s),
            View::Axial => vox(w, s, h),
            View::Sagittal => vox(s, w, h),
        }
    }
}

pub fn segment_slab2d_views(backend: &mut dyn Backend, spec: &ModelSpec, image: &Volume, opts: &RunOptions, progress: Progress) -> Result<Segmentation> {
    if opts.probabilities {
        return Err(Error::Spec(format!("{}: probabilities are not available for slab2d_views", spec.name)));
    }
    let raw = &spec.raw;
    let nclass = raw["full_classes"].as_u64().ok_or_else(|| Error::Spec("full_classes missing".into()))? as usize;
    let gather: Vec<usize> = raw["sagittal_gather"].as_array().ok_or_else(|| Error::Spec("sagittal_gather missing".into()))?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
    if gather.len() != nclass {
        return Err(Error::Spec("sagittal_gather must have full_classes entries".into()));
    }
    let class_ids: Vec<u16> = match raw["class_ids"].as_array() {
        Some(a) if a.len() == nclass => a.iter().map(|v| v.as_u64().unwrap_or(0) as u16).collect(),
        _ => (0..nclass as u16).collect(),
    };
    let weight = |v: View| raw["view_weights"][v.name()].as_f64().unwrap_or(0.4) as f32;
    let nview_classes = |v: View| raw["view_classes"][v.name()].as_u64().map(|x| x as usize).unwrap_or(nclass);

    // 1. conform
    let grid = conform_geometry(&image.geom, 1.0, [N, N, N], "LIA")?;
    let (src_min, scale) = getscale(&image.data);
    let mapped = resample(image, &grid, Interp::Linear, 0.0)?;
    let nv = N * N * N;
    let input: Vec<f32> = mapped
        .data
        .par_iter()
        .map(|&m| {
            if (m as f64).abs() <= 1e-8 {
                0.0
            } else {
                ((scale * (m as f64 - src_min)).clamp(0.0, 255.0)).round() as f32 / 255.0
            }
        })
        .collect();

    // 2./3. views → half-precision logit accumulator, class-major
    let mut acc = vec![0u16; nclass * nv];
    let total = 3 * N;
    let mut done = 0;
    let mut slab = vec![0.0f32; 7 * N * N];
    let timing = std::env::var_os("HASTY_INFER_TIMING").is_some();
    let (mut t_slab, mut t_fwd, mut t_acc) = (0.0f64, 0.0f64, 0.0f64);
    for view in [View::Axial, View::Coronal, View::Sagittal] {
        let kview = nview_classes(view);
        let wgt = weight(view);
        let role = view.name();
        for s in 0..N {
            let t0 = std::time::Instant::now();
            for t in 0..7 {
                let ss = (s as isize + t as isize - 3).clamp(0, N as isize - 1) as usize;
                let dst = &mut slab[t * N * N..(t + 1) * N * N];
                for h in 0..N {
                    for w in 0..N {
                        dst[h * N + w] = input[view.voxel(ss, h, w)];
                    }
                }
            }
            t_slab += t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            let out = backend.forward(role, &Tensor::new(vec![1, 7, N, N], slab.clone()))?;
            t_fwd += t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            if out.data.len() != kview * N * N {
                return Err(Error::Backend(format!("{}: {role} output has {} values, expected {kview} × {N} × {N}", backend.name(), out.data.len())));
            }
            let sagittal = view == View::Sagittal;
            acc.par_chunks_mut(nv).enumerate().for_each(|(c, plane)| {
                let src = &out.data[(if sagittal { gather[c] } else { c }) * N * N..][..N * N];
                for h in 0..N {
                    for w in 0..N {
                        let p = &mut plane[view.voxel(s, h, w)];
                        *p = f32_to_f16(f16_to_f32(*p) + wgt * src[h * N + w]);
                    }
                }
            });
            t_acc += t0.elapsed().as_secs_f64();
            done += 1;
            if !progress(done, total) {
                return Err(Error::Cancelled);
            }
        }
        if timing {
            eprintln!("slab2d {role}: slab {t_slab:.1}s forward {t_fwd:.1}s accumulate {t_acc:.1}s");
        }
    }

    // argmax over classes, streaming one class plane at a time
    let mut best = vec![f32::NEG_INFINITY; nv];
    let mut idx = vec![0u8; nv];
    for c in 0..nclass {
        let plane = &acc[c * nv..(c + 1) * nv];
        best.par_iter_mut().zip(idx.par_iter_mut()).zip(plane.par_iter()).for_each(|((b, i), &p)| {
            let v = f16_to_f32(p);
            if v > *b {
                *b = v;
                *i = c as u8;
            }
        });
    }
    drop(acc);
    let mut ids: Vec<u16> = idx.par_iter().map(|&i| class_ids[i as usize]).collect();
    if raw["relateralize"].as_bool().unwrap_or(false) {
        split_cortex_labels(&mut ids, [N, N, N]);
    }
    let lab: Vec<f32> = ids.iter().map(|&i| i as f32).collect();

    // 4. back onto the input grid
    let lv = Volume::new(grid, lab)?;
    let back = resample(&lv, &image.geom, Interp::Nearest, 0.0)?;
    Ok(Segmentation { labels: back.data.iter().map(|&x| x as u16).collect(), probabilities: None })
}


// --- cortex re-lateralisation (FastSurferCNN/data_loader/data_utils.py `split_cortex_labels`) -------------------------

/// Connected components (26-connectivity) of the voxels equal to `value`: `(centroid, size, voxel indices)` per component.
fn components(lab: &[u16], value: u16, d: [usize; 3]) -> Vec<([f64; 3], Vec<usize>)> {
    let [nx, ny, nz] = d;
    let mut seen = vec![false; lab.len()];
    let mut out = Vec::new();
    for start in 0..lab.len() {
        if lab[start] != value || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let mut members = Vec::new();
        let mut sum = [0.0f64; 3];
        while let Some(v) = stack.pop() {
            members.push(v);
            let (x, y, z) = (v % nx, (v / nx) % ny, v / (nx * ny));
            sum[0] += z as f64;
            sum[1] += y as f64;
            sum[2] += x as f64;
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (xx, yy, zz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                        if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i64 || yy >= ny as i64 || zz >= nz as i64 {
                            continue;
                        }
                        let u = (zz as usize * ny + yy as usize) * nx + xx as usize;
                        if lab[u] == value && !seen[u] {
                            seen[u] = true;
                            stack.push(u);
                        }
                    }
                }
            }
        }
        let n = members.len() as f64;
        out.push(([sum[0] / n, sum[1] / n, sum[2] / n], members));
    }
    out
}

/// separable Gaussian filter (scipy `gaussian_filter`: truncate 4, reflect boundaries) of a mask scaled by 1000
fn gaussian3(mask: &[bool], d: [usize; 3], sigma: f64) -> Vec<f32> {
    let [nx, ny, nz] = d;
    let r = (4.0 * sigma + 0.5) as isize;
    let mut k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp()).collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    let reflect = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let p = 2 * n;
        let m = i.rem_euclid(p);
        (if m >= n { p - 1 - m } else { m }) as usize
    };
    let mut a: Vec<f32> = mask.iter().map(|&b| if b { 1000.0 } else { 0.0 }).collect();
    let mut b = vec![0.0f32; a.len()];
    for axis in 0..3 {
        let (stride, len) = match axis {
            0 => (1, nx),
            1 => (nx, ny),
            _ => (nx * ny, nz),
        };
        b.par_iter_mut().enumerate().for_each(|(v, o)| {
            let pos = match axis {
                0 => v % nx,
                1 => (v / nx) % ny,
                _ => v / (nx * ny),
            };
            let base = v - pos * stride;
            let mut acc = 0.0f64;
            for (t, w) in k.iter().enumerate() {
                acc += w * a[base + reflect(pos as isize + t as isize - r, len) * stride] as f64;
            }
            *o = acc as f32;
        });
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// Re-lateralise the cortical labels the network predicts hemisphere-merged (19 DKT parcels 1xxx → 2xxx for the right
/// hemisphere, by the distance of each component to the centroids of the largest left / right white-matter components),
/// then resolve four overlapping parcels with a smoothed left-vs-right white-matter split. `lab` holds FreeSurfer ids,
/// x fastest with dims `d`.
pub fn split_cortex_labels(lab: &mut [u16], d: [usize; 3]) {
    let largest = |id: u16| -> Option<[f64; 3]> { components(lab, id, d).into_iter().max_by_key(|(_, m)| m.len()).map(|(c, _)| c) };
    let (Some(rh), Some(lh)) = (largest(41), largest(2)) else { return };
    let dist = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    for &l in &[1003u16, 1006, 1007, 1008, 1009, 1011, 1015, 1018, 1019, 1020, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035] {
        let comps = components(lab, l, d);
        for (c, members) in comps {
            if dist(c, rh) < dist(c, lh) {
                for v in members {
                    lab[v] = l + 1000;
                }
            }
        }
    }
    let lh_mask: Vec<bool> = lab.iter().map(|&v| v == 2).collect();
    let rh_mask: Vec<bool> = lab.iter().map(|&v| v == 41).collect();
    let (gl, gr) = (gaussian3(&lh_mask, d, 3.0), gaussian3(&rh_mask, d, 3.0));
    for &pl in &[1011u16, 1019, 1026, 1029] {
        let pr = pl + 1000;
        for v in 0..lab.len() {
            if lab[v] == pl || lab[v] == pr {
                lab[v] = if gl[v] >= gr[v] { pl } else { pr }; // argmax over (lh, rh): ties go to lh
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_roundtrip() {
        for &x in &[0.0f32, 1.0, -1.0, 0.5, 3.14159, -100.25, 1e-3, 60000.0, 65504.0, 1e-7] {
            let r = f16_to_f32(f32_to_f16(x));
            assert!((r - x).abs() <= x.abs() * 1e-3 + 1e-7, "{x} -> {r}");
        }
        assert!(f16_to_f32(f32_to_f16(1e6)).is_infinite());
    }

    #[test]
    fn cortex_split_by_hemisphere() {
        // 40³ grid: left WM blob around x=10, right around x=30, one cortical component near each, one 1011 straddling
        let d = [40usize, 20, 20];
        let mut lab = vec![0u16; d[0] * d[1] * d[2]];
        let at = |x: usize, y: usize, z: usize| (z * d[1] + y) * d[0] + x;
        for z in 5..15 {
            for y in 5..15 {
                for x in 4..16 {
                    lab[at(x, y, z)] = 2;
                }
                for x in 24..36 {
                    lab[at(x, y, z)] = 41;
                }
            }
        }
        for z in 5..8 {
            lab[at(2, 10, z)] = 1003; // near left WM
            lab[at(37, 10, z)] = 1003; // near right WM
        }
        split_cortex_labels(&mut lab, d);
        assert_eq!(lab[at(2, 10, 5)], 1003);
        assert_eq!(lab[at(37, 10, 5)], 2003);
    }

    #[test]
    fn scale_matches_formula() {
        // 1000 voxels 0..999 plus a hot spike: the spike must not set the upper bound
        let mut d: Vec<f32> = (0..100_000).map(|i| (i % 1000) as f32).collect();
        d.push(1e6);
        let (mn, sc) = getscale(&d);
        assert_eq!(mn, 0.0);
        assert!(255.0 / sc < 1e6 && 255.0 / sc > 500.0, "src_max {}", 255.0 / sc);
    }
}
