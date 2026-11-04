//! Learned 4D-flow prior for P²nP: adapter between hasty image stacks and the exported O_h-equivariant 7-point
//! denoiser (training and export: `rustlib/tools/denoiser`; design: `research/denoiser_design.md` §10).
//!
//! Hasty stack: `images[frame * 7 + e]`, each x-fastest (Fortran) `[X, Y, Z]`, encodes
//! `[ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]`. Read as a C-order tensor such an image is `[Z, Y, X]`; the model is
//! exactly equivariant to that axis permutation, so the adapter only reorders the encode pairs to
//! `[ref, z_lo, z_hi, y_lo, y_hi, x_lo, x_hi]` (no transpose) and zero-pads every axis to a size ≡ 1 mod
//! `pad_multiple`. Phase convention of the model: x_e ∝ exp(+i kv v). hasty-mrsim produces exp(−i kv v) data
//! (`phase_sign = −1`), mapped exactly by the involution x_e → conj(x_e) x_ref / conj(x_ref) before and after.
//! This crate is torch-free; the TorchScript backend (tch-rs) lives in `crates/hasty-denoise-torch`, a separate
//! cargo workspace so that libtorch and its crates never enter the main build.

use std::path::Path;

use num_complex::Complex32 as C32;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model info: {0}")]
    Info(String),
    #[error("layout: {0}")]
    Layout(String),
    #[error("backend: {0}")]
    Backend(String),
}

/// `model.json` of an export directory.
#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub pad_multiple: usize,
    pub rf_space: usize,
    pub rf_time: usize,
    pub group: String,
}

impl ModelInfo {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.json"))?).map_err(|e| Error::Info(e.to_string()))?;
        let u = |k: &str| j[k].as_u64().map(|v| v as usize).ok_or_else(|| Error::Info(format!("missing {k}")));
        Ok(Self { pad_multiple: u("pad_multiple")?, rf_space: u("rf_space")?, rf_time: u("rf_time")?, group: j["cfg"]["group"].as_str().unwrap_or("?").to_string() })
    }
}

/// Number of encodes of the 7-point dual-VENC layout.
pub const NE: usize = 7;
/// hasty slot (x-major) of each model slot (z-major, because the C-order view of an x-fastest image is [Z, Y, X]).
pub const MODEL_SLOT_SRC: [usize; NE] = [0, 5, 6, 3, 4, 1, 2];

/// smallest n' ≥ n with n' ≡ 1 mod m
pub fn padded(n: usize, m: usize) -> usize {
    n + (m - (n - 1) % m) % m
}

/// x_e → conj(x_e) · x_ref / conj(x_ref) for e ≥ 1 at voxel q of every frame (v → −v for all axes; an involution)
pub fn flip_velocity_sign(images: &mut [C32], nv: usize, frames: usize) {
    for t in 0..frames {
        let (r, rest) = images[t * NE * nv..(t + 1) * NE * nv].split_at_mut(nv);
        for q in 0..nv {
            let a = r[q].norm_sqr();
            let u2 = if a > 0.0 { r[q] * r[q] / a } else { C32::new(1.0, 0.0) };
            for e in 0..NE - 1 {
                let z = &mut rest[e * nv + q];
                *z = z.conj() * u2;
            }
        }
    }
}

/// Model tensor `[1, 7, 2, T, Zp, Yp, Xp]` (C-order f32) of a hasty stack; returns the data and `[Zp, Yp, Xp]`.
pub fn to_model(images: &[C32], dims: [usize; 3], frames: usize, pad_multiple: usize, phase_sign: i8) -> Result<(Vec<f32>, [usize; 3]), Error> {
    let nv = dims[0] * dims[1] * dims[2];
    if images.len() != NE * frames * nv {
        return Err(Error::Layout(format!("stack has {} values, expected 7 × {frames} × {nv}", images.len())));
    }
    let mut src;
    let images = if phase_sign < 0 {
        src = images.to_vec();
        flip_velocity_sign(&mut src, nv, frames);
        &src[..]
    } else {
        images
    };
    let [x, y, z] = dims;
    let p = [padded(z, pad_multiple), padded(y, pad_multiple), padded(x, pad_multiple)];
    let np = p[0] * p[1] * p[2];
    let mut out = vec![0f32; NE * 2 * frames * np];
    for (s, &e) in MODEL_SLOT_SRC.iter().enumerate() {
        for t in 0..frames {
            let img = &images[(t * NE + e) * nv..(t * NE + e + 1) * nv];
            let base_re = ((s * 2) * frames + t) * np;
            let base_im = ((s * 2 + 1) * frames + t) * np;
            for k in 0..z {
                for j in 0..y {
                    let row_in = (k * y + j) * x;
                    let row_out = (k * p[1] + j) * p[2];
                    for i in 0..x {
                        let v = img[row_in + i];
                        out[base_re + row_out + i] = v.re;
                        out[base_im + row_out + i] = v.im;
                    }
                }
            }
        }
    }
    Ok((out, p))
}

/// Inverse of [`to_model`]: crop and write the model output back into the hasty stack.
pub fn from_model(t: &[f32], padded_dims: [usize; 3], images: &mut [C32], dims: [usize; 3], frames: usize, phase_sign: i8) -> Result<(), Error> {
    let nv = dims[0] * dims[1] * dims[2];
    let p = padded_dims;
    let np = p[0] * p[1] * p[2];
    if t.len() != NE * 2 * frames * np || images.len() != NE * frames * nv {
        return Err(Error::Layout("tensor / stack size mismatch".into()));
    }
    let [x, y, z] = dims;
    for (s, &e) in MODEL_SLOT_SRC.iter().enumerate() {
        for f in 0..frames {
            let img = &mut images[(f * NE + e) * nv..(f * NE + e + 1) * nv];
            let base_re = ((s * 2) * frames + f) * np;
            let base_im = ((s * 2 + 1) * frames + f) * np;
            for k in 0..z {
                for j in 0..y {
                    let row_in = (k * p[1] + j) * p[2];
                    let row_out = (k * y + j) * x;
                    for i in 0..x {
                        img[row_out + i] = C32::new(t[base_re + row_in + i], t[base_im + row_in + i]);
                    }
                }
            }
        }
    }
    if phase_sign < 0 {
        flip_velocity_sign(images, nv, frames);
    }
    Ok(())
}

/// Exact overlap tiling of one padded axis of length `n` (≡ 1 mod L) with tiles of `tile` voxels (≡ 1 mod L) and a
/// `margin` (multiple of L, ≥ the receptive-field radius): `(start, len, keep_lo, keep_hi)` per tile, absolute
/// indices. Tiles start on the coarsest lattice (≡ 0 mod L) and their kept centres cover [0, n), so stitching gives
/// exactly the whole-axis result (port of `denoise_volume`, tools/denoiser/model.py).
pub fn axis_tiles(n: usize, tile: usize, margin: usize, l: usize) -> Result<Vec<(usize, usize, usize, usize)>, Error> {
    if n <= tile {
        return Ok(vec![(0, n, 0, n)]);
    }
    if (tile - 1) % l != 0 || margin % l != 0 || tile <= 2 * margin + 1 || (n - 1) % l != 0 {
        return Err(Error::Layout(format!("tiling: n {n}, tile {tile} (≡ 1 mod {l}), margin {margin} (≡ 0 mod {l}, < (tile − 1) / 2)")));
    }
    let step = tile - 1 - 2 * margin;
    let mut starts: Vec<usize> = (0..).map(|i| i * step).take_while(|&s| s < n - tile).collect();
    starts.push(n - tile);
    starts.dedup();
    Ok(starts.into_iter().map(|s| (s, tile, if s == 0 { 0 } else { s + margin }, if s + tile >= n { n } else { s + tile - margin })).collect())
}

/// tile margin for a model: the receptive-field radius rounded up to the coarsest lattice
pub fn tile_margin(info: &ModelInfo) -> usize {
    info.rf_space.div_ceil(info.pad_multiple) * info.pad_multiple
}

/// valid tile size ≥ `want` (≡ 1 mod L, larger than 2 margin + 1)
pub fn tile_size(info: &ModelInfo, want: usize) -> usize {
    padded(want.max(2 * tile_margin(info) + info.pad_multiple + 1), info.pad_multiple)
}

/// copy the box [lo, lo + ext) of a `[C, P0, P1, P2]` tensor `src` (dims `sp`) into `dst` (dims `ext`, all channels)
pub fn crop(src: &[f32], sp: [usize; 3], lo: [usize; 3], ext: [usize; 3], dst: &mut [f32]) {
    let (ns, nd) = (sp[0] * sp[1] * sp[2], ext[0] * ext[1] * ext[2]);
    for c in 0..src.len() / ns {
        for a in 0..ext[0] {
            for b in 0..ext[1] {
                let si = c * ns + ((lo[0] + a) * sp[1] + lo[1] + b) * sp[2] + lo[2];
                let di = c * nd + (a * ext[1] + b) * ext[2];
                dst[di..di + ext[2]].copy_from_slice(&src[si..si + ext[2]]);
            }
        }
    }
}

/// write the kept box [klo, khi) (absolute) of a tile output (dims `ext`, at `lo`) into `[C, P0, P1, P2]` `dst`
pub fn paste(tile: &[f32], ext: [usize; 3], lo: [usize; 3], klo: [usize; 3], khi: [usize; 3], dst: &mut [f32], sp: [usize; 3]) {
    let (ns, nt) = (sp[0] * sp[1] * sp[2], ext[0] * ext[1] * ext[2]);
    let w = khi[2] - klo[2];
    for c in 0..tile.len() / nt {
        for a in klo[0]..khi[0] {
            for b in klo[1]..khi[1] {
                let ti = c * nt + ((a - lo[0]) * ext[1] + b - lo[1]) * ext[2] + klo[2] - lo[2];
                let di = c * ns + (a * sp[1] + b) * sp[2] + klo[2];
                dst[di..di + w].copy_from_slice(&tile[ti..ti + w]);
            }
        }
    }
}

/// Apply `f` (a model call on `[1, 7, 2, Tw, t0, t1, t2]`, given (input, Tw, [t0, t1, t2])) to a padded model
/// tensor `[1, 7, 2, T, P0, P1, P2]` (dims `p`, `frames` = T, the full cardiac cycle) by exact overlap tiles: spatial
/// tiles of about `tile` voxels (margin [`tile_margin`]) and cyclic time windows that keep `keep_frames` frames each
/// (margin `rf_time`; the model pads time circularly, which only touches the margin frames). `None` / sizes that
/// cover the volume: one call along that dimension. Returns the stitched output.
pub fn tiled(data: &[f32], frames: usize, p: [usize; 3], info: &ModelInfo, tile: Option<usize>, keep_frames: Option<usize>, f: &mut dyn FnMut(&[f32], usize, [usize; 3]) -> Result<Vec<f32>, Error>) -> Result<Vec<f32>, Error> {
    let np = p[0] * p[1] * p[2];
    let nc = data.len() / (frames * np); // 14 = 7 encodes × (re, im)
    if nc * frames * np != data.len() {
        return Err(Error::Layout("tiled: data size".into()));
    }
    let ax: Vec<Vec<(usize, usize, usize, usize)>> = match tile.map(|t| tile_size(info, t)) {
        Some(t) => (0..3).map(|a| axis_tiles(p[a], t, tile_margin(info), info.pad_multiple)).collect::<Result<_, _>>()?,
        None => (0..3).map(|a| vec![(0, p[a], 0, p[a])]).collect(),
    };
    let mt = info.rf_time;
    // time windows: (first kept frame, kept count); window = kept ± mt frames, cyclic
    let tw: Vec<(usize, usize)> = match keep_frames {
        Some(k) if k.max(1) + 2 * mt < frames => (0..frames).step_by(k.max(1)).map(|a| (a, k.max(1).min(frames - a))).collect(),
        _ => vec![(0, frames)],
    };
    if tw.len() == 1 && ax.iter().all(|t| t.len() == 1) {
        return f(data, frames, p);
    }
    let mut out = vec![0f32; data.len()];
    for &(a, k) in &tw {
        let whole = tw.len() == 1;
        let (nw, src): (usize, Vec<usize>) = if whole { (frames, (0..frames).collect()) } else { (k + 2 * mt, (0..k + 2 * mt).map(|j| (a + frames + j - mt) % frames).collect()) };
        // the time window of the whole spatial volume, [nc, nw, P0, P1, P2]
        let mut win = vec![0f32; nc * nw * np];
        for c in 0..nc {
            for (j, &t) in src.iter().enumerate() {
                win[(c * nw + j) * np..(c * nw + j + 1) * np].copy_from_slice(&data[(c * frames + t) * np..(c * frames + t + 1) * np]);
            }
        }
        let keep_t = if whole { 0..frames } else { mt..mt + k };
        for t0 in &ax[0] {
            for t1 in &ax[1] {
                for t2 in &ax[2] {
                    let (lo, ext) = ([t0.0, t1.0, t2.0], [t0.1, t1.1, t2.1]);
                    let ne = ext[0] * ext[1] * ext[2];
                    let mut sub = vec![0f32; nc * nw * ne];
                    crop(&win, p, lo, ext, &mut sub);
                    let y = f(&sub, nw, ext)?;
                    // kept frames and the kept spatial box of this tile, into their cycle positions
                    let w = t2.3 - t2.2;
                    for c in 0..nc {
                        for (jj, j) in keep_t.clone().enumerate() {
                            let t = if whole { jj } else { a + jj };
                            for q0 in t0.2..t0.3 {
                                for q1 in t1.2..t1.3 {
                                    let d = (c * frames + t) * np + (q0 * p[1] + q1) * p[2] + t2.2;
                                    let s = (c * nw + j) * ne + ((q0 - lo[0]) * ext[1] + q1 - lo[1]) * ext[2] + t2.2 - lo[2];
                                    out[d..d + w].copy_from_slice(&y[s..s + w]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// kv (π/VENC) per hasty slot -> per model slot
pub fn kv_model(kv: [f32; NE]) -> [f32; NE] {
    MODEL_SLOT_SRC.map(|e| kv[e])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(dims: [usize; 3], frames: usize) -> Vec<C32> {
        let n = NE * frames * dims.iter().product::<usize>();
        (0..n).map(|i| C32::new(((i * 37 % 101) as f32 - 50.0) / 7.0, ((i * 53 % 97) as f32 - 48.0) / 5.0)).collect()
    }

    #[test]
    fn tiles_cover_exactly_once_on_the_lattice() {
        for (n, tile, m) in [(69usize, 65usize, 28usize), (181, 65, 28), (29, 65, 28), (201, 77, 28), (97, 61, 12)] {
            let t = axis_tiles(n, tile, m, 4).unwrap();
            let mut cover = vec![0; n];
            for &(s, len, lo, hi) in &t {
                assert!(s % 4 == 0 && s + len <= n && s <= lo && hi <= s + len);
                assert!(lo == 0 || lo >= s + m, "keep starts inside the margin");
                assert!(hi == n || hi + m <= s + len, "keep ends inside the margin");
                for q in lo..hi {
                    cover[q] = 1;
                }
            }
            assert!(cover.iter().all(|&c| c == 1), "n {n}: not covered");
        }
    }

    #[test]
    fn tiled_identity_and_local_maps_are_exact() {
        // a local map (3-point average along each axis, zero outside) has radius 1 per call: tiles with margin ≥ 1
        // must reproduce the whole-volume result exactly
        let info = ModelInfo { pad_multiple: 4, rf_space: 3, rf_time: 0, group: "test".into() };
        let p = [9usize, 21, 33];
        let nc = 3;
        let data: Vec<f32> = (0..nc * p.iter().product::<usize>()).map(|i| ((i * 7919) % 1013) as f32 / 101.0).collect();
        let f = |x: &[f32], e: [usize; 3]| -> Result<Vec<f32>, Error> {
            let n = e[0] * e[1] * e[2];
            let mut y = vec![0f32; x.len()];
            for c in 0..x.len() / n {
                for a in 0..e[0] {
                    for b in 0..e[1] {
                        for d in 0..e[2] {
                            let g = |a: isize, b: isize, d: isize| if a < 0 || b < 0 || d < 0 || a >= e[0] as isize || b >= e[1] as isize || d >= e[2] as isize { 0.0 } else { x[c * n + ((a as usize) * e[1] + b as usize) * e[2] + d as usize] };
                            let (a, b, d) = (a as isize, b as isize, d as isize);
                            y[c * n + ((a as usize) * e[1] + b as usize) * e[2] + d as usize] = g(a, b, d) + g(a - 1, b, d) + g(a + 1, b, d) + g(a, b - 1, d) + g(a, b + 1, d) + g(a, b, d - 1) + g(a, b, d + 1);
                        }
                    }
                }
            }
            Ok(y)
        };
        let whole = f(&data, p).unwrap();
        let mut g = |x: &[f32], _nt: usize, e: [usize; 3]| f(x, e);
        let t = tiled(&data, 1, p, &info, Some(13), None, &mut g).unwrap();
        assert_eq!(whole, t);
    }

    #[test]
    fn cyclic_time_windows_are_exact() {
        // a map with a cyclic 3-frame temporal stencil (radius 1) and a 1-voxel spatial stencil along axis 2:
        // time windows (margin rf_time = 1) and spatial tiles must reproduce the whole-cycle result
        let info = ModelInfo { pad_multiple: 4, rf_space: 1, rf_time: 1, group: "test".into() };
        let (p, frames, nc) = ([5usize, 5, 33], 10usize, 2usize);
        let np = p[0] * p[1] * p[2];
        let data: Vec<f32> = (0..nc * frames * np).map(|i| ((i * 7919) % 1013) as f32 / 101.0).collect();
        let mut f = |x: &[f32], nt: usize, e: [usize; 3]| -> Result<Vec<f32>, Error> {
            let n = e[0] * e[1] * e[2];
            let mut y = vec![0f32; x.len()];
            for c in 0..x.len() / (nt * n) {
                for t in 0..nt {
                    for v in 0..n {
                        let d = v % e[2];
                        let at = |tt: usize, vv: isize| -> f32 { if vv < 0 || vv >= e[2] as isize { 0.0 } else { x[(c * nt + tt) * n + v - d + vv as usize] } };
                        let (tm, tp) = ((t + nt - 1) % nt, (t + 1) % nt);
                        y[(c * nt + t) * n + v] = 2.0 * at(t, d as isize) + at(tm, d as isize) - 0.5 * at(tp, d as isize) + at(t, d as isize - 1) + 3.0 * at(t, d as isize + 1);
                    }
                }
            }
            Ok(y)
        };
        let whole = f(&data, frames, p).unwrap();
        for (tile, keep) in [(None, Some(3)), (None, Some(4)), (Some(13), Some(2)), (Some(13), None)] {
            let t = tiled(&data, frames, p, &info, tile, keep, &mut f).unwrap();
            assert_eq!(whole, t, "tile {tile:?} keep {keep:?}");
        }
    }

    #[test]
    fn padded_sizes() {
        assert_eq!(padded(1, 4), 1);
        assert_eq!(padded(5, 4), 5);
        assert_eq!(padded(6, 4), 9);
        assert_eq!(padded(23, 4), 25);
        assert_eq!(padded(33, 4), 33);
    }

    #[test]
    fn layout_round_trip() {
        let dims = [6, 5, 3];
        for sign in [1i8, -1] {
            let s = stack(dims, 2);
            let (t, p) = to_model(&s, dims, 2, 4, sign).unwrap();
            assert_eq!(p, [5, 5, 9]);
            let mut back = vec![C32::new(0.0, 0.0); s.len()];
            from_model(&t, p, &mut back, dims, 2, sign).unwrap();
            let err = s.iter().zip(&back).map(|(a, b)| (a - b).norm()).fold(0f32, f32::max);
            assert!(err < 1e-4, "sign {sign}: round trip error {err}");
        }
    }

    #[test]
    fn model_slots_follow_reversed_axes() {
        // model tensor axis 0 is z: its encode pair must be hasty's z pair (slots 5, 6), x pair last
        assert_eq!(MODEL_SLOT_SRC, [0, 5, 6, 3, 4, 1, 2]);
        let kv = kv_model([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(kv, [0.0, 5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn velocity_sign_flip_is_exact() {
        // x_ref = m e^{iθ}, x_e = m e^{iθ} e^{−iφ}: after the flip x_e x̄_ref has phase +φ and the same magnitude
        let (m, th, ph) = (2.0f32, 0.7f32, 0.4f32);
        let r = C32::from_polar(m, th);
        let mut s = vec![r; NE];
        for (e, z) in s.iter_mut().enumerate().skip(1) {
            *z = C32::from_polar(m, th - ph * e as f32);
        }
        flip_velocity_sign(&mut s, 1, 1);
        for (e, z) in s.iter().enumerate().skip(1) {
            let d = (z * r.conj()).arg();
            assert!((d - ph * e as f32).abs() < 1e-5, "encode {e}: phase {d}");
            assert!((z.norm() - m).abs() < 1e-5);
        }
    }
}
