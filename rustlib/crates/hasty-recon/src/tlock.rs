//! Static / dynamic tissue map and temporal lock (research/static_lock.md).
//!
//! Most voxels of an intracranial scan, and part of a thoracic one (spine, back muscles, much of the chest wall
//! between heartbeats), neither move nor carry flow: their images should be identical in every frame. The map finds
//! them from the reconstruction itself (no labels, no masks) and the lock pulls their frames to the common mean.
//!
//! Map, from the current iterate x[r, c, e] (respiratory bin r, cardiac frame c, encode e), every image low-passed by
//! a Gaussian (σ = `blur_mm`: incoherent aliasing is mostly high spatial frequency, organ motion is not):
//! - cardiac variation V_c(q) = mean_{r,e} of the unbiased variance over c,
//! - respiratory variation V_r(q) = mean_e of the unbiased variance over r of the cardiac means,
//! - noise / aliasing level σ² = median_q of mean |x_c − 2x_{c−1} + x_{c−2}|² / 6 (second differences in cardiac phase:
//!   frame-to-frame independent errors pass, smooth motion does not; the median over voxels removes the moving ones),
//! - scores s_c = √(V_c / σ²), s_r = √(V_r / (σ² / F)) (≈ 1 for a static voxel), s_v = |v̄| / σ_|v| for the velocity
//!   of the temporal mean images above `v_min` (steady flow does not vary in time but must not be locked),
//! - static weights w_c = ramp(max(s_c, s_v)) · conf_c, w_r = ramp(max(s_r, s_v)) · conf_r, ramp = 1 below `s_lo`, 0
//!   above `s_hi`. Undetectable is not static: conf (detection power) is 1 where a relative variation of `detect_ok`
//!   would have scored s_hi (s_hi σ ≤ detect_ok |m̄|, per frame for the cardiac lock, per bin mean for the respiratory
//!   one), 0 where not even `detect_max` would (heavy aliasing early on, or a very short scan), and 1 where the mean
//!   signal itself is not significant (air: nothing to lose). The dynamic region is then grown by `dilate_mm` (min
//!   filter of w).
//!
//! Lock (a proximal map, firmly non-expansive): the decomposition x = x̄ + (m_r − x̄) + (x_{r,c} − m_r) into the global
//! mean, the respiratory-bin deviation and the cardiac deviation inside the bin is orthogonal; each deviation is
//! scaled per voxel,
//! - quadratic: prox of (μ w / 2)‖deviation‖², factor 1 / (1 + μ w),
//! - group: prox of μ w σ_x √(dof) ‖deviation‖ (group soft threshold over the frames, σ_x the noise level of the
//!   unfiltered images): a static voxel is set exactly to its mean, a voxel that does move keeps its motion minus the
//!   threshold even where the map is wrong.
//! The cardiac lock acts inside each respiratory bin with w_c, the respiratory lock on the bin means with w_r: tissue
//! that moves with breathing but not with the heart is locked over the cardiac cycle only.

use hasty_cuda::C32;

type Cf = num_complex::Complex<f32>;

/// the stack's interleaved complex values as num_complex (both repr(C) { re, im })
pub fn cf(v: &[C32]) -> &[Cf] {
    // SAFETY: identical layout
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const Cf, v.len()) }
}

pub fn cf_mut(v: &mut [C32]) -> &mut [Cf] {
    // SAFETY: identical layout
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut Cf, v.len()) }
}
use rayon::prelude::*;

use crate::flow::FlowDecoder;

#[derive(Clone, Debug)]
pub struct LockOpts {
    /// quadratic: prox weight μ; group: threshold factor
    pub mu: f64,
    pub group: bool,
    pub blur_mm: f64,
    pub s_lo: f64,
    pub s_hi: f64,
    pub dilate_mm: f64,
    /// velocity floor of the flow criterion (m/s)
    pub v_min: f64,
    /// block size of the local noise level (mm)
    pub noise_block_mm: f64,
    /// detection power: a voxel with signal is locked only if a relative temporal variation (std / |mean|) of
    /// `detect_ok` would have been detected (s_hi σ ≤ detect_ok |m̄|), not at all if not even `detect_max` would;
    /// voxels without significant mean signal (air) stay lockable
    pub detect_ok: f64,
    pub detect_max: f64,
}

impl Default for LockOpts {
    fn default() -> Self {
        Self { mu: 20.0, group: false, blur_mm: 3.0, s_lo: 1.5, s_hi: 3.0, dilate_mm: 3.0, v_min: 0.03, noise_block_mm: 24.0, detect_ok: 0.1, detect_max: 0.3 }
    }
}

/// Stack layout: image index (× nv) of (respiratory bin r, cardiac frame c, encode e).
#[derive(Clone, Debug)]
pub struct FrameLayout {
    pub dims: [usize; 3],
    /// voxel size per axis (m)
    pub vox: [f64; 3],
    pub nc: usize,
    pub nr: usize,
    pub ne: usize,
    idx: Vec<usize>,
}

impl FrameLayout {
    /// frame f = r · nc + c; `encode_major`: image (e · nf + f), else (f · ne + e)
    pub fn new(dims: [usize; 3], vox: [f64; 3], nc: usize, nr: usize, ne: usize, encode_major: bool) -> Self {
        let nf = nc * nr;
        let mut idx = vec![0; nf * ne];
        for r in 0..nr {
            for c in 0..nc {
                for e in 0..ne {
                    let f = r * nc + c;
                    idx[(r * nc + c) * ne + e] = if encode_major { e * nf + f } else { f * ne + e };
                }
            }
        }
        Self { dims, vox, nc, nr, ne, idx }
    }

    pub fn nv(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn img(&self, r: usize, c: usize, e: usize) -> usize {
        self.idx[(r * self.nc + c) * self.ne + e]
    }
}

#[derive(Clone, Debug, Default)]
pub struct StaticMap {
    pub s_card: Vec<f32>,
    pub s_resp: Vec<f32>,
    pub s_flow: Vec<f32>,
    /// static weights in [0, 1] (1 = locked)
    pub w_card: Vec<f32>,
    pub w_resp: Vec<f32>,
    /// median noise variance per complex sample: low-passed images (the map) and unfiltered images
    pub sigma2_blur: f64,
    pub sigma2_raw: f64,
    /// local noise standard deviation of the unfiltered images per voxel (group thresholds)
    pub sigma_raw: Vec<f32>,
    /// fraction of voxels with w > 0.5
    pub static_card: f64,
    pub static_resp: f64,
    /// fraction of voxels with signal whose motion would not have been detected (confidence < 0.5, cardiac)
    pub undetermined: f64,
    /// iteration the map was estimated at
    pub iteration: usize,
}

impl StaticMap {
    /// the six maps (s_card, s_resp, s_flow, w_card, w_resp, σ_raw), image-major
    pub fn stacked(&self) -> Vec<f32> {
        [&self.s_card, &self.s_resp, &self.s_flow, &self.w_card, &self.w_resp, &self.sigma_raw].iter().flat_map(|v| v.iter().copied()).collect()
    }

    pub fn from_stacked(v: &[f32], nv: usize, iteration: usize) -> Self {
        let g = |i: usize| v[i * nv..(i + 1) * nv].to_vec();
        let frac = |w: &[f32]| w.iter().filter(|&&x| x > 0.5).count() as f64 / nv.max(1) as f64;
        let (w_card, w_resp) = (g(3), g(4));
        let sigma_raw = g(5);
        let sigma2_raw = median(sigma_raw.iter().map(|&x| (x as f64).powi(2)).collect());
        Self { s_card: g(0), s_resp: g(1), s_flow: g(2), static_card: frac(&w_card), static_resp: frac(&w_resp), undetermined: f64::NAN, w_card, w_resp, sigma2_blur: 0.0, sigma2_raw, sigma_raw, iteration }
    }
}

/// separable Gaussian low-pass of a complex image (σ per axis in voxels, replicated edges), out of place
pub fn blur(img: &[Cf], d: [usize; 3], sig: [f64; 3], tmp: &mut Vec<Cf>, out: &mut Vec<Cf>) {
    let nv = d[0] * d[1] * d[2];
    let kern = |s: f64| -> Vec<f32> { gauss(s).iter().map(|&v| v as f32).collect() };
    let (kx, ky, kz) = (kern(sig[0]), kern(sig[1]), kern(sig[2]));
    tmp.resize(nv, Cf::new(0.0, 0.0));
    out.resize(nv, Cf::new(0.0, 0.0));
    let cl = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    // x: per row
    let rx = (kx.len() / 2) as isize;
    out.par_chunks_mut(d[0]).zip(img.par_chunks(d[0])).for_each(|(o, row)| {
        for x in 0..d[0] {
            let mut a = Cf::new(0.0, 0.0);
            for (t, &g) in kx.iter().enumerate() {
                a += row[cl(x as isize + t as isize - rx, d[0])] * g;
            }
            o[x] = a;
        }
    });
    // y: per plane, rows of x as vectors
    let ry = (ky.len() / 2) as isize;
    let plane = d[0] * d[1];
    tmp.par_chunks_mut(plane).zip(out.par_chunks(plane)).for_each(|(o, p)| {
        for y in 0..d[1] {
            let orow = &mut o[y * d[0]..(y + 1) * d[0]];
            orow.iter_mut().for_each(|v| *v = Cf::new(0.0, 0.0));
            for (t, &g) in ky.iter().enumerate() {
                let ys = cl(y as isize + t as isize - ry, d[1]);
                for (a, b) in orow.iter_mut().zip(&p[ys * d[0]..(ys + 1) * d[0]]) {
                    *a += *b * g;
                }
            }
        }
    });
    // z: per output plane
    let rz = (kz.len() / 2) as isize;
    let src = &*tmp;
    out.par_chunks_mut(plane).enumerate().for_each(|(z, o)| {
        o.iter_mut().for_each(|v| *v = Cf::new(0.0, 0.0));
        for (t, &g) in kz.iter().enumerate() {
            let zs = cl(z as isize + t as isize - rz, d[2]);
            for (a, b) in o.iter_mut().zip(&src[zs * plane..(zs + 1) * plane]) {
                *a += *b * g;
            }
        }
    });
}

/// 1D Gaussian taps for σ voxels (a single tap below 0.3)
fn gauss(s: f64) -> Vec<f64> {
    if s < 0.3 {
        return vec![1.0];
    }
    let r = (3.0 * s).ceil() as isize;
    let k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / s).powi(2)).exp()).collect();
    let t: f64 = k.iter().sum();
    k.iter().map(|v| v / t).collect()
}

/// white-noise variance gain of the replicated-edge blur along one axis, per output index (the edges repeat samples)
fn blur_gain_axis(n: usize, k: &[f64]) -> Vec<f64> {
    let r = (k.len() / 2) as isize;
    (0..n)
        .map(|i| {
            let mut a = vec![0f64; n];
            for (t, &g) in k.iter().enumerate() {
                a[(i as isize + t as isize - r).clamp(0, n as isize - 1) as usize] += g;
            }
            a.iter().map(|v| v * v).sum()
        })
        .collect()
}

/// local robust level of a per-voxel noise estimate: medians over blocks of b voxels, trilinear between block centres
fn block_median(v: &[f64], d: [usize; 3], b: [usize; 3]) -> Vec<f64> {
    let b = b.map(|x| x.max(2));
    let nb = [0, 1, 2].map(|a| d[a].div_ceil(b[a]));
    let med: Vec<f64> = (0..nb[0] * nb[1] * nb[2])
        .into_par_iter()
        .map(|k| {
            let c = [k % nb[0], (k / nb[0]) % nb[1], k / (nb[0] * nb[1])];
            let lo = [0, 1, 2].map(|a| c[a] * b[a]);
            let hi = [0, 1, 2].map(|a| ((c[a] + 1) * b[a]).min(d[a]));
            let mut s = Vec::with_capacity(b[0] * b[1] * b[2]);
            for z in lo[2]..hi[2] {
                for y in lo[1]..hi[1] {
                    for x in lo[0]..hi[0] {
                        s.push(v[x + d[0] * (y + d[1] * z)]);
                    }
                }
            }
            median(s)
        })
        .collect();
    (0..d[0] * d[1] * d[2])
        .into_par_iter()
        .map(|q| {
            let p = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
            let (mut i0, mut f) = ([0usize; 3], [0f64; 3]);
            for a in 0..3 {
                let u = ((p[a] as f64 + 0.5) / b[a] as f64 - 0.5).clamp(0.0, (nb[a] - 1) as f64);
                i0[a] = (u.floor() as usize).min(nb[a].saturating_sub(2));
                f[a] = if nb[a] > 1 { u - i0[a] as f64 } else { 0.0 };
            }
            let mut acc = 0.0;
            for corner in 0..8 {
                let mut w = 1.0;
                let mut idx = [0usize; 3];
                for a in 0..3 {
                    let up = (corner >> a) & 1;
                    idx[a] = (i0[a] + up).min(nb[a] - 1);
                    w *= if up == 1 { f[a] } else { 1.0 - f[a] };
                }
                acc += w * med[idx[0] + nb[0] * (idx[1] + nb[1] * idx[2])];
            }
            acc
        })
        .collect()
}

/// separable min filter (cube of half-width r voxels per axis), in place
fn erode(w: &mut [f32], d: [usize; 3], r: [usize; 3]) {
    let mut t = w.to_vec();
    for a in 0..3 {
        if r[a] == 0 {
            continue;
        }
        let stride = [1, d[0], d[0] * d[1]][a];
        let n = d[a];
        let src = t.clone();
        t.par_iter_mut().enumerate().for_each(|(q, v)| {
            let i = (q / stride) % n;
            let (lo, hi) = (i.saturating_sub(r[a]), (i + r[a]).min(n - 1));
            let base = q - i * stride;
            *v = (lo..=hi).map(|j| src[base + j * stride]).fold(f32::INFINITY, f32::min);
        });
    }
    w.copy_from_slice(&t);
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = v.len() / 2;
    *v.select_nth_unstable_by(m, |a, b| a.partial_cmp(b).unwrap()).1
}

/// estimate the static / dynamic map from the stack `h` (layout `lay`); `dec`: the flow decoder (flow criterion)
pub fn estimate(h: &[C32], lay: &FrameLayout, opts: &LockOpts, dec: Option<&FlowDecoder>, iteration: usize) -> StaticMap {
    let h = cf(h);
    let (d, nv, nc, nr, ne) = (lay.dims, lay.nv(), lay.nc, lay.nr, lay.ne);
    let sig = [0, 1, 2].map(|a| opts.blur_mm * 1e-3 / lay.vox[a]);
    let mut vc = vec![0f64; nv];
    let mut vr = vec![0f64; nv];
    let mut d2b = vec![0f64; nv];
    let mut d2r = vec![0f64; nv];
    let mut means: Vec<Vec<Cf>> = vec![];
    let mut mbar = vec![0f64; nv];
    let (mut tmp, mut b) = (Vec::new(), Vec::new());
    let mut prev: [Vec<Cf>; 2] = [vec![Cf::new(0.0, 0.0); nv], vec![Cf::new(0.0, 0.0); nv]];
    for e in 0..ne {
        let mut t1 = vec![Cf::new(0.0, 0.0); nv];
        let mut t2 = vec![0f64; nv];
        for r in 0..nr {
            let mut s1 = vec![Cf::new(0.0, 0.0); nv];
            let mut s2 = vec![0f64; nv];
            for c in 0..nc {
                let raw = &h[lay.img(r, c, e) * nv..][..nv];
                blur(raw, d, sig, &mut tmp, &mut b);
                let second = c >= 2;
                let (p1, p2) = (&prev[0], &prev[1]);
                let (r1, r2) = if second { (&h[lay.img(r, c - 1, e) * nv..][..nv], &h[lay.img(r, c - 2, e) * nv..][..nv]) } else { (raw, raw) };
                (s1.par_iter_mut(), s2.par_iter_mut(), d2b.par_iter_mut(), d2r.par_iter_mut()).into_par_iter().enumerate().for_each(|(q, (a1, a2, db, dr))| {
                    let x = b[q];
                    *a1 += x;
                    *a2 += x.norm_sqr() as f64;
                    if second {
                        *db += (x - p1[q] * 2.0 + p2[q]).norm_sqr() as f64;
                        *dr += (raw[q] - r1[q] * 2.0 + r2[q]).norm_sqr() as f64;
                    }
                });
                // prev[1] ← prev[0] ← b
                prev.swap(0, 1);
                prev[0].copy_from_slice(&b);
            }
            let fc = nc as f64;
            (vc.par_iter_mut(), t1.par_iter_mut(), t2.par_iter_mut(), s1.par_iter(), s2.par_iter()).into_par_iter().for_each(|(v, a1, a2, &m1, &m2)| {
                if nc >= 2 {
                    *v += (m2 - m1.norm_sqr() as f64 / fc) / (fc - 1.0);
                }
                let m = m1 / nc as f32;
                *a1 += m;
                *a2 += m.norm_sqr() as f64;
            });
        }
        let fr = nr as f64;
        if nr >= 2 {
            vr.par_iter_mut().zip(t1.par_iter().zip(t2.par_iter())).for_each(|(v, (&m1, &m2))| *v += (m2 - m1.norm_sqr() as f64 / fr) / (fr - 1.0));
        }
        mbar.par_iter_mut().zip(t1.par_iter()).for_each(|(m, &t)| *m += (t / nr as f32).norm() as f64 / ne as f64);
        if dec.is_some() {
            t1.par_iter_mut().for_each(|m| *m /= nr as f32);
            means.push(t1);
        }
    }
    let ngrp = (ne * nr) as f64;
    vc.par_iter_mut().for_each(|v| *v /= ngrp);
    vr.par_iter_mut().for_each(|v| *v /= ne as f64);
    let n2 = (ne * nr * nc.saturating_sub(2)) as f64;
    // white-noise gain of the blur per voxel relative to the interior (edges repeat samples)
    let gains: Vec<Vec<f64>> = (0..3).map(|a| blur_gain_axis(d[a], &gauss(sig[a]))).collect();
    let gint: f64 = (0..3).map(|a| gauss(sig[a]).iter().map(|v| v * v).sum::<f64>()).product();
    let gq = |q: usize| -> f64 { gains[0][q % d[0]] * gains[1][(q / d[0]) % d[1]] * gains[2][q / (d[0] * d[1])] / gint };
    let blk = [0, 1, 2].map(|a| (opts.noise_block_mm * 1e-3 / lay.vox[a]).round() as usize);
    // local noise variance: block medians of the second-difference estimator (moving voxels are a minority per block),
    // or, with fewer than 3 cardiac frames, of the cardiac variance itself
    let (nb, nr_) = if n2 > 0.0 {
        let db: Vec<f64> = (0..nv).into_par_iter().map(|q| d2b[q] / (6.0 * n2) / gq(q)).collect();
        let dr: Vec<f64> = d2r.par_iter().map(|v| v / (6.0 * n2)).collect();
        (block_median(&db, d, blk), block_median(&dr, d, blk))
    } else {
        let db: Vec<f64> = (0..nv).into_par_iter().map(|q| vc[q] / gq(q)).collect();
        (block_median(&db, d, blk), vec![0.0; nv])
    };
    let step = nv.div_ceil(1 << 20).max(1);
    let sigma2_blur = median(nb.iter().step_by(step).cloned().collect());
    let sigma2_raw = median(nr_.iter().step_by(step).cloned().collect());
    // floors against empty blocks (air with no aliasing): a tenth of the median
    let s2q: Vec<f64> = (0..nv).into_par_iter().map(|q| nb[q].max(0.1 * sigma2_blur).max(1e-30) * gq(q)).collect();
    let sigma_raw: Vec<f32> = nr_.par_iter().map(|&v| v.max(0.1 * sigma2_raw).sqrt() as f32).collect();
    let s_card: Vec<f32> = (0..nv).into_par_iter().map(|q| (vc[q].max(0.0) / s2q[q]).sqrt() as f32).collect();
    let s_resp: Vec<f32> = (0..nv).into_par_iter().map(|q| if nr >= 2 { (vr[q].max(0.0) / (s2q[q] / nc as f64)).sqrt() as f32 } else { 0.0 }).collect();
    // flow: velocity of the temporal mean images against its noise
    let s_flow: Vec<f32> = match dec {
        Some(dec) => {
            let kv = dec.kv();
            let keff = (kv.iter().map(|k| k * k).sum::<f64>() / 3.0).sqrt().max(1e-12);
            (0..nv)
                .into_par_iter()
                .map(|q| {
                    let mut zb = [num_complex::Complex64::new(0.0, 0.0); crate::flow::MAX_ENC + 1];
                    for e in 0..ne {
                        let m = means[e][q];
                        zb[e] = num_complex::Complex64::new(m.re as f64, m.im as f64);
                    }
                    let z = &zb[..ne];
                    let mag = z[0].norm();
                    if mag <= 0.0 {
                        return 0.0;
                    }
                    let v = dec.velocity(z);
                    let vn = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if vn < opts.v_min {
                        return 0.0;
                    }
                    // σ_φ per encode, √2 for a difference, √3 for |v| of three components
                    let sphi = (s2q[q] / (nc * nr) as f64 / 2.0).sqrt() / mag;
                    let sv = 2f64.sqrt() * sphi / keff * 3f64.sqrt();
                    (vn / sv) as f32
                })
                .collect()
        }
        None => vec![0.0; nv],
    };
    let ramp = |s: f32| ((opts.s_hi as f32 - s) / (opts.s_hi - opts.s_lo) as f32).clamp(0.0, 1.0);
    // detection power: the smallest relative variation (std / |m̄|) that would score s_hi, per frame (cardiac) and per
    // bin mean (respiratory); no significant mean signal (|m̄| < 3 σ / √(frames)): nothing to lose, lockable
    let conf = |q: usize, per: f64| -> f32 {
        let sig = s2q[q].sqrt();
        if mbar[q] < 3.0 * sig / ((nc * nr) as f64).sqrt() {
            return 1.0;
        }
        let rho = opts.s_hi * sig / per.sqrt() / mbar[q];
        ((opts.detect_max - rho) / (opts.detect_max - opts.detect_ok)).clamp(0.0, 1.0) as f32
    };
    let conf_c: Vec<f32> = (0..nv).into_par_iter().map(|q| conf(q, 1.0)).collect();
    let undetermined = conf_c.iter().zip(&mbar).zip(&s2q).filter(|((&c, &m), &v)| c < 0.5 && m >= 3.0 * v.sqrt() / ((nc * nr) as f64).sqrt()).count() as f64 / nv.max(1) as f64;
    let mut w_card: Vec<f32> = (0..nv).into_par_iter().map(|q| ramp(s_card[q].max(s_flow[q])) * conf_c[q]).collect();
    let mut w_resp: Vec<f32> = (0..nv).into_par_iter().map(|q| if nr >= 2 { ramp(s_resp[q].max(s_flow[q])) * conf(q, nc as f64) } else { 0.0 }).collect();
    let rr = [0, 1, 2].map(|a| (opts.dilate_mm * 1e-3 / lay.vox[a]).round() as usize);
    erode(&mut w_card, d, rr);
    if nr >= 2 {
        erode(&mut w_resp, d, rr);
    }
    let frac = |w: &[f32]| w.iter().filter(|&&x| x > 0.5).count() as f64 / nv.max(1) as f64;
    StaticMap { static_card: frac(&w_card), static_resp: frac(&w_resp), undetermined, s_card, s_resp, s_flow, w_card, w_resp, sigma2_blur, sigma2_raw, sigma_raw, iteration }
}

/// the lock (in place)
pub fn apply(h: &mut [C32], lay: &FrameLayout, map: &StaticMap, opts: &LockOpts) {
    let h = cf_mut(h);
    let (nv, nc, nr, ne) = (lay.nv(), lay.nc, lay.nr, lay.ne);
    let mu = opts.mu as f32;
    let base = h.as_mut_ptr() as usize;
    // view of image i; every task below touches its own voxel range of each image
    unsafe fn view<'a>(base: usize, i: usize, nv: usize) -> &'a mut [Cf] {
        unsafe { std::slice::from_raw_parts_mut((base as *mut Cf).add(i * nv), nv) }
    }
    // SAFETY: disjoint voxel ranges per task, images inside h
    let img = |i: usize| unsafe { view(base, i, nv) };
    const CH: usize = 4096;
    for e in 0..ne {
        // cardiac lock inside each bin; the bin means are kept for the respiratory lock
        let mut bin_means: Vec<Vec<Cf>> = vec![vec![Cf::new(0.0, 0.0); nv]; nr];
        for r in 0..nr {
            let ids: Vec<usize> = (0..nc).map(|c| lay.img(r, c, e)).collect();
            bin_means[r].par_chunks_mut(CH).enumerate().for_each(|(k, m)| {
                let o = k * CH;
                for (j, mq) in m.iter_mut().enumerate() {
                    let q = o + j;
                    let mut s = Cf::new(0.0, 0.0);
                    for &i in &ids {
                        s += img(i)[q];
                    }
                    let mean = s / nc as f32;
                    *mq = mean;
                    let w = map.w_card[q];
                    if w <= 0.0 || nc < 2 {
                        continue;
                    }
                    let g = if opts.group {
                        let n2: f32 = ids.iter().map(|&i| (img(i)[q] - mean).norm_sqr()).sum();
                        let tau = mu * w * map.sigma_raw[q] * ((nc - 1) as f32).sqrt();
                        if n2 > 0.0 { (1.0 - tau / n2.sqrt()).max(0.0) } else { 0.0 }
                    } else {
                        1.0 / (1.0 + mu * w)
                    };
                    for &i in &ids {
                        let x = &mut img(i)[q];
                        *x = mean + (*x - mean) * g;
                    }
                }
            });
        }
        if nr < 2 {
            continue;
        }
        // respiratory lock on the bin means: x_{r,c} += (m̄ − m_r)(1 − g)
        let ids: Vec<Vec<usize>> = (0..nr).map(|r| (0..nc).map(|c| lay.img(r, c, e)).collect()).collect();
        let bm = &bin_means;
        (0..nv.div_ceil(CH)).into_par_iter().for_each(|k| {
            for q in k * CH..((k + 1) * CH).min(nv) {
                let w = map.w_resp[q];
                if w <= 0.0 {
                    continue;
                }
                let mbar = (0..nr).map(|r| bm[r][q]).fold(Cf::new(0.0, 0.0), |a, b| a + b) / nr as f32;
                let g = if opts.group {
                    let n2: f32 = (0..nr).map(|r| (bm[r][q] - mbar).norm_sqr()).sum();
                    // noise of a bin mean: σ² / F per sample
                    let tau = mu * w * map.sigma_raw[q] * (((nr - 1) as f32) / nc as f32).sqrt();
                    if n2 > 0.0 { (1.0 - tau / n2.sqrt()).max(0.0) } else { 0.0 }
                } else {
                    1.0 / (1.0 + mu * w)
                };
                for r in 0..nr {
                    let sh = (mbar - bm[r][q]) * (1.0 - g);
                    for &i in &ids[r] {
                        img(i)[q] += sh;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_finds_moving_and_flowing_voxels_and_lock_freezes_static_ones() {
        // 24³, 2 bins × 10 cardiac frames × 4 encodes (referenced): a static background, a block moving with the
        // heart, a block moving with breathing only, a vessel with steady flow; independent noise per frame
        let d = [24usize, 24, 24];
        let nv = d.iter().product::<usize>();
        let (nc, nr, ne) = (10usize, 2usize, 4usize);
        let venc = 1.0;
        let m1: Vec<[f64; 3]> = {
            let k = std::f64::consts::PI / venc / hasty_mrsim::GAMMA;
            vec![[0.0; 3], [k, 0.0, 0.0], [0.0, k, 0.0], [0.0, 0.0, k]]
        };
        let dec = FlowDecoder::new(&m1);
        let lay = FrameLayout::new(d, [2e-3; 3], nc, nr, ne, false);
        let inbox = |q: usize, lo: [usize; 3], hi: [usize; 3]| {
            let p = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
            (0..3).all(|a| p[a] >= lo[a] && p[a] < hi[a])
        };
        let mut s = 0x1234_5678u64;
        let mut g = || {
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32
        };
        let sig = 0.02f32;
        let mut h = vec![C32::new(0.0, 0.0); nv * nc * nr * ne];
        for r in 0..nr {
            for c in 0..nc {
                let ph = std::f64::consts::TAU * c as f64 / nc as f64;
                for q in 0..nv {
                    let p = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
                    let mut m = 1.0f64;
                    let mut v = [0.0f64; 3];
                    // cardiac block: its edge moves (magnitude oscillates in a slab)
                    if inbox(q, [4, 4, 4], [10, 10, 10]) {
                        m = 1.0 + 0.6 * ph.sin() * ((p[0] as f64 - 3.5) / 6.0);
                    }
                    // respiratory block: different magnitude per bin
                    if inbox(q, [14, 4, 4], [20, 10, 10]) {
                        m = 1.0 + 0.5 * r as f64;
                    }
                    // vessel with steady flow along z
                    if inbox(q, [10, 16, 0], [13, 19, 24]) {
                        v = [0.0, 0.0, 0.4];
                    }
                    for e in 0..ne {
                        let z = num_complex::Complex64::from_polar(m, 0.3 + dec.phase(e, &v));
                        h[lay.img(r, c, e) * nv + q] = C32::new(z.re as f32 + sig * g(), z.im as f32 + sig * g());
                    }
                }
            }
        }
        let opts = LockOpts { blur_mm: 2.0, dilate_mm: 2.0, ..Default::default() };
        let map = estimate(&h, &lay, &opts, Some(&dec), 0);
        let at = |p: [usize; 3]| p[0] + d[0] * (p[1] + d[1] * p[2]);
        // background: outside every structure by ≥ 3 voxels
        let bg: Vec<usize> = (0..nv).filter(|&q| !inbox(q, [1, 1, 1], [23, 13, 13]) && !inbox(q, [7, 13, 0], [16, 22, 24])).collect();
        let mean = |w: &[f32], ids: &[usize]| ids.iter().map(|&q| w[q] as f64).sum::<f64>() / ids.len() as f64;
        let (mc, mr) = (mean(&map.w_card, &bg), mean(&map.w_resp, &bg));
        println!("background: w_card {mc:.3}, w_resp {mr:.3}; σ² blur {:.2e} raw {:.2e}; static fractions {:.3} {:.3}", map.sigma2_blur, map.sigma2_raw, map.static_card, map.static_resp);
        for &q in bg.iter().filter(|&&q| map.w_card[q] < 0.5).take(12) {
            eprintln!("  unlocked bg {:?}: s_card {:.2} s_flow {:.2} s_resp {:.2}", [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])], map.s_card[q], map.s_flow[q], map.s_resp[q]);
        }
        assert!(mc > 0.97 && mr > 0.8, "static background not locked: w_card {mc}, w_resp {mr}");
        let (card, resp, flow) = (at([7, 7, 7]), at([17, 7, 7]), at([11, 17, 12]));
        assert!(map.w_card[card] == 0.0, "cardiac block locked: s_card {}", map.s_card[card]);
        assert!(map.w_resp[resp] == 0.0, "respiratory block locked over breathing: s_resp {}", map.s_resp[resp]);
        assert!(map.w_card[resp] > 0.99, "respiratory block should be locked over the cardiac cycle: {}", map.w_card[resp]);
        assert!(map.w_card[flow] == 0.0 && map.w_resp[flow] == 0.0, "vessel with steady flow locked: s_flow {}", map.s_flow[flow]);
        // lock: a fully static voxel's frames become equal; the cardiac block keeps its cardiac deviations
        let stat = *bg.iter().find(|&&q| map.w_card[q] == 1.0 && map.w_resp[q] == 1.0).expect("no fully static voxel");
        let dev = |x: &[C32], q: usize| -> Vec<Cf> {
            let x = cf(x);
            let m = (0..nc).map(|c| x[lay.img(0, c, 0) * nv + q]).fold(Cf::new(0.0, 0.0), |a, b| a + b) / nc as f32;
            (0..nc).map(|c| x[lay.img(0, c, 0) * nv + q] - m).collect()
        };
        let before = dev(&h, card);
        let mut x = h.clone();
        apply(&mut x, &lay, &map, &LockOpts { mu: 1e6, ..opts.clone() });
        let spread = |q: usize, x: &[C32]| { let x = cf(x); (0..nc).map(|c| (x[lay.img(1, c, 2) * nv + q] - x[lay.img(0, 0, 2) * nv + q]).norm()).fold(0f32, f32::max) };
        assert!(spread(stat, &x) < 1e-4, "static voxel not frozen: {}", spread(stat, &x));
        let after = dev(&x, card);
        assert!(before.iter().zip(&after).all(|(a, b)| (a - b).norm() < 1e-5), "cardiac motion changed");
        // group mode: exact for static voxels with a moderate threshold
        let mut y = h.clone();
        apply(&mut y, &lay, &map, &LockOpts { mu: 3.0, group: true, ..opts.clone() });
        assert!(spread(stat, &y) < 1e-6, "group lock left static variation {}", spread(stat, &y));
    }

    #[test]
    fn undetectable_motion_is_not_locked_air_is() {
        // tissue (|m| = 1) in x < 12, air in x ≥ 12, per-frame noise so strong that a 30 % variation would go unseen
        let d = [24usize, 16, 16];
        let nv = d.iter().product::<usize>();
        let (nc, nr, ne) = (12usize, 1usize, 1usize);
        let lay = FrameLayout::new(d, [2e-3; 3], nc, nr, ne, false);
        let mut s = 0x9999u64;
        let mut g = || {
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32
        };
        let mut h = vec![C32::new(0.0, 0.0); nv * nc];
        for c in 0..nc {
            for q in 0..nv {
                let m = if q % d[0] < 12 { 1.0 } else { 0.0 };
                h[lay.img(0, c, 0) * nv + q] = C32::new(m + 0.7 * g(), 0.7 * g());
            }
        }
        let map = estimate(&h, &lay, &LockOpts { blur_mm: 2.0, dilate_mm: 0.0, ..Default::default() }, None, 0);
        let mean = |f: &dyn Fn(usize) -> bool| { let ids: Vec<usize> = (0..nv).filter(|&q| f(q)).collect(); ids.iter().map(|&q| map.w_card[q] as f64).sum::<f64>() / ids.len() as f64 };
        let (tissue, air) = (mean(&|q| q % d[0] < 9), mean(&|q| q % d[0] >= 15));
        println!("tissue w_card {tissue:.3}, air {air:.3}, undetermined {:.3}", map.undetermined);
        assert!(tissue < 0.05, "tissue with undetectable motion locked: {tissue}");
        assert!(air > 0.9, "air not locked: {air}");
    }
}
