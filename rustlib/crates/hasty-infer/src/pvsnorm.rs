//! Intensity normalisation of the Pham et al. T1 PVS nnU-Net (`T1wPVSNormalization`), reimplemented from the
//! published algorithm and checked against `tools/models/pvs_t1_ref_norm.py` (skimage + dipy):
//!
//! 1. clip to the [min, max] of the non-zero voxels; percentile (1, 99) of the *whole* array → `rescale_intensity`
//! 2. noise σ = dipy `estimate_sigma` (N = 0, no background masking) / 2
//! 3. dipy `nlmeans` (classic algorithm, patch radius 1, block radius 2, Rician), mask = clipped image > 0
//! 4. percentile (0, 98) → `rescale_intensity`
//! 5. skimage `equalize_adapthist` (CLAHE, 3-D, kernel = shape / 8, clip 0.01, 256 bins, 2¹⁴ grey levels)
//!
//! Arrays are C-order with shape `[n0, n1, n2]` (= `[z, y, x]` for hasty volumes). dipy ≥ 1.12 also offers a
//! "blockwise" NLM that is the default there; this module follows the classic one (the default before 1.12).

use rayon::prelude::*;

/// numpy-style linear percentile of arbitrary data
pub fn percentile(data: &[f32], q: f64) -> f64 {
    let mut v: Vec<f32> = data.to_vec();
    v.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q / 100.0 * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let g = pos - lo as f64;
    v[lo] as f64 * (1.0 - g) + v[hi] as f64 * g
}

/// skimage `rescale_intensity(image, in_range=(lo, hi))` for float32 images
fn rescale_float(x: &mut [f32], lo: f64, hi: f64) {
    let (omin, omax) = if lo >= 0.0 { (0.0f32, 1.0f32) } else { (-1.0f32, 1.0f32) };
    let (lo, hi) = (lo as f32, hi as f32);
    x.par_iter_mut().for_each(|v| {
        let c = v.clamp(lo, hi);
        *v = if lo != hi { (c - lo) / (hi - lo) * (omax - omin) + omin } else { c * (omax - omin) + omin };
    });
}

/// skimage `rescale_intensity(image)`: in_range = image min / max
fn rescale_to_dtype(x: &mut [f32]) {
    let (mn, mx) = x.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    rescale_float(x, mn as f64, mx as f64);
}

fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let p = 2 * (n as isize - 1);
    let m = i.rem_euclid(p);
    (if m > n as isize - 1 { p - m } else { m }) as usize
}

/// dipy `estimate_sigma` for one 3-D volume, N = 0, no masking
pub fn estimate_sigma(x: &[f32], s: [usize; 3]) -> f64 {
    let [n0, n1, n2] = s;
    let at = |a: isize, b: isize, c: isize| -> f64 { x[(reflect(a, n0) * n1 + reflect(b, n1)) * n2 + reflect(c, n2)] as f64 };
    let k = (6.0f64 / 7.0).sqrt();
    let sum: f64 = (0..n0)
        .into_par_iter()
        .map(|i| {
            let mut acc = 0.0;
            for j in 0..n1 {
                for l in 0..n2 {
                    let (i, j, l) = (i as isize, j as isize, l as isize);
                    let conv = at(i - 1, j, l) + at(i + 1, j, l) + at(i, j - 1, l) + at(i, j + 1, l) + at(i, j, l - 1) + at(i, j, l + 1);
                    let m = k * (at(i, j, l) - conv / 6.0);
                    acc += m * m;
                }
            }
            acc
        })
        .sum();
    (sum / x.len() as f64).sqrt()
}

/// dipy classic `nlmeans_3d` (constant σ): returns the denoised volume
pub fn nlmeans_classic(x: &[f32], s: [usize; 3], mask: &[bool], sigma: f64, patch_radius: usize, block_radius: usize, rician: bool) -> Vec<f32> {
    let [n0, n1, n2] = s;
    let (p, b) = (patch_radius as isize, block_radius as isize);
    let bs = 2 * block_radius + 1;
    let pv = ((2 * patch_radius + 1) as f64).powi(3); // patch volume
    // reflect-padded copy
    let (q0, q1, q2) = (n0 + 2 * block_radius, n1 + 2 * block_radius, n2 + 2 * block_radius);
    let mut pad = vec![0.0f64; q0 * q1 * q2];
    for i in 0..q0 {
        let si = reflect(i as isize - b, n0);
        for j in 0..q1 {
            let sj = reflect(j as isize - b, n1);
            for l in 0..q2 {
                pad[(i * q1 + j) * q2 + l] = x[(si * n1 + sj) * n2 + reflect(l as isize - b, n2)] as f64;
            }
        }
    }
    // patch-mean σ is the constant σ, so dipy's denom = √2 · σ²
    let denom = 2f64.sqrt() * sigma * sigma;
    let mut out = vec![0.0f32; x.len()];
    out.par_chunks_mut(n1 * n2).enumerate().for_each(|(i, plane)| {
        let mut cache = vec![0.0f64; bs * bs * bs];
        let mut w = vec![0.0f64; bs * bs * bs];
        for j in 0..n1 {
            for l in 0..n2 {
                if !mask[(i * n1 + j) * n2 + l] {
                    continue;
                }
                // block around (i, j, l) in padded coordinates is [i, i + bs)
                for a in 0..bs {
                    for c in 0..bs {
                        let src = ((i + a) * q1 + (j + c)) * q2 + l;
                        cache[(a * bs + c) * bs..(a * bs + c) * bs + bs].copy_from_slice(&pad[src..src + bs]);
                    }
                }
                let ctr = |a: isize, c: isize, d: isize| cache[(((b + a) as usize * bs) + (b + c) as usize) * bs + (b + d) as usize];
                let (mut sumw, mut cnt) = (0.0, 0);
                for m in p..bs as isize - p {
                    for nn in p..bs as isize - p {
                        for o in p..bs as isize - p {
                            let mut sm = 0.0;
                            for a in -p..=p {
                                for c in -p..=p {
                                    for d in -p..=p {
                                        let dd = ctr(a, c, d) - cache[(((m + a) as usize) * bs + (nn + c) as usize) * bs + (o + d) as usize];
                                        sm += dd * dd;
                                    }
                                }
                            }
                            let wt = (-(sm / pv) / denom).exp();
                            sumw += wt;
                            w[cnt] = wt;
                            cnt += 1;
                        }
                    }
                }
                cnt = 0;
                let mut acc = 0.0;
                for m in p..bs as isize - p {
                    for nn in p..bs as isize - p {
                        for o in p..bs as isize - p {
                            let wn = if sumw > 0.0 { w[cnt] / sumw } else { 0.0 };
                            let v = cache[(m as usize * bs + nn as usize) * bs + o as usize];
                            acc += wn * v * v;
                            cnt += 1;
                        }
                    }
                }
                plane[j * n2 + l] = acc as f32;
            }
        }
    });
    // out holds the weighted mean of squares: subtract the Rician bias, sqrt
    out.iter().map(|&v| {
        let mut v = v as f64;
        if rician {
            v -= 2.0 * sigma * sigma;
            if v < 0.0 {
                v = 0.0;
            }
        }
        v.sqrt() as f32
    }).collect()
}

const NR_OF_GRAY: i64 = 1 << 14;

fn clip_histogram(hist: &mut [i64], clip: i64) {
    let mut n_excess: i64 = hist.iter().filter(|&&h| h > clip).map(|&h| h).sum::<i64>() - hist.iter().filter(|&&h| h > clip).count() as i64 * clip;
    for h in hist.iter_mut() {
        if *h > clip {
            *h = clip;
        }
    }
    let size = hist.len() as i64;
    let bin_incr = n_excess / size;
    let upper = clip - bin_incr;
    let n_low = hist.iter().filter(|&&h| h < upper).count() as i64;
    n_excess -= n_low * bin_incr;
    for h in hist.iter_mut() {
        if *h < upper {
            *h += bin_incr;
        }
    }
    let (mut mid_sum, mut mid_n) = (0i64, 0i64);
    for h in hist.iter_mut() {
        if *h >= upper && *h < clip {
            mid_sum += *h;
            mid_n += 1;
            *h = clip;
        }
    }
    n_excess += mid_sum - mid_n * clip;
    while n_excess > 0 {
        let prev = n_excess;
        for index in 0..hist.len() {
            let under_count = hist.iter().filter(|&&h| h < clip).count() as i64;
            let step = (under_count / n_excess).max(1) as usize;
            // numpy: under_mask = (hist < clip)[index::step]; hist[index::step][under_mask] += 1
            let mut k = index;
            let mut added = 0;
            // the mask is evaluated on the histogram before this pass's increments
            let under: Vec<bool> = hist.iter().map(|&h| h < clip).collect();
            while k < hist.len() {
                if under[k] {
                    hist[k] += 1;
                    added += 1;
                }
                k += step;
            }
            n_excess -= added;
            if n_excess <= 0 {
                break;
            }
        }
        if prev == n_excess {
            break;
        }
    }
}

/// skimage `equalize_adapthist` on a float32 volume in [0, 1] (3-D, default kernel, clip 0.01, 256 bins).
pub fn equalize_adapthist(x: &[f32], s: [usize; 3]) -> Vec<f32> {
    let nd = 3;
    // img_as_uint (float32 computation), rescale to [0, 2^14 - 1] with uint16 truncation
    let u16v: Vec<u16> = x.iter().map(|&v| (v * 65535.0f32).round().clamp(0.0, 65535.0) as u16).collect();
    let (imin, imax) = u16v.iter().fold((u16::MAX, 0u16), |(a, b), &v| (a.min(v), b.max(v)));
    let img: Vec<u16> = u16v
        .iter()
        .map(|&v| {
            if imin != imax {
                (((v as f64 - imin as f64) / (imax as f64 - imin as f64)) * (NR_OF_GRAY - 1) as f64) as u16
            } else {
                v.min(imin) // constant image: skimage returns image*(omax-omin)+omin with no scaling
            }
        })
        .collect();
    let k: Vec<usize> = s.iter().map(|&n| (n / 8).max(1)).collect();
    let pad_start: Vec<usize> = k.iter().map(|&k| k / 2).collect();
    let pad_end: Vec<usize> = (0..nd).map(|d| (k[d] - s[d] % k[d]) % k[d] + (k[d] as f64 / 2.0).ceil() as usize).collect();
    let ps: Vec<usize> = (0..nd).map(|d| s[d] + pad_start[d] + pad_end[d]).collect();
    let bin_size = 1 + NR_OF_GRAY / 256;
    // padded + binned image
    let mut pimg = vec![0u16; ps[0] * ps[1] * ps[2]];
    for i in 0..ps[0] {
        let si = reflect(i as isize - pad_start[0] as isize, s[0]);
        for j in 0..ps[1] {
            let sj = reflect(j as isize - pad_start[1] as isize, s[1]);
            for l in 0..ps[2] {
                let sl = reflect(l as isize - pad_start[2] as isize, s[2]);
                pimg[(i * ps[1] + j) * ps[2] + l] = (img[(si * s[1] + sj) * s[2] + sl] as i64 / bin_size) as u16;
            }
        }
    }
    let nh: Vec<usize> = (0..nd).map(|d| ps[d] / k[d] - 1).collect();
    let kernel_elements: usize = k.iter().product();
    let clim = ((0.01 * kernel_elements as f64) as i64).max(1);
    let nblocks: usize = nh.iter().product();
    // per-block mapping tables
    let maps: Vec<Vec<i64>> = (0..nblocks)
        .into_par_iter()
        .map(|bi| {
            let (b0, b1, b2) = (bi / (nh[1] * nh[2]), (bi / nh[2]) % nh[1], bi % nh[2]);
            let mut hist = vec![0i64; 256];
            for a in 0..k[0] {
                for b in 0..k[1] {
                    let row = ((k[0] / 2 + b0 * k[0] + a) * ps[1] + (k[1] / 2 + b1 * k[1] + b)) * ps[2] + k[2] / 2 + b2 * k[2];
                    for c in 0..k[2] {
                        hist[pimg[row + c] as usize] += 1;
                    }
                }
            }
            clip_histogram(&mut hist, clim);
            let mut cum = 0i64;
            hist.iter()
                .map(|&h| {
                    cum += h;
                    let v = cum as f64 * ((NR_OF_GRAY - 1) as f64 / kernel_elements as f64);
                    v.min((NR_OF_GRAY - 1) as f64) as i64
                })
                .collect()
        })
        .collect();
    // map_array: hist grid padded by one with edge replication → dims nh + 2
    let mh: Vec<usize> = nh.iter().map(|&n| n + 2).collect();
    let map_at = |i: usize, j: usize, l: usize| -> &Vec<i64> {
        let ci = |v: usize, n: usize| v.saturating_sub(1).min(n - 1);
        &maps[(ci(i, nh[0]) * nh[1] + ci(j, nh[1])) * nh[2] + ci(l, nh[2])]
    };
    debug_assert!(mh[0] >= 2);
    let np_: Vec<usize> = (0..nd).map(|d| ps[d] / k[d]).collect();
    // interpolate: block (bi, bj, bl), offset (a, b, c) within the block
    let mut result = vec![0u16; ps[0] * ps[1] * ps[2]];
    result.par_chunks_mut(ps[1] * ps[2]).enumerate().for_each(|(i, plane)| {
        let (bi, a) = (i / k[0], i % k[0]);
        for j in 0..ps[1] {
            let (bj, b) = (j / k[1], j % k[1]);
            for l in 0..ps[2] {
                let (bl, c) = (l / k[2], l % k[2]);
                let v = pimg[(i * ps[1] + j) * ps[2] + l] as usize;
                let t = [a as f64 / k[0] as f64, b as f64 / k[1] as f64, c as f64 / k[2] as f64];
                let mut acc = 0.0f32;
                for edge in 0..8usize {
                    let e = [edge >> 2 & 1, edge >> 1 & 1, edge & 1];
                    let m = map_at(bi + e[0], bj + e[1], bl + e[2]);
                    let mut coef = 1.0f64;
                    for d in (0..nd).rev() {
                        coef *= if e[d] == 0 { 1.0 - t[d] } else { t[d] };
                    }
                    acc += (m[v] as f64 * coef) as f32;
                }
                plane[j * ps[2] + l] = acc as u16;
            }
        }
    });
    let _ = np_;
    // unpad, back to float32 and rescale to [0, 1]
    let mut out = vec![0.0f32; x.len()];
    for i in 0..s[0] {
        for j in 0..s[1] {
            for l in 0..s[2] {
                out[(i * s[1] + j) * s[2] + l] = result[((i + pad_start[0]) * ps[1] + j + pad_start[1]) * ps[2] + l + pad_start[2]] as f32;
            }
        }
    }
    rescale_to_dtype(&mut out);
    out
}

/// Full `T1wPVSNormalization` of a (cropped) volume with C-order shape `s`.
pub fn t1w_pvs_norm(image: &[f32], s: [usize; 3]) -> Vec<f32> {
    pvs_norm(image, s, (1.0, 99.0))
}

/// `T2wPVSNormalization`: identical to the T1 one except for the first percentile pair (2, 98).
pub fn t2w_pvs_norm(image: &[f32], s: [usize; 3]) -> Vec<f32> {
    pvs_norm(image, s, (2.0, 98.0))
}

fn pvs_norm(image: &[f32], s: [usize; 3], pct: (f64, f64)) -> Vec<f32> {
    let mut x = image.to_vec();
    let (mn, mx) = x.iter().filter(|&&v| v != 0.0).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    if !mn.is_finite() {
        return x;
    }
    for v in x.iter_mut() {
        *v = v.clamp(mn, mx);
    }
    let mask: Vec<bool> = x.iter().map(|&v| v > 0.0).collect();
    let (p1, p2) = (percentile(&x, pct.0), percentile(&x, pct.1));
    rescale_float(&mut x, p1, p2);
    let sigma = (estimate_sigma(&x, s) as f32 / 2.0) as f64;
    let mut y = nlmeans_classic(&x, s, &mask, sigma, 1, 2, true);
    let (q1, q2) = (percentile(&y, 0.0), percentile(&y, 98.0));
    rescale_float(&mut y, q1, q2);
    equalize_adapthist(&y, s)
}
