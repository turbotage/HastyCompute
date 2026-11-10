//! Separable periodic 3D wavelet engine shared by [`crate::wavelet`] (Daubechies-4 on complex images) and
//! [`crate::dfwavelet`] (divergence-free wavelets on velocity fields).
//!
//! - Data: f32, `lanes` interleaved values per voxel (2 for complex images, 1 for a real field), x fastest, padded box
//!   `p` (every axis a multiple of 2^levels). In-place Mallat layout: after a level on the sub-box `s`, the octant
//!   (εx, εy, εz) ∈ {0, 1}³ of `s` holds subband ε (0 = low-pass, 1 = high-pass along that axis); the next level works
//!   on the octant (0, 0, 0).
//! - Filters: a two-channel bank per axis, periodic boundary (the transform is exact for any even length, also when a
//!   filter is longer than the line).
//! - Speed: every pass treats a line of n samples whose elements are contiguous vectors of V floats (x runs for the y
//!   and z passes; for the x pass a plane is transposed through a scratch buffer so that its vectors span the rows).
//!   The multiply-accumulate over V is SIMD (AVX2 + FMA when the CPU has them, chosen at run time). The x and y passes
//!   of a level run per z plane (the plane stays in cache), the z pass per y row; planes and rows are spread over the
//!   thread pool, so a single image uses every core.

use rayon::prelude::*;

/// Two-channel periodic filter bank along one axis.
///
/// Analysis: lo[k] = Σ_(r,c)∈alo c · x[(2k + r) mod n], hi[k] likewise with `ahi`.
/// Synthesis: x[(2k + r) mod n] += c · lo[k] for (r, c) ∈ slo, plus c · hi[k] for (r, c) ∈ shi.
#[derive(Clone, Debug)]
pub struct Bank {
    pub alo: Vec<(isize, f32)>,
    pub ahi: Vec<(isize, f32)>,
    /// synthesis in gather form, per output parity: (row offset j, coefficient, high band?) with
    /// x[2i + par] = Σ c · band[(i + j) mod n/2]
    gather: [Vec<(isize, f32, bool)>; 2],
}

impl Bank {
    pub fn new(alo: Vec<(isize, f32)>, ahi: Vec<(isize, f32)>, slo: &[(isize, f32)], shi: &[(isize, f32)]) -> Self {
        let mut gather: [Vec<(isize, f32, bool)>; 2] = [vec![], vec![]];
        for (taps, hi) in [(slo, false), (shi, true)] {
            for &(r, c) in taps {
                // x[2k + r] += c band[k]: output m = 2i + par with par ≡ r (mod 2), k = i + (par − r) / 2
                let par = r.rem_euclid(2);
                gather[par as usize].push(((par - r) / 2, c, hi));
            }
        }
        Self { alo, ahi, gather }
    }

    /// orthonormal bank from a low-pass filter h (synthesis = transposed analysis); high-pass g[t] = (−1)^t h[L−1−t]
    pub fn orthonormal(h: &[f64]) -> Self {
        let l = h.len();
        let alo: Vec<(isize, f32)> = h.iter().enumerate().map(|(t, &c)| (t as isize, c as f32)).collect();
        let ahi: Vec<(isize, f32)> = (0..l).map(|t| (t as isize, if t % 2 == 0 { h[l - 1 - t] } else { -h[l - 1 - t] } as f32)).collect();
        Self::new(alo.clone(), ahi.clone(), &alo, &ahi)
    }

    /// biorthogonal bank in BART's convention (src/dfwavelet, filters of length L, symmetric-free periodic form):
    /// analysis lo[i] = Σ_m x[2i + 1 − m] f[m], synthesis x[2j + m − (L − 2)] += g[m] lo[j]
    pub fn bart(lod: &[f64], hid: &[f64], lor: &[f64], hir: &[f64]) -> Self {
        let l = lod.len() as isize;
        let an = |f: &[f64]| f.iter().enumerate().filter(|(_, c)| **c != 0.0).map(|(m, &c)| (1 - m as isize, c as f32)).collect::<Vec<_>>();
        let sy = |f: &[f64]| f.iter().enumerate().filter(|(_, c)| **c != 0.0).map(|(m, &c)| (m as isize - (l - 2), c as f32)).collect::<Vec<_>>();
        Self::new(an(lod), an(hid), &sy(lor), &sy(hir))
    }
}

/// Raw pointer shared by tasks that touch disjoint parts of one buffer.
#[derive(Clone, Copy)]
struct Shared(*mut f32);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

// ── SIMD multiply-accumulate: dst = Σ c_t src_t ────────────────────────────────────────────────────────────────────

const MAXT: usize = 24;

#[inline(always)]
fn mac_impl<const FMA: bool>(dst: &mut [f32], taps: &[(f32, &[f32])]) {
    let v = dst.len();
    let mut j = 0;
    while j + 16 <= v {
        let mut acc = [0f32; 16];
        for (c, s) in taps {
            let s = &s[j..j + 16];
            for l in 0..16 {
                acc[l] = if FMA { c.mul_add(s[l], acc[l]) } else { acc[l] + c * s[l] };
            }
        }
        dst[j..j + 16].copy_from_slice(&acc);
        j += 16;
    }
    for jj in j..v {
        let mut a = 0f32;
        for (c, s) in taps {
            a += c * s[jj];
        }
        dst[jj] = a;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mac_avx2(dst: &mut [f32], taps: &[(f32, &[f32])]) {
    mac_impl::<true>(dst, taps)
}

#[inline]
fn mac(dst: &mut [f32], taps: &[(f32, &[f32])]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the features were detected on this CPU
            return unsafe { mac_avx2(dst, taps) };
        }
    }
    mac_impl::<false>(dst, taps)
}

/// analysis of a line of n vectors (src: n × v): row k of the low band → dst row k, high band → row n/2 + k;
/// `row(i)` gives the offset of output row i in `dst`
#[inline]
fn analyze_line(src: &[f32], n: usize, v: usize, b: &Bank, dst: *mut f32, row: impl Fn(usize) -> usize) {
    let h = n / 2;
    let ni = n as isize;
    for k in 0..h {
        for (taps, o) in [(&b.alo, k), (&b.ahi, h + k)] {
            let mut t: [(f32, &[f32]); MAXT] = [(0.0, &[]); MAXT];
            for (i, &(r, c)) in taps.iter().enumerate() {
                let s = (2 * k as isize + r).rem_euclid(ni) as usize;
                t[i] = (c, &src[s * v..s * v + v]);
            }
            // SAFETY: rows of dst are disjoint and inside the caller's region
            let d = unsafe { std::slice::from_raw_parts_mut(dst.add(row(o)), v) };
            mac(d, &t[..taps.len()]);
        }
    }
}

/// synthesis of a line from src = [lo (n/2 rows) | hi (n/2 rows)] × v into the n output rows `row(m)` of `dst`
#[inline]
fn synthesize_line(src: &[f32], n: usize, v: usize, b: &Bank, dst: *mut f32, row: impl Fn(usize) -> usize) {
    let h = n / 2;
    let hi = h as isize;
    for m in 0..n {
        let (i, par) = (m / 2, m % 2);
        let g = &b.gather[par];
        let mut t: [(f32, &[f32]); MAXT] = [(0.0, &[]); MAXT];
        for (q, &(j, c, high)) in g.iter().enumerate() {
            let k = (i as isize + j).rem_euclid(hi) as usize + if high { h } else { 0 };
            t[q] = (c, &src[k * v..k * v + v]);
        }
        // SAFETY: as in analyze_line
        let d = unsafe { std::slice::from_raw_parts_mut(dst.add(row(m)), v) };
        mac(d, &t[..g.len()]);
    }
}

/// x pass of one plane (sub-box s0 × s1, row stride rs floats): transpose into scratch, filter, transpose back
fn x_pass(plane: &mut [f32], s: [usize; 2], rs: usize, lanes: usize, b: &Bank, inverse: bool, a: &mut Vec<f32>, o: &mut Vec<f32>) {
    let (n, v) = (s[0], s[1] * lanes);
    a.resize(n * v, 0.0);
    o.resize(n * v, 0.0);
    for y in 0..s[1] {
        let r = &plane[y * rs..y * rs + n * lanes];
        for i in 0..n {
            for l in 0..lanes {
                a[i * v + y * lanes + l] = r[i * lanes + l];
            }
        }
    }
    if inverse {
        synthesize_line(a, n, v, b, o.as_mut_ptr(), |i| i * v);
    } else {
        analyze_line(a, n, v, b, o.as_mut_ptr(), |i| i * v);
    }
    for y in 0..s[1] {
        let r = &mut plane[y * rs..y * rs + n * lanes];
        for i in 0..n {
            for l in 0..lanes {
                r[i * lanes + l] = o[i * v + y * lanes + l];
            }
        }
    }
}

/// y pass of one plane: rows of the x run are the vectors
fn y_pass(plane: &mut [f32], s: [usize; 2], rs: usize, lanes: usize, b: &Bank, inverse: bool, a: &mut Vec<f32>) {
    let (n, v) = (s[1], s[0] * lanes);
    a.resize(n * v, 0.0);
    for y in 0..n {
        a[y * v..(y + 1) * v].copy_from_slice(&plane[y * rs..y * rs + v]);
    }
    let dst = plane.as_mut_ptr();
    if inverse {
        synthesize_line(a, n, v, b, dst, |i| i * rs);
    } else {
        analyze_line(a, n, v, b, dst, |i| i * rs);
    }
}

/// z pass of the sub-box s, parallel over y rows
fn z_pass(buf: &mut [f32], p: [usize; 3], s: [usize; 3], lanes: usize, b: &Bank, inverse: bool) {
    let (n, v) = (s[2], s[0] * lanes);
    let (rs, ps) = (p[0] * lanes, p[0] * p[1] * lanes);
    let base = Shared(buf.as_mut_ptr());
    let len = buf.len();
    (0..s[1]).into_par_iter().for_each_init(Vec::new, |a: &mut Vec<f32>, y| {
        let base = base;
        a.resize(n * v, 0.0);
        let off = y * rs;
        for z in 0..n {
            let o = off + z * ps;
            debug_assert!(o + v <= len);
            // SAFETY: row y of every plane is touched by this task only
            let src = unsafe { std::slice::from_raw_parts(base.0.add(o), v) };
            a[z * v..(z + 1) * v].copy_from_slice(src);
        }
        // SAFETY: as above
        let dst = unsafe { base.0.add(off) };
        if inverse {
            synthesize_line(a, n, v, b, dst, |i| i * ps);
        } else {
            analyze_line(a, n, v, b, dst, |i| i * ps);
        }
    });
}

/// x and y passes of every plane z < s[2] (forward: x then y; inverse: y then x)
fn xy_passes(buf: &mut [f32], p: [usize; 3], s: [usize; 3], lanes: usize, bx: &Bank, by: &Bank, inverse: bool) {
    let ps = p[0] * p[1] * lanes;
    let rs = p[0] * lanes;
    buf.par_chunks_mut(ps).take(s[2]).for_each_init(
        || (Vec::new(), Vec::new()),
        |(a, o), plane| {
            if inverse {
                y_pass(plane, [s[0], s[1]], rs, lanes, by, true, a);
                x_pass(plane, [s[0], s[1]], rs, lanes, bx, true, a, o);
            } else {
                x_pass(plane, [s[0], s[1]], rs, lanes, bx, false, a, o);
                y_pass(plane, [s[0], s[1]], rs, lanes, by, false, a);
            }
        },
    );
}

/// forward transform, `levels` levels; `banks[a]` filters axis a
pub fn forward(buf: &mut [f32], p: [usize; 3], lanes: usize, levels: usize, banks: [&Bank; 3]) {
    assert_eq!(buf.len(), p[0] * p[1] * p[2] * lanes);
    for l in 0..levels {
        let s = p.map(|d| d >> l);
        assert!(s.iter().all(|&d| d >= 2 && d % 2 == 0), "axis lengths {p:?} not divisible by 2^{levels}");
        xy_passes(buf, p, s, lanes, banks[0], banks[1], false);
        z_pass(buf, p, s, lanes, banks[2], false);
    }
}

/// inverse of [`forward`]
pub fn inverse(buf: &mut [f32], p: [usize; 3], lanes: usize, levels: usize, banks: [&Bank; 3]) {
    assert_eq!(buf.len(), p[0] * p[1] * p[2] * lanes);
    for l in (0..levels).rev() {
        let s = p.map(|d| d >> l);
        z_pass(buf, p, s, lanes, banks[2], true);
        xy_passes(buf, p, s, lanes, banks[0], banks[1], true);
    }
}

/// padded size of an axis: room for a cycle shift < 2^levels, divisible by 2^levels
pub fn padded(d: [usize; 3], levels: usize) -> [usize; 3] {
    let m = 1usize << levels;
    d.map(|n| (n + m - 1).div_ceil(m) * m)
}

#[inline]
fn refl(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i.rem_euclid(2 * n);
    if i >= n {
        i = 2 * n - 1 - i;
    }
    i as usize
}

/// copy an image (dims d, `lanes` floats per voxel) into the padded box p, cycle-shifted by `shift`, reflected beyond
/// the image
pub fn pad_in(img: &[f32], d: [usize; 3], p: [usize; 3], lanes: usize, shift: [usize; 3], out: &mut Vec<f32>) {
    out.resize(p[0] * p[1] * p[2] * lanes, 0.0);
    let sx: Vec<usize> = (0..p[0]).map(|x| refl(x as isize - shift[0] as isize, d[0])).collect();
    out.par_chunks_mut(p[0] * lanes).enumerate().for_each(|(row, o)| {
        let (y, z) = (row % p[1], row / p[1]);
        let (ys, zs) = (refl(y as isize - shift[1] as isize, d[1]), refl(z as isize - shift[2] as isize, d[2]));
        let src = &img[(d[0] * (ys + d[1] * zs)) * lanes..(d[0] * (ys + d[1] * zs) + d[0]) * lanes];
        for x in 0..p[0] {
            let s = sx[x] * lanes;
            o[x * lanes..x * lanes + lanes].copy_from_slice(&src[s..s + lanes]);
        }
    });
}

/// inverse of [`pad_in`]: the image region of the padded box back into the image
pub fn pad_out(buf: &[f32], d: [usize; 3], p: [usize; 3], lanes: usize, shift: [usize; 3], img: &mut [f32]) {
    img.par_chunks_mut(d[0] * lanes).enumerate().for_each(|(row, o)| {
        let (y, z) = (row % d[1], row / d[1]);
        let q = (shift[0] + p[0] * ((y + shift[1]) + p[1] * (z + shift[2]))) * lanes;
        o.copy_from_slice(&buf[q..q + d[0] * lanes]);
    });
}

/// view a complex slice as interleaved f32
pub fn as_f32(v: &[hasty_cuda::C32]) -> &[f32] {
    // SAFETY: Complex<f32> is repr(C) { re, im }
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() * 2) }
}

pub fn as_f32_mut(v: &mut [hasty_cuda::C32]) -> &mut [f32] {
    // SAFETY: as in as_f32
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut f32, v.len() * 2) }
}

/// The detail subbands of a level: (ε, offset, size) in the padded box, ε = (εx, εy, εz) ≠ 0.
pub fn subbands(p: [usize; 3], level: usize) -> Vec<([usize; 3], [usize; 3], [usize; 3])> {
    // level 1 = finest
    let s = p.map(|d| d >> level);
    (1..8usize)
        .map(|e| {
            let eps = [e & 1, (e >> 1) & 1, (e >> 2) & 1];
            (eps, [0, 1, 2].map(|a| eps[a] * s[a]), s)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, mut s: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn perfect_reconstruction_all_banks() {
        let db4 = Bank::orthonormal(&[-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523]);
        let (b0, b1) = crate::dfwavelet::cdf_banks();
        let p = [24usize, 16, 8];
        for lanes in [1usize, 2] {
            for (name, banks) in [("db4", [&db4, &db4, &db4]), ("cdf", [&b1, &b0, &b0]), ("cdf-z", [&b0, &b0, &b1])] {
                let x = rnd(p.iter().product::<usize>() * lanes, 7 + lanes as u64);
                let mut y = x.clone();
                forward(&mut y, p, lanes, 3, banks);
                inverse(&mut y, p, lanes, 3, banks);
                let err = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(err < 2e-5, "{name} lanes {lanes}: PR error {err}");
            }
        }
        // orthonormal: energy preserved
        let x = rnd(p.iter().product::<usize>() * 2, 99);
        let mut y = x.clone();
        forward(&mut y, p, 2, 3, [&db4, &db4, &db4]);
        let (ex, ey): (f64, f64) = (x.iter().map(|v| (*v as f64).powi(2)).sum(), y.iter().map(|v| (*v as f64).powi(2)).sum());
        assert!((ex / ey - 1.0).abs() < 1e-5, "db4 not orthonormal: {ex} vs {ey}");
    }
}
