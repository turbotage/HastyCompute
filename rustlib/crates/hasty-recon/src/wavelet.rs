//! Spatial wavelet shrinkage per image (frame × encode): a classical sparsity prior for P²nP, alone or mixed with the
//! learned denoiser (research/wavelet_prior.md).
//!
//! - 3D separable Daubechies-4 (orthonormal filters), `levels` levels, symmetric (half-sample) extension; axes whose
//!   length is not divisible by 2^levels are padded by reflection to the next multiple and cropped afterwards.
//!   Periodic transform of the padded box on the shared engine ([`crate::sepwt`]: f32, SIMD, all threads per image).
//! - Complex soft-thresholding by magnitude (the phase of each coefficient is kept): c ← c · max(0, 1 − τ / |c|).
//! - Thresholds per detail subband: τ = t · σ̂, σ̂ = median(|d|) / 0.6745 of the finest diagonal subband of that image
//!   (MAD noise estimate). Scaling the image scales σ̂: the map is scale-equivariant, like the learned denoiser. The
//!   approximation band is kept.
//! - Cycle spinning: a circular shift per call ([`Wavelet::set_iteration`]), different every iteration (fixed for a
//!   given iteration, so a resumed run repeats it), shifted back afterwards.

use hasty_cuda::C32;
use rayon::prelude::*;

use crate::sepwt::{as_f32, as_f32_mut, forward, inverse, pad_in, pad_out, padded, Bank};

/// Daubechies-4 (8 taps) low-pass analysis filter, orthonormal
const DB4: [f64; 8] = [
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
    0.6308807679295904,
    0.7148465705525415,
    0.23037781330885523,
];

pub struct Wavelet {
    pub dims: [usize; 3],
    pub levels: usize,
    /// threshold factor t (τ = t · σ̂)
    pub thr: f64,
    shift: [usize; 3],
    bank: Bank,
}

pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// cycle-spinning shift of iteration k: one value in [0, 2^levels) per axis, deterministic in (k, salt)
pub(crate) fn cycle_shift(k: u64, salt: u64, levels: usize) -> [usize; 3] {
    let h = mix64(k ^ salt);
    [0, 1, 2].map(|a| (mix64(h.wrapping_add(a as u64)) % (1u64 << levels)) as usize)
}

/// median of |c| over the complex coefficients of a sub-box, from at most 2^20 evenly strided samples
pub(crate) fn median_abs(buf: &[f32], p: [usize; 3], lanes: usize, off: [usize; 3], size: [usize; 3]) -> f64 {
    let n = size[0] * size[1] * size[2];
    if n == 0 {
        return 0.0;
    }
    let step = n.div_ceil(1 << 20).max(1);
    let mut v: Vec<f32> = (0..n)
        .step_by(step)
        .map(|i| {
            let (x, y, z) = (i % size[0], (i / size[0]) % size[1], i / (size[0] * size[1]));
            let q = ((off[0] + x) + p[0] * ((off[1] + y) + p[1] * (off[2] + z))) * lanes;
            buf[q..q + lanes].iter().map(|c| c * c).sum::<f32>().sqrt()
        })
        .collect();
    let mid = v.len() / 2;
    *v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1 as f64
}

impl Wavelet {
    pub fn new(dims: [usize; 3], levels: usize, thr: f64) -> Self {
        Self { dims, levels: levels.max(1), thr, shift: [0; 3], bank: Bank::orthonormal(&DB4) }
    }

    /// cycle-spinning shift of iteration k (deterministic in k)
    pub fn set_iteration(&mut self, k: u64) {
        self.shift = cycle_shift(k, 0x5741_5645_4c45_54, self.levels);
    }

    /// shrink every image of `stack` (n images of nv voxels, x fastest) in place; one image at a time, each spread
    /// over all threads (one padded copy in memory)
    pub fn apply(&self, stack: &mut [C32], n: usize) {
        let nv: usize = self.dims.iter().product();
        assert_eq!(stack.len(), n * nv);
        let mut buf = Vec::new();
        for img in stack.chunks_mut(nv) {
            self.shrink(img, &mut buf);
        }
    }

    fn shrink(&self, img: &mut [C32], buf: &mut Vec<f32>) {
        let d = self.dims;
        let p = padded(d, self.levels);
        pad_in(as_f32(img), d, p, 2, self.shift, buf);
        let b = &self.bank;
        forward(buf, p, 2, self.levels, [b, b, b]);
        // σ̂ from the finest diagonal subband (x, y, z all high-pass at level 1); |complex Gaussian| has median
        // σ √(ln 4) (σ per component): σ̂ = median / 1.1774
        let h = p.map(|v| v / 2);
        let tau = (self.thr * median_abs(buf, p, 2, h, h) / 1.1774) as f32;
        // every coefficient outside the coarsest approximation block is a detail coefficient
        let a = p.map(|v| v >> self.levels);
        let rs = p[0] * 2;
        buf.par_chunks_mut(p[0] * p[1] * 2).enumerate().for_each(|(z, plane)| {
            for y in 0..p[1] {
                let row = &mut plane[y * rs..(y + 1) * rs];
                let x0 = if z < a[2] && y < a[1] { a[0] } else { 0 };
                for c in row[x0 * 2..].chunks_exact_mut(2) {
                    let mag = (c[0] * c[0] + c[1] * c[1]).sqrt();
                    let s = if mag > tau { 1.0 - tau / mag } else { 0.0 };
                    c[0] *= s;
                    c[1] *= s;
                }
            }
        });
        inverse(buf, p, 2, self.levels, [b, b, b]);
        pad_out(buf, d, p, 2, self.shift, as_f32_mut(img));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_reconstruction_and_shrinkage() {
        let d = [20usize, 12, 9];
        let nv: usize = d.iter().product();
        let img: Vec<C32> = (0..nv).map(|q| C32::new(((q as f32) * 0.37).sin(), ((q as f32) * 0.11).cos())).collect();
        // zero threshold: identity (transform, shift and crop are exact)
        let mut w = Wavelet::new(d, 2, 0.0);
        w.set_iteration(3);
        let mut a = img.clone();
        w.apply(&mut a, 1);
        let err = a.iter().zip(&img).map(|(x, y)| (x.re - y.re).abs().max((x.im - y.im).abs())).fold(0f32, f32::max);
        assert!(err < 1e-5, "perfect reconstruction failed: {err}");
        // scale equivariance: W(μ x) = μ W(x)
        let mut w2 = Wavelet::new(d, 2, 1.0);
        w2.set_iteration(1);
        let (mut b, mut c) = (img.clone(), img.iter().map(|z| C32::new(z.re * 7.0, z.im * 7.0)).collect::<Vec<_>>());
        w2.apply(&mut b, 1);
        w2.apply(&mut c, 1);
        let rel = b.iter().zip(&c).map(|(x, y)| ((7.0 * x.re - y.re).powi(2) + (7.0 * x.im - y.im).powi(2)) as f64).sum::<f64>().sqrt() / c.iter().map(|y| (y.re * y.re + y.im * y.im) as f64).sum::<f64>().sqrt();
        assert!(rel < 1e-5, "not scale-equivariant: {rel}");
        // a smooth image plus noise: shrinkage reduces the error to the clean image
        let clean: Vec<C32> = (0..nv).map(|q| { let (x, y, z) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])); C32::new(((x as f32) * 0.2).cos() + ((y as f32) * 0.3).sin(), 0.5 * ((z as f32) * 0.4).cos()) }).collect();
        let mut s = 12345u64;
        let mut noisy: Vec<C32> = clean.iter().map(|z| { s = mix64(s); let u = (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5; s = mix64(s); let v = (s >> 11) as f32 / (1u64 << 53) as f32 - 0.5; C32::new(z.re + 0.3 * u, z.im + 0.3 * v) }).collect();
        let e0: f64 = noisy.iter().zip(&clean).map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)) as f64).sum();
        let mut w3 = Wavelet::new(d, 2, 1.0);
        w3.set_iteration(0);
        w3.apply(&mut noisy, 1);
        let e1: f64 = noisy.iter().zip(&clean).map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)) as f64).sum();
        assert!(e1 < 0.7 * e0, "shrinkage did not denoise: {e1} vs {e0}");
    }
}
