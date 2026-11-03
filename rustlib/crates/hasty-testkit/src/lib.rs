//! CPU references, error metrics and deterministic test data. No GPU dependency.

use num_complex::Complex;
use rustfft::FftPlanner;

pub type C64 = Complex<f64>;
pub type C32 = Complex<f32>;

/// Small deterministic RNG (xorshift64*), so tests and benches are reproducible without extra crates.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [-1, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
    pub fn complex_vec(&mut self, n: usize) -> Vec<C64> {
        (0..n)
            .map(|_| C64::new(self.uniform(), self.uniform()))
            .collect()
    }
}

/// Relative L2 error ||a - b|| / ||b|| (both f64).
pub fn rel_l2(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        num += (x - y).norm_sqr();
        den += y.norm_sqr();
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

pub fn to_c32(v: &[C64]) -> Vec<C32> {
    v.iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect()
}

pub fn to_c64(v: &[C32]) -> Vec<C64> {
    v.iter()
        .map(|c| C64::new(c.re as f64, c.im as f64))
        .collect()
}

/// Naive O(N^2) 1D DFT, sign = -1 forward / +1 inverse, unnormalised. Exact-ish oracle for small N.
pub fn dft_naive(x: &[C64], sign: i32) -> Vec<C64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut acc = C64::new(0.0, 0.0);
            for (j, v) in x.iter().enumerate() {
                let phase =
                    sign as f64 * 2.0 * std::f64::consts::PI * ((k * j) % n) as f64 / n as f64;
                acc += v * C64::new(phase.cos(), phase.sin());
            }
            acc
        })
        .collect()
}

/// f64 N-D FFT via rustfft, row-major with `dims[last]` contiguous. sign = -1 forward / +1 inverse, unnormalised.
pub fn fft_nd(data: &[C64], dims: &[usize], sign: i32) -> Vec<C64> {
    assert_eq!(data.len(), dims.iter().product::<usize>());
    let mut out = data.to_vec();
    let mut planner = FftPlanner::<f64>::new();
    let ndim = dims.len();
    for axis in 0..ndim {
        let n = dims[axis];
        let stride: usize = dims[axis + 1..].iter().product();
        let outer: usize = dims[..axis].iter().product();
        let plan = if sign < 0 {
            planner.plan_fft_forward(n)
        } else {
            planner.plan_fft_inverse(n)
        };
        let mut line = vec![C64::new(0.0, 0.0); n];
        for o in 0..outer {
            for s in 0..stride {
                let base = o * n * stride + s;
                for i in 0..n {
                    line[i] = out[base + i * stride];
                }
                plan.process(&mut line);
                for i in 0..n {
                    out[base + i * stride] = line[i];
                }
            }
        }
    }
    out
}

/// Direct non-uniform DFT oracle (type-1 style): out[m] = sum_j c_j exp(sign * i * k_j . m) over a mode grid.
/// Placeholder for the NUFFT track; kept here so the oracle lives next to the other references.
pub fn ndft_type2_1d(modes: &[C64], pts: &[f64], sign: i32) -> Vec<C64> {
    let n = modes.len() as i64;
    pts.iter()
        .map(|&x| {
            let mut acc = C64::new(0.0, 0.0);
            for (i, c) in modes.iter().enumerate() {
                let k = (i as i64 - n / 2) as f64;
                let p = sign as f64 * k * x;
                acc += c * C64::new(p.cos(), p.sin());
            }
            acc
        })
        .collect()
}

/// Direct non-uniform DFTs in 1-3 dimensions (f64 oracles). Conventions (same as `hasty-nufft`):
/// modes `k_a in [-(N_a/2) .. N_a - N_a/2)` (centred), mode arrays row-major with the LAST dim fastest (`dims = [.., Ny, Nx]`),
/// point coordinates `x[d][j]` with d = 0 <-> fastest (x), exponent `sign * i * sum_a k_a * x_a`.
pub fn ndft_type1(coords: &[Vec<f64>], c: &[C64], dims: &[usize], sign: i32) -> Vec<C64> {
    let d = dims.len();
    assert_eq!(coords.len(), d);
    let total: usize = dims.iter().product();
    let m = c.len();
    let mut out = vec![C64::new(0.0, 0.0); total];
    for (flat, o) in out.iter_mut().enumerate() {
        let mut r = flat;
        let mut k = vec![0f64; d];
        for a in (0..d).rev() {
            let na = dims[a];
            k[d - 1 - a] = (r % na) as f64 - (na / 2) as f64; // k[0] <-> fastest
            r /= na;
        }
        let mut acc = C64::new(0.0, 0.0);
        for j in 0..m {
            let mut ph = 0.0;
            for a in 0..d {
                ph += k[a] * coords[a][j];
            }
            let ph = sign as f64 * ph;
            acc += c[j] * C64::new(ph.cos(), ph.sin());
        }
        *o = acc;
    }
    out
}

pub fn ndft_type2(coords: &[Vec<f64>], f: &[C64], dims: &[usize], sign: i32) -> Vec<C64> {
    let d = dims.len();
    let m = coords[0].len();
    let total: usize = dims.iter().product();
    assert_eq!(f.len(), total);
    let ks: Vec<Vec<f64>> = (0..total)
        .map(|flat| {
            let mut r = flat;
            let mut k = vec![0f64; d];
            for a in (0..d).rev() {
                let na = dims[a];
                k[d - 1 - a] = (r % na) as f64 - (na / 2) as f64;
                r /= na;
            }
            k
        })
        .collect();
    (0..m)
        .map(|j| {
            let mut acc = C64::new(0.0, 0.0);
            for (flat, k) in ks.iter().enumerate() {
                let mut ph = 0.0;
                for a in 0..d {
                    ph += k[a] * coords[a][j];
                }
                let ph = sign as f64 * ph;
                acc += f[flat] * C64::new(ph.cos(), ph.sin());
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rustfft_matches_naive_dft() {
        let mut rng = Rng::new(1);
        for n in [1usize, 2, 3, 5, 7, 8, 12, 17, 64, 100] {
            let x = rng.complex_vec(n);
            let a = fft_nd(&x, &[n], -1);
            let b = dft_naive(&x, -1);
            assert!(rel_l2(&a, &b) < 1e-12, "n={n}");
        }
    }

    #[test]
    fn nd_fft_roundtrip() {
        let mut rng = Rng::new(2);
        let dims = [4usize, 6, 5];
        let n: usize = dims.iter().product();
        let x = rng.complex_vec(n);
        let y = fft_nd(&fft_nd(&x, &dims, -1), &dims, 1);
        let y: Vec<C64> = y.iter().map(|c| c / n as f64).collect();
        assert!(rel_l2(&y, &x) < 1e-13);
    }
}
