//! Host-side NUFFT kernel math: exponential-of-semicircle (ES) kernel, width selection, Fourier series (apodization factors).
//! Conventions follow FINUFFT: kernel support of `w` fine-grid cells, `phi(z) = exp(beta (sqrt(1-z^2) - 1))`, `z = (u - x) / (w/2)`.

use std::f64::consts::PI;

/// Smallest `n' >= n` that is even and has only prime factors in {2,3,5,7} (all supported by `hasty-fft`).
pub fn next_smooth_even(n: usize) -> usize {
    let mut m = n.max(2);
    loop {
        if m % 2 == 0 {
            let mut r = m;
            for p in [2usize, 3, 5, 7] {
                while r % p == 0 {
                    r /= p;
                }
            }
            if r == 1 {
                return m;
            }
        }
        m += 1;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EsKernel {
    pub w: usize,
    pub beta: f64,
    pub sigma: f64,
}

impl EsKernel {
    /// Width and shape from a requested relative tolerance and oversampling factor (FINUFFT rules, validated by the accuracy tests).
    pub fn from_tol(tol: f64, sigma: f64) -> Self {
        let w = Self::width_for(tol, sigma);
        Self::with_width(w, sigma)
    }

    pub fn width_for(tol: f64, sigma: f64) -> usize {
        let w = if (sigma - 2.0).abs() < 1e-12 {
            (10.0 / tol).log10().ceil() as usize
        } else {
            ((1.0 / tol).ln() / (PI * (1.0 - 1.0 / sigma).sqrt())).ceil() as usize + 1
        };
        w.clamp(2, 16)
    }

    pub fn with_width(w: usize, sigma: f64) -> Self {
        let beta = if (sigma - 2.0).abs() < 1e-12 {
            2.30 * w as f64
        } else {
            0.97 * PI * (1.0 - 1.0 / (2.0 * sigma)) * w as f64
        };
        Self { w, beta, sigma }
    }

    #[inline]
    pub fn eval(&self, z: f64) -> f64 {
        if z.abs() >= 1.0 {
            0.0
        } else {
            (self.beta * ((1.0 - z * z).sqrt() - 1.0)).exp()
        }
    }

    /// Fourier transform of the kernel (as a function of the fine-grid coordinate) at integer mode `k` of a grid with `nf` points:
    /// `phi_hat(k) = int phi(2t/w) e^{i 2 pi k t / nf} dt`, by Gauss-Legendre quadrature on the even half.
    pub fn fseries(&self, nf: usize, k: i64) -> f64 {
        let q = (3 * self.w / 2 + 4).max(8);
        let (nodes, weights) = gauss_legendre(q);
        let xi = 2.0 * PI * k as f64 / nf as f64;
        let w = self.w as f64;
        // int_0^1 over z=(1+node)/2 ; dt = (w/2) dz ; even => factor 2
        let mut acc = 0.0;
        for (x, wt) in nodes.iter().zip(&weights) {
            let z = 0.5 * (x + 1.0);
            acc += 0.5 * wt * self.eval(z) * (xi * w * 0.5 * z).cos();
        }
        acc * w // 2 * (w/2)
    }

    /// 1/phi_hat for the `n` centred modes `-n/2 .. n - n/2` of a grid with `nf` points.
    pub fn deconv_factors(&self, nf: usize, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 1.0 / self.fseries(nf, i as i64 - (n / 2) as i64))
            .collect()
    }
}

/// Gauss-Legendre nodes and weights on [-1, 1].
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p1, mut p2) = (1.0, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let dz = p1 / pp;
            z -= dz;
            if dz.abs() < 1e-15 {
                let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
                x[i] = z;
                w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
                break;
            }
        }
    }
    (x, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gl_integrates_polynomials() {
        let (x, w) = gauss_legendre(12);
        let i: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(10)).sum();
        assert!((i - 2.0 / 11.0).abs() < 1e-13);
    }
    #[test]
    fn fseries_matches_direct_sum() {
        let k = EsKernel::from_tol(1e-6, 2.0);
        let nf = 128usize;
        // phi_hat(0) = int phi dt ; compare with a fine trapezoid
        let n = 200000;
        let w = k.w as f64;
        let mut s = 0.0;
        for i in 0..n {
            let t = -w / 2.0 + (i as f64 + 0.5) * w / n as f64;
            s += k.eval(2.0 * t / w) * (w / n as f64);
        }
        assert!((k.fseries(nf, 0) - s).abs() < 1e-9 * s);
        assert!(k.fseries(nf, 30) > 0.0);
    }
    #[test]
    fn smooth() {
        assert_eq!(next_smooth_even(11), 12);
        assert_eq!(next_smooth_even(640), 640);
        assert_eq!(next_smooth_even(641), 648);
    }
}
