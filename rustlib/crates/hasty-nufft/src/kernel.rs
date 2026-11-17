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

/// Kaiser–Bessel window `φ(z) = I₀(β√(1 − z²)) / I₀(β)` on |z| < 1 with Beatty's β = π√((W/σ)²(σ − ½)² − 0.8), evaluated
/// on the GPU without special functions (research/stencil_kernel_optimality.md §2.3, proposal 5; temporal_operator.md
/// §20): φ and its z-derivatives φ⁽ⁿ⁾, n ≤ 4, are entire in z, even / odd, so each is one polynomial
///   φ⁽ⁿ⁾(z) = z^(n mod 2) · Σ_k c_{n,k} y^k,   y = 2z² − 1 ∈ [−1, 1],
/// fitted (Chebyshev in y, then monomials; Horner on the device, coefficients as literals: no tables, no divergent
/// loads) to ≤ 1e-8 of max|φ⁽ⁿ⁾| per order (HASTY_SEG_KB_TOL; degrees 13–15 at W = 10, σ = 1.25). Unlike ES (√ cut-off) the derivatives are bounded up to the edge, so the
/// correction terms need no clamp. In f32 the evaluation error is ≤ 8.5e-7 of max|φ⁽ⁿ⁾| (order 4; 2.3e-7 for φ; unit test).
#[derive(Clone, Debug)]
pub struct KbPoly {
    pub w: usize,
    pub beta: f64,
    pub sigma: f64,
    /// per derivative order n = 0..=4: the coefficients c_{n,k} of the y polynomial (k ascending)
    pub coef: Vec<Vec<f64>>,
    /// per order: max|φ⁽ⁿ⁾| on [−1, 1] and the fit's max error relative to it
    pub scale: Vec<f64>,
    pub fit_err: Vec<f64>,
}

impl KbPoly {
    pub fn beta_for(w: usize, sigma: f64) -> f64 {
        PI * ((w as f64 / sigma).powi(2) * (sigma - 0.5).powi(2) - 0.8).max(1e-6).sqrt()
    }

    /// fit tolerance per order (relative to max|φ⁽ⁿ⁾|): env HASTY_SEG_KB_TOL, default 1e-8
    pub fn default_tol() -> f64 {
        std::env::var("HASTY_SEG_KB_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-8)
    }

    pub fn new(w: usize, sigma: f64) -> Self {
        Self::with_tol(w, sigma, Self::default_tol())
    }

    /// as [`new`](Self::new) with an explicit fit tolerance (the lowest degree per order that reaches it)
    pub fn with_tol(w: usize, sigma: f64, tol: f64) -> Self {
        let beta = Self::beta_for(w, sigma);
        let nodes = 400usize;
        let (mut coef, mut scale, mut fit_err) = (vec![], vec![], vec![]);
        // reference points for the error check: dense in z, including both ends
        let zs: Vec<f64> = (0..=4000).map(|i| (PI * i as f64 / 4000.0).cos()).collect();
        for n in 0..=4usize {
            let sc = zs.iter().map(|&z| Self::exact_of(beta, z, n).abs()).fold(0.0, f64::max);
            let par = (n % 2) as i32;
            // g(y) = φ⁽ⁿ⁾(z) / z^par at z = √((1 + y)/2) (even / odd part), on Chebyshev points of y
            let g = |y: f64| {
                let z = ((1.0 + y) * 0.5).max(0.0).sqrt();
                if par == 1 {
                    // odd orders: φ⁽ⁿ⁾(z)/z is smooth; at z → 0 use a small symmetric difference of the even (n+1) part
                    if z < 1e-4 { Self::exact_of(beta, 1e-4, n) / 1e-4 } else { Self::exact_of(beta, z, n) / z }
                } else {
                    Self::exact_of(beta, z, n)
                }
            };
            let mut chosen = None;
            for d in 6..=24usize {
                let c = cheb_fit(&g, d, nodes);
                let mono = cheb_to_mono(&c);
                let err = zs.iter().map(|&z| (horner(&mono, 2.0 * z * z - 1.0) * z.powi(par) - Self::exact_of(beta, z, n)).abs()).fold(0.0, f64::max) / sc;
                if err <= tol || d == 24 {
                    chosen = Some((mono, err));
                    break;
                }
            }
            let (mono, err) = chosen.unwrap();
            coef.push(mono);
            scale.push(sc);
            fit_err.push(err);
        }
        Self { w, beta, sigma, coef, scale, fit_err }
    }

    /// exact φ⁽ⁿ⁾(z) (f64): g(s) = I₀(β√s)/I₀(β) = Σ_k a_k s^k / I₀(β), a_k = (β²/4)^k/(k!)², s = 1 − z², and
    /// Faà di Bruno with s' = −2z, s'' = −2
    pub fn exact(&self, z: f64, n: usize) -> f64 {
        Self::exact_of(self.beta, z, n)
    }

    fn exact_of(beta: f64, z: f64, n: usize) -> f64 {
        if z.abs() > 1.0 {
            return 0.0;
        }
        let s = 1.0 - z * z;
        let q = beta * beta / 4.0;
        // G_m(s) = Σ_{k ≥ m} a_k k!/(k−m)! s^{k−m}, all terms positive
        let gm = |m: usize, s: f64| {
            let (mut acc, mut ak) = (0.0f64, 1.0f64);
            for k in 0..200usize {
                if k >= m {
                    let mut f = 1.0;
                    for i in 0..m {
                        f *= (k - i) as f64;
                    }
                    acc += ak * f * s.powi((k - m) as i32);
                }
                ak *= q / ((k + 1) as f64 * (k + 1) as f64);
                if ak < 1e-30 * acc.max(1e-300) && k > m + 10 {
                    break;
                }
            }
            acc
        };
        let i0b = gm(0, 1.0);
        let v = match n {
            0 => gm(0, s),
            1 => -2.0 * z * gm(1, s),
            2 => 4.0 * z * z * gm(2, s) - 2.0 * gm(1, s),
            3 => -8.0 * z.powi(3) * gm(3, s) + 12.0 * z * gm(2, s),
            4 => 16.0 * z.powi(4) * gm(4, s) - 48.0 * z * z * gm(3, s) + 12.0 * gm(2, s),
            _ => panic!("KB derivative order {n} > 4"),
        };
        v / i0b
    }

    /// the fitted polynomial in f64 (host checks)
    pub fn eval(&self, z: f64, n: usize) -> f64 {
        if z.abs() > 1.0 {
            return 0.0;
        }
        horner(&self.coef[n], 2.0 * z * z - 1.0) * z.powi((n % 2) as i32)
    }

    /// as [`EsKernel::fseries`], for the KB window
    pub fn fseries(&self, nf: usize, k: i64) -> f64 {
        let (nodes, weights) = gauss_legendre(2 * self.w + 24);
        let xi = 2.0 * PI * k as f64 / nf as f64;
        let w = self.w as f64;
        let mut acc = 0.0;
        for (x, wt) in nodes.iter().zip(&weights) {
            let z = 0.5 * (x + 1.0);
            acc += 0.5 * wt * self.exact(z, 0) * (xi * w * 0.5 * z).cos();
        }
        acc * w
    }

    /// 1/φ̂ for the `n` centred modes of a grid with `nf` points
    pub fn deconv_factors(&self, nf: usize, n: usize) -> Vec<f64> {
        (0..n).map(|i| 1.0 / self.fseries(nf, i as i64 - (n / 2) as i64)).collect()
    }

    /// CUDA source defining `KER_KB` and `kb_eval(z, o)` (orders 0..nk−1, literal coefficients, f32 Horner)
    pub fn cuda(&self) -> String {
        let mut s = format!("#define KER_KB 1\n// Kaiser-Bessel W = {}, beta = {:.17e} (sigma {}), polynomial fit errors {:?}\n", self.w, self.beta, self.sigma, self.fit_err.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>());
        s.push_str("template <int NKO>\n__device__ __forceinline__ void kb_eval(float z, float* o) {\n  if (fabsf(z) >= 1.f) {\n    #pragma unroll\n    for (int n = 0; n < NKO; ++n) o[n] = 0.f;\n    return;\n  }\n  const float y = fmaf(2.f * z, z, -1.f);\n");
        for (n, c) in self.coef.iter().enumerate() {
            let mut e = format!("{:.9e}f", c[c.len() - 1]);
            for k in (0..c.len() - 1).rev() {
                e = format!("fmaf({e}, y, {:.9e}f)", c[k]);
            }
            s.push_str(&format!("  if (NKO > {n}) o[{n}] = {}{e};\n", if n % 2 == 1 { "z * " } else { "" }));
        }
        s.push_str("}\n");
        s
    }
}

/// Chebyshev interpolation coefficients (degree d) of g on [−1, 1] from `nodes` Chebyshev points (least squares =
/// discrete orthogonality)
fn cheb_fit(g: &dyn Fn(f64) -> f64, d: usize, nodes: usize) -> Vec<f64> {
    let th: Vec<f64> = (0..nodes).map(|i| PI * (i as f64 + 0.5) / nodes as f64).collect();
    let gv: Vec<f64> = th.iter().map(|t| g(t.cos())).collect();
    (0..=d)
        .map(|k| {
            let s: f64 = th.iter().zip(&gv).map(|(t, v)| v * (k as f64 * t).cos()).sum();
            s * if k == 0 { 1.0 } else { 2.0 } / nodes as f64
        })
        .collect()
}

/// Chebyshev series → monomial coefficients (ascending)
fn cheb_to_mono(c: &[f64]) -> Vec<f64> {
    let d = c.len();
    let mut out = vec![0.0; d];
    // T_0 = 1, T_1 = x, T_{k+1} = 2x T_k − T_{k−1}
    let (mut tm, mut t) = (vec![0.0; d], vec![0.0; d]);
    tm[0] = 1.0;
    if d > 1 {
        t[1] = 1.0;
    }
    for (k, &ck) in c.iter().enumerate() {
        let tk = if k == 0 { &tm } else { &t };
        for i in 0..d {
            out[i] += ck * tk[i];
        }
        if k >= 1 && k + 1 < d {
            let mut nx = vec![0.0; d];
            for i in 0..d - 1 {
                nx[i + 1] += 2.0 * t[i];
            }
            for i in 0..d {
                nx[i] -= tm[i];
            }
            tm = std::mem::replace(&mut t, nx);
        }
    }
    out
}

fn horner(c: &[f64], y: f64) -> f64 {
    c.iter().rev().fold(0.0, |a, &ck| a * y + ck)
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
    fn kb_poly_fits_and_derivatives() {
        for (w, sigma) in [(10usize, 1.25f64), (8, 1.25), (6, 1.5)] {
            for tol in [1e-7, 1e-6] {
                let k = KbPoly::with_tol(w, sigma, tol);
                eprintln!("KB W {w} σ {sigma} tol {tol:.0e}: degrees {:?}", k.coef.iter().map(|c| c.len() - 1).collect::<Vec<_>>());
            }
            let k = KbPoly::with_tol(w, sigma, 1e-8);
            eprintln!("KB W {w} σ {sigma}: β {:.4}, degrees {:?}, fit errors {:?}", k.beta, k.coef.iter().map(|c| c.len() - 1).collect::<Vec<_>>(), k.fit_err);
            assert!(k.fit_err.iter().all(|&e| e <= 1e-8), "fit errors {:?}", k.fit_err);
            // derivatives of the exact form against central differences of the order below
            for &z in &[0.0, 0.3, -0.7, 0.95, 0.999] {
                for n in 1..=4usize {
                    let h = 1e-4;
                    let fd = (k.exact(z + h, n - 1) - k.exact(z - h, n - 1)) / (2.0 * h);
                    if (z as f64).abs() + h < 1.0 {
                        assert!((fd - k.exact(z, n)).abs() <= 1e-5 * k.scale[n], "order {n} at {z}: {fd} vs {}", k.exact(z, n));
                    }
                }
            }
            // φ̂(0) = ∫φ against a fine midpoint sum
            let n = 200000;
            let wf = w as f64;
            let s: f64 = (0..n).map(|i| k.exact(-1.0 + (i as f64 + 0.5) * 2.0 / n as f64, 0) * wf / n as f64).sum();
            assert!((k.fseries(128, 0) - s).abs() < 1e-8 * s);
            assert!(k.cuda().contains("kb_eval"));
            // f32 evaluation as on the device (f32 coefficients, fmaf Horner in y = 2z² − 1)
            let ef: Vec<f64> = (0..=4usize)
                .map(|n| {
                    let c: Vec<f32> = k.coef[n].iter().map(|&v| v as f32).collect();
                    (0..=20000)
                        .map(|i| {
                            let z = -1.0 + 2.0 * i as f64 / 20000.0;
                            let zf = z as f32;
                            let y = (2.0 * zf).mul_add(zf, -1.0);
                            let mut e = c[c.len() - 1];
                            for q in (0..c.len() - 1).rev() {
                                e = e.mul_add(y, c[q]);
                            }
                            let v = if n % 2 == 1 { zf * e } else { e };
                            (v as f64 - k.exact(zf as f64, n)).abs() / k.scale[n]
                        })
                        .fold(0.0, f64::max)
                })
                .collect();
            eprintln!("KB W {w} σ {sigma}: f32 evaluation max error / max|φ⁽ⁿ⁾| {:?}", ef.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>());
            assert!(ef.iter().all(|&e| e < 3e-6), "f32 KB evaluation error {ef:?}");
        }
    }
    /// ES and its z-derivatives as segnufft.cu `es5` (clamped to 0 for |z| > 0.98)
    fn es_derivs(beta: f64, z: f64) -> [f64; 5] {
        let a = 1.0 - z * z;
        if a <= 0.0 {
            return [0.0; 5];
        }
        let f = (beta * (a.sqrt() - 1.0)).exp();
        if z.abs() > 0.98 {
            return [f, 0.0, 0.0, 0.0, 0.0];
        }
        let (is, ia) = (1.0 / a.sqrt(), 1.0 / a);
        let (is3, is5, is7) = (is * ia, is * ia * ia, is * ia * ia * ia);
        let (u1, u2, u3, u4) = (-beta * z * is, -beta * is3, -3.0 * beta * z * is5, -3.0 * beta * (1.0 + 4.0 * z * z) * is7);
        [f, u1 * f, (u2 + u1 * u1) * f, (u3 + 3.0 * u1 * u2 + u1.powi(3)) * f, (u4 + 4.0 * u1 * u3 + 3.0 * u2 * u2 + 6.0 * u1 * u1 * u2 + u1.powi(4)) * f]
    }

    /// KB polynomial evaluated in f32 as on the device
    fn kb_f32(k: &KbPoly, z: f64) -> [f64; 5] {
        let zf = z as f32;
        if zf.abs() >= 1.0 {
            return [0.0; 5];
        }
        let y = (2.0 * zf).mul_add(zf, -1.0);
        [0, 1, 2, 3, 4].map(|n| {
            let c = &k.coef[n];
            let mut e = c[c.len() - 1] as f32;
            for q in (0..c.len() - 1).rev() {
                e = e.mul_add(y, c[q] as f32);
            }
            (if n % 2 == 1 { zf * e } else { e }) as f64
        })
    }

    /// 1-D interpolation errors of the window and its derivatives (research/stencil_kernel_optimality.md §2.1):
    /// e_n(f, ξ) = |Σ_m ψ⁽ⁿ⁾(f − m) e^{i(f−m)ξ} / ψ̂(ξ) − (−iξ)ⁿ| / (π/σ)ⁿ, ψ(u) = φ(2u/W), |ξ| ≤ π/σ; returns per n
    /// (max, rms) over 48 offsets × 121 frequencies
    fn interp_errors(w: usize, sigma: f64, der: &dyn Fn(f64) -> [f64; 5]) -> [(f64, f64); 5] {
        let (nodes, wts) = gauss_legendre(4 * w + 40);
        let hw = w as f64 / 2.0;
        let mut out = [(0.0f64, 0.0f64); 5];
        let mut cnt = 0usize;
        for iq in 0..121 {
            let xi = PI / sigma * (2.0 * iq as f64 / 120.0 - 1.0);
            // ψ̂(ξ) = ∫ ψ(u) cos(uξ) du (even), u = hw · node
            let ph: f64 = nodes.iter().zip(&wts).map(|(x, wt)| wt * hw * der(*x)[0] * (hw * x * xi).cos()).sum();
            for io in 0..48 {
                let f = io as f64 / 48.0;
                let mut sn = [(0.0f64, 0.0f64); 5];
                for m in -(w as i64)..=(w as i64) {
                    let u = f - m as f64;
                    if u.abs() >= hw {
                        continue;
                    }
                    let d = der(u / hw);
                    let (c, s) = ((u * xi).cos(), (u * xi).sin());
                    for n in 0..5 {
                        let v = d[n] / hw.powi(n as i32);
                        sn[n].0 += v * c;
                        sn[n].1 += v * s;
                    }
                }
                cnt += 1;
                for n in 0..5 {
                    // (−iξ)ⁿ
                    let (mut tr, mut ti) = (1.0, 0.0);
                    for _ in 0..n {
                        (tr, ti) = (ti * xi, -tr * xi);
                    }
                    let e = ((sn[n].0 / ph - tr).powi(2) + (sn[n].1 / ph - ti).powi(2)).sqrt() / (PI / sigma).powi(n as i32);
                    out[n].0 = out[n].0.max(e);
                    out[n].1 += e * e;
                }
            }
        }
        out.map(|(m, s)| (m, (s / cnt as f64).sqrt()))
    }

    /// Interpolation errors e_0..e_4 (max / rms) of the production ES (clamped) and the device KB polynomials (f32
    /// Horner, tolerances 1e-8 and 1e-6) at W 6 / 8 / 10, σ 1.25 (research/temporal_operator.md §20.2 table). Checks:
    /// the f32 polynomials add no material error to the exact KB at the default tolerance (≤ 1.5× + 1e-6 per order),
    /// and KB's rms value error e_0 is below ES's. (KB is NOT uniformly better: its max-norm e_2 / e_4 exceed ES's at
    /// every W, which is why it is not the default.)
    #[test]
    fn kb_poly_interpolation_errors() {
        let fmt = |e: &[(f64, f64); 5]| e.iter().map(|(m, r)| format!("{m:.1e}/{r:.1e}")).collect::<Vec<_>>().join(", ");
        for w in [6usize, 8, 10] {
            let sigma = 1.25;
            let es = EsKernel::with_width(w, sigma);
            let ee = interp_errors(w, sigma, &|z| es_derivs(es.beta, z));
            eprintln!("W {w}, σ {sigma}: e0..e4 max/rms ES (clamped)    [{}]", fmt(&ee));
            let k8 = KbPoly::with_tol(w, sigma, 1e-8);
            let ex = interp_errors(w, sigma, &|z| [0, 1, 2, 3, 4].map(|n| k8.exact(z, n)));
            eprintln!("               KB exact (f64)       [{}]", fmt(&ex));
            for tol in [1e-8, 1e-6] {
                let k = KbPoly::with_tol(w, sigma, tol);
                let ek = interp_errors(w, sigma, &|z| kb_f32(&k, z));
                eprintln!("               KB f32 poly tol {tol:.0e} [{}]", fmt(&ek));
                if tol == 1e-8 {
                    for n in 0..5 {
                        assert!(ek[n].0 <= 1.5 * ex[n].0 + 1e-6 && ek[n].1 <= 1.5 * ex[n].1 + 1e-6, "W {w}: f32 polynomial order {n} {:?} vs exact {:?}", ek[n], ex[n]);
                    }
                    assert!(ek[0].1 < ee[0].1, "W {w}: KB rms e0 {:.2e} ≥ ES {:.2e}", ek[0].1, ee[0].1);
                }
            }
        }
    }
    #[test]
    fn cheb_mono_roundtrip() {
        let c = [0.3, -0.2, 0.5, 0.1, -0.05];
        let m = cheb_to_mono(&c);
        for &x in &[-0.9f64, -0.1, 0.4, 1.0] {
            let t = [1.0, x, 2.0 * x * x - 1.0, 4.0 * x * x * x - 3.0 * x, 8.0 * x.powi(4) - 8.0 * x * x + 1.0];
            let v: f64 = c.iter().zip(&t).map(|(a, b)| a * b).sum();
            assert!((horner(&m, x) - v).abs() < 1e-14);
        }
    }
    #[test]
    fn smooth() {
        assert_eq!(next_smooth_even(11), 12);
        assert_eq!(next_smooth_even(640), 640);
        assert_eq!(next_smooth_even(641), 648);
    }
}
