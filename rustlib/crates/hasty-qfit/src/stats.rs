//! Fisher information and Cramér–Rao bounds of a symbolic model (Gaussian residuals; a complex signal's real and
//! imaginary parts are two residuals with the same variance). Used to check that fits are efficient (reach the CRLB)
//! and to design sampling (echo times) by the bound.

use crate::model::Model;

/// F = Σ_i w_i J_iᵀ J_i / σ² at `p` (n×n row-major). `vars` [n_var][n_data], `fixed` [n_fixed], `w` [n_data] relative
/// weights (None = 1), `sigma2` the noise variance per real residual at w = 1
pub fn fisher(m: &Model, p: &[f64], vars: &[f64], n_data: usize, fixed: &[f64], w: Option<&[f64]>, sigma2: f64) -> Vec<f64> {
    let (n, nout) = (m.n_param(), m.n_out());
    let prog = &m.prog_gn;
    let mut slots = vec![0.0; prog.ops.len()];
    let mut v = vec![0.0; m.n_var()];
    let mut f = vec![0.0; n * n];
    for i in 0..n_data {
        for (k, x) in v.iter_mut().enumerate() {
            *x = vars[k * n_data + i];
        }
        prog.run(p, &v, fixed, &mut slots);
        let wi = w.map_or(1.0, |w| w[i]) / sigma2;
        for k in 0..nout {
            let jk = |j: usize| slots[prog.outputs[nout + k * n + j] as usize];
            for a in 0..n {
                for b in 0..n {
                    f[a * n + b] += wi * jk(a) * jk(b);
                }
            }
        }
    }
    f
}

/// inverse of a symmetric positive definite matrix by Cholesky of the Jacobi-scaled matrix D^{-½} A D^{-½} (parameters
/// of very different units, e.g. amplitudes and ω in rad/s, would otherwise look singular); None if not (numerically)
/// positive definite
pub fn inv_spd(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let d: Vec<f64> = (0..n).map(|j| a[j * n + j]).collect();
    if d.iter().any(|x| !(*x > 0.0)) {
        return None;
    }
    let s: Vec<f64> = d.iter().map(|x| 1.0 / x.sqrt()).collect();
    let a: Vec<f64> = (0..n * n).map(|ij| a[ij] * s[ij / n] * s[ij % n]).collect();
    let inv = inv_spd_unscaled(&a, n)?;
    Some((0..n * n).map(|ij| inv[ij] * s[ij / n] * s[ij % n]).collect())
}

fn inv_spd_unscaled(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    let dmax = (0..n).fold(0.0f64, |m, j| m.max(a[j * n + j]));
    for j in 0..n {
        let mut v = a[j * n + j];
        for k in 0..j {
            v -= l[j * n + k] * l[j * n + k];
        }
        if !(v > 1e3 * f64::EPSILON * dmax) {
            return None;
        }
        l[j * n + j] = v.sqrt();
        for i in j + 1..n {
            let mut u = a[i * n + j];
            for k in 0..j {
                u -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = u / l[j * n + j];
        }
    }
    // A⁻¹ = L⁻ᵀ L⁻¹, column by column
    let mut inv = vec![0.0; n * n];
    let mut z = vec![0.0; n];
    for c in 0..n {
        for i in 0..n {
            let mut s = if i == c { 1.0 } else { 0.0 };
            for k in 0..i {
                s -= l[i * n + k] * z[k];
            }
            z[i] = s / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = z[i];
            for k in i + 1..n {
                s -= l[k * n + i] * inv[k * n + c];
            }
            inv[i * n + c] = s / l[i * n + i];
        }
    }
    Some(inv)
}

/// Cramér–Rao bound: the covariance F⁻¹ (n×n) of any unbiased estimator at `p`
pub fn crlb(m: &Model, p: &[f64], vars: &[f64], n_data: usize, fixed: &[f64], w: Option<&[f64]>, sigma2: f64) -> Option<Vec<f64>> {
    inv_spd(&fisher(m, p, vars, n_data, fixed, w, sigma2), m.n_param())
}
