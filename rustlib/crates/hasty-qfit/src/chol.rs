//! Modified Cholesky factorisations for (possibly indefinite) small symmetric matrices: P (A + E) Pᵀ = L Lᵀ with E
//! diagonal, E ≥ 0, E = 0 when A is safely positive definite. Written on a full row-major n×n array so the CUDA
//! versions in `gpu.rs` are line-by-line translations.
//!  - GMW81: Gill, Murray & Wright, *Practical Optimization* (1981), §4.4.2.2, with symmetric pivoting on the
//!    largest remaining diagonal.
//!  - SE99: Schnabel & Eskow, "A revised modified Cholesky factorization algorithm", SIAM J. Optim. 9 (1999):
//!    plain pivoted Cholesky while the matrix stays sufficiently positive definite (phase 1), then Gerschgorin-bound
//!    diagonal increments (phase 2); typically a smaller ‖E‖ than GMW81.

/// GMW81 in place: on return the lower triangle of `a` holds L (P (A + E) Pᵀ = L Lᵀ), `perm[k]` is the original
/// index at position k. Returns max_j E_jj.
pub fn gmw81(a: &mut [f64], n: usize, perm: &mut [usize], eps: f64) -> f64 {
    let mut d = vec![0.0f64; n];
    let mut l = vec![0.0f64; n * n];
    for (k, p) in perm.iter_mut().enumerate().take(n) {
        *p = k;
    }
    let (mut gamma, mut xi) = (0.0f64, 0.0f64);
    for i in 0..n {
        gamma = gamma.max(a[i * n + i].abs());
        for j in 0..i {
            xi = xi.max(a[i * n + j].abs());
        }
    }
    let nu = ((n * n) as f64 - 1.0).sqrt().max(1.0);
    let beta2 = gamma.max(xi / nu).max(eps);
    let delta = eps * (gamma + xi).max(1.0);
    let mut emax = 0.0f64;
    for j in 0..n {
        // pivot on the largest remaining |c_kk|
        let mut q = j;
        for k in j + 1..n {
            if a[k * n + k].abs() > a[q * n + q].abs() {
                q = k;
            }
        }
        if q != j {
            for c in 0..n {
                a.swap(j * n + c, q * n + c);
            }
            for r in 0..n {
                a.swap(r * n + j, r * n + q);
            }
            for s in 0..j {
                l.swap(j * n + s, q * n + s);
            }
            perm.swap(j, q);
        }
        for s in 0..j {
            l[j * n + s] = a[j * n + s] / d[s];
        }
        let mut theta = 0.0f64;
        for i in j + 1..n {
            let mut c = a[i * n + j];
            for s in 0..j {
                c -= l[j * n + s] * a[i * n + s];
            }
            a[i * n + j] = c;
            theta = theta.max(c.abs());
        }
        let cjj = a[j * n + j];
        d[j] = delta.max(cjj.abs()).max(theta * theta / beta2);
        emax = emax.max(d[j] - cjj);
        for i in j + 1..n {
            let c = a[i * n + j];
            a[i * n + i] -= c * c / d[j];
        }
    }
    // L D Lᵀ → Cholesky factor
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if j < i {
                l[i * n + j] * d[j].sqrt()
            } else if j == i {
                d[i].sqrt()
            } else {
                0.0
            };
        }
    }
    emax
}

fn swap_sym(a: &mut [f64], n: usize, i: usize, j: usize) {
    if i == j {
        return;
    }
    for c in 0..n {
        a.swap(i * n + c, j * n + c);
    }
    for r in 0..n {
        a.swap(r * n + i, r * n + j);
    }
}

/// one right-looking Cholesky step on column j (the remaining block stays symmetric in both triangles)
fn chol_step(a: &mut [f64], n: usize, j: usize) {
    let ljj = a[j * n + j].sqrt();
    a[j * n + j] = ljj;
    for i in j + 1..n {
        a[i * n + j] /= ljj;
    }
    for i in j + 1..n {
        for k in j + 1..=i {
            let v = a[i * n + k] - a[i * n + j] * a[k * n + j];
            a[i * n + k] = v;
            a[k * n + i] = v;
        }
    }
}

/// SE99 in place (same output convention as [`gmw81`]). Returns max_j E_jj.
pub fn se99(a: &mut [f64], n: usize, perm: &mut [usize], eps: f64) -> f64 {
    for (k, p) in perm.iter_mut().enumerate().take(n) {
        *p = k;
    }
    let tau = eps.cbrt();
    let taubar = tau * tau;
    let mu = 0.1;
    let mut gamma = 0.0f64;
    for i in 0..n {
        gamma = gamma.max(a[i * n + i].abs());
    }
    if gamma == 0.0 {
        gamma = 1.0;
    }
    let mut j = 0usize;
    let mut phase1 = true;
    let mut delta_prev = 0.0f64;
    let mut emax = 0.0f64;
    // phase 1: pivoted Cholesky while the remaining diagonal stays sufficiently positive
    while j < n {
        let mut imax = j;
        for i in j + 1..n {
            if a[i * n + i] > a[imax * n + imax] {
                imax = i;
            }
        }
        if a[imax * n + imax] < taubar * gamma {
            phase1 = false;
            break;
        }
        swap_sym(a, n, j, imax);
        perm.swap(j, imax);
        let ajj = a[j * n + j];
        let mut ok = true;
        for i in j + 1..n {
            let aij = a[i * n + j];
            if a[i * n + i] - aij * aij / ajj < -mu * gamma {
                ok = false;
                break;
            }
        }
        if !ok {
            phase1 = false;
            break;
        }
        chol_step(a, n, j);
        j += 1;
    }
    if !phase1 {
        if j == n - 1 {
            // only the last element remains
            let ajj = a[j * n + j];
            let dl = (-ajj + (tau * (-ajj) / (1.0 - tau)).max(taubar * gamma)).max(0.0);
            a[j * n + j] = ajj + dl;
            emax = emax.max(dl);
            a[j * n + j] = a[j * n + j].sqrt();
        } else {
            // phase 2: Gerschgorin lower bounds of the remaining block
            let k0 = j;
            let mut g = vec![0.0f64; n];
            for i in k0..n {
                let mut s = 0.0;
                for c in k0..n {
                    if c != i {
                        s += a[i * n + c].abs();
                    }
                }
                g[i] = a[i * n + i] - s;
            }
            for jj in k0..n - 2 {
                let mut imax = jj;
                for i in jj + 1..n {
                    if g[i] > g[imax] {
                        imax = i;
                    }
                }
                if imax != jj {
                    swap_sym(a, n, jj, imax);
                    perm.swap(jj, imax);
                    g.swap(jj, imax);
                }
                let mut normj = 0.0;
                for i in jj + 1..n {
                    normj += a[i * n + jj].abs();
                }
                let ajj = a[jj * n + jj];
                let dl = (-ajj + normj.max(taubar * gamma)).max(delta_prev).max(0.0);
                if dl > 0.0 {
                    a[jj * n + jj] = ajj + dl;
                    delta_prev = dl;
                    emax = emax.max(dl);
                }
                let ajj = a[jj * n + jj];
                if ajj != normj {
                    let t = 1.0 - normj / ajj;
                    for i in jj + 1..n {
                        g[i] += a[i * n + jj].abs() * t;
                    }
                }
                chol_step(a, n, jj);
            }
            // final 2×2 block
            let (p, q) = (n - 2, n - 1);
            let (aa, bb, cc) = (a[p * n + p], a[q * n + p], a[q * n + q]);
            let tr = 0.5 * (aa + cc);
            let disc = (0.25 * (aa - cc) * (aa - cc) + bb * bb).sqrt();
            let (lhi, llo) = (tr + disc, tr - disc);
            let dl = (-llo + (tau * (lhi - llo) / (1.0 - tau)).max(taubar * gamma)).max(delta_prev).max(0.0);
            let (aa, cc) = if dl > 0.0 {
                emax = emax.max(dl);
                (aa + dl, cc + dl)
            } else {
                (aa, cc)
            };
            let l11 = aa.sqrt();
            let l21 = bb / l11;
            a[p * n + p] = l11;
            a[q * n + p] = l21;
            a[q * n + q] = (cc - l21 * l21).max(0.0).sqrt();
        }
    }
    for i in 0..n {
        for c in i + 1..n {
            a[i * n + c] = 0.0;
        }
    }
    emax
}

/// solve (A + E) x = b with the factor from [`gmw81`] / [`se99`]
pub fn chol_solve(l: &[f64], n: usize, perm: &[usize], b: &[f64], x: &mut [f64]) {
    let mut y = vec![0.0f64; n];
    for k in 0..n {
        let mut s = b[perm[k]];
        for c in 0..k {
            s -= l[k * n + c] * y[c];
        }
        y[k] = s / l[k * n + k];
    }
    for k in (0..n).rev() {
        let mut s = y[k];
        for r in k + 1..n {
            s -= l[r * n + k] * y[r];
        }
        y[k] = s / l[k * n + k];
    }
    for k in 0..n {
        x[perm[k]] = y[k];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// check P (A + E) Pᵀ = L Lᵀ with E diagonal ≥ 0; returns ‖E‖∞
    fn check(f: fn(&mut [f64], usize, &mut [usize], f64) -> f64, a0: &[f64], n: usize) -> f64 {
        let mut a = a0.to_vec();
        let mut perm = vec![0; n];
        let emax = f(&mut a, n, &mut perm, f64::EPSILON);
        let mut llt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    llt[i * n + j] += a[i * n + k] * a[j * n + k];
                }
            }
        }
        let scale = a0.iter().fold(1.0f64, |m, x| m.max(x.abs()));
        for i in 0..n {
            assert!(a[i * n + i] > 0.0, "non-positive pivot");
            for j in 0..n {
                let orig = a0[perm[i] * n + perm[j]];
                let diff = llt[i * n + j] - orig;
                if i == j {
                    assert!(diff >= -1e-10 * scale, "negative E at {i}: {diff}");
                    assert!(diff <= emax + 1e-9 * scale, "E above reported max");
                } else {
                    assert!(diff.abs() < 1e-9 * scale, "off-diagonal mismatch {diff}");
                }
            }
        }
        emax
    }

    #[test]
    fn modified_cholesky_properties() {
        let mut s = 7u64;
        for n in 1..=9 {
            for trial in 0..200 {
                // random symmetric, half of them positive definite (B Bᵀ + c I)
                let mut b = vec![0.0; n * n];
                for x in b.iter_mut() {
                    *x = lcg(&mut s) * 3.0;
                }
                let mut a = vec![0.0; n * n];
                let pd = trial % 2 == 0;
                for i in 0..n {
                    for j in 0..n {
                        a[i * n + j] = if pd { (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum::<f64>() } else { 0.5 * (b[i * n + j] + b[j * n + i]) };
                    }
                    if pd {
                        a[i * n + i] += 0.5;
                    }
                }
                let e1 = check(gmw81, &a, n);
                let e2 = check(se99, &a, n);
                if pd {
                    assert!(e2 == 0.0, "SE99 modified a PD matrix (n {n}): {e2}");
                }
                let _ = e1;
            }
        }
    }
}
