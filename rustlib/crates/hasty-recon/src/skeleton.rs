//! Low-rank (skeleton / CUR) separation of a sample×voxel phase matrix Φ(j, u) = exp(−i ψ(j, u)):
//! Φ ≈ Σ_l a_l(j) b_l(u), with b_l = Φ(j_l, ·) (rows at selected samples) and a = Φ(·, U) G⁺ (columns at selected
//! voxels times the pseudo-inverse of the core). Pivots come from adaptive cross approximation with full pivoting on a
//! random submatrix. Generic in ψ: off-resonance, concomitant fields, anything evaluable per (sample, voxel).

use nalgebra::DMatrix;
use num_complex::Complex64;

pub struct Skeleton {
    /// selected sample indices (rows) and voxel indices (columns)
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    /// G⁺ (L × L), with a(j, ·) = Φ(j, cols) · G⁺
    pub core_pinv: DMatrix<Complex64>,
    /// relative Frobenius error of the approximation on an independent random submatrix
    pub heldout_err: f64,
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % n
    }
}

/// `phase(j, u)` returns ψ. `n_rows`/`n_cols` = full sizes; `cols_pool` restricts columns (e.g. the object support).
pub fn build(phase: &(dyn Fn(usize, usize) -> f64 + Sync), n_rows: usize, cols_pool: &[usize], sub: usize, tol: f64, l_max: usize) -> Skeleton {
    let mut rng = Lcg(0x5EED);
    let rs: Vec<usize> = (0..sub).map(|_| rng.next(n_rows)).collect();
    let cs: Vec<usize> = (0..sub).map(|_| cols_pool[rng.next(cols_pool.len())]).collect();
    let ev = |j: usize, u: usize| Complex64::from_polar(1.0, -phase(j, u));
    // residual matrix on the subsample
    let mut r = DMatrix::<Complex64>::from_fn(sub, sub, |a, b| ev(rs[a], cs[b]));
    let norm0 = r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for _ in 0..l_max {
        let (mut best, mut bi, mut bj) = (0.0, 0, 0);
        for b in 0..sub {
            for a in 0..sub {
                let v = r[(a, b)].norm();
                if v > best {
                    best = v;
                    bi = a;
                    bj = b;
                }
            }
        }
        let rel = r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() / norm0;
        if rel < tol {
            break;
        }
        rows.push(rs[bi]);
        cols.push(cs[bj]);
        let piv = r[(bi, bj)];
        let colv: Vec<Complex64> = (0..sub).map(|a| r[(a, bj)]).collect();
        let rowv: Vec<Complex64> = (0..sub).map(|b| r[(bi, b)] / piv).collect();
        for b in 0..sub {
            for a in 0..sub {
                r[(a, b)] -= colv[a] * rowv[b];
            }
        }
    }
    let l = rows.len();
    let g = DMatrix::<Complex64>::from_fn(l, l, |a, b| ev(rows[a], cols[b]));
    let svd = g.svd(true, true);
    let smax = svd.singular_values.iter().cloned().fold(0.0, f64::max);
    let core_pinv = svd.pseudo_inverse(smax * 1e-10).unwrap();
    // held-out check
    let rs2: Vec<usize> = (0..400).map(|_| rng.next(n_rows)).collect();
    let cs2: Vec<usize> = (0..400).map(|_| cols_pool[rng.next(cols_pool.len())]).collect();
    let cj = DMatrix::<Complex64>::from_fn(400, l, |a, b| ev(rs2[a], cols[b])) * &core_pinv;
    let ru = DMatrix::<Complex64>::from_fn(l, 400, |a, b| ev(rows[a], cs2[b]));
    let approx = cj * ru;
    let (mut num, mut den) = (0.0, 0.0);
    for b in 0..400 {
        for a in 0..400 {
            let t = ev(rs2[a], cs2[b]);
            num += (approx[(a, b)] - t).norm_sqr();
            den += t.norm_sqr();
        }
    }
    Skeleton { rows, cols, core_pinv, heldout_err: (num / den).sqrt() }
}
