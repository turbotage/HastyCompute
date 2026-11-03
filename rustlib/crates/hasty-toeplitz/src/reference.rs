//! f64 CPU reference operators on tiny grids (dense `V x V`), exact and series-truncated, used as oracles.

use num_complex::Complex64;

use crate::fields::{m_eval, xi, NQ};
use crate::poly::factorial;

pub struct SampleSetHost {
    /// k[a][j], a = 0 (x, fastest) .. 2 (z), radians per voxel
    pub k: [Vec<f64>; 3],
    pub wt: Vec<f64>,
    /// concomitant coefficients w[q][j]
    pub w: [Vec<f64>; NQ],
}

impl SampleSetHost {
    pub fn m(&self) -> usize {
        self.wt.len()
    }
    pub fn phi(&self, j: usize, x: f64, y: f64, z: f64) -> f64 {
        (0..NQ).map(|q| self.w[q][j] * m_eval(q, x, y, z)).sum()
    }
}

/// which approximation of `exp(-i Phi)` / of the pair phase to use
#[derive(Clone, Debug)]
pub enum Series {
    /// exact
    Exact,
    /// ignore Phi
    None,
    /// Taylor of u = e^{-i Phi} to total order P in Phi (then conj(u(r)) u(r'))
    TaylorPhi(usize),
    /// truncated pair series: sum_{m<=n} (i x)^m / m!, x = Phi(r) - Phi(r')
    WeylPair(usize),
    /// u_j(r) ~ sum_l c_l(j) b_l(r): `basis[l][v]` = b_l at voxel v, `coef[l][j]` = c_l(j)
    Basis {
        basis: Vec<Vec<Complex64>>,
        coef: Vec<Vec<Complex64>>,
    },
}

/// `N[r][r'] = sum_j wt_j e^{i k_j.(r - r')} * pair_j(r, r')`, voxels row-major (x fastest) for `dims = [Zn, Yn, Xn]`.
pub fn dense_series_operator(
    dims: [usize; 3],
    s: &SampleSetHost,
    series: &Series,
) -> Vec<Complex64> {
    let v = dims[0] * dims[1] * dims[2];
    let pos: Vec<[f64; 3]> = (0..v)
        .map(|i| {
            [
                i % dims[2],
                (i / dims[2]) % dims[1],
                i / (dims[2] * dims[1]),
            ]
            .map(|x| x as f64)
        })
        .collect();
    let nrm: Vec<[f64; 3]> = (0..v)
        .map(|i| {
            [
                xi(i % dims[2], dims[2]),
                xi((i / dims[2]) % dims[1], dims[1]),
                xi(i / (dims[2] * dims[1]), dims[0]),
            ]
        })
        .collect();
    let mut out = vec![Complex64::new(0.0, 0.0); v * v];
    let i_unit = Complex64::new(0.0, 1.0);
    for j in 0..s.m() {
        // per-voxel quantities for this sample
        let phi: Vec<f64> = nrm.iter().map(|p| s.phi(j, p[0], p[1], p[2])).collect();
        let lin: Vec<Complex64> = pos
            .iter()
            .map(|p| {
                Complex64::from_polar(1.0, s.k[0][j] * p[0] + s.k[1][j] * p[1] + s.k[2][j] * p[2])
            })
            .collect();
        let u: Vec<Complex64> = match series {
            Series::Exact | Series::WeylPair(_) => phi
                .iter()
                .map(|&f| Complex64::from_polar(1.0, -f))
                .collect(),
            Series::None => vec![Complex64::new(1.0, 0.0); v],
            Series::TaylorPhi(p) => phi
                .iter()
                .map(|&f| {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for m in 0..=*p {
                        acc += (-i_unit * f).powi(m as i32) / factorial(m);
                    }
                    acc
                })
                .collect(),
            Series::Basis { basis, coef } => (0..v)
                .map(|r| (0..basis.len()).map(|l| coef[l][j] * basis[l][r]).sum())
                .collect(),
        };
        for r in 0..v {
            for rp in 0..v {
                let pair = match series {
                    Series::WeylPair(n) => {
                        let x = phi[r] - phi[rp];
                        let mut acc = Complex64::new(0.0, 0.0);
                        for m in 0..=*n {
                            acc += (i_unit * x).powi(m as i32) / factorial(m);
                        }
                        acc
                    }
                    _ => u[r].conj() * u[rp],
                };
                out[r * v + rp] += lin[r] * lin[rp].conj() * pair * s.wt[j];
            }
        }
    }
    out
}

pub fn dense_operator(dims: [usize; 3], s: &SampleSetHost) -> Vec<Complex64> {
    dense_series_operator(dims, s, &Series::Exact)
}

pub fn apply_dense(n: &[Complex64], x: &[Complex64]) -> Vec<Complex64> {
    let v = x.len();
    (0..v)
        .map(|r| (0..v).map(|c| n[r * v + c] * x[c]).sum())
        .collect()
}
