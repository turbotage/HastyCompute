//! Coil compression guided by the singular values of the (whitened) coil data.
//!
//! The Gram G = Σ_n y_n y_nᴴ (y_n ∈ Cᵗᶜ: one k-space sample over all coils, or one voxel of the coil maps) has
//! eigenvalues σ_1² ≥ … ≥ σ_nc² and eigenvectors u_v. Virtual coil v is u_vᴴ y. Its rows are orthonormal, so white noise
//! stays white (whiten first, Ψ^{-1/2}). The rule picks how many to keep: a fixed count, or the fewest whose
//! eigenvalues hold a fraction of the total energy (default 99.9 %: the discarded part is ≤ 0.1 % of ‖y‖²).
//!
//! ```ignore
//! let mut g = CoilGram::new(nc);
//! g.add(&y, m, stride);                       // coil-major nc × m block, every stride-th sample
//! let cc = CoilCompression::from_gram(&g, CoilRule::Energy(0.999));
//! let yv = cc.apply(&y, m);                   // nvc × m
//! println!("{}", cc.report());
//! ```

use hasty_cuda::C32;
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;
use rayon::prelude::*;

/// How many virtual coils to keep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CoilRule {
    /// a fixed number (capped at the number of coils)
    Fixed(usize),
    /// the fewest virtual coils whose eigenvalues sum to at least this fraction of the total
    Energy(f64),
}

impl CoilRule {
    /// default retained energy of [`CoilRule::Energy`]
    pub const DEFAULT_ENERGY: f64 = 0.999;

    /// "auto" (or "energy") → `Energy(energy)`; a number n → `Fixed(n)`
    pub fn parse(s: &str, energy: f64) -> Result<Self, String> {
        match s {
            "auto" | "energy" => {
                if !(energy > 0.0 && energy <= 1.0) {
                    return Err(format!("coil energy {energy} not in (0, 1]"));
                }
                Ok(Self::Energy(energy))
            }
            n => n.parse::<usize>().map(Self::Fixed).map_err(|_| format!("--vcoils {n}: expected a number or auto")),
        }
    }
}

impl std::fmt::Display for CoilRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(n) => write!(f, "fixed {n}"),
            Self::Energy(e) => write!(f, "energy ≥ {:.3} %", 100.0 * e),
        }
    }
}

/// nc × nc Hermitian Gram Σ y yᴴ, accumulated over samples (data) or voxels (maps).
#[derive(Clone, Debug)]
pub struct CoilGram {
    pub nc: usize,
    /// row-major, g[a nc + b] = Σ y_a conj(y_b)
    g: Vec<Complex64>,
    /// number of vectors added
    pub n: usize,
}

impl CoilGram {
    pub fn new(nc: usize) -> Self {
        Self { nc, g: vec![Complex64::new(0.0, 0.0); nc * nc], n: 0 }
    }

    /// add every `stride`-th column of a coil-major block x (nc × len)
    pub fn add(&mut self, x: &[C32], len: usize, stride: usize) {
        assert!(x.len() >= self.nc * len, "CoilGram::add: block shorter than nc × len");
        self.add_with(len, stride, &|c, j| { let z = x[c * len + j]; Complex64::new(z.re as f64, z.im as f64) });
    }

    /// [`add`](Self::add) for a double-precision block
    pub fn add_c64(&mut self, x: &[Complex64], len: usize, stride: usize) {
        assert!(x.len() >= self.nc * len, "CoilGram::add_c64: block shorter than nc × len");
        self.add_with(len, stride, &|c, j| x[c * len + j]);
    }

    /// add every `stride`-th of `len` vectors, element (coil c, vector j) from `get(c, j)`
    pub fn add_with(&mut self, len: usize, stride: usize, get: &(dyn Fn(usize, usize) -> Complex64 + Sync)) {
        let nc = self.nc;
        let stride = stride.max(1);
        let cols: Vec<usize> = (0..len).step_by(stride).collect();
        let part = cols
            .par_chunks(4096)
            .map(|ch| {
                let mut g = vec![Complex64::new(0.0, 0.0); nc * nc];
                let mut v = vec![Complex64::new(0.0, 0.0); nc];
                for &j in ch {
                    for (c, vc) in v.iter_mut().enumerate() {
                        *vc = get(c, j);
                    }
                    for a in 0..nc {
                        let va = v[a];
                        for b in a..nc {
                            g[a * nc + b] += va * v[b].conj();
                        }
                    }
                }
                g
            })
            .reduce(|| vec![Complex64::new(0.0, 0.0); nc * nc], |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                a
            });
        for a in 0..nc {
            for b in a..nc {
                self.g[a * nc + b] += part[a * nc + b];
            }
        }
        self.n += cols.len();
    }

    /// add another Gram (e.g. accumulated over another chunk)
    pub fn merge(&mut self, o: &CoilGram) {
        assert_eq!(self.nc, o.nc);
        self.g.iter_mut().zip(&o.g).for_each(|(a, b)| *a += b);
        self.n += o.n;
    }

    /// the full Hermitian matrix
    pub fn matrix(&self) -> DMatrix<Complex64> {
        let nc = self.nc;
        DMatrix::from_fn(nc, nc, |a, b| if a <= b { self.g[a * nc + b] } else { self.g[b * nc + a].conj() })
    }
}

/// fewest leading eigenvalues (descending) holding at least `frac` of the total
pub fn count_for_energy(eig_desc: &[f64], frac: f64) -> usize {
    let tot: f64 = eig_desc.iter().map(|v| v.max(0.0)).sum();
    if tot <= 0.0 {
        return eig_desc.len().min(1);
    }
    let mut acc = 0.0;
    for (i, v) in eig_desc.iter().enumerate() {
        acc += v.max(0.0);
        if acc >= frac * tot * (1.0 - 1e-12) {
            return i + 1;
        }
    }
    eig_desc.len()
}

/// A coil compression: virtual coil v = Σ_c rows[v][c] · coil c (orthonormal rows).
#[derive(Clone, Debug)]
pub struct CoilCompression {
    pub rows: Vec<Vec<Complex64>>,
    /// all eigenvalues of the Gram, descending
    pub eig: Vec<f64>,
    /// fraction of the energy kept
    pub retained: f64,
    pub rule: CoilRule,
}

impl CoilCompression {
    pub fn from_gram(g: &CoilGram, rule: CoilRule) -> Self {
        let nc = g.nc;
        let e = SymmetricEigen::new(g.matrix());
        let mut order: Vec<usize> = (0..nc).collect();
        order.sort_by(|&a, &b| e.eigenvalues[b].partial_cmp(&e.eigenvalues[a]).unwrap());
        let eig: Vec<f64> = order.iter().map(|&i| e.eigenvalues[i].max(0.0)).collect();
        let n = match rule {
            CoilRule::Fixed(n) => n.clamp(1, nc),
            CoilRule::Energy(f) => count_for_energy(&eig, f),
        };
        let tot: f64 = eig.iter().sum();
        let retained = if tot > 0.0 { eig[..n].iter().sum::<f64>() / tot } else { 1.0 };
        let rows = order[..n].iter().map(|&i| (0..nc).map(|c| e.eigenvectors[(c, i)].conj()).collect()).collect();
        Self { rows, eig, retained, rule }
    }

    /// no compression (nc virtual coils = the coils)
    pub fn identity(nc: usize) -> Self {
        let rows = (0..nc).map(|v| (0..nc).map(|c| Complex64::new(if c == v { 1.0 } else { 0.0 }, 0.0)).collect()).collect();
        Self { rows, eig: vec![], retained: 1.0, rule: CoilRule::Fixed(nc) }
    }

    pub fn nvc(&self) -> usize {
        self.rows.len()
    }

    pub fn nc(&self) -> usize {
        self.rows.first().map_or(0, |r| r.len())
    }

    /// coil-major x (nc × len) → virtual coils (nvc × len)
    pub fn apply(&self, x: &[C32], len: usize) -> Vec<C32> {
        let mut out = vec![C32::new(0.0, 0.0); self.nvc() * len];
        self.apply_into(x, len, &mut out);
        out
    }

    pub fn apply_into(&self, x: &[C32], len: usize, out: &mut [C32]) {
        let nc = self.nc();
        assert!(x.len() >= nc * len && out.len() >= self.nvc() * len);
        out[..self.nvc() * len].par_chunks_mut(len).zip(self.rows.par_iter()).for_each(|(o, row)| {
            for (j, oj) in o.iter_mut().enumerate() {
                let mut acc = Complex64::new(0.0, 0.0);
                for (c, r) in row.iter().enumerate() {
                    let z = x[c * len + j];
                    acc += r * Complex64::new(z.re as f64, z.im as f64);
                }
                *oj = C32::new(acc.re as f32, acc.im as f32);
            }
        });
    }

    /// the compression of whitened coils composed with the whitener W (nc × nc): one matrix from the raw coils
    pub fn compose_real(&self, w: &[Vec<f64>]) -> Vec<Vec<Complex64>> {
        let nc = self.nc();
        self.rows.iter().map(|r| (0..nc).map(|c| (0..nc).map(|c2| r[c2] * w[c2][c]).sum()).collect()).collect()
    }

    /// one line: kept / total, energy, rule, leading spectrum
    pub fn report(&self) -> String {
        let nc = self.nc();
        let tot: f64 = self.eig.iter().sum();
        let rel: Vec<String> = self.eig.iter().take(nc).map(|v| format!("{:.1e}", if tot > 0.0 { v / tot } else { 0.0 })).collect();
        format!("{} of {nc} virtual coils ({}), {:.3} % energy kept; σ²/Σσ²: [{}]", self.nvc(), self.rule, 100.0 * self.retained, rel.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift normal pairs
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed.max(1);
        move || {
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (a, b) = (u(), u());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
    }

    /// rank-r signal on nc coils (+ optional white noise), coil-major nc × m
    fn data(nc: usize, r: usize, m: usize, noise: f64, seed: u64) -> Vec<C32> {
        let mut g = rng(seed);
        let mix: Vec<Complex64> = (0..nc * r).map(|_| Complex64::new(g(), g())).collect();
        let mut y = vec![C32::new(0.0, 0.0); nc * m];
        for j in 0..m {
            let src: Vec<Complex64> = (0..r).map(|k| Complex64::new(g(), g()) * (1.0 / (1.0 + k as f64))).collect();
            for c in 0..nc {
                let mut z: Complex64 = (0..r).map(|k| mix[c * r + k] * src[k]).sum();
                z += Complex64::new(noise * g(), noise * g());
                y[c * m + j] = C32::new(z.re as f32, z.im as f32);
            }
        }
        y
    }

    #[test]
    fn energy_rule_finds_the_signal_rank() {
        let (nc, r, m) = (16, 5, 4000);
        let y = data(nc, r, m, 0.0, 7);
        let mut g = CoilGram::new(nc);
        g.add(&y, m, 1);
        let cc = CoilCompression::from_gram(&g, CoilRule::Energy(0.999999));
        assert_eq!(cc.nvc(), r, "{}", cc.report());
        // the kept part reproduces the data: ‖Uᴴ U y − y‖ ≈ 0
        let yv = cc.apply(&y, m);
        let back: f64 = (0..m)
            .map(|j| {
                (0..nc)
                    .map(|c| {
                        let z: Complex64 = (0..cc.nvc()).map(|v| cc.rows[v][c].conj() * Complex64::new(yv[v * m + j].re as f64, yv[v * m + j].im as f64)).sum();
                        let d = z - Complex64::new(y[c * m + j].re as f64, y[c * m + j].im as f64);
                        d.norm_sqr()
                    })
                    .sum::<f64>()
            })
            .sum();
        let tot: f64 = y.iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum();
        assert!(back / tot < 1e-9, "residual {}", back / tot);
    }

    #[test]
    fn retained_energy_matches_the_projection_and_noise_stays_white() {
        let (nc, r, m) = (12, 4, 6000);
        let y = data(nc, r, m, 0.3, 11);
        let mut g = CoilGram::new(nc);
        g.add(&y, m, 1);
        let cc = CoilCompression::from_gram(&g, CoilRule::Energy(0.99));
        assert!(cc.retained >= 0.99 && cc.nvc() < nc, "{}", cc.report());
        let yv = cc.apply(&y, m);
        let kept: f64 = yv.iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum();
        let tot: f64 = y.iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum();
        assert!((kept / tot - cc.retained).abs() < 1e-5, "kept {} vs retained {}", kept / tot, cc.retained);
        // orthonormal rows: R Rᴴ = I
        for a in 0..cc.nvc() {
            for b in 0..cc.nvc() {
                let d: Complex64 = (0..nc).map(|c| cc.rows[a][c] * cc.rows[b][c].conj()).sum();
                assert!((d - Complex64::new(if a == b { 1.0 } else { 0.0 }, 0.0)).norm() < 1e-10);
            }
        }
        // pure white noise stays white with unit variance per virtual coil
        let n = data(nc, 0, m, 1.0, 5);
        let nv = cc.apply(&n, m);
        for v in 0..cc.nvc() {
            let p: f64 = nv[v * m..(v + 1) * m].iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>() / m as f64;
            assert!((p - 2.0).abs() < 0.15, "virtual coil {v} noise power {p}");
        }
    }

    #[test]
    fn fixed_rule_and_strided_gram() {
        let (nc, m) = (8, 3000);
        let y = data(nc, 6, m, 0.01, 3);
        let mut a = CoilGram::new(nc);
        a.add(&y, m, 3);
        assert_eq!(a.n, 1000);
        let cc = CoilCompression::from_gram(&a, CoilRule::Fixed(3));
        assert_eq!(cc.nvc(), 3);
        assert_eq!(CoilCompression::from_gram(&a, CoilRule::Fixed(99)).nvc(), nc);
        assert_eq!(CoilRule::parse("auto", 0.999).unwrap(), CoilRule::Energy(0.999));
        assert_eq!(CoilRule::parse("12", 0.999).unwrap(), CoilRule::Fixed(12));
        assert!(CoilRule::parse("x", 0.999).is_err());
        assert_eq!(count_for_energy(&[4.0, 3.0, 2.0, 1.0], 0.7), 2);
        assert_eq!(count_for_energy(&[4.0, 3.0, 2.0, 1.0], 0.71), 3);
    }
}
