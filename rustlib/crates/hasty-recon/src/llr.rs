//! Locally-low-rank denoiser for multi-frame, multi-encode complex images (the classical 4D-flow prior).
//!
//! The image stack x[t][v] (t = frame × encode) is cut into B×B×B blocks (random shift per call = cycle spinning over
//! iterations); every block forms a (B³ × T) Casorati matrix whose singular values are soft-thresholded. The threshold
//! is κ·σ̂ with σ̂ the median, over blocks, of the median singular value of each block's lower half: scaling x by μ
//! scales σ̂ by μ, so D(μx) = μD(x) (normalisation-equivariant, as P²nP requires). Blocks without any support voxel
//! are left unchanged.

use hasty_cuda::C32;
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex32;

pub struct Llr {
    pub block: usize,
    pub kappa: f64,
    pub dims: [usize; 3],
    pub support: Vec<bool>,
    /// rank cap R: a block's threshold is raised to its (R+1)-th largest singular value (soft threshold there; κ = 0
    /// for the cap alone). None (default): κσ̂ only. With T = 280 frames the κσ̂ rule keeps ~200 components (σ̂ lands
    /// in the per-frame aliasing spectrum, research/recon_diag_2026-11-13.md §1); the truth's 99.9 % rank is ~7 per
    /// block for 140 frames.
    pub rank: Option<usize>,
    /// hard threshold: keep the singular values above the threshold unchanged (weight 1), drop the rest; default soft
    /// (shrink by the threshold). With a rank cap the soft rule subtracts s_{R+1} from every kept value and shrinks
    /// the weaker cardiac components most (pf_rank_2mm: dynamics gain falling over the iterations).
    pub hard: bool,
    seed: u64,
}

impl Llr {
    pub fn new(dims: [usize; 3], block: usize, kappa: f64, support: Vec<bool>) -> Self {
        Self { block, kappa, dims, support, rank: None, hard: false, seed: 0x1234_5678 }
    }

    /// in-place on `x` laid out [t][v] (t = 0..nt)
    /// make the block shift of the next `apply` a function of the iteration `k` only (reproducible after a resume)
    pub fn set_iteration(&mut self, k: u64) {
        self.seed = 0x1234_5678 ^ k.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    /// The blocks of the next [`apply`](Self::apply) (voxel indices, x fastest; shifted grid, partial blocks at the
    /// borders included, blocks without a support voxel dropped). Advances the shift seed exactly as `apply` does, so
    /// the GPU path ([`crate::llr_gpu`]) cuts the same blocks.
    pub fn next_blocks(&mut self) -> Vec<Vec<usize>> {
        let d = self.dims;
        let b = self.block;
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let sh = [(self.seed >> 20) as usize % b, (self.seed >> 30) as usize % b, (self.seed >> 40) as usize % b];
        let starts = |n: usize, s: usize| -> Vec<usize> {
            let mut v = vec![0usize];
            let mut o = if s == 0 { b } else { s };
            while o < n {
                v.push(o);
                o += b;
            }
            v
        };
        let (sx, sy, sz) = (starts(d[0], sh[0]), starts(d[1], sh[1]), starts(d[2], sh[2]));
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        for (iz, &z0) in sz.iter().enumerate() {
            let z1 = sz.get(iz + 1).copied().unwrap_or(d[2]);
            for (iy, &y0) in sy.iter().enumerate() {
                let y1 = sy.get(iy + 1).copied().unwrap_or(d[1]);
                for (ix, &x0) in sx.iter().enumerate() {
                    let x1 = sx.get(ix + 1).copied().unwrap_or(d[0]);
                    let mut idx = Vec::with_capacity(b * b * b);
                    let mut any = false;
                    for z in z0..z1 {
                        for y in y0..y1 {
                            for xx in x0..x1 {
                                let q = xx + d[0] * (y + d[1] * z);
                                any |= self.support[q];
                                idx.push(q);
                            }
                        }
                    }
                    if any && idx.len() >= 2 {
                        blocks.push(idx);
                    }
                }
            }
        }
        blocks
    }

    /// σ̂ from the singular values of every block (empty = a skipped block): the median over blocks of each block's
    /// sorted s[len / 4]. None without any decomposed block.
    pub fn sigma_hat<'a>(svs: impl Iterator<Item = &'a [f32]>) -> Option<f64> {
        let mut lows: Vec<f32> = svs
            .filter(|sv| !sv.is_empty())
            .map(|sv| {
                let mut s = sv.to_vec();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s[s.len() / 4]
            })
            .collect();
        lows.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (!lows.is_empty()).then(|| lows[lows.len() / 2] as f64)
    }

    /// a block's threshold: the global `thr`, raised to the block's (rank+1)-th largest singular value with a cap
    pub fn block_threshold(rank: Option<usize>, sv: &[f32], thr: f32) -> f32 {
        match rank {
            Some(r) if r < sv.len() => {
                let mut s = sv.to_vec();
                s.sort_by(|a, b| b.partial_cmp(a).unwrap());
                thr.max(s[r])
            }
            _ => thr,
        }
    }

    /// the singular-value filter shrink(s) / s at threshold `thr`
    pub fn weight(s: f32, thr: f32) -> f32 {
        if s > thr { (s - thr) / s } else { 0.0 }
    }

    /// the filter weight of this operator's rule (soft: [`Self::weight`]; hard: 1 above the threshold, else 0)
    pub fn weight_rule(s: f32, thr: f32, hard: bool) -> f32 {
        if hard {
            if s > thr { 1.0 } else { 0.0 }
        } else {
            Self::weight(s, thr)
        }
    }

    pub fn apply(&mut self, x: &mut [C32], nt: usize) {
        let nv = self.dims[0] * self.dims[1] * self.dims[2];
        let blocks = self.next_blocks();
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let xs: &[C32] = x;
        // pass 1: eigen-decompositions of the Gram matrices, singular values
        let decomp: Vec<(Vec<f32>, DMatrix<Complex32>)> = std::thread::scope(|sc| {
            let chunk = blocks.len().div_ceil(nthreads).max(1);
            let hs: Vec<_> = blocks
                .chunks(chunk)
                .map(|bl| {
                    sc.spawn(move || {
                        bl.iter()
                            .map(|idx| {
                                // f64: the f32 Hermitian eigensolver breaks down on exactly rank-deficient Gram
                                // matrices (identical columns, e.g. equal spline nodes) and returns NaN
                                let a = DMatrix::<num_complex::Complex64>::from_fn(idx.len(), nt, |r, t| {
                                    let v = xs[t * nv + idx[r]];
                                    num_complex::Complex64::new(v.re as f64, v.im as f64)
                                });
                                let g = a.adjoint() * &a;
                                let e = SymmetricEigen::new(g);
                                let ok = e.eigenvalues.iter().all(|l| l.is_finite()) && e.eigenvectors.iter().all(|c| c.re.is_finite() && c.im.is_finite());
                                if !ok {
                                    // leave the block unchanged (empty singular values = skip)
                                    return (vec![], DMatrix::<Complex32>::zeros(0, 0));
                                }
                                let sv: Vec<f32> = e.eigenvalues.iter().map(|&l| l.max(0.0).sqrt() as f32).collect();
                                (sv, e.eigenvectors.map(|c| Complex32::new(c.re as f32, c.im as f32)))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        // σ̂: median over blocks of each block's s[len / 4]
        let Some(sigma) = Self::sigma_hat(decomp.iter().map(|(sv, _)| sv.as_slice())) else {
            return;
        };
        let thr = (self.kappa * sigma) as f32;
        let (rank, hard) = (self.rank, self.hard);
        // pass 2: A ← A V diag(shrink(s)/s) V^H, written back
        let xp = x.as_mut_ptr() as usize;
        std::thread::scope(|sc| {
            let chunk = blocks.len().div_ceil(nthreads).max(1);
            for (bl, de) in blocks.chunks(chunk).zip(decomp.chunks(chunk)) {
                sc.spawn(move || {
                    for (idx, (sv, v)) in bl.iter().zip(de) {
                        if sv.is_empty() {
                            continue;
                        }
                        let tb = Self::block_threshold(rank, sv, thr);
                        let w: Vec<f32> = sv.iter().map(|&s| Self::weight_rule(s, tb, hard)).collect();
                        // filter F = V diag(w) V^H
                        let mut vw = v.clone();
                        for (c, &wc) in w.iter().enumerate() {
                            vw.column_mut(c).scale_mut(wc);
                        }
                        let f = vw * v.adjoint();
                        let d = xp as *mut C32;
                        let a = DMatrix::<Complex32>::from_fn(idx.len(), nt, |r, t| unsafe {
                            let q = *d.add(t * nv + idx[r]);
                            Complex32::new(q.re, q.im)
                        });
                        let out = a * f;
                        // SAFETY: blocks are disjoint
                        for (r, &q) in idx.iter().enumerate() {
                            for t in 0..nt {
                                let o = out[(r, t)];
                                unsafe { *d.add(t * nv + q) = C32::new(o.re, o.im) };
                            }
                        }
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod rank_deficient_tests {
    use super::*;

    #[test]
    fn identical_columns_stay_finite() {
        // 24 images, 3 distinct (all velocity nodes equal at initialisation): exactly rank-deficient blocks
        let dims = [16usize, 16, 8];
        let nv = 16 * 16 * 8;
        let mut x = vec![C32::new(0.0, 0.0); 24 * nv];
        for t in 0..24 {
            for q in 0..nv {
                let c = if t < 8 { 0 } else { t % 3 };
                // complex, as the z nodes are (the case that produced NaN in f32)
                let v = ((q * 7 + c * 13) % 17) as f32 * 0.1 * (q % 5 != 0) as u8 as f32;
                x[t * nv + q] = C32::new(v * 0.8, v * 0.6 - 0.01 * c as f32);
            }
        }
        let mut l = Llr::new(dims, 8, 1.0, vec![true; nv]);
        l.apply(&mut x, 24);
        let bad = x.iter().filter(|c| !c.re.is_finite() || !c.im.is_finite()).count();
        assert_eq!(bad, 0, "{bad} non-finite values after LLR");
    }
}
