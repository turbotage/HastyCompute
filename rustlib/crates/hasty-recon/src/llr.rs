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
    seed: u64,
}

impl Llr {
    pub fn new(dims: [usize; 3], block: usize, kappa: f64, support: Vec<bool>) -> Self {
        Self { block, kappa, dims, support, seed: 0x1234_5678 }
    }

    /// in-place on `x` laid out [t][v] (t = 0..nt)
    pub fn apply(&mut self, x: &mut [C32], nt: usize) {
        let d = self.dims;
        let nv = d[0] * d[1] * d[2];
        let b = self.block;
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let sh = [(self.seed >> 20) as usize % b, (self.seed >> 30) as usize % b, (self.seed >> 40) as usize % b];
        // block origins (shifted grid, partial blocks at the borders included)
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
                                let a = DMatrix::<Complex32>::from_fn(idx.len(), nt, |r, t| {
                                    let v = xs[t * nv + idx[r]];
                                    Complex32::new(v.re, v.im)
                                });
                                let g = a.adjoint() * &a;
                                let e = SymmetricEigen::new(g);
                                let sv: Vec<f32> = e.eigenvalues.iter().map(|&l| l.max(0.0).sqrt()).collect();
                                (sv, e.eigenvectors)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        // σ̂: median over blocks of the median of each block's lower half of singular values
        let mut lows: Vec<f32> = decomp
            .iter()
            .map(|(sv, _)| {
                let mut s = sv.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s[s.len() / 4]
            })
            .collect();
        lows.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sigma = lows[lows.len() / 2] as f64;
        let thr = (self.kappa * sigma) as f32;
        // pass 2: A ← A V diag(shrink(s)/s) V^H, written back
        let xp = x.as_mut_ptr() as usize;
        std::thread::scope(|sc| {
            let chunk = blocks.len().div_ceil(nthreads).max(1);
            for (bl, de) in blocks.chunks(chunk).zip(decomp.chunks(chunk)) {
                sc.spawn(move || {
                    for (idx, (sv, v)) in bl.iter().zip(de) {
                        let w: Vec<f32> = sv.iter().map(|&s| if s > thr { (s - thr) / s } else { 0.0 }).collect();
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
