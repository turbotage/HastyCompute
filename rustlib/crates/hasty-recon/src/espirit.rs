//! ESPIRiT coil sensitivities (Uecker et al., MRM 71:990, 2014) from a Cartesian calibration region, on the CPU.
//!
//! Calibration: every k³ window a_p = R_p x of the multi-coil calibration k-space x (nc × k³ entries) is a row of A.
//! The right singular vectors of A with σ_j > `sv_rel` σ_max span the signal subspace V∥ (from the eigenvectors of the
//! Gram AᴴA). Consistent k-space satisfies x = (1/k³) Σ_p R_pᴴ V∥V∥ᴴ R_p x, a multi-channel convolution, i.e. in image
//! space a pointwise nc × nc operator
//!     W(r) = (1/k³) Σ_j ṽ_j(r) ṽ_j(r)ᴴ,   ṽ_{j,c}(r) = Σ_d v_j[c, d] e^{+2πi d·r / FOV},
//! whose eigenvector with eigenvalue ≈ 1 is the (unit-norm) coil sensitivity vector at r; the eigenvalue drops where
//! there is no signal. W is evaluated on a coarse grid covering the same FOV (the maps are smooth), its top eigenpair
//! per voxel computed there, the per-voxel phase aligned to a virtual reference coil (the dominant direction of all
//! maps), then interpolated (periodic trilinear, in FOV fractions) to the output grid and renormalised: Σ_c |s_c|² = 1
//! where the interpolated eigenvalue exceeds `eig_min`, 0 elsewhere.
//!
//! Conventions match the NUFFT (modes centred): grid index i is at position i − ⌊N/2⌋ voxels; calibration index κ is
//! k = κ − ⌊c/2⌋ in units of 1/FOV, with x[k] = Σ_r ρ(r) e^{−2πi k·r/N} ([`centered_fft3`]).
//! The calibration can come from Cartesian data or from a low-resolution reconstruction of non-Cartesian data
//! (centred FFT of the low-resolution coil images, no density compensation).

use std::sync::Arc;

use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use hasty_nufft::{NufftOpts, NufftPlan};
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};

use crate::gpu::Kernels;
use crate::Error;

#[derive(Clone, Debug)]
pub struct EspiritOpts {
    /// kernel (window) edge in k-space points; capped by the calibration size per axis (1 for a 2D calibration)
    pub kernel: usize,
    /// keep kernels with singular value σ > sv_rel · σ_max
    pub sv_rel: f64,
    /// sensitivities where the (interpolated) top eigenvalue exceeds this; 0 elsewhere
    pub eig_min: f64,
    /// grid of the eigenproblem (same FOV, x fastest); None: half the output grid per axis (≥ 2 kernel, even)
    pub coarse: Option<[usize; 3]>,
    /// full dense eigendecomposition of the calibration Gram (exact, O(n³) single-threaded; n = nc · kernel³) instead
    /// of the randomized subspace iteration for the kept kernels only
    pub dense_eigen: bool,
}

impl Default for EspiritOpts {
    fn default() -> Self {
        Self { kernel: 6, sv_rel: 0.02, eig_min: 0.8, coarse: None, dense_eigen: false }
    }
}

pub struct EspiritMaps {
    pub dims: [usize; 3],
    pub nc: usize,
    /// coil-major nc × nv
    pub maps: Vec<C32>,
    /// interpolated top eigenvalue per output voxel
    pub eig: Vec<f32>,
    /// kernels kept / calibration singular values (descending)
    pub n_kernels: usize,
    pub sv: Vec<f64>,
    pub coarse: [usize; 3],
    /// seconds: calibration Gram, its eigendecomposition, W on the coarse grid, per-voxel eigenpairs + interpolation
    pub timings: [f64; 4],
}

fn nvox(d: [usize; 3]) -> usize {
    d[0] * d[1] * d[2]
}

/// 3D FFT plans (x fastest), unnormalised; inverse uses the + sign
pub struct Fft3 {
    dims: [usize; 3],
    f: [Arc<dyn Fft<f32>>; 3],
}

impl Fft3 {
    pub fn new(dims: [usize; 3], inverse: bool) -> Self {
        let mut p = FftPlanner::<f32>::new();
        let mk = |p: &mut FftPlanner<f32>, n: usize| if inverse { p.plan_fft_inverse(n) } else { p.plan_fft_forward(n) };
        let f = [mk(&mut p, dims[0]), mk(&mut p, dims[1]), mk(&mut p, dims[2])];
        Self { dims, f }
    }

    pub fn process(&self, buf: &mut [Complex32]) {
        let [nx, ny, nz] = self.dims;
        assert_eq!(buf.len(), nx * ny * nz);
        if nx > 1 {
            let f = &self.f[0];
            buf.par_chunks_mut(nx * ny.max(1)).for_each(|s| f.process(s));
        }
        if ny > 1 {
            let f = &self.f[1];
            buf.par_chunks_mut(nx * ny).for_each(|slab| {
                let mut col = vec![Complex32::new(0.0, 0.0); ny];
                for x in 0..nx {
                    for y in 0..ny {
                        col[y] = slab[x + nx * y];
                    }
                    f.process(&mut col);
                    for y in 0..ny {
                        slab[x + nx * y] = col[y];
                    }
                }
            });
        }
        if nz > 1 {
            let f = &self.f[2];
            let nxy = nx * ny;
            let mut t = vec![Complex32::new(0.0, 0.0); buf.len()];
            {
                let b: &[Complex32] = buf;
                t.par_chunks_mut(nz).enumerate().for_each(|(q, col)| {
                    for z in 0..nz {
                        col[z] = b[q + nxy * z];
                    }
                    f.process(col);
                });
            }
            buf.par_chunks_mut(nxy).enumerate().for_each(|(z, slab)| {
                for q in 0..nxy {
                    slab[q] = t[q * nz + z];
                }
            });
        }
    }
}

/// circular shift by s[a] per axis: out[(i + s) mod N] = x[i]
pub fn roll3(x: &[Complex32], dims: [usize; 3], s: [usize; 3]) -> Vec<Complex32> {
    let [nx, ny, nz] = dims;
    let mut out = vec![Complex32::new(0.0, 0.0); x.len()];
    out.par_chunks_mut(nx * ny).enumerate().for_each(|(zo, slab)| {
        let z = (zo + nz - s[2] % nz) % nz;
        for yo in 0..ny {
            let y = (yo + ny - s[1] % ny) % ny;
            for xo in 0..nx {
                let xi = (xo + nx - s[0] % nx) % nx;
                slab[xo + nx * yo] = x[xi + nx * (y + ny * z)];
            }
        }
    });
    out
}

/// centred DFT (both index sets centred at ⌊N/2⌋): X[κ] = Σ_i x[i] e^{∓2πi (κ−⌊N/2⌋)(i−⌊N/2⌋)/N}, unnormalised
pub fn centered_fft3(x: &[Complex32], dims: [usize; 3], inverse: bool) -> Vec<Complex32> {
    let h = dims.map(|n| n / 2);
    let back = [0, 1, 2].map(|a| dims[a] - h[a]);
    let mut b = roll3(x, dims, back);
    Fft3::new(dims, inverse).process(&mut b);
    roll3(&b, dims, h)
}

/// crop the centred calibration region (c per axis) of centred k-space on `dims`
pub fn crop_center(k: &[Complex32], dims: [usize; 3], c: [usize; 3]) -> Vec<Complex32> {
    let o = [0, 1, 2].map(|a| dims[a] / 2 - c[a] / 2);
    let mut out = Vec::with_capacity(nvox(c));
    for z in 0..c[2] {
        for y in 0..c[1] {
            for x in 0..c[0] {
                out.push(k[(x + o[0]) + dims[0] * ((y + o[1]) + dims[1] * (z + o[2]))]);
            }
        }
    }
    out
}

/// all eigenpairs of a Hermitian n × n (row-major), descending: (values, vectors)
fn dense_top(g: &[Complex64], n: usize, _thr: f64) -> (Vec<f64>, Vec<Vec<Complex64>>) {
    let e = SymmetricEigen::new(DMatrix::<Complex64>::from_fn(n, n, |i, j| g[i * n + j]));
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| e.eigenvalues[b].partial_cmp(&e.eigenvalues[a]).unwrap());
    (order.iter().map(|&i| e.eigenvalues[i]).collect(), order.iter().map(|&i| (0..n).map(|r| e.eigenvectors[(r, i)]).collect()).collect())
}

/// out (n × b, row-major) = G (n × n, row-major) · Q (n × b, row-major)
fn matmul(g: &[Complex64], q: &[Complex64], n: usize, b: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * b];
    out.par_chunks_mut(b).enumerate().for_each(|(i, oi)| {
        let gi = &g[i * n..(i + 1) * n];
        for (k, gik) in gi.iter().enumerate() {
            if gik.re == 0.0 && gik.im == 0.0 {
                continue;
            }
            let qk = &q[k * b..(k + 1) * b];
            for (o, x) in oi.iter_mut().zip(qk) {
                *o += gik * x;
            }
        }
    });
    out
}

/// Qᴴ Y (b × b) for row-major n × b blocks
fn gram_ab(q: &[Complex64], y: &[Complex64], n: usize, b: usize) -> DMatrix<Complex64> {
    let parts: Vec<Complex64> = (0..n)
        .into_par_iter()
        .fold(|| vec![Complex64::new(0.0, 0.0); b * b], |mut acc, i| {
            let (qi, yi) = (&q[i * b..(i + 1) * b], &y[i * b..(i + 1) * b]);
            for a in 0..b {
                let qa = qi[a].conj();
                for (c, yc) in yi.iter().enumerate() {
                    acc[a * b + c] += qa * yc;
                }
            }
            acc
        })
        .reduce(|| vec![Complex64::new(0.0, 0.0); b * b], |mut x, y| {
            x.iter_mut().zip(&y).for_each(|(a, c)| *a += c);
            x
        });
    DMatrix::from_fn(b, b, |a, c| parts[a * b + c])
}

/// orthonormal basis of the columns of Y (n × b): Y V Λ^{-1/2} over the eigenpairs of YᴴY above a relative floor
fn orth(y: &[Complex64], n: usize, b: usize) -> (Vec<Complex64>, usize) {
    let s = gram_ab(y, y, n, b);
    let s = (&s + s.adjoint()) * Complex64::new(0.5, 0.0);
    let e = SymmetricEigen::new(s);
    let lmax = e.eigenvalues.iter().cloned().fold(0.0, f64::max);
    let keep: Vec<usize> = (0..b).filter(|&i| e.eigenvalues[i] > 1e-13 * lmax).collect();
    let nb = keep.len();
    let t: Vec<Complex64> = (0..b).flat_map(|r| keep.iter().map(move |&i| (r, i))).map(|(r, i)| e.eigenvectors[(r, i)] / e.eigenvalues[i].sqrt()).collect();
    (matmul_small(y, &t, n, b, nb), nb)
}

/// Y (n × b) · T (b × c), row-major
fn matmul_small(y: &[Complex64], t: &[Complex64], n: usize, b: usize, c: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * c];
    out.par_chunks_mut(c).enumerate().for_each(|(i, oi)| {
        let yi = &y[i * b..(i + 1) * b];
        for (k, yk) in yi.iter().enumerate() {
            let tk = &t[k * c..(k + 1) * c];
            for (o, x) in oi.iter_mut().zip(tk) {
                *o += yk * x;
            }
        }
    });
    out
}

/// leading eigenpairs of a Hermitian PSD n × n (row-major) down to `thr` · λ_max, descending, by randomized subspace
/// iteration (block b, 4 power steps, Rayleigh–Ritz); b doubles while the smallest Ritz value is still above the
/// threshold. Returns all Ritz pairs of the final block (the caller thresholds).
fn subspace_top(g: &[Complex64], n: usize, thr: f64) -> (Vec<f64>, Vec<Vec<Complex64>>) {
    let mut b = n.min(256);
    let mut seed = 0x9E3779B97F4A7C15u64;
    loop {
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let omega: Vec<Complex64> = (0..n * b).map(|_| Complex64::new(rnd(), rnd())).collect();
        let (mut q, mut nb) = orth(&matmul(g, &omega, n, b), n, b);
        for _ in 0..4 {
            let (q2, nb2) = orth(&matmul(g, &q, n, nb), n, nb);
            q = q2;
            nb = nb2;
        }
        // Rayleigh–Ritz
        let gq = matmul(g, &q, n, nb);
        let t = gram_ab(&q, &gq, n, nb);
        let t = (&t + t.adjoint()) * Complex64::new(0.5, 0.0);
        let e = SymmetricEigen::new(t);
        let mut order: Vec<usize> = (0..nb).collect();
        order.sort_by(|&a, &c| e.eigenvalues[c].partial_cmp(&e.eigenvalues[a]).unwrap());
        let vals: Vec<f64> = order.iter().map(|&i| e.eigenvalues[i]).collect();
        let lmax = vals[0].max(1e-300);
        if b < n && vals[nb - 1] > thr * lmax {
            b = (2 * b).min(n);
            continue;
        }
        let u: Vec<Complex64> = (0..nb).flat_map(|r| order.iter().map(move |&i| (r, i))).map(|(r, i)| e.eigenvectors[(r, i)]).collect();
        let v = matmul_small(&q, &u, n, nb, nb);
        let vecs = (0..nb).map(|j| (0..n).map(|r| v[r * nb + j]).collect()).collect();
        return (vals, vecs);
    }
}

fn to32(z: C32) -> Complex32 {
    Complex32::new(z.re, z.im)
}

/// ESPIRiT maps on the output grid `out` from a centred calibration k-space (coil-major nc × cdims, x fastest)
pub fn espirit(calib: &[C32], cdims: [usize; 3], nc: usize, out: [usize; 3], o: &EspiritOpts) -> EspiritMaps {
    let ncal = nvox(cdims);
    assert_eq!(calib.len(), nc * ncal, "calibration: nc × cdims");
    let mut tm = [0f64; 4];
    let mut clock = std::time::Instant::now();
    let mut lap = |i: usize, tm: &mut [f64; 4]| { tm[i] = clock.elapsed().as_secs_f64(); clock = std::time::Instant::now(); };
    let kk = cdims.map(|c| o.kernel.min(c).max(1));
    let nk = nvox(kk);
    let n = nc * nk;
    let nw = [0, 1, 2].map(|a| cdims[a] - kk[a] + 1);
    let np = nvox(nw);
    // windows: column i = (c, d) of A over all positions p (Aᵀ, n × P, contiguous per column)
    let mut at = vec![Complex32::new(0.0, 0.0); n * np];
    at.par_chunks_mut(np).enumerate().for_each(|(i, col)| {
        let (c, d) = (i / nk, i % nk);
        let (dx, dy, dz) = (d % kk[0], (d / kk[0]) % kk[1], d / (kk[0] * kk[1]));
        for pz in 0..nw[2] {
            for py in 0..nw[1] {
                for px in 0..nw[0] {
                    let p = px + nw[0] * (py + nw[1] * pz);
                    col[p] = to32(calib[c * ncal + (px + dx) + cdims[0] * ((py + dy) + cdims[1] * (pz + dz))]);
                }
            }
        }
    });
    // Gram AᴴA: G[i][j] = Σ_p conj(A_pi) A_pj
    let rows: Vec<Vec<Complex64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let ci = &at[i * np..(i + 1) * np];
            (0..n)
                .map(|j| {
                    if j < i {
                        return Complex64::new(0.0, 0.0);
                    }
                    let cj = &at[j * np..(j + 1) * np];
                    let mut s = Complex64::new(0.0, 0.0);
                    for p in 0..np {
                        let (a, b) = (ci[p], cj[p]);
                        s += Complex64::new((a.re * b.re + a.im * b.im) as f64, (a.re * b.im - a.im * b.re) as f64);
                    }
                    s
                })
                .collect()
        })
        .collect();
    drop(at);
    // full row-major Hermitian Gram
    let mut g = vec![Complex64::new(0.0, 0.0); n * n];
    g.par_chunks_mut(n).enumerate().for_each(|(i, gi)| {
        for j in 0..n {
            gi[j] = if i <= j { rows[i][j] } else { rows[j][i].conj() };
        }
    });
    drop(rows);
    lap(0, &mut tm);
    let thr = o.sv_rel * o.sv_rel; // on eigenvalues σ² relative to σ²_max
    let (vals, vecs) = if o.dense_eigen { dense_top(&g, n, thr) } else { subspace_top(&g, n, thr) };
    drop(g);
    let sv: Vec<f64> = vals.iter().map(|v| v.max(0.0).sqrt()).collect();
    let nkeep = sv.iter().take_while(|&&s| s > o.sv_rel * sv[0]).count().max(1);
    // A = UΣVᴴ: a window a_p = Σ_j U_pj σ_j conj(v_j), so the signal subspace of the windows is spanned by conj(v_j)
    let kernels: Vec<Vec<Complex32>> = vecs[..nkeep].iter().map(|v| v.iter().map(|z| Complex32::new(z.re as f32, -z.im as f32)).collect()).collect();
    lap(1, &mut tm);
    // coarse grid
    let cd = o.coarse.unwrap_or_else(|| [0, 1, 2].map(|a| {
        if out[a] <= 1 {
            1
        } else {
            let h = out[a].div_ceil(2).max(2 * kk[a]).min(out[a]);
            h + (h % 2)
        }
    }));
    let ncv = nvox(cd);
    let half = cd.map(|n| n / 2);
    let ifft = Fft3::new(cd, true);
    // W per coarse voxel, full nc × nc (row a, column b), accumulated over kernel batches
    let mut w = vec![Complex32::new(0.0, 0.0); ncv * nc * nc];
    let batch = 8usize;
    for b0 in (0..nkeep).step_by(batch) {
        let js: Vec<usize> = (b0..(b0 + batch).min(nkeep)).collect();
        // ṽ_{j,c} on the coarse grid, centred index order
        let vt: Vec<Vec<Complex32>> = (0..js.len() * nc)
            .into_par_iter()
            .map(|i| (js[i / nc], i % nc))
            .map(|(j, c)| {
                let mut buf = vec![Complex32::new(0.0, 0.0); ncv];
                for d in 0..nk {
                    let (dx, dy, dz) = (d % kk[0], (d / kk[0]) % kk[1], d / (kk[0] * kk[1]));
                    buf[(dx % cd[0]) + cd[0] * ((dy % cd[1]) + cd[1] * (dz % cd[2]))] += kernels[j][c * nk + d];
                }
                ifft.process(&mut buf);
                // position r − ⌊N/2⌋ ↔ FFT index (r − ⌊N/2⌋) mod N
                roll3(&buf, cd, half)
            })
            .collect();
        let nb = js.len();
        w.par_chunks_mut(nc * nc).enumerate().for_each(|(q, wq)| {
            for jb in 0..nb {
                let v = &vt[jb * nc..(jb + 1) * nc];
                for a in 0..nc {
                    let va = v[a][q];
                    for bb in 0..nc {
                        wq[a * nc + bb] += va * v[bb][q].conj();
                    }
                }
            }
        });
    }
    lap(2, &mut tm);
    let scale = 1.0 / nk as f32;
    // top eigenpair per coarse voxel
    let top: Vec<(f32, Vec<Complex32>)> = w
        .par_chunks(nc * nc)
        .map(|wq| {
            let m = DMatrix::<Complex64>::from_fn(nc, nc, |a, b| {
                let z = wq[a * nc + b] * scale;
                Complex64::new(z.re as f64, z.im as f64)
            });
            let m = (&m + m.adjoint()) * Complex64::new(0.5, 0.0);
            let e = SymmetricEigen::new(m);
            let (mut bi, mut bv) = (0, f64::NEG_INFINITY);
            for i in 0..nc {
                if e.eigenvalues[i] > bv {
                    bv = e.eigenvalues[i];
                    bi = i;
                }
            }
            (bv as f32, (0..nc).map(|c| { let z = e.eigenvectors[(c, bi)]; Complex32::new(z.re as f32, z.im as f32) }).collect())
        })
        .collect();
    drop(w);
    // phase reference: dominant direction of Σ_q λ_q s_q s_qᴴ over the voxels with signal
    let mut mref = DMatrix::<Complex64>::zeros(nc, nc);
    for (l, s) in &top {
        if (*l as f64) > o.eig_min {
            for a in 0..nc {
                for b in 0..nc {
                    let z = s[a] * s[b].conj() * *l;
                    mref[(a, b)] += Complex64::new(z.re as f64, z.im as f64);
                }
            }
        }
    }
    let er = SymmetricEigen::new(mref);
    let ri = (0..nc).max_by(|&a, &b| er.eigenvalues[a].partial_cmp(&er.eigenvalues[b]).unwrap()).unwrap();
    let uref: Vec<Complex32> = (0..nc).map(|c| { let z = er.eigenvectors[(c, ri)]; Complex32::new(z.re as f32, z.im as f32) }).collect();
    // aligned, eigenvalue-weighted maps z_c = λ s_c e^{−i arg(u_refᴴ s)} and λ on the coarse grid (coil-major)
    let mut zc = vec![Complex32::new(0.0, 0.0); nc * ncv];
    let mut lam = vec![0f32; ncv];
    for (q, (l, s)) in top.iter().enumerate() {
        let phi: Complex32 = (0..nc).map(|c| uref[c].conj() * s[c]).sum();
        let ph = if phi.norm() > 0.0 { phi.conj() / phi.norm() } else { Complex32::new(1.0, 0.0) };
        let lw = l.max(0.0);
        for c in 0..nc {
            zc[c * ncv + q] = s[c] * ph * lw;
        }
        lam[q] = lw;
    }
    // periodic trilinear interpolation to the output grid
    let nvo = nvox(out);
    let mut maps = vec![C32::new(0.0, 0.0); nc * nvo];
    let mut eig = vec![0f32; nvo];
    let corner = |i: usize, a: usize| -> (usize, usize, f32) {
        let f = (i as f64 - (out[a] / 2) as f64) / out[a] as f64;
        let g = f * cd[a] as f64 + half[a] as f64;
        let g0 = g.floor();
        let fr = (g - g0) as f32;
        let i0 = (g0 as i64).rem_euclid(cd[a] as i64) as usize;
        (i0, (i0 + 1) % cd[a], fr)
    };
    let (mx, my, mz) = ((0..out[0]).map(|i| corner(i, 0)).collect::<Vec<_>>(), (0..out[1]).map(|i| corner(i, 1)).collect::<Vec<_>>(), (0..out[2]).map(|i| corner(i, 2)).collect::<Vec<_>>());
    let vals: Vec<(f32, Vec<Complex32>)> = (0..nvo)
        .into_par_iter()
        .map(|q| {
            let (x, y, z) = (q % out[0], (q / out[0]) % out[1], q / (out[0] * out[1]));
            let (cx, cy, cz) = (mx[x], my[y], mz[z]);
            let mut l = 0f32;
            let mut s = vec![Complex32::new(0.0, 0.0); nc];
            for (iz, wz) in [(cz.0, 1.0 - cz.2), (cz.1, cz.2)] {
                for (iy, wy) in [(cy.0, 1.0 - cy.2), (cy.1, cy.2)] {
                    for (ix, wx) in [(cx.0, 1.0 - cx.2), (cx.1, cx.2)] {
                        let wt = wx * wy * wz;
                        if wt == 0.0 {
                            continue;
                        }
                        let qc = ix + cd[0] * (iy + cd[1] * iz);
                        l += wt * lam[qc];
                        for c in 0..nc {
                            s[c] += zc[c * ncv + qc] * wt;
                        }
                    }
                }
            }
            (l, s)
        })
        .collect();
    for (q, (l, s)) in vals.into_iter().enumerate() {
        eig[q] = l;
        let nrm = s.iter().map(|z| z.norm_sqr()).sum::<f32>().sqrt();
        if (l as f64) > o.eig_min && nrm > 0.0 {
            for c in 0..nc {
                let z = s[c] / nrm;
                maps[c * nvo + q] = C32::new(z.re, z.im);
            }
        }
    }
    lap(3, &mut tm);
    EspiritMaps { dims: out, nc, maps, eig, n_kernels: nkeep, sv, coarse: cd, timings: tm }
}

/// Calibration k-space (coil-major nc × cdims, centred, for [`espirit`]) from non-Cartesian data: the samples inside the
/// calibration box (|k_a| < cdims_a / 2 in units of 1/FOV) give low-resolution coil images on a cdims grid of the same
/// FOV by CG on ‖F ρ_c − y_c‖² + λ ‖ρ_c‖² (λ = `lambda_rel` · samples; no density compensation), whose centred FFT is
/// the calibration. `coords`: k·Δ in radians per voxel of the `dims` grid; `y`: coil-major nc × m (whitened /
/// compressed). Returns (calibration, samples used).
#[allow(clippy::too_many_arguments)]
pub fn calib_from_noncart(
    dev: &Device,
    st: &Arc<CudaStream>,
    kern: &Kernels,
    coords: [&[f32]; 3],
    y: &[C32],
    nc: usize,
    dims: [usize; 3],
    cdims: [usize; 3],
    iters: usize,
    lambda_rel: f64,
) -> Result<(Vec<C32>, usize), Error> {
    let m = coords[0].len();
    assert!(y.len() >= nc * m);
    let sc = [0, 1, 2].map(|a| dims[a] as f64 / cdims[a] as f64);
    let lim = std::f64::consts::PI * (1.0 - 1e-6);
    let sel: Vec<usize> = (0..m).filter(|&j| (0..3).all(|a| (coords[a][j] as f64 * sc[a]).abs() < lim)).collect();
    let ms = sel.len();
    if ms == 0 {
        return Err(Error::Format("calib_from_noncart: no samples inside the calibration box".into()));
    }
    let lc: Vec<Vec<f32>> = (0..3).map(|a| sel.iter().map(|&j| (coords[a][j] as f64 * sc[a]) as f32).collect()).collect();
    let ys: Vec<C32> = (0..nc).flat_map(|c| sel.iter().map(move |&j| y[c * m + j])).collect();
    let ncv = nvox(cdims);
    let mut plan = NufftPlan::<f32>::new(dev, &cdims, NufftOpts { tol: 1e-5, sigma: 2.0, ntransf: nc, ..Default::default() })?;
    let dc: Vec<_> = lc.iter().map(|c| st.clone_htod(c)).collect::<Result<Vec<_>, _>>()?;
    plan.set_points(st, &[&dc[0], &dc[1], &dc[2]], ms)?;
    let dy = st.clone_htod(&ys)?;
    let lam = lambda_rel * ms as f64;
    let one = Complex64::new(1.0, 0.0);
    let mut ksp = st.alloc_zeros::<C32>(nc * ms)?;
    // b = Fᴴ y; x = 0; r = b; p = r
    let mut b = st.alloc_zeros::<C32>(nc * ncv)?;
    plan.type1(st, &dy, &mut b, 1)?;
    let mut x = st.alloc_zeros::<C32>(nc * ncv)?;
    let mut r = st.clone_dtod(&b)?;
    let mut p = st.clone_dtod(&b)?;
    let mut ap = st.alloc_zeros::<C32>(nc * ncv)?;
    let mut rr = kern.dot(st, &r, &r)?.re;
    let r0 = rr;
    for _ in 0..iters {
        if rr <= 1e-12 * r0 {
            break;
        }
        plan.type2(st, &p, &mut ksp, -1)?;
        plan.type1(st, &ksp, &mut ap, 1)?;
        kern.axpby(st, Complex64::new(lam, 0.0), &p, one, &mut ap)?;
        let alpha = rr / kern.dot(st, &p, &ap)?.re;
        kern.axpby(st, Complex64::new(alpha, 0.0), &p, one, &mut x)?;
        kern.axpby(st, Complex64::new(-alpha, 0.0), &ap, one, &mut r)?;
        let rn = kern.dot(st, &r, &r)?.re;
        kern.axpby(st, one, &r, Complex64::new(rn / rr, 0.0), &mut p)?;
        rr = rn;
    }
    let img = st.clone_dtoh(&x)?;
    let mut calib = Vec::with_capacity(nc * ncv);
    for c in 0..nc {
        let ic: Vec<Complex32> = img[c * ncv..(c + 1) * ncv].iter().map(|&z| to32(z)).collect();
        calib.extend(centered_fft3(&ic, cdims, false).into_iter().map(|z| C32::new(z.re, z.im)));
    }
    Ok((calib, ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// smooth complex coil maps around the FOV (Gaussian magnitude, linear phase), coil-major
    fn coil_maps(dims: [usize; 3], nc: usize) -> Vec<Complex32> {
        let nv = nvox(dims);
        let mut s = vec![Complex32::new(0.0, 0.0); nc * nv];
        for c in 0..nc {
            let th = 2.0 * std::f32::consts::PI * c as f32 / nc as f32;
            let (cx, cy, cz) = (0.45 * th.cos(), 0.45 * th.sin(), if c % 2 == 0 { 0.2 } else { -0.2 });
            for q in 0..nv {
                let p = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
                let u = [0, 1, 2].map(|a| (p[a] as f32 - (dims[a] / 2) as f32) / dims[a] as f32);
                let r2 = (u[0] - cx).powi(2) + (u[1] - cy).powi(2) + (u[2] - cz).powi(2);
                let mag = (-r2 / 0.18).exp();
                let ph = 1.5 * (u[0] * th.sin() - u[1] * th.cos()) + 0.3 * c as f32;
                s[c * nv + q] = Complex32::from_polar(mag, ph);
            }
        }
        s
    }

    /// textured ellipsoid
    fn object(dims: [usize; 3]) -> Vec<f32> {
        (0..nvox(dims))
            .map(|q| {
                let p = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
                let u = [0, 1, 2].map(|a| (p[a] as f32 - (dims[a] / 2) as f32) / dims[a] as f32);
                let r = (u[0] / 0.36).powi(2) + (u[1] / 0.32).powi(2) + (u[2] / 0.34).powi(2);
                if r < 1.0 { 1.0 + 0.3 * (9.0 * u[0]).sin() * (7.0 * u[1]).cos() + 0.2 * (5.0 * u[2]).sin() } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn centered_fft_matches_the_direct_sum() {
        let dims = [5, 4, 3];
        let x: Vec<Complex32> = (0..60).map(|i| Complex32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos())).collect();
        let f = centered_fft3(&x, dims, false);
        for kq in [0usize, 7, 23, 59] {
            let k = [kq % 5, (kq / 5) % 4, kq / 20].map(|v| v as f64);
            let mut s = Complex64::new(0.0, 0.0);
            for q in 0..60 {
                let r = [q % 5, (q / 5) % 4, q / 20].map(|v| v as f64);
                let ph: f64 = (0..3).map(|a| (k[a] - (dims[a] / 2) as f64) * (r[a] - (dims[a] / 2) as f64) / dims[a] as f64).sum::<f64>() * -2.0 * std::f64::consts::PI;
                s += Complex64::new(x[q].re as f64, x[q].im as f64) * Complex64::from_polar(1.0, ph);
            }
            assert!((Complex64::new(f[kq].re as f64, f[kq].im as f64) - s).norm() < 1e-4, "k {kq}");
        }
    }

    #[test]
    fn recovers_smooth_coil_maps_inside_the_object() {
        let (dims, nc) = ([40usize, 36, 28], 8usize);
        let nv = nvox(dims);
        let s = coil_maps(dims, nc);
        let rho = object(dims);
        let cal = [20usize, 20, 16];
        let mut calib = Vec::with_capacity(nc * nvox(cal));
        for c in 0..nc {
            let img: Vec<Complex32> = (0..nv).map(|q| s[c * nv + q] * rho[q]).collect();
            let k = centered_fft3(&img, dims, false);
            calib.extend(crop_center(&k, dims, cal).into_iter().map(|z| C32::new(z.re, z.im)));
        }
        let t = std::time::Instant::now();
        let m = espirit(&calib, cal, nc, dims, &EspiritOpts { kernel: 5, ..Default::default() });
        let md = espirit(&calib, cal, nc, dims, &EspiritOpts { kernel: 5, dense_eigen: true, ..Default::default() });
        assert_eq!(m.n_kernels, md.n_kernels, "subspace vs dense kernel count");
        let dmax = m.maps.iter().zip(&md.maps).map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt()).fold(0f32, f32::max);
        assert!(dmax < 1e-3, "subspace vs dense maps differ by {dmax}");
        eprintln!("espirit {:.2}s, {} kernels of {}, coarse {:?}", t.elapsed().as_secs_f64(), m.n_kernels, nc * 125, m.coarse);
        // inside the object (away from its edge): |⟨ŝ, s/‖s‖⟩| ≈ 1, eigenvalue ≈ 1; far outside: no map
        let (mut err, mut cnt, mut inside_lo) = (0.0f64, 0usize, 0usize);
        for q in 0..nv {
            let p = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            let u = [0, 1, 2].map(|a| (p[a] as f32 - (dims[a] / 2) as f32) / dims[a] as f32);
            let r = (u[0] / 0.36).powi(2) + (u[1] / 0.32).powi(2) + (u[2] / 0.34).powi(2);
            if r < 0.6 {
                let n: f32 = (0..nc).map(|c| s[c * nv + q].norm_sqr()).sum::<f32>().sqrt();
                let ip: Complex32 = (0..nc).map(|c| { let e = m.maps[c * nv + q]; Complex32::new(e.re, e.im).conj() * s[c * nv + q] / n }).sum();
                err += 1.0 - ip.norm() as f64;
                cnt += 1;
                if m.eig[q] < 0.95 {
                    inside_lo += 1;
                }
            }
        }
        let err = err / cnt as f64;
        eprintln!("mean 1 − |⟨ŝ, s⟩| inside {err:.2e}, {inside_lo} of {cnt} inside voxels with eigenvalue < 0.95");
        assert!(err < 2e-3, "mean 1 − |⟨ŝ, s⟩| inside = {err:.2e}");
        assert!(inside_lo * 50 < cnt, "{inside_lo} of {cnt} inside voxels with eigenvalue < 0.95");
        // a corner of the FOV (no object) gets no map
        let q = 1 + dims[0] * (1 + dims[1] * 1);
        assert!(m.eig[q] < 0.5 && (0..nc).all(|c| m.maps[c * nv + q] == C32::new(0.0, 0.0)), "corner eig {}", m.eig[q]);
        // unit norm where kept
        let k = dims[0] / 2 + dims[0] * (dims[1] / 2 + dims[1] * (dims[2] / 2));
        let n: f32 = (0..nc).map(|c| m.maps[c * nv + k].re.powi(2) + m.maps[c * nv + k].im.powi(2)).sum();
        assert!((n - 1.0).abs() < 1e-4);
    }

    /// production-size timing (16 virtual coils, 6³ kernel, 24³ calibration, 130 × 80 × 44 output): cargo test --release
    /// -p hasty-recon --lib espirit::tests::timing -- --ignored --nocapture
    #[test]
    #[ignore]
    fn timing_production_size() {
        let (dims, nc) = ([130usize, 80, 44], 16usize);
        let nv = nvox(dims);
        let s = coil_maps(dims, nc);
        let rho = object(dims);
        let cal = [24usize, 24, 24];
        let mut calib = Vec::with_capacity(nc * nvox(cal));
        for c in 0..nc {
            let img: Vec<Complex32> = (0..nv).map(|q| s[c * nv + q] * rho[q]).collect();
            calib.extend(crop_center(&centered_fft3(&img, dims, false), dims, cal).into_iter().map(|z| C32::new(z.re, z.im)));
        }
        let t = std::time::Instant::now();
        let m = espirit(&calib, cal, nc, dims, &EspiritOpts::default());
        eprintln!("espirit {:.1}s (gram, eigen, W, voxels: {:?}), {} kernels of {}, coarse {:?}", t.elapsed().as_secs_f64(), m.timings, m.n_kernels, nc * 216, m.coarse);
    }
}
