//! Temporal subspace: solve for coefficients c with x = T c (T: F frames × K basis functions, the same for every voxel
//! and encode), gradient Tᴴ ∇f(T c). The data term's operator A is untouched: [`Subspace`] wraps any [`Objective`]
//! on frame stacks, so the set-streamed (and later multi-GPU) gradient runs as before on x = T c.
//!
//! Why it can help (research/mathematics_of_paper/reconstruction_operator_section.tex, "Temporal coupling"): a unitary
//! T (K = F) changes nothing (Tᴴ N T has N's spectrum). A truncated basis (K < F) compresses: by Cauchy interlacing
//! the spectrum of Tᴴ N T lies inside N's, the null space of the per-frame operator shrinks, and every coefficient
//! sees the k-space samples of all frames its basis function covers (effective acceleration (K/F) R_frame). It is the
//! constrained version of the cardiac low-pass prior (P²nP with D = TTᴴ projects after each step; here the iterate
//! and the quasi-Newton secant pairs live in range(T)).
//!
//! T and Tᴴ are pointwise in voxels and dense only over frames: per voxel chunk (all frames and coefficients of one
//! encode in cache), out[o] = Σ_i W[o, i] in[i] with the zeros of W skipped (block-diagonal bases: one respiratory bin
//! per block). Memory-bound: one read of the input stack and one write of the output per application, ~1 s at 2 mm on
//! the host, against minutes for the gradient. Host-resident: the sets are streamed to the device(s) by the gradient
//! anyway, so the transform needs no transposition between devices.
//!
//! Layouts: frame stack sets t = f · ne + e (one image of nv voxels each), coefficient stack sets k · ne + e.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use rayon::prelude::*;

use crate::gpu::Kernels;
use crate::hoststack::HostStack;
use crate::p2np::{Objective, Stack};
use crate::Error;

/// A real temporal basis B (nf frames × nk functions, row-major)
#[derive(Clone, Debug)]
pub struct TemporalBasis {
    pub nf: usize,
    pub nk: usize,
    pub b: Vec<f32>,
    pub name: String,
}

impl TemporalBasis {
    /// Orthonormal real cardiac Fourier basis per respiratory bin: 1, √2 cos(2πkc/n), √2 sin(2πkc/n) (k = 1..kmax, the
    /// Nyquist cosine alone when 2 kmax = n), all over √n; block-diagonal over the nr bins (frame f = r · n + c).
    /// kmax ≥ n/2 is the full (unitary) basis.
    pub fn cardiac_fourier(n: usize, nr: usize, kmax: usize) -> Self {
        let tau = 2.0 * std::f64::consts::PI;
        let mut cols: Vec<Vec<f64>> = vec![vec![1.0 / (n as f64).sqrt(); n]];
        for k in 1..=kmax.min(n / 2) {
            let ph = |c: usize| tau * k as f64 * c as f64 / n as f64;
            if 2 * k == n {
                cols.push((0..n).map(|c| ph(c).cos() / (n as f64).sqrt()).collect());
            } else {
                let s = (2.0 / n as f64).sqrt();
                cols.push((0..n).map(|c| s * ph(c).cos()).collect());
                cols.push((0..n).map(|c| s * ph(c).sin()).collect());
            }
        }
        Self::block_diagonal(&cols, n, nr, format!("fourier{kmax}"))
    }

    /// Periodic (cardiac-cyclic) uniform B-splines of degree p with m control points over n frames, per respiratory
    /// bin; not orthonormal (Tᴴ T ≠ I). The velocity model's time basis at the frame centres.
    pub fn periodic_bspline(n: usize, nr: usize, m: usize, p: usize) -> Self {
        assert!(m >= p + 1, "periodic B-spline: m ≥ p + 1 control points");
        let cols: Vec<Vec<f64>> = (0..m)
            .map(|j| {
                (0..n)
                    .map(|c| {
                        // frame centre in control-point units, distance to the knot of j, cyclic
                        let u = c as f64 * m as f64 / n as f64;
                        let mut s = 0.0;
                        for w in -1..=1 {
                            s += cardinal_bspline(u - j as f64 + (w * m as i64) as f64, p);
                        }
                        s
                    })
                    .collect()
            })
            .collect();
        Self::block_diagonal(&cols, n, nr, format!("bspline{m}d{p}"))
    }

    fn block_diagonal(cols: &[Vec<f64>], n: usize, nr: usize, name: String) -> Self {
        let kb = cols.len();
        let (nf, nk) = (n * nr, kb * nr);
        let mut b = vec![0f32; nf * nk];
        for r in 0..nr {
            for (j, col) in cols.iter().enumerate() {
                for c in 0..n {
                    b[(r * n + c) * nk + r * kb + j] = col[c] as f32;
                }
            }
        }
        Self { nf, nk, b, name }
    }

    /// Bᵀ (nk × nf)
    pub fn transpose(&self) -> Vec<f32> {
        let mut t = vec![0f32; self.nk * self.nf];
        for f in 0..self.nf {
            for k in 0..self.nk {
                t[k * self.nf + f] = self.b[f * self.nk + k];
            }
        }
        t
    }

    /// Gram matrix BᵀB (nk × nk, f64)
    pub fn gram(&self) -> Vec<f64> {
        let (nf, nk) = (self.nf, self.nk);
        let mut g = vec![0f64; nk * nk];
        for f in 0..nf {
            for a in 0..nk {
                let ba = self.b[f * nk + a] as f64;
                if ba == 0.0 {
                    continue;
                }
                for c in 0..nk {
                    g[a * nk + c] += ba * self.b[f * nk + c] as f64;
                }
            }
        }
        g
    }

    /// (BᵀB)⁻¹ Bᵀ (nk × nf): the least-squares coefficients of a frame series
    pub fn pinv(&self) -> Vec<f32> {
        let nk = self.nk;
        let g = nalgebra::DMatrix::from_row_slice(nk, nk, &self.gram());
        let gi = g.try_inverse().expect("temporal basis: singular Gram matrix");
        let bt = self.transpose();
        let mut p = vec![0f32; nk * self.nf];
        for k in 0..nk {
            for f in 0..self.nf {
                let mut s = 0.0;
                for j in 0..nk {
                    s += gi[(k, j)] * bt[j * self.nf + f] as f64;
                }
                p[k * self.nf + f] = s as f32;
            }
        }
        p
    }

    /// λ_max(BᵀB): the step bound of the coefficient problem is λ_max(N) λ_max(BᵀB) (1 for an orthonormal basis)
    pub fn gram_max_eig(&self) -> f64 {
        let g = nalgebra::DMatrix::from_row_slice(self.nk, self.nk, &self.gram());
        nalgebra::SymmetricEigen::new(g).eigenvalues.iter().cloned().fold(0.0, f64::max)
    }

    /// ‖B[:, k]‖² per basis function
    pub fn col_norms2(&self) -> Vec<f64> {
        (0..self.nk).map(|k| (0..self.nf).map(|f| (self.b[f * self.nk + k] as f64).powi(2)).sum()).collect()
    }
}

/// cardinal B-spline of degree p, support [0, p + 1)
fn cardinal_bspline(x: f64, p: usize) -> f64 {
    if p == 0 {
        return if (0.0..1.0).contains(&x) { 1.0 } else { 0.0 };
    }
    let pf = p as f64;
    (x * cardinal_bspline(x, p - 1) + (pf + 1.0 - x) * cardinal_bspline(x - 1.0, p - 1)) / pf
}

/// dst[o · ne + e] = Σ_i w[o, i] src[i · ne + e] over all voxels (sets of nv voxels; w: nout × nin row-major, zeros
/// skipped). Parallel over (encode, voxel chunk): every task reads its chunk of the nin input sets and writes its chunk
/// of the nout output sets (disjoint).
pub fn apply_sets(w: &[f32], nout: usize, nin: usize, ne: usize, nv: usize, src: &[C32], dst: &mut [C32]) {
    assert_eq!(w.len(), nout * nin);
    assert_eq!(src.len(), nin * ne * nv, "apply_sets: input stack size");
    assert_eq!(dst.len(), nout * ne * nv, "apply_sets: output stack size");
    let nz: Vec<Vec<(usize, f32)>> = (0..nout).map(|o| (0..nin).filter_map(|i| { let v = w[o * nin + i]; (v != 0.0).then_some((i, v)) }).collect()).collect();
    const CH: usize = 2048;
    let nch = nv.div_ceil(CH);
    let dp = dst.as_mut_ptr() as usize;
    (0..ne * nch).into_par_iter().for_each(|job| {
        let (e, ci) = (job / nch, job % nch);
        let (q0, q1) = (ci * CH, ((ci + 1) * CH).min(nv));
        let len = q1 - q0;
        let mut acc = vec![C32::new(0.0, 0.0); len];
        for (o, row) in nz.iter().enumerate() {
            acc.iter_mut().for_each(|a| *a = C32::new(0.0, 0.0));
            for &(i, wv) in row {
                let s = &src[(i * ne + e) * nv + q0..(i * ne + e) * nv + q1];
                for (a, v) in acc.iter_mut().zip(s) {
                    a.re += wv * v.re;
                    a.im += wv * v.im;
                }
            }
            // SAFETY: (o, e, q0..q1) ranges of different jobs are disjoint
            let d = unsafe { std::slice::from_raw_parts_mut((dp as *mut C32).add((o * ne + e) * nv + q0), len) };
            d.copy_from_slice(&acc);
        }
    });
}

struct UnionFind(Vec<usize>);

impl UnionFind {
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }
    fn find(&mut self, mut a: usize) -> usize {
        while self.0[a] != a {
            self.0[a] = self.0[self.0[a]];
            a = self.0[a];
        }
        a
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra.max(rb)] = ra.min(rb);
        }
    }
}

/// f(T c) on coefficient stacks around an objective on frame stacks (see the module docs)
pub struct Subspace<'a> {
    pub inner: &'a mut dyn Objective,
    pub basis: TemporalBasis,
    pub ne: usize,
    pub nv: usize,
    bt: Vec<f32>,
    norms2: Vec<f64>,
    /// the frame set whose metric stands for coefficient k (the frame where |B[f, k]| is largest)
    rep: Vec<usize>,
    x: HostStack,
    gx: HostStack,
}

impl<'a> Subspace<'a> {
    pub fn new(inner: &'a mut dyn Objective, basis: TemporalBasis, ne: usize) -> Self {
        let lens = inner.lens();
        assert_eq!(lens.len(), basis.nf * ne, "temporal subspace: the frame stack needs nf × ne sets (one frame and encode each)");
        let nv = lens[0];
        assert!(lens.iter().all(|&l| l == nv), "temporal subspace: sets of one image each");
        let bt = basis.transpose();
        let norms2 = basis.col_norms2();
        let rep = (0..basis.nk).map(|k| (0..basis.nf).max_by(|&a, &b| basis.b[a * basis.nk + k].abs().total_cmp(&basis.b[b * basis.nk + k].abs())).unwrap()).collect();
        Self { inner, basis, ne, nv, bt, norms2, rep, x: HostStack::default(), gx: HostStack::default() }
    }

    /// coefficient-stack set lengths
    pub fn coeff_lens(&self) -> Vec<usize> {
        vec![self.nv; self.basis.nk * self.ne]
    }

    /// x = T c (frame stack from a coefficient stack)
    pub fn synth(&self, c: &[C32], x: &mut [C32]) {
        apply_sets(&self.basis.b, self.basis.nf, self.basis.nk, self.ne, self.nv, c, x);
    }

    /// c = (TᵀT)⁻¹ Tᵀ x: the least-squares coefficients of a frame stack (the projection of a start image)
    pub fn coeffs(&self, x: &[C32], c: &mut [C32]) {
        apply_sets(&self.basis.pinv(), self.basis.nk, self.basis.nf, self.ne, self.nv, x, c);
    }

    fn scratch(&mut self) {
        if self.x.lens.is_empty() {
            let l = self.inner.lens();
            self.x = HostStack::zeros(&l);
            self.gx = HostStack::zeros(&l);
        }
    }
}

impl Objective for Subspace<'_> {
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn lens(&self) -> Vec<usize> {
        self.coeff_lens()
    }
    fn begin_iteration(&mut self, k: usize, iters: usize) {
        self.inner.begin_iteration(k, iters);
    }
    fn curvature_pair_ok(&self) -> bool {
        self.inner.curvature_pair_ok()
    }
    /// coefficient sets coupled through a shared frame (TᴴNT is not block-diagonal over k), or through the inner
    /// objective's own groups of frame sets: connected components of the bipartite graph coefficient set (k, e) –
    /// frame set (f, e) with B[f, k] ≠ 0, plus the edges between frame sets of one inner group
    fn secant_groups(&self) -> Option<Vec<usize>> {
        let (nf, nk, ne) = (self.basis.nf, self.basis.nk, self.ne);
        let nc = nk * ne;
        let mut uf = UnionFind::new(nc + nf * ne);
        for k in 0..nk {
            for f in 0..nf {
                if self.basis.b[f * nk + k] != 0.0 {
                    for e in 0..ne {
                        uf.union(k * ne + e, nc + f * ne + e);
                    }
                }
            }
        }
        if let Some(gi) = self.inner.secant_groups() {
            let mut first = std::collections::HashMap::new();
            for (t, &g) in gi.iter().enumerate() {
                let r = *first.entry(g).or_insert(t);
                uf.union(nc + r, nc + t);
            }
        }
        let mut id = std::collections::HashMap::new();
        Some((0..nc).map(|t| { let r = uf.find(t); let n = id.len(); *id.entry(r).or_insert(n) }).collect())
    }
    fn gradient_host(&mut self, st: &Arc<CudaStream>, c: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        self.scratch();
        let (nf, nk, ne, nv) = (self.basis.nf, self.basis.nk, self.ne, self.nv);
        apply_sets(&self.basis.b, nf, nk, ne, nv, &c.data, &mut self.x.data);
        let f = self.inner.gradient_host(st, &self.x, &mut self.gx)?;
        apply_sets(&self.bt, nk, nf, ne, nv, &self.gx.data, &mut g.data);
        Ok(f)
    }
    fn gradient(&mut self, st: &Arc<CudaStream>, c: &Stack, g: &mut Stack) -> Result<f64, Error> {
        let lc = self.coeff_lens();
        let mut ch = HostStack::zeros(&lc);
        for t in 0..lc.len() {
            st.memcpy_dtoh(&c.v[t], ch.chunk_mut(t))?;
        }
        let mut gh = HostStack::zeros(&lc);
        let f = self.gradient_host(st, &ch, &mut gh)?;
        for t in 0..lc.len() {
            st.memcpy_htod(gh.chunk(t), &mut g.v[t])?;
        }
        Ok(f)
    }
    /// coefficient k of encode e: the metric of the frame set where B[:, k] peaks, divided by ‖B[:, k]‖² (exact for
    /// TᴴNT when the frames share one normal operator and the basis is orthogonal)
    fn metric_host(&mut self, st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inverse: bool) -> Result<(), Error> {
        let (k, e) = (t / self.ne, t % self.ne);
        self.inner.metric_host(st, self.rep[k] * self.ne + e, r, z, inverse)?;
        let s = if inverse { self.norms2[k] } else { 1.0 / self.norms2[k] } as f32;
        if s != 1.0 {
            z.par_iter_mut().for_each(|v| *v = C32::new(s * v.re, s * v.im));
        }
        Ok(())
    }
    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        let rh = st.clone_dtoh(r)?;
        let mut zh = vec![C32::new(0.0, 0.0); rh.len()];
        self.metric_host(st, t, &rh, &mut zh, inverse)?;
        st.memcpy_htod(&zh, z)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn fourier_is_orthonormal_and_full_basis_is_unitary() {
        for (n, kmax) in [(20usize, 4usize), (20, 10), (7, 3)] {
            let b = TemporalBasis::cardiac_fourier(n, 2, kmax);
            let g = b.gram();
            for a in 0..b.nk {
                for c in 0..b.nk {
                    assert!(close(g[a * b.nk + c], if a == c { 1.0 } else { 0.0 }, 1e-6), "n {n} kmax {kmax}: G[{a},{c}] = {}", g[a * b.nk + c]);
                }
            }
            if 2 * kmax >= n {
                assert_eq!(b.nk, b.nf, "full basis: square");
            }
        }
        assert_eq!(TemporalBasis::cardiac_fourier(20, 2, 4).nk, 18);
    }

    #[test]
    fn bspline_partition_of_unity_and_pinv() {
        let b = TemporalBasis::periodic_bspline(20, 2, 8, 3);
        for f in 0..b.nf {
            let s: f64 = (0..b.nk).map(|k| b.b[f * b.nk + k] as f64).sum();
            assert!(close(s, 1.0, 1e-5), "frame {f}: Σ = {s}");
        }
        let p = b.pinv();
        // pinv · B = I
        for a in 0..b.nk {
            for c in 0..b.nk {
                let s: f64 = (0..b.nf).map(|f| p[a * b.nf + f] as f64 * b.b[f * b.nk + c] as f64).sum();
                assert!(close(s, if a == c { 1.0 } else { 0.0 }, 1e-4), "pinv B [{a},{c}] = {s}");
            }
        }
    }

    #[test]
    fn apply_sets_matches_dense_and_adjoint() {
        let (ne, nv) = (3usize, 5000usize);
        let b = TemporalBasis::cardiac_fourier(10, 2, 2);
        let mut s = 12345u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 20001) as f32 / 10000.0 - 1.0
        };
        let c: Vec<C32> = (0..b.nk * ne * nv).map(|_| C32::new(rnd(), rnd())).collect();
        let y: Vec<C32> = (0..b.nf * ne * nv).map(|_| C32::new(rnd(), rnd())).collect();
        let mut x = vec![C32::new(0.0, 0.0); b.nf * ne * nv];
        apply_sets(&b.b, b.nf, b.nk, ne, nv, &c, &mut x);
        // dense reference at a few voxels
        for &(e, q) in &[(0usize, 0usize), (1, 2047), (2, 2048), (2, 4999)] {
            for f in 0..b.nf {
                let mut r = C32::new(0.0, 0.0);
                for k in 0..b.nk {
                    let w = b.b[f * b.nk + k];
                    let v = c[(k * ne + e) * nv + q];
                    r = C32::new(r.re + w * v.re, r.im + w * v.im);
                }
                let got = x[(f * ne + e) * nv + q];
                assert!((got.re - r.re).abs() < 1e-5 && (got.im - r.im).abs() < 1e-5, "x[{f},{e},{q}]");
            }
        }
        // ⟨y, T c⟩ = ⟨Tᵀ y, c⟩
        let mut ty = vec![C32::new(0.0, 0.0); b.nk * ne * nv];
        apply_sets(&b.transpose(), b.nk, b.nf, ne, nv, &y, &mut ty);
        let dot = |a: &[C32], b: &[C32]| a.iter().zip(b).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum::<f64>();
        let (l, r) = (dot(&y, &x), dot(&ty, &c));
        assert!((l - r).abs() < 1e-6 * l.abs().max(1.0), "adjoint: {l} vs {r}");
    }

    #[test]
    fn full_fourier_round_trip() {
        let (ne, nv) = (2usize, 300usize);
        let b = TemporalBasis::cardiac_fourier(8, 1, 4);
        assert_eq!(b.nk, 8);
        let x: Vec<C32> = (0..b.nf * ne * nv).map(|i| C32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos())).collect();
        let mut c = vec![C32::new(0.0, 0.0); b.nk * ne * nv];
        apply_sets(&b.pinv(), b.nk, b.nf, ne, nv, &x, &mut c);
        let mut x2 = vec![C32::new(0.0, 0.0); x.len()];
        apply_sets(&b.b, b.nf, b.nk, ne, nv, &c, &mut x2);
        let err = x.iter().zip(&x2).map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs())).fold(0f32, f32::max);
        assert!(err < 1e-5, "round trip {err}");
    }
}
