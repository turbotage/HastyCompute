//! Knot-space solver pieces for temporal / respiratory B-spline bases (research/temporal_operator.md §16–§19).
//!
//! **Kronecker preconditioner.** For a knot stack c = (c_κ), κ = (respiratory knot, cardiac knot), the normal operator
//! is AᴴA = Σ_j w_κ(j) w_κ'(j) a_jᴴ a_j. If the k-space pattern of the samples near any time is statistically the same
//! (interleaved radial / spiral views, as in our sequences), it factors approximately as
//!   AᴴA ≈ G ⊗ (A_sᴴ A_s),   G_κκ' = mean_j w_κ(j) w_κ'(j)   (the temporal Gram of the basis under the sample times),
//! with A_s the static operator of all samples. The preconditioner is P = G⁻¹ ⊗ C⁻¹ with C⁻¹ ≈ (A_sᴴA_s)⁻¹ the circulant
//! metric. G is K × K (K ≤ 127), inverted on the host with a small ridge. It equalises the knots: sparsely sampled
//! knots and high-degree bases (whose G is ill-conditioned, §13) no longer stall CG.
//!
//! **Held-out split.** `heldout_mask` marks a deterministic, evenly spread 1/n of the readouts (by a hash of the
//! readout index, so every breathing / cardiac / encode stratum loses the same fraction) to keep out of the fit. Their
//! residual ‖A x − y‖ on the held-out samples measures prediction error without ground truth: early stopping, basis
//! choice and the acceptance of model components (motion, B0, …: research/model_robustness.md).
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{Device, C32};

use crate::cg::Precond;
use crate::precond::CirculantPrecond;
use crate::Error;

/// Temporal Gram G (K × K, row-major) of per-sample basis weights given sparsely: for every sample its nonzero
/// (knot, weight) pairs. Mean over the samples.
pub fn temporal_gram(k: usize, weights: impl Iterator<Item = Vec<(usize, f64)>>) -> Vec<f64> {
    let mut g = vec![0.0f64; k * k];
    let mut n = 0usize;
    for w in weights {
        for &(a, wa) in &w {
            for &(b, wb) in &w {
                g[a * k + b] += wa * wb;
            }
        }
        n += 1;
    }
    g.iter_mut().for_each(|v| *v /= n.max(1) as f64);
    g
}

/// inverse of a symmetric positive semi-definite K × K matrix with ridge `eps` × mean diagonal
pub fn gram_inverse(g: &[f64], k: usize, eps: f64) -> Vec<f64> {
    let md = (0..k).map(|i| g[i * k + i]).sum::<f64>() / k as f64;
    let m = nalgebra::DMatrix::<f64>::from_fn(k, k, |i, j| g[i * k + j] + if i == j { eps * md } else { 0.0 });
    let inv = m.try_inverse().expect("temporal Gram not invertible");
    (0..k * k).map(|q| inv[(q / k, q % k)]).collect()
}

/// condition number of the (ridged) Gram, for logs
pub fn gram_condition(g: &[f64], k: usize) -> f64 {
    let m = nalgebra::DMatrix::<f64>::from_fn(k, k, |i, j| g[i * k + j]);
    let e = m.symmetric_eigen().eigenvalues;
    let (lo, hi) = e.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
    hi / lo.max(1e-300)
}

const SRC: &str = r#"
struct cx { float x, y; };
// in place: z[a][v] ← Σ_b Ginv[a][b] z[b][v] (knot mixing per voxel, real K × K matrix); a block holds VB voxels'
// K values in shared memory (K ≤ 127)
#define VB 16
extern "C" __global__ void knot_mix(cx* __restrict__ z, const float* __restrict__ ginv, int k, unsigned long long nv) {
    __shared__ cx t[128 * VB];
    const unsigned long long v0 = (unsigned long long)blockIdx.x * VB;
    for (int q = threadIdx.x; q < k * VB; q += blockDim.x) {
        const int b = q / VB, v = q % VB;
        t[q] = v0 + v < nv ? z[(unsigned long long)b * nv + v0 + v] : cx{0.f, 0.f};
    }
    __syncthreads();
    for (int q = threadIdx.x; q < k * VB; q += blockDim.x) {
        const int a = q / VB, v = q % VB;
        if (v0 + v >= nv) continue;
        float zr = 0.f, zi = 0.f;
        for (int b = 0; b < k; ++b) {
            const float g = ginv[a * k + b];
            const cx w = t[b * VB + v];
            zr = fmaf(g, w.x, zr); zi = fmaf(g, w.y, zi);
        }
        z[(unsigned long long)a * nv + v0 + v] = cx{zr, zi};
    }
}
"#;

/// P = G⁻¹ ⊗ C⁻¹ on knot stacks [K][nv] (see the module docs); `circ` = None: G⁻¹ ⊗ I
pub struct KnotPrecond {
    k: usize,
    nv: usize,
    ginv: CudaSlice<f32>,
    circ: Option<CirculantPrecond>,
    a: CudaSlice<C32>,
    b: CudaSlice<C32>,
    mix: CudaFunction,
}

impl KnotPrecond {
    pub fn new(dev: &Device, st: &Arc<CudaStream>, k: usize, nv: usize, ginv: &[f64], circ: Option<CirculantPrecond>) -> Result<Self, Error> {
        assert_eq!(ginv.len(), k * k);
        let mix = dev.load_function("knotsolve", SRC, "knot_mix", &hasty_cuda::CompileSettings::default())?;
        Ok(Self {
            k,
            nv,
            ginv: st.clone_htod(&ginv.iter().map(|&v| v as f32).collect::<Vec<_>>())?,
            circ,
            a: st.alloc_zeros(nv)?,
            b: st.alloc_zeros(nv)?,
            mix,
        })
    }

    /// device bytes it holds (the ledger's preconditioner row): two images (+ the metric's own); no knot-stack scratch
    pub fn bytes(k: usize, nv: usize) -> usize {
        2 * nv * 8 + k * k * 4
    }
}

impl Precond for KnotPrecond {
    fn apply(&mut self, st: &Arc<CudaStream>, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (k, nv) = (self.k, self.nv);
        assert!(k <= 128, "knot mixing holds ≤ 128 knots per voxel");
        // per-knot circulant inverse straight into z, then the in-place knot mixing
        for q in 0..k {
            match self.circ.as_mut() {
                Some(c) => {
                    st.memcpy_dtod(&r.slice(q * nv..(q + 1) * nv), &mut self.a)?;
                    c.apply(st, &self.a, &mut self.b)?;
                    st.memcpy_dtod(&self.b, &mut z.slice_mut(q * nv..(q + 1) * nv))?;
                }
                None => st.memcpy_dtod(&r.slice(q * nv..(q + 1) * nv), &mut z.slice_mut(q * nv..(q + 1) * nv))?,
            }
        }
        let (ki, nvu) = (k as i32, nv as u64);
        unsafe {
            st.launch_builder(&self.mix)
                .arg(z)
                .arg(&self.ginv)
                .arg(&ki)
                .arg(&nvu)
                .launch(LaunchConfig { grid_dim: (nv.div_ceil(16) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;
        }
        Ok(())
    }
}

/// held-out readouts: a deterministic, evenly spread 1 / n of them (splitmix hash of the readout index)
pub fn heldout_mask(n_readouts: usize, n: usize) -> Vec<bool> {
    (0..n_readouts)
        .map(|i| {
            let mut z = (i as u64).wrapping_add(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            z % n as u64 == 0
        })
        .collect()
}

/// a deterministic, evenly spread subset of 1 / n of the readouts (splitmix hash of the readout index with `salt`, so
/// it is independent of [`heldout_mask`]): shorter scans with every breathing / cardiac / encode stratum thinned alike
pub fn subset_mask(n_readouts: usize, n: usize, salt: u64) -> Vec<bool> {
    (0..n_readouts)
        .map(|i| {
            let mut z = (i as u64 ^ salt.wrapping_mul(0xD1B54A32D192ED03)).wrapping_add(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            n <= 1 || z % n as u64 == 0
        })
        .collect()
}
