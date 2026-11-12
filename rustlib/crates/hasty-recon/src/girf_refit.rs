//! Reconstruction-side refit of a few gradient-chain parameters (research/recon_physics_wiring.md item 5).
//!
//! The model chain is a measured / estimated GIRF (hasty_seq `GirfMeasurement`: tables per axis, B0 terms, the
//! resonance list of its parametric fit) with a small parameter vector θ on top:
//!   self term a:   H_a(f; θ) = (1 + s_a) e^{−i2πf δ_a} H_a^est(f) Π_{j on a} R_j(f; f0_j (1 + φ_j), g_j (1 + γ_j)) / R_j(f; f0_j, g_j)
//!   cross a ← b:   (1 + s_a) e^{−i2πf δ_a} H_ab^est(f)      (the delay / gain act on the output axis)
//!   B0 term b:     (1 + β) H0_b^est(f)
//! (R_j = the resonance factor 1 + g·bp(f; f0, ζ) of `Response::Resonant`). θ = 0 is the estimate itself.
//!
//! Sensitivities ∂k/∂θ, ∂φ0/∂θ per ADC sample (the played trajectory's dependence on θ): a delay of output axis a
//! shifts its played gradient, ∂k_a/∂δ_a(t) = −γ [G_a(t) − G_a(t_exc)]; a gain scales it, ∂k_a/∂s_a = k_a(t); the B0
//! amplitude scales the k0 phase, ∂φ0/∂β = φ0(t) — all from the played window of the readout (no extra work). A
//! resonance parameter needs the played response of the derivative chain ∂H/∂θ_p: by linearity one more
//! `PlayedStream` (LTI superposition of precomputed block responses, base axes combined with each readout's rotation
//! on the fly), i.e. "small base-spoke sensitivity tables per base axis".
//!
//! Data Jacobian: y(n, c) = e^{−iφ0_n} [A x](k_n) ⇒ ∂y/∂θ_p = Σ_a ∂_{κ_a} y · ∂κ_a/∂θ_p − i y ∂φ0/∂θ_p, κ = k·Δ the
//! operator's coordinates (rad / voxel, logical axes) and ∂_{κ_a} y = A(−i r_a ⊙ x) (r_a = voxel index − n_a/2: the
//! operator's diagonals P_l, S_c commute with r_a). Per sample, with g = (∂_{κ_0} y, ∂_{κ_1} y, ∂_{κ_2} y, −i y):
//! G(n) = Re Σ_c g_a* g_b (4 × 4) and h(n) = Re Σ_c g_a* r (r = data − model), reduced over coils on the GPU
//! ([`Reducer`]) or the CPU ([`reduce_cpu`]); then H = Σ_n Sᵀ G S, b = Σ_n Sᵀ h with S = the 4 × P sensitivity of the
//! sample: one Gauss–Newton step Δθ = (H + λ diag H)⁻¹ b, the image fixed. Fisher information (complex Gaussian noise
//! of variance σ² per whitened coil sample): F = 2 H / σ², the Cramér–Rao bound F⁻¹.
//!
//! Navigator monitor ([`navigator_fit`]): the same per-sample G, h summed over each TR's navigator samples (the v2
//! rewinder, outside the reconstruction window) give a per-TR fit of a constant k offset (3) and a global phase.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use hasty_mrsim::sequence::{sample_record, SAMPLE_FLOATS};
use hasty_seq::girf::{played_from, Girf, GirfMeasurement, Response, ResonanceSpec};
use hasty_seq::{Mat3, Vec3, GAMMA};
use num_complex::Complex64;

use crate::played::FlowDesign;

/// which parameters are refitted
#[derive(Clone, Copy, Debug)]
pub struct RefitOpts {
    /// per-axis delay of the played gradient (3)
    pub delay: bool,
    /// per-axis gain (3; nearly absorbed by the image scale / an affine image map when the image is free)
    pub gain: bool,
    /// amplitude of the B0 (k0) terms (1)
    pub b0amp: bool,
    /// relative f0 and height of every resonance of the estimate's list (2 per resonance)
    pub res: bool,
}

impl Default for RefitOpts {
    fn default() -> Self {
        Self { delay: true, gain: false, b0amp: true, res: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Param {
    Delay(usize),
    Gain(usize),
    B0Amp,
    /// relative f0 change of resonance j
    ResF0(usize),
    /// relative height change of resonance j
    ResGain(usize),
}

impl Param {
    pub fn name(&self) -> String {
        match self {
            Param::Delay(a) => format!("delay_{}", ["x", "y", "z"][*a]),
            Param::Gain(a) => format!("gain_{}", ["x", "y", "z"][*a]),
            Param::B0Amp => "b0amp".into(),
            Param::ResF0(j) => format!("res{j}_f0"),
            Param::ResGain(j) => format!("res{j}_gain"),
        }
    }
    /// the parameter in natural units: delays in ns, the rest in % (relative)
    pub fn natural(&self, v: f64) -> f64 {
        match self {
            Param::Delay(_) => v * 1e9,
            _ => v * 100.0,
        }
    }
}

/// The estimated chain with the refit parameters (see the module docs).
#[derive(Clone)]
pub struct ChainModel {
    pub meas: GirfMeasurement,
    pub params: Vec<Param>,
    /// tabulation grid (Hz, ascending): the estimate's grids ∪ a fine grid around every resonance ∪ a log grid to 5 MHz
    pub grid: Vec<f64>,
}

impl ChainModel {
    pub fn new(meas: GirfMeasurement, o: RefitOpts) -> Self {
        let mut params = vec![];
        if o.delay {
            params.extend((0..3).map(Param::Delay));
        }
        if o.gain {
            params.extend((0..3).map(Param::Gain));
        }
        if o.b0amp && meas.girf.b0.iter().any(|b| b.is_some()) {
            params.push(Param::B0Amp);
        }
        if o.res {
            for j in 0..meas.resonances.len() {
                params.push(Param::ResF0(j));
                params.push(Param::ResGain(j));
            }
        }
        let mut grid: Vec<f64> = vec![];
        let mut add_tab = |r: &Response| {
            if let Response::Table { f, .. } = r {
                grid.extend_from_slice(f);
            }
        };
        meas.girf.axis.iter().for_each(&mut add_tab);
        meas.girf.cross.iter().for_each(|(_, _, r)| add_tab(r));
        meas.girf.b0.iter().flatten().for_each(&mut add_tab);
        grid.extend((0..=4000).map(|i| i as f64 * 5.0));
        grid.extend((0..=600).map(|i| 2e4 * 250f64.powf(i as f64 / 600.0)));
        for r in &meas.resonances {
            let w = (r.zeta * r.f0).max(1.0);
            grid.extend((-400..=400).map(|i| r.f0 + i as f64 * 0.05 * w));
        }
        grid.retain(|f| f.is_finite() && *f >= 0.0);
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        grid.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        Self { meas, params, grid }
    }

    pub fn n(&self) -> usize {
        self.params.len()
    }

    fn get(&self, th: &[f64], p: Param) -> f64 {
        self.params.iter().position(|q| *q == p).map_or(0.0, |i| th[i])
    }

    fn res_at(&self, th: &[f64], j: usize) -> ResonanceSpec {
        let r = &self.meas.resonances[j];
        ResonanceSpec { axis: r.axis, f0: r.f0 * (1.0 + self.get(th, Param::ResF0(j))), zeta: r.zeta, gain: r.gain * (1.0 + self.get(th, Param::ResGain(j))) }
    }

    /// output-axis factor (1 + s_a) e^{−i2πfδ_a}
    fn out_factor(&self, th: &[f64], a: usize, f: f64) -> Complex64 {
        (1.0 + self.get(th, Param::Gain(a))) * Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f * self.get(th, Param::Delay(a)))
    }

    /// resonance correction of axis a at θ: Π_j R_j(θ) / R_j(0)
    fn res_factor(&self, th: &[f64], a: usize, f: f64) -> Complex64 {
        let mut z = Complex64::new(1.0, 0.0);
        for (j, r) in self.meas.resonances.iter().enumerate() {
            if r.axis == a {
                z *= self.res_at(th, j).factor(f) / r.factor(f);
            }
        }
        z
    }

    fn table(&self, h: impl Fn(f64) -> Complex64) -> Response {
        Response::Table { f: self.grid.clone(), h: self.grid.iter().map(|&f| h(f)).collect() }
    }

    /// the model chain at θ (every term tabulated on [`ChainModel::grid`])
    pub fn chain(&self, th: &[f64]) -> Girf {
        let g = &self.meas.girf;
        let axis: [Response; 3] = [0, 1, 2].map(|a| self.table(|f| g.axis[a].at(f) * self.out_factor(th, a, f) * self.res_factor(th, a, f)));
        let cross = g.cross.iter().map(|(o, i, r)| (*o, *i, self.table(|f| r.at(f) * self.out_factor(th, *o, f)))).collect();
        let b = 1.0 + self.get(th, Param::B0Amp);
        let b0 = [0, 1, 2].map(|a| g.b0[a].as_ref().map(|r| self.table(|f| r.at(f) * b)));
        Girf { axis, cross, b0, tail: g.tail }
    }

    /// the derivative chain ∂H/∂θ_p for a resonance parameter (None for the parameters with analytic sensitivities)
    pub fn dchain(&self, th: &[f64], p: usize) -> Option<Girf> {
        let j = match self.params[p] {
            Param::ResF0(j) | Param::ResGain(j) => j,
            _ => return None,
        };
        let r0 = &self.meas.resonances[j];
        let a = r0.axis;
        let base = &self.meas.girf.axis[a];
        let rj = self.res_at(th, j);
        // ∂R_j/∂θ by a central difference of the analytic factor (θ relative: f0 (1 + φ), g (1 + γ))
        let eps = 1e-5;
        let dr = |f: f64| -> Complex64 {
            let (mut up, mut dn) = (rj.clone(), rj.clone());
            match self.params[p] {
                Param::ResF0(_) => {
                    up.f0 += eps * r0.f0;
                    dn.f0 -= eps * r0.f0;
                }
                _ => {
                    up.gain += eps * r0.gain;
                    dn.gain -= eps * r0.gain;
                }
            }
            (up.factor(f) - dn.factor(f)) / (2.0 * eps)
        };
        let zero = Response::Table { f: vec![0.0, 1e9], h: vec![Complex64::new(0.0, 0.0); 2] };
        let mut axis = [zero.clone(), zero.clone(), zero.clone()];
        axis[a] = self.table(|f| base.at(f) * self.out_factor(th, a, f) * self.res_factor(th, a, f) / rj.factor(f) * dr(f));
        Some(Girf { axis, cross: vec![], b0: [None, None, None], tail: self.meas.girf.tail })
    }
}

/// Per-readout played records and sensitivities of a [`ChainModel`] at θ.
pub struct Sens {
    /// SAMPLE_FLOATS records per sample (the model trajectory at θ), physical axes
    pub rec: Vec<f32>,
    /// per parameter, per sample: (∂k (rad/m, physical axes), ∂φ0 (rad)) per unit θ_p
    pub d: Vec<Vec<[f64; 4]>>,
}

/// The played trajectory of a [`FlowDesign`] through a [`ChainModel`] at θ and its sensitivities, for readouts given
/// by their TR history (`seq[i]` = (encode, rotation) of the previous TRs and the readout, acquisition order) at
/// `times[i]` (s from the TR start). `hist` previous TRs carried.
pub fn sensitivities(model: &ChainModel, th: &[f64], design: &FlowDesign, hist: usize, b0: f64, seqs: &[Vec<(usize, Mat3)>], times: &[Vec<f64>]) -> Vec<Sens> {
    let s0 = design.stream(&model.chain(th), hist);
    let ds: Vec<Option<hasty_seq::girf::PlayedStream>> = hasty_seq::par_map(&(0..model.n()).collect::<Vec<_>>(), |&p| model.dchain(th, p).map(|g| design.stream(&g, hist)));
    let t_exc = design.t_rf;
    let idx: Vec<usize> = (0..seqs.len()).collect();
    hasty_seq::par_map(&idx, |&i| {
        let (g, b0f) = s0.played_tr(&seqs[i]);
        let pl = played_from(&g, &b0f, t_exc, &times[i]);
        let mut rec = Vec::with_capacity(times[i].len() * SAMPLE_FLOATS);
        for (m, &t) in pl.iter().zip(&times[i]) {
            let mut r = sample_record(&m.m0, &m.m1, &m.m2, &m.p, b0, t - t_exc);
            r[15] = m.phi_b0 as f32;
            rec.extend_from_slice(&r);
        }
        let g_exc = g.at(t_exc);
        let d: Vec<Vec<[f64; 4]>> = model
            .params
            .iter()
            .enumerate()
            .map(|(p, par)| match par {
                Param::Delay(a) => times[i].iter().map(|&t| { let mut v = [0.0; 4]; v[*a] = -GAMMA * (g.at(t)[*a] - g_exc[*a]); v }).collect(),
                Param::Gain(a) => pl.iter().map(|m| { let mut v = [0.0; 4]; v[*a] = GAMMA * m.m0[*a]; v }).collect(),
                Param::B0Amp => pl.iter().map(|m| [0.0, 0.0, 0.0, m.phi_b0]).collect(),
                _ => {
                    let (dg, db) = ds[p].as_ref().unwrap().played_tr(&seqs[i]);
                    played_from(&dg, &db, t_exc, &times[i]).iter().map(|m| [GAMMA * m.m0[0], GAMMA * m.m0[1], GAMMA * m.m0[2], m.phi_b0]).collect()
                }
            })
            .collect();
        Sens { rec, d }
    })
}

/// per-sample reduction: G = Re Σ_c g_a* g_b (upper triangle, 10: 00 01 02 03 11 12 13 22 23 33) and h = Re Σ_c g_a* r
/// (4), g = (∂_{κ0} y, ∂_{κ1} y, ∂_{κ2} y, −i y); all arrays coil-major (nc × m)
pub fn reduce_cpu(nc: usize, m: usize, dk: [&[C32]; 3], y: &[C32], r: &[C32]) -> Vec<[f32; 14]> {
    use rayon::prelude::*;
    (0..m)
        .into_par_iter()
        .map(|n| {
            let mut o = [0f64; 14];
            for c in 0..nc {
                let i = c * m + n;
                let yv = y[i];
                let g = [dk[0][i], dk[1][i], dk[2][i], C32::new(yv.im, -yv.re)];
                let rv = r[i];
                let mut q = 0;
                for a in 0..4 {
                    for b in a..4 {
                        o[q] += (g[a].re * g[b].re + g[a].im * g[b].im) as f64;
                        q += 1;
                    }
                    o[10 + a] += (g[a].re * rv.re + g[a].im * rv.im) as f64;
                }
            }
            o.map(|v| v as f32)
        })
        .collect()
}

const RED_SRC: &str = r#"
struct cx { float x, y; };
// per sample n: G (10, upper triangle of Re g* g) and h (4, Re g* r) over the coils, g = (dk0, dk1, dk2, -i y)
extern "C" __global__ void refit_reduce(const cx* __restrict__ d0, const cx* __restrict__ d1, const cx* __restrict__ d2,
                                        const cx* __restrict__ y, const cx* __restrict__ r, float* __restrict__ out, int nc, long long m) {
  long long n = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= m) return;
  float o[14];
  #pragma unroll
  for (int q = 0; q < 14; ++q) o[q] = 0.f;
  for (int c = 0; c < nc; ++c) {
    const long long i = (long long)c * m + n;
    const cx yv = y[i];
    const cx g[4] = {d0[i], d1[i], d2[i], cx{yv.y, -yv.x}};
    const cx rv = r[i];
    int q = 0;
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
      #pragma unroll
      for (int b = a; b < 4; ++b) { o[q] = fmaf(g[a].x, g[b].x, fmaf(g[a].y, g[b].y, o[q])); ++q; }
      o[10 + a] = fmaf(g[a].x, rv.x, fmaf(g[a].y, rv.y, o[10 + a]));
    }
  }
  #pragma unroll
  for (int q = 0; q < 14; ++q) out[n * 14 + q] = o[q];
}
// x_a = -i r_a x (r_a = voxel index - n_a/2): the image whose forward transform is the k-gradient along axis a
extern "C" __global__ void refit_ramp(const cx* __restrict__ x, cx* __restrict__ out, int a, int n0, int n1, int n2) {
  long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long nv = (long long)n0 * n1 * n2;
  if (q >= nv) return;
  const int i = q % n0, j = (q / n0) % n1, k = q / ((long long)n0 * n1);
  const float r = a == 0 ? (float)(i - n0 / 2) : (a == 1 ? (float)(j - n1 / 2) : (float)(k - n2 / 2));
  const cx v = x[q];
  out[q] = cx{r * v.y, -r * v.x};
}
"#;

/// GPU helpers of the refit: the coil reduction and the coordinate ramps r_a ⊙ x.
pub struct Reducer {
    red: CudaFunction,
    ramp: CudaFunction,
}

impl Reducer {
    pub fn new(dev: &Device) -> Self {
        let m = dev.load_module("hasty_girf_refit_v1", RED_SRC, &CompileSettings { options: vec![], no_cache: false }).expect("refit kernels");
        Self { red: m.load_function("refit_reduce").unwrap(), ramp: m.load_function("refit_ramp").unwrap() }
    }

    /// out = −i r_a ⊙ x on a grid `dims` (x fastest)
    pub fn ramp(&self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, a: usize, dims: [usize; 3], out: &mut CudaSlice<C32>) {
        let nv = dims[0] * dims[1] * dims[2];
        let cfg = LaunchConfig { grid_dim: (nv.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (ai, n0, n1, n2) = (a as i32, dims[0] as i32, dims[1] as i32, dims[2] as i32);
        let mut b = st.launch_builder(&self.ramp);
        b.arg(x).arg(out).arg(&ai).arg(&n0).arg(&n1).arg(&n2);
        unsafe { b.launch(cfg) }.expect("refit_ramp");
    }

    /// per-sample reduction (see [`reduce_cpu`]) into `out` (m × 14 floats)
    #[allow(clippy::too_many_arguments)]
    pub fn reduce(&self, st: &Arc<CudaStream>, nc: usize, m: usize, d: [&CudaSlice<C32>; 3], y: &CudaSlice<C32>, r: &CudaSlice<C32>, out: &mut CudaSlice<f32>) {
        let cfg = LaunchConfig { grid_dim: (m.div_ceil(128) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (nci, mi) = (nc as i32, m as i64);
        let mut b = st.launch_builder(&self.red);
        b.arg(d[0]).arg(d[1]).arg(d[2]).arg(y).arg(r).arg(out).arg(&nci).arg(&mi);
        unsafe { b.launch(cfg) }.expect("refit_reduce");
    }
}

/// Normal equations of the Gauss–Newton step (P × P, row-major) and the right-hand side.
#[derive(Clone, Debug)]
pub struct Normal {
    pub p: usize,
    pub h: Vec<f64>,
    pub b: Vec<f64>,
    /// samples accumulated
    pub m: usize,
}

impl Normal {
    pub fn new(p: usize) -> Self {
        Self { p, h: vec![0.0; p * p], b: vec![0.0; p], m: 0 }
    }

    /// add samples: `red[n]` from the reduction, `s[n]` = the 4 × P sensitivity of sample n in operator units
    /// (∂κ_a/∂θ_p in rad / voxel, logical axes; ∂φ0/∂θ_p in rad), stored as s[n][p] = [κ0, κ1, κ2, φ0]
    pub fn add(&mut self, red: &[[f32; 14]], s: &[Vec<[f64; 4]>]) {
        let p = self.p;
        let tri = [[0usize, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]];
        for (n, rv) in red.iter().enumerate() {
            let sn = &s[n];
            // G S (4 × P)
            let mut gs = vec![[0f64; 4]; p];
            for q in 0..p {
                for a in 0..4 {
                    gs[q][a] = (0..4).map(|b| rv[tri[a][b]] as f64 * sn[q][b]).sum();
                }
            }
            for q in 0..p {
                for r in 0..p {
                    self.h[q * p + r] += (0..4).map(|a| sn[q][a] * gs[r][a]).sum::<f64>();
                }
                self.b[q] += (0..4).map(|a| sn[q][a] * rv[10 + a] as f64).sum::<f64>();
            }
        }
        self.m += red.len();
    }

    pub fn merge(&mut self, o: &Normal) {
        self.h.iter_mut().zip(&o.h).for_each(|(a, b)| *a += b);
        self.b.iter_mut().zip(&o.b).for_each(|(a, b)| *a += b);
        self.m += o.m;
    }

    /// Δθ = (H + λ diag H)⁻¹ b (Levenberg–Marquardt damping λ)
    pub fn step(&self, lambda: f64) -> Vec<f64> {
        let p = self.p;
        let mut a = nalgebra::DMatrix::<f64>::from_row_slice(p, p, &self.h);
        for i in 0..p {
            a[(i, i)] *= 1.0 + lambda;
        }
        let b = nalgebra::DVector::from_column_slice(&self.b);
        // scale to unit diagonal (parameters of very different units)
        let d: Vec<f64> = (0..p).map(|i| a[(i, i)].max(1e-300).sqrt()).collect();
        let an = nalgebra::DMatrix::<f64>::from_fn(p, p, |i, j| a[(i, j)] / (d[i] * d[j]));
        let bn = nalgebra::DVector::from_fn(p, |i, _| b[i] / d[i]);
        let x = an.svd(true, true).solve(&bn, 1e-12).expect("normal equations");
        (0..p).map(|i| x[i] / d[i]).collect()
    }

    /// Cramér–Rao bound (standard deviations) for complex Gaussian noise of variance σ² per sample and coil:
    /// F = 2 H / σ²
    pub fn crb(&self, sigma: f64) -> Vec<f64> {
        let p = self.p;
        let a = nalgebra::DMatrix::<f64>::from_row_slice(p, p, &self.h) * (2.0 / (sigma * sigma));
        let d: Vec<f64> = (0..p).map(|i| a[(i, i)].max(1e-300).sqrt()).collect();
        let an = nalgebra::DMatrix::<f64>::from_fn(p, p, |i, j| a[(i, j)] / (d[i] * d[j]));
        match an.try_inverse() {
            Some(inv) => (0..p).map(|i| (inv[(i, i)]).max(0.0).sqrt() / d[i]).collect(),
            None => vec![f64::INFINITY; p],
        }
    }

    /// correlation matrix of the estimate (from H⁻¹)
    pub fn correlation(&self) -> Vec<f64> {
        let p = self.p;
        let a = nalgebra::DMatrix::<f64>::from_row_slice(p, p, &self.h);
        let d: Vec<f64> = (0..p).map(|i| a[(i, i)].max(1e-300).sqrt()).collect();
        let an = nalgebra::DMatrix::<f64>::from_fn(p, p, |i, j| a[(i, j)] / (d[i] * d[j]));
        match an.try_inverse() {
            Some(inv) => {
                let s: Vec<f64> = (0..p).map(|i| inv[(i, i)].max(1e-300).sqrt()).collect();
                (0..p * p).map(|q| inv[(q / p, q % p)] / (s[q / p] * s[q % p])).collect()
            }
            None => vec![f64::NAN; p * p],
        }
    }
}

/// physical sensitivities (rad/m, rad) of a sample → operator units: κ = Rᵀ k · Δ (logical axes, rad / voxel)
pub fn to_operator_units(d: &[f64; 4], rot: &Mat3, vox: [f64; 3]) -> [f64; 4] {
    let l: Vec3 = [0, 1, 2].map(|a| (0..3).map(|b| rot[b][a] * d[b]).sum::<f64>());
    [l[0] * vox[0], l[1] * vox[1], l[2] * vox[2], d[3]]
}

/// Per-TR navigator fit: for every TR (consecutive groups of `per_tr` samples of `red`), the constant k offset
/// (operator units, 3) and global phase (rad) explaining its residual: (Σ G)⁻¹ Σ h over the TR's samples, and the
/// Cramér–Rao standard deviations for noise σ. Returns per TR ([Δκ0, Δκ1, Δκ2, Δφ], [sd …]).
pub fn navigator_fit(red: &[[f32; 14]], per_tr: usize, sigma: f64) -> Vec<([f64; 4], [f64; 4])> {
    let tri = [[0usize, 1, 2, 3], [1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]];
    red.chunks(per_tr)
        .map(|ch| {
            let mut g = nalgebra::Matrix4::<f64>::zeros();
            let mut h = nalgebra::Vector4::<f64>::zeros();
            for rv in ch {
                for a in 0..4 {
                    for b in 0..4 {
                        g[(a, b)] += rv[tri[a][b]] as f64;
                    }
                    h[a] += rv[10 + a] as f64;
                }
            }
            let d: Vec<f64> = (0..4).map(|i| g[(i, i)].max(1e-300).sqrt()).collect();
            let gn = nalgebra::Matrix4::<f64>::from_fn(|i, j| g[(i, j)] / (d[i] * d[j]));
            match gn.try_inverse() {
                Some(inv) => {
                    let x: [f64; 4] = std::array::from_fn(|i| (0..4).map(|j| inv[(i, j)] * h[j] / d[j]).sum::<f64>() / d[i]);
                    let sd: [f64; 4] = std::array::from_fn(|i| (inv[(i, i)] * sigma * sigma / 2.0).max(0.0).sqrt() / d[i]);
                    (x, sd)
                }
                None => ([f64::NAN; 4], [f64::INFINITY; 4]),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meas() -> GirfMeasurement {
        let g = Girf::variants().into_iter().find(|v| v.0 == "resonance").unwrap().1;
        let f: Vec<f64> = (0..=8000).map(|i| i as f64 * 2.5).chain((1..=600).map(|i| 2e4 * 250f64.powf(i as f64 / 600.0))).collect();
        let dir = std::env::temp_dir().join(format!("hasty_refit_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let res = vec![ResonanceSpec { axis: 0, f0: 1400.0, zeta: 0.03, gain: 0.06 }, ResonanceSpec { axis: 1, f0: 1250.0, zeta: 0.03, gain: 0.08 }, ResonanceSpec { axis: 2, f0: 900.0, zeta: 0.03, gain: 0.05 }];
        g.to_measurement(&dir.join("g.json"), &f, &res, "").unwrap();
        let m = GirfMeasurement::load(&dir.join("g.json")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        m
    }

    #[test]
    fn chain_at_zero_is_the_estimate_and_derivatives_match_differences() {
        let cm = ChainModel::new(meas(), RefitOpts { delay: true, gain: true, b0amp: true, res: true });
        assert_eq!(cm.n(), 3 + 3 + 1 + 6);
        let z = vec![0.0; cm.n()];
        let c0 = cm.chain(&z);
        for f in [10.0, 899.0, 1250.0, 5e3, 1e5] {
            for a in 0..3 {
                let (p, q) = (cm.meas.girf.axis[a].at(f), c0.axis[a].at(f));
                assert!((p - q).norm() < 1e-9 * p.norm().max(1e-9), "θ = 0 is not the estimate at {f} Hz");
            }
        }
        // resonance derivative chains vs central differences of the chain
        for (p, par) in cm.params.iter().enumerate() {
            let Some(d) = cm.dchain(&z, p) else { continue };
            let h = 1e-4;
            let mut up = z.clone();
            up[p] = h;
            let mut dn = z.clone();
            dn[p] = -h;
            let (cu, cd) = (cm.chain(&up), cm.chain(&dn));
            for f in [500.0, 880.0, 905.0, 1240.0, 1255.0, 1400.0, 3000.0] {
                for a in 0..3 {
                    let fd = (cu.axis[a].at(f) - cd.axis[a].at(f)) / (2.0 * h);
                    let an = d.axis[a].at(f);
                    assert!((fd - an).norm() < 2e-3 * fd.norm().max(1e-4), "{} axis {a} at {f} Hz: {an} vs {fd}", par.name());
                }
            }
        }
    }

    #[test]
    fn normal_step_solves_a_linear_problem() {
        // samples whose residual is exactly J θ*: one GN step recovers θ*
        let p = 3;
        let mut nrm = Normal::new(p);
        let th = [0.3, -1.2, 0.05];
        let mut red = vec![];
        let mut s = vec![];
        let mut rng = 1u64;
        let mut u = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
        for _ in 0..200 {
            // one coil: g arbitrary, sensitivities arbitrary, r = Σ_p θ_p J_p
            let g: Vec<C32> = (0..4).map(|_| C32::new(u() as f32, u() as f32)).collect();
            let sn: Vec<[f64; 4]> = (0..p).map(|_| [u(), u(), u(), u()]).collect();
            let mut r = C32::new(0.0, 0.0);
            for q in 0..p {
                for a in 0..4 {
                    r.re += (th[q] * sn[q][a]) as f32 * g[a].re;
                    r.im += (th[q] * sn[q][a]) as f32 * g[a].im;
                }
            }
            // g[3] must be −i y: choose y = i g[3]
            let y = C32::new(-g[3].im, g[3].re);
            let one = |v: C32| vec![v];
            red.push(reduce_cpu(1, 1, [&one(g[0]), &one(g[1]), &one(g[2])], &one(y), &one(r))[0]);
            s.push(sn);
        }
        nrm.add(&red, &s);
        let d = nrm.step(0.0);
        for q in 0..p {
            assert!((d[q] - th[q]).abs() < 1e-4, "{d:?}");
        }
    }
}
