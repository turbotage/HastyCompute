//! Per-axis static / dynamic lock on respiratory × cardiac knot bases (research/static_lock.md §7).
//!
//! The knot signal of a voxel is x(t) = Σ_{k,b} ψ_k(r(t)) φ_b(c(t)) C[k, b] (respiratory knot k, cardiac knot b, image
//! κ = k · mc + b as in the 2-D knot engine). Both bases are partitions of unity, so "constant along an axis" is
//! "equal coefficients along that axis":
//! - S_c (static over the cardiac cycle, moves with breathing): C[k, b] = a_k;
//! - S_r (static over breathing, moves with the heart): C[k, b] = d_b;
//! - S_0 = S_c ∩ S_r (fully static): C[k, b] = m.
//!
//! **Metric.** The data term's Hessian on a knot stack is ≈ G ⊗ AᴴA_static with G the temporal Gram under the sample
//! times (knotsolve), the metric of the knot-preconditioned solver. The lock is a prox in the separable approximation
//! G_s = G_r ⊗ G_c, with G_r, G_c the exact marginals of G (partial sums over the other axis: partition of unity).
//! In G_s the three projections are weighted coefficient averages with the knots' **sample shares** ω = G1 (the mean
//! basis weight of each knot over the samples), and they commute, so the four pieces
//!   m = P_0 C, R = (P_c − P_0) C (breathing only), Cd = (P_r − P_0) C (heart only), I = C − P_c C − P_r C + P_0 C
//! are G_s-orthogonal. The share-weighted mean is also the minimum-variance (BLUE) static value when the coefficient
//! noise has covariance ∝ G⁻¹ (least squares / preconditioned iterates). The plain coefficient average is the special
//! case of equal shares (uniform periodic cardiac knots). Linear respiratory knots at bellows quantiles do not have
//! equal shares (sim_flow_10min: 0.091 0.217 0.192 0.195 0.256 0.049), and there the plain average is an oblique
//! projection with G-operator norm 1.58 — not non-expansive. G_s differs from G by the breathing / heart-phase
//! dependence of the sampling (5 % there; `nonsep`, `metric_bound`); [`KnotBasis::gs_inverse`] in the knot
//! preconditioner removes the mismatch.
//!
//! **Lock** (prox of a penalty diagonal in the four pieces, firmly non-expansive in the G_s norm):
//! - quadratic: R ← R / (1 + μ w_r), Cd ← Cd / (1 + μ w_c), I ← I / (1 + μ (w_c + w_x));
//! - group: block soft thresholds in the G_s norm, τ = μ w s √n (s the local noise per degree of freedom, n the
//!   piece's noise dimension): a static voxel goes exactly to its mean, a moving one keeps its motion minus τ.
//! - `Coupling::Nested` reproduces `tlock` (cardiac lock on the whole cardiac deviation I + Cd, respiratory lock on the
//!   cardiac means only; groups per bin when G_r is diagonal); `Additive` (default) also locks the interaction I
//!   with the respiratory weight w_x, so "static over breathing" means the whole cardiac curve is the same in every
//!   breathing state.
//! The mean m is never changed: steady flow (mean velocity) is never locked to zero.
//!
//! **Map** (no labels), as `tlock` but on the knot coefficients with exact normalisations: piece energies in the G_s
//! norm divided by their expected noise energy s² n (n = tr(G_s Π M), M the coefficient noise covariance model:
//! G_s⁻¹ (`NoiseModel::Ls`), ∝ I (`White`) or G_s⁻¹ + t I with t fitted from the data (`Auto`)), noise s² from
//! cardiac second differences of the coefficients divided by Σ dᵀ M d, block medians, blur gains, detection power,
//! flow criterion, dilation. The colour matters: for 16 cubic cardiac knots the second differences of LS-coloured
//! noise are 5× stronger than of white noise of the same scores' level, a 2.2× error in s if the model is wrong.
//!
//! Velocity model (z_κ complex, v_κ three reals per voxel): the map is estimated from the pseudo-images
//! z_κ exp(i φ_e(v_κ)) (streamed per encode), the z lock is the image lock with z's own noise level, the v lock acts on
//! the three components jointly (group norm over components) with v's own noise level. The static anchor v ≈ 0 stays
//! in `vobj`.
use std::ops::{Add, Mul, Sub};

use hasty_cuda::C32;
use rayon::prelude::*;

use crate::flow::FlowDecoder;
use crate::tlock::{self, cf, cf_mut, FrameLayout, LockOpts};

type Cf = num_complex::Complex<f32>;

/// coefficient noise covariance model of the iterate (normalisation of the map's scores)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseModel {
    /// Σ ∝ G_s⁻¹: least-squares / knot-preconditioned iterates (data noise and aliasing through (G ⊗ AᴴA)⁻¹)
    Ls,
    /// Σ ∝ I: white coefficients
    White,
    /// Σ ∝ G_s⁻¹ + t I with t fitted per coefficient set from the data: the ratio of the interaction piece's energy
    /// to the second differences' (both pure noise in static voxels, the majority) picks t (needs kr ≥ 2, mc ≥ 3;
    /// else Ls). Noise smoother in time than white (e.g. after a temporal low-rank prior) clamps at White: the scores
    /// are then inflated, which locks less (the safe side).
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coupling {
    /// `tlock`'s decomposition: cardiac lock on (I − P_c), respiratory lock on (P_c − P_0)
    Nested,
    /// four pieces; the interaction is locked by w_c + w_x
    Additive,
}

#[derive(Clone, Debug)]
pub struct KnotLockOpts {
    /// μ, group, blur, score ramp, dilation, v_min, noise blocks, detection power: as `tlock`
    pub base: LockOpts,
    pub coupling: Coupling,
}

impl Default for KnotLockOpts {
    fn default() -> Self {
        Self { base: LockOpts::default(), coupling: Coupling::Additive }
    }
}

/// noise dimensions of the pieces under one noise model (n(Π) = tr(G_s Π M), s² units)
#[derive(Clone, Debug, Default)]
pub struct Norms {
    /// time-averaged noise of the evaluated signal: tr(G_s M)
    pub k_eff: f64,
    pub n0: f64,
    pub n_pc: f64,
    pub n_rm: f64,
    pub n_co: f64,
    pub n_int: f64,
    pub n_card: f64,
    /// per respiratory knot: E_k (I − P_c) (nested groups per bin, diagonal G_r)
    pub n_bin: Vec<f64>,
    /// Σ over the cardiac second-difference terms of (e_k ⊗ d_b)ᵀ M (e_k ⊗ d_b) (0: fewer than 3 cardiac knots)
    pub n_d: f64,
}

/// the respiratory × cardiac knot basis seen by the lock: marginal Grams, shares, noise normalisations
#[derive(Clone, Debug)]
pub struct KnotBasis {
    pub kr: usize,
    pub mc: usize,
    /// marginal Grams, normalised to 1ᵀG1 = 1 (row-major)
    pub g_r: Vec<f64>,
    pub g_c: Vec<f64>,
    /// sample shares ω = G 1 (sum 1)
    pub om_r: Vec<f64>,
    pub om_c: Vec<f64>,
    /// the cardiac axis wraps (periodic B-splines) for the second differences; false = `tlock`'s frame convention
    pub periodic: bool,
    pub noise: NoiseModel,
    /// ‖G − G_r ⊗ G_c‖_F / ‖G‖_F of the full Gram (0 when built from marginals)
    pub nonsep: f64,
    /// √(λ_max / λ_min) of G_s⁻¹ G: bound on the G-norm Lipschitz constant of a G_s-non-expansive map (1 = separable)
    pub metric_bound: f64,
    pub norms_ls: Norms,
    pub norms_white: Norms,
    diag_r: bool,
    nz_r: Vec<Vec<(usize, f32)>>,
    nz_c: Vec<Vec<(usize, f32)>>,
    om_r32: Vec<f32>,
    om_c32: Vec<f32>,
}

fn sym_inverse(g: &[f64], n: usize, ridge: f64) -> Vec<f64> {
    let md = (0..n).map(|i| g[i * n + i]).sum::<f64>() / n as f64;
    let m = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| g[i * n + j] + if i == j { ridge * md } else { 0.0 });
    let inv = m.try_inverse().expect("knot Gram not invertible");
    (0..n * n).map(|q| inv[(q / n, q % n)]).collect()
}

fn kron(a: &[f64], na: usize, b: &[f64], nb: usize) -> Vec<f64> {
    let n = na * nb;
    let mut o = vec![0.0; n * n];
    for i in 0..na {
        for j in 0..na {
            for p in 0..nb {
                for q in 0..nb {
                    o[(i * nb + p) * n + j * nb + q] = a[i * na + j] * b[p * nb + q];
                }
            }
        }
    }
    o
}

impl KnotBasis {
    /// from the full K × K temporal Gram (knotsolve::temporal_gram, κ = k · mc + b): marginals by partial sums
    pub fn from_gram(g: &[f64], kr: usize, mc: usize, periodic: bool, noise: NoiseModel) -> Self {
        let k = kr * mc;
        assert_eq!(g.len(), k * k);
        let tot: f64 = g.iter().sum();
        assert!(tot > 0.0, "empty Gram");
        let mut g_r = vec![0.0; kr * kr];
        let mut g_c = vec![0.0; mc * mc];
        for a in 0..k {
            for b in 0..k {
                let v = g[a * k + b] / tot;
                g_r[(a / mc) * kr + b / mc] += v;
                g_c[(a % mc) * mc + b % mc] += v;
            }
        }
        let mut s = Self::from_marginals(&g_r, &g_c, kr, mc, periodic, noise);
        let gs = kron(&s.g_r, kr, &s.g_c, mc);
        let gn: Vec<f64> = g.iter().map(|v| v / tot).collect();
        let num: f64 = gn.iter().zip(&gs).map(|(a, b)| (a - b).powi(2)).sum();
        let den: f64 = gn.iter().map(|a| a * a).sum();
        s.nonsep = (num / den).sqrt();
        // generalised eigenvalues of (G, G_s): L⁻¹ G L⁻ᵀ with G_s = L Lᵀ (ridged)
        let md = (0..k).map(|i| gs[i * k + i]).sum::<f64>() / k as f64;
        let gsm = nalgebra::DMatrix::<f64>::from_fn(k, k, |i, j| gs[i * k + j] + if i == j { 1e-9 * md } else { 0.0 });
        let gm = nalgebra::DMatrix::<f64>::from_fn(k, k, |i, j| gn[i * k + j] + if i == j { 1e-9 * md } else { 0.0 });
        if let Some(ch) = gsm.cholesky() {
            let l = ch.l();
            let li = l.clone().try_inverse().unwrap();
            let a = &li * gm * li.transpose();
            let a = (&a + a.transpose()) * 0.5;
            let e = a.symmetric_eigen().eigenvalues;
            let (lo, hi) = e.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
            s.metric_bound = (hi / lo.max(1e-300)).sqrt();
        }
        s
    }

    /// from marginal Grams (row-major kr × kr and mc × mc; any positive scale)
    pub fn from_marginals(g_r: &[f64], g_c: &[f64], kr: usize, mc: usize, periodic: bool, noise: NoiseModel) -> Self {
        assert!(kr >= 1 && mc >= 1 && kr * mc <= 128, "knot lock holds ≤ 128 knots");
        let nrm = |g: &[f64]| -> Vec<f64> {
            let t: f64 = g.iter().sum();
            g.iter().map(|v| v / t).collect()
        };
        let (g_r, g_c) = (nrm(g_r), nrm(g_c));
        let share = |g: &[f64], n: usize| -> Vec<f64> { (0..n).map(|i| (0..n).map(|j| g[i * n + j]).sum()).collect() };
        let (om_r, om_c) = (share(&g_r, kr), share(&g_c, mc));
        let sparse = |g: &[f64], n: usize| -> Vec<Vec<(usize, f32)>> {
            let mx = g.iter().fold(0f64, |a, &b| a.max(b.abs()));
            (0..n).map(|i| (0..n).filter(|&j| g[i * n + j].abs() > 1e-12 * mx).map(|j| (j, g[i * n + j] as f32)).collect()).collect()
        };
        let diag_r = (0..kr).all(|i| (0..kr).all(|j| i == j || g_r[i * kr + j] == 0.0));
        let mut s = Self {
            kr,
            mc,
            nz_r: sparse(&g_r, kr),
            nz_c: sparse(&g_c, mc),
            om_r32: om_r.iter().map(|&v| v as f32).collect(),
            om_c32: om_c.iter().map(|&v| v as f32).collect(),
            g_r,
            g_c,
            om_r,
            om_c,
            periodic,
            noise,
            nonsep: 0.0,
            metric_bound: 1.0,
            norms_ls: Norms::default(),
            norms_white: Norms::default(),
            diag_r,
        };
        s.norms_ls = s.norms_for(NoiseModel::Ls);
        s.norms_white = s.norms_for(NoiseModel::White);
        s
    }

    /// `tlock`'s frame stacks: nr bins × nc frames, equal shares (identity Gram), no cardiac wrap
    pub fn frames(nr: usize, nc: usize) -> Self {
        let eye = |n: usize| -> Vec<f64> { (0..n * n).map(|q| (q / n == q % n) as u8 as f64).collect() };
        Self::from_marginals(&eye(nr), &eye(nc), nr, nc, false, NoiseModel::Ls)
    }

    pub fn k(&self) -> usize {
        self.kr * self.mc
    }

    /// the separable Gram G_s = G_r ⊗ G_c (K × K, row-major, κ = k · mc + b)
    pub fn gs(&self) -> Vec<f64> {
        kron(&self.g_r, self.kr, &self.g_c, self.mc)
    }

    /// G_s⁻¹ = G_r⁻¹ ⊗ G_c⁻¹ (each factor ridged by `eps` × its mean diagonal): pass it to
    /// `knotsolve::KnotPrecond::new` instead of the full G⁻¹ to make the solver metric G_s ⊗ C, in which the lock is
    /// exactly firmly non-expansive (temporal part)
    pub fn gs_inverse(&self, eps: f64) -> Vec<f64> {
        kron(&sym_inverse(&self.g_r, self.kr, eps), self.kr, &sym_inverse(&self.g_c, self.mc, eps), self.mc)
    }

    /// noise dimensions under Σ ∝ G_s⁻¹ + t I (t = ∞: White); n(Π) is linear in Σ
    pub fn norms_t(&self, t: f64) -> Norms {
        if t.is_infinite() {
            return self.norms_white.clone();
        }
        let (a, b) = (&self.norms_ls, &self.norms_white);
        let f = |x: f64, y: f64| x + t * y;
        Norms {
            k_eff: f(a.k_eff, b.k_eff),
            n0: f(a.n0, b.n0),
            n_pc: f(a.n_pc, b.n_pc),
            n_rm: f(a.n_rm, b.n_rm),
            n_co: f(a.n_co, b.n_co),
            n_int: f(a.n_int, b.n_int),
            n_card: f(a.n_card, b.n_card),
            n_bin: a.n_bin.iter().zip(&b.n_bin).map(|(&x, &y)| f(x, y)).collect(),
            n_d: f(a.n_d, b.n_d),
        }
    }

    /// the noise mixture t for a measured ratio ρ = median(interaction energy) / median(second-difference energy)
    /// (Ls: 0, White: ∞, Auto: solve (n_int^LS + t n_int^W) / (n_D^LS + t n_D^W) = ρ, clamped to [0, ∞])
    pub fn resolve_mix(&self, rho: f64) -> f64 {
        match self.noise {
            NoiseModel::Ls => 0.0,
            NoiseModel::White => f64::INFINITY,
            NoiseModel::Auto => {
                let (a, b) = (&self.norms_ls, &self.norms_white);
                if self.kr < 2 || self.mc < 3 || !rho.is_finite() || a.n_d <= 0.0 {
                    return 0.0;
                }
                let (r0, r1) = (a.n_int / a.n_d, b.n_int / b.n_d);
                // f(t) runs monotonically from r0 (t = 0) to r1 (t = ∞)
                if (rho - r0) * (r1 - r0) <= 0.0 {
                    return 0.0;
                }
                if (rho - r1) * (r1 - r0) >= 0.0 {
                    return f64::INFINITY;
                }
                (rho * a.n_d - a.n_int) / (b.n_int - rho * b.n_d)
            }
        }
    }

    /// projector matrices (K × K) and noise dimensions under a model
    fn norms_for(&self, model: NoiseModel) -> Norms {
        let (kr, mc, k) = (self.kr, self.mc, self.k());
        let gs = kron(&self.g_r, kr, &self.g_c, mc);
        let m = match model {
            NoiseModel::Ls | NoiseModel::Auto => kron(&sym_inverse(&self.g_r, kr, 1e-9), kr, &sym_inverse(&self.g_c, mc, 1e-9), mc),
            NoiseModel::White => {
                let tr: f64 = (0..k).map(|i| gs[i * k + i]).sum();
                let lam = k as f64 / tr;
                (0..k * k).map(|q| if q / k == q % k { lam } else { 0.0 }).collect()
            }
        };
        // MG = M G_s; n(Π) = Σ_ij Π_ij (M G_s)_ji
        let mut mg = vec![0.0; k * k];
        for i in 0..k {
            for l in 0..k {
                let a = m[i * k + l];
                if a != 0.0 {
                    for j in 0..k {
                        mg[i * k + j] += a * gs[l * k + j];
                    }
                }
            }
        }
        let n_of = |pi: &dyn Fn(usize, usize) -> f64| -> f64 {
            let mut s = 0.0;
            for i in 0..k {
                for j in 0..k {
                    let p = pi(i, j);
                    if p != 0.0 {
                        s += p * mg[j * k + i];
                    }
                }
            }
            s
        };
        let (om_r, om_c) = (&self.om_r, &self.om_c);
        let d = |i: usize, j: usize| (i == j) as u8 as f64;
        let pc = |i: usize, j: usize| d(i / mc, j / mc) * om_c[j % mc];
        let pr = |i: usize, j: usize| d(i % mc, j % mc) * om_r[j / mc];
        let p0 = |_i: usize, j: usize| om_r[j / mc] * om_c[j % mc];
        let k_eff = n_of(&|i, j| d(i, j));
        let n0 = n_of(&p0);
        let n_pc = n_of(&pc);
        let n_rm = n_of(&|i, j| pc(i, j) - p0(i, j));
        let n_co = n_of(&|i, j| pr(i, j) - p0(i, j));
        let n_int = n_of(&|i, j| d(i, j) - pc(i, j) - pr(i, j) + p0(i, j));
        let n_card = n_of(&|i, j| d(i, j) - pc(i, j));
        let n_bin = (0..kr).map(|kk| n_of(&|i, j| if i / mc == kk { d(i, j) - pc(i, j) } else { 0.0 })).collect();
        let mut n_d = 0.0;
        if mc >= 3 {
            for kk in 0..kr {
                for (b, b1, b2) in self.d2_terms() {
                    let idx = [kk * mc + b, kk * mc + b1, kk * mc + b2];
                    let w = [1.0, -2.0, 1.0];
                    for p in 0..3 {
                        for q in 0..3 {
                            n_d += w[p] * w[q] * m[idx[p] * k + idx[q]];
                        }
                    }
                }
            }
        }
        Norms { k_eff, n0, n_pc, n_rm, n_co, n_int, n_card, n_bin, n_d }
    }

    /// cardiac second-difference stencils (b, b − 1, b − 2): all b with wrap if periodic, b ≥ 2 otherwise
    fn d2_terms(&self) -> Vec<(usize, usize, usize)> {
        let mc = self.mc;
        if mc < 3 {
            return vec![];
        }
        if self.periodic {
            (0..mc).map(|b| (b, (b + mc - 1) % mc, (b + mc - 2) % mc)).collect()
        } else {
            (2..mc).map(|b| (b, b - 1, b - 2)).collect()
        }
    }

    /// the four pieces of a voxel's coefficients c (K): mean, R (kr), Cd (mc), I (K)
    fn decompose<T: Elem>(&self, c: &[T], r: &mut [T], cd: &mut [T], i: &mut [T]) -> T {
        let (kr, mc) = (self.kr, self.mc);
        let mut m = T::default();
        for k in 0..kr {
            let mut s = T::default();
            for b in 0..mc {
                s = s + c[k * mc + b] * self.om_c32[b];
            }
            r[k] = s;
            m = m + s * self.om_r32[k];
        }
        for b in 0..mc {
            let mut s = T::default();
            for k in 0..kr {
                s = s + c[k * mc + b] * self.om_r32[k];
            }
            cd[b] = s;
        }
        for k in 0..kr {
            for b in 0..mc {
                i[k * mc + b] = c[k * mc + b] - r[k] - cd[b] + m;
            }
        }
        for v in r.iter_mut() {
            *v = *v - m;
        }
        for v in cd.iter_mut() {
            *v = *v - m;
        }
        m
    }

    fn norm_r<T: Elem>(&self, r: &[T]) -> f64 {
        let mut s = 0f64;
        for (k, row) in self.nz_r.iter().enumerate() {
            for &(j, g) in row {
                s += (g * r[k].rdot(r[j])) as f64;
            }
        }
        s
    }

    fn norm_c<T: Elem>(&self, cd: &[T]) -> f64 {
        let mut s = 0f64;
        for (b, row) in self.nz_c.iter().enumerate() {
            for &(j, g) in row {
                s += (g * cd[b].rdot(cd[j])) as f64;
            }
        }
        s
    }

    /// ‖x‖² in G_r ⊗ G_c (x: K), y scratch (K)
    fn norm_rc<T: Elem>(&self, x: &[T], y: &mut [T]) -> f64 {
        let mc = self.mc;
        for k in 0..self.kr {
            for b in 0..mc {
                let mut a = T::default();
                for &(j, g) in &self.nz_c[b] {
                    a = a + x[k * mc + j] * g;
                }
                y[k * mc + b] = a;
            }
        }
        let mut s = 0f64;
        for (k, row) in self.nz_r.iter().enumerate() {
            for &(j, g) in row {
                let mut t = 0f32;
                for b in 0..mc {
                    t += x[k * mc + b].rdot(y[j * mc + b]);
                }
                s += (g * t) as f64;
            }
        }
        s
    }

    /// G_s norm² of one respiratory knot's row (mc values)
    fn norm_row<T: Elem>(&self, k: usize, row: &[T]) -> f64 {
        self.g_r[k * self.kr + k] * self.norm_c(row)
    }
}

/// element of a coefficient image: complex (images, z) or real (velocity components)
pub trait Elem: Copy + Send + Sync + Default + Add<Output = Self> + Sub<Output = Self> + Mul<f32, Output = Self> {
    /// Re(conj(a) b)
    fn rdot(self, o: Self) -> f32;
}

impl Elem for Cf {
    fn rdot(self, o: Self) -> f32 {
        self.re * o.re + self.im * o.im
    }
}

impl Elem for f32 {
    fn rdot(self, o: Self) -> f32 {
        self * o
    }
}

#[derive(Clone, Debug, Default)]
pub struct KnotMap {
    pub s_card: Vec<f32>,
    pub s_resp: Vec<f32>,
    /// score of the breathing-dependence of the cardiac curve (interaction piece)
    pub s_int: Vec<f32>,
    pub s_flow: Vec<f32>,
    /// static weights in [0, 1]: over the cardiac cycle, over breathing (cardiac means), interaction under the
    /// respiratory lock (Additive; 0 for Nested)
    pub w_card: Vec<f32>,
    pub w_resp: Vec<f32>,
    pub w_rx: Vec<f32>,
    /// local noise per degree of freedom (G_s norm) of the unfiltered coefficients (images or z): group thresholds
    pub sigma_raw: Vec<f32>,
    /// the same for the velocity components (velocity model only, else empty)
    pub sigma_v: Vec<f32>,
    pub sigma2_blur: f64,
    pub sigma2_raw: f64,
    pub static_card: f64,
    pub static_resp: f64,
    /// fraction locked over both axes (w_card, w_resp and, Additive, w_rx > 0.5)
    pub static_full: f64,
    pub undetermined: f64,
    /// noise-model check: median interaction-piece noise / median second-difference noise of the blurred stack,
    /// normalised under (Ls, White); ≈ 1 for the model that matches the iterate (NaN without an interaction piece)
    pub model_check: [f64; 2],
    /// noise mixture t (Σ ∝ G_s⁻¹ + t I; ∞ = White) of (the blurred map statistics, the raw images / z, v); the lock
    /// uses [1] and [2] for its group thresholds (save it with the maps)
    pub noise_mix: [f64; 3],
    pub iteration: usize,
}

impl KnotMap {
    /// the maps (s_card, s_resp, s_int, s_flow, w_card, w_resp, w_rx, sigma_raw[, sigma_v]), image-major
    pub fn stacked(&self) -> Vec<f32> {
        let mut v: Vec<&Vec<f32>> = vec![&self.s_card, &self.s_resp, &self.s_int, &self.s_flow, &self.w_card, &self.w_resp, &self.w_rx, &self.sigma_raw];
        if !self.sigma_v.is_empty() {
            v.push(&self.sigma_v);
        }
        v.iter().flat_map(|x| x.iter().copied()).collect()
    }

    pub fn n_stacked(&self) -> usize {
        8 + (!self.sigma_v.is_empty()) as usize
    }

    /// maps saved by [`KnotMap::stacked`]; `noise_mix` as saved (needed by group locks)
    pub fn from_stacked(v: &[f32], nv: usize, iteration: usize, noise_mix: [f64; 3]) -> Self {
        let n = v.len() / nv;
        assert!(n == 8 || n == 9, "knot map: 8 or 9 maps, got {n}");
        let g = |i: usize| v[i * nv..(i + 1) * nv].to_vec();
        let frac = |w: &[f32]| w.iter().filter(|&&x| x > 0.5).count() as f64 / nv.max(1) as f64;
        let (w_card, w_resp, w_rx) = (g(4), g(5), g(6));
        let sigma_raw = g(7);
        let sigma2_raw = tlock::median(sigma_raw.iter().map(|&x| (x as f64).powi(2)).collect());
        let full = (0..nv).filter(|&q| w_card[q] > 0.5 && w_resp[q] > 0.5).count() as f64 / nv.max(1) as f64;
        Self {
            s_card: g(0),
            s_resp: g(1),
            s_int: g(2),
            s_flow: g(3),
            static_card: frac(&w_card),
            static_resp: frac(&w_resp),
            static_full: full,
            w_card,
            w_resp,
            w_rx,
            sigma_raw,
            sigma_v: if n == 9 { g(8) } else { vec![] },
            sigma2_blur: 0.0,
            sigma2_raw,
            undetermined: f64::NAN,
            model_check: [f64::NAN; 2],
            noise_mix,
            iteration,
        }
    }
}

/// per-voxel accumulators of the blurred statistics
struct Stats {
    e_card: Vec<f64>,
    e_rm: Vec<f64>,
    e_int: Vec<f64>,
    d2b: Vec<f64>,
    mbar: Vec<f64>,
    means: Vec<Vec<Cf>>,
}

/// blurred statistics of ne encodes of K images each, `fill(κ, e, out)` writes image κ of encode e
fn blur_stats(ne: usize, dims: [usize; 3], vox: [f64; 3], basis: &KnotBasis, opts: &LockOpts, want_means: bool, fill: &(dyn Fn(usize, usize, &mut [Cf]) + Sync)) -> Stats {
    let nv: usize = dims.iter().product();
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let sig = [0, 1, 2].map(|a| opts.blur_mm * 1e-3 / vox[a]);
    let mut st = Stats { e_card: vec![0.0; nv], e_rm: vec![0.0; nv], e_int: vec![0.0; nv], d2b: vec![0.0; nv], mbar: vec![0.0; nv], means: vec![] };
    let mut blurred: Vec<Vec<Cf>> = (0..k).map(|_| Vec::new()).collect();
    let (mut raw, mut tmp) = (vec![Cf::new(0.0, 0.0); nv], Vec::new());
    let terms = basis.d2_terms();
    for e in 0..ne {
        for (kk, b) in blurred.iter_mut().enumerate() {
            fill(kk, e, &mut raw);
            tlock::blur(&raw, dims, sig, &mut tmp, b);
        }
        let bl = &blurred;
        let mut mean_e = if want_means { vec![Cf::new(0.0, 0.0); nv] } else { vec![] };
        const CH: usize = 1024;
        let res: Vec<[f64; 5]> = (0..nv)
            .into_par_iter()
            .with_min_len(CH)
            .map_init(
                || (vec![Cf::new(0.0, 0.0); k], vec![Cf::new(0.0, 0.0); kr], vec![Cf::new(0.0, 0.0); mc], vec![Cf::new(0.0, 0.0); k], vec![Cf::new(0.0, 0.0); k]),
                |(c, r, cd, i, y), q| {
                    for kk in 0..k {
                        c[kk] = bl[kk][q];
                    }
                    let m = basis.decompose(c, r, cd, i);
                    let ni = if kr >= 2 && mc >= 2 { basis.norm_rc(i, y) } else { 0.0 };
                    let nco = basis.norm_c(cd);
                    let nrm = basis.norm_r(r);
                    let mut d2 = 0f64;
                    for kk in 0..kr {
                        for &(b, b1, b2) in &terms {
                            d2 += (c[kk * mc + b] - c[kk * mc + b1] * 2.0 + c[kk * mc + b2]).norm_sqr() as f64;
                        }
                    }
                    [nco + ni, nrm, ni, d2, (m.re as f64).hypot(m.im as f64)]
                },
            )
            .collect();
        if want_means {
            mean_e.par_iter_mut().enumerate().for_each(|(q, mq)| {
                let mut m = Cf::new(0.0, 0.0);
                for kk in 0..kr {
                    let mut s = Cf::new(0.0, 0.0);
                    for b in 0..mc {
                        s += bl[kk * mc + b][q] * basis.om_c32[b];
                    }
                    m += s * basis.om_r32[kk];
                }
                *mq = m;
            });
        }
        (st.e_card.par_iter_mut(), st.e_rm.par_iter_mut(), st.e_int.par_iter_mut(), st.d2b.par_iter_mut(), st.mbar.par_iter_mut(), res.par_iter()).into_par_iter().for_each(|(a, b, c, d, m, r)| {
            *a += r[0];
            *b += r[1];
            *c += r[2];
            *d += r[3];
            *m += r[4] / ne as f64;
        });
        if want_means {
            st.means.push(mean_e);
        }
    }
    st
}

/// local raw noise per degree of freedom from cardiac second differences of `ncomp` coefficient sets of K images each
/// (`get(κ, comp)`): block medians, floored at a tenth of the median; the noise mixture t from the interaction /
/// second-difference ratio on a voxel subsample; returns (σ per voxel, median σ², t)
fn raw_noise<'a, T: Elem>(ncomp: usize, nv: usize, dims: [usize; 3], vox: [f64; 3], basis: &KnotBasis, opts: &LockOpts, get: &(dyn Fn(usize, usize) -> &'a [T] + Sync)) -> (Vec<f32>, f64, f64) {
    let terms = basis.d2_terms();
    if terms.is_empty() {
        return (vec![0.0; nv], 0.0, basis.resolve_mix(f64::NAN));
    }
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let d2: Vec<f64> = (0..nv)
        .into_par_iter()
        .with_min_len(1024)
        .map(|q| {
            let mut s = 0f64;
            for comp in 0..ncomp {
                for kk in 0..kr {
                    for &(b, b1, b2) in &terms {
                        let x = get(kk * mc + b, comp)[q] - get(kk * mc + b1, comp)[q] * 2.0 + get(kk * mc + b2, comp)[q];
                        s += x.rdot(x) as f64;
                    }
                }
            }
            s
        })
        .collect();
    let step = nv.div_ceil(1 << 16).max(1);
    let rho = if kr >= 2 && mc >= 2 {
        let ei: Vec<f64> = (0..nv)
            .into_par_iter()
            .step_by(step)
            .map_init(
                || (vec![T::default(); k], vec![T::default(); kr], vec![T::default(); mc], vec![T::default(); k], vec![T::default(); k]),
                |(c, r, cd, i, y), q| {
                    let mut s = 0f64;
                    for comp in 0..ncomp {
                        for kk in 0..k {
                            c[kk] = get(kk, comp)[q];
                        }
                        basis.decompose(c, r, cd, i);
                        s += basis.norm_rc(i, y);
                    }
                    s
                },
            )
            .collect();
        tlock::median(ei) / tlock::median(d2.iter().step_by(step).cloned().collect()).max(1e-300)
    } else {
        f64::NAN
    };
    let t = basis.resolve_mix(rho);
    let nd = basis.norms_t(t).n_d * ncomp as f64;
    let d2: Vec<f64> = d2.par_iter().map(|v| v / nd).collect();
    let blk = [0, 1, 2].map(|a| (opts.noise_block_mm * 1e-3 / vox[a]).round() as usize);
    let nb = tlock::block_median(&d2, dims, blk);
    let step = nv.div_ceil(1 << 20).max(1);
    let med = tlock::median(nb.iter().step_by(step).cloned().collect());
    (nb.par_iter().map(|&v| v.max(0.1 * med).sqrt() as f32).collect(), med, t)
}

/// map from blurred statistics (+ raw noise); the shared tail of [`estimate`] and [`estimate_vel`]
fn finish_map(st: Stats, ne: usize, dims: [usize; 3], vox: [f64; 3], basis: &KnotBasis, kopts: &KnotLockOpts, dec: Option<&FlowDecoder>, raw: (Vec<f32>, f64, f64), iteration: usize) -> KnotMap {
    let (sigma_raw, sigma2_raw, t_raw) = raw;
    let opts = &kopts.base;
    let nv: usize = dims.iter().product();
    let (kr, mc) = (basis.kr, basis.mc);
    let sig = [0, 1, 2].map(|a| opts.blur_mm * 1e-3 / vox[a]);
    let gains: Vec<Vec<f64>> = (0..3).map(|a| tlock::blur_gain_axis(dims[a], &tlock::gauss(sig[a]))).collect();
    let gint: f64 = (0..3).map(|a| tlock::gauss(sig[a]).iter().map(|v| v * v).sum::<f64>()).product();
    let d = dims;
    let gq = |q: usize| -> f64 { gains[0][q % d[0]] * gains[1][(q / d[0]) % d[1]] * gains[2][q / (d[0] * d[1])] / gint };
    let blk = [0, 1, 2].map(|a| (opts.noise_block_mm * 1e-3 / vox[a]).round() as usize);
    let nef = ne as f64;
    let step = nv.div_ceil(1 << 20).max(1);
    // noise-model check and mixture: the interaction piece's noise level against the second differences' (both pure
    // noise in static voxels; their ratio depends on the coefficient covariance model)
    let have_d2 = !basis.d2_terms().is_empty();
    let rho = if have_d2 && kr >= 2 && mc >= 2 {
        let raw_d: Vec<f64> = (0..nv).step_by(step).map(|q| st.d2b[q] / gq(q)).collect();
        let raw_i: Vec<f64> = (0..nv).step_by(step).map(|q| st.e_int[q] / gq(q)).collect();
        tlock::median(raw_i) / tlock::median(raw_d).max(1e-300)
    } else {
        f64::NAN
    };
    let model_check = [rho * basis.norms_ls.n_d / basis.norms_ls.n_int, rho * basis.norms_white.n_d / basis.norms_white.n_int];
    let t_blur = basis.resolve_mix(rho);
    let nm = basis.norms_t(t_blur);
    let db: Vec<f64> = (0..nv).into_par_iter().map(|q| if have_d2 { st.d2b[q] / (nef * nm.n_d) / gq(q) } else { st.e_card[q] / (nef * nm.n_card) / gq(q) }).collect();
    let nb = tlock::block_median(&db, d, blk);
    let sigma2_blur = tlock::median(nb.iter().step_by(step).cloned().collect());
    let s2q: Vec<f64> = (0..nv).into_par_iter().map(|q| nb[q].max(0.1 * sigma2_blur).max(1e-30) * gq(q)).collect();
    let score = |e: f64, n: f64, q: usize| -> f32 { if n > 0.0 { (e.max(0.0) / (nef * n) / s2q[q]).sqrt() as f32 } else { 0.0 } };
    let s_card: Vec<f32> = (0..nv).into_par_iter().map(|q| score(st.e_card[q], nm.n_card, q)).collect();
    let s_resp: Vec<f32> = (0..nv).into_par_iter().map(|q| if kr >= 2 { score(st.e_rm[q], nm.n_rm, q) } else { 0.0 }).collect();
    let s_int: Vec<f32> = (0..nv).into_par_iter().map(|q| if kr >= 2 && mc >= 2 { score(st.e_int[q], nm.n_int, q) } else { 0.0 }).collect();
    let s_flow: Vec<f32> = match dec {
        Some(dec) => {
            let kv = dec.kv();
            let keff = (kv.iter().map(|k| k * k).sum::<f64>() / 3.0).sqrt().max(1e-12);
            (0..nv)
                .into_par_iter()
                .map(|q| {
                    let mut zb = [num_complex::Complex64::new(0.0, 0.0); crate::flow::MAX_ENC + 1];
                    for e in 0..ne {
                        let m = st.means[e][q];
                        zb[e] = num_complex::Complex64::new(m.re as f64, m.im as f64);
                    }
                    let z = &zb[..ne];
                    let mag = z[0].norm();
                    if mag <= 0.0 {
                        return 0.0;
                    }
                    let v = dec.velocity(z);
                    let vn = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if vn < opts.v_min {
                        return 0.0;
                    }
                    // noise of the mean (P_0) per encode: s² n_0
                    let sphi = (s2q[q] * nm.n0 / 2.0).sqrt() / mag;
                    let sv = 2f64.sqrt() * sphi / keff * 3f64.sqrt();
                    (vn / sv) as f32
                })
                .collect()
        }
        None => vec![0.0; nv],
    };
    let ramp = |s: f32| ((opts.s_hi as f32 - s) / (opts.s_hi - opts.s_lo) as f32).clamp(0.0, 1.0);
    // detection power: σ_t = √(s² K_eff) is the time-averaged noise of the evaluated signal ("per frame"); the
    // respiratory means average K_eff / n(P_c) of it; air: |m̄| below 3 noise levels of the mean
    let per_r = if nm.n_pc > 0.0 { nm.k_eff / nm.n_pc } else { 1.0 };
    let air = |q: usize| st.mbar[q] < 3.0 * (s2q[q] * nm.n0).sqrt();
    let conf = |q: usize, per: f64| -> f32 {
        if air(q) {
            return 1.0;
        }
        let rho = opts.s_hi * (s2q[q] * nm.k_eff).sqrt() / per.sqrt() / st.mbar[q];
        ((opts.detect_max - rho) / (opts.detect_max - opts.detect_ok)).clamp(0.0, 1.0) as f32
    };
    let conf_c: Vec<f32> = (0..nv).into_par_iter().map(|q| conf(q, 1.0)).collect();
    let undetermined = (0..nv).filter(|&q| conf_c[q] < 0.5 && !air(q)).count() as f64 / nv.max(1) as f64;
    let mut w_card: Vec<f32> = (0..nv).into_par_iter().map(|q| ramp(s_card[q].max(s_flow[q])) * conf_c[q]).collect();
    let mut w_resp: Vec<f32> = (0..nv).into_par_iter().map(|q| if kr >= 2 { ramp(s_resp[q].max(s_flow[q])) * conf(q, per_r) } else { 0.0 }).collect();
    let additive = kopts.coupling == Coupling::Additive && kr >= 2 && mc >= 2;
    let mut w_rx: Vec<f32> = (0..nv).into_par_iter().map(|q| if additive { ramp(s_resp[q].max(s_int[q]).max(s_flow[q])) * conf(q, per_r).min(conf_c[q]) } else { 0.0 }).collect();
    let rr = [0, 1, 2].map(|a| (opts.dilate_mm * 1e-3 / vox[a]).round() as usize);
    tlock::erode(&mut w_card, d, rr);
    if kr >= 2 {
        tlock::erode(&mut w_resp, d, rr);
    }
    if additive {
        tlock::erode(&mut w_rx, d, rr);
    }
    let frac = |w: &[f32]| w.iter().filter(|&&x| x > 0.5).count() as f64 / nv.max(1) as f64;
    let static_full = (0..nv).filter(|&q| w_card[q] > 0.5 && w_resp[q] > 0.5 && (!additive || w_rx[q] > 0.5)).count() as f64 / nv.max(1) as f64;
    KnotMap {
        static_card: frac(&w_card),
        static_resp: frac(&w_resp),
        static_full,
        undetermined,
        s_card,
        s_resp,
        s_int,
        s_flow,
        w_card,
        w_resp,
        w_rx,
        sigma_raw,
        sigma_v: vec![],
        sigma2_blur,
        sigma2_raw,
        model_check,
        noise_mix: [t_blur, t_raw, f64::NAN],
        iteration,
    }
}

/// map of a knot-image stack (layout: `FrameLayout::new(dims, vox, mc, kr, ne, encode_major)`, i.e. "frame"
/// f = k · mc + b); `dec`: flow decoder (steady-flow criterion)
pub fn estimate(h: &[C32], lay: &FrameLayout, basis: &KnotBasis, opts: &KnotLockOpts, dec: Option<&FlowDecoder>, iteration: usize) -> KnotMap {
    assert_eq!((lay.nr, lay.nc), (basis.kr, basis.mc), "layout (bins, frames) must be (kr, mc)");
    let h = cf(h);
    let (nv, ne, mc) = (lay.nv(), lay.ne, basis.mc);
    let img = move |kk: usize, e: usize| -> &[Cf] { &h[lay.img(kk / mc, kk % mc, e) * nv..][..nv] };
    let fill = |kk: usize, e: usize, out: &mut [Cf]| out.copy_from_slice(img(kk, e));
    let st = blur_stats(ne, lay.dims, lay.vox, basis, &opts.base, dec.is_some(), &fill);
    let raw = raw_noise::<Cf>(ne, nv, lay.dims, lay.vox, basis, &opts.base, &img);
    finish_map(st, ne, lay.dims, lay.vox, basis, opts, dec, raw, iteration)
}

/// map of the knot velocity model: z[κ] (nv complex), v[κ] (3 nv reals, component-major), κ = k · mc + b. The scores
/// come from the pseudo-images z_κ exp(i φ_e(v_κ)) (φ_e = `dec.phase`, streamed per encode); `sigma_raw` is z's own
/// noise level, `sigma_v` v's (per component), for the group thresholds of [`apply_z`] / [`apply_v`].
pub fn estimate_vel(z: &[&[C32]], v: &[&[f32]], dims: [usize; 3], vox: [f64; 3], basis: &KnotBasis, opts: &KnotLockOpts, dec: &FlowDecoder, iteration: usize) -> KnotMap {
    let k = basis.k();
    let nv: usize = dims.iter().product();
    assert!(z.len() == k && v.len() == k, "z, v: one slice per knot");
    assert!(z.iter().all(|s| s.len() == nv) && v.iter().all(|s| s.len() == 3 * nv));
    let ne = dec.n_enc();
    let zc: Vec<&[Cf]> = z.iter().map(|s| cf(s)).collect();
    let fill = |kk: usize, e: usize, out: &mut [Cf]| {
        let (zk, vk) = (zc[kk], v[kk]);
        out.par_iter_mut().enumerate().with_min_len(4096).for_each(|(q, o)| {
            let ph = dec.phase(e, &[vk[q] as f64, vk[nv + q] as f64, vk[2 * nv + q] as f64]) as f32;
            *o = zk[q] * Cf::from_polar(1.0, ph);
        });
    };
    let st = blur_stats(ne, dims, vox, basis, &opts.base, true, &fill);
    let rz = raw_noise::<Cf>(1, nv, dims, vox, basis, &opts.base, &|kk, _| zc[kk]);
    let (sv, _, tv) = raw_noise::<f32>(3, nv, dims, vox, basis, &opts.base, &|kk, c| &v[kk][c * nv..(c + 1) * nv]);
    let mut m = finish_map(st, ne, dims, vox, basis, opts, Some(dec), rz, iteration);
    m.sigma_v = sv;
    m.noise_mix[2] = tv;
    m
}

/// per-voxel lock factors (rm, co, int) and the nested per-bin cardiac factors
struct Factors {
    g_rm: f32,
    g_co: f32,
    g_int: f32,
}

/// the lock on `ncomp` coefficient sets of K images (`ptrs[comp * K + κ]`, nv elements each), per-voxel group noise
/// `sigma`; groups span the components
fn lock_sets<T: Elem>(ptrs: &[usize], ncomp: usize, nv: usize, basis: &KnotBasis, map: &KnotMap, sigma: &[f32], t_mix: f64, opts: &KnotLockOpts) {
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    assert_eq!(ptrs.len(), ncomp * k);
    let o = &opts.base;
    let mu = o.mu as f32;
    let nm = basis.norms_t(if t_mix.is_nan() { basis.resolve_mix(f64::NAN) } else { t_mix });
    let nested = opts.coupling == Coupling::Nested || kr < 2 || mc < 2;
    let per_bin = nested && basis.diag_r;
    let ncf = ncomp as f64;
    let sq = |n: f64| (n * ncf).max(0.0).sqrt() as f32;
    let (r_rm, r_co, r_int, r_card) = (sq(nm.n_rm), sq(nm.n_co), sq(nm.n_int), sq(nm.n_card));
    let r_bin: Vec<f32> = nm.n_bin.iter().map(|&n| sq(n)).collect();
    let shrink = |tau: f32, n2: f64| -> f32 { if n2 > 0.0 { (1.0 - tau / (n2 as f32).sqrt()).max(0.0) } else { 0.0 } };
    const CH: usize = 2048;
    (0..nv.div_ceil(CH)).into_par_iter().for_each(|ch| {
        let q0 = ch * CH;
        let n = CH.min(nv - q0);
        // SAFETY: each task owns voxels q0..q0+n of every image; the images are disjoint allocations or ranges
        let view = |i: usize| unsafe { std::slice::from_raw_parts_mut((ptrs[i] as *mut T).add(q0), n) };
        // skip chunks without any lock
        if (q0..q0 + n).all(|q| map.w_card[q] <= 0.0 && map.w_resp[q] <= 0.0 && map.w_rx[q] <= 0.0) {
            return;
        }
        let mut buf = vec![T::default(); ncomp * k * n];
        for i in 0..ncomp * k {
            buf[i * n..(i + 1) * n].copy_from_slice(view(i));
        }
        let mut c = vec![T::default(); k];
        let (mut r, mut cd, mut it, mut y) = (vec![vec![T::default(); kr]; ncomp], vec![vec![T::default(); mc]; ncomp], vec![vec![T::default(); k]; ncomp], vec![T::default(); k]);
        let mut m = vec![T::default(); ncomp];
        let mut gbin = vec![1f32; kr];
        for j in 0..n {
            let q = q0 + j;
            let (wc, wr, wx) = (map.w_card[q].max(0.0), map.w_resp[q].max(0.0), if nested { 0.0 } else { map.w_rx[q].max(0.0) });
            if wc <= 0.0 && wr <= 0.0 && wx <= 0.0 {
                continue;
            }
            for comp in 0..ncomp {
                for kk in 0..k {
                    c[kk] = buf[(comp * k + kk) * n + j];
                }
                m[comp] = basis.decompose(&c, &mut r[comp], &mut cd[comp], &mut it[comp]);
            }
            let f = if !o.group {
                let g_co = 1.0 / (1.0 + mu * wc);
                Factors { g_rm: 1.0 / (1.0 + mu * wr), g_co, g_int: if nested { g_co } else { 1.0 / (1.0 + mu * (wc + wx)) } }
            } else {
                let s = sigma[q];
                let nrm: f64 = (0..ncomp).map(|cc| basis.norm_r(&r[cc])).sum();
                let g_rm = if wr > 0.0 { shrink(mu * wr * s * r_rm, nrm) } else { 1.0 };
                if per_bin {
                    // tlock: cardiac groups per bin (G_r diagonal ⇒ the bins' cardiac deviations are G_s-orthogonal)
                    for kk in 0..kr {
                        gbin[kk] = if wc > 0.0 {
                            let n2: f64 = (0..ncomp)
                                .map(|cc| {
                                    let row: Vec<T> = (0..mc).map(|b| cd[cc][b] + it[cc][kk * mc + b]).collect();
                                    basis.norm_row(kk, &row)
                                })
                                .sum();
                            shrink(mu * wc * s * r_bin[kk], n2)
                        } else {
                            1.0
                        };
                    }
                    Factors { g_rm, g_co: f32::NAN, g_int: f32::NAN }
                } else {
                    let nco: f64 = (0..ncomp).map(|cc| basis.norm_c(&cd[cc])).sum();
                    let ni: f64 = (0..ncomp).map(|cc| basis.norm_rc(&it[cc], &mut y)).sum();
                    if nested {
                        let g = if wc > 0.0 { shrink(mu * wc * s * r_card, nco + ni) } else { 1.0 };
                        Factors { g_rm, g_co: g, g_int: g }
                    } else {
                        let g_co = if wc > 0.0 { shrink(mu * wc * s * r_co, nco) } else { 1.0 };
                        let g_int = if wc + wx > 0.0 { shrink(mu * (wc + wx) * s * r_int, ni) } else { 1.0 };
                        Factors { g_rm, g_co, g_int }
                    }
                }
            };
            for comp in 0..ncomp {
                for kk in 0..kr {
                    let base = m[comp] + r[comp][kk] * f.g_rm;
                    for b in 0..mc {
                        let x = if per_bin && o.group { base + (cd[comp][b] + it[comp][kk * mc + b]) * gbin[kk] } else { base + cd[comp][b] * f.g_co + it[comp][kk * mc + b] * f.g_int };
                        buf[(comp * k + kk * mc + b) * n + j] = x;
                    }
                }
            }
        }
        for i in 0..ncomp * k {
            view(i).copy_from_slice(&buf[i * n..(i + 1) * n]);
        }
    });
}

/// the lock on a knot-image stack, in place (every encode separately; layout as [`estimate`])
pub fn apply(h: &mut [C32], lay: &FrameLayout, basis: &KnotBasis, map: &KnotMap, opts: &KnotLockOpts) {
    assert_eq!((lay.nr, lay.nc), (basis.kr, basis.mc));
    let (nv, mc) = (lay.nv(), basis.mc);
    let h = cf_mut(h);
    let base = h.as_mut_ptr() as usize;
    for e in 0..lay.ne {
        let ptrs: Vec<usize> = (0..basis.k()).map(|kk| base + lay.img(kk / mc, kk % mc, e) * nv * std::mem::size_of::<Cf>()).collect();
        lock_sets::<Cf>(&ptrs, 1, nv, basis, map, &map.sigma_raw, map.noise_mix[1], opts);
    }
}

/// the lock on the velocity model's complex references z[κ] (nv each), in place
pub fn apply_z(z: &mut [&mut [C32]], basis: &KnotBasis, map: &KnotMap, opts: &KnotLockOpts) {
    assert_eq!(z.len(), basis.k());
    let nv = map.w_card.len();
    assert!(z.iter().all(|s| s.len() == nv));
    let ptrs: Vec<usize> = z.iter_mut().map(|s| s.as_mut_ptr() as usize).collect();
    lock_sets::<Cf>(&ptrs, 1, nv, basis, map, &map.sigma_raw, map.noise_mix[1], opts);
}

/// the lock on the velocity model's velocities v[κ] (3 nv reals, component-major), in place: temporal constancy of
/// the 3-vector along the locked axes (group norms over the three components); the mean velocity is kept
pub fn apply_v(v: &mut [&mut [f32]], basis: &KnotBasis, map: &KnotMap, opts: &KnotLockOpts) {
    assert_eq!(v.len(), basis.k());
    let nv = map.w_card.len();
    assert!(v.iter().all(|s| s.len() == 3 * nv));
    let k = basis.k();
    let mut ptrs = vec![0usize; 3 * k];
    for (kk, s) in v.iter_mut().enumerate() {
        let b = s.as_mut_ptr() as usize;
        for c in 0..3 {
            ptrs[c * k + kk] = b + c * nv * std::mem::size_of::<f32>();
        }
    }
    let sig = if map.sigma_v.is_empty() { vec![0.0; nv] } else { map.sigma_v.clone() };
    assert!(!(opts.base.group && map.sigma_v.is_empty()), "group v lock needs sigma_v (estimate_vel)");
    lock_sets::<f32>(&ptrs, 3, nv, basis, map, &sig, map.noise_mix[2], opts);
}

/// local noise level per degree of freedom (G_s norm, as `KnotMap::sigma_raw`) and noise mixture of a real coefficient
/// set over the K knots (`sets[κ]`, nv each): the group thresholds of [`apply_real`] (spline velocity model: m, ψ)
pub fn real_noise(sets: &[&[f32]], dims: [usize; 3], vox: [f64; 3], basis: &KnotBasis, opts: &KnotLockOpts) -> (Vec<f32>, f64) {
    assert_eq!(sets.len(), basis.k(), "one set per knot");
    let nv: usize = dims.iter().product();
    let (s, _, t) = raw_noise::<f32>(1, nv, dims, vox, basis, &opts.base, &|kk, _| sets[kk]);
    (s, t)
}

/// the lock on a real coefficient set over the K knots (`sets[κ]`, nv each), in place: the parameter-spline velocity
/// model's magnitude m and zero-moment phase ψ (research/velocity_model_sr.md §F) with their own noise level
/// ([`real_noise`]); the map comes from [`estimate_vel`] on the pseudo-images m e^{iψ}
pub fn apply_real(sets: &mut [&mut [f32]], basis: &KnotBasis, map: &KnotMap, opts: &KnotLockOpts, sigma: &[f32], t_mix: f64) {
    assert_eq!(sets.len(), basis.k());
    let nv = map.w_card.len();
    assert!(sets.iter().all(|s| s.len() == nv) && sigma.len() == nv);
    let ptrs: Vec<usize> = sets.iter_mut().map(|s| s.as_mut_ptr() as usize).collect();
    lock_sets::<f32>(&ptrs, 1, nv, basis, map, sigma, t_mix, opts);
}

/// G_s inner product of two knot coefficient sets per voxel, summed over voxels (tests, diagnostics):
/// Σ_q Re⟨a_q, b_q⟩_{G_r ⊗ G_c}
pub fn gs_dot<T: Elem>(basis: &KnotBasis, a: &[&[T]], b: &[&[T]]) -> f64 {
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    assert!(a.len() == k && b.len() == k);
    let nv = a[0].len();
    (0..nv)
        .into_par_iter()
        .map(|q| {
            let mut s = 0f64;
            for k1 in 0..kr {
                for &(k2, gr) in &basis.nz_r[k1] {
                    for b1 in 0..mc {
                        for &(b2, gc) in &basis.nz_c[b1] {
                            s += (gr * gc * a[k1 * mc + b1][q].rdot(b[k2 * mc + b2][q])) as f64;
                        }
                    }
                }
            }
            s
        })
        .sum()
}
