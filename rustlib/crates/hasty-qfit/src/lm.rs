//! Second-order Levenberg–Marquardt with a trust-region-style λ update (CPU reference; `gpu.rs` runs the same
//! algorithm one problem per CUDA thread).
//!
//! Per iteration: A = H + λ diag(max(D_j, d_min · max D)) with H the full Hessian Σ w (JᵀJ + r ∇²m) (or Gauss–Newton
//! JᵀJ) and D = diag(JᵀJ) (Marquardt scaling); A is factored by a modified Cholesky (GMW81 or SE99) so the step
//! s = −A⁻¹g is a descent direction even where H is indefinite. Box bounds: variables held at a bound by an outward
//! gradient are fixed for the step (active set, projected Newton), the rest is solved and projected onto the box. The gain
//! ratio ρ = (f − f(p + s)) / (−gᵀs − ½ sᵀHs) decides acceptance and the λ update (classic: accept if ρ > μ, shrink λ
//! if ρ > η, grow otherwise; or Nielsen's continuous rule).
//!
//! # Scaling and statistics
//!
//! The solver works in scaled variables u = S⁻¹p (S = diag(scale) > 0, the typical magnitude of each parameter), i.e.
//! on f̃(u) = f(S u). Everything is formed in u: gradient S g, Hessian S H S, damping λ diag(max(S²D, floor)) with an
//! isotropic floor = d_min · max_j S_j² D_j, the modified-Cholesky shift E, the box [lo/S, hi/S] and the step test.
//! Consequences:
//! - The damping is a Tikhonov term in the scaled variables. Its Marquardt part λ S²D equals λ D of the unscaled
//!   problem, so S does not change it at all; only the floor (which keeps the system well posed for parameters with
//!   ~zero information, e.g. Δf where there is no signal) and E depend on S, and they act in units of each
//!   parameter's typical magnitude, not in arbitrary physical units.
//! - S is a bijective reparametrisation: f̃ and f have the same value at corresponding points, so the minimiser (the
//!   maximum-likelihood estimate when w = 1/σ², see below) is the same point for every S, and the box projection
//!   commutes with a positive diagonal S. The iteration stops only where the projected gradient vanishes (s = 0 ⇔
//!   P(S g) = 0 because A is positive definite), so its fixed points are the KKT points of the original problem.
//!   S changes the path (conditioning, especially in f32), never the answer.
//! - Statistics are fixed by the objective f = ½ Σ w r², not by the solver: f is the negative log-likelihood (up to
//!   a constant) iff w_i ∝ 1/σ_i² of independent Gaussian residuals (complex data: real and imaginary parts as two
//!   residuals with equal weight — circular noise; never fit magnitudes, which are Rician). Multiplying all residuals
//!   of one problem by one constant (e.g. normalising a voxel's data by its RMS) only scales f and keeps the MLE;
//!   per-sample rescaling of data or residuals does change it and must go through w instead.
//! - Stopping: `tol_stat` stops when the remaining (projected) Newton step is shorter than `tol_stat` estimated
//!   standard errors, δ² = (S g)ᵀ (S H S)⁻¹ (S g) ≤ tol_stat² σ̂², σ̂² = 2f / (m − n) (H ≈ Fisher information / σ̂²).
//!   It is invariant to S, to the data scale and to the weight scale, so the returned point is the MLE to a
//!   statistically negligible fraction of its standard error whatever scaling is used. On noise-free data it never
//!   triggers (σ̂ → 0); the step/function/gradient tolerances are the backstop there and at the precision floor.

use crate::chol::{chol_solve, gmw81, se99};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianMode {
    GaussNewton,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModChol {
    Gmw81,
    Se99,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LambdaRule {
    /// accept if ρ > mu; λ ·= down if ρ > eta; reject: λ ·= up
    Classic { mu: f64, eta: f64, down: f64, up: f64 },
    /// Nielsen (1999): accept if ρ > 0, λ ·= max(1/3, 1 − (2ρ − 1)³); reject: λ ·= ν, ν ·= 2
    Nielsen,
}

#[derive(Clone, Debug)]
pub struct LmOptions {
    pub max_iter: u32,
    /// on the projected gradient, ∞-norm
    pub tol_grad: f64,
    /// relative step: ‖s‖∞ ≤ tol_step (‖p‖∞ + tol_step)
    pub tol_step: f64,
    /// relative decrease of an accepted step: f_old − f ≤ tol_fun · f_old
    pub tol_fun: f64,
    /// statistical stop: remaining Newton step ≤ tol_stat estimated standard errors (0 = off); see the module docs
    pub tol_stat: f64,
    pub lambda0: f64,
    /// Marquardt-scaling floor, relative to max_j D_j
    pub d_min: f64,
    pub hessian: HessianMode,
    pub modchol: ModChol,
    pub rule: LambdaRule,
    /// typical magnitude per parameter; the solver works in u = p / scale (conditioning, important in f32).
    /// None: max(|lo|, |hi|) where finite and non-zero, else 1
    pub scale: Option<Vec<f64>>,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol_grad: 1e-12,
            tol_step: 1e-10,
            tol_fun: 1e-14,
            tol_stat: 1e-4,
            lambda0: 1e-3,
            d_min: 1e-6,
            hessian: HessianMode::Full,
            modchol: ModChol::Se99,
            rule: LambdaRule::Classic { mu: 1e-4, eta: 0.75, down: 1.0 / 3.0, up: 4.0 },
            scale: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    GradTol = 1,
    StepTol = 2,
    FunTol = 3,
    MaxIter = 4,
    NonFinite = 5,
    LambdaOverflow = 6,
    /// remaining Newton step below `tol_stat` standard errors
    StatTol = 7,
}

impl Status {
    pub fn from_i32(x: i32) -> Status {
        match x {
            1 => Status::GradTol,
            2 => Status::StepTol,
            3 => Status::FunTol,
            4 => Status::MaxIter,
            5 => Status::NonFinite,
            7 => Status::StatTol,
            _ => Status::LambdaOverflow,
        }
    }
    pub fn converged(&self) -> bool {
        matches!(self, Status::GradTol | Status::StepTol | Status::FunTol | Status::StatTol)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FitOut {
    pub f: f64,
    pub iters: u32,
    pub status: Status,
}

/// what the solver needs from a problem: symbolic models implement it via `SymProblem`; hand-written objectives
/// (e.g. VarPro) implement it directly
pub trait Objective {
    fn n_param(&self) -> usize;
    /// number of real residuals m (for the residual-variance estimate σ̂² = 2f / (m − n) of the statistical stop)
    fn n_resid(&self) -> usize;
    /// f = ½ Σ w r²
    fn value(&self, p: &[f64]) -> f64;
    /// returns f; fills g (n), h (n×n, full, row-major) and d = diag of the Gauss–Newton part (n)
    fn derivs(&self, p: &[f64], full: bool, g: &mut [f64], h: &mut [f64], d: &mut [f64]) -> f64;
}

/// the parameter scaling used by the solver (see [`LmOptions::scale`]); every entry finite and > 0
pub fn scales(o: &LmOptions, lo: &[f64], hi: &[f64]) -> Vec<f64> {
    let s: Vec<f64> = match &o.scale {
        Some(s) => s.clone(),
        None => (0..lo.len())
            .map(|j| {
                let m = [lo[j], hi[j]].iter().filter(|x| x.is_finite()).fold(0.0f64, |a, x| a.max(x.abs()));
                if m > 0.0 { m } else { 1.0 }
            })
            .collect(),
    };
    assert_eq!(s.len(), lo.len(), "one scale per parameter");
    assert!(s.iter().all(|x| x.is_finite() && *x > 0.0), "scales must be finite and positive: {s:?}");
    s
}

/// free coordinates: not held at a bound by a gradient pushing outwards
fn free(p: f64, lo: f64, hi: f64, g: f64) -> bool {
    !((p <= lo && g > 0.0) || (p >= hi && g < 0.0))
}

/// projected Newton decrement in the scaled variables, δ² = (S g)_F ᵀ ((S H S)_FF)⁻¹ (S g)_F over the free set F;
/// None if (S H S)_FF is not safely positive definite (not near a minimum: the test does not apply). `a`: n×n scratch
pub fn newton_decrement(h: &[f64], g: &[f64], sc: &[f64], p: &[f64], lo: &[f64], hi: &[f64], a: &mut [f64]) -> Option<f64> {
    let n = g.len();
    let idx: Vec<usize> = (0..n).filter(|&j| free(p[j], lo[j], hi[j], g[j])).collect();
    let nf = idx.len();
    if nf == 0 {
        return Some(0.0);
    }
    let mut dmax = 0.0f64;
    for (r, &i) in idx.iter().enumerate() {
        for (c, &j) in idx.iter().enumerate() {
            a[r * nf + c] = sc[i] * sc[j] * h[i * n + j];
        }
        dmax = dmax.max(a[r * nf + r]);
    }
    if !(dmax > 0.0) {
        return None;
    }
    // unpivoted Cholesky (backward stable for PD matrices) + forward substitution: δ² = ‖L⁻¹ S g‖²
    let mut d2 = 0.0;
    let mut z = vec![0.0f64; nf];
    for j in 0..nf {
        let mut v = a[j * nf + j];
        for k in 0..j {
            v -= a[j * nf + k] * a[j * nf + k];
        }
        if !(v > 1e3 * f64::EPSILON * dmax) {
            return None;
        }
        let ljj = v.sqrt();
        a[j * nf + j] = ljj;
        for i in j + 1..nf {
            let mut v = a[i * nf + j];
            for k in 0..j {
                v -= a[i * nf + k] * a[j * nf + k];
            }
            a[i * nf + j] = v / ljj;
        }
        let mut t = g[idx[j]] * sc[idx[j]];
        for k in 0..j {
            t -= a[j * nf + k] * z[k];
        }
        z[j] = t / ljj;
        d2 += z[j] * z[j];
    }
    Some(d2)
}

fn clamp(p: &mut [f64], lo: &[f64], hi: &[f64]) {
    for j in 0..p.len() {
        p[j] = p[j].clamp(lo[j], hi[j]);
    }
}

/// minimise `obj` from `p` (in/out) inside [lo, hi]
pub fn lm(obj: &dyn Objective, p: &mut [f64], lo: &[f64], hi: &[f64], o: &LmOptions) -> FitOut {
    let n = obj.n_param();
    let m = obj.n_resid();
    let full = o.hessian == HessianMode::Full;
    clamp(p, lo, hi);
    let (mut g, mut h, mut d) = (vec![0.0; n], vec![0.0; n * n], vec![0.0; n]);
    let mut f = obj.derivs(p, full, &mut g, &mut h, &mut d);
    let mut lambda = o.lambda0;
    let mut nu = 2.0f64;
    let (mut a, mut s, mut pt) = (vec![0.0; n * n], vec![0.0; n], vec![0.0; n]);
    let mut perm = vec![0usize; n];
    let mut neg_g = vec![0.0; n];
    let sc = scales(o, lo, hi);
    for it in 1..=o.max_iter {
        if !f.is_finite() {
            return FitOut { f, iters: it - 1, status: Status::NonFinite };
        }
        // projected gradient (components pushing out of an active bound do not count)
        let mut pg = 0.0f64;
        for j in 0..n {
            if free(p[j], lo[j], hi[j], g[j]) {
                pg = pg.max((g[j] * sc[j]).abs());
            }
        }
        if pg <= o.tol_grad {
            return FitOut { f, iters: it - 1, status: Status::GradTol };
        }
        if o.tol_stat > 0.0 && m > n {
            let sig2 = 2.0 * f / (m - n) as f64;
            if let Some(d2) = newton_decrement(&h, &g, &sc, p, lo, hi, &mut a) {
                if d2 <= o.tol_stat * o.tol_stat * sig2 {
                    return FitOut { f, iters: it - 1, status: Status::StatTol };
                }
            }
        }
        // scaled system: (S H S + λ diag(max(S² D, floor))) u = −S g, s = S u
        let dmax = (0..n).fold(0.0f64, |m, j| m.max(d[j] * sc[j] * sc[j]));
        let floor = (o.d_min * dmax).max(f64::MIN_POSITIVE);
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = sc[i] * sc[j] * h[i * n + j];
            }
        }
        for j in 0..n {
            a[j * n + j] += lambda * (d[j] * sc[j] * sc[j]).max(floor);
        }
        // active set: a variable held at a bound by an outward gradient does not move (projected Newton); the step
        // is the solve on the free variables only, so clipping cannot stall the iteration along a bound
        for j in 0..n {
            if !free(p[j], lo[j], hi[j], g[j]) {
                for c in 0..n {
                    a[j * n + c] = 0.0;
                    a[c * n + j] = 0.0;
                }
                a[j * n + j] = 1.0;
            }
        }
        match o.modchol {
            ModChol::Gmw81 => gmw81(&mut a, n, &mut perm, f64::EPSILON),
            ModChol::Se99 => se99(&mut a, n, &mut perm, f64::EPSILON),
        };
        for j in 0..n {
            neg_g[j] = if free(p[j], lo[j], hi[j], g[j]) { -g[j] * sc[j] } else { 0.0 };
        }
        chol_solve(&a, n, &perm, &neg_g, &mut s);
        for j in 0..n {
            pt[j] = p[j] + s[j] * sc[j];
        }
        clamp(&mut pt, lo, hi);
        let mut smax = 0.0f64;
        let mut pmax = 0.0f64;
        for j in 0..n {
            s[j] = pt[j] - p[j];
            smax = smax.max((s[j] / sc[j]).abs());
            pmax = pmax.max((p[j] / sc[j]).abs());
        }
        if smax <= o.tol_step * (pmax + o.tol_step) {
            return FitOut { f, iters: it, status: Status::StepTol };
        }
        let mut pred = 0.0;
        for i in 0..n {
            let mut hs = 0.0;
            for j in 0..n {
                hs += h[i * n + j] * s[j];
            }
            pred -= g[i] * s[i] + 0.5 * s[i] * hs;
        }
        let ft = obj.value(&pt);
        let act = f - ft;
        let rho = if pred > 0.0 { act / pred } else { -1.0 };
        let accept = ft.is_finite()
            && pred > 0.0
            && act > 0.0
            && match o.rule {
                LambdaRule::Classic { mu, .. } => rho > mu,
                LambdaRule::Nielsen => rho > 0.0,
            };
        if accept {
            p.copy_from_slice(&pt);
            let f_old = f;
            f = obj.derivs(p, full, &mut g, &mut h, &mut d);
            match o.rule {
                LambdaRule::Classic { eta, down, .. } => {
                    if rho > eta {
                        lambda *= down;
                    }
                }
                LambdaRule::Nielsen => {
                    lambda *= (1.0 / 3.0f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
                    nu = 2.0;
                }
            }
            lambda = lambda.max(1e-15);
            if f_old - f <= o.tol_fun * f_old {
                return FitOut { f, iters: it, status: Status::FunTol };
            }
        } else {
            match o.rule {
                LambdaRule::Classic { up, .. } => lambda *= up,
                LambdaRule::Nielsen => {
                    lambda *= nu;
                    nu *= 2.0;
                }
            }
            if lambda > 1e16 {
                return FitOut { f, iters: it, status: Status::LambdaOverflow };
            }
        }
    }
    FitOut { f, iters: o.max_iter, status: Status::MaxIter }
}
