//! λ_max estimation for the step size α = 1 / λ̂ (research/temporal_operator.md, "Step size by Lanczos").
//!
//! The operator is T = M A with A Hermitian positive semi-definite (the normal operator N, or the Gauss–Newton
//! Hessian H) and M Hermitian positive definite (the circulant / diagonal metric; M = I without one). T is
//! self-adjoint in the M⁻¹ inner product ⟨x, y⟩ = xᴴ M⁻¹ y, so the Lanczos process runs in that inner product and
//! its tridiagonal T_k has real Ritz values θ_1 ≥ … ≥ θ_k inside [λ_min, λ_max] of T.
//!
//! Two equivalent forms of the three-term recurrence (q_j M⁻¹-orthonormal, p_j = M⁻¹ q_j):
//!
//! * **p-form** (no M⁻¹; [`PForm`] doc): w = A q_j (= M⁻¹ T q_j), α_j = Re q_jᴴ w (= ⟨T q_j, q_j⟩),
//!   s = w − α_j p_j − β_j p_{j−1} (= M⁻¹ r), r = M s, β_{j+1} = √(Re sᴴ r) (= ‖r‖_{M⁻¹}),
//!   q_{j+1} = r / β_{j+1}, p_{j+1} = s / β_{j+1}. One A and one M per step, four vectors (three when M = I).
//! * **q-form** (M⁻¹ only inside the norm, as the power iteration already does): w = A q_j, α_j = Re q_jᴴ w,
//!   r = M w − α_j q_j − β_j q_{j−1}, β_{j+1} = √(Re rᴴ M⁻¹ r). Three vectors; everything after w = A q_j is
//!   elementwise per chunk, so the velocity objective keeps q_{j−1} in host memory and needs only q and w on the device.
//!
//! No reorthogonalisation: with k ≤ ~10 steps and only the top Ritz value wanted, loss of orthogonality only produces
//! copies ("ghosts") of an already converged extreme Ritz value — it never moves θ_1 above λ_max by more than
//! O(ε ‖T‖) (Paige). The residual of the top Ritz pair (θ, y = Q_k s) is ‖T y − θ y‖_{M⁻¹} = β_{k+1} |s_k|, so some
//! eigenvalue of T lies in [θ − b, θ + b] with b = β_{k+1} |s_k|. θ ≤ λ_max always; when the Ritz value approximates
//! the top eigenvalue (the start vector is not deficient in it) λ_max ≤ θ + b. The step uses λ̂ = θ + b: an upper
//! estimate whose excess over λ_max is the (measured) residual bound, versus the power iteration's unguarded
//! underestimate.

use std::fmt;

/// how the step size's λ_max is estimated
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LamMethod {
    /// the power iteration with the caller's iteration count (the original path, unchanged)
    Power,
    /// `k` Lanczos steps; λ̂ = θ + residual bound
    Lanczos(usize),
}

impl LamMethod {
    /// `power` | `lanczos` | `lanczos:K` (default K = 8)
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "power" => Some(Self::Power),
            "lanczos" => Some(Self::Lanczos(8)),
            _ => s.strip_prefix("lanczos:").and_then(|k| k.parse().ok()).filter(|&k: &usize| k >= 1).map(Self::Lanczos),
        }
    }
}

impl fmt::Display for LamMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Power => write!(f, "power"),
            Self::Lanczos(k) => write!(f, "lanczos:{k}"),
        }
    }
}

/// a λ_max estimate: top Ritz value (power: the last Rayleigh quotient / growth), its residual bound (0 for power),
/// the operator applications spent
#[derive(Clone, Copy, Debug, Default)]
pub struct LamEst {
    pub theta: f64,
    pub bound: f64,
    pub applies: usize,
}

impl LamEst {
    pub fn power(lam: f64, applies: usize) -> Self {
        Self { theta: lam, bound: 0.0, applies }
    }
    /// the conservative value for α = 1 / λ̂ (power: λ itself, exactly as before)
    pub fn hat(&self) -> f64 {
        self.theta + self.bound
    }
    /// max over several estimates (sets, coil draws): the largest λ̂, applies summed
    pub fn max(self, o: Self) -> Self {
        let a = if o.hat() > self.hat() { o } else { self };
        Self { applies: self.applies + o.applies, ..a }
    }
}

/// The vector space of one Lanczos run (p-form or q-form, see the module doc). The driver only sees scalars.
pub trait LanczosSpace {
    type E;
    /// set the start vector (q_0, and p_0 = M⁻¹ q_0 in the p-form); return ‖q_0‖²_{M⁻¹}
    fn start(&mut self) -> Result<f64, Self::E>;
    /// scale the current vector (q_j, and p_j) by c
    fn scale(&mut self, c: f64) -> Result<(), Self::E>;
    /// w = A q_j; return Re q_jᴴ w = ⟨T q_j, q_j⟩_{M⁻¹}
    fn apply(&mut self) -> Result<f64, Self::E>;
    /// form the unnormalised next vector from w with the coefficients α_j, β_j, shift q_j → q_{j−1}; return
    /// ‖q_{j+1}‖²_{M⁻¹} (before normalisation)
    fn advance(&mut self, alpha: f64, beta: f64) -> Result<f64, Self::E>;
    /// called with the top Ritz vector's coefficients s (in the basis q_0 … q_{k−1}) after the run; a space that kept
    /// its basis can form the Ritz vector for a warm start
    fn ritz(&mut self, _s: &[f64]) -> Result<(), Self::E> {
        Ok(())
    }
}

/// Run `k` Lanczos steps (k operator applications, k + 1 metric applications in the p-form)
pub fn lanczos<S: LanczosSpace>(sp: &mut S, k: usize) -> Result<LamEst, S::E> {
    let n2 = sp.start()?;
    sp.scale(1.0 / n2.max(1e-300).sqrt())?;
    let (mut al, mut be) = (Vec::with_capacity(k), Vec::with_capacity(k));
    let mut b = 0.0;
    for _ in 0..k {
        let a = sp.apply()?;
        al.push(a);
        let b2 = sp.advance(a, b)?;
        b = b2.max(0.0).sqrt();
        be.push(b);
        // invariant subspace found (exact for the start vector): the Ritz values are eigenvalues
        if b <= 1e-7 * a.abs().max(1e-300) {
            break;
        }
        sp.scale(1.0 / b)?;
    }
    let (theta, s) = top_ritz(&al, &be[..al.len() - 1]);
    let bound = be[al.len() - 1] * s[al.len() - 1].abs();
    sp.ritz(&s)?;
    Ok(LamEst { theta, bound, applies: al.len() })
}

/// largest eigenvalue and its unit eigenvector of the real symmetric tridiagonal matrix (diag `a`, off-diag `b`),
/// by cyclic Jacobi on the dense k×k matrix (k ≤ a few tens)
pub fn top_ritz(a: &[f64], b: &[f64]) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut m = vec![0.0f64; n * n];
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = a[i];
        v[i * n + i] = 1.0;
        if i + 1 < n {
            m[i * n + i + 1] = b[i];
            m[(i + 1) * n + i] = b[i];
        }
    }
    let scale = m.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-300);
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| m[i * n + j] * m[i * n + j]).sum();
        if off.sqrt() <= 1e-15 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let (app, aqq) = (m[p * n + p], m[q * n + q]);
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for r in 0..n {
                    let (mrp, mrq) = (m[r * n + p], m[r * n + q]);
                    m[r * n + p] = c * mrp - s * mrq;
                    m[r * n + q] = s * mrp + c * mrq;
                }
                for r in 0..n {
                    let (mpr, mqr) = (m[p * n + r], m[q * n + r]);
                    m[p * n + r] = c * mpr - s * mqr;
                    m[q * n + r] = s * mpr + c * mqr;
                }
                for r in 0..n {
                    let (vrp, vrq) = (v[r * n + p], v[r * n + q]);
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    let i = (0..n).max_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j])).unwrap();
    let mut s: Vec<f64> = (0..n).map(|r| v[r * n + i]).collect();
    let nn = s.iter().map(|x| x * x).sum::<f64>().sqrt();
    s.iter_mut().for_each(|x| *x /= nn);
    (m[i * n + i], s)
}

#[cfg(test)]
mod tests {
    use super::*;

    type Mat = Vec<f64>;

    fn rng(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
    fn mv(a: &Mat, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect()
    }
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn mm(a: &Mat, b: &Mat, n: usize) -> Mat {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
        c
    }
    fn tr(a: &Mat, n: usize) -> Mat {
        (0..n * n).map(|i| a[(i % n) * n + i / n]).collect()
    }
    /// dense symmetric λ_max by Jacobi (reuse top_ritz on a dense matrix: run it through the tridiagonal path is not
    /// possible, so a small dense Jacobi here)
    fn eig_max(a: &Mat, n: usize) -> f64 {
        let mut m = a.clone();
        for _ in 0..200 {
            for p in 0..n {
                for q in p + 1..n {
                    let apq = m[p * n + q];
                    if apq.abs() < 1e-300 {
                        continue;
                    }
                    let tau = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                    let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    for r in 0..n {
                        let (x, y) = (m[r * n + p], m[r * n + q]);
                        m[r * n + p] = c * x - s * y;
                        m[r * n + q] = s * x + c * y;
                    }
                    for r in 0..n {
                        let (x, y) = (m[p * n + r], m[q * n + r]);
                        m[p * n + r] = c * x - s * y;
                        m[q * n + r] = s * x + c * y;
                    }
                }
            }
        }
        (0..n).map(|i| m[i * n + i]).fold(f64::MIN, f64::max)
    }
    /// A = Bᵀ B (PSD, spread spectrum), M = C Cᵀ (SPD), C lower triangular
    fn problem(n: usize, seed: u64) -> (Mat, Mat, Mat) {
        let mut s = seed;
        let mut b: Mat = (0..n * n).map(|_| rng(&mut s)).collect();
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] *= 1.0 + 3.0 * (i as f64 / n as f64).powi(2);
            }
        }
        let a = mm(&tr(&b, n), &b, n);
        let mut c: Mat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                c[i * n + j] = if i == j { 0.5 + rng(&mut s).abs() * 2.0 } else { 0.3 * rng(&mut s) };
            }
        }
        let m = mm(&c, &tr(&c, n), n);
        (a, m, c)
    }

    struct P<'a> {
        a: &'a Mat,
        m: Option<&'a Mat>,
        pp: Vec<f64>,
        p: Vec<f64>,
        q: Vec<f64>,
        w: Vec<f64>,
    }
    impl LanczosSpace for P<'_> {
        type E = ();
        fn start(&mut self) -> Result<f64, ()> {
            let n = self.p.len();
            self.p = (0..n).map(|i| 1.0 + 0.1 * (i % 7) as f64).collect();
            self.q = self.m.map_or(self.p.clone(), |m| mv(m, &self.p));
            Ok(dot(&self.p, &self.q))
        }
        fn scale(&mut self, c: f64) -> Result<(), ()> {
            self.p.iter_mut().chain(self.q.iter_mut()).for_each(|x| *x *= c);
            Ok(())
        }
        fn apply(&mut self) -> Result<f64, ()> {
            self.w = mv(self.a, &self.q);
            Ok(dot(&self.q, &self.w))
        }
        fn advance(&mut self, al: f64, be: f64) -> Result<f64, ()> {
            let s: Vec<f64> = (0..self.w.len()).map(|i| self.w[i] - al * self.p[i] - be * self.pp[i]).collect();
            let r = self.m.map_or(s.clone(), |m| mv(m, &s));
            self.pp = std::mem::replace(&mut self.p, s);
            self.q = r;
            Ok(dot(&self.p, &self.q))
        }
    }

    struct Q<'a> {
        a: &'a Mat,
        m: Option<&'a Mat>,
        minv: Option<Mat>,
        qp: Vec<f64>,
        q: Vec<f64>,
        w: Vec<f64>,
    }
    impl Q<'_> {
        fn n2(&self, x: &[f64]) -> f64 {
            self.minv.as_ref().map_or(dot(x, x), |mi| dot(x, &mv(mi, x)))
        }
    }
    impl LanczosSpace for Q<'_> {
        type E = ();
        fn start(&mut self) -> Result<f64, ()> {
            let n = self.q.len();
            // the same start as the p-form (q_0 = M p_0)
            let p0: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * (i % 7) as f64).collect();
            self.q = self.m.map_or(p0.clone(), |m| mv(m, &p0));
            Ok(self.n2(&self.q))
        }
        fn scale(&mut self, c: f64) -> Result<(), ()> {
            self.q.iter_mut().for_each(|x| *x *= c);
            Ok(())
        }
        fn apply(&mut self) -> Result<f64, ()> {
            self.w = mv(self.a, &self.q);
            Ok(dot(&self.q, &self.w))
        }
        fn advance(&mut self, al: f64, be: f64) -> Result<f64, ()> {
            let u = self.m.map_or(self.w.clone(), |m| mv(m, &self.w));
            let r: Vec<f64> = (0..u.len()).map(|i| u[i] - al * self.q[i] - be * self.qp[i]).collect();
            self.qp = std::mem::replace(&mut self.q, r);
            Ok(self.n2(&self.q))
        }
    }

    fn inv_spd(m: &Mat, n: usize) -> Mat {
        // Gauss–Jordan
        let mut a = m.clone();
        let mut b: Mat = (0..n * n).map(|i| (i / n == i % n) as u8 as f64).collect();
        for c in 0..n {
            let d = a[c * n + c];
            for j in 0..n {
                a[c * n + j] /= d;
                b[c * n + j] /= d;
            }
            for r in 0..n {
                if r != c {
                    let f = a[r * n + c];
                    for j in 0..n {
                        a[r * n + j] -= f * a[c * n + j];
                        b[r * n + j] -= f * b[c * n + j];
                    }
                }
            }
        }
        b
    }

    #[test]
    fn ritz_of_tridiagonal() {
        // T = tridiag(1, 2, 1) of size 5: eigenvalues 2 + 2 cos(kπ/6)
        let (th, s) = top_ritz(&[2.0; 5], &[1.0; 4]);
        assert!((th - (2.0 + 2.0 * (std::f64::consts::PI / 6.0).cos())).abs() < 1e-12, "{th}");
        assert!((s.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lanczos_matches_eig_plain_and_metric() {
        let n = 40;
        for seed in [7u64, 99, 12345] {
            let (a, m, c) = problem(n, seed);
            // reference: λ_max(A) and λ_max(M A) = λ_max(Cᵀ A C)
            let lam_a = eig_max(&a, n);
            let lam_ma = eig_max(&mm(&tr(&c, n), &mm(&a, &c, n), n), n);
            let z = vec![0.0; n];
            for k in [3usize, 6, 10, 20] {
                let mut pa = P { a: &a, m: None, pp: z.clone(), p: z.clone(), q: z.clone(), w: z.clone() };
                let mut pm = P { a: &a, m: Some(&m), pp: z.clone(), p: z.clone(), q: z.clone(), w: z.clone() };
                let mut qm = Q { a: &a, m: Some(&m), minv: Some(inv_spd(&m, n)), qp: z.clone(), q: z.clone(), w: z.clone() };
                let ea = lanczos(&mut pa, k).unwrap();
                let em = lanczos(&mut pm, k).unwrap();
                let eq = lanczos(&mut qm, k).unwrap();
                eprintln!("seed {seed} k {k}: A θ/λ {:.6} (θ+b)/λ {:.6} | MA θ/λ {:.6} (θ+b)/λ {:.6} q-form θ/λ {:.6}", ea.theta / lam_a, ea.hat() / lam_a, em.theta / lam_ma, em.hat() / lam_ma, eq.theta / lam_ma);
                // Rayleigh–Ritz: never above λ_max; the residual bound brackets it
                assert!(ea.theta <= lam_a * (1.0 + 1e-10) && em.theta <= lam_ma * (1.0 + 1e-10));
                // the residual interval holds an eigenvalue; once the top Ritz value has separated it is λ_max
                if k >= 10 {
                    assert!(ea.hat() >= lam_a * (1.0 - 1e-9) && em.hat() >= lam_ma * (1.0 - 1e-9), "bound misses λ_max");
                }
                // both forms are the same recurrence
                assert!((em.theta - eq.theta).abs() <= 1e-8 * lam_ma, "{} vs {}", em.theta, eq.theta);
                if k == 20 {
                    assert!((em.theta / lam_ma - 1.0).abs() < 1e-6 && (ea.theta / lam_a - 1.0).abs() < 1e-6);
                }
            }
        }
    }
}
