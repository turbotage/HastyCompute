//! Velocity decoding of the encoding schemes, coprime dual-VENC unwrapping, and the noise model.
//!
//! Every scheme is decoded from phase-difference *pairs* (plus, minus) of encodes: φ = arg(s₊ s₋*) = g·v wrapped to
//! (−π, π], with sensitivity g = γ (M1₊ − M1₋) (rad per m/s), computed from the actual first moments. Three pairs with
//! independent directions ĉ give the velocity. Referenced / seven / eight layouts have ĉ along the axes; Hadamard
//! pairs measure v_x + v_y, v_x + v_z, v_y + v_z (so each pair aliases when |v_i + v_j| exceeds its VENC).
//!
//! Dual VENC: each direction ĉ is measured by a low- and a high-VENC pair with parallel sensitivities, |g_lo| / |g_hi|
//! = b / a. For coprime a, b (Beghella Bartoli et al., JCMR 2025: a = p, b = p + 1), with VENC_lo = a·u,
//! VENC_hi = b·u the wrapped phases determine ĉ·v modulo 2a·u and 2b·u, hence modulo 2ab·u (Chinese remainder): the
//! unaliased range is ±ab·u = ±a·VENC_hi = ±b·VENC_lo. With φ_lo = π v/VENC_lo − 2π n_lo, φ_hi = π v/VENC_hi − 2π n_hi,
//!   m = nint((a φ_lo − b φ_hi) / 2π) = b n_hi − a n_lo,
//! and (n_hi, n_lo) follow from m through a table (unique inside the range; the paper's Eq. 4 and Table 1). The two
//! unwrapped estimates are then combined by generalised least squares (inverse covariance, including the correlation
//! of a shared reference).
//! Unwrap failure: the noise of (a φ_lo − b φ_hi)/2π has σ_m = σ_d √(a² + b² − 2ab ρ) / 2π (σ_d: phase-difference
//! noise, ρ: correlation of the two pairs, ½ with a shared reference), and m is wrong when it exceeds ½:
//! P ≈ erfc(1 / (2√2 σ_m)) (small-noise); `unwrap_failure_exact` uses the exact phase-noise distribution instead
//! (≈ 3× higher around SNR 3). The paper has no analytic bound; this sets how far p can be pushed at a given SNR.

use std::collections::HashMap;

use crate::encode::{scheme_m1, Scheme};
use crate::{dot, norm, Vec3, GAMMA};

#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub plus: usize,
    pub minus: usize,
    /// phase(plus) − phase(minus) = g·v, rad per m/s
    pub g: Vec3,
}

/// Chinese-remainder unwrapping for VENC_lo / VENC_hi = a / b, gcd(a, b) = 1
#[derive(Clone, Debug)]
pub struct Crt {
    pub a: i64,
    pub b: i64,
    table: HashMap<i64, (i64, i64)>,
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

impl Crt {
    pub fn new(a: i64, b: i64) -> Crt {
        assert!(a > 0 && b > a && gcd(a, b) == 1, "need 0 < a < b coprime");
        // all (n_hi, n_lo) whose velocity intervals overlap inside (−ab, ab] (velocities in units of u)
        let mut table = HashMap::new();
        let (lo, hi) = (a as f64, b as f64);
        for nh in -(a + 1)..=(a + 1) {
            for nl in -(b + 1)..=(b + 1) {
                // v ∈ (2 nh hi − hi, 2 nh hi + hi] ∩ (2 nl lo − lo, 2 nl lo + lo] ∩ (−ab, ab]
                let l = (2 * nh - 1) as f64 * hi;
                let l = l.max((2 * nl - 1) as f64 * lo).max(-(a * b) as f64);
                let r = ((2 * nh + 1) as f64 * hi).min((2 * nl + 1) as f64 * lo).min((a * b) as f64);
                if r > l {
                    let m = b * nh - a * nl;
                    let prev = table.insert(m, (nh, nl));
                    assert!(prev.is_none() || prev == Some((nh, nl)), "CRT table not unique");
                }
            }
        }
        Crt { a, b, table }
    }
    /// Nyquist numbers (n_hi, n_lo) for the wrapped phases; None if m falls outside the table (noise beyond the range)
    pub fn wraps(&self, phi_lo: f64, phi_hi: f64) -> Option<(i64, i64)> {
        let m = ((self.a as f64 * phi_lo - self.b as f64 * phi_hi) / (2.0 * std::f64::consts::PI)).round() as i64;
        self.table.get(&m).copied()
    }
    /// the values m can take
    pub fn len(&self) -> usize {
        self.table.len()
    }
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

/// a direction measured once (single VENC) or by a low/high pair (dual VENC)
#[derive(Clone, Debug)]
pub struct Combo {
    /// unit direction ĉ: the pair measures ĉ·v
    pub c: Vec3,
    pub lo: Pair,
    pub hi: Option<Pair>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub combos: Vec<Combo>,
    pub crt: Option<Crt>,
    /// correlation of the low and high pair of a combo (½: shared minus encode)
    pub rho: f64,
}

/// x = b / a > 1 with small coprime integers, if it is one
fn rational(x: f64) -> Option<(i64, i64)> {
    for a in 1..=64i64 {
        let b = (x * a as f64).round() as i64;
        if b > a && ((b as f64 / a as f64) - x).abs() < 1e-9 * x {
            let g = gcd(a, b);
            return Some((a / g, b / g));
        }
    }
    None
}

fn pair(m1: &[Vec3], plus: usize, minus: usize) -> Pair {
    Pair { plus, minus, g: [0, 1, 2].map(|q| GAMMA * (m1[plus][q] - m1[minus][q])) }
}

/// decoding plan of a scheme (from its first moments; `balanced` does not change the pairs' sensitivities)
pub fn plan(scheme: Scheme, venc: f64) -> Plan {
    use crate::encode::DualLayout::*;
    let m1 = scheme_m1(scheme, venc, false);
    let (lo, hi): (Vec<(usize, usize)>, Vec<(usize, usize)>) = match scheme {
        Scheme::Referenced => (vec![(1, 0), (2, 0), (3, 0)], vec![]),
        Scheme::Hadamard => (vec![(1, 0), (2, 0), (3, 0)], vec![]),
        Scheme::DualVenc { .. } | Scheme::Coprime { layout: Seven, .. } => (vec![(1, 0), (3, 0), (5, 0)], vec![(2, 0), (4, 0), (6, 0)]),
        Scheme::Coprime { layout: Eight | EightHadamard, .. } => (vec![(1, 0), (2, 0), (3, 0)], vec![(5, 4), (6, 4), (7, 4)]),
    };
    let mut combos = vec![];
    for (i, &(p, q)) in lo.iter().enumerate() {
        let l = pair(&m1, p, q);
        let c = crate::scale(&l.g, 1.0 / norm(&l.g));
        let h = hi.get(i).map(|&(p, q)| pair(&m1, p, q));
        if let Some(h) = &h {
            let ch = crate::scale(&h.g, 1.0 / norm(&h.g));
            assert!((dot(&c, &ch) - 1.0).abs() < 1e-12, "low / high pairs not parallel");
        }
        combos.push(Combo { c, lo: l, hi: h });
    }
    // a = 1 is the classic dual VENC (range VENC_hi); a > 1 coprime extends it to a·VENC_hi; no small rational ratio:
    // classic unwrapping of the low by the high estimate
    let crt = combos[0].hi.as_ref().and_then(|h| rational(norm(&combos[0].lo.g) / norm(&h.g)).map(|(a, b)| Crt::new(a, b)));
    let rho = match &combos[0].hi {
        Some(h) if h.minus == combos[0].lo.minus => 0.5,
        _ => 0.0,
    };
    Plan { combos, crt, rho }
}

#[derive(Clone, Debug)]
pub struct Decoded {
    pub v: Vec3,
    /// every combo's Nyquist numbers were found
    pub ok: bool,
}

fn solve3(a: [Vec3; 3], y: Vec3) -> Vec3 {
    let det = dot(&a[0], &crate::cross(&a[1], &a[2]));
    let inv = [crate::cross(&a[1], &a[2]), crate::cross(&a[2], &a[0]), crate::cross(&a[0], &a[1])];
    // x = Σ y_i · (row-dual vectors) — A x = y with rows a_i: x = (y0 (a1×a2) + y1 (a2×a0) + y2 (a0×a1)) / det
    [0, 1, 2].map(|q| (y[0] * inv[0][q] + y[1] * inv[1][q] + y[2] * inv[2][q]) / det)
}

fn wrap(x: f64) -> f64 {
    let tp = 2.0 * std::f64::consts::PI;
    x - tp * ((x + std::f64::consts::PI) / tp).floor()
}

/// GLS weight of the low estimate (variances ∝ 1/|g|², correlation ρ)
pub fn lo_weight(g_lo: f64, g_hi: f64, rho: f64) -> f64 {
    let (sl, sh) = (1.0 / g_lo, 1.0 / g_hi);
    (sh * sh - rho * sl * sh) / (sl * sl + sh * sh - 2.0 * rho * sl * sh)
}

impl Plan {
    /// velocity from the per-encode phases (any common offset)
    pub fn decode(&self, phase: &[f64]) -> Decoded {
        let mut ok = true;
        let mut y = [0.0; 3];
        for (i, cb) in self.combos.iter().enumerate() {
            let pl = wrap(phase[cb.lo.plus] - phase[cb.lo.minus]);
            let gl = norm(&cb.lo.g);
            y[i] = match (&cb.hi, &self.crt) {
                (None, _) => pl / gl,
                (Some(h), Some(crt)) => {
                    let ph = wrap(phase[h.plus] - phase[h.minus]);
                    let gh = norm(&h.g);
                    let tp = 2.0 * std::f64::consts::PI;
                    match crt.wraps(pl, ph) {
                        Some((nh, nl)) => {
                            let w = lo_weight(gl, gh, self.rho);
                            w * (pl + tp * nl as f64) / gl + (1.0 - w) * (ph + tp * nh as f64) / gh
                        }
                        None => {
                            ok = false;
                            ph / gh
                        }
                    }
                }
                (Some(h), None) => {
                    let ph = wrap(phase[h.plus] - phase[h.minus]);
                    let gh = norm(&h.g);
                    let tp = 2.0 * std::f64::consts::PI;
                    let nl = ((ph / gh * gl - pl) / tp).round();
                    let w = lo_weight(gl, gh, self.rho);
                    w * (pl + tp * nl) / gl + (1.0 - w) * ph / gh
                }
            };
        }
        let a = [self.combos[0].c, self.combos[1].c, self.combos[2].c];
        Decoded { v: solve3(a, y), ok }
    }
    /// standard deviation of each velocity component (m/s) for independent phase noise σ_e per encode (≈ 1/SNR),
    /// by linear propagation through `decode` (unwrapping correct)
    pub fn velocity_std(&self, n_enc: usize, sigma_e: f64) -> Vec3 {
        let base = vec![0.0; n_enc];
        let v0 = self.decode(&base).v;
        let h = 1e-6;
        let mut var = [0.0; 3];
        for e in 0..n_enc {
            let mut p = base.clone();
            p[e] = h;
            let v = self.decode(&p).v;
            for q in 0..3 {
                var[q] += ((v[q] - v0[q]) / h * sigma_e).powi(2);
            }
        }
        var.map(f64::sqrt)
    }
    /// probability that at least one combo unwraps wrongly (independent combos assumed), phase noise σ_e per encode
    pub fn unwrap_failure(&self, sigma_e: f64) -> f64 {
        let Some(crt) = &self.crt else { return 0.0 };
        let (a, b) = (crt.a as f64, crt.b as f64);
        let sd = sigma_e * 2f64.sqrt();
        let sm = sd * (a * a + b * b - 2.0 * a * b * self.rho).sqrt() / (2.0 * std::f64::consts::PI);
        let p1 = erfc(1.0 / (2.0 * 2f64.sqrt() * sm));
        1.0 - (1.0 - p1).powi(self.combos.len() as i32)
    }
}

impl Plan {
    /// integer coefficient of each encode's phase noise in the CRT statistic a·φ_lo − b·φ_hi of combo `i`
    fn crt_coefficients(&self, i: usize, n_enc: usize) -> Vec<i64> {
        let crt = self.crt.as_ref().expect("no CRT");
        let cb = &self.combos[i];
        let h = cb.hi.as_ref().unwrap();
        let mut c = vec![0i64; n_enc];
        c[cb.lo.plus] += crt.a;
        c[cb.lo.minus] -= crt.a;
        c[h.plus] -= crt.b;
        c[h.minus] += crt.b;
        c
    }
    /// unwrap-failure probability with the exact phase-noise distribution of a complex Gaussian (signal 1, complex
    /// noise std `sigma` = 1/SNR): P(|Σ c_e δφ_e| > π) per combo by numerical convolution, union over the combos
    /// (treated as independent). The small-noise `unwrap_failure` underestimates this ~3× around SNR 3: the phase
    /// noise has heavier tails than a Gaussian.
    pub fn unwrap_failure_exact(&self, n_enc: usize, sigma: f64) -> f64 {
        if self.crt.is_none() {
            return 0.0;
        }
        let rho = 1.0 / sigma;
        let pdf = |t: f64| {
            let c = t.cos();
            let rc = rho * c;
            (-(rho * rho)).exp() / (2.0 * std::f64::consts::PI) * (1.0 + std::f64::consts::PI.sqrt() * rc * (rc * rc).exp() * (2.0 - erfc(rc)))
        };
        let pi = std::f64::consts::PI;
        let nh = 384usize; // grid points per π of the sum
        let h = pi / nh as f64;
        let mut p_ok = 1.0;
        for i in 0..self.combos.len() {
            let coef = self.crt_coefficients(i, n_enc);
            let mut dist = vec![1.0f64]; // mass at 0
            for &c in coef.iter().filter(|&&c| c != 0) {
                let ca = c.unsigned_abs() as usize;
                // c·θ on the grid j·h, j ∈ [−ca·nh, ca·nh]: mass of θ in the bin ((j − ½) h / ca, (j + ½) h / ca]
                let m = ca * nh;
                let sub = 8;
                let mass: Vec<f64> = (0..=2 * m)
                    .map(|k| {
                        let j = k as f64 - m as f64;
                        let (l, r) = (((j - 0.5) * h / ca as f64).max(-pi), ((j + 0.5) * h / ca as f64).min(pi));
                        if r <= l {
                            return 0.0;
                        }
                        let w = (r - l) / sub as f64;
                        (0..sub).map(|q| pdf(l + (q as f64 + 0.5) * w) * w).sum()
                    })
                    .collect();
                let mut out = vec![0.0; dist.len() + mass.len() - 1];
                for (x, &dx) in dist.iter().enumerate() {
                    if dx < 1e-300 {
                        continue;
                    }
                    for (y, &my) in mass.iter().enumerate() {
                        out[x + y] += dx * my;
                    }
                }
                dist = out;
            }
            let centre = (dist.len() - 1) / 2;
            let total: f64 = dist.iter().sum();
            let inside: f64 = dist.iter().enumerate().filter(|(k, _)| ((*k as f64 - centre as f64) * h).abs() <= pi).map(|(_, v)| v).sum();
            p_ok *= inside / total;
        }
        1.0 - p_ok
    }
}

/// complementary error function (Numerical Recipes erfcc, fractional error < 1.2e-7)
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.265_512_23 + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::DualLayout;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn n(&mut self) -> f64 {
            let (a, b) = (self.u().max(1e-300), self.u());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
    }

    #[test]
    fn crt_table_matches_the_paper() {
        // p = 3 (VENC_lo / VENC_hi = 3/4): Table 1 of Beghella Bartoli et al., m = 4 n_hi − 3 n_lo
        let c = Crt::new(3, 4);
        assert_eq!(c.len(), 7);
        for (m, nh, nl) in [(-3, 0, 1), (-2, 1, 2), (-1, -1, -1), (0, 0, 0), (1, 1, 1), (2, -1, -2), (3, 0, -1)] {
            assert_eq!(c.table[&m], (nh, nl), "m = {m}");
        }
    }

    fn phases(m1: &[Vec3], v: &Vec3) -> Vec<f64> {
        m1.iter().map(|m| GAMMA * dot(m, v)).collect()
    }

    #[test]
    fn noiseless_decoding_over_the_extended_range() {
        let venc = 1.0;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for scheme in [Scheme::Referenced, Scheme::Hadamard, Scheme::Coprime { p: 2, layout: DualLayout::Seven }, Scheme::Coprime { p: 3, layout: DualLayout::Eight }, Scheme::Coprime { p: 2, layout: DualLayout::EightHadamard }] {
            let pl = plan(scheme, venc);
            let m1 = scheme_m1(scheme, venc, true);
            let lim = scheme.unaliased(venc);
            for _ in 0..2000 {
                // inside the unaliased region of every combo: |ĉ·v| < lim·|ĉ|₁ scale (ĉ unit, Hadamard ĉ = (1,1,0)/√2 with
                // VENC along ĉ = lim/√2 … so keep each component below lim/2 for Hadamard, lim otherwise)
                let s = if matches!(scheme, Scheme::Hadamard) || matches!(scheme, Scheme::Coprime { layout: DualLayout::EightHadamard, .. }) { 0.5 } else { 1.0 };
                let v = [0, 1, 2].map(|_| (2.0 * rng.u() - 1.0) * lim * s * 0.999);
                let d = pl.decode(&phases(&m1, &v));
                assert!(d.ok);
                for q in 0..3 {
                    assert!((d.v[q] - v[q]).abs() < 1e-9, "{scheme:?} v {v:?} → {:?}", d.v);
                }
            }
        }
    }

    #[test]
    fn noise_matches_the_model() {
        // complex Gaussian noise (σ/√2 per channel, unit signal): velocity std at high SNR against linear propagation,
        // CRT failure rate at low SNR against the erfc model
        let mut rng = Rng(12345);
        let cases = [
            (Scheme::Coprime { p: 2, layout: DualLayout::Seven }, 2.5),
            (Scheme::Coprime { p: 3, layout: DualLayout::Eight }, 4.5),
            (Scheme::Coprime { p: 2, layout: DualLayout::EightHadamard }, 3.0),
        ];
        for (scheme, snr_fail) in cases {
            let venc = 0.6;
            let pl = plan(scheme, venc);
            let m1 = scheme_m1(scheme, venc, true);
            let ne = m1.len();
            for (snr, check_std) in [(30.0, true), (snr_fail, false), (snr_fail * 1.3, false)] {
                let sig = 1.0 / snr;
                let n = 100_000;
                let (mut fail, mut good, mut se) = (0usize, 0usize, [0.0; 3]);
                for _ in 0..n {
                    let s = if matches!(scheme, Scheme::Coprime { layout: DualLayout::EightHadamard, .. }) { 0.2 } else { 0.4 };
                    let v = [0, 1, 2].map(|_| (2.0 * rng.u() - 1.0) * s * scheme.unaliased(venc));
                    let ph: Vec<f64> = phases(&m1, &v)
                        .iter()
                        .map(|p| {
                            let (re, im) = (p.cos() + sig / 2f64.sqrt() * rng.n(), p.sin() + sig / 2f64.sqrt() * rng.n());
                            im.atan2(re)
                        })
                        .collect();
                    let d = pl.decode(&ph);
                    let err = [0, 1, 2].map(|q| d.v[q] - v[q]);
                    // a wrong Nyquist number moves the estimate by ≳ VENC_lo
                    if !d.ok || err.iter().any(|e| e.abs() > 0.5 * venc) {
                        fail += 1;
                    } else {
                        good += 1;
                        for q in 0..3 {
                            se[q] += err[q] * err[q];
                        }
                    }
                }
                let sigma_e = sig / 2f64.sqrt();
                if check_std {
                    let std = se.map(|s| (s / good as f64).sqrt());
                    let model = pl.velocity_std(ne, sigma_e);
                    for q in 0..3 {
                        assert!((std[q] / model[q] - 1.0).abs() < 0.05, "{scheme:?}: std {std:?} vs model {model:?}");
                    }
                    assert_eq!(fail, 0);
                } else {
                    let pg = pl.unwrap_failure(sigma_e);
                    let pf = pl.unwrap_failure_exact(ne, sig);
                    let rate = fail as f64 / n as f64;
                    println!("{scheme:?} SNR {snr}: failure {rate:.5}, exact model {pf:.5}, gaussian {pg:.5}");
                    assert!(pf > 1e-4 && rate < 1.3 * pf + 1e-4 && rate > 0.75 * pf - 1e-4, "{scheme:?}: failure {rate} vs model {pf}");
                }
            }
        }
    }
}
