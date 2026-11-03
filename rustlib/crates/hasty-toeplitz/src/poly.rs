//! Tiny sparse multivariate polynomial algebra with complex coefficients (used to derive the Taylor / commutator-Weyl expansions symbolically).

use std::collections::BTreeMap;

use num_complex::Complex64;

pub type Mono = Vec<u8>;

#[derive(Clone, Debug, Default)]
pub struct Poly {
    pub nv: usize,
    pub t: BTreeMap<Mono, Complex64>,
}

impl Poly {
    pub fn zero(nv: usize) -> Self {
        Self {
            nv,
            t: BTreeMap::new(),
        }
    }
    pub fn constant(nv: usize, c: Complex64) -> Self {
        let mut p = Self::zero(nv);
        p.t.insert(vec![0; nv], c);
        p
    }
    pub fn var(nv: usize, i: usize) -> Self {
        let mut p = Self::zero(nv);
        let mut m = vec![0u8; nv];
        m[i] = 1;
        p.t.insert(m, Complex64::new(1.0, 0.0));
        p
    }
    pub fn add_assign(&mut self, o: &Poly) {
        for (m, c) in &o.t {
            *self.t.entry(m.clone()).or_insert(Complex64::new(0.0, 0.0)) += *c;
        }
    }
    pub fn scale(&self, c: Complex64) -> Poly {
        Poly {
            nv: self.nv,
            t: self.t.iter().map(|(m, v)| (m.clone(), v * c)).collect(),
        }
    }
    pub fn mul(&self, o: &Poly) -> Poly {
        let mut r = Poly::zero(self.nv);
        for (m1, c1) in &self.t {
            for (m2, c2) in &o.t {
                let m: Mono = m1.iter().zip(m2).map(|(a, b)| a + b).collect();
                *r.t.entry(m).or_insert(Complex64::new(0.0, 0.0)) += c1 * c2;
            }
        }
        r
    }
    pub fn pow(&self, n: usize) -> Poly {
        let mut r = Poly::constant(self.nv, Complex64::new(1.0, 0.0));
        for _ in 0..n {
            r = r.mul(self);
        }
        r
    }
    pub fn prune(&mut self, eps: f64) {
        self.t.retain(|_, c| c.norm() > eps);
    }
}

pub fn factorial(n: usize) -> f64 {
    (1..=n).map(|x| x as f64).product()
}

pub fn binom(n: usize, k: usize) -> f64 {
    factorial(n) / (factorial(k) * factorial(n - k))
}
