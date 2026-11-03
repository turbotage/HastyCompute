//! Mixing specifications: how a normal operator is assembled from pre-diagonals, Toeplitz kernels and post-diagonals.
//!
//!   N x = sum_{(o,i,k,c) in entries}  post_o . T_k ( pre_i . x ) * c
//! where `T_k` is the Toeplitz operator with k-space multiplier `K_k` (circulant embedding on the padded grid).
//! All expansions in this crate (plain, Taylor-Hankel, commutator/Weyl, Tucker, CP) are instances.

use std::collections::BTreeMap;

use hasty_fft::PolyTerm;
use num_complex::Complex64;

use crate::fields::{m_poly, NQ};
use crate::poly::{binom, factorial, Poly};

/// Which sample-dependent weights and which kernel-domain polynomial define one Toeplitz kernel:
/// `H(D) = sum_j wt_j * w_j^mu * theta_j * (D/(N/2))^gamma * exp(+i k_j . D)`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelKey {
    /// exponents of the 4 concomitant coefficients
    pub mu: [u8; NQ],
    /// exponents of the normalised kernel-domain coordinates (x, y, z)
    pub gamma: [u8; 3],
    /// extra user weight set (index into the caller-supplied weight table), `None` = only wt_j
    pub extra: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum FieldDiag {
    None,
    /// real polynomial in the normalised voxel coordinates
    Poly(Vec<PolyTerm>),
    /// caller-supplied complex array `index` (post: conjugated)
    Array(usize),
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub out: usize,
    pub inp: usize,
    pub kernel: usize,
    pub coef: Complex64,
    /// use conj(K_k) instead of K_k (Hermitian-pair kernel sharing)
    pub conj: bool,
}

#[derive(Clone, Debug)]
pub struct MixSpec {
    pub pre: Vec<FieldDiag>,
    pub post: Vec<FieldDiag>,
    pub kernels: Vec<KernelKey>,
    pub entries: Vec<Entry>,
    pub label: String,
}

impl MixSpec {
    pub fn nfield(&self) -> usize {
        self.pre.len()
    }
    /// number of FFT pairs (= number of fields) and of distinct kernels
    pub fn cost(&self) -> (usize, usize) {
        (self.nfield(), self.kernels.len())
    }
}

fn poly_to_terms(p: &Poly) -> Vec<PolyTerm> {
    p.t.iter()
        .filter(|(_, c)| c.norm() > 1e-14)
        .map(|(m, c)| PolyTerm {
            coef: c.re,
            p: [m[0], m[1], m[2]],
        })
        .collect()
}

/// all multi-indices of dimension `d` with total degree <= `n`, graded order
pub fn multi_indices(d: usize, n: usize) -> Vec<Vec<u8>> {
    fn rec(d: usize, left: usize, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        if cur.len() == d {
            out.push(cur.clone());
            return;
        }
        for v in 0..=left {
            cur.push(v as u8);
            rec(d, left - v, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(d, n, &mut Vec::new(), &mut out);
    out.sort_by_key(|m| (m.iter().map(|&x| x as usize).sum::<usize>(), m.clone()));
    out
}

struct KernelTable {
    map: BTreeMap<KernelKey, usize>,
    keys: Vec<KernelKey>,
}
impl KernelTable {
    fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            keys: vec![],
        }
    }
    fn id(&mut self, k: KernelKey) -> usize {
        if let Some(&i) = self.map.get(&k) {
            return i;
        }
        let i = self.keys.len();
        self.map.insert(k.clone(), i);
        self.keys.push(k);
        i
    }
}

/// `T0`: ignore the concomitant phase.
pub fn plain() -> MixSpec {
    MixSpec {
        pre: vec![FieldDiag::None],
        post: vec![FieldDiag::None],
        kernels: vec![KernelKey {
            mu: [0; NQ],
            gamma: [0; 3],
            extra: None,
        }],
        entries: vec![Entry {
            out: 0,
            inp: 0,
            kernel: 0,
            coef: Complex64::new(1.0, 0.0),
            conj: false,
        }],
        label: "plain".into(),
    }
}

/// Taylor expansion of `e^{-i Phi}` to total order `p` in the fields (multinomial / Hankel form):
/// `N = sum_{a,a'} i^{|a|} (-i)^{|a'|} / (a! a'!) F_a . T^{a+a'} . F_{a'}`.
pub fn taylor_hankel(p: usize) -> MixSpec {
    let alphas = multi_indices(NQ, p);
    let one = Complex64::new(1.0, 0.0);
    let ms: Vec<Poly> = (0..NQ).map(m_poly).collect();
    let fields: Vec<FieldDiag> = alphas
        .iter()
        .map(|a| {
            let mut f = Poly::constant(3, one);
            for q in 0..NQ {
                f = f.mul(&ms[q].pow(a[q] as usize));
            }
            if a.iter().all(|&x| x == 0) {
                FieldDiag::None
            } else {
                FieldDiag::Poly(poly_to_terms(&f))
            }
        })
        .collect();
    let mut kt = KernelTable::new();
    let mut entries = Vec::new();
    for (o, a1) in alphas.iter().enumerate() {
        for (i, a2) in alphas.iter().enumerate() {
            let mut mu = [0u8; NQ];
            for q in 0..NQ {
                mu[q] = a1[q] + a2[q];
            }
            let s1: usize = a1.iter().map(|&x| x as usize).sum();
            let s2: usize = a2.iter().map(|&x| x as usize).sum();
            let f1: f64 = a1.iter().map(|&x| factorial(x as usize)).product();
            let f2: f64 = a2.iter().map(|&x| factorial(x as usize)).product();
            let phase = Complex64::new(0.0, 1.0).powi(s1 as i32)
                * Complex64::new(0.0, -1.0).powi(s2 as i32);
            let k = kt.id(KernelKey {
                mu,
                gamma: [0; 3],
                extra: None,
            });
            entries.push(Entry {
                out: o,
                inp: i,
                kernel: k,
                coef: phase / (f1 * f2),
                conj: false,
            });
        }
    }
    MixSpec {
        pre: fields.clone(),
        post: fields,
        kernels: kt.keys,
        entries,
        label: format!("taylor-hankel(P={p})"),
    }
}

/// Commutator / Weyl expansion in the *pair* phase, truncated at order `n`:
/// `e^{i x} ~ sum_{m<=n} (i x)^m / m!`, `x = Phi(r) - Phi(r') = sum_q w_q grad m_q(rbar) . Delta`, `rbar = (r + r')/2`.
/// Image-domain factors are coordinate monomials only; kernels are w-moment kernels with Delta-polynomial weights.
pub fn weyl(n: usize) -> MixSpec {
    // variables: w1..w4 (0..3), rbar_x,y,z (4..6), d_x,y,z (7..9)
    const NV: usize = 10;
    let w = |q: usize| Poly::var(NV, q);
    let rb = |a: usize| Poly::var(NV, 4 + a);
    let d = |a: usize| Poly::var(NV, 7 + a);
    let two = Complex64::new(2.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    // x = w1 2 rz dz + w2 (2 rx dx + 2 ry dy) + w3 (rz dx + rx dz) + w4 (rz dy + ry dz)
    let mut x = Poly::zero(NV);
    x.add_assign(&w(0).mul(&rb(2)).mul(&d(2)).scale(two));
    x.add_assign(&w(1).mul(&rb(0)).mul(&d(0)).scale(two));
    x.add_assign(&w(1).mul(&rb(1)).mul(&d(1)).scale(two));
    x.add_assign(&w(2).mul(&rb(2)).mul(&d(0)));
    x.add_assign(&w(2).mul(&rb(0)).mul(&d(2)));
    x.add_assign(&w(3).mul(&rb(2)).mul(&d(1)));
    x.add_assign(&w(3).mul(&rb(1)).mul(&d(2)));
    let mut series = Poly::zero(NV);
    let ix = x.scale(Complex64::new(0.0, 1.0));
    for m in 0..=n {
        series.add_assign(&ix.pow(m).scale(one / factorial(m)));
    }
    series.prune(1e-14);
    // rbar^b = ((r + r')/2)^b : expand into r^b1 r'^b2 and collect: key (b1, b2, mu, gamma) -> coef
    let mut acc: BTreeMap<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), Complex64> = BTreeMap::new();
    for (mono, c) in &series.t {
        let mu: Vec<u8> = mono[0..4].to_vec();
        let b: Vec<u8> = mono[4..7].to_vec();
        let g: Vec<u8> = mono[7..10].to_vec();
        // product over axes of sum_{k} C(b_a,k) r^k r'^{b_a-k} / 2^{b_a}
        let mut parts: Vec<(Vec<u8>, Vec<u8>, f64)> = vec![(vec![], vec![], 1.0)];
        for a in 0..3 {
            let mut next = Vec::new();
            for (b1, b2, f) in &parts {
                for k in 0..=b[a] as usize {
                    let mut nb1 = b1.clone();
                    nb1.push(k as u8);
                    let mut nb2 = b2.clone();
                    nb2.push(b[a] - k as u8);
                    next.push((
                        nb1,
                        nb2,
                        f * binom(b[a] as usize, k) / 2f64.powi(b[a] as i32),
                    ));
                }
            }
            parts = next;
        }
        for (b1, b2, f) in parts {
            *acc.entry((b1, b2, mu.clone(), g.clone()))
                .or_insert(Complex64::new(0.0, 0.0)) += c * f;
        }
    }
    let monos = multi_indices(3, n);
    let idx = |m: &Vec<u8>| monos.iter().position(|x| x == m).unwrap();
    let fields: Vec<FieldDiag> = monos
        .iter()
        .map(|m| {
            if m.iter().all(|&x| x == 0) {
                FieldDiag::None
            } else {
                FieldDiag::Poly(vec![PolyTerm {
                    coef: 1.0,
                    p: [m[0], m[1], m[2]],
                }])
            }
        })
        .collect();
    let mut kt = KernelTable::new();
    let mut entries = Vec::new();
    for ((b1, b2, mu, g), c) in acc {
        if c.norm() < 1e-14 {
            continue;
        }
        let k = kt.id(KernelKey {
            mu: [mu[0], mu[1], mu[2], mu[3]],
            gamma: [g[0], g[1], g[2]],
            extra: None,
        });
        entries.push(Entry {
            out: idx(&b1),
            inp: idx(&b2),
            kernel: k,
            coef: c,
            conj: false,
        });
    }
    MixSpec {
        pre: fields.clone(),
        post: fields,
        kernels: kt.keys,
        entries,
        label: format!("weyl(n={n})"),
    }
}

/// Tucker form from `l` spatial basis arrays `b_l(r)` (caller supplies the complex arrays) and per-sample coefficient sets `c_l(j)`:
/// `u_j(r) ~ sum_l c_l(j) b_l(r)`  =>  `N = sum_{l,l'} conj(b_l) . T[conj(c_l) c_l' wt] . b_l'`, with Hermitian kernel sharing (l' < l uses conj of K_{l' l}).
/// Kernel `k(l,l')` for `l <= l'` carries `extra = Some(l * L + l')`.
pub fn tucker(l: usize) -> MixSpec {
    let mut kt = KernelTable::new();
    let mut entries = Vec::new();
    for o in 0..l {
        for i in 0..l {
            let (a, b, conj) = if o <= i { (o, i, false) } else { (i, o, true) };
            let k = kt.id(KernelKey {
                mu: [0; NQ],
                gamma: [0; 3],
                extra: Some(a * l + b),
            });
            entries.push(Entry {
                out: o,
                inp: i,
                kernel: k,
                coef: Complex64::new(1.0, 0.0),
                conj,
            });
        }
    }
    MixSpec {
        pre: (0..l).map(FieldDiag::Array).collect(),
        post: (0..l).map(FieldDiag::Array).collect(),
        kernels: kt.keys,
        entries,
        label: format!("tucker(L={l})"),
    }
}

/// Rank-one-atom (CP) form: `u_j u_j^H ~ sum_l g_l(j) d_l d_l^H` => `N = sum_l conj(d_l) . T[wt g_l] . d_l`.
pub fn cp(l: usize) -> MixSpec {
    let mut kt = KernelTable::new();
    let mut entries = Vec::new();
    for o in 0..l {
        let k = kt.id(KernelKey {
            mu: [0; NQ],
            gamma: [0; 3],
            extra: Some(o),
        });
        entries.push(Entry {
            out: o,
            inp: o,
            kernel: k,
            coef: Complex64::new(1.0, 0.0),
            conj: false,
        });
    }
    MixSpec {
        pre: (0..l).map(FieldDiag::Array).collect(),
        post: (0..l).map(FieldDiag::Array).collect(),
        kernels: kt.keys,
        entries,
        label: format!("cp(L={l})"),
    }
}
