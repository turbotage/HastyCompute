//! Velocity from multi-point phase-contrast encodes (any count / layout: referenced 4-point, Hadamard, 7-point
//! dual-VENC, ...). Convention of hasty-mrsim: x_e ∝ m e^{iθ} exp(−iγ M1_e·v).
//!
//! Phase differences to encode 0, φ_e = −arg(x_e x̄_0) ≡ k_e·v (mod 2π) with k_e = γ (M1_e − M1_0), are solved by
//! hierarchical unwrapping: the least-sensitive encodes that span 3D give a first estimate (dynamic range of the
//! highest VENC), each more sensitive group is unwrapped against the current estimate and joins an unweighted least
//! squares solve (equal phase noise per encode → the normal equations weight each encode by |k_e|², as ML does).

use hasty_mrsim::{Mat3, Vec3, GAMMA};
use num_complex::Complex64;

use crate::data::inv3;

/// largest number of velocity-encoded images (beyond the reference) [`FlowDecoder::velocity`] handles
pub const MAX_ENC: usize = 32;

pub struct FlowDecoder {
    m1: Vec<Vec3>,
    /// k_e for e ≥ 1 (rad per m/s)
    k: Vec<Vec3>,
    /// unwrapping stages: (encodes added at this stage (indices into k), inverse normal matrix of all encodes so far)
    stages: Vec<(Vec<usize>, Mat3)>,
    /// optional search over the first stage's aliases ([`FlowDecoder::with_alias_search`])
    alias: Option<Alias>,
}

/// Search beyond the first stage's range, for designs whose combined unaliased range is larger than that of their
/// least sensitive encodes (coprime or dual-VENC layouts whose low-sensitivity group alone does not span it).
///
/// The first stage's three encodes K₁ (rows k_e) leave v ambiguous exactly on the lattice v₀ + 2π K₁⁻¹ n, n ∈ ℤ³.
/// Every lattice point inside the box |v_a| ≤ `vext` is refined through all stages (each encode unwrapped against it,
/// least squares) and scored by the phase cost Σ_e (1 − cos(φ_e − k_e·v)), the negative log-likelihood of equal,
/// small phase noise per encode. An aliased solution (n ≠ 0) wins only if its cost is lower than the in-range one by
/// `margin`: a prior for the first stage's range that keeps noise from selecting a distant alias at low SNR
/// (margin 0 = maximum likelihood over the box).
#[derive(Clone, Copy, Debug)]
pub struct AliasSearch {
    /// half-width of the velocity box per axis (m/s), e.g. the design's unaliased range
    pub vext: f64,
    /// cost advantage an aliased solution needs over the in-range one (dimensionless, Σ (1 − cos))
    pub margin: f64,
}

#[derive(Clone, Debug)]
struct Alias {
    opts: AliasSearch,
    /// 2π K₁⁻¹: columns are the alias lattice generators
    gen: Mat3,
    /// lattice offsets n to try (n = 0 excluded)
    offsets: Vec<[i32; 3]>,
}

fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

impl FlowDecoder {
    /// `m1`: logical first moment of each encode (T s / m · s), e.g. `FlowSpiralParams::encode_m1()`
    pub fn new(m1: &[Vec3]) -> Self {
        assert!(m1.len() >= 4, "need ≥ 4 encodes for 3 velocity components");
        let k: Vec<Vec3> = m1[1..].iter().map(|m| [0, 1, 2].map(|a| GAMMA * (m[a] - m1[0][a]))).collect();
        let mut order: Vec<usize> = (0..k.len()).collect();
        let norm = |e: usize| dot(&k[e], &k[e]).sqrt();
        order.sort_by(|&a, &b| norm(a).partial_cmp(&norm(b)).unwrap());
        // groups of equal sensitivity
        let mut groups: Vec<Vec<usize>> = vec![];
        for &e in &order {
            match groups.last_mut() {
                Some(g) if (norm(e) - norm(g[0])).abs() <= 1e-6 * norm(e) => g.push(e),
                _ => groups.push(vec![e]),
            }
        }
        let mut stages = vec![];
        let (mut n, mut pending) = ([[0.0; 3]; 3], vec![]);
        let kmax = order.last().map(|&e| norm(e)).unwrap_or(1.0);
        for g in groups {
            for &e in &g {
                for i in 0..3 {
                    for j in 0..3 {
                        n[i][j] += k[e][i] * k[e][j];
                    }
                }
            }
            pending.extend(g);
            if det3(&n).abs() > 1e-9 * kmax.powi(6) {
                stages.push((std::mem::take(&mut pending), inv3(&n)));
            }
        }
        assert!(pending.is_empty() && !stages.is_empty(), "encodes do not span 3D velocity space");
        Self { m1: m1.to_vec(), k, stages, alias: None }
    }

    /// enable the alias search ([`AliasSearch`]). A no-op (returns `self` unchanged) when the first stage does not
    /// consist of exactly three encodes or when no alias lies inside the box.
    pub fn with_alias_search(mut self, opts: AliasSearch) -> Self {
        let first = &self.stages[0].0;
        if first.len() != 3 {
            return self;
        }
        let k1: Mat3 = [self.k[first[0]], self.k[first[1]], self.k[first[2]]];
        if det3(&k1).abs() < 1e-12 {
            return self;
        }
        let inv = inv3(&k1);
        let tau = 2.0 * std::f64::consts::PI;
        let gen: Mat3 = [0, 1, 2].map(|i| [0, 1, 2].map(|j| tau * inv[i][j]));
        // |n_i| ≤ (K₁ (v − v₀))_i / 2π with |v_a| ≤ vext and |v₀_a| ≤ the first stage's range
        let r = self.range();
        let m: [i32; 3] = [0, 1, 2].map(|i| ((0..3).map(|a| k1[i][a].abs() * (opts.vext + r[a])).sum::<f64>() / tau).ceil() as i32);
        let mut offsets = vec![];
        for n0 in -m[0]..=m[0] {
            for n1 in -m[1]..=m[1] {
                for n2 in -m[2]..=m[2] {
                    if (n0, n1, n2) == (0, 0, 0) {
                        continue;
                    }
                    let d = [0, 1, 2].map(|a| gen[a][0] * n0 as f64 + gen[a][1] * n1 as f64 + gen[a][2] * n2 as f64);
                    // an alias of an in-range v₀ can only land in the box if its shift is at most vext + range per axis
                    // (strictly: a shift of exactly vext + range only reaches the box corner from the range's edge)
                    if (0..3).all(|a| d[a].abs() < opts.vext + r[a] - 1e-6) {
                        offsets.push([n0, n1, n2]);
                    }
                }
            }
        }
        if !offsets.is_empty() {
            self.alias = Some(Alias { opts, gen, offsets });
        }
        self
    }

    /// the alias search in use, if any
    pub fn alias_search(&self) -> Option<AliasSearch> {
        self.alias.as_ref().map(|a| a.opts)
    }

    pub fn n_enc(&self) -> usize {
        self.m1.len()
    }

    /// largest |v| along each axis that the first stage resolves without aliasing (m/s), i.e. the effective VENC
    pub fn range(&self) -> Vec3 {
        let first = &self.stages[0].0;
        [0, 1, 2].map(|a| std::f64::consts::PI / first.iter().map(|&e| self.k[e][a].abs()).fold(0.0, f64::max))
    }

    /// sensitivity |k_e| = γ |M1_e − M1_0| per encode (π / VENC_e; 0 for the reference)
    pub fn kv(&self) -> Vec<f64> {
        std::iter::once(0.0).chain(self.k.iter().map(|k| dot(k, k).sqrt())).collect()
    }

    /// k_e = γ (M1_e − M1_0) (rad per m/s), e ≥ 1
    pub fn k(&self) -> &[Vec3] {
        &self.k
    }

    /// encode phase −γ M1_e·v of a velocity (for synthesising model-consistent data)
    pub fn phase(&self, e: usize, v: &Vec3) -> f64 {
        -GAMMA * dot(&self.m1[e], v)
    }

    /// logical velocity (m/s) from the encodes of one voxel, z[e] for e in 0..n_enc
    pub fn velocity(&self, z: &[Complex64]) -> Vec3 {
        // fixed-size scratch (no allocation per voxel; MAX_ENC encodes)
        let ne = z.len() - 1;
        assert!(ne <= MAX_ENC, "at most {MAX_ENC} encodes");
        let mut phi = [0f64; MAX_ENC];
        for (p, ze) in phi.iter_mut().zip(&z[1..]) {
            *p = -(ze * z[0].conj()).arg();
        }
        let v0 = self.refine(&phi[..ne], [0.0; 3]);
        let Some(al) = &self.alias else { return v0 };
        let cost = |v: &Vec3| (0..ne).map(|e| 1.0 - (phi[e] - dot(&self.k[e], v)).cos()).sum::<f64>();
        let c0 = cost(&v0);
        // the first stage's own solution (before the later stages): the lattice is anchored there
        let first = self.first_stage(&phi[..ne]);
        let (mut best, mut cbest) = (v0, c0 - al.opts.margin);
        for n in &al.offsets {
            let start = [0, 1, 2].map(|a| first[a] + al.gen[a][0] * n[0] as f64 + al.gen[a][1] * n[1] as f64 + al.gen[a][2] * n[2] as f64);
            if start.iter().any(|x| x.abs() > al.opts.vext + 1e-9) {
                continue;
            }
            let v = self.refine(&phi[..ne], start);
            if v.iter().any(|x| x.abs() > al.opts.vext + 1e-6) {
                continue;
            }
            let c = cost(&v);
            if c < cbest {
                best = v;
                cbest = c;
            }
        }
        best
    }

    /// least-squares solution of the first stage alone (principal phases)
    fn first_stage(&self, phi: &[f64]) -> Vec3 {
        let (add, ninv) = &self.stages[0];
        let mut b = [0.0; 3];
        for &e in add {
            for a in 0..3 {
                b[a] += self.k[e][a] * phi[e];
            }
        }
        [0, 1, 2].map(|i| dot(&ninv[i], &b))
    }

    /// hierarchical unwrapping from a start velocity: each stage's encodes unwrapped against the current estimate, then
    /// least squares over all encodes so far (start 0 = the plain hierarchical decoder)
    fn refine(&self, phi: &[f64], start: Vec3) -> Vec3 {
        let tau = 2.0 * std::f64::consts::PI;
        let mut used = [(0usize, 0f64); MAX_ENC];
        let mut nu = 0;
        let mut v = start;
        for (add, ninv) in &self.stages {
            for &e in add {
                let pred = dot(&self.k[e], &v);
                used[nu] = (e, phi[e] + tau * ((pred - phi[e]) / tau).round());
                nu += 1;
            }
            let mut b = [0.0; 3];
            for &(e, p) in &used[..nu] {
                for a in 0..3 {
                    b[a] += self.k[e][a] * p;
                }
            }
            v = [0, 1, 2].map(|i| dot(&ninv[i], &b));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hasty_mrsim::sequence::{Encoding, FlowSpiralParams};

    fn synth(d: &FlowDecoder, v: &Vec3, m: f64, theta: f64) -> Vec<Complex64> {
        (0..d.n_enc()).map(|e| Complex64::from_polar(m, theta + d.phase(e, v))).collect()
    }

    fn check(enc: Encoding, vmax: f64) {
        let mut p = FlowSpiralParams::aorta_3t_default();
        p.venc = 1.5;
        p.encoding = enc;
        let d = FlowDecoder::new(&p.encode_m1());
        let mut s = 0x9e37u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut worst = 0.0f64;
        for _ in 0..2000 {
            let v = [0, 1, 2].map(|_| (2.0 * u() - 1.0) * vmax);
            let got = d.velocity(&synth(&d, &v, 0.3 + u(), 6.0 * u()));
            worst = worst.max((0..3).map(|a| (got[a] - v[a]).abs()).fold(0.0, f64::max));
        }
        assert!(worst < 1e-9, "{enc:?}: max error {worst:e} m/s");
    }

    #[test]
    fn referenced() {
        check(Encoding::Referenced, 0.999 * 1.5);
    }

    #[test]
    fn hadamard() {
        // Hadamard differences have k components ±γΔM1 per axis, |k·v| < π needs |v|_1 < VENC
        check(Encoding::Hadamard, 0.33 * 1.5);
    }

    #[test]
    fn dual_venc_unwraps_beyond_the_low_venc() {
        let p = {
            let mut p = FlowSpiralParams::aorta_3t_default();
            p.venc = 1.5;
            p.encoding = Encoding::DualVenc { venc_hi: 4.5 };
            p
        };
        let d = FlowDecoder::new(&p.encode_m1());
        assert!((d.range()[0] - 4.5).abs() < 1e-9, "range {:?}", d.range());
        // velocities up to 2.5 VENC_lo: the low-VENC phases wrap and must be unwrapped by the high-VENC estimate
        check(Encoding::DualVenc { venc_hi: 4.5 }, 0.99 * 1.5 * 2.5);
    }

    /// coprime dual-VENC layout ("coprime7", p = 2): per axis one encode at VENC 1.0 and one at 1.5 m/s against a common
    /// reference; combined unaliased range ±3 m/s, first (least sensitive) stage alone ±1.5 m/s
    fn coprime7() -> Vec<Vec3> {
        let mut m1 = vec![[0.0; 3]];
        for a in 0..3 {
            for venc in [1.0f64, 1.5] {
                let mut m = [0.0; 3];
                m[a] = std::f64::consts::PI / venc / GAMMA;
                m1.push(m);
            }
        }
        m1
    }

    #[test]
    fn coprime_alias_search_covers_the_combined_range() {
        let plain = FlowDecoder::new(&coprime7());
        assert!((plain.range()[0] - 1.5).abs() < 1e-9, "first-stage range {:?}", plain.range());
        let d = FlowDecoder::new(&coprime7()).with_alias_search(AliasSearch { vext: 3.0, margin: 0.0 });
        assert!(d.alias_search().is_some());
        let mut s = 0x77u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut worst, mut plain_bad) = (0.0f64, 0usize);
        for _ in 0..4000 {
            let v = [0, 1, 2].map(|_| (2.0 * u() - 1.0) * 2.99);
            let z = synth(&d, &v, 0.3 + u(), 6.0 * u());
            let got = d.velocity(&z);
            worst = worst.max((0..3).map(|a| (got[a] - v[a]).abs()).fold(0.0, f64::max));
            let gp = plain.velocity(&z);
            plain_bad += ((0..3).map(|a| (gp[a] - v[a]).abs()).fold(0.0, f64::max) > 0.1) as usize;
        }
        assert!(worst < 1e-9, "alias search: max error {worst:e} m/s");
        // the hierarchical decoder alone aliases whenever a component exceeds 1.5 m/s (most of the ±2.99 box)
        assert!(plain_bad > 2000, "hierarchical decoder failures {plain_bad} / 4000");
    }

    #[test]
    fn alias_search_keeps_in_range_answers_and_reference_layout() {
        // the referenced layout's first stage already spans its range: no alias lies in a box of the same size
        let mut p = FlowSpiralParams::aorta_3t_default();
        p.venc = 1.5;
        let d = FlowDecoder::new(&p.encode_m1()).with_alias_search(AliasSearch { vext: 1.5, margin: 0.0 });
        assert!(d.alias_search().is_none(), "no alias inside the first stage's own range");
        // coprime, velocities inside ±1.5: identical to the hierarchical decoder (noise-free)
        let plain = FlowDecoder::new(&coprime7());
        let d = FlowDecoder::new(&coprime7()).with_alias_search(AliasSearch { vext: 3.0, margin: 0.5 });
        for v in [[0.3, -1.2, 1.45], [0.0, 0.0, 0.0], [-1.49, 1.0, -0.2]] {
            let z = synth(&d, &v, 1.0, 0.4);
            let (a, b) = (d.velocity(&z), plain.velocity(&z));
            assert!((0..3).all(|i| (a[i] - b[i]).abs() < 1e-12 && (a[i] - v[i]).abs() < 1e-9), "{a:?} {b:?} {v:?}");
        }
    }

    #[test]
    fn dual_venc_noise_is_set_by_the_low_venc() {
        // phase noise σ_φ on every encode: the solve's velocity noise should be close to σ_φ VENC_lo / π
        let mut p = FlowSpiralParams::aorta_3t_default();
        p.venc = 1.0;
        p.encoding = Encoding::DualVenc { venc_hi: 3.0 };
        let d = FlowDecoder::new(&p.encode_m1());
        let mut s = 0x1234u64;
        let mut g = || {
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
        };
        let (sig, n) = (0.02, 20000);
        let mut e2 = 0.0;
        for _ in 0..n {
            let v = [0.7, -1.9, 0.2];
            let z: Vec<Complex64> = (0..7).map(|e| Complex64::from_polar(1.0, d.phase(e, &v) + if e > 0 { sig * g() } else { 0.0 })).collect();
            let got = d.velocity(&z);
            e2 += (got[0] - v[0]).powi(2);
        }
        let rms = (e2 / n as f64).sqrt();
        // LS over lo + hi: 1 / sqrt(1 + 1/9) of the lo-only noise
        let expect = sig / std::f64::consts::PI / (1.0f64 + 1.0 / 9.0).sqrt();
        assert!((rms / expect - 1.0).abs() < 0.05, "rms {rms:.4e}, expected {expect:.4e}");
    }
}
