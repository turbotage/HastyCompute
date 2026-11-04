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

pub struct FlowDecoder {
    m1: Vec<Vec3>,
    /// k_e for e ≥ 1 (rad per m/s)
    k: Vec<Vec3>,
    /// unwrapping stages: (encodes added at this stage (indices into k), inverse normal matrix of all encodes so far)
    stages: Vec<(Vec<usize>, Mat3)>,
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
        Self { m1: m1.to_vec(), k, stages }
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
        let tau = 2.0 * std::f64::consts::PI;
        let phi: Vec<f64> = z[1..].iter().map(|ze| -(ze * z[0].conj()).arg()).collect();
        let mut used: Vec<(usize, f64)> = Vec::with_capacity(phi.len());
        let mut v = [0.0; 3];
        for (add, ninv) in &self.stages {
            for &e in add {
                let pred = dot(&self.k[e], &v);
                used.push((e, phi[e] + tau * ((pred - phi[e]) / tau).round()));
            }
            let mut b = [0.0; 3];
            for &(e, p) in &used {
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
