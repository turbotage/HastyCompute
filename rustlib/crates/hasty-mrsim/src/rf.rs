//! Slab-selective RF excitation: a windowed-sinc pulse under a slice-select gradient, its rephasing lobe, and the
//! excitation profile from a Bloch simulation along the slab normal (rotations per time step, relaxation neglected
//! during the pulse).
//!
//! The profile gives the local flip angle α(z) and the phase of the excited magnetization. A steady-state spoiled GRE
//! then has M_xy = M0 sin α(z) (1 − E1) / (1 − cos α(z) E1). A spin entering the slab with M_z = s M0 gives s M0 sin α(z)
//! on its first excitation. Off-resonant spins (fat at −3.4 ppm) see the profile displaced by Δz = −Δf / (γ̄ G_ss).

use crate::GAMMA_BAR;

/// a slab-selective excitation
#[derive(Clone, Debug)]
pub struct SlabPulse {
    /// pulse duration, s
    pub duration: f64,
    /// time-bandwidth product (zero crossings of the sinc: TBW / 2 on each side)
    pub tbw: f64,
    /// nominal flip angle at the slab centre, rad
    pub flip: f64,
    /// slab thickness (FWHM of the profile), m
    pub thickness: f64,
    /// Hamming apodisation
    pub hamming: bool,
    /// time steps of the waveform
    pub n: usize,
}

/// the excitation profile on a z grid (relative to the slab centre)
#[derive(Clone, Debug)]
pub struct SlabProfile {
    pub z0: f64,
    pub dz: f64,
    /// local flip angle, rad
    pub alpha: Vec<f32>,
    /// phase of the excited magnetization, rad (after the rephasing lobe)
    pub phase: Vec<f32>,
    /// slice-select gradient, T/m
    pub g_ss: f64,
    /// RF bandwidth, Hz
    pub bandwidth: f64,
}

impl SlabPulse {
    pub fn new(duration: f64, tbw: f64, flip: f64, thickness: f64) -> Self {
        Self { duration, tbw, flip, thickness, hamming: true, n: 400 }
    }

    /// RF bandwidth (Hz) and the slice-select gradient (T/m) for the thickness
    pub fn bandwidth(&self) -> f64 {
        self.tbw / self.duration
    }

    pub fn g_ss(&self) -> f64 {
        self.bandwidth() / (GAMMA_BAR * self.thickness)
    }

    /// B1 envelope samples (relative, real), scaled so the small-tip flip at resonance is `flip`
    pub fn envelope(&self) -> Vec<f64> {
        let n = self.n;
        let h = self.tbw / 2.0;
        let mut b: Vec<f64> = (0..n)
            .map(|k| {
                let x = (k as f64 + 0.5) / n as f64 * 2.0 - 1.0; // −1 … 1
                let t = x * h;
                let s = if t.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
                let w = if self.hamming { 0.54 + 0.46 * (std::f64::consts::PI * x).cos() } else { 1.0 };
                s * w
            })
            .collect();
        let dt = self.duration / n as f64;
        let area: f64 = b.iter().sum::<f64>() * dt;
        // γ ∫ B1 dt = flip (small-tip)
        let scale = self.flip / (2.0 * std::f64::consts::PI * GAMMA_BAR * area);
        for v in &mut b {
            *v *= scale;
        }
        b
    }

    /// Bloch simulation over z ∈ [−extent, extent] (m from the slab centre) at `nz` points: the rotation per step about
    /// (γ B1, 0, γ G z + Δω), then the rephasing lobe (half the slice-select area); `df` an off-resonance (Hz)
    pub fn profile(&self, extent: f64, nz: usize, df: f64) -> SlabProfile {
        let b1 = self.envelope();
        let dt = self.duration / self.n as f64;
        let g = self.g_ss();
        let gam = 2.0 * std::f64::consts::PI * GAMMA_BAR;
        let dz = 2.0 * extent / (nz - 1) as f64;
        let mut alpha = Vec::with_capacity(nz);
        let mut phase = Vec::with_capacity(nz);
        for iz in 0..nz {
            let z = -extent + iz as f64 * dz;
            let mut m = [0.0f64, 0.0, 1.0];
            for &b in &b1 {
                let w = [gam * b, 0.0, gam * g * z + 2.0 * std::f64::consts::PI * df];
                let wn = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
                let th = wn * dt;
                if th < 1e-15 {
                    continue;
                }
                let k = [w[0] / wn, w[1] / wn, w[2] / wn];
                // Rodrigues: m' = m cos θ + (k × m) sin θ + k (k·m)(1 − cos θ); precession is left-handed (−θ)
                let (c, s) = (th.cos(), -th.sin());
                let kxm = [k[1] * m[2] - k[2] * m[1], k[2] * m[0] - k[0] * m[2], k[0] * m[1] - k[1] * m[0]];
                let kd = k[0] * m[0] + k[1] * m[1] + k[2] * m[2];
                m = [0, 1, 2].map(|a| m[a] * c + kxm[a] * s + k[a] * kd * (1.0 - c));
            }
            // rephasing lobe: −½ the slice-select area (a free precession about z)
            let ph_re = gam * g * z * 0.5 * self.duration;
            let mxy = (m[0] * m[0] + m[1] * m[1]).sqrt();
            let ph = m[1].atan2(m[0]) + ph_re;
            alpha.push(mxy.atan2(m[2]) as f32);
            phase.push(((ph + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI) as f32);
        }
        SlabProfile { z0: -extent, dz, alpha, phase, g_ss: g, bandwidth: self.bandwidth() }
    }
}

impl SlabProfile {
    /// local flip angle at z (m from the slab centre), linear in the table, 0 outside
    pub fn alpha_at(&self, z: f64) -> f64 {
        let f = (z - self.z0) / self.dz;
        if f < 0.0 || f > (self.alpha.len() - 1) as f64 {
            return 0.0;
        }
        let i = (f as usize).min(self.alpha.len() - 2);
        let a = f - i as f64;
        self.alpha[i] as f64 * (1.0 - a) + self.alpha[i + 1] as f64 * a
    }

    /// full width at half the centre's sin α
    pub fn fwhm(&self) -> f64 {
        let s: Vec<f64> = self.alpha.iter().map(|&a| (a as f64).sin()).collect();
        let c = s[s.len() / 2];
        let above: Vec<usize> = (0..s.len()).filter(|&i| s[i] >= 0.5 * c).collect();
        (above.last().unwrap() - above.first().unwrap()) as f64 * self.dz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_slab_profile_has_its_thickness_and_flip() {
        let p = SlabPulse::new(0.8e-3, 6.0, 12f64.to_radians(), 0.16);
        let pr = p.profile(0.16, 801, 0.0);
        let c = pr.alpha_at(0.0).to_degrees();
        let fw = pr.fwhm();
        let edge = pr.alpha_at(0.12).to_degrees();
        // phase flat across the slab after the rephasing lobe
        let ph: Vec<f32> = (0..801).filter(|&i| (pr.z0 + i as f64 * pr.dz).abs() < 0.05).map(|i| pr.phase[i]).collect();
        let pspread = ph.iter().cloned().fold(f32::MIN, f32::max) - ph.iter().cloned().fold(f32::MAX, f32::min);
        eprintln!("centre α {c:.2}°, FWHM {:.1} mm, α at 120 mm {edge:.3}°, G_ss {:.3} mT/m, BW {:.0} Hz, phase spread {pspread:.3} rad", fw * 1e3, pr.g_ss * 1e3, pr.bandwidth);
        assert!((c - 12.0).abs() < 0.3, "centre flip {c}");
        assert!((fw / 0.16 - 1.0).abs() < 0.06, "FWHM {fw}");
        assert!(edge < 0.5, "outside the slab {edge}");
        assert!(pspread < 0.15, "phase spread {pspread}");
    }

    #[test]
    fn fat_profile_is_displaced() {
        let p = SlabPulse::new(0.8e-3, 6.0, 12f64.to_radians(), 0.16);
        let df = -3.4e-6 * GAMMA_BAR * 3.0; // fat at 3 T
        let w = p.profile(0.16, 1601, 0.0);
        let f = p.profile(0.16, 1601, df);
        // the half-maximum edges move by Δz = Δf / (γ̄ G)
        let edge = |pr: &SlabProfile| -> f64 {
            let c = pr.alpha_at(0.0).sin();
            (0..pr.alpha.len()).map(|i| pr.z0 + i as f64 * pr.dz).filter(|&z| z > 0.0 && pr.alpha_at(z).sin() >= 0.5 * c).fold(0.0, f64::max)
        };
        let want = -df / (GAMMA_BAR * p.g_ss()); // resonance γ G z + Δω = 0
        eprintln!("upper edge water {:.1} mm, fat {:.1} mm, Δz {:.1} mm (expected {:.1})", edge(&w) * 1e3, edge(&f) * 1e3, (edge(&f) - edge(&w)) * 1e3, want * 1e3);
        assert!(((edge(&f) - edge(&w)) - want).abs() < 1.5e-3);
    }
}
