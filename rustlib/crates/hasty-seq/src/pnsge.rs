//! GE's PNS model as implemented in TOPPE `pns.m` (https://github.com/toppeMRI/toppe, +toppe/pns.m) and used by the
//! pge2 checks (normal mode ≤ 80 %, first controlled mode ≤ 100 %; https://github.com/HarmonizedMRI/pge2), after
//! DIN EN 60601-2-33: per physical axis
//!   P_a(l) = 100 · Δt · c / SR_min · Σ_{k ≤ l} SR_a[k] / (c + (l − k + ½) Δt)²,   SR_min = rheobase / α,
//! SR[k] = (G[k] − G[k−1]) / Δt (signed; no absolute value before the sum), combined P = sqrt(Σ_a P_a²).
//! Coil parameters from the same file (from the scanner's `Scandbdt.cfg` / `GRSubsystemHWO.xml`); the SIGNA Premier
//! coil is listed as `hrmw` [CHECK against the target scanner's own config file].

use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::wave::Grad;

#[derive(Clone, Debug)]
pub struct GeCoil {
    pub name: &'static str,
    /// s
    pub chronaxie: f64,
    /// T/s
    pub rheobase: f64,
    /// effective dB/dt length, m
    pub alpha: f64,
}

impl GeCoil {
    pub fn hrmw() -> Self {
        Self { name: "hrmw", chronaxie: 642.4e-6, rheobase: 17.9, alpha: 0.310 }
    }
    pub fn xrm() -> Self {
        Self { name: "xrm", chronaxie: 334e-6, rheobase: 23.4, alpha: 0.333 }
    }
    pub fn hrmbuhp() -> Self {
        Self { name: "hrmbuhp", chronaxie: 359e-6, rheobase: 26.5, alpha: 0.370 }
    }
    pub fn by_name(n: &str) -> Option<Self> {
        match n {
            "hrmw" | "premier" => Some(Self::hrmw()),
            "xrm" => Some(Self::xrm()),
            "hrmb" | "hrmbuhp" | "uhp" => Some(Self::hrmbuhp()),
            _ => None,
        }
    }
    pub fn sr_min(&self) -> f64 {
        self.rheobase / self.alpha
    }
}

/// GE PNS of a physical-axis waveform (corner samples), % per raster point: per axis and combined; `pad` zero samples
/// are appended so the response after the waveform is included
pub fn pns_ge(g: &Grad, coil: &GeCoil, pad: usize) -> (Vec<[f64; 3]>, Vec<f64>, f64) {
    let dt = g.dt;
    let n = g.len() + pad;
    let c = coil.chronaxie;
    let pre = 100.0 * dt * c / coil.sr_min();
    let nfft = (2 * n).next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(nfft);
    let inv = planner.plan_fft_inverse(nfft);
    let mut ker: Vec<Complex64> = (0..nfft).map(|j| if j < n { Complex64::new(1.0 / (c + (j as f64 + 0.5) * dt).powi(2), 0.0) } else { Complex64::new(0.0, 0.0) }).collect();
    fwd.process(&mut ker);
    let mut axis = vec![[0.0; 3]; n];
    for a in 0..3 {
        let mut sr: Vec<Complex64> = (0..nfft)
            .map(|k| {
                if k == 0 || k >= n {
                    return Complex64::new(0.0, 0.0);
                }
                let gk = if k < g.len() { g.g[k][a] } else { 0.0 };
                let gp = if k - 1 < g.len() { g.g[k - 1][a] } else { 0.0 };
                Complex64::new((gk - gp) / dt, 0.0)
            })
            .collect();
        fwd.process(&mut sr);
        for (s, k) in sr.iter_mut().zip(&ker) {
            *s *= k;
        }
        inv.process(&mut sr);
        for l in 0..n {
            axis[l][a] = pre * sr[l].re / nfft as f64;
        }
    }
    let tot: Vec<f64> = axis.iter().map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).collect();
    let mx = tot.iter().cloned().fold(0.0, f64::max);
    (axis, tot, mx)
}

/// steady state of a periodic block: maximum over the last of `repeats` copies
pub fn pns_ge_periodic(tr: &Grad, coil: &GeCoil, repeats: usize) -> f64 {
    let mut g = tr.clone();
    for _ in 1..repeats {
        g.g.extend_from_slice(&tr.g[1..]);
    }
    let (_, tot, _) = pns_ge(&g, coil, 0);
    let start = (repeats - 1) * (tr.len() - 1);
    tot[start..].iter().cloned().fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_slew_approaches_rheobase_scaling_and_matches_direct_sum() {
        // a long constant slew S: P → 100 S / SR_min · (1 − c / (c + T))
        let dt = 4e-6;
        let s = 20.0;
        let n = 20000; // 80 ms
        let ramp: Vec<f64> = (0..=n).map(|i| s * i as f64 * dt).collect();
        let w = Grad::from_axis(0, &ramp, dt, 0.0);
        let coil = GeCoil::hrmw();
        let (ax, _, _) = pns_ge(&w, &coil, 0);
        let t = n as f64 * dt;
        let expect = 100.0 * s / coil.sr_min() * (1.0 - coil.chronaxie / (coil.chronaxie + t));
        let got = ax[n][0];
        assert!((got - expect).abs() / expect < 2e-3, "{got} vs {expect}");
        // FFT convolution = TOPPE's direct sum on a short triangle
        let tri: Vec<f64> = (0..=100).map(|i| 0.02 * (1.0 - ((i as f64 - 50.0) / 50.0).abs())).collect();
        let w2 = Grad::from_axis(1, &tri, dt, 0.0);
        let (a2, _, _) = pns_ge(&w2, &coil, 50);
        let c = coil.chronaxie;
        let sr: Vec<f64> = (0..151).map(|k| if k == 0 { 0.0 } else { (tri.get(k).copied().unwrap_or(0.0) - tri.get(k - 1).copied().unwrap_or(0.0)) / dt }).collect();
        for l in [10usize, 50, 75, 120] {
            let d: f64 = 100.0 * dt * c / coil.sr_min() * (0..=l).map(|k| sr[k] / (c + ((l - k) as f64 + 0.5) * dt).powi(2)).sum::<f64>();
            assert!((a2[l][1] - d).abs() < 1e-9 * d.abs().max(1.0), "l {l}: {} vs {d}", a2[l][1]);
        }
    }
}
