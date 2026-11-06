//! Linear time-invariant model of the gradient chain (GIRF): what the scanner actually plays for a nominal waveform.
//!
//! Per axis a frequency response H(f) (self term), optionally cross terms H_ab(f) (axis b's input appearing on axis a)
//! and a B0 term H0_a(f) (T per T/m: the spatially constant field caused by axis a, i.e. the k0 / eddy-B0 phase).
//! Parametric self term:
//!   H(f) = e^{−i2πf·delay} · 1 / (1 + i2πf τ_lp) · [1 − Σ_k a_k (i2πf τ_k) / (1 + i2πf τ_k)],
//! i.e. a pure delay, the amplifier / coil low-pass, and multi-exponential eddy currents (each driven by dG/dt with
//! amplitude a_k and time constant τ_k), or a measured table (linear interpolation of re / im on |f|, Hermitian).
//! Applied by FFT convolution on the waveform raster with zero padding (no wrap-around), so the output keeps the
//! raster and is extended by the response tail.

use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::wave::Grad;
use crate::{Vec3, GAMMA};

#[derive(Clone, Debug)]
pub enum Response {
    Ideal,
    Parametric { delay: f64, tau_lp: f64, eddy: Vec<(f64, f64)> },
    /// measured: frequencies (Hz, ascending, ≥ 0) and complex response
    Table { f: Vec<f64>, h: Vec<Complex64> },
}

impl Response {
    pub fn at(&self, f: f64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        match self {
            Response::Ideal => Complex64::new(1.0, 0.0),
            Response::Parametric { delay, tau_lp, eddy } => {
                let w = 2.0 * std::f64::consts::PI * f;
                let mut h = (-i * w * *delay).exp() / (1.0 + i * w * *tau_lp);
                let mut e = Complex64::new(1.0, 0.0);
                for &(a, tau) in eddy {
                    e -= a * (i * w * tau) / (1.0 + i * w * tau);
                }
                h *= e;
                h
            }
            Response::Table { f: ff, h } => {
                let fa = f.abs();
                let j = match ff.binary_search_by(|x| x.partial_cmp(&fa).unwrap()) {
                    Ok(j) => return if f >= 0.0 { h[j] } else { h[j].conj() },
                    Err(j) => j,
                };
                let v = if j == 0 {
                    h[0]
                } else if j >= ff.len() {
                    h[ff.len() - 1]
                } else {
                    let a = (fa - ff[j - 1]) / (ff[j] - ff[j - 1]);
                    h[j - 1] * (1.0 - a) + h[j] * a
                };
                if f >= 0.0 {
                    v
                } else {
                    v.conj()
                }
            }
        }
    }
}

/// gradient chain of the three physical axes
#[derive(Clone, Debug)]
pub struct Girf {
    pub axis: [Response; 3],
    /// cross terms (output axis, input axis, response)
    pub cross: Vec<(usize, usize, Response)>,
    /// B0 terms per input axis (T per T/m)
    pub b0: [Option<Response>; 3],
    /// extra samples appended for the response tail
    pub tail: f64,
}

impl Girf {
    pub fn ideal() -> Self {
        Self { axis: [Response::Ideal, Response::Ideal, Response::Ideal], cross: vec![], b0: [None, None, None], tail: 0.0 }
    }
    /// a plausible clinical chain for studies until a measured GIRF exists: ~2–3 µs delays, 8 µs low-pass, 0.2 % / 3 ms
    /// and 0.1 % / 50 µs eddy currents (illustrative numbers, not a scanner measurement)
    pub fn clinical_example() -> Self {
        let p = |d: f64| Response::Parametric { delay: d, tau_lp: 8e-6, eddy: vec![(0.002, 3e-3), (0.001, 50e-6)] };
        Self { axis: [p(2.0e-6), p(2.5e-6), p(3.0e-6)], cross: vec![], b0: [None, None, None], tail: 1e-3 }
    }

    /// samples kept before the input start (room for negative delays and the slight pre-ringing of a band-limited table)
    pub const PRE: usize = 32;

    /// actual gradients (physical axes) for the nominal `g`, starting `PRE` raster steps earlier; and the B0 field (T)
    /// from the B0 terms on the same raster. Parametric responses run as exact time-domain recursions on the
    /// piecewise-linear input (delay by linear interpolation, first-order sections in closed form); measured tables by
    /// zero-padded FFT convolution.
    pub fn apply(&self, g: &Grad) -> (Grad, Vec<f64>) {
        let dt = g.dt;
        let pre = Self::PRE;
        let ntail = (self.tail / dt).ceil() as usize;
        let n = pre + g.len() + ntail;
        let input: Vec<Vec<f64>> = (0..3).map(|a| (0..n).map(|i| if i >= pre && i - pre < g.len() { g.g[i - pre][a] } else { 0.0 }).collect()).collect();
        let mut fft: Option<(Vec<Vec<Complex64>>, std::sync::Arc<dyn rustfft::Fft<f64>>, usize)> = None;
        let mut filter = |resp: &Response, src: usize| -> Vec<f64> {
            match resp {
                Response::Ideal => input[src].clone(),
                Response::Parametric { delay, tau_lp, eddy } => time_domain(&input[src], dt, *delay, *tau_lp, eddy),
                Response::Table { .. } => {
                    let (spec, inv, nfft) = fft.get_or_insert_with(|| {
                        let nfft = (2 * n).next_power_of_two();
                        let mut planner = FftPlanner::<f64>::new();
                        let fwd = planner.plan_fft_forward(nfft);
                        let spec = (0..3)
                            .map(|a| {
                                let mut x: Vec<Complex64> = (0..nfft).map(|i| Complex64::new(if i < n { input[a][i] } else { 0.0 }, 0.0)).collect();
                                fwd.process(&mut x);
                                x
                            })
                            .collect();
                        (spec, planner.plan_fft_inverse(nfft), nfft)
                    });
                    let nf = *nfft;
                    let freq = |k: usize| if k <= nf / 2 { k as f64 / (nf as f64 * dt) } else { (k as f64 - nf as f64) / (nf as f64 * dt) };
                    let mut y: Vec<Complex64> = (0..nf).map(|k| spec[src][k] * resp.at(freq(k))).collect();
                    inv.process(&mut y);
                    y.iter().take(n).map(|c| c.re / nf as f64).collect()
                }
            }
        };
        let mut out = vec![[0.0; 3]; n];
        for a in 0..3 {
            let y = filter(&self.axis[a], a);
            for i in 0..n {
                out[i][a] += y[i];
            }
        }
        for (oa, ia, r) in &self.cross {
            let y = filter(r, *ia);
            for i in 0..n {
                out[i][*oa] += y[i];
            }
        }
        let mut b0 = vec![0.0; n];
        for a in 0..3 {
            if let Some(r) = &self.b0[a] {
                let y = filter(r, a);
                for i in 0..n {
                    b0[i] += y[i];
                }
            }
        }
        (Grad { dt, t0: g.t0 - pre as f64 * dt, g: out }, b0)
    }
}

/// first-order low-pass y' = (x − y)/τ, exact for x linear between the samples
fn lowpass_pl(x: &[f64], dt: f64, tau: f64) -> Vec<f64> {
    if tau <= 0.0 {
        return x.to_vec();
    }
    let e = (-dt / tau).exp();
    let mut y = vec![0.0; x.len()];
    y[0] = x[0];
    for i in 0..x.len() - 1 {
        let m = (x[i + 1] - x[i]) / dt;
        y[i + 1] = x[i + 1] - m * tau + (y[i] - x[i] + m * tau) * e;
    }
    y
}

/// parametric response in the time domain: delay ∘ low-pass ∘ (1 − Σ a_k high-pass_k)
fn time_domain(x: &[f64], dt: f64, delay: f64, tau_lp: f64, eddy: &[(f64, f64)]) -> Vec<f64> {
    let d = delay / dt;
    let n = x.len();
    let at = |f: f64| -> f64 {
        if f < 0.0 || f > (n - 1) as f64 {
            return 0.0;
        }
        let i = (f.floor() as usize).min(n - 2);
        let a = f - i as f64;
        x[i] * (1.0 - a) + x[i + 1] * a
    };
    let xd: Vec<f64> = (0..n).map(|i| at(i as f64 - d)).collect();
    let y = lowpass_pl(&xd, dt, tau_lp);
    let mut z = y.clone();
    for &(a, tau) in eddy {
        let l = lowpass_pl(&y, dt, tau);
        for i in 0..n {
            z[i] -= a * (y[i] - l[i]);
        }
    }
    z
}

/// k-space error between the actual and the nominal waveform (cycles/m) at the nominal raster points (the actual one
/// on the same raster, possibly starting earlier), and the k0 phase (rad) of the B0 term along the actual raster
pub fn trajectory_error(nominal: &Grad, actual: &Grad, b0: &[f64]) -> (Vec<Vec3>, Vec<f64>) {
    let off = ((nominal.t0 - actual.t0) / nominal.dt).round() as usize;
    let kn = nominal.moments(nominal.t0).k();
    let ka = actual.moments(actual.t0).k();
    let err = (0..kn.len()).map(|i| [0, 1, 2].map(|q| ka[i + off][q] - kn[i][q])).collect();
    let mut ph = vec![0.0; b0.len()];
    for i in 1..b0.len() {
        ph[i] = ph[i - 1] + GAMMA * 0.5 * (b0[i - 1] + b0[i]) * nominal.dt;
    }
    (err, ph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave::trap_shape;

    #[test]
    fn delay_shifts_and_lowpass_preserves_area() {
        let dt = 10e-6;
        let w = Grad::from_axis(0, &trap_shape(10, 50, 10, 0.02), dt, 0.0);
        // a delay of exactly 3 raster steps: the output is the shifted input
        let g = Girf { axis: [Response::Parametric { delay: 3.0 * dt, tau_lp: 0.0, eddy: vec![] }, Response::Ideal, Response::Ideal], cross: vec![], b0: [None, None, None], tail: 1e-3 };
        let (a, _) = g.apply(&w);
        for i in 0..w.len() {
            let j = i + 3 + Girf::PRE;
            assert!((a.g[j][0] - w.g[i][0]).abs() < 1e-12, "i {i}: {} vs {}", a.g[j][0], w.g[i][0]);
        }
        // a low-pass keeps the area (DC gain 1) and smooths the corners
        let g2 = Girf { axis: [Response::Parametric { delay: 0.0, tau_lp: 20e-6, eddy: vec![] }, Response::Ideal, Response::Ideal], cross: vec![], b0: [None, None, None], tail: 2e-3 };
        let (b, _) = g2.apply(&w);
        let area_in: f64 = w.moments(0.0).last().0[0];
        let area_out: f64 = b.moments(0.0).last().0[0];
        assert!((area_in - area_out).abs() / area_in < 1e-8, "{area_in} {area_out}");
        assert!(b.limits().s_axis < w.limits().s_axis);
    }

    #[test]
    fn eddy_current_reduces_a_long_plateau_by_its_amplitude() {
        // after a step, an eddy term a e^{−t/τ} pulls the gradient down by a, recovering with τ
        let dt = 10e-6;
        let w = Grad::from_axis(1, &trap_shape(5, 2000, 5, 0.01), dt, 0.0);
        let g = Girf { axis: [Response::Ideal, Response::Parametric { delay: 0.0, tau_lp: 0.0, eddy: vec![(0.01, 2e-3)] }, Response::Ideal], cross: vec![], b0: [None, None, None], tail: 0.0 };
        let (a, _) = g.apply(&w);
        let just_after = a.g[10 + Girf::PRE][1] / 0.01;
        assert!((just_after - 0.99).abs() < 2e-3, "{just_after}");
        let later = a.g[5 + 1000 + Girf::PRE][1] / 0.01; // 10 ms: recovered by e^{−5}
        assert!((later - (1.0 - 0.01 * (-(1000.0 * dt) / 2e-3f64).exp())).abs() < 1e-3, "{later}");
    }
}
