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
    /// mechanical resonance on top of a base response: H = H_base · (1 + g · 2iζω0ω / (ω0² − ω² + 2iζω0ω)), a
    /// band-pass peak of relative height g at f0 (half width ζ f0) that leaves DC and high frequencies unchanged, as
    /// in measured GIRFs (applied by FFT)
    Resonant { base: Box<Response>, f0: f64, zeta: f64, gain: f64 },
    /// a pure scale on top of a base shape
    Scaled { base: Box<Response>, k: f64 },
    /// k · (iωτ)/(1 + iωτ): an eddy-current term driven by dG/dt, decaying with τ (the shape of cross and B0 terms)
    HighPass { tau: f64, k: f64 },
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
            Response::Resonant { base, f0, zeta, gain } => {
                let (w, w0) = (2.0 * std::f64::consts::PI * f, 2.0 * std::f64::consts::PI * f0);
                let bp = Complex64::new(0.0, 2.0 * zeta * w0 * w) / Complex64::new(w0 * w0 - w * w, 2.0 * zeta * w0 * w);
                base.at(f) * (1.0 + *gain * bp)
            }
            Response::Scaled { base, k } => base.at(f) * *k,
            Response::HighPass { tau, k } => {
                let iwt = Complex64::new(0.0, 2.0 * std::f64::consts::PI * f * tau);
                *k * iwt / (1.0 + iwt)
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

impl Response {
    /// the discrete response for piecewise-linear signals on raster `dt`, sampled at the raster again: the input's
    /// spectrum is the samples' (periodic) spectrum times the hat's sinc²(f·dt), so the response seen by the samples
    /// is the alias sum Σ_m sinc²(f·dt + m) H(f + m/dt). Consistent with the PL moments of `wave::Grad` (plain H on the
    /// samples would treat them as band-limited and make every corner a spurious k error).
    pub fn at_pl(&self, f: f64, dt: f64) -> Complex64 {
        if let Response::Ideal = self {
            return Complex64::new(1.0, 0.0);
        }
        let sinc2 = |x: f64| if x.abs() < 1e-12 { 1.0 } else { ((std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)).powi(2) };
        let mut h = Complex64::new(0.0, 0.0);
        for m in -40i64..=40 {
            let w = sinc2(f * dt + m as f64);
            if w > 0.0 {
                h += self.at(f + m as f64 / dt) * w;
            }
        }
        h
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

    /// plausible, *different* gradient chains for robustness studies until a GIRF is measured (illustrative values,
    /// not a scanner): each adds one effect class to the previous one, so a design's sensitivity can be attributed.
    /// delays (axis differences 0.6–1.1 µs) → + first-order low-pass (7–9 µs) → + per-axis eddy currents (0.1–0.3 %,
    /// 50 µs – 3 ms) → + cross terms (0.05 % eddy-shaped) → + B0 eddy terms (1.5–3·10⁻⁵ m, 1–4 ms) → + mechanical
    /// resonances (1.25 kHz on y, 1.4 kHz on x, 0.9 kHz on z; ζ 0.03, band-pass peaks of 5–8 %)
    pub fn variants() -> Vec<(&'static str, Girf)> {
        let delays = [2.0e-6, 2.6e-6, 3.1e-6];
        let lp = [7e-6, 8e-6, 9e-6];
        let eddy = [vec![(0.0025, 2.5e-3), (0.001, 50e-6)], vec![(0.002, 3e-3), (0.0012, 80e-6)], vec![(0.003, 1e-3), (0.0008, 40e-6)]];
        let par = |a: usize, with_lp: bool, with_eddy: bool| Response::Parametric { delay: delays[a], tau_lp: if with_lp { lp[a] } else { 0.0 }, eddy: if with_eddy { eddy[a].clone() } else { vec![] } };
        let mk = |with_lp: bool, with_eddy: bool| -> [Response; 3] { [par(0, with_lp, with_eddy), par(1, with_lp, with_eddy), par(2, with_lp, with_eddy)] };
        let hp = |k: f64, tau: f64| Response::HighPass { tau, k };
        let cross = vec![(0, 2, hp(5e-4, 1e-3)), (1, 0, hp(5e-4, 2e-3)), (2, 1, hp(4e-4, 1.5e-3))];
        let b0 = [Some(hp(2e-5, 2e-3)), Some(hp(1.5e-5, 4e-3)), Some(hp(3e-5, 1e-3))];
        let res = |r: Response, f0: f64, gain: f64| Response::Resonant { base: Box::new(r), f0, zeta: 0.03, gain };
        let tail = 30e-3;
        let base = |axis: [Response; 3]| Girf { axis, cross: vec![], b0: [None, None, None], tail };
        vec![
            ("delays", base(mk(false, false))),
            ("lowpass", base(mk(true, false))),
            ("eddy", base(mk(true, true))),
            ("cross", Girf { cross: cross.clone(), ..base(mk(true, true)) }),
            ("b0eddy", Girf { cross: cross.clone(), b0: b0.clone(), ..base(mk(true, true)) }),
            ("resonance", {
                let [x, y, z] = mk(true, true);
                Girf { axis: [res(x, 1400.0, 0.06), res(y, 1250.0, 0.08), res(z, 900.0, 0.05)], cross, b0, tail }
            }),
        ]
    }

    /// the same chain known with relative error `eps` (a model of it, for "what is left after modelling the played
    /// gradients" studies): delays, low-pass and eddy / high-pass τ, eddy, cross and B0 amplitudes, resonance gain and
    /// damping scaled by (1 + eps) (time constants of eddy terms by (1 + eps/2)), resonance frequencies by (1 + eps/10)
    pub fn perturbed(&self, eps: f64) -> Girf {
        fn p(r: &Response, eps: f64) -> Response {
            match r {
                Response::Ideal => Response::Ideal,
                Response::Parametric { delay, tau_lp, eddy } => Response::Parametric {
                    delay: delay * (1.0 + eps),
                    tau_lp: tau_lp * (1.0 + eps),
                    eddy: eddy.iter().map(|&(a, t)| (a * (1.0 + eps), t * (1.0 + 0.5 * eps))).collect(),
                },
                Response::HighPass { tau, k } => Response::HighPass { tau: tau * (1.0 + 0.5 * eps), k: k * (1.0 + eps) },
                Response::Resonant { base, f0, zeta, gain } => Response::Resonant { base: Box::new(p(base, eps)), f0: f0 * (1.0 + 0.1 * eps), zeta: zeta * (1.0 + eps), gain: gain * (1.0 + eps) },
                Response::Scaled { base, k } => Response::Scaled { base: Box::new(p(base, eps)), k: *k },
                Response::Table { f, h } => Response::Table { f: f.clone(), h: h.iter().map(|v| Complex64::new(1.0, 0.0) + (v - 1.0) * (1.0 + eps)).collect() },
            }
        }
        Girf {
            axis: [p(&self.axis[0], eps), p(&self.axis[1], eps), p(&self.axis[2], eps)],
            cross: self.cross.iter().map(|(o, i, r)| (*o, *i, p(r, eps))).collect(),
            b0: [0, 1, 2].map(|a| self.b0[a].as_ref().map(|r| p(r, eps))),
            tail: self.tail,
        }
    }

    /// single-class model errors of relative size `eps` (each a model of this chain that is off in one parameter
    /// class only, same sign on every axis): independent errors whose effects add in quadrature. Classes absent from
    /// the chain are skipped.
    pub fn perturbations(&self, eps: f64) -> Vec<(&'static str, Girf)> {
        #[derive(Clone, Copy, PartialEq)]
        enum C {
            Delay,
            LowPass,
            EddyA,
            EddyTau,
            Cross,
            B0,
            ResF0,
            ResZeta,
            ResGain,
        }
        fn p(r: &Response, c: C, eps: f64, role: C) -> Response {
            let s = |on: bool| if on { 1.0 + eps } else { 1.0 };
            match r {
                Response::Parametric { delay, tau_lp, eddy } => Response::Parametric {
                    delay: delay * s(c == C::Delay),
                    tau_lp: tau_lp * s(c == C::LowPass),
                    eddy: eddy.iter().map(|&(a, t)| (a * s(c == C::EddyA), t * s(c == C::EddyTau))).collect(),
                },
                Response::HighPass { tau, k } => Response::HighPass { tau: tau * s(c == C::EddyTau), k: k * s(c == role) },
                Response::Resonant { base, f0, zeta, gain } => Response::Resonant {
                    base: Box::new(p(base, c, eps, role)),
                    f0: f0 * if c == C::ResF0 { 1.0 + 0.1 * eps } else { 1.0 },
                    zeta: zeta * s(c == C::ResZeta),
                    gain: gain * s(c == C::ResGain),
                },
                Response::Scaled { base, k } => Response::Scaled { base: Box::new(p(base, c, eps, role)), k: *k },
                other => other.clone(),
            }
        }
        let all = [
            ("delay", C::Delay),
            ("lowpass", C::LowPass),
            ("eddy_a", C::EddyA),
            ("eddy_tau", C::EddyTau),
            ("cross", C::Cross),
            ("b0", C::B0),
            ("res_f0", C::ResF0),
            ("res_zeta", C::ResZeta),
            ("res_gain", C::ResGain),
        ];
        let me = format!("{self:?}");
        all.iter()
            .filter_map(|&(name, c)| {
                let g = Girf {
                    axis: [0, 1, 2].map(|a| p(&self.axis[a], c, eps, C::Delay)),
                    cross: self.cross.iter().map(|(o, i, r)| (*o, *i, p(r, c, eps, C::Cross))).collect(),
                    b0: [0, 1, 2].map(|a| self.b0[a].as_ref().map(|r| p(r, c, eps, C::B0))),
                    tail: self.tail,
                };
                (format!("{g:?}") != me).then_some((name, g))
            })
            .collect()
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
                Response::HighPass { tau, k } => {
                    let l = lowpass_pl(&input[src], dt, *tau);
                    input[src].iter().zip(&l).map(|(x, y)| k * (x - y)).collect()
                }
                _ => {
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
                    let mut y: Vec<Complex64> = (0..nf).map(|k| spec[src][k] * resp.at_pl(freq(k), dt)).collect();
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

/// moments of one *played* TR at sample times, what a simulator / reconstruction needs per sample
#[derive(Clone, Copy, Debug)]
pub struct Played {
    /// M0, M1, M2 (time origin = the excitation) and P = ∫GGᵀ, all integrated from the excitation
    pub m0: Vec3,
    pub m1: Vec3,
    pub m2: Vec3,
    pub p: [f64; 6],
    /// phase of the B0 terms since the excitation, rad
    pub phi_b0: f64,
}

/// moments of the TR `this` (physical axes, raster from its start, t0 = 0) as played through `girf`, at `times` (s
/// from the TR start), integrated from the excitation at `t_exc`. `prev` = the samples of the TR before (its eddy
/// tails are still in the chain after this excitation; empty for the first TR). With `Girf::ideal()` this is the
/// nominal waveform's moments.
pub fn played_moments(girf: &Girf, prev: &[Vec3], this: &Grad, t_exc: f64, times: &[f64]) -> Vec<Played> {
    let dt = this.dt;
    let off = prev.len();
    let mut g = prev.to_vec();
    g.extend_from_slice(&this.g);
    let w = Grad { dt, t0: this.t0 - off as f64 * dt, g };
    let (p, b0) = girf.apply(&w);
    let mut all = vec![t_exc];
    all.extend_from_slice(times);
    let m = p.moments_at(t_exc, &all);
    let mut ph = vec![0.0; b0.len()];
    for i in 1..b0.len() {
        ph[i] = ph[i - 1] + GAMMA * 0.5 * (b0[i - 1] + b0[i]) * dt;
    }
    let phase_at = |t: f64| {
        let f = ((t - p.t0) / dt).clamp(0.0, (ph.len() - 1) as f64 - 1e-9);
        let i = f.floor() as usize;
        ph[i] + (ph[(i + 1).min(ph.len() - 1)] - ph[i]) * (f - i as f64)
    };
    let (z0, z1, z2, zp) = m[0];
    let p0 = phase_at(t_exc);
    times
        .iter()
        .zip(&m[1..])
        .map(|(&t, &(a, b, c, d))| Played {
            m0: [0, 1, 2].map(|q| a[q] - z0[q]),
            m1: [0, 1, 2].map(|q| b[q] - z1[q]),
            m2: [0, 1, 2].map(|q| c[q] - z2[q]),
            p: [0, 1, 2, 3, 4, 5].map(|q| d[q] - zp[q]),
            phi_b0: phase_at(t) - p0,
        })
        .collect()
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
    fn fft_and_time_domain_paths_agree() {
        // a resonance of gain 0 (FFT path) reproduces the time-domain parametric response; HighPass time domain
        // matches its frequency response through a Table
        let dt = 10e-6;
        let w = Grad::from_axis(0, &trap_shape(12, 40, 12, 0.02), dt, 0.0);
        let par = Response::Parametric { delay: 2.5e-6, tau_lp: 8e-6, eddy: vec![(0.002, 1e-3)] };
        let td = Girf { axis: [par.clone(), Response::Ideal, Response::Ideal], cross: vec![], b0: [None, None, None], tail: 10e-3 };
        let fd = Girf { axis: [Response::Resonant { base: Box::new(par), f0: 1200.0, zeta: 0.03, gain: 0.0 }, Response::Ideal, Response::Ideal], ..td.clone() };
        let (a, _) = td.apply(&w);
        let (b, _) = fd.apply(&w);
        let e = a.g.iter().zip(&b.g).map(|(p, q)| (p[0] - q[0]).abs()).fold(0.0, f64::max);
        // both paths treat the input as piecewise linear (the FFT path through the alias sum of `at_pl`); what is
        // left is the time-domain path's PL treatment of its intermediate (low-passed) signal, O(dt²)
        // what is left is the time-domain path's delay: linear interpolation of the samples at shifted times is not
        // the delayed PL waveform at the corners (≈ 0.5 % of G for a quarter-raster delay); without the delay the paths
        // agree to the cascade's O(dt²)
        eprintln!("time vs FFT: max |Δg| {e:.2e} of 0.02");
        assert!(e < 7e-3 * 0.02, "time vs FFT {e}");
        let par0 = Response::Parametric { delay: 0.0, tau_lp: 8e-6, eddy: vec![(0.002, 1e-3)] };
        let td0 = Girf { axis: [par0.clone(), Response::Ideal, Response::Ideal], ..td.clone() };
        let fd0 = Girf { axis: [Response::Resonant { base: Box::new(par0), f0: 1200.0, zeta: 0.03, gain: 0.0 }, Response::Ideal, Response::Ideal], ..td.clone() };
        let (a0, _) = td0.apply(&w);
        let (b0_, _) = fd0.apply(&w);
        let e0 = a0.g.iter().zip(&b0_.g).map(|(p, q)| (p[0] - q[0]).abs()).fold(0.0, f64::max);
        eprintln!("time vs FFT without delay: max |Δg| {e0:.2e} of 0.02");
        assert!(e0 < 5e-4 * 0.02, "time vs FFT without delay {e0}");
        let hp = Response::HighPass { tau: 1e-3, k: 0.01 };
        let fs: Vec<f64> = (0..=4000).map(|i| i as f64 * 12.5).collect();
        let tab = Response::Table { h: fs.iter().map(|&f| hp.at(f)).collect(), f: fs };
        let g1 = Girf { b0: [Some(hp), None, None], ..Girf::ideal() };
        let g2 = Girf { b0: [Some(tab), None, None], ..Girf::ideal() };
        let g1 = Girf { tail: 10e-3, ..g1 };
        let g2 = Girf { tail: 10e-3, ..g2 };
        let (_, b1) = g1.apply(&w);
        let (_, b2) = g2.apply(&w);
        let m = b1.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let e2 = b1.iter().zip(&b2).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(m > 0.0 && e2 < 0.05 * m, "high-pass time vs table {e2} of {m}");
    }

    #[test]
    fn played_moments_of_the_ideal_chain_are_the_nominal_ones() {
        let dt = 10e-6;
        let this = Grad::from_axes(&trap_shape(5, 20, 5, 0.01), &[0.0; 31], &trap_shape(8, 10, 8, -0.02), dt, 0.0);
        let t_exc = 2.0 * dt;
        let times: Vec<f64> = (0..25).map(|i| 3e-5 + i as f64 * 1.3e-5).collect();
        let pl = played_moments(&Girf::ideal(), &[[0.0; 3]; 7], &this, t_exc, &times);
        let nom = this.moments_at(t_exc, &times);
        let z = this.moments_at(t_exc, &[t_exc])[0];
        for (a, b) in pl.iter().zip(&nom) {
            for q in 0..3 {
                assert!((a.m0[q] - (b.0[q] - z.0[q])).abs() < 1e-15 && (a.m1[q] - (b.1[q] - z.1[q])).abs() < 1e-18, "{a:?} vs {b:?}");
            }
        }
        // a previous TR's eddy tail shows up after this excitation
        let g = Girf { axis: [Response::Parametric { delay: 0.0, tau_lp: 0.0, eddy: vec![(0.01, 1e-3)] }, Response::Ideal, Response::Ideal], tail: 0.0, ..Girf::ideal() };
        let prev: Vec<[f64; 3]> = trap_shape(5, 200, 5, 0.02).iter().map(|&v| [v, 0.0, 0.0]).collect();
        let zero = Grad::zeros(40, dt, 0.0);
        let q = played_moments(&g, &prev, &zero, t_exc, &times);
        assert!(q.last().unwrap().m0[0].abs() > 1e-9, "{:?}", q.last().unwrap().m0);
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
