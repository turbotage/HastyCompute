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

// ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// Additions (research/recon_physics_wiring.md, 2026-11-11): the played trajectory with the chain's TR history
// (`PlayedStream`, LTI superposition of precomputed block responses) and measured / estimated responses from a table
// file (`Girf::from_measurement`, format `hasty-girf-1`). Nothing above this line is changed.
// ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────

use crate::Mat3;

impl Girf {
    /// [`Girf::apply`] for several inputs of equal length at once (physical axes, raster `dt`, input sample 0 at raster
    /// index 0): the outputs cover raster indices −PRE .. n_out − 1 (n_out ≥ the input length; the rest is the
    /// response tail), with the B0 field. Same arithmetic per response type as `apply` (parametric responses in the
    /// time domain, tables / resonances by zero-padded FFT through `at_pl`), but every FFT-path transfer function is
    /// evaluated once for the whole batch.
    pub fn apply_batch(&self, inputs: &[Vec<Vec3>], dt: f64, n_out: usize) -> Vec<(Vec<Vec3>, Vec<f64>)> {
        let pre = Self::PRE;
        let n = pre + n_out;
        let nin = inputs.iter().map(|g| g.len()).max().unwrap_or(0);
        assert!(nin <= n_out, "apply_batch: the output must cover the input");
        let chan = |g: &[Vec3], a: usize| -> Vec<f64> { (0..n).map(|i| if i >= pre && i - pre < g.len() { g[i - pre][a] } else { 0.0 }).collect() };
        let nfft = (2 * n).next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let (fwd, inv) = (planner.plan_fft_forward(nfft), planner.plan_fft_inverse(nfft));
        let freq = |k: usize| if k <= nfft / 2 { k as f64 / (nfft as f64 * dt) } else { (k as f64 - nfft as f64) / (nfft as f64 * dt) };
        let fft_path = |r: &Response| !matches!(r, Response::Ideal | Response::Parametric { .. } | Response::HighPass { .. });
        // transfer functions of the FFT-path responses, once
        let mut resp: Vec<&Response> = self.axis.iter().collect();
        resp.extend(self.cross.iter().map(|(_, _, r)| r));
        resp.extend(self.b0.iter().flatten());
        let tf: Vec<Option<Vec<Complex64>>> = resp.iter().map(|r| fft_path(r).then(|| (0..nfft).map(|k| r.at_pl(freq(k), dt)).collect())).collect();
        let filter = |r: &Response, h: &Option<Vec<Complex64>>, x: &[f64], spec: &mut Option<Vec<Complex64>>| -> Vec<f64> {
            match r {
                Response::Ideal => x.to_vec(),
                Response::Parametric { delay, tau_lp, eddy } => time_domain(x, dt, *delay, *tau_lp, eddy),
                Response::HighPass { tau, k } => {
                    let l = lowpass_pl(x, dt, *tau);
                    x.iter().zip(&l).map(|(a, b)| k * (a - b)).collect()
                }
                _ => {
                    let s = spec.get_or_insert_with(|| {
                        let mut v: Vec<Complex64> = (0..nfft).map(|i| Complex64::new(if i < n { x[i] } else { 0.0 }, 0.0)).collect();
                        fwd.process(&mut v);
                        v
                    });
                    let h = h.as_ref().unwrap();
                    let mut y: Vec<Complex64> = (0..nfft).map(|k| s[k] * h[k]).collect();
                    inv.process(&mut y);
                    y.iter().take(n).map(|c| c.re / nfft as f64).collect()
                }
            }
        };
        let na = self.axis.len();
        let nx = self.cross.len();
        crate::par_map(inputs, |g| {
            let x: [Vec<f64>; 3] = [chan(g, 0), chan(g, 1), chan(g, 2)];
            let mut spec: [Option<Vec<Complex64>>; 3] = [None, None, None];
            let mut out = vec![[0.0; 3]; n];
            for a in 0..3 {
                let y = filter(&self.axis[a], &tf[a], &x[a], &mut spec[a]);
                for i in 0..n {
                    out[i][a] += y[i];
                }
            }
            for (j, (oa, ia, r)) in self.cross.iter().enumerate() {
                let y = filter(r, &tf[na + j], &x[*ia], &mut spec[*ia]);
                for i in 0..n {
                    out[i][*oa] += y[i];
                }
            }
            let mut b0 = vec![0.0; n];
            let mut j = na + nx;
            for a in 0..3 {
                if let Some(r) = &self.b0[a] {
                    let y = filter(r, &tf[j], &x[a], &mut spec[a]);
                    j += 1;
                    for i in 0..n {
                        b0[i] += y[i];
                    }
                }
            }
            (out, b0)
        })
    }
}

/// The played trajectory of a periodic TR sequence with the chain's history. The chain is linear and time-invariant,
/// so the played gradient of TR i is the sum of the responses to every earlier TR, each a shifted copy of the
/// response of its block. A TR of type e with readout rotation R is split into its rotation-independent part F_e
/// (encode / tail / spoiler on physical axes, readout samples zeroed) and the rotated base readout R·B placed at raster
/// index `i_ro`; by linearity chain(R·B) = Σ_b Σ_c R_bc chain(B_c on input axis b). The responses of every F_e and of
/// the nine (b, c) pairs are computed once (over `hist` + 1 TRs), so each readout costs (hist + 1)·n_tr·(3 + 27)
/// multiply-adds plus the moment integration, whatever the chain. Exact for the LTI model (the same responses as
/// [`Girf::apply`] on the concatenated waveform, up to rounding and the FFT path's tail truncation).
/// With `hist` previous TRs (10 ≈ 141 ms for the v2 flow TR) eddy terms up to ≈ 30 ms are carried (research/
/// girf_sensitivity.md: 2 vs 10 TRs differ by 0.08 cycles/m rms with 10–20 ms eddy terms).
pub struct PlayedStream {
    pub dt: f64,
    /// raster samples of one TR (TR j starts at raster j·n_tr)
    pub n_tr: usize,
    /// previous TRs carried
    pub hist: usize,
    /// raster index of the readout start in the TR
    pub i_ro: usize,
    /// response of each TR type's fixed part over (hist + 1) TRs, from raster −PRE: (gradient, B0 field)
    fixed: Vec<(Vec<Vec3>, Vec<f64>)>,
    /// response of base component c on input axis b (index 3b + c), input start = the TR start (readout at i_ro)
    base: Vec<(Vec<Vec3>, Vec<f64>)>,
}

impl PlayedStream {
    /// `tr_fixed[e]`: TR type e on physical axes with its readout samples zeroed (n_tr + 1 samples from the TR start;
    /// the last sample, = the next TR's first, must be zero); `base`: the base readout (unrotated) whose sample i is
    /// played at raster i_ro + i; `hist`: previous TRs carried.
    pub fn new(girf: &Girf, tr_fixed: &[Vec<Vec3>], base: &[Vec3], i_ro: usize, n_tr: usize, dt: f64, hist: usize) -> Self {
        assert!(i_ro + base.len() <= n_tr + 1, "PlayedStream: the readout must lie inside the TR");
        let len = (hist + 1) * n_tr + 1;
        let mut inputs: Vec<Vec<Vec3>> = tr_fixed
            .iter()
            .map(|g| {
                assert!(g.len() <= n_tr + 1, "PlayedStream: TR type longer than n_tr + 1 samples");
                let mut v = g.clone();
                if v.len() == n_tr + 1 {
                    assert!(norm(&v[n_tr]) == 0.0, "PlayedStream: the last TR sample (the next TR's first) must be zero");
                }
                v.truncate(n_tr);
                v
            })
            .collect();
        for b in 0..3 {
            for c in 0..3 {
                let mut v = vec![[0.0; 3]; i_ro + base.len()];
                for (i, s) in base.iter().enumerate() {
                    v[i_ro + i][b] = s[c];
                }
                inputs.push(v);
            }
        }
        // the chain's own tail beyond the window keeps the FFT path's circular convolution from wrapping into it
        let ntail = (girf.tail / dt).ceil() as usize;
        let mut out = girf.apply_batch(&inputs, dt, len + ntail);
        let base_r = out.split_off(tr_fixed.len());
        Self { dt, n_tr, hist, i_ro, fixed: out, base: base_r }
    }

    /// TR types
    pub fn n_types(&self) -> usize {
        self.fixed.len()
    }

    /// Played gradient (physical axes) and B0 field (T) over the window of the last TR of `seq`, raster 0 ..= n_tr
    /// from its start (t0 = 0). `seq` = (TR type, readout rotation) of consecutive TRs in acquisition order, the last
    /// one being the TR played; at most hist + 1 are used (older ones are dropped), fewer = the acquisition started
    /// from rest.
    pub fn played_tr(&self, seq: &[(usize, Mat3)]) -> (Grad, Vec<f64>) {
        let n = self.n_tr + 1;
        let pre = Girf::PRE;
        let use_n = seq.len().min(self.hist + 1);
        let (mut g, mut b0) = (vec![[0.0; 3]; n], vec![0.0; n]);
        for (d, (e, r)) in seq.iter().rev().take(use_n).enumerate() {
            // the TR d TRs before: our sample s is its response sample d·n_tr + s
            let off = pre + d * self.n_tr;
            let (fg, fb) = &self.fixed[*e];
            for s in 0..n {
                let v = fg[off + s];
                g[s][0] += v[0];
                g[s][1] += v[1];
                g[s][2] += v[2];
                b0[s] += fb[off + s];
            }
            for b in 0..3 {
                for c in 0..3 {
                    let w = r[b][c];
                    if w == 0.0 {
                        continue;
                    }
                    let (bg, bb) = &self.base[3 * b + c];
                    for s in 0..n {
                        let v = bg[off + s];
                        g[s][0] += w * v[0];
                        g[s][1] += w * v[1];
                        g[s][2] += w * v[2];
                        b0[s] += w * bb[off + s];
                    }
                }
            }
        }
        (Grad { dt: self.dt, t0: 0.0, g }, b0)
    }

    /// moments of the last TR of `seq` at `times` (s from its start), integrated from the excitation `t_exc`, as
    /// [`played_moments`] returns them
    pub fn moments(&self, seq: &[(usize, Mat3)], t_exc: f64, times: &[f64]) -> Vec<Played> {
        let (p, b0) = self.played_tr(seq);
        played_from(&p, &b0, t_exc, times)
    }
}

/// moments of a played window `p` (and its B0 field on the same raster) at `times`, integrated from `t_exc`
pub fn played_from(p: &Grad, b0: &[f64], t_exc: f64, times: &[f64]) -> Vec<Played> {
    let dt = p.dt;
    let mut all = vec![t_exc];
    all.extend_from_slice(times);
    let m = p.moments_at(t_exc, &all);
    let mut ph = vec![0.0; b0.len()];
    for i in 1..b0.len() {
        ph[i] = ph[i - 1] + GAMMA * 0.5 * (b0[i - 1] + b0[i]) * dt;
    }
    let phase_at = |t: f64| {
        if ph.len() < 2 {
            return 0.0;
        }
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

fn norm(v: &Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// One resonance of a measured / estimated chain (metadata of a `hasty-girf-1` file: the parametric fit's list, the
/// starting point of a reconstruction refit of f0 / height). On axis `axis` (self term), the factor
/// 1 + gain · 2iζω0ω / (ω0² − ω² + 2iζω0ω) of [`Response::Resonant`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResonanceSpec {
    pub axis: usize,
    pub f0: f64,
    pub zeta: f64,
    pub gain: f64,
}

impl ResonanceSpec {
    /// the band-pass factor 1 + g·bp(f)
    pub fn factor(&self, f: f64) -> Complex64 {
        let (w, w0) = (2.0 * std::f64::consts::PI * f, 2.0 * std::f64::consts::PI * self.f0);
        let bp = Complex64::new(0.0, 2.0 * self.zeta * w0 * w) / Complex64::new(w0 * w0 - w * w, 2.0 * self.zeta * w0 * w);
        1.0 + self.gain * bp
    }
}

/// Content of a `hasty-girf-1` file: the chain (self terms, cross terms, B0 terms as [`Response::Table`] of the
/// continuous response) and its metadata.
#[derive(Clone, Debug)]
pub struct GirfMeasurement {
    pub girf: Girf,
    pub resonances: Vec<ResonanceSpec>,
    pub note: String,
}

impl Girf {
    /// Load a measured / estimated chain from a `hasty-girf-1` JSON file (format in research/recon_physics_wiring.md):
    /// one frequency grid `f` (Hz, ascending, ≥ 0), per axis the complex continuous self response, optional cross
    /// terms {out, in, h} and B0 terms (T per T/m), the tail (s). Arrays inline ({"re": [..], "im": [..]}) or `.npy`
    /// files next to the JSON (complex128 / complex64, or float64 for `f`). Every term becomes a
    /// [`Response::Table`] (linear interpolation of re / im, Hermitian, held constant beyond the grid — tabulate to
    /// ≳ 40 / dt so the raster alias sum of `at_pl` sees the real roll-off).
    pub fn from_measurement(path: &std::path::Path) -> Result<Girf, String> {
        GirfMeasurement::load(path).map(|m| m.girf)
    }

    /// Write the chain as a `hasty-girf-1` JSON file (inline arrays), every response tabulated at `f` (Hz, ascending,
    /// ≥ 0); `resonances` / `note` as metadata.
    pub fn to_measurement(&self, path: &std::path::Path, f: &[f64], resonances: &[ResonanceSpec], note: &str) -> Result<(), String> {
        let tab = |r: &Response| -> serde_json::Value {
            let h: Vec<Complex64> = f.iter().map(|&x| r.at(x)).collect();
            serde_json::json!({"re": h.iter().map(|c| c.re).collect::<Vec<_>>(), "im": h.iter().map(|c| c.im).collect::<Vec<_>>()})
        };
        let j = serde_json::json!({
            "format": "hasty-girf-1",
            "convention": "continuous",
            "f": f,
            "self": self.axis.iter().map(tab).collect::<Vec<_>>(),
            "cross": self.cross.iter().map(|(o, i, r)| serde_json::json!({"out": o, "in": i, "h": tab(r)})).collect::<Vec<_>>(),
            "b0": self.b0.iter().map(|r| r.as_ref().map_or(serde_json::Value::Null, tab)).collect::<Vec<_>>(),
            "tail": self.tail,
            "resonances": resonances.iter().map(|r| serde_json::json!({"axis": r.axis, "f0": r.f0, "zeta": r.zeta, "gain": r.gain})).collect::<Vec<_>>(),
            "note": note,
        });
        std::fs::write(path, serde_json::to_string(&j).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))
    }
}

impl GirfMeasurement {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let err = |m: String| format!("{}: {m}", path.display());
        let txt = std::fs::read_to_string(path).map_err(|e| err(e.to_string()))?;
        let j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| err(e.to_string()))?;
        if j["format"] != "hasty-girf-1" {
            return Err(err("format is not hasty-girf-1".into()));
        }
        if !(j["convention"].is_null() || j["convention"] == "continuous") {
            return Err(err(format!("convention {} (only 'continuous' is supported)", j["convention"])));
        }
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        let reals = |v: &serde_json::Value| -> Result<Vec<f64>, String> {
            match v {
                serde_json::Value::String(s) => npy_1d(&dir.join(s)).and_then(|(re, im)| if im.iter().all(|x| *x == 0.0) { Ok(re) } else { Err(format!("{s}: complex where real expected")) }),
                serde_json::Value::Array(a) => a.iter().map(|x| x.as_f64().ok_or("non-numeric entry".to_string())).collect(),
                _ => Err("expected an array or an .npy file name".into()),
            }
        };
        let cplx = |v: &serde_json::Value| -> Result<Vec<Complex64>, String> {
            match v {
                serde_json::Value::String(s) => npy_1d(&dir.join(s)).map(|(re, im)| re.into_iter().zip(im).map(|(a, b)| Complex64::new(a, b)).collect()),
                serde_json::Value::Object(_) => {
                    let re = reals(&v["re"])?;
                    let im = if v["im"].is_null() { vec![0.0; re.len()] } else { reals(&v["im"])? };
                    if re.len() != im.len() {
                        return Err("re / im lengths differ".into());
                    }
                    Ok(re.into_iter().zip(im).map(|(a, b)| Complex64::new(a, b)).collect())
                }
                _ => Err("expected {re, im} or an .npy file name".into()),
            }
        };
        let f = reals(&j["f"]).map_err(err)?;
        if f.is_empty() || f[0] < 0.0 || !f.windows(2).all(|w| w[1] > w[0]) {
            return Err(err("f must be non-negative and strictly ascending".into()));
        }
        let table = |v: &serde_json::Value| -> Result<Response, String> {
            let ff = if v.get("f").is_some() && v.get("h").is_some() { reals(&v["f"])? } else { f.clone() };
            let h = cplx(v.get("h").unwrap_or(v))?;
            if h.len() != ff.len() {
                return Err(format!("table of {} values on {} frequencies", h.len(), ff.len()));
            }
            Ok(Response::Table { f: ff, h })
        };
        let s = j["self"].as_array().ok_or_else(|| err("self: three responses expected".into()))?;
        if s.len() != 3 {
            return Err(err("self: three responses expected".into()));
        }
        let axis = [table(&s[0]).map_err(err)?, table(&s[1]).map_err(err)?, table(&s[2]).map_err(err)?];
        let mut cross = vec![];
        for c in j["cross"].as_array().map(|v| v.as_slice()).unwrap_or(&[]) {
            let (o, i) = (c["out"].as_u64().ok_or_else(|| err("cross.out".into()))? as usize, c["in"].as_u64().ok_or_else(|| err("cross.in".into()))? as usize);
            if o > 2 || i > 2 || o == i {
                return Err(err(format!("cross term {o} <- {i}")));
            }
            cross.push((o, i, table(&c["h"]).map_err(err)?));
        }
        let mut b0: [Option<Response>; 3] = [None, None, None];
        if let Some(b) = j["b0"].as_array() {
            for (a, v) in b.iter().enumerate().take(3) {
                if !v.is_null() {
                    b0[a] = Some(table(v).map_err(err)?);
                }
            }
        }
        let tail = j["tail"].as_f64().unwrap_or(30e-3);
        let resonances = j["resonances"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|r| ResonanceSpec { axis: r["axis"].as_u64().unwrap_or(0) as usize, f0: r["f0"].as_f64().unwrap_or(0.0), zeta: r["zeta"].as_f64().unwrap_or(0.0), gain: r["gain"].as_f64().unwrap_or(0.0) })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self { girf: Girf { axis, cross, b0, tail }, resonances, note: j["note"].as_str().unwrap_or("").to_string() })
    }
}

/// a 1-D `.npy` array (C order, little-endian: float64, complex128, complex64; or float64 [n, 2] = re, im) → (re, im)
fn npy_1d(path: &std::path::Path) -> Result<(Vec<f64>, Vec<f64>), String> {
    let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if buf.len() < 10 || &buf[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not npy", path.display()));
    }
    let (hlen, hs) = if buf[6] == 1 { (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10) } else { (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12) };
    let h = std::str::from_utf8(&buf[hs..hs + hlen]).map_err(|e| e.to_string())?;
    if h.contains("'fortran_order': True") {
        return Err(format!("{}: Fortran order", path.display()));
    }
    let descr = h.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).unwrap_or("").to_string();
    let shp: Vec<usize> = h.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).unwrap_or("").split(',').filter_map(|t| t.trim().parse().ok()).collect();
    let d = &buf[hs + hlen..];
    let f8 = |i: usize| f64::from_le_bytes(d[8 * i..8 * i + 8].try_into().unwrap());
    let f4 = |i: usize| f32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as f64;
    match (descr.as_str(), shp.as_slice()) {
        ("<f8", [n]) => Ok(((0..*n).map(f8).collect(), vec![0.0; *n])),
        ("<f8", [n, 2]) => Ok(((0..*n).map(|i| f8(2 * i)).collect(), (0..*n).map(|i| f8(2 * i + 1)).collect())),
        ("<c16", [n]) => Ok(((0..*n).map(|i| f8(2 * i)).collect(), (0..*n).map(|i| f8(2 * i + 1)).collect())),
        ("<c8", [n]) => Ok(((0..*n).map(|i| f4(2 * i)).collect(), (0..*n).map(|i| f4(2 * i + 1)).collect())),
        (d, s) => Err(format!("{}: dtype {d} shape {s:?} (1-D <f8 / <c16 / <c8 or [n, 2] <f8)", path.display())),
    }
}

#[cfg(test)]
mod tests_stream {
    use super::*;
    use crate::wave::trap_shape;
    use crate::{mat_vec, quat_to_mat, Mat3};

    /// a small periodic "TR": encode-like lobes on all axes (type-dependent), a readout of 60 samples at i_ro
    fn toy(n_types: usize) -> (Vec<Vec<Vec3>>, Vec<Vec3>, usize, usize) {
        let n_tr = 140;
        let i_ro = 40;
        let fixed: Vec<Vec<Vec3>> = (0..n_types)
            .map(|e| {
                let s = 1.0 + 0.3 * e as f64;
                let mut g = vec![[0.0; 3]; n_tr + 1];
                for (i, v) in trap_shape(3, 10, 3, 0.012 * s).iter().enumerate() {
                    g[5 + i][0] = *v;
                    g[8 + i][1] = -0.7 * v;
                    g[110 + i][2] = 0.9 * v;
                    g[118 + i][0] = -0.5 * v;
                }
                g
            })
            .collect();
        let base: Vec<Vec3> = (0..60).map(|i| {
            let t = i as f64 / 59.0;
            let r = (std::f64::consts::PI * t).sin();
            [0.02 * r * (6.0 * t).cos(), 0.015 * r * (5.0 * t).sin(), 0.01 * r]
        }).collect();
        (fixed, base, i_ro, n_tr)
    }

    fn tr_wave(fixed: &[Vec<Vec3>], base: &[Vec3], i_ro: usize, e: usize, r: &Mat3) -> Vec<Vec3> {
        let mut g = fixed[e].clone();
        for (i, v) in base.iter().enumerate() {
            g[i_ro + i] = mat_vec(r, v);
        }
        g
    }

    fn chain() -> Girf {
        // every response type: time-domain parametric, high-pass cross / B0 terms, FFT-path resonances and a table
        let mut g = Girf::variants().into_iter().find(|v| v.0 == "resonance").unwrap().1;
        let fs: Vec<f64> = (0..=20000).map(|i| i as f64 * 25.0).chain((1..=200).map(|i| 5e5 + i as f64 * 2.5e4)).collect();
        let tab = Response::Table { h: fs.iter().map(|&f| g.axis[1].at(f) * Complex64::from_polar(1.0, 1e-3 * (f / 3e3).sin())).collect(), f: fs };
        g.axis[1] = tab;
        g.tail = 40e-3;
        g
    }

    #[test]
    fn stream_matches_the_concatenated_waveform() {
        let dt = 10e-6;
        let (fixed, base, i_ro, n_tr) = toy(3);
        let g = chain();
        let rots: Vec<Mat3> = (0..6).map(|j| quat_to_mat([1.0, 0.3 * j as f64, -0.2, 0.1 * j as f64])).collect();
        let seq: Vec<(usize, Mat3)> = (0..6).map(|j| (j % 3, rots[j])).collect();
        let t_exc = 20e-6;
        let times: Vec<f64> = (0..50).map(|i| i_ro as f64 * dt + i as f64 * 1.3e-5).collect();
        for hist in [1usize, 3] {
            let ps = PlayedStream::new(&g, &fixed, &base, i_ro, n_tr, dt, hist);
            for last in [0usize, 2, 5] {
                let s = &seq[..=last];
                let a = ps.moments(s, t_exc, &times);
                // reference: the previous min(hist, last) TRs concatenated through `played_moments`
                let first = last.saturating_sub(hist);
                let mut prev: Vec<Vec3> = vec![];
                for (e, r) in &seq[first..last] {
                    let mut w = tr_wave(&fixed, &base, i_ro, *e, r);
                    w.truncate(n_tr);
                    prev.extend(w);
                }
                let this = Grad { dt, t0: 0.0, g: tr_wave(&fixed, &base, i_ro, seq[last].0, &seq[last].1) };
                let b = played_moments(&g, &prev, &this, t_exc, &times);
                let kmax = b.iter().map(|p| norm(&p.m0)).fold(0.0, f64::max);
                let (mut dk, mut dp, mut dw) = (0.0f64, 0.0f64, 0.0f64);
                for (p, q) in a.iter().zip(&b) {
                    dk = dk.max(norm(&[p.m0[0] - q.m0[0], p.m0[1] - q.m0[1], p.m0[2] - q.m0[2]]));
                    dp = dp.max((p.phi_b0 - q.phi_b0).abs());
                    dw = dw.max((0..6).map(|k| (p.p[k] - q.p[k]).abs()).fold(0.0, f64::max));
                }
                let pmax = b.iter().flat_map(|p| p.p).map(f64::abs).fold(0.0, f64::max);
                eprintln!("hist {hist} last {last}: max |ΔM0| {:.2e} of {kmax:.2e}, |Δφ_B0| {dp:.2e} rad, |ΔP| {:.2e} of {pmax:.2e}", dk, dw);
                assert!(dk < 1e-9 * kmax, "M0 {dk} of {kmax}");
                assert!(dp < 1e-9, "B0 phase {dp}");
                assert!(dw < 1e-9 * pmax, "P {dw} of {pmax}");
            }
        }
    }

    #[test]
    fn apply_batch_equals_apply() {
        let dt = 10e-6;
        let g = chain();
        let w: Vec<Vec3> = (0..300).map(|i| [((i as f64) * 0.05).sin() * 0.01, ((i as f64) * 0.031).cos() * 0.008 - 0.008, if i > 150 { 0.01 } else { 0.0 }]).collect();
        let (a, ab0) = g.apply(&Grad { dt, t0: 0.0, g: w.clone() });
        let n_out = a.len() - Girf::PRE;
        let out = g.apply_batch(&[w], dt, n_out);
        let (b, bb0) = &out[0];
        let e = a.g.iter().zip(b).map(|(p, q)| (0..3).map(|k| (p[k] - q[k]).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        let eb = ab0.iter().zip(bb0).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(e == 0.0 && eb == 0.0, "batch vs apply: {e:.3e}, B0 {eb:.3e}");
    }

    #[test]
    fn ideal_stream_is_the_nominal_trajectory() {
        let dt = 10e-6;
        let (fixed, base, i_ro, n_tr) = toy(2);
        let ps = PlayedStream::new(&Girf::ideal(), &fixed, &base, i_ro, n_tr, dt, 4);
        let r = quat_to_mat([0.9, 0.1, 0.4, -0.3]);
        let seq = vec![(1usize, r), (0usize, r)];
        let times: Vec<f64> = (0..40).map(|i| i_ro as f64 * dt + i as f64 * 1.1e-5).collect();
        let a = ps.moments(&seq, 15e-6, &times);
        let nom = Grad { dt, t0: 0.0, g: tr_wave(&fixed, &base, i_ro, 0, &r) };
        let b = played_moments(&Girf::ideal(), &[], &nom, 15e-6, &times);
        for (p, q) in a.iter().zip(&b) {
            for k in 0..3 {
                assert!((p.m0[k] - q.m0[k]).abs() < 1e-15 && (p.m1[k] - q.m1[k]).abs() < 1e-18);
            }
        }
    }

    #[test]
    fn measurement_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("hasty_girf_meas_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let g = Girf::variants().into_iter().find(|v| v.0 == "resonance").unwrap().1;
        let f: Vec<f64> = (0..=4000).map(|i| i as f64 * 5.0).chain((1..=400).map(|i| 2e4 * (250.0f64).powf(i as f64 / 400.0))).collect();
        let res = vec![ResonanceSpec { axis: 0, f0: 1400.0, zeta: 0.03, gain: 0.06 }];
        g.to_measurement(&dir.join("g.json"), &f, &res, "test").unwrap();
        let m = GirfMeasurement::load(&dir.join("g.json")).unwrap();
        assert_eq!(m.resonances, res);
        assert_eq!(m.girf.cross.len(), g.cross.len());
        for fq in [0.0, 13.0, 777.0, 1400.0, 1e4, 3e5] {
            for a in 0..3 {
                let (p, q) = (g.axis[a].at(fq), m.girf.axis[a].at(fq));
                assert!((p - q).norm() < 2e-3 * p.norm().max(1e-6), "axis {a} f {fq}: {p} vs {q}");
            }
            let (p, q) = (g.b0[2].as_ref().unwrap().at(fq), m.girf.b0[2].as_ref().unwrap().at(fq));
            assert!((p - q).norm() < 1e-2 * p.norm().max(1e-12) + 1e-12, "b0 f {fq}: {p} vs {q}");
        }
        // the played gradient through the table chain is close to the parametric one (table interpolation only)
        let w = Grad::from_axis(2, &trap_shape(20, 100, 20, 0.02), 10e-6, 0.0);
        let (a, _) = Girf { tail: 3e-3, ..g.clone() }.apply(&w);
        let (b, _) = Girf { tail: 3e-3, ..m.girf.clone() }.apply(&w);
        let e = a.g.iter().zip(&b.g).map(|(p, q)| (p[2] - q[2]).abs()).fold(0.0, f64::max);
        assert!(e < 2e-3 * 0.02, "table vs parametric chain: {e:.2e} T/m");
        // npy arrays: f as float64, a self term as complex128
        let npy = |name: &str, descr: &str, n: usize, bytes: Vec<u8>| {
            let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}").into_bytes();
            while (10 + h.len() + 1) % 64 != 0 {
                h.push(b' ');
            }
            h.push(b'\n');
            let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
            out.extend_from_slice(&(h.len() as u16).to_le_bytes());
            out.extend_from_slice(&h);
            out.extend_from_slice(&bytes);
            std::fs::write(dir.join(name), out).unwrap();
        };
        let fs = [0.0f64, 100.0, 1e6];
        npy("f.npy", "<f8", 3, fs.iter().flat_map(|x| x.to_le_bytes()).collect());
        npy("h.npy", "<c16", 3, [1.0f64, 0.0, 0.9, -0.1, 0.5, 0.0].iter().flat_map(|x| x.to_le_bytes()).collect());
        std::fs::write(dir.join("n.json"), r#"{"format": "hasty-girf-1", "f": "f.npy", "self": ["h.npy", {"re": [1, 1, 1]}, {"re": [1, 1, 1], "im": [0, 0, 0]}], "tail": 0.01}"#).unwrap();
        let n = Girf::from_measurement(&dir.join("n.json")).unwrap();
        assert!((n.axis[0].at(100.0) - Complex64::new(0.9, -0.1)).norm() < 1e-12 && (n.axis[0].at(-100.0) - Complex64::new(0.9, 0.1)).norm() < 1e-12);
        assert!(n.b0.iter().all(|b| b.is_none()) && n.cross.is_empty());
        std::fs::write(dir.join("bad.json"), r#"{"format": "x"}"#).unwrap();
        assert!(Girf::from_measurement(&dir.join("bad.json")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
