//! Initialisation of the velocity model from a time-averaged (warm-start) reconstruction
//! (`research/velocity_model_sr.md` §A3 step 1):
//! - residual phase: per encode e ≥ 1, the polynomial φ_e fitted to the encode phase difference d_e = arg(x_e x̄_0)
//!   over the object, by iteratively reweighted least squares (Tukey biweight on the wrapped residual, signal
//!   weight |x_0||x_e|). In static tissue d_e = φ_e; vessels and noise are outliers and get weight 0.
//! - static-tissue mask: voxels that every encode's fit accepts (robust weight > ½) with enough signal; it anchors
//!   v ≈ 0 against the residual phase (the degeneracy of §A3) without simulator truth.
//! - velocity: the hierarchical decoder on the phase-corrected images.

use num_complex::Complex64 as C;

use crate::flow::FlowDecoder;
use crate::vmodel::{poly_at, poly_exps};

pub struct StaticFit {
    /// a[(e − 1)·np + p] (rad), the order of `vmodel::poly_exps`
    pub a: Vec<f64>,
    /// 1 = static tissue
    pub mask: Vec<f32>,
    /// robust residual scale (rad) per encode e ≥ 1
    pub sigma: Vec<f64>,
}

fn wrap(x: f64) -> f64 {
    let t = std::f64::consts::TAU;
    x - t * ((x + std::f64::consts::PI) / t).floor()
}

fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    // Gaussian elimination with partial pivoting (n ≤ 20), small ridge for safety
    let tr: f64 = (0..n).map(|i| a[i * n + i]).sum::<f64>() / n as f64;
    for i in 0..n {
        a[i * n + i] += 1e-12 * tr.max(1e-300);
    }
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().partial_cmp(&a[j * n + c].abs()).unwrap()).unwrap();
        if p != c {
            for k in 0..n {
                a.swap(c * n + k, p * n + k);
            }
            b.swap(c, p);
        }
        let d = a[c * n + c];
        for i in c + 1..n {
            let f = a[i * n + c] / d;
            for k in c..n {
                a[i * n + k] -= f * a[c * n + k];
            }
            b[i] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| a[i * n + k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i * n + i];
    }
    x
}

/// Robust static-tissue fit of the residual phase. `x[e][r]`: time-averaged image of encode e; `object`: voxels to
/// consider (the support); `deg`: polynomial degree; `min_signal`: fraction of the 95th percentile of |x_0| below
/// which voxels are ignored.
pub fn fit_static_phase(x: &[Vec<C>], object: &[bool], dims: [usize; 3], deg: usize, min_signal: f64) -> StaticFit {
    let ex = poly_exps(deg);
    let np = ex.len();
    let ne = x.len();
    let nv = object.len();
    let mut mags: Vec<f64> = (0..nv).filter(|&r| object[r]).map(|r| x[0][r].norm()).collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = mags.get(mags.len() * 95 / 100).copied().unwrap_or(1.0);
    let vox: Vec<usize> = (0..nv).filter(|&r| object[r] && x[0][r].norm() > min_signal * p95).collect();
    let pm: Vec<Vec<f64>> = vox.iter().map(|&r| ex.iter().map(|e| poly_at(e, r, dims)).collect()).collect();
    let mut a = vec![0.0; (ne - 1) * np];
    let mut sigma = vec![0.0; ne - 1];
    let mut keep = vec![true; vox.len()];
    for e in 1..ne {
        let d: Vec<f64> = vox.iter().map(|&r| (x[e][r] * x[0][r].conj()).arg()).collect();
        let sw: Vec<f64> = vox.iter().map(|&r| x[e][r].norm() * x[0][r].norm()).collect();
        let mut rw = vec![1.0; vox.len()];
        let mut coef = vec![0.0; np];
        for it in 0..25 {
            let (mut m, mut b) = (vec![0.0; np * np], vec![0.0; np]);
            for (i, p) in pm.iter().enumerate() {
                let w = sw[i] * rw[i];
                if w == 0.0 {
                    continue;
                }
                // unwrap the data against the current model so the fit is linear
                let fit: f64 = (0..np).map(|q| coef[q] * p[q]).sum();
                let di = fit + wrap(d[i] - fit);
                for u in 0..np {
                    b[u] += w * p[u] * di;
                    for v in 0..np {
                        m[u * np + v] += w * p[u] * p[v];
                    }
                }
            }
            coef = solve(m, b, np);
            // residuals, robust scale (MAD over signal-weighted voxels) and Tukey weights
            let res: Vec<f64> = pm.iter().zip(&d).map(|(p, &di)| wrap(di - (0..np).map(|q| coef[q] * p[q]).sum::<f64>())).collect();
            let mut ar: Vec<f64> = res.iter().map(|v| v.abs()).collect();
            ar.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let s = (1.4826 * ar[ar.len() / 2]).max(1e-6);
            let c = 4.685 * s;
            for (w, r) in rw.iter_mut().zip(&res) {
                let u = r / c;
                *w = if u.abs() < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
            }
            sigma[e - 1] = s;
            let _ = it;
        }
        for (i, k) in keep.iter_mut().enumerate() {
            *k &= rw[i] > 0.5;
        }
        a[(e - 1) * np..e * np].copy_from_slice(&coef);
    }
    let mut mask = vec![0f32; nv];
    for (i, &r) in vox.iter().enumerate() {
        if keep[i] {
            mask[r] = 1.0;
        }
    }
    StaticFit { a, mask, sigma }
}

/// images with the fitted residual phase removed: x_e · e^{−iφ_e}
pub fn remove_phase(x: &[Vec<C>], a: &[f64], dims: [usize; 3], deg: usize) -> Vec<Vec<C>> {
    let ex = poly_exps(deg);
    let np = ex.len();
    x.iter()
        .enumerate()
        .map(|(e, xe)| {
            if e == 0 {
                return xe.clone();
            }
            xe.iter().enumerate().map(|(r, &z)| z * C::from_polar(1.0, -(0..np).map(|p| a[(e - 1) * np + p] * poly_at(&ex[p], r, dims)).sum::<f64>())).collect()
        })
        .collect()
}

/// velocity per voxel by the hierarchical decoder (0 outside `object`)
pub fn decode(x: &[Vec<C>], object: &[bool], dec: &FlowDecoder) -> Vec<[f64; 3]> {
    (0..object.len()).map(|r| if object[r] { dec.velocity(&x.iter().map(|xe| xe[r]).collect::<Vec<_>>()) } else { [0.0; 3] }).collect()
}

/// (m, ψ) knot values of complex zero-moment values z_k (one voxel, K knots): m = |z|, ψ unwrapped along the knots as
/// arg z̄ + arg(z_k z̄*) (z̄ the mean phasor: continuous as long as the phase stays within ±π of its mean)
pub fn polar_of(z: &[C]) -> Vec<(f64, f64)> {
    let zb: C = z.iter().sum();
    let a0 = if zb.norm() > 0.0 { zb.arg() } else { 0.0 };
    z.iter().map(|c| (c.norm(), a0 + if zb.norm() > 0.0 { (c * zb.conj()).arg() } else { c.arg() })).collect()
}

/// cardiac B-spline coefficients ↔ values at the knot centres, per respiratory row, in place over K = kr · mc entries
/// (degree 3: values = (1/6, 2/3, 1/6) ⊛ coefficients; inverse by the periodic circulant inverse; degree 1: identity)
pub fn cardiac_values(x: &mut [f64], kr: usize, mc: usize, cdeg: u32, to_values: bool) {
    if cdeg != 3 {
        return;
    }
    let inv = crate::vknot::OpFrames::with_kre(2, mc, 3, mc, 3, 2).prefilter();
    for r in 0..kr {
        let row: Vec<f64> = x[r * mc..(r + 1) * mc].to_vec();
        for c in 0..mc {
            x[r * mc + c] = if to_values { (2.0 / 3.0) * row[c] + (row[(c + 1) % mc] + row[(c + mc - 1) % mc]) / 6.0 } else { (0..mc).map(|n| inv[n] * row[(c + mc - n) % mc]).sum() };
        }
    }
}

/// The spline-mode robust start of one voxel (research/velocity_model_sr.md §F, safeguards §F.2): from complex
/// parameter-basis knot coefficients per encode to the (m, ψ', v) cardiac spline coefficients of the velocity model
/// x_e = m e^{i(ψ' + φ_e − k'_e·v)}, k' = k − k̄.
pub struct SplineStart<'a> {
    pub dec: &'a FlowDecoder,
    /// encode sensitivities k_e = γ M1_e (rad per m/s), uncentred
    pub kenc: &'a [[f64; 3]],
    /// the centring k̄ (VelSpec::kbar)
    pub kbar: [f64; 3],
    pub kr: usize,
    pub mc: usize,
    pub cdeg: u32,
    /// decode a knot only at SNR ≥ this (else the voxel's time-mean decode, else 0); None = every knot
    pub snr_gate: Option<f64>,
    /// |v_a| ≤ this for the decoded values and the coefficients; None = no box
    pub vbox: Option<f64>,
}

impl SplineStart<'_> {
    /// `coef(e, k)`: the complex knot coefficient of encode e, knot k = r · mc + c; `phi[e]`: the residual phase to
    /// remove (0 for the reference). Returns (m, ψ') and v coefficients per knot and the counts [decoded per knot,
    /// time-mean fallback, zero] (support voxels; outside the support v = 0 and nothing is counted).
    pub fn voxel(&self, coef: &dyn Fn(usize, usize) -> C, phi: &[f64], in_support: bool) -> (Vec<(f32, f32)>, Vec<[f32; 3]>, [usize; 3]) {
        let (kr, mc, cdeg) = (self.kr, self.mc, self.cdeg);
        let nk = kr * mc;
        let ne = self.kenc.len();
        // values at the knot centres (cubic: (1/6, 2/3, 1/6) along the cardiac knots), residual phase removed
        let val = |e: usize, k: usize| -> C {
            let (r, c) = (k / mc, k % mc);
            let at = |cc: usize| coef(e, r * mc + cc);
            let v = if cdeg == 3 { at(c) * (2.0 / 3.0) + (at((c + 1) % mc) + at((c + mc - 1) % mc)) / 6.0 } else { at(c) };
            v * C::from_polar(1.0, -phi[e])
        };
        let zs: Vec<Vec<C>> = (0..nk).map(|k| (0..ne).map(|e| val(e, k)).collect()).collect();
        // SNR gate: the noise per real component from the dispersion of the encode magnitudes (every encode sees the
        // same m; var|z_e| ≈ σ² per component), the median over the knots over the χ²_{ne−1}/(ne−1) median
        // (Wilson–Hilferty); the time mean's noise taken as σ/√K
        let mean_mag = |z: &[C]| z.iter().map(|c| c.norm()).sum::<f64>() / ne as f64;
        let (sig, vbar, use_bar) = match self.snr_gate {
            Some(smin) if in_support && ne > 2 => {
                let mut var: Vec<f64> = zs.iter().map(|z| { let mu = mean_mag(z); z.iter().map(|c| (c.norm() - mu).powi(2)).sum::<f64>() / (ne - 1) as f64 }).collect();
                var.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let d = (ne - 1) as f64;
                let sig = (var[nk / 2] / (1.0 - 2.0 / (9.0 * d)).powi(3)).sqrt().max(1e-30);
                let zb: Vec<C> = (0..ne).map(|e| zs.iter().map(|z| z[e]).sum::<C>() / nk as f64).collect();
                let ok = mean_mag(&zb) * (nk as f64).sqrt() / sig >= smin;
                (sig, if ok { self.dec.velocity(&zb) } else { [0.0; 3] }, ok)
            }
            _ => (0.0, [0.0; 3], false),
        };
        let mut cnt = [0usize; 3];
        let mut z0 = vec![C::new(0.0, 0.0); nk];
        let mut vv = vec![[0.0f64; 3]; nk];
        let k0 = self.kenc[0];
        for k in 0..nk {
            let z = &zs[k];
            let v = if !in_support {
                [0.0; 3]
            } else if self.snr_gate.is_none_or(|smin| mean_mag(z) / sig >= smin) {
                cnt[0] += 1;
                self.dec.velocity(z)
            } else if use_bar {
                cnt[1] += 1;
                vbar
            } else {
                cnt[2] += 1;
                [0.0; 3]
            };
            let v = self.vbox.map_or(v, |b| v.map(|x| x.clamp(-b, b)));
            // zero-moment value: the reference encode with its velocity phase −k_0·v undone
            z0[k] = z[0] * C::from_polar(1.0, k0[0] * v[0] + k0[1] * v[1] + k0[2] * v[2]);
            vv[k] = v;
        }
        let (mut m, mut ps): (Vec<f64>, Vec<f64>) = polar_of(&z0).into_iter().unzip();
        cardiac_values(&mut m, kr, mc, cdeg, false);
        cardiac_values(&mut ps, kr, mc, cdeg, false);
        let mut vc: Vec<[f32; 3]> = vec![[0.0; 3]; nk];
        for a in 0..3 {
            let mut x: Vec<f64> = vv.iter().map(|v| v[a]).collect();
            cardiac_values(&mut x, kr, mc, cdeg, false);
            for k in 0..nk {
                // coefficients inside the box too (the circulant inverse can overshoot the values)
                vc[k][a] = self.vbox.map_or(x[k], |b| x[k].clamp(-b, b)) as f32;
            }
        }
        let kb = self.kbar;
        let zp: Vec<(f32, f32)> = (0..nk).map(|k| (m[k] as f32, (ps[k] - (0..3).map(|a| kb[a] * vc[k][a] as f64).sum::<f64>()) as f32)).collect();
        (zp, vc, cnt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the spline-mode start (§F, §F.2) on model-consistent knot values of the coprime-7 dual-VENC design (VENC 1.0 /
    /// 1.5 m/s, unaliased ±3 m/s): the decoder's sign / VENC convention, the zero-moment phase, the centring and the
    /// cardiac circulant inverse give back the model's (m, ψ' = ψ0 − k̄·v, v) coefficients, and x_e = m e^{i(ψ' + φ_e −
    /// k'_e·v)} reproduces the encode values; the SNR gate falls back to the time mean at low SNR
    #[test]
    fn spline_start_recovers_model_coefficients() {
        let (lo, hi, r) = (5.871648784821618e-09, 1.957216261607207e-09, -5.871648784821618e-09);
        let mut m1: Vec<[f64; 3]> = vec![[r; 3]];
        for a in 0..3 {
            for x in [lo, hi] {
                let mut e = [r; 3];
                e[a] = x;
                m1.push(e);
            }
        }
        let dec = FlowDecoder::new(&m1).with_alias_search(crate::flow::AliasSearch { vext: 3.0, margin: 1.0 });
        let kenc: Vec<[f64; 3]> = m1.iter().map(|m| m.map(|x| hasty_mrsim::GAMMA * x)).collect();
        let ne = kenc.len();
        let kbar = [0, 1, 2].map(|a| kenc.iter().map(|k| k[a]).sum::<f64>() / ne as f64);
        let (kr, mc, cdeg) = (2usize, 16usize, 3u32);
        let nk = kr * mc;
        let tw = |c: usize| 2.0 * std::f64::consts::PI * c as f64 / mc as f64;
        // truth coefficients (smooth, |v| up to 2.2 m/s: beyond the 1.5 m/s first stage, inside the design range)
        let mt: Vec<f64> = (0..nk).map(|k| 2.0 + 0.4 * tw(k % mc).cos() + 0.1 * (k / mc) as f64).collect();
        let pt: Vec<f64> = (0..nk).map(|k| 0.6 + 0.5 * tw(k % mc).sin()).collect();
        let vt: Vec<[f64; 3]> = (0..nk).map(|k| [0, 1, 2].map(|a| [1.6, -2.2, 0.9][a] * (tw(k % mc) + a as f64).sin() + 0.1 * (k / mc) as f64)).collect();
        let vals = |x: &[f64]| { let mut y = x.to_vec(); cardiac_values(&mut y, kr, mc, cdeg, true); y };
        let (mv, pv) = (vals(&mt), vals(&pt));
        let vv: Vec<Vec<f64>> = (0..3).map(|a| vals(&vt.iter().map(|v| v[a]).collect::<Vec<_>>())).collect();
        let phi = [0.0, 0.3, -0.2, 0.15, 0.05, -0.4, 0.25];
        // complex encode values at the knot centres (residual phase included), then complex coefficients
        let coefs = |noise: &dyn Fn(usize, usize) -> C| -> Vec<Vec<C>> {
            (0..ne).map(|e| {
                let z: Vec<C> = (0..nk).map(|k| C::from_polar(mv[k], pv[k] + phi[e] - (0..3).map(|a| kenc[e][a] * vv[a][k]).sum::<f64>()) + noise(e, k)).collect();
                let (mut re, mut im): (Vec<f64>, Vec<f64>) = z.iter().map(|c| (c.re, c.im)).unzip();
                cardiac_values(&mut re, kr, mc, cdeg, false);
                cardiac_values(&mut im, kr, mc, cdeg, false);
                re.into_iter().zip(im).map(|(a, b)| C::new(a, b)).collect()
            }).collect()
        };
        let ss = SplineStart { dec: &dec, kenc: &kenc, kbar, kr, mc, cdeg, snr_gate: Some(3.28), vbox: Some(3.0) };
        let cf = coefs(&|_, _| C::new(0.0, 0.0));
        let (zp, vc, cnt) = ss.voxel(&|e, k| cf[e][k], &phi, true);
        assert_eq!(cnt, [nk, 0, 0], "noise-free: every knot decoded");
        for k in 0..nk {
            for a in 0..3 {
                assert!((vc[k][a] as f64 - vt[k][a]).abs() < 1e-4, "v coefficient {k} {a}: {} vs {}", vc[k][a], vt[k][a]);
            }
            assert!((zp[k].0 as f64 - mt[k]).abs() < 1e-4, "m coefficient {k}");
            let psi_c = pt[k] - (0..3).map(|a| kbar[a] * vt[k][a]).sum::<f64>();
            assert!((zp[k].1 as f64 - psi_c).abs() < 1e-4, "ψ' coefficient {k}: {} vs {psi_c}", zp[k].1);
        }
        // the model's forward form at the knot centres (VelSpec::kenc_model = k − k̄) gives the encode values back
        let m_s = vals(&zp.iter().map(|p| p.0 as f64).collect::<Vec<_>>());
        let p_s = vals(&zp.iter().map(|p| p.1 as f64).collect::<Vec<_>>());
        let v_s: Vec<Vec<f64>> = (0..3).map(|a| vals(&vc.iter().map(|v| v[a] as f64).collect::<Vec<_>>())).collect();
        for e in 0..ne {
            for k in 0..nk {
                let ph = p_s[k] + phi[e] - (0..3).map(|a| (kenc[e][a] - kbar[a]) * v_s[a][k]).sum::<f64>();
                let want = C::from_polar(mv[k], pv[k] + phi[e] - (0..3).map(|a| kenc[e][a] * vv[a][k]).sum::<f64>());
                assert!((C::from_polar(m_s[k], ph) - want).norm() < 1e-4, "encode {e} knot {k}");
            }
        }
        // low SNR (σ = m per component): no knot passes; the time mean (σ/√K) does
        let mut sgen = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || { sgen ^= sgen << 13; sgen ^= sgen >> 7; sgen ^= sgen << 17; (sgen >> 11) as f64 / (1u64 << 53) as f64 };
        let nz: Vec<C> = (0..ne * nk).map(|_| { let (u, w) = (rnd().max(1e-12), rnd()); let r = 2.0 * (-2.0 * u.ln()).sqrt(); C::new(r * (6.283185307 * w).cos(), r * (6.283185307 * w).sin()) }).collect();
        let cf = coefs(&|e, k| nz[e * nk + k]);
        let (_, vc, cnt) = ss.voxel(&|e, k| cf[e][k], &phi, true);
        assert!(cnt[0] <= nk / 8 && cnt[0] + cnt[1] + cnt[2] == nk, "low SNR gated: {cnt:?}");
        assert!(vc.iter().all(|v| v.iter().all(|x| x.abs() <= 3.0)), "inside the box");
        // outside the support: v = 0, nothing counted
        let (_, vc, cnt) = ss.voxel(&|e, k| cf[e][k], &phi, false);
        assert_eq!(cnt, [0, 0, 0]);
        assert!(vc.iter().all(|v| *v == [0.0; 3]));
    }

    #[test]
    fn robust_fit_recovers_phase_and_rejects_the_vessel() {
        // 40×30×12 object; static tissue with a degree-1 residual phase per encode; a "vessel" tube with flow
        // (phase differences far from the polynomial) and noise; the fit must recover the coefficients and exclude
        // the vessel from the static mask
        let dims = [40usize, 30, 12];
        let nv = dims.iter().product();
        let (lo, hi) = (std::f64::consts::PI / 1.0, std::f64::consts::PI / 3.0);
        let mut kenc = vec![[0.0; 3]];
        for ax in 0..3 {
            for k in [lo, hi] {
                let mut v = [0.0; 3];
                v[ax] = k;
                kenc.push(v);
            }
        }
        let ex = poly_exps(1);
        let a_true: Vec<f64> = (0..6 * ex.len()).map(|i| 0.5 * (((i * 7919) % 13) as f64 / 6.0 - 1.0)).collect();
        let mut s = 0x1234u64;
        let mut g = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let vessel = |r: usize| {
            let (i, j) = ((r % 40) as f64, ((r / 40) % 30) as f64);
            (i - 20.0).hypot(j - 15.0) < 4.0
        };
        let mut x = vec![vec![C::new(0.0, 0.0); nv]; 7];
        for r in 0..nv {
            let m = 1.0 + 0.3 * g();
            let v = if vessel(r) { [0.2, -0.7, 0.9] } else { [0.0; 3] };
            for e in 0..7 {
                let phi: f64 = if e == 0 { 0.0 } else { (0..ex.len()).map(|p| a_true[(e - 1) * ex.len() + p] * poly_at(&ex[p], r, dims)).sum() };
                let ph = phi - (kenc[e][0] * v[0] + kenc[e][1] * v[1] + kenc[e][2] * v[2]) + 0.4;
                x[e][r] = C::from_polar(m, ph) + C::new(0.03 * g(), 0.03 * g());
            }
        }
        let object = vec![true; nv];
        let fit = fit_static_phase(&x, &object, dims, 1, 0.05);
        let err = fit.a.iter().zip(&a_true).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / (a_true.len() as f64).sqrt();
        assert!(err < 5e-3, "coefficient RMS error {err:e}");
        let in_vessel = (0..nv).filter(|&r| vessel(r) && fit.mask[r] > 0.0).count();
        let static_kept = (0..nv).filter(|&r| !vessel(r) && fit.mask[r] > 0.0).count() as f64 / (0..nv).filter(|&r| !vessel(r)).count() as f64;
        assert!(in_vessel == 0, "{in_vessel} vessel voxels in the static mask");
        assert!(static_kept > 0.9, "only {static_kept:.2} of the static tissue kept");
        // decoded velocity after phase removal: the vessel speed, static ≈ 0
        let dec = FlowDecoder::new(&kenc.iter().map(|k| k.map(|v| v / hasty_mrsim::GAMMA)).collect::<Vec<_>>());
        let v = decode(&remove_phase(&x, &fit.a, dims, 1), &object, &dec);
        let r0 = 15 * 40 + 20; // vessel centre, slice 0
        assert!((v[r0][1] + 0.7).abs() < 0.05 && (v[r0][2] - 0.9).abs() < 0.05, "{:?}", v[r0]);
        assert!(v[5][0].abs() < 0.02, "static voxel velocity {:?}", v[5]);
    }
}
