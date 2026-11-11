//! WP1 step 2 (research/plan_girf_b0_qfit.md): how much the gradient chain (GIRF) breaks what the reconstructions
//! assume, for the final designs, over the illustrative GIRF variants (`hasty_seq::girf::Girf::variants`, each adding
//! one effect class). CPU only, seconds.
//!
//!     girfcheck [--flow data/hasty/seq/flow_thorax] [--out data/hasty/seq/studies/girfcheck] [--n-rot 200]
//!
//! 4D-flow TR (`seqdesign` arrays):
//! - rotation structure: the played readouts R_j g are compared with R_j k̄, k̄ = the mean of R_jᵀ k_played over the
//!   rotations (the best single rotated base): max |error| in Nyquist cells (× FOV) over the ADC window. An isotropic
//!   chain gives exactly 0, so this isolates axis differences and cross terms;
//! - concomitant structure: the same for P = ∫GGᵀ, as the phase at 15 cm (rad) at the readout end;
//! - per encode: the k error and B0-eddy phase the encode block leaves over the readout, averaged with the body-scale
//!   least-squares weights of a linear-phase fit (`robust::ball_weights`, as `encrobust`, which also has the vessel
//!   scale and the model-error residuals); the encode-to-encode differences → background velocity at 10 cm,
//!   (|Δφ̄| + 2π·0.1·|Δc|)/(γ|ΔM1|). (Earlier versions used the k offset at TE itself, which overstates the short
//!   low-pass / delay transients that are gone ≈ 30 µs into the readout.)
//! - full played sequence: every encode's TR played after its real predecessor (acquisition order: the encodes of a
//!   rotation back to back), so the previous TR's flow-compensated tail and readout leak in; encode-to-encode
//!   background velocity of the part that is the same every rotation (coherent: a background phase) and the rms of
//!   what changes from TR to TR (encode 0 follows the previous rotation's readout), body and vessel scale, 16 rotations.
//! B0 train (2 mm, 24 lobes, as `b0seq`):
//! - per echo: B0-eddy phase and |Δk| at the k = 0 crossings; the phase slope over TE is a global ω bias (Hz) the
//!   B0 map would absorb, the residual after the slope is model misfit;
//! - rotation structure of the lobes as above (50 TRs).
//! Writes <out>/girfcheck.json.

use std::path::PathBuf;

use hasty_recon::data::read_npy;
use hasty_seq::girf::{played_moments, trajectory_error, Girf};
use hasty_seq::robust;
use hasty_seq::multiecho::{golden_dirs, EchoSpec, EchoTrain};
use hasty_seq::system::System;
use hasty_seq::traj::{acquisition_order, rotations, RotSet};
use hasty_seq::wave::Grad;
use hasty_seq::{mat_vec, transpose, Mat3, Vec3, GAMMA, GAMMA_BAR};
use serde_json::json;

fn grad_from(v: &[f64], n: usize, dt: f64) -> Grad {
    Grad { dt, t0: 0.0, g: (0..n).map(|i| [v[3 * i], v[3 * i + 1], v[3 * i + 2]]).collect() }
}

/// played k (cycles/m) and P (xx yy zz xy xz yz) of `g` through `girf`, on g's raster (index 0 = g's start)
fn played(girf: &Girf, g: &Grad) -> (Vec<Vec3>, Vec<[f64; 6]>, Vec<f64>) {
    let (a, b0) = girf.apply(g);
    let m = a.moments(a.t0);
    let off = Girf::PRE;
    let k: Vec<Vec3> = m.k()[off..off + g.len()].to_vec();
    let p: Vec<[f64; 6]> = m.p[off..off + g.len()].to_vec();
    // B0-eddy phase (rad), accumulated from the start
    let mut ph = vec![0.0; b0.len()];
    for i in 1..b0.len() {
        ph[i] = ph[i - 1] + GAMMA * 0.5 * (b0[i - 1] + b0[i]) * g.dt;
    }
    (k, p, ph[off..off + g.len()].to_vec())
}

fn rot_p(r: &Mat3, p: &[f64; 6]) -> [[f64; 3]; 3] {
    let m = [[p[0], p[3], p[4]], [p[3], p[1], p[5]], [p[4], p[5], p[2]]];
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for a in 0..3 {
                for b in 0..3 {
                    o[i][j] += r[i][a] * m[a][b] * r[j][b];
                }
            }
        }
    }
    o
}

/// rotation-structure errors of base waveform `gb` (base frame) over `rots`: (max k error in cycles/m, max
/// concomitant-structure phase at radius `x` in rad)
fn rotation_structure(girf: &Girf, gb: &Grad, rots: &[Mat3], b0: f64, x: f64) -> (f64, f64) {
    let n = gb.len();
    let runs: Vec<(Vec<Vec3>, Vec<[f64; 6]>)> = hasty_seq::par_map(rots, |r| {
        let (k, p, _) = played(girf, &gb.rotated(r));
        (k, p)
    });
    // mean base-frame k and P
    let mut kb = vec![[0.0; 3]; n];
    let mut pb = vec![[[0.0; 3]; 3]; n];
    for (r, (k, p)) in rots.iter().zip(&runs) {
        let rt = transpose(r);
        for i in 0..n {
            let kk = mat_vec(&rt, &k[i]);
            for a in 0..3 {
                kb[i][a] += kk[a] / rots.len() as f64;
            }
            let pp = rot_p(&rt, &p[i]);
            for a in 0..3 {
                for b in 0..3 {
                    pb[i][a][b] += pp[a][b] / rots.len() as f64;
                }
            }
        }
    }
    let (mut ek, mut ep) = (0.0f64, 0.0f64);
    let wc = GAMMA / (2.0 * b0);
    for (r, (k, p)) in rots.iter().zip(&runs) {
        for i in 0..n {
            let kr = mat_vec(r, &kb[i]);
            ek = ek.max(hasty_seq::norm(&hasty_seq::sub(&k[i], &kr)));
        }
        // concomitant: P_played − R P̄ Rᵀ at the end, its largest eigen-ish magnitude × wc × x²
        let last = n - 1;
        let pr = rot_p(r, &[pb[last][0][0], pb[last][1][1], pb[last][2][2], pb[last][0][1], pb[last][0][2], pb[last][1][2]]);
        let pa = &p[last];
        let d = [pa[0] - pr[0][0], pa[1] - pr[1][1], pa[2] - pr[2][2], pa[3] - pr[0][1], pa[4] - pr[0][2], pa[5] - pr[1][2]];
        let mag = d.iter().map(|v| v.abs()).fold(0.0, f64::max);
        ep = ep.max(wc * mag * x * x);
    }
    (ek, ep)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |n: &str| args.iter().position(|a| a == n).map(|i| args[i + 1].clone());
    let flow = opt("--flow").map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("seq/flow_thorax"));
    let out = opt("--out").map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("seq/studies/girfcheck"));
    let n_rot: usize = opt("--n-rot").map_or(200, |s| s.parse().unwrap());
    std::fs::create_dir_all(&out).unwrap();
    let sys = System::prisma_xr();
    let fin: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(flow.join("final.json")).unwrap()).unwrap();
    let summ: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(flow.join("summary.json")).unwrap()).unwrap();
    let dt = fin["dt"].as_f64().unwrap();
    let (t_rf, te) = (fin["t_rf"].as_f64().unwrap(), fin["te"][0].as_f64().unwrap());
    let n_enc = fin["n_enc"].as_u64().unwrap() as usize;
    let i_adc = summ["best"]["i_adc"].as_u64().unwrap() as usize;
    let fov = summ["problem"]["fov"].as_f64().unwrap();
    let venc_lo = summ["encoding"]["venc_lo"].as_f64().unwrap();
    let spg = read_npy(&flow.join("spoke_grad.npy")).unwrap().f64s();
    let base = grad_from(&spg, i_adc + 1, dt);
    let trg = read_npy(&flow.join("tr_grad.npy")).unwrap().f64s();
    let ntr = trg.len() / (3 * n_enc);
    let rots_all = rotations(RotSet::SuperFibonacci, 2208);
    let rots: Vec<Mat3> = (0..n_rot).map(|j| rots_all[j * rots_all.len() / n_rot]).collect();
    let i_te = ((t_rf + te) / dt).round() as usize;

    // B0 train
    let mut spec = EchoSpec::new(2e-3, 24);
    let t0_ = hasty_seq::multiecho::fit_slew_to_pns(&spec, &sys, &hasty_seq::pns::SafeHw::example(), 80.0, 6);
    spec.slew_frac = t0_.spec.slew_frac;
    spec.te_shifts = vec![0.0, sys.ceil_raster(0.5 * t0_.t_lobe)];
    let train = EchoTrain::design(&spec, &sys);
    let dpt = train.dirs_per_tr();
    let t_rf_b = sys.rf_dead + 0.5 * spec.rf_dur;

    // readout weights (as `encrobust`): body-scale least-squares weight of a linear-phase fit, 1 µs grid
    let i_ro = i_te;
    let i_enc = summ["best"]["i_enc"].as_u64().unwrap() as usize;
    let dtf = 1e-6;
    let times: Vec<f64> = (0..(base.duration() / dtf) as usize).map(|i| i as f64 * dtf).collect();
    let kabs: Vec<f64> = base.moments_at(0.0, &times).iter().map(|m| GAMMA_BAR * hasty_seq::norm(&m.0)).collect();
    let w_body = robust::ball_weights(&kabs, 0.12);
    let w_vessel = robust::ball_weights(&kabs, 0.002);
    // full played sequence: TR (e, R) = the exported encode / flow-compensated tail (physical) with the readout R·base;
    // played after its real predecessor in acquisition order (encodes back to back, then the next rotation)
    assert_eq!(summ["best"]["spoil_mode"].as_str(), Some("flowcomp"), "the full-sequence check assumes the physical per-encode tail");
    let n_tr = summ["best"]["n_tr"].as_u64().unwrap() as usize;
    let ord = acquisition_order(RotSet::SuperFibonacci, rots_all.len());
    let racq: Vec<Mat3> = ord[..17].iter().map(|&i| rots_all[i]).collect();
    // --same-tail: every TR gets encode 0's tail (what an encode-independent tail, i.e. no per-encode M1 rewind, leaks)
    let same_tail = args.iter().any(|a| a == "--same-tail");
    let tr_of = |e: usize, r: &Mat3| -> Vec<Vec3> {
        let mut g: Vec<Vec3> = (0..ntr).map(|i| [trg[(e * ntr + i) * 3], trg[(e * ntr + i) * 3 + 1], trg[(e * ntr + i) * 3 + 2]]).collect();
        if same_tail {
            for i in i_te + i_adc..ntr {
                g[i] = [trg[i * 3], trg[i * 3 + 1], trg[i * 3 + 2]];
            }
        }
        for i in 0..=i_adc {
            g[i_te + i] = mat_vec(r, &base.g[i]);
        }
        g
    };
    let abs_t: Vec<f64> = times.iter().map(|t| i_te as f64 * dt + t).collect();
    // readout resonance exposure (`robust::band_exposure`, the seqdesign objective term) for the generic band and the
    // variant chain's resonances as if measured, Nyquist cells
    let exp_band = robust::band_exposure(&base, &robust::rotation_band(&robust::RobustOpts::default())) * fov;
    let known = Girf::variants().into_iter().find(|v| v.0 == "resonance").unwrap().1;
    let exp_known = robust::band_exposure(&base, &robust::rotation_band(&robust::RobustOpts { band: None, ..Default::default() }.with_resonances(&known, 300.0, 3000.0, 10.0))) * fov;
    println!("readout resonance exposure: generic band {exp_band:.3}, variant resonances {exp_known:.3} Nyquist cells");
    let mut rows = vec![];
    println!("{:<10} | flow: rot k [cells] conc@15cm [rad] | enc: weighted Δφ [mrad] Δk [1/m] v_bg [mm/s] | B0 train: rot k [cells] |Δk|@TE [cells] ω bias [Hz] φ resid [mrad]", "GIRF");
    for (name, girf) in Girf::variants() {
        // flow: rotation structure of the base readout
        let (ek, ep) = rotation_structure(&girf, &base, &rots, sys.b0, 0.15);
        // flow: per encode, the k error / B0 phase the encode block leaves over the readout, least-squares weighted
        // (body scale, `robust::ball_weights` 12 cm) → background velocity at 10 cm (as `encrobust`)
        let blocks: Vec<Grad> = (0..n_enc)
            .map(|e| {
                let g = grad_from(&trg[e * ntr * 3..(e + 1) * ntr * 3], ntr, dt);
                Grad { dt, t0: i_enc as f64 * dt, g: g.g[i_enc..=i_ro].to_vec() }
            })
            .collect();
        let offs: Vec<(Vec3, f64)> = blocks
            .iter()
            .map(|b| {
                let (dk, ph) = robust::played_offsets(&girf, None, b, &times);
                robust::weighted_offset(&dk, &ph, &w_body)
            })
            .collect();
        let m1s: Vec<Vec3> = blocks.iter().map(|b| b.moments(t_rf).last().1).collect();
        let mut v_bg: f64 = 0.0;
        let (mut dphi_max, mut dk_max) = (0.0f64, 0.0f64);
        for e in 1..n_enc {
            let dphi = (offs[e].1 - offs[0].1).abs();
            let dk = hasty_seq::norm(&hasty_seq::sub(&offs[e].0, &offs[0].0));
            dphi_max = dphi_max.max(dphi);
            dk_max = dk_max.max(dk);
            v_bg = v_bg.max((dphi + 2.0 * std::f64::consts::PI * dk * 0.10) / (GAMMA * hasty_seq::norm(&hasty_seq::sub(&m1s[e], &m1s[0]))));
        }
        let _ = (venc_lo, te);
        // B0 train: echo phases / k at the crossings over a few TRs
        let mut om_bias = vec![];
        let mut resid = 0.0f64;
        let mut dk_te = 0.0f64;
        for tr_i in 0..8 {
            let s = spec.te_shifts[tr_i % 2];
            let w = train.tr_waveform(&golden_dirs(tr_i * dpt, dpt), s, &sys);
            let (k, _, ph) = played(&girf, &w);
            let kn = w.moments(0.0).k();
            let tes = train.te(s);
            let idx: Vec<usize> = tes.iter().map(|t| ((t_rf_b + t) / dt).round() as usize).collect();
            let p: Vec<f64> = idx.iter().map(|&i| ph[i]).collect();
            for &i in &idx {
                dk_te = dk_te.max(hasty_seq::norm(&hasty_seq::sub(&k[i], &kn[i])));
            }
            // least-squares slope of phase over TE
            let n = tes.len() as f64;
            let (mt, mp) = (tes.iter().sum::<f64>() / n, p.iter().sum::<f64>() / n);
            let sl = tes.iter().zip(&p).map(|(t, q)| (t - mt) * (q - mp)).sum::<f64>() / tes.iter().map(|t| (t - mt).powi(2)).sum::<f64>();
            om_bias.push(sl / (2.0 * std::f64::consts::PI));
            for (t, q) in tes.iter().zip(&p) {
                resid = resid.max((q - mp - sl * (t - mt)).abs());
            }
        }
        // B0 train rotation structure of one lobe pair waveform under 50 directions
        let lobe2 = train.readout(&vec![[0.0, 0.0, 1.0]; dpt], 0.0);
        let lobe2 = Grad { g: lobe2.g[..2 * (train.lobe.len() - 1) + 1].to_vec(), ..lobe2 };
        let rots_b: Vec<Mat3> = golden_dirs(0, 50)
            .iter()
            .map(|d| {
                // rotation taking z to d
                let z = [0.0, 0.0, 1.0];
                let ax = hasty_seq::cross(&z, d);
                let s = hasty_seq::norm(&ax);
                if s < 1e-12 { hasty_seq::IDENTITY } else { hasty_seq::axis_angle(&ax, s.atan2(d[2])) }
            })
            .collect();
        let (ekb, _) = rotation_structure(&girf, &lobe2, &rots_b, sys.b0, 0.15);
        let omb = om_bias.iter().map(|x| x.abs()).fold(0.0, f64::max);
        // full played sequence: per rotation and encode the weighted offsets [body, vessel] of the k error / phase
        // left by `model` (Girf::ideal() = unmodelled)
        let offs_for = |model: &Girf, nr: usize| -> Vec<Vec<[(Vec3, f64); 2]>> {
            hasty_seq::par_map(&(1..nr.min(racq.len())).collect::<Vec<_>>(), |&j| {
                (0..n_enc)
                    .map(|e| {
                        let this = Grad { dt, t0: 0.0, g: tr_of(e, &racq[j]) };
                        let prev = if e == 0 { tr_of(n_enc - 1, &racq[j - 1]) } else { tr_of(e - 1, &racq[j]) };
                        let prev = &prev[..n_tr.min(prev.len())];
                        let pl = played_moments(&girf, prev, &this, t_rf, &abs_t);
                        let nom = played_moments(model, prev, &this, t_rf, &abs_t);
                        let dk: Vec<Vec3> = pl.iter().zip(&nom).map(|(a, b)| [0, 1, 2].map(|q| GAMMA_BAR * (a.m0[q] - b.m0[q]))).collect();
                        let ph: Vec<f64> = pl.iter().zip(&nom).map(|(a, b)| a.phi_b0 - b.phi_b0).collect();
                        [robust::weighted_offset(&dk, &ph, &w_body), robust::weighted_offset(&dk, &ph, &w_vessel)]
                    })
                    .collect()
            })
        };
        // coherent part (the same every rotation: a background phase) and what fluctuates from TR to TR (the previous
        // rotation's readout leaking into encode 0: a trajectory error that blurs rather than biases)
        let vel = |d_phi: f64, d_k: &Vec3, e: usize| (d_phi.abs() + 2.0 * std::f64::consts::PI * 0.10 * hasty_seq::norm(d_k)) / (GAMMA * hasty_seq::norm(&hasty_seq::sub(&m1s[e], &m1s[0])));
        let split = |offs_rot: &Vec<Vec<[(Vec3, f64); 2]>>| -> ([f64; 2], [f64; 2]) {
            let nrot = offs_rot.len() as f64;
            let mut coh = [0.0f64; 2];
            let mut fluct = [0.0f64; 2];
            for sc in 0..2 {
                let mean_of = |e: usize| -> (Vec3, f64) {
                    let mut c = [0.0; 3];
                    let mut p = 0.0;
                    for o in offs_rot {
                        for q in 0..3 {
                            c[q] += o[e][sc].0[q] / nrot;
                        }
                        p += o[e][sc].1 / nrot;
                    }
                    (c, p)
                };
                let (c0, p0) = mean_of(0);
                for e in 1..n_enc {
                    let (ce, pe) = mean_of(e);
                    coh[sc] = coh[sc].max(vel(pe - p0, &hasty_seq::sub(&ce, &c0), e));
                    let ms = offs_rot
                        .iter()
                        .map(|o| {
                            let dk = hasty_seq::sub(&hasty_seq::sub(&o[e][sc].0, &o[0][sc].0), &hasty_seq::sub(&ce, &c0));
                            let dp = (o[e][sc].1 - o[0][sc].1) - (pe - p0);
                            vel(dp, &dk, e).powi(2)
                        })
                        .sum::<f64>()
                        / nrot;
                    fluct[sc] = fluct[sc].max(ms.sqrt());
                }
            }
            (coh, fluct)
        };
        let offs_rot = offs_for(&Girf::ideal(), racq.len());
        let (coh, fluct) = split(&offs_rot);
        // after modelling the played sequence with single-class 10 % model errors: RSS of the coherent remainder
        let perts = girf.perturbations(0.1);
        let both: Vec<([f64; 2], [f64; 2])> = perts.iter().map(|(_, m)| split(&offs_for(m, 9))).collect();
        let modelled: Vec<[f64; 2]> = both.iter().map(|b| b.0).collect();
        let fl_mod: Vec<[f64; 2]> = both.iter().map(|b| b.1).collect();
        let by_class: Vec<serde_json::Value> = perts
            .iter()
            .zip(&both)
            .map(|((n, _), (c, f))| json!({"class": n, "coherent_body": c[0], "coherent_vessel": c[1], "fluct_body": f[0], "fluct_vessel": f[1]}))
            .collect();
        if name == "resonance" || name == "eddy" {
            let mut v: Vec<(&str, f64)> = perts.iter().zip(&both).map(|((n, _), (c, f))| (*n, c[0].max(c[1]).max(f[0]).max(f[1]))).collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("{:<10}   largest model-error classes (10 %, f0 1 %): {}", "", v.iter().take(4).map(|(n, x)| format!("{n} {:.1}", x * 1e3)).collect::<Vec<_>>().join(", "));
        }
        let coh_mod = [0, 1].map(|sc| modelled.iter().map(|v| v[sc] * v[sc]).sum::<f64>().sqrt());
        let fluct_mod = [0, 1].map(|sc| fl_mod.iter().map(|v| v[sc] * v[sc]).sum::<f64>().sqrt());
        println!(
            "{name:<10} | {:>8.3} {:>12.4} | {:>14.3} {:>10.3} {:>9.3} | {:>8.3} {:>10.3} {:>9.3} {:>9.3}",
            ek * fov,
            ep,
            dphi_max * 1e3,
            dk_max,
            v_bg * 1e3,
            ekb * 0.40,
            dk_te * 0.40,
            omb,
            resid * 1e3
        );
        println!(
            "{:<10}   full played sequence ({} rotations): coherent v_bg body {:.1}, vessel {:.1} | modelled {:.1}, {:.1} mm/s; TR-to-TR rms body {:.1}, vessel {:.1} | modelled {:.1}, {:.1}",
            "", offs_rot.len(), coh[0] * 1e3, coh[1] * 1e3, coh_mod[0] * 1e3, coh_mod[1] * 1e3, fluct[0] * 1e3, fluct[1] * 1e3, fluct_mod[0] * 1e3, fluct_mod[1] * 1e3
        );
        rows.push(json!({"girf": name, "flow": {"rot_k_cells": ek * fov, "conc_15cm_rad": ep, "enc_dphi_weighted_rad": dphi_max, "enc_dk_weighted": dk_max, "v_background_m_s": v_bg,
                                                 "full_sequence": {"v_bg_body": coh[0], "v_bg_vessel": coh[1], "fluct_rms_body": fluct[0], "fluct_rms_vessel": fluct[1], "rotations": offs_rot.len(),
                                                                   "modelled": {"v_bg_body": coh_mod[0], "v_bg_vessel": coh_mod[1], "fluct_rms_body": fluct_mod[0], "fluct_rms_vessel": fluct_mod[1], "eps": 0.1, "by_class": by_class}}},
                         "b0_train": {"rot_k_cells": ekb * 0.40, "dk_te_cells": dk_te * 0.40, "omega_bias_hz": omb, "phase_resid_rad": resid}}));
    }
    let _ = (GAMMA_BAR, trajectory_error as fn(&Grad, &Grad, &[f64]) -> (Vec<Vec3>, Vec<f64>));
    std::fs::write(out.join("girfcheck.json"), serde_json::to_string_pretty(&json!({"flow": flow, "n_rot": n_rot, "fov_flow": fov, "fov_b0": 0.40, "rows": rows,
        "readout_exposure_cells": {"band": exp_band, "variant_resonances": exp_known}})).unwrap()).unwrap();
    println!("→ {}", out.join("girfcheck.json").display());
}
