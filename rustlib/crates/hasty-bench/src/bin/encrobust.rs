//! WP1 step 3 (research/plan_girf_b0_qfit.md): GIRF-robust velocity encodes for the final 4D-flow TR. CPU only.
//!
//!     encrobust [--flow data/hasty/seq/flow_thorax] [--out data/hasty/seq/studies/encrobust]
//!               [--extra 0,20,50,80] [--robust-slew 1.0,1.25] [--eps 0.1] [--r-off 0.10]
//!
//! Compares the minimum-time two-lobe encodes of the final design (`seqdesign`) with the same block plus a short gap
//! and with GIRF-robust blocks (`hasty_seq::robust`, QP) of the same and longer lengths, over the illustrative
//! chains `Girf::variants()`.
//!
//! Metric (what phase contrast sees): the chain is linear, so encode e's played readout differs from encode 0's
//! exactly by the played (g_e − g_0); its k error Δk_e(t) and B0-term phase φ_e(t) over the ADC window are averaged
//! with least-squares weights of a linear-phase fit (`robust::ball_weights`: a 12 cm ball = body-scale background,
//! a 2 mm ball = vessel scale). Background velocity at r_off from the isocentre:
//!   v_bg,e = (|Δφ̄_e| + 2π r_off |Δc_e|) / (γ |ΔM1_e|),
//! max over encodes. "modelled": what is left after the reconstruction models the played gradients with a chain that
//! is off by a relative `eps` in one parameter class (`Girf::perturbations`: delays, low-pass, eddy amplitudes / time
//! constants, cross, B0, resonance f0 (eps/10) / damping / height), root-sum-square over the classes. PNS: SAFE (example hardware) of the 7 TRs of rotation 0 with the
//! encode replaced (readout and tail unchanged).
//! Writes <out>/encrobust.json and arrays for rustlib/tools/seq/plot_encrobust.py.

use std::path::PathBuf;

use hasty_recon::data::read_npy;
use hasty_seq::encode::{design as two_lobe_design, DualLayout, EncodeSpec, Scheme};
use hasty_seq::girf::Girf;
use hasty_seq::pns::{pns, SafeHw};
use hasty_seq::robust::{ball_weights, objective, played_offsets, sensitivity, spectrum, weighted_offset, RobustOpts};
use hasty_seq::system::System;
use hasty_seq::wave::Grad;
use hasty_seq::{norm, sub, Vec3, GAMMA};
use serde_json::json;

fn write_f8(p: &std::path::Path, shape: &[usize], v: &[f64]) {
    let hdr = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({}{}), }}", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "), if shape.len() == 1 { "," } else { "" });
    let mut h = hdr.into_bytes();
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(b' ');
    }
    h.push(b'\n');
    let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(&h);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(p, out).unwrap();
}

struct Cand {
    name: String,
    waves: Vec<Grad>,
    m1: Vec<Vec3>,
    /// segments of the block (incl. a gap)
    n: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |n: &str| args.iter().position(|a| a == n).map(|i| args[i + 1].clone());
    let flow = PathBuf::from(opt("--flow").unwrap_or("data/hasty/seq/flow_thorax".into()));
    let out = PathBuf::from(opt("--out").unwrap_or("data/hasty/seq/studies/encrobust".into()));
    let extra: Vec<usize> = opt("--extra").unwrap_or("0,20,50,80".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let eps: f64 = opt("--eps").map_or(0.1, |s| s.parse().unwrap());
    let r_off: f64 = opt("--r-off").map_or(0.10, |s| s.parse().unwrap());
    std::fs::create_dir_all(&out).unwrap();
    let t_start = std::time::Instant::now();

    let fin: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(flow.join("final.json")).unwrap()).unwrap();
    let summ: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(flow.join("summary.json")).unwrap()).unwrap();
    let dt = fin["dt"].as_f64().unwrap();
    let t_rf = fin["t_rf"].as_f64().unwrap();
    let n_enc = fin["n_enc"].as_u64().unwrap() as usize;
    let enc_slew = fin["slew_factors"]["encode"].as_f64().unwrap();
    let b = &summ["best"];
    let (i_enc, i_ro, i_adc) = (b["i_enc"].as_u64().unwrap() as usize, b["i_ro"].as_u64().unwrap() as usize, b["i_adc"].as_u64().unwrap() as usize);
    let venc = summ["encoding"]["venc_lo"].as_f64().unwrap();
    let p_cop = summ["encoding"]["p"].as_u64().unwrap() as u32;
    assert_eq!(summ["encoding"]["scheme"].as_str().unwrap(), "coprime7", "encrobust is set up for the coprime 7-point layout");
    let scheme = Scheme::Coprime { p: p_cop, layout: DualLayout::Seven };
    let sys = System::prisma_xr();
    let enc_sys = System { smax: sys.smax * enc_slew, ..sys.clone() };
    let es = EncodeSpec { m0: [0.0; 3], m1_offset: [0.0; 3], scheme, venc, balanced: true, t_start: i_enc as f64 * dt, t_ref: t_rf };

    // readout k(t) from the echo and the least-squares weights (1 µs grid over the ADC window)
    let spg = read_npy(&flow.join("spoke_grad.npy")).unwrap().f64s();
    let spoke = Grad { dt, t0: 0.0, g: (0..=i_adc).map(|i| [spg[3 * i], spg[3 * i + 1], spg[3 * i + 2]]).collect() };
    let t_adc = spoke.duration();
    let dtf = 1e-6;
    let times: Vec<f64> = (0..(t_adc / dtf) as usize).map(|i| i as f64 * dtf).collect();
    let kabs: Vec<f64> = spoke.moments_at(0.0, &times).iter().map(|m| hasty_seq::GAMMA_BAR * norm(&m.0)).collect();
    let w_body = ball_weights(&kabs, 0.12);
    let w_vessel = ball_weights(&kabs, 0.002);
    let half = |w: &[f64]| {
        let s: f64 = w.iter().sum();
        let mut c = 0.0;
        w.iter().position(|v| {
            c += v;
            c >= 0.5 * s
        }).unwrap() as f64 * dtf
    };
    println!("readout weight: half of the body-scale weight within {:.0} µs, vessel-scale within {:.0} µs of the echo", half(&w_body) * 1e6, half(&w_vessel) * 1e6);
    let sw = |w: &[f64]| w.iter().sum::<f64>();
    let w_comb: Vec<f64> = w_body.iter().zip(&w_vessel).map(|(a, b)| a / sw(&w_body) + b / sw(&w_vessel)).collect();
    let ropts = RobustOpts::default().persist(&w_comb, dtf);

    // candidates
    let base = two_lobe_design(&es, &enc_sys);
    let n0 = base.waves[0].len() - 1;
    assert_eq!(i_enc + n0, i_ro, "encode block of the final design");
    let mut cands = vec![Cand { name: "two-lobe".into(), waves: base.waves.clone(), m1: base.m1.clone(), n: n0 }];
    for gap in [3usize, 6] {
        let waves: Vec<Grad> = base
            .waves
            .iter()
            .map(|w| {
                let mut g = w.g.clone();
                g.extend(std::iter::repeat([0.0; 3]).take(gap));
                Grad { g, ..w.clone() }
            })
            .collect();
        cands.push(Cand { name: format!("two-lobe+gap{:.0}us", gap as f64 * dt * 1e6), waves, m1: base.m1.clone(), n: n0 + gap });
    }
    // robust blocks at the design's encode slew and scaled up (the smoother blocks have PNS headroom)
    let slews: Vec<f64> = opt("--robust-slew").unwrap_or("1.0,1.25".into()).split(',').map(|s| s.parse().unwrap()).collect();
    for &sc in &slews {
        let rsys = System { smax: sys.smax * enc_slew * sc, ..sys.clone() };
        let n_min = hasty_seq::robust::min_len(&es, &rsys, (n0 as f64 / sc.cbrt()) as usize, &ropts);
        println!("robust at slew ×{sc}: feasible from {n_min} segments ({:.2} ms; two-lobe {n0})", n_min as f64 * dt * 1e3);
        let mut done = vec![];
        for &x in &extra {
            let n = (n0 + x).max(n_min);
            if done.contains(&n) {
                continue;
            }
            done.push(n);
            let (d, _) = hasty_seq::robust::design(&es, &rsys, n, &ropts).expect("robust design");
            for (e, w) in d.waves.iter().enumerate() {
                assert!(w.limits().ok_axis(&rsys), "robust encode {e} limits {:?}", w.limits());
                assert!(norm(&sub(&d.m1[e], &base.m1[e])) < 1e-6 * norm(&base.m1[e]).max(1e-12), "M1 mismatch");
            }
            let tag = if sc == 1.0 { String::new() } else { format!("s{sc}") };
            cands.push(Cand { name: format!("robust{tag}{:+.2}ms", (n as f64 - n0 as f64) * dt * 1e3), waves: d.waves, m1: d.m1, n });
        }
    }

    // tuned to a known chain: the 'resonance' variant's resonances as if measured (WP2), at the design slew ×1.25
    let known = Girf::variants().into_iter().find(|v| v.0 == "resonance").unwrap().1;
    let topts = RobustOpts { band: None, ..ropts.clone() }.with_resonances(&known, 300.0, 3000.0, 10.0);
    let tsys = System { smax: sys.smax * enc_slew * 1.25, ..sys.clone() };
    let tmin = hasty_seq::robust::min_len(&es, &tsys, 120, &topts);
    let mut tl: Vec<usize> = [tmin, n0, n0 + 10, n0 + 20, n0 + 30].into_iter().filter(|&n| n >= tmin).collect();
    tl.dedup();
    for n in tl {
        if let Some((d, _)) = hasty_seq::robust::design(&es, &tsys, n, &topts) {
            cands.push(Cand { name: format!("tuned{:+.2}ms", (n as f64 - n0 as f64) * dt * 1e3), waves: d.waves, m1: d.m1, n });
        }
    }

    // PNS: the 7 TRs of rotation 0 with the encode replaced
    let trg = read_npy(&flow.join("tr_grad.npy")).unwrap().f64s();
    let ntr = trg.len() / (3 * n_enc);
    let hw = SafeHw::example();
    let pns_of = |c: &Cand| -> (f64, f64) {
        let mut g: Vec<Vec3> = vec![];
        let mut enc_ranges = vec![];
        for _rep in 0..2 {
            for e in 0..n_enc {
                let tr = |i: usize| [trg[(e * ntr + i) * 3], trg[(e * ntr + i) * 3 + 1], trg[(e * ntr + i) * 3 + 2]];
                let s = g.len();
                for i in 0..i_enc {
                    g.push(tr(i));
                }
                let e0 = g.len();
                g.extend_from_slice(&c.waves[e].g[..c.waves[e].len() - 1]);
                enc_ranges.push((e0, g.len()));
                for i in i_ro..ntr {
                    g.push(tr(i));
                }
                let _ = s;
            }
        }
        g.push([0.0; 3]);
        let p = pns(&Grad { dt, t0: 0.0, g }, &hw);
        let enc_max = enc_ranges.iter().flat_map(|&(a, b)| (a..b).map(|j| p.norm[p.pad + j])).fold(0.0, f64::max);
        (enc_max, p.max)
    };

    let mut chains = Girf::variants();
    if args.iter().any(|a| a == "--diag") {
        // resonances alone (ideal base), and the full chain without its resonances' frequency error
        use hasty_seq::girf::Response;
        let res = |f0: f64, gain: f64| Response::Resonant { base: Box::new(Response::Ideal), f0, zeta: 0.03, gain };
        chains.push(("res_only", Girf { axis: [res(1400.0, 0.06), res(1250.0, 0.08), res(900.0, 0.05)], tail: 30e-3, ..Girf::ideal() }));
    }
    let mut rows = vec![];
    println!(
        "{:<18} {:>7} {:>6} {:>9} | v_bg @{:.0} cm [mm/s] body / vessel scale, per chain (unmodelled | modelled: RSS of single-class model errors, eps {eps})",
        "design", "TE[ms]", "PNS%", "obj",
        r_off * 100.0
    );
    let mut traces = vec![];
    for c in &cands {
        let te = (i_enc + c.n) as f64 * dt - t_rf;
        let (pns_enc, pns_all) = pns_of(c);
        let obj: f64 = (0..n_enc).flat_map(|e| (0..3).map(move |a| (e, a))).map(|(e, a)| objective(&c.waves[e].g.iter().map(|v| v[a]).collect::<Vec<_>>(), dt, &ropts, a)).fold(0.0, f64::max);
        let mut per_chain = vec![];
        let mut line = format!("{:<18} {:>7.3} {:>6.1} {:>9.3} |", c.name, te * 1e3, pns_enc, obj);
        for (cname, girf) in chains.iter() {
            let perts = girf.perturbations(eps);
            let res: Vec<Vec<(Vec<Vec3>, Vec<f64>)>> = hasty_seq::par_map(&(0..n_enc).collect::<Vec<_>>(), |&e| {
                let mut r = vec![played_offsets(girf, None, &c.waves[e], &times)];
                for (_, m) in &perts {
                    r.push(played_offsets(girf, Some(m), &c.waves[e], &times));
                }
                r
            });
            // background velocity, max over encodes, for result index m (0 = unmodelled, 1.. = each model error)
            let vbg = |m: usize, w: &[f64]| -> f64 {
                let (c0, p0) = weighted_offset(&res[0][m].0, &res[0][m].1, w);
                (1..n_enc)
                    .map(|e| {
                        let (ce, pe) = weighted_offset(&res[e][m].0, &res[e][m].1, w);
                        let dm1 = norm(&sub(&c.m1[e], &c.m1[0]));
                        ((pe - p0).abs() + 2.0 * std::f64::consts::PI * r_off * norm(&sub(&ce, &c0))) / (GAMMA * dm1)
                    })
                    .fold(0.0, f64::max)
            };
            // [unmodelled, modelled = RSS over the single-class model errors][body, vessel]
            let mut v = [[0.0f64; 2]; 2];
            let mut by_class = vec![];
            for (wi, w) in [&w_body, &w_vessel].iter().enumerate() {
                v[0][wi] = vbg(0, w);
                let per: Vec<f64> = (0..perts.len()).map(|j| vbg(j + 1, w)).collect();
                v[1][wi] = per.iter().map(|x| x * x).sum::<f64>().sqrt();
                if wi == 0 {
                    by_class = perts.iter().zip(&per).map(|((n, _), x)| json!({"class": n, "v_bg_body": x})).collect();
                }
            }
            if *cname == "resonance" {
                // traces of the full chain: encode e − encode 0, unmodelled, every 10 µs over the first 4 ms
                for e in 0..n_enc {
                    for i in (0..times.len().min(4000)).step_by(10) {
                        let d = sub(&res[e][0].0[i], &res[0][0].0[i]);
                        traces.extend_from_slice(&[d[0], d[1], d[2], res[e][0].1[i] - res[0][0].1[i]]);
                    }
                }
            }
            line += &format!(" {cname} {:.0}/{:.0} | {:.1}/{:.1};", v[0][0] * 1e3, v[0][1] * 1e3, v[1][0] * 1e3, v[1][1] * 1e3);
            per_chain.push(json!({"girf": cname, "v_bg_body": v[0][0], "v_bg_vessel": v[0][1], "v_bg_body_modelled": v[1][0], "v_bg_vessel_modelled": v[1][1], "modelled_by_class": by_class}));
        }
        // resonance scan: one band-pass peak (height 6 %, ζ 0.03) at f0 on every axis, f0 over 0.4–2.0 kHz; unmodelled
        // and after modelling with the single-class errors (f0 by eps/10, ζ, height by eps), RSS — mean and worst
        // over f0 (body scale)
        let f0s: Vec<f64> = (0..65).map(|i| 400.0 + 25.0 * i as f64).collect();
        let scan: Vec<(f64, f64)> = hasty_seq::par_map(&f0s, |&f0| {
            use hasty_seq::girf::Response;
            let r = || Response::Resonant { base: Box::new(Response::Ideal), f0, zeta: 0.03, gain: 0.06 };
            let girf = Girf { axis: [r(), r(), r()], tail: 30e-3, ..Girf::ideal() };
            let perts = girf.perturbations(eps);
            let res: Vec<Vec<(Vec<Vec3>, Vec<f64>)>> = (0..n_enc)
                .map(|e| {
                    let mut r = vec![played_offsets(&girf, None, &c.waves[e], &times[..2000])];
                    for (_, m) in &perts {
                        r.push(played_offsets(&girf, Some(m), &c.waves[e], &times[..2000]));
                    }
                    r
                })
                .collect();
            let wb = &w_body[..2000];
            let vbg = |m: usize| -> f64 {
                let (c0, p0) = weighted_offset(&res[0][m].0, &res[0][m].1, wb);
                (1..n_enc)
                    .map(|e| {
                        let (ce, pe) = weighted_offset(&res[e][m].0, &res[e][m].1, wb);
                        ((pe - p0).abs() + 2.0 * std::f64::consts::PI * r_off * norm(&sub(&ce, &c0))) / (GAMMA * norm(&sub(&c.m1[e], &c.m1[0])))
                    })
                    .fold(0.0, f64::max)
            };
            (vbg(0), (1..=perts.len()).map(|j| vbg(j).powi(2)).sum::<f64>().sqrt())
        });
        let mean = |f: &dyn Fn(&(f64, f64)) -> f64| scan.iter().map(f).sum::<f64>() / scan.len() as f64;
        let worst = |f: &dyn Fn(&(f64, f64)) -> f64| scan.iter().map(f).fold(0.0, f64::max);
        let (um, uw, mm, mw) = (mean(&|x| x.0), worst(&|x| x.0), mean(&|x| x.1), worst(&|x| x.1));
        line += &format!(" f0-scan mean/worst {:.0}/{:.0} | {:.1}/{:.1}", um * 1e3, uw * 1e3, mm * 1e3, mw * 1e3);
        println!("{line}");
        rows.push(json!({"design": c.name, "segments": c.n, "te": te, "pns_encode": pns_enc, "pns_all": pns_all, "objective_max": obj, "chains": per_chain,
                         "res_scan": {"f0": f0s, "v_bg_body": scan.iter().map(|x| x.0).collect::<Vec<_>>(), "v_bg_body_modelled": scan.iter().map(|x| x.1).collect::<Vec<_>>(),
                                      "mean": um, "worst": uw, "mean_modelled": mm, "worst_modelled": mw}}));
    }

    // arrays: encode waveforms (padded), sensitivity curves, spectra, traces, weights
    let nmax = cands.iter().map(|c| c.n).max().unwrap() + 1;
    let mut wv = vec![0.0; cands.len() * n_enc * nmax * 3];
    for (ci, c) in cands.iter().enumerate() {
        for e in 0..n_enc {
            for (i, g) in c.waves[e].g.iter().enumerate() {
                for a in 0..3 {
                    wv[((ci * n_enc + e) * nmax + i) * 3 + a] = g[a];
                }
            }
        }
    }
    write_f8(&out.join("enc_waves.npy"), &[cands.len(), n_enc, nmax, 3], &wv);
    let taus: Vec<f64> = (0..61).map(|i| 1e-6 * 10f64.powf(i as f64 * 4.0 / 60.0)).collect();
    let freqs: Vec<f64> = (0..301).map(|i| i as f64 * 10.0).collect();
    let mut sens = vec![];
    let mut spec = vec![];
    for c in &cands {
        // encode 1 − encode 0 on axis x (what the x velocity sees)
        let d: Vec<f64> = c.waves[1].g.iter().zip(&c.waves[0].g).map(|(a, b)| a[0] - b[0]).collect();
        sens.extend(taus.iter().map(|&t| hasty_seq::GAMMA_BAR * sensitivity(&d, dt, t)));
        spec.extend(freqs.iter().map(|&f| hasty_seq::GAMMA_BAR * spectrum(&d, dt, f)));
    }
    write_f8(&out.join("sens_tau.npy"), &[taus.len()], &taus);
    write_f8(&out.join("sens.npy"), &[cands.len(), taus.len()], &sens);
    write_f8(&out.join("spec_f.npy"), &[freqs.len()], &freqs);
    write_f8(&out.join("spec.npy"), &[cands.len(), freqs.len()], &spec);
    let nt = (0..times.len().min(4000)).step_by(10).count();
    write_f8(&out.join("traces.npy"), &[cands.len(), n_enc, nt, 4], &traces);
    let sub_w: Vec<f64> = (0..times.len().min(4000)).step_by(10).flat_map(|i| [times[i], w_body[i], w_vessel[i], kabs[i]]).collect();
    write_f8(&out.join("weights.npy"), &[nt, 4], &sub_w);
    let summary = json!({
        "flow": flow, "eps": eps, "r_off": r_off, "dt": dt, "t_rf": t_rf, "i_enc": i_enc, "venc_lo": venc, "enc_slew": enc_slew,
        "designs": cands.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        "chains": chains.iter().map(|c| c.0).collect::<Vec<_>>(),
        "robust_opts": {"taus": ropts.taus, "band": ropts.band, "jerk": ropts.jerk},
        "weights_half_us": {"body": half(&w_body) * 1e6, "vessel": half(&w_vessel) * 1e6},
        "rows": rows,
    });
    std::fs::write(out.join("encrobust.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("→ {} ({:.1} s)", out.display(), t_start.elapsed().as_secs_f64());
}
