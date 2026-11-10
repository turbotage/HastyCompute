//! Spoiling check of designed TRs (hasty_seq::spoil): for several readout/spoiler variants of the same thorax design,
//! the multi-isochromat RF-spoiled steady state with the actual per-TR moments (rotations in acquisition order, all
//! encodes per rotation) for static tissue and plug-flowing blood. Reports the steady-state signal relative to the ideal
//! (Ernst) value, its TR-to-TR fluctuation, and for flowing blood the velocity error of every encode pair (mean complex
//! echo per encode). Writes <out>/spoilcheck.json and <out>/traces.npy; figures: tools/seq/plot_spoilcheck.py <out>.
//!   spoilcheck <out> [--t-adc 8e-3] [--res 2e-3] [--fov 0.32] [--flip 10] [--n-iso 16384]
use std::path::PathBuf;

use hasty_phantom::volume::write_npy;
use hasty_seq::encode::{DualLayout, Scheme};
use hasty_seq::optim::{Problem, Shape};
use hasty_seq::pns::SafeHw;
use hasty_seq::spoil::{ernst, simulate, SpoilSim, Tissue, TrMoments};
use hasty_seq::system::System;
use hasty_seq::tr::Spoil;
use hasty_seq::traj::{acquisition_order, rotations, RotSet};
use hasty_seq::venc::plan;
use hasty_seq::{norm, GAMMA_BAR};
use serde_json::json;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let t_adc: f64 = take("--t-adc").map(|s| s.parse().unwrap()).unwrap_or(8e-3);
    let res: f64 = take("--res").map(|s| s.parse().unwrap()).unwrap_or(2e-3);
    let fov: f64 = take("--fov").map(|s| s.parse().unwrap()).unwrap_or(0.32);
    let flip: f64 = take("--flip").map(|s| s.parse::<f64>().unwrap()).unwrap_or(10.0).to_radians();
    let n_iso: usize = take("--n-iso").map(|s| s.parse().unwrap()).unwrap_or(16384);
    // --vsweep: blood only, many plug-flow velocities, for the variants named by their first letters
    let vsweep: Option<String> = take("--vsweep");
    // voxel centre distance from isocentre (m), along a fixed oblique direction
    let off: f64 = take("--offset").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let offset = { let d = [0.6f64, 0.3, 0.74]; let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(); d.map(|x| x / n * off) };
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    let kmax = 0.5 / res;
    let scheme = Scheme::Coprime { p: 2, layout: DualLayout::Seven };
    let venc = 1.0;
    let base = Problem {
        sys: System::prisma_xr(),
        hw: SafeHw::example(),
        pns_target: 80.0,
        kmax,
        fov,
        edge_density: 1.0,
        t_adc,
        vmax: None,
        rf_dur: 100e-6,
        scheme,
        venc,
        balanced: true,
        spoil: Spoil::Endpoint,
        rot_kind: RotSet::SuperFibonacci,
        n_rot_pns: 400,
        aniso_weight: 0.0,
        res_weight: 0.0,
        res_band: vec![],
        curve_pts: 60_000,
        ret: Some(0.0),
    };
    let zk = |c: f64| Spoil::Physical([0.0, 0.0, c / (GAMMA_BAR * res)]);
    let rad = |c: f64| Spoil::Radial(c / (GAMMA_BAR * res));
    // (name, return radius (None = centre-out, not at rest), spoil, M1-equalised what-if)
    // last flag: every TR its own rotation (encode e of group i takes rotation 7i + e of a 7N set) instead of all
    // encodes of a group sharing one
    // (…, M1-equalised, own rotation per TR, encode order cycled by the group index)
    let variants: Vec<(&str, Option<f64>, Spoil, bool, bool, bool)> = vec![
        ("A out-and-back + separate z spoil 0.5 cyc/vox (= kmax)", Some(0.0), zk(0.5), false, false, false),
        ("B out-and-back + separate z spoil 2 cyc/vox", Some(0.0), zk(2.0), false, false, false),
        ("C out, stop at kmax = spoil 0.5 cyc/vox rotating", Some(1.0), Spoil::Endpoint, false, false, false),
        ("D centre-out, continues out to 2 cyc/vox rotating", None, rad(2.0), false, false, false),
        ("E centre-out, continues out to 4 cyc/vox rotating", None, rad(4.0), false, false, false),
        ("F out-and-back, no spoiler (net 0)", Some(0.0), Spoil::Endpoint, false, false, false),
        ("B' = B with equal TR M1 across encodes (what-if)", Some(0.0), zk(2.0), true, false, false),
        ("D' = D with equal TR M1 across encodes (what-if)", None, rad(2.0), true, false, false),
        ("Du = D, own rotation every TR", None, rad(2.0), false, true, false),
        ("Eu = E, own rotation every TR", None, rad(4.0), false, true, false),
        ("Cu = C, own rotation every TR", Some(1.0), Spoil::Endpoint, false, true, false),
        ("Bp = B, encode order cycled per group", Some(0.0), zk(2.0), false, false, true),
        ("Dp = D, encode order cycled per group", None, rad(2.0), false, false, true),
        ("Ep = E, encode order cycled per group", None, rad(4.0), false, false, true),
        ("G out-and-back + per-encode block: z spoil 2 cyc/vox + equal TR M1 (FlowComp)", Some(0.0), Spoil::FlowComp([0.0, 0.0, 2.0 / (GAMMA_BAR * res)]), false, false, false),
    ];
    let shape = Shape { kg0: 0.0, wobble: 0.0, wobble_freq: 2.0 };
    let pl = plan(scheme, venc);
    let mut designs = vec![];
    for (vi, (name, ret, spoil, m1eq, ungroup)) in variants.iter().map(|v| (v.0, v.1, v.2, v.3, v.4)).enumerate() {
        let cyc = variants[vi].5;
        let (ret, spoil, m1eq, ungroup) = (&ret, &spoil, &m1eq, &ungroup);
        let p = Problem { ret: *ret, spoil: *spoil, ..base.clone() };
        let rots_pns = p.pns_rotations();
        let e = p.evaluate(&shape, &rots_pns).expect("design");
        let n = e.n_eff.ceil() as usize;
        let ne = e.tr.n_enc();
        let nr = if *ungroup { n * ne } else { n };
        let rs = rotations(RotSet::SuperFibonacci, nr);
        let ord = acquisition_order(RotSet::SuperFibonacci, nr);
        let tr = &e.tr;
        // tails of the acquisition set may exceed the PNS-set TR: rebuild the TR for the acquisition rotations
        let acq: Vec<_> = ord.iter().map(|&i| rs[i]).collect();
        let trb = hasty_seq::tr::Tr::build(&p.tr_spec(&e.slew), tr.spoke.clone(), &acq);
        let mut trs = vec![];
        let mut labels = vec![];
        let groups: Vec<Vec<hasty_seq::Mat3>> = if *ungroup { acq.chunks(ne).map(|c| c.to_vec()).collect() } else { acq.iter().map(|r| vec![*r; ne]).collect() };
        for (gi, g) in groups.iter().enumerate() {
            let mut m1_0 = [0.0; 3];
            for pos in 0..trb.n_enc() {
                let ee = if cyc { (pos + gi) % ne } else { pos };
                labels.push(ee);
                let m = trb.physical(ee, &g[pos]).moments(0.0);
                let (m0, m1, _, _) = m.last();
                if ee == 0 {
                    m1_0 = m1;
                }
                // what-if: an extra lobe makes every encode's TR first moment equal (time cost ignored)
                trs.push(TrMoments { m1_echo: m.m1[trb.i_ro], k_net: m0.map(|x| x * GAMMA_BAR), m1_net: if *m1eq { m1_0 } else { m1 } });
            }
        }
        let kn = trs.iter().map(|t| norm(&t.k_net)).fold(0.0, f64::max) / kmax;
        // diagnostics: per rotation group, spread across encodes of the TR's net M1 (as phase at |v| = 1 m/s) and net k
        let (mut dm1, mut dk, mut m1rot): (f64, f64, f64) = (0.0, 0.0, 0.0);
        let ne_ = trb.n_enc();
        for (gi, grp) in trs.chunks(ne_).enumerate() {
            for t in grp {
                dm1 = dm1.max(hasty_seq::GAMMA * norm(&hasty_seq::sub(&t.m1_net, &grp[0].m1_net)));
                dk = dk.max(norm(&hasty_seq::sub(&t.k_net, &grp[0].k_net)));
            }
            if gi > 0 {
                m1rot = m1rot.max(hasty_seq::GAMMA * norm(&hasty_seq::sub(&grp[0].m1_net, &trs[0].m1_net)));
            }
        }
        println!("   diag: net-M1 spread across encodes {dm1:.3e} rad/(m/s), across rotations {m1rot:.3e} rad/(m/s); net-k spread across encodes {dk:.3e} /m");
        println!("{name}: N {n}, TR {:.3} ms, TE {:.3} ms, |k_net| ≤ {kn:.3} kmax, {} TRs", trb.tr() * 1e3, trb.te() * 1e3, trs.len());
        designs.push((name.to_string(), trb.tr(), trb.te(), trs, trb.n_enc(), e.objective, labels));
    }
    if let Some(sel) = &vsweep {
        let pick: Vec<usize> = designs.iter().enumerate().filter(|(_, d)| sel.split(',').any(|k| d.0.starts_with(&format!("{k} ")))).map(|(i, _)| i).collect();
        let blood = Tissue { t1: 1.7, t2: 0.2 };
        let dirs = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.577, 0.577, 0.577], [0.707, -0.5, 0.5]];
        let mags = [0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 2.5];
        let mut jobs = vec![];
        for &d in &pick {
            for dir in &dirs {
                for &m in &mags {
                    for inc in [117.0f64, 50.0] {
                        let n = norm(dir);
                        jobs.push((d, dir.map(|x| x / n * m), inc));
                    }
                }
            }
        }
        let res_v = hasty_seq::par_map(&jobs, |&(d, v, inc)| {
            let (_, tr, te, trs, ne, _, labels) = &designs[d];
            let sim = SpoilSim { flip, tr: *tr, te: *te, rf_inc: inc.to_radians(), voxel: res, n_iso, offset };
            let s = simulate(&sim, &blood, trs, &v);
            let ideal = ernst(&sim, &blood);
            let ss = &s[s.len() / 2..];
            let mags: Vec<f64> = ss.iter().map(|(a, b)| (a * a + b * b).sqrt()).collect();
            let mean = mags.iter().sum::<f64>() / mags.len() as f64;
            let sd = (mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / mags.len() as f64).sqrt();
            let i0 = s.len() / 2 / ne * ne;
            let mut acc = vec![(0.0f64, 0.0f64); *ne];
            for (i, (a, b)) in s.iter().enumerate().skip(i0) {
                acc[labels[i]].0 += a;
                acc[labels[i]].1 += b;
            }
            let mut verr: f64 = 0.0;
            for cb in &pl.combos {
                for pr in std::iter::once(&cb.lo).chain(cb.hi.iter()) {
                    let (p, m) = (acc[pr.plus], acc[pr.minus]);
                    let meas = (p.1 * m.0 - p.0 * m.1).atan2(p.0 * m.0 + p.1 * m.1);
                    let want = pr.g[0] * v[0] + pr.g[1] * v[1] + pr.g[2] * v[2];
                    let e = (meas - want + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
                    verr = verr.max((e / norm(&pr.g)).abs());
                }
            }
            (mean / ideal, sd / mean, verr)
        });
        let mut rows = vec![];
        println!("\nvelocity sweep (blood T1 1.7 s / T2 200 ms, {} velocities × 2 RF increments per variant)", dirs.len() * mags.len());
        println!(" variant | |S|/ideal min–max | fluct median / max | velocity error median / 90% / max (mm/s)");
        for &d in &pick {
            let r: Vec<_> = jobs.iter().zip(&res_v).filter(|(j, _)| j.0 == d).collect();
            let mut ve: Vec<f64> = r.iter().map(|(_, x)| x.2 * 1e3).collect();
            let mut fl: Vec<f64> = r.iter().map(|(_, x)| x.1).collect();
            ve.sort_by(|a, b| a.partial_cmp(b).unwrap());
            fl.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let smin = r.iter().map(|(_, x)| x.0).fold(f64::INFINITY, f64::min);
            let smax = r.iter().map(|(_, x)| x.0).fold(0.0, f64::max);
            let q = |v: &[f64], f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
            println!(" {} | {smin:.3}–{smax:.3} | {:.3} / {:.3} | {:.2} / {:.2} / {:.2}", designs[d].0, q(&fl, 0.5), q(&fl, 1.0), q(&ve, 0.5), q(&ve, 0.9), q(&ve, 1.0));
            for (j, x) in &r {
                rows.push(json!({"variant": designs[d].0, "v": j.1, "rf_inc_deg": j.2, "rel_signal": x.0, "fluct": x.1, "v_err": x.2}));
            }
        }
        std::fs::write(out.join("vsweep.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        return;
    }
    let tissues = [("myocardium", Tissue { t1: 1.45, t2: 0.045 }, [0.0; 3]), ("blood static", Tissue { t1: 1.7, t2: 0.2 }, [0.0; 3]), ("blood 0.71 m/s", Tissue { t1: 1.7, t2: 0.2 }, [0.5, 0.3, -0.4])];
    let incs = [117.0f64, 50.0];
    let mut jobs = vec![];
    for d in 0..designs.len() {
        for t in 0..tissues.len() {
            for &inc in &incs {
                jobs.push((d, t, inc));
            }
        }
    }
    let res_all = hasty_seq::par_map(&jobs, |&(d, t, inc)| {
        let (_, tr, te, trs, _, _, _) = &designs[d];
        let sim = SpoilSim { flip, tr: *tr, te: *te, rf_inc: inc.to_radians(), voxel: res, n_iso, offset };
        simulate(&sim, &tissues[t].1, trs, &tissues[t].2)
    });
    let mut rows = vec![];
    let nmax = designs.iter().map(|d| d.3.len()).max().unwrap();
    let mut traces = vec![f64::NAN; jobs.len() * nmax];
    println!("\n variant | tissue | RF inc | |S|/ideal | TR-to-TR fluct. | max velocity error (mm/s)");
    for (j, (&(d, t, inc), s)) in jobs.iter().zip(&res_all).enumerate() {
        let (name, tr, te, _, ne, _, labels) = &designs[d];
        let sim = SpoilSim { flip, tr: *tr, te: *te, rf_inc: inc.to_radians(), voxel: res, n_iso, offset };
        let ideal = ernst(&sim, &tissues[t].1);
        let ss = &s[s.len() / 2..];
        let mags: Vec<f64> = ss.iter().map(|(a, b)| (a * a + b * b).sqrt()).collect();
        let mean = mags.iter().sum::<f64>() / mags.len() as f64;
        let sd = (mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / mags.len() as f64).sqrt();
        for (i, (a, b)) in s.iter().enumerate() {
            traces[j * nmax + i] = (a * a + b * b).sqrt() / ideal;
        }
        // mean complex echo per encode over the steady state (TR index i ↔ encode i % ne)
        let i0 = s.len() / 2 / ne * ne;
        let mut acc = vec![(0.0f64, 0.0f64); *ne];
        for (i, (a, b)) in s.iter().enumerate().skip(i0) {
            acc[labels[i]].0 += a;
            acc[labels[i]].1 += b;
        }
        let v = tissues[t].2;
        let mut verr: f64 = 0.0;
        for cb in &pl.combos {
            for pr in std::iter::once(&cb.lo).chain(cb.hi.iter()) {
                let (p, m) = (acc[pr.plus], acc[pr.minus]);
                let meas = (p.1 * m.0 - p.0 * m.1).atan2(p.0 * m.0 + p.1 * m.1);
                let want = pr.g[0] * v[0] + pr.g[1] * v[1] + pr.g[2] * v[2];
                let mut e = meas - want;
                e = (e + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
                verr = verr.max((e / norm(&pr.g)).abs());
            }
        }
        println!(" {name} | {} | {inc:.0}° | {:.3} | {:.3} | {:.2}", tissues[t].0, mean / ideal, sd / mean, verr * 1e3);
        rows.push(json!({"variant": name, "tissue": tissues[t].0, "rf_inc_deg": inc, "rel_signal": mean / ideal, "fluct": sd / mean,
                         "v_err_max": verr, "tr": tr, "te": te, "n_tr": s.len(), "velocity": v, "t1": tissues[t].1.t1, "t2": tissues[t].1.t2}));
    }
    let b: Vec<u8> = traces.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy(&out.join("traces.npy"), "<f8", &[jobs.len(), nmax], false, &b).unwrap();
    let summary = json!({"flip_deg": flip.to_degrees(), "t_adc": t_adc, "res": res, "fov": fov, "n_iso": n_iso, "scheme": format!("{scheme:?}"), "venc_lo": venc,
                         "designs": designs.iter().map(|d| json!({"variant": d.0, "tr": d.1, "te": d.2, "n_tr_s": d.5})).collect::<Vec<_>>(), "rows": rows});
    std::fs::write(out.join("spoilcheck.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("wrote {}", out.display());
}
