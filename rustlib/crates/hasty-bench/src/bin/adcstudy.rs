//! Readout-length study: how far does a longer ADC window pay off once B0, concomitant fields and the GIRF are part of
//! the reconstruction operator? For each readout budget T_adc and a few base-spoke shapes, the full design (N fit,
//! part-wise PNS derating on the played sequence) is evaluated and reported:
//!  - efficiency: N, TR, N·TR per encode, ADC duty T_adc/TR (scan-time-normalised SNR ∝ √duty), and the duty if the
//!    ADC also runs during the encode and the tail;
//!  - what grows with T_adc and is *not* fixed by the operator unless modelled: mean sample time of the outer k shell
//!    (T2* weighting, off-resonance / field-map-error phase, fat phase), readout γM1 (flow displacement), concomitant
//!    phase scale of the readout;
//!  - encode and tail k-excursions (relevant when sampling them): max |k| during the encode, share of tail samples
//!    inside kmax.
//! Results are appended to <out>/results.jsonl as they finish (resumable; STOP file / SIGINT stops).
//!   adcstudy <out> [--res 2e-3] [--fov 0.32] [--scheme coprime7] [--p 2] [--v-ext 3.0] [--t-adcs 2,3,4,6,8,10,12,16]
//!            [--shapes "0,0,2;0.086,0.375,1.494;0.5,1,2"] [--n-rot-pns 400] [--spoil-cycles 2] [--threads 16]
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use hasty_seq::encode::{DualLayout, Scheme};
use hasty_seq::metrics::{concomitant_phase, readout_m1};
use hasty_seq::optim::{Problem, Shape};
use hasty_seq::pns::SafeHw;
use hasty_seq::system::System;
use hasty_seq::traj::RotSet;
use hasty_seq::{norm, GAMMA, GAMMA_BAR};
use serde_json::json;

fn key(t_adc: f64, s: &Shape) -> String {
    format!("{:016x}{:016x}{:016x}{:016x}", t_adc.to_bits(), s.kg0.to_bits(), s.wobble.to_bits(), s.wobble_freq.to_bits())
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let res: f64 = take("--res").map(|s| s.parse().unwrap()).unwrap_or(2.0e-3);
    let fov: f64 = take("--fov").map(|s| s.parse().unwrap()).unwrap_or(0.32);
    let p_cop: u32 = take("--p").map(|s| s.parse().unwrap()).unwrap_or(2);
    let v_ext: f64 = take("--v-ext").map(|s| s.parse().unwrap()).unwrap_or(3.0);
    let scheme = match take("--scheme").as_deref().unwrap_or("coprime7") {
        "coprime7" => Scheme::Coprime { p: p_cop, layout: DualLayout::Seven },
        "coprime8" => Scheme::Coprime { p: p_cop, layout: DualLayout::Eight },
        "referenced" => Scheme::Referenced,
        s => panic!("unsupported scheme {s}"),
    };
    let venc = match scheme {
        Scheme::Coprime { p, .. } => v_ext / (p as f64 + 1.0),
        _ => v_ext,
    };
    let t_adcs: Vec<f64> = take("--t-adcs").unwrap_or_else(|| "2,3,4,6,8,10,12,16".into()).split(',').map(|s| s.parse::<f64>().unwrap() * 1e-3).collect();
    let shapes: Vec<Shape> = take("--shapes")
        .unwrap_or_else(|| "0,0,2;0.086,0.375,1.494;0.5,1,2".into())
        .split(';')
        .map(|s| {
            let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
            Shape { kg0: v[0], wobble: v[1], wobble_freq: v[2] }
        })
        .collect();
    let n_rot_pns: usize = take("--n-rot-pns").map(|s| s.parse().unwrap()).unwrap_or(400);
    let spoil_cycles: f64 = take("--spoil-cycles").map(|s| s.parse().unwrap()).unwrap_or(2.0);
    let threads: usize = take("--threads").map(|s| s.parse().unwrap()).unwrap_or(16);
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    let ctl = hasty_recon::checkpoint::RunCtl::new(&out, 1);

    let sys = System::prisma_xr();
    let kmax = 0.5 / res;
    let p = Problem {
        sys: sys.clone(),
        hw: SafeHw::example(),
        pns_target: 80.0,
        kmax,
        fov,
        edge_density: 1.0,
        t_adc: 0.0,
        vmax: None,
        rf_dur: 100e-6,
        scheme,
        venc,
        balanced: true,
        spoil: hasty_seq::tr::Spoil::Physical([0.0, 0.0, spoil_cycles / (GAMMA_BAR * res)]),
        rot_kind: RotSet::SuperFibonacci,
        n_rot_pns,
        aniso_weight: 0.0,
        curve_pts: 60_000,
        ret: None,
    };
    let fp = format!("{res}|{fov}|{scheme:?}|{venc}|{n_rot_pns}|{spoil_cycles}|{:?}", p.hw);
    let fp_file = out.join("problem.txt");
    match std::fs::read_to_string(&fp_file) {
        Ok(s) if s != fp => panic!("{} holds a different problem; use a new output dir", out.display()),
        Ok(_) => {}
        Err(_) => std::fs::write(&fp_file, &fp).unwrap(),
    }
    let res_file = out.join("results.jsonl");
    let done: HashSet<String> = std::fs::read_to_string(&res_file)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v["key"].as_str().map(String::from))
        .collect();
    let jobs: Vec<(f64, Shape)> = t_adcs.iter().flat_map(|&t| shapes.iter().map(move |s| (t, *s))).filter(|(t, s)| !done.contains(&key(*t, s))).collect();
    println!(
        "adcstudy: res {:.2} mm, FOV {:.0} mm, {scheme:?} VENC {venc:.3} m/s, {} T_adc × {} shapes, {} to do ({} cached), EXAMPLE PNS hw",
        res * 1e3, fov * 1e3, t_adcs.len(), shapes.len(), jobs.len(), done.len()
    );
    let rots = p.pns_rotations();
    let file = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&res_file).unwrap());
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads.min(jobs.len()) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= jobs.len() || ctl.stop_requested() {
                    break;
                }
                let (t_adc, s) = jobs[i];
                let pt = Problem { t_adc, ..p.clone() };
                let t0 = std::time::Instant::now();
                let rec = match pt.evaluate(&s, &rots) {
                    None => json!({"key": key(t_adc, &s), "t_adc_budget": t_adc, "shape": [s.kg0, s.wobble, s.wobble_freq], "feasible": false}),
                    Some(e) => {
                        let tr = &e.tr;
                        let sp = &tr.spoke;
                        let dt = tr.dt;
                        // sample times of the outer k shell (0.9–1.0 kmax) and of the 0.45–0.55 shell
                        let m = sp.grad.moments(0.0);
                        let k = m.k();
                        let shell = |a: f64, b: f64| {
                            let ts: Vec<f64> = k.iter().enumerate().filter(|(_, q)| (a * kmax..=b * kmax).contains(&norm(q))).map(|(j, _)| j as f64 * dt).collect();
                            if ts.is_empty() { f64::NAN } else { ts.iter().sum::<f64>() / ts.len() as f64 }
                        };
                        let (t_mid, t_out) = (shell(0.45, 0.55), shell(0.9, 1.0001));
                        let m1 = readout_m1(&sp.grad, 0.0, kmax, &[0.5, 1.0]);
                        let m1_end = GAMMA * norm(&m.m1[sp.i_adc]);
                        let conc = concomitant_phase(&m.p[sp.i_adc], sys.b0, 0.15);
                        // encode / tail k excursions (physical axes, rotation 0, every encode)
                        let phys = tr.physical_all(&rots[0]);
                        let (mut kenc, mut tail_in, mut tail_n, mut enc_in, mut enc_n) = (0.0f64, 0usize, 0usize, 0usize, 0usize);
                        for g in &phys {
                            let mut kk = [0.0f64; 3];
                            for j in 1..tr.n {
                                for a in 0..3 {
                                    kk[a] += 0.5 * (g[j - 1][a] + g[j][a]) * dt * GAMMA_BAR;
                                }
                                if j < tr.i_ro && j >= tr.i_enc {
                                    kenc = kenc.max(norm(&kk));
                                    enc_n += 1;
                                    if norm(&kk) <= kmax {
                                        enc_in += 1;
                                    }
                                }
                                if j > tr.i_tail {
                                    tail_n += 1;
                                    if norm(&kk) <= kmax {
                                        tail_in += 1;
                                    }
                                }
                            }
                        }
                        let t_enc0 = tr.i_enc as f64 * dt;
                        json!({
                            "key": key(t_adc, &s), "t_adc_budget": t_adc, "shape": [s.kg0, s.wobble, s.wobble_freq], "feasible": e.feasible,
                            "n": e.n_design, "tr": tr.tr(), "te": tr.te(), "t_adc": sp.t_adc, "n_tr": e.objective,
                            "scan_nyquist_all_enc": e.objective * tr.n_enc() as f64,
                            "duty": sp.t_adc / tr.tr(), "duty_all": (tr.tr() - t_enc0) / tr.tr(),
                            "encode": (tr.i_ro - tr.i_enc) as f64 * dt, "tail": tr.n_tail as f64 * dt,
                            "pns": e.pns.max, "pns_part": format!("{:?}", e.pns.part), "slew": e.slew, "aniso": e.aniso,
                            "t_shell_mid": t_mid, "t_shell_outer": t_out, "length": sp.length,
                            "gm1_first50": m1[0], "gm1_first100": m1[1], "gm1_end": m1_end, "conc_15cm": conc,
                            "k_enc_max_over_kmax": kenc / kmax, "tail_inside_kmax": tail_in as f64 / tail_n.max(1) as f64,
                            "enc_inside_kmax": enc_in as f64 / enc_n.max(1) as f64,
                            // ADC duty counting only samples with |k| ≤ kmax when the ADC runs from the encode start to the TR end
                            "duty_all_in_kmax": (sp.t_adc / dt + (enc_in + tail_in) as f64 / phys.len() as f64) / tr.n as f64,
                            "secs": t0.elapsed().as_secs_f64(),
                        })
                    }
                };
                println!(
                    "  T_adc {:5.1} ms shape {:?}: N {} TR {} ms N·TR {} s duty {} ({:.0} s)",
                    t_adc * 1e3, [s.kg0, s.wobble, s.wobble_freq], rec["n"].as_f64().map_or("-".into(), |x| format!("{x:.0}")),
                    rec["tr"].as_f64().map_or("-".into(), |x| format!("{:.2}", x * 1e3)), rec["n_tr"].as_f64().map_or("-".into(), |x| format!("{x:.2}")),
                    rec["duty"].as_f64().map_or("-".into(), |x| format!("{x:.2}")), t0.elapsed().as_secs_f64()
                );
                let mut f = file.lock().unwrap();
                writeln!(f, "{}", rec).unwrap();
                f.flush().unwrap();
            });
        }
    });
    if ctl.stop_requested() {
        println!("stopped; rerun to resume");
    } else {
        println!("done → {}", res_file.display());
    }
}
