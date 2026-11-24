//! CSF inflow-signal experiments (Söderström et al., MRM 2026) on the spin-history simulator `hasty_mrsim::inflow`.
//! CPU only. Every command writes arrays (.npy, f64) and a summary.json under <out>; figures: research/experiments/csf_inflow/plot.py <out>.
//!
//!   inflowsim knee      <out>   signal vs constant plug velocity (static → fresh), slice gap / crosstalk reference
//!   inflowsim phantom   <out>   the paper's phantom grid: 0.93 / 0.23 / 0.077 Hz sinusoids at six amplitudes each, 2 min
//!   inflowsim alias     <out>   spectrum of the cardiac-driven signal at TR 2 s (aliasing to 0.07 Hz) vs short TR
//!   inflowsim sweeps    <out>   dispersion, T1, slice order, slice gap, flip angle: modulation per unit velocity
//!   inflowsim roundtrip <out>   physiology-driven truth (Monte Carlo, non-sinusoidal) → quasi-stationary estimator
//! Options: --n-sim 6 --dz 80e-6 --noise 0.0 --seed 1
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};

use hasty_mrsim::inflow::*;
use hasty_mrsim::physio::Ecg;

fn write_npy(path: &Path, shape: &[usize], data: &[f64]) {
    let shp = if shape.len() == 1 { format!("({},)", shape[0]) } else { format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")) };
    let mut h = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shp}, }}");
    while (h.len() + 11) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(h.len() as u16).to_le_bytes()).unwrap();
    f.write_all(h.as_bytes()).unwrap();
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

struct Opts {
    n_sim: usize,
    dz: f64,
    noise: f64,
    seed: u64,
}

fn paper_acq(o: &Opts) -> Acq {
    let mut a = Acq::paper();
    a.n_sim = o.n_sim;
    a.dz = o.dz;
    a
}

fn sinusoid(f: f64, amp_mm_s: f64, t_end: f64) -> Component {
    Component::new(PhaseClock::constant(f, 0.0, t_end, 0.01), amp_mm_s * 1e-3, 0.0)
}

/// modulation amplitude (fraction of the mean) at the aliased frequency of f
fn modulation(sim: &InflowSim, f: f64, amp_mm_s: f64, o: &Opts) -> f64 {
    let acq = &sim.acq;
    let flow = flow_from_components(&[sinusoid(f, amp_mm_s, acq.t_end() + 1.0)], acq.t_end() + 1.0, 2e-3);
    let out = sim.run(&flow);
    let mut s = out.normalized(0.0);
    if o.noise > 0.0 {
        add_noise(&mut s, o.noise, o.seed);
    }
    tone_amplitude(&out.t, &s, alias_freq(f, 1.0 / acq.tr))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let o = Opts {
        n_sim: take("--n-sim").map(|s| s.parse().unwrap()).unwrap_or(6),
        dz: take("--dz").map(|s| s.parse().unwrap()).unwrap_or(80e-6),
        noise: take("--noise").map(|s| s.parse().unwrap()).unwrap_or(0.0),
        seed: take("--seed").map(|s| s.parse().unwrap()).unwrap_or(1),
    };
    if args.len() < 2 {
        eprintln!("usage: inflowsim <knee|phantom|alias|sweeps|roundtrip> <out> [--n-sim 6] [--dz 80e-6] [--noise 0] [--seed 1]");
        std::process::exit(2);
    }
    let cmd = args[0].clone();
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let mut summary = String::new();
    let t0 = std::time::Instant::now();
    match cmd.as_str() {
        "knee" => {
            // 4 TRs at constant velocity: element reaching the slice has been excited 0..few times; D = 0 so the knee is geometric
            let mut acq = paper_acq(&o);
            acq.n_dummy = 1;
            acq.n_scan = 3;
            let mut med = Medium::csf();
            med.diffusion = 0.0;
            let vs: Vec<f64> = (0..=40).map(|i| i as f64 * 0.25).collect(); // mm/s
            let mut rows = vec![];
            for &v in &vs {
                let sim = InflowSim::new(acq.clone(), med.clone());
                let o2 = sim.run(&Flow::from_vel(acq.t_end() + 1.0, 0.01, |_| v * 1e-3));
                rows.push(o2.raw[2]);
            }
            let fresh = *rows.last().unwrap();
            let rel: Vec<f64> = rows.iter().map(|s| s / fresh).collect();
            write_npy(&out.join("knee_v_mm_s.npy"), &[vs.len()], &vs);
            write_npy(&out.join("knee_signal_rel_fresh.npy"), &[rel.len()], &rel);
            writeln!(summary, "\"thickness_over_tr_mm_s\": {:.3}, \"static_rel_fresh\": {:.4}", acq.thickness / acq.tr * 1e3, rel[0]).unwrap();
            for (v, r) in vs.iter().zip(&rel).step_by(4) {
                println!("v {v:5.2} mm/s  signal/fresh {r:.4}");
            }
        }
        "phantom" => {
            let mut acq = paper_acq(&o);
            acq.n_scan = 60; // 2 min
            acq.n_dummy = 10;
            let sim = InflowSim::new(acq.clone(), Medium::csf());
            let comps = [("cardiac", 0.93, 20.0), ("respiratory", 0.23, 5.0), ("low", 0.077, 2.5)];
            let mut json = vec![];
            for (name, f, vmax) in comps {
                let amps: Vec<f64> = (0..6).map(|i| vmax * i as f64 / 5.0).collect();
                let m: Vec<f64> = amps.iter().map(|&a| modulation(&sim, f, a, &o)).collect();
                // linearity: Pearson r of modulation against amplitude, and the same restricted to the lowest four levels
                let r = pearson(&amps, &m);
                let r4 = pearson(&amps[..4], &m[..4]);
                println!("{name:12} f {f} Hz (alias {:.3}): modulation % {:?}  r {r:.4}  r(lowest 4) {r4:.4}", alias_freq(f, 0.5), m.iter().map(|x| (x * 1e4).round() / 100.0).collect::<Vec<_>>());
                write_npy(&out.join(format!("phantom_{name}_amp_mm_s.npy")), &[6], &amps);
                write_npy(&out.join(format!("phantom_{name}_modulation.npy")), &[6], &m);
                json.push(format!("\"{name}\": {{\"f\": {f}, \"alias\": {:.4}, \"r\": {r:.5}, \"r_lowest4\": {r4:.5}}}", alias_freq(f, 0.5)));
            }
            summary = json.join(", ");
        }
        "alias" => {
            let comps_f = 0.93;
            let mut json = vec![];
            for (tag, tr, nsl) in [("tr2.0", 2.0, 37usize), ("tr0.5", 0.5, 9)] {
                let mut acq = paper_acq(&o);
                acq.tr = tr;
                acq.n_slices = nsl;
                acq.n_scan = (360.0 / tr) as usize;
                acq.n_dummy = (20.0 / tr) as usize;
                let sim = InflowSim::new(acq.clone(), Medium::csf());
                let flow = flow_from_components(&[sinusoid(comps_f, 12.0, acq.t_end() + 1.0)], acq.t_end() + 1.0, 2e-3);
                let outp = sim.run(&flow);
                let s = outp.normalized(0.0);
                let fs = 1.0 / tr;
                let fgrid: Vec<f64> = (1..=(0.5 * fs / 0.002) as usize).map(|i| i as f64 * 0.002).collect();
                let amp: Vec<f64> = fgrid.iter().map(|&f| tone_amplitude(&outp.t, &s, f)).collect();
                let (ip, ap) = amp.iter().enumerate().fold((0, 0.0), |b, (i, &a)| if a > b.1 { (i, a) } else { b });
                println!("{tag}: fs {fs} Hz, peak at {:.3} Hz (expected alias {:.3}), amplitude {:.3} %", fgrid[ip], alias_freq(comps_f, fs), ap * 100.0);
                write_npy(&out.join(format!("alias_{tag}_f.npy")), &[fgrid.len()], &fgrid);
                write_npy(&out.join(format!("alias_{tag}_amp.npy")), &[amp.len()], &amp);
                json.push(format!("\"{tag}\": {{\"peak_hz\": {:.4}, \"expected_hz\": {:.4}, \"amp\": {ap:.5}}}", fgrid[ip], alias_freq(comps_f, fs)));
            }
            summary = json.join(", ");
        }
        "sweeps" => {
            let mut json = vec![];
            let mut run = |tag: &str, vals: &[f64], make: &dyn Fn(f64) -> (Acq, Medium)| {
                let mut rows = vec![];
                for &x in vals {
                    let (mut acq, med) = make(x);
                    acq.n_scan = 60;
                    let sim = InflowSim::new(acq, med);
                    let mc = modulation(&sim, 0.93, 12.0, &o);
                    let mr = modulation(&sim, 0.23, 1.2, &o);
                    println!("{tag} {x:8.3}: cardiac(12 mm/s) {:.3} %   resp(1.2 mm/s) {:.3} %", mc * 100.0, mr * 100.0);
                    rows.extend([x, mc, mr]);
                }
                write_npy(&out.join(format!("sweep_{tag}.npy")), &[vals.len(), 3], &rows);
                json.push(format!("\"{tag}\": {:?}", rows));
            };
            run("dispersion_cm2_min", &[0.0, 2.0, 4.0, 6.0, 8.0], &|d| { let mut m = Medium::csf(); m.diffusion = d * 1e-4 / 60.0; (paper_acq(&o), m) });
            run("t1_s", &[3.5, 4.2, 4.738, 5.2], &|t| { let mut m = Medium::csf(); m.t1 = t; (paper_acq(&o), m) });
            run("flip_deg", &[50.0, 65.0, 80.0, 90.0], &|f| { let mut a = paper_acq(&o); a.flip = f.to_radians(); (a, Medium::csf()) });
            run("gap_mm", &[0.0, 0.5, 1.0, 2.0], &|g| { let mut a = paper_acq(&o); a.gap = g * 1e-3; (a, Medium::csf()) });
            run("order_0seq_1inter", &[0.0, 1.0], &|x| { let mut a = paper_acq(&o); a.order = if x > 0.5 { SliceOrder::Interleaved } else { SliceOrder::Sequential }; (a, Medium::csf()) });
            summary = json.join(", ");
        }
        "phantomfit" => {
            // the paper's phantom analysis: prescribed vs estimated velocity amplitude per component (2 min scans, known pump
            // frequency). Truth: Monte-Carlo engine with random-walk dispersion; estimator: continuum engine + quasi-stationary.
            let mut acq = paper_acq(&o);
            acq.n_sim = o.n_sim.min(4);
            acq.dz = o.dz.max(100e-6);
            acq.n_scan = 60;
            let t_end = acq.t_end() + 1.0;
            let med = Medium::csf();
            let sim = InflowSim::new(acq.clone(), med.clone());
            let mut json = vec![];
            for (name, f, vmax) in [("cardiac", 0.93, 20.0), ("respiratory", 0.23, 5.0), ("low", 0.077, 2.5)] {
                let amps: Vec<f64> = (0..6).map(|i| vmax * i as f64 / 5.0).collect();
                let mut est = vec![];
                for (i, &a) in amps.iter().enumerate() {
                    let clock = PhaseClock::constant(f, 0.0, t_end, 0.01);
                    let flow = flow_from_components(&[Component::new(clock.clone(), a * 1e-3, 0.5)], t_end, 2e-3);
                    let raw = lagrangian_signal(&acq, &med, &flow, 300_000, o.seed + i as u64);
                    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
                    let mut data: Vec<f64> = raw.iter().map(|x| x / mean).collect();
                    if o.noise > 0.0 {
                        add_noise(&mut data, o.noise, o.seed + 100 + i as u64);
                    }
                    let mut quiet = |_: &str| {};
                    let fit = fit::fit_quasi(&sim, &[clock], &data, &[(a.max(0.5)) * 1e-3], 8, &mut quiet);
                    println!("{name} true {a:.2} mm/s → est {:.2} mm/s (R² {:.3})", fit.v[0] * 1e3, fit.r2);
                    est.push(fit.v[0] * 1e3);
                }
                write_npy(&out.join(format!("pf_{name}_true.npy")), &[6], &amps);
                write_npy(&out.join(format!("pf_{name}_est.npy")), &[6], &est);
                json.push(format!("\"{name}\": {{\"true\": {:?}, \"est\": {:?}}}", amps, est));
            }
            summary = json.join(", ");
        }
        "mismatch" => {
            // How far do the estimated velocity amplitudes move when the estimator's model or the data deviate from the nominal
            // case? Truth: Monte-Carlo engine, physiology clocks with variability, non-sinusoidal cardiac waveform, 2 % noise.
            // Every variant is warm-started from the nominal fit (joint LM only), so the table is a local sensitivity.
            let mut acq = paper_acq(&o);
            acq.n_sim = o.n_sim.min(4);
            acq.dz = o.dz.max(100e-6);
            let t_end = acq.t_end() + 1.0;
            let ecg = Ecg::new(t_end + 5.0, 1.0 / 0.93, 0.05, o.seed);
            let cc = PhaseClock::cardiac(&ecg, t_end, 0.01);
            let rc = PhaseClock::wandering(0.23, 0.15, 20.0, t_end, 0.05, o.seed + 1);
            let sc = PhaseClock::wandering(0.05, 0.4, 30.0, t_end, 0.05, o.seed + 2);
            let clocks = vec![cc.clone(), rc.clone(), sc.clone()];
            let truth_v = [12.0, 1.2, 0.18];
            let truth_d = [0.7, 2.1, 4.0];
            let med = Medium::csf();
            let mut comps: Vec<Component> = (0..3).map(|k| Component::new(clocks[k].clone(), truth_v[k] * 1e-3, truth_d[k])).collect();
            comps[0].harmonics = vec![(0.5, 1.0), (0.25, 2.0), (0.12, 3.0)];
            let flow = flow_from_components(&comps, t_end, 2e-3);
            let normed = |raw: &[f64]| -> Vec<f64> { let m = raw.iter().sum::<f64>() / raw.len() as f64; raw.iter().map(|x| x / m).collect() };
            let clean = normed(&lagrangian_signal(&acq, &med, &flow, 600_000, o.seed + 10));
            let noise_sigma = if o.noise > 0.0 { o.noise } else { 0.02 };
            let noisy = |seed: u64, base: &[f64]| -> Vec<f64> { let mut d = base.to_vec(); add_noise(&mut d, noise_sigma, seed); d };
            let sim0 = InflowSim::new(acq.clone(), med.clone());
            let mut log = |s: &str| println!("  {s}");
            let data0 = noisy(o.seed + 20, &clean);
            let base = fit::fit_quasi(&sim0, &clocks, &data0, &[8e-3, 1e-3, 0.3e-3], 8, &mut log);
            let p0: Vec<f64> = base.v.iter().cloned().chain(base.delta.iter().cloned()).collect();
            let mut names: Vec<String> = vec![];
            let mut rows: Vec<f64> = vec![];
            let mut record = |name: &str, f: &fit::FitResult| {
                println!("{name:34} v {:6.2} {:5.2} {:5.2} mm/s   δ {:5.2} {:5.2} {:5.2}   R² {:.3}", f.v[0] * 1e3, f.v[1] * 1e3, f.v[2] * 1e3, f.delta[0], f.delta[1], f.delta[2], f.r2);
                names.push(name.to_string());
                rows.extend(f.v.iter().map(|x| x * 1e3));
                rows.extend(f.delta.iter().cloned());
                rows.push(f.r2);
            };
            record("nominal", &base);
            // statistical scatter: same truth, new noise
            for k in 1..=5u64 {
                let d = noisy(o.seed + 20 + k, &clean);
                record(&format!("noise seed {k}"), &fit::fit_from(&sim0, &clocks, &d, &p0, 8));
            }
            // estimator model mismatch (data unchanged)
            let variants: Vec<(String, Acq, Medium)> = {
                let mut v = vec![];
                for d in [3.0, 9.0] { let mut m = med.clone(); m.diffusion = d * 1e-4 / 60.0; v.push((format!("estimator D = {d} cm²/min (truth 6)"), acq.clone(), m)); }
                for t1 in [4.2, 5.3] { let mut m = med.clone(); m.t1 = t1; v.push((format!("estimator T1 = {t1} s (truth 4.74)"), acq.clone(), m)); }
                for fl in [70.0, 90.0] { let mut a = acq.clone(); a.flip = f64::to_radians(fl); v.push((format!("estimator flip = {fl}° (truth 80)"), a, med.clone())); }
                let mut a = acq.clone(); a.gap = 1.0e-3; v.push(("estimator gap = 1.0 mm (truth 0.5)".into(), a, med.clone()));
                v
            };
            for (name, a, m) in variants {
                let sim = InflowSim::new(a, m);
                record(&name, &fit::fit_from(&sim, &clocks, &data0, &p0, 8));
            }
            // timing errors of the recorded physiology (estimator clocks shifted / jittered, data unchanged)
            for (name, tau) in [("cardiac clock late 0.10 s", [0.10, 0.0, 0.0]), ("cardiac clock late 0.25 s", [0.25, 0.0, 0.0]), ("respiratory clock late 1.0 s", [0.0, 1.0, 0.0])] {
                let cl: Vec<PhaseClock> = (0..3).map(|k| clocks[k].shifted(tau[k])).collect();
                record(name, &fit::fit_from(&sim0, &cl, &data0, &p0, 8));
            }
            for jit in [0.03, 0.06] {
                let mut rng = Rng::new(o.seed + 300);
                let mut e2 = ecg.clone();
                for r in e2.r_peaks.iter_mut() { *r += jit * rng.normal(); }
                e2.r_peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cl = vec![PhaseClock::cardiac(&e2, t_end, 0.01), rc.clone(), sc.clone()];
                record(&format!("R-peak jitter {:.0} ms", jit * 1e3), &fit::fit_from(&sim0, &cl, &data0, &p0, 8));
            }
            // slow confounds added to the data
            let t_rec: Vec<f64> = (0..acq.n_scan).map(|i| (acq.n_dummy + i) as f64 * acq.tr).collect();
            for a in [0.0025, 0.005, 0.01] {
                let mut d = data0.clone();
                for (x, &t) in d.iter_mut().zip(&t_rec) { *x += a * (sc.at(t) + 1.0).cos(); }
                record(&format!("slow confound locked to regressor {:.2} %", a * 100.0), &fit::fit_from(&sim0, &clocks, &d, &p0, 8));
            }
            {
                // band-limited (0.01–0.1 Hz) noise not tied to any regressor, 1 % rms
                let mut rng = Rng::new(o.seed + 400);
                let tones: Vec<(f64, f64)> = (0..40).map(|_| (0.01 + 0.09 * rng.uniform(), 2.0 * std::f64::consts::PI * rng.uniform())).collect();
                let series: Vec<f64> = t_rec.iter().map(|&t| tones.iter().map(|&(f, p)| (2.0 * std::f64::consts::PI * f * t + p).cos()).sum::<f64>()).collect();
                let rms = (series.iter().map(|x| x * x).sum::<f64>() / series.len() as f64).sqrt();
                let mut d = data0.clone();
                for (x, y) in d.iter_mut().zip(&series) { *x += 0.01 * y / rms; }
                record("slow noise 1 % rms, unlocked", &fit::fit_from(&sim0, &clocks, &d, &p0, 8));
            }
            // non-plug truth: mean velocity prescribed, profile across the lumen
            for (name, m) in [("laminar profile truth (m=2)", 2.0), ("blunt profile truth (m=6)", 6.0)] {
                let fac = radial_factors(m, 600_000, o.seed + 50);
                let d = noisy(o.seed + 20, &normed(&lagrangian_signal_with(&acq, &med, &flow, 600_000, o.seed + 10, Some(&fac))));
                record(name, &fit::fit_from(&sim0, &clocks, &d, &p0, 8));
            }
            write_npy(&out.join("mismatch_table.npy"), &[names.len(), 7], &rows);
            write_npy(&out.join("mismatch_truth.npy"), &[6], &[truth_v[0], truth_v[1], truth_v[2], truth_d[0], truth_d[1], truth_d[2]]);
            summary = format!("\"names\": {:?}, \"truth_v_mm_s\": {:?}, \"truth_delta\": {:?}, \"noise_sigma\": {noise_sigma}", names, truth_v, truth_d);
        }
        "roundtrip" => {
            // truth: Monte-Carlo engine (random-walk dispersion), physiology clocks with variability, non-sinusoidal cardiac waveform
            // estimator: continuum engine + quasi-stationary sinusoids of the same clocks (the paper's model)
            let mut acq = paper_acq(&o);
            acq.n_sim = o.n_sim.min(4);
            acq.dz = o.dz.max(100e-6);
            let t_end = acq.t_end() + 1.0;
            let ecg = Ecg::new(t_end + 5.0, 1.0 / 0.93, 0.05, o.seed);
            let cc = PhaseClock::cardiac(&ecg, t_end, 0.01);
            let rc = PhaseClock::wandering(0.23, 0.15, 20.0, t_end, 0.05, o.seed + 1);
            let sc = PhaseClock::wandering(0.05, 0.4, 30.0, t_end, 0.05, o.seed + 2);
            let truth_v = [12.0, 1.2, 0.18]; // paper medians, mm/s
            let truth_d = [0.7, 2.1, 4.0];
            let med = Medium::csf();
            let mut rt_json: Vec<String> = vec![];
            let clocks = vec![cc.clone(), rc.clone(), sc.clone()];
            for (tag, harmonics) in [("sinusoidal_truth", false), ("nonsinusoidal_truth", true)] {
                let mut comps: Vec<Component> = (0..3).map(|k| Component::new(clocks[k].clone(), truth_v[k] * 1e-3, truth_d[k])).collect();
                if harmonics {
                    comps[0].harmonics = vec![(0.5, 1.0), (0.25, 2.0), (0.12, 3.0)];
                }
                let flow = flow_from_components(&comps, t_end, 2e-3);
                let raw = lagrangian_signal(&acq, &med, &flow, 600_000, o.seed + 10);
                let mean = raw.iter().sum::<f64>() / raw.len() as f64;
                let mut data: Vec<f64> = raw.iter().map(|x| x / mean).collect();
                if o.noise > 0.0 {
                    add_noise(&mut data, o.noise, o.seed + 20);
                }
                let sim = InflowSim::new(acq.clone(), med.clone());
                let mut log = |s: &str| println!("[{tag}] {s}");
                let fit = fit::fit_quasi(&sim, &clocks, &data, &[8e-3, 1e-3, 0.3e-3], 8, &mut log);
                println!("[{tag}] R² {:.4}  v fit {:?} mm/s  (truth {:?}, fundamental)  δ fit {:?}", fit.r2, fit.v.iter().map(|x| (x * 1e5).round() / 100.0).collect::<Vec<_>>(), truth_v, fit.delta.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
                write_npy(&out.join(format!("rt_{tag}_data.npy")), &[data.len()], &data);
                write_npy(&out.join(format!("rt_{tag}_model.npy")), &[data.len()], &fit.model);
                rt_json.push(format!("\"{tag}\": {{\"r2\": {:.4}, \"v_fit_mm_s\": {:?}, \"v_truth_mm_s\": {:?}, \"delta_fit\": {:?}, \"delta_truth\": {:?}}}", fit.r2, fit.v.iter().map(|x| x * 1e3).collect::<Vec<_>>(), truth_v, fit.delta, truth_d));
            }
            summary = rt_json.join(", ");
        }
        other => {
            eprintln!("unknown command {other}");
            std::process::exit(2);
        }
    }
    std::fs::write(out.join("summary.json"), format!("{{\"cmd\": \"{cmd}\", \"seconds\": {:.1}, {summary}}}\n", t0.elapsed().as_secs_f64())).unwrap();
    println!("done in {:.1} s → {}", t0.elapsed().as_secs_f64(), out.display());
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    sxy / (sxx * syy).sqrt()
}
