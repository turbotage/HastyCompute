//! B0 / R2* / fat estimation: model convention, profile initialisation, statistics (efficiency, scaling invariance).

use hasty_qfit::fit::{fit_cpu, Data};
use hasty_qfit::lm::{LmOptions, Status};
use hasty_qfit::stats::crlb;
use hasty_qfit::waterfat::{normalize, FatSpectrum, Grid, Profile, Species, WaterFat, WfConfig};
use hasty_qfit::Model;
use num_complex::Complex64 as C;

const TAU: f64 = 2.0 * std::f64::consts::PI;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (TAU * b).cos()
    }
}

fn echoes() -> Vec<f64> {
    (0..12).map(|i| 1.2e-3 + i as f64 * 1.0e-3).collect()
}

/// [wr, wi, fr, fi, r2, om] → parameters of `species`
fn params(sp: Species, ff: f64, ph: f64, r2: f64, om: f64) -> Vec<f64> {
    let (w, f) = (C::from_polar(1.0 - ff, ph), C::from_polar(ff, ph));
    match sp {
        Species::Water => vec![w.re, w.im, r2, om],
        Species::WaterFat => vec![w.re, w.im, f.re, f.im, r2, om],
        Species::WaterFatCommonPhase => vec![1.0 - ff, ff, ph, r2, om],
        Species::WaterFatDualR2 => vec![w.re, w.im, f.re, f.im, r2, r2 * 1.3, om],
    }
}

/// noisy data [2][nd][n] for per-voxel truths [np][n]; complex Gaussian noise with σ per component
fn data(wf: &WaterFat, truth: &[f64], n: usize, sigma: f64, rng: &mut Rng) -> Vec<f64> {
    let (np, nd) = (wf.n_param(), wf.n_echo());
    let mut y = vec![0.0; 2 * nd * n];
    for v in 0..n {
        let p: Vec<f64> = (0..np).map(|j| truth[j * n + v]).collect();
        for (i, s) in wf.signal(&p).iter().enumerate() {
            y[i * n + v] = s.re + sigma * rng.gauss();
            y[(nd + i) * n + v] = s.im + sigma * rng.gauss();
        }
    }
    y
}

fn ff_of(sp: Species, p: &[f64]) -> f64 {
    let (w, f) = match sp {
        Species::Water => return 0.0,
        Species::WaterFatCommonPhase => (p[0].abs(), p[1].abs()),
        _ => (C::new(p[0], p[1]).norm(), C::new(p[2], p[3]).norm()),
    };
    f / (w + f).max(1e-30)
}

#[test]
fn model_matches_simulator_convention() {
    // simulator: each source contributes a e^{−R2* t} e^{−iωt}; fat = one source per peak at ω + ω_p, amplitude ρ_f α_p
    let fat = FatSpectrum::six_peak();
    let te = echoes();
    let b0 = 3.0;
    let wp = fat.omega(b0);
    let sim = |rw: C, rf: C, r2w: f64, r2f: f64, om: f64, t: f64| -> C {
        let mut s = rw * C::from_polar((-r2w * t).exp(), -om * t);
        for (a, w) in fat.amp.iter().zip(&wp) {
            s += rf * *a * C::from_polar((-r2f * t).exp(), -(om + w) * t);
        }
        s
    };
    let mut rng = Rng(5);
    for sp in [Species::Water, Species::WaterFat, Species::WaterFatCommonPhase, Species::WaterFatDualR2] {
        let wf = WaterFat::new(sp, te.clone(), b0, fat.clone());
        let vars = wf.vars();
        let nd = te.len();
        for _ in 0..10 {
            let p = params(sp, if sp == Species::Water { 0.0 } else { rng.u() }, TAU * rng.u(), 10.0 + 90.0 * rng.u(), TAU * (600.0 * rng.u() - 300.0));
            let direct = wf.signal(&p);
            for i in 0..nd {
                let t = te[i];
                let want = match sp {
                    Species::Water => sim(C::new(p[0], p[1]), C::new(0.0, 0.0), p[2], p[2], p[3], t),
                    Species::WaterFat => sim(C::new(p[0], p[1]), C::new(p[2], p[3]), p[4], p[4], p[5], t),
                    Species::WaterFatCommonPhase => sim(C::from_polar(p[0], p[2]), C::from_polar(p[1], p[2]), p[3], p[3], p[4], t),
                    Species::WaterFatDualR2 => sim(C::new(p[0], p[1]), C::new(p[2], p[3]), p[4], p[5], p[6], t),
                };
                let v = [vars[i], vars[nd + i], vars[2 * nd + i]];
                let sym = C::new(wf.model.graph.eval(wf.model.outputs[0], &p, &v, &[]), wf.model.graph.eval(wf.model.outputs[1], &p, &v, &[]));
                assert!((sym - want).norm() < 1e-12, "{sp:?} symbolic {sym} vs simulator {want}");
                assert!((direct[i] - want).norm() < 1e-12, "{sp:?} direct {} vs simulator {want}", direct[i]);
            }
        }
    }
    // 3 T: main methylene peak −3.40 ppm ≈ −434 Hz
    let f_main = fat.omega(3.0)[1] / TAU;
    assert!((f_main + 434.2).abs() < 0.5, "{f_main}");
}

/// one batch of noisy voxels with random truths over the clinically relevant ranges
fn random_batch(sp: Species, n: usize, sigma: f64, seed: u64) -> (WaterFat, Vec<f64>, Vec<f64>) {
    let wf = WaterFat::new(sp, echoes(), 3.0, FatSpectrum::six_peak());
    let np = wf.n_param();
    let mut rng = Rng(seed);
    let mut truth = vec![0.0; np * n];
    for v in 0..n {
        let ff = if sp == Species::Water { 0.0 } else { rng.u() };
        let p = params(sp, ff, TAU * rng.u(), 10.0 + 70.0 * rng.u(), TAU * (700.0 * rng.u() - 350.0));
        for j in 0..np {
            truth[j * n + v] = p[j];
        }
    }
    let y = data(&wf, &truth, n, sigma, &mut rng);
    (wf, truth, y)
}

/// fraction of voxels in the right basin: Δf and ΔFF within 5 CRLB standard deviations (at the truth, noise σ) and
/// Δf < 0.4 × the 434 Hz fat shift (a swap/wrap is a jump of about that size)
fn basin_rate(wf: &WaterFat, truth: &[f64], est: &[f64], n: usize, sigma: f64) -> f64 {
    let np = wf.n_param();
    let io = np - 1;
    let vars = wf.vars();
    let mut ok = 0;
    for v in 0..n {
        let t: Vec<f64> = (0..np).map(|j| truth[j * n + v]).collect();
        let e: Vec<f64> = (0..np).map(|j| est[j * n + v]).collect();
        let c = crlb(&wf.model, &t, &vars, wf.n_echo(), &[], None, sigma * sigma).unwrap();
        let sd_om = c[io * np + io].sqrt();
        // FF standard deviation by the delta method on the amplitudes (≤ the larger amplitude sd)
        let sd_amp = sp_amp_sd(wf.species, &c, np);
        let dom = (t[io] - e[io]).abs();
        let dff = (ff_of(wf.species, &t) - ff_of(wf.species, &e)).abs();
        let amp = match wf.species {
            Species::Water => 1.0,
            Species::WaterFatCommonPhase => t[0].abs() + t[1].abs(),
            _ => C::new(t[0], t[1]).norm() + C::new(t[2], t[3]).norm(),
        };
        if dom < (5.0 * sd_om).min(0.4 * TAU * 434.0) && dff < 5.0 * sd_amp / amp + 1e-9 {
            ok += 1;
        }
    }
    ok as f64 / n as f64
}

fn sp_amp_sd(sp: Species, c: &[f64], np: usize) -> f64 {
    sp.amp_idx().iter().map(|&j| c[j * np + j].sqrt()).fold(0.0, f64::max) * 2.0
}

#[test]
fn profile_starts_find_the_right_basin() {
    let n = 4000;
    for (sp, sigma) in [(Species::WaterFat, 0.02), (Species::WaterFatCommonPhase, 0.02), (Species::Water, 0.02), (Species::WaterFat, 0.1), (Species::WaterFatCommonPhase, 0.1)] {
        let (wf, truth, y) = random_batch(sp, n, sigma, 17);
        println!("SNR {}:", 1.0 / sigma);
        for n_start in [1, 3] {
            let cfg = WfConfig { n_start, ..WfConfig::default() };
            let t0 = std::time::Instant::now();
            let r = wf.fit_cpu(&y, n, &cfg);
            let el = t0.elapsed().as_secs_f64();
            let rate = basin_rate(&wf, &truth, &r.params, n, sigma);
            let conv = r.status.iter().filter(|s| s.converged()).count();
            let it = r.iters.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            let mut hist = std::collections::BTreeMap::new();
            for s in &r.status {
                *hist.entry(format!("{s:?}")).or_insert(0) += 1;
            }
            println!("{sp:?} starts {n_start}: right basin {:.2} %, converged {conv}/{n}, iters {it:.1} (winning start), {el:.2} s CPU, {hist:?}", 100.0 * rate);
            // misses: did the multi-start lose the global per-voxel optimum (a start at the truth finds a lower
            // objective), or is the per-voxel MLE itself in the other basin (only spatial priors can fix that)?
            let (yn, cn) = normalize(&y, wf.n_echo(), n);
            let vars = wf.vars();
            let d = Data { n_prob: n, n_data: wf.n_echo(), vars: &vars[..], fixed: &[][..], y: &yn[..], w: None };
            let (lo, hi) = wf.bounds(&cfg);
            let mut pt: Vec<f64> = truth.clone();
            for &j in sp.amp_idx() {
                for v in 0..n {
                    pt[j * n + v] /= cn[v];
                }
            }
            let ot = fit_cpu(&wf.model, &d, &mut pt, &lo, &hi, &wf.options(&cfg));
            let lost = (0..n).filter(|&v| ot[v].f * cn[v] * cn[v] < r.f[v] * (1.0 - 1e-9) - 1e-12).count();
            println!("    lost global per-voxel optimum (truth start lower): {lost}/{n}; wrong-basin MLE: {:.2} %", 100.0 * (1.0 - rate) - 100.0 * lost as f64 / n as f64);
            if n_start == 3 {
                assert!(lost as f64 <= 0.002 * n as f64, "{sp:?}: multi-start lost the optimum in {lost} voxels");
            }
            assert_eq!(conv, n, "{sp:?}: not all converged");
            if n_start == 3 {
                assert!(rate >= if sigma <= 0.02 { 0.995 } else { 0.95 }, "{sp:?}: only {:.2} % in the right basin", 100.0 * rate);
            }
        }
    }
}

/// sample mean / covariance of the estimates vs truth and the CRLB, for many noise realisations of one voxel
#[test]
fn mle_is_efficient_reaches_crlb() {
    let n: usize = std::env::var("QN").map_or(4000, |x| x.parse().unwrap());
    let sigma: f64 = std::env::var("QS").map_or(0.02, |x| x.parse().unwrap());
    let seed: u64 = std::env::var("QSEED").map_or(23, |x| x.parse().unwrap());
    for sp in [Species::WaterFat, Species::WaterFatCommonPhase] {
        let wf = WaterFat::new(sp, echoes(), 3.0, FatSpectrum::six_peak());
        let np = wf.n_param();
        let p0 = params(sp, 0.3, 0.5, 40.0, TAU * 60.0);
        let truth: Vec<f64> = (0..np).flat_map(|j| vec![p0[j]; n]).collect();
        let y = data(&wf, &truth, n, sigma, &mut Rng(seed));
        let r = wf.fit_cpu(&y, n, &WfConfig::default());
        let cov = crlb(&wf.model, &p0, &wf.vars(), wf.n_echo(), &[], None, sigma * sigma).unwrap();
        println!("{sp:?}: parameter, truth, bias / SE(bias), sample var / CRLB");
        for j in 0..np {
            let e: Vec<f64> = (0..n).map(|v| r.params[j * n + v]).collect();
            let mean = e.iter().sum::<f64>() / n as f64;
            let var = e.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let bound = cov[j * np + j];
            let z = (mean - p0[j]) / (bound / n as f64).sqrt();
            println!("  {:>3} {:>10.4} {:>7.2} {:>7.3}", sp.names()[j], p0[j], z, var / bound);
            // efficiency: the variance reaches the bound (±10 % ≈ 3σ of the variance estimate for n = 4000)
            assert!((var / bound - 1.0).abs() < 0.10, "{sp:?} {}: var / CRLB = {}", sp.names()[j], var / bound);
            // no bias beyond the O(σ²) MLE bias
            assert!(z.abs() < 4.5, "{sp:?} {}: bias {z} SE", sp.names()[j]);
        }
    }
}

/// replace whole identifiers in an expression
fn subst(e: &str, map: &[(&str, String)]) -> String {
    let mut out = String::new();
    let cs: Vec<char> = e.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_ascii_alphabetic() || cs[i] == '_' {
            let st = i;
            while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            let id: String = cs[st..i].iter().collect();
            match map.iter().find(|(k, _)| *k == id) {
                Some((_, v)) => out.push_str(v),
                None => out.push_str(&id),
            }
        } else {
            out.push(cs[i]);
            i += 1;
        }
    }
    out
}

/// the solver in scaled variables is exactly the solver on the reparametrised problem u = p / S with S = 1:
/// same iterations, same damping, same answer
#[test]
fn scaling_is_a_reparametrisation() {
    let n = 2000;
    let sp = Species::WaterFat;
    let (wf, _truth, y) = random_batch(sp, n, 0.02, 31);
    let cfg = WfConfig::default();
    let (yn, _) = normalize(&y, wf.n_echo(), n);
    let prof = Profile::new(&wf, &Grid::for_echoes(&wf.te, cfg.om_max, cfg.grid_r2.clone()), None);
    let cand = prof.candidates_cpu(&yn, n, 1);
    let p0 = wf.starts(&prof, &cand, 0);
    let vars = wf.vars();
    let d = Data { n_prob: n, n_data: wf.n_echo(), vars: &vars[..], fixed: &[][..], y: &yn[..], w: None };
    let (lo, hi) = wf.bounds(&cfg);
    let np = wf.n_param();
    // deliberately odd scales
    let s: Vec<f64> = vec![0.3, 7.0, 0.05, 2.0, 123.0, 4321.0];
    let o_a = LmOptions { scale: Some(s.clone()), ..LmOptions::default() };
    let mut pa = p0.clone();
    let oa = fit_cpu(&wf.model, &d, &mut pa, &lo, &hi, &o_a);
    // reparametrised model in u, solved with unit scale
    let names = sp.names();
    let map: Vec<(&str, String)> = names.iter().zip(&s).map(|(nm, sj)| (*nm, format!("({sj:e} * {nm})"))).collect();
    let ex = WaterFat::output_exprs(sp);
    let mu = Model::new(names, &["t", "cr", "ci"], &[], &[&subst(&ex[0], &map), &subst(&ex[1], &map)]).unwrap();
    let mut pb: Vec<f64> = (0..np * n).map(|i| p0[i] / s[i / n]).collect();
    let lob: Vec<f64> = lo.iter().zip(&s).map(|(a, b)| a / b).collect();
    let hib: Vec<f64> = hi.iter().zip(&s).map(|(a, b)| a / b).collect();
    let o_b = LmOptions { scale: Some(vec![1.0; np]), ..LmOptions::default() };
    let ob = fit_cpu(&mu, &d, &mut pb, &lob, &hib, &o_b);
    let same_it = oa.iter().zip(&ob).filter(|(a, b)| a.iters == b.iters && a.status == b.status).count();
    let mut dmax: f64 = 0.0;
    for i in 0..np * n {
        let j = i / n;
        dmax = dmax.max((pa[i] - pb[i] * s[j]).abs() / (s[j] * 1e-3));
    }
    println!("scaled vs reparametrised: identical iteration counts and status {same_it}/{n}, max |Δp| = {dmax:.2e} × 1e-3 scale");
    assert!(same_it as f64 >= 0.99 * n as f64);
    assert!(dmax < 1e-4, "{dmax}");
}

/// whatever the scaling (parameters, data units, weight units), the estimate is the same MLE up to the
/// statistical stopping tolerance, and the statistical stop sits within tol_stat standard errors of the tight fit
#[test]
fn scaling_does_not_change_the_mle() {
    let n = 2000;
    let sp = Species::WaterFat;
    let sigma = 0.02;
    let (wf, _truth, y) = random_batch(sp, n, sigma, 41);
    let np = wf.n_param();
    let cfg = WfConfig::default();
    let reference = wf.fit_cpu(&y, n, &WfConfig { lm: LmOptions { tol_stat: 0.0, tol_step: 1e-14, tol_fun: 1e-16, max_iter: 500, ..LmOptions::default() }, ..cfg.clone() });
    // per-voxel standard errors (CRLB at the reference estimate)
    let vars = wf.vars();
    let se: Vec<Vec<f64>> = (0..n)
        .map(|v| {
            let p: Vec<f64> = (0..np).map(|j| reference.params[j * n + v]).collect();
            let c = crlb(&wf.model, &p, &vars, wf.n_echo(), &[], None, sigma * sigma).unwrap();
            (0..np).map(|j| c[j * np + j].sqrt()).collect()
        })
        .collect();
    let dist = |a: &[f64]| -> f64 {
        let mut m: f64 = 0.0;
        for v in 0..n {
            for j in 0..np {
                m = m.max((a[j * n + v] - reference.params[j * n + v]).abs() / se[v][j]);
            }
        }
        m
    };
    let mut rng = Rng(3);
    let variants: Vec<(&str, WfConfig, f64)> = vec![
        ("default scaling (stat stop 1e-4)", cfg.clone(), 1.0),
        ("unit scaling", WfConfig { lm: LmOptions { scale: Some(vec![1.0; np]), ..cfg.lm.clone() }, ..cfg.clone() }, 1.0),
        ("data × 1e4", cfg.clone(), 1e4),
        ("data × 1e-6", cfg.clone(), 1e-6),
    ];
    for (name, c, ys) in variants {
        let yy: Vec<f64> = y.iter().map(|x| x * ys).collect();
        let r = if name == "unit scaling" {
            // bypass WaterFat::options (which sets the scale) to run with S = 1
            let (yn, cn) = normalize(&yy, wf.n_echo(), n);
            let prof = Profile::new(&wf, &Grid::for_echoes(&wf.te, c.om_max, c.grid_r2.clone()), None);
            let cand = prof.candidates_cpu(&yn, n, c.n_start);
            let d = Data { n_prob: n, n_data: wf.n_echo(), vars: &vars[..], fixed: &[][..], y: &yn[..], w: None };
            let (lo, hi) = wf.bounds(&c);
            let mut best = vec![0.0; np * n];
            let mut bf = vec![f64::INFINITY; n];
            for k in 0..c.n_start {
                let mut p = wf.starts(&prof, &cand, k);
                let out = fit_cpu(&wf.model, &d, &mut p, &lo, &hi, &c.lm);
                for v in 0..n {
                    if out[v].f < bf[v] {
                        bf[v] = out[v].f;
                        for j in 0..np {
                            best[j * n + v] = p[j * n + v];
                        }
                    }
                }
            }
            for &j in sp.amp_idx() {
                for v in 0..n {
                    best[j * n + v] *= cn[v];
                }
            }
            hasty_qfit::waterfat::canonicalize(sp, &mut best, n);
            best
        } else {
            wf.fit_cpu(&yy, n, &c).params
        };
        let back: Vec<f64> = r.iter().enumerate().map(|(i, x)| if sp.amp_idx().contains(&(i / n)) { x / ys } else { *x }).collect();
        let dd = dist(&back);
        println!("{name:>34}: max |Δp| vs tight reference = {dd:.2e} standard errors");
        assert!(dd < 1e-3, "{name}: {dd} standard errors from the MLE");
    }
    // the statistical stop saves iterations without moving the estimate
    let it_ref = reference.iters.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let def = wf.fit_cpu(&y, n, &cfg);
    let it_def = def.iters.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let stat = def.status.iter().filter(|s| **s == Status::StatTol).count();
    println!("mean iterations: tight {it_ref:.1}, with tol_stat 1e-4 {it_def:.1} ({stat}/{n} stopped by StatTol)");
    let _ = rng.u();
}

#[test]
fn gpu_matches_cpu() {
    let Ok(dev) = hasty_cuda::Device::new(0) else {
        println!("no CUDA device; skipped");
        return;
    };
    use hasty_qfit::waterfat::gpu::WaterFatGpu;
    let n = 4000;
    let sigma = 0.02;
    for sp in [Species::WaterFat, Species::WaterFatCommonPhase] {
        let (wf, truth, y) = random_batch(sp, n, sigma, 77);
        let np = wf.n_param();
        let cfg = WfConfig::default();
        let rc = wf.fit_cpu(&y, n, &cfg);
        let vars = wf.vars();
        let se = |v: usize, j: usize| -> f64 {
            let p: Vec<f64> = (0..np).map(|q| rc.params[q * n + v]).collect();
            crlb(&wf.model, &p, &vars, wf.n_echo(), &[], None, sigma * sigma).map_or(1.0, |c| c[j * np + j].sqrt())
        };
        let g64 = WaterFatGpu::<f64>::new(&dev, &wf, &cfg).unwrap();
        let r64 = g64.fit(&y, n).unwrap();
        // the best profile minimum must agree; lower-ranked near-tied minima may swap order by rounding
        let same_node = (0..n).filter(|&i| rc.cand.node[i] == r64.cand.node[i]).count();
        let same_all = (0..n * cfg.n_start).filter(|&i| rc.cand.node[i] == r64.cand.node[i]).count();
        let mut d64: f64 = 0.0;
        for v in 0..n {
            for j in 0..np {
                d64 = d64.max((rc.params[j * n + v] - r64.params[j * n + v]).abs() / se(v, j));
            }
        }
        let g32 = WaterFatGpu::<f32>::new(&dev, &wf, &WfConfig { lm: LmOptions { tol_step: 1e-6, tol_fun: 1e-10, tol_grad: 1e-9, ..LmOptions::default() }, ..cfg.clone() }).unwrap();
        let t0 = std::time::Instant::now();
        let r32 = g32.fit(&y, n).unwrap();
        let el = t0.elapsed().as_secs_f64();
        let mut d32: f64 = 0.0;
        let mut far = 0;
        for v in 0..n {
            let mut m: f64 = 0.0;
            for j in 0..np {
                m = m.max((rc.params[j * n + v] - r32.params[j * n + v]).abs() / se(v, j));
            }
            if m > 0.05 {
                far += 1;
            }
            d32 = d32.max(m);
        }
        let rate = basin_rate(&wf, &truth, &r32.params, n, sigma);
        println!(
            "{sp:?}: GPU f64 same best candidate {same_node}/{n} (all ranks {same_all}/{}), max |Δp| vs CPU {d64:.2e} SE; GPU f32 {el:.3} s, right basin {:.2} %, max |Δp| vs CPU {d32:.2e} SE, {far} voxels > 0.05 SE",
            n * cfg.n_start,
            100.0 * rate
        );
        assert!(same_node as f64 >= 0.999 * n as f64);
        assert!(d64 < 1e-3, "f64 GPU vs CPU {d64}");
        assert!(far as f64 <= 0.002 * n as f64, "{far} f32 voxels differ by > 0.05 SE");
        assert!(rate >= 0.995);
    }
}

#[test]
#[ignore]
fn gpu_benchmark_million_voxels() {
    let dev = hasty_cuda::Device::new(0).unwrap();
    use hasty_qfit::waterfat::gpu::WaterFatGpu;
    let n = 1_000_000;
    let (wf, truth, y) = random_batch(Species::WaterFat, n, 0.02, 99);
    let cfg = WfConfig { lm: LmOptions { tol_step: 1e-6, tol_fun: 1e-10, tol_grad: 1e-9, ..LmOptions::default() }, ..WfConfig::default() };
    let g = WaterFatGpu::<f32>::new(&dev, &wf, &cfg).unwrap();
    let _ = g.fit(&y[..], n).unwrap();
    let t0 = std::time::Instant::now();
    let r = g.fit(&y, n).unwrap();
    let el = t0.elapsed().as_secs_f64();
    println!(
        "water–fat (6 params, 12 echoes, {} grid nodes, {} starts): {n} voxels in {el:.3} s on the GPU incl. transfers and normalisation; right basin {:.2} %",
        g.prof.om.len(),
        cfg.n_start,
        100.0 * basin_rate(&wf, &truth, &r.params, n, 0.02)
    );
}
