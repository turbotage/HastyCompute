use hasty_qfit::expr::Graph;
use hasty_qfit::gpu::GpuFit;
use hasty_qfit::model::hidx;
use hasty_qfit::{fit_cpu, Data, HessianMode, LambdaRule, LmOptions, ModChol, Model};

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn parser_and_constant_folding() {
    let mut g = Graph::new(&["a", "b"], &["t"], &["k"]);
    let cases = [
        ("a + b * t", 1.5 + 2.0 * 0.5),
        ("-a^2", -(1.5f64.powi(2))),
        ("a ** 3 / 2", 1.5f64.powi(3) / 2.0),
        ("pow(a, b)", 1.5f64.powf(2.0)),
        ("exp(-t * b) * k", (-0.5f64 * 2.0).exp() * 3.0),
        ("2.5e-1 * pi + sqrt(a) - abs(-b)", 0.25 * std::f64::consts::PI + 1.5f64.sqrt() - 2.0),
        ("atan(t) + tanh(a) + sign(-k)", 0.5f64.atan() + 1.5f64.tanh() - 1.0),
        ("(a - a) * exp(b) + 1", 1.0),
    ];
    for (s, want) in cases {
        let e = g.parse(s).unwrap();
        let got = g.eval(e, &[1.5, 2.0], &[0.5], &[3.0]);
        assert!((got - want).abs() < 1e-12, "{s}: {got} vs {want}");
    }
    assert!(g.parse("a +").is_err());
    assert!(g.parse("foo(a)").is_err());
    assert!(g.parse("c * a").is_err());
}

#[test]
fn symbolic_derivatives_match_finite_differences() {
    let m = Model::new(
        &["s0", "r2", "df", "ph"],
        &["te"],
        &["b1"],
        &["s0 * b1 * exp(-r2 * te) * cos(2 * pi * df * te + ph)", "s0 * exp(-r2 * te) * sin(2 * pi * df * te + ph) / (1 + te^2) + log(1 + s0^2) * atan(r2 * te)"],
    )
    .unwrap();
    let mut s = 3u64;
    for _ in 0..20 {
        let p = [0.5 + lcg(&mut s), 10.0 + 30.0 * lcg(&mut s), -50.0 + 100.0 * lcg(&mut s), lcg(&mut s) * 3.0];
        let v = [0.002 + 0.02 * lcg(&mut s)];
        let fx = [0.8 + 0.4 * lcg(&mut s)];
        for k in 0..m.n_out() {
            for j in 0..4 {
                let h = 1e-6 * p[j].abs().max(1.0);
                let (mut pp, mut pm) = (p, p);
                pp[j] += h;
                pm[j] -= h;
                let fd = (m.graph.eval(m.outputs[k], &pp, &v, &fx) - m.graph.eval(m.outputs[k], &pm, &v, &fx)) / (2.0 * h);
                let an = m.graph.eval(m.jac[k][j], &p, &v, &fx);
                assert!((fd - an).abs() <= 1e-6 * an.abs().max(1.0), "J[{k}][{j}] {an} vs fd {fd}");
                for i in 0..=j {
                    let hi = 1e-5 * p[i].abs().max(1.0);
                    let (mut qp, mut qm) = (p, p);
                    qp[i] += hi;
                    qm[i] -= hi;
                    let fd2 = (m.graph.eval(m.jac[k][j], &qp, &v, &fx) - m.graph.eval(m.jac[k][j], &qm, &v, &fx)) / (2.0 * hi);
                    let an2 = m.graph.eval(m.hes[k][hidx(i, j)], &p, &v, &fx);
                    assert!((fd2 - an2).abs() <= 1e-5 * an2.abs().max(1.0), "H[{k}][{i},{j}] {an2} vs fd {fd2}");
                }
            }
        }
    }
    // the straight-line programs agree with the graph evaluation
    let p = [1.1, 20.0, 12.0, 0.3];
    let (v, fx) = ([0.007], [0.9]);
    let mut slots = vec![0.0; m.prog_full.ops.len()];
    m.prog_full.run(&p, &v, &fx, &mut slots);
    let n = m.n_param();
    for k in 0..m.n_out() {
        assert!((slots[m.prog_full.outputs[k] as usize] - m.graph.eval(m.outputs[k], &p, &v, &fx)).abs() < 1e-12);
        for j in 0..n {
            assert!((slots[m.prog_full.outputs[m.n_out() + k * n + j] as usize] - m.graph.eval(m.jac[k][j], &p, &v, &fx)).abs() < 1e-9);
        }
    }
    println!("program sizes: value {} ops, Gauss–Newton {} ops, full {} ops", m.prog_val.ops.len(), m.prog_gn.ops.len(), m.prog_full.ops.len());
}

/// a batch of synthetic problems: params [np][nprob], y [nout][ndata][nprob]
struct Batch {
    m: Model,
    nprob: usize,
    vars: Vec<f64>,
    fixed: Vec<f64>,
    y: Vec<f64>,
    truth: Vec<f64>,
    start: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

fn batch(m: Model, vars: Vec<f64>, nprob: usize, gen: &dyn Fn(&mut u64) -> (Vec<f64>, Vec<f64>), lo: Vec<f64>, hi: Vec<f64>, noise: f64) -> Batch {
    let np = m.n_param();
    let nd = vars.len() / m.n_var().max(1);
    let nout = m.n_out();
    let mut s = 11u64;
    let (mut truth, mut start, mut y) = (vec![0.0; np * nprob], vec![0.0; np * nprob], vec![0.0; nout * nd * nprob]);
    for k in 0..nprob {
        let (t, s0) = gen(&mut s);
        for j in 0..np {
            truth[j * nprob + k] = t[j];
            start[j * nprob + k] = s0[j];
        }
        for i in 0..nd {
            let v: Vec<f64> = (0..m.n_var()).map(|q| vars[q * nd + i]).collect();
            for o in 0..nout {
                let clean = m.graph.eval(m.outputs[o], &t, &v, &[]);
                let n = noise * (lcg(&mut s) + lcg(&mut s) + lcg(&mut s) - 1.5) * 2.0;
                y[(o * nd + i) * nprob + k] = clean + n;
            }
        }
    }
    Batch { m, nprob, vars, fixed: vec![], y, truth, start, lo, hi }
}

fn ivim() -> Batch {
    let m = Model::new(&["s0", "f", "d", "ds"], &["b"], &[], &["s0 * ((1 - f) * exp(-b * d) + f * exp(-b * ds))"]).unwrap();
    let b = vec![0.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0];
    batch(
        m,
        b,
        2000,
        &|s| {
            let t = vec![1.0, 0.05 + 0.25 * lcg(s), 0.0007 + 0.0015 * lcg(s), 0.01 + 0.04 * lcg(s)];
            (t, vec![0.9, 0.1, 0.001, 0.02])
        },
        vec![0.0, 0.0, 1e-5, 2e-3],
        vec![5.0, 1.0, 5e-3, 0.2],
        0.0,
    )
}

fn multiecho() -> Batch {
    // complex multi-echo signal: ρ e^{−R2* t} e^{i 2π Δf t}, ρ complex → outputs (re, im)
    let m = Model::new(
        &["rr", "ri", "r2", "df"],
        &["te"],
        &[],
        &["exp(-r2 * te) * (rr * cos(2 * pi * df * te) - ri * sin(2 * pi * df * te))", "exp(-r2 * te) * (rr * sin(2 * pi * df * te) + ri * cos(2 * pi * df * te))"],
    )
    .unwrap();
    let te: Vec<f64> = (0..12).map(|i| 1.5e-3 + i as f64 * 1.1e-3).collect();
    batch(
        m,
        te,
        2000,
        &|s| {
            let ph = std::f64::consts::TAU * lcg(s);
            let t = vec![ph.cos(), ph.sin(), 10.0 + 60.0 * lcg(s), -60.0 + 120.0 * lcg(s)];
            // start near the truth for Δf (the wrapped phase problem is non-convex; real use: unwrap / multi-start)
            (t.clone(), vec![t[0] * 0.8, t[1] * 0.8, 30.0, t[3] + 8.0 * (lcg(s) - 0.5)])
        },
        vec![-10.0, -10.0, 0.0, -300.0],
        vec![10.0, 10.0, 500.0, 300.0],
        0.0,
    )
}

fn check_cpu(b: &Batch, o: &LmOptions, tol: f64) -> f64 {
    let np = b.m.n_param();
    let d = Data { n_prob: b.nprob, n_data: b.vars.len() / b.m.n_var(), vars: &b.vars[..], fixed: &b.fixed[..], y: &b.y[..], w: None };
    let mut p = b.start.clone();
    let t0 = std::time::Instant::now();
    let out = fit_cpu(&b.m, &d, &mut p, &b.lo, &b.hi, o);
    let el = t0.elapsed().as_secs_f64();
    let conv = out.iter().filter(|x| x.status.converged()).count();
    let mut worst: f64 = 0.0;
    for k in 0..b.nprob {
        for j in 0..np {
            let (t, e) = (b.truth[j * b.nprob + k], p[j * b.nprob + k]);
            worst = worst.max((t - e).abs() / t.abs().max(1e-3));
        }
    }
    let it = out.iter().map(|x| x.iters as f64).sum::<f64>() / b.nprob as f64;
    let mut hist = std::collections::BTreeMap::new();
    for x in &out {
        *hist.entry(format!("{:?}", x.status)).or_insert(0) += 1;
    }
    let bad: Vec<usize> = (0..b.nprob).filter(|&k| (0..np).any(|j| (b.truth[j * b.nprob + k] - p[j * b.nprob + k]).abs() / b.truth[j * b.nprob + k].abs().max(1e-3) > tol)).collect();
    for &k in bad.iter().take(3) {
        println!("  bad problem {k}: truth {:?} fit {:?} status {:?} f {:.3e}", (0..np).map(|j| b.truth[j * b.nprob + k]).collect::<Vec<_>>(), (0..np).map(|j| p[j * b.nprob + k]).collect::<Vec<_>>(), out[k].status, out[k].f);
    }
    println!("  status {hist:?}");
    println!("CPU {:?} {:?}: {} problems in {el:.3} s, converged {conv}, mean iters {it:.1}, worst rel. error {worst:.2e}, {} off the truth", o.hessian, o.modchol, b.nprob, bad.len());
    // GMW81 over-modifies some Hessians (a known property), which can change the path into IVIM's swapped
    // (D ↔ D*) basin for a rare start; allow 0.1 % there, none otherwise
    let allowed = if o.modchol == ModChol::Gmw81 { b.nprob / 1000 } else { 0 };
    assert!(bad.len() <= allowed, "{} problems off the truth (worst relative error {worst})", bad.len());
    worst
}

#[test]
fn cpu_fits_noise_free() {
    for (name, b) in [("ivim", ivim()), ("multiecho", multiecho())] {
        for hessian in [HessianMode::Full, HessianMode::GaussNewton] {
            for modchol in [ModChol::Se99, ModChol::Gmw81] {
                let o = LmOptions { hessian, modchol, max_iter: 300, ..LmOptions::default() };
                println!("{name}:");
                check_cpu(&b, &o, if name == "ivim" { 1e-4 } else { 1e-6 });
            }
        }
        let o = LmOptions { rule: LambdaRule::Nielsen, max_iter: 300, ..LmOptions::default() };
        check_cpu(&b, &o, if name == "ivim" { 1e-4 } else { 1e-6 });
    }
}

#[test]
fn gpu_matches_cpu() {
    let Ok(dev) = hasty_cuda::Device::new(0) else {
        println!("no CUDA device; skipped");
        return;
    };
    for (name, b) in [("ivim", ivim()), ("multiecho", multiecho())] {
        for hessian in [HessianMode::Full, HessianMode::GaussNewton] {
            let o = LmOptions { hessian, max_iter: 300, ..LmOptions::default() };
            let nd = b.vars.len() / b.m.n_var();
            let d = Data { n_prob: b.nprob, n_data: nd, vars: &b.vars[..], fixed: &b.fixed[..], y: &b.y[..], w: None };
            let mut pc = b.start.clone();
            let oc = fit_cpu(&b.m, &d, &mut pc, &b.lo, &b.hi, &o);
            // f64 on the GPU: same algorithm, should agree closely
            let g64 = GpuFit::new::<f64>(&dev, &b.m, &o).unwrap();
            let mut pg = b.start.clone();
            let og = g64.fit(&d, &mut pg, &b.lo, &b.hi, &o).unwrap();
            let mut dmax: f64 = 0.0;
            for i in 0..pc.len() {
                dmax = dmax.max((pc[i] - pg[i]).abs() / pc[i].abs().max(1e-3));
            }
            let same_it = oc.iter().zip(&og.iters).filter(|(a, b)| a.iters as i32 == **b).count();
            println!("{name} {hessian:?}: GPU f64 vs CPU max rel diff {dmax:.2e}, identical iteration counts {same_it}/{}", b.nprob);
            assert!(dmax < 1e-6, "GPU f64 differs from CPU: {dmax}");
            // f32: tolerances scaled to single precision
            let o32 = LmOptions { tol_grad: 1e-7, tol_step: 1e-6, tol_fun: 1e-10, ..o.clone() };
            let g32 = GpuFit::new::<f32>(&dev, &b.m, &o32).unwrap();
            let y32: Vec<f32> = b.y.iter().map(|&x| x as f32).collect();
            let v32: Vec<f32> = b.vars.iter().map(|&x| x as f32).collect();
            let d32 = Data { n_prob: b.nprob, n_data: nd, vars: &v32[..], fixed: &[][..], y: &y32[..], w: None };
            let mut p32: Vec<f32> = b.start.iter().map(|&x| x as f32).collect();
            let lo32: Vec<f32> = b.lo.iter().map(|&x| x as f32).collect();
            let hi32: Vec<f32> = b.hi.iter().map(|&x| x as f32).collect();
            let t0 = std::time::Instant::now();
            let o3 = g32.fit(&d32, &mut p32, &lo32, &hi32, &o32).unwrap();
            let el = t0.elapsed().as_secs_f64();
            let mut worst: f64 = 0.0;
            for i in 0..p32.len() {
                worst = worst.max((p32[i] as f64 - b.truth[i]).abs() / b.truth[i].abs().max(1e-3));
            }
            let conv = o3.status.iter().filter(|s| s.converged()).count();
            println!("{name} {hessian:?}: GPU f32 {el:.3} s (incl. transfer), converged {conv}/{}, worst rel. error vs truth {worst:.2e}", b.nprob);
            assert!(worst < 2e-3, "f32 fit error {worst}");
        }
    }
}

#[test]
#[ignore]
fn gpu_benchmark_million_voxels() {
    let dev = hasty_cuda::Device::new(0).unwrap();
    let base = ivim();
    let reps = 500;
    let n = base.nprob * reps;
    let nd = base.vars.len();
    let mut y = vec![0f32; nd * n];
    let mut start = vec![0f32; 4 * n];
    for r in 0..reps {
        for k in 0..base.nprob {
            for i in 0..nd {
                y[i * n + r * base.nprob + k] = base.y[i * base.nprob + k] as f32;
            }
            for j in 0..4 {
                start[j * n + r * base.nprob + k] = base.start[j * base.nprob + k] as f32;
            }
        }
    }
    let v32: Vec<f32> = base.vars.iter().map(|&x| x as f32).collect();
    let lo: Vec<f32> = base.lo.iter().map(|&x| x as f32).collect();
    let hi: Vec<f32> = base.hi.iter().map(|&x| x as f32).collect();
    for hessian in [HessianMode::Full, HessianMode::GaussNewton] {
        let o = LmOptions { hessian, max_iter: 100, tol_grad: 1e-7, tol_step: 1e-6, tol_fun: 1e-10, ..LmOptions::default() };
        let g = GpuFit::new::<f32>(&dev, &base.m, &o).unwrap();
        let d = Data { n_prob: n, n_data: nd, vars: &v32[..], fixed: &[][..], y: &y[..], w: None };
        let mut p = start.clone();
        let _ = g.fit(&d, &mut p, &lo, &hi, &o).unwrap(); // warm-up (module load)
        let mut p = start.clone();
        let t0 = std::time::Instant::now();
        let out = g.fit(&d, &mut p, &lo, &hi, &o).unwrap();
        let el = t0.elapsed().as_secs_f64();
        let it = out.iters.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        println!("IVIM {hessian:?}: {n} voxels × {nd} b-values on the GPU in {el:.3} s (incl. transfers), mean iters {it:.1}");
    }
}
