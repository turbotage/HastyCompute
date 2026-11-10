//! Cardiac lattice-Boltzmann flow of a phantom (hasty-cfd; research/flow_phantom.md Tier B): the case from
//! `rustlib/tools/phantom/lbm_case.py`, the run into <out> (flow_bins.f32 / .json: phase-binned Eulerian velocity mean
//! and covariance, log.jsonl, ckpt.bin). Interruptible (Ctrl-C or a STOP file in <out>) and resumable (rerun).
//!
//!     cardiac_lbm <case_dir> <out_dir> [--cpu] [--plan] [--cycles n] [--warmup n] [--bins n] [--quiet]
//!
//! --plan: units, active tiles, nodes, memory and an estimate of the run time, without running (no GPU needed).
//! --cpu:  the CPU backend (the same C core; small cases and checks only).
//! --beats: also write the per-cycle beat library (<out>/beats/beat_<cycle>.bin: per occupied cell and phase bin the
//!         coverage, mean velocity, covariance, third central moments and wall shear stress, f16; see beats/index.json).
//! --probe <steps>[:every]: debugging, a fresh start (no checkpoint, no accumulation) for <steps> steps with the fastest
//!         nodes and speed percentiles every <every> steps (default steps / 20) into <out>/probe.jsonl, and sparse
//!         snapshots (hasty_cfd::probe::snapshot) at the steps of --snap s1,s2,… into <out>/snaps.
//! --spin [k=v,…]: the blood's spin history for an imaging protocol, the relative M_z s = M_z / M0 as a passive scalar
//!         (hasty_cfd case::SpinSpec; its cell mean and std go into the beat library as mz, mz_std). Keys (defaults: the
//!         simphantom thorax protocol, `--seq flow` of data/hasty/seq/flow_thorax): tr 11.29e-3 s, flip 12°, t1 the
//!         phantom's blood T1 (case.json "blood", its contrast state), center x;y;z 0.02;-0.03;1.31 m
//!         (lab), normal x;y;z 0;0;1, thk 0.16 m (slab), tbw 6, dur 0.8e-3 s (windowed-sinc slab pulse,
//!         hasty_mrsim::rf, Bloch profile). Inlets enter with s from their case "upstream" path (fresh, or the
//!         flux-weighted outflow of the "from" cuts carried through the path), else 1.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use hasty_cfd::case::Case;
use hasty_cfd::solver::{active_tiles, Solver, Units};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |a: &mut Vec<String>, n: &str| a.iter().position(|x| x == n).map(|i| { a.remove(i); true }).unwrap_or(false);
    let take = |a: &mut Vec<String>, n: &str| a.iter().position(|x| x == n).map(|i| { a.remove(i); a.remove(i) });
    let cpu = flag(&mut args, "--cpu");
    let plan = flag(&mut args, "--plan");
    let quiet = flag(&mut args, "--quiet");
    let beats = flag(&mut args, "--beats");
    let cycles = take(&mut args, "--cycles").map(|s| s.parse::<usize>().unwrap());
    let warmup = take(&mut args, "--warmup").map(|s| s.parse::<usize>().unwrap());
    let bins = take(&mut args, "--bins").map(|s| s.parse::<usize>().unwrap());
    let probe = take(&mut args, "--probe");
    let snaps: Vec<u64> = take(&mut args, "--snap").map_or(vec![], |s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let spin_arg = args.iter().position(|x| x == "--spin").map(|i| {
        args.remove(i);
        if i < args.len() && !args[i].starts_with("--") && args[i].contains('=') { args.remove(i) } else { String::new() }
    });
    if args.len() < 2 {
        eprintln!("usage: cardiac_lbm <case_dir> <out_dir> [--cpu] [--plan] [--cycles n] [--warmup n] [--bins n] [--beats] [--quiet]");
        std::process::exit(2);
    }
    let (case_dir, out) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]));
    let mut case = Case::load(&case_dir).unwrap_or_else(|e| panic!("{}: {e}", case_dir.display()));
    if let Some(c) = cycles {
        case.run.cycles = c;
    }
    if let Some(w) = warmup {
        case.run.warmup = w;
    }
    if let Some(b) = bins {
        case.run.bins = b;
    }
    if beats {
        case.run.beats = true;
    }
    if let Some(sa) = &spin_arg {
        // the phantom's blood T1 (its contrast state) from the case, unless given
        let cj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(case_dir.join("case.json")).unwrap()).unwrap();
        case.spin = Some(spin_spec(sa, case.cuts.len(), cj["blood"]["t1"].as_f64()));
    }
    let u = Units::new(&case);
    println!(
        "case {}: grid {:?} at {:.2} mm, period {:.3} s → dt {:.2} µs ({} steps / cycle), ν_lattice {:.2e}, τ0 {:.5}, {} valves, {} cuts",
        case_dir.display(),
        case.grid.dims,
        case.grid.dx * 1e3,
        u.period,
        u.dt * 1e6,
        u.steps_per_cycle,
        u.nu_lat,
        u.tau0,
        case.valves.len(),
        case.cuts.len()
    );
    if plan {
        let tiles = active_tiles(&case, 24);
        let f = 8 / case.run.out_factor;
        let cells = f * f * f;
        let bytes = hasty_cfd::gpu::GpuBackend::estimate_bytes(tiles.len(), cells, case.motion.voxels(), case.motion.nt, case.sdf.data.len());
        let nodes = tiles.len() * 512;
        let steps = case.run.cycles as u64 * u.steps_per_cycle;
        let acc_file = case.run.bins * tiles.len() * cells * 10 * 4;
        // ≈ 1.5–2 GLUPS on an RTX 4090 for a sparse D3Q19 pull scheme (all nodes of active tiles)
        println!(
            "plan: {} active tiles = {:.1} M nodes, device ≈ {:.1} GB, accumulator file {:.1} GB, {} steps ({} cycles) ≈ {:.0}–{:.0} min at 2–1.5 GLUPS",
            tiles.len(),
            nodes as f64 / 1e6,
            bytes as f64 / 1e9,
            acc_file as f64 / 1e9,
            steps,
            case.run.cycles,
            nodes as f64 * steps as f64 / 2e9 / 60.0,
            nodes as f64 * steps as f64 / 1.5e9 / 60.0
        );
        return;
    }
    std::fs::create_dir_all(&out).unwrap();
    if let Some(p) = probe {
        let mut it = p.split(':');
        let steps: u64 = it.next().unwrap().parse().unwrap();
        let every: u64 = it.next().map_or((steps / 20).max(1), |x| x.parse().unwrap());
        if cpu {
            run_probe(Solver::new_cpu(case).unwrap(), &out, steps, every, &snaps);
        } else {
            let dev = hasty_cuda::Device::new(0).unwrap();
            run_probe(Solver::new_gpu(case, &dev).unwrap(), &out, steps, every, &snaps);
        }
        return;
    }
    let ctl = hasty_recon::checkpoint::RunCtl::new(&out, 0);
    let calls = AtomicU64::new(0);
    // the STOP file is polled every 200 steps (a file system call per step would be noticeable on small cases)
    let stop = || calls.fetch_add(1, Ordering::Relaxed) % 200 == 0 && ctl.stop_requested();
    let rep = if cpu {
        let mut s = Solver::new_cpu(case).unwrap();
        println!("CPU backend: {} tiles, {:.2} M nodes", s.topo.ntiles, s.topo.nodes() as f64 / 1e6);
        s.run(&out, &stop, !quiet)
    } else {
        let dev = hasty_cuda::Device::new(0).unwrap();
        let mut s = Solver::new_gpu(case, &dev).unwrap();
        println!("GPU backend: {} tiles, {:.2} M nodes", s.topo.ntiles, s.topo.nodes() as f64 / 1e6);
        s.run(&out, &stop, !quiet)
    };
    match rep {
        Ok(r) => println!("{} after {} steps in {:.0} s ({:.0} MLUPS) → {}", if r.stopped { "stopped (resumable)" } else { "done" }, r.steps, r.seconds, r.mlups, out.display()),
        Err(e) => {
            eprintln!("cardiac_lbm: {e}");
            std::process::exit(1);
        }
    }
}

fn run_probe<B: hasty_cfd::backend::Backend>(mut s: Solver<B>, out: &std::path::Path, steps: u64, every: u64, snaps: &[u64]) {
    use std::io::Write;
    let mut log = std::fs::File::create(out.join("probe.jsonl")).unwrap();
    s.init().unwrap();
    let mut marks: Vec<u64> = (1..=steps / every).map(|k| k * every).chain(snaps.iter().copied()).filter(|&x| x <= steps).collect();
    marks.sort_unstable();
    marks.dedup();
    for m in marks {
        // diagnostics every 1000 steps on the way (they drive the inlet flux controllers, as in a run)
        while s.step < m {
            let next = (s.step / 1000 + 1) * 1000;
            s.advance(next.min(m) - s.step).unwrap();
            if s.step % 1000 == 0 {
                s.diagnostics().unwrap();
            }
        }
        let d = s.diagnostics().unwrap();
        let h = hasty_cfd::probe::hot_spots(&mut s, 12).unwrap();
        let top = &h["top"][0];
        println!(
            "step {} t {:.1} ms Δρ̄ {:.2e}: speed p99 {:.2} p99.9 {:.2} p99.99 {:.2} m/s; classes {}; fastest {:.2} m/s {} at {:?} mm (valve {} {:.1} mm, cut {} {:.1} mm) ρ {:.3} φ {:.2} mm B {:.2}",
            s.step,
            s.time() * 1e3,
            d["drho_mean"].as_f64().unwrap(),
            h["p99"].as_f64().unwrap(),
            h["p999"].as_f64().unwrap(),
            h["p9999"].as_f64().unwrap(),
            h["classes"],
            top["speed"].as_f64().unwrap_or(0.0),
            top["class"],
            top["x_mm"].as_array().map(|a| a.iter().map(|v| (v.as_f64().unwrap() * 10.0).round() / 10.0).collect::<Vec<_>>()).unwrap_or_default(),
            top["valve"]["name"],
            top["valve"]["mm"].as_f64().unwrap_or(0.0),
            top["cut"]["name"],
            top["cut"]["mm"].as_f64().unwrap_or(0.0),
            top["rho"].as_f64().unwrap_or(0.0),
            top["phi_mm"].as_f64().unwrap_or(0.0),
            top["B"].as_f64().unwrap_or(0.0)
        );
        writeln!(log, "{}", serde_json::json!({"diag": d, "hot": h})).unwrap();
        if snaps.contains(&s.step) {
            let p = hasty_cfd::probe::snapshot(&mut s, &out.join("snaps")).unwrap();
            println!("   snapshot → {}", p.display());
        }
    }
}

/// the protocol's spin history: the slab pulse's Bloch profile along the slab normal (see the --spin doc)
fn spin_spec(arg: &str, ncuts: usize, blood_t1: Option<f64>) -> hasty_cfd::case::SpinSpec {
    let mut kv: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for p in arg.split(',').filter(|p| !p.is_empty()) {
        let (k, v) = p.split_once('=').unwrap_or_else(|| panic!("--spin: {p} is not key=value"));
        kv.insert(k.trim().into(), v.trim().into());
    }
    let num = |k: &str, d: f64| kv.get(k).map_or(d, |v| v.parse().unwrap_or_else(|_| panic!("--spin {k}={v}")));
    let vec3 = |k: &str, d: [f64; 3]| {
        kv.get(k).map_or(d, |v| {
            let x: Vec<f64> = v.split(';').map(|t| t.parse().unwrap_or_else(|_| panic!("--spin {k}={v}"))).collect();
            assert!(x.len() == 3, "--spin {k}: three ';'-separated numbers");
            [x[0], x[1], x[2]]
        })
    };
    for k in kv.keys() {
        assert!(["tr", "flip", "t1", "center", "normal", "thk", "tbw", "dur"].contains(&k.as_str()), "--spin: unknown key {k}");
    }
    let t1 = match (kv.contains_key("t1"), blood_t1) {
        (true, _) => num("t1", 0.0),
        (false, Some(t)) => t,
        (false, None) => panic!("--spin: the case has no blood T1 (rebuild it with lbm_case.py) — give t1=…"),
    };
    let (tr, flip, thk, tbw, dur) = (num("tr", 11.29e-3), num("flip", 12.0), num("thk", 0.16), num("tbw", 6.0), num("dur", 0.8e-3));
    let center = vec3("center", [0.02, -0.03, 1.31]);
    let nr = vec3("normal", [0.0, 0.0, 1.0]);
    let nn = (nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]).sqrt();
    let normal = nr.map(|x| x / nn);
    let pulse = hasty_mrsim::rf::SlabPulse::new(dur, tbw, flip.to_radians(), thk);
    let prof = pulse.profile(0.75 * thk + 0.02, 3001, 0.0);
    println!(
        "spin history: TR {:.2} ms, T1 {t1:.2} s, slab {:.0} mm (FWHM {:.1} mm) at {:?} along {:?}, centre flip {:.1}°, sinc TBW {tbw} over {:.2} ms",
        tr * 1e3,
        thk * 1e3,
        prof.fwhm() * 1e3,
        center,
        normal,
        prof.alpha_at(0.0).to_degrees(),
        dur * 1e3
    );
    let protocol = serde_json::json!({"tr": tr, "flip_deg": flip, "t1": t1, "thk": thk, "tbw": tbw, "dur": dur, "center": center, "normal": normal, "fwhm": prof.fwhm()});
    hasty_cfd::case::SpinSpec { tr, t1, center, normal, z0: prof.z0, dz: prof.dz, alpha: prof.alpha, s_in: vec![1.0; ncuts], protocol }
}
