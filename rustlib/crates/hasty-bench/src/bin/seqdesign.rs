//! Sequence design run (hasty-seq): finds the yarn base spoke with the shortest Nyquist-complete scan N·TR for a readout
//! budget under G_max / S_max and the SAFE PNS limit of the played sequence (encode on physical axes, rotated spoke,
//! per-rotation tail to a constant physical spoil moment), then reports density, coverage of concrete rotation sets, concomitant anisotropy, readout M1, PNS and the
//! GIRF-actual trajectory, and the encoding moments at TE from the designed waveforms. Writes arrays + summary.json into <out>; figures: tools/seq/plot_seqdesign.py <out>.
//! Interruptible (STOP file / SIGINT) and resumable: evaluations are cached in <out>/evals.jsonl and the search is
//! deterministic, so a rerun replays the cache and continues.
//!   seqdesign <out> [--res 2e-3] [--fov 0.32] [--t-adc 3e-3] [--venc 1.5 | --v-ext V]
//!             [--scheme hadamard|referenced|dual7|coprime7|coprime8|coprime8h] [--p 2]
//!             [--pns-target 80] [--asc FILE] [--edge 1] [--dwell S] [--rot super|fib|golden] [--n-rot-pns 400]
//!             [--aniso-weight 0] [--spoil-cycles 2 | 0.5 with --return] [--iters 80] [--check]
//!             [--return R]   out-and-back readout back to R·kmax, ending at rest; per-rotation straight move to the spoil
//!                            moment (|spoil| ≤ kmax keeps every sample after the encode inside kmax)
//!             [--spoil physical|endpoint|radial|flowcomp] constant z moment with a per-rotation tail | no tail, the
//!                            readout end point is the moment | the trajectory continues out along its end direction
//!                            (rotating) | constant z moment + equal TR M1 per encode (needs --return 0)
//!             [--threads T]  total worker threads (default: all cores); several runs in parallel → split the cores
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use hasty_phantom::volume::write_npy;
use hasty_seq::encode::{DualLayout, Scheme};
use hasty_seq::girf::{trajectory_error, Girf};
use hasty_seq::metrics::{concomitant_anisotropy, concomitant_phase, coverage, density, max_k_error, readout_m1};
use hasty_seq::optim::{nelder_mead, shape_of, Eval, Problem, Shape};
use hasty_seq::pns::{pns, SafeHw};
use hasty_seq::system::System;
use hasty_seq::tr::{sequence, Spoil};
use hasty_seq::traj::{acquisition_order, rotations, RotSet};
use hasty_seq::wave::{trap_shape, Grad};
use hasty_seq::{Vec3, GAMMA_BAR};
use serde_json::{json, Value};

fn f8(path: &Path, shape: &[usize], v: &[f64]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy(path, "<f8", shape, false, &b).unwrap();
}
fn flat3(v: &[Vec3]) -> Vec<f64> {
    v.iter().flat_map(|p| p.iter().copied()).collect()
}

#[derive(Clone, Debug)]
struct Rec {
    shape: Shape,
    objective: f64,
    feasible: bool,
    n: f64,
    tr: f64,
    te: f64,
    t_adc: f64,
    pns: f64,
    part: String,
    slew: [f64; 3],
    aniso: f64,
}

impl Rec {
    fn of(e: &Eval) -> Rec {
        Rec { shape: e.shape, objective: e.objective, feasible: e.feasible, n: e.n_eff, tr: e.tr.tr(), te: e.tr.te(), t_adc: e.tr.spoke.t_adc, pns: e.pns.max, part: format!("{:?}", e.pns.part), slew: e.slew, aniso: e.aniso }
    }
    fn infeasible(s: Shape) -> Rec {
        Rec { shape: s, objective: f64::INFINITY, feasible: false, n: 0.0, tr: 0.0, te: 0.0, t_adc: 0.0, pns: 0.0, part: "none".into(), slew: [0.0; 3], aniso: 0.0 }
    }
    fn json(&self) -> Value {
        json!({"bits": key(&self.shape).map(|b| format!("{b:016x}")), "kg0": self.shape.kg0, "wobble": self.shape.wobble, "wobble_freq": self.shape.wobble_freq,
               "objective": if self.objective.is_finite() { self.objective } else { -1.0 }, "feasible": self.feasible, "n": self.n,
               "tr": self.tr, "te": self.te, "t_adc": self.t_adc, "pns": self.pns, "part": self.part, "slew": self.slew, "aniso": self.aniso})
    }
    fn from_json(v: &Value) -> Rec {
        let f = |k: &str| v[k].as_f64().unwrap();
        let o = f("objective");
        let sl: Vec<f64> = v["slew"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        // exact shape from the stored bit patterns (serde_json's default float parsing is not always correctly rounded)
        let shape = match v["bits"].as_array() {
            Some(b) => {
                let u: Vec<f64> = b.iter().map(|x| f64::from_bits(u64::from_str_radix(x.as_str().unwrap(), 16).unwrap())).collect();
                Shape { kg0: u[0], wobble: u[1], wobble_freq: u[2] }
            }
            None => Shape { kg0: f("kg0"), wobble: f("wobble"), wobble_freq: f("wobble_freq") },
        };
        Rec {
            shape,
            objective: if o < 0.0 { f64::INFINITY } else { o },
            feasible: v["feasible"].as_bool().unwrap(),
            n: f("n"),
            tr: f("tr"),
            te: f("te"),
            t_adc: f("t_adc"),
            pns: f("pns"),
            part: v["part"].as_str().unwrap().to_string(),
            slew: [sl[0], sl[1], sl[2]],
            aniso: f("aniso"),
        }
    }
}

fn key(s: &Shape) -> [u64; 3] {
    [s.kg0.to_bits(), s.wobble.to_bits(), s.wobble_freq.to_bits()]
}

struct Cache {
    map: Mutex<HashMap<[u64; 3], Rec>>,
    file: Mutex<std::fs::File>,
}

impl Cache {
    fn open(path: &Path) -> Cache {
        let mut map = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(path) {
            for l in text.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(v) = serde_json::from_str::<Value>(l) {
                    let r = Rec::from_json(&v);
                    map.insert(key(&r.shape), r);
                }
            }
        }
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
        Cache { map: Mutex::new(map), file: Mutex::new(file) }
    }
    fn get(&self, s: &Shape) -> Option<Rec> {
        self.map.lock().unwrap().get(&key(s)).cloned()
    }
    fn put(&self, r: Rec) {
        let mut f = self.file.lock().unwrap();
        writeln!(f, "{}", serde_json::to_string(&r.json()).unwrap()).unwrap();
        f.flush().unwrap();
        self.map.lock().unwrap().insert(key(&r.shape), r);
    }
    fn len(&self) -> usize {
        self.map.lock().unwrap().len()
    }
}

fn eval_cached(p: &Problem, rots: &[hasty_seq::Mat3], cache: &Cache, s: &Shape) -> Rec {
    if let Some(r) = cache.get(s) {
        return r;
    }
    let r = match p.evaluate(s, rots) {
        Some(e) => Rec::of(&e),
        None => Rec::infeasible(*s),
    };
    cache.put(r.clone());
    r
}

/// pypulseq cross-check data: waveforms + Rust SAFE PNS + exact moments
fn write_check(out: &Path, p: &Problem, best: &Eval) {
    let d = out.join("check");
    std::fs::create_dir_all(&d).unwrap();
    let dt = p.sys.grad_raster;
    let rots = p.pns_rotations();
    let mut cases: Vec<(String, Grad)> = vec![
        ("trapezoid_x".into(), Grad::from_axis(0, &trap_shape(20, 30, 20, 0.04), dt, 0.0)),
        ("triangle_train_y".into(), {
            let mut s = vec![0.0];
            for k in 0..12 {
                let sg = if k % 2 == 0 { 1.0 } else { -1.0 };
                for i in 1..=20 {
                    s.push(sg * 0.035 * i as f64 / 20.0);
                }
                for i in (0..20).rev() {
                    s.push(sg * 0.035 * i as f64 / 20.0);
                }
            }
            Grad::from_axis(1, &s, dt, 0.0)
        }),
    ];
    cases.push(("sequence_3rot".into(), sequence(&best.tr, &rots[..3])));
    let mut names = vec![];
    for (name, g) in &cases {
        f8(&d.join(format!("{name}_gwf.npy")), &[g.len(), 3], &flat3(&g.g));
        let pr = pns(g, &p.hw);
        let mut a = Vec::new();
        for (ax, nn) in pr.axis.iter().zip(&pr.norm) {
            a.extend_from_slice(ax);
            a.push(*nn);
        }
        f8(&d.join(format!("{name}_pns.npy")), &[pr.norm.len(), 4], &a);
        let m = g.moments(0.0);
        let mut mm = Vec::new();
        for i in 0..m.t.len() {
            mm.extend_from_slice(&m.m0[i]);
            mm.extend_from_slice(&m.m1[i]);
            mm.extend_from_slice(&m.m2[i]);
        }
        f8(&d.join(format!("{name}_moments.npy")), &[m.t.len(), 9], &mm);
        names.push(name.clone());
    }
    let hw = &p.hw;
    let hwj: Vec<Value> = hw.axes.iter().map(|a| json!({"tau": a.tau, "a": a.a, "stim_limit": a.stim_limit, "stim_thresh": a.stim_thresh, "g_scale": a.g_scale})).collect();
    std::fs::write(d.join("check.json"), serde_json::to_string_pretty(&json!({"dt": dt, "cases": names, "hw": hwj, "hw_name": hw.name})).unwrap()).unwrap();
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let res: f64 = take("--res").map(|s| s.parse().unwrap()).unwrap_or(2.0e-3);
    let fov: f64 = take("--fov").map(|s| s.parse().unwrap()).unwrap_or(0.32);
    let t_adc: f64 = take("--t-adc").map(|s| s.parse().unwrap()).unwrap_or(3.0e-3);
    // --venc is the (low) VENC; --v-ext instead sets it from the speed that must not alias in any direction
    let venc_arg: Option<f64> = take("--venc").map(|s| s.parse().unwrap());
    let v_ext: Option<f64> = take("--v-ext").map(|s| s.parse().unwrap());
    let p_cop: u32 = take("--p").map(|s| s.parse().unwrap()).unwrap_or(2);
    let scheme_name = take("--scheme").unwrap_or_else(|| "hadamard".into());
    let scheme = match scheme_name.as_str() {
        "referenced" => Scheme::Referenced,
        "hadamard" => Scheme::Hadamard,
        "dual7" => Scheme::DualVenc { venc_hi: v_ext.expect("dual7 needs --v-ext (= VENC_hi)") },
        "coprime7" => Scheme::Coprime { p: p_cop, layout: DualLayout::Seven },
        "coprime8" => Scheme::Coprime { p: p_cop, layout: DualLayout::Eight },
        "coprime8h" => Scheme::Coprime { p: p_cop, layout: DualLayout::EightHadamard },
        s => panic!("unknown scheme {s}"),
    };
    // Hadamard pairs measure v_i + v_j: their range must be √2 larger for the same protection in every direction
    let had = matches!(scheme, Scheme::Hadamard | Scheme::Coprime { layout: DualLayout::EightHadamard, .. });
    let venc = match (venc_arg, v_ext) {
        (Some(v), _) => v,
        (None, Some(ve)) => {
            let ve = if had { ve * 2f64.sqrt() } else { ve };
            match scheme {
                Scheme::Coprime { p, .. } => ve / (p as f64 + 1.0),
                Scheme::DualVenc { venc_hi } => venc_hi / 2.0,
                _ => ve,
            }
        }
        (None, None) => 1.5,
    };
    let pns_target: f64 = take("--pns-target").map(|s| s.parse().unwrap()).unwrap_or(80.0);
    let asc = take("--asc");
    let edge: f64 = take("--edge").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let dwell: Option<f64> = take("--dwell").map(|s| s.parse().unwrap());
    let rot_kind = match take("--rot").as_deref() {
        Some("fib") => RotSet::FibonacciGolden,
        Some("golden") => RotSet::GoldenMeans,
        _ => RotSet::SuperFibonacci,
    };
    let n_rot_pns: usize = take("--n-rot-pns").map(|s| s.parse().unwrap()).unwrap_or(400);
    let aniso_weight: f64 = take("--aniso-weight").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let ret: Option<f64> = take("--return").map(|s| s.parse().unwrap());
    // --spoil physical (constant z moment, per-rotation tail) | endpoint (no tail: the readout end point, rotating)
    //         | radial (the trajectory continues out along its end direction to the spoil moment, rotating)
    let spoil_mode = take("--spoil").unwrap_or_else(|| "physical".into());
    let spoil_endpoint = spoil_mode == "endpoint";
    assert!(!spoil_endpoint || ret.is_some(), "--spoil endpoint needs --return (the readout must end at rest)");
    let threads: usize = take("--threads").map(|s| s.parse().unwrap()).unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let spoil_cycles: f64 = take("--spoil-cycles").map(|s| s.parse().unwrap()).unwrap_or(if ret.is_some() { 0.5 } else { 2.0 });
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(80);
    let flag = |a: &mut Vec<String>, n: &str| a.iter().position(|x| x == n).map(|i| { a.remove(i); true }).unwrap_or(false);
    let check = flag(&mut args, "--check");
    let out = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    let ctl = hasty_recon::checkpoint::RunCtl::new(&out, 1);
    let t0 = Instant::now();

    let sys = System::prisma_xr();
    let (hw, hw_note) = match &asc {
        Some(f) => (SafeHw::from_asc(&std::fs::read_to_string(f).unwrap(), false).unwrap(), format!("from {f}")),
        None => (SafeHw::example(), "pypulseq EXAMPLE hardware (not a real scanner) — supply --asc for real PNS".to_string()),
    };
    let kmax = 0.5 / res;
    let p = Problem {
        sys: sys.clone(),
        hw,
        pns_target,
        kmax,
        fov,
        edge_density: edge,
        t_adc,
        vmax: dwell.map(|d| 1.0 / (fov * d)),
        rf_dur: 100e-6,
        scheme,
        venc,
        balanced: true,
        spoil: match spoil_mode.as_str() {
            "endpoint" => Spoil::Endpoint,
            "radial" => Spoil::Radial(spoil_cycles / (GAMMA_BAR * res)),
            "flowcomp" => Spoil::FlowComp([0.0, 0.0, spoil_cycles / (GAMMA_BAR * res)]),
            _ => Spoil::Physical([0.0, 0.0, spoil_cycles / (GAMMA_BAR * res)]),
        },
        ret,
        rot_kind,
        n_rot_pns,
        aniso_weight,
        curve_pts: 60_000,
    };
    assert!(spoil_mode != "flowcomp" || ret == Some(0.0), "--spoil flowcomp needs --return 0");
    if ret.is_some() && spoil_mode == "physical" && spoil_cycles > 0.5 {
        println!("  note: spoil {spoil_cycles} cycles/voxel > 0.5 (= kmax): the tail leaves the kmax ball");
    }
    println!("seqdesign: res {:.2} mm, FOV {:.0} mm, kmax {kmax:.0} /m, T_adc {:.2} ms, VENC {venc} m/s {scheme:?}, PNS ≤ {pns_target}% ({hw_note})", res * 1e3, fov * 1e3, t_adc * 1e3);
    let rots = p.pns_rotations();
    // the cache is only valid for the same problem
    let fp = format!("{:?}|{:?}|{pns_target}|{kmax}|{fov}|{edge}|{t_adc}|{:?}|{scheme:?}|{venc}|{:?}|{rot_kind:?}|{n_rot_pns}|{aniso_weight}|{}{}", sys, p.hw, p.vmax, p.spoil, p.curve_pts, ret.map_or(String::new(), |r| format!("|ret{r}")));
    let fpf = out.join("problem.txt");
    match std::fs::read_to_string(&fpf) {
        Ok(old) if old != fp => {
            eprintln!("{} holds evaluations of a different problem; use another output directory", out.display());
            std::process::exit(2);
        }
        Ok(_) => {}
        Err(_) => std::fs::write(&fpf, &fp).unwrap(),
    }
    let cache = Cache::open(&out.join("evals.jsonl"));
    println!("  cache: {} evaluations", cache.len());

    // ── coarse grid (parallel) ──
    let mut grid = vec![];
    for kg0 in [0.0, 0.5, 1.5] {
        for wob in [0.0, 0.5, 1.0, 2.0] {
            for fr in [0.5, 2.0, 6.0] {
                if wob == 0.0 && fr != 2.0 {
                    continue;
                }
                grid.push(Shape { kg0, wobble: wob, wobble_freq: fr });
            }
        }
    }
    let next = AtomicUsize::new(0);
    let stopped = std::sync::atomic::AtomicBool::new(false);
    // grid: shapes in parallel, the rest of the thread budget inside each evaluation; Nelder–Mead: all inside
    let outer = threads.min(grid.len()).max(1);
    hasty_seq::set_threads((threads / outer).max(1));
    std::thread::scope(|sc| {
        for _ in 0..outer {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= grid.len() {
                    break;
                }
                if ctl.stop_requested() {
                    stopped.store(true, Ordering::SeqCst);
                    break;
                }
                eval_cached(&p, &rots, &cache, &grid[i]);
            });
        }
    });
    if stopped.load(Ordering::SeqCst) {
        println!("stopped during the grid ({} cached); rerun to resume", cache.len());
        return;
    }
    let grid_recs: Vec<Rec> = grid.iter().map(|s| cache.get(s).unwrap()).collect();
    println!("  grid ({:.0} s):  kg0  wobble  freq |     N   TR ms  T_adc  PNS%  part     aniso  N·TR s", t0.elapsed().as_secs_f64());
    for r in &grid_recs {
        println!(
            "    {:5.2} {:6.2} {:5.2} | {:6.0} {:6.3} {:5.2} {:5.1} {:8} {:5.2} {}",
            r.shape.kg0, r.shape.wobble, r.shape.wobble_freq, r.n, r.tr * 1e3, r.t_adc * 1e3, r.pns, r.part, r.aniso,
            if r.objective.is_finite() { format!("{:.2}", r.objective) } else { "infeasible".into() }
        );
    }
    let gbest = grid_recs.iter().filter(|r| r.objective.is_finite()).min_by(|a, b| a.objective.partial_cmp(&b.objective).unwrap()).cloned();
    let Some(gbest) = gbest else {
        println!("no feasible grid point (readout budget too short or PNS target unreachable)");
        return;
    };

    // ── Nelder–Mead from the best grid point ──
    hasty_seq::set_threads(threads);
    let mut stop_flag = false;
    let x0 = [gbest.shape.kg0, gbest.shape.wobble.max(0.05), gbest.shape.wobble_freq.ln()];
    let mut f = |x: &[f64]| -> f64 {
        if stop_flag || ctl.stop_requested() {
            stop_flag = true;
            return f64::INFINITY;
        }
        let r = eval_cached(&p, &rots, &cache, &shape_of(x));
        if r.objective.is_finite() { r.objective } else { 1e30 }
    };
    let (xb, fb) = nelder_mead(&mut f, &x0, &[0.3, 0.3, 0.5], iters, 1e-4);
    if stop_flag {
        println!("stopped during Nelder–Mead ({} cached); rerun to resume", cache.len());
        return;
    }
    let best_shape = shape_of(&xb);
    println!("  Nelder–Mead ({:.0} s, {} evaluations): {best_shape:?} → N·TR {fb:.3} s (grid best {:.3})", t0.elapsed().as_secs_f64(), cache.len(), gbest.objective);

    // ── final check with a denser PNS rotation set ──
    let pf = Problem { n_rot_pns: n_rot_pns * 8, ..p.clone() };
    let rots_f = pf.pns_rotations();
    let best = pf.evaluate(&best_shape, &rots_f).expect("best shape infeasible on the dense check");
    let tr = &best.tr;
    let sp = &tr.spoke;
    let n_enc = tr.n_enc();
    println!(
        "  final: N {:.0} (yarn design {:.0}), TR {:.3} ms, TE {:.3} ms, T_adc {:.3} ms, encode {:.3} ms, tail {:.3} ms, PNS {:.1}% ({:?}) slew factors {:?}",
        best.n_eff, best.n_design, tr.tr() * 1e3, tr.te() * 1e3, sp.t_adc * 1e3, (tr.i_ro - tr.i_enc) as f64 * tr.dt * 1e3, tr.n_tail as f64 * tr.dt * 1e3,
        best.pns.max, best.pns.part, best.slew
    );

    // ── encoding at the echo, from the designed waveforms ──
    let mut enc_rows = vec![];
    let mut m1_te = vec![];
    println!("  moments at TE (readout start), RF centre origin:  M0 (mT·ms/m)  M1 (mT·ms²/m)  M2 (mT·ms³/m)");
    for e in 0..n_enc {
        let m = tr.physical(e, &rots_f[0]).moments(tr.t_rf);
        let (m0, m1, m2) = (m.m0[tr.i_ro], m.m1[tr.i_ro], m.m2[tr.i_ro]);
        let pe = m.p[tr.i_ro];
        println!(
            "    enc {e}: M0 [{:+.1e} {:+.1e} {:+.1e}]  M1 [{:+.4} {:+.4} {:+.4}]  M2 [{:+.4} {:+.4} {:+.4}]",
            m0[0] * 1e6, m0[1] * 1e6, m0[2] * 1e6, m1[0] * 1e9, m1[1] * 1e9, m1[2] * 1e9, m2[0] * 1e12, m2[1] * 1e12, m2[2] * 1e12
        );
        m1_te.push(m1);
        enc_rows.push(json!({"m0": m0, "m1": m1, "m2": m2, "p": pe}));
    }
    // pair VENCs from the designed first moments (must equal the scheme's), and the encode-to-encode concomitant spread
    let plan = hasty_seq::venc::plan(scheme, venc);
    let mut pair_vencs = vec![];
    for cb in &plan.combos {
        for pr in std::iter::once(&cb.lo).chain(cb.hi.iter()) {
            let g = [0, 1, 2].map(|q| hasty_seq::GAMMA * (m1_te[pr.plus][q] - m1_te[pr.minus][q]));
            let v_des = std::f64::consts::PI / hasty_seq::norm(&g);
            let v_nom = std::f64::consts::PI / hasty_seq::norm(&pr.g);
            assert!((v_des / v_nom - 1.0).abs() < 1e-9, "designed pair VENC {v_des} ≠ scheme {v_nom}");
            pair_vencs.push(v_des);
        }
    }
    let p0 = enc_rows[0]["p"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>();
    let dp = enc_rows.iter().map(|r| r["p"].as_array().unwrap().iter().zip(&p0).map(|(a, b)| (a.as_f64().unwrap() - b).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
    let maxwell = hasty_seq::GAMMA / (2.0 * sys.b0) * dp * 0.15 * 0.15;
    let (vlo, vhi) = scheme.vencs(venc);
    let sd20 = plan.velocity_std(n_enc, 1.0 / 20.0 / 2f64.sqrt());
    let snr_fail = if plan.crt.is_some() && plan.combos[0].hi.is_some() {
        let (mut lo, mut hi) = (1.0f64, 200.0f64);
        for _ in 0..40 {
            let mid = (lo * hi).sqrt();
            if plan.unwrap_failure_exact(n_enc, 1.0 / mid) > 1e-3 { lo = mid } else { hi = mid }
        }
        Some(hi)
    } else {
        None
    };
    println!(
        "  encoding: VENC lo/hi {vlo:.3}/{vhi:.3} m/s, unaliased {:.2} m/s per pair direction{}; pair VENCs from the waveforms {:?}; σv at SNR 20 {:.1} mm/s; unwrap 1e-3 failures below SNR {}; encode-to-encode concomitant ΔP → {:.3} rad at 15 cm",
        scheme.unaliased(venc), if had { " (Hadamard pairs: v_i+v_j)" } else { "" }, pair_vencs.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        sd20.iter().sum::<f64>() / 3.0 * 1e3, snr_fail.map_or("-".to_string(), |x| format!("{x:.1}")), maxwell
    );

    // ── metrics ──
    let up = ((2.0 * GAMMA_BAR * sys.gmax * sys.grad_raster * fov).ceil() as usize).max(1);
    let k = sp.k_adc(up);
    let yarn = p.yarn(&best_shape, best.n_design);
    let den = density(&k, kmax, fov, |r| yarn.w(r), 60);
    let lim = sp.grad.limits();
    let m1 = readout_m1(&sp.grad, 0.0, kmax, &[0.1, 0.25, 0.5, 0.99]);
    let pm = sp.grad.moments(0.0);
    let p_end = pm.p[sp.i_adc];
    let vmax_along = sp.grad.g.iter().map(hasty_seq::norm).fold(0.0, f64::max) * GAMMA_BAR;
    println!(
        "  density: N* {:.0} at r {:.0} /m (design N {:.0}); |G| {:.1} mT/m, |S| {:.0} T/m/s; aniso {:.2}, conc. phase at 15 cm {:.2} rad; readout γM1 at first reach of 10/25/50/99% kmax: {:.2e} {:.2e} {:.2e} {:.2e} rad/(m/s)",
        den.n_star, den.r_star, best.n_eff, lim.g_vec * 1e3, lim.s_vec, concomitant_anisotropy(&p_end), concomitant_phase(&p_end, sys.b0, 0.15), m1[0], m1[1], m1[2], m1[3]
    );
    let n_cov = best.n_eff.ceil() as usize;
    let mut cov_rows = vec![];
    let mut cov_json = serde_json::Map::new();
    for (name, kind) in [("super", RotSet::SuperFibonacci), ("fib", RotSet::FibonacciGolden), ("golden", RotSet::GoldenMeans)] {
        let rs = rotations(kind, n_cov);
        let c = coverage(&k, &rs, kmax, fov, 20);
        // a temporally contiguous 1/8 of the acquisition (what one cardiac bin sees, roughly)
        let ord = acquisition_order(kind, n_cov);
        let sub: Vec<_> = ord.iter().take(n_cov / 8).map(|&i| rs[i]).collect();
        let c8 = coverage(&k, &sub, kmax, fov, 20);
        let outer = |v: &[f64]| v[v.len() * 3 / 4..].iter().sum::<f64>() / (v.len() - v.len() * 3 / 4) as f64;
        println!("  coverage {name:6}: empty Nyquist cells (outer quarter) full set {:.3}, 1/8 subset {:.3} (Poisson e^-1/8 = {:.3})", outer(&c.empty), outer(&c8.empty), (-1.0f64 / 8.0).exp());
        for i in 0..c.r.len() {
            cov_rows.extend_from_slice(&[c.r[i], c.empty[i], c.mean_hits[i], c8.empty[i]]);
        }
        cov_json.insert(name.into(), json!({"empty_outer": outer(&c.empty), "empty_outer_eighth": outer(&c8.empty)}));
    }
    // GIRF: actual trajectory for a few rotations
    let girf = Girf::clinical_example();
    let mut gerr_max: f64 = 0.0;
    let mut gerr0 = vec![];
    for (j, r) in rots.iter().take(16).enumerate() {
        let nom = sp.grad.rotated(r);
        let (act, b0) = girf.apply(&nom);
        let (err, _) = trajectory_error(&nom, &act, &b0);
        gerr_max = gerr_max.max(max_k_error(&err, sp.i_adc + 1, fov));
        if j == 0 {
            gerr0 = err;
        }
    }
    println!("  GIRF (clinical example): max |Δk| over the ADC {:.2} Nyquist cells (16 rotations)", gerr_max);

    // ── where the samples are: |k| / kmax per TR part over the dense rotation set (readout and tail do not depend on
    // the encode: M0 = 0 at the echo); encode excursions are navigator-only ──
    let kparts = hasty_seq::par_map(&rots_f, |r| {
        let m = tr.physical(0, r).moments(tr.t_rf);
        let k = m.k();
        let (mut ro, mut tl, mut n_ro, mut n_tl) = (0.0f64, 0.0f64, 0usize, 0usize);
        for (i, q) in k.iter().enumerate().skip(tr.i_ro) {
            let a = hasty_seq::norm(q) / kmax;
            if i <= tr.i_tail {
                ro = ro.max(a);
                n_ro += (a > 1.0 + 1e-9) as usize;
            } else {
                tl = tl.max(a);
                n_tl += (a > 1.0 + 1e-9) as usize;
            }
        }
        (ro, tl, n_ro, n_tl)
    });
    let ro_max = kparts.iter().map(|x| x.0).fold(0.0, f64::max);
    let tl_max = kparts.iter().map(|x| x.1).fold(0.0, f64::max);
    let ro_out: usize = kparts.iter().map(|x| x.2).sum();
    let tl_out: usize = kparts.iter().map(|x| x.3).sum();
    let mut enc_max = 0.0f64;
    let mut ktr = vec![];
    for e in 0..n_enc {
        let k = tr.physical(e, &rots_f[0]).moments(tr.t_rf).k();
        for (i, q) in k.iter().enumerate() {
            if i >= tr.i_enc && i < tr.i_ro {
                enc_max = enc_max.max(hasty_seq::norm(q) / kmax);
            }
        }
        ktr.extend(flat3(&k));
    }
    println!(
        "  |k|/kmax over {} rotations: readout max {ro_max:.4} ({ro_out} raster samples outside), tail max {tl_max:.4} ({tl_out} outside); encode (navigator) max {enc_max:.2}",
        rots_f.len()
    );
    let kcheck = json!({"readout_max": ro_max, "tail_max": tl_max, "readout_outside": ro_out, "tail_outside": tl_out, "encode_max": enc_max, "rotations": rots_f.len()});
    f8(&out.join("k_tr.npy"), &[n_enc, tr.n + 1, 3], &ktr);
    let parts: Vec<f64> = (0..=tr.n).map(|i| tr.part(i.min(tr.n - 1)) as usize as f64).collect();
    f8(&out.join("parts.npy"), &[tr.n + 1], &parts);

    // ── arrays ──
    let mut spg = sp.grad.g.clone();
    spg.extend_from_slice(&tr.tail(&rots[0]).g[1..]);
    f8(&out.join("spoke_grad.npy"), &[spg.len(), 3], &flat3(&spg));
    f8(&out.join("k_adc.npy"), &[k.len(), 3], &flat3(&k));
    let mut trp = vec![];
    for e in 0..n_enc {
        trp.extend(flat3(&tr.physical(e, &rots[0]).g));
    }
    f8(&out.join("tr_phys.npy"), &[n_enc, tr.n + 1, 3], &trp);
    // PNS trace: the first 4 rotations of the sequence, and a window ending at the worst TR
    let trace = |r0: usize, nr: usize| -> Vec<f64> {
        let pr = pns(&sequence(tr, &rots_f[r0..(r0 + nr).min(rots_f.len())]), &p.hw);
        let mut a = vec![];
        for (ax, nn) in pr.axis.iter().zip(&pr.norm).skip(pr.pad) {
            a.extend_from_slice(ax);
            a.push(*nn);
        }
        a
    };
    let a = trace(0, 4);
    f8(&out.join("pns_trace.npy"), &[a.len() / 4, 4], &a);
    let w0 = best.pns.rot.saturating_sub(1);
    let a = trace(w0, 3);
    f8(&out.join("pns_worst.npy"), &[a.len() / 4, 4], &a);
    let mut dn = vec![];
    for i in 0..den.r.len() {
        dn.extend_from_slice(&[den.r[i], den.l1[i], den.need[i]]);
    }
    f8(&out.join("density.npy"), &[den.r.len(), 3], &dn);
    f8(&out.join("coverage.npy"), &[3, cov_rows.len() / 12, 4], &cov_rows);
    f8(&out.join("girf_err.npy"), &[gerr0.len(), 3], &flat3(&gerr0));
    let mut gt = vec![];
    for r in &grid_recs {
        gt.extend_from_slice(&[r.shape.kg0, r.shape.wobble, r.shape.wobble_freq, r.n, r.tr, r.pns, r.aniso, if r.objective.is_finite() { r.objective } else { f64::NAN }]);
    }
    f8(&out.join("grid.npy"), &[grid_recs.len(), 8], &gt);
    let summary = json!({
        "problem": {"res": res, "fov": fov, "kmax": kmax, "t_adc_budget": t_adc, "venc": venc, "scheme": format!("{scheme:?}"), "pns_target": pns_target,
                    "pns_hw": p.hw.name, "pns_hw_note": hw_note, "edge_density": edge, "dwell": dwell, "rot": format!("{rot_kind:?}"),
                    "gmax": sys.gmax, "smax": sys.smax, "dt": sys.grad_raster, "b0": sys.b0},
        "best": {"shape": {"kg0": best_shape.kg0, "wobble": best_shape.wobble, "wobble_freq": best_shape.wobble_freq},
                 "n_design": best.n_design, "n_star": den.n_star, "r_star": den.r_star, "tr": tr.tr(), "te": tr.te(), "t_adc": sp.t_adc,
                 "i_adc": sp.i_adc, "i_enc": tr.i_enc, "i_ro": tr.i_ro, "i_tail": tr.i_tail, "n_tail": tr.n_tail, "n_tr": tr.n, "n_enc": n_enc, "spoil_cycles": spoil_cycles, "spoil_mode": spoil_mode,
                 "n_eff": best.n_eff, "ret": ret, "kmax_check": kcheck,
                 "scan_nyquist_s": best.n_eff * tr.tr() * n_enc as f64, "objective": best.objective,
                 "pns": best.pns.max, "pns_part": format!("{:?}", best.pns.part), "slew": best.slew, "g_vec": lim.g_vec, "s_vec": lim.s_vec,
                 "aniso": concomitant_anisotropy(&p_end), "readout_gamma_m1": m1, "curve_length": sp.length, "sigma_total": yarn.sigma_total(),
                 "vmax_along": vmax_along, "girf_max_err_cells": gerr_max, "coverage": cov_json, "upsample": up},
        "encoding": {"scheme": scheme_name, "p": p_cop, "venc_lo": vlo, "venc_hi": vhi, "unaliased": scheme.unaliased(venc), "hadamard_pairs": had,
                     "pair_vencs": pair_vencs, "sigma_v_snr20": sd20, "snr_unwrap_1e-3": snr_fail, "maxwell_dphase_15cm": maxwell, "at_te": enc_rows},
        "grid_best": gbest.json(),
        "evaluations": cache.len(),
        "elapsed_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    if check {
        write_check(&out, &p, &best);
        println!("  wrote check/ for tools/seq/check_pypulseq.py");
    }
    println!("done in {:.0} s → {}", t0.elapsed().as_secs_f64(), out.display());
}
