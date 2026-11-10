//! CPU validation of the solver (the same C core as the GPU kernels): rest state, Poiseuille, a translating capsule
//! (Galilean invariance with moving walls and refill), the expanding closed-end tube Q(x) = 2πRṘx
//! (research/cfd_training_data.md §6 gate 2), valve geometry and blocking.

use crate::backend::Backend;
use crate::case::{Case, CutKind, CutSpec, Grid, RunSpec, Sdf, Table, ValveKind, ValveSpec};
use crate::motion::Motion;
use crate::solver::Solver;

/// a case on a box of `dims` nodes (spacing dx, node (0,0,0) at the origin), SDF sampled from `sdf` on a grid with
/// `pad` extra nodes around it (voxel centres on the nodes)
#[allow(clippy::too_many_arguments)]
fn mk_case(dims: [usize; 3], dx: f64, pad: usize, sdf: impl Fn([f64; 3]) -> f64, motion: Motion, nu: f64, v_max: f64, u_lat: f64, valves: Vec<ValveSpec>, cuts: Vec<CutSpec>) -> Case {
    let origin = [0.0; 3];
    let sd = [dims[0] + 2 * pad, dims[1] + 2 * pad, dims[2] + 2 * pad];
    let so = [-(pad as f64) * dx - 0.5 * dx; 3];
    let mut data = vec![0f32; sd[0] * sd[1] * sd[2]];
    for z in 0..sd[2] {
        for y in 0..sd[1] {
            for x in 0..sd[0] {
                let p = [so[0] + (x as f64 + 0.5) * dx, so[1] + (y as f64 + 0.5) * dx, so[2] + (z as f64 + 0.5) * dx];
                data[(z * sd[1] + y) * sd[0] + x] = sdf(p) as f32;
            }
        }
    }
    Case {
        dir: std::path::PathBuf::from("."),
        grid: Grid { origin, dx, dims },
        sdf: Sdf { origin: so, h: dx, dims: sd, data },
        motion,
        nu,
        rho: 1060.0,
        v_max,
        u_lattice: u_lat,
        valves,
        cuts,
        run: RunSpec { geom_every: 2, cwale: 0.0, uclip: 0.0, inv_iters: 3, ..RunSpec::default() },
        regions: None,
        spin: None,
    }
}

/// mean fluid velocity (m/s) over nodes selected by `sel(g)` (lattice coordinates)
fn mean_u(s: &mut Solver<crate::backend::CpuBackend>, sel: impl Fn([i32; 3]) -> bool) -> ([f64; 3], usize) {
    let f = s.be.fields().unwrap();
    let n = s.be.nodes();
    let mut acc = [0.0; 3];
    let mut cnt = 0;
    for k in 0..n {
        if f.flags[k] & crate::params::F_FLUID == 0 || f.flags[k] & crate::params::F_BC != 0 {
            continue;
        }
        if !sel(s.topo.coords(k)) {
            continue;
        }
        for a in 0..3 {
            acc[a] += f.ru[(a + 1) * n + k] as f64;
        }
        cnt += 1;
    }
    let sc = s.u.dx / s.u.dt;
    (acc.map(|x| x / cnt.max(1) as f64 * sc), cnt)
}

#[test]
fn closed_box_stays_at_rest() {
    let dx = 1e-3;
    let c = mk_case([24, 24, 24], dx, 4, |p| [p[0] - 4e-3, 20e-3 - p[0], p[1] - 4e-3, 20e-3 - p[1], p[2] - 4e-3, 20e-3 - p[2]].into_iter().fold(f64::INFINITY, f64::min), Motion::still([0.0; 3], 1e-2, 1.0), 1e-4, 0.1, 0.05, vec![], vec![]);
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    s.advance(200).unwrap();
    let d = s.diagnostics().unwrap();
    assert!(d["u_max_lattice"].as_f64().unwrap() < 1e-6, "{d}");
    assert!(d["drho_mean"].as_f64().unwrap().abs() < 1e-6, "{d}");
}

fn pipe(r: f64, x0: f64, x1: f64, yc: f64, zc: f64) -> impl Fn([f64; 3]) -> f64 {
    move |p: [f64; 3]| {
        let rho = ((p[1] - yc).powi(2) + (p[2] - zc).powi(2)).sqrt();
        (r - rho).min(p[0] - x0).min(x1 - p[0])
    }
}

#[test]
fn poiseuille_pipe_profile_and_flux() {
    let dx = 1e-3;
    let (r, yc, zc) = (6e-3, 8e-3, 8e-3);
    let (dt, u_lat_mean) = (5e-4, 0.015);
    let um = u_lat_mean * dx / dt; // 0.03 m/s
    let q = um * std::f64::consts::PI * r * r;
    let nu = (0.1 / 3.0) * dx * dx / dt; // τ = 0.6 (the viscous pressure drop stays ≲ 2 % in density)
    let cuts = vec![
        CutSpec { name: "in".into(), center: [3e-3, yc, zc], normal: [-1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Inlet { q: Table::constant(q, 1.0) } },
        CutSpec { name: "out".into(), center: [53e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } },
    ];
    let c = mk_case([58, 17, 17], dx, 4, pipe(r, 0.5e-3, 56e-3, yc, zc), Motion::still([0.0; 3], 1e-2, 1.0), nu, 0.1, 0.05, vec![], cuts);
    let mut s = Solver::new_cpu(c).unwrap();
    assert!((s.u.dt - dt).abs() < 1e-9 && (s.u.tau0 - 0.6).abs() < 1e-6, "{:?}", s.u);
    s.init().unwrap();
    // diagnostics every 250 steps (they also drive the inlet flux controller, as in a run)
    for _ in 0..10 {
        s.advance(250).unwrap();
        s.diagnostics().unwrap();
    }
    let d = s.diagnostics().unwrap();
    // centre-line and mean over the cross-section at mid length
    let (uc, _) = mean_u(&mut s, |g| g[0] == 28 && (g[1] - 8).abs() <= 0 && (g[2] - 8).abs() <= 0);
    let (ua, na) = mean_u(&mut s, |g| g[0] == 28);
    let qout = d["cuts"][1]["q_out_ml_s"].as_f64().unwrap() * 1e-6;
    eprintln!("Poiseuille: centre {:.4} m/s (2 ū = {:.4}), area mean {:.4} over {na} nodes, outlet {:.3} vs inlet {:.3} mL/s; {d}", uc[0], 2.0 * um, ua[0], qout * 1e6, q * 1e6);
    // fully developed Poiseuille: centre = 2 × the section mean (the section mean itself carries the slight
    // compressibility of the pressure drop); outlet volume flux = inlet within the density ratio
    assert!((uc[0] / (2.0 * ua[0] * na as f64 / (std::f64::consts::PI * 36.0)) - 1.0).abs() < 0.08, "centre {} vs 2 × mean {}", uc[0], ua[0]);
    assert!((qout / q - 1.0).abs() < 0.05, "outlet {qout} vs inlet {q}");
    // wall shear stress from the accumulator against τ_w = 4 μ ū / R over the middle of the pipe (x 16–40 mm)
    for _ in 0..20 {
        s.advance(10).unwrap();
        s.be.accumulate(&s.step_params()).unwrap();
    }
    let acc = s.be.take_acc().unwrap();
    let (st, cpt) = (crate::params::ACC_STRIDE, s.cells);
    let c1 = 8 / s.case.run.out_factor;
    let (mut wsum, mut tsum, mut txs) = (0.0, 0.0, 0.0);
    for t in 0..s.topo.ntiles {
        for c in 0..cpt {
            let x = s.topo.txyz[3 * t] as usize * 8 + (c % c1) * s.case.run.out_factor;
            if !(16..=40).contains(&x) {
                continue;
            }
            let a = &acc[(t * cpt + c) * st..(t * cpt + c + 1) * st];
            wsum += a[10] as f64;
            tsum += a[14] as f64;
            txs += a[11] as f64;
        }
    }
    let mu = s.case.rho * s.case.nu;
    let want = 4.0 * mu * (ua[0] * na as f64 / (std::f64::consts::PI * 36.0)) / r;
    eprintln!("WSS: mean |τ_w| {:.3} Pa (x component {:.3}) over {wsum:.0} wall samples vs 4μū/R = {want:.3} Pa", tsum / wsum, txs / wsum);
    assert!(wsum > 100.0 && (tsum / wsum / want - 1.0).abs() < 0.25, "WSS {} vs {want}", tsum / wsum);
}

#[test]
fn translating_capsule_moves_with_its_walls() {
    // a closed capsule translating along x: d(t) = A sin(ωt); the fluid inside moves rigidly with it
    let dx = 1e-3;
    let (a, per) = (3e-3, 0.5);
    let w = 2.0 * std::f64::consts::PI / per;
    let motion = Motion::from_fn([24, 12, 12], [-6e-3, -6e-3, -6e-3], 4e-3, 64, per, move |_, t| [a * (w * t).sin(), 0.0, 0.0]);
    let sdf = move |p: [f64; 3]| {
        // capsule: axis x ∈ [14, 34] mm, radius 5 mm, centre (y, z) = (8, 8) mm
        let xc = p[0].clamp(14e-3, 34e-3);
        5e-3 - ((p[0] - xc).powi(2) + (p[1] - 8e-3).powi(2) + (p[2] - 8e-3).powi(2)).sqrt()
    };
    let nu = 0.1 * dx * dx / 5e-4;
    let c = mk_case([48, 17, 17], dx, 4, sdf, motion, nu, 0.1, 0.05, vec![], vec![]);
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    let spc = s.u.steps_per_cycle;
    s.advance(spc / 2).unwrap(); // t = T/2: wall velocity −A ω (the largest, reversed since t = 0)
    let t = s.time();
    let vw = a * w * (w * t).cos();
    let xc = 24.0 + a * (w * t).sin() / dx;
    let (u, n) = mean_u(&mut s, |g| ((g[0] as f64 - xc).abs() < 6.0) && (g[1] - 8).abs() <= 2 && (g[2] - 8).abs() <= 2);
    let d = s.diagnostics().unwrap();
    eprintln!("capsule at t {t:.3}: fluid {:.4} m/s over {n} nodes vs wall {vw:.4}; {d}", u[0]);
    assert!(n > 20);
    assert!((u[0] - vw).abs() < 0.08 * vw.abs(), "fluid {} vs wall {vw}", u[0]);
    assert!(u[1].abs() < 0.05 * vw.abs() && u[2].abs() < 0.05 * vw.abs());
}

#[test]
fn expanding_closed_tube_flux_grows_linearly() {
    // radial dilation (y, z) of a tube closed at x = x_c, open (pressure) at the far end: ū(x) = 2 Ṙ (x − x_c) / R
    let dx = 1e-3;
    let (r0, yc, zc, per, amp) = (6e-3, 10e-3, 10e-3, 0.4, 0.15);
    let motion = Motion::from_fn([16, 8, 8], [-8e-3, -6e-3, -6e-3], 5e-3, 96, per, move |p, t| {
        let a = amp * (std::f64::consts::PI * t / per).sin().powi(2);
        [0.0, a * (p[1] - yc), a * (p[2] - zc)]
    });
    let xcl = 2e-3;
    let cuts = vec![CutSpec { name: "open".into(), center: [44e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: 1.3 * r0, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } }];
    let c = mk_case([50, 21, 21], dx, 6, pipe(r0, xcl, 47e-3, yc, zc), motion, 0.1 * dx * dx / 3.3333e-4, 0.15, 0.05, vec![], cuts);
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    let spc = s.u.steps_per_cycle;
    s.advance(spc / 4).unwrap(); // t = T/4: largest dilation rate
    let t = s.time();
    let a = amp * (std::f64::consts::PI * t / per).sin().powi(2);
    let adot = amp * std::f64::consts::PI / per * (2.0 * std::f64::consts::PI * t / per).sin();
    let (r, rdot) = (r0 * (1.0 + a), r0 * adot);
    // the dilating tube draws fluid in through the open end: the flux in +x through the section at x is
    // Q(x) = −2πRṘ (x − x_c); measured as Σ u_x dx² over the section's nodes
    let mut worst: f64 = 0.0;
    for xm in [12.0, 20.0, 28.0, 36.0] {
        // flux = section-mean velocity × the true section area (the node count over-counts the staircase area)
        let (u, n) = mean_u(&mut s, |g| (g[0] as f64 - xm).abs() < 0.5);
        let q = u[0] * std::f64::consts::PI * r * r;
        let want = -2.0 * std::f64::consts::PI * r * rdot * (xm * dx - xcl);
        eprintln!("x {xm} mm: Q {:.3} vs {:.3} mL/s ({n} nodes, πR²/dx² = {:.1})", q * 1e6, want * 1e6, std::f64::consts::PI * r * r / (dx * dx));
        worst = worst.max((q / want - 1.0).abs());
    }
    assert!(worst < 0.10, "worst relative error {worst}");
}

fn valve_case(kind: ValveKind, open: f64) -> Case {
    let dx = 1e-3;
    let (r, yc, zc) = (7e-3, 10e-3, 10e-3);
    let v = ValveSpec {
        name: "v".into(),
        kind,
        center: [24e-3, yc, zc],
        normal: [1.0, 0.0, 0.0],
        e1: [0.0, 1.0, 0.0],
        radius: r,
        ring: 2e-3,
        thickness: 3e-3,
        theta_open: 75f64.to_radians(),
        opening: Table::constant(open, 1.0),
        tissue: true,
        trigger: crate::case::ValveTrigger { enabled: false, ..Default::default() },
        push: 0.0,
    };
    let cuts = vec![
        CutSpec { name: "in".into(), center: [3e-3, yc, zc], normal: [-1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 20.0 } },
        CutSpec { name: "out".into(), center: [45e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } },
    ];
    mk_case([50, 21, 21], dx, 4, pipe(r, 0.5e-3, 48e-3, yc, zc), Motion::still([0.0; 3], 1e-2, 1.0), 0.1 * dx * dx / 5e-4, 0.1, 0.05, vec![v], cuts)
}

#[test]
fn valve_geometry_closed_and_open() {
    for (kind, label) in [(ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) }, "tri"), (ValveKind::Leaflets { polys: crate::case::presets::bileaflet(0.2, 5, 7) }, "bi"), (ValveKind::Orifice { hole: [0.9, 0.9] }, "orifice")] {
        for (open, want_closed) in [(0.0, true), (1.0, false)] {
            let mut s = Solver::new_cpu(valve_case(kind.clone(), open)).unwrap();
            s.init().unwrap();
            let f = s.be.fields().unwrap();
            // the valve plane x = 24, nodes within the hinge circle
            let mut blocked = 0;
            let mut total = 0;
            let mut axis = 0.0;
            for k in 0..s.be.nodes() {
                let g = s.topo.coords(k);
                let rr = ((g[1] - 10).pow(2) + (g[2] - 10).pow(2)) as f64;
                if g[0] == 24 && rr <= 5.0f64.powi(2) && f.phi[k] > 0.0 {
                    total += 1;
                    if f.bsol[k] >= 0.5 {
                        blocked += 1;
                    }
                    if rr == 0.0 {
                        axis = f.bsol[k];
                    }
                }
            }
            eprintln!("{label} open {open}: {blocked}/{total} plane nodes blocked, axis B {axis}");
            if want_closed {
                assert_eq!(blocked, total, "{label}: closed valve leaks");
            } else {
                assert!(axis < 0.5 && (blocked as f64) < 0.45 * total as f64, "{label}: open valve still blocked ({blocked}/{total}, axis {axis})");
            }
        }
    }
}

#[test]
fn closed_valve_blocks_the_flow() {
    let mut q = vec![];
    for open in [0.0, 1.0] {
        let mut s = Solver::new_cpu(valve_case(ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) }, open)).unwrap();
        s.init().unwrap();
        s.advance(1000).unwrap();
        // time average over the last 500 steps (acoustic sloshing between the two pressure ends)
        let mut acc = 0.0;
        for _ in 0..10 {
            s.advance(50).unwrap();
            acc += s.diagnostics().unwrap()["cuts"][1]["q_out_ml_s"].as_f64().unwrap() / 10.0;
        }
        q.push(acc);
        eprintln!("valve open {open}: mean outlet {:.4} mL/s", acc);
    }
    assert!(q[1] > 0.0 && q[0].abs() < 0.05 * q[1], "closed {} vs open {}", q[0], q[1]);
}

/// LBM_CASE=<case dir> LBM_STEPS=n cargo test --release -p hasty-cfd probe_case -- --ignored --nocapture
#[test]
#[ignore]
fn probe_case() {
    let dir = std::env::var("LBM_CASE").expect("LBM_CASE");
    let steps: u64 = std::env::var("LBM_STEPS").map_or(2000, |s| s.parse().unwrap());
    let case = Case::load(std::path::Path::new(&dir)).unwrap();
    let mut s = Solver::new_cpu(case).unwrap();
    s.init().unwrap();
    let chunk = (steps / 4).max(1);
    for _ in 0..4 {
        s.advance(chunk).unwrap();
        let f = s.be.fields().unwrap();
        let n = s.be.nodes();
        let mut top: Vec<(f32, usize)> = (0..n).filter(|&k| f.flags[k] & 1 != 0).map(|k| ((f.ru[n + k].powi(2) + f.ru[2 * n + k].powi(2) + f.ru[3 * n + k].powi(2)).sqrt(), k)).collect();
        top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let sc = s.u.dx / s.u.dt;
        let pct = |q: f64| top[((top.len() as f64 * (1.0 - q)) as usize).min(top.len() - 1)].0 as f64 * sc;
        let wall_top = top.iter().take(200).filter(|&&(_, k)| f.flags[k] & 0x20 != 0).count();
        eprintln!("step {} t {:.1} ms: speed p99 {:.2} p99.9 {:.2} p99.99 {:.2} max {:.2} m/s; {wall_top}/200 fastest are wall nodes", s.step, s.time() * 1e3, pct(0.99), pct(0.999), pct(0.9999), top[0].0 as f64 * sc);
        let d = s.diagnostics().unwrap();
        let cs: Vec<String> = d["cuts"].as_array().unwrap().iter().map(|c| format!("{} {:.0}/{:.0}", &c["name"].as_str().unwrap()[..12.min(c["name"].as_str().unwrap().len())], c["q_out_ml_s"].as_f64().unwrap(), c["target_ml_s"].as_f64().unwrap_or(f64::NAN))).collect();
        eprintln!("   cuts measured/expected out mL/s: {}; Δρ̄ {:.2e}", cs.join(", "), d["drho_mean"].as_f64().unwrap());
        for &(sp, k) in top.iter().take(2) {
            let g = s.topo.coords(k);
            let x = [0, 1, 2].map(|a| s.case.grid.origin[a] + g[a] as f64 * s.u.dx);
            let near_v = s.case.valves.iter().map(|v| (v.name.clone(), ((0..3).map(|a| (x[a] - s.case.motion.forward(&v.center, s.time())[a]).powi(2)).sum::<f64>()).sqrt() * 1e3)).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            let near_c = s.case.cuts.iter().map(|c| (c.name.clone(), ((0..3).map(|a| (x[a] - s.case.motion.forward(&c.center, s.time())[a]).powi(2)).sum::<f64>()).sqrt() * 1e3)).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            eprintln!(
                "   |u| {:.3} ({:.2} m/s) ρ {:.3} flags {:#04x} φ {:.2} B {:.2} wall vel {:.3} | valve {} {:.1} mm, cut {} {:.1} mm",
                sp,
                sp as f64 * s.u.dx / s.u.dt,
                f.ru[k],
                f.flags[k],
                f.phi[k],
                f.bsol[k],
                (f.um[k].powi(2) + f.um[n + k].powi(2) + f.um[2 * n + k].powi(2)).sqrt(),
                near_v.0,
                near_v.1,
                &near_c.0[..near_c.0.len().min(24)],
                near_c.1
            );
        }
    }
}

/// the GPU kernels (NVRTC, --use_fast_math) against the CPU build of the same core: a dilating tube with a half-open
/// valve, an inlet and a pressure outlet, a few hundred steps (geometry updates, refill, Bouzidi, PSM, WALE, cuts,
/// accumulator). Needs a CUDA device:
///     cargo test --release -p hasty-cfd gpu_matches_cpu -- --ignored --nocapture
#[test]
#[ignore]
fn gpu_matches_cpu() {
    let dx = 1e-3;
    let (r0, yc, zc, per, amp) = (7e-3, 10e-3, 10e-3, 0.3, 0.12);
    let motion = Motion::from_fn([16, 8, 8], [-8e-3, -6e-3, -6e-3], 5e-3, 96, per, move |p, t| {
        let a = amp * (std::f64::consts::PI * t / per).sin().powi(2);
        [0.0, a * (p[1] - yc), a * (p[2] - zc)]
    });
    let mut c = valve_case(ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) }, 0.6);
    c.motion = motion;
    c.cuts[0].kind = CutKind::Inlet { q: Table::constant(0.5e-6, 1.0) };
    c.run.cwale = 0.5;
    c.run.uclip = 0.3;
    c.run.geom_every = 4;
    c.cuts[1].sponge = 4e-3;
    c.cuts[1].sponge_tau = 0.3;
    // cut planes off the lattice planes: membership is decided on the reference coordinate from the Newton inverse,
    // which rounds differently under --use_fast_math (an exact tie s = 0 flips between slab and layer)
    c.cuts[0].center[0] += 0.27e-3;
    c.cuts[1].center[0] -= 0.23e-3;
    c.sdf.data = {
        // the pipe for this radius
        let f = pipe(r0, 0.5e-3, 48e-3, yc, zc);
        let sd = c.sdf.dims;
        let mut d = vec![0f32; sd[0] * sd[1] * sd[2]];
        for z in 0..sd[2] {
            for y in 0..sd[1] {
                for x in 0..sd[0] {
                    let p = [c.sdf.origin[0] + (x as f64 + 0.5) * dx, c.sdf.origin[1] + (y as f64 + 0.5) * dx, c.sdf.origin[2] + (z as f64 + 0.5) * dx];
                    d[(z * sd[1] + y) * sd[0] + x] = f(p) as f32;
                }
            }
        }
        d
    };
    // spin history with a slab edge across the pipe (α 0 → 30° over z), TR 2 ms: the BFECC scalar on both backends
    c.spin = Some(crate::case::SpinSpec::along_z(2e-3, 1.0, 4e-3, 4e-3, [0.0f32, 10.0, 20.0, 30.0].iter().map(|a| a.to_radians()).collect(), vec![1.0; c.cuts.len()]));
    let dev = hasty_cuda::Device::new(0).expect("CUDA device");
    let mut a = Solver::new_cpu(c.clone()).unwrap();
    let mut b = Solver::new_gpu(c, &dev).unwrap();
    a.init().unwrap();
    b.init().unwrap();
    for round in 0..4 {
        a.advance(150).unwrap();
        b.advance(150).unwrap();
        a.be.accumulate(&a.step_params()).unwrap();
        b.be.accumulate(&b.step_params()).unwrap();
        let da = a.diagnostics().unwrap();
        let db = b.diagnostics().unwrap();
        let fa = a.be.fields().unwrap();
        let fb = b.be.fields().unwrap();
        let n = a.be.nodes();
        let flag_diff = (0..n).filter(|&k| fa.flags[k] != fb.flags[k]).count();
        // rounding flips a wall / refill threshold at an odd node (a local, physical-size difference there), so the
        // RMS difference is the measure and the largest one only a loose bound
        let (mut du, mut umax, mut s2, mut u2) = (0f64, 0f64, 0f64, 0f64);
        for k in 0..n {
            if fa.flags[k] & crate::params::F_FLUID == 0 {
                continue;
            }
            for c in 1..4 {
                let d = (fa.ru[c * n + k] - fb.ru[c * n + k]) as f64;
                du = du.max(d.abs());
                umax = umax.max(fa.ru[c * n + k].abs() as f64);
                s2 += d * d;
                u2 += (fa.ru[c * n + k] as f64).powi(2);
            }
        }
        let rel_rms = (s2 / u2.max(1e-30)).sqrt();
        let bdiff = (0..n).map(|k| (fa.bsol[k] - fb.bsol[k]).abs()).fold(0f32, f32::max);
        eprintln!("round {round}: flag mismatches {flag_diff}/{n}, relative RMS Δu {rel_rms:.2e}, max |Δu| {du:.2e} of max |u| {umax:.2e}, max |ΔB| {bdiff:.2e}\n   cpu {da}\n   gpu {db}");
        assert!(flag_diff <= n / 2000, "flags differ at {flag_diff} nodes");
        assert!(rel_rms < 2e-3 && du < 0.05 * umax.max(1e-3), "velocity differs: RMS {rel_rms}, max {du} vs {umax}");
        let (sa, sb) = (a.be.save_scalar().unwrap(), b.be.save_scalar().unwrap());
        let fl: Vec<usize> = (0..n).filter(|&k| fa.flags[k] & crate::params::F_FLUID != 0).collect();
        let ds = (fl.iter().map(|&k| ((sa[k] - sb[k]) as f64).powi(2)).sum::<f64>() / fl.len() as f64).sqrt();
        let smin = fl.iter().map(|&k| sa[k]).fold(1f32, f32::min);
        eprintln!("   spin: RMS Δs {ds:.2e}, min s {smin:.3}");
        assert!(ds < 2e-3 && smin < 0.9, "spin scalar differs: RMS {ds}, min {smin}");
    }
    let acc_a = a.be.take_acc().unwrap();
    let acc_b = b.be.take_acc().unwrap();
    let rel = acc_a.iter().zip(&acc_b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / acc_a.iter().map(|x| x.abs() as f64).fold(1e-12, f64::max);
    let tot: f64 = acc_a.iter().step_by(crate::params::ACC_STRIDE).map(|&x| x as f64).sum();
    eprintln!("accumulator max relative difference {rel:.2e} (Σw {tot:.0})");
    assert!(tot > 1000.0 && rel < 5e-3);
}

#[test]
fn pressure_trigger_opens_a_closed_valve() {
    // the closed valve of `closed_valve_blocks_the_flow` with 20 Pa across it: the trigger (p_open 5 Pa) opens it (the
    // filter outlasts the ±100 Pa acoustic sloshing between the two pressure ends, period ≈ 80 ms here)
    let mut q = vec![];
    for enabled in [false, true] {
        let mut c = valve_case(ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) }, 0.0);
        c.valves[0].trigger = crate::case::ValveTrigger { enabled, p_open: 5.0, p_close: 1.0, tau_open: 0.02, tau_close: 0.02, tau_filter: 0.2, window: f64::INFINITY };
        c.run.fb_every = 10;
        let mut s = Solver::new_cpu(c).unwrap();
        s.init().unwrap();
        s.advance(1000).unwrap();
        let mut acc = 0.0;
        let mut open = 0.0;
        for _ in 0..10 {
            s.advance(50).unwrap();
            let d = s.diagnostics().unwrap();
            acc += d["cuts"][1]["q_out_ml_s"].as_f64().unwrap() / 10.0;
            open = d["valves"][0]["open"].as_f64().unwrap();
            if std::env::var("LBM_VERBOSE").is_ok() {
                eprintln!("  {}", d["valves"][0]);
            }
        }
        eprintln!("trigger {enabled}: mean outlet {acc:.4} mL/s, opening {open:.2}");
        q.push((acc, open));
    }
    assert!(q[0].0.abs() < 0.02 && q[0].1 < 0.01, "disabled: {:?}", q[0]);
    assert!(q[1].0 > 0.3 && q[1].1 > 0.9, "enabled: {:?}", q[1]);
}

/// a valve opening in a sealed tube (no cuts, no motion): the fluid mass Σρ over the fluid nodes must not jump while the
/// closed disk dissolves and the leaflets swing open; per half (upstream / downstream of the plane) the mass exchange
/// is reported. LBM_VERBOSE=1 prints the time series
#[test]
fn opening_valve_conserves_mass() {
    let kind = ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) };
    let mut c = valve_case(kind, 0.0);
    c.cuts.clear();
    c.valves[0].push = std::env::var("HASTY_CFD_PUSH").map_or(0.0, |x| x.parse().unwrap());
    // heart-like lattice speed (c_s ≈ 11 m/s: the leaflets swing at Ma ≈ 0.02), opening 0 → 1 over 30 ms at 50 ms
    c.v_max = 1.0;
    c.valves[0].opening = Table { t: vec![0.0, 0.05, 0.08, 0.6, 0.63], v: vec![0.0, 0.0, 1.0, 1.0, 0.0], period: 1.0 };
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    // mean density deviation per half (upstream / downstream of the plane x = 24): a pressure difference, mmHg-free
    let dens = |s: &mut Solver<crate::backend::CpuBackend>| {
        let f = s.be.fields().unwrap();
        let n = s.be.nodes();
        let mut a = [(0.0, 0.0); 2];
        for k in 0..n {
            if f.flags[k] & crate::params::F_FLUID != 0 {
                let w = 1.0 - f.bsol[k] as f64;
                let h = (s.topo.coords(k)[0] >= 24) as usize;
                a[h].0 += w * (f.ru[k] as f64 - 1.0);
                a[h].1 += w;
            }
        }
        [a[0].0 / a[0].1, a[1].0 / a[1].1]
    };
    let spc = (0.002 / s.u.dt).round() as u64;
    let mut worst: f64 = 0.0;
    while s.time() < 0.12 {
        s.advance(spc).unwrap();
        let d = dens(&mut s);
        worst = worst.max(d[0].abs()).max(d[1].abs());
        if std::env::var("LBM_VERBOSE").is_ok() {
            eprintln!("t {:.3} opening {:.2}: Δρ up {:+.2e} down {:+.2e}", s.time(), s.case.valves[0].opening.at(s.time()), d[0], d[1]);
        }
    }
    // Δρ 1e-3 is ≈ 0.4 mmHg at this lattice speed
    eprintln!("opening valve: worst mean density deviation per half {worst:.2e}");
    assert!(worst < 2e-3, "density deviation {worst}");
}

#[test]
fn spin_history_saturates_along_a_pipe() {
    // the Poiseuille pipe in a uniform 10° slab, TR 5 ms, T1 1.6 s, fresh inflow: at steady state the centre line's
    // s = M_z / M0 follows the exact transit solution s(x / u_c)
    let dx = 1e-3;
    let (r, yc, zc) = (6e-3, 8e-3, 8e-3);
    let (dt, u_lat_mean) = (5e-4, 0.015);
    let um = u_lat_mean * dx / dt;
    let q = um * std::f64::consts::PI * r * r;
    let nu = (0.1 / 3.0) * dx * dx / dt;
    let cuts = vec![
        CutSpec { name: "in".into(), center: [3e-3, yc, zc], normal: [-1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Inlet { q: Table::constant(q, 1.0) } },
        CutSpec { name: "out".into(), center: [53e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } },
    ];
    let mut c = mk_case([58, 17, 17], dx, 4, pipe(r, 0.5e-3, 56e-3, yc, zc), Motion::still([0.0; 3], 1e-2, 1.0), nu, 0.1, 0.05, vec![], cuts);
    // a recirculating loop: the inflow is the outflow after 0.4 s outside the slab (T1 recovery: the `from` path)
    c.cuts[0].upstream = Some(crate::case::Upstream { points: vec![[0.0, 0.0, 5.0]], dt: vec![0.4], from: vec!["out".into()] });
    let spin = crate::case::SpinSpec::along_z(5e-3, 1.6, -1.0, 2.0, vec![10f32.to_radians(); 2], vec![1.0, 1.0]);
    c.spin = Some(spin.clone());
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    for _ in 0..40 {
        s.advance(250).unwrap();
        s.diagnostics().unwrap();
    }
    let f = s.be.fields().unwrap();
    let sc = s.be.save_scalar().unwrap();
    let n = s.be.nodes();
    // the centre line is a streamline: transit time Σ Δx / u from the last inlet-slab node (s held at exactly 1); the
    // developing entrance profile makes u_c rise over the first diameter, so a constant-u_c transit is wrong
    let line: Vec<(f64, f64)> = (0..58)
        .filter_map(|x| (0..n).find(|&k| s.topo.coords(k) == [x, 8, 8]).map(|k| (f.ru[n + k] as f64 * dx / dt, sc[k] as f64)))
        .collect();
    let s_in = s.s_in[0];
    // the slab nodes hold the s_in of the last cut upload (s_in moves slowly with the loop's average)
    let x0 = (0..8).filter(|&i| line[i].0 > 1e-6 && (line[i].1 - s_in).abs() < 1e-3).max().unwrap_or_else(|| panic!("no inlet slab: s_in {s_in}, line {:?}", &line[..8]));
    let mut worst: f64 = 0.0;
    let mut t = 0.0;
    for i in x0 + 1..line.len() - 6 {
        t += 0.5 * dx * (1.0 / line[i - 1].0.max(1e-6) + 1.0 / line[i].0);
        if (i - x0) % 10 == 0 {
            let want = spin.along_from(s_in, &[0.0], &[t]);
            eprintln!("x {i} mm: s {:.3} vs {:.3} (u_c {:.3} m/s, t {:.2} s)", line[i].1, want, line[i].0, t);
            worst = worst.max((line[i].1 / want - 1.0).abs());
        }
    }
    // the loop's fixed point: the flux-weighted outflow s carried 0.3 s further
    let (a, b) = (s.s_flux_of(1)[0], s.s_flux_of(1)[1]);
    let loop_s = spin.along_from(a / b, &[5.0], &[0.4]);
    eprintln!("inlet slab ends at x {x0} mm; s_in {s_in:.4}, outflow s {:.4} → {loop_s:.4} after the loop", a / b);
    assert!((loop_s - s_in).abs() < 0.01 && s_in < 0.7, "loop s_in {s_in} vs {loop_s}");
    assert!(worst < 0.05, "centre-line s off by {worst}");
}
