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
        transpiration: None,
        wall_pump: vec![],
        refine: vec![],
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
    // HASTY_CFD_PIPE_R (mm): another radius for the flux-layer study (the asserts below are for the default 6 mm)
    let r = std::env::var("HASTY_CFD_PIPE_R").map_or(6e-3, |x| x.parse::<f64>().unwrap() * 1e-3);
    let (yc, zc) = (8e-3, 8e-3);
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
    // the section flux node by node along the pipe (the inlet's flux layer is the first interior layer, x = 3 mm)
    let mut prof = String::new();
    for x in [2, 3, 4, 5, 6, 8, 12, 28, 44, 50, 52, 53] {
        let (u_, n_) = mean_u(&mut s, |g| g[0] == x);
        prof += &format!(" x{x}: {:.3}", u_[0] * n_ as f64 * dx * dx * 1e6);
    }
    eprintln!("section flux along the pipe [mL/s]:{prof}");
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
fn moving_pipe_cut_flux() {
    // the cut flux layers of a pipe that moves with the tissue (research/flow_phantom.md: in the moving heart the
    // layers read 5–8 % below the binned field): a constant inflow through a pipe translating sinusoidally across
    // (HASTY_CFD_MOVE_AXIS=1, default) or along (0) its axis; over the last cycle the inlet / outlet layer fluxes
    // against the section flux at mid length (u_x summed over the section, fixed in the lab: the axial motion's
    // wall velocity averages out over a cycle)
    let axis: usize = std::env::var("HASTY_CFD_MOVE_AXIS").map_or(1, |x| x.parse().unwrap());
    let dx = 1e-3;
    let (r, yc, zc) = (4e-3, 10e-3, 10e-3);
    let (amp, per) = (2e-3, 0.25);
    let w = 2.0 * std::f64::consts::PI / per;
    let um = 0.03;
    let q = um * std::f64::consts::PI * r * r;
    let nu = (0.1 / 3.0) * dx * dx / 5e-4;
    let motion = Motion::from_fn([17, 9, 9], [-6e-3, -6e-3, -6e-3], 4e-3, 64, per, move |_, t| {
        let mut d = [0.0; 3];
        d[axis] = amp * (w * t).sin();
        d
    });
    let cuts = vec![
        CutSpec { name: "in".into(), center: [5e-3, yc, zc], normal: [-1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Inlet { q: Table::constant(q, per) } },
        CutSpec { name: "out".into(), center: [51e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } },
    ];
    let c = mk_case([58, 21, 21], dx, 4, pipe(r, 0.5e-3, 56e-3, yc, zc), motion, nu, 0.1, 0.05, vec![], cuts);
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    let spc = s.u.steps_per_cycle;
    for _ in 0..3 * spc / 25 {
        s.advance(25).unwrap();
        s.diagnostics().unwrap();
    }
    let (mut qi, mut qo, mut qs, mut n) = (0.0, 0.0, 0.0, 0.0);
    let (mut li, mut lo) = (0.0, 0.0);
    for _ in 0..spc / 25 {
        s.advance(25).unwrap();
        let d = s.diagnostics().unwrap();
        qi += -d["cuts"][0]["q_out_ml_s"].as_f64().unwrap();
        qo += d["cuts"][1]["q_out_ml_s"].as_f64().unwrap();
        li += -d["cuts"][0]["q_link_ml_s"].as_f64().unwrap();
        lo += d["cuts"][1]["q_link_ml_s"].as_f64().unwrap();
        let (u_, k_) = mean_u(&mut s, |g| g[0] == 28);
        qs += u_[0] * k_ as f64 * dx * dx * 1e6;
        n += 1.0;
    }
    let (qi, qo, qs) = (qi / n, qo / n, qs / n);
    let (li, lo) = (li / n, lo / n);
    eprintln!("moving pipe (axis {axis}): cycle means inlet layer {qi:.4} / link {li:.4}, section x 28 {qs:.4}, outlet layer {qo:.4} / link {lo:.4} mL/s (prescribed {:.4})", q * 1e6);
    // the inlet controller steers the exact link flux (RunSpec::kappa_link): it and the section flux near mid length
    // carry the prescribed flow; the u-based layer reads a few % low
    assert!((li / (q * 1e6) - 1.0).abs() < 0.02, "inlet link flux {li} vs prescribed {}", q * 1e6);
    assert!((qs / (q * 1e6) - 1.0).abs() < 0.03, "section flux {qs} vs prescribed {}", q * 1e6);
}

#[test]
fn contracting_chamber_pumps_its_volume_change() {
    // a sphere (R 8 mm) with an outlet pipe (r 3 mm), the whole domain scaled about the sphere centre by s(t) =
    // 1 − ε (1 − cos ωt) / 2: the outflow through the (moving) outlet cut must be −dV/dt of the domain upstream of it,
    // V(t) = V0 s(t)³ — the share of the wall motion the moving walls turn into flow (research/flow_phantom.md: the
    // heart's chambers delivered ≈ 0.9 of their stroke to the cuts). HASTY_CFD_PUMP_EPS (default 0.08), _PER (0.4 s)
    let dx = 1e-3;
    let eps: f64 = std::env::var("HASTY_CFD_PUMP_EPS").map_or(0.08, |x| x.parse().unwrap());
    let per: f64 = std::env::var("HASTY_CFD_PUMP_PER").map_or(0.4, |x| x.parse().unwrap());
    let w = 2.0 * std::f64::consts::PI / per;
    let (c0, rs, rp, x1) = ([14e-3, 12e-3, 12e-3], 8e-3, 3e-3, 40e-3);
    let sdf = move |p: [f64; 3]| {
        let ds = rs - ((p[0] - c0[0]).powi(2) + (p[1] - c0[1]).powi(2) + (p[2] - c0[2]).powi(2)).sqrt();
        let dp = (rp - ((p[1] - c0[1]).powi(2) + (p[2] - c0[2]).powi(2)).sqrt()).min(x1 - p[0]).min(p[0] - c0[0]);
        ds.max(dp)
    };
    let motion = Motion::from_fn([14, 8, 8], [-6e-3, -6e-3, -6e-3], 4e-3, 64, per, move |x, t| {
        let sc = -eps * (1.0 - (w * t).cos()) / 2.0;
        [sc * (x[0] - c0[0]), sc * (x[1] - c0[1]), sc * (x[2] - c0[2])]
    });
    let xc = 34e-3;
    let cuts = vec![CutSpec { name: "out".into(), center: [xc, c0[1], c0[2]], normal: [1.0, 0.0, 0.0], radius: rp, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } }];
    let nu = (0.1 / 3.0) * dx * dx / 2e-4;
    let vmax: f64 = std::env::var("HASTY_CFD_PUMP_VMAX").map_or(1.0, |x| x.parse().unwrap()); // c_s 11.5 m/s at 1.0 (heart cases: 14)
    let mut c = mk_case([46, 25, 25], dx, 4, sdf, motion, nu * vmax / 0.25, vmax, 0.05, vec![], cuts);
    c.run.wall_mass = std::env::var("HASTY_CFD_WALL_MASS").is_ok();
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    let spc = s.u.steps_per_cycle;
    // reference volume upstream of the cut: the sphere + the pipe from its junction to the cut (h = the cap height)
    let hc = rs - (rs * rs - rp * rp).sqrt();
    let v0 = 4.0 / 3.0 * std::f64::consts::PI * rs.powi(3) + std::f64::consts::PI * rp * rp * (xc - c0[0] - (rs - hc)) - std::f64::consts::PI * hc * hc * (3.0 * rp - hc) / 3.0 * 0.0;
    for _ in 0..spc / 25 {
        s.advance(25).unwrap();
        s.diagnostics().unwrap();
    }
    // over one cycle: the outflow while contracting (first half) against the volume expelled
    // over one cycle: the first harmonic of the outflow against that of −dV/dt (the weakly compressible lattice delays
    // the outflow, its amplitude is the pumped share)
    let (mut hq, mut hw) = ((0.0f64, 0.0f64), (0.0f64, 0.0f64));
    let mut series = vec![];
    for _ in 0..spc / 25 {
        s.advance(25).unwrap();
        let d = s.diagnostics().unwrap();
        let t = s.time();
        let sc = 1.0 - eps * (1.0 - (w * t).cos()) / 2.0;
        let dsdt = -eps * w * (w * t).sin() / 2.0;
        let want = -v0 * 3.0 * sc * sc * dsdt; // −dV/dt, outward positive
        let ql = d["cuts"][0]["q_link_ml_s"].as_f64().unwrap() * 1e-6;
        let (cw, sw) = ((w * t).cos(), (w * t).sin());
        hq = (hq.0 + ql * cw, hq.1 + ql * sw);
        hw = (hw.0 + want * cw, hw.1 + want * sw);
        series.push(format!("{:.2}/{:.2}", ql * 1e6, want * 1e6));
    }
    let (aq, aw) = ((hq.0 * hq.0 + hq.1 * hq.1).sqrt(), (hw.0 * hw.0 + hw.1 * hw.1).sqrt());
    let lag = (hq.1.atan2(hq.0) - hw.1.atan2(hw.0)).to_degrees();
    eprintln!("contracting chamber (ε {eps}, T {per} s, {} steps/cycle, c_s {:.1} m/s): first-harmonic outflow / −dV/dt = {:.3}, lag {lag:.0}°", spc, s.u.dx / s.u.dt / 3f64.sqrt(), aq / aw);
    eprintln!("   link / want per 25 steps: {}", series.iter().step_by(2).cloned().collect::<Vec<_>>().join(" "));
    assert!((aq / aw - 1.0).abs() < 0.05, "the chamber pumps {} of its volume change", aq / aw);
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

#[test]
fn transpiring_closed_tube_flux_grows_linearly() {
    // a rigid tube closed at x = x_c, open (pressure) at the far end, whose side wall transpires outward at u0 over
    // x_s ≤ x ≤ x_e (case.rs Transpiration; a constant q_w and a constant weight w = u0 / q_w): the wall removes
    // 2πR u0 per unit length, so the flux in +x is Q(x) = −2πR u0 (x − x_s) inside the band — the dilating tube's law
    // with Ṙ = u0, without moving geometry
    let dx = 1e-3;
    let (r0, yc, zc) = (6e-3, 10e-3, 10e-3);
    let motion = Motion::from_fn([16, 8, 8], [-8e-3, -6e-3, -6e-3], 5e-3, 8, 0.4, |_, _| [0.0, 0.0, 0.0]);
    let xcl = 2e-3;
    let (xs, xe) = (6e-3, 38e-3);
    let cuts = vec![CutSpec { name: "open".into(), center: [44e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: 1.3 * r0, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } }];
    let mut c = mk_case([50, 21, 21], dx, 6, pipe(r0, xcl, 47e-3, yc, zc), motion, 0.1 * dx * dx / 3.3333e-4, 0.15, 0.05, vec![], cuts);
    let (q0, u0) = (1e-6, 0.01);
    let sd = c.sdf.dims;
    let mut group = vec![0u8; c.sdf.data.len()];
    let mut weight = vec![0f32; c.sdf.data.len()];
    for z in 0..sd[2] {
        for y in 0..sd[1] {
            for x in 0..sd[0] {
                let px = c.sdf.origin[0] + (x as f64 + 0.5) * dx;
                let i = (z * sd[1] + y) * sd[0] + x;
                if px >= xs && px <= xe && (c.sdf.data[i] as f64).abs() <= 2.5 * dx {
                    group[i] = 1;
                    weight[i] = (u0 / q0) as f32;
                }
            }
        }
    }
    c.transpiration = Some(crate::case::Transpiration { band: 2.0, group, weight, groups: vec![(1, "tube".into(), Table::constant(q0, 0.4))] });
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    s.advance(900).unwrap();
    let mut worst: f64 = 0.0;
    for xm in [14.0, 22.0, 30.0] {
        let (u, n) = mean_u(&mut s, |g| (g[0] as f64 - xm).abs() < 0.5);
        let q = u[0] * std::f64::consts::PI * r0 * r0;
        let want = -2.0 * std::f64::consts::PI * r0 * u0 * (xm * dx - xs);
        eprintln!("x {xm} mm: Q {:.4} vs {:.4} mL/s ({n} nodes)", q * 1e6, want * 1e6);
        worst = worst.max((q / want - 1.0).abs());
    }
    // beyond the band the flux stays at the band's total
    let (u, _) = mean_u(&mut s, |g| (g[0] as f64 - 41.0).abs() < 0.5);
    let q_out = u[0] * std::f64::consts::PI * r0 * r0;
    let want_out = -2.0 * std::f64::consts::PI * r0 * u0 * (xe - xs);
    eprintln!("x 41 mm (past the band): Q {:.4} vs {:.4} mL/s", q_out * 1e6, want_out * 1e6);
    worst = worst.max((q_out / want_out - 1.0).abs());
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
        hole: None,
        membrane: std::env::var("HASTY_CFD_MEMBRANE").is_ok(),
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
fn closed_valve_with_a_regurgitant_hole() {
    // a closed bileaflet valve with a slit on its coaptation line (valve disease): open inside the ellipse, sealed
    // elsewhere (e1 = y, e2 = n × e1 = z; R 7 mm, hole centre (0, 0.2 R), semi-axes 0.45 R × 0.2 R)
    let mut c = valve_case(ValveKind::Leaflets { polys: crate::case::presets::bileaflet(0.2, 5, 7) }, 0.0);
    c.valves[0].hole = Some([0.0, 0.2, 0.45, 0.2]);
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    let f = s.be.fields().unwrap();
    let (mut in_open, mut in_tot, mut out_blk, mut out_tot) = (0, 0, 0, 0);
    for k in 0..s.be.nodes() {
        let g = s.topo.coords(k);
        if g[0] != 24 || f.phi[k] <= 0.0 {
            continue;
        }
        let (x, y) = ((g[1] - 10) as f64 / 7.0, (g[2] - 10) as f64 / 7.0);
        let re = ((x / 0.45).powi(2) + ((y - 0.2) / 0.2).powi(2)).sqrt();
        if re < 0.6 {
            in_tot += 1;
            in_open += (f.bsol[k] < 0.5) as i32;
        } else if re > 1.6 && x * x + y * y <= (5.0f64 / 7.0).powi(2) {
            out_tot += 1;
            out_blk += (f.bsol[k] >= 0.5) as i32;
        }
    }
    eprintln!("regurgitant hole: {in_open}/{in_tot} nodes open inside, {out_blk}/{out_tot} blocked outside");
    assert!(in_tot > 0 && in_open == in_tot, "hole not open");
    assert!(out_tot > 0 && out_blk == out_tot, "closed valve leaks outside the hole");
}

/// the moving sheet pumps 0.981 of its swept volume, 0.998 with RunSpec::wall_mass (HASTY_CFD_WALL_MASS=1; research/
/// flow_phantom.md); a Gaussian bump instead of the flat top moves the sheet's faces only 0.92 as far as its centre
#[test]
fn moving_closed_valve_pumps_like_a_piston() {
    // a closed tissue-frame valve carried along the static tube by an axial displacement bump around it (the tube walls
    // are axially invariant, so only the valve moves): the outflow must be the disk area × the sheet's velocity — the
    // AV-plane descent is the ventricles' main pump, and a sheet that advances by covering nodes on one side and
    // refilling them on the other moves mass across without pushing it (research/flow_phantom.md). First harmonics over
    // the last cycle; HASTY_CFD_PISTON_VMAX (default 1.0 m/s: c_s 11.5 m/s)
    let vmax: f64 = std::env::var("HASTY_CFD_PISTON_VMAX").map_or(1.0, |x| x.parse().unwrap());
    let (amp, per) = (3e-3, 0.4);
    let w = 2.0 * std::f64::consts::PI / per;
    let mut c = valve_case(ValveKind::Leaflets { polys: crate::case::presets::bileaflet(0.2, 5, 7) }, 0.0);
    // flat-top bump: the valve region (±6 mm, the closed sheet ±2 mm) moves rigidly, a cosine taper to zero by ±12 mm,
    // the cuts (21 mm away) stay put (a Gaussian bump moved the sheet's faces only 0.92 as far as its centre)
    let bump = |x: f64| {
        let d = (x - 24e-3).abs();
        if d <= 6e-3 { 1.0 } else if d >= 12e-3 { 0.0 } else { 0.5 * (1.0 + (std::f64::consts::PI * (d - 6e-3) / 6e-3).cos()) }
    };
    c.motion = Motion::from_fn([32, 18, 18], [-6e-3, -6e-3, -6e-3], 2e-3, 64, per, move |x, t| [amp * (w * t).sin() * bump(x[0]), 0.0, 0.0]);
    let dx = c.grid.dx;
    c.v_max = vmax;
    c.nu = 0.1 * dx * dx / 5e-4 * vmax / 0.1;
    if let CutKind::Pressure { p } = &mut c.cuts[0].kind {
        *p = 0.0;
    }
    c.run.valve_q = std::env::var("HASTY_CFD_VQ").is_ok();
    // HASTY_CFD_WALL_MASS: the wall operations mass-neutral per region; the regions are the two sides of the valve
    // plane (reference pose, the sdf grid)
    if std::env::var("HASTY_CFD_WALL_MASS").is_ok() {
        c.run.wall_mass = true;
        let (sd, so, h) = (c.sdf.dims, c.sdf.origin, c.sdf.h);
        let mut map = vec![0u8; sd[0] * sd[1] * sd[2]];
        for z in 0..sd[2] {
            for y in 0..sd[1] {
                for x in 0..sd[0] {
                    let px = so[0] + (x as f64 + 0.5) * h;
                    map[(z * sd[1] + y) * sd[0] + x] = if (px < 24e-3) != std::env::var("HASTY_CFD_WM_SWAP").is_ok() { 1 } else { 2 };
                }
            }
        }
        c.regions = Some(crate::case::Regions { names: [(1u16, "blood_up".to_string()), (2u16, "blood_down".to_string())].into_iter().collect(), map, ..Default::default() });
    }
    let r = c.valves[0].radius;
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    let spc = s.u.steps_per_cycle;
    for _ in 0..spc / 25 {
        s.advance(25).unwrap();
        s.diagnostics().unwrap();
    }
    let (mut ho, mut hi, mut hw) = ((0.0f64, 0.0f64), (0.0f64, 0.0f64), (0.0f64, 0.0f64));
    for _ in 0..spc / 25 {
        s.advance(25).unwrap();
        let d = s.diagnostics().unwrap();
        let t = s.time();
        let want = std::f64::consts::PI * r * r * amp * w * (w * t).cos(); // the swept volume rate, + along x (out)
        let qo = d["cuts"][1]["q_link_ml_s"].as_f64().unwrap() * 1e-6;
        let qi = -d["cuts"][0]["q_link_ml_s"].as_f64().unwrap() * 1e-6;
        let (cw, sw) = ((w * t).cos(), (w * t).sin());
        ho = (ho.0 + qo * cw, ho.1 + qo * sw);
        hi = (hi.0 + qi * cw, hi.1 + qi * sw);
        hw = (hw.0 + want * cw, hw.1 + want * sw);
    }
    let a = |h: (f64, f64)| (h.0 * h.0 + h.1 * h.1).sqrt();
    eprintln!("moving closed valve (R {:.0} mm, ±{:.0} mm, T {per} s, c_s {:.1} m/s): first harmonic outlet {:.3} / inlet {:.3} of the swept volume rate", r * 1e3, amp * 1e3, s.u.dx / s.u.dt / 3f64.sqrt(), a(ho) / a(hw), a(hi) / a(hw));
    assert!((a(ho) / a(hw) - 1.0).abs() < 0.03 && (a(hi) / a(hw) - 1.0).abs() < 0.03, "the moving valve pumps {} / {} of its swept volume", a(ho) / a(hw), a(hi) / a(hw));
}

#[test]
fn closed_valve_blocks_the_flow() {
    let mut q = vec![];
    for open in [0.0, 1.0] {
        let mut s = Solver::new_cpu(valve_case(ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) }, open)).unwrap();
        s.init().unwrap();
        s.advance(1000).unwrap();
        // time average over the last 500 steps (acoustic sloshing between the two pressure ends)
        let (mut acc, mut qv, mut ql) = (0.0, 0.0, 0.0);
        for _ in 0..10 {
            s.advance(50).unwrap();
            let d = s.diagnostics().unwrap();
            acc += d["cuts"][1]["q_out_ml_s"].as_f64().unwrap() / 10.0;
            ql += d["cuts"][1]["q_link_ml_s"].as_f64().unwrap() / 10.0;
            qv += d["valves"][0]["q_link_ml_s"].as_f64().unwrap() / 10.0;
        }
        q.push(acc);
        eprintln!("valve open {open}: mean outlet {acc:.4} (link {ql:.4}) mL/s, through the valve plane (link) {qv:.4} mL/s");
        // the exact valve-plane flux carries the outlet's flow (still tube: no plane motion)
        assert!((qv - ql).abs() < 0.03 * ql.abs().max(0.05), "valve plane {qv} vs outlet {ql}");
    }
    assert!(q[1] > 0.0 && q[0].abs() < 0.05 * q[1], "closed {} vs open {}", q[0], q[1]);
}

#[test]
fn membrane_valve_blocks_and_passes() {
    // zero-thickness membrane leaflets: every link crossing a leaflet or the closed disk bounces back, so the closed
    // valve seals exactly whatever the thickness; open, the valve-plane link flux carries the outlet's flow
    for (polys, label) in [(crate::case::presets::trileaflet(90.0, 4), "tri"), (crate::case::presets::bileaflet(0.2, 5, 7), "bi")] {
        let mut q = vec![];
        for open in [0.0, 1.0] {
            let mut c = valve_case(ValveKind::Leaflets { polys: polys.clone() }, open);
            c.valves[0].membrane = true;
            let mut s = Solver::new_cpu(c).unwrap();
            s.init().unwrap();
            s.advance(1000).unwrap();
            let (mut acc, mut qv, mut ql) = (0.0, 0.0, 0.0);
            for _ in 0..10 {
                s.advance(50).unwrap();
                let d = s.diagnostics().unwrap();
                acc += d["cuts"][1]["q_out_ml_s"].as_f64().unwrap() / 10.0;
                ql += d["cuts"][1]["q_link_ml_s"].as_f64().unwrap() / 10.0;
                qv += d["valves"][0]["q_link_ml_s"].as_f64().unwrap() / 10.0;
            }
            // every node stays fluid (no leaflet solid): the membrane is links only
            let f = s.be.fields().unwrap();
            let solid_in_disk = (0..s.be.nodes()).filter(|&k| { let g = s.topo.coords(k); f.phi[k] > 0.0 && f.flags[k] & crate::params::F_FLUID == 0 && ((g[1] - 10).pow(2) + (g[2] - 10).pow(2)) <= 16 }).count();
            eprintln!("{label} membrane open {open}: outlet {acc:.4} (link {ql:.4}) mL/s, valve plane {qv:.4} mL/s, {solid_in_disk} solid nodes within 4 mm of the axis");
            assert_eq!(solid_in_disk, 0, "{label}: membrane leaflets left solid nodes");
            assert!((qv - ql).abs() < 0.03 * ql.abs().max(0.05), "{label}: valve plane {qv} vs outlet {ql}");
            q.push(acc);
        }
        assert!(q[1] > 0.0 && q[0].abs() < 0.05 * q[1], "{label}: closed {} vs open {}", q[0], q[1]);
    }
}

/// steady flow through a fully open valve: the peak axial velocity one radius downstream over the mean (Q / A), plates
/// (partially saturated, `thickness` thick) against membranes; HASTY_CFD_OPEN_DEG (default 90)
#[test]
fn open_valve_profile_plates_vs_membrane() {
    let deg: f64 = std::env::var("HASTY_CFD_OPEN_DEG").map_or(90.0, |x| x.parse().unwrap());
    let mut ratio = vec![];
    for membrane in [false, true] {
        let mut c = valve_case(ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) }, 1.0);
        c.valves[0].membrane = membrane;
        c.valves[0].theta_open = deg.to_radians();
        let mut s = Solver::new_cpu(c).unwrap();
        s.init().unwrap();
        s.advance(1500).unwrap();
        let (mut um, mut ux) = (0.0, 0.0);
        for _ in 0..10 {
            s.advance(50).unwrap();
            // the plane x = 31 (one radius past the valve at x = 24): mean and peak axial velocity over its fluid nodes
            let (u, cnt) = mean_u(&mut s, |g| g[0] == 31);
            let f = s.be.fields().unwrap();
            let sc = s.u.dx / s.u.dt;
            let mut mx = 0.0f64;
            for k in 0..s.be.nodes() {
                if s.topo.coords(k)[0] == 31 && f.flags[k] & crate::params::F_FLUID != 0 {
                    mx = mx.max(f.ru[s.be.nodes() + k] as f64 * sc);
                }
            }
            um += u[0] / 10.0;
            ux += mx / 10.0;
            let _ = cnt;
        }
        eprintln!("open valve at {deg}° ({}): mean axial {um:.4} m/s, peak {ux:.4} m/s, peak / mean {:.3}", if membrane { "membrane" } else { "plates" }, ux / um);
        ratio.push(ux / um);
    }
    assert!(ratio[1] <= ratio[0] + 0.02, "membrane profile {} not flatter than plates {}", ratio[1], ratio[0]);
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
    c.run.wall_mass = std::env::var("HASTY_CFD_WALL_MASS").is_ok();
    c.motion = motion;
    c.cuts[0].kind = CutKind::Inlet { q: Table::constant(0.5e-6, 1.0) };
    c.run.cwale = std::env::var("HASTY_CFD_TEST_CWALE").map_or(0.5, |x| x.parse().unwrap());
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
    // wall transpiration over x 10–30 mm with a time-varying q_w (the cofactor path under the dilation, both backends)
    {
        let sd = c.sdf.dims;
        let mut group = vec![0u8; c.sdf.data.len()];
        let mut weight = vec![0f32; c.sdf.data.len()];
        for z in 0..sd[2] {
            for y in 0..sd[1] {
                for x in 0..sd[0] {
                    let px = c.sdf.origin[0] + (x as f64 + 0.5) * dx;
                    let i = (z * sd[1] + y) * sd[0] + x;
                    if (10e-3..=30e-3).contains(&px) && (c.sdf.data[i] as f64).abs() <= 2.5 * dx {
                        group[i] = 1;
                        weight[i] = if std::env::var("HASTY_TEST_TW0").is_ok() { 0.0 } else { 1.5e4 };
                    }
                }
            }
        }
        c.transpiration = Some(crate::case::Transpiration { band: 2.0, group, weight, groups: vec![(1, "t".into(), Table { t: vec![0.0, 0.15], v: vec![0.3e-6, -0.3e-6], period: per })] });
    }
    // spin history with a slab edge across the pipe (α 0 → 30° over z), TR 2 ms: the BFECC scalar on both backends
    c.spin = Some(crate::case::SpinSpec::along_z(2e-3, 1.0, 4e-3, 4e-3, [0.0f32, 10.0, 20.0, 30.0].iter().map(|a| a.to_radians()).collect(), vec![1.0; c.cuts.len()]));
    // HASTY_CFD_REFINE: the valve region refined to 0.5 mm (x 16–32 mm; the interface crosses the moving walls, the
    // transpiration band and the spin slab), "layers" for the per-layer transfer
    if let Ok(m) = std::env::var("HASTY_CFD_REFINE") {
        c.refine = refine_box([16e-3, -1.0, -1.0], [32e-3, 1.0, 1.0]);
        if m == "layers" {
            c.run.refine_neq = "layers".into();
        }
    }
    let dev = hasty_cuda::Device::new(0).expect("CUDA device");
    let mut a = Solver::new_cpu(c.clone()).unwrap();
    let mut b = Solver::new_gpu(c, &dev).unwrap();
    a.init().unwrap();
    b.init().unwrap();
    for round in 0..4 {
        a.advance(150).unwrap();
        b.advance(150).unwrap();
        a.accumulate_all().unwrap();
        b.accumulate_all().unwrap();
        // the finer level: flags and velocity
        if !a.fine.is_empty() {
            let fa = a.fine[0].be.fields().unwrap();
            let fb = b.fine[0].be.fields().unwrap();
            let n = fa.flags.len();
            let fd = (0..n).filter(|&k| fa.flags[k] != fb.flags[k]).count();
            let (mut s2, mut u2) = (0f64, 0f64);
            for k in 0..n {
                if fa.flags[k] & crate::params::F_FLUID == 0 || fa.flags[k] & crate::params::F_IFACE != 0 {
                    continue;
                }
                for c in 1..4 {
                    s2 += ((fa.ru[c * n + k] - fb.ru[c * n + k]) as f64).powi(2);
                    u2 += (fa.ru[c * n + k] as f64).powi(2);
                }
            }
            let rr = (s2 / u2.max(1e-30)).sqrt();
            eprintln!("round {round} level 1: flag mismatches {fd}/{n}, relative RMS Δu {rr:.2e}");
            assert!(fd <= n / 2000 && rr < 2e-3, "level 1 differs: flags {fd}, RMS {rr}");
        }
        let da = a.diagnostics().unwrap();
        let db = b.diagnostics().unwrap();
        let fa = a.be.fields().unwrap();
        let fb = b.be.fields().unwrap();
        let n = a.be.nodes();
        let flag_diff = (0..n).filter(|&k| fa.flags[k] != fb.flags[k]).count();
        let link_diff: Vec<usize> = (0..n).filter(|&k| fa.links[k] != fb.links[k]).collect();
        for &k in link_diff.iter().take(4) {
            eprintln!("   links differ at {:?}: cpu {:#021b} gpu {:#021b}", a.topo.coords(k), fa.links[k], fb.links[k]);
        }
        let qd: Vec<(usize, usize)> = (1..crate::params::Q).flat_map(|i| (0..n).map(move |k| (i, k))).filter(|&(i, k)| (fa.links[k] >> i) & 1 == 1 && fa.qlink[(i - 1) * n + k] != fb.qlink[(i - 1) * n + k]).collect();
        for &(i, k) in qd.iter().take(4) {
            eprintln!("   q differs at {:?} link {i}: cpu {} gpu {}", a.topo.coords(k), fa.qlink[(i - 1) * n + k], fb.qlink[(i - 1) * n + k]);
        }
        eprintln!("   link mismatches {}/{n}, q mismatches {}", link_diff.len(), qd.len());
        for k in (0..n).filter(|&k| fa.flags[k] != fb.flags[k]).take(6) {
            eprintln!("   flags differ at {:?}: cpu {:#010b} gpu {:#010b}, phi {:.3} / {:.3}, B {:e} / {:e}", a.topo.coords(k), fa.flags[k], fb.flags[k], fa.phi[k], fb.phi[k], fa.bsol[k], fb.bsol[k]);
        }
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
        if std::env::var("HASTY_TEST_DBG").is_ok() {
            let mut v: Vec<(f64, usize)> = (0..n).filter(|&k| fa.flags[k] & crate::params::F_FLUID != 0).map(|k| ((1..4).map(|c| ((fa.ru[c * n + k] - fb.ru[c * n + k]) as f64).powi(2)).sum::<f64>().sqrt(), k)).collect();
            v.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            for &(d, k) in v.iter().take(8) {
                let dum = (0..3).map(|c| (fa.um[c * n + k] - fb.um[c * n + k]).abs()).fold(0f32, f32::max);
                eprintln!("   node {:?} |Δu| {d:.2e} phi cpu {:.3} gpu {:.3} |Δum| {dum:.2e} um cpu {:?}", a.topo.coords(k), fa.phi[k], fb.phi[k], [fa.um[k], fa.um[n + k], fa.um[2 * n + k]]);
            }
            let nd = (0..3 * n).filter(|&i| (fa.um[i] - fb.um[i]).abs() > 1e-6).count();
            eprintln!("   um entries differing > 1e-6: {nd}");
            let mut w: Vec<(f32, usize)> = (0..n).map(|k| ((0..3).map(|c| (fa.um[c * n + k] - fb.um[c * n + k]).abs()).fold(0f32, f32::max), k)).collect();
            w.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            for &(d, k) in w.iter().take(6) {
                eprintln!("   um diff {d:.2e} at {:?} phi cpu {:.3} gpu {:.3} flags {} {} B cpu {:e} gpu {:e} um cpu {:?} gpu {:?}", a.topo.coords(k), fa.phi[k], fb.phi[k], fa.flags[k], fb.flags[k], fa.bsol[k], fb.bsol[k], [fa.um[k], fa.um[n + k], fa.um[2 * n + k]], [fb.um[k], fb.um[n + k], fb.um[2 * n + k]]);
            }
            // fluid-side band sum of the outward um (transpiration + motion) per backend
            let (mut sa, mut sb) = (0f64, 0f64);
            for k in 0..n {
                let g = a.topo.coords(k);
                let (y, z) = (g[1] as f64 - 10.0, g[2] as f64 - 10.0);
                let r = (y * y + z * z).sqrt().max(1e-9);
                if fa.phi[k].abs() < 2.0 && (10..=30).contains(&g[0]) {
                    sa += ((fa.um[n + k] as f64) * y + (fa.um[2 * n + k] as f64) * z) / r;
                    sb += ((fb.um[n + k] as f64) * y + (fb.um[2 * n + k] as f64) * z) / r;
                }
            }
            eprintln!("   band Σ radial um: cpu {sa:.6e} gpu {sb:.6e}");
        }
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
    {
        // the largest differences by slot (slot max |Δ| / slot max |value|)
        let st = crate::params::ACC_STRIDE;
        let mut v: Vec<String> = vec![];
        for k in 0..st {
            let (mut dm, mut vm, mut at) = (0f64, 0f64, 0usize);
            for c in 0..acc_a.len() / st {
                let d = (acc_a[c * st + k] - acc_b[c * st + k]).abs() as f64;
                vm = vm.max(acc_a[c * st + k].abs() as f64);
                if d > dm {
                    dm = d;
                    at = c;
                }
            }
            if dm > 1e-3 * vm.max(1e-12) {
                v.push(format!("slot {k}: max |Δ| {dm:.3e} at cell {at} (cpu {:.4e} gpu {:.4e}), slot max {vm:.3e}", acc_a[at * st + k], acc_b[at * st + k]));
            }
        }
        eprintln!("   {}", v.join("\n   "));
    }
    // membrane leaflets pass 2–3× faster flow through the 0.6-open valve; WALE's gradient-based eddy viscosity then
    // amplifies the GPU's fast-math rounding (2026-11-27: 6.3e-3 with WALE, 1.2e-5 with HASTY_CFD_TEST_CWALE=0 — links
    // and q fractions identical between the backends)
    let tol = if std::env::var("HASTY_CFD_MEMBRANE").is_ok() { 1e-2 } else { 5e-3 };
    // refined: the wall shear stress (slots 10–14) is compared apart. It divides the non-equilibrium by 1 − 1/τ, and the
    // fine level of this test runs at τ 1.1 (1 − 1/τ = 0.09): the fast-math rounding of the neq is amplified 11×
    // (2026-11-27: 6.6e-2 of the slot maximum with membranes, the velocity moments 2e-3); the production τ ≈ ½ gives
    // 1 − 1/τ ≈ −1
    if std::env::var("HASTY_CFD_REFINE").is_ok() {
        let st = crate::params::ACC_STRIDE;
        let part = |ks: &[usize]| -> f64 {
            let (mut dm, mut vm) = (0f64, 1e-12f64);
            for c in 0..acc_a.len() / st {
                for &k in ks {
                    dm = dm.max((acc_a[c * st + k] - acc_b[c * st + k]).abs() as f64);
                    vm = vm.max(acc_a[c * st + k].abs() as f64);
                }
            }
            dm / vm
        };
        let vel: Vec<usize> = (0..st).filter(|k| !(10..15).contains(k)).collect();
        let (rv, rw) = (part(&vel), part(&[10, 11, 12, 13, 14]));
        eprintln!("refined: accumulator velocity moments {rv:.2e}, wall shear stress {rw:.2e}");
        assert!(tot > 1000.0 && rv < tol && rw < 0.15, "accumulators differ: velocity moments {rv} (tolerance {tol}), WSS {rw}");
        return;
    }
    assert!(tot > 1000.0 && rel < tol, "accumulators differ by {rel} (tolerance {tol})");
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

// ── grid refinement (refine.rs, kernels/lbm_core.h "grid refinement") ──

/// world positions of the nodes of refinement level l
fn level_pos(s: &Solver<crate::backend::CpuBackend>, l: usize) -> Vec<[f64; 3]> {
    let t = s.level_topo(l);
    let o = crate::refine::level_origin(s.case.grid.origin, s.case.grid.dx, l);
    let dx = s.case.grid.dx / (1u64 << l) as f64;
    (0..t.nodes()).map(|k| { let g = t.coords(k); [0, 1, 2].map(|a| o[a] + g[a] as f64 * dx) }).collect()
}

fn level_fields(s: &mut Solver<crate::backend::CpuBackend>, l: usize) -> crate::backend::Fields {
    if l == 0 { s.be.fields().unwrap() } else { s.fine[l - 1].be.fields().unwrap() }
}

/// Σ ρ over the owned fluid nodes of every level by volume (8^−l per node): the lattice mass in level-0 units
fn total_mass(s: &mut Solver<crate::backend::CpuBackend>) -> f64 {
    let mut m = 0.0;
    for l in 0..=s.fine.len() {
        let f = level_fields(s, l);
        let n = f.flags.len();
        for k in 0..n {
            if f.flags[k] & crate::params::F_FLUID != 0 && f.flags[k] & crate::params::F_IFACE == 0 {
                m += f.ru[k] as f64 * 0.125f64.powi(l as i32);
            }
        }
    }
    m
}

/// equilibrium populations of ρ(x), u(x) (lattice velocity) on every level
fn set_state(s: &mut Solver<crate::backend::CpuBackend>, st: impl Fn([f64; 3]) -> (f64, [f64; 3])) {
    for l in 0..=s.fine.len() {
        let pos = level_pos(s, l);
        let n = pos.len();
        let mut f = vec![0f32; crate::params::Q * n];
        let mut ru = vec![0f32; 4 * n];
        for (k, x) in pos.iter().enumerate() {
            let (rho, u) = st(*x);
            let uf = u.map(|v| v as f32);
            for i in 0..crate::params::Q {
                f[i * n + k] = crate::params::geq(i, rho as f32, uf);
            }
            ru[k] = rho as f32;
            for a in 0..3 {
                ru[(a + 1) * n + k] = uf[a];
            }
        }
        if l == 0 { s.be.restore(&f, &ru).unwrap() } else { s.fine[l - 1].be.restore(&f, &ru).unwrap() }
    }
    s.sync_hidden().unwrap();
}

fn refine_box(lo: [f64; 3], hi: [f64; 3]) -> Vec<crate::case::RefineSpec> {
    vec![crate::case::RefineSpec { level: 1, boxes: vec![(lo, hi)], spheres: vec![] }]
}

#[test]
fn refined_box_stays_at_rest() {
    let dx = 1e-3;
    let mut c = mk_case([24, 24, 24], dx, 4, |p| [p[0] - 4e-3, 20e-3 - p[0], p[1] - 4e-3, 20e-3 - p[1], p[2] - 4e-3, 20e-3 - p[2]].into_iter().fold(f64::INFINITY, f64::min), Motion::still([0.0; 3], 1e-2, 1.0), 1e-4, 0.1, 0.05, vec![], vec![]);
    c.refine = refine_box([8e-3; 3], [15e-3; 3]);
    let mut s = Solver::new_cpu(c).unwrap();
    assert_eq!(s.fine.len(), 1);
    assert!(s.fine[0].cells == 512, "refined cells {}", s.fine[0].cells);
    s.init().unwrap();
    let m0 = total_mass(&mut s);
    s.advance(200).unwrap();
    let d = s.diagnostics().unwrap();
    let m1 = total_mass(&mut s);
    eprintln!("refined box at rest: {d}, mass {m0} → {m1}");
    assert!(d["u_max_lattice"].as_f64().unwrap() < 1e-6, "{d}");
    assert!(d["drho_mean"].as_f64().unwrap().abs() < 1e-6, "{d}");
    assert!((m1 / m0 - 1.0).abs() < 1e-7, "mass {m0} → {m1}");
}

/// a closed box with half-way walls (q = ½ on both levels: exactly mass conserving) and a density / velocity pulse
/// across the refinement interface: the total mass of both levels stays exact, with the fine region inside the fluid
/// and crossing the walls; the uniform run is the control
#[test]
fn refined_mass_is_exact() {
    let dx = 1e-3;
    let walls = |p: [f64; 3]| [p[0] - 3.5e-3, 28.5e-3 - p[0], p[1] - 3.5e-3, 28.5e-3 - p[1], p[2] - 3.5e-3, 28.5e-3 - p[2]].into_iter().fold(f64::INFINITY, f64::min);
    let pulse = |x: [f64; 3]| {
        let r2 = (x[0] - 14e-3).powi(2) + (x[1] - 15e-3).powi(2) + (x[2] - 16e-3).powi(2);
        let g = (-r2 / (2.0 * 3e-3f64.powi(2))).exp();
        (1.0 + 0.01 * g, [0.03 * g, 0.01 * g, -0.006 * g])
    };
    // τ0 = 0.52 on level 0 (0.54 on level 1), no LES: the low-viscosity regime of the heart (HASTY_CFD_TEST_TAU: τ0)
    let tau: f64 = std::env::var("HASTY_CFD_TEST_TAU").map_or(0.52, |x| x.parse().unwrap());
    let nu = ((tau - 0.5) / 3.0) * dx * dx / 5e-4;
    for (label, refine) in [("uniform", vec![]), ("interior", refine_box([12e-3, 10e-3, 10e-3], [20e-3, 22e-3, 22e-3])), ("across walls", refine_box([12e-3, 0.0, 10e-3], [20e-3, 32e-3, 22e-3]))] {
        let mut c = mk_case([32, 32, 32], dx, 4, walls, Motion::still([0.0; 3], 1e-2, 1.0), nu, 0.1, 0.05, vec![], vec![]);
        c.refine = refine;
        c.run.refine_neq = std::env::var("HASTY_CFD_REFINE_NEQ").unwrap_or("copy".into());
        let mut s = Solver::new_cpu(c).unwrap();
        s.init().unwrap();
        set_state(&mut s, pulse);
        let m0 = total_mass(&mut s);
        let mut worst = 0f64;
        let mut umax = 0f64;
        let every: u64 = std::env::var("HASTY_CFD_TEST_EVERY").map_or(50, |x| x.parse().unwrap());
        for _ in 0..300 / every {
            s.advance(every).unwrap();
            let m = total_mass(&mut s);
            worst = worst.max((m / m0 - 1.0).abs());
            umax = s.diagnostics().unwrap()["u_max_lattice"].as_f64().unwrap();
            if std::env::var("HASTY_CFD_TEST_EVERY").is_ok() {
                eprintln!("   {label} step {}: |u|max {umax:.4}, mass {:.3e}", s.step, m / m0 - 1.0);
            }
        }
        eprintln!("{label}: {} fine nodes, mass drift max {worst:.2e} over 300 steps, |u|max {umax:.4}", s.fine.first().map_or(0, |l| l.topo.nodes()));
        assert!(umax.is_finite() && umax < 0.05, "{label}: unstable |u| {umax}");
        assert!(worst < 1e-6, "{label}: mass drift {worst}");
    }
}

/// mean over a level's owned fluid nodes selected by world position: (u (m/s), ρ − 1, count)
fn level_mean(s: &mut Solver<crate::backend::CpuBackend>, l: usize, sel: impl Fn([f64; 3]) -> bool) -> ([f64; 3], f64, usize) {
    let pos = level_pos(s, l);
    let f = level_fields(s, l);
    let n = pos.len();
    let (mut u, mut r, mut c) = ([0.0; 3], 0.0, 0);
    for k in 0..n {
        if f.flags[k] & crate::params::F_FLUID == 0 || f.flags[k] & (crate::params::F_IFACE | crate::params::F_BC) != 0 || !sel(pos[k]) {
            continue;
        }
        for a in 0..3 {
            u[a] += f.ru[(a + 1) * n + k] as f64;
        }
        r += f.ru[k] as f64 - 1.0;
        c += 1;
    }
    let sc = s.u.dx / s.u.dt;
    (u.map(|x| x / c.max(1) as f64 * sc), r / c.max(1) as f64, c)
}

/// Poiseuille flow through a pipe refined over x 16–40 mm: flux continuity along the pipe across both interfaces, the
/// centre / mean ratio 2 in the fine section, the axial pressure gradient the same on both levels; HASTY_CFD_NO_RESCALE
/// for the unscaled transfer
#[test]
fn refined_poiseuille() {
    let dx = 1e-3;
    let r = 6e-3;
    let (yc, zc) = (8e-3, 8e-3);
    let (dt, u_lat_mean) = (5e-4, 0.015);
    let um = u_lat_mean * dx / dt;
    let q = um * std::f64::consts::PI * r * r;
    let nu = (0.1 / 3.0) * dx * dx / dt;
    let cuts = vec![
        CutSpec { name: "in".into(), center: [3e-3, yc, zc], normal: [-1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Inlet { q: Table::constant(q, 1.0) } },
        CutSpec { name: "out".into(), center: [53e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } },
    ];
    let mut c = mk_case([58, 17, 17], dx, 4, pipe(r, 0.5e-3, 56e-3, yc, zc), Motion::still([0.0; 3], 1e-2, 1.0), nu, 0.1, 0.05, vec![], cuts);
    c.refine = refine_box([16e-3, -1.0, -1.0], [40e-3, 1.0, 1.0]);
    c.run.refine_neq = std::env::var("HASTY_CFD_REFINE_NEQ").unwrap_or("copy".into());
    let mut s = Solver::new_cpu(c).unwrap();
    s.init().unwrap();
    for _ in 0..10 {
        s.advance(250).unwrap();
        s.diagnostics().unwrap();
    }
    let d = s.diagnostics().unwrap();
    let qout = d["cuts"][1]["q_out_ml_s"].as_f64().unwrap() * 1e-6;
    // section flux and mean density per plane; coarse planes at integer mm, fine planes at x = k/2 − ¼ mm
    let mut prof = vec![];
    for x in [6.0, 10.0, 13.0, 14.0, 15.0, 16.25, 16.75, 18.25, 22.25, 28.25, 34.25, 38.75, 39.75, 41.0, 42.0, 44.0, 48.0] {
        let lv = if (16.0..40.0).contains(&x) { 1 } else { 0 };
        let h = if lv == 1 { 0.5e-3 } else { 1e-3 };
        let (u, dr, n) = level_mean(&mut s, lv, |p| (p[0] - x * 1e-3).abs() < 1e-5);
        prof.push((x, u[0] * n as f64 * h * h * 1e6, dr, lv));
    }
    let line: Vec<String> = prof.iter().map(|(x, qf, dr, lv)| format!("x{x}{}: {qf:.4} mL/s Δρ {dr:.2e}", if *lv == 1 { "f" } else { "" })).collect();
    eprintln!("section flux / density along the refined pipe (inlet {:.4} mL/s):\n  {}", q * 1e6, line.join("\n  "));
    let (uc, _, _) = level_mean(&mut s, 1, |p| (p[0] - 28.25e-3).abs() < 1e-5 && (p[1] - yc).abs() < 0.3e-3 && (p[2] - zc).abs() < 0.3e-3);
    let (ua, _, na) = level_mean(&mut s, 1, |p| (p[0] - 28.25e-3).abs() < 1e-5);
    let area_ratio = na as f64 * 0.25e-6 / (std::f64::consts::PI * r * r);
    eprintln!("fine section x 28.25: centre {:.4} m/s, 2 × area mean {:.4} (fluid area / π r² {area_ratio:.3}), outlet {:.4} vs inlet {:.4} mL/s", uc[0], 2.0 * ua[0] * area_ratio, qout * 1e6, q * 1e6);
    // pressure gradients: coarse upstream (6–14), fine (18–38), coarse downstream (42–48)
    let slope = |a: &(f64, f64, f64, usize), b: &(f64, f64, f64, usize)| (b.2 - a.2) / (b.0 - a.0);
    let g_up = slope(&prof[0], &prof[3]);
    let g_f = slope(&prof[7], &prof[10]);
    let g_dn = slope(&prof[13], &prof[16]);
    eprintln!("dρ/dx per mm: coarse upstream {g_up:.3e}, fine {g_f:.3e}, coarse downstream {g_dn:.3e}");
    // the pressure step at each interface: the density extrapolated from both sides to the plane (x 16, x 40)
    let step_in = (prof[3].2 + g_up * (16.0 - prof[3].0)) - (prof[7].2 + g_f * (16.0 - prof[7].0));
    let step_out = (prof[10].2 + g_f * (40.0 - prof[10].0)) - (prof[13].2 + g_dn * (40.0 - prof[13].0));
    eprintln!("interface steps Δρ (upstream − downstream side): x 16 {step_in:.2e}, x 40 {step_out:.2e} (= {:.2} / {:.2} mm of pipe)", step_in / -g_f, step_out / -g_f);
    for &(x, qf, _, _) in &prof {
        assert!((qf / (q * 1e6) - 1.0).abs() < 0.06, "section flux {qf} at x {x} vs inlet {}", q * 1e6);
    }
    assert!((qout / q - 1.0).abs() < 0.05, "outlet {qout} vs inlet {q}");
    assert!((uc[0] / (2.0 * ua[0] * area_ratio) - 1.0).abs() < 0.03, "fine section centre {} vs 2 × mean", uc[0]);
    // the interfaces leave a first-order pressure step (piecewise-constant explosion); the gradients agree loosely
    assert!((g_f / g_up - 1.0).abs() < 0.25 && (g_dn / g_up - 1.0).abs() < 0.25, "pressure gradients {g_up} / {g_f} / {g_dn}");
}

/// the valve case of `valve_case` at spacing dx (the same physical case: geometry, ν, cuts)
fn valve_case_dx(kind: ValveKind, open: f64, dx: f64, nu: f64) -> Case {
    let mut c = valve_case(kind, open);
    let k = (1e-3 / dx).round() as usize;
    let walls = pipe(7e-3, 0.5e-3, 48e-3, 10e-3, 10e-3);
    let mut d = mk_case([50 * k, 21 * k, 21 * k], dx, 4, walls, Motion::still([0.0; 3], 1e-2, 1.0), nu, 0.1, 0.05, c.valves.clone(), c.cuts.clone());
    d.valves[0].ring = c.valves[0].ring;
    d.valves[0].thickness = c.valves[0].thickness;
    c = d;
    c
}

/// the jet one radius past a fully open membrane valve: peak and mean axial velocity on the plane x ≈ 31 mm and the
/// outlet flow for the uniform 1 mm lattice, 1 mm refined to 0.5 mm around the valve (x 14–36 mm), and the uniform
/// 0.5 mm lattice; the refined jet must follow the fine one (HASTY_CFD_TEST_TAU: τ0 of the 1 mm lattice, default 0.8)
#[test]
fn refined_valve_jet_follows_the_fine_lattice() {
    let tau: f64 = std::env::var("HASTY_CFD_TEST_TAU").map_or(0.8, |x| x.parse().unwrap());
    let nu = (tau - 0.5) / 3.0 * 1e-6 / 5e-4;
    let steps: u64 = std::env::var("HASTY_CFD_TEST_STEPS").map_or(1500, |x| x.parse().unwrap());
    let mut res = vec![];
    for (label, dx, refine) in [("1 mm", 1e-3, false), ("1 mm + 0.5 mm box", 1e-3, true), ("0.5 mm", 0.5e-3, false)] {
        let mut c = valve_case_dx(ValveKind::Leaflets { polys: crate::case::presets::trileaflet(90.0, 4) }, 1.0, dx, nu);
        c.valves[0].membrane = true;
        c.valves[0].theta_open = 90f64.to_radians();
        // flow-driven (the same flow on every lattice; a pressure drop gives each its own pipe resistance)
        c.cuts[0].kind = CutKind::Inlet { q: Table::constant(1.9e-6, 1.0) };
        if refine {
            c.refine = refine_box([14e-3, -1.0, -1.0], [36e-3, 1.0, 1.0]);
        }
        c.run.refine_neq = std::env::var("HASTY_CFD_REFINE_NEQ").unwrap_or("copy".into());
        let mut s = Solver::new_cpu(c).unwrap();
        s.init().unwrap();
        // the same physical time on every lattice: steps × (1 mm dt / dt)
        let k = (1e-3 / dx).round() as u64;
        s.advance(steps * k).unwrap();
        let (mut um, mut ux, mut qo) = (0.0, 0.0, 0.0);
        let nav = 10;
        for _ in 0..nav {
            s.advance(50 * k).unwrap();
            // planes centred on x = 31 mm: the node plane of the uniform lattices, the two fine planes 30.75 / 31.25 of
            // the refined one (mean of their means and peaks)
            let lv = if refine { 1 } else { 0 };
            let planes: Vec<f64> = if refine { vec![30.75e-3, 31.25e-3] } else { vec![31e-3] };
            let pos = level_pos(&s, lv);
            let f = level_fields(&mut s, lv);
            let n = pos.len();
            let sc = s.u.dx / s.u.dt;
            for xp in &planes {
                let (mut sum, mut cnt, mut mx) = (0.0, 0usize, 0.0f64);
                for kk in 0..n {
                    if (pos[kk][0] - xp).abs() > 1e-5 || f.flags[kk] & crate::params::F_FLUID == 0 || f.flags[kk] & crate::params::F_IFACE != 0 {
                        continue;
                    }
                    let v = f.ru[n + kk] as f64 * sc;
                    sum += v;
                    cnt += 1;
                    mx = mx.max(v);
                }
                um += sum / cnt.max(1) as f64 / (nav * planes.len()) as f64;
                ux += mx / (nav * planes.len()) as f64;
            }
            qo += s.diagnostics().unwrap()["cuts"][1]["q_link_ml_s"].as_f64().unwrap() / nav as f64;
        }
        eprintln!("{label} (τ0 {:.3}): plane x ≈ 31 mm mean {um:.4} m/s, peak {ux:.4} m/s, peak / mean {:.3}; outlet {qo:.3} mL/s", s.u.tau0, ux / um);
        res.push((ux / um, ux, qo));
    }
    let (c, r, f) = (res[0], res[1], res[2]);
    eprintln!("peak / mean: coarse {:.3}, refined {:.3}, fine {:.3}; peak: {:.4} / {:.4} / {:.4} m/s; outlet {:.3} / {:.3} / {:.3} mL/s", c.0, r.0, f.0, c.1, r.1, f.1, c.2, r.2, f.2);
    assert!((r.1 - f.1).abs() < 0.5 * (c.1 - f.1).abs().max(0.02 * f.1), "refined peak {} not closer to the fine {} than the coarse {}", r.1, f.1, c.1);
    assert!((r.0 - f.0).abs() < 0.5 * (c.0 - f.0).abs().max(0.02 * f.0), "refined peak / mean {} not closer to the fine {} than the coarse {}", r.0, f.0, c.0);
}

/// diagnostics: the post-collision non-equilibrium shear Π*_xy at mid radius of a developed Poiseuille pipe on the
/// 1 mm (τ 0.6) and 0.5 mm (τ 0.7) lattices; Chapman–Enskog: Π* = −2 (τ − 1) ρ c_s² S, ratio fine / coarse (τ_f − 1) /
/// (2 (τ_c − 1)) = 0.375
#[test]
#[ignore]
fn poiseuille_stress_by_level() {
    let r = 6e-3;
    let (yc, zc) = (8e-3, 8e-3);
    let nu = (0.1 / 3.0) * 1e-6 / 5e-4;
    let um = 0.015 * 1e-3 / 5e-4;
    let q = um * std::f64::consts::PI * r * r;
    let mut out = vec![];
    for dx in [1e-3f64, 0.5e-3] {
        let k = (1e-3 / dx).round() as usize;
        let cuts = vec![
            CutSpec { name: "in".into(), center: [3e-3, yc, zc], normal: [-1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Inlet { q: Table::constant(q, 1.0) } },
            CutSpec { name: "out".into(), center: [53e-3, yc, zc], normal: [1.0, 0.0, 0.0], radius: r, slab: 2e-3, sponge: 0.0, sponge_tau: 0.0, q_ref: None, share: None, inertance: 0.0, relax: 0.25, resistance: 0.0, upstream: None, kind: CutKind::Pressure { p: 0.0 } },
        ];
        let c = mk_case([58 * k, 17 * k, 17 * k], dx, 4, pipe(r, 0.5e-3, 56e-3, yc, zc), Motion::still([0.0; 3], 1e-2, 1.0), nu, 0.1, 0.05, vec![], cuts);
        let mut s = Solver::new_cpu(c).unwrap();
        s.init().unwrap();
        for _ in 0..10 {
            s.advance(250 * k as u64).unwrap();
            s.diagnostics().unwrap();
        }
        let (f, ru) = s.be.save().unwrap();
        let n = s.be.nodes();
        let mut acc = (0.0, 0.0, 0);
        for kk in 0..n {
            let g = s.topo.coords(kk);
            let x = [g[0] as f64 * dx, g[1] as f64 * dx, g[2] as f64 * dx];
            if (x[0] - 28e-3).abs() > 1e-6 || (x[2] - zc).abs() > 1e-6 || (x[1] - (yc + 3e-3)).abs() > 1e-6 {
                continue;
            }
            let (rho, u) = (ru[kk], [ru[n + kk], ru[2 * n + kk], ru[3 * n + kk]]);
            let mut pxy = 0.0f64;
            for i in 0..crate::params::Q {
                pxy += (crate::params::CX[i] * crate::params::CY[i]) as f64 * (f[i * n + kk] - crate::params::geq(i, rho, u)) as f64;
            }
            // S_xy from central differences of u_x along y (lattice units of this level)
            let up = s.topo.node_of([g[0], g[1] + 1, g[2]]).unwrap();
            let dn = s.topo.node_of([g[0], g[1] - 1, g[2]]).unwrap();
            let sxy = 0.5 * 0.5 * (ru[n + up] - ru[n + dn]) as f64;
            acc = (pxy, sxy, 1);
        }
        let ce = -2.0 * (s.u.tau0 - 1.0) * (1.0 / 3.0) * acc.1;
        eprintln!("dx {dx}: τ {:.3}, Π*_xy {:.4e}, S_xy {:.4e} (lattice), Chapman–Enskog −2(τ−1)c_s² S = {ce:.4e}, ratio {:.3}", s.u.tau0, acc.0, acc.1, acc.0 / ce);
        out.push(acc.0);
    }
    eprintln!("fine / coarse Π*_xy {:.3} (Chapman–Enskog 0.375)", out[1] / out[0]);
}
