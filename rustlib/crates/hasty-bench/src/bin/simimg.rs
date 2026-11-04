//! Static image object → synthetic spiral 4D-flow k-space with the full physics (operator validation).
//!   simimg <case_dir> <out_dir> [--n-int n] [--visits k] [--gnl on|off] [--b0-hz a] [--no-conc] [--venc v]
//!          [--fov-edge r] [--noise rel] [--centre-mm x,y,z | case]
//! `case_dir` from rustlib/tools/raclette/export_case.py (object per encode, coil maps, mask, geometry). Model-exact
//! level: one source per reconstruction voxel at its *apparent* (GNL-warped) position, so the object is exactly
//! representable; true positions, coil maps, B0 and the concomitant monomials come from `SimData::physics_grid`, i.e.
//! the same physics the reconstruction model uses, evaluated exactly per sample by the GPU direct sum (no segments, no
//! interpolation kernels). The output directory has the `sim4d` layout (loads with `SimData::load`); additionally
//! truth_object.npy (complex [x, y, z, 4]). The object is static: velocities live in the encode phases.
//! B0: analytic smooth field (±`--b0-hz` scale, several length scales). Coverage: stack of spirals, every partition
//! visited `--visits` × n_int times per encode with golden-angle rotations (dense, ≥ Nyquist for visits ≥ 1).

use std::path::PathBuf;
use std::time::Instant;

use hasty_cuda::{Device, C32, C64};
use hasty_mrsim::direct::{pack_sources, DirectSim, Source};
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::sequence::{Acquisition, FlowSpiralParams, SAMPLE_FLOATS};
use hasty_mrsim::{mat_vec, GAMMA_BAR};
use hasty_recon::data::{read_npy, SimData};
use num_complex::Complex64;

fn save(path: &std::path::Path, descr: &str, shape: &[usize], bytes: Vec<u8>) {
    hasty_phantom::volume::write_npy(path, descr, shape, true, &bytes).unwrap();
}
fn c8(v: impl IntoIterator<Item = Complex64>) -> Vec<u8> {
    v.into_iter().flat_map(|c| [(c.re as f32).to_le_bytes(), (c.im as f32).to_le_bytes()].concat()).collect()
}
fn f4(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let no_conc = args.iter().position(|a| a == "--no-conc").map(|i| { args.remove(i); true }).unwrap_or(false);
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let n_int: usize = take("--n-int").map(|s| s.parse().unwrap()).unwrap_or(8);
    let visits: usize = take("--visits").map(|s| s.parse().unwrap()).unwrap_or(2);
    let gnl_on = take("--gnl").map_or(true, |s| s == "on");
    let b0_hz: f64 = take("--b0-hz").map(|s| s.parse().unwrap()).unwrap_or(150.0);
    let venc: f64 = take("--venc").map(|s| s.parse().unwrap()).unwrap_or(2.5);
    let fov_edge: f64 = take("--fov-edge").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let noise_rel: f64 = take("--noise").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    // FOV centre in the scanner frame (isocentre = origin), mm; "case" = RACLETTE's stored coordinates (≈19 cm off
    // isocentre for n001, whether those are isocentre-relative is not documented)
    let centre_arg = take("--centre-mm").unwrap_or("0,40,20".into());
    let (case, out) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]));
    std::fs::create_dir_all(&out).unwrap();
    let t0 = Instant::now();
    let geom: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(case.join("geom.json")).unwrap()).unwrap();
    let dims: [usize; 3] = [0, 1, 2].map(|a| geom["dims"][a].as_u64().unwrap() as usize);
    let res = geom["res_m"].as_f64().unwrap();
    let rot: [[f64; 3]; 3] = [0, 1, 2].map(|r| [0, 1, 2].map(|c| geom["rot_logical_to_lps"][r][c].as_f64().unwrap()));
    let centre: [f64; 3] = if centre_arg == "case" {
        [0, 1, 2].map(|a| geom["fov_centre_lps_m"][a].as_f64().unwrap())
    } else {
        let v: Vec<f64> = centre_arg.split(',').map(|s| s.parse::<f64>().unwrap() * 1e-3).collect();
        [v[0], v[1], v[2]]
    };
    let nv = dims[0] * dims[1] * dims[2];
    let obj = read_npy(&case.join("object.npy")).unwrap();
    assert!(obj.fortran && obj.shape == vec![dims[0], dims[1], dims[2], 4]);
    let obj = obj.c32s();
    let sm = read_npy(&case.join("smaps.npy")).unwrap();
    let nc = sm.shape[3];
    let smaps = sm.c32s();
    let mask = read_npy(&case.join("mask.npy")).unwrap().f32s();

    // ------------------------------------------------ acquisition
    let mut fp = FlowSpiralParams::aorta_3t_default();
    fp.fov = [0, 1, 2].map(|a| dims[a] as f64 * res);
    fp.res = [res; 3];
    fp.n_int = n_int;
    fp.fov_edge_ratio = fov_edge;
    fp.venc = venc;
    fp.rot_presc = rot;
    fp.fov_center = centre;
    fp.n_frames = 1;
    let tr = Acquisition::new(FlowSpiralParams { scan_time: 0.1, ..fp.clone() }).tr;
    let n_part = fp.n_partitions();
    fp.scan_time = (4 * n_part * visits * n_int) as f64 * tr;
    let acq = Acquisition::new(fp.clone());
    let na = acq.n_adc();
    let gnl = if gnl_on { GradientCoils { r0: 0.25, transverse: [-0.03, -0.004], longitudinal: [-0.045, -0.004] } } else { GradientCoils::ideal() };
    println!(
        "{}: grid {dims:?} @ {:.1} mm, FOV centre (scanner) {:?} mm; {} readouts x {} ADC ({:.2} ms readout, TR {:.2} ms), {n_part} partitions x {} visits per encode, {nc} coils, GNL {}, concomitant {}",
        geom["source"].as_str().unwrap(),
        res * 1e3,
        centre.map(|v| (v * 1e3).round()),
        acq.readouts.len(),
        na,
        acq.adc_times[na - 1] * 1e3 - acq.adc_times[0] * 1e3,
        tr * 1e3,
        visits * n_int,
        if gnl_on { "on" } else { "off" },
        if no_conc { "off" } else { "on" }
    );

    // ------------------------------------------------ physics files in the sim4d layout (the loader evaluates them)
    let vlog = |q: usize| -> [f64; 3] {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [(i as f64 - (dims[0] / 2) as f64) * res, (j as f64 - (dims[1] / 2) as f64) * res, (k as f64 - (dims[2] / 2) as f64) * res]
    };
    // smooth off-resonance with several length scales (Hz), defined on the logical grid
    let field_hz = |p: [f64; 3]| -> f64 {
        let (x, y, z) = (p[0], p[1], p[2]);
        b0_hz * (0.8 * (2.0 * std::f64::consts::PI * x / 0.17).sin() * (2.0 * std::f64::consts::PI * y / 0.13).cos() + 0.6 * (y / 0.08) + 0.5 * (z / 0.024).powi(2) - 0.4
            + 1.5 * (-((x - 0.03).powi(2) + (y + 0.02).powi(2)) / (2.0 * 0.025f64.powi(2))).exp())
    };
    let ppm: Vec<f32> = (0..nv).map(|q| (field_hz(vlog(q)) / (GAMMA_BAR * fp.b0) * 1e6) as f32).collect();
    save(&out.join("field_ppm_recon.npy"), "<f4", &dims, f4(ppm.iter().copied()));
    save(&out.join("smaps_true.npy"), "<c8", &[dims[0], dims[1], dims[2], nc], c8(smaps.iter().map(|c| Complex64::new(c.re as f64, c.im as f64))));
    save(&out.join("truth_lumen_fraction.npy"), "<f4", &dims, f4(mask.iter().copied()));
    save(&out.join("truth_object.npy"), "<c8", &[dims[0], dims[1], dims[2], 4], c8(obj.iter().map(|c| Complex64::new(c.re as f64, c.im as f64))));
    // truth velocity: the encode phases, with the reconstruction's convention v = −arg(x_a x_0*)·VENC/π
    let mut vel = vec![0f32; 3 * nv];
    for q in 0..nv {
        let r0 = obj[q];
        for a in 0..3 {
            let ra = obj[q + nv * (a + 1)];
            let z = Complex64::new(ra.re as f64, ra.im as f64) * Complex64::new(r0.re as f64, -r0.im as f64);
            vel[q + nv * a] = (-z.arg() * venc / std::f64::consts::PI) as f32;
        }
    }
    save(&out.join("truth_vel_logical.npy"), "<f4", &[dims[0], dims[1], dims[2], 3, 1], f4(vel.iter().copied()));
    let meta = |sigma: f64| {
        serde_json::json!({
            "subject": geom["source"], "case_dir": case, "level": "model-exact (sources on the apparent reconstruction grid)",
            "b0": fp.b0, "venc": fp.venc, "encoding": format!("{:?}", fp.encoding), "coverage": format!("{:?}", fp.coverage),
            "fov": fp.fov, "res": fp.res, "recon_dims": dims, "n_int": fp.n_int, "fov_edge_ratio": fp.fov_edge_ratio,
            "dt": fp.dt, "dwell": fp.dwell, "t_rf": fp.t_rf, "t_spoil": fp.t_spoil, "scan_time": fp.scan_time,
            "gmax": fp.lim.gmax, "smax": fp.lim.smax, "rot_presc": fp.rot_presc, "fov_center_scanner": fp.fov_center,
            "mean_rr": fp.mean_rr, "rr_rel_std": fp.rr_rel_std, "n_frames": fp.n_frames, "seed": fp.seed,
            "tr": acq.tr, "te": acq.t_ro, "n_readouts": acq.readouts.len(), "n_adc": na, "n_coils": nc,
            "noise_sigma": sigma, "noise_rel": noise_rel, "visits": visits, "b0_hz": b0_hz, "concomitant": !no_conc,
            "psi": (0..nc).map(|i| (0..nc).map(|j| if i == j { 1.0 } else { 0.0 }).collect::<Vec<f64>>()).collect::<Vec<_>>(),
            "gnl": {"r0": gnl.r0, "transverse": gnl.transverse, "longitudinal": gnl.longitudinal},
        })
    };
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta(0.0)).unwrap()).unwrap();
    save(&out.join("data_clean.npy"), "<c8", &[acq.readouts.len(), na, nc], vec![0u8; acq.readouts.len() * na * nc * 8]);
    let sd = SimData::load(&out).unwrap();
    let w: Vec<Vec<f64>> = (0..nc).map(|i| (0..nc).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let grid = sd.physics_grid(gnl_on, &w);
    let disp: f64 = (0..nv).map(|q| { let v = vlog(q); (0..3).map(|a| (grid.x_log[q][a] - v[a]).powi(2)).sum::<f64>().sqrt() }).fold(0.0, f64::max);
    println!("physics: max GNL displacement in the FOV {:.2} mm, off-resonance {:.0}..{:.0} Hz", disp * 1e3, grid.omega.iter().cloned().fold(f64::MAX, f64::min) / (2.0 * std::f64::consts::PI), grid.omega.iter().cloned().fold(f64::MIN, f64::max) / (2.0 * std::f64::consts::PI));

    // ------------------------------------------------ sources (one per voxel with signal) and per-encode amplitudes
    let live: Vec<usize> = (0..nv).filter(|&q| (0..4).any(|e| obj[q + nv * e].re != 0.0 || obj[q + nv * e].im != 0.0)).collect();
    let src: Vec<Source> = live
        .iter()
        .map(|&q| {
            let u = mat_vec(&rot, &vlog(q));
            let xl = grid.x_log[q];
            let p = mat_vec(&rot, &xl);
            Source { u, x: [p[0] + centre[0], p[1] + centre[1], p[2] + centre[2]], omega: grid.omega[q], ..Default::default() }
        })
        .collect();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let sim = DirectSim::new(&dev, nc).unwrap();
    let d_src = st.clone_htod(&pack_sources(&src)).unwrap();
    let mut data = vec![Complex64::new(0.0, 0.0); acq.readouts.len() * na * nc];
    let t1 = Instant::now();
    for e in 0..4 {
        let amp: Vec<C32> = live.iter().flat_map(|&q| (0..nc).map(move |c| (q, c))).map(|(q, c)| {
            let o = obj[q + nv * e];
            let s = grid.smaps[c * nv + q];
            C32::new(o.re * s.re - o.im * s.im, o.re * s.im + o.im * s.re)
        }).collect();
        let d_amp = st.clone_htod(&amp).unwrap();
        let ros: Vec<usize> = (0..acq.readouts.len()).filter(|&r| acq.readouts[r].enc == e).collect();
        let per = ((1usize << 31) / (nc * 16) / na).max(1);
        for chunk in ros.chunks(per) {
            let mut smp = Vec::with_capacity(chunk.len() * na * SAMPLE_FLOATS);
            for &r in chunk {
                let mut s = acq.sample_params(r..r + 1);
                if no_conc {
                    for n in 0..na {
                        for k in 9..13 {
                            s[n * SAMPLE_FLOATS + k] = 0.0;
                        }
                    }
                }
                smp.extend(s);
            }
            let ns = chunk.len() * na;
            let d_smp = st.clone_htod(&smp).unwrap();
            let mut d_out = st.alloc_zeros::<C64>(ns * nc).unwrap();
            sim.accumulate(&st, &d_smp, ns, &d_src, &d_amp, src.len(), &mut d_out).unwrap();
            let o = st.clone_dtoh(&d_out).unwrap();
            for (k, &r) in chunk.iter().enumerate() {
                for n in 0..na {
                    for c in 0..nc {
                        let v = o[(k * na + n) * nc + c];
                        data[(r * na + n) * nc + c] = Complex64::new(v.re, v.im);
                    }
                }
            }
        }
    }
    let rms = (data.iter().map(|c| c.norm_sqr()).sum::<f64>() / data.len() as f64).sqrt();
    println!("direct sum: {} sources x {} samples x {nc} coils in {:.1}s; data rms {rms:.3e}", src.len(), acq.n_samples(), t1.elapsed().as_secs_f64());
    save(&out.join("data_clean.npy"), "<c8", &[acq.readouts.len(), na, nc], c8(data));
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta(noise_rel * rms)).unwrap()).unwrap();
    println!("done in {:.1}s -> {}", t0.elapsed().as_secs_f64(), out.display());
}
