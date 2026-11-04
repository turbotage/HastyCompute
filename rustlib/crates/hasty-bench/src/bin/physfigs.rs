//! What the reconstruction uses, for one simulation: B0, sequence, concomitant fields, gradient nonlinearity, coils.
//!   physfigs <sim_dir> [--out dir]      (default <sim_dir>/physics; figures: tools/figs/physics_figs.py, run at the end)
//! Arrays (Fortran npy, hasty layout x fastest) and summary.json:
//! - b0_hz [X,Y,Z]: off-resonance the operator applies (field map × γ̄ B0), and the phase it accrues over the readout;
//! - gnl_disp_mm [X,Y,Z]: |true − nominal| voxel position (gradient nonlinearity warp), gnl_vscale_pct [X,Y,Z]:
//!   max |σ_i(J) − 1| of the gradient-field Jacobian = velocity-encoding scale error;
//! - conc_voffset [X,Y,Z,E−1]: concomitant (Maxwell) phase of each encode relative to the reference at the first ADC
//!   sample, as velocity along that encode's direction (m/s); conc_ro_rad [X,Y,Z]: concomitant phase accrued during
//!   the readout (max over the ADC window, reference encode, middle partition);
//! - coil_sos [X,Y,Z] (whitened, true maps), psi [C,C] (noise covariance);
//! - k_traj [R,N,3] (cycles/m, the 16 interleaves of the middle partition), ro_g [M,3] (T/m) and ro_t [M] (s): readout
//!   gradient; coverage [F,E] readouts per frame and encode, coverage_kz [F,P] for the reference encode, card_phase
//!   [readouts] (cardiac phase of each readout).

use std::path::PathBuf;

use hasty_phantom::volume::write_npy;
use hasty_recon::data::{inv3, SimData};
use hasty_recon::flow::FlowDecoder;

fn f4(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(&args[0]);
    let out = args.iter().position(|a| a == "--out").map(|i| PathBuf::from(&args[i + 1])).unwrap_or_else(|| dir.join("physics"));
    std::fs::create_dir_all(&out).unwrap();
    let sd = SimData::load(&dir).unwrap();
    let (nv, nc, d) = (sd.nv(), sd.nc, sd.dims);
    let save = |name: &str, shape: &[usize], v: &[f32]| write_npy(&out.join(name), "<f4", shape, true, &f4(v)).unwrap();
    let w = sd.whitener();
    let meta = &sd.meta;
    let ideal = meta["ideal"].as_bool().unwrap_or(false) || dir.file_name().unwrap().to_string_lossy().contains("ideal");
    // physics grids with and without the GNL warp
    let gg = sd.physics_grid(true, &w);
    let gn = sd.physics_grid(false, &w);
    let lumen: Vec<bool> = (0..nv).map(|q| sd.sample_log(&sd.truth_frac, 0, &gg.x_log[q]) > 0.9).collect();
    // B0
    let hz: Vec<f32> = gg.omega.iter().map(|w| (w / (2.0 * std::f64::consts::PI)) as f32).collect();
    save("b0_hz.npy", &d, &hz);
    let ro_dur = sd.acq.adc_times.last().unwrap() - sd.acq.adc_times[0];
    // GNL
    let disp: Vec<f32> = (0..nv).map(|q| ((0..3).map(|a| (gg.x_log[q][a] - gn.x_log[q][a]).powi(2)).sum::<f64>().sqrt() * 1e3) as f32).collect();
    let vscale: Vec<f32> = (0..nv)
        .map(|q| {
            // velocity correction J⁻¹: deviation of its columns' norms / of J from identity, as a percentage
            let j = &gg.jac[q];
            let ji = inv3(j);
            let mut m = 0f64;
            for a in 0..3 {
                let n = (0..3).map(|b| j[b][a] * j[b][a]).sum::<f64>().sqrt();
                let ni = (0..3).map(|b| ji[b][a] * ji[b][a]).sum::<f64>().sqrt();
                m = m.max((n - 1.0).abs()).max((ni - 1.0).abs());
            }
            (100.0 * m) as f32
        })
        .collect();
    save("gnl_disp_mm.npy", &d, &disp);
    save("gnl_vscale_pct.npy", &d, &vscale);
    // concomitant: per-sample coefficients w_q (rad/m², monomials [z², x²+y², xz, yz] of the scanner-frame position)
    let m1 = sd.acq.params.encode_m1();
    let ne = m1.len();
    let dec = FlowDecoder::new(&m1);
    let kv = dec.kv();
    let nparts = sd.acq.kz.len();
    let ro_of = |e: usize| (0..sd.acq.readouts.len()).find(|&i| sd.acq.readouts[i].enc == e && sd.acq.readouts[i].part == nparts / 2).unwrap();
    let wq_of = |ro: usize| -> Vec<[f64; 4]> { sd.sample_set(&[ro], &w).wq };
    let phase = |wq: &[f64; 4], q: usize| -> f64 { (0..4).map(|i| wq[i] * gg.mono[q][i]).sum() };
    let w0 = wq_of(ro_of(0));
    let mut voff = vec![0f32; nv * (ne - 1)];
    for e in 1..ne {
        let we = wq_of(ro_of(e));
        for q in 0..nv {
            voff[q + nv * (e - 1)] = ((phase(&we[0], q) - phase(&w0[0], q)) / kv[e].max(1e-12)) as f32;
        }
    }
    save("conc_voffset.npy", &[d[0], d[1], d[2], ne - 1], &voff);
    let ro_c: Vec<f32> = (0..nv).map(|q| { let p0 = phase(&w0[0], q); w0.iter().map(|wq| (phase(wq, q) - p0).abs()).fold(0.0, f64::max) as f32 }).collect();
    save("conc_ro_rad.npy", &d, &ro_c);
    // coils, noise
    let sos: Vec<f32> = (0..nv).map(|q| (0..nc).map(|c| { let s = gg.smaps[c * nv + q]; s.re * s.re + s.im * s.im }).sum::<f32>().sqrt()).collect();
    save("coil_sos.npy", &d, &sos);
    let psi: Vec<f32> = (0..nc * nc).map(|i| sd.psi[i % nc][i / nc] as f32).collect();
    save("psi.npy", &[nc, nc], &psi);
    // sequence
    let nint = sd.acq.params.n_int;
    let mid: Vec<usize> = (0..sd.acq.readouts.len()).filter(|&i| sd.acq.readouts[i].part == nparts / 2 && sd.acq.readouts[i].enc == 0).take(nint).collect();
    let na = sd.na;
    let mut kt = vec![0f32; mid.len() * na * 3];
    for (r, &ro) in mid.iter().enumerate() {
        let k = sd.acq.k_logical(ro);
        for n in 0..na {
            for a in 0..3 {
                // hasty layout [R, N, 3] Fortran: index r + R (n + N a)
                kt[r + mid.len() * (n + na * a)] = (k[n][a] / (2.0 * std::f64::consts::PI)) as f32;
            }
        }
    }
    save("k_traj.npy", &[mid.len(), na, 3], &kt);
    let wave = &sd.acq.readout.wave;
    let mg = wave.g.len();
    let mut rg = vec![0f32; mg * 3];
    for (i, g) in wave.g.iter().enumerate() {
        for a in 0..3 {
            rg[i + mg * a] = g[a] as f32;
        }
    }
    save("ro_g.npy", &[mg, 3], &rg);
    save("ro_t.npy", &[mg], &(0..mg).map(|i| (wave.t0 + i as f64 * wave.dt) as f32).collect::<Vec<_>>());
    let nf = sd.n_frames;
    let mut cov = vec![0f32; nf * ne];
    let mut covz = vec![0f32; nf * nparts];
    for r in &sd.acq.readouts {
        cov[r.frame + nf * r.enc] += 1.0;
        if r.enc == 0 {
            covz[r.frame + nf * r.part] += 1.0;
        }
    }
    save("coverage.npy", &[nf, ne], &cov);
    save("coverage_kz.npy", &[nf, nparts], &covz);
    save("card_phase.npy", &[sd.acq.readouts.len()], &sd.acq.readouts.iter().map(|r| r.card.phase as f32).collect::<Vec<_>>());
    // summary
    let stat = |v: &[f32], m: Option<&[bool]>| -> serde_json::Value {
        let mut s: Vec<f32> = v.iter().enumerate().filter(|(q, x)| x.is_finite() && m.is_none_or(|m| m[*q % nv])).map(|(_, &x)| x).collect();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |f: f64| s.get(((s.len() as f64 - 1.0) * f) as usize).copied().unwrap_or(0.0);
        serde_json::json!({"p1": p(0.01), "p50": p(0.5), "p99": p(0.99), "max_abs": s.iter().map(|x| x.abs()).fold(0.0f32, f32::max)})
    };
    let full = vec![true; nv];
    let per_frame: Vec<f64> = (0..nf).map(|f| (0..ne).map(|e| cov[f + nf * e] as f64).sum::<f64>() / ne as f64).collect();
    let full_vol = (sd.acq.params.n_int * nparts) as f64;
    let voxel_mm = sd.vox.map(|v| v * 1e3);
    let fov_mm: [f64; 3] = [0, 1, 2].map(|a| d[a] as f64 * sd.vox[a] * 1e3);
    let j = serde_json::json!({
        "sim": dir, "ideal_flag_in_name": ideal, "encoding": meta["encoding"], "venc": sd.venc, "kv_rad_per_mps": kv,
        "m1_logical": m1, "tr_s": sd.acq.tr, "te_s": sd.acq.t_ro, "readout_s": ro_dur, "dwell_s": sd.acq.params.dwell, "n_adc": na,
        "n_interleaves": nint, "n_partitions": nparts, "n_readouts": sd.acq.readouts.len(), "scan_time_s": sd.acq.params.scan_time,
        "frames": nf, "rr_mean_s": sd.acq.params.mean_rr, "readouts_per_frame_encode": per_frame.iter().sum::<f64>() / nf as f64,
        "acceleration_per_frame": full_vol / (per_frame.iter().sum::<f64>() / nf as f64),
        "dims": d, "voxel_mm": voxel_mm, "fov_mm": fov_mm, "coils": nc,
        "b0_hz_fov": stat(&hz, Some(&full)), "b0_hz_lumen": stat(&hz, Some(&lumen)),
        "b0_readout_phase_rad_p99": stat(&hz, Some(&full))["max_abs"].as_f64().unwrap() * 2.0 * std::f64::consts::PI * ro_dur,
        "gnl_disp_mm_fov": stat(&disp, Some(&full)), "gnl_disp_mm_lumen": stat(&disp, Some(&lumen)),
        "gnl_vscale_pct_fov": stat(&vscale, Some(&full)), "gnl_vscale_pct_lumen": stat(&vscale, Some(&lumen)),
        "conc_voffset_mps_fov": stat(&voff, Some(&full)), "conc_voffset_mps_lumen": stat(&voff, Some(&lumen)),
        "conc_readout_rad_fov": stat(&ro_c, Some(&full)), "conc_readout_rad_lumen": stat(&ro_c, Some(&lumen)),
        "lumen_voxels": lumen.iter().filter(|&&b| b).count(),
        "note": "the simulator's 'ideal' runs zero B0 / concomitant / GNL in the data; these maps are what the physics model of the reconstruction would apply (--phys full --gnl)",
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&j).unwrap());
    let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    let tool = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/figs/physics_figs.py");
    let _ = std::process::Command::new(&py).arg("-I").arg(&tool).arg(&out).arg(&dir).status();
}
