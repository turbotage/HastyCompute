//! Ratemap fit (hasty_bench::ratemap) on saved echo images, without the echo reconstruction:
//!
//!     ratemap_fit <recon_b0 run with echoes.npy> <out_dir> [--fit smooth|voxel] [--fit-thr 0.01] [--om-max-hz 880]
//!                 [--window-hz 200] [--knots 48,32,20] [--lam 0.05] [--cauchy-hz 40] [--irls 6]
//!     ratemap_fit --synth <sim scan dir> <out_dir> [--grid-scale g] [...]   (control: noise-free voxel signal from the
//!                 phantom's truth maps, the fit's own error; truth enters only the synthetic input)
//!
//! Writes maps_{omega_hz,omega_smooth_hz,r2,ff,water,fat,relres}.npy [dims, bins] and summary.json in the recon_b0
//! layout (sim_dir, dims, res, te, resp.bin_amp), so `recon_p2np --ratemap est:<out_dir>[:bin]` reads it.
//! Evaluation: tools/b0/ratemap_eval.py <out_dir>.

use std::path::PathBuf;
use std::time::Instant;

use hasty_bench::ratemap::{fit, Cfg, Method};
use hasty_cuda::Device;
use hasty_phantom::volume::write_npy;
use hasty_qfit::waterfat::FatSpectrum;
use hasty_recon::data::read_npy;
use num_complex::Complex64 as C;

fn f32npy(path: &std::path::Path, dims: &[usize], v: &[f32]) {
    write_npy(path, "<f4", dims, true, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

/// g-block mean (x fastest) of a real volume
fn block_mean(v: &[f32], d: [usize; 3], g: usize) -> Vec<f32> {
    let o = d.map(|n| n / g);
    let mut out = vec![0f32; o[0] * o[1] * o[2]];
    for k in 0..o[2] {
        for j in 0..o[1] {
            for i in 0..o[0] {
                let mut s = 0f64;
                for c in 0..g {
                    for b in 0..g {
                        for a in 0..g {
                            s += v[(g * i + a) + d[0] * ((g * j + b) + d[1] * (g * k + c))] as f64;
                        }
                    }
                }
                out[i + o[0] * (j + o[1] * k)] = (s / (g * g * g) as f64) as f32;
            }
        }
    }
    out
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |n: &str| -> Option<String> { args.iter().position(|a| a == n).map(|i| { args.remove(i); args.remove(i) }) };
    let method = Method::parse(&take("--fit").unwrap_or_else(|| "smooth".into()));
    let mut cfg = Cfg { method, ..Default::default() };
    if method == Method::Voxel {
        cfg.om_max_hz = 500.0;
    }
    if let Some(v) = take("--om-max-hz") { cfg.om_max_hz = v.parse().unwrap() }
    if let Some(v) = take("--window-hz") { cfg.window_hz = v.parse().unwrap() }
    if let Some(v) = take("--knots") { cfg.knots_mm = v.split(',').map(|x| x.parse().unwrap()).collect() }
    if let Some(v) = take("--lam") { cfg.lam = v.parse().unwrap() }
    if let Some(v) = take("--cauchy-hz") { cfg.huber_hz = v.parse().unwrap() }
    if let Some(v) = take("--irls") { cfg.irls = v.parse().unwrap() }
    let fit_thr: f64 = take("--fit-thr").map_or(if method == Method::Voxel { 0.05 } else { 0.01 }, |s| s.parse().unwrap());
    let synth: Option<PathBuf> = take("--synth").map(PathBuf::from);
    let gs: usize = take("--grid-scale").map_or(1, |s| s.parse().unwrap());
    let t0 = Instant::now();
    let (inp, out) = match &synth {
        Some(_) => (None, PathBuf::from(&args[0])),
        None => (Some(PathBuf::from(&args[0])), PathBuf::from(&args[1])),
    };
    std::fs::create_dir_all(&out).unwrap();
    let tw = 2.0 * std::f64::consts::PI;

    // echo images [set][voxel] (set = bin · nd + d), TEs, grid
    let (dims, res, b0, te, nb, sim_dir, resp, imgs, support): ([usize; 3], f64, f64, Vec<f64>, usize, PathBuf, serde_json::Value, Vec<Vec<C>>, Vec<bool>) = match (&inp, &synth) {
        (Some(run), _) => {
            let s: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run.join("summary.json")).unwrap()).unwrap();
            let dims: [usize; 3] = [0, 1, 2].map(|a| s["dims"][a].as_u64().unwrap() as usize);
            let nv = dims[0] * dims[1] * dims[2];
            let e = read_npy(&run.join("echoes.npy")).expect("no echoes.npy: rerun recon_b0 with --save-echoes").c32s();
            let nset = e.len() / nv;
            let n_te = s["te"].as_array().unwrap().len();
            let nb = s["resp"]["bins"].as_u64().unwrap() as usize;
            let nb_fit = if s["moco"].is_null() { nb } else { 1 };
            assert_eq!(nset % nb_fit, 0);
            let nd = nset / nb_fit;
            assert!(nd * nb_fit == nset && nd <= n_te);
            let te: Vec<f64> = s["te"].as_array().unwrap()[..nd].iter().map(|x| x.as_f64().unwrap()).collect();
            let imgs = (0..nset).map(|t| e[t * nv..(t + 1) * nv].iter().map(|c| C::new(c.re as f64, c.im as f64)).collect()).collect();
            let sup: Vec<bool> = read_npy(&run.join("support.npy")).unwrap().f32s().iter().map(|&x| x > 0.5).collect();
            (dims, s["res"].as_f64().unwrap(), s["b0"].as_f64().unwrap(), te, nb_fit, PathBuf::from(s["sim_dir"].as_str().unwrap()), s["resp"].clone(), imgs, sup)
        }
        (None, Some(dir)) => {
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
            let df: [usize; 3] = [0, 1, 2].map(|a| meta["recon_dims"][a].as_u64().unwrap() as usize);
            let dims = df.map(|n| n / gs);
            let nv = dims[0] * dims[1] * dims[2];
            let b0 = meta["b0"].as_f64().unwrap();
            let te: Vec<f64> = meta["sequence"]["te"].as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>()).collect();
            let rd = |n: &str| block_mean(&read_npy(&dir.join(n)).unwrap().f32s().iter().map(|x| if x.is_finite() { *x } else { 0.0 }).collect::<Vec<_>>(), df, gs);
            let (pw, pf, r2, fld) = (rd("truth_pd_water.npy"), rd("truth_pd_fat.npy"), rd("truth_r2s.npy"), rd("field_ppm_recon.npy"));
            let fat = FatSpectrum::six_peak();
            let basis: Vec<C> = te.iter().map(|&t| fat.basis(b0, t)).collect();
            let gbar = hasty_qfit::waterfat::GAMMA_BAR;
            let imgs: Vec<Vec<C>> = (0..te.len())
                .map(|d| {
                    (0..nv)
                        .map(|q| {
                            let om = tw * gbar * b0 * fld[q] as f64 * 1e-6;
                            (C::new(pw[q] as f64, 0.0) + basis[d] * pf[q] as f64) * (-(C::new(r2[q] as f64, om)) * te[d]).exp()
                        })
                        .collect()
                })
                .collect();
            let sup = (0..nv).map(|q| pw[q] + pf[q] > 0.0).collect();
            (dims, meta["res"][0].as_f64().unwrap() * gs as f64, b0, te, 1, dir.clone(), serde_json::json!({"bins": 1, "mode": "synth"}), imgs, sup)
        }
        _ => unreachable!(),
    };
    let nv = dims[0] * dims[1] * dims[2];
    let nd = te.len();
    let e0mag: Vec<f64> = (0..nv).map(|q| imgs[0][q].norm()).collect();
    let e0_99 = {
        let mut v: Vec<f64> = (0..nv).filter(|&q| support[q]).map(|q| e0mag[q]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(0.99 * (v.len() - 1) as f64) as usize]
    };
    let vox: Vec<usize> = (0..nv).filter(|&q| support[q] && e0mag[q] > fit_thr * e0_99).collect();
    let nvx = vox.len();
    let nprob = nb * nvx;
    let mut y = vec![0.0f64; 2 * nd * nprob];
    for b in 0..nb {
        for (vi, &q) in vox.iter().enumerate() {
            for d in 0..nd {
                let z = imgs[b * nd + d][q];
                y[d * nprob + b * nvx + vi] = z.re;
                y[(nd + d) * nprob + b * nvx + vi] = z.im;
            }
        }
    }
    drop(imgs);
    println!("{} echo sets × {nb} bins, {nvx} voxels above {fit_thr} × the first-echo p99 ({:.0}s)", nd, t0.elapsed().as_secs_f64());
    let dev = Device::new(0).unwrap();
    let m = fit(&dev, b0, &te, dims, res, &vox, nb, &y, &cfg);
    let db = [dims[0], dims[1], dims[2], nb];
    for (n, v) in [("maps_omega_hz", &m.om_hz), ("maps_omega_smooth_hz", &m.om_smooth_hz), ("maps_r2", &m.r2), ("maps_ff", &m.ff), ("maps_water", &m.wm), ("maps_fat", &m.fm), ("maps_relres", &m.rel_res)] {
        f32npy(&out.join(format!("{n}.npy")), &db, v);
    }
    let summary = serde_json::json!({
        "sim_dir": sim_dir, "source": inp.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "synth (truth voxel signal)".into()),
        "dims": dims, "res": res, "grid_scale": if synth.is_some() { gs } else { 0 }, "b0": b0, "te": te, "fit": cfg.json(), "fit_thr": fit_thr, "fit_voxels": nvx,
        "resp": resp, "log": m.log, "seconds": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("done in {:.0}s → {}", t0.elapsed().as_secs_f64(), out.display());
}
