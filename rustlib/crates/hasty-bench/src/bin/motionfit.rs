//! Respiratory motion model from a breathing-resolved `recon_b0` run (step 2 of the free-breathing plan in
//! research/plan_girf_b0_qfit.md): one smooth field S shared by all bins, I_b(x) ≈ I_0(x − r_b S(x))
//! (hasty-recon `motion`), with r_b the bin's mean self-gating surrogate. Optional truth comparison with the phantom's
//! respiratory basis (Lagrangian x = X + r S_true(X); equal to the fitted Eulerian pull-back to first order).
//!
//!     motionfit <recon_b0_run> [--truth-resp <phantom>/motion/resp_motion.hv] [--sp 8] [--levels 2] [--lambda 1e-3]
//!
//! Writes motion_S_mm.npy [x, y, z, 3] (displacement at r = 1, mm), motion_pred.npy / motion_bins.npy (pulled-back
//! reference vs measured magnitude per bin), motion.json; figures: tools/b0/plot_motion.py <run>.

use std::path::PathBuf;

use hasty_phantom::volume::write_npy;
use hasty_recon::data::read_npy;
use hasty_recon::motion::{fit_shared, warp};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |n: &str| -> Option<String> { args.iter().position(|a| a == n).map(|i| { args.remove(i); args.remove(i) }) };
    let truth = take("--truth-resp").map(PathBuf::from);
    let sp: f64 = take("--sp").map_or(8.0, |s| s.parse().unwrap());
    let levels: usize = take("--levels").map_or(2, |s| s.parse().unwrap());
    let lam: f64 = take("--lambda").map_or(1e-3, |s| s.parse().unwrap());
    let iters: usize = take("--iters").map_or(60, |s| s.parse().unwrap());
    let run = PathBuf::from(&args[0]);
    let summ: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run.join("summary.json")).unwrap()).unwrap();
    let dims: [usize; 3] = [0, 1, 2].map(|a| summ["dims"][a].as_u64().unwrap() as usize);
    let res = summ["res"].as_f64().unwrap();
    let nv = dims[0] * dims[1] * dims[2];
    let w = read_npy(&run.join("maps_water.npy")).unwrap().f32s();
    let f = read_npy(&run.join("maps_fat.npy")).unwrap().f32s();
    let nb = w.len() / nv;
    assert!(nb >= 2, "needs a breathing-resolved run (--resp-bins ≥ 2)");
    let sup: Vec<bool> = read_npy(&run.join("support.npy")).unwrap().f32s().iter().map(|&x| x > 0.5).collect();
    // per-bin magnitude, normalised by its mean over the support (removes global intensity changes)
    let ims: Vec<Vec<f32>> = (0..nb)
        .map(|b| {
            let m: Vec<f32> = (0..nv).map(|q| w[b * nv + q] + f[b * nv + q]).collect();
            let mean = (0..nv).filter(|&q| sup[q]).map(|q| m[q] as f64).sum::<f64>() / sup.iter().filter(|&&s| s).count() as f64;
            m.iter().map(|&x| (x as f64 / mean) as f32).collect()
        })
        .collect();
    // bin surrogates
    let g = read_npy(&run.join("selfgate.npy")).unwrap().f64s();
    let n_tr = g.len() / 4;
    let rb: Vec<f64> = (0..nb)
        .map(|b| {
            let v: Vec<f64> = (0..n_tr).filter(|&i| g[i * 4 + 3] as usize == b).map(|i| g[i * 4 + 1]).collect();
            v.iter().sum::<f64>() / v.len() as f64
        })
        .collect();
    let r_rel: Vec<f64> = rb.iter().map(|r| r - rb[0]).collect();
    // fit over voxels with signal
    let m99 = {
        let mut v: Vec<f32> = (0..nv).filter(|&q| sup[q]).map(|q| ims[0][q]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(0.99 * (v.len() - 1) as f64) as usize]
    };
    let mask: Vec<bool> = (0..nv).map(|q| sup[q]).collect();
    let t0 = std::time::Instant::now();
    let fit = fit_shared(&ims, &r_rel, dims, &mask, sp, levels, lam, iters);
    let s = fit.field.dense();
    println!(
        "shared field: {} control points, relative residual {:.3} (1 = no motion model), min Jacobian at r_max {:.3}, {} L-BFGS steps ({:.1}s)",
        fit.field.n_ctrl(),
        fit.rel_resid,
        fit.min_jac,
        fit.history.len(),
        t0.elapsed().as_secs_f64()
    );
    let smm: Vec<f32> = (0..3).flat_map(|a| s[a * nv..(a + 1) * nv].iter().map(|x| (x * res * 1e3) as f32).collect::<Vec<_>>()).collect();
    write_npy(&run.join("motion_S_mm.npy"), "<f4", &[dims[0], dims[1], dims[2], 3], true, &smm.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let mut pred = Vec::with_capacity(nb * nv);
    let mut meas = Vec::with_capacity(nb * nv);
    let mut per_bin = vec![];
    for b in 0..nb {
        let p = warp(&ims[0], dims, &s, r_rel[b]);
        let (mut e_m, mut e_0) = (0.0, 0.0);
        for q in 0..nv {
            if mask[q] {
                e_m += (p[q] - ims[b][q]).powi(2) as f64;
                e_0 += (ims[0][q] - ims[b][q]).powi(2) as f64;
            }
        }
        per_bin.push(serde_json::json!({"bin": b, "r_gate": rb[b], "r_rel": r_rel[b], "resid_model": e_m, "resid_static": e_0}));
        pred.extend(p);
        meas.extend(ims[b].iter().copied());
    }
    let wf = |name: &str, v: &[f32]| write_npy(&run.join(name), "<f4", &[dims[0], dims[1], dims[2], nb], true, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    wf("motion_pred.npy", &pred);
    wf("motion_bins.npy", &meas);

    // truth: the phantom's respiratory basis (sum of modes, m at r = 1) at the recon voxel positions
    let mut truth_json = serde_json::Value::Null;
    if let Some(tp) = truth {
        let rmo = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(&tp).unwrap())).unwrap();
        let bt = rmo.get("B").and_then(|x| x.as_tensor()).unwrap();
        let sh = bt.shape().to_vec();
        let bv = bt.to_vec::<f32>().unwrap();
        let rn = sh[2] * sh[3] * sh[4];
        let mut st = vec![0f32; 3 * rn];
        for k in 0..sh[0] {
            for q in 0..3 * rn {
                st[q] += bv[k * 3 * rn + q];
            }
        }
        let fl = |v: &hasty_value::Value| -> Vec<f64> {
            match v {
                hasty_value::Value::List(l) => l.iter().map(|x| match x { hasty_value::Value::Float(f) => *f, hasty_value::Value::Int(i) => *i as f64, _ => 0.0 }).collect(),
                hasty_value::Value::Tensor(t) => (0..t.numel()).map(|i| t.get_f64(i).0).collect(),
                hasty_value::Value::Float(f) => vec![*f],
                _ => vec![],
            }
        };
        let ro = fl(rmo.get("origin_m").unwrap());
        let rh = fl(rmo.get("voxel_m").unwrap())[0];
        let sim_meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(PathBuf::from(summ["sim_dir"].as_str().unwrap()).join("meta.json")).unwrap()).unwrap();
        let c: Vec<f64> = (0..3).map(|a| sim_meta["presc_center_subject"][a].as_f64().unwrap()).collect();
        let tri = |p: [f64; 3]| -> [f64; 3] {
            let fx = [0, 1, 2].map(|a| ((p[a] - ro[a]) / rh - 0.5).clamp(0.0, [sh[4], sh[3], sh[2]][a] as f64 - 1.001));
            let i0 = fx.map(|x| x.floor() as usize);
            let fr = [0, 1, 2].map(|a| fx[a] - i0[a] as f64);
            let mut v = [0.0; 3];
            for dk in 0..2 {
                for dj in 0..2 {
                    for di in 0..2 {
                        let wgt = (if di == 1 { fr[0] } else { 1.0 - fr[0] }) * (if dj == 1 { fr[1] } else { 1.0 - fr[1] }) * (if dk == 1 { fr[2] } else { 1.0 - fr[2] });
                        let q = (i0[0] + di) + sh[4] * ((i0[1] + dj) + sh[3] * (i0[2] + dk));
                        for a in 0..3 {
                            v[a] += wgt * st[a * rn + q] as f64;
                        }
                    }
                }
            }
            v
        };
        // per-bin displacement (mm) estimated r_rel,b S vs true (r_true,b − r_true,0) S_true, over signal voxels
        let rt: Vec<f64> = summ["resp"]["bin_mean_true_r"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
        let b = nb - 1;
        let (mut se, mut st2, mut sxy, mut sxx, mut syy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut si_est = vec![];
        let mut si_true = vec![];
        for q in 0..nv {
            if !(mask[q] && ims[0][q] > 0.2 * m99) {
                continue;
            }
            let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
            let p = [c[0] + (i as f64 - (dims[0] / 2) as f64) * res, c[1] + (j as f64 - (dims[1] / 2) as f64) * res, c[2] + (k as f64 - (dims[2] / 2) as f64) * res];
            let tv = tri(p);
            for a in 0..3 {
                let ue = r_rel[b] * s[a * nv + q] * res * 1e3;
                let ut = (rt[b] - rt[0]) * tv[a] * 1e3;
                se += (ue - ut).powi(2);
                st2 += ut * ut;
                sxy += ue * ut;
                sxx += ue * ue;
                syy += ut * ut;
                n += 1.0;
                if a == 2 {
                    si_est.push(ue);
                    si_true.push(ut);
                }
            }
        }
        let rmse = (se / n).sqrt();
        let rms_t = (st2 / n).sqrt();
        let cc = sxy / (sxx * syy).sqrt();
        let mx = |v: &[f64]| v.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "vs phantom truth (bin {b} vs bin 0, {} voxels): displacement RMSE {rmse:.2} mm (truth RMS {rms_t:.2} mm), correlation {cc:.3}; max SI displacement est {:.1} / true {:.1} mm",
            si_est.len(),
            mx(&si_est),
            mx(&si_true)
        );
        truth_json = serde_json::json!({"bin": b, "rmse_mm": rmse, "truth_rms_mm": rms_t, "corr": cc, "si_max_est_mm": mx(&si_est), "si_max_true_mm": mx(&si_true), "n": si_est.len()});
    }
    let js = serde_json::json!({"dims": dims, "res": res, "bins": nb, "r_gate": rb, "control_spacing_vox": fit.field.sp, "levels": levels, "lambda_rel": lam,
                                "rel_resid": fit.rel_resid, "min_jacobian": fit.min_jac, "history": fit.history, "per_bin": per_bin, "truth": truth_json});
    std::fs::write(run.join("motion.json"), serde_json::to_string_pretty(&js).unwrap()).unwrap();
    println!("→ {}", run.join("motion.json").display());
}
