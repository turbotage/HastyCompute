//! Operator validation: dense static data (from `simimg`) reconstructed per encode with (almost) no regularisation,
//! error against the known object for several physics models of the encoding operator A.
//!   opval <sim_dir> [--models plain,b0,conc,full] [--iters n] [--lambda rel] [--nseg L] [--order o] [--every k] [--out dir]
//! plain: NUFFT only; b0: + off-resonance segments; conc: + concomitant (segments for the per-ADC-sample mean, per-sample
//! φ0 / k-shift / ΔC kernel correction); full: + GNL (image in apparent coordinates, physics at the true positions);
//! or any '+'-combination of b0, conc, gnl (e.g. `gnl`, `b0+gnl`).
//! The true coil maps are used (this validates A, not map estimation). PCGLS with the circulant preconditioner,
//! Tikhonov λ = rel·λ_max(AᴴA). Writes recon_<model>.npy, metrics.json, run.json and figures (tools/figs/opval_figs.py).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::cg::{pcgls_cb, power_norm};
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::precond::CirculantPrecond;
use hasty_recon::segphys::{fit_monomials, mean_coefficients, seg_tables, segment_phase};
use hasty_recon::setop::{SegSets, SetView};
use num_complex::Complex64;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let models: Vec<String> = take("--models").unwrap_or("plain,b0,conc,full".into()).split(',').map(String::from).collect();
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(100);
    let lam_rel: f64 = take("--lambda").map(|s| s.parse().unwrap()).unwrap_or(1e-6);
    let nseg_arg: usize = take("--nseg").map(|s| s.parse().unwrap()).unwrap_or(10);
    let order_arg: u32 = take("--order").map(|s| s.parse().unwrap()).unwrap_or(2);
    let every: usize = take("--every").map(|s| s.parse().unwrap()).unwrap_or(5);
    let out_arg = take("--out");
    let dir = PathBuf::from(&args[0]);
    let out = out_arg.map(PathBuf::from).unwrap_or_else(|| {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/runs").join(format!("opval_{}_{ts}", dir.file_name().unwrap().to_string_lossy()))
    });
    std::fs::create_dir_all(&out).unwrap();
    let t0 = Instant::now();
    std::env::set_var("HASTY_NOISE_SIGMA", std::env::var("HASTY_NOISE_SIGMA").unwrap_or("0".into()));
    let sd = SimData::load(&dir).unwrap();
    let (nv, nc, na, dims, res) = (sd.nv(), sd.nc, sd.na, sd.dims, sd.res);
    let w = sd.whitener();
    let truth = read_npy(&dir.join("truth_object.npy")).unwrap().c32s();
    let lumen: Vec<bool> = sd.truth_frac.iter().map(|&f| f > 0.5).collect();
    let mags: Vec<f64> = (0..nv).map(|q| (truth[q].re as f64).hypot(truth[q].im as f64)).collect();
    let mut sorted = mags.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = 0.02 * sorted[nv * 99 / 100];
    let support: Vec<usize> = (0..nv).filter(|&q| mags[q] > thr).collect();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let sets: Vec<_> = (0..4).map(|e| sd.sample_set(&sd.readouts_of(e, None), &w)).collect();
    println!(
        "{}: {:?} @ {:.1} mm, {} coils, {} samples per encode, support {} voxels, lumen {}; model of the data: {}",
        dir.display(),
        dims,
        res * 1e3,
        nc,
        sets[0].tau.len(),
        support.len(),
        lumen.iter().filter(|&&b| b).count(),
        sd.meta["level"].as_str().unwrap_or("?")
    );
    let tnorm: Vec<f64> = (0..4).map(|e| support.iter().map(|&q| { let t = truth[q + nv * e]; (t.re as f64).powi(2) + (t.im as f64).powi(2) }).sum::<f64>().sqrt()).collect();
    let mut metrics = serde_json::Map::new();
    for model in &models {
        let tm = Instant::now();
        // plain | b0 | conc (= b0+conc) | full (= b0+conc+gnl) | any '+'-combination of b0, conc, gnl
        let (with_b0, with_conc, gnl) = match model.as_str() {
            "plain" => (false, false, false),
            "conc" => (true, true, false),
            "full" => (true, true, true),
            m => {
                let t: Vec<&str> = m.split('+').collect();
                assert!(t.iter().all(|x| ["b0", "conc", "gnl"].contains(x)), "model {m}");
                (t.contains(&"b0"), t.contains(&"conc"), t.contains(&"gnl"))
            }
        };
        let (nseg, order) = if with_b0 || with_conc { (nseg_arg, if with_conc { order_arg } else { 0 }) } else { (1, 0) };
        let grid = sd.physics_grid(gnl, &w);
        let fit = fit_monomials(&sd, &grid, &support);
        let mut eng = SegNufft::new(&dev, dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames: 1, bspline: 1, max_grid_bytes: 6 << 30 }).unwrap();
        eng.set_maps(&st, &grid.smaps).unwrap();
        let tau_adc: Vec<f64> = (0..na).map(|n| sets[0].tau[n]).collect();
        let seg_fit: Vec<usize> = support.iter().copied().step_by(3).collect();
        let mut pts = Vec::new();
        let mut segs = Vec::new();
        let mut resid = 0.0f64;
        for set in &sets {
            let wbar = if with_conc { mean_coefficients(&[set], na) } else { vec![[0.0; 4]; na] };
            let psi = segment_phase(&grid, &wbar, &tau_adc, with_b0);
            let (sp, at, ta, dt) = build_segments(nv, nseg, tau_adc[0], tau_adc[na - 1], 256, &seg_fit, &psi);
            segs.push(eng.segments(&st, &sp, &at, ta, dt).unwrap());
            let tab = seg_tables(&sd, set, &fit, with_conc, Some(&wbar));
            resid = resid.max(tab.max_resid_phase);
            pts.push(eng.points(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, None).unwrap());
        }
        eng.autotune(&st, &segs[0], &pts[0]).unwrap();
        let mut ops = SegSets { eng, pts, segs, seg_of_set: vec![0, 1, 2, 3], kern: kern.clone() };
        let mut recon = vec![C32::new(0.0, 0.0); 4 * nv];
        let mut curves = Vec::new();
        let mut finals = Vec::new();
        for e in 0..4 {
            let mut view = SetView { op: &mut ops, t: e };
            let lmax = power_norm(&mut view, &st, 15).unwrap();
            let coords = &sets[e].coords;
            let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &grid.smaps, nc, 1e-3).unwrap();
            let y = st.clone_htod(&sets[e].y).unwrap();
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            let mut curve: Vec<(usize, f64)> = Vec::new();
            let err = |h: &[C32]| -> f64 {
                support.iter().map(|&q| { let (a, b) = (h[q], truth[q + nv * e]); ((a.re - b.re) as f64).powi(2) + ((a.im - b.im) as f64).powi(2) }).sum::<f64>().sqrt() / tnorm[e]
            };
            let mut cb = |k: usize, xk: &cudarc::driver::CudaSlice<C32>| {
                if k % every == 0 || k == iters {
                    curve.push((k, err(&st.clone_dtoh(xk).unwrap())));
                }
            };
            pcgls_cb(&mut view, Some(&mut pre), &st, &y, &mut x, lam_rel * lmax, iters, &mut cb).unwrap();
            let h = st.clone_dtoh(&x).unwrap();
            let fe = err(&h);
            recon[e * nv..(e + 1) * nv].copy_from_slice(&h);
            finals.push(fe);
            curves.push(curve);
        }
        // velocity from the encode phases (same convention as the truth), lumen and static support
        let vel = |img: &dyn Fn(usize, usize) -> Complex64, q: usize, a: usize| -> f64 { -(img(q, a + 1) * img(q, 0).conj()).arg() * sd.venc / std::f64::consts::PI };
        let rec = |q: usize, e: usize| Complex64::new(recon[q + nv * e].re as f64, recon[q + nv * e].im as f64);
        let (mut l2, mut lm, mut ln, mut s2, mut sn) = (0.0f64, [0.0f64; 3], 0usize, 0.0f64, 0usize);
        for &q in &support {
            for a in 0..3 {
                let d = vel(&rec, q, a) - sd.truth_vel[q + nv * a] as f64;
                let d = (d + sd.venc).rem_euclid(2.0 * sd.venc) - sd.venc; // phase wrap
                if lumen[q] {
                    l2 += d * d;
                    lm[a] += d;
                } else {
                    s2 += d * d;
                }
            }
            if lumen[q] { ln += 1 } else { sn += 1 }
        }
        let vrmse_l = (l2 / (3 * ln.max(1)) as f64).sqrt();
        let vbias: Vec<f64> = lm.iter().map(|v| v / ln.max(1) as f64).collect();
        let vrmse_s = (s2 / (3 * sn.max(1)) as f64).sqrt();
        println!(
            "  {model:<5} (L {nseg}, order {order}, GNL {}): relative image error {} | velocity RMSE lumen {:.4} m/s (bias {:?}), static {:.4} m/s | fit-left phase {:.1e} rad | {:.0}s",
            if gnl { "on" } else { "off" },
            finals.iter().map(|v| format!("{v:.2e}")).collect::<Vec<_>>().join(" "),
            vrmse_l,
            vbias.iter().map(|v| format!("{v:+.4}")).collect::<Vec<_>>(),
            vrmse_s,
            resid,
            tm.elapsed().as_secs_f64()
        );
        let bytes: Vec<u8> = recon.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat()).collect();
        hasty_phantom::volume::write_npy(&out.join(format!("recon_{}.npy", model.replace('+', "_"))), "<c8", &[dims[0], dims[1], dims[2], 4], true, &bytes).unwrap();
        metrics.insert(model.clone(), serde_json::json!({
            "nseg": nseg, "order": order, "gnl": gnl, "rel_error_final": finals, "rel_error_curves": curves,
            "vel_rmse_lumen": vrmse_l, "vel_bias_lumen": vbias, "vel_rmse_static": vrmse_s, "fit_left_phase": resid,
        }));
    }
    std::fs::write(out.join("metrics.json"), serde_json::to_string_pretty(&serde_json::Value::Object(metrics)).unwrap()).unwrap();
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&serde_json::json!({
        "cmdline": std::env::args().collect::<Vec<_>>().join(" "), "sim_dir": dir, "models": models, "iters": iters,
        "lambda_rel": lam_rel, "dims": dims, "venc": sd.venc, "support_threshold": thr,
    })).unwrap()).unwrap();
    let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/figs/opval_figs.py");
    let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
        Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
        r => println!("figure script failed ({r:?})"),
    }
    println!("done in {:.0}s -> {}", t0.elapsed().as_secs_f64(), out.display());
}
