//! Time-resolved P²nP-ISTA reconstruction of the synthetic spiral 4D-flow data (frames × encodes), no density
//! compensation, whitened + coil-compressed data, LLR prior over all frames × encodes or the learned 7-point prior.
//! Any encoding of the simulator (referenced / Hadamard 4-point, 7-point dual-VENC): velocities by hasty_recon::flow.
//!   recon_p2np <sim_dir> [--model sense|seg] [--temporal off|linear|cubic] [--phys none|b0|full] [--gnl]
//!              [--nseg L] [--order o] [--precond none,circ,zmshr1,zmshr1m] [--iters n] [--kappa k,k,..] [--block b]
//!              [--vcoils n] [--every n] [--warm n] [--eps e] [--conc-dc] [--synthetic] [--out dir]
//!              [--learned <export_dir> [--tile n] [--keep-frames n] [--learned-prec bf16|fp16|fp32] [--learned-cpu]]
//!              [--fresh] [--ckpt-every n]
//! --learned: the exported denoiser (tools/denoiser/export.py) through the libtorch plugin (hasty_recon::learned) in
//!   place of LLR, x ← x + κ (D(x) − x) per P²nP iteration; --tile / --keep-frames: exact overlap tiles in space /
//!   cyclic time windows (bounded GPU memory); bf16 autocast by default (the training regime).
//! Interrupt / resume: Ctrl-C, SIGTERM or a file STOP in the run directory checkpoints (every --ckpt-every iterations
//! too) and exits; rerunning the same command continues (warm start per encode, then each preconditioner / κ run).
//! --fresh discards the saved state.
//! Every run writes `dir` (default <repo>/data/hasty/runs/<sim>_<model>…, stable so that reruns resume): run.json, coil maps, masks, warm
//! start, final complex images and velocities per (preconditioner, κ), metrics CSV, and figures (`figs/`, made by
//! rustlib/tools/figs/run_figs.py at the end).
//! sense: plain NUFFT per (frame, encode) set; `--conc-dc` folds the per-encode mean concomitant phase into the coil maps.
//! seg:   segmented physics NUFFT (`hasty_nufft::SegNufft`): B0 + full concomitant history per sample (segments for the
//!        readout-time mean, per-sample k-shift / phase / ΔC kernel correction), coil maps and fields at the true (GNL)
//!        positions with `--gnl`; per (frame, encode) set, or per encode over all frames with temporal B-spline
//!        interpolation in cardiac time (`--temporal`).
//! Reports data fidelity and the per-frame velocity RMSE in the lumen (velocities J⁻¹-corrected and compared at the
//! true positions with `--gnl`).

use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::CudaSlice;
use hasty_cuda::{Device, C32};
use hasty_mrsim::{mat_vec, transpose};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::cg::pcgls_ctl;
use hasty_recon::checkpoint::{value_dev, Outcome, RunCtl};
use hasty_recon::data::{inv3, SampleSet, SimData};
use hasty_recon::flow::FlowDecoder;
use hasty_recon::gpu::Kernels;
use hasty_recon::llr::Llr;
use hasty_recon::op::SenseOp;
use hasty_recon::p2np::{P2np, Precond, Stack};
use hasty_recon::precond::CirculantPrecond;
use hasty_recon::segphys::{fit_monomials, mean_coefficients, seg_tables, segment_phase};
use hasty_recon::setop::{SegSets, SenseSets, SetOp};
use nalgebra::DMatrix;
use num_complex::Complex64;

fn compression(y: &[C32], m: usize, nc: usize, nvc: usize) -> Vec<Vec<Complex64>> {
    let step = (m / 20000).max(1);
    let cols: Vec<usize> = (0..m).step_by(step).collect();
    let a = DMatrix::<Complex64>::from_fn(nc, cols.len(), |c, j| {
        let v = y[c * m + cols[j]];
        Complex64::new(v.re as f64, v.im as f64)
    });
    let svd = (&a * a.adjoint()).svd(true, false);
    let u = svd.u.unwrap();
    (0..nvc).map(|v| (0..nc).map(|c| u[(c, v)].conj()).collect()).collect()
}

fn apply_cc(x: &[C32], len: usize, nc: usize, t: &[Vec<Complex64>]) -> Vec<C32> {
    let mut out = vec![C32::new(0.0, 0.0); t.len() * len];
    for (v, row) in t.iter().enumerate() {
        for i in 0..len {
            let mut acc = Complex64::new(0.0, 0.0);
            for c in 0..nc {
                let s = x[c * len + i];
                acc += row[c] * Complex64::new(s.re as f64, s.im as f64);
            }
            out[v * len + i] = C32::new(acc.re as f32, acc.im as f32);
        }
    }
    out
}

fn save(path: &std::path::Path, descr: &str, shape: &[usize], bytes: Vec<u8>) {
    hasty_phantom::volume::write_npy(path, descr, shape, true, &bytes).unwrap();
}
fn c8(v: &[C32]) -> Vec<u8> {
    v.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat()).collect()
}
fn f4(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| args.iter().position(|a| a == name).map(|i| { args.remove(i); true }).unwrap_or(false);
    let (synthetic, conc_dc, gnl, fresh) = (flag("--synthetic"), flag("--conc-dc"), flag("--gnl"), flag("--fresh"));
    let learned_cpu = flag("--learned-cpu");
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(40);
    let kappas: Vec<f64> = take("--kappa").unwrap_or("1.0".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let block: usize = take("--block").map(|s| s.parse().unwrap()).unwrap_or(8);
    let nvc: usize = take("--vcoils").map(|s| s.parse().unwrap()).unwrap_or(12);
    let warm_iters: usize = take("--warm").map(|s| s.parse().unwrap()).unwrap_or(0);
    let every: usize = take("--every").map(|s| s.parse().unwrap()).unwrap_or(5);
    let model = take("--model").unwrap_or("seg".into());
    let temporal: Option<u32> = match take("--temporal").as_deref() {
        None | Some("off") => None,
        Some("linear") => Some(1),
        Some("cubic") => Some(3),
        Some(o) => panic!("--temporal {o}"),
    };
    let phys = take("--phys").unwrap_or("full".into());
    let nseg: usize = take("--nseg").map(|s| s.parse().unwrap()).unwrap_or(if phys == "none" { 1 } else { 10 });
    let order: u32 = take("--order").map(|s| s.parse().unwrap()).unwrap_or(if phys == "full" { 2 } else { 0 });
    // relative floor of the circulant preconditioner's Fourier diagonal (larger = weaker, better-posed metric)
    let eps: f64 = take("--eps").map(|s| s.parse().unwrap()).unwrap_or(1e-3);
    let preconds: Vec<String> = take("--precond").unwrap_or("zmshr1".into()).split(',').map(|s| s.to_string()).collect();
    let out_arg = take("--out");
    let ckpt_every: usize = take("--ckpt-every").map(|s| s.parse().unwrap()).unwrap_or(10);
    // velocity-parametrised model (research/velocity_model_sr.md): --vmodel KZ,KV [--vdeg DZ,DV] [--vpoly d] [--vcont n]
    let vmodel: Option<(usize, usize)> = take("--vmodel").map(|s| { let v: Vec<usize> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) });
    let vdeg: (u32, u32) = take("--vdeg").map(|s| { let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).unwrap_or((1, 1));
    let vpoly: Option<usize> = take("--vpoly").map(|s| s.parse().unwrap());
    let vcont: usize = take("--vcont").map(|s| s.parse().unwrap()).unwrap_or(0);
    // z warm-up: first n iterations update z only (v and the residual phase frozen): makes z consistent with the
    // operator (B0, coils, physics) before velocity moves, so encode-common phase errors cannot leak into v
    let vzwarm: usize = take("--vzwarm").map(|s| s.parse().unwrap()).unwrap_or(0);
    // velocity-model prior strengths (LLR on the z nodes and on p = |z̄|·v; 0 = no prior), one run each
    let vkappas: Vec<f64> = take("--vkappa").unwrap_or("0".into()).split(',').map(|s| s.parse().unwrap()).collect();
    // memory planner (velocity path): device-usage cap (e.g. 8G; default: the whole card) and placement mode
    let vram_budget: Option<usize> = take("--vram-budget").map(|s| hasty_recon::plan::parse_bytes(&s).expect("--vram-budget e.g. 8G"));
    let plan_mode = take("--plan").unwrap_or("auto".into());
    // reconstruction grid relative to the native one: s or sx,sy,sz (research/velocity_model_sr.md §B2)
    // voxel basis function: dirac (default) | rect | bspline3, a per-sample weight Π sinc(k Δ / 2)^p
    let voxel_basis: Option<i32> = match take("--voxel-basis").as_deref() { None | Some("dirac") => None, Some("rect") => Some(1), Some("bspline3") => Some(4), Some(o) => panic!("--voxel-basis {o}") };
    let grid_scale: Option<[f64; 3]> = take("--grid-scale").map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); if v.len() == 1 { [v[0]; 3] } else { [v[0], v[1], v[2]] } });
    // test data: per-encode residual phase c_e + g_e·r (Legendre degree 1, amplitude AMP rad, seeded), injected exactly
    // as a coordinate shift + data phase (E_k(x e^{i g (r − n/2)}) = E_{k−g}(x)); recorded in run.json
    let inject: Option<(u64, f64)> = take("--inject-phase").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v[1].parse().unwrap()) });
    // static-tissue anchor for --vpoly: relative strength μ; mask from the simulator truth (outside the lumen band)
    // --vanchor μ[,auto|truth]: auto = static mask from the robust residual-phase fit (vinit), truth = simulator
    // auto refinement: pulsatile = temporal std of v over the nodes > mult × median (static mask), dilated by dil voxels
    let vanchor: Option<(f64, String)> = take("--vanchor").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v.get(1).unwrap_or(&"auto").to_string()) });
    let vanchor_refine: (f64, usize) = take("--vanchor-refine").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v.get(1).map_or(2, |d| d.parse().unwrap())) }).unwrap_or((1.5, 3));
    // learned prior (7-point dual-VENC data): export directory of tools/denoiser/export.py; κ mixes x + κ (D(x) − x)
    let learned_dir = take("--learned").map(PathBuf::from);
    let tile: Option<usize> = take("--tile").map(|s| s.parse().unwrap()).filter(|&t: &usize| t > 0);
    let keep_frames: Option<usize> = take("--keep-frames").map(|s| s.parse().unwrap()).filter(|&t: &usize| t > 0);
    // learned prior precision on CUDA: fp32 | bf16 (autocast, the training regime) | fp16
    let learned_prec = take("--learned-prec").unwrap_or("bf16".into());
    let cmdline = std::env::args().collect::<Vec<_>>().join(" ");
    let dir = PathBuf::from(&args[0]);
    assert!(model == "seg" || temporal.is_none(), "--temporal needs --model seg");
    // run output directory: arrays, metrics and figures (tools/figs/run_figs.py)
    let out = out_arg.map(PathBuf::from).unwrap_or_else(|| {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let tag = format!("{name}_{model}{}_{phys}{}", temporal.map_or(String::new(), |b| format!("_t{b}")), if gnl { "_gnl" } else { "" });
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/runs").join(tag)
    });
    std::fs::create_dir_all(&out).unwrap();
    if fresh {
        for e in std::fs::read_dir(&out).unwrap().flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with("state_") || n.starts_with("metrics_") || n.starts_with("recon_") || n.starts_with("vel_") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    let ctl = RunCtl::new(&out, ckpt_every);
    println!("output directory {} (checkpoint every {ckpt_every} iterations; Ctrl-C or touch {}/STOP to pause)", out.display(), out.display());
    let t0 = Instant::now();
    let mut sd = SimData::load(&dir).unwrap();
    if let Some(s) = grid_scale {
        sd.set_recon_grid(s);
        println!("reconstruction grid {:?} (native {:?}), voxel {:?} mm", sd.dims, sd.tdims, sd.vox.map(|v| (v * 1e4).round() / 10.0));
    }
    let sd = sd;
    let (nv, nc, nf) = (sd.nv(), sd.nc, sd.n_frames);
    // encodes (4-point referenced / Hadamard, 7-point dual-VENC, ...) and their velocity decoder
    let dec = FlowDecoder::new(&sd.acq.params.encode_m1());
    let ne = dec.n_enc();
    let learned = learned_dir.as_ref().map(|d| {
        assert_eq!(ne, 7, "--learned needs 7-point dual-VENC data (sim4d --venc-hi)");
        // the model's slot layout [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]: encode e ≥ 1 along axis (e − 1) / 2
        for (i, k) in dec.k().iter().enumerate() {
            let a = i / 2;
            assert!((0..3).all(|b| b == a || k[b].abs() < 1e-9 * k[a].abs()), "encode {} is not along axis {a}", i + 1);
        }
        let p = hasty_recon::learned::LearnedPrior::load(&hasty_recon::learned::default_plugin(), d, !learned_cpu).unwrap();
        if p.on_cuda {
            p.set_autocast(match learned_prec.as_str() { "fp32" => 0, "fp16" => 2, _ => 1 }).unwrap();
        }
        println!("learned prior {} on {} ({}), tiles {tile:?}, kept frames per window {keep_frames:?}", d.display(), if p.on_cuda { "CUDA" } else { "CPU" }, if p.on_cuda { learned_prec.as_str() } else { "fp32" });
        p
    });
    let kv7: [f32; 7] = if ne == 7 { std::array::from_fn(|e| dec.kv()[e] as f32) } else { [0.0; 7] };
    let w = sd.whitener();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    // sets: static t = frame·ne + encode (one frame each); temporal t = encode (all frames)
    let fe: Vec<(Option<usize>, usize)> = if temporal.is_some() { (0..ne).map(|e| (None, e)).collect() } else { (0..nf).flat_map(|f| (0..ne).map(move |e| (Some(f), e))).collect() };
    let enc_of = |t: usize| fe[t].1;
    let mut host_sets: Vec<SampleSet> = std::thread::scope(|sc| {
        let hs: Vec<_> = fe.iter().map(|&(f, e)| { let (sd, w) = (&sd, &w); sc.spawn(move || sd.sample_set(&sd.readouts_of(e, f), w)) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // injected residual phase: a[e-1] = [a0, a_x, a_y, a_z] (Legendre P0, P1(u_x), P1(u_y), P1(u_z), u = (2i+1)/n − 1)
    let injected: Vec<[f64; 4]> = match inject {
        None => vec![],
        Some((seed, amp)) => {
            let mut sgen = seed.max(1) ^ 0x2545F4914F6CDD1D;
            let mut u = || { sgen ^= sgen << 13; sgen ^= sgen >> 7; sgen ^= sgen << 17; 2.0 * ((sgen >> 11) as f64 / (1u64 << 53) as f64) - 1.0 };
            let a: Vec<[f64; 4]> = (1..ne).map(|_| [amp * u(), amp * u(), amp * u(), amp * u()]).collect();
            for (t, &(_, e)) in fe.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                let c = &a[e - 1];
                // φ = c0 + Σ_a (2 a_a / n_a)(i_a − n_a/2): constant c0 + Σ a_a / n_a, slope g_a = 2 a_a / n_a rad / voxel
                let c0 = c[0] + (0..3).map(|ax| c[1 + ax] / sd.dims[ax] as f64).sum::<f64>();
                let g: [f64; 3] = [0, 1, 2].map(|ax| 2.0 * c[1 + ax] / sd.dims[ax] as f64);
                let s = &mut host_sets[t];
                for ax in 0..3 {
                    for k in s.coords[ax].iter_mut() {
                        *k += g[ax] as f32;
                    }
                }
                let rot = Complex64::from_polar(1.0, c0);
                for y in s.y.iter_mut() {
                    let z = Complex64::new(y.re as f64, y.im as f64) * rot;
                    *y = C32::new(z.re as f32, z.im as f32);
                }
            }
            println!("injected residual phase (rad; P0, P1x, P1y, P1z per encode ≥ 1): {:?}", a.iter().map(|c| c.map(|v| (v * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
            a
        }
    };
    let cc = compression(&host_sets[0].y, host_sets[0].tau.len(), nc, nvc.min(nc));
    let nvc = cc.len();
    let grid = sd.physics_grid(gnl, &w);
    let mut smaps = apply_cc(&grid.smaps, nv, nc, &cc);
    let ys: Vec<CudaSlice<C32>> = host_sets.iter().map(|s| st.clone_htod(&apply_cc(&s.y, s.tau.len(), nc, &cc)).unwrap()).collect();
    let coords: Vec<[CudaSlice<f32>; 3]> = host_sets.iter().map(|s| [0, 1, 2].map(|a| st.clone_htod(&s.coords[a]).unwrap())).collect();
    let ms: Vec<usize> = host_sets.iter().map(|s| s.tau.len()).collect();
    let mut op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&host_sets[0].coords[0], &host_sets[0].coords[1], &host_sets[0].coords[2]], &smaps, nvc, vec![], 1e-4).unwrap();
    println!(
        "{} sets ({}), {}..{} readouts per set, {} virtual coils, model {model}, physics {phys}{}, setup {:.1}s",
        fe.len(),
        if temporal.is_some() { format!("{ne} encodes x {nf} frames, temporal B-spline {}", temporal.unwrap()) } else { format!("{nf} frames x {ne} encodes") },
        host_sets.iter().map(|s| s.readouts.len()).min().unwrap(),
        host_sets.iter().map(|s| s.readouts.len()).max().unwrap(),
        nvc,
        if gnl { " + GNL" } else { "" },
        t0.elapsed().as_secs_f64()
    );
    // ESPIRiT-style normalisation Σ_c |S_c|² = 1 (receive bias moves into the image intensity; phases and hence
    // velocities are unchanged). Raw Biot–Savart maps blow up next to the loops and dominate λ_max of the normal operator.
    if std::env::var("HASTY_RAW_MAPS").is_err() {
        let dmax = (0..nv).map(|q| (0..nvc).map(|c| smaps[c * nv + q].re.powi(2) + smaps[c * nv + q].im.powi(2)).sum::<f32>()).fold(0f32, f32::max);
        for q in 0..nv {
            let d: f32 = (0..nvc).map(|c| smaps[c * nv + q].re.powi(2) + smaps[c * nv + q].im.powi(2)).sum();
            let sc = 1.0 / (d + 1e-6 * dmax).sqrt();
            for c in 0..nvc {
                smaps[c * nv + q].re *= sc;
                smaps[c * nv + q].im *= sc;
            }
        }
    }
    // per-encode mean concomitant phase folded into coil maps (sense model only)
    let wc = hasty_mrsim::GAMMA / (2.0 * sd.b0);
    let mut maps_host: Vec<Vec<C32>> = if conc_dc && model == "sense" {
        (0..ne)
            .map(|e| {
                let np = sd.acq.pre[e].len();
                let mut pp = [0.0f64; 6];
                for j in 0..np {
                    let ph = sd.acq.pre[e][j].rotate(&sd.rot).p;
                    for q in 0..6 {
                        pp[q] += ph[q] / np as f64;
                    }
                }
                let wq = [wc * (pp[0] + pp[1]), wc * pp[2] / 4.0, -wc * pp[4], -wc * pp[5]];
                let mut m = smaps.clone();
                for q in 0..nv {
                    let phi: f64 = (0..4).map(|i| wq[i] * grid.mono[q][i]).sum();
                    let e = Complex64::from_polar(1.0, -phi);
                    for c in 0..nvc {
                        let s = m[c * nv + q];
                        let z = Complex64::new(s.re as f64, s.im as f64) * e;
                        m[c * nv + q] = C32::new(z.re as f32, z.im as f32);
                    }
                }
                m
            })
            .collect()
    } else {
        vec![]
    };
    // warm start: time-averaged plain-model reconstruction per encode (all frames pooled), PCGLS + circulant
    let warm: Vec<Vec<C32>> = if warm_iters > 0 {
        (0..ne)
            .map(|e| {
                let idx: Vec<usize> = (0..fe.len()).filter(|&t| enc_of(t) == e).collect();
                let cat: [Vec<f32>; 3] = [0, 1, 2].map(|a| idx.iter().flat_map(|&t| host_sets[t].coords[a].iter().copied()).collect());
                let m: usize = idx.iter().map(|&t| ms[t]).sum();
                let mut y = vec![C32::new(0.0, 0.0); nvc * m];
                let mut off = 0;
                for &t in &idx {
                    let yt = st.clone_dtoh(&ys[t]).unwrap();
                    for c in 0..nvc {
                        y[c * m + off..c * m + off + ms[t]].copy_from_slice(&yt[c * ms[t]..(c + 1) * ms[t]]);
                    }
                    off += ms[t];
                }
                let maps_e: &[C32] = if !maps_host.is_empty() { &maps_host[e] } else { &smaps };
                let mut op_e = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&cat[0], &cat[1], &cat[2]], maps_e, nvc, vec![], 1e-4).unwrap();
                let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&cat[0], &cat[1], &cat[2]], maps_e, nvc, 1e-3).unwrap();
                let done = format!("warm_done_{e}");
                if let Some(v) = ctl.load(&done).unwrap() {
                    return Some(st.clone_dtoh(&value_dev(&st, &v, nv).unwrap()).unwrap());
                }
                let dy = st.clone_htod(&y).unwrap();
                let mut xe = st.alloc_zeros::<C32>(nv).unwrap();
                let (_, oc) = pcgls_ctl(&mut op_e, Some(&mut pre), &st, &dy, &mut xe, 0.0, warm_iters, &mut |_, _| {}, Some((&ctl, &format!("warm{e}")))).unwrap();
                if let Outcome::Stopped(k) = oc {
                    println!("stopped in the warm start (encode {e}, CG iteration {k}): state saved, rerun the same command to resume");
                    return None;
                }
                ctl.save(&done, &hasty_recon::checkpoint::dev_value(&st, &xe).unwrap()).unwrap();
                Some(st.clone_dtoh(&xe).unwrap())
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_else(|| std::process::exit(0))
    } else {
        vec![]
    };
    println!("warm start: {} PCGLS iterations per encode ({:.0}s)", warm_iters, t0.elapsed().as_secs_f64());
    // object support: smoothed magnitude of the warm start (or of the summed adjoint of the reference encode without
    // one), thresholded, enclosed low-signal regions filled per slice (background = what a flood fill from the slice
    // border reaches through sub-threshold voxels)
    let support: Vec<bool> = {
        let mut acc = vec![0f32; nv];
        if !warm.is_empty() {
            for w in &warm {
                for q in 0..nv {
                    acc[q] += (w[q].re * w[q].re + w[q].im * w[q].im).sqrt();
                }
            }
        } else {
            let mut tmp = st.alloc_zeros::<C32>(nv).unwrap();
            for t in (0..fe.len()).filter(|&t| enc_of(t) == 0) {
                let c = &coords[t];
                op.set_points(&st, [&c[0], &c[1], &c[2]], ms[t]).unwrap();
                op.adjoint(&st, &ys[t], &mut tmp).unwrap();
                let h = st.clone_dtoh(&tmp).unwrap();
                for q in 0..nv {
                    acc[q] += (h[q].re * h[q].re + h[q].im * h[q].im).sqrt();
                }
            }
        }
        // smooth (3 box passes, radius 4 in-plane / 1 through-plane, σ ≈ 4.5 / 1.4 voxels) so that dark structures
        // inside the body (vessel lumen with flow dephasing, lungs) do not open holes in the mask
        let d = sd.dims;
        for (axis, r) in [(0usize, 4usize), (1, 4), (2, 1)] {
            let (n, stride) = (d[axis], [1, d[0], d[0] * d[1]][axis]);
            for _ in 0..3 {
                let src = acc.clone();
                for q in 0..nv {
                    let i = (q / stride) % n;
                    let (lo, hi) = (i.saturating_sub(r), (i + r).min(n - 1));
                    let base = q - i * stride;
                    acc[q] = (lo..=hi).map(|k| src[base + k * stride]).sum::<f32>() / (hi - lo + 1) as f32;
                }
            }
        }
        let mut sorted = acc.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = 0.05 * sorted[sorted.len() * 99 / 100];
        let fg: Vec<bool> = acc.iter().map(|&v| v > thr).collect();
        let mut bg = vec![false; nv];
        for k in 0..d[2] {
            let mut stack: Vec<(usize, usize)> = Vec::new();
            for i in 0..d[0] {
                stack.push((i, 0));
                stack.push((i, d[1] - 1));
            }
            for j in 0..d[1] {
                stack.push((0, j));
                stack.push((d[0] - 1, j));
            }
            while let Some((i, j)) = stack.pop() {
                let q = i + d[0] * (j + d[1] * k);
                if bg[q] || fg[q] {
                    continue;
                }
                bg[q] = true;
                if i > 0 { stack.push((i - 1, j)); }
                if i + 1 < d[0] { stack.push((i + 1, j)); }
                if j > 0 { stack.push((i, j - 1)); }
                if j + 1 < d[1] { stack.push((i, j + 1)); }
            }
        }
        (0..nv).map(|q| !bg[q]).collect()
    };
    println!("support {} voxels", support.iter().filter(|&&b| b).count());
    // coil maps zero outside the (dilated) object support, as ESPIRiT maps are: no unconstrained air modes
    let mut dil = support.clone();
    {
        let d = sd.dims;
        for _ in 0..3 {
            let prev = dil.clone();
            for q in 0..nv {
                if prev[q] {
                    continue;
                }
                let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
                let nb = [(i > 0).then(|| q - 1), (i + 1 < d[0]).then(|| q + 1), (j > 0).then(|| q - d[0]), (j + 1 < d[1]).then(|| q + d[0]), (k > 0).then(|| q - d[0] * d[1]), (k + 1 < d[2]).then(|| q + d[0] * d[1])];
                dil[q] = nb.iter().flatten().any(|&n| prev[n]);
            }
        }
    }
    if std::env::var("HASTY_NO_MASK").is_err() {
        for q in (0..nv).filter(|&q| !dil[q]) {
            for c in 0..nvc {
                smaps[c * nv + q] = C32::new(0.0, 0.0);
                for m in maps_host.iter_mut() {
                    m[c * nv + q] = C32::new(0.0, 0.0);
                }
            }
        }
        op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&host_sets[0].coords[0], &host_sets[0].coords[1], &host_sets[0].coords[2]], &smaps, nvc, vec![], 1e-4).unwrap();
        println!("coil maps masked to {} voxels", dil.iter().filter(|&&b| b).count());
    }
    let maps: Vec<CudaSlice<C32>> = maps_host.iter().map(|m| st.clone_htod(m).unwrap()).collect();
    // lumen and truth at the true positions of the reconstruction voxels
    let lumen: Vec<usize> = (0..nv).filter(|&q| sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) > 0.9).collect();
    println!("lumen {} voxels: {} in the support, {} in the masked maps", lumen.len(), lumen.iter().filter(|&&q| support[q]).count(), lumen.iter().filter(|&&q| dil[q]).count());
    let dims4 = |n: usize| [sd.dims[0], sd.dims[1], sd.dims[2], n];
    save(&out.join("smaps.npy"), "<c8", &dims4(nvc), c8(&smaps));
    save(&out.join("mask_support.npy"), "<f4", &sd.dims, f4(support.iter().map(|&b| b as u8 as f32)));
    save(&out.join("mask_maps.npy"), "<f4", &sd.dims, f4(dil.iter().map(|&b| b as u8 as f32)));
    if !warm.is_empty() {
        save(&out.join("warm.npy"), "<c8", &dims4(ne), c8(&warm.concat()));
    }
    // fixed circulant preconditioners per frame (block-diagonal over the frames of a time-resolved set)
    // (the velocity model uses its own Gauss–Newton metric: no circulant kernels, they would cost ~8 GB)
    let need_circ = vmodel.is_none() && preconds.iter().any(|p| p == "circ" || p == "zmshr1m");
    let circ: Vec<Vec<CirculantPrecond>> = if need_circ {
        host_sets
            .iter()
            .map(|s| {
                let frames: Vec<Option<usize>> = if temporal.is_some() { (0..nf).map(Some).collect() } else { vec![None] };
                frames
                    .iter()
                    .map(|&f| {
                        let sel: Vec<usize> = (0..s.readouts.len()).filter(|&r| f.is_none_or(|f| sd.acq.readouts[s.readouts[r]].frame == f)).collect();
                        let c: [Vec<f32>; 3] = [0, 1, 2].map(|a| sel.iter().flat_map(|&r| s.coords[a][r * sd.na..(r + 1) * sd.na].iter().copied()).collect());
                        CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&c[0], &c[1], &c[2]], &smaps, nvc, eps).unwrap()
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };
    if !warm.is_empty() {
        // diagnostic: warm-start velocity vs the time-averaged truth over the lumen
        let (mut mr, mut mt, mut e2, mut n) = ([0.0f64; 3], [0.0f64; 3], 0.0f64, 0usize);
        for &q in &lumen {
            let z: Vec<Complex64> = (0..ne).map(|e| Complex64::new(warm[e][q].re as f64, warm[e][q].im as f64)).collect();
            let vq = dec.velocity(&z);
            for a in 0..3 {
                let v = vq[a];
                let t = (0..nf).map(|f| sd.sample_log(&sd.truth_vel, a + 3 * f, &grid.x_log[q])).sum::<f64>() / nf as f64;
                mr[a] += v;
                mt[a] += t;
                e2 += (v - t).powi(2);
            }
            n += 1;
        }
        let mut wv = vec![0f32; 3 * nv];
        for q in 0..nv {
            let z: Vec<Complex64> = (0..ne).map(|e| Complex64::new(warm[e][q].re as f64, warm[e][q].im as f64)).collect();
            let v = dec.velocity(&z);
            for a in 0..3 {
                wv[q + nv * a] = v[a] as f32;
            }
        }
        save(&out.join("warm_vel.npy"), "<f4", &dims4(3), f4(wv));
        println!("warm-start velocity in the lumen: mean {:?} m/s, truth (time average) {:?}, RMSE {:.3} m/s", mr.map(|v| format!("{:.3}", v / n as f64)), mt.map(|v| format!("{:.3}", v / n as f64)), (e2 / n as f64).sqrt());
    }
    // the operator stack
    let mut sense_sets;
    let mut seg_sets;
    let sop: &mut dyn SetOp = if model == "sense" {
        sense_sets = SenseSets { op: &mut op, coords, ms: ms.clone(), maps, map_of_set: if conc_dc { (0..fe.len()).map(enc_of).collect() } else { vec![] } };
        &mut sense_sets
    } else {
        let ts = Instant::now();
        let (with_b0, with_conc) = (phys != "none", phys == "full");
        let fit_vox: Vec<usize> = (0..nv).filter(|&q| dil[q]).collect();
        let fit = fit_monomials(&sd, &grid, &fit_vox);
        println!("monomial quadratic fit over {} voxels: rms {:.1e} / max {:.1e} m² (z², x²+y², xz, yz: {:?})", fit_vox.len(), fit.rms.iter().cloned().fold(0.0, f64::max), fit.max.iter().cloned().fold(0.0, f64::max), fit.max.map(|v| format!("{v:.1e}")));
        let mut eng = SegNufft::new(&dev, sd.dims, sd.vox, nvc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames: if temporal.is_some() { nf } else { 1 }, bspline: temporal.unwrap_or(1), max_grid_bytes: 6 << 30 }).unwrap();
        eng.set_maps(&st, &smaps).unwrap();
        // segments per encode from the encode-wide mean coefficients (all frames)
        let wbar: Vec<Vec<[f64; 4]>> = (0..ne).map(|e| mean_coefficients(&(0..fe.len()).filter(|&t| enc_of(t) == e).map(|t| &host_sets[t]).collect::<Vec<_>>(), sd.na)).collect();
        let wbar: Vec<Vec<[f64; 4]>> = if with_conc { wbar } else { vec![vec![[0.0; 4]; sd.na]; ne] };
        let tau_adc: Vec<f64> = (0..sd.na).map(|n| host_sets[0].tau[n]).collect();
        let seg_fit: Vec<usize> = fit_vox.iter().copied().step_by(3).collect();
        let segs: Vec<_> = (0..ne)
            .map(|e| {
                let psi = segment_phase(&grid, &wbar[e], &tau_adc, with_b0);
                let (sp, at, ta, dt) = build_segments(nv, nseg, tau_adc[0], tau_adc[sd.na - 1], 256, &seg_fit, &psi);
                eng.segments(&st, &sp, &at, ta, dt).unwrap()
            })
            .collect();
        let mut resid = 0.0f64;
        let pts: Vec<_> = (0..fe.len())
            .map(|t| {
                let tab = seg_tables(&sd, &host_sets[t], &fit, with_conc, with_conc.then(|| &wbar[enc_of(t)][..]));
                resid = resid.max(tab.max_resid_phase);
                eng.points(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, temporal.map(|_| &tab.t[..])).unwrap()
            })
            .collect();
        let (kf, ka) = eng.autotune(&st, &segs[0], &pts[0]).unwrap();
        println!(
            "SegNufft: L = {nseg}, order {order}, frames/set {}, coil batch {}, kernels {kf:.1}/{ka:.1} ms (set 0), max phase left out by the quadratic fit {resid:.2e} rad, built in {:.1}s",
            eng.frames,
            eng.cb,
            ts.elapsed().as_secs_f64()
        );
        seg_sets = SegSets { eng, pts, segs, seg_of_set: (0..fe.len()).map(enc_of).collect(), kern: kern.clone() };
        &mut seg_sets
    };
    let mut wsets;
    let sop: &mut dyn SetOp = match voxel_basis {
        None => sop,
        Some(pw) => {
            let w: Vec<Vec<C32>> = host_sets.iter().map(|s| hasty_recon::setop::WeightedSets::basis_weights([&s.coords[0], &s.coords[1], &s.coords[2]], pw)).collect();
            let wmin = w.iter().flatten().map(|c| c.re).fold(f32::INFINITY, f32::min);
            println!("voxel basis {}: per-sample weights, min {wmin:.3}", if pw == 1 { "rect" } else { "cubic B-spline" });
            wsets = hasty_recon::setop::WeightedSets::new(&st, sop, w).unwrap();
            &mut wsets
        }
    };
    let ns = sop.nsets();
    let lens: Vec<usize> = (0..ns).map(|t| sop.len_x(t)).collect();
    // image (frame f, encode e) inside the stack: (set, offset)
    let at = |f: usize, e: usize| -> (usize, usize) { if temporal.is_some() { (e, f * nv) } else { (f * ne + e, 0) } };
    // model-consistent test data: y_t = E_t x_true,t + noise (solver + prior test without simulator physics)
    let mut ys = ys;
    if synthetic {
        assert!(!warm.is_empty(), "--synthetic needs --warm");
        assert!(sd.native_grid(), "--synthetic indexes the truth by voxel: native grid only");
        let sigma = sd.meta["noise_sigma"].as_f64().unwrap() / 2f64.sqrt() * std::env::var("NOISE_SCALE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
        let mut seed = 0x51u64;
        let mut gauss = || {
            let mut u = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
            (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
        };
        let mut xt = vec![vec![C32::new(0.0, 0.0); 0]; ns];
        for t in 0..ns {
            xt[t] = vec![C32::new(0.0, 0.0); lens[t]];
        }
        for fr in 0..nf {
            for e in 0..ne {
                let (t, o) = at(fr, e);
                for q in 0..nv {
                    let r = warm[0][q];
                    let mut z = Complex64::new(r.re as f64, r.im as f64);
                    if sd.truth_frac[q] > 0.0 {
                        let v = [0, 1, 2].map(|a| sd.truth_vel[q + nv * (a + 3 * fr)] as f64);
                        z *= Complex64::from_polar(1.0, dec.phase(e, &v) - dec.phase(0, &v));
                    }
                    xt[t][o + q] = C32::new(z.re as f32, z.im as f32);
                }
            }
        }
        for t in 0..ns {
            let dx = st.clone_htod(&xt[t]).unwrap();
            let mut y = st.alloc_zeros::<C32>(sop.len_y(t)).unwrap();
            sop.forward(&st, t, &dx, &mut y).unwrap();
            let mut h = st.clone_dtoh(&y).unwrap();
            for v in h.iter_mut() {
                v.re += (sigma * gauss()) as f32;
                v.im += (sigma * gauss()) as f32;
            }
            ys[t] = st.clone_htod(&h).unwrap();
        }
        println!("synthetic model-consistent data (noise sigma {:.2} per component)", sigma);
    }
    if let Some((kz, kv)) = vmodel {
        assert!(!warm.is_empty(), "--vmodel needs --warm (initialisation from the time-averaged reconstruction)");
        use hasty_recon::vmodel::{cpu::Theta, VelModel, VelSpec};
        use hasty_recon::vobj::VelProblem;
        let m1 = sd.acq.params.encode_m1();
        // centred encodings k̃_e = γ M1_e − mean_e: z becomes the mean-encode reference, which makes the Gauss–Newton
        // z–v cross term Σ_e k̃_e |x_e|² vanish exactly, so encode-common phase errors (B0, coil, warm-start
        // mismatch) no longer drive v. Velocity is unchanged (only encode differences carry it).
        let kraw: Vec<[f64; 3]> = m1.iter().map(|m| m.map(|x| hasty_mrsim::GAMMA * x)).collect();
        let kmean: [f64; 3] = [0, 1, 2].map(|a| kraw.iter().map(|k| k[a]).sum::<f64>() / kraw.len() as f64);
        // (opt-in: centring makes z carry the velocity-dependent mean-encode phase, which breaks its temporal
        // smoothness: 7-point 0.138 → 0.174 m/s; the z warm-up below fixes the coupling instead)
        let kenc: Vec<[f64; 3]> = if std::env::var("HASTY_VM_CENTRED").is_err() { kraw.clone() } else { kraw.iter().map(|k| [0, 1, 2].map(|a| k[a] - kmean[a])).collect() };
        let spec = VelSpec { dims: sd.dims, kenc: kenc.clone(), nf, kz, deg_z: vdeg.0, kv, deg_v: vdeg.1, poly: vpoly };
        let vm = VelModel::new(&dev, &st, spec.clone()).unwrap();
        // initialisation: time-averaged warm start, z from the reference encode, v by the hierarchical decoder (inside
        // the support), no residual phase
        let mut th = Theta::zeros(&spec);
        let warm64: Vec<Vec<Complex64>> = warm.iter().map(|w| w.iter().map(|c| Complex64::new(c.re as f64, c.im as f64)).collect()).collect();
        // residual phase: robust static-tissue fit to the warm-start phase differences; it also yields the static mask
        let sfit = vpoly.map(|d| {
            let f = hasty_recon::vinit::fit_static_phase(&warm64, &support, sd.dims, d, 0.05);
            println!("residual-phase init (robust static fit, degree {d}): σ per encode {:?} rad, static mask {} voxels", f.sigma.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>(), f.mask.iter().filter(|&&m| m > 0.0).count());
            if !injected.is_empty() && d == 1 {
                let np = 4;
                let err: f64 = (1..ne).map(|e| (0..np).map(|p| (f.a[(e - 1) * np + p] - injected[e - 1][p]).powi(2)).sum::<f64>()).sum::<f64>().sqrt() / ((ne - 1) as f64 * 4.0).sqrt();
                println!("  init residual-phase RMS error vs injected: {err:.4} rad");
            }
            th.a.copy_from_slice(&f.a);
            f
        });
        let warm_c = match &sfit { Some(f) => hasty_recon::vinit::remove_phase(&warm64, &f.a, sd.dims, vpoly.unwrap()), None => warm64.clone() };
        for q in 0..nv {
            let z: Vec<Complex64> = (0..ne).map(|e| warm_c[e][q]).collect();
            let v = if support[q] { dec.velocity(&z) } else { [0.0; 3] };
            let z0 = z[0] * Complex64::from_polar(1.0, kenc[0][0] * v[0] + kenc[0][1] * v[1] + kenc[0][2] * v[2]);
            for k in 0..kz {
                th.z[k][q] = z0;
            }
            for k in 0..kv {
                th.v[k][q] = v;
            }
        }
        let coil: Vec<f32> = (0..nv).map(|q| (0..nvc).map(|c| { let s = smaps[c * nv + q]; s.re * s.re + s.im * s.im }).sum()).collect();
        let lens = spec.lens();
        let mut pr = VelProblem::new(&st, kern.clone(), sop, ys, vm, Some(coil)).unwrap();
        if let Some((mu, mode)) = &vanchor {
            let w: Vec<f32> = match (mode.as_str(), &sfit) {
                // oracle: inside the coil-map mask, no blood at the voxel's true position
                ("truth", _) => (0..nv).map(|q| (dil[q] && sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) < 1e-3) as u8 as f32).collect(),
                (_, Some(f)) => f.mask.clone(),
                _ => panic!("--vanchor auto needs --vpoly (the static mask comes from the residual-phase fit)"),
            };
            let wrong = (0..nv).filter(|&q| w[q] > 0.0 && sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) > 0.1).count();
            println!("static-tissue anchor μ {mu} ({mode}) on {} voxels ({wrong} of them blood by the simulator truth)", w.iter().filter(|&&x| x > 0.0).count());
            pr.set_anchor(&st, &w, *mu).unwrap();
        }
        // memory plan: what is resident now, what the run needs, where the placeable buffers go
        use hasty_recon::plan::{choose, probe, Need};
        let pb = probe(&dev.ctx, &st).unwrap();
        let (data_b, set_max) = pr.data_bytes();
        let budget = vram_budget.unwrap_or(pb.total.saturating_sub(512 << 20)).min(pb.total);
        let quasi = preconds.iter().any(|p| p == "zmshr1" || p == "zmshr1m");
        let mut need = Need { fixed: (pb.total - pb.free).saturating_sub(data_b), stack: spec.stack_bytes(), chunk_max: lens.iter().max().unwrap() * 8, data: data_b, set_max, quasi, gradient_s: 0.0 };
        let forced = |data: bool, prev: bool, w: hasty_recon::p2np::WMode| -> hasty_recon::plan::Plan {
            let mut p = choose(&need, &pb, usize::MAX, Some(data)).unwrap();
            p.prev_on_host = prev;
            p.w_mode = w;
            p
        };
        let mut plan = match plan_mode.as_str() {
            "device" => forced(false, false, hasty_recon::p2np::WMode::Device),
            "host" => forced(true, true, hasty_recon::p2np::WMode::Host),
            _ => choose(&need, &pb, budget, None).unwrap_or_else(|e| panic!("{e}")),
        };
        println!("device: {:.2} GiB total, {:.2} GiB free; pinned bandwidth h2d {:.1} / d2h {:.1} GB/s; budget {:.2} GiB", pb.total as f64 / 1073741824.0, pb.free as f64 / 1073741824.0, pb.h2d / 1e9, pb.d2h / 1e9, budget as f64 / 1073741824.0);
        if plan.data_on_host {
            let freed = pr.data_to_host(&dev.ctx, &st).unwrap();
            println!("k-space data moved to pinned host memory ({:.2} GiB freed)", freed as f64 / 1073741824.0);
        }
        print!("{}", plan.describe());
        // VENC continuation: the low-VENC encodes (all but the least sensitive group) are left out of the first stage
        let kn: Vec<f64> = (0..ne).map(|e| (0..3).map(|a| (kenc[e][a] - kenc[0][a]).powi(2)).sum::<f64>().sqrt()).collect();
        let kmin = kn.iter().copied().filter(|&x| x > 0.0).fold(f64::INFINITY, f64::min);
        let low: Vec<bool> = kn.iter().map(|&x| x > kmin * (1.0 + 1e-6)).collect();
        let vtag = |pc: &str, kap: f64| format!("{model}_{pc}_vm{kz}x{kv}_d{}{}{}{}{}{}", vdeg.0, vdeg.1, vpoly.map_or(String::new(), |d| format!("_p{d}")), if vcont > 0 { format!("_c{vcont}") } else { String::new() }, if vzwarm > 0 { format!("_z{vzwarm}") } else { String::new() }, if kap > 0.0 { format!("_llr{kap}") } else { String::new() });
        let runs: Vec<String> = preconds.iter().flat_map(|p| vkappas.iter().map(move |&k| (p, k))).map(|(p, k)| vtag(p, k)).collect();
        let info = serde_json::json!({
            "cmdline": cmdline, "sim_dir": dir, "model": model, "phys": phys, "gnl": gnl, "iters": iters, "warm_iters": warm_iters, "vcoils": nvc, "dims": sd.dims, "vox": sd.vox, "native_dims": sd.tdims,
            "n_frames": nf, "n_enc": ne, "venc": sd.venc, "venc_range": dec.range(), "encoding": sd.meta["encoding"], "runs": runs,
            "prior": if vkappas.iter().any(|&k| k > 0.0) { "llr(z, p)" } else { "none" }, "vkappas": vkappas, "vmodel": {"kz": kz, "kv": kv, "deg": [vdeg.0, vdeg.1], "poly": vpoly, "vcont": vcont, "low_venc_encodes": low, "anchor": vanchor.as_ref().map(|(m, s)| format!("{m},{s}"))}, "injected_phase": injected, "setup_s": t0.elapsed().as_secs_f64(),
        });
        std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
        let rt = transpose(&sd.rot);
        for (pc, kap) in preconds.iter().flat_map(|p| vkappas.iter().map(move |&k| (p, k))) {
            let tag = vtag(pc, kap);
            let precond = match pc.as_str() { "none" => Precond::None, "circ" => Precond::Circulant, "zmshr1" => Precond::Zmshr1, _ => Precond::Zmshr1M };
            let with_metric = matches!(precond, Precond::Circulant | Precond::Zmshr1M);
            let resuming = out.join(format!("state_{tag}_s0.hv")).exists() || out.join(format!("state_{tag}_s1.hv")).exists() || out.join(format!("vdone_{tag}_s0")).exists();
            if !resuming && out.join(format!("recon_{tag}.npy")).exists() {
                println!("  {tag}: finished in an earlier run, skipped (--fresh to redo)");
                continue;
            }
            let mut x = Stack { v: th.pack(&spec).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
            pr.reset();
            let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(out.join(format!("metrics_{tag}.csv"))).unwrap();
            if !resuming {
                use std::io::Write;
                csv.set_len(0).unwrap();
                writeln!(csv, "iter,f,rmse,rel,wrap_frac,seconds,{}", (0..nf).map(|f| format!("frame{f}")).collect::<Vec<_>>().join(",")).unwrap();
            }
            let tr = Instant::now();
            let wv = pr.model.wv.clone();
            // velocity of frame fr at voxel q from host copies of the v chunks
            let vel_of = |vh: &[Vec<f32>], fr: usize, q: usize| -> [f64; 3] {
                let mut v = [0.0; 3];
                for k in 0..kv {
                    let w = wv[fr * kv + k];
                    if w != 0.0 {
                        for a in 0..3 {
                            v[a] += w * vh[k][a * nv + q] as f64;
                        }
                    }
                }
                if gnl {
                    v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v)));
                }
                v
            };
            let mut stages: Vec<(usize, bool)> = if vcont > 0 && low.iter().any(|&b| b) { vec![(vcont.min(iters), true), (iters.saturating_sub(vcont), false)] } else { vec![(iters, false)] };
            // z warm-up as an extra first stage (all encodes, v frozen); stage indices shift by one
            let zstage = vzwarm > 0;
            if zstage {
                stages.insert(0, (vzwarm, false));
            }
            let mut k_off = 0;
            // the automatic anchor waits for pulsatility: stage 0 without it, refined mask from stage 1 on
            let anchor_auto = matches!(&vanchor, Some((_, m)) if m == "auto") && stages.len() > 1;
            let fit_mask: Option<Vec<f32>> = sfit.as_ref().map(|f| f.mask.clone());
            for (si, &(n_it, only_hi)) in stages.iter().enumerate() {
                let stag = format!("{tag}_s{si}");
                pr.freeze_v = zstage && si == 0;
                let si = if zstage { si.saturating_sub(1) } else { si }; // continuation / anchor logic sees the original stages
                if anchor_auto && si == 0 {
                    pr.clear_anchor();
                }
                if anchor_auto && si == 1 {
                    // static = accepted by the robust fit, low temporal variation of v over the nodes, and not
                    // within 2 voxels of a pulsatile voxel
                    let vh: Vec<Vec<f32>> = (0..kv).map(|kk| st.clone_dtoh(&x.v[spec.chunk_v(kk)]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect()).collect();
                    let std: Vec<f64> = (0..nv).map(|q| {
                        let mean: [f64; 3] = [0, 1, 2].map(|a| (0..kv).map(|kk| vh[kk][a * nv + q] as f64).sum::<f64>() / kv as f64);
                        ((0..kv).map(|kk| (0..3).map(|a| (vh[kk][a * nv + q] as f64 - mean[a]).powi(2)).sum::<f64>()).sum::<f64>() / kv as f64).sqrt()
                    }).collect();
                    let fm = fit_mask.as_ref().unwrap();
                    let mut sv: Vec<f64> = (0..nv).filter(|&q| fm[q] > 0.0).map(|q| std[q]).collect();
                    sv.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let tau = (vanchor_refine.0 * sv.get(sv.len() / 2).copied().unwrap_or(0.0)).max(0.02);
                    let mut puls: Vec<bool> = (0..nv).map(|q| support[q] && std[q] > tau).collect();
                    let d = sd.dims;
                    for _ in 0..vanchor_refine.1 {
                        let prev = puls.clone();
                        for q in 0..nv {
                            if prev[q] {
                                continue;
                            }
                            let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
                            let nb = [(i > 0).then(|| q - 1), (i + 1 < d[0]).then(|| q + 1), (j > 0).then(|| q - d[0]), (j + 1 < d[1]).then(|| q + d[0]), (k > 0).then(|| q - d[0] * d[1]), (k + 1 < d[2]).then(|| q + d[0] * d[1])];
                            puls[q] = nb.iter().flatten().any(|&n| prev[n]);
                        }
                    }
                    let w: Vec<f32> = (0..nv).map(|q| (fm[q] > 0.0 && !puls[q]) as u8 as f32).collect();
                    let wrong = (0..nv).filter(|&q| w[q] > 0.0 && sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) > 0.1).count();
                    println!("  anchor refined by pulsatility (std of v over nodes > {tau:.3} m/s, dilated {}):", vanchor_refine.1);
                    println!("    {} voxels ({wrong} blood by the simulator truth)", w.iter().filter(|&&x| x > 0.0).count());
                    pr.set_anchor(&st, &w, vanchor.as_ref().unwrap().0).unwrap();
                }
                let done = out.join(format!("vdone_{stag}"));
                // set t holds encode t % E (per-frame sets t = f·E + e; temporal sets t = e)
                for t in 0..pr.weights.len() {
                    pr.weights[t] = if only_hi && low[t % ne] { 0.0 } else { 1.0 };
                }
                if done.exists() {
                    if let Some(v) = ctl.load(&format!("{stag}_final")).unwrap() {
                        x = hasty_recon::checkpoint::value_stack(&st, &v, &lens).unwrap();
                    }
                    k_off += n_it;
                    continue;
                }
                let tl = Instant::now();
                let lam = pr.lambda_max(&st, &x, with_metric, 12).unwrap();
                st.synchronize().unwrap();
                println!("  {tag} stage {si} ({}): {n_it} iterations, λ_max({}) = {lam:.3e}", if only_hi { "reference + high-VENC encodes" } else { "all encodes" }, if with_metric { "M H" } else { "H" });
                if vanchor.is_some() {
                    println!("    anchor λ {:.3e} (n̂ {:.3e})", pr.anchor_lam, pr.nhat);
                }
                // refine the state placement with the measured cost of one gradient (≈ one Hessian-vector product)
                if plan_mode == "auto" && need.gradient_s == 0.0 {
                    need.gradient_s = tl.elapsed().as_secs_f64() / 12.0;
                    plan = choose(&need, &pb, budget, Some(plan.data_on_host)).unwrap_or_else(|e| panic!("{e}"));
                    print!("  measured {:.3} s per gradient; {}", need.gradient_s, plan.describe());
                }
                let sopts = plan.state_opts(&dev.ctx);
                let mut report = |k: usize, xs: &Stack, f: f64| {
                    let k = if k == usize::MAX { k_off } else { k + 1 + k_off };
                    if k % every != 0 && k != iters && k != 0 {
                        return;
                    }
                    let vh: Vec<Vec<f32>> = (0..kv).map(|kk| st.clone_dtoh(&xs.v[spec.chunk_v(kk)]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect()).collect();
                    let (mut e2, mut t2, mut n, mut wrap) = (0.0f64, 0.0f64, 0usize, 0usize);
                    let mut per_frame = Vec::with_capacity(nf);
                    for fr in 0..nf {
                        let (mut fe2, mut fn_) = (0.0, 0usize);
                        for &q in &lumen {
                            let v = vel_of(&vh, fr, q);
                            for a in 0..3 {
                                let t = sd.sample_log(&sd.truth_vel, a + 3 * fr, &grid.x_log[q]);
                                let d = v[a] - t;
                                e2 += d * d;
                                t2 += t * t;
                                fe2 += d * d;
                                wrap += (d.abs() > 0.8 * sd.venc.min(dec.range()[a])) as usize;
                            }
                            n += 1;
                            fn_ += 1;
                        }
                        per_frame.push((fe2 / fn_.max(1) as f64).sqrt());
                    }
                    let wf = wrap as f64 / (3 * n).max(1) as f64;
                    {
                        use std::io::Write;
                        writeln!(csv, "{k},{f:.6e},{:.5},{:.5},{wf:.5},{:.1},{}", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), tr.elapsed().as_secs_f64(), per_frame.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")).unwrap();
                    }
                    println!("  {pc:<7} vm it {k:>3}: f = {f:.4e}, velocity RMSE {:.3} m/s (rel {:.3}), wrapped {:.2} %, frames {}, {:.0}s", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), 100.0 * wf, per_frame.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "), tr.elapsed().as_secs_f64());
                };
                // prior: LLR on the z nodes and, separately, on p = |z̄|·v (3 K_v real images), then v = p / |z̄|
                let (mut llr_z, mut llr_p) = (Llr::new(sd.dims, block, kap, support.clone()), Llr::new(sd.dims, block, kap, support.clone()));
                let voff: Vec<usize> = (0..kv).map(|k| kz * nv + k * (3 * nv).div_ceil(2)).collect();
                let mut vden = |k: usize, h: &mut Vec<C32>| {
                    if kap <= 0.0 {
                        return;
                    }
                    let it = (k + k_off) as u64;
                    let nonfinite = |h: &[C32]| h.iter().filter(|c| !c.re.is_finite() || !c.im.is_finite()).count();
                    let n0 = nonfinite(h);
                    llr_z.set_iteration(it);
                    llr_z.apply(&mut h[..kz * nv], kz);
                    let n1 = nonfinite(&h[..kz * nv]);
                    let zbar: Vec<f32> = (0..nv).map(|q| (0..kz).map(|kk| { let c = h[kk * nv + q]; (c.re * c.re + c.im * c.im).sqrt() }).sum::<f32>() / kz as f32).collect();
                    let zmax = zbar.iter().cloned().fold(0f32, f32::max);
                    let mut pimg = vec![C32::new(0.0, 0.0); 3 * kv * nv];
                    for kk in 0..kv {
                        let f: Vec<f32> = h[voff[kk]..voff[kk] + (3 * nv).div_ceil(2)].iter().flat_map(|c| [c.re, c.im]).collect();
                        for c in 0..3 {
                            for q in 0..nv {
                                pimg[(3 * kk + c) * nv + q] = C32::new(zbar[q] * f[c * nv + q], 0.0);
                            }
                        }
                    }
                    let n2 = nonfinite(&pimg);
                    llr_p.set_iteration(it);
                    llr_p.apply(&mut pimg, 3 * kv);
                    let n3 = nonfinite(&pimg);
                    if n0 + n1 + n2 + n3 > 0 {
                        eprintln!("    vden it {k}: non-finite before {n0}, after LLR(z) {n1}, p {n2}, after LLR(p) {n3}");
                    }
                    for kk in 0..kv {
                        let mut f: Vec<f32> = h[voff[kk]..voff[kk] + (3 * nv).div_ceil(2)].iter().flat_map(|c| [c.re, c.im]).collect();
                        for c in 0..3 {
                            for q in 0..nv {
                                if zbar[q] > 1e-3 * zmax {
                                    f[c * nv + q] = pimg[(3 * kk + c) * nv + q].re / zbar[q];
                                }
                            }
                        }
                        for (i, pair) in f.chunks(2).enumerate() {
                            h[voff[kk] + i] = C32::new(pair[0], *pair.get(1).unwrap_or(&0.0));
                        }
                    }
                };
                if let Outcome::Stopped(k) = hasty_recon::p2np::run_objective_with(&mut pr, &st, &mut x, precond, 1.0 / lam, n_it, &mut vden, &mut report, Some((&ctl, &stag)), &sopts).unwrap() {
                    println!("stopped {stag} at iteration {k} / {n_it}: state saved in {}, rerun the same command to resume", out.display());
                    std::process::exit(0);
                }
                ctl.save(&format!("{stag}_final"), &hasty_recon::checkpoint::stack_value(&st, &x).unwrap()).unwrap();
                std::fs::write(&done, b"").unwrap();
                k_off += n_it;
            }
            // outputs: velocities and model images per frame, parameters
            let vh: Vec<Vec<f32>> = (0..kv).map(|kk| st.clone_dtoh(&x.v[spec.chunk_v(kk)]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect()).collect();
            let mut vout = vec![0f32; nv * 3 * nf];
            let mut xout = vec![C32::new(0.0, 0.0); nv * ne * nf];
            let mut xi = st.alloc_zeros::<C32>(nv).unwrap();
            for fr in 0..nf {
                for q in 0..nv {
                    let v = vel_of(&vh, fr, q);
                    for a in 0..3 {
                        vout[q + nv * (a + 3 * fr)] = v[a] as f32;
                    }
                }
                for e in 0..ne {
                    pr.model.eval(&st, &x, e, fr, &mut xi).unwrap();
                    xout[nv * (e + ne * fr)..nv * (e + ne * fr + 1)].copy_from_slice(&st.clone_dtoh(&xi).unwrap());
                }
            }
            save(&out.join(format!("vel_{tag}.npy")), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3, nf], f4(vout));
            save(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf], c8(&xout));
            if spec.np() > 0 {
                let af: Vec<f32> = st.clone_dtoh(&x.v[spec.chunk_a()]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect();
                let np = spec.np();
                let rec: Vec<Vec<f64>> = (1..ne).map(|e| (0..np).map(|p| (af[(e - 1) * np + p] as f64 * 1000.0).round() / 1000.0).collect()).collect();
                println!("  recovered residual phase (rad, Legendre order of vmodel::poly_exps): {rec:?}");
                if !injected.is_empty() && vpoly == Some(1) {
                    let err: f64 = (1..ne).map(|e| (0..4).map(|p| (af[(e - 1) * np + p] as f64 - injected[e - 1][p]).powi(2)).sum::<f64>()).sum::<f64>().sqrt() / ((ne - 1) as f64 * 4.0).sqrt();
                    println!("  residual-phase RMS error vs injected: {err:.4} rad");
                }
            }
            let params = hasty_recon::checkpoint::stack_value(&st, &x).unwrap();
            hasty_value::write(&params, &hasty_value::Options::default(), &mut std::io::BufWriter::new(std::fs::File::create(out.join(format!("vparams_{tag}.hv"))).unwrap())).unwrap();
            for si in 0..stages.len() {
                ctl.clear(&format!("{tag}_s{si}_final"));
                let _ = std::fs::remove_file(out.join(format!("vdone_{tag}_s{si}")));
            }
        }
        let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/figs/run_figs.py");
        let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
        match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
            Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
            r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
        }
        return;
    }
    let mut p = P2np::new(sop, ys.iter().collect(), circ, &st).unwrap();
    let lmax = p.step_norm(&st, false, 12).unwrap();
    let lmax_c = if need_circ { p.step_norm(&st, true, 12).unwrap() } else { 1.0 };
    println!("lambda_max(N) = {lmax:.3e}, lambda_max(M N) = {lmax_c:.3e}, setup {:.0}s", t0.elapsed().as_secs_f64());
    let ttag = temporal.map_or(String::new(), |b| format!("_t{b}"));
    let runs: Vec<String> = preconds.iter().flat_map(|p| kappas.iter().map(|k| (p.clone(), *k)).collect::<Vec<_>>()).map(|(p, k)| format!("{model}{ttag}_{p}{}_k{k}", if learned.is_some() { "_learned" } else { "" })).collect();
    let info = serde_json::json!({
        "cmdline": cmdline, "sim_dir": dir, "model": model, "temporal": temporal, "phys": phys, "gnl": gnl, "nseg": nseg,
        "order": order, "eps": eps, "iters": iters, "warm_iters": warm_iters, "block": block, "vcoils": nvc, "dims": sd.dims, "vox": sd.vox, "native_dims": sd.tdims,
        "n_frames": nf, "n_enc": ne, "venc": sd.venc, "venc_range": dec.range(), "encoding": sd.meta["encoding"], "lambda_max": lmax, "lambda_max_mn": lmax_c, "setup_s": t0.elapsed().as_secs_f64(),
        "noise_sigma_env": std::env::var("HASTY_NOISE_SIGMA").ok(), "runs": runs,
        "prior": if learned.is_some() { "learned" } else { "llr" }, "learned_dir": learned_dir, "tile": tile, "keep_frames": keep_frames, "learned_prec": learned_prec,
    });
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
    let rt = transpose(&sd.rot);
    for (pc, kappa) in preconds.iter().flat_map(|p| kappas.iter().map(move |&k| (p, k))) {
        let (precond, alpha) = match pc.as_str() {
            "none" => (Precond::None, 1.0 / lmax),
            "circ" => (Precond::Circulant, 1.0 / lmax_c),
            "zmshr1m" => (Precond::Zmshr1M, 1.0 / lmax_c),
            _ => (Precond::Zmshr1, 1.0 / lmax),
        };
        let mut x = Stack::zeros_lens(&st, &lens).unwrap();
        if !warm.is_empty() {
            for fr in 0..nf {
                for e in 0..ne {
                    let (t, o) = at(fr, e);
                    st.memcpy_htod(&warm[e], &mut x.v[t].slice_mut(o..o + nv)).unwrap();
                }
            }
        }
        let mut llr = Llr::new(sd.dims, block, kappa, support.clone());
        let nimg = nf * ne;
        let tr = Instant::now();
        let (mut td, mut tn) = (0.0f64, 0usize);
        let mut denoise = |k: usize, h: &mut Vec<C32>| {
            if kappa <= 0.0 {
                return;
            }
            let Some(prior) = &learned else {
                llr.set_iteration(k as u64);
                llr.apply(h, nimg);
                return;
            };
            // stack -> images[frame * 7 + e] (the temporal model stores encode-major sets of all frames)
            let mut img: Vec<C32> = if temporal.is_some() { (0..nf * ne).flat_map(|i| { let (f, e) = (i / ne, i % ne); h[(e * nf + f) * nv..(e * nf + f + 1) * nv].iter().copied() }).collect() } else { h.clone() };
            let t = Instant::now();
            prior.denoise(&mut img, sd.dims, nf, kv7, sd.vox[0] as f32, tile, keep_frames).unwrap();
            td += t.elapsed().as_secs_f64();
            tn += 1;
            let kf = kappa as f32;
            for i in 0..nf * ne {
                let (f, e) = (i / ne, i % ne);
                let o = if temporal.is_some() { (e * nf + f) * nv } else { i * nv };
                for q in 0..nv {
                    let d = img[i * nv + q];
                    let x = &mut h[o + q];
                    *x = C32::new(x.re + (d.re - x.re) * kf, x.im + (d.im - x.im) * kf);
                }
            }
            if tn == 1 || tn % 10 == 0 {
                println!("    learned prior: {:.2}s per call ({tn} calls)", td / tn as f64);
            }
        };
        let ptag = if learned.is_some() { "_learned" } else { "" };
        let tag = format!("{model}{}_{pc}{ptag}_k{kappa}", temporal.map_or(String::new(), |b| format!("_t{b}")));
        let resuming = out.join(format!("state_{tag}.hv")).exists();
        if !resuming && out.join(format!("recon_{tag}.npy")).exists() {
            println!("  {tag}: finished in an earlier run (recon_{tag}.npy), skipped (--fresh to redo)");
            continue;
        }
        let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(out.join(format!("metrics_{tag}.csv"))).unwrap();
        if !resuming {
            use std::io::Write;
            csv.set_len(0).unwrap();
            writeln!(csv, "iter,f,rmse,rel,seconds,{}", (0..nf).map(|f| format!("frame{f}")).collect::<Vec<_>>().join(",")).unwrap();
        }
        let mut report = |k: usize, xs: &Stack, f: f64| {
            let k = k.wrapping_add(1);
            if k % every != 0 && k != iters && k != 0 {
                return;
            }
            let imgs: Vec<Vec<C32>> = xs.v.iter().map(|v| st.clone_dtoh(v).unwrap()).collect();
            let img = |fr: usize, e: usize, q: usize| -> Complex64 {
                let (t, o) = at(fr, e);
                let c = imgs[t][o + q];
                Complex64::new(c.re as f64, c.im as f64)
            };
            // velocity at voxel q, frame fr (logical components; GNL: apparent → true via J⁻¹)
            let vel = |fr: usize, q: usize| -> [f64; 3] {
                let z: Vec<Complex64> = (0..ne).map(|e| img(fr, e, q)).collect();
                let mut v = dec.velocity(&z);
                if gnl {
                    v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v)));
                }
                v
            };
            let (mut e2, mut t2, mut n) = (0.0f64, 0.0f64, 0usize);
            let mut per_frame = Vec::with_capacity(nf);
            for fr in 0..nf {
                let (mut fe2, mut fn_) = (0.0, 0usize);
                for &q in &lumen {
                    let v = vel(fr, q);
                    for a in 0..3 {
                        let t = sd.sample_log(&sd.truth_vel, a + 3 * fr, &grid.x_log[q]);
                        e2 += (v[a] - t).powi(2);
                        t2 += t * t;
                        fe2 += (v[a] - t).powi(2);
                    }
                    n += 1;
                    fn_ += 1;
                }
                per_frame.push((fe2 / fn_.max(1) as f64).sqrt());
            }
            {
                use std::io::Write;
                writeln!(csv, "{k},{f:.6e},{:.5},{:.5},{:.1},{}", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), tr.elapsed().as_secs_f64(), per_frame.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")).unwrap();
            }
            if k == iters {
                let mut vout = vec![0f32; nv * 3 * nf];
                let mut xout = vec![C32::new(0.0, 0.0); nv * ne * nf];
                for fr in 0..nf {
                    for q in 0..nv {
                        let v = vel(fr, q);
                        for a in 0..3 {
                            vout[q + nv * (a + 3 * fr)] = v[a] as f32;
                        }
                        for e in 0..ne {
                            let z = img(fr, e, q);
                            xout[q + nv * (e + ne * fr)] = C32::new(z.re as f32, z.im as f32);
                        }
                    }
                }
                save(&out.join(format!("vel_{tag}.npy")), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3, nf], f4(vout));
                save(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf], c8(&xout));
            }
            println!(
                "  {:<7} κ {kappa:<4} it {:>3}: f = {:.4e}, velocity RMSE {:.3} m/s (rel {:.3}), frames {}, {:.0}s",
                pc,
                k,
                f,
                (e2 / n as f64).sqrt(),
                (e2 / t2).sqrt(),
                per_frame.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "),
                tr.elapsed().as_secs_f64()
            );
        };
        if let Outcome::Stopped(k) = p.run_from(&st, &mut x, precond, alpha, iters, &mut denoise, &mut report, Some((&ctl, &tag))).unwrap() {
            println!("stopped {tag} at iteration {k} / {iters}: state saved in {}, rerun the same command to resume", out.display());
            std::process::exit(0);
        }
    }
    let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/figs/run_figs.py");
    let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
        Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
        r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
    }
}
