//! Truth simulation of a spiral 4D-flow scan of a prepared subject (direct summation on the GPU).
//!   sim4d <subject_dir> <out_dir> [--scan-time s] [--noise rel] [--bin-ms ms] [--venc m/s] [--venc-hi m/s]
//! Writes data_clean.npy (complex64 [readout][adc][coil], noise-free; `--write-noisy` also data.npy), readouts.npy,
//! meta.json (incl. noise_sigma: the loader adds Ψ-correlated noise at that level), truth_*.npy, diagnostics.
//! Interruptible: Ctrl-C / SIGTERM / `touch <out_dir>/STOP` stops after the current chunk of flow-time bins; each
//! finished chunk is kept as `state_sim_bins_<b0>_<b1>.hv` (complex64), so the same command line resumes. The chunk
//! grid is a function of the arguments only; the states are removed when the run completes.

use std::path::PathBuf;
use std::time::Instant;

use hasty_cuda::{Device, C32, C64};
use hasty_mrsim::direct::{pack_sources, DirectSim};
use hasty_mrsim::sequence::{Acquisition, FlowSpiralParams, SAMPLE_FLOATS};
use hasty_phantom::flow::{AorticWaveform, CenterlineFlow, FlowField};
use hasty_phantom::prescription::aorta_slab;
use hasty_phantom::scene::{cholesky, Scene, SceneParams};
use hasty_phantom::subject::Subject;
use hasty_phantom::tissue::TissueModel;
use hasty_phantom::volume::{write_npy, write_npy_f32, Volume};
use num_complex::Complex64;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        let mut s = self.0;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.0 = s;
        ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
    }
}

fn c64_bytes(v: &[Complex64]) -> Vec<u8> {
    v.iter().flat_map(|c| [(c.re as f32).to_le_bytes(), (c.im as f32).to_le_bytes()].concat()).collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let scan_time: f64 = take("--scan-time").map(|s| s.parse().unwrap()).unwrap_or(60.0);
    let noise_rel: f64 = take("--noise").map(|s| s.parse().unwrap()).unwrap_or(0.02);
    let bin_ms: f64 = take("--bin-ms").map(|s| s.parse().unwrap()).unwrap_or(10.0);
    let venc: f64 = take("--venc").map(|s| s.parse().unwrap()).unwrap_or(2.0);
    // seven-point dual-VENC (the learned denoiser's layout): --venc is the low (sensitive) VENC
    let venc_hi: Option<f64> = take("--venc-hi").map(|s| s.parse().unwrap());
    // continue the body beyond the CT crop (head–foot) by this much on both ends: real bodies extend past the FOV
    let pad_mm: f64 = take("--pad-mm").map(|s| s.parse().unwrap()).unwrap_or(120.0);
    // also write the noisy data (data.npy); by default only data_clean.npy, the loader adds the noise (half the disk)
    let write_noisy = args.iter().position(|a| a == "--write-noisy").map(|i| { args.remove(i); true }).unwrap_or(false);
    let static_blood = args.iter().position(|a| a == "--static-blood").map(|i| { args.remove(i); true }).unwrap_or(false);
    let no_tissue = args.iter().position(|a| a == "--no-tissue").map(|i| { args.remove(i); true }).unwrap_or(false);
    // ideal scanner: linear gradients, no susceptibility off-resonance, no concomitant fields (diagnostics)
    let ideal = args.iter().position(|a| a == "--ideal").map(|i| { args.remove(i); true }).unwrap_or(false);
    // velocity encoded by the bipolars only (no motion during the readout): consistent with a frame-static model
    let no_ro_motion = args.iter().position(|a| a == "--no-readout-motion").map(|i| { args.remove(i); true }).unwrap_or(false);
    let subj_dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let ctl = hasty_recon::checkpoint::RunCtl::new(&out, 1);
    let t0 = Instant::now();
    let mut subj = Subject::load(&subj_dir).unwrap();
    subj.pad_z((pad_mm * 1e-3 / subj.hu.affine[2][2].abs()).round() as usize);
    let mut fp = FlowSpiralParams::aorta_3t_default();
    fp.scan_time = scan_time;
    fp.venc = venc;
    if let Some(venc_hi) = venc_hi {
        fp.encoding = hasty_mrsim::sequence::Encoding::DualVenc { venc_hi };
    }
    let heart = subj.label_centroid(&[51]).unwrap();
    let wave = AorticWaveform::healthy(fp.mean_rr);
    let flow = CenterlineFlow::new(subj.centerline.clone(), subj.sdf.clone(), heart, wave, 30, 3.3e-6, 5.0);
    let (main_pts, main_r) = flow.main_path(0.5);
    let presc = aorta_slab(&main_pts, main_r, 0.002, [0.36, 0.36], 0.008);
    fp.rot_presc = presc.rot;
    fp.fov = presc.fov;
    fp.res = presc.res;
    let tr = Acquisition::new(FlowSpiralParams { scan_time: 0.1, ..fp.clone() }).tr;
    println!("subject {}: {} centerline branches (fraction, radius mm): {:?}", subj.name, flow.n_branches(), flow.branch_info().iter().map(|(f, r)| (format!("{f:.2}"), format!("{:.1}", r * 1e3))).collect::<Vec<_>>());
    let mut sp = SceneParams::default_3t(tr);
    sp.static_blood = static_blood;
    if ideal {
        sp.gnl = hasty_mrsim::gnl::GradientCoils::ideal();
        sp.no_b0 = true;
    }
    let scene = Scene::new(&subj, TissueModel::default_3t(), sp, presc.clone(), &flow);
    fp.fov_center = scene.fov_center_scanner();
    let acq = Acquisition::new(fp.clone());
    let nc = scene.nc();
    println!(
        "scene ready in {:.1}s: iso {:?} mm, FOV centre (scanner) {:?} mm, slab {:.0} mm, TR {:.2} ms, TE {:.2} ms, {} readouts x {} ADC, {} coils",
        t0.elapsed().as_secs_f64(),
        scene.iso.map(|v| (v * 1e3).round()),
        scene.fov_center_scanner().map(|v| (v * 1e3).round()),
        presc.fov[2] * 1e3,
        acq.tr * 1e3,
        acq.t_ro * 1e3,
        acq.readouts.len(),
        acq.n_adc(),
        nc
    );
    // field statistics inside the slab tissue
    {
        let mut v: Vec<f32> = Vec::new();
        for q in 0..scene.field_ppm.len() {
            if scene.cls.data[q] >= 3 {
                v.push(scene.field_ppm.data[q]);
            }
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let hz = |p: f32| p as f64 * 1e-6 * hasty_mrsim::GAMMA_BAR * 3.0;
        println!("off-resonance in tissue (Hz): 1% {:.0}, 50% {:.0}, 99% {:.0}", hz(v[v.len() / 100]), hz(v[v.len() / 2]), hz(v[99 * v.len() / 100]));
    }
    write_npy_f32(&out.join("field_ppm_1mm.npy"), &scene.field_ppm).unwrap();
    write_npy_f32(&out.join("class_1mm.npy"), &scene.cls.map(|v| v as f32)).unwrap();

    // ------------------------------------------------ sources
    let t1 = Instant::now();
    let (tsrc, tamp) = if no_tissue { (vec![hasty_mrsim::direct::Source::default()], vec![Complex64::new(0.0, 0.0); nc]) } else { scene.tissue_sources() };
    let pts = scene.blood_points();
    println!("sources: {} tissue (1 mm), {} blood (0.5 mm) in {:.1}s", tsrc.len(), pts.len(), t1.elapsed().as_secs_f64());
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let sim = DirectSim::new(&dev, nc).unwrap();
    let to32 = |a: &[Complex64]| a.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>();
    let d_tsrc = stream.clone_htod(&pack_sources(&tsrc)).unwrap();
    let d_tamp = stream.clone_htod(&to32(&tamp)).unwrap();
    drop(tamp);

    // ------------------------------------------------ simulate, grouped by flow-time bins
    let period = flow.period();
    let bin = bin_ms * 1e-3;
    let nb = (period / bin).ceil() as usize;
    let tflow: Vec<f64> = acq.readouts.iter().map(|r| r.card.phase * period).collect();
    let mut by_bin: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (i, &t) in tflow.iter().enumerate() {
        by_bin[((t / bin) as usize).min(nb - 1)].push(i);
    }
    let na = acq.n_adc();
    let mut data = vec![Complex64::new(0.0, 0.0); acq.readouts.len() * na * nc];
    let t2 = Instant::now();
    let mut t_blood = 0.0;
    // chunks of consecutive bins (≤ max_chunk samples): tissue summed once per chunk at full occupancy, blood per bin
    let max_chunk = (1usize << 31) / (nc * 16); // ≤ 2 GiB of f64 output
    let mut b0 = 0;
    let mut chunk_states = vec![];
    while b0 < nb {
        let mut b1 = b0;
        let mut ns = 0;
        while b1 < nb && (ns == 0 || ns + by_bin[b1].len() * na <= max_chunk) {
            ns += by_bin[b1].len() * na;
            b1 += 1;
        }
        if ns == 0 {
            b0 = b1;
            continue;
        }
        let mut smp = Vec::with_capacity(ns * SAMPLE_FLOATS);
        let mut order = Vec::new();
        let mut ranges = Vec::new();
        for b in b0..b1 {
            let tb = (b as f64 + 0.5) * bin;
            let start = order.len() * na;
            for &r in &by_bin[b] {
                let mut s = acq.sample_params(r..r + 1);
                let delta = (tflow[r] - tb) as f32;
                for n in 0..na {
                    s[n * SAMPLE_FLOATS + 14] = delta;
                    if no_ro_motion {
                        let r0 = &acq.readouts[r];
                        let pm = acq.pre[r0.enc][r0.part].rotate(&acq.params.rot_presc);
                        for a in 0..3 {
                            s[n * SAMPLE_FLOATS + 3 + a] = (hasty_mrsim::GAMMA * pm.m1[a]) as f32;
                            s[n * SAMPLE_FLOATS + 6 + a] = (0.5 * hasty_mrsim::GAMMA * pm.m2[a]) as f32;
                        }
                    }
                    if ideal {
                        // no concomitant fields
                        for q in 9..13 {
                            s[n * SAMPLE_FLOATS + q] = 0.0;
                        }
                    }
                }
                smp.extend(s);
                order.push(r);
            }
            ranges.push((b, start, order.len() * na));
        }
        let state = format!("sim_bins_{b0}_{b1}");
        chunk_states.push(state.clone());
        if let Some(v) = ctl.load(&state).unwrap() {
            let o: Vec<num_complex::Complex32> = v.as_tensor().expect("chunk state").to_vec().unwrap();
            assert_eq!(o.len(), ns * nc, "{state}: stale chunk state (other arguments?), delete the state_sim_* files");
            for (k, &r) in order.iter().enumerate() {
                for n in 0..na {
                    for c in 0..nc {
                        let v = o[(k * na + n) * nc + c];
                        data[(r * na + n) * nc + c] = Complex64::new(v.re as f64, v.im as f64);
                    }
                }
            }
            println!("  bins {b0}..{b1} of {nb}: {} readouts, from the saved chunk", order.len());
            b0 = b1;
            continue;
        }
        let d_smp = stream.clone_htod(&smp).unwrap();
        let mut d_out = stream.alloc_zeros::<C64>(ns * nc).unwrap();
        sim.accumulate(&stream, &d_smp, ns, &d_tsrc, &d_tamp, tsrc.len(), &mut d_out).unwrap();
        for &(b, s0, s1) in &ranges {
            if s1 == s0 {
                continue;
            }
            let tq = Instant::now();
            let (bsrc, bamp) = scene.blood_sources(&pts, (b as f64 + 0.5) * bin);
            let d_bsrc = stream.clone_htod(&pack_sources(&bsrc)).unwrap();
            let d_bamp = stream.clone_htod(&to32(&bamp)).unwrap();
            t_blood += tq.elapsed().as_secs_f64();
            let smp_b = d_smp.slice(s0 * SAMPLE_FLOATS..s1 * SAMPLE_FLOATS);
            let mut out_b = d_out.slice_mut(s0 * nc..s1 * nc);
            sim.accumulate_view(&stream, &smp_b, s1 - s0, &d_bsrc, &d_bamp, bsrc.len(), &mut out_b).unwrap();
        }
        let o = stream.clone_dtoh(&d_out).unwrap();
        for (k, &r) in order.iter().enumerate() {
            for n in 0..na {
                for c in 0..nc {
                    let v = o[(k * na + n) * nc + c];
                    data[(r * na + n) * nc + c] = Complex64::new(v.re, v.im);
                }
            }
        }
        // complex64 like the final data_clean.npy
        let o32: Vec<num_complex::Complex32> = o.iter().map(|v| num_complex::Complex32::new(v.re as f32, v.im as f32)).collect();
        ctl.save(&state, &hasty_value::Value::Tensor(hasty_value::Tensor::from_slice(&[ns, nc], &o32).unwrap())).unwrap();
        println!("  bins {b0}..{b1} of {nb}: {} readouts, elapsed {:.1}s", order.len(), t2.elapsed().as_secs_f64());
        b0 = b1;
        if b0 < nb && ctl.stop_requested() {
            println!("stopped after bins ..{b0} of {nb}: chunks saved in {}, rerun the same command to resume", out.display());
            std::process::exit(0);
        }
    }
    let gpu_terms = (acq.n_samples() as f64) * ((tsrc.len() + pts.len()) as f64) * nc as f64;
    println!(
        "simulation {:.1}s ({:.1}s building blood sources), {:.2e} source-sample-coil terms ({:.2e}/s)",
        t2.elapsed().as_secs_f64(),
        t_blood,
        gpu_terms,
        gpu_terms / (t2.elapsed().as_secs_f64() - t_blood)
    );

    // ------------------------------------------------ noise (coil covariance Ψ)
    // noise-free data as well: the reconstruction side can add Ψ-correlated noise at any level (SNR calibration)
    write_npy(&out.join("data_clean.npy"), "<c8", &[acq.readouts.len(), na, nc], false, &c64_bytes(&data)).unwrap();
    let rms = (data.iter().map(|c| c.norm_sqr()).sum::<f64>() / data.len() as f64).sqrt();
    let sigma = noise_rel * rms;
    if write_noisy {
        let l = cholesky(&scene.psi);
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut white = vec![Complex64::new(0.0, 0.0); nc];
        for s in data.chunks_mut(nc) {
            for w in white.iter_mut() {
                *w = Complex64::new(rng.gauss(), rng.gauss()) * (sigma / 2f64.sqrt());
            }
            for c in 0..nc {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..=c {
                    acc += white[k] * l[c][k];
                }
                s[c] += acc;
            }
        }
        write_npy(&out.join("data.npy"), "<c8", &[acq.readouts.len(), na, nc], false, &c64_bytes(&data)).unwrap();
    }
    let ro_tab: Vec<u8> = acq
        .readouts
        .iter()
        .flat_map(|r| {
            let mut v = vec![r.t_exc, r.enc as f64, r.part as f64, r.frame as f64, r.card.phase];
            for a in 0..3 {
                for b in 0..3 {
                    v.push(r.rot[a][b]);
                }
            }
            v.into_iter().flat_map(|x| x.to_le_bytes())
        })
        .collect();
    write_npy(&out.join("readouts.npy"), "<f8", &[acq.readouts.len(), 14], false, &ro_tab).unwrap();

    // ------------------------------------------------ truth on the reconstruction grid (logical frame)
    let res = presc.res[0];
    let dims = [0, 1, 2].map(|a| (presc.fov[a] / res).round() as usize);
    let nvox = dims[0] * dims[1] * dims[2];
    let vox = |x: &[f64; 3]| -> Option<usize> {
        let l = scene.to_logical(x);
        let mut q = [0usize; 3];
        for a in 0..3 {
            let f = (l[a] / res + (dims[a] / 2) as f64).round();
            if f < 0.0 || f >= dims[a] as f64 {
                return None;
            }
            q[a] = f as usize;
        }
        Some(q[0] + dims[0] * (q[1] + dims[1] * q[2]))
    };
    let mut frac = vec![0f32; nvox];
    for x in &pts {
        if let Some(q) = vox(x) {
            frac[q] += 1.0 / 64.0;
        }
    }
    let nf = fp.n_frames;
    let mut vel = vec![0f32; nvox * 3 * nf];
    let mut acc = vec![[0f64; 4]; nvox];
    for f in 0..nf {
        // truth = velocity averaged over the frame's cardiac window (what the binned data contain), 5 sub-times
        for sub in 0..5 {
        let t = (f as f64 + (sub as f64 + 0.5) / 5.0) / nf as f64 * period;
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let parts: Vec<Vec<(usize, [f64; 3])>> = std::thread::scope(|sc| {
            let hs: Vec<_> = pts
                .chunks(pts.len().div_ceil(nthreads))
                .map(|ps| {
                    let (flow, scene, vox) = (&flow, &scene, &vox);
                    sc.spawn(move || {
                        ps.iter()
                            .filter_map(|x| {
                                let q = vox(x)?;
                                let (v, _) = flow.eval(x, t);
                                Some((q, hasty_mrsim::mat_vec(&hasty_mrsim::transpose(&scene.presc.rot), &v)))
                            })
                            .collect()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        if sub == 0 {
            acc.iter_mut().for_each(|a| *a = [0.0; 4]);
        }
        for (q, vl) in parts.into_iter().flatten() {
            for a in 0..3 {
                acc[q][a] += vl[a];
            }
            acc[q][3] += 1.0;
        }
        }
        for q in 0..nvox {
            if acc[q][3] > 0.0 {
                for a in 0..3 {
                    vel[q + nvox * (a + 3 * f)] = (acc[q][a] / acc[q][3]) as f32;
                }
            }
        }
    }
    {
        let mut sp: Vec<f32> = (0..nvox).filter(|&q| frac[q] > 0.5).map(|q| (0..3).map(|a| vel[q + nvox * (a + 3 * 3)].powi(2)).sum::<f32>().sqrt()).collect();
        sp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !sp.is_empty() {
            println!("truth |v| in lumen, frame 3 (systole): median {:.2}, 99% {:.2}, max {:.2} m/s", sp[sp.len() / 2], sp[99 * sp.len() / 100], sp[sp.len() - 1]);
        }
    }
    write_npy(&out.join("truth_vel_logical.npy"), "<f4", &[dims[0], dims[1], dims[2], 3, nf], true, &vel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    write_npy(&out.join("truth_lumen_fraction.npy"), "<f4", &dims, true, &frac.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    // true sensitivities and field on the reconstruction grid (reference only; the reconstruction estimates its own)
    let mut smaps = vec![Complex64::new(0.0, 0.0); nvox * nc];
    let mut field = vec![0f32; nvox];
    let mut sbuf = vec![Complex64::new(0.0, 0.0); nc];
    let rot = presc.rot;
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let lg = [(i as f64 - (dims[0] / 2) as f64) * res, (j as f64 - (dims[1] / 2) as f64) * res, (k as f64 - (dims[2] / 2) as f64) * res];
                let p = hasty_mrsim::mat_vec(&rot, &lg);
                let x = [p[0] + presc.center[0], p[1] + presc.center[1], p[2] + presc.center[2]];
                let q = i + dims[0] * (j + dims[1] * k);
                scene.sens.at(&x, &mut sbuf);
                for c in 0..nc {
                    smaps[q + nvox * c] = sbuf[c];
                }
                field[q] = scene.field_ppm.sample(&x, 0.0);
            }
        }
    }
    write_npy(&out.join("smaps_true.npy"), "<c8", &[dims[0], dims[1], dims[2], nc], true, &c64_bytes(&smaps)).unwrap();
    let fv = Volume { dims, affine: [[0.0; 4]; 4], data: field };
    write_npy_f32(&out.join("field_ppm_recon.npy"), &fv).unwrap();
    let meta = serde_json::json!({
        "subject": subj.name,
        "b0": fp.b0, "venc": fp.venc, "venc_hi": venc_hi, "encoding": format!("{:?}", fp.encoding), "coverage": format!("{:?}", fp.coverage),
        "fov": fp.fov, "res": fp.res, "recon_dims": dims, "n_int": fp.n_int, "fov_edge_ratio": fp.fov_edge_ratio,
        "dt": fp.dt, "dwell": fp.dwell, "t_rf": fp.t_rf, "t_spoil": fp.t_spoil, "scan_time": fp.scan_time,
        "gmax": fp.lim.gmax, "smax": fp.lim.smax,
        "rot_presc": fp.rot_presc, "fov_center_scanner": fp.fov_center, "iso_subject": scene.iso, "presc_center_subject": presc.center,
        "mean_rr": fp.mean_rr, "rr_rel_std": fp.rr_rel_std, "n_frames": fp.n_frames, "seed": fp.seed,
        "tr": acq.tr, "te": acq.t_ro, "n_readouts": acq.readouts.len(), "n_adc": na, "n_coils": nc,
        "noise_sigma": sigma, "noise_rel": noise_rel, "pad_mm": pad_mm, "psi": scene.psi,
        "gnl": {"r0": scene.params.gnl.r0, "transverse": scene.params.gnl.transverse, "longitudinal": scene.params.gnl.longitudinal},
        "flow": {"model": "centerline Womersley (M0)", "period": period, "branches": flow.branch_info()},
        "n_tissue_sources": tsrc.len(), "n_blood_sources": pts.len(),
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    for s in &chunk_states {
        ctl.clear(s);
    }
    println!("done in {:.1}s -> {}", t0.elapsed().as_secs_f64(), out.display());
}
