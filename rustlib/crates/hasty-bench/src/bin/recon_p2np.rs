//! Time-resolved P²nP-ISTA reconstruction of the synthetic spiral 4D-flow data (frames × encodes), no density
//! compensation, whitened + coil-compressed data, LLR prior over all frames × encodes.
//!   recon_p2np <sim_dir> [--model sense|seg] [--temporal off|linear|cubic] [--phys none|b0|full] [--gnl]
//!              [--nseg L] [--order o] [--precond none,circ,zmshr1,zmshr1m] [--iters n] [--kappa k,k,..] [--block b]
//!              [--vcoils n] [--every n] [--warm n] [--eps e] [--conc-dc] [--synthetic] [--out dir]
//! Every run writes `dir` (default <repo>/data/hasty/runs/<sim>_<model>…_<unix time>): run.json, coil maps, masks, warm
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
use hasty_recon::cg::pcgls;
use hasty_recon::data::{inv3, SampleSet, SimData};
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
    let (synthetic, conc_dc, gnl) = (flag("--synthetic"), flag("--conc-dc"), flag("--gnl"));
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
    let cmdline = std::env::args().collect::<Vec<_>>().join(" ");
    let dir = PathBuf::from(&args[0]);
    assert!(model == "seg" || temporal.is_none(), "--temporal needs --model seg");
    // run output directory: arrays, metrics and figures (tools/figs/run_figs.py)
    let out = out_arg.map(PathBuf::from).unwrap_or_else(|| {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let tag = format!("{name}_{model}{}_{phys}{}_{ts}", temporal.map_or(String::new(), |b| format!("_t{b}")), if gnl { "_gnl" } else { "" });
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/runs").join(tag)
    });
    std::fs::create_dir_all(&out).unwrap();
    println!("output directory {}", out.display());
    let t0 = Instant::now();
    let sd = SimData::load(&dir).unwrap();
    let (nv, nc, nf) = (sd.nv(), sd.nc, sd.n_frames);
    let w = sd.whitener();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    // sets: static t = frame·4 + encode (one frame each); temporal t = encode (all frames)
    let fe: Vec<(Option<usize>, usize)> = if temporal.is_some() { (0..4).map(|e| (None, e)).collect() } else { (0..nf).flat_map(|f| (0..4).map(move |e| (Some(f), e))).collect() };
    let enc_of = |t: usize| fe[t].1;
    let host_sets: Vec<SampleSet> = std::thread::scope(|sc| {
        let hs: Vec<_> = fe.iter().map(|&(f, e)| { let (sd, w) = (&sd, &w); sc.spawn(move || sd.sample_set(&sd.readouts_of(e, f), w)) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
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
        if temporal.is_some() { format!("4 encodes x {nf} frames, temporal B-spline {}", temporal.unwrap()) } else { format!("{nf} frames x 4 encodes") },
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
        (0..4)
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
        (0..4)
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
                let dy = st.clone_htod(&y).unwrap();
                let mut xe = st.alloc_zeros::<C32>(nv).unwrap();
                pcgls(&mut op_e, Some(&mut pre), &st, &dy, &mut xe, 0.0, warm_iters).unwrap();
                st.clone_dtoh(&xe).unwrap()
            })
            .collect()
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
        save(&out.join("warm.npy"), "<c8", &dims4(4), c8(&warm.concat()));
    }
    // fixed circulant preconditioners per frame (block-diagonal over the frames of a time-resolved set)
    let need_circ = preconds.iter().any(|p| p == "circ" || p == "zmshr1m");
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
            let r0 = Complex64::new(warm[0][q].re as f64, warm[0][q].im as f64);
            for a in 0..3 {
                let ra = Complex64::new(warm[a + 1][q].re as f64, warm[a + 1][q].im as f64);
                let v = -(ra * r0.conj()).arg() * sd.venc / std::f64::consts::PI;
                let t = (0..nf).map(|f| sd.sample_log(&sd.truth_vel, a + 3 * f, &grid.x_log[q])).sum::<f64>() / nf as f64;
                mr[a] += v;
                mt[a] += t;
                e2 += (v - t).powi(2);
            }
            n += 1;
        }
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
        let mut eng = SegNufft::new(&dev, sd.dims, [sd.res; 3], nvc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames: if temporal.is_some() { nf } else { 1 }, bspline: temporal.unwrap_or(1), max_grid_bytes: 6 << 30 }).unwrap();
        eng.set_maps(&st, &smaps).unwrap();
        // segments per encode from the encode-wide mean coefficients (all frames)
        let wbar: Vec<Vec<[f64; 4]>> = (0..4).map(|e| mean_coefficients(&(0..fe.len()).filter(|&t| enc_of(t) == e).map(|t| &host_sets[t]).collect::<Vec<_>>(), sd.na)).collect();
        let wbar: Vec<Vec<[f64; 4]>> = if with_conc { wbar } else { vec![vec![[0.0; 4]; sd.na]; 4] };
        let tau_adc: Vec<f64> = (0..sd.na).map(|n| host_sets[0].tau[n]).collect();
        let seg_fit: Vec<usize> = fit_vox.iter().copied().step_by(3).collect();
        let segs: Vec<_> = (0..4)
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
    let ns = sop.nsets();
    let lens: Vec<usize> = (0..ns).map(|t| sop.len_x(t)).collect();
    // image (frame f, encode e) inside the stack: (set, offset)
    let at = |f: usize, e: usize| -> (usize, usize) { if temporal.is_some() { (e, f * nv) } else { (f * 4 + e, 0) } };
    // model-consistent test data: y_t = E_t x_true,t + noise (solver + prior test without simulator physics)
    let mut ys = ys;
    if synthetic {
        assert!(!warm.is_empty(), "--synthetic needs --warm");
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
            for e in 0..4 {
                let (t, o) = at(fr, e);
                for q in 0..nv {
                    let r = warm[0][q];
                    let mut z = Complex64::new(r.re as f64, r.im as f64);
                    if e > 0 && sd.truth_frac[q] > 0.0 {
                        let v = sd.truth_vel[q + nv * (e - 1 + 3 * fr)] as f64;
                        z *= Complex64::from_polar(1.0, -std::f64::consts::PI * v / sd.venc);
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
    let mut p = P2np::new(sop, ys.iter().collect(), circ, &st).unwrap();
    let lmax = p.step_norm(&st, false, 12).unwrap();
    let lmax_c = if need_circ { p.step_norm(&st, true, 12).unwrap() } else { 1.0 };
    println!("lambda_max(N) = {lmax:.3e}, lambda_max(M N) = {lmax_c:.3e}, setup {:.0}s", t0.elapsed().as_secs_f64());
    let ttag = temporal.map_or(String::new(), |b| format!("_t{b}"));
    let runs: Vec<String> = preconds.iter().flat_map(|p| kappas.iter().map(|k| (p.clone(), *k)).collect::<Vec<_>>()).map(|(p, k)| format!("{model}{ttag}_{p}_k{k}")).collect();
    let info = serde_json::json!({
        "cmdline": cmdline, "sim_dir": dir, "model": model, "temporal": temporal, "phys": phys, "gnl": gnl, "nseg": nseg,
        "order": order, "eps": eps, "iters": iters, "warm_iters": warm_iters, "block": block, "vcoils": nvc, "dims": sd.dims,
        "n_frames": nf, "venc": sd.venc, "lambda_max": lmax, "lambda_max_mn": lmax_c, "setup_s": t0.elapsed().as_secs_f64(),
        "noise_sigma_env": std::env::var("HASTY_NOISE_SIGMA").ok(), "runs": runs,
    });
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
    let venc = sd.venc;
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
                for e in 0..4 {
                    let (t, o) = at(fr, e);
                    st.memcpy_htod(&warm[e], &mut x.v[t].slice_mut(o..o + nv)).unwrap();
                }
            }
        }
        let mut llr = Llr::new(sd.dims, block, kappa, support.clone());
        let nimg = nf * 4;
        let tr = Instant::now();
        let mut denoise = |h: &mut Vec<C32>| {
            if kappa > 0.0 {
                llr.apply(h, nimg)
            }
        };
        let tag = format!("{model}{}_{pc}_k{kappa}", temporal.map_or(String::new(), |b| format!("_t{b}")));
        let mut csv = std::fs::File::create(out.join(format!("metrics_{tag}.csv"))).unwrap();
        {
            use std::io::Write;
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
                let r0 = img(fr, 0, q);
                let mut v = [0, 1, 2].map(|a| -(img(fr, a + 1, q) * r0.conj()).arg() * venc / std::f64::consts::PI);
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
                let mut xout = vec![C32::new(0.0, 0.0); nv * 4 * nf];
                for fr in 0..nf {
                    for q in 0..nv {
                        let v = vel(fr, q);
                        for a in 0..3 {
                            vout[q + nv * (a + 3 * fr)] = v[a] as f32;
                        }
                        for e in 0..4 {
                            let z = img(fr, e, q);
                            xout[q + nv * (e + 4 * fr)] = C32::new(z.re as f32, z.im as f32);
                        }
                    }
                }
                save(&out.join(format!("vel_{tag}.npy")), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3, nf], f4(vout));
                save(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], 4, nf], c8(&xout));
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
        p.run(&st, &mut x, precond, alpha, iters, &mut denoise, &mut report).unwrap();
    }
    let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/figs/run_figs.py");
    let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
        Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
        r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
    }
}
