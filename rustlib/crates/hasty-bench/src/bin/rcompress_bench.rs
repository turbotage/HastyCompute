//! Per-readout sample compression (`hasty_recon::rcompress`, research/sample_compression.md) on a real trajectory:
//! design time, achieved reduction, direct-sum gate, and on the GPU the SegNufft forward + adjoint per encode pass on
//! all samples against the kept samples plus the banded R step, with the compressed operator's accuracy (normal
//! operator, gradient and objective against the uncompressed one).
//!
//!   rcompress_bench <scan dir> --dims NX,NY,NZ --vox MM [--kscale S] [--roi box|ball|ball:R_MM] [--os 1.35] [--q 32]
//!       [--encode 0] [--stride N] [--coils 8] [--nseg 10] [--order 2] [--knots K] [--reps 3] [--gate N]
//!       [--grid-gb 6] [--no-gpu] [--json out.json]
//!
//! Grid: `--dims` at `--vox` (mm), centred on the scan's FOV centre (a box recon's grid; the operator's image origin
//! is voxel d/2). Samples: encode `--encode`, every `--stride`-th readout, k × `--kscale` (the study's "1 mm proxy":
//! the 2 mm readout with k × 2, concomitant coefficients × kscale²), cropped to the grid's Nyquist box. ROI: the grid
//! box (per-readout designs) or the ball enclosing it (one shared design; `ball:R_MM`: a smaller ball, then the test
//! images are supported inside it). Band: ±1.5 kHz, decay ≤ 300 1/s (no ratemap here).
//! Operator: static (`--knots 0|1`) or the knot-major cubic temporal engine with K periodic knots (`--knots K`), W 10,
//! σ 1.25, `--nseg` segments of a smooth synthetic ±300 Hz field, order `--order` (3: order-3 tables with zero
//! degree-3/4 terms), synthetic coil maps; per-sample concomitant residuals zero (timing does not depend on them; the
//! gate uses the real concomitant coefficients).
use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, grid_default_size, SegNufft, SegNufftOpts, NHI};
use hasty_recon::data::SimData;
use hasty_recon::kload::LoadOpts;
use hasty_recon::rcompress::{gate, Band, Compressor, GateStats, RcDev, RcOpts, Roi, SetComp};
use num_complex::Complex64;

const GIB: f64 = (1u64 << 30) as f64;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |args: &mut Vec<String>, n: &str| args.iter().position(|a| a == n).map(|i| { args.remove(i); true }).unwrap_or(false);
    let no_gpu = flag(&mut args, "--no-gpu");
    let mut take = |n: &str| -> Option<String> { args.iter().position(|a| a == n).map(|i| { args.remove(i); args.remove(i) }) };
    let dims: [usize; 3] = { let v: Vec<usize> = take("--dims").expect("--dims NX,NY,NZ").split(',').map(|s| s.parse().unwrap()).collect(); [v[0], v[1], v[2]] };
    let vox_mm: f64 = take("--vox").expect("--vox MM").parse().unwrap();
    let vox = [vox_mm * 1e-3; 3];
    let kscale: f64 = take("--kscale").map_or(1.0, |v| v.parse().unwrap());
    let roi_s = take("--roi").unwrap_or("box".into());
    let os: f64 = take("--os").map_or(1.35, |v| v.parse().unwrap());
    let q: usize = take("--q").map_or(32, |v| v.parse().unwrap());
    let enc: usize = take("--encode").map_or(0, |v| v.parse().unwrap());
    let stride: usize = take("--stride").map_or(1, |v| v.parse().unwrap());
    let nc: usize = take("--coils").map_or(8, |v| v.parse().unwrap());
    let nseg: usize = take("--nseg").map_or(10, |v| v.parse().unwrap());
    let order: u32 = take("--order").map_or(2, |v| v.parse().unwrap());
    let knots: usize = take("--knots").map_or(0, |v| v.parse().unwrap());
    // --knots2d KR,KC: the 2-D respiratory (KR linear, bellows quantiles) × cardiac (KC periodic cubic) engine
    let knots2d: Option<(usize, usize)> = take("--knots2d").map(|v| { let (a, b) = v.split_once(',').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) });
    // --upsample F: the readout sampled F× faster (dwell / F; cubic interpolation of k, τ, concomitant along the ADC
    // index): the protocol-change study
    let upsample: usize = take("--upsample").map_or(1, |v| v.parse().unwrap());
    let reps: usize = take("--reps").map_or(3, |v| v.parse().unwrap());
    let ngate: usize = take("--gate").map_or(4, |v| v.parse().unwrap());
    let grid_gb: f64 = take("--grid-gb").map_or(6.0, |v| v.parse().unwrap());
    // NUFFT kernel (accuracy rule: W 10, σ 1.25; other values only to measure the NUFFT's own error)
    let width: usize = take("--width").map_or(10, |v| v.parse().unwrap());
    let sigma: f64 = take("--sigma").map_or(1.25, |v| v.parse().unwrap());
    let json_out: Option<PathBuf> = take("--json").map(PathBuf::from);
    // --gpu-lock FILE: flock(2) on FILE (the `flock` tool's lock) taken only around the GPU part, after the CPU design
    let gpu_lock: Option<PathBuf> = take("--gpu-lock").map(PathBuf::from);
    let dir = PathBuf::from(&args[0]);
    let t0 = Instant::now();
    let sd = SimData::load_lazy(&dir).unwrap();
    let mut ros = sd.readouts_of(enc, None);
    ros = ros.into_iter().step_by(stride.max(1)).collect();
    let w = sd.whitener();
    let set = sd.load_set(&ros, &LoadOpts { w: &w, cc: None, keep: None, data: false }).unwrap();
    let na0 = sd.na;
    assert_eq!(set.tau.len(), ros.len() * na0, "fixed ADC count per readout expected");
    // --upsample: Catmull–Rom along the ADC index (τ exactly uniform)
    let set = if upsample > 1 {
        let f = upsample;
        let na1 = (na0 - 1) * f + 1;
        let mut o = hasty_recon::data::SampleSet { readouts: set.readouts.clone(), coords: [vec![], vec![], vec![]], y: vec![], tau: vec![], wq: vec![] };
        let cr = |p: [f64; 4], t: f64| -> f64 { 0.5 * (2.0 * p[1] + (-p[0] + p[2]) * t + (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3]) * t * t + (-p[0] + 3.0 * p[1] - 3.0 * p[2] + p[3]) * t * t * t) };
        for r in 0..ros.len() {
            let b = r * na0;
            let at = |n: i64| -> usize { b + n.clamp(0, na0 as i64 - 1) as usize };
            for i in 0..na1 {
                let (n, t) = ((i / f) as i64, (i % f) as f64 / f as f64);
                let idx = [at(n - 1), at(n), at(n + 1), at(n + 2)];
                for a in 0..3 {
                    o.coords[a].push(cr(idx.map(|j| set.coords[a][j] as f64), t) as f32);
                }
                o.tau.push(set.tau[at(n)] + t * (set.tau[at(n + 1)] - set.tau[at(n)]));
                o.wq.push([0, 1, 2, 3].map(|c| cr(idx.map(|j| set.wq[j][c]), t)));
            }
        }
        o
    } else {
        set
    };
    let na = (na0 - 1) * upsample + 1;
    // k (rad/m) × kscale, cropped to the grid's Nyquist box
    let kin = |j: usize| -> [f64; 3] { [0, 1, 2].map(|a| set.coords[a][j] as f64 / sd.vox[a] * kscale) };
    let mut keep: Vec<usize> = vec![];
    let mut lens: Vec<usize> = vec![];
    for r in 0..ros.len() {
        let before = keep.len();
        for n in 0..na {
            let j = r * na + n;
            let k = kin(j);
            if (0..3).all(|a| (k[a] * vox[a]).abs() < std::f64::consts::PI) {
                keep.push(j);
            }
        }
        lens.push(keep.len() - before);
    }
    let m = keep.len();
    let mut cs = hasty_recon::data::SampleSet { readouts: ros.clone(), coords: [vec![], vec![], vec![]], y: vec![], tau: vec![], wq: vec![] };
    for &j in &keep {
        let k = kin(j);
        for a in 0..3 {
            cs.coords[a].push((k[a] * vox[a]) as f32);
        }
        cs.tau.push(set.tau[j]);
        cs.wq.push(set.wq[j].map(|v| v * kscale * kscale));
    }
    drop(set);
    let roi = match roi_s.as_str() {
        "box" => Roi::grid_box(dims, vox),
        "ball" => Roi::grid_ball(dims, vox),
        s if s.starts_with("ball:") => Roi::Ball { r: s[5..].parse::<f64>().unwrap() * 1e-3 },
        "ellipsoid" => match Roi::grid_box(dims, vox) { Roi::Box { h } => Roi::Ellipsoid { h }, _ => unreachable!() },
        o => panic!("--roi box|ball|ball:R_MM, got {o}"),
    };
    let band = Band::default_wide();
    let opts = RcOpts { os, q, ..RcOpts::new(roi, band) };
    println!("{}: encode {enc}, {} readouts (stride {stride}), {m} samples in the grid band of {} per encode before the crop; grid {dims:?} at {vox_mm} mm, k × {kscale}, dwell / {upsample}; ROI {}, band {}, os {os}, q {q} ({:.1}s)", dir.display(), ros.len(), ros.len() * na, roi.describe(), band.describe(), t0.elapsed().as_secs_f64());
    // Landau bound n / (∫ B dt / 2π) per readout, mean
    let mut landau = (0.0, 0.0);
    {
        let offs: Vec<usize> = lens.iter().scan(0, |o, &l| { let r = *o; *o += l; Some(r) }).collect();
        for r in (0..lens.len()).step_by((lens.len() / 64).max(1)) {
            let (o, l) = (offs[r], lens[r]);
            if l < 2 {
                continue;
            }
            let k: Vec<[f64; 3]> = (o..o + l).map(|j| [0, 1, 2].map(|a| cs.coords[a][j] as f64 / vox[a])).collect();
            let tau = &cs.tau[o..o + l];
            let (b, _) = hasty_recon::rcompress::band_of(&k, tau, &opts);
            let ib: f64 = (1..l).map(|j| b[j] * (tau[j] - tau[j - 1]).min(5e-6)).sum::<f64>() / (2.0 * std::f64::consts::PI);
            landau.0 += l as f64;
            landau.1 += ib;
        }
    }
    let td = Instant::now();
    let mut comp = Compressor::new(opts.clone());
    let sc: SetComp = comp.plan_set(&cs, &lens, vox);
    let t_design = td.elapsed().as_secs_f64();
    let red = m as f64 / sc.m() as f64;
    println!("design: {} in {t_design:.1}s; reduction {red:.3}× ({m} → {} samples), Landau bound n / ∫B dt = {:.3}; R tables {:.3} GiB", comp.describe(), sc.m(), landau.0 / landau.1.max(1e-300), comp.table_bytes() as f64 / GIB);
    // gate on a few readouts (CPU direct sums, real concomitant coefficients, cardiac knot weights)
    let offs: Vec<usize> = lens.iter().scan(0, |o, &l| { let r = *o; *o += l; Some(r) }).collect();
    let card = |r: usize, j: usize| -> f64 { let c = &sd.ro.card[ros[r]]; ((c.phase + cs.tau[j] / c.rr) * 16.0 - 0.5).rem_euclid(16.0) };
    let mut gs = GateStats::default();
    let tg = Instant::now();
    for g in 0..ngate.min(lens.len()) {
        let r = g * lens.len() / ngate.max(1);
        let (o, l) = (offs[r], lens[r]);
        let k: Vec<[f64; 3]> = (o..o + l).map(|j| [0, 1, 2].map(|a| cs.coords[a][j] as f64 / vox[a])).collect();
        let kt: Vec<f64> = (o..o + l).map(|j| card(r, j)).collect();
        let d = hasty_recon::rcompress::Design::build(&k, &cs.tau[o..o + l], &opts);
        assert_eq!(d.sel.len(), comp.designs[sc.design[r] as usize].sel.len());
        let st = gate(&d, &k, &cs.tau[o..o + l], Some(&cs.wq[o..o + l]), Some(&kt), &opts, 240, 11 + g as u64);
        gs = gs.max(st);
    }
    println!("gate ({} readouts, 240 sources each, half on the band edge; {:.1}s): worst source {:.2e}, Frobenius {:.2e}, Gram (normal operator on the sources) {:.2e}", ngate.min(lens.len()), tg.elapsed().as_secs_f64(), gs.worst, gs.frob, gs.gram);
    let mut js = serde_json::json!({
        "scan": dir, "encode": enc, "readouts": ros.len(), "stride": stride, "dims": dims, "vox_mm": vox_mm, "kscale": kscale, "upsample": upsample,
        "roi": roi.describe(), "os": os, "q": q, "samples": m, "kept": sc.m(), "reduction": red, "landau": landau.0 / landau.1.max(1e-300),
        "design_s": t_design, "designs": comp.designs.len(), "table_gib": comp.table_bytes() as f64 / GIB,
        "gate": {"worst": gs.worst, "frob": gs.frob, "gram": gs.gram},
    });
    if no_gpu {
        if let Some(p) = &json_out {
            std::fs::write(p, serde_json::to_string_pretty(&js).unwrap()).unwrap();
        }
        return;
    }
    // ---------------------------------------------------------------- GPU
    let _lock = gpu_lock.map(|p| {
        let f = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
        let tl = Instant::now();
        f.lock().unwrap();
        println!("gpu lock {} taken after {:.0}s", p.display(), tl.elapsed().as_secs_f64());
        f
    });
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let ev = (dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap(), dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap());
    let time = |f: &mut dyn FnMut()| -> f32 {
        let mut t = Vec::new();
        for i in 0..=reps {
            ev.0.record(&st).unwrap();
            f();
            ev.1.record(&st).unwrap();
            ev.1.synchronize().unwrap();
            if i >= 1 {
                t.push(ev.0.elapsed_ms(&ev.1).unwrap());
            }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t[t.len() / 2]
    };
    let nv: usize = dims.iter().product();
    let nimg = knots2d.map_or(knots.max(1), |(a, b)| a * b);
    let nx = nv * nimg;
    let ge: usize = dims.iter().map(|&d| grid_default_size(d, sigma, width)).product();
    let frames = knots2d.map_or(knots.max(1), |k| k.1);
    let mk_so = |width: usize, sigma: f64| SegNufftOpts { sigma, width, nseg, order, frames, bspline: 3, max_grid_bytes: (grid_gb * GIB) as usize };
    let so = mk_so(width, sigma);
    let cb = (so.max_grid_bytes / (nseg * ge * 8)).clamp(1, nc.min(16));
    // sequential: the all-samples case (tables, y) is freed before the kept-samples case is built
    let plan = [("grid slot", cb * nseg * ge * 8), ("maps", nc * nv * 8), ("diagonals", nseg * nv * 8), ("tables (build peak)", m * 232), ("y", m * nc * 8), ("x, x0, adjoint out", 3 * nx * 8), ("R tables", comp.table_bytes())];
    let need: usize = plan.iter().map(|p| p.1).sum();
    let free = dev.ctx.mem_get_info().unwrap().0;
    println!("memory plan: {:.2} GiB ({}) of {:.2} GiB free", need as f64 / GIB, plan.iter().map(|(n, b)| format!("{n} {:.2}", *b as f64 / GIB)).collect::<Vec<_>>().join(", "), free as f64 / GIB);
    if (need as f64 * 1.05) as usize + (512 << 20) > free && std::env::var("HASTY_BENCH_FORCE").map_or(true, |v| v == "0") {
        println!("does not fit: skipped (HASTY_BENCH_FORCE=1 to try)");
        js["gpu"] = serde_json::json!("skipped: memory");
        if let Some(p) = &json_out {
            std::fs::write(p, serde_json::to_string_pretty(&js).unwrap()).unwrap();
        }
        return;
    }
    let new_eng = |so: SegNufftOpts| -> SegNufft {
        match knots2d {
            Some((kr, _)) => SegNufft::new_temporal2d(&dev, dims, vox, nc, so, kr, 1).unwrap(),
            None if knots > 1 => SegNufft::new_temporal(&dev, dims, vox, nc, so, true).unwrap(),
            None => SegNufft::new(&dev, dims, vox, nc, so).unwrap(),
        }
    };
    let mut s = 12345u64;
    let mut u = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
    // smooth synthetic coil maps (Gaussian lobes on a ring) and field
    let pos = |q: usize| -> [f64; 3] { let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])); [(i as f64 - (dims[0] / 2) as f64) * vox[0], (j as f64 - (dims[1] / 2) as f64) * vox[1], (k as f64 - (dims[2] / 2) as f64) * vox[2]] };
    let ext = [0, 1, 2].map(|a| dims[a] as f64 * vox[a] / 2.0);
    let smaps: Vec<C32> = (0..nc * nv).map(|i| {
        let (c, qv) = (i / nv, i % nv);
        let p = pos(qv);
        let ang = 2.0 * std::f64::consts::PI * c as f64 / nc as f64;
        let cc = [ext[0] * ang.cos(), ext[1] * ang.sin(), ext[2] * (0.5 - (c % 2) as f64)];
        let d2: f64 = (0..3).map(|a| ((p[a] - cc[a]) / (ext[a].max(1e-3) * 1.2)).powi(2)).sum();
        let z = Complex64::from_polar((-d2).exp(), 0.3 * c as f64 + 2.0 * p[0] / ext[0].max(1e-3));
        C32::new(z.re as f32, z.im as f32)
    }).collect();
    let om = |q: usize| -> f64 { let p = pos(q); 2.0 * std::f64::consts::PI * 300.0 * ((p[0] / ext[0].max(1e-3) * 1.7).sin() * (p[1] / ext[1].max(1e-3) * 1.3).cos() * 0.7 + 0.3 * p[2] / ext[2].max(1e-3)) };
    let (tmin, tmax) = cs.tau.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| (a.min(t), b.max(t)));
    let fit: Vec<usize> = (0..nv).step_by((nv / 20000).max(1)).collect();
    let psi = |q: usize, t: f64| om(q) * (t - tmin);
    let segtab = build_segments(nv, nseg, tmin, tmax, 128, &fit, &psi);
    // per-sample knot coordinates: cardiac (phase + τ / RR) in knot units; respiratory linear hats at the bellows
    // amplitude quantiles (per readout)
    let kt_all: Vec<f32> = (0..lens.len()).flat_map(|r| (offs[r]..offs[r] + lens[r]).map(move |j| (r, j))).map(|(r, j)| { let c = &sd.ro.card[ros[r]]; ((c.phase + cs.tau[j] / c.rr) * frames as f64 - 0.5).rem_euclid(frames as f64) as f32 }).collect();
    let (ri_all, rv_all): (Vec<u32>, Vec<f32>) = match knots2d {
        Some((kr, _)) => {
            let a = hasty_recon::data::read_npy(&dir.join("physio.npy")).unwrap();
            let (wd, v) = (a.shape[1], a.f64s());
            let amp: Vec<f64> = (0..sd.n_readouts()).map(|i| v[i * wd + 3]).collect();
            let mut srt = amp.clone();
            srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let qk: Vec<f64> = (0..kr).map(|k| srt[((k as f64 / (kr - 1) as f64) * (srt.len() - 1) as f64).round() as usize]).collect();
            let loc = |x: f64| -> (u32, f32) { let i = (0..kr - 1).rev().find(|&i| x >= qk[i]).unwrap_or(0); (i as u32, ((x - qk[i]) / (qk[i + 1] - qk[i]).max(1e-12)).clamp(0.0, 1.0) as f32) };
            (0..lens.len()).flat_map(|r| std::iter::repeat(loc(amp[ros[r]])).take(lens[r])).unzip()
        }
        None => (vec![], vec![]),
    };
    let mk = |eng: &mut SegNufft, idx: Option<&[u32]>| -> hasty_nufft::SegPoints {
        let g = |v: &[f32]| -> Vec<f32> { match idx { Some(i) => i.iter().map(|&j| v[j as usize]).collect(), None => v.to_vec() } };
        let c = [g(&cs.coords[0]), g(&cs.coords[1]), g(&cs.coords[2])];
        let tau: Vec<f32> = g(&cs.tau.iter().map(|&t| t as f32).collect::<Vec<_>>());
        let mm = tau.len();
        let (ph0, dc) = (vec![0f32; mm], vec![0f32; 6 * mm]);
        let tt = g(&kt_all);
        let (uu, hh) = (vec![0f32; 4 * mm], vec![0f64; 4 * NHI]);
        match (knots2d.is_some(), order >= 3) {
            (true, hi) => {
                let ri: Vec<u32> = match idx { Some(i) => i.iter().map(|&j| ri_all[j as usize]).collect(), None => ri_all.clone() };
                let rv = g(&rv_all);
                if hi { eng.points2d_hi(&st, [&c[0], &c[1], &c[2]], &tau, &ph0, &dc, &tt, &ri, &rv, &uu, &hh).unwrap() } else { eng.points2d(&st, [&c[0], &c[1], &c[2]], &tau, &ph0, &dc, &tt, &ri, &rv).unwrap() }
            }
            (false, true) => eng.points_hi(&st, [&c[0], &c[1], &c[2]], &tau, &ph0, &dc, &uu, &hh, (knots > 1).then_some(&tt[..])).unwrap(),
            (false, false) => eng.points(&st, [&c[0], &c[1], &c[2]], &tau, &ph0, &dc, (knots > 1).then_some(&tt[..])).unwrap(),
        }
    };
    // test image: random in the ROI (all voxels for the box and the enclosing ball); x0 = 0.6 x
    let inside = |q: usize| -> bool { let p = pos(q); match roi { Roi::Ball { r } => (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() <= r, Roi::Box { .. } => true, Roi::Ellipsoid { h } => (0..3).map(|a| (p[a] / h[a]).powi(2)).sum::<f64>() <= 1.0 } };
    let xh: Vec<C32> = (0..nx).map(|i| if inside(i % nv) { C32::new(u() as f32, u() as f32) } else { C32::new(0.0, 0.0) }).collect();
    let x = st.clone_htod(&xh).unwrap();
    let x0 = st.clone_htod(&xh.iter().map(|v| C32::new(v.re * 0.6, v.im * 0.6)).collect::<Vec<_>>()).unwrap();
    drop(xh);
    let kern = hasty_recon::gpu::Kernels::new(&dev).unwrap();
    let one = Complex64::new(1.0, 0.0);
    let host = |v: &cudarc::driver::CudaSlice<C32>| -> Vec<C32> { st.clone_dtoh(v).unwrap() };
    let rel = |a: &[C32], b: &[C32]| -> f64 { let (n, d) = a.iter().zip(b).fold((0.0f64, 0.0f64), |s, (p, q)| (s.0 + ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2), s.1 + (p.re as f64).powi(2) + (p.im as f64).powi(2))); (n / d.max(1e-300)).sqrt() };
    // All samples on an engine: (fwd ms, adj ms, A^H A x, gradient at x0, objective at x0, table bytes, table s); the
    // data y (host) are made on the first engine run (the reference when there is one): y = A x + 2 % noise
    let mut yh: Vec<C32> = vec![];
    let mut run_full = |so: SegNufftOpts, timed: bool, yh: &mut Vec<C32>| -> (f32, f32, Vec<C32>, Vec<C32>, f64, usize, f64) {
        let mut eng = new_eng(mk_so(width, sigma));
        eng.set_maps(&st, &smaps).unwrap();
        let seg = eng.segments(&st, &segtab.0, &segtab.1, segtab.2, segtab.3).unwrap();
        let tp = Instant::now();
        let pts = mk(&mut eng, None);
        st.synchronize().unwrap();
        let tp = tp.elapsed().as_secs_f64();
        if timed && std::env::var("HASTY_BENCH_TUNE").as_deref() == Ok("1") {
            let (kf, ka) = eng.autotune(&st, &seg, &pts).unwrap();
            println!("autotune (all samples): {kf:.1} / {ka:.1} ms");
        }
        let mut yf = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xo = st.alloc_zeros::<C32>(nx).unwrap();
        let (tf, ta) = if timed {
            eng.forward(&st, &seg, &pts, &x, &mut yf).unwrap();
            (time(&mut || eng.forward(&st, &seg, &pts, &x, &mut yf).unwrap()), time(&mut || eng.adjoint(&st, &seg, &pts, &yf, &mut xo).unwrap()))
        } else {
            (0.0, 0.0)
        };
        eng.forward(&st, &seg, &pts, &x, &mut yf).unwrap();
        eng.adjoint(&st, &seg, &pts, &yf, &mut xo).unwrap();
        let n = host(&xo);
        if yh.is_empty() {
            *yh = host(&yf);
            let sig = 0.02 * (yh.iter().map(|v| (v.re as f64).powi(2) + (v.im as f64).powi(2)).sum::<f64>() / yh.len() as f64).sqrt();
            for v in yh.iter_mut() {
                *v = C32::new(v.re + (sig * u() * 2.4) as f32, v.im + (sig * u() * 2.4) as f32);
            }
        }
        let ydev = st.clone_htod(yh).unwrap();
        eng.forward(&st, &seg, &pts, &x0, &mut yf).unwrap();
        kern.axpby(&st, -one, &ydev, one, &mut yf).unwrap();
        let f = kern.dot(&st, &yf, &yf).unwrap().re;
        eng.adjoint(&st, &seg, &pts, &yf, &mut xo).unwrap();
        (tf, ta, n, host(&xo), f, pts.table_bytes(), tp)
    };
    // reference: a more accurate kernel on all samples (the NUFFT's own error at W 10 / σ 1.25 is of the order of the
    // compression error, so both operators are compared with it)
    let refk: Option<(usize, f64)> = std::env::var("RC_BENCH_REF").ok().filter(|v| v != "none").map_or(Some((12, 2.0)), |v| { let (a, b) = v.split_once(',').unwrap(); Some((a.parse().unwrap(), b.parse().unwrap())) });
    let reference = refk.map(|(w, sg)| {
        let r = run_full(mk_so(w, sg), false, &mut yh);
        (r.2, r.3, r.4)
    });
    let (tf_full, ta_full, n_full, g_full, f_full, pts_full_b, tp_full) = run_full(mk_so(width, sigma), true, &mut yh);
    // kept samples on the main engine
    let mut eng = new_eng(mk_so(width, sigma));
    eng.set_maps(&st, &smaps).unwrap();
    let seg = eng.segments(&st, &segtab.0, &segtab.1, segtab.2, segtab.3).unwrap();
    let td2 = Instant::now();
    let (sc2, yt) = comp.compress_set(&cs, &lens, vox, &yh, nc);
    assert_eq!(sc2.sel, sc.sel);
    let t_cdata = td2.elapsed().as_secs_f64();
    drop(yh);
    let cnst: f64 = sc2.cnorm.iter().sum();
    let tp = Instant::now();
    let pts_c = mk(&mut eng, Some(&sc.sel));
    st.synchronize().unwrap();
    let tp_c = tp.elapsed().as_secs_f64();
    let rc = RcDev::new(&dev, &st, &comp, std::slice::from_ref(&sc)).unwrap();
    let mut yc = st.alloc_zeros::<C32>(nc * sc.m()).unwrap();
    let mut xo = st.alloc_zeros::<C32>(nx).unwrap();
    eng.forward(&st, &seg, &pts_c, &x, &mut yc).unwrap();
    let tf_c = time(&mut || eng.forward(&st, &seg, &pts_c, &x, &mut yc).unwrap());
    let ta_c = time(&mut || eng.adjoint(&st, &seg, &pts_c, &yc, &mut xo).unwrap());
    let tr_f = time(&mut || rc.apply(&st, 0, &mut yc, false).unwrap());
    let tr_a = time(&mut || rc.apply(&st, 0, &mut yc, true).unwrap());
    let (full, cmp, rstep) = (tf_full + ta_full, tf_c + ta_c, tr_f + tr_a);
    println!("GPU ({nimg} knot images{}, {nc} coils, coil batch {}, L {nseg}, order {order}, W {width}, σ {sigma}): all samples fwd {tf_full:.1} + adj {ta_full:.1} = {full:.1} ms; kept fwd {tf_c:.1} + adj {ta_c:.1} = {cmp:.1} ms; R step {tr_f:.2} + {tr_a:.2} ms; pass speed-up {:.3}× (tables {tp_full:.1}s / {tp_c:.1}s; data compression {t_cdata:.1}s on the CPU)", knots2d.map_or(String::new(), |k| format!(" ({} × {})", k.0, k.1)), eng.cb, full / (cmp + rstep));
    eng.forward(&st, &seg, &pts_c, &x, &mut yc).unwrap();
    rc.apply(&st, 0, &mut yc, false).unwrap();
    rc.apply(&st, 0, &mut yc, true).unwrap();
    eng.adjoint(&st, &seg, &pts_c, &yc, &mut xo).unwrap();
    let n_c = host(&xo);
    let ytdev = st.clone_htod(&yt).unwrap();
    eng.forward(&st, &seg, &pts_c, &x0, &mut yc).unwrap();
    rc.apply(&st, 0, &mut yc, false).unwrap();
    kern.axpby(&st, -one, &ytdev, one, &mut yc).unwrap();
    let f_c = kern.dot(&st, &yc, &yc).unwrap().re + cnst;
    rc.apply(&st, 0, &mut yc, true).unwrap();
    eng.adjoint(&st, &seg, &pts_c, &yc, &mut xo).unwrap();
    let g_c = host(&xo);
    let (e_normal, e_grad, e_obj) = (rel(&n_full, &n_c), rel(&g_full, &g_c), (f_full - f_c).abs() / f_full);
    println!("accuracy, compressed vs all samples (both W {width}, σ {sigma}): normal operator {e_normal:.2e}, gradient at 0.6 x_true {e_grad:.2e}, objective {e_obj:.2e} (data constant {:.3e} of ‖y‖² {:.3e})", cnst, sc2.ynorm.iter().sum::<f64>());
    let vs_ref = reference.as_ref().map(|(nr, gr, fr)| {
        let v = (rel(nr, &n_full), rel(nr, &n_c), rel(gr, &g_full), rel(gr, &g_c), (fr - f_full).abs() / fr, (fr - f_c).abs() / fr);
        println!("accuracy against the reference (all samples, W {}, σ {}): normal operator all samples {:.2e} / compressed {:.2e}; gradient {:.2e} / {:.2e}; objective {:.2e} / {:.2e}", refk.unwrap().0, refk.unwrap().1, v.0, v.1, v.2, v.3, v.4, v.5);
        v
    });
    js["gpu"] = serde_json::json!({
        "knot_images": nimg, "knots2d": knots2d, "coils": nc, "coil_batch": eng.cb, "nseg": nseg, "order": order, "width": width, "sigma": sigma,
        "fwd_full_ms": tf_full, "adj_full_ms": ta_full, "fwd_kept_ms": tf_c, "adj_kept_ms": ta_c, "r_fwd_ms": tr_f, "r_adj_ms": tr_a,
        "speedup": full / (cmp + rstep), "tables_full_s": tp_full, "tables_kept_s": tp_c, "data_compress_s": t_cdata,
        "err_normal": e_normal, "err_grad": e_grad, "err_obj": e_obj,
        "ref": refk, "vs_ref": vs_ref.map(|v| serde_json::json!({"normal_full": v.0, "normal_kept": v.1, "grad_full": v.2, "grad_kept": v.3, "obj_full": v.4, "obj_kept": v.5})),
        "pts_full_gib": pts_full_b as f64 / GIB, "pts_kept_gib": pts_c.table_bytes() as f64 / GIB,
    });
    if let Some(p) = &json_out {
        std::fs::write(p, serde_json::to_string_pretty(&js).unwrap()).unwrap();
    }
    println!("done ({:.0}s)", t0.elapsed().as_secs_f64());
}
