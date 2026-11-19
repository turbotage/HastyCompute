//! Segmented physics NUFFT vs plain NUFFT: forward + adjoint time (ms) per call for all coils.
//!   segnufft_bench [nx ny nz M nc] ...
//! HASTY_BENCH_FRAMES="F,deg": time-resolved operator over F frames (M samples per frame, random times), B-spline deg.
//! HASTY_BENCH_GRID_GB: fine-grid budget (GiB, default 6; sets the coil batch).
//! HASTY_BENCH_KNOT=1: knot-major temporal operator (`SegNufft::new_temporal(.., true)`).
//! HASTY_BENCH_2D="kr,rdeg": respiratory × cardiac knots (`SegNufft::new_temporal2d`, kr respiratory knots of degree
//! rdeg, the FRAMES knots cardiac; M samples per knot image, so kr · F · M samples).
//! HASTY_BENCH_W: kernel width (default 10); HASTY_BENCH_RES: voxel size in m (default 0.002).
//! HASTY_BENCH_TRAJ: sample distribution in k·Δ: cube (default, uniform in [−3.1, 3.1]³), ball (uniform in the ball of
//! radius 3.1: yarn-ball coverage), koosh (3-D radial / kooshball: random spoke directions, radius uniform in [0, 3.1],
//! so density ∝ 1/k², centre-dense) or sos (stack of spirals: uniform in the disk of radius 3.1 × kz).
//! HASTY_SEG_KPRUNE=0 turns the k-support pruning of the grid FFTs off, HASTY_SEG_KERNEL=kb selects the KB window,
//! HASTY_NUFFT_GRID=auto the measured fine-grid sizes (hasty-nufft; research/temporal_operator.md §20).
//! Order 3 uses the order-3 sample tables (`points_hi` / `points2d_hi`, random degree-3/4 terms); HASTY_BENCH_HI=0
//! keeps the plain tables (the order-2 operator on an order-3 engine, the old behaviour).
//! HASTY_BENCH_REPS: timed calls per direction (median; default 4, plus one warm-up).
//! Memory plan: before each case the device bytes it will allocate are estimated per component (printed) and the case
//! is skipped when they exceed the free memory (HASTY_BENCH_FORCE=1 runs it anyway); allocation / launch errors skip
//! the case instead of aborting. The measured device-memory growth of each stage is printed with the timings.
use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, grid_default_size, NufftOpts, NufftPlan, SegNufft, SegNufftOpts, NHI};

const GIB: f64 = (1u64 << 30) as f64;

/// device bytes in use (other processes included), after the stream is idle
fn used(dev: &Device, st: &std::sync::Arc<cudarc::driver::CudaStream>) -> usize {
    st.synchronize().unwrap();
    let (free, total) = dev.ctx.mem_get_info().unwrap();
    total - free
}

/// return the stream-ordered pool's free blocks to the device (freed buffers stay in the pool otherwise, and the next
/// case's free-memory check would see them as used)
fn trim_pool(dev: &Device, st: &std::sync::Arc<cudarc::driver::CudaStream>) {
    st.synchronize().unwrap();
    unsafe {
        let mut pool = std::mem::MaybeUninit::uninit();
        if cudarc::driver::sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), dev.ctx.cu_device()) == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            cudarc::driver::sys::cuMemPoolTrimTo(pool.assume_init(), 0);
        }
    }
}

/// print the estimated per-component bytes of a case and whether they fit in the free device memory
fn plan_fits(dev: &Device, label: &str, parts: &[(&str, usize)]) -> bool {
    let need: usize = parts.iter().map(|p| p.1).sum();
    let free = dev.ctx.mem_get_info().map(|(f, _)| f).unwrap_or(usize::MAX);
    // 3 % + 256 MiB headroom: allocator granularity, kernel modules, cuFFT-free but twiddle / tile tables
    let fits = (need as f64 * 1.03) as usize + (256 << 20) <= free;
    let list: Vec<String> = parts.iter().filter(|p| p.1 > 0).map(|(n, b)| format!("{n} {:.2}", *b as f64 / GIB)).collect();
    println!("    memory plan {label}: {:.2} GiB ({}) of {:.2} GiB free{}", need as f64 / GIB, list.join(", "), free as f64 / GIB, if fits { "" } else { " -> does not fit" });
    fits || std::env::var("HASTY_BENCH_FORCE").is_ok_and(|v| v != "0")
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let cases: Vec<[usize; 5]> = if a.len() >= 5 { a.chunks(5).map(|c| [c[0], c[1], c[2], c[3], c[4]]).collect() } else { vec![[180, 180, 28, 220_000, 12], [128, 128, 128, 1_000_000, 8]] };
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let ev = (dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap(), dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap());
    let reps: usize = std::env::var("HASTY_BENCH_REPS").ok().map_or(4, |v| v.parse().unwrap());
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
    let (frames, bspline): (usize, u32) = std::env::var("HASTY_BENCH_FRAMES").ok().map(|v| { let mut it = v.split(','); (it.next().unwrap().parse().unwrap(), it.next().map_or(1, |b| b.parse().unwrap())) }).unwrap_or((1, 1));
    let r2d: Option<(usize, u32)> = std::env::var("HASTY_BENCH_2D").ok().map(|v| { let mut it = v.split(','); (it.next().unwrap().parse().unwrap(), it.next().map_or(1, |b| b.parse().unwrap())) });
    let width: usize = std::env::var("HASTY_BENCH_W").ok().map_or(10, |v| v.parse().unwrap());
    let res: f64 = std::env::var("HASTY_BENCH_RES").ok().map_or(0.002, |v| v.parse().unwrap());
    for c in cases {
        let dims = [c[0], c[1], c[2]];
        let nimg = frames * r2d.map_or(1, |r| r.0);
        let (m, nc) = (c[3] * nimg, c[4]);
        let nv = dims.iter().product::<usize>();
        let mut s = 12345u64;
        let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
        let traj = std::env::var("HASTY_BENCH_TRAJ").unwrap_or_else(|_| "cube".into());
        let coords: [Vec<f32>; 3] = match traj.as_str() {
            "cube" => [0, 1, 2].map(|_| (0..m).map(|_| (u() * 6.2) as f32).collect()),
            "ball" | "sos" => {
                let mut c: [Vec<f32>; 3] = Default::default();
                while c[0].len() < m {
                    let p = [u() * 6.2, u() * 6.2, u() * 6.2];
                    let r2 = p[0] * p[0] + p[1] * p[1] + if traj == "ball" { p[2] * p[2] } else { 0.0 };
                    if r2 <= 3.1 * 3.1 {
                        (0..3).for_each(|a| c[a].push(p[a] as f32));
                    }
                }
                c
            }
            "koosh" => {
                // 3-D radial: random spoke directions, radius uniform in [0, 3.1] (density ∝ 1/k², centre-dense)
                let mut c: [Vec<f32>; 3] = Default::default();
                for _ in 0..m {
                    let (cz, ph, r) = (2.0 * u(), 2.0 * std::f64::consts::PI * (u() + 0.5), 3.1 * (u() + 0.5));
                    let sz = (1.0 - cz * cz).max(0.0).sqrt();
                    let p = [r * sz * ph.cos(), r * sz * ph.sin(), r * cz];
                    (0..3).for_each(|a| c[a].push(p[a] as f32));
                }
                c
            }
            t => panic!("HASTY_BENCH_TRAJ={t}: cube, ball, sos or koosh"),
        };
        let x: Vec<C32> = (0..nv * nimg).map(|_| C32::new(u() as f32, u() as f32)).collect();
        let tfr: Vec<f32> = (0..m).map(|_| ((u() + 0.5) * frames as f64) as f32).collect();
        // 2-D: respiratory interval and fraction (nri = kr for bins, kr − 1 for linear)
        let (rir, rvr): (Vec<u32>, Vec<f32>) = match r2d {
            Some((kr, rdeg)) => {
                let nri = if rdeg == 0 { kr } else { kr - 1 };
                (0..m).map(|_| { let r = (u() + 0.5) * nri as f64; ((r.floor() as u32).min(nri as u32 - 1), (r - r.floor()) as f32) }).unzip()
            }
            None => (vec![], vec![]),
        };
        let dx = st.clone_htod(&x).unwrap();
        println!("dims {dims:?}, M = {m}, coils = {nc}, frames = {frames} (B-spline {bspline}), 2-D {r2d:?}, W {width}");
        // plain NUFFT reference (σ = 2, tol 1e-5 and σ = 1.25 W = 10)
        let plain: &[(f64, Option<usize>)] = if std::env::var("HASTY_BENCH_PLAIN").as_deref() == Ok("0") { &[] } else { &[(2.0, None), (1.25, Some(10usize))] };
        for &(sig, w) in plain {
            let gb = 3usize << 30;
            let ge: usize = dims.iter().map(|&d| grid_default_size(d, sig, w.unwrap_or(8))).product();
            let pl = [("grid", (gb / (8 * ge)).clamp(1, nc) * ge * 8), ("x + out", 2 * nv * nc * 8), ("y", m * nc * 8), ("points", m * 40)];
            trim_pool(&dev, &st);
            if !plan_fits(&dev, &format!("plain σ={sig}"), &pl) {
                println!("  plain NUFFT σ={sig}: skipped (memory)");
                continue;
            }
            let mut p = NufftPlan::<f32>::new(&dev, &dims, NufftOpts { tol: 1e-5, sigma: sig, width: w, ntransf: nc, ..Default::default() }).unwrap();
            let dc: Vec<_> = coords.iter().map(|v| st.clone_htod(v).unwrap()).collect();
            p.set_points(&st, &[&dc[0], &dc[1], &dc[2]], m).unwrap();
            let fx = st.clone_htod(&x.iter().cycle().take(nv * nc).cloned().collect::<Vec<_>>()).unwrap();
            let mut y = st.alloc_zeros::<C32>(m * nc).unwrap();
            let mut fo = st.alloc_zeros::<C32>(nv * nc).unwrap();
            let tf = time(&mut || p.type2(&st, &fx, &mut y, -1).unwrap());
            let ta = time(&mut || p.type1(&st, &y, &mut fo, 1).unwrap());
            println!("  plain NUFFT σ={sig} W={}: forward {tf:.1} ms, adjoint {ta:.1} ms", p.ker.w);
        }
        let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new(1.0 / (1.0 + (i % 7) as f32), 0.1)).collect();
        // HASTY_BENCH_CFG="nseg,order;nseg,order" restricts the configurations; HASTY_BENCH_PLAIN=0 skips the plain NUFFT
        let cfgs: Vec<(usize, u32)> = match std::env::var("HASTY_BENCH_CFG") {
            Ok(v) => v.split(';').map(|c| { let mut it = c.split(','); (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap()) }).collect(),
            Err(_) => vec![(1, 0), (8, 0), (8, 1), (8, 2), (10, 2), (12, 2)],
        };
        for (nseg, order) in cfgs {
            let knot = std::env::var("HASTY_BENCH_KNOT").is_ok_and(|v| v != "0");
            let so = SegNufftOpts { sigma: 1.25, width, nseg, order, frames, bspline, max_grid_bytes: std::env::var("HASTY_BENCH_GRID_GB").ok().map_or(6 << 30, |v| (v.parse::<f64>().unwrap() * (1u64 << 30) as f64) as usize) };
            let hi = order >= 3 && std::env::var("HASTY_BENCH_HI").as_deref() != Ok("0");
            // memory plan (estimates of what the engine and this case allocate; `dx` is already resident)
            let ge: usize = dims.iter().map(|&d| grid_default_size(d, 1.25, width)).product();
            let nslots = if (knot || r2d.is_some()) && frames > 1 { 1 } else if frames > 1 { bspline as usize + 1 } else { 1 };
            let cb = (so.max_grid_bytes / (nslots * nseg * ge * 8)).clamp(1, nc.min(16));
            let mem = [
                ("grid slots", nslots * cb * nseg * ge * 8),
                ("plan scratch (released)", ge * 8),
                ("coil maps", nc * nv * 8),
                ("segment diagonals", nseg * nv * 8),
                // ≈ 200 B per sample at the stage's peak (measured: upload, sorting, 48 B records, tile lists)
                ("sample tables", m * (200 + if hi { 16 } else { 0 })),
                ("y", m * nc * 8),
                ("forward coil-batch y", cb * m * 8),
                ("adjoint x", nv * nimg * 8),
            ];
            trim_pool(&dev, &st);
            if !plan_fits(&dev, &format!("L={nseg} order {order} (coil batch {cb}, grid {ge} cells)"), &mem) {
                println!("  SegNufft L={nseg:>2} order {order}: skipped (memory; HASTY_BENCH_FORCE=1 to try)");
                continue;
            }
            let outcome: Result<(), Box<dyn std::error::Error>> = (|| {
                let u0 = used(&dev, &st);
                let mut op = match r2d {
                    Some((kr, rdeg)) => SegNufft::new_temporal2d(&dev, dims, [res; 3], nc, so, kr, rdeg)?,
                    None => SegNufft::new_temporal(&dev, dims, [res; 3], nc, so, knot)?,
                };
                let u1 = used(&dev, &st);
                op.set_maps(&st, &smaps)?;
                let u2 = used(&dev, &st);
                let fit: Vec<usize> = (0..nv).step_by(97).collect();
                let psi = |v: usize, t: f64| 2000.0 * ((v % 113) as f64 / 113.0 - 0.5) * t;
                let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 128, &fit, &psi);
                let seg = op.segments(&st, &sp, &at, t0, dt)?;
                drop(sp);
                let u3 = used(&dev, &st);
                let tau: Vec<f32> = (0..m).map(|_| (5e-3 + u() * 6e-3) as f32).collect();
                let dcv: Vec<f32> = (0..6 * m).map(|_| (u() * 10.0) as f32).collect();
                // order 3: degree-3/4 terms of the size of a GNL-warped concomitant fit's remainder (random)
                let uh: Vec<f32> = if hi { (0..4 * m).map(|_| (u() * 2.0) as f32).collect() } else { vec![] };
                let hh: Vec<f64> = if hi { (0..4 * NHI).map(|_| u() * 50.0).collect() } else { vec![] };
                let tp = std::time::Instant::now();
                let ph0 = vec![0f32; m];
                let pts = match (r2d.is_some(), hi) {
                    (true, true) => op.points2d_hi(&st, [&coords[0], &coords[1], &coords[2]], &tau, &ph0, &dcv, &tfr, &rir, &rvr, &uh, &hh)?,
                    (true, false) => op.points2d(&st, [&coords[0], &coords[1], &coords[2]], &tau, &ph0, &dcv, &tfr, &rir, &rvr)?,
                    (false, true) => op.points_hi(&st, [&coords[0], &coords[1], &coords[2]], &tau, &ph0, &dcv, &uh, &hh, (frames > 1).then_some(&tfr[..]))?,
                    (false, false) => op.points(&st, [&coords[0], &coords[1], &coords[2]], &tau, &ph0, &dcv, (frames > 1).then_some(&tfr[..]))?,
                };
                st.synchronize()?;
                let tp = tp.elapsed().as_secs_f64() * 1e3;
                let u4 = used(&dev, &st);
                let def = (op.tiles_fwd, op.tiles_adj);
                if std::env::var("HASTY_BENCH_TUNE").as_deref() != Ok("0") {
                    let tt = std::time::Instant::now();
                    let (kf, ka) = op.autotune(&st, &seg, &pts)?;
                    println!("    autotune {:.1} s: kernels fwd {kf:.1} ms {:?}, adj {ka:.1} ms {:?} (default fwd {:?}, adj {:?})", tt.elapsed().as_secs_f64(), op.tiles_fwd, op.tiles_adj, def.0, def.1);
                }
                let mut y = st.alloc_zeros::<C32>(m * nc)?;
                let mut xo = st.alloc_zeros::<C32>(nv * nimg)?;
                op.forward(&st, &seg, &pts, &dx, &mut y)?;
                op.adjoint(&st, &seg, &pts, &y, &mut xo)?;
                st.synchronize()?;
                let tf = time(&mut || op.forward(&st, &seg, &pts, &dx, &mut y).unwrap());
                let ta = time(&mut || op.adjoint(&st, &seg, &pts, &y, &mut xo).unwrap());
                let u5 = used(&dev, &st);
                let nfg = [op.plan.nf[0], op.plan.nf[1], op.plan.nf[2]];
                let ks = op.grid_fft_ksupport().map(|k| (k.fractions(nfg), k.cell_fraction(nfg)));
                println!("  SegNufft L={nseg:>2} order {order}{}: forward {tf:.1} ms, adjoint {ta:.1} ms (coil batch {}, set_points {tp:.0} ms; traj {traj}, grid {:?}, pruned FFT {}, window {:?}, k-support {})", if order >= 3 && !hi { " (plain tables)" } else { "" }, op.cb, op.plan.nf, op.grid_fft_pruned(), op.kernel_kind(), ks.map_or("off".to_string(), |((c, p), v)| format!("columns {c:.2}, proj {p:.2?}, cells {v:.2}")));
                let g = |a: usize, b: usize| b.saturating_sub(a) as f64 / GIB;
                println!("    device memory measured (GiB, stage peaks: pool growth): engine {:.2}, maps {:.2}, segments {:.2}, sample tables {:.2}, y + x + forward work {:.2}; peak case total {:.2}", g(u0, u1), g(u1, u2), g(u2, u3), g(u3, u4), g(u4, u5), g(u0, u5));
                Ok(())
            })();
            if let Err(e) = outcome {
                println!("  SegNufft L={nseg:>2} order {order}: failed: {e}");
            }
            trim_pool(&dev, &st);
        }
    }
}
