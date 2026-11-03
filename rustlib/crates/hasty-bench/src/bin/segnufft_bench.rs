//! Segmented physics NUFFT vs plain NUFFT: forward + adjoint time (ms) per call for all coils.
//!   segnufft_bench [nx ny nz M nc] ...
//! HASTY_BENCH_FRAMES="F,deg": time-resolved operator over F frames (M samples per frame, random times), B-spline deg.
use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, NufftOpts, NufftPlan, SegNufft, SegNufftOpts};

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let cases: Vec<[usize; 5]> = if a.len() >= 5 { a.chunks(5).map(|c| [c[0], c[1], c[2], c[3], c[4]]).collect() } else { vec![[180, 180, 28, 220_000, 12], [128, 128, 128, 1_000_000, 8]] };
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let ev = (dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap(), dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap());
    let time = |f: &mut dyn FnMut()| -> f32 {
        let mut t = Vec::new();
        for i in 0..5 {
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
    for c in cases {
        let dims = [c[0], c[1], c[2]];
        let (m, nc) = (c[3] * frames, c[4]);
        let nv = dims.iter().product::<usize>();
        let mut s = 12345u64;
        let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
        let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (u() * 6.2) as f32).collect());
        let x: Vec<C32> = (0..nv * frames).map(|_| C32::new(u() as f32, u() as f32)).collect();
        let tfr: Vec<f32> = (0..m).map(|_| ((u() + 0.5) * frames as f64) as f32).collect();
        let dx = st.clone_htod(&x).unwrap();
        println!("dims {dims:?}, M = {m}, coils = {nc}, frames = {frames} (B-spline {bspline})");
        // plain NUFFT reference (σ = 2, tol 1e-5 and σ = 1.25 W = 10)
        let plain: &[(f64, Option<usize>)] = if std::env::var("HASTY_BENCH_PLAIN").as_deref() == Ok("0") { &[] } else { &[(2.0, None), (1.25, Some(10usize))] };
        for &(sig, w) in plain {
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
            let mut op = SegNufft::new(&dev, dims, [0.002; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames, bspline, max_grid_bytes: 6 << 30 }).unwrap();
            op.set_maps(&st, &smaps).unwrap();
            let fit: Vec<usize> = (0..nv).step_by(97).collect();
            let psi = |v: usize, t: f64| 2000.0 * ((v % 113) as f64 / 113.0 - 0.5) * t;
            let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 128, &fit, &psi);
            let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
            let tau: Vec<f32> = (0..m).map(|_| (5e-3 + u() * 6e-3) as f32).collect();
            let dcv: Vec<f32> = (0..6 * m).map(|_| (u() * 10.0) as f32).collect();
            let tp = std::time::Instant::now();
            let pts = op.points(&st, [&coords[0], &coords[1], &coords[2]], &tau, &vec![0f32; m], &dcv, (frames > 1).then_some(&tfr[..])).unwrap();
            st.synchronize().unwrap();
            let tp = tp.elapsed().as_secs_f64() * 1e3;
            let def = (op.tiles_fwd, op.tiles_adj);
            if std::env::var("HASTY_BENCH_TUNE").as_deref() != Ok("0") {
                let tt = std::time::Instant::now();
                let (kf, ka) = op.autotune(&st, &seg, &pts).unwrap();
                println!("    autotune {:.1} s: kernels fwd {kf:.1} ms {:?}, adj {ka:.1} ms {:?} (default fwd {:?}, adj {:?})", tt.elapsed().as_secs_f64(), op.tiles_fwd, op.tiles_adj, def.0, def.1);
            }
            let mut y = st.alloc_zeros::<C32>(m * nc).unwrap();
            let mut xo = st.alloc_zeros::<C32>(nv * frames).unwrap();
            let tf = time(&mut || op.forward(&st, &seg, &pts, &dx, &mut y).unwrap());
            let ta = time(&mut || op.adjoint(&st, &seg, &pts, &y, &mut xo).unwrap());
            println!("  SegNufft L={nseg:>2} order {order}: forward {tf:.1} ms, adjoint {ta:.1} ms (coil batch {}, set_points {tp:.0} ms)", op.cb);
        }
    }
}
