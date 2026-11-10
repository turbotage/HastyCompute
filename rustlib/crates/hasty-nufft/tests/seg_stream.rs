//! GPU: host-resident (streamed) sample tables (`SegPoints::to_host`, the engine's two device slots, optional
//! prefetch on a second stream) give bit-identical results to resident tables: forward / adjoint, temporal frames,
//! order 3, the multi-image and joint water + fat paths, several sets cycling through the two slots, prefetch on and
//! off, and the round trip back to the device.
//!   cargo test --release -p hasty-nufft --test seg_stream -- --nocapture

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, build_segments_fat, hi_exponents, SegNufft, SegNufftOpts, SegPoints, SegSegments, NHI};
use hasty_testkit::Rng;

const PI: f64 = std::f64::consts::PI;
const DIMS: [usize; 3] = [20, 18, 12];
const RES: f64 = 3e-3;
const NC: usize = 4;

struct Set {
    k: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    t: Vec<f32>,
    u: Vec<f32>,
}

/// sets of different sizes (spiral-like readouts, random ΔC, frame times, order-3 weights)
fn sets(n: usize) -> Vec<Set> {
    let mut r = Rng::new(91);
    (0..n).map(|s| {
        let (nro, na) = (4 + 2 * s, 250usize);
        let mut k: [Vec<f32>; 3] = Default::default();
        let mut tau = vec![];
        for _ in 0..nro {
            let (kz, th) = (0.97 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
            for i in 0..na {
                let q = i as f64 / (na - 1) as f64;
                let (rad, ang) = (0.97 * PI * q.sqrt(), th + 2.0 * PI * 7.0 * q.sqrt());
                k[0].push((rad * ang.cos()) as f32);
                k[1].push((rad * ang.sin()) as f32);
                k[2].push(kz as f32);
                tau.push((1.6e-3 + 8e-3 * q) as f32);
            }
        }
        let m = tau.len();
        Set {
            k,
            phi0: (0..m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect(),
            dc: (0..6 * m).map(|_| (200.0 * (2.0 * r.uniform() - 1.0)) as f32).collect(),
            t: (0..m).map(|_| (4.0 * r.uniform()) as f32).collect(),
            u: (0..4 * m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect(),
            tau,
        }
    }).collect()
}

fn c32(v: &[hasty_testkit::C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}

fn same(a: &[C32], b: &[C32]) -> f64 {
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        n += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        d += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (n / d.max(1e-300)).sqrt()
}

/// forward + adjoint of every set in a cycling order, resident vs streamed tables, for one engine configuration
fn check(dev: &Device, order: u32, frames: usize, prefetch: bool, label: &str) {
    check_packed(dev, order, frames, prefetch, None, label)
}

/// `packed`: compress the host tables (`SegPoints::pack`) with this codec
fn check_packed(dev: &Device, order: u32, frames: usize, prefetch: bool, packed: Option<hasty_value::Options>, label: &str) {
    let st: Arc<CudaStream> = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    let l = 6;
    let ss = sets(3);
    let mut eng = SegNufft::new(dev, DIMS, [RES; 3], NC, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order, frames, bspline: 3, max_grid_bytes: 1 << 30 }).unwrap();
    let mut r = Rng::new(7);
    eng.set_maps(&st, &c32(&r.complex_vec(NC * nv))).unwrap();
    let om: Vec<f64> = (0..nv).map(|q| 2.0 * PI * 120.0 * ((q % DIMS[0]) as f64 / DIMS[0] as f64 - 0.5)).collect();
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let (sp, at, t0, dt) = build_segments(nv, l, 1.6e-3, 9.6e-3, 250, &fitv, &psi);
    let seg: SegSegments = eng.segments(&st, &sp, &at, t0, dt).unwrap();
    let ex = hi_exponents();
    let h: Vec<f64> = (0..4 * NHI).map(|i| 0.02 * (r.uniform() - 0.5) / 0.03f64.powi(ex[i % NHI].iter().sum::<u32>() as i32)).collect();
    let build = |eng: &mut SegNufft, s: &Set| -> SegPoints {
        let tt = (frames > 1).then_some(&s.t[..]);
        let dc: &[f32] = if order > 0 { &s.dc } else { &[] };
        if order >= 3 {
            eng.points_hi(&st, [&s.k[0], &s.k[1], &s.k[2]], &s.tau, &s.phi0, dc, &s.u, &h, tt).unwrap()
        } else {
            eng.points(&st, [&s.k[0], &s.k[1], &s.k[2]], &s.tau, &s.phi0, dc, tt).unwrap()
        }
    };
    let res: Vec<SegPoints> = ss.iter().map(|s| build(&mut eng, s)).collect();
    let mut host: Vec<SegPoints> = ss.iter().map(|s| build(&mut eng, s)).collect();
    let mut freed = 0;
    for p in host.iter_mut() {
        freed += p.to_host(&dev.ctx, &st).unwrap();
        assert!(p.is_host());
        if let Some(o) = &packed {
            let (a, b) = p.pack(o).unwrap();
            assert!(p.is_packed() && b < a, "packing did not shrink the tables: {a} → {b}");
        }
    }
    let stored: usize = host.iter().map(|p| p.host_stored_bytes()).sum();
    let logical: usize = host.iter().map(|p| p.table_bytes()).sum();
    eprintln!("{label}: host tables {:.2} MB stored for {:.2} MB (×{:.2})", stored as f64 / 1e6, logical as f64 / 1e6, logical as f64 / stored.max(1) as f64);
    eng.stream_reserve(&st, &host.iter().collect::<Vec<_>>()).unwrap();
    eng.stream_prefetch = prefetch;
    let xs: Vec<CudaSlice<C32>> = (0..3).map(|_| st.clone_htod(&c32(&r.complex_vec(frames * nv))).unwrap()).collect();
    let ys: Vec<CudaSlice<C32>> = ss.iter().map(|s| st.clone_htod(&c32(&r.complex_vec(NC * s.tau.len()))).unwrap()).collect();
    // cycle 0, 1, 2, 0, 2, 1 (three sets through two slots, prefetch hits and misses)
    let (mut worst, mut spread) = (0.0f64, 0.0f64);
    for &t in &[0usize, 1, 2, 0, 2, 1] {
        let m = ss[t].tau.len();
        let run = |eng: &mut SegNufft, p: &SegPoints| -> (Vec<C32>, Vec<C32>) {
            let mut y = st.alloc_zeros::<C32>(NC * m).unwrap();
            let mut x = st.alloc_zeros::<C32>(frames * nv).unwrap();
            eng.forward(&st, &seg, p, &xs[t], &mut y).unwrap();
            if prefetch {
                eng.prefetch(&st, &host[(t + 1) % 3]).unwrap();
            }
            eng.adjoint(&st, &seg, p, &ys[t], &mut x).unwrap();
            (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&x).unwrap())
        };
        let (yr, xr) = run(&mut eng, &res[t]);
        let (yh, xh) = run(&mut eng, &host[t]);
        worst = worst.max(same(&yh, &yr)).max(same(&xh, &xr));
        // the resident path's own run-to-run spread (float32 atomics in the adjoint)
        let (yr2, xr2) = run(&mut eng, &res[t]);
        spread = spread.max(same(&yr2, &yr)).max(same(&xr2, &xr));
        // the multi-image path (two images, a weight) on streamed tables
        let w = st.clone_htod(&c32(&r.complex_vec(m))).unwrap();
        let mut ym = [st.alloc_zeros::<C32>(NC * m).unwrap(), st.alloc_zeros::<C32>(NC * m).unwrap()];
        for (i, p) in [&res[t], &host[t]].iter().enumerate() {
            eng.forward_multi(&st, &seg, p, &[&xs[t], &xs[(t + 1) % 3]], &[None, Some(&w)], &mut ym[i]).unwrap();
        }
        worst = worst.max(same(&st.clone_dtoh(&ym[1]).unwrap(), &st.clone_dtoh(&ym[0]).unwrap()));
    }
    let (n_up, bytes) = eng.stream_stats();
    eprintln!("{label}: streamed vs resident max rel. difference {worst:.1e} (resident repeat {spread:.1e}); {} uploads, {:.1} MB, {:.1} MB freed, slots {:.1} MB", n_up, bytes as f64 / 1e6, freed as f64 / 1e6, eng.stream_slot_bytes() as f64 / 1e6);
    assert!(worst <= (3.0 * spread).max(2e-5), "{label}: streamed tables differ from resident ones: {worst:.1e} (resident repeat {spread:.1e})");
    assert!(n_up >= 3, "tables were not streamed");
    // back to the device
    for p in host.iter_mut() {
        p.to_device(&st).unwrap();
        assert!(!p.is_host());
    }
    let mut y = st.alloc_zeros::<C32>(NC * ss[0].tau.len()).unwrap();
    let mut y2 = st.alloc_zeros::<C32>(NC * ss[0].tau.len()).unwrap();
    eng.forward(&st, &seg, &host[0], &xs[0], &mut y).unwrap();
    eng.forward(&st, &seg, &res[0], &xs[0], &mut y2).unwrap();
    assert!(same(&st.clone_dtoh(&y).unwrap(), &st.clone_dtoh(&y2).unwrap()) <= 1e-6);
}

#[test]
fn streamed_tables_match_resident() {
    let dev = Device::new(0).expect("CUDA device");
    for prefetch in [false, true] {
        check(&dev, 2, 1, prefetch, &format!("order 2, static, prefetch {prefetch}"));
        check(&dev, 2, 4, prefetch, &format!("order 2, 4 frames (cubic), prefetch {prefetch}"));
        check(&dev, 3, 1, prefetch, &format!("order 3, static, prefetch {prefetch}"));
    }
    check(&dev, 0, 1, true, "order 0, static, prefetch true");
}

/// packed (compressed) host tables decode into the slots' pinned staging buffers: same results
#[test]
fn packed_tables_match_resident() {
    let dev = Device::new(0).expect("CUDA device");
    for prefetch in [false, true] {
        check_packed(&dev, 2, 1, prefetch, Some(hasty_value::Options::zstd(1)), &format!("packed zstd1, order 2, static, prefetch {prefetch}"));
        check_packed(&dev, 3, 1, prefetch, Some(hasty_value::Options::default()), &format!("packed lz4, order 3, static, prefetch {prefetch}"));
    }
    check_packed(&dev, 2, 4, true, Some(hasty_value::Options::zstd(1)), "packed zstd1, order 2, 4 frames, prefetch true");
}

/// the joint water + fat path (`forward_fat`) on streamed tables
#[test]
fn streamed_tables_joint_fat() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    let l = 8;
    let ss = sets(2);
    let mut eng = SegNufft::new(&dev, DIMS, [RES; 3], NC, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    let mut r = Rng::new(8);
    eng.set_maps(&st, &c32(&r.complex_vec(NC * nv))).unwrap();
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| 2.0 * PI * 100.0 * (v % 7) as f64 / 7.0 * t;
    let c = |t: f64| hasty_testkit::C64::from_polar(1.0, -2.0 * PI * 420.0 * t);
    let (sw, aw, cl, t0, dt) = build_segments_fat(nv, l, 1.6e-3, 9.6e-3, 250, &fitv, &psi, &c);
    let seg = eng.segments_fat(&st, &sw, &aw, &cl, t0, dt).unwrap();
    let maps = eng.fat_maps(&st, &(0..nv).map(|_| r.uniform() as f32).collect::<Vec<_>>()).unwrap();
    let s = &ss[1];
    let m = s.tau.len();
    let res = eng.points(&st, [&s.k[0], &s.k[1], &s.k[2]], &s.tau, &s.phi0, &[], None).unwrap();
    let mut host = eng.points(&st, [&s.k[0], &s.k[1], &s.k[2]], &s.tau, &s.phi0, &[], None).unwrap();
    host.to_host(&dev.ctx, &st).unwrap();
    let x = st.clone_htod(&c32(&r.complex_vec(nv))).unwrap();
    let y = st.clone_htod(&c32(&r.complex_vec(NC * m))).unwrap();
    let mut out = vec![];
    for p in [&res, &host] {
        let mut a = st.alloc_zeros::<C32>(NC * m).unwrap();
        let mut b = st.alloc_zeros::<C32>(nv).unwrap();
        eng.forward_fat(&st, &seg, p, &maps, Some(0), &x, &mut a).unwrap();
        eng.adjoint_fat(&st, &seg, p, &maps, Some(0), &y, &mut b).unwrap();
        out.push((st.clone_dtoh(&a).unwrap(), st.clone_dtoh(&b).unwrap()));
    }
    let d = same(&out[1].0, &out[0].0).max(same(&out[1].1, &out[0].1));
    // the resident path's own run-to-run spread (float32 atomics in the adjoint)
    let spread = {
        let mut b = st.alloc_zeros::<C32>(nv).unwrap();
        eng.adjoint_fat(&st, &seg, &res, &maps, Some(0), &y, &mut b).unwrap();
        same(&st.clone_dtoh(&b).unwrap(), &out[0].1)
    };
    eprintln!("joint water + fat, streamed vs resident: {d:.1e} (resident repeat {spread:.1e})");
    assert!(d <= (3.0 * spread).max(2e-5), "joint water + fat streamed vs resident {d:.1e} (resident repeat {spread:.1e})");
}
