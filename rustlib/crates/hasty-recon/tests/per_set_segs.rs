//! GPU: per-set segment tables (`setop::PerSetSegs`, one ratemap per operator frame) give exactly the operator whose
//! tables are all resident and selected by `seg_of_set`: forward, adjoint, forward_residual and the group calls, with
//! sets of 3 frames × 2 encodes (set t = frame t / 2, encode t % 2) and distinct fields per frame.
//!   cargo test --release -p hasty-recon --test per_set_segs -- --nocapture

use std::f64::consts::PI;
use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::gpu::Kernels;
use hasty_recon::setop::{HostSegTab, PerSetSegs, SegSets, SetOp};
use hasty_testkit::Rng;
use num_complex::Complex64 as C64;

const DIMS: [usize; 3] = [24, 20, 14];
const NF: usize = 3;
const NE: usize = 2;

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

#[test]
fn per_set_segs_equal_resident_tables() {
    let res = 3e-3;
    let nv: usize = DIMS.iter().product();
    let nc = 4;
    let idx: Vec<[f64; 3]> = (0..nv)
        .map(|q| {
            let (i, j, k) = (q % DIMS[0], (q / DIMS[0]) % DIMS[1], q / (DIMS[0] * DIMS[1]));
            [i as f64 - (DIMS[0] / 2) as f64, j as f64 - (DIMS[1] / 2) as f64, k as f64 - (DIMS[2] / 2) as f64]
        })
        .collect();
    let smaps: Vec<C32> = (0..nc)
        .flat_map(|c| {
            let th = 2.0 * PI * c as f64 / nc as f64;
            idx.iter().map(move |p| {
                let z = C64::from_polar(1.0 / (1.0 + 0.02 * ((p[0] - 8.0 * th.cos()).powi(2) + (p[1] - 8.0 * th.sin()).powi(2))), 0.1 * (p[0] * th.sin() - p[1] * th.cos()));
                C32::new(z.re as f32, z.im as f32)
            })
        })
        .collect::<Vec<_>>();
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let l = 6;
    let opts = || SegNufftOpts { sigma: 1.25, width: 8, nseg: l, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 };
    let mut eng_ref = SegNufft::new(&dev, DIMS, [res; 3], nc, opts()).unwrap();
    let mut eng_tst = SegNufft::new(&dev, DIMS, [res; 3], nc, opts()).unwrap();
    // control: a second engine with every table resident, the same as the reference. Two engines differ at fp32 rounding
    // level (spreading atomics in the adjoint, autotuned kernel choices); their difference is the floor for the test.
    let mut eng_ctl = SegNufft::new(&dev, DIMS, [res; 3], nc, opts()).unwrap();
    eng_ref.set_maps(&st, &smaps).unwrap();
    eng_tst.set_maps(&st, &smaps).unwrap();
    eng_ctl.set_maps(&st, &smaps).unwrap();
    // sample tables per set (3 frames × 2 encodes)
    let mut r = Rng::new(11);
    let (nro, na, te, t_ro) = (6usize, 300usize, 1.5e-3, 4e-3);
    let nsets = NF * NE;
    let (mut pr, mut pt, mut pc, mut ms) = (vec![], vec![], vec![], vec![]);
    let mut taus: Vec<f64> = vec![];
    for _ in 0..nsets {
        let mut k: [Vec<f32>; 3] = Default::default();
        let mut tau = vec![];
        for _ in 0..nro {
            let (kz, th) = (0.95 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
            for n in 0..na {
                let s = n as f64 / (na - 1) as f64;
                let (rad, ang) = (0.95 * PI * s.sqrt(), th + 2.0 * PI * 4.0 * s.sqrt());
                k[0].push((rad * ang.cos()) as f32);
                k[1].push((rad * ang.sin()) as f32);
                k[2].push(kz as f32);
                tau.push(te + t_ro * s);
            }
        }
        let m = tau.len();
        let tau32: Vec<f32> = tau.iter().map(|&t| t as f32).collect();
        pr.push(eng_ref.points(&st, [&k[0], &k[1], &k[2]], &tau32, &vec![0f32; m], &[], None).unwrap());
        pt.push(eng_tst.points(&st, [&k[0], &k[1], &k[2]], &tau32, &vec![0f32; m], &[], None).unwrap());
        pc.push(eng_ctl.points(&st, [&k[0], &k[1], &k[2]], &tau32, &vec![0f32; m], &[], None).unwrap());
        ms.push(m);
        taus = tau;
    }
    // one field per frame (distinct), host tables
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let tabs: Vec<HostSegTab> = (0..NF)
        .map(|f| {
            let a = 60.0 * (1.0 + 0.7 * f as f64);
            let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * a * (p[0] / 12.0 * 0.6 - 0.3 * (p[1] / 10.0).powi(2) + 0.1 * f as f64 * p[2] / 7.0)).collect();
            let psi = |v: usize, t: f64| om[v] * t;
            let (segph, atab, t0, dt) = build_segments(nv, l, taus[0], taus[taus.len() - 1], na, &fitv, &psi);
            HostSegTab { segph, atab, t0, dt }
        })
        .collect();
    let frame = |t: usize| t / NE;
    // reference: every frame's table resident, selected by seg_of_set
    let segs_ref = tabs.iter().map(|h| eng_ref.segments(&st, &h.segph, &h.atab, h.t0, h.dt).unwrap()).collect();
    let mut refs = SegSets { eng: eng_ref, pts: pr, segs: segs_ref, seg_of_set: (0..nsets).map(frame).collect(), kern: kern.clone(), surrogate: None };
    let segs_ctl = tabs.iter().map(|h| eng_ctl.segments(&st, &h.segph, &h.atab, h.t0, h.dt).unwrap()).collect();
    let mut ctl = SegSets { eng: eng_ctl, pts: pc, segs: segs_ctl, seg_of_set: (0..nsets).map(frame).collect(), kern: kern.clone(), surrogate: None };
    // test: one slot holding a wrong (frame 0 × 0.5 field) table at the start, loaded per set by PerSetSegs
    let wrong = {
        let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * 30.0 * p[0] / 12.0).collect();
        let psi = |v: usize, t: f64| om[v] * t;
        let (segph, atab, t0, dt) = build_segments(nv, l, taus[0], taus[taus.len() - 1], na, &fitv, &psi);
        eng_tst.segments(&st, &segph, &atab, t0, dt).unwrap()
    };
    let mut inner = SegSets { eng: eng_tst, pts: pt, segs: vec![wrong], seg_of_set: vec![0; nsets], kern: kern.clone(), surrogate: None };
    let mut tst = PerSetSegs::new(&st, &mut inner, Arc::new(tabs), (0..nsets).map(frame).collect(), vec![0; nsets]).unwrap();
    // objects
    let xs: Vec<Vec<C32>> = (0..nsets)
        .map(|_| (0..nv).map(|_| C32::new(r.uniform() as f32 - 0.5, r.uniform() as f32 - 0.5)).collect())
        .collect();
    let dxs: Vec<_> = xs.iter().map(|x| st.clone_htod(x).unwrap()).collect();
    let (mut worst, mut floor) = (0.0f64, 0.0f64);
    // per set, in a scrambled order (frames change back and forth)
    for &t in &[3usize, 0, 5, 1, 4, 2, 0, 5] {
        let mut ya = st.alloc_zeros::<C32>(nc * ms[t]).unwrap();
        let mut yb = st.alloc_zeros::<C32>(nc * ms[t]).unwrap();
        refs.forward(&st, t, &dxs[t], &mut ya).unwrap();
        tst.forward(&st, t, &dxs[t], &mut yb).unwrap();
        let e_f = rel(&st.clone_dtoh(&yb).unwrap(), &st.clone_dtoh(&ya).unwrap());
        let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
        let mut xb = st.alloc_zeros::<C32>(nv).unwrap();
        refs.adjoint(&st, t, &ya, &mut xa).unwrap();
        tst.adjoint(&st, t, &ya, &mut xb).unwrap();
        let e_a = rel(&st.clone_dtoh(&xb).unwrap(), &st.clone_dtoh(&xa).unwrap());
        let mut ra = st.alloc_zeros::<C32>(nc * ms[t]).unwrap();
        let mut rb = st.alloc_zeros::<C32>(nc * ms[t]).unwrap();
        refs.forward_residual(&st, t, &dxs[(t + 1) % nsets], &ya, &mut ra).unwrap();
        tst.forward_residual(&st, t, &dxs[(t + 1) % nsets], &ya, &mut rb).unwrap();
        let e_r = rel(&st.clone_dtoh(&rb).unwrap(), &st.clone_dtoh(&ra).unwrap());
        // control engine, same calls
        let mut yc = st.alloc_zeros::<C32>(nc * ms[t]).unwrap();
        ctl.forward(&st, t, &dxs[t], &mut yc).unwrap();
        let c_f = rel(&st.clone_dtoh(&yc).unwrap(), &st.clone_dtoh(&ya).unwrap());
        let mut xc = st.alloc_zeros::<C32>(nv).unwrap();
        ctl.adjoint(&st, t, &ya, &mut xc).unwrap();
        let c_a = rel(&st.clone_dtoh(&xc).unwrap(), &st.clone_dtoh(&xa).unwrap());
        let mut rc = st.alloc_zeros::<C32>(nc * ms[t]).unwrap();
        ctl.forward_residual(&st, t, &dxs[(t + 1) % nsets], &ya, &mut rc).unwrap();
        let c_r = rel(&st.clone_dtoh(&rc).unwrap(), &st.clone_dtoh(&ra).unwrap());
        eprintln!("set {t} (frame {}): forward {e_f:.2e}, adjoint {e_a:.2e}, residual {e_r:.2e}  (control engine: {c_f:.2e}, {c_a:.2e}, {c_r:.2e})", frame(t));
        worst = worst.max(e_f).max(e_a).max(e_r);
        floor = floor.max(c_f).max(c_a).max(c_r);
    }
    // group calls over all sets with one shared image (the pooled warm start's pattern): split by table
    let members: Vec<usize> = (0..nsets).collect();
    let mut ya: Vec<_> = (0..nsets).map(|t| st.alloc_zeros::<C32>(nc * ms[t]).unwrap()).collect();
    let mut yb: Vec<_> = (0..nsets).map(|t| st.alloc_zeros::<C32>(nc * ms[t]).unwrap()).collect();
    {
        let mut a: Vec<&mut _> = ya.iter_mut().collect();
        refs.forward_group(&st, &members, &dxs[0], &mut a).unwrap();
        let mut b: Vec<&mut _> = yb.iter_mut().collect();
        tst.forward_group(&st, &members, &dxs[0], &mut b).unwrap();
    }
    let e_g = (0..nsets).map(|t| rel(&st.clone_dtoh(&yb[t]).unwrap(), &st.clone_dtoh(&ya[t]).unwrap())).fold(0.0, f64::max);
    let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
    let mut xb = st.alloc_zeros::<C32>(nv).unwrap();
    let yr: Vec<&_> = ya.iter().collect();
    refs.adjoint_group(&st, &members, &yr, &mut xa).unwrap();
    tst.adjoint_group(&st, &members, &yr, &mut xb).unwrap();
    let e_ag = rel(&st.clone_dtoh(&xb).unwrap(), &st.clone_dtoh(&xa).unwrap());
    let mut ra: Vec<_> = (0..nsets).map(|t| st.alloc_zeros::<C32>(nc * ms[t]).unwrap()).collect();
    let mut rb: Vec<_> = (0..nsets).map(|t| st.alloc_zeros::<C32>(nc * ms[t]).unwrap()).collect();
    {
        let mut a: Vec<&mut _> = ra.iter_mut().collect();
        refs.forward_residual_group(&st, &members, &dxs[1], &yr, &mut a).unwrap();
        let mut b: Vec<&mut _> = rb.iter_mut().collect();
        tst.forward_residual_group(&st, &members, &dxs[1], &yr, &mut b).unwrap();
    }
    let e_rg = (0..nsets).map(|t| rel(&st.clone_dtoh(&rb[t]).unwrap(), &st.clone_dtoh(&ra[t]).unwrap())).fold(0.0, f64::max);
    eprintln!("groups: forward {e_g:.2e}, adjoint {e_ag:.2e}, residual {e_rg:.2e}; uploads {}", tst.uploads);
    worst = worst.max(e_g).max(e_ag).max(e_rg);
    // the tables differ: a wrong table must show (guards against a test that compares the same thing)
    let mut y0 = st.alloc_zeros::<C32>(nc * ms[0]).unwrap();
    let mut y1 = st.alloc_zeros::<C32>(nc * ms[0]).unwrap();
    refs.forward(&st, 0, &dxs[0], &mut y0).unwrap();
    refs.seg_of_set[0] = 2;
    refs.forward(&st, 0, &dxs[0], &mut y1).unwrap();
    let e_wrong = rel(&st.clone_dtoh(&y1).unwrap(), &st.clone_dtoh(&y0).unwrap());
    eprintln!("frame 2's table on set 0 instead of frame 0's: {e_wrong:.2e} (must be large)");
    assert!(e_wrong > 1e-2, "the per-frame tables are too similar for the test to discriminate");
    // fp32 floor: the control engine's worst difference from the reference. A wrapper fault (stale or wrong table) is
    // of the size of e_wrong, orders above it.
    eprintln!("worst PerSetSegs difference {worst:.2e}; control-engine floor {floor:.2e}");
    assert!(worst < 1e-5 && worst < 10.0 * floor.max(1e-7), "PerSetSegs differs from the resident tables: {worst:.2e} (floor {floor:.2e})");
}
