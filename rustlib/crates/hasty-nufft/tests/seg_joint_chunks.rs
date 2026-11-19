//! Joint operator changes of research/coil_segment_compression.md §12 (GPU), each against the previous path to f32
//! rounding:
//!   * Φ generation as a cuBLAS CGEMM (default) vs the `joint_phi` kernel;
//!   * sample chunks (`forward_chunks` / `adjoint_chunks`: g buffer of one chunk) vs the whole set, static order 0,
//!     static order 3 (multi kernels) and 2-D knots order 3; adjointness of the chunked pair;
//!   * inner batches with the new default shapes: a prime batch (13, ragged groups) and a blocked batch (24 channels,
//!     nseg = 1: acb / fcb = 4) vs batches of 8 (no blocking).
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_joint_basis, seg_time_grid, set_thread_cb_cap, JointBasis, JointBuildOpts, JointSegNufft, SegNufft, SegNufftOpts, SegPoints, NHI};
use hasty_testkit::{Rng, C64};
use num_complex::Complex64;

const PI: f64 = std::f64::consts::PI;
const TE: f64 = 1.6e-3;
const TRO: f64 = 8e-3;

struct Setup {
    dims: [usize; 3],
    nc: usize,
    nv: usize,
    smaps: Vec<C32>,
    hz: Vec<f64>,
    r2: Vec<f64>,
    k: [Vec<f32>; 3],
    tau: Vec<f32>,
    tau_adc: Vec<f64>,
    na: usize,
    nro: usize,
}

fn setup() -> Setup {
    let dims = [24usize, 20, 16];
    let res = 3e-3;
    let nv = dims.iter().product::<usize>();
    let nc = 6;
    let pos: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [(i as f64 - (dims[0] / 2) as f64) * res, (j as f64 - (dims[1] / 2) as f64) * res, (k as f64 - (dims[2] / 2) as f64) * res]
    }).collect();
    let mut smaps = Vec::with_capacity(nc * nv);
    for c in 0..nc {
        let ang = 2.0 * PI * c as f64 / nc as f64;
        let ctr = [0.05 * ang.cos(), 0.05 * ang.sin(), 0.01 * (c % 2) as f64];
        for p in &pos {
            let d2 = (p[0] - ctr[0]).powi(2) + (p[1] - ctr[1]).powi(2) + 0.5 * (p[2] - ctr[2]).powi(2);
            let z = C64::from_polar((-d2 / 0.003).exp() + 0.15, 30.0 * (c as f64 + 1.0) * p[0] - 20.0 * p[1] + 8.0 * c as f64 * p[2]);
            smaps.push(C32::new(z.re as f32, z.im as f32));
        }
    }
    let ext = [0, 1, 2].map(|a| dims[a] as f64 * res);
    let hz: Vec<f64> = pos.iter().map(|p| 120.0 * (2.0 * PI * p[0] / ext[0]).sin() * (2.0 * PI * p[1] / ext[1]).cos() + 150.0 * ((p[0] - 0.006) / 0.004).tanh() + 40.0 * 2.0 * p[2] / ext[2]).collect();
    let r2: Vec<f64> = pos.iter().map(|p| 20.0 + 30.0 * (p[1] / ext[1] + 0.5)).collect();
    let (nro, na) = (12usize, 256usize);
    let mut r = Rng::new(77);
    let mut k: [Vec<f32>; 3] = Default::default();
    let mut tau = Vec::with_capacity(nro * na);
    let tau_adc: Vec<f64> = (0..na).map(|n| (TE + TRO * n as f64 / (na - 1) as f64) as f32 as f64).collect();
    for _ in 0..nro {
        let kz = 0.97 * PI * r.uniform();
        let th = PI * r.uniform();
        for n in 0..na {
            let s = n as f64 / (na - 1) as f64;
            let (rad, ang) = (0.97 * PI * s.sqrt(), th + 2.0 * PI * 7.0 * s.sqrt());
            k[0].push((rad * ang.cos()) as f32);
            k[1].push((rad * ang.sin()) as f32);
            k[2].push(kz as f32);
            tau.push(tau_adc[n] as f32);
        }
    }
    Setup { dims, nc, nv, smaps, hz, r2, k, tau, tau_adc, na, nro }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|y| (y.re as f64).powi(2) + (y.im as f64).powi(2)).sum();
    (n / d.max(1e-30)).sqrt()
}
fn dot(a: &[C32], b: &[C32]) -> Complex64 {
    a.iter().zip(b).map(|(x, y)| Complex64::new(x.re as f64, x.im as f64).conj() * Complex64::new(y.re as f64, y.im as f64)).sum()
}

/// per-sample tables of samples j0 .. j1 (planar arrays)
struct Tabs {
    k: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    uh: Vec<f32>,
    tc: Vec<f32>,
    ri: Vec<u32>,
    rv: Vec<f32>,
}
impl Tabs {
    fn slice(&self, j0: usize, j1: usize) -> Tabs {
        let m = self.tau.len();
        let pl = |v: &Vec<f32>, n: usize| (0..n).flat_map(|q| v[q * m + j0..q * m + j1].iter().copied()).collect::<Vec<f32>>();
        Tabs { k: [0, 1, 2].map(|a| self.k[a][j0..j1].to_vec()), tau: self.tau[j0..j1].to_vec(), phi0: self.phi0[j0..j1].to_vec(), dc: pl(&self.dc, 6), uh: pl(&self.uh, 4), tc: self.tc[j0..j1].to_vec(), ri: self.ri[j0..j1].to_vec(), rv: self.rv[j0..j1].to_vec() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Eng {
    Static0,
    Static3,
    Knot2d3,
}

fn tables(e: &mut SegNufft, st: &std::sync::Arc<cudarc::driver::CudaStream>, eng: Eng, t: &Tabs, hh: &[f64]) -> SegPoints {
    let c = [&t.k[0][..], &t.k[1][..], &t.k[2][..]];
    match eng {
        Eng::Static0 => e.points(st, c, &t.tau, &t.phi0, &t.dc, None).unwrap(),
        Eng::Static3 => e.points_hi(st, c, &t.tau, &t.phi0, &t.dc, &t.uh, hh, None).unwrap(),
        Eng::Knot2d3 => e.points2d_hi(st, c, &t.tau, &t.phi0, &t.dc, &t.tc, &t.ri, &t.rv, &t.uh, hh).unwrap(),
    }
}

fn copy(st: &std::sync::Arc<cudarc::driver::CudaStream>, b: &JointBasis) -> JointBasis {
    JointBasis { nc: b.nc, nv: b.nv, k: b.k, lam: b.lam.clone(), u: b.u.clone(), ncol: b.ncol, ts: b.ts.clone(), stw: b.stw.clone(), v: b.v.clone(), time_ev: b.time_ev.clone(), tab_times: b.tab_times.clone(), vt: b.vt.clone(), h: st.clone_dtod(&b.h).unwrap(), timing: vec![] }
}

fn inner(dev: &Device, s: &Setup, eng: Eng, cbj: usize) -> SegNufft {
    set_thread_cb_cap(Some(cbj.max(16)));
    let e = match eng {
        Eng::Static0 => SegNufft::new(dev, s.dims, [3e-3; 3], cbj, SegNufftOpts { sigma: 1.25, width: 10, nseg: 1, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap(),
        Eng::Static3 => SegNufft::new(dev, s.dims, [3e-3; 3], cbj, SegNufftOpts { sigma: 1.25, width: 10, nseg: 1, order: 3, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap(),
        Eng::Knot2d3 => SegNufft::new_temporal2d(dev, s.dims, [3e-3; 3], cbj, SegNufftOpts { sigma: 1.25, width: 10, nseg: 1, order: 3, frames: 4, bspline: 3, max_grid_bytes: 1 << 30 }, 2, 1).unwrap(),
    };
    set_thread_cb_cap(None);
    e
}

#[test]
fn joint_phi_gemm_chunks_and_inner_shapes() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let s = setup();
    let (nv, m, nc) = (s.nv, s.tau.len(), s.nc);
    let gf = |q: usize, t: f64| Complex64::from_polar((-s.r2[q] * (t - TE)).exp(), -2.0 * PI * s.hz[q] * t);
    let fit: Vec<usize> = (0..nv).collect();
    let w = vec![1.0f64; nv];
    let (ts, tw) = seg_time_grid(&s.tau_adc, &vec![1.0; s.na], 96);
    let basis = build_joint_basis(&dev, &st, nv, nc, &s.smaps, &fit, &w, &ts, &tw, &s.tau_adc, &gf, &JointBuildOpts::default()).unwrap();
    let r = basis.count_for(1e-5);
    let mut rng = Rng::new(9);
    let tabs = Tabs {
        k: s.k.clone(),
        tau: s.tau.clone(),
        phi0: (0..m).map(|_| (0.3 * rng.uniform()) as f32).collect(),
        dc: (0..6 * m).map(|_| (rng.uniform() * 5.0) as f32).collect(),
        uh: (0..4 * m).map(|_| rng.uniform() as f32).collect(),
        tc: (0..m).map(|j| ((j as f64 * 0.37) % 4.0) as f32).collect(),
        ri: vec![0u32; m],
        rv: (0..m).map(|j| ((j as f64 * 0.13) % 1.0) as f32).collect(),
    };
    let hh: Vec<f64> = (0..4 * NHI).map(|_| rng.uniform() * 20.0).collect();
    // chunks of whole readouts: 5 + 4 + 3 of the 12
    let bounds = [0usize, 5, 9, s.nro];
    for eng in [Eng::Static0, Eng::Static3, Eng::Knot2d3] {
        let nimg = if eng == Eng::Knot2d3 { 8 } else { 1 };
        let mut rr = Rng::new(3);
        let xs: Vec<C32> = rr.complex_vec(nv * nimg).iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect();
        let ys: Vec<C32> = rr.complex_vec(nc * m).iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect();
        let (xd, yd) = (st.clone_htod(&xs).unwrap(), st.clone_htod(&ys).unwrap());
        let run = |op: &mut JointSegNufft, pts: &SegPoints| -> (Vec<C32>, Vec<C32>) {
            let s1 = op.segment1(&st, None).unwrap();
            let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
            let mut x = st.alloc_zeros::<C32>(nv * nimg).unwrap();
            op.forward(&st, &s1, pts, &xd, &mut y).unwrap();
            op.adjoint(&st, &s1, pts, &yd, &mut x).unwrap();
            (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&x).unwrap())
        };
        // reference: batches of 8, Φ by the kernel
        let mut op = JointSegNufft::new(&dev, &st, inner(&dev, &s, eng, 8), copy(&st, &basis), &s.smaps, r).unwrap();
        op.phi_gemm = false;
        let pts = tables(&mut op.inner, &st, eng, &tabs, &hh);
        let (y0, x0) = run(&mut op, &pts);
        // Φ by CGEMM
        op.phi_gemm = true;
        let (y1, x1) = run(&mut op, &pts);
        eprintln!("{eng:?} R = {r}: Φ CGEMM vs kernel: forward {:.1e}, adjoint {:.1e}", rel(&y1, &y0), rel(&x1, &x0));
        assert!(rel(&y1, &y0) < 1e-5 && rel(&x1, &x0) < 1e-5);
        // sample chunks
        let cps: Vec<SegPoints> = bounds.windows(2).map(|b| tables(&mut op.inner, &st, eng, &tabs.slice(b[0] * s.na, b[1] * s.na), &hh)).collect();
        let refs: Vec<&SegPoints> = cps.iter().collect();
        let offs: Vec<usize> = bounds[..3].iter().map(|&b| b * s.na).collect();
        let s1 = op.segment1(&st, None).unwrap();
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut x = st.alloc_zeros::<C32>(nv * nimg).unwrap();
        op.forward_chunks(&st, &s1, &refs, &offs, m, &xd, &mut y).unwrap();
        op.adjoint_chunks(&st, &s1, &refs, &offs, m, &yd, &mut x).unwrap();
        let (y2, x2) = (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&x).unwrap());
        let e = (dot(&ys, &y2) - dot(&x2, &xs)).norm() / (dot(&ys, &ys).re * dot(&y2, &y2).re).sqrt();
        eprintln!("{eng:?}: 3 chunks vs whole set: forward {:.1e}, adjoint {:.1e}; adjointness {e:.1e}", rel(&y2, &y1), rel(&x2, &x1));
        assert!(rel(&y2, &y1) < 3e-5 && rel(&x2, &x1) < 6e-5 && e < 2e-6);
        drop(op);
        // inner batches with the new default shapes: prime 13 (ragged groups), 24 (nseg = 1 blocking)
        for cbj in [13usize, 24] {
            let mut op = JointSegNufft::new(&dev, &st, inner(&dev, &s, eng, cbj), copy(&st, &basis), &s.smaps, r).unwrap();
            let pts = tables(&mut op.inner, &st, eng, &tabs, &hh);
            let (y3, x3) = run(&mut op, &pts);
            eprintln!("{eng:?}: batch {cbj} (tiles fwd {:?}, adj {:?}, multi {:?}) vs batch 8: forward {:.1e}, adjoint {:.1e}", op.inner.tiles_fwd, op.inner.tiles_adj, op.inner.multi_shapes(1), rel(&y3, &y1), rel(&x3, &x1));
            assert!(rel(&y3, &y1) < 3e-5 && rel(&x3, &x1) < 6e-5);
        }
    }
}
