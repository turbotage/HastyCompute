use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use num_complex::Complex64;

struct R(u64);
impl R {
    fn u(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
    }
}

fn check(seg_terms: usize) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let dims = [24usize, 20, 12];
    let nv = dims.iter().product::<usize>();
    let (nc, m) = (3usize, 5000usize);
    let mut r = R(77);
    let coords: Vec<Vec<f32>> = (0..3).map(|_| (0..m).map(|_| r.u() * 6.0).collect()).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|_| C32::new(r.u(), r.u())).collect();
    let seg: Vec<(Vec<C32>, Vec<C32>)> = (0..seg_terms).map(|_| ((0..m).map(|_| C32::new(r.u(), r.u())).collect(), (0..nv).map(|_| C32::new(r.u(), r.u())).collect())).collect();
    let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, seg, 1e-6).unwrap();
    let x: Vec<C32> = (0..nv).map(|_| C32::new(r.u(), r.u())).collect();
    let y: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u(), r.u())).collect();
    let dx = st.clone_htod(&x).unwrap();
    let dy = st.clone_htod(&y).unwrap();
    let mut ex = st.alloc_zeros::<C32>(nc * m).unwrap();
    let mut ehy = st.alloc_zeros::<C32>(nv).unwrap();
    op.forward(&st, &dx, &mut ex).unwrap();
    op.adjoint(&st, &dy, &mut ehy).unwrap();
    let a = kern.dot(&st, &dy, &ex).unwrap(); // <y, E x>
    let b = kern.dot(&st, &ehy, &dx).unwrap(); // <E^H y, x>
    eprintln!("L={seg_terms}: <y,Ex> = {a}, <E^H y,x> = {b}, rel diff {:.2e}", (a - b).norm() / a.norm());
    assert!((a - b).norm() < 1e-4 * a.norm());
    let mut nx = st.alloc_zeros::<C32>(nv).unwrap();
    op.normal(&st, &dx, &mut nx).unwrap();
    let q: Complex64 = kern.dot(&st, &dx, &nx).unwrap();
    assert!(q.re > 0.0 && q.im.abs() < 1e-4 * q.re, "<x, N x> = {q}");
}

#[test]
fn sense_operator_adjoint_plain() {
    check(0);
}

#[test]
fn sense_operator_adjoint_segmented() {
    check(3);
}

#[test]
fn nufft_type1_type2_adjoint() {
    use hasty_nufft::{NufftOpts, NufftPlan};
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Kernels::new(&dev).unwrap();
    let dims = [24usize, 20, 12];
    let nv = dims.iter().product::<usize>();
    let m = 5000usize;
    let mut r = R(5);
    for nt in [1usize, 3] {
        let coords: Vec<Vec<f32>> = (0..3).map(|_| (0..m).map(|_| r.u() * 6.0).collect()).collect();
        let dc: Vec<_> = coords.iter().map(|c| st.clone_htod(c).unwrap()).collect();
        let mut p = NufftPlan::<f32>::new(&dev, &dims, NufftOpts { tol: 1e-6, ntransf: nt, ..Default::default() }).unwrap();
        p.set_points(&st, &[&dc[0], &dc[1], &dc[2]], m).unwrap();
        let x: Vec<C32> = (0..nt * nv).map(|_| C32::new(r.u(), r.u())).collect();
        let y: Vec<C32> = (0..nt * m).map(|_| C32::new(r.u(), r.u())).collect();
        let dx = st.clone_htod(&x).unwrap();
        let dy = st.clone_htod(&y).unwrap();
        let mut fx = st.alloc_zeros::<C32>(nt * m).unwrap();
        let mut fy = st.alloc_zeros::<C32>(nt * nv).unwrap();
        p.type2(&st, &dx, &mut fx, -1).unwrap();
        p.type1(&st, &dy, &mut fy, 1).unwrap();
        let a = kern.dot(&st, &dy, &fx).unwrap();
        let b = kern.dot(&st, &fy, &dx).unwrap();
        eprintln!("nufft ntransf={nt}: <y,Fx> = {a}, <F^H y, x> = {b}");
    }
}
