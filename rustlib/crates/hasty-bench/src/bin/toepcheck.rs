//! Toeplitz kernel (hasty_toeplitz::build_kernels + ConvPlan) vs NUFFT normal operator, single coil with unit map.
//!   toepcheck <sim_dir>
use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_fft::{ConvPlan, ConvSpec, KernelFormat};
use hasty_recon::data::SimData;
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_toeplitz::{build_kernels, plain, KernelBuildOpts, Samples};
use num_complex::Complex64;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let sd = SimData::load(&dir).unwrap();
    let ident: Vec<Vec<f64>> = (0..sd.nc).map(|i| (0..sd.nc).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let ros: Vec<usize> = sd.readouts_of(0, None).into_iter().take(100).collect();
    let set = sd.sample_set(&ros, &ident);
    let m = set.tau.len();
    let dims = sd.dims;
    let nv = sd.nv();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let ones = vec![C32::new(1.0, 0.0); nv];
    let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &ones, 1, vec![], 1e-6).unwrap();
    // smooth-ish random test image
    let mut r = 99u64;
    let x: Vec<C32> = (0..nv).map(|_| { r ^= r << 13; r ^= r >> 7; r ^= r << 17; C32::new(((r >> 40) as f32) / 16777216.0 - 0.5, 0.0) }).collect();
    let dx = st.clone_htod(&x).unwrap();
    let mut nx = st.alloc_zeros::<C32>(nv).unwrap();
    op.normal(&st, &dx, &mut nx).unwrap();
    let sdims = [dims[2], dims[1], dims[0]];
    let dc: Vec<_> = set.coords.iter().map(|c| st.clone_htod(c).unwrap()).collect();
    let wt = st.clone_htod(&vec![1f32; m]).unwrap();
    let zero = st.alloc_zeros::<f32>(m).unwrap();
    let extra: Vec<cudarc::driver::CudaSlice<C32>> = vec![];
    let samples = Samples::<f32> { m, coords: [&dc[0], &dc[1], &dc[2]], wt: &wt, w: [&zero, &zero, &zero, &zero], extra: &extra };
    let k = build_kernels(&dev, &st, sdims, &samples, &plain().kernels, &KernelBuildOpts { tol: 1e-6, sigma: 2.0 }).unwrap().remove(0);
    let conv = ConvPlan::<f32>::new(&dev, &ConvSpec { dims: sdims.to_vec(), dims_pad: sdims.iter().map(|d| 2 * d).collect(), pre: vec![], post: vec![], accumulate: false, kernel_format: KernelFormat::Complex }).unwrap();
    let mut scratch = st.alloc_zeros::<C32>(8 * nv).unwrap();
    let mut tx = st.alloc_zeros::<C32>(nv).unwrap();
    conv.execute(&st, &dx, &k, 1.0, &mut scratch, &mut tx, &[], &[]).unwrap();
    let a = st.clone_dtoh(&nx).unwrap();
    let b = st.clone_dtoh(&tx).unwrap();
    let ab: Complex64 = a.iter().zip(&b).map(|(p, q)| Complex64::new(q.re as f64, -q.im as f64) * Complex64::new(p.re as f64, p.im as f64)).sum();
    let bb: f64 = b.iter().map(|q| (q.re as f64).powi(2) + (q.im as f64).powi(2)).sum();
    let s = ab / bb;
    let err: f64 = a.iter().zip(&b).map(|(p, q)| (Complex64::new(p.re as f64, p.im as f64) - s * Complex64::new(q.re as f64, q.im as f64)).norm_sqr()).sum::<f64>();
    let an: f64 = a.iter().map(|p| (p.re as f64).powi(2) + (p.im as f64).powi(2)).sum();
    println!("Toeplitz vs NUFFT normal op: best scale {s:.4e}, rel err {:.3e}", (err / an).sqrt());
    let kh = st.clone_dtoh(&k).unwrap();
    let re: Vec<f32> = kh.iter().map(|c| c.re).collect();
    let mut sorted = re.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max_im = kh.iter().map(|c| c.im.abs()).fold(0f32, f32::max);
    // Fourier-mode test of the circulant spectrum
    let pre = hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &ones, 1, 1e-6).unwrap();
    println!("preconditioner dynamic range {:.2e}", pre.dyn_range);
    for f in [[0usize, 0, 0], [1, 0, 0], [5, 3, 1], [20, 10, 2], [45, 40, 5], [90, 90, 14], [80, 20, 10]] {
        let xm: Vec<C32> = (0..nv).map(|q| {
            let (i, j, kk) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
            let ph = 2.0 * std::f64::consts::PI * (f[0] as f64 * i as f64 / dims[0] as f64 + f[1] as f64 * j as f64 / dims[1] as f64 + f[2] as f64 * kk as f64 / dims[2] as f64);
            C32::new(ph.cos() as f32, ph.sin() as f32)
        }).collect();
        let dxm = st.clone_htod(&xm).unwrap();
        let mut nxm = st.alloc_zeros::<C32>(nv).unwrap();
        op.normal(&st, &dxm, &mut nxm).unwrap();
        let h = st.clone_dtoh(&nxm).unwrap();
        let ray: Complex64 = h.iter().zip(&xm).map(|(a, b)| Complex64::new(b.re as f64, -b.im as f64) * Complex64::new(a.re as f64, a.im as f64)).sum::<Complex64>() / nv as f64;
        // λ index for frequency f: FFT order on the N grid; forward FFT convention e^{-i}: mode +f sits at index f or N−f
        let idx = |a: [usize; 3]| a[0] + dims[0] * (a[1] + dims[1] * a[2]);
        let neg = [0, 1, 2].map(|a| (dims[a] - f[a]) % dims[a]);
        println!("mode {f:?}: Rayleigh <x,Nx>/|x|² = {:.4e}, λ[+f] = {:.4e}, λ[−f] = {:.4e}", ray.re, pre.lambda[idx(f)], pre.lambda[idx(neg)]);
    }
    println!("K̂: min {:.3e}, 1% {:.3e}, median {:.3e}, max {:.3e}, max|Im| {:.3e}", sorted[0], sorted[sorted.len() / 100], sorted[sorted.len() / 2], sorted[sorted.len() - 1], max_im);
}
