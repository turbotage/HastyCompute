//! ESPIRiT from non-Cartesian data (GPU): synthetic smooth coil maps × a textured ellipsoid, sampled at centre-dense
//! random 3D k-space points by the type-2 NUFFT, calibration by `calib_from_noncart` (low-resolution CG, no density
//! compensation), maps by `espirit`; inside the object the estimated maps match the true ones up to a per-voxel phase.

use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_nufft::{NufftOpts, NufftPlan};
use hasty_recon::espirit::{calib_from_noncart, espirit, EspiritOpts};
use hasty_recon::gpu::Kernels;
use num_complex::Complex32;

fn nvox(d: [usize; 3]) -> usize {
    d[0] * d[1] * d[2]
}

fn pos(q: usize, dims: [usize; 3]) -> [f32; 3] {
    let p = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
    [0, 1, 2].map(|a| (p[a] as f32 - (dims[a] / 2) as f32) / dims[a] as f32)
}

fn coil_maps(dims: [usize; 3], nc: usize) -> Vec<Complex32> {
    let nv = nvox(dims);
    let mut s = vec![Complex32::new(0.0, 0.0); nc * nv];
    for c in 0..nc {
        let th = 2.0 * std::f32::consts::PI * c as f32 / nc as f32;
        let (cx, cy, cz) = (0.45 * th.cos(), 0.45 * th.sin(), if c % 2 == 0 { 0.2 } else { -0.2 });
        for q in 0..nv {
            let u = pos(q, dims);
            let r2 = (u[0] - cx).powi(2) + (u[1] - cy).powi(2) + (u[2] - cz).powi(2);
            s[c * nv + q] = Complex32::from_polar((-r2 / 0.18).exp(), 1.5 * (u[0] * th.sin() - u[1] * th.cos()) + 0.3 * c as f32);
        }
    }
    s
}

fn inside(u: [f32; 3]) -> f32 {
    (u[0] / 0.36).powi(2) + (u[1] / 0.32).powi(2) + (u[2] / 0.34).powi(2)
}

#[test]
fn espirit_from_noncartesian_samples() {
    let (dims, nc) = ([40usize, 36, 28], 8usize);
    let nv = nvox(dims);
    let s = coil_maps(dims, nc);
    let rho: Vec<f32> = (0..nv).map(|q| { let u = pos(q, dims); if inside(u) < 1.0 { 1.0 + 0.3 * (9.0 * u[0]).sin() * (7.0 * u[1]).cos() + 0.2 * (5.0 * u[2]).sin() } else { 0.0 } }).collect();
    // centre-dense random samples in the ball |k| < π (rad/voxel)
    let m = 200_000;
    let mut seed = 12345u64;
    let mut u = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut coords: [Vec<f32>; 3] = [vec![], vec![], vec![]];
    for _ in 0..m {
        let (z, ph, r) = (2.0 * u() - 1.0, 2.0 * std::f64::consts::PI * u(), std::f64::consts::PI * u().powf(1.5) * 0.999);
        let st = (1.0 - z * z).sqrt();
        let k = [r * st * ph.cos(), r * st * ph.sin(), r * z];
        for a in 0..3 {
            coords[a].push(k[a] as f32);
        }
    }
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let mut plan = NufftPlan::<f32>::new(&dev, &dims, NufftOpts { tol: 1e-6, sigma: 2.0, ntransf: nc, ..Default::default() }).unwrap();
    let dc: Vec<_> = coords.iter().map(|c| st.clone_htod(c).unwrap()).collect();
    plan.set_points(&st, &[&dc[0], &dc[1], &dc[2]], m).unwrap();
    let imgs: Vec<C32> = (0..nc * nv).map(|i| { let z = s[i] * rho[i % nv]; C32::new(z.re, z.im) }).collect();
    let di = st.clone_htod(&imgs).unwrap();
    let mut dy = st.alloc_zeros::<C32>(nc * m).unwrap();
    plan.type2(&st, &di, &mut dy, -1).unwrap();
    let y = st.clone_dtoh(&dy).unwrap();
    let cal = [20usize, 20, 16];
    let (calib, used) = calib_from_noncart(&dev, &st, &kern, [&coords[0], &coords[1], &coords[2]], &y, nc, dims, cal, 40, 1e-4).unwrap();
    let em = espirit(&calib, cal, nc, dims, &EspiritOpts { kernel: 5, ..Default::default() });
    let (mut err, mut cnt) = (0.0f64, 0usize);
    for q in 0..nv {
        if inside(pos(q, dims)) < 0.6 {
            let n: f32 = (0..nc).map(|c| s[c * nv + q].norm_sqr()).sum::<f32>().sqrt();
            let ip: Complex32 = (0..nc).map(|c| { let e = em.maps[c * nv + q]; Complex32::new(e.re, -e.im) * s[c * nv + q] / n }).sum();
            err += 1.0 - ip.norm() as f64;
            cnt += 1;
        }
    }
    let err = err / cnt as f64;
    eprintln!("{used} of {m} samples in the calibration box, {} kernels, mean 1 − |⟨ŝ, s⟩| inside {err:.2e}", em.n_kernels);
    assert!(err < 2e-3, "mean 1 − |⟨ŝ, s⟩| = {err:.2e}");
}
