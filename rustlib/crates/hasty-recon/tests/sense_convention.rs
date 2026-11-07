//! SenseOp geometry convention: image index q = i + n0 (j + n1 k) (x fastest) sits at ((i − n0/2), (j − n1/2),
//! (k − n2/2)) voxels, coords[a] = k_a·Δ (radians per voxel), forward y = Σ x e^{−i coords·(index − n/2)}.
use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;

#[test]
fn forward_of_a_delta_is_a_plane_wave_with_x_fastest_indexing() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let dims = [20usize, 16, 12];
    let nv = dims[0] * dims[1] * dims[2];
    let m = 200;
    let mut s = 3u64;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
    };
    let c: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| 6.0 * u()).collect());
    let smaps = vec![C32::new(1.0, 0.0); nv];
    let mut op = SenseOp::new(&dev, &st, kern, dims, [&c[0], &c[1], &c[2]], &smaps, 1, vec![], 1e-6).unwrap();
    let (i0, j0, k0) = (13usize, 4usize, 9usize);
    let mut x = vec![C32::new(0.0, 0.0); nv];
    x[i0 + dims[0] * (j0 + dims[1] * k0)] = C32::new(1.0, 0.0);
    let dx = st.clone_htod(&x).unwrap();
    let mut dy = st.alloc_zeros::<C32>(m).unwrap();
    op.forward(&st, &dx, &mut dy).unwrap();
    let y = st.clone_dtoh(&dy).unwrap();
    let pos = [i0 as f64 - (dims[0] / 2) as f64, j0 as f64 - (dims[1] / 2) as f64, k0 as f64 - (dims[2] / 2) as f64];
    let mut err = 0f64;
    for n in 0..m {
        let ph = (0..3).map(|a| c[a][n] as f64 * pos[a]).sum::<f64>();
        let e = num_complex::Complex64::from_polar(1.0, -ph);
        err = err.max(((y[n].re as f64 - e.re).powi(2) + (y[n].im as f64 - e.im).powi(2)).sqrt());
    }
    assert!(err < 1e-3, "max |y − e^(−i k·x)| = {err}");
}
