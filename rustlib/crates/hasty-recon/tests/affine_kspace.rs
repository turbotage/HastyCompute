//! `motion::affine_samples` against image-domain motion on the grid: a translation by whole voxels and a 90° rotation
//! in x–y (both exact permutations of the grid) must give the same data as transforming the image.
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};

fn run(op: &mut SegNufft, st: &std::sync::Arc<cudarc::driver::CudaStream>, coords: &[Vec<f32>; 3], phi0: &[f32], x: &[C32]) -> Vec<C32> {
    let m = phi0.len();
    let nv = op.nv();
    let fit: Vec<usize> = (0..nv).collect();
    let (sp, at, t0, dt) = build_segments(nv, 1, 1e-3, 2e-3, 8, &fit, &|_, _| 0.0);
    let seg = op.segments(st, &sp, &at, t0, dt).unwrap();
    let pts = op.points(st, [&coords[0], &coords[1], &coords[2]], &vec![1.5e-3f32; m], phi0, &[], None).unwrap();
    let dx = st.clone_htod(x).unwrap();
    let mut y = st.alloc_zeros::<C32>(m).unwrap();
    op.forward(st, &seg, &pts, &dx, &mut y).unwrap();
    st.clone_dtoh(&y).unwrap()
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|q| (q.re as f64).powi(2) + (q.im as f64).powi(2)).sum();
    (n / d).sqrt()
}

#[test]
fn affine_kspace_matches_image_motion() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let n = [16usize, 16, 8];
    let nv = n[0] * n[1] * n[2];
    let vox = [2e-3, 2e-3, 3e-3];
    let mut op = SegNufft::new(&dev, n, vox, 1, SegNufftOpts { sigma: 2.0, width: 8, nseg: 1, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 26 }).unwrap();
    op.set_maps(&st, &vec![C32::new(1.0, 0.0); nv]).unwrap();
    let mut s = 99u64;
    let mut u = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5 };
    // smooth-ish image away from the edges
    let x: Vec<C32> = (0..nv).map(|v| { let (i, j, k) = (v % 16, (v / 16) % 16, v / 256); let w = if (3..13).contains(&i) && (3..13).contains(&j) && (2..6).contains(&k) { 1.0 } else { 0.0 }; C32::new((w * (1.0 + 0.3 * u())) as f32, (w * 0.3 * u()) as f32) }).collect();
    let m = 3000;
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (u() * 6.0) as f32).collect());
    let s1 = vec![1.0f64; m];
    // translation by (2, −1, 1) voxels: x_s(p) = x(p − d)
    let d = [2i64, -1, 1];
    let xt: Vec<C32> = (0..nv).map(|v| { let (i, j, k) = ((v % 16) as i64, ((v / 16) % 16) as i64, (v / 256) as i64); let q = (((k - d[2]).rem_euclid(8) * 16 + (j - d[1]).rem_euclid(16)) * 16 + (i - d[0]).rem_euclid(16)) as usize; x[q] }).collect();
    let (mut c1, mut p1) = (coords.clone(), vec![0f32; m]);
    let t1 = [d[0] as f64 * vox[0], d[1] as f64 * vox[1], d[2] as f64 * vox[2]];
    let det = hasty_recon::motion::affine_samples(&mut c1, &mut p1, &s1, &[[0.0; 3]; 3], &t1, vox);
    assert!(det.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    let e_t = rel(&run(&mut op, &st, &c1, &p1, &x), &run(&mut op, &st, &coords, &vec![0f32; m], &xt));
    // rotation by +90° about z (x → y): p = R q, R = [[0, −1, 0], [1, 0, 0], [0, 0, 1]] (square x–y voxels), A₁ = R − I
    let xr: Vec<C32> = (0..nv).map(|v| { let (i, j, k) = ((v % 16) as i64, ((v / 16) % 16) as i64, (v / 256) as i64); let (pi, pj) = (i - 8, j - 8); let (qi, qj) = (pj, -pi); let q = ((k * 16 + (qj + 8).rem_euclid(16)) * 16 + (qi + 8).rem_euclid(16)) as usize; x[q] }).collect();
    let a1 = [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]];
    let (mut c2, mut p2) = (coords.clone(), vec![0f32; m]);
    hasty_recon::motion::affine_samples(&mut c2, &mut p2, &s1, &a1, &[0.0; 3], vox);
    let e_r = rel(&run(&mut op, &st, &c2, &p2, &x), &run(&mut op, &st, &coords, &vec![0f32; m], &xr));
    eprintln!("affine k-space vs image motion: translation {e_t:.1e}, 90° rotation {e_r:.1e}");
    assert!(e_t < 1e-4 && e_r < 1e-4, "translation {e_t:.1e}, rotation {e_r:.1e}");
}
