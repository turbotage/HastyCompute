//! Water and fat as separate images (`SegNufft::forward_wf` / `adjoint_wf`) against the fat-fraction-map path
//! (`forward_fat` / `adjoint_fat`): with W = (1 − f) x and Fa = f x the forwards agree, and the adjoints agree after
//! (1 − f) g_W + f g_Fa. Static, and temporal (cubic frames) with the water + fat pair on a knot-major engine against
//! per-frame fat maps on the interval-major multi path.
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|y| (y.re as f64).powi(2) + (y.im as f64).powi(2)).sum();
    (n / d.max(1e-30)).sqrt()
}

fn case(frames: usize, bspline: u32, knot: bool) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [20usize, 18, 14];
    let nv: usize = dims.iter().product();
    let (nc, nseg, m) = (3, 5, 4000);
    let mut r = R(99 + frames as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * frames as f64) as f32).collect();
    let x: Vec<C32> = (0..frames * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let ff: Vec<f32> = (0..frames * nv).map(|_| (r.u() + 0.5) as f32).collect();
    let yd: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let cl: Vec<C32> = (0..nseg).map(|l| { let a = 1.3 * l as f32 + 0.4; C32::new(0.9 * a.cos(), 0.9 * a.sin()) }).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    let opts = || SegNufftOpts { sigma: 1.25, width: 8, nseg, order: 2, frames, bspline, max_grid_bytes: 1 << 28 };
    let tt = (frames > 1).then_some(&t[..]);
    // reference: fat-fraction maps on the interval-major engine
    let (yref, xref) = {
        let mut op = SegNufft::new(&dev, dims, [0.004; 3], nc, opts()).unwrap();
        op.set_maps(&st, &smaps).unwrap();
        let seg = op.segments_fat(&st, &sp, &at, &cl, t0, dt).unwrap();
        let maps = op.fat_maps(&st, &ff).unwrap();
        let pts = op.points(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, tt).unwrap();
        let (dx, dy) = (st.clone_htod(&x).unwrap(), st.clone_htod(&yd).unwrap());
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(frames * nv).unwrap();
        let map = if frames == 1 { Some(0) } else { None };
        op.forward_fat(&st, &seg, &pts, &maps, map, &dx, &mut y).unwrap();
        op.adjoint_fat(&st, &seg, &pts, &maps, map, &dy, &mut xa).unwrap();
        (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap())
    };
    // water + fat pair
    let mut op = SegNufft::new_temporal(&dev, dims, [0.004; 3], nc, opts(), knot).unwrap();
    op.set_maps(&st, &smaps).unwrap();
    let seg = op.segments_fat(&st, &sp, &at, &cl, t0, dt).unwrap();
    let pts = op.points(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, tt).unwrap();
    let wf: Vec<C32> = (0..2 * frames * nv).map(|i| { let (k, f) = (i % (frames * nv), if i < frames * nv { 1.0 - ff[i % (frames * nv)] } else { ff[i % (frames * nv)] }); C32::new(f * x[k].re, f * x[k].im) }).collect();
    let (dx, dy) = (st.clone_htod(&wf).unwrap(), st.clone_htod(&yd).unwrap());
    let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
    let mut xa = st.alloc_zeros::<C32>(2 * frames * nv).unwrap();
    op.forward_wf(&st, &seg, &pts, &dx, &mut y).unwrap();
    op.adjoint_wf(&st, &seg, &pts, &dy, &mut xa).unwrap();
    let (y, xa) = (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap());
    let n = frames * nv;
    let comb: Vec<C32> = (0..n).map(|k| { let f = ff[k]; C32::new((1.0 - f) * xa[k].re + f * xa[n + k].re, (1.0 - f) * xa[k].im + f * xa[n + k].im) }).collect();
    let (ef, ea) = (rel(&y, &yref), rel(&comb, &xref));
    eprintln!("frames {frames} deg {bspline} knot {knot}: water + fat images vs fat maps: forward {ef:.1e}, adjoint {ea:.1e}");
    assert!(ef < 3e-5 && ea < 6e-5, "water + fat pair differs from the fat-map path: forward {ef:.1e}, adjoint {ea:.1e}");
}

#[test]
fn water_fat_images_match_fat_maps_static() {
    case(1, 1, false);
}

#[test]
fn water_fat_images_match_fat_maps_temporal() {
    case(5, 3, false);   // interval-major, single-image kernels
    case(5, 3, true);    // knot-major
    case(4, 1, true);
}
