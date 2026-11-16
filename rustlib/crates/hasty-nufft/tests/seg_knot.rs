//! Knot-major temporal operator (`SegNufft::new_temporal(.., true)`) against the interval-major one: same operator up
//! to f32 summation order, forward and adjoint, linear and cubic B-splines, with split tile lists, a wrapping window
//! (periodic knots) and the minimum knot count; plus adjointness of the knot-major pair.
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

fn dot(a: &[C32], b: &[C32]) -> num_complex::Complex64 {
    a.iter().zip(b).map(|(x, y)| num_complex::Complex64::new(x.re as f64, x.im as f64).conj() * num_complex::Complex64::new(y.re as f64, y.im as f64)).sum()
}

fn case(frames: usize, bspline: u32, m: usize, order: u32) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 20, 16];
    let nv: usize = dims.iter().product();
    let (nc, nseg) = (3, 4);
    let mut r = R(4242 + frames as u64 * 7 + bspline as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * frames as f64) as f32).collect();
    let x: Vec<C32> = (0..frames * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yd: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    let opts = || SegNufftOpts { sigma: 1.25, width: 8, nseg, order, frames, bspline, max_grid_bytes: 1 << 28 };
    let mut out = Vec::new();
    let mut adj = Vec::new();
    for knot in [false, true] {
        let mut op = SegNufft::new_temporal(&dev, dims, [0.004; 3], nc, opts(), knot).unwrap();
        assert_eq!(op.ks > 0, knot);
        op.set_maps(&st, &smaps).unwrap();
        let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
        let pts = op.points(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, Some(&t)).unwrap();
        let dx = st.clone_htod(&x).unwrap();
        let dy = st.clone_htod(&yd).unwrap();
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(frames * nv).unwrap();
        op.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
        op.adjoint(&st, &seg, &pts, &dy, &mut xa).unwrap();
        // the fused residual too
        let mut rr = st.alloc_zeros::<C32>(nc * m).unwrap();
        op.forward_residual(&st, &seg, &pts, &dx, &dy, &mut rr).unwrap();
        let (y, xa, rr) = (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap(), st.clone_dtoh(&rr).unwrap());
        let resid: Vec<C32> = y.iter().zip(&yd).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect();
        assert!(rel(&rr, &resid) < 1e-5, "residual differs from forward − y: {:.1e}", rel(&rr, &resid));
        let (l, rgt) = (dot(&yd, &y), dot(&xa, &x));
        // normalised by ‖y‖‖Ax‖ (the dot of random vectors cancels ~√N-fold, which would amplify f32 rounding)
        let e = (l - rgt).norm() / (dot(&yd, &yd).re * dot(&y, &y).re).sqrt();
        eprintln!("frames {frames} deg {bspline} m {m}: {} adjointness {e:.1e}", if knot { "knot-major" } else { "interval-major" });
        adj.push(e);
        out.push((y, xa));
    }
    let (ef, ea) = (rel(&out[1].0, &out[0].0), rel(&out[1].1, &out[0].1));
    eprintln!("frames {frames} deg {bspline} m {m} order {order}: knot-major vs interval-major forward {ef:.1e}, adjoint {ea:.1e}");
    assert!(adj[1] < 2e-6 && adj[0] < 2e-6, "adjointness: knot-major {:.1e}, interval-major {:.1e}", adj[1], adj[0]);
    // f32 summation order only: both paths sum the same terms in different association (the interval path through
    // read-modify-writes of KS slots); the measured gap is at the level of either path's own adjointness error
    assert!(ef < 3e-5 && ea < 6e-5, "knot-major differs from interval-major: forward {ef:.1e}, adjoint {ea:.1e}");
}

#[test]
fn knot_major_linear_matches_interval_major() {
    case(6, 1, 3000, 2);
    case(2, 1, 3000, 0);    // minimum knot count: the window covers the period
}

#[test]
fn knot_major_cubic_matches_interval_major() {
    case(7, 3, 3000, 2);
    case(4, 3, 3000, 1);    // minimum knot count
}

#[test]
fn knot_major_order0_matches_interval_major() {
    case(6, 1, 3000, 0);
    case(7, 3, 3000, 0);
}

#[test]
fn knot_major_split_lists_match_interval_major() {
    // ~40 k samples on a 30 × 25 × 20 fine grid: tile lists far above the 1024-sample item limit (atomic items)
    case(5, 3, 40_000, 2);
}
