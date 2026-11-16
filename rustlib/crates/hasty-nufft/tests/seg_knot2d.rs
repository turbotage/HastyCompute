//! 2-D knots (respiratory × cardiac, `SegNufft::new_temporal2d`) against an explicit reference: per respiratory
//! interval i, the 1-D cardiac knot-major operator A_i on the interval's samples, combined with the respiratory weights
//! on the host: y = Σ_k γ_k ⊙ A_i x_k (bins: γ = 1 for knot i; linear: 1 − v for knot i, v for knot i + 1), and the
//! adjoint x_k = Σ_i A_iᴴ(γ_k ⊙ y_i). Plus adjointness of the 2-D pair.
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

fn case(mc: usize, cdeg: u32, kr: usize, rdeg: u32, m: usize) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [20usize, 18, 14];
    let nv: usize = dims.iter().product();
    let (nc, nseg) = (3, 4);
    let nri = if rdeg == 0 { kr } else { kr - 1 };
    let mut r = R(777 + kr as u64 * 13 + rdeg as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * nri as f64) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| if rdeg == 0 { 0.0 } else { (r.u() + 0.5) as f32 }).collect();
    let x: Vec<C32> = (0..kr * mc * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yd: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    let opts = || SegNufftOpts { sigma: 1.25, width: 8, nseg, order: 2, frames: mc, bspline: cdeg, max_grid_bytes: 1 << 28 };
    // 2-D engine
    let (y2, x2) = {
        let mut op = SegNufft::new_temporal2d(&dev, dims, [0.004; 3], nc, opts(), kr, rdeg).unwrap();
        op.set_maps(&st, &smaps).unwrap();
        let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
        let pts = op.points2d(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, &t, &ri, &rv).unwrap();
        let (dx, dy) = (st.clone_htod(&x).unwrap(), st.clone_htod(&yd).unwrap());
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(kr * mc * nv).unwrap();
        op.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
        op.adjoint(&st, &seg, &pts, &dy, &mut xa).unwrap();
        let (y, xa) = (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap());
        let dot = |a: &[C32], b: &[C32]| a.iter().zip(b).map(|(p, q)| num_complex::Complex64::new(p.re as f64, -p.im as f64) * num_complex::Complex64::new(q.re as f64, q.im as f64)).sum::<num_complex::Complex64>();
        let e = (dot(&yd, &y) - dot(&xa, &x)).norm() / (dot(&yd, &yd).re * dot(&y, &y).re).sqrt();
        eprintln!("2-D knots M {mc} deg {cdeg}, kr {kr} rdeg {rdeg}: normalised adjointness {e:.1e}");
        assert!(e < 2e-6, "2-D pair not adjoint: {e:.1e}");
        (y, xa)
    };
    // reference: per respiratory interval, the 1-D engine on its samples
    let mut yr = vec![C32::new(0.0, 0.0); nc * m];
    let mut xr = vec![C32::new(0.0, 0.0); kr * mc * nv];
    let mut op = SegNufft::new_temporal(&dev, dims, [0.004; 3], nc, opts(), true).unwrap();
    op.set_maps(&st, &smaps).unwrap();
    let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
    for i in 0..nri {
        let sel: Vec<usize> = (0..m).filter(|&j| ri[j] as usize == i).collect();
        if sel.is_empty() {
            continue;
        }
        let ms = sel.len();
        let g = |v: &Vec<f32>| -> Vec<f32> { sel.iter().map(|&j| v[j]).collect() };
        let cs = [g(&coords[0]), g(&coords[1]), g(&coords[2])];
        let dcs: Vec<f32> = (0..6).flat_map(|q| sel.iter().map(move |&j| q * m + j)).map(|k| dcv[k]).collect();
        let pts = op.points(&st, [&cs[0], &cs[1], &cs[2]], &g(&tau), &g(&phi0), &dcs, Some(&g(&t))).unwrap();
        let knots: Vec<(usize, Vec<f32>)> = if rdeg == 0 { vec![(i, vec![1.0; ms])] } else { vec![(i, sel.iter().map(|&j| 1.0 - rv[j]).collect()), (i + 1, sel.iter().map(|&j| rv[j]).collect())] };
        for (k, w) in knots {
            let dx = st.clone_htod(&x[k * mc * nv..(k + 1) * mc * nv]).unwrap();
            let mut y = st.alloc_zeros::<C32>(nc * ms).unwrap();
            op.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
            let y = st.clone_dtoh(&y).unwrap();
            let mut ys = vec![C32::new(0.0, 0.0); nc * ms];
            for c in 0..nc {
                for (jj, &j) in sel.iter().enumerate() {
                    let v = y[c * ms + jj];
                    { let o = yr[c * m + j]; yr[c * m + j] = C32::new(o.re + w[jj] * v.re, o.im + w[jj] * v.im); }
                    let d = yd[c * m + j];
                    ys[c * ms + jj] = C32::new(w[jj] * d.re, w[jj] * d.im);
                }
            }
            let dys = st.clone_htod(&ys).unwrap();
            let mut xa = st.alloc_zeros::<C32>(mc * nv).unwrap();
            op.adjoint(&st, &seg, &pts, &dys, &mut xa).unwrap();
            for (q, v) in st.clone_dtoh(&xa).unwrap().into_iter().enumerate() {
                { let o = xr[k * mc * nv + q]; xr[k * mc * nv + q] = C32::new(o.re + v.re, o.im + v.im); }
            }
        }
    }
    let (ef, ea) = (rel(&y2, &yr), rel(&x2, &xr));
    eprintln!("2-D knots M {mc} deg {cdeg}, kr {kr} rdeg {rdeg}, {m} samples: vs per-interval 1-D reference forward {ef:.1e}, adjoint {ea:.1e}");
    assert!(ef < 3e-5 && ea < 6e-5, "2-D knots differ from the reference: forward {ef:.1e}, adjoint {ea:.1e}");
}

#[test]
fn knot2d_bins_match_reference() {
    case(6, 3, 3, 0, 4000);
}

#[test]
fn knot2d_linear_match_reference() {
    case(6, 3, 4, 1, 4000);
    case(4, 1, 2, 1, 3000);
}

#[test]
fn knot2d_split_lists_match_reference() {
    case(5, 3, 3, 1, 40_000);
}
