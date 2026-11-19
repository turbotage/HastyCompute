//! Per-frame segments (`SegNufft::segments_frames`): one ratemap per operator frame, so frame b's image is padded with
//! its own diagonals P_{l,b} and its samples use its own interpolators a_{l,b}(τ). By linearity the per-frame operator
//! equals the sum over frames of the shared-segment operator built from frame b's segments, applied to frame b alone:
//!   A_pf x = Σ_b A[seg_b](x ⊙ 1_b),      (A_pfᴴ y)_b = (A[seg_b]ᴴ y)_b.
//! Every frame gets a different field (different ratemap), on a 2-D (respiratory linear × cardiac cubic) knot engine.
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

fn case(mc: usize, cdeg: u32, kr: usize, order: u32, m: usize) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [20usize, 18, 14];
    let nv: usize = dims.iter().product();
    let (nc, nseg) = (3, 4);
    let nri = kr - 1;
    let nfr = kr * mc;
    let mut r = R(4242 + kr as u64 * 31 + mc as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * nri as f64) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| (r.u() + 0.5) as f32).collect();
    let x: Vec<C32> = (0..nfr * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yd: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    // a different ratemap per frame: field amplitude and offset depend on the frame
    let mut sp_all = Vec::new();
    let mut at_all = Vec::new();
    let mut segs1 = Vec::new();
    let (mut t0, mut dt) = (0.0, 0.0);
    for b in 0..nfr {
        let (amp, off) = (900.0 + 60.0 * b as f64, 40.0 * ((b * 5) % 7) as f64);
        let psi = move |v: usize, tt: f64| (amp * ((v % 113) as f64 / 113.0 - 0.5) + off) * tt;
        let (sp, at, a0, d0) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
        t0 = a0;
        dt = d0;
        segs1.push((sp.clone(), at.clone()));
        sp_all.extend_from_slice(&sp);
        at_all.extend_from_slice(&at);
    }
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg, order, frames: mc, bspline: cdeg, max_grid_bytes: 1 << 28 };
    let mut op = SegNufft::new_temporal2d(&dev, dims, [0.004; 3], nc, opts, kr, 1).unwrap();
    op.set_maps(&st, &smaps).unwrap();
    let pts = op.points2d(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, &t, &ri, &rv).unwrap();
    let (dx, dy) = (st.clone_htod(&x).unwrap(), st.clone_htod(&yd).unwrap());
    // per-frame operator
    let segf = op.segments_frames(&st, &sp_all, &at_all, t0, dt).unwrap();
    assert_eq!(segf.per_frame(), nfr);
    let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
    let mut xa = st.alloc_zeros::<C32>(nfr * nv).unwrap();
    op.forward(&st, &segf, &pts, &dx, &mut y).unwrap();
    op.adjoint(&st, &segf, &pts, &dy, &mut xa).unwrap();
    let (y, xa) = (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap());
    let dot = |a: &[C32], b: &[C32]| a.iter().zip(b).map(|(p, q)| num_complex::Complex64::new(p.re as f64, -p.im as f64) * num_complex::Complex64::new(q.re as f64, q.im as f64)).sum::<num_complex::Complex64>();
    let e = (dot(&yd, &y) - dot(&xa, &x)).norm() / (dot(&yd, &yd).re * dot(&y, &y).re).sqrt();
    // reference: Σ_b shared operator of frame b on frame b alone
    let mut yr = vec![C32::new(0.0, 0.0); nc * m];
    let mut xr = vec![C32::new(0.0, 0.0); nfr * nv];
    for (b, (sp, at)) in segs1.iter().enumerate() {
        let seg = op.segments(&st, sp, at, t0, dt).unwrap();
        let mut xb = vec![C32::new(0.0, 0.0); nfr * nv];
        xb[b * nv..(b + 1) * nv].copy_from_slice(&x[b * nv..(b + 1) * nv]);
        let dxb = st.clone_htod(&xb).unwrap();
        let mut yb = st.alloc_zeros::<C32>(nc * m).unwrap();
        op.forward(&st, &seg, &pts, &dxb, &mut yb).unwrap();
        for (o, v) in yr.iter_mut().zip(st.clone_dtoh(&yb).unwrap()) {
            *o = C32::new(o.re + v.re, o.im + v.im);
        }
        let mut xab = st.alloc_zeros::<C32>(nfr * nv).unwrap();
        op.adjoint(&st, &seg, &pts, &dy, &mut xab).unwrap();
        let xab = st.clone_dtoh(&xab).unwrap();
        xr[b * nv..(b + 1) * nv].copy_from_slice(&xab[b * nv..(b + 1) * nv]);
    }
    let (ef, ea) = (rel(&y, &yr), rel(&xa, &xr));
    eprintln!("per-frame segments, {kr}×{mc} frames (cardiac degree {cdeg}), order {order}, {m} samples: vs Σ_b shared reference forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}");
    assert!(e < 2e-6, "per-frame pair not adjoint: {e:.1e}");
    assert!(ef < 3e-5 && ea < 6e-5, "per-frame segments differ from the reference: forward {ef:.1e}, adjoint {ea:.1e}");
}

#[test]
fn per_frame_segments_match_reference() {
    case(6, 3, 3, 2, 4000);
    case(4, 1, 2, 2, 3000);
}
