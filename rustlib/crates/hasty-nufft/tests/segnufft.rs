use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts, SegPoints, SegSegments};
use num_complex::Complex64;

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

struct Setup {
    dims: [usize; 3],
    res: [f64; 3],
    nc: usize,
    x: Vec<Complex64>,
    smaps: Vec<Complex64>,
    omega: Vec<f64>,
    /// per sample: coords (rad/voxel), tau, full quadratic C (6), residual ΔC (6)
    k: [Vec<f64>; 3],
    tau: Vec<f64>,
    c: Vec<[f64; 6]>,
    dc: Vec<[f64; 6]>,
    cbar0: [f64; 6],
    t: (f64, f64),
    /// frames (x holds frames × nv), temporal B-spline degree, per-sample acquisition time in frame units
    frames: usize,
    bspline: u32,
    tt: Vec<f64>,
    /// per-sample constant phase (rad): concomitant φ0 / per-encode terms
    phi0: Vec<f64>,
}

fn rho(s: &Setup, v: usize) -> [f64; 3] {
    let d = s.dims;
    let (i, j, k) = (v % d[0], (v / d[0]) % d[1], v / (d[0] * d[1]));
    [(i as f64 - (d[0] / 2) as f64) * s.res[0], (j as f64 - (d[1] / 2) as f64) * s.res[1], (k as f64 - (d[2] / 2) as f64) * s.res[2]]
}
fn quad(c: &[f64; 6], p: &[f64; 3]) -> f64 {
    c[0] * p[0] * p[0] + c[1] * p[1] * p[1] + c[2] * p[2] * p[2] + 2.0 * (c[3] * p[0] * p[1] + c[4] * p[0] * p[2] + c[5] * p[1] * p[2])
}

fn setup(m: usize, dc_scale: f64) -> Setup {
    setup_t(m, dc_scale, 1, 1)
}

fn setup_t(m: usize, dc_scale: f64, frames: usize, bspline: u32) -> Setup {
    let dims = [16usize, 14, 10];
    let res = [0.012, 0.012, 0.012];
    let nv = dims.iter().product::<usize>();
    let nc = 3;
    let mut r = R(17);
    let x: Vec<Complex64> = (0..frames * nv).map(|_| Complex64::new(r.u(), r.u())).collect();
    let mut s = Setup { dims, res, nc, x, smaps: vec![], omega: vec![], k: [vec![], vec![], vec![]], tau: vec![], c: vec![], dc: vec![], cbar0: [0.0; 6], t: (2e-3, 8e-3), frames, bspline, tt: vec![], phi0: vec![] };
    // smooth coil maps and a smooth-but-structured field map (±~200 Hz)
    for c in 0..nc {
        for v in 0..nv {
            let p = rho(&s, v);
            let ph = 3.0 * (c as f64 + 1.0) * p[0] + 2.0 * p[1];
            s.smaps.push(Complex64::from_polar((-(p[0] - 0.05 * c as f64).powi(2) / 0.02).exp(), ph));
        }
    }
    for v in 0..nv {
        let p = rho(&s, v);
        s.omega.push(2.0 * std::f64::consts::PI * (150.0 * (12.0 * p[0]).sin() * (9.0 * p[1]).cos() + 80.0 * (20.0 * p[2]).sin()));
    }
    // τ-dependent quadratic phase reaching ~2 rad at the FOV corner by the end of the readout
    let corner = 0.1f64;
    s.cbar0 = [2.0 / (corner * corner) * 0.4, 2.0 / (corner * corner) * 0.35, 2.0 / (corner * corner) * 0.2, 10.0, -8.0, 5.0];
    for _ in 0..m {
        for a in 0..3 {
            s.k[a].push(r.u() * 2.0 * std::f64::consts::PI * 0.98);
        }
        let t = s.t.0 + (r.u() + 0.5) * (s.t.1 - s.t.0);
        s.tau.push(t);
        let f = (t - s.t.0) / (s.t.1 - s.t.0);
        let d: [f64; 6] = [0, 1, 2, 3, 4, 5].map(|_| r.u() * dc_scale / (corner * corner));
        let mut c = [0.0; 6];
        for q in 0..6 {
            c[q] = s.cbar0[q] * f + d[q];
        }
        s.c.push(c);
        s.dc.push(d);
        s.tt.push((r.u() + 0.5) * frames as f64);
        s.phi0.push(r.u() * 7.0);
    }
    s
}

/// temporal weight of frame f for a sample at time t (periodic over `frames`): centred hat (degree 1) or cubic B₃
fn bweight(s: &Setup, f: usize, t: f64) -> f64 {
    if s.frames == 1 {
        return 1.0;
    }
    let n = s.frames as f64;
    let mut d = (t - f as f64).rem_euclid(n);
    if d > 0.5 * n {
        d -= n;
    }
    let a = d.abs();
    match s.bspline {
        1 => (1.0 - a).max(0.0),
        _ if a <= 1.0 => 2.0 / 3.0 - a * a + 0.5 * a * a * a,
        _ if a <= 2.0 => (2.0 - a).powi(3) / 6.0,
        _ => 0.0,
    }
}

fn exact(s: &Setup) -> Vec<Complex64> {
    let nv = s.x.len() / s.frames;
    let m = s.tau.len();
    let mut y = vec![Complex64::new(0.0, 0.0); s.nc * m];
    for j in 0..m {
        let bw: Vec<f64> = (0..s.frames).map(|f| bweight(s, f, s.tt[j])).collect();
        for v in 0..nv {
            let d = s.dims;
            let (i, jj, k) = (v % d[0], (v / d[0]) % d[1], v / (d[0] * d[1]));
            let lin = s.k[0][j] * (i as f64 - (d[0] / 2) as f64) + s.k[1][j] * (jj as f64 - (d[1] / 2) as f64) + s.k[2][j] * (k as f64 - (d[2] / 2) as f64);
            let ph = s.phi0[j] + lin + s.omega[v] * s.tau[j] + quad(&s.c[j], &rho(s, v));
            let xv: Complex64 = (0..s.frames).filter(|&f| bw[f] != 0.0).map(|f| bw[f] * s.x[f * nv + v]).sum();
            let e = Complex64::from_polar(1.0, -ph) * xv;
            for c in 0..s.nc {
                y[c * m + j] += s.smaps[c * nv + v] * e;
            }
        }
    }
    y
}

fn build(dev: &Device, s: &Setup, nseg: usize, order: u32) -> (SegNufft, SegSegments, SegPoints) {
    let st = dev.default_stream();
    let nv = s.x.len() / s.frames;
    let mut op = SegNufft::new(dev, s.dims, s.res, s.nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames: s.frames, bspline: s.bspline, max_grid_bytes: 1 << 30 }).unwrap();
    op.set_maps(&st, &s.smaps.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect::<Vec<_>>()).unwrap();
    let cb = s.cbar0;
    let (t0, t1) = s.t;
    let psi = move |v: usize, t: f64| -> f64 {
        let f = (t - t0) / (t1 - t0);
        let c: [f64; 6] = [0, 1, 2, 3, 4, 5].map(|q| cb[q] * f);
        s.omega[v] * t + quad(&c, &rho(s, v))
    };
    let all: Vec<usize> = (0..nv).collect();
    let (segph, atab, ta, dt) = build_segments(nv, nseg, s.t.0, s.t.1, 256, &all, &psi);
    let seg = op.segments(&st, &segph, &atab, ta, dt).unwrap();
    let m = s.tau.len();
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|a| s.k[a].iter().map(|&v| v as f32).collect());
    let tau: Vec<f32> = s.tau.iter().map(|&v| v as f32).collect();
    let mut dc = vec![0f32; 6 * m];
    for j in 0..m {
        for q in 0..6 {
            dc[q * m + j] = s.dc[j][q] as f32;
        }
    }
    let tt: Vec<f32> = s.tt.iter().map(|&v| v as f32).collect();
    let phi0: Vec<f32> = s.phi0.iter().map(|&v| v as f32).collect();
    let pts = op.points(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dc, (s.frames > 1).then_some(&tt[..])).unwrap();
    (op, seg, pts)
}

fn rel(a: &[C32], b: &[Complex64]) -> f64 {
    let (mut n, mut d) = (0.0, 0.0);
    for (p, q) in a.iter().zip(b) {
        n += (Complex64::new(p.re as f64, p.im as f64) - q).norm_sqr();
        d += q.norm_sqr();
    }
    (n / d).sqrt()
}

fn adjoint_check(frames: usize, bspline: u32) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let s = setup_t(2000, 0.15, frames, bspline);
    let (mut op, seg, pts) = build(&dev, &s, 6, 2);
    let nx = s.x.len();
    let m = s.tau.len();
    let mut r = R(5);
    let xr: Vec<C32> = (0..nx).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yr: Vec<C32> = (0..s.nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let (dx, dy) = (st.clone_htod(&xr).unwrap(), st.clone_htod(&yr).unwrap());
    let mut ax = st.alloc_zeros::<C32>(s.nc * m).unwrap();
    let mut ahy = st.alloc_zeros::<C32>(nx).unwrap();
    op.forward(&st, &seg, &pts, &dx, &mut ax).unwrap();
    op.adjoint(&st, &seg, &pts, &dy, &mut ahy).unwrap();
    let (ax, ahy) = (st.clone_dtoh(&ax).unwrap(), st.clone_dtoh(&ahy).unwrap());
    let dot = |a: &[C32], b: &[C32]| -> Complex64 { a.iter().zip(b).map(|(p, q)| Complex64::new(p.re as f64, -p.im as f64) * Complex64::new(q.re as f64, q.im as f64)).sum() };
    let (l, rr) = (dot(&yr, &ax), dot(&ahy, &xr));
    eprintln!("frames {frames}, B-spline {bspline}: <y, A x> = {l:.6e}, <A^H y, x> = {rr:.6e}");
    assert!((l - rr).norm() < 1e-4 * l.norm());
}

#[test]
fn segnufft_is_adjoint() {
    adjoint_check(1, 1);
}

#[test]
fn segnufft_temporal_is_adjoint() {
    for (frames, bspline) in [(2, 1), (5, 1), (4, 3), (7, 3)] {
        adjoint_check(frames, bspline);
    }
}

#[test]
fn segnufft_temporal_matches_direct_sum() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    for (frames, bspline) in [(5, 1), (6, 3)] {
        let s = setup_t(3000, 0.15, frames, bspline);
        let want = exact(&s);
        let dx = st.clone_htod(&s.x.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect::<Vec<_>>()).unwrap();
        let m = s.tau.len();
        let (mut op, seg, pts) = build(&dev, &s, 10, 2);
        let mut y = st.alloc_zeros::<C32>(s.nc * m).unwrap();
        op.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
        let e = rel(&st.clone_dtoh(&y).unwrap(), &want);
        eprintln!("frames {frames}, B-spline {bspline}, L = 10, order 2: relative error {e:.2e}");
        assert!(e < 1e-3, "error {e}");
    }
}

#[test]
fn segnufft_matches_direct_sum() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let s = setup(3000, 0.15);
    let want = exact(&s);
    let dx = st.clone_htod(&s.x.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect::<Vec<_>>()).unwrap();
    let m = s.tau.len();
    for (nseg, order) in [(4, 0), (6, 0), (6, 1), (8, 1), (10, 1), (6, 2), (8, 2), (10, 2)] {
        let (mut op, seg, pts) = build(&dev, &s, nseg, order);
        let mut y = st.alloc_zeros::<C32>(s.nc * m).unwrap();
        op.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
        let e = rel(&st.clone_dtoh(&y).unwrap(), &want);
        eprintln!("L = {nseg:>2}, correction order {order}: relative error {e:.2e}");
        if nseg == 10 && order == 2 {
            assert!(e < 1e-3, "error {e}");
        }
    }
}
