//! Host-resident circulant preconditioners (one shared device workspace) apply exactly like device-resident ones,
//! also when several of them alternate on the same stream.

use hasty_cuda::{Device, C32};
use hasty_recon::precond::{CircShared, CirculantPrecond};
use hasty_testkit::Rng;

#[test]
fn host_resident_circulant_matches_device() {
    let dims = [20usize, 16, 12];
    let nv: usize = dims.iter().product();
    let nc = 4;
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(hasty_recon::gpu::Kernels::new(&dev).unwrap());
    let mut r = Rng::new(5);
    let smaps: Vec<C32> = (0..nc * nv).map(|_| C32::new(r.uniform() as f32, r.uniform() as f32 - 0.5)).collect();
    let mk = |seed: u64| -> CirculantPrecond {
        let mut r = Rng::new(seed);
        let m = 4000;
        let c: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (std::f64::consts::PI * (2.0 * r.uniform() - 1.0) * r.uniform()) as f32).collect());
        CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&c[0], &c[1], &c[2]], &smaps, nc, 1e-3).unwrap()
    };
    let (mut d1, mut d2) = (mk(1), mk(2));
    let ws = CircShared::new(&dev, &st, dims).unwrap();
    let (mut h1, mut h2) = (mk(1).to_host(&st, &ws).unwrap(), mk(2).to_host(&st, &ws).unwrap());
    assert!(h1.on_host() && !d1.on_host());
    let x: Vec<C32> = (0..nv).map(|_| C32::new(r.uniform() as f32 - 0.5, r.uniform() as f32 - 0.5)).collect();
    let dx = st.clone_htod(&x).unwrap();
    let mut out = |p: &mut CirculantPrecond, inv: bool| -> Vec<C32> {
        let mut z = st.alloc_zeros::<C32>(nv).unwrap();
        if inv { p.apply_inv(&st, &dx, &mut z).unwrap() } else { p.apply(&st, &dx, &mut z).unwrap() }
        st.clone_dtoh(&z).unwrap()
    };
    let rel = |a: &[C32], b: &[C32]| {
        let n: f64 = a.iter().zip(b).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum();
        let d: f64 = b.iter().map(|q| (q.re as f64).powi(2) + (q.im as f64).powi(2)).sum();
        (n / d).sqrt()
    };
    for inv in [false, true] {
        // alternate the two host preconditioners through the shared slots
        let (a1, a2, b1, b2) = (out(&mut h1, inv), out(&mut h2, inv), out(&mut h1, inv), out(&mut h2, inv));
        let (e1, e2) = (out(&mut d1, inv), out(&mut d2, inv));
        for (name, a, e) in [("1", &a1, &e1), ("2", &a2, &e2), ("1 again", &b1, &e1), ("2 again", &b2, &e2)] {
            let err = rel(a, e);
            assert!(err < 1e-6, "host-resident preconditioner {name} (inverse {inv}) differs: {err:e}");
        }
        assert!(rel(&a1, &a2) > 1e-3, "the two preconditioners should differ");
    }
}
