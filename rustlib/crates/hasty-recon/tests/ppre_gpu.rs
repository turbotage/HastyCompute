//! The device circulant metric built from given factors (`CirculantPrecond::from_factors`, research/precond_estimator.md
//! P0–P2) applies exactly like the host reference `ppre::SymPre` (same FFT conventions: a probed symbol transfers
//! between them unmirrored), with and without a deflation (P3), host-resident too; apply ∘ apply_inv = identity.

use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_recon::ppre::{kradius, Deflation, Fft3, SymPre};
use hasty_recon::precond::{CircShared, CirculantPrecond};
use hasty_testkit::Rng;

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|q| (q.re as f64).powi(2) + (q.im as f64).powi(2)).sum();
    (n / d).sqrt()
}

#[test]
fn from_factors_matches_host_symbol_and_deflation() {
    let dims = [20usize, 16, 12];
    let nv: usize = dims.iter().product();
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(hasty_recon::gpu::Kernels::new(&dev).unwrap());
    let mut r = Rng::new(7);
    // an asymmetric symbol (λ(k) ≠ λ(−k)) so that a sign-convention mismatch would show
    let lam: Vec<f32> = (0..nv).map(|q| 0.05 + 5.0 * (1.0 - kradius(dims, q) as f32).max(0.0) + if q % dims[0] < 3 { 1.0 } else { 0.0 }).collect();
    let dinv: Vec<f32> = (0..nv).map(|_| 1.0 + 0.5 * r.uniform() as f32).collect();
    let fft = Arc::new(Fft3::new(dims));
    let host = SymPre::new(fft.clone(), lam.clone(), Some(dinv.clone()));
    let x: Vec<C32> = (0..nv).map(|_| C32::new(r.uniform() as f32 - 0.5, r.uniform() as f32 - 0.5)).collect();
    let dev_apply = |p: &mut CirculantPrecond, inv: bool, v: &[C32]| -> Vec<C32> {
        let dv = st.clone_htod(v).unwrap();
        let mut z = st.alloc_zeros::<C32>(nv).unwrap();
        if inv { p.apply_inv(&st, &dv, &mut z).unwrap() } else { p.apply(&st, &dv, &mut z).unwrap() }
        st.clone_dtoh(&z).unwrap()
    };
    let mut pd = CirculantPrecond::from_factors(&dev, &st, kern.clone(), dims, &lam, &dinv).unwrap();
    let mut want = vec![C32::default(); nv];
    host.apply(&x, &mut want);
    let got = dev_apply(&mut pd, false, &x);
    assert!(rel(&got, &want) < 1e-5, "device vs host metric: {:e}", rel(&got, &want));
    host.apply_inv(&x, &mut want);
    let got = dev_apply(&mut pd, true, &x);
    assert!(rel(&got, &want) < 1e-5, "device vs host inverse: {:e}", rel(&got, &want));
    // the factors come back unchanged
    let (l2, d2) = pd.factors(&st).unwrap();
    assert!(l2.iter().zip(&lam).all(|(a, b)| (a - b).abs() <= 1e-6 * b) && d2.iter().zip(&dinv).all(|(a, b)| (a - b).abs() <= 1e-6 * b));
    // deflation by two M⁻¹-orthonormal vectors: u_i = M^{1/2}-images of orthonormal e_i (u = M w with wᴴ M w = 1)
    let mut us: Vec<Vec<C32>> = vec![];
    for s in 0..2u64 {
        let mut rr = Rng::new(100 + s);
        let w: Vec<C32> = (0..nv).map(|_| C32::new(rr.uniform() as f32 - 0.5, rr.uniform() as f32 - 0.5)).collect();
        let mut u = vec![C32::default(); nv];
        host.apply(&w, &mut u);
        // Gram–Schmidt in the M⁻¹ inner product: ⟨a, b⟩ = aᴴ M⁻¹ b
        for p in &us {
            let mut mp = vec![C32::default(); nv];
            host.apply_inv(p, &mut mp);
            let c = hasty_recon::ppre::cdot(&mp, &u);
            hasty_recon::ppre::caxpy(-c, p, &mut u);
        }
        let mut mu = vec![C32::default(); nv];
        host.apply_inv(&u, &mut mu);
        let n = hasty_recon::ppre::cdot(&u, &mu).re.sqrt();
        u.iter_mut().for_each(|v| { v.re /= n as f32; v.im /= n as f32; });
        us.push(u);
    }
    let df = Deflation { u: us.clone(), theta: vec![40.0, 12.0], target: vec![2.0, 2.0] };
    let mut pdd = CirculantPrecond::from_factors(&dev, &st, kern.clone(), dims, &lam, &dinv).unwrap().with_deflation(&st, &us, &df.theta, &df.target).unwrap();
    assert_eq!(pdd.deflation_rank(), 2);
    host.apply(&x, &mut want);
    df.add(&x, &mut want);
    let got = dev_apply(&mut pdd, false, &x);
    assert!(rel(&got, &want) < 1e-5, "deflated device vs host: {:e}", rel(&got, &want));
    // M' (M')⁻¹ x = x
    let back = { let w = dev_apply(&mut pdd, true, &x); dev_apply(&mut pdd, false, &w) };
    assert!(rel(&back, &x) < 1e-4, "deflated inverse: {:e}", rel(&back, &x));
    // host-resident copy applies the same
    let ws = CircShared::new(&dev, &st, dims).unwrap();
    let mut ph = CirculantPrecond::from_factors(&dev, &st, kern.clone(), dims, &lam, &dinv).unwrap().to_host(&st, &ws).unwrap().with_deflation(&st, &us, &df.theta, &df.target).unwrap();
    let got = dev_apply(&mut ph, false, &x);
    assert!(rel(&got, &want) < 1e-5, "host-resident deflated: {:e}", rel(&got, &want));
}
