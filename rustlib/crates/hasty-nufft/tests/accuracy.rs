use hasty_cuda::{Device, Real};
use hasty_nufft::{NufftOpts, NufftPlan, SpreadMethod};
use hasty_testkit::{ndft_type1, ndft_type2, rel_l2, Rng, C64 as RC64};

fn to_dev<R: Real>(v: &[RC64]) -> Vec<R::Complex> {
    v.iter().map(|c| R::complex(c.re, c.im)).collect()
}
fn from_dev<R: Real>(v: &[R::Complex]) -> Vec<RC64> {
    v.iter()
        .map(|&c| {
            let (a, b) = R::parts(c);
            RC64::new(a, b)
        })
        .collect()
}

/// returns (type1 err, type2 err) for the first transform
fn run<R: Real>(
    dev: &Device,
    modes: &[usize],
    m: usize,
    tol: f64,
    sigma: f64,
    method: SpreadMethod,
    ntransf: usize,
    clustered: bool,
) -> (f64, f64) {
    let ndim = modes.len();
    let stream = dev.default_stream();
    let mut rng = Rng::new(7 + m as u64 + modes[0] as u64);
    // points (f64 master copy), optionally clustered near the centre (radial-like density)
    let coords: Vec<Vec<f64>> = (0..ndim)
        .map(|_| {
            (0..m)
                .map(|_| {
                    let u = rng.uniform();
                    let v = if clustered { u * u * u.signum() } else { u };
                    v * std::f64::consts::PI * 0.999
                })
                .collect()
        })
        .collect();
    let dims: Vec<usize> = modes.iter().rev().cloned().collect(); // slowest first for the oracle
    let nm: usize = modes.iter().product();
    let c: Vec<RC64> = rng.complex_vec(m * ntransf);
    let f: Vec<RC64> = rng.complex_vec(nm * ntransf);
    let want1 = ndft_type1(&coords, &c[..m], &dims, 1);
    let want2 = ndft_type2(&coords, &f[..nm], &dims, -1);

    let opts = NufftOpts {
        tol,
        sigma,
        spread: method,
        ntransf,
        ..Default::default()
    };
    let mut plan = NufftPlan::<R>::new(dev, modes, opts).unwrap();
    let dc: Vec<_> = coords
        .iter()
        .map(|v| {
            stream
                .clone_htod(&v.iter().map(|&x| R::from_f64(x)).collect::<Vec<R>>())
                .unwrap()
        })
        .collect();
    let refs: Vec<&_> = dc.iter().collect();
    plan.set_points(&stream, &refs, m).unwrap();
    let dcc = stream.clone_htod(&to_dev::<R>(&c)).unwrap();
    let mut df = stream.alloc_zeros::<R::Complex>(nm * ntransf).unwrap();
    plan.type1(&stream, &dcc, &mut df, 1).unwrap();
    let got1 = from_dev::<R>(&stream.clone_dtoh(&df).unwrap());
    let e1 = rel_l2(&got1[..nm], &want1);

    let dff = stream.clone_htod(&to_dev::<R>(&f)).unwrap();
    let mut dcout = stream.alloc_zeros::<R::Complex>(m * ntransf).unwrap();
    plan.type2(&stream, &dff, &mut dcout, -1).unwrap();
    let got2 = from_dev::<R>(&stream.clone_dtoh(&dcout).unwrap());
    let e2 = rel_l2(&got2[..m], &want2);
    // all transforms in the batch are independent: check the last one against the oracle too
    if ntransf > 1 {
        let t = ntransf - 1;
        let w1 = ndft_type1(&coords, &c[t * m..(t + 1) * m], &dims, 1);
        assert!(
            rel_l2(&got1[t * nm..(t + 1) * nm], &w1) < 4.0 * e1.max(1e-7),
            "last transform of the batch"
        );
    }
    (e1, e2)
}

#[test]
fn f64_matches_ndft_1d_2d_3d() {
    let dev = Device::new(0).unwrap();
    for (modes, m) in [
        (vec![32usize], 300usize),
        (vec![20, 24], 500),
        (vec![14, 12, 10], 700),
    ] {
        for method in [SpreadMethod::Gm, SpreadMethod::Sm] {
            for tol in [1e-3, 1e-6, 1e-9] {
                let (e1, e2) = run::<f64>(&dev, &modes, m, tol, 2.0, method, 1, false);
                assert!(
                    e1 < 6.0 * tol && e2 < 6.0 * tol,
                    "{modes:?} {method:?} tol {tol:e}: e1 {e1:e} e2 {e2:e}"
                );
            }
        }
    }
}

#[test]
fn f32_accuracy_and_low_sigma() {
    let dev = Device::new(0).unwrap();
    for sigma in [2.0, 1.5, 1.25] {
        for tol in [1e-2, 1e-4] {
            for method in [SpreadMethod::Gm, SpreadMethod::Sm] {
                let (e1, e2) = run::<f32>(&dev, &[16, 18, 12], 800, tol, sigma, method, 1, false);
                assert!(
                    e1 < 8.0 * tol && e2 < 8.0 * tol,
                    "sigma {sigma} tol {tol:e} {method:?}: {e1:e} {e2:e}"
                );
            }
        }
    }
    let (e1, e2) = run::<f32>(
        &dev,
        &[24, 24, 24],
        3000,
        1e-6,
        2.0,
        SpreadMethod::Auto,
        1,
        false,
    );
    assert!(e1 < 2e-5 && e2 < 2e-5, "f32 1e-6: {e1:e} {e2:e}");
}

#[test]
fn batched_clustered_odd_sizes() {
    let dev = Device::new(0).unwrap();
    let (e1, e2) = run::<f32>(
        &dev,
        &[15, 13, 11],
        2000,
        1e-5,
        2.0,
        SpreadMethod::Sm,
        5,
        true,
    );
    assert!(e1 < 1e-4 && e2 < 1e-4, "{e1:e} {e2:e}");
    let (e1, e2) = run::<f32>(
        &dev,
        &[15, 13, 11],
        2000,
        1e-5,
        2.0,
        SpreadMethod::Gm,
        5,
        true,
    );
    assert!(e1 < 1e-4 && e2 < 1e-4, "{e1:e} {e2:e}");
}

#[test]
fn adjointness() {
    // <type2(f), c> == <f, type1(c)> with opposite signs
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let (modes, m) = (vec![16usize, 14, 12], 1500usize);
    let mut rng = Rng::new(5);
    let coords: Vec<Vec<f32>> = (0..3)
        .map(|_| (0..m).map(|_| (rng.uniform() * 3.1) as f32).collect())
        .collect();
    let nm: usize = modes.iter().product();
    let f = rng.complex_vec(nm);
    let c = rng.complex_vec(m);
    let mut plan = NufftPlan::<f32>::new(
        &dev,
        &modes,
        NufftOpts {
            tol: 1e-6,
            ..Default::default()
        },
    )
    .unwrap();
    let dc: Vec<_> = coords
        .iter()
        .map(|v| stream.clone_htod(v).unwrap())
        .collect();
    let refs: Vec<&_> = dc.iter().collect();
    plan.set_points(&stream, &refs, m).unwrap();
    let df = stream.clone_htod(&to_dev::<f32>(&f)).unwrap();
    let dcc = stream.clone_htod(&to_dev::<f32>(&c)).unwrap();
    let mut out_c = stream.alloc_zeros::<hasty_cuda::C32>(m).unwrap();
    let mut out_f = stream.alloc_zeros::<hasty_cuda::C32>(nm).unwrap();
    plan.type2(&stream, &df, &mut out_c, -1).unwrap();
    plan.type1(&stream, &dcc, &mut out_f, 1).unwrap();
    let a = from_dev::<f32>(&stream.clone_dtoh(&out_c).unwrap());
    let b = from_dev::<f32>(&stream.clone_dtoh(&out_f).unwrap());
    let lhs: RC64 = a.iter().zip(&c).map(|(x, y)| x * y.conj()).sum();
    let rhs: RC64 = f.iter().zip(&b).map(|(x, y)| x * y.conj()).sum();
    // <A f, c> = sum (Af)_j conj(c_j) ;  <f, A^H c> = sum f_k conj((A^H c)_k)
    assert!((lhs - rhs).norm() / lhs.norm() < 2e-5, "{lhs} vs {rhs}");
}
