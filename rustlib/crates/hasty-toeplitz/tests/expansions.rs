use cudarc::driver::CudaSlice;
use hasty_cuda::{Device, C32};
use hasty_testkit::{rel_l2, Rng};
use hasty_toeplitz::*;
use num_complex::Complex64;

const DIMS: [usize; 3] = [5, 6, 7]; // [Zn, Yn, Xn]

fn sample_set(m: usize, phi_scale: f64, seed: u64) -> SampleSetHost {
    let mut r = Rng::new(seed);
    let k: [Vec<f64>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| r.uniform() * 2.8).collect());
    let wt: Vec<f64> = (0..m).map(|_| 0.5 + 0.5 * (r.uniform() + 1.0)).collect();
    // w ~ sample-dependent coefficients with a shared growth (like an accumulated gradient-squared integral)
    let w: [Vec<f64>; 4] = [0, 1, 2, 3].map(|q| {
        (0..m)
            .map(|_| phi_scale * r.uniform() * if q == 0 { 1.0 } else { 0.6 })
            .collect()
    });
    SampleSetHost { k, wt, w }
}

struct Gpu {
    dev: Device,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    k: [CudaSlice<f32>; 3],
    wt: CudaSlice<f32>,
    w: [CudaSlice<f32>; 4],
    m: usize,
}

fn upload(dev: &Device, s: &SampleSetHost) -> Gpu {
    let stream = dev.default_stream();
    let f = |v: &Vec<f64>| {
        stream
            .clone_htod(&v.iter().map(|&x| x as f32).collect::<Vec<f32>>())
            .unwrap()
    };
    Gpu {
        dev: dev.clone(),
        stream: stream.clone(),
        k: [f(&s.k[0]), f(&s.k[1]), f(&s.k[2])],
        wt: f(&s.wt),
        w: [f(&s.w[0]), f(&s.w[1]), f(&s.w[2]), f(&s.w[3])],
        m: s.m(),
    }
}

fn to_c32(v: &[Complex64]) -> Vec<C32> {
    v.iter()
        .map(|c| C32::new(c.re as f32, c.im as f32))
        .collect()
}
fn from_c32(v: &[C32]) -> Vec<Complex64> {
    v.iter()
        .map(|c| Complex64::new(c.re as f64, c.im as f64))
        .collect()
}

fn run_op(
    g: &Gpu,
    spec: MixSpec,
    extra: &[CudaSlice<C32>],
    arrays: Vec<CudaSlice<C32>>,
    x: &[Complex64],
) -> Vec<Complex64> {
    let samples = Samples::<f32> {
        m: g.m,
        coords: [&g.k[0], &g.k[1], &g.k[2]],
        wt: &g.wt,
        w: [&g.w[0], &g.w[1], &g.w[2], &g.w[3]],
        extra,
    };
    let kernels = build_kernels(
        &g.dev,
        &g.stream,
        DIMS,
        &samples,
        &spec.kernels,
        &KernelBuildOpts {
            tol: 1e-7,
            sigma: 2.0,
        },
    )
    .unwrap();
    let mut op = MixOp::<f32>::new(&g.dev, spec, DIMS, kernels, arrays).unwrap();
    let dx = g.stream.clone_htod(&to_c32(x)).unwrap();
    let mut dout = g.stream.alloc_zeros::<C32>(x.len()).unwrap();
    op.apply(&g.stream, &dx, &mut dout, false).unwrap();
    let fused_result = from_c32(&g.stream.clone_dtoh(&dout).unwrap());
    // the fused multi-field Z pass and the unfused path must agree to rounding
    if op.is_fused() {
        op.set_fused(false);
        let mut dout2 = g.stream.alloc_zeros::<C32>(x.len()).unwrap();
        op.apply(&g.stream, &dx, &mut dout2, false).unwrap();
        let unfused = from_c32(&g.stream.clone_dtoh(&dout2).unwrap());
        assert!(rel_l2(&fused_result, &unfused) < 3e-6, "fused vs unfused: {:e}", rel_l2(&fused_result, &unfused));
    } else {
        eprintln!("(fused path not available for this configuration)");
    }
    fused_result
}

fn dense_apply(s: &SampleSetHost, series: &Series, x: &[Complex64]) -> Vec<Complex64> {
    apply_dense(&dense_series_operator(DIMS, s, series), x)
}

#[test]
fn plain_equals_dense_without_phase() {
    let dev = Device::new(0).unwrap();
    let s = sample_set(300, 0.0, 1);
    let g = upload(&dev, &s);
    let x = Rng::new(2).complex_vec(DIMS.iter().product());
    let got = run_op(&g, plain(), &[], vec![], &x);
    let want = dense_apply(&s, &Series::Exact, &x);
    assert!(
        rel_l2(&got, &want) < 2e-5,
        "plain: {:e}",
        rel_l2(&got, &want)
    );
}

#[test]
fn taylor_hankel_matches_its_series_and_converges() {
    let dev = Device::new(0).unwrap();
    let s = sample_set(300, 0.9, 3);
    let g = upload(&dev, &s);
    let x = Rng::new(4).complex_vec(DIMS.iter().product());
    let exact = dense_apply(&s, &Series::Exact, &x);
    let mut prev = f64::MAX;
    for p in [1usize, 2, 3, 4] {
        let got = run_op(&g, taylor_hankel(p), &[], vec![], &x);
        let series = dense_apply(&s, &Series::TaylorPhi(p), &x);
        let e_series = rel_l2(&got, &series);
        let e_exact = rel_l2(&got, &exact);
        eprintln!("taylor P={p}: vs own series {e_series:e}, vs exact {e_exact:e}");
        assert!(e_series < 5e-5, "P={p} series mismatch {e_series:e}");
        assert!(e_exact < prev * 1.0001, "no convergence at P={p}");
        prev = e_exact;
    }
}

#[test]
fn weyl_matches_its_series_and_converges() {
    let dev = Device::new(0).unwrap();
    let s = sample_set(300, 0.9, 5);
    let g = upload(&dev, &s);
    let x = Rng::new(6).complex_vec(DIMS.iter().product());
    let exact = dense_apply(&s, &Series::Exact, &x);
    let mut prev = f64::MAX;
    for n in [1usize, 2, 3, 4] {
        let spec = weyl(n);
        let (nf, nk) = spec.cost();
        let got = run_op(&g, spec, &[], vec![], &x);
        let series = dense_apply(&s, &Series::WeylPair(n), &x);
        let e_series = rel_l2(&got, &series);
        let e_exact = rel_l2(&got, &exact);
        eprintln!("weyl n={n}: fields {nf} kernels {nk}: vs own series {e_series:e}, vs exact {e_exact:e}");
        assert!(e_series < 5e-5, "n={n} series mismatch {e_series:e}");
        assert!(e_exact < prev * 1.0001, "no convergence at n={n}");
        prev = e_exact;
    }
}

#[test]
fn tucker_with_svd_basis() {
    use nalgebra::DMatrix;
    let dev = Device::new(0).unwrap();
    let s = sample_set(260, 1.2, 7);
    let m = s.m();
    let v: usize = DIMS.iter().product();
    // E[j][r] = e^{-i Phi_j(r)}
    let mut e = DMatrix::<Complex64>::zeros(m, v);
    for j in 0..m {
        for r in 0..v {
            let (ix, iy, iz) = (
                r % DIMS[2],
                (r / DIMS[2]) % DIMS[1],
                r / (DIMS[2] * DIMS[1]),
            );
            let ph = s.phi(j, xi(ix, DIMS[2]), xi(iy, DIMS[1]), xi(iz, DIMS[0]));
            e[(j, r)] = Complex64::from_polar(1.0, -ph);
        }
    }
    let svd = e.clone().svd(true, true);
    let (u, sv, vt) = (svd.u.unwrap(), svd.singular_values, svd.v_t.unwrap());
    let g = upload(&dev, &s);
    let x = Rng::new(8).complex_vec(v);
    let exact = dense_apply(&s, &Series::Exact, &x);
    for l in [2usize, 4, 6] {
        let basis: Vec<Vec<Complex64>> = (0..l)
            .map(|a| (0..v).map(|r| vt[(a, r)]).collect())
            .collect(); // E = U S V^H => b_l(r) = (V^H)[l][r]
        let coef: Vec<Vec<Complex64>> = (0..l)
            .map(|a| (0..m).map(|j| u[(j, a)] * sv[a]).collect())
            .collect();
        // extra weight sets for kernels (l <= l'): conj(c_l) c_l'
        let mut extra: Vec<CudaSlice<C32>> = Vec::new();
        for a in 0..l {
            for b in 0..l {
                let w: Vec<Complex64> = (0..m).map(|j| coef[a][j].conj() * coef[b][j]).collect();
                extra.push(g.stream.clone_htod(&to_c32(&w)).unwrap());
            }
        }
        let arrays: Vec<CudaSlice<C32>> = basis
            .iter()
            .map(|b| g.stream.clone_htod(&to_c32(b)).unwrap())
            .collect();
        let got = run_op(&g, tucker(l), &extra, arrays, &x);
        let series = dense_apply(&s, &Series::Basis { basis, coef }, &x);
        let e_series = rel_l2(&got, &series);
        let e_exact = rel_l2(&got, &exact);
        eprintln!("tucker L={l}: vs own series {e_series:e}, vs exact {e_exact:e}");
        assert!(e_series < 1e-4, "L={l} mismatch {e_series:e}");
    }
}
