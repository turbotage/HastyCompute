use hasty_cuda::{Device, Real, C32, C64};
use hasty_fft::{AxisPlan, Direction, FftPlan, Norm};
use hasty_testkit::{fft_nd, rel_l2, Rng, C64 as RC64};

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

fn run_nd<R: Real>(dev: &Device, dims: &[usize], dir: Direction, tol: f64) {
    let total: usize = dims.iter().product();
    let mut rng = Rng::new(dims.iter().sum::<usize>() as u64 + 11);
    let x = rng.complex_vec(total);
    let sign = if dir == Direction::Forward { -1 } else { 1 };
    let want = fft_nd(&x, dims, sign);
    let stream = dev.default_stream();
    let din = stream.clone_htod(&to_dev::<R>(&x)).unwrap();
    let mut dout = stream.alloc_zeros::<R::Complex>(total).unwrap();
    let plan = FftPlan::<R>::new(dev, dims, dir, Norm::None).unwrap();
    plan.execute(&stream, &din, &mut dout).unwrap();
    let got = from_dev::<R>(&stream.clone_dtoh(&dout).unwrap());
    let e = rel_l2(&got, &want);
    assert!(
        e < tol,
        "out-of-place {dims:?} {dir:?}: rel err {e:e} >= {tol:e}"
    );
    // in place
    let mut dbuf = stream.clone_htod(&to_dev::<R>(&x)).unwrap();
    plan.execute_inplace(&stream, &mut dbuf).unwrap();
    let got2 = from_dev::<R>(&stream.clone_dtoh(&dbuf).unwrap());
    assert!(rel_l2(&got2, &want) < tol, "in-place {dims:?}");
    // input untouched by out-of-place
    let back = from_dev::<R>(&stream.clone_dtoh(&din).unwrap());
    assert!(rel_l2(&back, &x) < 1e-6);
}

#[test]
fn nd_f32() {
    let dev = Device::new(0).unwrap();
    for dims in [
        vec![2],
        vec![3],
        vec![5],
        vec![7],
        vec![8],
        vec![13],
        vec![16],
        vec![12],
        vec![100],
        vec![128],
        vec![1000],
        vec![1024],
        vec![2048],
        vec![360],
        vec![4, 6],
        vec![9, 10],
        vec![16, 16],
        vec![7, 11],
        vec![6, 5, 4],
        vec![8, 9, 10],
        vec![16, 16, 16],
        vec![13, 7, 11],
        vec![32, 1, 24],
        vec![1, 17 - 4, 2],
    ] {
        run_nd::<f32>(&dev, &dims, Direction::Forward, 3e-6);
        run_nd::<f32>(&dev, &dims, Direction::Inverse, 3e-6);
    }
}

#[test]
fn nd_f64() {
    let dev = Device::new(0).unwrap();
    for dims in [
        vec![2],
        vec![3],
        vec![5],
        vec![7],
        vec![8],
        vec![11],
        vec![13],
        vec![64],
        vec![81],
        vec![1000],
        vec![1024],
        vec![4, 6, 5],
        vec![12, 10, 14],
    ] {
        run_nd::<f64>(&dev, &dims, Direction::Forward, 1e-13);
        run_nd::<f64>(&dev, &dims, Direction::Inverse, 1e-13);
    }
}

#[test]
fn every_length_up_to_600_contiguous_batched() {
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let mut worst = 0.0f64;
    for n in 2..=600usize {
        if hasty_fft::factorize(n).is_none() {
            continue;
        }
        let batch = 5;
        let mut rng = Rng::new(n as u64);
        let x = rng.complex_vec(n * batch);
        let mut want = Vec::new();
        for b in 0..batch {
            want.extend(fft_nd(&x[b * n..(b + 1) * n], &[n], -1));
        }
        let plan = AxisPlan::<f32>::new(&dev, n, 1, batch, Direction::Forward, 1.0).unwrap();
        let din = stream.clone_htod(&to_dev::<f32>(&x)).unwrap();
        let mut dout = stream.alloc_zeros::<C32>(n * batch).unwrap();
        plan.execute(&stream, &din, &mut dout).unwrap();
        let got = from_dev::<f32>(&stream.clone_dtoh(&dout).unwrap());
        let e = rel_l2(&got, &want);
        worst = worst.max(e);
        assert!(e < 3e-6, "n={n} err {e:e}");
    }
    eprintln!("worst f32 rel err over all supported n<=600: {worst:e}");
}

#[test]
fn strided_axis_matches_reference() {
    // transform axis 1 of a [3, n, 37] array (inner = 37 not a multiple of the tile width)
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    for n in [2usize, 6, 8, 15, 64, 100, 128] {
        let (outer, inner) = (3usize, 37usize);
        let mut rng = Rng::new(n as u64 + 5);
        let x = rng.complex_vec(outer * n * inner);
        let mut want = x.clone();
        for o in 0..outer {
            for i in 0..inner {
                let line: Vec<RC64> = (0..n).map(|k| x[o * n * inner + k * inner + i]).collect();
                let y = fft_nd(&line, &[n], -1);
                for k in 0..n {
                    want[o * n * inner + k * inner + i] = y[k];
                }
            }
        }
        let plan = AxisPlan::<f64>::new(&dev, n, inner, outer, Direction::Forward, 1.0).unwrap();
        let din = stream.clone_htod(&to_dev::<f64>(&x)).unwrap();
        let mut dout = stream.alloc_zeros::<C64>(x.len()).unwrap();
        plan.execute(&stream, &din, &mut dout).unwrap();
        let got = from_dev::<f64>(&stream.clone_dtoh(&dout).unwrap());
        assert!(rel_l2(&got, &want) < 1e-13, "strided n={n}");
    }
}

#[test]
fn normalised_roundtrip() {
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let dims = [12usize, 10, 14];
    let total: usize = dims.iter().product();
    let x = Rng::new(3).complex_vec(total);
    let f = FftPlan::<f32>::new(&dev, &dims, Direction::Forward, Norm::None).unwrap();
    let b = FftPlan::<f32>::new(&dev, &dims, Direction::Inverse, Norm::Inverse).unwrap();
    let mut d = stream.clone_htod(&to_dev::<f32>(&x)).unwrap();
    f.execute_inplace(&stream, &mut d).unwrap();
    b.execute_inplace(&stream, &mut d).unwrap();
    let y = from_dev::<f32>(&stream.clone_dtoh(&d).unwrap());
    assert!(rel_l2(&y, &x) < 3e-6);
}
