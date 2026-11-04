//! Lengths with a prime factor > 13 (Bluestein core) against the CPU reference.
use hasty_cuda::{Device, Real};
use hasty_fft::{AxisPlan, Direction, FftPlan, Norm};
use hasty_testkit::{fft_nd, rel_l2, Rng, C64 as RC64};

fn to_dev<R: Real>(v: &[RC64]) -> Vec<R::Complex> {
    v.iter().map(|c| R::complex(c.re, c.im)).collect()
}
fn from_dev<R: Real>(v: &[R::Complex]) -> Vec<RC64> {
    v.iter().map(|&c| { let (a, b) = R::parts(c); RC64::new(a, b) }).collect()
}

fn batched<R: Real>(dev: &Device, n: usize, inner: usize, outer: usize, dir: Direction, tol: f64) {
    let st = dev.default_stream();
    let mut rng = Rng::new(n as u64 * 7 + inner as u64);
    let x = rng.complex_vec(outer * n * inner);
    let sign = if dir == Direction::Forward { -1 } else { 1 };
    let mut want = x.clone();
    for o in 0..outer {
        for i in 0..inner {
            let line: Vec<RC64> = (0..n).map(|k| x[(o * n + k) * inner + i]).collect();
            let f = fft_nd(&line, &[n], sign);
            for k in 0..n {
                want[(o * n + k) * inner + i] = f[k];
            }
        }
    }
    let plan = AxisPlan::<R>::new(dev, n, inner, outer, dir, 1.0).unwrap();
    let din = st.clone_htod(&to_dev::<R>(&x)).unwrap();
    let mut dout = st.alloc_zeros::<R::Complex>(x.len()).unwrap();
    plan.execute(&st, &din, &mut dout).unwrap();
    let e = rel_l2(&from_dev::<R>(&st.clone_dtoh(&dout).unwrap()), &want);
    assert!(e < tol, "n={n} inner={inner} {dir:?}: rel err {e:e}");
    let mut buf = st.clone_htod(&to_dev::<R>(&x)).unwrap();
    plan.execute_inplace(&st, &mut buf).unwrap();
    let e2 = rel_l2(&from_dev::<R>(&st.clone_dtoh(&buf).unwrap()), &want);
    assert!(e2 < tol, "in place n={n}: rel err {e2:e}");
    eprintln!("n={n:5} inner={inner:3} {dir:?}: {e:.2e}");
}

#[test]
fn bluestein_contiguous_f32() {
    let dev = Device::new(0).unwrap();
    for n in [17usize, 29, 31, 58, 97, 116, 232, 1009] {
        batched::<f32>(&dev, n, 1, 3, Direction::Forward, 3e-6);
    }
    batched::<f32>(&dev, 37, 1, 4, Direction::Inverse, 3e-6);
}

#[test]
fn bluestein_strided_and_f64() {
    let dev = Device::new(0).unwrap();
    batched::<f32>(&dev, 29, 37, 3, Direction::Forward, 3e-6);
    batched::<f32>(&dev, 58, 5, 2, Direction::Inverse, 3e-6);
    batched::<f64>(&dev, 29, 1, 2, Direction::Forward, 1e-12);
    batched::<f64>(&dev, 232, 7, 2, Direction::Inverse, 1e-12);
}

#[test]
fn bluestein_nd_plan() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [6usize, 17, 29];
    let x = Rng::new(3).complex_vec(dims.iter().product());
    for (dir, sign) in [(Direction::Forward, -1), (Direction::Inverse, 1)] {
        let want = fft_nd(&x, &dims, sign);
        let plan = FftPlan::<f32>::new(&dev, &dims, dir, Norm::None).unwrap();
        let mut buf = st.clone_htod(&to_dev::<f32>(&x)).unwrap();
        plan.execute_inplace(&st, &mut buf).unwrap();
        let e = rel_l2(&from_dev::<f32>(&st.clone_dtoh(&buf).unwrap()), &want);
        assert!(e < 3e-6, "{dims:?} {dir:?}: {e:e}");
    }
}
