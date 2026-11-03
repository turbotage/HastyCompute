use hasty_cuda::{Device, Real};
use hasty_fft::{ConvPlan, ConvSpec, DiagOp, KernelFormat};
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

fn reference(
    x: &[RC64],
    k: &[RC64],
    dims: &[usize],
    pad: &[usize],
    pre: &[(DiagOp, Vec<RC64>)],
    post: &[(DiagOp, Vec<RC64>)],
    acc: Option<&[RC64]>,
) -> Vec<RC64> {
    let d = dims.len();
    let tot_in: usize = dims.iter().product();
    let tot_pad: usize = pad.iter().product();
    let ap = |op: DiagOp, a: RC64, b: RC64| {
        if op == DiagOp::Mult {
            a * b
        } else {
            a * b.conj()
        }
    };
    let mut xp = vec![RC64::new(0.0, 0.0); tot_pad];
    let idx_pad = |c: &[usize]| c.iter().zip(pad).fold(0, |acc, (&ci, &p)| acc * p + ci);
    let mut coord = vec![0usize; d];
    for i in 0..tot_in {
        let mut r = i;
        for a in (0..d).rev() {
            coord[a] = r % dims[a];
            r /= dims[a];
        }
        let mut v = x[i];
        for (op, arr) in pre {
            v = ap(*op, v, arr[i]);
        }
        xp[idx_pad(&coord)] = v;
    }
    let f = fft_nd(&xp, pad, -1);
    let g: Vec<RC64> = f.iter().zip(k).map(|(a, b)| a * b).collect();
    let y = fft_nd(&g, pad, 1);
    let mut out = vec![RC64::new(0.0, 0.0); tot_in];
    for i in 0..tot_in {
        let mut r = i;
        for a in (0..d).rev() {
            coord[a] = r % dims[a];
            r /= dims[a];
        }
        let mut v = y[idx_pad(&coord)] / tot_pad as f64;
        for (op, arr) in post {
            v = ap(*op, v, arr[i]);
        }
        out[i] = match acc {
            Some(a0) => a0[i] + v,
            None => v,
        };
    }
    out
}

fn case<R: Real>(
    dev: &Device,
    dims: &[usize],
    pad: &[usize],
    pre_ops: &[DiagOp],
    post_ops: &[DiagOp],
    accumulate: bool,
    tol: f64,
) {
    let stream = dev.default_stream();
    let tot_in: usize = dims.iter().product();
    let tot_pad: usize = pad.iter().product();
    let mut rng = Rng::new((tot_in * 31 + tot_pad) as u64);
    let x = rng.complex_vec(tot_in);
    let k = rng.complex_vec(tot_pad);
    let pre: Vec<(DiagOp, Vec<RC64>)> = pre_ops
        .iter()
        .map(|&o| (o, rng.complex_vec(tot_in)))
        .collect();
    let post: Vec<(DiagOp, Vec<RC64>)> = post_ops
        .iter()
        .map(|&o| (o, rng.complex_vec(tot_in)))
        .collect();
    let acc0 = rng.complex_vec(tot_in);
    let want = reference(
        &x,
        &k,
        dims,
        pad,
        &pre,
        &post,
        if accumulate { Some(&acc0) } else { None },
    );

    let plan = ConvPlan::<R>::new(
        dev,
        &ConvSpec {
            dims: dims.to_vec(),
            dims_pad: pad.to_vec(),
            pre: pre_ops.to_vec(),
            post: post_ops.to_vec(),
            accumulate,
            kernel_format: KernelFormat::Complex,
        },
    )
    .unwrap();
    let dx = stream.clone_htod(&to_dev::<R>(&x)).unwrap();
    let dk = stream.clone_htod(&to_dev::<R>(&k)).unwrap();
    // scratch deliberately filled with garbage: the plan must never read stale/padding data
    let junk: Vec<RC64> = (0..tot_pad)
        .map(|i| RC64::new(1e3 * ((i % 7) as f64 + 1.0), -1e3))
        .collect();
    let mut ds = stream.clone_htod(&to_dev::<R>(&junk)).unwrap();
    let mut dout = stream
        .clone_htod(&to_dev::<R>(&if accumulate {
            acc0.clone()
        } else {
            vec![RC64::new(7.0, 7.0); tot_in]
        }))
        .unwrap();
    let dpre: Vec<_> = pre
        .iter()
        .map(|(_, a)| stream.clone_htod(&to_dev::<R>(a)).unwrap())
        .collect();
    let dpost: Vec<_> = post
        .iter()
        .map(|(_, a)| stream.clone_htod(&to_dev::<R>(a)).unwrap())
        .collect();
    let pre_r: Vec<&_> = dpre.iter().collect();
    let post_r: Vec<&_> = dpost.iter().collect();
    plan.execute(&stream, &dx, &dk, 1.0, &mut ds, &mut dout, &pre_r, &post_r)
        .unwrap();
    let got = from_dev::<R>(&stream.clone_dtoh(&dout).unwrap());
    let e = rel_l2(&got, &want);
    assert!(e < tol, "dims {dims:?} pad {pad:?} pre {pre_ops:?} post {post_ops:?} acc {accumulate}: rel err {e:e}");
}

#[test]
fn conv_f32_all_ranks() {
    let dev = Device::new(0).unwrap();
    use DiagOp::*;
    let cases: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![20], vec![40]),
        (vec![7], vec![14]),
        (vec![16], vec![16]),
        (vec![5, 6], vec![10, 12]),
        (vec![8, 8], vec![16, 16]),
        (vec![9, 4], vec![18, 9]),
        (vec![6, 5, 7], vec![12, 10, 14]),
        (vec![8, 8, 8], vec![16, 16, 16]),
        (vec![4, 5, 6], vec![6, 10, 9]),
        (vec![10, 12, 9], vec![20, 24, 18]),
        (vec![16, 12, 20], vec![32, 24, 40]),
    ];
    for (dims, pad) in cases {
        case::<f32>(&dev, &dims, &pad, &[], &[], false, 4e-6);
        case::<f32>(&dev, &dims, &pad, &[Mult], &[MultConj], false, 4e-6);
        case::<f32>(
            &dev,
            &dims,
            &pad,
            &[Mult, MultConj],
            &[Mult, Mult],
            true,
            4e-6,
        );
        case::<f32>(&dev, &dims, &pad, &[], &[], true, 4e-6);
    }
}

#[test]
fn conv_f64() {
    let dev = Device::new(0).unwrap();
    use DiagOp::*;
    for (dims, pad) in [
        (vec![20usize], vec![40usize]),
        (vec![5, 6], vec![10, 12]),
        (vec![6, 5, 7], vec![12, 10, 14]),
        (vec![8, 8, 8], vec![16, 16, 16]),
    ] {
        case::<f64>(&dev, &dims, &pad, &[Mult], &[MultConj], true, 1e-13);
    }
}

#[test]
fn conv_is_a_toeplitz_product() {
    // K = FFT of a kernel h(d), d in [-(N-1), N-1] wrapped on the 2N circle  ==>  out = linear convolution (Toeplitz matvec).
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let n = 12usize;
    let mut rng = Rng::new(9);
    let h = rng.complex_vec(2 * n - 1); // h[d + n - 1]
    let x = rng.complex_vec(n);
    let mut want = vec![RC64::new(0.0, 0.0); n];
    for i in 0..n {
        for j in 0..n {
            want[i] += h[i + n - 1 - j] * x[j];
        }
    }
    let mut circ = vec![RC64::new(0.0, 0.0); 2 * n];
    for d in -(n as i64 - 1)..=(n as i64 - 1) {
        circ[d.rem_euclid(2 * n as i64) as usize] = h[(d + n as i64 - 1) as usize];
    }
    let kk = fft_nd(&circ, &[2 * n], -1);
    let plan = ConvPlan::<f64>::new(
        &dev,
        &ConvSpec {
            dims: vec![n],
            dims_pad: vec![2 * n],
            pre: vec![],
            post: vec![],
            accumulate: false,
            kernel_format: KernelFormat::Complex,
        },
    )
    .unwrap();
    let dx = stream.clone_htod(&to_dev::<f64>(&x)).unwrap();
    let dk = stream.clone_htod(&to_dev::<f64>(&kk)).unwrap();
    let mut ds = stream.alloc_zeros::<hasty_cuda::C64>(2 * n).unwrap();
    let mut dout = stream.alloc_zeros::<hasty_cuda::C64>(n).unwrap();
    plan.execute(&stream, &dx, &dk, 1.0, &mut ds, &mut dout, &[], &[])
        .unwrap();
    let got = from_dev::<f64>(&stream.clone_dtoh(&dout).unwrap());
    assert!(rel_l2(&got, &want) < 1e-13);
}

#[test]
fn conv_kernel_formats() {
    // Real / fp16 / bf16 packed kernels vs the f64 reference with the same (rounded) kernel values
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let (dims, pad) = (vec![6usize, 5, 7], vec![12usize, 10, 14]);
    let (tot_in, tot_pad) = (
        dims.iter().product::<usize>(),
        pad.iter().product::<usize>(),
    );
    let mut rng = Rng::new(77);
    let x = rng.complex_vec(tot_in);
    // kernel with large dynamic range to exercise kscale: |K| up to ~5e5
    let k: Vec<RC64> = rng
        .complex_vec(tot_pad)
        .into_iter()
        .map(|c| c * 3.0e5)
        .collect();
    let kr: Vec<RC64> = k.iter().map(|c| RC64::new(c.re, 0.0)).collect();
    let dx = stream.clone_htod(&to_dev::<f32>(&x)).unwrap();
    let mut ds = stream.alloc_zeros::<hasty_cuda::C32>(tot_pad).unwrap();
    let rf = |fmt: KernelFormat| {
        ConvPlan::<f32>::new(
            &dev,
            &ConvSpec {
                dims: dims.clone(),
                dims_pad: pad.clone(),
                pre: vec![],
                post: vec![],
                accumulate: false,
                kernel_format: fmt,
            },
        )
        .unwrap()
    };
    let kscale = 3.0e5f64;
    // real
    {
        let plan = rf(KernelFormat::Real);
        let host: Vec<f32> = kr.iter().map(|c| (c.re / kscale) as f32).collect();
        let dk = stream.clone_htod(&host).unwrap();
        let mut dout = stream.alloc_zeros::<hasty_cuda::C32>(tot_in).unwrap();
        plan.execute(&stream, &dx, &dk, kscale, &mut ds, &mut dout, &[], &[])
            .unwrap();
        let got = from_dev::<f32>(&stream.clone_dtoh(&dout).unwrap());
        let want = reference(&x, &kr, &dims, &pad, &[], &[], None);
        assert!(rel_l2(&got, &want) < 5e-6, "real kernel");
    }
    for (fmt, tol) in [
        (KernelFormat::ComplexHalf, 2e-3),
        (KernelFormat::ComplexBf16, 1.5e-2),
    ] {
        let plan = rf(fmt);
        let host: Vec<u32> = k
            .iter()
            .map(|c| {
                let (a, b) = ((c.re / kscale) as f32, (c.im / kscale) as f32);
                if fmt == KernelFormat::ComplexHalf {
                    hasty_fft::pack_half(a, b)
                } else {
                    hasty_fft::pack_bf16(a, b)
                }
            })
            .collect();
        let dk = stream.clone_htod(&host).unwrap();
        let mut dout = stream.alloc_zeros::<hasty_cuda::C32>(tot_in).unwrap();
        plan.execute(&stream, &dx, &dk, kscale, &mut ds, &mut dout, &[], &[])
            .unwrap();
        let got = from_dev::<f32>(&stream.clone_dtoh(&dout).unwrap());
        let want = reference(&x, &k, &dims, &pad, &[], &[], None);
        let e = rel_l2(&got, &want);
        assert!(e < tol, "{fmt:?}: rel err {e:e}");
        eprintln!("{fmt:?}: rel err {e:e}");
    }
}
