//! Fused pruned convolution (Toeplitz-multiplication shape) benchmark: hasty-fft `ConvPlan` vs the C++ baselines in `bench/cpp/conv_baseline.cu`.
//! Same operation: out += conj-mult(crop(IFFT(K . FFT(pad(x . d1)))), d2) / prod(pad), pad = 2x, f32.
//!
//!   conv_bench [Zn,Yn,Xn ...]

use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_fft::{ConvPlan, ConvSpec, DiagOp, KernelFormat};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let specs: Vec<Vec<usize>> = if args.is_empty() {
        [
            "64,64,64",
            "128,128,128",
            "160,160,160",
            "192,192,192",
            "256,256,256",
            "320,320,320",
            "512,512,256",
            "512,512,320",
        ]
        .iter()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .collect()
    } else {
        args.iter()
            .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
            .collect()
    };
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let ev = (
        dev.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap(),
        dev.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap(),
    );
    println!(
        "{:<16} {:>9} | hasty fused+pruned ms: complex-K   real-K   fp16-K",
        "N (pad=2N)", "pad MiB"
    );
    for dims in specs {
        let pad: Vec<usize> = dims.iter().map(|d| 2 * d).collect();
        let nin: usize = dims.iter().product();
        let npad: usize = pad.iter().product();
        let free = dev.ctx.mem_get_info().map(|(f, _)| f).unwrap_or(usize::MAX);
        if npad * 16 + nin * 32 + (300 << 20) > free {
            println!("{:<16} skipped", format!("{dims:?}"));
            continue;
        }
        let mut r = hasty_testkit::Rng::new(1);
        let mut rnd = |n: usize| -> Vec<C32> {
            (0..n)
                .map(|_| C32::new(r.uniform() as f32, r.uniform() as f32))
                .collect()
        };
        let x = stream.clone_htod(&rnd(nin)).unwrap();
        let d1 = stream.clone_htod(&rnd(nin)).unwrap();
        let d2 = stream.clone_htod(&rnd(nin)).unwrap();
        let k = stream.clone_htod(&rnd(npad)).unwrap();
        let mut scratch = stream.alloc_zeros::<C32>(npad).unwrap();
        let mut out = stream.alloc_zeros::<C32>(nin).unwrap();
        let mut ms_fmt = Vec::new();
        for fmt in [
            KernelFormat::Complex,
            KernelFormat::Real,
            KernelFormat::ComplexHalf,
        ] {
            let plan = ConvPlan::<f32>::new(
                &dev,
                &ConvSpec {
                    dims: dims.clone(),
                    dims_pad: pad.clone(),
                    pre: vec![DiagOp::Mult],
                    post: vec![DiagOp::MultConj],
                    accumulate: true,
                    kernel_format: fmt,
                },
            )
            .unwrap();
            let run = |f: &mut dyn FnMut()| -> f32 {
                let mut t = Vec::new();
                for i in 0..13 {
                    ev.0.record(&stream).unwrap();
                    f();
                    ev.1.record(&stream).unwrap();
                    ev.1.synchronize().unwrap();
                    if i >= 3 {
                        t.push(ev.0.elapsed_ms(&ev.1).unwrap());
                    }
                }
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                t[t.len() / 2]
            };
            let ms = match fmt {
                KernelFormat::Complex => run(&mut || {
                    plan.execute(&stream, &x, &k, 1.0, &mut scratch, &mut out, &[&d1], &[&d2])
                        .unwrap()
                }),
                KernelFormat::Real => {
                    let kr = stream.alloc_zeros::<f32>(npad).unwrap();
                    run(&mut || {
                        plan.execute(
                            &stream,
                            &x,
                            &kr,
                            1.0,
                            &mut scratch,
                            &mut out,
                            &[&d1],
                            &[&d2],
                        )
                        .unwrap()
                    })
                }
                _ => {
                    let kh = stream.alloc_zeros::<u32>(npad).unwrap();
                    run(&mut || {
                        plan.execute(
                            &stream,
                            &x,
                            &kh,
                            1.0,
                            &mut scratch,
                            &mut out,
                            &[&d1],
                            &[&d2],
                        )
                        .unwrap()
                    })
                }
            };
            ms_fmt.push(ms);
            if fmt == KernelFormat::Complex && std::env::var("PROFILE").is_ok() {
                let t = plan
                    .execute_profiled(&stream, &x, &k, 1.0, &mut scratch, &mut out, &[&d1], &[&d2])
                    .unwrap();
                println!(
                    "   per-pass ms (complex kernel): {:?}",
                    t.iter()
                        .map(|v| (v * 1000.0).round() / 1000.0)
                        .collect::<Vec<_>>()
                );
            }
        }
        println!(
            "{:<16} {:>9} | {:>10.3} {:>10.3} {:>10.3}",
            dims.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(","),
            (npad * 8) >> 20,
            ms_fmt[0],
            ms_fmt[1],
            ms_fmt[2]
        );
    }
}
