//! Baseline: cuFFT C2C f32, out-of-place, 3D (or 1D/2D), batched. Reports median ms and effective GB/s
//! (counting one read + one write of the whole array, i.e. a *lower bound* of memory traffic for a
//! single-pass transform; real multi-pass transforms exceed this so the number is "effective", not achieved).
//!
//! usage: cufft_baseline [n0[,n1[,n2]] ...]   (default: a sweep relevant to 2N Toeplitz sizes)

use cudarc::cufft::{result::CufftError, sys, CudaFft, FftDirection};
use cudarc::driver::CudaContext;
use hasty_bench::median;

fn parse(s: &str) -> Vec<usize> {
    s.split(',').map(|t| t.parse().expect("dim")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shapes: Vec<Vec<usize>> = if args.is_empty() {
        vec![
            vec![256, 256, 256],
            vec![384, 384, 384],
            vec![512, 512, 512],
            vec![768, 768, 768],
            vec![1024, 1024, 1024],
        ]
    } else {
        args.iter().map(|a| parse(a)).collect()
    };

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    println!(
        "{:<18} {:>10} {:>12} {:>12}",
        "shape", "MiB", "median_ms", "eff_GB/s"
    );
    for dims in shapes {
        let n: usize = dims.iter().product();
        let bytes = n * 8;
        let free = ctx.mem_get_info().map(|(f, _)| f).unwrap_or(usize::MAX);
        if bytes * 2 > free {
            println!(
                "{:<18} skipped (needs {} MiB)",
                format!("{dims:?}"),
                bytes * 2 >> 20
            );
            continue;
        }
        let plan: Result<CudaFft, CufftError> = match dims.len() {
            1 => CudaFft::plan_1d(dims[0] as i32, sys::cufftType::CUFFT_C2C, 1, stream.clone()),
            2 => CudaFft::plan_2d(
                dims[0] as i32,
                dims[1] as i32,
                sys::cufftType::CUFFT_C2C,
                stream.clone(),
            ),
            3 => CudaFft::plan_3d(
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                sys::cufftType::CUFFT_C2C,
                stream.clone(),
            ),
            _ => panic!("1-3 dims"),
        };
        let plan = plan.expect("plan");
        let mut buf = stream.alloc_zeros::<sys::float2>(n).expect("alloc");
        let mut out = stream.alloc_zeros::<sys::float2>(n).expect("alloc");
        let (ev0, ev1) = (
            ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap(),
            ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap(),
        );
        let mut times = Vec::new();
        for it in 0..13 {
            ev0.record(&stream).unwrap();
            plan.exec_c2c(&mut buf, &mut out, FftDirection::Forward)
                .expect("exec");
            ev1.record(&stream).unwrap();
            ev1.synchronize().unwrap();
            if it >= 3 {
                times.push(ev0.elapsed_ms(&ev1).unwrap());
            }
        }
        let ms = median(&times);
        let gbs = 2.0 * bytes as f64 / (ms as f64 * 1e-3) / 1e9;
        println!(
            "{:<18} {:>10} {:>12.3} {:>12.1}",
            format!("{dims:?}"),
            bytes >> 20,
            ms,
            gbs
        );
    }
}
