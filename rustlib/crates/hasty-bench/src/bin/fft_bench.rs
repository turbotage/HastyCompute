//! hasty-fft vs cuFFT. Out-of-place C2C f32 (default) or f64 (`--f64`).
//!
//!   fft_bench [--f64] [--iters N] [1d|3d|all]            built-in sweeps
//!   fft_bench [--f64] shape...                           e.g. 1d:1024:16384 (n:batch)  3d:320,320,320
//!
//! "eff GB/s" = 2 * bytes / time (one read + one write); a 1-D single-kernel transform can reach ~memory roofline,
//! a 3-D transform needs 3 passes so its roofline in this unit is ~1/3 of memory bandwidth.

use cudarc::cufft::{result::CufftError, sys, CudaFft, FftDirection};
use cudarc::driver::{sys::CUevent_flags, CudaContext};
use hasty_cuda::{Device, Real};
use hasty_fft::{AxisPlan, Direction, FftPlan, Norm};

fn median(v: &mut Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Ev {
    a: cudarc::driver::CudaEvent,
    b: cudarc::driver::CudaEvent,
}

fn time<F: FnMut()>(ev: &Ev, stream: &cudarc::driver::CudaStream, iters: usize, mut f: F) -> f32 {
    for _ in 0..3 {
        f();
    }
    let mut t = Vec::new();
    for _ in 0..iters {
        ev.a.record(stream).unwrap();
        f();
        ev.b.record(stream).unwrap();
        ev.b.synchronize().unwrap();
        t.push(ev.a.elapsed_ms(&ev.b).unwrap());
    }
    median(&mut t)
}

fn run<R: Real>(dev: &Device, spec: &str, iters: usize)
where
    R::Complex: cudarc::driver::DeviceRepr,
{
    let ctx: &std::sync::Arc<CudaContext> = &dev.ctx;
    let stream = dev.default_stream();
    let ev = Ev {
        a: ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap(),
        b: ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap(),
    };
    let (kind, rest) = spec.split_once(':').unwrap();
    if kind == "ax" {
        // ax:n:inner:outer -- one strided axis pass, hasty only
        let v: Vec<usize> = rest.split(':').map(|x| x.parse().unwrap()).collect();
        let (n, inner, outer) = (v[0], v[1], v[2]);
        let n_total = n * inner * outer;
        let host: Vec<R::Complex> = {
            let mut r = hasty_testkit::Rng::new(1);
            (0..n_total)
                .map(|_| R::complex(r.uniform(), r.uniform()))
                .collect()
        };
        let din = stream.clone_htod(&host).unwrap();
        let mut dout = stream.alloc_zeros::<R::Complex>(n_total).unwrap();
        let p = AxisPlan::<R>::new(dev, n, inner, outer, Direction::Forward, 1.0).unwrap();
        let t = time(&ev, &stream, iters, || {
            p.execute(&stream, &din, &mut dout).unwrap()
        });
        let bytes = n_total * 2 * R::BYTES;
        println!(
            "{spec:<28} lpb={:<3} tpl={:<4} radices={:?} | {:>8.3} ms {:>6.0} GB/s",
            p.cfg.lpb,
            p.cfg.tpl,
            p.cfg.radices,
            t,
            2.0 * bytes as f64 / (t as f64 * 1e-3) / 1e9
        );
        return;
    }
    let (dims, batch): (Vec<usize>, usize) = match kind {
        "1d" => {
            let (n, b) = rest.split_once(':').unwrap();
            (vec![n.parse().unwrap()], b.parse().unwrap())
        }
        _ => (rest.split(',').map(|x| x.parse().unwrap()).collect(), 1),
    };
    let n_total: usize = dims.iter().product::<usize>() * batch;
    let bytes = n_total * 2 * R::BYTES;
    let free = ctx.mem_get_info().map(|(f, _)| f).unwrap_or(usize::MAX);
    if bytes * 4 + (256 << 20) > free {
        println!("{spec:<24} skipped (needs {} MiB)", bytes * 2 >> 20);
        return;
    }
    let host: Vec<R::Complex> = {
        let mut r = hasty_testkit::Rng::new(1);
        (0..n_total)
            .map(|_| R::complex(r.uniform(), r.uniform()))
            .collect()
    };
    let din = stream.clone_htod(&host).unwrap();
    let mut dout = stream.alloc_zeros::<R::Complex>(n_total).unwrap();

    // hasty
    let th = if kind == "1d" {
        let p = AxisPlan::<R>::new(dev, dims[0], 1, batch, Direction::Forward, 1.0).unwrap();
        time(&ev, &stream, iters, || {
            p.execute(&stream, &din, &mut dout).unwrap()
        })
    } else {
        let p = FftPlan::<R>::new(dev, &dims, Direction::Forward, Norm::None).unwrap();
        time(&ev, &stream, iters, || {
            p.execute(&stream, &din, &mut dout).unwrap()
        })
    };

    // cuFFT
    let tc = {
        let ty = if R::IS_F64 {
            sys::cufftType::CUFFT_Z2Z
        } else {
            sys::cufftType::CUFFT_C2C
        };
        let plan: Result<CudaFft, CufftError> = match (kind, dims.len()) {
            ("1d", _) => CudaFft::plan_1d(dims[0] as i32, ty, batch as i32, stream.clone()),
            (_, 1) => CudaFft::plan_1d(dims[0] as i32, ty, 1, stream.clone()),
            (_, 2) => CudaFft::plan_2d(dims[0] as i32, dims[1] as i32, ty, stream.clone()),
            _ => CudaFft::plan_3d(
                dims[0] as i32,
                dims[1] as i32,
                dims[2] as i32,
                ty,
                stream.clone(),
            ),
        };
        let plan = plan.unwrap();
        if R::IS_F64 {
            let mut a = stream
                .clone_htod(
                    &host
                        .iter()
                        .map(|c| {
                            let (x, y) = R::parts(*c);
                            sys::double2 { x, y }
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut b = stream.alloc_zeros::<sys::double2>(n_total).unwrap();
            time(&ev, &stream, iters, || {
                plan.exec_z2z(&mut a, &mut b, FftDirection::Forward)
                    .unwrap()
            })
        } else {
            let mut a = stream
                .clone_htod(
                    &host
                        .iter()
                        .map(|c| {
                            let (x, y) = R::parts(*c);
                            sys::float2 {
                                x: x as f32,
                                y: y as f32,
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut b = stream.alloc_zeros::<sys::float2>(n_total).unwrap();
            time(&ev, &stream, iters, || {
                plan.exec_c2c(&mut a, &mut b, FftDirection::Forward)
                    .unwrap()
            })
        }
    };
    let gb = |ms: f32| 2.0 * bytes as f64 / (ms as f64 * 1e-3) / 1e9;
    println!(
        "{spec:<24} {:>8} MiB | hasty {:>8.3} ms {:>6.0} GB/s | cuFFT {:>8.3} ms {:>6.0} GB/s | hasty/cuFFT speed {:>5.2}x",
        bytes >> 20,
        th,
        gb(th),
        tc,
        gb(tc),
        tc / th
    );
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let f64_mode = if let Some(i) = args.iter().position(|a| a == "--f64") {
        args.remove(i);
        true
    } else {
        false
    };
    let iters = if let Some(i) = args.iter().position(|a| a == "--iters") {
        args.remove(i);
        args.remove(i).parse().unwrap()
    } else {
        10
    };
    let dev = Device::new(0).unwrap();
    let mut specs: Vec<String> = Vec::new();
    let what = args.first().cloned().unwrap_or_else(|| "all".into());
    if what == "1d" || what == "all" {
        for n in [
            16usize, 64, 128, 256, 512, 1000, 1024, 2048, 384, 640, 768, 1536,
        ] {
            specs.push(format!("1d:{n}:{}", (1usize << 24) / n));
        }
    }
    if what == "3d" || what == "all" {
        for d in [128usize, 192, 256, 320, 384, 400, 512, 640] {
            specs.push(format!("3d:{d},{d},{d}"));
        }
        specs.push("3d:512,512,256".into());
        specs.push("3d:1024,1024,512".into());
    }
    if !["1d", "3d", "all"].contains(&what.as_str()) {
        specs = args.clone();
    }
    println!("precision: {}", if f64_mode { "f64" } else { "f32" });
    for s in specs {
        if f64_mode {
            run::<f64>(&dev, &s, iters)
        } else {
            run::<f32>(&dev, &s, iters)
        }
    }
}
