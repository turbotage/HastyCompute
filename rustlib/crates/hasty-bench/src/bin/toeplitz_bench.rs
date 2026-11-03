//! Cost of the normal-operator families on the GPU (f32), one coil, random data (timing only).
//!   toeplitz_bench [N ...]       cube image size N (padded to 2N)
//! Plain = fused ConvPlan (reference); everything else goes through MixOp (separate forward/mix/inverse stages).

use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_fft::{ConvPlan, ConvSpec, KernelFormat};
use hasty_toeplitz::*;

fn main() {
    let sizes: Vec<usize> = {
        let a: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|s| s.parse().ok())
            .collect();
        if a.is_empty() {
            vec![64, 96, 128]
        } else {
            a
        }
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
        "{:<10} {:<22} {:>7} {:>8} {:>10} {:>12} {:>14}",
        "N (pad 2N)", "operator", "fields", "kernels", "kernel GB", "ms / coil", "x plain(fused)"
    );
    for n in sizes {
        let dims = [n, n, n];
        let pad = [2 * n, 2 * n, 2 * n];
        let nin = n * n * n;
        let npad = 8 * nin;
        let mut r = hasty_testkit::Rng::new(1);
        let mut rnd = |c: usize| -> Vec<C32> {
            (0..c)
                .map(|_| C32::new(r.uniform() as f32, r.uniform() as f32))
                .collect()
        };
        let x = stream.clone_htod(&rnd(nin)).unwrap();
        let mut out = stream.alloc_zeros::<C32>(nin).unwrap();
        let time = |f: &mut dyn FnMut()| -> f32 {
            let mut t = Vec::new();
            for i in 0..8 {
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
        // plain fused reference
        let k = stream.clone_htod(&rnd(npad)).unwrap();
        let mut scratch = stream.alloc_zeros::<C32>(npad).unwrap();
        let conv = ConvPlan::<f32>::new(
            &dev,
            &ConvSpec {
                dims: dims.to_vec(),
                dims_pad: pad.to_vec(),
                pre: vec![],
                post: vec![],
                accumulate: false,
                kernel_format: KernelFormat::Complex,
            },
        )
        .unwrap();
        let t_plain = time(&mut || {
            conv.execute(&stream, &x, &k, 1.0, &mut scratch, &mut out, &[], &[])
                .unwrap()
        });
        println!(
            "{:<10} {:<22} {:>7} {:>8} {:>10.2} {:>12.3} {:>14.2}",
            n,
            "plain (fused conv)",
            1,
            1,
            npad as f64 * 8e-9,
            t_plain,
            1.0
        );
        drop((k, scratch));
        let specs: Vec<(MixSpec, usize)> = vec![
            (plain(), 0),
            (weyl(1), 0),
            (taylor_hankel(1), 0),
            (weyl(2), 0),
            (taylor_hankel(2), 0),
            (tucker(4), 4),
            (tucker(6), 6),
            (cp(8), 8),
            (cp(12), 12),
        ];
        for (spec, nbasis) in specs {
            let (nf, nk) = spec.cost();
            let gb = (nk * npad * 8) as f64 * 1e-9 + (nf * npad * 8) as f64 * 1e-9;
            let free = dev.ctx.mem_get_info().map(|(f, _)| f).unwrap_or(usize::MAX);
            if gb * 1e9 + 1e9 > free as f64 {
                println!(
                    "{:<10} {:<22} {:>7} {:>8} {:>10.2} {:>12}",
                    n,
                    spec.label,
                    nf,
                    nk,
                    nk as f64 * npad as f64 * 8e-9,
                    "skipped (memory)"
                );
                continue;
            }
            let kernels: Vec<_> = (0..nk)
                .map(|_| stream.clone_htod(&rnd(npad)).unwrap())
                .collect();
            let arrays: Vec<_> = (0..nbasis)
                .map(|_| stream.clone_htod(&rnd(nin)).unwrap())
                .collect();
            let mut op = MixOp::<f32>::new(&dev, spec.clone(), dims, kernels, arrays).unwrap();
            let t = time(&mut || op.apply(&stream, &x, &mut out, false).unwrap());
            let fused = op.is_fused();
            let t_unfused = if fused {
                op.set_fused(false);
                let tu = time(&mut || op.apply(&stream, &x, &mut out, false).unwrap());
                op.set_fused(true);
                Some(tu)
            } else {
                None
            };
            println!(
                "{:<10} {:<22} {:>7} {:>8} {:>10.2} {:>12.3} {:>14.2}   {}",
                n,
                spec.label,
                nf,
                nk,
                nk as f64 * npad as f64 * 8e-9,
                t,
                t / t_plain,
                match t_unfused {
                    Some(tu) => format!("fused Z pass (unfused {:.3} ms, {:.2}x)", tu, tu / t),
                    None => "unfused (does not fit / narrow tiles)".to_string(),
                }
            );
        }
    }
}
